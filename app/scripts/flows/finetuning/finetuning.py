import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from app.scripts.flows.finetuning.export_to_ollama import (
    DEFAULT_MODEL_NAME as OLLAMA_MODEL_NAME,
)
from app.scripts.flows.finetuning.export_to_ollama import (
    SYSTEM_PROMPT as OLLAMA_SYSTEM_PROMPT,
)

HF_MODEL_ID = "google/gemma-4-E2B-it"
DATA_DIR = Path("data/llm_finetune")

# M1 Pro / 32 GB unified memory — tuned for ~22 GB peak with grad checkpointing
BATCH_SIZE = 2
NUM_LAYERS = 16  # Gemma 4 E2B has 35 layers; 16 LoRA layers keeps training practical
ITERS = 1500
LEARNING_RATE = 1e-5
MAX_SEQ_LEN = 2048
STEPS_PER_REPORT = 25  # was 1; reduces per-iteration I/O overhead
STEPS_PER_EVAL = 200
SAVE_EVERY = 500


def _run(cmd: list[str], cwd: Path | None = None, log_file: Path | None = None) -> None:
    display_cwd = str(cwd) if cwd else "(current)"
    # Use the current interpreter so the correct venv/environment is always used
    resolved = [sys.executable if arg == "python" else arg for arg in cmd]
    print(f"\n>>> Running: {' '.join(resolved)}")
    print(f"    in: {display_cwd}\n")

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            resolved,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
        )
        with open(log_file, "w") as lf:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, resolved)
    else:
        subprocess.run(resolved, check=True, cwd=cwd)


def split_long_sequences(
    data_dir: Path,
    hf_model_id: str,
    max_seq_len: int = MAX_SEQ_LEN,
) -> None:
    """
    Split JSONL records whose token length exceeds max_seq_len into smaller chunks.

    Operates in-place on train.jsonl / valid.jsonl / test.jsonl.
    Resolves the mlx_lm truncation warning for sequences > 2048 tokens, which
    wastes memory on padding and causes inconsistent batch sizes on M1.
    """
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    print("Loading tokenizer for sequence pre-splitting...")
    # Gemma 4's tokenizer_config ships `extra_special_tokens` as a list, but
    # transformers expects a mapping here; override it with an empty dict.
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, extra_special_tokens={})

    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        jsonl_path = data_dir / split
        if not jsonl_path.exists():
            continue

        records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        output: list[dict] = []
        long_count = 0

        for record in records:
            text: str = record.get("text", "")
            if not text:
                output.append(record)
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_seq_len:
                output.append(record)
                continue

            long_count += 1
            for start in range(0, len(token_ids), max_seq_len):
                chunk_ids = token_ids[start : start + max_seq_len]
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                output.append({**record, "text": chunk_text})

        if long_count:
            print(f"  {split}: split {long_count} long record(s) -> {len(output)} total")
            jsonl_path.write_text("\n".join(json.dumps(r) for r in output) + "\n")
        else:
            print(f"  {split}: all {len(records)} records within {max_seq_len} tokens")


def convert_to_mlx(hf_model_id: str, mlx_model_dir: Path) -> Path:
    """Convert HF model to MLX format (4-bit quantised)."""
    mlx_model_dir.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            hf_model_id,
            "--mlx-path",
            str(mlx_model_dir),
            "--q-bits",
            "4",
            "--q-group-size",
            "64",
        ]
    )

    return mlx_model_dir


def finetune_lora(
    mlx_model_dir: Path,
    adapter_dir: Path,
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_layers: int = NUM_LAYERS,
    iters: int = ITERS,
    learning_rate: float = LEARNING_RATE,
    max_seq_len: int = MAX_SEQ_LEN,
    steps_per_report: int = STEPS_PER_REPORT,
    steps_per_eval: int = STEPS_PER_EVAL,
    save_every: int = SAVE_EVERY,
    log_file: Path | None = None,
) -> Path:
    """Launch LoRA fine-tuning using mlx_lm."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            str(mlx_model_dir),
            "--train",
            "--data",
            str(data_dir),
            "--fine-tune-type",
            "lora",
            "--batch-size",
            str(batch_size),
            "--num-layers",
            str(num_layers),
            "--iters",
            str(iters),
            "--learning-rate",
            str(learning_rate),
            "--max-seq-length",
            str(max_seq_len),
            "--grad-checkpoint",  # ~30% lower peak memory -> fits batch_size=4
            "--adapter-path",
            str(adapter_dir),
            "--steps-per-report",
            str(steps_per_report),
            "--steps-per-eval",
            str(steps_per_eval),
            "--save-every",
            str(save_every),
        ],
        log_file=log_file,
    )

    return adapter_dir


def save_to_ollama(
    mlx_model_dir: Path,
    adapter_dir: Path,
    model_name: str = OLLAMA_MODEL_NAME,
) -> None:
    """Fuse adapter, convert to GGUF via llama.cpp, and register with Ollama.

    Uses a two-step process because mlx_lm's --export-gguf does not support
    the Gemma 4 text export path used here:
      1. mlx_lm fuse --save-path  →  HuggingFace safetensors format
      2. convert_hf_to_gguf.py    →  GGUF (F16)
    """
    output_dir = mlx_model_dir.parent / "ollama"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    fused_hf_dir = output_dir / "fused_hf"
    gguf_path = output_dir / "chembl-drug-chat.gguf"

    # Step 1: fuse LoRA into base model, save as HF safetensors
    _run(
        [
            "python",
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(mlx_model_dir),
            "--adapter-path",
            str(adapter_dir),
            "--save-path",
            str(fused_hf_dir),
            "--de-quantize",
        ]
    )

    # Step 2: convert HF safetensors → GGUF (no PyTorch needed)
    from app.scripts.flows.finetuning.convert_gemma3_gguf import convert as _gguf_convert

    _gguf_convert(fused_hf_dir, gguf_path)

    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        f"FROM {gguf_path.resolve()}\n\n"
        f'SYSTEM """\n{OLLAMA_SYSTEM_PROMPT}\n"""\n\n'
        'TEMPLATE """'
        "{{- if .System }}{{ .System }}\n\n{{ end -}}"
        "{{- range .Messages -}}"
        '{{- if eq .Role "user" }}### Question\n{{ .Content }}\n\n### Answer\n'
        '{{- else if eq .Role "assistant" }}{{ .Content }}\n\n{{ end -}}'
        '{{- end -}}"""\n\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.5\n"  # raised from 1.3 — harder penalty for repetition
        "PARAMETER repeat_last_n 512\n"  # raised from 256 — catches longer repeated phrases
        "PARAMETER num_ctx 1024\n"  # reduced from 2048 — limits context carry-over
        "PARAMETER num_predict 300\n"  # reduced from 400 — shorter, less rambling
        'PARAMETER stop "### Question"\n'
        'PARAMETER stop "### Answer"\n'
    )

    _run(["ollama", "create", model_name, "-f", str(modelfile_path)])

    print(f"\nOllama model '{model_name}' created successfully!")
    print(f"  Run with: ollama run {model_name}")


def gemma4_chembl_toon_finetune_flow(
    hf_model_id: str = HF_MODEL_ID,
    data_dir: str = str(DATA_DIR),
    run_name: str | None = None,
) -> None:
    """
    Finetuning pipeline optimised for Apple Silicon (M1 Pro, 32 GB):

    1. Pre-split training sequences > 2048 tokens to eliminate truncation waste.
    2. Convert Gemma 4 HF model -> MLX format (4-bit quantised).
    3. LoRA fine-tuning with gradient checkpointing, log capture.

    When run standalone, also exports the adapter to Ollama (step 4).
    When run via the Prefect pipeline, Ollama export is handled as a
    separate task after this flow completes.

    Args:
        hf_model_id: HuggingFace model ID
        data_dir: Path to training data directory (must contain train.jsonl)
        run_name: Optional custom run name, defaults to timestamp
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path("artifacts") / run_name
    mlx_model_dir = run_dir / "mlx" / "gemma-4-e2b-it-mlx"
    adapter_dir = run_dir / "adapters" / "gemma4-e2b-it-chembl-toon"
    log_file = run_dir / "logs" / "finetune.log"

    print(f"\n{'=' * 60}")
    print(f"Starting finetuning run: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 60}\n")

    split_long_sequences(Path(data_dir), hf_model_id)
    mlx_model_dir = convert_to_mlx(hf_model_id, mlx_model_dir)
    adapter_dir = finetune_lora(mlx_model_dir, adapter_dir, Path(data_dir), log_file=log_file)

    print(f"\n{'=' * 60}")
    print("Finetuning complete!")
    print(f"MLX model:    {mlx_model_dir}")
    print(f"LoRA adapter: {adapter_dir}")
    print(f"Training log: {log_file}")
    print(f"{'=' * 60}\n")

    save_to_ollama(
        mlx_model_dir=mlx_model_dir,
        adapter_dir=adapter_dir,
    )


gemma3_chembl_toon_finetune_flow = gemma4_chembl_toon_finetune_flow


if __name__ == "__main__":
    gemma4_chembl_toon_finetune_flow()
