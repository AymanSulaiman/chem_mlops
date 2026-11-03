"""
Export a fine-tuned ChEMBL model to GGUF and register it with Ollama.

Use this script to register (or re-register) any completed fine-tuning run with
Ollama without re-running training. Training can happen elsewhere (Colab): copy
the run's ``model/`` folder into ``artifacts/<run>/`` and export here.

Usage
-----
    # Latest run (auto-detected):
    uv run python -m app.scripts.flows.finetuning.export_to_ollama

    # Specific run:
    uv run python -m app.scripts.flows.finetuning.export_to_ollama \\
        --run-dir artifacts/20260403_220717

    # After export, start a chat:
    ollama run chembl-drug-chat:270m
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
MODEL_SUBDIR = "model"  # the fine-tuned Hugging Face model inside a run dir
CHECKPOINTS_SUBDIR = "checkpoints"
OLLAMA_SUBDIR = "ollama"  # GGUF + Modelfile, portable as a pair
DEFAULT_MODEL_NAME = "chembl-drug-chat:270m"
# llama.cpp quantization type. The quantized GGUF is the file that ships, so the
# deployed model is byte-for-byte the one registered (and measured) here.
# Done with llama-quantize because Ollama 0.40 can only quantize safetensors
# imports, and that path does not support Gemma 3.
QUANTIZE = "Q4_K_M"


def _quantize(src: Path, dst: Path, qtype: str) -> None:
    tool = shutil.which("llama-quantize")
    if tool is None:
        raise FileNotFoundError(
            "llama-quantize not found. Install llama.cpp (macOS: `brew install llama.cpp`), "
            "or export unquantized with --quantize ''."
        )
    _run([tool, str(src), str(dst), qtype])


def _run(cmd: list[str]) -> None:
    resolved = [sys.executable if arg == "python" else arg for arg in cmd]
    print(f"\n>>> {' '.join(resolved)}\n")
    subprocess.run(resolved, check=True)


def latest_run_dir(artifacts_dir: Path) -> Path:
    """Return the most recently created run directory."""
    runs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found in {artifacts_dir}")
    return runs[-1]


def write_modelfile(modelfile_path: Path, gguf_name: str) -> None:
    """Write the Modelfile that pairs with *gguf_name* in the same folder."""
    # No SYSTEM block. The model is trained and evaluated on bare
    # "### Question / ### Answer" text; a baked-in system prompt is prepended by
    # Ollama to every chat and pulls it out of distribution. Measured on
    # 20260921_184541_tools: 0/8 tool calls with the old SYSTEM block, 4/8 without.
    modelfile_path.write_text(
        # Relative to the Modelfile, so the folder can be copied anywhere (e.g. a container).
        f"FROM ./{gguf_name}\n\n"
        # Multi-turn template: each user message is wrapped in ### Question / ### Answer
        # so the model sees the exact format it was trained on, while Ollama passes the
        # full conversation history via .Messages on every turn.
        #
        # IMPORTANT: use "{{ else" (no leading dash) after ### Answer so the newline
        # between "### Answer" and the model's response is preserved.  The training data
        # was formatted as "### Answer\n{answer}", so stripping that newline ({{- else)
        # causes the model to emit an immediate EOS token and return empty content.
        'TEMPLATE """'
        "{{- if .System }}{{ .System }}\n\n{{ end -}}"
        "{{- range .Messages -}}"
        '{{- if eq .Role "user" }}### Question\n{{ .Content }}\n\n### Answer\n'
        '{{ else if eq .Role "assistant" }}{{ .Content }}\n\n{{ end -}}'
        '{{- end -}}"""\n\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.1\n"
        "PARAMETER repeat_last_n 512\n"
        # 2048 was fine for prose turns; a tool result (truncated at 2000 chars)
        # plus history overflows it and silently drops the oldest turns —
        # including the tool result the answer depends on.
        "PARAMETER num_ctx 8192\n"
        "PARAMETER num_predict 512\n"
        'PARAMETER stop "### Question"\n'
    )


def export_to_ollama(
    run_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    force: bool = False,
    quantize: str = QUANTIZE,
) -> None:
    """
    Convert the run's fine-tuned model to GGUF, write a Modelfile, and register
    it with Ollama. The GGUF is quantized to *quantize* (empty string keeps F16).

    Args:
        run_dir:    Artifact run directory (e.g. ``artifacts/20260403_220717``).
        model_name: Ollama model name to create.
        force:      If True, overwrite an existing ollama export directory.
        quantize:   llama.cpp quantization type, e.g. ``Q4_K_M``.
    """
    model_dir = run_dir / MODEL_SUBDIR
    output_dir = run_dir / OLLAMA_SUBDIR
    gguf_path = output_dir / "chembl-drug-chat.gguf"
    modelfile_path = output_dir / "Modelfile"

    # The Trainer writes model/ only when training finishes. A crashed run
    # leaves checkpoints/; fail here rather than inside the converter.
    if not (model_dir / "config.json").exists():
        checkpoints = sorted(
            (run_dir / CHECKPOINTS_SUBDIR).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        hint = (
            f"Latest checkpoint is {checkpoints[-1].name}: re-run training with the same "
            "--run-name to resume and finish."
            if checkpoints
            else "No checkpoints either. Check the training log and re-run the fine-tune."
        )
        raise FileNotFoundError(f"Training did not finish: no model in {model_dir}. {hint}")

    print(f"\n{'=' * 60}")
    print(f"Exporting run: {run_dir}")
    print(f"  Model    : {model_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Ollama   : {model_name} ({quantize or 'F16'})")
    print(f"{'=' * 60}\n")

    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory already exists: {output_dir}")
            print("Use --force to overwrite.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/4 — Converting to GGUF (F16)...")
    from app.scripts.flows.finetuning.convert_gemma3_gguf import convert as _gguf_convert

    f16_path = gguf_path.with_suffix(".f16.gguf") if quantize else gguf_path
    _gguf_convert(model_dir, f16_path)

    if quantize:
        print(f"Step 2/4 — Quantizing to {quantize}...")
        _quantize(f16_path, gguf_path, quantize)
        f16_path.unlink(missing_ok=True)  # rebuildable from model/; only the shipped file is kept

    print("Step 3/4 — Writing Modelfile...")
    write_modelfile(modelfile_path, gguf_path.name)
    print(f"  Written to {modelfile_path}")

    print(f"Step 4/4 — Registering '{model_name}' with Ollama...")
    _run(["ollama", "create", model_name, "-f", str(modelfile_path)])

    print(f"\n{'=' * 60}")
    print(f"Model '{model_name}' is ready.")
    print("\nStart a chat:")
    print(f"  ollama run {model_name}")
    print("\nExample questions:")
    print("  What does Aspirin target?")
    print("  Is it safe to take warfarin with aspirin?")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned run to GGUF and register it with Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Artifact run directory (e.g. artifacts/20260403_220717). "
            "Defaults to the most recent directory inside artifacts/."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        metavar="NAME",
        help=f"Ollama model name to create (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--quantize",
        default=QUANTIZE,
        help=f"llama.cpp quantization, '' for F16 (default: {QUANTIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing ollama export directory.",
    )

    args = parser.parse_args()

    export_to_ollama(
        run_dir=args.run_dir or latest_run_dir(ARTIFACTS_DIR),
        model_name=args.model_name,
        force=args.force,
        quantize=args.quantize,
    )
