from pathlib import Path
import subprocess
from datetime import datetime
import shutil

HF_MODEL_ID = "google/gemma-3-1b-pt"
DATA_DIR = Path("data/llm_finetune")

BATCH_SIZE = 2
NUM_LAYERS = 8
ITERS = 1500
LEARNING_RATE = 1e-5


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    display_cwd = str(cwd) if cwd else "(current)"
    print(f"\n>>> Running: {' '.join(cmd)}")
    print(f"    in: {display_cwd}\n")

    result = subprocess.run(
        cmd,
        check=False,
        cwd=cwd,
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def convert_to_mlx(hf_model_id: str, mlx_model_dir: Path) -> Path:
    """Convert HF model to MLX format."""
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
):
    # TODO: Find a way to save the logs to analyse the loss, tokens/sec etc. example of logs below
    """
    Iter 393: Train loss 0.568, Learning Rate 1.000e-05, It/sec 0.061, Tokens/sec 185.796, Trained Tokens 682568, Peak mem 25.933 GB
    Iter 394: Train loss 0.603, Learning Rate 1.000e-05, It/sec 0.175, Tokens/sec 218.927, Trained Tokens 683816, Peak mem 25.933 GB
    Iter 395: Train loss 0.415, Learning Rate 1.000e-05, It/sec 0.195, Tokens/sec 249.194, Trained Tokens 685091, Peak mem 25.933 GB
    Iter 396: Train loss 0.573, Learning Rate 1.000e-05, It/sec 0.078, Tokens/sec 193.704, Trained Tokens 687569, Peak mem 25.933 GB
    Iter 397: Train loss 0.593, Learning Rate 1.000e-05, It/sec 0.111, Tokens/sec 196.816, Trained Tokens 689347, Peak mem 25.933 GB
    Iter 398: Train loss 0.616, Learning Rate 1.000e-05, It/sec 0.162, Tokens/sec 215.451, Trained Tokens 690679, Peak mem 25.933 GB
    Iter 399: Train loss 0.594, Learning Rate 1.000e-05, It/sec 0.125, Tokens/sec 227.212, Trained Tokens 692493, Peak mem 25.933 GB
    """

    # TODO [WARNING] Some sequences are longer than 2048 tokens. The longest sentence 5161 will be truncated to 2048. Consider pre-splitting your data to save memory. <- figure this out

    """
    Launch LoRA fine-tuning using updated mlx_lm CLI.
    """
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
            "--adapter-path",
            str(adapter_dir),
            "--steps-per-report",
            "1",
        ]
    )

    return adapter_dir


def convert_mlx_to_gguf(
    mlx_model_dir: Path, adapter_dir: Path, output_dir: Path
) -> Path:
    """Convert MLX model + adapter to GGUF format for Ollama."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = output_dir / "model.gguf"

    fused_dir = output_dir / "fused"
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
            str(fused_dir),
        ]
    )

    hf_dir = output_dir / "hf_model"
    _run(
        [
            "python",
            "-m",
            "mlx_lm",
            "convert",
            "--mlx-path",
            str(fused_dir),
            "--hf-path",
            str(hf_dir),
        ]
    )

    _run(
        [
            "python",
            "-m",
            "llama_cpp.convert_hf_to_gguf",
            str(hf_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            "q4_k_m",
        ]
    )

    return gguf_path


def save_to_ollama(mlx_model_dir: Path, adapter_dir: Path, model_name: str) -> None:
    """Fuse adapter and create Ollama model."""
    output_dir = mlx_model_dir.parent / "ollama"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    fused_hf_dir = output_dir / "fused_hf"

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

    modelfile_path = output_dir / "Modelfile"

    modelfile_path.write_text(f"""FROM {fused_hf_dir.resolve()}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
""")

    _run(["ollama", "create", model_name, "-f", str(modelfile_path)])

    print(f"\nOllama model '{model_name}' created successfully!")
    print(f"  Test with: ollama run {model_name}")


def gemma3_chembl_toon_finetune_flow(
    hf_model_id: str = HF_MODEL_ID,
    data_dir: str = str(DATA_DIR),
    run_name: str | None = None,
) -> None:
    """
    Minimal finetuning pipeline:

    1. Convert Gemma 3 HF model -> MLX format (4-bit quantized).
    2. Run LoRA fine-tuning on the ChEMBL -> TOON JSONL dataset.

    Args:
        hf_model_id: HuggingFace model ID
        data_dir: Path to training data
        run_name: Optional custom run name, defaults to timestamp
    """
    # Generate timestamped run directory
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path("artifacts") / run_name
    mlx_model_dir = run_dir / "mlx" / "gemma-3-4b-pt-mlx"
    adapter_dir = run_dir / "adapters" / "gemma3-4b-pt-chembl-toon"

    print(f"\n{'=' * 60}")
    print(f"Starting finetuning run: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 60}\n")

    mlx_model_dir = convert_to_mlx(hf_model_id, mlx_model_dir)
    adapter_dir = finetune_lora(mlx_model_dir, adapter_dir, Path(data_dir))

    print(f"\n{'=' * 60}")
    print(f"Finetuning complete!")
    print(f"MLX model: {mlx_model_dir}")
    print(f"LoRA adapter: {adapter_dir}")
    print(f"{'=' * 60}\n")

    save_to_ollama(
        mlx_model_dir=mlx_model_dir,
        adapter_dir=adapter_dir,
        model_name=f"chembl-toon:1b",
    )


if __name__ == "__main__":
    gemma3_chembl_toon_finetune_flow()
