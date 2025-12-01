import polars as pl
from pathlib import Path
import subprocess
from datetime import datetime

HF_MODEL_ID = "google/gemma-3-4b-pt"
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
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print("---- STDOUT ----")
        print(result.stdout)

    if result.stderr:
        print("---- STDERR ----")
        print(result.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


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
            "25",
        ]
    )

    return adapter_dir


def gemma3_chembl_toon_finetune_flow(
    hf_model_id: str = HF_MODEL_ID,
    data_dir: str = str(DATA_DIR),
    run_name: str | None = None,
) -> None:
    """
    Minimal finetuning pipeline:

    1. Convert Gemma 3 HF model -> MLX format (4-bit quantized).
    2. Run LoRA fine-tuning on the ChEMBL→TOON JSONL dataset.

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


if __name__ == "__main__":
    gemma3_chembl_toon_finetune_flow()
