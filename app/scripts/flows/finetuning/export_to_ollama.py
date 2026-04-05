"""
Export a fine-tuned ChEMBL LoRA adapter to Ollama.

Use this script to register (or re-register) any completed fine-tuning run with
Ollama without re-running the full training pipeline.

Usage
-----
    # Latest run (auto-detected):
    uv run python -m app.scripts.flows.finetuning.export_to_ollama

    # Specific run:
    uv run python -m app.scripts.flows.finetuning.export_to_ollama \\
        --run-dir artifacts/20260403_220717

    # Custom Ollama model name:
    uv run python -m app.scripts.flows.finetuning.export_to_ollama \\
        --model-name my-chem-model:latest

    # After export, start a chat:
    ollama run chembl-drug-chat:1b
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MLX_SUBDIR = "mlx/gemma-3-1b-pt-mlx"
DEFAULT_ADAPTER_SUBDIR = "adapters/gemma3-1b-pt-chembl-toon"
DEFAULT_MODEL_NAME = "chembl-drug-chat:1b"

SYSTEM_PROMPT = """\
You are a chemistry and pharmacology assistant with expert knowledge of drug \
interactions, mechanisms of action, pharmacokinetics, and clinical pharmacology. \
Your knowledge is sourced from the ChEMBL database. Answer questions clearly \
and cite ChEMBL identifiers where relevant.\
"""


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


def export_to_ollama(
    run_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    mlx_subdir: str = DEFAULT_MLX_SUBDIR,
    adapter_subdir: str = DEFAULT_ADAPTER_SUBDIR,
    force: bool = False,
) -> None:
    """
    Fuse a LoRA adapter into the base MLX model, export to GGUF, and register
    the resulting model with Ollama.

    Args:
        run_dir:        Artifact run directory (e.g. ``artifacts/20260403_220717``).
        model_name:     Ollama model name to create (e.g. ``chembl-drug-chat:1b``).
        mlx_subdir:     Subdirectory within run_dir containing the MLX base model.
        adapter_subdir: Subdirectory within run_dir containing the LoRA adapter.
        force:          If True, overwrite an existing ollama export directory.
    """
    mlx_model_dir = run_dir / mlx_subdir
    adapter_dir = run_dir / adapter_subdir
    output_dir = run_dir / "mlx" / "ollama"
    gguf_path = output_dir / "chembl-drug-chat.gguf"
    modelfile_path = output_dir / "Modelfile"

    if not mlx_model_dir.exists():
        raise FileNotFoundError(f"MLX model not found: {mlx_model_dir}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    print(f"\n{'=' * 60}")
    print(f"Exporting run: {run_dir}")
    print(f"  MLX model : {mlx_model_dir}")
    print(f"  Adapter   : {adapter_dir}")
    print(f"  Output    : {output_dir}")
    print(f"  Ollama    : {model_name}")
    print(f"{'=' * 60}\n")

    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory already exists: {output_dir}")
            print("Use --force to overwrite.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    fused_hf_dir = output_dir / "fused_hf"

    # Step 1 — fuse adapter into base model, save as HF safetensors
    # (mlx_lm --export-gguf does not support gemma3_text, so we use --save-path
    #  and convert to GGUF in step 2 via llama.cpp's convert script)
    print("Step 1/4 — Fusing LoRA adapter into base model (HF format)...")
    _run(
        [
            "python", "-m", "mlx_lm", "fuse",
            "--model",        str(mlx_model_dir),
            "--adapter-path", str(adapter_dir),
            "--save-path",    str(fused_hf_dir),
            "--de-quantize",
        ]
    )

    # Step 2 — convert HF safetensors → GGUF using built-in converter (no PyTorch needed)
    print("Step 2/4 — Converting to GGUF (mlx + gguf, no PyTorch)...")
    from app.scripts.flows.finetuning.convert_gemma3_gguf import convert as _gguf_convert
    _gguf_convert(fused_hf_dir, gguf_path)

    # Step 3 — write Modelfile
    print("Step 3/4 — Writing Modelfile...")
    # The model was fine-tuned on raw "### Question / ### Answer" text (no chat
    # template).  Override Ollama's default Gemma template so prompts are sent
    # in the exact format the model expects.
    modelfile_path.write_text(
        f'FROM {gguf_path.resolve()}\n\n'
        f'SYSTEM """\n{SYSTEM_PROMPT}\n"""\n\n'
        'TEMPLATE """'
        '{{ if .System }}{{ .System }}\n\n{{ end }}'
        '### Question\n{{ .Prompt }}\n\n### Answer\n"""\n\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        'PARAMETER stop "### Question"\n'
    )
    print(f"  Written to {modelfile_path}")

    # Step 4 — register with Ollama
    print(f"Step 4/4 — Registering '{model_name}' with Ollama...")
    _run(["ollama", "create", model_name, "-f", str(modelfile_path)])

    print(f"\n{'=' * 60}")
    print(f"Model '{model_name}' is ready.")
    print("\nStart a chat:")
    print(f"  ollama run {model_name}")
    print("\nExample questions:")
    print("  What does Aspirin target?")
    print("  How is Warfarin metabolised?")
    print("  Which drugs share the CYP2C9 pathway with Warfarin?")
    print("  What are the black box warnings for Methotrexate?")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse a LoRA adapter and register the model with Ollama.",
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
        "--mlx-subdir",
        default=DEFAULT_MLX_SUBDIR,
        metavar="PATH",
        help=f"MLX model subdirectory within run-dir (default: {DEFAULT_MLX_SUBDIR})",
    )
    parser.add_argument(
        "--adapter-subdir",
        default=DEFAULT_ADAPTER_SUBDIR,
        metavar="PATH",
        help=f"Adapter subdirectory within run-dir (default: {DEFAULT_ADAPTER_SUBDIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing ollama export directory.",
    )

    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir(ARTIFACTS_DIR)

    export_to_ollama(
        run_dir=run_dir,
        model_name=args.model_name,
        mlx_subdir=args.mlx_subdir,
        adapter_subdir=args.adapter_subdir,
        force=args.force,
    )
