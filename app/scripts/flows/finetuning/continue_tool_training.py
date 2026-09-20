"""Continue training an existing adapter on a tool-heavy slice of the dataset.

A full run trains 950 K records for 2-4 hours, and the 60 K tool-call records in
it compete with ~900 K prose records that answer the same question shapes from
memory. This does the cheap experiment instead: start from the adapter that is
already good at pharmacology prose and keep teaching it, on a mix where tool
calls are a third of what it sees rather than six percent.

    uv run python -m app.scripts.flows.finetuning.continue_tool_training

Writes a new run directory, so the adapter it started from is never modified.
Skips the export — check the tool-call rate with eval_finetuned_model first:

    uv run python -m app.scripts.flows.eval.eval_finetuned_model --run-dir <new run>

# ponytail: the prose:tool ratio is one number to tune, not a search space.
# Start at 2 and move it if the model forgets prose (ratio up) or still will not
# call tools (ratio down).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

from app.scripts.flows.finetuning.export_to_ollama import (
    ARTIFACTS_DIR,
    DEFAULT_ADAPTER_SUBDIR,
    DEFAULT_MLX_SUBDIR,
    latest_run_dir,
)
from app.scripts.flows.finetuning.finetuning import finetune_lora
from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    TOOL_RESULT_HEADER,
    write_jsonl_splits,
)

DATA_DIR = Path("data/llm_finetune")
MIX_DIR = Path("data/llm_finetune_tools")
PROSE_RATIO = 2  # prose records kept per tool record
CONTINUE_ITERS = 600
CONTINUE_LEARNING_RATE = 1e-5


def build_tool_mix(
    data_dir: Path = DATA_DIR,
    out_dir: Path = MIX_DIR,
    prose_ratio: int = PROSE_RATIO,
    seed: int = 42,
) -> tuple[int, int]:
    """Write a train/valid split of every tool record plus sampled prose.

    Prose is kept so the model does not forget how to answer without a tool;
    the sample is drawn with a fixed seed so a rerun trains on the same mix.

    Returns:
        (tool record count, prose record count).

    Raises:
        FileNotFoundError: If train.jsonl is missing.
        RuntimeError: If it contains no tool-call records — the dataset predates
            generate_tool_call_qa and needs rebuilding.
    """
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — build the dataset first.")

    tool_records: list[dict] = []
    prose_records: list[dict] = []
    with train_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if TOOL_RESULT_HEADER in record.get("text", ""):
                tool_records.append(record)
            else:
                prose_records.append(record)

    if not tool_records:
        raise RuntimeError(
            f"No '{TOOL_RESULT_HEADER}' records in {train_path}. Rebuild the dataset: "
            "python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset"
        )

    rng = random.Random(seed)
    keep = min(len(prose_records), len(tool_records) * prose_ratio)
    sampled_prose = rng.sample(prose_records, keep)

    mixed = tool_records + sampled_prose
    rng.shuffle(mixed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    write_jsonl_splits(mixed, out_dir)

    print(f"Tool-call records : {len(tool_records):,}")
    print(f"Prose records kept: {len(sampled_prose):,} (ratio 1:{prose_ratio})")
    print(f"Written to {out_dir}/")
    return len(tool_records), len(sampled_prose)


def continue_tool_training(
    from_run: Path | None = None,
    data_dir: Path = DATA_DIR,
    mix_dir: Path = MIX_DIR,
    iters: int = CONTINUE_ITERS,
    prose_ratio: int = PROSE_RATIO,
    run_name: str | None = None,
) -> Path:
    """Continue the latest (or given) run's adapter on the tool mix.

    Returns:
        The new run directory.
    """
    source_run = from_run if from_run is not None else latest_run_dir(ARTIFACTS_DIR)
    source_adapter = source_run / DEFAULT_ADAPTER_SUBDIR / "adapters.safetensors"
    if not source_adapter.exists():
        raise FileNotFoundError(f"No trained adapter at {source_adapter}.")

    build_tool_mix(data_dir, mix_dir, prose_ratio=prose_ratio)

    run_name = run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tools"
    run_dir = ARTIFACTS_DIR / run_name
    adapter_dir = run_dir / DEFAULT_ADAPTER_SUBDIR
    log_file = run_dir / "logs" / "finetune.log"

    # The MLX base model is shared, not reconverted: only the adapter changes.
    mlx_model_dir = run_dir / DEFAULT_MLX_SUBDIR
    mlx_model_dir.parent.mkdir(parents=True, exist_ok=True)
    if not mlx_model_dir.exists():
        mlx_model_dir.symlink_to((source_run / DEFAULT_MLX_SUBDIR).resolve())

    print(f"\n{'=' * 60}")
    print(f"Continuing {source_run.name} -> {run_name}")
    print(f"  Adapter : {source_adapter}")
    print(f"  Data    : {mix_dir}")
    print(f"  Iters   : {iters}")
    print(f"{'=' * 60}\n")

    finetune_lora(
        mlx_model_dir,
        adapter_dir,
        mix_dir,
        iters=iters,
        learning_rate=CONTINUE_LEARNING_RATE,
        log_file=log_file,
        resume_from=source_adapter,
    )

    print(
        f"\nDone. Measure before exporting:\n  uv run python -m "
        f"app.scripts.flows.eval.eval_finetuned_model --run-dir {run_dir}\n"
    )
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--from-run", type=Path, default=None, help="Run to continue (default: latest)"
    )
    parser.add_argument("--iters", type=int, default=CONTINUE_ITERS)
    parser.add_argument("--prose-ratio", type=int, default=PROSE_RATIO)
    parser.add_argument("--mix-only", action="store_true", help="Build the mix and stop")
    args = parser.parse_args()

    if args.mix_only:
        build_tool_mix(prose_ratio=args.prose_ratio)
    else:
        continue_tool_training(
            from_run=args.from_run, iters=args.iters, prose_ratio=args.prose_ratio
        )
