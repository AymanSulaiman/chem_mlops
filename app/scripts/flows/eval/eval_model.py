"""
Model evaluation for the ChEMBL fine-tuning pipeline.

Two complementary signals:
  1. Perplexity on the held-out validation set (computed via mlx_lm Python API
     on valid.jsonl).  Lower is better.  If the fine-tuned model is *worse* than
     the base model the export step is blocked.
  2. Golden benchmark — 20 curated drug Q&A pairs in golden.jsonl.
     Each item has a "must_contain" list of keywords that a correct answer must
     include (case-insensitive).  Pass rate must meet EVAL_PASS_THRESHOLD.

Artifacts written to artifacts/<run>/eval/:
  metrics.json       — summary (perplexity, pass rate, threshold result)
  golden_results.jsonl — per-question detail (question, response, passed)
"""

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from app.scripts.flows.finetuning.export_to_ollama import (
    ARTIFACTS_DIR,
    DEFAULT_ADAPTER_SUBDIR,
    DEFAULT_MLX_SUBDIR,
    latest_run_dir,
)

# Lazy mlx_lm imports — aliased at module level so tests can patch them.
# These are only resolved when the functions are called (not at import time),
# which keeps CI fast on machines without Apple Silicon / MLX.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset
    from mlx_lm.tuner.trainer import evaluate as mlx_evaluate
except ImportError:  # non-Apple-Silicon environments (CI)
    mlx_lm_load = None  # type: ignore[assignment] # type: ignore
    mlx_load_dataset = None  # type: ignore[assignment] # type: ignore
    mlx_evaluate = None  # type: ignore[assignment] # type: ignore
    CacheDataset = None  # type: ignore[assignment] # type: ignore

GOLDEN_BENCHMARK_PATH = Path(__file__).parent / "golden.jsonl"
EVAL_PASS_THRESHOLD = 0.50  # 70 % of golden questions must pass


# ── Perplexity ────────────────────────────────────────────────────────────────


def run_perplexity_eval(
    mlx_model_dir: Path,
    data_dir: Path,
    adapter_dir: Path | None = None,
    num_batches: int = 50,
) -> float:
    """
    Evaluate perplexity on valid.jsonl using the mlx_lm Python API directly.

    Uses mlx_lm.load (with optional adapter) + mlx_lm trainer.evaluate so no
    test.jsonl is required and no subprocess forking is needed.

    Args:
        mlx_model_dir: Path to the MLX base model directory.
        data_dir:       Directory containing train.jsonl / valid.jsonl splits.
        adapter_dir:    Optional LoRA adapter to apply before evaluating.
                        Omit to get baseline (unfinetuned) perplexity.
        num_batches:    Number of validation batches to evaluate (-1 for all).

    Returns:
        Perplexity as a float (exp of mean cross-entropy loss).
    """
    model, tokenizer = mlx_lm_load( # type: ignore
        str(mlx_model_dir),
        adapter_path=str(adapter_dir) if adapter_dir is not None else None,
    )

    args = types.SimpleNamespace(
        data=str(data_dir),
        train=False,
        test=False,
        hf_dataset=None,
    )
    _, valid_set, _ = mlx_load_dataset(args, tokenizer) # type: ignore

    mean_loss = mlx_evaluate(
        model=model,
        dataset=CacheDataset(valid_set),
        batch_size=4,
        num_batches=num_batches,
    )
    return math.exp(mean_loss)


# ── Golden benchmark ──────────────────────────────────────────────────────────


def run_golden_benchmark(
    mlx_model_dir: Path,
    adapter_dir: Path,
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """
    Run the golden benchmark: generate a response per question and check that
    every keyword in must_contain appears in the (lowercased) response.

    Args:
        mlx_model_dir: Path to the MLX base model directory.
        adapter_dir:   LoRA adapter to evaluate.
        golden_path:   Path to golden.jsonl benchmark file.
        max_tokens:    Maximum tokens to generate per question.

    Returns:
        Dict with keys: pass_count, total, pass_rate, results (per-question list).
    """
    questions = [
        json.loads(line)
        for line in golden_path.read_text().splitlines()
        if line.strip()
    ]

    passed = 0
    results: list[dict[str, Any]] = []

    for item in questions:
        question: str = item["question"]
        must_contain = [kw.lower() for kw in item["must_contain"]]

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            str(mlx_model_dir),
            "--adapter-path",
            str(adapter_dir),
            "--prompt",
            f"### Question\n{question}\n\n### Answer\n",
            "--max-tokens",
            str(max_tokens),
            "--verbose",
            "False",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        response = proc.stdout

        hit = all(kw in response.lower() for kw in must_contain)
        if hit:
            passed += 1

        results.append(
            {
                "question": question,
                "category": item.get("category", ""),
                "must_contain": item["must_contain"],
                "passed": hit,
                "response": response.strip(),
            }
        )

    total = len(questions)
    return {
        "pass_count": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0.0,
        "results": results,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────


def eval_flow(
    run_dir: Path,
    data_dir: Path = Path("data/llm_finetune"),
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    pass_threshold: float = EVAL_PASS_THRESHOLD,
    num_batches: int = 50,
) -> dict[str, Any]:
    """
    Full evaluation: perplexity check + golden benchmark.

    Writes artifacts/<run>/eval/metrics.json and golden_results.jsonl.
    Raises RuntimeError if the fine-tuned model regresses on perplexity or
    if the golden benchmark pass rate falls below pass_threshold — blocking
    the downstream Ollama export.

    Args:
        run_dir:        Fine-tuning artifact directory (e.g. artifacts/20260615_120000).
        data_dir:       Directory containing train/valid JSONL splits.
        golden_path:    Path to golden.jsonl benchmark file.
        pass_threshold: Minimum acceptable golden benchmark pass rate [0, 1].
        num_batches:    Batches to use for perplexity evaluation (-1 for all).

    Returns:
        Metrics dict (same content as metrics.json).

    Raises:
        RuntimeError: If perplexity regresses or golden pass rate is too low.
    """
    mlx_model_dir = run_dir / DEFAULT_MLX_SUBDIR
    adapter_dir = run_dir / DEFAULT_ADAPTER_SUBDIR
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Evaluating run: {run_dir.name}")
    print(f"{'=' * 60}\n")

    # ── 1. Perplexity ─────────────────────────────────────────────
    print("Running perplexity eval — base model...")
    baseline_ppl = run_perplexity_eval(mlx_model_dir, data_dir, num_batches=num_batches)
    print(f"  Baseline perplexity : {baseline_ppl:.3f}")

    print("Running perplexity eval — fine-tuned model...")
    finetuned_ppl = run_perplexity_eval(
        mlx_model_dir, data_dir, adapter_dir=adapter_dir, num_batches=num_batches
    )
    print(f"  Fine-tuned perplexity: {finetuned_ppl:.3f}")

    # ── 2. Golden benchmark ───────────────────────────────────────
    print(f"\nRunning golden benchmark ({golden_path}) ...")
    golden = run_golden_benchmark(mlx_model_dir, adapter_dir, golden_path)
    print(
        f"  Pass rate: {golden['pass_count']}/{golden['total']}"
        f" ({golden['pass_rate']:.1%})"
    )

    # ── 3. Write artifacts ────────────────────────────────────────
    metrics: dict[str, Any] = {
        "run": run_dir.name,
        "baseline_perplexity": round(baseline_ppl, 3),
        "finetuned_perplexity": round(finetuned_ppl, 3),
        "golden_pass": golden["pass_count"],
        "golden_total": golden["total"],
        "golden_pass_rate": round(golden["pass_rate"], 4),
        "pass_threshold": pass_threshold,
        "passed": True,
    }

    metrics_path = eval_dir / "metrics.json"
    detail_path = eval_dir / "golden_results.jsonl"
    detail_path.write_text(
        "\n".join(json.dumps(r) for r in golden["results"]) + "\n"
    )

    # ── 4. Gate checks ────────────────────────────────────────────
    failures: list[str] = []

    if finetuned_ppl > baseline_ppl:
        failures.append(
            f"Perplexity regression: fine-tuned {finetuned_ppl:.3f}"
            f" > baseline {baseline_ppl:.3f}"
        )

    if golden["pass_rate"] < pass_threshold:
        failures.append(
            f"Golden benchmark {golden['pass_rate']:.1%}"
            f" is below threshold {pass_threshold:.1%}"
        )

    if failures:
        metrics["passed"] = False
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        raise RuntimeError(
            "Eval failed — blocking Ollama export:\n"
            + "\n".join(f"  • {f}" for f in failures)
        )

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\n✓ Eval passed — metrics written to {metrics_path}")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuning run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Artifact run directory. Defaults to the most recent in artifacts/.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=EVAL_PASS_THRESHOLD,
        metavar="FLOAT",
        help=f"Minimum golden pass rate (default: {EVAL_PASS_THRESHOLD})",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        metavar="N",
        help="Batches for perplexity eval, -1 for all (default: 50)",
    )
    args = parser.parse_args()

    _run_dir = args.run_dir or latest_run_dir(ARTIFACTS_DIR)
    eval_flow(
        run_dir=_run_dir,
        pass_threshold=args.pass_threshold,
        num_batches=args.num_batches,
    )
