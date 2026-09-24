"""
Model evaluation for the ChEMBL fine-tuning pipeline.

Four signals, three of which block the export:

  1. Perplexity on the held-out validation set (computed via mlx_lm Python API
     on valid.jsonl).  Lower is better.  GATE: a fine-tuned model worse than
     the base model blocks export.
  2. Tool-call benchmark — tool-shaped questions about held-out drugs, scored
     through the same resolver the agent loop uses.  GATE: parse rate must meet
     TOOL_CALL_PARSE_THRESHOLD, because a model that cannot emit a usable tool
     call cannot answer a lookup at all.
  3. Tool-result benchmark — the other half of the agent loop: handed a tool
     result, does the model read it or echo its shape?  RECORDED, NOT GATED.
     This is the number that decides whether the temporary bridge in
     web/src/tools.ts can be deleted (roadmap item 3).
  4. Golden benchmark — drug Q&A pairs in golden.jsonl, scored on a
     "must_contain" keyword list.  RECORDED, NOT GATED: every question asks for
     a fact about a molecule deliberately held out of training, which is a
     lookup rather than something weights can supply.  See eval_flow.

Artifacts written to data/eval/<run>/:
  finetuned_eval_metrics.json         — summary (perplexity, both rates, gate result)
  finetuned_golden_results.jsonl      — per-question detail (question, response, passed)
  finetuned_tool_call_results.jsonl   — per-call detail (named tool, resolved tool, args)
  finetuned_tool_result_results.jsonl — per-answer detail (expected marker, grounded, prose)
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
EVAL_PASS_THRESHOLD = 0.7  # golden pass rate that would clear the gate — see eval_flow
TOOL_CALL_PARSE_THRESHOLD = 0.5  # roadmap item 3 acceptance: a stated majority parses

# Mirrored from build_drug_interaction_dataset: the model is trained on this
# exact header and this truncation, and the server sends both. A benchmark that
# drifts from them measures a prompt the model never sees.
TOOL_RESULT_HEADER = "### Tool result"
TOOL_RESULT_LIMIT = 2_000

# ── Tool-call benchmark ───────────────────────────────────────────────────────
# Roadmap item 3's acceptance criterion: does the model emit a *valid* tool call
# for a question a tool can answer, on drugs it never saw in training?
#
# ponytail: TOOL_SYSTEM_PROMPT is duplicated from web/src/tools.ts rather than
# shared. A parity test keeps the tool names honest; move both to one JSON file
# if the prompt text itself starts drifting.
#
# query_compounds is deliberately absent: it is the only tool with no training
# records, and advertising it made the model copy its example SMILES verbatim
# for unrelated questions (17 of 40 calls on run 20260920_114710_tools). It is
# still callable — see the `advertised` flag in web/src/tools.ts.
#
# NOT SENT TO THE FINE-TUNE. Training records carry no system prompt at all
# (_tool_call_record is bare ### Question / ### Answer), so prepending this tool
# list at inference is out-of-distribution text the model copies from rather
# than reasons over: every polypharmacy question collapsed onto whichever tool
# example sat last in the list. Removing it fixed routing outright. Kept for a
# general tool-capable model, which does need to be told what the tools are,
# and passed via the `system_prompt` argument of the benchmarks below.
TOOL_SYSTEM_PROMPT = """You can look up real pharmacological data with tools. To call one, reply with
ONLY a JSON object and nothing else:
{"tool": "<name>", "args": {...}}

Tools:
- get_compound_by_name {"name": "Aspirin"} — ChEMBL record for a drug by name: chembl_id, molecular weight, properties.
- query_drug_side_effects {"drug_name": "Warfarin", "n": 10} — Known interaction partners of one drug, strongest TWOSIDES signal first.
- query_polypharmacy {"drug_1": "Warfarin", "drug_2": "Aspirin"} — Side effects reported for one specific drug pair taken together.
- draw_molecule {"name": "Ibuprofen"} — Draw a molecule. Give the drug NAME and the structure is looked up in ChEMBL — never invent a SMILES string. Pass {"smiles": "..."} only for a structure the user typed.

The tool result comes back as a '### Tool result' message. Then answer the
question in prose using it. Do not invent ChEMBL IDs or side effects: look them
up. If no tool is needed, just answer."""

# question template, expected tool, argument the call must carry
TOOL_CASES: list[tuple[str, str, str]] = [
    ("What is the molecular weight of {drug}?", "get_compound_by_name", "name"),
    ("Draw {drug}.", "draw_molecule", "name"),
    ("Is it safe to take {drug} with warfarin?", "query_polypharmacy", "drug_1"),
    ("Which drugs interact with {drug}?", "query_drug_side_effects", "drug_name"),
]
TOOL_CALL_DRUGS = 10  # holdout molecules to test; each costs len(TOOL_CASES) generate calls


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
    model, tokenizer = mlx_lm_load(  # type: ignore
        str(mlx_model_dir),
        adapter_path=str(adapter_dir) if adapter_dir is not None else None,
    )

    args = types.SimpleNamespace(
        data=str(data_dir),
        train=False,
        test=False,
        hf_dataset=None,
    )
    _, valid_set, _ = mlx_load_dataset(args, tokenizer)  # type: ignore

    mean_loss = mlx_evaluate(
        model=model,
        dataset=CacheDataset(valid_set),
        batch_size=4,
        num_batches=num_batches,
    )
    return math.exp(mean_loss)


# ── Golden benchmark ──────────────────────────────────────────────────────────


def _first_json_object(text: str) -> dict[str, Any] | None:
    """First balanced JSON object in *text*, or None.

    The model wraps its call in chatter or a ``` fence as often as not, so a
    plain json.loads of the whole reply understates how often it got it right.
    Mirrors parseToolCall in web/src/tools.ts.
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _preamble(system_prompt: str) -> str:
    """The system prompt with its separator, or nothing at all."""
    return f"{system_prompt}\n\n" if system_prompt else ""


def _generate(
    mlx_model_dir: Path, adapter_dir: Path, prompt: str, max_tokens: int
) -> str:
    """One completion from the fine-tuned model. Returns stdout, stripped.

    Raises RuntimeError if the subprocess fails or produces nothing. That
    distinction matters: a *bad* reply is data the benchmarks score, but a
    *missing* one is a broken run, and scoring it as a wrong answer manufactures
    findings. Running two evals at once exhausted Metal and yielded 40 empty
    completions, which the rates reported as "0.0% routing" — indistinguishable
    from a real regression except that golden, which shares no code with the
    tool prompt, went to zero at the same time.
    """
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
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--verbose",
        "False",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    response = proc.stdout.strip()
    if proc.returncode != 0 or not response:
        raise RuntimeError(
            f"mlx_lm generate failed (exit {proc.returncode}, {len(response)} chars). "
            f"Is another eval or training run using the GPU? {proc.stderr.strip()[-300:]}"
        )
    return response


def _holdout_drugs(golden_path: Path, limit: int) -> list[str]:
    """Drug names from the golden benchmark — molecules withheld from training."""
    names: list[str] = []
    for line in golden_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        # "drug" was added later; older golden.jsonl files only have the question.
        drug = item.get("drug")
        if not drug:
            question = item.get("question", "")
            drug = question.removeprefix("What does ").removesuffix(" target?").strip()
        if drug and drug not in names:
            names.append(drug)
        if len(names) >= limit:
            break
    return names


def run_tool_call_benchmark(
    mlx_model_dir: Path,
    adapter_dir: Path,
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    drugs: int = TOOL_CALL_DRUGS,
    max_tokens: int = 120,
    model_label: str = "chembl-drug-chat (MLX LoRA)",
    system_prompt: str = "",
) -> dict[str, Any]:
    """Measure tool-call quality on held-out drugs (roadmap item 3).

    *system_prompt* defaults to none, matching how the fine-tune was trained.
    Pass TOOL_SYSTEM_PROMPT to measure a model that needs to be told the tools.

    Three rates, because they fail independently and the fixes differ:
      parse_rate   — a JSON object came back at all (a decoding problem)
      known_rate   — it named a tool that exists (a naming problem)
      correct_rate — it named the *right* tool with the expected argument
                     (a routing problem, the one that decides whether the
                     agent is real)

    Returns the rates plus per-case detail. Never raises on a bad reply: an
    unparseable answer is a result, not an error.
    """
    from app.scripts.flows.vector_store.tools import TOOLS, resolve_tool_call

    known_tools = set(TOOLS)
    results: list[dict[str, Any]] = []

    for drug in _holdout_drugs(golden_path, drugs):
        for template, expected_tool, expected_arg in TOOL_CASES:
            question = template.format(drug=drug)
            response = _generate(
                mlx_model_dir,
                adapter_dir,
                f"{_preamble(system_prompt)}### Question\n{question}\n\n### Answer\n",
                max_tokens,
            )

            call = _first_json_object(response)
            named = call.get("tool") or call.get("name") if call else None
            args = (call.get("args") or call.get("arguments") or {}) if call else {}
            parsed = isinstance(named, str)
            # Score what the serving path runs, not the raw string: a near-miss
            # name the agent loop resolves is a call that works for the user.
            tool = named
            if isinstance(named, str) and isinstance(args, dict):
                tool, args = resolve_tool_call(named, args)
            known = parsed and tool in known_tools
            correct = known and tool == expected_tool and expected_arg in args

            results.append(
                {
                    "model": model_label,
                    "drug": drug,
                    "question": question,
                    "expected_tool": expected_tool,
                    "called_tool": tool,
                    "named_tool": named,
                    "args": args if isinstance(args, dict) else {},
                    "parsed": parsed,
                    "known_tool": known,
                    "correct_tool": correct,
                    "response": response[:400],
                }
            )

    total = len(results) or 1
    return {
        "total": len(results),
        "parse_rate": sum(r["parsed"] for r in results) / total,
        "known_rate": sum(r["known_tool"] for r in results) / total,
        "correct_rate": sum(r["correct_tool"] for r in results) / total,
        "results": results,
    }


# ── Tool-result benchmark ─────────────────────────────────────────────────────
# The other half of the agent loop. The tool-call benchmark asks "does it ask
# for the right lookup"; this asks "having been handed the answer, can it read
# it". It was written to decide whether a deterministic bridge in web/src/tools.ts
# could be removed — that bridge existed because the model, handed a tool result,
# echoed the JSON shape back instead of using it. This number reached 100% and
# the bridge is gone; the benchmark stays, because it is what would catch the
# behaviour coming back.
#
# Every expected value below is invented. A real molecular weight could be
# recalled from training; 481.27 cannot, so an answer containing it is proof the
# model read the result rather than its own weights. The drugs are held out for
# the same reason.
TOOL_RESULT_CASES: list[tuple[str, str, dict[str, Any], Any, str]] = [
    (
        "What is the molecular weight of {drug}?",
        "get_compound_by_name",
        {"name": "{drug}"},
        {
            "chembl_id": "CHEMBL9900001",
            "pref_name": "{drug}",
            "mw_freebase": 481.27,
            "full_molformula": "C23H31N5O4",
        },
        "481.27",
    ),
    (
        "Draw {drug}.",
        "draw_molecule",
        {"name": "{drug}"},
        {
            "image": "the picture is already displayed to the user",
            "smiles": "CC(=O)Nc1ccc(O)cc1",
            "source": "ChEMBL",
            "chembl_id": "CHEMBL9900002",
            "pref_name": "{drug}",
            "full_molformula": "C23H31N5O4",
        },
        "C23H31N5O4",
    ),
    (
        "Which drugs interact with {drug}?",
        "query_drug_side_effects",
        {"drug_name": "{drug}", "n": 3},
        [{"partner": "Zalbovir", "side_effect": "tachycardia", "prr": 9.81}],
        "Zalbovir",
    ),
    (
        "Is it safe to take {drug} with warfarin?",
        "query_polypharmacy",
        {"drug_1": "{drug}", "drug_2": "warfarin"},
        [{"side_effect": "photosensitivity", "prr": 7.43}],
        "7.43",
    ),
]


def run_tool_result_benchmark(
    mlx_model_dir: Path,
    adapter_dir: Path,
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    drugs: int = TOOL_CALL_DRUGS,
    max_tokens: int = 120,
    model_label: str = "chembl-drug-chat (MLX LoRA)",
    system_prompt: str = "",
) -> dict[str, Any]:
    """Measure whether the model can *use* a tool result it is handed.

    The prompt is assembled exactly as ``_tool_call_record`` lays out a training
    example and exactly as the server sends one at inference — system prompt,
    question, the model's own call, then the result inside a ``### Question``
    block. Train-serve mismatch is the failure this whole item is about, so the
    benchmark must not invent its own format.

    Two rates, because they fail independently:
      grounded_rate — the answer contains a value that appears *only* in the
                      tool result, so it cannot have come from memory
      prose_rate    — the answer is prose rather than an echo of the JSON shape,
                      which is the specific failure the temporary bridge exists
                      to hide

    Returns the rates plus per-case detail. Never raises on a bad reply.
    """
    results: list[dict[str, Any]] = []

    for drug in _holdout_drugs(golden_path, drugs):
        for template, tool, arg_shape, result_shape, expected in TOOL_RESULT_CASES:
            question = template.format(drug=drug)
            args = {k: v.format(drug=drug) if isinstance(v, str) else v
                    for k, v in arg_shape.items()}
            result = json.loads(json.dumps(result_shape).replace("{drug}", drug))
            call = json.dumps({"tool": tool, "args": args})
            body = json.dumps(result, default=str)

            prompt = (
                f"{_preamble(system_prompt)}"
                f"### Question\n{question}\n\n"
                f"### Answer\n{call}\n\n"
                f"### Question\n{TOOL_RESULT_HEADER} ({tool})\n{body}\n\n"
                f"### Answer\n"
            )
            response = _generate(mlx_model_dir, adapter_dir, prompt, max_tokens)

            grounded = expected.lower() in response.lower()
            # An answer that opens with a brace or re-emits a tool call is the
            # model mimicking the shape it was just shown, not answering.
            echoed = response.lstrip().startswith("{") or '"tool"' in response
            results.append(
                {
                    "model": model_label,
                    "drug": drug,
                    "question": question,
                    "tool": tool,
                    "expected_in_answer": expected,
                    "grounded": grounded,
                    "prose": not echoed,
                    "response": response[:400],
                }
            )

    total = len(results) or 1
    return {
        "total": len(results),
        "grounded_rate": sum(r["grounded"] for r in results) / total,
        "prose_rate": sum(r["prose"] for r in results) / total,
        "results": results,
    }


def run_golden_benchmark(
    mlx_model_dir: Path,
    adapter_dir: Path,
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    max_tokens: int = 300,
    model_label: str = "chembl-drug-chat (MLX LoRA)",
    use_tools: bool = True,
) -> dict[str, Any]:
    """
    Run the golden benchmark through the agent loop and keyword-score the answer.

    **Golden is a lookup benchmark.** Every question asks for a fact about a
    molecule item 1 deliberately withheld from training, so the answer cannot be
    in the weights — it is in the vector store, and the model's job is to go and
    get it. Scoring the bare model here measured whether it would *guess* a
    protein, which it did, wrongly, 39 times out of 40.

    So: one turn to let the model ask for a lookup, the tool actually runs, the
    result comes back, and the answer it then writes is what gets scored. A
    model that answers without asking is scored on that answer directly, so a
    question needing no tool still works.

    Scoring: keyword_match — a question passes if every word in must_contain
    appears anywhere in the final answer (case-insensitive).

    Args:
        mlx_model_dir: Path to the MLX base model directory.
        adapter_dir:   LoRA adapter to evaluate.
        golden_path:   Path to golden.jsonl benchmark file.
        max_tokens:    Maximum tokens to generate per question.
        model_label:   Human-readable model identifier written into each result.
        use_tools:     Run the agent loop. False scores the bare model, which is
                       only meaningful for questions the weights should answer.

    Returns:
        Dict with keys: pass_count, total, pass_rate, tool_used_count,
        results (per-question list).
    """
    from app.scripts.flows.vector_store.tools import resolve_tool_call, run_tool

    questions = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]

    passed = 0
    tool_used = 0
    results: list[dict[str, Any]] = []

    for item in questions:
        question: str = item["question"]
        must_contain = [kw.lower() for kw in item["must_contain"]]

        first = _generate(
            mlx_model_dir, adapter_dir, f"### Question\n{question}\n\n### Answer\n", max_tokens
        )

        response = first
        tool_name: str | None = None
        call = _first_json_object(first) if use_tools else None
        named = (call.get("tool") or call.get("name")) if call else None
        if call is not None and isinstance(named, str):
            args = call.get("args") or call.get("arguments") or {}
            tool_name, resolved_args = resolve_tool_call(
                named, args if isinstance(args, dict) else {}
            )
            outcome = run_tool(tool_name, resolved_args)
            body = json.dumps(outcome.get("result", outcome), default=str)[:TOOL_RESULT_LIMIT]
            # Same layout the server sends and the dataset trains on.
            response = _generate(
                mlx_model_dir,
                adapter_dir,
                f"### Question\n{question}\n\n"
                f"### Answer\n{json.dumps({'tool': tool_name, 'args': resolved_args})}\n\n"
                f"### Question\n{TOOL_RESULT_HEADER} ({tool_name})\n{body}\n\n"
                f"### Answer\n",
                max_tokens,
            )
            tool_used += 1

        hit = all(kw in response.lower() for kw in must_contain)
        if hit:
            passed += 1

        results.append(
            {
                "model": model_label,
                "question": question,
                "category": item.get("category", ""),
                "must_contain": item["must_contain"],
                "keyword_match_passed": hit,
                "tool_called": tool_name,
                "response": response.strip(),
            }
        )

    total = len(questions)
    return {
        "pass_count": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0.0,
        "tool_used_count": tool_used,
        "results": results,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────


def eval_flow(
    run_dir: Path,
    data_dir: Path = Path("data/llm_finetune"),
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    pass_threshold: float = EVAL_PASS_THRESHOLD,
    num_batches: int = 50,
    eval_output_dir: Path | None = None,
    tool_call_drugs: int = TOOL_CALL_DRUGS,
    tool_call_threshold: float = TOOL_CALL_PARSE_THRESHOLD,
) -> dict[str, Any]:
    """
    Full evaluation: perplexity check, golden benchmark, tool-call benchmark.

    Writes metrics.json and golden_results.jsonl to eval_output_dir
    (default: data/eval/<run_name>/).
    Raises RuntimeError if the fine-tuned model regresses on perplexity, or if
    the golden pass rate or tool-call parse rate falls below its threshold —
    blocking the downstream Ollama export. A threshold of 0 disables that gate,
    which is how an intermediate adapter is measured without blocking on it.

    Args:
        run_dir:          Fine-tuning artifact directory (e.g. artifacts/20260615_120000).
        data_dir:         Directory containing train/valid JSONL splits.
        golden_path:      Path to golden.jsonl benchmark file.
        pass_threshold:   Minimum golden pass rate [0, 1]; 0 disables the gate.
        num_batches:      Batches to use for perplexity evaluation (-1 for all).
        eval_output_dir:  Where to write metrics/results. Defaults to
                          data/eval/<run_dir.name>/.
        tool_call_drugs:  Held-out molecules to test tool calls against.
        tool_call_threshold: Minimum tool-call parse rate [0, 1]; 0 disables it.

    Returns:
        Metrics dict (same content as metrics.json).

    Raises:
        RuntimeError: If perplexity regresses, or a gated rate is too low.
    """
    mlx_model_dir = run_dir / DEFAULT_MLX_SUBDIR
    adapter_dir = run_dir / DEFAULT_ADAPTER_SUBDIR
    eval_dir = eval_output_dir if eval_output_dir is not None else Path("data/eval") / run_dir.name
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
    model_label = f"chembl-drug-chat (MLX LoRA, run={run_dir.name})"
    print(f"\nRunning golden benchmark ({golden_path}) ...")
    golden = run_golden_benchmark(mlx_model_dir, adapter_dir, golden_path, model_label=model_label)
    print(
        f"  Pass rate: {golden['pass_count']}/{golden['total']} ({golden['pass_rate']:.1%})"
        f" · looked it up in {golden.get('tool_used_count', 0)}/{golden['total']}"
    )

    # ── 3. Tool calls (the gate) ──────────────────────────────────
    print(f"\nRunning tool-call benchmark ({tool_call_drugs} held-out drugs) ...")
    tools = run_tool_call_benchmark(
        mlx_model_dir, adapter_dir, golden_path, drugs=tool_call_drugs, model_label=model_label
    )
    print(
        f"  Parsed {tools['parse_rate']:.1%} · known tool {tools['known_rate']:.1%} · "
        f"right tool {tools['correct_rate']:.1%} of {tools['total']}"
    )
    if tools["correct_rate"] < 1.0:
        print(
            f"  note: the agent routes to the right tool {tools['correct_rate']:.1%} of the time. "
            "Not gated yet — see roadmap item 3."
        )

    # ── 3b. Tool results (measured, not gated) ────────────────────
    print(f"\nRunning tool-result benchmark ({tool_call_drugs} held-out drugs) ...")
    tool_results = run_tool_result_benchmark(
        mlx_model_dir, adapter_dir, golden_path, drugs=tool_call_drugs, model_label=model_label
    )
    print(
        f"  Used the result {tool_results['grounded_rate']:.1%} · "
        f"answered in prose {tool_results['prose_rate']:.1%} of {tool_results['total']}"
    )

    # ── 4. Write artifacts ────────────────────────────────────────
    metrics: dict[str, Any] = {
        "eval_type": "finetuned_model_eval",
        "scoring_method": "keyword_match — question passes if all must_contain keywords appear in response (case-insensitive)",
        "run": run_dir.name,
        "model": model_label,
        "mlx_model_dir": str(run_dir / DEFAULT_MLX_SUBDIR),
        "adapter_dir": str(run_dir / DEFAULT_ADAPTER_SUBDIR),
        "baseline_perplexity": round(baseline_ppl, 3),
        "finetuned_perplexity": round(finetuned_ppl, 3),
        "golden_pass_count": golden["pass_count"],
        "golden_total": golden["total"],
        "golden_pass_rate": round(golden["pass_rate"], 4),
        # How often the model asked for a lookup instead of answering from
        # memory. A low pass rate with a low count here is a routing problem;
        # a low pass rate with a high count is a reading or data problem.
        "golden_tool_used_count": golden.get("tool_used_count", 0),
        "pass_threshold": pass_threshold,
        "tool_call_total": tools["total"],
        "tool_call_parse_rate": round(tools["parse_rate"], 3),
        "tool_call_known_rate": round(tools["known_rate"], 3),
        "tool_call_correct_rate": round(tools["correct_rate"], 3),
        "tool_result_total": tool_results["total"],
        "tool_result_grounded_rate": round(tool_results["grounded_rate"], 3),
        "tool_result_prose_rate": round(tool_results["prose_rate"], 3),
        "tool_call_parse_threshold": tool_call_threshold,
        # A threshold of 0 cannot fire, so the run is measured and not gated —
        # which is how an intermediate adapter is evaluated. Record that
        # honestly rather than claiming a gate that could never fail.
        "tool_call_gated": tool_call_threshold > 0,
        # 0 cannot fire, so the run is measured and not gated — how an
        # intermediate adapter is evaluated. Same convention as the tool-call
        # threshold above.
        "golden_gated": pass_threshold > 0,
        # Named so "eval_gate_passed" cannot be read as "every number is good".
        # It means these gates passed, and nothing more.
        "gates_applied": ["perplexity"]
        + (["golden_pass_rate"] if pass_threshold > 0 else [])
        + (["tool_call_parse_rate"] if tool_call_threshold > 0 else []),
        "ungated_metrics": [
            "tool_call_correct_rate",
            "tool_result_grounded_rate",
            "tool_result_prose_rate",
        ]
        + ([] if tool_call_threshold > 0 else ["tool_call_parse_rate"])
        + ([] if pass_threshold > 0 else ["golden_pass_rate"]),
        "eval_gate_passed": True,
    }

    metrics_path = eval_dir / "finetuned_eval_metrics.json"
    detail_path = eval_dir / "finetuned_golden_results.jsonl"
    detail_path.write_text("\n".join(json.dumps(r) for r in golden["results"]) + "\n")
    (eval_dir / "finetuned_tool_call_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in tools["results"]) + "\n"
    )
    (eval_dir / "finetuned_tool_result_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in tool_results["results"]) + "\n"
    )

    # ── 5. Gate checks ────────────────────────────────────────────
    failures: list[str] = []

    if finetuned_ppl > baseline_ppl:
        failures.append(
            f"Perplexity regression: fine-tuned {finetuned_ppl:.3f} > baseline {baseline_ppl:.3f}"
        )

    # Golden gates again. It was demoted when it scored the *bare model* on
    # "What does {drug} target?" about molecules item 1 deliberately removed
    # from training — an unpassable bar that blocked every export for a
    # capability the fine-tune was never meant to have. It now runs through the
    # agent loop, where that question is an ordinary lookup, and clears the
    # threshold with room (97.5% on 20260921_053213_tools).
    #
    # It is the only gate on *answer quality*. Perplexity and the tool-call rate
    # would both pass a model that routes perfectly and then writes nonsense
    # from the result it was handed.
    if golden["pass_rate"] < pass_threshold:
        failures.append(
            f"Golden benchmark {golden['pass_rate']:.1%} is below threshold "
            f"{pass_threshold:.1%} (looked it up in "
            f"{golden.get('tool_used_count', 0)}/{golden['total']})"
        )

    if tools["parse_rate"] < tool_call_threshold:
        failures.append(
            f"Tool-call parse rate {tools['parse_rate']:.1%} is below threshold "
            f"{tool_call_threshold:.1%} — the model is not emitting usable tool calls"
        )

    if failures:
        metrics["eval_gate_passed"] = False
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        raise RuntimeError(
            "Eval failed — blocking Ollama export:\n" + "\n".join(f"  • {f}" for f in failures)
        )

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    # Deliberately not "✓ Eval passed": the gates are a floor, not a verdict on
    # answer quality, and golden at 5% under a bare "passed" reads as a green
    # light on a model that answers 2 of 40 benchmark questions correctly.
    gated = ", ".join(
        ["perplexity"]
        + (["golden"] if pass_threshold > 0 else [])
        + (["tool-call parse"] if tool_call_threshold > 0 else [])
    )
    print(
        f"\n✓ Gates passed ({gated}) — tool-call parse {tools['parse_rate']:.1%}, "
        f"routing {tools['correct_rate']:.1%}, golden {golden['pass_rate']:.1%}"
        f"{'' if tool_call_threshold > 0 else ' — all recorded, NOT gated'}"
    )
    print(f"  Metrics written to {metrics_path}")
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
        help=f"Golden pass rate to report against, not gated (default: {EVAL_PASS_THRESHOLD})",
    )
    parser.add_argument(
        "--tool-call-threshold",
        type=float,
        default=TOOL_CALL_PARSE_THRESHOLD,
        metavar="FLOAT",
        help=f"Minimum tool-call parse rate; blocks export (default: {TOOL_CALL_PARSE_THRESHOLD})",
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
        tool_call_threshold=args.tool_call_threshold,
    )
