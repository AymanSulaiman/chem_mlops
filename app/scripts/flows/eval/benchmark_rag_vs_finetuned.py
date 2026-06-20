"""
Benchmark: RAG (Python) vs Fine-tuned (Standard) on the golden question set.

Both modes call Ollama via HTTP. RAG mode extracts drug names from each
question, queries LanceDB (compounds + polypharmacy tables), and prepends
the result as a system message — an independent Python implementation of the
TypeScript RAG in web/src/rag.ts.

Results are written to data/benchmarks/<timestamp>/.

Usage:
    uv run python -m app.scripts.flows.eval.benchmark_rag_vs_finetuned
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.scripts.flows.eval.eval_finetuned_model import GOLDEN_BENCHMARK_PATH
from app.scripts.flows.vector_store.ingest_to_lancedb import LANCEDB_DIR
from app.scripts.flows.vector_store.query_lancedb import (
    get_compound_by_name,
    query_drug_side_effects,
    query_polypharmacy,
)

BENCHMARKS_DIR = Path("data/benchmarks")
RAG_PASS_THRESHOLD = 0.4  # minimum RAG pass rate to allow Ollama export
# 40% catches a broken RAG (empty LanceDB, gemma3:1b not available) while
# tolerating the expected gap between a base model and a domain-fine-tuned model
# on pharmaceutical keyword-match questions.

DEFAULT_FINETUNED_MODEL = "chembl-drug-chat:1b"
DEFAULT_RAG_MODEL = "gemma3:1b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

_DRUG_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_STOPWORDS = {
    "What", "Which", "How", "When", "Where", "Why", "Who",
    "The", "This", "That", "These", "Those", "There", "Their",
    "And", "But", "For", "With", "From", "Into", "Over", "Under",
    "Does", "Should", "Would", "Could", "Have", "Has", "Had",
    "Are", "Was", "Were", "Been", "Being", "CYP", "DDI",
}


# ── RAG context builder ───────────────────────────────────────────────────────


def extract_drug_candidates(text: str) -> list[str]:
    """Return title-cased drug name candidates from *text*, deduped in order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _DRUG_RE.finditer(text):
        word = m.group(0).title()
        if word not in _STOPWORDS and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def _compound_line(row: dict[str, Any]) -> str:
    parts: list[str] = [f"• {row.get('pref_name', '?')} ({row.get('chembl_id', '?')})"]
    for label, key, limit in [
        ("indications", "indications", 120),
        ("mechanism", "mechanisms", 120),
        ("metabolised by", "metabolic_enzymes", 80),
        ("warnings", "warning_types", 80),
    ]:
        val = row.get(key)
        if val:
            parts.append(f"{label}: {str(val)[:limit]}")
    return " | ".join(parts)


def build_rag_context(
    question: str,
    lancedb_dir: str | Path = LANCEDB_DIR,
) -> str | None:
    """Query LanceDB for drug names found in *question*.

    Returns a context string to prepend as a system message, or None if no
    relevant records are found.
    """
    lancedb_dir = str(lancedb_dir)
    candidates = extract_drug_candidates(question)
    if not candidates:
        return None

    lines: list[str] = []
    seen_pairs: set[frozenset[str]] = set()

    for name in candidates:
        try:
            row = get_compound_by_name(name, lancedb_dir=lancedb_dir)
            if row:
                lines.append(_compound_line(row))
        except Exception:
            pass

        try:
            for pair in query_drug_side_effects(name, n=5, lancedb_dir=lancedb_dir)[:3]:
                d1, d2 = pair.get("drug_1_name", ""), pair.get("drug_2_name", "")
                key: frozenset[str] = frozenset([d1, d2])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                effects = str(pair.get("side_effects", ""))[:100]
                lines.append(
                    f"• {d1} + {d2} polypharmacy: {effects} "
                    f"(max PRR {pair.get('max_prr', '?')})"
                )
        except Exception:
            pass

    for i, d1 in enumerate(candidates):
        for d2 in candidates[i + 1 :]:
            key = frozenset([d1, d2])
            if key in seen_pairs:
                continue
            try:
                pair = query_polypharmacy(d1, d2, lancedb_dir=lancedb_dir)
                if pair:
                    seen_pairs.add(key)
                    effects = str(pair.get("side_effects", ""))[:100]
                    lines.append(
                        f"• {d1} + {d2} polypharmacy: {effects} "
                        f"(max PRR {pair.get('max_prr', '?')})"
                    )
            except Exception:
                pass

    if not lines:
        return None

    return (
        "Use the following ChEMBL/TWOSIDES data to answer accurately. "
        "Do not invent facts not present below.\n\n" + "\n".join(lines)
    )


# ── Ollama client ─────────────────────────────────────────────────────────────


def call_ollama(
    messages: list[dict[str, str]],
    model: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> str:
    """POST to Ollama /api/chat and return the assistant reply text."""
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_response(response: str, must_contain: list[str]) -> bool:
    low = response.lower()
    return all(kw.lower() in low for kw in must_contain)


# ── Benchmark runner ──────────────────────────────────────────────────────────


def run_benchmark(
    golden_path: Path = GOLDEN_BENCHMARK_PATH,
    lancedb_dir: str | Path = LANCEDB_DIR,
    finetuned_model: str = DEFAULT_FINETUNED_MODEL,
    rag_model: str = DEFAULT_RAG_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    """Run fine-tuned and RAG modes against every golden question.

    Returns a results dict with per-question detail and aggregate pass rates.
    """
    questions = [
        json.loads(line)
        for line in Path(golden_path).read_text().splitlines()
        if line.strip()
    ]
    total = len(questions)
    ft_results: list[dict[str, Any]] = []
    rag_results: list[dict[str, Any]] = []

    for i, item in enumerate(questions, 1):
        q: str = item["question"]
        must: list[str] = item["must_contain"]
        cat: str = item.get("category", "")
        print(f"  [{i:02d}/{total}] {q[:65]}")

        try:
            ft_reply = call_ollama(
                [{"role": "user", "content": q}],
                finetuned_model,
                ollama_base_url,
            )
        except Exception as e:
            ft_reply = f"[ERROR: {e}]"
        ft_passed = score_response(ft_reply, must)
        ft_results.append(
            {"model": finetuned_model, "question": q, "category": cat,
             "must_contain": must, "keyword_match_passed": ft_passed, "response": ft_reply}
        )
        print(f"         fine-tuned : {'✓' if ft_passed else '✗'}")

        context = build_rag_context(q, lancedb_dir=lancedb_dir)
        rag_messages: list[dict[str, str]] = []
        if context:
            rag_messages.append({"role": "system", "content": context})
        rag_messages.append({"role": "user", "content": q})
        try:
            rag_reply = call_ollama(rag_messages, rag_model, ollama_base_url)
        except Exception as e:
            rag_reply = f"[ERROR: {e}]"
        rag_passed = score_response(rag_reply, must)
        rag_results.append(
            {"model": rag_model, "question": q, "category": cat,
             "must_contain": must, "keyword_match_passed": rag_passed, "response": rag_reply,
             "context_used": bool(context)}
        )
        print(f"         rag        : {'✓' if rag_passed else '✗'}  "
              f"(context: {'yes' if context else 'no'})")

    ft_pass = sum(r["keyword_match_passed"] for r in ft_results)
    rag_pass = sum(r["keyword_match_passed"] for r in rag_results)

    delta = round((rag_pass - ft_pass) / total, 4)
    winner = "rag" if delta > 0 else "finetuned" if delta < 0 else "tie"

    per_question = [
        {
            "question": ft["question"],
            "category": ft["category"],
            "must_contain": ft["must_contain"],
            "finetuned": {
                "model": ft["model"],
                "keyword_match_passed": ft["keyword_match_passed"],
                "response": ft["response"],
            },
            "rag": {
                "model": rag["model"],
                "keyword_match_passed": rag["keyword_match_passed"],
                "response": rag["response"],
                "context_used": rag["context_used"],
            },
        }
        for ft, rag in zip(ft_results, rag_results)
    ]

    return {
        "eval_type": "rag_vs_finetuned_benchmark",
        "scoring_method": "keyword_match — question passes if all must_contain keywords appear in response (case-insensitive)",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "finetuned_model": finetuned_model,
        "rag_model": rag_model,
        "total": total,
        "finetuned": {
            "model": finetuned_model,
            "pass_count": ft_pass,
            "pass_rate": round(ft_pass / total, 4),
            "results": ft_results,
        },
        "rag": {
            "model": rag_model,
            "pass_count": rag_pass,
            "pass_rate": round(rag_pass / total, 4),
            "results": rag_results,
        },
        "per_question": per_question,
        "delta_pass_rate": delta,
        "delta_description": "rag_pass_rate minus finetuned_pass_rate: positive = RAG answers more questions correctly; negative = fine-tuned answers more",
        "winner": winner,
    }


def check_rag_quality(
    results: dict[str, Any],
    threshold: float = RAG_PASS_THRESHOLD,
) -> None:
    """Raise RuntimeError if RAG pass rate falls below *threshold*.

    Called by benchmark_op in the Dagster pipeline to block the Ollama export
    when the LanceDB RAG component does not meet the minimum quality bar.
    """
    pass_rate: float = results["rag"]["pass_rate"]
    if pass_rate < threshold:
        raise RuntimeError(
            f"RAG benchmark {pass_rate:.1%} is below threshold {threshold:.1%} — "
            "blocking Ollama export. Check LanceDB ingestion and gemma3:1b availability."
        )


def _model_slug(name: str) -> str:
    """Convert a model name to a filesystem-safe slug, e.g. 'gemma3:1b' → 'gemma3_1b'."""
    return name.replace(":", "_").replace("/", "_")


def write_benchmark_artifacts(
    results: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Write <finetuned_model>_vs_<rag_model>_benchmark.json to out_dir and return the path."""
    ft_slug = _model_slug(results.get("finetuned_model", "finetuned"))
    rag_slug = _model_slug(results.get("rag_model", "rag"))
    filename = f"{ft_slug}_vs_{rag_slug}_benchmark.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    out.write_text(json.dumps(results, indent=2) + "\n")
    return out


def print_summary(results: dict[str, Any]) -> None:
    total = results["total"]
    ft = results["finetuned"]
    rag = results["rag"]
    delta = results["delta_pass_rate"]
    winner_label = "RAG" if delta > 0 else ("Fine-tuned" if delta < 0 else "Tie")

    print(f"\n{'─' * 54}")
    print(
        f"  Fine-tuned  ({results['finetuned_model']}): "
        f"{ft['pass_count']}/{total} ({ft['pass_rate']:.1%})"
    )
    print(
        f"  RAG         ({results['rag_model']}):    "
        f"{rag['pass_count']}/{total} ({rag['pass_rate']:.1%})"
    )
    print(f"  Δ pass rate : {delta:+.1%}  →  {winner_label} wins")
    print(f"{'─' * 54}")

    cats: dict[str, dict[str, int]] = {}
    for r_ft, r_rag in zip(ft["results"], rag["results"]):
        cat = r_ft.get("category", "other")
        if cat not in cats:
            cats[cat] = {"ft": 0, "rag": 0, "n": 0}
        cats[cat]["n"] += 1
        cats[cat]["ft"] += int(r_ft["keyword_match_passed"])
        cats[cat]["rag"] += int(r_rag["keyword_match_passed"])

    print(f"\n  {'Category':<30} {'Fine-tuned':>12} {'RAG':>6}")
    print(f"  {'─' * 52}")
    for cat, v in sorted(cats.items()):
        print(f"  {cat:<30} {v['ft']}/{v['n']:>10} {v['rag']}/{v['n']:>4}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark RAG vs fine-tuned on the golden question set."
    )
    parser.add_argument("--finetuned-model", default=DEFAULT_FINETUNED_MODEL)
    parser.add_argument("--rag-model", default=DEFAULT_RAG_MODEL)
    parser.add_argument("--lancedb-dir", type=Path, default=LANCEDB_DIR, metavar="PATH")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--golden", type=Path, default=GOLDEN_BENCHMARK_PATH, metavar="PATH")
    parser.add_argument("--benchmarks-dir", type=Path, default=BENCHMARKS_DIR, metavar="PATH")
    args = parser.parse_args()

    n_questions = sum(1 for ln in open(args.golden) if ln.strip())
    print("\nBenchmark  : RAG vs Fine-tuned")
    print(f"Golden set : {args.golden} ({n_questions} questions)\n")

    results = run_benchmark(
        golden_path=args.golden,
        lancedb_dir=args.lancedb_dir,
        finetuned_model=args.finetuned_model,
        rag_model=args.rag_model,
        ollama_base_url=args.ollama_base_url,
    )
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_path = write_benchmark_artifacts(results, out_dir=args.benchmarks_dir / ts)
    print_summary(results)
    print(f"Results written to {out_path}")
    sys.exit(0)
