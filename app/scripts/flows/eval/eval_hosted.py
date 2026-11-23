"""
Golden benchmark against the served chat app: model + tool loop + LanceDB,
exactly what a visitor gets. Scores each golden.jsonl answer on must_contain.

Local copy of the deployed stack (same images, same model as vm.sh's RUN):
  CHATS_PER_HOUR=1000 MODEL_DIR=artifacts/gcp_v1_tools/ollama docker compose up --build -d
  uv run python -m app.scripts.flows.eval.eval_hosted

Live demo (signed in; copy the `session` cookie from the browser):
  EVAL_COOKIE='session=...' uv run python -m app.scripts.flows.eval.eval_hosted \
      --url https://demo.chemmlops.app --limit 15
The demo allows 20 chats/hour per account and takes the only seat while it runs.

Writes data/eval/hosted_<timestamp>/{metrics.json,results.jsonl}.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from app.scripts.flows.eval.eval_finetuned_model import GOLDEN_BENCHMARK_PATH

EVAL_DIR = Path("data/eval")


def ask(url: str, question: str, cookie: str = "", timeout: float = 300) -> dict:
    """One chat turn through /api/chat; folds the NDJSON event stream into one result."""
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps({"messages": [{"role": "user", "content": question}]}).encode(),
        headers={"content-type": "application/json", "cookie": cookie},
    )
    answer, tools, errors = [], [], []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        model = response.headers.get("x-model", "")
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if "message" in event:
                answer.append(event["message"].get("content", ""))
            if "tool" in event:
                tools.append(event["tool"].get("tool"))
            if event.get("toolResult", {}).get("error"):
                errors.append(event["toolResult"]["error"])
            if "error" in event:
                errors.append(event["error"])
    return {"model": model, "response": "".join(answer), "tools": tools, "errors": errors}


def passed(response: str, must_contain: list[str]) -> bool:
    text = response.lower()
    return all(keyword.lower() in text for keyword in must_contain)


def run(
    url: str, cookie: str = "", limit: int | None = None, golden_path: Path = GOLDEN_BENCHMARK_PATH
) -> dict:
    cases = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()][
        :limit
    ]
    out_dir = EVAL_DIR / f"hosted_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out_dir / "results.jsonl", "w") as f:
        for i, case in enumerate(cases, 1):
            start = time.monotonic()
            try:
                result: dict = ask(url, case["question"], cookie)
            except Exception as error:  # one bad turn (429, timeout) shouldn't sink the run
                result = {"model": "", "response": "", "tools": [], "errors": [str(error)]}
            result |= {
                **case,
                "passed": passed(result["response"], case["must_contain"]),
                "latency_s": round(time.monotonic() - start, 1),
            }
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(
                f"[{i}/{len(cases)}] {'PASS' if result['passed'] else 'FAIL'} {case['drug']} tools={result['tools']}"
            )

    n = len(results) or 1
    metrics = {
        "url": url,
        "model": next((r["model"] for r in results if r["model"]), ""),
        "questions": len(results),
        "pass_rate": sum(r["passed"] for r in results) / n,
        "tool_call_rate": sum(bool(r["tools"]) for r in results) / n,
        "error_rate": sum(bool(r["errors"]) for r in results) / n,
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / n, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), f"\nWrote {out_dir}", sep="")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden benchmark against the served chat app.")
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--limit", type=int, default=None, help="first N golden questions")
    args = parser.parse_args()
    run(args.url, os.environ.get("EVAL_COOKIE", ""), args.limit)
