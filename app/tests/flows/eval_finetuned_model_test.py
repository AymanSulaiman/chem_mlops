"""Tests for app/scripts/flows/eval/eval_finetuned_model.py."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.scripts.flows.eval.eval_finetuned_model import (
    GOLDEN_BENCHMARK_PATH,
    eval_flow,
    run_golden_benchmark,
    run_perplexity_eval,
)
from app.scripts.flows.finetuning.export_to_ollama import (
    DEFAULT_ADAPTER_SUBDIR,
    DEFAULT_MLX_SUBDIR,
)

_EVAL_MODULE = "app.scripts.flows.eval.eval_finetuned_model"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _gen_output(text: str) -> MagicMock:
    """Fake subprocess.run result that looks like mlx_lm generate output."""
    m = MagicMock()
    m.stdout = text
    m.stderr = ""
    return m


def _write_golden(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")


def _mock_ppl_context(loss: float):
    """Context manager that patches the mlx_lm Python API for perplexity eval."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_valid_set = MagicMock()
    mock_valid_set.__len__ = MagicMock(return_value=100)

    load_patch = patch(f"{_EVAL_MODULE}.mlx_lm_load", return_value=(mock_model, mock_tokenizer))
    dataset_patch = patch(
        f"{_EVAL_MODULE}.mlx_load_dataset", return_value=(None, mock_valid_set, None)
    )
    evaluate_patch = patch(f"{_EVAL_MODULE}.mlx_evaluate", return_value=loss)
    cache_patch = patch(f"{_EVAL_MODULE}.CacheDataset", side_effect=lambda ds: ds)
    return load_patch, dataset_patch, evaluate_patch, cache_patch


# ── run_perplexity_eval ───────────────────────────────────────────────────────


class TestRunPerplexityEval:
    def test_returns_exp_of_loss(self, tmp_path: Path) -> None:
        loss = 1.5
        load_p, ds_p, eval_p, cache_p = _mock_ppl_context(loss)
        with load_p, ds_p, eval_p, cache_p:
            ppl = run_perplexity_eval(tmp_path / "model", tmp_path / "data")
        assert ppl == pytest.approx(math.exp(loss))

    def test_loads_without_adapter_for_baseline(self, tmp_path: Path) -> None:
        load_p, ds_p, eval_p, cache_p = _mock_ppl_context(1.0)
        with load_p as mock_load, ds_p, eval_p, cache_p:
            run_perplexity_eval(tmp_path / "model", tmp_path / "data")
        assert mock_load.call_args.kwargs.get("adapter_path") is None

    def test_loads_with_adapter_when_given(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        load_p, ds_p, eval_p, cache_p = _mock_ppl_context(1.0)
        with load_p as mock_load, ds_p, eval_p, cache_p:
            run_perplexity_eval(tmp_path / "model", tmp_path / "data", adapter_dir=adapter)
        assert mock_load.call_args.kwargs.get("adapter_path") == str(adapter)

    def test_num_batches_forwarded_to_evaluate(self, tmp_path: Path) -> None:
        load_p, ds_p, eval_p, cache_p = _mock_ppl_context(1.0)
        with load_p, ds_p, eval_p as mock_eval, cache_p:
            run_perplexity_eval(tmp_path / "model", tmp_path / "data", num_batches=25)
        assert mock_eval.call_args.kwargs.get("num_batches") == 25

    def test_returns_float(self, tmp_path: Path) -> None:
        load_p, ds_p, eval_p, cache_p = _mock_ppl_context(2.0)
        with load_p, ds_p, eval_p, cache_p:
            result = run_perplexity_eval(tmp_path / "model", tmp_path / "data")
        assert isinstance(result, float)


# ── run_golden_benchmark ──────────────────────────────────────────────────────


class TestRunGoldenBenchmark:
    def test_all_questions_pass(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "golden.jsonl"
        _write_golden(
            golden_path,
            [
                {"question": "What does Aspirin target?", "must_contain": ["cyclooxygenase"]},
                {"question": "What enzyme metabolises Warfarin?", "must_contain": ["cyp2c9"]},
            ],
        )
        responses = [
            _gen_output("Aspirin inhibits Cyclooxygenase (COX-1 and COX-2)."),
            _gen_output("Warfarin is metabolised by CYP2C9."),
        ]
        with patch("subprocess.run", side_effect=responses):
            result = run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        assert result["pass_count"] == 2
        assert result["total"] == 2
        assert result["pass_rate"] == pytest.approx(1.0)

    def test_some_questions_fail(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "golden.jsonl"
        _write_golden(
            golden_path,
            [
                {"question": "What does Aspirin target?", "must_contain": ["cyclooxygenase"]},
                {"question": "What enzyme metabolises Warfarin?", "must_contain": ["cyp2c9"]},
            ],
        )
        responses = [
            _gen_output("Aspirin is an NSAID."),  # missing keyword
            _gen_output("Warfarin is metabolised by CYP2C9."),
        ]
        with patch("subprocess.run", side_effect=responses):
            result = run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        assert result["pass_count"] == 1
        assert result["pass_rate"] == pytest.approx(0.5)

    def test_keyword_check_is_case_insensitive(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "golden.jsonl"
        _write_golden(
            golden_path,
            [{"question": "Q?", "must_contain": ["CYP2C9"]}],
        )
        with patch("subprocess.run", return_value=_gen_output("metabolised by cyp2c9.")):
            result = run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        assert result["pass_count"] == 1

    def test_all_must_contain_keywords_required(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "golden.jsonl"
        _write_golden(
            golden_path,
            [{"question": "Q?", "must_contain": ["cyp2c9", "bleeding"]}],
        )
        # only one keyword present
        with patch("subprocess.run", return_value=_gen_output("CYP2C9 substrate.")):
            result = run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        assert result["pass_count"] == 0

    def test_results_list_length_matches_questions(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "golden.jsonl"
        items = [{"question": f"Q{i}?", "must_contain": ["x"]} for i in range(5)]
        _write_golden(golden_path, items)

        with patch("subprocess.run", return_value=_gen_output("x")):
            result = run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        assert len(result["results"]) == 5

    def test_uses_ignore_chat_template_prompt_format(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "golden.jsonl"
        _write_golden(golden_path, [{"question": "What targets Aspirin?", "must_contain": ["x"]}])

        with patch("subprocess.run", return_value=_gen_output("x")) as mock_run:
            run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        cmd = mock_run.call_args.args[0]
        prompt_idx = cmd.index("--prompt") + 1
        assert "### Question" in cmd[prompt_idx]
        assert "### Answer" in cmd[prompt_idx]

    def test_empty_golden_file_returns_zero_pass_rate(self, tmp_path: Path) -> None:
        golden_path = tmp_path / "empty.jsonl"
        golden_path.write_text("")
        result = run_golden_benchmark(tmp_path / "model", tmp_path / "adapter", golden_path)

        assert result["total"] == 0
        assert result["pass_rate"] == pytest.approx(0.0)


# ── eval_flow ─────────────────────────────────────────────────────────────────


_EVAL_PATCH = "app.scripts.flows.eval.eval_finetuned_model"


class TestEvalFlow:
    def _make_run_dir(self, tmp_path: Path) -> Path:
        run_dir = tmp_path / "20260615_120000"
        (run_dir / DEFAULT_MLX_SUBDIR).mkdir(parents=True)
        (run_dir / DEFAULT_ADAPTER_SUBDIR).mkdir(parents=True)
        return run_dir

    def _make_golden(self, tmp_path: Path) -> Path:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [{"question": "Q?", "must_contain": ["x"], "category": "test"}])
        return golden

    def test_writes_metrics_json_on_success(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)
        eval_out = tmp_path / "eval"

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={"pass_count": 1, "total": 1, "pass_rate": 1.0, "results": []},
            ),
        ):
            eval_flow(run_dir, golden_path=golden, eval_output_dir=eval_out)

        metrics_path = eval_out / "finetuned_eval_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["eval_gate_passed"] is True
        assert metrics["run"] == "20260615_120000"

    def test_metrics_contain_perplexity_values(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.5, 3.2]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={"pass_count": 1, "total": 1, "pass_rate": 1.0, "results": []},
            ),
        ):
            metrics = eval_flow(run_dir, golden_path=golden, eval_output_dir=tmp_path / "eval")

        assert metrics["baseline_perplexity"] == pytest.approx(6.5, abs=0.01)
        assert metrics["finetuned_perplexity"] == pytest.approx(3.2, abs=0.01)

    def test_raises_on_perplexity_regression(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[4.0, 6.0]),  # regression
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={"pass_count": 1, "total": 1, "pass_rate": 1.0, "results": []},
            ),
        ):
            with pytest.raises(RuntimeError, match="Perplexity regression"):
                eval_flow(run_dir, golden_path=golden, eval_output_dir=tmp_path / "eval")

    def test_raises_on_low_golden_pass_rate(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={
                    "pass_count": 5,
                    "total": 20,
                    "pass_rate": 0.25,  # below 0.70 threshold
                    "results": [],
                },
            ),
        ):
            with pytest.raises(RuntimeError, match="below threshold"):
                eval_flow(run_dir, golden_path=golden, pass_threshold=0.70,
                          eval_output_dir=tmp_path / "eval")

    def test_metrics_json_written_even_on_failure(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)
        eval_out = tmp_path / "eval"

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[4.0, 6.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={"pass_count": 0, "total": 1, "pass_rate": 0.0, "results": []},
            ),
        ):
            with pytest.raises(RuntimeError):
                eval_flow(run_dir, golden_path=golden, eval_output_dir=eval_out)

        metrics_path = eval_out / "finetuned_eval_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["eval_gate_passed"] is False

    def test_golden_results_jsonl_written(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)
        eval_out = tmp_path / "eval"
        fake_results = [{"question": "Q?", "passed": True, "response": "x", "must_contain": ["x"]}]

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={
                    "pass_count": 1,
                    "total": 1,
                    "pass_rate": 1.0,
                    "results": fake_results,
                },
            ),
        ):
            eval_flow(run_dir, golden_path=golden, eval_output_dir=eval_out)

        detail_path = eval_out / "finetuned_golden_results.jsonl"
        assert detail_path.exists()
        rows = [json.loads(line) for line in detail_path.read_text().splitlines() if line.strip()]
        assert rows == fake_results

    def test_custom_pass_threshold_respected(self, tmp_path: Path) -> None:
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)

        # pass_rate=0.60 should pass with threshold=0.50 but fail with default 0.70
        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={"pass_count": 6, "total": 10, "pass_rate": 0.60, "results": []},
            ),
        ):
            metrics = eval_flow(run_dir, golden_path=golden, pass_threshold=0.50,
                                eval_output_dir=tmp_path / "eval")

        assert metrics["eval_gate_passed"] is True


# ── golden.jsonl integrity ────────────────────────────────────────────────────


class TestGoldenJsonl:
    def test_golden_file_is_valid_jsonl(self) -> None:
        assert GOLDEN_BENCHMARK_PATH.exists(), f"{GOLDEN_BENCHMARK_PATH} not found"
        for i, line in enumerate(GOLDEN_BENCHMARK_PATH.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Invalid JSON on line {i}: {exc}")  # type: ignore

    def test_golden_file_has_required_keys(self) -> None:
        for i, line in enumerate(GOLDEN_BENCHMARK_PATH.read_text().splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            assert "question" in item, f"Line {i} missing 'question'"
            assert "must_contain" in item, f"Line {i} missing 'must_contain'"
            assert isinstance(item["must_contain"], list), f"Line {i} 'must_contain' must be a list"
            assert len(item["must_contain"]) > 0, f"Line {i} 'must_contain' is empty"

    def test_golden_file_has_at_least_20_questions(self) -> None:
        lines = [ln for ln in GOLDEN_BENCHMARK_PATH.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 20
