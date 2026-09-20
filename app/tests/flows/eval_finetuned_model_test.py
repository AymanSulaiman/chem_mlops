"""Tests for app/scripts/flows/eval/eval_finetuned_model.py."""

import json
import math
import re
import types
from pathlib import Path
from typing import Any
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
    m.returncode = 0  # a MagicMock here reads as a crash to _generate
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

    @pytest.fixture(autouse=True)
    def _stub_tool_results(self) -> Any:
        """The tool-result benchmark is not what these tests measure."""
        with patch(
            f"{_EVAL_PATCH}.run_tool_result_benchmark",
            return_value={"total": 4, "grounded_rate": 1.0, "prose_rate": 1.0, "results": []},
        ) as stub:
            yield stub

    def _tools(self, parse_rate: float = 1.0) -> Any:
        """Stub the tool-call benchmark — it is the gate, so it must be explicit."""
        return patch(
            f"{_EVAL_PATCH}.run_tool_call_benchmark",
            return_value={
                "total": 4,
                "parse_rate": parse_rate,
                "known_rate": parse_rate,
                "correct_rate": parse_rate,
                "results": [],
            },
        )

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
            self._tools(),
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
            self._tools(),
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
            self._tools(),
        ):
            with pytest.raises(RuntimeError, match="Perplexity regression"):
                eval_flow(run_dir, golden_path=golden, eval_output_dir=tmp_path / "eval")

    def test_low_golden_pass_rate_is_recorded_not_gated(self, tmp_path: Path) -> None:
        """Golden asks for facts held out of training — a lookup, not a fine-tune result.

        It was blocking every export for a capability the model is not supposed
        to have. Recorded so the number stays visible; the tool-call rate gates.
        """
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={
                    "pass_count": 5,
                    "total": 20,
                    "pass_rate": 0.25,  # far below the reported 0.70
                    "results": [],
                },
            ),
            self._tools(),
        ):
            metrics = eval_flow(
                run_dir,
                golden_path=golden,
                pass_threshold=0.70,
                eval_output_dir=tmp_path / "eval",
            )

        assert metrics["eval_gate_passed"] is True
        assert metrics["golden_pass_rate"] == 0.25
        assert metrics["golden_gated"] is False

    def test_raises_on_low_tool_call_parse_rate(self, tmp_path: Path) -> None:
        """A model that cannot emit a usable tool call cannot answer a lookup."""
        run_dir = self._make_run_dir(tmp_path)
        golden = self._make_golden(tmp_path)

        with (
            patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
            patch(
                f"{_EVAL_PATCH}.run_golden_benchmark",
                return_value={"pass_count": 1, "total": 1, "pass_rate": 1.0, "results": []},
            ),
            self._tools(parse_rate=0.1),
        ):
            with pytest.raises(RuntimeError, match="Tool-call parse rate"):
                eval_flow(run_dir, golden_path=golden, eval_output_dir=tmp_path / "eval")

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
            self._tools(),
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
            self._tools(),
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
            self._tools(),
        ):
            metrics = eval_flow(
                run_dir, golden_path=golden, pass_threshold=0.50, eval_output_dir=tmp_path / "eval"
            )

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


TRAIN_PATH = Path("data/llm_finetune/train.jsonl")


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="train.jsonl not built in this checkout")
def test_no_golden_molecule_appears_in_train() -> None:
    """Acceptance criterion for the eval holdout: golden.jsonl must be unseen data.

    Skipped where the dataset has not been built. Rebuild both together with
    `python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset`.
    """
    # Whole IDs only: a substring check reports CHEMBL413 as leaked because
    # CHEMBL413552 — a different molecule — appears in the training text.
    train_ids = set(re.findall(r"CHEMBL\d+", TRAIN_PATH.read_text()))
    items = [json.loads(ln) for ln in GOLDEN_BENCHMARK_PATH.read_text().splitlines() if ln.strip()]
    for item in items:
        chembl_id = item.get("chembl_id")
        assert chembl_id, (
            f"golden entry {item['question']!r} has no chembl_id — rebuild golden.jsonl"
        )
        assert chembl_id not in train_ids, (
            f"{chembl_id} ({item['question']!r}) leaked into train.jsonl"
        )


# ── Tool-call benchmark ───────────────────────────────────────────────────────


class TestToolCallBenchmark:
    @staticmethod
    def _golden(tmp_path: Path, drugs: list[str]) -> Path:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "question": f"What does {d} target?",
                        "must_contain": ["x"],
                        "category": "mechanism_of_action",
                        "chembl_id": f"CHEMBL{i}",
                        "drug": d,
                    }
                )
                for i, d in enumerate(drugs)
            )
            + "\n"
        )
        return path

    def test_first_json_object_digs_the_call_out_of_prose(self) -> None:
        from app.scripts.flows.eval.eval_finetuned_model import _first_json_object

        assert _first_json_object(
            'Sure! {"tool": "draw_molecule", "args": {"name": "X"}} done'
        ) == {
            "tool": "draw_molecule",
            "args": {"name": "X"},
        }
        assert _first_json_object('```json\n{"tool": "a"}\n```') == {"tool": "a"}
        assert _first_json_object("no json here") is None
        assert _first_json_object('{"unterminated": ') is None

    def test_holdout_drugs_reads_new_and_old_golden_files(self, tmp_path: Path) -> None:
        from app.scripts.flows.eval.eval_finetuned_model import _holdout_drugs

        assert _holdout_drugs(self._golden(tmp_path, ["SIROLIMUS", "BRECANAVIR"]), 10) == [
            "SIROLIMUS",
            "BRECANAVIR",
        ]

        # A golden.jsonl written before the "drug" field existed.
        old = tmp_path / "old.jsonl"
        old.write_text(json.dumps({"question": "What does ASPIRIN target?"}) + "\n")
        assert _holdout_drugs(old, 10) == ["ASPIRIN"]

    def test_rates_separate_parsing_naming_and_routing(self, tmp_path: Path) -> None:
        """The three rates fail independently — that is the point of having three."""
        from app.scripts.flows.eval.eval_finetuned_model import TOOL_CASES, run_tool_call_benchmark

        replies = [
            '{"tool": "get_compound_by_name", "args": {"name": "SIROLIMUS"}}',  # correct
            '{"tool": "draw_it_now", "args": {"name": "SIROLIMUS"}}',  # parses, unknown tool
            '{"tool": "get_compound_by_name", "args": {"name": "SIROLIMUS"}}',  # known, wrong tool
            "Sirolimus interacts with many drugs.",  # no JSON at all
        ]
        assert len(replies) == len(TOOL_CASES)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                types.SimpleNamespace(stdout=r, stderr="", returncode=0) for r in replies
            ]
            report = run_tool_call_benchmark(
                Path("mlx"), Path("adapter"), self._golden(tmp_path, ["SIROLIMUS"]), drugs=1
            )

        assert report["total"] == 4
        assert report["parse_rate"] == 0.75  # three replies contained JSON
        assert report["known_rate"] == 0.5  # one named a tool that does not exist
        assert report["correct_rate"] == 0.25  # only the first was the right tool
        assert report["results"][1]["called_tool"] == "draw_it_now"

    def test_prompt_advertises_every_tool_except_the_untrained_one(self) -> None:
        """Guard against drift from web/src/tools.ts and the tool registry.

        query_compounds is callable but not advertised: it has no training
        records, and listing it made the model copy its example SMILES verbatim
        for unrelated questions. Keep it out of the prompt until it is trained.
        """
        from app.scripts.flows.eval.eval_finetuned_model import TOOL_SYSTEM_PROMPT
        from app.scripts.flows.vector_store.tools import TOOLS

        advertised = {
            line.split()[1] for line in TOOL_SYSTEM_PROMPT.splitlines() if line.startswith("- ")
        }
        assert advertised == set(TOOLS) - {"query_compounds"}
        assert "query_compounds" in TOOLS  # still dispatchable


def test_metrics_name_what_was_gated(tmp_path: Path) -> None:
    """eval_gate_passed must not read as "every number is good".

    Golden sits at 5% and is not gated, so a bare pass flag with nothing naming
    the gates would misrepresent the run to anyone skimming.
    """
    run_dir = tmp_path / "20260615_120000"
    (run_dir / DEFAULT_MLX_SUBDIR).mkdir(parents=True)
    (run_dir / DEFAULT_ADAPTER_SUBDIR).mkdir(parents=True)
    golden = tmp_path / "golden.jsonl"
    _write_golden(golden, [{"question": "Q?", "must_contain": ["x"], "category": "test"}])

    with (
        patch(f"{_EVAL_PATCH}.run_perplexity_eval", side_effect=[6.0, 4.0]),
        patch(
            f"{_EVAL_PATCH}.run_golden_benchmark",
            return_value={"pass_count": 1, "total": 20, "pass_rate": 0.05, "results": []},
        ),
        patch(
            f"{_EVAL_PATCH}.run_tool_call_benchmark",
            return_value={
                "total": 4,
                "parse_rate": 0.9,
                "known_rate": 0.9,
                "correct_rate": 0.5,
                "results": [],
            },
        ),
        patch(
            f"{_EVAL_PATCH}.run_tool_result_benchmark",
            return_value={"total": 4, "grounded_rate": 0.5, "prose_rate": 0.5, "results": []},
        ),
    ):
        metrics = eval_flow(run_dir, golden_path=golden, eval_output_dir=tmp_path / "eval")

    assert metrics["eval_gate_passed"] is True
    assert metrics["gates_applied"] == ["perplexity", "tool_call_parse_rate"]
    # The weak numbers stay in the file and stay labelled as ungated.
    assert "golden_pass_rate" in metrics["ungated_metrics"]
    assert metrics["golden_pass_rate"] == 0.05


class TestToolResultBenchmark:
    """The other half of the loop: can the model read a result it is handed?"""

    def _golden(self, tmp_path: Path, drugs: list[str]) -> Path:
        golden = tmp_path / "golden.jsonl"
        _write_golden(
            golden,
            [{"question": f"What does {d} target?", "must_contain": ["x"], "drug": d}
             for d in drugs],
        )
        return golden

    def test_scores_grounding_and_prose_separately(self, tmp_path: Path) -> None:
        from app.scripts.flows.eval.eval_finetuned_model import run_tool_result_benchmark

        replies = [
            "SIROLIMUS has a molecular weight of 481.27.",  # grounded, prose
            '{"tool": "draw_molecule", "args": {"name": "SIROLIMUS"}}',  # echoed the shape
            "It interacts with several drugs.",  # prose, but not grounded
            "The pair shows photosensitivity at 7.43.",  # grounded, prose
        ]
        with patch(f"{_EVAL_PATCH}._generate", side_effect=replies):
            report = run_tool_result_benchmark(
                Path("mlx"), Path("adapter"), self._golden(tmp_path, ["SIROLIMUS"]), drugs=1
            )

        assert report["total"] == 4
        assert report["grounded_rate"] == 0.5  # first and last quoted the result
        assert report["prose_rate"] == 0.75  # only the second echoed JSON

    def test_expected_values_cannot_be_recalled_from_training(self) -> None:
        """The whole design rests on this: a memorised answer must not score.

        If an expected value were a real property of a real drug, a model
        answering from weights would pass without reading the tool result.
        """
        from app.scripts.flows.eval.eval_finetuned_model import TOOL_RESULT_CASES

        expected = [case[-1] for case in TOOL_RESULT_CASES]
        assert len(set(expected)) == len(expected), "each case needs its own marker"
        # Every marker must actually appear in that case's fabricated result.
        for *_, result_shape, marker in TOOL_RESULT_CASES:
            assert marker in json.dumps(result_shape), f"{marker} is not in the tool result"

    def test_prompt_matches_the_training_record_layout(self, tmp_path: Path) -> None:
        """Train-serve mismatch is the failure this item is about."""
        from app.scripts.flows.eval.eval_finetuned_model import run_tool_result_benchmark

        seen: list[str] = []

        def capture(_model: Path, _adapter: Path, prompt: str, _max_tokens: int) -> str:
            seen.append(prompt)
            return "ok"

        with patch(f"{_EVAL_PATCH}._generate", side_effect=capture):
            run_tool_result_benchmark(
                Path("mlx"), Path("adapter"), self._golden(tmp_path, ["SIROLIMUS"]), drugs=1
            )

        prompt = seen[0]
        # The result arrives inside a ### Question block, exactly as
        # _tool_call_record lays it out and as the server sends it.
        assert "### Question\n### Tool result (get_compound_by_name)\n" in prompt
        assert prompt.endswith("### Answer\n")
        assert prompt.count("### Answer") == 2  # the model's call, then its answer


def test_generate_raises_instead_of_scoring_a_crashed_subprocess() -> None:
    """A missing completion is a broken run, not a wrong answer.

    Scoring it as wrong manufactures findings: two concurrent evals once
    produced 40 empty completions, reported as "0.0% routing".
    """
    from app.scripts.flows.eval.eval_finetuned_model import _generate

    crashed = types.SimpleNamespace(returncode=1, stdout="", stderr="Metal out of memory")
    with patch(f"{_EVAL_PATCH}.subprocess.run", return_value=crashed):
        with pytest.raises(RuntimeError, match="mlx_lm generate failed"):
            _generate(Path("m"), Path("a"), "prompt", 10)

    # Exit 0 but nothing on stdout is equally broken.
    empty = types.SimpleNamespace(returncode=0, stdout="   \n", stderr="")
    with patch(f"{_EVAL_PATCH}.subprocess.run", return_value=empty):
        with pytest.raises(RuntimeError, match="mlx_lm generate failed"):
            _generate(Path("m"), Path("a"), "prompt", 10)
