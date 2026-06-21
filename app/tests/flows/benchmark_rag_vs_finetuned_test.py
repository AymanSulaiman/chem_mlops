"""Tests for app/scripts/flows/eval/benchmark_rag_vs_finetuned.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.scripts.flows.eval.benchmark_rag_vs_finetuned import (
    RAG_PASS_THRESHOLD,
    build_rag_context,
    call_ollama,
    check_rag_quality,
    extract_drug_candidates,
    run_benchmark,
    score_response,
    write_benchmark_artifacts,
)

_MOD = "app.scripts.flows.eval.benchmark_rag_vs_finetuned"


# ── extract_drug_candidates ───────────────────────────────────────────────────


class TestExtractDrugCandidates:
    def test_picks_capitalised_words(self) -> None:
        result = extract_drug_candidates("What does Aspirin target?")
        assert "Aspirin" in result

    def test_excludes_stopwords(self) -> None:
        result = extract_drug_candidates("What does Warfarin do?")
        assert "What" not in result
        assert "Warfarin" in result

    def test_deduplicates(self) -> None:
        result = extract_drug_candidates("Aspirin and Aspirin")
        assert result.count("Aspirin") == 1

    def test_title_cases_result(self) -> None:
        result = extract_drug_candidates("ASPIRIN inhibits COX")
        assert "Aspirin" in result

    def test_empty_string_returns_empty_list(self) -> None:
        assert extract_drug_candidates("") == []

    def test_multiple_drugs_preserved_in_order(self) -> None:
        result = extract_drug_candidates("Warfarin and Aspirin interaction")
        assert result.index("Warfarin") < result.index("Aspirin")

    def test_short_words_excluded(self) -> None:
        # 2-char words (D+o, I+t) don't satisfy [a-zA-Z]{2,} after the capital
        result = extract_drug_candidates("Do It something")
        assert "Do" not in result
        assert "It" not in result


# ── score_response ────────────────────────────────────────────────────────────


class TestScoreResponse:
    def test_returns_true_when_all_keywords_present(self) -> None:
        assert score_response("CYP2C9 metabolises Warfarin", ["cyp2c9", "warfarin"])

    def test_returns_false_when_keyword_missing(self) -> None:
        assert not score_response("Warfarin is anticoagulant", ["cyp2c9"])

    def test_case_insensitive(self) -> None:
        assert score_response("CYP2C9 substrate", ["cyp2c9"])
        assert score_response("cyp2c9 substrate", ["CYP2C9"])

    def test_empty_must_contain_always_passes(self) -> None:
        assert score_response("anything", [])

    def test_all_keywords_must_match(self) -> None:
        assert not score_response("cyp2c9 only", ["cyp2c9", "bleeding"])


# ── build_rag_context ─────────────────────────────────────────────────────────


class TestBuildRagContext:
    def test_returns_none_when_no_drug_candidates(self) -> None:
        result = build_rag_context("what is the weather today?", lancedb_dir="/nonexistent")
        assert result is None

    def test_returns_none_when_lancedb_missing(self) -> None:
        # All LanceDB calls will raise FileNotFoundError — context should be None.
        result = build_rag_context("What does Aspirin target?", lancedb_dir="/nonexistent")
        assert result is None

    def test_returns_string_when_compound_found(self) -> None:
        fake_row = {
            "pref_name": "Aspirin", "chembl_id": "CHEMBL25",
            "indications": "pain; fever", "mechanisms": "COX inhibitor",
            "metabolic_enzymes": "CYP2C9", "warning_types": "bleeding",
        }
        with (
            patch(f"{_MOD}.get_compound_by_name", return_value=fake_row),
            patch(f"{_MOD}.query_drug_side_effects", return_value=[]),
            patch(f"{_MOD}.query_polypharmacy", return_value=None),
        ):
            result = build_rag_context("What does Aspirin target?", lancedb_dir="/fake")

        assert result is not None
        assert "Aspirin" in result
        assert "CHEMBL25" in result

    def test_includes_polypharmacy_signals(self) -> None:
        fake_pair = {
            "drug_1_name": "Warfarin", "drug_2_name": "Aspirin",
            "side_effects": "bleeding; bruising", "max_prr": 5.2,
        }
        with (
            patch(f"{_MOD}.get_compound_by_name", return_value=None),
            patch(f"{_MOD}.query_drug_side_effects", return_value=[fake_pair]),
            patch(f"{_MOD}.query_polypharmacy", return_value=None),
        ):
            result = build_rag_context("What is the DDI for Warfarin and Aspirin?",
                                       lancedb_dir="/fake")

        assert result is not None
        assert "Warfarin" in result
        assert "Aspirin" in result

    def test_context_has_instruction_prefix(self) -> None:
        fake_row = {"pref_name": "Aspirin", "chembl_id": "CHEMBL25"}
        with (
            patch(f"{_MOD}.get_compound_by_name", return_value=fake_row),
            patch(f"{_MOD}.query_drug_side_effects", return_value=[]),
            patch(f"{_MOD}.query_polypharmacy", return_value=None),
        ):
            result = build_rag_context("What does Aspirin target?", lancedb_dir="/fake")

        assert result is not None
        assert result.startswith("Use the following")

    def test_deduplicates_polypharmacy_pairs(self) -> None:
        fake_pair = {
            "drug_1_name": "Warfarin", "drug_2_name": "Aspirin",
            "side_effects": "bleeding", "max_prr": 5.0,
        }
        with (
            patch(f"{_MOD}.get_compound_by_name", return_value=None),
            patch(f"{_MOD}.query_drug_side_effects", return_value=[fake_pair]),
            patch(f"{_MOD}.query_polypharmacy", return_value=fake_pair),
        ):
            result = build_rag_context("Warfarin Aspirin interaction?", lancedb_dir="/fake")

        assert result is not None
        # Pair should appear only once
        assert result.count("Warfarin + Aspirin") + result.count("Aspirin + Warfarin") == 1


# ── call_ollama ───────────────────────────────────────────────────────────────


class TestCallOllama:
    def test_returns_content_string(self) -> None:
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "CYP2C9 metabolises Warfarin."}}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp):
            result = call_ollama([{"role": "user", "content": "Q?"}], "gemma3:1b")

        assert result == "CYP2C9 metabolises Warfarin."

    def test_posts_to_correct_endpoint(self) -> None:
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "ok"}}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            call_ollama([], "gemma3:1b", base_url="http://localhost:11434")

        assert mock_post.call_args.args[0] == "http://localhost:11434/api/chat"

    def test_sends_stream_false(self) -> None:
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"message": {"content": "ok"}}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            call_ollama([], "gemma3:1b")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is False


# ── run_benchmark ─────────────────────────────────────────────────────────────


def _write_golden(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")


class TestRunBenchmark:
    def _patch_ollama(self, ft_reply: str, rag_reply: str):
        replies = [ft_reply, rag_reply]
        call_count = {"n": 0}

        def fake_call(messages, model, base_url=None, timeout=None):
            r = replies[call_count["n"] % len(replies)]
            call_count["n"] += 1
            return r

        return patch(f"{_MOD}.call_ollama", side_effect=fake_call)

    def test_returns_correct_structure(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [{"question": "Q?", "must_contain": ["x"], "category": "test"}])

        with (
            self._patch_ollama("x answer", "x answer"),
            patch(f"{_MOD}.build_rag_context", return_value=None),
        ):
            results = run_benchmark(golden_path=golden, lancedb_dir="/fake")

        assert "finetuned" in results
        assert "rag" in results
        assert results["total"] == 1

    def test_fine_tuned_pass_rate_calculated(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [
            {"question": "Q1?", "must_contain": ["cyp2c9"], "category": "metabolism"},
            {"question": "Q2?", "must_contain": ["bleeding"], "category": "warning"},
        ])

        call_count = {"n": 0}
        # interleaved: ft_Q1, rag_Q1, ft_Q2, rag_Q2
        replies = ["cyp2c9 present", "no keyword", "bleeding here", "no keyword"]

        def fake_call(messages, model, *_a, **_kw):
            r = replies[call_count["n"]]
            call_count["n"] += 1
            return r

        with (
            patch(f"{_MOD}.call_ollama", side_effect=fake_call),
            patch(f"{_MOD}.build_rag_context", return_value=None),
        ):
            results = run_benchmark(golden_path=golden, lancedb_dir="/fake")

        assert results["finetuned"]["pass_count"] == 2  # cyp2c9 present / bleeding here
        assert results["rag"]["pass_count"] == 0        # both "no keyword"

    def test_delta_pass_rate_sign(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [{"question": "Q?", "must_contain": ["cyp2c9"], "category": "c"}])

        # fine-tuned passes, RAG fails
        replies = ["cyp2c9 yes", "no keyword"]
        call_count = {"n": 0}

        def fake_call(messages, model, *_a, **_kw):
            r = replies[call_count["n"]]
            call_count["n"] += 1
            return r

        with (
            patch(f"{_MOD}.call_ollama", side_effect=fake_call),
            patch(f"{_MOD}.build_rag_context", return_value=None),
        ):
            results = run_benchmark(golden_path=golden, lancedb_dir="/fake")

        assert results["delta_pass_rate"] < 0  # RAG worse than fine-tuned

    def test_rag_context_prepended_as_system_message(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [{"question": "Q?", "must_contain": ["x"], "category": "c"}])

        captured: list[list[dict]] = []

        def fake_call(messages, model, *_a, **_kw):
            captured.append(messages)
            return "x"

        with (
            patch(f"{_MOD}.call_ollama", side_effect=fake_call),
            patch(f"{_MOD}.build_rag_context", return_value="fake context"),
        ):
            run_benchmark(golden_path=golden, lancedb_dir="/fake")

        # Second call is the RAG call — should have a system message first
        rag_messages = captured[1]
        assert rag_messages[0]["role"] == "system"
        assert "fake context" in rag_messages[0]["content"]

    def test_error_in_ollama_recorded_not_raised(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [{"question": "Q?", "must_contain": ["x"], "category": "c"}])

        with (
            patch(f"{_MOD}.call_ollama", side_effect=Exception("connection refused")),
            patch(f"{_MOD}.build_rag_context", return_value=None),
        ):
            results = run_benchmark(golden_path=golden, lancedb_dir="/fake")

        assert results["finetuned"]["pass_count"] == 0
        assert "[ERROR:" in results["finetuned"]["results"][0]["response"]

    def test_results_use_keyword_match_passed_field(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden, [{"question": "Q?", "must_contain": ["x"], "category": "c"}])

        with (
            self._patch_ollama("x answer", "x answer"),
            patch(f"{_MOD}.build_rag_context", return_value=None),
        ):
            results = run_benchmark(golden_path=golden, lancedb_dir="/fake")

        assert "keyword_match_passed" in results["finetuned"]["results"][0]
        assert "keyword_match_passed" in results["rag"]["results"][0]
        assert "passed" not in results["finetuned"]["results"][0]


# ── write_benchmark_artifacts ─────────────────────────────────────────────────


class TestWriteBenchmarkArtifacts:
    def _results(self, ft: str = "chembl-drug-chat:1b", rag: str = "gemma3:1b") -> dict:
        return {"finetuned_model": ft, "rag_model": rag, "total": 1}

    def test_filename_contains_model_slugs(self, tmp_path: Path) -> None:
        out_path = write_benchmark_artifacts(self._results(), out_dir=tmp_path)
        assert out_path.name == "chembl-drug-chat_1b_vs_gemma3_1b_benchmark.json"

    def test_filename_sanitises_colon(self, tmp_path: Path) -> None:
        out_path = write_benchmark_artifacts(
            self._results(ft="my-model:v2", rag="base:3b"), out_dir=tmp_path
        )
        assert ":" not in out_path.name
        assert "my-model_v2_vs_base_3b_benchmark.json" == out_path.name

    def test_writes_to_given_directory(self, tmp_path: Path) -> None:
        out_path = write_benchmark_artifacts(self._results(), out_dir=tmp_path)
        assert out_path.exists()
        assert out_path.parent == tmp_path

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        results = {**self._results(), "total": 5, "finetuned": {"pass_count": 4}, "rag": {"pass_count": 3}}
        out_path = write_benchmark_artifacts(results, out_dir=tmp_path)
        loaded = json.loads(out_path.read_text())
        assert loaded["total"] == 5

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "eval_run"
        out_path = write_benchmark_artifacts(self._results(), out_dir=out_dir)
        assert out_path.exists()


# ── check_rag_quality ─────────────────────────────────────────────────────────


def _make_results(rag_pass_count: int, total: int) -> dict:
    return {
        "rag": {
            "pass_count": rag_pass_count,
            "pass_rate": rag_pass_count / total if total > 0 else 0.0,
        }
    }


class TestCheckRagQuality:
    def test_passes_when_above_threshold(self) -> None:
        check_rag_quality(_make_results(12, 20), threshold=0.5)  # 60% ≥ 50% — should not raise

    def test_passes_exactly_at_threshold(self) -> None:
        check_rag_quality(_make_results(10, 20), threshold=0.5)  # 50% == 50% — should not raise

    def test_raises_when_below_threshold(self) -> None:
        with pytest.raises(RuntimeError, match="below threshold"):
            check_rag_quality(_make_results(5, 20), threshold=0.5)  # 25% < 50%

    def test_error_message_contains_pass_rate(self) -> None:
        with pytest.raises(RuntimeError, match="25.0%"):
            check_rag_quality(_make_results(5, 20), threshold=0.5)

    def test_uses_module_default_threshold(self) -> None:
        # Passing 0 / 20 should always fail with the default threshold (0.5)
        with pytest.raises(RuntimeError):
            check_rag_quality(_make_results(0, 20))

    def test_default_threshold_is_rag_pass_threshold_constant(self) -> None:
        # Ensures the gate and the constant stay in sync
        results_at_threshold = _make_results(int(RAG_PASS_THRESHOLD * 20), 20)
        check_rag_quality(results_at_threshold)  # exactly at threshold — must not raise
