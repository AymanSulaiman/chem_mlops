"""Tests for app/scripts/flows/finetuning/finetuning.py."""

import json
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

from app.scripts.flows.finetuning.finetuning import (
    HF_MODEL_ID,
    convert_to_mlx,
    finetune_lora,
    gemma3_chembl_toon_finetune_flow,
    split_long_sequences,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_tokenizer(token_length_per_char: int = 1) -> MagicMock:
    """
    Mock tokenizer where encode returns one token per character,
    and decode returns a string of 'x' characters of the same length.
    """
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **_: list(range(len(text) * token_length_per_char))
    tok.decode.side_effect = lambda ids, **_: "x" * len(ids)
    return tok


# ── split_long_sequences ──────────────────────────────────────────────────────


class TestSplitLongSequences:
    def test_short_sequences_pass_through_unchanged(self, tmp_path: Path) -> None:
        records = [{"text": "short"}]  # 5 chars < max_seq_len=10
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=10)

        result = _read_jsonl(tmp_path / "train.jsonl")
        assert result == records

    def test_long_sequence_is_split_into_chunks(self, tmp_path: Path) -> None:
        # 12 chars → 12 tokens with our mock; max_seq_len=5 → 3 chunks
        records = [{"text": "a" * 12}]
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        result = _read_jsonl(tmp_path / "train.jsonl")
        assert len(result) == 3  # ceil(12 / 5) = 3 chunks

    def test_chunk_lengths_respect_max_seq_len(self, tmp_path: Path) -> None:
        records = [{"text": "a" * 12}]
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        result = _read_jsonl(tmp_path / "train.jsonl")
        # decode returns "x" * len(ids), so text length == chunk token count
        assert all(len(r["text"]) <= 5 for r in result)

    def test_extra_fields_preserved_across_chunks(self, tmp_path: Path) -> None:
        records = [{"text": "a" * 12, "source": "chembl", "category": "moa"}]
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        result = _read_jsonl(tmp_path / "train.jsonl")
        for chunk in result:
            assert chunk["source"] == "chembl"
            assert chunk["category"] == "moa"

    def test_empty_text_passes_through(self, tmp_path: Path) -> None:
        records = [{"text": ""}]
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        result = _read_jsonl(tmp_path / "train.jsonl")
        assert result == records

    def test_record_without_text_key_passes_through(self, tmp_path: Path) -> None:
        records = [{"other_field": "value"}]
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        result = _read_jsonl(tmp_path / "train.jsonl")
        assert result == records

    def test_missing_split_file_is_skipped(self, tmp_path: Path) -> None:
        # Only train.jsonl exists; valid.jsonl and test.jsonl are absent
        _write_jsonl(tmp_path / "train.jsonl", [{"text": "hi"}])

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=100)

        # Should not raise, and only train.jsonl should exist
        assert (tmp_path / "train.jsonl").exists()
        assert not (tmp_path / "valid.jsonl").exists()

    def test_processes_all_three_splits(self, tmp_path: Path) -> None:
        for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            _write_jsonl(tmp_path / split, [{"text": "a" * 12}])

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            result = _read_jsonl(tmp_path / split)
            assert len(result) == 3

    def test_file_not_rewritten_when_no_long_sequences(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        records = [{"text": "hi"}]
        _write_jsonl(path, records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), patch(
            "pathlib.Path.write_text"
        ) as mock_write_text:
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=100)

        mock_write_text.assert_not_called()

    def test_mixed_short_and_long_records(self, tmp_path: Path) -> None:
        records = [
            {"text": "short"},  # 5 tokens, passes through
            {"text": "a" * 12},  # 12 tokens, split into 3
        ]
        _write_jsonl(tmp_path / "train.jsonl", records)

        tok = _make_tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            split_long_sequences(tmp_path, HF_MODEL_ID, max_seq_len=5)

        result = _read_jsonl(tmp_path / "train.jsonl")
        assert len(result) == 4  # 1 unchanged + 3 chunks


# ── convert_to_mlx ────────────────────────────────────────────────────────────


class TestConvertToMlx:
    def test_returns_mlx_model_dir(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "mlx" / "model"
        with patch("app.scripts.flows.finetuning.finetuning._run"):
            result = convert_to_mlx(HF_MODEL_ID, mlx_dir)
        assert result == mlx_dir

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "mlx" / "nested" / "model"
        with patch("app.scripts.flows.finetuning.finetuning._run"):
            convert_to_mlx(HF_MODEL_ID, mlx_dir)
        assert mlx_dir.parent.exists()

    def test_run_called_with_mlx_lm_convert(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "model"
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            convert_to_mlx(HF_MODEL_ID, mlx_dir)

        cmd = mock_run.call_args.args[0]
        assert "-m" in cmd
        assert "mlx_lm" in cmd
        assert "convert" in cmd

    def test_run_called_with_hf_model_id(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "model"
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            convert_to_mlx(HF_MODEL_ID, mlx_dir)

        cmd = mock_run.call_args.args[0]
        assert HF_MODEL_ID in cmd

    def test_run_called_with_4bit_quantisation(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "model"
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            convert_to_mlx(HF_MODEL_ID, mlx_dir)

        cmd = mock_run.call_args.args[0]
        assert "--q-bits" in cmd
        assert "4" in cmd


# ── finetune_lora ─────────────────────────────────────────────────────────────


class TestFinetuneLora:
    def test_returns_adapter_dir(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "mlx"
        adapter_dir = tmp_path / "adapter"
        with patch("app.scripts.flows.finetuning.finetuning._run"):
            result = finetune_lora(mlx_dir, adapter_dir)
        assert result == adapter_dir

    def test_creates_adapter_dir(self, tmp_path: Path) -> None:
        mlx_dir = tmp_path / "mlx"
        adapter_dir = tmp_path / "nested" / "adapter"
        with patch("app.scripts.flows.finetuning.finetuning._run"):
            finetune_lora(mlx_dir, adapter_dir)
        assert adapter_dir.exists()

    def test_run_called_with_mlx_lm_lora(self, tmp_path: Path) -> None:
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            finetune_lora(tmp_path / "mlx", tmp_path / "adapter")

        cmd = mock_run.call_args.args[0]
        assert "mlx_lm" in cmd
        assert "lora" in cmd

    def test_run_called_with_grad_checkpoint(self, tmp_path: Path) -> None:
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            finetune_lora(tmp_path / "mlx", tmp_path / "adapter")

        cmd = mock_run.call_args.args[0]
        assert "--grad-checkpoint" in cmd

    def test_log_file_passed_to_run(self, tmp_path: Path) -> None:
        log = tmp_path / "logs" / "finetune.log"
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            finetune_lora(tmp_path / "mlx", tmp_path / "adapter", log_file=log)

        assert mock_run.call_args.kwargs.get("log_file") == log

    def test_custom_iters_passed_to_run(self, tmp_path: Path) -> None:
        with patch("app.scripts.flows.finetuning.finetuning._run") as mock_run:
            finetune_lora(tmp_path / "mlx", tmp_path / "adapter", iters=42)

        cmd = mock_run.call_args.args[0]
        assert "--iters" in cmd
        assert "42" in cmd


# ── gemma3_chembl_toon_finetune_flow ──────────────────────────────────────────

_FLOW_PATCH = "app.scripts.flows.finetuning.finetuning"


class TestGemma3ChemblToonFinetuneFlow:
    def test_all_stages_called(self) -> None:
        with patch.multiple(
            _FLOW_PATCH,
            split_long_sequences=DEFAULT,
            convert_to_mlx=DEFAULT,
            finetune_lora=DEFAULT,
            save_to_ollama=DEFAULT,
        ) as mocks:
            mocks["convert_to_mlx"].side_effect = lambda hf, d: d
            mocks["finetune_lora"].side_effect = lambda m, d, *a, **kw: d
            gemma3_chembl_toon_finetune_flow(run_name="test_run")

        assert mocks["split_long_sequences"].called
        assert mocks["convert_to_mlx"].called
        assert mocks["finetune_lora"].called
        assert mocks["save_to_ollama"].called

    def test_respects_custom_run_name(self) -> None:
        with patch.multiple(
            _FLOW_PATCH,
            split_long_sequences=DEFAULT,
            convert_to_mlx=DEFAULT,
            finetune_lora=DEFAULT,
            save_to_ollama=DEFAULT,
        ) as mocks:
            mocks["convert_to_mlx"].side_effect = lambda hf, d: d
            mocks["finetune_lora"].side_effect = lambda m, d, *a, **kw: d
            gemma3_chembl_toon_finetune_flow(run_name="my_custom_run")

        # run_name is embedded in mlx_model_dir: artifacts/<run_name>/mlx/...
        mlx_dir_arg: Path = mocks["convert_to_mlx"].call_args.args[1]
        assert "my_custom_run" in str(mlx_dir_arg)

    def test_run_name_defaults_to_timestamp(self) -> None:
        with patch.multiple(
            _FLOW_PATCH,
            split_long_sequences=DEFAULT,
            convert_to_mlx=DEFAULT,
            finetune_lora=DEFAULT,
            save_to_ollama=DEFAULT,
        ) as mocks:
            mocks["convert_to_mlx"].side_effect = lambda hf, d: d
            mocks["finetune_lora"].side_effect = lambda m, d, *a, **kw: d
            gemma3_chembl_toon_finetune_flow()

        # mlx_model_dir = artifacts/<run_name>/mlx/gemma-3-1b-pt-mlx
        mlx_dir_arg: Path = mocks["convert_to_mlx"].call_args.args[1]
        run_name = mlx_dir_arg.parts[1]  # ('artifacts', '<run_name>', 'mlx', ...)
        assert len(run_name) == 15  # YYYYMMDD_HHMMSS

    def test_hf_model_id_forwarded_to_split(self) -> None:
        custom_id = "google/gemma-3-4b-pt"
        with patch.multiple(
            _FLOW_PATCH,
            split_long_sequences=DEFAULT,
            convert_to_mlx=DEFAULT,
            finetune_lora=DEFAULT,
            save_to_ollama=DEFAULT,
        ) as mocks:
            mocks["convert_to_mlx"].side_effect = lambda hf, d: d
            mocks["finetune_lora"].side_effect = lambda m, d, *a, **kw: d
            gemma3_chembl_toon_finetune_flow(hf_model_id=custom_id, run_name="test_run")

        assert mocks["split_long_sequences"].call_args.args[1] == custom_id
