"""Tests for app/scripts/flows/finetuning/finetuning.py."""

import json
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import torch

# transformers loads these lazily; touch them once so the patches below stick.
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer  # noqa: F401

from app.scripts.flows.finetuning.finetuning import (
    HF_MODEL_ID,
    collate,
    finetune,
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
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
            patch("pathlib.Path.write_text") as mock_write_text,
        ):
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


# ── collate ───────────────────────────────────────────────────────────────────


class _CharTokenizer:
    """One token per character, right padding with 0 — enough to check masking."""

    eos_token = "|"

    def __call__(self, texts, max_length, **_):
        ids = [[ord(c) for c in t][:max_length] for t in texts]
        width = max(len(i) for i in ids)
        return {
            "input_ids": torch.tensor([i + [0] * (width - len(i)) for i in ids]),
            "attention_mask": torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids]),
        }


class TestCollate:
    def test_appends_eos_so_the_model_learns_to_stop(self) -> None:
        batch = collate(_CharTokenizer(), ["ab"])
        assert batch["input_ids"][0].tolist() == [ord("a"), ord("b"), ord("|")]

    def test_padding_is_masked_out_of_the_loss(self) -> None:
        batch = collate(_CharTokenizer(), ["a", "abc"])
        assert batch["labels"][0].tolist() == [ord("a"), ord("|"), -100, -100]
        assert batch["labels"][1].tolist() == batch["input_ids"][1].tolist()

    def test_truncates_to_max_seq_len(self) -> None:
        batch = collate(_CharTokenizer(), ["abcdef"], max_seq_len=3)
        assert batch["input_ids"].shape == (1, 3)


# ── finetune ──────────────────────────────────────────────────────────────────


def _finetune_with_mocks(tmp_path: Path, run_dir: Path) -> MagicMock:
    """Run finetune() with model, tokenizer and Trainer mocked; return the Trainer mock."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_jsonl(data_dir / "train.jsonl", [{"text": "q1"}, {"text": "q2"}])
    _write_jsonl(data_dir / "valid.jsonl", [{"text": "v1"}])
    trainer = MagicMock()
    trainer.state.log_history = [{"loss": 1.0}]
    with (
        patch("transformers.AutoTokenizer.from_pretrained"),
        patch("transformers.AutoModelForCausalLM.from_pretrained"),
        patch("transformers.Trainer", return_value=trainer) as trainer_cls,
    ):
        finetune("base", run_dir, data_dir, iters=10)
    trainer.cls_kwargs = trainer_cls.call_args.kwargs
    return trainer


class TestFinetune:
    def test_trains_on_the_text_of_each_record(self, tmp_path: Path) -> None:
        trainer = _finetune_with_mocks(tmp_path, tmp_path / "run")
        assert trainer.cls_kwargs["train_dataset"] == ["q1", "q2"]
        assert trainer.cls_kwargs["eval_dataset"] == ["v1"]

    def test_fresh_run_does_not_resume(self, tmp_path: Path) -> None:
        trainer = _finetune_with_mocks(tmp_path, tmp_path / "run")
        trainer.train.assert_called_once_with(resume_from_checkpoint=None)

    def test_resumes_when_a_checkpoint_exists(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints" / "checkpoint-200").mkdir(parents=True)
        trainer = _finetune_with_mocks(tmp_path, run_dir)
        trainer.train.assert_called_once_with(resume_from_checkpoint=True)

    def test_saves_model_and_training_log(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        trainer = _finetune_with_mocks(tmp_path, run_dir)
        trainer.save_model.assert_called_once_with(str(run_dir / "model"))
        assert json.loads((run_dir / "logs" / "train_log.json").read_text()) == [{"loss": 1.0}]


# ── gemma3_chembl_toon_finetune_flow ──────────────────────────────────────────


class TestGemma3ChemblToonFinetuneFlow:
    def _run(self, **kwargs) -> dict[str, MagicMock]:
        with patch.multiple(
            "app.scripts.flows.finetuning.finetuning",
            split_long_sequences=DEFAULT,
            finetune=DEFAULT,
            export_to_ollama=DEFAULT,
        ) as mocks:
            gemma3_chembl_toon_finetune_flow(**kwargs)
        return mocks

    def test_trains_into_the_named_run(self) -> None:
        mocks = self._run(run_name="my_run", export=False)
        base, run_dir = mocks["finetune"].call_args.args[:2]
        assert base == HF_MODEL_ID
        assert run_dir == Path("artifacts") / "my_run"

    def test_run_name_defaults_to_timestamp(self) -> None:
        mocks = self._run(export=False)
        run_dir: Path = mocks["finetune"].call_args.args[1]
        assert run_dir.name[:8].isdigit()

    def test_hf_model_id_forwarded_to_split_and_training(self) -> None:
        mocks = self._run(hf_model_id="org/other", run_name="r", export=False)
        assert mocks["split_long_sequences"].call_args.args[1] == "org/other"
        assert mocks["finetune"].call_args.args[0] == "org/other"

    def test_export_only_when_asked(self) -> None:
        assert not self._run(run_name="r", export=False)["export_to_ollama"].called
        assert self._run(run_name="r", export=True)["export_to_ollama"].called
