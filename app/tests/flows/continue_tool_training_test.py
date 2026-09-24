"""Tests for app/scripts/flows/finetuning/continue_tool_training.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.scripts.flows.finetuning.continue_tool_training import (
    build_tool_mix,
    continue_tool_training,
)
from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    TOOL_RESULT_HEADER,
)


def _dataset(tmp_path: Path, tool: int, prose: int) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    records = [
        {
            "text": f"### Question\nq{i}\n\n### Answer\n{{...}}\n\n### Question\n{TOOL_RESULT_HEADER} (t)\n{{}}\n\n### Answer\na"
        }
        for i in range(tool)
    ] + [{"text": f"### Question\np{i}\n\n### Answer\nprose"} for i in range(prose)]
    (data_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return data_dir


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestBuildToolMix:
    def test_keeps_every_tool_record_and_samples_prose(self, tmp_path: Path) -> None:
        data_dir = _dataset(tmp_path, tool=10, prose=500)
        out_dir = tmp_path / "mix"

        tools, prose = build_tool_mix(data_dir, out_dir, prose_ratio=2)

        assert (tools, prose) == (10, 20)
        written = _read(out_dir / "train.jsonl") + _read(out_dir / "valid.jsonl")
        assert len(written) == 30
        assert sum(TOOL_RESULT_HEADER in r["text"] for r in written) == 10

    def test_takes_all_prose_when_there_is_less_than_the_ratio_asks(self, tmp_path: Path) -> None:
        data_dir = _dataset(tmp_path, tool=10, prose=5)
        assert build_tool_mix(data_dir, tmp_path / "mix", prose_ratio=2) == (10, 5)

    def test_sampling_is_reproducible(self, tmp_path: Path) -> None:
        data_dir = _dataset(tmp_path, tool=5, prose=200)
        first = build_tool_mix(data_dir, tmp_path / "a")
        second = build_tool_mix(data_dir, tmp_path / "b")
        assert first == second
        assert _read(tmp_path / "a" / "train.jsonl") == _read(tmp_path / "b" / "train.jsonl")

    def test_refuses_a_dataset_built_before_the_tool_category(self, tmp_path: Path) -> None:
        data_dir = _dataset(tmp_path, tool=0, prose=20)
        with pytest.raises(RuntimeError, match="Rebuild the dataset"):
            build_tool_mix(data_dir, tmp_path / "mix")

    def test_missing_dataset_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train.jsonl"):
            build_tool_mix(tmp_path / "nope", tmp_path / "mix")


class TestContinueToolTraining:
    def test_resumes_from_the_source_adapter_into_a_new_run(self, tmp_path: Path) -> None:
        data_dir = _dataset(tmp_path, tool=4, prose=40)
        source = tmp_path / "artifacts" / "20260101_000000"
        (source / "adapters" / "gemma3-1b-pt-chembl-toon").mkdir(parents=True)
        (source / "adapters" / "gemma3-1b-pt-chembl-toon" / "adapters.safetensors").touch()
        (source / "mlx" / "gemma-3-1b-pt-mlx").mkdir(parents=True)

        with (
            patch(
                "app.scripts.flows.finetuning.continue_tool_training.ARTIFACTS_DIR",
                tmp_path / "artifacts",
            ),
            patch(
                "app.scripts.flows.finetuning.continue_tool_training.finetune_lora"
            ) as mock_train,
        ):
            run_dir = continue_tool_training(
                from_run=source,
                data_dir=data_dir,
                mix_dir=tmp_path / "mix",
                iters=120,
                run_name="20260102_000000_tools",
            )

        kwargs = mock_train.call_args.kwargs
        assert kwargs["resume_from"].name == "adapters.safetensors"
        assert kwargs["iters"] == 120
        # The source adapter must not be the training output: it stays intact.
        assert run_dir != source
        assert mock_train.call_args.args[1] != source / "adapters" / "gemma3-1b-pt-chembl-toon"

    def test_refuses_a_run_that_never_finished_training(self, tmp_path: Path) -> None:
        source = tmp_path / "artifacts" / "20260101_000000"
        (source / "adapters" / "gemma3-1b-pt-chembl-toon").mkdir(parents=True)  # config only
        with pytest.raises(FileNotFoundError, match="No trained adapter"):
            continue_tool_training(from_run=source, data_dir=_dataset(tmp_path, 2, 2))
