"""Tests for app/scripts/flows/finetuning/export_to_ollama.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.scripts.flows.finetuning.export_to_ollama import (
    DEFAULT_MODEL_NAME,
    QUANTIZE,
    export_to_ollama,
    latest_run_dir,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _llama_quantize_installed():
    """Tests mock the subprocess calls; don't depend on llama.cpp being installed."""
    with patch("shutil.which", return_value="/bin/llama-quantize"):
        yield


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A run directory as a finished training run leaves it."""
    rd = tmp_path / "20260101_000000"
    (rd / "model").mkdir(parents=True)
    (rd / "model" / "config.json").write_text("{}")
    return rd


# ── latest_run_dir ────────────────────────────────────────────────────────────


class TestLatestRunDir:
    def test_returns_lexicographically_last_dir(self, tmp_path: Path) -> None:
        for name in ("20260101_000000", "20260102_000000", "20260103_000000"):
            (tmp_path / name).mkdir()
        assert latest_run_dir(tmp_path).name == "20260103_000000"

    def test_single_dir_is_returned(self, tmp_path: Path) -> None:
        (tmp_path / "20260101_000000").mkdir()
        assert latest_run_dir(tmp_path).name == "20260101_000000"

    def test_raises_if_no_dirs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No run directories found"):
            latest_run_dir(tmp_path)

    def test_ignores_files(self, tmp_path: Path) -> None:
        (tmp_path / "20260101_000000").mkdir()
        (tmp_path / "README.md").write_text("not a run dir")
        assert latest_run_dir(tmp_path).name == "20260101_000000"

    def test_returns_path_object(self, tmp_path: Path) -> None:
        (tmp_path / "20260101_000000").mkdir()
        result = latest_run_dir(tmp_path)
        assert isinstance(result, Path)


# ── export_to_ollama ──────────────────────────────────────────────────────────


class TestExportToOllama:
    def test_raises_if_training_never_started(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No checkpoints either"):
            export_to_ollama(run_dir=tmp_path / "run")

    def test_names_the_checkpoint_to_resume_from(self, tmp_path: Path) -> None:
        rd = tmp_path / "run"
        for step in (200, 1000):  # numeric, not lexical, order
            (rd / "checkpoints" / f"checkpoint-{step}").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="checkpoint-1000"):
            export_to_ollama(run_dir=rd)

    def test_skips_without_force_if_output_exists(self, run_dir: Path) -> None:
        output_dir = run_dir / "ollama"
        output_dir.mkdir(parents=True)

        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run,
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir, force=False)
            mock_run.assert_not_called()

    def test_overwrites_with_force(self, run_dir: Path) -> None:
        output_dir = run_dir / "ollama"
        output_dir.mkdir(parents=True)
        sentinel = output_dir / "old_file.txt"
        sentinel.write_text("stale")

        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run"),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir, force=True)

        assert not sentinel.exists()

    def test_modelfile_has_no_system_prompt(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run"),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "ollama" / "Modelfile").read_text()
        # The model is trained without one; a baked-in SYSTEM stopped all tool calls.
        assert "\nSYSTEM" not in modelfile

    def test_modelfile_contains_gguf_path(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run"),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "ollama" / "Modelfile").read_text()
        assert modelfile.startswith("FROM ./chembl-drug-chat.gguf\n")

    def test_modelfile_contains_stop_tokens(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run"),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "ollama" / "Modelfile").read_text()
        assert 'PARAMETER stop "### Question"' in modelfile
        # "### Answer" must NOT be a stop token — it appears in the prompt template
        # itself, and including it as a stop token caused the model to emit an
        # immediate EOS and return empty content.
        assert 'PARAMETER stop "### Answer"' not in modelfile

    def test_modelfile_contains_sampling_params(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run"),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "ollama" / "Modelfile").read_text()
        assert "PARAMETER temperature" in modelfile
        assert "PARAMETER repeat_penalty" in modelfile
        assert "PARAMETER num_ctx" in modelfile

    def test_ollama_create_called_with_model_name(self, run_dir: Path) -> None:
        custom_name = "my-model:test"
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run,
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir, model_name=custom_name)

        ollama_call = mock_run.call_args_list[-1]
        cmd = ollama_call.args[0]
        assert "ollama" in cmd
        assert "create" in cmd
        assert custom_name in cmd

    def test_gguf_convert_is_called(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run"),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert") as mock_convert,
        ):
            export_to_ollama(run_dir=run_dir)

        mock_convert.assert_called_once()

    def test_quantizes_before_registering(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run,
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert") as mock_convert,
        ):
            # The converter writes the F16 file that the quantizer then deletes.
            mock_convert.side_effect = lambda src, dst: dst.touch()
            export_to_ollama(run_dir=run_dir)

        quantize_cmd, create_cmd = (c.args[0] for c in mock_run.call_args_list)
        out = run_dir / "ollama"
        assert quantize_cmd == [
            "/bin/llama-quantize",
            str(out / "chembl-drug-chat.f16.gguf"),
            str(out / "chembl-drug-chat.gguf"),
            QUANTIZE,
        ]
        assert create_cmd[:2] == ["ollama", "create"]
        assert not (out / "chembl-drug-chat.f16.gguf").exists()

    def test_empty_quantize_ships_f16(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run,
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert") as mock_convert,
        ):
            export_to_ollama(run_dir=run_dir, quantize="")

        assert mock_convert.call_args.args[1] == run_dir / "ollama" / "chembl-drug-chat.gguf"
        assert [c.args[0][0] for c in mock_run.call_args_list] == ["ollama"]

    def test_missing_quantizer_is_named(self, run_dir: Path) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
            pytest.raises(FileNotFoundError, match="llama-quantize"),
        ):
            export_to_ollama(run_dir=run_dir)

    def test_default_model_name_is_used(self, run_dir: Path) -> None:
        with (
            patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run,
            patch("app.scripts.flows.finetuning.convert_gemma3_gguf.convert"),
        ):
            export_to_ollama(run_dir=run_dir)

        ollama_call = mock_run.call_args_list[-1]
        assert DEFAULT_MODEL_NAME in ollama_call.args[0]
