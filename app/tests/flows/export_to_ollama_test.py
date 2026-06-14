"""Tests for app/scripts/flows/finetuning/export_to_ollama.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.scripts.flows.finetuning.export_to_ollama import (
    DEFAULT_ADAPTER_SUBDIR,
    DEFAULT_MLX_SUBDIR,
    DEFAULT_MODEL_NAME,
    SYSTEM_PROMPT,
    export_to_ollama,
    latest_run_dir,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A run directory with the expected mlx model and adapter subdirs present."""
    rd = tmp_path / "20260101_000000"
    (rd / DEFAULT_MLX_SUBDIR).mkdir(parents=True)
    (rd / DEFAULT_ADAPTER_SUBDIR).mkdir(parents=True)
    return rd


def _patch_export(run_dir: Path, **kwargs):
    """Context manager that patches both expensive calls in export_to_ollama."""
    return patch.multiple(
        "app.scripts.flows.finetuning.export_to_ollama",
        _run=kwargs.get("_run", patch("app.scripts.flows.finetuning.export_to_ollama._run").start()),
    )


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
    def test_raises_if_mlx_model_dir_missing(self, tmp_path: Path) -> None:
        rd = tmp_path / "20260101_000000"
        # Only adapter dir exists — mlx model dir is absent
        (rd / DEFAULT_ADAPTER_SUBDIR).mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="MLX model not found"):
            export_to_ollama(run_dir=rd)

    def test_raises_if_adapter_dir_missing(self, tmp_path: Path) -> None:
        rd = tmp_path / "20260101_000000"
        # Only mlx model dir exists — adapter is absent
        (rd / DEFAULT_MLX_SUBDIR).mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            export_to_ollama(run_dir=rd)

    def test_skips_without_force_if_output_exists(self, run_dir: Path) -> None:
        output_dir = run_dir / "mlx" / "ollama"
        output_dir.mkdir(parents=True)

        with patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run, patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir, force=False)
            mock_run.assert_not_called()

    def test_overwrites_with_force(self, run_dir: Path) -> None:
        output_dir = run_dir / "mlx" / "ollama"
        output_dir.mkdir(parents=True)
        sentinel = output_dir / "old_file.txt"
        sentinel.write_text("stale")

        with patch("app.scripts.flows.finetuning.export_to_ollama._run"), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir, force=True)

        assert not sentinel.exists()

    def test_modelfile_contains_system_prompt(self, run_dir: Path) -> None:
        with patch("app.scripts.flows.finetuning.export_to_ollama._run"), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "mlx" / "ollama" / "Modelfile").read_text()
        assert SYSTEM_PROMPT in modelfile

    def test_modelfile_contains_gguf_path(self, run_dir: Path) -> None:
        with patch("app.scripts.flows.finetuning.export_to_ollama._run"), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "mlx" / "ollama" / "Modelfile").read_text()
        assert "chembl-drug-chat.gguf" in modelfile

    def test_modelfile_contains_stop_tokens(self, run_dir: Path) -> None:
        with patch("app.scripts.flows.finetuning.export_to_ollama._run"), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "mlx" / "ollama" / "Modelfile").read_text()
        assert 'PARAMETER stop "### Question"' in modelfile
        assert 'PARAMETER stop "### Answer"' in modelfile

    def test_modelfile_contains_sampling_params(self, run_dir: Path) -> None:
        with patch("app.scripts.flows.finetuning.export_to_ollama._run"), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir)

        modelfile = (run_dir / "mlx" / "ollama" / "Modelfile").read_text()
        assert "PARAMETER temperature" in modelfile
        assert "PARAMETER repeat_penalty" in modelfile
        assert "PARAMETER num_ctx" in modelfile

    def test_ollama_create_called_with_model_name(self, run_dir: Path) -> None:
        custom_name = "my-model:test"
        with patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run, patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir, model_name=custom_name)

        ollama_call = mock_run.call_args_list[-1]
        cmd = ollama_call.args[0]
        assert "ollama" in cmd
        assert "create" in cmd
        assert custom_name in cmd

    def test_gguf_convert_is_called(self, run_dir: Path) -> None:
        with patch("app.scripts.flows.finetuning.export_to_ollama._run"), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ) as mock_convert:
            export_to_ollama(run_dir=run_dir)

        mock_convert.assert_called_once()

    def test_fuse_called_before_convert(self, run_dir: Path) -> None:
        call_order: list[str] = []

        def record_run(cmd):
            call_order.append("fuse" if "fuse" in cmd else "ollama")

        def record_convert(src, dst):
            call_order.append("convert")

        with patch("app.scripts.flows.finetuning.export_to_ollama._run", side_effect=record_run), patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert", side_effect=record_convert
        ):
            export_to_ollama(run_dir=run_dir)

        assert call_order.index("fuse") < call_order.index("convert")
        assert call_order.index("convert") < call_order.index("ollama")

    def test_default_model_name_is_used(self, run_dir: Path) -> None:
        with patch("app.scripts.flows.finetuning.export_to_ollama._run") as mock_run, patch(
            "app.scripts.flows.finetuning.convert_gemma3_gguf.convert"
        ):
            export_to_ollama(run_dir=run_dir)

        ollama_call = mock_run.call_args_list[-1]
        assert DEFAULT_MODEL_NAME in ollama_call.args[0]
