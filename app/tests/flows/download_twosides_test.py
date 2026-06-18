"""Tests for app/scripts/flows/llm_finetuning_data/download_twosides.py."""

import contextlib
import gzip
from pathlib import Path

import httpx
import polars as pl
import pytest

from app.scripts.flows.llm_finetuning_data.download_twosides import download_twosides

_SAMPLE_CSV = (
    "drug_1_rxnorn_id,drug_1_concept_name,drug_2_rxnorm_id,drug_2_concept_name,"
    "condition_meddra_id,condition_concept_name,A,B,C,D,PRR,PRR_error,mean_reporting_frequency\n"
    "10355,Temazepam,136411,Sildenafil,10003239,Arthralgia,7,149,24,1536,2.92,0.42,0.045\n"
    "1808,Bumetanide,7824,Oxytocin,10003239,Arthralgia,1,13,2,138,5.0,1.19,0.071\n"
)


def _mock_stream(csv_text: str):
    """Return a mock for httpx.stream that yields gzip-compressed CSV bytes."""
    compressed = gzip.compress(csv_text.encode())

    @contextlib.contextmanager
    def _ctx(method, url, **kwargs):
        class _Resp:
            headers = {"content-length": str(len(compressed))}

            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self, chunk_size: int = 65_536):
                yield compressed

        yield _Resp()

    return _ctx


class TestDownloadTwosides:
    def test_skips_if_file_already_exists(self, tmp_path: Path) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        output.write_bytes(b"existing")
        result = download_twosides(output_path=output)
        assert result == output
        assert output.read_bytes() == b"existing"  # not overwritten

    def test_returns_output_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        result = download_twosides(output_path=output)
        assert result == output

    def test_writes_parquet_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output)
        assert output.exists()

    def test_parquet_row_count_matches_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output)
        df = pl.read_parquet(output)
        assert len(df) == 2

    def test_parquet_has_expected_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output)
        df = pl.read_parquet(output)
        for col in ("drug_1_concept_name", "drug_2_concept_name", "condition_concept_name", "PRR", "A"):
            assert col in df.columns, f"Missing column: {col}"

    def test_numeric_columns_have_correct_types(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output)
        df = pl.read_parquet(output)
        assert df["PRR"].dtype in (pl.Float32, pl.Float64)
        assert df["A"].dtype in (pl.Int32, pl.Int64)

    def test_force_overwrites_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        output.write_bytes(b"stale data")
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output, force=True)
        df = pl.read_parquet(output)
        assert len(df) == 2  # new data, not stale bytes

    def test_creates_parent_directory_if_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "nested" / "subdir" / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output)
        assert output.exists()

    def test_data_values_preserved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = tmp_path / "TWOSIDES.parquet"
        monkeypatch.setattr(httpx, "stream", _mock_stream(_SAMPLE_CSV))
        download_twosides(output_path=output)
        df = pl.read_parquet(output)
        names = df["drug_1_concept_name"].to_list()
        assert "Temazepam" in names
        assert "Bumetanide" in names
