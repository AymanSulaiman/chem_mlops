"""Tests for app/scripts/flows/vector_store/ingest_twosides_to_lancedb.py."""

from pathlib import Path

import lancedb
import polars as pl
import pytest

from app.scripts.flows.vector_store.ingest_twosides_to_lancedb import (
    POLYPHARMACY_TABLE,
    ingest_twosides_to_lancedb,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def twosides_parquet(tmp_path: Path) -> Path:
    """Write a minimal TWOSIDES-schema Parquet to a temp directory."""
    df = pl.DataFrame(
        {
            "drug_1_rxnorn_id": [10355, 10355, 1808, 1808],
            "drug_1_concept_name": ["Temazepam", "Temazepam", "Bumetanide", "Bumetanide"],
            "drug_2_rxnorm_id": [136411, 136411, 7824, 7824],
            "drug_2_concept_name": ["Sildenafil", "Sildenafil", "Oxytocin", "Oxytocin"],
            "condition_meddra_id": [10003239, 10012345, 10003239, 10054321],
            "condition_concept_name": ["Arthralgia", "Nausea", "Arthralgia", "Dizziness"],
            "A": [7, 5, 8, 6],
            "B": [149, 100, 80, 60],
            "C": [24, 20, 15, 12],
            "D": [1536, 1000, 800, 600],
            "PRR": [4.5, 3.5, 5.0, 3.5],  # both Temazepam+Sildenafil rows exceed MIN_PRR=3.0
            "PRR_error": [0.42, 0.3, 1.19, 0.5],
            "mean_reporting_frequency": [0.045, 0.05, 0.071, 0.06],
        }
    )
    path = tmp_path / "TWOSIDES.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def lancedb_dir(tmp_path: Path) -> Path:
    """Create a temp LanceDB directory with a chembl_CHEMBL_36 subdirectory."""
    db_path = tmp_path / "lancedb"
    db_subdir = db_path / "chembl_CHEMBL_36"
    db_subdir.mkdir(parents=True)
    # Connect to initialise a valid LanceDB at this path
    lancedb.connect(str(db_subdir))
    return db_path


# ── Error cases ───────────────────────────────────────────────────────────────


class TestIngestTwosidesErrors:
    def test_raises_if_parquet_missing(self, lancedb_dir: Path) -> None:
        missing = lancedb_dir / "nonexistent.parquet"
        with pytest.raises(FileNotFoundError, match="TWOSIDES not found"):
            ingest_twosides_to_lancedb(twosides_path=missing, lancedb_dir=lancedb_dir)

    def test_raises_if_no_chembl_lancedb(self, twosides_parquet: Path, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_lancedb"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=empty_dir)


# ── Successful ingestion ──────────────────────────────────────────────────────


class TestIngestTwosidesSuccess:
    def test_creates_polypharmacy_table(self, twosides_parquet: Path, lancedb_dir: Path) -> None:
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(lancedb_dir / "chembl_CHEMBL_36"))
        assert POLYPHARMACY_TABLE in db.list_tables().tables

    def test_aggregates_side_effects_per_pair(self, twosides_parquet: Path, lancedb_dir: Path) -> None:
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(lancedb_dir / "chembl_CHEMBL_36"))
        rows = db.open_table(POLYPHARMACY_TABLE).search().to_list()
        # 4 raw rows → 2 unique pairs after aggregation
        assert len(rows) == 2

    def test_side_effects_are_joined(self, twosides_parquet: Path, lancedb_dir: Path) -> None:
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(lancedb_dir / "chembl_CHEMBL_36"))
        rows = db.open_table(POLYPHARMACY_TABLE).search().to_list()
        tem_sil = next(r for r in rows if r["drug_1_name"] == "Temazepam")
        # Both Arthralgia and Nausea must appear in the aggregated side_effects string
        assert "Arthralgia" in tem_sil["side_effects"]
        assert "Nausea" in tem_sil["side_effects"]

    def test_max_prr_is_correct(self, twosides_parquet: Path, lancedb_dir: Path) -> None:
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(lancedb_dir / "chembl_CHEMBL_36"))
        rows = db.open_table(POLYPHARMACY_TABLE).search().to_list()
        tem_sil = next(r for r in rows if r["drug_1_name"] == "Temazepam")
        assert tem_sil["max_prr"] == pytest.approx(4.5, abs=0.01)

    def test_pair_key_format(self, twosides_parquet: Path, lancedb_dir: Path) -> None:
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(lancedb_dir / "chembl_CHEMBL_36"))
        rows = db.open_table(POLYPHARMACY_TABLE).search().to_list()
        for row in rows:
            assert "|" in row["pair_key"]

    def test_overwrites_on_rerun(self, twosides_parquet: Path, lancedb_dir: Path) -> None:
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        ingest_twosides_to_lancedb(twosides_path=twosides_parquet, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(lancedb_dir / "chembl_CHEMBL_36"))
        rows = db.open_table(POLYPHARMACY_TABLE).search().to_list()
        assert len(rows) == 2  # no duplicates from double-run
