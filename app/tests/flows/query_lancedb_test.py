"""Unit tests for app/scripts/flows/vector_store/query_lancedb.py."""

from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pytest
from rdkit import Chem

from app.scripts.flows.vector_store.ingest_to_lancedb import (
    COMPOUNDS_TABLE,
    _FP_GEN,
)
from app.scripts.flows.vector_store.query_lancedb import (
    _open_table,
    _resolve_lancedb_uri,
    _smiles_to_query_vector,
    _run_sanity_check,
    get_compound,
    query_compounds,
)


ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
INVALID_SMILES = "not_a_smiles"


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_record(smiles: str, chembl_id: str, pref_name: str, mw: float) -> dict[str, Any]:
    fp = _FP_GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.float32)
    return {
        "chembl_id": chembl_id,
        "pref_name": pref_name,
        "mw_freebase": mw,
        "canonical_smiles": smiles,
        "vector": fp.tolist(),
    }


@pytest.fixture()
def lancedb_dir(tmp_path: Path) -> str:
    """Populate a temp LanceDB with three known compounds."""
    uri = str(tmp_path / "chembl_CHEMBL_36")
    db = lancedb.connect(uri)
    records = [
        _make_record(ASPIRIN_SMILES, "CHEMBL25", "Aspirin", 180.16),
        _make_record(CAFFEINE_SMILES, "CHEMBL113", "Caffeine", 194.19),
        _make_record(IBUPROFEN_SMILES, "CHEMBL521", "Ibuprofen", 206.29),
    ]
    db.create_table(COMPOUNDS_TABLE, data=records, mode="overwrite")
    return str(tmp_path)


# ── _resolve_lancedb_uri ──────────────────────────────────────────────────────


class TestResolveLancedbUri:
    def test_returns_latest_versioned_dir(self, lancedb_dir: str) -> None:
        uri = _resolve_lancedb_uri(lancedb_dir)
        assert uri.endswith("chembl_CHEMBL_36")

    def test_picks_lexicographically_last(self, tmp_path: Path) -> None:
        for name in ("chembl_CHEMBL_34", "chembl_CHEMBL_35", "chembl_CHEMBL_36"):
            (tmp_path / name).mkdir()
        uri = _resolve_lancedb_uri(str(tmp_path))
        assert uri.endswith("chembl_CHEMBL_36")

    def test_raises_if_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_lancedb_uri(str(tmp_path / "nonexistent"))

    def test_raises_if_no_chembl_subdir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No chembl_CHEMBL"):
            _resolve_lancedb_uri(str(tmp_path))


# ── _smiles_to_query_vector ───────────────────────────────────────────────────


class TestSmilesToQueryVector:
    def test_valid_smiles_returns_float_list(self) -> None:
        vec = _smiles_to_query_vector(ASPIRIN_SMILES)
        assert isinstance(vec, list)
        assert len(vec) == 2048
        assert all(isinstance(v, float) for v in vec)

    def test_invalid_smiles_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            _smiles_to_query_vector(INVALID_SMILES)

    def test_none_like_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            _smiles_to_query_vector("")


# ── _open_table ───────────────────────────────────────────────────────────────


class TestOpenTable:
    def test_opens_table_successfully(self, lancedb_dir: str) -> None:
        table = _open_table(lancedb_dir)
        assert table is not None

    def test_raises_if_table_missing(self, tmp_path: Path) -> None:
        uri = str(tmp_path / "chembl_CHEMBL_36")
        lancedb.connect(uri)  # create empty db, no tables
        with pytest.raises(FileNotFoundError, match="not found"):
            _open_table(str(tmp_path))


# ── query_compounds ───────────────────────────────────────────────────────────


class TestQueryCompounds:
    def test_returns_n_results(self, lancedb_dir: str) -> None:
        results = query_compounds(ASPIRIN_SMILES, n=2, lancedb_dir=lancedb_dir)
        assert len(results) == 2

    def test_top_hit_is_self(self, lancedb_dir: str) -> None:
        results = query_compounds(ASPIRIN_SMILES, n=3, lancedb_dir=lancedb_dir)
        assert results[0]["chembl_id"] == "CHEMBL25"

    def test_results_ordered_by_distance(self, lancedb_dir: str) -> None:
        results = query_compounds(ASPIRIN_SMILES, n=3, lancedb_dir=lancedb_dir)
        distances = [r["_distance"] for r in results]
        assert distances == sorted(distances)

    def test_vector_column_stripped(self, lancedb_dir: str) -> None:
        results = query_compounds(ASPIRIN_SMILES, n=1, lancedb_dir=lancedb_dir)
        assert "vector" not in results[0]

    def test_metadata_present(self, lancedb_dir: str) -> None:
        results = query_compounds(ASPIRIN_SMILES, n=1, lancedb_dir=lancedb_dir)
        row = results[0]
        assert "pref_name" in row
        assert "mw_freebase" in row

    def test_invalid_smiles_raises(self, lancedb_dir: str) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            query_compounds(INVALID_SMILES, lancedb_dir=lancedb_dir)

    def test_missing_lancedb_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            query_compounds(ASPIRIN_SMILES, lancedb_dir=str(tmp_path / "missing"))


# ── get_compound ──────────────────────────────────────────────────────────────


class TestGetCompound:
    def test_returns_correct_record(self, lancedb_dir: str) -> None:
        record = get_compound("CHEMBL25", lancedb_dir=lancedb_dir)
        assert record is not None
        assert record["chembl_id"] == "CHEMBL25"
        assert record["pref_name"] == "Aspirin"

    def test_returns_none_for_unknown_id(self, lancedb_dir: str) -> None:
        record = get_compound("CHEMBL_DOES_NOT_EXIST", lancedb_dir=lancedb_dir)
        assert record is None

    def test_vector_column_stripped(self, lancedb_dir: str) -> None:
        record = get_compound("CHEMBL113", lancedb_dir=lancedb_dir)
        assert record is not None
        assert "vector" not in record

    def test_metadata_present(self, lancedb_dir: str) -> None:
        record = get_compound("CHEMBL521", lancedb_dir=lancedb_dir)
        assert record is not None
        assert record["mw_freebase"] == pytest.approx(206.29)

    def test_missing_lancedb_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_compound("CHEMBL25", lancedb_dir=str(tmp_path / "missing"))


# ── _run_sanity_check ─────────────────────────────────────────────────────────


class TestRunSanityCheck:
    def test_passes_against_known_data(self, lancedb_dir: str) -> None:
        # Should complete without raising — all 4 internal assertions must hold
        _run_sanity_check(lancedb_dir=lancedb_dir)

    def test_fails_on_empty_store(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _run_sanity_check(lancedb_dir=str(tmp_path / "missing"))
