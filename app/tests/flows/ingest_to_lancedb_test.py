from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import pytest

from app.scripts.flows.vector_store.ingest_to_lancedb import (
    COMPOUNDS_TABLE,
    MORGAN_BITS,
    _build_flat_df,
    _collect,
    _resolve_chembl_version,
    _smiles_to_fp,
    _write_to_lancedb,
    ingest_compounds_to_lancedb,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
INVALID_SMILES = "not_a_smiles"


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> str:
    """Write minimal ChEMBL-shaped parquet stubs into a temp directory."""
    d = str(tmp_path)

    pl.DataFrame(
        {
            "chembl_release_id": [34, 35, 36],
            "chembl_release": ["CHEMBL_34", "CHEMBL_35", "CHEMBL_36"],
        }
    ).write_parquet(f"{d}/chembl_release.parquet")

    pl.DataFrame(
        {
            "molregno": [1, 2, 3],
            "pref_name": ["Aspirin", "Caffeine", "Dummy"],
            "chembl_id": ["CHEMBL25", "CHEMBL113", "CHEMBL999"],
            "structure_type": ["MOL", "MOL", "NONE"],  # row 3 should be filtered out
            "molecule_type": ["Small molecule", "Small molecule", "Small molecule"],
            "max_phase": [4, 4, None],
            "therapeutic_flag": [1, 1, 0],
            "dosed_ingredient": [True, True, False],
            "first_approval": [1950, 1962, None],
            "oral": [True, True, None],
            "parenteral": [False, False, None],
            "topical": [False, False, None],
            "black_box_warning": [0, 0, 0],
            "natural_product": [0, 0, 0],
            "first_in_class": [0, 0, 0],
            "chirality": [0, 0, 0],
            "prodrug": [0, 0, 0],
            "inorganic_flag": [0, 0, 0],
            "usan_year": pl.Series([None, None, None], dtype=pl.Int64),
            "availability_type": [2, 2, None],
            "usan_stem": pl.Series([None, None, None], dtype=pl.Utf8),
            "polymer_flag": [0, 0, 0],
            "usan_substem": pl.Series([None, None, None], dtype=pl.Utf8),
            "usan_stem_definition": pl.Series([None, None, None], dtype=pl.Utf8),
            "withdrawn_flag": [0, 0, 0],
            "chemical_probe": [0, 0, 0],
            "orphan": [0, 0, 0],
            "veterinary": [0, 0, 0],
        }
    ).write_parquet(f"{d}/molecule_dictionary.parquet")

    pl.DataFrame(
        {
            "molregno": [1, 2, 3],
            "canonical_smiles": [ASPIRIN_SMILES, CAFFEINE_SMILES, None],  # row 3 has no SMILES
            "standard_inchi_key": [
                "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                None,
            ],
        }
    ).write_parquet(f"{d}/compound_structures.parquet")

    pl.DataFrame(
        {
            "molregno": [1, 2],
            "mw_freebase": [180.16, 194.19],
            "alogp": [1.19, -0.07],
            "hba": [3, 3],
            "hbd": [1, 0],
            "psa": [63.6, 58.4],
            "qed_weighted": [0.55, 0.67],
            "full_molformula": ["C9H8O4", "C8H10N4O2"],
            "num_ro5_violations": [0, 0],
            "heavy_atoms": [13, 14],
        }
    ).write_parquet(f"{d}/compound_properties.parquet")

    pl.DataFrame(
        {
            "molregno": [1, 2],
            "parent_molregno": [1, 2],
            "active_molregno": [1, 2],
        }
    ).write_parquet(f"{d}/molecule_hierarchy.parquet")

    pl.DataFrame(
        {"stem": ["salicyl"], "annotation": ["salicylates"], "stem_class": ["INN"]}
    ).write_parquet(f"{d}/usan_stems.parquet")
    pl.DataFrame(
        {
            "molregno": pl.Series([], dtype=pl.Int64),
            "mechanism_of_action": pl.Series([], dtype=pl.Utf8),
            "action_type": pl.Series([], dtype=pl.Utf8),
            "tid": pl.Series([], dtype=pl.Int64),
        }
    ).write_parquet(f"{d}/drug_mechanism.parquet")
    pl.DataFrame(
        {"tid": pl.Series([], dtype=pl.Int64), "pref_name": pl.Series([], dtype=pl.Utf8)}
    ).write_parquet(f"{d}/target_dictionary.parquet")
    pl.DataFrame(
        {
            "molregno": pl.Series([], dtype=pl.Int64),
            "max_phase_for_ind": pl.Series([], dtype=pl.Float64),
            "mesh_heading": pl.Series([], dtype=pl.Utf8),
            "efo_term": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/drug_indication.parquet")
    pl.DataFrame(
        {
            "molregno": pl.Series([], dtype=pl.Int64),
            "warning_type": pl.Series([], dtype=pl.Utf8),
            "warning_class": pl.Series([], dtype=pl.Utf8),
            "warning_description": pl.Series([], dtype=pl.Utf8),
            "warning_country": pl.Series([], dtype=pl.Utf8),
            "warning_year": pl.Series([], dtype=pl.Int64),
        }
    ).write_parquet(f"{d}/drug_warning.parquet")
    pl.DataFrame(
        {"molregno": pl.Series([], dtype=pl.Int64), "synonyms": pl.Series([], dtype=pl.Utf8)}
    ).write_parquet(f"{d}/molecule_synonyms.parquet")
    pl.DataFrame(
        {"molregno": pl.Series([], dtype=pl.Int64), "level5": pl.Series([], dtype=pl.Utf8)}
    ).write_parquet(f"{d}/molecule_atc_classification.parquet")
    pl.DataFrame(
        {
            "level5": pl.Series([], dtype=pl.Utf8),
            "who_name": pl.Series([], dtype=pl.Utf8),
            "level1_description": pl.Series([], dtype=pl.Utf8),
            "level2_description": pl.Series([], dtype=pl.Utf8),
            "level3_description": pl.Series([], dtype=pl.Utf8),
            "level4_description": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/atc_classification.parquet")
    pl.DataFrame(
        {"record_id": pl.Series([], dtype=pl.Int64), "molregno": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(f"{d}/compound_records.parquet")
    pl.DataFrame(
        {
            "met_id": pl.Series([], dtype=pl.Int64),
            "substrate_record_id": pl.Series([], dtype=pl.Int64),
            "enzyme_name": pl.Series([], dtype=pl.Utf8),
            "met_conversion": pl.Series([], dtype=pl.Utf8),
            "organism": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/metabolism.parquet")
    pl.DataFrame(
        {"product_id": pl.Series([], dtype=pl.Utf8), "molregno": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(f"{d}/formulations.parquet")
    pl.DataFrame(
        {
            "product_id": pl.Series([], dtype=pl.Utf8),
            "trade_name": pl.Series([], dtype=pl.Utf8),
            "route": pl.Series([], dtype=pl.Utf8),
            "dosage_form": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/products.parquet")
    pl.DataFrame(
        {
            "cpd_str_alert_id": pl.Series([], dtype=pl.Int64),
            "molregno": pl.Series([], dtype=pl.Int64),
            "alert_id": pl.Series([], dtype=pl.Int64),
        }
    ).write_parquet(f"{d}/compound_structural_alerts.parquet")
    pl.DataFrame(
        {
            "alert_id": pl.Series([], dtype=pl.Int64),
            "alert_set_id": pl.Series([], dtype=pl.Int64),
            "alert_name": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/structural_alerts.parquet")
    pl.DataFrame(
        {"alert_set_id": pl.Series([], dtype=pl.Int64), "set_name": pl.Series([], dtype=pl.Utf8)}
    ).write_parquet(f"{d}/structural_alert_sets.parquet")
    pl.DataFrame(
        {
            "atc_code": pl.Series([], dtype=pl.Utf8),
            "ddd_value": pl.Series([], dtype=pl.Float64),
            "ddd_units": pl.Series([], dtype=pl.Utf8),
            "ddd_admr": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/defined_daily_dose.parquet")
    pl.DataFrame(
        {
            "activity_id": pl.Series([], dtype=pl.Int64),
            "assay_id": pl.Series([], dtype=pl.Int64),
            "molregno": pl.Series([], dtype=pl.Int64),
            "standard_flag": pl.Series([], dtype=pl.Int64),
            "pchembl_value": pl.Series([], dtype=pl.Float64),
            "standard_type": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(f"{d}/activities.parquet")
    pl.DataFrame(
        {"assay_id": pl.Series([], dtype=pl.Int64), "tid": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(f"{d}/assays.parquet")

    return d


@pytest.fixture()
def lancedb_dir(tmp_path: Path) -> str:
    return str(tmp_path / "lancedb")


# ── _collect ──────────────────────────────────────────────────────────────────


class TestCollect:
    def test_returns_dataframe(self) -> None:
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = _collect(lf)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 1)

    def test_preserves_values(self) -> None:
        lf = pl.LazyFrame({"x": [10, 20], "y": ["a", "b"]})
        result = _collect(lf)
        assert result["x"].to_list() == [10, 20]
        assert result["y"].to_list() == ["a", "b"]


# ── _smiles_to_fp ─────────────────────────────────────────────────────────────


class TestSmilesToFp:
    def test_valid_smiles_returns_ndarray(self) -> None:
        fp = _smiles_to_fp(ASPIRIN_SMILES)
        assert fp is not None
        assert isinstance(fp, np.ndarray)

    def test_fp_length_equals_morgan_bits(self) -> None:
        fp = _smiles_to_fp(CAFFEINE_SMILES)
        assert fp is not None
        assert len(fp) == MORGAN_BITS

    def test_fp_dtype_is_float32(self) -> None:
        fp = _smiles_to_fp(ASPIRIN_SMILES)
        assert fp is not None
        assert fp.dtype == np.float32

    def test_fp_values_are_binary(self) -> None:
        fp = _smiles_to_fp(ASPIRIN_SMILES)
        assert fp is not None
        assert set(fp.tolist()).issubset({0.0, 1.0})

    def test_invalid_smiles_returns_none(self) -> None:
        assert _smiles_to_fp(INVALID_SMILES) is None

    def test_different_molecules_produce_different_fps(self) -> None:
        fp1 = _smiles_to_fp(ASPIRIN_SMILES)
        fp2 = _smiles_to_fp(CAFFEINE_SMILES)
        assert fp1 is not None and fp2 is not None
        assert not np.array_equal(fp1, fp2)

    def test_same_smiles_produces_same_fp(self) -> None:
        fp1 = _smiles_to_fp(ASPIRIN_SMILES)
        fp2 = _smiles_to_fp(ASPIRIN_SMILES)
        assert fp1 is not None and fp2 is not None
        assert np.array_equal(fp1, fp2)


# ── _resolve_chembl_version ───────────────────────────────────────────────────


class TestResolveChemblVersion:
    def test_returns_latest_version(self, parquet_dir: str) -> None:
        version = _resolve_chembl_version(parquet_dir)
        assert version == "CHEMBL_36"

    def test_returns_string(self, parquet_dir: str) -> None:
        version = _resolve_chembl_version(parquet_dir)
        assert isinstance(version, str)


# ── _build_flat_df ────────────────────────────────────────────────────────────


class TestBuildFlatDf:
    def test_filters_to_mol_structure_type(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        # Row with structure_type == "NONE" should be excluded
        assert "Dummy" not in df["pref_name"].to_list()

    def test_drops_rows_without_smiles(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        assert df["canonical_smiles"].null_count() == 0

    def test_contains_expected_compounds(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        assert set(df["chembl_id"].to_list()) == {"CHEMBL25", "CHEMBL113"}

    def test_has_compound_properties_columns(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        for col in ("mw_freebase", "alogp", "qed_weighted", "full_molformula"):
            assert col in df.columns, f"Missing column: {col}"

    def test_has_inchi_key_column(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        assert "standard_inchi_key" in df.columns

    def test_returns_polars_dataframe(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        assert isinstance(df, pl.DataFrame)

    def test_row_count_matches_valid_smiles(self, parquet_dir: str) -> None:
        df = _build_flat_df(parquet_dir)
        # 2 rows have valid SMILES (aspirin + caffeine); 1 is NONE structure type, 1 has null SMILES
        assert len(df) == 2


# ── _write_to_lancedb ─────────────────────────────────────────────────────────


class TestWriteToLancedb:
    def _make_flat_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "chembl_id": ["CHEMBL25", "CHEMBL113", "CHEMBL999"],
                "canonical_smiles": [ASPIRIN_SMILES, CAFFEINE_SMILES, INVALID_SMILES],
                "standard_inchi_key": ["KEY1", "KEY2", "KEY3"],
            }
        )

    def test_writes_valid_rows(self, lancedb_dir: str) -> None:
        db = lancedb.connect(lancedb_dir)
        df = self._make_flat_df()
        written, skipped = _write_to_lancedb(df, db, cpu_count=1)
        assert written == 2
        assert skipped == 1

    def test_skips_invalid_smiles(self, lancedb_dir: str) -> None:
        db = lancedb.connect(lancedb_dir)
        df = self._make_flat_df()
        _, skipped = _write_to_lancedb(df, db, cpu_count=1)
        assert skipped == 1

    def test_creates_compounds_table(self, lancedb_dir: str) -> None:
        db = lancedb.connect(lancedb_dir)
        _write_to_lancedb(self._make_flat_df(), db, cpu_count=1)
        assert COMPOUNDS_TABLE in db.list_tables().tables

    def test_vector_column_has_correct_length(self, lancedb_dir: str) -> None:
        db = lancedb.connect(lancedb_dir)
        _write_to_lancedb(self._make_flat_df(), db, cpu_count=1)
        table = db.open_table(COMPOUNDS_TABLE)
        row = table.to_pandas().iloc[0]
        assert len(row["vector"]) == MORGAN_BITS

    def test_returns_zero_written_when_all_smiles_invalid(self, lancedb_dir: str) -> None:
        db = lancedb.connect(lancedb_dir)
        df = pl.DataFrame(
            {
                "molregno": [1],
                "chembl_id": ["CHEMBL1"],
                "canonical_smiles": [INVALID_SMILES],
                "standard_inchi_key": ["KEY1"],
            }
        )
        written, skipped = _write_to_lancedb(df, db, cpu_count=1)
        assert written == 0
        assert skipped == 1

    def test_creates_scalar_indices(self, lancedb_dir: str) -> None:
        db = lancedb.connect(lancedb_dir)
        df = self._make_flat_df()
        _write_to_lancedb(df, db, cpu_count=1)
        table = db.open_table(COMPOUNDS_TABLE)
        index_names = [idx.name for idx in table.list_indices()]
        assert any("chembl_id" in n for n in index_names)
        assert any("standard_inchi_key" in n for n in index_names)


# ── ingest_compounds_to_lancedb ───────────────────────────────────────────────


class TestIngestCompoundsToLancedb:
    def test_creates_versioned_lancedb_dir(self, parquet_dir: str, lancedb_dir: str) -> None:
        ingest_compounds_to_lancedb(parquet_dir=parquet_dir, lancedb_dir=lancedb_dir)
        assert (Path(lancedb_dir) / "chembl_CHEMBL_36").exists()

    def test_compounds_table_exists_after_ingest(self, parquet_dir: str, lancedb_dir: str) -> None:
        ingest_compounds_to_lancedb(parquet_dir=parquet_dir, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(Path(lancedb_dir) / "chembl_CHEMBL_36"))
        assert COMPOUNDS_TABLE in db.list_tables().tables

    def test_ingest_is_idempotent(self, parquet_dir: str, lancedb_dir: str) -> None:
        ingest_compounds_to_lancedb(parquet_dir=parquet_dir, lancedb_dir=lancedb_dir)
        ingest_compounds_to_lancedb(parquet_dir=parquet_dir, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(Path(lancedb_dir) / "chembl_CHEMBL_36"))
        table = db.open_table(COMPOUNDS_TABLE)
        # Re-running should drop & recreate, so row count stays the same (2 valid SMILES)
        assert table.count_rows() == 2

    def test_correct_row_count(self, parquet_dir: str, lancedb_dir: str) -> None:
        ingest_compounds_to_lancedb(parquet_dir=parquet_dir, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(Path(lancedb_dir) / "chembl_CHEMBL_36"))
        table = db.open_table(COMPOUNDS_TABLE)
        assert table.count_rows() == 2

    def test_vector_search_returns_results(self, parquet_dir: str, lancedb_dir: str) -> None:
        ingest_compounds_to_lancedb(parquet_dir=parquet_dir, lancedb_dir=lancedb_dir)
        db = lancedb.connect(str(Path(lancedb_dir) / "chembl_CHEMBL_36"))
        table = db.open_table(COMPOUNDS_TABLE)
        query_fp = _smiles_to_fp(ASPIRIN_SMILES)
        assert query_fp is not None
        results = table.search(query_fp.tolist()).limit(1).to_pandas()
        assert len(results) == 1
        assert results.iloc[0]["chembl_id"] == "CHEMBL25"
