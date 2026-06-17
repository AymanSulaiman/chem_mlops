import polars as pl
import pytest

from app.scripts.flows.llm_finetuning_data.build_finetune_dataset import (
    filter_activities,
    join_tables,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def molecule_dict():
    return pl.DataFrame(
        {
            "molregno": [1, 2, 3],
            "pref_name": ["Aspirin", "Warfarin", "Metformin"],
            "chembl_id": ["CHEMBL25", "CHEMBL1378", "CHEMBL1431"],
            "max_phase": [4, 4, 4],
            "therapeutic_flag": [1, 1, 1],
            "molecule_type": ["Small molecule", "Small molecule", "Small molecule"],
            "structure_type": ["MOL", "MOL", "MOL"],
            "natural_product": [0, 0, 0],
            "first_in_class": [0, 0, 0],
            "black_box_warning": [0, 1, 0],
        }
    )


@pytest.fixture
def compound_structures():
    return pl.DataFrame(
        {
            "molregno": [1, 2],  # molregno=3 intentionally missing (left join test)
            "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)c1ccc(cc1)OCCCN2CCOCC2"],
            "standard_inchi": ["InChI=1S/C9H8O4/...", "InChI=1S/C19H16O4/..."],
            "standard_inchi_key": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "PJVWKTKQMONHTI-UHFFFAOYSA-N"],
        }
    )


@pytest.fixture
def activities():
    return pl.DataFrame(
        {
            "activity_id": [10, 20, 30, 40],
            "assay_id": [100, 200, 300, 400],
            "doc_id": [1, 2, 3, 4],
            "record_id": [1, 2, 3, 4],
            "molregno": [1, 2, 3, 1],
            "standard_relation": ["=", "=", "=", "<"],
            "standard_value": [1.0, None, 500.0, 2.0],
            "standard_units": ["nM", None, "nM", "nM"],
            "standard_type": ["IC50", "Ki", "IC50", "IC50"],
            "pchembl_value": [9.0, 7.5, None, 8.7],
            "data_validity_comment": [None, None, None, None],
            "activity_comment": [None, None, None, None],
        }
    )


@pytest.fixture
def activities_all_null():
    """All rows have both standard_value and pchembl_value as null — should all be filtered."""
    return pl.DataFrame(
        {
            "activity_id": [1, 2],
            "assay_id": [10, 20],
            "doc_id": [1, 2],
            "record_id": [1, 2],
            "molregno": [1, 2],
            "standard_relation": [None, None],
            "standard_value": [None, None],
            "standard_units": [None, None],
            "standard_type": [None, None],
            "pchembl_value": [None, None],
            "data_validity_comment": [None, None],
            "activity_comment": [None, None],
        }
    )


# ---------------------------------------------------------------------------
# filter_activities
# ---------------------------------------------------------------------------


def test_filter_keeps_rows_with_pchembl(activities):
    result = filter_activities(activities)
    assert result.filter(pl.col("pchembl_value").is_not_null()).shape[0] > 0


def test_filter_keeps_rows_with_standard_value(activities):
    result = filter_activities(activities)
    # molregno=3 has standard_value=500.0 but pchembl_value=null — must be kept
    assert result.filter(pl.col("molregno") == 3).shape[0] == 1


def test_filter_drops_rows_with_both_null(activities_all_null):
    result = filter_activities(activities_all_null)
    assert result.shape[0] == 0


def test_filter_selects_expected_columns(activities):
    result = filter_activities(activities)
    expected = {
        "activity_id",
        "assay_id",
        "doc_id",
        "record_id",
        "molregno",
        "standard_relation",
        "standard_value",
        "standard_units",
        "standard_type",
        "pchembl_value",
        "data_validity_comment",
        "activity_comment",
    }
    assert set(result.columns) == expected


def test_filter_preserves_row_count_when_all_valid(activities):
    # All 4 rows have at least one of pchembl_value or standard_value
    result = filter_activities(activities)
    assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# join_tables
# ---------------------------------------------------------------------------


def test_join_returns_dataframe(compound_structures, activities, molecule_dict):
    result = join_tables(compound_structures, activities, molecule_dict)
    assert isinstance(result, pl.DataFrame)


def test_join_inner_on_molregno(compound_structures, activities, molecule_dict):
    # All 4 activity rows have molregno in molecule_dict (1, 2, 3) — all kept
    result = join_tables(compound_structures, activities, molecule_dict)
    assert result.shape[0] == 4


def test_join_includes_mol_columns(compound_structures, activities, molecule_dict):
    result = join_tables(compound_structures, activities, molecule_dict)
    assert "pref_name" in result.columns
    assert "chembl_id" in result.columns
    assert "black_box_warning" in result.columns


def test_join_includes_structure_columns(compound_structures, activities, molecule_dict):
    result = join_tables(compound_structures, activities, molecule_dict)
    assert "canonical_smiles" in result.columns
    assert "standard_inchi_key" in result.columns


def test_join_left_join_for_structures(compound_structures, activities, molecule_dict):
    # molregno=3 has no structure — row should still be present with null smiles
    result = join_tables(compound_structures, activities, molecule_dict)
    row = result.filter(pl.col("molregno") == 3)
    assert row.shape[0] == 1
    assert row["canonical_smiles"][0] is None


def test_join_drops_molregno_not_in_molecule_dict(compound_structures, molecule_dict):
    # Activity row with unknown molregno should be dropped (inner join)
    acts_with_unknown = pl.DataFrame(
        {
            "activity_id": [99],
            "assay_id": [999],
            "doc_id": [9],
            "record_id": [9],
            "molregno": [999],
            "standard_relation": ["="],
            "standard_value": [1.0],
            "standard_units": ["nM"],
            "standard_type": ["IC50"],
            "pchembl_value": [7.0],
            "data_validity_comment": [None],
            "activity_comment": [None],
        }
    )
    result = join_tables(compound_structures, acts_with_unknown, molecule_dict)
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# create_finetuning_dataset (integration via monkeypatching)
# ---------------------------------------------------------------------------


def test_create_finetuning_dataset_writes_parquet(
    tmp_path, monkeypatch, compound_structures, activities, molecule_dict
):
    import app.scripts.flows.llm_finetuning_data.build_finetune_dataset as mod

    output_path = tmp_path / "dataset.parquet"
    monkeypatch.setattr(mod, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(
        mod, "load_tables", lambda: (compound_structures, activities, molecule_dict)
    )

    result = mod.create_finetuning_dataset()

    assert output_path.exists()
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] > 0


def test_create_finetuning_dataset_output_matches_join(
    tmp_path, monkeypatch, compound_structures, activities, molecule_dict
):
    import app.scripts.flows.llm_finetuning_data.build_finetune_dataset as mod

    monkeypatch.setattr(mod, "OUTPUT_PATH", tmp_path / "out.parquet")
    monkeypatch.setattr(
        mod, "load_tables", lambda: (compound_structures, activities, molecule_dict)
    )

    result = mod.create_finetuning_dataset()
    on_disk = pl.read_parquet(tmp_path / "out.parquet")

    assert result.shape == on_disk.shape
