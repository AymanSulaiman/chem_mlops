import json
import pytest
import polars as pl
from pathlib import Path
import tempfile
import shutil

from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    _drug_name,
    _mol_lookup,
    _record_to_molregno,
    generate_mechanism_qa,
    generate_indication_qa,
    generate_metabolism_qa,
    generate_ddi_qa,
    generate_activity_qa,
    write_jsonl_splits,
    build_drug_interaction_dataset,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def molecule_dict():
    return pl.DataFrame(
        {
            "molregno": [1, 2, 3],
            "pref_name": ["Aspirin", "Warfarin", None],
            "chembl_id": ["CHEMBL25", "CHEMBL1378", "CHEMBL999"],
            "max_phase": [4, 4, 2],
        }
    )


@pytest.fixture
def target_dict():
    return pl.DataFrame(
        {
            "chembl_id": ["CHEMBL_TGT_1", "CHEMBL_TGT_2"],
            "pref_name": ["Cyclooxygenase-1", "Vitamin K epoxide reductase"],
            "target_type": ["SINGLE PROTEIN", "SINGLE PROTEIN"],
        }
    )


@pytest.fixture
def drug_mechanism(target_dict):
    return pl.DataFrame(
        {
            "molregno": [1, 2],
            "mechanism_of_action": [
                "Cyclooxygenase inhibitor",
                "Vitamin K antagonist",
            ],
            "target_chembl_id": ["CHEMBL_TGT_1", "CHEMBL_TGT_2"],
            "action_type": ["INHIBITOR", "INHIBITOR"],
        }
    )


@pytest.fixture
def drug_indication():
    return pl.DataFrame(
        {
            "molregno": [1, 1, 2],
            "mesh_heading": ["Pain", "Fever", "Atrial Fibrillation"],
            "efo_term": [None, None, None],
            "max_phase_for_ind": [4, 4, 4],
        }
    )


@pytest.fixture
def compound_records():
    return pl.DataFrame(
        {
            "record_id": [101, 102, 103],
            "molregno": [1, 2, 3],
        }
    )


@pytest.fixture
def metabolism(compound_records):
    return pl.DataFrame(
        {
            "substrate_record_id": [101, 102, 101],
            "drug_record_id": [101, 102, 101],
            "enzyme_name": ["CYP2C9", "CYP2C9", "CYP3A4"],
        }
    )


@pytest.fixture
def activities():
    return pl.DataFrame(
        {
            "molregno": [1, 2],
            "pchembl_value": [7.5, 5.2],
            "standard_type": ["IC50", "Ki"],
            "standard_value": [30.0, 600.0],
            "standard_units": ["nM", "nM"],
        }
    )


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_drug_name_prefers_pref_name():
    assert _drug_name({"pref_name": "Aspirin", "chembl_id": "CHEMBL25"}) == "Aspirin"


def test_drug_name_falls_back_to_chembl_id():
    assert _drug_name({"pref_name": None, "chembl_id": "CHEMBL25"}) == "CHEMBL25"


def test_drug_name_unknown():
    assert _drug_name({}) == "Unknown"


def test_mol_lookup_keys(molecule_dict):
    lookup = _mol_lookup(molecule_dict)
    assert set(lookup.keys()) == {1, 2, 3}
    assert lookup[1]["pref_name"] == "Aspirin"


def test_record_to_molregno(compound_records):
    mapping = _record_to_molregno(compound_records)
    assert mapping[101] == 1
    assert mapping[102] == 2


# ---------------------------------------------------------------------------
# generate_mechanism_qa
# ---------------------------------------------------------------------------


def test_mechanism_qa_produces_pairs(drug_mechanism, molecule_dict, target_dict):
    pairs = list(generate_mechanism_qa(drug_mechanism, molecule_dict, target_dict))
    assert len(pairs) > 0


def test_mechanism_qa_contains_drug_name(drug_mechanism, molecule_dict, target_dict):
    pairs = list(generate_mechanism_qa(drug_mechanism, molecule_dict, target_dict))
    texts = [p["text"] for p in pairs]
    assert any("Aspirin" in t for t in texts)
    assert any("Warfarin" in t for t in texts)


def test_mechanism_qa_contains_target(drug_mechanism, molecule_dict, target_dict):
    pairs = list(generate_mechanism_qa(drug_mechanism, molecule_dict, target_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "Cyclooxygenase-1" in texts


def test_mechanism_qa_skips_unknown_molregno(molecule_dict, target_dict):
    bad_mech = pl.DataFrame(
        {
            "molregno": [999],
            "mechanism_of_action": ["Unknown inhibitor"],
            "target_chembl_id": ["CHEMBL_TGT_1"],
            "action_type": ["INHIBITOR"],
        }
    )
    pairs = list(generate_mechanism_qa(bad_mech, molecule_dict, target_dict))
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# generate_indication_qa
# ---------------------------------------------------------------------------


def test_indication_qa_produces_pairs(drug_indication, molecule_dict):
    pairs = list(generate_indication_qa(drug_indication, molecule_dict))
    assert len(pairs) > 0


def test_indication_qa_groups_by_drug(drug_indication, molecule_dict):
    pairs = list(generate_indication_qa(drug_indication, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    # Aspirin should appear with both indications
    assert "Aspirin" in texts
    assert "Pain" in texts
    assert "Fever" in texts


def test_indication_qa_reverse_lookup(drug_indication, molecule_dict):
    pairs = list(generate_indication_qa(drug_indication, molecule_dict))
    # Should also generate "What drugs treat X?" pairs
    texts = " ".join(p["text"] for p in pairs)
    assert "What drugs are used to treat" in texts


# ---------------------------------------------------------------------------
# generate_metabolism_qa
# ---------------------------------------------------------------------------


def test_metabolism_qa_produces_pairs(metabolism, molecule_dict, compound_records):
    pairs = list(generate_metabolism_qa(metabolism, molecule_dict, compound_records))
    assert len(pairs) > 0


def test_metabolism_qa_cyp_specific_pair(metabolism, molecule_dict, compound_records):
    pairs = list(generate_metabolism_qa(metabolism, molecule_dict, compound_records))
    texts = " ".join(p["text"] for p in pairs)
    assert "CYP" in texts
    assert "CYP2C9" in texts


def test_metabolism_qa_no_enzyme_skipped(molecule_dict, compound_records):
    empty_met = pl.DataFrame(
        {"substrate_record_id": [101], "drug_record_id": [101], "enzyme_name": [None]}
    )
    pairs = list(generate_metabolism_qa(empty_met, molecule_dict, compound_records))
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# generate_ddi_qa
# ---------------------------------------------------------------------------


def test_ddi_qa_generates_pairs(metabolism, molecule_dict, compound_records):
    pairs = list(generate_ddi_qa(metabolism, molecule_dict, compound_records))
    assert len(pairs) > 0


def test_ddi_qa_mentions_shared_enzyme(metabolism, molecule_dict, compound_records):
    pairs = list(generate_ddi_qa(metabolism, molecule_dict, compound_records))
    texts = " ".join(p["text"] for p in pairs)
    assert "CYP2C9" in texts


def test_ddi_qa_no_duplicate_pairs(metabolism, molecule_dict, compound_records):
    pairs = list(generate_ddi_qa(metabolism, molecule_dict, compound_records))
    # Each drug pair should appear at most once across the two pair types per enzyme
    drug_pair_questions = [
        p["text"].split("\n\n")[0]
        for p in pairs
        if "co-administered" in p["text"]
    ]
    assert len(drug_pair_questions) == len(set(drug_pair_questions))


def test_ddi_qa_max_pairs_respected(metabolism, molecule_dict, compound_records):
    pairs = list(
        generate_ddi_qa(metabolism, molecule_dict, compound_records, max_pairs=1)
    )
    # max_pairs=1 yields at most 2 records (2 QA variants per pair)
    assert len(pairs) <= 2


def test_ddi_qa_only_named_drugs(metabolism, molecule_dict, compound_records):
    pairs = list(generate_ddi_qa(metabolism, molecule_dict, compound_records))
    texts = " ".join(p["text"] for p in pairs)
    # molregno=3 has no pref_name so CHEMBL999 / None should not appear as a drug name
    assert "CHEMBL999" not in texts


# ---------------------------------------------------------------------------
# generate_activity_qa
# ---------------------------------------------------------------------------


def test_activity_qa_produces_pairs(activities, molecule_dict):
    pairs = list(generate_activity_qa(activities, molecule_dict))
    assert len(pairs) > 0


def test_activity_qa_high_potency_label(activities, molecule_dict):
    pairs = list(generate_activity_qa(activities, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "high potency" in texts  # pChEMBL 7.5 >= 7


def test_activity_qa_moderate_potency_label(activities, molecule_dict):
    pairs = list(generate_activity_qa(activities, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "moderate potency" in texts  # pChEMBL 5.2 in [5, 7)


def test_activity_qa_max_records(activities, molecule_dict):
    pairs = list(generate_activity_qa(activities, molecule_dict, max_records=1))
    assert len(pairs) == 1


# ---------------------------------------------------------------------------
# write_jsonl_splits
# ---------------------------------------------------------------------------


def test_write_jsonl_splits_creates_files(temp_dir):
    records = [{"text": f"pair {i}"} for i in range(10)]
    n_train, n_valid = write_jsonl_splits(records, temp_dir, train_ratio=0.8, seed=0)
    assert (temp_dir / "train.jsonl").exists()
    assert (temp_dir / "valid.jsonl").exists()
    assert n_train == 8
    assert n_valid == 2


def test_write_jsonl_splits_valid_json(temp_dir):
    records = [{"text": f"Question: A? Answer: B {i}"} for i in range(20)]
    write_jsonl_splits(records, temp_dir)
    for line in (temp_dir / "train.jsonl").read_text().splitlines():
        obj = json.loads(line)
        assert "text" in obj


def test_write_jsonl_splits_total_count(temp_dir):
    records = [{"text": str(i)} for i in range(100)]
    n_train, n_valid = write_jsonl_splits(records, temp_dir)
    assert n_train + n_valid == 100


# ---------------------------------------------------------------------------
# build_drug_interaction_dataset (integration)
# ---------------------------------------------------------------------------


class TestBuildDrugInteractionDataset:
    @pytest.fixture
    def parquet_data_dir(
        self,
        tmp_path,
        molecule_dict,
        drug_mechanism,
        drug_indication,
        metabolism,
        compound_records,
        activities,
        target_dict,
    ):
        molecule_dict.write_parquet(tmp_path / "molecule_dictionary.parquet")
        drug_mechanism.write_parquet(tmp_path / "drug_mechanism.parquet")
        drug_indication.write_parquet(tmp_path / "drug_indication.parquet")
        metabolism.write_parquet(tmp_path / "metabolism.parquet")
        compound_records.write_parquet(tmp_path / "compound_records.parquet")
        activities.write_parquet(tmp_path / "activities.parquet")
        target_dict.write_parquet(tmp_path / "target_dictionary.parquet")
        return tmp_path

    def test_creates_train_and_valid(self, parquet_data_dir, tmp_path):
        output_dir = tmp_path / "output"
        build_drug_interaction_dataset(
            data_dir=parquet_data_dir, output_dir=output_dir
        )
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "valid.jsonl").exists()

    def test_all_records_are_valid_json(self, parquet_data_dir, tmp_path):
        output_dir = tmp_path / "output"
        build_drug_interaction_dataset(
            data_dir=parquet_data_dir, output_dir=output_dir
        )
        for fname in ("train.jsonl", "valid.jsonl"):
            for line in (output_dir / fname).read_text().splitlines():
                obj = json.loads(line)
                assert "text" in obj
                assert "### Question" in obj["text"]
                assert "### Answer" in obj["text"]

    def test_missing_molecule_dict_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        # Write a dummy parquet so ChemblDataLoader doesn't raise FileNotFoundError
        pl.DataFrame({"x": [1]}).write_parquet(empty_dir / "dummy.parquet")
        with pytest.raises(RuntimeError, match="molecule_dictionary"):
            build_drug_interaction_dataset(
                data_dir=empty_dir, output_dir=tmp_path / "out"
            )
