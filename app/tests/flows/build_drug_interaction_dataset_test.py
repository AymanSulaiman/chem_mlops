import inspect
import json
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest

import app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset as dataset_module
from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    _drug_name,
    _mol_lookup,
    _record_to_molregno,
    build_drug_interaction_dataset,
    generate_activity_qa,
    generate_cyp_inhibition_qa,
    generate_ddi_qa,
    generate_indication_qa,
    generate_mechanism_qa,
    generate_metabolism_qa,
    write_jsonl_splits,
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
def cyp_assays():
    return pl.DataFrame(
        {
            "assay_id": [30, 40],
            "tid": [3, 4],
            "description": [
                "CYP3A4 inhibition assay in human liver microsomes",
                "CYP2D6 inhibition assay in human recombinant enzyme",
            ],
            "assay_type": ["B", "B"],
            "assay_organism": ["Homo sapiens", "Homo sapiens"],
            "assay_tissue": ["Liver", "Recombinant enzyme"],
            "assay_cell_type": [None, None],
        }
    )


@pytest.fixture
def cyp_inhibition_activities():
    return pl.DataFrame(
        {
            "activity_id": [300, 400],
            "assay_id": [30, 40],
            "molregno": [1, 2],
            "standard_type": ["IC50", "Ki"],
            "standard_relation": ["=", "="],
            "standard_value": [45.0, 6.0],
            "standard_units": ["nM", "uM"],
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


# ---------------------------------------------------------------------------
# generate_cyp_inhibition_qa
# ---------------------------------------------------------------------------


def test_cyp_inhibition_qa_produces_pairs(
    cyp_inhibition_activities, cyp_assays, target_dict_with_tid, molecule_dict
):
    pairs = list(
        generate_cyp_inhibition_qa(
            cyp_inhibition_activities, cyp_assays, target_dict_with_tid, molecule_dict
        )
    )
    assert len(pairs) > 0


def test_cyp_inhibition_qa_mentions_potency(
    cyp_inhibition_activities, cyp_assays, target_dict_with_tid, molecule_dict
):
    pairs = list(
        generate_cyp_inhibition_qa(
            cyp_inhibition_activities, cyp_assays, target_dict_with_tid, molecule_dict
        )
    )
    texts = " ".join(p["text"] for p in pairs)
    assert "strong inhibitor" in texts
    assert "moderate inhibitor" in texts
    assert "CYP3A4" in texts
    assert "CYP2D6" in texts


def test_cyp_inhibition_qa_weak_label(target_dict_with_tid, molecule_dict):
    cyp_assays = pl.DataFrame(
        {
            "assay_id": [50],
            "tid": [3],
            "description": ["CYP3A4 inhibition assay"],
            "assay_type": ["B"],
            "assay_organism": ["Homo sapiens"],
            "assay_tissue": [None],
            "assay_cell_type": [None],
        }
    )
    weak_activity = pl.DataFrame(
        {
            "activity_id": [500],
            "assay_id": [50],
            "molregno": [1],
            "standard_type": ["IC50"],
            "standard_relation": ["="],
            "standard_value": [20.0],
            "standard_units": ["uM"],
        }
    )
    texts = " ".join(
        p["text"]
        for p in generate_cyp_inhibition_qa(
            weak_activity, cyp_assays, target_dict_with_tid, molecule_dict
        )
    )
    assert "weak inhibitor" in texts


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
        p["text"].split("\n\n")[0] for p in pairs if "co-administered" in p["text"]
    ]
    assert len(drug_pair_questions) == len(set(drug_pair_questions))


def test_ddi_qa_has_no_pair_limit_parameter():
    assert "max_pairs" not in inspect.signature(generate_ddi_qa).parameters
    assert not hasattr(dataset_module, "MAX_DDI_PAIRS")


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


def test_activity_qa_returns_all_records(activities, molecule_dict):
    """All pchembl-valued rows should be yielded — no cap applied."""
    pairs = list(generate_activity_qa(activities, molecule_dict))
    assert len(pairs) >= 1


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
        build_drug_interaction_dataset(data_dir=parquet_data_dir, output_dir=output_dir)
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "valid.jsonl").exists()

    def test_all_records_are_valid_json(self, parquet_data_dir, tmp_path):
        output_dir = tmp_path / "output"
        build_drug_interaction_dataset(data_dir=parquet_data_dir, output_dir=output_dir)
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
            build_drug_interaction_dataset(data_dir=empty_dir, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# New generator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def drug_warning():
    return pl.DataFrame(
        {
            "molregno": [1, 1, 2],
            "warning_type": ["Black Box Warning", "Black Box Warning", "Withdrawal"],
            "warning_class": ["hepatotoxicity", "cardiac toxicity", "neurotoxicity"],
            "warning_country": ["USA", "EU", "USA"],
            "warning_year": [2005, 2010, 2015],
            "efo_term": [None, None, None],
            "efo_id": [None, None, None],
            "efo_id_for_warning_class": [None, None, None],
        }
    )


@pytest.fixture
def molecule_synonyms():
    return pl.DataFrame(
        {
            "molregno": [1, 1, 2, 2],
            "syn_type": ["TRADE_NAME", "INN", "TRADE_NAME", "USAN"],
            "molsyn_id": [1, 2, 3, 4],
            "synonyms": ["Bayer Aspirin", "acetylsalicylic acid", "Coumadin", "warfarin"],
        }
    )


@pytest.fixture
def compound_properties():
    return pl.DataFrame(
        {
            "molregno": [1, 2],
            "mw_freebase": [180.16, 308.33],
            "alogp": [1.19, 2.70],
            "hba": [4, 4],
            "hbd": [1, 1],
            "psa": [63.6, 73.3],
            "rtb": [2, 4],
            "ro3_pass": [None, None],
            "num_ro5_violations": [0, 0],
            "full_mwt": [180.16, 308.33],
            "aromatic_rings": [1, 2],
            "heavy_atoms": [13, 21],
            "qed_weighted": [0.55, 0.67],
            "full_molformula": ["C9H8O4", "C19H16O4"],
            "np_likeness_score": [-1.2, -0.8],
        }
    )


@pytest.fixture
def atc_classification():
    return pl.DataFrame(
        {
            "who_name": ["ACETYLSALICYLIC ACID", "WARFARIN"],
            "level1": ["N", "B"],
            "level2": ["N02", "B01"],
            "level3": ["N02B", "B01A"],
            "level4": ["N02BA", "B01AA"],
            "level5": ["N02BA01", "B01AA03"],
            "level1_description": ["NERVOUS SYSTEM", "BLOOD AND BLOOD FORMING ORGANS"],
            "level2_description": ["OTHER ANALGESICS AND ANTIPYRETICS", "ANTITHROMBOTIC AGENTS"],
            "level3_description": [None, None],
            "level4_description": ["Salicylic acid and derivatives", "Vitamin K antagonists"],
        }
    )


@pytest.fixture
def molecule_atc_classification():
    return pl.DataFrame(
        {
            "mol_atc_id": [1, 2],
            "level5": ["N02BA01", "B01AA03"],
            "molregno": [1, 2],
        }
    )


@pytest.fixture
def products():
    return pl.DataFrame(
        {
            "product_id": ["P001", "P002"],
            "trade_name": ["BAYER ASPIRIN", "COUMADIN"],
            "dosage_form": ["TABLET", "TABLET"],
            "route": ["ORAL", "ORAL"],
            "approval_date": ["1965-01-01", "1954-01-01"],
            "ad_type": [None, None],
            "oral": [1, 1],
            "topical": [0, 0],
            "parenteral": [0, 0],
            "black_box_warning": [0, 1],
            "applicant_full_name": ["Bayer", "Bristol-Myers Squibb"],
            "innovator_company": [1, 1],
            "nda_type": ["N", "N"],
        }
    )


@pytest.fixture
def formulations(compound_records):
    return pl.DataFrame(
        {
            "product_id": ["P001", "P002"],
            "ingredient": ["Aspirin", "Warfarin sodium"],
            "strength": ["325 mg", "5 mg"],
            "record_id": [101, 102],
            "molregno": [1, 2],
            "formulation_id": [1001, 1002],
        }
    )


# ---------------------------------------------------------------------------
# generate_drug_warning_qa
# ---------------------------------------------------------------------------


def test_drug_warning_qa_produces_pairs(drug_warning, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_drug_warning_qa,
    )

    pairs = list(generate_drug_warning_qa(drug_warning, molecule_dict))
    assert len(pairs) > 0


def test_drug_warning_qa_black_box_pair(drug_warning, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_drug_warning_qa,
    )

    pairs = list(generate_drug_warning_qa(drug_warning, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "Black Box Warning" in texts
    assert "Aspirin" in texts


def test_drug_warning_qa_skips_unknown_molregno(molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_drug_warning_qa,
    )

    bad = pl.DataFrame(
        {
            "molregno": [999],
            "warning_type": ["Withdrawal"],
            "warning_class": ["x"],
            "warning_country": ["USA"],
            "warning_year": [2000],
            "efo_term": [None],
            "efo_id": [None],
            "efo_id_for_warning_class": [None],
        }
    )
    assert list(generate_drug_warning_qa(bad, molecule_dict)) == []


# ---------------------------------------------------------------------------
# generate_synonym_qa
# ---------------------------------------------------------------------------


def test_synonym_qa_produces_pairs(molecule_synonyms, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_synonym_qa,
    )

    pairs = list(generate_synonym_qa(molecule_synonyms, molecule_dict))
    assert len(pairs) > 0


def test_synonym_qa_trade_name_pair(molecule_synonyms, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_synonym_qa,
    )

    pairs = list(generate_synonym_qa(molecule_synonyms, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "Bayer Aspirin" in texts
    assert "Coumadin" in texts


def test_synonym_qa_inn_pair(molecule_synonyms, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_synonym_qa,
    )

    pairs = list(generate_synonym_qa(molecule_synonyms, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "INN" in texts or "generic" in texts.lower()


# ---------------------------------------------------------------------------
# generate_physicochemical_qa
# ---------------------------------------------------------------------------


def test_physicochemical_qa_produces_pairs(compound_properties, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_physicochemical_qa,
    )

    pairs = list(generate_physicochemical_qa(compound_properties, molecule_dict))
    assert len(pairs) > 0


def test_physicochemical_qa_contains_mw(compound_properties, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_physicochemical_qa,
    )

    pairs = list(generate_physicochemical_qa(compound_properties, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "MW=" in texts
    assert "LogP=" in texts


def test_physicochemical_qa_skips_null_mw(molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_physicochemical_qa,
    )

    no_mw = pl.DataFrame(
        {
            "molregno": [1],
            "mw_freebase": [None],
            "alogp": [1.0],
            "hba": [2],
            "hbd": [1],
            "psa": [60.0],
            "rtb": [2],
            "ro3_pass": [None],
            "num_ro5_violations": [0],
            "full_mwt": [None],
            "aromatic_rings": [1],
            "heavy_atoms": [10],
            "qed_weighted": [0.5],
            "full_molformula": [None],
            "np_likeness_score": [-1.0],
        }
    )
    assert list(generate_physicochemical_qa(no_mw, molecule_dict)) == []


# ---------------------------------------------------------------------------
# generate_atc_classification_qa
# ---------------------------------------------------------------------------


def test_atc_qa_produces_pairs(atc_classification, molecule_atc_classification, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_atc_classification_qa,
    )

    pairs = list(
        generate_atc_classification_qa(
            atc_classification, molecule_atc_classification, molecule_dict
        )
    )
    assert len(pairs) > 0


def test_atc_qa_contains_who_name(atc_classification, molecule_atc_classification, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_atc_classification_qa,
    )

    pairs = list(
        generate_atc_classification_qa(
            atc_classification, molecule_atc_classification, molecule_dict
        )
    )
    texts = " ".join(p["text"] for p in pairs)
    assert "ACETYLSALICYLIC ACID" in texts or "N02BA01" in texts


def test_atc_qa_contains_code(atc_classification, molecule_atc_classification, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_atc_classification_qa,
    )

    pairs = list(
        generate_atc_classification_qa(
            atc_classification, molecule_atc_classification, molecule_dict
        )
    )
    texts = " ".join(p["text"] for p in pairs)
    assert "ATC" in texts


def test_assay_context_qa_has_no_pair_cap():
    assert not hasattr(dataset_module, "MAX_ASSAY_PAIRS")


# ---------------------------------------------------------------------------
# generate_approved_product_qa
# ---------------------------------------------------------------------------


def test_approved_product_qa_produces_pairs(formulations, products, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_approved_product_qa,
    )

    pairs = list(generate_approved_product_qa(formulations, products, molecule_dict))
    assert len(pairs) > 0


def test_approved_product_qa_trade_name(formulations, products, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_approved_product_qa,
    )

    pairs = list(generate_approved_product_qa(formulations, products, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "BAYER ASPIRIN" in texts or "COUMADIN" in texts


def test_approved_product_qa_black_box(formulations, products, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_approved_product_qa,
    )

    pairs = list(generate_approved_product_qa(formulations, products, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "Black Box" in texts
    assert "COUMADIN" in texts


# ---------------------------------------------------------------------------
# Extended integration test — all tables present
# ---------------------------------------------------------------------------


class TestBuildDrugInteractionDatasetExtended:
    @pytest.fixture
    def full_parquet_dir(
        self,
        tmp_path,
        molecule_dict,
        drug_mechanism,
        drug_indication,
        metabolism,
        compound_records,
        activities,
        target_dict,
        drug_warning,
        molecule_synonyms,
        compound_properties,
        atc_classification,
        molecule_atc_classification,
        formulations,
        products,
    ):
        for name, df in [
            ("molecule_dictionary", molecule_dict),
            ("drug_mechanism", drug_mechanism),
            ("drug_indication", drug_indication),
            ("metabolism", metabolism),
            ("compound_records", compound_records),
            ("activities", activities),
            ("target_dictionary", target_dict),
            ("drug_warning", drug_warning),
            ("molecule_synonyms", molecule_synonyms),
            ("compound_properties", compound_properties),
            ("atc_classification", atc_classification),
            ("molecule_atc_classification", molecule_atc_classification),
            ("formulations", formulations),
            ("products", products),
        ]:
            df.write_parquet(tmp_path / f"{name}.parquet")
        return tmp_path

    def test_all_generators_run(self, full_parquet_dir, tmp_path):
        output_dir = tmp_path / "output"
        build_drug_interaction_dataset(data_dir=full_parquet_dir, output_dir=output_dir)
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "valid.jsonl").exists()

    def test_more_pairs_with_all_tables(self, full_parquet_dir, tmp_path):
        """Dataset with all 14 tables should have more pairs than 7-table subset."""
        import shutil

        # Build with all tables
        full_out = tmp_path / "full"
        build_drug_interaction_dataset(data_dir=full_parquet_dir, output_dir=full_out)
        full_count = sum(1 for _ in (full_out / "train.jsonl").read_text().splitlines()) + sum(
            1 for _ in (full_out / "valid.jsonl").read_text().splitlines()
        )

        # Build with only the original 7 tables
        partial_dir = tmp_path / "partial_src"
        partial_dir.mkdir()
        for t in [
            "molecule_dictionary",
            "drug_mechanism",
            "drug_indication",
            "metabolism",
            "compound_records",
            "activities",
            "target_dictionary",
        ]:
            shutil.copy(full_parquet_dir / f"{t}.parquet", partial_dir / f"{t}.parquet")
        partial_out = tmp_path / "partial"
        build_drug_interaction_dataset(data_dir=partial_dir, output_dir=partial_out)
        partial_count = sum(
            1 for _ in (partial_out / "train.jsonl").read_text().splitlines()
        ) + sum(1 for _ in (partial_out / "valid.jsonl").read_text().splitlines())

        assert full_count > partial_count


# ---------------------------------------------------------------------------
# New fixtures for the 8 additional generators
# ---------------------------------------------------------------------------


@pytest.fixture
def docs():
    return pl.DataFrame(
        {
            "doc_id": [1, 2],
            "title": ["Aspirin inhibits COX-1", "Warfarin pharmacokinetics"],
            "abstract": [
                "This study demonstrates that aspirin irreversibly inhibits "
                "cyclooxygenase-1 and reduces thromboxane A2 synthesis.",
                "Warfarin exhibits stereoselective metabolism via CYP2C9 "
                "and has a narrow therapeutic index requiring careful monitoring.",
            ],
            "journal": ["J Pharmacol", "Clin Pharmacokinet"],
            "year": [2020, 2019],
            "pubmed_id": [12345678, 87654321],
        }
    )


@pytest.fixture
def assays():
    return pl.DataFrame(
        {
            "assay_id": [10, 20, 30, 40],
            "tid": [1, 2, 3, 4],
            "description": [
                "Inhibition of COX-1 in human platelets",
                "Anticoagulant activity in rat plasma",
                "CYP3A4 inhibition assay in human liver microsomes",
                "CYP2D6 inhibition assay in human recombinant enzyme",
            ],
            "assay_type": ["B", "F", "B", "B"],
            "assay_organism": ["Homo sapiens", "Rattus norvegicus", "Homo sapiens", "Homo sapiens"],
            "assay_tissue": ["Blood", "Plasma", "Liver", "Recombinant enzyme"],
            "assay_cell_type": [None, None, None, None],
        }
    )


@pytest.fixture
def activities_with_assay(activities):
    """Activities that include assay_id and activity_id columns."""
    return pl.DataFrame(
        {
            "molregno": [1, 2, 1, 2],
            "pchembl_value": [7.5, 5.2, None, None],
            "standard_type": ["IC50", "Ki", "IC50", "Ki"],
            "standard_relation": ["=", "=", "=", "="],
            "standard_value": [30.0, 600.0, 45.0, 6.0],
            "standard_units": ["nM", "nM", "nM", "uM"],
            "assay_id": [10, 20, 30, 40],
            "activity_id": [100, 200, 300, 400],
        }
    )


@pytest.fixture
def ligand_eff():
    return pl.DataFrame(
        {
            "activity_id": [100, 200],
            "le": [0.35, 0.22],
            "lle": [3.1, 1.8],
            "bei": [12.5, 8.7],
            "sei": [5.2, 3.9],
        }
    )


@pytest.fixture
def target_dict_with_tid():
    return pl.DataFrame(
        {
            "tid": [1, 2, 3, 4],
            "pref_name": [
                "Cyclooxygenase-1",
                "Vitamin K epoxide reductase",
                "CYP3A4",
                "CYP2D6",
            ],
            "chembl_id": ["CHEMBL_TGT_1", "CHEMBL_TGT_2", "CHEMBL_TGT_3", "CHEMBL_TGT_4"],
            "organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "target_type": [
                "SINGLE PROTEIN",
                "SINGLE PROTEIN",
                "SINGLE PROTEIN",
                "SINGLE PROTEIN",
            ],
        }
    )


@pytest.fixture
def component_sequences():
    return pl.DataFrame(
        {
            "component_id": [101, 102],
            "accession": ["P23219", "P56817"],
            "description": ["Prostaglandin G/H synthase 1", "Vitamin K epoxide reductase"],
            "organism": ["Homo sapiens", "Homo sapiens"],
            "sequence": ["MSELAAC", "MSWLKL"],
        }
    )


@pytest.fixture
def target_components():
    return pl.DataFrame(
        {
            "tid": [1, 2],
            "component_id": [101, 102],
        }
    )


@pytest.fixture
def protein_classification():
    return pl.DataFrame(
        {
            "protein_class_id": [1, 2],
            "pref_name": ["Enzyme", "Reductase"],
            "short_name": ["Enzyme", "Reductase"],
            "protein_class_desc": ["Enzyme superfamily", "Reductase family"],
            "definition": ["Catalyses chemical reactions.", "Reduces substrates."],
        }
    )


@pytest.fixture
def component_class():
    return pl.DataFrame(
        {
            "component_id": [101, 102],
            "protein_class_id": [1, 2],
        }
    )


@pytest.fixture
def biotherapeutics():
    return pl.DataFrame(
        {
            "molregno": [1, 2],
            "description": ["monoclonal antibody", None],
        }
    )


@pytest.fixture
def target_relations():
    return pl.DataFrame(
        {
            "tid": [1],
            "related_tid": [2],
            "relationship": ["SUBSET OF"],
        }
    )


# ---------------------------------------------------------------------------
# generate_literature_qa
# ---------------------------------------------------------------------------


def test_literature_qa_produces_pairs(docs):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_literature_qa,
    )

    pairs = list(generate_literature_qa(docs))
    assert len(pairs) >= 2


def test_literature_qa_contains_title_and_abstract(docs):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_literature_qa,
    )

    pairs = list(generate_literature_qa(docs))
    texts = " ".join(p["text"] for p in pairs)
    assert "Aspirin inhibits COX-1" in texts
    assert "cyclooxygenase-1" in texts


def test_literature_qa_skips_short_abstract():
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_literature_qa,
    )

    df = pl.DataFrame(
        {
            "doc_id": [1],
            "title": ["Short paper"],
            "abstract": ["Too short."],
            "journal": ["J Test"],
            "year": [2020],
            "pubmed_id": [None],
        }
    )
    assert list(generate_literature_qa(df)) == []


# ---------------------------------------------------------------------------
# generate_assay_context_qa
# ---------------------------------------------------------------------------


def test_assay_context_qa_produces_pairs(assays, activities_with_assay, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_assay_context_qa,
    )

    pairs = list(generate_assay_context_qa(assays, activities_with_assay, molecule_dict))
    assert len(pairs) > 0


def test_assay_context_qa_contains_drug_name(assays, activities_with_assay, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_assay_context_qa,
    )

    pairs = list(generate_assay_context_qa(assays, activities_with_assay, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "Aspirin" in texts or "Warfarin" in texts


def test_assay_context_qa_no_pairs_without_description(activities_with_assay, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_assay_context_qa,
    )

    empty_assays = pl.DataFrame(
        {
            "assay_id": [10],
            "description": [None],
            "assay_type": ["B"],
            "assay_organism": ["Homo sapiens"],
            "assay_tissue": [None],
            "assay_cell_type": [None],
        }
    )
    pairs = list(generate_assay_context_qa(empty_assays, activities_with_assay, molecule_dict))
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# generate_ligand_efficiency_qa
# ---------------------------------------------------------------------------


def test_ligand_efficiency_qa_produces_pairs(ligand_eff, activities_with_assay, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_ligand_efficiency_qa,
    )

    pairs = list(generate_ligand_efficiency_qa(ligand_eff, activities_with_assay, molecule_dict))
    assert len(pairs) > 0


def test_ligand_efficiency_qa_contains_le_value(ligand_eff, activities_with_assay, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_ligand_efficiency_qa,
    )

    pairs = list(generate_ligand_efficiency_qa(ligand_eff, activities_with_assay, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "LE=" in texts
    assert "good ligand efficiency" in texts or "moderate ligand efficiency" in texts


def test_ligand_efficiency_qa_quality_labels(ligand_eff, activities_with_assay, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_ligand_efficiency_qa,
    )

    pairs = list(generate_ligand_efficiency_qa(ligand_eff, activities_with_assay, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    # 0.35 → good, 0.22 → moderate
    assert "good" in texts
    assert "moderate" in texts


# ---------------------------------------------------------------------------
# generate_target_sequence_qa
# ---------------------------------------------------------------------------


def test_target_sequence_qa_produces_pairs(
    component_sequences, target_components, target_dict_with_tid
):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_target_sequence_qa,
    )

    pairs = list(
        generate_target_sequence_qa(component_sequences, target_components, target_dict_with_tid)
    )
    assert len(pairs) > 0


def test_target_sequence_qa_contains_accession(
    component_sequences, target_components, target_dict_with_tid
):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_target_sequence_qa,
    )

    pairs = list(
        generate_target_sequence_qa(component_sequences, target_components, target_dict_with_tid)
    )
    texts = " ".join(p["text"] for p in pairs)
    assert "P23219" in texts or "P56817" in texts


# ---------------------------------------------------------------------------
# generate_protein_family_qa
# ---------------------------------------------------------------------------


def test_protein_family_qa_produces_pairs(
    protein_classification, component_class, target_components, target_dict_with_tid
):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_protein_family_qa,
    )

    pairs = list(
        generate_protein_family_qa(
            protein_classification, component_class, target_components, target_dict_with_tid
        )
    )
    assert len(pairs) > 0


def test_protein_family_qa_contains_class_name(
    protein_classification, component_class, target_components, target_dict_with_tid
):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_protein_family_qa,
    )

    pairs = list(
        generate_protein_family_qa(
            protein_classification, component_class, target_components, target_dict_with_tid
        )
    )
    texts = " ".join(p["text"] for p in pairs)
    assert "Enzyme" in texts or "Reductase" in texts


# ---------------------------------------------------------------------------
# generate_biotherapeutic_qa
# ---------------------------------------------------------------------------


def test_biotherapeutic_qa_produces_pairs(biotherapeutics, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_biotherapeutic_qa,
    )

    pairs = list(generate_biotherapeutic_qa(biotherapeutics, molecule_dict))
    assert len(pairs) > 0


def test_biotherapeutic_qa_contains_biologic_label(biotherapeutics, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_biotherapeutic_qa,
    )

    pairs = list(generate_biotherapeutic_qa(biotherapeutics, molecule_dict))
    texts = " ".join(p["text"] for p in pairs)
    assert "biologic" in texts


def test_biotherapeutic_qa_skips_null_molregno(molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_biotherapeutic_qa,
    )

    df = pl.DataFrame({"molregno": [None], "description": ["antibody"]})
    assert list(generate_biotherapeutic_qa(df, molecule_dict)) == []


# ---------------------------------------------------------------------------
# generate_target_relations_qa
# ---------------------------------------------------------------------------


def test_target_relations_qa_produces_pairs(target_relations, target_dict_with_tid):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_target_relations_qa,
    )

    pairs = list(generate_target_relations_qa(target_relations, target_dict_with_tid))
    assert len(pairs) > 0


def test_target_relations_qa_subset_of(target_relations, target_dict_with_tid):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_target_relations_qa,
    )

    pairs = list(generate_target_relations_qa(target_relations, target_dict_with_tid))
    texts = " ".join(p["text"] for p in pairs)
    assert "subset" in texts.lower() or "subtype" in texts.lower()


def test_target_relations_qa_skips_unknown_tid(target_dict_with_tid):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_target_relations_qa,
    )

    bad = pl.DataFrame(
        {
            "tid": [999],
            "related_tid": [888],
            "relationship": ["SUBSET OF"],
        }
    )
    assert list(generate_target_relations_qa(bad, target_dict_with_tid)) == []


# ---------------------------------------------------------------------------
# Extended integration test — all 17 generators
# ---------------------------------------------------------------------------


class TestBuildDrugInteractionDatasetFull:
    @pytest.fixture
    def all_tables_dir(
        self,
        tmp_path,
        molecule_dict,
        drug_mechanism,
        drug_indication,
        metabolism,
        compound_records,
        activities_with_assay,
        target_dict_with_tid,
        drug_warning,
        molecule_synonyms,
        compound_properties,
        atc_classification,
        molecule_atc_classification,
        formulations,
        products,
        docs,
        assays,
        ligand_eff,
        component_sequences,
        target_components,
        protein_classification,
        component_class,
        biotherapeutics,
        target_relations,
    ):
        # target_dict_with_tid already has tid column needed for new generators
        for name, df in [
            ("molecule_dictionary", molecule_dict),
            ("drug_mechanism", drug_mechanism),
            ("drug_indication", drug_indication),
            ("metabolism", metabolism),
            ("compound_records", compound_records),
            ("activities", activities_with_assay),
            ("target_dictionary", target_dict_with_tid),
            ("drug_warning", drug_warning),
            ("molecule_synonyms", molecule_synonyms),
            ("compound_properties", compound_properties),
            ("atc_classification", atc_classification),
            ("molecule_atc_classification", molecule_atc_classification),
            ("formulations", formulations),
            ("products", products),
            ("docs", docs),
            ("assays", assays),
            ("ligand_eff", ligand_eff),
            ("component_sequences", component_sequences),
            ("target_components", target_components),
            ("protein_classification", protein_classification),
            ("component_class", component_class),
            ("biotherapeutics", biotherapeutics),
            ("target_relations", target_relations),
        ]:
            df.write_parquet(tmp_path / f"{name}.parquet")
        return tmp_path

    def test_full_pipeline_runs(self, all_tables_dir, tmp_path):
        out = tmp_path / "out"
        build_drug_interaction_dataset(data_dir=all_tables_dir, output_dir=out)
        assert (out / "train.jsonl").exists()
        assert (out / "valid.jsonl").exists()

    def test_full_pipeline_more_pairs_than_partial(self, all_tables_dir, tmp_path):
        full_out = tmp_path / "full"
        build_drug_interaction_dataset(data_dir=all_tables_dir, output_dir=full_out)
        full_count = sum(1 for _ in (full_out / "train.jsonl").read_text().splitlines()) + sum(
            1 for _ in (full_out / "valid.jsonl").read_text().splitlines()
        )
        assert full_count > 10
