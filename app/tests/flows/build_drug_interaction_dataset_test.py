import json
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest

from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    _drug_name,
    _mol_lookup,
    _record_to_molregno,
    build_drug_interaction_dataset,
    generate_activity_qa,
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
    bad = pl.DataFrame({
        "molregno": [999], "warning_type": ["Withdrawal"],
        "warning_class": ["x"], "warning_country": ["USA"],
        "warning_year": [2000], "efo_term": [None],
        "efo_id": [None], "efo_id_for_warning_class": [None],
    })
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
    no_mw = pl.DataFrame({
        "molregno": [1], "mw_freebase": [None], "alogp": [1.0], "hba": [2],
        "hbd": [1], "psa": [60.0], "rtb": [2], "ro3_pass": [None],
        "num_ro5_violations": [0], "full_mwt": [None], "aromatic_rings": [1],
        "heavy_atoms": [10], "qed_weighted": [0.5], "full_molformula": [None],
        "np_likeness_score": [-1.0],
    })
    assert list(generate_physicochemical_qa(no_mw, molecule_dict)) == []


# ---------------------------------------------------------------------------
# generate_atc_classification_qa
# ---------------------------------------------------------------------------


def test_atc_qa_produces_pairs(atc_classification, molecule_atc_classification, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_atc_classification_qa,
    )
    pairs = list(generate_atc_classification_qa(
        atc_classification, molecule_atc_classification, molecule_dict
    ))
    assert len(pairs) > 0


def test_atc_qa_contains_who_name(atc_classification, molecule_atc_classification, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_atc_classification_qa,
    )
    pairs = list(generate_atc_classification_qa(
        atc_classification, molecule_atc_classification, molecule_dict
    ))
    texts = " ".join(p["text"] for p in pairs)
    assert "ACETYLSALICYLIC ACID" in texts or "N02BA01" in texts


def test_atc_qa_contains_code(atc_classification, molecule_atc_classification, molecule_dict):
    from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
        generate_atc_classification_qa,
    )
    pairs = list(generate_atc_classification_qa(
        atc_classification, molecule_atc_classification, molecule_dict
    ))
    texts = " ".join(p["text"] for p in pairs)
    assert "ATC" in texts


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
        full_count = sum(
            1 for _ in (full_out / "train.jsonl").read_text().splitlines()
        ) + sum(
            1 for _ in (full_out / "valid.jsonl").read_text().splitlines()
        )

        # Build with only the original 7 tables
        partial_dir = tmp_path / "partial_src"
        partial_dir.mkdir()
        for t in ["molecule_dictionary", "drug_mechanism", "drug_indication",
                  "metabolism", "compound_records", "activities", "target_dictionary"]:
            shutil.copy(full_parquet_dir / f"{t}.parquet", partial_dir / f"{t}.parquet")
        partial_out = tmp_path / "partial"
        build_drug_interaction_dataset(data_dir=partial_dir, output_dir=partial_out)
        partial_count = sum(
            1 for _ in (partial_out / "train.jsonl").read_text().splitlines()
        ) + sum(
            1 for _ in (partial_out / "valid.jsonl").read_text().splitlines()
        )

        assert full_count > partial_count
