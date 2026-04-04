"""
Build a QA-formatted JSONL dataset for finetuning a drug-interaction chatbot.

Pulls six ChEMBL tables beyond the basic activities/structures used elsewhere
and emits four categories of training pair:

  1. Mechanism of action   — "What does {drug} target?"
  2. Therapeutic indication — "What is {drug} indicated for?"
  3. Metabolic pathways    — "How is {drug} metabolised?" / CYP-specific
  4. Drug-drug interactions — pairs inferred from shared CYP substrates
  5. Bioactivity potency   — pChEMBL-based potency statements

Output: data/llm_finetune/train.jsonl  (90 %)
        data/llm_finetune/valid.jsonl  (10 %)
"""

import json
import random
from pathlib import Path
from typing import Iterator

import polars as pl

from app.scripts.load_data.load_data import ChemblDataLoader

DATA_DIR = Path("data/chembl_transform")
OUTPUT_DIR = Path("data/llm_finetune")

TRAIN_RATIO = 0.9
RANDOM_SEED = 42
MAX_DDI_PAIRS = 5_000
MAX_ACTIVITY_RECORDS = 20_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drug_name(mol: dict) -> str:
    return mol.get("pref_name") or mol.get("chembl_id") or "Unknown"


def _mol_lookup(molecule_dict: pl.DataFrame) -> dict[int, dict]:
    return {
        row["molregno"]: row
        for row in molecule_dict.select(
            ["molregno", "pref_name", "chembl_id", "max_phase"]
        ).to_dicts()
    }


def _record_to_molregno(compound_records: pl.DataFrame) -> dict[int, int]:
    """Map compound_records.record_id -> molecule_dictionary.molregno."""
    return {
        row["record_id"]: row["molregno"]
        for row in compound_records.select(["record_id", "molregno"]).to_dicts()
    }


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

_REQUIRED_TABLES = {
    "molecule_dictionary",
    "drug_mechanism",
    "drug_indication",
    "metabolism",
    "target_dictionary",
    "compound_records",
    "activities",
}


def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pl.DataFrame]:
    loader = ChemblDataLoader(data_dir=data_dir)
    available = set(loader.list_tables())
    missing = _REQUIRED_TABLES - available
    if missing:
        print(f"  Warning: tables not found (skipping): {missing}")
    return {name: loader.load_table(name) for name in _REQUIRED_TABLES & available}


# ---------------------------------------------------------------------------
# QA generators
# ---------------------------------------------------------------------------


def generate_mechanism_qa(
    drug_mechanism: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    target_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Mechanism-of-action pairs from drug_mechanism + target_dictionary."""
    mols = _mol_lookup(molecule_dict)

    # target_dictionary uses chembl_id as the natural key referenced by drug_mechanism
    target_by_chembl_id = {
        row["chembl_id"]: row
        for row in target_dict.select(["chembl_id", "pref_name", "target_type"]).to_dicts()
    }

    for row in drug_mechanism.to_dicts():
        mol = mols.get(row.get("molregno"))
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        moa = (row.get("mechanism_of_action") or "").strip()
        action = (row.get("action_type") or "").strip().lower()
        target = target_by_chembl_id.get(row.get("target_chembl_id") or "", {})
        target_name = target.get("pref_name") or "an unspecified target"

        if not moa and not action:
            continue

        action_phrase = f"{action} of {target_name}" if action else f"modulator of {target_name}"

        yield {
            "text": (
                f"### Question\nWhat is the mechanism of action of {drug}?\n\n"
                f"### Answer\n{drug} ({chembl_id}) acts as a {action_phrase}. "
                f"{moa}"
            ).strip()
        }

        if target_name != "an unspecified target":
            yield {
                "text": (
                    f"### Question\nWhat does {drug} target?\n\n"
                    f"### Answer\n{drug} primarily targets {target_name}. "
                    f"It acts as a {action if action else 'modulator'} at this target."
                )
            }


def generate_indication_qa(
    drug_indication: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Therapeutic-indication pairs grouped by drug."""
    mols = _mol_lookup(molecule_dict)

    drug_indications: dict[int, list[str]] = {}
    for row in drug_indication.to_dicts():
        molregno = row.get("molregno")
        indication = row.get("mesh_heading") or row.get("efo_term")
        if molregno and indication:
            drug_indications.setdefault(molregno, []).append(indication)

    for molregno, indications in drug_indications.items():
        mol = mols.get(molregno)
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        max_phase = mol.get("max_phase")
        phase_str = f" (max clinical phase: {int(max_phase)})" if max_phase else ""
        unique_indications = list(dict.fromkeys(indications))[:6]
        indication_list = "; ".join(unique_indications)

        yield {
            "text": (
                f"### Question\nWhat is {drug} indicated for?\n\n"
                f"### Answer\n{drug} ({chembl_id}){phase_str} is indicated for: {indication_list}."
            )
        }

        for indication in unique_indications:
            yield {
                "text": (
                    f"### Question\nWhat drugs are used to treat {indication}?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) is an approved treatment for "
                    f"{indication}{phase_str}."
                )
            }


def generate_metabolism_qa(
    metabolism: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    compound_records: pl.DataFrame,
) -> Iterator[dict]:
    """Metabolic pathway pairs, emphasising CYP enzymes."""
    mols = _mol_lookup(molecule_dict)
    rec_to_mol = _record_to_molregno(compound_records)

    drug_enzymes: dict[int, list[str]] = {}
    for row in metabolism.to_dicts():
        record_id = row.get("substrate_record_id") or row.get("drug_record_id")
        enzyme = (row.get("enzyme_name") or "").strip()
        if not record_id or not enzyme:
            continue
        molregno = rec_to_mol.get(record_id)
        if not molregno:
            continue
        drug_enzymes.setdefault(molregno, []).append(enzyme)

    for molregno, enzymes in drug_enzymes.items():
        mol = mols.get(molregno)
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        unique_enzymes = list(dict.fromkeys(enzymes))
        enzyme_str = ", ".join(unique_enzymes)

        yield {
            "text": (
                f"### Question\nHow is {drug} metabolized?\n\n"
                f"### Answer\n{drug} ({chembl_id}) is metabolized by: {enzyme_str}. "
                f"Understanding these pathways is essential for predicting drug-drug interactions."
            )
        }

        cyp_enzymes = [e for e in unique_enzymes if "CYP" in e.upper()]
        if cyp_enzymes:
            cyp_str = ", ".join(cyp_enzymes)
            yield {
                "text": (
                    f"### Question\nWhich CYP enzymes are involved in the metabolism of {drug}?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) is a substrate of {cyp_str}. "
                    f"Co-administration with inhibitors or inducers of these enzymes may alter "
                    f"its plasma concentrations and clinical effect."
                )
            }


def generate_ddi_qa(
    metabolism: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    compound_records: pl.DataFrame,
    max_pairs: int = MAX_DDI_PAIRS,
) -> Iterator[dict]:
    """
    Infer drug-drug interactions from shared CYP substrates.

    Two drugs that share a CYP enzyme compete for metabolism; co-administration
    may raise or lower plasma levels of either drug.
    Only named drugs (pref_name set) are included to keep outputs readable.
    """
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(
            ["molregno", "pref_name", "chembl_id"]
        ).to_dicts()
        if row.get("pref_name")  # named drugs only
    }
    rec_to_mol = _record_to_molregno(compound_records)

    enzyme_drugs: dict[str, list[int]] = {}
    for row in metabolism.to_dicts():
        record_id = row.get("substrate_record_id") or row.get("drug_record_id")
        enzyme = (row.get("enzyme_name") or "").strip()
        if not record_id or "CYP" not in enzyme.upper():
            continue
        molregno = rec_to_mol.get(record_id)
        if molregno and molregno in mols:
            enzyme_drugs.setdefault(enzyme, []).append(molregno)

    rng = random.Random(RANDOM_SEED)
    pairs_seen: set[frozenset] = set()
    pair_count = 0

    for enzyme, drug_list in enzyme_drugs.items():
        unique_drugs = list(set(drug_list))
        if len(unique_drugs) < 2:
            continue
        rng.shuffle(unique_drugs)

        for i in range(len(unique_drugs)):
            for j in range(i + 1, len(unique_drugs)):
                if pair_count >= max_pairs:
                    return

                pair_key: frozenset = frozenset([unique_drugs[i], unique_drugs[j]])
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)

                mol_a = mols[unique_drugs[i]]
                mol_b = mols[unique_drugs[j]]
                name_a, id_a = _drug_name(mol_a), mol_a.get("chembl_id", "")
                name_b, id_b = _drug_name(mol_b), mol_b.get("chembl_id", "")

                yield {
                    "text": (
                        f"### Question\nCan {name_a} and {name_b} be safely co-administered?\n\n"
                        f"### Answer\nCaution is advised when co-administering {name_a} ({id_a}) "
                        f"and {name_b} ({id_b}), as both are substrates of {enzyme}. "
                        f"Competition for this enzyme may alter the plasma concentrations of "
                        f"either drug, potentially affecting efficacy or increasing toxicity risk. "
                        f"Always consult prescribing guidelines and consider therapeutic drug monitoring."
                    )
                }

                yield {
                    "text": (
                        f"### Question\nWhat is the drug interaction between {name_a} and {name_b}?\n\n"
                        f"### Answer\n{name_a} and {name_b} share a common metabolic pathway "
                        f"via {enzyme}. When co-administered, competition for {enzyme} may increase "
                        f"or decrease plasma levels of one or both drugs."
                    )
                }

                pair_count += 1


def generate_activity_qa(
    activities: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    max_records: int = MAX_ACTIVITY_RECORDS,
) -> Iterator[dict]:
    """Bioactivity potency pairs from measured pChEMBL values."""
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(
            ["molregno", "pref_name", "chembl_id"]
        ).to_dicts()
        if row.get("pref_name")
    }

    count = 0
    for row in activities.filter(pl.col("pchembl_value").is_not_null()).to_dicts():
        if count >= max_records:
            break
        mol = mols.get(row.get("molregno"))
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        pchembl = row.get("pchembl_value")
        std_type = row.get("standard_type") or "activity"
        std_value = row.get("standard_value")
        std_units = row.get("standard_units") or ""

        potency = (
            "high" if pchembl >= 7
            else "moderate" if pchembl >= 5
            else "low"
        )

        yield {
            "text": (
                f"### Question\nHow potent is {drug} based on its measured bioactivity?\n\n"
                f"### Answer\n{drug} ({chembl_id}) shows {potency} potency with a pChEMBL "
                f"value of {pchembl:.2f}. The measured {std_type} is "
                f"{std_value} {std_units}."
            )
        }
        count += 1


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def write_jsonl_splits(
    records: list[dict],
    output_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[int, int]:
    """Shuffle records and write train.jsonl / valid.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(records)

    split_idx = int(len(records) * train_ratio)
    train_records = records[:split_idx]
    valid_records = records[split_idx:]

    (output_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_records) + "\n"
    )
    (output_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(r) for r in valid_records) + "\n"
    )

    return len(train_records), len(valid_records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_drug_interaction_dataset(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Build QA-formatted JSONL training data for a drug-interaction chatbot.

    Reads ChEMBL parquet files from data_dir and writes:
      output_dir/train.jsonl  (90 % of pairs)
      output_dir/valid.jsonl  (10 % of pairs)
    """
    print("Loading ChEMBL tables...")
    tables = load_tables(data_dir)

    mol = tables.get("molecule_dictionary")
    if mol is None:
        raise RuntimeError("molecule_dictionary table is required but not found")

    compound_records = tables.get("compound_records")

    all_records: list[dict] = []

    if "drug_mechanism" in tables and "target_dictionary" in tables:
        print("Generating mechanism-of-action QA pairs...")
        pairs = list(generate_mechanism_qa(tables["drug_mechanism"], mol, tables["target_dictionary"]))
        print(f"  -> {len(pairs):,} pairs")
        all_records.extend(pairs)

    if "drug_indication" in tables:
        print("Generating drug-indication QA pairs...")
        pairs = list(generate_indication_qa(tables["drug_indication"], mol))
        print(f"  -> {len(pairs):,} pairs")
        all_records.extend(pairs)

    if "metabolism" in tables and compound_records is not None:
        print("Generating metabolism QA pairs...")
        pairs = list(generate_metabolism_qa(tables["metabolism"], mol, compound_records))
        print(f"  -> {len(pairs):,} pairs")
        all_records.extend(pairs)

        print("Generating drug-drug interaction pairs...")
        pairs = list(generate_ddi_qa(tables["metabolism"], mol, compound_records))
        print(f"  -> {len(pairs):,} pairs")
        all_records.extend(pairs)

    if "activities" in tables:
        print("Generating bioactivity QA pairs...")
        pairs = list(generate_activity_qa(tables["activities"], mol))
        print(f"  -> {len(pairs):,} pairs")
        all_records.extend(pairs)

    print(f"\nTotal QA pairs: {len(all_records):,}")
    n_train, n_valid = write_jsonl_splits(all_records, output_dir)
    print(f"Written to {output_dir}/")
    print(f"  train.jsonl : {n_train:,} records")
    print(f"  valid.jsonl : {n_valid:,} records")


if __name__ == "__main__":
    build_drug_interaction_dataset()
