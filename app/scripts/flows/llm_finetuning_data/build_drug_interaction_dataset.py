"""
Build a QA-formatted JSONL dataset for finetuning a drug-interaction chatbot.

Pulls ChEMBL tables and emits ten categories of training pairs:

  1. Mechanism of action      — "What does {drug} target?"
  2. Therapeutic indication   — "What is {drug} indicated for?"
  3. Metabolic pathways       — "How is {drug} metabolised?" / CYP-specific
  4. Drug-drug interactions   — pairs inferred from shared CYP substrates
  5. Bioactivity potency      — pChEMBL-based potency statements
  6. Drug warnings            — black box warnings and safety alerts
  7. Drug synonyms            — trade name / INN / USAN lookups
  8. Physicochemical props    — Lipinski, LogP, PSA, QED
  9. ATC classification       — WHO therapeutic class hierarchy
 10. Approved products        — formulations, routes, dosage forms

Output: data/llm_finetune/train.jsonl  (90 %)
        data/llm_finetune/valid.jsonl  (10 %)
"""

import json
import random
from collections.abc import Iterator
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/chembl_transform")
OUTPUT_DIR = Path("data/llm_finetune")

TRAIN_RATIO = 0.9
RANDOM_SEED = 42
MAX_DDI_PAIRS = 5_000
MAX_ACTIVITY_RECORDS = 50_000
MAX_COMPOUND_PROPERTY_RECORDS = 200_000

# Tables that benefit from row-capping during load (large parquet files).
# Others are loaded fully — they are small enough to fit in RAM.
_TABLE_ROW_LIMITS: dict[str, int] = {
    "activities": MAX_ACTIVITY_RECORDS * 3,   # 3× cap to survive pchembl filter
    "compound_properties": MAX_COMPOUND_PROPERTY_RECORDS,
}

_REQUIRED_TABLES = {
    "molecule_dictionary",
    "drug_mechanism",
    "drug_indication",
    "metabolism",
    "target_dictionary",
    "compound_records",
    "activities",
    "drug_warning",
    "molecule_synonyms",
    "compound_properties",
    "atc_classification",
    "molecule_atc_classification",
    "formulations",
    "products",
}


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


def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pl.DataFrame]:
    """
    Load all required ChEMBL tables from parquet.

    Large tables (activities, compound_properties) are lazily scanned and
    capped to avoid OOM. All other tables are read fully.
    """
    tables: dict[str, pl.DataFrame] = {}
    for name in _REQUIRED_TABLES:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  Warning: {name}.parquet not found, skipping")
            continue
        limit = _TABLE_ROW_LIMITS.get(name)
        if limit:
            tables[name] = pl.read_parquet(path, n_rows=limit)
        else:
            tables[name] = pl.read_parquet(path)
    return tables


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

    target_by_chembl_id = {
        row["chembl_id"]: row
        for row in target_dict.select(["chembl_id", "pref_name", "target_type"]).to_dicts()
    }

    for row in drug_mechanism.to_dicts():
        molregno: int | None = row.get("molregno")
        mol = mols.get(molregno) if molregno is not None else None
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
        if row.get("pref_name")
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
        if pchembl is None:
            continue
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


def generate_drug_warning_qa(
    drug_warning: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Safety warning pairs from drug_warning."""
    mols = _mol_lookup(molecule_dict)

    drug_warnings: dict[int, list[dict]] = {}
    for row in drug_warning.to_dicts():
        molregno = row.get("molregno")
        if molregno:
            drug_warnings.setdefault(molregno, []).append(row)

    for molregno, warnings in drug_warnings.items():
        mol = mols.get(molregno)
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")

        for w in warnings:
            wtype = (w.get("warning_type") or "Warning").strip()
            wclass = (w.get("warning_class") or "").strip()
            wcountry = (w.get("warning_country") or "").strip()
            wyear = w.get("warning_year")

            class_str = f" ({wclass})" if wclass else ""
            country_str = f" in {wcountry}" if wcountry else ""
            year_str = f" ({int(wyear)})" if wyear else ""

            yield {
                "text": (
                    f"### Question\nDoes {drug} have any safety warnings?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) has a {wtype}{class_str} issued"
                    f"{country_str}{year_str}. "
                    f"Prescribers should review current labeling for complete safety information."
                )
            }

            if wtype == "Black Box Warning":
                yield {
                    "text": (
                        f"### Question\nWhat is the black box warning for {drug}?\n\n"
                        f"### Answer\n{drug} ({chembl_id}) carries a Black Box Warning{class_str}. "
                        f"This is the most serious FDA warning, indicating significant risk of "
                        f"serious adverse effects. Always consult current prescribing information."
                    )
                }


def generate_synonym_qa(
    molecule_synonyms: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Trade name / INN / USAN synonym lookup pairs."""
    mols = _mol_lookup(molecule_dict)

    useful_types = {"TRADE_NAME", "INN", "USAN", "BAN"}
    drug_names: dict[int, dict[str, list[str]]] = {}
    for row in molecule_synonyms.filter(
        pl.col("syn_type").is_in(list(useful_types))
    ).to_dicts():
        molregno = row.get("molregno")
        syn_type = row.get("syn_type")
        synonym = (row.get("synonyms") or "").strip()
        if molregno and synonym:
            drug_names.setdefault(int(molregno), {}).setdefault(str(syn_type), []).append(synonym)

    for molregno, names_by_type in drug_names.items():
        mol = mols.get(molregno)
        if not mol:
            continue

        generic = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")

        trade_names = list(dict.fromkeys(names_by_type.get("TRADE_NAME", [])))
        inn_names = list(dict.fromkeys(
            names_by_type.get("INN", []) + names_by_type.get("USAN", []) + names_by_type.get("BAN", [])
        ))

        if trade_names:
            trade_str = ", ".join(trade_names[:5])
            yield {
                "text": (
                    f"### Question\nWhat are the brand names for {generic}?\n\n"
                    f"### Answer\n{generic} ({chembl_id}) is marketed under the trade "
                    f"name{'s' if len(trade_names) > 1 else ''}: {trade_str}."
                )
            }
            for trade in trade_names[:3]:
                yield {
                    "text": (
                        f"### Question\nWhat is {trade}?\n\n"
                        f"### Answer\n{trade} is a brand name for the drug {generic} ({chembl_id})."
                    )
                }

        if inn_names:
            inn_str = ", ".join(inn_names[:3])
            yield {
                "text": (
                    f"### Question\nWhat is the INN or generic name for {trade_names[0] if trade_names else generic}?\n\n"
                    f"### Answer\nThe International Nonproprietary Name (INN) is {inn_str} "
                    f"(ChEMBL ID: {chembl_id})."
                )
            }


def generate_physicochemical_qa(
    compound_properties: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Lipinski / physicochemical property pairs."""
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
    }

    for row in compound_properties.to_dicts():
        mol = mols.get(row.get("molregno"))
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")

        mw = row.get("full_mwt")
        logp = row.get("alogp")
        hba = row.get("hba")
        hbd = row.get("hbd")
        psa = row.get("psa")
        ro5 = row.get("num_ro5_violations")
        qed = row.get("qed_weighted")
        formula = row.get("full_molformula")

        if not mw:
            continue

        props = []
        if mw is not None:
            props.append(f"MW={mw:.1f} Da")
        if logp is not None:
            props.append(f"LogP={logp:.2f}")
        if hba is not None:
            props.append(f"HBA={hba}")
        if hbd is not None:
            props.append(f"HBD={hbd}")
        if psa is not None:
            props.append(f"PSA={psa:.1f} Å²")

        ro5_phrase = (
            f" with {ro5} Lipinski rule violation{'s' if ro5 != 1 else ''}"
            if ro5 else " (drug-like by Lipinski's rule of 5)"
        )

        yield {
            "text": (
                f"### Question\nWhat are the physicochemical properties of {drug}?\n\n"
                f"### Answer\n{drug} ({chembl_id}) has: {', '.join(props)}{ro5_phrase}."
                + (f" Molecular formula: {formula}." if formula else "")
                + (f" QED score: {qed:.3f}." if qed is not None else "")
            )
        }

        if ro5 and ro5 > 1:
            yield {
                "text": (
                    f"### Question\nDoes {drug} follow Lipinski's rule of 5?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) has {ro5} violation{'s' if ro5 != 1 else ''} "
                    f"of Lipinski's rule of 5, which may reduce oral bioavailability. "
                    + (f"MW={mw:.1f} Da, LogP={logp:.2f}." if logp is not None else f"MW={mw:.1f} Da.")
                )
            }


def generate_atc_classification_qa(
    atc_classification: pl.DataFrame,
    molecule_atc_classification: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """WHO ATC therapeutic class pairs."""
    mols = _mol_lookup(molecule_dict)

    atc_lookup = {row["level5"]: row for row in atc_classification.to_dicts()}

    mol_atc: dict[int, list[str]] = {}
    for row in molecule_atc_classification.to_dicts():
        molregno = row.get("molregno")
        level5 = row.get("level5")
        if molregno and level5:
            mol_atc.setdefault(molregno, []).append(level5)

    for molregno, atc_codes in mol_atc.items():
        mol = mols.get(molregno)
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")

        for code in atc_codes[:2]:
            atc = atc_lookup.get(code)
            if not atc:
                continue

            l1 = (atc.get("level1_description") or "").strip()
            l2 = (atc.get("level2_description") or "").strip()
            l4 = (atc.get("level4_description") or "").strip()
            who_name = (atc.get("who_name") or "").strip()

            if not who_name:
                continue

            yield {
                "text": (
                    f"### Question\nWhat therapeutic class does {drug} belong to?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) is classified under the WHO ATC system as "
                    f"{who_name} (code: {code})."
                    + (f" Anatomical group: {l1}." if l1 else "")
                    + (f" Pharmacological subgroup: {l2}." if l2 else "")
                    + (f" Chemical subgroup: {l4}." if l4 else "")
                )
            }

            yield {
                "text": (
                    f"### Question\nWhat is the ATC code for {drug}?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) has ATC code {code}: {who_name}."
                    + (f" It belongs to the {l1} anatomical group." if l1 else "")
                )
            }


def generate_approved_product_qa(
    formulations: pl.DataFrame,
    products: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Approved product / formulation pairs from FDA product data."""
    mols = _mol_lookup(molecule_dict)

    product_lookup = {row["product_id"]: row for row in products.to_dicts()}

    mol_products: dict[int, list[dict]] = {}
    for row in formulations.to_dicts():
        molregno = row.get("molregno")
        product_id = row.get("product_id")
        if not molregno or not product_id:
            continue
        product = product_lookup.get(product_id, {})
        trade_name = (product.get("trade_name") or "").strip()
        if not trade_name:
            continue
        mol_products.setdefault(molregno, []).append(
            {
                "trade_name": trade_name,
                "route": (product.get("route") or "").strip(),
                "dosage_form": (product.get("dosage_form") or "").strip(),
                "strength": (row.get("strength") or "").strip(),
                "black_box": product.get("black_box_warning") == 1,
            }
        )

    for molregno, prods in mol_products.items():
        mol = mols.get(molregno)
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")

        seen: set[str] = set()
        unique_prods = [p for p in prods if not (seen.add(p["trade_name"]) or p["trade_name"] in seen - {p["trade_name"]})]  # type: ignore[func-returns-value]

        routes = list(dict.fromkeys(p["route"] for p in unique_prods if p["route"]))
        forms = list(dict.fromkeys(p["dosage_form"] for p in unique_prods if p["dosage_form"]))
        names = [p["trade_name"] for p in unique_prods[:5]]

        if names:
            yield {
                "text": (
                    f"### Question\nWhat are the approved formulations of {drug}?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) is available as: {', '.join(names[:3])}."
                    + (f" Administration routes: {', '.join(routes)}." if routes else "")
                    + (f" Dosage forms: {', '.join(forms)}." if forms else "")
                )
            }

        bbw_prods = [p for p in unique_prods if p["black_box"]]
        if bbw_prods:
            bbw_names = ", ".join(p["trade_name"] for p in bbw_prods[:3])
            yield {
                "text": (
                    f"### Question\nDoes any formulation of {drug} carry a black box warning?\n\n"
                    f"### Answer\nYes, {drug} ({chembl_id}) has formulations with an FDA Black Box "
                    f"Warning. Affected products include: {bbw_names}. "
                    f"Review current prescribing information before use."
                )
            }


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
    # Narrow to a non-None local so lambdas below pass ty's type checks.
    # The "enabled" flag in each tuple guarantees this is only called when not None.
    _cr: pl.DataFrame = compound_records if compound_records is not None else pl.DataFrame()

    all_records: list[dict] = []

    _generators = [
        (
            "mechanism-of-action",
            lambda: generate_mechanism_qa(
                tables["drug_mechanism"], mol, tables["target_dictionary"]
            ),
            "drug_mechanism" in tables and "target_dictionary" in tables,
        ),
        (
            "drug-indication",
            lambda: generate_indication_qa(tables["drug_indication"], mol),
            "drug_indication" in tables,
        ),
        (
            "metabolism",
            lambda: generate_metabolism_qa(tables["metabolism"], mol, _cr),
            "metabolism" in tables and compound_records is not None,
        ),
        (
            "drug-drug interactions",
            lambda: generate_ddi_qa(tables["metabolism"], mol, _cr),
            "metabolism" in tables and compound_records is not None,
        ),
        (
            "bioactivity",
            lambda: generate_activity_qa(tables["activities"], mol),
            "activities" in tables,
        ),
        (
            "drug warnings",
            lambda: generate_drug_warning_qa(tables["drug_warning"], mol),
            "drug_warning" in tables,
        ),
        (
            "synonyms",
            lambda: generate_synonym_qa(tables["molecule_synonyms"], mol),
            "molecule_synonyms" in tables,
        ),
        (
            "physicochemical properties",
            lambda: generate_physicochemical_qa(tables["compound_properties"], mol),
            "compound_properties" in tables,
        ),
        (
            "ATC classification",
            lambda: generate_atc_classification_qa(
                tables["atc_classification"],
                tables["molecule_atc_classification"],
                mol,
            ),
            "atc_classification" in tables and "molecule_atc_classification" in tables,
        ),
        (
            "approved products",
            lambda: generate_approved_product_qa(
                tables["formulations"], tables["products"], mol
            ),
            "formulations" in tables and "products" in tables,
        ),
    ]

    for label, generator_fn, enabled in _generators:
        if not enabled:
            continue
        print(f"Generating {label} QA pairs...")
        pairs = list(generator_fn())
        print(f"  -> {len(pairs):,} pairs")
        all_records.extend(pairs)

    print(f"\nTotal QA pairs: {len(all_records):,}")
    n_train, n_valid = write_jsonl_splits(all_records, output_dir)
    print(f"Written to {output_dir}/")
    print(f"  train.jsonl : {n_train:,} records")
    print(f"  valid.jsonl : {n_valid:,} records")


if __name__ == "__main__":
    build_drug_interaction_dataset()
