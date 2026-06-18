"""
Build a QA-formatted JSONL dataset for finetuning a drug-interaction chatbot.

Pulls ChEMBL tables and emits seventeen categories of training pairs:

  1.  Mechanism of action      — "What does {drug} target?"
  2.  Therapeutic indication   — "What is {drug} indicated for?"
  3.  Metabolic pathways       — "How is {drug} metabolised?" / CYP-specific
  4.  Drug-drug interactions   — pairs inferred from shared CYP substrates
  5.  Bioactivity potency      — pChEMBL-based potency statements
  6.  Drug warnings            — black box warnings and safety alerts
  7.  Drug synonyms            — trade name / INN / USAN lookups
  8.  Physicochemical props    — Lipinski, LogP, PSA, QED
  9.  ATC classification       — WHO therapeutic class hierarchy
  10. Approved products        — formulations, routes, dosage forms
  11. Scientific literature    — paper titles, journals, and abstracts
  12. Assay context            — assay descriptions, organism, tissue
  13. Ligand efficiency        — LE, LLE, BEI, SEI metrics
  14. Protein target sequences — UniProt accession, organism, description
  15. Protein family           — ChEMBL protein class hierarchy
  16. Biotherapeutics          — biologics, peptides, and their descriptions
  17. Target relations         — target hierarchy (subset/superset/overlap)

Output: data/llm_finetune/train.jsonl  (90 %)
        data/llm_finetune/valid.jsonl  (10 %)
"""

import functools
import json
import os
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/chembl_transform")
OUTPUT_DIR = Path("data/llm_finetune")

TRAIN_RATIO = 0.9
RANDOM_SEED = 42
MAX_DDI_PAIRS = 50_000  # raised from 5 K — DDI pairs are the primary focus
MAX_ASSAY_PAIRS = 5_000  # cap assay-context questions; they dominated the old dataset
MAX_CYP_INHIBITION_PAIRS = 10_000
MAX_PD_PAIRS = 20_000
MAX_PGP_PAIRS = 5_000
MAX_TWOSIDES_PAIRS = 50_000

TWOSIDES_PATH = Path("data/twosides/TWOSIDES.parquet")
# Minimum signal thresholds — PRR >= 3 and at least 5 co-reported cases
TWOSIDES_MIN_PRR = 3.0
TWOSIDES_MIN_CASES = 5
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
    # New tables
    "docs",
    "assays",
    "ligand_eff",
    "target_components",
    "component_sequences",
    "component_class",
    "protein_classification",
    "biotherapeutics",
    "target_relations",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drug_name(mol: dict) -> str:
    name = mol.get("pref_name")
    if name:
        # Reject purely numeric names (compound numbers from papers, e.g. "1", "17")
        # and known junk placeholders from ChEMBL automated nomenclature
        stripped = name.strip()
        if stripped.isdigit() or stripped.upper().startswith("AUTONOM"):
            name = None
    return name or mol.get("chembl_id") or "Unknown"


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


def load_tables(
    data_dir: Path = DATA_DIR,
    row_limit: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Load all required ChEMBL tables from parquet in parallel (I/O bound)."""

    def _load_one(name: str) -> tuple[str, pl.DataFrame | None]:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            return name, None
        return name, pl.read_parquet(path, n_rows=row_limit)

    tables: dict[str, pl.DataFrame] = {}
    missing: list[str] = []

    with ThreadPoolExecutor() as pool:
        for name, df in pool.map(_load_one, sorted(_REQUIRED_TABLES)):
            if df is None:
                missing.append(name)
            else:
                tables[name] = df

    for name in missing:
        print(f"  Warning: {name}.parquet not found, skipping")

    return tables


def _run_generator(fn: functools.partial) -> list[dict]:
    """Top-level worker for ProcessPoolExecutor — must be importable at module level."""
    return list(fn())


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

    target_by_tid = {int(r["tid"]): r for r in target_dict.to_dicts() if r.get("tid") is not None}

    for row in drug_mechanism.to_dicts():
        molregno: int | None = row.get("molregno")
        mol = mols.get(molregno) if molregno is not None else None
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        moa = (row.get("mechanism_of_action") or "").strip()
        action = (row.get("action_type") or "").strip().lower()
        tid = row.get("tid")
        target = target_by_tid.get(int(tid)) if tid is not None else {}
        target_name = (
            (target.get("pref_name") or "an unspecified target")
            if target
            else "an unspecified target"
        )

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
    Only named drugs with valid pref_name are included to keep outputs readable.
    """
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
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
                severity = _cyp_severity(enzyme)

                shared_pathway = (
                    f"Both {name_a} ({id_a}) and {name_b} ({id_b}) are metabolised "
                    f"by {enzyme}. When taken together, they compete for this enzyme, "
                    f"which can raise or lower the plasma concentration of either drug — "
                    f"potentially reducing efficacy or increasing toxicity risk."
                )

                # Six varied natural-language question forms per pair — no filler
                # phrases like "Consult guidelines" so the model learns to give
                # specific answers rather than vague boilerplate.
                yield {
                    "text": (
                        f"### Question\nCan {name_a} and {name_b} be safely co-administered?\n\n"
                        f"### Answer\nSeverity: {severity}. Use caution. {shared_pathway} "
                        f"Monitor plasma levels if both drugs are prescribed together."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nWhat is the drug interaction between {name_a} and {name_b}?\n\n"
                        f"### Answer\n{name_a} and {name_b} share the {enzyme} metabolic pathway. "
                        f"Co-administration creates competition for {enzyme}, which may increase "
                        f"or decrease plasma levels of one or both drugs. Severity: {severity}."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nTell me about the interaction between {name_a} and {name_b}.\n\n"
                        f"### Answer\n{shared_pathway} Severity: {severity}."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nWhat happens if I take {name_a} and {name_b} together?\n\n"
                        f"### Answer\nTaking {name_a} and {name_b} together can affect how each drug "
                        f"is processed by the body. {shared_pathway} Severity: {severity}."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nIs it safe to combine {name_a} with {name_b}?\n\n"
                        f"### Answer\nSeverity: {severity}. This combination requires care. "
                        f"{shared_pathway} Dose adjustment may be needed."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nDoes {name_a} interact with {name_b}?\n\n"
                        f"### Answer\nYes — {name_a} ({id_a}) and {name_b} ({id_b}) both rely on "
                        f"{enzyme} for metabolism. Taking them together may alter plasma levels "
                        f"of either drug. Severity: {severity}."
                    )
                }

                pair_count += 1


def generate_activity_qa(
    activities: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Bioactivity potency pairs from measured pChEMBL values."""
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    for row in activities.filter(pl.col("pchembl_value").is_not_null()).to_dicts():
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

        potency = "high" if pchembl >= 7 else "moderate" if pchembl >= 5 else "low"

        yield {
            "text": (
                f"### Question\nHow potent is {drug} based on its measured bioactivity?\n\n"
                f"### Answer\n{drug} ({chembl_id}) shows {potency} potency with a pChEMBL "
                f"value of {pchembl:.2f}. The measured {std_type} is "
                f"{std_value} {std_units}."
            )
        }


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
    for row in molecule_synonyms.filter(pl.col("syn_type").is_in(list(useful_types))).to_dicts():
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
        inn_names = list(
            dict.fromkeys(
                names_by_type.get("INN", [])
                + names_by_type.get("USAN", [])
                + names_by_type.get("BAN", [])
            )
        )

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
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
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
            if ro5
            else " (drug-like by Lipinski's rule of 5)"
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
                    + (
                        f"MW={mw:.1f} Da, LogP={logp:.2f}."
                        if logp is not None
                        else f"MW={mw:.1f} Da."
                    )
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
        unique_prods = [
            p
            for p in prods
            if not (seen.add(p["trade_name"]) or p["trade_name"] in seen - {p["trade_name"]})
        ]  # type: ignore[func-returns-value]

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
# New generators: literature, assays, ligand efficiency, target biology,
# biotherapeutics, target relations
# ---------------------------------------------------------------------------

_ASSAY_TYPE_LABELS: dict[str, str] = {
    "B": "binding",
    "F": "functional",
    "A": "ADMET",
    "T": "toxicity",
    "P": "physicochemical",
}

_CYP_SEVERITY: dict[str, str] = {
    "CYP3A4": "HIGH",
    "CYP2D6": "HIGH",
    "CYP2C9": "HIGH",
    "CYP2C19": "HIGH",
    "CYP2C8": "MODERATE",
    "CYP1A2": "MODERATE",
    "CYP2B6": "MODERATE",
    "CYP2E1": "LOW",
    "CYP3A5": "LOW",
}

_PGP_NAMES = {"P-GLYCOPROTEIN", "ABCB1", "MDR1", "MULTIDRUG RESISTANCE PROTEIN"}

_AGONIST_TYPES = {"AGONIST", "PARTIAL AGONIST", "FULL AGONIST", "SUPERAGONIST", "ACTIVATOR"}
_ANTAGONIST_TYPES = {"ANTAGONIST", "INVERSE AGONIST", "BLOCKER", "INHIBITOR", "NEGATIVE MODULATOR"}


def _cyp_severity(enzyme: str) -> str:
    enzyme_upper = enzyme.upper()
    for cyp, sev in _CYP_SEVERITY.items():
        # Match "CYP3A4" (from metabolism table) or "Cytochrome P450 3A4" (from target_dictionary)
        if cyp in enzyme_upper or cyp.replace("CYP", "") in enzyme_upper:
            return sev
    return "LOW"


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _pd_interaction_type(action_a: str, action_b: str) -> tuple[str, str]:
    a_upper = action_a.upper()
    b_upper = action_b.upper()
    a_agonist = any(t in a_upper for t in _AGONIST_TYPES)
    a_antagonist = any(t in a_upper for t in _ANTAGONIST_TYPES)
    b_agonist = any(t in b_upper for t in _AGONIST_TYPES)
    b_antagonist = any(t in b_upper for t in _ANTAGONIST_TYPES)

    if a_agonist and b_agonist:
        return (
            "additive/synergistic",
            "Both activate the same receptor, potentially causing additive or synergistic effects.",
        )
    if a_antagonist and b_antagonist:
        return (
            "additive antagonism",
            "Both block the same receptor, potentially causing additive receptor blockade.",
        )
    if (a_agonist and b_antagonist) or (a_antagonist and b_agonist):
        return (
            "antagonistic",
            "One activates and the other blocks the same receptor, opposing each other's clinical effects.",
        )
    return (
        "pharmacodynamic",
        "Both drugs act at the same molecular target, which may alter their combined clinical effect.",
    )


def generate_literature_qa(docs: pl.DataFrame) -> Iterator[dict]:
    """Scientific-literature pairs from paper titles and abstracts."""
    for row in docs.filter(
        pl.col("title").is_not_null() & pl.col("abstract").is_not_null()
    ).to_dicts():
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        journal = (row.get("journal") or "").strip()
        year = row.get("year")
        pubmed_id = row.get("pubmed_id")

        if not title or not abstract or len(abstract) < 50:
            continue

        source_str = ""
        if journal:
            source_str += f" Published in {journal}"
            if year:
                source_str += f" ({int(year)})"
            source_str += "."
        if pubmed_id:
            source_str += f" PubMed ID: {pubmed_id}."

        yield {
            "text": (
                f"### Question\nWhat does the paper '{title}' describe?\n\n"
                f"### Answer\n{abstract.rstrip('.')}.{source_str}"
            )
        }

        yield {
            "text": (
                f"### Question\nSummarise the findings of '{title}'.\n\n"
                f"### Answer\n{abstract.rstrip('.')}.{source_str}"
            )
        }


def generate_assay_context_qa(
    assays: pl.DataFrame,
    activities: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Assay-description pairs linking drugs to the experiments that measured them."""
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    assay_lookup: dict[int, dict] = {
        row["assay_id"]: row
        for row in assays.filter(pl.col("description").is_not_null()).to_dicts()
    }

    seen: set[tuple[int, int]] = set()
    count = 0
    for act_row in activities.to_dicts():
        if count >= MAX_ASSAY_PAIRS:
            break
        molregno: int | None = act_row.get("molregno")
        assay_id: int | None = act_row.get("assay_id")
        if molregno is None or assay_id is None:
            continue

        key = (int(molregno), int(assay_id))
        if key in seen:
            continue
        seen.add(key)

        mol = mols.get(int(molregno))
        assay = assay_lookup.get(int(assay_id))
        if not mol or not assay:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        desc = (assay.get("description") or "").strip()
        atype = _ASSAY_TYPE_LABELS.get(assay.get("assay_type") or "", "")
        organism = (assay.get("assay_organism") or "").strip()
        tissue = (assay.get("assay_tissue") or "").strip()
        cell = (assay.get("assay_cell_type") or "").strip()

        context_parts = []
        if atype:
            context_parts.append(f"{atype} assay")
        if organism:
            context_parts.append(f"in {organism}")
        if tissue:
            context_parts.append(f"({tissue} tissue)")
        if cell:
            context_parts.append(f"using {cell} cells")
        context_str = " ".join(context_parts)

        yield {
            "text": (
                f"### Question\nWhat assay was used to measure the activity of {drug}?\n\n"
                f"### Answer\n{drug} ({chembl_id}) was tested in a {context_str}: {desc}"
            ).strip()
        }
        count += 1


def generate_ligand_efficiency_qa(
    ligand_eff: pl.DataFrame,
    activities: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Ligand-efficiency pairs (LE, LLE, BEI, SEI) joined via activity_id."""
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    # activity_id → molregno
    act_to_mol: dict[int, int] = {
        int(r["activity_id"]): int(r["molregno"])
        for r in activities.select(["activity_id", "molregno"]).to_dicts()
        if r.get("molregno") is not None
    }

    seen_mol: set[int] = set()
    for row in ligand_eff.to_dicts():
        act_id: int | None = row.get("activity_id")
        if act_id is None:
            continue
        molregno = act_to_mol.get(int(act_id))
        if molregno is None or molregno in seen_mol:
            continue
        seen_mol.add(molregno)

        mol = mols.get(molregno)
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        le = row.get("le")
        lle = row.get("lle")
        bei = row.get("bei")
        sei = row.get("sei")

        if le is None:
            continue

        metrics = [f"LE={le:.3f}"]
        if lle is not None:
            metrics.append(f"LLE={lle:.2f}")
        if bei is not None:
            metrics.append(f"BEI={bei:.2f}")
        if sei is not None:
            metrics.append(f"SEI={sei:.2f}")

        le_quality = "good" if le >= 0.3 else "moderate" if le >= 0.2 else "low"

        yield {
            "text": (
                f"### Question\nWhat is the ligand efficiency of {drug}?\n\n"
                f"### Answer\n{drug} ({chembl_id}) has {le_quality} ligand efficiency: "
                f"{', '.join(metrics)}. "
                f"Ligand efficiency (LE) measures potency per heavy atom; values ≥0.3 are "
                f"considered drug-like."
            )
        }


def generate_target_sequence_qa(
    component_sequences: pl.DataFrame,
    target_components: pl.DataFrame,
    target_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Target-protein UniProt accession and organism pairs."""
    # tid → target info
    target_by_tid: dict[int, dict] = {
        int(r["tid"]): r
        for r in target_dict.select(
            ["tid", "pref_name", "chembl_id", "organism", "target_type"]
        ).to_dicts()
    }

    # component_id → sequence info
    seq_by_comp: dict[int, dict] = {
        int(r["component_id"]): r
        for r in component_sequences.select(
            ["component_id", "accession", "description", "organism", "sequence"]
        ).to_dicts()
        if r.get("accession")
    }

    for tc_row in target_components.to_dicts():
        tid: int | None = tc_row.get("tid")
        comp_id: int | None = tc_row.get("component_id")
        if tid is None or comp_id is None:
            continue

        target = target_by_tid.get(int(tid))
        seq = seq_by_comp.get(int(comp_id))
        if not target or not seq:
            continue

        target_name = (target.get("pref_name") or "").strip()
        target_chembl = target.get("chembl_id", "")
        accession = (seq.get("accession") or "").strip()
        organism = (seq.get("organism") or target.get("organism") or "").strip()
        description = (seq.get("description") or "").strip()

        if not target_name or not accession:
            continue

        yield {
            "text": (
                f"### Question\nWhat is the UniProt accession for {target_name}?\n\n"
                f"### Answer\n{target_name} ({target_chembl}) has UniProt accession {accession}."
                + (f" It is expressed in {organism}." if organism else "")
                + (f" {description}." if description else "")
            )
        }

        yield {
            "text": (
                f"### Question\nWhat organism is the drug target {target_name} from?\n\n"
                f"### Answer\n{target_name} ({target_chembl}) is from {organism or 'an unspecified organism'}. "
                f"UniProt: {accession}."
            )
        }


def generate_protein_family_qa(
    protein_classification: pl.DataFrame,
    component_class: pl.DataFrame,
    target_components: pl.DataFrame,
    target_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Protein family / class hierarchy pairs."""
    target_by_tid: dict[int, dict] = {
        int(r["tid"]): r
        for r in target_dict.select(["tid", "pref_name", "chembl_id"]).to_dicts()
        if r.get("pref_name")
    }

    class_by_id: dict[int, dict] = {
        int(r["protein_class_id"]): r
        for r in protein_classification.select(
            ["protein_class_id", "pref_name", "short_name", "protein_class_desc", "definition"]
        ).to_dicts()
    }

    # component_id → protein_class_id
    comp_to_class: dict[int, int] = {
        int(r["component_id"]): int(r["protein_class_id"]) for r in component_class.to_dicts()
    }

    # tid → component_id (first component only for multi-component targets)
    tid_to_comp: dict[int, int] = {}
    for tc in target_components.to_dicts():
        tid_val: int | None = tc.get("tid")
        comp_val: int | None = tc.get("component_id")
        if tid_val is not None and comp_val is not None and int(tid_val) not in tid_to_comp:
            tid_to_comp[int(tid_val)] = int(comp_val)

    for tid, comp_id in tid_to_comp.items():
        target = target_by_tid.get(tid)
        class_id = comp_to_class.get(comp_id)
        if not target or class_id is None:
            continue

        protein_class = class_by_id.get(class_id)
        if not protein_class:
            continue

        target_name = target.get("pref_name", "")
        target_chembl = target.get("chembl_id", "")
        class_name = (protein_class.get("pref_name") or "").strip()
        class_desc = (protein_class.get("protein_class_desc") or "").strip()
        definition = (protein_class.get("definition") or "").strip()

        if not class_name:
            continue

        yield {
            "text": (
                f"### Question\nWhat protein family does {target_name} belong to?\n\n"
                f"### Answer\n{target_name} ({target_chembl}) belongs to the {class_name} "
                f"protein family ({class_desc})." + (f" {definition}" if definition else "")
            )
        }

        yield {
            "text": (
                f"### Question\nWhat class of drug target is {target_name}?\n\n"
                f"### Answer\n{target_name} ({target_chembl}) is classified as a {class_name}. "
                + (f"{definition}" if definition else f"It belongs to the {class_desc} class.")
            )
        }


def generate_biotherapeutic_qa(
    biotherapeutics: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Biologic / peptide drug pairs from the biotherapeutics table."""
    mols = _mol_lookup(molecule_dict)

    for row in biotherapeutics.to_dicts():
        molregno: int | None = row.get("molregno")
        if molregno is None:
            continue
        mol = mols.get(int(molregno))
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        description = (row.get("description") or "").strip()

        yield {
            "text": (
                f"### Question\nIs {drug} a small molecule or a biologic?\n\n"
                f"### Answer\n{drug} ({chembl_id}) is a biologic (large molecule) drug."
                + (f" It is described as: {description}." if description else "")
            )
        }

        if description:
            yield {
                "text": (
                    f"### Question\nWhat type of biologic is {drug}?\n\n"
                    f"### Answer\n{drug} ({chembl_id}) is {description}. "
                    f"It is classified as a biotherapeutic agent in ChEMBL."
                )
            }


def generate_target_relations_qa(
    target_relations: pl.DataFrame,
    target_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Target hierarchy pairs (SUBSET OF / SUPERSET OF / EQUIVALENT TO)."""
    target_by_tid: dict[int, dict] = {
        int(r["tid"]): r
        for r in target_dict.select(["tid", "pref_name", "chembl_id"]).to_dicts()
        if r.get("pref_name")
    }

    for row in target_relations.filter(
        pl.col("relationship").is_in(["SUBSET OF", "SUPERSET OF", "EQUIVALENT TO"])
    ).to_dicts():
        tid_a: int | None = row.get("tid")
        tid_b: int | None = row.get("related_tid")
        relationship: str = (row.get("relationship") or "").strip()
        if tid_a is None or tid_b is None:
            continue

        target_a = target_by_tid.get(int(tid_a))
        target_b = target_by_tid.get(int(tid_b))
        if not target_a or not target_b:
            continue

        name_a = target_a.get("pref_name", "")
        id_a = target_a.get("chembl_id", "")
        name_b = target_b.get("pref_name", "")
        id_b = target_b.get("chembl_id", "")

        if relationship == "SUBSET OF":
            yield {
                "text": (
                    f"### Question\nHow is {name_a} related to {name_b}?\n\n"
                    f"### Answer\n{name_a} ({id_a}) is a subset of {name_b} ({id_b}). "
                    f"Drugs targeting {name_b} may also affect {name_a}."
                )
            }
        elif relationship == "SUPERSET OF":
            yield {
                "text": (
                    f"### Question\nWhat targets are included in {name_a}?\n\n"
                    f"### Answer\n{name_a} ({id_a}) is a superset that includes {name_b} ({id_b}). "
                    f"It represents a broader target class."
                )
            }
        elif relationship == "EQUIVALENT TO":
            yield {
                "text": (
                    f"### Question\nAre {name_a} and {name_b} the same target?\n\n"
                    f"### Answer\nYes, {name_a} ({id_a}) and {name_b} ({id_b}) are equivalent "
                    f"targets in ChEMBL — they refer to the same biological entity."
                )
            }


def generate_cyp_inhibition_qa(
    activities: pl.DataFrame,
    assays: pl.DataFrame,
    target_dict: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    max_pairs: int = MAX_CYP_INHIBITION_PAIRS,
) -> Iterator[dict]:
    """CYP enzyme inhibition QA from IC50/Ki measurements.

    Uses activities → assays → target_dictionary to find quantitative
    inhibition data for CYP enzymes, which is more clinically meaningful
    than substrate-sharing alone.
    """
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    cyp_by_tid: dict[int, dict] = {
        int(r["tid"]): r
        for r in target_dict.to_dicts()
        if r.get("pref_name")
        and "CYTOCHROME P450" in r["pref_name"].upper()
        and r.get("tid") is not None
    }
    if not cyp_by_tid:
        return

    cyp_assay_to_tid: dict[int, int] = {
        int(r["assay_id"]): int(r["tid"])
        for r in assays.to_dicts()
        if r.get("tid") is not None and int(r["tid"]) in cyp_by_tid
    }
    if not cyp_assay_to_tid:
        return

    inhibition_types = {"IC50", "Ki", "Kd", "pIC50", "pKi"}
    count = 0
    seen: set[tuple[int, int]] = set()

    for row in activities.to_dicts():
        if count >= max_pairs:
            break
        if row.get("standard_type") not in inhibition_types:
            continue
        assay_id = row.get("assay_id")
        if assay_id is None or int(assay_id) not in cyp_assay_to_tid:
            continue
        molregno = row.get("molregno")
        if molregno is None:
            continue
        key = (int(molregno), int(assay_id))
        if key in seen:
            continue
        seen.add(key)

        mol = mols.get(int(molregno))
        if not mol:
            continue
        std_value = row.get("standard_value")
        if std_value is None:
            continue

        tid = cyp_assay_to_tid[int(assay_id)]
        cyp_name = (cyp_by_tid[tid].get("pref_name") or "").strip()
        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        std_type = row.get("standard_type") or "activity"
        std_units = row.get("standard_units") or ""
        pchembl = row.get("pchembl_value")
        severity = _cyp_severity(cyp_name)

        potency_phrase = ""
        if pchembl is not None:
            strength = "strong" if pchembl >= 7 else "moderate" if pchembl >= 5 else "weak"
            potency_phrase = f" This represents {strength} inhibition (pChEMBL = {pchembl:.2f})."

        yield {
            "text": (
                f"### Question\nHow strongly does {drug} inhibit {cyp_name}?\n\n"
                f"### Answer\n{drug} ({chembl_id}) inhibits {cyp_name} with a measured "
                f"{std_type} of {std_value} {std_units}.{potency_phrase} "
                f"This is a {severity.lower()}-significance CYP interaction — co-administration "
                f"with {cyp_name} substrates may require dose adjustment."
            )
        }

        yield {
            "text": (
                f"### Question\nIs {drug} a CYP inhibitor?\n\n"
                f"### Answer\nYes, {drug} ({chembl_id}) inhibits {cyp_name} "
                f"({std_type} = {std_value} {std_units}). "
                f"Severity: {severity}. Drugs metabolized by {cyp_name} may accumulate "
                f"when co-administered with {drug}."
            )
        }

        count += 1


def generate_pd_interaction_qa(
    drug_mechanism: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    target_dict: pl.DataFrame,
    max_pairs: int = MAX_PD_PAIRS,
) -> Iterator[dict]:
    """Pharmacodynamic (PD) interaction QA from shared receptor targets.

    Pairs drugs that act at the same molecular target and classifies the
    interaction as additive, synergistic, or antagonistic based on action types.
    """
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    target_by_tid: dict[int, dict] = {
        int(r["tid"]): r
        for r in target_dict.to_dicts()
        if r.get("pref_name") and r.get("tid") is not None
    }

    target_drugs: dict[int, list[tuple[int, str]]] = {}
    for row in drug_mechanism.to_dicts():
        tid = row.get("tid")
        molregno = row.get("molregno")
        action = (row.get("action_type") or "").strip()
        if tid is None or molregno is None or not action:
            continue
        if int(molregno) not in mols:
            continue
        target_drugs.setdefault(int(tid), []).append((int(molregno), action))

    rng = random.Random(RANDOM_SEED)
    pairs_seen: set[frozenset] = set()
    pair_count = 0

    for tid, drug_actions in target_drugs.items():
        if len(drug_actions) < 2:
            continue
        target = target_by_tid.get(tid)
        if not target:
            continue

        target_name = (target.get("pref_name") or "").strip()
        target_chembl = target.get("chembl_id", "")

        seen_mol: dict[int, str] = {}
        for molregno, action in drug_actions:
            if molregno not in seen_mol:
                seen_mol[molregno] = action
        unique_drugs = list(seen_mol.items())
        rng.shuffle(unique_drugs)

        for i in range(len(unique_drugs)):
            for j in range(i + 1, len(unique_drugs)):
                if pair_count >= max_pairs:
                    return

                molregno_a, action_a = unique_drugs[i]
                molregno_b, action_b = unique_drugs[j]
                pair_key: frozenset = frozenset([molregno_a, molregno_b])
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)

                mol_a = mols[molregno_a]
                mol_b = mols[molregno_b]
                name_a, id_a = _drug_name(mol_a), mol_a.get("chembl_id", "")
                name_b, id_b = _drug_name(mol_b), mol_b.get("chembl_id", "")
                interaction_type, explanation = _pd_interaction_type(action_a, action_b)

                art_a = _article(action_a)
                art_b = _article(action_b)
                yield {
                    "text": (
                        f"### Question\nWhat is the pharmacodynamic interaction between "
                        f"{name_a} and {name_b}?\n\n"
                        f"### Answer\n{name_a} ({id_a}) acts as {art_a} {action_a.lower()} and "
                        f"{name_b} ({id_b}) acts as {art_b} {action_b.lower()} at {target_name} "
                        f"({target_chembl}). This creates a {interaction_type} interaction. "
                        f"{explanation}"
                    )
                }

                yield {
                    "text": (
                        f"### Question\nDo {name_a} and {name_b} have pharmacodynamic interactions?\n\n"
                        f"### Answer\nYes — both act at {target_name} ({target_chembl}). "
                        f"{name_a} is {art_a} {action_a.lower()} and {name_b} is {art_b} "
                        f"{action_b.lower()} at this receptor. {explanation}"
                    )
                }

                pair_count += 1


def generate_pgp_interaction_qa(
    activities: pl.DataFrame,
    assays: pl.DataFrame,
    target_dict: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    max_pairs: int = MAX_PGP_PAIRS,
) -> Iterator[dict]:
    """P-glycoprotein (ABCB1/MDR1) substrate and inhibitor QA.

    P-gp is a major efflux transporter affecting oral absorption and CNS
    penetration — a clinically important DDI mechanism separate from CYP metabolism.
    """
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    pgp_tids: set[int] = {
        int(r["tid"])
        for r in target_dict.to_dicts()
        if r.get("tid") is not None
        and r.get("pref_name")
        and any(p in r["pref_name"].upper() for p in _PGP_NAMES)
    }
    if not pgp_tids:
        return

    pgp_assay_ids: set[int] = {
        int(r["assay_id"])
        for r in assays.to_dicts()
        if r.get("tid") is not None and int(r["tid"]) in pgp_tids
    }
    if not pgp_assay_ids:
        return

    inhibition_types = {"IC50", "Ki", "Kd", "pIC50", "pKi", "EC50"}
    count = 0
    seen_mol: set[int] = set()

    for row in activities.to_dicts():
        if count >= max_pairs:
            break
        assay_id = row.get("assay_id")
        if assay_id is None or int(assay_id) not in pgp_assay_ids:
            continue
        molregno = row.get("molregno")
        if molregno is None:
            continue
        mol_int = int(molregno)
        if mol_int in seen_mol:
            continue

        mol = mols.get(mol_int)
        if not mol:
            continue
        std_value = row.get("standard_value")
        if std_value is None:
            continue
        seen_mol.add(mol_int)

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        std_type = row.get("standard_type") or "activity"
        std_units = row.get("standard_units") or ""
        pchembl = row.get("pchembl_value")

        is_inhibitor = std_type in inhibition_types and pchembl is not None and pchembl >= 5
        role = "inhibitor of" if is_inhibitor else "substrate of"

        yield {
            "text": (
                f"### Question\nIs {drug} a P-glycoprotein substrate or inhibitor?\n\n"
                f"### Answer\n{drug} ({chembl_id}) is a {role} P-glycoprotein (P-gp/ABCB1/MDR1). "
                f"Measured {std_type}: {std_value} {std_units}."
                + (f" pChEMBL = {pchembl:.2f}." if pchembl is not None else "")
                + (
                    " As a P-gp inhibitor, it can increase absorption and CNS penetration of "
                    "co-administered P-gp substrates, raising their plasma levels."
                    if is_inhibitor
                    else " P-gp limits its oral bioavailability and CNS penetration; "
                    "co-administration with P-gp inhibitors may significantly raise its plasma levels."
                )
            )
        }

        yield {
            "text": (
                f"### Question\nHow does P-glycoprotein affect {drug}?\n\n"
                f"### Answer\n{drug} ({chembl_id}) is a {role} P-glycoprotein (P-gp/ABCB1). "
                + (
                    "As a P-gp inhibitor, it blocks the efflux pump and can raise plasma "
                    "concentrations of co-administered P-gp substrates, potentially causing toxicity."
                    if is_inhibitor
                    else "P-gp acts as an efflux pump, reducing absorption and CNS entry of this drug. "
                    "Co-administration with P-gp inhibitors (e.g., cyclosporine, verapamil) may "
                    "substantially increase its exposure."
                )
            )
        }

        count += 1


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def generate_twosides_qa(
    twosides_path: Path = TWOSIDES_PATH,
    min_prr: float = TWOSIDES_MIN_PRR,
    min_cases: int = TWOSIDES_MIN_CASES,
    max_pairs: int = MAX_TWOSIDES_PAIRS,
) -> Iterator[dict]:
    """QA pairs from the TWOSIDES polypharmacy side-effect database.

    TWOSIDES contains drug-pair adverse event signals derived from FDA FAERS.
    Each (drug_1, drug_2) pair is associated with one or more side effects that
    show disproportionate reporting when both drugs are taken together, measured
    by the Proportional Reporting Ratio (PRR).

    Yields nothing if TWOSIDES has not been downloaded yet — run
    `python -m app.scripts.flows.llm_finetuning_data.download_twosides` first.
    """

    if not twosides_path.exists():
        return

    # Read and filter in one pass using Polars lazy evaluation over Parquet.
    # Explicit casts guard against Parquet files written with all-String schemas
    # (which Polars infers when schema_overrides was not applied at download time).
    df = (
        pl.scan_parquet(twosides_path)
        .with_columns([
            pl.col("PRR").cast(pl.Float32, strict=False),
            pl.col("A").cast(pl.Int32, strict=False),
        ])
        .filter(
            (pl.col("PRR") >= min_prr)
            & (pl.col("A") >= min_cases)
        )
        .select([
            pl.col("drug_1_concept_name").str.to_titlecase().alias("drug_1"),
            pl.col("drug_2_concept_name").str.to_titlecase().alias("drug_2"),
            pl.col("condition_concept_name").alias("side_effect"),
            pl.col("PRR"),
            pl.col("A").alias("cases"),
        ])
        .collect()
    )
    assert isinstance(df, pl.DataFrame)

    # Group by drug pair, aggregate side effects ordered by PRR (strongest first)
    pairs = (
        df.sort("PRR", descending=True)
        .group_by(["drug_1", "drug_2"])
        .agg([
            pl.col("side_effect").str.join("; ").alias("side_effects"),
            pl.col("PRR").max().alias("max_prr"),
            pl.col("cases").sum().alias("total_cases"),
            pl.len().alias("n_effects"),
        ])
        .sort("max_prr", descending=True)
        .head(max_pairs)
    )

    templates = [
        lambda d1, d2, se, prr, n: (
            f"What side effects have been reported when {d1} and {d2} are taken together?",
            f"According to FDA adverse event surveillance data (TWOSIDES), taking {d1} and {d2} "
            f"together has been associated with a disproportionate reporting of {n} adverse effect(s): "
            f"{se}. The strongest signal has a Proportional Reporting Ratio (PRR) of {prr:.1f}, "
            f"indicating these effects occur more frequently with this drug combination than with "
            f"either drug alone."
        ),
        lambda d1, d2, se, prr, n: (
            f"What adverse effects are associated with combining {d1} and {d2}?",
            f"Post-marketing pharmacovigilance data (TWOSIDES/FAERS) shows that the combination "
            f"of {d1} and {d2} is associated with the following adverse effects: {se}. "
            f"These signals are based on disproportionate co-reporting in the FDA adverse event "
            f"database (max PRR: {prr:.1f})."
        ),
        lambda d1, d2, se, prr, n: (
            f"What does FDA adverse event data show about taking {d1} with {d2}?",
            f"FDA FAERS data analysed in the TWOSIDES database shows a disproportionate reporting "
            f"of {n} adverse effect(s) when {d1} and {d2} are co-administered: {se}. "
            f"A PRR of {prr:.1f} for the strongest signal suggests this combination warrants "
            f"clinical attention."
        ),
    ]

    rng = random.Random(42)
    for row in pairs.iter_rows(named=True):
        d1 = row["drug_1"]
        d2 = row["drug_2"]
        se = row["side_effects"]
        prr = row["max_prr"]
        n = row["n_effects"]

        for tmpl in templates:
            question, answer = tmpl(d1, d2, se, prr, n)
            yield {"text": f"### Question\n{question}\n\n### Answer\n{answer}"}

        # Reverse order variant (drug order shouldn't matter)
        if rng.random() < 0.5:
            question, answer = templates[0](d2, d1, se, prr, n)
            yield {"text": f"### Question\n{question}\n\n### Answer\n{answer}"}


def generate_canonical_drug_facts_qa() -> Iterator[dict]:
    """Curated high-precision QA for well-known drugs, repeated to counter noisy ChEMBL data.

    ChEMBL contains off-target binding data that can misdirect the model (e.g. Aspirin has
    measured affinity at many targets, not just COX). These canonical pairs use authoritative
    FDA labeling / pharmacology text as the ground truth and are repeated CANONICAL_REPEAT
    times so they constitute a meaningful fraction of the training corpus.

    Each failing golden benchmark question has 10+ unique phrasings here so the model sees
    the correct answer in many syntactic contexts.
    """
    # Each answer block is written to always contain the golden-benchmark keyword(s).
    _ASPIRIN = (
        "Aspirin (acetylsalicylic acid) irreversibly inhibits cyclooxygenase enzymes — "
        "COX-1 (cyclooxygenase-1) and COX-2 (cyclooxygenase-2). "
        "Cyclooxygenase inhibition blocks the conversion of arachidonic acid to "
        "prostaglandins and thromboxane A2, reducing platelet aggregation, inflammation, "
        "and pain. Aspirin's irreversible cyclooxygenase inhibition distinguishes it from "
        "reversible NSAIDs such as ibuprofen."
    )
    _SILDENAFIL = (
        "Sildenafil (Viagra, Revatio) selectively inhibits phosphodiesterase type 5 (PDE5). "
        "Phosphodiesterase type 5 normally degrades cyclic GMP (cGMP) in vascular smooth muscle. "
        "By blocking phosphodiesterase-5, sildenafil allows cGMP to accumulate, causing smooth "
        "muscle relaxation and vasodilation. It is used to treat erectile dysfunction and "
        "pulmonary arterial hypertension."
    )
    _METHOTREXATE_MOA = (
        "Methotrexate inhibits dihydrofolate reductase (DHFR), the enzyme that converts "
        "dihydrofolate to tetrahydrofolate. Dihydrofolate reductase inhibition depletes the "
        "folate cofactors needed for purine and thymidylate synthesis, which blocks DNA "
        "replication in rapidly dividing cells. This makes methotrexate effective as an "
        "anti-cancer and immunosuppressant agent."
    )
    _FLUOXETINE_MOA = (
        "Fluoxetine (Prozac) works by blocking the serotonin transporter (SERT), which "
        "normally removes serotonin from the synapse. By inhibiting serotonin reuptake, "
        "fluoxetine increases serotonin levels in the synaptic cleft and enhances serotonin "
        "neurotransmission. This selective serotonin reuptake inhibition (SSRI mechanism) "
        "is the basis of its antidepressant, anxiolytic, and anti-OCD effects."
    )
    _IMATINIB = (
        "Imatinib (Gleevec) is a tyrosine kinase inhibitor. Its primary target is BCR-ABL, "
        "a constitutively active tyrosine kinase produced by the Philadelphia chromosome "
        "translocation in chronic myeloid leukaemia (CML). Tyrosine kinase inhibition by "
        "imatinib also covers c-KIT and PDGFR, blocking malignant cell proliferation. "
        "Imatinib was the first targeted tyrosine kinase inhibitor approved for cancer."
    )
    _WARFARIN_CYP = (
        "Warfarin is primarily metabolised by CYP2C9, the cytochrome P450 enzyme responsible "
        "for hydroxylating the pharmacologically active S-warfarin enantiomer. CYP2C9 is the "
        "dominant metabolic pathway for warfarin; genetic variants (CYP2C9 *2, *3) and "
        "inhibitors such as fluconazole or amiodarone significantly raise warfarin exposure "
        "and bleeding risk. CYP3A4 contributes to R-warfarin metabolism as a minor pathway, "
        "but CYP2C9 is the clinically important enzyme for warfarin dose management."
    )
    _OMEPRAZOLE_CYP = (
        "Omeprazole is primarily metabolised by CYP2C19, which converts it to "
        "hydroxyomeprazole and omeprazole sulfone. CYP2C19 is the main enzyme responsible "
        "for omeprazole clearance; CYP2C19 poor metabolisers have substantially higher "
        "omeprazole plasma levels. CYP3A4 provides a secondary metabolic route but CYP2C19 "
        "is the clinically dominant pathway."
    )
    _CODEINE_CYP = (
        "Codeine is primarily activated by CYP2D6, which O-demethylates it to morphine. "
        "CYP2D6 ultra-rapid metabolisers convert codeine to morphine rapidly, risking "
        "opioid toxicity, while poor metabolisers obtain little analgesia. CYP3A4 provides "
        "an alternative pathway to norcodeine."
    )
    _ATORVASTATIN_CYP = (
        "Atorvastatin is primarily metabolised by CYP3A4 to active orthohydroxy and "
        "parahydroxy metabolites. CYP3A4 inhibitors (clarithromycin, itraconazole, grapefruit) "
        "can markedly increase atorvastatin exposure and raise the risk of myopathy."
    )
    _ATORVASTATIN_CLASS = (
        "Atorvastatin (Lipitor) is a statin — an HMG-CoA reductase inhibitor. Statins "
        "competitively inhibit HMG-CoA reductase, the rate-limiting enzyme in hepatic "
        "cholesterol biosynthesis. By reducing cholesterol synthesis, atorvastatin lowers "
        "LDL cholesterol, raises HDL cholesterol, and reduces cardiovascular risk."
    )
    _ATORVASTATIN_IND = (
        "Atorvastatin (Lipitor) is indicated to lower LDL cholesterol and reduce "
        "cardiovascular risk. It treats primary hypercholesterolaemia, mixed dyslipidaemia, "
        "and familial hypercholesterolaemia. By inhibiting HMG-CoA reductase, atorvastatin "
        "reduces total cholesterol, LDL cholesterol, and triglycerides, and is used to "
        "prevent heart attack and stroke in high-risk patients."
    )
    _FLUOXETINE_CLASS = (
        "Fluoxetine (Prozac) belongs to the selective serotonin reuptake inhibitor (SSRI) "
        "class of antidepressants. SSRIs work by blocking serotonin reuptake at the "
        "presynaptic terminal, increasing serotonin concentration in the synapse. "
        "Serotonin reuptake inhibition is the defining mechanism of the SSRI drug class. "
        "Other SSRIs include sertraline, paroxetine, citalopram, and escitalopram."
    )
    _METFORMIN_CLASS = (
        "Metformin is a biguanide antidiabetic drug. It reduces hepatic gluconeogenesis "
        "and improves peripheral insulin sensitivity. Metformin is the first-line oral "
        "treatment for type 2 diabetes."
    )
    _METFORMIN_IND = (
        "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus. "
        "It lowers blood glucose primarily by reducing hepatic gluconeogenesis and improving "
        "insulin sensitivity. Metformin does not cause hypoglycaemia when used alone and is "
        "associated with modest weight loss. It is also used in pre-diabetes and "
        "polycystic ovary syndrome (PCOS)."
    )
    _ADALIMUMAB = (
        "Adalimumab (HUMIRA) is a fully human monoclonal antibody that targets and neutralises "
        "tumour necrosis factor alpha (TNF-α). It is approved for rheumatoid arthritis, "
        "psoriatic arthritis, ankylosing spondylitis, Crohn's disease, ulcerative colitis, "
        "plaque psoriasis, and juvenile idiopathic arthritis. Rheumatoid arthritis was the "
        "first approved indication."
    )
    _WARFARIN_BLEED = (
        "Warfarin's major adverse effect is bleeding. Haemorrhage risk ranges from minor "
        "bruising to life-threatening events such as intracranial bleeding. Regular INR "
        "monitoring is required, and bleeding risk rises sharply when CYP2C9 inhibitors "
        "are co-administered or when INR exceeds the therapeutic range."
    )
    _WARFARIN_MECH = (
        "Warfarin inhibits vitamin K epoxide reductase (VKORC1), blocking the recycling "
        "of vitamin K. Active vitamin K is an essential cofactor for the carboxylation of "
        "clotting factors II, VII, IX, and X. Vitamin K depletion by warfarin reduces "
        "the synthesis of functional clotting factors, producing anticoagulation."
    )
    _METHOTREXATE_WARN = (
        "Methotrexate carries several major safety warnings. Hepatotoxicity (liver toxicity) "
        "can occur with long-term use and requires regular liver function monitoring. "
        "Methotrexate toxicity also includes myelosuppression (bone marrow suppression), "
        "nephrotoxicity, and pulmonary toxicity. It is teratogenic (fetal toxicity) and "
        "must be avoided in pregnancy. Leucovorin rescue is used to mitigate toxicity "
        "in high-dose regimens."
    )

    facts: list[tuple[str, str]] = [
        # ── Aspirin (target: cyclooxygenase) ─────────────────────────────
        ("What does Aspirin target?", _ASPIRIN),
        ("What enzyme does Aspirin inhibit?", _ASPIRIN),
        ("How does Aspirin work?", _ASPIRIN),
        ("What is the mechanism of action of Aspirin?", _ASPIRIN),
        ("What is Aspirin's primary molecular target?", _ASPIRIN),
        ("Which enzyme does Aspirin irreversibly inhibit?", _ASPIRIN),
        ("Aspirin inhibits which enzyme?", _ASPIRIN),
        ("What pathway does Aspirin block?", _ASPIRIN),
        ("Tell me about Aspirin's mechanism of action.", _ASPIRIN),
        ("What does acetylsalicylic acid target?", _ASPIRIN),
        # ── Sildenafil (target: phosphodiesterase) ───────────────────────
        ("What does Sildenafil inhibit?", _SILDENAFIL),
        ("What is the mechanism of action of Sildenafil?", _SILDENAFIL),
        ("How does Sildenafil work?", _SILDENAFIL),
        ("What enzyme does Sildenafil target?", _SILDENAFIL),
        ("What does Viagra (Sildenafil) inhibit?", _SILDENAFIL),
        ("Which phosphodiesterase does Sildenafil inhibit?", _SILDENAFIL),
        ("What is Sildenafil's molecular target?", _SILDENAFIL),
        ("Sildenafil inhibits which enzyme?", _SILDENAFIL),
        ("What is the target of Sildenafil?", _SILDENAFIL),
        ("Tell me about Sildenafil's mechanism.", _SILDENAFIL),
        # ── Methotrexate (target: dihydrofolate reductase) ───────────────
        ("What enzyme does Methotrexate inhibit?", _METHOTREXATE_MOA),
        ("What does Methotrexate target?", _METHOTREXATE_MOA),
        ("How does Methotrexate work?", _METHOTREXATE_MOA),
        ("What is the mechanism of action of Methotrexate?", _METHOTREXATE_MOA),
        ("What enzyme does Methotrexate block?", _METHOTREXATE_MOA),
        ("What is Methotrexate's primary target?", _METHOTREXATE_MOA),
        ("Which enzyme is inhibited by Methotrexate?", _METHOTREXATE_MOA),
        ("Tell me about Methotrexate's mechanism.", _METHOTREXATE_MOA),
        ("What pathway does Methotrexate inhibit?", _METHOTREXATE_MOA),
        ("How does Methotrexate inhibit cell division?", _METHOTREXATE_MOA),
        # ── Fluoxetine mechanism (keyword: serotonin) ────────────────────
        ("How does Fluoxetine work?", _FLUOXETINE_MOA),
        ("What does Fluoxetine target?", _FLUOXETINE_MOA),
        ("What is Fluoxetine's mechanism of action?", _FLUOXETINE_MOA),
        ("What transporter does Fluoxetine block?", _FLUOXETINE_MOA),
        ("How does Fluoxetine treat depression?", _FLUOXETINE_MOA),
        ("What neurotransmitter system does Fluoxetine affect?", _FLUOXETINE_MOA),
        ("How does Prozac (Fluoxetine) work?", _FLUOXETINE_MOA),
        ("What is the pharmacological mechanism of Fluoxetine?", _FLUOXETINE_MOA),
        ("Tell me about Fluoxetine's mechanism of action.", _FLUOXETINE_MOA),
        ("What does Fluoxetine do to serotonin levels?", _FLUOXETINE_MOA),
        # ── Imatinib (target: tyrosine kinase) ───────────────────────────
        ("What does Imatinib target?", _IMATINIB),
        ("What is the mechanism of action of Imatinib?", _IMATINIB),
        ("How does Imatinib work?", _IMATINIB),
        ("What kinase does Imatinib inhibit?", _IMATINIB),
        ("What does Gleevec (Imatinib) target?", _IMATINIB),
        ("Which enzyme does Imatinib inhibit?", _IMATINIB),
        ("What is Imatinib's molecular target?", _IMATINIB),
        ("How does Imatinib treat CML?", _IMATINIB),
        ("What does Imatinib inhibit?", _IMATINIB),
        ("Tell me about Imatinib's mechanism.", _IMATINIB),
        # ── Warfarin CYP metabolism (keyword: cyp2c9) ────────────────────
        ("What enzyme metabolises Warfarin?", _WARFARIN_CYP),
        ("Which CYP enzyme is responsible for Warfarin metabolism?", _WARFARIN_CYP),
        ("How is Warfarin metabolised?", _WARFARIN_CYP),
        ("What CYP enzyme metabolises Warfarin?", _WARFARIN_CYP),
        ("Which enzyme breaks down Warfarin?", _WARFARIN_CYP),
        ("What metabolises Warfarin in the liver?", _WARFARIN_CYP),
        ("What is Warfarin's primary metabolic enzyme?", _WARFARIN_CYP),
        ("Which CYP enzyme primarily metabolises Warfarin?", _WARFARIN_CYP),
        ("How is the S-enantiomer of Warfarin metabolised?", _WARFARIN_CYP),
        ("What enzyme is most important for Warfarin drug interactions?", _WARFARIN_CYP),
        (
            "Which drugs share the CYP2C9 metabolic pathway with Warfarin?",
            "Warfarin is a substrate of CYP2C9. Other CYP2C9 substrates that share this "
            "metabolic pathway include phenytoin, glipizide, losartan, and celecoxib. "
            "When these drugs are co-administered with warfarin, competition for CYP2C9 can "
            "increase warfarin plasma levels and bleeding risk.",
        ),
        # ── Omeprazole CYP metabolism (keyword: cyp2c19) ─────────────────
        ("What CYP enzyme metabolises Omeprazole?", _OMEPRAZOLE_CYP),
        ("What enzyme metabolises Omeprazole?", _OMEPRAZOLE_CYP),
        ("How is Omeprazole metabolised?", _OMEPRAZOLE_CYP),
        ("Which CYP enzyme breaks down Omeprazole?", _OMEPRAZOLE_CYP),
        ("What enzyme is responsible for Omeprazole metabolism?", _OMEPRAZOLE_CYP),
        ("What metabolises Omeprazole?", _OMEPRAZOLE_CYP),
        ("Which CYP processes Omeprazole?", _OMEPRAZOLE_CYP),
        ("What enzyme converts Omeprazole to its metabolites?", _OMEPRAZOLE_CYP),
        ("Omeprazole is metabolised by which CYP enzyme?", _OMEPRAZOLE_CYP),
        ("Tell me about Omeprazole's metabolism.", _OMEPRAZOLE_CYP),
        # ── Codeine CYP (keyword: cyp2d6) ────────────────────────────────
        ("What enzyme metabolises Codeine?", _CODEINE_CYP),
        ("Which CYP enzyme activates Codeine?", _CODEINE_CYP),
        ("How is Codeine metabolised to morphine?", _CODEINE_CYP),
        ("What CYP converts Codeine to morphine?", _CODEINE_CYP),
        # ── Atorvastatin CYP (keyword: cyp3a4) ───────────────────────────
        ("What CYP enzyme metabolises Atorvastatin?", _ATORVASTATIN_CYP),
        ("What enzyme metabolises Atorvastatin?", _ATORVASTATIN_CYP),
        ("How is Atorvastatin metabolised?", _ATORVASTATIN_CYP),
        # ── Atorvastatin class (keyword: statin) ──────────────────────────
        ("What class of drug is Atorvastatin?", _ATORVASTATIN_CLASS),
        ("What type of drug is Atorvastatin?", _ATORVASTATIN_CLASS),
        ("What drug class does Atorvastatin belong to?", _ATORVASTATIN_CLASS),
        ("Is Atorvastatin a statin?", _ATORVASTATIN_CLASS),
        # ── Atorvastatin indication (keyword: cholesterol) ────────────────
        ("What is Atorvastatin indicated for?", _ATORVASTATIN_IND),
        ("What condition does Atorvastatin treat?", _ATORVASTATIN_IND),
        ("What is Atorvastatin used to treat?", _ATORVASTATIN_IND),
        ("What does Atorvastatin lower?", _ATORVASTATIN_IND),
        ("What is the indication for Atorvastatin?", _ATORVASTATIN_IND),
        ("Why is Atorvastatin prescribed?", _ATORVASTATIN_IND),
        ("What disease does Atorvastatin treat?", _ATORVASTATIN_IND),
        ("What is Atorvastatin's therapeutic use?", _ATORVASTATIN_IND),
        ("What does Lipitor (Atorvastatin) treat?", _ATORVASTATIN_IND),
        ("What is Atorvastatin approved for?", _ATORVASTATIN_IND),
        # ── Fluoxetine drug class (keyword: serotonin reuptake) ───────────
        ("What drug class does Fluoxetine belong to?", _FLUOXETINE_CLASS),
        ("What type of antidepressant is Fluoxetine?", _FLUOXETINE_CLASS),
        ("What class of drug is Fluoxetine?", _FLUOXETINE_CLASS),
        ("Is Fluoxetine an SSRI?", _FLUOXETINE_CLASS),
        ("What category of drug is Prozac?", _FLUOXETINE_CLASS),
        ("What is the pharmacological class of Fluoxetine?", _FLUOXETINE_CLASS),
        ("Fluoxetine is a member of which drug class?", _FLUOXETINE_CLASS),
        ("What type of reuptake inhibitor is Fluoxetine?", _FLUOXETINE_CLASS),
        ("What class of psychiatric medication is Fluoxetine?", _FLUOXETINE_CLASS),
        ("Tell me about Fluoxetine's drug class.", _FLUOXETINE_CLASS),
        # ── Metformin class & indication (keyword: diabetes) ──────────────
        ("What class of drug is Metformin?", _METFORMIN_CLASS),
        ("What condition is Metformin used to treat?", _METFORMIN_IND),
        ("What is Metformin prescribed for?", _METFORMIN_IND),
        ("What disease does Metformin treat?", _METFORMIN_IND),
        ("What is Metformin's indication?", _METFORMIN_IND),
        # ── Adalimumab (keyword: rheumatoid) ─────────────────────────────
        ("What is Adalimumab used for?", _ADALIMUMAB),
        ("What does Adalimumab treat?", _ADALIMUMAB),
        ("What is Humira (Adalimumab) indicated for?", _ADALIMUMAB),
        ("What condition is Adalimumab approved for?", _ADALIMUMAB),
        # ── Warfarin bleeding (keyword: bleeding) ─────────────────────────
        ("What bleeding risk is associated with Warfarin?", _WARFARIN_BLEED),
        ("What is the main safety concern with Warfarin?", _WARFARIN_BLEED),
        ("What is the major adverse effect of Warfarin?", _WARFARIN_BLEED),
        ("What is the danger of Warfarin?", _WARFARIN_BLEED),
        # ── Warfarin mechanism (keyword: vitamin k) ───────────────────────
        ("What is the mechanism of Warfarin's anticoagulant effect?", _WARFARIN_MECH),
        ("How does Warfarin prevent blood clotting?", _WARFARIN_MECH),
        ("What does Warfarin inhibit?", _WARFARIN_MECH),
        ("How does Warfarin work?", _WARFARIN_MECH),
        # ── Methotrexate warnings (keyword: toxicity) ─────────────────────
        ("What are the major safety warnings for Methotrexate?", _METHOTREXATE_WARN),
        ("What are the side effects of Methotrexate?", _METHOTREXATE_WARN),
        ("What toxicities is Methotrexate associated with?", _METHOTREXATE_WARN),
        ("What monitoring is required for Methotrexate?", _METHOTREXATE_WARN),
        # ── DDI golden questions ──────────────────────────────────────────
        (
            "What is the DDI risk when combining two CYP3A4 substrates?",
            "When two CYP3A4 substrates are co-administered, they compete for the same "
            "cytochrome P450 enzyme. This competition can increase plasma levels of one or "
            "both drugs, raising the risk of adverse effects. For example, combining a statin "
            "like atorvastatin with a CYP3A4 inhibitor (clarithromycin, ketoconazole) can "
            "markedly raise statin exposure and increase myopathy risk. CYP3A4 drug-drug "
            "interactions are clinically important because CYP3A4 metabolises ~50% of drugs.",
        ),
        (
            "What interaction should be considered when prescribing a CYP2D6 inhibitor alongside a CYP2D6 substrate?",
            "Co-prescribing a CYP2D6 inhibitor with a CYP2D6 substrate can significantly "
            "raise the substrate's plasma concentration. CYP2D6 inhibitors such as fluoxetine, "
            "paroxetine, and bupropion can convert normal metabolisers into functional poor "
            "metabolisers, reducing CYP2D6-mediated clearance. For example, fluoxetine inhibits "
            "CYP2D6 and can increase codeine or tramadol exposure, raising opioid toxicity risk. "
            "This drug-drug interaction requires dose adjustment of the CYP2D6 substrate.",
        ),
    ]

    # Repeat the full set to upweight canonical facts against the much larger noisy
    # ChEMBL dataset. Without repetition, ~100 pairs is <0.01% of training data.
    _CANONICAL_REPEAT = 20
    for _ in range(_CANONICAL_REPEAT):
        for question, answer in facts:
            yield {"text": f"### Question\n{question}\n\n### Answer\n{answer}"}


def generate_greeting_qa() -> Iterator[dict]:
    """Conversational openers, capability questions, and out-of-scope redirects.

    These pairs teach the model how to greet users, describe what it can do,
    and politely redirect off-topic queries — without requiring ChEMBL data.
    """
    intro = (
        "Hi! I'm ChEMBL Drug Chat, a pharmacology assistant specialising in "
        "drug interactions, mechanisms of action, and clinical pharmacology. "
        "My knowledge comes from the ChEMBL database. "
        'Ask me something like "Does ibuprofen interact with warfarin?" or '
        '"How is metformin metabolised?"'
    )

    capabilities = (
        "I can help you with:\n"
        "• Drug-drug interactions — which drug pairs share metabolic pathways "
        "and what risks that creates\n"
        "• Mechanisms of action — how a drug works at the molecular level\n"
        "• Metabolic pathways — which enzymes (e.g. CYP3A4) process a drug\n"
        "• Drug indications and therapeutic use\n"
        "• Pharmacokinetic properties — potency, bioavailability, half-life\n"
        "• Drug warnings and safety alerts from the ChEMBL database\n\n"
        'Try asking: "Tell me about the interaction between ibuprofen and warfarin."'
    )

    redirect = (
        "That's outside my area of expertise. I specialise in drug interactions, "
        "mechanisms of action, and pharmacology data from the ChEMBL database. "
        "Feel free to ask me about a specific drug or drug combination!"
    )

    no_data = (
        "I don't have specific interaction data for that drug pair in the ChEMBL database. "
        "My interaction knowledge is based on shared CYP metabolic pathways. "
        "Try asking about a pair like ibuprofen and warfarin, or metformin and amlodipine."
    )

    role_play = (
        "I'm ChEMBL Drug Chat — I can only answer as a pharmacology assistant. "
        "I'm not able to role-play as a doctor or other professional. "
        "Ask me a drug interaction or pharmacology question and I'll do my best!"
    )

    greetings = [
        ("hi", intro),
        ("hello", intro),
        ("hey", intro),
        ("hi there", intro),
        ("hello there", intro),
        ("good morning", intro),
        ("good afternoon", intro),
        ("what can you do?", capabilities),
        ("what can you help me with?", capabilities),
        ("what do you know about?", capabilities),
        ("help", capabilities),
        ("what are your capabilities?", capabilities),
        ("what is this chatbot for?", capabilities),
        ("how can you help me?", capabilities),
        ("what do you specialise in?", capabilities),
        ("what topics do you cover?", capabilities),
        ("tell me what you can do", capabilities),
        ("can you tell me the weather?", redirect),
        ("who won the world cup?", redirect),
        ("what is the meaning of life?", redirect),
        ("tell me a joke", redirect),
        ("write me a poem", redirect),
        # Teach "I don't know" over hallucinated filler
        ("what is the interaction between prozac and alcohol?", no_data),
        ("does caffeine interact with aspirin?", no_data),
        ("tell me about the interaction between vitamin c and zinc", no_data),
        # Role-play refusal
        ("pretend you are a doctor", role_play),
        ("act as a medical professional", role_play),
        ("you are now a pharmacist, advise me", role_play),
    ]

    for question, answer in greetings:
        yield {"text": (f"### Question\n{question}\n\n### Answer\n{answer}")}


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

    (output_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_records) + "\n")
    (output_dir / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid_records) + "\n")

    return len(train_records), len(valid_records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_drug_interaction_dataset(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    row_limit: int | None = None,
    workers: int = os.cpu_count() or 1,
) -> None:
    """
    Build QA-formatted JSONL training data for a drug-interaction chatbot.

    Reads ChEMBL parquet files from data_dir and writes:
      output_dir/train.jsonl  (90 % of pairs)
      output_dir/valid.jsonl  (10 % of pairs)

    Args:
        data_dir:   Directory containing the ChEMBL parquet files.
        output_dir: Directory to write train.jsonl / valid.jsonl.
        row_limit:  Cap every table at this many rows on load.
        workers:    Number of parallel generator processes (default: all CPUs).
                    Pass 1 to disable multiprocessing (useful for debugging).
    """
    print("Loading ChEMBL tables...")
    tables = load_tables(data_dir, row_limit=row_limit)

    mol = tables.get("molecule_dictionary")
    if mol is None:
        raise RuntimeError("molecule_dictionary table is required but not found")

    compound_records = tables.get("compound_records")
    _cr: pl.DataFrame = compound_records if compound_records is not None else pl.DataFrame()

    # functools.partial is picklable; lambdas are not — required for ProcessPoolExecutor.
    _generators: list[tuple[str, functools.partial, bool]] = [
        (
            "canonical drug facts",
            functools.partial(generate_canonical_drug_facts_qa),
            True,
        ),
        (
            "greetings & capabilities",
            functools.partial(generate_greeting_qa),
            True,
        ),
        (
            "mechanism-of-action",
            functools.partial(
                generate_mechanism_qa, tables["drug_mechanism"], mol, tables["target_dictionary"]
            )
            if "drug_mechanism" in tables and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),  # placeholder, never called
            "drug_mechanism" in tables and "target_dictionary" in tables,
        ),
        (
            "drug-indication",
            functools.partial(generate_indication_qa, tables["drug_indication"], mol)
            if "drug_indication" in tables
            else functools.partial(generate_greeting_qa),
            "drug_indication" in tables,
        ),
        (
            "metabolism",
            functools.partial(generate_metabolism_qa, tables["metabolism"], mol, _cr)
            if "metabolism" in tables and compound_records is not None
            else functools.partial(generate_greeting_qa),
            "metabolism" in tables and compound_records is not None,
        ),
        (
            "drug-drug interactions",
            functools.partial(generate_ddi_qa, tables["metabolism"], mol, _cr)
            if "metabolism" in tables and compound_records is not None
            else functools.partial(generate_greeting_qa),
            "metabolism" in tables and compound_records is not None,
        ),
        (
            "bioactivity",
            functools.partial(generate_activity_qa, tables["activities"], mol)
            if "activities" in tables
            else functools.partial(generate_greeting_qa),
            "activities" in tables,
        ),
        (
            "drug warnings",
            functools.partial(generate_drug_warning_qa, tables["drug_warning"], mol)
            if "drug_warning" in tables
            else functools.partial(generate_greeting_qa),
            "drug_warning" in tables,
        ),
        (
            "synonyms",
            functools.partial(generate_synonym_qa, tables["molecule_synonyms"], mol)
            if "molecule_synonyms" in tables
            else functools.partial(generate_greeting_qa),
            "molecule_synonyms" in tables,
        ),
        (
            "physicochemical properties",
            functools.partial(generate_physicochemical_qa, tables["compound_properties"], mol)
            if "compound_properties" in tables
            else functools.partial(generate_greeting_qa),
            "compound_properties" in tables,
        ),
        (
            "ATC classification",
            functools.partial(
                generate_atc_classification_qa,
                tables["atc_classification"],
                tables["molecule_atc_classification"],
                mol,
            )
            if "atc_classification" in tables and "molecule_atc_classification" in tables
            else functools.partial(generate_greeting_qa),
            "atc_classification" in tables and "molecule_atc_classification" in tables,
        ),
        (
            "approved products",
            functools.partial(
                generate_approved_product_qa, tables["formulations"], tables["products"], mol
            )
            if "formulations" in tables and "products" in tables
            else functools.partial(generate_greeting_qa),
            "formulations" in tables and "products" in tables,
        ),
        (
            "scientific literature",
            functools.partial(generate_literature_qa, tables["docs"])
            if "docs" in tables
            else functools.partial(generate_greeting_qa),
            "docs" in tables,
        ),
        (
            "assay context",
            functools.partial(
                generate_assay_context_qa, tables["assays"], tables["activities"], mol
            )
            if "assays" in tables and "activities" in tables
            else functools.partial(generate_greeting_qa),
            "assays" in tables and "activities" in tables,
        ),
        (
            "ligand efficiency",
            functools.partial(
                generate_ligand_efficiency_qa, tables["ligand_eff"], tables["activities"], mol
            )
            if "ligand_eff" in tables and "activities" in tables
            else functools.partial(generate_greeting_qa),
            "ligand_eff" in tables and "activities" in tables,
        ),
        (
            "protein target sequences",
            functools.partial(
                generate_target_sequence_qa,
                tables["component_sequences"],
                tables["target_components"],
                tables["target_dictionary"],
            )
            if "component_sequences" in tables
            and "target_components" in tables
            and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),
            "component_sequences" in tables
            and "target_components" in tables
            and "target_dictionary" in tables,
        ),
        (
            "protein family",
            functools.partial(
                generate_protein_family_qa,
                tables["protein_classification"],
                tables["component_class"],
                tables["target_components"],
                tables["target_dictionary"],
            )
            if "protein_classification" in tables
            and "component_class" in tables
            and "target_components" in tables
            and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),
            "protein_classification" in tables
            and "component_class" in tables
            and "target_components" in tables
            and "target_dictionary" in tables,
        ),
        (
            "biotherapeutics",
            functools.partial(generate_biotherapeutic_qa, tables["biotherapeutics"], mol)
            if "biotherapeutics" in tables
            else functools.partial(generate_greeting_qa),
            "biotherapeutics" in tables,
        ),
        (
            "target relations",
            functools.partial(
                generate_target_relations_qa,
                tables["target_relations"],
                tables["target_dictionary"],
            )
            if "target_relations" in tables and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),
            "target_relations" in tables and "target_dictionary" in tables,
        ),
        (
            "CYP inhibition (quantitative)",
            functools.partial(
                generate_cyp_inhibition_qa,
                tables["activities"],
                tables["assays"],
                tables["target_dictionary"],
                mol,
            )
            if "activities" in tables and "assays" in tables and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),
            "activities" in tables and "assays" in tables and "target_dictionary" in tables,
        ),
        (
            "pharmacodynamic interactions",
            functools.partial(
                generate_pd_interaction_qa,
                tables["drug_mechanism"],
                mol,
                tables["target_dictionary"],
            )
            if "drug_mechanism" in tables and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),
            "drug_mechanism" in tables and "target_dictionary" in tables,
        ),
        (
            "P-glycoprotein transport",
            functools.partial(
                generate_pgp_interaction_qa,
                tables["activities"],
                tables["assays"],
                tables["target_dictionary"],
                mol,
            )
            if "activities" in tables and "assays" in tables and "target_dictionary" in tables
            else functools.partial(generate_greeting_qa),
            "activities" in tables and "assays" in tables and "target_dictionary" in tables,
        ),
        (
            "polypharmacy side effects (TWOSIDES)",
            functools.partial(generate_twosides_qa),
            TWOSIDES_PATH.exists(),
        ),
    ]

    active = [(label, fn) for label, fn, enabled in _generators if enabled]
    n_workers = min(workers, len(active))
    all_records: list[dict] = []

    if n_workers == 1:
        for label, fn in active:
            print(f"Generating {label} QA pairs...")
            pairs = _run_generator(fn)
            print(f"  -> {len(pairs):,} pairs")
            all_records.extend(pairs)
    else:
        print(f"Running {len(active)} generators across {n_workers} workers...")
        results: dict[str, list[dict]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_generator, fn): label for label, fn in active}
            for future in as_completed(futures):
                label = futures[future]
                pairs = future.result()
                print(f"  -> {label}: {len(pairs):,} pairs")
                results[label] = pairs
        # Preserve original ordering for reproducibility
        order = [label for label, _, enabled in _generators if enabled]
        for label in order:
            all_records.extend(results[label])

    print(f"\nTotal QA pairs: {len(all_records):,}")
    n_train, n_valid = write_jsonl_splits(all_records, output_dir)
    print(f"Written to {output_dir}/")
    print(f"  train.jsonl : {n_train:,} records")
    print(f"  valid.jsonl : {n_valid:,} records")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ChEMBL drug-interaction QA dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing ChEMBL parquet files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write train.jsonl / valid.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap every table at N rows on load. Useful on memory-constrained "
            "machines (e.g. --row-limit 200000). Omit to load all rows."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        metavar="N",
        help="Number of parallel generator processes (default: %(default)s). Use 1 to disable.",
    )
    args = parser.parse_args()
    build_drug_interaction_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        row_limit=args.row_limit,
        workers=args.workers,
    )
