"""
Build a QA-formatted JSONL dataset for finetuning a drug-interaction chatbot.

    Pulls ChEMBL tables and emits eighteen categories of training pairs:

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
   18. CYP inhibition           — quantitative IC50/Ki CYP inhibitor data

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


def _standard_value_to_micromolar(value: float | int | None, units: str | None) -> float | None:
    """Convert a standard value to micromolar when the unit is known."""
    if value is None or not units:
        return None

    normalized_units = units.strip().replace("µ", "u").replace("μ", "u").lower()
    multipliers = {
        "pm": 1e-6,
        "nm": 1e-3,
        "um": 1.0,
        "mm": 1e3,
        "m": 1e6,
    }
    multiplier = multipliers.get(normalized_units)
    if multiplier is None:
        return None

    return float(value) * multiplier


def _cyp_potency_label(value_um: float) -> str:
    """Classify CYP inhibition potency from a micromolar value."""
    if value_um < 1:
        return "strong"
    if value_um <= 10:
        return "moderate"
    return "weak"


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------


def load_tables(
    data_dir: Path = DATA_DIR,
    row_limit: int | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Load all required ChEMBL tables from parquet.

    Args:
        data_dir:  Directory containing the parquet files.
        row_limit: Optional cap applied uniformly to every table. Useful on
                   memory-constrained machines.  Pass e.g. ``200_000`` to load
                   at most 200 K rows per table.  ``None`` (default) loads
                   everything.
    """
    tables: dict[str, pl.DataFrame] = {}
    for name in _REQUIRED_TABLES:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  Warning: {name}.parquet not found, skipping")
            continue
        tables[name] = pl.read_parquet(path, n_rows=row_limit)
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


def generate_cyp_inhibition_qa(
    activities: pl.DataFrame,
    assays: pl.DataFrame,
    target_dict: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> Iterator[dict]:
    """Quantitative CYP inhibition QA pairs from IC50/Ki activity measurements."""
    mols = {
        row["molregno"]: row
        for row in molecule_dict.select(["molregno", "pref_name", "chembl_id"]).to_dicts()
        if row.get("pref_name")
        and not row["pref_name"].strip().isdigit()
        and not row["pref_name"].strip().upper().startswith("AUTONOM")
    }

    assay_by_id: dict[int, dict] = {
        int(row["assay_id"]): row
        for row in assays.select(["assay_id", "tid"]).to_dicts()
        if row.get("assay_id") is not None and row.get("tid") is not None
    }
    target_by_tid: dict[int, dict] = {
        int(row["tid"]): row
        for row in target_dict.select(["tid", "chembl_id", "pref_name", "target_type"]).to_dicts()
        if row.get("tid") is not None
    }

    seen: set[tuple[int, int, str, float]] = set()

    for row in activities.to_dicts():
        molregno = row.get("molregno")
        assay_id = row.get("assay_id")
        standard_type = (row.get("standard_type") or "").strip()
        standard_relation = (row.get("standard_relation") or "").strip()
        standard_value = row.get("standard_value")
        standard_units = row.get("standard_units")

        if (
            molregno is None
            or assay_id is None
            or standard_type.upper() not in {"IC50", "KI"}
            or standard_relation not in {"", "=", None}
        ):
            continue

        value_um = _standard_value_to_micromolar(standard_value, standard_units)
        if value_um is None:
            continue

        assay = assay_by_id.get(int(assay_id))
        if not assay:
            continue

        target = target_by_tid.get(int(assay["tid"]))
        if not target:
            continue

        target_name = (target.get("pref_name") or "").strip()
        target_type = (target.get("target_type") or "").strip().upper()
        if "CYP" not in target_name.upper() and "CYTOCHROME P450" not in target_name.upper():
            continue
        if target_type and target_type != "SINGLE PROTEIN":
            continue

        mol = mols.get(int(molregno))
        if not mol:
            continue

        drug = _drug_name(mol)
        chembl_id = mol.get("chembl_id", "")
        potency = _cyp_potency_label(value_um)
        formatted_value = f"{float(standard_value):g} {standard_units} (~{value_um:.3g} µM)"

        dedupe_key = (int(molregno), int(assay_id), standard_type.upper(), round(value_um, 6))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        yield {
            "text": (
                f"### Question\nHow potent is {drug} as a {target_name} inhibitor?\n\n"
                f"### Answer\n{drug} ({chembl_id}) inhibits {target_name} with {standard_type} "
                f"{formatted_value}. This is a {potency} inhibitor by ChEMBL potency bins "
                f"(<1 µM strong, 1-10 µM moderate, >10 µM weak)."
            )
        }

        yield {
            "text": (
                f"### Question\nWhat is the interaction risk of combining {drug} with a "
                f"{target_name} substrate?\n\n"
                f"### Answer\n{drug} ({chembl_id}) is a {potency} {target_name} inhibitor "
                f"based on a {standard_type} of {formatted_value}. Co-administering it with "
                f"a {target_name} substrate may increase substrate exposure and toxicity risk."
            )
        }


def generate_ddi_qa(
    metabolism: pl.DataFrame,
    molecule_dict: pl.DataFrame,
    compound_records: pl.DataFrame,
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

    for enzyme, drug_list in enzyme_drugs.items():
        unique_drugs = list(set(drug_list))
        if len(unique_drugs) < 2:
            continue
        rng.shuffle(unique_drugs)

        for i in range(len(unique_drugs)):
            for j in range(i + 1, len(unique_drugs)):
                pair_key: frozenset = frozenset([unique_drugs[i], unique_drugs[j]])
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)

                mol_a = mols[unique_drugs[i]]
                mol_b = mols[unique_drugs[j]]
                name_a, id_a = _drug_name(mol_a), mol_a.get("chembl_id", "")
                name_b, id_b = _drug_name(mol_b), mol_b.get("chembl_id", "")

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
                        f"### Answer\nUse caution. {shared_pathway} "
                        f"Monitor plasma levels if both drugs are prescribed together."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nWhat is the drug interaction between {name_a} and {name_b}?\n\n"
                        f"### Answer\n{name_a} and {name_b} share the {enzyme} metabolic pathway. "
                        f"Co-administration creates competition for {enzyme}, which may increase "
                        f"or decrease plasma levels of one or both drugs."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nTell me about the interaction between {name_a} and {name_b}.\n\n"
                        f"### Answer\n{shared_pathway}"
                    )
                }
                yield {
                    "text": (
                        f"### Question\nWhat happens if I take {name_a} and {name_b} together?\n\n"
                        f"### Answer\nTaking {name_a} and {name_b} together can affect how each drug "
                        f"is processed by the body. {shared_pathway}"
                    )
                }
                yield {
                    "text": (
                        f"### Question\nIs it safe to combine {name_a} with {name_b}?\n\n"
                        f"### Answer\nThis combination requires care. {shared_pathway} "
                        f"Dose adjustment may be needed."
                    )
                }
                yield {
                    "text": (
                        f"### Question\nDoes {name_a} interact with {name_b}?\n\n"
                        f"### Answer\nYes — {name_a} ({id_a}) and {name_b} ({id_b}) both rely on "
                        f"{enzyme} for metabolism. Taking them together may alter plasma levels "
                        f"of either drug."
                    )
                }


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
    for act_row in activities.to_dicts():
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


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


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
) -> None:
    """
    Build QA-formatted JSONL training data for a drug-interaction chatbot.

    Reads ChEMBL parquet files from data_dir and writes:
      output_dir/train.jsonl  (90 % of pairs)
      output_dir/valid.jsonl  (10 % of pairs)

    Args:
        data_dir:   Directory containing the ChEMBL parquet files.
        output_dir: Directory to write train.jsonl / valid.jsonl.
        row_limit:  Cap every table at this many rows on load.  Useful on
                    memory-constrained machines (e.g. ``--row-limit 200000``).
                    ``None`` (default) loads all rows.
    """
    print("Loading ChEMBL tables...")
    tables = load_tables(data_dir, row_limit=row_limit)

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
            "greetings & capabilities",
            generate_greeting_qa,
            True,  # no external data needed
        ),
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
            "CYP inhibition",
            lambda: generate_cyp_inhibition_qa(
                tables["activities"], tables["assays"], tables["target_dictionary"], mol
            ),
            "activities" in tables and "assays" in tables and "target_dictionary" in tables,
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
            lambda: generate_approved_product_qa(tables["formulations"], tables["products"], mol),
            "formulations" in tables and "products" in tables,
        ),
        (
            "scientific literature",
            lambda: generate_literature_qa(tables["docs"]),
            "docs" in tables,
        ),
        (
            "assay context",
            lambda: generate_assay_context_qa(tables["assays"], tables["activities"], mol),
            "assays" in tables and "activities" in tables,
        ),
        (
            "ligand efficiency",
            lambda: generate_ligand_efficiency_qa(tables["ligand_eff"], tables["activities"], mol),
            "ligand_eff" in tables and "activities" in tables,
        ),
        (
            "protein target sequences",
            lambda: generate_target_sequence_qa(
                tables["component_sequences"],
                tables["target_components"],
                tables["target_dictionary"],
            ),
            "component_sequences" in tables
            and "target_components" in tables
            and "target_dictionary" in tables,
        ),
        (
            "protein family",
            lambda: generate_protein_family_qa(
                tables["protein_classification"],
                tables["component_class"],
                tables["target_components"],
                tables["target_dictionary"],
            ),
            "protein_classification" in tables
            and "component_class" in tables
            and "target_components" in tables
            and "target_dictionary" in tables,
        ),
        (
            "biotherapeutics",
            lambda: generate_biotherapeutic_qa(tables["biotherapeutics"], mol),
            "biotherapeutics" in tables,
        ),
        (
            "target relations",
            lambda: generate_target_relations_qa(
                tables["target_relations"], tables["target_dictionary"]
            ),
            "target_relations" in tables and "target_dictionary" in tables,
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
    args = parser.parse_args()
    build_drug_interaction_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        row_limit=args.row_limit,
    )
