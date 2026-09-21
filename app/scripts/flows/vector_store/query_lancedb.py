# ChEMBL LanceDB Query Interface
# Provides similarity search and exact lookup against the ingested compounds table,
# and polypharmacy side-effect lookup against the TWOSIDES polypharmacy table.

from __future__ import annotations

import os
from typing import Any

import lancedb
import numpy as np
from lancedb.db import DBConnection
from lancedb.table import Table

from app.scripts.flows.vector_store.ingest_to_lancedb import (
    COMPOUNDS_TABLE,
    LANCEDB_DIR,
    _smiles_to_fp,
)
from app.scripts.flows.vector_store.ingest_twosides_to_lancedb import POLYPHARMACY_TABLE

# ── Private helpers ───────────────────────────────────────────────────────────


def _resolve_lancedb_uri(lancedb_dir: str) -> str:
    """Return the path of the latest chembl_CHEMBL_* subdirectory.

    Scans *lancedb_dir* for directories whose names start with ``chembl_CHEMBL``
    and returns the lexicographically last one (e.g. ``chembl_CHEMBL_36``).

    Raises:
        FileNotFoundError: If no matching subdirectory exists.
    """
    if not os.path.isdir(lancedb_dir):
        raise FileNotFoundError(
            f"LanceDB directory not found: '{lancedb_dir}'. "
            "Run ingest_compounds_to_lancedb() first."
        )
    candidates: list[str] = sorted(
        d
        for d in os.listdir(lancedb_dir)
        if d.startswith("chembl_CHEMBL") and os.path.isdir(os.path.join(lancedb_dir, d))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No chembl_CHEMBL_* subdirectory found in '{lancedb_dir}'. "
            "Run ingest_compounds_to_lancedb() first."
        )
    return os.path.join(lancedb_dir, candidates[-1])


def _open_table(lancedb_dir: str, table_name: str) -> Table:
    """Connect to the latest ChEMBL LanceDB and return the named table."""
    uri: str = _resolve_lancedb_uri(lancedb_dir)
    db: DBConnection = lancedb.connect(uri)
    if table_name not in db.list_tables().tables:
        raise FileNotFoundError(
            f"Table '{table_name}' not found in '{uri}'. Run the appropriate ingest step first."
        )
    return db.open_table(table_name)


def _smiles_to_query_vector(smiles: str) -> list[float]:
    """Convert a SMILES string to a query vector, raising on invalid input."""
    if not smiles.strip():
        raise ValueError(f"Invalid SMILES — could not parse: '{smiles}'")
    fp: np.ndarray | None = _smiles_to_fp(smiles)
    if fp is None:
        raise ValueError(f"Invalid SMILES — could not parse: '{smiles}'")
    return fp.tolist()


# ── Public API ────────────────────────────────────────────────────────────────


def query_compounds(
    smiles: str,
    n: int = 5,
    lancedb_dir: str = LANCEDB_DIR,
) -> list[dict[str, Any]]:
    """Find the *n* most similar compounds by Morgan fingerprint similarity.

    Computes a 2048-bit ECFP4 fingerprint from *smiles* and runs an
    approximate nearest-neighbour search against the LanceDB compounds table.
    The ``vector`` column is dropped from each returned record.

    Args:
        smiles: Query molecule as a SMILES string.
        n: Number of results to return (default 5).
        lancedb_dir: Root directory that contains the ``chembl_CHEMBL_*``
            subdirectory (default ``data/lancedb``).

    Molecules with no parsable structure — biologics, mostly — are stored with a
    zero vector so the name-lookup tools can reach their mechanism and target
    data. They are filtered out here: a zero vector sits at a fixed distance
    from every query and would otherwise surface as a spurious "similar"
    compound to an antibody.

    Returns:
        List of compound dicts ordered by descending similarity, each
        containing all metadata columns plus a ``_distance`` field.

    Raises:
        ValueError: If *smiles* cannot be parsed by RDKit.
        FileNotFoundError: If the LanceDB table does not exist.
    """
    query_vector: list[float] = _smiles_to_query_vector(smiles)
    table: Table = _open_table(lancedb_dir, COMPOUNDS_TABLE)
    results: list[dict[str, Any]] = (
        table.search(query_vector).where("has_structure = true").limit(n).to_list()
    )
    # Drop the raw vector column — callers need metadata, not the 2048-float blob
    for row in results:
        row.pop("vector", None)
    return results


def get_compound(
    chembl_id: str,
    lancedb_dir: str = LANCEDB_DIR,
) -> dict[str, Any] | None:
    """Exact lookup by ChEMBL ID.

    Uses the scalar index on ``chembl_id`` for a fast filtered search.

    Args:
        chembl_id: ChEMBL identifier string, e.g. ``"CHEMBL25"``.
        lancedb_dir: Root directory that contains the ``chembl_CHEMBL_*``
            subdirectory (default ``data/lancedb``).

    Returns:
        A compact compound dict — the fields in COMPOUND_SUMMARY_FIELDS that
        are present — or ``None`` if no matching row is found. Use
        :func:`get_compound` for the whole row.

    Raises:
        FileNotFoundError: If the LanceDB table does not exist.
    """
    table: Table = _open_table(lancedb_dir, COMPOUNDS_TABLE)
    rows: list[dict[str, Any]] = (
        table.search().where(f"chembl_id = '{chembl_id}'").limit(1).to_list()
    )
    if not rows:
        return None
    row = rows[0]
    # Projected, not the full 75-column row — see COMPOUND_SUMMARY_FIELDS.
    return {k: row[k] for k in COMPOUND_SUMMARY_FIELDS if row.get(k) is not None}


# What a name lookup returns. The compounds table has 75 columns; a model asked
# "what does X target?" had to read past molregno, max_phase, therapeutic_flag
# and twenty more before reaching mechanism_targets, and mostly did not — it
# re-emitted its tool call instead of answering, failing 20 of 40 golden
# questions. Handed these fields alone it answered every one. The training
# records use this shape too, so projecting here also closes a train-serve gap:
# the model is taught on a compact result and was being served a 14 KB blob
# truncated mid-record.
#
# Ordered deliberately: identity, then what the drug does, then what it is.
# get_compound() still returns the whole row for programmatic callers.
#
# Fields are omitted as carefully as they are included. `indications`,
# `max_phase`, `first_approval` and `has_structure` were in an earlier version
# and each one cost answers: asked what UNASNEMAB targets the model replied
# "Spinal Cord Injuries" (its indication), and asked about AMG-517 it replied
# "the compound does not have a structure assigned" (has_structure). A field the
# training records never carry is a distractor, not context. Indication lookups
# want their own converted training records, not a wider result here.
COMPOUND_SUMMARY_FIELDS: tuple[str, ...] = (
    "chembl_id",
    "pref_name",
    "mechanism_targets",
    "mechanisms",
    "action_types",
    "mw_freebase",
    "full_molformula",
    "alogp",
    "canonical_smiles",
)


def get_compound_by_name(
    name: str,
    lancedb_dir: str = LANCEDB_DIR,
) -> dict[str, Any] | None:
    """Lookup by preferred name, falling back to trade names and synonyms.

    ChEMBL's ``pref_name`` is not always the name people use: paracetamol is
    stored as ACETAMINOPHEN, and Tylenol only appears in ``synonyms``. The
    fallback scans that column and accepts a row only when the query matches a
    whole synonym, so "Codeine" cannot match "Codeine component of ...".

    Args:
        name: Drug name, e.g. ``"Aspirin"``, ``"Paracetamol"``, ``"Tylenol"``.
        lancedb_dir: Root directory that contains the ``chembl_CHEMBL_*``
            subdirectory (default ``data/lancedb``).

    Returns:
        A compact compound dict — the fields in COMPOUND_SUMMARY_FIELDS that
        are present — or ``None`` if no matching row is found. Use
        :func:`get_compound` for the whole row.

    Raises:
        FileNotFoundError: If the LanceDB table does not exist.
    """
    table: Table = _open_table(lancedb_dir, COMPOUNDS_TABLE)
    safe = name.strip().replace("'", "''")
    rows: list[dict[str, Any]] = (
        table.search().where(f"LOWER(pref_name) = '{safe.lower()}'").limit(1).to_list()
    )
    if not rows:
        rows = _search_synonyms(table, safe)
    if not rows:
        return None
    row = rows[0]
    # Projected, not the full 75-column row — see COMPOUND_SUMMARY_FIELDS.
    return {k: row[k] for k in COMPOUND_SUMMARY_FIELDS if row.get(k) is not None}


def _search_synonyms(table: Table, safe_name: str) -> list[dict[str, Any]]:
    """Rows whose ``synonyms`` list contains *safe_name* as a whole entry."""
    wanted = safe_name.lower()
    # LIKE narrows 2.8M rows to a handful (~0.1s); the exact check happens here,
    # because LIKE '%codeine%' also hits "Codeine component of ..." entries.
    like = wanted.replace("%", "").replace("_", "")
    candidates: list[dict[str, Any]] = (
        table.search().where(f"LOWER(synonyms) LIKE '%{like}%'").limit(25).to_list()
    )
    return [
        row
        for row in candidates
        if wanted in {s.strip().lower() for s in (row.get("synonyms") or "").split(";")}
    ]


def query_polypharmacy(
    drug_1: str,
    drug_2: str,
    lancedb_dir: str = LANCEDB_DIR,
) -> dict[str, Any] | None:
    """Look up polypharmacy side-effect signals for a specific drug pair.

    Drug name matching is case-insensitive and checks both orderings, since
    TWOSIDES does not guarantee a canonical (drug_1, drug_2) order.

    Args:
        drug_1: Name of the first drug (e.g. ``"Warfarin"``).
        drug_2: Name of the second drug (e.g. ``"Aspirin"``).
        lancedb_dir: Root LanceDB directory (default ``data/lancedb``).

    Returns:
        A dict with ``side_effects``, ``max_prr``, ``total_cases``, etc.,
        or ``None`` if the pair has no TWOSIDES signal above the ingestion thresholds.

    Raises:
        FileNotFoundError: If the polypharmacy table has not been ingested yet.
    """
    table: Table = _open_table(lancedb_dir, POLYPHARMACY_TABLE)
    d1 = drug_1.strip().title()
    d2 = drug_2.strip().title()
    rows: list[dict[str, Any]] = (
        table.search()
        .where(
            f"(drug_1_name = '{d1}' AND drug_2_name = '{d2}') OR "
            f"(drug_1_name = '{d2}' AND drug_2_name = '{d1}')"
        )
        .limit(1)
        .to_list()
    )
    return rows[0] if rows else None


def query_drug_side_effects(
    drug_name: str,
    n: int = 20,
    lancedb_dir: str = LANCEDB_DIR,
) -> list[dict[str, Any]]:
    """Find all known polypharmacy signals involving a given drug.

    Returns all drug pairs in the TWOSIDES table where *drug_name* appears as
    either drug_1 or drug_2, ordered by descending max PRR (strongest signals first).

    Args:
        drug_name: Drug name to search for (e.g. ``"Sildenafil"``).
        n: Maximum number of pairs to return (default 20).
        lancedb_dir: Root LanceDB directory (default ``data/lancedb``).

    Returns:
        List of polypharmacy dicts ordered by descending ``max_prr``.
        Each dict includes the partner drug name, aggregated side effects, and signal stats.

    Raises:
        FileNotFoundError: If the polypharmacy table has not been ingested yet.
    """
    table: Table = _open_table(lancedb_dir, POLYPHARMACY_TABLE)
    name = drug_name.strip().title()
    rows: list[dict[str, Any]] = (
        table.search().where(f"drug_1_name = '{name}' OR drug_2_name = '{name}'").limit(n).to_list()
    )
    return sorted(rows, key=lambda r: r.get("max_prr", 0), reverse=True)


# ── Internal self-check ───────────────────────────────────────────────────────


def _run_sanity_check(lancedb_dir: str = LANCEDB_DIR) -> None:
    """Query the live vector store with known molecules and print a summary.

    Checks:
    1. Aspirin (CHEMBL25) is the top similarity hit for its own SMILES.
    2. Exact lookup by ChEMBL ID returns the expected preferred name.
    3. An invalid SMILES raises ValueError.
    4. An unknown ChEMBL ID returns None.
    """
    ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # noqa: N806
    ASPIRIN_ID = "CHEMBL25"  # noqa: N806

    print("── Sanity check: ChEMBL LanceDB ─────────────────────────────")

    # Check 1: similarity search returns aspirin at rank 1
    print(f"\n[1] Similarity search for aspirin ({ASPIRIN_ID})...")
    hits = query_compounds(ASPIRIN_SMILES, n=5, lancedb_dir=lancedb_dir)
    assert hits, "No results returned from similarity search"
    top = hits[0]
    top_id: str = str(top.get("chembl_id", ""))
    top_name: str = str(top.get("pref_name", ""))
    print(f"    Top hit: {top_id} ({top_name})  _distance={top.get('_distance', '?'):.6f}")
    assert top_id == ASPIRIN_ID, f"Expected top hit to be {ASPIRIN_ID}, got {top_id}"
    print("    ✓ Correct top hit")

    # Check 2: exact lookup
    print(f"\n[2] Exact lookup: {ASPIRIN_ID}...")
    record = get_compound(ASPIRIN_ID, lancedb_dir=lancedb_dir)
    assert record is not None, f"{ASPIRIN_ID} not found via exact lookup"
    print(f"    pref_name={record.get('pref_name')}  mw={record.get('mw_freebase')}")
    assert str(record.get("chembl_id")) == ASPIRIN_ID
    print("    ✓ Exact lookup returned correct record")

    # Check 3: invalid SMILES raises ValueError
    print("\n[3] Invalid SMILES raises ValueError...")
    try:
        query_compounds("not_a_smiles", lancedb_dir=lancedb_dir)
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        print("    ✓ ValueError raised as expected")

    # Check 4: unknown ID returns None
    print("\n[4] Unknown ChEMBL ID returns None...")
    missing = get_compound("CHEMBL_DOES_NOT_EXIST", lancedb_dir=lancedb_dir)
    assert missing is None, f"Expected None, got {missing}"
    print("    ✓ None returned for unknown ID")

    # Checks 5–6: polypharmacy table (skipped gracefully if not yet ingested)
    try:
        print("\n[5] Polypharmacy pair lookup (Temazepam + Sildenafil)...")
        result = query_polypharmacy("Temazepam", "Sildenafil", lancedb_dir=lancedb_dir)
        if result is None:
            print("    — pair not found (may be filtered by PRR threshold)")
        else:
            print(f"    max_prr={result.get('max_prr')}  n_effects={result.get('n_side_effects')}")
            print(f"    side_effects={result.get('side_effects', '')[:80]}...")
            print("    ✓ Polypharmacy pair lookup returned a result")

        print("\n[6] Drug side-effect query (Warfarin)...")
        pairs = query_drug_side_effects("Warfarin", n=5, lancedb_dir=lancedb_dir)
        print(f"    Found {len(pairs)} pair(s) involving Warfarin")
        if pairs:
            top = pairs[0]
            partner = (
                top.get("drug_2_name")
                if top.get("drug_1_name", "").title() == "Warfarin"
                else top.get("drug_1_name")
            )
            print(f"    Strongest signal: Warfarin + {partner}  max_prr={top.get('max_prr')}")
            print("    ✓ Drug side-effect query succeeded")
    except FileNotFoundError:
        print("    — polypharmacy table not ingested yet, skipping checks 5-6")

    print("\n── All checks passed ✓ ──────────────────────────────────────")


if __name__ == "__main__":
    _run_sanity_check()
