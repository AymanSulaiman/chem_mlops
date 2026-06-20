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
            f"Table '{table_name}' not found in '{uri}'. "
            "Run the appropriate ingest step first."
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

    Returns:
        List of compound dicts ordered by descending similarity, each
        containing all metadata columns plus a ``_distance`` field.

    Raises:
        ValueError: If *smiles* cannot be parsed by RDKit.
        FileNotFoundError: If the LanceDB table does not exist.
    """
    query_vector: list[float] = _smiles_to_query_vector(smiles)
    table: Table = _open_table(lancedb_dir, COMPOUNDS_TABLE)
    results: list[dict[str, Any]] = table.search(query_vector).limit(n).to_list()
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
        A single compound dict, or ``None`` if no matching row is found.

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
    row.pop("vector", None)
    return row


def get_compound_by_name(
    name: str,
    lancedb_dir: str = LANCEDB_DIR,
) -> dict[str, Any] | None:
    """Exact lookup by preferred name (case-insensitive).

    Uses a scalar filter on ``pref_name`` for a fast filtered search.

    Args:
        name: Drug preferred name, e.g. ``"Aspirin"``.
        lancedb_dir: Root directory that contains the ``chembl_CHEMBL_*``
            subdirectory (default ``data/lancedb``).

    Returns:
        A single compound dict, or ``None`` if no matching row is found.

    Raises:
        FileNotFoundError: If the LanceDB table does not exist.
    """
    table: Table = _open_table(lancedb_dir, COMPOUNDS_TABLE)
    safe = name.strip().replace("'", "''")
    rows: list[dict[str, Any]] = (
        table.search().where(f"LOWER(pref_name) = '{safe.lower()}'").limit(1).to_list()
    )
    if not rows:
        return None
    row = rows[0]
    row.pop("vector", None)
    return row


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
        table.search()
        .where(f"drug_1_name = '{name}' OR drug_2_name = '{name}'")
        .limit(n)
        .to_list()
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
            partner = top.get("drug_2_name") if top.get("drug_1_name", "").title() == "Warfarin" else top.get("drug_1_name")
            print(f"    Strongest signal: Warfarin + {partner}  max_prr={top.get('max_prr')}")
            print("    ✓ Drug side-effect query succeeded")
    except FileNotFoundError:
        print("    — polypharmacy table not ingested yet, skipping checks 5-6")

    print("\n── All checks passed ✓ ──────────────────────────────────────")


if __name__ == "__main__":
    _run_sanity_check()
