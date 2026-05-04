# ChEMBL LanceDB Query Interface
# Provides similarity search and exact lookup against the ingested compounds table.

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
    _FP_GEN,
    _smiles_to_fp,
)


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


def _open_table(lancedb_dir: str) -> Table:
    """Connect to the latest ChEMBL LanceDB and return the compounds table."""
    uri: str = _resolve_lancedb_uri(lancedb_dir)
    db: DBConnection = lancedb.connect(uri)
    if COMPOUNDS_TABLE not in db.list_tables().tables:
        raise FileNotFoundError(
            f"Table '{COMPOUNDS_TABLE}' not found in '{uri}'. "
            "Run ingest_compounds_to_lancedb() first."
        )
    return db.open_table(COMPOUNDS_TABLE)


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
    table: Table = _open_table(lancedb_dir)
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
    table: Table = _open_table(lancedb_dir)
    rows: list[dict[str, Any]] = (
        table.search().where(f"chembl_id = '{chembl_id}'").limit(1).to_list()
    )
    if not rows:
        return None
    row = rows[0]
    row.pop("vector", None)
    return row


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

    print("\n── All checks passed ✓ ──────────────────────────────────────")


if __name__ == "__main__":
    _run_sanity_check()
