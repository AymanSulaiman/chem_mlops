"""Tool bridge — exposes the LanceDB query functions to the web agent loop.

The web server (Bun/TypeScript) spawns this module once per tool call:

    uv run python -m app.scripts.flows.vector_store.tools \
        get_compound_by_name '{"name": "Aspirin"}'

Prints one JSON object on stdout: ``{"result": ...}`` or ``{"error": "..."}``.

# ponytail: one subprocess per call — RDKit/LanceDB import is ~1-2s. If tool
# latency matters, turn this into a stdin/stdout worker loop reading one JSON
# request per line; the TOOLS dispatch below does not change.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections.abc import Callable
from typing import Any

from app.scripts.flows.vector_store.query_lancedb import (
    get_compound_by_name,
    query_compounds,
    query_drug_side_effects,
    query_polypharmacy,
)


def draw_molecule(
    name: str | None = None,
    smiles: str | None = None,
    size: int = 320,
) -> dict[str, str]:
    """Draw a molecule, preferring ChEMBL's structure over a supplied SMILES.

    Models invent SMILES strings — asked for ibuprofen they produce something
    that parses but is the wrong molecule. So *name* is the primary argument:
    the structure then comes from the ``compounds`` table. A *smiles* that does
    not parse is retried as a name, because models put drug names in that field.

    Args:
        name: Drug name, e.g. ``"Ibuprofen"``. Looked up in ChEMBL.
        smiles: Structure supplied by the user. Only used when *name* is absent.
        size: Image edge length in pixels (default 320).

    Returns:
        ``{"image": "data:image/png;base64,...", "smiles": ..., "source": ...}``
        plus ``chembl_id`` and ``pref_name`` when the molecule came from ChEMBL.

    Raises:
        ValueError: If neither argument yields a molecule RDKit can parse.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    record: dict[str, Any] | None = None
    if name:
        record = get_compound_by_name(name)
        if record is None:
            raise ValueError(f"No ChEMBL compound named '{name}' — check the spelling.")
    elif smiles and Chem.MolFromSmiles(smiles) is None:
        record = get_compound_by_name(smiles)  # a drug name in the smiles field
        if record is None:
            raise ValueError(f"Invalid SMILES — could not parse: '{smiles}'")

    drawn = str(record["canonical_smiles"]) if record else (smiles or "")
    mol = Chem.MolFromSmiles(drawn)
    if mol is None:
        raise ValueError(f"Invalid SMILES — could not parse: '{drawn}'")

    buf = io.BytesIO()
    Draw.MolToImage(mol, size=(size, size)).save(buf, format="PNG")
    result = {
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "smiles": drawn,
        "source": "ChEMBL" if record else "supplied by the user",
    }
    if record:
        result["chembl_id"] = str(record["chembl_id"])
        result["pref_name"] = str(record["pref_name"])
    return result


TOOLS: dict[str, Callable[..., Any]] = {
    "query_compounds": query_compounds,
    "get_compound_by_name": get_compound_by_name,
    "query_polypharmacy": query_polypharmacy,
    "query_drug_side_effects": query_drug_side_effects,
    "draw_molecule": draw_molecule,
    "draw_smiles": draw_molecule,  # old name, kept so a stale prompt still works
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool call, returning ``{"result": ...}`` or ``{"error": ...}``.

    Every failure — unknown tool, bad arguments, missing table, invalid SMILES —
    comes back as an ``error`` string, because the caller feeds it to a model
    that can retry rather than to a human reading a traceback.
    """
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(sorted(TOOLS))}"}
    try:
        return {"result": fn(**args)}
    except Exception as exc:  # noqa: BLE001 — the model is the error handler here
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str]) -> dict[str, Any]:
    if len(argv) < 2:
        return {"error": f"Usage: tools <tool_name> '<json args>'. Tools: {', '.join(sorted(TOOLS))}"}
    try:
        args = json.loads(argv[2]) if len(argv) > 2 else {}
    except json.JSONDecodeError as exc:
        return {"error": f"Arguments are not valid JSON: {exc}"}
    if not isinstance(args, dict):
        return {"error": "Arguments must be a JSON object."}
    return run_tool(argv[1], args)


if __name__ == "__main__":
    # default=str: LanceDB rows carry numpy scalars that json cannot encode.
    print(json.dumps(main(sys.argv), default=str))
