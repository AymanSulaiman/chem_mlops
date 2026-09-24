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
import inspect
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
            # The caller may have routed a SMILES the user typed into `name`.
            if Chem.MolFromSmiles(name) is not None:
                smiles = name
            else:
                raise ValueError(f"No ChEMBL compound named '{name}' — check the spelling.")
    elif smiles and Chem.MolFromSmiles(smiles) is None:
        record = get_compound_by_name(smiles)  # a drug name in the smiles field
        if record is None:
            raise ValueError(f"Invalid SMILES — could not parse: '{smiles}'")

    # Biologics are now in the compounds table — reachable by name for their
    # mechanism and target data, but there is no structure to draw. Say so,
    # rather than failing on the string "None".
    if record is not None and not record.get("canonical_smiles"):
        raise ValueError(
            f"{record.get('pref_name') or name} has no small-molecule structure "
            "(it is a biologic), so there is nothing to draw."
        )

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
        # Carry the facts a caption would quote. Without them the model invents
        # a formula to put next to the picture.
        for field in ("full_molformula", "mw_freebase"):
            if record.get(field) is not None:
                result[field] = str(record[field])
    return result


TOOLS: dict[str, Callable[..., Any]] = {
    "query_compounds": query_compounds,
    "get_compound_by_name": get_compound_by_name,
    "query_polypharmacy": query_polypharmacy,
    "query_drug_side_effects": query_drug_side_effects,
    "draw_molecule": draw_molecule,
}

# Near-miss names the fine-tune invents. Measured on the item 3 benchmark:
# 11 of 40 calls named a tool that does not exist (query_compound, query_drugs)
# or picked query_compounds — a SMILES similarity search — for a lookup by drug
# name. The intent was right and the spelling was not, so mapping them recovers
# the call instead of returning an error the model has to recover from.
# Delete an entry once a trained model stops emitting it.
TOOL_ALIASES: dict[str, str] = {
    "draw_smiles": "draw_molecule",  # old name, kept so a stale prompt still works
    "draw": "draw_molecule",
    "query_compound": "get_compound_by_name",
    "query_drug": "get_compound_by_name",
    "query_drugs": "get_compound_by_name",
    "get_compound": "get_compound_by_name",
    "get_drug_by_name": "get_compound_by_name",
    "compound_by_name": "get_compound_by_name",
    "query_compound_by_name": "get_compound_by_name",
    "query_side_effects": "query_drug_side_effects",
    "query_interactions": "query_drug_side_effects",
    "query_drug_interactions": "query_drug_side_effects",
}

# What each tool calls its primary argument, against the keys a model reaches
# for instead. Same failure as the names: right intent, wrong spelling.
ARG_ALIASES: dict[str, dict[str, str]] = {
    "get_compound_by_name": {
        "drug_name": "name",
        "compound_name": "name",
        "compound": "name",
        "drug": "name",
    },
    "draw_molecule": {
        "drug_name": "name",
        "compound_name": "name",
        "compound": "name",
        "drug": "name",
        "molecule": "name",
    },
    "query_drug_side_effects": {"name": "drug_name", "drug": "drug_name", "compound": "drug_name"},
    "query_polypharmacy": {
        "drug_a": "drug_1",
        "drug_b": "drug_2",
        "drug1": "drug_1",
        "drug2": "drug_2",
    },
    "query_compounds": {"smiles_string": "smiles", "structure": "smiles"},
}

_NAME_KEYS = ("name", "drug_name", "compound_name", "compound", "drug")


def resolve_tool_call(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Map a near-miss tool name and argument keys onto the real registry.

    Mirrored by ``resolveToolCall`` in ``web/src/tools.ts`` — the serving path
    and the benchmark have to agree, or the measured rate is not the rate users
    get. Unknown names that match nothing are returned untouched, so the caller
    still reports them as unknown.

    Returns:
        (tool name, arguments with aliased keys renamed).
    """
    tool = TOOL_ALIASES.get(name, name)

    # A similarity search needs a SMILES string. Handed a drug name instead,
    # the model meant the by-name lookup: 4 of 40 benchmark calls did this.
    if tool == "query_compounds" and "smiles" not in args:
        if any(key in args for key in _NAME_KEYS):
            tool = "get_compound_by_name"

    renames = ARG_ALIASES.get(tool, {})
    resolved = {k: v for k, v in args.items() if k not in renames}
    for key, value in args.items():
        # A correctly named key already present wins over the aliased spelling.
        if key in renames and renames[key] not in resolved:
            resolved[renames[key]] = value
    return tool, resolved


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool call, returning ``{"result": ...}`` or ``{"error": ...}``.

    The name and argument keys go through :func:`resolve_tool_call` first, so a
    near-miss spelling runs instead of erroring.

    Every failure — unknown tool, bad arguments, missing table, invalid SMILES —
    comes back as an ``error`` string, because the caller feeds it to a model
    that can retry rather than to a human reading a traceback.
    """
    name, args = resolve_tool_call(name, args)
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(sorted(TOOLS))}"}
    # Models pad calls with arguments the tool does not take ("n": 10 on a
    # single-record lookup). Dropping them beats a TypeError the model then has
    # to interpret; a *missing* required argument still raises and is reported.
    accepted = inspect.signature(fn).parameters
    args = {k: v for k, v in args.items() if k in accepted}
    try:
        return {"result": fn(**args)}
    except Exception as exc:  # noqa: BLE001 — the model is the error handler here
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str]) -> dict[str, Any]:
    if len(argv) < 2:
        return {
            "error": f"Usage: tools <tool_name> '<json args>'. Tools: {', '.join(sorted(TOOLS))}"
        }
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
