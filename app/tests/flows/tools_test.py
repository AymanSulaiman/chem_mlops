"""Tests for app/scripts/flows/vector_store/tools.py."""

from __future__ import annotations

import base64

import pytest

from app.scripts.flows.vector_store.tools import TOOLS, draw_molecule, main, run_tool


def test_tools_registry_covers_the_agent_tools() -> None:
    assert set(TOOLS) == {
        "query_compounds",
        "get_compound_by_name",
        "query_polypharmacy",
        "query_drug_side_effects",
        "draw_molecule",
        "draw_smiles",  # alias for draw_molecule
    }
    assert TOOLS["draw_smiles"] is TOOLS["draw_molecule"]


def test_draw_molecule_returns_a_png_data_url() -> None:
    result = draw_molecule(smiles="CCO", size=64)
    prefix = "data:image/png;base64,"
    assert result["image"].startswith(prefix)
    assert base64.b64decode(result["image"][len(prefix) :])[:4] == b"\x89PNG"


def test_run_tool_reports_failures_as_errors_not_exceptions() -> None:
    # Unknown tool, bad arguments and invalid input all come back as strings,
    # because the caller feeds them to a model that can retry.
    assert "Unknown tool" in run_tool("rm_rf", {})["error"]
    assert "TypeError" in run_tool("draw_molecule", {"wrong_kwarg": 1})["error"]


def test_main_parses_argv() -> None:
    assert "not valid JSON" in main(["tools", "draw_molecule", "{nope}"])["error"]
    assert "must be a JSON object" in main(["tools", "draw_molecule", "[1, 2]"])["error"]
    assert "Usage" in main(["tools"])["error"]
    assert main(["tools", "draw_molecule", '{"smiles": "CCO", "size": 64}'])["result"]["image"]


def test_draw_molecule_prefers_the_chembl_structure_over_a_supplied_one(monkeypatch) -> None:
    """A name is resolved in ChEMBL; a name mistakenly sent as SMILES is too."""
    aspirin = {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "chembl_id": "CHEMBL25", "pref_name": "ASPIRIN"}
    monkeypatch.setattr(
        "app.scripts.flows.vector_store.tools.get_compound_by_name",
        lambda n: aspirin if n.strip().lower() in {"aspirin", "acetylsalicylic acid"} else None,
    )

    by_name = draw_molecule(name="Aspirin", size=64)
    assert by_name["smiles"] == aspirin["canonical_smiles"]
    assert by_name["source"] == "ChEMBL"
    assert by_name["chembl_id"] == "CHEMBL25"

    # Models put drug names in the smiles field; that is retried as a lookup.
    assert draw_molecule(smiles="Aspirin", size=64)["smiles"] == aspirin["canonical_smiles"]

    # A user-supplied structure is drawn as given, and flagged as unverified.
    supplied = draw_molecule(smiles="CCO", size=64)
    assert supplied["smiles"] == "CCO"
    assert supplied["source"] == "supplied by the user"

    with pytest.raises(ValueError, match="No ChEMBL compound named"):
        draw_molecule(name="Notadrug")
    with pytest.raises(ValueError, match="Invalid SMILES"):
        draw_molecule(smiles="zzz")
