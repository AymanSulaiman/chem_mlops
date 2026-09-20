"""Tests for app/scripts/flows/vector_store/tools.py."""

from __future__ import annotations

import base64

import pytest

from app.scripts.flows.vector_store.tools import (
    ARG_ALIASES,
    TOOL_ALIASES,
    TOOLS,
    draw_molecule,
    main,
    resolve_tool_call,
    run_tool,
)


def test_tools_registry_covers_the_agent_tools() -> None:
    assert set(TOOLS) == {
        "query_compounds",
        "get_compound_by_name",
        "query_polypharmacy",
        "query_drug_side_effects",
        "draw_molecule",
    }


def test_every_alias_points_at_a_real_tool() -> None:
    assert set(TOOL_ALIASES.values()) <= set(TOOLS)
    assert set(ARG_ALIASES) <= set(TOOLS)
    # An alias that shadows a real tool would silently reroute a correct call.
    assert not set(TOOL_ALIASES) & set(TOOLS)


def test_resolve_maps_the_names_the_finetune_actually_emits() -> None:
    # Measured on the item 3 benchmark: 11 of 40 calls missed by a near-miss
    # name while the intent was unambiguous.
    assert resolve_tool_call("query_compound", {"drug_name": "SIROLIMUS"}) == (
        "get_compound_by_name",
        {"name": "SIROLIMUS"},
    )
    assert resolve_tool_call("draw_smiles", {"name": "Aspirin"})[0] == "draw_molecule"


def test_a_name_handed_to_the_similarity_search_is_a_lookup() -> None:
    # query_compounds needs a SMILES; given a drug name it cannot be what the
    # model meant, so the call is rerouted rather than run on bad input.
    assert resolve_tool_call("query_compounds", {"drug_name": "SIROLIMUS", "n": 10}) == (
        "get_compound_by_name",
        {"n": 10, "name": "SIROLIMUS"},
    )
    # A real similarity search is left alone.
    assert resolve_tool_call("query_compounds", {"smiles": "CCO"}) == (
        "query_compounds",
        {"smiles": "CCO"},
    )


def test_resolve_leaves_correct_calls_and_unknown_names_untouched() -> None:
    assert resolve_tool_call("get_compound_by_name", {"name": "Aspirin"}) == (
        "get_compound_by_name",
        {"name": "Aspirin"},
    )
    # Nothing to map onto: the caller still gets to report it as unknown.
    assert resolve_tool_call("rm_rf", {})[0] == "rm_rf"


def test_a_correct_key_wins_over_an_aliased_one() -> None:
    _, args = resolve_tool_call("get_compound_by_name", {"name": "Aspirin", "drug_name": "Ibu"})
    assert args == {"name": "Aspirin"}


def test_draw_molecule_returns_a_png_data_url() -> None:
    result = draw_molecule(smiles="CCO", size=64)
    prefix = "data:image/png;base64,"
    assert result["image"].startswith(prefix)
    assert base64.b64decode(result["image"][len(prefix) :])[:4] == b"\x89PNG"


def test_run_tool_reports_failures_as_errors_not_exceptions() -> None:
    # Unknown tool, bad arguments and invalid input all come back as strings,
    # because the caller feeds them to a model that can retry.
    assert "Unknown tool" in run_tool("rm_rf", {})["error"]
    # A *missing* required argument still fails loudly...
    assert "TypeError" in run_tool("query_polypharmacy", {})["error"]


def test_run_tool_drops_arguments_the_tool_does_not_take() -> None:
    # ...but padding the call ("n": 10 on a single-record lookup) is the model
    # being chatty, not a failure. Dropping beats a TypeError it has to parse.
    assert "TypeError" not in run_tool("draw_molecule", {"smiles": "CCO", "n": 10})


def test_main_parses_argv() -> None:
    assert "not valid JSON" in main(["tools", "draw_molecule", "{nope}"])["error"]
    assert "must be a JSON object" in main(["tools", "draw_molecule", "[1, 2]"])["error"]
    assert "Usage" in main(["tools"])["error"]
    assert main(["tools", "draw_molecule", '{"smiles": "CCO", "size": 64}'])["result"]["image"]


def test_draw_molecule_prefers_the_chembl_structure_over_a_supplied_one(monkeypatch) -> None:
    """A name is resolved in ChEMBL; a name mistakenly sent as SMILES is too."""
    aspirin = {
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "chembl_id": "CHEMBL25",
        "pref_name": "ASPIRIN",
    }
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


def test_alias_tables_match_the_typescript_agent() -> None:
    """The serving path is TypeScript; the benchmark is Python. They must agree.

    If they drift, the tool-call rate in the eval is not the rate users get —
    which is the whole reason the benchmark exists.
    """
    import re
    from pathlib import Path

    source = Path(__file__).parents[3] / "web" / "src" / "tools.ts"
    block = re.search(r"const TOOL_ALIASES[^{]*\{(.*?)\n\};", source.read_text(), re.S)
    assert block, "TOOL_ALIASES not found in web/src/tools.ts"

    ts_aliases = dict(re.findall(r'(\w+):\s*"([^"]+)"', block.group(1)))
    assert ts_aliases == TOOL_ALIASES
