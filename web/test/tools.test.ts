import { expect, test } from "bun:test";

import {
  describeDrawResult,
  formatToolResult,
  parseToolCall,
  routeDirectToolCall,
  TOOL_SPECS,
  TOOL_SYSTEM_PROMPT,
} from "../src/tools";

test("parseToolCall reads a bare tool call", () => {
  expect(parseToolCall(`{"tool": "get_compound_by_name", "args": {"name": "Aspirin"}}`)).toEqual({
    tool: "get_compound_by_name",
    args: { name: "Aspirin" },
  });
});

test("parseToolCall finds the call inside prose and a code fence", () => {
  const text = 'Sure! I will look it up.\n```json\n{"tool": "draw_molecule", "args": {"name": "Ethanol"}}\n```\nDone.';
  expect(parseToolCall(text)).toEqual({ tool: "draw_molecule", args: { name: "Ethanol" } });
});

test("parseToolCall maps the old draw_smiles name onto draw_molecule", () => {
  expect(parseToolCall(`{"tool": "draw_smiles", "args": {"smiles": "CCO"}}`)).toEqual({
    tool: "draw_molecule",
    args: { smiles: "CCO" },
  });
});

test("parseToolCall accepts the OpenAI-shaped name/arguments spelling", () => {
  expect(parseToolCall(`{"name": "query_polypharmacy", "arguments": {"drug_1": "A", "drug_2": "B"}}`)).toEqual({
    tool: "query_polypharmacy",
    args: { drug_1: "A", drug_2: "B" },
  });
});

test("parseToolCall skips JSON that is not a known tool", () => {
  expect(parseToolCall(`{"tool": "rm_rf", "args": {}}`)).toBeNull();
  expect(parseToolCall(`{"answer": "Aspirin is CHEMBL25"}`)).toBeNull();
  expect(parseToolCall("Aspirin inhibits COX-1 and COX-2.")).toBeNull();
});

test("parseToolCall handles braces inside strings and nested args", () => {
  const text = `Note: not {a tool}. {"tool": "query_compounds", "args": {"smiles": "C{C}", "n": 3}}`;
  expect(parseToolCall(text)).toEqual({ tool: "query_compounds", args: { smiles: "C{C}", n: 3 } });
});

test("parseToolCall ignores an unterminated object", () => {
  expect(parseToolCall(`{"tool": "draw_molecule", "args": {"smiles": "CCO"`)).toBeNull();
});

test("every tool in the prompt has an example argument object", () => {
  for (const spec of TOOL_SPECS) {
    expect(TOOL_SYSTEM_PROMPT).toContain(spec.name);
    expect(JSON.parse(spec.args)).toBeInstanceOf(Object);
  }
});

test("formatToolResult truncates and keeps images out of the prompt", () => {
  const call = { tool: "draw_molecule", args: { name: "Ethanol" } };
  const formatted = formatToolResult(call, { result: { image: "data:image/png;base64,AAAA" } });
  expect(formatted).toContain("### Tool result (draw_molecule)");
  expect(formatted).not.toContain("base64");

  const long = formatToolResult({ tool: "query_compounds", args: {} }, { result: "x".repeat(5000) }, 100);
  expect(long.length).toBeLessThan(200);

  expect(formatToolResult(call, { error: "ValueError: bad SMILES" })).toContain("bad SMILES");
});

test("the tool-result header matches what the fine-tune is trained on", () => {
  // Mirrored by TOOL_RESULT_HEADER / TOOL_RESULT_LIMIT in
  // app/scripts/flows/llm_finetuning_data/build_drug_interaction_dataset.py.
  // Changing either side alone gives the model a format it never saw in training.
  const formatted = formatToolResult({ tool: "get_compound_by_name", args: {} }, { result: { a: 1 } });
  expect(formatted.startsWith("### Tool result (get_compound_by_name)\n")).toBe(true);

  const long = formatToolResult({ tool: "query_compounds", args: {} }, { result: "x".repeat(9000) });
  expect(long.length).toBeLessThanOrEqual("### Tool result (query_compounds)\n".length + 2000);
});

// ── Deterministic bridge (temporary; see routeDirectToolCall) ────────────────

test("routeDirectToolCall catches explicit structure requests", () => {
  const cases: [string, string][] = [
    ["Show me the molecular structure of prozac", "prozac"],
    ["show me the structure of Ibuprofen", "Ibuprofen"],
    ["Draw the structure of aspirin.", "aspirin"],
    ["Draw ibuprofen", "ibuprofen"],
    ["draw paracetamol please", "paracetamol"],
    ["Render caffeine", "caffeine"],
    ["What does warfarin look like?", "warfarin"],
    ["Draw the acetylsalicylic acid molecule", "acetylsalicylic acid"],
  ];
  for (const [message, name] of cases) {
    expect(routeDirectToolCall(message)).toEqual({ tool: "draw_molecule", args: { name } });
  }
});

test("routeDirectToolCall leaves every other question to the model", () => {
  const untouched = [
    "What is the molecular weight of Aspirin?",
    "Is it safe to take warfarin with aspirin?",
    "Which drugs interact with Warfarin?",
    "What is a CYP3A4 inhibitor?",
    "hi",
    "Draw it",          // names nothing to look up
    "render this",
  ];
  for (const message of untouched) {
    expect(routeDirectToolCall(message)).toBeNull();
  }
});

test("describeDrawResult writes the caption from the tool result", () => {
  expect(
    describeDrawResult({
      result: {
        source: "ChEMBL",
        pref_name: "IBUPROFEN",
        chembl_id: "CHEMBL521",
        full_molformula: "C13H18O2",
        mw_freebase: "206.28",
        smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
      },
    }),
  ).toBe("IBUPROFEN (CHEMBL521) — C13H18O2, MW 206.28. SMILES: CC(C)Cc1ccc(C(C)C(=O)O)cc1");

  expect(describeDrawResult({ result: { source: "supplied by the user", smiles: "CCO" } })).toBe(
    "Structure drawn from the SMILES you supplied: CCO",
  );
  expect(describeDrawResult({ error: "No ChEMBL compound named 'xyz'" })).toContain("Could not draw");
});

test("every tool carries an example a person can click and a name the parser knows", () => {
  for (const spec of TOOL_SPECS) {
    expect(spec.ask.length).toBeGreaterThan(0);
    // The manual's examples must actually reach the tool they advertise:
    // draw phrasings via the bridge, the rest by the model deciding.
    if (spec.name === "draw_molecule") {
      expect(routeDirectToolCall(spec.ask)?.tool).toBe("draw_molecule");
    } else {
      expect(routeDirectToolCall(spec.ask)).toBeNull();
    }
  }
});
