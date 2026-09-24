import { expect, test } from "bun:test";

import {
  formatToolResult,
  parseToolCall,
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

test("every advertised tool has an example argument object", () => {
  for (const spec of TOOL_SPECS) {
    if (spec.advertised !== false) expect(TOOL_SYSTEM_PROMPT).toContain(spec.name);
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

test("every tool carries an example a person can click and a name the parser knows", () => {
  for (const spec of TOOL_SPECS) {
    expect(spec.ask.length).toBeGreaterThan(0);
    // Which tool an example reaches is now the model's decision, measured by
    // the tool-call benchmark rather than asserted here.
    expect(parseToolCall(`{"tool": "${spec.name}", "args": ${spec.args}}`)?.tool).toBe(spec.name);
  }
});

test("parseToolCall corrects the near-miss names the fine-tune emits", () => {
  // Measured on the roadmap item 3 benchmark: 11 of 40 calls missed by a name
  // while the intent was unambiguous. Correcting beats erroring.
  expect(parseToolCall(`{"tool": "query_compound", "args": {"drug_name": "SIROLIMUS"}}`)).toEqual({
    tool: "get_compound_by_name",
    args: { name: "SIROLIMUS" },
  });
});

test("parseToolCall rereads a similarity search given a drug name as a lookup", () => {
  // query_compounds needs a SMILES, so a drug name cannot be what was meant.
  expect(
    parseToolCall(`{"tool": "query_compounds", "args": {"drug_name": "SIROLIMUS", "n": 10}}`),
  ).toEqual({ tool: "get_compound_by_name", args: { n: 10, name: "SIROLIMUS" } });

  // A real similarity search is left alone.
  expect(parseToolCall(`{"tool": "query_compounds", "args": {"smiles": "CCO"}}`)).toEqual({
    tool: "query_compounds",
    args: { smiles: "CCO" },
  });
});

test("the untrained tool is callable but not advertised to the model", () => {
  // query_compounds has no training records, and listing it made the model copy
  // its example SMILES verbatim for unrelated questions (17/40 calls on run
  // 20260920_114710_tools). Out of the prompt, still dispatchable.
  expect(TOOL_SYSTEM_PROMPT).not.toContain("query_compounds");
  expect(parseToolCall(`{"tool": "query_compounds", "args": {"smiles": "CCO"}}`)).toEqual({
    tool: "query_compounds",
    args: { smiles: "CCO" },
  });
  // Every other tool is still advertised.
  for (const spec of TOOL_SPECS.filter(t => t.advertised !== false)) {
    expect(TOOL_SYSTEM_PROMPT).toContain(spec.name);
  }
});
