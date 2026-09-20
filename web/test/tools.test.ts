import { expect, test } from "bun:test";

import { formatToolResult, parseToolCall, TOOL_SPECS, TOOL_SYSTEM_PROMPT } from "../src/tools";

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
