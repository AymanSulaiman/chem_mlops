// Tool definitions for the agentic pane.
//
// The model asks for a lookup by emitting a JSON object; the server runs it
// against LanceDB via app/scripts/flows/vector_store/tools.py and feeds the
// result back. Nothing is injected unless a tool asked for it.

// Resolved relative to this file: web/src/tools.ts → the project root.
export const PROJECT_ROOT = new URL("../../", import.meta.url).pathname;

export type ToolCall = { tool: string; args: Record<string, unknown> };
export type ToolOutcome = { result?: unknown; error?: string };

// draw_smiles was the old name for draw_molecule; a model that learned the old
// name still gets routed rather than ignored.
const TOOL_ALIASES: Record<string, string> = { draw_smiles: "draw_molecule" };

export const TOOL_SPECS = [
  {
    name: "get_compound_by_name",
    args: `{"name": "Aspirin"}`,
    description: "ChEMBL record for a drug by name: chembl_id, molecular weight, properties.",
  },
  {
    name: "query_drug_side_effects",
    args: `{"drug_name": "Warfarin", "n": 10}`,
    description: "Known interaction partners of one drug, strongest TWOSIDES signal first.",
  },
  {
    name: "query_polypharmacy",
    args: `{"drug_1": "Warfarin", "drug_2": "Aspirin"}`,
    description: "Side effects reported for one specific drug pair taken together.",
  },
  {
    name: "query_compounds",
    args: `{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "n": 5}`,
    description: "Compounds structurally similar to a SMILES string (Morgan fingerprint).",
  },
  {
    name: "draw_molecule",
    args: `{"name": "Ibuprofen"}`,
    description:
      "Draw a molecule. Give the drug NAME and the structure is looked up in ChEMBL — " +
      "never invent a SMILES string. Pass {\"smiles\": \"...\"} only for a structure the user typed.",
  },
] as const;

export const TOOL_SYSTEM_PROMPT = [
  "You can look up real pharmacological data with tools. To call one, reply with",
  "ONLY a JSON object and nothing else:",
  `{"tool": "<name>", "args": {...}}`,
  "",
  "Tools:",
  ...TOOL_SPECS.map(t => `- ${t.name} ${t.args} — ${t.description}`),
  "",
  "The tool result comes back as a '### Tool result' message. Then answer the",
  "question in prose using it. Do not invent ChEMBL IDs or side effects: look them",
  "up. If no tool is needed, just answer.",
].join("\n");

// Find the first balanced {...} in the text and read it as a tool call.
// The fine-tuned model is a prose completer (see roadmap item 3), so the JSON
// usually arrives wrapped in chatter or a ``` fence — a plain JSON.parse of the
// whole reply is not enough.
export function parseToolCall(text: string): ToolCall | null {
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const end = matchBrace(text, start);
    if (end === -1) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text.slice(start, end + 1));
    } catch {
      continue;
    }
    const call = parsed as { tool?: unknown; name?: unknown; args?: unknown; arguments?: unknown };
    const named = typeof call.tool === "string" ? call.tool : call.name;
    if (typeof named !== "string") continue;
    const tool = TOOL_ALIASES[named] ?? named;
    if (!TOOL_SPECS.some(t => t.name === tool)) continue;
    const args = call.args ?? call.arguments ?? {};
    return {
      tool,
      args: typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {},
    };
  }
  return null;
}

// Index of the '}' closing the '{' at `start`, or -1. Skips braces in strings.
function matchBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i;
  }
  return -1;
}

// Run one tool call in the project's Python environment.
export async function runTool(call: ToolCall, cwd: string = PROJECT_ROOT): Promise<ToolOutcome> {
  const proc = Bun.spawn(
    [
      "uv", "run", "python", "-m", "app.scripts.flows.vector_store.tools",
      call.tool, JSON.stringify(call.args),
    ],
    { cwd, stdout: "pipe", stderr: "pipe" },
  );
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  await proc.exited;
  try {
    return JSON.parse(stdout.trim().split("\n").at(-1) ?? "") as ToolOutcome;
  } catch {
    return { error: `Tool '${call.tool}' produced no result. ${stderr.trim().slice(-400)}` };
  }
}

// What the model sees after a tool runs. Truncated: a similarity search can
// return thousands of characters and the context window is shared with history.
export function formatToolResult(call: ToolCall, outcome: ToolOutcome, limit = 2000): string {
  const body = outcome.error
    ? `error: ${outcome.error}`
    : JSON.stringify(redactImages(outcome.result)).slice(0, limit);
  return `### Tool result (${call.tool})\n${body}`;
}

// Base64 images are for the browser, not the prompt. The replacement is plain
// prose: a placeholder-looking value gets echoed back as markdown by the model.
function redactImages(result: unknown): unknown {
  if (typeof result === "object" && result !== null && "image" in result) {
    return { ...result, image: "the picture is already displayed to the user" };
  }
  return result;
}
