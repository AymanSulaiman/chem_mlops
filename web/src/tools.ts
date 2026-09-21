// Tool definitions for the agentic pane.
//
// The model asks for a lookup by emitting a JSON object; the server runs it
// against LanceDB via app/scripts/flows/vector_store/tools.py and feeds the
// result back. Nothing is injected unless a tool asked for it.

// Resolved relative to this file: web/src/tools.ts → the project root.
export const PROJECT_ROOT = new URL("../../", import.meta.url).pathname;

export type ToolCall = { tool: string; args: Record<string, unknown> };

// Mirrored by TOOL_RESULT_HEADER in the dataset builder: the fine-tune is
// trained on this exact header, so the two must not drift.
export const TOOL_RESULT_HEADER = "### Tool result";
export type ToolOutcome = { result?: unknown; error?: string };

// Near-miss names the fine-tune invents. Measured on the item 3 benchmark:
// 11 of 40 calls named a tool that does not exist (query_compound, query_drugs)
// or picked query_compounds — a SMILES similarity search — for a lookup by drug
// name. The intent was right and the spelling was not, so mapping them recovers
// the call instead of returning an error the model has to recover from.
// Mirrored by TOOL_ALIASES in app/scripts/flows/vector_store/tools.py.
const TOOL_ALIASES: Record<string, string> = {
  draw_smiles: "draw_molecule", // old name, kept so a stale prompt still works
  draw: "draw_molecule",
  query_compound: "get_compound_by_name",
  query_drug: "get_compound_by_name",
  query_drugs: "get_compound_by_name",
  get_compound: "get_compound_by_name",
  get_drug_by_name: "get_compound_by_name",
  compound_by_name: "get_compound_by_name",
  query_compound_by_name: "get_compound_by_name",
  query_side_effects: "query_drug_side_effects",
  query_interactions: "query_drug_side_effects",
  query_drug_interactions: "query_drug_side_effects",
};

// What each tool calls its primary argument, against the keys a model reaches
// for instead. Same failure as the names: right intent, wrong spelling.
// Mirrored by ARG_ALIASES in app/scripts/flows/vector_store/tools.py.
const ARG_ALIASES: Record<string, Record<string, string>> = {
  get_compound_by_name: { drug_name: "name", compound_name: "name", compound: "name", drug: "name" },
  draw_molecule: {
    drug_name: "name",
    compound_name: "name",
    compound: "name",
    drug: "name",
    molecule: "name",
  },
  query_drug_side_effects: { name: "drug_name", drug: "drug_name", compound: "drug_name" },
  query_polypharmacy: { drug_a: "drug_1", drug_b: "drug_2", drug1: "drug_1", drug2: "drug_2" },
  query_compounds: { smiles_string: "smiles", structure: "smiles" },
};

const NAME_KEYS = ["name", "drug_name", "compound_name", "compound", "drug"];

// Map a near-miss tool name and argument keys onto the real registry. An
// unknown name that matches nothing comes back untouched, so unknownToolName
// still reports it. Mirrored by resolve_tool_call in the Python bridge — the
// serving path and the benchmark have to agree, or the measured rate is not
// the rate users get.
export function resolveToolCall(name: string, args: Record<string, unknown>): ToolCall {
  let tool = TOOL_ALIASES[name] ?? name;

  // A similarity search needs a SMILES string. Handed a drug name instead, the
  // model meant the by-name lookup: 4 of 40 benchmark calls did this.
  if (tool === "query_compounds" && !("smiles" in args) && NAME_KEYS.some(k => k in args)) {
    tool = "get_compound_by_name";
  }

  const renames = ARG_ALIASES[tool] ?? {};
  const resolved: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    if (!(key in renames)) resolved[key] = value;
  }
  for (const [key, value] of Object.entries(args)) {
    // A correctly named key already present wins over the aliased spelling.
    const renamed = renames[key];
    if (renamed !== undefined && !(renamed in resolved)) resolved[renamed] = value;
  }
  return { tool, args: resolved };
}

type ToolSpec = {
  readonly name: string;
  readonly args: string;
  readonly description: string;
  readonly ask: string;
  readonly help?: string;
  // Listed in TOOL_SYSTEM_PROMPT? Callable either way — this only decides what
  // the model is told about. See query_compounds below for why that differs.
  readonly advertised?: boolean;
};

export const TOOL_SPECS: readonly ToolSpec[] = [
  {
    name: "get_compound_by_name",
    args: `{"name": "Aspirin"}`,
    description: "ChEMBL record for a drug by name: chembl_id, molecular weight, properties.",
    ask: "What is the molecular weight of Aspirin?",
  },
  {
    name: "query_drug_side_effects",
    args: `{"drug_name": "Warfarin", "n": 10}`,
    description: "Known interaction partners of one drug, strongest TWOSIDES signal first.",
    ask: "Which drugs interact most strongly with Warfarin?",
  },
  {
    name: "query_polypharmacy",
    args: `{"drug_1": "Warfarin", "drug_2": "Aspirin"}`,
    description: "Side effects reported for one specific drug pair taken together.",
    ask: "Is it safe to take warfarin with aspirin?",
  },
  {
    name: "query_compounds",
    args: `{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "n": 5}`,
    description: "Compounds structurally similar to a SMILES string (Morgan fingerprint).",
    ask: "What compounds are similar to CC(=O)Oc1ccccc1C(=O)O?",
    // Not advertised to the model: it is the one tool with no training records
    // (generate_tool_call_qa excludes it — a similarity search needs a SMILES
    // the model does not know), and on run 20260920_114710_tools it absorbed
    // 17 of 40 tool calls, 10 of them copying the example SMILES above
    // verbatim — aspirin, emitted for questions about other drugs entirely.
    // Unsure which tool to use, the model copies the nearest literal example in
    // the prompt, so an untrained tool in the prompt is a decoy. Still callable
    // and still in the manual: a person can ask for a similarity search, and a
    // trained model can be re-advertised by deleting this line.
    advertised: false,
  },
  {
    name: "draw_molecule",
    args: `{"name": "Ibuprofen"}`,
    description:
      "Draw a molecule. Give the drug NAME and the structure is looked up in ChEMBL — " +
      "never invent a SMILES string. Pass {\"smiles\": \"...\"} only for a structure the user typed.",
    ask: "Draw ibuprofen",
    // `description` is written for the model; `help` is what the in-app manual
    // shows a person. Only set it where the two would differ.
    help: "Draws the structure, looked up in ChEMBL by name. You can paste a SMILES instead.",
  },
];

export const TOOL_SYSTEM_PROMPT = [
  "You can look up real pharmacological data with tools. To call one, reply with",
  "ONLY a JSON object and nothing else:",
  `{"tool": "<name>", "args": {...}}`,
  "",
  "Tools:",
  ...TOOL_SPECS.filter(t => t.advertised !== false).map(
    t => `- ${t.name} ${t.args} — ${t.description}`,
  ),
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
    const raw = call.args ?? call.arguments ?? {};
    const resolved = resolveToolCall(
      named,
      typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : {},
    );
    if (!TOOL_SPECS.some(t => t.name === resolved.tool)) continue;
    return resolved;
  }
  return null;
}

// The name in a tool-call-shaped object that names a tool we do not have
// ("query_compound_by_name"). Worth catching: the reply is unusable as prose,
// so it is better to hand the model the error than to show the JSON to a user.
export function unknownToolName(text: string): string | null {
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
    const raw = call.args ?? call.arguments ?? {};
    const { tool } = resolveToolCall(
      named,
      typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : {},
    );
    if (TOOL_SPECS.some(t => t.name === tool)) continue;
    return named;
  }
  return null;
}

// Error text handed back to the model when it names a tool that does not exist.
export function unknownToolError(name: string): string {
  return `error: Unknown tool '${name}'. Available: ${TOOL_SPECS.map(t => t.name).join(", ")}.`;
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
  return `${TOOL_RESULT_HEADER} (${call.tool})\n${body}`;
}

// Base64 images are for the browser, not the prompt. The replacement is plain
// prose: a placeholder-looking value gets echoed back as markdown by the model.
function redactImages(result: unknown): unknown {
  if (typeof result === "object" && result !== null && "image" in result) {
    return { ...result, image: "the picture is already displayed to the user" };
  }
  return result;
}
