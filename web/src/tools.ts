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

// draw_smiles was the old name for draw_molecule; a model that learned the old
// name still gets routed rather than ignored.
const TOOL_ALIASES: Record<string, string> = { draw_smiles: "draw_molecule" };

type ToolSpec = {
  readonly name: string;
  readonly args: string;
  readonly description: string;
  readonly ask: string;
  readonly help?: string;
};

export const TOOL_SPECS = [
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
] as const satisfies readonly ToolSpec[];

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

// ── Deterministic bridge ─────────────────────────────────────────────────────
// TEMPORARY. This is not the model calling a tool: chembl-drug-chat:1b never
// emits a tool call (roadmap item 3), so an explicit "draw X" is routed straight
// to draw_molecule. Delete this and its call site in app.ts once a fine-tune
// trained on the "tool calls" dataset category does the routing itself.
//
// Deliberately narrow: only requests that name a molecule to draw. Anything
// else — interactions, properties, lookups — still needs the model to decide,
// and keyword-routing those would rebuild the RAG guesswork item 2 removed.
const DRAW_PATTERNS = [
  // "show me the molecular structure of prozac", "draw the structure of X"
  /\b(?:draw|render|display|show|give)\b(?:\s+me)?\s+(?:the\s+|a\s+)?(?:\w+\s+)?structure\s+(?:of|for)\s+(.+)/i,
  // "draw ibuprofen", "render aspirin"
  /^\s*(?:draw|render)\s+(?:me\s+)?(?:the\s+|a\s+)?(.+)/i,
  // "what does ibuprofen look like?"
  /^\s*what\s+does\s+(.+?)\s+look\s+like/i,
];

export function routeDirectToolCall(message: string): ToolCall | null {
  for (const pattern of DRAW_PATTERNS) {
    const name = pattern.exec(message)?.[1];
    if (!name) continue;
    const cleaned = (
      // "draw ibuprofen and tell me its side effects" names one molecule.
      name.split(/\s+(?:and|then|plus|also)\s+|[,;]/)[0] ?? ""
    )
      .replace(/\b(?:molecule|compound|structure|please)\b/gi, "")
      .replace(/['"“”]/g, "")
      .replace(/[.?!,;:]+\s*$/, "")
      .trim();
    // A bare verb ("draw it", "render this") names nothing to look up.
    if (!cleaned || /^(?:it|this|that|one)$/i.test(cleaned) || cleaned.length > 60) continue;
    return { tool: "draw_molecule", args: { name: cleaned } };
  }
  return null;
}

// Caption for a bridged draw, written from the tool result rather than by the
// model. Part of the same temporary bridge: this fine-tune, handed a tool
// result, mimics its JSON shape instead of reading it — real captures include
// "{ CHEMBL1201082 }" and an invented ebi.ac.uk image URL. Deterministic text
// beats a hallucinated caption until the model is trained to use tool output.
export function describeDrawResult(outcome: ToolOutcome): string {
  if (outcome.error) return `Could not draw that: ${outcome.error}`;
  const r = (outcome.result ?? {}) as Record<string, string>;
  if (!r.smiles) return "Structure drawn.";
  if (r.source !== "ChEMBL") return `Structure drawn from the SMILES you supplied: ${r.smiles}`;

  const facts = [r.full_molformula, r.mw_freebase ? `MW ${r.mw_freebase}` : ""].filter(Boolean);
  return (
    `${r.pref_name} (${r.chembl_id})` +
    (facts.length ? ` — ${facts.join(", ")}` : "") +
    `. SMILES: ${r.smiles}`
  );
}

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
    const call = parsed as { tool?: unknown; name?: unknown };
    const named = typeof call.tool === "string" ? call.tool : call.name;
    if (typeof named !== "string") continue;
    if (TOOL_ALIASES[named] || TOOL_SPECS.some(t => t.name === named)) continue;
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
