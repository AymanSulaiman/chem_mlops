import { connect } from "@lancedb/lancedb";
import { readdir } from "node:fs/promises";
import path from "node:path";

// Resolved relative to this file: web/src/rag.ts → data/lancedb at the project root.
export const DEFAULT_LANCEDB_DIR = new URL("../../data/lancedb", import.meta.url).pathname;

const STOPWORDS = new Set([
  "What", "When", "Where", "Why", "Who", "How", "The", "This", "That", "With",
  "From", "Have", "Has", "Can", "Will", "Would", "Should", "Could", "About",
  "Which", "Their", "There", "They", "Then", "Than", "Some", "More", "Been",
  "Into", "Also", "Most", "Both", "Each", "Such", "These", "Those", "Your",
  "Take", "Taking", "Does", "Drug", "Side", "Effects", "Used", "Treat", "Pain",
  "Tell", "List", "Give", "Show", "Explain", "Between", "Effect", "After",
  "Before", "Common", "Cause", "Causes", "Drugs", "Together", "Possible",
]);

// Prevent SQL injection in filter strings.
function sqlEscape(s: string): string {
  return s.replace(/'/g, "''");
}

async function resolveDbUri(lancedbDir: string): Promise<string> {
  const entries = await readdir(lancedbDir, { withFileTypes: true });
  const match = entries
    .filter(e => e.isDirectory() && e.name.startsWith("chembl_CHEMBL"))
    .sort((a, b) => a.name.localeCompare(b.name))
    .at(-1);
  if (!match) throw new Error(`No chembl_CHEMBL_* directory found in ${lancedbDir}`);
  return path.join(lancedbDir, match.name);
}

// Extract capitalised words from user text as drug name candidates.
// The polypharmacy table stores names in title-case (Polars str.to_titlecase),
// so we normalise each candidate to Title Case before querying.
export function extractDrugCandidates(text: string): string[] {
  const words = text.match(/\b[A-Z][a-zA-Z]{2,}\b/g) ?? [];
  const unique = [...new Set(words.filter(w => !STOPWORDS.has(w)))];
  return unique.map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase());
}

// Build a compact pharmacological context block from ChEMBL and TWOSIDES.
// Returns null when the LanceDB database is unavailable or no relevant rows
// are found, so the caller can fall back to a plain standard-mode chat.
export async function buildRagContext(
  userMessage: string,
  lancedbDir: string,
): Promise<string | null> {
  let dbUri: string;
  try {
    dbUri = await resolveDbUri(lancedbDir);
  } catch {
    return null;
  }

  const db = await connect(dbUri);
  const candidates = extractDrugCandidates(userMessage);
  if (candidates.length === 0) return null;

  const parts: string[] = [];

  // Compounds table: look up each candidate by preferred name.
  try {
    const tbl = await db.openTable("compounds");
    for (const name of candidates.slice(0, 4)) {
      const safe = sqlEscape(name.toLowerCase());
      const rows: Record<string, unknown>[] = await tbl
        .query()
        .filter(`LOWER(pref_name) = '${safe}'`)
        .select(["chembl_id", "pref_name", "mw_freebase"])
        .limit(1)
        .toArray();
      if (rows.length > 0) {
        const r = rows[0];
        parts.push(`${r["pref_name"]} (${r["chembl_id"]}), MW ${r["mw_freebase"]}`);
      }
    }
  } catch {
    // compounds table unavailable — not a fatal error
  }

  // Polypharmacy table: explicit pair signals then top per-drug partners.
  try {
    const polyTbl = await db.openTable("polypharmacy");

    // Explicit pair interactions (all ordered combinations of candidates)
    for (let i = 0; i < candidates.length; i++) {
      for (let j = i + 1; j < candidates.length; j++) {
        const d1 = sqlEscape(candidates[i]);
        const d2 = sqlEscape(candidates[j]);
        const rows: Record<string, unknown>[] = await polyTbl
          .query()
          .filter(
            `(drug_1_name = '${d1}' AND drug_2_name = '${d2}') OR ` +
            `(drug_1_name = '${d2}' AND drug_2_name = '${d1}')`,
          )
          .select(["drug_1_name", "drug_2_name", "side_effects", "max_prr", "total_cases"])
          .limit(1)
          .toArray();
        if (rows.length > 0) {
          const r = rows[0];
          const prr = typeof r["max_prr"] === "number" ? (r["max_prr"] as number).toFixed(1) : "?";
          const topEffects = (r["side_effects"] as string)
            ?.split(";").slice(0, 3).map(s => s.trim()).join(", ");
          parts.push(
            `${candidates[i]} + ${candidates[j]} interaction (max PRR ${prr}, ${r["total_cases"]} cases): ${topEffects}`,
          );
        }
      }
    }

    // Top partners for each drug individually (strongest signals first)
    for (const name of candidates.slice(0, 2)) {
      const safe = sqlEscape(name);
      const rows: Record<string, unknown>[] = await polyTbl
        .query()
        .filter(`drug_1_name = '${safe}' OR drug_2_name = '${safe}'`)
        .select(["drug_1_name", "drug_2_name", "side_effects", "max_prr"])
        .limit(5)
        .toArray();
      const sorted = rows.sort(
        (a, b) => ((b["max_prr"] as number) ?? 0) - ((a["max_prr"] as number) ?? 0),
      );
      for (const r of sorted.slice(0, 3)) {
        const partner =
          (r["drug_1_name"] as string) === name ? r["drug_2_name"] : r["drug_1_name"];
        const prr = typeof r["max_prr"] === "number" ? (r["max_prr"] as number).toFixed(1) : "?";
        const topEffect = (r["side_effects"] as string)?.split(";")[0]?.trim();
        parts.push(`${name} + ${partner}: ${topEffect} (PRR ${prr})`);
      }
    }
  } catch {
    // polypharmacy table unavailable — not a fatal error
  }

  if (parts.length === 0) return null;

  return [
    "Relevant pharmacological context retrieved from ChEMBL and TWOSIDES databases:",
    ...parts.map(p => `- ${p}`),
  ].join("\n");
}

// Prepend a system message carrying RAG context to the conversation history.
export function augmentMessages<T extends { role: string; content: string }>(
  messages: T[],
  context: string,
): Array<{ role: string; content: string }> {
  return [{ role: "system", content: context }, ...messages];
}
