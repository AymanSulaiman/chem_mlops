import { expect, test } from "bun:test";

import { augmentMessages, buildRagContext, extractDrugCandidates } from "../src/rag";

test("extractDrugCandidates picks capitalised words and excludes stopwords", () => {
  const candidates = extractDrugCandidates("What are the side effects of Warfarin and Aspirin?");
  expect(candidates).toContain("Warfarin");
  expect(candidates).toContain("Aspirin");
  // Common stopwords must not appear
  expect(candidates).not.toContain("What");
  expect(candidates).not.toContain("Effects");
});

test("extractDrugCandidates returns an empty array for plain lowercase text", () => {
  expect(extractDrugCandidates("tell me about warfarin")).toEqual([]);
});

test("extractDrugCandidates deduplicates repeated names", () => {
  const candidates = extractDrugCandidates("Aspirin is common. Aspirin was also studied.");
  expect(candidates.filter(c => c === "Aspirin").length).toBe(1);
});

test("augmentMessages prepends a system message", () => {
  const messages = [{ role: "user", content: "hello" }];
  const result = augmentMessages(messages, "Some context");
  expect(result[0]).toEqual({ role: "system", content: "Some context" });
  expect(result[1]).toEqual({ role: "user", content: "hello" });
  expect(result.length).toBe(2);
});

test("augmentMessages does not mutate the original array", () => {
  const messages = [{ role: "user", content: "hello" }];
  augmentMessages(messages, "ctx");
  expect(messages.length).toBe(1);
});

test("buildRagContext returns null when lancedb dir does not exist", async () => {
  const result = await buildRagContext("What is Warfarin?", "/nonexistent/lancedb");
  expect(result).toBeNull();
});

test("buildRagContext returns null when no capitalised drug names are found", async () => {
  const result = await buildRagContext("tell me about drugs in general", "/nonexistent/lancedb");
  expect(result).toBeNull();
});
