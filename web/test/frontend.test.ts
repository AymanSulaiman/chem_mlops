import { expect, test } from "bun:test";

import { formatModelLabel, formatReplyText } from "../src/frontend-helpers";

test("formatModelLabel shows the model and its source", () => {
  expect(formatModelLabel({ model: "chembl-drug-chat:gemma4-e2b", source: "ollama-tags" })).toBe(
    "Latest model: chembl-drug-chat:gemma4-e2b (ollama-tags)",
  );
});

test("formatReplyText falls back when the reply is blank", () => {
  expect(formatReplyText("")).toBe("(empty response)");
  expect(formatReplyText("  ")).toBe("(empty response)");
  expect(formatReplyText("hello")).toBe("hello");
});
