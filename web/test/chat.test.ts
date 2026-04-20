import { expect, test } from "bun:test";

import { normalizeMessages } from "../src/app";

test("normalizeMessages keeps valid messages and trims whitespace", () => {
  const messages = normalizeMessages([
    { role: "user", content: "  hello  " },
    { role: "assistant", content: "hi" },
    { role: "system", content: "ignore me" },
    { role: "user", content: "   " },
    null,
  ]);

  expect(messages).toEqual([
    { role: "user", content: "hello" },
    { role: "assistant", content: "hi" },
  ]);
});

test("normalizeMessages returns an empty array for non-arrays", () => {
  expect(normalizeMessages(undefined)).toEqual([]);
  expect(normalizeMessages("hello")).toEqual([]);
});
