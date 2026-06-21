import { expect, test } from "bun:test";
import { formatReplyText, renderMarkdown } from "../src/frontend-helpers";

test("formatReplyText falls back when the reply is blank", () => {
  expect(formatReplyText("")).toBe("(empty response)");
  expect(formatReplyText("  ")).toBe("(empty response)");
  expect(formatReplyText("hello")).toBe("hello");
});

test("renderMarkdown escapes HTML before processing", () => {
  expect(renderMarkdown("<script>alert(1)</script>")).toBe(
    "&lt;script&gt;alert(1)&lt;/script&gt;"
  );
});

test("renderMarkdown renders code blocks", () => {
  expect(renderMarkdown("```\nconst x = 1;\n```")).toContain("<pre><code>");
});

test("renderMarkdown renders inline code", () => {
  expect(renderMarkdown("`foo`")).toBe("<code>foo</code>");
});

test("renderMarkdown renders bold and italic", () => {
  expect(renderMarkdown("**bold**")).toBe("<strong>bold</strong>");
  expect(renderMarkdown("*italic*")).toBe("<em>italic</em>");
});

test("renderMarkdown converts newlines to br", () => {
  expect(renderMarkdown("a\nb")).toBe("a<br>b");
});
