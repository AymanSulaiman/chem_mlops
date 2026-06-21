import { expect, test } from "bun:test";

import { createChatRequestHandler } from "../src/app";

function ndjson(...chunks: object[]): Response {
  const body = chunks.map(c => JSON.stringify(c)).join("\n") + "\n";
  return new Response(body, { headers: { "content-type": "application/x-ndjson" } });
}

async function collectStream(response: Response): Promise<string> {
  const text = await response.text();
  return text.trim().split("\n").reduce((acc, line) => {
    try {
      const c = JSON.parse(line) as { done?: boolean; message?: { content?: string } };
      return acc + ((!c.done && c.message?.content) ? c.message.content : "");
    } catch { return acc; }
  }, "");
}

test("request handler serves the app shell and health route", async () => {
  const handler = createChatRequestHandler({
    fetchImpl: fetch,
    publicDir: new URL("../public/", import.meta.url),
  });

  const home = await handler(new Request("http://localhost/"));
  const health = await handler(new Request("http://localhost/api/health"));

  expect(home.status).toBe(200);
  expect(await home.text()).toContain('<script type="module" src="/frontend.js"></script>');
  expect(await health.json()).toEqual({ ok: true, ollamaBaseUrl: "http://127.0.0.1:11434" });
});

test("request handler chats with the latest model and caches the lookup", async () => {
  let tagsCalls = 0;
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.endsWith("/api/tags")) {
        tagsCalls += 1;
        return Response.json({
          models: [
            { name: "chembl-drug-chat:older", modified_at: "2026-04-18T10:00:00Z" },
            { name: "chembl-drug-chat:newer", modified_at: "2026-04-19T10:00:00Z" },
          ],
        });
      }

      if (url.endsWith("/api/chat")) {
        const body = init?.body ? JSON.parse(init.body as string) : null;
        expect(body.model).toBe("chembl-drug-chat:newer");
        expect(body.stream).toBe(true);
        expect(body.messages).toEqual([{ role: "user", content: "hello" }]);
        return ndjson(
          { message: { role: "assistant", content: "hi " }, done: false },
          { message: { role: "assistant", content: "there" }, done: false },
          { message: { role: "assistant", content: "" }, done: true, eval_count: 10, prompt_eval_count: 5, eval_duration: 1_000_000_000, total_duration: 2_000_000_000 },
        );
      }

      throw new Error(`Unexpected request: ${url}`);
    },
    now: () => 0,
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    }),
  );

  expect(response.status).toBe(200);
  expect(response.headers.get("x-model")).toBe("chembl-drug-chat:newer");
  expect(response.headers.get("x-source")).toBe("ollama-tags");
  expect(await collectStream(response)).toBe("hi there");

  const response2 = await handler(new Request("http://localhost/api/model"));
  expect(response2.status).toBe(200);
  expect(tagsCalls).toBe(1);
});

test("RAG mode routes to gemma3:1b, not the fine-tuned model", async () => {
  let capturedModel: string | null = null;

  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    ragModelName: "gemma3:1b",
    fetchImpl: async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/chat")) {
        capturedModel = (JSON.parse(init?.body as string)).model;
        return ndjson(
          { message: { role: "assistant", content: "rag reply" }, done: false },
          { message: { role: "assistant", content: "" }, done: true },
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "What is Warfarin?" }], mode: "rag" }),
    }),
  );

  expect(response.status).toBe(200);
  expect(capturedModel).toBe("gemma3:1b");
  expect(response.headers.get("x-model")).toBe("gemma3:1b");
  expect(response.headers.get("x-source")).toBe("rag-model");
  expect(await collectStream(response)).toBe("rag reply");
});

test("finetuned mode routes to the fine-tuned model", async () => {
  let capturedModel: string | null = null;

  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    ragModelName: "gemma3:1b",
    fetchImpl: async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) {
        return Response.json({ models: [{ name: "chembl-drug-chat:1b", modified_at: "2026-04-19T10:00:00Z" }] });
      }
      if (url.endsWith("/api/chat")) {
        capturedModel = (JSON.parse(init?.body as string)).model;
        return ndjson(
          { message: { role: "assistant", content: "standard reply" }, done: false },
          { message: { role: "assistant", content: "" }, done: true },
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    },
    now: () => 0,
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    }),
  );

  expect(response.status).toBe(200);
  expect(capturedModel).toBe("chembl-drug-chat:1b");
  expect(response.headers.get("x-source")).toBe("ollama-tags");
});

test("request handler handles errors gracefully", async () => {
  const handler = createChatRequestHandler({
    fetchImpl: async () => { throw new Error("Network error"); },
    publicDir: new URL("../public/", import.meta.url),
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "test" }] }),
    }),
  );

  expect(response.status).toBe(503);
});
