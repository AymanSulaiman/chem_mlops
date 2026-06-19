import { expect, test } from "bun:test";

import { createChatRequestHandler } from "../src/app";

test("request handler serves the app shell and health route", async () => {
  const handler = createChatRequestHandler({
    fetchImpl: fetch,
    publicDir: new URL("../public/", import.meta.url),
  });

  const home = await handler(new Request("http://localhost/"));
  const health = await handler(new Request("http://localhost/api/health"));

  expect(home.status).toBe(200);
  expect(await home.text()).toContain('<script type="module" src="/frontend.js"></script>');
  expect(await health.json()).toEqual({
    ok: true,
    ollamaBaseUrl: "http://127.0.0.1:11434",
  });
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
        expect(body.messages).toEqual([{ role: "user", content: "hello" }]);

        return Response.json({
          message: { content: "hi there" },
        });
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
  expect(await response.json()).toEqual({
    model: "chembl-drug-chat:newer",
    source: "ollama-tags",
    reply: "hi there",
  });

  const response2 = await handler(
    new Request("http://localhost/api/model"),
  );

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
        const body = init?.body ? JSON.parse(init.body as string) : null;
        capturedModel = body.model;
        return Response.json({ message: { content: "rag reply" } });
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
  const data = await response.json();
  expect(data.model).toBe("gemma3:1b");
  expect(data.source).toBe("rag-model");
  expect(data.reply).toBe("rag reply");
});

test("standard mode still routes to the fine-tuned model", async () => {
  let capturedModel: string | null = null;

  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    ragModelName: "gemma3:1b",
    fetchImpl: async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.endsWith("/api/tags")) {
        return Response.json({
          models: [{ name: "chembl-drug-chat:1b", modified_at: "2026-04-19T10:00:00Z" }],
        });
      }

      if (url.endsWith("/api/chat")) {
        const body = init?.body ? JSON.parse(init.body as string) : null;
        capturedModel = body.model;
        return Response.json({ message: { content: "standard reply" } });
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
  expect((await response.json()).source).toBe("ollama-tags");
});

test("request handler handles errors gracefully", async () => {
  const handler = createChatRequestHandler({
    fetchImpl: async () => {
      throw new Error("Network error");
    },
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
