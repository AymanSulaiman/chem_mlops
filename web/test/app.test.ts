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
