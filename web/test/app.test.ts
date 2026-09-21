import { expect, test } from "bun:test";

import { createChatRequestHandler } from "../src/app";

type Event = {
  tool?: { tool: string; args: Record<string, unknown> };
  toolResult?: { tool: string; result?: unknown; error?: string };
  message?: { content?: string };
  done?: boolean;
  error?: string;
};

// Ollama streams NDJSON: one chunk per token, counts on the final chunk.
function reply(content: string, chunkSize = 1000): Response {
  const pieces = content.match(new RegExp(`[\\s\\S]{1,${chunkSize}}`, "g")) ?? [""];
  const lines = pieces.map(piece =>
    JSON.stringify({ message: { role: "assistant", content: piece }, done: false }),
  );
  lines.push(JSON.stringify({ message: { content: "" }, done: true, eval_count: 7, prompt_eval_count: 11 }));
  return new Response(`${lines.join("\n")}\n`, {
    headers: { "content-type": "application/x-ndjson" },
  });
}

async function collectEvents(response: Response): Promise<Event[]> {
  const text = await response.text();
  return text.trim().split("\n").flatMap(line => {
    try {
      return [JSON.parse(line) as Event];
    } catch {
      return [];
    }
  });
}

function answerOf(events: Event[]): string {
  return events.map(e => e.message?.content ?? "").join("");
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
    fetchImpl: async (input) => {
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
      if (url.endsWith("/api/chat")) return reply("hi there");
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
  expect(answerOf(await collectEvents(response))).toBe("hi there");

  const response2 = await handler(new Request("http://localhost/api/model"));
  expect(response2.status).toBe(200);
  expect(tagsCalls).toBe(1);
});

test("a tool call is run, reported, and fed back for a grounded answer", async () => {
  const sentMessages: { role: string; content: string }[][] = [];
  const replies = [
    `Let me look that up.\n{"tool": "get_compound_by_name", "args": {"name": "Aspirin"}}`,
    "Aspirin is CHEMBL25, MW 180.16.",
  ];

  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fallbackModelName: "chembl-drug-chat:1b",
    fetchImpl: async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) {
        sentMessages.push(JSON.parse(init?.body as string).messages);
        return reply(replies.shift() ?? "no more replies");
      }
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async (call) => ({ result: { chembl_id: "CHEMBL25", pref_name: "ASPIRIN", asked: call.args } }),
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "What is Aspirin?" }] }),
    }),
  );

  const events = await collectEvents(response);
  // Prose before the call streams first, so find the events rather than index them.
  expect(events.find(e => e.tool)?.tool).toEqual({
    tool: "get_compound_by_name",
    args: { name: "Aspirin" },
  });
  expect(events.find(e => e.toolResult)?.toolResult?.tool).toBe("get_compound_by_name");
  expect(answerOf(events)).toBe("Let me look that up.\nAspirin is CHEMBL25, MW 180.16.");
  expect(events.at(-1)?.done).toBe(true);

  // The second turn must carry the tool result — that is what grounds the answer.
  expect(sentMessages).toHaveLength(2);
  expect(sentMessages[1]?.at(-1)?.content).toContain("CHEMBL25");
});

test("no tool call means no LanceDB lookup happens", async () => {
  let toolRuns = 0;
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) return reply("Paracetamol is an analgesic.");
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => { toolRuns += 1; return { result: null }; },
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "What is an analgesic?" }] }),
    }),
  );

  const events = await collectEvents(response);
  expect(toolRuns).toBe(0);
  expect(events.some(e => e.tool)).toBe(false);
  expect(answerOf(events)).toBe("Paracetamol is an analgesic.");
});

test("the tool loop stops at maxToolSteps", async () => {
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    maxToolSteps: 2,
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      // A model stuck in a loop: every turn asks for another lookup.
      if (url.endsWith("/api/chat")) return reply(`{"tool": "draw_molecule", "args": {"name": "Ethanol"}}`);
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => ({ result: { image: "data:image/png;base64,AAA" } }),
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "tell me about ethanol" }] }),
    }),
  );

  const events = await collectEvents(response);
  expect(events.filter(e => e.tool).length).toBe(2);
  expect(events.at(-1)?.done).toBe(true);
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

test("prose streams token by token, and a tool call never leaks to the client", async () => {
  const replies = [
    // Narration then a tool call: the prose streams, the JSON is held back.
    `Let me look that up.\n{"tool": "get_compound_by_name", "args": {"name": "Aspirin"}}`,
    "Aspirin is CHEMBL25.",
  ];

  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      // 4-character chunks: forces the tool-call JSON to arrive split up.
      if (url.endsWith("/api/chat")) return reply(replies.shift() ?? "no more replies", 4);
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => ({ result: { chembl_id: "CHEMBL25" } }),
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "What is Aspirin?" }] }),
    }),
  );

  const events = await collectEvents(response);
  const deltas = events.filter(e => e.message?.content !== undefined);
  expect(deltas.length).toBeGreaterThan(1); // streamed, not one blob
  const text = answerOf(events);
  expect(text).toContain("Let me look that up.");
  expect(text).toContain("Aspirin is CHEMBL25.");
  expect(text).not.toContain('"tool"'); // the call itself is never shown
  expect(events.find(e => e.tool)?.tool?.tool).toBe("get_compound_by_name");
});

test("a brace in prose is released, not swallowed", async () => {
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) return reply('Formula {C9H8O4} is aspirin.', 4);
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => ({ result: null }),
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "formula?" }] }),
    }),
  );

  expect(answerOf(await collectEvents(response))).toBe("Formula {C9H8O4} is aspirin.");
});

test("a draw request is routed by the model, not by a keyword bridge", async () => {
  // This used to be intercepted before the model got a turn, because the
  // fine-tune could neither emit a tool call nor read a result. It now does
  // both at 100% on held-out drugs (roadmap item 3), so the bridge is gone and
  // the request goes through the ordinary agent loop.
  const calls: { tool: string; args: Record<string, unknown> }[] = [];
  const replies = [
    `{"tool": "draw_molecule", "args": {"name": "prozac"}}`,
    "FLUOXETINE HYDROCHLORIDE (CHEMBL1201082), C17H19ClF3NO.",
  ];
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) return reply(replies.shift() ?? "done");
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async (call) => {
      calls.push(call);
      return {
        result: {
          image: "data:image/png;base64,AAA",
          smiles: "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1.Cl",
          source: "ChEMBL",
          chembl_id: "CHEMBL1201082",
          pref_name: "FLUOXETINE HYDROCHLORIDE",
          full_molformula: "C17H19ClF3NO",
        },
      };
    },
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: "Show me the molecular structure of prozac" }],
      }),
    }),
  );

  const events = await collectEvents(response);
  expect(calls).toEqual([{ tool: "draw_molecule", args: { name: "prozac" } }]);
  expect(events[0]?.tool).toEqual({ tool: "draw_molecule", args: { name: "prozac" } });
  expect(events[1]?.toolResult?.tool).toBe("draw_molecule");
  // The caption is the model's, written from the result it was handed.
  expect(answerOf(events)).toBe("FLUOXETINE HYDROCHLORIDE (CHEMBL1201082), C17H19ClF3NO.");
  expect(events.at(-1)?.done).toBe(true);
});

test("an ordinary question does not trigger the bridge", async () => {
  let toolRuns = 0;
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) return reply("Aspirin inhibits COX-1.");
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => { toolRuns += 1; return { result: null }; },
  });

  await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "What does Aspirin target?" }] }),
    }),
  ).then(collectEvents);

  expect(toolRuns).toBe(0);
});

test("a call naming a tool we do not have is corrected, not shown to the user", async () => {
  const replies = [
    // No such tool, and nothing TOOL_ALIASES can map it onto — a near-miss name
    // like query_compound_by_name is corrected before it ever reaches here.
    `{"tool": "lookup_the_drug_thing", "args": {"name": "Prozac"}}`,
    `{"tool": "get_compound_by_name", "args": {"name": "Prozac"}}`,
    "Prozac is fluoxetine, CHEMBL1201082.",
  ];
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) return reply(replies.shift() ?? "done");
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => ({ result: { chembl_id: "CHEMBL1201082" } }),
  });

  const response = await handler(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "what is prozac" }] }),
    }),
  );

  const events = await collectEvents(response);
  const answer = answerOf(events);
  expect(answer).toBe("Prozac is fluoxetine, CHEMBL1201082.");
  expect(answer).not.toContain("lookup_the_drug_thing"); // never leaked as prose
  const errored = events.find(e => e.toolResult?.error);
  expect(errored?.toolResult?.tool).toBe("lookup_the_drug_thing");
  expect(errored?.toolResult?.error).toContain("Unknown tool");
});

test("the fine-tune is not sent a system prompt", async () => {
  // Its training records carry none, so the tool list is out-of-distribution
  // text it copies from: routing was 50% with it and 100% without.
  let sentRoles: string[] = [];
  const handler = createChatRequestHandler({
    publicDir: new URL("../public/", import.meta.url),
    fetchImpl: async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) {
        const body = JSON.parse(String((init as RequestInit)?.body ?? "{}"));
        sentRoles = (body.messages ?? []).map((m: { role: string }) => m.role);
        return reply("Aspirin is a painkiller.");
      }
      throw new Error(`Unexpected request: ${url}`);
    },
    toolRunner: async () => ({ result: {} }),
  });

  await collectEvents(
    await handler(
      new Request("http://localhost/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: "what is aspirin" }] }),
      }),
    ),
  );

  expect(sentRoles).not.toContain("system");
  expect(sentRoles).toEqual(["user"]);
});
