import {
  formatToolResult,
  parseToolCall,
  runTool,
  TOOL_RESULT_HEADER,
  unknownToolError,
  unknownToolName,
  type ToolCall,
  type ToolOutcome,
} from "./tools";
import { MAX_MESSAGE_CHARS, MAX_MESSAGES, MAX_USER_CHARS } from "./frontend-helpers";
import { createGate, createGoogleVerifier, createSessions, type GateOptions } from "./access";
import { appendFile } from "node:fs/promises";

// Usage log: one JSON line per sign-in and per chat, to stdout and, when
// USAGE_LOG is set (the VM: a Docker volume), to a file that survives redeploys.
function logEvent(event: Record<string, unknown>) {
  const line = JSON.stringify({ ts: new Date().toISOString(), ...event });
  console.log(line);
  const file = Bun.env.USAGE_LOG;
  if (file) appendFile(file, `${line}\n`).catch((error) => console.error(error));
}

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type OllamaModel = {
  name?: string;
  modified_at?: string;
};

export type ChatAppOptions = {
  ollamaBaseUrl?: string;
  ollamaModelPrefix?: string;
  fallbackModelName?: string;
  publicDir?: URL;
  // Only the shape the handler uses, so tests can pass a plain function.
  fetchImpl?: (input: URL | RequestInfo, init?: RequestInit) => Promise<Response>;
  toolRunner?: (call: ToolCall) => Promise<ToolOutcome>;
  maxToolSteps?: number;
  now?: () => number;
  cacheTtlMs?: number;
  // Google OAuth client ID for "Sign in with Google". Unset = no sign-in
  // (local runs); every caller is then treated as one user, "local".
  googleClientId?: string | null;
  gate?: GateOptions;
};

export function oversizedReason(messages: ChatMessage[]): string | null {
  if (messages.length > MAX_MESSAGES) return `Conversations are limited to ${MAX_MESSAGES} messages.`;
  for (const m of messages) {
    if (m.role === "user" && m.content.length > MAX_USER_CHARS)
      return `Messages are limited to ${MAX_USER_CHARS.toLocaleString("en")} characters.`;
    if (m.content.length > MAX_MESSAGE_CHARS) return "A message in this conversation is too long.";
  }
  return null;
}


type CachedModel = {
  model: string;
  source: "ollama-tags" | "fallback";
  fetchedAt: number;
};

type OllamaReply = {
  message?: { content?: string };
  prompt_eval_count?: number;
  eval_count?: number;
};

export function isValidChatMessage(value: unknown): value is ChatMessage {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as ChatMessage).role !== undefined &&
    ((value as ChatMessage).role === "user" || (value as ChatMessage).role === "assistant") &&
    typeof (value as ChatMessage).content === "string" &&
    (value as ChatMessage).content.trim().length > 0
  );
}

export function normalizeMessages(messages: unknown): ChatMessage[] {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.filter(isValidChatMessage).map((message) => ({
    role: message.role,
    content: message.content.trim(),
  }));
}

export function pickLatestModel(models: OllamaModel[], prefix: string): string | null {
  const matches = models.filter(
    (model) => typeof model.name === "string" && model.name.startsWith(prefix),
  );

  if (matches.length === 0) {
    return null;
  }

  matches.sort((left, right) => {
    const leftTime = Date.parse(left.modified_at ?? "") || 0;
    const rightTime = Date.parse(right.modified_at ?? "") || 0;
    return rightTime - leftTime;
  });

  return matches[0]?.name ?? null;
}

function json(data: unknown, status = 200) {
  return Response.json(data, { status });
}

function resolveContentType(path: string) {
  if (path.endsWith(".css")) return "text/css; charset=utf-8";
  if (path.endsWith(".js")) return "application/javascript; charset=utf-8";
  return "text/html; charset=utf-8";
}

function readTextFile(path: string, publicDir: URL) {
  return new Response(Bun.file(new URL(path, publicDir)), {
    headers: {
      "content-type": resolveContentType(path),
    },
  });
}

export function createChatRequestHandler(options: ChatAppOptions = {}) {
  const ollamaBaseUrl = options.ollamaBaseUrl ?? Bun.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434";
  const ollamaModelPrefix = options.ollamaModelPrefix ?? Bun.env.OLLAMA_MODEL_PREFIX ?? "chembl-drug-chat";
  const fallbackModelName = options.fallbackModelName ?? Bun.env.OLLAMA_MODEL_NAME ?? "chembl-drug-chat:270m";
  const publicDir = options.publicDir ?? new URL("../public/", import.meta.url);
  const fetchImpl = options.fetchImpl ?? fetch;
  const toolRunner = options.toolRunner ?? ((call: ToolCall) => runTool(call));
  const maxToolSteps = options.maxToolSteps ?? 3;
  const now = options.now ?? (() => Date.now());
  const cacheTtlMs = options.cacheTtlMs ?? 15_000;
  const googleClientId =
    options.googleClientId === undefined ? Bun.env.GOOGLE_CLIENT_ID || null : options.googleClientId;
  const verifyGoogle = googleClientId ? createGoogleVerifier(googleClientId, fetchImpl, now) : null;
  const sessions = createSessions(now);
  // MAX_USERS: seats in the chat at once (the VM's deploy/<vm>.env sets it).
  const gate = createGate({ now, maxUsers: Number(Bun.env.MAX_USERS) || undefined, ...options.gate });

  let cachedModel: CachedModel | null = null;

  // One chat per seat at a time (Ollama's OLLAMA_NUM_PARALLEL matches), and
  // queuing here (not in Ollama) lets a waiting visitor be told so.
  let running = 0;
  let busy = 0;
  const waiters: (() => void)[] = [];
  async function takeTurn(): Promise<() => void> {
    if (running < gate.maxUsers) running++;
    else await new Promise<void>((resolve) => waiters.push(resolve));
    // Hand the slot straight to the next waiter, so `running` stays put.
    return () => {
      const next = waiters.shift();
      if (next) next();
      else running--;
    };
  }

  function visitor(request: Request): string | null {
    return verifyGoogle ? sessions.read(request.headers.get("cookie")) : "local";
  }

  const unauthorized = () => json({ error: "Sign in with Google to use the chat." }, 401);
  const queued = (position: number) =>
    json({ error: `You're #${position} in the queue. The chat opens when it's your turn.`, queued: true, position }, 503);

  async function detectLatestModel() {
    if (cachedModel && now() - cachedModel.fetchedAt < cacheTtlMs) {
      return cachedModel;
    }

    const response = await fetchImpl(`${ollamaBaseUrl}/api/tags`);
    if (!response.ok) {
      throw new Error(`Could not read Ollama models: ${response.status} ${response.statusText}`);
    }

    const payload = (await response.json()) as { models?: OllamaModel[] };
    const latestModel = pickLatestModel(Array.isArray(payload.models) ? payload.models : [], ollamaModelPrefix);

    cachedModel = {
      model: latestModel ?? fallbackModelName,
      source: latestModel ? "ollama-tags" : "fallback",
      fetchedAt: now(),
    };

    return cachedModel;
  }

  // One streamed turn. Ollama refuses its native `tools` field for
  // chembl-drug-chat ("does not support tools" — Gemma 3 has no tool template),
  // so tool calls are prompted and parsed out of the prose instead.
  // onDelta sees each token as it arrives; the caller decides what to forward.
  async function streamOllamaTurn(
    model: string,
    messages: { role: string; content: string }[],
    onDelta: (delta: string) => void,
  ): Promise<OllamaReply> {
    const response = await fetchImpl(`${ollamaBaseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, messages, stream: true }),
    });
    if (!response.ok) {
      throw new Error(`Ollama chat failed: ${response.status} ${await response.text()}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let partial = "";
    const reply: OllamaReply = { message: { content: "" } };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      partial += decoder.decode(value, { stream: true });
      const lines = partial.split("\n");
      partial = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let chunk: OllamaReply;
        try {
          chunk = JSON.parse(line) as OllamaReply;
        } catch {
          continue;
        }
        const delta = chunk.message?.content ?? "";
        if (delta) {
          reply.message!.content += delta;
          onDelta(delta);
        }
        // Token counts only appear on the final chunk.
        reply.prompt_eval_count = chunk.prompt_eval_count ?? reply.prompt_eval_count;
        reply.eval_count = chunk.eval_count ?? reply.eval_count;
      }
    }

    return reply;
  }

  // Agentic chat: loop on tool calls, streaming NDJSON events out as they happen.
  // Prose streams token by token; text from the first "{" onwards is held back
  // until the turn ends, because a tool call cannot be recognised until its JSON
  // closes. Held text that turns out not to be a call is released at the end, so
  // nothing is ever dropped.
  async function agentChat(messages: ChatMessage[], email: string): Promise<Response> {
    const modelInfo = await detectLatestModel();
    const startMs = now();
    // No system prompt. The fine-tune's training records carry none — they are
    // bare ### Question / ### Answer — so prepending the tool list is
    // out-of-distribution text it copies from rather than reasons over: every
    // "is it safe to take X with Y?" collapsed onto whichever tool example sat
    // last in the list. Dropping it took routing from 50% to 100% and
    // tool-result grounding from 60% to 100% on run 20260920_114710_tools.
    // TOOL_SYSTEM_PROMPT stays exported for a general tool-capable model, which
    // does need to be told what the tools are; this model already knows.
    const convo: { role: string; content: string }[] = [...messages];

    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        const encoder = new TextEncoder();
        const send = (event: unknown) =>
          controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
        let promptTokens = 0;
        let completionTokens = 0;
        let steps = 0;
        const toolNames: string[] = [];
        if (busy >= gate.maxUsers) send({ waiting: true });
        busy++;
        const release = await takeTurn();

        try {
          for (let step = 0; ; step++) {
            let forwarded = 0; // characters of this turn already sent to the client
            let holding = false;
            const reply = await streamOllamaTurn(modelInfo.model, convo, (delta) => {
              if (holding) return;
              const brace = delta.indexOf("{");
              const head = brace === -1 ? delta : delta.slice(0, brace);
              if (head) {
                send({ message: { content: head } });
                forwarded += head.length;
              }
              if (brace !== -1) holding = true;
            });
            const content = reply.message?.content ?? "";
            promptTokens += reply.prompt_eval_count ?? 0;
            completionTokens += reply.eval_count ?? 0;

            const call = step < maxToolSteps ? parseToolCall(content) : null;
            if (!call) {
              // A call naming a tool we do not have: hand the model the error
              // rather than releasing unusable JSON into the transcript.
              const unknown = step < maxToolSteps ? unknownToolName(content) : null;
              if (unknown) {
                send({ toolResult: { tool: unknown, error: unknownToolError(unknown) } });
                convo.push({ role: "assistant", content });
                convo.push({
                  role: "user",
                  content: `${TOOL_RESULT_HEADER} (${unknown})\n${unknownToolError(unknown)}`,
                });
                continue;
              }
              // Release whatever was held back: it was prose, not a tool call.
              const tail = content.slice(forwarded);
              if (tail) send({ message: { content: tail } });
              break;
            }

            steps++;
            toolNames.push(call.tool);
            send({ tool: call });
            const outcome = await toolRunner(call);
            send({ toolResult: { tool: call.tool, ...outcome } });
            if (outcome.error) {
              // Handed an error, the model invents an answer (a made-up
              // structure for Prozac, with no fluorine). Say what failed instead.
              send({ message: { content: `I couldn't get that from the database (${call.tool} failed), so I won't guess. Try rephrasing, or ask about another drug.` } });
              break;
            }
            convo.push({ role: "assistant", content });
            convo.push({ role: "user", content: formatToolResult(call, outcome) });
          }
          send({ done: true });
        } catch (error) {
          send({ error: error instanceof Error ? error.message : "Chat request failed." });
        } finally {
          release();
          busy--;
          // One line per chat: who, when, how long, how much. The question
          // text is not logged.
          logEvent({
            event: "chat",
            email,
            questionChars: messages.at(-1)?.content.length ?? 0,
            tools: toolNames,
            model: modelInfo.model,
            source: modelInfo.source,
            latencyMs: now() - startMs,
            toolCalls: steps,
            promptTokens,
            completionTokens,
          });
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "content-type": "application/x-ndjson",
        "x-model": modelInfo.model,
        "x-source": modelInfo.source,
      },
    });
  }

  return async function handleRequest(request: Request) {
    const url = new URL(request.url);

    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      return readTextFile("index.html", publicDir);
    }

    if (request.method === "GET" && url.pathname === "/style.css") {
      return readTextFile("style.css", publicDir);
    }

    if (request.method === "GET" && url.pathname === "/frontend.js") {
      return readTextFile("frontend.js", publicDir);
    }

    if (request.method === "GET" && url.pathname === "/api/health") {
      return json({ ok: true, ollamaBaseUrl });
    }

    if (request.method === "GET" && url.pathname === "/api/model") {
      try {
        return json(await detectLatestModel());
      } catch (error) {
        return json(
          {
            error: error instanceof Error ? error.message : "Could not detect model.",
          },
          503,
        );
      }
    }

    // What the page needs to show the sign-in button.
    if (request.method === "GET" && url.pathname === "/api/config") {
      return json({ googleClientId, signInRequired: Boolean(verifyGoogle) });
    }

    // The Google button's ID token in, a session cookie out.
    if (request.method === "POST" && url.pathname === "/api/login") {
      if (!verifyGoogle) return json({ error: "Sign-in is not enabled." }, 404);
      let credential: unknown;
      try {
        credential = ((await request.json()) as { credential?: unknown }).credential;
      } catch {
        return json({ error: "Send the sign-in token as JSON." }, 400);
      }
      let email: string | null = null;
      try {
        email = await verifyGoogle(credential);
      } catch (error) {
        console.error(error);
      }
      if (!email) return unauthorized();
      logEvent({ event: "login", email });
      return new Response(JSON.stringify({ email }), {
        headers: { "content-type": "application/json", "set-cookie": sessions.issue(email) },
      });
    }

    if (request.method === "POST" && url.pathname === "/api/logout") {
      const email = visitor(request);
      if (email) gate.leave(email);
      return new Response(null, { status: 204, headers: { "set-cookie": sessions.clear() } });
    }

    // Heartbeat while the chat page is open: joins the queue or keeps your place.
    // { seated: true, position: 0 } = in the chat; otherwise position = place in line.
    if (request.method === "POST" && url.pathname === "/api/session") {
      const email = visitor(request);
      if (!email) return unauthorized();
      return json({ ...gate.claim(email), email });
    }

    // Sent by navigator.sendBeacon when the tab closes, so the next person goes in at once.
    if (request.method === "POST" && url.pathname === "/api/session/leave") {
      const email = visitor(request);
      if (email) gate.leave(email);
      return new Response(null, { status: 204 });
    }

    if (request.method === "POST" && url.pathname === "/api/chat") {
      const email = visitor(request);
      if (!email) return unauthorized();
      const place = gate.claim(email);
      if (!place.seated) return queued(place.position);

      let body: { messages?: unknown };
      try {
        body = (await request.json()) as { messages?: unknown };
      } catch {
        return json({ error: "Send the chat as JSON." }, 400);
      }
      const messages = normalizeMessages(body.messages);

      if (messages.length === 0) {
        return json({ error: "Send at least one user message." }, 400);
      }
      const oversized = oversizedReason(messages);
      if (oversized) return json({ error: oversized }, 413);
      if (!gate.allowChat(email)) {
        return json({ error: "You've reached the chat limit for now. Try again later." }, 429);
      }

      try {
        return await agentChat(messages, email);
      } catch (error) {
        return json(
          {
            error: error instanceof Error ? error.message : "Chat request failed.",
          },
          503,
        );
      }
    }

    return new Response("Not found", { status: 404 });
  };
}
