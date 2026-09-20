import {
  describeDrawResult,
  formatToolResult,
  parseToolCall,
  routeDirectToolCall,
  runTool,
  TOOL_RESULT_HEADER,
  unknownToolError,
  unknownToolName,
  type ToolCall,
  type ToolOutcome,
} from "./tools";

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
};

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
  const fallbackModelName = options.fallbackModelName ?? Bun.env.OLLAMA_MODEL_NAME ?? "chembl-drug-chat:1b";
  const publicDir = options.publicDir ?? new URL("../public/", import.meta.url);
  const fetchImpl = options.fetchImpl ?? fetch;
  const toolRunner = options.toolRunner ?? ((call: ToolCall) => runTool(call));
  const maxToolSteps = options.maxToolSteps ?? 3;
  const now = options.now ?? (() => Date.now());
  const cacheTtlMs = options.cacheTtlMs ?? 15_000;

  let cachedModel: CachedModel | null = null;

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
  async function agentChat(messages: ChatMessage[]): Promise<Response> {
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

        try {
          // TEMPORARY bridge — see routeDirectToolCall in tools.ts. An explicit
          // "draw X" runs the tool before the model gets a turn, so the answer
          // is grounded even though this fine-tune never calls tools itself.
          const direct = routeDirectToolCall(messages.at(-1)?.content ?? "");
          if (direct) {
            steps++;
            send({ tool: direct });
            const outcome = await toolRunner(direct);
            send({ toolResult: { tool: direct.tool, ...outcome } });
            // The caption is written from the tool result, and the model gets no
            // turn at all: handed a tool result this fine-tune echoes its JSON
            // shape rather than reading it. Both halves go when the bridge goes.
            send({ message: { content: describeDrawResult(outcome) } });
            send({ done: true });
            return;
          }

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
            send({ tool: call });
            const outcome = await toolRunner(call);
            send({ toolResult: { tool: call.tool, ...outcome } });
            convo.push({ role: "assistant", content });
            convo.push({ role: "user", content: formatToolResult(call, outcome) });
          }
          send({ done: true });
        } catch (error) {
          send({ error: error instanceof Error ? error.message : "Chat request failed." });
        } finally {
          console.log(JSON.stringify({
            ts: new Date().toISOString(),
            model: modelInfo.model,
            source: modelInfo.source,
            latencyMs: now() - startMs,
            toolCalls: steps,
            promptTokens,
            completionTokens,
          }));
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

    if (request.method === "POST" && url.pathname === "/api/chat") {
      const body = (await request.json()) as { messages?: unknown };
      const messages = normalizeMessages(body.messages);

      if (messages.length === 0) {
        return json({ error: "Send at least one user message." }, 400);
      }

      try {
        return await agentChat(messages);
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
