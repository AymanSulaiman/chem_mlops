import { augmentMessages, buildRagContext } from "./rag";

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
  ragModelName?: string;
  ragLancedbDir?: string;
  publicDir?: URL;
  fetchImpl?: typeof fetch;
  now?: () => number;
  cacheTtlMs?: number;
};

type CachedModel = {
  model: string;
  source: "ollama-tags" | "fallback";
  fetchedAt: number;
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
  const ragModelName = options.ragModelName ?? Bun.env.RAG_MODEL_NAME ?? "gemma3:1b";
  const ragLancedbDir = options.ragLancedbDir ?? Bun.env.LANCEDB_DIR ?? null;
  const publicDir = options.publicDir ?? new URL("../public/", import.meta.url);
  const fetchImpl = options.fetchImpl ?? fetch;
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

  async function chat(messages: ChatMessage[], modelOverride?: string) {
    const modelInfo = modelOverride
      ? { model: modelOverride, source: "rag-model" as const }
      : await detectLatestModel();
    const response = await fetchImpl(`${ollamaBaseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: modelInfo.model,
        messages,
        stream: false,
      }),
    });

    if (!response.ok) {
      const details = await response.text();
      throw new Error(`Ollama chat failed: ${response.status} ${details}`);
    }

    const payload = (await response.json()) as { message?: { content?: string } };

    return {
      model: modelInfo.model,
      source: modelInfo.source,
      reply: payload.message?.content ?? "",
    };
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
      const body = (await request.json()) as { messages?: unknown; mode?: string };
      const messages = normalizeMessages(body.messages);

      if (messages.length === 0) {
        return json({ error: "Send at least one user message." }, 400);
      }

      try {
        const isRag = body.mode === "rag";
        let outgoing = messages;
        if (isRag && ragLancedbDir !== null) {
          const lastUserMessage = [...messages].reverse().find(m => m.role === "user");
          if (lastUserMessage) {
            const context = await buildRagContext(lastUserMessage.content, ragLancedbDir);
            if (context) {
              outgoing = augmentMessages(messages, context);
            }
          }
        }
        return json(await chat(outgoing, isRag ? ragModelName : undefined));
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
