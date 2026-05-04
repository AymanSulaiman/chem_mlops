# Bun Backend Reference

A comprehensive reference for using Bun as the backend runtime for the `chem_mlops` chat app, written for someone with strong Python + general engineering background.

Covers the batteries-included APIs you'll actually use, the architectural decisions specific to your chat app (Bun + Python sidecar for RAG, Ollama proxying, SSE streaming), and the LanceDB integration story honestly.

Roughly ordered by "what you'll need first" rather than alphabetical.

---

## 1. The mental model

**Bun is three things in one binary:**
1. A JavaScript/TypeScript runtime (alternative to Node.js).
2. A package manager (alternative to npm/yarn/pnpm).
3. A bundler and test runner (alternative to esbuild + Jest).

For your purposes, the runtime and package manager are what matter. The bundler is occasionally useful for compiling client-side TS.

**How it differs from Node.js:**
- Implements ~95% of the Node.js API. Most npm packages "just work."
- Adds Bun-native APIs (`Bun.serve`, `Bun.file`, `bun:sqlite`, etc.) that are faster and more ergonomic than the Node equivalents.
- Runs TypeScript directly — no `tsc`, no `ts-node`. `bun run server.ts` just works.
- Built on JavaScriptCore (Safari's engine), not V8.
- One sharp edge: **Node native addons that depend on V8 internals don't work.** Native addons that use the stable N-API mostly do work, but module-resolution edge cases exist.

**For your chat app:** the runtime gives you a fast HTTP server, streaming, fetch, file I/O, SQLite, and child processes — all without needing npm packages for any of it. That's the whole point.

---

## 2. Running TypeScript

```bash
bun run server.ts          # run a TS file directly
bun --watch run server.ts  # auto-restart on changes
bun --hot run server.ts    # hot reload (faster, may have edge cases)
```

No `tsc`, no separate compile step. Bun parses, transpiles, and runs.

**What Bun does NOT do:**
Bun *transpiles* TS but does not *type-check* it at runtime. Type errors don't stop your code from running. Run `tsc --noEmit` separately if you want strict type-checking — typically as a pre-commit hook or in CI.

```bash
bun add -d typescript
bunx tsc --noEmit  # type check without emitting
```

**`package.json` scripts:**
```json
{
  "scripts": {
    "dev": "bun --watch run server.ts",
    "start": "bun run server.ts",
    "typecheck": "tsc --noEmit",
    "test": "bun test"
  }
}
```

Run with `bun run dev`, `bun run typecheck`, etc.

---

## 3. Package management

```bash
bun init              # scaffold a new project
bun add lodash        # add a dependency
bun add -d @types/x   # add a dev dependency
bun remove lodash     # remove
bun install           # install everything in package.json
bun update            # update to latest within version ranges
bunx <pkg>            # run a package without installing (like npx)
```

`bun install` is dramatically faster than `npm install` (often 10-100x). The lockfile is `bun.lockb` (binary) — don't hand-edit it; check it into git.

**Note:** mixing `npm install` and `bun install` in the same project can cause version conflicts because they use different lockfile formats. Pick one and stick with it.

---

## 4. `Bun.serve` — the HTTP server

The single most important Bun API for your chat app.

### Basic shape

```ts
const server = Bun.serve({
  port: 3000,
  fetch(req: Request): Response | Promise<Response> {
    return new Response("hello world");
  },
});

console.log(`Listening on http://localhost:${server.port}`);
```

The `fetch` handler takes a standard `Request` (from the Web Fetch API, not Node's `http.IncomingMessage`) and returns a `Response`. Same shape as Cloudflare Workers, Deno, and modern web APIs.

### Routing — manual

There's no built-in router; you switch on the URL. For your chat app's small surface, this is fine.

```ts
Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/api/chat" && req.method === "POST") {
      return handleChat(req);
    }

    if (url.pathname === "/health") {
      return new Response("ok");
    }

    // Static files
    if (req.method === "GET") {
      return serveStatic(url.pathname);
    }

    return new Response("Not Found", { status: 404 });
  },
});
```

For larger apps, libraries like Hono or Elysia provide routing, but you don't need them yet.

### Reading the request body

```ts
async fetch(req) {
  // JSON
  const body = await req.json();

  // Text
  const text = await req.text();

  // Binary
  const buffer = await req.arrayBuffer();

  // FormData (file uploads)
  const form = await req.formData();
}
```

`req.json()` returns `Promise<any>`. Treat the result as `unknown` and validate before using.

### Writing the response — basics

```ts
new Response("body text");
new Response(JSON.stringify({ ok: true }), {
  headers: { "Content-Type": "application/json" },
});
new Response(null, { status: 404 });
new Response("redirecting", { status: 302, headers: { Location: "/foo" } });
```

### Streaming responses (the key thing for SSE)

For the streaming chat responses, you build a `ReadableStream` and pass it as the body:

```ts
async fetch(req) {
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      for (const char of "Hello world") {
        controller.enqueue(encoder.encode(`data: ${char}\n\n`));
        await Bun.sleep(30);
      }
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
```

The `data: <text>\n\n` format is Server-Sent Events — the simplest streaming protocol the browser understands natively.

For real Ollama / RAG streaming, you'd pipe tokens from the upstream into this stream as they arrive. See section 12 for the full pattern.

### Static file serving

```ts
async fetch(req) {
  const url = new URL(req.url);
  let path = url.pathname === "/" ? "/index.html" : url.pathname;
  const file = Bun.file(`./public${path}`);
  if (await file.exists()) {
    return new Response(file);  // Bun.file streams + sets MIME automatically
  }
  return new Response("Not Found", { status: 404 });
}
```

`Bun.file()` is lazy — it doesn't read the file until you await it or pass it to `Response`. Streaming and MIME detection are automatic.

### WebSockets

If you ever want bidirectional comms (you don't, for the basic chat app — SSE is fine), Bun.serve has them built in:

```ts
Bun.serve({
  port: 3000,
  fetch(req, server) {
    if (server.upgrade(req)) return;  // WebSocket upgrade
    return new Response("not a websocket");
  },
  websocket: {
    message(ws, msg) { ws.send(`echo: ${msg}`); },
    open(ws) { ws.send("hello"); },
    close(ws) { /* ... */ },
  },
});
```

### Lifecycle and graceful shutdown

```ts
const server = Bun.serve({ /* ... */ });

process.on("SIGINT", () => {
  console.log("shutting down");
  server.stop();
  process.exit(0);
});
```

`server.stop()` stops accepting new connections. `server.stop(true)` aborts in-flight ones too.

### Performance notes

`Bun.serve` is significantly faster than Node's `http.createServer` and Express — typically 3-5x throughput, much lower latency variance. For a personal chat app this doesn't matter. For a production RAG service it does.

---

## 5. `Bun.file` — file I/O

Lazy, fast, MIME-aware file API.

```ts
const f = Bun.file("./data.json");

await f.exists();         // boolean
await f.text();           // string
await f.json();           // parsed JSON, type any
await f.arrayBuffer();    // binary
await f.bytes();          // Uint8Array (newer alias)
f.size;                   // bytes (sync, after stat)
f.type;                   // MIME type (e.g., "application/json")
f.stream();               // ReadableStream<Uint8Array>
```

Writing:

```ts
await Bun.write("./out.txt", "hello");
await Bun.write("./out.json", JSON.stringify(data));
await Bun.write("./copy.bin", Bun.file("./source.bin"));  // file-to-file
await Bun.write("./out.txt", new Response("from response"));
```

`Bun.write` is overloaded — it accepts strings, Blobs, Files, Responses, Uint8Arrays, etc. Just pass whatever you have.

---

## 6. `fetch` — outbound HTTP

Standard Web Fetch API, available globally — no import.

```ts
const res = await fetch("https://api.example.com/data");
const data = await res.json();

// POST
const res2 = await fetch("https://api.example.com/data", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "Alice" }),
});

// Streaming (e.g., reading from Ollama)
const res3 = await fetch("http://localhost:11434/api/chat", {
  method: "POST",
  body: JSON.stringify({ model: "...", messages: [...], stream: true }),
});
const reader = res3.body!.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value, { stream: true });
  // process chunk
}
```

Bun's `fetch` is fast and supports HTTP/2 by default. Same API as the browser, so streaming code is portable.

---

## 7. Streams — the conceptual key

Both Bun and the browser use the WHATWG Streams API. Three types:

- **`ReadableStream<T>`** — produces values
- **`WritableStream<T>`** — consumes values
- **`TransformStream<I, O>`** — both

For your chat app, you mostly create `ReadableStream`s for SSE responses and consume `ReadableStream`s from `fetch`.

### Consuming a stream — three ways

```ts
// 1. .getReader() loop (most flexible)
const reader = stream.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  // value is a Uint8Array (for binary streams)
}

// 2. for-await-of (often cleanest)
for await (const chunk of stream) {
  // chunk is Uint8Array
}

// 3. Convert to string/buffer/etc
const text = await new Response(stream).text();
```

### Creating a stream

```ts
const stream = new ReadableStream({
  async start(controller) {
    controller.enqueue("first");
    controller.enqueue("second");
    controller.close();
  },
  // optional:
  pull(controller) { /* called when consumer wants more */ },
  cancel(reason) { /* called if consumer aborts */ },
});
```

### `TextEncoder` and `TextDecoder` — string ↔ bytes

```ts
const encoder = new TextEncoder();
const bytes: Uint8Array = encoder.encode("hello");

const decoder = new TextDecoder();
const str: string = decoder.decode(bytes);

// For chunked streams, use { stream: true } so partial UTF-8 sequences
// are buffered correctly across chunks
decoder.decode(chunk, { stream: true });
```

You'll write this exact pattern dozens of times in streaming code. Memorize it.

### Async generators — alternative to manual stream loops

```ts
async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      if (event.startsWith("data: ")) {
        yield event.slice(6);
      }
    }
  }
}

for await (const chunk of parseSSE(response.body!)) {
  console.log(chunk);
}
```

Async generators (`async function*`) compose better than manual reader loops for parsing. Use them for stream transformations.

---

## 8. Environment variables

```bash
# .env
PORT=3000
OLLAMA_URL=http://localhost:11434
LANCEDB_PATH=./data/vectors
```

Bun loads `.env` automatically. Access:

```ts
const port = Bun.env.PORT;          // string | undefined
const port2 = process.env.PORT;     // same thing
```

For typed access:

```ts
function envOrThrow(key: string): string {
  const v = Bun.env[key];
  if (!v) throw new Error(`missing env: ${key}`);
  return v;
}

const port = parseInt(envOrThrow("PORT"));
```

`.env.local`, `.env.production`, etc. — Bun loads them based on `NODE_ENV`. Don't commit secrets.

---

## 9. `Bun.spawn` — child processes

Critical for the chat app: this is how Bun talks to the Python RAG service.

```ts
const proc = Bun.spawn(["python", "rag.py", "--query", "aspirin"], {
  cwd: "./rag",
  env: { ...process.env, FOO: "bar" },
  stdout: "pipe",   // default
  stderr: "pipe",
  stdin: "pipe",
});

// Read all stdout at once
const output = await new Response(proc.stdout).text();
const exitCode = await proc.exited;

// Write to stdin
proc.stdin.write("input data\n");
proc.stdin.end();

// Stream stdout
for await (const chunk of proc.stdout) {
  console.log(decoder.decode(chunk));
}

// Kill
proc.kill();
```

For one-shot calls (run a script, get output) this is great. For continuous interaction, see section 12 for the sidecar pattern.

---

## 10. `bun:sqlite` — embedded database

Built-in synchronous SQLite. Useful if you ever want to persist chat history, eval results, or RAG metadata.

```ts
import { Database } from "bun:sqlite";

const db = new Database("./chat.db");
db.exec("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)");

// Prepared statements (recommended for repeated queries)
const insert = db.prepare("INSERT INTO messages (role, content) VALUES (?, ?)");
insert.run("user", "Hello");
insert.run("assistant", "Hi there");

// Query
const all = db.prepare("SELECT * FROM messages ORDER BY id DESC LIMIT 10");
const rows = all.all();   // array of all results
const one = all.get();    // first result or undefined

// Named parameters
const byRole = db.prepare("SELECT * FROM messages WHERE role = $role");
byRole.all({ $role: "user" });

// Transactions
const insertMany = db.transaction((items: { role: string; content: string }[]) => {
  for (const item of items) insert.run(item.role, item.content);
});
insertMany([
  { role: "user", content: "Q1" },
  { role: "assistant", content: "A1" },
]);

// WAL mode for better concurrency (recommended for most apps)
db.exec("PRAGMA journal_mode = WAL;");
```

The API is *synchronous* — operations block. SQLite is fast enough that this is fine for local apps. For high-concurrency servers it's still fine because SQLite operations are microsecond-scale.

You probably won't need this for the basic chat app. It becomes useful when you add chat history, eval logging, or persistent RAG metadata.

---

## 11. LanceDB on Bun — the honest story

Three relevant facts:

1. LanceDB ships an official TypeScript SDK: `@lancedb/lancedb`.
2. The SDK uses a Rust core wrapped via Node-API native bindings (`.node` files).
3. Bun implements ~95-100% of Node-API and runs most native addons that use it.

**The optimistic answer:** install `@lancedb/lancedb` with `bun add @lancedb/lancedb`, import it, use it. It often works.

**The honest answer:** it sometimes works, and there are known sharp edges. Issues seen in the wild include:
- Module resolution edge cases for the platform-specific binary (e.g., `@lancedb/lancedb-darwin-arm64` not being found in some configurations)
- Bundler interactions (esbuild, webpack) breaking native binary loading; this matters less for plain Bun runtime usage
- Occasional N-API behavioral parity gaps (mostly around `async_hooks`, threading, and `uv_*` semantics) that affect specific edge cases

The Bun team actively works to fix these and the situation has been improving steadily. Status as of mid-2026: **try it first; have a fallback plan.**

### Direct LanceDB-on-Bun (the optimistic path)

```ts
import * as lancedb from "@lancedb/lancedb";

const db = await lancedb.connect("./data/lancedb");

const table = await db.createTable("chunks", [
  { id: 1, vector: [0.1, 0.2, 0.3], text: "first chunk" },
  { id: 2, vector: [0.5, 0.4, 0.6], text: "second chunk" },
]);

const results = await table
  .vectorSearch([0.1, 0.3, 0.2])
  .limit(5)
  .toArray();
```

If that runs without complaint, you're set. Go this route.

### Python sidecar path (the reliable path)

Given:
- You're already building the RAG side in Python
- LanceDB's Python SDK is mature, well-trodden, and actively maintained
- Embedding models, reranking, and chunking ergonomics are nicer in Python today
- You don't gain much by porting to TS

…the cleanest architecture is **Python owns LanceDB and the RAG pipeline; Bun talks to it over HTTP.**

```
[Browser]
   │  POST /api/chat (mode=rag)
   ▼
[Bun: server.ts]                 ← HTTP server, SSE streaming, Ollama proxy
   │  POST /rag (query, k)
   ▼
[Python: FastAPI sidecar]        ← LanceDB + retrieval + prompt building
   │  retrieves chunks → builds prompt → calls Ollama or external LLM
   ▼
[returns answer + chunks, streamed back through Bun to the browser]
```

Why this is better for your project:
- Both runtimes do what they're good at. Python keeps the data/ML side; Bun handles the user-facing server.
- The boundary is HTTP — language-agnostic, debuggable, replaceable.
- You can swap the Python implementation later (or rewrite it in Rust/Go) without touching Bun.
- Decouples deployment: you can host the sidecar on a GPU instance and the Bun server elsewhere.

The cost: an extra process to run. In dev: `python -m uvicorn rag:app --port 8001` in one terminal, Bun in another. In prod: docker compose or two separate services.

### Sidecar talking pattern in Bun

```ts
async function ragQuery(query: string): Promise<{ answer: ReadableStream; chunks: string[] }> {
  const res = await fetch("http://localhost:8001/rag", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k: 5 }),
  });

  if (!res.ok) {
    throw new Error(`RAG sidecar error: ${res.status}`);
  }

  // If sidecar streams, forward the stream
  return {
    answer: res.body!,
    chunks: [],  // or pulled from a separate header / leading event
  };
}
```

For chat-app-shaped RAG, have the Python sidecar stream SSE itself. Bun just relays the stream to the browser. The full pattern is in section 12.

### Recommendation

Document the choice in `DECISIONS.md`. Both approaches are valid. **Default to the sidecar.** It's the boring, correct architectural choice for your build, and "boring + correct" is a feature in interviews.

---

## 12. Architectural pattern: SSE streaming with mode-based routing

The full shape of the chat app's `/api/chat` endpoint:

```ts
import { Database } from "bun:sqlite";  // optional, if logging

interface ChatRequest {
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  mode: "finetune" | "rag";
}

Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/api/chat" && req.method === "POST") {
      const body = await req.json() as ChatRequest;
      return handleChat(body);
    }

    // ... static file serving, etc.
    return new Response("Not Found", { status: 404 });
  },
});

async function handleChat(body: ChatRequest): Promise<Response> {
  const upstream =
    body.mode === "finetune"
      ? await callOllama(body.messages)
      : await callRagSidecar(body.messages);

  // Convert upstream NDJSON or SSE to SSE for the browser
  const stream = transformToSSE(upstream);

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}

async function callOllama(messages: ChatRequest["messages"]): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "chembl-drug-chat:1b",
      messages,
      stream: true,
    }),
  });
  if (!res.body) throw new Error("Ollama returned no body");
  return res.body;
}

async function callRagSidecar(messages: ChatRequest["messages"]): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch("http://localhost:8001/rag/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.body) throw new Error("RAG sidecar returned no body");
  return res.body;
}

function transformToSSE(upstream: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const reader = upstream.getReader();
      let buffer = "";
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // Ollama emits NDJSON: one JSON object per line
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            const text = event.message?.content ?? "";
            if (text) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "token", text })}\n\n`));
            }
          }
        }
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "done" })}\n\n`));
      } catch (err) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "error", error: String(err) })}\n\n`));
      } finally {
        controller.close();
      }
    },
  });
}
```

This is the rough shape. You'll evolve it. Keep the **transform step** as a clean function so swapping upstream protocols (NDJSON → SSE → JSON-Lines) doesn't ripple through the rest.

---

## 13. Testing with `bun:test`

Built-in Jest-compatible test runner.

```ts
// chat.test.ts
import { describe, test, expect, beforeEach } from "bun:test";
import { parseSSE } from "./stream";

describe("parseSSE", () => {
  test("parses single event", async () => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode("data: hello\n\n"));
        c.close();
      },
    });
    const events: string[] = [];
    for await (const e of parseSSE(stream)) events.push(e);
    expect(events).toEqual(["hello"]);
  });
});
```

Run with:
```bash
bun test                    # all tests
bun test stream             # files matching "stream"
bun test --watch
```

Most Jest APIs work (`describe`, `test`, `expect`, `beforeEach`, `mock.fn()`, etc.). Some — `jest.mock`, advanced mocking — differ. For your build, basic test cases are plenty.

---

## 14. Bundling client TypeScript

For your chat app's frontend (`src/main.ts` etc.), Bun's bundler ships TS to JS for the browser.

```bash
# One-shot
bun build ./src/main.ts --outdir ./public/js

# With watch
bun build ./src/main.ts --outdir ./public/js --watch

# Minified for prod
bun build ./src/main.ts --outdir ./public/js --minify

# Multiple entrypoints
bun build ./src/main.ts ./src/admin.ts --outdir ./public/js
```

`package.json` scripts:
```json
{
  "scripts": {
    "build:client": "bun build ./src/main.ts --outdir ./public/js",
    "dev:client": "bun build ./src/main.ts --outdir ./public/js --watch"
  }
}
```

For the chat app, you don't need anything more sophisticated.

---

## 15. Useful Bun-specific globals

Things that are just there, no import needed:

```ts
Bun.version           // e.g., "1.3.0"
Bun.env               // alias for process.env
Bun.argv              // command line args
Bun.sleep(ms)         // returns a Promise that resolves after ms
Bun.password.hash("secret");   // bcrypt-shaped, no native addon
Bun.password.verify(hash, "secret");
Bun.hash(data)        // fast non-crypto hash
Bun.escapeHTML(str)   // string escape
crypto.randomUUID()   // standard Web Crypto
```

These are nice-to-haves. Most you'll never use. `Bun.sleep` is genuinely useful for testing streams.

---

## 16. Things you can punt on

These exist, you may see them, you don't need them right now:

- **`Bun.s3`** — S3 client. For when you deploy and want to serve model artifacts from cloud storage.
- **`Bun.SQL` / `Bun.sql`** — built-in Postgres/MySQL/SQLite tagged template literal API. Nice if you ever add Postgres.
- **`Bun.redis`** — built-in Redis client.
- **`Bun.FileSystemRouter`** — file-based routing (Next.js style). Overkill for your app.
- **`Bun.Glob`** — glob pattern matching for file lookup.
- **`bun:ffi`** — call C functions from JS. For when you need raw native performance.
- **`Bun.$`** — shell scripting in JS. Replace bash scripts with TS.
- **`Bun.Transpiler`** — programmatic access to the TS transpiler. Almost never needed.
- **HTMLRewriter** — HTML manipulation, Cloudflare Workers-compatible.

If you ever wonder "does Bun have a built-in for X?" — check `bun.com/docs`. Often it does.

---

## 17. Common pitfalls

**1. Mixing `bun install` and `npm install`.** Different lockfiles. Pick one. Stick with it.

**2. Native addons crashing or failing to load.** If a package depends on `node-gyp` (check `binding.gyp` in the package), it might not work. The error is usually visible at install or first import, not silent.

**3. Long-running scripts exit unexpectedly.** Unlike Node, Bun doesn't keep the process alive without something pending — no listener, no interval, no open handle, and the process exits. `Bun.serve` keeps it alive; pure scripts may not.

**4. `bun run` script has different semantics than running directly.** `bun run server.ts` runs the file. `bun run start` runs the `start` script in `package.json` (which is itself often `bun server.ts`). Easy to confuse.

**5. Hot reload (`--hot`) caching.** Sometimes module state persists across reloads in surprising ways. Use `--watch` for full restarts unless you specifically want hot-reload semantics.

**6. Bun's `fetch` and Node's `fetch` are subtly different.** Bun's is closer to the browser. Most code is portable; some streaming edge cases differ.

**7. `bun:test` is Jest-compatible, not identical.** `jest.mock()` doesn't exist; use `mock.module()`. If you're not coming from Jest, you won't notice.

**8. Native binaries fail in Docker for the wrong arch.** If you build on M1 and deploy to Linux x64, make sure the lockfile resolves the right platform binary. `bun install --frozen-lockfile` on the target arch is the safe path.

---

## 18. Recommended mental defaults for this project

When in doubt:
- Use `Bun.serve` for HTTP. Don't reach for Express.
- Use `fetch` for outbound HTTP. Don't reach for axios or node-fetch.
- Use `bun:sqlite` if you need persistence. Don't reach for better-sqlite3 or sqlite3.
- Use `bun:test` for tests. Don't reach for Jest or Vitest.
- Use `Bun.spawn` for subprocesses. Don't reach for child_process unless you specifically need its quirks.
- Use SSE for streaming to the browser. Don't reach for WebSockets unless you genuinely need bidirectional.
- Use a Python sidecar for LanceDB and RAG. Don't try to port the RAG layer to TS.
- Use `Bun.file` and `Bun.write` for file I/O. Don't reach for `fs/promises` unless you need a specific Node API.

When you violate these defaults, write down why in `DECISIONS.md`. Future you will thank present you.

---

## End

This doc covers the 95% of Bun APIs you'll actually touch in the chat app build. The remaining 5% is in the official docs at `bun.com/docs` — well-organized and worth a skim once you've shipped the basic version.

The single most valuable thing you can internalize from this document: **Bun is batteries-included on purpose, and the batteries are usually the right answer.** When you find yourself reaching for a Node.js library to solve a problem Bun has built-in, stop and check. The built-in is almost always faster, simpler, and what hiring managers will recognize as "they actually understood the runtime."
