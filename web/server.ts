import { createChatRequestHandler } from "./src/app";
import { DEFAULT_LANCEDB_DIR } from "./src/rag";

const PORT = Number(Bun.env.PORT ?? 3000);
const FRONTEND_ENTRY = "./src/frontend.ts";

// Build the browser module from TypeScript so the page can load a plain JS file.
await Bun.build({
  entrypoints: [FRONTEND_ENTRY],
  outdir: "./public",
  target: "browser",
});

// Serve the web app, the compiled frontend bundle, and the Ollama-backed API.
Bun.serve({
  port: PORT,
  fetch: createChatRequestHandler({
    ragModelName: Bun.env.RAG_MODEL_NAME ?? "gemma3:1b",
    ragLancedbDir: Bun.env.LANCEDB_DIR ?? DEFAULT_LANCEDB_DIR,
  }),
});

console.log(`Server running at http://localhost:${PORT}`);
