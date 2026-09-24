import { createChatRequestHandler } from "./src/app";

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
  // A tool loop makes several model calls before the answer; the 10s default
  // cuts the response off mid-stream. 255s is Bun's maximum.
  idleTimeout: 255,
  fetch: createChatRequestHandler(),
});

console.log(`Server running at http://localhost:${PORT}`);
