import { createChatRequestHandler } from "./src/app";

const PORT = Number(Bun.env.PORT ?? 3000);
const FRONTEND_ENTRY = "./src/frontend.ts";

// Build the browser module from TypeScript so the page can load a plain JS file.
// The Docker image builds it at image time and sets FRONTEND_PREBUILT to skip this.
if (!Bun.env.FRONTEND_PREBUILT) {
  await Bun.build({
    entrypoints: [FRONTEND_ENTRY],
    outdir: "./public",
    target: "browser",
  });
}

// Serve the web app, the compiled frontend bundle, and the Ollama-backed API.
Bun.serve({
  port: PORT,
  // A tool loop makes several model calls before the answer; the 10s default
  // cuts the response off mid-stream. 255s is Bun's maximum.
  idleTimeout: 255,
  // A full 20-message conversation is well under this; Bun's default is 128 MB.
  maxRequestBodySize: 256 * 1024,
  fetch: createChatRequestHandler(),
});

if (!Bun.env.GOOGLE_CLIENT_ID) {
  console.warn(
    "Sign-in is OFF (GOOGLE_CLIENT_ID unset): every visitor counts as one person. " +
      "Fine locally; set it before exposing the demo.",
  );
}
