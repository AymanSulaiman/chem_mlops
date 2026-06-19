# Bun Web App

Browser chat interface for the ChEMBL Drug Chat models. Supports two inference modes selectable via a Standard / RAG pill toggle.

## Modes

| Mode | Model | How it works |
|------|-------|-------------|
| **Standard** | `chembl-drug-chat:1b` | Fine-tuned Gemma 3 1B, no retrieval |
| **RAG** | `gemma3:1b` | Base Gemma 3 1B + LanceDB context injected as system message |

In RAG mode the Bun backend extracts drug-name candidates from the user's message, queries the `compounds` and `polypharmacy` LanceDB tables via the `@lancedb/lancedb` TypeScript client (reads the same Lance files written by the Python pipeline — no Python server needed), and prepends a pharmacological context block before forwarding to Ollama.

## Folder structure

```text
web/
├── public/
│   ├── index.html          # Standard / RAG pill toggle + chat UI
│   └── style.css
├── src/
│   ├── app.ts              # Request handler, model detection, Standard/RAG routing
│   ├── rag.ts              # extractDrugCandidates, buildRagContext, augmentMessages
│   ├── frontend.ts         # Browser: mode toggle state, chat history, submit handler
│   └── frontend-helpers.ts # Formatting helpers (testable, no DOM deps)
├── test/
│   ├── app.test.ts         # Handler routing, model detection, RAG model selection
│   ├── rag.test.ts         # extractDrugCandidates, augmentMessages, buildRagContext
│   ├── chat.test.ts
│   ├── frontend.test.ts
│   └── model.test.ts
├── package.json
└── server.ts               # Bun.serve entry point
```

## How it works

```mermaid
flowchart LR
    BR([Browser]) -->|POST /api/chat\nmode: standard | raft| BUN[Bun backend\nsrc/app.ts]

    BUN -->|Standard\ndetect latest chembl-drug-chat| OLL[(Ollama\nchembl-drug-chat:1b)]

    BUN -->|RAG\nbuildRagContext| LDB[(LanceDB\ncompounds +\npolypharmacy)]
    LDB -->|context string| BUN
    BUN -->|augmented messages| OLL2[(Ollama\ngemma3:1b)]

    OLL -->|reply| BR
    OLL2 -->|reply| BR
```

### `server.ts`

- Bundles `src/frontend.ts` into `public/frontend.js` at startup
- Starts Bun.serve on port `3000`
- Passes `ragModelName` and `ragLancedbDir` to the app handler

### `src/app.ts`

- Normalises chat messages
- Detects the latest `chembl-drug-chat:*` Ollama model (cached 15 s)
- Routes `mode: "standard"` → fine-tuned model
- Routes `mode: "rag"` → `buildRagContext` → `augmentMessages` → base `gemma3:1b`
- Serves `index.html`, `style.css`, `frontend.js`
- `/api/health` and `/api/model` debug endpoints

### `src/rag.ts`

- `extractDrugCandidates(text)` — regex extracts capitalised words, filters stopwords, title-cases to match DB format
- `buildRagContext(message, lancedbDir)` — queries `compounds` table by name and `polypharmacy` table for pair signals and top per-drug partners; returns a bullet-list context string or `null` if nothing is found
- `augmentMessages(messages, context)` — prepends `{ role: "system", content: context }` to the history

### `src/frontend.ts`

- Gets page elements including `#mode-standard` and `#mode-rag` toggle buttons
- Tracks `currentMode: "standard" | "rag"` (default `"standard"`)
- Includes `mode: currentMode` in every POST body
- Renders replies in the message feed; supports Enter-to-send / Shift+Enter for newline

### `src/frontend-helpers.ts`

- Keeps small UI formatting logic (`formatModelLabel`, `formatReplyText`) separate and testable

## Run the app

```bash
# Dev mode — hot reload on file changes
cd web
bun run dev

# Production
cd web
bun run start
```

Then open `http://localhost:3000`.

## Run the tests

```bash
cd web
bun test
```

## Required services

```bash
ollama serve
```

The Standard mode uses the newest `chembl-drug-chat:*` model. RAG mode uses `gemma3:1b`. Both must be present in Ollama:

```bash
ollama list | grep -E "chembl-drug-chat|gemma3"
```

## Environment variables

A `.env` file with working defaults is committed at `web/.env` — Bun loads it automatically, so no setup is needed. Edit it to point at a different Ollama host, swap the RAG model, or override the LanceDB path:

```bash
PORT=3000
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_PREFIX=chembl-drug-chat
OLLAMA_MODEL_NAME=chembl-drug-chat:1b   # fallback if /api/tags fails
RAG_MODEL_NAME=gemma3:1b                # model used in RAG mode
# LANCEDB_DIR=                          # default: auto-resolved from web/src/rag.ts
```

## UI behaviour

- Dark theme by default
- **Standard / RAG pill toggle** in the header — persists for the session
- Enter sends; Shift+Enter adds a newline
- Input textarea grows as you type
- Model label shows which Ollama tag is active and how it was selected
