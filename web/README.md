# Bun Web App

This folder contains the browser app that chats with the latest Ollama model created by this project.

## What the app does

1. Serves a simple chat page
2. Loads the latest `chembl-drug-chat` model from Ollama
3. Sends the user’s chat history to the Bun backend
4. Returns the model reply to the browser

## Folder structure

```text
web/
├── public/
│   ├── index.html
│   └── style.css
├── src/
│   ├── app.ts
│   ├── frontend.ts
│   └── frontend-helpers.ts
├── test/
│   ├── app.test.ts
│   ├── chat.test.ts
│   ├── frontend.test.ts
│   └── model.test.ts
├── package.json
└── server.ts
```

## How it works

### `server.ts`

- builds `src/frontend.ts` into `public/frontend.js`
- starts Bun on port `3000`
- hands requests to the app handler from `src/app.ts`

### `src/app.ts`

- normalizes chat messages
- finds the latest Ollama model
- forwards chat requests to Ollama
- serves `index.html`, `style.css`, and `frontend.js`

### `src/frontend.ts`

- reads the page elements
- loads the current model label
- sends chat messages to `/api/chat`
- renders replies in the message feed
- supports Enter-to-send and Shift+Enter for newlines

### `src/frontend-helpers.ts`

- keeps small UI formatting logic separate and testable

## Run the app

From the repo root:

```bash
cd web
bun run server.ts
```

Then open:

```text
http://localhost:3000
```

## Run the tests

```bash
cd web
bun test
```

Or:

```bash
cd web
bun run test
```

These use Bun’s built-in test runner and `bun:test`.

## Required services

The app talks to Ollama locally.

```bash
ollama serve
```

If you have already exported a model from this project, the app will use the newest `chembl-drug-chat*` model it finds.

## Environment variables

```bash
PORT=3000
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_PREFIX=chembl-drug-chat
OLLAMA_MODEL_NAME=chembl-drug-chat:gemma4-e2b
```

## How to think about it

Browser:

- user types a question
- browser sends JSON to the Bun backend

Backend:

- validates the message list
- picks the latest model
- calls Ollama
- returns JSON

Ollama:

- generates the answer

## UI behavior

The chat UI is designed to feel closer to ChatGPT:

- dark theme by default
- Enter sends the message
- Shift+Enter adds a newline
- the input grows as you type
