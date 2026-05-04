# Chat App Build Plan

A vanilla TypeScript + Bun + DOM frontend for `chem_mlops`, with a model-mode dropdown that toggles between the fine-tuned Gemma model (via Ollama) and a RAG pipeline (LanceDB + base model).

No frameworks. No virtual DOM. No build pipeline beyond Bun's built-in TypeScript handling. Just code I wrote and can read end-to-end.

**This plan assumes I don't know TypeScript yet.** Step 0 covers what I need before writing any chat-app code. Each subsequent step explains the TS concepts as they appear, so I'm learning the language by building, not by reading a textbook.

---

## Goals

- Working chat UI in the browser, served from the existing `web/` Bun app
- Streaming token-by-token responses (SSE)
- A dropdown that switches between two modes:
  - **Fine-tune** — proxies to local Ollama running `chembl-drug-chat:1b`
  - **RAG** — retrieves from LanceDB, augments the prompt, calls a base model
- Markdown rendering for assistant messages
- Collapsible "retrieved chunks" panel in RAG mode

## Non-goals (explicitly out of scope)

- Auth / user accounts
- Conversation persistence across reloads
- Multi-conversation history sidebar
- Mobile-perfect responsive design
- Deployment to anything other than localhost (for now)
- Tests beyond a couple of basic sanity tests
- Becoming a TypeScript expert. Goal is "competent enough to build this app and read TS code in interviews," not "advanced generics."

If I find myself adding any of these mid-build, stop and ask: *is this needed for the demo, or am I yak-shaving?*

---

## Time budget (honest)

Roughly **5-6 weekend cafe sessions** of 4-6 hours each. The first two sessions are heavy on learning, the rest are on building. If it's taking 8+ sessions, something is wrong — pause and reassess.

Per-step estimates below are realistic, not aspirational. They include "things will break and I'll Google" time, plus "I need to look up what this syntax means" time.

---

## Mental model — TypeScript in one paragraph

TypeScript is JavaScript with type annotations. The browser doesn't run TypeScript; a tool (Bun, in our case) strips out the types and produces plain JavaScript. The types exist only at "compile time" to catch mistakes — like calling a function with the wrong kind of argument — before the code runs. **Functionally, every TypeScript program is also a JavaScript program with extra metadata.** When in doubt, mentally delete the type annotations and ask "would this JavaScript work?"

The types I'll see most:

- `string`, `number`, `boolean` — the obvious ones
- `string[]` — array of strings
- `{ name: string; age: number }` — an object with specific fields
- `string | number` — a value that could be either (called a "union")
- `(x: number) => string` — a function that takes a number and returns a string
- `Promise<string>` — a value that will eventually be a string (async result)

That's 80% of what I need.

---

## Mental model — vanilla DOM in one paragraph

Three concepts:

1. **The DOM is a tree of objects I poke at.** HTML becomes a JS object called `document`. Find elements with `document.querySelector`, create them with `document.createElement`, attach with `parent.appendChild`, listen with `addEventListener`. That's 90% of frontend.

2. **State is just plain variables.** When state changes, re-render the affected DOM nodes by clearing and recreating them. No hooks, no effects, no reactivity magic.

3. **Streaming is reading from a response one chunk at a time.** `fetch()` returns a `Response`. `response.body` is a `ReadableStream`. Read it in a loop, decode each chunk, append to the UI. ~15 lines of utility code.

If any step starts feeling "magic," I've forgotten one of the six bullets above (3 TS, 3 DOM). Re-read.

---

## Project structure (target end state)

```
web/
├── server.ts                 # existing Bun server, extended with /api/chat
├── public/
│   ├── index.html            # page shell
│   ├── styles.css            # styling (or Tailwind via CDN)
│   └── js/                   # compiled JS output (gitignored)
├── src/
│   ├── main.ts               # entry point — wires everything together
│   ├── chat.ts               # chat state + DOM rendering
│   ├── stream.ts             # SSE/streaming utility
│   ├── api.ts                # fetch wrappers for /api/chat
│   └── types.ts              # shared types (Message, Mode, etc.)
├── tsconfig.json
└── package.json
```

Don't try to create all these files upfront. They get added as needed.

---

## The single most important rule

**Each step must work end-to-end before starting the next one.**

Do not skip ahead. Do not combine steps. Each step has a checkpoint. Hit the checkpoint, commit, move on. The vanilla approach feels hard only when too many layers are wrong at once.

After each checkpoint: `git commit` with a clear message. Future me will thank present me.

---

## Step 0 — TypeScript familiarization

**Goal:** Be able to *read* TypeScript without panicking. I don't need to be fluent — I need to know what `:`, `?`, `|`, `<>`, and `as` mean when I see them. This is the only step where I'm allowed to learn without building.

**Time:** 4-6 hours total (one cafe session, or split over a few evenings)

**Tasks:**

- **Read the official TypeScript handbook's intro page.** It's short and assumes I already know JS basics. https://www.typescriptlang.org/docs/handbook/typescript-from-scratch.html
- **Watch one short YouTube intro** if I prefer video to reading — pick anything ~30-60 minutes that covers the basics. Don't watch a 6-hour bootcamp; we don't need that.
- **Play in the official TypeScript Playground.** This is the in-browser one at https://www.typescriptlang.org/play. Hover over things, see the type errors, get a feel.
- **Do not skip this step thinking "I'll learn as I go."** Trying to learn TS *and* DOM *and* streaming *and* the project architecture simultaneously is the failure mode that kills these builds. One pre-step of dedicated TS time pays for itself ten times over.

**Concepts I should be comfortable with by the end:**
- Type annotations on variables: `let x: number = 5;`
- Type annotations on function parameters and return values: `function greet(name: string): string { return "hi " + name; }`
- Interfaces and type aliases: `interface User { name: string; age: number; }` — these define the shape of objects
- Optional fields: `interface User { name: string; age?: number; }` — `age` may or may not be present
- Union types: `type Mode = "finetune" | "rag";` — a value can be one of these specific strings
- Arrays: `string[]` and `number[]`
- Async/await with `Promise<T>` (probably already familiar from Python's `async def`)
- The `as` keyword for type assertions (use sparingly, but I will see it)
- Generics in shallow form: I should know that `Array<string>` is the same as `string[]`, and not freak out when I see `<T>` somewhere

**Concepts I can punt on:**
- Advanced generics, conditional types, mapped types
- Decorators
- Namespaces
- Most utility types beyond `Partial<T>` and `Record<K, V>`
- The differences between `interface` and `type` (use either; pick by feel)

**Checkpoint:** I can read a 50-line TypeScript file and not be lost. I don't need to write it from scratch — I need to read it without panic. If I open someone else's `.ts` file and feel lost, I haven't hit the checkpoint yet.

**Commit:** No commit. This is reading, not building.

**Anti-perfectionism rule:** Once the checkpoint is roughly true, *move on*. I will learn way more by Step 5 than I ever will from one more tutorial. Tutorials hit diminishing returns fast.

---

## Step 1 — Static HTML mockup

**Goal:** A non-interactive HTML page that *looks* like the chat app. Zero TypeScript involved.

**Time:** 60-90 min

**Tasks:**
- Create `web/public/index.html`
- Page structure: header, scrollable message area, input bar (textarea + send button), mode dropdown in the corner
- Add 2-3 hardcoded message bubbles (one user, one assistant) so styling has something to bite on
- Style with plain CSS in `web/public/styles.css`, OR use Tailwind via CDN: `<script src="https://cdn.tailwindcss.com"></script>` (this is fine for prototypes — no build step needed, ignore Tailwind's own warnings about CDN)
- Open the file directly in a browser (`file://` is fine here) — no server needed yet

**Why no TS yet:** I want the page to look right before I write a single line of code. This separates "what does it look like" from "what does it do." It also means if styling breaks something later, I know exactly when it broke.

**Checkpoint:** I can open the page in a browser and it looks roughly like a chat app. No interactivity expected.

**Anti-yak-shave rule:** ugly is fine. Don't spend 3 hours on the perfect message bubble. The fast version is good enough.

**Commit:** `feat(web): static chat UI mockup`

---

## Step 2 — Serve the page from the Bun server

**Goal:** Move from `file://` to `http://localhost:3000` (or whatever port `server.ts` uses). Still no client-side TS.

**Time:** 30-45 min

**Tasks:**
- Extend `web/server.ts` to serve files from `web/public/` for any GET request that isn't an API route
- Verify `index.html`, `styles.css`, and any other static assets load correctly
- Handle the basics: 404 for missing files, correct MIME types (Bun's `Bun.file()` handles MIME automatically)

**TS knowledge introduced:** None new. `server.ts` is already TS but I'm just extending an existing pattern.

**Checkpoint:** Run `bun run server.ts`, open `http://localhost:3000`, see the same page from Step 1.

**Commit:** `feat(web): serve static chat UI from Bun server`

---

## Step 3 — TypeScript setup + first interactive line

**Goal:** Wire up TypeScript for the *client*, write one `.ts` file, prove the toolchain works end-to-end.

**Time:** 60-90 min (most of it is config + understanding what's happening, not writing code)

**Tasks:**
- Create `web/tsconfig.json`. Minimal config:
  ```json
  {
    "compilerOptions": {
      "target": "ES2022",
      "module": "ESNext",
      "moduleResolution": "Bundler",
      "strict": true,
      "lib": ["ES2022", "DOM", "DOM.Iterable"],
      "types": ["bun-types"],
      "outDir": "public/js"
    },
    "include": ["src/**/*"]
  }
  ```
- **What this config means** (read this once, then move on):
  - `target` — what version of JavaScript to compile *to*. ES2022 is widely supported.
  - `module` / `moduleResolution` — how `import` statements work.
  - `strict` — turn on all the type-checking strictness. Annoying at first, life-saving later. Keep it on.
  - `lib` — what built-in types are available. `DOM` is what gives us `document`, `window`, etc.
  - `types` — extra type definitions to include. `bun-types` gives us TypeScript awareness of Bun's runtime.
  - `outDir` — where compiled `.js` files go.
- Create `web/src/main.ts` with a single line: `console.log("hello");`
- Add a build script to `web/package.json`:
  ```json
  "scripts": {
    "build:client": "bun build ./src/main.ts --outdir ./public/js"
  }
  ```
- Run `bun run build:client`, verify `web/public/js/main.js` is generated. Open it — notice it's just plain JavaScript. The TS compiled away.
- In `index.html`, add `<script type="module" src="/js/main.js"></script>` before `</body>`
- Reload page, open DevTools console, see `hello`
- Add `web/public/js/` to `.gitignore` — compiled output shouldn't be committed

**TS knowledge introduced:** Compilation pipeline. *I write `.ts`, the build step produces `.js`, the browser runs `.js`.* This is the single most important mental model for TS in this project.

**Optional but recommended:** Add a watch script: `"dev:client": "bun build ./src/main.ts --outdir ./public/js --watch"`. Now `main.js` rebuilds on save.

**Checkpoint:** I can edit `main.ts`, run the build (or have watch running), reload, and see my console output. The pipeline is real. I understand what each line of `tsconfig.json` does *roughly*.

**Commit:** `chore(web): typescript build pipeline`

---

## Step 4 — Send button reads input and logs it

**Goal:** First DOM interaction. No state, no rendering, just an event listener. First real TS code.

**Time:** 60-90 min

**Tasks:**
- Add IDs to the relevant elements in `index.html` if not already present (e.g., `id="message-input"`, `id="send-button"`)
- In `main.ts`: query the textarea and send button by ID
- Add a `click` listener to the send button that reads textarea value and `console.log`s it
- Also handle Enter (without Shift) to submit (Shift+Enter should add a newline)
- Clear the textarea after submit

**TS knowledge introduced:** The `as` keyword and DOM type narrowing.

When I write:
```ts
const input = document.getElementById("message-input");
```
TypeScript types `input` as `HTMLElement | null`. The `| null` is because the element might not exist. The `HTMLElement` is generic — TS doesn't know it's specifically a `<textarea>`.

To tell TS "trust me, this is a textarea":
```ts
const input = document.getElementById("message-input") as HTMLTextAreaElement;
```

Now I can call `input.value` without TS complaining. **But**: `as` is an unchecked assertion. If I'm wrong, the runtime breaks. Use it only when I'm sure.

I'll see `as HTMLTextAreaElement`, `as HTMLButtonElement`, `as HTMLSelectElement` etc. throughout this project. That's normal.

The `?.` operator is also useful here:
```ts
const input = document.getElementById("message-input");
input?.addEventListener(...);  // only adds the listener if input is not null
```

**Checkpoint:** Type into the box, click send (or hit Enter), see the message in the console. Textarea clears. Shift+Enter adds a newline instead of sending.

**Commit:** `feat(web): wire send button to input`

---

## Step 5 — Render a user message bubble in the DOM

**Goal:** Replace `console.log` with actual DOM rendering. Introduce the `Message` type. Start splitting code into multiple files.

**Time:** 90-120 min

**Tasks:**
- Create `web/src/types.ts`:
  ```ts
  export type Mode = "finetune" | "rag";
  export type Role = "user" | "assistant";

  export interface Message {
    role: Role;
    content: string;
    chunks?: string[]; // for RAG mode; optional
  }
  ```
- Create `web/src/chat.ts` with:
  - Module-level state: `let messages: Message[] = [];`
  - A function `renderMessage(msg: Message)` that creates a DOM node and appends to the message container
  - A function `addUserMessage(text: string)` that pushes to `messages` and calls `renderMessage`
- In `main.ts`, replace `console.log` with a call to `addUserMessage(text)`
- Auto-scroll the message area to the bottom when a new message appears

**TS knowledge introduced:**

- **Modules and `import` / `export`.** TS uses ES modules. To use something from another file, the source file must `export` it and the consumer must `import` it. Example:
  ```ts
  // types.ts
  export interface Message { ... }

  // chat.ts
  import type { Message } from "./types";
  ```
  The `import type` is a TS-specific thing — it makes clear we only want the type, not runtime code. Use it for types whenever possible; it's slightly more efficient.
- **Interfaces and type aliases.** `interface Message { ... }` defines the shape of a message object. `type Mode = "finetune" | "rag"` is a string literal union — `Mode` can only be exactly those two strings.
- **Optional properties.** `chunks?: string[]` means `chunks` may or may not exist on the object. When I read it, TS will warn me to handle the undefined case.

**Checkpoint:** Type a message, hit send, see a user bubble appear. State is real (it's in the array), DOM is real. Multiple files now, importing from each other.

**Commit:** `feat(web): render user messages in DOM`

---

## Step 6 — Backend `/api/chat` returning a fake non-streamed response

**Goal:** Frontend talks to backend over HTTP. No models yet. No streaming yet. Just round-trip.

**Time:** 90-120 min

**Tasks:**
- Create `web/src/api.ts` with a function:
  ```ts
  import type { Message, Mode } from "./types";

  export async function sendMessage(messages: Message[], mode: Mode): Promise<string> {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, mode }),
    });
    return await response.text();
  }
  ```
- Extend `server.ts` to handle `POST /api/chat`:
  - Parse JSON body
  - Return a hardcoded fake assistant response, e.g. `"Got: " + lastUserMessage`
- In `chat.ts`, after rendering the user message, call `sendMessage`, then push and render the assistant message

**TS knowledge introduced:**

- **`async`/`await` with `Promise<T>`.** Same model as Python's `async def` and `await`. `Promise<string>` means "an eventual string." `await` unwraps it.
- **Reading typed JSON from `fetch`.** `response.json()` returns `Promise<any>` by default — TS doesn't know what's in it. For now, that's fine. In a more careful version we'd type-narrow it.
- **The relationship between client and server types.** Both ends share `Message` and `Mode` from `types.ts`. This is a power move of TS: the same interface defines what the frontend sends *and* what the backend expects, and they can't drift apart silently. Notice this when I wire it up.

**Checkpoint:** Type a message → user bubble appears → ~half a second later → assistant bubble appears with the fake echo response.

**Commit:** `feat(web): wire chat to backend with fake response`

---

## Step 7 — Streaming responses end-to-end (still fake)

**Goal:** Replace the one-shot response with a streamed character-by-character response. This is the moment the UX starts to feel real, and the conceptually heaviest step.

**Time:** 3-4 hours (the conceptual leap, not the LOC)

**Tasks:**
- In `server.ts`, change `/api/chat` to return a Server-Sent Events stream:
  - `Content-Type: text/event-stream`
  - Stream the message `"Hello, this is a fake response"` one character at a time with a 30ms delay between chunks
  - Format each chunk as `data: <char>\n\n`
- Create `web/src/stream.ts` with a utility:
  ```ts
  export async function* streamSSE(response: Response): AsyncGenerator<string> {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          yield line.slice(6);
        }
      }
    }
  }
  ```
- In `chat.ts`: when sending a message, create an empty assistant bubble first, then for each chunk from the stream, append to its content and re-render that one bubble
- **Don't re-render the entire message list on every chunk.** Only the in-progress bubble.

**TS knowledge introduced:** This step has the most new TS in the whole build. Don't rush it.

- **Async generators.** `async function*` is a function that produces values *over time*, asynchronously. The caller iterates with `for await (const chunk of streamSSE(response))`. If I've never used these in any language, expect to spend an hour just understanding the concept. They're worth knowing.
- **The non-null assertion `!`.** `response.body!` tells TS "I know this might be null in general, but here, trust me, it's not." Use sparingly.
- **Destructuring with type inference.** `const { done, value } = await reader.read();` — TS knows the shape of what `read()` returns. I can hover in my editor to see the types.
- **`AsyncGenerator<string>`.** The generic `<string>` says "this generator yields strings." Without it, TS would default to `unknown`.

**Why this step is the hardest:** I'm dealing with async generators, partial DOM updates, and SSE format all at once. If stuck:
1. First, console.log every yielded chunk and don't touch the DOM yet. Make sure streaming works at all.
2. Then, render to a single `<div>` separate from the chat UI. Make sure DOM rendering of streamed content works.
3. *Then* wire it into the chat UI.

That layering is the difference between "this took 3 hours" and "this took a full weekend in frustration."

**Checkpoint:** Send a message → user bubble appears → assistant bubble starts empty → text streams in character by character.

**Commit:** `feat(web): streaming responses via SSE`

---

## Step 8 — Wire to Ollama (fine-tune mode is real)

**Goal:** Replace the fake stream with actual Ollama responses from `chembl-drug-chat:1b`.

**Time:** 90-120 min

**Tasks:**
- Confirm Ollama is running locally and the model responds: `ollama run chembl-drug-chat:1b`
- In `server.ts`, change `/api/chat` to:
  - Accept `{ messages, mode }` in the body
  - For now, ignore `mode` (next step handles it) and always call Ollama
  - POST to `http://localhost:11434/api/chat` with `{ model: "chembl-drug-chat:1b", messages, stream: true }`
  - Ollama returns NDJSON (one JSON object per line, *not* SSE) — read its body, parse each line, and re-emit as SSE to the frontend
- Test with real drug interaction questions: "What does Aspirin target?"

**TS knowledge introduced:** None new. This is mostly applying what I already know to a real API.

**Checkpoint:** Real model answers stream into the chat UI. Sometimes the model is dumb because it's a 1B param model — that's fine, that's a known property of the model, not a bug in the app.

**Commit:** `feat(web): proxy to Ollama for fine-tune mode`

---

## Step 9 — Mode dropdown + RAG path

**Goal:** The dropdown actually does something. RAG mode does retrieval, builds a prompt, calls a model.

**Time:** 4-6 hours (this is real work — depends on RAG implementation state)

**Prereq:** The RAG version of chem_mlops needs to be at least scaffolded — LanceDB indexed, retrieval function callable from Python.

**Tasks:**
- Add the dropdown to `index.html` with two options: `finetune` and `rag`
- In `chat.ts`, track current mode as a module variable, listen for dropdown changes
- Pass `mode` in the request body to `/api/chat`
- In `server.ts`:
  - If `mode === "finetune"`: existing Ollama path
  - If `mode === "rag"`: call into the Python RAG layer. Easiest is probably a small FastAPI/Flask sidecar exposing `/rag` that the Bun server proxies to.
  - Document the choice in `DECISIONS.md`.

**TS knowledge introduced:** None new.

**Checkpoint:** Toggle the dropdown, ask the same question in both modes, get visibly different responses.

**Commit:** `feat(web): RAG mode via dropdown toggle`

---

## Step 10 — Show retrieved chunks in RAG mode

**Goal:** The non-obvious upgrade that makes the project genuinely impressive. Surface what was retrieved so the system is inspectable.

**Time:** 90 min

**Tasks:**
- Have the RAG endpoint return retrieved chunks as part of the stream (e.g., a special `data: {"type":"chunks", ...}` event before the answer tokens)
- Update `streamSSE` (or make a typed variant) to handle different event types
- When chunks arrive, attach them to the in-progress assistant message
- Render a collapsible `<details>` element under the assistant bubble showing the retrieved chunks, with their similarity scores if available
- Only show this for messages where `chunks` is set

**TS knowledge introduced:** *Discriminated unions* — a useful pattern.

```ts
type StreamEvent =
  | { type: "chunks"; chunks: string[] }
  | { type: "token"; text: string }
  | { type: "done" };
```

When I check `event.type === "chunks"`, TS automatically narrows `event` to that variant and lets me access `event.chunks`. This is one of the parts of TS that genuinely makes code safer.

**Checkpoint:** In RAG mode, every assistant message has an expandable "Retrieved context" section. In fine-tune mode, no such section.

**Commit:** `feat(web): surface retrieved chunks in RAG mode`

---

## Step 11 — Markdown rendering

**Goal:** Assistant responses render as markdown (lists, code blocks, bold, etc.) instead of plain text.

**Time:** 30-45 min

**Tasks:**
- Add `marked` (single import, no framework): `bun add marked`
- In `chat.ts`, when rendering an assistant message, set `innerHTML` to `marked.parse(content)` instead of `textContent = content`
- Sanitization: for now, since the only source of HTML is my own model, this is fine. Note in `DECISIONS.md` that production would need DOMPurify.
- Add basic CSS for code blocks and lists

**TS knowledge introduced:** Adding a third-party library. `bun add marked` installs the package and its TS type definitions in one step. Nothing dramatic.

**Checkpoint:** Ask a question that produces a list or code block, see it rendered properly.

**Commit:** `feat(web): markdown rendering for assistant messages`

---

## Step 12 — Polish + ship

**Goal:** Make it look like something I'd be willing to demo to a hiring manager.

**Time:** 2-4 hours

**Tasks:**
- Loading indicator while waiting for the first token
- Auto-scroll to bottom (re-check after streaming changes)
- Keyboard shortcut: Cmd/Ctrl+Enter to send
- Error states: what happens if the backend is down, model is not loaded, RAG sidecar is down
- Empty state: nice landing message when there are no messages yet
- A few example prompts as clickable suggestions on the empty state ("What does Aspirin target?", "How is Warfarin metabolized?")
- Basic visual polish: spacing, font, max-width, dark mode if it's quick
- Update `web/README.md` with how to run the app end-to-end
- Add a screenshot or short GIF to the main repo `README.md`

**Checkpoint:** I'd be willing to share the URL/demo with a friend without apologizing for it.

**Commit:** `feat(web): polish and ship`

---

## Stretch goals (only after Step 12)

These are *not* part of the core build. Only do them if Steps 1-12 are shipped and committed.

- **Deploy somewhere public.** Modal, Fly.io, Railway, or a small cloud VM. The challenge: the model has to live somewhere. A quantized GGUF on a small GPU instance is the realistic option.
- **A hosted version of the model with a public URL** so a hiring manager can try it without cloning.
- **Side-by-side mode:** a UI option to send the same query to both modes simultaneously, render two columns. This is interview gold but only after the basic version works.
- **Eval harness UI:** a small page where I can upload a JSON of question/answer pairs and run them through both modes, get a scorecard.

---

## Failure modes to watch for

1. **Yak-shaving the styling.** "Just one more refinement" before Step 5 is done. Ugly but functional > beautiful but broken.
2. **Combining steps.** "I'll do streaming and the dropdown together since they're related." No. One at a time. Always.
3. **Adding scope mid-build.** "Wouldn't it be cool if the chat had voice input?" Write it down in an `IDEAS.md`, do not implement.
4. **Letting the agent drive without reading the code.** If I can't explain why a function is structured the way it is, stop and read it. Familiarity is the goal, not just function.
5. **Optimizing the wrong layer.** Don't tune Ollama parameters or LanceDB indexing while the basic UI is broken. End-to-end first, optimize second.
6. **Skipping commits.** Each checkpoint = a commit. If I burn a session and have nothing committed, I've drifted.
7. **Trying to learn advanced TS while building.** If I'm hitting a wall on a TS concept, write the code in plain JS in my head first ("if this had no types, what would it look like?"), then add types back. The types are documentation, not the substance.
8. **Skipping Step 0.** The biggest risk in this revised plan. Trying to learn TS while debugging streaming responses is the failure mode. Do Step 0 properly. It's not procrastination; it's the cheapest hour I'll spend.

---

## Definition of done (for this whole plan)

I can:
1. Run `bun run server.ts` and the Python RAG sidecar, open localhost in a browser
2. Have a streaming conversation with both the fine-tuned and RAG modes via the dropdown
3. Inspect retrieved chunks in RAG mode
4. See markdown-rendered responses
5. Demo it to a friend without apologizing
6. Push the whole thing to the chem_mlops repo with a clear `web/README.md`
7. Read every file in `web/src/` and explain what it does and why

Everything else is bonus.
