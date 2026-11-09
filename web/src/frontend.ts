/// <reference lib="dom" />
import {
  formatReplyText,
  MAX_MESSAGES,
  MAX_USER_CHARS,
  renderMarkdown,
  type ChatResult,
  type ModelInfo,
  type Place,
} from "./frontend-helpers";
import type { ChatMessage } from "./app";
import { TOOL_SPECS } from "./tools";

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} missing`);
  return e as T;
}

const form = el<HTMLFormElement>("chat-form");
const input = el<HTMLTextAreaElement>("input");
const btn = el<HTMLButtonElement>("btn");
const modelLabel = el<HTMLElement>("model-label");
const messages = el<HTMLElement>("messages");
const status = el<HTMLElement>("status");
const signin = el<HTMLElement>("signin");
const signinError = el<HTMLElement>("signin-error");
const account = el<HTMLElement>("account");
const accountEmail = el<HTMLElement>("account-email");
const signout = el<HTMLButtonElement>("signout");

const history: ChatMessage[] = [];
input.maxLength = MAX_USER_CHARS;

// ── Sign in with Google ──────────────────────────────────────────────────────
// When the server has a Google client ID, the chat needs a signed-in visitor.
// Google's button hands the page an ID token; the server checks it and sets a
// session cookie. Without a client ID (local runs) there is no sign-in at all.
type GoogleId = {
  initialize(options: { client_id: string; callback: (response: { credential: string }) => void }): void;
  renderButton(parent: HTMLElement, options: Record<string, string>): void;
  disableAutoSelect(): void;
};
declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } };
  }
}

let config: { googleClientId: string | null; signInRequired: boolean } = {
  googleClientId: null,
  signInRequired: false,
};
let googleReady: Promise<GoogleId> | null = null;

function loadGoogle(clientId: string): Promise<GoogleId> {
  googleReady ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onerror = () => reject(new Error("Could not load Google sign-in."));
    script.onload = () => {
      const id = window.google!.accounts.id;
      id.initialize({ client_id: clientId, callback: ({ credential }) => void completeSignIn(credential) });
      resolve(id);
    };
    document.head.appendChild(script);
  });
  return googleReady;
}

async function completeSignIn(credential: string) {
  signinError.hidden = true;
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ credential }),
  });
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as ChatResult;
    signinError.textContent = data.error ?? "Sign-in failed. Try again.";
    signinError.hidden = false;
    return;
  }
  signin.hidden = true;
  heartbeat();
}

async function showSignIn() {
  clearTimeout(pingTimer);
  seated = false;
  status.hidden = true;
  account.hidden = true;
  input.disabled = true;
  btn.disabled = true;
  signin.hidden = false;
  try {
    const id = await loadGoogle(config.googleClientId!);
    const target = el<HTMLElement>("google-button");
    target.replaceChildren();
    id.renderButton(target, { theme: "filled_black", size: "large", text: "signin_with", shape: "pill" });
  } catch (error) {
    signinError.textContent = error instanceof Error ? error.message : "Could not load Google sign-in.";
    signinError.hidden = false;
  }
}

signout.addEventListener("click", async () => {
  await fetch("/api/logout", { method: "POST" });
  window.google?.accounts.id.disableAutoSelect();
  showSignIn();
});

// ── Place in the queue ───────────────────────────────────────────────────────
// One person uses the chat at a time; everyone else sees their number in line.
// The page pings to keep its place: every few seconds while waiting (so the
// number moves), every 30 s once in. A tab in the chat that is hidden stops
// pinging, so its seat passes on after ~2 minutes; a waiting tab keeps its place
// even when hidden. Closing the tab gives up the place at once.
const QUEUED_PING_MS = 5_000;
const SEATED_PING_MS = 30_000;
const baseTitle = document.title;
let seated = false;
let sending = false;
let pingTimer: ReturnType<typeof setTimeout> | undefined;

function showPlace(place: Place & { email?: string }) {
  if (config.signInRequired && place.email) {
    accountEmail.textContent = place.email;
    account.hidden = false;
  }
  const wasWaiting = !seated && !status.hidden && status.dataset.queued === "1";
  seated = place.seated;
  status.hidden = place.seated;
  status.dataset.queued = place.seated ? "" : "1";
  if (!place.seated) {
    status.textContent =
      place.error ??
      `You're #${place.position} in the queue. The chat opens when it's your turn; keep this tab open.`;
  }
  // Flag the tab if the turn arrived while the visitor was looking elsewhere.
  if (place.seated && wasWaiting && document.visibilityState !== "visible") {
    document.title = `Your turn · ${baseTitle}`;
  }
  input.disabled = !place.seated;
  btn.disabled = !place.seated || sending;
}

function showProblem(message: string) {
  seated = false;
  status.hidden = false;
  status.dataset.queued = "";
  status.textContent = `${message} Retrying automatically…`;
  input.disabled = true;
  btn.disabled = true;
}

async function heartbeat() {
  clearTimeout(pingTimer);
  if (!(seated && document.visibilityState !== "visible")) {
    try {
      const res = await fetch("/api/session", { method: "POST" });
      const data = (await res.json().catch(() => ({}))) as Place;
      if (res.status === 401 && config.signInRequired) return void showSignIn();
      if (res.ok) showPlace(data);
      else showProblem(data.error ?? "The demo is unavailable.");
    } catch {
      // Network blip: keep the current state and try again on the next ping.
    }
  }
  pingTimer = setTimeout(heartbeat, seated ? SEATED_PING_MS : QUEUED_PING_MS);
}

// In-app manual, built from TOOL_SPECS so it cannot drift from the tools that
// are actually wired up. Each example fills the input, so it is usable as well
// as readable.
function renderManual() {
  const body = el<HTMLElement>("manual-body");

  const intro = document.createElement("p");
  intro.className = "manual__intro";
  intro.textContent =
    "Answers come from the ChEMBL and TWOSIDES databases through tools. " +
    "When a tool runs you'll see a dashed bubble with the call and what it returned. " +
    "Structure requests always run a tool; the rest depend on the model choosing to call one.";
  body.appendChild(intro);

  for (const spec of TOOL_SPECS) {
    const row = document.createElement("div");
    row.className = "manual__row";

    const example = document.createElement("button");
    example.type = "button";
    example.className = "manual__ask";
    example.textContent = spec.ask;
    example.addEventListener("click", () => {
      input.value = spec.ask;
      resizeInput();
      input.focus();
    });

    const detail = document.createElement("p");
    detail.className = "manual__detail";
    detail.textContent = spec.help ?? spec.description;

    const name = document.createElement("code");
    name.className = "manual__name";
    name.textContent = spec.name;

    row.append(example, detail, name);
    body.appendChild(row);
  }
}

async function loadModel() {
  try {
    const res = await fetch("/api/model");
    const data = (await res.json()) as ModelInfo;
    modelLabel.textContent = `${data.model} · ${data.source}`;
  } catch {
    modelLabel.textContent = "model unavailable";
  }
}

function addBubble(role: "user" | "assistant" | "tool", text?: string) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  if (text) {
    role === "assistant"
      ? (bubble.innerHTML = renderMarkdown(text))
      : (bubble.textContent = text);
  } else {
    bubble.classList.add("thinking");
  }
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

// A tool call and its result, shown as its own bubble so the lookup is visible.
function addToolBubble(call: { tool: string; args: unknown }) {
  const bubble = addBubble("tool", `${call.tool}(${JSON.stringify(call.args)})`);
  bubble.classList.add("thinking");
  return bubble;
}

function fillToolResult(bubble: HTMLElement, payload: { error?: string; result?: unknown }) {
  bubble.classList.remove("thinking");
  const image =
    typeof payload.result === "object" && payload.result !== null
      ? (payload.result as { image?: string }).image
      : undefined;
  const summary = payload.error ?? (image ? "" : JSON.stringify(payload.result));
  if (summary) {
    const pre = document.createElement("pre");
    pre.textContent = summary.length > 600 ? `${summary.slice(0, 600)}…` : summary;
    bubble.appendChild(pre);
  }
  if (image) {
    const img = document.createElement("img");
    img.src = image;
    img.alt = "Rendered molecule";
    bubble.appendChild(img);
  }
  messages.scrollTop = messages.scrollHeight;
}

// Read the server's NDJSON event stream: tool calls, tool results, final answer.
async function readEvents(res: Response, bubble: HTMLElement) {
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as ChatResult;
    bubble.classList.remove("thinking");
    bubble.textContent = data.error || "Chat request failed.";
    if (data.queued) showPlace({ seated: false, position: data.position ?? 1 });
    if (res.status === 401 && config.signInRequired) showSignIn();
    return;
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let partial = "";
  let toolBubble: HTMLElement | null = null;
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    partial += decoder.decode(value, { stream: true });
    const lines = partial.split("\n");
    partial = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      let event: {
        waiting?: boolean;
        tool?: { tool: string; args: unknown };
        toolResult?: { tool: string; error?: string; result?: unknown };
        message?: { content?: string };
        error?: string;
      };
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }

      if (event.waiting) {
        bubble.textContent = "Waiting for your turn… someone else's question is being answered.";
      } else if (event.tool) {
        toolBubble = addToolBubble(event.tool);
      } else if (event.toolResult && toolBubble) {
        fillToolResult(toolBubble, event.toolResult);
        toolBubble = null;
      } else if (event.message?.content !== undefined) {
        // The answer bubble is created before the tool calls are known, so move
        // it below them the moment real text starts arriving.
        if (!answer) messages.appendChild(bubble);
        // Deltas: plain text while streaming, markdown once the turn is done.
        answer += event.message.content;
        bubble.classList.remove("thinking");
        bubble.textContent = answer;
      } else if (event.error) {
        bubble.classList.remove("thinking");
        bubble.textContent = event.error;
      }
      messages.scrollTop = messages.scrollHeight;
    }
  }

  bubble.classList.remove("thinking");
  if (answer) {
    bubble.innerHTML = renderMarkdown(formatReplyText(answer));
    history.push({ role: "assistant", content: answer });
    messages.scrollTop = messages.scrollHeight;
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

form.addEventListener("submit", async (event: SubmitEvent) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  history.push({ role: "user", content: text });
  addBubble("user", text);
  input.value = "";
  resizeInput();
  sending = true;
  btn.disabled = true;

  const bubble = addBubble("assistant");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      // Only the latest turns: the server caps a conversation's length.
      body: JSON.stringify({ messages: history.slice(-MAX_MESSAGES) }),
    });
    await readEvents(res, bubble);
  } catch (error) {
    bubble.classList.remove("thinking");
    bubble.textContent = error instanceof Error ? error.message : "Chat request failed.";
  }

  sending = false;
  btn.disabled = !seated;
  input.focus();
});

input.addEventListener("keydown", (event: KeyboardEvent) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  form.requestSubmit();
});

input.addEventListener("input", resizeInput);
resizeInput();
renderManual();
loadModel();
showProblem("Joining the demo…");
status.textContent = "Joining the demo…";
fetch("/api/config")
  .then((res) => res.json())
  .then((data) => (config = data))
  .catch(() => {})
  .finally(heartbeat);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") document.title = baseTitle;
  heartbeat();
});
window.addEventListener("pageshow", heartbeat);
window.addEventListener("pagehide", () => navigator.sendBeacon("/api/session/leave"));
