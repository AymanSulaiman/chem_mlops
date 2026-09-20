/// <reference lib="dom" />
import { formatReplyText, renderMarkdown, type ChatResult, type ModelInfo } from "./frontend-helpers";
import type { ChatMessage } from "./app";

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

const history: ChatMessage[] = [];

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
    const data = (await res.json()) as ChatResult;
    bubble.classList.remove("thinking");
    bubble.textContent = data.error || "Chat request failed.";
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

      if (event.tool) {
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
  btn.disabled = true;

  const bubble = addBubble("assistant");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    await readEvents(res, bubble);
  } catch (error) {
    bubble.classList.remove("thinking");
    bubble.textContent = error instanceof Error ? error.message : "Chat request failed.";
  }

  btn.disabled = false;
  input.focus();
});

input.addEventListener("keydown", (event: KeyboardEvent) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  form.requestSubmit();
});

input.addEventListener("input", resizeInput);
resizeInput();
loadModel();
