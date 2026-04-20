/// <reference lib="dom" />

import { formatModelLabel, formatReplyText, type ChatResult, type ModelInfo } from "./frontend-helpers";
import type { ChatMessage } from "./app";

// Cache the page elements once so the browser code stays simple to follow.
function getRequiredElement<T extends HTMLElement>(id: string, guard: (value: HTMLElement) => value is T) {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement) || !guard(element)) {
    throw new Error(`${id} is missing.`);
  }

  return element;
}

const form = getRequiredElement("chat-form", (value): value is HTMLFormElement => value instanceof HTMLFormElement);
const input = getRequiredElement(
  "input",
  (value): value is HTMLTextAreaElement => value instanceof HTMLTextAreaElement,
);
const btn = getRequiredElement("btn", (value): value is HTMLButtonElement => value instanceof HTMLButtonElement);
const messagesEl = getRequiredElement("messages", (value): value is HTMLElement => value instanceof HTMLElement);
const modelLabel = getRequiredElement("model-label", (value): value is HTMLElement => value instanceof HTMLElement);

const history: ChatMessage[] = [];

// Resize the prompt box so longer messages feel natural to type and review.
function resizePromptBox() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

// Ask the Bun backend which exported model is newest.
async function loadModel() {
  try {
    const response = await fetch("/api/model");
    const data = (await response.json()) as ModelInfo;
    modelLabel.textContent = formatModelLabel(data);
  } catch {
    modelLabel.textContent = "Latest model: unavailable";
  }
}

// Render one chat bubble and keep the scroll pinned to the newest message.
function addBubble(role: ChatMessage["role"], text: string) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  bubble.textContent = text;
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

// Submit the chat message to the Bun API and render the reply.
form.addEventListener("submit", async (event: SubmitEvent) => {
  event.preventDefault();

  const text = input.value.trim();
  if (!text) {
    return;
  }

  history.push({ role: "user", content: text });
  addBubble("user", text);
  input.value = "";
  btn.disabled = true;

  const replyBubble = addBubble("assistant", "Thinking...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });

    const data = (await response.json()) as ChatResult;
    if (!response.ok) {
      throw new Error(data.error || "Chat request failed.");
    }

    replyBubble.textContent = formatReplyText(data.reply);
    history.push({ role: "assistant", content: data.reply || "" });
  } catch (error) {
    replyBubble.textContent =
      error instanceof Error ? error.message : "Error: could not reach server.";
  } finally {
    btn.disabled = false;
    input.focus();
  }
});

// ChatGPT-style keyboard behavior: Enter sends, Shift+Enter inserts a newline.
input.addEventListener("keydown", (event: KeyboardEvent) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }

  event.preventDefault();
  form.requestSubmit();
});

// Keep the textarea height in sync with what the user types.
input.addEventListener("input", resizePromptBox);

// Load the model label as soon as the page opens.
resizePromptBox();
loadModel();
