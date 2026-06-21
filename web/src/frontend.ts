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

const containers = {
  finetuned: el<HTMLElement>("messages-finetuned"),
  rag: el<HTMLElement>("messages-rag"),
};

const histories: Record<"finetuned" | "rag", ChatMessage[]> = { finetuned: [], rag: [] };

async function loadModel() {
  try {
    const res = await fetch("/api/model");
    const data = (await res.json()) as ModelInfo;
    modelLabel.textContent = data.source;
    el<HTMLElement>("header-finetuned").textContent = `Finetuned — ${data.model}`;
    el<HTMLElement>("header-rag").textContent = `RAG — ${data.ragModel}`;
  } catch {
    modelLabel.textContent = "model unavailable";
  }
}

function addBubble(container: HTMLElement, role: ChatMessage["role"], text?: string) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  if (text) {
    role === "assistant"
      ? (bubble.innerHTML = renderMarkdown(text))
      : (bubble.textContent = text);
  } else {
    bubble.classList.add("thinking");
  }
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

async function streamInto(
  res: Response,
  bubble: HTMLElement,
  container: HTMLElement,
  history: ChatMessage[],
) {
  if (!res.ok) {
    const data = (await res.json()) as ChatResult;
    bubble.classList.remove("thinking");
    bubble.textContent = data.error || "Chat request failed.";
    return;
  }

  bubble.classList.remove("thinking");
  let accumulated = "";
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let partial = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    partial += decoder.decode(value, { stream: true });
    const lines = partial.split("\n");
    partial = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const chunk = JSON.parse(line) as { done?: boolean; message?: { content?: string } };
        if (!chunk.done && chunk.message?.content) {
          accumulated += chunk.message.content;
          bubble.textContent = accumulated;
          container.scrollTop = container.scrollHeight;
        }
      } catch {}
    }
  }

  bubble.innerHTML = renderMarkdown(formatReplyText(accumulated));
  history.push({ role: "assistant", content: accumulated });
  container.scrollTop = container.scrollHeight;
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

form.addEventListener("submit", async (event: SubmitEvent) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  for (const mode of ["finetuned", "rag"] as const) {
    histories[mode].push({ role: "user", content: text });
    addBubble(containers[mode], "user", text);
  }

  input.value = "";
  btn.disabled = true;

  const bubbles = {
    finetuned: addBubble(containers.finetuned, "assistant"),
    rag: addBubble(containers.rag, "assistant"),
  };

  const [finRes, ragRes] = await Promise.all([
    fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: histories.finetuned, mode: "finetuned" }),
    }),
    fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: histories.rag, mode: "rag" }),
    }),
  ]).catch(err => { throw err; });

  await Promise.allSettled([
    streamInto(finRes, bubbles.finetuned, containers.finetuned, histories.finetuned),
    streamInto(ragRes, bubbles.rag, containers.rag, histories.rag),
  ]);

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
