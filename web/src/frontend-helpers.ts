export type ModelInfo = { model: string; source: string };
export type ChatResult = { reply?: string; error?: string; queued?: boolean; position?: number };
export type Place = { seated: boolean; position: number; error?: string };

// Request size caps, shared by the server (rejects) and the page (trims history,
// caps the textarea). Assistant turns are capped too because history comes from
// the client; a 512-token reply fits.
export const MAX_MESSAGES = 20;
export const MAX_USER_CHARS = 2_000;
export const MAX_MESSAGE_CHARS = 4_000;

export function formatReplyText(reply?: string): string {
  return reply?.trim() || "(empty response)";
}

export function renderMarkdown(text: string): string {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/```[\w]*\n?([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}
