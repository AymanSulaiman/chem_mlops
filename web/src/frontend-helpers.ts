export type ModelInfo = {
  model: string;
  source: string;
};

export type ChatResult = {
  reply?: string;
  error?: string;
};

export function formatModelLabel(data: ModelInfo) {
  return `Latest model: ${data.model} (${data.source})`;
}

export function formatReplyText(reply?: string) {
  return reply && reply.trim().length > 0 ? reply : "(empty response)";
}
