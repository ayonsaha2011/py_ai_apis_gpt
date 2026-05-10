export function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function errorMessage(value: unknown): string {
  if (value instanceof Error) return value.message;
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    const nested = object.error && typeof object.error === "object" ? (object.error as Record<string, unknown>) : null;
    return String(nested?.message || object.detail || object.message || JSON.stringify(value));
  }
  return "Request failed";
}
