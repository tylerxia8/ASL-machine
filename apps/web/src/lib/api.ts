const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type SignMeta = {
  sign_id: string;
  gloss: string;
  category: string;
  unit: string;
  trained: boolean;
};

export type HintResponse = {
  sign_id: string;
  gloss: string;
  message: string;
  handshape: string;
  movement: string;
  location: string;
  framing: string;
};

export type ProgressSummary = {
  total_attempts: number;
  total_passes: number;
  mastered_count: number;
  recent_attempts: {
    id: string;
    sign_id: string;
    outcome: string;
    confidence: number | null;
    created_at: string;
  }[];
};

async function headers(userId: string, token?: string): Promise<HeadersInit> {
  const h: Record<string, string> = { "Content-Type": "application/json", "X-User-Id": userId };
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

export async function checkApiHealth(): Promise<{ ok: boolean; url: string }> {
  const url = API_URL;
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(3000) });
    return { ok: res.ok, url };
  } catch {
    return { ok: false, url };
  }
}

export async function fetchSigns(wave?: number): Promise<SignMeta[]> {
  const q = wave ? `?wave=${wave}` : "";
  const res = await fetch(`${API_URL}/signs${q}`);
  if (!res.ok) throw new Error("Failed to load signs");
  return res.json();
}

export async function fetchHint(signId: string, reason: string, userId: string): Promise<HintResponse> {
  const res = await fetch(`${API_URL}/signs/${signId}/hint?reason=${reason}`, {
    headers: await headers(userId),
  });
  if (!res.ok) throw new Error("Hint not found");
  return res.json();
}

export async function createSession(userId: string, token?: string) {
  const res = await fetch(`${API_URL}/sessions`, {
    method: "POST",
    headers: await headers(userId, token),
    body: "{}",
  });
  if (!res.ok) throw new Error("Session create failed");
  return res.json() as Promise<{ id: string }>;
}

export async function recordAttempt(
  userId: string,
  body: {
    sign_id: string;
    outcome: string;
    confidence?: number;
    predicted_label?: string;
    session_id?: string;
  },
  token?: string
) {
  const res = await fetch(`${API_URL}/attempts`, {
    method: "POST",
    headers: await headers(userId, token),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Failed to record attempt");
  return res.json();
}

export async function fetchProgress(userId: string, token?: string): Promise<ProgressSummary> {
  const res = await fetch(`${API_URL}/progress`, { headers: await headers(userId, token) });
  if (!res.ok) throw new Error("Progress fetch failed");
  return res.json();
}

export function trackEvent(name: string, props?: Record<string, unknown>) {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("asl-analytics", { detail: { name, props, ts: Date.now() } }));
  }
  console.debug("[analytics]", name, props);
}
