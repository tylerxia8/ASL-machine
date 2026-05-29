export type PhraseLogEntry = {
  phrase: string;
  outcome: string;
  ts?: number;
};

export const PHRASE_LOG_KEY = "phrase_practice_log";

export function readPhraseLog(): PhraseLogEntry[] {
  try {
    const raw = localStorage.getItem(PHRASE_LOG_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function writePhraseLog(entries: PhraseLogEntry[]) {
  localStorage.setItem(PHRASE_LOG_KEY, JSON.stringify(entries));
}

export function summarizePhraseLog(entries: PhraseLogEntry[]) {
  const summary = {
    total: entries.length,
    pass: 0,
    retry: 0,
    byPhrase: {} as Record<string, { total: number; pass: number; retry: number }>,
  };
  for (const entry of entries) {
    if (entry.outcome === "pass") summary.pass += 1;
    else summary.retry += 1;
    const row = summary.byPhrase[entry.phrase] ?? (summary.byPhrase[entry.phrase] = { total: 0, pass: 0, retry: 0 });
    row.total += 1;
    if (entry.outcome === "pass") row.pass += 1;
    else row.retry += 1;
  }
  return summary;
}
