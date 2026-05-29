export type RecognitionFeedbackEntry = {
  ts?: number;
  signId?: string;
  predictedLabel?: string;
  confidence?: number;
  accepted?: boolean;
};

export type RecognitionFeedbackSummary = {
  total: number;
  accepted: number;
  rejected: number;
  bySign: Record<
    string,
    {
      total: number;
      accepted: number;
      rejected: number;
      commonPredictions: Record<string, number>;
    }
  >;
};

const STORAGE_KEY = "recognition_feedback";

export function readRecognitionFeedback(): RecognitionFeedbackEntry[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function summarizeRecognitionFeedback(entries: RecognitionFeedbackEntry[]): RecognitionFeedbackSummary {
  const summary: RecognitionFeedbackSummary = {
    total: entries.length,
    accepted: 0,
    rejected: 0,
    bySign: {},
  };

  for (const entry of entries) {
    if (entry.accepted) summary.accepted += 1;
    else summary.rejected += 1;

    const signId = entry.signId || "unknown";
    const row =
      summary.bySign[signId] ??
      (summary.bySign[signId] = { total: 0, accepted: 0, rejected: 0, commonPredictions: {} });
    row.total += 1;
    if (entry.accepted) row.accepted += 1;
    else row.rejected += 1;

    const predicted = entry.predictedLabel || "unknown";
    row.commonPredictions[predicted] = (row.commonPredictions[predicted] ?? 0) + 1;
  }

  return summary;
}

export function recognitionFeedbackCsv(entries: RecognitionFeedbackEntry[]) {
  const header = ["timestamp", "sign_id", "predicted_label", "confidence", "accepted"];
  const rows = entries.map((entry) => [
    entry.ts ? new Date(entry.ts).toISOString() : "",
    entry.signId ?? "",
    entry.predictedLabel ?? "",
    typeof entry.confidence === "number" ? entry.confidence.toFixed(6) : "",
    entry.accepted ? "true" : "false",
  ]);
  return [header, ...rows]
    .map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(","))
    .join("\n");
}

export function downloadText(filename: string, text: string, type = "text/plain") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function clearRecognitionFeedback() {
  localStorage.removeItem(STORAGE_KEY);
}
