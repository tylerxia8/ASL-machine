export type RecognitionFeedbackEntry = {
  ts?: number;
  signId?: string;
  sign_id?: string;
  predictedLabel?: string;
  predicted_label?: string;
  confidence?: number;
  accepted?: boolean;
  correct?: boolean;
  top_predictions?: { label: string; confidence: number }[];
  tracking_ratio?: number | null;
  model_version?: string;
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

export function feedbackSignId(entry: RecognitionFeedbackEntry) {
  return entry.signId || entry.sign_id || "unknown";
}

export function feedbackPredictedLabel(entry: RecognitionFeedbackEntry) {
  return entry.predictedLabel || entry.predicted_label || "unknown";
}

export function feedbackAccepted(entry: RecognitionFeedbackEntry) {
  if (typeof entry.accepted === "boolean") return entry.accepted;
  if (typeof entry.correct === "boolean") return entry.correct;
  return false;
}

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
    const accepted = feedbackAccepted(entry);
    if (accepted) summary.accepted += 1;
    else summary.rejected += 1;

    const signId = feedbackSignId(entry);
    const row =
      summary.bySign[signId] ??
      (summary.bySign[signId] = { total: 0, accepted: 0, rejected: 0, commonPredictions: {} });
    row.total += 1;
    if (accepted) row.accepted += 1;
    else row.rejected += 1;

    const predicted = feedbackPredictedLabel(entry);
    row.commonPredictions[predicted] = (row.commonPredictions[predicted] ?? 0) + 1;
  }

  return summary;
}

export function recognitionFeedbackCsv(entries: RecognitionFeedbackEntry[]) {
  const header = ["timestamp", "sign_id", "predicted_label", "confidence", "accepted"];
  const rows = entries.map((entry) => [
    entry.ts ? new Date(entry.ts).toISOString() : "",
    feedbackSignId(entry),
    feedbackPredictedLabel(entry),
    typeof entry.confidence === "number" ? entry.confidence.toFixed(6) : "",
    feedbackAccepted(entry) ? "true" : "false",
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
