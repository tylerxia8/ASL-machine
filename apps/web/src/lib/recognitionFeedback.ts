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
  window_predictions?: { label: string; confidence: number }[];
  agreement?: number;
  tracking_ratio?: number | null;
  model_version?: string;
  routed_by?: string;
  primary_predicted_label?: string;
  primary_confidence?: number;
  specialist_predicted_label?: string;
  specialist_confidence?: number;
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
  routed: {
    total: number;
    accepted: number;
    rejected: number;
    byRoute: Record<string, { total: number; accepted: number; rejected: number }>;
  };
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
    routed: { total: 0, accepted: 0, rejected: 0, byRoute: {} },
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

    if (entry.routed_by) {
      summary.routed.total += 1;
      if (accepted) summary.routed.accepted += 1;
      else summary.routed.rejected += 1;
      const route =
        summary.routed.byRoute[entry.routed_by] ??
        (summary.routed.byRoute[entry.routed_by] = { total: 0, accepted: 0, rejected: 0 });
      route.total += 1;
      if (accepted) route.accepted += 1;
      else route.rejected += 1;
    }
  }

  return summary;
}

export function recognitionFeedbackCsv(entries: RecognitionFeedbackEntry[]) {
  const header = [
    "timestamp",
    "sign_id",
    "predicted_label",
    "confidence",
    "agreement",
    "window_predictions",
    "accepted",
    "routed_by",
    "primary_predicted_label",
    "primary_confidence",
    "specialist_predicted_label",
    "specialist_confidence",
  ];
  const rows = entries.map((entry) => [
    entry.ts ? new Date(entry.ts).toISOString() : "",
    feedbackSignId(entry),
    feedbackPredictedLabel(entry),
    typeof entry.confidence === "number" ? entry.confidence.toFixed(6) : "",
    typeof entry.agreement === "number" ? entry.agreement.toFixed(6) : "",
    entry.window_predictions?.map((p) => `${p.label}:${p.confidence.toFixed(4)}`).join(";") ?? "",
    feedbackAccepted(entry) ? "true" : "false",
    entry.routed_by ?? "",
    entry.primary_predicted_label ?? "",
    typeof entry.primary_confidence === "number" ? entry.primary_confidence.toFixed(6) : "",
    entry.specialist_predicted_label ?? "",
    typeof entry.specialist_confidence === "number" ? entry.specialist_confidence.toFixed(6) : "",
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
