import type { SignMeta } from "./api";
import type { RecognitionCalibration } from "./recognitionCalibration";
import type { RecognitionFeedbackSummary } from "./recognitionFeedback";

export type LearningPriority = {
  sign: SignMeta;
  score: number;
  reasons: string[];
  f1?: number;
  support?: number;
  localWrong?: number;
  localTotal?: number;
};

export function buildLearningPriorities(
  signs: SignMeta[],
  calibration: RecognitionCalibration | null,
  feedback: RecognitionFeedbackSummary,
  counts: Record<string, number> = {}
) {
  return signs
    .map((sign) => {
      const threshold = calibration?.thresholds?.[sign.sign_id];
      const row = feedback.bySign[sign.sign_id];
      const f1 = threshold?.f1 ?? 1;
      const support = threshold?.support ?? 999;
      const localWrong = row?.rejected ?? 0;
      const localTotal = row?.total ?? 0;
      const clipCount = counts[sign.sign_id] ?? 0;
      const reasons: string[] = [];
      let score = 0;

      if (f1 < 0.5) {
        score += 90;
        reasons.push(`low model F1 ${Math.round(f1 * 100)}%`);
      } else if (f1 < 0.75) {
        score += 45;
        reasons.push(`model F1 ${Math.round(f1 * 100)}%`);
      }

      if (support < 10) {
        score += 45;
        reasons.push(`${support} test clips`);
      } else if (support < 25) {
        score += 20;
        reasons.push(`${support} test clips`);
      }

      if (localWrong > 0) {
        score += localWrong * 20 + Math.max(0, localWrong - (localTotal - localWrong)) * 10;
        reasons.push(`${localWrong}/${localTotal} local misses`);
      }

      if (clipCount < 10) {
        score += 30 - clipCount;
        reasons.push(`${clipCount} learner clips`);
      }

      if (reasons.length === 0) reasons.push("healthy");
      return { sign, score, reasons, f1, support, localWrong, localTotal } satisfies LearningPriority;
    })
    .sort((a, b) => b.score - a.score || a.sign.gloss.localeCompare(b.sign.gloss));
}

export function buildConfusionDrillSigns(signs: SignMeta[], calibration: RecognitionCalibration | null) {
  const byId = new Map(signs.map((sign) => [sign.sign_id, sign]));
  const seen = new Set<string>();
  const out: SignMeta[] = [];
  const rows = Object.entries(calibration?.confusions ?? {})
    .map(([pair, row]) => {
      const [prompt, predicted] = pair.split("->");
      return { prompt, predicted, count: row.count ?? 0 };
    })
    .filter((row) => byId.has(row.prompt) && byId.has(row.predicted))
    .sort((a, b) => b.count - a.count);

  for (const row of rows) {
    for (const signId of [row.prompt, row.predicted]) {
      if (seen.has(signId)) continue;
      const sign = byId.get(signId);
      if (!sign) continue;
      seen.add(signId);
      out.push(sign);
    }
  }

  for (const sign of signs) {
    if (!seen.has(sign.sign_id)) out.push(sign);
  }
  return out;
}
