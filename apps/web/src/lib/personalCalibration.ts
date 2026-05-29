import type { SignThresholds } from "./recognitionCalibration";
import type { RecognitionFeedbackSummary } from "./recognitionFeedback";

export type PersonalThresholds = Required<Pick<SignThresholds, "passThreshold" | "retryThreshold">> & {
  adjustment: number;
};

export function personalizeThresholds(
  base: Required<Pick<SignThresholds, "passThreshold" | "retryThreshold">>,
  signId: string,
  feedback: RecognitionFeedbackSummary
): PersonalThresholds {
  const row = feedback.bySign[signId];
  if (!row || row.total < 3) return { ...base, adjustment: 0 };

  const missRate = row.rejected / row.total;
  const adjustment = missRate >= 0.5 ? 0.06 : missRate <= 0.15 ? -0.03 : 0;
  return {
    passThreshold: clamp(base.passThreshold + adjustment, 0.55, 0.96),
    retryThreshold: clamp(base.retryThreshold + adjustment / 2, 0.4, 0.9),
    adjustment,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
