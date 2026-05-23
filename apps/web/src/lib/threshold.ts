export type EvalOutcome = "pass" | "fail" | "retry";

export type ThresholdResult = {
  outcome: EvalOutcome;
  confidence: number;
  predictedLabel: string;
  promptLabel: string;
};

export function evaluateAttempt(
  promptLabel: string,
  predictedLabel: string,
  confidence: number,
  passThreshold = 0.9,
  retryThreshold = 0.7
): ThresholdResult {
  if (predictedLabel !== promptLabel) {
    return { outcome: "fail", confidence, predictedLabel, promptLabel };
  }
  if (confidence >= passThreshold) {
    return { outcome: "pass", confidence, predictedLabel, promptLabel };
  }
  if (confidence >= retryThreshold) {
    return { outcome: "retry", confidence, predictedLabel, promptLabel };
  }
  return { outcome: "fail", confidence, predictedLabel, promptLabel };
}
