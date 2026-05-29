import { describe, expect, it } from "vitest";
import { personalizeThresholds } from "./personalCalibration";
import type { RecognitionFeedbackSummary } from "./recognitionFeedback";

function summary(rejected: number, total: number): RecognitionFeedbackSummary {
  return {
    total,
    accepted: total - rejected,
    rejected,
    bySign: {
      how: {
        total,
        accepted: total - rejected,
        rejected,
        commonPredictions: {},
      },
    },
  };
}

describe("personalizeThresholds", () => {
  it("tightens thresholds when local feedback has frequent misses", () => {
    const next = personalizeThresholds({ passThreshold: 0.8, retryThreshold: 0.6 }, "how", summary(3, 4));
    expect(next.passThreshold).toBeCloseTo(0.86);
    expect(next.retryThreshold).toBeCloseTo(0.63);
  });

  it("does not adjust until enough labels exist", () => {
    const next = personalizeThresholds({ passThreshold: 0.8, retryThreshold: 0.6 }, "how", summary(1, 2));
    expect(next.adjustment).toBe(0);
  });
});
