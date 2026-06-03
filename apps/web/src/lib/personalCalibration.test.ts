import { describe, expect, it } from "vitest";
import { personalizeThresholds } from "./personalCalibration";
import type { RecognitionFeedbackSummary } from "./recognitionFeedback";

function summary(rejected: number, total: number): RecognitionFeedbackSummary {
  return {
    total,
    accepted: total - rejected,
    rejected,
    routed: { total: 0, accepted: 0, rejected: 0, byRoute: {} },
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

  it("does not adjust non-autopass weak-sign thresholds", () => {
    const next = personalizeThresholds({ passThreshold: 1.01, retryThreshold: 0.82 }, "how", summary(3, 4));
    expect(next.passThreshold).toBe(1.01);
    expect(next.retryThreshold).toBe(0.82);
    expect(next.adjustment).toBe(0);
  });
});
