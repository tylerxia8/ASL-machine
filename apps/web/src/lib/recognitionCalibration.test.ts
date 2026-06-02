import { describe, expect, it } from "vitest";
import { reliabilityFor, thresholdsFor } from "./recognitionCalibration";

describe("recognition calibration", () => {
  it("marks weak low-support signs as non-autopass", () => {
    const calibration = {
      thresholds: {
        five: { passThreshold: 0.95, retryThreshold: 0.82, f1: 0, support: 1 },
      },
    };

    expect(reliabilityFor(calibration.thresholds.five)).toBe("weak");
    expect(thresholdsFor(calibration, "five")).toEqual({ passThreshold: 1.01, retryThreshold: 0.82 });
  });

  it("keeps calibrated thresholds for reliable signs", () => {
    const calibration = {
      thresholds: {
        eat: { passThreshold: 0.86, retryThreshold: 0.62, f1: 0.9, support: 75 },
      },
    };

    expect(reliabilityFor(calibration.thresholds.eat)).toBe("strong");
    expect(thresholdsFor(calibration, "eat")).toEqual({ passThreshold: 0.86, retryThreshold: 0.62 });
  });
});
