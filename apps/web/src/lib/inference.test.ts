import { describe, expect, it } from "vitest";
import { summarizeProbabilitiesForLabels, type LabelsFile } from "./inference";

const labels: LabelsFile = {
  sign_ids: ["hello", "who", "four"],
  label_to_idx: { hello: 0, who: 1, four: 2 },
  model_version: "test",
};

describe("summarizeProbabilitiesForLabels", () => {
  it("reports capture-window agreement for averaged predictions", () => {
    const summary = summarizeProbabilitiesForLabels(
      [0.2, 0.6, 0.2],
      labels,
      [
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.7, 0.2, 0.1],
      ]
    );

    expect(summary.predictedLabel).toBe("who");
    expect(summary.agreement).toBeCloseTo(2 / 3);
    expect(summary.windowPredictions).toEqual([
      { label: "who", confidence: 0.8 },
      { label: "who", confidence: 0.7 },
      { label: "hello", confidence: 0.7 },
    ]);
  });
});
