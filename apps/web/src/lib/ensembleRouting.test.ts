import { describe, expect, it } from "vitest";
import { routeSpecialistPrediction, type SpecialistRoute } from "./ensembleRouting";
import type { InferenceSummary } from "./inference";

function summary(label: string, confidence: number, margin = 0.1): InferenceSummary {
  return {
    predictedLabel: label,
    confidence,
    margin,
    topPredictions: [{ label, confidence }],
    probs: [confidence],
  };
}

const route: SpecialistRoute = {
  id: "specialist",
  releaseTag: "test",
  allowedSigns: ["four"],
  minConfidence: 0.4,
  minMargin: 0.05,
  maxPrimaryConfidence: 0.85,
};

describe("routeSpecialistPrediction", () => {
  it("uses a specialist only for approved confident signs", () => {
    const routed = routeSpecialistPrediction(summary("where", 0.5), summary("four", 0.72), route);
    expect(routed.predictedLabel).toBe("four");
    expect(routed.routedBy).toBe("specialist");
    expect(routed.primaryPrediction?.predictedLabel).toBe("where");
  });

  it("keeps primary when specialist sign is not approved", () => {
    const routed = routeSpecialistPrediction(summary("where", 0.5), summary("deaf", 0.9), route);
    expect(routed.predictedLabel).toBe("where");
    expect(routed.routedBy).toBeUndefined();
  });

  it("keeps primary when primary is already highly confident", () => {
    const routed = routeSpecialistPrediction(summary("where", 0.91), summary("four", 0.9), route);
    expect(routed.predictedLabel).toBe("where");
    expect(routed.routedBy).toBeUndefined();
  });
});
