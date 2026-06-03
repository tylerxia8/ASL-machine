import { describe, expect, it } from "vitest";
import { recognitionFeedbackCsv, summarizeRecognitionFeedback } from "./recognitionFeedback";

describe("recognition feedback", () => {
  it("normalizes legacy camelCase and current snake_case entries", () => {
    const summary = summarizeRecognitionFeedback([
      { signId: "hello", predictedLabel: "hello", accepted: true },
      { sign_id: "hello", predicted_label: "who", correct: false },
    ]);

    expect(summary.total).toBe(2);
    expect(summary.accepted).toBe(1);
    expect(summary.rejected).toBe(1);
    expect(summary.bySign.hello.commonPredictions).toEqual({ hello: 1, who: 1 });
  });

  it("exports normalized csv rows", () => {
    const csv = recognitionFeedbackCsv([
      {
        ts: 0,
        sign_id: "how",
        predicted_label: "who",
        confidence: 0.42,
        agreement: 0.67,
        window_predictions: [{ label: "who", confidence: 0.8 }],
        correct: false,
      },
    ]);

    expect(csv).toContain('"how","who","0.420000","0.670000","who:0.8000","false"');
  });

  it("summarizes routed specialist feedback", () => {
    const summary = summarizeRecognitionFeedback([
      {
        sign_id: "four",
        predicted_label: "four",
        accepted: true,
        routed_by: "v32-weighted",
        primary_predicted_label: "where",
        specialist_predicted_label: "four",
      },
      {
        sign_id: "four",
        predicted_label: "four",
        accepted: false,
        routed_by: "v32-weighted",
      },
    ]);

    expect(summary.routed.total).toBe(2);
    expect(summary.routed.accepted).toBe(1);
    expect(summary.routed.byRoute["v32-weighted"].rejected).toBe(1);
  });
});
