import { describe, expect, it } from "vitest";
import { summarizePhraseLog } from "./phraseLog";

describe("summarizePhraseLog", () => {
  it("counts phrase attempts by outcome and phrase", () => {
    const summary = summarizePhraseLog([
      { phrase: "hello_name", outcome: "pass" },
      { phrase: "hello_name", outcome: "retry" },
      { phrase: "thank_you", outcome: "pass" },
    ]);

    expect(summary.total).toBe(3);
    expect(summary.pass).toBe(2);
    expect(summary.retry).toBe(1);
    expect(summary.byPhrase.hello_name).toEqual({ total: 2, pass: 1, retry: 1 });
  });
});
