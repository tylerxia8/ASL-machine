import { describe, expect, it } from "vitest";
import { evaluateAttempt } from "./threshold";

describe("evaluateAttempt", () => {
  it("fails on wrong label", () => {
    const r = evaluateAttempt("hello", "goodbye", 0.99);
    expect(r.outcome).toBe("fail");
  });

  it("passes at 90%+ correct label", () => {
    const r = evaluateAttempt("hello", "hello", 0.91);
    expect(r.outcome).toBe("pass");
  });

  it("retries between 70-90%", () => {
    const r = evaluateAttempt("hello", "hello", 0.75);
    expect(r.outcome).toBe("retry");
  });

  it("fails below 70%", () => {
    const r = evaluateAttempt("hello", "hello", 0.5);
    expect(r.outcome).toBe("fail");
  });
});
