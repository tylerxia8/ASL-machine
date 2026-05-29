import { describe, expect, it } from "vitest";
import { buildConfusionDrillSigns } from "./learningPlan";
import type { SignMeta } from "./api";

const signs: SignMeta[] = [
  { sign_id: "help", gloss: "help", category: "core", unit: "1", trained: true },
  { sign_id: "how", gloss: "how", category: "core", unit: "1", trained: true },
  { sign_id: "who", gloss: "who", category: "core", unit: "1", trained: true },
  { sign_id: "yes", gloss: "yes", category: "core", unit: "1", trained: true },
];

describe("buildConfusionDrillSigns", () => {
  it("orders prompt/predicted pairs by confusion count", () => {
    const ordered = buildConfusionDrillSigns(signs, {
      confusions: {
        "help->how": { count: 11, message: "different movement" },
        "help->who": { count: 8, message: "different location" },
      },
    });

    expect(ordered.map((sign) => sign.sign_id)).toEqual(["help", "how", "who", "yes"]);
  });
});
