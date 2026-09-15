import { describe, expect, it } from "vitest";

import { averageError, resultError } from "../src/evaluation.js";

describe("candidate classifier error", () => {
  it("normalizes numeric differences against the declared range", () => {
    const result = { type: "number" as const, min: -10, max: 30 };

    expect(resultError(result, 10, 6)).toBe(0.1);
  });

  it("scores boolean and closed-string matches as zero and misses as one", () => {
    expect(resultError({ type: "boolean" }, true, true)).toBe(0);
    expect(resultError({ type: "boolean" }, true, false)).toBe(1);
    expect(
      resultError({ type: "string", values: ["gbp", "usd"] }, "gbp", "gbp"),
    ).toBe(0);
    expect(
      resultError({ type: "string", values: ["gbp", "usd"] }, "gbp", "usd"),
    ).toBe(1);
  });

  it("averages error across held-out examples", () => {
    expect(
      averageError(
        { type: "number", min: 0, max: 1 },
        [
          { reference: 0.9, candidate: 0.8 },
          { reference: 0.4, candidate: 0.4 },
          { reference: 0.2, candidate: 0.4 },
          { reference: 0.7, candidate: 0.6 },
        ],
      ),
    ).toBeCloseTo(0.1);
  });

  it("refuses to report an average without held-out examples", () => {
    expect(() =>
      averageError({ type: "boolean" }, []),
    ).toThrow(expect.objectContaining({ code: "invalid_result" }));
  });
});
