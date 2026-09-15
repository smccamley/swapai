import { describe, expect, it } from "vitest";

import { normalizeConfig, validateResult } from "../src/config.js";
import { SwapAIError } from "../src/errors.js";

const baseConfig = {
  name: "accountant-relevance",
  result: { type: "number" as const, min: 0, max: 1 },
  retrainOnCount: 50,
  acceptableError: "10%" as const,
  retestInterval: 100,
  retestRevertOn: 3,
  model: "needle2" as const,
  maxTrainingSet: 10_000,
};

describe("classifier configuration", () => {
  it("treats percent and decimal acceptable error values identically", () => {
    expect(normalizeConfig(baseConfig).acceptableError).toBe(0.1);
    expect(
      normalizeConfig({ ...baseConfig, acceptableError: 0.1 }).acceptableError,
    ).toBe(0.1);
  });

  it.each([
    ["classifier name", { name: "" }],
    ["finite minimum", { result: { type: "number", min: Number.NaN, max: 1 } }],
    ["ordered bounds", { result: { type: "number", min: 1, max: 1 } }],
    ["closed string values", { result: { type: "string", values: [] } }],
    [
      "unique string values",
      { result: { type: "string", values: ["rabbit", "rabbit"] } },
    ],
    ["positive retrain count", { retrainOnCount: 0 }],
    ["acceptable error range", { acceptableError: "101%" }],
    ["whole retest interval", { retestInterval: 1.5 }],
    ["positive revert count", { retestRevertOn: 0 }],
    ["known model", { model: "needle3" }],
    ["training set large enough", { maxTrainingSet: 49 }],
  ])("rejects an invalid %s", (_description, change) => {
    expect(() => normalizeConfig({ ...baseConfig, ...change } as never)).toThrow(
      expect.objectContaining({ code: "invalid_configuration" }),
    );
  });
});

describe("classification results", () => {
  it("accepts finite bounded numbers, booleans, and declared strings", () => {
    expect(validateResult({ type: "number", min: 0, max: 1 }, 0.92)).toBe(
      0.92,
    );
    expect(validateResult({ type: "boolean" }, false)).toBe(false);
    expect(
      validateResult(
        { type: "string", values: ["rabbit", "fish", "pig"] },
        "fish",
      ),
    ).toBe("fish");
  });

  it.each([
    [{ type: "number", min: 0, max: 1 }, Number.POSITIVE_INFINITY],
    [{ type: "number", min: 0, max: 1 }, 1.01],
    [{ type: "boolean" }, 1],
    [{ type: "string", values: ["gbp", "usd", "eur"] }, "cad"],
  ])("rejects a result outside the declared allowed results", (result, value) => {
    expect(() => validateResult(result as never, value)).toThrow(
      expect.objectContaining({ code: "invalid_result" }),
    );
  });

  it("uses a stable public error type", () => {
    const error = new SwapAIError("not_trained", "No active model");

    expect(error).toMatchObject({
      name: "SwapAIError",
      code: "not_trained",
      message: "No active model",
    });
  });
});
