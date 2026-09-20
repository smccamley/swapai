import { describe, expect, it } from "vitest";

import { evaluateCandidatePredictions } from "../src/candidate-evaluation.js";

describe("candidate promotion evidence", () => {
  it("counts a missing Needle classification as complete error", () => {
    const predictions = [
      {
        purpose: "validation" as const,
        resultBin: "false",
        reference: false,
        candidate: false,
        classificationFailed: true,
        input: "missing-call",
      },
      {
        purpose: "representative_test" as const,
        resultBin: "false",
        reference: false,
        candidate: false,
        input: "classified",
      },
      {
        purpose: "coverage_test" as const,
        resultBin: "false",
        reference: false,
        candidate: false,
        input: "covered",
      },
    ];

    const evidence = evaluateCandidatePredictions(
      { type: "boolean" },
      0.1,
      predictions,
    );

    expect(evidence.metrics).toContainEqual({
      purpose: "validation",
      resultBin: null,
      exampleCount: 1,
      error: 1,
      passed: false,
    });
    expect(evidence.passed).toBe(false);
    expect(evidence.maximumError).toBe(1);
  });

  it("rejects a low constant prediction even when its aggregate error looks good", () => {
    const predictions = [
      {
        purpose: "validation" as const,
        resultBin: "0..0.2",
        reference: 0,
        candidate: 0,
        input: "validation-negative",
      },
      ...Array.from({ length: 98 }, (_, index) => ({
        purpose: "representative_test" as const,
        resultBin: "0..0.2",
        reference: 0,
        candidate: 0,
        input: `representative-negative-${index}`,
      })),
      ...Array.from({ length: 2 }, (_, index) => ({
        purpose: "representative_test" as const,
        resultBin: "0.8..1",
        reference: 1,
        candidate: 0,
        input: `representative-positive-${index}`,
      })),
      ...Array.from({ length: 30 }, (_, index) => ({
        purpose: "coverage_test" as const,
        resultBin: "0..0.2",
        reference: 0,
        candidate: 0,
        input: `coverage-negative-${index}`,
      })),
      ...Array.from({ length: 30 }, (_, index) => ({
        purpose: "coverage_test" as const,
        resultBin: "0.8..1",
        reference: 1,
        candidate: 0,
        input: `coverage-positive-${index}`,
      })),
    ];

    const evidence = evaluateCandidatePredictions(
      { type: "number", min: 0, max: 1 },
      0.1,
      predictions,
    );

    expect(evidence.metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          purpose: "representative_test",
          resultBin: null,
          error: 0.02,
          passed: true,
        }),
        expect.objectContaining({
          purpose: "coverage_test",
          resultBin: "0.8..1",
          error: 1,
          passed: false,
        }),
      ]),
    );
    expect(evidence.passed).toBe(false);
    expect(evidence.maximumError).toBe(1);
  });
});
