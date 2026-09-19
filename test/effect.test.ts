import { Context, Effect, Either, Fiber } from "effect";
import { describe, expect, expectTypeOf, it } from "vitest";

import {
  classify,
  classifyConfigured,
  classifyWithReference,
  close,
  erase,
  flush,
  inspect,
  isTrained,
  logClassification,
  promoteCandidate,
  reconcileTraining,
  requestTraining,
} from "../src/effect.js";
import { SwapAIError } from "../src/errors.js";
import type { Classifier, ConfiguredClassifier } from "../src/types.js";

function classifierWith(
  run: Classifier<number>["classify"],
): Classifier<number> {
  return {
    isTrained: () => true,
    logClassification: () => undefined,
    clearTrainingData: () => undefined,
    erase: async () => undefined,
    classify: run,
    flush: () => Promise.resolve(),
    close: () => Promise.resolve(),
  };
}

describe("Effect adapter", () => {
  it("wraps the complete configured classifier lifecycle", async () => {
    const calls: string[] = [];
    const configured: ConfiguredClassifier<boolean, "family"> = {
      isTrained: () => true,
      classify: async () => true,
      logClassification: () => { calls.push("log"); },
      inspect: () => ({
        name: "configured", totalExamplesLogged: 0, retainedExamples: 0,
        readyForTraining: false, resultBins: [], facetCoverage: [], deficits: [],
        examplesByPurpose: {
          training: 0, validation: 0, representative_test: 0, coverage_test: 0,
        },
        latestTrainingRun: null, trainingRuns: [],
      }),
      requestTraining: async () => ({ status: "not_ready", deficits: [] }),
      promoteCandidate: async (trainingRunId) => ({
        status: "promoted", trainingRunId, datasetRevisionId: "revision",
      }),
      reconcileTraining: async () => [],
      erase: async () => { calls.push("erase"); },
      flush: async () => { calls.push("flush"); },
      close: async () => { calls.push("close"); },
    };

    await expect(Effect.runPromise(classifyConfigured(
      configured,
      "invoice",
      { family: "invoice" },
    ))).resolves.toBe(true);
    await Effect.runPromise(logClassification(
      configured,
      "invoice",
      true,
      { family: "invoice" },
    ));
    await expect(Effect.runPromise(isTrained(configured))).resolves.toBe(true);
    await expect(Effect.runPromise(inspect(configured))).resolves.toMatchObject({
      name: "configured",
    });
    await expect(Effect.runPromise(requestTraining(configured))).resolves
      .toMatchObject({ status: "not_ready" });
    await expect(Effect.runPromise(promoteCandidate(configured, "run"))).resolves
      .toMatchObject({ status: "promoted" });
    await expect(Effect.runPromise(reconcileTraining(configured))).resolves.toEqual([]);
    await Effect.runPromise(erase(configured));
    await Effect.runPromise(flush(configured));
    await Effect.runPromise(close(configured));
    expect(calls).toEqual(["log", "erase", "flush", "close"]);
  });

  it("classifies through a native Effect", async () => {
    const classifier = classifierWith(() => Promise.resolve(0.81));

    const program = classify(classifier, "I owe you £5");

    expectTypeOf(program).toEqualTypeOf<
      Effect.Effect<number, SwapAIError, never>
    >();
    await expect(Effect.runPromise(program)).resolves.toBe(0.81);
  });

  it("keeps SwapAI failures in the Effect error channel", async () => {
    const failure = new SwapAIError(
      "not_trained",
      "No trained classifier is available",
    );
    const classifier = classifierWith(() => Promise.reject(failure));

    const result = await Effect.runPromise(
      Effect.either(classify(classifier, "input")),
    );

    expect(Either.isLeft(result)).toBe(true);
    if (Either.isLeft(result)) {
      expect(result.left).toBe(failure);
    }
  });

  it("preserves the reference Effect environment", async () => {
    class ReferenceScore extends Context.Tag("ReferenceScore")<
      ReferenceScore,
      number
    >() {}

    const classifier = classifierWith((_input, reference) => {
      if (reference === undefined) {
        return Promise.reject(new Error("reference missing"));
      }
      return Promise.resolve(reference("reference input"));
    });
    const program = classifyWithReference(
      classifier,
      "managed input",
      () => ReferenceScore,
    );

    expectTypeOf(program).toEqualTypeOf<
      Effect.Effect<number, SwapAIError, ReferenceScore>
    >();
    await expect(
      Effect.runPromise(Effect.provideService(program, ReferenceScore, 0.92)),
    ).resolves.toBe(0.92);
  });

  it("preserves the reference Effect error value", async () => {
    const referenceFailure = { _tag: "ReferenceUnavailable" as const };
    const classifier = classifierWith((_input, reference) => {
      if (reference === undefined) {
        return Promise.reject(new Error("reference missing"));
      }
      return Promise.resolve(reference("reference input"));
    });
    const program = classifyWithReference(classifier, "managed input", () =>
      Effect.fail(referenceFailure),
    );

    const result = await Effect.runPromise(Effect.either(program));

    expect(Either.isLeft(result)).toBe(true);
    if (Either.isLeft(result)) {
      expect(result.left).toBe(referenceFailure);
    }
  });

  it("interrupts the running reference Effect", async () => {
    let interrupted = false;
    const classifier = classifierWith((_input, reference) => {
      if (reference === undefined) {
        return Promise.reject(new Error("reference missing"));
      }
      return Promise.resolve(reference("reference input"));
    });
    const program = classifyWithReference(classifier, "managed input", () =>
      Effect.async<number>((_resume, signal) => {
        signal.addEventListener("abort", () => {
          interrupted = true;
        });
      }),
    );

    const fiber = Effect.runFork(program);
    await new Promise<void>((resolve) => setImmediate(resolve));
    await Effect.runPromise(Fiber.interrupt(fiber));

    expect(interrupted).toBe(true);
  });
});
