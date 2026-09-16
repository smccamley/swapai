import { mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  expectTypeOf,
  it,
  vi,
} from "vitest";

import { init } from "../src/index.js";
import { SwapAIError } from "../src/errors.js";
import { assignExampleSplit, openStorage } from "../src/storage.js";

const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite") as typeof import("node:sqlite");

const needle = vi.hoisted(() => {
  class NeedleModelArtifactError extends Error {}
  const model = {
    classify: vi.fn<(input: string) => Promise<number | boolean | string>>(),
    close: vi.fn(async () => undefined),
  };
  const runtime = {
    ready: vi.fn(async () => undefined),
    train: vi.fn(async (options: { resultConfig: { type: string } }) => ({
      modelPath: "/tmp/swapai-candidate.cact",
      needleVersion: options.resultConfig.type === "number"
        ? "2.0.14/number-buckets-v1"
        : "2.0.14",
    })),
    loadModel: vi.fn(async () => model),
    clearClassifierArtifacts: vi.fn(async () => undefined),
    clearClassifierGenerationArtifacts: vi.fn(async () => undefined),
    clearClassifierArtifactsThroughGeneration: vi.fn(async () => undefined),
    close: vi.fn(async () => undefined),
  };
  return { NeedleModelArtifactError, model, runtime };
});

vi.mock("../src/runtime.js", () => ({
  NEEDLE_VERSION: "2.0.14",
  needleModelVersion: (result: { type: string }) =>
    result.type === "number"
      ? "2.0.14/number-buckets-v1"
      : "2.0.14",
  NeedleModelArtifactError: needle.NeedleModelArtifactError,
  hasNeedleModelArtifacts: vi.fn(() => true),
  createNeedleRuntime: () => needle.runtime,
}));

const temporaryDirectories: string[] = [];

function makeDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "swapai-classifier-"));
  temporaryDirectories.push(directory);
  return directory;
}

function inputsForBothSplits(name: string): [string, string] {
  let training = "";
  let heldOut = "";
  for (let index = 0; training === "" || heldOut === ""; index += 1) {
    const input = `example-${index}`;
    if (assignExampleSplit(name, input) === "training") training = input;
    else heldOut = input;
  }
  return [training, heldOut];
}

function booleanConfig(dataDirectory: string, name = "relevance") {
  return {
    name,
    result: { type: "boolean" } as const,
    retrainOnCount: 2,
    acceptableError: "0%" as const,
    retestInterval: 2,
    retestRevertOn: 2,
    model: "needle2" as const,
    maxTrainingSet: 10,
    dataDirectory,
  };
}

function setLegacyTrainingLease(
  dataDirectory: string,
  name: string,
  leaseUntil: number,
): void {
  const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 2);
  database.prepare(`
    UPDATE classifiers
    SET training_lease_owner = 'legacy-owner',
        training_lease_until = ?,
        training_lease_epoch = NULL
    WHERE name = ?
  `).run(leaseUntil, name);
  database.close();
}

beforeEach(() => {
  vi.clearAllMocks();
  needle.model.classify.mockResolvedValue(true);
});

afterEach(() => {
  vi.restoreAllMocks();
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("classifier", () => {
  it("initializes synchronously and uses the reference until trained", async () => {
    const classifier = init(booleanConfig(makeDirectory()));
    const reference = vi.fn(async () => true);

    expect(classifier.isTrained()).toBe(false);
    expect(() => classifier.logClassification("bad", "true" as never)).toThrow(
      /declared allowed results/i,
    );
    await expect(classifier.classify("I owe you £5", reference)).resolves.toBe(
      true,
    );
    await classifier.flush();

    expect(reference).toHaveBeenCalledOnce();
    await classifier.close();
  });

  it("supports public numeric and closed-string classifiers", async () => {
    const dataDirectory = makeDirectory();
    const common = {
      retrainOnCount: 2,
      acceptableError: "10%" as const,
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2" as const,
      maxTrainingSet: 10,
      dataDirectory,
    };
    const score = init({
      ...common,
      name: "score",
      result: { type: "number" as const, min: 0, max: 1 },
    });
    const currency = init({
      ...common,
      name: "currency",
      result: {
        type: "string" as const,
        values: ["gbp", "usd", "eur"],
      },
    });

    await expect(score.classify("score input", () => 0.92)).resolves.toBe(0.92);
    const currencyResult = await currency.classify("currency input", () => "gbp");
    expect(currencyResult).toBe("gbp");
    expectTypeOf(currencyResult).toEqualTypeOf<"gbp" | "usd" | "eur">();
    await score.close();
    await currency.close();
  });

  it("rebuilds old numeric models from retained examples", async () => {
    const dataDirectory = makeDirectory();
    const name = "old-numeric-model";
    const config = {
      name,
      result: { type: "number" as const, min: 0, max: 1 },
      retrainOnCount: 2,
      acceptableError: "10%" as const,
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2" as const,
      maxTrainingSet: 10,
      dataDirectory,
    };
    const stored = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0.1,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    stored.addExample(trainingInput, 0.92);
    stored.addExample(heldOutInput, 0.92);
    stored.markTrainingAttempted(2);
    stored.promoteGeneration({
      modelPath: "/tmp/old-numeric.cact",
      needleVersion: "2.0.14",
    });
    stored.close();
    needle.runtime.loadModel.mockClear();
    needle.model.classify.mockResolvedValue(0.92);

    const classifier = init(config);

    expect(classifier.isTrained()).toBe(false);
    await classifier.flush();
    expect(needle.runtime.loadModel).not.toHaveBeenCalledWith(
      expect.objectContaining({ modelPath: "/tmp/old-numeric.cact" }),
    );
    expect(needle.runtime.train).toHaveBeenCalledWith(
      expect.objectContaining({
        generation: 2,
        examples: [expect.objectContaining({ input: trainingInput })],
      }),
    );
    expect(classifier.isTrained()).toBe(true);
    const reopened = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0.1,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    expect(reopened.snapshot()).toMatchObject({
      activeGeneration: 2,
      trained: true,
      activeExampleCount: 2,
    });
    reopened.close();
    await classifier.close();
  });

  it("rebuilds a trained numeric model when its label sidecar cannot load", async () => {
    const dataDirectory = makeDirectory();
    const name = "missing-number-labels";
    const config = {
      name,
      result: { type: "number" as const, min: 0, max: 1 },
      retrainOnCount: 2,
      acceptableError: "10%" as const,
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2" as const,
      maxTrainingSet: 10,
      dataDirectory,
    };
    const stored = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: 10,
      config: {
        result: config.result,
        retrainOnCount: 2,
        acceptableError: 0.1,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
    });
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    stored.addExample(trainingInput, 0.92);
    stored.addExample(heldOutInput, 0.92);
    stored.markTrainingAttempted(2);
    stored.promoteGeneration({
      modelPath: "/tmp/missing-labels.cact",
      needleVersion: "2.0.14/number-buckets-v1",
    });
    stored.close();
    const runtimeModule = await import("../src/runtime.js") as typeof import("../src/runtime.js") & {
      hasNeedleModelArtifacts: ReturnType<typeof vi.fn>;
    };
    runtimeModule.hasNeedleModelArtifacts.mockReturnValueOnce(false);
    needle.model.classify.mockResolvedValue(0.92);

    const classifier = init(config);
    expect(classifier.isTrained()).toBe(false);
    await classifier.flush();

    expect(classifier.isTrained()).toBe(true);
    expect(needle.runtime.train).toHaveBeenCalledWith(
      expect.objectContaining({ generation: 2 }),
    );
    await classifier.close();
  });

  it("trains, tests the exported model on held-out examples, then promotes it", async () => {
    const name = "promoted";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init(booleanConfig(makeDirectory(), name));

    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();

    expect(needle.runtime.train).toHaveBeenCalledWith(
      expect.objectContaining({
        classifierName: name,
        generation: 1,
        examples: [expect.objectContaining({ input: trainingInput, result: true })],
      }),
    );
    expect(needle.model.classify).toHaveBeenCalledWith(heldOutInput);
    expect(classifier.isTrained()).toBe(true);
    await expect(classifier.classify("new input")).resolves.toBe(true);
    await classifier.close();
  });

  it("does not promote a model that exceeds the average acceptable error", async () => {
    const name = "rejected";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    needle.model.classify.mockResolvedValue(false);
    const classifier = init(booleanConfig(makeDirectory(), name));

    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();

    expect(classifier.isTrained()).toBe(false);
    expect(needle.model.close).toHaveBeenCalled();
    await expect(classifier.classify("manual only")).rejects.toMatchObject({
      code: "not_trained",
    });
    await classifier.close();
  });

  it("rejects a candidate that cannot classify held-out examples and retries after the next batch", async () => {
    const name = "classification-failed-candidate";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classificationFailure = new SwapAIError(
      "classification_failed",
      "Needle did not return exactly one classify call",
    );
    needle.model.classify
      .mockRejectedValueOnce(classificationFailure)
      .mockResolvedValue(true);
    const classifier = init(booleanConfig(makeDirectory(), name));

    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await expect(classifier.flush()).resolves.toBeUndefined();
    expect(classifier.isTrained()).toBe(false);
    expect(needle.runtime.train).toHaveBeenCalledTimes(1);

    classifier.logClassification(`${trainingInput}-next`, true);
    classifier.logClassification(`${heldOutInput}-next`, true);
    await classifier.flush();

    expect(needle.runtime.train).toHaveBeenCalledTimes(2);
    expect(classifier.isTrained()).toBe(true);
    await classifier.close();
  });

  it("restores trained state and the saved model after restart", async () => {
    const dataDirectory = makeDirectory();
    const name = "restart";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const first = init(booleanConfig(dataDirectory, name));
    first.logClassification(trainingInput, true);
    first.logClassification(heldOutInput, true);
    await first.flush();
    expect(first.isTrained()).toBe(true);
    await first.close();

    const second = init(booleanConfig(dataDirectory, name));
    expect(second.isTrained()).toBe(true);
    await expect(second.classify("after restart")).resolves.toBe(true);
    expect(needle.runtime.loadModel).toHaveBeenCalledWith(
      expect.objectContaining({ modelPath: "/tmp/swapai-candidate.cact" }),
    );
    await second.close();
  });

  it("clears only this classifier's training data and stays untrained after restart", async () => {
    const dataDirectory = makeDirectory();
    const clearedName = "customer-erasure";
    const preservedName = "other-classifier";
    const [clearedTraining, clearedHeldOut] = inputsForBothSplits(clearedName);
    const [preservedTraining, preservedHeldOut] =
      inputsForBothSplits(preservedName);
    const cleared = init(booleanConfig(dataDirectory, clearedName));
    const preserved = init(booleanConfig(dataDirectory, preservedName));

    cleared.logClassification(clearedTraining, true);
    cleared.logClassification(clearedHeldOut, true);
    preserved.logClassification(preservedTraining, true);
    preserved.logClassification(preservedHeldOut, true);
    await Promise.all([cleared.flush(), preserved.flush()]);
    expect(cleared.isTrained()).toBe(true);
    expect(preserved.isTrained()).toBe(true);

    expect(cleared.clearTrainingData()).toBeUndefined();
    expect(cleared.isTrained()).toBe(false);
    await cleared.flush();
    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration,
    ).toHaveBeenCalledWith(
      clearedName,
      expect.any(Number),
    );
    expect(preserved.isTrained()).toBe(true);
    await cleared.close();
    await preserved.close();

    const clearedAfterRestart = init(
      booleanConfig(dataDirectory, clearedName),
    );
    const preservedAfterRestart = init(
      booleanConfig(dataDirectory, preservedName),
    );
    expect(clearedAfterRestart.isTrained()).toBe(false);
    expect(preservedAfterRestart.isTrained()).toBe(true);
    await clearedAfterRestart.close();
    await preservedAfterRestart.close();
  });

  it("does not retain a classification that was still running when data was cleared", async () => {
    const dataDirectory = makeDirectory();
    const name = "in-flight-erasure";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init(booleanConfig(dataDirectory, name));
    let finishReference!: (result: boolean) => void;
    const referenceResult = new Promise<boolean>((resolve) => {
      finishReference = resolve;
    });

    const inFlight = classifier.classify("old customer input", () =>
      referenceResult,
    );
    classifier.clearTrainingData();
    finishReference(true);
    await expect(inFlight).resolves.toBe(true);

    classifier.logClassification(trainingInput, true);
    await classifier.flush();
    expect(needle.runtime.train).not.toHaveBeenCalled();

    classifier.logClassification(heldOutInput, true);
    await classifier.flush();
    expect(needle.runtime.train).toHaveBeenCalledOnce();
    await classifier.close();
  });

  it("waits for running training before deleting its artifacts", async () => {
    const dataDirectory = makeDirectory();
    const name = "training-erasure";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    let finishTraining!: () => void;
    const trainingGate = new Promise<void>((resolve) => {
      finishTraining = resolve;
    });
    needle.runtime.train.mockImplementationOnce(async () => {
      await trainingGate;
      return {
        modelPath: "/tmp/swapai-candidate.cact",
        needleVersion: "2.0.14",
      };
    });
    const classifier = init(booleanConfig(dataDirectory, name));

    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await new Promise<void>((resolve) => setImmediate(resolve));
    classifier.clearTrainingData();
    expect(classifier.isTrained()).toBe(false);

    let clearFinished = false;
    const flushed = classifier.flush().then(() => {
      clearFinished = true;
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(clearFinished).toBe(false);
    finishTraining();
    await flushed;

    expect(classifier.isTrained()).toBe(false);
    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration,
    ).toHaveBeenCalledWith(name, expect.any(Number));
    expect(
      needle.runtime.train.mock.invocationCallOrder[0],
    ).toBeLessThan(
      needle.runtime.clearClassifierArtifactsThroughGeneration.mock
        .invocationCallOrder[0]!,
    );
    await classifier.close();
  });

  it("rejects an in-flight local result before closing and deleting its cleared model", async () => {
    const dataDirectory = makeDirectory();
    const name = "classification-erasure";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init(booleanConfig(dataDirectory, name));
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();

    let finishClassification!: (result: boolean) => void;
    const classificationGate = new Promise<boolean>((resolve) => {
      finishClassification = resolve;
    });
    needle.model.classify.mockReturnValueOnce(classificationGate);
    const classification = classifier.classify("currently classifying");
    await new Promise<void>((resolve) => setImmediate(resolve));

    classifier.clearTrainingData();
    let clearFinished = false;
    const flushed = classifier.flush().then(() => {
      clearFinished = true;
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(clearFinished).toBe(false);

    finishClassification(true);
    await expect(classification).rejects.toMatchObject({
      code: "not_trained",
    });
    await flushed;
    expect(classifier.isTrained()).toBe(false);
    expect(needle.model.close).toHaveBeenCalled();
    await classifier.close();
  });

  it("attempts model artifact deletion when SQLite WAL truncation fails", async () => {
    const dataDirectory = makeDirectory();
    const name = "checkpoint-erasure";
    const classifier = init(booleanConfig(dataDirectory, name));
    classifier.logClassification("private input", true);
    await classifier.flush();
    const reader = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    reader.exec("BEGIN");
    reader.prepare("SELECT * FROM examples").get();

    classifier.clearTrainingData();
    await expect(classifier.flush()).rejects.toMatchObject({
      code: "storage_failed",
      cause: expect.objectContaining({
        message: expect.stringMatching(/WAL could not be truncated/i),
      }),
    });
    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration,
    ).toHaveBeenCalledWith(name, expect.any(Number));

    await expect(classifier.flush()).rejects.toMatchObject({
      code: "storage_failed",
      cause: expect.objectContaining({
        message: expect.stringMatching(/WAL could not be truncated/i),
      }),
    });

    reader.exec("ROLLBACK");
    reader.close();
    await expect(classifier.flush()).resolves.toBeUndefined();
    await classifier.close();
  });

  it("stops another live instance from serving or recording the erased epoch", async () => {
    const dataDirectory = makeDirectory();
    const name = "shared-instance-erasure";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const first = init(booleanConfig(dataDirectory, name));
    first.logClassification(trainingInput, true);
    first.logClassification(heldOutInput, true);
    await first.flush();
    const second = init(booleanConfig(dataDirectory, name));
    await second.flush();
    expect(second.isTrained()).toBe(true);
    needle.model.classify.mockClear();
    needle.runtime.train.mockClear();

    let finishCandidate!: (result: boolean) => void;
    const candidateResult = new Promise<boolean>((resolve) => {
      finishCandidate = resolve;
    });
    needle.model.classify.mockReturnValueOnce(candidateResult);
    let finishReference!: (result: boolean) => void;
    const referenceResult = new Promise<boolean>((resolve) => {
      finishReference = resolve;
    });
    const staleClassification = second.classify("old private input", () =>
      referenceResult,
    );
    await new Promise<void>((resolve) => setImmediate(resolve));
    first.clearTrainingData();

    expect(second.isTrained()).toBe(false);
    finishCandidate(true);
    await new Promise<void>((resolve) => setImmediate(resolve));
    finishReference(true);
    await expect(staleClassification).resolves.toBe(true);
    await Promise.all([first.flush(), second.flush()]);
    expect(needle.model.classify).toHaveBeenCalledOnce();
    await first.close();
    await second.close();

    const restarted = init(booleanConfig(dataDirectory, name));
    expect(restarted.isTrained()).toBe(false);
    restarted.logClassification(trainingInput, true);
    await restarted.flush();
    expect(needle.runtime.train).not.toHaveBeenCalled();
    await restarted.close();
  });

  it("prevents another live instance from promoting training from an erased epoch", async () => {
    const dataDirectory = makeDirectory();
    const name = "shared-training-erasure";
    const config = booleanConfig(dataDirectory, name);
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const childArtifactLock = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    let finishTraining!: () => void;
    const trainingGate = new Promise<void>((resolve) => {
      finishTraining = resolve;
    });
    needle.runtime.train.mockImplementationOnce(async () => {
      expect(childArtifactLock.claimArtifactWriteLock()).toBe(true);
      await trainingGate;
      childArtifactLock.releaseArtifactWriteLock();
      return {
        modelPath: "/tmp/stale-candidate.cact",
        needleVersion: "2.0.14",
      };
    });
    const clearingInstance = init(config);
    const trainingInstance = init(config);
    trainingInstance.logClassification(trainingInput, true);
    trainingInstance.logClassification(heldOutInput, true);
    await new Promise<void>((resolve) => setImmediate(resolve));
    const afterLeaseExpiry = Date.now() + 10 * 60 * 1000;
    const clock = vi.spyOn(Date, "now").mockReturnValue(afterLeaseExpiry);

    clearingInstance.clearTrainingData();
    let clearFinished = false;
    const clearing = clearingInstance.flush().then(() => {
      clearFinished = true;
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    expect(clearFinished).toBe(false);
    finishTraining();
    await clearing;
    await trainingInstance.flush();
    clock.mockRestore();

    expect(trainingInstance.isTrained()).toBe(false);
    expect(
      needle.runtime.clearClassifierGenerationArtifacts,
    ).toHaveBeenCalledWith(name, 1);
    await clearingInstance.close();
    await trainingInstance.close();
    childArtifactLock.close();
    const restarted = init(config);
    expect(restarted.isTrained()).toBe(false);
    await restarted.close();
  });

  it("does not promote training after another instance resets the generation", async () => {
    const dataDirectory = makeDirectory();
    const name = "reset-during-training";
    const config = booleanConfig(dataDirectory, name);
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    let finishTraining!: () => void;
    const trainingGate = new Promise<void>((resolve) => {
      finishTraining = resolve;
    });
    needle.runtime.train.mockImplementationOnce(async () => {
      await trainingGate;
      return {
        modelPath: "/tmp/reset-stale-candidate.cact",
        needleVersion: "2.0.14",
      };
    });
    const classifier = init(config);
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await vi.waitFor(() =>
      expect(needle.runtime.train).toHaveBeenCalledOnce(),
    );
    const resetting = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    expect(resetting.archiveAndReset(0)).not.toBeNull();
    resetting.close();
    finishTraining();
    await classifier.flush();

    expect(classifier.isTrained()).toBe(false);
    expect(
      needle.runtime.clearClassifierGenerationArtifacts,
    ).toHaveBeenCalledWith(name, 1);
    await classifier.close();
  });

  it("allows only one concurrent failed retest to reset a generation", async () => {
    const dataDirectory = makeDirectory();
    const name = "concurrent-reset";
    const config = {
      ...booleanConfig(dataDirectory, name),
      retestInterval: 1,
      retestRevertOn: 1,
    };
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const first = init(config);
    first.logClassification(trainingInput, true);
    first.logClassification(heldOutInput, true);
    await first.flush();
    const second = init(config);
    await second.flush();
    needle.model.classify.mockResolvedValue(false);

    await Promise.all([
      first.classify("first failed retest", async () => true),
      second.classify("second failed retest", async () => true),
    ]);

    const check = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    expect(check.listGenerations()).toHaveLength(2);
    check.close();
    await first.close();
    await second.close();
  });

  it("finishes a persisted pending clear when the classifier is reopened", async () => {
    const dataDirectory = makeDirectory();
    const name = "interrupted-erasure";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const first = init(booleanConfig(dataDirectory, name));
    first.logClassification(trainingInput, true);
    first.logClassification(heldOutInput, true);
    await first.flush();
    needle.runtime.clearClassifierArtifactsThroughGeneration.mockRejectedValue(
      new Error("disk unavailable"),
    );

    first.clearTrainingData();
    await expect(first.flush()).rejects.toMatchObject({ code: "storage_failed" });
    await expect(first.close()).rejects.toMatchObject({ code: "storage_failed" });

    needle.runtime.clearClassifierArtifactsThroughGeneration.mockResolvedValue(
      undefined,
    );
    const reopened = init(booleanConfig(dataDirectory, name));
    expect(reopened.isTrained()).toBe(false);
    await expect(reopened.flush()).resolves.toBeUndefined();
    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration,
    ).toHaveBeenLastCalledWith(
      name,
      expect.any(Number),
    );
    await reopened.close();
    await expect(first.close()).resolves.toBeUndefined();
  });

  it("finishes a clear marked before a process can queue deletion work", async () => {
    const dataDirectory = makeDirectory();
    const name = "crash-before-deletion";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const config = booleanConfig(dataDirectory, name);
    const seeded = init(config);
    seeded.logClassification(trainingInput, true);
    seeded.logClassification(heldOutInput, true);
    await seeded.flush();
    await seeded.close();
    const interruptedStorage = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    interruptedStorage.beginClearTrainingData();
    expect(interruptedStorage.snapshot().clearPending).toBe(true);
    interruptedStorage.close();
    needle.runtime.clearClassifierArtifactsThroughGeneration.mockClear();

    const reopened = init(config);
    expect(reopened.isTrained()).toBe(false);
    await reopened.flush();

    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration,
    ).toHaveBeenCalledWith(name, expect.any(Number));
    await reopened.close();
  });

  it("finishes pending deletion before reporting a Python startup failure", async () => {
    const dataDirectory = makeDirectory();
    const name = "clear-before-runtime";
    const config = booleanConfig(dataDirectory, name);
    const storage = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    storage.addExample("private input", true);
    storage.beginClearTrainingData();
    storage.close();
    needle.runtime.ready.mockRejectedValueOnce(new Error("Python unavailable"));
    needle.runtime.clearClassifierArtifactsThroughGeneration.mockClear();

    const classifier = init(config);
    await expect(classifier.flush()).rejects.toMatchObject({
      code: "service_unavailable",
    });

    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration,
    ).toHaveBeenCalledWith(name, expect.any(Number));
    const check = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    expect(check.snapshot().clearPending).toBe(false);
    check.close();
    await classifier.close();
  });

  it("recovers from a transient runtime startup failure without retiring the model", async () => {
    const dataDirectory = makeDirectory();
    const name = "runtime-recovers";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    needle.runtime.ready.mockRejectedValueOnce(new Error("Python unavailable"));
    const stored = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: 10,
      config: {
        result: { type: "boolean" },
        retrainOnCount: 2,
        acceptableError: 0,
        retestInterval: 2,
        retestRevertOn: 2,
        model: "needle2",
      },
    });
    stored.addExample(trainingInput, true);
    stored.addExample(heldOutInput, true);
    stored.promoteGeneration({
      modelPath: "/tmp/transient-model.cact",
      needleVersion: "2.0.14",
    });
    stored.close();
    const classifier = init(booleanConfig(dataDirectory, name));

    expect(classifier.isTrained()).toBe(true);
    await expect(classifier.classify("after recovery")).resolves.toBe(true);
    expect(needle.runtime.loadModel).toHaveBeenCalledWith(
      expect.objectContaining({ modelPath: "/tmp/transient-model.cact" }),
    );
    await expect(classifier.flush()).rejects.toMatchObject({
      code: "service_unavailable",
    });
    await classifier.close();
  });

  it("retries closing the loaded model until the clear can finish", async () => {
    const dataDirectory = makeDirectory();
    const name = "model-close-retry";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init(booleanConfig(dataDirectory, name));
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();
    needle.model.close.mockRejectedValue(new Error("worker would not stop"));

    classifier.clearTrainingData();
    await expect(classifier.flush()).rejects.toMatchObject({
      code: "storage_failed",
    });
    const callsAfterFirstFlush = needle.model.close.mock.calls.length;
    await expect(classifier.flush()).rejects.toMatchObject({
      code: "storage_failed",
    });
    expect(needle.model.close.mock.calls.length).toBeGreaterThan(
      callsAfterFirstFlush,
    );

    needle.model.close.mockResolvedValue(undefined);
    await expect(classifier.flush()).resolves.toBeUndefined();
    expect(classifier.isTrained()).toBe(false);
    await classifier.close();
  });

  it("attempts runtime teardown when closing the loaded model fails", async () => {
    const dataDirectory = makeDirectory();
    const name = "close-teardown";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init(booleanConfig(dataDirectory, name));
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();
    needle.model.close.mockRejectedValueOnce(new Error("model close failed"));

    await expect(classifier.close()).rejects.toThrow("model close failed");
    expect(needle.runtime.close).toHaveBeenCalledOnce();
  });

  it("rejects late work and shares concurrent close teardown", async () => {
    const classifier = init(booleanConfig(makeDirectory(), "closing-gate"));
    let finishRuntimeClose!: () => void;
    const runtimeCloseGate = new Promise<void>((resolve) => {
      finishRuntimeClose = resolve;
    });
    needle.runtime.close.mockImplementationOnce(async () => {
      await runtimeCloseGate;
      return undefined;
    });

    const firstClose = classifier.close();
    await vi.waitFor(() => expect(needle.runtime.close).toHaveBeenCalledOnce());
    expect(() => classifier.isTrained()).toThrow(/closed/i);
    expect(() => classifier.logClassification("late", true)).toThrow(/closed/i);
    expect(() => classifier.clearTrainingData()).toThrow(/closed/i);
    expect(() => classifier.classify("late")).toThrow(/closed/i);
    await expect(classifier.flush()).rejects.toThrow(/closed/i);
    const secondClose = classifier.close();

    finishRuntimeClose();
    await Promise.all([firstClose, secondClose]);
    expect(needle.runtime.close).toHaveBeenCalledOnce();
  });

  it("treats repeated clear calls as separate data cutoffs", async () => {
    const dataDirectory = makeDirectory();
    const name = "repeated-clear";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init(booleanConfig(dataDirectory, name));

    classifier.clearTrainingData();
    classifier.logClassification("between clear calls", true);
    classifier.clearTrainingData();
    await classifier.flush();
    classifier.logClassification(trainingInput, true);
    await classifier.flush();
    expect(needle.runtime.train).not.toHaveBeenCalled();
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();
    expect(needle.runtime.train).toHaveBeenCalledOnce();
    await classifier.close();
  });

  it("sees a replacement trained by another live instance on the first check", async () => {
    const dataDirectory = makeDirectory();
    const name = "external-replacement";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const replacing = init(booleanConfig(dataDirectory, name));
    const observing = init(booleanConfig(dataDirectory, name));
    replacing.clearTrainingData();
    await replacing.flush();
    expect(observing.isTrained()).toBe(false);
    replacing.logClassification(trainingInput, true);
    replacing.logClassification(heldOutInput, true);
    await replacing.flush();

    expect(observing.isTrained()).toBe(true);
    await replacing.close();
    await observing.close();
  });

  it("serializes concurrent clear workers and keeps their generation cutoffs separate", async () => {
    const dataDirectory = makeDirectory();
    const name = "concurrent-clears";
    const first = init(booleanConfig(dataDirectory, name));
    const second = init(booleanConfig(dataDirectory, name));
    let releaseFirstDeletion!: () => void;
    const firstDeletion = new Promise<void>((resolve) => {
      releaseFirstDeletion = resolve;
    });
    needle.runtime.clearClassifierArtifactsThroughGeneration
      .mockImplementationOnce(async () => {
        await firstDeletion;
        return undefined;
      })
      .mockResolvedValue(undefined);

    first.clearTrainingData();
    const firstFlush = first.flush();
    await new Promise<void>((resolve) => setImmediate(resolve));
    second.clearTrainingData();
    let secondFinished = false;
    const secondFlush = second.flush().then(() => {
      secondFinished = true;
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    expect(secondFinished).toBe(false);
    releaseFirstDeletion();
    await Promise.all([firstFlush, secondFlush]);

    expect(
      needle.runtime.clearClassifierArtifactsThroughGeneration.mock.calls,
    ).toEqual([
      [name, 1],
      [name, 2],
    ]);
    await first.close();
    await second.close();
  });

  it("keeps a model-close failure when another instance supersedes the clear", async () => {
    const dataDirectory = makeDirectory();
    const name = "superseded-close-retry";
    const config = booleanConfig(dataDirectory, name);
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const firstModel = {
      classify: vi.fn(async () => true),
      close: vi
        .fn<() => Promise<void>>()
        .mockRejectedValueOnce(new Error("worker still running"))
        .mockResolvedValue(undefined),
    };
    const secondModel = {
      classify: vi.fn(async () => true),
      close: vi.fn(async () => undefined),
    };
    needle.runtime.loadModel
      .mockResolvedValueOnce(firstModel)
      .mockResolvedValueOnce(secondModel);
    const first = init(config);
    first.logClassification(trainingInput, true);
    first.logClassification(heldOutInput, true);
    await first.flush();
    const second = init(config);
    await second.flush();
    const lock = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    expect(lock.claimArtifactWriteLock()).toBe(true);

    first.clearTrainingData();
    const firstFlush = first.flush();
    await vi.waitFor(() => expect(firstModel.close).toHaveBeenCalledOnce());
    second.clearTrainingData();
    lock.releaseArtifactWriteLock();
    lock.close();

    await expect(firstFlush).rejects.toMatchObject({ code: "storage_failed" });
    await expect(second.flush()).resolves.toBeUndefined();
    await expect(first.flush()).resolves.toBeUndefined();
    expect(firstModel.close).toHaveBeenCalledTimes(2);
    await first.close();
    await second.close();
  });

  it("times out without certifying erasure while an artifact lock is held", async () => {
    const dataDirectory = makeDirectory();
    const name = "stuck-artifact-writer";
    const config = booleanConfig(dataDirectory, name);
    const classifier = init(config);
    await classifier.flush();
    const lock = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: 0,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
    expect(lock.claimArtifactWriteLock()).toBe(true);
    let now = Date.now();
    const clock = vi.spyOn(Date, "now").mockImplementation(() => now);

    classifier.clearTrainingData();
    const flushing = classifier.flush();
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    now += 30_001;
    await expect(flushing).rejects.toMatchObject({ code: "storage_failed" });
    expect(lock.snapshot().clearPending).toBe(true);

    lock.releaseArtifactWriteLock();
    clock.mockRestore();
    await expect(classifier.flush()).resolves.toBeUndefined();
    lock.close();
    await classifier.close();
  });

  it("clears a migrated training lease that was already stale", async () => {
    const dataDirectory = makeDirectory();
    const name = "stale-legacy-lease";
    const config = booleanConfig(dataDirectory, name);
    const setup = init(config);
    await setup.close();
    setLegacyTrainingLease(dataDirectory, name, Date.now() - 1);

    const classifier = init(config);
    classifier.clearTrainingData();
    await expect(classifier.flush()).resolves.toBeUndefined();
    expect(classifier.isTrained()).toBe(false);
    await classifier.close();
  });

  it("retries a migrated live lease after it expires", async () => {
    const dataDirectory = makeDirectory();
    const name = "expiring-legacy-lease";
    const config = booleanConfig(dataDirectory, name);
    const setup = init(config);
    await setup.close();
    let now = Date.now();
    setLegacyTrainingLease(dataDirectory, name, now + 60_000);
    const clock = vi.spyOn(Date, "now").mockImplementation(() => now);
    const classifier = init(config);

    classifier.clearTrainingData();
    const firstFlush = classifier.flush();
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    now += 30_001;
    await expect(firstFlush).rejects.toMatchObject({ code: "storage_failed" });
    now += 30_001;
    await expect(classifier.flush()).resolves.toBeUndefined();

    clock.mockRestore();
    await classifier.close();
  });

  it("archives the trained generation after consecutive failed retests", async () => {
    const dataDirectory = makeDirectory();
    const name = "retest-reset";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init({
      ...booleanConfig(dataDirectory, name),
      retestInterval: 1,
    });
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();
    expect(classifier.isTrained()).toBe(true);

    needle.model.classify.mockResolvedValue(false);
    const reference = vi.fn(async () => true);
    await expect(classifier.classify("first miss", reference)).resolves.toBe(
      true,
    );
    expect(classifier.isTrained()).toBe(true);
    await expect(classifier.classify("second miss", reference)).resolves.toBe(
      true,
    );
    expect(classifier.isTrained()).toBe(false);
    await classifier.close();

    const restarted = init({
      ...booleanConfig(dataDirectory, name),
      retestInterval: 1,
    });
    expect(restarted.isTrained()).toBe(false);
    await expect(
      restarted.classify("reference resumes", reference),
    ).resolves.toBe(true);
    await restarted.close();
  });

  it("keeps the consecutive retest failure count after restart", async () => {
    const dataDirectory = makeDirectory();
    const name = "retest-restart";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const config = {
      ...booleanConfig(dataDirectory, name),
      retestInterval: 1,
    };
    const reference = vi.fn(async () => true);
    const first = init(config);
    first.logClassification(trainingInput, true);
    first.logClassification(heldOutInput, true);
    await first.flush();

    needle.model.classify.mockResolvedValue(false);
    await expect(first.classify("first miss", reference)).resolves.toBe(true);
    expect(first.isTrained()).toBe(true);
    await first.close();

    const second = init(config);
    expect(second.isTrained()).toBe(true);
    await expect(second.classify("second miss", reference)).resolves.toBe(true);
    expect(second.isTrained()).toBe(false);
    await second.close();
  });

  it("keeps a due retest pending during manual candidate-only calls", async () => {
    const dataDirectory = makeDirectory();
    const name = "pending-retest";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init({
      ...booleanConfig(dataDirectory, name),
      retestInterval: 1,
    });
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();

    await expect(classifier.classify("manual")).resolves.toBe(true);
    const reference = vi.fn(async () => true);
    await expect(classifier.classify("managed", reference)).resolves.toBe(true);
    expect(reference).toHaveBeenCalledOnce();
    await classifier.close();
  });

  it("counts a due candidate crash as a failed retest and resets after the limit", async () => {
    const dataDirectory = makeDirectory();
    const name = "crashed-retest";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    const classifier = init({
      ...booleanConfig(dataDirectory, name),
      retestInterval: 1,
    });
    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    await classifier.flush();

    needle.model.classify.mockRejectedValue(new Error("worker stopped"));
    const reference = vi.fn(async () => true);
    await expect(classifier.classify("first crash", reference)).resolves.toBe(
      true,
    );
    expect(classifier.isTrained()).toBe(true);
    await expect(classifier.classify("second crash", reference)).resolves.toBe(
      true,
    );
    expect(classifier.isTrained()).toBe(false);
    await classifier.close();
  });

  it("stops retrying training after maxTrainingSet is reached", async () => {
    const dataDirectory = makeDirectory();
    const name = "training-limit";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    needle.model.classify.mockResolvedValue(false);
    const classifier = init({
      ...booleanConfig(dataDirectory, name),
      maxTrainingSet: 4,
    });

    classifier.logClassification(trainingInput, true);
    classifier.logClassification(heldOutInput, true);
    classifier.logClassification(`${trainingInput}-burst`, true);
    classifier.logClassification(`${heldOutInput}-burst`, true);
    await classifier.flush();
    expect(needle.runtime.train).toHaveBeenCalledOnce();
    expect(classifier.isTrained()).toBe(false);

    classifier.logClassification(`${trainingInput}-new`, true);
    classifier.logClassification(`${heldOutInput}-new`, true);
    await classifier.flush();
    expect(needle.runtime.train).toHaveBeenCalledOnce();
    await classifier.close();
  });

  it("returns the reference result without waiting for background training", async () => {
    const dataDirectory = makeDirectory();
    const name = "non-blocking-training";
    const [trainingInput, heldOutInput] = inputsForBothSplits(name);
    let finishTraining!: () => void;
    const trainingGate = new Promise<void>((resolve) => {
      finishTraining = resolve;
    });
    needle.runtime.train.mockImplementationOnce(async () => {
      await trainingGate;
      return {
        modelPath: "/tmp/swapai-candidate.cact",
        needleVersion: "2.0.14",
      };
    });
    const classifier = init(booleanConfig(dataDirectory, name));
    const reference = vi.fn(async () => true);

    await classifier.classify(trainingInput, reference);
    let returned = false;
    const second = classifier.classify(heldOutInput, reference).then((value) => {
      returned = true;
      return value;
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    const returnedBeforeTrainingFinished = returned;
    finishTraining();
    await expect(second).resolves.toBe(true);
    await classifier.flush();

    expect(returnedBeforeTrainingFinished).toBe(true);
    await classifier.close();
  });
});
