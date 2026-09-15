import { mkdtempSync, rmSync } from "node:fs";
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
import { assignExampleSplit } from "../src/storage.js";

const needle = vi.hoisted(() => {
  const model = {
    classify: vi.fn<(input: string) => Promise<number | boolean | string>>(),
    close: vi.fn(async () => undefined),
  };
  const runtime = {
    ready: vi.fn(async () => undefined),
    train: vi.fn(async () => ({
      modelPath: "/tmp/swapai-candidate.cact",
      needleVersion: "2.0.14",
    })),
    loadModel: vi.fn(async () => model),
    close: vi.fn(async () => undefined),
  };
  return { model, runtime };
});

vi.mock("../src/runtime.js", () => ({
  NEEDLE_VERSION: "2.0.14",
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

beforeEach(() => {
  vi.clearAllMocks();
  needle.model.classify.mockResolvedValue(true);
});

afterEach(() => {
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
