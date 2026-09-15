import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  assignExampleSplit,
  openStorage,
  type Storage,
} from "../src/storage.js";

const temporaryDirectories: string[] = [];

function makeDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "swapai-storage-"));
  temporaryDirectories.push(directory);
  return directory;
}

function openTestStorage(
  dataDirectory: string,
  overrides: Partial<Parameters<typeof openStorage>[0]> = {},
): Storage {
  return openStorage({
    dataDirectory,
    name: "accountant-relevance",
    config: {
      result: { type: "number", min: 0, max: 1 },
      acceptableError: 0.1,
      retrainOnCount: 50,
      retestInterval: 100,
      retestRevertOn: 3,
    },
    maxTrainingSet: 10_000,
    ...overrides,
  });
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("SQLite storage", () => {
  it("persists classifier state, examples, models, and counters across restarts", () => {
    const dataDirectory = makeDirectory();
    const first = openTestStorage(dataDirectory);

    first.addExample("I owe you £5", 0.92);
    first.addExample("Paid rent", 0.81);
    first.markTrainingAttempted(2);
    first.promoteGeneration({
      modelPath: "/models/accountant-relevance-1.cact",
      needleVersion: "2.4.0",
    });
    expect(first.recordLocalClassification()).toBe(1);
    expect(first.recordLocalClassification()).toBe(2);
    expect(first.recordRetest(false)).toBe(1);
    expect(first.recordRetest(false)).toBe(2);
    first.close();

    const second = openTestStorage(dataDirectory);
    expect(second.snapshot()).toMatchObject({
      name: "accountant-relevance",
      config: {
        result: { type: "number", min: 0, max: 1 },
        acceptableError: 0.1,
      },
      activeGeneration: 1,
      activeExampleCount: 2,
      totalExamplesLogged: 2,
      newExamplesSinceTraining: 0,
      trainingAttempts: 1,
      examplesUsedForTraining: 2,
      localClassificationsSinceRetest: 0,
      totalLocalClassifications: 2,
      totalRetests: 2,
      consecutiveRetestFailures: 2,
      trained: true,
      modelPath: "/models/accountant-relevance-1.cact",
      needleVersion: "2.4.0",
    });
    expect(second.listExamples().map(({ input, result }) => ({ input, result }))).toEqual([
      { input: "I owe you £5", result: 0.92 },
      { input: "Paid rent", result: 0.81 },
    ]);

    second.close();
  });

  it("always assigns repeated inputs to the same training or held-out set", () => {
    const name = "currency";
    const input = "Paid in GBP";
    const expectedSplit = assignExampleSplit(name, input);
    const dataDirectory = makeDirectory();
    const storage = openTestStorage(dataDirectory, { name });

    storage.addExample(input, "gbp");
    storage.addExample(input, "usd");

    expect(storage.listExamples()).toEqual([
      expect.objectContaining({ input, split: expectedSplit }),
      expect.objectContaining({ input, split: expectedSplit }),
    ]);
    storage.close();
  });

  it("keeps the newest active examples when the training set limit is reached", () => {
    const storage = openTestStorage(makeDirectory(), { maxTrainingSet: 3 });

    storage.addExample("one", true);
    storage.addExample("two", false);
    storage.addExample("three", true);
    storage.addExample("four", false);

    expect(storage.listExamples().map((example) => example.input)).toEqual([
      "two",
      "three",
      "four",
    ]);
    expect(storage.snapshot()).toMatchObject({
      activeExampleCount: 3,
      totalExamplesLogged: 4,
      newExamplesSinceTraining: 4,
    });
    storage.close();
  });

  it("archives the trained generation and starts with an empty active generation", () => {
    const dataDirectory = makeDirectory();
    const storage = openTestStorage(dataDirectory);

    storage.addExample("old example", 0.8);
    storage.promoteGeneration({
      modelPath: "/models/old.cact",
      needleVersion: "2.4.0",
    });
    storage.recordLocalClassification();
    storage.recordRetest(false);

    const nextGeneration = storage.archiveAndReset();

    expect(nextGeneration).toMatchObject({
      generation: 2,
      status: "active",
      trained: false,
      modelPath: null,
      needleVersion: null,
    });
    expect(storage.snapshot()).toMatchObject({
      activeGeneration: 2,
      activeExampleCount: 0,
      totalExamplesLogged: 1,
      newExamplesSinceTraining: 0,
      examplesUsedForTraining: 0,
      localClassificationsSinceRetest: 0,
      consecutiveRetestFailures: 0,
      trained: false,
      modelPath: null,
      needleVersion: null,
    });
    expect(storage.listGenerations()).toEqual([
      expect.objectContaining({
        generation: 1,
        status: "archived",
        trained: true,
        modelPath: "/models/old.cact",
      }),
      expect.objectContaining({
        generation: 2,
        status: "active",
        trained: false,
      }),
    ]);
    expect(storage.listExamples(undefined, 1)).toEqual([
      expect.objectContaining({ input: "old example", generation: 1 }),
    ]);
    expect(storage.listExamples()).toEqual([]);

    storage.close();
  });

  it("allows only one process to hold the training lease", () => {
    const dataDirectory = makeDirectory();
    const first = openTestStorage(dataDirectory);
    const second = openTestStorage(dataDirectory);

    expect(first.claimTrainingLease("process-one", 60_000)).toBe(true);
    expect(second.claimTrainingLease("process-two", 60_000)).toBe(false);
    first.releaseTrainingLease("process-one");
    expect(second.claimTrainingLease("process-two", 60_000)).toBe(true);

    second.releaseTrainingLease("process-two");
    first.close();
    second.close();
  });

  it.each([
    ["result", { type: "boolean" }],
    ["retrainOnCount", 25],
    ["acceptableError", 0.2],
    ["retestInterval", 50],
    ["retestRevertOn", 5],
    ["model", "another-model"],
  ])("rejects reopening with a different %s", (key, value) => {
    const dataDirectory = makeDirectory();
    const originalConfig = {
      result: { type: "number", min: 0, max: 1 },
      acceptableError: "10%",
      retrainOnCount: 50,
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2",
    };
    const first = openTestStorage(dataDirectory, { config: originalConfig });
    first.addExample("existing example", 0.9);
    first.promoteGeneration({
      modelPath: "/models/existing.cact",
      needleVersion: "2.4.0",
    });
    first.close();

    expect(() => openTestStorage(dataDirectory, {
      config: { ...originalConfig, [key]: value },
    })).toThrowError(TypeError);

    const unchanged = openTestStorage(dataDirectory, {
      config: { ...originalConfig, acceptableError: 0.1 },
      maxTrainingSet: 1,
    });
    expect(unchanged.snapshot()).toMatchObject({
      activeExampleCount: 1,
      trained: true,
      modelPath: "/models/existing.cact",
    });
    unchanged.close();
  });
});
