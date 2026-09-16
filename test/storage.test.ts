import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  assignExampleSplit,
  openStorage,
  type Storage,
} from "../src/storage.js";

const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite") as typeof import("node:sqlite");

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

  it("deletes all retained generations and counters for only one classifier", () => {
    const dataDirectory = makeDirectory();
    const removed = openTestStorage(dataDirectory, { name: "removed" });
    const preserved = openTestStorage(dataDirectory, { name: "preserved" });
    removed.addExample("private input", true);
    removed.promoteGeneration({
      modelPath: "/models/private.cact",
      needleVersion: "2.4.0",
    });
    removed.archiveAndReset();
    removed.addExample("new private input", false);
    removed.recordLocalClassification();
    preserved.addExample("keep this", true);

    const emptyGeneration = removed.clearTrainingData();

    expect(emptyGeneration).toMatchObject({
      status: "active",
      trained: false,
      modelPath: null,
      needleVersion: null,
    });
    expect(removed.listGenerations()).toEqual([
      expect.objectContaining({ generation: emptyGeneration.generation }),
    ]);
    expect(removed.listExamples()).toEqual([]);
    expect(removed.snapshot()).toMatchObject({
      activeExampleCount: 0,
      totalExamplesLogged: 0,
      newExamplesSinceTraining: 0,
      trainingAttempts: 0,
      examplesUsedForTraining: 0,
      localClassificationsSinceRetest: 0,
      totalLocalClassifications: 0,
      totalRetests: 0,
      consecutiveRetestFailures: 0,
      trained: false,
      modelPath: null,
      needleVersion: null,
    });
    expect(preserved.listExamples()).toEqual([
      expect.objectContaining({ input: "keep this", result: true }),
    ]);
    removed.close();
    preserved.close();
  });

  it("overwrites deleted example bytes and truncates the SQLite WAL", () => {
    const dataDirectory = makeDirectory();
    const storage = openTestStorage(dataDirectory, { name: "secure-erasure" });
    const privateMarker = "SWAPAI_PRIVATE_ERASURE_MARKER_7f4c9a";
    storage.addExample(`${privateMarker}:${"private".repeat(1_000)}`, true);

    storage.clearTrainingData();

    const databaseFiles = [
      storage.databasePath,
      `${storage.databasePath}-wal`,
      `${storage.databasePath}-shm`,
    ].filter(existsSync);
    for (const path of databaseFiles) {
      expect(readFileSync(path).includes(Buffer.from(privateMarker))).toBe(false);
    }
    storage.close();
  });

  it("reports when another database reader prevents WAL truncation", () => {
    const storage = openTestStorage(makeDirectory(), {
      name: "checkpoint-failure",
    });
    storage.addExample("private input", true);
    const reader = new DatabaseSync(storage.databasePath);
    reader.exec("BEGIN");
    reader.prepare("SELECT * FROM examples").get();

    expect(() => storage.clearTrainingData()).toThrow(/WAL could not be truncated/i);
    expect(storage.snapshot().clearPending).toBe(true);
    expect(
      storage.addExample("must remain blocked", false, storage.snapshot().dataEpoch),
    ).toBeNull();

    reader.exec("ROLLBACK");
    reader.close();
    expect(() => storage.clearTrainingData()).not.toThrow();
    storage.close();
  });

  it("migrates a database created before persisted erasure epochs", () => {
    const dataDirectory = makeDirectory();
    const original = openTestStorage(dataDirectory, { name: "old-database" });
    original.close();
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.exec("ALTER TABLE classifiers DROP COLUMN clear_pending");
    database.exec("ALTER TABLE classifiers DROP COLUMN data_epoch");
    database.exec("ALTER TABLE classifiers DROP COLUMN clear_erased");
    database.exec(
      "ALTER TABLE classifiers DROP COLUMN clear_artifact_generation_max",
    );
    database.exec("ALTER TABLE classifiers DROP COLUMN training_lease_epoch");
    database.close();

    const migrated = openTestStorage(dataDirectory, { name: "old-database" });

    expect(migrated.snapshot()).toMatchObject({
      dataEpoch: 0,
      clearPending: false,
    });
    migrated.close();
  });

  it("rejects writes from a live pre-0.2 database connection", () => {
    const dataDirectory = makeDirectory();
    const storage = openTestStorage(dataDirectory, { name: "legacy-writer" });
    const legacy = new DatabaseSync(storage.databasePath);

    expect(() =>
      legacy.prepare(`
        INSERT INTO examples (
          classifier_name, generation, input, result_json, split, created_at
        ) VALUES (?, 1, ?, ?, 'training', ?)
      `).run("legacy-writer", "private legacy input", "true", Date.now()),
    ).toThrow(/swapai_writer_version|too old/i);

    legacy.close();
    storage.close();
  });

  it("rejects every old-epoch mutation from another live storage instance", () => {
    const dataDirectory = makeDirectory();
    const first = openTestStorage(dataDirectory, { name: "shared-epoch" });
    const second = openTestStorage(dataDirectory, { name: "shared-epoch" });
    first.addExample("old input", true);
    const oldEpoch = first.snapshot().dataEpoch;

    const clearEpoch = second.beginClearTrainingData();

    expect(clearEpoch).toBe(oldEpoch + 1);
    expect(first.addExample("must not return", false, oldEpoch)).toBeNull();
    expect(first.markTrainingAttempted(1, oldEpoch)).toBe(false);
    expect(
      first.listExamplesForTraining("training", 1, oldEpoch),
    ).toBeNull();
    expect(
      first.promoteGeneration(
        { modelPath: "/models/stale.cact", needleVersion: "2.4.0" },
        oldEpoch,
      ),
    ).toBeNull();
    expect(first.archiveAndReset(oldEpoch)).toBeNull();
    expect(first.recordLocalClassification(oldEpoch)).toBeNull();
    expect(first.recordRetest(false, oldEpoch)).toBeNull();
    expect(first.snapshot()).toMatchObject({
      dataEpoch: clearEpoch,
      clearPending: true,
      totalExamplesLogged: 1,
      trainingAttempts: 0,
      totalLocalClassifications: 0,
      totalRetests: 0,
      trained: false,
    });

    expect(second.eraseTrainingData(clearEpoch)).not.toBeNull();
    expect(second.finishClearTrainingData(clearEpoch)).toBe(true);
    first.close();
    second.close();
  });

  it("releases the artifact write barrier when a training process exits", () => {
    const dataDirectory = makeDirectory();
    const trainingProcess = openTestStorage(dataDirectory, {
      name: "crashed-training",
    });
    const clearingProcess = openTestStorage(dataDirectory, {
      name: "crashed-training",
    });
    expect(trainingProcess.claimArtifactWriteLock()).toBe(true);
    const clearEpoch = clearingProcess.beginClearTrainingData();
    expect(clearingProcess.claimArtifactWriteLock()).toBe(false);

    trainingProcess.close();

    expect(clearingProcess.claimArtifactWriteLock()).toBe(true);
    expect(clearingProcess.eraseTrainingData(clearEpoch)).not.toBeNull();
    expect(clearingProcess.finishClearTrainingData(clearEpoch)).toBe(true);
    clearingProcess.releaseArtifactWriteLock();
    clearingProcess.close();
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

  it("closes database handles when reopening with incompatible config", () => {
    const dataDirectory = makeDirectory();
    const original = openTestStorage(dataDirectory);
    original.close();
    const close = vi.spyOn(DatabaseSync.prototype, "close");
    try {
      expect(() => openTestStorage(dataDirectory, {
        config: {
          result: { type: "boolean" },
          acceptableError: 0.1,
          retrainOnCount: 50,
          retestInterval: 100,
          retestRevertOn: 3,
        },
      })).toThrowError(TypeError);
      expect(close).toHaveBeenCalledTimes(2);
    } finally {
      close.mockRestore();
    }
  });

  it("closes database handles when its initialization transaction fails", () => {
    const dataDirectory = makeDirectory();
    const original = openTestStorage(dataDirectory);
    original.close();
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.exec(`
      CREATE TRIGGER reject_storage_init
      BEFORE UPDATE ON classifiers
      BEGIN SELECT RAISE(ABORT, 'forced init failure'); END
    `);
    database.close();
    const close = vi.spyOn(DatabaseSync.prototype, "close");
    try {
      expect(() => openTestStorage(dataDirectory)).toThrow(/forced init failure/i);
      expect(close).toHaveBeenCalledTimes(2);
    } finally {
      close.mockRestore();
    }
  });
});
