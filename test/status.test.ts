import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { openStorage } from "../src/storage.js";
import { readClassifierStatuses } from "../src/classifiers-ui.js";

const temporaryDirectories: string[] = [];

function makeDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "swapai-status-"));
  temporaryDirectories.push(directory);
  return directory;
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("classifier status", () => {
  it("returns no classifiers before SwapAI has created its database", () => {
    expect(readClassifierStatuses({ dataDirectory: makeDirectory() })).toEqual([]);
  });

  it("reports collected data, error target, training, loaded, and trained state", () => {
    const dataDirectory = makeDirectory();
    const storage = openStorage({
      dataDirectory,
      name: "accountant-relevance",
      config: {
        result: { type: "number", min: 0, max: 1 },
        acceptableError: 0.1,
        retrainOnCount: 50,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
      maxTrainingSet: 10_000,
    });
    storage.addExample("I owe you £5", 0.92);
    storage.markTrainingAttempted(1);
    storage.recordEvaluation(0.08);
    storage.promoteGeneration({ modelPath: "/tmp/model.cact", needleVersion: "2.0.14" });
    storage.registerRuntime("1234:runtime", 1234);
    storage.claimTrainingLease("1234:runtime", 60_000);

    expect(readClassifierStatuses({ dataDirectory })).toEqual([
      expect.objectContaining({
        name: "accountant-relevance",
        resultType: "number",
        retainedExamples: 1,
        totalExamplesLogged: 1,
        lastEvaluatedError: 0.08,
        acceptableError: 0.1,
        loaded: true,
        loadedProcessCount: 1,
        training: true,
        trained: true,
        needleVersion: "2.0.14",
      }),
    ]);
    storage.close();
  });
});
