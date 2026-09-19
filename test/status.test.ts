import { mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { assignExampleSplit, openStorage } from "../src/storage.js";
import { readClassifierStatuses } from "../src/classifiers-ui.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

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

  it("inspects legacy examples before a classifier opens and migrates them", () => {
    const dataDirectory = makeDirectory();
    const name = "legacy-accountant-relevance";
    const storage = openStorage({
      dataDirectory,
      name,
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
    let input = "";
    for (let index = 0; input === ""; index += 1) {
      const candidate = `legacy-invoice-${index}`;
      if (assignExampleSplit(name, candidate) === "training") input = candidate;
    }
    storage.addExample(input, 0.92);
    storage.close();

    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.exec("ALTER TABLE examples DROP COLUMN result_bin");
    database.exec("ALTER TABLE examples DROP COLUMN purpose");
    database.exec("ALTER TABLE examples DROP COLUMN facets_json");
    database.close();

    const [status] = readClassifierStatuses({ dataDirectory });
    expect(status).toMatchObject({
      name,
      retainedExamples: 1,
      examplesByPurpose: {
        training: 1,
        validation: 0,
        representative_test: 0,
        coverage_test: 0,
      },
    });
    expect(
      status!.resultBins.reduce((total, bin) => total + bin.total, 0),
    ).toBe(1);
  });
});
