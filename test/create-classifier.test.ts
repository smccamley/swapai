import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

import { afterEach, describe, expect, it, vi } from "vitest";

import { createClassifier, init } from "../src/index.js";
import type { TrainingJob, TrainingProvider } from "../src/index.js";

const temporaryDirectories: string[] = [];
const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

const makeDirectory = (): string => {
  const directory = mkdtempSync(join(tmpdir(), "swapai-simple-classifier-"));
  temporaryDirectories.push(directory);
  return directory;
};

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("createClassifier", () => {
  it("migrates the exact database schema published by 0.3.0", async () => {
    const dataDirectory = makeDirectory();
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.exec(`
      CREATE TABLE classifiers (
        name TEXT PRIMARY KEY, config_json TEXT NOT NULL,
        max_training_set INTEGER NOT NULL, active_generation INTEGER NOT NULL DEFAULT 1,
        total_examples_logged INTEGER NOT NULL DEFAULT 0,
        new_examples_since_training INTEGER NOT NULL DEFAULT 0,
        training_attempts INTEGER NOT NULL DEFAULT 0,
        examples_used_for_training INTEGER NOT NULL DEFAULT 0,
        last_evaluated_error REAL, last_evaluated_at INTEGER,
        local_classifications_since_retest INTEGER NOT NULL DEFAULT 0,
        total_local_classifications INTEGER NOT NULL DEFAULT 0,
        total_retests INTEGER NOT NULL DEFAULT 0,
        consecutive_retest_failures INTEGER NOT NULL DEFAULT 0,
        training_lease_owner TEXT, training_lease_until INTEGER,
        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
        data_epoch INTEGER NOT NULL DEFAULT 0,
        clear_pending INTEGER NOT NULL DEFAULT 0,
        clear_erased INTEGER NOT NULL DEFAULT 0,
        clear_artifact_generation_max INTEGER,
        training_lease_epoch INTEGER
      );
      CREATE TABLE generations (
        classifier_name TEXT NOT NULL, generation INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
        trained INTEGER NOT NULL DEFAULT 0, model_path TEXT, needle_version TEXT,
        created_at INTEGER NOT NULL, archived_at INTEGER,
        PRIMARY KEY (classifier_name, generation),
        FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
      );
      CREATE TABLE examples (
        id INTEGER PRIMARY KEY AUTOINCREMENT, classifier_name TEXT NOT NULL,
        generation INTEGER NOT NULL, input TEXT NOT NULL, result_json TEXT NOT NULL,
        split TEXT NOT NULL CHECK (split IN ('training', 'held_out')),
        created_at INTEGER NOT NULL,
        FOREIGN KEY (classifier_name, generation)
          REFERENCES generations(classifier_name, generation) ON DELETE CASCADE
      );
    `);
    const now = Date.now();
    const config = JSON.stringify({
      result: { type: "number", min: 0, max: 1 },
      retrainOnCount: 50,
      acceptableError: 0.1,
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2",
    });
    database.prepare(`
      INSERT INTO classifiers (
        name, config_json, max_training_set, total_examples_logged,
        new_examples_since_training, created_at, updated_at
      ) VALUES (?, ?, 10000, 1, 1, ?, ?)
    `).run("accountant-relevance", config, now, now);
    database.prepare(`
      INSERT INTO generations (classifier_name, generation, status, created_at)
      VALUES (?, 1, 'active', ?)
    `).run("accountant-relevance", now);
    database.prepare(`
      INSERT INTO examples (
        classifier_name, generation, input, result_json, split, created_at
      ) VALUES (?, 1, ?, ?, 'training', ?)
    `).run("accountant-relevance", "historic invoice", "0.92", now);
    database.close();

    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      dataDirectory,
    });

    expect(relevance.inspect().retainedExamples).toBe(1);
    await relevance.close();
  });

  it("opens data created by the 0.3 application configuration", async () => {
    const dataDirectory = makeDirectory();
    const legacy = init({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      retrainOnCount: 50,
      acceptableError: "10%",
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2",
      maxTrainingSet: 10_000,
      automaticTraining: false,
      dataDirectory,
    });
    legacy.logClassification("historic relevant invoice", 0.92);
    legacy.logClassification("historic irrelevant newsletter", 0.08);
    legacy.logClassification("historic boundary high", 0.51);
    legacy.logClassification("historic boundary low", 0.49);
    await legacy.close();

    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      dataDirectory,
    });

    const inspection = relevance.inspect();
    expect(
      inspection.examplesByPurpose.training +
        inspection.examplesByPurpose.validation,
    ).toBe(4);
    expect(inspection.examplesByPurpose.representative_test).toBe(0);
    expect(inspection.examplesByPurpose.coverage_test).toBe(0);
    expect(
      inspection.resultBins.reduce((total, bin) => total + bin.total, 0),
    ).toBe(4);

    await expect(relevance.close()).resolves.toBeUndefined();
  });

  it("needs only a classifier definition and reference function", async () => {
    const reference = vi.fn(async (input: string) => input.includes("invoice"));
    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "boolean" },
      reference,
      facets: ["documentFamily"] as const,
      dataDirectory: makeDirectory(),
    });

    await expect(
      relevance.classify("Supplier invoice", { documentFamily: "invoice" }),
    ).resolves.toBe(true);
    await relevance.flush();

    expect(reference).toHaveBeenCalledWith("Supplier invoice");
    expect(relevance.isTrained()).toBe(false);
    await relevance.close();
  });

  it("accepts decision boundaries and declared facet names without training options", async () => {
    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.92,
      decisionBoundaries: [0.5],
      facets: ["documentFamily", "sourceKind"] as const,
      dataDirectory: makeDirectory(),
    });

    await expect(
      relevance.classify("Supplier invoice", {
        documentFamily: "invoice",
        sourceKind: "gmail",
      }),
    ).resolves.toBe(0.92);
    await relevance.close();
  });

  it("explains missing range and test coverage instead of reporting only a total", async () => {
    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async (input) => (input.startsWith("relevant") ? 0.92 : 0.08),
      decisionBoundaries: [0.5],
      facets: ["documentFamily"] as const,
      dataDirectory: makeDirectory(),
    });

    for (let index = 0; index < 100; index += 1) {
      await relevance.classify(`irrelevant-${index}`, {
        documentFamily: "newsletter",
      });
    }
    for (let index = 0; index < 20; index += 1) {
      await relevance.classify(`relevant-${index}`, {
        documentFamily: "invoice",
      });
    }
    await relevance.flush();

    const inspection = relevance.inspect();
    expect(inspection).toMatchObject({
      totalExamplesLogged: 120,
      readyForTraining: false,
    });
    expect(inspection.resultBins).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          range: { minimum: 0, maximum: 0.2, includesMaximum: false },
          total: expect.any(Number),
        }),
        expect.objectContaining({
          range: { minimum: 0.4, maximum: 0.5, includesMaximum: false },
          total: 0,
        }),
        expect.objectContaining({
          range: { minimum: 0.5, maximum: 0.6, includesMaximum: false },
          total: 0,
        }),
        expect.objectContaining({
          range: { minimum: 0.8, maximum: 1, includesMaximum: true },
          total: expect.any(Number),
        }),
      ]),
    );
    expect(inspection.deficits).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          purpose: "training",
          resultBin: "0.4..0.5",
        }),
        expect.objectContaining({
          purpose: "coverage_test",
          resultBin: "0.5..0.6",
        }),
      ]),
    );
    await relevance.close();
  });

  it("keeps rare result coverage when the retained dataset reaches its limit", async () => {
    const relevance = createClassifier({
      name: "bounded-relevance",
      result: { type: "boolean" },
      reference: async () => false,
      maxTrainingSet: 8,
      dataDirectory: makeDirectory(),
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });

    relevance.logClassification("rare-positive", true);
    for (let index = 0; index < 30; index += 1) {
      relevance.logClassification(`common-negative-${index}`, false);
    }
    await relevance.flush();

    const inspection = relevance.inspect();
    expect(inspection.retainedExamples).toBe(8);
    expect(inspection.resultBins.find((bin) => bin.id === "true")?.total).toBe(
      1,
    );
    await relevance.close();
  });

  it("keeps declared facet groups when the retained dataset reaches its limit", async () => {
    const relevance = createClassifier({
      name: "bounded-facet-coverage",
      result: { type: "boolean" },
      reference: async () => false,
      facets: ["documentFamily"] as const,
      maxTrainingSet: 8,
      dataDirectory: makeDirectory(),
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });

    for (const documentFamily of ["invoice", "newsletter"] as const) {
      for (let index = 0; index < 30; index += 1) {
        relevance.logClassification(`${documentFamily}-${index}`, false, {
          documentFamily,
        });
      }
    }
    await relevance.flush();

    const coverage = relevance.inspect().facetCoverage;
    expect(coverage.find((group) => group.value === "invoice")?.total)
      .toBeGreaterThan(0);
    expect(coverage.find((group) => group.value === "newsletter")?.total)
      .toBeGreaterThan(0);
    await relevance.close();
  });

  it("counts repeated observations once in the retained dataset", async () => {
    const relevance = createClassifier({
      name: "deduplicated-relevance",
      result: { type: "boolean" },
      reference: async () => true,
      dataDirectory: makeDirectory(),
    });

    for (let index = 0; index < 10; index += 1) {
      relevance.logClassification("same input", true);
    }
    await relevance.flush();

    expect(relevance.inspect()).toMatchObject({
      totalExamplesLogged: 10,
      retainedExamples: 1,
    });
    await relevance.close();
  });

  it("keeps an input in its original dataset purpose when its result changes", async () => {
    const relevance = createClassifier({
      name: "stable-purpose",
      result: { type: "boolean" },
      reference: async () => false,
      dataDirectory: makeDirectory(),
    });

    relevance.logClassification("same-0", false);
    await relevance.flush();
    const before = relevance.inspect().examplesByPurpose;

    relevance.logClassification("same-0", true);
    await relevance.flush();

    expect(relevance.inspect().examplesByPurpose).toEqual(before);
    await relevance.close();
  });

  it("shows declared facet coverage including unlabelled examples", async () => {
    const relevance = createClassifier({
      name: "facet-coverage",
      result: { type: "boolean" },
      reference: async () => true,
      facets: ["documentFamily"] as const,
      dataDirectory: makeDirectory(),
    });
    relevance.logClassification("invoice", true, { documentFamily: "invoice" });
    relevance.logClassification("newsletter", false, {
      documentFamily: "newsletter",
    });
    relevance.logClassification("unknown", false);
    await relevance.flush();

    expect(relevance.inspect().facetCoverage).toEqual(expect.arrayContaining([
      expect.objectContaining({ facet: "documentFamily", value: null, total: 1 }),
      expect.objectContaining({ facet: "documentFamily", value: "invoice", total: 1 }),
      expect.objectContaining({ facet: "documentFamily", value: "newsletter", total: 1 }),
    ]));
    await relevance.close();
  });

  it("hands only frozen training examples to a trainer", async () => {
    const receivedJobs: TrainingJob[] = [];
    const expectedFailure = new Error("stop after inspecting the handoff");
    const training: TrainingProvider = {
      name: "recording-trainer",
      train: async (job, lifecycle) => {
        receivedJobs.push(job);
        lifecycle?.recordProviderRun({
          providerRunId: "provider-run-failed",
          resources: [{ type: "test-machine", id: "machine-1" }],
        });
        lifecycle?.recordCleanup({
          status: "failed",
          message: "machine did not terminate",
        });
        throw expectedFailure;
      },
    };
    const relevance = createClassifier({
      name: "provider-boundary",
      result: { type: "boolean" },
      reference: async (input) => input.includes("yes"),
      training,
      dataDirectory: makeDirectory(),
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });

    for (let index = 0; index < 200; index += 1) {
      await relevance.classify(`${index % 2 === 0 ? "yes" : "no"}-${index}`);
    }
    await relevance.flush();

    await expect(relevance.requestTraining()).rejects.toBe(expectedFailure);
    expect(receivedJobs).toHaveLength(1);
    expect(receivedJobs[0]).toMatchObject({
      classifierName: "provider-boundary",
      result: { type: "boolean" },
      datasetRevisionId: expect.stringMatching(/^[a-f0-9]{64}$/),
    });
    expect(receivedJobs[0]!.examples.length).toBeGreaterThan(0);
    expect(
      receivedJobs[0]!.examples.every(
        (example) =>
          example.purpose === "training",
      ),
    ).toBe(true);
    expect(
      receivedJobs[0]!.examples.some(
        (example) => example.purpose === "training",
      ),
    ).toBe(true);
    expect(relevance.inspect().latestTrainingRun).toMatchObject({
      id: receivedJobs[0]!.id,
      provider: "recording-trainer",
      status: "failed",
      datasetRevisionId: receivedJobs[0]!.datasetRevisionId,
      failureMessage: "stop after inspecting the handoff",
      providerRunId: "provider-run-failed",
      resources: [{ type: "test-machine", id: "machine-1" }],
      cleanup: {
        status: "failed",
        message: "machine did not terminate",
      },
    });
    await relevance.close();
  });
});
