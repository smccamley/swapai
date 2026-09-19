import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createClassifier,
  init,
  inspectLegacyHeldOutMigration,
  migrateLegacyHeldOutExamples,
} from "../src/index.js";
import { createDatasetPolicy } from "../src/dataset-policy.js";
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

const zeroBinMinimums = {
  minimumTrainingExamples: 1,
  minimumTrainingExamplesPerResultBin: 0,
  minimumValidationExamplesPerResultBin: 0,
  minimumRepresentativeTestExamples: 0,
  minimumCoverageTestExamplesPerResultBin: 0,
} as const;

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("createClassifier", () => {
  it("opens data collected by the legacy API through the configured operator API", async () => {
    const dataDirectory = makeDirectory();
    const name = "accountant-relevance:source-1";
    const result = { type: "number" as const, min: 0, max: 1 };
    const datasetRequirements = {
      minimumTrainingExamples: 1,
      minimumTrainingExamplesPerResultBin: 0,
      minimumValidationExamplesPerResultBin: 0,
      minimumRepresentativeTestExamples: 0,
      minimumCoverageTestExamplesPerResultBin: 0,
    };
    const datasetPolicy = createDatasetPolicy(result, {
      decisionBoundaries: [0.5],
      facets: ["documentFamily"],
      requirements: datasetRequirements,
    });
    const runtime = init({
      name,
      result,
      retrainOnCount: 50,
      acceptableError: "10%",
      retestInterval: 100,
      retestRevertOn: 3,
      model: "needle2",
      maxTrainingSet: 10_000,
      automaticTraining: false,
      datasetPolicy,
      dataDirectory,
    });
    runtime.logClassification("invoice", 0.9, {
      documentFamily: "invoice",
    });
    await runtime.flush();
    await runtime.close();

    const operator = createClassifier({
      name,
      result,
      reference: async () => 0.9,
      decisionBoundaries: [0.5],
      facets: ["documentFamily"] as const,
      acceptableError: "10%",
      maxTrainingSet: 10_000,
      datasetRequirements,
      dataDirectory,
    });
    expect(operator.inspect()).toMatchObject({
      name,
      totalExamplesLogged: 1,
      retainedExamples: 1,
    });
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"), {
      readOnly: true,
    });
    try {
      expect(
        database
          .prepare(
            `
        SELECT COUNT(*) AS count FROM classifiers WHERE name = ?
      `,
          )
          .get(name),
      ).toEqual({ count: 1 });
    } finally {
      database.close();
    }
    await operator.close();
  });

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
    database
      .prepare(
        `
      INSERT INTO classifiers (
        name, config_json, max_training_set, total_examples_logged,
        new_examples_since_training, created_at, updated_at
      ) VALUES (?, ?, 10000, 1, 1, ?, ?)
    `,
      )
      .run("accountant-relevance", config, now, now);
    database
      .prepare(
        `
      INSERT INTO generations (classifier_name, generation, status, created_at)
      VALUES (?, 1, 'active', ?)
    `,
      )
      .run("accountant-relevance", now);
    database
      .prepare(
        `
      INSERT INTO examples (
        classifier_name, generation, input, result_json, split, created_at
      ) VALUES (?, 1, ?, ?, 'training', ?)
    `,
      )
      .run("accountant-relevance", "historic invoice", "0.92", now);
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
    for (let index = 0; index < 1_319; index += 1) {
      legacy.logClassification(
        `historic-example-${index}`,
        index % 2 === 0 ? 0.92 : 0.08,
      );
    }
    await legacy.close();
    mimicVersion050LegacyAdoption(dataDirectory, "accountant-relevance");

    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      dataDirectory,
    });

    const inspection = relevance.inspect();
    expect(inspection.examplesByPurpose).toEqual({
      training: 819,
      validation: 197,
      representative_test: 141,
      coverage_test: 162,
    });
    expect(
      inspection.resultBins.reduce((total, bin) => total + bin.total, 0),
    ).toBe(1_319);

    await expect(relevance.close()).resolves.toBeUndefined();

    const reopened = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      dataDirectory,
    });
    expect(reopened.inspect().examplesByPurpose).toEqual(
      inspection.examplesByPurpose,
    );
    await reopened.close();
  }, 15_000);

  it("turns never-evaluated legacy held-out rows into protected suites without moving training rows", async () => {
    const dataDirectory = makeDirectory();
    const name = "production-legacy-counters";
    const legacy = init({
      name,
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
    for (let index = 0; index < 1_319; index += 1) {
      legacy.logClassification(
        `historic-example-${index}`,
        index % 2 === 0 ? 0.92 : 0.08,
      );
    }
    await legacy.close();

    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.function("swapai_writer_version", { deterministic: true }, () => 3);
    database
      .prepare(
        `
        UPDATE examples
        SET split = CASE WHEN id <= 1049 THEN 'training' ELSE 'held_out' END
        WHERE classifier_name = ?
      `,
      )
      .run(name);
    database
      .prepare(
        `
        UPDATE classifiers
        SET training_attempts = 24,
            examples_used_for_training = 1309
        WHERE name = ?
      `,
      )
      .run(name);
    database.close();
    mimicVersion050LegacyAdoption(dataDirectory, name);

    const firstOpen = createClassifier({
      name,
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      datasetRequirements: {
        minimumTrainingExamples: 800,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 100,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
      dataDirectory,
    });
    expect(firstOpen.inspect().examplesByPurpose).toMatchObject({
      training: 1_049,
      validation: 270,
      representative_test: 0,
      coverage_test: 0,
    });
    await firstOpen.close();
    const originalTrainingRowsSha256 = trainingRowsSha256(
      dataDirectory,
      name,
    );

    const classifierDirectory = join(
      dataDirectory,
      "classifiers",
      createHash("sha256").update(name).digest("hex"),
    );
    mkdirSync(join(classifierDirectory, "generation-1", "candidates", "failed"), {
      recursive: true,
    });
    mkdirSync(join(classifierDirectory, "generation-1", "checkpoints"), {
      recursive: true,
    });
    writeFileSync(
      join(
        classifierDirectory,
        "generation-1",
        "candidates",
        "failed",
        "training.jsonl",
      ),
      "historic trainer input",
    );
    writeFileSync(
      join(classifierDirectory, "generation-1", "checkpoints", "needle2.pkl"),
      "shared base checkpoint",
    );

    const targetExamplesByPurpose = {
      validation: 135,
      representative_test: 100,
      coverage_test: 35,
    } as const;
    const plan = inspectLegacyHeldOutMigration({
      dataDirectory,
      classifierName: name,
      targetExamplesByPurpose,
    });
    expect(plan).toMatchObject({
      status: "ready",
      preservedTrainingExamples: 1_049,
      eligibleHeldOutExamples: 270,
      observedTrainingAttempts: 24,
      observedExamplesUsedForTraining: 1_309,
      blockers: [],
      planSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
    });
    expect(
      plan.heldOutGroups.reduce((total, group) => total + group.examples, 0),
    ).toBe(270);
    expect(plan.heldOutGroups).toHaveLength(2);

    const result = migrateLegacyHeldOutExamples({
      dataDirectory,
      classifierName: name,
      targetExamplesByPurpose,
      expectedPlanSha256: plan.planSha256,
      attestation: {
        heldOutExamplesWereNeverUsedForModelSelection: true,
        operator: "production-migration",
        reason: "No candidate model or evaluation exists",
      },
    });
    expect(result).toMatchObject({
      status: "migrated",
      preservedTrainingExamples: 1_049,
      examplesByPurpose: targetExamplesByPurpose,
      planSha256: plan.planSha256,
      evidence: {
        observedTrainingAttempts: 24,
        observedExamplesUsedForTraining: 1_309,
      },
    });
    expect(trainingRowsSha256(dataDirectory, name)).toBe(
      originalTrainingRowsSha256,
    );

    const classifier = createClassifier({
      name,
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      datasetRequirements: {
        minimumTrainingExamples: 800,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 100,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
      dataDirectory,
    });

    const purposes = classifier.inspect().examplesByPurpose;
    expect(purposes.training).toBe(1_049);
    expect(purposes.validation).toBe(135);
    expect(purposes.representative_test).toBe(100);
    expect(purposes.coverage_test).toBe(35);
    for (const resultBin of classifier.inspect().resultBins.filter(
      (bin) => bin.total > 0,
    )) {
      expect(resultBin.purposes.validation).toBeGreaterThan(0);
      expect(resultBin.purposes.representative_test).toBeGreaterThan(0);
      expect(resultBin.purposes.coverage_test).toBeGreaterThan(0);
    }
    await classifier.close();

    expect(
      inspectLegacyHeldOutMigration({
        dataDirectory,
        classifierName: name,
        targetExamplesByPurpose,
      }),
    ).toMatchObject({
      status: "already_migrated",
      planSha256: plan.planSha256,
      priorMigration: {
        operator: "production-migration",
        reason: "No candidate model or evaluation exists",
      },
    });

    const eraser = createClassifier({
      name,
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.8,
      decisionBoundaries: [0.5],
      datasetRequirements: {
        minimumTrainingExamples: 800,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 100,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
      dataDirectory,
    });
    await eraser.erase();
    await eraser.close();
    expect(
      inspectLegacyHeldOutMigration({
        dataDirectory,
        classifierName: name,
        targetExamplesByPurpose,
      }).priorMigration,
    ).toBeNull();
  }, 15_000);

  it.each(["model.cact", "swapai-lora.pkl", "model.cact.numbers.json"])(
    "blocks legacy held-out migration when candidate output %s exists",
    async (candidateFile) => {
      const dataDirectory = makeDirectory();
      const name = `candidate-output-${candidateFile}`;
      const legacy = init({
        name,
        result: { type: "boolean" },
        retrainOnCount: 50,
        acceptableError: "10%",
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
        maxTrainingSet: 100,
        automaticTraining: false,
        dataDirectory,
      });
      for (let index = 0; index < 20; index += 1) {
        legacy.logClassification(`historic-${index}`, index % 2 === 0);
      }
      await legacy.close();
      const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
      database.function(
        "swapai_writer_version",
        { deterministic: true },
        () => 3,
      );
      database
        .prepare(
          `
          UPDATE examples
          SET split = CASE WHEN id <= 15 THEN 'training' ELSE 'held_out' END
          WHERE classifier_name = ?
        `,
        )
        .run(name);
      database
        .prepare(
          "UPDATE classifiers SET training_attempts = 1 WHERE name = ?",
        )
        .run(name);
      database.close();
      mimicVersion050LegacyAdoption(dataDirectory, name);

      const configured = createClassifier({
        name,
        result: { type: "boolean" },
        reference: async () => true,
        datasetRequirements: {
          minimumTrainingExamples: 1,
          minimumTrainingExamplesPerResultBin: 0,
          minimumValidationExamplesPerResultBin: 0,
          minimumRepresentativeTestExamples: 0,
          minimumCoverageTestExamplesPerResultBin: 0,
        },
        dataDirectory,
      });
      await configured.close();

      const candidateDirectory = join(
        dataDirectory,
        "classifiers",
        createHash("sha256").update(name).digest("hex"),
        "generation-1",
        "candidates",
        "historic",
      );
      mkdirSync(candidateDirectory, { recursive: true });
      writeFileSync(join(candidateDirectory, candidateFile), "candidate");

      expect(
        inspectLegacyHeldOutMigration({
          dataDirectory,
          classifierName: name,
          targetExamplesByPurpose: {
            validation: 2,
            representative_test: 2,
            coverage_test: 1,
          },
        }),
      ).toMatchObject({
        status: "blocked",
        blockers: [
          expect.objectContaining({ code: "candidate_artifact" }),
        ],
      });
    },
  );

  it("requires every classifier runtime to stop before legacy held-out migration", async () => {
    const dataDirectory = makeDirectory();
    const name = "offline-held-out-migration";
    await prepareAttemptedLegacyDataset({
      dataDirectory,
      name,
      totalExamples: 20,
      trainingExamples: 15,
    });
    const configured = createClassifier({
      name,
      result: { type: "boolean" },
      reference: async () => true,
      datasetRequirements: zeroBinMinimums,
      dataDirectory,
    });
    const options = {
      dataDirectory,
      classifierName: name,
      targetExamplesByPurpose: {
        validation: 2,
        representative_test: 2,
        coverage_test: 1,
      },
    } as const;

    expect(inspectLegacyHeldOutMigration(options)).toMatchObject({
      status: "blocked",
      blockers: [expect.objectContaining({ code: "active_runtime" })],
    });
    await configured.close();
    expect(inspectLegacyHeldOutMigration(options)).toMatchObject({
      status: "ready",
      blockers: [],
    });
  });

  it("rejects a held-out migration when the reviewed plan changes", async () => {
    const dataDirectory = makeDirectory();
    const name = "changed-held-out-plan";
    await prepareAttemptedLegacyDataset({
      dataDirectory,
      name,
      totalExamples: 20,
      trainingExamples: 15,
    });
    const configured = createClassifier({
      name,
      result: { type: "boolean" },
      reference: async () => true,
      datasetRequirements: zeroBinMinimums,
      dataDirectory,
    });
    await configured.close();
    const options = {
      dataDirectory,
      classifierName: name,
      targetExamplesByPurpose: {
        validation: 2,
        representative_test: 2,
        coverage_test: 1,
      },
    } as const;
    const reviewed = inspectLegacyHeldOutMigration(options);

    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.function("swapai_writer_version", { deterministic: true }, () => 3);
    database
      .prepare(
        `
        UPDATE examples SET facets_json = '{"changed":"yes"}'
        WHERE id = (
          SELECT MIN(id) FROM examples
          WHERE classifier_name = ? AND split = 'held_out'
        )
      `,
      )
      .run(name);
    database.close();

    expect(() =>
      migrateLegacyHeldOutExamples({
        ...options,
        expectedPlanSha256: reviewed.planSha256,
        attestation: {
          heldOutExamplesWereNeverUsedForModelSelection: true,
          operator: "test-operator",
          reason: "No candidate was produced",
        },
      }),
    ).toThrowError(
      expect.objectContaining({
        code: "storage_failed",
        message: expect.stringContaining("plan changed"),
      }),
    );
  });

  it("blocks migration when a configured result bin cannot receive protected examples", async () => {
    const dataDirectory = makeDirectory();
    const name = "missing-held-out-result-bin";
    await prepareAttemptedLegacyDataset({
      dataDirectory,
      name,
      totalExamples: 20,
      trainingExamples: 15,
    });
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.function("swapai_writer_version", { deterministic: true }, () => 3);
    database
      .prepare(
        "UPDATE examples SET result_json = 'true' WHERE classifier_name = ? AND split = 'held_out'",
      )
      .run(name);
    database.close();
    mimicVersion050LegacyAdoption(dataDirectory, name);
    const configured = createClassifier({
      name,
      result: { type: "boolean" },
      reference: async () => true,
      datasetRequirements: {
        ...zeroBinMinimums,
        minimumValidationExamplesPerResultBin: 1,
      },
      dataDirectory,
    });
    await configured.close();

    expect(
      inspectLegacyHeldOutMigration({
        dataDirectory,
        classifierName: name,
        targetExamplesByPurpose: {
          validation: 2,
          representative_test: 2,
          coverage_test: 1,
        },
      }),
    ).toMatchObject({
      status: "blocked",
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: "result_bin_lacks_protected_examples",
        }),
      ]),
    });
  });

  it("blocks migration when legacy held-out evaluation evidence exists", async () => {
    const dataDirectory = makeDirectory();
    const name = "evaluated-held-out-data";
    await prepareAttemptedLegacyDataset({
      dataDirectory,
      name,
      totalExamples: 20,
      trainingExamples: 15,
    });
    const configured = createClassifier({
      name,
      result: { type: "boolean" },
      reference: async () => true,
      datasetRequirements: zeroBinMinimums,
      dataDirectory,
    });
    await configured.close();
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.function("swapai_writer_version", { deterministic: true }, () => 3);
    database
      .prepare(
        `
        UPDATE classifiers
        SET last_evaluated_error = 0.2, last_evaluated_at = ?
        WHERE name = ?
      `,
      )
      .run(Date.now(), name);
    database.close();

    expect(
      inspectLegacyHeldOutMigration({
        dataDirectory,
        classifierName: name,
        targetExamplesByPurpose: {
          validation: 2,
          representative_test: 2,
          coverage_test: 1,
        },
      }),
    ).toMatchObject({
      status: "blocked",
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: "candidate_evaluation" }),
      ]),
    });
  });

  it.each([
    "local attempt",
    "provider run",
    "model artifact",
    "trained generation",
  ] as const)(
    "does not turn legacy examples exposed by a %s into protected evidence",
    async (evidence) => {
      const dataDirectory = makeDirectory();
      const legacy = init({
        name: "previously-trained",
        result: { type: "boolean" },
        retrainOnCount: 50,
        acceptableError: "10%",
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
        maxTrainingSet: 10_000,
        automaticTraining: false,
        dataDirectory,
      });
      for (let index = 0; index < 200; index += 1) {
        legacy.logClassification(`historic-example-${index}`, index % 2 === 0);
      }
      await legacy.close();
      mimicVersion050LegacyAdoption(dataDirectory, "previously-trained");

      const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
      database.function(
        "swapai_writer_version",
        { deterministic: true },
        () => 3,
      );
      if (evidence === "local attempt") {
        database
          .prepare(
            "UPDATE classifiers SET training_attempts = 1 WHERE name = ?",
          )
          .run("previously-trained");
      } else if (evidence === "provider run") {
        database
          .prepare(
            `
          INSERT INTO dataset_revisions (
            id, classifier_name, generation, data_epoch, created_at
          ) VALUES ('historic-revision', 'previously-trained', 1, 0, ?)
        `,
          )
          .run(Date.now());
        database
          .prepare(
            `
          INSERT INTO training_runs (
            id, dataset_revision_id, classifier_name, provider_name, status,
            cleanup_status, started_at, finished_at
          ) VALUES (
            'historic-run', 'historic-revision', 'previously-trained',
            'historic-provider', 'failed', 'succeeded', ?, ?
          )
        `,
          )
          .run(Date.now(), Date.now());
      } else if (evidence === "model artifact") {
        database
          .prepare(
            `
          INSERT INTO model_artifacts (
            classifier_name, sha256, model_path, needle_version,
            size_bytes, created_at
          ) VALUES (
            'previously-trained', 'historic-sha', '/historic/model.cact',
            '2.0.14', 1, ?
          )
        `,
          )
          .run(Date.now());
      } else {
        database
          .prepare(
            `
          UPDATE generations
          SET trained = 1,
              model_path = '/historic/model.cact',
              needle_version = '2.0.14'
          WHERE classifier_name = 'previously-trained' AND generation = 1
        `,
          )
          .run();
      }
      database.close();

      const classifier = createClassifier({
        name: "previously-trained",
        result: { type: "boolean" },
        reference: async () => true,
        dataDirectory,
      });

      expect(classifier.inspect().examplesByPurpose).toMatchObject({
        representative_test: 0,
        coverage_test: 0,
      });
      expect(
        classifier.inspect().examplesByPurpose.training +
          classifier.inspect().examplesByPurpose.validation,
      ).toBe(200);
      await classifier.close();
    },
  );

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
    expect(
      coverage.find((group) => group.value === "invoice")?.total,
    ).toBeGreaterThan(0);
    expect(
      coverage.find((group) => group.value === "newsletter")?.total,
    ).toBeGreaterThan(0);
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

    expect(relevance.inspect().facetCoverage).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          facet: "documentFamily",
          value: null,
          total: 1,
        }),
        expect.objectContaining({
          facet: "documentFamily",
          value: "invoice",
          total: 1,
        }),
        expect.objectContaining({
          facet: "documentFamily",
          value: "newsletter",
          total: 1,
        }),
      ]),
    );
    await relevance.close();
  });

  it("does not spend when any protected evaluation suite is empty", async () => {
    const train = vi.fn(async (_job: TrainingJob) => {
      throw new Error("training must not start");
    });
    const relevance = createClassifier({
      name: "empty-protected-suites",
      result: { type: "boolean" },
      reference: async () => true,
      training: { name: "must-not-run", train },
      dataDirectory: makeDirectory(),
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });
    relevance.logClassification("training-example-1", true);
    await relevance.flush();

    expect(relevance.inspect()).toMatchObject({
      readyForTraining: false,
      deficits: expect.arrayContaining([
        {
          purpose: "validation",
          resultBin: null,
          required: 1,
          available: 0,
        },
        {
          purpose: "representative_test",
          resultBin: null,
          required: 1,
          available: 0,
        },
        {
          purpose: "coverage_test",
          resultBin: null,
          required: 1,
          available: 0,
        },
      ]),
    });
    await expect(relevance.requestTraining()).resolves.toMatchObject({
      status: "not_ready",
    });
    expect(train).not.toHaveBeenCalled();
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
        (example) => example.purpose === "training",
      ),
    ).toBe(true);
    expect(
      receivedJobs[0]!.examples.some(
        (example) => example.purpose === "training",
      ),
    ).toBe(true);
    const failedRun = relevance.inspect().latestTrainingRun!;
    expect(failedRun).toMatchObject({
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
    await expect(relevance.requestTraining()).resolves.toEqual({
      status: "already_failed",
      trainingRunId: failedRun.id,
      datasetRevisionId: failedRun.datasetRevisionId,
    });
    expect(receivedJobs).toHaveLength(1);
    await expect(relevance.retryTraining(failedRun.id)).rejects.toThrow(
      /reconciled before retry/i,
    );
    expect(receivedJobs).toHaveLength(1);
    await relevance.close();
  });

  it("spends again after failure only through explicit retry", async () => {
    const attempts: TrainingJob[] = [];
    const trainingFailure = new Error("trainer unavailable");
    const relevance = createClassifier({
      name: "explicit-training-retry",
      result: { type: "boolean" },
      reference: async (input) => input.includes("yes"),
      training: {
        name: "retry-test",
        train: async (job) => {
          attempts.push(job);
          throw trainingFailure;
        },
      },
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

    await expect(relevance.requestTraining()).rejects.toBe(trainingFailure);
    const failedRun = relevance.inspect().latestTrainingRun!;
    await expect(relevance.requestTraining()).resolves.toMatchObject({
      status: "already_failed",
      trainingRunId: failedRun.id,
    });
    expect(attempts).toHaveLength(1);
    await expect(relevance.retryTraining(failedRun.id)).rejects.toBe(
      trainingFailure,
    );
    expect(attempts).toHaveLength(2);
    expect(attempts[1]!.id).not.toBe(failedRun.id);
    await relevance.close();
  });
});

const mimicVersion050LegacyAdoption = (
  dataDirectory: string,
  classifierName: string,
): void => {
  const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.function("test_input_hash", { deterministic: true }, (input) =>
    createHash("sha256").update(String(input)).digest("hex"),
  );
  database
    .prepare(
      `
      UPDATE examples
      SET input_hash = test_input_hash(input),
          result_bin = CASE result_json WHEN 'true' THEN 'true'
            WHEN 'false' THEN 'false'
            ELSE CASE WHEN CAST(result_json AS REAL) < 0.5
              THEN '0..0.2' ELSE '0.8..1' END
          END,
          purpose = CASE split WHEN 'training' THEN 'training'
            ELSE 'validation' END
      WHERE classifier_name = ?
    `,
    )
    .run(classifierName);
  database.close();
};

const prepareAttemptedLegacyDataset = async (options: {
  dataDirectory: string;
  name: string;
  totalExamples: number;
  trainingExamples: number;
}): Promise<void> => {
  const legacy = init({
    name: options.name,
    result: { type: "boolean" },
    retrainOnCount: 50,
    acceptableError: "10%",
    retestInterval: 100,
    retestRevertOn: 3,
    model: "needle2",
    maxTrainingSet: 100,
    automaticTraining: false,
    dataDirectory: options.dataDirectory,
  });
  for (let index = 0; index < options.totalExamples; index += 1) {
    legacy.logClassification(`historic-${index}`, index % 2 === 0);
  }
  await legacy.close();
  const database = new DatabaseSync(
    join(options.dataDirectory, "swapai.sqlite"),
  );
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database
    .prepare(
      `
      UPDATE examples
      SET split = CASE WHEN id <= ? THEN 'training' ELSE 'held_out' END
      WHERE classifier_name = ?
    `,
    )
    .run(options.trainingExamples, options.name);
  database
    .prepare(
      `
      UPDATE classifiers
      SET training_attempts = 1,
          examples_used_for_training = ?
      WHERE name = ?
    `,
    )
    .run(options.totalExamples, options.name);
  database.close();
  mimicVersion050LegacyAdoption(options.dataDirectory, options.name);
};

const trainingRowsSha256 = (
  dataDirectory: string,
  classifierName: string,
): string => {
  const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"), {
    readOnly: true,
  });
  try {
    const rows = database
      .prepare(
        `
        SELECT id, generation, input, result_json, split, input_hash,
               result_bin, purpose, facets_json, created_at
        FROM examples
        WHERE classifier_name = ? AND split = 'training'
        ORDER BY id
      `,
      )
      .all(classifierName);
    return createHash("sha256").update(JSON.stringify(rows)).digest("hex");
  } finally {
    database.close();
  }
};
