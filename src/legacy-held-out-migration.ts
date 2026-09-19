import { createHash } from "node:crypto";
import { existsSync, readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join, relative } from "node:path";
import type { DatabaseSync as DatabaseSyncType } from "node:sqlite";

import { SwapAIError } from "./errors.js";
import { resultBins } from "./dataset-policy.js";
import type {
  DatasetPolicy,
  DatasetRequirements,
  ResultConfig,
} from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

const LIVE_RUNTIME_WINDOW_MILLISECONDS = 10_000;

export interface LegacyHeldOutPurposeTargets {
  readonly validation: number;
  readonly representative_test: number;
  readonly coverage_test: number;
}

export interface LegacyHeldOutMigrationOptions {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly targetExamplesByPurpose: LegacyHeldOutPurposeTargets;
}

export type LegacyHeldOutMigrationBlockerCode =
  | "active_runtime"
  | "candidate_evaluation"
  | "candidate_artifact"
  | "trained_generation"
  | "previous_generation"
  | "training_run"
  | "invalid_legacy_dataset"
  | "target_count_mismatch"
  | "target_below_configured_minimum"
  | "result_bin_lacks_protected_examples";

export interface LegacyHeldOutMigrationBlocker {
  readonly code: LegacyHeldOutMigrationBlockerCode;
  readonly message: string;
}

export interface LegacyHeldOutMigrationEvidence {
  readonly operator: string;
  readonly reason: string;
  readonly migratedAt: number;
  readonly observedTrainingAttempts: number;
  readonly observedExamplesUsedForTraining: number;
}

export interface LegacyHeldOutMigrationGroup {
  readonly resultBin: string;
  readonly facets: Readonly<Record<string, string>>;
  readonly examples: number;
}

export interface LegacyHeldOutMigrationInspection {
  readonly status: "ready" | "blocked" | "already_migrated";
  readonly classifierName: string;
  readonly planSha256: string;
  readonly preservedTrainingExamples: number;
  readonly eligibleHeldOutExamples: number;
  readonly observedTrainingAttempts: number;
  readonly observedExamplesUsedForTraining: number;
  readonly heldOutGroups: readonly LegacyHeldOutMigrationGroup[];
  readonly targetExamplesByPurpose: LegacyHeldOutPurposeTargets;
  readonly blockers: readonly LegacyHeldOutMigrationBlocker[];
  readonly priorMigration: LegacyHeldOutMigrationEvidence | null;
}

export interface LegacyHeldOutMigrationAttestation {
  readonly heldOutExamplesWereNeverUsedForModelSelection: true;
  readonly operator: string;
  readonly reason: string;
}

export interface MigrateLegacyHeldOutExamplesOptions
  extends LegacyHeldOutMigrationOptions {
  readonly expectedPlanSha256: string;
  readonly attestation: LegacyHeldOutMigrationAttestation;
}

export interface LegacyHeldOutMigrationResult {
  readonly status: "migrated" | "already_migrated";
  readonly classifierName: string;
  readonly planSha256: string;
  readonly preservedTrainingExamples: number;
  readonly examplesByPurpose: LegacyHeldOutPurposeTargets;
  readonly evidence: LegacyHeldOutMigrationEvidence;
}

interface LegacyExample {
  readonly id: number;
  readonly input_hash: string;
  readonly result_bin: string;
  readonly facets_json: string;
}

interface PlannedMigration extends LegacyHeldOutMigrationInspection {
  readonly assignments: ReadonlyMap<number, keyof LegacyHeldOutPurposeTargets>;
}

interface StoredClassifier {
  readonly active_generation: number;
  readonly config_json: string;
  readonly last_evaluated_error: number | null;
  readonly last_evaluated_at: number | null;
  readonly training_attempts: number;
  readonly examples_used_for_training: number;
}

interface StoredMigration {
  readonly plan_sha256: string;
  readonly preserved_training_examples: number;
  readonly validation_examples: number;
  readonly representative_test_examples: number;
  readonly coverage_test_examples: number;
  readonly operator: string;
  readonly reason: string;
  readonly migrated_at: number;
  readonly observed_training_attempts: number;
  readonly observed_examples_used_for_training: number;
}

const MIGRATION_TABLE = `
  CREATE TABLE IF NOT EXISTS legacy_held_out_purpose_migrations (
    classifier_name TEXT PRIMARY KEY,
    plan_sha256 TEXT NOT NULL,
    preserved_training_examples INTEGER NOT NULL,
    validation_examples INTEGER NOT NULL,
    representative_test_examples INTEGER NOT NULL,
    coverage_test_examples INTEGER NOT NULL,
    operator TEXT NOT NULL,
    reason TEXT NOT NULL,
    observed_training_attempts INTEGER NOT NULL,
    observed_examples_used_for_training INTEGER NOT NULL,
    migrated_at INTEGER NOT NULL,
    FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
  )
`;

export const inspectLegacyHeldOutMigration = (
  options: LegacyHeldOutMigrationOptions,
): LegacyHeldOutMigrationInspection => {
  const databasePath = join(options.dataDirectory, "swapai.sqlite");
  if (!existsSync(databasePath)) {
    throw new SwapAIError(
      "invalid_configuration",
      `Classifier "${options.classifierName}" has no SwapAI database`,
    );
  }
  const database = new DatabaseSync(databasePath, { readOnly: true });
  try {
    database.exec("PRAGMA busy_timeout = 5000");
    return withoutAssignments(buildPlan(database, options));
  } finally {
    database.close();
  }
};

export const migrateLegacyHeldOutExamples = (
  options: MigrateLegacyHeldOutExamplesOptions,
): LegacyHeldOutMigrationResult => {
  requireAttestation(options.attestation);
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  database.exec("BEGIN EXCLUSIVE");
  try {
    database.exec(MIGRATION_TABLE);
    const plan = buildPlan(database, options);
    if (plan.status === "already_migrated") {
      database.exec("COMMIT");
      return resultFromPriorMigration(plan);
    }
    if (plan.planSha256 !== options.expectedPlanSha256) {
      throw new SwapAIError(
        "storage_failed",
        "Legacy held-out migration plan changed; inspect it again before applying",
      );
    }
    if (plan.status !== "ready") {
      throw new SwapAIError(
        "invalid_configuration",
        plan.blockers.map((blocker) => blocker.message).join("; "),
      );
    }

    const update = database.prepare(`
      UPDATE examples
      SET purpose = ?
      WHERE id = ?
        AND classifier_name = ?
        AND generation = ?
        AND split = 'held_out'
        AND purpose = 'validation'
    `);
    const activeGeneration = requiredClassifier(
      database,
      options.classifierName,
    ).active_generation;
    for (const [id, purpose] of plan.assignments) {
      const result = update.run(
        purpose,
        id,
        options.classifierName,
        activeGeneration,
      );
      if (result.changes !== 1) {
        throw new SwapAIError(
          "storage_failed",
          "Legacy held-out migration lost an example while holding the offline lock",
        );
      }
    }

    const migratedAt = Date.now();
    database
      .prepare(
        `
        INSERT INTO legacy_held_out_purpose_migrations (
          classifier_name, plan_sha256, preserved_training_examples,
          validation_examples, representative_test_examples,
          coverage_test_examples, operator, reason,
          observed_training_attempts, observed_examples_used_for_training,
          migrated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `,
      )
      .run(
        options.classifierName,
        plan.planSha256,
        plan.preservedTrainingExamples,
        options.targetExamplesByPurpose.validation,
        options.targetExamplesByPurpose.representative_test,
        options.targetExamplesByPurpose.coverage_test,
        options.attestation.operator.trim(),
        options.attestation.reason.trim(),
        plan.observedTrainingAttempts,
        plan.observedExamplesUsedForTraining,
        migratedAt,
      );
    database.exec("COMMIT");
    return {
      status: "migrated",
      classifierName: options.classifierName,
      planSha256: plan.planSha256,
      preservedTrainingExamples: plan.preservedTrainingExamples,
      examplesByPurpose: options.targetExamplesByPurpose,
      evidence: {
        operator: options.attestation.operator.trim(),
        reason: options.attestation.reason.trim(),
        migratedAt,
        observedTrainingAttempts: plan.observedTrainingAttempts,
        observedExamplesUsedForTraining: plan.observedExamplesUsedForTraining,
      },
    };
  } catch (error) {
    database.exec("ROLLBACK");
    throw error;
  } finally {
    database.close();
  }
};

const buildPlan = (
  database: DatabaseSyncType,
  options: LegacyHeldOutMigrationOptions,
): PlannedMigration => {
  validateTargets(options.targetExamplesByPurpose);
  const classifier = requiredClassifier(database, options.classifierName);
  const prior = readPriorMigration(database, options.classifierName);
  if (prior !== null) {
    const migratedHeldOut = readHeldOutExamples(
      database,
      options.classifierName,
      classifier.active_generation,
    );
    return {
      status: "already_migrated",
      classifierName: options.classifierName,
      planSha256: prior.plan_sha256,
      preservedTrainingExamples: prior.preserved_training_examples,
      eligibleHeldOutExamples:
        prior.validation_examples +
        prior.representative_test_examples +
        prior.coverage_test_examples,
      observedTrainingAttempts: prior.observed_training_attempts,
      observedExamplesUsedForTraining:
        prior.observed_examples_used_for_training,
      heldOutGroups: heldOutGroups(migratedHeldOut),
      targetExamplesByPurpose: {
        validation: prior.validation_examples,
        representative_test: prior.representative_test_examples,
        coverage_test: prior.coverage_test_examples,
      },
      blockers: [],
      priorMigration: {
        operator: prior.operator,
        reason: prior.reason,
        migratedAt: prior.migrated_at,
        observedTrainingAttempts: prior.observed_training_attempts,
        observedExamplesUsedForTraining:
          prior.observed_examples_used_for_training,
      },
      assignments: new Map(),
    };
  }

  const trainingCount = countExamples(
    database,
    options.classifierName,
    classifier.active_generation,
    "training",
  );
  const heldOut = database
    .prepare(
      `
      SELECT id, input_hash, result_bin, facets_json
      FROM examples
      WHERE classifier_name = ?
        AND generation = ?
        AND split = 'held_out'
        AND purpose = 'validation'
      ORDER BY id
    `,
    )
    .all(options.classifierName, classifier.active_generation)
    .map((value) => value as unknown as LegacyExample);
  const totalHeldOut = countExamples(
    database,
    options.classifierName,
    classifier.active_generation,
    "held_out",
  );
  const blockers = evidenceBlockers(
    database,
    options,
    classifier,
    heldOut.length,
    totalHeldOut,
  );
  const storedPolicy = readStoredPolicy(classifier.config_json);
  addTargetBlockers(
    blockers,
    options.targetExamplesByPurpose,
    heldOut,
    storedPolicy.requirements,
    storedPolicy.resultBinIds,
  );
  const assignments = blockers.length === 0
    ? assignProtectedPurposes(
        heldOut,
        options.targetExamplesByPurpose,
        storedPolicy.requirements,
      )
    : new Map<number, keyof LegacyHeldOutPurposeTargets>();
  const planSha256 = hashPlan({
    classifierName: options.classifierName,
    generation: classifier.active_generation,
    trainingCount,
    trainingAttempts: classifier.training_attempts,
    examplesUsedForTraining: classifier.examples_used_for_training,
    heldOut,
    targets: options.targetExamplesByPurpose,
    requirements: storedPolicy.requirements,
    configuredResultBinIds: storedPolicy.resultBinIds,
    blockerCodes: blockers.map((blocker) => blocker.code),
  });
  return {
    status: blockers.length === 0 ? "ready" : "blocked",
    classifierName: options.classifierName,
    planSha256,
    preservedTrainingExamples: trainingCount,
    eligibleHeldOutExamples: heldOut.length,
    observedTrainingAttempts: classifier.training_attempts,
    observedExamplesUsedForTraining: classifier.examples_used_for_training,
    heldOutGroups: heldOutGroups(heldOut),
    targetExamplesByPurpose: options.targetExamplesByPurpose,
    blockers,
    priorMigration: null,
    assignments,
  };
};

const readHeldOutExamples = (
  database: DatabaseSyncType,
  classifierName: string,
  generation: number,
): readonly LegacyExample[] =>
  database
    .prepare(
      `
      SELECT id, input_hash, result_bin, facets_json
      FROM examples
      WHERE classifier_name = ? AND generation = ? AND split = 'held_out'
      ORDER BY id
    `,
    )
    .all(classifierName, generation)
    .map((value) => value as unknown as LegacyExample);

const evidenceBlockers = (
  database: DatabaseSyncType,
  options: LegacyHeldOutMigrationOptions,
  classifier: StoredClassifier,
  eligibleHeldOutCount: number,
  totalHeldOutCount: number,
): LegacyHeldOutMigrationBlocker[] => {
  const blockers: LegacyHeldOutMigrationBlocker[] = [];
  const liveRuntime = database
    .prepare(
      `
      SELECT 1 FROM classifier_runtimes
      WHERE classifier_name = ? AND heartbeat_at >= ?
      LIMIT 1
    `,
    )
    .get(
      options.classifierName,
      Date.now() - LIVE_RUNTIME_WINDOW_MILLISECONDS,
    );
  if (liveRuntime !== undefined) {
    blockers.push({
      code: "active_runtime",
      message: "Stop every classifier runtime before migrating legacy held-out examples",
    });
  }
  if (
    classifier.last_evaluated_at !== null ||
    classifier.last_evaluated_error !== null ||
    countRows(database, "training_evaluations", options.classifierName) > 0 ||
    countRows(database, "shadow_evaluations", options.classifierName) > 0
  ) {
    blockers.push({
      code: "candidate_evaluation",
      message: "Stored candidate evaluation evidence proves held-out model selection",
    });
  }
  const generations = database
    .prepare(
      `
      SELECT generation, status, trained, model_path, needle_version
      FROM generations WHERE classifier_name = ?
    `,
    )
    .all(options.classifierName) as unknown as Array<{
      generation: number;
      status: string;
      trained: number;
      model_path: string | null;
      needle_version: string | null;
    }>;
  if (
    generations.some(
      (generation) =>
        generation.trained !== 0 ||
        generation.model_path !== null ||
        generation.needle_version !== null,
    ) || countRows(database, "model_artifacts", options.classifierName) > 0
  ) {
    blockers.push({
      code: "trained_generation",
      message: "A trained generation or indexed model artifact exists",
    });
  }
  if (
    generations.some(
      (generation) =>
        generation.generation !== classifier.active_generation ||
        generation.status !== "active",
    )
  ) {
    blockers.push({
      code: "previous_generation",
      message: "A previous classifier generation may have used held-out model selection",
    });
  }
  if (countRows(database, "training_runs", options.classifierName) > 0) {
    blockers.push({
      code: "training_run",
      message: "A provider Training Run exists for this classifier",
    });
  }
  if (candidateArtifactPaths(options).length > 0) {
    blockers.push({
      code: "candidate_artifact",
      message: "A candidate model, LoRA, or candidate sidecar exists on disk",
    });
  }
  if (eligibleHeldOutCount !== totalHeldOutCount) {
    blockers.push({
      code: "invalid_legacy_dataset",
      message: "Original held-out rows are not all in the legacy validation purpose",
    });
  }
  return blockers;
};

const addTargetBlockers = (
  blockers: LegacyHeldOutMigrationBlocker[],
  targets: LegacyHeldOutPurposeTargets,
  heldOut: readonly LegacyExample[],
  requirements: DatasetRequirements,
  configuredResultBinIds: readonly string[],
): void => {
  const targetTotal =
    targets.validation + targets.representative_test + targets.coverage_test;
  if (targetTotal !== heldOut.length) {
    blockers.push({
      code: "target_count_mismatch",
      message: `Protected targets total ${targetTotal}; exactly ${heldOut.length} legacy held-out examples must be assigned`,
    });
  }
  const bins = configuredResultBinIds;
  const minimumValidation = Math.max(
    1,
    requirements.minimumValidationExamplesPerResultBin * bins.length,
  );
  const minimumRepresentative = Math.max(
    1,
    requirements.minimumRepresentativeTestExamples,
  );
  const minimumCoverage = Math.max(
    1,
    requirements.minimumCoverageTestExamplesPerResultBin * bins.length,
  );
  if (
    targets.validation < minimumValidation ||
    targets.representative_test < minimumRepresentative ||
    targets.coverage_test < minimumCoverage
  ) {
    blockers.push({
      code: "target_below_configured_minimum",
      message: `Protected targets must be at least validation=${minimumValidation}, representative_test=${minimumRepresentative}, coverage_test=${minimumCoverage}`,
    });
  }
  for (const bin of bins) {
    const available = heldOut.filter(
      (example) => example.result_bin === bin,
    ).length;
    const required =
      requirements.minimumValidationExamplesPerResultBin +
      requirements.minimumCoverageTestExamplesPerResultBin;
    if (available < required) {
      blockers.push({
        code: "result_bin_lacks_protected_examples",
        message: `Result bin "${bin}" has ${available} held-out examples but needs ${required} for configured validation and coverage minima`,
      });
    }
  }
};

const assignProtectedPurposes = (
  heldOut: readonly LegacyExample[],
  targets: LegacyHeldOutPurposeTargets,
  requirements: DatasetRequirements,
): ReadonlyMap<number, keyof LegacyHeldOutPurposeTargets> => {
  const assignments = new Map<number, keyof LegacyHeldOutPurposeTargets>();
  const byBin = groupExamples(heldOut, (example) => example.result_bin);
  for (const bin of [...byBin.keys()].sort()) {
    const ordered = stratifiedOrder(byBin.get(bin)!);
    let offset = 0;
    for (
      let count = 0;
      count < requirements.minimumValidationExamplesPerResultBin;
      count += 1
    ) {
      assignments.set(ordered[offset++]!.id, "validation");
    }
    for (
      let count = 0;
      count < requirements.minimumCoverageTestExamplesPerResultBin;
      count += 1
    ) {
      assignments.set(ordered[offset++]!.id, "coverage_test");
    }
  }
  const remaining = stratifiedOrder(
    heldOut.filter((example) => !assignments.has(example.id)),
  );
  const assignedValidation = countAssigned(assignments, "validation");
  const assignedCoverage = countAssigned(assignments, "coverage_test");
  const schedule: Array<keyof LegacyHeldOutPurposeTargets> = [
    ...Array.from(
      { length: targets.validation - assignedValidation },
      () => "validation" as const,
    ),
    ...Array.from(
      { length: targets.representative_test },
      () => "representative_test" as const,
    ),
    ...Array.from(
      { length: targets.coverage_test - assignedCoverage },
      () => "coverage_test" as const,
    ),
  ];
  for (const [index, purpose] of schedule.entries()) {
    assignments.set(remaining[index]!.id, purpose);
  }
  return assignments;
};

const stratifiedOrder = (
  examples: readonly LegacyExample[],
): readonly LegacyExample[] => {
  const groups = groupExamples(
    examples,
    (example) => `${example.result_bin}\0${example.facets_json}`,
  );
  const orderedGroups = [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, group]) =>
      [...group].sort((left, right) =>
        migrationOrder(left).localeCompare(migrationOrder(right)),
      ),
    );
  const ordered: LegacyExample[] = [];
  for (let index = 0; ordered.length < examples.length; index += 1) {
    for (const group of orderedGroups) {
      const example = group[index];
      if (example !== undefined) ordered.push(example);
    }
  }
  return ordered;
};

const migrationOrder = (example: LegacyExample): string =>
  createHash("sha256")
    .update("swapai-legacy-held-out-purpose-v1")
    .update("\0")
    .update(example.input_hash)
    .digest("hex");

const heldOutGroups = (
  examples: readonly LegacyExample[],
): readonly LegacyHeldOutMigrationGroup[] =>
  [...groupExamples(
    examples,
    (example) => `${example.result_bin}\0${example.facets_json}`,
  ).entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, group]) => ({
      resultBin: group[0]!.result_bin,
      facets: JSON.parse(group[0]!.facets_json) as Record<string, string>,
      examples: group.length,
    }));

const groupExamples = (
  examples: readonly LegacyExample[],
  keyFor: (example: LegacyExample) => string,
): Map<string, LegacyExample[]> => {
  const groups = new Map<string, LegacyExample[]>();
  for (const example of examples) {
    const key = keyFor(example);
    const group = groups.get(key) ?? [];
    group.push(example);
    groups.set(key, group);
  }
  return groups;
};

const countAssigned = (
  assignments: ReadonlyMap<number, keyof LegacyHeldOutPurposeTargets>,
  purpose: keyof LegacyHeldOutPurposeTargets,
): number =>
  [...assignments.values()].filter((candidate) => candidate === purpose).length;

const candidateArtifactPaths = (
  options: Pick<LegacyHeldOutMigrationOptions, "dataDirectory" | "classifierName">,
): readonly string[] => {
  const classifierDirectory = join(
    options.dataDirectory,
    "classifiers",
    createHash("sha256").update(options.classifierName).digest("hex"),
  );
  if (!existsSync(classifierDirectory)) return [];
  return readdirSync(classifierDirectory, {
    recursive: true,
    withFileTypes: true,
  })
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name))
    .filter((path) => {
      const relativePath = relative(classifierDirectory, path);
      const segments = relativePath.split(/[\\/]/u);
      const name = segments.at(-1)!.toLowerCase();
      if (name.includes("lora") || name.startsWith("model.cact")) return true;
      const candidatesIndex = segments.indexOf("candidates");
      return candidatesIndex >= 0 && name !== "training.jsonl";
    });
};

const readStoredPolicy = (
  configJson: string,
): {
  readonly requirements: DatasetRequirements;
  readonly resultBinIds: readonly string[];
} => {
  const config = JSON.parse(configJson) as {
    result?: ResultConfig;
    acceptableError?: number;
    datasetPolicy?: DatasetPolicy;
  };
  if (
    config.result === undefined ||
    config.acceptableError === undefined ||
    config.datasetPolicy?.requirements === undefined
  ) {
    throw new SwapAIError(
      "invalid_configuration",
      "Classifier has no stored dataset requirements for a held-out migration",
    );
  }
  return {
    requirements: config.datasetPolicy.requirements,
    resultBinIds: resultBins(
      config.result,
      config.acceptableError,
      config.datasetPolicy,
    ).map((bin) => bin.id),
  };
};

const requiredClassifier = (
  database: DatabaseSyncType,
  classifierName: string,
): StoredClassifier => {
  const classifier = database
    .prepare(
      `
      SELECT active_generation, config_json,
             training_attempts, examples_used_for_training,
             last_evaluated_error, last_evaluated_at
      FROM classifiers WHERE name = ?
    `,
    )
    .get(classifierName) as StoredClassifier | undefined;
  if (classifier === undefined) {
    throw new SwapAIError(
      "invalid_configuration",
      `Classifier "${classifierName}" does not exist`,
    );
  }
  return classifier;
};

const countExamples = (
  database: DatabaseSyncType,
  classifierName: string,
  generation: number,
  split: "training" | "held_out",
): number =>
  (
    database
      .prepare(
        `
        SELECT COUNT(*) AS count FROM examples
        WHERE classifier_name = ? AND generation = ? AND split = ?
      `,
      )
      .get(classifierName, generation, split) as { count: number }
  ).count;

const countRows = (
  database: DatabaseSyncType,
  table: "training_runs" | "training_evaluations" | "model_artifacts" | "shadow_evaluations",
  classifierName: string,
): number => {
  if (!hasTable(database, table)) return 0;
  const column = table === "training_evaluations" || table === "shadow_evaluations"
    ? "training_run_id IN (SELECT id FROM training_runs WHERE classifier_name = ?)"
    : "classifier_name = ?";
  return (
    database
      .prepare(`SELECT COUNT(*) AS count FROM ${table} WHERE ${column}`)
      .get(classifierName) as { count: number }
  ).count;
};

const hasTable = (database: DatabaseSyncType, table: string): boolean =>
  database
    .prepare(
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
    )
    .get(table) !== undefined;

const readPriorMigration = (
  database: DatabaseSyncType,
  classifierName: string,
): StoredMigration | null => {
  if (!hasTable(database, "legacy_held_out_purpose_migrations")) return null;
  return (
    (database
      .prepare(
        `
        SELECT plan_sha256, preserved_training_examples,
               validation_examples, representative_test_examples,
               coverage_test_examples, operator, reason,
               observed_training_attempts,
               observed_examples_used_for_training, migrated_at
        FROM legacy_held_out_purpose_migrations
        WHERE classifier_name = ?
      `,
      )
      .get(classifierName) as StoredMigration | undefined) ?? null
  );
};

const validateTargets = (targets: LegacyHeldOutPurposeTargets): void => {
  for (const [purpose, target] of Object.entries(targets)) {
    if (!Number.isSafeInteger(target) || target < 0) {
      throw new TypeError(`${purpose} target must be a non-negative integer`);
    }
  }
};

const requireAttestation = (
  attestation: LegacyHeldOutMigrationAttestation,
): void => {
  if (
    attestation.heldOutExamplesWereNeverUsedForModelSelection !== true ||
    attestation.operator.trim() === "" ||
    attestation.reason.trim() === ""
  ) {
    throw new TypeError(
      "Legacy held-out migration needs a named operator, reason, and explicit model-selection attestation",
    );
  }
};

const hashPlan = (value: unknown): string =>
  createHash("sha256").update(JSON.stringify(value)).digest("hex");

const withoutAssignments = (
  plan: PlannedMigration,
): LegacyHeldOutMigrationInspection => {
  const { assignments: _assignments, ...inspection } = plan;
  return inspection;
};

const resultFromPriorMigration = (
  plan: PlannedMigration,
): LegacyHeldOutMigrationResult => ({
  status: "already_migrated",
  classifierName: plan.classifierName,
  planSha256: plan.planSha256,
  preservedTrainingExamples: plan.preservedTrainingExamples,
  examplesByPurpose: plan.targetExamplesByPurpose,
  evidence: plan.priorMigration!,
});
