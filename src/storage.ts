import { createHash } from "node:crypto";
import { chmodSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import type { DatabaseSync as DatabaseSyncType } from "node:sqlite";

import type { PreparedExampleMetadata } from "./dataset-policy.js";
import type { ClassificationFacets, DatasetPurpose } from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

export type StoredResult = number | boolean | string;
export type ExampleSplit = "training" | "held_out";
export type GenerationStatus = "active" | "archived";

export interface OpenStorageOptions {
  dataDirectory: string;
  name: string;
  config: unknown;
  maxTrainingSet: number;
  prepareLegacyExampleMetadata?: (
    input: string,
    result: StoredResult,
  ) => PreparedExampleMetadata;
}

export interface StoredExample {
  id: number;
  generation: number;
  input: string;
  result: StoredResult;
  split: ExampleSplit;
  inputHash: string;
  resultBin: string;
  purpose: DatasetPurpose;
  facets: ClassificationFacets<string>;
  createdAt: number;
}

export interface StoredGeneration {
  generation: number;
  status: GenerationStatus;
  trained: boolean;
  modelPath: string | null;
  needleVersion: string | null;
  createdAt: number;
  archivedAt: number | null;
}

export interface StorageSnapshot {
  name: string;
  config: unknown;
  activeGeneration: number;
  activeExampleCount: number;
  totalExamplesLogged: number;
  newExamplesSinceTraining: number;
  trainingAttempts: number;
  examplesUsedForTraining: number;
  lastEvaluatedError: number | null;
  lastEvaluatedAt: number | null;
  localClassificationsSinceRetest: number;
  totalLocalClassifications: number;
  totalRetests: number;
  consecutiveRetestFailures: number;
  trained: boolean;
  modelPath: string | null;
  needleVersion: string | null;
  dataEpoch: number;
  clearPending: boolean;
  clearErased: boolean;
  clearArtifactGenerationMax: number | null;
  trainingLeaseOwner: string | null;
  trainingLeaseEpoch: number | null;
  trainingLeaseUntil: number | null;
}

export interface StoredRuntime {
  runtimeId: string;
  pid: number;
  startedAt: number;
  heartbeatAt: number;
}

export interface PromotedModel {
  modelPath: string;
  needleVersion: string;
}

export interface ArchiveAndResetOptions {
  readonly retainExamples?: boolean;
}

export interface Storage {
  readonly databasePath: string;
  snapshot(): StorageSnapshot;
  addExample(
    input: string,
    result: StoredResult,
    expectedDataEpoch?: number,
    metadata?: PreparedExampleMetadata,
  ): StoredExample | null;
  listExamples(split?: ExampleSplit, generation?: number): StoredExample[];
  listExamplesForTraining(
    split: ExampleSplit,
    generation: number,
    expectedDataEpoch: number,
  ): StoredExample[] | null;
  markTrainingAttempted(
    exampleCount: number,
    expectedDataEpoch?: number,
  ): boolean;
  recordEvaluation(error: number, expectedDataEpoch?: number): boolean;
  promoteGeneration(
    model: PromotedModel,
    expectedDataEpoch?: number,
  ): StoredGeneration | null;
  recordLocalClassification(expectedDataEpoch?: number): number | null;
  recordRetest(passed: boolean, expectedDataEpoch?: number): number | null;
  claimTrainingLease(
    owner: string,
    durationMs: number,
    expectedDataEpoch?: number,
  ): boolean;
  releaseTrainingLease(owner: string): void;
  registerRuntime(runtimeId: string, pid: number): void;
  heartbeatRuntime(runtimeId: string): void;
  removeRuntime(runtimeId: string): void;
  listLiveRuntimes(heartbeatAfter: number): StoredRuntime[];
  archiveAndReset(
    expectedDataEpoch?: number,
    options?: ArchiveAndResetOptions,
  ): StoredGeneration | null;
  beginClearTrainingData(): number;
  eraseTrainingData(dataEpoch: number): StoredGeneration | null;
  finishClearTrainingData(dataEpoch: number): boolean;
  clearTrainingData(): StoredGeneration;
  claimArtifactWriteLock(): boolean;
  releaseArtifactWriteLock(): void;
  listGenerations(): StoredGeneration[];
  close(): void;
}

interface ClassifierRow {
  name: string;
  config_json: string;
  active_generation: number;
  total_examples_logged: number;
  new_examples_since_training: number;
  training_attempts: number;
  examples_used_for_training: number;
  last_evaluated_error: number | null;
  last_evaluated_at: number | null;
  local_classifications_since_retest: number;
  total_local_classifications: number;
  total_retests: number;
  consecutive_retest_failures: number;
  trained: number;
  model_path: string | null;
  needle_version: string | null;
  active_example_count: number;
  data_epoch: number;
  clear_pending: number;
  clear_erased: number;
  clear_artifact_generation_max: number | null;
  training_lease_owner: string | null;
  training_lease_epoch: number | null;
  training_lease_until: number | null;
}

interface ExampleRow {
  id: number;
  generation: number;
  input: string;
  result_json: string;
  split: ExampleSplit;
  input_hash: string;
  result_bin: string;
  purpose: DatasetPurpose;
  facets_json: string;
  created_at: number;
}

interface GenerationRow {
  generation: number;
  status: GenerationStatus;
  trained: number;
  model_path: string | null;
  needle_version: string | null;
  created_at: number;
  archived_at: number | null;
}

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS classifiers (
    name TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    max_training_set INTEGER NOT NULL,
    active_generation INTEGER NOT NULL DEFAULT 1,
    total_examples_logged INTEGER NOT NULL DEFAULT 0,
    new_examples_since_training INTEGER NOT NULL DEFAULT 0,
    training_attempts INTEGER NOT NULL DEFAULT 0,
    examples_used_for_training INTEGER NOT NULL DEFAULT 0,
    last_evaluated_error REAL,
    last_evaluated_at INTEGER,
    local_classifications_since_retest INTEGER NOT NULL DEFAULT 0,
    total_local_classifications INTEGER NOT NULL DEFAULT 0,
    total_retests INTEGER NOT NULL DEFAULT 0,
    consecutive_retest_failures INTEGER NOT NULL DEFAULT 0,
    training_lease_owner TEXT,
    training_lease_until INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    data_epoch INTEGER NOT NULL DEFAULT 0,
    clear_pending INTEGER NOT NULL DEFAULT 0,
    clear_erased INTEGER NOT NULL DEFAULT 0,
    clear_artifact_generation_max INTEGER,
    training_lease_epoch INTEGER
  );

  CREATE TABLE IF NOT EXISTS generations (
    classifier_name TEXT NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    trained INTEGER NOT NULL DEFAULT 0,
    model_path TEXT,
    needle_version TEXT,
    created_at INTEGER NOT NULL,
    archived_at INTEGER,
    PRIMARY KEY (classifier_name, generation),
    FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    classifier_name TEXT NOT NULL,
    generation INTEGER NOT NULL,
    input TEXT NOT NULL,
    result_json TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('training', 'held_out')),
    input_hash TEXT NOT NULL DEFAULT '',
    result_bin TEXT NOT NULL DEFAULT 'legacy',
    purpose TEXT NOT NULL DEFAULT 'legacy_seen',
    facets_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    FOREIGN KEY (classifier_name, generation)
      REFERENCES generations(classifier_name, generation) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS examples_by_generation
    ON examples(classifier_name, generation, id);
  CREATE INDEX IF NOT EXISTS examples_by_split
    ON examples(classifier_name, generation, split, id);
  CREATE UNIQUE INDEX IF NOT EXISTS examples_by_input
    ON examples(classifier_name, generation, input_hash)
    WHERE input_hash != '';

  CREATE TABLE IF NOT EXISTS classifier_runtimes (
    classifier_name TEXT NOT NULL,
    runtime_id TEXT NOT NULL,
    pid INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    heartbeat_at INTEGER NOT NULL,
    PRIMARY KEY (classifier_name, runtime_id),
    FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS classifier_runtimes_by_heartbeat
    ON classifier_runtimes(classifier_name, heartbeat_at);

  CREATE TABLE IF NOT EXISTS dataset_revisions (
    id TEXT PRIMARY KEY,
    classifier_name TEXT NOT NULL,
    generation INTEGER NOT NULL,
    data_epoch INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS dataset_revision_examples (
    dataset_revision_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    input TEXT NOT NULL,
    result_json TEXT NOT NULL,
    result_bin TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN (
      'training', 'validation', 'representative_test', 'coverage_test'
    )),
    facets_json TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    PRIMARY KEY (dataset_revision_id, position),
    FOREIGN KEY (dataset_revision_id) REFERENCES dataset_revisions(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    dataset_revision_id TEXT NOT NULL,
    classifier_name TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'failed', 'rejected', 'promoted')),
    provider_run_id TEXT,
    cost_usd REAL,
    artifact_sha256 TEXT,
    failure_message TEXT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    FOREIGN KEY (dataset_revision_id) REFERENCES dataset_revisions(id) ON DELETE CASCADE,
    FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS training_evaluations (
    training_run_id TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('representative_test', 'coverage_test')),
    result_bin TEXT,
    example_count INTEGER NOT NULL,
    error REAL NOT NULL,
    passed INTEGER NOT NULL,
    PRIMARY KEY (training_run_id, purpose, result_bin),
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS model_artifacts (
    classifier_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    model_path TEXT NOT NULL,
    needle_version TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (classifier_name, sha256),
    FOREIGN KEY (classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
  );

  CREATE TRIGGER IF NOT EXISTS swapai_v2_classifiers_insert
    BEFORE INSERT ON classifiers
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_classifiers_update
    BEFORE UPDATE ON classifiers
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_classifiers_delete
    BEFORE DELETE ON classifiers
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_generations_insert
    BEFORE INSERT ON generations
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_generations_update
    BEFORE UPDATE ON generations
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_generations_delete
    BEFORE DELETE ON generations
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_examples_insert
    BEFORE INSERT ON examples
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_examples_update
    BEFORE UPDATE ON examples
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
  CREATE TRIGGER IF NOT EXISTS swapai_v2_examples_delete
    BEFORE DELETE ON examples
    WHEN swapai_writer_version() < 2
    BEGIN SELECT RAISE(ABORT, 'SwapAI writer is too old'); END;
`;

export function assignExampleSplit(name: string, input: string): ExampleSplit {
  const hash = createHash("sha256")
    .update(name)
    .update("\0")
    .update(input)
    .digest();

  return hash.readUInt32BE(0) % 10 < 8 ? "training" : "held_out";
}

export function openStorage(options: OpenStorageOptions): Storage {
  if (
    !Number.isSafeInteger(options.maxTrainingSet) ||
    options.maxTrainingSet <= 0
  ) {
    throw new TypeError("maxTrainingSet must be a positive integer");
  }

  mkdirSync(options.dataDirectory, { recursive: true, mode: 0o700 });
  chmodSync(options.dataDirectory, 0o700);
  const databasePath = join(options.dataDirectory, "swapai.sqlite");
  const database = new DatabaseSync(databasePath);
  let artifactLockDatabase: DatabaseSyncType | null = null;
  try {
    database.function(
      "swapai_writer_version",
      { deterministic: true },
      () => 2,
    );
    chmodSync(databasePath, 0o600);
    database.exec("PRAGMA journal_mode = WAL");
    database.exec("PRAGMA foreign_keys = ON");
    database.exec("PRAGMA secure_delete = ON");
    database.exec("PRAGMA busy_timeout = 5000");
    const secureDelete = requiredRow<{ secure_delete: number }>(
      database.prepare("PRAGMA secure_delete").get(),
    );
    if (secureDelete.secure_delete !== 1) {
      throw new Error("SQLite secure deletion could not be enabled");
    }
    database.exec(SCHEMA);
    migrateClassifierColumns(database);
    migrateExampleColumns(database);
    const lockDirectory = join(options.dataDirectory, "locks");
    mkdirSync(lockDirectory, { recursive: true, mode: 0o700 });
    chmodSync(lockDirectory, 0o700);
    const lockPath = join(
      lockDirectory,
      `${createHash("sha256").update(options.name).digest("hex")}.sqlite`,
    );
    artifactLockDatabase = new DatabaseSync(lockPath);
    chmodSync(lockPath, 0o600);
    artifactLockDatabase.exec("PRAGMA busy_timeout = 0");
    artifactLockDatabase.exec(
      "CREATE TABLE IF NOT EXISTS artifact_lock (id INTEGER PRIMARY KEY)",
    );

    const now = Date.now();
    const configJson = stringifyJson(options.config, "classifier config");
    const existing = database
      .prepare(
        `
      SELECT config_json FROM classifiers WHERE name = ?
    `,
      )
      .get(options.name) as { config_json: string } | undefined;

    if (
      existing !== undefined &&
      criticalConfigJson(existing.config_json) !==
        criticalConfigJson(configJson)
    ) {
      throw new TypeError(
        `Classifier "${options.name}" already exists with different result or behavior settings`,
      );
    }

    transaction(database, () => {
      database
        .prepare(
          `
        INSERT INTO classifiers (
          name, config_json, max_training_set, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
          config_json = excluded.config_json,
          max_training_set = excluded.max_training_set,
          updated_at = excluded.updated_at
      `,
        )
        .run(options.name, configJson, options.maxTrainingSet, now, now);

      database
        .prepare(
          `
        INSERT OR IGNORE INTO generations (
          classifier_name, generation, status, created_at
        ) VALUES (?, 1, 'active', ?)
      `,
        )
        .run(options.name, now);

      if (options.prepareLegacyExampleMetadata !== undefined) {
        adoptLegacyExamples(
          database,
          options.name,
          activeGeneration(database, options.name),
          options.prepareLegacyExampleMetadata,
        );
      }

      trimExamples(
        database,
        options.name,
        activeGeneration(database, options.name),
        options.maxTrainingSet,
      );
    });
  } catch (error) {
    try {
      artifactLockDatabase?.close();
    } catch {
      // Preserve the initialization error after attempting to close both handles.
    }
    try {
      database.close();
    } catch {
      // Preserve the initialization error after attempting to close both handles.
    }
    throw error;
  }

  const initializedArtifactLockDatabase = artifactLockDatabase;

  let closed = false;
  let artifactLockHeld = false;

  const storage: Storage = {
    databasePath,

    snapshot() {
      assertOpen(closed);
      const row = requiredRow<ClassifierRow>(
        database
          .prepare(
            `
        SELECT
          c.name,
          c.config_json,
          c.active_generation,
          c.total_examples_logged,
          c.new_examples_since_training,
          c.training_attempts,
          c.examples_used_for_training,
          c.last_evaluated_error,
          c.last_evaluated_at,
          c.local_classifications_since_retest,
          c.total_local_classifications,
          c.total_retests,
          c.consecutive_retest_failures,
          g.trained,
          g.model_path,
          g.needle_version,
          c.data_epoch,
          c.clear_pending,
          c.clear_erased,
          c.clear_artifact_generation_max,
          c.training_lease_owner,
          c.training_lease_epoch,
          c.training_lease_until,
          (
            SELECT COUNT(*)
            FROM examples e
            WHERE e.classifier_name = c.name
              AND e.generation = c.active_generation
          ) AS active_example_count
        FROM classifiers c
        JOIN generations g
          ON g.classifier_name = c.name
          AND g.generation = c.active_generation
        WHERE c.name = ?
      `,
          )
          .get(options.name),
      );

      return {
        name: row.name,
        config: JSON.parse(row.config_json) as unknown,
        activeGeneration: row.active_generation,
        activeExampleCount: row.active_example_count,
        totalExamplesLogged: row.total_examples_logged,
        newExamplesSinceTraining: row.new_examples_since_training,
        trainingAttempts: row.training_attempts,
        examplesUsedForTraining: row.examples_used_for_training,
        lastEvaluatedError: row.last_evaluated_error,
        lastEvaluatedAt: row.last_evaluated_at,
        localClassificationsSinceRetest: row.local_classifications_since_retest,
        totalLocalClassifications: row.total_local_classifications,
        totalRetests: row.total_retests,
        consecutiveRetestFailures: row.consecutive_retest_failures,
        trained: row.trained === 1,
        modelPath: row.model_path,
        needleVersion: row.needle_version,
        dataEpoch: row.data_epoch,
        clearPending: row.clear_pending === 1,
        clearErased: row.clear_erased === 1,
        clearArtifactGenerationMax: row.clear_artifact_generation_max,
        trainingLeaseOwner: row.training_lease_owner,
        trainingLeaseEpoch: row.training_lease_epoch,
        trainingLeaseUntil: row.training_lease_until,
      };
    },

    addExample(input, result, expectedDataEpoch, metadata) {
      assertOpen(closed);
      const split =
        metadata === undefined
          ? assignExampleSplit(options.name, input)
          : metadata.purpose === "training"
            ? "training"
            : "held_out";
      const inputHash = metadata?.inputHash ?? "";
      const resultBin = metadata?.resultBin ?? "legacy";
      const purpose = metadata?.purpose ?? "legacy_seen";
      const facetsJson = stringifyJson(
        metadata?.facets ?? {},
        "classification facets",
      );
      const createdAt = Date.now();
      const resultJson = stringifyJson(result, "classification result");
      let id = 0;
      let generation = 0;
      let accepted = false;

      transaction(database, () => {
        const state = classifierState(database, options.name);
        const epoch = expectedDataEpoch ?? state.data_epoch;
        if (state.data_epoch !== epoch || state.clear_pending === 1) return;
        generation = state.active_generation;
        if (metadata !== undefined) {
          database
            .prepare(
              `
            DELETE FROM examples
            WHERE classifier_name = ? AND generation = ? AND input_hash = ?
          `,
            )
            .run(options.name, generation, inputHash);
        }
        const insertion = database
          .prepare(
            `
          INSERT INTO examples (
            classifier_name, generation, input, result_json, split,
            input_hash, result_bin, purpose, facets_json, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
          )
          .run(
            options.name,
            generation,
            input,
            resultJson,
            split,
            inputHash,
            resultBin,
            purpose,
            facetsJson,
            createdAt,
          );
        id = Number(insertion.lastInsertRowid);
        accepted = true;

        database
          .prepare(
            `
          UPDATE classifiers
          SET total_examples_logged = total_examples_logged + 1,
              new_examples_since_training = new_examples_since_training + 1,
              updated_at = ?
          WHERE name = ?
        `,
          )
          .run(createdAt, options.name);

        trimExamples(
          database,
          options.name,
          generation,
          options.maxTrainingSet,
        );
      });

      return accepted
        ? {
            id,
            generation,
            input,
            result,
            split,
            inputHash,
            resultBin,
            purpose,
            facets: metadata?.facets ?? {},
            createdAt,
          }
        : null;
    },

    listExamples(split, generation) {
      assertOpen(closed);
      const selectedGeneration =
        generation ?? activeGeneration(database, options.name);
      const rows =
        split === undefined
          ? database
              .prepare(
                `
            SELECT id, generation, input, result_json, split,
                   input_hash, result_bin, purpose, facets_json, created_at
            FROM examples
            WHERE classifier_name = ? AND generation = ?
            ORDER BY id
          `,
              )
              .all(options.name, selectedGeneration)
          : database
              .prepare(
                `
            SELECT id, generation, input, result_json, split,
                   input_hash, result_bin, purpose, facets_json, created_at
            FROM examples
            WHERE classifier_name = ? AND generation = ? AND split = ?
            ORDER BY id
          `,
              )
              .all(options.name, selectedGeneration, split);

      return rows.map((value) => mapExample(row<ExampleRow>(value)));
    },

    listExamplesForTraining(split, generation, expectedDataEpoch) {
      assertOpen(closed);
      let examples: StoredExample[] | null = null;
      transaction(database, () => {
        const state = classifierState(database, options.name);
        if (
          state.data_epoch !== expectedDataEpoch ||
          state.clear_pending === 1 ||
          state.active_generation !== generation
        ) {
          return;
        }
        examples = database
          .prepare(
            `
          SELECT id, generation, input, result_json, split,
                 input_hash, result_bin, purpose, facets_json, created_at
          FROM examples
          WHERE classifier_name = ? AND generation = ? AND split = ?
          ORDER BY id
        `,
          )
          .all(options.name, generation, split)
          .map((value) => mapExample(row<ExampleRow>(value)));
      });
      return examples;
    },

    markTrainingAttempted(exampleCount, expectedDataEpoch) {
      assertOpen(closed);
      if (!Number.isSafeInteger(exampleCount) || exampleCount <= 0) {
        throw new TypeError(
          "training example count must be a positive integer",
        );
      }
      const epoch = expectedDataEpoch ?? storage.snapshot().dataEpoch;
      const result = database
        .prepare(
          `
        UPDATE classifiers
        SET new_examples_since_training = 0,
            training_attempts = training_attempts + 1,
            examples_used_for_training = ?,
            updated_at = ?
        WHERE name = ? AND data_epoch = ? AND clear_pending = 0
      `,
        )
        .run(exampleCount, Date.now(), options.name, epoch);
      return result.changes === 1;
    },

    recordEvaluation(error, expectedDataEpoch) {
      assertOpen(closed);
      if (!Number.isFinite(error) || error < 0 || error > 1) {
        throw new TypeError("evaluation error must be between 0 and 1");
      }
      const epoch = expectedDataEpoch ?? storage.snapshot().dataEpoch;
      const result = database
        .prepare(
          `
        UPDATE classifiers
        SET last_evaluated_error = ?,
            last_evaluated_at = ?,
            updated_at = ?
        WHERE name = ? AND data_epoch = ? AND clear_pending = 0
      `,
        )
        .run(error, Date.now(), Date.now(), options.name, epoch);
      return result.changes === 1;
    },

    promoteGeneration(model, expectedDataEpoch) {
      assertOpen(closed);
      let generation = 0;
      let promoted = false;
      transaction(database, () => {
        const state = classifierState(database, options.name);
        const epoch = expectedDataEpoch ?? state.data_epoch;
        if (state.data_epoch !== epoch || state.clear_pending === 1) return;
        generation = state.active_generation;
        database
          .prepare(
            `
          UPDATE generations
          SET trained = 1, model_path = ?, needle_version = ?
          WHERE classifier_name = ? AND generation = ?
        `,
          )
          .run(model.modelPath, model.needleVersion, options.name, generation);
        promoted = true;
      });
      return promoted
        ? generationByNumber(database, options.name, generation)
        : null;
    },

    recordLocalClassification(expectedDataEpoch) {
      assertOpen(closed);
      const epoch = expectedDataEpoch ?? storage.snapshot().dataEpoch;
      const result = database
        .prepare(
          `
        UPDATE classifiers
        SET local_classifications_since_retest = local_classifications_since_retest + 1,
            total_local_classifications = total_local_classifications + 1,
            updated_at = ?
        WHERE name = ? AND data_epoch = ? AND clear_pending = 0
      `,
        )
        .run(Date.now(), options.name, epoch);
      if (result.changes !== 1) return null;
      return storage.snapshot().localClassificationsSinceRetest;
    },

    recordRetest(passed, expectedDataEpoch) {
      assertOpen(closed);
      const epoch = expectedDataEpoch ?? storage.snapshot().dataEpoch;
      const result = database
        .prepare(
          `
        UPDATE classifiers
        SET local_classifications_since_retest = 0,
            total_retests = total_retests + 1,
            consecutive_retest_failures = CASE
              WHEN ? = 1 THEN 0
              ELSE consecutive_retest_failures + 1
            END,
            updated_at = ?
        WHERE name = ? AND data_epoch = ? AND clear_pending = 0
      `,
        )
        .run(passed ? 1 : 0, Date.now(), options.name, epoch);
      if (result.changes !== 1) return null;
      return storage.snapshot().consecutiveRetestFailures;
    },

    claimTrainingLease(owner, durationMs, expectedDataEpoch) {
      assertOpen(closed);
      if (
        owner.trim() === "" ||
        !Number.isSafeInteger(durationMs) ||
        durationMs <= 0
      ) {
        throw new TypeError(
          "training lease needs an owner and positive duration",
        );
      }
      const now = Date.now();
      const epoch = expectedDataEpoch ?? storage.snapshot().dataEpoch;
      const result = database
        .prepare(
          `
        UPDATE classifiers
        SET training_lease_owner = ?,
            training_lease_until = ?,
            training_lease_epoch = ?,
            updated_at = ?
        WHERE name = ?
          AND data_epoch = ?
          AND clear_pending = 0
          AND (
            training_lease_owner IS NULL
            OR training_lease_until <= ?
            OR training_lease_owner = ?
          )
      `,
        )
        .run(
          owner,
          now + durationMs,
          epoch,
          now,
          options.name,
          epoch,
          now,
          owner,
        );
      return result.changes === 1;
    },

    releaseTrainingLease(owner) {
      assertOpen(closed);
      database
        .prepare(
          `
        UPDATE classifiers
        SET training_lease_owner = NULL,
            training_lease_until = NULL,
            training_lease_epoch = NULL,
            updated_at = ?
        WHERE name = ? AND training_lease_owner = ?
      `,
        )
        .run(Date.now(), options.name, owner);
    },

    registerRuntime(runtimeId, pid) {
      assertOpen(closed);
      if (runtimeId.trim() === "" || !Number.isSafeInteger(pid) || pid <= 0) {
        throw new TypeError("runtime needs an id and positive process id");
      }
      const now = Date.now();
      database
        .prepare(
          `
        INSERT INTO classifier_runtimes (
          classifier_name, runtime_id, pid, started_at, heartbeat_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(classifier_name, runtime_id) DO UPDATE SET
          pid = excluded.pid,
          heartbeat_at = excluded.heartbeat_at
      `,
        )
        .run(options.name, runtimeId, pid, now, now);
    },

    heartbeatRuntime(runtimeId) {
      assertOpen(closed);
      database
        .prepare(
          `
        UPDATE classifier_runtimes
        SET heartbeat_at = ?
        WHERE classifier_name = ? AND runtime_id = ?
      `,
        )
        .run(Date.now(), options.name, runtimeId);
    },

    removeRuntime(runtimeId) {
      assertOpen(closed);
      database
        .prepare(
          `
        DELETE FROM classifier_runtimes
        WHERE classifier_name = ? AND runtime_id = ?
      `,
        )
        .run(options.name, runtimeId);
    },

    listLiveRuntimes(heartbeatAfter) {
      assertOpen(closed);
      if (!Number.isFinite(heartbeatAfter)) {
        throw new TypeError("heartbeat cutoff must be finite");
      }
      return database
        .prepare(
          `
        SELECT
          runtime_id AS runtimeId,
          pid,
          started_at AS startedAt,
          heartbeat_at AS heartbeatAt
        FROM classifier_runtimes
        WHERE classifier_name = ? AND heartbeat_at >= ?
        ORDER BY started_at, runtime_id
      `,
        )
        .all(options.name, heartbeatAfter)
        .map((value) => row<StoredRuntime>(value));
    },

    archiveAndReset(expectedDataEpoch, resetOptions) {
      assertOpen(closed);
      let nextGeneration = 0;
      const changedAt = Date.now();
      let reset = false;

      transaction(database, () => {
        const state = classifierState(database, options.name);
        const epoch = expectedDataEpoch ?? state.data_epoch;
        if (state.data_epoch !== epoch || state.clear_pending === 1) return;
        const currentGeneration = state.active_generation;
        database
          .prepare(
            `
          UPDATE generations
          SET status = 'archived', archived_at = ?
          WHERE classifier_name = ? AND generation = ?
        `,
          )
          .run(changedAt, options.name, currentGeneration);

        const maximum = requiredRow<{ maximum: number }>(
          database
            .prepare(
              `
          SELECT MAX(generation) AS maximum
          FROM generations
          WHERE classifier_name = ?
        `,
            )
            .get(options.name),
        );
        nextGeneration = maximum.maximum + 1;

        database
          .prepare(
            `
          INSERT INTO generations (
            classifier_name, generation, status, created_at
          ) VALUES (?, ?, 'active', ?)
        `,
          )
          .run(options.name, nextGeneration, changedAt);

        let retainedExampleCount = 0;
        if (resetOptions?.retainExamples) {
          retainedExampleCount = Number(
            database
              .prepare(
                `
            UPDATE examples
            SET generation = ?
            WHERE classifier_name = ? AND generation = ?
          `,
              )
              .run(nextGeneration, options.name, currentGeneration).changes,
          );
        }

        database
          .prepare(
            `
          UPDATE classifiers
          SET active_generation = ?,
              data_epoch = data_epoch + 1,
              new_examples_since_training = ?,
              training_attempts = 0,
              examples_used_for_training = 0,
              last_evaluated_error = NULL,
              last_evaluated_at = NULL,
              training_lease_owner = NULL,
              training_lease_until = NULL,
              training_lease_epoch = NULL,
              local_classifications_since_retest = 0,
              consecutive_retest_failures = 0,
              updated_at = ?
          WHERE name = ?
        `,
          )
          .run(nextGeneration, retainedExampleCount, changedAt, options.name);
        reset = true;
      });

      return reset
        ? generationByNumber(database, options.name, nextGeneration)
        : null;
    },

    beginClearTrainingData() {
      assertOpen(closed);
      database
        .prepare(
          `
        UPDATE classifiers
        SET data_epoch = data_epoch + 1,
            clear_pending = 1,
            clear_erased = 0,
            clear_artifact_generation_max = (
              SELECT MAX(generation)
              FROM generations
              WHERE classifier_name = ?
            ),
            updated_at = ?
        WHERE name = ?
      `,
        )
        .run(options.name, Date.now(), options.name);
      return classifierState(database, options.name).data_epoch;
    },

    eraseTrainingData(dataEpoch) {
      assertOpen(closed);
      let nextGeneration = 0;
      const changedAt = Date.now();
      let erased = false;

      transaction(database, () => {
        const state = classifierState(database, options.name);
        if (state.data_epoch !== dataEpoch || state.clear_pending !== 1) return;
        if (state.clear_erased === 1) return;
        const maximum = requiredRow<{ maximum: number }>(
          database
            .prepare(
              `
          SELECT MAX(generation) AS maximum
          FROM generations
          WHERE classifier_name = ?
        `,
            )
            .get(options.name),
        );
        nextGeneration = maximum.maximum + 1;

        database
          .prepare(
            `
          DELETE FROM examples WHERE classifier_name = ?
        `,
          )
          .run(options.name);
        database
          .prepare(
            `
          DELETE FROM dataset_revisions WHERE classifier_name = ?
        `,
          )
          .run(options.name);
        database
          .prepare(
            `
          DELETE FROM model_artifacts WHERE classifier_name = ?
        `,
          )
          .run(options.name);
        database
          .prepare(
            `
          DELETE FROM generations WHERE classifier_name = ?
        `,
          )
          .run(options.name);
        database
          .prepare(
            `
          INSERT INTO generations (
            classifier_name, generation, status, created_at
          ) VALUES (?, ?, 'active', ?)
        `,
          )
          .run(options.name, nextGeneration, changedAt);
        database
          .prepare(
            `
          UPDATE classifiers
          SET active_generation = ?,
              total_examples_logged = 0,
              new_examples_since_training = 0,
              training_attempts = 0,
              examples_used_for_training = 0,
              last_evaluated_error = NULL,
              last_evaluated_at = NULL,
              local_classifications_since_retest = 0,
              total_local_classifications = 0,
              total_retests = 0,
              consecutive_retest_failures = 0,
              training_lease_owner = NULL,
              training_lease_until = NULL,
              training_lease_epoch = NULL,
              clear_erased = 1,
              updated_at = ?
          WHERE name = ?
        `,
          )
          .run(nextGeneration, changedAt, options.name);
        erased = true;
      });

      if (!erased) return null;

      truncateWriteAheadLog(database);

      return generationByNumber(database, options.name, nextGeneration);
    },

    finishClearTrainingData(dataEpoch) {
      assertOpen(closed);
      const state = classifierState(database, options.name);
      if (
        state.data_epoch !== dataEpoch ||
        state.clear_pending !== 1 ||
        state.clear_erased !== 1
      ) {
        return false;
      }
      truncateWriteAheadLog(database);
      const result = database
        .prepare(
          `
        UPDATE classifiers
        SET clear_pending = 0, updated_at = ?
        WHERE name = ?
          AND data_epoch = ?
          AND clear_pending = 1
          AND clear_erased = 1
      `,
        )
        .run(Date.now(), options.name, dataEpoch);
      return result.changes === 1;
    },

    clearTrainingData() {
      const dataEpoch = storage.beginClearTrainingData();
      if (!storage.claimArtifactWriteLock()) {
        throw new Error("Classifier artifacts are currently being written");
      }
      try {
        const generation = storage.eraseTrainingData(dataEpoch);
        if (
          generation === null ||
          !storage.finishClearTrainingData(dataEpoch)
        ) {
          throw new Error("Training data clear was superseded");
        }
        return generation;
      } finally {
        storage.releaseArtifactWriteLock();
      }
    },

    claimArtifactWriteLock() {
      assertOpen(closed);
      if (artifactLockHeld) return true;
      try {
        initializedArtifactLockDatabase.exec("BEGIN IMMEDIATE");
        artifactLockHeld = true;
        return true;
      } catch (error) {
        if (
          typeof error === "object" &&
          error !== null &&
          "code" in error &&
          error.code === "ERR_SQLITE_ERROR" &&
          "message" in error &&
          typeof error.message === "string" &&
          /locked|busy/i.test(error.message)
        ) {
          return false;
        }
        throw error;
      }
    },

    releaseArtifactWriteLock() {
      assertOpen(closed);
      if (!artifactLockHeld) return;
      initializedArtifactLockDatabase.exec("COMMIT");
      artifactLockHeld = false;
    },

    listGenerations() {
      assertOpen(closed);
      return database
        .prepare(
          `
        SELECT
          generation, status, trained, model_path, needle_version,
          created_at, archived_at
        FROM generations
        WHERE classifier_name = ?
        ORDER BY generation
      `,
        )
        .all(options.name)
        .map((value) => mapGeneration(row<GenerationRow>(value)));
    },

    close() {
      if (closed) {
        return;
      }
      if (artifactLockHeld) {
        initializedArtifactLockDatabase.exec("ROLLBACK");
        artifactLockHeld = false;
      }
      initializedArtifactLockDatabase.close();
      closed = true;
      database.close();
    },
  };

  return storage;
}

function adoptLegacyExamples(
  database: DatabaseSyncType,
  name: string,
  generation: number,
  prepareMetadata: (
    input: string,
    result: StoredResult,
  ) => PreparedExampleMetadata,
): void {
  const examples = database
    .prepare(
      `
      SELECT id, input, result_json, split
      FROM examples
      WHERE classifier_name = ?
        AND generation = ?
        AND purpose = 'legacy_seen'
      ORDER BY id
    `,
    )
    .all(name, generation)
    .map((value) =>
      row<{
        id: number;
        input: string;
        result_json: string;
        split: ExampleSplit;
      }>(value),
    );
  const update = database.prepare(`
    UPDATE examples
    SET input_hash = ?, result_bin = ?, purpose = ?, facets_json = ?
    WHERE id = ?
      AND classifier_name = ?
      AND generation = ?
      AND purpose = 'legacy_seen'
  `);
  for (const example of examples) {
    const metadata = prepareMetadata(
      example.input,
      JSON.parse(example.result_json) as StoredResult,
    );
    update.run(
      metadata.inputHash,
      metadata.resultBin,
      example.split === "training" ? "training" : "validation",
      stringifyJson(metadata.facets, "classification facets"),
      example.id,
      name,
      generation,
    );
  }
}

function activeGeneration(database: DatabaseSyncType, name: string): number {
  const active = requiredRow<{ active_generation: number }>(
    database
      .prepare(
        `
    SELECT active_generation FROM classifiers WHERE name = ?
  `,
      )
      .get(name),
  );
  return active.active_generation;
}

function truncateWriteAheadLog(database: DatabaseSyncType): void {
  database.exec("PRAGMA busy_timeout = 0");
  let checkpoint: { busy: number; log: number; checkpointed: number };
  try {
    checkpoint = requiredRow<{
      busy: number;
      log: number;
      checkpointed: number;
    }>(database.prepare("PRAGMA wal_checkpoint(TRUNCATE)").get());
  } finally {
    database.exec("PRAGMA busy_timeout = 5000");
  }
  if (checkpoint.busy !== 0) {
    throw new Error(
      "SQLite WAL could not be truncated after clearing training data",
    );
  }
}

function classifierState(
  database: DatabaseSyncType,
  name: string,
): {
  active_generation: number;
  data_epoch: number;
  clear_pending: number;
  clear_erased: number;
  clear_artifact_generation_max: number | null;
  training_lease_owner: string | null;
  training_lease_epoch: number | null;
  training_lease_until: number | null;
} {
  return requiredRow(
    database
      .prepare(
        `
    SELECT
      active_generation,
      data_epoch,
      clear_pending,
      clear_erased,
      clear_artifact_generation_max,
      training_lease_owner,
      training_lease_epoch,
      training_lease_until
    FROM classifiers
    WHERE name = ?
  `,
      )
      .get(name),
  );
}

function migrateClassifierColumns(database: DatabaseSyncType): void {
  transaction(database, () => {
    const columns = new Set(
      database
        .prepare("PRAGMA table_info(classifiers)")
        .all()
        .map((value) => row<{ name: string }>(value).name),
    );
    if (!columns.has("data_epoch")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN data_epoch INTEGER NOT NULL DEFAULT 0",
      );
    }
    if (!columns.has("clear_pending")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN clear_pending INTEGER NOT NULL DEFAULT 0",
      );
    }
    if (!columns.has("clear_erased")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN clear_erased INTEGER NOT NULL DEFAULT 0",
      );
    }
    if (!columns.has("clear_artifact_generation_max")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN clear_artifact_generation_max INTEGER",
      );
    }
    if (!columns.has("training_lease_epoch")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN training_lease_epoch INTEGER",
      );
    }
    if (!columns.has("last_evaluated_error")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN last_evaluated_error REAL",
      );
    }
    if (!columns.has("last_evaluated_at")) {
      database.exec(
        "ALTER TABLE classifiers ADD COLUMN last_evaluated_at INTEGER",
      );
    }
  });
}

function migrateExampleColumns(database: DatabaseSyncType): void {
  transaction(database, () => {
    const columns = new Set(
      database
        .prepare("PRAGMA table_info(examples)")
        .all()
        .map((value) => row<{ name: string }>(value).name),
    );
    if (!columns.has("input_hash")) {
      database.exec(
        "ALTER TABLE examples ADD COLUMN input_hash TEXT NOT NULL DEFAULT ''",
      );
    }
    if (!columns.has("result_bin")) {
      database.exec(
        "ALTER TABLE examples ADD COLUMN result_bin TEXT NOT NULL DEFAULT 'legacy'",
      );
    }
    if (!columns.has("purpose")) {
      database.exec(
        "ALTER TABLE examples ADD COLUMN purpose TEXT NOT NULL DEFAULT 'legacy_seen'",
      );
    }
    if (!columns.has("facets_json")) {
      database.exec(
        "ALTER TABLE examples ADD COLUMN facets_json TEXT NOT NULL DEFAULT '{}'",
      );
    }
  });
}

function generationByNumber(
  database: DatabaseSyncType,
  name: string,
  generation: number,
): StoredGeneration {
  const stored = requiredRow<GenerationRow>(
    database
      .prepare(
        `
    SELECT
      generation, status, trained, model_path, needle_version,
      created_at, archived_at
    FROM generations
    WHERE classifier_name = ? AND generation = ?
  `,
      )
      .get(name, generation),
  );
  return mapGeneration(stored);
}

function trimExamples(
  database: DatabaseSyncType,
  name: string,
  generation: number,
  maxTrainingSet: number,
): void {
  const count = requiredRow<{ example_count: number }>(
    database
      .prepare(
        `
    SELECT COUNT(*) AS example_count
    FROM examples
    WHERE classifier_name = ? AND generation = ?
  `,
      )
      .get(name, generation),
  ).example_count;
  let excess = count - maxTrainingSet;
  while (excess > 0) {
    const fullest = requiredRow<{
      result_bin: string;
      purpose: DatasetPurpose;
      facets_json: string;
    }>(
      database
        .prepare(
          `
      SELECT result_bin, purpose, facets_json
      FROM examples
      WHERE classifier_name = ? AND generation = ?
      GROUP BY result_bin, purpose, facets_json
      ORDER BY COUNT(*) DESC, MIN(id)
      LIMIT 1
    `,
        )
        .get(name, generation),
    );
    database
      .prepare(
        `
      DELETE FROM examples
      WHERE id = (
        SELECT id
        FROM examples
        WHERE classifier_name = ?
          AND generation = ?
          AND result_bin = ?
          AND purpose = ?
          AND facets_json = ?
        ORDER BY id
        LIMIT 1
      )
    `,
      )
      .run(
        name,
        generation,
        fullest.result_bin,
        fullest.purpose,
        fullest.facets_json,
      );
    excess -= 1;
  }
}

function mapExample(stored: ExampleRow): StoredExample {
  return {
    id: stored.id,
    generation: stored.generation,
    input: stored.input,
    result: JSON.parse(stored.result_json) as StoredResult,
    split: stored.split,
    inputHash: stored.input_hash,
    resultBin: stored.result_bin,
    purpose: stored.purpose,
    facets: JSON.parse(stored.facets_json) as ClassificationFacets<string>,
    createdAt: stored.created_at,
  };
}

function mapGeneration(stored: GenerationRow): StoredGeneration {
  return {
    generation: stored.generation,
    status: stored.status,
    trained: stored.trained === 1,
    modelPath: stored.model_path,
    needleVersion: stored.needle_version,
    createdAt: stored.created_at,
    archivedAt: stored.archived_at,
  };
}

function transaction(database: DatabaseSyncType, action: () => void): void {
  database.exec("BEGIN IMMEDIATE");
  try {
    action();
    database.exec("COMMIT");
  } catch (error) {
    database.exec("ROLLBACK");
    throw error;
  }
}

function stringifyJson(value: unknown, description: string): string {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new TypeError(`${description} must be JSON serializable`);
  }
  return serialized;
}

function criticalConfigJson(configJson: string): string {
  const config = JSON.parse(configJson) as unknown;
  if (!isRecord(config)) {
    throw new TypeError("classifier config must be an object");
  }

  const critical: Record<string, unknown> = {};
  for (const key of [
    "result",
    "retrainOnCount",
    "acceptableError",
    "retestInterval",
    "retestRevertOn",
    "model",
  ]) {
    if (Object.hasOwn(config, key)) {
      critical[key] = config[key];
    }
  }

  if (typeof critical.acceptableError === "string") {
    const percentage = /^(\d+(?:\.\d+)?)%$/.exec(critical.acceptableError);
    if (percentage !== null) {
      critical.acceptableError = Number(percentage[1]) / 100;
    }
  }

  if (isRecord(critical.result) && Array.isArray(critical.result.values)) {
    critical.result = {
      ...critical.result,
      values: [...critical.result.values].sort(compareJsonValues),
    };
  }

  return JSON.stringify(sortObjectKeys(critical));
}

function sortObjectKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortObjectKeys);
  }
  if (!isRecord(value)) {
    return value;
  }

  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, sortObjectKeys(child)]),
  );
}

function compareJsonValues(left: unknown, right: unknown): number {
  return JSON.stringify(sortObjectKeys(left)).localeCompare(
    JSON.stringify(sortObjectKeys(right)),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertOpen(closed: boolean): void {
  if (closed) {
    throw new Error("Storage is closed");
  }
}

function row<T>(value: unknown): T {
  return value as T;
}

function requiredRow<T>(value: unknown): T {
  if (value === undefined) {
    throw new Error("Expected stored classifier data");
  }
  return row<T>(value);
}
