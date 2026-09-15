import { createHash } from "node:crypto";
import { chmodSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import type { DatabaseSync as DatabaseSyncType } from "node:sqlite";

const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite") as typeof import("node:sqlite");

export type StoredResult = number | boolean | string;
export type ExampleSplit = "training" | "held_out";
export type GenerationStatus = "active" | "archived";

export interface OpenStorageOptions {
  dataDirectory: string;
  name: string;
  config: unknown;
  maxTrainingSet: number;
}

export interface StoredExample {
  id: number;
  generation: number;
  input: string;
  result: StoredResult;
  split: ExampleSplit;
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
  localClassificationsSinceRetest: number;
  totalLocalClassifications: number;
  totalRetests: number;
  consecutiveRetestFailures: number;
  trained: boolean;
  modelPath: string | null;
  needleVersion: string | null;
}

export interface PromotedModel {
  modelPath: string;
  needleVersion: string;
}

export interface Storage {
  readonly databasePath: string;
  snapshot(): StorageSnapshot;
  addExample(input: string, result: StoredResult): StoredExample;
  listExamples(split?: ExampleSplit, generation?: number): StoredExample[];
  markTrainingAttempted(exampleCount: number): void;
  promoteGeneration(model: PromotedModel): StoredGeneration;
  recordLocalClassification(): number;
  recordRetest(passed: boolean): number;
  claimTrainingLease(owner: string, durationMs: number): boolean;
  releaseTrainingLease(owner: string): void;
  archiveAndReset(): StoredGeneration;
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
  local_classifications_since_retest: number;
  total_local_classifications: number;
  total_retests: number;
  consecutive_retest_failures: number;
  trained: number;
  model_path: string | null;
  needle_version: string | null;
  active_example_count: number;
}

interface ExampleRow {
  id: number;
  generation: number;
  input: string;
  result_json: string;
  split: ExampleSplit;
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
    local_classifications_since_retest INTEGER NOT NULL DEFAULT 0,
    total_local_classifications INTEGER NOT NULL DEFAULT 0,
    total_retests INTEGER NOT NULL DEFAULT 0,
    consecutive_retest_failures INTEGER NOT NULL DEFAULT 0,
    training_lease_owner TEXT,
    training_lease_until INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
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
    created_at INTEGER NOT NULL,
    FOREIGN KEY (classifier_name, generation)
      REFERENCES generations(classifier_name, generation) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS examples_by_generation
    ON examples(classifier_name, generation, id);
  CREATE INDEX IF NOT EXISTS examples_by_split
    ON examples(classifier_name, generation, split, id);
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
  if (!Number.isSafeInteger(options.maxTrainingSet) || options.maxTrainingSet <= 0) {
    throw new TypeError("maxTrainingSet must be a positive integer");
  }

  mkdirSync(options.dataDirectory, { recursive: true, mode: 0o700 });
  chmodSync(options.dataDirectory, 0o700);
  const databasePath = join(options.dataDirectory, "swapai.sqlite");
  const database = new DatabaseSync(databasePath);
  chmodSync(databasePath, 0o600);
  database.exec("PRAGMA journal_mode = WAL");
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  database.exec(SCHEMA);

  const now = Date.now();
  const configJson = stringifyJson(options.config, "classifier config");
  const existing = database.prepare(`
    SELECT config_json FROM classifiers WHERE name = ?
  `).get(options.name) as { config_json: string } | undefined;

  if (
    existing !== undefined
    && criticalConfigJson(existing.config_json) !== criticalConfigJson(configJson)
  ) {
    database.close();
    throw new TypeError(
      `Classifier "${options.name}" already exists with different result or behavior settings`,
    );
  }

  transaction(database, () => {
    database.prepare(`
      INSERT INTO classifiers (
        name, config_json, max_training_set, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(name) DO UPDATE SET
        config_json = excluded.config_json,
        max_training_set = excluded.max_training_set,
        updated_at = excluded.updated_at
    `).run(options.name, configJson, options.maxTrainingSet, now, now);

    database.prepare(`
      INSERT OR IGNORE INTO generations (
        classifier_name, generation, status, created_at
      ) VALUES (?, 1, 'active', ?)
    `).run(options.name, now);

    trimExamples(database, options.name, activeGeneration(database, options.name), options.maxTrainingSet);
  });

  let closed = false;

  const storage: Storage = {
    databasePath,

    snapshot() {
      assertOpen(closed);
      const row = requiredRow<ClassifierRow>(database.prepare(`
        SELECT
          c.name,
          c.config_json,
          c.active_generation,
          c.total_examples_logged,
          c.new_examples_since_training,
          c.training_attempts,
          c.examples_used_for_training,
          c.local_classifications_since_retest,
          c.total_local_classifications,
          c.total_retests,
          c.consecutive_retest_failures,
          g.trained,
          g.model_path,
          g.needle_version,
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
      `).get(options.name));

      return {
        name: row.name,
        config: JSON.parse(row.config_json) as unknown,
        activeGeneration: row.active_generation,
        activeExampleCount: row.active_example_count,
        totalExamplesLogged: row.total_examples_logged,
        newExamplesSinceTraining: row.new_examples_since_training,
        trainingAttempts: row.training_attempts,
        examplesUsedForTraining: row.examples_used_for_training,
        localClassificationsSinceRetest: row.local_classifications_since_retest,
        totalLocalClassifications: row.total_local_classifications,
        totalRetests: row.total_retests,
        consecutiveRetestFailures: row.consecutive_retest_failures,
        trained: row.trained === 1,
        modelPath: row.model_path,
        needleVersion: row.needle_version,
      };
    },

    addExample(input, result) {
      assertOpen(closed);
      const generation = activeGeneration(database, options.name);
      const split = assignExampleSplit(options.name, input);
      const createdAt = Date.now();
      const resultJson = stringifyJson(result, "classification result");
      let id = 0;

      transaction(database, () => {
        const insertion = database.prepare(`
          INSERT INTO examples (
            classifier_name, generation, input, result_json, split, created_at
          ) VALUES (?, ?, ?, ?, ?, ?)
        `).run(options.name, generation, input, resultJson, split, createdAt);
        id = Number(insertion.lastInsertRowid);

        database.prepare(`
          UPDATE classifiers
          SET total_examples_logged = total_examples_logged + 1,
              new_examples_since_training = new_examples_since_training + 1,
              updated_at = ?
          WHERE name = ?
        `).run(createdAt, options.name);

        trimExamples(database, options.name, generation, options.maxTrainingSet);
      });

      return { id, generation, input, result, split, createdAt };
    },

    listExamples(split, generation) {
      assertOpen(closed);
      const selectedGeneration = generation ?? activeGeneration(database, options.name);
      const rows = split === undefined
        ? database.prepare(`
            SELECT id, generation, input, result_json, split, created_at
            FROM examples
            WHERE classifier_name = ? AND generation = ?
            ORDER BY id
          `).all(options.name, selectedGeneration)
        : database.prepare(`
            SELECT id, generation, input, result_json, split, created_at
            FROM examples
            WHERE classifier_name = ? AND generation = ? AND split = ?
            ORDER BY id
          `).all(options.name, selectedGeneration, split);

      return rows.map((value) => mapExample(row<ExampleRow>(value)));
    },

    markTrainingAttempted(exampleCount) {
      assertOpen(closed);
      if (!Number.isSafeInteger(exampleCount) || exampleCount <= 0) {
        throw new TypeError("training example count must be a positive integer");
      }
      database.prepare(`
        UPDATE classifiers
        SET new_examples_since_training = 0,
            training_attempts = training_attempts + 1,
            examples_used_for_training = ?,
            updated_at = ?
        WHERE name = ?
      `).run(exampleCount, Date.now(), options.name);
    },

    promoteGeneration(model) {
      assertOpen(closed);
      const generation = activeGeneration(database, options.name);
      database.prepare(`
        UPDATE generations
        SET trained = 1, model_path = ?, needle_version = ?
        WHERE classifier_name = ? AND generation = ?
      `).run(model.modelPath, model.needleVersion, options.name, generation);
      return generationByNumber(database, options.name, generation);
    },

    recordLocalClassification() {
      assertOpen(closed);
      database.prepare(`
        UPDATE classifiers
        SET local_classifications_since_retest = local_classifications_since_retest + 1,
            total_local_classifications = total_local_classifications + 1,
            updated_at = ?
        WHERE name = ?
      `).run(Date.now(), options.name);
      return storage.snapshot().localClassificationsSinceRetest;
    },

    recordRetest(passed) {
      assertOpen(closed);
      database.prepare(`
        UPDATE classifiers
        SET local_classifications_since_retest = 0,
            total_retests = total_retests + 1,
            consecutive_retest_failures = CASE
              WHEN ? = 1 THEN 0
              ELSE consecutive_retest_failures + 1
            END,
            updated_at = ?
        WHERE name = ?
      `).run(passed ? 1 : 0, Date.now(), options.name);
      return storage.snapshot().consecutiveRetestFailures;
    },

    claimTrainingLease(owner, durationMs) {
      assertOpen(closed);
      if (owner.trim() === "" || !Number.isSafeInteger(durationMs) || durationMs <= 0) {
        throw new TypeError("training lease needs an owner and positive duration");
      }
      const now = Date.now();
      const result = database.prepare(`
        UPDATE classifiers
        SET training_lease_owner = ?,
            training_lease_until = ?,
            updated_at = ?
        WHERE name = ?
          AND (
            training_lease_owner IS NULL
            OR training_lease_until <= ?
            OR training_lease_owner = ?
          )
      `).run(owner, now + durationMs, now, options.name, now, owner);
      return result.changes === 1;
    },

    releaseTrainingLease(owner) {
      assertOpen(closed);
      database.prepare(`
        UPDATE classifiers
        SET training_lease_owner = NULL,
            training_lease_until = NULL,
            updated_at = ?
        WHERE name = ? AND training_lease_owner = ?
      `).run(Date.now(), options.name, owner);
    },

    archiveAndReset() {
      assertOpen(closed);
      let nextGeneration = 0;
      const changedAt = Date.now();

      transaction(database, () => {
        const currentGeneration = activeGeneration(database, options.name);
        database.prepare(`
          UPDATE generations
          SET status = 'archived', archived_at = ?
          WHERE classifier_name = ? AND generation = ?
        `).run(changedAt, options.name, currentGeneration);

        const maximum = requiredRow<{ maximum: number }>(database.prepare(`
          SELECT MAX(generation) AS maximum
          FROM generations
          WHERE classifier_name = ?
        `).get(options.name));
        nextGeneration = maximum.maximum + 1;

        database.prepare(`
          INSERT INTO generations (
            classifier_name, generation, status, created_at
          ) VALUES (?, ?, 'active', ?)
        `).run(options.name, nextGeneration, changedAt);

        database.prepare(`
          UPDATE classifiers
          SET active_generation = ?,
              new_examples_since_training = 0,
              training_attempts = 0,
              examples_used_for_training = 0,
              training_lease_owner = NULL,
              training_lease_until = NULL,
              local_classifications_since_retest = 0,
              consecutive_retest_failures = 0,
              updated_at = ?
          WHERE name = ?
        `).run(nextGeneration, changedAt, options.name);
      });

      return generationByNumber(database, options.name, nextGeneration);
    },

    listGenerations() {
      assertOpen(closed);
      return database.prepare(`
        SELECT
          generation, status, trained, model_path, needle_version,
          created_at, archived_at
        FROM generations
        WHERE classifier_name = ?
        ORDER BY generation
      `).all(options.name).map((value) => mapGeneration(row<GenerationRow>(value)));
    },

    close() {
      if (closed) {
        return;
      }
      closed = true;
      database.close();
    },
  };

  return storage;
}

function activeGeneration(database: DatabaseSyncType, name: string): number {
  const active = requiredRow<{ active_generation: number }>(database.prepare(`
    SELECT active_generation FROM classifiers WHERE name = ?
  `).get(name));
  return active.active_generation;
}

function generationByNumber(
  database: DatabaseSyncType,
  name: string,
  generation: number,
): StoredGeneration {
  const stored = requiredRow<GenerationRow>(database.prepare(`
    SELECT
      generation, status, trained, model_path, needle_version,
      created_at, archived_at
    FROM generations
    WHERE classifier_name = ? AND generation = ?
  `).get(name, generation));
  return mapGeneration(stored);
}

function trimExamples(
  database: DatabaseSyncType,
  name: string,
  generation: number,
  maxTrainingSet: number,
): void {
  database.prepare(`
    DELETE FROM examples
    WHERE classifier_name = ?
      AND generation = ?
      AND id NOT IN (
        SELECT id
        FROM examples
        WHERE classifier_name = ? AND generation = ?
        ORDER BY id DESC
        LIMIT ?
      )
  `).run(name, generation, name, generation, maxTrainingSet);
}

function mapExample(stored: ExampleRow): StoredExample {
  return {
    id: stored.id,
    generation: stored.generation,
    input: stored.input,
    result: JSON.parse(stored.result_json) as StoredResult,
    split: stored.split,
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
