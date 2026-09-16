import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import type { DatabaseSync as DatabaseSyncType } from "node:sqlite";

const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite") as typeof import("node:sqlite");

const DEFAULT_LIVE_WITHIN_MS = 10_000;

export interface ReadClassifierStatusesOptions {
  readonly dataDirectory: string;
  readonly liveWithinMs?: number;
}

export interface ClassifierStatus {
  readonly name: string;
  readonly resultType: "number" | "boolean" | "string";
  readonly retainedExamples: number;
  readonly totalExamplesLogged: number;
  readonly newExamplesSinceTraining: number;
  readonly examplesUsedForLastTraining: number;
  readonly lastEvaluatedError: number | null;
  readonly lastEvaluatedAt: number | null;
  readonly acceptableError: number;
  readonly loaded: boolean;
  readonly loadedProcessCount: number;
  readonly training: boolean;
  readonly trained: boolean;
  readonly trainingAttempts: number;
  readonly totalLocalClassifications: number;
  readonly totalRetests: number;
  readonly consecutiveRetestFailures: number;
  readonly needleVersion: string | null;
  readonly updatedAt: number;
}

interface StatusRow {
  name: string;
  config_json: string;
  total_examples_logged: number;
  new_examples_since_training: number;
  training_attempts: number;
  examples_used_for_training: number;
  last_evaluated_error: number | null;
  last_evaluated_at: number | null;
  total_local_classifications: number;
  total_retests: number;
  consecutive_retest_failures: number;
  training_lease_owner: string | null;
  training_lease_until: number | null;
  updated_at: number;
  trained: number;
  needle_version: string | null;
  retained_examples: number;
}

interface RuntimeRow {
  classifier_name: string;
  runtime_id: string;
}

export function readClassifierStatuses(
  options: ReadClassifierStatusesOptions,
): ClassifierStatus[] {
  const liveWithinMs = options.liveWithinMs ?? DEFAULT_LIVE_WITHIN_MS;
  if (!Number.isSafeInteger(liveWithinMs) || liveWithinMs <= 0) {
    throw new TypeError("liveWithinMs must be a positive integer");
  }
  const databasePath = join(options.dataDirectory, "swapai.sqlite");
  if (!existsSync(databasePath)) return [];

  const database = new DatabaseSync(databasePath, { readOnly: true });
  try {
    database.exec("PRAGMA busy_timeout = 5000");
    if (!hasTable(database, "classifiers") || !hasTable(database, "generations")) {
      return [];
    }
    const columns = tableColumns(database, "classifiers");
    const errorColumn = columns.has("last_evaluated_error")
      ? "c.last_evaluated_error"
      : "NULL";
    const evaluatedAtColumn = columns.has("last_evaluated_at")
      ? "c.last_evaluated_at"
      : "NULL";
    const rows = database.prepare(`
      SELECT
        c.name,
        c.config_json,
        c.total_examples_logged,
        c.new_examples_since_training,
        c.training_attempts,
        c.examples_used_for_training,
        ${errorColumn} AS last_evaluated_error,
        ${evaluatedAtColumn} AS last_evaluated_at,
        c.total_local_classifications,
        c.total_retests,
        c.consecutive_retest_failures,
        c.training_lease_owner,
        c.training_lease_until,
        c.updated_at,
        g.trained,
        g.needle_version,
        (
          SELECT COUNT(*)
          FROM examples e
          WHERE e.classifier_name = c.name
            AND e.generation = c.active_generation
        ) AS retained_examples
      FROM classifiers c
      JOIN generations g
        ON g.classifier_name = c.name
        AND g.generation = c.active_generation
      ORDER BY c.name COLLATE NOCASE, c.name
    `).all().map((value) => value as unknown as StatusRow);

    const now = Date.now();
    const liveAfter = now - liveWithinMs;
    const runtimes = hasTable(database, "classifier_runtimes")
      ? database.prepare(`
          SELECT classifier_name, runtime_id
          FROM classifier_runtimes
          WHERE heartbeat_at >= ?
        `).all(liveAfter).map((value) => value as unknown as RuntimeRow)
      : [];
    const liveByClassifier = new Map<string, Set<string>>();
    for (const runtime of runtimes) {
      const existing = liveByClassifier.get(runtime.classifier_name) ?? new Set<string>();
      existing.add(runtime.runtime_id);
      liveByClassifier.set(runtime.classifier_name, existing);
    }

    return rows.map((stored) => {
      const config = parseConfig(stored.config_json);
      const liveRuntimes = liveByClassifier.get(stored.name) ?? new Set<string>();
      return {
        name: stored.name,
        resultType: config.resultType,
        retainedExamples: stored.retained_examples,
        totalExamplesLogged: stored.total_examples_logged,
        newExamplesSinceTraining: stored.new_examples_since_training,
        examplesUsedForLastTraining: stored.examples_used_for_training,
        lastEvaluatedError: stored.last_evaluated_error,
        lastEvaluatedAt: stored.last_evaluated_at,
        acceptableError: config.acceptableError,
        loaded: liveRuntimes.size > 0,
        loadedProcessCount: liveRuntimes.size,
        training:
          stored.training_lease_owner !== null &&
          stored.training_lease_until !== null &&
          stored.training_lease_until > now &&
          liveRuntimes.has(stored.training_lease_owner),
        trained: stored.trained === 1,
        trainingAttempts: stored.training_attempts,
        totalLocalClassifications: stored.total_local_classifications,
        totalRetests: stored.total_retests,
        consecutiveRetestFailures: stored.consecutive_retest_failures,
        needleVersion: stored.needle_version,
        updatedAt: stored.updated_at,
      };
    });
  } finally {
    database.close();
  }
}

function parseConfig(configJson: string): {
  resultType: "number" | "boolean" | "string";
  acceptableError: number;
} {
  const config = JSON.parse(configJson) as {
    result?: { type?: unknown };
    acceptableError?: unknown;
  };
  const resultType = config.result?.type;
  if (resultType !== "number" && resultType !== "boolean" && resultType !== "string") {
    throw new Error("Stored classifier has an invalid result type");
  }
  const acceptableError = typeof config.acceptableError === "string"
    ? parsePercentage(config.acceptableError)
    : config.acceptableError;
  if (
    typeof acceptableError !== "number" ||
    !Number.isFinite(acceptableError) ||
    acceptableError < 0 ||
    acceptableError > 1
  ) {
    throw new Error("Stored classifier has an invalid acceptable error");
  }
  return { resultType, acceptableError };
}

function parsePercentage(value: string): number {
  const match = /^(\d+(?:\.\d+)?)%$/.exec(value);
  return match === null ? Number.NaN : Number(match[1]) / 100;
}

function hasTable(database: DatabaseSyncType, name: string): boolean {
  return database.prepare(`
    SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?
  `).get(name) !== undefined;
}

function tableColumns(database: DatabaseSyncType, table: string): Set<string> {
  return new Set(
    database.prepare(`PRAGMA table_info(${table})`).all().map((value) =>
      (value as { name: string }).name,
    ),
  );
}
