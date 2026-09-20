import { createHash, randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

import { SwapAIError } from "./errors.js";
import type {
  ResultConfig,
  TrainingExample,
  TrainingJob,
  TrainingCleanupStatus,
  TrainingResource,
} from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

interface ClassifierRow {
  active_generation: number;
  data_epoch: number;
  clear_pending: number;
}

interface TrainingExampleRow {
  id: number;
  input: string;
  result_json: string;
  result_bin: string;
  purpose:
    | "training"
    | "validation"
    | "representative_test"
    | "coverage_test";
  facets_json: string;
  input_hash: string;
}

export const createTrainingJob = (options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly result: ResultConfig;
  readonly acceptableError: number;
  readonly providerName: string;
}): TrainingJob | {
  readonly existing: true;
  readonly id: string;
  readonly datasetRevisionId: string;
  readonly status: "running" | "failed" | "rejected" | "candidate" | "promoted";
} => {
  const databasePath = join(options.dataDirectory, "swapai.sqlite");
  const database = new DatabaseSync(databasePath);
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    database.exec("BEGIN IMMEDIATE");
    try {
      const classifier = database.prepare(`
        SELECT active_generation, data_epoch, clear_pending
        FROM classifiers
        WHERE name = ?
      `).get(options.classifierName) as ClassifierRow | undefined;
      if (classifier === undefined || classifier.clear_pending === 1) {
        throw new SwapAIError(
          "storage_failed",
          `Classifier "${options.classifierName}" is unavailable`,
        );
      }
      const running = database.prepare(`
        SELECT id, dataset_revision_id FROM training_runs
        WHERE classifier_name = ? AND status = 'running'
        ORDER BY started_at DESC, id DESC
        LIMIT 1
      `).get(options.classifierName) as {
        id: string;
        dataset_revision_id: string;
      } | undefined;
      if (running !== undefined) {
        database.exec("COMMIT");
        return {
          existing: true,
          id: running.id,
          datasetRevisionId: running.dataset_revision_id,
          status: "running",
        };
      }

      const rows = database.prepare(`
        SELECT id, input, result_json, result_bin, purpose, facets_json, input_hash
        FROM examples
        WHERE classifier_name = ?
          AND generation = ?
          AND purpose != 'legacy_seen'
        ORDER BY id
      `).all(options.classifierName, classifier.active_generation) as unknown as TrainingExampleRow[];
      if (!rows.some((row) => row.purpose === "training")) {
        throw new SwapAIError(
          "not_trained",
          `Classifier "${options.classifierName}" has no training examples`,
        );
      }
      if (!rows.some((row) => row.purpose === "validation")) {
        throw new SwapAIError(
          "not_trained",
          `Classifier "${options.classifierName}" has no validation examples`,
        );
      }

      const revisionHash = createHash("sha256")
        .update(JSON.stringify({
          classifierName: options.classifierName,
          generation: classifier.active_generation,
          dataEpoch: classifier.data_epoch,
          result: options.result,
          acceptableError: options.acceptableError,
          examples: rows.map((row) => ({
            id: row.id,
            inputHash: row.input_hash,
            result: row.result_json,
            resultBin: row.result_bin,
            purpose: row.purpose,
            facets: row.facets_json,
          })),
        }))
        .digest("hex");
      const createdAt = Date.now();
      database.prepare(`
        INSERT OR IGNORE INTO dataset_revisions (
          id, classifier_name, generation, data_epoch, created_at
        ) VALUES (?, ?, ?, ?, ?)
      `).run(
        revisionHash,
        options.classifierName,
        classifier.active_generation,
        classifier.data_epoch,
        createdAt,
      );
      const insertExample = database.prepare(`
        INSERT OR IGNORE INTO dataset_revision_examples (
          dataset_revision_id, position, input, result_json, result_bin,
          purpose, facets_json, input_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `);
      rows.forEach((row, position) => {
        insertExample.run(
          revisionHash,
          position,
          row.input,
          row.result_json,
          row.result_bin,
          row.purpose,
          row.facets_json,
          row.input_hash,
        );
      });

      const existing = database.prepare(`
        SELECT id, status
        FROM training_runs
        WHERE classifier_name = ?
          AND dataset_revision_id = ?
          AND provider_name = ?
        ORDER BY started_at DESC, id DESC
        LIMIT 1
      `).get(
        options.classifierName,
        revisionHash,
        options.providerName,
      ) as {
        id: string;
        status: "running" | "failed" | "rejected" | "candidate" | "promoted";
      } | undefined;
      if (existing !== undefined) {
        database.exec("COMMIT");
        return {
          existing: true,
          id: existing.id,
          datasetRevisionId: revisionHash,
          status: existing.status,
        };
      }

      const runId = randomUUID();
      database.prepare(`
        INSERT INTO training_runs (
          id, dataset_revision_id, classifier_name, provider_name, status, started_at
        ) VALUES (?, ?, ?, ?, 'running', ?)
      `).run(
        runId,
        revisionHash,
        options.classifierName,
        options.providerName,
        createdAt,
      );
      database.exec("COMMIT");

      const outputDirectory = join(
        options.dataDirectory,
        "training-runs",
        runId,
      );
      mkdirSync(outputDirectory, { recursive: true, mode: 0o700 });
      return {
        id: runId,
        datasetRevisionId: revisionHash,
        classifierName: options.classifierName,
        generation: classifier.active_generation,
        dataEpoch: classifier.data_epoch,
        result: options.result,
        acceptableError: options.acceptableError,
        examples: rows
          .filter((row): row is TrainingExampleRow & {
            purpose: "training";
          } => row.purpose === "training")
          .map(mapTrainingExample),
        outputDirectory,
      };
    } catch (error) {
      database.exec("ROLLBACK");
      throw error;
    }
  } finally {
    database.close();
  }
};

export const retryTrainingJob = (options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly trainingRunId: string;
  readonly result: ResultConfig;
  readonly acceptableError: number;
  readonly providerName: string;
}): TrainingJob => {
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  database.exec("BEGIN IMMEDIATE");
  try {
    const failed = database.prepare(`
      SELECT r.dataset_revision_id, r.provider_name, r.cleanup_status,
             d.generation, d.data_epoch,
             c.active_generation, c.data_epoch AS current_data_epoch,
             c.clear_pending
      FROM training_runs r
      JOIN dataset_revisions d ON d.id = r.dataset_revision_id
      JOIN classifiers c ON c.name = r.classifier_name
      WHERE r.id = ? AND r.classifier_name = ?
        AND r.status IN ('failed', 'rejected')
    `).get(options.trainingRunId, options.classifierName) as {
      dataset_revision_id: string;
      provider_name: string;
      cleanup_status: TrainingCleanupStatus;
      generation: number;
      data_epoch: number;
      active_generation: number;
      current_data_epoch: number;
      clear_pending: number;
    } | undefined;
    if (failed === undefined) {
      throw new SwapAIError(
        "invalid_result",
        `Training run "${options.trainingRunId}" is not retryable`,
      );
    }
    if (failed.provider_name !== options.providerName) {
      throw new SwapAIError(
        "invalid_configuration",
        `Training run "${options.trainingRunId}" belongs to provider "${failed.provider_name}"`,
      );
    }
    if (
      failed.cleanup_status === "pending" ||
      failed.cleanup_status === "failed"
    ) {
      throw new SwapAIError(
        "service_unavailable",
        `Training run "${options.trainingRunId}" must be reconciled before retry`,
      );
    }
    if (
      failed.clear_pending !== 0 ||
      failed.generation !== failed.active_generation ||
      failed.data_epoch !== failed.current_data_epoch
    ) {
      throw new SwapAIError(
        "invalid_result",
        "Failed training can only be retried while its dataset revision is current",
      );
    }
    const newer = database.prepare(`
      SELECT id FROM training_runs
      WHERE classifier_name = ? AND dataset_revision_id = ? AND provider_name = ?
        AND id != ? AND status IN ('running', 'candidate', 'promoted')
      LIMIT 1
    `).get(
      options.classifierName,
      failed.dataset_revision_id,
      options.providerName,
      options.trainingRunId,
    );
    if (newer !== undefined) {
      throw new SwapAIError(
        "invalid_result",
        "This dataset revision already has an active or promoted training run",
      );
    }
    const rows = database.prepare(`
      SELECT position AS id, input, result_json, result_bin, purpose,
             facets_json, input_hash
      FROM dataset_revision_examples
      WHERE dataset_revision_id = ? AND purpose = 'training'
      ORDER BY position
    `).all(failed.dataset_revision_id) as unknown as TrainingExampleRow[];
    if (rows.length === 0) {
      throw new SwapAIError("not_trained", "Dataset revision has no training examples");
    }
    const runId = randomUUID();
    database.prepare(`
      INSERT INTO training_runs (
        id, dataset_revision_id, classifier_name, provider_name, status, started_at
      ) VALUES (?, ?, ?, ?, 'running', ?)
    `).run(
      runId,
      failed.dataset_revision_id,
      options.classifierName,
      options.providerName,
      Date.now(),
    );
    database.exec("COMMIT");
    const outputDirectory = join(options.dataDirectory, "training-runs", runId);
    mkdirSync(outputDirectory, { recursive: true, mode: 0o700 });
    return {
      id: runId,
      datasetRevisionId: failed.dataset_revision_id,
      classifierName: options.classifierName,
      generation: failed.generation,
      dataEpoch: failed.data_epoch,
      result: options.result,
      acceptableError: options.acceptableError,
      examples: rows.map((row) => mapTrainingExample({ ...row, purpose: "training" })),
      outputDirectory,
    };
  } catch (error) {
    if (database.isTransaction) database.exec("ROLLBACK");
    throw error;
  } finally {
    database.close();
  }
};

export const recordTrainingFailure = (options: {
  readonly dataDirectory: string;
  readonly trainingRunId: string;
  readonly error: unknown;
}): void => {
  updateTrainingRun(options.dataDirectory, options.trainingRunId, {
    status: "failed",
    failureMessage: trainingFailureMessage(options.error),
  });
};

const trainingFailureMessage = (failure: unknown): string => {
  const messages: string[] = [];
  const visited = new Set<unknown>();
  let current: unknown = failure;
  while (current !== undefined && current !== null && messages.length < 6) {
    if (visited.has(current)) break;
    visited.add(current);
    const message = current instanceof Error ? current.message : String(current);
    const normalized = message.replace(/\s+/g, " ").trim();
    if (normalized !== "" && messages.at(-1) !== normalized) messages.push(normalized);
    current = current instanceof Error ? current.cause : undefined;
  }
  return messages
    .map((message, index) => index === 0 ? message : `caused by: ${message}`)
    .join("; ")
    .slice(0, 2_000);
};

export const updateTrainingRun = (
  dataDirectory: string,
  trainingRunId: string,
  update: {
    readonly status: "failed" | "rejected" | "candidate" | "promoted";
    readonly providerRunId?: string;
    readonly costUsd?: number;
    readonly failureMessage?: string;
  },
): void => {
  const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    database.prepare(`
      UPDATE training_runs
      SET status = ?,
          provider_run_id = COALESCE(?, provider_run_id),
          cost_usd = COALESCE(?, cost_usd),
          failure_message = ?,
          finished_at = ?
      WHERE id = ? AND status = 'running'
    `).run(
      update.status,
      update.providerRunId ?? null,
      update.costUsd ?? null,
      update.failureMessage ?? null,
      Date.now(),
      trainingRunId,
    );
  } finally {
    database.close();
  }
};

export const recordTrainingLifecycle = (options: {
  readonly dataDirectory: string;
  readonly trainingRunId: string;
  readonly providerRunId?: string;
  readonly resources?: readonly TrainingResource[];
  readonly costUsd?: number;
  readonly cleanupStatus?: TrainingCleanupStatus;
  readonly cleanupMessage?: string | null;
}): void => {
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    database.prepare(`
      UPDATE training_runs
      SET provider_run_id = COALESCE(?, provider_run_id),
          resources_json = COALESCE(?, resources_json),
          cost_usd = COALESCE(?, cost_usd),
          cleanup_status = COALESCE(?, cleanup_status),
          cleanup_message = CASE
            WHEN ? IS NULL THEN cleanup_message
            ELSE ?
          END
      WHERE id = ?
    `).run(
      options.providerRunId ?? null,
      options.resources === undefined ? null : JSON.stringify(options.resources),
      options.costUsd ?? null,
      options.cleanupStatus ?? null,
      options.cleanupMessage === undefined ? null : 1,
      options.cleanupMessage ?? null,
      options.trainingRunId,
    );
  } finally {
    database.close();
  }
};

export interface CandidateEvaluationExample {
  readonly input: string;
  readonly result: TrainingExample["result"];
  readonly resultBin: string;
  readonly purpose: "validation" | "representative_test" | "coverage_test";
}

export const readCandidateEvaluationExamples = (
  dataDirectory: string,
  datasetRevisionId: string,
): readonly CandidateEvaluationExample[] => {
  const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"), {
    readOnly: true,
  });
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    const rows = database.prepare(`
      SELECT input, result_json, result_bin, purpose, facets_json, input_hash, 0 AS id
      FROM dataset_revision_examples
      WHERE dataset_revision_id = ?
        AND purpose IN ('validation', 'representative_test', 'coverage_test')
      ORDER BY position
    `).all(datasetRevisionId) as unknown as TrainingExampleRow[];
    return rows.map((row) => ({
      input: row.input,
      result: JSON.parse(row.result_json) as TrainingExample["result"],
      resultBin: row.result_bin,
      purpose: row.purpose as CandidateEvaluationExample["purpose"],
    }));
  } finally {
    database.close();
  }
};

const mapTrainingExample = (
  row: TrainingExampleRow & { purpose: "training" },
): TrainingExample => ({
  input: row.input,
  result: JSON.parse(row.result_json) as TrainingExample["result"],
  resultBin: row.result_bin,
  purpose: row.purpose,
  facets: JSON.parse(row.facets_json) as TrainingExample["facets"],
});
