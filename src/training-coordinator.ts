import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { chmod, copyFile, mkdir, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";

import {
  evaluateCandidatePredictions,
  type CandidateEvaluationEvidence,
  type CandidatePrediction,
} from "./candidate-evaluation.js";
import { validateResult } from "./config.js";
import { SwapAIError } from "./errors.js";
import {
  createNeedleRuntime,
  hasNeedleModelArtifacts,
  needleModelVersion,
} from "./runtime.js";
import { readProtectedTestExamples } from "./training-dataset.js";
import type {
  ResultConfig,
  TrainingCandidate,
  TrainingJob,
} from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

export const evaluateAndPromoteCandidate = async (options: {
  readonly dataDirectory: string;
  readonly job: TrainingJob;
  readonly candidate: TrainingCandidate;
}): Promise<"promoted" | "rejected"> => {
  const expectedVersion = needleModelVersion(options.job.result);
  if (options.candidate.needleVersion !== expectedVersion) {
    throw new SwapAIError(
      "invalid_result",
      `Trainer returned Needle ${options.candidate.needleVersion}; expected ${expectedVersion}`,
    );
  }
  if (!hasNeedleModelArtifacts({
    modelPath: options.candidate.modelPath,
    resultConfig: options.job.result,
  })) {
    throw new SwapAIError(
      "invalid_result",
      "Trainer returned an incomplete Needle model artifact",
    );
  }

  const protectedTests = readProtectedTestExamples(
    options.dataDirectory,
    options.job.datasetRevisionId,
  );
  const runtime = createNeedleRuntime({ dataDirectory: options.dataDirectory });
  const model = await runtime.loadModel({
    modelPath: options.candidate.modelPath,
    resultConfig: options.job.result,
  });
  const predictions: CandidatePrediction[] = [];
  try {
    for (const example of protectedTests) {
      predictions.push({
        input: example.input,
        purpose: example.purpose,
        resultBin: example.resultBin,
        reference: validateResult(options.job.result, example.result),
        candidate: validateResult(
          options.job.result,
          await model.classify(example.input),
        ),
      });
    }
  } finally {
    await model.close().catch(() => undefined);
    await runtime.close().catch(() => undefined);
  }

  const evidence = evaluateCandidatePredictions(
    options.job.result,
    options.job.acceptableError,
    predictions,
  );
  const artifact = await saveContentAddressedArtifact({
    dataDirectory: options.dataDirectory,
    classifierName: options.job.classifierName,
    candidate: options.candidate,
    result: options.job.result,
  });
  return recordEvaluationAndPromotion({
    dataDirectory: options.dataDirectory,
    job: options.job,
    candidate: options.candidate,
    evidence,
    artifact,
  });
};

const saveContentAddressedArtifact = async (options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly candidate: TrainingCandidate;
  readonly result: ResultConfig;
}): Promise<{ readonly sha256: string; readonly modelPath: string; readonly sizeBytes: number }> => {
  const sha256 = await hashFile(options.candidate.modelPath);
  const artifactDirectory = join(
    options.dataDirectory,
    "classifiers",
    createHash("sha256").update(options.classifierName).digest("hex"),
    "artifacts",
    "sha256",
    sha256,
  );
  const modelPath = join(artifactDirectory, "model.cact");
  await mkdir(artifactDirectory, { recursive: true, mode: 0o700 });
  await copyFile(options.candidate.modelPath, modelPath);
  await chmod(modelPath, 0o600);
  if (options.result.type === "number") {
    await copyFile(
      `${options.candidate.modelPath}.numbers.json`,
      `${modelPath}.numbers.json`,
    );
    await chmod(`${modelPath}.numbers.json`, 0o600);
  }
  return {
    sha256,
    modelPath,
    sizeBytes: (await stat(modelPath)).size,
  };
};

const recordEvaluationAndPromotion = (options: {
  readonly dataDirectory: string;
  readonly job: TrainingJob;
  readonly candidate: TrainingCandidate;
  readonly evidence: CandidateEvaluationEvidence;
  readonly artifact: {
    readonly sha256: string;
    readonly modelPath: string;
    readonly sizeBytes: number;
  };
}): "promoted" | "rejected" => {
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function(
    "swapai_writer_version",
    { deterministic: true },
    () => 2,
  );
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  database.exec("BEGIN IMMEDIATE");
  try {
    const revision = database.prepare(`
      SELECT classifier_name, generation, data_epoch
      FROM dataset_revisions
      WHERE id = ?
    `).get(options.job.datasetRevisionId) as {
      classifier_name: string;
      generation: number;
      data_epoch: number;
    } | undefined;
    const current = database.prepare(`
      SELECT active_generation, data_epoch, clear_pending
      FROM classifiers
      WHERE name = ?
    `).get(options.job.classifierName) as {
      active_generation: number;
      data_epoch: number;
      clear_pending: number;
    } | undefined;
    if (revision === undefined || current === undefined) {
      throw new SwapAIError("storage_failed", "Training dataset revision disappeared");
    }

    database.prepare(`
      INSERT OR IGNORE INTO model_artifacts (
        classifier_name, sha256, model_path, needle_version, size_bytes, created_at
      ) VALUES (?, ?, ?, ?, ?, ?)
    `).run(
      options.job.classifierName,
      options.artifact.sha256,
      options.artifact.modelPath,
      options.candidate.needleVersion,
      options.artifact.sizeBytes,
      Date.now(),
    );
    const insertMetric = database.prepare(`
      INSERT INTO training_evaluations (
        training_run_id, purpose, result_bin, example_count, error, passed
      ) VALUES (?, ?, ?, ?, ?, ?)
    `);
    for (const metric of options.evidence.metrics) {
      insertMetric.run(
        options.job.id,
        metric.purpose,
        metric.resultBin,
        metric.exampleCount,
        metric.error,
        metric.passed ? 1 : 0,
      );
    }

    const revisionIsCurrent =
      revision.classifier_name === options.job.classifierName &&
      revision.generation === current.active_generation &&
      revision.data_epoch === current.data_epoch &&
      current.clear_pending === 0;
    const promoted = options.evidence.passed && revisionIsCurrent;
    database.prepare(`
      UPDATE classifiers
      SET last_evaluated_error = ?, last_evaluated_at = ?, updated_at = ?
      WHERE name = ?
    `).run(
      options.evidence.maximumError,
      Date.now(),
      Date.now(),
      options.job.classifierName,
    );
    if (promoted) {
      database.prepare(`
        UPDATE generations
        SET trained = 1, model_path = ?, needle_version = ?
        WHERE classifier_name = ? AND generation = ?
      `).run(
        options.artifact.modelPath,
        options.candidate.needleVersion,
        options.job.classifierName,
        current.active_generation,
      );
    }
    database.prepare(`
      UPDATE training_runs
      SET status = ?, provider_run_id = ?, cost_usd = ?, artifact_sha256 = ?,
          failure_message = ?, finished_at = ?
      WHERE id = ? AND status = 'running'
    `).run(
      promoted ? "promoted" : "rejected",
      options.candidate.providerRunId ?? null,
      options.candidate.costUsd ?? null,
      options.artifact.sha256,
      revisionIsCurrent ? null : "Dataset changed before promotion",
      Date.now(),
      options.job.id,
    );
    database.exec("COMMIT");
    return promoted ? "promoted" : "rejected";
  } catch (error) {
    database.exec("ROLLBACK");
    throw error;
  } finally {
    database.close();
  }
};

const hashFile = async (path: string): Promise<string> => {
  const hash = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.once("error", reject);
    stream.once("end", resolve);
  });
  return hash.digest("hex");
};
