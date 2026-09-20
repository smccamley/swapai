import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { chmod, copyFile, mkdir, rm, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

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
import { readCandidateEvaluationExamples } from "./training-dataset.js";
import type {
  ResultConfig,
  TrainingCandidate,
  TrainingJob,
} from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

export const evaluateCandidate = async (options: {
  readonly dataDirectory: string;
  readonly job: TrainingJob;
  readonly candidate: TrainingCandidate;
}): Promise<"candidate" | "rejected"> => {
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

  const protectedTests = readCandidateEvaluationExamples(
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
      const reference = validateResult(options.job.result, example.result);
      const prediction = {
        input: example.input,
        purpose: example.purpose,
        resultBin: example.resultBin,
        reference,
      };
      try {
        predictions.push({
          ...prediction,
          candidate: validateResult(
            options.job.result,
            await model.classify(example.input),
          ),
        });
      } catch (error) {
        if (!(error instanceof SwapAIError) || error.code !== "classification_failed") {
          throw error;
        }
        predictions.push({
          ...prediction,
          candidate: reference,
          classificationFailed: true,
        });
      }
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
  const status = recordCandidateEvaluation({
    dataDirectory: options.dataDirectory,
    job: options.job,
    candidate: options.candidate,
    evidence,
    artifact,
  });
  if (status === "rejected") {
    await deleteUnreferencedArtifact({
      dataDirectory: options.dataDirectory,
      classifierName: options.job.classifierName,
      artifact,
    });
  }
  return status;
};

const deleteUnreferencedArtifact = async (options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly artifact: { readonly sha256: string; readonly modelPath: string };
}): Promise<void> => {
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    const retained = database.prepare(`
      SELECT 1 FROM training_runs
      WHERE classifier_name = ? AND artifact_sha256 = ?
        AND status IN ('candidate', 'promoted')
      LIMIT 1
    `).get(options.classifierName, options.artifact.sha256);
    if (retained !== undefined) return;
    database.prepare(`
      DELETE FROM model_artifacts WHERE classifier_name = ? AND sha256 = ?
    `).run(options.classifierName, options.artifact.sha256);
  } finally {
    database.close();
  }
  await rm(dirname(options.artifact.modelPath), {
    recursive: true,
    force: true,
  });
};

const saveContentAddressedArtifact = async (options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly candidate: TrainingCandidate;
  readonly result: ResultConfig;
}): Promise<{ readonly sha256: string; readonly modelPath: string; readonly sizeBytes: number }> => {
  const artifactIdentity = await hashModelArtifact(
    options.candidate.modelPath,
    options.result,
  );
  const sha256 = artifactIdentity.sha256;
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
    sizeBytes: artifactIdentity.sizeBytes,
  };
};

const recordCandidateEvaluation = (options: {
  readonly dataDirectory: string;
  readonly job: TrainingJob;
  readonly candidate: TrainingCandidate;
  readonly evidence: CandidateEvaluationEvidence;
  readonly artifact: {
    readonly sha256: string;
    readonly modelPath: string;
    readonly sizeBytes: number;
  };
}): "candidate" | "rejected" => {
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function(
    "swapai_writer_version",
    { deterministic: true },
    () => 3,
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
    const candidateReady = options.evidence.passed && revisionIsCurrent;
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
    database.prepare(`
      UPDATE training_runs
      SET status = ?, provider_run_id = ?, cost_usd = ?, artifact_sha256 = ?,
          failure_message = ?, finished_at = ?
      WHERE id = ? AND status = 'running'
    `).run(
      candidateReady ? "candidate" : "rejected",
      options.candidate.providerRunId ?? null,
      options.candidate.costUsd ?? null,
      options.artifact.sha256,
      revisionIsCurrent ? null : "Dataset changed before promotion",
      Date.now(),
      options.job.id,
    );
    database.exec("COMMIT");
    return candidateReady ? "candidate" : "rejected";
  } catch (error) {
    database.exec("ROLLBACK");
    throw error;
  } finally {
    database.close();
  }
};

export const promoteCandidate = async (options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly trainingRunId: string;
  readonly acceptableError: number;
  readonly result: ResultConfig;
}): Promise<{ readonly datasetRevisionId: string }> => {
  const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    const candidateStatement = database.prepare(`
      SELECT
        r.dataset_revision_id,
        r.artifact_sha256,
        d.generation,
        d.data_epoch,
        a.model_path,
        a.needle_version
      FROM training_runs r
      JOIN dataset_revisions d ON d.id = r.dataset_revision_id
      JOIN model_artifacts a
        ON a.classifier_name = r.classifier_name
       AND a.sha256 = r.artifact_sha256
      WHERE r.id = ? AND r.classifier_name = ? AND r.status = 'candidate'
    `);
    type CandidateRow = {
      dataset_revision_id: string;
      artifact_sha256: string;
      generation: number;
      data_epoch: number;
      model_path: string;
      needle_version: string;
    };
    const candidateBeforeHash = candidateStatement.get(
      options.trainingRunId,
      options.classifierName,
    ) as CandidateRow | undefined;
    if (candidateBeforeHash === undefined) {
      throw new SwapAIError(
        "invalid_result",
        `Training run "${options.trainingRunId}" is not a promotable candidate`,
      );
    }
    let artifactIdentity: Awaited<ReturnType<typeof hashModelArtifact>>;
    try {
      artifactIdentity = await hashModelArtifact(
        candidateBeforeHash.model_path,
        options.result,
      );
    } catch (error) {
      throw new SwapAIError(
        "invalid_result",
        "Candidate artifact is missing or unreadable",
        { cause: error },
      );
    }
    if (artifactIdentity.sha256 !== candidateBeforeHash.artifact_sha256) {
      throw new SwapAIError(
        "invalid_result",
        "Candidate artifact failed SHA-256 verification",
      );
    }
    database.exec("BEGIN IMMEDIATE");
    const run = candidateStatement.get(
      options.trainingRunId,
      options.classifierName,
    ) as CandidateRow | undefined;
    if (
      run === undefined ||
      run.dataset_revision_id !== candidateBeforeHash.dataset_revision_id ||
      run.artifact_sha256 !== artifactIdentity.sha256 ||
      run.model_path !== candidateBeforeHash.model_path ||
      run.needle_version !== candidateBeforeHash.needle_version
    ) {
      throw new SwapAIError(
        "invalid_result",
        "Candidate changed while its artifact was being verified",
      );
    }
    const shadow = database.prepare(`
      SELECT example_count, total_error, failure_count
      FROM shadow_evaluations WHERE training_run_id = ?
    `).get(options.trainingRunId) as {
      example_count: number;
      total_error: number;
      failure_count: number;
    } | undefined;
    if (
      shadow === undefined ||
      shadow.example_count === 0 ||
      shadow.failure_count > 0 ||
      shadow.total_error / shadow.example_count > options.acceptableError
    ) {
      throw new SwapAIError(
        "invalid_result",
        "Candidate needs passing fresh shadow evidence before promotion",
      );
    }
    const current = database.prepare(`
      SELECT active_generation, data_epoch, clear_pending
      FROM classifiers WHERE name = ?
    `).get(options.classifierName) as {
      active_generation: number;
      data_epoch: number;
      clear_pending: number;
    } | undefined;
    if (
      current === undefined ||
      current.clear_pending !== 0 ||
      current.active_generation !== run.generation ||
      current.data_epoch !== run.data_epoch
    ) {
      throw new SwapAIError(
        "invalid_result",
        "Candidate cannot be promoted because its dataset revision is no longer current",
      );
    }
    database.prepare(`
      UPDATE generations
      SET trained = 1, model_path = ?, needle_version = ?
      WHERE classifier_name = ? AND generation = ?
    `).run(
      run.model_path,
      run.needle_version,
      options.classifierName,
      current.active_generation,
    );
    database.prepare(`
      UPDATE training_runs SET status = 'promoted'
      WHERE id = ? AND status = 'candidate'
    `).run(options.trainingRunId);
    database.exec("COMMIT");
    return { datasetRevisionId: run.dataset_revision_id };
  } catch (error) {
    if (database.isTransaction) database.exec("ROLLBACK");
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

const hashModelArtifact = async (
  modelPath: string,
  result: ResultConfig,
): Promise<{ readonly sha256: string; readonly sizeBytes: number }> => {
  const paths = [
    modelPath,
    ...(result.type === "number" ? [`${modelPath}.numbers.json`] : []),
  ];
  const hash = createHash("sha256").update("swapai-model-artifact-v1\0");
  let sizeBytes = 0;
  for (const path of paths) {
    const file = await stat(path);
    sizeBytes += file.size;
    hash.update(`${file.size}\0`);
    hash.update(await hashFile(path));
  }
  return { sha256: hash.digest("hex"), sizeBytes };
};
