import { createRequire } from "node:module";
import { join } from "node:path";

import { validateResult } from "./config.js";
import { resultError } from "./evaluation.js";
import {
  createNeedleRuntime,
  type LoadedNeedleModel,
  type NeedleRuntime,
} from "./runtime.js";
import type { ResultConfig, ResultFor } from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

interface CandidateRow {
  readonly id: string;
  readonly model_path: string;
}

export interface ShadowEvaluator<Result> {
  observe(input: string, reference: Result): void;
  flush(): Promise<void>;
  reset(): Promise<void>;
  close(): Promise<void>;
}

export const createShadowEvaluator = <Config extends ResultConfig>(options: {
  readonly dataDirectory: string;
  readonly classifierName: string;
  readonly result: Config;
}): ShadowEvaluator<ResultFor<Config>> => {
  let runtime: NeedleRuntime | null = null;
  let queue = Promise.resolve();
  let loadedRunId: string | null = null;
  let loadedModel: LoadedNeedleModel | null = null;

  const candidate = (): CandidateRow | undefined => {
    const database = new DatabaseSync(join(options.dataDirectory, "swapai.sqlite"), {
      readOnly: true,
    });
    database.exec("PRAGMA busy_timeout = 5000");
    try {
      return database.prepare(`
        SELECT r.id, a.model_path
        FROM training_runs r
        JOIN model_artifacts a
          ON a.classifier_name = r.classifier_name
         AND a.sha256 = r.artifact_sha256
        WHERE r.classifier_name = ? AND r.status = 'candidate'
        ORDER BY r.finished_at DESC, r.id DESC
        LIMIT 1
      `).get(options.classifierName) as CandidateRow | undefined;
    } finally {
      database.close();
    }
  };

  const evaluate = async (input: string, reference: ResultFor<Config>) => {
    const current = candidate();
    if (current === undefined) return;
    try {
      if (loadedRunId !== current.id || loadedModel === null) {
        if (loadedModel !== null) await loadedModel.close();
        runtime ??= createNeedleRuntime({ dataDirectory: options.dataDirectory });
        await runtime.ready();
        loadedModel = await runtime.loadModel({
          modelPath: current.model_path,
          resultConfig: options.result,
        });
        loadedRunId = current.id;
      }
      const predicted = validateResult(
        options.result,
        await loadedModel.classify(input),
      );
      recordSuccess(
        options.dataDirectory,
        current.id,
        resultError(options.result, reference, predicted),
      );
    } catch (error) {
      recordFailure(
        options.dataDirectory,
        current.id,
        error instanceof Error ? error.message : String(error),
      );
    }
  };

  return {
    observe(input, reference) {
      queue = queue.then(() => evaluate(input, reference));
    },
    flush: () => queue,
    async reset() {
      await queue;
      if (loadedModel !== null) await loadedModel.close();
      if (runtime !== null) await runtime.close();
      loadedModel = null;
      loadedRunId = null;
      runtime = null;
    },
    async close() {
      await queue;
      if (loadedModel !== null) await loadedModel.close();
      if (runtime !== null) await runtime.close();
    },
  };
};

const recordSuccess = (
  dataDirectory: string,
  trainingRunId: string,
  error: number,
): void => updateShadow(dataDirectory, trainingRunId, { error });

const recordFailure = (
  dataDirectory: string,
  trainingRunId: string,
  failureMessage: string,
): void => updateShadow(dataDirectory, trainingRunId, { failureMessage });

const updateShadow = (
  dataDirectory: string,
  trainingRunId: string,
  result: { readonly error: number } | { readonly failureMessage: string },
): void => {
  const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
  database.function("swapai_writer_version", { deterministic: true }, () => 3);
  database.exec("PRAGMA foreign_keys = ON");
  database.exec("PRAGMA busy_timeout = 5000");
  try {
    if (
      database.prepare("SELECT 1 FROM training_runs WHERE id = ? AND status = 'candidate'")
        .get(trainingRunId) === undefined
    ) return;
    database.prepare(`
      INSERT INTO shadow_evaluations (
        training_run_id, example_count, total_error, failure_count,
        last_failure_message, last_evaluated_at
      ) VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(training_run_id) DO UPDATE SET
        example_count = example_count + excluded.example_count,
        total_error = total_error + excluded.total_error,
        failure_count = failure_count + excluded.failure_count,
        last_failure_message = COALESCE(
          excluded.last_failure_message,
          last_failure_message
        ),
        last_evaluated_at = excluded.last_evaluated_at
    `).run(
      trainingRunId,
      "error" in result ? 1 : 0,
      "error" in result ? result.error : 0,
      "failureMessage" in result ? 1 : 0,
      "failureMessage" in result ? result.failureMessage : null,
      Date.now(),
    );
  } finally {
    database.close();
  }
};
