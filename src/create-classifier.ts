import { init } from "./classifier.js";
import { rm } from "node:fs/promises";
import { readClassifierInspection } from "./dataset-inspection.js";
import { createDatasetPolicy } from "./dataset-policy.js";
import {
  createTrainingJob,
  recordTrainingLifecycle,
  recordTrainingFailure,
  retryTrainingJob,
} from "./training-dataset.js";
import {
  evaluateCandidate,
  promoteCandidate,
} from "./training-coordinator.js";
import type {
  ConfiguredClassifier,
  CreateClassifierConfig,
  ResultConfig,
  ResultFor,
  TrainingCompletionResult,
  TrainingJob,
  TrainingLifecycleReporter,
  TrainingRunInspection,
} from "./types.js";

const DEFAULT_MAX_TRAINING_SET = 10_000;

export const createClassifier = <
  const Config extends ResultConfig,
  const FacetName extends string = string,
>(
  config: CreateClassifierConfig<Config, FacetName>,
): ConfiguredClassifier<ResultFor<Config>, FacetName> => {
  const datasetPolicy = createDatasetPolicy(config.result, {
    ...(config.decisionBoundaries === undefined
      ? {}
      : { decisionBoundaries: config.decisionBoundaries }),
    ...(config.facets === undefined ? {} : { facets: config.facets }),
    ...(config.datasetRequirements === undefined
      ? {}
      : { requirements: config.datasetRequirements }),
  });
  const acceptableError = config.acceptableError ?? "10%";
  const normalizedAcceptableError = typeof acceptableError === "number"
    ? acceptableError
    : Number(acceptableError.slice(0, -1)) / 100;
  const dataDirectory = config.dataDirectory ?? `${process.cwd()}/.swapai`;

  const classifier = init({
    name: config.name,
    result: config.result,
    retrainOnCount: Math.min(50, config.maxTrainingSet ?? DEFAULT_MAX_TRAINING_SET),
    acceptableError,
    retestInterval: 100,
    retestRevertOn: 3,
    model: "needle2",
    maxTrainingSet: config.maxTrainingSet ?? DEFAULT_MAX_TRAINING_SET,
    automaticTraining: false,
    datasetPolicy,
    ...(config.dataDirectory === undefined
      ? {}
      : { dataDirectory: config.dataDirectory }),
    ...(config.onBackgroundError === undefined
      ? {}
      : { onBackgroundError: config.onBackgroundError }),
  });

  const lifecycleFor = (trainingRunId: string): TrainingLifecycleReporter => ({
    recordProviderRun: ({ providerRunId, resources }) =>
      recordTrainingLifecycle({
        dataDirectory,
        trainingRunId,
        providerRunId,
        resources,
        cleanupStatus: "pending",
      }),
    recordCleanup: ({ status, message }) =>
      recordTrainingLifecycle({
        dataDirectory,
        trainingRunId,
        cleanupStatus: status,
        cleanupMessage: message ?? null,
      }),
  });

  const inspect = () => readClassifierInspection({
    dataDirectory,
    name: config.name,
    result: config.result,
    acceptableError: normalizedAcceptableError,
    policy: datasetPolicy,
  });

  const reconcileTraining = async (): Promise<readonly TrainingRunInspection[]> => {
    const unfinished = inspect().trainingRuns.filter((run) =>
      run.status === "running" ||
      run.cleanup.status === "pending" ||
      run.cleanup.status === "failed"
    );
    for (const run of unfinished) {
      if (config.training?.name !== run.provider || config.training.reconcile === undefined) {
        continue;
      }
      const result = await config.training.reconcile(run, lifecycleFor(run.id));
      if (result.status === "failed") {
        recordTrainingFailure({
          dataDirectory,
          trainingRunId: run.id,
          error: new Error(result.failureMessage ?? "Training provider run failed"),
        });
      }
    }
    return inspect().trainingRuns;
  };

  const executeTrainingJob = async (
    job: TrainingJob,
  ): Promise<TrainingCompletionResult> => {
    if (config.training === undefined) {
      throw new TypeError(`Classifier "${config.name}" needs a training provider`);
    }
    try {
      const candidate = await config.training.train(job, lifecycleFor(job.id));
      const status = await evaluateCandidate({ dataDirectory, job, candidate });
      return {
        status,
        trainingRunId: job.id,
        datasetRevisionId: job.datasetRevisionId,
      };
    } catch (error) {
      recordTrainingFailure({ dataDirectory, trainingRunId: job.id, error });
      throw error;
    } finally {
      await rm(job.outputDirectory, { recursive: true, force: true });
    }
  };

  return {
    isTrained: () => classifier.isTrained(),
    classify: async (input, facets = {}) => {
      return classifier.classify(input, config.reference, facets);
    },
    logClassification: (input, result, facets = {}) => {
      classifier.logClassification(input, result, facets);
    },
    inspect,
    requestTraining: async () => {
      await classifier.flush();
      const inspection = readClassifierInspection({
        dataDirectory,
        name: config.name,
        result: config.result,
        acceptableError: normalizedAcceptableError,
        policy: datasetPolicy,
      });
      if (!inspection.readyForTraining) {
        return { status: "not_ready", deficits: inspection.deficits };
      }
      if (config.training === undefined) {
        throw new TypeError(
          `Classifier "${config.name}" needs a training provider`,
        );
      }
      const job = createTrainingJob({
        dataDirectory,
        classifierName: config.name,
        result: config.result,
        acceptableError: normalizedAcceptableError,
        providerName: config.training.name,
      });
      if ("existing" in job) {
        return {
          status: job.status === "running"
            ? "already_running"
            : job.status === "promoted"
              ? "already_promoted"
              : job.status === "failed"
                ? "already_failed"
                : job.status,
          trainingRunId: job.id,
          datasetRevisionId: job.datasetRevisionId,
        };
      }
      return executeTrainingJob(job);
    },
    retryTraining: async (trainingRunId) => {
      await classifier.flush();
      if (config.training === undefined) {
        throw new TypeError(`Classifier "${config.name}" needs a training provider`);
      }
      return executeTrainingJob(retryTrainingJob({
        dataDirectory,
        classifierName: config.name,
        trainingRunId,
        result: config.result,
        acceptableError: normalizedAcceptableError,
        providerName: config.training.name,
      }));
    },
    promoteCandidate: async (trainingRunId) => {
      await classifier.flush();
      const promoted = await promoteCandidate({
        dataDirectory,
        classifierName: config.name,
        trainingRunId,
        acceptableError: normalizedAcceptableError,
        result: config.result,
      });
      return {
        status: "promoted",
        trainingRunId,
        datasetRevisionId: promoted.datasetRevisionId,
      };
    },
    reconcileTraining,
    erase: async () => {
      await classifier.flush();
      const runs = inspect().trainingRuns;
      for (const run of runs.filter((candidate) =>
        candidate.status === "running" ||
        candidate.cleanup.status === "pending" ||
        candidate.cleanup.status === "failed"
      )) {
        if (config.training?.name !== run.provider || config.training.cancel === undefined) {
          throw new TypeError(
            `Training run "${run.id}" must be cancelled by provider "${run.provider}" before erasure`,
          );
        }
        await config.training.cancel(run, lifecycleFor(run.id));
        if (run.status === "running") {
          recordTrainingFailure({
            dataDirectory,
            trainingRunId: run.id,
            error: new Error("Training run cancelled for classifier erasure"),
          });
        }
      }
      await classifier.erase();
    },
    flush: async () => {
      await classifier.flush();
    },
    close: async () => {
      await classifier.close();
    },
  };
};
