import { init } from "./classifier.js";
import { readClassifierInspection } from "./dataset-inspection.js";
import { createDatasetPolicy } from "./dataset-policy.js";
import {
  createTrainingJob,
  recordTrainingFailure,
} from "./training-dataset.js";
import { evaluateAndPromoteCandidate } from "./training-coordinator.js";
import type {
  ConfiguredClassifier,
  CreateClassifierConfig,
  ResultConfig,
  ResultFor,
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
    retrainOnCount: config.maxTrainingSet ?? DEFAULT_MAX_TRAINING_SET,
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

  return {
    isTrained: () => classifier.isTrained(),
    classify: (input, facets = {}) =>
      classifier.classify(input, config.reference, facets),
    logClassification: (input, result, facets = {}) =>
      classifier.logClassification(input, result, facets),
    inspect: () =>
      readClassifierInspection({
        dataDirectory,
        name: config.name,
        result: config.result,
        acceptableError: normalizedAcceptableError,
        policy: datasetPolicy,
      }),
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
      try {
        const candidate = await config.training.train(job);
        const status = await evaluateAndPromoteCandidate({
          dataDirectory,
          job,
          candidate,
        });
        return {
          status,
          trainingRunId: job.id,
          datasetRevisionId: job.datasetRevisionId,
        };
      } catch (error) {
        recordTrainingFailure({
          dataDirectory,
          trainingRunId: job.id,
          error,
        });
        throw error;
      }
    },
    clearTrainingData: () => classifier.clearTrainingData(),
    flush: () => classifier.flush(),
    close: () => classifier.close(),
  };
};
