import { randomUUID } from "node:crypto";
import { join } from "node:path";

import { normalizeConfig, validateResult } from "./config.js";
import { averageError, resultError } from "./evaluation.js";
import { SwapAIError } from "./errors.js";
import {
  NEEDLE_VERSION,
  createNeedleRuntime,
  type LoadedNeedleModel,
  type NeedleRuntime,
} from "./runtime.js";
import { openStorage, type Storage } from "./storage.js";
import type {
  Classifier,
  InitConfig,
  NormalizedInitConfig,
  ReferenceClassifier,
  ResultConfig,
  ResultFor,
  ResultValue,
} from "./types.js";

const TRAINING_LEASE_DURATION_MS = 5 * 60 * 1000;
const TRAINING_LEASE_RENEWAL_MS = 60 * 1000;

export function init<const Config extends ResultConfig>(
  inputConfig: InitConfig<Config>,
): Classifier<ResultFor<Config>> {
  const config = normalizeConfig(inputConfig);
  const dataDirectory = config.dataDirectory ?? join(process.cwd(), ".swapai");
  const storage = openClassifierStorage(config, dataDirectory);
  const runtime = createNeedleRuntime({
    dataDirectory,
    onBackgroundError: (error) =>
      reportBackgroundError(
        config,
        new SwapAIError("service_unavailable", error.message, { cause: error }),
      ),
  });

  return createClassifier(config, storage, runtime);
}

function openClassifierStorage<const Config extends ResultConfig>(
  config: NormalizedInitConfig<Config>,
  dataDirectory: string,
): Storage {
  try {
    return openStorage({
      dataDirectory,
      name: config.name,
      maxTrainingSet: config.maxTrainingSet,
      config: {
        result: config.result,
        retrainOnCount: config.retrainOnCount,
        acceptableError: config.acceptableError,
        retestInterval: config.retestInterval,
        retestRevertOn: config.retestRevertOn,
        model: config.model,
      },
    });
  } catch (error) {
    throw new SwapAIError(
      error instanceof TypeError ? "invalid_configuration" : "storage_failed",
      error instanceof Error ? error.message : "Could not open SwapAI storage",
      { cause: error },
    );
  }
}

function createClassifier<const Config extends ResultConfig>(
  config: NormalizedInitConfig<Config>,
  storage: Storage,
  runtime: NeedleRuntime,
): Classifier<ResultFor<Config>> {
  type Result = ResultFor<Config>;

  let closed = false;
  let trained = storage.snapshot().trained;
  let loadedModel: LoadedNeedleModel | null = null;
  let operationQueue: Promise<void> = Promise.resolve();
  let trainingQueue: Promise<void> = Promise.resolve();
  let trainingScheduled = false;
  const trainingLeaseOwner = `${process.pid}:${randomUUID()}`;
  let classificationQueue: Promise<void> = Promise.resolve();
  let queuedFailure: SwapAIError | null = null;

  const saved = storage.snapshot();
  if (
    saved.trained &&
    (saved.modelPath === null || saved.needleVersion !== NEEDLE_VERSION)
  ) {
    storage.archiveAndReset();
    trained = false;
  }

  const startup = startRuntime();

  async function startRuntime(): Promise<void> {
    try {
      await runtime.ready();
      const snapshot = storage.snapshot();
      if (snapshot.trained && snapshot.modelPath !== null) {
        loadedModel = await runtime.loadModel({
          modelPath: snapshot.modelPath,
          resultConfig: config.result,
        });
      }
    } catch (error) {
      const swapAIError = toSwapAIError(
        error,
        "service_unavailable",
        "Needle could not start",
      );
      rememberBackgroundFailure(swapAIError);
    }
  }

  function rememberBackgroundFailure(error: SwapAIError): void {
    queuedFailure = error;
    reportBackgroundError(config, error);
  }

  function enqueue(operation: () => void | Promise<void>): Promise<void> {
    const result = operationQueue.then(operation);
    operationQueue = result.catch((error: unknown) => {
      rememberBackgroundFailure(
        toSwapAIError(error, "storage_failed", "Background work failed"),
      );
    });
    return result;
  }

  function persistExample(input: string, result: Result): Promise<void> {
    return enqueue(() => {
      storage.addExample(input, result);
      scheduleTraining();
    });
  }

  function shouldTrain(snapshot = storage.snapshot()): boolean {
    return (
      !snapshot.trained &&
      snapshot.examplesUsedForTraining < config.maxTrainingSet &&
      snapshot.newExamplesSinceTraining >= config.retrainOnCount
    );
  }

  function scheduleTraining(): void {
    if (trainingScheduled || !shouldTrain()) return;
    trainingScheduled = true;
    let completedAttempt = false;
    const attempt = operationQueue.then(async () => {
      completedAttempt = await trainWhenDue();
    });
    trainingQueue = attempt
      .catch((error: unknown) => {
        rememberBackgroundFailure(
          toSwapAIError(error, "service_unavailable", "Needle training failed"),
        );
      })
      .finally(() => {
        trainingScheduled = false;
        if (completedAttempt) scheduleTraining();
      });
  }

  async function trainWhenDue(): Promise<boolean> {
    if (!shouldTrain()) return false;
    if (!storage.claimTrainingLease(trainingLeaseOwner, TRAINING_LEASE_DURATION_MS)) {
      return false;
    }

    let leaseHeld = true;
    const renewTrainingLease = (): boolean => {
      if (!leaseHeld) return false;
      try {
        leaseHeld = storage.claimTrainingLease(
          trainingLeaseOwner,
          TRAINING_LEASE_DURATION_MS,
        );
      } catch {
        leaseHeld = false;
      }
      return leaseHeld;
    };
    const leaseRenewal = setInterval(
      renewTrainingLease,
      TRAINING_LEASE_RENEWAL_MS,
    );
    leaseRenewal.unref();

    try {
      const snapshot = storage.snapshot();
      if (!shouldTrain(snapshot)) return false;

      const trainingExamples = storage.listExamples("training");
      const heldOutExamples = storage.listExamples("held_out");
      storage.markTrainingAttempted(snapshot.activeExampleCount);

      if (trainingExamples.length === 0 || heldOutExamples.length === 0) {
        return true;
      }

      const candidate = await runtime.train({
        classifierName: config.name,
        generation: snapshot.activeGeneration,
        examples: trainingExamples.map(({ input, result }) => ({ input, result })),
        resultConfig: config.result,
      });
      const candidateModel = await runtime.loadModel({
        modelPath: candidate.modelPath,
        resultConfig: config.result,
      });

      try {
        const comparisons = [];
        for (const example of heldOutExamples) {
          const candidateResult = validateResult(
            config.result,
            await candidateModel.classify(example.input),
          );
          comparisons.push({
            reference: validateResult(config.result, example.result),
            candidate: candidateResult,
          });
        }

        if (averageError(config.result, comparisons) > config.acceptableError) {
          return true;
        }

        if (!renewTrainingLease()) {
          throw new SwapAIError(
            "service_unavailable",
            `Classifier "${config.name}" lost its training lease`,
          );
        }

        storage.promoteGeneration(candidate);
        if (loadedModel !== null) await loadedModel.close();
        loadedModel = candidateModel;
        trained = true;
        return true;
      } finally {
        if (loadedModel !== candidateModel) await candidateModel.close();
      }
    } finally {
      clearInterval(leaseRenewal);
      storage.releaseTrainingLease(trainingLeaseOwner);
    }
  }

  async function model(): Promise<LoadedNeedleModel> {
    await startup;
    if (!trained) {
      throw new SwapAIError(
        "not_trained",
        `Classifier "${config.name}" has not passed its held-out test`,
      );
    }
    if (loadedModel === null) {
      const snapshot = storage.snapshot();
      if (snapshot.modelPath === null) {
        throw new SwapAIError(
          "service_unavailable",
          `Classifier "${config.name}" is trained but has no saved model`,
        );
      }
      await runtime.ready();
      loadedModel = await runtime.loadModel({
        modelPath: snapshot.modelPath,
        resultConfig: config.result,
      });
    }
    return loadedModel;
  }

  async function callReference(
    input: string,
    reference: ReferenceClassifier<Result>,
  ): Promise<Result> {
    return validateResult(config.result, await reference(input));
  }

  async function fallbackToReference(
    input: string,
    reference: ReferenceClassifier<Result> | undefined,
    candidateError: unknown,
    retestDue: boolean,
  ): Promise<Result> {
    if (reference === undefined) {
      throw toSwapAIError(
        candidateError,
        "classification_failed",
        "Needle classification failed",
      );
    }
    reportBackgroundError(
      config,
      toSwapAIError(
        candidateError,
        "classification_failed",
        "Needle classification failed; the reference classifier was used",
      ),
    );
    const referenceResult = await callReference(input, reference);
    await persistExample(input, referenceResult);
    if (retestDue) {
      storage.recordLocalClassification();
      const failures = storage.recordRetest(false);
      if (failures >= config.retestRevertOn) await disableModel();
    }
    return referenceResult;
  }

  async function disableModel(): Promise<void> {
    storage.archiveAndReset();
    trained = false;
    const previousModel = loadedModel;
    loadedModel = null;
    if (previousModel !== null) await previousModel.close();
  }

  async function classifyNow(
    input: string,
    reference: ReferenceClassifier<Result> | undefined,
  ): Promise<Result> {
    assertOpen(closed);

    if (!trained) {
      if (reference === undefined) {
        throw new SwapAIError(
          "not_trained",
          `Classifier "${config.name}" has not passed its held-out test`,
        );
      }
      const referenceResult = await callReference(input, reference);
      await persistExample(input, referenceResult);
      return referenceResult;
    }

    const snapshot = storage.snapshot();
    const retestDue =
      config.retestInterval > 0 &&
      snapshot.localClassificationsSinceRetest + 1 >= config.retestInterval;

    let candidateResult: Result;
    try {
      candidateResult = validateResult(
        config.result,
        await (await model()).classify(input),
      );
    } catch (error) {
      return fallbackToReference(input, reference, error, retestDue);
    }

    if (!retestDue || reference === undefined) {
      storage.recordLocalClassification();
      return candidateResult;
    }

    const referenceResult = await callReference(input, reference);
    await persistExample(input, referenceResult);
    storage.recordLocalClassification();
    const passed =
      resultError(config.result, referenceResult, candidateResult) <=
      config.acceptableError;
    const failures = storage.recordRetest(passed);

    if (!passed && failures >= config.retestRevertOn) {
      await disableModel();
    }

    return referenceResult;
  }

  const classifier: Classifier<Result> = {
    isTrained() {
      return trained;
    },

    logClassification(input, result) {
      assertOpen(closed);
      const validResult = validateResult(config.result, result);
      void persistExample(input, validResult);
    },

    classify(input, reference) {
      assertOpen(closed);
      const task = classificationQueue.then(() => classifyNow(input, reference));
      classificationQueue = task.then(
        () => undefined,
        () => undefined,
      );
      return task;
    },

    async flush() {
      await startup;
      while (true) {
        const operations = operationQueue;
        await operations;
        const training = trainingQueue;
        await training;
        if (operations === operationQueue && training === trainingQueue) break;
      }
      if (queuedFailure !== null) {
        const failure = queuedFailure;
        queuedFailure = null;
        throw failure;
      }
    },

    async close() {
      if (closed) return;
      closed = true;
      let failure: unknown;
      try {
        await classificationQueue;
        await classifier.flush();
      } catch (error) {
        failure = error;
      } finally {
        if (loadedModel !== null) await loadedModel.close();
        loadedModel = null;
        await runtime.close();
        storage.close();
      }
      if (failure !== undefined) throw failure;
    },
  };

  return classifier;
}

function assertOpen(closed: boolean): void {
  if (closed) {
    throw new SwapAIError("service_unavailable", "Classifier is closed");
  }
}

function toSwapAIError(
  error: unknown,
  code: "classification_failed" | "service_unavailable" | "storage_failed",
  message: string,
): SwapAIError {
  if (error instanceof SwapAIError) return error;
  return new SwapAIError(code, message, { cause: error });
}

function reportBackgroundError(
  config: NormalizedInitConfig<ResultConfig>,
  error: SwapAIError,
): void {
  try {
    config.onBackgroundError?.(error);
  } catch {
    // A reporting callback must not interrupt classification or storage work.
  }
}
