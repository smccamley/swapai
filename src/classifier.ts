import { randomUUID } from "node:crypto";
import { join } from "node:path";

import { normalizeConfig, validateResult } from "./config.js";
import { averageError, resultError } from "./evaluation.js";
import { SwapAIError } from "./errors.js";
import {
  createNeedleRuntime,
  hasNeedleModelArtifacts,
  NeedleModelArtifactError,
  needleModelVersion,
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
const ARTIFACT_LOCK_WAIT_MS = 30_000;

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

  const initialState = storage.snapshot();
  let closed = false;
  let closing = false;
  let closePromise: Promise<void> | null = null;
  let trained = initialState.trained && !initialState.clearPending;
  let loadedModel: LoadedNeedleModel | null = null;
  let loadedModelEpoch: number | null = null;
  let operationQueue: Promise<void> = Promise.resolve();
  let trainingQueue: Promise<void> = Promise.resolve();
  let trainingScheduled = false;
  const trainingLeaseOwner = `${process.pid}:${randomUUID()}`;
  let classificationQueue: Promise<void> = Promise.resolve();
  let queuedFailure: SwapAIError | null = null;
  let clearFailure: SwapAIError | null = null;
  let dataEpoch = initialState.dataEpoch;
  let clearedEpoch = initialState.clearPending
    ? initialState.dataEpoch - 1
    : initialState.dataEpoch;

  const saved = initialState;
  if (
    !saved.clearPending &&
    saved.trained &&
    (saved.modelPath === null ||
      saved.needleVersion !== needleModelVersion(config.result) ||
      !hasNeedleModelArtifacts({
        modelPath: saved.modelPath,
        resultConfig: config.result,
      }))
  ) {
    if (
      storage.archiveAndReset(saved.dataEpoch, { retainExamples: true }) !== null
    ) {
      const reset = storage.snapshot();
      dataEpoch = reset.dataEpoch;
      clearedEpoch = reset.dataEpoch;
      trained = false;
    }
  }

  const startup = startRuntime();

  async function startRuntime(): Promise<void> {
    const startupEpoch = dataEpoch;
    const beforeRuntime = storage.snapshot();
    if (beforeRuntime.clearPending) {
      trained = false;
      dataEpoch = beforeRuntime.dataEpoch;
      try {
        await performDurableClear(beforeRuntime.dataEpoch);
      } catch (error) {
        rememberClearFailure(error);
      }
    }
    try {
      await runtime.ready();
    } catch (error) {
      rememberBackgroundFailure(toSwapAIError(
        error,
        "service_unavailable",
        "Needle could not start",
      ));
      return;
    }
    const snapshot = storage.snapshot();
    if (snapshot.clearPending) {
      trained = false;
      dataEpoch = snapshot.dataEpoch;
      return;
    }
    if (snapshot.trained && snapshot.modelPath !== null) {
      try {
        const restoredModel = await runtime.loadModel({
          modelPath: snapshot.modelPath,
          resultConfig: config.result,
        });
        const current = storage.snapshot();
        if (
          startupEpoch === current.dataEpoch &&
          !current.clearPending &&
          current.trained
        ) {
          loadedModel = restoredModel;
          loadedModelEpoch = startupEpoch;
          trained = true;
        } else {
          await restoredModel.close();
        }
      } catch (error) {
        if (
          error instanceof NeedleModelArtifactError &&
          storage.archiveAndReset(snapshot.dataEpoch, {
            retainExamples: true,
          }) !== null
        ) {
          const reset = refreshStoredState();
          dataEpoch = reset.dataEpoch;
          clearedEpoch = reset.dataEpoch;
          trained = false;
        }
        const loadError = toSwapAIError(
          error,
          "service_unavailable",
          error instanceof NeedleModelArtifactError
            ? `Could not load classifier "${config.name}"; retraining from saved examples`
            : `Could not load classifier "${config.name}"`,
        );
        if (error instanceof NeedleModelArtifactError) {
          reportBackgroundError(config, loadError);
        } else {
          rememberBackgroundFailure(loadError);
        }
      }
    }
    scheduleTraining();
  }

  function rememberBackgroundFailure(error: SwapAIError): void {
    queuedFailure = error;
    reportBackgroundError(config, error);
  }

  function rememberClearFailure(error: unknown): void {
    clearFailure = toSwapAIError(
      error,
      "storage_failed",
      `Could not completely clear classifier "${config.name}"`,
    );
    reportBackgroundError(config, clearFailure);
  }

  function refreshStoredState() {
    const snapshot = storage.snapshot();
    dataEpoch = snapshot.dataEpoch;
    if (snapshot.clearPending) {
      trained = false;
    } else {
      trained = snapshot.trained;
      clearedEpoch = snapshot.dataEpoch;
    }
    return snapshot;
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

  function persistExample(
    input: string,
    result: Result,
    epoch = dataEpoch,
  ): Promise<void> {
    if (epoch !== dataEpoch) return Promise.resolve();
    return enqueue(() => {
      if (epoch !== dataEpoch) return;
      if (storage.addExample(input, result, epoch) !== null) scheduleTraining();
    });
  }

  function shouldTrain(snapshot = storage.snapshot()): boolean {
    return (
      clearedEpoch === dataEpoch &&
      snapshot.dataEpoch === dataEpoch &&
      !snapshot.clearPending &&
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
    const trainingEpoch = dataEpoch;
    if (!shouldTrain()) return false;
    if (
      !storage.claimTrainingLease(
        trainingLeaseOwner,
        TRAINING_LEASE_DURATION_MS,
        trainingEpoch,
      )
    ) {
      return false;
    }

      let leaseHeld = true;
      const renewTrainingLease = (): boolean => {
        if (!leaseHeld) return false;
        try {
          leaseHeld = storage.claimTrainingLease(
            trainingLeaseOwner,
            TRAINING_LEASE_DURATION_MS,
            trainingEpoch,
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

      if (
        !storage.markTrainingAttempted(
          snapshot.activeExampleCount,
          trainingEpoch,
        )
      ) {
        return false;
      }
      const trainingExamples = storage.listExamplesForTraining(
        "training",
        snapshot.activeGeneration,
        trainingEpoch,
      );
      const heldOutExamples = storage.listExamplesForTraining(
        "held_out",
        snapshot.activeGeneration,
        trainingEpoch,
      );

      if (trainingExamples === null || heldOutExamples === null) return false;
      if (trainingExamples.length === 0 || heldOutExamples.length === 0) {
        return true;
      }

      const candidate = await runtime.train({
        classifierName: config.name,
        generation: snapshot.activeGeneration,
        expectedEpoch: trainingEpoch,
        examples: trainingExamples.map(({ input, result }) => ({ input, result })),
        resultConfig: config.result,
        acceptableError: config.acceptableError,
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
          const current = refreshStoredState();
          if (
            current.dataEpoch !== trainingEpoch ||
            current.clearPending
          ) {
            await runtime.clearClassifierGenerationArtifacts(
              config.name,
              snapshot.activeGeneration,
            );
            return false;
          }
          throw new SwapAIError(
            "service_unavailable",
            `Classifier "${config.name}" lost its training lease`,
          );
        }

        if (trainingEpoch !== refreshStoredState().dataEpoch) {
          await runtime.clearClassifierGenerationArtifacts(
            config.name,
            snapshot.activeGeneration,
          );
          return false;
        }
        if (storage.promoteGeneration(candidate, trainingEpoch) === null) {
          await runtime.clearClassifierGenerationArtifacts(
            config.name,
            snapshot.activeGeneration,
          );
          return false;
        }
        if (loadedModel !== null) await loadedModel.close();
        if (trainingEpoch !== refreshStoredState().dataEpoch) {
          await runtime.clearClassifierGenerationArtifacts(
            config.name,
            snapshot.activeGeneration,
          );
          return false;
        }
        loadedModel = candidateModel;
        loadedModelEpoch = trainingEpoch;
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

  async function model(expectedEpoch: number): Promise<LoadedNeedleModel> {
    await startup;
    const current = refreshStoredState();
    if (
      !trained ||
      current.clearPending ||
      current.dataEpoch !== expectedEpoch
    ) {
      throw new SwapAIError(
        "not_trained",
        `Classifier "${config.name}" has not passed its held-out test`,
      );
    }
    if (loadedModel !== null && loadedModelEpoch !== expectedEpoch) {
      await loadedModel.close();
      loadedModel = null;
      loadedModelEpoch = null;
    }
    if (loadedModel === null) {
      if (current.modelPath === null) {
        throw new SwapAIError(
          "service_unavailable",
          `Classifier "${config.name}" is trained but has no saved model`,
        );
      }
      await runtime.ready();
      let restored: LoadedNeedleModel;
      try {
        restored = await runtime.loadModel({
          modelPath: current.modelPath,
          resultConfig: config.result,
        });
      } catch (error) {
        if (
          error instanceof NeedleModelArtifactError &&
          storage.archiveAndReset(expectedEpoch, { retainExamples: true }) !== null
        ) {
          const reset = refreshStoredState();
          dataEpoch = reset.dataEpoch;
          clearedEpoch = reset.dataEpoch;
          trained = false;
          scheduleTraining();
        }
        throw error;
      }
      const afterLoad = refreshStoredState();
      if (
        afterLoad.dataEpoch !== expectedEpoch ||
        afterLoad.clearPending ||
        !afterLoad.trained
      ) {
        await restored.close();
        throw new SwapAIError(
          "not_trained",
          `Classifier "${config.name}" has been cleared`,
        );
      }
      loadedModel = restored;
      loadedModelEpoch = expectedEpoch;
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
    epoch: number,
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
    await persistExample(input, referenceResult, epoch);
    if (retestDue && epoch === dataEpoch) {
      if (storage.recordLocalClassification(epoch) === null) return referenceResult;
      const failures = storage.recordRetest(false, epoch);
      if (failures !== null && failures >= config.retestRevertOn) {
        await disableModel(epoch);
      }
    }
    return referenceResult;
  }

  async function disableModel(epoch: number): Promise<void> {
    if (storage.archiveAndReset(epoch) === null) return;
    refreshStoredState();
    trained = false;
    const previousModel = loadedModel;
    loadedModel = null;
    loadedModelEpoch = null;
    if (previousModel !== null) await previousModel.close();
  }

  async function classifyNow(
    input: string,
    reference: ReferenceClassifier<Result> | undefined,
    epoch: number,
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
      await persistExample(input, referenceResult, epoch);
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
        await (await model(epoch)).classify(input),
      );
    } catch (error) {
      return fallbackToReference(input, reference, error, retestDue, epoch);
    }

    const afterClassification = refreshStoredState();
    if (
      afterClassification.dataEpoch !== epoch ||
      afterClassification.clearPending
    ) {
      return fallbackToReference(
        input,
        reference,
        new SwapAIError(
          "not_trained",
          `Classifier "${config.name}" was cleared during classification`,
        ),
        retestDue,
        epoch,
      );
    }

    if (!retestDue || reference === undefined) {
      if (
        epoch !== dataEpoch ||
        storage.recordLocalClassification(epoch) === null
      ) {
        return fallbackToReference(
          input,
          reference,
          new SwapAIError(
            "not_trained",
            `Classifier "${config.name}" was cleared before returning its result`,
          ),
          retestDue,
          epoch,
        );
      }
      return candidateResult;
    }

    const referenceResult = await callReference(input, reference);
    await persistExample(input, referenceResult, epoch);
    if (epoch !== dataEpoch) return referenceResult;
    if (storage.recordLocalClassification(epoch) === null) return referenceResult;
    const passed =
      resultError(config.result, referenceResult, candidateResult) <=
      config.acceptableError;
    const failures = storage.recordRetest(passed, epoch);

    if (
      !passed &&
      failures !== null &&
      failures >= config.retestRevertOn
    ) {
      await disableModel(epoch);
    }

    return referenceResult;
  }

  async function performDurableClear(clearEpoch: number): Promise<void> {
    trained = false;
    const deletionFailures: unknown[] = [];
    const throwDeletionFailures = (): void => {
      if (deletionFailures.length === 1) throw deletionFailures[0];
      if (deletionFailures.length > 1) {
        throw new AggregateError(
          deletionFailures,
          `Could not completely clear classifier "${config.name}"`,
        );
      }
    };
    const previousModel = loadedModel;
    if (previousModel !== null) {
      try {
        await previousModel.close();
        if (loadedModel === previousModel) {
          loadedModel = null;
          loadedModelEpoch = null;
        }
      } catch (error) {
        deletionFailures.push(error);
      }
    }

    const lockWaitStarted = Date.now();
    let clearState = refreshStoredState();
    while (
      clearState.clearPending &&
      clearState.dataEpoch === clearEpoch &&
      ((clearState.trainingLeaseOwner !== null &&
        clearState.trainingLeaseEpoch === null &&
        clearState.trainingLeaseUntil !== null &&
        clearState.trainingLeaseUntil > Date.now()) ||
        !storage.claimArtifactWriteLock())
    ) {
      if (Date.now() - lockWaitStarted >= ARTIFACT_LOCK_WAIT_MS) {
        throw new SwapAIError(
          "storage_failed",
          `Timed out waiting to clear classifier "${config.name}" while training was still running`,
        );
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 10));
      clearState = refreshStoredState();
    }

    if (!clearState.clearPending || clearState.dataEpoch !== clearEpoch) {
      throwDeletionFailures();
      return;
    }
    try {
      try {
        storage.eraseTrainingData(clearEpoch);
      } catch (error) {
        deletionFailures.push(error);
      }
      clearState = refreshStoredState();
      if (
        clearState.clearPending &&
        clearState.dataEpoch === clearEpoch &&
        clearState.clearErased &&
        clearState.clearArtifactGenerationMax !== null
      ) {
        try {
          await runtime.clearClassifierArtifactsThroughGeneration(
            config.name,
            clearState.clearArtifactGenerationMax,
          );
        } catch (error) {
          deletionFailures.push(error);
        }
      }
      throwDeletionFailures();

      storage.finishClearTrainingData(clearEpoch);
      const current = refreshStoredState();
      if (!current.clearPending) {
        clearFailure = null;
        clearedEpoch = current.dataEpoch;
      }
    } finally {
      storage.releaseArtifactWriteLock();
    }
  }

  async function flushNow(): Promise<void> {
    const clearFailureBeforeFlush = clearFailure;
    await startup;
    while (true) {
      const operations = operationQueue;
      await operations;
      const training = trainingQueue;
      await training;
      if (operations === operationQueue && training === trainingQueue) break;
    }
    if (
      clearFailure !== null &&
      clearFailure !== clearFailureBeforeFlush
    ) {
      throw clearFailure;
    }
    const pending = refreshStoredState();
    if (pending.clearPending || clearFailureBeforeFlush !== null) {
      try {
        await performDurableClear(pending.dataEpoch);
        if (!refreshStoredState().clearPending) clearFailure = null;
      } catch (error) {
        rememberClearFailure(error);
      }
    }
    if (refreshStoredState().clearPending) {
      clearFailure ??= new SwapAIError(
        "storage_failed",
        `Could not completely clear classifier "${config.name}"`,
      );
      throw clearFailure;
    }
    if (clearFailure !== null) throw clearFailure;
    if (queuedFailure !== null) {
      const failure = queuedFailure;
      queuedFailure = null;
      throw failure;
    }
  }

  const classifier: Classifier<Result> = {
    isTrained() {
      assertOpen(closed || closing);
      refreshStoredState();
      return trained;
    },

    logClassification(input, result) {
      assertOpen(closed || closing);
      const validResult = validateResult(config.result, result);
      const epoch = refreshStoredState().dataEpoch;
      void persistExample(input, validResult, epoch);
    },

    clearTrainingData() {
      assertOpen(closed || closing);
      trained = false;
      const clearEpoch = storage.beginClearTrainingData();
      dataEpoch = clearEpoch;
      const trainingBeforeClear = trainingQueue;
      const classificationsBeforeClear = classificationQueue;
      void enqueue(async () => {
        await startup;
        await trainingBeforeClear;
        await classificationsBeforeClear;
        try {
          await performDurableClear(clearEpoch);
        } catch (error) {
          rememberClearFailure(error);
        }
      });
    },

    classify(input, reference) {
      assertOpen(closed || closing);
      const epoch = refreshStoredState().dataEpoch;
      const task = classificationQueue.then(() =>
        classifyNow(input, reference, epoch),
      );
      classificationQueue = task.then(
        () => undefined,
        () => undefined,
      );
      return task;
    },

    async flush() {
      assertOpen(closed || closing);
      await flushNow();
    },

    close() {
      if (closed) return Promise.resolve();
      if (closePromise !== null) return closePromise;
      closing = true;
      closePromise = (async () => {
        try {
          await classificationQueue;
          await flushNow();
        } catch (error) {
          closing = false;
          closePromise = null;
          throw error;
        }
        const failures: unknown[] = [];
        if (loadedModel !== null) {
          try {
            await loadedModel.close();
          } catch (error) {
            failures.push(error);
          }
        }
        loadedModel = null;
        loadedModelEpoch = null;
        try {
          await runtime.close();
        } catch (error) {
          failures.push(error);
        }
        try {
          storage.close();
        } catch (error) {
          failures.push(error);
        }
        closed = true;
        closing = false;
        if (failures.length === 1) throw failures[0];
        if (failures.length > 1) {
          throw new AggregateError(
            failures,
            `Could not completely close classifier "${config.name}"`,
          );
        }
      })();
      return closePromise;
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
