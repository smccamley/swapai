import { Effect, Either, Runtime } from "effect";

import { SwapAIError } from "./errors.js";
import { createClassifier as createPromiseClassifier } from "./create-classifier.js";
import type {
  ClassificationFacets,
  Classifier,
  ConfiguredClassifier,
  CreateClassifierConfig,
  ResultConfig,
  ResultFor,
  ResultValue,
} from "./types.js";

class ReferenceEffectFailure<ReferenceError> {
  readonly error: ReferenceError;

  constructor(error: ReferenceError) {
    this.error = error;
  }
}

function classificationFailure(error: unknown): SwapAIError {
  if (error instanceof SwapAIError) {
    return error;
  }

  return new SwapAIError("classification_failed", "Classification failed", {
    cause: error,
  });
}

export function classify<Result extends ResultValue>(
  classifier: Classifier<Result>,
  input: string,
): Effect.Effect<Result, SwapAIError> {
  return Effect.tryPromise({
    try: () => classifier.classify(input),
    catch: classificationFailure,
  });
}

export function classifyWithReference<
  Result extends ResultValue,
  ReferenceError,
  Requirements,
>(
  classifier: Classifier<Result>,
  input: string,
  reference: (
    input: string,
  ) => Effect.Effect<Result, ReferenceError, Requirements>,
  facets: ClassificationFacets<string> = {},
): Effect.Effect<Result, SwapAIError | ReferenceError, Requirements> {
  return Effect.runtime<Requirements>().pipe(
    Effect.flatMap((runtime) =>
      Effect.tryPromise({
        try: (signal) =>
          classifier.classify(input, async (referenceInput) => {
            const result = await Runtime.runPromise(runtime)(
              Effect.either(reference(referenceInput)),
              { signal },
            );

            if (Either.isLeft(result)) {
              throw new ReferenceEffectFailure(result.left);
            }

            return result.right;
          }, facets),
        catch: (error): SwapAIError | ReferenceError => {
          if (error instanceof ReferenceEffectFailure) {
            return error.error as ReferenceError;
          }

          return classificationFailure(error);
        },
      }),
    ),
  );
}

export const createClassifierEffect = <
  const Config extends ResultConfig,
  const FacetName extends string = string,
>(
  config: CreateClassifierConfig<Config, FacetName>,
): Effect.Effect<
  ConfiguredClassifier<ResultFor<Config>, FacetName>,
  SwapAIError
> => Effect.try({
  try: () => createPromiseClassifier(config),
  catch: (error) => error instanceof SwapAIError
    ? error
    : new SwapAIError("invalid_configuration", "Could not create classifier", {
        cause: error,
      }),
});

export const classifyConfigured = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
  input: string,
  facets: ClassificationFacets<FacetName> = {},
): Effect.Effect<Result, SwapAIError> => Effect.tryPromise({
  try: () => classifier.classify(input, facets),
  catch: classificationFailure,
});

export const inspect = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.try({
  try: () => classifier.inspect(),
  catch: classificationFailure,
});

export const isTrained = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.try({
  try: () => classifier.isTrained(),
  catch: classificationFailure,
});

export const logClassification = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
  input: string,
  result: Result,
  facets: ClassificationFacets<FacetName> = {},
) => Effect.try({
  try: () => classifier.logClassification(input, result, facets),
  catch: classificationFailure,
});

export const requestTraining = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.tryPromise({
  try: () => classifier.requestTraining(),
  catch: classificationFailure,
});

export const retryTraining = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
  trainingRunId: string,
) => Effect.tryPromise({
  try: () => classifier.retryTraining(trainingRunId),
  catch: classificationFailure,
});

export const promoteCandidate = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
  trainingRunId: string,
) => Effect.tryPromise({
  try: () => classifier.promoteCandidate(trainingRunId),
  catch: classificationFailure,
});

export const reconcileTraining = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.tryPromise({
  try: () => classifier.reconcileTraining(),
  catch: classificationFailure,
});

export const erase = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.tryPromise({
  try: () => classifier.erase(),
  catch: classificationFailure,
});

export const flush = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.tryPromise({
  try: () => classifier.flush(),
  catch: classificationFailure,
});

export const close = <Result extends ResultValue, FacetName extends string>(
  classifier: ConfiguredClassifier<Result, FacetName>,
) => Effect.tryPromise({
  try: () => classifier.close(),
  catch: classificationFailure,
});
