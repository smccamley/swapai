import { Effect, Either, Runtime } from "effect";

import { SwapAIError } from "./errors.js";
import type { Classifier, ResultValue } from "./types.js";

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
          }),
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
