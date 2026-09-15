import { validateResult } from "./config.js";
import { SwapAIError } from "./errors.js";
import type {
  ResultComparison,
  ResultConfig,
  ResultFor,
} from "./types.js";

export function resultError<const Config extends ResultConfig>(
  config: Config,
  reference: ResultFor<Config>,
  candidate: ResultFor<Config>,
): number {
  const validReference = validateResult(config, reference);
  const validCandidate = validateResult(config, candidate);

  if (config.type === "number") {
    return (
      Math.abs((validReference as number) - (validCandidate as number)) /
      (config.max - config.min)
    );
  }

  return validReference === validCandidate ? 0 : 1;
}

export function averageError<const Config extends ResultConfig>(
  config: Config,
  comparisons: readonly ResultComparison<ResultFor<Config>>[],
): number {
  if (comparisons.length === 0) {
    throw new SwapAIError(
      "invalid_result",
      "at least one held-out example is required to calculate average error",
    );
  }

  const total = comparisons.reduce(
    (sum, comparison) =>
      sum + resultError(config, comparison.reference, comparison.candidate),
    0,
  );

  return total / comparisons.length;
}
