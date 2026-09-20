import { resultError } from "./evaluation.js";
import type {
  ResultComparison,
  ResultConfig,
  ResultValue,
} from "./types.js";

export interface CandidatePrediction extends ResultComparison<ResultValue> {
  readonly input: string;
  readonly purpose: "validation" | "representative_test" | "coverage_test";
  readonly resultBin: string;
  readonly classificationFailed?: boolean;
}

export interface CandidateEvaluationMetric {
  readonly purpose: CandidatePrediction["purpose"];
  readonly resultBin: string | null;
  readonly exampleCount: number;
  readonly error: number;
  readonly passed: boolean;
}

export interface CandidateEvaluationEvidence {
  readonly passed: boolean;
  readonly maximumError: number;
  readonly metrics: readonly CandidateEvaluationMetric[];
}

export const evaluateCandidatePredictions = (
  result: ResultConfig,
  acceptableError: number,
  predictions: readonly CandidatePrediction[],
): CandidateEvaluationEvidence => {
  const metrics: CandidateEvaluationMetric[] = [];
  for (const purpose of [
    "validation",
    "representative_test",
    "coverage_test",
  ] as const) {
    const suite = predictions.filter(
      (prediction) => prediction.purpose === purpose,
    );
    if (suite.length === 0) {
      metrics.push({
        purpose,
        resultBin: null,
        exampleCount: 0,
        error: 1,
        passed: false,
      });
      continue;
    }
    metrics.push(measure(result, acceptableError, purpose, null, suite));
    const resultBins = [...new Set(suite.map((prediction) => prediction.resultBin))]
      .sort();
    for (const resultBin of resultBins) {
      metrics.push(measure(
        result,
        acceptableError,
        purpose,
        resultBin,
        suite.filter((prediction) => prediction.resultBin === resultBin),
      ));
    }
  }
  return {
    passed: metrics.every((metric) => metric.passed),
    maximumError: Math.max(...metrics.map((metric) => metric.error)),
    metrics,
  };
};

const measure = (
  result: ResultConfig,
  acceptableError: number,
  purpose: CandidatePrediction["purpose"],
  resultBin: string | null,
  predictions: readonly CandidatePrediction[],
): CandidateEvaluationMetric => {
  const error = predictions.reduce(
    (total, prediction) => total + (
      prediction.classificationFailed
        ? 1
        : resultError(result, prediction.reference, prediction.candidate)
    ),
    0,
  ) / predictions.length;
  return {
    purpose,
    resultBin,
    exampleCount: predictions.length,
    error,
    passed: error <= acceptableError,
  };
};
