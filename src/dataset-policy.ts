import { createHash } from "node:crypto";

import { SwapAIError } from "./errors.js";
import type {
  ClassificationFacets,
  DatasetPolicy,
  DatasetPolicyInput,
  DatasetPurpose,
  ResultConfig,
  ResultFor,
} from "./types.js";

export interface NumericResultBin {
  readonly id: string;
  readonly minimum: number;
  readonly maximum: number;
  readonly includesMaximum: boolean;
}

export interface DiscreteResultBin {
  readonly id: string;
  readonly value: boolean | string;
}

export type ResultBin = NumericResultBin | DiscreteResultBin;

export interface PreparedExampleMetadata {
  readonly inputHash: string;
  readonly resultBin: string;
  readonly purpose: Exclude<DatasetPurpose, "legacy_seen">;
  readonly facets: ClassificationFacets<string>;
}

const DEFAULT_REQUIREMENTS = {
  minimumTrainingExamples: 1_000,
  minimumTrainingExamplesPerResultBin: 50,
  minimumValidationExamplesPerResultBin: 20,
  minimumRepresentativeTestExamples: 200,
  minimumCoverageTestExamplesPerResultBin: 30,
} as const;

const requireNonNegativeInteger = (value: number, name: string): void => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SwapAIError(
      "invalid_configuration",
      `${name} must be a non-negative integer`,
    );
  }
};

export const createDatasetPolicy = (
  result: ResultConfig,
  input: DatasetPolicyInput,
): DatasetPolicy => {
  const decisionBoundaries = [...(input.decisionBoundaries ?? [])];
  if (result.type !== "number" && decisionBoundaries.length > 0) {
    throw new SwapAIError(
      "invalid_configuration",
      "decisionBoundaries are only valid for number results",
    );
  }
  if (
    result.type === "number" &&
    decisionBoundaries.some(
      (boundary) =>
        !Number.isFinite(boundary) ||
        boundary <= result.min ||
        boundary >= result.max,
    )
  ) {
    throw new SwapAIError(
      "invalid_configuration",
      "decisionBoundaries must be finite numbers inside the result range",
    );
  }
  if (new Set(decisionBoundaries).size !== decisionBoundaries.length) {
    throw new SwapAIError(
      "invalid_configuration",
      "decisionBoundaries must be unique",
    );
  }

  const facets = [...(input.facets ?? [])];
  if (
    facets.some((facet) => facet.trim() === "") ||
    new Set(facets).size !== facets.length
  ) {
    throw new SwapAIError(
      "invalid_configuration",
      "facets must contain unique non-empty names",
    );
  }

  const requirements = {
    ...DEFAULT_REQUIREMENTS,
    ...input.requirements,
  };
  for (const [name, value] of Object.entries(requirements)) {
    requireNonNegativeInteger(value, name);
  }

  return {
    decisionBoundaries: decisionBoundaries.sort((left, right) => left - right),
    facets,
    requirements,
  };
};

const stableNumber = (value: number): number => Number(value.toPrecision(12));
const displayNumber = (value: number): string => String(stableNumber(value));

export const resultBins = (
  result: ResultConfig,
  acceptableError: number,
  policy: DatasetPolicy,
): readonly ResultBin[] => {
  if (result.type === "boolean") {
    return [
      { id: "false", value: false },
      { id: "true", value: true },
    ];
  }
  if (result.type === "string") {
    return result.values.map((value) => ({ id: value, value }));
  }

  const range = result.max - result.min;
  const ordinaryWidth = Math.max(range * acceptableError * 2, range / 10);
  const boundaryWidth = Math.max(range * acceptableError, range / 20);
  const points = new Set<number>([result.min, result.max]);
  for (
    let point = result.min + ordinaryWidth;
    point < result.max;
    point += ordinaryWidth
  ) {
    points.add(stableNumber(point));
  }
  for (const boundary of policy.decisionBoundaries) {
    points.add(stableNumber(boundary));
    points.add(stableNumber(Math.max(result.min, boundary - boundaryWidth)));
    points.add(stableNumber(Math.min(result.max, boundary + boundaryWidth)));
  }

  const ordered = [...points].sort((left, right) => left - right);
  return ordered.slice(0, -1).map((minimum, index) => {
    const maximum = ordered[index + 1]!;
    return {
      id: `${displayNumber(minimum)}..${displayNumber(maximum)}`,
      minimum,
      maximum,
      includesMaximum: index === ordered.length - 2,
    };
  });
};

const resultBinFor = <Config extends ResultConfig>(
  bins: readonly ResultBin[],
  result: Config,
  value: ResultFor<Config>,
): string => {
  if (result.type !== "number") return String(value);
  const numeric = value as number;
  const bin = bins.find(
    (candidate) =>
      "minimum" in candidate &&
      numeric >= candidate.minimum &&
      (numeric < candidate.maximum ||
        (candidate.includesMaximum && numeric === candidate.maximum)),
  );
  if (bin === undefined) {
    throw new SwapAIError("invalid_result", "result did not fit a result bin");
  }
  return bin.id;
};

const purposeFor = (
  classifierName: string,
  inputHash: string,
  resultBin: string,
): Exclude<DatasetPurpose, "legacy_seen"> => {
  const bucket = createHash("sha256")
    .update(classifierName)
    .update("\0")
    .update(resultBin)
    .update("\0")
    .update(inputHash)
    .digest()
    .readUInt32BE(0) % 100;
  if (bucket < 65) return "training";
  if (bucket < 80) return "validation";
  if (bucket < 90) return "representative_test";
  return "coverage_test";
};

export const prepareExampleMetadata = <Config extends ResultConfig>(
  classifierName: string,
  result: Config,
  acceptableError: number,
  policy: DatasetPolicy,
  input: string,
  value: ResultFor<Config>,
  facets: ClassificationFacets<string> = {},
): PreparedExampleMetadata => {
  const undeclaredFacet = Object.keys(facets).find(
    (facet) => !policy.facets.includes(facet),
  );
  if (undeclaredFacet !== undefined) {
    throw new SwapAIError(
      "invalid_result",
      `classification facet is not declared: ${undeclaredFacet}`,
    );
  }
  if (
    Object.values(facets).some(
      (facet) => typeof facet !== "string" || facet.trim() === "",
    )
  ) {
    throw new SwapAIError(
      "invalid_result",
      "classification facets must be non-empty strings",
    );
  }
  const inputHash = createHash("sha256").update(input).digest("hex");
  const normalizedFacets = Object.fromEntries(
    policy.facets
      .filter((facet) => facets[facet] !== undefined)
      .map((facet) => [facet, facets[facet]!]),
  );
  const bin = resultBinFor(
    resultBins(result, acceptableError, policy),
    result,
    value,
  );
  return {
    inputHash,
    resultBin: bin,
    purpose: purposeFor(classifierName, inputHash, bin),
    facets: normalizedFacets,
  };
};
