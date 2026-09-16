export type NeedleResultConfig =
  | { readonly type: "number"; readonly min: number; readonly max: number }
  | { readonly type: "boolean" }
  | { readonly type: "string"; readonly values: readonly string[] };

export type NeedleResultValue = number | boolean | string;

export interface NeedleExample {
  readonly input: string;
  readonly result: NeedleResultValue;
}

export interface NeedleTool {
  readonly name: "classify";
  readonly description: string;
  readonly parameters: {
    readonly type: "object";
    readonly properties: {
      readonly result: Record<string, unknown>;
    };
    readonly required: readonly ["result"];
    readonly additionalProperties: false;
  };
}

export interface NeedleNumberLabels {
  readonly values: readonly number[];
}

export const MAX_NEEDLE_NUMBER_LABELS = 64;

export function createNeedleNumberLabels(
  values: readonly number[],
  config: Extract<NeedleResultConfig, { type: "number" }>,
  acceptableError = 0,
): NeedleNumberLabels {
  if (
    typeof acceptableError !== "number" ||
    !Number.isFinite(acceptableError) ||
    acceptableError < 0 ||
    acceptableError > 1
  ) {
    throw new TypeError("acceptableError must be between 0 and 1.");
  }
  const valid = values.map((value) =>
    validateNeedleResult(value, config) as number,
  );
  const unique = [...new Set(valid)].sort((left, right) => left - right);
  const desiredLabelCount = acceptableError === 0
    ? MAX_NEEDLE_NUMBER_LABELS
    : Math.min(
        MAX_NEEDLE_NUMBER_LABELS,
        Math.max(1, Math.floor(1 / (2 * acceptableError)) + 1),
      );
  if (unique.length <= desiredLabelCount) return { values: unique };

  const width = (config.max - config.min) / desiredLabelCount;
  const occupied = new Set<number>();
  for (const value of unique) {
    occupied.add(Math.min(
      desiredLabelCount - 1,
      Math.floor((value - config.min) / width),
    ));
  }
  return {
    values: [...occupied]
      .sort((left, right) => left - right)
      .map((index) => config.min + (index + 0.5) * width),
  };
}

function numberLabel(index: number, count: number): string {
  if (count === 1) return "only_score";
  if (count === 2) return index === 0 ? "lower_score" : "higher_score";
  return `score_bucket_${index}`;
}

function encodeNumberResult(
  value: number,
  labels: NeedleNumberLabels,
): string {
  if (labels.values.length === 0) {
    throw new TypeError("Numeric result has no Needle label.");
  }
  let closestIndex = 0;
  let closestDistance = Math.abs(value - labels.values[0]!);
  for (let index = 1; index < labels.values.length; index += 1) {
    const distance = Math.abs(value - labels.values[index]!);
    if (distance < closestDistance) {
      closestIndex = index;
      closestDistance = distance;
    }
  }
  return numberLabel(closestIndex, labels.values.length);
}

export function createNeedleTool(
  config: NeedleResultConfig,
  numberLabels?: NeedleNumberLabels,
): NeedleTool {
  const description = "The classification result.";
  let result: Record<string, unknown>;

  switch (config.type) {
    case "number":
      if (!numberLabels || numberLabels.values.length === 0) {
        throw new TypeError("Numeric Needle tools require at least one result label.");
      }
      result = {
        type: "string",
        enum: numberLabels.values.map((_, index) =>
          numberLabel(index, numberLabels.values.length),
        ),
        description:
          `The numeric classification bucket from ${config.min} to ${config.max}.`,
      };
      break;
    case "boolean":
      result = { type: "boolean", description };
      break;
    case "string":
      result = { type: "string", enum: [...config.values], description };
      break;
  }

  return {
    name: "classify",
    description: "Classify every supplied input and always return the learned result.",
    parameters: {
      type: "object",
      properties: { result },
      required: ["result"],
      additionalProperties: false,
    },
  };
}

export function createTrainingLine(
  input: string,
  value: NeedleResultValue,
  config: NeedleResultConfig,
  numberLabels?: NeedleNumberLabels,
): string {
  const result = validateNeedleResult(value, config);
  const needleResult =
    config.type === "number"
      ? encodeNumberResult(result as number, numberLabels ?? { values: [] })
      : result;
  return JSON.stringify({
    query: input,
    tools: [createNeedleTool(config, numberLabels)],
    answers: [{ name: "classify", arguments: { result: needleResult } }],
  });
}

export function validateNeedleResult(
  value: unknown,
  config: NeedleResultConfig,
): NeedleResultValue {
  switch (config.type) {
    case "number":
      if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        value < config.min ||
        value > config.max
      ) {
        throw new TypeError(
          `Needle returned a number outside ${config.min} to ${config.max}.`,
        );
      }
      return value;
    case "boolean":
      if (typeof value !== "boolean") {
        throw new TypeError("Needle did not return a boolean result.");
      }
      return value;
    case "string":
      if (typeof value !== "string" || !config.values.includes(value)) {
        throw new TypeError("Needle returned a value that is not an allowed string result.");
      }
      return value;
  }
}

export function readClassificationResult(
  response: unknown,
  config: NeedleResultConfig,
  numberLabels?: NeedleNumberLabels,
): NeedleResultValue {
  if (!isRecord(response) || !Array.isArray(response.function_calls)) {
    throw new TypeError("Needle did not return exactly one classify call.");
  }

  const calls = response.function_calls;
  const call = calls[0];
  if (
    calls.length !== 1 ||
    !isRecord(call) ||
    call.name !== "classify" ||
    !isRecord(call.arguments) ||
    !("result" in call.arguments)
  ) {
    throw new TypeError("Needle did not return exactly one classify call.");
  }

  return decodeNeedleResult(call.arguments.result, config, numberLabels);
}

export function decodeNeedleResult(
  result: unknown,
  config: NeedleResultConfig,
  numberLabels?: NeedleNumberLabels,
): NeedleResultValue {
  if (config.type === "number") {
    if (typeof result !== "string") {
      throw new TypeError(
        `Needle returned a number outside ${config.min} to ${config.max}.`,
      );
    }
    const index = numberLabels?.values.findIndex(
      (_, candidateIndex) =>
        result === numberLabel(candidateIndex, numberLabels.values.length),
    ) ?? -1;
    const decoded = numberLabels?.values[index];
    if (index < 0 || decoded === undefined) {
      throw new TypeError(
        `Needle returned a number outside ${config.min} to ${config.max}.`,
      );
    }
    return validateNeedleResult(decoded, config);
  }
  return validateNeedleResult(result, config);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
