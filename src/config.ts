import { SwapAIError } from "./errors.js";
import type {
  InitConfig,
  NormalizedInitConfig,
  ResultConfig,
  ResultFor,
} from "./types.js";

function invalidConfiguration(message: string): never {
  throw new SwapAIError("invalid_configuration", message);
}

function requirePositiveInteger(value: number, field: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    invalidConfiguration(`${field} must be a positive integer`);
  }
}

function requireNonNegativeInteger(value: number, field: string): void {
  if (!Number.isInteger(value) || value < 0) {
    invalidConfiguration(`${field} must be a non-negative integer`);
  }
}

function normalizeAcceptableError(value: number | `${number}%`): number {
  let normalized: number;

  if (typeof value === "number") {
    normalized = value;
  } else {
    const match = /^\s*(\d+(?:\.\d+)?)%\s*$/.exec(value);
    if (match === null) {
      invalidConfiguration(
        "acceptableError must be a number or a percentage such as \"10%\"",
      );
    }
    normalized = Number(match[1]) / 100;
  }

  if (!Number.isFinite(normalized) || normalized < 0 || normalized > 1) {
    invalidConfiguration("acceptableError must be between 0 and 1");
  }

  return normalized;
}

function validateResultConfig(result: ResultConfig): void {
  if (result.type === "number") {
    if (!Number.isFinite(result.min) || !Number.isFinite(result.max)) {
      invalidConfiguration("number result bounds must be finite");
    }
    if (result.max <= result.min) {
      invalidConfiguration("number result max must be greater than min");
    }
    if (!Number.isFinite(result.max - result.min)) {
      invalidConfiguration("number result range must be finite");
    }
    return;
  }

  if (result.type === "string") {
    if (result.values.length === 0) {
      invalidConfiguration("string result values must not be empty");
    }
    if (
      result.values.some((value) => typeof value !== "string") ||
      new Set(result.values).size !== result.values.length
    ) {
      invalidConfiguration("string result values must be unique strings");
    }
  }
}

export function normalizeConfig<const Config extends ResultConfig>(
  config: InitConfig<Config>,
): NormalizedInitConfig<Config> {
  if (typeof config.name !== "string" || config.name.trim() === "") {
    invalidConfiguration("name must not be empty");
  }

  validateResultConfig(config.result);
  requirePositiveInteger(config.retrainOnCount, "retrainOnCount");
  requireNonNegativeInteger(config.retestInterval, "retestInterval");
  requirePositiveInteger(config.retestRevertOn, "retestRevertOn");
  requirePositiveInteger(config.maxTrainingSet, "maxTrainingSet");

  if (config.maxTrainingSet < config.retrainOnCount) {
    invalidConfiguration(
      "maxTrainingSet must be greater than or equal to retrainOnCount",
    );
  }
  if (config.model !== "needle2") {
    invalidConfiguration('model must be "needle2"');
  }
  if (
    config.dataDirectory !== undefined &&
    (typeof config.dataDirectory !== "string" ||
      config.dataDirectory.trim() === "")
  ) {
    invalidConfiguration("dataDirectory must not be empty");
  }
  if (
    config.onBackgroundError !== undefined &&
    typeof config.onBackgroundError !== "function"
  ) {
    invalidConfiguration("onBackgroundError must be a function");
  }

  return {
    ...config,
    acceptableError: normalizeAcceptableError(config.acceptableError),
  };
}

export function validateResult<const Config extends ResultConfig>(
  config: Config,
  value: unknown,
): ResultFor<Config> {
  const valid =
    config.type === "number"
      ? typeof value === "number" &&
        Number.isFinite(value) &&
        value >= config.min &&
        value <= config.max
      : config.type === "boolean"
        ? typeof value === "boolean"
        : typeof value === "string" && config.values.includes(value);

  if (!valid) {
    throw new SwapAIError(
      "invalid_result",
      "classification result does not match the declared allowed results",
    );
  }

  return value as ResultFor<Config>;
}
