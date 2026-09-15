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

export function createNeedleTool(config: NeedleResultConfig): NeedleTool {
  const description = "The classification result.";
  let result: Record<string, unknown>;

  switch (config.type) {
    case "number":
      result = {
        type: "number",
        minimum: config.min,
        maximum: config.max,
        description,
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
): string {
  const result = validateNeedleResult(value, config);
  return JSON.stringify({
    query: input,
    tools: [createNeedleTool(config)],
    answers: [{ name: "classify", arguments: { result } }],
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

  return validateNeedleResult(call.arguments.result, config);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
