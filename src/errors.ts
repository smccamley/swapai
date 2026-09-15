export type SwapAIErrorCode =
  | "invalid_configuration"
  | "invalid_result"
  | "not_trained"
  | "service_unavailable"
  | "classification_failed"
  | "storage_failed";

export class SwapAIError extends Error {
  readonly _tag = "SwapAIError" as const;
  readonly code: SwapAIErrorCode;

  constructor(
    code: SwapAIErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SwapAIError";
    this.code = code;
  }
}
