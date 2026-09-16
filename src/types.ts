import type { SwapAIError } from "./errors.js";

export type ResultValue = number | boolean | string;

export interface NumberResultConfig {
  readonly type: "number";
  readonly min: number;
  readonly max: number;
}

export interface BooleanResultConfig {
  readonly type: "boolean";
}

export interface StringResultConfig<Value extends string = string> {
  readonly type: "string";
  readonly values: readonly Value[];
}

export type ResultConfig =
  | NumberResultConfig
  | BooleanResultConfig
  | StringResultConfig;

export type ResultFor<Config extends ResultConfig> =
  Config extends NumberResultConfig
    ? number
    : Config extends BooleanResultConfig
      ? boolean
      : Config extends StringResultConfig<infer Value>
        ? Value
        : never;

export type ReferenceClassifier<Result extends ResultValue> = (
  input: string,
) => Result | Promise<Result>;

export interface Classifier<Result extends ResultValue> {
  isTrained(): boolean;
  logClassification(input: string, result: Result): void;
  clearTrainingData(): void;
  classify(
    input: string,
    referenceClassifier?: ReferenceClassifier<Result>,
  ): Promise<Result>;
  flush(): Promise<void>;
  close(): Promise<void>;
}

export interface InitConfig<Config extends ResultConfig = ResultConfig> {
  readonly name: string;
  readonly result: Config;
  readonly retrainOnCount: number;
  readonly acceptableError: number | `${number}%`;
  readonly retestInterval: number;
  readonly retestRevertOn: number;
  readonly model: "needle2";
  readonly maxTrainingSet: number;
  readonly dataDirectory?: string;
  readonly onBackgroundError?: (error: SwapAIError) => void;
}

export type NormalizedInitConfig<Config extends ResultConfig = ResultConfig> =
  Omit<InitConfig<Config>, "acceptableError"> & {
    readonly acceptableError: number;
  };

export interface ResultComparison<Result extends ResultValue> {
  readonly reference: Result;
  readonly candidate: Result;
}
