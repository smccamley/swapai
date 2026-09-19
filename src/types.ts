import type { SwapAIError } from "./errors.js";

export type ResultValue = number | boolean | string;
export type DatasetPurpose =
  | "training"
  | "validation"
  | "representative_test"
  | "coverage_test"
  | "legacy_seen";

export interface DatasetRequirements {
  readonly minimumTrainingExamples: number;
  readonly minimumTrainingExamplesPerResultBin: number;
  readonly minimumValidationExamplesPerResultBin: number;
  readonly minimumRepresentativeTestExamples: number;
  readonly minimumCoverageTestExamplesPerResultBin: number;
}

export interface DatasetPolicy {
  readonly decisionBoundaries: readonly number[];
  readonly facets: readonly string[];
  readonly requirements: DatasetRequirements;
}

export interface DatasetRequirementOverrides {
  readonly minimumTrainingExamples?: number;
  readonly minimumTrainingExamplesPerResultBin?: number;
  readonly minimumValidationExamplesPerResultBin?: number;
  readonly minimumRepresentativeTestExamples?: number;
  readonly minimumCoverageTestExamplesPerResultBin?: number;
}

export interface DatasetPolicyInput {
  readonly decisionBoundaries?: readonly number[];
  readonly facets?: readonly string[];
  readonly requirements?: DatasetRequirementOverrides;
}

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
  logClassification(
    input: string,
    result: Result,
    facets?: ClassificationFacets<string>,
  ): void;
  clearTrainingData(): void;
  classify(
    input: string,
    referenceClassifier?: ReferenceClassifier<Result>,
    facets?: ClassificationFacets<string>,
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
  readonly automaticTraining?: boolean;
  readonly datasetPolicy?: DatasetPolicy;
  readonly dataDirectory?: string;
  readonly onBackgroundError?: (error: SwapAIError) => void;
}

export type ClassificationFacets<FacetName extends string> = Partial<
  Readonly<Record<FacetName, string>>
>;

export interface CreateClassifierConfig<
  Config extends ResultConfig = ResultConfig,
  FacetName extends string = string,
> {
  readonly name: string;
  readonly result: Config;
  readonly reference: ReferenceClassifier<ResultFor<Config>>;
  readonly training?: TrainingProvider;
  readonly decisionBoundaries?: readonly number[];
  readonly facets?: readonly FacetName[];
  readonly acceptableError?: number | `${number}%`;
  readonly maxTrainingSet?: number;
  readonly dataDirectory?: string;
  readonly onBackgroundError?: (error: SwapAIError) => void;
  readonly datasetRequirements?: DatasetRequirementOverrides;
}

export interface TrainingExample {
  readonly input: string;
  readonly result: ResultValue;
  readonly resultBin: string;
  readonly purpose: "training" | "validation";
  readonly facets: ClassificationFacets<string>;
}

export interface TrainingJob {
  readonly id: string;
  readonly datasetRevisionId: string;
  readonly classifierName: string;
  readonly generation: number;
  readonly dataEpoch: number;
  readonly result: ResultConfig;
  readonly acceptableError: number;
  readonly examples: readonly TrainingExample[];
  readonly outputDirectory: string;
}

export interface TrainingCandidate {
  readonly modelPath: string;
  readonly needleVersion: string;
  readonly providerRunId?: string;
  readonly costUsd?: number;
}

export interface TrainingProvider {
  readonly name: string;
  train(job: TrainingJob): Promise<TrainingCandidate>;
}

export type TrainingRequestResult =
  | {
      readonly status: "not_ready";
      readonly deficits: readonly DatasetDeficit[];
    }
  | {
      readonly status: "promoted" | "rejected";
      readonly trainingRunId: string;
      readonly datasetRevisionId: string;
    };

export interface NumericResultBinInspection {
  readonly id: string;
  readonly range: {
    readonly minimum: number;
    readonly maximum: number;
    readonly includesMaximum: boolean;
  };
  readonly total: number;
  readonly purposes: Record<Exclude<DatasetPurpose, "legacy_seen">, number>;
}

export interface DiscreteResultBinInspection {
  readonly id: string;
  readonly value: boolean | string;
  readonly total: number;
  readonly purposes: Record<Exclude<DatasetPurpose, "legacy_seen">, number>;
}

export type ResultBinInspection =
  | NumericResultBinInspection
  | DiscreteResultBinInspection;

export interface DatasetDeficit {
  readonly purpose: Exclude<DatasetPurpose, "legacy_seen">;
  readonly resultBin: string | null;
  readonly required: number;
  readonly available: number;
}

export interface ClassifierInspection {
  readonly name: string;
  readonly totalExamplesLogged: number;
  readonly retainedExamples: number;
  readonly readyForTraining: boolean;
  readonly resultBins: readonly ResultBinInspection[];
  readonly deficits: readonly DatasetDeficit[];
  readonly examplesByPurpose: Record<Exclude<DatasetPurpose, "legacy_seen">, number>;
  readonly latestTrainingRun: TrainingRunInspection | null;
}

export interface TrainingRunInspection {
  readonly id: string;
  readonly datasetRevisionId: string;
  readonly provider: string;
  readonly status: "running" | "failed" | "rejected" | "promoted";
  readonly providerRunId: string | null;
  readonly costUsd: number | null;
  readonly artifactSha256: string | null;
  readonly failureMessage: string | null;
  readonly startedAt: number;
  readonly finishedAt: number | null;
}

export interface ConfiguredClassifier<
  Result extends ResultValue,
  FacetName extends string = string,
> {
  isTrained(): boolean;
  classify(
    input: string,
    facets?: ClassificationFacets<FacetName>,
  ): Promise<Result>;
  logClassification(
    input: string,
    result: Result,
    facets?: ClassificationFacets<FacetName>,
  ): void;
  inspect(): ClassifierInspection;
  requestTraining(): Promise<TrainingRequestResult>;
  clearTrainingData(): void;
  flush(): Promise<void>;
  close(): Promise<void>;
}

export type NormalizedInitConfig<Config extends ResultConfig = ResultConfig> =
  Omit<InitConfig<Config>, "acceptableError"> & {
    readonly acceptableError: number;
  };

export interface ResultComparison<Result extends ResultValue> {
  readonly reference: Result;
  readonly candidate: Result;
}
