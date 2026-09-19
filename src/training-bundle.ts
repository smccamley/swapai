import { createHash } from "node:crypto";
import { chmod, mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import {
  createNeedleNumberLabels,
  createTrainingLine,
} from "./needle.js";
import type { TrainingCandidate, TrainingJob } from "./types.js";

export const writeTrainingBundle = async (
  job: TrainingJob,
): Promise<string> => {
  const bundleDirectory = join(job.outputDirectory, "bundle");
  await mkdir(bundleDirectory, { recursive: true, mode: 0o700 });
  const training = job.examples.filter((example) => example.purpose === "training");
  const numberLabels = job.result.type === "number"
    ? createNeedleNumberLabels(
        training.map((example) => example.result as number),
        job.result,
        job.acceptableError,
      )
    : undefined;
  const lines = (examples: typeof job.examples): string =>
    `${examples.map((example) =>
      createTrainingLine(
        example.input,
        example.result,
        job.result,
        numberLabels,
      )).join("\n")}\n`;

  const trainingContents = lines(training);
  const numberLabelContents = numberLabels === undefined
    ? undefined
    : JSON.stringify({ format: 1, values: numberLabels.values });
  await Promise.all([
    writePrivateFile(join(bundleDirectory, "training.jsonl"), trainingContents),
    ...(numberLabelContents === undefined
      ? []
      : [writePrivateFile(
          join(bundleDirectory, "number-labels.json"),
          numberLabelContents,
        )]),
  ]);
  await writePrivateFile(join(bundleDirectory, "job.json"), JSON.stringify({
      format: 2,
      id: job.id,
      datasetRevisionId: job.datasetRevisionId,
      classifierName: job.classifierName,
      result: job.result,
      acceptableError: job.acceptableError,
      trainingExamples: training.length,
      needleVersion: "2.0.14",
      inputs: {
        training: {
          path: "training.jsonl",
          sha256: sha256(trainingContents),
          examples: training.length,
        },
        ...(numberLabelContents === undefined
          ? {}
          : {
              numberLabels: {
                path: "number-labels.json",
                sha256: sha256(numberLabelContents),
              },
            }),
      },
    }));
  return bundleDirectory;
};

interface RunnerManifest {
  readonly format: 2;
  readonly id: string;
  readonly datasetRevisionId: string;
  readonly result: TrainingJob["result"];
  readonly inputs: {
    readonly training: RunnerFile;
    readonly numberLabels?: RunnerFile;
  };
}

interface RunnerFile {
  readonly path: string;
  readonly sha256: string;
  readonly examples?: number;
}

interface RunnerResult {
  readonly format: 2;
  readonly jobId: string;
  readonly datasetRevisionId: string;
  readonly needleVersion: string;
  readonly inputs: {
    readonly trainingSha256: string;
  };
  readonly outputs: {
    readonly model: RunnerFile;
    readonly numberLabels?: RunnerFile;
  };
}

export const verifyTrainingResult = async (
  bundleDirectory: string,
): Promise<TrainingCandidate> => {
  const manifest = await readJson<RunnerManifest>(
    join(bundleDirectory, "job.json"),
  );
  if (manifest.format !== 2) throw new Error("Unsupported SwapAI training job");
  await verifyFile(bundleDirectory, "training.jsonl", manifest.inputs.training);
  if (manifest.inputs.numberLabels !== undefined) {
    await verifyFile(bundleDirectory, "number-labels.json", manifest.inputs.numberLabels);
  }

  const result = await readJson<RunnerResult>(
    join(bundleDirectory, "result.json"),
  );
  if (
    result.format !== 2 ||
    result.jobId !== manifest.id ||
    result.datasetRevisionId !== manifest.datasetRevisionId
  ) {
    throw new Error("Trainer result does not match the training job");
  }
  if (
    result.inputs.trainingSha256 !== manifest.inputs.training.sha256
  ) {
    throw new Error("Trainer result does not attest to the supplied inputs");
  }
  await verifyFile(bundleDirectory, "model.cact", result.outputs.model);
  if (manifest.result.type === "number") {
    if (result.outputs.numberLabels === undefined) {
      throw new Error("Trainer result is missing numeric labels");
    }
    await verifyFile(
      bundleDirectory,
      "model.cact.numbers.json",
      result.outputs.numberLabels,
    );
  }
  return {
    modelPath: join(bundleDirectory, "model.cact"),
    needleVersion: result.needleVersion,
  };
};

const verifyFile = async (
  directory: string,
  expectedPath: string,
  file: RunnerFile,
): Promise<void> => {
  if (file.path !== expectedPath) {
    throw new Error(`Trainer supplied an unexpected path for ${expectedPath}`);
  }
  const contents = await readFile(join(directory, expectedPath));
  if (sha256(contents) !== file.sha256) {
    throw new Error(`${expectedPath} failed SHA-256 verification`);
  }
};

const readJson = async <Value>(path: string): Promise<Value> =>
  JSON.parse(await readFile(path, "utf8")) as Value;

const sha256 = (value: string | Uint8Array): string =>
  createHash("sha256").update(value).digest("hex");

const writePrivateFile = async (path: string, contents: string): Promise<void> => {
  await writeFile(path, contents, { encoding: "utf8", mode: 0o600 });
  await chmod(path, 0o600);
};
