import { chmod, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import {
  createNeedleNumberLabels,
  createTrainingLine,
} from "./needle.js";
import type { TrainingJob } from "./types.js";

export const writeTrainingBundle = async (
  job: TrainingJob,
): Promise<string> => {
  const bundleDirectory = join(job.outputDirectory, "bundle");
  await mkdir(bundleDirectory, { recursive: true, mode: 0o700 });
  const training = job.examples.filter((example) => example.purpose === "training");
  const validation = job.examples.filter((example) => example.purpose === "validation");
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

  await Promise.all([
    writePrivateFile(join(bundleDirectory, "training.jsonl"), lines(training)),
    writePrivateFile(join(bundleDirectory, "validation.jsonl"), lines(validation)),
    writePrivateFile(join(bundleDirectory, "job.json"), JSON.stringify({
      format: 1,
      id: job.id,
      datasetRevisionId: job.datasetRevisionId,
      classifierName: job.classifierName,
      result: job.result,
      acceptableError: job.acceptableError,
      trainingExamples: training.length,
      validationExamples: validation.length,
      needleVersion: "2.0.14",
    })),
    ...(numberLabels === undefined
      ? []
      : [writePrivateFile(
          join(bundleDirectory, "number-labels.json"),
          JSON.stringify({ format: 1, values: numberLabels.values }),
        )]),
  ]);
  return bundleDirectory;
};

const writePrivateFile = async (path: string, contents: string): Promise<void> => {
  await writeFile(path, contents, { encoding: "utf8", mode: 0o600 });
  await chmod(path, 0o600);
};
