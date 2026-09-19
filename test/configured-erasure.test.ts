import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createClassifier } from "../src/index.js";
import type {
  TrainingCandidate,
  TrainingLifecycleReporter,
  TrainingRunInspection,
} from "../src/index.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("configured classifier erasure", () => {
  it("cancels and verifies active provider resources before deleting all run data", async () => {
    const dataDirectory = mkdtempSync(join(tmpdir(), "swapai-erasure-"));
    directories.push(dataDirectory);
    let rejectTraining: ((error: Error) => void) | undefined;
    let started: (() => void) | undefined;
    const providerStarted = new Promise<void>((resolve) => { started = resolve; });
    const training = {
      name: "cancellable",
      train: async (
        _job: unknown,
        lifecycle?: TrainingLifecycleReporter,
      ): Promise<TrainingCandidate> => {
        lifecycle?.recordProviderRun({
          providerRunId: "machine-1",
          resources: [{ type: "machine", id: "machine-1" }],
        });
        started?.();
        return new Promise<TrainingCandidate>((_resolve, reject) => {
          rejectTraining = reject;
        });
      },
      cancel: async (
        _run: TrainingRunInspection,
        lifecycle?: TrainingLifecycleReporter,
      ) => {
        lifecycle?.recordCleanup({ status: "pending" });
        lifecycle?.recordCleanup({ status: "succeeded" });
        rejectTraining?.(new Error("cancelled"));
      },
    };
    const classifier = createClassifier({
      name: "erasable",
      result: { type: "boolean" },
      reference: async (input) => input.startsWith("yes"),
      training,
      dataDirectory,
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });
    for (let index = 0; index < 200; index += 1) {
      await classifier.classify(`${index % 2 === 0 ? "yes" : "no"}-${index}`);
    }
    await classifier.flush();

    const request = classifier.requestTraining();
    await providerStarted;
    const run = classifier.inspect().latestTrainingRun!;
    expect(run.cleanup.status).toBe("pending");
    const requestFailure = request.catch((error: unknown) => error);

    await expect(classifier.erase()).resolves.toBeUndefined();
    expect(await requestFailure).toBeInstanceOf(Error);
    expect(classifier.inspect()).toMatchObject({
      retainedExamples: 0,
      totalExamplesLogged: 0,
      trainingRuns: [],
    });
    expect(existsSync(join(dataDirectory, "training-runs", run.id))).toBe(false);
    await classifier.close();
  });
});
