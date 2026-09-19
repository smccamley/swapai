import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/runtime.js", () => ({
  NEEDLE_VERSION: "2.0.14",
  NEEDLE_NUMBER_MODEL_VERSION: "2.0.14/number-buckets-v1",
  NeedleModelArtifactError: class NeedleModelArtifactError extends Error {},
  needleModelVersion: (result: { type: string }) =>
    result.type === "number" ? "2.0.14/number-buckets-v1" : "2.0.14",
  hasNeedleModelArtifacts: () => true,
  createNeedleRuntime: () => ({
    ready: async () => undefined,
    train: vi.fn(),
    loadModel: async ({ resultConfig }: { resultConfig: { type: string } }) => ({
      classify: async (input: string) => resultConfig.type === "number"
        ? input.startsWith("high-") ? 0.9 : 0.1
        : input === "shadow-disagrees" || input.startsWith("relevant-"),
      close: async () => undefined,
    }),
    clearClassifierArtifacts: async () => undefined,
    clearClassifierGenerationArtifacts: async () => undefined,
    clearClassifierArtifactsThroughGeneration: async () => undefined,
    close: async () => undefined,
  }),
}));

import { createClassifier } from "../src/index.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("manual training promotion", () => {
  it("keeps a passing candidate in shadow mode until explicitly promoted", async () => {
    const dataDirectory = mkdtempSync(join(tmpdir(), "swapai-promotion-"));
    temporaryDirectories.push(dataDirectory);
    let transientOutputDirectory = "";
    const train = vi.fn(async (job: { outputDirectory: string }) => {
      transientOutputDirectory = job.outputDirectory;
      const modelPath = join(job.outputDirectory, "model.cact");
      writeFileSync(modelPath, "candidate model");
      return {
        modelPath,
        needleVersion: "2.0.14",
        providerRunId: "provider-run-1",
        costUsd: 0.04,
      };
    });
    const relevance = createClassifier({
      name: "promotion-evidence",
      result: { type: "boolean" },
      reference: async (input) => input.startsWith("relevant-"),
      training: {
        name: "test-provider",
        train,
      },
      dataDirectory,
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });
    for (let index = 0; index < 500; index += 1) {
      await relevance.classify(
        `${index % 2 === 0 ? "relevant" : "irrelevant"}-${index}`,
      );
    }
    await relevance.flush();

    const training = await relevance.requestTraining();
    expect(training).toMatchObject({ status: "candidate" });
    await expect(relevance.requestTraining()).resolves.toEqual(training);
    expect(train).toHaveBeenCalledOnce();
    expect(existsSync(transientOutputDirectory)).toBe(false);
    expect(relevance.isTrained()).toBe(false);
    if (training.status !== "candidate") throw new Error("expected candidate");
    await expect(
      relevance.promoteCandidate(training.trainingRunId),
    ).rejects.toThrow(/shadow/i);
    await expect(relevance.classify("shadow-agrees")).resolves.toBe(false);
    await relevance.flush();
    expect(relevance.inspect().latestTrainingRun).toMatchObject({
      provider: "test-provider",
      providerRunId: "provider-run-1",
      status: "candidate",
      costUsd: 0.04,
      artifactSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      shadow: {
        exampleCount: 1,
        meanError: 0,
        passed: true,
        failureCount: 0,
      },
    });
    await expect(
      relevance.promoteCandidate(training.trainingRunId),
    ).resolves.toEqual({
      status: "promoted",
      trainingRunId: training.trainingRunId,
      datasetRevisionId: training.datasetRevisionId,
    });
    expect(relevance.isTrained()).toBe(true);
    expect(relevance.inspect().latestTrainingRun?.status).toBe("promoted");
    await relevance.close();
  });

  it("includes the numeric label sidecar in artifact identity", async () => {
    const identities: string[] = [];
    for (const [index, labels] of ["labels-a", "labels-b"].entries()) {
      const dataDirectory = mkdtempSync(join(tmpdir(), "swapai-numeric-artifact-"));
      temporaryDirectories.push(dataDirectory);
      const classifier = createClassifier({
        name: `numeric-artifact-${index}`,
        result: { type: "number", min: 0, max: 1 },
        reference: async (input) => input.startsWith("high-") ? 0.9 : 0.1,
        training: {
          name: "numeric-test",
          train: async (job) => {
            const modelPath = join(job.outputDirectory, "model.cact");
            writeFileSync(modelPath, "identical model");
            writeFileSync(`${modelPath}.numbers.json`, labels);
            return {
              modelPath,
              needleVersion: "2.0.14/number-buckets-v1",
            };
          },
        },
        dataDirectory,
        datasetRequirements: {
          minimumTrainingExamples: 0,
          minimumTrainingExamplesPerResultBin: 0,
          minimumValidationExamplesPerResultBin: 0,
          minimumRepresentativeTestExamples: 0,
          minimumCoverageTestExamplesPerResultBin: 0,
        },
      });
      for (let example = 0; example < 100; example += 1) {
        await classifier.classify(
          `${example % 2 === 0 ? "high" : "low"}-${example}`,
        );
      }
      await classifier.flush();
      await classifier.requestTraining();
      identities.push(classifier.inspect().latestTrainingRun!.artifactSha256!);
      await classifier.close();
    }
    expect(identities[0]).not.toBe(identities[1]);
  });

  it("removes rejected candidate artifacts but retains evaluation evidence", async () => {
    const dataDirectory = mkdtempSync(join(tmpdir(), "swapai-rejected-artifact-"));
    temporaryDirectories.push(dataDirectory);
    const name = "rejected-artifact";
    const classifier = createClassifier({
      name,
      result: { type: "boolean" },
      reference: async () => false,
      training: {
        name: "rejecting-test",
        train: async (job) => {
          const modelPath = join(job.outputDirectory, "model.cact");
          writeFileSync(modelPath, "rejected model");
          return { modelPath, needleVersion: "2.0.14" };
        },
      },
      dataDirectory,
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });
    for (let index = 0; index < 500; index += 1) {
      await classifier.classify(`relevant-${index}`);
    }
    await classifier.flush();

    const training = await classifier.requestTraining();
    expect(training.status).toBe("rejected");
    const run = classifier.inspect().latestTrainingRun!;
    expect(run).toMatchObject({
      status: "rejected",
      artifactSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      evaluations: expect.arrayContaining([
        expect.objectContaining({ passed: false }),
      ]),
    });
    const artifactDirectory = join(
      dataDirectory,
      "classifiers",
      createHash("sha256").update(name).digest("hex"),
      "artifacts",
      "sha256",
      run.artifactSha256!,
    );
    expect(existsSync(artifactDirectory)).toBe(false);
    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"), {
      readOnly: true,
    });
    try {
      expect(database.prepare(`
        SELECT COUNT(*) AS count FROM model_artifacts
        WHERE classifier_name = ? AND sha256 = ?
      `).get(name, run.artifactSha256)).toEqual({ count: 0 });
    } finally {
      database.close();
    }
    await classifier.close();
  });
});
