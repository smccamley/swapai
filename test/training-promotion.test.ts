import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/runtime.js", () => ({
  NEEDLE_VERSION: "2.0.14",
  NEEDLE_NUMBER_MODEL_VERSION: "2.0.14/number-buckets-v1",
  NeedleModelArtifactError: class NeedleModelArtifactError extends Error {},
  needleModelVersion: () => "2.0.14",
  hasNeedleModelArtifacts: () => true,
  createNeedleRuntime: () => ({
    ready: async () => undefined,
    train: vi.fn(),
    loadModel: async () => ({
      classify: async (input: string) => input.startsWith("relevant-"),
      close: async () => undefined,
    }),
    clearClassifierArtifacts: async () => undefined,
    clearClassifierGenerationArtifacts: async () => undefined,
    clearClassifierArtifactsThroughGeneration: async () => undefined,
    close: async () => undefined,
  }),
}));

import { createClassifier } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("manual training promotion", () => {
  it("promotes a provider artifact only after both protected suites pass", async () => {
    const dataDirectory = mkdtempSync(join(tmpdir(), "swapai-promotion-"));
    temporaryDirectories.push(dataDirectory);
    const relevance = createClassifier({
      name: "promotion-evidence",
      result: { type: "boolean" },
      reference: async (input) => input.startsWith("relevant-"),
      training: {
        name: "test-provider",
        train: async (job) => {
          const modelPath = join(job.outputDirectory, "model.cact");
          writeFileSync(modelPath, "candidate model");
          return {
            modelPath,
            needleVersion: "2.0.14",
            providerRunId: "provider-run-1",
            costUsd: 0.04,
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
    for (let index = 0; index < 500; index += 1) {
      await relevance.classify(
        `${index % 2 === 0 ? "relevant" : "irrelevant"}-${index}`,
      );
    }
    await relevance.flush();

    await expect(relevance.requestTraining()).resolves.toMatchObject({
      status: "promoted",
    });
    expect(relevance.isTrained()).toBe(true);
    expect(relevance.inspect().latestTrainingRun).toMatchObject({
      provider: "test-provider",
      providerRunId: "provider-run-1",
      status: "promoted",
      costUsd: 0.04,
      artifactSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
    });
    await relevance.close();
  });
});
