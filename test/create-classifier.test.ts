import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { createClassifier } from "../src/index.js";
import type { TrainingJob, TrainingProvider } from "../src/index.js";

const temporaryDirectories: string[] = [];

const makeDirectory = (): string => {
  const directory = mkdtempSync(join(tmpdir(), "swapai-simple-classifier-"));
  temporaryDirectories.push(directory);
  return directory;
};

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("createClassifier", () => {
  it("needs only a classifier definition and reference function", async () => {
    const reference = vi.fn(async (input: string) => input.includes("invoice"));
    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "boolean" },
      reference,
      facets: ["documentFamily"] as const,
      dataDirectory: makeDirectory(),
    });

    await expect(
      relevance.classify("Supplier invoice", { documentFamily: "invoice" }),
    ).resolves.toBe(true);
    await relevance.flush();

    expect(reference).toHaveBeenCalledWith("Supplier invoice");
    expect(relevance.isTrained()).toBe(false);
    await relevance.close();
  });

  it("accepts decision boundaries and declared facet names without training options", async () => {
    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async () => 0.92,
      decisionBoundaries: [0.5],
      facets: ["documentFamily", "sourceKind"] as const,
      dataDirectory: makeDirectory(),
    });

    await expect(
      relevance.classify("Supplier invoice", {
        documentFamily: "invoice",
        sourceKind: "gmail",
      }),
    ).resolves.toBe(0.92);
    await relevance.close();
  });

  it("explains missing range and test coverage instead of reporting only a total", async () => {
    const relevance = createClassifier({
      name: "accountant-relevance",
      result: { type: "number", min: 0, max: 1 },
      reference: async (input) => input.startsWith("relevant") ? 0.92 : 0.08,
      decisionBoundaries: [0.5],
      facets: ["documentFamily"] as const,
      dataDirectory: makeDirectory(),
    });

    for (let index = 0; index < 100; index += 1) {
      await relevance.classify(`irrelevant-${index}`, {
        documentFamily: "newsletter",
      });
    }
    for (let index = 0; index < 20; index += 1) {
      await relevance.classify(`relevant-${index}`, {
        documentFamily: "invoice",
      });
    }
    await relevance.flush();

    const inspection = relevance.inspect();
    expect(inspection).toMatchObject({
      totalExamplesLogged: 120,
      readyForTraining: false,
    });
    expect(inspection.resultBins).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          range: { minimum: 0, maximum: 0.2, includesMaximum: false },
          total: expect.any(Number),
        }),
        expect.objectContaining({
          range: { minimum: 0.4, maximum: 0.5, includesMaximum: false },
          total: 0,
        }),
        expect.objectContaining({
          range: { minimum: 0.5, maximum: 0.6, includesMaximum: false },
          total: 0,
        }),
        expect.objectContaining({
          range: { minimum: 0.8, maximum: 1, includesMaximum: true },
          total: expect.any(Number),
        }),
      ]),
    );
    expect(inspection.deficits).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          purpose: "training",
          resultBin: "0.4..0.5",
        }),
        expect.objectContaining({
          purpose: "coverage_test",
          resultBin: "0.5..0.6",
        }),
      ]),
    );
    await relevance.close();
  });

  it("keeps rare result coverage when the retained dataset reaches its limit", async () => {
    const relevance = createClassifier({
      name: "bounded-relevance",
      result: { type: "boolean" },
      reference: async () => false,
      maxTrainingSet: 8,
      dataDirectory: makeDirectory(),
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });

    relevance.logClassification("rare-positive", true);
    for (let index = 0; index < 30; index += 1) {
      relevance.logClassification(`common-negative-${index}`, false);
    }
    await relevance.flush();

    const inspection = relevance.inspect();
    expect(inspection.retainedExamples).toBe(8);
    expect(
      inspection.resultBins.find((bin) => bin.id === "true")?.total,
    ).toBe(1);
    await relevance.close();
  });

  it("counts repeated observations once in the retained dataset", async () => {
    const relevance = createClassifier({
      name: "deduplicated-relevance",
      result: { type: "boolean" },
      reference: async () => true,
      dataDirectory: makeDirectory(),
    });

    for (let index = 0; index < 10; index += 1) {
      relevance.logClassification("same input", true);
    }
    await relevance.flush();

    expect(relevance.inspect()).toMatchObject({
      totalExamplesLogged: 10,
      retainedExamples: 1,
    });
    await relevance.close();
  });

  it("hands only frozen training and validation examples to a trainer", async () => {
    const receivedJobs: TrainingJob[] = [];
    const expectedFailure = new Error("stop after inspecting the handoff");
    const training: TrainingProvider = {
      name: "recording-trainer",
      train: async (job) => {
        receivedJobs.push(job);
        throw expectedFailure;
      },
    };
    const relevance = createClassifier({
      name: "provider-boundary",
      result: { type: "boolean" },
      reference: async (input) => input.includes("yes"),
      training,
      dataDirectory: makeDirectory(),
      datasetRequirements: {
        minimumTrainingExamples: 0,
        minimumTrainingExamplesPerResultBin: 0,
        minimumValidationExamplesPerResultBin: 0,
        minimumRepresentativeTestExamples: 0,
        minimumCoverageTestExamplesPerResultBin: 0,
      },
    });

    for (let index = 0; index < 200; index += 1) {
      await relevance.classify(`${index % 2 === 0 ? "yes" : "no"}-${index}`);
    }
    await relevance.flush();

    await expect(relevance.requestTraining()).rejects.toBe(expectedFailure);
    expect(receivedJobs).toHaveLength(1);
    expect(receivedJobs[0]).toMatchObject({
      classifierName: "provider-boundary",
      result: { type: "boolean" },
      datasetRevisionId: expect.stringMatching(/^[a-f0-9]{64}$/),
    });
    expect(receivedJobs[0]!.examples.length).toBeGreaterThan(0);
    expect(
      receivedJobs[0]!.examples.every(
        (example) =>
          example.purpose === "training" || example.purpose === "validation",
      ),
    ).toBe(true);
    expect(receivedJobs[0]!.examples.some((example) => example.purpose === "training")).toBe(true);
    expect(receivedJobs[0]!.examples.some((example) => example.purpose === "validation")).toBe(true);
    expect(relevance.inspect().latestTrainingRun).toMatchObject({
      id: receivedJobs[0]!.id,
      provider: "recording-trainer",
      status: "failed",
      datasetRevisionId: receivedJobs[0]!.datasetRevisionId,
      failureMessage: "stop after inspecting the handoff",
    });
    await relevance.close();
  });
});
