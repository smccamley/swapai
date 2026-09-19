import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  verifyTrainingResult,
  writeTrainingBundle,
} from "../src/training-bundle.js";
import type { TrainingJob } from "../src/index.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("training runner contract", () => {
  it("rejects changed input and accepts only hash-verified output", async () => {
    const outputDirectory = mkdtempSync(join(tmpdir(), "swapai-contract-"));
    directories.push(outputDirectory);
    const bundle = await writeTrainingBundle(trainingJob(outputDirectory));
    const manifest = JSON.parse(
      readFileSync(join(bundle, "job.json"), "utf8"),
    ) as { format: number; inputs: { training: { sha256: string } } };
    expect(manifest.format).toBe(2);
    expect(manifest.inputs.training.sha256).toMatch(/^[a-f0-9]{64}$/);

    writeFileSync(join(bundle, "training.jsonl"), "changed");
    await expect(verifyTrainingResult(bundle)).rejects.toThrow(
      /training\.jsonl.*SHA-256/i,
    );

    const restored = await writeTrainingBundle(trainingJob(outputDirectory));
    writeFileSync(join(restored, "model.cact"), "model bytes");
    const modelHash = createHash("sha256").update("model bytes").digest("hex");
    const current = JSON.parse(
      readFileSync(join(restored, "job.json"), "utf8"),
    ) as {
      id: string;
      datasetRevisionId: string;
      inputs: { training: { sha256: string } };
    };
    writeFileSync(join(restored, "result.json"), JSON.stringify({
      format: 2,
      jobId: current.id,
      datasetRevisionId: current.datasetRevisionId,
      needleVersion: "2.0.14",
      inputs: {
        trainingSha256: current.inputs.training.sha256,
      },
      outputs: {
        model: { path: "model.cact", sha256: modelHash },
      },
    }));

    await expect(verifyTrainingResult(restored)).resolves.toMatchObject({
      modelPath: join(restored, "model.cact"),
      needleVersion: "2.0.14",
    });
    writeFileSync(join(restored, "model.cact"), "tampered");
    await expect(verifyTrainingResult(restored)).rejects.toThrow(
      /model\.cact.*SHA-256/i,
    );
  });
});

const trainingJob = (outputDirectory: string): TrainingJob => ({
  id: "runner-contract-id",
  datasetRevisionId: "b".repeat(64),
  classifierName: "runner-contract",
  generation: 1,
  dataEpoch: 0,
  result: { type: "boolean" },
  acceptableError: 0.1,
  examples: [
    { input: "yes", result: true, resultBin: "true", purpose: "training", facets: {} },
  ],
  outputDirectory,
});
