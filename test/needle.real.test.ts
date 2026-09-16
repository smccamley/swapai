import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { createNeedleRuntime } from "../src/runtime.js";
import { openStorage } from "../src/storage.js";

describe("real Needle 2 runtime", () => {
  it("trains, exports, loads, and classifies with a real .cact", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-real-needle-"));
    const storage = openStorage({
      dataDirectory,
      name: "real-boolean-smoke-test",
      maxTrainingSet: 100,
      config: {
        result: { type: "boolean" },
        retrainOnCount: 2,
        acceptableError: 0,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
    });
    const runtime = createNeedleRuntime({ dataDirectory });
    try {
      const model = await runtime.train({
        classifierName: "real-boolean-smoke-test",
        generation: 1,
        expectedEpoch: 0,
        resultConfig: { type: "boolean" },
        examples: Array.from({ length: 32 }, (_, index) => ({
          input: `Classify this as true for accounting relevance or false otherwise: ${
            index % 2 === 0
              ? `invoice ${index}: customer owes £${index + 1}`
              : `weather note ${index}: blue sky and sunshine`
          }`,
          result: index % 2 === 0,
        })),
        epochs: 10,
      });

      expect(model.modelPath.endsWith(".cact")).toBe(true);
      const loaded = await runtime.loadModel({
        modelPath: model.modelPath,
        resultConfig: { type: "boolean" },
      });
      try {
        await expect(
          loaded.classify(
            "Classify this as true for accounting relevance or false otherwise: invoice 0: customer owes £1",
          ),
        ).resolves.toBe(true);
      } finally {
        await loaded.close();
      }
    } finally {
      await runtime.close();
      storage.close();
      await rm(dataDirectory, { recursive: true, force: true });
    }
  });

  it("trains and classifies numeric scores through closed Needle labels", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-real-number-"));
    const name = "real-number-smoke-test";
    const storage = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: 100,
      config: {
        result: { type: "number", min: 0, max: 1 },
        retrainOnCount: 128,
        acceptableError: 1,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
    });
    const runtime = createNeedleRuntime({ dataDirectory });
    try {
      const examples = Array.from({ length: 128 }, (_, index) => ({
        input:
          "Classify accountant relevance: " +
          (index % 2 === 0
            ? `invoice ${index}: customer owes GBP ${index + 1}`
            : `weather note ${index}: blue sky and sunshine`),
        result: index % 2 === 0 ? 0.92 : 0.08,
      }));
      const model = await runtime.train({
        classifierName: name,
        generation: 1,
        expectedEpoch: 0,
        resultConfig: { type: "number", min: 0, max: 1 },
        examples,
        epochs: 10,
      });
      const loaded = await runtime.loadModel({
        modelPath: model.modelPath,
        resultConfig: { type: "number", min: 0, max: 1 },
      });
      try {
        await expect(loaded.classify(examples[0]!.input)).resolves.toBe(0.92);
        await expect(loaded.classify(examples[1]!.input)).resolves.toBe(0.08);
      } finally {
        await loaded.close();
      }
    } finally {
      await runtime.close();
      storage.close();
      await rm(dataDirectory, { recursive: true, force: true });
    }
  });
});
