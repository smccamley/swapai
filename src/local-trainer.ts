import { dirname } from "node:path";

import { createNeedleRuntime } from "./runtime.js";
import type { TrainingProvider } from "./types.js";

export const localTrainer = (): TrainingProvider => ({
  name: "local",
  train: async (job) => {
    const runtime = createNeedleRuntime({
      dataDirectory: dirname(dirname(job.outputDirectory)),
    });
    try {
      return await runtime.train({
        classifierName: job.classifierName,
        generation: job.generation,
        expectedEpoch: job.dataEpoch,
        examples: job.examples
          .filter((example) => example.purpose === "training")
          .map((example) => ({ input: example.input, result: example.result })),
        resultConfig: job.result,
        acceptableError: job.acceptableError,
      });
    } finally {
      await runtime.close();
    }
  },
});
