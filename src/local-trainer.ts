import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { needleModelVersion } from "./runtime.js";
import {
  verifyTrainingResult,
  writeTrainingBundle,
} from "./training-bundle.js";
import type { TrainingProvider } from "./types.js";

export interface LocalTrainerOptions {
  readonly python?: string;
  readonly runnerPath?: string;
}

export const localTrainer = (
  options: LocalTrainerOptions = {},
): TrainingProvider => ({
  name: "local",
  train: async (job) => {
    const bundleDirectory = await writeTrainingBundle(job);
    const runnerPath = options.runnerPath ?? fileURLToPath(
      new URL("../trainer/train.py", import.meta.url),
    );
    await runProcess(options.python ?? "python3", [
      runnerPath,
      "--job-directory",
      bundleDirectory,
    ]);
    const candidate = await verifyTrainingResult(bundleDirectory);
    const expected = needleModelVersion(job.result);
    if (candidate.needleVersion !== expected) {
      throw new Error(
        `Trainer returned Needle ${candidate.needleVersion}; expected ${expected}`,
      );
    }
    return candidate;
  },
});

const runProcess = (command: string, args: readonly string[]): Promise<void> =>
  new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => { stderr += String(chunk); });
    child.once("error", reject);
    child.once("close", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(
        `${command} exited with ${signal ?? code}${stderr === "" ? "" : `: ${stderr}`}`,
      ));
    });
  });
