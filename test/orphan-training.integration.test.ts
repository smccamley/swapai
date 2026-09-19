import { createHash } from "node:crypto";
import { once } from "node:events";
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";
import { describe, expect, it } from "vitest";

import { createNeedleRuntime } from "../src/runtime.js";
import { openStorage } from "../src/storage.js";

const workerPath = fileURLToPath(
  new URL("../python/swapai_worker.py", import.meta.url),
);

describe.skipIf(process.platform === "win32")(
  "training whose Node parent crashes",
  () => {
    it(
      "keeps erasure blocked until the orphaned Python worker stops writing",
      async () => {
        const dataDirectory = await mkdtemp(
          join(tmpdir(), "swapai-orphan-training-"),
        );
        const classifierName = "orphaned-python-training";
        const controlDirectory = join(dataDirectory, "fake-needle-control");
        const fakeNeedleDirectory = join(dataDirectory, "fake-python");
        const parentScript = join(dataDirectory, "training-parent.mjs");
        const pythonPidPath = join(controlDirectory, "python-pid");
        const lockHeldPath = join(controlDirectory, "lock-held");
        const continuePath = join(controlDirectory, "continue");
        const orphanWritePath = join(controlDirectory, "orphan-write");
        const releasePath = join(controlDirectory, "release");
        const classifierHash = createHash("sha256")
          .update(classifierName)
          .digest("hex");
        const classifierKey = `${classifierName}-${classifierHash.slice(0, 12)}`;
        const generationDirectory = join(
          dataDirectory,
          "classifiers",
          classifierKey,
          "generation-1",
        );
        const modelPath = join(generationDirectory, "candidates", "old", "model.cact");
        const trainingPath = join(
          generationDirectory,
          "candidates",
          "old",
          "training.jsonl",
        );
        const checkpointDirectory = join(generationDirectory, "checkpoints");
        const lockPath = join(dataDirectory, "locks", `${classifierHash}.sqlite`);
        const storageOptions = {
          dataDirectory,
          name: classifierName,
          maxTrainingSet: 10,
          config: {
            result: { type: "boolean" as const },
            retrainOnCount: 2,
            acceptableError: 0,
            retestInterval: 100,
            retestRevertOn: 3,
            model: "needle2",
          },
        };
        let parent: ChildProcess | undefined;
        let pythonPid: number | undefined;
        let clearStorage: ReturnType<typeof openStorage> | undefined;
        let runtime: ReturnType<typeof createNeedleRuntime> | undefined;

        try {
          await mkdir(join(fakeNeedleDirectory, "needle", "model"), {
            recursive: true,
          });
          await mkdir(controlDirectory, { recursive: true });
          await writeFile(
            join(fakeNeedleDirectory, "needle", "__init__.py"),
            '__version__ = "2.0.14"\n',
          );
          await writeFile(
            join(fakeNeedleDirectory, "needle", "model", "__init__.py"),
            "",
          );
          await writeFile(
            join(fakeNeedleDirectory, "needle", "model", "finetune.py"),
            `from pathlib import Path
import os
import time

control = Path(os.environ["SWAPAI_FAKE_NEEDLE_CONTROL"])

def wait_for(name):
    deadline = time.monotonic() + 3
    path = control / name
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for " + name)
        time.sleep(0.01)

def finetune_local(args):
    (control / "lock-held").write_text(str(os.getpid()), encoding="utf-8")
    wait_for("continue")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint written by orphan", encoding="utf-8")
    (control / "orphan-write").write_text(str(os.getpid()), encoding="utf-8")
    wait_for("release")
    Path(args.out).write_text("adapter written by orphan", encoding="utf-8")

def build_main(args):
    Path(args.out).write_text("old model written by orphan", encoding="utf-8")
`,
          );
          await writeFile(
            parentScript,
            `import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { delimiter } from "node:path";

const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite");
const [python, worker, fakePython, control, pidPath, mainDatabase, classifierName, ...workerArgs] = process.argv.slice(2);
const database = new DatabaseSync(mainDatabase);
database.function("swapai_writer_version", { deterministic: true }, () => 3);
database.prepare(\`
  UPDATE classifiers
  SET training_lease_owner = ?,
      training_lease_epoch = data_epoch,
      training_lease_until = ?
  WHERE name = ?
\`).run(\`node-parent-\${process.pid}\`, Date.now() + 60_000, classifierName);
database.close();
const child = spawn(python, [worker, ...workerArgs], {
  detached: true,
  env: {
    ...process.env,
    PYTHONPATH: fakePython + (process.env.PYTHONPATH ? delimiter + process.env.PYTHONPATH : ""),
    SWAPAI_FAKE_NEEDLE_CONTROL: control,
  },
  stdio: ["pipe", "ignore", "ignore"],
});
writeFileSync(pidPath, String(child.pid));
child.stdin.end('{"messages":[{"role":"user","content":"old private example"}],"answers":[{"name":"classify","arguments":{"result":true}}]}\\n');
child.once("exit", (code) => process.exit(code ?? 1));
`,
          );

          const setupStorage = openStorage(storageOptions);
          setupStorage.addExample("old private example", true);
          setupStorage.close();

          parent = spawn(
            process.execPath,
            [
              parentScript,
              process.env.PYTHON ?? "python3",
              workerPath,
              fakeNeedleDirectory,
              controlDirectory,
              pythonPidPath,
              join(dataDirectory, "swapai.sqlite"),
              classifierName,
              "train",
              "--training-data",
              trainingPath,
              "--output",
              modelPath,
              "--checkpoint-dir",
              checkpointDirectory,
              "--epochs",
              "1",
              "--artifact-lock-database",
              lockPath,
              "--main-database",
              join(dataDirectory, "swapai.sqlite"),
              "--classifier-name",
              classifierName,
              "--expected-epoch",
              "0",
            ],
            { stdio: "ignore" },
          );

          await waitForFile(lockHeldPath);
          pythonPid = Number(await readFile(pythonPidPath, "utf8"));
          expect(Number.isSafeInteger(pythonPid)).toBe(true);
          expect(Number(await readFile(lockHeldPath, "utf8"))).toBe(pythonPid);

          const parentExit = once(parent, "exit");
          parent.kill("SIGKILL");
          const [, parentSignal] = await parentExit;
          expect(parentSignal).toBe("SIGKILL");
          expect(() => process.kill(pythonPid!, 0)).not.toThrow();

          await writeFile(continuePath, "continue");
          await waitForFile(orphanWritePath);
          expect(Number(await readFile(orphanWritePath, "utf8"))).toBe(pythonPid);

          clearStorage = openStorage(storageOptions);
          expect(clearStorage.snapshot()).toMatchObject({
            trainingLeaseOwner: `node-parent-${parent.pid}`,
            trainingLeaseEpoch: 0,
          });
          const clearEpoch = clearStorage.beginClearTrainingData();
          const generationCutoff = clearStorage.snapshot().clearArtifactGenerationMax;
          expect(generationCutoff).toBe(1);
          expect(clearStorage.snapshot()).toMatchObject({
            trainingLeaseOwner: `node-parent-${parent.pid}`,
            trainingLeaseEpoch: 0,
          });
          expect(clearStorage.claimArtifactWriteLock()).toBe(false);

          await writeFile(releasePath, "release");
          await waitForFile(modelPath);
          await waitFor(() => clearStorage!.claimArtifactWriteLock());
          expect(await readFile(modelPath, "utf8")).toBe(
            "old model written by orphan",
          );

          expect(clearStorage.eraseTrainingData(clearEpoch)).not.toBeNull();
          runtime = createNeedleRuntime({ dataDirectory });
          await runtime.clearClassifierArtifactsThroughGeneration(
            classifierName,
            generationCutoff!,
          );
          expect(clearStorage.finishClearTrainingData(clearEpoch)).toBe(true);
          clearStorage.releaseArtifactWriteLock();

          expect(clearStorage.snapshot()).toMatchObject({
            activeExampleCount: 0,
            clearPending: false,
            trained: false,
          });
          await expect(access(modelPath)).rejects.toMatchObject({ code: "ENOENT" });
        } finally {
          try {
            clearStorage?.releaseArtifactWriteLock();
          } catch {
            // The lock was not held or storage had already closed.
          }
          clearStorage?.close();
          await runtime?.close();
          if (parent?.exitCode === null && parent.signalCode === null) {
            parent.kill("SIGKILL");
          }
          if (pythonPid !== undefined) {
            try {
              process.kill(pythonPid, "SIGKILL");
            } catch {
              // The worker already exited normally.
            }
          }
          await rm(dataDirectory, { recursive: true, force: true });
        }
      },
      5_000,
    );
  },
);

async function waitForFile(path: string): Promise<void> {
  await waitFor(async () => {
    try {
      await access(path);
      return true;
    } catch {
      return false;
    }
  });
}

async function waitFor(check: () => boolean | Promise<boolean>): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!(await check())) {
    if (Date.now() >= deadline) {
      throw new Error("Timed out waiting for orphaned training state");
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
