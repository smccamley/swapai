import { EventEmitter } from "node:events";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { delimiter, dirname, isAbsolute, join, relative } from "node:path";
import { PassThrough, Writable } from "node:stream";

import { describe, expect, it, vi } from "vitest";

import {
  NEEDLE_VERSION,
  NeedleModelArtifactError,
  createNeedleRuntime,
  hasNeedleModelArtifacts,
  type NeedleSpawn,
  type RunCommand,
} from "../src/runtime.js";
import { openStorage } from "../src/storage.js";

const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite") as typeof import("node:sqlite");

async function makeFakeCommands(dataDirectory: string) {
  const calls: Array<{ command: string; args: readonly string[] }> = [];

  const runCommand: RunCommand = async (command, args, options) => {
    calls.push({ command, args });

    if (args[0] === "venv") {
      const environmentDirectory = args.at(-1)!;
      const python =
        process.platform === "win32"
          ? join(environmentDirectory, "Scripts", "python.exe")
          : join(environmentDirectory, "bin", "python");
      await mkdir(join(python, ".."), { recursive: true });
      await writeFile(python, "fake python");
    }

    const outputIndex = args.indexOf("--output");
    if (outputIndex >= 0) {
      const trainingIndex = args.indexOf("--training-data");
      let trainingInput = options?.stdin ?? "";
      let numberLabels: string | undefined;
      if (args.includes("--numeric-labels-from-stdin")) {
        const newline = trainingInput.indexOf("\n");
        numberLabels = trainingInput.slice(0, newline);
        trainingInput = trainingInput.slice(newline + 1);
      }
      await writeFile(args[trainingIndex + 1]!, trainingInput);
      await writeFile(args[outputIndex + 1]!, "fake cact");
      if (numberLabels !== undefined) {
        await writeFile(`${args[outputIndex + 1]!}.numbers.json`, numberLabels);
      }
    }

    return { stdout: "", stderr: "" };
  };

  return { dataDirectory, calls, runCommand };
}

class FakeNeedleProcess extends EventEmitter {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly stdin: Writable;
  readonly requests: unknown[] = [];

  constructor(
    private readonly stops = true,
    private readonly result: unknown = true,
  ) {
    super();
    let buffered = "";
    this.stdin = new Writable({
      write: (chunk, _encoding, callback) => {
        buffered += chunk.toString();
        const lines = buffered.split("\n");
        buffered = lines.pop() ?? "";
        for (const line of lines) {
          if (!line) continue;
          const request = JSON.parse(line) as {
            id?: number;
            type: string;
            input?: string;
          };
          this.requests.push(request);
          if (request.type === "classify") {
            this.stdout.write(
              `${JSON.stringify({ id: request.id, ok: true, result: this.result })}\n`,
            );
          } else if (request.type === "close") {
            this.stdout.write(`${JSON.stringify({ type: "closed" })}\n`);
            if (this.stops) queueMicrotask(() => this.emit("exit", 0, null));
          }
        }
        callback();
      },
    });
    queueMicrotask(() => {
      this.stdout.write(
        `${JSON.stringify({ type: "ready", needleVersion: NEEDLE_VERSION })}\n`,
      );
    });
  }

  kill(signal?: NodeJS.Signals) {
    if (this.stops) this.emit("exit", null, signal ?? "SIGTERM");
    return true;
  }
}

describe("managed Needle runtime", () => {
  it("creates a private environment and pins cactus-needle", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-runtime-"));
    const fake = await makeFakeCommands(dataDirectory);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand },
    });

    await runtime.ready();
    await runtime.ready();

    expect(fake.calls).toEqual([
      { command: "uv", args: ["--version"] },
      expect.objectContaining({
        command: "uv",
        args: expect.arrayContaining(["venv", "--python", "3.12"]),
      }),
      expect.objectContaining({
        command: "uv",
        args: expect.arrayContaining([
          "pip",
          "install",
          "--require-hashes",
          "--requirement",
        ]),
      }),
      expect.objectContaining({
        args: ["-c", expect.stringContaining("needle.__version__")],
      }),
    ]);

    const marker = JSON.parse(
      await readFile(
        join(
          dataDirectory,
          "runtime",
          `needle-${NEEDLE_VERSION}`,
          "installed.json",
        ),
        "utf8",
      ),
    );
    expect(marker).toMatchObject({
      needleVersion: NEEDLE_VERSION,
      requirementsDigest: expect.stringMatching(/^[a-f0-9]{64}$/),
    });

    await runtime.close();
  });

  it("repairs a marked environment when its imports fail", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-repair-"));
    const initial = await makeFakeCommands(dataDirectory);
    const first = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: initial.runCommand },
    });
    await first.ready();
    await first.close();

    const repaired = await makeFakeCommands(dataDirectory);
    let healthChecks = 0;
    const runCommand: RunCommand = async (command, args, options) => {
      if (args[0] === "-c" && healthChecks++ === 0) {
        repaired.calls.push({ command, args });
        throw new Error("broken environment");
      }
      return repaired.runCommand(command, args, options);
    };
    const second = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand },
    });

    await second.ready();

    expect(healthChecks).toBe(2);
    expect(
      repaired.calls.filter((call) =>
        call.args.includes("--require-hashes"),
      ),
    ).toHaveLength(1);
    expect(
      repaired.calls.find((call) => call.args.includes("--require-hashes"))?.args,
    ).toContain("--reinstall");
    await second.close();
  });

  it("writes Needle JSONL and trains a .cact in the classifier generation", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-train-"));
    const fake = await makeFakeCommands(dataDirectory);
    const runtime = createNeedleRuntime({
      dataDirectory: relative(process.cwd(), dataDirectory),
      dependencies: { runCommand: fake.runCommand },
    });

    const trained = await runtime.train({
      classifierName: "accountant/relevance",
      generation: 2,
      expectedEpoch: 0,
      examples: [
        { input: "I owe you £5", result: true },
        { input: "Sunny outside", result: false },
      ],
      resultConfig: { type: "boolean" },
    });

    expect(trained.needleVersion).toBe(NEEDLE_VERSION);
    expect(trained.modelPath).toMatch(
      /generation-2[/\\]candidates[/\\][^/\\]+[/\\]model\.cact$/,
    );
    expect(await readFile(trained.modelPath, "utf8")).toBe("fake cact");

    const trainCall = fake.calls.find((call) => call.args.includes("train"));
    expect(trainCall).toBeDefined();
    const trainingPath = trainCall!.args[trainCall!.args.indexOf("--training-data") + 1]!;
    expect(isAbsolute(trainingPath)).toBe(true);
    const rows = (await readFile(trainingPath, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(rows).toHaveLength(2);
    expect(rows[0].answers[0].arguments.result).toBe(true);
    expect(trainCall!.args).toEqual(
      expect.arrayContaining([
        "--artifact-lock-database",
        "--main-database",
        "--classifier-name",
        "accountant/relevance",
        "--expected-epoch",
        "0",
      ]),
    );

    await runtime.close();
  });

  it("keeps the artifact lock in the training command after its parent storage closes", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-child-lock-"));
    const name = "orphan-safe-training";
    const storageOptions = {
      dataDirectory,
      name,
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
    const parentStorage = openStorage(storageOptions);
    const fake = await makeFakeCommands(dataDirectory);
    let signalLockHeld!: () => void;
    const lockHeld = new Promise<void>((resolve) => {
      signalLockHeld = resolve;
    });
    let releaseTraining!: () => void;
    const trainingReleased = new Promise<void>((resolve) => {
      releaseTraining = resolve;
    });
    const runCommand: RunCommand = async (command, args, options) => {
      if (!args.includes("train")) return fake.runCommand(command, args, options);
      const lockPath = args[args.indexOf("--artifact-lock-database") + 1]!;
      const childLock = new DatabaseSync(lockPath);
      childLock.exec("BEGIN IMMEDIATE");
      signalLockHeld();
      await trainingReleased;
      const trainingPath = args[args.indexOf("--training-data") + 1]!;
      const outputPath = args[args.indexOf("--output") + 1]!;
      await mkdir(dirname(outputPath), { recursive: true });
      await writeFile(trainingPath, options?.stdin ?? "");
      await writeFile(outputPath, "fake cact");
      childLock.exec("COMMIT");
      childLock.close();
      return { stdout: "", stderr: "" };
    };
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand },
    });
    const training = runtime.train({
      classifierName: name,
      generation: 1,
      expectedEpoch: 0,
      examples: [{ input: "private input", result: true }],
      resultConfig: { type: "boolean" },
    });
    await lockHeld;

    parentStorage.close();
    const clearer = openStorage(storageOptions);
    clearer.beginClearTrainingData();
    expect(clearer.claimArtifactWriteLock()).toBe(false);
    releaseTraining();
    await training;
    expect(clearer.claimArtifactWriteLock()).toBe(true);
    clearer.releaseArtifactWriteLock();
    clearer.close();
    await runtime.close();
  });

  it("does not write numeric labels after the training epoch was cleared", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-number-clear-race-"));
    const name = "numeric-clear-race";
    const storage = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: 10,
      config: {
        result: { type: "number", min: 0, max: 1 },
        retrainOnCount: 2,
        acceptableError: 0.1,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
    });
    const fake = await makeFakeCommands(dataDirectory);
    let signalReadyCheck!: () => void;
    const readyCheckStarted = new Promise<void>((resolve) => {
      signalReadyCheck = resolve;
    });
    let resumeReady!: () => void;
    const readyCanFinish = new Promise<void>((resolve) => {
      resumeReady = resolve;
    });
    let paused = false;
    let modelPath = "";
    const runCommand: RunCommand = async (command, args, options) => {
      if (!paused && command === "uv" && args[0] === "--version") {
        paused = true;
        signalReadyCheck();
        await readyCanFinish;
      }
      if (args.includes("train")) {
        modelPath = args[args.indexOf("--output") + 1]!;
        expect(args).toContain("--numeric-labels-from-stdin");
        expect(JSON.parse((options?.stdin ?? "").split("\n")[0]!)).toEqual({
          format: 1,
          values: [0.08, 0.92],
        });
        const database = new DatabaseSync(
          args[args.indexOf("--main-database") + 1]!,
          { readOnly: true },
        );
        const row = database.prepare(`
          SELECT data_epoch, clear_pending FROM classifiers WHERE name = ?
        `).get(name) as { data_epoch: number; clear_pending: number };
        database.close();
        const expectedEpoch = Number(
          args[args.indexOf("--expected-epoch") + 1],
        );
        if (row.data_epoch !== expectedEpoch || row.clear_pending !== 0) {
          throw new Error("Classifier training was superseded by data erasure");
        }
      }
      return fake.runCommand(command, args, options);
    };
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand },
    });
    const training = runtime.train({
      classifierName: name,
      generation: 1,
      expectedEpoch: 0,
      acceptableError: 0.1,
      resultConfig: { type: "number", min: 0, max: 1 },
      examples: [
        { input: "weather", result: 0.08 },
        { input: "invoice", result: 0.92 },
      ],
    });

    await readyCheckStarted;
    storage.clearTrainingData();
    resumeReady();
    await expect(training).rejects.toMatchObject({
      message: "Needle training failed.",
      cause: expect.objectContaining({
        message: expect.stringMatching(/superseded by data erasure/i),
      }),
    });
    await expect(readFile(modelPath, "utf8")).rejects.toMatchObject({
      code: "ENOENT",
    });
    await expect(readFile(`${modelPath}.numbers.json`, "utf8")).rejects.toMatchObject({
      code: "ENOENT",
    });
    await runtime.close();
    storage.close();
  });

  it("deletes one classifier's artifacts without deleting other classifiers or the managed runtime", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-clear-"));
    const fake = await makeFakeCommands(dataDirectory);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand },
    });
    const removed = await runtime.train({
      classifierName: "customer-to-remove",
      generation: 1,
      expectedEpoch: 0,
      examples: [{ input: "private input", result: true }],
      resultConfig: { type: "boolean" },
    });
    const preserved = await runtime.train({
      classifierName: "other-customer",
      generation: 1,
      expectedEpoch: 0,
      examples: [{ input: "other input", result: false }],
      resultConfig: { type: "boolean" },
    });

    await runtime.clearClassifierArtifacts("customer-to-remove");

    await expect(readFile(removed.modelPath, "utf8")).rejects.toMatchObject({
      code: "ENOENT",
    });
    await expect(readFile(preserved.modelPath, "utf8")).resolves.toBe(
      "fake cact",
    );
    await expect(
      readFile(
        join(
          dataDirectory,
          "runtime",
          `needle-${NEEDLE_VERSION}`,
          "installed.json",
        ),
        "utf8",
      ),
    ).resolves.toContain(NEEDLE_VERSION);
    await runtime.close();
  });

  it("deletes one stale classifier generation without deleting its current generation", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-generation-clear-"));
    const fake = await makeFakeCommands(dataDirectory);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand },
    });
    const stale = await runtime.train({
      classifierName: "shared-classifier",
      generation: 1,
      expectedEpoch: 0,
      examples: [{ input: "stale input", result: true }],
      resultConfig: { type: "boolean" },
    });
    const current = await runtime.train({
      classifierName: "shared-classifier",
      generation: 2,
      expectedEpoch: 0,
      examples: [{ input: "current input", result: false }],
      resultConfig: { type: "boolean" },
    });

    await runtime.clearClassifierArtifactsThroughGeneration(
      "shared-classifier",
      1,
    );

    await expect(readFile(stale.modelPath, "utf8")).rejects.toMatchObject({
      code: "ENOENT",
    });
    await expect(readFile(current.modelPath, "utf8")).resolves.toBe("fake cact");
    await runtime.close();
  });

  it("rejects unsafe classifier generation deletion values", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-generation-path-"));
    const fake = await makeFakeCommands(dataDirectory);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand },
    });

    await expect(
      runtime.clearClassifierGenerationArtifacts("classifier", -1),
    ).rejects.toThrow(/non-negative safe integer/i);
    await expect(
      runtime.clearClassifierGenerationArtifacts(
        "classifier",
        Number.MAX_SAFE_INTEGER + 1,
      ),
    ).rejects.toThrow(/non-negative safe integer/i);
    await runtime.close();
  });

  it("loads one worker, classifies over NDJSON, and closes it", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-serve-"));
    const fake = await makeFakeCommands(dataDirectory);
    const worker = new FakeNeedleProcess();
    const spawnProcess: NeedleSpawn = vi.fn(() => worker);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand, spawnProcess },
    });
    const modelPath = join(dataDirectory, "model.cact");
    await writeFile(modelPath, "fake cact");

    const model = await runtime.loadModel({
      modelPath,
      resultConfig: { type: "boolean" },
    });

    await expect(model.classify("I owe you £5")).resolves.toBe(true);
    expect(worker.requests).toContainEqual({
      id: 1,
      type: "classify",
      input: "I owe you £5",
    });

    await model.close();
    expect(worker.requests).toContainEqual({ type: "close" });
    await runtime.close();
  });

  it("persists numeric labels with the model and decodes the selected label", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-number-labels-"));
    const name = "numeric-labels";
    const fake = await makeFakeCommands(dataDirectory);
    const worker = new FakeNeedleProcess(true, "higher_score");
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: {
        runCommand: fake.runCommand,
        spawnProcess: vi.fn(() => worker),
      },
    });
    const storage = openStorage({
      dataDirectory,
      name,
      maxTrainingSet: 10,
      config: {
        result: { type: "number", min: 0, max: 1 },
        retrainOnCount: 2,
        acceptableError: 0.1,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
    });

    const trained = await runtime.train({
      classifierName: name,
      generation: 1,
      expectedEpoch: 0,
      resultConfig: { type: "number", min: 0, max: 1 },
      examples: [
        { input: "newsletter", result: 0.08 },
        { input: "invoice", result: 0.92 },
      ],
    });
    expect(
      JSON.parse(await readFile(`${trained.modelPath}.numbers.json`, "utf8")),
    ).toEqual({ format: 1, values: [0.08, 0.92] });
    const trainCall = fake.calls.find((call) => call.args.includes("train"))!;
    const trainingPath = trainCall.args[trainCall.args.indexOf("--training-data") + 1]!;
    const rows = (await readFile(trainingPath, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(rows.map((row) => row.answers[0].arguments.result)).toEqual([
      "lower_score",
      "higher_score",
    ]);
    expect(rows[0].tools[0].parameters.properties.result.enum).toEqual([
      "lower_score",
      "higher_score",
    ]);

    const model = await runtime.loadModel({
      modelPath: trained.modelPath,
      resultConfig: { type: "number", min: 0, max: 1 },
    });
    await expect(model.classify("invoice")).resolves.toBe(0.92);
    await model.close();
    await runtime.close();
    storage.close();
  });

  it("rejects an empty numeric label sidecar as a broken model artifact", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-empty-labels-"));
    const fake = await makeFakeCommands(dataDirectory);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand },
    });
    const modelPath = join(dataDirectory, "model.cact");
    const resultConfig = { type: "number", min: 0, max: 1 } as const;
    await writeFile(modelPath, "fake cact");
    await writeFile(
      `${modelPath}.numbers.json`,
      JSON.stringify({ format: 1, values: [] }),
    );

    expect(hasNeedleModelArtifacts({ modelPath, resultConfig })).toBe(false);
    await expect(runtime.loadModel({ modelPath, resultConfig })).rejects.toBeInstanceOf(
      NeedleModelArtifactError,
    );
    await runtime.close();
  });

  it("reports failure when a model worker remains alive after SIGKILL", async () => {
    vi.useFakeTimers();
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-stubborn-model-"));
    const fake = await makeFakeCommands(dataDirectory);
    const worker = new FakeNeedleProcess(false);
    const spawnProcess: NeedleSpawn = vi.fn(() => worker);
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand: fake.runCommand, spawnProcess },
    });
    const modelPath = join(dataDirectory, "model.cact");
    await writeFile(modelPath, "fake cact");
    const model = await runtime.loadModel({
      modelPath,
      resultConfig: { type: "boolean" },
    });

    const closing = model.close();
    const rejected = expect(closing).rejects.toThrow(/SIGKILL/i);
    await vi.advanceTimersByTimeAsync(9_100);
    await rejected;
    vi.useRealTimers();
  });

  it("keeps all caches and telemetry controls inside dataDirectory", async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "swapai-env-"));
    const fake = await makeFakeCommands(dataDirectory);
    let environment: NodeJS.ProcessEnv | undefined;
    const runCommand: RunCommand = async (command, args, options) => {
      environment = options?.env;
      return fake.runCommand(command, args, options);
    };
    const runtime = createNeedleRuntime({
      dataDirectory,
      dependencies: { runCommand },
    });

    await runtime.ready();

    expect(environment).toMatchObject({
      NEEDLE_TELEMETRY: "0",
      DO_NOT_TRACK: "1",
      HF_HOME: join(dataDirectory, "runtime", "huggingface"),
      UV_CACHE_DIR: join(dataDirectory, "runtime", "uv-cache"),
    });
    expect(environment?.PATH?.split(delimiter)).toBeDefined();
    await runtime.close();
  });
});
