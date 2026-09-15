import { EventEmitter } from "node:events";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { PassThrough, Writable } from "node:stream";

import { describe, expect, it, vi } from "vitest";

import {
  NEEDLE_VERSION,
  createNeedleRuntime,
  type NeedleSpawn,
  type RunCommand,
} from "../src/runtime.js";

async function makeFakeCommands(dataDirectory: string) {
  const calls: Array<{ command: string; args: readonly string[] }> = [];

  const runCommand: RunCommand = async (command, args) => {
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
      await writeFile(args[outputIndex + 1]!, "fake cact");
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

  constructor() {
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
              `${JSON.stringify({ id: request.id, ok: true, result: true })}\n`,
            );
          } else if (request.type === "close") {
            this.stdout.write(`${JSON.stringify({ type: "closed" })}\n`);
            queueMicrotask(() => this.emit("exit", 0, null));
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

  kill() {
    this.emit("exit", null, "SIGTERM");
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
      dataDirectory,
      dependencies: { runCommand: fake.runCommand },
    });

    const trained = await runtime.train({
      classifierName: "accountant/relevance",
      generation: 2,
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
    const rows = (await readFile(trainingPath, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(rows).toHaveLength(2);
    expect(rows[0].answers[0].arguments.result).toBe(true);

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
