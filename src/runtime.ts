import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { readFileSync, statSync } from "node:fs";
import {
  chmod,
  copyFile,
  mkdir,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { createInterface } from "node:readline";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { SwapAIError } from "./errors.js";
import {
  createNeedleNumberLabels,
  createNeedleTool,
  createTrainingLine,
  decodeNeedleResult,
  validateNeedleResult,
  type NeedleExample,
  type NeedleNumberLabels,
  type NeedleResultConfig,
  type NeedleResultValue,
} from "./needle.js";

export const NEEDLE_VERSION = "2.0.14";
export const NEEDLE_NUMBER_MODEL_VERSION =
  `${NEEDLE_VERSION}/number-buckets-v1`;
const UV_VERSION = "0.11.4";
const PYTHON_VERSION = "3.12";

export interface CommandOptions {
  readonly cwd?: string;
  readonly env?: NodeJS.ProcessEnv;
  readonly timeoutMs?: number;
  readonly stdin?: string;
}

export type RunCommand = (
  command: string,
  args: readonly string[],
  options?: CommandOptions,
) => Promise<{ readonly stdout: string; readonly stderr: string }>;

export interface NeedleChildProcess {
  readonly stdin: NodeJS.WritableStream;
  readonly stdout: NodeJS.ReadableStream;
  readonly stderr: NodeJS.ReadableStream;
  on(event: string, listener: (...args: any[]) => void): this;
  once(event: string, listener: (...args: any[]) => void): this;
  kill(signal?: NodeJS.Signals): boolean;
}

export type NeedleSpawn = (
  command: string,
  args: readonly string[],
  options: { readonly cwd: string; readonly env: NodeJS.ProcessEnv },
) => NeedleChildProcess;

export interface NeedleRuntimeDependencies {
  readonly runCommand?: RunCommand;
  readonly spawnProcess?: NeedleSpawn;
  readonly fetch?: typeof globalThis.fetch;
  readonly workerPath?: string;
  readonly requirementsPath?: string;
}

export interface CreateNeedleRuntimeOptions {
  readonly dataDirectory: string;
  readonly onBackgroundError?: (error: NeedleRuntimeError) => void;
  readonly dependencies?: NeedleRuntimeDependencies;
}

export interface TrainNeedleModelOptions {
  readonly classifierName: string;
  readonly generation: number;
  readonly expectedEpoch: number;
  readonly examples: readonly NeedleExample[];
  readonly resultConfig: NeedleResultConfig;
  readonly acceptableError?: number;
  readonly epochs?: number;
}

export interface TrainedNeedleModel {
  readonly modelPath: string;
  readonly needleVersion: string;
}

export function needleModelVersion(
  resultConfig: NeedleResultConfig,
): string {
  return resultConfig.type === "number"
    ? NEEDLE_NUMBER_MODEL_VERSION
    : NEEDLE_VERSION;
}

export interface LoadNeedleModelOptions {
  readonly modelPath: string;
  readonly resultConfig: NeedleResultConfig;
}

export interface LoadedNeedleModel {
  classify(input: string): Promise<NeedleResultValue>;
  close(): Promise<void>;
}

export class NeedleModelArtifactError extends SwapAIError {
  constructor(message: string, cause?: unknown) {
    super(
      "service_unavailable",
      message,
      cause === undefined ? undefined : { cause },
    );
    this.name = "NeedleModelArtifactError";
  }
}

export function hasNeedleModelArtifacts(
  options: LoadNeedleModelOptions,
): boolean {
  try {
    if (!statSync(options.modelPath).isFile()) return false;
    if (options.resultConfig.type !== "number") return true;
    const saved = JSON.parse(
      readFileSync(`${options.modelPath}.numbers.json`, "utf8"),
    ) as { format?: unknown; values?: unknown };
    if (
      saved.format !== 1 ||
      !Array.isArray(saved.values) ||
      saved.values.length === 0
    ) return false;
    const savedValues = saved.values as number[];
    const labels = createNeedleNumberLabels(savedValues, options.resultConfig);
    return labels.values.length === savedValues.length && labels.values.every(
      (value, index) => value === savedValues[index],
    );
  } catch {
    return false;
  }
}

export interface NeedleRuntime {
  ready(): Promise<void>;
  train(options: TrainNeedleModelOptions): Promise<TrainedNeedleModel>;
  loadModel(options: LoadNeedleModelOptions): Promise<LoadedNeedleModel>;
  clearClassifierArtifacts(classifierName: string): Promise<void>;
  clearClassifierGenerationArtifacts(
    classifierName: string,
    generation: number,
  ): Promise<void>;
  clearClassifierArtifactsThroughGeneration(
    classifierName: string,
    maximumGeneration: number,
  ): Promise<void>;
  close(): Promise<void>;
}

export class NeedleRuntimeError extends SwapAIError {
  declare readonly code: "service_unavailable" | "classification_failed";

  constructor(
    code: "service_unavailable" | "classification_failed",
    message: string,
    cause?: unknown,
  ) {
    super(code, message, cause === undefined ? undefined : { cause });
    this.name = "NeedleRuntimeError";
  }
}

export function createNeedleRuntime(
  options: CreateNeedleRuntimeOptions,
): NeedleRuntime {
  return new ManagedNeedleRuntime(options);
}

class ManagedNeedleRuntime implements NeedleRuntime {
  readonly #dataDirectory: string;
  readonly #runtimeDirectory: string;
  readonly #runCommand: RunCommand;
  readonly #spawnProcess: NeedleSpawn;
  readonly #fetch: typeof globalThis.fetch;
  readonly #workerPath: string;
  readonly #requirementsPath: string;
  readonly #onBackgroundError: ((error: NeedleRuntimeError) => void) | undefined;
  readonly #models = new Set<NeedleModelProcess>();
  #readyPromise: Promise<void> | undefined;
  #pythonPath: string | undefined;
  #environment: NodeJS.ProcessEnv | undefined;
  #closed = false;

  constructor(options: CreateNeedleRuntimeOptions) {
    this.#dataDirectory = resolve(options.dataDirectory);
    this.#runtimeDirectory = join(this.#dataDirectory, "runtime");
    this.#runCommand = options.dependencies?.runCommand ?? runCommand;
    this.#spawnProcess = options.dependencies?.spawnProcess ?? spawnProcess;
    this.#fetch = options.dependencies?.fetch ?? globalThis.fetch;
    this.#workerPath =
      options.dependencies?.workerPath ??
      fileURLToPath(new URL("../python/swapai_worker.py", import.meta.url));
    this.#requirementsPath =
      options.dependencies?.requirementsPath ??
      fileURLToPath(new URL("../python/requirements.lock", import.meta.url));
    this.#onBackgroundError = options.onBackgroundError;
  }

  ready(): Promise<void> {
    if (this.#closed) {
      return Promise.reject(
        new NeedleRuntimeError("service_unavailable", "Needle runtime is closed."),
      );
    }
    this.#readyPromise ??= this.#prepare().catch((error: unknown) => {
      this.#readyPromise = undefined;
      throw toRuntimeError("Could not prepare Needle.", error);
    });
    return this.#readyPromise;
  }

  async train(options: TrainNeedleModelOptions): Promise<TrainedNeedleModel> {
    if (!Number.isSafeInteger(options.generation) || options.generation < 0) {
      throw new NeedleRuntimeError(
        "service_unavailable",
        "Needle generation must be a non-negative integer.",
      );
    }
    if (!Number.isSafeInteger(options.expectedEpoch) || options.expectedEpoch < 0) {
      throw new NeedleRuntimeError(
        "service_unavailable",
        "Needle data epoch must be a non-negative integer.",
      );
    }
    if (options.examples.length === 0) {
      throw new NeedleRuntimeError(
        "service_unavailable",
        "Needle needs at least one training example.",
      );
    }
    await this.ready();

    const classifierDirectory = join(
      this.#dataDirectory,
      "classifiers",
      classifierKey(options.classifierName),
    );
    const generationDirectory = join(
      classifierDirectory,
      `generation-${options.generation}`,
    );
    const checkpointDirectory = join(generationDirectory, "checkpoints");
    const candidateDirectory = join(
      generationDirectory,
      "candidates",
      randomUUID(),
    );
    const trainingPath = join(candidateDirectory, "training.jsonl");
    const modelPath = join(candidateDirectory, "model.cact");
    await mkdir(checkpointDirectory, { recursive: true, mode: 0o700 });
    await mkdir(candidateDirectory, { recursive: true, mode: 0o700 });

    const numberLabels =
      options.resultConfig.type === "number"
        ? createNeedleNumberLabels(
            options.examples.map((example) => example.result as number),
            options.resultConfig,
            options.acceptableError,
          )
        : undefined;
    const lines = options.examples.map((example) =>
      createTrainingLine(
        example.input,
        example.result,
        options.resultConfig,
        numberLabels,
      ),
    );
    const lockPath = join(
      this.#dataDirectory,
      "locks",
      `${createHash("sha256").update(options.classifierName).digest("hex")}.sqlite`,
    );

    try {
      const trainArguments = [
        this.#workerPath,
        "train",
        "--training-data",
        trainingPath,
        "--output",
        modelPath,
        "--checkpoint-dir",
        checkpointDirectory,
        "--epochs",
        String(options.epochs ?? 10),
        "--artifact-lock-database",
        lockPath,
        "--main-database",
        join(this.#dataDirectory, "swapai.sqlite"),
        "--classifier-name",
        options.classifierName,
        "--expected-epoch",
        String(options.expectedEpoch),
      ];
      if (numberLabels) {
        trainArguments.push("--numeric-labels-from-stdin");
      }
      await this.#runCommand(
        this.#pythonPath!,
        trainArguments,
        {
          cwd: candidateDirectory,
          env: this.#environment!,
          timeoutMs: 6 * 60 * 60 * 1000,
          stdin: numberLabels
            ? `${JSON.stringify({ format: 1, values: numberLabels.values })}\n${lines.join("\n")}\n`
            : `${lines.join("\n")}\n`,
        },
      );
      await stat(modelPath);
    } catch (error) {
      throw toRuntimeError("Needle training failed.", error);
    }

    return {
      modelPath,
      needleVersion: needleModelVersion(options.resultConfig),
    };
  }

  async loadModel(options: LoadNeedleModelOptions): Promise<LoadedNeedleModel> {
    await this.ready();
    try {
      await stat(options.modelPath);
    } catch (error) {
      throw new NeedleModelArtifactError(
        `Needle model does not exist: ${options.modelPath}`,
        error,
      );
    }

    let numberLabels: NeedleNumberLabels | undefined;
    if (options.resultConfig.type === "number") {
      try {
        const saved = JSON.parse(
          await readFile(`${options.modelPath}.numbers.json`, "utf8"),
        ) as { format?: unknown; values?: unknown };
        if (
          saved.format !== 1 ||
          !Array.isArray(saved.values) ||
          saved.values.length === 0
        ) {
          throw new TypeError("Numeric label map has an invalid format.");
        }
        const savedValues = saved.values as unknown[];
        numberLabels = createNeedleNumberLabels(
          savedValues as number[],
          options.resultConfig,
        );
        if (
          numberLabels.values.length !== savedValues.length ||
          numberLabels.values.some(
            (value, index) => value !== savedValues[index],
          )
        ) {
          throw new TypeError("Numeric label map is not sorted and unique.");
        }
      } catch (error) {
        throw new NeedleModelArtifactError(
          "Could not load Needle numeric labels.",
          error,
        );
      }
    }

    const schemaPath = `${options.modelPath}.schema.json`;
    await writeFile(
      schemaPath,
      JSON.stringify([createNeedleTool(options.resultConfig, numberLabels)]),
      "utf8",
    );

    const process = this.#spawnProcess(
      this.#pythonPath!,
      [
        "-u",
        this.#workerPath,
        "serve",
        "--model",
        options.modelPath,
        "--schema",
        schemaPath,
      ],
      { cwd: dirname(options.modelPath), env: this.#environment! },
    );
    const model = new NeedleModelProcess(
      process,
      options.resultConfig,
      numberLabels,
      (error) => this.#onBackgroundError?.(error),
      () => this.#models.delete(model),
    );
    this.#models.add(model);
    try {
      await model.ready();
      return model;
    } catch (error) {
      this.#models.delete(model);
      process.kill("SIGTERM");
      throw toRuntimeError("Needle model could not start.", error);
    }
  }

  async clearClassifierArtifacts(classifierName: string): Promise<void> {
    await rm(
      join(this.#dataDirectory, "classifiers", classifierKey(classifierName)),
      { recursive: true, force: true },
    );
  }

  async clearClassifierGenerationArtifacts(
    classifierName: string,
    generation: number,
  ): Promise<void> {
    if (!Number.isSafeInteger(generation) || generation < 0) {
      throw new NeedleRuntimeError(
        "service_unavailable",
        "Classifier generation must be a non-negative safe integer.",
      );
    }
    const classifierDirectory = resolve(
      this.#dataDirectory,
      "classifiers",
      classifierKey(classifierName),
    );
    const generationDirectory = resolve(
      classifierDirectory,
      `generation-${generation}`,
    );
    if (!generationDirectory.startsWith(`${classifierDirectory}${sep}`)) {
      throw new NeedleRuntimeError(
        "service_unavailable",
        "Classifier generation path is outside its classifier directory.",
      );
    }
    await rm(generationDirectory, { recursive: true, force: true });
  }

  async clearClassifierArtifactsThroughGeneration(
    classifierName: string,
    maximumGeneration: number,
  ): Promise<void> {
    if (!Number.isSafeInteger(maximumGeneration) || maximumGeneration < 0) {
      throw new NeedleRuntimeError(
        "service_unavailable",
        "Maximum classifier generation must be a non-negative safe integer.",
      );
    }
    const classifierDirectory = join(
      this.#dataDirectory,
      "classifiers",
      classifierKey(classifierName),
    );
    let entries;
    try {
      entries = await readdir(classifierDirectory, { withFileTypes: true });
    } catch (error) {
      if (
        typeof error === "object" &&
        error !== null &&
        "code" in error &&
        error.code === "ENOENT"
      ) {
        return;
      }
      throw error;
    }
    await Promise.all(
      entries.map(async (entry) => {
        const match = /^generation-(\d+)$/.exec(entry.name);
        if (!entry.isDirectory() || match === null) return;
        const generation = Number(match[1]);
        if (generation > maximumGeneration) return;
        await rm(join(classifierDirectory, entry.name), {
          recursive: true,
          force: true,
        });
      }),
    );
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    await Promise.allSettled([...this.#models].map((model) => model.close()));
    this.#models.clear();
  }

  async #prepare(): Promise<void> {
    const installDirectory = join(
      this.#runtimeDirectory,
      `needle-${NEEDLE_VERSION}`,
    );
    const environmentDirectory = join(installDirectory, ".venv");
    const markerPath = join(installDirectory, "installed.json");
    await mkdir(installDirectory, { recursive: true });

    const baseEnvironment = this.#makeEnvironment();
    const uvPath = await this.#findOrInstallUv(baseEnvironment);
    const pythonPath = pythonIn(environmentDirectory);
    const environment = this.#makeEnvironment(uvPath, environmentDirectory);

    if (!(await exists(pythonPath))) {
      await this.#runCommand(
        uvPath,
        ["venv", "--python", PYTHON_VERSION, environmentDirectory],
        { cwd: installDirectory, env: environment, timeoutMs: 15 * 60 * 1000 },
      );
    }

    const requirementsDigest = createHash("sha256")
      .update(await readFile(this.#requirementsPath))
      .digest("hex");
    const marker = await readJson(markerPath);
    let installRequired =
      marker?.needleVersion !== NEEDLE_VERSION ||
      marker.requirementsDigest !== requirementsDigest;
    let forceReinstall = false;
    const healthCheck = [
      "-c",
      `import needle, jax, flax, optax; assert needle.__version__ == "${NEEDLE_VERSION}"`,
    ] as const;

    if (!installRequired) {
      try {
        await this.#runCommand(pythonPath, healthCheck, {
          cwd: installDirectory,
          env: environment,
          timeoutMs: 60_000,
        });
      } catch {
        installRequired = true;
        forceReinstall = true;
      }
    }

    if (installRequired) {
      await this.#runCommand(
        uvPath,
        [
          "pip",
          "install",
          "--python",
          pythonPath,
          ...(forceReinstall ? ["--reinstall"] : []),
          "--require-hashes",
          "--requirement",
          this.#requirementsPath,
        ],
        { cwd: installDirectory, env: environment, timeoutMs: 30 * 60 * 1000 },
      );
      await this.#runCommand(pythonPath, healthCheck, {
        cwd: installDirectory,
        env: environment,
        timeoutMs: 60_000,
      });
      await writeFile(
        markerPath,
        JSON.stringify({ needleVersion: NEEDLE_VERSION, requirementsDigest }),
        "utf8",
      );
    }

    this.#pythonPath = pythonPath;
    this.#environment = environment;
  }

  #makeEnvironment(
    uvPath?: string,
    environmentDirectory?: string,
  ): NodeJS.ProcessEnv {
    const pathParts = [
      environmentDirectory ? dirname(pythonIn(environmentDirectory)) : undefined,
      uvPath ? dirname(uvPath) : undefined,
      process.env.PATH,
    ].filter((part): part is string => Boolean(part));
    return {
      ...process.env,
      PATH: pathParts.join(process.platform === "win32" ? ";" : ":"),
      PYTHONNOUSERSITE: "1",
      PYTHONUNBUFFERED: "1",
      NEEDLE_TELEMETRY: "0",
      DO_NOT_TRACK: "1",
      HF_HOME: join(this.#runtimeDirectory, "huggingface"),
      UV_CACHE_DIR: join(this.#runtimeDirectory, "uv-cache"),
      XDG_CACHE_HOME: join(this.#runtimeDirectory, "cache"),
    };
  }

  async #findOrInstallUv(environment: NodeJS.ProcessEnv): Promise<string> {
    const candidates = [process.env.SWAPAI_UV_PATH, "uv"].filter(
      (candidate): candidate is string => Boolean(candidate),
    );
    for (const candidate of candidates) {
      try {
        await this.#runCommand(candidate, ["--version"], {
          env: environment,
          timeoutMs: 30_000,
        });
        return candidate;
      } catch {
        // Try the managed copy next.
      }
    }

    const managed = join(
      this.#runtimeDirectory,
      "tools",
      process.platform === "win32" ? "uv.exe" : "uv",
    );
    if (!(await exists(managed))) {
      await this.#downloadUv(managed, environment);
    }
    await this.#runCommand(managed, ["--version"], {
      env: environment,
      timeoutMs: 30_000,
    });
    return managed;
  }

  async #downloadUv(
    destination: string,
    environment: NodeJS.ProcessEnv,
  ): Promise<void> {
    const asset = uvAsset();
    const toolsDirectory = dirname(destination);
    const archivePath = join(toolsDirectory, asset);
    const extractDirectory = join(toolsDirectory, `uv-${UV_VERSION}`);
    await mkdir(extractDirectory, { recursive: true });
    const release = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}`;

    const [archiveResponse, sumsResponse] = await Promise.all([
      this.#fetch(`${release}/${asset}`, {
        signal: AbortSignal.timeout(5 * 60 * 1000),
      }),
      this.#fetch(`${release}/sha256.sum`, {
        signal: AbortSignal.timeout(5 * 60 * 1000),
      }),
    ]);
    if (!archiveResponse.ok || !sumsResponse.ok) {
      throw new Error(`Could not download the managed uv ${UV_VERSION} binary.`);
    }
    const archive = Buffer.from(await archiveResponse.arrayBuffer());
    const sums = await sumsResponse.text();
    const expected = checksumFor(asset, sums);
    const actual = createHash("sha256").update(archive).digest("hex");
    if (actual !== expected) {
      throw new Error("The managed uv download failed its SHA-256 check.");
    }
    await writeFile(archivePath, archive);
    await this.#runCommand(
      "tar",
      [asset.endsWith(".tar.gz") ? "-xzf" : "-xf", archivePath, "-C", extractDirectory],
      { env: environment, timeoutMs: 5 * 60 * 1000 },
    );
    const extracted = await findFile(
      extractDirectory,
      process.platform === "win32" ? "uv.exe" : "uv",
    );
    if (!extracted) throw new Error("The managed uv archive did not contain uv.");
    await copyFile(extracted, destination);
    if (process.platform !== "win32") await chmod(destination, 0o755);
  }
}

class NeedleModelProcess implements LoadedNeedleModel {
  readonly #process: NeedleChildProcess;
  readonly #resultConfig: NeedleResultConfig;
  readonly #numberLabels: NeedleNumberLabels | undefined;
  readonly #onBackgroundError: (error: NeedleRuntimeError) => void;
  readonly #onClose: () => void;
  readonly #pending = new Map<
    number,
    {
      readonly resolve: (value: NeedleResultValue) => void;
      readonly reject: (error: NeedleRuntimeError) => void;
    }
  >();
  readonly #readyPromise: Promise<void>;
  readonly #exitPromise: Promise<void>;
  #resolveReady!: () => void;
  #rejectReady!: (error: NeedleRuntimeError) => void;
  #resolveExit!: () => void;
  #requestId = 0;
  #ready = false;
  #exited = false;
  #closing = false;
  #stderr = "";

  constructor(
    process: NeedleChildProcess,
    resultConfig: NeedleResultConfig,
    numberLabels: NeedleNumberLabels | undefined,
    onBackgroundError: (error: NeedleRuntimeError) => void,
    onClose: () => void,
  ) {
    this.#process = process;
    this.#resultConfig = resultConfig;
    this.#numberLabels = numberLabels;
    this.#onBackgroundError = onBackgroundError;
    this.#onClose = onClose;
    this.#readyPromise = new Promise((resolve, reject) => {
      this.#resolveReady = resolve;
      this.#rejectReady = reject;
    });
    this.#exitPromise = new Promise((resolve) => {
      this.#resolveExit = resolve;
    });
    this.#listen();
  }

  ready(): Promise<void> {
    return withTimeout(
      this.#readyPromise,
      300_000,
      "Needle did not become ready within five minutes.",
    );
  }

  classify(input: string): Promise<NeedleResultValue> {
    if (!this.#ready || this.#closing || this.#exited) {
      return Promise.reject(
        new NeedleRuntimeError("service_unavailable", "Needle model is not running."),
      );
    }
    const id = ++this.#requestId;
    const result = new Promise<NeedleResultValue>((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#process.stdin.write(
        `${JSON.stringify({ id, type: "classify", input })}\n`,
        (error?: Error | null) => {
          if (!error) return;
          this.#pending.delete(id);
          reject(
            new NeedleRuntimeError(
              "service_unavailable",
              "Could not send input to Needle.",
              error,
            ),
          );
        },
      );
    });
    return withTimeout(
      result,
      60_000,
      "Needle did not classify the input within one minute.",
      () => this.#pending.delete(id),
    );
  }

  async close(): Promise<void> {
    if (this.#exited) return;
    if (!this.#closing) {
      this.#closing = true;
      this.#process.stdin.write(`${JSON.stringify({ type: "close" })}\n`);
    }
    try {
      await withTimeout(this.#exitPromise, 5_000, "Needle did not stop.");
    } catch {
      this.#process.kill("SIGTERM");
      try {
        await withTimeout(this.#exitPromise, 2_000, "Needle ignored SIGTERM.");
      } catch {
        this.#process.kill("SIGKILL");
        await withTimeout(this.#exitPromise, 2_000, "Needle ignored SIGKILL.");
      }
    }
  }

  #listen(): void {
    const lines = createInterface({ input: this.#process.stdout });
    lines.on("line", (line) => this.#handleLine(line));
    this.#process.stderr.on("data", (chunk: Buffer | string) => {
      this.#stderr = `${this.#stderr}${chunk.toString()}`.slice(-16_384);
    });
    this.#process.once("error", (error: Error) => {
      this.#fail(
        new NeedleRuntimeError("service_unavailable", "Needle process failed.", error),
      );
    });
    this.#process.once(
      "exit",
      (code: number | null, signal: NodeJS.Signals | null) => {
        this.#exited = true;
        this.#resolveExit();
        this.#onClose();
        if (this.#closing && code === 0) {
          this.#ready = false;
          return;
        }
        const detail = this.#stderr.trim();
        this.#fail(
          new NeedleRuntimeError(
            "service_unavailable",
            `Needle stopped${code === null ? ` with ${signal ?? "a signal"}` : ` with code ${code}`}${detail ? `: ${detail}` : "."}`,
          ),
        );
      },
    );
  }

  #handleLine(line: string): void {
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch (error) {
      this.#fail(
        new NeedleRuntimeError(
          "service_unavailable",
          "Needle returned invalid process data.",
          error,
        ),
      );
      return;
    }
    if (!isRecord(message)) return;

    if (message.type === "ready") {
      if (message.needleVersion !== NEEDLE_VERSION) {
        this.#fail(
          new NeedleRuntimeError(
            "service_unavailable",
            `Needle ${NEEDLE_VERSION} is required; worker reported ${String(message.needleVersion)}.`,
          ),
        );
        return;
      }
      this.#ready = true;
      this.#resolveReady();
      return;
    }
    if (message.type === "error" && !this.#ready) {
      this.#fail(
        new NeedleRuntimeError(
          "service_unavailable",
          `Needle could not start: ${String(message.message ?? "unknown error")}`,
        ),
      );
      return;
    }
    if (typeof message.id !== "number") return;
    const pending = this.#pending.get(message.id);
    if (!pending) return;
    this.#pending.delete(message.id);

    if (message.ok !== true) {
      const workerError = isRecord(message.error)
        ? String(message.error.message ?? "Needle classification failed.")
        : "Needle classification failed.";
      pending.reject(
        new NeedleRuntimeError("classification_failed", workerError),
      );
      return;
    }
    try {
      pending.resolve(
        decodeNeedleResult(
          message.result,
          this.#resultConfig,
          this.#numberLabels,
        ),
      );
    } catch (error) {
      pending.reject(
        new NeedleRuntimeError(
          "classification_failed",
          "Needle returned an invalid classification.",
          error,
        ),
      );
    }
  }

  #fail(error: NeedleRuntimeError): void {
    const wasReady = this.#ready;
    this.#ready = false;
    if (!wasReady) this.#rejectReady(error);
    for (const pending of this.#pending.values()) pending.reject(error);
    this.#pending.clear();
    if (wasReady && !this.#closing) this.#onBackgroundError(error);
    if (!this.#exited && !this.#closing) this.#process.kill("SIGTERM");
  }
}

const spawnProcess: NeedleSpawn = (command, args, options) =>
  spawn(command, [...args], {
    cwd: options.cwd,
    env: options.env,
    stdio: ["pipe", "pipe", "pipe"],
  });

const runCommand: RunCommand = (command, args, options) =>
  new Promise((resolve, reject) => {
    const child = spawn(command, [...args], {
      cwd: options?.cwd,
      env: options?.env,
      stdio: [options?.stdin === undefined ? "ignore" : "pipe", "pipe", "pipe"],
    });
    if (options?.stdin !== undefined && child.stdin !== null) {
      child.stdin.on("error", () => undefined);
      child.stdin.end(options.stdin, "utf8");
    }
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    let forceKill: ReturnType<typeof setTimeout> | undefined;
    let rejectAfterKill: ReturnType<typeof setTimeout> | undefined;
    const timeout = options?.timeoutMs === undefined
      ? undefined
      : setTimeout(() => {
          timedOut = true;
          child.kill("SIGTERM");
          forceKill = setTimeout(() => {
            child.kill("SIGKILL");
            rejectAfterKill = setTimeout(() => {
              if (settled) return;
              settled = true;
              clearTimers();
              reject(new Error(`${command} exceeded its time limit.`));
            }, 2_000);
            rejectAfterKill.unref();
          }, 2_000);
          forceKill.unref();
        }, options.timeoutMs);
    timeout?.unref();
    const clearTimers = () => {
      if (timeout !== undefined) clearTimeout(timeout);
      if (forceKill !== undefined) clearTimeout(forceKill);
      if (rejectAfterKill !== undefined) clearTimeout(rejectAfterKill);
    };
    child.stdout!.on("data", (chunk: Buffer | string) => {
      stdout = `${stdout}${chunk.toString()}`.slice(-262_144);
    });
    child.stderr!.on("data", (chunk: Buffer | string) => {
      stderr = `${stderr}${chunk.toString()}`.slice(-262_144);
    });
    child.once("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimers();
      reject(error);
    });
    child.once("exit", (code, signal) => {
      if (settled) return;
      settled = true;
      clearTimers();
      if (timedOut) {
        reject(new Error(`${command} exceeded its time limit.`));
        return;
      }
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(
        new Error(
          `${command} failed${code === null ? ` with ${signal ?? "a signal"}` : ` with code ${code}`}${stderr.trim() ? `: ${stderr.trim()}` : "."}`,
        ),
      );
    });
  });

function classifierKey(name: string): string {
  const readable = name
    .normalize("NFKD")
    .replace(/[^a-zA-Z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40) || "classifier";
  const hash = createHash("sha256").update(name).digest("hex").slice(0, 12);
  return `${readable}-${hash}`;
}

function pythonIn(environmentDirectory: string): string {
  return process.platform === "win32"
    ? join(environmentDirectory, "Scripts", "python.exe")
    : join(environmentDirectory, "bin", "python");
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function readJson(path: string): Promise<Record<string, unknown> | undefined> {
  try {
    const value: unknown = JSON.parse(await readFile(path, "utf8"));
    return isRecord(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

function uvAsset(): string {
  const key = `${process.platform}-${process.arch}`;
  const assets: Record<string, string> = {
    "darwin-arm64": "uv-aarch64-apple-darwin.tar.gz",
    "darwin-x64": "uv-x86_64-apple-darwin.tar.gz",
    "linux-arm64": "uv-aarch64-unknown-linux-gnu.tar.gz",
    "linux-x64": "uv-x86_64-unknown-linux-gnu.tar.gz",
    "win32-arm64": "uv-aarch64-pc-windows-msvc.zip",
    "win32-x64": "uv-x86_64-pc-windows-msvc.zip",
  };
  const asset = assets[key];
  if (!asset) throw new Error(`SwapAI does not support uv on ${key}.`);
  return asset;
}

function checksumFor(asset: string, sums: string): string {
  for (const line of sums.split("\n")) {
    const [hash, file] = line.trim().split(/\s+\*?/);
    if (file === asset && /^[a-f0-9]{64}$/i.test(hash ?? "")) return hash!;
  }
  throw new Error(`The uv checksum list did not contain ${asset}.`);
}

async function findFile(directory: string, name: string): Promise<string | undefined> {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isFile() && entry.name === name) return path;
    if (entry.isDirectory()) {
      const nested = await findFile(path, name);
      if (nested) return nested;
    }
  }
  return undefined;
}

function withTimeout<T>(
  promise: Promise<T>,
  milliseconds: number,
  message: string,
  onTimeout?: () => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      onTimeout?.();
      reject(new NeedleRuntimeError("service_unavailable", message));
    }, milliseconds);
    timer.unref();
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function toRuntimeError(message: string, cause: unknown): NeedleRuntimeError {
  return cause instanceof NeedleRuntimeError
    ? cause
    : new NeedleRuntimeError("service_unavailable", message, cause);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
