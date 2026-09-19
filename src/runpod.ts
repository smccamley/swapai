import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { Effect } from "effect";

import { writeTrainingBundle } from "./training-bundle.js";
import type {
  TrainingCandidate,
  TrainingJob,
  TrainingProvider,
} from "./types.js";

const RUNPOD_API = "https://rest.runpod.io/v1";
const DEFAULT_IMAGE = "ghcr.io/smccamley/swapai-trainer:0.4.2";
const DEFAULT_GPU_TYPES = [
  "NVIDIA A40",
  "NVIDIA RTX A5000",
  "NVIDIA RTX A6000",
] as const;

export interface RunpodTrainerOptions {
  readonly apiKey: string;
  readonly maximumCostUsd: number;
  readonly maximumRuntimeMinutes?: number;
  readonly image?: string;
  readonly gpuTypes?: readonly string[];
  readonly cloudType?: "SECURE" | "COMMUNITY";
  readonly sshPrivateKey?: string;
}

export interface RunpodTrainerDependencies {
  readonly fetch?: typeof globalThis.fetch;
  readonly runCommand?: (
    command: string,
    args: readonly string[],
  ) => Promise<{ readonly stdout: string; readonly stderr: string }>;
  readonly sleep?: (milliseconds: number) => Promise<void>;
  readonly now?: () => number;
}

interface RunpodPod {
  readonly id: string;
  readonly name: string;
  readonly desiredStatus?: string;
  readonly publicIp?: string;
  readonly portMappings?: Readonly<Record<string, number>>;
  readonly costPerHr?: number;
  readonly adjustedCostPerHr?: number;
}

export const runpodTrainer = (
  options: RunpodTrainerOptions,
  dependencies: RunpodTrainerDependencies = {},
): TrainingProvider => {
  validateOptions(options);
  const runtimeMinutes = options.maximumRuntimeMinutes ?? 30;
  const fetchImplementation = dependencies.fetch ?? globalThis.fetch;
  const runCommand = dependencies.runCommand ?? runProcess;
  const sleep = dependencies.sleep ?? ((milliseconds) =>
    new Promise<void>((resolve) => setTimeout(resolve, milliseconds)));
  const now = dependencies.now ?? Date.now;
  const api = createRunpodApi(options.apiKey, fetchImplementation);

  return {
    name: "runpod",
    train: (job) => Effect.runPromise(Effect.acquireUseRelease(
      Effect.tryPromise({
        try: async () => {
          await removeExpiredManagedPods(api, now(), sleep);
          const privateKey = resolvePrivateKey(options.sshPrivateKey);
          const publicKey = (await runCommand(
            "ssh-keygen",
            ["-y", "-f", privateKey],
          )).stdout.trim();
          if (publicKey === "") throw new Error("SSH public key is empty");
          const deadline = now() + runtimeMinutes * 60_000;
          const pod = await api.createPod({
            name: `swapai-${deadline}-${job.id}`,
            cloudType: options.cloudType ?? "SECURE",
            computeType: "GPU",
            imageName: options.image ?? DEFAULT_IMAGE,
            gpuTypeIds: options.gpuTypes ?? DEFAULT_GPU_TYPES,
            gpuTypePriority: "availability",
            gpuCount: 1,
            containerDiskInGb: 30,
            volumeInGb: 0,
            minVCPUPerGPU: 4,
            minRAMPerGPU: 16,
            ports: ["22/tcp"],
            supportPublicIp: true,
            interruptible: false,
            env: { SSH_PUBLIC_KEY: publicKey, PUBLIC_KEY: publicKey },
          });
          return { pod, privateKey, startedAt: now() };
        },
        catch: asError,
      }),
      ({ pod, privateKey, startedAt }) => Effect.tryPromise({
        try: () => withTimeout(
          usePod({
            api,
            job,
            pod,
            privateKey,
            maximumCostUsd: options.maximumCostUsd,
            maximumRuntimeMinutes: runtimeMinutes,
            runCommand,
            sleep,
            now,
            startedAt,
          }),
          runtimeMinutes * 60_000,
          `Runpod training exceeded ${runtimeMinutes} minutes`,
        ),
        catch: asError,
      }),
      ({ pod }) => Effect.tryPromise({
        try: () => deletePodVerified(api, pod.id, sleep),
        catch: asError,
      }).pipe(Effect.orDie),
    )),
  };
};

const usePod = async (options: {
  readonly api: ReturnType<typeof createRunpodApi>;
  readonly job: TrainingJob;
  readonly pod: RunpodPod;
  readonly privateKey: string;
  readonly maximumCostUsd: number;
  readonly maximumRuntimeMinutes: number;
  readonly runCommand: NonNullable<RunpodTrainerDependencies["runCommand"]>;
  readonly sleep: NonNullable<RunpodTrainerDependencies["sleep"]>;
  readonly now: () => number;
  readonly startedAt: number;
}): Promise<TrainingCandidate> => {
  const hourlyCost = options.pod.adjustedCostPerHr ?? options.pod.costPerHr;
  if (hourlyCost === undefined || !Number.isFinite(hourlyCost)) {
    throw new Error("Runpod did not report the Pod hourly cost");
  }
  const maximumRunCost = hourlyCost * options.maximumRuntimeMinutes / 60;
  if (maximumRunCost > options.maximumCostUsd) {
    throw new Error(
      `Runpod Pod could cost $${maximumRunCost.toFixed(2)}; limit is $${options.maximumCostUsd.toFixed(2)}`,
    );
  }

  const connected = await waitForConnectablePod(
    options.api,
    options.pod.id,
    options.sleep,
  );
  const ssh = sshArguments(
    connected.publicIp!,
    connected.portMappings?.["22"]!,
    options.privateKey,
    join(options.job.outputDirectory, "runpod-known-hosts"),
  );
  await waitForSsh(ssh, options.runCommand, options.sleep);
  const bundleDirectory = await writeTrainingBundle(options.job);
  const remoteDirectory = `/workspace/swapai-${options.job.id}`;
  await options.runCommand("scp", [
    "-r",
    ...scpConnectionArguments(ssh),
    bundleDirectory,
    `${ssh.at(-1)!}:${remoteDirectory}`,
  ]);
  await options.runCommand("ssh", [
    ...ssh,
    `python /opt/swapai/train.py --job-directory ${remoteDirectory}`,
  ]);
  const modelPath = join(options.job.outputDirectory, "model.cact");
  await options.runCommand("scp", [
    ...scpConnectionArguments(ssh),
    `${ssh.at(-1)!}:${remoteDirectory}/model.cact`,
    modelPath,
  ]);
  if (options.job.result.type === "number") {
    await options.runCommand("scp", [
      ...scpConnectionArguments(ssh),
      `${ssh.at(-1)!}:${remoteDirectory}/model.cact.numbers.json`,
      `${modelPath}.numbers.json`,
    ]);
  }
  return {
    modelPath,
    needleVersion: "2.0.14",
    providerRunId: options.pod.id,
    costUsd: hourlyCost * Math.max(0, options.now() - options.startedAt) / 3_600_000,
  };
};

const createRunpodApi = (
  apiKey: string,
  fetchImplementation: typeof globalThis.fetch,
) => {
  const request = async <Value>(
    path: string,
    init: RequestInit = {},
  ): Promise<Value> => {
    const response = await fetchImplementation(`${RUNPOD_API}${path}`, {
      ...init,
      headers: {
        authorization: `Bearer ${apiKey}`,
        ...(init.body === undefined ? {} : { "content-type": "application/json" }),
        ...init.headers,
      },
    });
    const text = await response.text();
    if (!response.ok) {
      const error = new Error(
        `${init.method ?? "GET"} Runpod ${path} returned ${response.status}${text === "" ? "" : `: ${text}`}`,
      ) as Error & { status?: number };
      error.status = response.status;
      throw error;
    }
    return (text === "" ? null : JSON.parse(text)) as Value;
  };
  return {
    createPod: (body: unknown) => request<RunpodPod>("/pods", {
      method: "POST",
      body: JSON.stringify(body),
    }),
    listPods: async (): Promise<readonly RunpodPod[]> => {
      const response = await request<readonly RunpodPod[] | { pods?: readonly RunpodPod[] }>("/pods");
      return Array.isArray(response)
        ? response
        : "pods" in response
          ? response.pods ?? []
          : [];
    },
    getPod: (id: string) => request<RunpodPod>(`/pods/${id}`),
    deletePod: (id: string) => request<unknown>(`/pods/${id}`, { method: "DELETE" }),
  };
};

const removeExpiredManagedPods = async (
  api: ReturnType<typeof createRunpodApi>,
  now: number,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<void> => {
  const pods = await api.listPods();
  for (const pod of pods) {
    const match = /^swapai-(\d+)-/.exec(pod.name);
    if (match !== null && Number(match[1]) <= now && pod.desiredStatus !== "TERMINATED") {
      await deletePodVerified(api, pod.id, sleep);
    }
  }
};

const waitForConnectablePod = async (
  api: ReturnType<typeof createRunpodApi>,
  id: string,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<RunpodPod> => {
  for (let attempt = 0; attempt < 144; attempt += 1) {
    const pod = await api.getPod(id);
    if (
      pod.desiredStatus === "RUNNING" &&
      pod.publicIp !== undefined &&
      pod.portMappings?.["22"] !== undefined
    ) return pod;
    await sleep(5_000);
  }
  throw new Error(`Runpod Pod ${id} did not expose SSH within 12 minutes`);
};

const waitForSsh = async (
  ssh: readonly string[],
  runCommand: NonNullable<RunpodTrainerDependencies["runCommand"]>,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<void> => {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try {
      await runCommand("ssh", [...ssh, "true"]);
      return;
    } catch {
      await sleep(5_000);
    }
  }
  throw new Error("Runpod Pod SSH did not become ready within 10 minutes");
};

const deletePodVerified = async (
  api: ReturnType<typeof createRunpodApi>,
  id: string,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<void> => {
  try {
    await api.deletePod(id);
  } catch (error) {
    if (!hasStatus(error, 404)) throw error;
  }
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const pod = await api.getPod(id);
      if (pod.desiredStatus === "TERMINATED") return;
    } catch (error) {
      if (hasStatus(error, 404)) return;
      throw error;
    }
    await sleep(3_000);
  }
  throw new Error(`Runpod Pod ${id} still exists after deletion`);
};

const sshArguments = (
  host: string,
  port: number,
  privateKey: string,
  knownHostsPath: string,
): readonly string[] => [
  "-p",
  String(port),
  "-i",
  privateKey,
  "-o",
  "BatchMode=yes",
  "-o",
  "ConnectTimeout=15",
  "-o",
  "StrictHostKeyChecking=accept-new",
  "-o",
  `UserKnownHostsFile=${knownHostsPath}`,
  `root@${host}`,
];

const scpConnectionArguments = (ssh: readonly string[]): readonly string[] => {
  const args = ssh.slice(0, -1);
  if (args[0] === "-p") args[0] = "-P";
  return args;
};

const resolvePrivateKey = (configured: string | undefined): string => {
  const candidates = configured === undefined
    ? [join(homedir(), ".ssh", "id_ed25519"), join(homedir(), ".ssh", "id_rsa")]
    : [configured];
  const found = candidates.find(existsSync);
  if (found === undefined) {
    throw new Error(
      "Runpod training needs an SSH private key; set sshPrivateKey or create ~/.ssh/id_ed25519",
    );
  }
  return found;
};

const runProcess = (
  command: string,
  args: readonly string[],
): Promise<{ readonly stdout: string; readonly stderr: string }> =>
  new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += String(chunk); });
    child.stderr.on("data", (chunk) => { stderr += String(chunk); });
    child.once("error", reject);
    child.once("close", (code, signal) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${command} exited with ${signal ?? code}${stderr === "" ? "" : `: ${stderr}`}`));
    });
  });

const withTimeout = async <Value>(
  promise: Promise<Value>,
  milliseconds: number,
  message: string,
): Promise<Value> => {
  let timeout: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error(message)), milliseconds);
        timeout.unref();
      }),
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
};

const validateOptions = (options: RunpodTrainerOptions): void => {
  if (options.apiKey.trim() === "") throw new TypeError("Runpod apiKey must not be empty");
  if (!Number.isFinite(options.maximumCostUsd) || options.maximumCostUsd <= 0) {
    throw new TypeError("Runpod maximumCostUsd must be positive");
  }
  if (
    options.maximumRuntimeMinutes !== undefined &&
    (!Number.isFinite(options.maximumRuntimeMinutes) || options.maximumRuntimeMinutes < 1)
  ) throw new TypeError("Runpod maximumRuntimeMinutes must be at least 1");
  if (
    options.maximumRuntimeMinutes !== undefined &&
    options.maximumRuntimeMinutes > 360
  ) throw new TypeError("Runpod maximumRuntimeMinutes must not exceed 360");
  if (options.gpuTypes !== undefined && options.gpuTypes.length === 0) {
    throw new TypeError("Runpod gpuTypes must not be empty");
  }
};

const hasStatus = (error: unknown, status: number): boolean =>
  typeof error === "object" && error !== null && "status" in error && error.status === status;

const asError = (error: unknown): Error =>
  error instanceof Error ? error : new Error(String(error));
