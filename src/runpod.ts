import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { needleModelVersion } from "./runtime.js";
import {
  verifyTrainingResult,
  writeTrainingBundle,
} from "./training-bundle.js";
import type {
  TrainingCandidate,
  TrainingJob,
  TrainingLifecycleReporter,
  TrainingProvider,
} from "./types.js";

const RUNPOD_API = "https://api.runpod.io/v2";
const DEFAULT_IMAGE = "ghcr.io/smccamley/swapai-trainer:0.6.5";
const CLEANUP_HEADROOM_MS = 5 * 60_000;
const CONTAINER_DISK_GB = 30;
const CONTAINER_DISK_USD_PER_GB_MONTH = 0.1;
const CONSERVATIVE_MONTH_HOURS = 28 * 24;
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
  readonly cloud?: "SECURE";
  /** @deprecated Use cloud. */
  readonly cloudType?: "SECURE";
  readonly sshPrivateKey?: string;
  /** Register this trainer's public key before Pod creation and remove it after cleanup. */
  readonly registerSshPublicKeyForTraining?: boolean;
}

export interface RunpodTrainerDependencies {
  readonly fetch?: typeof globalThis.fetch;
  readonly runCommand?: (
    command: string,
    args: readonly string[],
    execution?: { readonly signal?: AbortSignal },
  ) => Promise<{ readonly stdout: string; readonly stderr: string }>;
  readonly sleep?: (milliseconds: number) => Promise<void>;
  readonly now?: () => number;
  readonly requestTimeoutMilliseconds?: number;
}

interface RunpodPod {
  readonly id: string;
  readonly name: string;
  readonly status?: string;
  readonly cost?: number;
  readonly ssh?: {
    readonly proxy?: {
      readonly host: string;
      readonly port: number;
      readonly username: string;
    } | null;
    readonly direct?: {
      readonly host: string;
      readonly port: number;
      readonly username: string;
    } | null;
  } | null;
}

interface RunpodPodListResponse {
  readonly pods?: readonly RunpodPod[];
  readonly pagination?: {
    readonly nextCursor: string | null;
    readonly hasNextPage: boolean;
  };
}

interface RunpodSshKeysResponse {
  readonly keys: readonly string[];
}

export const runpodTrainer = (
  options: RunpodTrainerOptions,
  dependencies: RunpodTrainerDependencies = {},
): TrainingProvider => {
  validateOptions(options);
  const runtimeMinutes = options.maximumRuntimeMinutes ?? 30;
  const fetchImplementation = dependencies.fetch ?? globalThis.fetch;
  const runCommand = dependencies.runCommand ?? runProcess;
  const sleep =
    dependencies.sleep ??
    ((milliseconds) =>
      new Promise<void>((resolve) => setTimeout(resolve, milliseconds)));
  const now = dependencies.now ?? Date.now;
  const api = createRunpodApi(
    options.apiKey,
    fetchImplementation,
    dependencies.requestTimeoutMilliseconds ?? 30_000,
  );

  return {
    name: "runpod",
    train: async (job, lifecycle) => {
      await removeExpiredManagedPods(api, now(), sleep);
      const privateKey = resolvePrivateKey(options.sshPrivateKey);
      const publicKey = (
        await withAbortTimeout(
          (signal) =>
            runCommand("ssh-keygen", ["-y", "-f", privateKey], { signal }),
          30_000,
          "SSH public-key derivation timed out",
        )
      ).stdout.trim();
      if (publicKey === "") throw new Error("SSH public key is empty");
      const removeSshPublicKey =
        options.registerSshPublicKeyForTraining === true
          ? await registerSshPublicKey(api, publicKey)
          : undefined;
      const billingStartedAt = now();
      const deadline = billingStartedAt + runtimeMinutes * 60_000;
      let pod: RunpodPod;
      try {
        pod = await createPodForFirstAvailableGpu(
          api,
          options.gpuTypes ?? DEFAULT_GPU_TYPES,
          {
            name: `swapai-${deadline}-${job.id}`,
            cloud: options.cloud ?? options.cloudType ?? "SECURE",
            image: options.image ?? DEFAULT_IMAGE,
            disk: CONTAINER_DISK_GB,
            ports: ["22/tcp"],
            startSsh: true,
            env: { SSH_PUBLIC_KEY: publicKey, PUBLIC_KEY: publicKey },
          },
          sleep,
        );
      } catch (podCreationError) {
        if (removeSshPublicKey === undefined) throw podCreationError;
        try {
          await removeSshPublicKey();
        } catch (sshKeyCleanupError) {
          throw new AggregateError(
            [podCreationError, sshKeyCleanupError],
            "Runpod Pod creation and temporary SSH key cleanup both failed",
          );
        }
        throw podCreationError;
      }
      let result: TrainingCandidate | undefined;
      let trainingError: unknown;
      try {
        lifecycle?.recordProviderRun({
          providerRunId: pod.id,
          resources: [{ type: "runpod-pod", id: pod.id }],
        });
        const trainingWindowMs = paidTrainingWindowMilliseconds({
          hourlyCost: totalHourlyPodCost(pod.cost),
          maximumCostUsd: options.maximumCostUsd,
          maximumRuntimeMinutes: runtimeMinutes,
          billingStartedAt,
          currentTime: now(),
        });
        if (trainingWindowMs <= 0) {
          throw new Error(
            "Runpod Pod leaves no positive training window after cleanup headroom",
          );
        }
        result = await withAbortTimeout(
          (signal) =>
            usePod({
              api,
              job,
              pod,
              privateKey,
              runCommand,
              sleep,
              signal,
            }),
          trainingWindowMs,
          "Runpod paid training window expired",
        );
      } catch (error) {
        trainingError = error;
      }
      let cleanupError: unknown;
      try {
        await deletePodAndReport(
          api,
          pod.id,
          sleep,
          lifecycle,
          removeSshPublicKey,
        );
      } catch (error) {
        cleanupError = error;
      }
      if (trainingError !== undefined && cleanupError !== undefined) {
        throw new AggregateError(
          [trainingError, cleanupError],
          "Runpod training and Pod cleanup both failed",
        );
      }
      if (trainingError !== undefined) throw trainingError;
      if (cleanupError !== undefined) throw cleanupError;
      return {
        ...result!,
        costUsd:
          (totalHourlyPodCost(pod.cost) *
            Math.max(0, now() - billingStartedAt)) /
          3_600_000,
      };
    },
    cancel: async (run, lifecycle) => {
      if (run.providerRunId === null) {
        throw new Error(`Training run ${run.id} has no Runpod Pod ID`);
      }
      await deletePodAndReport(api, run.providerRunId, sleep, lifecycle);
    },
    reconcile: async (run, lifecycle) => {
      let providerRunId = run.providerRunId;
      if (providerRunId === null) {
        const managedPods = (await api.listPods()).filter(
          (pod) =>
            pod.name.endsWith(`-${run.id}`) && pod.name.startsWith("swapai-"),
        );
        if (managedPods.length === 0) {
          if (now() < run.startedAt + runtimeMinutes * 60_000) {
            return { status: "running" };
          }
          lifecycle?.recordCleanup({ status: "succeeded" });
          return {
            status: "failed",
            failureMessage:
              "Runpod Pod ID was never recorded and no managed Pod exists",
          };
        }
        if (managedPods.length > 1) {
          const failures: unknown[] = [];
          try {
            lifecycle?.recordProviderRun({
              providerRunId: managedPods[0]!.id,
              resources: managedPods.map((pod) => ({
                type: "runpod-pod",
                id: pod.id,
              })),
            });
          } catch (error) {
            failures.push(error);
          }
          for (const pod of managedPods) {
            try {
              await deletePodAndReport(api, pod.id, sleep, lifecycle);
            } catch (error) {
              failures.push(error);
            }
          }
          if (failures.length === 1) throw failures[0];
          if (failures.length > 1) {
            throw new AggregateError(
              failures,
              `Could not safely clean ambiguous Runpod Pods for ${run.id}`,
            );
          }
          return {
            status: "failed",
            failureMessage:
              "Multiple managed Runpod Pods matched this training run",
          };
        }
        const managedPod = managedPods[0]!;
        providerRunId = managedPod.id;
        try {
          lifecycle?.recordProviderRun({
            providerRunId,
            resources: [{ type: "runpod-pod", id: providerRunId }],
          });
        } catch (error) {
          try {
            await deletePodAndReport(api, providerRunId, sleep, lifecycle);
          } catch (cleanupError) {
            throw new AggregateError(
              [error, cleanupError],
              `Could not persist or clean recovered Runpod Pod ${providerRunId}`,
            );
          }
          throw error;
        }
      }
      if (run.status !== "running") {
        await deletePodAndReport(api, providerRunId, sleep, lifecycle);
        return {
          status: "failed",
          failureMessage: "Cleaned provider resource for terminal training run",
        };
      }
      if (now() >= run.startedAt + runtimeMinutes * 60_000) {
        await deletePodAndReport(api, providerRunId, sleep, lifecycle);
        return {
          status: "failed",
          failureMessage: `Runpod training exceeded ${runtimeMinutes} minutes`,
        };
      }
      try {
        const pod = await api.getPod(providerRunId);
        if (pod.status === "TERMINATED") {
          lifecycle?.recordCleanup({ status: "succeeded" });
          return {
            status: "failed",
            failureMessage: `Runpod Pod ended with status ${pod.status}`,
          };
        }
        if (pod.status === "EXITED" || pod.status === "ERROR") {
          await deletePodAndReport(api, providerRunId, sleep, lifecycle);
          return {
            status: "failed",
            failureMessage: `Runpod Pod ended with status ${pod.status}`,
          };
        }
        let paidWindowMs: number;
        try {
          paidWindowMs = paidTrainingWindowMilliseconds({
            hourlyCost: totalHourlyPodCost(pod.cost),
            maximumCostUsd: options.maximumCostUsd,
            maximumRuntimeMinutes: runtimeMinutes,
            billingStartedAt: run.startedAt,
            currentTime: now(),
          });
        } catch (error) {
          await deletePodAndReport(api, providerRunId, sleep, lifecycle);
          return {
            status: "failed",
            failureMessage:
              error instanceof Error ? error.message : String(error),
          };
        }
        if (paidWindowMs <= 0) {
          await deletePodAndReport(api, providerRunId, sleep, lifecycle);
          return {
            status: "failed",
            failureMessage: "Runpod paid training window expired",
          };
        }
        return { status: "running" };
      } catch (error) {
        if (hasStatus(error, 404)) {
          lifecycle?.recordCleanup({ status: "succeeded" });
          return {
            status: "failed",
            failureMessage: "Runpod Pod no longer exists",
          };
        }
        throw error;
      }
    },
  };
};

const usePod = async (options: {
  readonly api: ReturnType<typeof createRunpodApi>;
  readonly job: TrainingJob;
  readonly pod: RunpodPod;
  readonly privateKey: string;
  readonly runCommand: NonNullable<RunpodTrainerDependencies["runCommand"]>;
  readonly sleep: NonNullable<RunpodTrainerDependencies["sleep"]>;
  readonly signal: AbortSignal;
}): Promise<TrainingCandidate> => {
  const hourlyCost = options.pod.cost;
  if (hourlyCost === undefined || !Number.isFinite(hourlyCost)) {
    throw new Error("Runpod did not report the Pod hourly cost");
  }

  const connected = await waitForConnectablePod(
    options.api,
    options.pod.id,
    options.sleep,
    options.signal,
  );
  const connections = [connected.ssh?.direct, connected.ssh?.proxy].filter(
    (connection): connection is NonNullable<typeof connection> =>
      connection !== undefined && connection !== null,
  );
  if (connections.length === 0) {
    throw new Error(`Runpod Pod ${options.pod.id} did not provide SSH details`);
  }
  const ssh = await waitForSsh(
    connections.map((connection) =>
      sshArguments(
        connection.host,
        connection.port,
        options.privateKey,
        join(options.job.outputDirectory, "runpod-known-hosts"),
        connection.username,
      ),
    ),
    options.runCommand,
    options.sleep,
    options.signal,
  );
  const bundleDirectory = await writeTrainingBundle(options.job);
  const remoteDirectory = `/workspace/swapai-${options.job.id}`;
  await options.runCommand(
    "scp",
    [
      "-r",
      ...scpConnectionArguments(ssh),
      bundleDirectory,
      `${ssh.at(-1)!}:${remoteDirectory}`,
    ],
    { signal: options.signal },
  );
  await options.runCommand(
    "ssh",
    [...ssh, `python /opt/swapai/train.py --job-directory ${remoteDirectory}`],
    { signal: options.signal },
  );
  await options.runCommand(
    "scp",
    [
      ...scpConnectionArguments(ssh),
      `${ssh.at(-1)!}:${remoteDirectory}/result.json`,
      join(bundleDirectory, "result.json"),
    ],
    { signal: options.signal },
  );
  const modelPath = join(bundleDirectory, "model.cact");
  await options.runCommand(
    "scp",
    [
      ...scpConnectionArguments(ssh),
      `${ssh.at(-1)!}:${remoteDirectory}/model.cact`,
      modelPath,
    ],
    { signal: options.signal },
  );
  if (options.job.result.type === "number") {
    await options.runCommand(
      "scp",
      [
        ...scpConnectionArguments(ssh),
        `${ssh.at(-1)!}:${remoteDirectory}/model.cact.numbers.json`,
        `${modelPath}.numbers.json`,
      ],
      { signal: options.signal },
    );
  }
  const verified = await verifyTrainingResult(bundleDirectory);
  if (verified.needleVersion !== needleModelVersion(options.job.result)) {
    throw new Error(
      `Trainer returned Needle ${verified.needleVersion}; expected ${needleModelVersion(options.job.result)}`,
    );
  }
  return {
    ...verified,
    providerRunId: options.pod.id,
  };
};

const paidTrainingWindowMilliseconds = (options: {
  readonly hourlyCost: number | undefined;
  readonly maximumCostUsd: number;
  readonly maximumRuntimeMinutes: number;
  readonly billingStartedAt: number;
  readonly currentTime: number;
}): number => {
  if (
    options.hourlyCost === undefined ||
    !Number.isFinite(options.hourlyCost)
  ) {
    throw new Error("Runpod did not report the Pod hourly cost");
  }
  if (options.hourlyCost <= 0) {
    throw new Error("Runpod reported a non-positive Pod hourly cost");
  }
  const runtimeDeadline =
    options.billingStartedAt + options.maximumRuntimeMinutes * 60_000;
  const costDeadline =
    options.billingStartedAt +
    (options.maximumCostUsd / options.hourlyCost) * 3_600_000;
  return Math.floor(
    Math.min(runtimeDeadline, costDeadline) -
      CLEANUP_HEADROOM_MS -
      options.currentTime,
  );
};

const totalHourlyPodCost = (reportedHourlyCost: number | undefined): number => {
  if (
    reportedHourlyCost === undefined ||
    !Number.isFinite(reportedHourlyCost)
  ) {
    throw new Error("Runpod did not report the Pod hourly cost");
  }
  return (
    reportedHourlyCost +
    (CONTAINER_DISK_GB * CONTAINER_DISK_USD_PER_GB_MONTH) /
      CONSERVATIVE_MONTH_HOURS
  );
};

const createRunpodApi = (
  apiKey: string,
  fetchImplementation: typeof globalThis.fetch,
  requestTimeoutMilliseconds: number,
) => {
  const request = async <Value>(
    path: string,
    init: RequestInit = {},
    externalSignal?: AbortSignal,
  ): Promise<Value> => {
    const requestSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
    const response = await fetchImplementation(`${RUNPOD_API}${path}`, {
      ...init,
      signal:
        externalSignal === undefined
          ? requestSignal
          : AbortSignal.any([externalSignal, requestSignal]),
      headers: {
        authorization: `Bearer ${apiKey}`,
        ...(init.body === undefined
          ? {}
          : { "content-type": "application/json" }),
        ...init.headers,
      },
    });
    const text = await response.text();
    if (!response.ok) {
      const error = new Error(
        `${init.method ?? "GET"} Runpod ${path} returned ${response.status}${text === "" ? "" : `: ${text}`}`,
      ) as Error & { status?: number; retryAfterMilliseconds?: number };
      error.status = response.status;
      const retryAfter = response.headers.get("retry-after");
      if (retryAfter !== null && /^\d+$/.test(retryAfter)) {
        error.retryAfterMilliseconds = Number(retryAfter) * 1_000;
      }
      throw error;
    }
    return (text === "" ? null : JSON.parse(text)) as Value;
  };
  return {
    createPod: (body: unknown) =>
      request<RunpodPod>("/pods", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    listPods: async (): Promise<readonly RunpodPod[]> => {
      const pods: RunpodPod[] = [];
      const seenCursors = new Set<string>();
      let cursor: string | null = null;
      do {
        const response: RunpodPodListResponse =
          await request<RunpodPodListResponse>(
            cursor === null
              ? "/pods"
              : `/pods?cursor=${encodeURIComponent(cursor)}`,
          );
        pods.push(...(response.pods ?? []));
        if (response.pagination?.hasNextPage !== true) break;
        const nextCursor: string | null = response.pagination.nextCursor;
        if (nextCursor === null || seenCursors.has(nextCursor)) {
          throw new Error("Runpod Pod pagination returned an invalid cursor");
        }
        seenCursors.add(nextCursor);
        cursor = nextCursor;
      } while (true);
      return pods;
    },
    getPod: (id: string, signal?: AbortSignal) =>
      request<RunpodPod>(`/pods/${id}`, {}, signal),
    deletePod: (id: string, signal?: AbortSignal) =>
      request<unknown>(`/pods/${id}`, { method: "DELETE" }, signal),
    listSshPublicKeys: () =>
      request<RunpodSshKeysResponse>("/account/ssh-keys"),
    replaceSshPublicKeys: (keys: readonly string[]) =>
      request<RunpodSshKeysResponse>("/account/ssh-keys", {
        method: "PUT",
        body: JSON.stringify({ keys }),
      }),
  };
};

const sshPublicKeyIdentity = (publicKey: string): string =>
  publicKey.trim().split(/\s+/, 2).join(" ");

const registerSshPublicKey = async (
  api: ReturnType<typeof createRunpodApi>,
  publicKey: string,
): Promise<(() => Promise<void>) | undefined> => {
  const identity = sshPublicKeyIdentity(publicKey);
  const current = await api.listSshPublicKeys();
  if (current.keys.some((key) => sshPublicKeyIdentity(key) === identity)) {
    return undefined;
  }
  await api.replaceSshPublicKeys([...current.keys, publicKey]);
  const registered = await api.listSshPublicKeys();
  if (!registered.keys.some((key) => sshPublicKeyIdentity(key) === identity)) {
    throw new Error("Runpod did not retain the temporary SSH public key");
  }
  return async () => {
    const keysBeforeCleanup = await api.listSshPublicKeys();
    if (
      !keysBeforeCleanup.keys.some(
        (key) => sshPublicKeyIdentity(key) === identity,
      )
    ) {
      return;
    }
    await api.replaceSshPublicKeys(
      keysBeforeCleanup.keys.filter(
        (key) => sshPublicKeyIdentity(key) !== identity,
      ),
    );
    const keysAfterCleanup = await api.listSshPublicKeys();
    if (
      keysAfterCleanup.keys.some(
        (key) => sshPublicKeyIdentity(key) === identity,
      )
    ) {
      throw new Error("Runpod did not remove the temporary SSH public key");
    }
  };
};

const createPodForFirstAvailableGpu = async (
  api: ReturnType<typeof createRunpodApi>,
  gpuTypes: readonly string[],
  body: Readonly<Record<string, unknown>>,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<RunpodPod> => {
  let lastError: unknown;
  for (const gpuType of gpuTypes) {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        return await api.createPod({
          ...body,
          gpu: {
            id: gpuType,
            count: 1,
            minCudaVersion: "12.8",
            minVcpuCountPerGpu: 4,
            minRamPerGpu: 16,
          },
        });
      } catch (error) {
        lastError = error;
        if (hasStatus(error, 429) || hasServerStatus(error)) {
          if (attempt === 4) throw error;
          await sleep(retryAfterMilliseconds(error) ?? 2 ** attempt * 1_000);
          continue;
        }
        if (hasStatus(error, 400) || hasStatus(error, 403)) break;
        throw error;
      }
    }
  }
  throw lastError ?? new Error("Runpod has no configured GPU type");
};

const removeExpiredManagedPods = async (
  api: ReturnType<typeof createRunpodApi>,
  now: number,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<void> => {
  const pods = await api.listPods();
  for (const pod of pods) {
    const match = /^swapai-(\d+)-/.exec(pod.name);
    if (
      match !== null &&
      Number(match[1]) <= now &&
      pod.status !== "TERMINATED"
    ) {
      await deletePodVerified(api, pod.id, sleep);
    }
  }
};

const waitForConnectablePod = async (
  api: ReturnType<typeof createRunpodApi>,
  id: string,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
  signal: AbortSignal,
): Promise<RunpodPod> => {
  for (let attempt = 0; attempt < 144; attempt += 1) {
    signal.throwIfAborted();
    const pod = await api.getPod(id);
    if (
      pod.status === "EXITED" ||
      pod.status === "ERROR" ||
      pod.status === "TERMINATED"
    ) {
      throw new Error(`Runpod Pod ${id} ended with status ${pod.status}`);
    }
    if (
      pod.status === "RUNNING" &&
      ((pod.ssh?.direct !== undefined && pod.ssh.direct !== null) ||
        (pod.ssh?.proxy !== undefined && pod.ssh.proxy !== null))
    )
      return pod;
    await waitWithSignal(sleep, 5_000, signal);
  }
  throw new Error(`Runpod Pod ${id} did not expose SSH within 12 minutes`);
};

const waitForSsh = async (
  connections: readonly (readonly string[])[],
  runCommand: NonNullable<RunpodTrainerDependencies["runCommand"]>,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
  signal: AbortSignal,
): Promise<readonly string[]> => {
  let lastError: unknown;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    signal.throwIfAborted();
    for (const ssh of connections) {
      try {
        await runCommand("ssh", [...ssh, "true"], { signal });
        return ssh;
      } catch (error) {
        if (signal.aborted) throw signal.reason ?? error;
        lastError = error;
      }
    }
    await waitWithSignal(sleep, 5_000, signal);
  }
  const failure = boundedFailureMessage(lastError);
  throw new Error(
    `Runpod Pod SSH did not become ready within 10 minutes${failure === "" ? "" : `; last failure: ${failure}`}`,
    lastError === undefined ? undefined : { cause: lastError },
  );
};

const boundedFailureMessage = (error: unknown): string =>
  (error instanceof Error ? error.message : String(error))
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 300);

const deletePodVerified = async (
  api: ReturnType<typeof createRunpodApi>,
  id: string,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
): Promise<void> =>
  withAbortTimeout(
    (signal) => deletePodWithinCleanupReserve(api, id, sleep, signal),
    CLEANUP_HEADROOM_MS,
    `Runpod Pod ${id} cleanup exceeded the five minute reserve`,
  );

const deletePodWithinCleanupReserve = async (
  api: ReturnType<typeof createRunpodApi>,
  id: string,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
  signal: AbortSignal,
): Promise<void> => {
  try {
    await api.deletePod(id, signal);
  } catch (error) {
    if (!hasStatus(error, 404)) throw error;
  }
  for (let attempt = 0; attempt < 60; attempt += 1) {
    signal.throwIfAborted();
    try {
      const pod = await api.getPod(id, signal);
      if (pod.status === "TERMINATED") return;
    } catch (error) {
      if (hasStatus(error, 404)) return;
      throw error;
    }
    await waitWithSignal(sleep, 3_000, signal);
  }
  throw new Error(`Runpod Pod ${id} still exists after deletion`);
};

const deletePodAndReport = async (
  api: ReturnType<typeof createRunpodApi>,
  id: string,
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
  lifecycle: TrainingLifecycleReporter | undefined,
  additionalCleanup?: () => Promise<void>,
): Promise<void> => {
  const failures: unknown[] = [];
  try {
    lifecycle?.recordCleanup({ status: "pending" });
  } catch (error) {
    failures.push(error);
  }
  let deletionError: unknown;
  try {
    await deletePodVerified(api, id, sleep);
  } catch (error) {
    deletionError = error;
    failures.push(error);
  }
  let additionalCleanupError: unknown;
  if (additionalCleanup !== undefined) {
    try {
      await additionalCleanup();
    } catch (error) {
      additionalCleanupError = error;
      failures.push(error);
    }
  }
  try {
    lifecycle?.recordCleanup(
      deletionError === undefined && additionalCleanupError === undefined
        ? { status: "succeeded" }
        : {
            status: "failed",
            message: [deletionError, additionalCleanupError]
              .filter((error) => error !== undefined)
              .map((error) =>
                error instanceof Error ? error.message : String(error),
              )
              .join("; "),
          },
    );
  } catch (error) {
    failures.push(error);
  }
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) {
    throw new AggregateError(failures, `Could not clean up Runpod Pod ${id}`);
  }
};

const sshArguments = (
  host: string,
  port: number,
  privateKey: string,
  knownHostsPath: string,
  username = "root",
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
  `${username}@${host}`,
];

const scpConnectionArguments = (ssh: readonly string[]): readonly string[] => {
  const args = ssh.slice(0, -1);
  if (args[0] === "-p") args[0] = "-P";
  return ["-O", ...args];
};

const resolvePrivateKey = (configured: string | undefined): string => {
  const candidates =
    configured === undefined
      ? [
          join(homedir(), ".ssh", "id_ed25519"),
          join(homedir(), ".ssh", "id_rsa"),
        ]
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
  execution?: { readonly signal?: AbortSignal },
): Promise<{ readonly stdout: string; readonly stderr: string }> =>
  new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
      ...(execution?.signal === undefined ? {} : { signal: execution.signal }),
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.once("error", reject);
    child.once("close", (code, signal) => {
      if (code === 0) resolve({ stdout, stderr });
      else
        reject(
          new Error(
            `${command} exited with ${signal ?? code}${stderr === "" ? "" : `: ${stderr}`}`,
          ),
        );
    });
  });

const withAbortTimeout = async <Value>(
  operation: (signal: AbortSignal) => Promise<Value>,
  milliseconds: number,
  message: string,
): Promise<Value> => {
  const controller = new AbortController();
  let timeout: NodeJS.Timeout | undefined;
  try {
    const aborted = new Promise<never>((_resolve, reject) => {
      controller.signal.addEventListener(
        "abort",
        () => reject(controller.signal.reason),
        { once: true },
      );
    });
    timeout = setTimeout(
      () => controller.abort(new Error(message)),
      milliseconds,
    );
    timeout.unref();
    return await Promise.race([operation(controller.signal), aborted]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
};

const waitWithSignal = async (
  sleep: NonNullable<RunpodTrainerDependencies["sleep"]>,
  milliseconds: number,
  signal: AbortSignal,
): Promise<void> => {
  signal.throwIfAborted();
  let rejectForAbort: (() => void) | undefined;
  const aborted = new Promise<never>((_resolve, reject) => {
    rejectForAbort = () => reject(signal.reason);
    signal.addEventListener("abort", rejectForAbort, { once: true });
  });
  try {
    await Promise.race([sleep(milliseconds), aborted]);
  } finally {
    if (rejectForAbort !== undefined) {
      signal.removeEventListener("abort", rejectForAbort);
    }
  }
};

const validateOptions = (options: RunpodTrainerOptions): void => {
  if (options.apiKey.trim() === "")
    throw new TypeError("Runpod apiKey must not be empty");
  if (!Number.isFinite(options.maximumCostUsd) || options.maximumCostUsd <= 0) {
    throw new TypeError("Runpod maximumCostUsd must be positive");
  }
  if (
    options.maximumRuntimeMinutes !== undefined &&
    (!Number.isFinite(options.maximumRuntimeMinutes) ||
      options.maximumRuntimeMinutes <= CLEANUP_HEADROOM_MS / 60_000)
  ) {
    throw new TypeError(
      "Runpod maximumRuntimeMinutes must exceed the 5 minute cleanup reserve",
    );
  }
  if (
    options.maximumRuntimeMinutes !== undefined &&
    options.maximumRuntimeMinutes > 360
  )
    throw new TypeError("Runpod maximumRuntimeMinutes must not exceed 360");
  if (options.gpuTypes !== undefined && options.gpuTypes.length === 0) {
    throw new TypeError("Runpod gpuTypes must not be empty");
  }
  const requestedCloud = options as {
    readonly cloud?: string;
    readonly cloudType?: string;
  };
  if (
    requestedCloud.cloud === "COMMUNITY" ||
    requestedCloud.cloudType === "COMMUNITY"
  ) {
    throw new TypeError(
      "Runpod training requires Secure Cloud because direct SSH is not guaranteed on Community Cloud",
    );
  }
};

const hasStatus = (error: unknown, status: number): boolean =>
  typeof error === "object" &&
  error !== null &&
  "status" in error &&
  error.status === status;

const hasServerStatus = (error: unknown): boolean =>
  typeof error === "object" &&
  error !== null &&
  "status" in error &&
  typeof error.status === "number" &&
  error.status >= 500;

const retryAfterMilliseconds = (error: unknown): number | undefined =>
  typeof error === "object" &&
  error !== null &&
  "retryAfterMilliseconds" in error &&
  typeof error.retryAfterMilliseconds === "number"
    ? error.retryAfterMilliseconds
    : undefined;
