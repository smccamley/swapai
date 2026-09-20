import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { runpodTrainer } from "../src/runpod.js";
import type { TrainingJob, TrainingProvider } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("runpodTrainer", () => {
  it("rejects Community Pods because direct SSH is not guaranteed", () => {
    expect(() =>
      runpodTrainer({
        apiKey: "test-key",
        maximumCostUsd: 1,
        // @ts-expect-error Community Cloud cannot guarantee this SSH boundary.
        cloud: "COMMUNITY",
      }),
    ).toThrow(/secure.*ssh/i);
  });

  it("aborts a Runpod API request that does not return", async () => {
    let requestSignal: AbortSignal | null = null;
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
      },
      {
        requestTimeoutMilliseconds: 10,
        fetch: vi.fn(
          async (_input: string | URL | Request, init?: RequestInit) => {
            requestSignal = init?.signal as AbortSignal;
            return new Promise<Response>((_resolve, reject) => {
              requestSignal!.addEventListener(
                "abort",
                () => reject(requestSignal!.reason),
                { once: true },
              );
            });
          },
        ) as typeof globalThis.fetch,
      },
    );

    await expect(
      trainer.reconcile!(
        inspectionRun({
          providerRunId: "pod-timeout",
          startedAt: Date.now(),
        }),
      ),
    ).rejects.toThrow(/timed out|timeout/i);
    expect(
      Boolean(requestSignal && (requestSignal as AbortSignal).aborted),
    ).toBe(true);
  });

  it("aborts a child process that does not return", async () => {
    vi.useFakeTimers();
    let processSignal: AbortSignal | null = null;
    const directory = mkdtempSync(
      join(tmpdir(), "swapai-runpod-process-timeout-"),
    );
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: vi.fn(async () => json({ pods: [] })) as typeof globalThis.fetch,
        runCommand: async (_command, _args, execution) => {
          processSignal = execution?.signal ?? null;
          return new Promise((_resolve, reject) => {
            processSignal!.addEventListener(
              "abort",
              () => reject(processSignal!.reason),
              { once: true },
            );
          });
        },
      },
    );

    try {
      const training = trainer.train(trainingJob(directory));
      const trainingFailure = training.then(
        () => undefined,
        (error: unknown) => error,
      );
      await vi.advanceTimersByTimeAsync(30_000);
      await expect(trainingFailure).resolves.toEqual(
        expect.objectContaining({
          message: expect.stringMatching(/public-key derivation timed out/i),
        }),
      );
      expect(
        Boolean(processSignal && (processSignal as AbortSignal).aborted),
      ).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("rejects a runtime that cannot contain the cleanup reserve before creating a Pod", () => {
    expect(() =>
      runpodTrainer({
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 5,
      }),
    ).toThrow(/exceed.*5 minute cleanup reserve/i);
  });

  it("deletes and verifies its exact Pod when remote training fails", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    const requests: Array<{ url: string; method: string; body?: unknown }> = [];
    let deleted = false;
    let registeredKeys = ["ssh-ed25519 existing account-key"];
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        const method = init?.method ?? "GET";
        requests.push({
          url,
          method,
          ...(init?.body === undefined
            ? {}
            : { body: JSON.parse(String(init.body)) }),
        });
        if (url.endsWith("/v2/account/ssh-keys") && method === "GET") {
          return json({ keys: registeredKeys });
        }
        if (url.endsWith("/v2/account/ssh-keys") && method === "PUT") {
          registeredKeys = (
            JSON.parse(String(init?.body)) as { keys: string[] }
          ).keys;
          return json({ keys: registeredKeys });
        }
        if (url.endsWith("/v2/pods") && method === "GET") {
          return json({
            pods: [],
            pagination: { nextCursor: null, hasNextPage: false },
          });
        }
        if (url.endsWith("/v2/pods") && method === "POST") {
          const body = JSON.parse(String(init?.body)) as {
            gpu?: { id?: string };
          };
          if (body.gpu?.id === "NVIDIA A40") {
            return new Response("unavailable", { status: 400 });
          }
          return json({
            id: "pod-123",
            name: "swapai-test",
            status: "PROVISIONING",
            cost: 0.5,
          });
        }
        if (url.endsWith("/v2/pods/pod-123") && method === "DELETE") {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-123") && deleted) {
          return new Response("not found", { status: 404 });
        }
        if (url.endsWith("/v2/pods/pod-123")) {
          return json({
            id: "pod-123",
            name: "swapai-test",
            status: "RUNNING",
            cost: 0.5,
            ssh: {
              proxy: {
                host: "ssh.runpod.io",
                port: 22,
                username: "pod-123-user",
              },
              direct: null,
            },
          });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const runCommand = vi.fn(
      async (
        command: string,
        args: readonly string[],
        execution?: { readonly signal?: AbortSignal },
      ) => {
        if (command === "ssh-keygen")
          return { stdout: "ssh-ed25519 public", stderr: "" };
        if (command === "ssh" && args.at(-1) === "true") {
          return { stdout: "", stderr: "" };
        }
        if (command === "scp") return { stdout: "", stderr: "" };
        if (command === "ssh") throw new Error("trainer exited 1");
        throw new Error(`unexpected command: ${command}`);
      },
    );
    const trainer = runpodTrainer(
      {
        apiKey: "runpod-test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
        registerSshPublicKeyForTraining: true,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        runCommand,
        sleep: async () => undefined,
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      "trainer exited 1",
    );
    expect(requests).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "POST",
          url: "https://api.runpod.io/v2/pods",
          body: expect.objectContaining({
            image: "ghcr.io/smccamley/swapai-trainer:0.6.4",
            gpu: expect.objectContaining({
              id: "NVIDIA RTX A5000",
              count: 1,
              minCudaVersion: "12.8",
            }),
            cloud: "SECURE",
            disk: 30,
            ports: ["22/tcp"],
            startSsh: true,
          }),
        }),
        expect.objectContaining({
          method: "DELETE",
          url: expect.stringMatching(/\/v2\/pods\/pod-123$/),
        }),
      ]),
    );
    expect(deleted).toBe(true);
    expect(registeredKeys).toEqual(["ssh-ed25519 existing account-key"]);
    const keyWrites = requests.filter(
      ({ method, url }) =>
        method === "PUT" && url.endsWith("/v2/account/ssh-keys"),
    );
    expect(keyWrites).toHaveLength(2);
    expect(keyWrites[0]?.body).toEqual({
      keys: ["ssh-ed25519 existing account-key", "ssh-ed25519 public"],
    });
    expect(keyWrites[1]?.body).toEqual({
      keys: ["ssh-ed25519 existing account-key"],
    });
  });

  it("falls back from an unreachable direct SSH endpoint to the Runpod proxy", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-ssh-fallback-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    const sshAttempts: string[] = [];
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        const method = init?.method ?? "GET";
        if (url.endsWith("/v2/pods") && method === "GET") {
          return json({ pods: [] });
        }
        if (url.endsWith("/v2/pods") && method === "POST") {
          return json({
            id: "pod-ssh-fallback",
            name: "swapai-ssh-fallback",
            status: "PROVISIONING",
            cost: 0.5,
          });
        }
        if (
          url.endsWith("/v2/pods/pod-ssh-fallback") &&
          method === "DELETE"
        ) {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-ssh-fallback") && deleted) {
          return new Response("not found", { status: 404 });
        }
        if (url.endsWith("/v2/pods/pod-ssh-fallback")) {
          return json({
            id: "pod-ssh-fallback",
            name: "swapai-ssh-fallback",
            status: "RUNNING",
            cost: 0.5,
            ssh: {
              direct: {
                host: "203.0.113.10",
                port: 22022,
                username: "root",
              },
              proxy: {
                host: "ssh.runpod.io",
                port: 22,
                username: "pod-user",
              },
            },
          });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const runCommand = vi.fn(
      async (command: string, args: readonly string[]) => {
        if (command === "ssh-keygen") {
          return { stdout: "ssh-ed25519 public", stderr: "" };
        }
        if (command === "ssh" && args.at(-1) === "true") {
          const destination = args.at(-2)!;
          sshAttempts.push(destination);
          if (destination === "root@203.0.113.10") {
            throw new Error("direct route is unreachable");
          }
          return { stdout: "", stderr: "" };
        }
        if (command === "scp") throw new Error("stop after SSH fallback");
        throw new Error(`unexpected command: ${command}`);
      },
    );
    const trainer = runpodTrainer(
      {
        apiKey: "runpod-test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        runCommand,
        sleep: async () => undefined,
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      "stop after SSH fallback",
    );
    expect(sshAttempts).toEqual([
      "root@203.0.113.10",
      "pod-user@ssh.runpod.io",
    ]);
    expect(deleted).toBe(true);
  });

  it("moves to the next GPU after a pool-specific forbidden response", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-forbidden-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let creates = 0;
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: vi.fn(
          async (input: string | URL | Request, init?: RequestInit) => {
            if (
              String(input).endsWith("/v2/pods") &&
              init?.method === undefined
            ) {
              return json({ pods: [] });
            }
            creates += 1;
            return new Response(creates === 1 ? "forbidden" : "invalid", {
              status: creates === 1 ? 403 : 422,
            });
          },
        ) as typeof globalThis.fetch,
        runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      /422.*invalid/i,
    );
    expect(creates).toBe(2);
  });

  it("does not retry a contract-invalid Pod request", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-invalid-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let creates = 0;
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: vi.fn(
          async (input: string | URL | Request, init?: RequestInit) => {
            if (
              String(input).endsWith("/v2/pods") &&
              init?.method === undefined
            ) {
              return json({ pods: [] });
            }
            creates += 1;
            return new Response("invalid", { status: 422 });
          },
        ) as typeof globalThis.fetch,
        runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      /422.*invalid/i,
    );
    expect(creates).toBe(1);
  });

  it("retries rate limits and server failures on the same GPU", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-retry-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    const requestedGpuTypes: string[] = [];
    const sleepDurations: number[] = [];
    const statuses = [429, 500, 422];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: vi.fn(
          async (input: string | URL | Request, init?: RequestInit) => {
            if (
              String(input).endsWith("/v2/pods") &&
              init?.method === undefined
            ) {
              return json({ pods: [] });
            }
            const body = JSON.parse(String(init?.body)) as {
              gpu: { id: string };
            };
            requestedGpuTypes.push(body.gpu.id);
            const status = statuses.shift()!;
            return new Response(`status-${status}`, {
              status,
              ...(status === 429 ? { headers: { "retry-after": "7" } } : {}),
            });
          },
        ) as typeof globalThis.fetch,
        runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
        sleep: async (milliseconds) => {
          sleepDurations.push(milliseconds);
        },
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      /422.*status-422/i,
    );
    expect(requestedGpuTypes).toEqual([
      "NVIDIA A40",
      "NVIDIA A40",
      "NVIDIA A40",
    ]);
    expect(sleepDurations).toEqual([7_000, 2_000]);
  });

  it("fails immediately when a provisioning Pod enters a terminal state", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-terminal-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const sleep = vi.fn(async () => undefined);
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: vi.fn(
          async (input: string | URL | Request, init?: RequestInit) => {
            const url = String(input);
            if (url.endsWith("/v2/pods") && init?.method === undefined) {
              return json({ pods: [] });
            }
            if (url.endsWith("/v2/pods") && init?.method === "POST") {
              return json({
                id: "pod-terminal",
                name: "swapai-terminal",
                status: "PROVISIONING",
                cost: 0.5,
              });
            }
            if (
              url.endsWith("/v2/pods/pod-terminal") &&
              init?.method === "DELETE"
            ) {
              deleted = true;
              return json(null);
            }
            if (url.endsWith("/v2/pods/pod-terminal") && deleted) {
              return new Response("not found", { status: 404 });
            }
            if (url.endsWith("/v2/pods/pod-terminal")) {
              return json({
                id: "pod-terminal",
                name: "swapai-terminal",
                status: "ERROR",
                cost: 0.5,
              });
            }
            return new Response("unexpected request", { status: 500 });
          },
        ) as typeof globalThis.fetch,
        runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
        sleep,
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      /ended with status ERROR/i,
    );
    expect(sleep).not.toHaveBeenCalled();
    expect(deleted).toBe(true);
  });

  it("terminates an orphaned Pod after the configured runtime limit", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-orphan-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods/pod-orphan") && init?.method === "DELETE") {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-orphan") && deleted) {
          return new Response("not found", { status: 404 });
        }
        if (url.endsWith("/v2/pods/pod-orphan")) {
          return json({
            id: "pod-orphan",
            name: "swapai-orphan",
            status: "RUNNING",
          });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const cleanup: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        sleep: async () => undefined,
        now: () => 31 * 60_000,
      },
    );

    await expect(
      trainer.reconcile!(
        {
          id: "run-orphan",
          datasetRevisionId: "a".repeat(64),
          provider: "runpod",
          status: "running",
          providerRunId: "pod-orphan",
          costUsd: null,
          artifactSha256: null,
          failureMessage: null,
          resources: [{ type: "runpod-pod", id: "pod-orphan" }],
          cleanup: { status: "pending", message: null },
          evaluations: [],
          shadow: null,
          startedAt: 0,
          finishedAt: null,
        },
        {
          recordProviderRun: () => undefined,
          recordCleanup: ({ status }) => cleanup.push(status),
        },
      ),
    ).resolves.toMatchObject({ status: "failed" });
    expect(deleted).toBe(true);
    expect(cleanup).toEqual(["pending", "succeeded"]);
  });

  it("deletes and verifies an exited Pod before reporting cleanup success", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-exited-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods/pod-exited") && init?.method === "DELETE") {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-exited") && deleted) {
          return new Response("not found", { status: 404 });
        }
        if (url.endsWith("/v2/pods/pod-exited")) {
          return json({
            id: "pod-exited",
            name: "swapai-exited",
            status: "EXITED",
          });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const cleanup: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        sleep: async () => undefined,
        now: () => 1_000,
      },
    );

    await expect(
      trainer.reconcile!(
        inspectionRun({
          providerRunId: "pod-exited",
          startedAt: 500,
        }),
        {
          recordProviderRun: () => undefined,
          recordCleanup: ({ status }) => cleanup.push(status),
        },
      ),
    ).resolves.toEqual({
      status: "failed",
      failureMessage: "Runpod Pod ended with status EXITED",
    });
    expect(deleted).toBe(true);
    expect(cleanup).toEqual(["pending", "succeeded"]);
  });

  it("finishes cleanup for a failed durable run before it can be retried", async () => {
    const directory = mkdtempSync(
      join(tmpdir(), "swapai-runpod-failed-cleanup-"),
    );
    temporaryDirectories.push(directory);
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods/pod-failed") && init?.method === "DELETE") {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-failed") && deleted) {
          return new Response("not found", { status: 404 });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const cleanup: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        sleep: async () => undefined,
      },
    );

    await expect(
      trainer.reconcile!(
        inspectionRun({
          status: "failed",
          providerRunId: "pod-failed",
          cleanup: { status: "failed", message: "first delete failed" },
        }),
        {
          recordProviderRun: () => undefined,
          recordCleanup: ({ status }) => cleanup.push(status),
        },
      ),
    ).resolves.toMatchObject({ status: "failed" });
    expect(deleted).toBe(true);
    expect(cleanup).toEqual(["pending", "succeeded"]);
  });

  it("deletes a newly created Pod when recording its provider ID throws", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-reporter-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods") && init?.method === undefined) {
          return json({ pods: [] });
        }
        if (url.endsWith("/v2/pods") && init?.method === "POST") {
          return json({
            id: "pod-reporter",
            name: "swapai-reporter",
            status: "PROVISIONING",
            cost: 0.5,
          });
        }
        if (
          url.endsWith("/v2/pods/pod-reporter") &&
          init?.method === "DELETE"
        ) {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-reporter") && deleted) {
          return new Response("not found", { status: 404 });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const providerRecordFailure = new Error("provider run record failed");
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        sshPrivateKey: privateKey,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
        sleep: async () => undefined,
      },
    );

    await expect(
      trainer.train(trainingJob(directory), {
        recordProviderRun: () => {
          throw providerRecordFailure;
        },
        recordCleanup: () => undefined,
      }),
    ).rejects.toBe(providerRecordFailure);
    expect(deleted).toBe(true);
  });

  it("deletes a Pod even when recording pending cleanup throws", async () => {
    const directory = mkdtempSync(
      join(tmpdir(), "swapai-runpod-cleanup-reporter-"),
    );
    temporaryDirectories.push(directory);
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods/pod-cancel") && init?.method === "DELETE") {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-cancel") && deleted) {
          return new Response("not found", { status: 404 });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const pendingRecordFailure = new Error("pending cleanup record failed");
    const statuses: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        sleep: async () => undefined,
      },
    );

    await expect(
      trainer.cancel!(
        inspectionRun({
          providerRunId: "pod-cancel",
        }),
        {
          recordProviderRun: () => undefined,
          recordCleanup: ({ status }) => {
            statuses.push(status);
            if (status === "pending") throw pendingRecordFailure;
          },
        },
      ),
    ).rejects.toBe(pendingRecordFailure);
    expect(deleted).toBe(true);
    expect(statuses).toEqual(["pending", "succeeded"]);
  });

  it("recovers a Pod created before its ID was durably recorded", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-recover-id-"));
    temporaryDirectories.push(directory);
    const fetch = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/v2/pods")) {
        return json({
          pods: [
            {
              id: "pod-recovered",
              name: "swapai-1800000-run-crash",
              status: "RUNNING",
              cost: 0.5,
            },
          ],
        });
      }
      if (url.endsWith("/v2/pods/pod-recovered")) {
        return json({
          id: "pod-recovered",
          name: "swapai-1800000-run-crash",
          status: "RUNNING",
          cost: 0.5,
        });
      }
      return new Response("unexpected request", { status: 500 });
    });
    const recovered: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        now: () => 1_000,
      },
    );

    await expect(
      trainer.reconcile!(
        inspectionRun({
          id: "run-crash",
          providerRunId: null,
          resources: [],
        }),
        {
          recordProviderRun: ({ providerRunId }) =>
            recovered.push(providerRunId),
          recordCleanup: () => undefined,
        },
      ),
    ).resolves.toEqual({ status: "running" });
    expect(recovered).toEqual(["pod-recovered"]);
  });

  it("searches every Pod list page when recovering an unrecorded Pod", async () => {
    const requestedUrls: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
      },
      {
        fetch: vi.fn(async (input: string | URL | Request) => {
          const url = String(input);
          requestedUrls.push(url);
          if (url.endsWith("/v2/pods")) {
            return json({
              pods: [{ id: "other", name: "unrelated", status: "RUNNING" }],
              pagination: { nextCursor: "page 2", hasNextPage: true },
            });
          }
          if (url.endsWith("/v2/pods?cursor=page%202")) {
            return json({
              pods: [
                {
                  id: "pod-page-2",
                  name: "swapai-1800000-run-page-2",
                  status: "RUNNING",
                  cost: 0.5,
                },
              ],
              pagination: { nextCursor: null, hasNextPage: false },
            });
          }
          if (url.endsWith("/v2/pods/pod-page-2")) {
            return json({
              id: "pod-page-2",
              name: "swapai-1800000-run-page-2",
              status: "RUNNING",
              cost: 0.5,
            });
          }
          return new Response("unexpected request", { status: 500 });
        }) as typeof globalThis.fetch,
        now: () => 1_000,
      },
    );
    const recovered: string[] = [];

    await expect(
      trainer.reconcile!(
        inspectionRun({
          id: "run-page-2",
          providerRunId: null,
          resources: [],
        }),
        {
          recordProviderRun: ({ providerRunId }) =>
            recovered.push(providerRunId),
          recordCleanup: () => undefined,
        },
      ),
    ).resolves.toEqual({ status: "running" });
    expect(recovered).toEqual(["pod-page-2"]);
    expect(requestedUrls).toContain(
      "https://api.runpod.io/v2/pods?cursor=page%202",
    );
  });

  it("records cleanup when no unrecorded Pod exists after the runtime deadline", async () => {
    const cleanup: string[] = [];
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
      },
      {
        fetch: vi.fn(async () => json({ pods: [] })) as typeof globalThis.fetch,
        now: () => 1_800_000,
      },
    );

    await expect(
      trainer.reconcile!(
        inspectionRun({
          id: "run-without-pod",
          providerRunId: null,
          resources: [],
        }),
        {
          recordProviderRun: () => undefined,
          recordCleanup: ({ status }) => cleanup.push(status),
        },
      ),
    ).resolves.toMatchObject({
      status: "failed",
      failureMessage: expect.stringMatching(/no managed Pod exists/i),
    });
    expect(cleanup).toEqual(["succeeded"]);
  });

  it("deletes every ambiguous recovered Pod and fails closed", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-ambiguous-"));
    temporaryDirectories.push(directory);
    const deleted = new Set<string>();
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods")) {
          return json({
            pods: ["pod-a", "pod-b"].map((id) => ({
              id,
              name: `swapai-1800000-run-ambiguous`,
              status: "RUNNING",
              cost: 0.5,
            })),
          });
        }
        const id = url.endsWith("pod-a") ? "pod-a" : "pod-b";
        if (init?.method === "DELETE") {
          deleted.add(id);
          return json(null);
        }
        if (deleted.has(id)) return new Response("not found", { status: 404 });
        return new Response("unexpected request", { status: 500 });
      },
    );
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        sleep: async () => undefined,
        now: () => 1_000,
      },
    );

    await expect(
      trainer.reconcile!(
        inspectionRun({
          id: "run-ambiguous",
          providerRunId: null,
          resources: [],
        }),
        {
          recordProviderRun: () => undefined,
          recordCleanup: () => undefined,
        },
      ),
    ).resolves.toMatchObject({
      status: "failed",
      failureMessage: expect.stringMatching(/multiple managed/i),
    });
    expect(deleted).toEqual(new Set(["pod-a", "pod-b"]));
  });

  it("rejects a Pod when the cost limit leaves no positive training window", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-no-window-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods") && init?.method === undefined) {
          return json({ pods: [] });
        }
        if (url.endsWith("/v2/pods") && init?.method === "POST") {
          return json({
            id: "pod-no-window",
            name: "swapai-no-window",
            status: "PROVISIONING",
            cost: 60,
          });
        }
        if (
          url.endsWith("/v2/pods/pod-no-window") &&
          init?.method === "DELETE"
        ) {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-no-window") && deleted) {
          return new Response("not found", { status: 404 });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 0.01,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
        sleep: async () => undefined,
        now: () => 0,
      },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow(
      /no positive training window/i,
    );
    expect(deleted).toBe(true);
  });

  it("includes container storage inside the cost deadline", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const directory = mkdtempSync(
      join(tmpdir(), "swapai-runpod-cost-deadline-"),
    );
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/v2/pods") && init?.method === undefined) {
          return json({ pods: [] });
        }
        if (url.endsWith("/v2/pods") && init?.method === "POST") {
          return json({
            id: "pod-deadline",
            name: "swapai-deadline",
            status: "PROVISIONING",
            cost: 10,
          });
        }
        if (
          url.endsWith("/v2/pods/pod-deadline") &&
          init?.method === "DELETE"
        ) {
          deleted = true;
          return json(null);
        }
        if (url.endsWith("/v2/pods/pod-deadline") && deleted) {
          return new Response("not found", { status: 404 });
        }
        if (url.endsWith("/v2/pods/pod-deadline")) {
          return json({
            id: "pod-deadline",
            name: "swapai-deadline",
            status: "RUNNING",
            cost: 10,
            ssh: {
              direct: { host: "203.0.113.10", port: 22022, username: "root" },
            },
          });
        }
        return new Response("unexpected request", { status: 500 });
      },
    );
    const runCommand = vi.fn(
      async (command: string, args: readonly string[]) => {
        if (command === "ssh-keygen")
          return { stdout: "ssh-ed25519 public", stderr: "" };
        if (command === "ssh" && args.at(-1) === "true") {
          return { stdout: "", stderr: "" };
        }
        if (command === "scp") return { stdout: "", stderr: "" };
        return new Promise<{ stdout: string; stderr: string }>(() => undefined);
      },
    );
    const trainer = runpodTrainer(
      {
        apiKey: "test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 6,
        sshPrivateKey: privateKey,
      },
      {
        fetch: fetch as typeof globalThis.fetch,
        runCommand,
        sleep: async () => undefined,
      },
    );

    try {
      const training = trainer.train(trainingJob(directory));
      const trainingFailure = training.then(
        () => undefined,
        (error: unknown) => error,
      );
      let expiredAtStorageInclusiveDeadline = false;
      void trainingFailure.then(() => {
        expiredAtStorageInclusiveDeadline = true;
      });
      await vi.advanceTimersByTimeAsync(59_838);
      expect(deleted).toBe(false);
      await vi.advanceTimersByTimeAsync(1);
      const includedStorage = expiredAtStorageInclusiveDeadline;
      if (!includedStorage) await vi.advanceTimersByTimeAsync(161);
      await expect(trainingFailure).resolves.toEqual(
        expect.objectContaining({
          message: expect.stringMatching(/paid training window expired/i),
        }),
      );
      expect(includedStorage).toBe(true);
      expect(deleted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});

const inspectionRun = (
  overrides: Partial<Parameters<NonNullable<TrainingProvider["reconcile"]>>[0]>,
) => ({
  id: "run-orphan",
  datasetRevisionId: "a".repeat(64),
  provider: "runpod",
  status: "running" as const,
  providerRunId: "pod-orphan",
  costUsd: null,
  artifactSha256: null,
  failureMessage: null,
  resources: [{ type: "runpod-pod", id: "pod-orphan" }],
  cleanup: { status: "pending" as const, message: null },
  evaluations: [],
  shadow: null,
  startedAt: 0,
  finishedAt: null,
  ...overrides,
});

const trainingJob = (outputDirectory: string): TrainingJob => ({
  id: "36eb6bff-391b-4245-89d8-cc8e9a19693d",
  datasetRevisionId: "a".repeat(64),
  classifierName: "accountant-relevance",
  generation: 1,
  dataEpoch: 0,
  result: { type: "boolean" },
  acceptableError: 0.1,
  examples: [
    {
      input: "supplier invoice",
      result: true,
      resultBin: "true",
      purpose: "training",
      facets: {},
    },
  ],
  outputDirectory,
});

const json = (value: unknown): Response =>
  new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
