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
  it("deletes and verifies its exact Pod when remote training fails", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    const requests: Array<{ url: string; method: string; body?: unknown }> = [];
    let deleted = false;
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      requests.push({
        url,
        method,
        ...(init?.body === undefined ? {} : { body: JSON.parse(String(init.body)) }),
      });
      if (url.endsWith("/v2/pods") && method === "GET") {
        return json({ pods: [], pagination: { nextCursor: null, hasNextPage: false } });
      }
      if (url.endsWith("/v2/pods") && method === "POST") {
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
            direct: { host: "203.0.113.10", port: 22022, username: "root" },
          },
        });
      }
      return new Response("unexpected request", { status: 500 });
    });
    const runCommand = vi.fn(async (command: string, args: readonly string[]) => {
      if (command === "ssh-keygen") return { stdout: "ssh-ed25519 public", stderr: "" };
      if (command === "ssh" && args.at(-1) === "true") {
        return { stdout: "", stderr: "" };
      }
      if (command === "scp") return { stdout: "", stderr: "" };
      if (command === "ssh") throw new Error("trainer exited 1");
      throw new Error(`unexpected command: ${command}`);
    });
    const trainer = runpodTrainer(
      {
        apiKey: "runpod-test-key",
        maximumCostUsd: 1,
        maximumRuntimeMinutes: 30,
        sshPrivateKey: privateKey,
      },
      { fetch: fetch as typeof globalThis.fetch, runCommand, sleep: async () => undefined },
    );

    await expect(trainer.train(trainingJob(directory))).rejects.toThrow("trainer exited 1");
    expect(requests).toEqual(expect.arrayContaining([
      expect.objectContaining({
        method: "POST",
        url: "https://api.runpod.io/v2/pods",
        body: expect.objectContaining({
          image: "ghcr.io/smccamley/swapai-trainer:0.5.0",
          gpu: expect.objectContaining({ id: "NVIDIA A40", count: 1 }),
          cloud: "SECURE",
          disk: 30,
          ports: ["22/tcp"],
        }),
      }),
      expect.objectContaining({ method: "DELETE", url: expect.stringMatching(/\/v2\/pods\/pod-123$/) }),
    ]));
    expect(deleted).toBe(true);
  });

  it("terminates an orphaned Pod after the configured runtime limit", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-orphan-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v2/pods/pod-orphan") && init?.method === "DELETE") {
        deleted = true;
        return json(null);
      }
      if (url.endsWith("/v2/pods/pod-orphan") && deleted) {
        return new Response("not found", { status: 404 });
      }
      if (url.endsWith("/v2/pods/pod-orphan")) {
        return json({ id: "pod-orphan", name: "swapai-orphan", status: "RUNNING" });
      }
      return new Response("unexpected request", { status: 500 });
    });
    const cleanup: string[] = [];
    const trainer = runpodTrainer({
      apiKey: "test-key",
      maximumCostUsd: 1,
      maximumRuntimeMinutes: 30,
      sshPrivateKey: privateKey,
    }, {
      fetch: fetch as typeof globalThis.fetch,
      sleep: async () => undefined,
      now: () => 31 * 60_000,
    });

    await expect(trainer.reconcile!({
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
    }, {
      recordProviderRun: () => undefined,
      recordCleanup: ({ status }) => cleanup.push(status),
    })).resolves.toMatchObject({ status: "failed" });
    expect(deleted).toBe(true);
    expect(cleanup).toEqual(["pending", "succeeded"]);
  });

  it("deletes and verifies an exited Pod before reporting cleanup success", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-exited-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v2/pods/pod-exited") && init?.method === "DELETE") {
        deleted = true;
        return json(null);
      }
      if (url.endsWith("/v2/pods/pod-exited") && deleted) {
        return new Response("not found", { status: 404 });
      }
      if (url.endsWith("/v2/pods/pod-exited")) {
        return json({ id: "pod-exited", name: "swapai-exited", status: "EXITED" });
      }
      return new Response("unexpected request", { status: 500 });
    });
    const cleanup: string[] = [];
    const trainer = runpodTrainer({
      apiKey: "test-key",
      maximumCostUsd: 1,
      maximumRuntimeMinutes: 30,
      sshPrivateKey: privateKey,
    }, {
      fetch: fetch as typeof globalThis.fetch,
      sleep: async () => undefined,
      now: () => 1_000,
    });

    await expect(trainer.reconcile!(inspectionRun({
      providerRunId: "pod-exited",
      startedAt: 500,
    }), {
      recordProviderRun: () => undefined,
      recordCleanup: ({ status }) => cleanup.push(status),
    })).resolves.toEqual({
      status: "failed",
      failureMessage: "Runpod Pod ended with status EXITED",
    });
    expect(deleted).toBe(true);
    expect(cleanup).toEqual(["pending", "succeeded"]);
  });

  it("deletes a newly created Pod when recording its provider ID throws", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-reporter-"));
    temporaryDirectories.push(directory);
    const privateKey = join(directory, "id_ed25519");
    writeFileSync(privateKey, "test-key");
    let deleted = false;
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
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
      if (url.endsWith("/v2/pods/pod-reporter") && init?.method === "DELETE") {
        deleted = true;
        return json(null);
      }
      if (url.endsWith("/v2/pods/pod-reporter") && deleted) {
        return new Response("not found", { status: 404 });
      }
      return new Response("unexpected request", { status: 500 });
    });
    const providerRecordFailure = new Error("provider run record failed");
    const trainer = runpodTrainer({
      apiKey: "test-key",
      maximumCostUsd: 1,
      sshPrivateKey: privateKey,
    }, {
      fetch: fetch as typeof globalThis.fetch,
      runCommand: async () => ({ stdout: "ssh-ed25519 public", stderr: "" }),
      sleep: async () => undefined,
    });

    await expect(trainer.train(trainingJob(directory), {
      recordProviderRun: () => { throw providerRecordFailure; },
      recordCleanup: () => undefined,
    })).rejects.toBe(providerRecordFailure);
    expect(deleted).toBe(true);
  });

  it("deletes a Pod even when recording pending cleanup throws", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swapai-runpod-cleanup-reporter-"));
    temporaryDirectories.push(directory);
    let deleted = false;
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v2/pods/pod-cancel") && init?.method === "DELETE") {
        deleted = true;
        return json(null);
      }
      if (url.endsWith("/v2/pods/pod-cancel") && deleted) {
        return new Response("not found", { status: 404 });
      }
      return new Response("unexpected request", { status: 500 });
    });
    const pendingRecordFailure = new Error("pending cleanup record failed");
    const statuses: string[] = [];
    const trainer = runpodTrainer({
      apiKey: "test-key",
      maximumCostUsd: 1,
    }, {
      fetch: fetch as typeof globalThis.fetch,
      sleep: async () => undefined,
    });

    await expect(trainer.cancel!(inspectionRun({
      providerRunId: "pod-cancel",
    }), {
      recordProviderRun: () => undefined,
      recordCleanup: ({ status }) => {
        statuses.push(status);
        if (status === "pending") throw pendingRecordFailure;
      },
    })).rejects.toBe(pendingRecordFailure);
    expect(deleted).toBe(true);
    expect(statuses).toEqual(["pending", "succeeded"]);
  });
});

const inspectionRun = (overrides: Partial<Parameters<NonNullable<TrainingProvider["reconcile"]>>[0]>) => ({
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

const json = (value: unknown): Response => new Response(JSON.stringify(value), {
  status: 200,
  headers: { "content-type": "application/json" },
});
