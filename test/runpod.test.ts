import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { runpodTrainer } from "../src/runpod.js";
import type { TrainingJob } from "../src/index.js";

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
    const requests: Array<{ url: string; method: string }> = [];
    let deleted = false;
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      requests.push({ url, method });
      if (url.endsWith("/pods") && method === "GET") return json([]);
      if (url.endsWith("/pods") && method === "POST") {
        return json({
          id: "pod-123",
          name: "swapai-test",
          costPerHr: 0.5,
        });
      }
      if (url.endsWith("/pods/pod-123") && method === "DELETE") {
        deleted = true;
        return json(null);
      }
      if (url.endsWith("/pods/pod-123") && deleted) {
        return new Response("not found", { status: 404 });
      }
      if (url.endsWith("/pods/pod-123")) {
        return json({
          id: "pod-123",
          name: "swapai-test",
          desiredStatus: "RUNNING",
          publicIp: "203.0.113.10",
          portMappings: { "22": 22022 },
          costPerHr: 0.5,
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
      expect.objectContaining({ method: "DELETE", url: expect.stringMatching(/\/pods\/pod-123$/) }),
    ]));
    expect(deleted).toBe(true);
  });
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
    {
      input: "newsletter",
      result: false,
      resultBin: "false",
      purpose: "validation",
      facets: {},
    },
  ],
  outputDirectory,
});

const json = (value: unknown): Response => new Response(JSON.stringify(value), {
  status: 200,
  headers: { "content-type": "application/json" },
});
