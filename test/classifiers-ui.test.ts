import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { startClassifiersUi } from "../src/classifiers-ui.js";
import { openStorage } from "../src/storage.js";

const temporaryDirectories: string[] = [];

function makeDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "swapai-ui-"));
  temporaryDirectories.push(directory);
  return directory;
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("classifier monitor web server", () => {
  it("serves the monitor and its JSON API", async () => {
    const dataDirectory = makeDirectory();
    const storage = openStorage({
      dataDirectory,
      name: "currency",
      config: {
        result: { type: "string", values: ["gbp", "usd", "eur"] },
        acceptableError: 0.05,
        retrainOnCount: 50,
        retestInterval: 100,
        retestRevertOn: 3,
        model: "needle2",
      },
      maxTrainingSet: 10_000,
    });
    storage.addExample("Pounds", "gbp");
    const server = await startClassifiersUi({ dataDirectory, port: 0 });

    try {
      const [page, api, health] = await Promise.all([
        fetch(server.url),
        fetch(`${server.url}api/classifiers`),
        fetch(`${server.url}health`),
      ]);
      expect(page.status).toBe(200);
      const pageBody = await page.text();
      expect(pageBody).toContain("SwapAI classifiers");
      expect(pageBody).toContain("Training deficits");
      expect(pageBody).toContain("Result-bin coverage");
      expect(pageBody).toContain("Facet coverage");
      expect(pageBody).toContain("Training run history");
      expect(pageBody).toContain("Protected evaluation");
      expect(pageBody).toContain("Shadow evidence");
      expect(pageBody).toContain("Provider resources");
      expect(pageBody).toContain("Cleanup");
      expect(pageBody).not.toContain("modelPath");
      expect(api.status).toBe(200);
      await expect(api.json()).resolves.toMatchObject({
        classifiers: [expect.objectContaining({ name: "currency" })],
      });
      await expect(health.json()).resolves.toEqual({
        service: "swapai-classifiers-ui",
        status: "ok",
      });
    } finally {
      await server.close();
      storage.close();
    }
  });
});
