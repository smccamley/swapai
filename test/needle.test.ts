import { describe, expect, it } from "vitest";

import {
  MAX_NEEDLE_NUMBER_LABELS,
  createNeedleTool,
  createNeedleNumberLabels,
  createTrainingLine,
  readClassificationResult,
} from "../src/needle.js";

describe("Needle classifier format", () => {
  it("encodes numeric results as closed Needle classification buckets", () => {
    const config = { type: "number", min: 0, max: 1 } as const;
    const labels = createNeedleNumberLabels([0.92, 0.08, 0.92], config);
    const tool = createNeedleTool(config, labels);
    expect(tool).toMatchObject({
      name: "classify",
      parameters: {
        properties: {
          result: {
            type: "string",
            description: "The numeric classification bucket from 0 to 1.",
          },
        },
      },
    });
    const values = tool.parameters.properties.result.enum as string[];
    expect(labels.values).toEqual([0.08, 0.92]);
    expect(values).toEqual(["lower_score", "higher_score"]);
  });

  it("bounds high-cardinality numeric results and maps each result to its nearest label", () => {
    const config = { type: "number", min: 0, max: 1 } as const;
    const values = Array.from({ length: 10_000 }, (_, index) => index / 9_999);
    const labels = createNeedleNumberLabels(values, config, 0.01);

    expect(labels.values).toHaveLength(51);
    expect(Math.abs(labels.values[0]! - config.min)).toBeLessThanOrEqual(0.01);
    expect(Math.abs(config.max - labels.values.at(-1)!)).toBeLessThanOrEqual(0.01);
    expect(new Set(labels.values).size).toBe(51);

    const rows = values.map((value, index) =>
      createTrainingLine(`score ${index}`, value, config, labels),
    );
    expect(Buffer.byteLength(rows.join("\n"))).toBeLessThan(20_000_000);
    expect(JSON.parse(rows[5_000]!).answers[0].arguments.result).toMatch(
      /^score_bucket_\d+$/,
    );
  });

  it("caps zero-error numeric labels at the private Needle limit", () => {
    const config = { type: "number", min: 0, max: 1 } as const;
    const values = Array.from({ length: 10_000 }, (_, index) => index / 9_999);
    expect(createNeedleNumberLabels(values, config, 0).values).toHaveLength(
      MAX_NEEDLE_NUMBER_LABELS,
    );
  });

  it("builds boolean and closed string classify tools", () => {
    expect(
      createNeedleTool({ type: "string", values: ["rabbit", "fish", "pig"] }),
    ).toMatchObject({
      parameters: {
        properties: {
          result: { type: "string", enum: ["rabbit", "fish", "pig"] },
        },
      },
    });

    expect(createNeedleTool({ type: "boolean" })).toMatchObject({
      parameters: { properties: { result: { type: "boolean" } } },
    });
  });

  it("renders exact training JSON without reasoning or confidence", () => {
    const config = { type: "boolean" } as const;
    const line = createTrainingLine("I owe you £5", true, config);

    expect(JSON.parse(line)).toEqual({
      query: "I owe you £5",
      tools: [createNeedleTool(config)],
      answers: [{ name: "classify", arguments: { result: true } }],
    });
    expect(line).not.toContain("confidence");
    expect(line).not.toContain("reasoning");
  });

  it("writes numeric training labels as closed buckets and decodes them back to numbers", () => {
    const config = { type: "number", min: 0, max: 1 } as const;
    const labels = createNeedleNumberLabels([0.92, 0.81], config);
    const line = createTrainingLine("Accountant relevance", 0.92, config, labels);

    expect(JSON.parse(line).answers).toEqual([
      { name: "classify", arguments: { result: "higher_score" } },
    ]);
    expect(
      readClassificationResult(
        {
          function_calls: [
            { name: "classify", arguments: { result: "lower_score" } },
          ],
        },
        config,
        labels,
      ),
    ).toBe(0.81);
    expect(() =>
      readClassificationResult(
        {
          function_calls: [
            { name: "classify", arguments: { result: "score_bucket_999" } },
          ],
        },
        config,
        labels,
      ),
    ).toThrow(/number outside 0 to 1/i);
    expect(() =>
      readClassificationResult(
        {
          function_calls: [
            { name: "classify", arguments: { result: "score_bucket_00" } },
          ],
        },
        config,
        labels,
      ),
    ).toThrow(/number outside 0 to 1/i);
  });

  it("accepts exactly one classify call and validates its result", () => {
    expect(
      readClassificationResult(
        {
          type: "call",
          function_calls: [
            { name: "classify", arguments: { result: "gbp" } },
          ],
        },
        { type: "string", values: ["gbp", "usd", "eur"] },
      ),
    ).toBe("gbp");

    expect(() =>
      readClassificationResult(
        { function_calls: [] },
        { type: "boolean" },
      ),
    ).toThrow(/exactly one classify call/i);

    expect(() =>
      readClassificationResult(
        {
          function_calls: [
            { name: "classify", arguments: { result: "cad" } },
          ],
        },
        { type: "string", values: ["gbp", "usd", "eur"] },
      ),
    ).toThrow(/not an allowed string result/i);
  });
});
