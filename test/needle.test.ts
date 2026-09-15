import { describe, expect, it } from "vitest";

import {
  createNeedleTool,
  createTrainingLine,
  readClassificationResult,
} from "../src/needle.js";

describe("Needle classifier format", () => {
  it("builds a bounded numeric classify tool", () => {
    expect(createNeedleTool({ type: "number", min: 0, max: 1 })).toEqual({
      name: "classify",
      description: "Classify every supplied input and always return the learned result.",
      parameters: {
        type: "object",
        properties: {
          result: {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: "The classification result.",
          },
        },
        required: ["result"],
        additionalProperties: false,
      },
    });
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
