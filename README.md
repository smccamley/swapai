# SwapAI

SwapAI learns from an existing classifier, trains a local Needle 2 classifier,
and only uses it after it has passed a held-out test.

```sh
npm install @swapai/core
```

## Use it

```ts
import { init } from "@swapai/core";

const relevance = init({
  name: "accountant-relevance",
  result: { type: "number", min: 0, max: 1 },
  retrainOnCount: 50,
  acceptableError: "10%",
  retestInterval: 100,
  retestRevertOn: 3,
  model: "needle2",
  maxTrainingSet: 10000,
});

const score = await relevance.classify(input, async (value) => {
  const result = await callExistingClassifier(value);
  return result.score;
});
```

Before the local classifier is trained, `classify` calls the supplied reference
classifier and records its answer. Once the local classifier passes its
held-out test, it takes over. SwapAI periodically calls both classifiers to
check that the local result remains within the allowed error.

`input` must be the complete input sent to the reference classifier, including
the instruction that defines the classification task. Do not log only the raw
text being classified: Needle needs the same task and context the reference saw.

## Manual control

The callback is optional. You can keep control of the switch yourself:

```ts
if (relevance.isTrained()) {
  return relevance.classify(input);
}

const score = await callExistingClassifier(input);
relevance.logClassification(input, score);
return score;
```

`isTrained()` and `logClassification()` are synchronous. Logging validates the
result immediately and queues storage work. Use `await relevance.flush()` when
you need to know all accepted examples are durable, such as during shutdown.

## Delete one classifier's training data

```ts
relevance.clearTrainingData();
await relevance.flush();
```

`clearTrainingData()` is synchronous and makes `isTrained()` return `false`
immediately. It queues deletion of every retained example and saved model for
this classifier name. `flush()` resolves after deletion is durable. Other
classifiers and SwapAI's shared Needle runtime are left intact.

A local classification already in progress cannot return an old-epoch result
after the clear. It falls back to the supplied reference classifier, or rejects
with `not_trained` when no reference was supplied. Its old-epoch result is not
retained. Calls made after deletion finishes can build a new training set.

If deletion cannot finish, `flush()` rejects and keeps the clear pending. A
later `flush()` retries it. If the process exits, the next `init()` for that
classifier finishes the pending deletion before any saved model can be used.

## Result types

SwapAI supports bounded numbers, booleans, and a closed list of strings:

```ts
init({
  // other options
  result: { type: "boolean" },
});

init({
  // other options
  result: { type: "string", values: ["rabbit", "fish", "pig"] },
});
```

The string result is inferred as `"rabbit" | "fish" | "pig"`.

Needle is a classifier, not a regression engine. For bounded numbers, SwapAI
converts reference scores into a small, closed set of internal labels, then
converts the selected label back to a number. When the training set contains
only a few scores, those scores stay exact. Larger score sets are quantized to
a bounded set based on `acceptableError`. The held-out test still compares the
decoded number with the original reference score, so quantization consumes the
same error budget and cannot bypass the accuracy gate.

## Effect

Effect is optional and lives in a separate import:

```ts
import { classifyWithReference } from "@swapai/core/effect";

const program = classifyWithReference(relevance, input, callReferenceEffect);
```

## Runtime

Needle runs locally. On first use SwapAI creates its private runtime under
`.swapai`, installs the pinned training package, and downloads Needle's model
files. No Python setup is part of the TypeScript API. Set `dataDirectory` to
store this state elsewhere.

Full documentation: [stuartmccamley.com/swapai](https://stuartmccamley.com/swapai)
