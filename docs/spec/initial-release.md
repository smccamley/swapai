# SwapAI initial release

## Purpose

`@swapai/core` records results from an existing classifier, trains a local
Needle 2 replacement, proves it against held-out examples, and only then uses
it in production.

`input` is the complete input sent to the Reference Classifier, including the
classification instruction and the text to classify. A raw record without the
task instruction is not an equivalent Classification Example.

## Public API

```ts
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

relevance.isTrained();
relevance.logClassification(input, result);
await relevance.classify(input);
await relevance.classify(input, referenceClassifier);
await relevance.flush();
await relevance.close();
```

`init`, `isTrained`, and `logClassification` are synchronous.
`logClassification` validates immediately and queues the SQLite write. `flush`
waits for queued writes and background work. `classify` is the only operation
that callers normally need to await.

## Result types

Every classifier declares exactly one result type:

- a number with a finite minimum and maximum;
- a boolean; or
- a string from a complete, closed list of allowed values.

Numeric error is `abs(reference - candidate) / (max - min)`. Boolean and string
error is `0` for a match and `1` for a mismatch. `acceptableError: "10%"` and
`acceptableError: 0.1` mean the same thing.

Needle is trained as a classifier. SwapAI maps numeric reference scores to a
small, closed set of internal labels and stores the label-to-number map with the
model. Small score sets stay exact. Larger sets are quantized using the declared
range and `acceptableError`, with a fixed maximum label count. Held-out error is
still calculated from the original reference score and decoded local score, so
the model is rejected when classification plus quantization exceeds the error
limit.

## Training

- SQLite stores examples, model generations, counters, and consecutive retest
  failure counts.
- A stable hash of classifier name and input assigns each input to the 80%
  training set or 20% held-out set. Repeated inputs can never cross sets.
- Every `retrainOnCount` newly logged examples starts a training attempt until
  `maxTrainingSet` examples have been retained.
- The newest examples replace the oldest examples above `maxTrainingSet`.
- TypeScript owns storage, scheduling, evaluation, and promotion.
- A managed Python worker only trains Needle and runs exported `.cact` models.
- The exported candidate model is tested against the held-out set. It becomes
  active only when its average error is at or below `acceptableError`.
- State survives process restarts.

## Classification and retesting

`classify(input)` uses the active local model. It fails with `not_trained` if no
model has passed evaluation.

`classify(input, referenceClassifier)` manages the replacement:

- before training, it calls the reference classifier, logs the result, and
  returns it;
- after training, it normally returns the local result;
- every `retestInterval` local classifications, it calls both classifiers,
  records the reference result, and returns the reference result;
- a retest fails when that single local result exceeds `acceptableError`;
- `retestRevertOn` consecutive failures disable the local model, archive the
  prior examples and model, create an empty generation, and resume the
  reference classifier;
- if the local worker fails, it returns the reference result.

With manual `classify(input)`, a due retest remains due until a later managed
call supplies a reference classifier.

## Effect adapter

`@swapai/core/effect` exports native Effect wrappers for local classification
and managed classification. Effect is an optional peer dependency; the main
package does not require it.

## Errors

Public errors have one of these codes:

- `invalid_configuration`
- `invalid_result`
- `not_trained`
- `service_unavailable`
- `classification_failed`
- `storage_failed`

Fire-and-forget failures are sent to `onBackgroundError` when configured.

## First use

SwapAI manages its Python environment inside `dataDirectory`, installs the
pinned `cactus-needle[train]` version, and disables Needle telemetry. Python is
an implementation detail; callers only use the TypeScript API.

## Acceptance tests

- Public API tests cover numbers, booleans, and closed string lists.
- Persistence and counters survive a restart using only the public API.
- Retest failures archive the old generation and start empty.
- Effect tests use actual Effect programs.
- A slow real test trains, exports, loads, and classifies with a `.cact` model.
