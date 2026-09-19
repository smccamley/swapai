# SwapAI

SwapAI collects answers from an existing classifier, trains a small Needle 2
model outside the application, evaluates it locally, and keeps the reference
classifier authoritative until you explicitly promote a candidate.

```sh
npm install @swapai/core effect
```

The canonical guide is [stuartmccamley.com/swapai](https://stuartmccamley.com/swapai).

## Quick start

```ts
import { createClassifier } from "@swapai/core";
import { runpodTrainer } from "@swapai/core/runpod";

const relevance = createClassifier({
  name: "accountant-relevance",
  result: { type: "number", min: 0, max: 1 },
  decisionBoundaries: [0.5],
  facets: ["documentFamily", "sourceKind"] as const,
  reference: async (input) => (await classifyWithGrok(input)).score,
  training: runpodTrainer({
    apiKey: process.env.RUNPOD_API_KEY!,
    maximumCostUsd: 1,
    maximumRuntimeMinutes: 30,
    sshPrivateKey: process.env.SWAPAI_RUNPOD_SSH_PRIVATE_KEY,
  }),
});

const score = await relevance.classify(input, {
  documentFamily: "invoice",
  sourceKind: "gmail",
});
```

Before promotion, `classify()` always returns the reference answer and records
it. It never starts training.

## Train, review, promote

```ts
const inspection = relevance.inspect();
if (inspection.readyForTraining) {
  const request = await relevance.requestTraining();

  if (request.status === "candidate") {
    // After normal traffic has called classify(), wait for its shadow writes.
    await relevance.flush();
    const run = relevance.inspect().trainingRuns.find(
      ({ id }) => id === request.trainingRunId,
    );

    if (run?.shadow?.passed) {
      await relevance.promoteCandidate(request.trainingRunId);
    }
  }
}
```

`requestTraining()` returns `not_ready`, `candidate`, `rejected`,
`already_running`, `already_promoted`, or `already_failed`. Every terminal
outcome is deduplicated for the same immutable dataset revision and provider,
so workflow retries cannot create a second paid run. Spending on an intentional
retry is explicit:

```ts
await relevance.retryTraining(failedOrRejectedRunId);
```

A passing protected evaluation creates an unpromoted candidate. Fresh traffic
is then evaluated in shadow mode: the reference answer remains authoritative,
candidate failures never affect classification, and aggregate shadow error is
persisted. `promoteCandidate()` requires both passing protected evidence and
passing fresh shadow evidence.

## Dataset protection

Each unique input has a stable purpose based only on classifier name and input:

- training: sent to the selected provider;
- validation: retained locally for candidate evaluation;
- representative test: retained locally;
- coverage test: retained locally.

Providers receive only training examples. All three evaluation sets remain
local, so a provider cannot inspect or train against its acceptance exam.
Numeric results are binned around declared decision boundaries. Boolean and
string results use one bin per allowed value. `inspect()` reports result-bin
coverage, declared-facet groups (including unlabelled examples), readiness
deficits, complete run history, provider resources, cleanup state, protected
metrics, cost, artifacts, and shadow evidence.

When `maxTrainingSet` is reached, retention balances result bin, purpose, and
facet groups rather than keeping only recent majority traffic.

## Runpod

`runpodTrainer()` uses Runpod REST API v2. It creates one Secure Pod by default,
enforces the declared time and cost ceilings, records the Pod ID immediately,
and verifies termination on success and failure. `reconcileTraining()` checks
durable running records after a controller restart and terminates expired Pods.
The paid training deadline reserves five minutes for verified deletion inside
the earlier of the runtime and cost limits.

The default image is
`ghcr.io/smccamley/swapai-trainer:0.5.0`. The image pins
`cactus-needle[train,gpu]` 2.0.14. Numeric artifacts report
`2.0.14/number-buckets-v1`.

```ts
await relevance.reconcileTraining();
```

Runpod authentication uses `RUNPOD_API_KEY`. SSH uses the configured private
key; the corresponding public key is injected only into the temporary Pod.

## Local and custom providers

`localTrainer()` uses the same version-2 runner contract as Runpod: SHA-256
verified training input, result manifest, model, and numeric sidecar. Training
bundles are deleted after every completed or failed request.

Custom providers implement `TrainingProvider`. Use its lifecycle reporter to
persist the provider run ID, resources, and cleanup result immediately. Supply
`reconcile()` and `cancel()` when the provider owns external resources.

## Observe

```sh
npx @swapai/core classifiers-ui --data-directory .swapai
```

Or use the read-only API:

```ts
import { readClassifierStatuses } from "@swapai/core/classifiers-ui";

const statuses = readClassifierStatuses({ dataDirectory: ".swapai" });
```

## Erase

```ts
await relevance.erase();
```

`erase()` first cancels and verifies provider cleanup, then removes examples,
dataset revisions, run rows, transient bundles, complete content-addressed
artifacts, and numeric sidecars. It refuses to certify erasure while an
external resource is still live or cleanup failed.

The legacy `init()` interface remains available for applications that supply a
reference callback per request. It also supports `erase()` and persisted shadow
evaluation. New integrations should normally use `createClassifier()`.

## Effect

`@swapai/core/effect` exposes typed Effects for the full configured lifecycle:
create, classify, log, inspect, request or explicitly retry training, promote,
reconcile, erase, flush, and close. The Promise interface remains available
from `@swapai/core`.

Needle source and fine-tuning details are documented by
[cactus-compute/needle](https://github.com/cactus-compute/needle). Runpod v2 is
documented at [docs.runpod.io](https://docs.runpod.io/api-reference-v2/overview).
