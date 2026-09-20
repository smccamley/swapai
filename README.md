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
    registerSshPublicKeyForTraining: true,
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
    const run = relevance
      .inspect()
      .trainingRuns.find(({ id }) => id === request.trainingRunId);

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
Readiness always requires at least one validation, representative-test, and
coverage-test example, even when configurable per-bin minima are zero;
`requestTraining()` returns `not_ready` before contacting a provider when any
protected suite is empty.
Numeric results are binned around declared decision boundaries. Boolean and
string results use one bin per allowed value. `inspect()` reports result-bin
coverage, declared-facet groups (including unlabelled examples), readiness
deficits, the newest 100 training runs, provider resources, cleanup state,
protected metrics, cost, artifacts, and shadow evidence.

On the first configured open of never-attempted legacy data, SwapAI repairs
the earlier 0.5 two-purpose adoption and assigns all four purposes
deterministically so existing observations can satisfy modern readiness. If
any local attempt, provider run, model artifact, or earlier generation exists,
legacy trainer-visible rows stay non-protected and cannot become validation or
test evidence.

### Explicit legacy held-out migration

SwapAI 0.5 counted a local attempt before Needle produced a candidate. For an
attempted legacy classifier, first stop every process using the classifier and
inspect a one-time migration plan. Targets are explicit; SwapAI does not invent
a statistically meaningful split:

```ts
import {
  inspectLegacyHeldOutMigration,
  migrateLegacyHeldOutExamples,
} from "@swapai/core";

const options = {
  dataDirectory: ".swapai",
  classifierName: "accountant-relevance",
  targetExamplesByPurpose: {
    validation: 135,
    representative_test: 100,
    coverage_test: 35,
  },
} as const;

const plan = inspectLegacyHeldOutMigration(options);
if (plan.status !== "ready") throw new Error(JSON.stringify(plan.blockers));

const migration = migrateLegacyHeldOutExamples({
  ...options,
  expectedPlanSha256: plan.planSha256,
  attestation: {
    heldOutExamplesWereNeverUsedForModelSelection: true,
    operator: "deployment-owner",
    reason: "Verified no candidate model or evaluation was produced",
  },
});
```

The command uses the durable legacy-adoption timestamp to select only original
pre-adoption `held_out` validation rows. Targets must total that eligible count,
not newer protected data. It holds an exclusive database transaction, rechecks
the reviewed plan hash, preserves every `training` row, and leaves every newer
validation, representative, and coverage row unchanged. A missing timestamp or
a held-out row exactly on its boundary is ambiguous and blocks the migration.
It also blocks active runtimes, completed evaluations, provider runs, trained
or previous generations, indexed artifacts, and candidate model, LoRA, or
sidecar files. A failed attempt's `training.jsonl` and shared base checkpoint do
not prove held-out evaluation. The plan validates configured minimums and
allocates deterministically across result-bin and facet groups. The operator,
reason, targets, counts, and plan digest are stored durably; repeating the
command returns `already_migrated`.

When `maxTrainingSet` is reached, retention balances result bin, purpose, and
facet groups rather than keeping only recent majority traffic.

## Runpod

`runpodTrainer()` uses Runpod REST API v2 and Secure Cloud only. It accepts the
API's proxy or direct SSH connection, requires a CUDA 12.8-or-newer host, and
creates one Pod after documented per-GPU placement retries. HTTP requests and
child processes are abortable; terminal Pod states fail immediately. Recovery
walks every API page, records the Pod ID immediately, and verifies termination
on success and failure. `reconcileTraining()` checks durable running records
after a controller restart and terminates expired Pods. The paid training
deadline includes both reported compute and conservatively priced container
storage, and reserves five minutes for verified deletion inside the earlier of
the runtime and cost limits.

The default image is
`ghcr.io/smccamley/swapai-trainer:0.6.5`. The image pins
`cactus-needle[train,gpu]` 2.0.14. Numeric artifacts report
`2.0.14/number-buckets-v1`.

```ts
await relevance.reconcileTraining();
```

Runpod authentication uses `RUNPOD_API_KEY`. SSH uses the configured private
key. Set `registerSshPublicKeyForTraining: true` when the key is not already on
the Runpod account. SwapAI then adds that exact public key before Pod creation,
enables Runpod's `startSsh` path, preserves every existing account key, and
removes only the key it added after verified Pod cleanup. Registration failure
happens before a paid Pod exists. File transfers force the legacy SCP protocol
because Runpod's proxy SSH endpoint does not provide the SFTP subsystem used by
modern `scp` by default.

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
