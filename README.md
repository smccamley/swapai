# SwapAI

SwapAI collects answers from an existing classifier, trains a small Needle model
elsewhere, and swaps only after the candidate passes protected tests.

```sh
npm install @swapai/core effect
```

## Replace a classifier

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
  }),
});

const score = await relevance.classify(input, {
  documentFamily: "invoice",
  sourceKind: "gmail",
});
```

That is the application interface. Before promotion, `classify` returns the
Reference Classifier's answer and records it. After promotion, it uses the
Needle model and periodically checks it against the Reference Classifier.

Training is manual and never starts inside `classify`:

```ts
const dataset = relevance.inspect();
if (dataset.readyForTraining) {
  const result = await relevance.requestTraining();
  console.log(result.status); // promoted or rejected
}
```

`inspect()` explains missing result ranges and protected-test coverage. A high
total is not treated as sufficient evidence.

## What SwapAI protects

Every unique input is assigned deterministically to one purpose:

- 65% training
- 15% validation
- 10% representative test
- 10% coverage test

Numeric results are automatically split into ordinary ranges plus smaller
ranges around declared decision boundaries. Boolean and string results use one
bin per allowed value. Optional facets preserve useful categories such as
document family and source kind.

When storage reaches `maxTrainingSet`, SwapAI evicts old examples from the
fullest result/purpose/facet cell. Repeated inputs replace their earlier copy.
This preserves rare examples instead of retaining only the latest traffic.

`requestTraining()` freezes an immutable Dataset Revision. The Training
Provider receives only training and validation examples. Representative and
coverage tests stay in the control plane. A candidate is promoted only when
its aggregate and per-result-bin errors all meet `acceptableError`.

## Runpod

The Runpod provider creates one Secure Pod, uploads one Dataset Revision, runs
the pinned trainer image, downloads the candidate, and deletes that exact Pod.
It verifies deletion on success and failure.

```ts
runpodTrainer({
  apiKey: process.env.RUNPOD_API_KEY!,
  maximumCostUsd: 1,
  maximumRuntimeMinutes: 30,
  // Optional. Defaults to ~/.ssh/id_ed25519, then ~/.ssh/id_rsa.
  sshPrivateKey: process.env.SWAPAI_RUNPOD_SSH_PRIVATE_KEY,
});
```

The provider rejects a Pod whose hourly price could exceed the declared cost
ceiling. Pod names contain their hard deadline; later runs remove expired
SwapAI Pods before allocating another. SwapAI never keeps a Pod intentionally.

The default image is
`ghcr.io/smccamley/swapai-trainer:0.4.2`. The image is built from
[`trainer/Dockerfile`](trainer/Dockerfile) and pins `cactus-needle` 2.0.14 with
its NVIDIA/JAX training dependencies.

## Other training infrastructure

Training infrastructure implements one small interface:

```ts
import type { TrainingProvider } from "@swapai/core";

const awsTrainer: TrainingProvider = {
  name: "aws",
  async train(job) {
    // Upload job.examples, run the trainer, download model.cact.
    return { modelPath, needleVersion: "2.0.14", providerRunId };
  },
};
```

For a Mac, Windows machine, or a persistent worker, use `localTrainer()` or
implement the same interface. `localTrainer()` is explicit; SwapAI does not
silently train on the application host.

## Observe classifiers

```sh
npx @swapai/core classifiers-ui --data-directory .swapai
```

Open `http://127.0.0.1:4789`. The read-only viewer shows retained versus
observed examples, every dataset purpose, readiness gaps, trainer/provider,
cost, candidate status, protected-test error, and promoted model state. It does
not expose classification inputs, credentials, or model paths.

Applications can use the same read-only library:

```ts
import { readClassifierStatuses } from "@swapai/core/classifiers-ui";

const classifiers = readClassifierStatuses({ dataDirectory: ".swapai" });
```

## Erasure

```ts
relevance.clearTrainingData();
await relevance.flush();
```

This prevents further use of the model immediately, then securely deletes that
classifier's examples, Dataset Revisions, Training Runs, and generation state.
Other classifiers are untouched.

## Compatibility

The original `init()` interface remains available for existing applications.
New applications should use `createClassifier()`: it configures the Reference
Classifier once and disables in-process automatic training.

Needle's official fine-tuning guide documents LoRA training, GPU/Metal extras,
and the `.cact` artifact format:
[cactus-compute/needle](https://github.com/cactus-compute/needle/blob/main/doc/finetuning.md).
Runpod documents Pods and API keys at
[docs.runpod.io](https://docs.runpod.io/).
