# SwapAI

SwapAI learns a cheaper classifier from an existing classifier, trains outside
the application process, and promotes only candidates supported by protected
test evidence.

## Language

**Classifier**:
The named unit that owns one result declaration, examples, Dataset Revisions,
Training Runs, Model Artifacts, and health.
_Avoid_: Namespace, AI instance

**Reference Classifier**:
The existing classifier whose answers SwapAI returns and records until a
Candidate Classifier is promoted.
_Avoid_: Ground truth, old model

**Classification Example**:
One unique input, Reference Classifier result, result bin, purpose, and declared
facets. Repeated inputs replace their retained copy while increasing observed
traffic.
_Avoid_: Log row, event

**Result Bin**:
One automatically calculated part of the allowed result space. Numeric bins are
narrower around declared Decision Boundaries.
_Avoid_: Bucket without a declared meaning

**Classification Facet**:
A declared low-cardinality category used to preserve dataset variety, such as
document family or source kind.
_Avoid_: Arbitrary metadata, tag

**Dataset Purpose**:
The deterministic, permanent role of a Classification Example: training,
validation, representative test, or coverage test.
_Avoid_: Random split

**Dataset Readiness**:
The exact training, validation, representative-test, and per-bin coverage
requirements still missing. A total example count is not readiness.
_Avoid_: Enough data

**Dataset Revision**:
An immutable snapshot of all examples used to train and test one candidate.
Only its training and validation examples cross the Training Provider boundary.
_Avoid_: Current dataset, mutable batch

**Training Provider**:
The infrastructure adapter that accepts one Training Job and returns one
Needle Model Artifact. Runpod, local, AWS, and owned machines are providers.
_Avoid_: Trainer implementation, cloud abstraction

**Trainer**:
The pinned program inside a training machine that converts training data into a
Needle Model Artifact. It does not create or delete infrastructure and does not
see protected tests.
_Avoid_: Provider, worker lifecycle

**Training Run**:
One attempt by one Training Provider to train one Dataset Revision, including
its provider run ID, status, estimated cost, failure, and candidate artifact.
_Avoid_: Background task

**Candidate Classifier**:
A Needle model returned by a Training Run but not yet permitted to replace the
Reference Classifier.
_Avoid_: Trained model

**Promotion Evidence**:
The recorded aggregate and per-result-bin errors from both protected test
suites. Every recorded slice must meet Acceptable Error.
_Avoid_: MAE, accuracy number

**Promoted Classifier**:
A Candidate Classifier whose Promotion Evidence passed and whose Dataset
Revision was still current at the promotion transaction.
_Avoid_: Trained, finished

**Model Artifact**:
An immutable, SHA-256-addressed `.cact` file plus required numeric label map.
_Avoid_: Model path, latest model

**Classifier Observer**:
The read-only library and local page that show dataset coverage, Training Runs,
Promotion Evidence, cost, and active model state without exposing inputs,
credentials, or artifact paths.
_Avoid_: Admin writer, training UI

## Boundaries

1. `classify()` never trains. Training begins only through `requestTraining()`.
2. A Training Provider owns infrastructure lifecycle; the Trainer owns only
   Needle training.
3. Protected test examples never cross into a Trainer.
4. Promotion is a local atomic decision made from a frozen Dataset Revision.
5. Model Artifacts are immutable and content-addressed.
6. The Classifier Observer is read-only.
