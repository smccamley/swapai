# Training architecture

The public API is deliberately smaller than the implementation:

```text
application
  createClassifier → classify / inspect / requestTraining / promoteCandidate
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  example store    training provider   observer
  + revisions      (Runpod/local/…)    (read-only)
                         │
                         ▼
                  pinned trainer image
                         │
                         ▼
                  candidate .cact
                         │
                         ▼
             protected local evaluation
                         │
                         ▼
               unpromoted candidate
                         │
                         ▼
              fresh shadow evidence
                         │
                         ▼ explicit decision
               promoted classifier
```

## Why these seams remain separate

The application process cannot own Needle training because JAX compilation and
fine-tuning are the resource-heavy operations that caused the original host to
fail. `requestTraining()` may coordinate a run and evaluate inference, but the
Training Provider performs fine-tuning elsewhere.

The provider and trainer are separate because they change for different
reasons. Runpod/AWS/local code allocates, limits, transfers, and cleans up
compute. The trainer consumes a versioned job directory and creates a pinned
Needle artifact. Either side can change without teaching the other about its
internals.

Validation and protected tests remain outside the trainer so a training
implementation cannot accidentally tune against its promotion exam. Aggregate error alone is
insufficient: an imbalanced relevance dataset can make a constant low score
look accurate. Promotion therefore requires every protected suite and result
bin to pass. Readiness requires every protected suite to be non-empty, even
when configurable per-bin minima are zero, so a guaranteed rejection cannot
incur provider cost. The provider receives training examples only. A candidate that
passes frozen evidence remains non-authoritative until fresh reference-backed
shadow evidence passes and the caller explicitly promotes its run ID.

The trainer boundary is a versioned on-disk protocol. Input and output files
are SHA-256 verified on both sides. Local and Runpod providers execute the same
contract, so infrastructure choice cannot change what a trainer is allowed to
see or return.

The Runpod adapter is Secure-Cloud-only because Community Cloud cannot
guarantee the SSH route this protocol needs. It supports both Runpod proxy SSH
and direct SSH, forces legacy SCP transport because the proxy omits SFTP,
constrains placement to CUDA 12.8 or newer, aborts bounded HTTP
and child-process work, and follows every list page during orphan recovery.
Its paid deadline includes reported compute, conservative container-storage
cost, and deletion headroom. When temporary SSH-key registration is enabled,
the adapter registers the exact derived public key before Pod creation and
removes only that key as part of the same reported cleanup boundary.

Attempted legacy datasets use a separate two-phase administrative boundary.
Inspection uses the durable legacy-adoption timestamp to identify pre-adoption
held-out validation rows, then produces a SHA-256 plan over every current row,
the cutoff, and explicit operator-selected purpose totals. Newer protected rows
are excluded from the targets and remain unchanged. A missing cutoff or a row
exactly on it fails closed. Application requires the reviewed digest, a named
model-selection attestation, no live classifier runtime, and no durable or
filesystem candidate evidence. An exclusive SQLite transaction rechecks those
facts before changing only eligible legacy held-out purposes; training rows
remain untouched. The attestation and allocation are durable and idempotent.

The project uses one npm package with explicit subpath imports instead of four
packages. This keeps installation to one dependency while retaining the same
boundaries as separate libraries:

- `@swapai/core` — application API and local provider
- `@swapai/core/runpod` — Runpod lifecycle adapter
- `@swapai/core/classifiers-ui` — read-only observer
- `ghcr.io/smccamley/swapai-trainer` — isolated trainer image

A second independently versioned consumer is the point at which a subpath
should become a separate package. Splitting earlier would add release and
version-coordination work without creating a stronger boundary.
