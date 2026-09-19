# Training architecture

The public API is deliberately smaller than the implementation:

```text
application
  createClassifier → classify / inspect / requestTraining
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
               content-addressed artifact
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

Protected tests remain outside the trainer so a training implementation cannot
accidentally tune against its promotion exam. Aggregate error alone is
insufficient: an imbalanced relevance dataset can make a constant low score
look accurate. Promotion therefore requires every protected suite and result
bin to pass.

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
