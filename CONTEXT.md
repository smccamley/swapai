# SwapAI

SwapAI learns to reproduce an existing classifier with a locally-run model, then checks that replacement against the existing classifier while it is in use.

## Language

**Classifier Name**:
The stable name that owns one classifier's examples, trained model, counters, and health.
_Avoid_: Namespace, key

**Reference Classifier**:
The existing classifier whose recorded results are the answers SwapAI learns and checks against.
_Avoid_: Old classifier, fallback model, ground truth

**Candidate Classifier**:
The locally-run model being trained to reproduce the Reference Classifier.
_Avoid_: Free model, replacement AI

**Classification Example**:
One input paired with the Reference Classifier's result.
_Avoid_: Event, log row, training event

**Allowed Results**:
The complete result shape declared for one classifier: a bounded number, a boolean, or an explicit list of strings.
_Avoid_: Output schema, labels

**Held-Out Examples**:
Classification Examples permanently reserved for measuring the Candidate Classifier and never used to train it.
_Avoid_: Validation data, test events

**Acceptable Error**:
The highest average error the Candidate Classifier may produce across Held-Out Examples before it can be used in place of the Reference Classifier. Numeric results use mean absolute difference; boolean and string results use the fraction of incorrect answers.
_Avoid_: Threshold, accuracy

**Trained**:
A Candidate Classifier whose error across Held-Out Examples is no greater than its Acceptable Error.
_Avoid_: Ready, finished

**Retest**:
A scheduled comparison where both classifiers process the same live input after the Candidate Classifier becomes Trained.
_Avoid_: Health check, validation run
