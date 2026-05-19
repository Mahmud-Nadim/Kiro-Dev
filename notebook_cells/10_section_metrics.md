---

## Section 10 — Evaluation Metrics

We define five metrics. The paper's main table reports all five for each method.

| Metric | What it tells reviewers |
|---|---|
| **Top-1 Accuracy** | Did the model pick exactly the gold candidate? |
| **CPS (Composite Pragmatic Score)** | Soft score, mean over 6 axes (each in [0,1]) |
| **Axis-Accuracy@k** | Per-axis hit rate — exposes which axes are hardest |
| **Honorific Register Accuracy (HRA)** | Specific test: did the model pick the right pronoun-class (apni/tumi/tui)? |
| **Capability Tax** | Drop on a held-out non-pragmatic Bengali task — alignment shouldn't break general competence |

A method is interesting if it improves CPS AND maintains capability. We report both.
