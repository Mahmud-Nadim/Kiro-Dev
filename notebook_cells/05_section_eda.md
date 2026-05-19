---

## Section 5 — Exploratory Data Analysis

We need three things from EDA before model training:

1. **Coverage**: are all 6 axes represented across non-trivial value ranges?
2. **Class balance**: gold answers should not all cluster on one axis configuration (otherwise the model can solve the task by ignoring the relationship graph).
3. **Hard-negatives ratio**: each example should have at least one *near-correct* distractor (axis distance ≤ 2) and at least one *very wrong* distractor (axis distance ≥ 4). This is what makes preference learning informative.

These plots end up as Figure 2 / Appendix A in the paper.
