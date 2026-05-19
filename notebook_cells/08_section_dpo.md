---

## Section 8 — Vanilla DPO Baseline

DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) treats alignment as a binary preference learning problem. For each (prompt, chosen, rejected) triple, the model is trained to assign higher likelihood to `chosen` than `rejected`.

This is the baseline our MA-DPO method must beat. The key contrast:
- **Vanilla DPO**: one preference pair per example (gold vs random distractor)
- **MA-DPO**: six preference signals per example, one per pragmatic axis, with relational graph conditioning

We construct DPO training data by pairing the gold candidate against the *most distant* distractor for each example.
