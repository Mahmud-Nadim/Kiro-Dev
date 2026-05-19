---

## Section 12 — Ablations

Reviewers will demand ablations. We pre-compute every reasonable one and report. The 7 ablations:

1. **MA-DPO − relational graph conditioning** (drop the graph from the prompt; pure axis-decomposed loss)
2. **MA-DPO − learned axis weights** (uniform alphas)
3. **MA-DPO with random axis pairs** (sanity baseline)
4. **6× single-axis DPO ensemble** (does jointness matter?)
5. **MA-DPO with high-IAA examples only** (annotation-quality robustness)
6. **MA-DPO at half data** (data efficiency)
7. **MA-DPO with smaller LoRA rank (r=4)** (parameter efficiency)

Each ablation produces a row in Table 3. To save Colab compute, ablations 1, 2, 3 use one epoch and the same train/test split; ablations 4–7 are flagged as `RUN_FULL_ABLATIONS = False` by default — set to `True` to run.
