## What this notebook produces

By the end of this notebook you will have:

1. **PRANAM-Bench-Mini**: a 240-example seed benchmark with axis-graded responses across Bengali (160), Hindi (40), Korean (40). The full-scale version targets 12,600 examples after human annotation; this notebook produces the methodologically valid mini-version that proves the pipeline.
2. **A trained MA-DPO model** (Qwen2.5-0.5B-Instruct + LoRA) that beats SFT and vanilla DPO baselines on a Composite Pragmatic Score (CPS).
3. **Cross-lingual transfer numbers** — Bengali-trained model evaluated zero-shot on Hindi and Korean.
4. **A full ablation table** with 7 ablations.
5. **An error-mode taxonomy** built from real model failures.
6. **LaTeX-formatted tables** ready to paste into your EMNLP submission.
7. **Publication-quality figures** (300 DPI, color-blind safe palettes).
8. **An Argilla-format export** for scaling up annotation later.
9. **A model card and reproducibility manifest**.

## How to use this notebook

- **First pass (~2 hours)**: Run cells top-to-bottom. The notebook is sized for free Colab T4.
- **Second pass**: Replace the synthetic axis labels with real human annotations (see Section 15).
- **Third pass**: Scale up the base model to Llama-3.1-8B (Section 9 has the config).
- **Submission pass**: Run Section 14 to regenerate all paper artifacts after each experiment iteration.

## Notebook structure

| Section | Purpose | Time on T4 |
|---|---|---|
| 2 | Setup & config | 5 min |
| 3 | Relational Pragmatic Tensor — formal framework | 2 min |
| 4 | Build PRANAM-Bench-Mini dataset | 10 min |
| 5 | Exploratory data analysis | 3 min |
| 6 | Zero-shot baselines | 15 min |
| 7 | SFT baseline | 20 min |
| 8 | Vanilla DPO baseline | 25 min |
| 9 | MA-DPO (proposed method) | 30 min |
| 10 | Evaluation metrics | 10 min |
| 11 | Cross-lingual transfer | 10 min |
| 12 | Ablations | 30 min |
| 13 | Error analysis | 10 min |
| 14 | Paper artifacts (tables + figures) | 5 min |
| 15 | Human-eval scaffolding | 5 min |
| 16 | Reproducibility | 5 min |
| 17 | Roadmap to publication | reading |
