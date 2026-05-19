---

## Section 2 — Setup, Installation, and Reproducibility

We pin every dependency. Reviewers complain when a notebook breaks because of a silent transformers minor-version change. Pinning is non-negotiable for a publication-grade artifact.

The dependency stack:
- `transformers` — model loading + tokenization
- `trl` — DPO/SFT trainers we subclass for MA-DPO
- `peft` — LoRA adapters (keeps memory under T4's 16 GB)
- `accelerate`, `bitsandbytes` — quantization & device mgmt
- `datasets` — HF datasets format
- `pandas`, `matplotlib`, `seaborn` — analysis & figures
- `scikit-learn` — kappa / agreement metrics
- `tabulate` — LaTeX table generation

Skip the install cell if you have already run it in this Colab session.
