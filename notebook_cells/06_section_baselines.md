---

## Section 6 — Zero-Shot Baselines (Prompted LLMs)

Before training anything, we need to know how badly off-the-shelf instruction-tuned models do. This is the *hook* of the paper — Figure 1 will show GPT-4 / Llama-3 picking the wrong honorific register.

We score each candidate by computing P(candidate | context) under the base instruction-tuned model. The model "picks" the candidate with highest log-prob. Then we compare its choice to the gold index.

This section runs in ~10 minutes on a T4 and produces three numbers per model:
- **Top-1 accuracy** (chose the gold)
- **Composite Pragmatic Score (CPS)** — soft-axis score against gold
- **Honorific Register Accuracy** — did it pick the right pronoun-class (apni/tumi/tui)?
