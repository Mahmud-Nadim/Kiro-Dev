---

## Section 7 — SFT Baseline

The simplest fine-tuning approach: supervised fine-tuning on the gold (correct) responses only. This is what most papers do as a "naive baseline" for alignment work. We expect SFT to improve top-1 over zero-shot but to *plateau* on CPS because it never sees the contrast between correct and incorrect register choices.

We use LoRA to keep the run feasible on T4 (~20 minutes).
