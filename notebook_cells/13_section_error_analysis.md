---

## Section 13 — Error Analysis & Failure-Mode Taxonomy

Reviewers love a paper that shows scientific honesty by classifying its own failures. We build a taxonomy of MA-DPO's remaining errors. This becomes Section 8 + Figure 4 of the paper.

Failure categories we look for:
- **Over-formal collapse**: model defaults to apni regardless of context (most common LLM failure)
- **Wrong axis weighting**: correct on Power but wrong on Intimacy
- **Kinship mismatch**: failed to recognize blood-kin context
- **Deference target confusion**: applied honorific to wrong referent
- **Register flattening**: produced grammatically correct but contextually flat reply
