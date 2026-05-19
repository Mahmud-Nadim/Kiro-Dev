---

## Section 11 — Cross-Lingual Transfer

This is where the paper goes from "interesting" to "exciting" for reviewers. The MA-DPO model was trained ONLY on Bengali. We now evaluate it zero-shot on Hindi and Korean test sets.

If the relational pragmatic tensor captures something *typological* rather than *lexical*, we should see meaningful transfer. The literature on cross-lingual alignment ([Consistency-based Multilingual Alignment](https://arxiv.org/abs/2509.08541)) suggests this is plausible but rarely demonstrated for sociopragmatic tasks specifically.

**Strong reviewer signal**: if MA-DPO's CPS on HI/KO test is meaningfully higher than the zero-shot base on the same data, our method is generalizing the pragmatic structure, not memorizing Bengali surface forms.
