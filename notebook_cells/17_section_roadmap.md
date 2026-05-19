---

## Section 17 — Roadmap to the EMNLP 2027 Publication

You have just produced a methodologically valid mini-version of the paper. The artifact inventory above is the *exact* set of files referenced from your manuscript. Here is the prioritized sequence to convert this notebook into an accepted paper.

### Pre-submission ranked actions (in order)

**1. Replace synthetic axis labels with real human annotation.**
- Spin up an Argilla instance.
- Upload `argilla_export_bn_test.json`.
- Recruit ≥ 3 native Bengali annotators per example.
- Compute Krippendorff's alpha — target ≥ 0.7 for ordinal axes, ≥ 0.8 for categorical.
- Re-run the entire notebook on the human-annotated split.

**2. Scale up base model.**
- Switch `CFG.base_model` to `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`.
- Use 8×A100 if available; otherwise BF16 + ZeRO-3.
- Expect ~5x absolute gain over the 0.5B model on CPS.

**3. Expand the dataset to 12,000 examples.**
- Mine Bengali drama scripts, novel dialogue, and Wikipedia talk pages.
- Use the same `RelationshipGraph` schema.
- Stratify by region, formality setting, and kinship category.

**4. Add a downstream task.**
- Pick a customer-service or healthcare dialogue dataset in Bengali (BanglaCHQ-Summ etc.).
- Show MA-DPO model has higher *acceptability* on real-world tasks.
- This becomes Section 9 of the paper and addresses the "does this matter for downstream tasks?" reviewer question.

**5. Cross-lingual scale.**
- Add Tamil, Marathi, Japanese to the test-only set.
- Show MA-DPO transfers more broadly than reviewers expect.

**6. Run human evaluation.**
- 200 test items × 5 raters per language for {Bengali, Hindi, Korean}.
- Pre-registered protocol on AnonGitHub or OSF.
- Pay rate documented in Ethics.

**7. Workshop preprint.**
- Submit a 4-page version to BLP-2026 or BLP-2027 (whenever the next workshop runs).
- Get reviewer familiarity. Cite this preprint in the EMNLP submission.

**8. Adversarial internal review.**
- Three lab-mates do "review-as-if-EMNLP" on the draft.
- Address every concern in the appendix or rebuttal-buffer.

**9. ARR submission with anonymized everything.**
- Strip GitHub URLs, university names, model org names.
- Upload supplementary: anonymized dataset sample (200 examples), code, model card stub.
- Primary area: "Multilinguality and Linguistic Diversity"; secondary: "Resources and Evaluation".

**10. Rebuttal cycle.**
- Pre-write rebuttals for the top 8 likely concerns (already in the prose plan).
- Run any reviewer-requested experiment within the 1-week window.
- Final camera-ready: add the experiments + acknowledgments + de-anonymize.

### Final EMNLP 2027 submission checklist

- [ ] Long paper (8 pages + unlimited references)
- [ ] Anonymized
- [ ] Supplementary: code + sample data
- [ ] Reproducibility checklist
- [ ] Ethics statement (annotator pay, dual-use)
- [ ] Limitations section (we listed candidates above)
- [ ] LaTeX tables 1–4 referenced in the body
- [ ] Figures 1–5 with informative captions
- [ ] Cross-lingual transfer section
- [ ] Failure-mode taxonomy
- [ ] Model card on HuggingFace under OpenRAIL-M
- [ ] Public dataset under CC-BY 4.0

### Where to find every paper artifact

| Paper element | File in this notebook's workdir |
|---|---|
| Table 1 (main results) | `tables/table1_main_results.tex` |
| Table 2 (cross-lingual) | `tables/table2_crosslingual.tex` |
| Table 3 (ablations) | `tables/table3_ablations.tex` |
| Table 4 (failures) | `tables/table4_failure_taxonomy.tex` |
| Figure 1 (motivating) | `figures/fig1_motivating_example.pdf` |
| Figure 2 (axis distributions) | `figures/fig2_axis_distributions.pdf` |
| Figure 3 (distractor difficulty) | `figures/fig3_distractor_distance.pdf` |
| Figure 4 (failure modes) | `figures/fig4_failure_taxonomy.pdf` |
| Figure 5 (cross-lingual heatmap) | `figures/fig5_crosslingual_heatmap.pdf` |
| Appendix A (data stats) | `tables/appendix_A_data_stats.csv` |
| Appendix C (failure examples) | `tables/appendix_C_failure_examples.json` |
| Appendix D (reproducibility) | `reproducibility_manifest.json` |
| Released artifacts | `models/madpo_lora/` and `models/MODEL_CARD.md` |

You now have a complete, runnable, paper-aligned pipeline. The only work between this notebook and an EMNLP 2027 acceptance is **annotation labor + scale + human evaluation** — none of which require further methodological invention.

Good luck.
