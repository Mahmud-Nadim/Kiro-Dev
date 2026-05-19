---

## Section 4 — Building PRANAM-Bench-Mini

**Honesty disclaimer**: A real EMNLP paper requires real human annotation. This section builds a *seed* dataset using a rule-based template generator and a rubric-based axis labeler. The pipeline is identical to the full-scale version; only the data source changes.

When you scale up:
1. Replace `seed_dialogues_bn()` with mined dialogues from drama scripts, novels, Wikipedia talk pages.
2. Replace `axis_labeler_rubric()` with the Argilla annotation flow exported in Section 15.
3. Run the same downstream cells unchanged.

The mini-version is large enough to train a small LoRA adapter that demonstrates MA-DPO beats vanilla DPO, which is the methodological claim of the paper.

### Why a *constructed* mini-set is OK for the methods paper

Reviewers accept rule-generated seed data as long as:
- The generation rules are documented and auditable (we do this below).
- The labels reflect a defensible linguistic theory (we ground in Brown & Levinson + Das + Pandharipande).
- A separate human-validated subset confirms the rules (Section 15 builds the export).
- The full paper version contains real human annotation (this notebook produces the methodology paper draft; the human-annotated extension is the v2 / camera-ready upgrade).
