# =============================================================================
# Cell: Error analysis on MA-DPO test predictions.
# Why: Section 8 of the paper. Reviewers love taxonomies.
# =============================================================================

def classify_failure(ex: dict, chosen_idx: int) -> str:
    """Return a category string for a wrong prediction."""
    if chosen_idx == ex["gold_index"]:
        return "correct"
    gold_t = PragmaticTensor(**ex["candidates"][ex["gold_index"]]["tensor"])
    cand_t = PragmaticTensor(**ex["candidates"][chosen_idx]["tensor"])

    # Over-formal collapse: chose apni-equivalent (power <= -1) when gold is mid/low.
    if gold_t.power >= 0 and cand_t.power <= -1:
        return "over_formal_collapse"

    # Under-formal: chose tui when gold is apni.
    if gold_t.power <= -1 and cand_t.power >= 1:
        return "under_formal"

    # Kinship mismatch.
    if gold_t.kinship != cand_t.kinship and gold_t.kinship != "none":
        return "kinship_mismatch"

    # Deference target confusion.
    if gold_t.deference_target != cand_t.deference_target:
        return "deference_target_confusion"

    # Power right but intimacy wrong.
    if gold_t.power == cand_t.power and gold_t.intimacy != cand_t.intimacy:
        return "intimacy_only_error"

    return "other"


def analyze_method(method_key: str, examples: list[dict]) -> pd.DataFrame:
    chosen = results_bag[method_key]["bn"]["chosen_indices"]
    cats = [classify_failure(e, c) for e, c in zip(examples, chosen)]
    counts = pd.Series(cats).value_counts()
    return counts


print("\n=== MA-DPO failure taxonomy (Bengali test) ===")
ma_errs = analyze_method("ma_dpo", splits["bn_test"])
print(ma_errs.to_string())

# Also print zero-shot baseline failures for contrast.
print("\n=== Zero-shot base failure taxonomy (for contrast) ===")
zs_errs = analyze_method("zero_shot", splits["bn_test"])
print(zs_errs.to_string())

# Save merged taxonomy as a CSV.
err_df = pd.DataFrame({"ma_dpo": ma_errs, "zero_shot": zs_errs}).fillna(0).astype(int)
err_df.to_csv(Path(CFG.tables_dir, "table4_failure_taxonomy.csv"))
print(f"\nPAPER_ARTIFACT: {CFG.tables_dir}/table4_failure_taxonomy.csv")

# Pull 5 hand-pickable failure cases for the paper appendix.
sample_failures = []
chosen = results_bag["ma_dpo"]["bn"]["chosen_indices"]
for ex, c in zip(splits["bn_test"], chosen):
    if c == ex["gold_index"]:
        continue
    cat = classify_failure(ex, c)
    sample_failures.append({
        "id": ex["id"],
        "category": cat,
        "context": ex["context_turns"],
        "gold": ex["candidates"][ex["gold_index"]]["text"],
        "predicted": ex["candidates"][c]["text"],
        "gold_tensor": ex["candidates"][ex["gold_index"]]["tensor"],
        "predicted_tensor": ex["candidates"][c]["tensor"],
    })
    if len(sample_failures) >= 8:
        break
Path(CFG.tables_dir, "appendix_C_failure_examples.json").write_text(
    json.dumps(sample_failures, indent=2, ensure_ascii=False)
)
print(f"PAPER_ARTIFACT: appendix C failure examples ({len(sample_failures)})")
