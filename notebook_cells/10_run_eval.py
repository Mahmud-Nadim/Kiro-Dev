# =============================================================================
# Cell: Build the main results table.
# Why: The Section 6 / Table 1 of the paper.
# =============================================================================

def render_main_table() -> pd.DataFrame:
    rows = []
    method_keys = [
        ("zero_shot", "Zero-shot (base)"),
        ("sft", "SFT"),
        ("dpo", "Vanilla DPO"),
        ("ma_dpo", "MA-DPO (ours)"),
    ]
    for key, label in method_keys:
        if key not in results_bag:
            continue
        r = results_bag[key]["bn"]
        row = {
            "Method": label,
            "Top-1 Acc": r.get("top1_accuracy", float("nan")),
            "CPS": r.get("cps", float("nan")),
            "HRA": r.get("honorific_register_accuracy", float("nan")),
        }
        # Per-axis breakdown.
        for ax in CFG.axis_names:
            row[f"Acc-{ax}"] = r["axis_scores"].get(ax, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


main_df = render_main_table()
print("\n=== Table 1: Main Results (Bengali test) ===")
print(main_df.to_string(index=False, float_format="%.3f"))

main_df.to_csv(Path(CFG.tables_dir, "table1_main_results.csv"), index=False)
print(f"\nPAPER_ARTIFACT: {CFG.tables_dir}/table1_main_results.csv")
