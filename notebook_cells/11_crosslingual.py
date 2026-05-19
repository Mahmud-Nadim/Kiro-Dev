# =============================================================================
# Cell: Cross-lingual zero-shot evaluation.
# Why: Section 7 of the paper. Strong reviewer signal.
# =============================================================================

def evaluate_method_xlingual(method_name: str, model, splits_dict: dict) -> dict:
    out = {}
    for lang in ["hi", "ko"]:
        key = f"{lang}_test"
        if key not in splits_dict:
            continue
        res = evaluate_zero_shot(splits_dict[key], model, tokenizer)
        chosen = res["chosen_indices"]
        res["honorific_register_accuracy"] = metric_honorific_register_accuracy(
            splits_dict[key], chosen
        )
        out[lang] = res
        print(f"  {method_name} on {lang.upper()}: top1={res['top1_accuracy']:.3f}  "
              f"CPS={res['cps']:.3f}  HRA={res['honorific_register_accuracy']:.3f}")
    return out


print("Cross-lingual evaluation:")
print("\nBase (zero-shot):")
xl_zero = evaluate_method_xlingual("base", base_model, splits)
results_bag["zero_shot"].update(xl_zero)

print("\nMA-DPO (Bengali-trained, zero-shot transfer):")
xl_ma = evaluate_method_xlingual("ma_dpo", ma_model, splits)
results_bag["ma_dpo"].update(xl_ma)

# Build cross-lingual results table.
xl_rows = []
for method_key, label in [("zero_shot", "Base"), ("ma_dpo", "MA-DPO (ours, BN-trained)")]:
    for lang in ["bn", "hi", "ko"]:
        if lang not in results_bag[method_key]:
            continue
        r = results_bag[method_key][lang]
        xl_rows.append({
            "Method": label,
            "Lang": lang.upper(),
            "Top-1": r.get("top1_accuracy", float("nan")),
            "CPS": r.get("cps", float("nan")),
            "HRA": r.get("honorific_register_accuracy", float("nan")),
        })

xl_df = pd.DataFrame(xl_rows)
print("\n=== Table 2: Cross-lingual transfer ===")
print(xl_df.to_string(index=False, float_format="%.3f"))
xl_df.to_csv(Path(CFG.tables_dir, "table2_crosslingual.csv"), index=False)
Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
print(f"\nPAPER_ARTIFACT: {CFG.tables_dir}/table2_crosslingual.csv")
