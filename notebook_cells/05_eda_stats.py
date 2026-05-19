# =============================================================================
# Cell: EDA stats.
# Why: Reviewers ask for these. We dump as a CSV that goes into Appendix A.
# =============================================================================

def load_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


splits = {
    "bn_train": load_split(paths["bn_train"]),
    "bn_dev":   load_split(paths["bn_dev"]),
    "bn_test":  load_split(paths["bn_test"]),
    "hi_test":  load_split(paths["hi_test"]),
    "ko_test":  load_split(paths["ko_test"]),
}

stats_rows = []
for name, exs in splits.items():
    n = len(exs)
    n_cands = sum(len(e["candidates"]) for e in exs)
    avg_cands = n_cands / max(n, 1)

    # Distance stats.
    dists_to_gold = []
    for e in exs:
        gold_t = PragmaticTensor(**e["candidates"][e["gold_index"]]["tensor"])
        for j, c in enumerate(e["candidates"]):
            if j == e["gold_index"]:
                continue
            d = axis_distance(PragmaticTensor(**c["tensor"]), gold_t)
            dists_to_gold.append(d)

    near = sum(1 for d in dists_to_gold if d <= 2)
    far = sum(1 for d in dists_to_gold if d >= 4)

    # Axis coverage.
    axis_var = {ax: set() for ax in CFG.axis_names}
    for e in exs:
        for c in e["candidates"]:
            for ax in CFG.axis_names:
                axis_var[ax].add(c["tensor"][ax])

    stats_rows.append({
        "split": name,
        "n_examples": n,
        "avg_candidates": round(avg_cands, 2),
        "near_distractors": near,
        "far_distractors": far,
        "power_unique": len(axis_var["power"]),
        "intimacy_unique": len(axis_var["intimacy"]),
        "formality_unique": len(axis_var["formality"]),
        "kinship_unique": len(axis_var["kinship"]),
        "dt_unique": len(axis_var["deference_target"]),
    })

stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))

stats_df.to_csv(Path(CFG.tables_dir, "appendix_A_data_stats.csv"), index=False)
print(f"\nPAPER_ARTIFACT: {CFG.tables_dir}/appendix_A_data_stats.csv")
