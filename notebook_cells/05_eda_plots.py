# =============================================================================
# Cell: EDA plots.
# Why: Visual evidence the dataset has axis diversity.
# =============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = sns.color_palette("colorblind")

# Figure: distribution of expected (gold) tensor values across the 4 ordinal-ish axes.
fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
axis_titles = ["Power", "Age", "Intimacy", "Formality"]
axis_keys = ["power", "age", "intimacy", "formality"]

for ax_obj, title, key in zip(axes, axis_titles, axis_keys):
    vals = []
    for e in splits["bn_train"] + splits["bn_dev"] + splits["bn_test"]:
        gold = e["candidates"][e["gold_index"]]["tensor"]
        vals.append(gold[key])
    ax_obj.hist(vals, bins=range(min(vals), max(vals) + 2),
                color=PALETTE[0], edgecolor="black", alpha=0.85)
    ax_obj.set_title(f"Gold {title}")
    ax_obj.set_xlabel(key)
    ax_obj.set_ylabel("count")

plt.tight_layout()
fig.savefig(Path(CFG.figures_dir, "fig2_axis_distributions.pdf"), dpi=300)
fig.savefig(Path(CFG.figures_dir, "fig2_axis_distributions.png"), dpi=300)
plt.show()
print(f"\nPAPER_ARTIFACT: {CFG.figures_dir}/fig2_axis_distributions.{{pdf,png}}")

# Figure: hard-negative coverage histogram.
fig, ax = plt.subplots(figsize=(7, 4))
all_distances = []
for e in splits["bn_train"]:
    gold_t = PragmaticTensor(**e["candidates"][e["gold_index"]]["tensor"])
    for j, c in enumerate(e["candidates"]):
        if j == e["gold_index"]:
            continue
        all_distances.append(axis_distance(PragmaticTensor(**c["tensor"]), gold_t))

ax.hist(all_distances, bins=range(0, int(max(all_distances)) + 2),
        color=PALETTE[2], edgecolor="black", alpha=0.85)
ax.set_xlabel("L1 axis-distance from gold")
ax.set_ylabel("count")
ax.set_title("Distractor difficulty distribution (Bengali train)")
plt.tight_layout()
fig.savefig(Path(CFG.figures_dir, "fig3_distractor_distance.pdf"), dpi=300)
plt.show()
print(f"PAPER_ARTIFACT: {CFG.figures_dir}/fig3_distractor_distance.pdf")
