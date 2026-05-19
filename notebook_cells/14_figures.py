# =============================================================================
# Cell: Generate publication figures.
# Why: 300 DPI, color-blind safe, vector PDFs for camera-ready.
# =============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- Figure 1: motivating example ---------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
scenarios = json.loads(Path(CFG.data_dir, "figure1_scenarios.json").read_text())
y_labels = [s["label"] for s in scenarios]
x_axes = ["Power", "Age", "Intimacy", "Formality"]
data = np.array([
    [s["tensor"]["power"], s["tensor"]["age"], s["tensor"]["intimacy"],
     s["tensor"]["formality"]] for s in scenarios
])
sns.heatmap(data, annot=True, cmap="vlag", center=0,
            xticklabels=x_axes, yticklabels=y_labels, cbar_kws={"label": "axis value"}, ax=ax)
ax.set_title("Figure 1: Diverse relationship configurations require diverse honorific responses")
plt.tight_layout()
fig.savefig(Path(CFG.figures_dir, "fig1_motivating_example.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(Path(CFG.figures_dir, "fig1_motivating_example.png"), dpi=300, bbox_inches="tight")
plt.show()
print(f"PAPER_ARTIFACT: fig1_motivating_example.pdf")


# --- Figure 4: failure treemap -----------------------------------------------
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ma_err_counts = ma_errs.drop("correct", errors="ignore")
total = ma_err_counts.sum()
if total > 0:
    cats = ma_err_counts.index.tolist()
    sizes = ma_err_counts.values
    colors = sns.color_palette("colorblind", len(cats))
    # Simple horizontal bar instead of treemap for portability.
    y = np.arange(len(cats))
    ax.barh(y, sizes, color=colors, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("count")
    ax.set_title("Figure 4: MA-DPO failure-mode taxonomy on Bengali test")
plt.tight_layout()
fig.savefig(Path(CFG.figures_dir, "fig4_failure_taxonomy.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(Path(CFG.figures_dir, "fig4_failure_taxonomy.png"), dpi=300, bbox_inches="tight")
plt.show()
print(f"PAPER_ARTIFACT: fig4_failure_taxonomy.pdf")


# --- Figure 5: cross-lingual heatmap -----------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.2))
methods = ["Base", "MA-DPO"]
langs = ["BN", "HI", "KO"]
matrix = np.zeros((2, 3))
for i, key in enumerate(["zero_shot", "ma_dpo"]):
    for j, lang in enumerate(["bn", "hi", "ko"]):
        if lang in results_bag.get(key, {}):
            matrix[i, j] = results_bag[key][lang].get("cps", 0.0)
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=langs, yticklabels=methods, ax=ax)
ax.set_title("Figure 5: CPS across languages — Base vs MA-DPO (BN-trained)")
plt.tight_layout()
fig.savefig(Path(CFG.figures_dir, "fig5_crosslingual_heatmap.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(Path(CFG.figures_dir, "fig5_crosslingual_heatmap.png"), dpi=300, bbox_inches="tight")
plt.show()
print(f"PAPER_ARTIFACT: fig5_crosslingual_heatmap.pdf")
