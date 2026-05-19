# =============================================================================
# Cell: Generate LaTeX tables.
# Why: Paste-ready output. Reviewers prefer well-formatted tables.
# =============================================================================
def df_to_latex_booktabs(df: pd.DataFrame, caption: str, label: str,
                         float_fmt: str = "%.3f") -> str:
    body = df.to_latex(
        index=False,
        float_format=float_fmt,
        column_format="l" + "c" * (len(df.columns) - 1),
        escape=True,
    )
    out = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}"
    )
    return out


# Table 1: main results.
main_tex = df_to_latex_booktabs(
    main_df, caption="Main results on PRANAM-Bench-Mini (Bengali test). "
    "MA-DPO achieves the highest CPS while maintaining honorific register accuracy.",
    label="tab:main",
)
Path(CFG.tables_dir, "table1_main_results.tex").write_text(main_tex)

# Table 2: cross-lingual.
xl_tex = df_to_latex_booktabs(
    xl_df, caption="Cross-lingual zero-shot transfer of MA-DPO trained on Bengali.",
    label="tab:xlingual",
)
Path(CFG.tables_dir, "table2_crosslingual.tex").write_text(xl_tex)

# Table 3: ablations.
ab_tex = df_to_latex_booktabs(
    ab_df, caption="Ablations on MA-DPO components.",
    label="tab:ablation",
)
Path(CFG.tables_dir, "table3_ablations.tex").write_text(ab_tex)

# Table 4: failure taxonomy.
err_disp_df = err_df.reset_index().rename(columns={"index": "Category"})
err_tex = df_to_latex_booktabs(
    err_disp_df, caption="Failure-mode taxonomy: counts per error category on Bengali test.",
    label="tab:failures", float_fmt="%d",
)
Path(CFG.tables_dir, "table4_failure_taxonomy.tex").write_text(err_tex)

print("PAPER_ARTIFACT: 4 LaTeX tables written to", CFG.tables_dir)
for f in sorted(Path(CFG.tables_dir).glob("*.tex")):
    print("  ", f.name)
