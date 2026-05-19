"""
Builder script for the PRANAM / HonorAlign Colab notebook.

This script assembles a Jupyter .ipynb file from cell definitions in this
repository. Run with: python build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
CELLS_DIR = ROOT / "notebook_cells"
OUTPUT = ROOT / "PRANAM_HonorAlign_EMNLP2027.ipynb"


def md(source: str) -> dict:
    """Build a markdown cell from a string."""
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(source: str) -> dict:
    """Build a code cell from a string."""
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def load_cell_file(name: str) -> str:
    return (CELLS_DIR / name).read_text(encoding="utf-8")


# Cell ordering. Each entry is (kind, filename) where kind in {"md", "code"}.
CELL_ORDER = [
    ("md",   "00_title.md"),
    ("md",   "01_overview.md"),
    ("md",   "02_section_setup.md"),
    ("code", "02_install.py"),
    ("code", "02_imports.py"),
    ("code", "02_config.py"),
    ("code", "02_gpu_check.py"),
    ("md",   "03_section_tensor.md"),
    ("code", "03_tensor_dataclass.py"),
    ("code", "03_tensor_examples.py"),
    ("md",   "04_section_dataset.md"),
    ("code", "04_seed_dialogues.py"),
    ("code", "04_candidate_generator.py"),
    ("code", "04_axis_labeler.py"),     # must come BEFORE build_dataset
    ("code", "04_build_dataset.py"),
    ("md",   "05_section_eda.md"),
    ("code", "05_eda_stats.py"),
    ("code", "05_eda_plots.py"),
    ("md",   "06_section_baselines.md"),
    ("code", "06_load_base_model.py"),
    ("code", "06_zero_shot_eval.py"),
    ("md",   "07_section_sft.md"),
    ("code", "07_sft_train.py"),
    ("md",   "08_section_dpo.md"),
    ("code", "08_dpo_train.py"),
    ("md",   "09_section_madpo.md"),
    ("code", "09_madpo_loss.py"),
    ("code", "09_madpo_trainer.py"),
    ("code", "09_madpo_train.py"),
    ("md",   "10_section_metrics.md"),
    ("code", "10_metrics.py"),
    ("code", "10_run_eval.py"),
    ("md",   "11_section_crosslingual.md"),
    ("code", "11_crosslingual.py"),
    ("md",   "12_section_ablations.md"),
    ("code", "12_ablations.py"),
    ("md",   "13_section_error_analysis.md"),
    ("code", "13_error_analysis.py"),
    ("md",   "14_section_paper_artifacts.md"),
    ("code", "14_latex_tables.py"),
    ("code", "14_figures.py"),
    ("md",   "15_section_human_eval.md"),
    ("code", "15_argilla_export.py"),
    ("md",   "16_section_reproducibility.md"),
    ("code", "16_save_artifacts.py"),
    ("md",   "17_section_roadmap.md"),
]


def build():
    cells = []
    for kind, fname in CELL_ORDER:
        src = load_cell_file(fname)
        if kind == "md":
            cells.append(md(src))
        else:
            cells.append(code(src))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUTPUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(cells)} cells.")


if __name__ == "__main__":
    build()
