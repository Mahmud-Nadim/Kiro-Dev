# PRANAM / HonorAlign — EMNLP 2027 Submission Pipeline

Runnable Colab notebook + builder for the paper:

> **PRANAM: Relational-Pragmatic Preference Optimization for Honorific-Rich Languages**

## Files

- `PRANAM_HonorAlign_EMNLP2027.ipynb` — the main 46-cell Colab notebook. Open in Colab, set runtime to T4 GPU, run top-to-bottom.
- `build_notebook.py` — builder script that assembles the notebook from `notebook_cells/`. Re-run after editing any cell.
- `notebook_cells/` — individual cell sources (one file per cell). Edit these, then re-run the builder.

## Quick start (Colab)

1. Upload `PRANAM_HonorAlign_EMNLP2027.ipynb` to Google Colab.
2. Runtime → Change runtime type → GPU (T4 is enough for the mini version).
3. Run the install cell (Cell 4) — uncomment the `!pip` lines on first run.
4. Run all remaining cells in order. Total time: ~2 hours on free T4.

## What the notebook produces

- `pranam_workdir/data/*.jsonl` — PRANAM-Bench-Mini dataset (240 examples).
- `pranam_workdir/tables/*.csv` and `*.tex` — paper-ready tables (Table 1–4).
- `pranam_workdir/figures/*.pdf` — paper-ready figures (Figure 1–5).
- `pranam_workdir/models/madpo_lora/` — trained LoRA adapter.
- `pranam_workdir/reproducibility_manifest.json` — for the EMNLP reproducibility checklist.
- `pranam_workdir/data/argilla_export_bn_test.json` — for scaling up annotation.

## Editing the notebook

Each cell lives as a separate file in `notebook_cells/`:

- `*.md` files = markdown cells.
- `*.py` files = code cells.
- The order is defined in `build_notebook.py`'s `CELL_ORDER` list.

After editing:

```bash
python3 build_notebook.py
```

This regenerates `PRANAM_HonorAlign_EMNLP2027.ipynb` cleanly.

## Roadmap to publication

See Section 17 of the notebook ("Roadmap to the EMNLP 2027 Publication") for the full step-by-step plan from this artifact to an accepted submission.
