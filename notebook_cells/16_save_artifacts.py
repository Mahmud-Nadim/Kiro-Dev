# =============================================================================
# Cell: Save reproducibility manifest + model card.
# Why: Reviewers / replicators need this. Also: ARR demands a checklist.
# =============================================================================
import hashlib
import platform

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


manifest = {
    "title": "PRANAM / HonorAlign — Reproducibility Manifest",
    "config": asdict(CFG),
    "platform": {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "torch_version": torch.__version__ if HAS_TORCH else None,
    "cuda_version": (torch.version.cuda if HAS_TORCH and torch.cuda.is_available() else None),
    "gpu": (torch.cuda.get_device_name(0) if HAS_TORCH and torch.cuda.is_available() else None),
    "data_files": {
        p.name: {
            "path": str(p),
            "sha256": file_sha256(p),
            "size_bytes": p.stat().st_size,
        }
        for p in Path(CFG.data_dir).glob("*.jsonl")
    },
    "tables": [str(p) for p in Path(CFG.tables_dir).glob("*")],
    "figures": [str(p) for p in Path(CFG.figures_dir).glob("*")],
    "models": [str(p) for p in Path(CFG.models_dir).iterdir() if p.is_dir()],
}
Path(CFG.workdir, "reproducibility_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)
print(f"PAPER_ARTIFACT: reproducibility_manifest.json")

# Model card.
card = f"""---
license: openrail
language:
  - bn
  - hi
  - ko
tags:
  - honorific
  - alignment
  - dpo
  - bengali
  - low-resource
---

# PRANAM-MA-DPO ({CFG.base_model.split('/')[-1]} + LoRA)

This is the released LoRA adapter from the PRANAM / HonorAlign paper at EMNLP 2027 (under review).

## Method
Multi-Axis DPO (MA-DPO) trained on PRANAM-Bench-Mini Bengali split. See paper for details.

## Limitations
- Trained on a synthetic seed dataset. v2 (after human annotation) is the publication target.
- Tested only on three languages (BN, HI, KO).
- May not generalize to other South Asian languages without further fine-tuning.

## Citation
```bibtex
@inproceedings{{pranam2027,
  title={{PRANAM: Relational-Pragmatic Preference Optimization for Honorific-Rich Languages}},
  author={{[Anonymous]}},
  booktitle={{EMNLP 2027}},
  year={{2027}},
}}
```
"""
Path(CFG.models_dir, "MODEL_CARD.md").write_text(card)
print(f"PAPER_ARTIFACT: MODEL_CARD.md")

# Final summary print.
print("\n" + "=" * 60)
print("FINAL ARTIFACT INVENTORY")
print("=" * 60)
for label, glob in [
    ("Data files", "data/*.jsonl"),
    ("Argilla export", "data/*.json"),
    ("Tables", "tables/*"),
    ("Figures", "figures/*"),
    ("Manifests", "*.json"),
]:
    paths = list(Path(CFG.workdir).glob(glob))
    print(f"\n{label}: {len(paths)}")
    for p in paths[:8]:
        print(f"  - {p.name}")
    if len(paths) > 8:
        print(f"  ... and {len(paths) - 8} more")
