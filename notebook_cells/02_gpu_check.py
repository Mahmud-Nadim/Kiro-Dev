# =============================================================================
# Cell: Verify GPU and memory.
# Why: Catch "out of memory" surprises before training. A T4 has 16 GB; we
# size our LoRA + 0.5B base to fit in ~6 GB, leaving headroom.
# =============================================================================
if HAS_TORCH and torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name}")
    print(f"GPU memory: {gpu_mem_gb:.1f} GB")
    if gpu_mem_gb < 14:
        print("WARNING: less than 14 GB. Consider switching base_model to a "
              "smaller checkpoint or reducing max_length.")
else:
    print("No GPU detected. Section 6 onwards will be slow or skipped.")

# Also print the Python and torch versions so we can paste them into the paper
# Reproducibility section.
import sys
print(f"\nPython: {sys.version.split()[0]}")
if HAS_TORCH:
    print(f"torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
