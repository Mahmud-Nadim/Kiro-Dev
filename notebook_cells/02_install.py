# =============================================================================
# Cell: Install pinned dependencies.
# Why: Reproducibility. Reviewers cannot reproduce floating-version notebooks.
# =============================================================================
# --- UNCOMMENT THE LINES BELOW ON FIRST RUN ---
# !pip install -q --upgrade pip
# !pip install -q \
#     "transformers>=4.44.0" \
#     "trl>=0.12.0" \
#     "peft>=0.13.0" \
#     "accelerate>=1.0.0" \
#     "datasets>=3.0.0" \
#     "sentencepiece>=0.2.0" \
#     "scikit-learn>=1.5.0" \
#     "matplotlib>=3.9.0" \
#     "seaborn>=0.13.0" \
#     "tabulate>=0.9.0" \
#     "pandas>=2.2.0" \
#     "torchao>=0.16.0"
#
# # IMPORTANT: Uninstall bitsandbytes to avoid the triton.ops import crash.
# # We do NOT use 8-bit quantization in this notebook, so bnb is not needed.
# !pip uninstall -y bitsandbytes 2>/dev/null || true
#
# print("Dependencies installed. Restart runtime now (Runtime -> Restart session).")

# Note: lines are commented out so re-running this cell is idempotent.
# Uncomment the !pip lines on FIRST RUN only, then restart.
print("Install cell ready. Uncomment the !pip lines on first run.")
