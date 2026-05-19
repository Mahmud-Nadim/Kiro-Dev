# =============================================================================
# Cell: Install pinned dependencies.
# Why: Reproducibility. Reviewers cannot reproduce floating-version notebooks.
# =============================================================================
# !pip install -q --upgrade pip
# !pip install -q \
#     "transformers==4.46.3" \
#     "trl==0.12.1" \
#     "peft==0.13.2" \
#     "accelerate==1.1.1" \
#     "bitsandbytes==0.44.1" \
#     "datasets==3.1.0" \
#     "sentencepiece==0.2.0" \
#     "scikit-learn==1.5.2" \
#     "matplotlib==3.9.2" \
#     "seaborn==0.13.2" \
#     "tabulate==0.9.0" \
#     "pandas==2.2.3"
# print("Dependencies installed. If running on Colab, restart runtime now if "
#       "this is your first install of the session.")

# Note: lines are commented out so re-running this cell is idempotent. Uncomment
# the !pip lines on first run only.
print("Install cell ready. Uncomment the !pip lines on first run.")
