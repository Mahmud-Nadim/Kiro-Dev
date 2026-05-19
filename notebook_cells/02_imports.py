# =============================================================================
# Cell: Imports and global state.
# Why: Single import block makes the notebook scannable and reduces cell-order
# fragility. All later cells assume these names are bound.
# =============================================================================
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Heavy ML deps — wrapped in try/except so the dataset/EDA cells run even on a
# CPU-only kernel for quick iteration.
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available — only data/EDA cells will run.")

try:
    from datasets import Dataset, DatasetDict
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

print(f"torch={HAS_TORCH}  hf_datasets={HAS_HF}  transformers={HAS_TRANSFORMERS}")
