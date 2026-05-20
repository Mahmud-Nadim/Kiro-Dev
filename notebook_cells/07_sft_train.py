# =============================================================================
# Cell: SFT training with LoRA.
# Why: A clean baseline that other methods (DPO, MA-DPO) must beat.
#
# FIX for Colab environment issues:
#   1. bitsandbytes has broken triton.ops dependency
#   2. torchao 0.10.0 (Colab default) is too old for PEFT >= 0.14
#   We neutralize BOTH by patching every location where they're referenced.
# =============================================================================
import importlib
import importlib.util
import sys

# =============================================================================
# NUCLEAR FIX: Remove torchao entirely so PEFT can never find/check it.
# This is safe because we use standard fp16 LoRA, not quantized layers.
# =============================================================================
_torchao_keys = [k for k in list(sys.modules.keys()) if k == "torchao" or k.startswith("torchao.")]
for k in _torchao_keys:
    del sys.modules[k]

# Block future torchao imports.
class _BlockModuleFinder:
    """Prevents specified modules from being imported."""
    def __init__(self, blocked_prefixes):
        self.blocked = blocked_prefixes

    def find_module(self, fullname, path=None):
        for prefix in self.blocked:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self
        return None

    def load_module(self, fullname):
        raise ImportError(f"{fullname} is blocked by notebook neutralizer")

# Block both bitsandbytes and torchao.
_blocker = _BlockModuleFinder(["bitsandbytes", "torchao"])
if _blocker not in sys.meta_path:
    sys.meta_path.insert(0, _blocker)

# Also remove bitsandbytes.
_bnb_keys = [k for k in list(sys.modules.keys()) if k == "bitsandbytes" or k.startswith("bitsandbytes.")]
for k in _bnb_keys:
    del sys.modules[k]

# =============================================================================
# Now patch PEFT's import_utils at EVERY level.
# The @lru_cache means we must clear it AND replace the function object AND
# patch any module that already imported a reference to the old function.
# =============================================================================
import peft.import_utils

# Patch all availability checkers.
_fns_to_kill = [
    "is_bnb_available",
    "is_bnb_4bit_available",
    "is_torchao_available",
]

for fn_name in _fns_to_kill:
    fn = getattr(peft.import_utils, fn_name, None)
    if fn is None:
        continue
    # Clear lru_cache if present.
    if hasattr(fn, "cache_clear"):
        fn.cache_clear()
    # Replace on the module.
    setattr(peft.import_utils, fn_name, lambda: False)

# CRITICAL: Also patch in peft.tuners.lora.model where it's imported directly.
try:
    import peft.tuners.lora.model as _lora_model
    if hasattr(_lora_model, "is_torchao_available"):
        _lora_model.is_torchao_available = lambda: False
    if hasattr(_lora_model, "is_bnb_available"):
        _lora_model.is_bnb_available = lambda: False
except (ImportError, AttributeError):
    pass

# Also patch in peft.tuners.tuners_utils if it imported these.
try:
    import peft.tuners.tuners_utils as _tuners_utils
    if hasattr(_tuners_utils, "is_torchao_available"):
        _tuners_utils.is_torchao_available = lambda: False
    if hasattr(_tuners_utils, "is_bnb_available"):
        _tuners_utils.is_bnb_available = lambda: False
except (ImportError, AttributeError):
    pass

# Also patch peft.mapping / peft.mapping_func.
try:
    import peft.mapping_func as _mapping
    if hasattr(_mapping, "is_torchao_available"):
        _mapping.is_torchao_available = lambda: False
except (ImportError, AttributeError):
    pass

# Patch ANY peft submodule that has a reference to these functions.
for mod_name, mod in list(sys.modules.items()):
    if mod is None or not mod_name.startswith("peft"):
        continue
    for fn_name in _fns_to_kill:
        if hasattr(mod, fn_name):
            setattr(mod, fn_name, lambda: False)

print("bitsandbytes + torchao fully neutralized across all PEFT submodules.")

# =============================================================================
# Actual SFT training code below.
# =============================================================================
from transformers import Trainer, DataCollatorForLanguageModeling

# Build SFT dataset: (prompt + gold_response) for each train example.
sft_records = []
for ex in splits["bn_train"]:
    prompt = build_prompt(ex)
    gold = ex["candidates"][ex["gold_index"]]["text"]
    sft_records.append({"prompt": prompt, "completion": gold})

print(f"SFT records: {len(sft_records)}")


def tokenize_sft(rec):
    full = rec["prompt"] + rec["completion"] + tokenizer.eos_token
    enc = tokenizer(full, truncation=True, max_length=CFG.max_length,
                    padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0)
    attn = enc["attention_mask"].squeeze(0)
    # Mask prompt tokens from the loss.
    prompt_len = len(tokenizer(rec["prompt"], truncation=True,
                               max_length=CFG.max_length)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[attn == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


sft_tokenized = [tokenize_sft(r) for r in sft_records]


class SimpleSFTDataset(TorchDataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


# LoRA wrap — safe because we disabled bnb + torchao dispatches above.
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=list(CFG.lora_target_modules),
    bias="none",
)
sft_model = get_peft_model(base_model, peft_config)
sft_model.print_trainable_parameters()

sft_training_args = TrainingArguments(
    output_dir=str(Path(CFG.models_dir, "sft")),
    num_train_epochs=CFG.sft_epochs,
    per_device_train_batch_size=CFG.batch_size,
    gradient_accumulation_steps=CFG.grad_accum,
    learning_rate=CFG.learning_rate,
    warmup_ratio=CFG.warmup_ratio,
    logging_steps=10,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=CFG.seed,
)

trainer = Trainer(
    model=sft_model,
    args=sft_training_args,
    train_dataset=SimpleSFTDataset(sft_tokenized),
    tokenizer=tokenizer,
)

trainer.train()
sft_model.save_pretrained(Path(CFG.models_dir, "sft_lora"))
print(f"SFT LoRA saved to {Path(CFG.models_dir, 'sft_lora')}")

# Evaluate.
sft_model.eval()
sft_results = evaluate_zero_shot(splits["bn_test"], sft_model, tokenizer)
print("\nSFT Results (Bengali test):")
print(json.dumps(sft_results, indent=2))
results_bag["sft"] = {"bn": sft_results}
Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
