# =============================================================================
# Cell: SFT training with LoRA.
# Why: A clean baseline that other methods (DPO, MA-DPO) must beat.
#
# FIX for bitsandbytes/triton crash on Colab:
#   PEFT calls importlib.util.find_spec("bitsandbytes") via an @lru_cache'd
#   function. If bnb is broken (triton.ops missing), this crashes. We:
#   1. Remove bitsandbytes from sys.modules entirely so find_spec returns None
#   2. Patch the @lru_cache'd checker to always return False
#   3. Clear the lru_cache so our patch takes effect
# =============================================================================
import importlib
import importlib.util
import sys
from functools import lru_cache

# --- Step 1: Completely remove bitsandbytes from sys.modules -----------------
_bnb_keys = [k for k in sys.modules if k == "bitsandbytes" or k.startswith("bitsandbytes.")]
for k in _bnb_keys:
    del sys.modules[k]

# --- Step 2: Block future imports of bitsandbytes ----------------------------
# Install an import hook that prevents bitsandbytes from ever loading.
import importlib.abc
import importlib.machinery


class _BlockBnbFinder(importlib.abc.MetaPathFinder):
    """Prevents bitsandbytes from being imported."""
    def find_module(self, fullname, path=None):
        if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
            return self
        return None

    def load_module(self, fullname):
        raise ImportError(f"{fullname} is blocked (bnb neutralizer active)")


# Insert at the FRONT of sys.meta_path so it takes priority.
sys.meta_path.insert(0, _BlockBnbFinder())

# --- Step 3: Patch PEFT's cached availability checks -------------------------
import peft.import_utils

# Clear the lru_cache on the existing functions (if they have it).
for fn_name in ["is_bnb_available", "is_bnb_4bit_available"]:
    fn = getattr(peft.import_utils, fn_name, None)
    if fn is not None and hasattr(fn, "cache_clear"):
        fn.cache_clear()

# Replace with simple lambdas that always return False.
peft.import_utils.is_bnb_available = lambda: False
peft.import_utils.is_bnb_4bit_available = lambda: False

# --- Step 4: Neutralize torchao version check --------------------------------
# PEFT >= 0.14 checks torchao version and raises ImportError if too old.
# We don't use torchao quantization, so we disable the check entirely.
for fn_name in ["is_torchao_available"]:
    fn = getattr(peft.import_utils, fn_name, None)
    if fn is not None and hasattr(fn, "cache_clear"):
        fn.cache_clear()
    if fn is not None:
        setattr(peft.import_utils, fn_name, lambda: False)

print("bitsandbytes + torchao neutralized — PEFT will use standard fp16 LoRA only.")

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


# LoRA wrap — now safe because we disabled the bnb dispatch above.
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
