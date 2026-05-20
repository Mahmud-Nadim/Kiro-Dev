# =============================================================================
# Cell: SFT training with LoRA.
# Why: A clean baseline that other methods (DPO, MA-DPO) must beat.
#
# FIX for bitsandbytes/triton crash:
#   We block bitsandbytes from being imported by PEFT. We do NOT use 8-bit
#   quantization, so this is safe. If bnb is already broken in your env,
#   this prevents the crash entirely.
# =============================================================================
import importlib
import sys

# --- Neutralize bitsandbytes so PEFT doesn't try to import it ----------------
# This avoids "ModuleNotFoundError: No module named 'triton.ops'" on Colab.
if "bitsandbytes" not in sys.modules:
    # Create a dummy module so `import bitsandbytes` doesn't crash
    import types
    _fake_bnb = types.ModuleType("bitsandbytes")
    _fake_bnb.__version__ = "0.0.0"
    sys.modules["bitsandbytes"] = _fake_bnb
    # Also block sub-imports that PEFT touches
    for sub in ["bitsandbytes.nn", "bitsandbytes.nn.modules",
                "bitsandbytes.functional", "bitsandbytes.autograd"]:
        sys.modules[sub] = types.ModuleType(sub)

# Now force peft to think bnb is NOT available so it skips the bnb dispatch.
import peft.import_utils
peft.import_utils.is_bnb_available = lambda: False
peft.import_utils.is_bnb_4bit_available = lambda: False

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
