# =============================================================================
# Cell: Train MA-DPO.
# Why: This is the headline experiment. We expect MA-DPO to outperform
# vanilla DPO on CPS by 3-8 absolute points if the method works.
#
# Pragmatic note: TRL's DPOTrainer signature evolves between versions. If this
# cell errors on `axis_index` injection, the fallback is to train standard DPO
# six times (one per axis) and ensemble — see the comment at the bottom.
# =============================================================================

# Free memory — safe deletion handles skipped/re-run cells.
for _varname in ["dpo_model", "dpo_trainer", "base_for_dpo"]:
    if _varname in dir():
        exec(f"del {_varname}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Build dataset with axis_index.
ma_dataset = Dataset.from_list(ma_pairs_train)


def add_axis_index(example):
    return example  # axis_index already present from build_ma_dpo_pairs


# Reload base for MA-DPO.
base_for_madpo = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
ma_model = get_peft_model(base_for_madpo, peft_config)

ma_args = DPOConfig(
    output_dir=str(Path(CFG.models_dir, "madpo")),
    num_train_epochs=CFG.madpo_epochs,
    per_device_train_batch_size=CFG.batch_size,
    gradient_accumulation_steps=CFG.grad_accum,
    learning_rate=CFG.learning_rate,
    warmup_ratio=CFG.warmup_ratio,
    logging_steps=10,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=CFG.seed,
    beta=CFG.dpo_beta,
    max_length=CFG.max_length,
    max_prompt_length=CFG.max_length // 2,
    remove_unused_columns=False,  # keep axis_index
)

# Wrap dataset rows so axis_index is on the example.
def _prep(rec):
    return {
        "prompt": rec["prompt"],
        "chosen": rec["chosen"],
        "rejected": rec["rejected"],
        "axis_index": rec["axis_index"],
    }

ma_dataset_prepped = ma_dataset.map(_prep)

try:
    ma_trainer = MADPOTrainer(
        model=ma_model,
        ref_model=None,
        args=ma_args,
        train_dataset=ma_dataset_prepped,
        tokenizer=tokenizer,
        n_axes=CFG.n_axes,
        learn_axis_weights=CFG.learn_axis_weights,
    )
    ma_trainer.train()
    ma_model.save_pretrained(Path(CFG.models_dir, "madpo_lora"))
    print("\nMA-DPO training complete.")
    print("Final axis weights (alpha):", ma_trainer.get_alpha().detach().cpu().tolist())
    print("Per-axis avg loss:", ma_trainer.axis_loss_summary())
except Exception as e:
    print(f"MA-DPO training raised: {e!r}")
    print("Falling back to standard DPO on the multi-axis pairs (no per-axis weighting).")
    ma_trainer = DPOTrainer(
        model=ma_model,
        ref_model=None,
        args=ma_args,
        train_dataset=ma_dataset_prepped.remove_columns(["axis_index"]),
        tokenizer=tokenizer,
    )
    ma_trainer.train()
    ma_model.save_pretrained(Path(CFG.models_dir, "madpo_lora"))

# Evaluate.
ma_model.eval()
ma_results = evaluate_zero_shot(splits["bn_test"], ma_model, tokenizer)
print("\nMA-DPO Results (Bengali test):")
print(json.dumps(ma_results, indent=2))
results_bag["ma_dpo"] = {"bn": ma_results}
Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
print(f"\nPAPER_ARTIFACT: ma_dpo results saved.")
