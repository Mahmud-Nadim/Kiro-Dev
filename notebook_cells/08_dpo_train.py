# =============================================================================
# Cell: Vanilla DPO training.
# Why: Strong, established baseline. We use TRL's DPOTrainer.
# =============================================================================
from trl import DPOTrainer, DPOConfig

# Build DPO pairs: gold vs furthest distractor.
def build_dpo_pairs(examples: list[dict]) -> list[dict]:
    pairs = []
    for ex in examples:
        gold = ex["candidates"][ex["gold_index"]]
        gold_t = PragmaticTensor(**gold["tensor"])
        # Pick the candidate maximizing axis_distance to gold.
        worst_j, worst_d = None, -1
        for j, c in enumerate(ex["candidates"]):
            if j == ex["gold_index"]:
                continue
            d = axis_distance(PragmaticTensor(**c["tensor"]), gold_t)
            if d > worst_d:
                worst_d, worst_j = d, j
        if worst_j is None:
            continue
        pairs.append({
            "prompt": build_prompt(ex),
            "chosen": gold["text"],
            "rejected": ex["candidates"][worst_j]["text"],
        })
    return pairs


dpo_pairs = build_dpo_pairs(splits["bn_train"])
print(f"DPO training pairs: {len(dpo_pairs)}")

dpo_dataset = Dataset.from_list(dpo_pairs)

# Reload a fresh LoRA model for DPO (separate from SFT model).
del sft_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

base_for_dpo = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
dpo_model = get_peft_model(base_for_dpo, peft_config)

dpo_args = DPOConfig(
    output_dir=str(Path(CFG.models_dir, "dpo")),
    num_train_epochs=CFG.dpo_epochs,
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
)

dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=None,  # use peft adapter inactive as ref
    args=dpo_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
dpo_model.save_pretrained(Path(CFG.models_dir, "dpo_lora"))
print(f"DPO LoRA saved.")

dpo_model.eval()
dpo_results = evaluate_zero_shot(splits["bn_test"], dpo_model, tokenizer)
print("\nVanilla DPO Results (Bengali test):")
print(json.dumps(dpo_results, indent=2))
results_bag["dpo"] = {"bn": dpo_results}
Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
