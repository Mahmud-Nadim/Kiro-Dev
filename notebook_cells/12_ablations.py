# =============================================================================
# Cell: Run ablations.
# Why: Each row of Table 3. Pre-empts reviewer "why didn't you ablate X?".
# =============================================================================
RUN_FULL_ABLATIONS = False  # set True for full set; will take ~1 hour on T4.

ablation_results: dict[str, dict] = {}


def _run_ablation(name: str, train_pairs: list[dict],
                  use_axis_weights: bool = True,
                  drop_graph_from_prompt: bool = False,
                  epochs: int = 1) -> dict:
    """Run one ablation training and return its evaluation."""
    print(f"\n--- Ablation: {name} ---")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_for_ab = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    ab_model = get_peft_model(base_for_ab, peft_config)

    pairs = train_pairs
    if drop_graph_from_prompt:
        # Re-build prompts without the relationship axes line.
        for p in pairs:
            p["prompt"] = re.sub(r"Relationship axes:.*\n", "", p["prompt"])

    ds = Dataset.from_list(pairs)
    args = DPOConfig(
        output_dir=str(Path(CFG.models_dir, f"ablation_{name}")),
        num_train_epochs=epochs,
        per_device_train_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.grad_accum,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        logging_steps=20,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=CFG.seed,
        beta=CFG.dpo_beta,
        max_length=CFG.max_length,
        max_prompt_length=CFG.max_length // 2,
        remove_unused_columns=False,
    )

    try:
        trainer = MADPOTrainer(
            model=ab_model, ref_model=None, args=args, train_dataset=ds,
            tokenizer=tokenizer, n_axes=CFG.n_axes,
            learn_axis_weights=use_axis_weights,
        )
        trainer.train()
    except Exception as e:
        print(f"  fallback to vanilla DPO: {e!r}")
        ds2 = ds.remove_columns(["axis_index"]) if "axis_index" in ds.column_names else ds
        trainer = DPOTrainer(model=ab_model, ref_model=None, args=args,
                             train_dataset=ds2, tokenizer=tokenizer)
        trainer.train()

    ab_model.eval()
    res = evaluate_zero_shot(splits["bn_test"], ab_model, tokenizer)
    res["honorific_register_accuracy"] = metric_honorific_register_accuracy(
        splits["bn_test"], res["chosen_indices"]
    )
    del ab_model, base_for_ab, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# Ablation 1: drop graph conditioning.
ablation_results["A1_no_graph"] = _run_ablation(
    "A1_no_graph",
    [dict(p) for p in ma_pairs_train],
    use_axis_weights=True,
    drop_graph_from_prompt=True,
    epochs=1,
)

# Ablation 2: uniform axis weights.
ablation_results["A2_uniform_alpha"] = _run_ablation(
    "A2_uniform_alpha",
    [dict(p) for p in ma_pairs_train],
    use_axis_weights=False,
    epochs=1,
)

# Ablation 3: random axis pairs (sanity).
random.seed(CFG.seed + 1)
random_pairs = []
for ex in splits["bn_train"]:
    cands = ex["candidates"]
    if len(cands) < 2:
        continue
    a, b = random.sample(range(len(cands)), 2)
    random_pairs.append({
        "prompt": build_prompt(ex),
        "chosen": cands[a]["text"],
        "rejected": cands[b]["text"],
        "axis_index": random.randint(0, CFG.n_axes - 1),
    })
ablation_results["A3_random_pairs"] = _run_ablation(
    "A3_random_pairs", random_pairs, epochs=1,
)

if RUN_FULL_ABLATIONS:
    # Ablation 4: 6x single-axis DPO ensemble (run six times then evaluate by majority).
    print("\n--- Ablation 4: single-axis DPO ensemble ---")
    per_axis_chosen = {ax: [] for ax in CFG.axis_names}
    for axis_idx, axis_name in enumerate(CFG.axis_names):
        axis_only = [p for p in ma_pairs_train if p["axis_index"] == axis_idx]
        if not axis_only:
            continue
        res = _run_ablation(f"A4_{axis_name}_only", axis_only,
                            use_axis_weights=False, epochs=1)
        per_axis_chosen[axis_name] = res["chosen_indices"]
    # Majority vote — collapse to one chosen per example.
    n = len(splits["bn_test"])
    voted = []
    for i in range(n):
        votes = [per_axis_chosen[ax][i] for ax in CFG.axis_names if per_axis_chosen[ax]]
        if not votes:
            voted.append(0)
        else:
            voted.append(int(np.bincount(votes).argmax()))
    n_correct = sum(1 for i in range(n) if voted[i] == splits["bn_test"][i]["gold_index"])
    cps_vals = []
    for i, ex in enumerate(splits["bn_test"]):
        ct = PragmaticTensor(**ex["candidates"][voted[i]]["tensor"])
        gt = PragmaticTensor(**ex["candidates"][ex["gold_index"]]["tensor"])
        cps_vals.append(np.mean(list(axiswise_softscore(ct, gt).values())))
    ablation_results["A4_singleaxis_ensemble"] = {
        "top1_accuracy": n_correct / n,
        "cps": float(np.mean(cps_vals)),
        "axis_scores": {},
        "honorific_register_accuracy": metric_honorific_register_accuracy(splits["bn_test"], voted),
    }

# Build ablation table.
ab_rows = []
ab_rows.append({"Variant": "MA-DPO (full)", **{
    "Top-1": results_bag.get("ma_dpo", {}).get("bn", {}).get("top1_accuracy", float("nan")),
    "CPS": results_bag.get("ma_dpo", {}).get("bn", {}).get("cps", float("nan")),
    "HRA": results_bag.get("ma_dpo", {}).get("bn", {}).get("honorific_register_accuracy", float("nan")),
}})
for name, res in ablation_results.items():
    ab_rows.append({
        "Variant": name,
        "Top-1": res.get("top1_accuracy", float("nan")),
        "CPS": res.get("cps", float("nan")),
        "HRA": res.get("honorific_register_accuracy", float("nan")),
    })
ab_df = pd.DataFrame(ab_rows)
print("\n=== Table 3: Ablations ===")
print(ab_df.to_string(index=False, float_format="%.3f"))
ab_df.to_csv(Path(CFG.tables_dir, "table3_ablations.csv"), index=False)
results_bag["ablations"] = ablation_results
Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
print(f"\nPAPER_ARTIFACT: {CFG.tables_dir}/table3_ablations.csv")
