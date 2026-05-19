# =============================================================================
# Cell: Zero-shot evaluation by candidate scoring.
# Why: Evaluate the BASE model on PRANAM-Bench by computing the log-likelihood
# of each candidate response and picking the highest. This is more robust
# than free-form generation for small models.
# =============================================================================

def build_prompt(example: dict) -> str:
    """Single string prompt summarizing context + relationship for the LM."""
    rel = example["relationship"]
    sa = rel["speaker_to_addressee"]
    sm = rel.get("speaker_meta", {})
    am = rel.get("addressee_meta", {})

    speaker_role = sm.get("role", "speaker")
    addressee_role = am.get("role", "addressee")

    sys = (
        "You are a culturally-aware Bengali speaker. Produce a reply in Bengali "
        "that respects the social hierarchy between speaker and addressee."
    )
    ctx = (
        f"Speaker: {speaker_role} (age {sm.get('age','?')}). "
        f"Addressee: {addressee_role} (age {am.get('age','?')}). "
        f"Relationship axes: power={sa['power']}, age={sa['age']}, "
        f"intimacy={sa['intimacy']}, formality={sa['formality']}, "
        f"kinship={sa['kinship']}, deference_target={sa['deference_target']}.\n"
    )
    user_turn = example["context_turns"][0]["text"] if example["context_turns"] else ""
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": ctx + "User says: " + user_turn},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_candidate(prompt: str, candidate: str, model, tok) -> float:
    """Average log-prob of candidate tokens given prompt."""
    full = prompt + candidate
    full_ids = tok(full, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    n_prompt = prompt_ids.size(1)
    out = model(full_ids, labels=full_ids)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    cand_logp = chosen[:, n_prompt - 1:].sum().item()
    n_tokens = max(1, full_ids.size(1) - n_prompt)
    return cand_logp / n_tokens


def evaluate_zero_shot(examples: list[dict], model, tok,
                       max_examples: Optional[int] = None) -> dict:
    if max_examples is not None:
        examples = examples[:max_examples]
    n = len(examples)
    n_correct = 0
    cps_scores = []
    per_axis = {ax: [] for ax in CFG.axis_names}
    chosen_indices = []
    for ex in examples:
        prompt = build_prompt(ex)
        scores = []
        for c in ex["candidates"]:
            s = score_candidate(prompt, c["text"], model, tok)
            scores.append(s)
        chosen = int(np.argmax(scores))
        chosen_indices.append(chosen)
        if chosen == ex["gold_index"]:
            n_correct += 1
        chosen_t = PragmaticTensor(**ex["candidates"][chosen]["tensor"])
        gold_t = PragmaticTensor(**ex["candidates"][ex["gold_index"]]["tensor"])
        soft = axiswise_softscore(chosen_t, gold_t)
        cps_scores.append(np.mean(list(soft.values())))
        for ax in CFG.axis_names:
            per_axis[ax].append(soft[ax])
    return {
        "n": n,
        "top1_accuracy": n_correct / n,
        "cps": float(np.mean(cps_scores)),
        "axis_scores": {ax: float(np.mean(vs)) for ax, vs in per_axis.items()},
        "chosen_indices": chosen_indices,
    }


# Run zero-shot baseline on Bengali test split.
zero_shot_bn = evaluate_zero_shot(splits["bn_test"], base_model, tokenizer)
print("Zero-shot baseline (Bengali test):")
print(json.dumps(zero_shot_bn, indent=2))

# Save for later table generation.
results_bag = {"zero_shot": {"bn": zero_shot_bn}}
Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
print(f"\nPAPER_ARTIFACT: zero-shot baseline saved to {CFG.workdir}/results.json")
