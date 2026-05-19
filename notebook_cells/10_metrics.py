# =============================================================================
# Cell: Define metrics functions.
# Why: Single canonical implementation reused across all methods.
# =============================================================================

def metric_top1_accuracy(eval_result: dict) -> float:
    return eval_result["top1_accuracy"]


def metric_cps(eval_result: dict) -> float:
    return eval_result["cps"]


def metric_axis_accuracies(eval_result: dict) -> dict:
    return eval_result["axis_scores"]


def metric_honorific_register_accuracy(examples: list[dict],
                                        chosen_indices: list[int]) -> float:
    """Did the model pick a candidate with the correct honorific register?
    We define register by the Power axis sign:
      power <= -1  → high (apni / aap / hapsyo-che)
      power == 0   → mid  (tumi / tum / haeyo-che)
      power >= 1   → low  (tui / tu / panmal)
    """
    n_correct = 0
    n_total = 0
    for ex, chosen in zip(examples, chosen_indices):
        gold_t = PragmaticTensor(**ex["candidates"][ex["gold_index"]]["tensor"])
        cand_t = PragmaticTensor(**ex["candidates"][chosen]["tensor"])
        def _bucket(p):
            if p <= -1: return "high"
            if p >= 1:  return "low"
            return "mid"
        if _bucket(gold_t.power) == _bucket(cand_t.power):
            n_correct += 1
        n_total += 1
    return n_correct / max(n_total, 1)


def metric_capability_tax(zero_shot_score: float,
                           method_score: float) -> float:
    """Negative = method dropped capability."""
    return method_score - zero_shot_score


# Compute HRA for every method we have run so far.
def attach_hra(method_key: str, examples: list[dict]):
    if method_key not in results_bag:
        return
    chosen = results_bag[method_key]["bn"].get("chosen_indices")
    if chosen is None:
        return
    hra = metric_honorific_register_accuracy(examples, chosen)
    results_bag[method_key]["bn"]["honorific_register_accuracy"] = hra
    print(f"{method_key}: HRA = {hra:.3f}")


for k in ["zero_shot", "sft", "dpo", "ma_dpo"]:
    attach_hra(k, splits["bn_test"])

Path(CFG.workdir, "results.json").write_text(json.dumps(results_bag, indent=2))
print("\nPAPER_ARTIFACT: results.json updated with HRA.")
