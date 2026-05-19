# =============================================================================
# Cell: MA-DPO data construction — axis-decomposed preference pairs.
# Why: Each example becomes up to 6 (one per axis) preference pairs. This is
# the ENTIRE point of the method.
# =============================================================================

def build_ma_dpo_pairs(examples: list[dict]) -> list[dict]:
    """For each axis k, build (winner, loser) pairs based on per-axis correctness.

    Winner = candidate whose axis-k value matches gold's axis-k value.
    Loser  = candidate whose axis-k value diverges most from gold's axis-k value.
    Skip the axis if no losing candidate exists (rare in our seed data).
    """
    pairs = []
    for ex in examples:
        gold = ex["candidates"][ex["gold_index"]]
        gold_t = PragmaticTensor(**gold["tensor"])
        prompt = build_prompt(ex)

        for axis_idx, axis_name in enumerate(CFG.axis_names):
            # Winners: candidates that match gold on this axis.
            winners = []
            losers = []
            for c in ex["candidates"]:
                ct = PragmaticTensor(**c["tensor"])
                gold_v = getattr(gold_t, axis_name)
                cand_v = getattr(ct, axis_name)
                if cand_v == gold_v:
                    winners.append(c["text"])
                else:
                    # Distance on this axis only.
                    if axis_name in ORDINAL_AXES or axis_name == "formality":
                        diff = abs(cand_v - gold_v)
                    else:
                        diff = 1
                    losers.append((diff, c["text"]))

            if not winners or not losers:
                continue

            # Pick the gold's text as the canonical winner.
            winner_text = gold["text"]
            # Pick the largest-difference loser.
            losers.sort(key=lambda t: -t[0])
            loser_text = losers[0][1]

            pairs.append({
                "prompt": prompt,
                "chosen": winner_text,
                "rejected": loser_text,
                "axis_index": axis_idx,
                "axis_name": axis_name,
                "example_id": ex["id"],
            })
    return pairs


ma_pairs_train = build_ma_dpo_pairs(splits["bn_train"])
print(f"MA-DPO train pairs: {len(ma_pairs_train)} "
      f"({len(ma_pairs_train) / max(len(splits['bn_train']), 1):.2f} per example)")

# Per-axis pair counts (for axis-balance check + a paper appendix table).
axis_counts = {ax: 0 for ax in CFG.axis_names}
for p in ma_pairs_train:
    axis_counts[p["axis_name"]] += 1
print("\nPer-axis pair counts:")
for ax, n in axis_counts.items():
    print(f"  {ax:>18s}: {n}")
