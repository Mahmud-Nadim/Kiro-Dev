# =============================================================================
# Cell: Assemble the final PRANAM-Bench-Mini dataset.
# Why: Single source of truth used by every downstream cell. We split into
# train / dev / test and serialize to disk for reproducibility.
# =============================================================================

def build_dialogue_examples(instances: list, lang_code: str) -> list[DialogueExample]:
    examples = []
    for i, inst in enumerate(instances):
        rel_dict = inst["rel"]
        rel = RelationshipGraph(
            speaker_to_addressee=PragmaticTensor(**rel_dict["speaker_to_addressee"]),
            speaker_meta=rel_dict.get("speaker_meta", {}),
            addressee_meta=rel_dict.get("addressee_meta", {}),
        )
        candidates = []
        for cand_text, axis_hint in inst["candidates_en_axes"]:
            candidates.append({
                "text": cand_text,
                "tensor": axis_labels_from_hint(axis_hint),
            })
        ex = DialogueExample(
            id=f"pranam_{lang_code}_{i:05d}",
            language=lang_code,
            context_turns=[
                {"speaker": "user", "text": inst.get("user_prompt_en", "")},
            ],
            relationship=rel,
            candidates=candidates,
            gold_index=inst["gold_index"],
            notes=f"template={inst['tag']} filler={inst.get('filler_name','')}",
        )
        examples.append(ex)
    return examples


bn_examples = build_dialogue_examples(bn_instances, "bn")
hi_examples = build_dialogue_examples(hi_instances, "hi")
ko_examples = build_dialogue_examples(ko_instances, "ko")

# 80/10/10 splits per language.
def _split(xs, train=0.8, dev=0.1):
    n = len(xs)
    n_train = int(n * train)
    n_dev = int(n * dev)
    return xs[:n_train], xs[n_train:n_train + n_dev], xs[n_train + n_dev:]


bn_tr, bn_dv, bn_te = _split(bn_examples)
hi_tr, hi_dv, hi_te = _split(hi_examples)
ko_tr, ko_dv, ko_te = _split(ko_examples)

print(f"Bengali split: {len(bn_tr)}/{len(bn_dv)}/{len(bn_te)}")
print(f"Hindi split:   {len(hi_tr)}/{len(hi_dv)}/{len(hi_te)}")
print(f"Korean split:  {len(ko_tr)}/{len(ko_dv)}/{len(ko_te)}")


# --- Serialize ----------------------------------------------------------------
def example_to_dict(ex: DialogueExample) -> dict:
    return {
        "id": ex.id,
        "language": ex.language,
        "context_turns": ex.context_turns,
        "relationship": {
            "speaker_to_addressee": asdict(ex.relationship.speaker_to_addressee),
            "speaker_meta": ex.relationship.speaker_meta,
            "addressee_meta": ex.relationship.addressee_meta,
        },
        "candidates": [
            {"text": c["text"], "tensor": asdict(c["tensor"])}
            for c in ex.candidates
        ],
        "gold_index": ex.gold_index,
        "notes": ex.notes,
    }


def dump_split(name: str, exs: list[DialogueExample]):
    path = Path(CFG.data_dir, f"{name}.jsonl")
    with path.open("w") as f:
        for ex in exs:
            f.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")
    return path


paths = {
    "bn_train": dump_split("bn_train", bn_tr),
    "bn_dev":   dump_split("bn_dev",   bn_dv),
    "bn_test":  dump_split("bn_test",  bn_te),
    "hi_train": dump_split("hi_train", hi_tr),
    "hi_dev":   dump_split("hi_dev",   hi_dv),
    "hi_test":  dump_split("hi_test",  hi_te),
    "ko_train": dump_split("ko_train", ko_tr),
    "ko_dev":   dump_split("ko_dev",   ko_dv),
    "ko_test":  dump_split("ko_test",  ko_te),
}
for k, v in paths.items():
    print(f"  saved: {v}")
print("\nPAPER_ARTIFACT: PRANAM-Bench-Mini v0.1 — JSONL files in data/.")
