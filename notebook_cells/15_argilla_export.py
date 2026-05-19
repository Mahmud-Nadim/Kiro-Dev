# =============================================================================
# Cell: Argilla / label-studio export.
# Why: When you scale up, this is the artifact you ship to annotators.
# =============================================================================

def to_argilla_record(ex: dict) -> dict:
    rel = ex["relationship"]
    sa = rel["speaker_to_addressee"]
    sm = rel.get("speaker_meta", {})
    am = rel.get("addressee_meta", {})
    text_block = (
        f"### Context\n"
        f"Speaker: {sm.get('role','?')} (age {sm.get('age','?')})\n"
        f"Addressee: {am.get('role','?')} (age {am.get('age','?')})\n"
        f"Power={sa['power']}, Age={sa['age']}, Intimacy={sa['intimacy']}, "
        f"Formality={sa['formality']}, Kinship={sa['kinship']}, "
        f"DefTarget={sa['deference_target']}\n\n"
        f"### Last user turn\n"
        f"{ex['context_turns'][0]['text'] if ex['context_turns'] else ''}"
    )
    record = {
        "id": ex["id"],
        "text": text_block,
        "metadata": {
            "language": ex["language"],
            "rule_tag": ex["notes"],
            "relationship": rel,
        },
        "fields": {
            "context": text_block,
            **{f"candidate_{i}": c["text"] for i, c in enumerate(ex["candidates"])},
        },
        "questions": [
            {
                "name": "preferred_response",
                "type": "rating",
                "options": [str(i) for i in range(len(ex["candidates"]))],
            },
        ] + [
            {
                "name": f"axis_{ax}",
                "type": "rating" if ax in ORDINAL_AXES or ax == "formality" else "label_selection",
                "options": (
                    [str(v) for v in range(-2, 3)] if ax in ORDINAL_AXES
                    else [str(v) for v in range(0, 5)] if ax == "formality"
                    else list(KINSHIP_VALUES) if ax == "kinship"
                    else list(DEFERENCE_TARGETS)
                ),
            }
            for ax in CFG.axis_names
        ] + [
            {"name": "notes", "type": "text"},
        ],
    }
    return record


export_records = [to_argilla_record(ex) for ex in splits["bn_test"]]
export_path = Path(CFG.data_dir, "argilla_export_bn_test.json")
export_path.write_text(json.dumps(export_records, indent=2, ensure_ascii=False))
print(f"PAPER_ARTIFACT: {export_path}")
print(f"  records: {len(export_records)}")
print("  upload via: rg.log(records, name='pranam_bn_test') after rg.init()")
