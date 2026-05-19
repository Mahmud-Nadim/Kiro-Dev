# =============================================================================
# Cell: Worked examples that illustrate the tensor.
# Why: Sanity-check that our representation captures real Bengali distinctions.
# These examples become Figure 1 of the paper.
# =============================================================================

EXAMPLE_SCENARIOS = [
    {
        "label": "Young employee → Elderly stranger boss",
        "rel": RelationshipGraph(
            speaker_to_addressee=PragmaticTensor(
                power=-2, age=-2, intimacy=-2, formality=4,
                kinship="none", deference_target="addressee",
            ),
            speaker_meta={"age": 25, "role": "junior employee"},
            addressee_meta={"age": 65, "role": "CEO"},
        ),
        "expected_pronoun": "apni",
        "ungrammatical_choice": "tui",  # would be a serious cultural offense
    },
    {
        "label": "Two close friends, same age",
        "rel": RelationshipGraph(
            speaker_to_addressee=PragmaticTensor(
                power=0, age=0, intimacy=2, formality=0,
                kinship="none", deference_target="neither",
            ),
            speaker_meta={"age": 28, "role": "friend"},
            addressee_meta={"age": 28, "role": "friend"},
        ),
        "expected_pronoun": "tui",
        "ungrammatical_choice": "apni",  # would feel cold/distant
    },
    {
        "label": "Speaker addressing their elder sister",
        "rel": RelationshipGraph(
            speaker_to_addressee=PragmaticTensor(
                power=-1, age=-1, intimacy=2, formality=1,
                kinship="elder_blood", deference_target="addressee",
            ),
            speaker_meta={"age": 22, "role": "younger sibling"},
            addressee_meta={"age": 32, "role": "elder sister"},
        ),
        "expected_pronoun": "tumi (with 'didi' kinship form)",
        "ungrammatical_choice": "apni",  # too distant for blood-kin elder
    },
    {
        "label": "Speaker referring to a respected absent third person",
        "rel": RelationshipGraph(
            referent_id="R",
            speaker_to_addressee=PragmaticTensor(
                power=0, age=0, intimacy=1, formality=2,
            ),
            speaker_to_referent=PragmaticTensor(
                power=-2, age=-2, intimacy=-1, formality=3,
                kinship="none", deference_target="referent",
            ),
            speaker_meta={"role": "speaker"},
            addressee_meta={"role": "peer"},
            referent_meta={"role": "respected scholar (third person)"},
        ),
        "expected_pronoun": "addressee=tumi; verb form for referent uses honorific -en suffix",
        "ungrammatical_choice": "non-honorific verb for referent",
    },
]

# Display
for ex in EXAMPLE_SCENARIOS:
    print(f"\n=== {ex['label']} ===")
    print(f"  Expected: {ex['expected_pronoun']}")
    print(f"  Wrong:    {ex['ungrammatical_choice']}")
    t = ex["rel"].speaker_to_addressee
    print(f"  Tensor: P={t.power}, A={t.age}, I={t.intimacy}, "
          f"F={t.formality}, K={t.kinship}, DT={t.deference_target}")

# Save examples for later use as Figure 1 source.
Path(CFG.data_dir, "figure1_scenarios.json").write_text(
    json.dumps(
        [
            {
                "label": ex["label"],
                "tensor": asdict(ex["rel"].speaker_to_addressee),
                "expected": ex["expected_pronoun"],
                "wrong": ex["ungrammatical_choice"],
            }
            for ex in EXAMPLE_SCENARIOS
        ],
        indent=2,
    )
)
print("\nPAPER_ARTIFACT: figure1_scenarios.json saved.")
