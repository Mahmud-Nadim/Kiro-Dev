# =============================================================================
# Cell: Candidate response generator.
# Why: Each template defines 4 candidate responses with different pragmatic
# profiles. Here we expand each template into multiple instances by varying
# names, locations, times. This is the dialogue-multiplication step.
# =============================================================================

# Filler banks for instantiation.
BN_NAMES = ["Rahim", "Karim", "Sumi", "Mou", "Tania", "Riad", "Sajid", "Faria",
            "Mahmud", "Tasmin", "Asif", "Nilima", "Saif", "Shimul"]
BN_PLACES = ["Dhaka", "Chittagong", "Sylhet", "Barishal", "Rajshahi", "Khulna",
             "Mymensingh"]
BN_TIMES = ["morning", "noon", "afternoon", "evening", "night"]

HI_NAMES = ["Rahul", "Priya", "Amit", "Neha", "Vikram", "Sunita"]
KO_NAMES = ["Min-jun", "Ji-woo", "Seo-yeon", "Hyun-woo", "Soo-jin"]


def _instantiate(template: dict, language: str, seed: int) -> dict:
    """Apply random fillers to a template to create one concrete dialogue.
    Deterministic given seed; we use seed to vary name slot only."""
    rng = random.Random(seed)
    if language == "bn":
        names = BN_NAMES
        places = BN_PLACES
    elif language == "hi":
        names = HI_NAMES
        places = ["Delhi", "Mumbai", "Kolkata", "Bangalore"]
    elif language == "ko":
        names = KO_NAMES
        places = ["Seoul", "Busan", "Incheon"]
    else:
        raise ValueError(language)

    instance = {
        "tag": template["tag"],
        "context_en": template["context_en"],
        "rel": template["rel"],
        "user_prompt_en": template.get("user_prompt_en", ""),
        "user_prompt_bn": template.get("user_prompt_bn", ""),
        "ideal_reply_en": template["ideal_reply_en"],
        "candidates_en_axes": template["candidates_en_axes"],
        "gold_index": template["gold_index"],
        "filler_name": rng.choice(names),
        "filler_place": rng.choice(places),
        "filler_time": rng.choice(BN_TIMES),
        "language": language,
    }
    return instance


def expand_templates(templates: list, language: str, target_n: int) -> list:
    """Round-robin over templates, instantiating with different seeds."""
    expanded = []
    i = 0
    while len(expanded) < target_n:
        tpl = templates[i % len(templates)]
        instance = _instantiate(tpl, language, seed=i)
        expanded.append(instance)
        i += 1
    return expanded


bn_instances = expand_templates(BN_TEMPLATES, "bn", CFG.n_dialogues_bn)
hi_instances = expand_templates(HI_TEMPLATES, "hi", CFG.n_dialogues_hi)
ko_instances = expand_templates(KO_TEMPLATES, "ko", CFG.n_dialogues_ko)

print(f"Instantiated dialogues: bn={len(bn_instances)}, hi={len(hi_instances)}, ko={len(ko_instances)}")
print("\nExample instantiation (Bengali):")
print(json.dumps({k: v for k, v in bn_instances[0].items() if k != "candidates_en_axes"}, indent=2))
