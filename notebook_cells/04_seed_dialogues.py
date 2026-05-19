# =============================================================================
# Cell: Seed dialogue templates.
# Why: Each template encodes a relationship configuration. We instantiate
# templates with diverse fillers to get a mini dataset that covers the
# relationship space adequately.
# =============================================================================

# --- Bengali templates ---------------------------------------------------------
# Each template has:
#   - a context (1-2 turns of dialogue setting up the scene, in English glosses
#     so this notebook is reviewable by non-Bengali speakers; the actual
#     Bengali surface forms appear in `candidates`)
#   - a relationship graph
#   - a target reply slot the model must fill

BN_TEMPLATES = [
    {
        "tag": "junior_to_senior_office",
        "context_en": "A 25-year-old new employee approaches the 60-year-old company chairman for the first time.",
        "rel": dict(
            speaker_to_addressee=dict(power=-2, age=-2, intimacy=-2, formality=4,
                                      kinship="none", deference_target="addressee"),
            speaker_meta=dict(age=25, role="new employee"),
            addressee_meta=dict(age=60, role="chairman"),
        ),
        "user_prompt_bn": "Sir, ami notun kormi. Apni ki amake ektu somay diben?",
        "user_prompt_en": "Sir, I am a new employee. Could you give me some time?",
        "ideal_reply_en": "Yes, I have time tomorrow at 11. Please come.",
        "candidates_en_axes": [  # (english gloss, axis hints)
            ("apni-form: 'Yes, please come tomorrow at 11.'",
             dict(power=-2, age=-2, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("tumi-form: 'Yes, come tomorrow at 11.'",
             dict(power=-1, age=-1, intimacy=0, formality=2, kinship="none", deference_target="addressee")),
            ("tui-form: 'Yes, come tomorrow at 11, ya.'",
             dict(power=0, age=0, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("hyper-formal Sanskritized: 'Affirmative, your presence is requested at 11 hours tomorrow.'",
             dict(power=-2, age=-2, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "best_friends_chat",
        "context_en": "Two 28-year-old close friends meeting at a tea stall.",
        "rel": dict(
            speaker_to_addressee=dict(power=0, age=0, intimacy=2, formality=0,
                                      kinship="none", deference_target="neither"),
            speaker_meta=dict(age=28, role="friend"),
            addressee_meta=dict(age=28, role="friend"),
        ),
        "user_prompt_bn": "Doctor, kemon achhish? Onek din por dekha.",
        "user_prompt_en": "Hey friend, how are you? Long time no see.",
        "ideal_reply_en": "Bhai I'm great! What about you?",
        "candidates_en_axes": [
            ("tui-form casual: 'Bhai I'm great, you tell me!'",
             dict(power=0, age=0, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("tumi-form mid: 'I'm fine, what about you?'",
             dict(power=0, age=0, intimacy=0, formality=2, kinship="none", deference_target="addressee")),
            ("apni-form distant: 'I am well, thank you. And yourself?'",
             dict(power=-1, age=-1, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("hostile rude: 'Why do you care?'",
             dict(power=1, age=0, intimacy=-2, formality=0, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "younger_to_elder_sister",
        "context_en": "A 22-year-old speaks to her 32-year-old elder sister at home.",
        "rel": dict(
            speaker_to_addressee=dict(power=-1, age=-1, intimacy=2, formality=1,
                                      kinship="elder_blood", deference_target="addressee"),
            speaker_meta=dict(age=22, role="younger sibling"),
            addressee_meta=dict(age=32, role="elder sister"),
        ),
        "user_prompt_bn": "Didi, kheyechho?",
        "user_prompt_en": "Didi, have you eaten?",
        "ideal_reply_en": "Yes, I ate. What about you, my dear sibling?",
        "candidates_en_axes": [
            ("tumi-form with kin: 'Yes I've eaten, you eat too.'",
             dict(power=-1, age=-1, intimacy=2, formality=1, kinship="elder_blood", deference_target="addressee")),
            ("apni-form: 'Yes, I have eaten, ma'am.'",
             dict(power=-2, age=-2, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("tui-form: 'Yeah I've eaten, you eat too.'",
             dict(power=0, age=0, intimacy=2, formality=0, kinship="peer_blood", deference_target="neither")),
            ("only the verb, no kin term: 'I've eaten.'",
             dict(power=0, age=0, intimacy=0, formality=1, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "stranger_asking_directions",
        "context_en": "A 30-year-old asks a 50-year-old stranger for directions.",
        "rel": dict(
            speaker_to_addressee=dict(power=-1, age=-1, intimacy=-2, formality=3,
                                      kinship="none", deference_target="addressee"),
            speaker_meta=dict(age=30, role="lost traveller"),
            addressee_meta=dict(age=50, role="local stranger"),
        ),
        "user_prompt_bn": "Kaka, station kothai?",
        "user_prompt_en": "Uncle, where is the station?",
        "ideal_reply_en": "It's just two streets ahead, please go straight.",
        "candidates_en_axes": [
            ("apni-form polite: 'Please go straight, it's two streets ahead.'",
             dict(power=-1, age=-1, intimacy=-2, formality=3, kinship="none", deference_target="addressee")),
            ("tumi-form: 'Go straight, two streets ahead.'",
             dict(power=0, age=0, intimacy=0, formality=2, kinship="none", deference_target="addressee")),
            ("tui-form: 'Go straight, dude.'",
             dict(power=1, age=1, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("indirect / evasive: 'I don't know.'",
             dict(power=0, age=0, intimacy=-1, formality=2, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "mother_to_adult_son",
        "context_en": "A 60-year-old mother speaks to her 35-year-old son.",
        "rel": dict(
            speaker_to_addressee=dict(power=1, age=2, intimacy=2, formality=1,
                                      kinship="younger", deference_target="neither"),
            speaker_meta=dict(age=60, role="mother"),
            addressee_meta=dict(age=35, role="adult son"),
        ),
        "user_prompt_bn": "Baba, tor ki khabar dorkar?",
        "user_prompt_en": "Son, do you need food?",
        "ideal_reply_en": "Yes Ma, please give me a little bit.",
        "candidates_en_axes": [
            ("apni-form to mother: 'Yes Ma, please give some.'",
             dict(power=-2, age=-2, intimacy=2, formality=2, kinship="elder_blood", deference_target="addressee")),
            ("tumi-form to mother: 'Yes Ma, give a little.'",
             dict(power=-1, age=-2, intimacy=2, formality=1, kinship="elder_blood", deference_target="addressee")),
            ("tui-form (unusual): 'Yeah Ma, give me some.'",
             dict(power=0, age=-1, intimacy=2, formality=0, kinship="elder_blood", deference_target="neither")),
            ("rude refusal: 'No I don't want it.'",
             dict(power=1, age=0, intimacy=-2, formality=0, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "teacher_to_student_classroom",
        "context_en": "A 45-year-old university teacher addresses a 20-year-old student in class.",
        "rel": dict(
            speaker_to_addressee=dict(power=2, age=2, intimacy=-1, formality=3,
                                      kinship="none", deference_target="neither"),
            speaker_meta=dict(age=45, role="university teacher"),
            addressee_meta=dict(age=20, role="undergraduate student"),
        ),
        "user_prompt_bn": "Sir, ami answer-ta jani na.",
        "user_prompt_en": "Sir, I do not know the answer.",
        "ideal_reply_en": "Sit down, study and try again next class.",
        "candidates_en_axes": [
            ("tumi-form, formal classroom: 'Sit down, prepare and try next time.'",
             dict(power=2, age=2, intimacy=-1, formality=3, kinship="none", deference_target="neither")),
            ("apni-form (overly formal toward student): 'Please sit, you may try later.'",
             dict(power=0, age=0, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("tui-form (too informal in classroom): 'Sit down, study next time.'",
             dict(power=2, age=2, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("dismissive rude: 'You always fail.'",
             dict(power=2, age=2, intimacy=-2, formality=2, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "patient_to_doctor",
        "context_en": "A 40-year-old patient consults a 38-year-old doctor.",
        "rel": dict(
            speaker_to_addressee=dict(power=-2, age=0, intimacy=-2, formality=4,
                                      kinship="none", deference_target="addressee"),
            speaker_meta=dict(age=40, role="patient"),
            addressee_meta=dict(age=38, role="doctor"),
        ),
        "user_prompt_bn": "Daktar shahab, amar matha betha kore.",
        "user_prompt_en": "Doctor sir, I have a headache.",
        "ideal_reply_en": "Please describe when it started.",
        "candidates_en_axes": [
            ("apni-form professional: 'Please describe when this began.'",
             dict(power=-2, age=0, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("tumi-form (inappropriate to patient): 'Tell me when it started.'",
             dict(power=-1, age=0, intimacy=0, formality=2, kinship="none", deference_target="addressee")),
            ("tui-form (rude): 'Just tell me already.'",
             dict(power=1, age=0, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("evasive: 'Hmm, hard to say.'",
             dict(power=0, age=0, intimacy=-1, formality=2, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "vendor_to_regular_customer",
        "context_en": "A 50-year-old vendor speaks to a regular 35-year-old customer.",
        "rel": dict(
            speaker_to_addressee=dict(power=0, age=1, intimacy=1, formality=2,
                                      kinship="none", deference_target="addressee"),
            speaker_meta=dict(age=50, role="vendor"),
            addressee_meta=dict(age=35, role="regular customer"),
        ),
        "user_prompt_bn": "Vai, dam koto?",
        "user_prompt_en": "Brother, what is the price?",
        "ideal_reply_en": "For you, just 100 taka, brother.",
        "candidates_en_axes": [
            ("vendor friendliness, tumi-form: 'For you 100 taka, dada.'",
             dict(power=0, age=1, intimacy=1, formality=2, kinship="none", deference_target="addressee")),
            ("apni-form formal: 'For you, sir, 100 taka.'",
             dict(power=-1, age=0, intimacy=-1, formality=3, kinship="none", deference_target="addressee")),
            ("tui-form rude: '100 taka, take it or leave it.'",
             dict(power=1, age=1, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("indirect upcharge: 'It's expensive today.'",
             dict(power=0, age=0, intimacy=0, formality=2, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
]

# Hindi parallels — fewer because Hindi is the secondary language for transfer.
HI_TEMPLATES = [
    {
        "tag": "hi_junior_to_senior",
        "context_en": "A junior employee speaks to a senior manager.",
        "rel": dict(
            speaker_to_addressee=dict(power=-2, age=-2, intimacy=-2, formality=4,
                                      kinship="none", deference_target="addressee"),
            speaker_meta=dict(age=25, role="junior"),
            addressee_meta=dict(age=55, role="manager"),
        ),
        "user_prompt_en": "Sir, may I have an appointment?",
        "ideal_reply_en": "aap kal subah aa jaaiye (please come tomorrow morning).",
        "candidates_en_axes": [
            ("aap-form (correct): 'aap kal subah aa jaaiye'",
             dict(power=-2, age=-2, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("tum-form: 'tum kal subah aa jana'",
             dict(power=-1, age=-1, intimacy=0, formality=2, kinship="none", deference_target="addressee")),
            ("tu-form: 'tu kal subah aa'",
             dict(power=1, age=0, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("dismissive: 'I am busy.'",
             dict(power=1, age=0, intimacy=-2, formality=2, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "hi_close_friends",
        "context_en": "Two close friends.",
        "rel": dict(
            speaker_to_addressee=dict(power=0, age=0, intimacy=2, formality=0,
                                      kinship="none", deference_target="neither"),
            speaker_meta=dict(age=28, role="friend"),
            addressee_meta=dict(age=28, role="friend"),
        ),
        "user_prompt_en": "Hey buddy, what's up?",
        "ideal_reply_en": "tu bata kya haal (you tell me, what's up)",
        "candidates_en_axes": [
            ("tu-form: 'tu bata kya haal'",
             dict(power=0, age=0, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("tum-form: 'tum batao'",
             dict(power=0, age=0, intimacy=0, formality=2, kinship="none", deference_target="addressee")),
            ("aap-form: 'aap bataaiye'",
             dict(power=-1, age=-1, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("rude: 'why do you care?'",
             dict(power=1, age=0, intimacy=-2, formality=0, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
]

# Korean parallels — even fewer; we want to demonstrate the SAME 6 axes
# capture honorific behavior in a typologically different language.
KO_TEMPLATES = [
    {
        "tag": "ko_junior_to_senior",
        "context_en": "A 25-year-old junior to a 60-year-old executive.",
        "rel": dict(
            speaker_to_addressee=dict(power=-2, age=-2, intimacy=-2, formality=4,
                                      kinship="none", deference_target="addressee"),
            speaker_meta=dict(age=25, role="junior"),
            addressee_meta=dict(age=60, role="executive"),
        ),
        "user_prompt_en": "Sir, may I ask a question?",
        "ideal_reply_en": "hapsyo-che (most formal): yes please ask.",
        "candidates_en_axes": [
            ("hapsyo-che -seumnida ending (correct): 'ne, mal-sseumhae jusip-syo'",
             dict(power=-2, age=-2, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("haeyo-che -yo ending: 'ne, malhae juseyo'",
             dict(power=-1, age=-1, intimacy=0, formality=3, kinship="none", deference_target="addressee")),
            ("panmal: 'eo, malhae'",
             dict(power=1, age=1, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("dismissive: 'I'm busy.'",
             dict(power=1, age=0, intimacy=-2, formality=2, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
    {
        "tag": "ko_close_friends",
        "context_en": "Two close friends in their 20s.",
        "rel": dict(
            speaker_to_addressee=dict(power=0, age=0, intimacy=2, formality=0,
                                      kinship="none", deference_target="neither"),
            speaker_meta=dict(age=25, role="friend"),
            addressee_meta=dict(age=25, role="friend"),
        ),
        "user_prompt_en": "What's up dude?",
        "ideal_reply_en": "panmal: yeah how's it going",
        "candidates_en_axes": [
            ("panmal (correct): 'eung, jal jinae?'",
             dict(power=0, age=0, intimacy=2, formality=0, kinship="none", deference_target="neither")),
            ("haeyo-che: 'ne, jal jinae-yo'",
             dict(power=0, age=0, intimacy=0, formality=3, kinship="none", deference_target="addressee")),
            ("hapsyo-che (overly formal): 'ne, jal jinaem-nida'",
             dict(power=-1, age=-1, intimacy=-2, formality=4, kinship="none", deference_target="addressee")),
            ("rude: 'why are you asking?'",
             dict(power=1, age=0, intimacy=-2, formality=0, kinship="none", deference_target="neither")),
        ],
        "gold_index": 0,
    },
]

print(f"Bengali templates: {len(BN_TEMPLATES)}")
print(f"Hindi templates:   {len(HI_TEMPLATES)}")
print(f"Korean templates:  {len(KO_TEMPLATES)}")
