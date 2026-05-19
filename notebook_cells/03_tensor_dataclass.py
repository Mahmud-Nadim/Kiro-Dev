# =============================================================================
# Cell: Define the Relational Pragmatic Tensor as a Python dataclass.
# Why: Reviewers asked (in the imagined adversarial review) for a formal,
# code-backed representation. This is it. Every dataset row references it.
# =============================================================================

# --- Axis definitions ----------------------------------------------------------

# Ordinal axes: integer in [-2, +2]. Negative = speaker subordinate to target.
ORDINAL_AXES = ("power", "age", "intimacy")

# Formality is unidirectional 0..4 (0 = casual, 4 = ceremonial).
FORMALITY_LEVELS = (0, 1, 2, 3, 4)

# Kinship is a controlled-vocabulary categorical.
KINSHIP_VALUES = (
    "none",         # non-relative
    "elder_blood",  # parent, uncle, aunt, grandparent
    "elder_inlaw",  # mother-in-law, father-in-law
    "peer_blood",   # sibling, cousin
    "peer_inlaw",   # brother-in-law of equal age
    "younger",      # younger sibling, child, niece/nephew
)

# Deference Target: who the honorific elevates.
DEFERENCE_TARGETS = ("addressee", "referent", "both", "neither")


@dataclass
class PragmaticTensor:
    """A 6-axis pragmatic profile of an utterance / a relationship slot."""
    power: int = 0          # [-2, +2]
    age: int = 0            # [-2, +2]
    intimacy: int = 0       # [-2, +2]
    formality: int = 2      # [0, 4]
    kinship: str = "none"   # KINSHIP_VALUES
    deference_target: str = "addressee"  # DEFERENCE_TARGETS

    def to_vector(self) -> np.ndarray:
        """Numeric vector for ML consumption. Categorical → one-hot."""
        kin_vec = [1.0 if self.kinship == v else 0.0 for v in KINSHIP_VALUES]
        dt_vec = [1.0 if self.deference_target == v else 0.0 for v in DEFERENCE_TARGETS]
        return np.array(
            [self.power, self.age, self.intimacy, self.formality]
            + kin_vec + dt_vec,
            dtype=np.float32,
        )

    def validate(self) -> None:
        assert -2 <= self.power <= 2, f"power out of range: {self.power}"
        assert -2 <= self.age <= 2
        assert -2 <= self.intimacy <= 2
        assert self.formality in FORMALITY_LEVELS
        assert self.kinship in KINSHIP_VALUES
        assert self.deference_target in DEFERENCE_TARGETS


@dataclass
class RelationshipGraph:
    """Speaker -> Addressee -> Referent triangle with edge-level pragmatic
    expectations. The model must produce a response whose tensor matches the
    expected edge profile."""
    speaker_id: str = "S"
    addressee_id: str = "A"
    referent_id: Optional[str] = None  # may be None (no third party)

    # The "expected" tensor for the speaker -> addressee edge.
    speaker_to_addressee: PragmaticTensor = field(default_factory=PragmaticTensor)
    # Optional: speaker -> referent (for honorific-marked third person)
    speaker_to_referent: Optional[PragmaticTensor] = None

    # Demographic metadata (free-form, used for analysis)
    speaker_meta: dict = field(default_factory=dict)
    addressee_meta: dict = field(default_factory=dict)
    referent_meta: dict = field(default_factory=dict)


@dataclass
class DialogueExample:
    """One unit of PRANAM-Bench."""
    id: str
    language: str               # "bn", "hi", "ko"
    context_turns: list         # list of {"speaker": ..., "text": ...}
    relationship: RelationshipGraph
    candidates: list            # list of {"text": str, "tensor": PragmaticTensor}
    gold_index: int = 0         # index of the gold response in candidates
    notes: str = ""             # human / linguist notes

    def expected_tensor(self) -> PragmaticTensor:
        return self.relationship.speaker_to_addressee


print("Relational Pragmatic Tensor defined.")
print(f"Vector dimensionality: "
      f"{len(PragmaticTensor().to_vector())} "
      f"(4 numeric + {len(KINSHIP_VALUES)} kinship one-hot + "
      f"{len(DEFERENCE_TARGETS)} DT one-hot)")
