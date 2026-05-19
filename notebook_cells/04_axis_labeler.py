# =============================================================================
# Cell: Rubric-based axis labeler.
# Why: For the seed dataset we need axis labels without human annotators.
# We use the explicit axis hints inside each candidate (defined by the
# linguist-author at template time). In the full pipeline, this cell is
# replaced by the Argilla export + native-speaker annotators.
# =============================================================================

def axis_labels_from_hint(axis_hint: dict) -> PragmaticTensor:
    """Convert a candidate's axis-hint dict into a PragmaticTensor."""
    t = PragmaticTensor(
        power=axis_hint.get("power", 0),
        age=axis_hint.get("age", 0),
        intimacy=axis_hint.get("intimacy", 0),
        formality=axis_hint.get("formality", 2),
        kinship=axis_hint.get("kinship", "none"),
        deference_target=axis_hint.get("deference_target", "addressee"),
    )
    t.validate()
    return t


def axis_distance(t1: PragmaticTensor, t2: PragmaticTensor) -> float:
    """L1 distance between tensors. Used to score candidate fitness."""
    d = (
        abs(t1.power - t2.power)
        + abs(t1.age - t2.age)
        + abs(t1.intimacy - t2.intimacy)
        + abs(t1.formality - t2.formality)
        + (0.0 if t1.kinship == t2.kinship else 1.0)
        + (0.0 if t1.deference_target == t2.deference_target else 1.0)
    )
    return float(d)


def axiswise_correctness(candidate_tensor: PragmaticTensor,
                         expected: PragmaticTensor) -> dict:
    """Per-axis 1/0 correctness — used by the Axis-Accuracy metric."""
    return {
        "power": int(candidate_tensor.power == expected.power),
        "age": int(candidate_tensor.age == expected.age),
        "intimacy": int(candidate_tensor.intimacy == expected.intimacy),
        "formality": int(candidate_tensor.formality == expected.formality),
        "kinship": int(candidate_tensor.kinship == expected.kinship),
        "deference_target": int(candidate_tensor.deference_target == expected.deference_target),
    }


def axiswise_softscore(candidate_tensor: PragmaticTensor,
                       expected: PragmaticTensor) -> dict:
    """Soft per-axis score for ordinal axes: 1 - |delta|/4."""
    def _ord(c, e):
        return max(0.0, 1.0 - abs(c - e) / 4.0)
    return {
        "power": _ord(candidate_tensor.power, expected.power),
        "age": _ord(candidate_tensor.age, expected.age),
        "intimacy": _ord(candidate_tensor.intimacy, expected.intimacy),
        "formality": _ord(candidate_tensor.formality, expected.formality),
        "kinship": float(candidate_tensor.kinship == expected.kinship),
        "deference_target": float(candidate_tensor.deference_target == expected.deference_target),
    }

# Quick sanity check.
t_a = PragmaticTensor(power=-2, age=-2, intimacy=-2, formality=4)
t_b = PragmaticTensor(power=0, age=0, intimacy=2, formality=0)
print("Distance(formal-elder vs casual-friend):", axis_distance(t_a, t_b))
print("Soft score:", axiswise_softscore(t_a, t_b))
