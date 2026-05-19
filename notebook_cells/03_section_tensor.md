---

## Section 3 — The Relational Pragmatic Tensor

This is the conceptual contribution of the paper. The claim is that honorific behavior in Bengali (and other honorific-rich languages) is *not* a scalar "politeness" dimension. It is a 6-axis structured representation:

| Axis | Symbol | What it captures | Bengali surface markers |
|---|---|---|---|
| Power | P | Hierarchical authority | apni > tumi > tui; -ji suffix |
| Age | A | Age differential | dada / didi / kaka kinship terms; verb honorific morphology |
| Intimacy | I | Solidarity / closeness | tui (intimate) vs apni (distant); diminutives |
| Formality | F | Setting register | sadhu vs cholito bhasha; Sanskritized lexicon |
| Kinship | K | Family / non-family role | use of relational kin vs name |
| Deference Target | DT | Whom the honorific elevates (addressee, referent, both) | third-person honorific verb forms |

A correct response in Bengali requires the model to choose the response whose pragmatic profile *matches* the relationship graph between speaker, addressee, and referent. Existing alignment methods compress this into a single preference — they will reliably pick the *most polite* response, which is wrong when the addressee is a young intimate friend.

The dataclass below operationalizes this for code. We use ordinal numeric scores [-2, +2] for the gradient axes (P, A, I), a 0–4 scale for Formality, and categorical for Kinship and Deference Target. We use the same six axes across Bengali / Hindi / Korean — the *values* differ but the typology holds.
