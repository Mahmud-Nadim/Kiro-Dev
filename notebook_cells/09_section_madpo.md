---

## Section 9 — MA-DPO: Multi-Axis Direct Preference Optimization (Proposed Method)

This is the methodological contribution of the paper. The standard DPO loss:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

becomes, in MA-DPO:

$$\mathcal{L}_{\text{MA-DPO}} = -\sum_{k=1}^{K} \alpha_k \cdot \mathbb{E}_{(x, g, y_w^{(k)}, y_l^{(k)})}\left[\log \sigma\left(\beta_k \log \frac{\pi_\theta(y_w^{(k)}|x, g)}{\pi_\text{ref}(y_w^{(k)}|x, g)} - \beta_k \log \frac{\pi_\theta(y_l^{(k)}|x, g)}{\pi_\text{ref}(y_l^{(k)}|x, g)}\right)\right]$$

where:
- $K=6$ pragmatic axes
- $y_w^{(k)}, y_l^{(k)}$ is the winner/loser pair *along axis k* (constructed from the same example by sorting candidates on that axis)
- $g$ is the relational graph condition (concatenated to the prompt)
- $\alpha_k$ is the axis weight (learned via meta-objective on dev set)

**Three innovation hooks** the paper highlights:

1. **Axis-conditional preferences**: the same example yields up to 6 preference signals
2. **Conflict-aware sampling**: when two axes disagree, we explicitly oversample to teach trade-off navigation
3. **Relational graph conditioning**: serialized into the prompt as a structured directive, lightweight and adapter-friendly

The implementation below subclasses TRL's DPOTrainer to add the multi-axis loss aggregation.
