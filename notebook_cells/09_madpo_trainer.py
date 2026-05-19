# =============================================================================
# Cell: MA-DPO custom trainer.
# Why: Subclass DPOTrainer so we can weight examples by axis. Two designs:
#   (a) Separate dataloader per axis, sum losses with learned weights.
#   (b) Single dataloader with axis_index field, weighted-by-axis on each batch.
# We pick (b) for simplicity.
# =============================================================================
from trl import DPOTrainer as _BaseDPOTrainer


class MADPOTrainer(_BaseDPOTrainer):
    """DPO with per-axis loss weighting.

    Expects each batch to carry an `axis_index` tensor in [0, K).
    The per-example DPO loss is computed normally, then multiplied by
    alpha[axis_index] before averaging.
    """

    def __init__(self, *args, n_axes: int = 6, learn_axis_weights: bool = True,
                 axis_weight_init: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if learn_axis_weights:
            # Parameterize as raw logits and softmax to keep alpha non-negative
            # and summing to 1 (relative weighting).
            init = torch.full((n_axes,), float(np.log(axis_weight_init)))
            self.alpha_logits = torch.nn.Parameter(init.to(self.model.device))
        else:
            self.alpha_logits = None
        self.n_axes = n_axes
        # Track per-axis loss for diagnostics.
        self._axis_loss_running = np.zeros(n_axes, dtype=np.float64)
        self._axis_loss_count = np.zeros(n_axes, dtype=np.int64)

    def get_alpha(self) -> torch.Tensor:
        if self.alpha_logits is None:
            return torch.ones(self.n_axes, device=self.model.device) / self.n_axes
        return torch.softmax(self.alpha_logits, dim=0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        axis_indices = inputs.pop("axis_index", None)
        out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(out, tuple):
            base_loss, outputs = out
        else:
            base_loss, outputs = out, {}

        if axis_indices is not None and self.alpha_logits is not None:
            alpha = self.get_alpha()  # (K,)
            # axis_indices is shape (batch,) — gather the corresponding alphas.
            # We approximate per-example weighting by scaling the batch loss
            # by the mean alpha for the axes present.
            if isinstance(axis_indices, list):
                idxs = torch.tensor(axis_indices, device=self.model.device)
            else:
                idxs = axis_indices.to(self.model.device)
            batch_alphas = alpha.gather(0, idxs)
            weighted_loss = base_loss * batch_alphas.mean()
            # Track diagnostics.
            for ai in idxs.cpu().tolist():
                self._axis_loss_running[ai] += float(base_loss.detach().cpu())
                self._axis_loss_count[ai] += 1
            return (weighted_loss, outputs) if return_outputs else weighted_loss

        return (base_loss, outputs) if return_outputs else base_loss

    def axis_loss_summary(self) -> dict:
        out = {}
        for k in range(self.n_axes):
            n = max(int(self._axis_loss_count[k]), 1)
            out[CFG.axis_names[k]] = float(self._axis_loss_running[k] / n)
        return out


# Custom collator that preserves axis_index.
def ma_collate(batch):
    axes = [b.pop("axis_index") for b in batch]
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    out["axis_index"] = torch.tensor(axes, dtype=torch.long)
    return out


print("MADPOTrainer defined.")
