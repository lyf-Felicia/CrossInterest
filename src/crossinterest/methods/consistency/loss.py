"""Four-view supervised likelihoods with optional conditional Bayes consistency."""

from __future__ import annotations

import torch


VIEW_ORDER = ("a_given_r", "b_given_r", "a_given_rb", "b_given_ra")
ARMS = {"ce": 0.0, "consistent": 1.0}


def dual_loss(logits: torch.Tensor, targets: torch.Tensor, arm: str):
    if arm not in ARMS:
        raise ValueError("unknown dual training arm")
    if logits.ndim != 3 or logits.shape[1] != 4 or logits.shape[0] == 0:
        raise ValueError("expected nonempty [pairs, four views, categories] logits")
    if targets.shape != logits.shape[:1] + (2,) or targets.dtype != torch.long:
        raise ValueError("expected [pairs, A/B] integer targets")
    if (targets < 0).any() or (targets >= logits.shape[-1]).any():
        raise ValueError("target outside category vocabulary")
    if (targets[:, 0] == targets[:, 1]).any():
        raise ValueError("historical A and newly observed B must differ")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite dual logits")
    log_probabilities = logits.float().log_softmax(dim=-1)
    ordered_targets = targets[:, (0, 1, 0, 1)].unsqueeze(-1)
    selected = log_probabilities.gather(-1, ordered_targets).squeeze(-1)
    ce = -selected.mean(dim=-1)
    reverse_lift = selected[:, 2] - selected[:, 0]
    forward_lift = selected[:, 3] - selected[:, 1]
    residual = forward_lift - reverse_lift
    consistency = residual.square()
    loss = (ce + ARMS[arm] * consistency).mean()
    diagnostics = {"ce": ce.detach(), "consistency": consistency.detach(),
                   "forward_lift": forward_lift.detach(),
                   "reverse_lift": reverse_lift.detach()}
    return loss, diagnostics