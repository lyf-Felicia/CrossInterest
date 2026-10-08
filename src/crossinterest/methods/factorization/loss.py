"""Factorized CIR supervision with matched first-ID gradient magnitude."""

from __future__ import annotations

import torch
from torch.nn import functional as F


ARMS = {
    "baseline": (1.0, 1.0, False),
    "relation_true": (4.0 / 3.0, 2.0 / 3.0, False),
    "relation_pseudo": (4.0 / 3.0, 2.0 / 3.0, True),
    "exclusion_true": (2.0 / 3.0, 4.0 / 3.0, False),
    "exclusion_pseudo": (2.0 / 3.0, 4.0 / 3.0, True),
}


def factorized_loss(logits: torch.Tensor, positive_mask: torch.Tensor,
                    seen_mask: torch.Tensor, arm: str) -> tuple[torch.Tensor, dict]:
    alpha, beta, _ = ARMS[arm]
    if logits.ndim != 2 or logits.shape != positive_mask.shape or logits.shape != seen_mask.shape:
        raise ValueError("factor masks must align with batch by label logits")
    if positive_mask.dtype != torch.bool or seen_mask.dtype != torch.bool:
        raise ValueError("factor masks must be boolean")
    if not (positive_mask.sum(-1) == 1).all() or (positive_mask & seen_mask).any():
        raise ValueError("each CIR target must be unique and unseen")
    scores = logits if logits.dtype == torch.float64 else logits.float()
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite factor logits")
    targets = positive_mask.to(scores.dtype)
    log_probability = F.log_softmax(scores, dim=-1)
    original = -(targets * log_probability).sum(-1)
    unseen_scores = scores.masked_fill(seen_mask, float("-inf"))
    all_partition = torch.logsumexp(scores, dim=-1)
    unseen_partition = torch.logsumexp(unseen_scores, dim=-1)
    target_scores = (targets * scores).sum(-1)
    relation = unseen_partition - target_scores
    exclusion = all_partition - unseen_partition
    with torch.no_grad():
        probability = log_probability.exp()
        conditional = F.softmax(unseen_scores, dim=-1)
        relation_gradient = conditional - targets
        exclusion_gradient = probability - conditional
        original_gradient = probability - targets
        proposed_gradient = alpha * relation_gradient + beta * exclusion_gradient
        original_norm = original_gradient.norm(dim=-1)
        proposed_norm = proposed_gradient.norm(dim=-1)
        fallback = (proposed_norm <= 1e-12) | (original_norm <= 1e-12)
        scale = torch.where(fallback, torch.ones_like(original_norm),
                            original_norm / proposed_norm.clamp_min(1e-12))
        effective = torch.where(fallback[:, None], original_gradient,
                                scale[:, None] * proposed_gradient)
        direction_delta = (effective - original_gradient).norm(dim=-1) / original_norm.clamp_min(1e-12)
        cosine = F.cosine_similarity(effective, original_gradient, dim=-1, eps=1e-12)
        diagnostics = {
            "relation_loss": relation.detach(), "exclusion_loss": exclusion.detach(),
            "original_loss": original.detach(), "seen_mass": (probability * seen_mask).sum(-1),
            "relation_gradient_norm": relation_gradient.norm(dim=-1),
            "exclusion_gradient_norm": exclusion_gradient.norm(dim=-1),
            "original_gradient_norm": original_norm, "scale": scale,
            "direction_delta": direction_delta, "gradient_cosine": cosine,
            "fallback": fallback, "seen_count": seen_mask.sum(-1),
        }
    if arm == "baseline":
        loss = original.mean()
    else:
        loss = torch.where(fallback, original, scale * (alpha * relation + beta * exclusion)).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite factor loss")
    return loss, diagnostics