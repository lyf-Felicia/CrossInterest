"""Numerically stable SuperLoss confidence weights for sequence-level losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def answer_logits_to_keep(labels: torch.Tensor) -> int:
    """Return the suffix length needed to predict every supervised token."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    supervised = labels.ne(-100)
    if torch.any(~supervised.any(dim=1)):
        raise ValueError("batch contains a sequence without answer tokens")
    first_supervised = supervised.to(torch.int64).argmax(dim=1)
    return min(
        labels.shape[1],
        labels.shape[1] - int(first_supervised.min().item()) + 1,
    )


def sequence_answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute one mean next-token NLL per sequence over non-masked answer tokens."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "logits and labels must have shapes [batch, sequence, vocab] and [batch, sequence]"
        )
    if logits.shape[0] != labels.shape[0] or logits.shape[1] > labels.shape[1]:
        raise ValueError("logits and labels have incompatible shapes")
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, -(logits.shape[1] - 1) :]
    mask = shifted_labels.ne(-100)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("batch contains a sequence without answer tokens")
    token_losses = functional.cross_entropy(
        shifted_logits[mask].float(),
        shifted_labels[mask],
        reduction="none",
    )
    sequence_ids = mask.nonzero(as_tuple=False)[:, 0]
    totals = torch.zeros(logits.shape[0], dtype=token_losses.dtype, device=logits.device)
    totals.scatter_add_(0, sequence_ids, token_losses)
    return totals / counts


def principal_lambert_w(values: torch.Tensor, iterations: int = 48) -> torch.Tensor:
    """Evaluate the real principal Lambert-W branch with monotone bisection."""
    branch_point = -1.0 / math.e
    if torch.any(values < branch_point):
        raise ValueError("principal Lambert-W requires values >= -1/e")

    lower = torch.where(values < 0, torch.full_like(values, -1.0), torch.zeros_like(values))
    upper = torch.where(
        values < 0,
        torch.zeros_like(values),
        torch.maximum(torch.ones_like(values), torch.log1p(values)),
    )
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        midpoint_value = midpoint * torch.exp(midpoint)
        lower = torch.where(midpoint_value < values, midpoint, lower)
        upper = torch.where(midpoint_value < values, upper, midpoint)
    return (lower + upper) / 2


@torch.no_grad()
def superloss_confidence(
    sample_losses: torch.Tensor,
    threshold: torch.Tensor | float,
    regularization: float,
    minimum: float = 0.1,
) -> torch.Tensor:
    """Return detached SuperLoss confidence weights from Castells et al. (2020)."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be in [0, 1]")
    threshold_tensor = torch.as_tensor(
        threshold, dtype=sample_losses.dtype, device=sample_losses.device
    )
    clipped = torch.clamp(
        (sample_losses.detach() - threshold_tensor) / regularization,
        min=-2.0 / math.e,
    )
    confidence = torch.exp(-principal_lambert_w(0.5 * clipped))
    return confidence.clamp_min(minimum)