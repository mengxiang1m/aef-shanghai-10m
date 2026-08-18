from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _elementwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    name: str,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if name == "bce":
        return F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    if name == "balanced_bce":
        valid = torch.ones_like(target, dtype=torch.bool) if mask is None else mask.bool()
        positives = target[valid].sum()
        negatives = valid.sum().to(target.dtype) - positives
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0)
        return F.binary_cross_entropy_with_logits(
            prediction, target, reduction="none", pos_weight=positive_weight
        )
    if name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="none")
    if name == "l1":
        return F.l1_loss(prediction, target, reduction="none")
    if name == "mse":
        return F.mse_loss(prediction, target, reduction="none")
    raise ValueError(f"Unknown loss '{name}'")


def _regrid(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, spacing: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if spacing <= 1:
        return prediction, target, mask
    mask_float = mask.to(prediction.dtype)
    denominator = F.avg_pool2d(mask_float, spacing, spacing, ceil_mode=True)
    prediction = F.avg_pool2d(prediction * mask_float, spacing, spacing, ceil_mode=True)
    target = F.avg_pool2d(target * mask_float, spacing, spacing, ceil_mode=True)
    prediction = prediction / denominator.clamp_min(1e-6)
    target = target / denominator.clamp_min(1e-6)
    return prediction, target, denominator > 0


def _shift_invariant_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_name: str,
    radius: int,
) -> torch.Tensor:
    losses = []
    height, width = prediction.shape[-2:]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pred_y = slice(max(0, dy), min(height, height + dy))
            pred_x = slice(max(0, dx), min(width, width + dx))
            target_y = slice(max(0, -dy), min(height, height - dy))
            target_x = slice(max(0, -dx), min(width, width - dx))
            shifted_prediction = prediction[..., pred_y, pred_x]
            shifted_target = target[..., target_y, target_x]
            shifted_mask = mask[..., target_y, target_x]
            losses.append(
                masked_mean(
                    _elementwise_loss(
                        shifted_prediction, shifted_target, loss_name, shifted_mask
                    ),
                    shifted_mask,
                )
            )
    return torch.stack(losses).min()


def reconstruction_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    layer_specs: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {}
    weighted = []
    weights = []
    for name, prediction in predictions.items():
        target = targets[name]
        mask = masks[name]
        loss_name = layer_specs[name]["loss"]
        prediction, target, mask = _regrid(
            prediction, target, mask, int(layer_specs[name].get("regrid_pixels", 1))
        )
        if not bool(mask.any()):
            losses[name] = prediction.sum() * 0.0
            continue
        shift_pixels = int(layer_specs[name].get("shift_pixels", 0))
        if shift_pixels > 0:
            losses[name] = _shift_invariant_loss(
                prediction, target, mask, loss_name, shift_pixels
            )
        else:
            losses[name] = masked_mean(
                _elementwise_loss(prediction, target, loss_name, mask), mask
            )
        weight = float(layer_specs[name].get("weight", 1.0))
        weighted.append(weight * losses[name])
        weights.append(weight)
    if weighted:
        total = torch.stack(weighted).sum() / max(sum(weights), 1e-12)
    else:
        total = sum(prediction.sum() * 0.0 for prediction in predictions.values())
    return total, losses


def batch_uniformity_loss(embedding: torch.Tensor) -> torch.Tensor:
    """AEF batch-rotation orthogonality objective over dense pixel embeddings."""
    if embedding.shape[0] < 2:
        return embedding.new_zeros(())
    paired = embedding.roll(1, dims=0)
    return embedding.mul(paired).sum(dim=1).abs().mean()


def consistency_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Shared-parameter teacher/student agreement from AEF equation (5)."""
    return ((1.0 - teacher.mul(student).sum(dim=1)) * 0.5).mean()
