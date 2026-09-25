"""Detached continuation-utility targets used by the StreamRefine stop head."""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _as_bk(values: Tensor, name: str) -> tuple[Tensor, bool]:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    squeezed = values.ndim == 1
    if squeezed:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError(f"{name} must have shape [B,K] or [K]")
    if not values.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    return (values, squeezed)


def _positive_scale(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _prefix_observation_mask(mask: Optional[Tensor], reference: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if not isinstance(mask, Tensor) or mask.shape != reference.shape:
        raise ValueError("observed_mask must match the [B,K] loss tensor")
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.shape[1] > 1 and bool((~mask[:, :-1] & mask[:, 1:]).any()):
        raise ValueError("observed_mask must be prefix-contiguous for every trajectory")
    if not bool(mask[:, 0].all()):
        raise ValueError("every trajectory must observe its first refinement state")
    return mask


def continuation_benefit_targets(
    translation_losses: Tensor,
    *,
    translation_scale: float,
    anatomy_losses: Optional[Tensor] = None,
    anatomy_scale: Optional[float] = None,
    lambda_anatomy: float = 0.0,
    lambda_compute: float = 0.0,
    observed_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute the approved best-future continuation target ``B*_k``."""
    translation, squeezed = _as_bk(translation_losses, "translation_losses")
    translation = translation.detach()
    scale_l = _positive_scale(translation_scale, "translation_scale")
    coefficient_a = float(lambda_anatomy)
    coefficient_c = float(lambda_compute)
    if not math.isfinite(coefficient_a) or coefficient_a < 0.0:
        raise ValueError("lambda_anatomy must be finite and >= 0")
    if not math.isfinite(coefficient_c) or coefficient_c < 0.0:
        raise ValueError("lambda_compute must be finite and >= 0")
    anatomy: Optional[Tensor]
    scale_a: Optional[float]
    if coefficient_a != 0.0:
        if anatomy_losses is None or anatomy_scale is None:
            raise ValueError(
                "anatomy losses and anatomy_scale are required when lambda_anatomy > 0"
            )
        anatomy, anatomy_squeezed = _as_bk(anatomy_losses, "anatomy_losses")
        if anatomy_squeezed != squeezed or anatomy.shape != translation.shape:
            raise ValueError("anatomy_losses must match translation_losses")
        anatomy = anatomy.detach().to(
            device=translation.device, dtype=translation.dtype
        )
        scale_a = _positive_scale(anatomy_scale, "anatomy_scale")
    else:
        anatomy = None
        scale_a = None
    normalized_observed_mask = observed_mask
    if squeezed and isinstance(observed_mask, Tensor) and (observed_mask.ndim == 1):
        normalized_observed_mask = observed_mask.unsqueeze(0)
    observed = _prefix_observation_mask(normalized_observed_mask, translation)
    target = torch.zeros_like(translation)
    num_states = translation.shape[1]
    for current in range(num_states - 1):
        candidate_indices = torch.arange(
            current + 1, num_states, device=translation.device, dtype=torch.long
        )
        delta_steps = candidate_indices.to(translation.dtype) - float(current)
        gain = (
            translation[:, current : current + 1] - translation[:, current + 1 :]
        ) / scale_l
        if anatomy is not None and scale_a is not None:
            anatomy_increase = (
                anatomy[:, current + 1 :] - anatomy[:, current : current + 1]
            )
            gain = gain - coefficient_a * anatomy_increase / scale_a
        gain = gain - coefficient_c * delta_steps.unsqueeze(0)
        valid_future = observed[:, current + 1 :] & observed[:, current : current + 1]
        masked_gain = gain.masked_fill(~valid_future, -torch.inf)
        has_future = valid_future.any(dim=1)
        best = masked_gain.max(dim=1).values
        target[:, current] = torch.where(has_future, best, torch.zeros_like(best))
    target = torch.where(observed, target, torch.zeros_like(target)).detach()
    return target.squeeze(0) if squeezed else target


def smooth_l1_benefit_loss(
    predicted_benefit: Tensor,
    realized_benefit: Tensor,
    *,
    weight: Optional[Tensor] = None,
    beta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Smooth-L1 stop-head loss with detached targets and explicit IPW support."""
    if predicted_benefit.shape != realized_benefit.shape:
        raise ValueError(
            "predicted_benefit and realized_benefit must have identical shapes"
        )
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be one of: none, sum, mean")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and > 0")
    elementwise = F.smooth_l1_loss(
        predicted_benefit,
        realized_benefit.detach().to(predicted_benefit),
        beta=float(beta),
        reduction="none",
    )
    if weight is None:
        weighted = elementwise
        denominator = torch.tensor(
            elementwise.numel(), device=elementwise.device, dtype=elementwise.dtype
        )
    else:
        if weight.shape != elementwise.shape:
            try:
                weight = torch.broadcast_to(weight, elementwise.shape)
            except RuntimeError as error:
                raise ValueError(
                    "weight is not broadcastable to the benefit loss"
                ) from error
        weight = weight.to(device=elementwise.device, dtype=elementwise.dtype)
        if bool((weight < 0).any()) or not bool(torch.isfinite(weight).all()):
            raise ValueError("weight must be finite and non-negative")
        weighted = elementwise * weight
        denominator = weight.sum()
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / denominator.clamp_min(1.0)


def masked_mean_absolute_error(
    prediction: Tensor, target: Tensor, valid_mask: Optional[Tensor] = None
) -> Tensor:
    """Per-case MAE used to construct translation-loss trajectories."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched shape")
    error = (prediction - target).abs()
    if valid_mask is None:
        return error.flatten(1).mean(dim=1)
    mask = valid_mask.to(device=error.device, dtype=error.dtype)
    if mask.ndim == error.ndim - 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == error.ndim - 1:
        mask = mask.unsqueeze(1)
    elif mask.ndim != error.ndim:
        raise ValueError(
            "valid_mask must be spatial, batched spatial, or prediction-shaped"
        )
    try:
        mask = torch.broadcast_to(mask, error.shape)
    except RuntimeError as exc:
        raise ValueError("valid_mask is not broadcastable to prediction") from exc
    numerator = (error * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError("valid_mask must select at least one element per case")
    return numerator / denominator


def anatomy_proxy_from_descriptors(
    translated_descriptor: Tensor,
    source_descriptor: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return per-case selected-anatomy descriptor MAE; inputs are precomputed."""
    return masked_mean_absolute_error(
        translated_descriptor, source_descriptor, valid_mask
    )


def robust_positive_scale(
    values: Tensor, *, quantile: float = 0.75, minimum: float = 1e-06
) -> float:
    """Deterministic robust absolute scale for an offline calibration tool."""
    if not isinstance(values, Tensor) or values.numel() == 0:
        raise ValueError("values must be a non-empty torch.Tensor")
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("quantile must lie in [0.5, 1.0)")
    if not math.isfinite(float(minimum)) or float(minimum) <= 0.0:
        raise ValueError("minimum must be finite and > 0")
    finite = values.detach().float().flatten()
    if not bool(torch.isfinite(finite).all()):
        raise ValueError("values must be finite")
    center = finite.median()
    scale = torch.quantile((finite - center).abs(), float(quantile))
    return max(float(scale), float(minimum))


__all__ = [
    "anatomy_proxy_from_descriptors",
    "continuation_benefit_targets",
    "masked_mean_absolute_error",
    "robust_positive_scale",
    "smooth_l1_benefit_loss",
]
