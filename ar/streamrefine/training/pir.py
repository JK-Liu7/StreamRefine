"""Minimal Pair-Invariant Relations (PIR) for detached stopping targets.

The implementation follows ``StreamRefine_ICLR_Strategy_Minimal_PIR.docx``:

* relations are channel-wise cosines in the frozen VidTok clean posterior-mean
  latent, never in the z-scored model space;
* the exact positive DHW offsets ``(2,0,0)``, ``(0,1,0)``, and ``(0,0,1)``
  are used;
* source--target agreement is a detached reliability weight;
* the reduction denominator is the number of valid-and-anatomy edges ``|E|`` (not the sum
  of reliability weights);
* every public computation is performed under ``torch.no_grad`` so PIR cannot
  silently become an editor regularizer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

PIR_FORMULA_VERSION = "streamrefine_minimal_pir_v2_anatomy_support"
PIR_OFFSETS_DHW = ((2, 0, 0), (0, 1, 0), (0, 0, 1))
PIR_FORMULA = "r_i_delta(z)=cos(z_i,z_i+delta); w_i_delta=(1-0.5*abs(r_i_delta(z_s)-r_i_delta(z_t)))^2; A_pir(z_hat)=sum_E(w_i_delta*abs(r_i_delta(z_hat)-r_i_delta(z_t)))/|E|"
_COSINE_EPS = 1e-08


@dataclass(frozen=True)
class MinimalPIRReference:
    """Detached target relations, pair reliability, and valid-edge support."""

    target_relations: tuple[Any, ...]
    reliability_weights: tuple[Any, ...]
    valid_edges: tuple[Any, ...]
    edge_count: Any
    latent_shape_bcdhw: tuple[int, int, int, int, int]
    formula_version: str = PIR_FORMULA_VERSION


def _batched_latent(value: Any, name: str):
    import torch

    tensor = torch.as_tensor(value).detach().float()
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"{name} must be [C,D,H,W] or [B,C,D,H,W]")
    minimum_shape = tuple(
        (max((offset[axis] for offset in PIR_OFFSETS_DHW)) + 1 for axis in range(3))
    )
    if int(tensor.shape[1]) <= 0 or any(
        (int(size) < minimum for size, minimum in zip(tensor.shape[-3:], minimum_shape))
    ):
        raise ValueError(
            f"{name} must have channels and spatial shape at least {minimum_shape}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor.contiguous()


def _batched_hard_mask(
    value: Any, *, name: str, batch_size: int, spatial_shape: tuple[int, int, int]
):
    import torch

    mask = torch.as_tensor(value).detach()
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 5 and int(mask.shape[1]) == 1:
        mask = mask[:, 0]
    if mask.ndim != 4:
        raise ValueError(f"{name} must be [D,H,W], [B,D,H,W], or [B,1,D,H,W]")
    if int(mask.shape[0]) != int(batch_size):
        raise ValueError(f"{name} batch dimension does not match the latents")
    if tuple((int(size) for size in mask.shape[-3:])) != spatial_shape:
        raise ValueError(f"{name} spatial shape does not match the latents")
    if mask.dtype != torch.bool:
        raise TypeError(
            f"Minimal PIR requires cache {name} with bool dtype; soft/thresholded support is not part of the formula contract"
        )
    return mask.contiguous()


def _edge_views(value: Any, axis: int, offset: int = 2):
    left = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left[axis] = slice(0, -int(offset))
    right[axis] = slice(int(offset), None)
    return (value[tuple(left)], value[tuple(right)])


def _relation(latent: Any, spatial_axis: int, offset: int):
    import torch.nn.functional as functional

    left, right = _edge_views(latent, spatial_axis + 2, offset)
    return functional.cosine_similarity(
        left, right, dim=1, eps=_COSINE_EPS
    ).contiguous()


def count_minimal_pir_edges(valid_mask: Any, anatomy_mask: Any):
    """Count valid-and-anatomy relation edges per case without latent values."""
    import torch

    valid = torch.as_tensor(valid_mask).detach()
    anatomy = torch.as_tensor(anatomy_mask).detach()
    if valid.ndim == 3:
        valid = valid.unsqueeze(0)
    if anatomy.ndim == 3:
        anatomy = anatomy.unsqueeze(0)
    if valid.ndim == 5 and int(valid.shape[1]) == 1:
        valid = valid[:, 0]
    if anatomy.ndim == 5 and int(anatomy.shape[1]) == 1:
        anatomy = anatomy[:, 0]
    if (
        valid.ndim != 4
        or anatomy.ndim != 4
        or tuple(valid.shape) != tuple(anatomy.shape)
    ):
        raise ValueError("PIR valid/anatomy masks must share [B,D,H,W]")
    if valid.dtype != torch.bool or anatomy.dtype != torch.bool:
        raise TypeError("PIR valid/anatomy masks must both use torch.bool")
    if bool((anatomy & ~valid).any()):
        raise ValueError("PIR anatomy mask must be a subset of the valid mask")
    support = valid & anatomy
    count = torch.zeros(int(valid.shape[0]), dtype=torch.int64, device=valid.device)
    for spatial_axis, offset_dhw in enumerate(PIR_OFFSETS_DHW):
        distance = int(offset_dhw[spatial_axis])
        left, right = _edge_views(support, spatial_axis + 1, distance)
        count = count + (left & right).flatten(1).sum(1)
    return count.detach()


def build_minimal_pir_reference(
    source_latent_raw: Any, target_latent_raw: Any, valid_mask: Any, anatomy_mask: Any
) -> MinimalPIRReference:
    """Build the detached pair-calibrated target reference.

    ``source_latent_raw`` and ``target_latent_raw`` must be the cached clean
    posterior means.  Passing model-space latents here is a contract violation;
    call sites therefore source these tensors from ``*_latent_raw`` explicitly.
    """
    import torch

    with torch.no_grad():
        source = _batched_latent(source_latent_raw, "source_latent_raw")
        target = _batched_latent(target_latent_raw, "target_latent_raw")
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError("source and target PIR latents must have identical shapes")
        if source.device != target.device:
            raise ValueError("source and target PIR latents must be on the same device")
        spatial_shape = tuple((int(size) for size in source.shape[-3:]))
        valid_support = _batched_hard_mask(
            valid_mask,
            name="valid_mask_latent_hard",
            batch_size=int(source.shape[0]),
            spatial_shape=spatial_shape,
        )
        anatomy_support = _batched_hard_mask(
            anatomy_mask,
            name="anatomy_mask_latent_hard",
            batch_size=int(source.shape[0]),
            spatial_shape=spatial_shape,
        )
        if (
            valid_support.device != source.device
            or anatomy_support.device != source.device
        ):
            raise ValueError("PIR valid_mask must be on the same device as the latents")
        if bool((anatomy_support & ~valid_support).any()):
            raise ValueError(
                "anatomy_mask_latent_hard must be a subset of valid_mask_latent_hard"
            )
        support = (valid_support & anatomy_support).contiguous()
        target_relations: list[Any] = []
        reliability_weights: list[Any] = []
        valid_edges: list[Any] = []
        edge_count = torch.zeros(
            int(source.shape[0]), device=source.device, dtype=torch.float32
        )
        for spatial_axis, offset_dhw in enumerate(PIR_OFFSETS_DHW):
            distance = int(offset_dhw[spatial_axis])
            source_relation = _relation(source, spatial_axis, distance)
            target_relation = _relation(target, spatial_axis, distance)
            support_left, support_right = _edge_views(
                support, spatial_axis + 1, distance
            )
            edge_mask = (support_left & support_right).contiguous()
            reliability = (
                (1.0 - 0.5 * (source_relation - target_relation).abs())
                .clamp_(0.0, 1.0)
                .square_()
                .contiguous()
            )
            target_relations.append(target_relation.detach())
            reliability_weights.append(reliability.detach())
            valid_edges.append(edge_mask.detach())
            edge_count = edge_count + edge_mask.flatten(1).sum(1).float()
        empty = torch.nonzero(edge_count <= 0, as_tuple=False).flatten().tolist()
        if empty:
            raise ValueError(
                f"Minimal PIR is undefined because no configured valid edge exists for batch indices {empty}"
            )
        return MinimalPIRReference(
            target_relations=tuple(target_relations),
            reliability_weights=tuple(reliability_weights),
            valid_edges=tuple(valid_edges),
            edge_count=edge_count.detach(),
            latent_shape_bcdhw=tuple((int(size) for size in source.shape)),
        )


def minimal_pir_loss(predicted_latent_raw: Any, reference: MinimalPIRReference):
    """Return detached per-case ``A_k^PIR`` values with shape ``[B]``."""
    import torch

    if not isinstance(reference, MinimalPIRReference):
        raise TypeError("reference must be a MinimalPIRReference")
    if reference.formula_version != PIR_FORMULA_VERSION:
        raise ValueError("PIR reference formula version is incompatible")
    with torch.no_grad():
        prediction = _batched_latent(predicted_latent_raw, "predicted_latent_raw")
        if tuple((int(size) for size in prediction.shape)) != tuple(
            reference.latent_shape_bcdhw
        ):
            raise ValueError("predicted PIR latent shape differs from its reference")
        numerator = torch.zeros(
            int(prediction.shape[0]), device=prediction.device, dtype=torch.float32
        )
        for spatial_axis, offset_dhw in enumerate(PIR_OFFSETS_DHW):
            relation = _relation(
                prediction, spatial_axis, int(offset_dhw[spatial_axis])
            )
            target_relation = reference.target_relations[spatial_axis].to(
                device=prediction.device, dtype=relation.dtype
            )
            reliability = reference.reliability_weights[spatial_axis].to(
                device=prediction.device, dtype=relation.dtype
            )
            edge_mask = reference.valid_edges[spatial_axis].to(device=prediction.device)
            numerator = numerator + (
                (relation - target_relation).abs()
                * reliability
                * edge_mask.to(relation)
            ).flatten(1).sum(1)
        result = numerator / reference.edge_count.to(
            device=prediction.device, dtype=numerator.dtype
        )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Minimal PIR produced NaN or Inf")
        return result.detach()


def minimal_pir_trajectory(predicted_latents_raw: Any, reference: MinimalPIRReference):
    """Return detached PIR losses for ``[B,K,C,D,H,W]`` raw states."""
    import torch

    states = torch.as_tensor(predicted_latents_raw).detach().float()
    if states.ndim != 6:
        raise ValueError("predicted_latents_raw must be [B,K,C,D,H,W]")
    if int(states.shape[0]) != int(reference.latent_shape_bcdhw[0]):
        raise ValueError("PIR trajectory/reference batch dimensions differ")
    return torch.stack(
        [
            minimal_pir_loss(states[:, index], reference)
            for index in range(states.shape[1])
        ],
        dim=1,
    ).detach()


__all__ = [
    "MinimalPIRReference",
    "PIR_FORMULA",
    "PIR_FORMULA_VERSION",
    "PIR_OFFSETS_DHW",
    "build_minimal_pir_reference",
    "count_minimal_pir_edges",
    "minimal_pir_loss",
    "minimal_pir_trajectory",
]
