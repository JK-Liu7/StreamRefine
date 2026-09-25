"""Case-shared image and latent occupancy masks."""

from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as functional
from torch import Tensor
from .grid_patch_with_coords import COMPRESSION_HWD, as_3tuple


def build_image_valid_mask_hwd(
    original_shape_hwd: Sequence[int], padded_shape_hwd: Sequence[int]
) -> Tensor:
    """Return ``[1,H_p,W_p,D_p]`` occupancy for end-padded volumes."""
    h, w, d = as_3tuple(original_shape_hwd, "original_shape_hwd")
    hp, wp, dp = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    if h > hp or w > wp or d > dp:
        raise ValueError(
            f"original_shape_hwd={(h, w, d)} cannot exceed padded_shape_hwd={(hp, wp, dp)}"
        )
    mask = torch.zeros((1, hp, wp, dp), dtype=torch.float32)
    mask[:, :h, :w, :d] = 1.0
    return mask


def downsample_valid_mask_to_latent(
    mask_hwd: Tensor, comp_hwd: Sequence[int] = COMPRESSION_HWD, threshold: float = 0.5
) -> tuple[Tensor, Tensor]:
    """Pool image occupancy into hard and soft latent ``[D_l,H_l,W_l]`` masks."""
    comp_h, comp_w, comp_d = as_3tuple(comp_hwd, "comp_hwd")
    if mask_hwd.ndim != 4 or int(mask_hwd.shape[0]) != 1:
        raise ValueError(f"Expected mask [1,H,W,D], got {tuple(mask_hwd.shape)}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    _, h, w, d = map(int, mask_hwd.shape)
    if h % comp_h or w % comp_w or d % comp_d:
        raise ValueError(
            f"Mask shape HWD={(h, w, d)} must be divisible by compression_hwd={(comp_h, comp_w, comp_d)}"
        )
    mask_float = mask_hwd.float()
    if not torch.isfinite(mask_float).all():
        raise ValueError("Image valid mask contains NaN or Inf")
    if bool((mask_float < 0).any()) or bool((mask_float > 1).any()):
        raise ValueError("Image valid mask values must lie in [0,1]")
    x = mask_float.permute(0, 3, 1, 2).unsqueeze(0)
    soft = functional.avg_pool3d(
        x, kernel_size=(comp_d, comp_h, comp_w), stride=(comp_d, comp_h, comp_w)
    )[0, 0]
    hard = soft >= float(threshold)
    if not torch.isfinite(soft).all():
        raise RuntimeError("Non-finite latent occupancy after average pooling")
    return (hard, soft)


def build_latent_valid_masks(
    original_shape_hwd: Sequence[int],
    padded_shape_hwd: Sequence[int],
    *,
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Convenience wrapper for the shared image-to-latent mask path."""
    image_mask = build_image_valid_mask_hwd(original_shape_hwd, padded_shape_hwd)
    return downsample_valid_mask_to_latent(
        image_mask, comp_hwd=comp_hwd, threshold=threshold
    )
