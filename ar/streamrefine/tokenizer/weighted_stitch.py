"""FP32 weighted stitching for continuous VidTok KL posterior means."""

from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
import torch
from torch import Tensor
from .grid_patch_with_coords import COMPRESSION_HWD, LATENT_TILE_DHW, as_3tuple


def _hann_with_floor(n: int, eps: float = 0.001) -> Tensor:
    n = int(n)
    if n <= 0:
        raise ValueError(f"Hann length must be positive, got {n}")
    if not math.isfinite(float(eps)) or not 0.0 < float(eps) <= 1.0:
        raise ValueError(f"eps must be finite and in (0, 1], got {eps}")
    if n == 1:
        return torch.ones(1, dtype=torch.float32)
    return torch.hann_window(n, periodic=False, dtype=torch.float32).clamp_min(
        float(eps)
    )


def _gaussian_with_floor(n: int, eps: float, sigma_scale: float) -> Tensor:
    n = int(n)
    if n <= 0:
        raise ValueError(f"Gaussian length must be positive, got {n}")
    if not math.isfinite(float(sigma_scale)) or float(sigma_scale) <= 0.0:
        raise ValueError(f"sigma_scale must be positive and finite, got {sigma_scale}")
    if n == 1:
        return torch.ones(1, dtype=torch.float32)
    coordinates = torch.arange(n, dtype=torch.float32) - (n - 1.0) / 2.0
    sigma = max(float(n) * float(sigma_scale), torch.finfo(torch.float32).eps)
    weights = torch.exp(-0.5 * (coordinates / sigma).square())
    return weights.clamp_min(float(eps))


def make_importance_map_dhw(
    tile_dhw: Sequence[int] = LATENT_TILE_DHW,
    eps: float = 0.001,
    *,
    mode: str = "hann",
    gaussian_sigma_scale: float = 0.125,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create a positive separable ``[1,D_l,H_l,W_l]`` importance map.

    Hann is the cache-v2 default.  Gaussian remains available for the documented
    D_l=4 sensitivity audit without changing the stitching interface.
    """
    tile = as_3tuple(tile_dhw, "tile_dhw")
    mode_key = str(mode).strip().lower()
    if mode_key == "hann":
        one_dimensional = [_hann_with_floor(n, eps) for n in tile]
    elif mode_key == "gaussian":
        one_dimensional = [
            _gaussian_with_floor(n, eps, gaussian_sigma_scale) for n in tile
        ]
    else:
        raise ValueError(f"importance mode must be 'hann' or 'gaussian', got {mode!r}")
    wd, wh, ww = one_dimensional
    weight = wd[:, None, None] * wh[None, :, None] * ww[None, None, :]
    weight = weight / weight.max().clamp_min(float(eps))
    weight = weight.clamp_min(float(eps) ** 3).unsqueeze(0).to(device=device)
    if weight.dtype != torch.float32 or tuple(weight.shape) != (1, *tile):
        raise RuntimeError(
            f"Invalid importance map shape/dtype: {weight.shape}, {weight.dtype}"
        )
    if not torch.isfinite(weight).all() or not (weight > 0).all():
        raise RuntimeError("Importance map must be finite and strictly positive")
    return weight


def allocate_full_accumulators(
    padded_shape_hwd: Sequence[int],
    channels: int = 16,
    *,
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
    device: torch.device | str | None = "cpu",
) -> tuple[Tensor, Tensor, tuple[int, int, int]]:
    """Allocate the only supported accumulator representation: FP32."""
    h, w, d = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    ch, cw, cd = as_3tuple(comp_hwd, "comp_hwd")
    channels = int(channels)
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if h % ch or w % cw or d % cd:
        raise ValueError(
            f"padded_shape_hwd={(h, w, d)} must be divisible by compression_hwd={(ch, cw, cd)}"
        )
    latent_shape_dhw = (d // cd, h // ch, w // cw)
    latent_sum = torch.zeros(
        (channels, *latent_shape_dhw), dtype=torch.float32, device=device
    )
    weight_sum = torch.zeros((1, *latent_shape_dhw), dtype=torch.float32, device=device)
    return (latent_sum, weight_sum, latent_shape_dhw)


def _latent_start_dhw(patch_meta: dict[str, Any]) -> tuple[int, int, int]:
    if all((key in patch_meta for key in ("ld0", "lh0", "lw0"))):
        return (int(patch_meta["ld0"]), int(patch_meta["lh0"]), int(patch_meta["lw0"]))
    if "latent_start_dhw" in patch_meta:
        values = patch_meta["latent_start_dhw"]
        if isinstance(values, (str, bytes)) or len(values) != 3:
            raise ValueError(f"latent_start_dhw must contain 3 values, got {values!r}")
        start = tuple((int(value) for value in values))
        if any((value < 0 for value in start)):
            raise ValueError(f"latent_start_dhw must be non-negative, got {start}")
        return start
    if all((key in patch_meta for key in ("d0", "h0", "w0"))):
        return (
            int(patch_meta["d0"]) // COMPRESSION_HWD[2],
            int(patch_meta["h0"]) // COMPRESSION_HWD[0],
            int(patch_meta["w0"]) // COMPRESSION_HWD[1],
        )
    raise KeyError("patch_meta must contain latent or image start coordinates")


def accumulate_patch_mean(
    latent_sum: Tensor,
    weight_sum: Tensor,
    patch_mu_cdhw: Tensor,
    patch_meta: dict[str, Any],
    importance_1dhw: Tensor,
) -> None:
    """Accumulate one continuous posterior-mean tile in FP32."""
    if latent_sum.ndim != 4 or weight_sum.ndim != 4 or patch_mu_cdhw.ndim != 4:
        raise ValueError(
            "Expected latent_sum [C,D,H,W], weight_sum [1,D,H,W], and patch [C,D,H,W]"
        )
    if latent_sum.dtype != torch.float32 or weight_sum.dtype != torch.float32:
        raise TypeError("latent_sum and weight_sum must both use torch.float32")
    if weight_sum.shape[0] != 1 or tuple(weight_sum.shape[1:]) != tuple(
        latent_sum.shape[1:]
    ):
        raise ValueError(
            f"Accumulator shape mismatch: latent_sum={tuple(latent_sum.shape)}, weight_sum={tuple(weight_sum.shape)}"
        )
    if patch_mu_cdhw.shape[0] != latent_sum.shape[0]:
        raise ValueError(
            f"Patch channels {patch_mu_cdhw.shape[0]} do not match accumulator channels {latent_sum.shape[0]}"
        )
    expected_weight_shape = (1, *tuple((int(v) for v in patch_mu_cdhw.shape[1:])))
    if tuple(importance_1dhw.shape) != expected_weight_shape:
        raise ValueError(
            f"Importance shape {tuple(importance_1dhw.shape)} does not match patch spatial shape {tuple(patch_mu_cdhw.shape[1:])}"
        )
    if not torch.isfinite(patch_mu_cdhw).all():
        raise ValueError("Posterior-mean patch contains NaN or Inf")
    if not torch.isfinite(importance_1dhw).all() or not (importance_1dhw > 0).all():
        raise ValueError("Importance weights must be finite and strictly positive")
    d0, h0, w0 = _latent_start_dhw(patch_meta)
    tile_d, tile_h, tile_w = map(int, patch_mu_cdhw.shape[1:])
    d1, h1, w1 = (d0 + tile_d, h0 + tile_h, w0 + tile_w)
    full_d, full_h, full_w = map(int, latent_sum.shape[1:])
    if d0 < 0 or h0 < 0 or w0 < 0 or (d1 > full_d) or (h1 > full_h) or (w1 > full_w):
        raise ValueError(
            f"Patch latent bbox {(d0, d1, h0, h1, w0, w1)} is outside full latent shape {(full_d, full_h, full_w)}"
        )
    patch_fp32 = patch_mu_cdhw.detach().to(
        device=latent_sum.device, dtype=torch.float32
    )
    weight_fp32 = importance_1dhw.detach().to(
        device=latent_sum.device, dtype=torch.float32
    )
    latent_sum[:, d0:d1, h0:h1, w0:w1].add_(patch_fp32 * weight_fp32)
    weight_sum[:, d0:d1, h0:h1, w0:w1].add_(weight_fp32)


def finalize_full_mean(latent_sum: Tensor, weight_sum: Tensor) -> Tensor:
    """Normalize FP32 sums, requiring complete finite positive coverage."""
    if latent_sum.dtype != torch.float32 or weight_sum.dtype != torch.float32:
        raise TypeError("Finalization requires FP32 latent_sum and weight_sum")
    if latent_sum.ndim != 4 or weight_sum.ndim != 4:
        raise ValueError("Expected 4D latent_sum and weight_sum")
    if weight_sum.shape[0] != 1 or tuple(weight_sum.shape[1:]) != tuple(
        latent_sum.shape[1:]
    ):
        raise ValueError(
            f"Accumulator shape mismatch: {tuple(latent_sum.shape)} vs {tuple(weight_sum.shape)}"
        )
    if not torch.isfinite(latent_sum).all() or not torch.isfinite(weight_sum).all():
        raise RuntimeError("Non-finite value in stitching accumulators")
    if not (weight_sum > 0).all():
        uncovered = int((weight_sum <= 0).sum().item())
        raise RuntimeError(f"Some latent cells have zero coverage ({uncovered} cells)")
    full_mu = latent_sum / weight_sum
    if full_mu.dtype != torch.float32 or not torch.isfinite(full_mu).all():
        raise RuntimeError("Non-finite full posterior mean after weighted finalization")
    return full_mu
