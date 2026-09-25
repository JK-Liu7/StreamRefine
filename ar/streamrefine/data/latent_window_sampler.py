from __future__ import annotations
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any
import torch
import torch.nn.functional as F

DEFAULT_WINDOW_DHW = (24, 12, 12)


def as_dhw_window(window: Sequence[int]) -> tuple[int, int, int]:
    if len(window) != 3:
        raise ValueError(f"window must contain D,H,W, got {window!r}")
    result = tuple((int(item) for item in window))
    if any((item <= 0 for item in result)):
        raise ValueError(f"window values must be positive, got {result}")
    return result


def as_dhw_origin(origin: Sequence[int]) -> tuple[int, int, int]:
    if len(origin) != 3:
        raise ValueError(f"origin must contain D,H,W, got {origin!r}")
    result = tuple((int(item) for item in origin))
    if any((item < 0 for item in result)):
        raise ValueError(f"origin values must be non-negative, got {result}")
    return result


def _validate_mask_and_window(
    valid_mask: torch.Tensor, window: Sequence[int]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected valid_mask [D,H,W], got {tuple(valid_mask.shape)}")
    shape = tuple((int(item) for item in valid_mask.shape))
    window_dhw = as_dhw_window(window)
    if any((dim < size for dim, size in zip(shape, window_dhw))):
        raise ValueError(
            f"Cache is too small: latent_shape_dhw={shape}, window_dhw={window_dhw}"
        )
    return (shape, window_dhw)


def window_valid_ratio(
    valid_mask: torch.Tensor,
    origin_dhw: Sequence[int],
    window: Sequence[int] = DEFAULT_WINDOW_DHW,
) -> float:
    shape, (wd, wh, ww) = _validate_mask_and_window(valid_mask, window)
    d0, h0, w0 = as_dhw_origin(origin_dhw)
    if not (d0 + wd <= shape[0] and h0 + wh <= shape[1] and (w0 + ww <= shape[2])):
        raise ValueError(
            f"origin_dhw={(d0, h0, w0)} with window={(wd, wh, ww)} exceeds shape={shape}"
        )
    crop = valid_mask[d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww]
    return float(crop.float().mean().item())


def mask_has_minimal_pir_edge(mask: torch.Tensor) -> bool:
    """Return whether a hard DHW support contains any configured PIR edge."""
    return minimal_pir_edge_count(mask) > 0


def minimal_pir_edge_count(mask: torch.Tensor) -> int:
    """Count the canonical PIR edges whose two endpoints are in ``mask``."""
    from streamrefine.training.pir import PIR_OFFSETS_DHW

    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("PIR edge support must be a boolean [D,H,W] mask")
    count = 0
    shape = tuple((int(value) for value in mask.shape))
    for offset_dhw in PIR_OFFSETS_DHW:
        if any((offset < 0 for offset in offset_dhw)) or not any(offset_dhw):
            raise RuntimeError(f"Invalid canonical PIR offset: {offset_dhw!r}")
        if any((offset >= size for offset, size in zip(offset_dhw, shape))):
            continue
        left = tuple(
            (slice(0, size - offset) for size, offset in zip(shape, offset_dhw))
        )
        right = tuple((slice(offset, size) for size, offset in zip(shape, offset_dhw)))
        count += int((mask[left] & mask[right]).sum().item())
    return count


def _window_mask(
    mask: torch.Tensor, origin: Sequence[int], window: Sequence[int]
) -> torch.Tensor:
    d0, h0, w0 = as_dhw_origin(origin)
    wd, wh, ww = as_dhw_window(window)
    return mask[d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww]


def find_best_valid_window(
    valid_mask: torch.Tensor, window: Sequence[int] = DEFAULT_WINDOW_DHW
) -> tuple[tuple[int, int, int], float]:
    """Find the globally highest-occupancy window, deterministically."""
    _, window_dhw = _validate_mask_and_window(valid_mask, window)
    scores = F.avg_pool3d(
        valid_mask.float().unsqueeze(0).unsqueeze(0), kernel_size=window_dhw, stride=1
    )[0, 0]
    flat_index = int(torch.argmax(scores.reshape(-1)).item())
    _, starts_h, starts_w = (int(item) for item in scores.shape)
    d0, remainder = divmod(flat_index, starts_h * starts_w)
    h0, w0 = divmod(remainder, starts_w)
    return ((d0, h0, w0), float(scores[d0, h0, w0].item()))


def find_best_pir_edge_window(
    valid_mask: torch.Tensor,
    pir_mask: torch.Tensor,
    window: Sequence[int] = DEFAULT_WINDOW_DHW,
    *,
    min_valid_ratio: float = 0.85,
) -> tuple[tuple[int, int, int], float, int]:
    """Prefer threshold-qualified windows with the most canonical PIR edges.

    If none qualifies (e.g. a padded short volume), maximize valid coverage
    among windows containing an edge, then edge count.  The returned ratio
    remains the actual coverage, possibly below ``min_valid_ratio``; callers
    expose this through ``valid_window_met_threshold``.  An edge is always
    required so PIR stays defined.  Ties use the earliest flattened DHW origin.
    """
    from streamrefine.training.pir import PIR_OFFSETS_DHW

    shape, window_dhw = _validate_mask_and_window(valid_mask, window)
    if pir_mask.dtype != torch.bool or tuple(pir_mask.shape) != shape:
        raise ValueError("pir_mask must be boolean and match valid_mask [D,H,W]")
    if not 0.0 <= float(min_valid_ratio) <= 1.0:
        raise ValueError(f"min_valid_ratio must be in [0,1], got {min_valid_ratio}")
    valid_scores = F.avg_pool3d(
        valid_mask.float().unsqueeze(0).unsqueeze(0), kernel_size=window_dhw, stride=1
    )[0, 0]
    edge_scores = torch.zeros_like(valid_scores)
    for offset_dhw in PIR_OFFSETS_DHW:
        kernel = tuple((size - offset for size, offset in zip(window_dhw, offset_dhw)))
        if any((size <= 0 for size in kernel)):
            continue
        left = tuple(
            (slice(0, size - offset) for size, offset in zip(shape, offset_dhw))
        )
        right = tuple((slice(offset, size) for size, offset in zip(shape, offset_dhw)))
        edge_map = (pir_mask[left] & pir_mask[right]).float()
        pooled = F.avg_pool3d(
            edge_map.unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1
        )[0, 0]
        edge_scores = edge_scores + pooled * float(math.prod(kernel))
    has_edges = edge_scores >= 0.5
    if not bool(has_edges.any()):
        raise ValueError(
            f"No crop contains at least one PIR anatomy edge: latent_shape_dhw={shape}, window_dhw={window_dhw}"
        )
    eligible = (valid_scores >= float(min_valid_ratio)) & has_edges
    if not bool(eligible.any()):
        best_ratio = valid_scores.masked_fill(~has_edges, -1.0).max()
        eligible = has_edges & (valid_scores == best_ratio)
        warnings.warn(
            "No PIR crop reaches min_valid_ratio; using the highest-valid-coverage crop with at least one PIR anatomy edge. Inspect valid_ratio and valid_window_met_threshold in the sample.",
            RuntimeWarning,
            stacklevel=2,
        )
    ranked = torch.where(eligible, edge_scores, torch.full_like(edge_scores, -1.0))
    flat_index = int(torch.argmax(ranked.reshape(-1)).item())
    _, starts_h, starts_w = (int(item) for item in ranked.shape)
    d0, remainder = divmod(flat_index, starts_h * starts_w)
    h0, w0 = divmod(remainder, starts_w)
    edge_count = int(round(float(edge_scores[d0, h0, w0].item())))
    ratio = float(valid_scores[d0, h0, w0].item())
    return ((d0, h0, w0), ratio, edge_count)


def sample_valid_latent_window(
    valid_mask: torch.Tensor,
    window: Sequence[int] = DEFAULT_WINDOW_DHW,
    min_valid_ratio: float = 0.85,
    max_tries: int = 100,
    generator: torch.Generator | None = None,
    required_pir_mask: torch.Tensor | None = None,
) -> tuple[tuple[int, int, int], float]:
    """Sample a valid-aware window, then use the true best window as fallback."""
    shape, window_dhw = _validate_mask_and_window(valid_mask, window)
    if not 0.0 <= float(min_valid_ratio) <= 1.0:
        raise ValueError(f"min_valid_ratio must be in [0,1], got {min_valid_ratio}")
    if int(max_tries) < 0:
        raise ValueError(f"max_tries must be non-negative, got {max_tries}")
    wd, wh, ww = window_dhw
    if required_pir_mask is not None:
        if (
            required_pir_mask.dtype != torch.bool
            or tuple(required_pir_mask.shape) != shape
        ):
            raise ValueError(
                "required_pir_mask must be boolean and match valid_mask [D,H,W]"
            )
        if not mask_has_minimal_pir_edge(required_pir_mask):
            raise ValueError("Full-volume valid & anatomy support has no PIR edge")
    for _ in range(int(max_tries)):
        origin = (
            int(torch.randint(0, shape[0] - wd + 1, (), generator=generator).item()),
            int(torch.randint(0, shape[1] - wh + 1, (), generator=generator).item()),
            int(torch.randint(0, shape[2] - ww + 1, (), generator=generator).item()),
        )
        ratio = window_valid_ratio(valid_mask, origin, window_dhw)
        has_required_edge = required_pir_mask is None or mask_has_minimal_pir_edge(
            _window_mask(required_pir_mask, origin, window_dhw)
        )
        if ratio >= float(min_valid_ratio) and has_required_edge:
            return (origin, ratio)
    if required_pir_mask is None:
        return find_best_valid_window(valid_mask, window_dhw)
    origin, ratio, _ = find_best_pir_edge_window(
        valid_mask, required_pir_mask, window_dhw, min_valid_ratio=min_valid_ratio
    )
    return (origin, ratio)


def center_latent_window(
    valid_mask: torch.Tensor, window: Sequence[int] = DEFAULT_WINDOW_DHW
) -> tuple[tuple[int, int, int], float]:
    shape, window_dhw = _validate_mask_and_window(valid_mask, window)
    origin = tuple(((dim - size) // 2 for dim, size in zip(shape, window_dhw)))
    return (origin, window_valid_ratio(valid_mask, origin, window_dhw))


def crop_aligned_pair(
    source_cache: Mapping[str, Any],
    target_cache: Mapping[str, Any],
    origin_dhw: Sequence[int],
    window: Sequence[int] = DEFAULT_WINDOW_DHW,
) -> dict[str, Any]:
    """Crop source, target, and the case-shared masks at identical coordinates."""
    source = torch.as_tensor(source_cache["latent_mu"])
    target = torch.as_tensor(target_cache["latent_mu"])
    if source.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"Expected latent_mu [C,D,H,W], got source={tuple(source.shape)}, target={tuple(target.shape)}"
        )
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"Source/target latent shapes differ: {tuple(source.shape)} vs {tuple(target.shape)}"
        )
    shape, window_dhw = _validate_mask_and_window(
        torch.as_tensor(source_cache["valid_mask_latent"]), window
    )
    if tuple(source.shape[1:]) != shape:
        raise ValueError(
            f"latent_mu shape {tuple(source.shape[1:])} does not match valid mask {shape}"
        )
    d0, h0, w0 = as_dhw_origin(origin_dhw)
    wd, wh, ww = window_dhw
    if d0 + wd > shape[0] or h0 + wh > shape[1] or w0 + ww > shape[2]:
        raise ValueError(
            f"origin_dhw={(d0, h0, w0)} with window={window_dhw} exceeds shape={shape}"
        )
    slices = (slice(d0, d0 + wd), slice(h0, h0 + wh), slice(w0, w0 + ww))
    hard = (
        torch.as_tensor(source_cache["valid_mask_latent"])[slices].bool().contiguous()
    )
    source_anatomy = torch.as_tensor(source_cache["anatomy_mask_latent_hard"])
    target_anatomy = torch.as_tensor(target_cache["anatomy_mask_latent_hard"])
    if tuple(source_anatomy.shape) != shape or not torch.equal(
        source_anatomy.bool(), target_anatomy.bool()
    ):
        raise ValueError(
            "Source/target anatomy_mask_latent_hard must be identical and aligned"
        )
    anatomy_hard = source_anatomy[slices].bool().contiguous()
    pir_hard = (hard & anatomy_hard).contiguous()
    soft_full = source_cache.get(
        "valid_mask_latent_soft", source_cache["valid_mask_latent"]
    )
    soft = torch.as_tensor(soft_full)[slices].float().contiguous()
    return {
        "source_latent_raw": source[slice(None), *slices].contiguous(),
        "target_latent_raw": target[slice(None), *slices].contiguous(),
        "valid_mask_latent_soft": soft,
        "valid_mask_latent": soft,
        "valid_mask_latent_hard": hard,
        "anatomy_mask_latent_hard": anatomy_hard,
        "pir_mask_latent_hard": pir_hard,
        "crop_origin_dhw": torch.as_tensor([d0, h0, w0], dtype=torch.long),
        "crop_shape_dhw": torch.as_tensor(window_dhw, dtype=torch.long),
        "valid_ratio": soft.mean(),
    }


def patchify_valid_mask(
    valid_mask: torch.Tensor, block_size_dhw: Sequence[int] = (2, 1, 1)
) -> torch.Tensor:
    """Convert a latent-cell mask to token blocks (default DiT patchify 2x1x1)."""
    block = as_dhw_window(block_size_dhw)
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected valid_mask [D,H,W], got {tuple(valid_mask.shape)}")
    d, h, w = (int(item) for item in valid_mask.shape)
    bd, bh, bw = block
    if d % bd or h % bh or w % bw:
        raise ValueError(
            f"valid_mask shape {(d, h, w)} must be divisible by block_size {block}"
        )
    return (
        valid_mask.reshape(d // bd, bd, h // bh, bh, w // bw, bw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(d // bd, h // bh, w // bw, bd * bh * bw)
    )
