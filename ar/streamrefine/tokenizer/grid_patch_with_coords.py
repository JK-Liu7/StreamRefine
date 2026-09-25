"""Explicit HWD patch geometry for StreamRefine VidTok latent caching.

Medical volumes use ``[C, H, W, D]`` throughout this module.  VidTok latent
coordinates are recorded as ``DHW`` because cached latents use
``[C_z, D_l, H_l, W_l]``.  Keeping those conventions in the field names is
intentional: it prevents a silent HWD/DHW axis swap at cache boundaries.
"""

from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any

PATCH_SIZE_HWD = (256, 256, 16)
STRIDE_HWD = (192, 192, 12)
COMPRESSION_HWD = (8, 8, 4)
LATENT_TILE_DHW = (4, 32, 32)
LATENT_STRIDE_DHW = (3, 24, 24)
BRATS24_SHAPE_HWD = (256, 256, 128)
MINIMUM_IMAGE_SHAPE_HWD = (96, 96, 96)


def as_3tuple(
    values: Sequence[int] | Sequence[float], name: str
) -> tuple[int, int, int]:
    """Return three positive integer values, with a useful contract error."""
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {values!r}")
    result = tuple((int(value) for value in values))
    if any((value <= 0 for value in result)):
        raise ValueError(f"{name} must contain positive values, got {result!r}")
    return result


def stride_from_overlap(
    patch_hwd: Sequence[int] = PATCH_SIZE_HWD, overlap: float = 0.25
) -> tuple[int, int, int]:
    """Compute an integer HWD stride for a fractional overlap."""
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    patch = as_3tuple(patch_hwd, "patch_hwd")
    stride = tuple((int(round(dim * (1.0 - float(overlap)))) for dim in patch))
    if any((value <= 0 for value in stride)):
        raise ValueError(
            f"overlap={overlap} produces a non-positive stride for {patch}"
        )
    return stride


def coverage_padded_size(
    n: int, patch: int, stride: int, comp: int, minimum: int = 0
) -> int:
    """Pad one dimension to ``patch + k * stride`` and compression alignment.

    ``minimum`` is applied before the coverage calculation.  This is important
    for StreamRefine: variable-size datasets must still support a later 96^3
    image-equivalent training crop.
    """
    n, patch, stride, comp, minimum = map(int, (n, patch, stride, comp, minimum))
    if n <= 0 or patch <= 0 or stride <= 0 or (comp <= 0):
        raise ValueError(
            f"n, patch, stride, and comp must be positive: {(n, patch, stride, comp)}"
        )
    if minimum < 0:
        raise ValueError(f"minimum must be non-negative, got {minimum}")
    if patch % comp != 0 or stride % comp != 0:
        raise ValueError(
            f"patch={patch} and stride={stride} must both be divisible by comp={comp}"
        )
    target = max(n, minimum)
    if target <= patch:
        out = patch
    else:
        out = patch + math.ceil((target - patch) / stride) * stride
    out = math.ceil(out / comp) * comp
    return int(out)


def _dataset_key(dataset: str) -> str:
    key = str(dataset).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "brats": "brats24",
        "brats2024": "brats24",
        "brats24": "brats24",
        "synthrad": "synthrad",
        "synthrad2023": "synthrad",
        "autopet": "autopet",
        "autopetii": "autopet",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected BraTS24, SynthRAD, or AutoPET"
        ) from exc


def compute_padded_shape_hwd(
    shape_hwd: Sequence[int],
    dataset: str,
    patch_hwd: Sequence[int] = PATCH_SIZE_HWD,
    stride_hwd: Sequence[int] = STRIDE_HWD,
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
    minimum_image_shape_hwd: Sequence[int] = MINIMUM_IMAGE_SHAPE_HWD,
    pad_mode: str = "auto",
) -> tuple[int, int, int]:
    """Apply the dataset-specific full-volume padding policy.

    BraTS24 is a strict no-padding path and must already be 256x256x128.
    SynthRAD and AutoPET use one case-shared coverage-padded shape.
    """
    shape = as_3tuple(shape_hwd, "shape_hwd")
    patch = as_3tuple(patch_hwd, "patch_hwd")
    stride = as_3tuple(stride_hwd, "stride_hwd")
    comp = as_3tuple(comp_hwd, "comp_hwd")
    minimum = as_3tuple(minimum_image_shape_hwd, "minimum_image_shape_hwd")
    dataset_key = _dataset_key(dataset)
    mode = str(pad_mode).strip().lower().replace("-", "_")
    mode_aliases = {
        "auto": "auto",
        "coverage": "coverage",
        "coverage_pad": "coverage",
        "no_pad": "no_pad",
        "none": "no_pad",
    }
    try:
        mode = mode_aliases[mode]
    except KeyError as exc:
        raise ValueError(
            f"pad_mode must be auto, coverage, or no_pad; got {pad_mode!r}"
        ) from exc
    if dataset_key == "brats24":
        if mode == "coverage":
            raise ValueError("BraTS24 explicitly forbids coverage padding")
        if shape != BRATS24_SHAPE_HWD:
            raise ValueError(
                f"BraTS24 must already be on the fixed {BRATS24_SHAPE_HWD} common grid; got {shape}. BraTS24 cache generation does not pad or resize."
            )
        return shape
    if mode == "no_pad":
        if any((n < p for n, p in zip(shape, patch))):
            raise ValueError(
                f"no_pad requires shape_hwd={shape} to be at least patch_hwd={patch}"
            )
        if any((n % c for n, c in zip(shape, comp))):
            raise ValueError(
                f"no_pad shape_hwd={shape} must be divisible by compression_hwd={comp}"
            )
        for n, p, s, c in zip(shape, patch, stride, comp):
            compute_starts(n, p, s, c, include_last=True)
        return shape
    return tuple(
        (
            coverage_padded_size(n, p, s, c, minimum=m)
            for n, p, s, c, m in zip(shape, patch, stride, comp, minimum)
        )
    )


def compute_starts(
    n: int,
    patch: int,
    stride: int,
    require_divisible_by: int,
    include_last: bool = True,
) -> list[int]:
    """Return deterministic patch starts with an optional exact last patch."""
    n, patch, stride, require_divisible_by = map(
        int, (n, patch, stride, require_divisible_by)
    )
    if n <= 0 or patch <= 0 or stride <= 0 or (require_divisible_by <= 0):
        raise ValueError(
            f"n, patch, stride, and require_divisible_by must be positive: {(n, patch, stride, require_divisible_by)}"
        )
    if n <= patch:
        starts = [0]
    else:
        starts = list(range(0, n - patch + 1, stride))
        if include_last:
            last = n - patch
            if not starts or starts[-1] != last:
                starts.append(last)
        starts = sorted(set(starts))
    bad = [start for start in starts if start % require_divisible_by != 0]
    if bad:
        raise ValueError(
            f"Patch starts must map exactly to the latent grid: n={n}, patch={patch}, stride={stride}, compression={require_divisible_by}, bad_starts={bad}"
        )
    return starts


def build_patch_index_table(
    shape_hwd: Sequence[int],
    patch_hwd: Sequence[int] = PATCH_SIZE_HWD,
    stride_hwd: Sequence[int] = STRIDE_HWD,
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
    include_last: bool = True,
) -> tuple[list[dict[str, Any]], tuple[list[int], list[int], list[int]]]:
    """Build the case-shared image/latent coordinate table."""
    shape = as_3tuple(shape_hwd, "shape_hwd")
    patch = as_3tuple(patch_hwd, "patch_hwd")
    stride = as_3tuple(stride_hwd, "stride_hwd")
    comp = as_3tuple(comp_hwd, "comp_hwd")
    if any((n < p for n, p in zip(shape, patch))):
        raise ValueError(
            f"shape_hwd={shape} is smaller than patch_hwd={patch}; apply coverage padding first"
        )
    if any((p % c != 0 or s % c != 0 for p, s, c in zip(patch, stride, comp))):
        raise ValueError(
            f"patch_hwd={patch} and stride_hwd={stride} must align to compression_hwd={comp}"
        )
    hs = compute_starts(shape[0], patch[0], stride[0], comp[0], include_last)
    ws = compute_starts(shape[1], patch[1], stride[1], comp[1], include_last)
    ds = compute_starts(shape[2], patch[2], stride[2], comp[2], include_last)
    table: list[dict[str, Any]] = []
    for ih, h0 in enumerate(hs):
        for iw, w0 in enumerate(ws):
            for id_, d0 in enumerate(ds):
                h1, w1, d1 = (h0 + patch[0], w0 + patch[1], d0 + patch[2])
                lh0, lw0, ld0 = (h0 // comp[0], w0 // comp[1], d0 // comp[2])
                lh1, lw1, ld1 = (h1 // comp[0], w1 // comp[1], d1 // comp[2])
                table.append(
                    {
                        "patch_id": len(table),
                        "h0": h0,
                        "h1": h1,
                        "w0": w0,
                        "w1": w1,
                        "d0": d0,
                        "d1": d1,
                        "lh0": lh0,
                        "lh1": lh1,
                        "lw0": lw0,
                        "lw1": lw1,
                        "ld0": ld0,
                        "ld1": ld1,
                        "ih": ih,
                        "iw": iw,
                        "id": id_,
                        "num_h": len(hs),
                        "num_w": len(ws),
                        "num_d": len(ds),
                        "is_first_h": ih == 0,
                        "is_last_h": ih == len(hs) - 1,
                        "is_first_w": iw == 0,
                        "is_last_w": iw == len(ws) - 1,
                        "is_first_d": id_ == 0,
                        "is_last_d": id_ == len(ds) - 1,
                        "grid_index_hwd": [ih, iw, id_],
                        "grid_shape_hwd": [len(hs), len(ws), len(ds)],
                        "image_start_hwd": [h0, w0, d0],
                        "image_bbox_hwd": [[h0, h1], [w0, w1], [d0, d1]],
                        "latent_start_dhw": [ld0, lh0, lw0],
                        "latent_bbox_dhw": [[ld0, ld1], [lh0, lh1], [lw0, lw1]],
                    }
                )
    return (table, (hs, ws, ds))


def pad_chwd_to_shape(x: Any, padded_shape_hwd: Sequence[int], value: float = -1.0):
    """Right-pad a ``[C,H,W,D]`` tensor to a case-shared HWD shape."""
    import torch.nn.functional as functional

    if x.ndim != 4:
        raise ValueError(f"Expected [C,H,W,D], got {tuple(x.shape)}")
    _, h, w, d = map(int, x.shape)
    hp, wp, dp = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    if h > hp or w > wp or d > dp:
        raise ValueError(f"Cannot pad shape {(h, w, d)} down to {(hp, wp, dp)}")
    return functional.pad(
        x, (0, dp - d, 0, wp - w, 0, hp - h), mode="constant", value=float(value)
    )


def crop_chwd(x: Any, patch_meta: dict[str, Any]):
    """Crop one patch using an entry produced by :func:`build_patch_index_table`."""
    if x.ndim != 4:
        raise ValueError(f"Expected [C,H,W,D], got {tuple(x.shape)}")
    required = ("h0", "h1", "w0", "w1", "d0", "d1")
    missing = [key for key in required if key not in patch_meta]
    if missing:
        raise KeyError(f"patch_meta is missing coordinate fields: {missing}")
    crop = x[
        :,
        int(patch_meta["h0"]) : int(patch_meta["h1"]),
        int(patch_meta["w0"]) : int(patch_meta["w1"]),
        int(patch_meta["d0"]) : int(patch_meta["d1"]),
    ]
    return crop.contiguous()


def padding_info(
    original_shape_hwd: Sequence[int],
    padded_shape_hwd: Sequence[int],
    *,
    pad_value: float,
    policy: str,
) -> dict[str, Any]:
    """Create JSON-safe, explicit end-padding metadata."""
    original = as_3tuple(original_shape_hwd, "original_shape_hwd")
    padded = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    if any((p < o for o, p in zip(original, padded))):
        raise ValueError(
            f"padded shape {padded} cannot be smaller than original shape {original}"
        )
    return {
        "pad_width_hwd": [[0, int(p - o)] for o, p in zip(original, padded)],
        "pad_value": float(pad_value),
        "pad_side": "end",
        "policy": str(policy),
    }
