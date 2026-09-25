"""Strict StreamRefine KL posterior-mean cache schema and atomic I/O."""

from __future__ import annotations
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from .grid_patch_with_coords import (
    BRATS24_SHAPE_HWD,
    COMPRESSION_HWD,
    PATCH_SIZE_HWD,
    STRIDE_HWD,
    as_3tuple,
    build_patch_index_table,
)

CACHE_VERSION = "streamrefine_kl_mean_v2_anatomy_mask"
LATENT_CHANNELS = 16
LATENT_LAYOUT = "C,D,H,W"
POSTERIOR_MODE = "mean"
STITCH_MODE = "weighted_posterior_mean_blending"
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "latent_logvar",
        "sampled_z",
        "packed_index",
        "packed_indices",
        "weight_sum",
        "posterior_sample",
        "latent_sample",
        "z_sample",
        "raw_moments",
        "posterior_moments",
    }
)
REQUIRED_KEYS = frozenset(
    {
        "cache_version",
        "latent_mu",
        "valid_mask_latent",
        "valid_mask_latent_soft",
        "anatomy_mask_latent_hard",
        "anatomy_mask_contract",
        "anatomy_mask_contract_fingerprint",
        "anatomy_mask_sha256",
        "anatomy_mask_statistics",
        "original_shape_hwd",
        "padded_shape_hwd",
        "latent_shape_dhw",
        "compression_hwd",
        "latent_layout",
        "posterior_mode",
        "save_logvar",
        "latent_channels",
        "cache_dtype",
        "stitch_mode",
        "importance_mode",
        "patch_size_hwd",
        "stride_hwd",
        "overlap",
        "tokenizer_type",
        "tokenizer_causal",
        "tokenizer_config",
        "tokenizer_config_sha256",
        "tokenizer_checkpoint",
        "tokenizer_checkpoint_sha256",
        "generation_contract",
        "generation_contract_sha256",
        "tokenizer_git_commit",
        "dataset",
        "case_id",
        "group_id",
        "split",
        "modality",
        "available_modalities",
        "modality_image_path",
        "modality_image_paths",
        "input_file_signatures",
        "spacing",
        "affine",
        "padding_info",
        "normalization_info",
        "patch_index_table",
    }
)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA256 without loading a checkpoint into memory."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def anatomy_mask_sha256(mask: Tensor) -> str:
    """Hash a boolean DHW mask with its shape using a platform-stable payload."""
    tensor = torch.as_tensor(mask).detach().cpu().contiguous()
    if tensor.ndim != 3 or tensor.dtype != torch.bool:
        raise TypeError("anatomy mask hash requires a boolean [D,H,W] tensor")
    shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    payload = tensor.to(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(shape + b"\n" + payload).hexdigest()


def _check_forbidden_keys(value: Any, path: str = "cache") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            key_lower = key.lower()
            forbidden_variance = key_lower != "save_logvar" and (
                "logvar" in key_lower or "log_variance" in key_lower
            )
            if (
                key_lower in FORBIDDEN_EXACT_KEYS
                or key_lower.startswith("fsq_")
                or forbidden_variance
            ):
                raise ValueError(
                    f"Forbidden legacy/stochastic cache field at {path}.{key}: cache-v2 stores only deterministic latent_mu"
                )
            _check_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_forbidden_keys(child, f"{path}[{index}]")


def _tensor_payload_paths(value: Any, path: str = "cache") -> list[str]:
    if isinstance(value, Tensor):
        return [path]
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(_tensor_payload_paths(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_tensor_payload_paths(child, f"{path}[{index}]"))
    return paths


def _require_equal(cache: Mapping[str, Any], key: str, expected: Any) -> None:
    if cache[key] != expected:
        raise ValueError(f"{key} must be {expected!r}, got {cache[key]!r}")


def _validate_sha256(value: Any, key: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(
        (character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(
            f"{key} must be a 64-character hexadecimal SHA256, got {value!r}"
        )
    return text


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def input_snapshot_sha256(cache: Mapping[str, Any]) -> str:
    """Hash the complete case-level input snapshot shared by all modalities."""
    available_raw = cache.get("available_modalities")
    paths_raw = cache.get("modality_image_paths")
    signatures_raw = cache.get("input_file_signatures")
    if (
        isinstance(available_raw, (str, bytes))
        or not isinstance(available_raw, Sequence)
        or (not isinstance(paths_raw, Mapping))
        or (not isinstance(signatures_raw, Mapping))
    ):
        raise TypeError(
            "Input snapshot requires available_modalities, modality_image_paths, and input_file_signatures"
        )
    available = sorted((str(item).strip().lower() for item in available_raw))
    paths = {
        str(key).strip().lower(): os.path.normcase(os.path.normpath(str(value)))
        for key, value in paths_raw.items()
    }
    signatures: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in signatures_raw.items():
        if not isinstance(raw_value, Mapping):
            raise TypeError(f"input_file_signatures[{raw_key!r}] must be a mapping")
        signatures[str(raw_key).strip().lower()] = {
            "resolved_path": os.path.normcase(
                os.path.normpath(str(raw_value.get("resolved_path", "")))
            ),
            "size_bytes": int(raw_value.get("size_bytes", -1)),
            "mtime_ns": int(raw_value.get("mtime_ns", -1)),
        }
    snapshot = {
        "available_modalities": available,
        "modality_image_paths": {key: paths[key] for key in sorted(paths)},
        "input_file_signatures": {key: signatures[key] for key in sorted(signatures)},
        "crop_source_modality": str(cache.get("crop_source_modality", ""))
        .strip()
        .lower(),
    }
    return _stable_json_sha256(snapshot)


def _validate_tensor_payload(cache: Mapping[str, Any]) -> tuple[int, int, int]:
    latent = cache["latent_mu"]
    if not isinstance(latent, Tensor):
        raise TypeError(f"latent_mu must be a torch.Tensor, got {type(latent)!r}")
    if latent.ndim != 4 or int(latent.shape[0]) != LATENT_CHANNELS:
        raise ValueError(
            f"latent_mu must have shape [16,D_l,H_l,W_l], got {tuple(latent.shape)}. A 32-channel tensor is raw KL moments and must never be cached."
        )
    if latent.dtype != torch.float16:
        raise TypeError(
            f"latent_mu must be stored as torch.float16, got {latent.dtype}"
        )
    if not torch.isfinite(latent).all():
        raise ValueError("latent_mu contains NaN or Inf")
    latent_shape = tuple((int(value) for value in latent.shape[1:]))
    hard = cache["valid_mask_latent"]
    soft = cache["valid_mask_latent_soft"]
    if not isinstance(hard, Tensor) or not isinstance(soft, Tensor):
        raise TypeError(
            "valid_mask_latent and valid_mask_latent_soft must be torch.Tensor values"
        )
    if tuple(hard.shape) != latent_shape or tuple(soft.shape) != latent_shape:
        raise ValueError(
            f"Latent masks must match latent_mu spatial shape: latent={latent_shape}, hard={tuple(hard.shape)}, soft={tuple(soft.shape)}"
        )
    if hard.dtype != torch.bool:
        raise TypeError(f"valid_mask_latent must use torch.bool, got {hard.dtype}")
    if soft.dtype != torch.float16:
        raise TypeError(
            f"valid_mask_latent_soft must use torch.float16, got {soft.dtype}"
        )
    if not torch.isfinite(soft).all():
        raise ValueError("valid_mask_latent_soft contains NaN or Inf")
    if bool((soft < 0).any()) or bool((soft > 1).any()):
        raise ValueError(
            "valid_mask_latent_soft must contain occupancy ratios in [0,1]"
        )
    anatomy = cache["anatomy_mask_latent_hard"]
    if not isinstance(anatomy, Tensor):
        raise TypeError("anatomy_mask_latent_hard must be a torch.Tensor")
    if tuple(anatomy.shape) != latent_shape:
        raise ValueError(
            f"anatomy_mask_latent_hard must match latent_mu spatial shape: latent={latent_shape}, anatomy={tuple(anatomy.shape)}"
        )
    if anatomy.dtype != torch.bool:
        raise TypeError(
            f"anatomy_mask_latent_hard must use torch.bool, got {anatomy.dtype}"
        )
    if not bool(anatomy.any()):
        raise ValueError("anatomy_mask_latent_hard must contain at least one true cell")
    if bool((anatomy & ~hard).any()):
        raise ValueError(
            "anatomy_mask_latent_hard must be a subset of valid_mask_latent"
        )
    stored_mask_digest = _validate_sha256(
        cache["anatomy_mask_sha256"], "anatomy_mask_sha256"
    )
    if stored_mask_digest != anatomy_mask_sha256(anatomy):
        raise ValueError("anatomy_mask_sha256 does not match anatomy_mask_latent_hard")
    return latent_shape


def _validate_shapes(
    cache: Mapping[str, Any], latent_shape: tuple[int, int, int]
) -> None:
    original = as_3tuple(cache["original_shape_hwd"], "original_shape_hwd")
    padded = as_3tuple(cache["padded_shape_hwd"], "padded_shape_hwd")
    declared_latent = as_3tuple(cache["latent_shape_dhw"], "latent_shape_dhw")
    compression = as_3tuple(cache["compression_hwd"], "compression_hwd")
    if compression != COMPRESSION_HWD:
        raise ValueError(
            f"cache-v2 compression_hwd must be {COMPRESSION_HWD}, got {compression}"
        )
    if any((o > p for o, p in zip(original, padded))):
        raise ValueError(
            f"original_shape_hwd={original} exceeds padded_shape_hwd={padded}"
        )
    if any((p % c for p, c in zip(padded, compression))):
        raise ValueError(
            f"padded_shape_hwd={padded} is not divisible by compression_hwd={compression}"
        )
    expected_latent = (
        padded[2] // compression[2],
        padded[0] // compression[0],
        padded[1] // compression[1],
    )
    if declared_latent != latent_shape or expected_latent != latent_shape:
        raise ValueError(
            f"latent_shape_dhw mismatch: tensor={latent_shape}, declared={declared_latent}, expected_from_padded={expected_latent}"
        )
    dataset_key = str(cache["dataset"]).lower().replace("-", "").replace("_", "")
    if dataset_key in {"brats", "brats24", "brats2024"}:
        if original != BRATS24_SHAPE_HWD or padded != BRATS24_SHAPE_HWD:
            raise ValueError(
                f"BraTS24 is a no-padding fixed-grid cache; original and padded shape must both be {BRATS24_SHAPE_HWD}"
            )
        if latent_shape != (32, 32, 32):
            raise ValueError(
                f"BraTS24 latent shape must be (32,32,32), got {latent_shape}"
            )
    info = cache["padding_info"]
    if not isinstance(info, Mapping):
        raise TypeError("padding_info must be a mapping")
    widths = info.get("pad_width_hwd")
    expected_widths = [[0, p - o] for o, p in zip(original, padded)]
    if widths is None or [list(map(int, pair)) for pair in widths] != expected_widths:
        raise ValueError(
            f"padding_info.pad_width_hwd must be {expected_widths}, got {widths!r}"
        )
    if info.get("pad_side") != "end":
        raise ValueError("cache-v2 only supports case-shared end padding")
    if "pad_value" not in info or "policy" not in info:
        raise ValueError("padding_info must record pad_value and policy")
    if not math.isfinite(float(info["pad_value"])):
        raise ValueError("padding_info.pad_value must be finite")
    if not isinstance(info["policy"], str) or not info["policy"].strip():
        raise ValueError("padding_info.policy must be a non-empty string")


def _validate_medical_metadata(cache: Mapping[str, Any]) -> None:
    spacing = torch.as_tensor(cache["spacing"], dtype=torch.float64)
    if (
        tuple(spacing.shape) != (3,)
        or not torch.isfinite(spacing).all()
        or (not (spacing > 0).all())
    ):
        raise ValueError(
            f"spacing must contain three finite positive values, got {cache['spacing']!r}"
        )
    affine = torch.as_tensor(cache["affine"], dtype=torch.float64)
    if tuple(affine.shape) != (4, 4) or not torch.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not isinstance(cache["normalization_info"], Mapping):
        raise TypeError("normalization_info must be a mapping")
    if not isinstance(cache["padding_info"], Mapping):
        raise TypeError("padding_info must be a mapping")
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
    )

    anatomy_contract = cache["anatomy_mask_contract"]
    if not isinstance(anatomy_contract, Mapping) or not anatomy_contract:
        raise TypeError("anatomy_mask_contract must be a non-empty mapping")
    stored_anatomy_fingerprint = _validate_sha256(
        cache["anatomy_mask_contract_fingerprint"], "anatomy_mask_contract_fingerprint"
    )
    computed_anatomy_fingerprint = _stable_json_sha256(anatomy_contract)
    if stored_anatomy_fingerprint != computed_anatomy_fingerprint:
        raise ValueError(
            "anatomy_mask_contract_fingerprint does not match anatomy_mask_contract"
        )
    active_contract = anatomy_mask_contract(
        cache["dataset"], anatomy_contract.get("policy")
    )
    if _stable_json_sha256(active_contract) != computed_anatomy_fingerprint:
        raise ValueError(
            "Cached anatomy-mask contract differs from the active deterministic policy"
        )
    if (
        anatomy_mask_contract_fingerprint(
            cache["dataset"], anatomy_contract.get("policy")
        )
        != stored_anatomy_fingerprint
    ):
        raise ValueError("Cached anatomy-mask fingerprint differs from active code")
    statistics = cache["anatomy_mask_statistics"]
    if not isinstance(statistics, Mapping):
        raise TypeError("anatomy_mask_statistics must be a mapping")
    required_statistics = {
        "image_voxel_count",
        "image_voxel_ratio",
        "latent_cell_count",
        "latent_cell_ratio",
        "pir_support_cell_count",
        "pir_support_cell_ratio",
        "pir_valid_edge_count",
        "image_voxel_total",
        "latent_cell_total",
    }
    missing_statistics = sorted(required_statistics.difference(statistics))
    if missing_statistics:
        raise ValueError(
            f"anatomy_mask_statistics is missing fields: {missing_statistics}"
        )
    anatomy = torch.as_tensor(cache["anatomy_mask_latent_hard"]).bool()
    valid = torch.as_tensor(cache["valid_mask_latent"]).bool()
    expected_counts = {
        "latent_cell_count": int(anatomy.sum().item()),
        "pir_support_cell_count": int((anatomy & valid).sum().item()),
        "latent_cell_total": int(anatomy.numel()),
    }
    for key, expected in expected_counts.items():
        if int(statistics[key]) != expected:
            raise ValueError(
                f"anatomy_mask_statistics.{key}={statistics[key]!r}, expected {expected}"
            )
    for key in ("image_voxel_ratio", "latent_cell_ratio", "pir_support_cell_ratio"):
        ratio = float(statistics[key])
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError(f"anatomy_mask_statistics.{key} must lie in (0,1]")
    if int(statistics["image_voxel_count"]) <= 0:
        raise ValueError("anatomy_mask_statistics.image_voxel_count must be positive")
    image_total = int(statistics["image_voxel_total"])
    image_count = int(statistics["image_voxel_count"])
    if image_total <= 0 or image_count > image_total:
        raise ValueError("anatomy_mask_statistics image counts are inconsistent")
    expected_ratios = {
        "image_voxel_ratio": image_count / image_total,
        "latent_cell_ratio": int(statistics["latent_cell_count"])
        / int(statistics["latent_cell_total"]),
        "pir_support_cell_ratio": int(statistics["pir_support_cell_count"])
        / int(statistics["latent_cell_total"]),
    }
    for key, expected in expected_ratios.items():
        if not math.isclose(
            float(statistics[key]), expected, rel_tol=1e-06, abs_tol=1e-08
        ):
            raise ValueError(
                f"anatomy_mask_statistics.{key} does not match its count/total"
            )
    from streamrefine.training.pir import count_minimal_pir_edges

    expected_edge_count = int(count_minimal_pir_edges(valid, anatomy)[0].item())
    if (
        expected_edge_count <= 0
        or int(statistics["pir_valid_edge_count"]) != expected_edge_count
    ):
        raise ValueError(
            "anatomy_mask_statistics.pir_valid_edge_count does not match the configured valid & anatomy edge support"
        )
    for key in ("dataset", "case_id", "group_id", "split", "modality"):
        if not isinstance(cache[key], str) or not cache[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    available = cache["available_modalities"]
    if isinstance(available, (str, bytes)) or not isinstance(available, Sequence):
        raise TypeError("available_modalities must be a sequence of modality strings")
    if not all((isinstance(item, str) and item for item in available)):
        raise ValueError("available_modalities contains an invalid modality")
    if cache["modality"] not in available:
        raise ValueError("modality must be present in available_modalities")
    if not isinstance(cache["modality_image_path"], (str, Path)):
        raise TypeError("modality_image_path must be a string or pathlib.Path")
    image_paths = cache["modality_image_paths"]
    if not isinstance(image_paths, Mapping):
        raise TypeError("modality_image_paths must be a modality-to-path mapping")
    if set(map(str, image_paths)) != set(available):
        raise ValueError(
            "modality_image_paths keys must exactly match available_modalities"
        )
    if not all(
        (
            isinstance(path, (str, Path)) and str(path).strip()
            for path in image_paths.values()
        )
    ):
        raise ValueError("modality_image_paths contains an empty/non-path value")
    if str(image_paths[cache["modality"]]) != str(cache["modality_image_path"]):
        raise ValueError(
            "modality_image_path must match modality_image_paths[modality]"
        )
    signatures = cache["input_file_signatures"]
    if not isinstance(signatures, Mapping) or set(map(str, signatures)) != set(
        available
    ):
        raise ValueError(
            "input_file_signatures keys must exactly match available_modalities"
        )
    for modality, signature in signatures.items():
        if not isinstance(signature, Mapping):
            raise TypeError(f"input_file_signatures[{modality!r}] must be a mapping")
        path = signature.get("resolved_path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"input_file_signatures[{modality!r}].resolved_path is invalid"
            )
        for key in ("size_bytes", "mtime_ns"):
            value = signature.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"input_file_signatures[{modality!r}].{key} must be non-negative int"
                )
    stored_snapshot_digest = cache.get("input_snapshot_sha256")
    if stored_snapshot_digest is not None:
        stored_snapshot_digest = _validate_sha256(
            stored_snapshot_digest, "input_snapshot_sha256"
        )
        computed_snapshot_digest = input_snapshot_sha256(cache)
        if stored_snapshot_digest != computed_snapshot_digest:
            raise ValueError(
                f"input_snapshot_sha256 does not match the case input metadata: stored={stored_snapshot_digest}, computed={computed_snapshot_digest}"
            )


def _validate_patch_table(cache: Mapping[str, Any]) -> None:
    table = cache["patch_index_table"]
    if not isinstance(table, list) or not table:
        raise ValueError("patch_index_table must be a non-empty list")
    required = {
        "patch_id",
        "h0",
        "h1",
        "w0",
        "w1",
        "d0",
        "d1",
        "lh0",
        "lh1",
        "lw0",
        "lw1",
        "ld0",
        "ld1",
        "ih",
        "iw",
        "id",
        "num_h",
        "num_w",
        "num_d",
        "is_first_h",
        "is_last_h",
        "is_first_w",
        "is_last_w",
        "is_first_d",
        "is_last_d",
        "grid_index_hwd",
        "grid_shape_hwd",
        "image_start_hwd",
        "image_bbox_hwd",
        "latent_start_dhw",
        "latent_bbox_dhw",
    }
    padded = as_3tuple(cache["padded_shape_hwd"], "padded_shape_hwd")
    expected_table, _ = build_patch_index_table(padded)
    if len(table) != len(expected_table):
        raise ValueError(
            f"patch_index_table has {len(table)} entries; canonical cache-v2 geometry requires {len(expected_table)}"
        )
    comp_h, comp_w, comp_d = COMPRESSION_HWD
    seen_ids: set[int] = set()
    intervals_h: set[tuple[int, int]] = set()
    intervals_w: set[tuple[int, int]] = set()
    intervals_d: set[tuple[int, int]] = set()
    for expected_id, entry in enumerate(table):
        if not isinstance(entry, Mapping):
            raise TypeError(f"patch_index_table[{expected_id}] must be a mapping")
        missing = sorted(required.difference(entry))
        if missing:
            raise ValueError(
                f"patch_index_table[{expected_id}] missing fields: {missing}"
            )
        patch_id = int(entry["patch_id"])
        if patch_id != expected_id or patch_id in seen_ids:
            raise ValueError(
                "patch_index_table patch_id values must be unique and sequential"
            )
        seen_ids.add(patch_id)
        h0, h1 = (int(entry["h0"]), int(entry["h1"]))
        w0, w1 = (int(entry["w0"]), int(entry["w1"]))
        d0, d1 = (int(entry["d0"]), int(entry["d1"]))
        if (h1 - h0, w1 - w0, d1 - d0) != PATCH_SIZE_HWD:
            raise ValueError(
                f"Patch {patch_id} does not have image tile shape {PATCH_SIZE_HWD}"
            )
        if (
            h0 < 0
            or w0 < 0
            or d0 < 0
            or (h1 > padded[0])
            or (w1 > padded[1])
            or (d1 > padded[2])
        ):
            raise ValueError(f"Patch {patch_id} lies outside padded_shape_hwd={padded}")
        if h0 % comp_h or w0 % comp_w or d0 % comp_d:
            raise ValueError(f"Patch {patch_id} is not compression aligned")
        expected_latent = (
            d0 // comp_d,
            d1 // comp_d,
            h0 // comp_h,
            h1 // comp_h,
            w0 // comp_w,
            w1 // comp_w,
        )
        declared_latent = (
            int(entry["ld0"]),
            int(entry["ld1"]),
            int(entry["lh0"]),
            int(entry["lh1"]),
            int(entry["lw0"]),
            int(entry["lw1"]),
        )
        if declared_latent != expected_latent:
            raise ValueError(
                f"Patch {patch_id} latent bbox {declared_latent} does not map from image bbox"
            )
        if list(entry["image_start_hwd"]) != [h0, w0, d0]:
            raise ValueError(f"Patch {patch_id} image_start_hwd is inconsistent")
        if list(entry["latent_start_dhw"]) != [
            expected_latent[0],
            expected_latent[2],
            expected_latent[4],
        ]:
            raise ValueError(f"Patch {patch_id} latent_start_dhw is inconsistent")
        intervals_h.add((h0, h1))
        intervals_w.add((w0, w1))
        intervals_d.add((d0, d1))
        expected_entry = expected_table[expected_id]
        for key in required:
            if entry[key] != expected_entry[key]:
                raise ValueError(
                    f"Patch {patch_id} field {key!r} is inconsistent with canonical geometry: got {entry[key]!r}, expected {expected_entry[key]!r}"
                )
    for axis, intervals, full_size in (
        ("H", intervals_h, padded[0]),
        ("W", intervals_w, padded[1]),
        ("D", intervals_d, padded[2]),
    ):
        ordered = sorted(intervals)
        if ordered[0][0] != 0 or ordered[-1][1] != full_size:
            raise ValueError(f"Patch table does not cover the full {axis} axis")
        if any(
            (
                right_start > left_end
                for (_, left_end), (right_start, _) in zip(ordered, ordered[1:])
            )
        ):
            raise ValueError(f"Patch table has an uncovered gap on the {axis} axis")


def validate_cache_dict(
    cache: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_generation_contract_sha256: str | None = None,
) -> None:
    """Validate cache-v2 and reject stochastic, legacy FSQ, or mask drift."""
    if not isinstance(cache, Mapping):
        raise TypeError(f"cache must be a mapping, got {type(cache)!r}")
    _check_forbidden_keys(cache)
    missing = sorted(REQUIRED_KEYS.difference(cache))
    if missing:
        raise ValueError(f"Cache is missing required fields: {missing}")
    allowed_tensor_paths = {
        "cache.latent_mu",
        "cache.valid_mask_latent",
        "cache.valid_mask_latent_soft",
        "cache.anatomy_mask_latent_hard",
        "cache.affine",
        "cache.spacing",
    }
    unexpected_tensor_fields = sorted(
        (
            path
            for path in _tensor_payload_paths(cache)
            if path not in allowed_tensor_paths
        )
    )
    if unexpected_tensor_fields:
        raise ValueError(
            f"Cache-v2 permits no additional tensor payloads beyond latent_mu and its masks/medical metadata; got {unexpected_tensor_fields}"
        )
    _require_equal(cache, "cache_version", CACHE_VERSION)
    _require_equal(cache, "latent_layout", LATENT_LAYOUT)
    _require_equal(cache, "posterior_mode", POSTERIOR_MODE)
    _require_equal(cache, "save_logvar", False)
    _require_equal(cache, "latent_channels", LATENT_CHANNELS)
    _require_equal(cache, "cache_dtype", "float16")
    _require_equal(cache, "stitch_mode", STITCH_MODE)
    _require_equal(cache, "tokenizer_type", "vidtok_kl")
    _require_equal(cache, "tokenizer_causal", False)
    if str(cache["importance_mode"]).lower() not in {"hann", "gaussian"}:
        raise ValueError("importance_mode must be hann or gaussian")
    if as_3tuple(cache["patch_size_hwd"], "patch_size_hwd") != PATCH_SIZE_HWD:
        raise ValueError(f"patch_size_hwd must be {PATCH_SIZE_HWD}")
    if as_3tuple(cache["stride_hwd"], "stride_hwd") != STRIDE_HWD:
        raise ValueError(f"stride_hwd must be {STRIDE_HWD}")
    if not math.isclose(float(cache["overlap"]), 0.25, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("cache-v2 overlap must be exactly 0.25")
    digest = _validate_sha256(
        cache["tokenizer_checkpoint_sha256"], "tokenizer_checkpoint_sha256"
    )
    config_digest = _validate_sha256(
        cache["tokenizer_config_sha256"], "tokenizer_config_sha256"
    )
    generation_digest = _validate_sha256(
        cache["generation_contract_sha256"], "generation_contract_sha256"
    )
    generation_contract = cache["generation_contract"]
    if not isinstance(generation_contract, Mapping) or not generation_contract:
        raise ValueError("generation_contract must be a non-empty mapping")
    computed_generation_digest = _stable_json_sha256(generation_contract)
    if generation_digest != computed_generation_digest:
        raise ValueError(
            f"generation_contract_sha256 does not match generation_contract: stored={generation_digest}, computed={computed_generation_digest}"
        )
    if str(generation_contract.get("cache_version", "")) != CACHE_VERSION:
        raise ValueError(
            "generation_contract.cache_version does not match cache_version"
        )
    if (
        generation_contract.get("anatomy_mask_contract")
        != cache["anatomy_mask_contract"]
    ):
        raise ValueError(
            "generation_contract anatomy_mask_contract differs from the cache payload"
        )
    if str(generation_contract.get("anatomy_mask_contract_fingerprint", "")) != str(
        cache["anatomy_mask_contract_fingerprint"]
    ):
        raise ValueError(
            "generation_contract anatomy-mask fingerprint differs from the cache payload"
        )
    if expected_checkpoint_sha256 is not None:
        expected = _validate_sha256(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
        if digest != expected:
            raise ValueError(
                f"Tokenizer checkpoint hash mismatch: cache={digest}, expected={expected}"
            )
    if expected_config_sha256 is not None:
        expected_config = _validate_sha256(
            expected_config_sha256, "expected_config_sha256"
        )
        if config_digest != expected_config:
            raise ValueError(
                f"Tokenizer config hash mismatch: cache={config_digest}, expected={expected_config}"
            )
    if expected_generation_contract_sha256 is not None:
        expected_generation = _validate_sha256(
            expected_generation_contract_sha256, "expected_generation_contract_sha256"
        )
        if generation_digest != expected_generation:
            raise ValueError(
                f"Generation contract hash mismatch: cache={generation_digest}, expected={expected_generation}"
            )
    for key in ("tokenizer_config", "tokenizer_checkpoint", "tokenizer_git_commit"):
        if not isinstance(cache[key], (str, Path)) or not str(cache[key]):
            raise ValueError(f"{key} must be a non-empty string/path")
    if "tokenizer_repo_root" in cache and (
        not isinstance(cache["tokenizer_repo_root"], (str, Path))
        or not str(cache["tokenizer_repo_root"]).strip()
    ):
        raise ValueError(
            "tokenizer_repo_root must be a non-empty string/path when present"
        )
    latent_shape = _validate_tensor_payload(cache)
    _validate_shapes(cache, latent_shape)
    _validate_medical_metadata(cache)
    _validate_patch_table(cache)


def atomic_torch_save(
    cache: Mapping[str, Any],
    output_path: Path | str,
    *,
    overwrite: bool = False,
    validate: bool = True,
) -> Path:
    """Validate and atomically save one cache using a same-directory temp file."""
    if validate:
        validate_cache_dict(cache)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.writer.lock")
    try:
        lock_descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another process is writing this cache (or left a stale lock): {lock_path}"
        ) from exc
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        os.write(
            lock_descriptor,
            f"pid={os.getpid()} destination={destination}".encode("utf-8"),
        )
        os.fsync(lock_descriptor)
        if destination.exists() and (not overwrite):
            raise FileExistsError(
                f"Refusing to overwrite existing cache without overwrite=True: {destination}"
            )
        torch.save(dict(cache), temporary)
        if destination.exists() and (not overwrite):
            raise FileExistsError(
                f"Destination appeared during cache write; refusing overwrite: {destination}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        os.close(lock_descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return destination


def load_and_validate_cache(
    cache_path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_generation_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Load one trusted project cache and immediately enforce cache-v2."""
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cache does not exist: {path}")
    try:
        cache = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location=map_location)
    validate_cache_dict(
        cache,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_generation_contract_sha256=expected_generation_contract_sha256,
    )
    return dict(cache)


def to_json_safe(value: Any) -> Any:
    """Recursively convert cache metadata into strict JSON-compatible values."""
    if is_dataclass(value) and (not isinstance(value, type)):
        value = asdict(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float is not valid strict JSON: {value}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return to_json_safe(value.detach().cpu().item())
        return to_json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        converted = {str(key): to_json_safe(child) for key, child in value.items()}
        json.dumps(converted, allow_nan=False)
        return converted
    if isinstance(value, (list, tuple, set)):
        converted = [to_json_safe(child) for child in value]
        json.dumps(converted, allow_nan=False)
        return converted
    if hasattr(value, "item"):
        try:
            return to_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return to_json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Value of type {type(value)!r} is not JSON-safe")


json_safe = to_json_safe
