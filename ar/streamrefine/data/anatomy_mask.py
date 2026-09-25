"""Deterministic dataset-specific anatomy support for Minimal PIR.

The helpers in this module deliberately keep the mask outside the learned
StreamRefine path.  A case-shared image-space support is built during medical
preprocessing, follows the same spatial transforms as every modality, and is
then cached once as a hard latent mask.
"""

from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ANATOMY_MASK_SCHEMA_VERSION = "streamrefine_anatomy_mask_v1"
ANATOMY_MASK_POLICIES = {
    "brats24": "brats24_nonzero_union_v1",
    "synthrad": "synthrad_ct_body_v1",
    "autopet": "autopet_ct_body_shared_v1",
}
CT_BODY_THRESHOLD_HU = -500.0
LATENT_OCCUPANCY_THRESHOLD = 0.5
ANATOMY_MASK_COMPRESSION_HWD = (8, 8, 4)
CT_CONNECTIVITY = 1


def _canonical_dataset(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "brats": "brats24",
        "brats24": "brats24",
        "brats2024": "brats24",
        "synthrad": "synthrad",
        "synthrad2025": "synthrad",
        "autopet": "autopet",
        "autopet3": "autopet",
        "autopetiii": "autopet",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported anatomy-mask dataset: {value!r}")
    return aliases[key]


def canonical_anatomy_mask_policy(dataset: Any, policy: Any | None = None) -> str:
    dataset_key = _canonical_dataset(dataset)
    expected = ANATOMY_MASK_POLICIES[dataset_key]
    selected = expected if policy is None else str(policy).strip().lower()
    if selected != expected:
        raise ValueError(
            f"Dataset {dataset_key!r} requires anatomy-mask policy {expected!r}; got {policy!r}. Policy constants are versioned rather than tunable."
        )
    return selected


def _normalized_source_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def anatomy_mask_contract(dataset: Any, policy: Any | None = None) -> dict[str, Any]:
    dataset_key = _canonical_dataset(dataset)
    selected = canonical_anatomy_mask_policy(dataset_key, policy)
    common: dict[str, Any] = {
        "schema_version": ANATOMY_MASK_SCHEMA_VERSION,
        "dataset": dataset_key,
        "policy": selected,
        "implementation_source_sha256": _normalized_source_sha256(Path(__file__)),
        "stage": "after_orientation_before_case_shared_crop_and_normalization",
        "spatial_alignment": "same_case_shared_crop_resize_as_modalities",
        "nonfinite_input": "reject_case",
        "padding": "end_padding_false",
        "latent_downsample": {
            "compression_hwd": list(ANATOMY_MASK_COMPRESSION_HWD),
            "reduction": "block_mean_occupancy",
            "hard_rule": "occupancy_greater_equal_0.5",
            "threshold": LATENT_OCCUPANCY_THRESHOLD,
            "compression_order": "H,W,D",
        },
        "pir_support": "valid_mask_latent_hard_and_anatomy_mask_latent_hard",
        "learned_component": False,
        "inference_input": False,
    }
    if dataset_key == "brats24":
        common["extractor"] = {
            "source": "all_available_raw_mri_modalities",
            "operation": "finite_nonzero_union",
            "raw_nonzero_threshold": 0.0,
            "pre_normalized_background_cutoff": -0.999,
            "crop_restriction": "union_is_then_restricted_by_existing_t1c_foreground_crop",
        }
    else:
        common["extractor"] = {
            "source": "paired_ct_shared_across_modalities",
            "threshold_hu_strict_greater": CT_BODY_THRESHOLD_HU,
            "threshold_comparison_dtype": "preserve_loaded_input_dtype",
            "largest_component": {
                "dimension": 3,
                "connectivity": CT_CONNECTIVITY,
                "tie_break": "lowest_scan_order_label",
            },
            "hole_fill": "independent_axial_2d_binary_fill_holes",
            "pre_normalized_input": "rejected_requires_raw_hu",
        }
    return common


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def anatomy_mask_contract_fingerprint(dataset: Any, policy: Any | None = None) -> str:
    return _canonical_json_sha256(anatomy_mask_contract(dataset, policy))


def _validate_image_mapping(
    images_by_modality: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[int, ...]]:
    import torch

    if not isinstance(images_by_modality, Mapping) or not images_by_modality:
        raise ValueError("images_by_modality must be a non-empty mapping")
    images = {
        str(key).strip().lower(): torch.as_tensor(value)
        for key, value in images_by_modality.items()
    }
    reference_shape: tuple[int, ...] | None = None
    for modality, image in images.items():
        if image.ndim != 4 or int(image.shape[0]) != 1:
            raise ValueError(
                f"Anatomy-mask input {modality!r} must be [1,H,W,D], got {tuple(image.shape)}"
            )
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f"Anatomy-mask input {modality!r} contains NaN or Inf")
        shape = tuple((int(item) for item in image.shape))
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise ValueError(
                f"Anatomy-mask modalities must share one grid: {reference_shape} vs {shape}"
            )
    assert reference_shape is not None
    return (images, reference_shape)


def _ct_body_mask(ct_hwd: Any):
    import numpy as np
    import torch

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required by the deterministic CT body-mask preprocessing policy"
        ) from exc
    ct = torch.as_tensor(ct_hwd).detach().cpu()
    candidate = (ct[0] > CT_BODY_THRESHOLD_HU).numpy().astype(np.bool_, copy=False)
    structure = ndimage.generate_binary_structure(3, CT_CONNECTIVITY)
    labels, component_count = ndimage.label(candidate, structure=structure)
    if int(component_count) <= 0:
        raise ValueError(
            f"CT body policy found no voxel above {CT_BODY_THRESHOLD_HU:g} HU"
        )
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest = labels == largest_label
    filled = np.empty_like(largest, dtype=np.bool_)
    for d_index in range(int(largest.shape[2])):
        filled[:, :, d_index] = ndimage.binary_fill_holes(largest[:, :, d_index])
    if not bool(filled.any()):
        raise RuntimeError(
            "CT body policy produced an empty support after component filtering"
        )
    return torch.from_numpy(filled.copy()).unsqueeze(0)


def build_image_anatomy_mask_hwd(
    dataset: Any,
    images_by_modality: Mapping[str, Any],
    *,
    policy: Any | None = None,
    already_normalized: bool = False,
):
    """Build a boolean case-shared anatomy mask on the oriented raw image grid."""
    import torch

    dataset_key = _canonical_dataset(dataset)
    canonical_anatomy_mask_policy(dataset_key, policy)
    images, _ = _validate_image_mapping(images_by_modality)
    if dataset_key == "brats24":
        mask = None
        for image in images.values():
            current = image > -0.999 if already_normalized else image != 0
            mask = current if mask is None else mask | current
        assert mask is not None
        result = mask.bool()
    else:
        if already_normalized:
            raise ValueError(
                f"{dataset_key} CT body policy requires raw HU and rejects pre-normalized input"
            )
        if "ct" not in images:
            raise ValueError(
                f"{dataset_key} anatomy-mask policy requires paired CT; available={sorted(images)}"
            )
        result = _ct_body_mask(images["ct"]).to(device=images["ct"].device)
    if result.dtype != torch.bool:
        result = result.bool()
    if not bool(result.any()):
        raise ValueError("Deterministic anatomy support is empty")
    return result.contiguous()


class BuildDeterministicAnatomyMaskd:
    """MONAI-compatible deterministic dictionary transform with no learned state."""

    def __init__(
        self,
        *,
        dataset: Any,
        image_keys: Sequence[str],
        modality_by_key: Mapping[str, str],
        output_key: str = "anatomy_mask",
        policy: Any | None = None,
        already_normalized: bool = False,
    ) -> None:
        self.dataset = _canonical_dataset(dataset)
        self.image_keys = tuple((str(item) for item in image_keys))
        self.modality_by_key = {
            str(key): str(value).strip().lower()
            for key, value in modality_by_key.items()
        }
        self.output_key = str(output_key)
        self.policy = canonical_anatomy_mask_policy(self.dataset, policy)
        self.already_normalized = bool(already_normalized)

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(data)
        images = {self.modality_by_key[key]: result[key] for key in self.image_keys}
        result[self.output_key] = build_image_anatomy_mask_hwd(
            self.dataset,
            images,
            policy=self.policy,
            already_normalized=self.already_normalized,
        )
        return result


def downsample_anatomy_mask_to_latent(
    anatomy_mask_hwd: Any,
    padded_shape_hwd: Sequence[int],
    *,
    compression_hwd: Sequence[int] = ANATOMY_MASK_COMPRESSION_HWD,
):
    """Pad with false and apply the fixed 50% block-occupancy latent rule."""
    import torch
    from streamrefine.tokenizer.valid_mask import downsample_valid_mask_to_latent

    mask = torch.as_tensor(anatomy_mask_hwd).bool()
    if mask.ndim != 4 or int(mask.shape[0]) != 1:
        raise ValueError(f"anatomy_mask_hwd must be [1,H,W,D], got {tuple(mask.shape)}")
    padded = tuple((int(item) for item in padded_shape_hwd))
    if len(padded) != 3 or any((item <= 0 for item in padded)):
        raise ValueError(
            f"padded_shape_hwd must contain three positive values, got {padded}"
        )
    source = tuple((int(item) for item in mask.shape[1:]))
    compression = tuple((int(item) for item in compression_hwd))
    if compression != ANATOMY_MASK_COMPRESSION_HWD:
        raise ValueError(
            f"Anatomy-mask downsampling is a versioned fixed contract: expected compression_hwd={ANATOMY_MASK_COMPRESSION_HWD}, got {compression}"
        )
    if any((left > right for left, right in zip(source, padded))):
        raise ValueError(f"anatomy mask shape {source} exceeds padded shape {padded}")
    padded_mask = torch.zeros((1, *padded), dtype=torch.float32, device=mask.device)
    padded_mask[:, : source[0], : source[1], : source[2]] = mask.float()
    hard, occupancy = downsample_valid_mask_to_latent(
        padded_mask, comp_hwd=compression, threshold=LATENT_OCCUPANCY_THRESHOLD
    )
    if not bool(hard.any()):
        raise ValueError("Anatomy support vanished during fixed latent downsampling")
    return (hard.bool().contiguous(), occupancy.float().contiguous())


def compose_pir_mask_latent_hard(valid_mask: Any, anatomy_mask: Any):
    """Return the only support accepted by Minimal PIR: padding-valid AND anatomy."""
    import torch

    valid = torch.as_tensor(valid_mask)
    anatomy = torch.as_tensor(anatomy_mask)
    if valid.dtype != torch.bool or anatomy.dtype != torch.bool:
        raise TypeError("PIR valid/anatomy latent masks must both use torch.bool")
    if tuple(valid.shape) != tuple(anatomy.shape) or valid.ndim != 3:
        raise ValueError(
            f"PIR masks must share [D,H,W], got valid={tuple(valid.shape)}, anatomy={tuple(anatomy.shape)}"
        )
    support = (valid & anatomy).contiguous()
    if not bool(support.any()):
        raise ValueError("valid & anatomy PIR support is empty")
    return support


__all__ = [
    "ANATOMY_MASK_POLICIES",
    "ANATOMY_MASK_SCHEMA_VERSION",
    "ANATOMY_MASK_COMPRESSION_HWD",
    "BuildDeterministicAnatomyMaskd",
    "CT_BODY_THRESHOLD_HU",
    "LATENT_OCCUPANCY_THRESHOLD",
    "anatomy_mask_contract",
    "anatomy_mask_contract_fingerprint",
    "build_image_anatomy_mask_hwd",
    "canonical_anatomy_mask_policy",
    "compose_pir_mask_latent_hard",
    "downsample_anatomy_mask_to_latent",
]
