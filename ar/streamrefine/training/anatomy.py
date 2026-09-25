"""Versioned anatomy-target selection shared by training and offline audits."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from .pir import PIR_FORMULA, PIR_FORMULA_VERSION, PIR_OFFSETS_DHW

ANATOMY_METRICS = frozenset({"pir"})


def _normalized_source_sha256(path: Path) -> str:
    """Hash normalized source so numerical-code drift invalidates old artifacts."""
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _implementation_source_sha256(filename: str) -> str:
    return _normalized_source_sha256(Path(__file__).with_name(filename))


def _integration_source_sha256() -> dict[str, str]:
    """Bind raw-latent inversion, masking, and all three anatomy call sites."""
    streamrefine_root = Path(__file__).resolve().parents[1]
    paths = {
        "training_trainer": streamrefine_root / "training/trainer.py",
        "training_validation": streamrefine_root / "training/validation.py",
        "inference_runner": streamrefine_root / "inference/runner.py",
        "latent_inverse": streamrefine_root / "data/persistent_pair_dataset.py",
    }
    return {
        name: _normalized_source_sha256(path) for name, path in sorted(paths.items())
    }


def _pir_support_integration_source_sha256() -> dict[str, str]:
    """Bind every deterministic hand-off that creates or moves PIR support."""
    streamrefine_root = Path(__file__).resolve().parents[1]
    project_root = streamrefine_root.parent
    paths = {
        "anatomy_mask": streamrefine_root / "data/anatomy_mask.py",
        "cache_precompute": project_root / "tools/precompute_vidtok_kl_mean_latents.py",
        "cache_schema": streamrefine_root / "tokenizer/cache_schema.py",
        "data_augmentation": streamrefine_root / "data/augmentation.py",
        "latent_window_sampler": streamrefine_root / "data/latent_window_sampler.py",
        "medical_preprocessing": streamrefine_root / "data/preprocess_medical.py",
        "persistent_pair_dataset": streamrefine_root
        / "data/persistent_pair_dataset.py",
    }
    return {
        name: _normalized_source_sha256(path) for name, path in sorted(paths.items())
    }


def _pir_dataset_mask_contracts() -> dict[str, dict[str, Any]]:
    """Return the complete, canonical dataset-policy table embedded in PIR."""
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
    )

    result: dict[str, dict[str, Any]] = {}
    for dataset in ("autopet", "brats24", "synthrad"):
        contract = anatomy_mask_contract(dataset)
        result[dataset] = {
            "policy": str(contract["policy"]),
            "contract": contract,
            "contract_fingerprint": anatomy_mask_contract_fingerprint(
                dataset, contract["policy"]
            ),
        }
    return result


def canonical_anatomy_metric(value: Any) -> str:
    metric = str(value).strip().lower()
    if metric not in ANATOMY_METRICS:
        raise ValueError(
            f"anatomy metric must be one of {sorted(ANATOMY_METRICS)}, got {value!r}"
        )
    return metric


def anatomy_metric_contract(metric: Any) -> dict[str, Any]:
    """Return the JSON-stable formula identity bound into calibration artifacts."""
    canonical_anatomy_metric(metric)
    return {
        "metric": "pir",
        "formula_version": PIR_FORMULA_VERSION,
        "implementation_source_sha256": _implementation_source_sha256("pir.py"),
        "integration_source_sha256": _integration_source_sha256(),
        "formula": PIR_FORMULA,
        "latent": "frozen_vidtok_clean_posterior_mean_raw_C,D,H,W",
        "relation": "channel_cosine",
        "cosine_eps": 1e-08,
        "offsets_dhw": [list(offset) for offset in PIR_OFFSETS_DHW],
        "point_support": "valid_mask_latent_hard & anatomy_mask_latent_hard",
        "valid_edge": "both_endpoints_in_point_support",
        "anatomy_mask_contract_by_dataset": _pir_dataset_mask_contracts(),
        "pir_support_integration_source_sha256": _pir_support_integration_source_sha256(),
        "reduction_denominator": "valid_edge_count",
        "reference": "paired_target",
        "reliability": "source_target_relation_agreement_squared",
        "stop_gradient": True,
        "inference_computation": False,
    }


def anatomy_metric_contract_fingerprint(metric: Any) -> str:
    payload = json.dumps(
        anatomy_metric_contract(metric),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ANATOMY_METRICS",
    "anatomy_metric_contract",
    "anatomy_metric_contract_fingerprint",
    "canonical_anatomy_metric",
]
