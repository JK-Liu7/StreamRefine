"""Versioned L_k definitions, independent of differentiable editor losses."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

TRANSLATION_METRICS = frozenset({"latent_mae"})


def canonical_translation_metric(value: Any) -> str:
    if not isinstance(value, str) or value not in TRANSLATION_METRICS:
        raise ValueError("benefit.translation_metric must be 'latent_mae'")
    return value


def translation_metric_contract(metric: str) -> dict[str, Any]:
    name = canonical_translation_metric(metric)
    source = (
        Path(__file__)
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    contract = {
        "metric": name,
        "formula_version": f"streamrefine_{name}_translation_v1",
        "implementation_source_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
        "prediction": "committed_self_generated_clean_refinement_state",
        "reference": "paired_target",
        "stop_gradient": True,
        "reduction_dtype": "float32",
        "aggregation": "one_scalar_per_case_per_state",
    }
    contract.update(
        {
            "space": "training_statistics_channel_standardized_posterior_mean",
            "formula": "sum_c,i valid_soft[i]*abs(z_hat_model[c,i]-z_target_model[c,i]) / (C*sum_i valid_soft[i])",
            "support": "valid_mask_latent_soft",
            "training_scope": "aligned_latent_crop",
            "full_volume_scope": "gaussian_blended_global_latent_state",
            "statistics_identity": "bound_by_latent_statistics_identity",
        }
    )
    return contract


def translation_metric_identity(metric: str) -> dict[str, Any]:
    contract = translation_metric_contract(metric)
    serialized = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return {
        "translation_metric": canonical_translation_metric(metric),
        "translation_contract": contract,
        "translation_contract_fingerprint": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
    }


def latent_state_mae(states: Any, target: Any, valid_mask: Any):
    """Detached FP32 MAE for [B,K,C,D,H,W] or [B,C,D,H,W] model latents.

    These are generated clean states, not paired flow-query endpoint estimates.
    The target must use the same training statistics as the states. Soft valid
    occupancy excludes padding; anatomy/PIR support does not redefine L_k.
    """
    import torch

    if states.ndim not in (5, 6) or target.ndim != 5:
        raise ValueError(
            "latent MAE expects [B,K,C,D,H,W] or [B,C,D,H,W] and [B,C,D,H,W] target"
        )
    squeeze = states.ndim == 5
    if squeeze:
        states = states.unsqueeze(1)
    if states.shape[0] != target.shape[0] or states.shape[2:] != target.shape[1:]:
        raise ValueError("latent MAE state/target geometry differs")
    if tuple(valid_mask.shape) != (target.shape[0], *target.shape[-3:]):
        raise ValueError("latent MAE valid mask must be [B,D,H,W]")
    with torch.no_grad():
        mask = valid_mask.detach().to(device=states.device, dtype=torch.float32)
        if not bool(torch.isfinite(mask).all()) or bool(
            ((mask < 0) | (mask > 1)).any()
        ):
            raise ValueError("latent MAE valid mask must be finite and in [0,1]")
        mass = mask.flatten(1).sum(1)
        if bool((mass <= 0).any()):
            raise ValueError("Every latent MAE case must contain valid voxels")
        error = (
            states.detach().float() - target.detach().to(states.device).float()[:, None]
        ).abs()
        error = error.masked_fill(mask[:, None, None] == 0, 0)
        result = (error * mask[:, None, None]).flatten(2).sum(2) / (
            mass[:, None] * target.shape[1]
        )
    return result[:, 0] if squeeze else result
