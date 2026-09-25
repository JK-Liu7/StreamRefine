"""One-run warmup/calibration state, using the existing full-volume and v5 contracts."""

from __future__ import annotations
import copy
import contextlib
import hashlib
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping
from streamrefine.config import (
    canonical_config_json,
    config_fingerprint,
    pair_modalities,
)
from streamrefine.runtime import atomic_write_json, atomic_write_text

STATE_VERSION = "streamrefine_auto_calibration_v1"
CACHE_KEYS = (
    "tokenizer_checkpoint_sha256",
    "tokenizer_config_sha256",
    "generation_contract_sha256",
    "cache_schema",
    "anatomy_mask_policy",
    "anatomy_mask_contract_fingerprint",
)


def enabled(config: Mapping[str, Any]) -> bool:
    return bool(config["benefit"].get("auto_calibration", {}).get("enabled", False))


def adaptive_start_step(config: Mapping[str, Any]) -> int:
    return int(config["method"]["policy_warmup_steps"]) + int(
        config["benefit"].get("auto_calibration", {}).get("head_warmup_steps", 1000)
    )


def phase(config: Mapping[str, Any], step: int, calibration: Any) -> str:
    if calibration is None:
        return (
            "warmup"
            if step < int(config["method"]["policy_warmup_steps"])
            else "calibrating"
        )
    return "head_warmup" if step < adaptive_start_step(config) else "adaptive"


def expectation(
    config: Mapping[str, Any], provenance: Mapping[str, Any], artifact: Any
):
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
    )
    from streamrefine.method.calibration import CalibrationExpectation
    from streamrefine.training.anatomy import (
        anatomy_metric_contract,
        anatomy_metric_contract_fingerprint,
    )
    from streamrefine.training.translation import translation_metric_identity

    source, target = pair_modalities(config)
    metric = str(config["benefit"]["anatomy_metric"])
    mask = anatomy_mask_contract(
        config["data"]["dataset"], config["data"]["anatomy_mask_policy"]
    )
    return CalibrationExpectation(
        **translation_metric_identity(config["benefit"]["translation_metric"]),
        dataset=config["data"]["dataset"],
        source_modality=source,
        target_modality=target,
        split="train",
        k_max=int(config["rollout"]["k_max"]),
        fixed_horizon_checkpoint_identity=artifact.fixed_horizon_checkpoint_identity,
        latent_statistics_identity=provenance["latent_statistics_sha256"],
        cache_provenance={key: provenance[key] for key in CACHE_KEYS},
        trajectory_contract_fingerprint=artifact.trajectory_contract_fingerprint,
        anatomy_metric=metric,
        anatomy_contract=anatomy_metric_contract(metric),
        anatomy_contract_fingerprint=anatomy_metric_contract_fingerprint(metric),
        anatomy_mask_policy=config["data"]["anatomy_mask_policy"]
        if metric == "pir"
        else None,
        anatomy_mask_contract=mask if metric == "pir" else None,
        anatomy_mask_contract_fingerprint=anatomy_mask_contract_fingerprint(
            config["data"]["dataset"], config["data"]["anatomy_mask_policy"]
        )
        if metric == "pir"
        else None,
    )


def state_dict(
    config: Mapping[str, Any], step: int, calibration: Any
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "phase": phase(config, step, calibration),
        "calibration_step": int(config["method"]["policy_warmup_steps"]),
        "adaptive_start_step": adaptive_start_step(config),
        "artifact": None if calibration is None else calibration.to_dict(),
        "artifact_fingerprint": None
        if calibration is None
        else calibration.fingerprint,
    }


def restore_state(
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    state: Any,
    step: int,
    *,
    require_adaptive: bool = False,
):
    """Fail closed before resuming optimizer state or using a checkpoint for inference."""
    from streamrefine.method.calibration import BenefitCalibration

    if not isinstance(state, Mapping) or state.get("version") != STATE_VERSION:
        raise ValueError(
            "Single-run anatomy checkpoint lacks valid automatic calibration state"
        )
    boundary = int(config["method"]["policy_warmup_steps"])
    if state.get("calibration_step") != boundary or state.get(
        "adaptive_start_step"
    ) != adaptive_start_step(config):
        raise ValueError("Automatic calibration schedule differs from the checkpoint")
    artifact = state.get("artifact")
    calibration = (
        None if artifact is None else BenefitCalibration.from_mapping(artifact)
    )
    fingerprint = None if calibration is None else calibration.fingerprint
    if state.get("artifact_fingerprint") != fingerprint:
        raise ValueError("Embedded automatic calibration fingerprint mismatch")
    if (
        step < 0
        or (calibration is None and step > boundary)
        or (calibration is not None and step < boundary)
    ):
        raise ValueError(
            "Automatic calibration state is inconsistent with checkpoint step"
        )
    if state.get("phase") != phase(config, step, calibration):
        raise ValueError(
            "Automatic calibration phase is inconsistent with checkpoint step"
        )
    if require_adaptive and state["phase"] != "adaptive":
        raise ValueError(
            "Single-run anatomy checkpoint has not reached the adaptive phase"
        )
    if calibration is not None:
        calibration.assert_compatible(expectation(config, provenance, calibration))
        metadata = calibration.metadata.get("automatic_calibration", {})
        expected = {
            "config_fingerprint": config_fingerprint(config),
            "calibration_step": boundary,
            "adaptive_start_step": adaptive_start_step(config),
            "train_manifest_sha256": provenance["train_manifest_sha256"],
            "weight_source": "ema"
            if config["inference"].get("use_ema", True)
            else "model",
        }
        if not isinstance(metadata, Mapping) or any(
            (metadata.get(k) != v for k, v in expected.items())
        ):
            raise ValueError(
                "Automatic calibration does not match its training snapshot/config/data"
            )
    return calibration


def checkpoint_calibration(
    checkpoint: Mapping[str, Any], *, require_adaptive: bool = False
):
    config = checkpoint.get("resolved_config", {})
    if not isinstance(config, Mapping) or not enabled(
        {"benefit": config.get("benefit", {})}
    ):
        return None
    compatibility = checkpoint.get("compatibility", {})
    if compatibility.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError(
            "Single-run checkpoint resolved config does not match its compatibility contract"
        )
    return restore_state(
        config,
        compatibility.get("provenance", {}),
        checkpoint.get("policy_state", {}).get("auto_calibration"),
        int(checkpoint["step"]),
        require_adaptive=require_adaptive,
    )


def ensure_artifact_file(calibration: Any, output_dir: Path) -> Path:
    """Recreate a missing sidecar from the checkpoint; never replace a different one."""
    from streamrefine.method.calibration import (
        load_benefit_calibration,
        save_benefit_calibration,
    )

    path = output_dir / "calibration" / "benefit_calibration.json"
    if path.exists():
        if load_benefit_calibration(path).fingerprint != calibration.fingerprint:
            raise ValueError("Calibration sidecar differs from the resumed checkpoint")
    else:
        save_benefit_calibration(calibration, path)
    return path


def rank_zero_call(callback, *, rank: int, world_size: int):
    """Propagate artifact write/validation failures to all ranks before they continue."""
    result = [None]
    if rank == 0:
        try:
            result[0] = {"value": callback()}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if world_size > 1:
        import torch

        torch.distributed.broadcast_object_list(result, src=0)
    if not isinstance(result[0], Mapping):
        raise RuntimeError("Automatic calibration broadcast returned no result")
    if "error" in result[0]:
        raise RuntimeError("Automatic calibration failed: " + result[0]["error"])
    return result[0]["value"]


def run_automatic_calibration(
    *,
    model: Any,
    decoder: Any,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    snapshot_path: Path,
    output_dir: Path,
    device: Any,
    autocast_factory: Any,
    rank: int,
    world_size: int,
    progress: Any = None,
):
    """Caller installs snapshot raw/EMA weights and preserves the training RNG."""
    from streamrefine.training.calibration_progress import CalibrationProgress

    context = (
        contextlib.nullcontext(progress)
        if progress is not None
        else CalibrationProgress(
            output_dir=output_dir,
            step=int(config["method"]["policy_warmup_steps"]),
            rank=rank,
            world_size=world_size,
            device=device,
        )
    )
    with context as reporter:
        return _run_automatic_calibration(
            model=model,
            decoder=decoder,
            config=config,
            provenance=provenance,
            snapshot_path=snapshot_path,
            output_dir=output_dir,
            device=device,
            autocast_factory=autocast_factory,
            rank=rank,
            world_size=world_size,
            progress=reporter,
        )


def uses_latent_metrics_only(config: Mapping[str, Any]) -> bool:
    from streamrefine.training.anatomy import canonical_anatomy_metric
    from streamrefine.training.translation import translation_metric_identity

    return (
        translation_metric_identity(config["benefit"]["translation_metric"])[
            "translation_metric"
        ]
        == "latent_mae"
        and canonical_anatomy_metric(config["benefit"]["anatomy_metric"]) == "pir"
    )


def _run_automatic_calibration(
    *,
    model: Any,
    decoder: Any,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    snapshot_path: Path,
    output_dir: Path,
    device: Any,
    autocast_factory: Any,
    rank: int,
    world_size: int,
    progress: Any,
):
    from torch.utils.data import Subset
    from streamrefine.data.full_volume_dataset import build_full_volume_dataset
    from streamrefine.method.calibration import BenefitCalibration
    from streamrefine.training.validation import run_full_volume_validation

    latent_only = uses_latent_metrics_only(config)
    with progress.stage(
        "dataset_setup",
        metric_path="latent_only" if latent_only else "image_and_latent",
    ):
        dataset = build_full_volume_dataset(
            config, split="train", include_images=not latent_only
        )
    count = min(
        len(dataset), int(config["benefit"]["auto_calibration"].get("max_cases", 32))
    )
    if count <= 0:
        raise ValueError("Automatic calibration training dataset is empty")
    seed = int(config["inference"].get("seed", 3407))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), count))
    progress.emit(
        "roster",
        selected_cases=count,
        dataset_cases=len(dataset),
        k_max=int(config["rollout"]["k_max"]),
        metric_path="latent_only" if latent_only else "image_and_latent",
    )
    export_config = copy.deepcopy(dict(config))
    export_config["method"]["mode"] = "fixed_k"
    export_config["benefit"]["auto_calibration"]["enabled"] = False
    export_config["validation"].update(
        max_cases=count,
        compute_anatomy=True,
        threshold_sweep=[],
        selection_metric="full_trajectory_utility",
        selection_direction="min",
    )
    export_config["diagnostics"]["trajectory"]["enabled"] = False
    summary = run_full_volume_validation(
        model=model,
        decoder=decoder,
        dataset=Subset(dataset, indices),
        config=export_config,
        device=device,
        autocast_factory=autocast_factory,
        calibration=None,
        rank=rank,
        world_size=world_size,
        synchronize_failures=True,
        latent_metrics_only=latent_only,
        progress=progress,
    )

    def write_and_calibrate():
        from streamrefine.data.anatomy_mask import (
            anatomy_mask_contract,
            anatomy_mask_contract_fingerprint,
        )
        from streamrefine.inference.runner import _contract_fingerprint
        from streamrefine.method.calibration import load_benefit_calibration
        from streamrefine.training.anatomy import (
            anatomy_metric_contract,
            anatomy_metric_contract_fingerprint,
        )
        from streamrefine.training.checkpoint import checkpoint_weight_identity
        from streamrefine.training.translation import translation_metric_identity
        from tools.calibrate_streamrefine_benefit import calibrate

        directory = output_dir / "calibration"
        directory.mkdir(parents=True, exist_ok=True)
        cache = {key: provenance[key] for key in CACHE_KEYS}
        cache_fp = hashlib.sha256(
            canonical_config_json(cache).encode("utf-8")
        ).hexdigest()
        identity = checkpoint_weight_identity(
            snapshot_path, use_ema=bool(config["inference"].get("use_ema", True))
        )
        trajectory_fp = _contract_fingerprint(export_config, snapshot_path, provenance)
        source, target = pair_modalities(config)
        metric = str(config["benefit"]["anatomy_metric"])
        shared = {
            **translation_metric_identity(config["benefit"]["translation_metric"]),
            "dataset": config["data"]["dataset"],
            "source_modality": source,
            "target_modality": target,
            "split": "train",
            "k_max": int(config["rollout"]["k_max"]),
            "fixed_horizon_checkpoint_identity": identity,
            "latent_statistics_identity": provenance["latent_statistics_sha256"],
            "cache_provenance": cache,
            "cache_provenance_fingerprint": cache_fp,
            "trajectory_contract_fingerprint": trajectory_fp,
            "anatomy_metric": metric,
            "anatomy_contract": anatomy_metric_contract(metric),
            "anatomy_contract_fingerprint": anatomy_metric_contract_fingerprint(metric),
        }
        policy = config["data"]["anatomy_mask_policy"]
        shared.update(
            anatomy_mask_policy=policy,
            anatomy_mask_contract=anatomy_mask_contract(
                config["data"]["dataset"], policy
            ),
            anatomy_mask_contract_fingerprint=anatomy_mask_contract_fingerprint(
                config["data"]["dataset"], policy
            ),
        )
        records = summary["val/per_case_records"]
        if len(records) != count:
            raise ValueError(
                "Automatic calibration did not cover the selected training roster"
            )
        rows = []
        for record in records:
            for key in (
                "dataset",
                "source_modality",
                "target_modality",
                "translation_metric",
                "anatomy_metric",
            ):
                if record.get(key) != shared[key]:
                    raise ValueError(f"Automatic calibration record differs in {key}")
            row = {
                **shared,
                "case_id": record["case_id"],
                "states": [
                    {
                        "translation_loss": state["translation_loss"],
                        "anatomy_loss": state["anatomy_loss"],
                    }
                    for state in record["states"]
                ],
            }
            for key in ("anatomy_mask_policy", "anatomy_mask_contract_fingerprint"):
                if record.get(key) != shared[key]:
                    raise ValueError(f"Automatic calibration record differs in {key}")
            row.update(
                {
                    key: record[key]
                    for key in ("pir_support_cell_count", "pir_valid_edge_count")
                }
            )
            rows.append(row)
        trajectories = directory / "calibration_trajectories.jsonl"
        atomic_write_text(
            trajectories,
            "".join(
                (
                    json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    for row in rows
                )
            ),
        )
        cache_path = directory / "calibration_cache_provenance.json"
        atomic_write_json(cache_path, cache)
        metadata_path = directory / "automatic_metadata.json"
        atomic_write_json(
            metadata_path,
            {
                "automatic_calibration": {
                    "config_fingerprint": config_fingerprint(config),
                    "calibration_step": int(config["method"]["policy_warmup_steps"]),
                    "adaptive_start_step": adaptive_start_step(config),
                    "train_manifest_sha256": provenance["train_manifest_sha256"],
                    "weight_source": "ema"
                    if config["inference"].get("use_ema", True)
                    else "model",
                    "dataset_length": len(dataset),
                    "selected_dataset_indices": indices,
                    "snapshot_path": str(snapshot_path),
                    "scope": "training_full_volume_fixed_horizon",
                }
            },
        )
        output = directory / "benefit_calibration.json"
        report = calibrate(
            Namespace(
                trajectory_jsonl=[trajectories],
                output=output,
                dataset=config["data"]["dataset"],
                source_modality=source,
                target_modality=target,
                translation_metric=config["benefit"]["translation_metric"],
                anatomy_metric=metric,
                split="train",
                k_max=int(config["rollout"]["k_max"]),
                fixed_horizon_checkpoint_identity=identity,
                latent_statistics_identity=provenance["latent_statistics_sha256"],
                cache_provenance_json=cache_path,
                metadata_json=metadata_path,
                quantile=0.75,
                minimum_scale=1e-06,
                seed=seed,
            )
        )
        atomic_write_json(directory / "calibration_report.json", report)
        return load_benefit_calibration(output).to_dict()

    def timed_write_and_calibrate():
        with progress.stage("fit_scales_and_write"):
            return write_and_calibrate()

    with progress.stage("artifact_consensus"):
        payload = rank_zero_call(
            timed_write_and_calibrate, rank=rank, world_size=world_size
        )
    calibration = BenefitCalibration.from_mapping(payload)
    restore_state(
        config,
        provenance,
        state_dict(config, int(config["method"]["policy_warmup_steps"]), calibration),
        int(config["method"]["policy_warmup_steps"]),
    )
    return calibration
