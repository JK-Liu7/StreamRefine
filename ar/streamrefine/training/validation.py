"""Deployment-faithful full-volume validation runtime for StreamRefine.

The caller owns EMA swapping.  This module only runs the exact inference model,
sliding-window, decoder, and metric path under whichever weights are currently
installed in ``model``.  It reveals Kmax once with ``controller=None`` and delegates
all stopping, oracle, utility, regret, strata, and Pareto semantics to the single
standard-library implementation in :mod:`streamrefine.validation_policy`.
"""

from __future__ import annotations
import hashlib
import json
import math
import contextlib
import time
from typing import Any, Mapping, Sequence


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number, not bool")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _calibration_scale(calibration: Any, config: Mapping[str, Any], name: str) -> float:
    if calibration is None:
        value = config["benefit"].get(name, 1.0)
    elif isinstance(calibration, Mapping):
        value = calibration.get(name, config["benefit"].get(name, 1.0))
    else:
        value = getattr(calibration, name)
    return _finite_positive(value, f"benefit.{name}")


def _validate_and_order_by_dataset_index(
    gathered: Sequence[Sequence[Mapping[str, Any]]], *, expected_indices: Sequence[int]
) -> tuple[list[dict[str, Any]], int]:
    """Prove exact rank-strided coverage and reject every duplicate index."""
    by_index: dict[int, dict[str, Any]] = {}
    duplicate_count = 0
    for rank_records in gathered:
        for raw in rank_records:
            record = dict(raw)
            index = int(record["dataset_index"])
            previous = by_index.get(index)
            if previous is None:
                by_index[index] = record
                continue
            duplicate_count += 1
            raise RuntimeError(
                f"Exact rank-strided validation produced duplicate dataset index {index} on ranks {previous.get('rank')} and {record.get('rank')}"
            )
    expected = [int(index) for index in expected_indices]
    missing = sorted(set(expected).difference(by_index))
    extra = sorted(set(by_index).difference(expected))
    if missing or extra:
        raise RuntimeError(
            f"Distributed validation roster mismatch: missing={missing}, extra={extra}"
        )
    ordered = [by_index[index] for index in expected]
    case_to_index: dict[str, int] = {}
    for record in ordered:
        case_id = str(record["case_id"])
        index = int(record["dataset_index"])
        if case_id in case_to_index and case_to_index[case_id] != index:
            raise RuntimeError(
                f"Validation case_id {case_id!r} occurs at multiple dataset indices"
            )
        case_to_index[case_id] = index
    return (ordered, duplicate_count)


def _gather_records(
    local_records: Sequence[Mapping[str, Any]], *, rank: int, world_size: int
) -> list[list[Mapping[str, Any]]]:
    if int(world_size) == 1:
        if int(rank) != 0:
            raise ValueError("rank must be zero when world_size is one")
        return [list(local_records)]
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "world_size > 1 requires an initialized torch.distributed group"
        )
    actual_rank = torch.distributed.get_rank()
    actual_world = torch.distributed.get_world_size()
    if (int(rank), int(world_size)) != (actual_rank, actual_world):
        raise RuntimeError(
            "Validation rank/world_size differs from the initialized process group"
        )
    gathered: list[Any] = [None for _ in range(actual_world)]
    torch.distributed.all_gather_object(gathered, list(local_records))
    if not all((isinstance(records, list) for records in gathered)):
        raise RuntimeError("Distributed validation gather returned a malformed roster")
    return gathered


def _roster_fingerprint(
    records: Sequence[Mapping[str, Any]], dataset_length: int
) -> str:
    payload = json.dumps(
        {
            "dataset_length": int(dataset_length),
            "cases": [
                {
                    "dataset_index": int(record["dataset_index"]),
                    "case_id": str(record["case_id"]),
                    "pair_id": str(record.get("pair_id", "")),
                }
                for record in records
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_fields(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    requested = str(
        config["validation"].get("selection_metric", "adaptive_utility")
    ).strip()
    key = requested if requested.startswith("val/") else f"val/{requested}"
    aliases = {
        "val/adaptive_quality": "val/adaptive_mae",
        "val/full_trajectory_utility": "val/full/fixed_k_terminal_utility",
    }
    resolved = aliases.get(key, key)
    if resolved not in summary:
        raise ValueError(
            f"validation.selection_metric={requested!r} does not name a scalar deployment-validation metric"
        )
    value = summary[resolved]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Selection metric {resolved} is not scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"Selection metric {resolved} is not finite")
    direction = str(config["validation"].get("selection_direction", "min"))
    if direction not in {"min", "max"}:
        raise ValueError("validation.selection_direction must be 'min' or 'max'")
    return {
        "val/selection_metric": resolved,
        "val/selection_direction": direction,
        "val/selection_value": float(value),
    }


def run_full_volume_validation(
    *,
    model: Any,
    decoder: Any,
    dataset: Any,
    config: Mapping[str, Any],
    device: Any,
    autocast_factory: Any,
    calibration: Any,
    rank: int,
    world_size: int,
    synchronize_failures: bool = False,
    latent_metrics_only: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Run full-volume Kmax validation and return an all-rank ``val/...`` summary.

    ``dataset=None`` constructs the configured validation split.  Otherwise the
    supplied dataset is used, which makes the runtime independently testable with
    generated fixtures.  The globally selected roster is the first
    ``validation.max_cases`` rows and each rank evaluates ``roster[rank::world_size]``
    without DistributedSampler padding.

    ``latent_metrics_only`` is a calibration collector for latent MAE + PIR:
    it preserves the complete trajectory and losses, skips images, and returns
    only the gathered records, not a deployment-quality summary.
    """
    import torch
    from streamrefine.data.full_volume_dataset import build_full_volume_dataset
    from streamrefine.data.anatomy_mask import anatomy_mask_contract_fingerprint
    from streamrefine.data.persistent_pair_dataset import inverse_model_latent
    from streamrefine.diagnostics import summarize_trajectory_diagnostics
    from streamrefine.inference.metrics import (
        compute_volume_metrics,
        crop_to_original_hwd,
    )
    from streamrefine.inference.runner import ModelWindowPredictor
    from streamrefine.inference.sliding_window import (
        synchronized_sliding_refinement,
        tiled_decode_raw_latent,
    )
    from streamrefine.training.anatomy import (
        anatomy_metric_contract_fingerprint,
        canonical_anatomy_metric,
    )
    from streamrefine.training.pir import build_minimal_pir_reference, minimal_pir_loss
    from streamrefine.validation_policy import summarize_validation_records
    from streamrefine.training.translation import (
        latent_state_mae,
        translation_metric_identity,
    )

    translation_identity = translation_metric_identity(
        config["benefit"].get("translation_metric", "latent_mae")
    )
    anatomy_metric = canonical_anatomy_metric(
        config["benefit"].get("anatomy_metric", "pir")
    )
    compute_anatomy = str(config["method"]["mode"]) == "anatomy_aware" or bool(
        config["validation"].get("compute_anatomy", True)
    )
    if latent_metrics_only and (
        translation_identity["translation_metric"] != "latent_mae"
        or anatomy_metric != "pir"
        or (not compute_anatomy)
    ):
        raise ValueError(
            "Latent-only calibration requires latent_mae + PIR with anatomy enabled"
        )
    stage = (
        progress.stage
        if progress is not None
        else lambda *args, **kwargs: contextlib.nullcontext()
    )
    if dataset is None:
        dataset = build_full_volume_dataset(
            config, split="val", include_images=not latent_metrics_only
        )
    dataset_length = len(dataset)
    max_cases = int(config["validation"].get("max_cases", dataset_length))
    if max_cases <= 0:
        raise ValueError("validation.max_cases must be positive")
    selected_indices = list(range(min(dataset_length, max_cases)))
    if not selected_indices:
        raise ValueError("Full-volume validation dataset is empty")
    if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
        raise ValueError("rank must lie in [0, world_size)")
    local_indices = selected_indices[int(rank) :: int(world_size)]
    predictor = ModelWindowPredictor(model, config, autocast_factory)
    local_records: list[dict[str, Any]] = []
    previous_training = bool(getattr(model, "training", False))
    local_failure = None
    model.eval()
    try:
        if progress is not None:
            progress.start_cases(
                local_count=len(local_indices), global_count=len(selected_indices)
            )
        for case_number, dataset_index in enumerate(local_indices, start=1):
            case_started = time.perf_counter()
            case_fields = {
                "case_number": case_number,
                "local_cases": len(local_indices),
                "dataset_index": dataset_index,
            }
            predictor.reset_case()
            with stage("load_case", **case_fields):
                sample = dataset[dataset_index]
            case_id = str(sample["case_id"])
            case_fields["case_id"] = case_id
            with stage("trajectory", **case_fields):
                source_latent = torch.as_tensor(sample["source_latent_model"]).to(
                    device
                )
                valid_mask = torch.as_tensor(sample["valid_mask_latent_soft"]).to(
                    device
                )
                coordinates = torch.as_tensor(sample["global_coords_dhw"]).to(device)
                refinement = synchronized_sliding_refinement(
                    source_latent=source_latent,
                    valid_mask_soft=valid_mask,
                    global_coordinates=coordinates,
                    predictor=predictor,
                    controller=None,
                    k_max=int(config["rollout"]["k_max"]),
                    window_shape_dhw=config["inference"]["window_size_dhw"],
                    overlap=float(config["inference"]["overlap"]),
                    seed=int(config["inference"].get("seed", 3407)),
                    case_id=case_id,
                )
            predictor.reset_case()
            with stage(
                "latent_metrics" if latent_metrics_only else "decode_and_metrics",
                **case_fields,
            ):
                source_image = target_image = None
                if not latent_metrics_only:
                    original_hwd = tuple(
                        (int(value) for value in sample["original_shape_hwd"])
                    )
                    target_image = crop_to_original_hwd(
                        torch.as_tensor(sample["preprocessed_gt_image"]), original_hwd
                    )
                    source_image = crop_to_original_hwd(
                        torch.as_tensor(sample["preprocessed_source_image"]),
                        original_hwd,
                    )
                pir_reference = None
                if compute_anatomy:
                    with torch.inference_mode():
                        pir_reference = build_minimal_pir_reference(
                            torch.as_tensor(sample["source_latent_raw"])
                            .unsqueeze(0)
                            .to(device),
                            torch.as_tensor(sample["target_latent_raw"])
                            .unsqueeze(0)
                            .to(device),
                            torch.as_tensor(sample["valid_mask_latent_hard"])
                            .unsqueeze(0)
                            .to(device),
                            torch.as_tensor(sample["anatomy_mask_latent_hard"])
                            .unsqueeze(0)
                            .to(device),
                        )
                state_records: list[dict[str, Any]] = []
                for state in refinement.states:
                    raw_latent = inverse_model_latent(
                        state.latent.float(),
                        sample["latent_stats_mean"],
                        sample["latent_stats_std"],
                    )
                    metrics = {}
                    if not latent_metrics_only:
                        decoded_padded = tiled_decode_raw_latent(
                            raw_latent,
                            decoder=decoder,
                            tile_shape_dhw=config["vidtok"]["decode_window_dhw"],
                            overlap=float(config["vidtok"]["decode_overlap"]),
                            compression_dhw=(4, 8, 8),
                            batch_size=int(config["vidtok"]["decode_batch_size"]),
                            precision=str(config["train"]["precision"]),
                        )
                        decoded = crop_to_original_hwd(decoded_padded, original_hwd)
                        prediction_metric = decoded.unsqueeze(0).to(device)
                        target_metric = target_image.unsqueeze(0).to(device)
                        metrics = compute_volume_metrics(
                            prediction_metric,
                            target_metric,
                            enabled={
                                "mae": True,
                                "psnr": True,
                                "ssim": True,
                                "lpips": False,
                            },
                            data_range=float(config["metrics"].get("data_range", 2.0)),
                        )
                        translation_loss = float(
                            (prediction_metric - target_metric).abs().mean().item()
                        )
                    if translation_identity["translation_metric"] == "latent_mae":
                        translation_loss = float(
                            latent_state_mae(
                                state.latent.unsqueeze(0),
                                torch.as_tensor(
                                    sample["target_latent_model"]
                                ).unsqueeze(0),
                                valid_mask.unsqueeze(0),
                            )[0].item()
                        )
                    anatomy_loss = None
                    if pir_reference is not None:
                        with torch.inference_mode():
                            anatomy_loss = float(
                                minimal_pir_loss(
                                    raw_latent.unsqueeze(0), pir_reference
                                )[0].item()
                            )
                    state_records.append(
                        {
                            "step": int(state.step),
                            "benefit_score": float(state.benefit_score),
                            "translation_loss": translation_loss,
                            "translation_metric": translation_identity[
                                "translation_metric"
                            ],
                            "anatomy_loss": anatomy_loss,
                            "anatomy_metric": anatomy_metric
                            if compute_anatomy
                            else None,
                            "metrics": metrics,
                            "window_score_variance": float(state.window_score_variance),
                            "seam_mse": float(state.seam_mse),
                            "sampling_seconds": float(state.sampling_seconds),
                            "window_count": int(state.window_count),
                            "model_evaluations": int(state.model_evaluations),
                        }
                    )
                    del raw_latent
                    if not latent_metrics_only:
                        del decoded_padded, decoded, prediction_metric, target_metric
            if len(state_records) != int(config["rollout"]["k_max"]):
                raise RuntimeError(
                    f"controller=None returned {len(state_records)} states instead of Kmax"
                )
            local_records.append(
                {
                    "dataset_index": int(dataset_index),
                    **translation_identity,
                    "rank": int(rank),
                    "case_id": case_id,
                    "pair_id": str(sample.get("pair_id", "")),
                    "dataset": str(sample.get("dataset", config["data"]["dataset"])),
                    "source_modality": str(sample.get("source_modality", "")),
                    "target_modality": str(sample.get("target_modality", "")),
                    "difficulty": float(state_records[0]["translation_loss"]),
                    "anatomy_metric": anatomy_metric if compute_anatomy else None,
                    "anatomy_mask_policy": str(
                        sample.get(
                            "anatomy_mask_policy", config["data"]["anatomy_mask_policy"]
                        )
                    ),
                    "anatomy_mask_contract_fingerprint": str(
                        sample.get(
                            "anatomy_mask_contract_fingerprint",
                            anatomy_mask_contract_fingerprint(
                                config["data"]["dataset"],
                                config["data"]["anatomy_mask_policy"],
                            ),
                        )
                    ),
                    "pir_support_cell_count": int(
                        torch.as_tensor(sample["pir_mask_latent_hard"]).sum().item()
                    )
                    if pir_reference is not None
                    else None,
                    "pir_valid_edge_count": int(pir_reference.edge_count[0].item())
                    if pir_reference is not None
                    else None,
                    "states": state_records,
                }
            )
            predictor.reset_case()
            del refinement, state, source_latent, valid_mask, coordinates
            del pir_reference, source_image, target_image, sample
            if progress is not None:
                progress.case_complete(
                    **case_fields,
                    states=len(state_records),
                    elapsed_sec=time.perf_counter() - case_started,
                )
    except Exception as exc:
        if not synchronize_failures or int(world_size) == 1:
            raise
        local_failure = f"rank {rank}: {type(exc).__name__}: {exc}"
    finally:
        predictor.reset_case()
        model.train(previous_training)
        if progress is not None:
            progress.finish_cases()
    with stage("gather_records", local_cases=len(local_records)):
        if synchronize_failures and int(world_size) > 1:
            failures: list[Any] = [None] * int(world_size)
            torch.distributed.all_gather_object(failures, local_failure)
            if any((failure is not None for failure in failures)):
                raise RuntimeError(
                    "Full-volume calibration failed: "
                    + "; ".join(
                        (failure for failure in failures if failure is not None)
                    )
                )
        gathered = _gather_records(
            local_records, rank=int(rank), world_size=int(world_size)
        )
        records, duplicate_count = _validate_and_order_by_dataset_index(
            gathered, expected_indices=selected_indices
        )
    if latent_metrics_only:
        return {
            "val/per_case_records": records,
            "val/case_count": len(records),
            "val/duplicate_record_count": duplicate_count,
            "val/world_size": int(world_size),
            "val/roster_fingerprint": _roster_fingerprint(records, dataset_length),
        }
    mode = str(config["method"]["mode"])
    lambda_anatomy = (
        float(config["benefit"].get("lambda_anatomy", 0.0))
        if mode == "anatomy_aware"
        else 0.0
    )
    translation_scale = _calibration_scale(calibration, config, "translation_scale")
    anatomy_scale = (
        _calibration_scale(calibration, config, "anatomy_scale")
        if lambda_anatomy > 0.0
        else None
    )
    summary = summarize_validation_records(
        records,
        mode=mode,
        stop_threshold=float(config["method"].get("stop_threshold", 0.0)),
        k_max=int(config["rollout"]["k_max"]),
        translation_scale=translation_scale,
        anatomy_scale=anatomy_scale,
        lambda_anatomy=lambda_anatomy,
        lambda_compute=float(config["benefit"].get("lambda_compute", 0.0)),
        oracle_tolerance=float(config["validation"].get("oracle_tolerance", 1e-08)),
        metric_names=("mae", "psnr", "ssim"),
        threshold_sweep=config["validation"].get("threshold_sweep", ()),
    )
    if bool(config["diagnostics"]["trajectory"].get("enabled", True)):
        anatomy_calibrated = compute_anatomy and calibration is not None
        diagnostic_lambda_anatomy = (
            float(config["benefit"].get("lambda_anatomy", 0.0))
            if anatomy_calibrated
            else 0.0
        )
        diagnostic_anatomy_scale = (
            _calibration_scale(calibration, config, "anatomy_scale")
            if diagnostic_lambda_anatomy > 0.0
            else None
        )
        anatomy_unavailable_reason = None
        if not compute_anatomy:
            anatomy_unavailable_reason = "anatomy_diagnostics_disabled"
        elif calibration is None:
            anatomy_unavailable_reason = (
                "calibration_artifact_required_for_anatomy_oracle"
            )
        mechanism = summarize_trajectory_diagnostics(
            records,
            translation_metric=translation_identity["translation_metric"],
            mode=mode,
            stop_threshold=float(config["method"].get("stop_threshold", 0.0)),
            k_max=int(config["rollout"]["k_max"]),
            translation_scale=translation_scale,
            anatomy_scale=diagnostic_anatomy_scale,
            lambda_anatomy=diagnostic_lambda_anatomy,
            lambda_compute=float(config["benefit"].get("lambda_compute", 0.0)),
            oracle_tolerance=float(config["validation"].get("oracle_tolerance", 1e-08)),
            anatomy_metric=anatomy_metric,
            pir_utility_key=None,
            pir_formula=None,
            anatomy_unavailable_reason=anatomy_unavailable_reason,
        )
        per_case_diagnostics = {
            str(row["case_id"]): dict(row)
            for row in mechanism.pop("diagnostics/per_case")
        }
        for record in records:
            case_id = str(record["case_id"])
            record["diagnostics"] = per_case_diagnostics[case_id]
        summary.update(mechanism)
        summary["val/wrong_stop_rate"] = mechanism["wrong_stop/rate"]
        summary["val/wrong_stop_regret"] = mechanism["wrong_stop/regret"]
    summary["pareto/quality_step"] = summary["val/quality_step_curve"]
    for point in summary["val/full/fixed_k_curve"]:
        step = int(point["steps"])
        summary[f"quality/{step}"] = point.get("translation_loss")
        summary[f"anatomy_drift/{step}"] = point.get("anatomy_loss")
    summary["val/full_trajectory_utility"] = summary[
        "val/full/fixed_k_terminal_utility"
    ]
    summary["val/translation_metric"] = translation_identity["translation_metric"]
    summary["val/translation_contract_fingerprint"] = translation_identity[
        "translation_contract_fingerprint"
    ]
    summary["val/case_count"] = int(summary["case_count"])
    summary.update(_selection_fields(summary, config))
    summary["val/duplicate_record_count"] = int(duplicate_count)
    summary["val/deduplicated_case_count"] = int(summary["case_count"])
    summary["val/per_case_records"] = records
    summary["val/roster_fingerprint"] = _roster_fingerprint(records, dataset_length)
    summary["val/world_size"] = int(world_size)
    summary["val/anatomy_metric"] = anatomy_metric
    summary["val/anatomy_mask_policy"] = str(config["data"]["anatomy_mask_policy"])
    summary["val/anatomy_mask_contract_fingerprint"] = (
        anatomy_mask_contract_fingerprint(
            config["data"]["dataset"], config["data"]["anatomy_mask_policy"]
        )
    )
    summary["val/anatomy_contract_fingerprint"] = anatomy_metric_contract_fingerprint(
        anatomy_metric
    )
    summary["val/weight_source"] = (
        "ema" if bool(config["inference"].get("use_ema", True)) else "raw"
    )
    return summary


__all__ = ["run_full_volume_validation"]
