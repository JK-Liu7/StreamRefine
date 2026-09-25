"""Shared trainer for all StreamRefine modes.

The implementation keeps the two gradient boundaries explicit: complete rollout
states are generated without graphs and committed detached, while each reached outer
state receives one independent analytic Rectified-Flow query with editor gradients.
"""

from __future__ import annotations
import copy
import contextlib
import datetime
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from streamrefine.config import config_fingerprint, pair_modalities
from streamrefine.runtime import atomic_write_json, configure_precision, seed_everything
from streamrefine.training.translation import (
    canonical_translation_metric,
    latent_state_mae,
    translation_metric_identity,
)
from . import auto_calibration as automatic
from .checkpoint import (
    atomic_torch_save,
    build_compatibility_contract,
    capture_rng_state,
    checkpoint_weight_identity,
    checkpoint_payload,
    initialize_model_checkpoint,
    resolve_checkpoint_action,
    restore_rng_state,
    restore_training_checkpoint,
)
from .logging import (
    RunningMeans,
    StructuredTrainingLog,
    format_progress,
    gpu_memory_snapshot,
    timed_training_batches,
    training_log_due,
)


@dataclass
class BatchObjective:
    total: Any
    flow: Any
    latent_l1: Any
    stop: Any
    logs: dict[str, float]


class ExponentialMovingAverage:
    def __init__(self, model: Any, decay: float = 0.9999) -> None:
        import torch

        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must lie in (0,1)")
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }
        self.nonfloating = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not torch.is_floating_point(value)
        }

    def update(self, model: Any) -> None:
        with __import__("torch").no_grad():
            for name, value in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].lerp_(value.detach().float(), 1.0 - self.decay)
                else:
                    self.nonfloating[name] = value.detach().clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "nonfloating": self.nonfloating,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("Checkpoint EMA decay differs from configuration")
        saved_shadow = state.get("shadow")
        saved_nonfloating = state.get("nonfloating")
        if not isinstance(saved_shadow, Mapping) or not isinstance(
            saved_nonfloating, Mapping
        ):
            raise ValueError("Checkpoint EMA payload is malformed")
        if set(saved_shadow) != set(self.shadow) or set(saved_nonfloating) != set(
            self.nonfloating
        ):
            raise ValueError("Checkpoint EMA keys differ from the active model")
        self.shadow = {
            name: saved_shadow[name]
            .detach()
            .to(device=current.device, dtype=current.dtype)
            .clone()
            for name, current in self.shadow.items()
        }
        self.nonfloating = {
            name: saved_nonfloating[name]
            .detach()
            .to(device=current.device, dtype=current.dtype)
            .clone()
            for name, current in self.nonfloating.items()
        }

    @contextlib.contextmanager
    def average_parameters(self, model: Any):
        """Temporarily evaluate the live module with EMA weights.

        Parameter and buffer objects are retained so optimizer references remain
        valid.  Raw values are copied to CPU before the swap and restored in a
        ``finally`` block; the EMA payload itself is never mutated.
        """
        import torch

        live_state = model.state_dict()
        averaged = {**self.nonfloating, **self.shadow}
        if set(live_state) != set(averaged):
            missing = sorted(set(live_state).difference(averaged))
            unexpected = sorted(set(averaged).difference(live_state))
            raise ValueError(
                f"EMA/model state keys differ before validation: missing={missing}, unexpected={unexpected}"
            )
        for name, live in live_state.items():
            candidate = averaged[name]
            if tuple(candidate.shape) != tuple(live.shape):
                raise ValueError(
                    f"EMA/model tensor shape differs for {name}: {tuple(candidate.shape)} != {tuple(live.shape)}"
                )
        was_training = bool(model.training)
        raw_backup = {
            name: value.detach().cpu().clone() for name, value in live_state.items()
        }
        try:
            with torch.no_grad():
                for name, live in live_state.items():
                    live.copy_(averaged[name].to(device=live.device, dtype=live.dtype))
            yield model
        finally:
            with torch.no_grad():
                restored = model.state_dict()
                for name, live in restored.items():
                    live.copy_(
                        raw_backup[name].to(device=live.device, dtype=live.dtype)
                    )
            model.train(was_training)


def _unwrap(model: Any) -> Any:
    return getattr(model, "module", model)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate_streamrefine_batch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    if not rows:
        raise ValueError("Cannot collate an empty StreamRefine batch")
    result: dict[str, Any] = {}
    keys = set.intersection(*(set(row) for row in rows))
    for key in keys:
        values = [row[key] for row in rows]
        if all((torch.is_tensor(value) for value in values)):
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                result[key] = values
        elif all((isinstance(value, bool) for value in values)):
            result[key] = torch.as_tensor(values, dtype=torch.bool)
        elif all((isinstance(value, (int, float)) for value in values)):
            result[key] = torch.as_tensor(values)
        else:
            result[key] = values
    return result


def build_model(config: Mapping[str, Any]):
    from streamrefine.models import VolumeCausalDiT

    model_cfg = config["model"]
    return VolumeCausalDiT(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        patch_size_dhw=model_cfg["patch_size_dhw"],
        hidden_size=int(model_cfg["hidden_size"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        rope_base=float(model_cfg["rope_base"]),
        max_refinement_steps=int(model_cfg["max_refinement_steps"]),
        detach_benefit_input=bool(model_cfg.get("detach_benefit_input", True)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
    )


def build_vidtok_decoder(config: Mapping[str, Any], device: Any):
    from streamrefine.tokenizer.vidtok_kl_wrapper import load_vidtok_kl_mean_encoder

    vidtok = config["vidtok"]
    config_path = str(vidtok.get("config", ""))
    checkpoint = str(vidtok.get("checkpoint", ""))
    if not config_path or not Path(config_path).is_file():
        raise FileNotFoundError(f"VidTok model config does not exist: {config_path!r}")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"VidTok checkpoint does not exist: {checkpoint!r}")
    repo = Path(str(vidtok.get("repo", ""))).expanduser()
    return load_vidtok_kl_mean_encoder(
        config_path,
        checkpoint,
        vidtok_root=repo,
        device=device,
        precision=str(config["train"]["precision"]),
        latent_channels=int(config["latent"]["channels"]),
        lightweight_loss=True,
    )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _per_case_masked(error: Any, valid_mask: Any):
    import torch

    mask = valid_mask.to(device=error.device, dtype=error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask, error.shape)
    numerator = (error * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError("Every training case must contain valid latent voxels")
    return numerator / denominator


def _distributed_weighted_ratio(numerator: Any, denominator: Any):
    import torch

    denominator = (
        torch.as_tensor(denominator, device=numerator.device, dtype=numerator.dtype)
        .detach()
        .clone()
    )
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    return numerator * float(world_size) / denominator.clamp_min(1.0)


def _weighted_state_mean(
    values: list[Any], weights: list[Any], *, ht_population_size: int | None = None
):
    import torch

    numerator = torch.stack(
        [(value * weight).sum() for value, weight in zip(values, weights)]
    ).sum()
    denominator = (
        numerator.new_tensor(float(ht_population_size))
        if ht_population_size is not None
        else torch.stack([weight.sum() for weight in weights]).sum()
    )
    return _distributed_weighted_ratio(numerator, denominator)


def _broadcast_model_parameters(model: Any, world_size: int) -> None:
    import torch

    if int(world_size) <= 1:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            torch.distributed.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            torch.distributed.broadcast(buffer.data, src=0)


def _synchronize_gradients(model: Any, world_size: int) -> None:
    """Average accumulated gradients without wrapping repeated forwards in DDP."""
    import torch

    if int(world_size) <= 1:
        return
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        torch.distributed.all_reduce(gradient, op=torch.distributed.ReduceOp.SUM)
        gradient.div_(float(world_size))
        if parameter.grad is None:
            parameter.grad = gradient


def _inverse_batch(model_latent: Any, mean: Any, std: Any):
    extra_state_axes = model_latent.ndim - 5
    shape = [mean.shape[0]] + [1] * extra_state_axes + [mean.shape[1], 1, 1, 1]
    return model_latent * std.reshape(shape) + mean.reshape(shape)


def _controller(config: Mapping[str, Any]):
    from streamrefine.method import AdaptiveStopController, ControllerConfig

    method = config["method"]
    return AdaptiveStopController(
        ControllerConfig(
            mode=method["mode"],
            k_max=int(config["rollout"]["k_max"]),
            fixed_steps=int(config["rollout"]["k_max"]),
            sentinel_probability=float(method["sentinel_probability"]),
            stop_threshold=float(method["stop_threshold"]),
            policy_warmup_steps=automatic.adaptive_start_step(config)
            if automatic.enabled(config)
            else int(method["policy_warmup_steps"]),
        )
    )


def compute_batch_objective(
    *,
    model: Any,
    decoder: Any,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    global_step: int,
    calibration: Any = None,
    controller: Any = None,
    trajectory_randomness: Mapping[str, Any] | None = None,
) -> BatchObjective:
    import torch
    from streamrefine.generation import RectifiedFlowScheduler, WholeVolumeSelfForcing
    from streamrefine.method import (
        compute_editor_reach_weights,
        compute_suffix_stop_weights,
        continuation_benefit_targets,
        smooth_l1_benefit_loss,
    )
    from streamrefine.training.pir import (
        build_minimal_pir_reference,
        minimal_pir_trajectory,
    )

    source = batch["source_latent_model"].float()
    target = batch["target_latent_model"].float()
    valid = batch["valid_mask_latent_soft"].float()
    coordinates = batch["global_coords_dhw"].float()
    active_controller = controller or _controller(config)
    adapter = WholeVolumeSelfForcing(
        model,
        RectifiedFlowScheduler(int(config["rollout"]["inner_steps"])),
        inner_steps=int(config["rollout"]["inner_steps"]),
        k_max=int(config["rollout"]["k_max"]),
        refinement_start=config["rollout"].get("refinement_start", "previous_state"),
        sync_flow_time_across_ranks=bool(
            config["rollout"].get("sync_flow_time_across_ranks", True)
        ),
    )
    trajectory = adapter.run_training_trajectory(
        source,
        target,
        controller=active_controller,
        global_step=int(global_step),
        rollout_noises=None
        if trajectory_randomness is None
        else trajectory_randomness.get("rollout_noises"),
        paired_noises=None
        if trajectory_randomness is None
        else trajectory_randomness.get("paired_noises"),
        paired_flow_times=None
        if trajectory_randomness is None
        else trajectory_randomness.get("paired_flow_times"),
        valid_mask=valid,
        global_coordinates=coordinates,
    )
    final_policy = trajectory.final_policy_state
    if final_policy is None:
        raise RuntimeError(
            "Shared trainer requires an explicit controller in every mode"
        )
    mode = str(config["method"]["mode"])
    epsilon = float(config["method"]["sentinel_probability"])
    flow_values, latent_values, editor_weights = ([], [], [])
    raw_flow_values, raw_latent_values = ([], [])
    for state_index, step in enumerate(trajectory.steps):
        query = step.paired_query
        flow_case = _per_case_masked(
            (query.predicted_velocity - query.target_velocity).square(), valid
        )
        latent_case = _per_case_masked((query.endpoint_estimate - target).abs(), valid)
        after_stop = (final_policy.first_stop_step > 0) & (
            state_index + 1 > final_policy.first_stop_step
        )
        weights = compute_editor_reach_weights(
            mode,
            reached=step.reached_mask,
            after_proposed_stop=after_stop,
            sentinel_selected=final_policy.sentinel_selected,
            sentinel_probability=epsilon,
            dtype=flow_case.dtype,
        )
        flow_values.append(flow_case)
        latent_values.append(latent_case)
        editor_weights.append(weights)
        if bool(step.reached_mask.any()):
            raw_flow_values.append(flow_case[step.reached_mask].mean())
            raw_latent_values.append(latent_case[step.reached_mask].mean())
    corrected_editor = mode in {"anatomy_aware"}
    editor_population = (
        int(source.shape[0]) * int(config["rollout"]["k_max"])
        if corrected_editor
        else None
    )
    flow_loss = _weighted_state_mean(
        flow_values, editor_weights, ht_population_size=editor_population
    )
    latent_loss = _weighted_state_mean(
        latent_values, editor_weights, ht_population_size=editor_population
    )
    losses = config["losses"]
    mean, std = (batch["latent_stats_mean"].float(), batch["latent_stats_std"].float())
    translation_metric = canonical_translation_metric(
        config["benefit"].get("translation_metric", "latent_mae")
    )
    with torch.no_grad():
        translation_losses = latent_state_mae(trajectory.states, target, valid)
        anatomy_losses = None
        if mode == "anatomy_aware":
            predicted_raw_states = _inverse_batch(trajectory.states.float(), mean, std)
            pir_reference = build_minimal_pir_reference(
                batch["source_latent_raw"].float(),
                batch["target_latent_raw"].float(),
                batch["valid_mask_latent_hard"],
                batch["anatomy_mask_latent_hard"],
            )
            anatomy_losses = minimal_pir_trajectory(predicted_raw_states, pir_reference)
        translation_scale = float(
            calibration.translation_scale
            if calibration is not None
            else config["benefit"].get("translation_scale", 1.0)
        )
        anatomy_scale = float(
            calibration.anatomy_scale
            if calibration is not None
            else config["benefit"].get("anatomy_scale", 1.0)
        )
        benefit_target = continuation_benefit_targets(
            translation_losses,
            translation_scale=translation_scale,
            anatomy_losses=anatomy_losses,
            anatomy_scale=anatomy_scale if anatomy_losses is not None else None,
            lambda_anatomy=float(config["benefit"]["lambda_anatomy"])
            if anatomy_losses is not None
            else 0.0,
            lambda_compute=float(config["benefit"]["lambda_compute"]),
            observed_mask=trajectory.reached_mask,
        )
    predicted = trajectory.predicted_benefits
    if mode in {"anatomy_aware"}:
        trajectory_weight = compute_suffix_stop_weights(
            mode,
            first_stop_step=final_policy.first_stop_step,
            sentinel_selected=final_policy.sentinel_selected,
            k_max=int(config["rollout"]["k_max"]),
            sentinel_probability=epsilon,
            dtype=predicted.dtype,
        )
        stop_weight = trajectory.reached_mask.to(predicted) * trajectory_weight[:, None]
    else:
        stop_weight = trajectory.reached_mask.to(predicted)
    sentinel_mode = mode in {"anatomy_aware"}
    stop_numerator = smooth_l1_benefit_loss(
        predicted, benefit_target, weight=stop_weight, reduction="sum"
    )
    stop_denominator = (
        int(source.shape[0]) * int(config["rollout"]["k_max"])
        if sentinel_mode
        else stop_weight.sum()
    )
    stop_loss = _distributed_weighted_ratio(stop_numerator, stop_denominator)
    stop_loss_raw = smooth_l1_benefit_loss(
        predicted, benefit_target, weight=trajectory.reached_mask.to(predicted)
    )
    total = (
        float(losses["flow_weight"]) * flow_loss
        + float(losses["latent_l1_weight"]) * latent_loss
        + float(losses["stop_weight"]) * stop_loss
    )
    reached_count = float(trajectory.reached_mask.sum().item())
    logs = {
        "translation_loss_mean": float(translation_losses.mean().item()),
        "translation_is_latent_mae": float(translation_metric == "latent_mae"),
        "vidtok_decoded_crops": 0.0,
        "loss_total": float(total.detach().item()),
        "loss_flow_weighted": float(flow_loss.detach().item()),
        "loss_flow_raw": float(torch.stack(raw_flow_values).mean().detach().item()),
        "loss_latent_weighted": float(latent_loss.detach().item()),
        "loss_latent_raw": float(torch.stack(raw_latent_values).mean().detach().item()),
        "loss_stop": float(stop_loss.detach().item()),
        "loss_stop_raw": float(stop_loss_raw.detach().item()),
        "reached_states": reached_count,
        "mean_reached_states": reached_count / source.shape[0],
        "sentinel_count": float(final_policy.sentinel_selected.sum().item()),
        "mean_editor_effective_weight": float(
            torch.stack(editor_weights, dim=1)
            .sum()
            .div(source.shape[0] * int(config["rollout"]["k_max"]))
            .item()
        ),
        "mean_stop_effective_weight": float(
            stop_weight.sum()
            .div(source.shape[0] * int(config["rollout"]["k_max"]))
            .item()
        ),
        "dense_model_forward_calls": float(trajectory.num_model_evaluations),
        "dense_case_model_evaluations": float(
            trajectory.num_model_evaluations * source.shape[0]
        ),
        "inner_steps": float(config["rollout"]["inner_steps"]),
        "policy_warmup_active": float(
            active_controller.is_policy_warmup(int(global_step))
        ),
    }
    if anatomy_losses is not None:
        logs["anatomy_loss_mean"] = float(anatomy_losses.mean().item())
    with torch.no_grad():
        decision_states = min(
            int(predicted.shape[1]), max(0, int(config["rollout"]["k_max"]) - 1)
        )
        if decision_states > 0:
            supervised = stop_weight[:, :decision_states] > 0
            if bool(supervised.any()):
                paired_predicted = predicted[:, :decision_states][supervised].float()
                paired_realized = benefit_target[:, :decision_states][
                    supervised
                ].float()
                difference = paired_predicted - paired_realized
                logs.update(
                    {
                        "benefit/pred_mean": float(paired_predicted.mean().item()),
                        "benefit/realized_mean": float(paired_realized.mean().item()),
                        "benefit/calibration_mae": float(
                            difference.abs().mean().item()
                        ),
                        "benefit/calibration_rmse": float(
                            difference.square().mean().sqrt().item()
                        ),
                        "benefit/calibration_bias": float(difference.mean().item()),
                        "benefit/decision_count": float(paired_predicted.numel()),
                    }
                )
                centered_predicted = paired_predicted - paired_predicted.mean()
                centered_realized = paired_realized - paired_realized.mean()
                denominator = (
                    centered_predicted.square().sum().sqrt()
                    * centered_realized.square().sum().sqrt()
                )
                if float(denominator.item()) > 0.0:
                    logs["benefit/pred_vs_realized"] = float(
                        (
                            (centered_predicted * centered_realized).sum() / denominator
                        ).item()
                    )
            for state_index in range(decision_states):
                state_supervised = supervised[:, state_index]
                if bool(state_supervised.any()):
                    logs[f"benefit/pred_state_{state_index + 1}"] = float(
                        predicted[state_supervised, state_index].mean().item()
                    )
                    logs[f"benefit/realized_state_{state_index + 1}"] = float(
                        benefit_target[state_supervised, state_index].mean().item()
                    )
        if int(predicted.shape[1]) == int(config["rollout"]["k_max"]):
            terminal_reached = trajectory.reached_mask[:, -1]
            if bool(terminal_reached.any()):
                logs["benefit/terminal_pred_mean"] = float(
                    predicted[terminal_reached, -1].mean().item()
                )
    logs.update(
        {
            f"reach_probability_state_{index + 1}": float(
                trajectory.reached_mask[:, index].float().mean().item()
            )
            for index in range(trajectory.reached_mask.shape[1])
        }
    )
    logs.update(
        {
            f"reach/{index + 1}": float(
                trajectory.reached_mask[:, index].float().mean().item()
            )
            for index in range(trajectory.reached_mask.shape[1])
        }
    )
    return BatchObjective(total, flow_loss, latent_loss, stop_loss, logs)


def _run_output_dir(config: Mapping[str, Any]) -> Path:
    configured = str(config["project"].get("output_dir", "")).strip()
    if configured:
        return Path(configured)
    return (
        Path(config["project"]["output_root"])
        / str(config["project"]["experiment"])
        / str(config["data"]["dataset"])
        / str(config["data"]["pair"])
        / str(config["method"]["mode"])
    )


def _build_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    from streamrefine.data.latent_pair_dataset import load_latent_stats
    from streamrefine.data.anatomy_mask import anatomy_mask_contract_fingerprint
    from streamrefine.training.anatomy import anatomy_metric_contract_fingerprint

    stats_path = Path(config["latent"]["stats_path"])
    stats = load_latent_stats(stats_path)
    return {
        **translation_metric_identity(
            config["benefit"].get("translation_metric", "latent_mae")
        ),
        "latent_statistics_sha256": _sha256_file(stats_path),
        "train_manifest_sha256": _sha256_file(config["data"]["train_manifest"]),
        "val_manifest_sha256": _sha256_file(config["data"]["val_manifest"]),
        "preprocessing_config_sha256": _sha256_file(
            config["data"]["preprocessing_config"]
        ),
        "tokenizer_checkpoint_sha256": stats["tokenizer_checkpoint_sha256"],
        "tokenizer_config_sha256": stats["tokenizer_config_sha256"],
        "generation_contract_sha256": stats["generation_contract_sha256"],
        "cache_schema": config["latent"]["cache_schema"],
        "anatomy_metric": str(config["benefit"]["anatomy_metric"]),
        "anatomy_contract_fingerprint": anatomy_metric_contract_fingerprint(
            config["benefit"]["anatomy_metric"]
        ),
        "anatomy_mask_policy": str(config["data"]["anatomy_mask_policy"]),
        "anatomy_mask_contract_fingerprint": anatomy_mask_contract_fingerprint(
            config["data"]["dataset"], config["data"]["anatomy_mask_policy"]
        ),
    }


def _load_calibration(
    config: Mapping[str, Any], provenance: Mapping[str, Any], init_checkpoint: Any
):
    if automatic.enabled(config):
        return (None, None)
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
    )
    from streamrefine.method.calibration import (
        CalibrationExpectation,
        load_benefit_calibration,
    )
    from streamrefine.training.anatomy import (
        anatomy_metric_contract,
        anatomy_metric_contract_fingerprint,
        canonical_anatomy_metric,
    )

    source, target = pair_modalities(config)
    path = str(config["benefit"].get("calibration_path", "")).strip()
    mode = str(config["method"]["mode"])
    configured_identity = str(
        config["benefit"].get("fixed_horizon_checkpoint_identity", "")
    ).strip()
    configured_trajectory_contract = str(
        config["benefit"].get("fixed_horizon_trajectory_contract_fingerprint", "")
    ).strip()
    calibration = load_benefit_calibration(path) if path else None
    artifact_identity = (
        "" if calibration is None else calibration.fixed_horizon_checkpoint_identity
    )
    initialization_identity = ""
    if init_checkpoint is not None:
        initialization_identity = checkpoint_weight_identity(
            init_checkpoint, use_ema=bool(config["train"].get("init_use_ema", True))
        )
    identity_candidates = [
        value
        for value in (configured_identity, initialization_identity, artifact_identity)
        if value
    ]
    if len(set(identity_candidates)) > 1:
        raise ValueError(
            "Configured, initialization-checkpoint, and calibration-artifact fixed-horizon identities do not agree"
        )
    fixed_identity = identity_candidates[0] if identity_candidates else ""
    if mode == "anatomy_aware" and calibration is None:
        raise ValueError("anatomy_aware mode requires benefit_calibration.json")
    if mode == "anatomy_aware" and (not fixed_identity):
        raise ValueError(
            "anatomy_aware requires a fixed-horizon checkpoint identity in its calibration artifact, config, or explicit --init-checkpoint"
        )
    if calibration is not None and (not configured_trajectory_contract):
        raise ValueError(
            "Using benefit calibration requires benefit.fixed_horizon_trajectory_contract_fingerprint"
        )
    anatomy_metric = canonical_anatomy_metric(config["benefit"]["anatomy_metric"])
    mask_policy = str(config["data"]["anatomy_mask_policy"])
    expectation = CalibrationExpectation(
        **translation_metric_identity(
            config["benefit"].get("translation_metric", "latent_mae")
        ),
        dataset=config["data"]["dataset"],
        source_modality=source,
        target_modality=target,
        split="train",
        fixed_horizon_checkpoint_identity=fixed_identity or None,
        latent_statistics_identity=provenance["latent_statistics_sha256"],
        anatomy_metric=anatomy_metric,
        anatomy_contract=anatomy_metric_contract(anatomy_metric),
        anatomy_contract_fingerprint=anatomy_metric_contract_fingerprint(
            anatomy_metric
        ),
        anatomy_mask_policy=mask_policy if anatomy_metric == "pir" else None,
        anatomy_mask_contract=anatomy_mask_contract(
            config["data"]["dataset"], mask_policy
        )
        if anatomy_metric == "pir"
        else None,
        anatomy_mask_contract_fingerprint=anatomy_mask_contract_fingerprint(
            config["data"]["dataset"], mask_policy
        )
        if anatomy_metric == "pir"
        else None,
        cache_provenance={
            key: provenance[key]
            for key in (
                "tokenizer_checkpoint_sha256",
                "tokenizer_config_sha256",
                "generation_contract_sha256",
                "cache_schema",
                "anatomy_mask_policy",
                "anatomy_mask_contract_fingerprint",
            )
        },
        trajectory_contract_fingerprint=configured_trajectory_contract or None,
        k_max=int(config["rollout"]["k_max"]),
    )
    if calibration is not None:
        calibration.assert_compatible(expectation)
    should_persist_identity = (
        mode == "anatomy_aware" or calibration is not None or bool(configured_identity)
    )
    return (
        calibration,
        fixed_identity if fixed_identity and should_persist_identity else None,
    )


def _distributed_context(
    torch: Any, requested_device: str, *, timeout_minutes: float = 120.0
):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if requested_device != "cuda":
            raise RuntimeError("Distributed StreamRefine currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(minutes=float(timeout_minutes)),
            )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
    return (world_size, rank, local_rank, device)


def run_training(
    config: Mapping[str, Any],
    *,
    explicit_resume: str | Path | None,
    init_checkpoint: str | Path | None,
) -> None:
    import sys
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from tqdm.auto import tqdm
    from streamrefine.data.augmentation import build_training_dataset
    from streamrefine.data.full_volume_dataset import build_full_volume_dataset
    from streamrefine.training.validation import run_full_volume_validation

    def write_console(message: str) -> None:
        tqdm.write(message)
        sys.stdout.flush()

    config = copy.deepcopy(dict(config))
    auto_calibrate = automatic.enabled(config)
    world_size, rank, local_rank, device = _distributed_context(
        torch,
        str(config["runtime"].get("device", "cuda")),
        timeout_minutes=float(
            config["runtime"].get("distributed_timeout_minutes", 120.0)
        ),
    )
    seed_everything(
        int(config["train"]["seed"]) + rank,
        deterministic=bool(config["runtime"].get("deterministic", False)),
    )
    autocast_factory, _ = configure_precision(config["train"]["precision"], device)
    output_dir = _run_output_dir(config)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[StreamRefine][setup] dataset={config['data']['dataset']} pair={config['data']['pair']} mode={config['method']['mode']} device={device} world_size={world_size} precision={config['train']['precision']} k_max={config['rollout']['k_max']} inner_steps={config['rollout']['inner_steps']} decode_batch_size={config['vidtok']['decode_batch_size']} decode_window_dhw={config['vidtok']['decode_window_dhw']}\n[StreamRefine][setup] translation_metric={config['benefit']['translation_metric']}\n[StreamRefine][setup] output_dir={output_dir}\n[StreamRefine][setup] Checking provenance and calibration...",
            flush=True,
        )
    provenance = _build_provenance(config)
    calibration, fixed_horizon_identity = _load_calibration(
        config, provenance, init_checkpoint
    )
    if fixed_horizon_identity is not None:
        config["benefit"]["fixed_horizon_checkpoint_identity"] = fixed_horizon_identity
    if calibration is not None:
        provenance = {
            **provenance,
            "benefit_calibration_fingerprint": calibration.fingerprint,
        }
    compatibility = build_compatibility_contract(
        config, provenance=provenance, world_size=world_size
    )
    if rank == 0:
        print("[StreamRefine][setup] Building train/validation datasets...", flush=True)
    train_dataset = build_training_dataset(config, split="train")
    val_dataset = build_training_dataset(config, split="val")
    deployment_val_dataset = build_full_volume_dataset(config, split="val")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["train"]["seed"]),
        drop_last=True,
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    )
    common_loader = {
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": bool(config["data"].get("pin_memory", True)),
        "persistent_workers": bool(config["data"].get("persistent_workers", True))
        and int(config["data"].get("num_workers", 4)) > 0,
        "collate_fn": collate_streamrefine_batch,
    }
    if common_loader["num_workers"] > 0:
        common_loader["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size_per_gpu"]),
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **common_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common_loader,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader is empty")
    dataset_summary = {
        "train_records": len(train_dataset),
        "val_crop_records": len(val_dataset),
        "val_full_volume_records": len(deployment_val_dataset),
        "train_microbatches_per_rank_per_epoch": len(train_loader),
        "train_sampled_records_per_rank": len(train_sampler),
        "train_consumed_crops_per_rank_per_epoch": len(train_loader)
        * int(config["train"]["batch_size_per_gpu"]),
        "val_crop_batches_per_rank": len(val_loader),
        "effective_global_batch_size": int(config["train"]["batch_size_per_gpu"])
        * int(config["train"]["grad_accumulation"])
        * world_size,
        "train_persistent_cache": str(
            getattr(getattr(train_dataset, "dataset", None), "cache_dir", "")
        ),
        "val_crop_persistent_cache": str(
            getattr(getattr(val_dataset, "dataset", None), "cache_dir", "")
        ),
        "val_full_persistent_cache": str(
            getattr(getattr(deployment_val_dataset, "dataset", None), "cache_dir", "")
        ),
        "num_workers": common_loader["num_workers"],
        "prefetch_factor": common_loader.get("prefetch_factor", None),
    }
    if rank == 0:
        print(
            "[StreamRefine][data] "
            + " ".join((f"{key}={value}" for key, value in dataset_summary.items())),
            flush=True,
        )
        print(
            "[StreamRefine][setup] Building editor, EMA and frozen VidTok decoder...",
            flush=True,
        )
    model = build_model(config).to(device)
    _broadcast_model_parameters(model, world_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
        betas=(0.9, 0.95),
    )
    max_steps, warmup = (
        int(config["train"]["max_steps"]),
        int(config["train"]["warmup_steps"]),
    )

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ema = (
        ExponentialMovingAverage(_unwrap(model), float(config["train"]["ema_decay"]))
        if bool(config["train"].get("ema", True))
        else None
    )
    decoder = build_vidtok_decoder(config, device)
    action = resolve_checkpoint_action(
        output_dir=output_dir,
        expected_compatibility=compatibility,
        explicit_resume=explicit_resume,
        init_checkpoint=init_checkpoint,
        auto_resume=bool(config["train"].get("auto_resume", True)),
    )
    if rank == 0:
        atomic_write_json(output_dir / "resolved_config.json", config)
    global_step, epoch, best_validation = (0, 0, None)
    best_validation_metric: str | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_selection_contract: dict[str, Any] | None = None
    batch_cursor = 0
    resume_rng_state = None
    restored_micro_since_update = 0
    if action.kind == "resume":
        restored = restore_training_checkpoint(
            action.path,
            model=_unwrap(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            ema=ema,
            expected_compatibility=compatibility,
        )
        global_step, epoch, best_validation = (
            int(restored["step"]),
            int(restored["epoch"]),
            restored["best_validation"],
        )
        restored_metric = restored.get("best_validation_metric")
        best_validation_metric = (
            None if restored_metric is None else str(restored_metric)
        )
        restored_metrics = restored.get("best_validation_metrics")
        if restored_metrics is not None and (not isinstance(restored_metrics, Mapping)):
            raise ValueError(
                "Checkpoint best_validation_metrics must be a mapping or null"
            )
        best_validation_metrics = (
            None if restored_metrics is None else copy.deepcopy(dict(restored_metrics))
        )
        restored_selection = restored.get("best_selection_contract")
        if restored_selection is not None and (
            not isinstance(restored_selection, Mapping)
        ):
            raise ValueError(
                "Checkpoint best_selection_contract must be a mapping or null"
            )
        best_selection_contract = (
            None
            if restored_selection is None
            else copy.deepcopy(dict(restored_selection))
        )
        rank_states = restored.get("rng_state_by_rank")
        if not isinstance(rank_states, list) or len(rank_states) != world_size:
            raise ValueError(
                "Resume checkpoint RNG rank roster differs from world size"
            )
        resume_rng_state = rank_states[rank]
        restore_rng_state(resume_rng_state)
        policy_state = restored.get("policy_state")
        if not isinstance(policy_state, Mapping):
            raise ValueError("Resume checkpoint lacks policy_state")
        batch_cursor = int(policy_state.get("microbatch_in_epoch", -1))
        if batch_cursor < 0:
            raise ValueError(
                "Resume checkpoint lacks a valid microbatch_in_epoch cursor"
            )
        if batch_cursor > len(train_loader):
            raise ValueError(
                "Resume microbatch cursor exceeds the active DataLoader length"
            )
        restored_micro_since_update = int(policy_state.get("micro_since_update", -1))
        if restored_micro_since_update != 0:
            raise ValueError(
                "StreamRefine checkpoints must be saved at optimizer boundaries"
            )
    elif action.kind == "initialize":
        initialize_model_checkpoint(
            action.path,
            model=_unwrap(model),
            expected_compatibility=compatibility,
            use_ema=bool(config["train"].get("init_use_ema", True)),
        )
        if ema is not None:
            ema = ExponentialMovingAverage(
                _unwrap(model), float(config["train"]["ema_decay"])
            )
    if auto_calibrate and action.kind == "resume":
        calibration = automatic.restore_state(
            config,
            provenance,
            restored["policy_state"].get("auto_calibration"),
            global_step,
        )
        if calibration is not None:
            automatic.rank_zero_call(
                lambda: str(automatic.ensure_artifact_file(calibration, output_dir)),
                rank=rank,
                world_size=world_size,
            )
    controller = _controller(config)
    selection_metric = str(
        config["validation"].get("selection_metric", "adaptive_utility")
    ).strip()
    selection_direction = (
        str(config["validation"].get("selection_direction", "min")).strip().lower()
    )
    validation_use_ema = bool(config["inference"].get("use_ema", True))
    validation_weight_source = "ema" if validation_use_ema else "raw"
    if validation_use_ema and ema is None:
        raise ValueError(
            "Deployment validation requests EMA via inference.use_ema=true, but train.ema=false"
        )
    if best_validation is not None:
        if best_validation_metric is None:
            raise ValueError(
                "Resumed checkpoint has best_validation but no metric identity"
            )
        if best_validation_metric != selection_metric:
            raise ValueError(
                f"Resumed checkpoint validation metric differs from the active selection contract: {best_validation_metric!r} != {selection_metric!r}"
            )
        if not isinstance(best_selection_contract, Mapping):
            raise ValueError(
                "Resumed checkpoint lacks the best-selection deployment contract"
            )
        if not isinstance(best_validation_metrics, Mapping):
            raise ValueError(
                "Resumed checkpoint lacks the best validation metric summary"
            )
        expected_selection = {
            "metric": selection_metric,
            "direction": selection_direction,
            "weight_source": validation_weight_source,
            "scope": "full_volume_synchronized_sliding",
            "method_mode": str(config["method"]["mode"]),
            "stop_threshold": float(config["method"]["stop_threshold"]),
            "k_max": int(config["rollout"]["k_max"]),
            "anatomy_metric": str(config["benefit"]["anatomy_metric"]),
            "anatomy_contract_fingerprint": provenance["anatomy_contract_fingerprint"],
            "anatomy_mask_policy": provenance["anatomy_mask_policy"],
            "anatomy_mask_contract_fingerprint": provenance[
                "anatomy_mask_contract_fingerprint"
            ],
        }
        mismatches = [
            key
            for key, expected in expected_selection.items()
            if best_selection_contract.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "Resumed best-selection contract differs in: " + ", ".join(mismatches)
            )
        if not str(best_selection_contract.get("roster_fingerprint", "")).strip():
            raise ValueError(
                "Resumed best-selection contract lacks a validation roster fingerprint"
            )
        selected_key = (
            selection_metric
            if selection_metric.startswith("val/")
            else f"val/{selection_metric}"
        )
        if selected_key not in best_validation_metrics:
            raise ValueError(f"Resumed best validation summary lacks {selected_key}")
        saved_selected_value = float(best_validation_metrics[selected_key])
        if (
            not math.isfinite(saved_selected_value)
            or not math.isfinite(float(best_validation))
            or saved_selected_value != float(best_validation)
        ):
            raise ValueError(
                "Resumed best_validation does not equal its selected metric summary"
            )
    accumulation = int(config["train"]["grad_accumulation"])
    if accumulation <= 0:
        raise ValueError("train.grad_accumulation must be positive")
    optimizer.zero_grad(set_to_none=True)
    running = RunningMeans()
    micro_since_update = restored_micro_since_update
    start_step = global_step
    start_time = time.perf_counter()
    last_log_time, last_log_step = (start_time, global_step)
    data_wait_since_update = interval_data_wait = 0.0
    logger = diagnostics_logger = None
    if rank == 0:
        metadata = {"config_fingerprint": config_fingerprint(config), "rank": rank}
        logger = StructuredTrainingLog(
            output_dir / "logs" / "train_log.json",
            output_dir=output_dir,
            run_metadata=metadata,
        )
        diagnostics_logger = StructuredTrainingLog(
            output_dir / "logs" / "diagnostics.json",
            output_dir=output_dir,
            run_metadata=metadata,
        )
        start_fields = {
            "step": global_step,
            "epoch": epoch,
            "max_steps": max_steps,
            "checkpoint_action": action.kind,
            "checkpoint_path": None if action.path is None else str(action.path),
            "dataset": dict(config["data"]),
            "dataset_counts": dataset_summary,
            "model": dict(config["model"]),
            "latent": dict(config["latent"]),
            "train": dict(config["train"]),
            "rollout": dict(config["rollout"]),
            "method": dict(config["method"]),
            "vidtok": dict(config["vidtok"]),
            "validation_weight_source": validation_weight_source,
            "training_phase": automatic.phase(config, global_step, calibration)
            if auto_calibrate
            else config["method"]["mode"],
            "selection_metric": selection_metric,
            "selection_direction": selection_direction,
            "world_size": world_size,
            "device": str(device),
            "trainable_parameters": sum(
                (p.numel() for p in model.parameters() if p.requires_grad)
            ),
            "diagnostics": dict(config["diagnostics"]),
            "metric_scope": "rank_mean_of_microbatch_means_since_previous_log",
            "data_wait_scope": "rank0_exposed_dataloader_wait_including_iterator_startup",
            "gpu_memory_scope": "rank0_allocator_process_lifetime_peaks",
        }
        for sink in (logger, diagnostics_logger):
            sink.log("train_start", **start_fields)
        print(
            f"[StreamRefine][start] action={action.kind} checkpoint={action.path} step={global_step}/{max_steps} epoch={epoch} parameters={start_fields['trainable_parameters']:,} accumulation={accumulation} log_every={config['train']['log_every']} optimizer steps validation_weights={validation_weight_source}\n[StreamRefine][logs] {logger.path}\n[StreamRefine][logs] {diagnostics_logger.path}\n[StreamRefine][train] Waiting for first optimizer update (data + forward/backward)...",
            flush=True,
        )

    def save_checkpoint(name: str) -> None:
        local_rng = capture_rng_state()
        if world_size > 1:
            rank_rng: list[Any] = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(rank_rng, local_rng)
        else:
            rank_rng = [local_rng]

        def write_on_rank_zero():
            payload = checkpoint_payload(
                model=_unwrap(model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                ema=ema,
                step=global_step,
                epoch=epoch,
                best_validation=best_validation,
                config=config,
                compatibility=compatibility,
                policy_state={
                    "global_step": global_step,
                    "policy_warmup_steps": controller.config.policy_warmup_steps,
                    "warmup_complete": global_step
                    >= controller.config.policy_warmup_steps,
                    "microbatch_in_epoch": batch_cursor,
                    "micro_since_update": micro_since_update,
                    **(
                        {
                            "auto_calibration": automatic.state_dict(
                                config, global_step, calibration
                            )
                        }
                        if auto_calibrate
                        else {}
                    ),
                },
                best_validation_metric=best_validation_metric,
                best_validation_metrics=best_validation_metrics,
                best_selection_contract=best_selection_contract,
            )
            payload["rng_state_by_rank"] = rank_rng
            atomic_torch_save(payload, output_dir / name)
            assert logger is not None
            fields = {
                "step": global_step,
                "epoch": epoch,
                "path": str(output_dir / name),
            }
            logger.log("checkpoint", **fields)
            write_console(format_progress("checkpoint", fields))

        if auto_calibrate:
            automatic.rank_zero_call(
                write_on_rank_zero, rank=rank, world_size=world_size
            )
        elif rank == 0:
            write_on_rank_zero()

    def maybe_calibrate() -> None:
        nonlocal calibration
        if not auto_calibrate or calibration is not None:
            return
        if global_step < int(config["method"]["policy_warmup_steps"]):
            return
        if micro_since_update:
            raise RuntimeError(
                "Automatic calibration must run at an optimizer boundary"
            )
        from streamrefine.training.calibration_progress import CalibrationProgress

        started = time.perf_counter()
        metric_path = (
            "latent_only"
            if automatic.uses_latent_metrics_only(config)
            else "image_and_latent"
        )
        if rank == 0:
            logger.log(
                "calibration_start",
                step=global_step,
                split="train",
                metric_path=metric_path,
                max_cases=config["benefit"]["auto_calibration"].get("max_cases", 32),
            )
            write_console(
                f"[StreamRefine][calibration_start] step={global_step} split=train weights={validation_weight_source} metric_path={metric_path}"
            )
        if progress is not None:
            progress.set_description(f"calibrating step={global_step}")
        snapshot = output_dir / "calibration" / "warmup.pt"
        try:
            with CalibrationProgress(
                output_dir=output_dir,
                step=global_step,
                rank=rank,
                world_size=world_size,
                device=device,
                training_bar=progress is not None,
            ) as calibration_progress:
                with calibration_progress.stage("save_latest_before"):
                    save_checkpoint("latest.pt")
                with calibration_progress.stage("save_warmup_snapshot"):
                    save_checkpoint("calibration/warmup.pt")
                    if world_size > 1:
                        torch.distributed.barrier()
                calibration_rng = capture_rng_state()
                was_training = bool(model.training)
                weight_context = (
                    ema.average_parameters(_unwrap(model))
                    if validation_use_ema
                    else contextlib.nullcontext()
                )
                try:
                    seed_everything(
                        int(config["inference"].get("seed", 3407)) + rank,
                        deterministic=False,
                    )
                    with weight_context:
                        model.eval()
                        calibration = automatic.run_automatic_calibration(
                            model=_unwrap(model),
                            decoder=decoder,
                            config=config,
                            provenance=provenance,
                            snapshot_path=snapshot,
                            output_dir=output_dir,
                            device=device,
                            autocast_factory=autocast_factory,
                            rank=rank,
                            world_size=world_size,
                            progress=calibration_progress,
                        )
                finally:
                    model.train(was_training)
                    restore_rng_state(calibration_rng)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                with calibration_progress.stage("save_latest_calibrated"):
                    save_checkpoint("latest.pt")
        finally:
            if progress is not None:
                progress.set_description(f"epoch={epoch}")
        if rank == 0:
            fields = {
                "step": global_step,
                "phase": automatic.phase(config, global_step, calibration),
                "translation_scale": calibration.translation_scale,
                "anatomy_scale": calibration.anatomy_scale,
                "sample_count": calibration.sample_count,
                "artifact_fingerprint": calibration.fingerprint,
                "artifact_path": str(
                    output_dir / "calibration" / "benefit_calibration.json"
                ),
                "adaptive_start_step": automatic.adaptive_start_step(config),
                "metric_path": metric_path,
                "calibration_elapsed_sec": time.perf_counter() - started,
                "progress_path": str(calibration_progress.path),
            }
            logger.log("calibration_complete", **fields)
            write_console(
                f"[StreamRefine][calibration_complete] step={global_step} s_L={calibration.translation_scale:.6g} s_A={calibration.anatomy_scale:.6g} cases={calibration.sample_count} elapsed={fields['calibration_elapsed_sec']:.1f}s adaptive_start={automatic.adaptive_start_step(config)}; training resumes"
            )

    def validate() -> dict[str, Any]:
        """Run optimizer diagnostics and the actual deployment policy on one EMA snapshot."""
        validation_rng = capture_rng_state()
        was_training = bool(model.training)
        weight_context = (
            ema.average_parameters(_unwrap(model))
            if validation_use_ema
            else contextlib.nullcontext(_unwrap(model))
        )
        try:
            seed_everything(
                int(config["inference"].get("seed", 3407)) + int(rank),
                deterministic=False,
            )
            with weight_context:
                model.eval()
                totals = torch.zeros(2, device=device, dtype=torch.float64)
                fixed_controller = _controller(
                    {**config, "method": {**config["method"], "mode": "fixed_k"}}
                )
                max_batches = int(config["validation"].get("max_batches", 16))
                with torch.no_grad():
                    for index, raw_batch in enumerate(val_loader):
                        if index >= max_batches:
                            break
                        batch = _move_batch(raw_batch, device)
                        with autocast_factory():
                            objective = compute_batch_objective(
                                model=model,
                                decoder=decoder,
                                batch=batch,
                                config=config,
                                global_step=0,
                                calibration=calibration,
                                controller=fixed_controller,
                            )
                        totals[0] += objective.total.detach().double()
                        totals[1] += 1
                        del objective, batch, raw_batch
                if world_size > 1:
                    torch.distributed.all_reduce(totals)
                if float(totals[1].item()) <= 0.0:
                    raise RuntimeError(
                        "Full-trajectory crop validation produced no samples"
                    )
                validation_config = config
                provisional = (
                    auto_calibrate
                    and automatic.phase(config, global_step, calibration) != "adaptive"
                )
                if provisional:
                    validation_config = copy.deepcopy(config)
                    validation_config["method"]["mode"] = "fixed_k"
                    validation_config["validation"]["selection_metric"] = (
                        "full_trajectory_utility"
                    )
                    validation_config["validation"]["selection_direction"] = "min"
                summary = run_full_volume_validation(
                    model=_unwrap(model),
                    decoder=decoder,
                    dataset=deployment_val_dataset,
                    config=validation_config,
                    device=device,
                    autocast_factory=autocast_factory,
                    calibration=calibration,
                    rank=rank,
                    world_size=world_size,
                )
                summary = dict(summary)
                summary["val/full_trajectory_objective"] = float(
                    (totals[0] / totals[1]).item()
                )
                summary["val/weight_source"] = validation_weight_source
                summary["val/policy_semantics"] = (
                    "warmup_full_horizon"
                    if provisional
                    else "deterministic_deployment_threshold_replay"
                )
                if auto_calibrate:
                    summary["val/training_phase"] = automatic.phase(
                        config, global_step, calibration
                    )
                    summary["val/calibration_frozen"] = calibration is not None
                return summary
        finally:
            restore_rng_state(validation_rng)
            model.train(was_training)

    def selected_validation_value(summary: Mapping[str, Any]) -> float:
        key = (
            selection_metric
            if selection_metric.startswith("val/")
            else f"val/{selection_metric}"
        )
        if key not in summary:
            raise KeyError(
                f"Configured validation selection metric {selection_metric!r} was not reported; available keys={sorted(summary)}"
            )
        value = float(summary[key])
        if not math.isfinite(value):
            raise ValueError(
                f"Validation selection metric {key} is not finite: {value}"
            )
        return value

    model.train()
    resume_skip = batch_cursor
    resume_rng_pending = resume_rng_state is not None
    progress = (
        tqdm(
            total=max_steps,
            initial=start_step,
            desc=f"epoch={epoch}",
            unit="step",
            dynamic_ncols=True,
        )
        if rank == 0
        else None
    )
    try:
        maybe_calibrate()
        while global_step < max_steps:
            if progress is not None:
                progress.set_description(f"epoch={epoch}")
            train_sampler.set_epoch(epoch)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            completed_epoch = True
            for _micro_index, raw_batch, data_wait_sec in timed_training_batches(
                train_loader
            ):
                if resume_skip and _micro_index < resume_skip:
                    continue
                data_wait_since_update += data_wait_sec
                interval_data_wait += data_wait_sec
                if resume_rng_pending:
                    restore_rng_state(resume_rng_state)
                    resume_rng_pending = False
                resume_skip = 0
                batch_cursor = _micro_index + 1
                batch = _move_batch(raw_batch, device)
                micro_since_update += 1
                sync_now = micro_since_update >= accumulation
                with autocast_factory():
                    objective = compute_batch_objective(
                        model=model,
                        decoder=decoder,
                        batch=batch,
                        config=config,
                        global_step=global_step,
                        calibration=calibration,
                        controller=controller,
                    )
                (objective.total / accumulation).backward()
                running.update(objective.logs)
                if not sync_now:
                    continue
                _synchronize_gradients(model, world_size)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["train"]["gradient_clip_norm"])
                )
                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_since_update = 0
                scheduler.step()
                global_step += 1
                if ema is not None:
                    ema.update(_unwrap(model))
                gradient_norm_value = (
                    float(gradient_norm.detach().item()) if rank == 0 else None
                )
                data_wait_last_step = data_wait_since_update
                data_wait_since_update = 0.0
                if progress is not None:
                    postfix = {
                        label: f"{objective.logs[key]:.3f}"
                        for label, key in (
                            ("loss", "loss_total"),
                            ("flow", "loss_flow_weighted"),
                            ("latent", "loss_latent_weighted"),
                            ("stop", "loss_stop"),
                            ("reach", "mean_reached_states"),
                        )
                        if key in objective.logs
                    }
                    postfix["lr"] = f"{learning_rate_used:.3e}"
                    postfix["grad"] = f"{gradient_norm_value:.3e}"
                    postfix["data_s"] = f"{data_wait_last_step:.3f}"
                    progress.set_postfix(postfix, refresh=False)
                    progress.update(1)
                if training_log_due(
                    global_step,
                    start_step=start_step,
                    max_steps=max_steps,
                    every=int(config["train"]["log_every"]),
                ):
                    log_values = running.compute()
                    if world_size > 1 and log_values:
                        log_keys = sorted(log_values)
                        log_tensor = torch.as_tensor(
                            [log_values[key] for key in log_keys],
                            device=device,
                            dtype=torch.float64,
                        )
                        torch.distributed.all_reduce(
                            log_tensor, op=torch.distributed.ReduceOp.SUM
                        )
                        log_tensor.div_(float(world_size))
                        log_values = {
                            key: float(log_tensor[index].item())
                            for index, key in enumerate(log_keys)
                        }
                    if rank == 0:
                        assert logger is not None
                        now = time.perf_counter()
                        interval_steps = global_step - last_log_step
                        wall_sec_per_step = (now - last_log_time) / interval_steps
                        fields = {
                            **log_values,
                            "step": global_step,
                            "epoch": epoch,
                            "max_steps": max_steps,
                            "training_phase": automatic.phase(
                                config, global_step, calibration
                            )
                            if auto_calibrate
                            else config["method"]["mode"],
                            "microbatch_in_epoch": batch_cursor,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "learning_rate_used": learning_rate_used,
                            "gradient_norm_last_step": gradient_norm_value,
                            "data_wait_sec_last_step": data_wait_last_step,
                            "data_wait_sec_per_step": interval_data_wait
                            / interval_steps,
                            "data_wait_fraction": interval_data_wait
                            / max(now - last_log_time, 1e-12),
                            "elapsed_sec": now - start_time,
                            "interval_optimizer_steps": interval_steps,
                            "wall_sec_per_step": wall_sec_per_step,
                            "eta_sec": wall_sec_per_step
                            * max(0, max_steps - global_step),
                            **gpu_memory_snapshot(torch, device),
                        }
                        logger.log("train", **fields)
                        write_console(format_progress("train", fields))
                        last_log_time, last_log_step = (now, global_step)
                    running.reset()
                    interval_data_wait = 0.0
                calibration_due = (
                    auto_calibrate
                    and calibration is None
                    and (global_step >= int(config["method"]["policy_warmup_steps"]))
                )
                validation_due = (
                    global_step % int(config["train"]["validate_every"]) == 0
                )
                if calibration_due or validation_due:
                    del objective, batch, raw_batch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if calibration_due:
                    maybe_calibrate()
                if auto_calibrate and global_step == automatic.adaptive_start_step(
                    config
                ):
                    if rank == 0:
                        logger.log(
                            "phase_transition", step=global_step, phase="adaptive"
                        )
                        write_console(
                            f"[StreamRefine][adaptive_start] step={global_step} frozen calibration active"
                        )
                    save_checkpoint("latest.pt")
                if validation_due:
                    validation_start = time.perf_counter()
                    if rank == 0:
                        assert logger is not None
                        logger.log(
                            "validation_start",
                            step=global_step,
                            weight_source=validation_weight_source,
                        )
                        write_console(
                            f"[StreamRefine][validation_start] step={global_step} weights={validation_weight_source} full-trajectory + adaptive policy..."
                        )
                    validation_metrics = validate()
                    selection_ready = (
                        not auto_calibrate
                        or automatic.phase(config, global_step, calibration)
                        == "adaptive"
                    )
                    current_selection_metric = (
                        selection_metric
                        if selection_ready
                        else "full_trajectory_objective"
                    )
                    validation = (
                        selected_validation_value(validation_metrics)
                        if selection_ready
                        else float(validation_metrics["val/full_trajectory_objective"])
                    )
                    validation_metrics["val/selection_metric"] = (
                        current_selection_metric
                        if current_selection_metric.startswith("val/")
                        else f"val/{current_selection_metric}"
                    )
                    validation_metrics["val/selection_eligible"] = selection_ready
                    current_selection_direction = (
                        selection_direction if selection_ready else "min"
                    )
                    validation_metrics["val/selection_direction"] = (
                        current_selection_direction
                    )
                    validation_metrics["val/selection_value"] = validation
                    improved = selection_ready and (
                        best_validation is None
                        or (
                            validation < float(best_validation)
                            if selection_direction == "min"
                            else validation > float(best_validation)
                        )
                    )
                    if improved:
                        best_validation = validation
                        best_validation_metric = selection_metric
                        best_validation_metrics = copy.deepcopy(validation_metrics)
                        best_selection_contract = {
                            "metric": selection_metric,
                            "direction": selection_direction,
                            "weight_source": validation_weight_source,
                            "scope": "full_volume_synchronized_sliding",
                            "method_mode": str(config["method"]["mode"]),
                            "stop_threshold": float(config["method"]["stop_threshold"]),
                            "k_max": int(config["rollout"]["k_max"]),
                            "anatomy_metric": str(config["benefit"]["anatomy_metric"]),
                            "anatomy_contract_fingerprint": provenance[
                                "anatomy_contract_fingerprint"
                            ],
                            "anatomy_mask_policy": provenance["anatomy_mask_policy"],
                            "anatomy_mask_contract_fingerprint": provenance[
                                "anatomy_mask_contract_fingerprint"
                            ],
                            "roster_fingerprint": validation_metrics.get(
                                "val/roster_fingerprint"
                            ),
                        }
                    if rank == 0:
                        assert logger is not None
                        fields = {
                            **validation_metrics,
                            "step": global_step,
                            "epoch": epoch,
                            "selection_metric": current_selection_metric,
                            "selection_direction": current_selection_direction,
                            "selection_value": validation,
                            "best_validation": best_validation,
                            "improved": improved,
                            "validation_elapsed_sec": time.perf_counter()
                            - validation_start,
                        }
                        logger.log("validation", **fields)
                        write_console(format_progress("validation", fields))
                    if improved:
                        save_checkpoint("best.pt")
                if global_step % int(config["train"]["checkpoint_every"]) == 0:
                    save_checkpoint(f"step_{global_step:08d}.pt")
                    save_checkpoint("latest.pt")
                if global_step >= max_steps:
                    completed_epoch = False
                    break
            if resume_rng_pending:
                restore_rng_state(resume_rng_state)
                resume_rng_pending = False
            resume_skip = 0
            if completed_epoch:
                epoch += 1
                batch_cursor = 0
    finally:
        if progress is not None:
            progress.close()
    save_checkpoint("last.pt")
    if rank == 0:
        assert logger is not None and diagnostics_logger is not None
        fields = {
            "step": global_step,
            "epoch": epoch,
            "best_validation": best_validation,
            "best_validation_metric": best_validation_metric,
            "best_selection_contract": best_selection_contract,
            "elapsed_sec": time.perf_counter() - start_time,
        }
        for sink in (logger, diagnostics_logger):
            sink.log("complete", **fields)
        write_console(format_progress("complete", fields))
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
