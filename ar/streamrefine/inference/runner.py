"""End-to-end synchronized full-volume inference runner."""

from __future__ import annotations
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping
from streamrefine.config import canonical_config_json, pair_modalities
from streamrefine.runtime import atomic_write_json, configure_precision, seed_everything
from streamrefine.training.translation import (
    latent_state_mae,
    translation_metric_identity,
)
from .artifacts import InferenceArtifactWriter
from .metrics import compute_volume_metrics, crop_to_original_hwd
from .sliding_window import synchronized_sliding_refinement, tiled_decode_raw_latent


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Inference checkpoint does not exist: {checkpoint_path}"
        )
    try:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(checkpoint_path, map_location="cpu")
    if (
        not isinstance(value, Mapping)
        or value.get("checkpoint_version") != "streamrefine_checkpoint_v1"
    ):
        raise ValueError(f"Unsupported StreamRefine checkpoint: {checkpoint_path}")
    return dict(value)


def _active_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    from streamrefine.data.anatomy_mask import anatomy_mask_contract_fingerprint
    from streamrefine.data.latent_pair_dataset import load_latent_stats
    from streamrefine.training.anatomy import anatomy_metric_contract_fingerprint

    path = Path(config["latent"]["stats_path"])
    stats = load_latent_stats(path)
    return {
        **translation_metric_identity(
            config["benefit"].get("translation_metric", "latent_mae")
        ),
        "latent_statistics_sha256": _sha256_file(path),
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


def _assert_inference_compatible(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    contract = checkpoint.get("compatibility")
    if not isinstance(contract, Mapping):
        raise ValueError("Checkpoint lacks a compatibility contract")
    expected = {
        "method_mode": config["method"]["mode"],
        "model_geometry": dict(config["model"]),
        "dataset": config["data"]["dataset"],
        "pair": config["data"]["pair"],
        "cache_schema": config["latent"]["cache_schema"],
    }
    mismatches = [
        name for name, value in expected.items() if contract.get(name) != value
    ]
    saved_config = checkpoint.get("resolved_config", {})
    serialized_config = contract.get("resolved_config_json")
    if serialized_config is not None:
        try:
            saved_config = json.loads(serialized_config)
        except (TypeError, ValueError) as exc:
            raise ValueError("Checkpoint has invalid resolved_config_json") from exc
    if not isinstance(saved_config, Mapping) or not isinstance(
        saved_config.get("rollout", {}), Mapping
    ):
        raise ValueError("Checkpoint has invalid rollout configuration")
    saved_start = saved_config.get("rollout", {}).get("refinement_start", "noise")
    if saved_start != "previous_state":
        raise ValueError("Checkpoint has invalid rollout.refinement_start")
    if saved_start != config["rollout"].get("refinement_start", "previous_state"):
        mismatches.append("rollout.refinement_start")
    checkpoint_provenance = contract.get("provenance", {})
    from streamrefine.training.auto_calibration import checkpoint_calibration

    embedded_calibration = checkpoint_calibration(checkpoint)
    for name, value in provenance.items():
        actual = checkpoint_provenance.get(name)
        if (
            name == "benefit_calibration_fingerprint"
            and embedded_calibration is not None
        ):
            actual = embedded_calibration.fingerprint
        if actual != value:
            mismatches.append(f"provenance.{name}")
    if mismatches:
        raise ValueError(
            "Inference checkpoint/config incompatibility: " + ", ".join(mismatches)
        )


def _load_model_state(
    model: Any, checkpoint: Mapping[str, Any], *, use_ema: bool
) -> None:
    state = checkpoint["model"]
    if use_ema:
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping):
            raise ValueError("inference.use_ema=true but checkpoint contains no EMA")
        shadow = ema.get("shadow")
        nonfloating = ema.get("nonfloating")
        if not isinstance(shadow, Mapping) or not isinstance(nonfloating, Mapping):
            raise ValueError("Checkpoint EMA payload is malformed")
        state = {**dict(nonfloating), **dict(shadow)}
    model.load_state_dict(state, strict=True)


def _inference_output_dir(config: Mapping[str, Any]) -> Path:
    explicit = str(config["inference"].get("output_dir", "")).strip()
    if explicit:
        return Path(explicit)
    return (
        Path(config["project"]["output_root"])
        / str(config["project"]["experiment"])
        / str(config["data"]["dataset"])
        / str(config["data"]["pair"])
        / str(config["method"]["mode"])
        / "inference"
    )


def _contract_fingerprint(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    provenance: Mapping[str, Any],
) -> str:
    payload = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "dataset": config["data"]["dataset"],
        "pair": config["data"]["pair"],
        "method": config["method"],
        "benefit": config["benefit"],
        "rollout": config["rollout"],
        "inference": config["inference"],
        "vidtok": config["vidtok"],
        "metrics": config["metrics"],
        "artifacts": config["artifacts"],
        "provenance": provenance,
    }
    return hashlib.sha256(canonical_config_json(payload).encode("utf-8")).hexdigest()


class ModelWindowPredictor:
    """Rebuild window histories from committed global states at each refinement."""

    def __init__(self, model, config, autocast_factory):
        from streamrefine.generation import (
            RectifiedFlowScheduler,
            WholeVolumeSelfForcing,
        )

        self.adapter = WholeVolumeSelfForcing(
            model,
            RectifiedFlowScheduler(int(config["rollout"]["inner_steps"])),
            inner_steps=int(config["rollout"]["inner_steps"]),
            k_max=int(config["rollout"]["k_max"]),
            refinement_start="previous_state",
            sync_flow_time_across_ranks=False,
        )
        self.autocast_factory = autocast_factory

    def reset_case(self):
        pass

    def __call__(
        self,
        *,
        source_window,
        history_windows,
        noise_window,
        valid_mask,
        global_coordinates,
        refinement_step,
    ):
        import torch

        history = tuple((state.unsqueeze(0) for state in history_windows))
        source, mask, coordinates = (
            source_window.unsqueeze(0),
            valid_mask.unsqueeze(0),
            global_coordinates.unsqueeze(0),
        )
        with torch.inference_mode(), self.autocast_factory():
            source_cache = self.adapter.prepare_source_cache(
                source, valid_mask=mask, global_coordinates=coordinates
            )
            rollout = self.adapter.rollout_state(
                source,
                refinement_step=refinement_step,
                history_states=history,
                noise=noise_window.unsqueeze(0),
                valid_mask=mask,
                global_coordinates=coordinates,
                source_cache=source_cache,
                target_history_cache=None,
            )
        return {
            "latent": rollout.state[0],
            "model_evaluations": rollout.num_model_evaluations,
        }

    def score_committed(
        self,
        *,
        source_window,
        history_windows,
        committed_window,
        valid_mask,
        global_coordinates,
        refinement_step,
    ):
        import torch

        history = tuple((state.unsqueeze(0) for state in history_windows))
        source, mask, coordinates = (
            source_window.unsqueeze(0),
            valid_mask.unsqueeze(0),
            global_coordinates.unsqueeze(0),
        )
        with torch.inference_mode(), self.autocast_factory():
            source_cache = self.adapter.prepare_source_cache(
                source, valid_mask=mask, global_coordinates=coordinates
            )
            target_history_cache = self.adapter.prepare_target_history_cache(
                source,
                history,
                valid_mask=mask,
                global_coordinates=coordinates,
                source_cache=source_cache,
            )
            benefit = self.adapter.benefit_query(
                committed_window.unsqueeze(0),
                source,
                refinement_step=refinement_step,
                history_states=history,
                valid_mask=mask,
                global_coordinates=coordinates,
                source_cache=source_cache,
                target_history_cache=target_history_cache,
                cache_target=False,
            )
        return {
            "benefit_score": benefit[0],
            "model_evaluations": (
                len(history) if target_history_cache is not None else 0
            )
            + 1,
        }


def _stop_callable(config: Mapping[str, Any]):
    from streamrefine.validation_policy import deployment_should_stop

    mode = str(config["method"]["mode"])
    threshold = float(config["method"]["stop_threshold"])

    def stop(score: float, step: int, k_max: int) -> bool:
        return deployment_should_stop(
            score, step=step, k_max=k_max, mode=mode, stop_threshold=threshold
        )

    return stop


def _case_id_without_loading(dataset: Any, index: int) -> str | None:
    persistent = getattr(dataset, "dataset", None)
    rows = getattr(persistent, "data", None)
    if (
        isinstance(rows, list)
        and index < len(rows)
        and isinstance(rows[index], Mapping)
    ):
        return str(rows[index].get("case_id", "")) or None
    return None


def _build_lpips_model(config: Mapping[str, Any], device: Any):
    if not bool(config["metrics"].get("lpips", True)):
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("metrics.lpips=true requires the lpips package") from exc
    model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    return model


def run_inference(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str = "val",
    max_cases: int | None = None,
    skip_completed: bool = True,
) -> None:
    import torch

    config = copy.deepcopy(dict(config))
    from streamrefine.data.full_volume_dataset import build_full_volume_dataset
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
    )
    from streamrefine.data.persistent_pair_dataset import inverse_model_latent
    from streamrefine.method.calibration import (
        CalibrationExpectation,
        load_benefit_calibration,
    )
    from streamrefine.training.checkpoint import checkpoint_weight_identity
    from streamrefine.training.anatomy import (
        anatomy_metric_contract,
        anatomy_metric_contract_fingerprint,
        canonical_anatomy_metric,
    )
    from streamrefine.training.trainer import build_model, build_vidtok_decoder

    anatomy_metric = canonical_anatomy_metric(
        config["benefit"].get("anatomy_metric", "pir")
    )
    anatomy_contract_fingerprint = anatomy_metric_contract_fingerprint(anatomy_metric)
    anatomy_metric_contract_value = anatomy_metric_contract(anatomy_metric)
    anatomy_mask_policy = str(config["data"]["anatomy_mask_policy"])
    pir_mask_contract = (
        anatomy_mask_contract(config["data"]["dataset"], anatomy_mask_policy)
        if anatomy_metric == "pir"
        else None
    )
    pir_mask_contract_fingerprint = (
        anatomy_mask_contract_fingerprint(
            config["data"]["dataset"], anatomy_mask_policy
        )
        if anatomy_metric == "pir"
        else None
    )
    export_calibration = bool(
        config["benefit"].get("export_calibration_trajectories", False)
    )
    if export_calibration and config["method"]["mode"] != "fixed_k":
        raise ValueError("Calibration trajectory export requires method.mode=fixed_k")
    if export_calibration and str(split).lower() not in {"train", "training"}:
        raise ValueError("Calibration trajectory export requires --split train")
    device = torch.device(str(config["runtime"].get("device", "cuda")))
    seed_everything(
        int(config["inference"].get("seed", 3407)),
        deterministic=bool(config["runtime"].get("deterministic", False)),
    )
    autocast_factory, _ = configure_precision(config["train"]["precision"], device)
    checkpoint = _load_checkpoint(checkpoint_path)
    provenance = _active_provenance(config)
    if config["method"]["mode"] == "anatomy_aware":
        source, target = pair_modalities(config)
        from streamrefine.training.auto_calibration import checkpoint_calibration

        embedded_calibration = checkpoint_calibration(checkpoint, require_adaptive=True)
        calibration_path = str(config["benefit"].get("calibration_path", ""))
        trajectory_contract = str(
            config["benefit"].get("fixed_horizon_trajectory_contract_fingerprint", "")
        ).strip()
        if embedded_calibration is not None:
            if (
                trajectory_contract
                and trajectory_contract
                != embedded_calibration.trajectory_contract_fingerprint
            ):
                raise ValueError(
                    "Configured trajectory contract differs from the checkpoint calibration"
                )
            trajectory_contract = embedded_calibration.trajectory_contract_fingerprint
        if not trajectory_contract:
            raise ValueError(
                "anatomy_aware inference requires benefit.fixed_horizon_trajectory_contract_fingerprint"
            )
        fixed_identity = str(
            config["benefit"].get("fixed_horizon_checkpoint_identity", "")
        ).strip()
        calibration = embedded_calibration or load_benefit_calibration(calibration_path)
        if embedded_calibration is not None and calibration_path:
            if (
                load_benefit_calibration(calibration_path).fingerprint
                != calibration.fingerprint
            ):
                raise ValueError(
                    "Explicit calibration differs from the checkpoint calibration"
                )
        config["benefit"]["fixed_horizon_trajectory_contract_fingerprint"] = (
            trajectory_contract
        )
        artifact_identity = calibration.fixed_horizon_checkpoint_identity
        if fixed_identity and fixed_identity != artifact_identity:
            raise ValueError(
                "Configured fixed-horizon identity differs from the calibration artifact"
            )
        fixed_identity = artifact_identity
        config["benefit"]["fixed_horizon_checkpoint_identity"] = fixed_identity
        expectation = CalibrationExpectation(
            **translation_metric_identity(
                config["benefit"].get("translation_metric", "latent_mae")
            ),
            dataset=config["data"]["dataset"],
            source_modality=source,
            target_modality=target,
            split="train",
            fixed_horizon_checkpoint_identity=fixed_identity,
            latent_statistics_identity=provenance["latent_statistics_sha256"],
            anatomy_metric=anatomy_metric,
            anatomy_contract=anatomy_metric_contract_value,
            anatomy_contract_fingerprint=anatomy_contract_fingerprint,
            anatomy_mask_policy=anatomy_mask_policy
            if anatomy_metric == "pir"
            else None,
            anatomy_mask_contract=pir_mask_contract,
            anatomy_mask_contract_fingerprint=pir_mask_contract_fingerprint,
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
            trajectory_contract_fingerprint=trajectory_contract,
            k_max=int(config["rollout"]["k_max"]),
        )
        calibration.assert_compatible(expectation)
        provenance = {
            **provenance,
            "benefit_calibration_fingerprint": calibration.fingerprint,
        }
    compatibility_provenance = provenance
    if export_calibration:
        compatibility_provenance = {
            key: value
            for key, value in provenance.items()
            if key
            not in {
                "anatomy_metric",
                "anatomy_contract_fingerprint",
                "translation_metric",
                "translation_contract",
                "translation_contract_fingerprint",
            }
        }
    _assert_inference_compatible(checkpoint, config, compatibility_provenance)
    model = build_model(config).to(device).eval()
    _load_model_state(
        model, checkpoint, use_ema=bool(config["inference"].get("use_ema", True))
    )
    decoder = build_vidtok_decoder(config, device)
    predictor = ModelWindowPredictor(model, config, autocast_factory)
    lpips_model = _build_lpips_model(config, device)
    dataset = build_full_volume_dataset(config, split=split)
    output_dir = _inference_output_dir(config)
    inference_contract_fingerprint = _contract_fingerprint(
        config, checkpoint_path, provenance
    )
    writer = InferenceArtifactWriter(
        output_dir, contract_fingerprint=inference_contract_fingerprint
    )
    checkpoint_identity = checkpoint_weight_identity(
        checkpoint_path, use_ema=bool(config["inference"].get("use_ema", True))
    )
    calibration_cache_provenance = {
        key: provenance[key]
        for key in (
            "tokenizer_checkpoint_sha256",
            "tokenizer_config_sha256",
            "generation_contract_sha256",
            "cache_schema",
            "anatomy_mask_policy",
            "anatomy_mask_contract_fingerprint",
        )
    }
    calibration_cache_provenance_fingerprint = hashlib.sha256(
        canonical_config_json(calibration_cache_provenance).encode("utf-8")
    ).hexdigest()
    if export_calibration:
        atomic_write_json(
            output_dir / "calibration_cache_provenance.json",
            calibration_cache_provenance,
        )
    processed = 0
    for index in range(len(dataset)):
        predictor.reset_case()
        prospective_case = _case_id_without_loading(dataset, index)
        if (
            skip_completed
            and prospective_case
            and writer.completed_record(prospective_case)
        ):
            continue
        sample = dataset[index]
        case_id = str(sample["case_id"])
        if skip_completed and writer.completed_record(case_id):
            continue
        source_latent = sample["source_latent_model"].to(device)
        valid_mask = sample["valid_mask_latent_soft"].to(device)
        coordinates = sample["global_coords_dhw"].to(device)
        refinement = synchronized_sliding_refinement(
            source_latent=source_latent,
            valid_mask_soft=valid_mask,
            global_coordinates=coordinates,
            predictor=predictor,
            controller=_stop_callable(config),
            k_max=int(config["rollout"]["k_max"]),
            window_shape_dhw=config["inference"]["window_size_dhw"],
            overlap=float(config["inference"]["overlap"]),
            seed=int(config["inference"].get("seed", 3407)),
            case_id=case_id,
        )
        mean = sample["latent_stats_mean"].to(device)
        std = sample["latent_stats_std"].to(device)
        original_hwd = tuple((int(value) for value in sample["original_shape_hwd"]))
        target_image = crop_to_original_hwd(
            sample["preprocessed_gt_image"], original_hwd
        )
        source_image = crop_to_original_hwd(
            sample["preprocessed_source_image"], original_hwd
        )
        pir_reference = None
        pir_support_cell_count = None
        pir_valid_edge_count = None
        if export_calibration:
            from streamrefine.training.pir import build_minimal_pir_reference

            with torch.no_grad():
                pir_reference = build_minimal_pir_reference(
                    sample["source_latent_raw"].unsqueeze(0).to(device),
                    sample["target_latent_raw"].unsqueeze(0).to(device),
                    sample["valid_mask_latent_hard"].unsqueeze(0).to(device),
                    sample["anatomy_mask_latent_hard"].unsqueeze(0).to(device),
                )
                pir_support_cell_count = int(
                    sample["pir_mask_latent_hard"].sum().item()
                )
                pir_valid_edge_count = int(pir_reference.edge_count[0].item())
        decoded_states, state_records = ([], [])
        for state in refinement.states:
            decode_start = time.perf_counter()
            raw = inverse_model_latent(state.latent.float(), mean, std)
            decoded_padded = tiled_decode_raw_latent(
                raw,
                decoder=decoder,
                tile_shape_dhw=config["vidtok"]["decode_window_dhw"],
                overlap=float(config["vidtok"]["decode_overlap"]),
                compression_dhw=(4, 8, 8),
                batch_size=int(config["vidtok"]["decode_batch_size"]),
                precision=str(config["train"]["precision"]),
            )
            decoded = crop_to_original_hwd(decoded_padded, original_hwd)
            decoder_seconds = time.perf_counter() - decode_start
            prediction_for_metric = decoded.unsqueeze(0).to(device)
            target_for_metric = target_image.unsqueeze(0).to(device)
            metrics = compute_volume_metrics(
                prediction_for_metric,
                target_for_metric,
                enabled=config["metrics"],
                data_range=float(config["metrics"].get("data_range", 2.0)),
                lpips_model=lpips_model,
                lpips_batch_size=int(config["metrics"].get("lpips_batch_size", 16)),
            )
            decoded_states.append(decoded.cpu())
            translation_loss = float(
                (prediction_for_metric - target_for_metric).abs().mean().item()
            )
            if (
                export_calibration
                and config["benefit"].get("translation_metric", "latent_mae")
                == "latent_mae"
            ):
                translation_loss = float(
                    latent_state_mae(
                        state.latent.unsqueeze(0),
                        sample["target_latent_model"].unsqueeze(0),
                        valid_mask.unsqueeze(0),
                    )[0].item()
                )
            anatomy_loss = None
            if export_calibration:
                with torch.no_grad():
                    from streamrefine.training.pir import minimal_pir_loss

                    anatomy_loss = float(
                        minimal_pir_loss(raw.unsqueeze(0), pir_reference)[0].item()
                    )
            state_records.append(
                {
                    "step": state.step,
                    "benefit_score": state.benefit_score,
                    "stopped": state.stopped,
                    "forced_stop": state.forced_stop,
                    "window_score_variance": state.window_score_variance,
                    "seam_mse": state.seam_mse,
                    "sampling_seconds": state.sampling_seconds,
                    "decoder_seconds": decoder_seconds,
                    "window_count": state.window_count,
                    "model_evaluations": state.model_evaluations,
                    "metrics": metrics,
                    "translation_loss": translation_loss
                    if export_calibration
                    else None,
                    "translation_metric": config["benefit"].get(
                        "translation_metric", "latent_mae"
                    )
                    if export_calibration
                    else None,
                    "anatomy_loss": anatomy_loss,
                    "anatomy_metric": anatomy_metric if export_calibration else None,
                }
            )
        writer.write_case(
            case_id=case_id,
            source=source_image,
            target=target_image,
            decoded_states=decoded_states,
            state_records=state_records,
            case_metadata={
                key: sample.get(key)
                for key in (
                    "dataset",
                    "split",
                    "pair_id",
                    "group_id",
                    "source_modality",
                    "target_modality",
                    "cohort",
                    "task",
                    "anatomy",
                    "tracer",
                    "tokenizer_checkpoint_sha256",
                    "tokenizer_config_sha256",
                    "generation_contract_sha256",
                    "input_snapshot_sha256",
                )
            }
            | {
                "calibration_export": export_calibration,
                **translation_metric_identity(
                    config["benefit"].get("translation_metric", "latent_mae")
                ),
                "anatomy_metric": anatomy_metric if export_calibration else "",
                "anatomy_contract": anatomy_metric_contract_value
                if export_calibration
                else {},
                "anatomy_contract_fingerprint": anatomy_contract_fingerprint
                if export_calibration
                else "",
                "fixed_horizon_checkpoint_identity": checkpoint_identity
                if export_calibration
                else "",
                "latent_statistics_identity": provenance["latent_statistics_sha256"]
                if export_calibration
                else "",
                "k_max": int(config["rollout"]["k_max"]),
                "cache_provenance": calibration_cache_provenance
                if export_calibration
                else {},
                "cache_provenance_fingerprint": calibration_cache_provenance_fingerprint
                if export_calibration
                else "",
                "trajectory_contract_fingerprint": inference_contract_fingerprint
                if export_calibration
                else "",
            }
            | (
                {
                    "anatomy_mask_policy": anatomy_mask_policy,
                    "anatomy_mask_contract": pir_mask_contract,
                    "anatomy_mask_contract_fingerprint": pir_mask_contract_fingerprint,
                    "pir_support_cell_count": pir_support_cell_count,
                    "pir_valid_edge_count": pir_valid_edge_count,
                }
                if export_calibration and anatomy_metric == "pir"
                else {}
            ),
            save_mid_slices=bool(config["artifacts"].get("save_mid_slices", True)),
            save_all_states=bool(config["artifacts"].get("save_all_states", True)),
        )
        processed += 1
        predictor.reset_case()
        if max_cases is not None and processed >= int(max_cases):
            break
