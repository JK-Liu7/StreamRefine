"""Configuration composition for StreamRefine.

The module deliberately has no Torch, MONAI, or VidTok imports so command-line
``--help`` and configuration inspection work on lightweight machines.
"""

from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping
from streamrefine.paths import resolve_config_path, resolve_runtime_paths

CONFIG_CONTRACT_VERSION = "streamrefine_release_v1"
ANATOMY_METRICS = frozenset({"pir"})
METHOD_MODES = frozenset({"anatomy_aware"})
DATASET_PAIRS: dict[str, tuple[str, ...]] = {
    "brats24": ("t2w_to_t2f", "t1c_to_t1n", "t2f_to_t2w", "t1n_to_t2w", "t1n_to_t1c"),
    "synthrad": ("mr_to_ct", "cbct_to_ct"),
    "autopet": ("ct_to_pet",),
}


class ConfigError(ValueError):
    """Raised when a composed configuration violates the public contract."""


def builtin_defaults():
    return {
        "config_contract_version": "streamrefine_release_v1",
        "project": {
            "root": ".",
            "output_root": "result/streamrefine",
            "experiment": "streamrefine_flow",
        },
        "data": {
            "dataset": "brats24",
            "pair": "t2w_to_t2f",
            "train_manifest": "result/streamrefine_latents/brats24/pair_manifest_train.jsonl",
            "val_manifest": "result/streamrefine_latents/brats24/pair_manifest_val.jsonl",
            "test_manifest": "",
            "path_remap": [],
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "anatomy_mask_policy": "brats24_nonzero_union_v1",
            "supported_pairs": [
                "t2w_to_t2f",
                "t1c_to_t1n",
                "t2f_to_t2w",
                "t1n_to_t2w",
                "t1n_to_t1c",
            ],
            "preprocessing_config": "ar/configs/cache/brats24_vidtok_kl16_mean.yaml",
        },
        "persistent_cache": {
            "root": "result/persistent_cache",
            "namespace": "streamrefine_pair_v3_anatomy_mask",
        },
        "latent": {
            "cache_schema": "streamrefine_kl_mean_v2_anatomy_mask",
            "channels": 16,
            "layout": "C,D,H,W",
            "stats_path": "result/streamrefine_latents/brats24/latent_stats_train.json",
            "crop_size_dhw": [24, 24, 24],
            "image_crop_size_dhw": [96, 192, 192],
        },
        "augmentation": {
            "enabled": True,
            "flip_probability_d": 0.5,
            "flip_probability_h": 0.5,
            "flip_probability_w": 0.5,
            "rotate90_hw_probability": 0.2,
        },
        "model": {
            "in_channels": 16,
            "out_channels": 16,
            "hidden_size": 512,
            "depth": 12,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "patch_size_dhw": [2, 2, 2],
            "rope_base": 10000.0,
            "max_refinement_steps": 4,
            "gradient_checkpointing": False,
            "detach_benefit_input": True,
        },
        "flow": {
            "objective": "rectified_flow",
            "prediction": "velocity",
            "train_time_distribution": "uniform",
            "inference_solver": "euler",
        },
        "rollout": {
            "k_max": 4,
            "inner_steps": 4,
            "refinement_start": "previous_state",
            "history_detach": True,
            "sync_flow_time_across_ranks": True,
        },
        "method": {
            "mode": "anatomy_aware",
            "sentinel_probability": 0.2,
            "stop_threshold": 0.0,
            "policy_warmup_steps": 10000,
        },
        "benefit": {
            "translation_metric": "latent_mae",
            "anatomy_metric": "pir",
            "lambda_anatomy": 1.0,
            "lambda_compute": 0.01,
            "translation_scale": 1.0,
            "anatomy_scale": 1.0,
            "fixed_horizon_checkpoint_identity": "",
            "fixed_horizon_trajectory_contract_fingerprint": "",
            "calibration_path": "",
            "calibration_required_for_anatomy": True,
            "export_calibration_trajectories": False,
            "auto_calibration": {
                "enabled": True,
                "max_cases": 32,
                "head_warmup_steps": 1000,
            },
        },
        "losses": {"flow_weight": 1.0, "latent_l1_weight": 0.2, "stop_weight": 1.0},
        "train": {
            "precision": "bf16",
            "batch_size_per_gpu": 4,
            "grad_accumulation": 1,
            "learning_rate": 2e-05,
            "weight_decay": 0.0,
            "max_steps": 100000,
            "warmup_steps": 2000,
            "gradient_clip_norm": 2.0,
            "ema": True,
            "ema_decay": 0.9995,
            "init_use_ema": True,
            "auto_resume": True,
            "checkpoint_every": 4000,
            "validate_every": 4000,
            "log_every": 200,
            "seed": 3407,
        },
        "validation": {
            "max_batches": 4,
            "max_cases": 16,
            "fixed_k": 4,
            "selection_metric": "adaptive_utility",
            "selection_direction": "min",
            "oracle_tolerance": "1e-8",
            "compute_anatomy": True,
            "threshold_sweep": [-0.05, 0.0, 0.05],
        },
        "diagnostics": {"trajectory": {"enabled": True}},
        "inference": {
            "window_size_dhw": [24, 12, 12],
            "overlap": 0.25,
            "blend_mode": "gaussian",
            "score_aggregation": "valid_weighted_mean",
            "use_ema": True,
            "seed": 3407,
        },
        "vidtok": {
            "repo": "external/VidTok",
            "config": "external/VidTok/configs/vidtok_kl_noncausal_488_16chn.yaml",
            "checkpoint": "checkpoints/vidtok_kl_noncausal_488_16chn.ckpt",
            "decode_window_dhw": [8, 32, 32],
            "decode_overlap": 0.25,
            "decode_batch_size": 1,
        },
        "metrics": {
            "mae": True,
            "psnr": True,
            "ssim": True,
            "lpips": True,
            "lpips_mode": "slice_wise_2d",
            "lpips_batch_size": 16,
            "data_range": 2.0,
        },
        "artifacts": {"save_mid_slices": True, "save_all_states": True},
        "runtime": {
            "device": "cuda",
            "deterministic": False,
            "distributed_timeout_minutes": 120.0,
        },
    }


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    """Load YAML, with JSON-compatible YAML supported without PyYAML."""
    if path in (None, ""):
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"PyYAML is unavailable and {config_path} is not JSON-compatible YAML"
            ) from exc
    else:
        value = yaml.safe_load(text)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Top level of {config_path} must be a mapping")
    return dict(value)


def _parse_override_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false", "null", "none"}:
        return {"true": True, "false": False, "null": None, "none": None}[lowered]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def set_dotted(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ConfigError("An override key cannot be empty")
    cursor: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            child: dict[str, Any] = {}
            cursor[key] = child
            cursor = child
        elif isinstance(existing, MutableMapping):
            cursor = existing
        else:
            raise ConfigError(f"Cannot assign {dotted_key!r}: {key!r} is not a mapping")
    cursor[keys[-1]] = value


def apply_overrides(
    config: Mapping[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must use KEY=VALUE syntax, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(result, key.strip(), _parse_override_value(raw.strip()))
    return result


def _triplet(value: Any, name: str, *, positive: bool = True) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{name} must contain exactly three integers")
    try:
        result = tuple((int(v) for v in value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must contain integers") from exc
    if positive and any((v <= 0 for v in result)):
        raise ConfigError(f"{name} values must be positive, got {result}")
    return result


def validate_config(
    config: Mapping[str, Any], *, require_runtime_paths: bool = False
) -> None:
    if set(config.get("losses", {})) != {
        "flow_weight",
        "latent_l1_weight",
        "stop_weight",
    }:
        raise ConfigError(
            "losses supports only flow_weight, latent_l1_weight, and stop_weight"
        )
    defaults = builtin_defaults()
    for section in ("method", "inference", "runtime", "diagnostics", "rollout"):
        allowed = set(defaults[section])
        if section == "inference":
            allowed.add("output_dir")
        unknown = set(config.get(section, {})) - allowed
        if unknown:
            raise ConfigError(f"Unsupported {section} options: {sorted(unknown)}")
    required_sections = {
        "project",
        "data",
        "persistent_cache",
        "latent",
        "model",
        "flow",
        "rollout",
        "method",
        "benefit",
        "losses",
        "train",
        "validation",
        "diagnostics",
        "inference",
        "vidtok",
        "metrics",
        "artifacts",
        "runtime",
    }
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")
    dataset = str(config["data"].get("dataset", "")).lower()
    pair = str(config["data"].get("pair", "")).lower()
    if dataset not in DATASET_PAIRS:
        raise ConfigError(
            f"Unsupported dataset {dataset!r}; expected one of {sorted(DATASET_PAIRS)}"
        )
    if pair not in DATASET_PAIRS[dataset]:
        raise ConfigError(
            f"Unsupported {dataset} pair {pair!r}; expected one of {DATASET_PAIRS[dataset]}"
        )
    mode = str(config["method"].get("mode", ""))
    if mode not in METHOD_MODES:
        raise ConfigError(
            f"Unsupported method.mode {mode!r}; expected one of {sorted(METHOD_MODES)}"
        )
    epsilon_value = config["method"].get("sentinel_probability", 0.0)
    if isinstance(epsilon_value, bool):
        raise ConfigError("method.sentinel_probability must lie in (0,1]")
    try:
        epsilon = float(epsilon_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("method.sentinel_probability must lie in (0,1]") from exc
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ConfigError("method.sentinel_probability must lie in (0,1]")
    threshold_value = config["method"].get("stop_threshold")
    if isinstance(threshold_value, bool):
        raise ConfigError("method.stop_threshold must be finite")
    try:
        stop_threshold = float(threshold_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("method.stop_threshold must be finite") from exc
    if not math.isfinite(stop_threshold):
        raise ConfigError("method.stop_threshold must be finite")
    warmup = config["method"].get("policy_warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ConfigError("method.policy_warmup_steps must be a non-negative integer")
    automatic = config["benefit"].get("auto_calibration", {})
    if not isinstance(automatic, Mapping):
        raise ConfigError("benefit.auto_calibration must be a mapping")
    auto_enabled = automatic.get("enabled", False)
    if not isinstance(auto_enabled, bool):
        raise ConfigError("benefit.auto_calibration.enabled must be boolean")
    for key, default, minimum in (("max_cases", 32, 1), ("head_warmup_steps", 1000, 0)):
        value = automatic.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(
                f"benefit.auto_calibration.{key} must be an integer >= {minimum}"
            )
    if auto_enabled:
        if mode != "anatomy_aware":
            raise ConfigError(
                "Automatic calibration requires method.mode=anatomy_aware"
            )
        if warmup <= 0:
            raise ConfigError("Automatic calibration requires policy_warmup_steps > 0")
        adaptive_start = warmup + int(automatic.get("head_warmup_steps", 1000))
        if int(config["train"]["max_steps"]) <= adaptive_start:
            raise ConfigError(
                "train.max_steps must exceed automatic calibration warmup plus head_warmup_steps"
            )
        if any(
            (
                str(config["benefit"].get(key, "")).strip()
                for key in (
                    "calibration_path",
                    "fixed_horizon_checkpoint_identity",
                    "fixed_horizon_trajectory_contract_fingerprint",
                )
            )
        ):
            raise ConfigError(
                "Automatic calibration cannot be combined with explicit calibration artifacts or identities"
            )
        if bool(config["benefit"].get("export_calibration_trajectories", False)):
            raise ConfigError("Automatic calibration manages its own trajectory export")
    benefit_values: dict[str, float] = {}
    translation_metric = config["benefit"].get("translation_metric")
    if not isinstance(translation_metric, str) or translation_metric not in {
        "latent_mae"
    }:
        raise ConfigError("benefit.translation_metric must be 'latent_mae'")
    anatomy_metric = config["benefit"].get("anatomy_metric")
    if anatomy_metric not in ANATOMY_METRICS:
        raise ConfigError("benefit.anatomy_metric must be 'pir'")
    for key in ("lambda_anatomy", "lambda_compute"):
        value = config["benefit"].get(key)
        if isinstance(value, bool):
            raise ConfigError(f"benefit.{key} must be finite and non-negative")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"benefit.{key} must be finite and non-negative") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ConfigError(f"benefit.{key} must be finite and non-negative")
        benefit_values[key] = numeric
    for key in ("translation_scale", "anatomy_scale"):
        value = config["benefit"].get(key)
        if isinstance(value, bool):
            raise ConfigError(f"benefit.{key} must be finite and positive")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"benefit.{key} must be finite and positive") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ConfigError(f"benefit.{key} must be finite and positive")
    if mode == "anatomy_aware" and benefit_values["lambda_anatomy"] <= 0.0:
        raise ConfigError("anatomy_aware mode requires benefit.lambda_anatomy > 0")
    channels = int(config["latent"].get("channels", -1))
    if channels != 16:
        raise ConfigError("StreamRefine cache-v2 requires latent.channels == 16")
    if config["latent"].get("cache_schema") != "streamrefine_kl_mean_v2_anatomy_mask":
        raise ConfigError(
            "StreamRefine training requires cache schema streamrefine_kl_mean_v2_anatomy_mask"
        )
    from streamrefine.data.anatomy_mask import canonical_anatomy_mask_policy

    try:
        canonical_anatomy_mask_policy(
            config["data"].get("dataset"), config["data"].get("anatomy_mask_policy")
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if (
        int(config["model"].get("in_channels", -1)) != channels
        or int(config["model"].get("out_channels", -1)) != channels
    ):
        raise ConfigError("model in/out channels must match latent.channels")
    hidden = int(config["model"].get("hidden_size", 0))
    heads = int(config["model"].get("num_heads", 0))
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ConfigError(
            "model.hidden_size must be positive and divisible by model.num_heads"
        )
    crop = _triplet(config["latent"].get("crop_size_dhw"), "latent.crop_size_dhw")
    image_crop = _triplet(
        config["latent"].get("image_crop_size_dhw"), "latent.image_crop_size_dhw"
    )
    if any((image % latent for image, latent in zip(image_crop, crop))):
        raise ConfigError(
            "latent.image_crop_size_dhw must be an integer multiple of latent.crop_size_dhw"
        )
    patch = _triplet(config["model"].get("patch_size_dhw"), "model.patch_size_dhw")
    if any((c % p for c, p in zip(crop, patch))):
        raise ConfigError(
            f"latent crop {crop} must be divisible by model patch {patch}"
        )
    window = _triplet(
        config["inference"].get("window_size_dhw"), "inference.window_size_dhw"
    )
    if any((w % p for w, p in zip(window, patch))):
        raise ConfigError(
            f"inference window {window} must be divisible by model patch {patch}"
        )
    overlap = float(config["inference"].get("overlap", -1.0))
    if not 0.0 <= overlap < 1.0:
        raise ConfigError("inference.overlap must satisfy 0 <= overlap < 1")
    if (
        config["flow"].get("objective") != "rectified_flow"
        or config["flow"].get("prediction") != "velocity"
    ):
        raise ConfigError(
            "The approved flow contract is rectified_flow velocity prediction"
        )
    if config["flow"].get("train_time_distribution") != "uniform":
        raise ConfigError("Only flow.train_time_distribution=uniform is implemented")
    if config["flow"].get("inference_solver") != "euler":
        raise ConfigError("Only the approved Euler solver is implemented")
    k_max = int(config["rollout"].get("k_max", 0))
    if k_max <= 0 or int(config["rollout"].get("inner_steps", 0)) <= 0:
        raise ConfigError("rollout.k_max and rollout.inner_steps must be positive")
    if config["rollout"].get("refinement_start", "previous_state") != "previous_state":
        raise ConfigError("rollout.refinement_start must be 'previous_state'")
    if int(config["model"].get("max_refinement_steps", k_max)) < k_max:
        raise ConfigError("model.max_refinement_steps must cover rollout.k_max")
    if not bool(config["rollout"].get("history_detach", False)):
        raise ConfigError("rollout.history_detach must remain true for Self-Forcing")
    if int(config["validation"].get("fixed_k", k_max)) != k_max:
        raise ConfigError("validation.fixed_k must equal rollout.k_max")
    max_cases = config["validation"].get("max_cases", 0)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        raise ConfigError("validation.max_cases must be a positive integer")
    selection_metric = str(config["validation"].get("selection_metric", "")).strip()
    normalized_selection_metric = (
        selection_metric[4:]
        if selection_metric.startswith("val/")
        else selection_metric
    )
    metric_directions = {
        "adaptive_utility": "min",
        "adaptive_mae": "min",
        "adaptive_quality": "min",
        "adaptive_ssim": "max",
        "wrong_stop_regret": "min",
        "under_stop_rate": "min",
        "over_stop_rate": "min",
        "full_trajectory_utility": "min",
    }
    if normalized_selection_metric not in metric_directions:
        raise ConfigError(
            f"validation.selection_metric must name a supported scalar deployment metric; expected one of {sorted(metric_directions)}"
        )
    if (
        normalized_selection_metric == "wrong_stop_regret"
        and (not auto_enabled)
        and (
            not str(config["benefit"].get("calibration_path", "")).strip()
            or not str(
                config["benefit"].get(
                    "fixed_horizon_trajectory_contract_fingerprint", ""
                )
            ).strip()
        )
    ):
        raise ConfigError(
            "validation.selection_metric=wrong_stop_regret requires an explicit benefit.calibration_path and fixed-horizon trajectory contract so anatomy-aware suffix regret has calibrated s_L/s_A scales"
        )
    selection_direction = config["validation"].get("selection_direction", "")
    if selection_direction not in {"min", "max"}:
        raise ConfigError("validation.selection_direction must be 'min' or 'max'")
    expected_direction = metric_directions[normalized_selection_metric]
    if selection_direction != expected_direction:
        raise ConfigError(
            f"validation.selection_direction for {normalized_selection_metric} must be {expected_direction!r}"
        )
    oracle_tolerance_value = config["validation"].get("oracle_tolerance", -1.0)
    if isinstance(oracle_tolerance_value, bool):
        raise ConfigError("validation.oracle_tolerance must be finite and non-negative")
    try:
        oracle_tolerance = float(oracle_tolerance_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "validation.oracle_tolerance must be finite and non-negative"
        ) from exc
    if not math.isfinite(oracle_tolerance) or oracle_tolerance < 0.0:
        raise ConfigError("validation.oracle_tolerance must be finite and non-negative")
    if not isinstance(config["validation"].get("compute_anatomy", False), bool):
        raise ConfigError("validation.compute_anatomy must be boolean")
    threshold_sweep = config["validation"].get("threshold_sweep", ())
    if not isinstance(threshold_sweep, (list, tuple)):
        raise ConfigError("validation.threshold_sweep must be a list of finite numbers")
    for index, threshold in enumerate(threshold_sweep):
        if isinstance(threshold, bool):
            raise ConfigError(
                f"validation.threshold_sweep[{index}] must be a finite number, not bool"
            )
        try:
            numeric_threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"validation.threshold_sweep[{index}] must be a finite number"
            ) from exc
        if not math.isfinite(numeric_threshold):
            raise ConfigError(
                f"validation.threshold_sweep[{index}] must be a finite number"
            )
    if "use_ema" in config["validation"]:
        raise ConfigError(
            "validation.use_ema is not a separate option; validation must reuse inference.use_ema so checkpoint selection and deployment use the same weights"
        )
    trajectory_diagnostics = config["diagnostics"].get("trajectory")
    if not isinstance(trajectory_diagnostics, Mapping) or not isinstance(
        trajectory_diagnostics.get("enabled"), bool
    ):
        raise ConfigError("diagnostics.trajectory.enabled must be boolean")
    if str(config["inference"].get("blend_mode", "")) != "gaussian":
        raise ConfigError("Only inference.blend_mode=gaussian is implemented")
    if str(config["inference"].get("score_aggregation", "")) != "valid_weighted_mean":
        raise ConfigError(
            "Only inference.score_aggregation=valid_weighted_mean is implemented"
        )
    if not isinstance(config["inference"].get("use_ema", True), bool):
        raise ConfigError("inference.use_ema must be boolean")
    if not isinstance(config["train"].get("ema", True), bool):
        raise ConfigError("train.ema must be boolean")
    if bool(config["inference"].get("use_ema", True)) and (
        not bool(config["train"].get("ema", True))
    ):
        raise ConfigError(
            "inference.use_ema=true requires train.ema=true; training validation reuses the deployment weight selection"
        )
    if str(config["train"].get("precision", "")).lower() not in {"bf16", "fp32"}:
        raise ConfigError(
            "train.precision must be bf16 or fp32; no silent fallback is allowed"
        )
    if not isinstance(config["train"].get("init_use_ema", True), bool):
        raise ConfigError("train.init_use_ema must be boolean")
    for key in (
        "batch_size_per_gpu",
        "grad_accumulation",
        "max_steps",
        "checkpoint_every",
        "validate_every",
        "log_every",
    ):
        if int(config["train"].get(key, 0)) <= 0:
            raise ConfigError(f"train.{key} must be positive")
    if int(config["validation"].get("max_batches", 0)) <= 0:
        raise ConfigError("validation.max_batches must be positive")
    decode_batch_size = config["vidtok"].get("decode_batch_size")
    if (
        isinstance(decode_batch_size, bool)
        or not isinstance(decode_batch_size, int)
        or decode_batch_size <= 0
    ):
        raise ConfigError("vidtok.decode_batch_size must be a positive integer")
    rotate_probability = float(
        config["augmentation"].get("rotate90_hw_probability", 0.0)
    )
    if not 0.0 <= rotate_probability <= 1.0:
        raise ConfigError("augmentation.rotate90_hw_probability must lie in [0,1]")
    if bool(config["augmentation"].get("enabled", True)) and rotate_probability > 0:
        if crop[1] != crop[2] or image_crop[1] != image_crop[2]:
            raise ConfigError(
                "90-degree H/W augmentation requires square latent and image H/W crops"
            )
    if str(config["metrics"].get("lpips_mode", "")) != "slice_wise_2d":
        raise ConfigError("Only metrics.lpips_mode=slice_wise_2d is implemented")
    if float(config["metrics"].get("data_range", 0.0)) <= 0:
        raise ConfigError("metrics.data_range must be positive")
    distributed_timeout = config["runtime"].get("distributed_timeout_minutes", 0.0)
    if isinstance(distributed_timeout, bool):
        raise ConfigError(
            "runtime.distributed_timeout_minutes must be finite and positive"
        )
    try:
        distributed_timeout = float(distributed_timeout)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "runtime.distributed_timeout_minutes must be finite and positive"
        ) from exc
    if not math.isfinite(distributed_timeout) or distributed_timeout <= 0.0:
        raise ConfigError(
            "runtime.distributed_timeout_minutes must be finite and positive"
        )
    if (
        mode == "anatomy_aware"
        and (not auto_enabled)
        and bool(config["benefit"].get("calibration_required_for_anatomy", True))
    ):
        calibration = str(config["benefit"].get("calibration_path", ""))
        trajectory_contract = str(
            config["benefit"].get("fixed_horizon_trajectory_contract_fingerprint", "")
        )
        if require_runtime_paths and (not calibration):
            raise ConfigError("anatomy_aware mode requires benefit.calibration_path")
        if require_runtime_paths and (not trajectory_contract):
            raise ConfigError(
                "anatomy_aware mode requires benefit.fixed_horizon_trajectory_contract_fingerprint"
            )
    if require_runtime_paths:
        for key in ("train_manifest", "val_manifest"):
            if not str(config["data"].get(key, "")):
                raise ConfigError(f"data.{key} must be configured")
        if not str(config["latent"].get("stats_path", "")):
            raise ConfigError("latent.stats_path must be configured")


def resolve_config(
    *,
    base_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    method_path: str | Path | None = None,
    overrides: Iterable[str] = (),
    explicit: Mapping[str, Any] | None = None,
    require_runtime_paths: bool = False,
) -> dict[str, Any]:
    """Compose built-ins < base < dataset < method < CLI overrides."""
    config = builtin_defaults()
    for path in (base_path, dataset_path, method_path):
        if path not in (None, ""):
            config = deep_merge(config, load_config_file(resolve_config_path(path)))
    config = apply_overrides(config, overrides)
    for key, value in (explicit or {}).items():
        if value is not None:
            set_dotted(config, key, value)
    config = resolve_runtime_paths(config)
    validate_config(config, require_runtime_paths=require_runtime_paths)
    return config


def canonical_config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def pair_modalities(config: Mapping[str, Any]) -> tuple[str, str]:
    pair = str(config["data"]["pair"])
    if "_to_" not in pair:
        raise ConfigError(f"Pair must use source_to_target form, got {pair!r}")
    source, target = pair.split("_to_", 1)
    return (source, target)
