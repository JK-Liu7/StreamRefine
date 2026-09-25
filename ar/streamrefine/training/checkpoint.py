"""Strict, atomic StreamRefine checkpoint and resume handling."""

from __future__ import annotations
import contextlib
import copy
import hashlib
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from streamrefine.config import canonical_config_json, config_fingerprint

CHECKPOINT_VERSION = "streamrefine_checkpoint_v1"
INITIALIZATION_PROVENANCE_KEYS = (
    "latent_statistics_sha256",
    "tokenizer_checkpoint_sha256",
    "tokenizer_config_sha256",
    "generation_contract_sha256",
    "cache_schema",
)


class CheckpointCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointAction:
    kind: str
    path: Path | None


def build_compatibility_contract(
    config: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    """Bind exact config and runtime data provenance for deterministic resume."""
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "resolved_config_json": canonical_config_json(config),
        "method_mode": str(config["method"]["mode"]),
        "model_geometry": copy.deepcopy(dict(config["model"])),
        "dataset": str(config["data"]["dataset"]),
        "pair": str(config["data"]["pair"]),
        "cache_schema": str(config["latent"]["cache_schema"]),
        "distributed_world_size": int(world_size),
        "latent_stats_path": str(config["latent"].get("stats_path", "")),
        "optimizer": {
            "learning_rate": config["train"]["learning_rate"],
            "weight_decay": config["train"]["weight_decay"],
        },
        "ema": {
            "enabled": bool(config["train"].get("ema", False)),
            "decay": config["train"].get("ema_decay"),
        },
        "provenance": copy.deepcopy(dict(provenance or {})),
    }


def assert_compatible(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if dict(actual) != dict(expected):
        keys = sorted(set(actual).union(expected))
        mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
        raise CheckpointCompatibilityError(
            "Checkpoint compatibility contract differs in: " + ", ".join(mismatches)
        )


def assert_initialization_compatible(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Allow method/calibration changes while binding model and cache identity."""
    for key in ("model_geometry", "dataset", "pair", "cache_schema"):
        if actual.get(key) != expected.get(key):
            raise CheckpointCompatibilityError(
                f"Initialization checkpoint is incompatible in {key}"
            )
    actual_provenance = actual.get("provenance")
    expected_provenance = expected.get("provenance")
    if not isinstance(actual_provenance, Mapping) or not isinstance(
        expected_provenance, Mapping
    ):
        raise CheckpointCompatibilityError("Initialization checkpoint lacks provenance")
    mismatches = [
        key
        for key in INITIALIZATION_PROVENANCE_KEYS
        if actual_provenance.get(key) != expected_provenance.get(key)
    ]
    if mismatches:
        raise CheckpointCompatibilityError(
            "Initialization checkpoint provenance differs in: " + ", ".join(mismatches)
        )


def checkpoint_weight_identity(path: str | Path, *, use_ema: bool) -> str:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    kind = "ema" if bool(use_ema) else "model"
    return f"sha256:{digest.hexdigest()}:weights={kind}"


def _initialization_model_state(
    checkpoint: Mapping[str, Any], *, use_ema: bool
) -> Mapping[str, Any]:
    if not use_ema:
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise CheckpointCompatibilityError("Checkpoint model state is malformed")
        return state
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping):
        raise CheckpointCompatibilityError(
            "train.init_use_ema=true but initialization checkpoint contains no EMA"
        )
    shadow, nonfloating = (ema.get("shadow"), ema.get("nonfloating"))
    if not isinstance(shadow, Mapping) or not isinstance(nonfloating, Mapping):
        raise CheckpointCompatibilityError("Initialization checkpoint EMA is malformed")
    return {**dict(nonfloating), **dict(shadow)}


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()
    try:
        import torch
    except ImportError:
        return state
    state["torch_cpu"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        import torch

        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    ema: Any,
    step: int,
    epoch: int,
    best_validation: float | None,
    config: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    policy_state: Mapping[str, Any],
    best_validation_metric: str | None = None,
    best_validation_metrics: Mapping[str, Any] | None = None,
    best_selection_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "best_validation": best_validation,
        "best_validation_metric": best_validation_metric,
        "best_validation_metrics": copy.deepcopy(
            None if best_validation_metrics is None else dict(best_validation_metrics)
        ),
        "best_selection_contract": copy.deepcopy(
            None if best_selection_contract is None else dict(best_selection_contract)
        ),
        "resolved_config": copy.deepcopy(dict(config)),
        "compatibility": copy.deepcopy(dict(compatibility)),
        "policy_state": copy.deepcopy(dict(policy_state)),
        "rng_state": capture_rng_state(),
    }


def atomic_torch_save(value: Any, path: str | Path) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, tmp_name)
        with open(tmp_name, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _torch_load(path: Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    result = dict(value)
    if result.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint version in {path}: {result.get('checkpoint_version')!r}"
        )
    from .auto_calibration import checkpoint_calibration

    checkpoint_calibration(result)
    return result


def load_checkpoint_header(path: str | Path) -> dict[str, Any]:
    value = _torch_load(Path(path), map_location="cpu")
    return {
        "checkpoint_version": value["checkpoint_version"],
        "compatibility": value.get("compatibility"),
        "step": value.get("step"),
    }


def resolve_checkpoint_action(
    *,
    output_dir: str | Path,
    expected_compatibility: Mapping[str, Any],
    explicit_resume: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    auto_resume: bool = True,
) -> CheckpointAction:
    if explicit_resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if explicit_resume:
        path = Path(explicit_resume).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(
                f"Explicit resume checkpoint does not exist: {path}"
            )
        header = load_checkpoint_header(path)
        assert_compatible(header["compatibility"], expected_compatibility)
        return CheckpointAction("resume", path)
    if bool(auto_resume):
        root = Path(output_dir)
        incompatible: list[tuple[Path, str]] = []
        for name in ("last.pt", "latest.pt"):
            candidate = root / name
            if not candidate.is_file():
                continue
            header = load_checkpoint_header(candidate)
            try:
                assert_compatible(header["compatibility"], expected_compatibility)
            except CheckpointCompatibilityError as exc:
                incompatible.append((candidate.resolve(), str(exc)))
                continue
            return CheckpointAction("resume", candidate.resolve())
        if incompatible:
            details = "; ".join((f"{path}: {reason}" for path, reason in incompatible))
            raise CheckpointCompatibilityError(
                f"Auto-resume checkpoint candidates exist but none is compatible. Use a new output directory, or pass --no-auto-resume only when an intentional fresh/initialization overwrite is desired. {details}"
            )
    if init_checkpoint:
        path = Path(init_checkpoint).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {path}")
        header = load_checkpoint_header(path)
        assert_initialization_compatible(
            dict(header["compatibility"]), dict(expected_compatibility)
        )
        return CheckpointAction("initialize", path)
    return CheckpointAction("fresh", None)


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    ema: Any,
    expected_compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    value = _torch_load(Path(path), map_location="cpu")
    assert_compatible(value["compatibility"], expected_compatibility)
    model.load_state_dict(value["model"], strict=True)
    for name, target in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
        ("ema", ema),
    ):
        saved = value.get(name)
        if (target is None) != (saved is None):
            raise CheckpointCompatibilityError(
                f"Checkpoint/runtime {name} presence differs"
            )
        if target is not None:
            target.load_state_dict(saved)
    restore_rng_state(value["rng_state"])
    return value


def initialize_model_checkpoint(
    path: str | Path,
    *,
    model: Any,
    expected_compatibility: Mapping[str, Any],
    use_ema: bool = False,
) -> dict[str, Any]:
    value = _torch_load(Path(path), map_location="cpu")
    assert_initialization_compatible(value["compatibility"], expected_compatibility)
    model.load_state_dict(
        _initialization_model_state(value, use_ema=bool(use_ema)), strict=True
    )
    return value
