from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

THIS_FILE = Path(__file__).resolve()
AR_ROOT = THIS_FILE.parents[1]
if str(AR_ROOT) not in sys.path:
    sys.path.insert(0, str(AR_ROOT))
from streamrefine.data.path_resolver import resolve_recorded_path
from streamrefine.paths import resolve_path_string

READY_STATUSES = frozenset({"ready", "ok"})
TRAIN_SPLITS = frozenset({"train", "training"})
SHA256_RE = re.compile("^[0-9a-fA-F]{64}$")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to compute latent statistics. Activate the StreamRefine training environment first."
        ) from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a cache dictionary at {path}, got {type(value).__name__}"
        )
    return value


def _validate_cache(
    cache: dict[str, Any],
    expected_checkpoint_sha256: str | None,
    expected_config_sha256: str | None,
    expected_generation_contract_sha256: str | None,
) -> None:
    try:
        from streamrefine.tokenizer.cache_schema import validate_cache_dict
    except ImportError as exc:
        raise RuntimeError(
            "Could not import streamrefine.tokenizer.cache_schema from the ar checkout."
        ) from exc
    validate_cache_dict(
        cache,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_generation_contract_sha256=expected_generation_contract_sha256,
    )


def _normalise_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA256")
    return digest


def _merge_cache_sha256(
    expected: str | None, cache: dict[str, Any], *, key: str, cache_path: Path
) -> str:
    current = _normalise_sha256(cache.get(key), f"Cache {key} in {cache_path}")
    if expected is None:
        return current
    if current != expected:
        raise ValueError(
            f"{key} mismatch in {cache_path}: expected {expected}, got {current}"
        )
    return expected


def _normalise_split(value: Any) -> str:
    return str(value or "").strip().lower()


def _split_modality_values(values: Iterable[str] | str | None) -> list[str] | None:
    if values is None:
        return None
    items = [values] if isinstance(values, str) else list(values)
    result: list[str] = []
    for item in items:
        result.extend((part.strip() for part in str(item).split(",") if part.strip()))
    if not result:
        raise ValueError("--expected-modalities must contain at least one modality")
    return result


def _resolve_modality_coverage(
    datasets: Iterable[str],
    observed_modalities: Iterable[str],
    *,
    expected_modalities: Iterable[str] | str | None,
    allow_modality_subset: bool,
) -> dict[str, Any]:
    """Resolve one dataset's global training-modality coverage, failing closed."""
    from streamrefine.data.preprocess_medical import (
        DEFAULT_MODALITIES,
        canonical_modality_name,
        normalize_dataset_name,
    )

    raw_datasets = {str(value).strip() for value in datasets if str(value).strip()}
    if not raw_datasets:
        raise ValueError("Training caches do not declare a dataset")
    normalized_datasets = {normalize_dataset_name(value) for value in raw_datasets}
    if len(normalized_datasets) != 1:
        raise ValueError(
            f"Latent statistics must be computed per dataset; observed datasets: {sorted(normalized_datasets)}"
        )
    dataset = next(iter(normalized_datasets))
    canonical = tuple(
        (canonical_modality_name(value) for value in DEFAULT_MODALITIES[dataset])
    )
    canonical_set = set(canonical)
    observed_set = {
        canonical_modality_name(value)
        for value in observed_modalities
        if canonical_modality_name(value)
    }
    if not observed_set:
        raise ValueError("Training caches do not declare any modalities")
    unexpected_observed = sorted(observed_set.difference(canonical_set))
    if unexpected_observed:
        raise ValueError(
            f"Dataset {dataset!r} contains non-canonical observed modalities: {unexpected_observed}; canonical roster is {list(canonical)}"
        )
    requested = _split_modality_values(expected_modalities)
    if requested is None:
        if allow_modality_subset:
            raise ValueError(
                "--allow-modality-subset requires an explicit --expected-modalities roster"
            )
        expected_set = canonical_set
        policy = "canonical_dataset_full_roster_required"
    else:
        expected_set = {canonical_modality_name(value) for value in requested}
        expected_set.discard("")
        unexpected_expected = sorted(expected_set.difference(canonical_set))
        if unexpected_expected:
            raise ValueError(
                f"--expected-modalities contains non-canonical values for {dataset}: {unexpected_expected}; canonical roster is {list(canonical)}"
            )
        omitted_canonical = sorted(canonical_set.difference(expected_set))
        if omitted_canonical and (not allow_modality_subset):
            raise ValueError(
                f"A partial --expected-modalities roster is fail-closed by default; omitted canonical modalities for {dataset}: {omitted_canonical}. Pass --allow-modality-subset only for an intentional subset/ablation."
            )
        policy = (
            "explicit_modality_subset_opt_in"
            if omitted_canonical
            else "explicit_canonical_full_roster_required"
        )
    expected = [value for value in canonical if value in expected_set]
    observed = [value for value in canonical if value in observed_set]
    missing = [value for value in expected if value not in observed_set]
    if missing:
        raise ValueError(
            f"Training modality coverage incomplete for dataset {dataset!r}: expected={expected}, observed={observed}, missing={missing}. Use the complete cache_manifest_train.jsonl, or explicitly opt into a documented subset with --expected-modalities and --allow-modality-subset."
        )
    return {
        "dataset": dataset,
        "policy": policy,
        "allow_modality_subset": bool(allow_modality_subset),
        "subset_opt_in_active": policy == "explicit_modality_subset_opt_in",
        "canonical_training_modalities": list(canonical),
        "expected_modalities": expected,
        "observed_modalities": observed,
        "missing_expected_modalities": [],
        "coverage_complete": True,
    }


def iter_training_manifest_rows(
    manifest_path: Path,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield successful cache rows and reject ambiguous or non-training rows."""
    ready_count = 0
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on manifest line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"Manifest line {line_number} must be a JSON object")
            status = str(row.get("status", "")).strip().lower()
            if status not in READY_STATUSES:
                continue
            split = _normalise_split(row.get("split"))
            if split not in TRAIN_SPLITS:
                shown = split or "<missing>"
                raise ValueError(
                    f"Manifest line {line_number} has status={status!r} but split={shown!r}. compute_latent_stats.py accepts training-manifest records only."
                )
            ready_count += 1
            yield (line_number, row)
    if ready_count == 0:
        raise ValueError("Manifest contains no status=ready/ok training cache records")


def _cache_path_from_row(
    row: dict[str, Any],
    manifest_path: Path,
    line_number: int,
    *,
    path_remap: Any = None,
) -> Path:
    value = row.get("cache_path", row.get("path", row.get("file")))
    if not value:
        raise KeyError(f"Manifest line {line_number} has no cache_path/path/file field")
    path = resolve_recorded_path(
        resolve_path_string(str(value), base_dir=manifest_path.parent)
        if path_remap is None
        else str(value),
        remaps=() if path_remap is None else path_remap,
        base_dir=manifest_path.parent,
        must_exist=False,
        role=f"manifest line {line_number} cache",
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest line {line_number} cache does not exist: {path}; check the manifest path or explicit path_remap rules."
        )
    return path


class ChannelWelford:
    """Mergeable per-channel (optionally weighted) population moments."""

    def __init__(self, channels: int) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.weight_sum = [0.0] * self.channels
        self.mean = [0.0] * self.channels
        self.m2 = [0.0] * self.channels
        self.minimum = [math.inf] * self.channels
        self.maximum = [-math.inf] * self.channels
        self.finite_count = [0] * self.channels

    def update(self, channel: int, values: Any, weights: Any | None = None) -> None:
        import torch

        if not 0 <= channel < self.channels:
            raise IndexError(channel)
        x = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
        if weights is None:
            w = torch.ones_like(x)
        else:
            w = torch.as_tensor(weights, dtype=torch.float64, device="cpu").reshape(-1)
            if w.shape != x.shape:
                raise ValueError(
                    f"values/weights shape mismatch: {tuple(x.shape)} vs {tuple(w.shape)}"
                )
            if not bool(torch.isfinite(w).all().item()) or bool((w < 0).any().item()):
                raise ValueError("Weights must be finite and non-negative")
        positive = w > 0
        x = x[positive]
        w = w[positive]
        if x.numel() == 0:
            return
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError(f"Channel {channel} contains NaN/Inf in valid cells")
        batch_weight = float(w.sum().item())
        if batch_weight <= 0:
            return
        batch_mean = float((x * w).sum().item() / batch_weight)
        batch_m2 = float(((x - batch_mean) ** 2 * w).sum().item())
        old_weight = self.weight_sum[channel]
        total_weight = old_weight + batch_weight
        delta = batch_mean - self.mean[channel]
        if old_weight == 0:
            self.mean[channel] = batch_mean
            self.m2[channel] = batch_m2
        else:
            self.mean[channel] += delta * batch_weight / total_weight
            self.m2[channel] += (
                batch_m2 + delta * delta * old_weight * batch_weight / total_weight
            )
        self.weight_sum[channel] = total_weight
        self.minimum[channel] = min(self.minimum[channel], float(x.min().item()))
        self.maximum[channel] = max(self.maximum[channel], float(x.max().item()))
        self.finite_count[channel] += int(x.numel())

    def population_std(self) -> list[float]:
        result: list[float] = []
        for channel in range(self.channels):
            weight = self.weight_sum[channel]
            if weight <= 0:
                result.append(float("nan"))
            else:
                result.append(math.sqrt(max(0.0, self.m2[channel] / weight)))
        return result


class PriorityReservoir:
    """A bounded uniform reservoir implemented with independent random priorities."""

    def __init__(self, channels: int, capacity: int, seed: int) -> None:
        import torch

        if channels <= 0 or capacity <= 0:
            raise ValueError("channels and capacity must be positive")
        self.channels = int(channels)
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.values = [torch.empty(0, dtype=torch.float32) for _ in range(channels)]
        self.weights = [torch.empty(0, dtype=torch.float32) for _ in range(channels)]
        self.priorities = [torch.empty(0, dtype=torch.float64) for _ in range(channels)]
        self.seen = [0] * channels
        self.generators = []
        for channel in range(channels):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 1000003 * channel)
            self.generators.append(generator)

    def update(self, channel: int, values: Any, weights: Any | None = None) -> None:
        import torch

        x = torch.as_tensor(values, dtype=torch.float32, device="cpu").reshape(-1)
        if weights is None:
            w = torch.ones_like(x)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32, device="cpu").reshape(-1)
            if w.shape != x.shape:
                raise ValueError("values/weights shape mismatch in reservoir")
        keep = w > 0
        x = x[keep]
        w = w[keep]
        if x.numel() == 0:
            return
        if not bool(torch.isfinite(x).all().item()) or not bool(
            torch.isfinite(w).all().item()
        ):
            raise ValueError(f"Channel {channel} reservoir input contains NaN/Inf")
        priorities = torch.rand(
            x.numel(), dtype=torch.float64, generator=self.generators[channel]
        )
        all_values = torch.cat((self.values[channel], x))
        all_weights = torch.cat((self.weights[channel], w))
        all_priorities = torch.cat((self.priorities[channel], priorities))
        if all_values.numel() > self.capacity:
            selected = torch.topk(
                all_priorities, k=self.capacity, largest=True, sorted=False
            ).indices
            all_values = all_values[selected]
            all_weights = all_weights[selected]
            all_priorities = all_priorities[selected]
        self.values[channel] = all_values
        self.weights[channel] = all_weights
        self.priorities[channel] = all_priorities
        self.seen[channel] += int(x.numel())

    def quantile(self, channel: int, q: float, *, weighted: bool) -> float:
        import torch

        if not 0.0 <= q <= 1.0:
            raise ValueError(q)
        values = self.values[channel]
        if values.numel() == 0:
            return float("nan")
        if not weighted:
            return float(torch.quantile(values, q).item())
        order = torch.argsort(values)
        sorted_values = values[order]
        sorted_weights = self.weights[channel][order].to(torch.float64)
        cumulative = torch.cumsum(sorted_weights, dim=0)
        total = cumulative[-1]
        threshold = total * float(q)
        index = int(torch.searchsorted(cumulative, threshold, right=False).item())
        index = min(max(index, 0), int(sorted_values.numel()) - 1)
        return float(sorted_values[index].item())


def _atomic_write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and (not overwrite):
        raise FileExistsError(f"Output exists (pass --overwrite to replace it): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def compute_latent_stats(
    manifest_path: Path,
    *,
    mask_mode: str,
    reservoir_size: int,
    seed: int,
    std_epsilon: float,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_generation_contract_sha256: str | None = None,
    expected_modalities: Iterable[str] | str | None = None,
    allow_modality_subset: bool = False,
    path_remap: Any = None,
    progress_callback: Callable[[int, Path], None] | None = None,
) -> dict[str, Any]:
    """Compute shared statistics; optional path_remap is the combined runtime path map."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to compute latent statistics. Activate the StreamRefine training environment first."
        ) from exc
    if path_remap is None:
        manifest_path = (
            Path(resolve_path_string(manifest_path, base_dir=Path.cwd()))
            .expanduser()
            .resolve()
        )
    else:
        manifest_path = resolve_recorded_path(
            manifest_path,
            remaps=path_remap,
            base_dir=Path.cwd(),
            must_exist=False,
            role="training modality-cache manifest",
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if mask_mode not in {"hard", "soft"}:
        raise ValueError("mask_mode must be hard or soft")
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")
    if std_epsilon <= 0:
        raise ValueError("std_epsilon must be positive")
    accumulator: ChannelWelford | None = None
    reservoir: PriorityReservoir | None = None
    checkpoint_hash = (
        _normalise_sha256(expected_checkpoint_sha256, "--expected-checkpoint-sha256")
        if expected_checkpoint_sha256
        else None
    )
    config_hash = (
        _normalise_sha256(expected_config_sha256, "--expected-config-sha256")
        if expected_config_sha256
        else None
    )
    generation_hash = (
        _normalise_sha256(
            expected_generation_contract_sha256, "--expected-generation-contract-sha256"
        )
        if expected_generation_contract_sha256
        else None
    )
    cache_version: str | None = None
    datasets: set[str] = set()
    modalities: set[str] = set()
    files_used = 0
    for line_number, row in iter_training_manifest_rows(manifest_path):
        path_options = {} if path_remap is None else {"path_remap": path_remap}
        cache_path = _cache_path_from_row(
            row, manifest_path, line_number, **path_options
        )
        cache = _torch_load(cache_path)
        _validate_cache(cache, checkpoint_hash, config_hash, generation_hash)
        cache_split = _normalise_split(cache.get("split"))
        if cache_split not in TRAIN_SPLITS:
            shown = cache_split or "<missing>"
            raise ValueError(
                f"Cache {cache_path} has split={shown!r}; training statistics require both the manifest row and cache payload to declare train/training"
            )
        checkpoint_hash = _merge_cache_sha256(
            checkpoint_hash,
            cache,
            key="tokenizer_checkpoint_sha256",
            cache_path=cache_path,
        )
        config_hash = _merge_cache_sha256(
            config_hash, cache, key="tokenizer_config_sha256", cache_path=cache_path
        )
        generation_hash = _merge_cache_sha256(
            generation_hash,
            cache,
            key="generation_contract_sha256",
            cache_path=cache_path,
        )
        current_version = str(cache.get("cache_version", ""))
        if cache_version is None:
            cache_version = current_version
        elif current_version != cache_version:
            raise ValueError(
                f"Mixed cache versions are not allowed: expected {cache_version}, got {current_version}"
            )
        latent = cache["latent_mu"]
        hard_mask = cache["valid_mask_latent"].bool()
        soft_mask = cache["valid_mask_latent_soft"].float()
        if latent.ndim != 4 or int(latent.shape[0]) != 16:
            raise ValueError(
                f"Expected latent_mu [16,D,H,W], got {tuple(latent.shape)} in {cache_path}"
            )
        if tuple(hard_mask.shape) != tuple(latent.shape[1:]) or tuple(
            soft_mask.shape
        ) != tuple(latent.shape[1:]):
            raise ValueError(f"Latent/mask shape mismatch in {cache_path}")
        if not bool(torch.isfinite(latent).all().item()):
            raise ValueError(f"latent_mu contains NaN/Inf: {cache_path}")
        if not bool(torch.isfinite(soft_mask).all().item()):
            raise ValueError(f"valid_mask_latent_soft contains NaN/Inf: {cache_path}")
        if bool(((soft_mask < 0) | (soft_mask > 1)).any().item()):
            raise ValueError(f"valid_mask_latent_soft must lie in [0,1]: {cache_path}")
        if accumulator is None:
            accumulator = ChannelWelford(channels=16)
            reservoir = PriorityReservoir(
                channels=16, capacity=reservoir_size, seed=seed
            )
        assert reservoir is not None
        if mask_mode == "hard":
            selected = hard_mask
            selected_weights = None
        else:
            selected = soft_mask > 0
            selected_weights = soft_mask[selected]
        if not bool(selected.any().item()):
            raise ValueError(
                f"Cache has no valid latent cells under mask_mode={mask_mode}: {cache_path}"
            )
        for channel in range(16):
            values = latent[channel][selected].float()
            accumulator.update(channel, values, selected_weights)
            reservoir.update(channel, values, selected_weights)
        datasets.add(str(cache.get("dataset", row.get("dataset", ""))))
        modalities.add(str(cache.get("modality", row.get("modality", ""))))
        files_used += 1
        if progress_callback is not None:
            progress_callback(files_used, cache_path)
    if (
        accumulator is None
        or reservoir is None
        or checkpoint_hash is None
        or (config_hash is None)
        or (generation_hash is None)
    ):
        raise RuntimeError("No cache values were accumulated")
    modality_coverage = _resolve_modality_coverage(
        datasets,
        modalities,
        expected_modalities=expected_modalities,
        allow_modality_subset=allow_modality_subset,
    )
    std = accumulator.population_std()
    collapsed = [
        channel
        for channel, value in enumerate(std)
        if not math.isfinite(value) or value <= std_epsilon
    ]
    if collapsed:
        raise ValueError(
            f"Collapsed/non-finite latent channels (std <= {std_epsilon:g}): {collapsed}; statistics were not written"
        )
    count: list[int | float]
    if mask_mode == "hard":
        count = [int(round(value)) for value in accumulator.weight_sum]
        count_kind = "valid_cells"
    else:
        count = [float(value) for value in accumulator.weight_sum]
        count_kind = "soft_valid_weight_sum"
    weighted_percentiles = mask_mode == "soft"
    result = {
        "cache_version": cache_version,
        "tokenizer_checkpoint_sha256": checkpoint_hash,
        "tokenizer_config_sha256": config_hash,
        "generation_contract_sha256": generation_hash,
        "channels": 16,
        "split": "train",
        "policy": "shared_across_training_modalities",
        "mask_mode": mask_mode,
        "count_kind": count_kind,
        "count": count,
        "finite_count": [int(value) for value in accumulator.finite_count],
        "mean": [float(value) for value in accumulator.mean],
        "std": [float(value) for value in std],
        "min": [float(value) for value in accumulator.minimum],
        "max": [float(value) for value in accumulator.maximum],
        "p01": [
            reservoir.quantile(channel, 0.01, weighted=weighted_percentiles)
            for channel in range(16)
        ],
        "p50": [
            reservoir.quantile(channel, 0.5, weighted=weighted_percentiles)
            for channel in range(16)
        ],
        "p99": [
            reservoir.quantile(channel, 0.99, weighted=weighted_percentiles)
            for channel in range(16)
        ],
        "population_std": True,
        "std_epsilon": float(std_epsilon),
        "files_used": files_used,
        "dataset": modality_coverage["dataset"],
        "datasets": [modality_coverage["dataset"]],
        "modalities": modality_coverage["observed_modalities"],
        "expected_modalities": modality_coverage["expected_modalities"],
        "observed_modalities": modality_coverage["observed_modalities"],
        "modality_coverage_policy": modality_coverage["policy"],
        "modality_coverage": modality_coverage,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "accepted_statuses": sorted(READY_STATUSES),
        "reservoir_sampling": {
            "strategy": "bounded_uniform_random_priority_reservoir",
            "capacity_per_channel": int(reservoir_size),
            "seed": int(seed),
            "observations_seen_per_channel": [int(value) for value in reservoir.seen],
            "sample_size_per_channel": [
                int(value.numel()) for value in reservoir.values
            ],
            "soft_percentiles": "weighted_quantile_over_uniform_reservoir"
            if weighted_percentiles
            else None,
        },
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute shared per-channel VidTok-KL latent statistics from status=ready/ok rows in cache_manifest_train.jsonl only."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Training cache manifest JSONL."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Atomic JSON output path."
    )
    parser.add_argument(
        "--mask-mode",
        choices=("hard", "soft"),
        default="hard",
        help="Use hard valid cells (default) or soft occupancy weights.",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=65536,
        help="Maximum percentile samples per channel.",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Reservoir sampling seed."
    )
    parser.add_argument(
        "--std-epsilon",
        type=float,
        default=1e-06,
        help="Reject channels at or below this std.",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        "--expected_checkpoint_sha256",
        dest="expected_checkpoint_sha256",
        default=None,
    )
    parser.add_argument(
        "--expected-config-sha256",
        "--expected_config_sha256",
        dest="expected_config_sha256",
        default=None,
    )
    parser.add_argument(
        "--expected-generation-contract-sha256",
        "--expected_generation_contract_sha256",
        dest="expected_generation_contract_sha256",
        default=None,
    )
    parser.add_argument(
        "--expected-modalities",
        "--expected_modalities",
        dest="expected_modalities",
        nargs="+",
        default=None,
        metavar="MODALITY",
        help="Expected global training-modality roster (space- or comma-separated). Defaults to the dataset's full canonical roster.",
    )
    parser.add_argument(
        "--allow-modality-subset",
        "--allow_modality_subset",
        dest="allow_modality_subset",
        action="store_true",
        help="Explicitly allow --expected-modalities to omit canonical modalities; intended only for a documented subset/ablation.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing statistics JSON."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path = Path(resolve_path_string(args.output, base_dir=Path.cwd()))
        result = compute_latent_stats(
            args.manifest,
            mask_mode=args.mask_mode,
            reservoir_size=args.reservoir_size,
            seed=args.seed,
            std_epsilon=args.std_epsilon,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_config_sha256=args.expected_config_sha256,
            expected_generation_contract_sha256=args.expected_generation_contract_sha256,
            expected_modalities=args.expected_modalities,
            allow_modality_subset=args.allow_modality_subset,
        )
        _atomic_write_json(output_path, result, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {result['channels']}-channel training latent statistics from {result['files_used']} caches to {output_path.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
