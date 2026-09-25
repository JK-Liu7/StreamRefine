from __future__ import annotations
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, default_collate, get_worker_info
from .latent_window_sampler import (
    DEFAULT_WINDOW_DHW,
    crop_aligned_pair,
    find_best_valid_window,
    sample_valid_latent_window,
)
from .preprocess_medical import (
    DEFAULT_MODALITIES,
    canonical_modality_name,
    normalize_dataset_name,
)

TRANSLATION_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "brats24": (
        ("t2w", "t2f"),
        ("t1c", "t1n"),
        ("t2f", "t2w"),
        ("t1n", "t2w"),
        ("t1n", "t1c"),
    ),
    "synthrad": (("mr", "ct"), ("cbct", "ct")),
    "autopet": (("ct", "pet"),),
}
_SUCCESS_STATUSES = {
    "",
    "ok",
    "ready",
    "complete",
    "completed",
    "cached",
    "written",
    "reused",
    "success",
    "valid",
}
_DATASET_SHARED_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "brats24": ("cohort",),
    "synthrad": ("task", "anatomy"),
    "autopet": ("tracer",),
}
_PRESERVE_AS_LIST_FIELDS = frozenset(
    {
        "padding_info",
        "normalization_info",
        "source_normalization_info",
        "target_normalization_info",
    }
)
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
_FULL_MODALITY_COVERAGE_POLICIES = frozenset(
    {
        "canonical_dataset_full_roster_required",
        "explicit_canonical_full_roster_required",
    }
)
_SUBSET_MODALITY_COVERAGE_POLICY = "explicit_modality_subset_opt_in"


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _shape_tuple(value: Any, name: str) -> tuple[int, int, int]:
    plain = _plain(value)
    if not isinstance(plain, (list, tuple)) or len(plain) != 3:
        raise ValueError(f"{name} must contain D,H,W or H,W,D, got {plain!r}")
    result = tuple((int(item) for item in plain))
    if any((item <= 0 for item in result)):
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _canonical_split(value: Any, *, name: str, required: bool) -> str:
    text = str(value or "").strip().lower()
    if not text:
        if required:
            raise ValueError(f"{name} must explicitly declare train or val")
        return ""
    if text not in _SPLIT_ALIASES:
        raise ValueError(f"{name} must be train or val, got {value!r}")
    return _SPLIT_ALIASES[text]


def _canonical_json_value(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_path_text(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _canonical_modality_mapping(
    value: Any, *, name: str, paths: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a modality-keyed mapping")
    result: dict[str, Any] = {}
    for raw_modality, raw_value in value.items():
        modality = canonical_modality_name(str(raw_modality))
        if not modality or modality in result:
            raise ValueError(
                f"{name} contains an empty or duplicate canonical modality"
            )
        if paths:
            result[modality] = _canonical_path_text(raw_value)
        else:
            signature = _canonical_json_value(raw_value)
            if isinstance(signature, Mapping) and "resolved_path" in signature:
                signature = dict(signature)
                signature["resolved_path"] = _canonical_path_text(
                    signature["resolved_path"]
                )
            result[modality] = signature
    return {key: result[key] for key in sorted(result)}


def _canonical_input_snapshot(cache: Mapping[str, Any]) -> dict[str, Any]:
    available_value = cache.get("available_modalities", ())
    if isinstance(available_value, (str, bytes)) or not isinstance(
        available_value, Sequence
    ):
        raise TypeError("available_modalities must be a sequence")
    available = [canonical_modality_name(str(item)) for item in available_value]
    if any((not item for item in available)) or len(set(available)) != len(available):
        raise ValueError(
            "available_modalities contains an empty or duplicate canonical modality"
        )
    return {
        "available_modalities": sorted(available),
        "modality_image_paths": _canonical_modality_mapping(
            cache.get("modality_image_paths", {}),
            name="modality_image_paths",
            paths=True,
        ),
        "input_file_signatures": _canonical_modality_mapping(
            cache.get("input_file_signatures", {}),
            name="input_file_signatures",
            paths=False,
        ),
        "crop_source_modality": canonical_modality_name(
            cache.get("crop_source_modality")
        ),
    }


def _input_snapshot_sha256(cache: Mapping[str, Any]) -> str:
    from streamrefine.tokenizer.cache_schema import input_snapshot_sha256

    digest = input_snapshot_sha256(cache)
    declared = str(cache.get("input_snapshot_sha256", "")).strip().lower()
    if declared:
        if len(declared) != 64 or any(
            (character not in "0123456789abcdef" for character in declared)
        ):
            raise ValueError(
                "input_snapshot_sha256 must be a 64-character hexadecimal SHA256"
            )
        if declared != digest:
            raise ValueError(
                f"input_snapshot_sha256 does not match canonical input snapshot: stored={declared}, computed={digest}"
            )
    return digest


def _canonical_pairs(
    dataset: str,
    translation_pairs: Mapping[str, Sequence[Sequence[str]]]
    | Sequence[Sequence[str]]
    | None,
) -> tuple[tuple[str, str], ...]:
    dataset_key = normalize_dataset_name(dataset)
    values: Sequence[Sequence[str]]
    if translation_pairs is None:
        values = TRANSLATION_PAIRS[dataset_key]
    elif isinstance(translation_pairs, Mapping):
        selected = None
        for key, candidate in translation_pairs.items():
            if normalize_dataset_name(str(key)) == dataset_key:
                selected = candidate
                break
        if selected is None:
            raise KeyError(
                f"translation_pairs has no entry for dataset={dataset_key!r}"
            )
        values = selected
    else:
        values = translation_pairs
    result: list[tuple[str, str]] = []
    for pair in values:
        if len(pair) != 2:
            raise ValueError(
                f"Each translation pair must contain source,target, got {pair!r}"
            )
        source, target = (canonical_modality_name(item) for item in pair)
        if not source or not target or source == target:
            raise ValueError(f"Invalid translation pair: {pair!r}")
        result.append((source, target))
    return tuple(dict.fromkeys(result))


def translation_pairs_for_dataset(
    dataset: str,
    translation_pairs: Mapping[str, Sequence[Sequence[str]]]
    | Sequence[Sequence[str]]
    | None = None,
) -> tuple[tuple[str, str], ...]:
    return _canonical_pairs(dataset, translation_pairs)


def _iter_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(dict(value))
        return rows
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("pairs", "data", "records", "caches"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(
            (isinstance(item, dict) for item in value)
        ):
            raise ValueError(f"Unsupported pair manifest format: {path}")
        return [dict(item) for item in value]
    raise ValueError(f"Pair manifest must be .jsonl or .json, got {path}")


def _resolve_manifest_path(value: Any, base_dir: Path) -> Path:
    if not value:
        raise ValueError("Manifest cache path is empty")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _path_from_row(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key):
            return row[key]
    return None


def _explicit_pair_row(row: Mapping[str, Any], base_dir: Path) -> dict[str, Any] | None:
    source_value = _path_from_row(
        row, "source_cache_path", "source_cache", "source_path"
    )
    target_value = _path_from_row(
        row, "target_cache_path", "target_cache", "target_path"
    )
    if source_value is None and target_value is None:
        return None
    if source_value is None or target_value is None:
        raise ValueError(
            f"Pair manifest row must contain both source and target cache paths: {row!r}"
        )
    source_path = _resolve_manifest_path(source_value, base_dir)
    target_path = _resolve_manifest_path(target_value, base_dir)
    source_modality = canonical_modality_name(row.get("source_modality"))
    target_modality = canonical_modality_name(row.get("target_modality"))
    case_id = str(row.get("case_id", "")).strip()
    pair_id = str(row.get("pair_id", "")).strip()
    if not pair_id:
        pair_id = f"{case_id or source_path.parent.name}_{source_modality}_to_{target_modality}"
    return {
        **dict(row),
        "source_cache_path": str(source_path),
        "target_cache_path": str(target_path),
        "source_modality": source_modality,
        "target_modality": target_modality,
        "pair_id": pair_id,
    }


def _cache_group_key(row: Mapping[str, Any], dataset_key: str) -> tuple[str, ...]:
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        raise ValueError(f"Modality cache manifest row has no case_id: {row!r}")
    group_id = str(row.get("group_id", "")).strip()
    if group_id:
        return (dataset_key, group_id)
    if dataset_key == "brats24":
        return (dataset_key, str(row.get("cohort", "")), case_id)
    if dataset_key == "synthrad":
        return (
            dataset_key,
            str(row.get("task", "")),
            str(row.get("anatomy", "")),
            case_id,
        )
    return (dataset_key, case_id)


def load_pair_manifest(
    manifest: Path | str | Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    translation_pairs: Mapping[str, Sequence[Sequence[str]]]
    | Sequence[Sequence[str]]
    | None = None,
    expected_split: str | None = None,
    require_files: bool = True,
) -> list[dict[str, Any]]:
    """Load explicit pair rows or derive pair rows from a modality cache manifest.

    Relative cache paths are resolved against the manifest directory. Derivation
    never fabricates endpoints: a pair is emitted only when both modality cache
    files are represented and, by default, exist on disk.
    """
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Pair/cache manifest does not exist: {manifest_path}"
            )
        rows = _iter_manifest_rows(manifest_path)
        base_dir = manifest_path.parent
    else:
        rows = [dict(row) for row in manifest]
        base_dir = Path.cwd()
    requested_dataset = normalize_dataset_name(dataset) if dataset else None
    requested_split = (
        _canonical_split(expected_split, name="expected_split", required=True)
        if expected_split is not None
        else None
    )
    explicit: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status not in _SUCCESS_STATUSES:
            continue
        row_split = _canonical_split(
            row.get("split"),
            name=f"Manifest row {row.get('pair_id') or row.get('case_id') or '<unknown>'!r} split",
            required=requested_split is not None,
        )
        if requested_split is not None and row_split != requested_split:
            continue
        row = dict(row)
        if row_split:
            row["split"] = row_split
        pair = _explicit_pair_row(row, base_dir)
        if pair is not None:
            row_dataset = pair.get("dataset")
            if not row_dataset and requested_dataset is None:
                raise ValueError(
                    f"Explicit pair {pair['pair_id']!r} must declare dataset when dataset= is not supplied"
                )
            row_dataset_key = normalize_dataset_name(
                str(row_dataset or requested_dataset)
            )
            if requested_dataset and row_dataset_key != requested_dataset:
                continue
            direction = (str(pair["source_modality"]), str(pair["target_modality"]))
            allowed_directions = _canonical_pairs(row_dataset_key, translation_pairs)
            if direction not in allowed_directions:
                raise ValueError(
                    f"Explicit pair {pair['pair_id']!r} direction {direction[0]}->{direction[1]} is not allowed for dataset={row_dataset_key!r}; allowed={allowed_directions}"
                )
            pair["dataset_key"] = row_dataset_key
            if require_files:
                for field in ("source_cache_path", "target_cache_path"):
                    if not Path(pair[field]).is_file():
                        raise FileNotFoundError(
                            f"Manifest pair {pair['pair_id']!r} references missing {field}: {pair[field]}"
                        )
            explicit.append(pair)
            continue
        cache_value = _path_from_row(row, "cache_path", "path", "file")
        if cache_value is None:
            continue
        row = dict(row)
        row["cache_path"] = str(_resolve_manifest_path(cache_value, base_dir))
        row["modality"] = canonical_modality_name(row.get("modality"))
        if not row["modality"]:
            raise ValueError(f"Modality cache row has no modality: {row!r}")
        if not row.get("dataset") and (not requested_dataset):
            raise ValueError(
                "Modality cache rows must contain dataset when dataset= is not supplied"
            )
        row_dataset = normalize_dataset_name(
            str(row.get("dataset") or requested_dataset)
        )
        if requested_dataset and row_dataset != requested_dataset:
            continue
        row["dataset_key"] = row_dataset
        modality_rows.append(row)
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in modality_rows:
        dataset_key = str(row["dataset_key"])
        key = _cache_group_key(row, dataset_key)
        modality = str(row["modality"])
        previous = grouped[key].get(modality)
        if previous and previous["cache_path"] != row["cache_path"]:
            raise ValueError(f"Conflicting {modality} cache paths for group {key}")
        grouped[key][modality] = row
    derived: list[dict[str, Any]] = []
    for key in sorted(grouped):
        by_modality = grouped[key]
        dataset_key = key[0]
        for source, target in _canonical_pairs(dataset_key, translation_pairs):
            if source not in by_modality or target not in by_modality:
                continue
            source_row, target_row = (by_modality[source], by_modality[target])
            source_path = Path(source_row["cache_path"])
            target_path = Path(target_row["cache_path"])
            if require_files and (
                not source_path.is_file() or not target_path.is_file()
            ):
                continue
            case_id = str(
                source_row.get("case_id")
                or target_row.get("case_id")
                or source_path.parent.name
            )
            derived.append(
                {
                    "source_cache_path": str(source_path),
                    "target_cache_path": str(target_path),
                    "case_id": case_id,
                    "group_id": str(
                        source_row.get("group_id") or target_row.get("group_id", "")
                    ),
                    "dataset": str(
                        source_row.get("dataset")
                        or target_row.get("dataset")
                        or dataset_key
                    ),
                    "dataset_key": dataset_key,
                    "split": str(
                        source_row.get("split") or target_row.get("split", "")
                    ),
                    "source_modality": source,
                    "target_modality": target,
                    "pair_id": f"{case_id}_{source}_to_{target}",
                    "cohort": str(
                        source_row.get("cohort") or target_row.get("cohort", "")
                    ),
                    "task": str(source_row.get("task") or target_row.get("task", "")),
                    "anatomy": str(
                        source_row.get("anatomy") or target_row.get("anatomy", "")
                    ),
                    "tracer": str(
                        source_row.get("tracer") or target_row.get("tracer", "")
                    ),
                    **{
                        field: source_row[field]
                        for field in (
                            "available_modalities",
                            "modality_image_paths",
                            "input_file_signatures",
                            "crop_source_modality",
                            "input_snapshot_sha256",
                        )
                        if field in source_row
                    },
                }
            )
    result = [*explicit, *derived]
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in result:
        key = (
            str(row["source_cache_path"]),
            str(row["target_cache_path"]),
            str(row["pair_id"]),
        )
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def _allclose_metadata(
    left: Any, right: Any, name: str, *, atol: float = 1e-05
) -> None:
    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64)
    if tuple(left_tensor.shape) != tuple(right_tensor.shape) or not torch.allclose(
        left_tensor, right_tensor, atol=atol, rtol=1e-05
    ):
        raise ValueError(
            f"Source/target {name} differ: {_plain(left)!r} vs {_plain(right)!r}"
        )


def validate_shared_pair_metadata(
    source_cache: Mapping[str, Any], target_cache: Mapping[str, Any]
) -> None:
    """Enforce the case-shared cache contract after individual schema checks."""
    for field in ("original_shape_hwd", "padded_shape_hwd", "latent_shape_dhw"):
        left = _shape_tuple(source_cache.get(field), field)
        right = _shape_tuple(target_cache.get(field), field)
        if left != right:
            raise ValueError(f"Source/target {field} differ: {left} vs {right}")
    for field in ("spacing", "affine"):
        _allclose_metadata(
            source_cache.get(field), target_cache.get(field), field, atol=0.001
        )
    if _plain(source_cache.get("padding_info")) != _plain(
        target_cache.get("padding_info")
    ):
        raise ValueError(
            "Source/target padding_info differ; caches are not on a shared padded grid"
        )
    source_hard = torch.as_tensor(source_cache.get("valid_mask_latent")).bool()
    target_hard = torch.as_tensor(target_cache.get("valid_mask_latent")).bool()
    if tuple(source_hard.shape) != tuple(target_hard.shape) or not torch.equal(
        source_hard, target_hard
    ):
        raise ValueError(
            "Source/target valid_mask_latent differ; paired PIR requires one shared hard valid-edge support"
        )
    source_anatomy = torch.as_tensor(
        source_cache.get("anatomy_mask_latent_hard")
    ).bool()
    target_anatomy = torch.as_tensor(
        target_cache.get("anatomy_mask_latent_hard")
    ).bool()
    if tuple(source_anatomy.shape) != tuple(target_anatomy.shape) or not torch.equal(
        source_anatomy, target_anatomy
    ):
        raise ValueError(
            "Source/target anatomy_mask_latent_hard differ; paired PIR requires one deterministic case-shared anatomy support"
        )
    for field in (
        "anatomy_mask_contract",
        "anatomy_mask_contract_fingerprint",
        "anatomy_mask_sha256",
        "anatomy_mask_statistics",
    ):
        if _plain(source_cache.get(field)) != _plain(target_cache.get(field)):
            raise ValueError(f"Source/target {field} differ")
    source_hash = str(source_cache.get("tokenizer_checkpoint_sha256", "")).lower()
    target_hash = str(target_cache.get("tokenizer_checkpoint_sha256", "")).lower()
    if source_hash != target_hash:
        raise ValueError(
            f"Source/target tokenizer checkpoint hashes differ: {source_hash} vs {target_hash}"
        )
    source_config_hash = str(source_cache.get("tokenizer_config_sha256", "")).lower()
    target_config_hash = str(target_cache.get("tokenizer_config_sha256", "")).lower()
    if source_config_hash != target_config_hash:
        raise ValueError(
            f"Source/target tokenizer config hashes differ: {source_config_hash or '<missing>'} vs {target_config_hash or '<missing>'}"
        )
    source_generation_hash = (
        str(source_cache.get("generation_contract_sha256", "")).strip().lower()
    )
    target_generation_hash = (
        str(target_cache.get("generation_contract_sha256", "")).strip().lower()
    )
    if source_generation_hash != target_generation_hash:
        raise ValueError(
            f"Source/target generation contract hashes differ: {source_generation_hash or '<missing>'} vs {target_generation_hash or '<missing>'}"
        )
    source_dataset = normalize_dataset_name(str(source_cache.get("dataset")))
    target_dataset = normalize_dataset_name(str(target_cache.get("dataset")))
    if source_dataset != target_dataset:
        raise ValueError("Source/target dataset metadata differ")
    if str(source_cache.get("case_id")) != str(target_cache.get("case_id")):
        raise ValueError(
            f"Source/target case_id differ: {source_cache.get('case_id')!r} vs {target_cache.get('case_id')!r}"
        )
    for field in (
        "group_id",
        "split",
        *_DATASET_SHARED_METADATA_FIELDS[source_dataset],
    ):
        source_value = str(source_cache.get(field, "")).strip()
        target_value = str(target_cache.get(field, "")).strip()
        if source_value != target_value:
            raise ValueError(
                f"Source/target {field} differ: {source_value!r} vs {target_value!r}"
            )
    source_snapshot = _canonical_input_snapshot(source_cache)
    target_snapshot = _canonical_input_snapshot(target_cache)
    for field in (
        "available_modalities",
        "modality_image_paths",
        "input_file_signatures",
        "crop_source_modality",
    ):
        if source_snapshot[field] != target_snapshot[field]:
            raise ValueError(f"Source/target {field} differ")
    source_snapshot_hash = _input_snapshot_sha256(source_cache)
    target_snapshot_hash = _input_snapshot_sha256(target_cache)
    if source_snapshot_hash != target_snapshot_hash:
        raise ValueError(
            f"Source/target input snapshot hashes differ: {source_snapshot_hash} vs {target_snapshot_hash}"
        )
    source_hard = torch.as_tensor(source_cache["valid_mask_latent"]).bool()
    target_hard = torch.as_tensor(target_cache["valid_mask_latent"]).bool()
    if tuple(source_hard.shape) != tuple(target_hard.shape) or not torch.equal(
        source_hard, target_hard
    ):
        raise ValueError(
            "Source/target valid_mask_latent differ; masks must be case-shared"
        )
    source_soft = torch.as_tensor(source_cache["valid_mask_latent_soft"]).float()
    target_soft = torch.as_tensor(target_cache["valid_mask_latent_soft"]).float()
    if tuple(source_soft.shape) != tuple(target_soft.shape) or not torch.allclose(
        source_soft, target_soft, atol=0.001, rtol=0.0
    ):
        raise ValueError(
            "Source/target valid_mask_latent_soft differ; masks must be case-shared"
        )


def _load_and_validate_cache(
    path: Path | str,
    expected_checkpoint_sha256: str | None,
    expected_config_sha256: str | None,
    expected_generation_contract_sha256: str | None,
) -> dict[str, Any]:
    from streamrefine.tokenizer.cache_schema import load_and_validate_cache

    return load_and_validate_cache(
        path,
        map_location="cpu",
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_generation_contract_sha256=expected_generation_contract_sha256,
    )


def _canonical_stats_modalities(
    value: Any, *, name: str, canonical_roster: Sequence[str]
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or (not value)
    ):
        raise ValueError(f"Latent stats {name} must be a non-empty modality list")
    raw = tuple((str(item).strip() for item in value))
    canonical = tuple((canonical_modality_name(item) for item in raw))
    if any((not item for item in canonical)) or raw != canonical:
        raise ValueError(
            f"Latent stats {name} must use canonical modality names, got {list(raw)!r}"
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Latent stats {name} contains duplicate modalities")
    roster = tuple(canonical_roster)
    unexpected = sorted(set(canonical).difference(roster))
    if unexpected:
        raise ValueError(
            f"Latent stats {name} contains modalities outside the dataset roster: {unexpected}"
        )
    canonical_order = tuple((item for item in roster if item in set(canonical)))
    if canonical != canonical_order:
        raise ValueError(
            f"Latent stats {name} must follow canonical roster order {list(roster)}, got {list(canonical)}"
        )
    return canonical


def _validate_stats_modality_coverage(stats: dict[str, Any]) -> None:
    dataset_raw = str(stats.get("dataset", "")).strip()
    if not dataset_raw:
        raise ValueError("Latent stats must contain dataset")
    dataset_key = normalize_dataset_name(dataset_raw)
    if dataset_raw != dataset_key:
        raise ValueError(
            f"Latent stats dataset must be canonical {dataset_key!r}, got {dataset_raw!r}"
        )
    canonical_roster = tuple(
        (canonical_modality_name(item) for item in DEFAULT_MODALITIES[dataset_key])
    )
    expected = _canonical_stats_modalities(
        stats.get("expected_modalities"),
        name="expected_modalities",
        canonical_roster=canonical_roster,
    )
    observed = _canonical_stats_modalities(
        stats.get("observed_modalities"),
        name="observed_modalities",
        canonical_roster=canonical_roster,
    )
    policy = str(stats.get("modality_coverage_policy", "")).strip()
    allowed_policies = {
        *_FULL_MODALITY_COVERAGE_POLICIES,
        _SUBSET_MODALITY_COVERAGE_POLICY,
    }
    if policy not in allowed_policies:
        raise ValueError(
            f"Latent stats modality_coverage_policy is missing or unsupported: {stats.get('modality_coverage_policy')!r}"
        )
    coverage = stats.get("modality_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("Latent stats modality_coverage must be present as a mapping")
    required_coverage_fields = {
        "dataset",
        "policy",
        "allow_modality_subset",
        "subset_opt_in_active",
        "canonical_training_modalities",
        "expected_modalities",
        "observed_modalities",
        "missing_expected_modalities",
        "coverage_complete",
    }
    missing = sorted(required_coverage_fields.difference(coverage))
    if missing:
        raise ValueError(f"Latent stats modality_coverage is missing fields: {missing}")
    coverage_dataset_raw = str(coverage["dataset"]).strip()
    coverage_dataset = normalize_dataset_name(coverage_dataset_raw)
    if coverage_dataset_raw != coverage_dataset or coverage_dataset != dataset_key:
        raise ValueError(
            "Latent stats modality_coverage.dataset must equal canonical stats.dataset"
        )
    if str(coverage["policy"]).strip() != policy:
        raise ValueError(
            "Latent stats modality_coverage.policy must match modality_coverage_policy"
        )
    coverage_roster = _canonical_stats_modalities(
        coverage["canonical_training_modalities"],
        name="modality_coverage.canonical_training_modalities",
        canonical_roster=canonical_roster,
    )
    coverage_expected = _canonical_stats_modalities(
        coverage["expected_modalities"],
        name="modality_coverage.expected_modalities",
        canonical_roster=canonical_roster,
    )
    coverage_observed = _canonical_stats_modalities(
        coverage["observed_modalities"],
        name="modality_coverage.observed_modalities",
        canonical_roster=canonical_roster,
    )
    if coverage_roster != canonical_roster:
        raise ValueError(
            "Latent stats canonical_training_modalities must equal the dataset full roster"
        )
    if coverage_expected != expected or coverage_observed != observed:
        raise ValueError(
            "Latent stats top-level modality rosters must match modality_coverage"
        )
    if (
        coverage["coverage_complete"] is not True
        or coverage["missing_expected_modalities"] != []
    ):
        raise ValueError(
            "Latent stats modality coverage must be complete with no missing modalities"
        )
    if not set(expected).issubset(observed):
        raise ValueError(
            "Latent stats expected_modalities must be contained in observed_modalities"
        )
    allow_subset = coverage["allow_modality_subset"]
    subset_active = coverage["subset_opt_in_active"]
    if not isinstance(allow_subset, bool) or not isinstance(subset_active, bool):
        raise ValueError("Latent stats modality subset flags must be booleans")
    if policy == _SUBSET_MODALITY_COVERAGE_POLICY:
        if (
            set(expected) == set(canonical_roster)
            or not allow_subset
            or (not subset_active)
        ):
            raise ValueError(
                "Explicit subset policy requires a strict expected-modality subset and active opt-in"
            )
    else:
        if expected != canonical_roster or observed != canonical_roster:
            raise ValueError(
                "Only explicit_modality_subset_opt_in may use less than the dataset full roster"
            )
        if subset_active:
            raise ValueError(
                "Full-roster modality coverage cannot mark subset_opt_in_active"
            )
        if policy == "canonical_dataset_full_roster_required" and allow_subset:
            raise ValueError(
                "Default full-roster coverage cannot enable allow_modality_subset"
            )
    stats["dataset"] = dataset_key
    stats["expected_modalities"] = list(expected)
    stats["observed_modalities"] = list(observed)


def load_latent_stats(path: Path | str) -> dict[str, Any]:
    stats_path = Path(path).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Latent stats do not exist: {stats_path}")
    if stats_path.suffix.lower() in {".pt", ".pth"}:
        try:
            value = torch.load(stats_path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(stats_path, map_location="cpu")
    else:
        value = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"Latent stats must be a mapping, got {type(value)!r}")
    stats = dict(value)
    if str(stats.get("cache_version", "")) != "streamrefine_kl_mean_v2_anatomy_mask":
        raise ValueError(
            f"Latent stats cache_version must be 'streamrefine_kl_mean_v2_anatomy_mask', got {stats.get('cache_version')!r}"
        )
    if str(stats.get("policy", "")) != "shared_across_training_modalities":
        raise ValueError(
            f"Latent stats policy must be 'shared_across_training_modalities', got {stats.get('policy')!r}"
        )
    stats_split = str(stats.get("split", "")).strip().lower()
    if stats_split not in {"train", "training"}:
        raise ValueError(
            f"Latent stats split must be train/training; validation or unlabeled statistics are forbidden, got {stats.get('split')!r}"
        )
    stats["split"] = "train"
    _validate_stats_modality_coverage(stats)
    channels = int(stats.get("channels", 0))
    if channels != 16:
        raise ValueError(f"Latent stats channels must be 16, got {channels}")
    mean = torch.as_tensor(stats.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(stats.get("std"), dtype=torch.float32)
    if tuple(mean.shape) != (channels,) or tuple(std.shape) != (channels,):
        raise ValueError(
            f"Latent stats mean/std must have shape [16], got {tuple(mean.shape)} and {tuple(std.shape)}"
        )
    std_epsilon = float(stats.get("std_epsilon", 1e-06))
    if not math.isfinite(std_epsilon) or std_epsilon <= 0:
        raise ValueError(
            f"Latent stats std_epsilon must be finite and positive, got {std_epsilon}"
        )
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or bool((std <= std_epsilon).any())
    ):
        raise ValueError(
            f"Latent stats mean/std must be finite and every std must exceed {std_epsilon:g}"
        )
    checkpoint_hash = str(stats.get("tokenizer_checkpoint_sha256", "")).strip().lower()
    if len(checkpoint_hash) != 64 or any(
        (character not in "0123456789abcdef" for character in checkpoint_hash)
    ):
        raise ValueError(
            "Latent stats must contain a valid tokenizer_checkpoint_sha256"
        )
    config_hash = str(stats.get("tokenizer_config_sha256", "")).strip().lower()
    if len(config_hash) != 64 or any(
        (character not in "0123456789abcdef" for character in config_hash)
    ):
        raise ValueError("Latent stats must contain a valid tokenizer_config_sha256")
    generation_hash = str(stats.get("generation_contract_sha256", "")).strip().lower()
    if len(generation_hash) != 64 or any(
        (character not in "0123456789abcdef" for character in generation_hash)
    ):
        raise ValueError("Latent stats must contain a valid generation_contract_sha256")
    stats["mean_tensor"] = mean[:, None, None, None]
    stats["std_tensor"] = std[:, None, None, None]
    stats["_path"] = str(stats_path)
    return stats


class LatentPairDataset(Dataset):
    """Read aligned KL-mean caches and return fixed StreamRefine latent windows.

    No VidTok model or encoder module is imported or called on this training path.
    Disk values remain raw FP16 posterior means; model-space normalization is
    applied only to the returned FP32 tensors.
    """

    def __init__(
        self,
        pair_manifest: Path | str | Sequence[Mapping[str, Any]],
        *,
        stats_path: Path | str | None = None,
        window_dhw: Sequence[int] = DEFAULT_WINDOW_DHW,
        min_valid_ratio: float = 0.85,
        max_tries: int = 100,
        seed: int = 2026,
        training: bool = True,
        normalize_with_stats: bool = True,
        expected_checkpoint_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_generation_contract_sha256: str | None = None,
        expected_split: str | None = None,
        dataset: str | None = None,
        translation_pairs: Mapping[str, Sequence[Sequence[str]]]
        | Sequence[Sequence[str]]
        | None = None,
    ) -> None:
        self.training = bool(training)
        requested_split = (
            _canonical_split(expected_split, name="expected_split", required=True)
            if expected_split is not None
            else None
        )
        if self.training:
            if requested_split is not None and requested_split != "train":
                raise ValueError("training=True requires expected_split='train'")
            requested_split = "train"
        self.records = load_pair_manifest(
            pair_manifest,
            dataset=dataset,
            translation_pairs=translation_pairs,
            expected_split=requested_split,
            require_files=True,
        )
        if not self.records:
            raise ValueError(
                "Pair manifest did not yield any source-target cache pairs"
            )
        if requested_split is None:
            inferred_splits = {
                _canonical_split(
                    record.get("split"),
                    name=f"Pair manifest record {record.get('pair_id', '<unknown>')!r} split",
                    required=True,
                )
                for record in self.records
            }
            if len(inferred_splits) != 1:
                raise ValueError(
                    "training=False requires expected_split when a pair manifest contains mixed splits"
                )
            requested_split = inferred_splits.pop()
        self.expected_split = requested_split
        self.window_dhw = _shape_tuple(window_dhw, "window_dhw")
        self.min_valid_ratio = float(min_valid_ratio)
        if not 0.0 <= self.min_valid_ratio <= 1.0:
            raise ValueError(f"min_valid_ratio must be in [0,1], got {min_valid_ratio}")
        self.max_tries = int(max_tries)
        if self.max_tries < 0:
            raise ValueError(f"max_tries must be non-negative, got {max_tries}")
        self.seed = int(seed)
        self.normalize_with_stats = bool(normalize_with_stats)
        self.epoch = 0
        self._rng_identity: tuple[int, int] | None = None
        self._rng_generator: torch.Generator | None = None
        self.stats = load_latent_stats(stats_path) if stats_path is not None else None
        if self.normalize_with_stats and self.stats is None:
            raise ValueError(
                "normalize_with_stats=True requires stats_path. Pass a training-split shared latent-stats file, or explicitly set normalize_with_stats=False for raw decoder-space latents."
            )
        manifest_datasets = {
            normalize_dataset_name(
                str(record.get("dataset") or record.get("dataset_key"))
            )
            for record in self.records
        }
        if self.stats is not None:
            if len(manifest_datasets) != 1:
                raise ValueError(
                    f"Latent stats are dataset-specific but pair manifest contains {sorted(manifest_datasets)}"
                )
            manifest_dataset = next(iter(manifest_datasets))
            if str(self.stats["dataset"]) != manifest_dataset:
                raise ValueError(
                    f"Stats dataset {self.stats['dataset']!r} != pair manifest dataset {manifest_dataset!r}"
                )
        expected = str(expected_checkpoint_sha256 or "").strip().lower() or None
        expected_config = str(expected_config_sha256 or "").strip().lower() or None
        expected_generation = (
            str(expected_generation_contract_sha256 or "").strip().lower() or None
        )
        if self.stats is not None:
            stats_hash = str(self.stats["tokenizer_checkpoint_sha256"]).lower()
            stats_config_hash = str(self.stats["tokenizer_config_sha256"]).lower()
            stats_generation_hash = str(
                self.stats["generation_contract_sha256"]
            ).lower()
            if expected is not None and stats_hash != expected:
                raise ValueError(
                    f"Stats/checkpoint hash mismatch: stats={stats_hash}, expected={expected}"
                )
            if expected_config is not None and stats_config_hash != expected_config:
                raise ValueError(
                    f"Stats/model-config hash mismatch: stats={stats_config_hash}, expected={expected_config}"
                )
            if (
                expected_generation is not None
                and stats_generation_hash != expected_generation
            ):
                raise ValueError(
                    f"Stats/generation-contract hash mismatch: stats={stats_generation_hash}, expected={expected_generation}"
                )
            expected = stats_hash
            expected_config = stats_config_hash
            expected_generation = stats_generation_hash
        self.expected_checkpoint_sha256 = expected
        self.expected_config_sha256 = expected_config
        self.expected_generation_contract_sha256 = expected_generation

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self) -> torch.Generator:
        worker = get_worker_info()
        if worker is None:
            identity = (-1, self.seed)
            stream_seed = self.seed
        else:
            identity = (int(worker.id), int(worker.seed))
            stream_seed = int(worker.seed) ^ self.seed
        if self._rng_generator is None or self._rng_identity != identity:
            generator = torch.Generator()
            generator.manual_seed(stream_seed % ((1 << 63) - 1))
            self._rng_generator = generator
            self._rng_identity = identity
        return self._rng_generator

    def _model_latent(self, raw: torch.Tensor) -> torch.Tensor:
        model = raw.float()
        if self.normalize_with_stats and self.stats is not None:
            model = (model - self.stats["mean_tensor"]) / self.stats["std_tensor"]
        return model.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_cache = _load_and_validate_cache(
            record["source_cache_path"],
            self.expected_checkpoint_sha256,
            self.expected_config_sha256,
            self.expected_generation_contract_sha256,
        )
        target_cache = _load_and_validate_cache(
            record["target_cache_path"],
            self.expected_checkpoint_sha256,
            self.expected_config_sha256,
            self.expected_generation_contract_sha256,
        )
        validate_shared_pair_metadata(source_cache, target_cache)
        cache_split = _canonical_split(
            source_cache.get("split"),
            name=f"Cache {source_cache.get('case_id', '<unknown>')!r} split",
            required=True,
        )
        if cache_split != self.expected_split:
            raise ValueError(
                f"Cache split {cache_split!r} != expected_split {self.expected_split!r}"
            )
        source_dataset = normalize_dataset_name(str(source_cache.get("dataset")))
        recorded_dataset = str(
            record.get("dataset") or record.get("dataset_key") or ""
        ).strip()
        if (
            recorded_dataset
            and normalize_dataset_name(recorded_dataset) != source_dataset
        ):
            raise ValueError(
                f"Pair manifest dataset {recorded_dataset!r} != cache {source_cache.get('dataset')!r}"
            )
        cache_snapshot = _canonical_input_snapshot(source_cache)
        for field in (
            "available_modalities",
            "modality_image_paths",
            "input_file_signatures",
            "crop_source_modality",
        ):
            if field not in record:
                continue
            candidate = dict(source_cache)
            candidate[field] = record[field]
            recorded_value = _canonical_input_snapshot(candidate)[field]
            if recorded_value != cache_snapshot[field]:
                raise ValueError(f"Pair manifest {field} != cache {field}")
        cache_snapshot_hash = _input_snapshot_sha256(source_cache)
        recorded_snapshot_hash = (
            str(record.get("input_snapshot_sha256", "")).strip().lower()
        )
        if recorded_snapshot_hash and (
            len(recorded_snapshot_hash) != 64
            or any(
                (
                    character not in "0123456789abcdef"
                    for character in recorded_snapshot_hash
                )
            )
        ):
            raise ValueError(
                "Pair manifest input_snapshot_sha256 must be a valid SHA256"
            )
        if recorded_snapshot_hash and recorded_snapshot_hash != cache_snapshot_hash:
            raise ValueError(
                f"Pair manifest input_snapshot_sha256 {recorded_snapshot_hash!r} != cache {cache_snapshot_hash!r}"
            )
        source_modality = canonical_modality_name(source_cache.get("modality"))
        target_modality = canonical_modality_name(target_cache.get("modality"))
        if self.stats is not None:
            stats_dataset = str(self.stats["dataset"])
            if source_dataset != stats_dataset:
                raise ValueError(
                    f"Stats dataset {stats_dataset!r} != cache dataset {source_dataset!r}"
                )
            expected_modalities = set(self.stats["expected_modalities"])
            observed_modalities = set(self.stats["observed_modalities"])
            for role, modality in (
                ("source", source_modality),
                ("target", target_modality),
            ):
                if (
                    modality not in expected_modalities
                    or modality not in observed_modalities
                ):
                    raise ValueError(
                        f"Pair {role} modality {modality!r} is not covered by both stats expected_modalities={sorted(expected_modalities)} and observed_modalities={sorted(observed_modalities)}"
                    )
        recorded_source = canonical_modality_name(record.get("source_modality"))
        recorded_target = canonical_modality_name(record.get("target_modality"))
        if recorded_source and source_modality != recorded_source:
            raise ValueError(
                f"Pair manifest source modality {recorded_source!r} != cache {source_modality!r}"
            )
        if recorded_target and target_modality != recorded_target:
            raise ValueError(
                f"Pair manifest target modality {recorded_target!r} != cache {target_modality!r}"
            )
        recorded_case = str(record.get("case_id", "")).strip()
        if recorded_case and recorded_case != str(source_cache["case_id"]):
            raise ValueError(
                f"Pair manifest case_id {recorded_case!r} != cache {source_cache['case_id']!r}"
            )
        for field in ("group_id", "split"):
            recorded_value = str(record.get(field, "")).strip()
            cached_value = str(source_cache.get(field, "")).strip()
            if recorded_value and recorded_value != cached_value:
                raise ValueError(
                    f"Pair manifest {field} {recorded_value!r} != cache {cached_value!r}"
                )
        valid_mask = torch.as_tensor(source_cache["valid_mask_latent"]).bool()
        if self.training:
            origin, sampled_ratio = sample_valid_latent_window(
                valid_mask,
                window=self.window_dhw,
                min_valid_ratio=self.min_valid_ratio,
                max_tries=self.max_tries,
                generator=self._generator(),
            )
        else:
            origin, sampled_ratio = find_best_valid_window(valid_mask, self.window_dhw)
        sample = crop_aligned_pair(source_cache, target_cache, origin, self.window_dhw)
        source_raw = sample["source_latent_raw"]
        target_raw = sample["target_latent_raw"]
        source_model = self._model_latent(source_raw)
        target_model = self._model_latent(target_raw)
        sample.update(
            {
                "source_latent_raw": source_raw,
                "target_latent_raw": target_raw,
                "source_latent_model": source_model,
                "target_latent_model": target_model,
                "source_latent": source_model,
                "target_latent": target_model,
                "valid_ratio": torch.tensor(float(sampled_ratio), dtype=torch.float32),
                "valid_window_met_threshold": bool(
                    sampled_ratio >= self.min_valid_ratio
                ),
                "full_latent_shape_dhw": torch.as_tensor(
                    source_cache["latent_shape_dhw"], dtype=torch.long
                ),
                "latent_shape_dhw": torch.as_tensor(
                    source_cache["latent_shape_dhw"], dtype=torch.long
                ),
                "original_shape_hwd": torch.as_tensor(
                    source_cache["original_shape_hwd"], dtype=torch.long
                ),
                "padded_shape_hwd": torch.as_tensor(
                    source_cache["padded_shape_hwd"], dtype=torch.long
                ),
                "spacing_hwd": torch.as_tensor(
                    source_cache["spacing"], dtype=torch.float32
                ),
                "affine": torch.as_tensor(source_cache["affine"], dtype=torch.float64),
                "case_id": str(source_cache["case_id"]),
                "group_id": str(
                    record.get("group_id", source_cache.get("group_id", ""))
                ),
                "pair_id": str(
                    record.get("pair_id")
                    or f"{source_cache['case_id']}_{source_modality}_to_{target_modality}"
                ),
                "dataset": str(source_cache["dataset"]),
                "split": cache_split,
                "source_modality": source_modality,
                "target_modality": target_modality,
                "source_cache_path": str(record["source_cache_path"]),
                "target_cache_path": str(record["target_cache_path"]),
                "tokenizer_checkpoint_sha256": str(
                    source_cache["tokenizer_checkpoint_sha256"]
                ),
                "tokenizer_config_sha256": str(source_cache["tokenizer_config_sha256"]),
                "generation_contract_sha256": str(
                    source_cache["generation_contract_sha256"]
                ),
                "input_snapshot_sha256": cache_snapshot_hash,
                "padding_info": dict(source_cache["padding_info"]),
                "normalization_info": dict(source_cache["normalization_info"]),
                "source_normalization_info": dict(source_cache["normalization_info"]),
                "target_normalization_info": dict(target_cache["normalization_info"]),
                "latent_stats_path": ""
                if self.stats is None
                else str(self.stats["_path"]),
                "latent_stats_applied": bool(
                    self.normalize_with_stats and self.stats is not None
                ),
            }
        )
        return sample


def collate_latent_pair_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack training fields while retaining heterogeneous metadata as lists."""
    if not batch:
        raise ValueError("Cannot collate an empty latent-pair batch")
    rows = [dict(sample) for sample in batch]
    preserved: dict[str, list[Any]] = {}
    for field in _PRESERVE_AS_LIST_FIELDS:
        present = [field in row for row in rows]
        if any(present) and (not all(present)):
            raise KeyError(f"Latent-pair batch has inconsistent presence of {field!r}")
        if all(present):
            preserved[field] = [row.pop(field) for row in rows]
    collated = dict(default_collate(rows))
    collated.update(preserved)
    return collated


def build_latent_pair_dataloader(
    pair_manifest: Path | str | Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    collate_fn: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = LatentPairDataset(pair_manifest, **dataset_kwargs)
    training = bool(dataset_kwargs.get("training", True))
    if shuffle is None:
        shuffle = training
    if persistent_workers is None:
        persistent_workers = int(num_workers) > 0
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        collate_fn=collate_latent_pair_batch if collate_fn is None else collate_fn,
    )
