"""Deterministic full-pair loading through MONAI PersistentDataset.

No object in this module discovers a dataset root. Every cache and image path is
obtained from an explicit manifest/cache record and resolved by configured prefix
maps. Random crops and augmentation live in :mod:`streamrefine.data.augmentation`.
"""

from __future__ import annotations
import copy
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence
from .path_resolver import is_explicitly_remapped_path, resolve_recorded_path

PERSISTENT_PAIR_VERSION = "streamrefine_persistent_pair_v5_latent_first"


def _file_guard(path: str | Path, *, role: str) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    stat = resolved.stat()
    return {
        "role": str(role),
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_preprocessing_snapshot(
    dataset: str, preprocessing_config: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve both cache and medical YAMLs once in the parent process."""
    from streamrefine.data.preprocess_medical import (
        load_yaml_config,
        resolve_preprocessing_config,
    )

    outer_path = Path(preprocessing_config).expanduser().resolve(strict=False)
    if not outer_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing configuration does not exist: {outer_path}"
        )
    outer_cfg = load_yaml_config(outer_path)
    outer_cfg["_config_path"] = str(outer_path)
    resolved_path, resolved_cfg = resolve_preprocessing_config(
        outer_cfg, str(dataset).lower()
    )
    resolved_path = Path(resolved_path).expanduser().resolve(strict=False)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Resolved medical preprocessing configuration does not exist: {resolved_path}"
        )
    data = resolved_cfg.get("data")
    if not isinstance(data, Mapping):
        raise ValueError(
            f"Resolved medical preprocessing configuration lacks data mapping: {resolved_path}"
        )
    snapshot = {
        "outer_path": str(outer_path),
        "outer_sha256": _sha256_file(outer_path),
        "resolved_path": str(resolved_path),
        "resolved_sha256": _sha256_file(resolved_path),
    }
    return (copy.deepcopy(dict(data)), snapshot)


def _verify_recorded_cache_guard(path: str | Path, guard: Any, *, role: str) -> None:
    """Validate copied caches by content, independently of copy-time metadata.

    A filesystem copy may change mtime without changing the latent payload.
    The generated size and SHA-256 must still match. Dataset construction then
    captures the current size/mtime in runtime guards for subsequent lookups.
    """
    if not isinstance(guard, Mapping):
        raise ValueError(f"Pair manifest lacks the generated {role} file guard")
    resolved = Path(path).resolve(strict=False)
    stat = resolved.stat()
    expected_sha256 = str(guard.get("sha256", "")).strip().lower()
    if len(expected_sha256) != 64 or any(
        (character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"Pair manifest {role} guard lacks a valid SHA-256")
    try:
        expected_size = int(guard.get("size_bytes", -1))
        expected_mtime = int(guard.get("mtime_ns", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Pair manifest {role} guard has malformed size/mtime"
        ) from exc
    if expected_size < 0 or expected_mtime < 0:
        raise ValueError(f"Pair manifest {role} guard has malformed size/mtime")
    if int(stat.st_size) != expected_size:
        raise ValueError(
            f"Resolved {role} size differs from the generated pair manifest: {resolved} (recorded={expected_size} bytes, actual={stat.st_size} bytes). Check that the cache copy is complete and belongs to this manifest."
        )
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Resolved {role} SHA-256 differs from the generated pair manifest: {resolved}"
        )


def _verify_runtime_file_guards(record: Mapping[str, Any]) -> None:
    guards = record.get("_runtime_file_guards")
    if not isinstance(guards, list) or not guards:
        raise ValueError("Pair record lacks runtime file guards")
    for guard in guards:
        if not isinstance(guard, Mapping):
            raise ValueError("Runtime file guard is malformed")
        path = Path(str(guard.get("path", "")))
        try:
            stat = path.stat()
        except OSError as exc:
            raise FileNotFoundError(
                f"Guarded {guard.get('role', 'file')} is unavailable: {path}"
            ) from exc
        if int(stat.st_size) != int(guard.get("size_bytes", -1)) or int(
            stat.st_mtime_ns
        ) != int(guard.get("mtime_ns", -1)):
            raise ValueError(
                f"Guarded {guard.get('role', 'file')} changed after dataset construction: {path}"
            )


def guarded_pickle_hashing(record: Any) -> bytes:
    """MONAI data hash that rechecks explicit files before every cache lookup."""
    if not isinstance(record, Mapping):
        raise TypeError("PersistentDataset records must be mappings")
    _verify_runtime_file_guards(record)
    from monai.data.utils import pickle_hashing

    return pickle_hashing(record)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"Manifest {path}:{line_number} must contain a JSON object"
                    )
                rows.append(dict(value))
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("records", value.get("pairs"))
    if not isinstance(value, list):
        raise TypeError(f"Manifest {path} must contain a JSON list or JSONL objects")
    if not all((isinstance(row, Mapping) for row in value)):
        raise TypeError(f"Every manifest row in {path} must be a mapping")
    return [dict(row) for row in value]


def load_explicit_pair_records(
    manifest: str | Path,
    *,
    dataset: str,
    pair: str,
    split: str,
    path_remap: Any = (),
    require_files: bool = True,
    include_images: bool = True,
    allow_remapped_image_mtime: bool = False,
) -> list[dict[str, Any]]:
    """Load generated explicit pair rows and resolve endpoints fail-closed."""
    manifest_path = Path(manifest).expanduser().resolve(strict=False)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pair manifest does not exist: {manifest_path}")
    if "_to_" not in pair:
        raise ValueError(f"pair must use source_to_target form, got {pair!r}")
    source_expected, target_expected = pair.lower().split("_to_", 1)
    selected: list[dict[str, Any]] = []
    portable_image_paths: set[str] = set()
    for row in _read_json_rows(manifest_path):
        status = str(row.get("status", "success")).strip().lower()
        if status not in {"ready", "success", "ok", "complete", "completed", "cached"}:
            continue
        row_dataset = (
            str(row.get("dataset") or row.get("dataset_key") or "").strip().lower()
        )
        if row_dataset and row_dataset != dataset.lower():
            continue
        if str(row.get("split", "")).strip().lower() != split.lower():
            continue
        source_modality = str(row.get("source_modality", "")).strip().lower()
        target_modality = str(row.get("target_modality", "")).strip().lower()
        if (source_modality, target_modality) != (source_expected, target_expected):
            continue
        source_value = row.get("source_cache_path", row.get("source_cache"))
        target_value = row.get("target_cache_path", row.get("target_cache"))
        if not source_value or not target_value:
            raise ValueError(
                f"Pair row {row.get('pair_id', '<unknown>')!r} lacks source_cache/target_cache"
            )
        resolved = dict(row)
        resolved["dataset"] = dataset.lower()
        resolved["split"] = split.lower()
        resolved["source_modality"] = source_expected
        resolved["target_modality"] = target_expected
        resolved["source_cache_path"] = str(
            resolve_recorded_path(
                source_value,
                remaps=path_remap,
                base_dir=manifest_path.parent,
                must_exist=require_files,
                role="source cache",
            )
        )
        resolved["target_cache_path"] = str(
            resolve_recorded_path(
                target_value,
                remaps=path_remap,
                base_dir=manifest_path.parent,
                must_exist=require_files,
                role="target cache",
            )
        )
        if require_files:
            _verify_recorded_cache_guard(
                resolved["source_cache_path"],
                row.get("source_cache_file_guard"),
                role="source cache",
            )
            _verify_recorded_cache_guard(
                resolved["target_cache_path"],
                row.get("target_cache_file_guard"),
                role="target cache",
            )
        elif not isinstance(
            row.get("source_cache_file_guard"), Mapping
        ) or not isinstance(row.get("target_cache_file_guard"), Mapping):
            raise ValueError(
                "Pair manifest lacks generated source/target cache file guards"
            )
        available = row.get("available_modalities")
        image_paths = row.get("modality_image_paths")
        image_signatures = row.get("input_file_signatures")
        if (
            not isinstance(available, list)
            or not isinstance(image_paths, Mapping)
            or (not isinstance(image_signatures, Mapping))
        ):
            raise ValueError(
                "Pair manifest must include available_modalities, modality_image_paths, and input_file_signatures from the cache generator"
            )
        mask_policy = str(row.get("anatomy_mask_policy", "")).strip()
        mask_fingerprint = (
            str(row.get("anatomy_mask_contract_fingerprint", "")).strip().lower()
        )
        if not mask_policy:
            raise ValueError("Pair manifest lacks anatomy_mask_policy")
        if len(mask_fingerprint) != 64 or any(
            (character not in "0123456789abcdef" for character in mask_fingerprint)
        ):
            raise ValueError(
                "Pair manifest lacks a valid anatomy_mask_contract_fingerprint"
            )
        mask_digest = str(row.get("anatomy_mask_sha256", "")).strip().lower()
        if len(mask_digest) != 64 or any(
            (character not in "0123456789abcdef" for character in mask_digest)
        ):
            raise ValueError("Pair manifest lacks a valid anatomy_mask_sha256")
        guards = [
            _file_guard(resolved["source_cache_path"], role="source cache"),
            _file_guard(resolved["target_cache_path"], role="target cache"),
        ]
        for modality_value in available:
            modality = str(modality_value).strip().lower()
            path_value = image_paths.get(modality)
            signature = image_signatures.get(modality)
            if path_value is None or not isinstance(signature, Mapping):
                raise ValueError(
                    f"Pair manifest lacks path/signature for modality {modality!r}"
                )
            if not include_images:
                continue
            image_path = resolve_recorded_path(
                path_value,
                remaps=path_remap,
                base_dir=manifest_path.parent,
                must_exist=require_files,
                role=f"{modality} image",
            )
            relaxed = _verify_resolved_image_signature(
                image_path,
                signature,
                modality=modality,
                path_remap=path_remap,
                allow_remapped_mtime=allow_remapped_image_mtime,
            )
            if relaxed and (not signature.get("sha256")):
                portable_image_paths.add(str(image_path))
            guards.append(_file_guard(image_path, role=f"{modality} image"))
        resolved["_runtime_file_guards"] = guards
        resolved.setdefault(
            "pair_id",
            f"{row.get('case_id', 'case')}_{source_expected}_to_{target_expected}",
        )
        selected.append(resolved)
    if not selected:
        raise ValueError(
            f"Manifest {manifest_path} contains no {dataset}/{pair}/{split} successful pair rows"
        )
    if portable_image_paths:
        warnings.warn(
            f"Accepted changed mtime for {len(portable_image_paths)} explicitly remapped raw images in {manifest_path.name}: recorded sizes match. Legacy image signatures have no content SHA-256; preprocessing and runtime file guards remain active. Use data.allow_remapped_image_mtime=false for strict mtime checks.",
            UserWarning,
            stacklevel=2,
        )
    return selected


def _plain_tensor(value: Any):
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if type(tensor) is not torch.Tensor:
        tensor = tensor.as_subclass(torch.Tensor).contiguous()
    return tensor


def _verify_resolved_image_signature(
    path: str | Path,
    signature: Mapping[str, Any],
    *,
    modality: str,
    path_remap: Any = (),
    allow_remapped_mtime: bool = False,
) -> bool:
    """Check raw-image identity; return whether a relocated mtime was accepted.

    Legacy caches have no raw-image digest, so relocation mode verifies size
    and the explicit source-to-destination mapping, not byte-for-byte equality.
    Same-path mutations and size changes remain errors. Optional content hashes
    are always checked when provided; recorded cache metadata is not rewritten.
    """
    resolved = Path(path)
    stat = resolved.stat()
    expected_size = int(signature.get("size_bytes", -1))
    expected_mtime = int(signature.get("mtime_ns", -1))
    if expected_size < 0 or expected_mtime < 0:
        raise ValueError(f"Cache input signature for {modality!r} is malformed")
    if int(stat.st_size) != expected_size:
        raise ValueError(
            f"Resolved {modality!r} image no longer matches its cache-generation size/mtime signature: {resolved} (size recorded={expected_size}, actual={stat.st_size}; mtime_ns recorded={expected_mtime}, actual={stat.st_mtime_ns}). Timestamp relocation cannot accept different file sizes."
        )
    expected_hash = signature.get("sha256")
    if expected_hash is not None:
        expected_hash = str(expected_hash).strip().lower()
        if len(expected_hash) != 64 or any(
            (c not in "0123456789abcdef" for c in expected_hash)
        ):
            raise ValueError(
                f"Cache input signature for {modality!r} has an invalid SHA-256"
            )
        if _sha256_file(resolved) != expected_hash:
            raise ValueError(
                f"Resolved {modality!r} image SHA-256 differs from its input signature: {resolved}"
            )
    if int(stat.st_mtime_ns) == expected_mtime:
        return False
    if allow_remapped_mtime and is_explicitly_remapped_path(
        str(signature.get("resolved_path", "")), resolved, remaps=path_remap
    ):
        return True
    raise ValueError(
        f"Resolved {modality!r} image no longer matches its cache-generation size/mtime signature: {resolved} (size={stat.st_size}; mtime_ns recorded={expected_mtime}, actual={stat.st_mtime_ns}). Only explicitly remapped images may use data.allow_remapped_image_mtime=true."
    )


def _pad_chwd(volume: Any, shape_hwd: Sequence[int], value: float = -1.0):
    from streamrefine.tokenizer.grid_patch_with_coords import pad_chwd_to_shape

    return _plain_tensor(
        pad_chwd_to_shape(volume, tuple((int(v) for v in shape_hwd)), value=value)
    )


def build_dense_global_coordinates(shape_dhw: Sequence[int], *, device: Any = None):
    import torch

    d, h, w = (int(value) for value in shape_dhw)
    axes = [
        torch.arange(length, dtype=torch.float32, device=device) for length in (d, h, w)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def normalize_model_latent(raw: Any, mean: Any, std: Any):
    import torch

    raw_tensor = torch.as_tensor(raw).float()
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=torch.float32).reshape(-1, 1, 1, 1)
    if (
        raw_tensor.shape[0] != mean_tensor.shape[0]
        or mean_tensor.shape != std_tensor.shape
    ):
        raise ValueError("Latent and statistics channel dimensions do not match")
    if (
        not torch.isfinite(raw_tensor).all()
        or not torch.isfinite(mean_tensor).all()
        or (not torch.isfinite(std_tensor).all())
    ):
        raise ValueError("Latent normalization inputs must be finite")
    if bool((std_tensor <= 0).any()):
        raise ValueError("Latent standard deviations must be positive")
    return ((raw_tensor - mean_tensor) / std_tensor).contiguous()


def inverse_model_latent(model: Any, mean: Any, std: Any):
    import torch

    model_tensor = torch.as_tensor(model).float()
    mean_tensor = torch.as_tensor(
        mean, dtype=torch.float32, device=model_tensor.device
    ).reshape(-1, 1, 1, 1)
    std_tensor = torch.as_tensor(
        std, dtype=torch.float32, device=model_tensor.device
    ).reshape(-1, 1, 1, 1)
    if model_tensor.shape[-4] != mean_tensor.shape[0]:
        raise ValueError("Model latent and statistics channel dimensions do not match")
    return (model_tensor * std_tensor + mean_tensor).contiguous()


class DeterministicPairTransform:
    """Dependency-lazy implementation for the concrete MONAI loading Transform."""

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        preprocessing_config: str | Path,
        path_remap: Any,
        stats: Mapping[str, Any],
        contract_identity: str,
        preprocessing_data: Mapping[str, Any] | None = None,
        preprocessing_snapshot: Mapping[str, Any] | None = None,
        include_images: bool = True,
        allow_remapped_image_mtime: bool = False,
    ) -> None:
        self.dataset = str(dataset).lower()
        self.split = str(split).lower()
        self.preprocessing_config = str(preprocessing_config)
        self.path_remap = path_remap
        if preprocessing_data is None or preprocessing_snapshot is None:
            preprocessing_data, preprocessing_snapshot = (
                _resolve_preprocessing_snapshot(self.dataset, self.preprocessing_config)
            )
        self.preprocessing_data = copy.deepcopy(dict(preprocessing_data))
        self.preprocessing_snapshot = copy.deepcopy(dict(preprocessing_snapshot))
        stats_dataset = str(stats["dataset"]).lower()
        if stats_dataset != self.dataset:
            raise ValueError(
                f"Latent statistics dataset {stats_dataset!r} does not match {self.dataset!r}"
            )
        self.stats = {
            "mean": list(stats["mean"]),
            "std": list(stats["std"]),
            "dataset": stats_dataset,
            "expected_modalities": [
                str(value).lower() for value in stats["expected_modalities"]
            ],
            "observed_modalities": [
                str(value).lower() for value in stats["observed_modalities"]
            ],
            "tokenizer_checkpoint_sha256": str(stats["tokenizer_checkpoint_sha256"]),
            "tokenizer_config_sha256": str(stats["tokenizer_config_sha256"]),
            "generation_contract_sha256": str(stats["generation_contract_sha256"]),
            "path": str(stats.get("_path", "")),
        }
        self.contract_identity = str(contract_identity)
        self.include_images = bool(include_images)
        self.allow_remapped_image_mtime = bool(allow_remapped_image_mtime)
        self._medical_transform: Any = None
        self._medical_modalities: tuple[str, ...] | None = None

    def _build_medical_transform(self, modalities: Sequence[str]):
        from streamrefine.data.preprocess_medical import (
            build_modality_full_volume_transform,
        )

        transform = build_modality_full_volume_transform(
            self.dataset, self.preprocessing_data, modalities
        )
        self._medical_transform = transform
        self._medical_modalities = tuple(modalities)
        return transform

    def __call__(self, record: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from streamrefine.data.latent_pair_dataset import (
            _canonical_input_snapshot,
            _input_snapshot_sha256,
            _load_and_validate_cache,
            validate_shared_pair_metadata,
        )
        from streamrefine.data.preprocess_medical import (
            canonical_modality_name,
            load_and_preprocess_modality_case,
            normalize_dataset_name,
        )
        from streamrefine.data.anatomy_mask import (
            anatomy_mask_contract,
            anatomy_mask_contract_fingerprint,
            compose_pir_mask_latent_hard,
            downsample_anatomy_mask_to_latent,
        )

        expected_checkpoint = self.stats["tokenizer_checkpoint_sha256"]
        expected_config = self.stats["tokenizer_config_sha256"]
        expected_generation = self.stats["generation_contract_sha256"]
        source_cache = _load_and_validate_cache(
            record["source_cache_path"],
            expected_checkpoint,
            expected_config,
            expected_generation,
        )
        target_cache = _load_and_validate_cache(
            record["target_cache_path"],
            expected_checkpoint,
            expected_config,
            expected_generation,
        )
        generation_contract = source_cache.get("generation_contract")
        if not isinstance(generation_contract, Mapping):
            raise ValueError("Source cache lacks generation_contract mapping")
        if (
            str(generation_contract.get("preprocessing_config_sha256", "")).lower()
            != str(self.preprocessing_snapshot.get("resolved_sha256", "")).lower()
        ):
            raise ValueError(
                "Current medical preprocessing YAML differs from the configuration that generated the latent cache"
            )
        validate_shared_pair_metadata(source_cache, target_cache)
        recorded_mask_digest = str(record.get("anatomy_mask_sha256", "")).lower()
        if recorded_mask_digest != str(source_cache["anatomy_mask_sha256"]).lower():
            raise ValueError("Manifest/cache anatomy_mask_sha256 mismatch")
        if normalize_dataset_name(str(source_cache["dataset"])) != self.dataset:
            raise ValueError("Cache dataset does not match configured dataset")
        recorded_dataset = str(
            record.get("dataset") or record.get("dataset_key") or ""
        ).strip()
        if (
            recorded_dataset
            and normalize_dataset_name(recorded_dataset) != self.dataset
        ):
            raise ValueError("Manifest dataset does not match configured/cache dataset")
        if str(source_cache.get("split", "")).lower() != self.split:
            raise ValueError("Cache split does not match configured split")
        if str(source_cache.get("case_id")) != str(
            record.get("case_id", source_cache.get("case_id"))
        ):
            raise ValueError("Manifest/cache case_id mismatch")
        for field in ("group_id", "split"):
            recorded_value = str(record.get(field, "")).strip()
            cached_value = str(source_cache.get(field, "")).strip()
            if recorded_value and recorded_value != cached_value:
                raise ValueError(f"Manifest/cache {field} mismatch")
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
            if _canonical_input_snapshot(candidate)[field] != cache_snapshot[field]:
                raise ValueError(f"Manifest/cache {field} mismatch")
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
                "Manifest input_snapshot_sha256 must be a hexadecimal SHA256"
            )
        if recorded_snapshot_hash and recorded_snapshot_hash != cache_snapshot_hash:
            raise ValueError("Manifest/cache input_snapshot_sha256 mismatch")
        available_raw = source_cache.get("available_modalities")
        paths_raw = source_cache.get("modality_image_paths")
        if not isinstance(available_raw, (list, tuple)) or not isinstance(
            paths_raw, Mapping
        ):
            raise ValueError(
                "Cache must record available_modalities and modality_image_paths"
            )
        modalities = tuple(
            (canonical_modality_name(str(item)) for item in available_raw)
        )
        paths_by_modality = {
            canonical_modality_name(str(key)): value for key, value in paths_raw.items()
        }
        signatures_raw = source_cache.get("input_file_signatures")
        if not isinstance(signatures_raw, Mapping):
            raise ValueError("Cache must record input_file_signatures")
        signatures_by_modality = {
            canonical_modality_name(str(key)): value
            for key, value in signatures_raw.items()
        }
        resolved_images: dict[str, str] = {}
        for modality in modalities:
            if modality not in paths_by_modality:
                raise ValueError(
                    f"Cache lacks recorded image path for modality {modality!r}"
                )
            if modality not in signatures_by_modality or not isinstance(
                signatures_by_modality[modality], Mapping
            ):
                raise ValueError(
                    f"Cache lacks a file signature for modality {modality!r}"
                )
            if not self.include_images:
                resolved_images[modality] = str(paths_by_modality[modality])
                continue
            resolved = resolve_recorded_path(
                paths_by_modality[modality],
                remaps=self.path_remap,
                must_exist=True,
                role=f"{modality} image",
            )
            _verify_resolved_image_signature(
                resolved,
                signatures_by_modality[modality],
                modality=modality,
                path_remap=self.path_remap,
                allow_remapped_mtime=self.allow_remapped_image_mtime,
            )
            resolved_images[modality] = str(resolved)
        source_modality = canonical_modality_name(str(source_cache["modality"]))
        target_modality = canonical_modality_name(str(target_cache["modality"]))
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
                    f"Pair {role} modality {modality!r} is not covered by the training statistics"
                )
        if (
            source_modality != str(record["source_modality"]).lower()
            or target_modality != str(record["target_modality"]).lower()
        ):
            raise ValueError("Manifest/cache modality direction mismatch")
        original_hwd = tuple((int(v) for v in source_cache["original_shape_hwd"]))
        padded_hwd = tuple((int(v) for v in source_cache["padded_shape_hwd"]))
        cached_anatomy = _plain_tensor(source_cache["anatomy_mask_latent_hard"]).bool()
        mask_policy = str(self.preprocessing_data.get("anatomy_mask_policy", ""))
        active_mask_contract = anatomy_mask_contract(self.dataset, mask_policy)
        active_mask_fingerprint = anatomy_mask_contract_fingerprint(
            self.dataset, mask_policy
        )
        if dict(source_cache["anatomy_mask_contract"]) != active_mask_contract:
            raise ValueError(
                "Cache anatomy-mask contract differs from active preprocessing"
            )
        if (
            str(source_cache["anatomy_mask_contract_fingerprint"])
            != active_mask_fingerprint
        ):
            raise ValueError(
                "Cache anatomy-mask fingerprint differs from active preprocessing"
            )
        if str(record.get("anatomy_mask_policy", "")) != mask_policy:
            raise ValueError("Manifest/cache anatomy_mask_policy mismatch")
        if (
            str(record.get("anatomy_mask_contract_fingerprint", ""))
            != active_mask_fingerprint
        ):
            raise ValueError("Manifest/cache anatomy-mask fingerprint mismatch")
        image_payload: dict[str, Any] = {}
        image_meta: dict[str, Any] = {}
        if self.include_images:
            case_record = {
                "dataset": self.dataset,
                "dataset_key": self.dataset,
                "case_id": str(source_cache["case_id"]),
                "group_id": str(source_cache.get("group_id", "")),
                "split": self.split,
                "modalities": resolved_images,
                "cohort": str(source_cache.get("cohort", "")),
                "task": str(source_cache.get("task", "")),
                "anatomy": str(source_cache.get("anatomy", "")),
                "tracer": str(source_cache.get("tracer", "")),
            }
            transform = (
                self._medical_transform
                if self._medical_transform is not None
                and self._medical_modalities == modalities
                else self._build_medical_transform(modalities)
            )
            volumes, image_meta = load_and_preprocess_modality_case(
                transform, case_record
            )
            for modality, volume in volumes.items():
                actual = tuple((int(v) for v in volume.shape[1:]))
                if actual != original_hwd:
                    raise ValueError(
                        f"Preprocessed {modality} shape {actual} does not match cache {original_hwd}; use the exact cache-generation preprocessing configuration"
                    )
            anatomy_mask_image = image_meta.get("anatomy_mask_image_hwd")
            if anatomy_mask_image is None:
                raise RuntimeError(
                    "Replayed medical preprocessing did not return anatomy_mask_image_hwd"
                )
            replayed_anatomy, _ = downsample_anatomy_mask_to_latent(
                anatomy_mask_image,
                padded_hwd,
                compression_hwd=source_cache["compression_hwd"],
            )
            if tuple(replayed_anatomy.shape) != tuple(
                cached_anatomy.shape
            ) or not torch.equal(replayed_anatomy, cached_anatomy):
                raise ValueError(
                    "Replayed deterministic anatomy mask differs from the cached mask; rebuild caches with the active preprocessing policy"
                )
            if str(image_meta.get("anatomy_mask_policy", "")) != mask_policy:
                raise ValueError(
                    "Replayed anatomy_mask_policy differs from active preprocessing"
                )
            image_payload = {
                "preprocessed_source_image": _pad_chwd(
                    volumes[source_modality], padded_hwd, value=-1.0
                )
                .permute(0, 3, 1, 2)
                .contiguous(),
                "preprocessed_gt_image": _pad_chwd(
                    volumes[target_modality], padded_hwd, value=-1.0
                )
                .permute(0, 3, 1, 2)
                .contiguous(),
            }
        source_raw = _plain_tensor(source_cache["latent_mu"])
        target_raw = _plain_tensor(target_cache["latent_mu"])
        source_model = normalize_model_latent(
            source_raw, self.stats["mean"], self.stats["std"]
        )
        target_model = normalize_model_latent(
            target_raw, self.stats["mean"], self.stats["std"]
        )
        hard = _plain_tensor(source_cache["valid_mask_latent"]).bool()
        anatomy_hard = cached_anatomy
        pir_hard = compose_pir_mask_latent_hard(hard, anatomy_hard)
        soft = (
            _plain_tensor(source_cache["valid_mask_latent_soft"])
            .float()
            .clamp_(0.0, 1.0)
        )
        coords = build_dense_global_coordinates(source_raw.shape[-3:])
        if (
            not torch.isfinite(source_model).all()
            or not torch.isfinite(target_model).all()
        ):
            raise ValueError("Normalized full-volume latents contain NaN or Inf")
        return {
            "source_latent_raw": source_raw,
            "target_latent_raw": target_raw,
            "source_latent_model": source_model,
            "target_latent_model": target_model,
            "source_latent": source_model,
            "target_latent": target_model,
            "valid_mask_latent_soft": soft,
            "valid_mask_latent": soft,
            "valid_mask_latent_hard": hard,
            "anatomy_mask_latent_hard": anatomy_hard,
            "pir_mask_latent_hard": pir_hard,
            **image_payload,
            "global_coords_dhw": coords,
            "crop_origin_dhw": torch.zeros(3, dtype=torch.long),
            "full_latent_shape_dhw": torch.as_tensor(
                source_raw.shape[-3:], dtype=torch.long
            ),
            "latent_shape_dhw": torch.as_tensor(
                source_raw.shape[-3:], dtype=torch.long
            ),
            "original_shape_hwd": torch.as_tensor(original_hwd, dtype=torch.long),
            "padded_shape_hwd": torch.as_tensor(padded_hwd, dtype=torch.long),
            "spacing_hwd": torch.as_tensor(
                source_cache["spacing"], dtype=torch.float32
            ),
            "affine": torch.as_tensor(source_cache["affine"], dtype=torch.float64),
            "case_id": str(source_cache["case_id"]),
            "group_id": str(source_cache.get("group_id", "")),
            "pair_id": str(record["pair_id"]),
            "dataset": self.dataset,
            "split": self.split,
            "source_modality": source_modality,
            "target_modality": target_modality,
            "source_cache_path": str(record["source_cache_path"]),
            "target_cache_path": str(record["target_cache_path"]),
            "source_image_path": resolved_images[source_modality],
            "target_image_path": resolved_images[target_modality],
            "available_modalities": list(modalities),
            "modality_image_paths": resolved_images,
            "metadata_by_modality": image_meta.get("metadata_by_modality", {}),
            "cohort": str(source_cache.get("cohort", "")),
            "task": str(source_cache.get("task", "")),
            "anatomy": str(source_cache.get("anatomy", "")),
            "tracer": str(source_cache.get("tracer", "")),
            "tokenizer_checkpoint_sha256": expected_checkpoint,
            "tokenizer_config_sha256": expected_config,
            "generation_contract_sha256": expected_generation,
            "anatomy_mask_policy": mask_policy,
            "anatomy_mask_contract": active_mask_contract,
            "anatomy_mask_contract_fingerprint": active_mask_fingerprint,
            "anatomy_mask_sha256": str(source_cache["anatomy_mask_sha256"]),
            "input_snapshot_sha256": cache_snapshot_hash,
            "latent_stats_path": self.stats["path"],
            "latent_stats_mean": torch.as_tensor(
                self.stats["mean"], dtype=torch.float32
            ),
            "latent_stats_std": torch.as_tensor(self.stats["std"], dtype=torch.float32),
            "padding_info": dict(source_cache["padding_info"]),
            "normalization_info": dict(source_cache["normalization_info"]),
            "persistent_contract_identity": self.contract_identity,
            "image_layout": "C,D,H,W",
            "latent_layout": "C,D,H,W",
        }


def persistent_contract_identity(
    *,
    dataset: str,
    split: str,
    preprocessing_config: str | Path,
    cache_schema: str,
    preprocessing_snapshot: Mapping[str, Any] | None = None,
    include_images: bool = True,
) -> str:
    path = Path(preprocessing_config).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessing configuration does not exist: {path}")
    payload = {
        "version": PERSISTENT_PAIR_VERSION,
        "dataset": str(dataset).lower(),
        "split": str(split).lower(),
        "cache_schema": str(cache_schema),
        "include_images": bool(include_images),
        "preprocessing_path": str(path),
        "preprocessing_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resolved_preprocessing_snapshot": copy.deepcopy(
            dict(preprocessing_snapshot or {})
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_persistent_pair_dataset(
    config: Mapping[str, Any], *, split: str, include_images: bool = True
):
    """Build MONAI PersistentDataset; importing this function does not import MONAI."""
    try:
        from monai.data.utils import pickle_hashing
        from monai.transforms import Compose
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required to build the StreamRefine PersistentDataset"
        ) from exc
    from streamrefine.data.latent_pair_dataset import load_latent_stats
    from streamrefine.data.monai_transforms import (
        LoadLatentPairTransform,
        StreamRefinePersistentDataset,
    )
    from streamrefine.paths import runtime_path_remaps

    data_cfg = config["data"]
    allow_remapped_image_mtime = data_cfg.get("allow_remapped_image_mtime", False)
    if not isinstance(allow_remapped_image_mtime, bool):
        raise ValueError("data.allow_remapped_image_mtime must be a boolean")
    path_remap = runtime_path_remaps(config)
    manifest_key = f"{split}_manifest"
    records = load_explicit_pair_records(
        data_cfg[manifest_key],
        dataset=data_cfg["dataset"],
        pair=data_cfg["pair"],
        split=split,
        path_remap=path_remap,
        require_files=True,
        include_images=include_images,
        allow_remapped_image_mtime=allow_remapped_image_mtime,
    )
    stats = load_latent_stats(config["latent"]["stats_path"])
    preprocessing_data, preprocessing_snapshot = _resolve_preprocessing_snapshot(
        data_cfg["dataset"], data_cfg["preprocessing_config"]
    )
    configured_mask_policy = str(data_cfg.get("anatomy_mask_policy", "")).strip()
    preprocessing_mask_policy = str(
        preprocessing_data.get("anatomy_mask_policy", "")
    ).strip()
    if configured_mask_policy != preprocessing_mask_policy:
        raise ValueError(
            f"Dataset config anatomy_mask_policy differs from the cache preprocessing config: {configured_mask_policy!r} vs {preprocessing_mask_policy!r}"
        )
    identity = persistent_contract_identity(
        dataset=data_cfg["dataset"],
        split=split,
        preprocessing_config=data_cfg["preprocessing_config"],
        cache_schema=config["latent"]["cache_schema"],
        preprocessing_snapshot=preprocessing_snapshot,
        include_images=include_images,
    )
    transform = LoadLatentPairTransform(
        dataset=data_cfg["dataset"],
        split=split,
        preprocessing_config=data_cfg["preprocessing_config"],
        path_remap=path_remap,
        stats=stats,
        contract_identity=identity,
        preprocessing_data=preprocessing_data,
        preprocessing_snapshot=preprocessing_snapshot,
        include_images=include_images,
        allow_remapped_image_mtime=allow_remapped_image_mtime,
    )
    payload_scope = "latent_and_images" if include_images else "latent_only"
    cache_dir = (
        Path(config["persistent_cache"]["root"])
        / str(config["persistent_cache"]["namespace"])
        / f"{data_cfg['dataset']}_{split}_{data_cfg['pair']}_{payload_scope}"
        / identity
    )
    if os.name == "nt":
        cache_path = str(cache_dir.expanduser().resolve(strict=False))
        if not cache_path.startswith("\\\\?\\"):
            cache_path = (
                "\\\\?\\UNC\\" + cache_path[2:]
                if cache_path.startswith("\\\\")
                else "\\\\?\\" + cache_path
            )
        cache_dir = Path(cache_path)
    return StreamRefinePersistentDataset(
        data=records,
        transform=Compose([transform]),
        cache_dir=str(cache_dir),
        hash_func=guarded_pickle_hashing,
        hash_transform=pickle_hashing,
    )
