from __future__ import annotations
import importlib
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

STREAMREFINE_ROOT = Path(__file__).resolve().parents[3]
STREAMREFINE_TOKENIZER_ROOT = (
    STREAMREFINE_ROOT / "vidtok_rec" / "streamrefine_tokenizer"
)
DEFAULT_FINETUNE_CONFIGS = {
    "brats24": STREAMREFINE_ROOT
    / "vidtok_rec"
    / "configs"
    / "vidtok_kl488_brats24_evaluate.yaml",
    "synthrad": STREAMREFINE_ROOT
    / "vidtok_rec"
    / "configs"
    / "vidtok_kl488_synthrad_evaluate.yaml",
    "autopet": STREAMREFINE_ROOT
    / "vidtok_rec"
    / "configs"
    / "vidtok_kl488_autopet_evaluate.yaml",
}
DATASET_DISPLAY_NAMES = {
    "brats24": "BraTS24",
    "synthrad": "SynthRAD",
    "autopet": "AutoPET",
}
DEFAULT_MODALITIES = {
    "brats24": ("t1n", "t1c", "t2w", "t2f"),
    "synthrad": ("mr", "ct", "cbct"),
    "autopet": ("ct", "pet"),
}
DATASET_PATH_FIELDS = (
    "image",
    "label",
    "image_path",
    "label_path",
    "paired_image",
    "paired_image_path",
    "source",
    "target",
    "source_image_path",
    "target_image_path",
)
MODALITY_ALIASES = {
    "t1": "t1n",
    "t1n": "t1n",
    "t1ce": "t1c",
    "t1c": "t1c",
    "t2": "t2w",
    "t2w": "t2w",
    "flair": "t2f",
    "t2f": "t2f",
    "mr": "mr",
    "mri": "mr",
    "ct": "ct",
    "cbct": "cbct",
    "pet": "pet",
}


def ensure_streamrefine_tokenizer_path() -> None:
    """Make the sibling medical VidTok evaluator importable without installation."""
    for path in (STREAMREFINE_ROOT, STREAMREFINE_TOKENIZER_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _tokenizer_module(name: str):
    ensure_streamrefine_tokenizer_path()
    try:
        return importlib.import_module(f"vidtok_rec.streamrefine_tokenizer.{name}")
    except ModuleNotFoundError as error:
        if error.name not in {"vidtok_rec", "vidtok_rec.streamrefine_tokenizer"}:
            raise
        return importlib.import_module(name)


def normalize_dataset_name(name: str | None) -> str:
    if not name:
        raise ValueError("dataset name must be provided")
    key = str(name).lower().replace("-", "").replace("_", "").strip()
    if key in {"brats", "brats24", "brats2024"}:
        return "brats24"
    if key in {"synthrad", "synthrad2025"}:
        return "synthrad"
    if key in {"autopet", "autopetiii", "autopet3"}:
        return "autopet"
    raise ValueError(f"Unsupported dataset: {name}")


def default_finetune_config_path(dataset: str) -> Path:
    """Resolve the evaluator config that defines tokenizer-matched preprocessing."""
    dataset_key = normalize_dataset_name(dataset)
    registry = _tokenizer_module("dataset_registry")
    configured = getattr(registry, "DEFAULT_DATASET_CONFIGS", {})
    path = Path(configured.get(dataset_key, DEFAULT_FINETUNE_CONFIGS[dataset_key]))
    return path.expanduser().resolve()


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load YAML through the sibling evaluator's canonical config loader."""
    return _tokenizer_module("config").load_yaml_config(path)


def load_dataset_matching_finetune_config(
    config_path: Path | str | None, dataset: str, *, explicit: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Load a preprocessing config and reject accidental cross-dataset reuse."""
    dataset_key = normalize_dataset_name(dataset)
    path = (
        default_finetune_config_path(dataset_key)
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(
                f"StreamRefine preprocessing config does not exist: {path}"
            )
        path = default_finetune_config_path(dataset_key)
    cfg = load_yaml_config(path)
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError(f"Config must contain a data mapping: {path}")
    configured_dataset = normalize_dataset_name(
        str(data_cfg.get("dataset", dataset_key))
    )
    if configured_dataset != dataset_key:
        if explicit:
            raise ValueError(
                f"Explicit StreamRefine config dataset mismatch: requested {dataset_key!r}, but {path} declares {configured_dataset!r}."
            )
        path = default_finetune_config_path(dataset_key)
        cfg = load_yaml_config(path)
    return (path, cfg)


def load_dataset_module(dataset: str):
    """Reuse the live evaluator registry and its three dataset modules."""
    registry = _tokenizer_module("dataset_registry")
    return registry.load_dataset_module(normalize_dataset_name(dataset))


def _is_placeholder(value: Any) -> bool:
    text = str(value or "").strip().upper()
    return not text or text.startswith("REPLACE_WITH_")


def _resolve_from_config(value: Path | str, cfg: Mapping[str, Any]) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = cfg.get("_config_path")
    base = (
        Path(str(config_path)).expanduser().resolve().parent
        if config_path
        else Path.cwd()
    )
    return (base / path).resolve()


def resolve_preprocessing_config(
    cfg: dict[str, Any], dataset: str
) -> tuple[Path, dict[str, Any]]:
    """Resolve a cache YAML to the live VidTok medical preprocessing config.

    Cache YAMLs intentionally keep cache/tool settings separate from the
    evaluator YAML that owns datalists and medical preprocessing. Dataset
    discovery must retain the evaluator YAML as its relative-path base. The
    outer config's ``data`` mapping is updated with the resolved preprocessing
    values so the later full-volume transform sees the exact same parameters.
    """
    dataset_key = normalize_dataset_name(dataset)
    outer_data = cfg.get("data")
    if not isinstance(outer_data, Mapping):
        raise ValueError("Config must contain a data mapping")
    outer_data = dict(outer_data)
    reference = outer_data.get("preprocessing_config")
    if not _is_placeholder(reference):
        path = _resolve_from_config(str(reference), cfg)
        path, preprocessing_cfg = load_dataset_matching_finetune_config(
            path, dataset_key, explicit=True
        )
    elif "train_datalist" in outer_data and "val_datalist" in outer_data:
        config_path = cfg.get("_config_path")
        path = (
            Path(str(config_path)).expanduser().resolve()
            if config_path
            else default_finetune_config_path(dataset_key)
        )
        preprocessing_cfg = dict(cfg)
    else:
        path, preprocessing_cfg = load_dataset_matching_finetune_config(
            None, dataset_key, explicit=False
        )
    base_data = preprocessing_cfg.get("data")
    if not isinstance(base_data, Mapping):
        raise ValueError(f"Preprocessing config must contain a data mapping: {path}")
    discovery_data = dict(base_data)
    path_override_keys = {
        "data_root",
        "label_dir",
        "splits_path",
        "cache_dir",
        "train_datalist",
        "val_datalist",
    }
    override_keys = {
        *path_override_keys,
        "fold",
        "cohorts",
        "tasks",
        "anatomies",
        "val_fraction",
        "orientation",
        "normalization",
        "full_spatial_size_hwd",
        "allow_label_fallback_from_image",
        "input_already_normalized",
        "skip_intensity_normalization",
        "pre_normalized_range_tolerance",
        "anatomy_mask_policy",
        "brats_affine_policy",
    }
    for key in override_keys:
        value = outer_data.get(key)
        if value is not None and (not _is_placeholder(value)):
            discovery_data[key] = (
                str(_resolve_from_config(value, cfg))
                if key in path_override_keys
                else value
            )
    discovery_data["dataset"] = DATASET_DISPLAY_NAMES[dataset_key]
    dataset_cfg = dict(preprocessing_cfg)
    dataset_cfg["data"] = discovery_data
    dataset_cfg["_config_path"] = str(path)
    runtime_data = dict(discovery_data)
    runtime_data.update(
        {key: value for key, value in outer_data.items() if key not in override_keys}
    )
    runtime_data["dataset"] = DATASET_DISPLAY_NAMES[dataset_key]
    runtime_data["preprocessing_config"] = str(path)
    cfg["data"] = runtime_data
    cfg["_preprocessing_config_path"] = str(path)
    return (path, dataset_cfg)


def _read_direct_manifest(path: Path) -> list[dict[str, Any]]:
    import json

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
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for split_name, keys in (
            ("train", ("training", "train")),
            ("val", ("validation", "val")),
        ):
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, list):
                    rows.extend(
                        (
                            {**dict(item), "split": split_name}
                            for item in candidate
                            if isinstance(item, Mapping)
                        )
                    )
                    break
        if rows:
            return rows
        for key in ("data", "records"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [dict(item) for item in candidate if isinstance(item, Mapping)]
    raise ValueError(f"Unsupported direct input manifest format: {path}")


def _configured_datalist_path(dataset_cfg: Mapping[str, Any], key: str) -> Path | None:
    data_cfg = dataset_cfg.get("data")
    if not isinstance(data_cfg, Mapping) or _is_placeholder(data_cfg.get(key)):
        return None
    value = data_cfg[key]
    resolver = _tokenizer_module("config").resolve_path
    resolved = resolver(value, dataset_cfg.get("_config_path"))
    return None if resolved is None else Path(resolved).expanduser().resolve()


def _with_split(
    records: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        recorded = str(row.get("split", "")).strip().lower()
        aliases = {"training": "train", "validation": "val", "valid": "val"}
        recorded = aliases.get(recorded, recorded)
        if recorded and recorded != split:
            raise ValueError(
                f"Configured {split} datalist contains record marked split={row.get('split')!r}: case_id={row.get('case_id')!r}."
            )
        row["split"] = split
        rows.append(row)
    return rows


def _rebase_dataset_record_paths(records, dataset_cfg, dataset):
    """Resolve relative image names against the configured dataset root."""
    root = Path(dataset_cfg["data"]["data_root"]).expanduser().resolve()
    result = []
    for record in records:
        row = dict(record)
        for key in DATASET_PATH_FIELDS:
            value = row.get(key)
            if isinstance(value, str) and value and (not Path(value).is_absolute()):
                row[key] = str((root / value).resolve())
        result.append(row)
    return result


def load_configured_records(
    cfg: dict[str, Any], dataset: str, split: str, rebuild: bool = False
) -> list[dict[str, Any]]:
    """Load train/val records and always attach an unambiguous split label."""
    dataset_key = normalize_dataset_name(dataset)
    split_key = str(split).strip().lower()
    split_key = {"training": "train", "validation": "val", "valid": "val"}.get(
        split_key, split_key
    )
    if split_key not in {"train", "val", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    _, dataset_cfg = resolve_preprocessing_config(cfg, dataset_key)
    runtime_data = cfg.get("data", dataset_cfg.get("data", {}))
    direct_manifest = (
        runtime_data.get("manifest") if isinstance(runtime_data, Mapping) else None
    )
    if not _is_placeholder(direct_manifest):
        manifest_path = _resolve_from_config(str(direct_manifest), cfg)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Configured input manifest does not exist: {manifest_path}"
            )
        manifest_rows = _rebase_dataset_record_paths(
            _read_direct_manifest(manifest_path), dataset_cfg, dataset_key
        )
        labelled: list[dict[str, Any]] = []
        for row in manifest_rows:
            row = dict(row)
            for image_field in ("image_path", "image"):
                image_value = row.get(image_field)
                if image_value:
                    image_path = Path(str(image_value)).expanduser()
                    if not image_path.is_absolute():
                        image_path = manifest_path.parent / image_path
                    row[image_field] = str(image_path.resolve())
            recorded = str(row.get("split", "")).strip().lower()
            recorded = {"training": "train", "validation": "val", "valid": "val"}.get(
                recorded, recorded
            )
            if not recorded:
                if split_key == "all":
                    raise ValueError(
                        f"Direct manifest row {row.get('case_id')!r} has no train/val split label"
                    )
                recorded = split_key
            elif recorded not in {"train", "val"}:
                raise ValueError(
                    f"Direct manifest row {row.get('case_id')!r} has unsupported split={row.get('split')!r}; expected train or val"
                )
            row["split"] = recorded
            labelled.append(row)
        if split_key == "all":
            return labelled
        return [row for row in labelled if row["split"] == split_key]
    module = load_dataset_module(dataset_key)
    train_path = _configured_datalist_path(dataset_cfg, "train_datalist")
    val_path = _configured_datalist_path(dataset_cfg, "val_datalist")
    if (
        not rebuild
        and train_path is not None
        and (val_path is not None)
        and train_path.is_file()
        and val_path.is_file()
    ):
        train_records = module.read_datalist(train_path)
        val_records = module.read_datalist(val_path)
    else:
        train_records, val_records = module.load_configured_datalists(
            dataset_cfg, rebuild=rebuild
        )
    train = _with_split(train_records, "train")
    val = _with_split(val_records, "val")
    if split_key == "train":
        selected = train
    elif split_key == "val":
        selected = val
    else:
        selected = [*train, *val]
    return _rebase_dataset_record_paths(selected, dataset_cfg, dataset_key)


def canonical_modality_name(modality: str | None) -> str:
    """Return the lowercase cache/manifest modality vocabulary."""
    text = str(modality or "").strip()
    if not text:
        return ""
    return MODALITY_ALIASES.get(text.lower(), text.lower())


def modality_field_name(modality: str) -> str:
    text = re.sub("[^a-z0-9]+", "_", canonical_modality_name(modality)).strip("_")
    return f"mod_{text or 'image'}"


def _requested_modalities(
    dataset: str, modalities: str | Sequence[str] | None
) -> tuple[str, ...]:
    if modalities is None:
        return DEFAULT_MODALITIES[normalize_dataset_name(dataset)]
    if isinstance(modalities, str):
        if modalities.strip().lower() in {"", "all", "any", "*"}:
            return DEFAULT_MODALITIES[normalize_dataset_name(dataset)]
        values = [item.strip() for item in modalities.split(",") if item.strip()]
    else:
        values = [str(item) for item in modalities]
    result = tuple(dict.fromkeys((canonical_modality_name(item) for item in values)))
    if not result or any((not item for item in result)):
        raise ValueError(f"No usable modalities were provided: {modalities!r}")
    return result


def _record_path(record: Mapping[str, Any]) -> str:
    for key in ("image_path", "image"):
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError(f"Record is missing image_path/image: keys={sorted(record)}")


def _case_key(record: Mapping[str, Any], dataset: str) -> tuple[str, ...]:
    dataset_key = normalize_dataset_name(dataset)
    case_id = str(record.get("case_id", "")).strip()
    if not case_id:
        raise ValueError(f"Record is missing case_id: {record!r}")
    if dataset_key == "brats24":
        return (str(record.get("cohort", "")), case_id)
    if dataset_key == "synthrad":
        return (str(record.get("task", "")), str(record.get("anatomy", "")), case_id)
    return (case_id,)


def build_modality_case_records(
    records: Sequence[Mapping[str, Any]],
    dataset: str,
    modalities: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Group per-volume datalist rows into case-shared modality records.

    BraTS is grouped by ``(cohort, case_id)``, SynthRAD by
    ``(task, anatomy, case_id)``, and AutoPET by ``case_id``. Missing requested
    modalities are retained as partial cases so the cache generator can encode
    every modality that actually exists; pair construction later requires both
    endpoints.
    """
    dataset_key = normalize_dataset_name(dataset)
    requested = set(_requested_modalities(dataset_key, modalities))
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw_record in records:
        record = dict(raw_record)
        modality = canonical_modality_name(record.get("modality"))
        if not modality or modality not in requested:
            continue
        key = _case_key(record, dataset_key)
        previous = grouped[key].get(modality)
        if previous is not None and _record_path(previous) != _record_path(record):
            raise ValueError(
                f"Conflicting {modality} paths for {DATASET_DISPLAY_NAMES[dataset_key]} case {key}: {_record_path(previous)!r} vs {_record_path(record)!r}."
            )
        if previous is not None:
            previous_split = str(previous.get("split", "")).strip().lower()
            current_split = str(record.get("split", "")).strip().lower()
            if previous_split and current_split and (previous_split != current_split):
                raise ValueError(
                    f"Duplicate {modality} record for case {key} crosses splits: {previous_split!r} vs {current_split!r}."
                )
        grouped[key][modality] = record
    cases: list[dict[str, Any]] = []
    order = DEFAULT_MODALITIES[dataset_key]
    for key in sorted(grouped):
        by_modality = grouped[key]
        if not by_modality:
            continue
        split_values = {
            str(row.get("split", "")).strip().lower()
            for row in by_modality.values()
            if str(row.get("split", "")).strip()
        }
        split_values = {
            "train"
            if item == "training"
            else "val"
            if item in {"validation", "valid"}
            else item
            for item in split_values
        }
        if len(split_values) > 1:
            raise ValueError(
                f"Case {key} crosses dataset splits: {sorted(split_values)}"
            )
        first = next(iter(by_modality.values()))
        sorted_modalities = [item for item in order if item in by_modality]
        sorted_modalities.extend(sorted(set(by_modality) - set(sorted_modalities)))
        modality_paths = {
            item: _record_path(by_modality[item]) for item in sorted_modalities
        }
        case_id = str(first.get("case_id", key[-1]))
        cases.append(
            {
                "dataset": DATASET_DISPLAY_NAMES[dataset_key],
                "dataset_key": dataset_key,
                "case_id": case_id,
                "group_id": "_".join((part for part in key if part)) or case_id,
                "modalities": modality_paths,
                "available_modalities": list(modality_paths),
                "modality_records": {
                    item: dict(by_modality[item]) for item in sorted_modalities
                },
                "cohort": str(first.get("cohort", "")),
                "task": str(first.get("task", "")),
                "anatomy": str(first.get("anatomy", "")),
                "tracer": str(first.get("tracer", "")),
                "split": next(iter(split_values), ""),
            }
        )
    return cases


def modality_foreground_mask(
    image: Any, modality: str, *, already_normalized: bool = False
):
    if already_normalized:
        threshold = -0.999
    else:
        threshold = (
            -950.0 if canonical_modality_name(modality) in {"ct", "cbct"} else 0.0
        )
    return image > threshold


def _affines_close(left: Any, right: Any, *, atol: float = 0.001) -> bool:
    if left is None or right is None:
        return left is None and right is None
    import torch

    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64)
    return tuple(left_tensor.shape) == tuple(right_tensor.shape) and bool(
        torch.allclose(left_tensor, right_tensor, atol=atol, rtol=1e-05)
    )


class MultiModalityForegroundMaskd:
    """Validate a common oriented grid and build one union foreground mask."""

    def __init__(
        self,
        image_keys: Sequence[str],
        modality_by_key: Mapping[str, str],
        foreground_key: str = "foreground",
        already_normalized: bool = False,
    ) -> None:
        if not image_keys:
            raise ValueError("image_keys must not be empty")
        self.image_keys = list(image_keys)
        self.modality_by_key = dict(modality_by_key)
        self.foreground_key = foreground_key
        self.already_normalized = bool(already_normalized)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        reference_key = self.image_keys[0]
        reference = data[reference_key]
        reference_shape = tuple(reference.shape)
        reference_affine = getattr(reference, "affine", None)
        mask = None
        for key in self.image_keys:
            image = data[key]
            if tuple(image.shape) != reference_shape:
                raise ValueError(
                    f"All case modalities must share the same oriented grid before foreground crop: {reference_key}={reference_shape}, {key}={tuple(image.shape)}."
                )
            if not _affines_close(reference_affine, getattr(image, "affine", None)):
                raise ValueError(
                    f"All case modalities must share the same affine before foreground crop; mismatch at {key}."
                )
            current = modality_foreground_mask(
                image,
                self.modality_by_key[key],
                already_normalized=self.already_normalized,
            )
            mask = current if mask is None else mask | current
        data[self.foreground_key] = mask
        return data


class ValidateCommonGridd:
    """Validate shape/affine equality without changing the spatial pipeline."""

    def __init__(self, image_keys: Sequence[str]) -> None:
        if not image_keys:
            raise ValueError("image_keys must not be empty")
        self.image_keys = list(image_keys)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        reference_key = self.image_keys[0]
        reference = data[reference_key]
        reference_shape = tuple(reference.shape)
        reference_affine = getattr(reference, "affine", None)
        for key in self.image_keys[1:]:
            image = data[key]
            if tuple(image.shape) != reference_shape:
                raise ValueError(
                    f"All case modalities must share the same oriented grid before shared crop: {reference_key}={reference_shape}, {key}={tuple(image.shape)}."
                )
            if not _affines_close(reference_affine, getattr(image, "affine", None)):
                raise ValueError(
                    f"All case modalities must share the same affine before shared crop; mismatch at {key}, reference={reference_key}, case={data.get('streamrefine_case_id', 'unknown')}. Unknown affine mismatches require a voxel-alignment audit; do not bypass this check or blindly resample using conflicting headers."
                )
        return data


class ModalityNormalizeIntensityd:
    """Match medical VidTok fine-tuning normalization for CT/CBCT and MR/PET."""

    def __init__(
        self,
        key: str,
        modality: str,
        *,
        low: float = 0.5,
        high: float = 99.5,
        max_quantile_samples: int | None = 1000000,
        info_key: str | None = None,
    ) -> None:
        self.key = key
        self.modality = canonical_modality_name(modality)
        self.low = float(low)
        self.high = float(high)
        self.max_quantile_samples = (
            None if max_quantile_samples is None else int(max_quantile_samples)
        )
        self.info_key = info_key or f"{key}_normalization_info"

    def _sample(self, values: Any):
        if self.max_quantile_samples is None or self.max_quantile_samples <= 0:
            return values
        if values.numel() <= self.max_quantile_samples:
            return values
        stride = math.ceil(values.numel() / self.max_quantile_samples)
        return values[::stride]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch

        image = data[self.key].float()
        input_min = float(image.min().item())
        input_max = float(image.max().item())
        if self.modality in {"ct", "cbct"}:
            image = (image.clamp(-1000.0, 1000.0) + 1000.0) / 2000.0
            info = {
                "policy": "hu_clip_to_minus1_1",
                "input_clip": [-1000.0, 1000.0],
                "actual_input_range": [input_min, input_max],
                "output_range": [-1.0, 1.0],
                "clipped": True,
            }
        else:
            values = image[image > 0]
            foreground_selection = "positive_values"
            if values.numel() == 0:
                values = image.reshape(-1)
                foreground_selection = "all_values_fallback"
            values = self._sample(values).contiguous()
            low = torch.quantile(values, self.low / 100.0)
            high = torch.quantile(values, self.high / 100.0)
            image = ((image - low) / (high - low).clamp_min(1e-06)).clamp(0.0, 1.0)
            info = {
                "policy": "positive_percentile_to_minus1_1",
                "percentile_q": [self.low, self.high],
                "actual_input_low_high": [float(low.item()), float(high.item())],
                "actual_input_range": [input_min, input_max],
                "percentile_selection": foreground_selection,
                "max_quantile_samples": self.max_quantile_samples,
                "output_range": [-1.0, 1.0],
                "clipped": True,
            }
        data[self.key] = image.mul(2.0).sub(1.0)
        data[self.info_key] = info
        return data


class RecordPercentileNormalizationParametersd:
    """Record the input percentiles used by the unchanged MONAI BraTS scaler."""

    def __init__(
        self, key: str, *, low: float, high: float, info_key: str | None = None
    ) -> None:
        self.key = key
        self.low = float(low)
        self.high = float(high)
        self.info_key = info_key or f"{key}_normalization_info"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch

        image = data[self.key].float()
        if not torch.isfinite(image).all():
            raise ValueError(f"BraTS input {self.key!r} contains NaN or Inf")
        values = image.reshape(-1).contiguous()
        if values.numel() > 2**24:
            from monai.transforms.utils_pytorch_numpy_unification import percentile

            low = percentile(values, self.low)
            high = percentile(values, self.high)
        else:
            low = torch.quantile(values, self.low / 100.0)
            high = torch.quantile(values, self.high / 100.0)
        data[self.info_key] = {
            "policy": "percentile_to_minus1_1",
            "percentile_q": [self.low, self.high],
            "actual_input_low_high": [float(low.item()), float(high.item())],
            "actual_input_range": [
                float(values.min().item()),
                float(values.max().item()),
            ],
            "percentile_selection": "all_values",
            "output_range": [-1.0, 1.0],
            "clipped": True,
        }
        return data


class ValidatePreNormalizedIntensityd:
    """Validate a disk-standardized volume before normalization passthrough."""

    def __init__(self, key: str, *, tolerance: float = 0.05) -> None:
        self.key = key
        self.tolerance = float(tolerance)
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch

        image = data[self.key].float()
        if not torch.isfinite(image).all():
            raise ValueError(f"Pre-normalized input {self.key!r} contains NaN or Inf")
        minimum = float(image.min().item())
        maximum = float(image.max().item())
        low, high = (-1.0 - self.tolerance, 1.0 + self.tolerance)
        if minimum < low or maximum > high:
            raise ValueError(
                f"Pre-normalized input {self.key!r} must be approximately [-1,1] (tolerance={self.tolerance}), got range [{minimum}, {maximum}]"
            )
        data[self.key] = image
        data[f"{self.key}_normalization_info"] = {
            "policy": "pre_normalized_passthrough",
            "validated_range": [-1.0, 1.0],
            "range_tolerance": self.tolerance,
            "actual_input_range": [minimum, maximum],
            "output_range": [-1.0, 1.0],
            "clipped": False,
        }
        return data


def configured_full_size_hwd(data_cfg: Mapping[str, Any]) -> tuple[int, int, int]:
    value = data_cfg.get("full_spatial_size_hwd", (256, 256, 128))
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(
            f"data.full_spatial_size_hwd must contain H,W,D, got {value!r}"
        )
    shape = tuple((int(item) for item in value))
    if any((item <= 0 for item in shape)):
        raise ValueError(f"data.full_spatial_size_hwd must be positive, got {shape}")
    return shape


def brats_crop_source_modality(modalities: Sequence[str]) -> str:
    canonical = sorted(
        {
            canonical_modality_name(item)
            for item in modalities
            if canonical_modality_name(item)
        }
    )
    if not canonical:
        raise ValueError("BraTS crop-source selection requires at least one modality")
    return "t1c" if "t1c" in canonical else canonical[0]


def build_modality_full_volume_transform(
    dataset: str, data_cfg: Mapping[str, Any], modalities: Sequence[str]
):
    """Build one deterministic transform over every modality in a case.

    Coverage padding intentionally is not part of this transform. It is computed
    once by the cache generator after all modalities have been checked.
    """
    from monai.transforms import (
        Compose,
        CropForegroundd,
        DeleteItemsd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        Resized,
        ScaleIntensityRangePercentilesd,
    )
    from streamrefine.data.anatomy_mask import (
        BuildDeterministicAnatomyMaskd,
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
        canonical_anatomy_mask_policy,
    )

    dataset_key = normalize_dataset_name(dataset)
    canonical_modalities = [canonical_modality_name(item) for item in modalities]
    if not canonical_modalities or any((not item for item in canonical_modalities)):
        raise ValueError(f"modalities must be non-empty, got {modalities!r}")
    if len(set(canonical_modalities)) != len(canonical_modalities):
        raise ValueError(
            f"modalities contain aliases of the same modality: {modalities!r}"
        )
    image_keys = [modality_field_name(item) for item in canonical_modalities]
    modality_by_key = dict(zip(image_keys, canonical_modalities))
    foreground_key = "foreground"
    anatomy_mask_key = "anatomy_mask"
    norm_cfg = data_cfg.get("normalization", {})
    if not isinstance(norm_cfg, Mapping):
        raise ValueError("data.normalization must be a mapping")
    percentile_low = float(norm_cfg.get("percentile_low", 0.5))
    percentile_high = float(norm_cfg.get("percentile_high", 99.5))
    already_normalized = bool(
        data_cfg.get("input_already_normalized", False)
        or data_cfg.get("skip_intensity_normalization", False)
    )
    range_tolerance = float(data_cfg.get("pre_normalized_range_tolerance", 0.05))
    anatomy_mask_policy = canonical_anatomy_mask_policy(
        dataset_key, data_cfg.get("anatomy_mask_policy")
    )
    transforms: list[Any] = [
        LoadImaged(keys=image_keys),
        EnsureChannelFirstd(keys=image_keys),
        Orientationd(keys=image_keys, axcodes=str(data_cfg.get("orientation", "RAS"))),
    ]
    if already_normalized:
        transforms.extend(
            (
                ValidatePreNormalizedIntensityd(key, tolerance=range_tolerance)
                for key in image_keys
            )
        )
    if dataset_key == "brats24":
        from streamrefine.data.brats_affine import POLICY, ReconcileVerifiedBraTSAffined

        transforms.append(
            ReconcileVerifiedBraTSAffined(
                image_keys,
                modality_by_key,
                policy=str(data_cfg.get("brats_affine_policy", POLICY)),
            )
        )
        transforms.append(ValidateCommonGridd(image_keys))
        transforms.append(
            BuildDeterministicAnatomyMaskd(
                dataset=dataset_key,
                image_keys=image_keys,
                modality_by_key=modality_by_key,
                output_key=anatomy_mask_key,
                policy=anatomy_mask_policy,
                already_normalized=already_normalized,
            )
        )
        crop_source_modality = brats_crop_source_modality(canonical_modalities)
        crop_source_key = modality_field_name(crop_source_modality)
        crop_kwargs: dict[str, Any] = {}
        if already_normalized:
            crop_kwargs["select_fn"] = lambda image: image > -0.999
        transforms.append(
            CropForegroundd(
                keys=[*image_keys, anatomy_mask_key],
                source_key=crop_source_key,
                allow_smaller=True,
                **crop_kwargs,
            )
        )
        if not already_normalized:
            transforms.extend(
                (
                    RecordPercentileNormalizationParametersd(
                        key, low=percentile_low, high=percentile_high
                    )
                    for key in image_keys
                )
            )
            transforms.append(
                ScaleIntensityRangePercentilesd(
                    keys=image_keys,
                    lower=percentile_low,
                    upper=percentile_high,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                )
            )
        transforms.append(
            Resized(
                keys=[*image_keys, anatomy_mask_key],
                spatial_size=configured_full_size_hwd(data_cfg),
                mode=tuple([*("trilinear" for _ in image_keys), "nearest"]),
            )
        )
    else:
        crop_source_modality = "case_shared_union"
        transforms.extend(
            [
                MultiModalityForegroundMaskd(
                    image_keys,
                    modality_by_key,
                    foreground_key,
                    already_normalized=already_normalized,
                ),
                BuildDeterministicAnatomyMaskd(
                    dataset=dataset_key,
                    image_keys=image_keys,
                    modality_by_key=modality_by_key,
                    output_key=anatomy_mask_key,
                    policy=anatomy_mask_policy,
                    already_normalized=already_normalized,
                ),
                CropForegroundd(
                    keys=[*image_keys, foreground_key, anatomy_mask_key],
                    source_key=foreground_key,
                    allow_smaller=True,
                ),
                DeleteItemsd(keys=[foreground_key]),
            ]
        )
        if not already_normalized:
            for modality, key in zip(canonical_modalities, image_keys):
                transforms.append(
                    ModalityNormalizeIntensityd(
                        key,
                        modality,
                        low=percentile_low,
                        high=percentile_high,
                        max_quantile_samples=norm_cfg.get(
                            "max_quantile_samples", 1000000
                        ),
                    )
                )
    if already_normalized:
        normalization_info = {
            "policy": "pre_normalized_passthrough",
            "input_already_normalized": True,
            "validated_range": [-1.0, 1.0],
            "range_tolerance": range_tolerance,
            "output_range": [-1.0, 1.0],
            "per_modality": True,
        }
    elif dataset_key == "brats24":
        normalization_info = {
            "policy": "percentile_to_minus1_1",
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
            "output_range": [-1.0, 1.0],
            "per_modality": True,
        }
    else:
        normalization_info = {
            "policy": "modality_specific",
            "ct_cbct_hu_clip": [-1000.0, 1000.0],
            "mr_pet_positive_percentiles": [percentile_low, percentile_high],
            "output_range": [-1.0, 1.0],
            "per_modality": True,
        }
    transforms.append(EnsureTyped(keys=[*image_keys, anatomy_mask_key]))
    transform = Compose(transforms)
    transform.streamrefine_dataset = dataset_key
    transform.streamrefine_modalities = tuple(canonical_modalities)
    transform.streamrefine_crop_source_modality = crop_source_modality
    transform.streamrefine_normalization_info = normalization_info
    transform.streamrefine_anatomy_mask_key = anatomy_mask_key
    transform.streamrefine_anatomy_mask_policy = anatomy_mask_policy
    transform.streamrefine_anatomy_mask_contract = anatomy_mask_contract(
        dataset_key, anatomy_mask_policy
    )
    transform.streamrefine_anatomy_mask_contract_fingerprint = (
        anatomy_mask_contract_fingerprint(dataset_key, anatomy_mask_policy)
    )
    return transform


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def affine_to_spacing_hwd(affine: Any) -> list[float] | None:
    if affine is None:
        return None
    import numpy as np

    array = np.asarray(_plain(affine), dtype=float)
    if array.ndim == 3:
        array = array[0]
    if array.shape != (4, 4):
        return None
    spacing = [float(np.linalg.norm(array[:3, axis])) for axis in range(3)]
    return (
        spacing if all((math.isfinite(item) and item > 0 for item in spacing)) else None
    )


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    meta = getattr(tensor, "meta", None)
    if not isinstance(meta, Mapping):
        meta = {}
    affine = getattr(tensor, "affine", None)
    if affine is None:
        affine = meta.get("affine", meta.get("original_affine"))
    return {
        "affine": _plain(affine),
        "spacing": affine_to_spacing_hwd(affine),
        "meta": {
            key: _plain(value)
            for key, value in meta.items()
            if key
            in {
                "filename_or_obj",
                "original_affine",
                "spatial_shape",
                "original_channel_dim",
            }
        },
    }


def load_and_preprocess_modality_case(
    transform: Any, case_record: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load all case modalities together and verify the resulting common grid."""
    import torch

    raw_modalities = case_record.get("modalities")
    if not isinstance(raw_modalities, Mapping) or not raw_modalities:
        raise ValueError("case_record.modalities must be a non-empty mapping")
    modalities = [canonical_modality_name(item) for item in raw_modalities]
    transform_modalities = tuple(
        getattr(transform, "streamrefine_modalities", modalities)
    )
    if set(modalities) != set(transform_modalities):
        raise ValueError(
            f"Transform modalities {transform_modalities!r} do not match case modalities {modalities!r}."
        )
    data = {
        modality_field_name(modality): str(raw_modalities[raw_name])
        for raw_name, modality in zip(raw_modalities, modalities)
    }
    data["streamrefine_case_id"] = str(case_record.get("case_id", ""))
    transformed = transform(data)
    volumes: dict[str, Any] = {}
    metadata_by_modality: dict[str, dict[str, Any]] = {}
    normalization_by_modality: dict[str, dict[str, Any]] = {}
    reference_shape: tuple[int, int, int] | None = None
    reference_affine: Any = None
    reference_modality = ""
    for modality in transform_modalities:
        key = modality_field_name(modality)
        value = transformed[key]
        volume = torch.as_tensor(value).float().contiguous()
        if volume.ndim != 4 or int(volume.shape[0]) != 1:
            raise ValueError(
                f"Expected {modality} [1,H,W,D], got {tuple(volume.shape)} for case {case_record.get('case_id')!r}."
            )
        shape = tuple((int(item) for item in volume.shape[1:]))
        metadata = tensor_metadata(value)
        if reference_shape is None:
            reference_shape = shape
            reference_affine = metadata["affine"]
            reference_modality = modality
        else:
            if shape != reference_shape:
                raise ValueError(
                    f"Shared preprocessing produced different shapes for case {case_record.get('case_id')!r}: {reference_modality}={reference_shape}, {modality}={shape}."
                )
            if not _affines_close(reference_affine, metadata["affine"]):
                raise ValueError(
                    f"Shared preprocessing produced different affines for case {case_record.get('case_id')!r}: {reference_modality} vs {modality}."
                )
        volumes[modality] = volume
        metadata_by_modality[modality] = metadata
        normalization = transformed.get(f"{key}_normalization_info")
        if not isinstance(normalization, Mapping):
            raise RuntimeError(
                f"Normalization metadata was not recorded for {modality} in case {case_record.get('case_id')!r}."
            )
        normalization_by_modality[modality] = dict(normalization)
    assert reference_shape is not None
    anatomy_mask_key = str(
        getattr(transform, "streamrefine_anatomy_mask_key", "anatomy_mask")
    )
    if anatomy_mask_key not in transformed:
        raise RuntimeError(
            "Medical preprocessing did not return the required deterministic anatomy mask"
        )
    anatomy_mask_image_hwd = torch.as_tensor(transformed[anatomy_mask_key])
    if anatomy_mask_image_hwd.ndim != 4 or int(anatomy_mask_image_hwd.shape[0]) != 1:
        raise ValueError(
            f"Preprocessed anatomy mask must be [1,H,W,D], got {tuple(anatomy_mask_image_hwd.shape)}"
        )
    if (
        tuple((int(item) for item in anatomy_mask_image_hwd.shape[1:]))
        != reference_shape
    ):
        raise ValueError(
            f"Preprocessed anatomy mask is not aligned with the modalities: mask={tuple(anatomy_mask_image_hwd.shape[1:])}, image={reference_shape}"
        )
    anatomy_mask_image_hwd = (anatomy_mask_image_hwd > 0.5).bool().contiguous()
    if not bool(anatomy_mask_image_hwd.any()):
        raise ValueError("Preprocessed deterministic anatomy mask is empty")
    first_metadata = metadata_by_modality[reference_modality]
    meta = {
        "case_id": str(case_record.get("case_id", "")),
        "group_id": str(case_record.get("group_id", "")),
        "dataset": str(case_record.get("dataset", "")),
        "dataset_key": str(case_record.get("dataset_key", "")),
        "split": str(case_record.get("split", "")),
        "available_modalities": list(transform_modalities),
        "modality_image_paths": {
            canonical_modality_name(name): str(path)
            for name, path in raw_modalities.items()
        },
        "original_shape_hwd": list(reference_shape),
        "spacing": first_metadata["spacing"],
        "affine": first_metadata["affine"],
        "normalization_info": dict(
            getattr(transform, "streamrefine_normalization_info", {})
        ),
        "normalization_by_modality": normalization_by_modality,
        "anatomy_mask_image_hwd": anatomy_mask_image_hwd,
        "anatomy_mask_policy": str(
            getattr(transform, "streamrefine_anatomy_mask_policy", "")
        ),
        "anatomy_mask_contract": dict(
            getattr(transform, "streamrefine_anatomy_mask_contract", {})
        ),
        "anatomy_mask_contract_fingerprint": str(
            getattr(transform, "streamrefine_anatomy_mask_contract_fingerprint", "")
        ),
        "crop_source_modality": str(
            getattr(transform, "streamrefine_crop_source_modality", "")
        ),
        "metadata_by_modality": metadata_by_modality,
        "affine_corrections": _plain(
            transformed.get("streamrefine_affine_corrections", [])
        ),
        "cohort": str(case_record.get("cohort", "")),
        "task": str(case_record.get("task", "")),
        "anatomy": str(case_record.get("anatomy", "")),
        "tracer": str(case_record.get("tracer", "")),
    }
    return (volumes, meta)


def safe_modality_cache_stem(case_record: Mapping[str, Any], modality: str) -> str:
    parts = [
        str(case_record.get("task", "")),
        str(case_record.get("anatomy", "")),
        str(case_record.get("cohort", "")),
        str(case_record.get("tracer", "")),
        str(case_record.get("case_id", "case")),
        canonical_modality_name(modality),
    ]
    stem = "_".join((part for part in parts if part))
    return re.sub("[^A-Za-z0-9_.=-]+", "_", stem).strip("_") or "case"


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path | str) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


ensure_tokenizer_path = ensure_streamrefine_tokenizer_path
