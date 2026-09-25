from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

THIS_FILE = Path(__file__).resolve()
AR_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = AR_ROOT.parent
if str(AR_ROOT) not in sys.path:
    sys.path.insert(0, str(AR_ROOT))
DATASET_CONFIGS = {
    "brats24": "brats24_vidtok_kl16_mean.yaml",
    "synthrad": "synthrad_vidtok_kl16_mean.yaml",
    "autopet": "autopet_vidtok_kl16_mean.yaml",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute deterministic per-modality full-volume VidTok-KL posterior-mean caches for StreamRefine."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS),
        default=None,
        help="Dataset selector. Inferred from --config when omitted; otherwise defaults to brats24.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Cache YAML. By default the dataset-specific YAML under ar/configs/cache is used.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "validation", "all"),
        default=None,
        help="Split selector. Defaults to data.split in the chosen YAML (or all if absent).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved paths without reading images or loading VidTok.",
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help="Optional comma-separated modality filter. Dataset defaults are used when omitted.",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, default=None
    )
    parser.add_argument(
        "--manifest-out", "--manifest_out", dest="manifest_out", type=Path, default=None
    )
    parser.add_argument(
        "--pair-manifest-out",
        "--pair_manifest_out",
        dest="pair_manifest_out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--failure-log", "--failure_log", dest="failure_log", type=Path, default=None
    )
    parser.add_argument(
        "--vidtok-root", "--vidtok_root", dest="vidtok_root", type=Path, default=None
    )
    parser.add_argument(
        "--vidtok-config",
        "--vidtok_config",
        dest="vidtok_config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--vidtok-ckpt",
        "--vidtok_ckpt",
        "--ckpt",
        dest="vidtok_ckpt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, for example cuda, cuda:1, or cpu."
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=None
    )
    parser.add_argument(
        "--patch-size-hwd",
        "--patch_size_hwd",
        dest="patch_size_hwd",
        nargs=3,
        type=int,
        default=None,
    )
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument(
        "--compression-hwd",
        "--compression_hwd",
        dest="compression_hwd",
        nargs=3,
        type=int,
        default=None,
    )
    parser.add_argument(
        "--pad-mode", choices=("auto", "no_pad", "coverage"), default=None
    )
    parser.add_argument("--importance-mode", choices=("hann", "gaussian"), default=None)
    parser.add_argument("--importance-floor", type=float, default=None)
    parser.add_argument("--gaussian-sigma-scale", type=float, default=None)
    parser.add_argument("--rebuild-datalists", action="store_true")
    parser.add_argument(
        "--start-index", "--start_index", dest="start_index", type=int, default=0
    )
    parser.add_argument(
        "--max-cases", "--max_cases", dest="max_cases", type=int, default=None
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse only existing caches whose schema and tokenizer hash validate.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace existing caches atomically. This takes precedence over --resume.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record a failed case and continue with the remaining cases.",
    )
    parser.add_argument(
        "--verify-determinism",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Encode the first pending tile twice and require matching posterior means.",
    )
    parser.add_argument(
        "--verify-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode the first pending mean tile to verify the cache/decoder contract.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve records, verify selected input files, and print a preview without loading VidTok.",
    )
    return parser.parse_args(argv)


def _default_config_path(dataset: str) -> Path:
    return AR_ROOT / "configs" / "cache" / DATASET_CONFIGS[dataset]


def _dataset_slug(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "brats": "brats24",
        "brats24": "brats24",
        "brats2024": "brats24",
        "synthrad": "synthrad",
        "synthrad2025": "synthrad",
        "autopet": "autopet",
        "autopetiii": "autopet",
        "autopet3": "autopet",
    }
    try:
        return aliases[key]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset name: {value!r}") from error


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to read cache configs. Install pyyaml in the project environment."
        ) from error
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Cache config does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    value["_config_path"] = str(resolved)
    return value


def _resolve_path(
    value: str | os.PathLike[str] | None, config_path: Path
) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _is_placeholder(value: Any) -> bool:
    return str(value or "").strip().upper().startswith("REPLACE_WITH_")


def _as_hwd(
    value: Sequence[int] | None, default: Sequence[int], name: str
) -> tuple[int, int, int]:
    selected = default if value is None else value
    if len(selected) != 3:
        raise ValueError(f"{name} must have exactly three values, got {selected!r}")
    result = tuple((int(item) for item in selected))
    if any((item <= 0 for item in result)):
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _config_value(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    current = cfg.get(section, {})
    return current.get(key, default) if isinstance(current, dict) else default


def _numeric_input_range(value: Any) -> tuple[float, float] | None:
    if (
        isinstance(value, str)
        or not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        return None
    lower, upper = (float(value[0]), float(value[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or (not lower < upper):
        raise ValueError(f"Invalid normalization/input range: {value!r}")
    return (lower, upper)


def _preprocessing_output_range(
    preprocessing_cfg: dict[str, Any],
) -> tuple[float, float]:
    data_cfg = preprocessing_cfg.get("data", {})
    normalization = (
        data_cfg.get("normalization", {}) if isinstance(data_cfg, dict) else {}
    )
    candidates = [
        normalization.get("output_range") if isinstance(normalization, dict) else None,
        _config_value(preprocessing_cfg, "vidtok", "input_range", None),
    ]
    for candidate in candidates:
        parsed = _numeric_input_range(candidate)
        if parsed is not None:
            return parsed
    normalization_type = (
        str(normalization.get("type", "")).strip().lower()
        if isinstance(normalization, dict)
        else ""
    )
    if "minus1_1" in normalization_type or "minus_one_to_one" in normalization_type:
        return (-1.0, 1.0)
    raise ValueError(
        "Could not determine the medical preprocessing output range. Set data.normalization.output_range or vidtok.input_range in the preprocessing config."
    )


def _resolve_padding_value(
    cfg: dict[str, Any], preprocessing_cfg: dict[str, Any]
) -> tuple[float, tuple[float, float], str]:
    """Bind boundary padding to the medical fine-tuning input convention."""
    preprocessing_range = _preprocessing_output_range(preprocessing_cfg)
    configured = float(
        _config_value(cfg, "patching", "padding_value", preprocessing_range[0])
    )
    if not math.isfinite(configured):
        raise ValueError("patching.padding_value must be finite")
    from_preprocessing = bool(
        _config_value(cfg, "data", "padding_value_from_preprocessing", False)
    )
    outer_range = _numeric_input_range(
        _config_value(cfg, "vidtok", "input_range", None)
    )
    if outer_range is not None and any(
        (
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-08)
            for a, b in zip(outer_range, preprocessing_range)
        )
    ):
        raise ValueError(
            f"Cache vidtok.input_range disagrees with the medical preprocessing config: cache={outer_range}, preprocessing={preprocessing_range}"
        )
    if not from_preprocessing:
        return (configured, preprocessing_range, "patching.padding_value")
    expected = float(preprocessing_range[0])
    if not math.isclose(configured, expected, rel_tol=0.0, abs_tol=1e-08):
        raise ValueError(
            f"data.padding_value_from_preprocessing=true requires patching.padding_value to equal the preprocessing lower bound: configured={configured}, expected={expected}"
        )
    return (expected, preprocessing_range, "medical_preprocessing_output_lower_bound")


def _check_input_paths(case_records: Sequence[dict[str, Any]]) -> int:
    """Fail before loading VidTok or touching manifests if input paths are stale."""
    checked = 0
    missing = 0
    examples: list[str] = []
    for case in case_records:
        for modality, raw_path in case["modalities"].items():
            path = Path(str(raw_path)).expanduser().resolve()
            checked += 1
            if not path.is_file():
                missing += 1
                if len(examples) < 5:
                    examples.append(
                        f"  case={case.get('group_id') or case.get('case_id')} modality={modality}: {path}"
                    )
    if missing:
        raise FileNotFoundError(
            f"Input path preflight failed: {missing}/{checked} volume path(s) do not exist.\n"
            + "\n".join(examples)
            + ""
        )
    return checked


def _report_case_failure(
    failure: dict[str, Any], count: int, failure_log: Path
) -> None:
    """Expose the original error promptly while bounding terminal output."""
    if count <= 3:
        print(
            f"\n[Case failure {count}] {failure['group_id'] or failure['case_id']}\n{failure['traceback']}",
            file=sys.stderr,
            flush=True,
        )
    elif count == 4:
        print(
            f"[Case failures] Further tracebacks are suppressed; full report: {failure_log}",
            file=sys.stderr,
            flush=True,
        )


def _safe_component(value: str) -> str:
    text = re.sub("[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return text or "case"


def _case_cache_dir(output_dir: Path, case_record: dict[str, Any]) -> Path:
    return output_dir / _safe_component(
        str(case_record.get("group_id") or case_record.get("case_id") or "case")
    )


@contextlib.contextmanager
def _exclusive_case_writer(case_dir: Path):
    """Enforce the cache-v2 one-writer-per-case rule across processes."""
    case_dir.mkdir(parents=True, exist_ok=True)
    lock_path = case_dir / ".streamrefine_case_writer.lock"
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            owner = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "<unreadable>"
        raise RuntimeError(
            f"Case cache is already reserved by another writer: {case_dir}; lock={lock_path}, owner={owner}. If the job crashed, verify no writer is active before removing the stale lock."
        ) from exc
    try:
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_unix": time.time(),
                "case_dir": str(case_dir),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join((json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    _atomic_write_text(path, payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file_local(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _python_source_tree_fingerprint(roots: dict[str, Path]) -> dict[str, Any]:
    """Hash executed tokenizer Python sources even when the tree has no Git metadata."""
    root_summaries: dict[str, dict[str, Any]] = {}
    combined_entries: dict[str, str] = {}
    for label, root in sorted(roots.items()):
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Tokenizer source root does not exist: {resolved}")
        entries = {
            path.relative_to(resolved).as_posix(): _sha256_file_local(path)
            for path in sorted(resolved.rglob("*.py"), key=lambda item: item.as_posix())
            if path.is_file()
        }
        if not entries:
            raise RuntimeError(
                f"Tokenizer source root contains no Python files: {resolved}"
            )
        root_summaries[label] = {
            "file_count": len(entries),
            "tree_sha256": _stable_json_sha256(entries),
        }
        combined_entries.update(
            {f"{label}/{name}": digest for name, digest in entries.items()}
        )
    return {
        "policy": "recursive_python_source_sha256_v1",
        "roots": root_summaries,
        "file_count": len(combined_entries),
        "tree_sha256": _stable_json_sha256(combined_entries),
    }


def _canonical_path_text(value: Any) -> str:
    if value is None or not str(value).strip():
        return ""
    return os.path.normcase(str(Path(str(value)).expanduser().resolve()))


def _input_file_signatures(case_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for raw_modality, raw_path in dict(case_record.get("modalities", {})).items():
        modality = str(raw_modality).strip().lower()
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Input volume for {modality} does not exist: {path}"
            )
        stat = path.stat()
        signatures[modality] = {
            "resolved_path": os.path.normcase(str(path)),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return signatures


def _cache_file_guard(path: Path) -> dict[str, Any]:
    """Bind a manifest row to the exact serialized cache file on disk."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Serialized cache does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file_local(resolved),
    }


def _merge_jsonl_rows(
    path: Path, new_rows: Sequence[dict[str, Any]], *, key_fields: Sequence[str]
) -> None:
    """Atomically merge incremental manifest rows, replacing equal logical keys."""
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in [*_read_jsonl(path), *new_rows]:
        key = tuple((str(row.get(field, "")) for field in key_fields))
        if not any(key):
            raise ValueError(
                f"Cannot merge {path}: row has no logical key fields {tuple(key_fields)}"
            )
        merged[key] = row
    ordered = [merged[key] for key in sorted(merged)]
    _write_jsonl(path, ordered)


def _git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _translation_pairs(cfg: dict[str, Any], dataset: str) -> list[tuple[str, str]]:
    value = cfg.get("translation_pairs", {})
    if isinstance(value, dict):
        selected: Any = None
        for key, candidate in value.items():
            if str(key).strip().lower() == dataset:
                selected = candidate
                break
        value = selected or []
    if not isinstance(value, list):
        raise ValueError("translation_pairs must be a list or a dataset-keyed mapping")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid translation pair: {item!r}")
        pairs.append((str(item[0]).lower(), str(item[1]).lower()))
    return pairs


def _manifest_row_from_cache(
    cache_path: Path, cache: dict[str, Any], *, status: str
) -> dict[str, Any]:
    latent = cache["latent_mu"]
    valid = cache["valid_mask_latent"]
    anatomy = cache["anatomy_mask_latent_hard"]
    return {
        "status": "ready",
        "write_action": status,
        "cache_path": str(cache_path.resolve()),
        "cache_file_guard": _cache_file_guard(cache_path),
        "dataset": str(cache.get("dataset", "")),
        "case_id": str(cache.get("case_id", "")),
        "group_id": str(cache.get("group_id", "")),
        "split": str(cache.get("split", "")),
        "modality": str(cache.get("modality", "")),
        "available_modalities": list(cache.get("available_modalities", [])),
        "modality_image_paths": dict(cache.get("modality_image_paths", {})),
        "input_file_signatures": dict(cache.get("input_file_signatures", {})),
        "latent_shape_cdhw": [int(item) for item in latent.shape],
        "original_shape_hwd": list(cache.get("original_shape_hwd", [])),
        "padded_shape_hwd": list(cache.get("padded_shape_hwd", [])),
        "num_patches": len(cache.get("patch_index_table", [])),
        "valid_ratio": float(valid.float().mean().item()),
        "anatomy_mask_ratio": float(anatomy.float().mean().item()),
        "pir_support_ratio": float((valid & anatomy).float().mean().item()),
        "anatomy_mask_policy": str(
            cache.get("anatomy_mask_contract", {}).get("policy", "")
        ),
        "anatomy_mask_contract_fingerprint": str(
            cache.get("anatomy_mask_contract_fingerprint", "")
        ),
        "anatomy_mask_sha256": str(cache.get("anatomy_mask_sha256", "")),
        "tokenizer_checkpoint_sha256": str(
            cache.get("tokenizer_checkpoint_sha256", "")
        ),
        "tokenizer_config_sha256": str(cache.get("tokenizer_config_sha256", "")),
        "generation_contract_sha256": str(cache.get("generation_contract_sha256", "")),
        "input_snapshot_sha256": str(cache.get("input_snapshot_sha256", "")),
        "cache_version": str(cache.get("cache_version", "")),
        "cohort": str(cache.get("cohort", "")),
        "task": str(cache.get("task", "")),
        "anatomy": str(cache.get("anatomy", "")),
        "tracer": str(cache.get("tracer", "")),
        "crop_source_modality": str(cache.get("crop_source_modality", "")),
    }


PAIR_SHARED_MANIFEST_FIELDS = (
    "cache_version",
    "tokenizer_checkpoint_sha256",
    "tokenizer_config_sha256",
    "generation_contract_sha256",
    "anatomy_mask_policy",
    "anatomy_mask_contract_fingerprint",
    "anatomy_mask_sha256",
    "input_snapshot_sha256",
    "available_modalities",
    "modality_image_paths",
    "input_file_signatures",
    "crop_source_modality",
    "original_shape_hwd",
    "padded_shape_hwd",
    "latent_shape_cdhw",
)


def _pair_manifest_mismatches(
    source_row: dict[str, Any], target_row: dict[str, Any]
) -> list[str]:
    return [
        field
        for field in PAIR_SHARED_MANIFEST_FIELDS
        if source_row.get(field) != target_row.get(field)
    ]


def _build_pair_rows(
    modality_rows: Sequence[dict[str, Any]], pairs: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in modality_rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("group_id", "")),
            str(row.get("split", "")),
        )
        grouped.setdefault(key, {})[str(row.get("modality", "")).lower()] = row
    result: list[dict[str, Any]] = []
    for (dataset, group_id, split), by_modality in sorted(grouped.items()):
        for source, target in pairs:
            if source not in by_modality or target not in by_modality:
                continue
            source_row = by_modality[source]
            target_row = by_modality[target]
            if _pair_manifest_mismatches(source_row, target_row):
                continue
            result.append(
                {
                    "status": "ready",
                    "dataset": dataset,
                    "case_id": source_row.get("case_id", ""),
                    "group_id": group_id,
                    "split": split,
                    "source_modality": source,
                    "target_modality": target,
                    "source_cache": source_row["cache_path"],
                    "target_cache": target_row["cache_path"],
                    "source_cache_file_guard": source_row.get("cache_file_guard"),
                    "target_cache_file_guard": target_row.get("cache_file_guard"),
                    "tokenizer_checkpoint_sha256": source_row.get(
                        "tokenizer_checkpoint_sha256", ""
                    ),
                    "tokenizer_config_sha256": source_row.get(
                        "tokenizer_config_sha256", ""
                    ),
                    "generation_contract_sha256": source_row.get(
                        "generation_contract_sha256", ""
                    ),
                    "anatomy_mask_policy": source_row.get("anatomy_mask_policy", ""),
                    "anatomy_mask_contract_fingerprint": source_row.get(
                        "anatomy_mask_contract_fingerprint", ""
                    ),
                    "anatomy_mask_sha256": source_row.get("anatomy_mask_sha256", ""),
                    "input_snapshot_sha256": source_row.get(
                        "input_snapshot_sha256", ""
                    ),
                    "available_modalities": source_row.get("available_modalities", []),
                    "modality_image_paths": source_row.get("modality_image_paths", {}),
                    "input_file_signatures": source_row.get(
                        "input_file_signatures", {}
                    ),
                    "crop_source_modality": source_row.get("crop_source_modality", ""),
                    "cohort": source_row.get("cohort", ""),
                    "task": source_row.get("task", ""),
                    "anatomy": source_row.get("anatomy", ""),
                    "tracer": source_row.get("tracer", ""),
                    "pair_id": f"{group_id}:{source}->{target}",
                }
            )
    return result


def _build_missing_pair_rows(
    modality_rows: Sequence[dict[str, Any]], pairs: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Report unavailable configured pairs without fabricating cache endpoints."""
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in modality_rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("group_id", "")),
            str(row.get("split", "")),
        )
        grouped.setdefault(key, {})[str(row.get("modality", "")).lower()] = row
    result: list[dict[str, Any]] = []
    for (dataset, group_id, split), by_modality in sorted(grouped.items()):
        example = next(iter(by_modality.values()))
        for source, target in pairs:
            missing = [name for name in (source, target) if name not in by_modality]
            mismatched_fields: list[str] = []
            if not missing:
                mismatched_fields = _pair_manifest_mismatches(
                    by_modality[source], by_modality[target]
                )
            if not missing and (not mismatched_fields):
                continue
            result.append(
                {
                    "status": "skipped_missing_modality"
                    if missing
                    else "skipped_incompatible_cache_provenance",
                    "dataset": dataset,
                    "case_id": example.get("case_id", ""),
                    "group_id": group_id,
                    "split": split,
                    "source_modality": source,
                    "target_modality": target,
                    "available_modalities": sorted(by_modality),
                    "missing_modalities": missing,
                    "mismatched_shared_fields": mismatched_fields,
                    "pair_id": f"{group_id}:{source}->{target}",
                }
            )
    return result


def _write_manifests(
    *,
    rows: Sequence[dict[str, Any]],
    translation_pairs: Sequence[tuple[str, str]],
    output_dir: Path,
    requested_split: str,
    manifest_out: Path | None,
    pair_manifest_out: Path | None,
    write_split_specific: bool = True,
    merge_existing: bool = False,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    split_slug = "val" if requested_split == "validation" else requested_split
    modality_path = manifest_out or output_dir / (
        "cache_manifest.jsonl"
        if split_slug == "all"
        else f"cache_manifest_{split_slug}.jsonl"
    )
    pair_path = pair_manifest_out or output_dir / (
        "pair_manifest.jsonl"
        if split_slug == "all"
        else f"pair_manifest_{split_slug}.jsonl"
    )
    if merge_existing:
        _merge_jsonl_rows(
            modality_path, rows, key_fields=("dataset", "group_id", "split", "modality")
        )
    else:
        _write_jsonl(modality_path, rows)
    materialized_rows = _read_jsonl(modality_path)
    materialized_pair_rows = _build_pair_rows(materialized_rows, translation_pairs)
    _write_jsonl(pair_path, materialized_pair_rows)
    if split_slug == "all" and write_split_specific:
        for name in ("train", "val"):
            selected_rows = [
                row
                for row in materialized_rows
                if str(row.get("split", "")).lower() == name
            ]
            selected_pairs = [
                row
                for row in materialized_pair_rows
                if str(row.get("split", "")).lower() == name
            ]
            _write_jsonl(output_dir / f"cache_manifest_{name}.jsonl", selected_rows)
            _write_jsonl(output_dir / f"pair_manifest_{name}.jsonl", selected_pairs)
    return (modality_path, pair_path, materialized_rows, materialized_pair_rows)


def _prepare_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    from streamrefine.paths import (
        resolve_config_path,
        resolve_runtime_paths,
        RELEASE_ROOT,
    )

    selected_from_cli = _dataset_slug(args.dataset) if args.dataset else None
    config_path = resolve_config_path(
        args.config or _default_config_path(selected_from_cli or "brats24")
    ).resolve()
    cfg = resolve_runtime_paths(_load_yaml(config_path))
    for name in (
        "output_dir",
        "manifest_out",
        "pair_manifest_out",
        "failure_log",
        "vidtok_root",
        "vidtok_config",
        "vidtok_ckpt",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    configured_value = _config_value(
        cfg, "data", "dataset", selected_from_cli or "brats24"
    )
    configured_dataset = _dataset_slug(configured_value)
    dataset = selected_from_cli or configured_dataset
    if configured_dataset != dataset:
        raise ValueError(
            f"--dataset={dataset!r} does not match data.dataset={configured_value!r} in {config_path}"
        )
    cfg.setdefault("data", {})["dataset"] = dataset
    if _is_placeholder(cfg.get("cache", {}).get("output_dir")) or not cfg.get(
        "cache", {}
    ).get("output_dir"):
        cfg.setdefault("cache", {})["output_dir"] = str(
            RELEASE_ROOT / "result" / "streamrefine_latents" / dataset
        )
    cfg.setdefault("vidtok", {})
    path_overrides = {
        "repo_root": args.vidtok_root,
        "config_path": args.vidtok_config,
        "ckpt_path": args.vidtok_ckpt,
    }
    for key, override in path_overrides.items():
        if override is not None:
            resolved = override.expanduser().resolve()
        else:
            resolved = _resolve_path(cfg["vidtok"].get(key), config_path)
        if resolved is None:
            raise ValueError(f"vidtok.{key} is required")
        cfg["vidtok"][key] = str(resolved)
    return (cfg, config_path, dataset)


def _load_existing_cache(
    path: Path,
    *,
    checkpoint_sha256: str,
    tokenizer_config_sha256: str,
    generation_contract_sha256: str,
    overwrite: bool,
    resume: bool,
    generation_contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not path.exists() or overwrite:
        return None
    if not resume:
        raise FileExistsError(
            f"Cache already exists; pass --overwrite or --resume: {path}"
        )
    import torch
    from streamrefine.tokenizer.cache_schema import validate_cache_dict

    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_cache_dict(
        cache,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_config_sha256=tokenizer_config_sha256,
        expected_generation_contract_sha256=generation_contract_sha256,
    )
    return cache


def _validate_existing_cache_identity(
    cache: dict[str, Any],
    *,
    case_record: dict[str, Any],
    dataset: str,
    modality: str,
    cache_path: Path,
    input_file_signatures: dict[str, dict[str, Any]],
) -> None:
    """Reject a valid cache copied into the wrong case/modality destination."""
    expected_case = str(case_record.get("case_id", ""))
    expected_group = str(case_record.get("group_id", ""))
    expected_split = str(case_record.get("split", "")).strip().lower()
    actual_split = str(cache.get("split", "")).strip().lower()
    expected_modality = str(modality).strip().lower()
    checks = {
        "dataset": (_dataset_slug(cache.get("dataset")), _dataset_slug(dataset)),
        "case_id": (str(cache.get("case_id", "")), expected_case),
        "group_id": (str(cache.get("group_id", "")), expected_group),
        "split": (actual_split, expected_split),
        "modality": (str(cache.get("modality", "")).strip().lower(), expected_modality),
    }
    mismatches = [
        f"{field}: cache={actual!r}, expected={expected!r}"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    current_modalities = {
        str(name).strip().lower(): _canonical_path_text(path)
        for name, path in dict(case_record.get("modalities", {})).items()
    }
    cached_modalities = {
        str(name).strip().lower(): _canonical_path_text(path)
        for name, path in dict(cache.get("modality_image_paths", {})).items()
    }
    if cached_modalities != current_modalities:
        mismatches.append(
            f"modality_image_paths/available preprocessing modalities changed: cache={cached_modalities!r}, expected={current_modalities!r}"
        )
    cached_signatures = cache.get("input_file_signatures")
    if (
        not isinstance(cached_signatures, dict)
        or cached_signatures != input_file_signatures
    ):
        mismatches.append(
            f"input file size/mtime/path signatures changed: cache={cached_signatures!r}, expected={input_file_signatures!r}"
        )
    expected_image = current_modalities.get(expected_modality, "")
    actual_image = _canonical_path_text(cache.get("modality_image_path"))
    if actual_image != expected_image:
        mismatches.append(
            f"modality_image_path: cache={actual_image!r}, expected={expected_image!r}"
        )
    if mismatches:
        raise ValueError(
            f"Refusing to resume mismatched cache at {cache_path}: "
            + "; ".join(mismatches)
        )


def _normalization_info(
    cfg: dict[str, Any],
    modality: str,
    padding_value: float,
    preprocessing_info: Any = None,
) -> dict[str, Any]:
    normalization = _config_value(cfg, "data", "normalization", {})
    result = dict(preprocessing_info) if isinstance(preprocessing_info, dict) else {}
    configured_range = _config_value(cfg, "vidtok", "input_range", [-1, 1])
    if isinstance(configured_range, str):
        if configured_range.strip().lower() != "same_as_finetune":
            raise ValueError(
                f"Unsupported vidtok.input_range sentinel: {configured_range!r}"
            )
        input_range = result.get("output_range", [-1.0, 1.0])
    else:
        input_range = configured_range
    if not isinstance(input_range, (list, tuple)) or len(input_range) != 2:
        raise ValueError(
            f"vidtok.input_range must contain two values, got {input_range!r}"
        )
    result.update(
        {
            "input_range": [float(input_range[0]), float(input_range[1])],
            "configured_policy": normalization,
            "padding_value": float(padding_value),
            "modality": modality,
            "preprocessing_scope": "case_shared_all_modalities",
        }
    )
    return result


def _process_case_unlocked(
    *,
    case_record: dict[str, Any],
    cfg: dict[str, Any],
    dataset: str,
    output_dir: Path,
    encoder: Any,
    device: Any,
    checkpoint_sha256: str,
    tokenizer_config_sha256: str,
    generation_contract: dict[str, Any],
    generation_contract_sha256: str,
    tokenizer_git_commit: str | None,
    patch_size_hwd: tuple[int, int, int],
    stride_hwd: tuple[int, int, int],
    compression_hwd: tuple[int, int, int],
    minimum_image_shape_hwd: tuple[int, int, int],
    pad_mode: str,
    padding_value: float,
    include_last: bool,
    importance_mode: str,
    importance_floor: float,
    gaussian_sigma_scale: float,
    overlap: float,
    precision: str,
    batch_size: int,
    num_workers: int,
    resume: bool,
    overwrite: bool,
    verify_determinism: bool,
    verify_decode: bool,
    verification_state: dict[str, Any],
    selected_modalities: set[str] | None,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from streamrefine.data.latent_generation_dataset import ModalityPatchDataset
    from streamrefine.data.preprocess_medical import (
        build_modality_full_volume_transform,
        load_and_preprocess_modality_case,
    )
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
        compose_pir_mask_latent_hard,
        downsample_anatomy_mask_to_latent,
    )
    from streamrefine.tokenizer.cache_schema import (
        CACHE_VERSION,
        anatomy_mask_sha256,
        atomic_torch_save,
        input_snapshot_sha256,
        validate_cache_dict,
    )
    from streamrefine.tokenizer.grid_patch_with_coords import (
        build_patch_index_table,
        compute_padded_shape_hwd,
        pad_chwd_to_shape,
        padding_info,
    )
    from streamrefine.tokenizer.valid_mask import (
        build_image_valid_mask_hwd,
        downsample_valid_mask_to_latent,
    )
    from streamrefine.tokenizer.weighted_stitch import (
        accumulate_patch_mean,
        allocate_full_accumulators,
        finalize_full_mean,
        make_importance_map_dhw,
    )

    all_modalities = [
        str(item).lower() for item in case_record.get("modalities", {}).keys()
    ]
    if not all_modalities:
        raise ValueError(f"Case has no modalities: {case_record}")
    modalities = [
        modality
        for modality in all_modalities
        if selected_modalities is None or modality in selected_modalities
    ]
    if not modalities:
        return []
    input_file_signatures = _input_file_signatures(case_record)
    case_dir = _case_cache_dir(output_dir, case_record)
    existing_by_modality: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for modality in modalities:
        path = case_dir / f"{_safe_component(modality.lower())}.pt"
        cache = _load_existing_cache(
            path,
            checkpoint_sha256=checkpoint_sha256,
            tokenizer_config_sha256=tokenizer_config_sha256,
            generation_contract_sha256=generation_contract_sha256,
            overwrite=overwrite,
            resume=resume,
            generation_contract=generation_contract,
        )
        if cache is None:
            pending.append(modality)
        else:
            _validate_existing_cache_identity(
                cache,
                case_record=case_record,
                dataset=dataset,
                modality=modality,
                cache_path=path,
                input_file_signatures=input_file_signatures,
            )
            existing_by_modality[modality] = cache
    if not pending:
        return [
            _manifest_row_from_cache(
                case_dir / f"{_safe_component(modality.lower())}.pt",
                existing_by_modality[modality],
                status="reused",
            )
            for modality in modalities
        ]
    transform = build_modality_full_volume_transform(
        dataset, cfg["data"], all_modalities
    )
    volumes, meta = load_and_preprocess_modality_case(transform, case_record)
    first_volume = next(iter(volumes.values()))
    original_shape_hwd = tuple((int(dim) for dim in first_volume.shape[1:]))
    padded_shape_hwd = compute_padded_shape_hwd(
        original_shape_hwd,
        dataset=dataset,
        patch_hwd=patch_size_hwd,
        stride_hwd=stride_hwd,
        comp_hwd=compression_hwd,
        pad_mode=pad_mode,
        minimum_image_shape_hwd=minimum_image_shape_hwd,
    )
    patch_table, starts_hwd = build_patch_index_table(
        padded_shape_hwd,
        patch_hwd=patch_size_hwd,
        stride_hwd=stride_hwd,
        comp_hwd=compression_hwd,
        include_last=include_last,
    )
    image_valid = build_image_valid_mask_hwd(original_shape_hwd, padded_shape_hwd)
    valid_hard, valid_soft = downsample_valid_mask_to_latent(
        image_valid, comp_hwd=compression_hwd, threshold=0.5
    )
    anatomy_image = meta.get("anatomy_mask_image_hwd")
    if anatomy_image is None:
        raise RuntimeError(
            "Medical preprocessing did not provide anatomy_mask_image_hwd"
        )
    anatomy_hard, _ = downsample_anatomy_mask_to_latent(
        anatomy_image, padded_shape_hwd, compression_hwd=compression_hwd
    )
    pir_support_hard = compose_pir_mask_latent_hard(valid_hard, anatomy_hard)
    from streamrefine.training.pir import count_minimal_pir_edges

    pir_valid_edge_count = int(
        count_minimal_pir_edges(valid_hard, anatomy_hard)[0].item()
    )
    if pir_valid_edge_count <= 0:
        raise ValueError(
            "Deterministic valid & anatomy support contains no configured PIR edge"
        )
    mask_policy = str(meta.get("anatomy_mask_policy", ""))
    mask_contract = anatomy_mask_contract(dataset, mask_policy)
    mask_contract_fingerprint = anatomy_mask_contract_fingerprint(dataset, mask_policy)
    if dict(meta.get("anatomy_mask_contract", {})) != mask_contract:
        raise RuntimeError(
            "Preprocessing anatomy-mask contract drifted before cache serialization"
        )
    if (
        str(meta.get("anatomy_mask_contract_fingerprint", ""))
        != mask_contract_fingerprint
    ):
        raise RuntimeError(
            "Preprocessing anatomy-mask fingerprint drifted before cache serialization"
        )
    anatomy_image_bool = torch.as_tensor(anatomy_image).bool()
    image_voxel_total = int(anatomy_image_bool.numel())
    latent_cell_total = int(anatomy_hard.numel())
    anatomy_mask_statistics = {
        "image_voxel_count": int(anatomy_image_bool.sum().item()),
        "image_voxel_ratio": float(anatomy_image_bool.float().mean().item()),
        "latent_cell_count": int(anatomy_hard.sum().item()),
        "latent_cell_ratio": float(anatomy_hard.float().mean().item()),
        "pir_support_cell_count": int(pir_support_hard.sum().item()),
        "pir_support_cell_ratio": float(pir_support_hard.float().mean().item()),
        "pir_valid_edge_count": pir_valid_edge_count,
        "image_voxel_total": image_voxel_total,
        "latent_cell_total": latent_cell_total,
    }
    latent_shape_dhw = tuple((int(item) for item in valid_hard.shape))
    latent_tile_dhw = (
        patch_size_hwd[2] // compression_hwd[2],
        patch_size_hwd[0] // compression_hwd[0],
        patch_size_hwd[1] // compression_hwd[1],
    )
    importance = make_importance_map_dhw(
        latent_tile_dhw,
        mode=importance_mode,
        eps=importance_floor,
        gaussian_sigma_scale=gaussian_sigma_scale,
    )
    rows: list[dict[str, Any]] = []
    for modality in modalities:
        cache_path = case_dir / f"{_safe_component(modality.lower())}.pt"
        if modality in existing_by_modality:
            rows.append(
                _manifest_row_from_cache(
                    cache_path, existing_by_modality[modality], status="reused"
                )
            )
            continue
        volume = pad_chwd_to_shape(
            volumes[modality], padded_shape_hwd, value=padding_value
        )
        patch_dataset = ModalityPatchDataset(volume, patch_table)
        loader = DataLoader(
            patch_dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=max(0, num_workers),
            pin_memory=bool(
                device.type == "cuda"
                and _config_value(cfg, "precompute", "pin_memory", True)
            ),
            persistent_workers=bool(
                num_workers > 0
                and _config_value(cfg, "precompute", "persistent_workers", True)
            ),
        )
        latent_sum, weight_sum, allocated_shape = allocate_full_accumulators(
            padded_shape_hwd, channels=16, comp_hwd=compression_hwd
        )
        if tuple(allocated_shape) != latent_shape_dhw:
            raise RuntimeError(
                f"Accumulator/mask latent shapes disagree: {allocated_shape} vs {latent_shape_dhw}"
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        encoded_patches = 0
        checked_determinism_here = False
        checked_decode_here = False
        determinism_max_abs = 0.0
        batch_metrics: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(loader):
            patch_ids = [int(item) for item in batch["patch_id"].tolist()]
            try:
                crops = batch["crop"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                batch_start = time.perf_counter()
                means_device = encoder.encode_mean(crops, precision=precision)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"CUDA out of memory while encoding VidTok tiles; lower --batch-size (current value: {batch_size}) and resume the cache job."
                ) from error
            batch_seconds = time.perf_counter() - batch_start
            batch_metrics.append(
                {
                    "batch_index": batch_index,
                    "num_patches": len(patch_ids),
                    "encode_seconds": batch_seconds,
                    "latent_min": float(means_device.min().item()),
                    "latent_max": float(means_device.max().item()),
                }
            )
            if verify_determinism and (
                not bool(verification_state.get("determinism_done", False))
            ):
                repeated = encoder.encode_mean(crops, precision=precision)
                difference = (means_device - repeated).abs()
                determinism_max_abs = float(difference.max().item())
                atol = float(_config_value(cfg, "vidtok", "determinism_atol", 1e-05))
                rtol = float(_config_value(cfg, "vidtok", "determinism_rtol", 0.0001))
                if not torch.allclose(means_device, repeated, atol=atol, rtol=rtol):
                    raise RuntimeError(
                        f"Repeated posterior-mean encoding was not deterministic: max_abs={determinism_max_abs:.6g}, atol={atol}, rtol={rtol}"
                    )
                verification_state.update(
                    {
                        "determinism_done": True,
                        "determinism_max_abs": determinism_max_abs,
                        "determinism_atol": atol,
                        "determinism_rtol": rtol,
                        "determinism_case_id": case_record.get("case_id", ""),
                        "determinism_modality": modality,
                    }
                )
                checked_determinism_here = True
            if verify_decode and (
                not bool(verification_state.get("decode_done", False))
            ):
                decoded = encoder.decode_raw_mean(means_device[:1], precision=precision)
                expected_decoded = (
                    1,
                    3,
                    patch_size_hwd[2],
                    patch_size_hwd[0],
                    patch_size_hwd[1],
                )
                if tuple(decoded.shape) != expected_decoded:
                    raise RuntimeError(
                        f"Mean-latent decode expected {expected_decoded}, got {tuple(decoded.shape)}"
                    )
                verification_state.update(
                    {
                        "decode_done": True,
                        "decode_shape_btchw": list(decoded.shape),
                        "decode_case_id": case_record.get("case_id", ""),
                        "decode_modality": modality,
                    }
                )
                checked_decode_here = True
            means = means_device.cpu()
            for item_index, patch_id in enumerate(patch_ids):
                accumulate_patch_mean(
                    latent_sum,
                    weight_sum,
                    means[item_index],
                    patch_table[patch_id],
                    importance,
                )
                encoded_patches += 1
        full_mu = finalize_full_mean(latent_sum, weight_sum)
        if tuple(full_mu.shape) != (16, *latent_shape_dhw):
            raise RuntimeError(
                f"Unexpected full posterior-mean shape: {tuple(full_mu.shape)}"
            )
        elapsed = time.perf_counter() - start_time
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        modality_metadata = meta.get("metadata_by_modality", {}).get(modality, {})
        modality_normalization = meta.get("normalization_by_modality", {}).get(
            modality, meta.get("normalization_info", {})
        )
        pad_info = padding_info(
            original_shape_hwd,
            padded_shape_hwd,
            pad_value=padding_value,
            policy=pad_mode,
        )
        cache: dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "latent_mu": full_mu.to(torch.float16),
            "valid_mask_latent": valid_hard.bool(),
            "valid_mask_latent_soft": valid_soft.to(torch.float16),
            "anatomy_mask_latent_hard": anatomy_hard.bool(),
            "anatomy_mask_contract": mask_contract,
            "anatomy_mask_contract_fingerprint": mask_contract_fingerprint,
            "anatomy_mask_sha256": anatomy_mask_sha256(anatomy_hard),
            "anatomy_mask_statistics": anatomy_mask_statistics,
            "original_shape_hwd": list(original_shape_hwd),
            "padded_shape_hwd": list(padded_shape_hwd),
            "latent_shape_dhw": list(latent_shape_dhw),
            "compression_hwd": list(compression_hwd),
            "latent_layout": "C,D,H,W",
            "posterior_mode": "mean",
            "save_logvar": False,
            "latent_channels": 16,
            "cache_dtype": "float16",
            "stitch_mode": "weighted_posterior_mean_blending",
            "importance_mode": importance_mode,
            "importance_floor": importance_floor,
            "gaussian_sigma_scale": gaussian_sigma_scale
            if importance_mode == "gaussian"
            else None,
            "patch_size_hwd": list(patch_size_hwd),
            "stride_hwd": list(stride_hwd),
            "overlap": float(overlap),
            "patch_starts_hwd": [list(axis) for axis in starts_hwd],
            "patch_index_table": patch_table,
            "tokenizer_type": "vidtok_kl",
            "tokenizer_causal": False,
            "tokenizer_repo_root": str(cfg["vidtok"]["repo_root"]),
            "tokenizer_config": str(cfg["vidtok"]["config_path"]),
            "tokenizer_config_sha256": tokenizer_config_sha256,
            "tokenizer_checkpoint": str(cfg["vidtok"]["ckpt_path"]),
            "tokenizer_checkpoint_sha256": checkpoint_sha256,
            "generation_contract": generation_contract,
            "generation_contract_sha256": generation_contract_sha256,
            "tokenizer_git_commit": tokenizer_git_commit,
            "dataset": str(
                meta.get("dataset") or case_record.get("dataset") or dataset
            ),
            "case_id": str(meta.get("case_id") or case_record.get("case_id", "")),
            "group_id": str(meta.get("group_id") or case_record.get("group_id", "")),
            "split": str(meta.get("split") or case_record.get("split", "")),
            "modality": modality.lower(),
            "available_modalities": [str(item).lower() for item in all_modalities],
            "modality_image_path": meta.get("modality_image_paths", {}).get(
                modality, ""
            ),
            "modality_image_paths": meta.get("modality_image_paths", {}),
            "input_file_signatures": input_file_signatures,
            "task": str(meta.get("task", "")),
            "anatomy": str(meta.get("anatomy", "")),
            "cohort": str(meta.get("cohort", "")),
            "tracer": str(meta.get("tracer", "")),
            "crop_source_modality": str(meta.get("crop_source_modality", "")),
            "spacing": modality_metadata.get("spacing"),
            "affine": modality_metadata.get("affine"),
            "modality_metadata": modality_metadata,
            "metadata_by_modality": meta.get("metadata_by_modality", {}),
            "affine_corrections": meta.get("affine_corrections", []),
            "padding_info": pad_info,
            "normalization_info": _normalization_info(
                cfg, modality, padding_value, modality_normalization
            ),
            "encoding_info": {
                "precision": precision,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "num_patches": encoded_patches,
                "elapsed_seconds": elapsed,
                "patches_per_second": encoded_patches / max(elapsed, 1e-09),
                "cuda_peak_memory_bytes": peak_memory,
                "latent_min": float(full_mu.min().item()),
                "latent_max": float(full_mu.max().item()),
                "determinism_checked": checked_determinism_here,
                "determinism_max_abs": determinism_max_abs
                if checked_determinism_here
                else None,
                "decode_contract_checked": checked_decode_here,
                "batch_metrics": batch_metrics,
            },
        }
        cache["input_snapshot_sha256"] = input_snapshot_sha256(cache)
        validate_cache_dict(
            cache,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_config_sha256=tokenizer_config_sha256,
            expected_generation_contract_sha256=generation_contract_sha256,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(cache, cache_path, overwrite=overwrite)
        rows.append(_manifest_row_from_cache(cache_path, cache, status="written"))
    return rows


def _process_case(**kwargs: Any) -> list[dict[str, Any]]:
    case_record = kwargs["case_record"]
    output_dir = kwargs["output_dir"]
    case_dir = _case_cache_dir(output_dir, case_record)
    with _exclusive_case_writer(case_dir):
        return _process_case_unlocked(**kwargs)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg, config_path, configured_dataset = _prepare_config(args)
    if args.print_config:
        if args.output_dir is not None:
            cfg["cache"]["output_dir"] = str(args.output_dir)
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return
    from streamrefine.data.preprocess_medical import (
        build_modality_case_records,
        canonical_modality_name,
        load_configured_records,
        load_yaml_config,
        normalize_dataset_name,
    )

    dataset = normalize_dataset_name(configured_dataset)
    expected_config_schema = "streamrefine_cache_config_v2_anatomy_mask"
    if str(cfg.get("schema_version", "")) != expected_config_schema:
        raise ValueError(
            f"Anatomy-aware cache generation requires schema_version={expected_config_schema!r}"
        )
    expected_cache_version = "streamrefine_kl_mean_v2_anatomy_mask"
    if str(_config_value(cfg, "cache", "cache_version", "")) != expected_cache_version:
        raise ValueError(
            f"Anatomy-aware cache generation requires cache.cache_version={expected_cache_version!r}"
        )
    if _config_value(cfg, "cache", "save_anatomy_mask_latent_hard", None) is not True:
        raise ValueError("cache.save_anatomy_mask_latent_hard must be exactly true")
    from streamrefine.data.anatomy_mask import canonical_anatomy_mask_policy

    canonical_anatomy_mask_policy(
        dataset, _config_value(cfg, "data", "anatomy_mask_policy", None)
    )
    requested_split = args.split or str(_config_value(cfg, "data", "split", "all"))
    split = "val" if requested_split == "validation" else requested_split.lower()
    if split not in {"train", "val", "all"}:
        raise ValueError(f"Unsupported configured split: {requested_split!r}")
    records = load_configured_records(
        cfg, dataset=dataset, split=split, rebuild=args.rebuild_datalists
    )
    case_records = build_modality_case_records(
        records, dataset=dataset, modalities=None
    )
    selected_modalities: set[str] | None = None
    if args.modalities and args.modalities.strip().lower() not in {"all", "any", "*"}:
        selected_modalities = {
            canonical_modality_name(item)
            for item in args.modalities.split(",")
            if item.strip()
        }
        if not selected_modalities or "" in selected_modalities:
            raise ValueError(
                f"--modalities did not contain a usable modality: {args.modalities!r}"
            )
        case_records = [
            row
            for row in case_records
            if selected_modalities.intersection(
                (str(item).lower() for item in row.get("modalities", {}))
            )
        ]
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            "Need --shard-count > 0 and 0 <= --shard-index < --shard-count"
        )
    case_records = case_records[args.start_index :]
    case_records = [
        row
        for index, row in enumerate(case_records)
        if index % args.shard_count == args.shard_index
    ]
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("--max-cases must be positive")
        case_records = case_records[: args.max_cases]
    if not case_records:
        raise RuntimeError(
            "No cases matched the requested dataset, split, modality, and shard filters."
        )
    configured_output = _config_value(cfg, "cache", "output_dir", None)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else None
        if _is_placeholder(configured_output)
        else _resolve_path(configured_output, config_path)
    )
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT / "result" / "streamrefine_latents" / dataset
        ).resolve()
    manifest_out = args.manifest_out.resolve() if args.manifest_out else None
    pair_manifest_out = (
        args.pair_manifest_out.resolve() if args.pair_manifest_out else None
    )
    split_tag = "" if split == "all" else f"_{split}"
    shard_tag = None
    if args.shard_count > 1:
        shard_tag = f"shard-{args.shard_index:05d}-of-{args.shard_count:05d}"
        if manifest_out is None:
            manifest_out = output_dir / f"cache_manifest{split_tag}.{shard_tag}.jsonl"
        if pair_manifest_out is None:
            pair_manifest_out = (
                output_dir / f"pair_manifest{split_tag}.{shard_tag}.jsonl"
            )
    audit_suffix = split_tag + (f".{shard_tag}" if shard_tag else "")
    failure_log = (
        args.failure_log.resolve()
        if args.failure_log
        else output_dir / f"failures{audit_suffix}.jsonl"
    )
    print(f"Dataset: {dataset}")
    print(f"Split: {split}")
    print(f"Config: {config_path}")
    print(f"Data root: {cfg['data'].get('data_root', '(not configured)')}")
    print(f"Cases selected: {len(case_records)}")
    print(f"Output directory: {output_dir}")
    input_count = _check_input_paths(case_records)
    print(f"Input paths verified: {input_count}")
    if args.dry_run:
        print(json.dumps(case_records[:5], ensure_ascii=False, indent=2))
        return
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for latent generation.") from error
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
            return iterable

    from streamrefine.tokenizer.cache_schema import sha256_file
    from streamrefine.tokenizer.grid_patch_with_coords import (
        COMPRESSION_HWD,
        PATCH_SIZE_HWD,
        STRIDE_HWD,
        stride_from_overlap,
    )
    from streamrefine.tokenizer.vidtok_kl_wrapper import load_vidtok_kl_mean_encoder

    checkpoint_path = Path(cfg["vidtok"]["ckpt_path"])
    tokenizer_config_path = Path(cfg["vidtok"]["config_path"])
    vidtok_root = Path(cfg["vidtok"]["repo_root"])
    for label, path, expected_kind in (
        ("VidTok checkpoint", checkpoint_path, "file"),
        ("VidTok model config", tokenizer_config_path, "file"),
        ("VidTok repository", vidtok_root, "directory"),
    ):
        valid = path.is_file() if expected_kind == "file" else path.is_dir()
        if not valid:
            raise FileNotFoundError(f"{label} is not a usable {expected_kind}: {path}")
    print("Hashing tokenizer checkpoint (performed once for cache provenance)...")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    tokenizer_config_sha256 = sha256_file(tokenizer_config_path)
    tokenizer_git_commit = _git_commit(vidtok_root) or "unknown"
    requested_device = args.device or str(
        _config_value(cfg, "precompute", "device", "cuda")
    )
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    requested_precision = (
        args.precision
        or str(_config_value(cfg, "precompute", "precision", "bf16")).lower()
    )
    if requested_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {requested_precision!r}")
    precision = requested_precision
    if device.type != "cuda" and precision != "fp32":
        print(
            f"WARNING: precision={precision} was requested on {device.type}; the verified CPU debug path uses fp32 instead."
        )
        precision = "fp32"
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else _config_value(cfg, "precompute", "batch_size", 8)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else _config_value(cfg, "precompute", "num_workers", 8)
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    resume = bool(
        _config_value(cfg, "precompute", "resume", True)
        if args.resume is None
        else args.resume
    )
    overwrite = bool(
        _config_value(cfg, "precompute", "overwrite", False)
        if args.overwrite is None
        else args.overwrite
    )
    continue_on_error = bool(
        _config_value(cfg, "precompute", "continue_on_error", True)
        if args.continue_on_error is None
        else args.continue_on_error
    )
    verify_determinism = bool(
        _config_value(cfg, "vidtok", "verify_determinism", True)
        if args.verify_determinism is None
        else args.verify_determinism
    )
    verify_decode = bool(
        _config_value(cfg, "vidtok", "verify_decode_subset", True)
        if args.verify_decode is None
        else args.verify_decode
    )
    patch_size_hwd = _as_hwd(
        args.patch_size_hwd,
        _config_value(cfg, "patching", "patch_size_hwd", [256, 256, 16]),
        "patch_size_hwd",
    )
    compression_hwd = _as_hwd(
        args.compression_hwd,
        _config_value(cfg, "patching", "compression_hwd", [8, 8, 4]),
        "compression_hwd",
    )
    overlap = float(
        args.overlap
        if args.overlap is not None
        else _config_value(cfg, "patching", "overlap", 0.25)
    )
    stride_hwd = stride_from_overlap(patch_size_hwd, overlap)
    configured_stride = _config_value(cfg, "patching", "stride_hwd", None)
    if (
        configured_stride is not None
        and tuple((int(item) for item in configured_stride)) != stride_hwd
    ):
        raise ValueError(
            f"Configured stride_hwd={configured_stride} disagrees with patch/overlap derived {stride_hwd}"
        )
    if patch_size_hwd != PATCH_SIZE_HWD:
        raise ValueError(
            f"cache-v2 requires patch_size_hwd={PATCH_SIZE_HWD}, got {patch_size_hwd}"
        )
    if compression_hwd != COMPRESSION_HWD:
        raise ValueError(
            f"cache-v2 requires compression_hwd={COMPRESSION_HWD}, got {compression_hwd}"
        )
    if stride_hwd != STRIDE_HWD or not math.isclose(
        overlap, 0.25, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"cache-v2 requires overlap=0.25 and stride_hwd={STRIDE_HWD}, got overlap={overlap}, stride_hwd={stride_hwd}"
        )
    minimum_image_shape_hwd = _as_hwd(
        None,
        _config_value(cfg, "patching", "minimum_image_shape_hwd", [96, 96, 96]),
        "minimum_image_shape_hwd",
    )
    pad_mode = args.pad_mode or str(_config_value(cfg, "patching", "pad_mode", "auto"))
    importance_mode = args.importance_mode or str(
        _config_value(cfg, "patching", "importance_mode", "hann")
    )
    importance_floor = float(
        args.importance_floor
        if args.importance_floor is not None
        else _config_value(cfg, "patching", "importance_floor", 0.001)
    )
    gaussian_sigma_scale = float(
        args.gaussian_sigma_scale
        if args.gaussian_sigma_scale is not None
        else _config_value(cfg, "patching", "gaussian_sigma_scale", 0.125)
    )
    include_last = bool(_config_value(cfg, "patching", "include_last_patch", True))
    if not include_last:
        raise ValueError("cache-v2 requires patching.include_last_patch=true")
    if not 0.0 < importance_floor <= 1.0 or not math.isfinite(importance_floor):
        raise ValueError("importance_floor must be finite and in (0,1]")
    if not math.isfinite(gaussian_sigma_scale) or gaussian_sigma_scale <= 0.0:
        raise ValueError("gaussian_sigma_scale must be positive and finite")
    preprocessing_config_path = (
        Path(str(cfg.get("_preprocessing_config_path", ""))).expanduser().resolve()
    )
    if not preprocessing_config_path.is_file():
        raise FileNotFoundError(
            f"Resolved medical preprocessing config does not exist: {preprocessing_config_path}"
        )
    preprocessing_cfg = load_yaml_config(preprocessing_config_path)
    padding_value, preprocessing_input_range, padding_value_source = (
        _resolve_padding_value(cfg, preprocessing_cfg)
    )
    implementation_paths = {
        "precompute": THIS_FILE,
        "preprocess_medical": AR_ROOT
        / "streamrefine"
        / "data"
        / "preprocess_medical.py",
        "grid_patch_with_coords": AR_ROOT
        / "streamrefine"
        / "tokenizer"
        / "grid_patch_with_coords.py",
        "valid_mask": AR_ROOT / "streamrefine" / "tokenizer" / "valid_mask.py",
        "weighted_stitch": AR_ROOT
        / "streamrefine"
        / "tokenizer"
        / "weighted_stitch.py",
        "vidtok_kl_wrapper": AR_ROOT
        / "streamrefine"
        / "tokenizer"
        / "vidtok_kl_wrapper.py",
        "cache_schema": AR_ROOT / "streamrefine" / "tokenizer" / "cache_schema.py",
        "anatomy_mask": AR_ROOT / "streamrefine" / "data" / "anatomy_mask.py",
        "medical_preprocessing": AR_ROOT
        / "streamrefine"
        / "data"
        / "preprocess_medical.py",
        "brats_affine": AR_ROOT / "streamrefine" / "data" / "brats_affine.py",
        "cache_precompute_entrypoint": THIS_FILE,
    }
    implementation_sha256 = {
        name: sha256_file(path) for name, path in implementation_paths.items()
    }
    tokenizer_source_tree = _python_source_tree_fingerprint(
        {
            "vidtok_package": vidtok_root / "vidtok",
            "medical_wrapper": PROJECT_ROOT / "vidtok_rec" / "streamrefine_tokenizer",
        }
    )
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
    )

    active_anatomy_mask_contract = anatomy_mask_contract(
        dataset, cfg["data"].get("anatomy_mask_policy")
    )
    runtime_dependency_versions: dict[str, str] = {"python": platform.python_version()}
    for distribution in ("torch", "monai", "numpy", "scipy", "PyYAML"):
        try:
            runtime_dependency_versions[distribution.lower()] = (
                importlib.metadata.version(distribution)
            )
        except importlib.metadata.PackageNotFoundError:
            runtime_dependency_versions[distribution.lower()] = "not-installed"
    generation_contract: dict[str, Any] = {
        "cache_version": "streamrefine_kl_mean_v2_anatomy_mask",
        "dataset": dataset,
        "outer_cache_config": str(config_path),
        "preprocessing_config": str(preprocessing_config_path),
        "preprocessing_config_sha256": sha256_file(preprocessing_config_path),
        "effective_data_config": {
            key: value
            for key, value in dict(cfg.get("data", {})).items()
            if key != "split"
        },
        "anatomy_mask_contract": active_anatomy_mask_contract,
        "anatomy_mask_contract_fingerprint": anatomy_mask_contract_fingerprint(
            dataset, active_anatomy_mask_contract["policy"]
        ),
        "tokenizer_checkpoint_sha256": checkpoint_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "tokenizer_git_commit": tokenizer_git_commit,
        "precision": precision,
        "patch_size_hwd": list(patch_size_hwd),
        "stride_hwd": list(stride_hwd),
        "compression_hwd": list(compression_hwd),
        "minimum_image_shape_hwd": list(minimum_image_shape_hwd),
        "pad_mode": pad_mode,
        "padding_value": padding_value,
        "padding_value_source": padding_value_source,
        "preprocessing_input_range": list(preprocessing_input_range),
        "include_last_patch": include_last,
        "importance_mode": importance_mode,
        "importance_floor": importance_floor,
        "gaussian_sigma_scale": gaussian_sigma_scale,
        "implementation_sha256": implementation_sha256,
        "tokenizer_source_tree": tokenizer_source_tree,
        "runtime_dependency_versions": runtime_dependency_versions,
    }
    encoder = load_vidtok_kl_mean_encoder(
        cfg, checkpoint_path=checkpoint_path, device=device, lightweight_loss=True
    )
    generation_contract["checkpoint_state_dict_audit"] = dict(encoder.checkpoint_audit)
    generation_contract_sha256 = _stable_json_sha256(generation_contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / f"cache_run_config{audit_suffix}.json"
    _write_json(
        run_config_path,
        {
            "dataset": dataset,
            "split": split,
            "config": str(config_path),
            "output_dir": str(output_dir),
            "vidtok_root": str(vidtok_root),
            "vidtok_model_config": str(tokenizer_config_path),
            "vidtok_checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
            "generation_contract": generation_contract,
            "generation_contract_sha256": generation_contract_sha256,
            "patch_size_hwd": list(patch_size_hwd),
            "stride_hwd": list(stride_hwd),
            "compression_hwd": list(compression_hwd),
            "importance_mode": importance_mode,
            "precision": precision,
            "requested_precision": requested_precision,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "resume": resume,
            "overwrite": overwrite,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "verify_determinism": verify_determinism,
            "verify_decode": verify_decode,
        },
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fatal_error: Exception | None = None
    verification_state: dict[str, Any] = {
        "determinism_requested": verify_determinism,
        "decode_requested": verify_decode,
        "determinism_done": False,
        "decode_done": False,
    }
    for case_record in tqdm(
        case_records, desc="Precomputing deterministic VidTok-KL means"
    ):
        try:
            rows.extend(
                _process_case(
                    case_record=case_record,
                    cfg=cfg,
                    dataset=dataset,
                    output_dir=output_dir,
                    encoder=encoder,
                    device=device,
                    checkpoint_sha256=checkpoint_sha256,
                    tokenizer_config_sha256=tokenizer_config_sha256,
                    generation_contract=generation_contract,
                    generation_contract_sha256=generation_contract_sha256,
                    tokenizer_git_commit=tokenizer_git_commit,
                    patch_size_hwd=patch_size_hwd,
                    stride_hwd=stride_hwd,
                    compression_hwd=compression_hwd,
                    minimum_image_shape_hwd=minimum_image_shape_hwd,
                    pad_mode=pad_mode,
                    padding_value=padding_value,
                    include_last=include_last,
                    importance_mode=importance_mode,
                    importance_floor=importance_floor,
                    gaussian_sigma_scale=gaussian_sigma_scale,
                    overlap=overlap,
                    precision=precision,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    resume=resume,
                    overwrite=overwrite,
                    verify_determinism=verify_determinism,
                    verify_decode=verify_decode,
                    verification_state=verification_state,
                    selected_modalities=selected_modalities,
                )
            )
        except Exception as error:
            failure = {
                "dataset": dataset,
                "case_id": case_record.get("case_id", ""),
                "group_id": case_record.get("group_id", ""),
                "split": case_record.get("split", ""),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
            failures.append(failure)
            _report_case_failure(failure, len(failures), failure_log)
            if not continue_on_error:
                fatal_error = error
                break
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    configured_pairs = _translation_pairs(cfg, dataset)
    incremental_selection = bool(
        args.start_index > 0
        or args.max_cases is not None
        or selected_modalities is not None
    )
    modality_manifest, pair_manifest, manifest_rows, pair_rows = _write_manifests(
        rows=rows,
        translation_pairs=configured_pairs,
        output_dir=output_dir,
        requested_split=split,
        manifest_out=manifest_out,
        pair_manifest_out=pair_manifest_out,
        write_split_specific=args.shard_count == 1,
        merge_existing=incremental_selection or fatal_error is not None,
    )
    missing_pair_rows = _build_missing_pair_rows(manifest_rows, configured_pairs)
    _write_jsonl(failure_log, failures)
    validation_path = output_dir / f"cache_run_validation{audit_suffix}.json"
    pair_validation_path = output_dir / f"pair_manifest_validation{audit_suffix}.jsonl"
    _write_jsonl(pair_validation_path, missing_pair_rows)
    verification_state["available_pair_count"] = len(pair_rows)
    verification_state["missing_pair_count"] = len(missing_pair_rows)
    verification_state["pair_validation_report"] = str(pair_validation_path)
    _write_json(validation_path, verification_state)
    print(f"Modality caches in manifest: {len(manifest_rows)} -> {modality_manifest}")
    print(f"Available translation pairs: {len(pair_rows)} -> {pair_manifest}")
    print(f"Failures: {len(failures)} -> {failure_log}")
    print(
        f"Unavailable configured pairs: {len(missing_pair_rows)} -> {pair_validation_path}"
    )
    print(f"Tokenizer contract checks -> {validation_path}")
    wrote_new_cache = any((row.get("write_action") == "written" for row in rows))
    if (
        wrote_new_cache
        and verify_determinism
        and (not verification_state.get("determinism_done"))
    ):
        raise RuntimeError(
            "A new cache was written without completing the requested determinism check."
        )
    if (
        wrote_new_cache
        and verify_decode
        and (not verification_state.get("decode_done"))
    ):
        raise RuntimeError(
            "A new cache was written without completing the requested decode-contract check."
        )
    if fatal_error is not None:
        raise RuntimeError(
            "Latent generation stopped after a case failure; see the failure log."
        ) from fatal_error
    if failures:
        raise RuntimeError(
            f"Latent generation completed with {len(failures)} failed case(s); first error: {failures[0]['error_type']}: {failures[0]['error']}. successful manifests were written, and details are in {failure_log}."
        )


if __name__ == "__main__":
    main()
