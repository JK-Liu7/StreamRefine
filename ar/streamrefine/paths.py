"""Portable paths rooted at the extracted release directory."""

from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any, Mapping

RELEASE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path_string(
    value: str | Path, *, base_dir: str | Path | None = None
) -> str:
    text = str(value)
    if not text.strip() or text.startswith("REPLACE_WITH_"):
        return text
    path = Path(os.path.expandvars(text)).expanduser()
    return str(
        (
            path if path.is_absolute() else Path(base_dir or RELEASE_ROOT) / path
        ).resolve()
    )


def resolve_config_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    local = Path.cwd() / path
    return local.resolve() if local.is_file() else (RELEASE_ROOT / path).resolve()


def resolve_runtime_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve path-valued YAML fields against the release root.

    Empty optional paths stay empty. Paths in manifests are resolved by the
    manifest reader, and explicit data.path_remap rules remain user-controlled.
    """
    fields = {
        "root",
        "repo",
        "repo_root",
        "config",
        "config_path",
        "checkpoint",
        "ckpt_path",
        "data_root",
        "label_dir",
        "splits_path",
        "cache_dir",
        "output_dir",
        "output_root",
        "stats_path",
        "train_manifest",
        "val_manifest",
        "test_manifest",
        "manifest",
        "train_datalist",
        "val_datalist",
        "preprocessing_config",
        "calibration_path",
    }

    def visit(value: Any, key: str = "") -> Any:
        if isinstance(value, Mapping):
            return {
                k: copy.deepcopy(v)
                if k in {"path_remap", "_config_path"}
                else visit(v, k)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, (str, Path)) and key in fields:
            return resolve_path_string(value)
        return copy.deepcopy(value)

    return visit(config)


def runtime_path_remaps(config: Mapping[str, Any]) -> list[dict[str, str]]:
    from streamrefine.data.path_resolver import normalise_prefix_remaps

    return [
        {"from": source, "to": resolve_path_string(target)}
        for source, target in normalise_prefix_remaps(
            config.get("data", {}).get("path_remap", ())
        )
    ]
