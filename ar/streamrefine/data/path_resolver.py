"""Fail-closed path resolution for portable StreamRefine manifests."""

from __future__ import annotations
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping


class PathResolutionError(FileNotFoundError):
    pass


_WINDOWS_ABSOLUTE = re.compile("^[A-Za-z]:[\\\\/]")


def _normalise_windows(text: str) -> str:
    return str(PureWindowsPath(text)).replace("/", "\\").rstrip("\\").casefold()


def normalise_prefix_remaps(value: Any) -> tuple[tuple[str, str], ...]:
    """Accept a mapping or a list of ``{from,to}`` entries."""
    if value in (None, ""):
        return ()
    pairs: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        iterable: Iterable[Any] = (
            {"from": key, "to": target} for key, target in value.items()
        )
    elif isinstance(value, (list, tuple)):
        iterable = value
    else:
        raise TypeError("path_remap must be a mapping or a list of {from,to} mappings")
    for entry in iterable:
        if not isinstance(entry, Mapping):
            raise TypeError(f"Invalid path-remap entry: {entry!r}")
        source = str(entry.get("from", "")).strip()
        target = str(entry.get("to", "")).strip()
        if not source or not target:
            raise ValueError(
                f"Path-remap entries require non-empty from/to values: {entry!r}"
            )
        pairs.append((source, target))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(pairs)


def _apply_remap(text: str, remaps: tuple[tuple[str, str], ...]) -> Path | None:
    windows = bool(_WINDOWS_ABSOLUTE.match(text) or text.startswith("\\\\"))
    for source, target in remaps:
        if windows:
            source_key = _normalise_windows(source)
            text_key = _normalise_windows(text)
            if text_key == source_key:
                suffix_parts: tuple[str, ...] = ()
            elif text_key.startswith(source_key + "\\"):
                original_parts = PureWindowsPath(text).parts
                source_parts = PureWindowsPath(source).parts
                suffix_parts = tuple(original_parts[len(source_parts) :])
            else:
                continue
        else:
            source_norm = os.path.normpath(source)
            text_norm = os.path.normpath(text)
            try:
                common = os.path.commonpath([source_norm, text_norm])
            except ValueError:
                continue
            if common != source_norm:
                continue
            suffix = os.path.relpath(text_norm, source_norm)
            suffix_parts = () if suffix == "." else tuple(Path(suffix).parts)
        return Path(target).expanduser().joinpath(*suffix_parts)
    return None


def is_explicitly_remapped_path(
    value: str | Path, resolved: str | Path, *, remaps: Any = ()
) -> bool:
    """Whether a configured prefix moves this recorded path to this endpoint."""
    text = str(value).strip()
    if not text:
        return False
    mapped = _apply_remap(text, normalise_prefix_remaps(remaps))
    if mapped is None:
        return False
    endpoint = Path(resolved).expanduser().resolve(strict=False)
    return (
        mapped.resolve(strict=False) == endpoint
        and Path(text).expanduser().resolve(strict=False) != endpoint
    )


def resolve_recorded_path(
    value: str | Path,
    *,
    remaps: Any = (),
    base_dir: str | Path | None = None,
    must_exist: bool = True,
    role: str = "path",
) -> Path:
    """Resolve an explicit recorded path without guessing cross-platform roots."""
    text = str(value).strip()
    if not text:
        raise PathResolutionError(f"Required {role} is empty")
    mappings = normalise_prefix_remaps(remaps)
    is_windows_absolute = bool(_WINDOWS_ABSOLUTE.match(text) or text.startswith("\\\\"))
    if is_windows_absolute:
        remapped = _apply_remap(text, mappings)
        if remapped is not None:
            candidate = remapped
        elif os.name == "nt":
            candidate = Path(text).expanduser()
        elif remapped is None:
            raise PathResolutionError(
                f"Recorded Windows {role} {text!r} requires an explicit data.path_remap prefix"
            )
    else:
        candidate = Path(text).expanduser()
        if candidate.is_absolute() or text.startswith("/"):
            remapped = _apply_remap(text, mappings)
            candidate = remapped if remapped is not None else candidate
        else:
            candidate = Path(base_dir or Path.cwd()).expanduser() / candidate
    candidate = candidate.resolve(strict=False)
    if must_exist and (not candidate.exists()):
        raise PathResolutionError(
            f"Resolved {role} does not exist: {candidate} (recorded {text!r})"
        )
    return candidate
