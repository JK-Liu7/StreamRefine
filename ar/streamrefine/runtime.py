"""Small runtime helpers shared by train and inference."""

from __future__ import annotations
import contextlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping


class RuntimeContractError(RuntimeError):
    pass


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "StreamRefine runtime requires PyTorch; install it in the portable environment"
        ) from exc
    return torch


def require_monai():
    try:
        import monai
    except ImportError as exc:
        raise RuntimeError(
            "StreamRefine data/runtime requires MONAI in the portable environment"
        ) from exc
    return monai


def configure_precision(precision: str, device: Any):
    """Return autocast context and dtype, rejecting unsupported BF16."""
    torch = require_torch()
    value = str(precision).lower()
    device_type = torch.device(device).type
    if value == "fp32":
        return (contextlib.nullcontext, torch.float32)
    if value != "bf16":
        raise RuntimeContractError(f"Unsupported precision {precision!r}")
    if device_type != "cuda":
        raise RuntimeContractError(
            "BF16 training is configured but the selected device is not CUDA"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeContractError(
            "BF16 was requested but CUDA BF16 support is unavailable; no fallback is used"
        )

    def factory():
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        )

    return (factory, torch.bfloat16)


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(int(seed))
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(int(seed) % 2**32)
    try:
        torch = require_torch()
    except RuntimeError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


@contextlib.contextmanager
def advisory_file_lock(path: str | Path) -> Iterator[None]:
    """Cross-platform cooperative lock based on exclusive lock-file creation."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeContractError(f"Lock already held: {lock_path}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
