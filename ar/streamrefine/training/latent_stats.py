"""Prepare immutable training statistics before CUDA/process-group initialization."""

from __future__ import annotations
import contextlib
import errno
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from streamrefine.config import pair_modalities
from streamrefine.data.path_resolver import resolve_recorded_path
from streamrefine.runtime import atomic_write_json
from streamrefine.paths import resolve_path_string, runtime_path_remaps


def _log(message: str) -> None:
    print(f"[latent_stats] {message}", flush=True)


@contextlib.contextmanager
def _statistics_lock(
    path: Path, *, timeout_seconds: float, log: Callable[[str], None]
) -> Iterator[None]:
    """An OS lock released even after a killed worker; never unlink its inode."""
    if timeout_seconds <= 0:
        raise ValueError("Latent-statistics lock timeout must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                handle.write(b"\x00")
                handle.flush()

            def acquire() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout_seconds
        waiting = False
        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if not waiting:
                    log(f"Waiting for another process to prepare statistics: {path}")
                    waiting = True
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for latent statistics lock: {path}"
                    ) from exc
                time.sleep(0.2)
        try:
            yield
        finally:
            release()


def _validate_for_config(stats: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    from streamrefine.data.preprocess_medical import (
        canonical_modality_name,
        normalize_dataset_name,
    )

    dataset = normalize_dataset_name(config["data"]["dataset"])
    if stats["dataset"] != dataset:
        raise ValueError(
            f"Latent stats dataset mismatch: expected {dataset}, got {stats['dataset']}"
        )
    required = {canonical_modality_name(value) for value in pair_modalities(config)}
    covered = set(stats["expected_modalities"]) & set(stats["observed_modalities"])
    if not required.issubset(covered):
        raise ValueError(
            f"Latent stats do not cover training pair {config['data']['pair']}: {sorted(covered)}"
        )


def _check_existing(path: Path, config: Mapping[str, Any]) -> None:
    from streamrefine.data.latent_pair_dataset import load_latent_stats

    _validate_for_config(load_latent_stats(path), config)


def _read_attempt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def prepare_training_latent_stats(
    config: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    enabled: bool = True,
    timeout_seconds: float = 86400,
    log: Callable[[str], None] = _log,
) -> Path:
    """Reuse valid stats or generate missing JSON using the offline tool's defaults.

    Every torchrun worker calls this before creating a process group. A file lock
    elects one producer on the shared stats path. Failures are shared with current
    waiters; a later invocation can retry. No training configuration is mutated.
    """
    started_ns = time.time_ns()
    output = Path(config["latent"]["stats_path"]).expanduser().resolve()
    if output.exists():
        _check_existing(output, config)
        log(f"Reusing validated statistics: {output}")
        return output
    if not enabled:
        raise FileNotFoundError(
            f"Latent stats do not exist: {output}; omit --no-auto-latent-stats to generate them."
        )
    if output.suffix.lower() in {".pt", ".pth"}:
        raise ValueError(
            "Automatic latent statistics are JSON; use --latent-stats with a .json path"
        )
    remaps = runtime_path_remaps(config)
    recorded = (
        config["data"]["train_manifest"] if manifest_path is None else manifest_path
    )
    if (
        manifest_path is not None
        and (not str(recorded).startswith("/"))
        and (not Path(recorded).is_absolute())
    ):
        recorded = resolve_path_string(recorded, base_dir=Path.cwd())
    manifest = resolve_recorded_path(
        recorded,
        remaps=remaps,
        base_dir=Path.cwd(),
        must_exist=False,
        role="training modality-cache manifest",
    )
    if manifest_path is None:
        manifest = manifest.with_name("cache_manifest_train.jsonl")
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Automatic latent statistics require the full training modality-cache manifest: {manifest}. Pass --latent-stats-manifest <cache_manifest_train.jsonl> if stored elsewhere. Existing continuous-latent caches are required; pair/val/test manifests cannot replace it."
        )
    lock_path = output.with_name(output.name + ".lock")
    attempt_path = output.with_name(output.name + ".bootstrap.json")
    request = hashlib.sha256(
        json.dumps(
            {
                "manifest": str(manifest),
                "dataset": config["data"]["dataset"],
                "remaps": remaps,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with _statistics_lock(lock_path, timeout_seconds=timeout_seconds, log=log):
        if output.exists():
            _check_existing(output, config)
            log(f"Reusing statistics prepared by another process: {output}")
            return output
        previous = _read_attempt(attempt_path)
        if (
            previous.get("request") == request
            and previous.get("status") == "failed"
            and (previous.get("finished_ns", 0) >= started_ns)
        ):
            raise RuntimeError(
                f"Latent-statistics preparation failed in another process: {previous.get('error')}"
            )
        attempt = {
            "request": request,
            "status": "running",
            "pid": os.getpid(),
            "started_ns": time.time_ns(),
        }
        atomic_write_json(attempt_path, attempt)
        log(
            f"Missing statistics; scanning all training modality caches on CPU: {manifest}"
        )
        scan_start = time.monotonic()
        last_report = scan_start

        def progress(count: int, cache_path: Path) -> None:
            nonlocal last_report
            now = time.monotonic()
            if count == 1 or now - last_report >= 30:
                log(
                    f"Processed {count} training caches in {now - scan_start:.1f}s ({cache_path.name})"
                )
                last_report = now

        try:
            from tools.compute_latent_stats import (
                _atomic_write_json,
                compute_latent_stats,
            )

            stats = compute_latent_stats(
                manifest,
                mask_mode="hard",
                reservoir_size=65536,
                seed=2026,
                std_epsilon=1e-06,
                path_remap=remaps,
                progress_callback=progress,
            )
            _validate_for_config(stats, config)
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}.", dir=output.parent
            ) as staging:
                candidate = Path(staging) / "stats.json"
                _atomic_write_json(candidate, stats, overwrite=False)
                _check_existing(candidate, config)
                if output.exists():
                    _check_existing(output, config)
                else:
                    os.replace(candidate, output)
        except Exception as exc:
            atomic_write_json(
                attempt_path,
                {
                    **attempt,
                    "status": "failed",
                    "finished_ns": time.time_ns(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        atomic_write_json(
            attempt_path,
            {**attempt, "status": "complete", "finished_ns": time.time_ns()},
        )
        log(
            f"Statistics ready: {output} ({stats['files_used']} training caches, {time.monotonic() - scan_start:.1f}s)"
        )
        return output
