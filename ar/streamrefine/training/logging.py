"""Atomic, categorized JSON training logs and compact console progress."""

from __future__ import annotations
import copy
import datetime
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator, Mapping
from streamrefine.runtime import atomic_write_text, json_safe

_EVENT_ARRAYS = {
    "train": "train_steps",
    "validation": "validation",
    "checkpoint": "checkpoints",
}
_RUN_ARRAYS = (*_EVENT_ARRAYS.values(), "other_events")


def _strict_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r}")


def _sanitize(value: Any) -> Any:
    value = json_safe(value)
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


class StructuredTrainingLog:
    """One strict JSON document; each invocation adds a run, never erases history.

    Writes use fsync plus same-directory atomic replacement. Invalid existing
    JSON is rejected unchanged. An abrupt termination leaves status=running;
    the next start marks that previous run interrupted before appending a run.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        output_dir: str | Path,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_metadata = _sanitize(dict(run_metadata or {}))
        self._active_run_id: str | None = None
        if self.path.exists():
            try:
                self._document = json.loads(
                    self.path.read_text(encoding="utf-8"),
                    parse_constant=_strict_constant,
                )
                self._validate(self._document)
            except (OSError, UnicodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot load training JSON {self.path}: {exc}. The existing file was not modified."
                ) from exc
        else:
            self._document = {
                "format_version": 1,
                "experiment": {"output_dir": str(output_dir)},
                "runs": [],
            }

    @staticmethod
    def _validate(document: Any) -> None:
        if not isinstance(document, dict) or document.get("format_version") != 1:
            raise ValueError("Expected a training log object with format_version=1")
        if not isinstance(document.get("experiment"), dict) or not isinstance(
            document.get("runs"), list
        ):
            raise ValueError("Expected experiment object and runs array")
        ids: set[str] = set()
        for run in document["runs"]:
            if not isinstance(run, dict):
                raise ValueError("Each run must be an object")
            run_id = run.get("run_id")
            if not isinstance(run_id, str) or not run_id or run_id in ids:
                raise ValueError("Invalid or duplicate run_id")
            ids.add(run_id)
            status = run.get("status")
            if status not in {"running", "interrupted", "completed"}:
                raise ValueError("Invalid run status")
            if not isinstance(run.get("start"), dict):
                raise ValueError("Expected run start object")
            for field in _RUN_ARRAYS:
                if not isinstance(run.get(field), list):
                    raise ValueError(f"Expected run {field} array")
            if status == "completed":
                if not isinstance(run.get("end"), dict):
                    raise ValueError("Completed run requires end object")
            elif run.get("end") is not None:
                raise ValueError("Unfinished run must have end=null")
            if status == "interrupted" and (
                not isinstance(run.get("interrupted_at"), str)
            ):
                raise ValueError("Interrupted run requires interrupted_at")

    def log(self, event: str, **fields: Any) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("Log event must be a non-empty string")
        row = _sanitize(
            {
                **self.run_metadata,
                **fields,
                "timestamp": time.time(),
                "time": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "event": event,
            }
        )
        updated = copy.deepcopy(self._document)
        next_active = self._active_run_id
        if event == "train_start":
            if next_active is not None:
                raise RuntimeError("Training logger already has an active run")
            for previous in updated["runs"]:
                if previous["status"] == "running":
                    previous["status"] = "interrupted"
                    previous["interrupted_at"] = row["time"]
            sequence = len(updated["runs"]) + 1
            existing_ids = {run["run_id"] for run in updated["runs"]}
            while f"run_{sequence:04d}" in existing_ids:
                sequence += 1
            next_active = f"run_{sequence:04d}"
            updated["runs"].append(
                {
                    "run_id": next_active,
                    "status": "running",
                    "start": row,
                    **{name: [] for name in _RUN_ARRAYS},
                    "end": None,
                }
            )
        else:
            if next_active is None:
                raise RuntimeError("Cannot log an event without an active training run")
            run = next(
                (item for item in updated["runs"] if item["run_id"] == next_active)
            )
            if event == "complete":
                run["status"], run["end"] = ("completed", row)
                next_active = None
            else:
                run[_EVENT_ARRAYS.get(event, "other_events")].append(row)
        self._validate(updated)
        atomic_write_text(
            self.path,
            json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        self._document, self._active_run_id = (updated, next_active)


def training_log_due(step: int, *, start_step: int, max_steps: int, every: int) -> bool:
    """Cadence is in optimizer updates, with a first/resumed and final update."""
    return step == start_step + 1 or step % every == 0 or step == max_steps


def timed_training_batches(loader: Any) -> Iterator[tuple[int, Any, float]]:
    """Measure exposed DataLoader wait, including iterator startup, using host wall time.

    Consumer work between yields is excluded. No CUDA synchronization is added;
    this measures host waiting for the next batch, not worker CPU or GPU time.
    """
    started = time.perf_counter()
    iterator = iter(loader)
    index = 0
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield (index, batch, time.perf_counter() - started)
        index += 1
        started = time.perf_counter()


def gpu_memory_snapshot(torch: Any, device: Any) -> dict[str, float]:
    """Local-rank allocator stats only, without synchronization or peak reset."""
    if torch.device(device).type != "cuda":
        return {}
    return {
        "gpu/allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "gpu/reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "gpu/process_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
        / 2**30,
        "gpu/process_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def format_progress(event: str, fields: Mapping[str, Any]) -> str:
    """A small subset for the terminal; JSON retains the complete metrics."""
    parts = [f"[StreamRefine][{event}]"]
    if "step" in fields:
        suffix = f"/{fields['max_steps']}" if "max_steps" in fields else ""
        parts.append(f"step={fields['step']}{suffix}")
    if "epoch" in fields:
        parts.append(f"epoch={fields['epoch']}")
    metrics = (
        ("loss", "loss_total"),
        ("flow", "loss_flow_weighted"),
        ("latent", "loss_latent_weighted"),
        ("stop", "loss_stop"),
        ("lr_used", "learning_rate_used"),
        ("grad", "gradient_norm_last_step"),
        ("reached", "mean_reached_states"),
        ("sentinel_count", "sentinel_count"),
        ("wall_s/step", "wall_sec_per_step"),
        ("eta_s", "eta_sec"),
        ("data_s/step", "data_wait_sec_per_step"),
        ("full_objective", "val/full_trajectory_objective"),
        ("adaptive_mae", "val/adaptive_mae"),
        ("psnr", "val/adaptive_psnr"),
        ("ssim", "val/adaptive_ssim"),
        ("mean_steps", "val/adaptive_mean_steps"),
        ("p50_steps", "val/adaptive_p50_steps"),
        ("p95_steps", "val/adaptive_p95_steps"),
        ("wrong_stop_regret", "val/wrong_stop_regret"),
        ("selection", "selection_value"),
        ("best", "best_validation"),
        ("elapsed_s", "elapsed_sec"),
        ("validation_s", "validation_elapsed_sec"),
    )
    for label, key in metrics:
        if fields.get(key) is not None:
            parts.append(f"{label}={float(fields[key]):.5g}")
    for key in ("val/weight_source", "selection_metric", "improved", "path"):
        if key in fields:
            parts.append(f"{key}={fields[key]}")
    return " ".join(parts)


class RunningMeans:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def update(self, values: Mapping[str, Any], *, weight: float = 1.0) -> None:
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            self.sums[name] = self.sums.get(name, 0.0) + number * float(weight)
            self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def compute(self) -> dict[str, float]:
        return {
            name: self.sums[name] / max(self.weights[name], 1e-12)
            for name in sorted(self.sums)
        }

    def reset(self) -> None:
        self.sums.clear()
        self.weights.clear()
