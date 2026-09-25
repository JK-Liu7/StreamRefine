"""Per-rank calibration progress; timing never enters scientific artifacts."""

from __future__ import annotations
import contextlib
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any


class CalibrationProgress:
    """Append phase/case events and display a local-case tqdm below training."""

    def __init__(
        self,
        *,
        output_dir: Path,
        step: int,
        rank: int,
        world_size: int,
        device: Any,
        training_bar: bool = False,
    ) -> None:
        self.path = Path(output_dir) / "calibration" / f"progress_rank_{rank:04d}.jsonl"
        self.step, self.rank, self.world_size = (int(step), int(rank), int(world_size))
        self.device = device
        self.position = 1 if training_bar and rank == 0 else 0
        self.bar = self.stream = None
        self.started = time.perf_counter()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.started = time.perf_counter()
        self.emit("runtime_start", progress_path=str(self.path))
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.finish_cases()
            self.emit(
                "runtime_failed" if exc is not None else "runtime_complete",
                elapsed_sec=time.perf_counter() - self.started,
                **{"error": f"{exc_type.__name__}: {exc}"} if exc is not None else {},
            )
        finally:
            if self.stream is not None:
                self.stream.close()
                self.stream = None

    def emit(self, event: str, **fields: Any) -> None:
        from tqdm.auto import tqdm

        row = {
            "event": f"calibration_{event}",
            "step": self.step,
            "rank": self.rank,
            "world_size": self.world_size,
            "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "timestamp": time.time(),
            **fields,
        }
        if self.stream is not None:
            self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.stream.flush()
        details = " ".join(
            (
                f"{key}={value:.3f}s"
                if key.endswith("_sec") and isinstance(value, (int, float))
                else f"{key}={value}"
                for key, value in fields.items()
            )
        )
        tqdm.write(
            f"[StreamRefine][calibration_{event}] step={self.step} rank={self.rank}/{self.world_size} {details}"
        )
        sys.stdout.flush()

    def _synchronize(self) -> None:
        if getattr(self.device, "type", str(self.device).split(":")[0]) == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def stage(self, stage: str, **fields: Any):
        gpu_stage = stage in {"trajectory", "latent_metrics", "decode_and_metrics"}
        if self.bar is not None:
            self.bar.set_postfix(
                stage=stage, case_id=fields.get("case_id", ""), refresh=True
            )
        self.emit("stage_start", stage=stage, **fields)
        if gpu_stage:
            self._synchronize()
        started = time.perf_counter()
        try:
            yield
            if gpu_stage:
                self._synchronize()
        except BaseException as exc:
            self.emit(
                "stage_failed",
                stage=stage,
                elapsed_sec=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
                **fields,
            )
            raise
        else:
            self.emit(
                "stage_complete",
                stage=stage,
                elapsed_sec=time.perf_counter() - started,
                **fields,
            )

    def start_cases(self, *, local_count: int, global_count: int) -> None:
        from tqdm.auto import tqdm

        self.emit("cases_start", local_cases=local_count, global_cases=global_count)
        self.bar = tqdm(
            total=local_count,
            desc=f"Calibration rank={self.rank}",
            unit="case",
            position=self.position,
            dynamic_ncols=True,
            leave=True,
        )

    def case_complete(self, **fields: Any) -> None:
        self.emit("case_complete", **fields)
        if self.bar is not None:
            self.bar.update(1)

    def finish_cases(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
