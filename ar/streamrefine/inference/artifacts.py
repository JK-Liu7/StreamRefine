"""Recoverable per-case trajectory and middle-slice artifact writer."""

from __future__ import annotations
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from streamrefine.runtime import (
    advisory_file_lock,
    atomic_write_json,
    atomic_write_text,
    json_safe,
)


def safe_case_id(value: str) -> str:
    cleaned = re.sub("[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return cleaned or "case"


def _middle_views(volume: Any) -> dict[str, Any]:
    import torch

    tensor = torch.as_tensor(volume).detach().float().cpu()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError("Middle-slice writer accepts a single volume")
        tensor = tensor[0]
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Expected [1,D,H,W], got {tuple(tensor.shape)}")
    image = tensor[0]
    return {
        "axial": image[image.shape[0] // 2],
        "coronal": image[:, image.shape[1] // 2, :],
        "sagittal": image[:, :, image.shape[2] // 2],
    }


def _save_png_atomic(path: Path, image: Any) -> None:
    import numpy as np
    import torch
    from PIL import Image

    tensor = torch.as_tensor(image).float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Cannot save non-finite slice to {path}")
    scaled = ((tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        Image.fromarray(np.asarray(scaled), mode="L").save(tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def save_middle_slices(case_dir: str | Path, prefix: str, volume: Any) -> list[str]:
    directory = Path(case_dir)
    saved: list[str] = []
    for view, image in _middle_views(volume).items():
        path = directory / f"{prefix}_mid_{view}.png"
        _save_png_atomic(path, image)
        saved.append(str(path))
    return saved


class InferenceArtifactWriter:
    def __init__(self, root: str | Path, *, contract_fingerprint: str) -> None:
        self.root = Path(root)
        self.cases_dir = self.root / "cases"
        self.contract_fingerprint = str(contract_fingerprint)

    def case_dir(self, case_id: str) -> Path:
        return self.cases_dir / safe_case_id(case_id)

    def completed_record(self, case_id: str) -> dict[str, Any] | None:
        path = self.case_dir(case_id) / "trajectory.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            return None
        if (
            value.get("status") != "complete"
            or value.get("contract_fingerprint") != self.contract_fingerprint
        ):
            return None
        return dict(value)

    def write_case(
        self,
        *,
        case_id: str,
        source: Any,
        target: Any,
        decoded_states: Sequence[Any],
        state_records: Sequence[Mapping[str, Any]],
        case_metadata: Mapping[str, Any],
        save_mid_slices: bool = True,
        save_all_states: bool = True,
    ) -> dict[str, Any]:
        if not decoded_states or len(decoded_states) != len(state_records):
            raise ValueError(
                "decoded_states and state_records must be non-empty and aligned"
            )
        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        if save_mid_slices:
            files.extend(save_middle_slices(directory, "source", source))
            files.extend(save_middle_slices(directory, "gt", target))
            if save_all_states:
                for index, state in enumerate(decoded_states, 1):
                    files.extend(
                        save_middle_slices(directory, f"state_{index:02d}", state)
                    )
            files.extend(save_middle_slices(directory, "final", decoded_states[-1]))
        record = {
            "status": "complete",
            "contract_fingerprint": self.contract_fingerprint,
            "case_id": str(case_id),
            "metadata": json_safe(case_metadata),
            "states": json_safe(list(state_records)),
            "stop_step": len(state_records),
            "artifacts": [str(Path(path).relative_to(directory)) for path in files],
        }
        atomic_write_json(directory / "trajectory.json", record)
        self.rebuild_aggregate_files()
        return record

    def _compatible_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.cases_dir.is_dir():
            return records
        for path in sorted(self.cases_dir.glob("*/trajectory.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(value, Mapping)
                and value.get("status") == "complete"
                and (value.get("contract_fingerprint") == self.contract_fingerprint)
            ):
                records.append(dict(value))
        return records

    def rebuild_aggregate_files(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(self.root / ".metrics.lock"):
            records = self._compatible_records()
            rows: list[dict[str, Any]] = []
            calibration_rows: list[dict[str, Any]] = []
            final_metrics: dict[str, list[float]] = {}
            for record in records:
                states = record.get("states", [])
                metadata = record.get("metadata", {})
                for state in states:
                    if not isinstance(state, Mapping):
                        continue
                    metrics = state.get("metrics", {})
                    row = {
                        "contract_fingerprint": self.contract_fingerprint,
                        "case_id": record.get("case_id"),
                        "state": state.get("step"),
                        "is_final": state is states[-1],
                        **(dict(metrics) if isinstance(metrics, Mapping) else {}),
                    }
                    rows.append(row)
                if states and isinstance(states[-1], Mapping):
                    metrics = states[-1].get("metrics", {})
                    if isinstance(metrics, Mapping):
                        for name, value in metrics.items():
                            number = float(value)
                            if math.isfinite(number):
                                final_metrics.setdefault(str(name), []).append(number)
                if isinstance(metadata, Mapping) and bool(
                    metadata.get("calibration_export")
                ):
                    anatomy_metric = (
                        str(metadata.get("anatomy_metric", "")).strip().lower()
                    )
                    if anatomy_metric not in {"pir"}:
                        raise ValueError(
                            "Calibration export metadata must select anatomy_metric 'pir'"
                        )
                    anatomy_contract = metadata.get("anatomy_contract")
                    if (
                        not isinstance(anatomy_contract, Mapping)
                        or not anatomy_contract
                    ):
                        raise ValueError(
                            "Calibration export metadata lacks canonical anatomy_contract"
                        )
                    anatomy_contract_fingerprint = str(
                        metadata.get("anatomy_contract_fingerprint", "")
                    ).strip()
                    if not anatomy_contract_fingerprint:
                        raise ValueError(
                            "Calibration export metadata lacks anatomy_contract_fingerprint"
                        )
                    from streamrefine.training.translation import (
                        translation_metric_identity,
                    )

                    translation_identity = translation_metric_identity(
                        metadata.get("translation_metric")
                    )
                    for key, expected in translation_identity.items():
                        if metadata.get(key) != expected:
                            raise ValueError(
                                f"Calibration export metadata has incompatible {key}"
                            )
                    calibration_row = {
                        **translation_identity,
                        "case_id": record.get("case_id"),
                        "dataset": metadata.get("dataset"),
                        "source_modality": metadata.get("source_modality"),
                        "target_modality": metadata.get("target_modality"),
                        "split": metadata.get("split"),
                        "fixed_horizon_checkpoint_identity": metadata.get(
                            "fixed_horizon_checkpoint_identity"
                        ),
                        "latent_statistics_identity": metadata.get(
                            "latent_statistics_identity"
                        ),
                        "anatomy_metric": anatomy_metric,
                        "anatomy_contract": dict(anatomy_contract),
                        "anatomy_contract_fingerprint": anatomy_contract_fingerprint,
                        "cache_provenance": metadata.get("cache_provenance"),
                        "cache_provenance_fingerprint": metadata.get(
                            "cache_provenance_fingerprint"
                        ),
                        "trajectory_contract_fingerprint": metadata.get(
                            "trajectory_contract_fingerprint",
                            record.get("contract_fingerprint"),
                        ),
                        "k_max": metadata.get("k_max"),
                        "states": [
                            {
                                "translation_loss": state.get("translation_loss"),
                                "anatomy_loss": state.get("anatomy_loss"),
                            }
                            for state in states
                            if isinstance(state, Mapping)
                        ],
                    }
                    mask_policy = str(metadata.get("anatomy_mask_policy", "")).strip()
                    mask_contract = metadata.get("anatomy_mask_contract")
                    mask_fingerprint = str(
                        metadata.get("anatomy_mask_contract_fingerprint", "")
                    ).strip()
                    if (
                        not mask_policy
                        or not isinstance(mask_contract, Mapping)
                        or (not mask_contract)
                        or (not mask_fingerprint)
                    ):
                        raise ValueError(
                            "PIR calibration export metadata lacks its complete dataset anatomy-mask contract"
                        )
                    support_cell_count = int(metadata.get("pir_support_cell_count", 0))
                    valid_edge_count = int(metadata.get("pir_valid_edge_count", 0))
                    if support_cell_count <= 0 or valid_edge_count <= 0:
                        raise ValueError(
                            "PIR calibration export metadata must record positive support-cell and valid-edge counts"
                        )
                    calibration_row.update(
                        {
                            "anatomy_mask_policy": mask_policy,
                            "anatomy_mask_contract": dict(mask_contract),
                            "anatomy_mask_contract_fingerprint": mask_fingerprint,
                            "pir_support_cell_count": support_cell_count,
                            "pir_valid_edge_count": valid_edge_count,
                        }
                    )
                    calibration_rows.append(calibration_row)
            text = "".join(
                (json.dumps(json_safe(row), sort_keys=True) + "\n" for row in rows)
            )
            atomic_write_text(self.root / "metrics.jsonl", text)
            calibration_text = "".join(
                (
                    json.dumps(json_safe(row), sort_keys=True) + "\n"
                    for row in calibration_rows
                )
            )
            atomic_write_text(
                self.root / "calibration_trajectories.jsonl", calibration_text
            )
            summary = {
                "contract_fingerprint": self.contract_fingerprint,
                "completed_cases": len(records),
                "calibration_trajectory_cases": len(calibration_rows),
                "final_metrics": {
                    name: {
                        "mean": sum(values) / len(values),
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                    }
                    for name, values in sorted(final_metrics.items())
                },
            }
            atomic_write_json(self.root / "summary.json", summary)
