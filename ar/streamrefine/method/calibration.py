"""Fail-closed loading of anatomy-aware continuation-benefit calibration."""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

CALIBRATION_SCHEMA_VERSION = "streamrefine_benefit_calibration_v5_translation_bound"
CALIBRATION_IMPLEMENTATION_VERSION = "streamrefine_benefit_v4_translation_bound"
_COMMON_REQUIRED_FIELDS = {
    "schema_version",
    "implementation_version",
    "translation_scale",
    "translation_metric",
    "translation_contract",
    "translation_contract_fingerprint",
    "anatomy_scale",
    "anatomy_metric",
    "anatomy_contract",
    "anatomy_contract_fingerprint",
    "dataset",
    "source_modality",
    "target_modality",
    "split",
    "fixed_horizon_checkpoint_identity",
    "latent_statistics_identity",
    "cache_provenance",
    "cache_provenance_fingerprint",
    "trajectory_contract_fingerprint",
    "sample_count",
    "seed",
    "k_max",
}
_PIR_ANATOMY_MASK_FIELDS = {
    "anatomy_mask_policy",
    "anatomy_mask_contract",
    "anatomy_mask_contract_fingerprint",
}
_OPTIONAL_FIELDS = {"scale_estimator", "metadata"}


def _canonical_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"calibration {field_name} must be a non-empty string")
    return value.strip().lower()


def _anatomy_metric(value: object) -> str:
    result = _canonical_name(value, "anatomy_metric")
    if result not in {"pir"}:
        raise ValueError("calibration anatomy_metric must be 'pir'")
    return result


def _validate_anatomy_contract(
    metric: str, contract_value: object, fingerprint_value: object
) -> tuple[dict[str, Any], str]:
    contract = _json_mapping(contract_value, "anatomy_contract")
    fingerprint = _identity(fingerprint_value, "anatomy_contract_fingerprint")
    from streamrefine.training.anatomy import (
        anatomy_metric_contract,
        anatomy_metric_contract_fingerprint,
    )

    active_contract = anatomy_metric_contract(metric)
    active_fingerprint = anatomy_metric_contract_fingerprint(metric)
    if _canonical_json(contract) != _canonical_json(active_contract):
        raise ValueError(
            f"calibration anatomy_contract does not match the active {metric!r} formula contract"
        )
    if fingerprint != _mapping_fingerprint(contract):
        raise ValueError(
            "calibration anatomy_contract_fingerprint does not match anatomy_contract"
        )
    if fingerprint != active_fingerprint:
        raise ValueError(
            f"calibration anatomy_contract_fingerprint does not match the active {metric!r} formula contract"
        )
    return (contract, fingerprint)


def _validate_pir_anatomy_mask_contract(
    dataset: str,
    policy_value: object,
    contract_value: object,
    fingerprint_value: object,
) -> tuple[str, dict[str, Any], str]:
    from streamrefine.data.anatomy_mask import (
        anatomy_mask_contract,
        anatomy_mask_contract_fingerprint,
        canonical_anatomy_mask_policy,
    )

    policy = canonical_anatomy_mask_policy(dataset, policy_value)
    contract = _json_mapping(contract_value, "anatomy_mask_contract")
    fingerprint = _identity(fingerprint_value, "anatomy_mask_contract_fingerprint")
    active_contract = anatomy_mask_contract(dataset, policy)
    active_fingerprint = anatomy_mask_contract_fingerprint(dataset, policy)
    if _canonical_json(contract) != _canonical_json(active_contract):
        raise ValueError(
            f"calibration anatomy_mask_contract does not match the active dataset policy for {dataset!r}"
        )
    if fingerprint != _mapping_fingerprint(contract):
        raise ValueError(
            "calibration anatomy_mask_contract_fingerprint does not match anatomy_mask_contract"
        )
    if fingerprint != active_fingerprint:
        raise ValueError(
            f"calibration anatomy_mask_contract_fingerprint does not match the active dataset policy for {dataset!r}"
        )
    return (policy, contract, fingerprint)


def _identity(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"calibration {field_name} must be a non-empty string")
    return value.strip()


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"calibration {field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration {field_name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"calibration {field_name} must be finite and > 0")
    return result


def _non_negative_int(value: object, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"calibration {field_name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"calibration {field_name} must be {relation}")
    return int(value)


def _json_mapping(
    value: object, field_name: str, *, allow_empty: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"calibration {field_name} must be a JSON object")
    result = dict(value)
    if not allow_empty and (not result):
        raise ValueError(f"calibration {field_name} must not be empty")
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"calibration {field_name} must contain valid finite JSON values"
        ) from exc
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationExpectation:
    """Run/config identities that an artifact must match exactly when supplied."""

    dataset: Optional[str] = None
    translation_metric: Optional[str] = None
    translation_contract: Optional[Mapping[str, Any]] = None
    translation_contract_fingerprint: Optional[str] = None
    source_modality: Optional[str] = None
    target_modality: Optional[str] = None
    split: Optional[str] = None
    fixed_horizon_checkpoint_identity: Optional[str] = None
    latent_statistics_identity: Optional[str] = None
    anatomy_metric: Optional[str] = None
    anatomy_contract: Optional[Mapping[str, Any]] = None
    anatomy_contract_fingerprint: Optional[str] = None
    anatomy_mask_policy: Optional[str] = None
    anatomy_mask_contract: Optional[Mapping[str, Any]] = None
    anatomy_mask_contract_fingerprint: Optional[str] = None
    cache_provenance: Optional[Mapping[str, Any]] = None
    trajectory_contract_fingerprint: Optional[str] = None
    k_max: Optional[int] = None
    implementation_version: str = CALIBRATION_IMPLEMENTATION_VERSION

    def assert_complete(self) -> None:
        required = {
            "translation_metric": self.translation_metric,
            "translation_contract": self.translation_contract,
            "translation_contract_fingerprint": self.translation_contract_fingerprint,
            "dataset": self.dataset,
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "split": self.split,
            "fixed_horizon_checkpoint_identity": self.fixed_horizon_checkpoint_identity,
            "latent_statistics_identity": self.latent_statistics_identity,
            "anatomy_metric": self.anatomy_metric,
            "anatomy_contract": self.anatomy_contract,
            "anatomy_contract_fingerprint": self.anatomy_contract_fingerprint,
            "cache_provenance": self.cache_provenance,
            "trajectory_contract_fingerprint": self.trajectory_contract_fingerprint,
            "k_max": self.k_max,
            "implementation_version": self.implementation_version,
        }
        missing = sorted((name for name, value in required.items() if value is None))
        if missing:
            raise ValueError(
                "anatomy-aware calibration expectations are incomplete: "
                + ", ".join(missing)
            )
        _anatomy_metric(self.anatomy_metric)
        mask_values = {
            "anatomy_mask_policy": self.anatomy_mask_policy,
            "anatomy_mask_contract": self.anatomy_mask_contract,
            "anatomy_mask_contract_fingerprint": self.anatomy_mask_contract_fingerprint,
        }
        missing_mask = sorted(
            (name for name, value in mask_values.items() if value is None)
        )
        if missing_mask:
            raise ValueError(
                "PIR calibration expectations are incomplete: "
                + ", ".join(missing_mask)
            )


@dataclass(frozen=True)
class BenefitCalibration:
    """Validated robust scales and the provenance that makes them applicable."""

    translation_scale: float
    translation_metric: str
    translation_contract: Mapping[str, Any]
    translation_contract_fingerprint: str
    anatomy_scale: float
    anatomy_metric: str
    anatomy_contract: Mapping[str, Any]
    anatomy_contract_fingerprint: str
    dataset: str
    source_modality: str
    target_modality: str
    split: str
    fixed_horizon_checkpoint_identity: str
    latent_statistics_identity: str
    cache_provenance: Mapping[str, Any]
    cache_provenance_fingerprint: str
    trajectory_contract_fingerprint: str
    sample_count: int
    seed: int
    k_max: int
    anatomy_mask_policy: Optional[str] = None
    anatomy_mask_contract: Optional[Mapping[str, Any]] = None
    anatomy_mask_contract_fingerprint: Optional[str] = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    implementation_version: str = CALIBRATION_IMPLEMENTATION_VERSION
    scale_estimator: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BenefitCalibration":
        if not isinstance(payload, Mapping):
            raise ValueError("benefit calibration must be a JSON object")
        keys = set(payload)
        if "schema_version" not in payload:
            raise ValueError(
                "benefit calibration is missing required field: schema_version"
            )
        schema = _identity(payload["schema_version"], "schema_version")
        if schema != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"incompatible calibration schema {schema!r}; expected {CALIBRATION_SCHEMA_VERSION!r}"
            )
        missing = sorted(_COMMON_REQUIRED_FIELDS - keys)
        if missing:
            raise ValueError(
                f"benefit calibration is missing required fields: {missing}"
            )
        from streamrefine.training.translation import translation_metric_identity

        translation_identity = translation_metric_identity(
            payload["translation_metric"]
        )
        for name, expected in translation_identity.items():
            if payload[name] != expected:
                raise ValueError(
                    f"calibration {name} does not match the active translation contract"
                )
        anatomy_metric = _anatomy_metric(payload["anatomy_metric"])
        missing_mask = sorted(_PIR_ANATOMY_MASK_FIELDS - keys)
        if missing_mask:
            raise ValueError(
                f"PIR benefit calibration is missing required anatomy-mask fields: {missing_mask}"
            )
        allowed = _COMMON_REQUIRED_FIELDS | _PIR_ANATOMY_MASK_FIELDS | _OPTIONAL_FIELDS
        unexpected = sorted(keys - allowed)
        if unexpected:
            raise ValueError(
                f"benefit calibration has unsupported top-level fields; put extensions under metadata: {unexpected}"
            )
        implementation = _identity(
            payload["implementation_version"], "implementation_version"
        )
        if implementation != CALIBRATION_IMPLEMENTATION_VERSION:
            raise ValueError(
                f"incompatible calibration implementation {implementation!r}; expected {CALIBRATION_IMPLEMENTATION_VERSION!r}"
            )
        cache_provenance = _json_mapping(
            payload["cache_provenance"], "cache_provenance"
        )
        estimator = _json_mapping(
            payload.get("scale_estimator", {}), "scale_estimator", allow_empty=True
        )
        metadata = _json_mapping(
            payload.get("metadata", {}), "metadata", allow_empty=True
        )
        cache_fingerprint = _identity(
            payload["cache_provenance_fingerprint"], "cache_provenance_fingerprint"
        )
        expected_cache_fingerprint = _mapping_fingerprint(cache_provenance)
        if cache_fingerprint != expected_cache_fingerprint:
            raise ValueError(
                "calibration cache_provenance_fingerprint does not match cache_provenance"
            )
        anatomy_contract, anatomy_contract_fingerprint = _validate_anatomy_contract(
            anatomy_metric,
            payload["anatomy_contract"],
            payload["anatomy_contract_fingerprint"],
        )
        dataset = _canonical_name(payload["dataset"], "dataset")
        anatomy_mask_policy: Optional[str] = None
        anatomy_mask_contract: Optional[dict[str, Any]] = None
        anatomy_mask_contract_fingerprint: Optional[str] = None
        (
            anatomy_mask_policy,
            anatomy_mask_contract,
            anatomy_mask_contract_fingerprint,
        ) = _validate_pir_anatomy_mask_contract(
            dataset,
            payload["anatomy_mask_policy"],
            payload["anatomy_mask_contract"],
            payload["anatomy_mask_contract_fingerprint"],
        )
        return cls(
            **translation_identity,
            translation_scale=_positive_float(
                payload["translation_scale"], "translation_scale"
            ),
            anatomy_scale=_positive_float(payload["anatomy_scale"], "anatomy_scale"),
            anatomy_metric=anatomy_metric,
            anatomy_contract=anatomy_contract,
            anatomy_contract_fingerprint=anatomy_contract_fingerprint,
            dataset=dataset,
            source_modality=_canonical_name(
                payload["source_modality"], "source_modality"
            ),
            target_modality=_canonical_name(
                payload["target_modality"], "target_modality"
            ),
            split=_canonical_name(payload["split"], "split"),
            fixed_horizon_checkpoint_identity=_identity(
                payload["fixed_horizon_checkpoint_identity"],
                "fixed_horizon_checkpoint_identity",
            ),
            latent_statistics_identity=_identity(
                payload["latent_statistics_identity"], "latent_statistics_identity"
            ),
            cache_provenance=cache_provenance,
            cache_provenance_fingerprint=cache_fingerprint,
            trajectory_contract_fingerprint=_identity(
                payload["trajectory_contract_fingerprint"],
                "trajectory_contract_fingerprint",
            ),
            sample_count=_non_negative_int(
                payload["sample_count"], "sample_count", positive=True
            ),
            seed=_non_negative_int(payload["seed"], "seed"),
            k_max=_non_negative_int(payload["k_max"], "k_max", positive=True),
            anatomy_mask_policy=anatomy_mask_policy,
            anatomy_mask_contract=anatomy_mask_contract,
            anatomy_mask_contract_fingerprint=anatomy_mask_contract_fingerprint,
            schema_version=schema,
            implementation_version=implementation,
            scale_estimator=estimator,
            metadata=metadata,
        )

    @property
    def pair(self) -> str:
        return f"{self.source_modality}->{self.target_modality}"

    def to_dict(self) -> dict[str, Any]:
        metric = _anatomy_metric(self.anatomy_metric)
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "implementation_version": self.implementation_version,
            "translation_scale": self.translation_scale,
            "translation_metric": self.translation_metric,
            "translation_contract": dict(self.translation_contract),
            "translation_contract_fingerprint": self.translation_contract_fingerprint,
            "anatomy_scale": self.anatomy_scale,
            "anatomy_metric": metric,
            "anatomy_contract": dict(self.anatomy_contract),
            "anatomy_contract_fingerprint": self.anatomy_contract_fingerprint,
            "dataset": self.dataset,
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "split": self.split,
            "fixed_horizon_checkpoint_identity": self.fixed_horizon_checkpoint_identity,
            "latent_statistics_identity": self.latent_statistics_identity,
            "cache_provenance": dict(self.cache_provenance),
            "cache_provenance_fingerprint": self.cache_provenance_fingerprint,
            "trajectory_contract_fingerprint": self.trajectory_contract_fingerprint,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "k_max": self.k_max,
            "scale_estimator": dict(self.scale_estimator),
            "metadata": dict(self.metadata),
        }
        mask_values = (
            self.anatomy_mask_policy,
            self.anatomy_mask_contract,
            self.anatomy_mask_contract_fingerprint,
        )
        if any((value is None for value in mask_values)):
            raise ValueError("PIR calibration lacks its required anatomy-mask contract")
        payload.update(
            {
                "anatomy_mask_policy": self.anatomy_mask_policy,
                "anatomy_mask_contract": dict(self.anatomy_mask_contract or {}),
                "anatomy_mask_contract_fingerprint": self.anatomy_mask_contract_fingerprint,
            }
        )
        return payload

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def assert_compatible(self, expected: CalibrationExpectation) -> None:
        comparisons = {
            "dataset": (
                self.dataset,
                None
                if expected.dataset is None
                else _canonical_name(expected.dataset, "dataset"),
            ),
            "source_modality": (
                self.source_modality,
                None
                if expected.source_modality is None
                else _canonical_name(expected.source_modality, "source_modality"),
            ),
            "target_modality": (
                self.target_modality,
                None
                if expected.target_modality is None
                else _canonical_name(expected.target_modality, "target_modality"),
            ),
            "split": (
                self.split,
                None
                if expected.split is None
                else _canonical_name(expected.split, "split"),
            ),
            "fixed_horizon_checkpoint_identity": (
                self.fixed_horizon_checkpoint_identity,
                expected.fixed_horizon_checkpoint_identity,
            ),
            "latent_statistics_identity": (
                self.latent_statistics_identity,
                expected.latent_statistics_identity,
            ),
            "translation_metric": (
                self.translation_metric,
                expected.translation_metric,
            ),
            "translation_contract": (
                dict(self.translation_contract),
                expected.translation_contract,
            ),
            "translation_contract_fingerprint": (
                self.translation_contract_fingerprint,
                expected.translation_contract_fingerprint,
            ),
            "anatomy_metric": (
                self.anatomy_metric,
                None
                if expected.anatomy_metric is None
                else _anatomy_metric(expected.anatomy_metric),
            ),
            "anatomy_contract_fingerprint": (
                self.anatomy_contract_fingerprint,
                expected.anatomy_contract_fingerprint,
            ),
            "anatomy_mask_policy": (
                self.anatomy_mask_policy,
                None
                if expected.anatomy_mask_policy is None
                else str(expected.anatomy_mask_policy).strip().lower(),
            ),
            "anatomy_mask_contract_fingerprint": (
                self.anatomy_mask_contract_fingerprint,
                expected.anatomy_mask_contract_fingerprint,
            ),
            "k_max": (self.k_max, expected.k_max),
            "implementation_version": (
                self.implementation_version,
                expected.implementation_version,
            ),
            "trajectory_contract_fingerprint": (
                self.trajectory_contract_fingerprint,
                expected.trajectory_contract_fingerprint,
            ),
        }
        mismatches = [
            f"{name}: artifact={actual!r}, expected={wanted!r}"
            for name, (actual, wanted) in comparisons.items()
            if wanted is not None and actual != wanted
        ]
        if expected.anatomy_contract is not None:
            wanted_anatomy_contract = _json_mapping(
                expected.anatomy_contract, "expected anatomy_contract"
            )
            if _canonical_json(dict(self.anatomy_contract)) != _canonical_json(
                wanted_anatomy_contract
            ):
                mismatches.append(
                    "anatomy_contract differs from the active formula contract"
                )
        if expected.anatomy_mask_contract is not None:
            wanted_mask_contract = _json_mapping(
                expected.anatomy_mask_contract, "expected anatomy_mask_contract"
            )
            if self.anatomy_mask_contract is None or _canonical_json(
                dict(self.anatomy_mask_contract)
            ) != _canonical_json(wanted_mask_contract):
                mismatches.append(
                    "anatomy_mask_contract differs from the active dataset policy"
                )
        if expected.cache_provenance is not None:
            wanted_provenance = _json_mapping(
                expected.cache_provenance, "expected cache_provenance"
            )
            if _canonical_json(dict(self.cache_provenance)) != _canonical_json(
                wanted_provenance
            ):
                mismatches.append(
                    "cache_provenance differs from the active cache contract"
                )
            wanted_fingerprint = _mapping_fingerprint(wanted_provenance)
            if self.cache_provenance_fingerprint != wanted_fingerprint:
                mismatches.append(
                    "cache_provenance_fingerprint differs from the active cache contract"
                )
        if mismatches:
            raise ValueError(
                "incompatible benefit calibration: " + "; ".join(mismatches)
            )


def load_benefit_calibration(
    path: str | Path, *, expected: Optional[CalibrationExpectation] = None
) -> BenefitCalibration:
    """Load an explicitly named artifact and validate every configured identity."""
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"benefit calibration does not exist: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"benefit calibration is not valid JSON: {artifact_path}"
        ) from exc
    calibration = BenefitCalibration.from_mapping(payload)
    if expected is not None:
        calibration.assert_compatible(expected)
    return calibration


def save_benefit_calibration(calibration: BenefitCalibration, path: str | Path) -> Path:
    """Write a deterministic artifact; intended for the explicit offline tool."""
    artifact_path = Path(path).expanduser()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_name(artifact_path.name + ".tmp")
    temporary.write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(artifact_path)
    return artifact_path


def require_benefit_calibration(
    mode: object,
    path: Optional[str | Path],
    *,
    expected: Optional[CalibrationExpectation] = None,
) -> Optional[BenefitCalibration]:
    """Fail closed at anatomy-aware startup; other modes do not touch the path."""
    mode_value = getattr(mode, "value", mode)
    normalized = str(mode_value).strip().lower()
    if normalized != "anatomy_aware":
        return None
    if path is None or not str(path).strip():
        raise ValueError("anatomy_aware mode requires benefit_calibration.json")
    if expected is None:
        raise ValueError(
            "anatomy_aware mode requires explicit calibration compatibility expectations"
        )
    expected.assert_complete()
    return load_benefit_calibration(path, expected=expected)


load_calibration = load_benefit_calibration
require_calibration = require_benefit_calibration
__all__ = [
    "CALIBRATION_IMPLEMENTATION_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "BenefitCalibration",
    "CalibrationExpectation",
    "load_benefit_calibration",
    "load_calibration",
    "require_benefit_calibration",
    "require_calibration",
    "save_benefit_calibration",
]
