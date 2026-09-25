"""Training, checkpoint, and logging components."""

from .anatomy import (
    ANATOMY_METRICS,
    anatomy_metric_contract,
    anatomy_metric_contract_fingerprint,
    canonical_anatomy_metric,
)
from .checkpoint import CHECKPOINT_VERSION
from .pir import (
    PIR_FORMULA,
    PIR_FORMULA_VERSION,
    PIR_OFFSETS_DHW,
    build_minimal_pir_reference,
    minimal_pir_loss,
    minimal_pir_trajectory,
)

__all__ = [
    "ANATOMY_METRICS",
    "CHECKPOINT_VERSION",
    "PIR_FORMULA",
    "PIR_FORMULA_VERSION",
    "PIR_OFFSETS_DHW",
    "anatomy_metric_contract",
    "anatomy_metric_contract_fingerprint",
    "build_minimal_pir_reference",
    "canonical_anatomy_metric",
    "minimal_pir_loss",
    "minimal_pir_trajectory",
]
