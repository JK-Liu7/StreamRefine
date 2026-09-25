"""Stopping policies and continuation-benefit targets for StreamRefine."""

from .benefit import continuation_benefit_targets, smooth_l1_benefit_loss
from .calibration import (
    BenefitCalibration,
    CalibrationExpectation,
    load_benefit_calibration,
    require_benefit_calibration,
)
from .controller import (
    AdaptiveStopController,
    ControllerConfig,
    MethodMode,
    compute_editor_reach_weights,
    compute_suffix_stop_weights,
)

__all__ = [
    "AdaptiveStopController",
    "BenefitCalibration",
    "CalibrationExpectation",
    "ControllerConfig",
    "MethodMode",
    "compute_editor_reach_weights",
    "compute_suffix_stop_weights",
    "continuation_benefit_targets",
    "load_benefit_calibration",
    "require_benefit_calibration",
    "smooth_l1_benefit_loss",
]
