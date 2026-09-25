"""Pure-Python policy replay and aggregation for StreamRefine validation.

This module intentionally imports only the Python standard library.  It accepts
already-computed, per-case full trajectories and never opens datasets, manifests,
latent caches, checkpoints, or model weights.  The deployment policy is deterministic:
training-only sentinel continuation and policy warmup are deliberately absent.

Each input record has the following compact contract::

    {
        "case_id": "case-001",
        "difficulty": 0.3,                  # optional; defaults to state-1 L
        "states": [
            {
                "step": 1,                 # optional, but must match list position
                "benefit_score": 0.2,
                "translation_loss": 0.4,   # falls back to metrics.mae
                "anatomy_loss": 0.1,       # required only when lambda_anatomy > 0
                "metrics": {"mae": 0.4, "psnr": 20.0, "ssim": 0.8},
            },
            ...
        ],
    }

All steps are one-indexed in public results.  ``summarize_validation_records`` first
deduplicates records by ``case_id`` so padding introduced by a distributed sampler
cannot silently bias means, histograms, or quantiles.  Conflicting duplicates fail
closed.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

DEPLOYMENT_METHOD_MODES = frozenset({"fixed_k", "anatomy_aware"})
DEFAULT_METRIC_NAMES = ("mae", "psnr", "ssim")


@dataclass(frozen=True)
class OracleResult:
    """Earliest (within tolerance) minimizer of a complete utility curve."""

    step: int
    utility: float
    minimum_utility: float


@dataclass(frozen=True)
class PolicyReplay:
    """Result of replaying the deterministic deployment policy on one trajectory."""

    stop_step: int
    forced_stop: bool
    selected_state: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedCase:
    case_id: str
    states: tuple[Mapping[str, Any], ...]
    scores: tuple[float, ...]
    translation_losses: tuple[float, ...]
    anatomy_losses: tuple[float, ...] | None
    metrics: tuple[Mapping[str, float], ...]
    utilities: tuple[float, ...]
    continuation_benefits: tuple[float, ...]
    oracle_step: int
    oracle_utility: float
    oracle_minimum_utility: float
    difficulty: float


@dataclass(frozen=True)
class _PolicyCase:
    prepared: _PreparedCase
    stop_step: int
    oracle_global_regret: float
    suffix_regret: float


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _nonnegative_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _numeric_sequence(values: Sequence[Any], name: str) -> tuple[float, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or (not values)
    ):
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(
        (
            _finite_number(value, f"{name}[{index}]")
            for index, value in enumerate(values)
        )
    )


def _normalize_mode(mode: str) -> str:
    result = str(mode).strip().lower()
    if result not in DEPLOYMENT_METHOD_MODES:
        raise ValueError(
            f"unsupported deployment mode {mode!r}; expected one of {sorted(DEPLOYMENT_METHOD_MODES)}"
        )
    return result


def deployment_stop_step(
    benefit_scores: Sequence[Any],
    *,
    mode: str,
    stop_threshold: float = 0.0,
    k_max: int | None = None,
) -> int:
    """Return the one-indexed deterministic deployment stop step.

    ``fixed_k`` ignores all learned scores and always returns ``k_max``.  Every other
    mode stops at the first non-terminal score at or below the configured threshold;
    the last step is forced.  Training-only sentinel exploration and policy warmup are
    intentionally not parameters of this function.
    """
    scores = _numeric_sequence(benefit_scores, "benefit_scores")
    horizon = len(scores) if k_max is None else int(k_max)
    if isinstance(k_max, bool) or horizon <= 0:
        raise ValueError("k_max must be a positive integer")
    if len(scores) != horizon:
        raise ValueError("benefit_scores must contain exactly k_max entries")
    selected_mode = _normalize_mode(mode)
    threshold = _finite_number(stop_threshold, "stop_threshold")
    if selected_mode == "fixed_k":
        return horizon
    for index, score in enumerate(scores[:-1], 1):
        if deployment_should_stop(
            score,
            step=index,
            k_max=horizon,
            mode=selected_mode,
            stop_threshold=threshold,
        ):
            return index
    return horizon


def deployment_should_stop(
    benefit_score: Any, *, step: int, k_max: int, mode: str, stop_threshold: float = 0.0
) -> bool:
    """Shared online deployment decision used by inference and offline replay."""
    horizon = int(k_max)
    current = int(step)
    if isinstance(k_max, bool) or horizon <= 0:
        raise ValueError("k_max must be a positive integer")
    if isinstance(step, bool) or current <= 0 or current > horizon:
        raise ValueError("step must lie in [1, k_max]")
    selected_mode = _normalize_mode(mode)
    score = _finite_number(benefit_score, "benefit_score")
    threshold = _finite_number(stop_threshold, "stop_threshold")
    if current >= horizon:
        return True
    if selected_mode == "fixed_k":
        return False
    return score <= threshold


def replay_deployment_policy(
    states: Sequence[Mapping[str, Any]],
    benefit_scores: Sequence[Any] | None = None,
    *,
    mode: str,
    stop_threshold: float = 0.0,
) -> PolicyReplay:
    """Select the state that the deterministic deployment policy would return."""
    if (
        isinstance(states, (str, bytes))
        or not isinstance(states, Sequence)
        or (not states)
    ):
        raise ValueError("states must be a non-empty sequence of mappings")
    if not all((isinstance(state, Mapping) for state in states)):
        raise TypeError("every replay state must be a mapping")
    if benefit_scores is None:
        scores = tuple(
            (
                _finite_number(
                    state.get("benefit_score"), f"states[{index}].benefit_score"
                )
                for index, state in enumerate(states)
            )
        )
    else:
        scores = _numeric_sequence(benefit_scores, "benefit_scores")
    if len(scores) != len(states):
        raise ValueError("states and benefit_scores must have equal length")
    step = deployment_stop_step(
        scores, mode=mode, stop_threshold=stop_threshold, k_max=len(states)
    )
    return PolicyReplay(
        stop_step=step,
        forced_stop=step == len(states),
        selected_state=dict(states[step - 1]),
    )


def compute_utility_curve(
    translation_losses: Sequence[Any],
    *,
    translation_scale: float,
    anatomy_losses: Sequence[Any] | None = None,
    anatomy_scale: float | None = None,
    lambda_anatomy: float = 0.0,
    lambda_compute: float = 0.0,
) -> tuple[float, ...]:
    """Compute ``U_k = L_k/s_L + lambda_A A_k/s_A + lambda_C k``."""
    translation = _numeric_sequence(translation_losses, "translation_losses")
    scale_l = _positive_number(translation_scale, "translation_scale")
    coefficient_a = _nonnegative_number(lambda_anatomy, "lambda_anatomy")
    coefficient_c = _nonnegative_number(lambda_compute, "lambda_compute")
    anatomy: tuple[float, ...] | None = None
    scale_a: float | None = None
    if coefficient_a > 0.0:
        if anatomy_losses is None or anatomy_scale is None:
            raise ValueError(
                "anatomy_losses and anatomy_scale are required when lambda_anatomy > 0"
            )
        anatomy = _numeric_sequence(anatomy_losses, "anatomy_losses")
        if len(anatomy) != len(translation):
            raise ValueError("anatomy_losses must match translation_losses")
        scale_a = _positive_number(anatomy_scale, "anatomy_scale")
    elif anatomy_losses is not None:
        anatomy = _numeric_sequence(anatomy_losses, "anatomy_losses")
        if len(anatomy) != len(translation):
            raise ValueError("anatomy_losses must match translation_losses")
    result: list[float] = []
    for index, loss in enumerate(translation, 1):
        utility = loss / scale_l + coefficient_c * float(index)
        if coefficient_a > 0.0:
            assert anatomy is not None and scale_a is not None
            utility += coefficient_a * anatomy[index - 1] / scale_a
        if not math.isfinite(utility):
            raise ValueError("computed utility is not finite")
        result.append(utility)
    return tuple(result)


def oracle_from_utility(
    utilities: Sequence[Any], *, tolerance: float = 0.0
) -> OracleResult:
    """Return the earliest step within ``tolerance`` of the global minimum."""
    curve = _numeric_sequence(utilities, "utilities")
    epsilon = _nonnegative_number(tolerance, "tolerance")
    minimum = min(curve)
    for index, value in enumerate(curve, 1):
        if value <= minimum + epsilon:
            return OracleResult(step=index, utility=value, minimum_utility=minimum)
    raise AssertionError("a finite utility curve must have an oracle step")


def continuation_benefit_curve(utilities: Sequence[Any]) -> tuple[float, ...]:
    """Return best-future gains ``B*_k=max_{j>k}(U_k-U_j)`` and terminal zero."""
    curve = _numeric_sequence(utilities, "utilities")
    benefits: list[float] = []
    for index, value in enumerate(curve[:-1]):
        benefits.append(max((value - future for future in curve[index + 1 :])))
    benefits.append(0.0)
    return tuple(benefits)


def nearest_rank_quantile(values: Sequence[Any], quantile: float) -> float:
    """Return an observed-value quantile using the deterministic nearest-rank rule.

    For ``0 < q <= 1`` the rank is ``ceil(q*N)``.  ``q=0`` returns the minimum.  This
    is particularly useful for integer refinement steps because p50/p95 remain actual
    observed steps instead of interpolated fractions.
    """
    ordered = sorted(_numeric_sequence(values, "values"))
    q = _finite_number(quantile, "quantile")
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    index = 0 if q == 0.0 else int(math.ceil(q * len(ordered))) - 1
    return ordered[index]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted(((str(key), _freeze(item)) for key, item in value.items())))
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return tuple((_freeze(item) for item in value))
    return value


def deduplicate_case_records(
    records: Iterable[Mapping[str, Any]], *, key: str = "case_id"
) -> tuple[dict[str, Any], ...]:
    """Remove identical distributed-padding duplicates and reject conflicts."""
    selected: dict[str, tuple[Any, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        identifier = str(record.get(key, "")).strip()
        if not identifier:
            raise ValueError(f"records[{index}] lacks a non-empty {key}")
        copied = dict(record)
        frozen = _freeze(copied)
        if identifier in selected:
            if selected[identifier][0] != frozen:
                raise ValueError(
                    f"conflicting duplicate validation record for {key}={identifier!r}"
                )
            continue
        selected[identifier] = (frozen, copied)
    return tuple((selected[identifier][1] for identifier in sorted(selected)))


def merge_rank_case_records(
    rank_records: Iterable[Iterable[Mapping[str, Any]]], *, key: str = "case_id"
) -> tuple[dict[str, Any], ...]:
    """Merge uneven or empty rank-local record lists with exact global deduplication."""
    flattened: list[Mapping[str, Any]] = []
    for rank, shard in enumerate(rank_records):
        if isinstance(shard, (str, bytes)):
            raise TypeError(f"rank_records[{rank}] must be an iterable of mappings")
        try:
            flattened.extend(shard)
        except TypeError as exc:
            raise TypeError(f"rank_records[{rank}] must be iterable") from exc
    return deduplicate_case_records(flattened, key=key)


def _state_metric(state: Mapping[str, Any], name: str, path: str) -> float:
    nested = state.get("metrics")
    value = (
        nested.get(name)
        if isinstance(nested, Mapping) and name in nested
        else state.get(name)
    )
    if str(name).lower() == "psnr":
        result = float(value)
        if result == math.inf:
            return result
    return _finite_number(value, f"{path}.{name}")


def _prepare_case(
    record: Mapping[str, Any],
    *,
    k_max: int,
    metric_names: Sequence[str],
    translation_scale: float,
    anatomy_scale: float | None,
    lambda_anatomy: float,
    lambda_compute: float,
    oracle_tolerance: float,
) -> _PreparedCase:
    case_id = str(record.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("validation record lacks case_id")
    states_raw = record.get("states")
    if (
        isinstance(states_raw, (str, bytes))
        or not isinstance(states_raw, Sequence)
        or len(states_raw) != int(k_max)
    ):
        raise ValueError(f"case {case_id!r} must contain exactly k_max states")
    states: list[Mapping[str, Any]] = []
    scores: list[float] = []
    translation: list[float] = []
    anatomy: list[float] = []
    metrics: list[Mapping[str, float]] = []
    for index, state in enumerate(states_raw, 1):
        if not isinstance(state, Mapping):
            raise TypeError(f"case {case_id!r} state {index} must be a mapping")
        explicit_step = state.get("step")
        if explicit_step is not None and (
            isinstance(explicit_step, bool) or int(explicit_step) != index
        ):
            raise ValueError(
                f"case {case_id!r} state steps must be contiguous and one-indexed"
            )
        path = f"case[{case_id}].states[{index - 1}]"
        state_metrics = {
            str(name): _state_metric(state, str(name), path) for name in metric_names
        }
        translation_value = state.get("translation_loss")
        if translation_value is None:
            if record.get("translation_metric", "latent_mae") != "latent_mae":
                raise ValueError(
                    f"{path} requires explicit latent translation_loss; image MAE cannot substitute"
                )
            translation_value = state_metrics.get("mae")
        translation.append(
            _finite_number(translation_value, f"{path}.translation_loss")
        )
        anatomy_value = state.get("anatomy_loss")
        if anatomy_value is not None:
            anatomy.append(_finite_number(anatomy_value, f"{path}.anatomy_loss"))
        elif float(lambda_anatomy) > 0.0:
            raise ValueError(f"{path}.anatomy_loss is required when lambda_anatomy > 0")
        scores.append(
            _finite_number(state.get("benefit_score"), f"{path}.benefit_score")
        )
        metrics.append(state_metrics)
        states.append(dict(state))
    if anatomy and len(anatomy) != len(states):
        raise ValueError(
            f"case {case_id!r} must provide anatomy_loss for every state or none"
        )
    anatomy_values: tuple[float, ...] | None = tuple(anatomy) if anatomy else None
    utilities = compute_utility_curve(
        translation,
        translation_scale=translation_scale,
        anatomy_losses=anatomy_values,
        anatomy_scale=anatomy_scale,
        lambda_anatomy=lambda_anatomy,
        lambda_compute=lambda_compute,
    )
    oracle = oracle_from_utility(utilities, tolerance=oracle_tolerance)
    difficulty_raw = record.get("difficulty", translation[0])
    difficulty = _finite_number(difficulty_raw, f"case[{case_id}].difficulty")
    return _PreparedCase(
        case_id=case_id,
        states=tuple(states),
        scores=tuple(scores),
        translation_losses=tuple(translation),
        anatomy_losses=anatomy_values,
        metrics=tuple(metrics),
        utilities=utilities,
        continuation_benefits=continuation_benefit_curve(utilities),
        oracle_step=oracle.step,
        oracle_utility=oracle.utility,
        oracle_minimum_utility=oracle.minimum_utility,
        difficulty=difficulty,
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / float(len(values))


def _policy_cases(
    cases: Sequence[_PreparedCase],
    *,
    mode: str,
    stop_threshold: float,
    regret_tolerance: float,
) -> tuple[_PolicyCase, ...]:
    result: list[_PolicyCase] = []
    for case in cases:
        stop_step = deployment_stop_step(
            case.scores,
            mode=mode,
            stop_threshold=stop_threshold,
            k_max=len(case.states),
        )
        regret = case.utilities[stop_step - 1] - case.oracle_utility
        if regret < -regret_tolerance:
            raise ValueError(
                f"case {case.case_id!r} has negative wrong-stop regret beyond tolerance"
            )
        suffix_regret = max(0.0, float(case.continuation_benefits[stop_step - 1]))
        result.append(_PolicyCase(case, stop_step, max(0.0, regret), suffix_regret))
    return tuple(result)


def _step_histogram(policy_cases: Sequence[_PolicyCase], k_max: int) -> dict[str, int]:
    histogram = {str(step): 0 for step in range(1, int(k_max) + 1)}
    for item in policy_cases:
        histogram[str(item.stop_step)] += 1
    return histogram


def _policy_point(
    policy_cases: Sequence[_PolicyCase],
    *,
    metric_names: Sequence[str],
    k_max: int,
    regret_tolerance: float,
) -> dict[str, Any]:
    steps = [float(item.stop_step) for item in policy_cases]
    point: dict[str, Any] = {
        "case_count": len(policy_cases),
        "mean_steps": _mean(steps),
        "p50_steps": None if not steps else int(nearest_rank_quantile(steps, 0.5)),
        "p95_steps": None if not steps else int(nearest_rank_quantile(steps, 0.95)),
        "step_histogram": _step_histogram(policy_cases, k_max),
        "utility": _mean(
            [item.prepared.utilities[item.stop_step - 1] for item in policy_cases]
        ),
        "active_utility_suffix_regret": _mean(
            [item.suffix_regret for item in policy_cases]
        ),
        "active_utility_wrong_stop_rate": _mean(
            [float(item.suffix_regret > regret_tolerance) for item in policy_cases]
        ),
        "oracle_global_regret": _mean(
            [item.oracle_global_regret for item in policy_cases]
        ),
        "oracle_step_mismatch_rate": _mean(
            [
                float(item.stop_step != item.prepared.oracle_step)
                for item in policy_cases
            ]
        ),
        "under_stop_rate": _mean(
            [float(item.stop_step < item.prepared.oracle_step) for item in policy_cases]
        ),
        "over_stop_rate": _mean(
            [float(item.stop_step > item.prepared.oracle_step) for item in policy_cases]
        ),
        "mean_signed_stop_gap": _mean(
            [float(item.stop_step - item.prepared.oracle_step) for item in policy_cases]
        ),
        "mean_abs_stop_gap": _mean(
            [
                float(abs(item.stop_step - item.prepared.oracle_step))
                for item in policy_cases
            ]
        ),
    }
    for metric in metric_names:
        point[str(metric)] = _mean(
            [
                item.prepared.metrics[item.stop_step - 1][str(metric)]
                for item in policy_cases
            ]
        )
    return point


def _stratum_summary(
    selected: Sequence[_PolicyCase], *, metric_names: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(selected),
        "mean_difficulty": _mean([item.prepared.difficulty for item in selected]),
        "mean_steps": _mean([float(item.stop_step) for item in selected]),
        "mean_oracle_steps": _mean(
            [float(item.prepared.oracle_step) for item in selected]
        ),
        "active_utility_suffix_regret": _mean(
            [item.suffix_regret for item in selected]
        ),
        "oracle_global_regret": _mean([item.oracle_global_regret for item in selected]),
    }
    for metric in metric_names:
        result[str(metric)] = _mean(
            [
                item.prepared.metrics[item.stop_step - 1][str(metric)]
                for item in selected
            ]
        )
    return result


def _difficulty_strata(
    policy_cases: Sequence[_PolicyCase], *, fraction: float, metric_names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    proportion = _finite_number(fraction, "easy_hard_fraction")
    if not 0.0 < proportion <= 0.5:
        raise ValueError("easy_hard_fraction must lie in (0, 0.5]")
    ordered = sorted(
        policy_cases, key=lambda item: (item.prepared.difficulty, item.prepared.case_id)
    )
    if not ordered:
        return (
            _stratum_summary((), metric_names=metric_names),
            _stratum_summary((), metric_names=metric_names),
        )
    count = max(1, int(math.ceil(len(ordered) * proportion)))
    count = min(count, len(ordered) // 2)
    if count == 0:
        easy, hard = (ordered, [])
    else:
        easy, hard = (ordered[:count], ordered[-count:])
    return (
        _stratum_summary(easy, metric_names=metric_names),
        _stratum_summary(hard, metric_names=metric_names),
    )


def summarize_validation_records(
    records: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    stop_threshold: float,
    k_max: int,
    translation_scale: float,
    anatomy_scale: float | None = None,
    lambda_anatomy: float = 0.0,
    lambda_compute: float = 0.0,
    oracle_tolerance: float = 1e-08,
    metric_names: Sequence[str] = DEFAULT_METRIC_NAMES,
    easy_hard_fraction: float = 0.25,
    threshold_sweep: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Aggregate full-horizon and deterministic adaptive validation records.

    Means are case-weighted.  The full fixed-K curve and the adaptive point come from
    the same complete trajectories.  Oracle and terminal fixed-K utilities are
    reported separately.  The trainer owns ``val/full_trajectory_objective`` because
    that name denotes its legacy fixed-horizon crop loss diagnostic.  Optional
    threshold sweeps never select or mutate the configured deployment threshold.
    """
    horizon = int(k_max)
    if isinstance(k_max, bool) or horizon <= 0:
        raise ValueError("k_max must be a positive integer")
    selected_mode = _normalize_mode(mode)
    threshold = _finite_number(stop_threshold, "stop_threshold")
    tolerance = _nonnegative_number(oracle_tolerance, "oracle_tolerance")
    names = tuple((str(name).strip() for name in metric_names))
    if not names or any((not name for name in names)) or len(set(names)) != len(names):
        raise ValueError("metric_names must contain unique non-empty names")
    unique = deduplicate_case_records(records)
    if not unique:
        raise ValueError("validation requires at least one unique case record")
    prepared = tuple(
        (
            _prepare_case(
                record,
                k_max=horizon,
                metric_names=names,
                translation_scale=translation_scale,
                anatomy_scale=anatomy_scale,
                lambda_anatomy=lambda_anatomy,
                lambda_compute=lambda_compute,
                oracle_tolerance=tolerance,
            )
            for record in unique
        )
    )
    policy_cases = _policy_cases(
        prepared,
        mode=selected_mode,
        stop_threshold=threshold,
        regret_tolerance=tolerance,
    )
    adaptive = _policy_point(
        policy_cases, metric_names=names, k_max=horizon, regret_tolerance=tolerance
    )
    fixed_curve: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        point: dict[str, Any] = {
            "kind": "fixed_k",
            "steps": step,
            "utility": _mean([case.utilities[step - 1] for case in prepared]),
            "translation_loss": _mean(
                [case.translation_losses[step - 1] for case in prepared]
            ),
            "predicted_benefit": _mean([case.scores[step - 1] for case in prepared]),
            "continuation_benefit": _mean(
                [case.continuation_benefits[step - 1] for case in prepared]
            ),
        }
        if all((case.anatomy_losses is not None for case in prepared)):
            point["anatomy_loss"] = _mean(
                [
                    case.anatomy_losses[step - 1]
                    for case in prepared
                    if case.anatomy_losses is not None
                ]
            )
        for metric in names:
            point[metric] = _mean([case.metrics[step - 1][metric] for case in prepared])
        fixed_curve.append(point)
    oracle_point: dict[str, Any] = {
        "kind": "oracle",
        "mean_steps": _mean([float(case.oracle_step) for case in prepared]),
        "utility": _mean([case.oracle_utility for case in prepared]),
        "minimum_utility": _mean([case.oracle_minimum_utility for case in prepared]),
    }
    for metric in names:
        oracle_point[metric] = _mean(
            [case.metrics[case.oracle_step - 1][metric] for case in prepared]
        )
    adaptive_curve_point = {
        "kind": "adaptive",
        "mean_steps": adaptive["mean_steps"],
        "utility": adaptive["utility"],
        **{metric: adaptive[metric] for metric in names},
    }
    benefit_errors = [
        case.scores[index] - case.continuation_benefits[index]
        for case in prepared
        for index in range(max(0, horizon - 1))
    ]
    decision_total = len(prepared) * max(0, horizon - 1)
    threshold_false_stops = threshold_false_continues = 0
    hidden_positive_stops = optimal_false_continues = 0
    predicted_stop_count = 0
    for case in prepared:
        for index in range(horizon - 1):
            predicted_continue = case.scores[index] > threshold
            calibrated_continue = case.continuation_benefits[index] > threshold
            optimal_continue = case.continuation_benefits[index] > tolerance
            predicted_stop_count += int(not predicted_continue)
            threshold_false_stops += int(not predicted_continue and calibrated_continue)
            threshold_false_continues += int(
                predicted_continue and (not calibrated_continue)
            )
            hidden_positive_stops += int(not predicted_continue and optimal_continue)
            optimal_false_continues += int(
                predicted_continue and (not optimal_continue)
            )
    premature = [
        float(item.prepared.continuation_benefits[item.stop_step - 1] > tolerance)
        for item in policy_cases
    ]
    easy, hard = _difficulty_strata(
        policy_cases, fraction=easy_hard_fraction, metric_names=names
    )
    low_demand = _stratum_summary(
        [item for item in policy_cases if item.prepared.oracle_step == 1],
        metric_names=names,
    )
    high_demand = _stratum_summary(
        [item for item in policy_cases if item.prepared.oracle_step > 1],
        metric_names=names,
    )
    benefit_mse = _mean([value * value for value in benefit_errors])
    summary: dict[str, Any] = {
        "case_count": len(prepared),
        "mode": selected_mode,
        "stop_threshold": threshold,
        "k_max": horizon,
        "val/full/fixed_k_terminal_utility": fixed_curve[-1]["utility"],
        "val/full/benefit_mae": _mean([abs(value) for value in benefit_errors]),
        "val/full/benefit_rmse": None
        if benefit_mse is None
        else math.sqrt(benefit_mse),
        "val/full/benefit_bias": _mean(benefit_errors),
        "val/full/benefit_calibration_scope": "decision_states_only_k_lt_kmax",
        "val/full/threshold_metrics_applicable": selected_mode != "fixed_k",
        "val/full/false_stop_rate": None
        if selected_mode == "fixed_k" or decision_total == 0
        else threshold_false_stops / float(decision_total),
        "val/full/false_continue_rate": None
        if selected_mode == "fixed_k" or decision_total == 0
        else threshold_false_continues / float(decision_total),
        "val/full/head_false_stop_rate": None
        if decision_total == 0
        else threshold_false_stops / float(decision_total),
        "val/full/head_false_continue_rate": None
        if decision_total == 0
        else threshold_false_continues / float(decision_total),
        "val/full/hidden_positive_future_rate": None
        if predicted_stop_count == 0
        else hidden_positive_stops / float(predicted_stop_count),
        "val/full/hidden_positive_future_joint_rate": None
        if decision_total == 0
        else hidden_positive_stops / float(decision_total),
        "val/full/predicted_stop_count": int(predicted_stop_count),
        "val/full/optimal_false_continue_rate": None
        if decision_total == 0
        else optimal_false_continues / float(decision_total),
        "val/full/fixed_k_curve": fixed_curve,
        "val/adaptive": adaptive,
        "val/adaptive/premature_stop_rate": _mean(premature),
        "val/adaptive/easy": easy,
        "val/adaptive/hard": hard,
        "val/adaptive/oracle_step_1": low_demand,
        "val/adaptive/oracle_step_gt_1": high_demand,
        "val/adaptive/easy_hard_fraction": float(easy_hard_fraction),
        "val/adaptive/difficulty_definition": "record.difficulty if present, otherwise state-1 translation_loss; global rank with case_id tie-break",
        "val/quality_step_curve": [*fixed_curve, adaptive_curve_point, oracle_point],
        "val/adaptive_quality": adaptive.get("mae"),
        "val/adaptive_utility": adaptive["utility"],
        "val/adaptive_mean_steps": adaptive["mean_steps"],
        "val/adaptive_p50_steps": adaptive["p50_steps"],
        "val/adaptive_p95_steps": adaptive["p95_steps"],
        "val/under_stop_rate": adaptive["under_stop_rate"],
        "val/over_stop_rate": adaptive["over_stop_rate"],
        "val/active_utility_wrong_stop_rate": adaptive[
            "active_utility_wrong_stop_rate"
        ],
        "val/active_utility_suffix_regret": adaptive["active_utility_suffix_regret"],
        "val/oracle_step_mismatch_rate": adaptive["oracle_step_mismatch_rate"],
        "val/oracle_global_regret": adaptive["oracle_global_regret"],
        "val/predicted_oracle_step_gap": adaptive["mean_signed_stop_gap"],
        "val/predicted_oracle_abs_step_gap": adaptive["mean_abs_stop_gap"],
        "val/oracle_mean_steps": oracle_point["mean_steps"],
        "val/oracle_utility": oracle_point["utility"],
        "val/oracle_minimum_utility": oracle_point["minimum_utility"],
    }
    for metric in names:
        summary[f"val/adaptive_{metric}"] = adaptive[metric]
    for point in fixed_curve:
        step = int(point["steps"])
        for name, value in point.items():
            if name not in {"kind", "steps"}:
                summary[f"val/full/state_{step}/{name}"] = value
    if threshold_sweep is not None:
        thresholds = sorted(
            {
                _finite_number(value, f"threshold_sweep[{index}]")
                for index, value in enumerate(threshold_sweep)
            }
        )
        sweep: list[dict[str, Any]] = []
        for candidate in thresholds:
            candidate_cases = _policy_cases(
                prepared,
                mode=selected_mode,
                stop_threshold=candidate,
                regret_tolerance=tolerance,
            )
            point = _policy_point(
                candidate_cases,
                metric_names=names,
                k_max=horizon,
                regret_tolerance=tolerance,
            )
            sweep.append({"threshold": candidate, **point})
        summary["val/threshold_sweep"] = sweep
    return summary


__all__ = [
    "DEFAULT_METRIC_NAMES",
    "DEPLOYMENT_METHOD_MODES",
    "OracleResult",
    "PolicyReplay",
    "compute_utility_curve",
    "continuation_benefit_curve",
    "deduplicate_case_records",
    "deployment_should_stop",
    "deployment_stop_step",
    "merge_rank_case_records",
    "nearest_rank_quantile",
    "oracle_from_utility",
    "replay_deployment_policy",
    "summarize_validation_records",
]
