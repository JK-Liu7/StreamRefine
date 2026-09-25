"""Trajectory diagnostics for the released PIR stopping policy."""

from __future__ import annotations
import math
from typing import Any, Iterable, Mapping, Sequence
from streamrefine.validation_policy import (
    compute_utility_curve,
    continuation_benefit_curve,
    deduplicate_case_records,
    deployment_stop_step,
    nearest_rank_quantile,
    oracle_from_utility,
)
from streamrefine.training.pir import PIR_FORMULA, PIR_FORMULA_VERSION

GRADIENT_FORMULA_VERSION = "streamrefine_matched_editor_gradient_v2"
TRAJECTORY_DIAGNOSTICS_VERSION = "streamrefine_trajectory_diagnostics_v2_minimal_pir"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / float(len(values))


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("Pearson inputs must have equal length")
    if len(x_values) < 2:
        return None
    mean_x = math.fsum(x_values) / len(x_values)
    mean_y = math.fsum(y_values) / len(y_values)
    centered_x = [value - mean_x for value in x_values]
    centered_y = [value - mean_y for value in y_values]
    norm_x = math.sqrt(math.fsum((value * value for value in centered_x)))
    norm_y = math.sqrt(math.fsum((value * value for value in centered_y)))
    if norm_x == 0.0 or norm_y == 0.0:
        return None
    result = math.fsum(
        (left * right for left, right in zip(centered_x, centered_y))
    ) / (norm_x * norm_y)
    return max(-1.0, min(1.0, result))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return deterministic one-indexed average ranks, including ties."""
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0 for _ in values]
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = 0.5 * (cursor + 1 + end)
        for position in range(cursor, end):
            result[indexed[position][0]] = average
        cursor = end
    return result


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("Spearman inputs must have equal length")
    if len(x_values) < 2:
        return None
    return _pearson(_average_ranks(x_values), _average_ranks(y_values))


def _histogram(steps: Sequence[int], k_max: int) -> dict[str, int]:
    result = {str(step): 0 for step in range(1, int(k_max) + 1)}
    for step in steps:
        if step not in range(1, int(k_max) + 1):
            raise ValueError("diagnostic step lies outside [1,k_max]")
        result[str(step)] += 1
    return result


def _curve_from_states(
    states: Sequence[Mapping[str, Any]], key: str, *, case_id: str
) -> tuple[float, ...]:
    return tuple(
        (
            _finite(state.get(key), f"case[{case_id}].states[{index}].{key}")
            for index, state in enumerate(states)
        )
    )


def _benefit_summary(
    predicted: Sequence[float], realized: Sequence[float]
) -> dict[str, Any]:
    if len(predicted) != len(realized):
        raise ValueError("predicted and realized benefit arrays must have equal length")
    errors = [left - right for left, right in zip(predicted, realized)]
    return {
        "count": len(errors),
        "pred_mean": _mean(list(predicted)),
        "realized_mean": _mean(list(realized)),
        "mae": _mean([abs(value) for value in errors]),
        "rmse": None
        if not errors
        else math.sqrt(math.fsum((value * value for value in errors)) / len(errors)),
        "bias": _mean(errors),
        "pearson_r": _pearson(list(predicted), list(realized)),
    }


def summarize_trajectory_diagnostics(
    records: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    stop_threshold: float,
    k_max: int,
    translation_scale: float,
    anatomy_scale: float | None,
    lambda_anatomy: float,
    lambda_compute: float,
    oracle_tolerance: float = 1e-08,
    anatomy_metric: str = "pir",
    translation_metric: str = "latent_mae",
    pir_utility_key: str | None = None,
    pir_formula: str | None = None,
    anatomy_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Summarize the complete PIR and translation trajectory."""
    horizon = int(k_max)
    from streamrefine.training.translation import canonical_translation_metric

    translation_metric = canonical_translation_metric(translation_metric)
    if isinstance(k_max, bool) or horizon <= 0:
        raise ValueError("k_max must be a positive integer")
    threshold = _finite(stop_threshold, "stop_threshold")
    tolerance = _finite(oracle_tolerance, "oracle_tolerance")
    if tolerance < 0.0:
        raise ValueError("oracle_tolerance must be non-negative")
    coefficient_anatomy = _finite(lambda_anatomy, "lambda_anatomy")
    coefficient_compute = _finite(lambda_compute, "lambda_compute")
    if coefficient_anatomy < 0.0 or coefficient_compute < 0.0:
        raise ValueError("diagnostic utility coefficients must be non-negative")
    anatomy_reason = str(anatomy_unavailable_reason or "").strip() or None
    metric = str(anatomy_metric).strip().lower()
    if metric not in {"pir"}:
        raise ValueError("anatomy_metric must be 'pir'")
    if anatomy_reason is not None and coefficient_anatomy > 0.0:
        raise ValueError(
            "anatomy_unavailable_reason cannot accompany a positive lambda_anatomy"
        )
    if pir_utility_key is not None:
        pir_utility_key = str(pir_utility_key).strip()
        if not pir_utility_key:
            raise ValueError("pir_utility_key must be non-empty when supplied")
        if not str(pir_formula or "").strip():
            raise ValueError(
                "pir_formula is required with pir_utility_key; PIR cannot be inferred from anatomy_loss"
            )
    unique = deduplicate_case_records(records)
    if not unique:
        raise ValueError("trajectory diagnostics require at least one case")
    per_case: list[dict[str, Any]] = []
    decision_predicted: list[float] = []
    decision_realized: list[float] = []
    per_step_predicted: list[list[float]] = [[] for _ in range(horizon - 1)]
    per_step_realized: list[list[float]] = [[] for _ in range(horizon - 1)]
    terminal_predictions: list[float] = []
    anatomy_presence: list[bool] = []
    for record in unique:
        case_id = str(record.get("case_id", "")).strip()
        states_raw = record.get("states")
        if (
            not case_id
            or isinstance(states_raw, (str, bytes))
            or (not isinstance(states_raw, Sequence))
            or (len(states_raw) != horizon)
            or (not all((isinstance(state, Mapping) for state in states_raw)))
        ):
            raise ValueError(
                f"case {case_id!r} must contain exactly k_max state mappings"
            )
        states = tuple((dict(state) for state in states_raw))
        scores = _curve_from_states(states, "benefit_score", case_id=case_id)
        translation = _curve_from_states(states, "translation_loss", case_id=case_id)
        anatomy_values: tuple[float, ...] | None = None
        if all((state.get("anatomy_loss") is not None for state in states)):
            anatomy_values = _curve_from_states(states, "anatomy_loss", case_id=case_id)
        elif any((state.get("anatomy_loss") is not None for state in states)):
            raise ValueError(
                f"case {case_id!r} must expose anatomy_loss for every state or none"
            )
        anatomy_presence.append(anatomy_values is not None)
        image_utility = compute_utility_curve(
            translation,
            translation_scale=translation_scale,
            lambda_compute=coefficient_compute,
        )
        image_oracle = oracle_from_utility(image_utility, tolerance=tolerance)
        anatomy_utility: tuple[float, ...] | None = None
        anatomy_oracle = None
        if anatomy_values is not None and coefficient_anatomy > 0.0:
            if anatomy_scale is None:
                raise ValueError(
                    "anatomy_scale is required for the anatomy-aware diagnostic oracle"
                )
            anatomy_utility = compute_utility_curve(
                translation,
                translation_scale=translation_scale,
                anatomy_losses=anatomy_values,
                anatomy_scale=anatomy_scale,
                lambda_anatomy=coefficient_anatomy,
                lambda_compute=coefficient_compute,
            )
            anatomy_oracle = oracle_from_utility(anatomy_utility, tolerance=tolerance)
        active_utility = (
            anatomy_utility
            if str(mode).strip().lower() == "anatomy_aware"
            else image_utility
        )
        if active_utility is None:
            raise ValueError(
                "anatomy_aware diagnostics require a complete anatomy curve"
            )
        active_benefit = continuation_benefit_curve(active_utility)
        regret_utility = anatomy_utility or active_utility
        regret_benefit = continuation_benefit_curve(regret_utility)
        stop_step = deployment_stop_step(
            scores, mode=mode, stop_threshold=threshold, k_max=horizon
        )
        suffix_regret = max(0.0, regret_benefit[stop_step - 1])
        wrong_stop_available = anatomy_reason is None
        active_oracle = oracle_from_utility(active_utility, tolerance=tolerance)
        global_regret = max(0.0, active_utility[stop_step - 1] - active_oracle.utility)
        pir_oracle = anatomy_oracle if metric == "pir" else None
        if pir_utility_key is not None:
            pir_utility = _curve_from_states(states, pir_utility_key, case_id=case_id)
            pir_oracle = oracle_from_utility(pir_utility, tolerance=tolerance)
        for index in range(horizon - 1):
            decision_predicted.append(scores[index])
            decision_realized.append(active_benefit[index])
            per_step_predicted[index].append(scores[index])
            per_step_realized[index].append(active_benefit[index])
        terminal_predictions.append(scores[-1])
        difficulty = _finite(
            record.get("difficulty", translation[0]), f"case[{case_id}].difficulty"
        )
        per_case.append(
            {
                "case_id": case_id,
                "stop_k": stop_step,
                "wrong_stop": None
                if not wrong_stop_available
                else suffix_regret > tolerance,
                "wrong_stop_suffix_regret": None
                if not wrong_stop_available
                else suffix_regret,
                "wrong_stop_basis": None
                if not wrong_stop_available
                else metric
                if anatomy_utility is not None
                else "active_utility",
                "oracle_global_regret": global_regret,
                "oracle_image_k": image_oracle.step,
                "oracle_anatomy_k": None
                if anatomy_oracle is None
                else anatomy_oracle.step,
                "oracle_pir_k": None if pir_oracle is None else pir_oracle.step,
                "anatomy_metric": metric,
                "difficulty": difficulty,
                "state_1_realized_benefit": active_benefit[0],
            }
        )
    if anatomy_presence and any(anatomy_presence) and (not all(anatomy_presence)):
        raise ValueError(
            "anatomy curves must be available for every diagnostic case or none"
        )
    if coefficient_anatomy > 0.0 and (not all(anatomy_presence)):
        raise ValueError(
            "positive lambda_anatomy requires an anatomy curve for every case"
        )
    steps = [int(case["stop_k"]) for case in per_case]
    image_steps = [int(case["oracle_image_k"]) for case in per_case]
    anatomy_available = all((case["oracle_anatomy_k"] is not None for case in per_case))
    anatomy_steps = (
        [int(case["oracle_anatomy_k"]) for case in per_case]
        if anatomy_available
        else []
    )
    pir_available = all((case["oracle_pir_k"] is not None for case in per_case))
    pir_steps = (
        [int(case["oracle_pir_k"]) for case in per_case] if pir_available else []
    )
    comparator_name = (
        ("pir" if metric == "pir" else "anatomy") if anatomy_available else None
    )
    comparator_steps = anatomy_steps
    disagreement = (
        None
        if comparator_name is None
        else _mean(
            [float(left != right) for left, right in zip(image_steps, comparator_steps)]
        )
    )
    signed_oracle_gaps = [
        float(right - left) for left, right in zip(image_steps, comparator_steps)
    ]
    pir_signed_oracle_gaps = [
        float(right - left) for left, right in zip(image_steps, pir_steps)
    ]
    benefit = _benefit_summary(decision_predicted, decision_realized)
    benefit_by_step = {
        str(index + 1): _benefit_summary(predicted, realized)
        for index, (predicted, realized) in enumerate(
            zip(per_step_predicted, per_step_realized)
        )
    }
    wrong_stop_available = all((case["wrong_stop"] is not None for case in per_case))
    wrong_regrets = (
        [float(case["wrong_stop_suffix_regret"]) for case in per_case]
        if wrong_stop_available
        else []
    )
    wrong_flags = (
        [float(bool(case["wrong_stop"])) for case in per_case]
        if wrong_stop_available
        else []
    )
    ordered = sorted(per_case, key=lambda case: (case["difficulty"], case["case_id"]))
    stratum_count = min(len(ordered) // 2, max(1, int(math.ceil(len(ordered) * 0.25))))
    easy = ordered[:stratum_count] if stratum_count else ordered
    hard = ordered[-stratum_count:] if stratum_count else []
    easy_steps = [float(case["stop_k"]) for case in easy]
    hard_steps = [float(case["stop_k"]) for case in hard]
    easy_future = [float(case["state_1_realized_benefit"]) for case in easy]
    hard_future = [float(case["state_1_realized_benefit"]) for case in hard]
    result: dict[str, Any] = {
        "diagnostics/schema": TRAJECTORY_DIAGNOSTICS_VERSION,
        "anatomy/metric": metric,
        "wrong_stop/rate": _mean(wrong_flags),
        "wrong_stop/regret": _mean(wrong_regrets),
        "wrong_stop/regret_on_wrong": _mean(
            [value for value in wrong_regrets if value > tolerance]
        ),
        "wrong_stop/definition": "positive_best_suffix_benefit_at_deployment_stop",
        "wrong_stop/available": wrong_stop_available,
        "wrong_stop/unavailable_reason": None
        if wrong_stop_available
        else anatomy_reason,
        "benefit/pred_vs_realized": benefit["pearson_r"],
        "benefit/pred_mean": benefit["pred_mean"],
        "benefit/realized_mean": benefit["realized_mean"],
        "benefit/calibration_mae": benefit["mae"],
        "benefit/calibration_rmse": benefit["rmse"],
        "benefit/calibration_bias": benefit["bias"],
        "benefit/decision_count": benefit["count"],
        "benefit/by_step": benefit_by_step,
        "benefit/terminal_pred_mean": _mean(terminal_predictions),
        "benefit/calibration_scope": "decision_states_only_k_lt_kmax",
        "oracle/image_k": _mean([float(value) for value in image_steps]),
        "oracle/image_k_histogram": _histogram(image_steps, horizon),
        "oracle/anatomy_k": None
        if not anatomy_available
        else _mean([float(value) for value in anatomy_steps]),
        "oracle/anatomy_k_histogram": None
        if not anatomy_available
        else _histogram(anatomy_steps, horizon),
        "oracle/anatomy_available": anatomy_available,
        "oracle/anatomy_unavailable_reason": None
        if anatomy_available
        else anatomy_reason
        or (
            "anatomy_curve_unavailable"
            if not any(anatomy_presence)
            else "positive_lambda_anatomy_and_scale_required"
        ),
        "oracle/pir_k": None
        if not pir_available
        else _mean([float(value) for value in pir_steps]),
        "oracle/pir_k_histogram": None
        if not pir_available
        else _histogram(pir_steps, horizon),
        "oracle/pir_available": pir_available,
        "oracle/pir_unavailable_reason": None
        if pir_available
        else anatomy_reason
        if metric == "pir" and anatomy_reason is not None
        else "pir_anatomy_curve_unavailable"
        if metric == "pir"
        else "anatomy_unavailable",
        "oracle/disagreement": disagreement,
        "oracle/disagreement_basis": None
        if comparator_name is None
        else f"image_vs_{comparator_name}",
        "oracle/disagreement_mean_signed_gap": _mean(signed_oracle_gaps),
        "oracle/disagreement_mean_abs_gap": _mean(
            [abs(value) for value in signed_oracle_gaps]
        ),
        "oracle/disagreement_deeper_rate": _mean(
            [float(value > 0.0) for value in signed_oracle_gaps]
        ),
        "oracle/disagreement_shallower_rate": _mean(
            [float(value < 0.0) for value in signed_oracle_gaps]
        ),
        "oracle/image_objective": "L_k/translation_scale + lambda_compute*k",
        "oracle/anatomy_objective": f"L_k/translation_scale + lambda_anatomy*A_k^{metric}/anatomy_scale + lambda_compute*k",
        "oracle/pir_formula": None
        if not pir_available
        else str(pir_formula)
        if pir_utility_key is not None
        else f"{PIR_FORMULA}; U_k=L_k/s_L+lambda_anatomy*A_k^PIR/s_A+lambda_compute*k",
        "oracle/pir_formula_version": PIR_FORMULA_VERSION
        if pir_available and metric == "pir"
        else None,
        "oracle/pir_disagreement": None
        if not pir_available
        else _mean(
            [float(left != right) for left, right in zip(image_steps, pir_steps)]
        ),
        "oracle/pir_disagreement_mean_signed_gap": _mean(pir_signed_oracle_gaps),
        "oracle/pir_disagreement_mean_abs_gap": _mean(
            [abs(value) for value in pir_signed_oracle_gaps]
        ),
        "step/mean": _mean([float(value) for value in steps]),
        "step/p50": int(nearest_rank_quantile(steps, 0.5)),
        "step/p95": int(nearest_rank_quantile(steps, 0.95)),
        "step/histogram": _histogram(steps, horizon),
        "step/difficulty_spearman": _spearman(
            [float(case["difficulty"]) for case in per_case],
            [float(case["stop_k"]) for case in per_case],
        ),
        "step/difficulty_definition": "record.difficulty if present, otherwise state_1_translation_loss; bottom/top quartile with case_id tie-break",
        "step/easy_count": len(easy),
        "step/hard_count": len(hard),
        "step/easy_mean": _mean(easy_steps),
        "step/hard_mean": _mean(hard_steps),
        "step/hard_minus_easy": None
        if not easy_steps or not hard_steps
        else float(_mean(hard_steps) - _mean(easy_steps)),
        "future_utility/hard_minus_easy": None
        if not easy_future or not hard_future
        else float(_mean(hard_future) - _mean(easy_future)),
        "future_utility/easy_mean": _mean(easy_future),
        "future_utility/hard_mean": _mean(hard_future),
        "future_utility/basis": "state_1_active_continuation_benefit",
        "diagnostics/per_case": per_case,
    }
    result["diagnostics/translation_metric"] = translation_metric
    if translation_metric == "latent_mae":
        if result["oracle/disagreement_basis"] is not None:
            result["oracle/disagreement_basis"] = result[
                "oracle/disagreement_basis"
            ].replace("image_vs_", "latent_vs_")
        for suffix in ("k", "k_histogram", "objective"):
            result[f"oracle/latent_{suffix}"] = result.pop(f"oracle/image_{suffix}")
        for case in per_case:
            case["oracle_latent_k"] = case.pop("oracle_image_k")
    return result


__all__ = ["TRAJECTORY_DIAGNOSTICS_VERSION", "summarize_trajectory_diagnostics"]
