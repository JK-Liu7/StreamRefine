"""Case-level stopping, sentinel continuation, and reach correction."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional
import torch

Tensor = torch.Tensor


class MethodMode(str, Enum):
    FIXED_K = "fixed_k"
    ANATOMY_AWARE = "anatomy_aware"

    @property
    def uses_sentinel(self) -> bool:
        return self in {MethodMode.ANATOMY_AWARE}

    @property
    def corrects_editor_reach(self) -> bool:
        return self in {MethodMode.ANATOMY_AWARE}


def normalize_method_mode(value: MethodMode | str) -> MethodMode:
    if isinstance(value, MethodMode):
        return value
    try:
        return MethodMode(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join((item.value for item in MethodMode))
        raise ValueError(
            f"unknown StreamRefine method mode {value!r}; expected one of {choices}"
        ) from exc


def _validate_probability(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("sentinel_probability must lie in (0, 1]")
    return result


@dataclass(frozen=True)
class ControllerConfig:
    mode: MethodMode | str = MethodMode.FIXED_K
    k_max: int = 4
    fixed_steps: Optional[int] = None
    sentinel_probability: float = 0.15
    stop_threshold: float = 0.0
    policy_warmup_steps: int = 10000

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_method_mode(self.mode))
        if isinstance(self.k_max, bool) or int(self.k_max) <= 0:
            raise ValueError("k_max must be a positive integer")
        object.__setattr__(self, "k_max", int(self.k_max))
        fixed = self.k_max if self.fixed_steps is None else int(self.fixed_steps)
        if fixed <= 0 or fixed > self.k_max:
            raise ValueError("fixed_steps must lie in [1, k_max]")
        object.__setattr__(self, "fixed_steps", fixed)
        probability = _validate_probability(self.sentinel_probability)
        object.__setattr__(self, "sentinel_probability", probability)
        if not math.isfinite(float(self.stop_threshold)):
            raise ValueError("stop_threshold must be finite")
        if (
            isinstance(self.policy_warmup_steps, bool)
            or int(self.policy_warmup_steps) < 0
        ):
            raise ValueError("policy_warmup_steps must be a non-negative integer")
        object.__setattr__(self, "policy_warmup_steps", int(self.policy_warmup_steps))


@dataclass(frozen=True)
class PolicyState:
    """Vectorized trajectory-level policy state for a batch of whole volumes."""

    active: Tensor
    first_stop_step: Tensor
    sentinel_selected: Tensor
    sentinel_drawn: Tensor

    def detached(self) -> "PolicyState":
        return PolicyState(
            active=self.active.detach(),
            first_stop_step=self.first_stop_step.detach(),
            sentinel_selected=self.sentinel_selected.detach(),
            sentinel_drawn=self.sentinel_drawn.detach(),
        )


@dataclass(frozen=True)
class PolicyDecision:
    refinement_step: int
    reached: Tensor
    proposed_stop: Tensor
    forced_stop: Tensor
    sentinel_sampled: Tensor
    sentinel_selected_now: Tensor
    continue_next: Tensor
    next_state: PolicyState
    policy_warmup_active: bool


class AdaptiveStopController:
    """Apply one case-level stopping decision after each blended/refined state."""

    def __init__(
        self, config: ControllerConfig | None = None, **config_kwargs: object
    ) -> None:
        if config is not None and config_kwargs:
            raise ValueError(
                "pass either ControllerConfig or keyword configuration, not both"
            )
        self.config = (
            config if config is not None else ControllerConfig(**config_kwargs)
        )

    @property
    def mode(self) -> MethodMode:
        return self.config.mode

    def is_policy_warmup(self, global_step: int) -> bool:
        if int(global_step) < 0:
            raise ValueError("global_step must be non-negative")
        return (
            self.mode is not MethodMode.FIXED_K
            and int(global_step) < self.config.policy_warmup_steps
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> PolicyState:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        size = int(batch_size)
        return PolicyState(
            active=torch.ones(size, device=device, dtype=torch.bool),
            first_stop_step=torch.full((size,), -1, device=device, dtype=torch.long),
            sentinel_selected=torch.zeros(size, device=device, dtype=torch.bool),
            sentinel_drawn=torch.zeros(size, device=device, dtype=torch.bool),
        )

    def advance(
        self,
        predicted_benefit: Tensor,
        *,
        refinement_step: int,
        global_step: int,
        state: PolicyState,
        generator: Optional[torch.Generator] = None,
        sentinel_override: Optional[Tensor | bool] = None,
    ) -> PolicyDecision:
        """Update the trajectory policy after observing state ``k``.

        A single Bernoulli sentinel is sampled at the first learned stop proposal.
        Selected sentinel trajectories ignore subsequent head decisions and reveal the
        entire suffix.  Every mode stops unconditionally at its configured horizon.
        """
        step = int(refinement_step)
        if step <= 0 or step > self.config.k_max:
            raise ValueError("refinement_step must lie in [1, k_max]")
        score = predicted_benefit
        if score.ndim == 2 and score.shape[1] == 1:
            score = score[:, 0]
        if score.ndim != 1 or score.shape != state.active.shape:
            raise ValueError(
                "predicted_benefit must have shape [B] matching PolicyState"
            )
        if not bool(torch.isfinite(score).all()):
            raise ValueError("predicted_benefit must be finite")
        for field in (
            state.first_stop_step,
            state.sentinel_selected,
            state.sentinel_drawn,
        ):
            if field.shape != state.active.shape or field.device != state.active.device:
                raise ValueError("PolicyState fields must share shape and device")
        if score.device != state.active.device:
            raise ValueError("predicted_benefit and PolicyState must share a device")
        reached = state.active
        warmup = self.is_policy_warmup(global_step)
        if self.mode is MethodMode.FIXED_K:
            horizon = int(self.config.fixed_steps)
            forced_stop = reached & (step >= horizon)
            learned_stop = torch.zeros_like(reached)
        elif warmup:
            forced_stop = reached & (step >= self.config.k_max)
            learned_stop = torch.zeros_like(reached)
        else:
            forced_stop = reached & (step >= self.config.k_max)
            eligible = reached & ~state.sentinel_selected & ~forced_stop
            learned_stop = eligible & (score <= float(self.config.stop_threshold))
        proposed_stop = forced_stop | learned_stop
        new_first_stop = proposed_stop & (state.first_stop_step < 0)
        sentinel_sampled = (
            learned_stop
            & new_first_stop
            & self.mode.uses_sentinel
            & ~state.sentinel_drawn
        )
        selected_now = torch.zeros_like(reached)
        if bool(sentinel_sampled.any()):
            if sentinel_override is None:
                draws = (
                    torch.rand(
                        reached.shape,
                        device=reached.device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    < self.config.sentinel_probability
                )
            elif isinstance(sentinel_override, Tensor):
                override = sentinel_override.to(device=reached.device, dtype=torch.bool)
                if override.ndim == 0 or override.numel() == 1:
                    draws = override.reshape(1).expand_as(reached)
                elif override.shape == reached.shape:
                    draws = override
                else:
                    raise ValueError("sentinel_override must be scalar or [B]")
            else:
                draws = torch.full_like(reached, bool(sentinel_override))
            selected_now = sentinel_sampled & draws
        selected = state.sentinel_selected | selected_now
        drawn = state.sentinel_drawn | sentinel_sampled
        first_stop = torch.where(
            new_first_stop,
            torch.full_like(state.first_stop_step, step),
            state.first_stop_step,
        )
        stop_now = forced_stop | learned_stop & ~selected_now
        continue_next = reached & ~stop_now
        next_state = PolicyState(
            active=continue_next,
            first_stop_step=first_stop,
            sentinel_selected=selected,
            sentinel_drawn=drawn,
        ).detached()
        return PolicyDecision(
            refinement_step=step,
            reached=reached.detach(),
            proposed_stop=proposed_stop.detach(),
            forced_stop=forced_stop.detach(),
            sentinel_sampled=sentinel_sampled.detach(),
            sentinel_selected_now=selected_now.detach(),
            continue_next=continue_next.detach(),
            next_state=next_state,
            policy_warmup_active=warmup,
        )


def _broadcast_bool(value: Tensor, reference: Tensor, name: str) -> Tensor:
    result = value.to(device=reference.device, dtype=torch.bool)
    if result.ndim + 1 == reference.ndim and result.shape == reference.shape[:-1]:
        result = result.unsqueeze(-1)
    try:
        return torch.broadcast_to(result, reference.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} is not broadcastable to reached") from exc


def compute_editor_reach_weights(
    mode: MethodMode | str,
    *,
    reached: Tensor,
    after_proposed_stop: Tensor,
    sentinel_selected: Tensor,
    sentinel_probability: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return per-state editor weights, including ``1/epsilon`` after a stop."""
    selected_mode = normalize_method_mode(mode)
    if not isinstance(reached, Tensor):
        raise TypeError("reached must be a torch.Tensor")
    reached_bool = reached.to(dtype=torch.bool)
    after = _broadcast_bool(after_proposed_stop, reached_bool, "after_proposed_stop")
    sentinel = _broadcast_bool(sentinel_selected, reached_bool, "sentinel_selected")
    epsilon = _validate_probability(sentinel_probability)
    inconsistent = reached_bool & after & ~sentinel
    if selected_mode.uses_sentinel and bool(inconsistent.any()):
        raise ValueError(
            "a reached post-stop state must belong to a selected sentinel trajectory"
        )
    weights = reached_bool.to(dtype=dtype)
    if selected_mode.corrects_editor_reach:
        weights = torch.where(
            reached_bool & after, torch.full_like(weights, 1.0 / epsilon), weights
        )
    return weights


def compute_suffix_stop_weights(
    mode: MethodMode | str,
    *,
    first_stop_step: Tensor,
    sentinel_selected: Tensor,
    k_max: int,
    sentinel_probability: float,
    supervised: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return trajectory weights for targets that require the complete suffix.

    For an early proposal, a non-sentinel trajectory receives zero because its suffix
    is unobserved.  A sentinel trajectory receives ``S/epsilon``.  A trajectory that
    naturally reaches Kmax receives unit weight.
    """
    selected_mode = normalize_method_mode(mode)
    if not isinstance(first_stop_step, Tensor) or first_stop_step.ndim != 1:
        raise ValueError("first_stop_step must have shape [B]")
    if int(k_max) <= 0:
        raise ValueError("k_max must be positive")
    sentinel = _broadcast_bool(sentinel_selected, first_stop_step, "sentinel_selected")
    if supervised is None:
        eligible = torch.ones_like(first_stop_step, dtype=torch.bool)
    else:
        eligible = _broadcast_bool(supervised, first_stop_step, "supervised")
    epsilon = _validate_probability(sentinel_probability)
    weights = eligible.to(dtype=dtype)
    if selected_mode.uses_sentinel:
        early = (first_stop_step > 0) & (first_stop_step < int(k_max))
        weights = torch.where(
            early & sentinel & eligible,
            torch.full_like(weights, 1.0 / epsilon),
            weights,
        )
        weights = torch.where(early & ~sentinel, torch.zeros_like(weights), weights)
    return weights


def state_reach_probabilities(
    mode: MethodMode | str,
    *,
    first_stop_step: Tensor,
    k_max: int,
    sentinel_probability: float,
) -> Tensor:
    """Theoretical ``[B,Kmax]`` reach probability for logging and audits."""
    selected_mode = normalize_method_mode(mode)
    if not isinstance(first_stop_step, Tensor) or first_stop_step.ndim != 1:
        raise ValueError("first_stop_step must have shape [B]")
    if int(k_max) <= 0:
        raise ValueError("k_max must be positive")
    epsilon = _validate_probability(sentinel_probability)
    steps = torch.arange(1, int(k_max) + 1, device=first_stop_step.device).unsqueeze(0)
    stop = first_stop_step.unsqueeze(1)
    after = (stop > 0) & (stop < int(k_max)) & (steps > stop)
    probability = torch.ones(
        (first_stop_step.shape[0], int(k_max)),
        device=first_stop_step.device,
        dtype=torch.float32,
    )
    if selected_mode.uses_sentinel:
        probability = torch.where(
            after, torch.full_like(probability, epsilon), probability
        )
    return probability


__all__ = [
    "AdaptiveStopController",
    "ControllerConfig",
    "MethodMode",
    "PolicyDecision",
    "PolicyState",
    "compute_editor_reach_weights",
    "compute_suffix_stop_weights",
    "normalize_method_mode",
    "state_reach_probabilities",
]
