"""Whole-volume Self-Forcing: detached rollout, paired flow query, and committed history. K1 starts from Gaussian noise; later states start from the last committed state."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING
import torch
from .flow_scheduler import EulerSampleResult, RectifiedFlowScheduler

if TYPE_CHECKING:
    from streamrefine.method.controller import (
        AdaptiveStopController,
        PolicyDecision,
        PolicyState,
    )
Tensor = torch.Tensor


class VolumeVelocityModel(Protocol):
    def __call__(
        self,
        noisy_state: Tensor,
        source_state: Tensor,
        flow_time: Tensor,
        refinement_step: Tensor,
        history_states: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
        target_history_cache: Any = None,
        cache_target: bool = False,
        detach_benefit_input: bool = True,
    ) -> Any: ...


@dataclass(frozen=True)
class RolloutResult:
    state: Tensor
    initial_noise: Tensor
    solver_states: Tensor
    solver_times: Tensor
    num_model_evaluations: int
    source_cache: Any = None
    target_history_cache: Any = None


@dataclass(frozen=True)
class PairedGradientQuery:
    """The only editor-gradient-carrying query for one reached outer state."""

    noisy_state: Tensor
    noise: Tensor
    flow_time: Tensor
    predicted_velocity: Tensor
    target_velocity: Tensor
    endpoint_estimate: Tensor
    history_states: Optional[Tensor]
    reached_mask: Tensor


@dataclass(frozen=True)
class SelfForcingStep:
    refinement_step: int
    rollout: RolloutResult
    paired_query: PairedGradientQuery
    predicted_benefit: Tensor
    reached_mask: Tensor
    policy_decision: Optional["PolicyDecision"] = None
    target_history_cache: Any = None


@dataclass(frozen=True)
class SelfForcingTrajectory:
    """A batch of whole-volume refinement trajectories."""

    states: Tensor
    reached_mask: Tensor
    predicted_benefits: Tensor
    paired_flow_times: Tensor
    steps: tuple[SelfForcingStep, ...]
    final_policy_state: Optional["PolicyState"]
    num_model_evaluations: int
    target_history_cache: Any = None

    @property
    def num_outer_states(self) -> int:
        return int(self.states.shape[1])


def _require_volume(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 5:
        raise ValueError(
            f"{name} must have shape [B,C,D,H,W], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{name} must have non-empty batch and channel dimensions")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _any_active_across_ranks(active: Tensor) -> bool:
    """Use one collective per outer state so every rank executes the same calls."""
    indicator = active.any().to(dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(indicator, op=torch.distributed.ReduceOp.MAX)
    return bool(indicator.item())


def detached_history(
    history_states: Optional[Tensor | Sequence[Tensor]],
    *,
    reference: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """Normalize history to detached ``[B,K,C,D,H,W]`` form."""
    if history_states is None:
        return None
    if isinstance(history_states, Tensor):
        history = history_states
    else:
        values = tuple(history_states)
        if not values:
            return None
        for index, value in enumerate(values):
            _require_volume(value, f"history_states[{index}]")
        history = torch.stack(values, dim=1)
    if history.ndim != 6:
        raise ValueError(
            f"history_states must have shape [B,K,C,D,H,W], got {tuple(history.shape)}"
        )
    if reference is not None:
        _require_volume(reference, "reference")
        expected = (reference.shape[0], reference.shape[1], *reference.shape[2:])
        actual = (history.shape[0], history.shape[2], *history.shape[3:])
        if actual != expected:
            raise ValueError(
                f"history state volumes must match the reference, got {actual} versus {expected}"
            )
        if history.device != reference.device:
            raise ValueError("history_states and reference must share a device")
    return history.detach()


def _extract_tensor(output: Any, names: Sequence[str], what: str) -> Tensor:
    if isinstance(output, Tensor):
        if what == "velocity":
            return output
        raise TypeError(f"model tensor output does not expose {what}")
    if isinstance(output, Mapping):
        for name in names:
            value = output.get(name)
            if isinstance(value, Tensor):
                return value
    for name in names:
        value = getattr(output, name, None)
        if isinstance(value, Tensor):
            return value
    raise TypeError(
        f"model output must expose a tensor {what} through one of {tuple(names)}"
    )


def _extract_velocity(output: Any) -> Tensor:
    return _extract_tensor(
        output,
        ("velocity", "predicted_velocity", "velocity_prediction", "model_output"),
        "velocity",
    )


def _extract_benefit(output: Any) -> Tensor:
    score = _extract_tensor(
        output,
        ("benefit_score", "benefit", "predicted_benefit", "stop_score"),
        "benefit",
    )
    if score.ndim == 2 and score.shape[1] == 1:
        score = score[:, 0]
    if score.ndim != 1:
        raise ValueError(
            f"benefit score must have shape [B] or [B,1], got {tuple(score.shape)}"
        )
    return score


def sample_synchronized_exit_indices(
    num_outer_states: int,
    num_candidates: int,
    *,
    device: torch.device | str,
    generator: Optional[torch.Generator] = None,
    same_across_states: bool = False,
    last_candidate_only: bool = False,
    sync_across_ranks: bool = True,
) -> Tensor:
    """Geometry-free form of the official random-exit sampler.

    StreamRefine's active gradient path samples continuous flow times instead.  This
    helper is kept explicit for provenance tests and any future discrete ablation.
    """
    if int(num_outer_states) <= 0 or int(num_candidates) <= 0:
        raise ValueError("num_outer_states and num_candidates must be positive")
    distributed = (
        sync_across_ranks
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    rank = torch.distributed.get_rank() if distributed else 0
    count = 1 if same_across_states else int(num_outer_states)
    if rank == 0:
        if last_candidate_only:
            indices = torch.full(
                (count,), int(num_candidates) - 1, device=device, dtype=torch.long
            )
        else:
            indices = torch.randint(
                0,
                int(num_candidates),
                (count,),
                device=device,
                dtype=torch.long,
                generator=generator,
            )
    else:
        indices = torch.empty(count, device=device, dtype=torch.long)
    if distributed:
        torch.distributed.broadcast(indices, src=0)
    if same_across_states:
        indices = indices.expand(int(num_outer_states)).clone()
    return indices


class WholeVolumeSelfForcing:
    """Continuous-volume rollout and paired-gradient adapter."""

    def __init__(
        self,
        model: VolumeVelocityModel,
        scheduler: Optional[RectifiedFlowScheduler] = None,
        *,
        inner_steps: int = 4,
        k_max: int = 4,
        refinement_start: str = "previous_state",
        sync_flow_time_across_ranks: bool = True,
    ) -> None:
        if int(inner_steps) <= 0 or int(k_max) <= 0:
            raise ValueError("inner_steps and k_max must be positive")
        if refinement_start != "previous_state":
            raise ValueError("refinement_start must be 'previous_state'")
        self.model = model
        self.scheduler = scheduler or RectifiedFlowScheduler(inner_steps)
        self.inner_steps = int(inner_steps)
        self.k_max = int(k_max)
        self.refinement_start = refinement_start
        self.sync_flow_time_across_ranks = bool(sync_flow_time_across_ranks)

    def _flow_start(
        self, refinement_step: int, history: Optional[Tensor], noise: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Use the last detached committed state for both paths after K1."""
        step = int(refinement_step)
        history_count = 0 if history is None else int(history.shape[1])
        if step <= 0 or history_count != step - 1:
            raise ValueError(
                "previous_state requires exactly refinement_step - 1 committed history states"
            )
        if step == 1:
            return noise
        return history[:, -1].detach()

    @staticmethod
    def _step_tensor(reference: Tensor, refinement_step: int) -> Tensor:
        step = int(refinement_step)
        if step <= 0:
            raise ValueError("refinement_step must be positive")
        return torch.full(
            (reference.shape[0],), step, device=reference.device, dtype=torch.long
        )

    def _model_forward(
        self,
        noisy_state: Tensor,
        source_state: Tensor,
        flow_time: Tensor,
        refinement_step: int,
        *,
        history_states: Optional[Tensor],
        valid_mask: Optional[Tensor],
        global_coordinates: Optional[Tensor],
        source_cache: Any,
        target_history_cache: Any,
        cache_target: bool,
        detach_benefit_input: bool,
    ) -> Any:
        kwargs = dict(
            noisy_state=noisy_state,
            source_state=source_state,
            flow_time=flow_time,
            refinement_step=self._step_tensor(noisy_state, refinement_step),
            history_states=history_states,
            valid_mask=valid_mask,
            global_coordinates=global_coordinates,
            source_cache=source_cache,
            detach_benefit_input=detach_benefit_input,
        )
        if target_history_cache is not None or cache_target:
            kwargs["target_history_cache"] = target_history_cache
            kwargs["cache_target"] = bool(cache_target)
        return self.model(**kwargs)

    def prepare_source_cache(
        self,
        source_state: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
    ) -> Any:
        """Prepare detached static source K/V when the model exposes that API."""
        _require_volume(source_state, "source_state")
        model_with_helpers = getattr(self.model, "module", self.model)
        prepare = getattr(model_with_helpers, "prepare_source_cache", None)
        if not callable(prepare):
            return None
        with torch.no_grad():
            return prepare(
                source_state=source_state,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
            )

    def prepare_target_history_cache(
        self,
        source_state: Tensor,
        history_states: Optional[Tensor | Sequence[Tensor]] = None,
        *,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
    ) -> Any:
        """Prepare detached committed-target K/V when the model supports it."""
        _require_volume(source_state, "source_state")
        history = detached_history(history_states, reference=source_state)
        model_with_helpers = getattr(self.model, "module", self.model)
        prepare = getattr(model_with_helpers, "prepare_target_history_cache", None)
        if not callable(prepare):
            return None
        with torch.no_grad():
            return prepare(
                source_state=source_state,
                history_states=history,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=source_cache,
                max_states=self.k_max,
            )

    def rollout_state(
        self,
        source_state: Tensor,
        *,
        refinement_step: int,
        history_states: Optional[Tensor | Sequence[Tensor]] = None,
        noise: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
        target_history_cache: Any = None,
        generator: Optional[torch.Generator] = None,
    ) -> RolloutResult:
        """Run the complete few-step sampler with no retained editor graph."""
        _require_volume(source_state, "source_state")
        history = detached_history(history_states, reference=source_state)
        noise = self._flow_start(refinement_step, history, noise)
        if noise is None:
            noise = torch.randn(
                source_state.shape,
                device=source_state.device,
                dtype=source_state.dtype,
                generator=generator,
            )
        _require_volume(noise, "noise")
        if noise.shape != source_state.shape or noise.device != source_state.device:
            raise ValueError("rollout noise must match source_state shape and device")
        active_source_cache = source_cache
        if active_source_cache is None:
            active_source_cache = self.prepare_source_cache(
                source_state,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
            )
        active_target_cache = target_history_cache
        target_cache_was_prepared = False
        if active_target_cache is None:
            active_target_cache = self.prepare_target_history_cache(
                source_state,
                history,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=active_source_cache,
            )
            target_cache_was_prepared = active_target_cache is not None
        with torch.no_grad():
            result: EulerSampleResult = self.scheduler.euler_sample(
                lambda current, flow_time: _extract_velocity(
                    self._model_forward(
                        current,
                        source_state,
                        flow_time,
                        refinement_step,
                        history_states=history,
                        valid_mask=valid_mask,
                        global_coordinates=global_coordinates,
                        source_cache=active_source_cache,
                        target_history_cache=active_target_cache,
                        cache_target=False,
                        detach_benefit_input=True,
                    )
                ),
                noise.detach(),
                num_steps=self.inner_steps,
            )
        return RolloutResult(
            state=result.final_state.detach(),
            initial_noise=noise.detach(),
            solver_states=result.states.detach(),
            solver_times=result.times.detach(),
            num_model_evaluations=result.num_model_evaluations
            + (
                0
                if not target_cache_was_prepared or history is None
                else int(history.shape[1])
            ),
            source_cache=active_source_cache,
            target_history_cache=active_target_cache,
        )

    def paired_gradient_query(
        self,
        source_state: Tensor,
        target_state: Tensor,
        *,
        refinement_step: int,
        history_states: Optional[Tensor | Sequence[Tensor]] = None,
        noise: Optional[Tensor] = None,
        flow_time: Optional[Tensor | float] = None,
        reached_mask: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
        target_history_cache: Any = None,
        generator: Optional[torch.Generator] = None,
    ) -> PairedGradientQuery:
        """Execute one differentiable analytic flow query for a reached outer state."""
        _require_volume(source_state, "source_state")
        _require_volume(target_state, "target_state")
        if (
            target_state.shape != source_state.shape
            or target_state.device != source_state.device
        ):
            raise ValueError("target_state must match source_state shape and device")
        if source_cache is not None:
            raise ValueError(
                "paired_gradient_query must recompute source K/V; a detached source_cache would cut editor gradients"
            )
        history = detached_history(history_states, reference=source_state)
        noise = self._flow_start(refinement_step, history, noise)
        active_target_cache = target_history_cache
        if active_target_cache is None:
            active_target_cache = self.prepare_target_history_cache(
                source_state,
                history,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=None,
            )
        pair = self.scheduler.make_training_pair(
            target_state,
            noise=noise,
            flow_time=flow_time,
            generator=generator,
            sync_flow_time_across_ranks=self.sync_flow_time_across_ranks,
        )
        output = self._model_forward(
            pair.noisy_state,
            source_state,
            pair.flow_time,
            refinement_step,
            history_states=history,
            valid_mask=valid_mask,
            global_coordinates=global_coordinates,
            source_cache=source_cache,
            target_history_cache=active_target_cache,
            cache_target=False,
            detach_benefit_input=True,
        )
        velocity = _extract_velocity(output)
        if velocity.shape != target_state.shape:
            raise ValueError("model velocity must match target_state shape")
        endpoint = self.scheduler.endpoint_from_velocity(
            pair.noisy_state, velocity, pair.flow_time
        )
        if reached_mask is None:
            reached = torch.ones(
                target_state.shape[0], device=target_state.device, dtype=torch.bool
            )
        else:
            reached = reached_mask.to(device=target_state.device, dtype=torch.bool)
            if reached.shape != (target_state.shape[0],):
                raise ValueError("reached_mask must have shape [B]")
        return PairedGradientQuery(
            noisy_state=pair.noisy_state,
            noise=pair.noise,
            flow_time=pair.flow_time,
            predicted_velocity=velocity,
            target_velocity=pair.velocity_target.detach(),
            endpoint_estimate=endpoint,
            history_states=history,
            reached_mask=reached.detach(),
        )

    def benefit_query(
        self,
        clean_rollout_state: Tensor,
        source_state: Tensor,
        *,
        refinement_step: int,
        history_states: Optional[Tensor | Sequence[Tensor]] = None,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
        target_history_cache: Any = None,
        cache_target: bool = False,
    ) -> Tensor:
        """Train/query the scalar head while requiring detached backbone features.

        ``VolumeCausalDiT`` must honor ``detach_benefit_input=True`` by detaching the
        valid-mask-pooled hidden representation before its benefit head.  This keeps
        benefit gradients out of the editor while preserving head gradients.
        """
        _require_volume(clean_rollout_state, "clean_rollout_state")
        _require_volume(source_state, "source_state")
        if clean_rollout_state.shape != source_state.shape:
            raise ValueError("clean_rollout_state and source_state must match")
        history = detached_history(history_states, reference=source_state)
        active_target_cache = target_history_cache
        if active_target_cache is None:
            active_target_cache = self.prepare_target_history_cache(
                source_state,
                history,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=source_cache,
            )
        should_cache_target = bool(cache_target and active_target_cache is not None)
        model_with_helpers = getattr(self.model, "module", self.model)
        score_clean_state = getattr(model_with_helpers, "score_clean_state", None)
        if callable(score_clean_state):
            score_kwargs = dict(
                clean_state=clean_rollout_state.detach(),
                source_state=source_state,
                refinement_step=self._step_tensor(clean_rollout_state, refinement_step),
                history_states=history,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=source_cache,
                cache_source=False,
            )
            if active_target_cache is not None or should_cache_target:
                score_kwargs["target_history_cache"] = active_target_cache
                score_kwargs["cache_target"] = should_cache_target
            output = score_clean_state(**score_kwargs)
        else:
            time = torch.ones(
                clean_rollout_state.shape[0],
                device=clean_rollout_state.device,
                dtype=torch.float32,
            )
            output = self._model_forward(
                clean_rollout_state.detach(),
                source_state,
                time,
                refinement_step,
                history_states=history,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=source_cache,
                target_history_cache=active_target_cache,
                cache_target=should_cache_target,
                detach_benefit_input=True,
            )
        score = _extract_benefit(output)
        if score.shape[0] != clean_rollout_state.shape[0]:
            raise ValueError(
                "benefit score batch size does not match the rollout state"
            )
        return score

    def training_step(
        self,
        source_state: Tensor,
        target_state: Tensor,
        *,
        refinement_step: int,
        history_states: Optional[Tensor | Sequence[Tensor]] = None,
        rollout_noise: Optional[Tensor] = None,
        paired_noise: Optional[Tensor] = None,
        paired_flow_time: Optional[Tensor | float] = None,
        reached_mask: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
        target_history_cache: Any = None,
        generator: Optional[torch.Generator] = None,
    ) -> SelfForcingStep:
        """Run the explicit rollout, paired-gradient, and benefit-head paths once."""
        history = detached_history(history_states, reference=source_state)
        rollout = self.rollout_state(
            source_state,
            refinement_step=refinement_step,
            history_states=history,
            noise=rollout_noise,
            valid_mask=valid_mask,
            global_coordinates=global_coordinates,
            source_cache=source_cache,
            target_history_cache=target_history_cache,
            generator=generator,
        )
        query = self.paired_gradient_query(
            source_state,
            target_state,
            refinement_step=refinement_step,
            history_states=history,
            noise=paired_noise,
            flow_time=paired_flow_time,
            reached_mask=reached_mask,
            valid_mask=valid_mask,
            global_coordinates=global_coordinates,
            source_cache=None,
            target_history_cache=rollout.target_history_cache,
            generator=generator,
        )
        reached = query.reached_mask
        if history is None:
            if not bool(reached.all()):
                raise RuntimeError(
                    "the first outer state must be reached by every case"
                )
            committed_for_cache = rollout.state
        else:
            expanded = reached.reshape(source_state.shape[0], 1, 1, 1, 1)
            committed_for_cache = torch.where(
                expanded, rollout.state, history[:, -1]
            ).detach()
        benefit = self.benefit_query(
            committed_for_cache,
            source_state,
            refinement_step=refinement_step,
            history_states=history,
            valid_mask=valid_mask,
            global_coordinates=global_coordinates,
            source_cache=rollout.source_cache,
            target_history_cache=rollout.target_history_cache,
            cache_target=rollout.target_history_cache is not None,
        )
        return SelfForcingStep(
            refinement_step=int(refinement_step),
            rollout=rollout,
            paired_query=query,
            predicted_benefit=benefit,
            reached_mask=query.reached_mask,
            target_history_cache=rollout.target_history_cache,
        )

    @staticmethod
    def _outer_value(
        values: Optional[Tensor | Sequence[Tensor | float]],
        index: int,
        reference: Tensor,
        name: str,
    ) -> Optional[Tensor | float]:
        if values is None:
            return None
        if isinstance(values, Tensor):
            if name == "paired_flow_times":
                if values.ndim == 1:
                    if values.shape[0] <= index:
                        raise ValueError(f"{name} does not contain outer index {index}")
                    return values[index]
                if values.ndim == 2 and values.shape[0] == reference.shape[0]:
                    if values.shape[1] <= index:
                        raise ValueError(f"{name} does not contain outer index {index}")
                    return values[:, index]
                raise ValueError(f"{name} must have shape [K] or [B,K]")
            if (
                values.ndim != reference.ndim + 1
                or values.shape[0] != reference.shape[0]
            ):
                raise ValueError(f"{name} tensor must have shape [B,K,C,D,H,W]")
            if values.shape[1] <= index:
                raise ValueError(f"{name} does not contain outer index {index}")
            selected = values[:, index]
            if selected.shape != reference.shape:
                raise ValueError(f"{name} state shape does not match source_state")
            return selected
        if len(values) <= index:
            raise ValueError(f"{name} does not contain outer index {index}")
        return values[index]

    def run_training_trajectory(
        self,
        source_state: Tensor,
        target_state: Tensor,
        *,
        controller: Optional["AdaptiveStopController"] = None,
        global_step: int = 0,
        rollout_noises: Optional[Tensor | Sequence[Tensor]] = None,
        paired_noises: Optional[Tensor | Sequence[Tensor]] = None,
        paired_flow_times: Optional[Tensor | Sequence[Tensor | float]] = None,
        valid_mask: Optional[Tensor] = None,
        global_coordinates: Optional[Tensor] = None,
        source_cache: Any = None,
        target_history_cache: Any = None,
        generator: Optional[torch.Generator] = None,
    ) -> SelfForcingTrajectory:
        """Run adaptive outer refinement while retaining an explicit reach mask.

        To keep tensor batches dense, model calls are vectorized over the batch while
        any case remains active.  Loss callers must apply ``reached_mask``; stopped
        cases commit no new value and carry their last detached state.  Thus no target
        state is ever inserted as history.
        """
        _require_volume(source_state, "source_state")
        _require_volume(target_state, "target_state")
        if (
            source_state.shape != target_state.shape
            or source_state.device != target_state.device
        ):
            raise ValueError(
                "source_state and target_state must match in shape and device"
            )
        if controller is not None and controller.config.k_max != self.k_max:
            raise ValueError("controller k_max must match WholeVolumeSelfForcing k_max")
        batch_size = source_state.shape[0]
        policy_state = (
            None
            if controller is None
            else controller.initial_state(batch_size, source_state.device)
        )
        active = torch.ones(batch_size, device=source_state.device, dtype=torch.bool)
        active_source_cache = source_cache
        if active_source_cache is None:
            active_source_cache = self.prepare_source_cache(
                source_state,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
            )
        active_target_cache = target_history_cache
        if active_target_cache is None:
            active_target_cache = self.prepare_target_history_cache(
                source_state,
                history_states=None,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=active_source_cache,
            )
        committed_states: list[Tensor] = []
        reached_masks: list[Tensor] = []
        benefit_scores: list[Tensor] = []
        query_times: list[Tensor] = []
        completed_steps: list[SelfForcingStep] = []
        evaluations = 0
        for outer_index in range(self.k_max):
            refinement_step = outer_index + 1
            if not _any_active_across_ranks(active):
                break
            history = detached_history(committed_states, reference=source_state)
            rollout_noise = self._outer_value(
                rollout_noises, outer_index, source_state, "rollout_noises"
            )
            paired_noise = self._outer_value(
                paired_noises, outer_index, source_state, "paired_noises"
            )
            paired_time = self._outer_value(
                paired_flow_times, outer_index, source_state, "paired_flow_times"
            )
            step_result = self.training_step(
                source_state,
                target_state,
                refinement_step=refinement_step,
                history_states=history,
                rollout_noise=rollout_noise
                if isinstance(rollout_noise, Tensor)
                else None,
                paired_noise=paired_noise if isinstance(paired_noise, Tensor) else None,
                paired_flow_time=paired_time,
                reached_mask=active,
                valid_mask=valid_mask,
                global_coordinates=global_coordinates,
                source_cache=active_source_cache,
                target_history_cache=active_target_cache,
                generator=generator,
            )
            active_target_cache = step_result.target_history_cache
            reached = active.detach()
            if committed_states:
                expanded = reached.reshape(batch_size, 1, 1, 1, 1)
                committed = torch.where(
                    expanded, step_result.rollout.state, committed_states[-1]
                ).detach()
            else:
                if not bool(reached.all()):
                    raise RuntimeError(
                        "the first outer state must be reached by every case"
                    )
                committed = step_result.rollout.state.detach()
            score = torch.where(
                reached,
                step_result.predicted_benefit,
                torch.zeros_like(step_result.predicted_benefit),
            )
            decision = None
            if controller is not None:
                assert policy_state is not None
                decision = controller.advance(
                    score.detach(),
                    refinement_step=refinement_step,
                    global_step=int(global_step),
                    state=policy_state,
                    generator=generator,
                )
                policy_state = decision.next_state
                active = decision.continue_next
            else:
                active = reached & (refinement_step < self.k_max)
            finalized_step = SelfForcingStep(
                refinement_step=refinement_step,
                rollout=step_result.rollout,
                paired_query=step_result.paired_query,
                predicted_benefit=score,
                reached_mask=reached,
                policy_decision=decision,
                target_history_cache=active_target_cache,
            )
            committed_states.append(committed)
            reached_masks.append(reached)
            benefit_scores.append(score)
            query_times.append(step_result.paired_query.flow_time)
            completed_steps.append(finalized_step)
            evaluations += step_result.rollout.num_model_evaluations + 2
        if not committed_states:
            raise RuntimeError("trajectory produced no states")
        return SelfForcingTrajectory(
            states=torch.stack(committed_states, dim=1),
            reached_mask=torch.stack(reached_masks, dim=1),
            predicted_benefits=torch.stack(benefit_scores, dim=1),
            paired_flow_times=torch.stack(query_times, dim=1),
            steps=tuple(completed_steps),
            final_policy_state=policy_state,
            num_model_evaluations=evaluations,
            target_history_cache=active_target_cache,
        )


__all__ = [
    "PairedGradientQuery",
    "RolloutResult",
    "SelfForcingStep",
    "SelfForcingTrajectory",
    "VolumeVelocityModel",
    "WholeVolumeSelfForcing",
    "detached_history",
    "sample_synchronized_exit_indices",
]
