"""Rectified-flow training and Euler integration for continuous 3D latents.

This module is a traceable medical-volume adaptation of the flow scheduling surface
in the official Self-Forcing source ``utils/scheduler.py``.  The upstream scheduler
uses a noise-level parameter that decreases from noise to data.  StreamRefine uses
the approved, equivalent increasing-time convention explicitly:

``z_t = (1 - t) * z_0 + t * z_1`` and ``v_target = z_1 - z_0``.
The legacy ``noise`` arguments denote the flow start z_0: Gaussian in the
original mode, or the detached previous refinement state in previous_state mode.

There are deliberately no video-frame, Wan-token, or model-cache assumptions here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch

Tensor = torch.Tensor
VelocityFunction = Callable[[Tensor, Tensor], Tensor]


def _require_sample(sample: Tensor, name: str) -> None:
    if not isinstance(sample, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if sample.ndim < 2:
        raise ValueError(
            f"{name} must have a batch dimension and at least one feature dimension"
        )
    if sample.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one batch item")
    if not sample.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _as_batch_time(
    time: Tensor | float, sample: Tensor, *, name: str = "time"
) -> Tensor:
    """Return a float32 ``[B]`` time tensor on ``sample.device``."""
    if isinstance(time, Tensor):
        value = time.to(device=sample.device, dtype=torch.float32)
    else:
        value = torch.tensor(time, device=sample.device, dtype=torch.float32)
    if value.ndim == 0 or value.numel() == 1:
        value = value.reshape(1).expand(sample.shape[0])
    elif value.ndim != 1 or value.shape[0] != sample.shape[0]:
        raise ValueError(
            f"{name} must be scalar or [B], got shape {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return value.contiguous()


def _broadcast_time(time: Tensor, sample: Tensor) -> Tensor:
    return time.to(dtype=sample.dtype).reshape(time.shape[0], *[1] * (sample.ndim - 1))


def _distributed_broadcast_(value: Tensor, enabled: bool) -> Tensor:
    if not enabled:
        return value
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    torch.distributed.broadcast(value, src=0)
    return value


@dataclass(frozen=True)
class RectifiedFlowTrainingPair:
    """One analytic paired-gradient query for flow matching."""

    noise: Tensor
    flow_time: Tensor
    noisy_state: Tensor
    velocity_target: Tensor


@dataclass(frozen=True)
class EulerSampleResult:
    """Result of an Euler solve from ``t=0`` to ``t=1``."""

    final_state: Tensor
    states: Tensor
    times: Tensor
    num_model_evaluations: int


class RectifiedFlowScheduler:
    """Approved StreamRefine rectified-flow convention and few-step solver."""

    def __init__(self, num_inference_steps: int = 4) -> None:
        if isinstance(num_inference_steps, bool) or int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        self.num_inference_steps = int(num_inference_steps)

    @staticmethod
    def interpolate(noise: Tensor, target: Tensor, flow_time: Tensor | float) -> Tensor:
        """Construct ``z_t=(1-t)z_0+t z_1`` without detaching either endpoint."""
        _require_sample(noise, "noise")
        _require_sample(target, "target")
        if noise.shape != target.shape:
            raise ValueError(
                f"noise and target must have identical shapes, got {tuple(noise.shape)} and {tuple(target.shape)}"
            )
        if noise.device != target.device:
            raise ValueError("noise and target must be on the same device")
        time = _as_batch_time(flow_time, target)
        expanded = _broadcast_time(time, target)
        return (1.0 - expanded) * noise + expanded * target

    @staticmethod
    def velocity_target(noise: Tensor, target: Tensor) -> Tensor:
        """Return the constant rectified-flow target ``z_1-z_0``."""
        _require_sample(noise, "noise")
        _require_sample(target, "target")
        if noise.shape != target.shape:
            raise ValueError("noise and target must have identical shapes")
        return target - noise

    @staticmethod
    def endpoint_from_velocity(
        noisy_state: Tensor, predicted_velocity: Tensor, flow_time: Tensor | float
    ) -> Tensor:
        """Estimate ``z_1`` as ``z_t + (1-t) v_theta``."""
        _require_sample(noisy_state, "noisy_state")
        _require_sample(predicted_velocity, "predicted_velocity")
        if noisy_state.shape != predicted_velocity.shape:
            raise ValueError(
                "noisy_state and predicted_velocity must have identical shapes"
            )
        time = _as_batch_time(flow_time, noisy_state)
        return (
            noisy_state
            + (1.0 - _broadcast_time(time, noisy_state)) * predicted_velocity
        )

    @staticmethod
    def euler_step(
        state: Tensor,
        predicted_velocity: Tensor,
        flow_time: Tensor | float,
        next_flow_time: Tensor | float,
    ) -> Tensor:
        """Advance one explicit Euler step in increasing flow time."""
        _require_sample(state, "state")
        _require_sample(predicted_velocity, "predicted_velocity")
        if state.shape != predicted_velocity.shape:
            raise ValueError("state and predicted_velocity must have identical shapes")
        current = _as_batch_time(flow_time, state, name="flow_time")
        following = _as_batch_time(next_flow_time, state, name="next_flow_time")
        if bool((following < current).any()):
            raise ValueError("next_flow_time must be >= flow_time for every batch item")
        delta = _broadcast_time(following - current, state)
        return state + delta * predicted_velocity

    @staticmethod
    def sample_flow_time(
        batch_size: int,
        *,
        device: torch.device | str,
        generator: Optional[torch.Generator] = None,
        sync_across_ranks: bool = False,
    ) -> Tensor:
        """Sample one continuous flow time per batch item, optionally DDP-synchronized."""
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        rank = 0
        distributed = (
            sync_across_ranks
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if distributed:
            rank = torch.distributed.get_rank()
        if rank == 0:
            sampled = torch.rand(
                int(batch_size), device=device, dtype=torch.float32, generator=generator
            )
        else:
            sampled = torch.empty(int(batch_size), device=device, dtype=torch.float32)
        return _distributed_broadcast_(sampled, distributed)

    def make_training_pair(
        self,
        target: Tensor,
        *,
        noise: Optional[Tensor] = None,
        flow_time: Optional[Tensor | float] = None,
        generator: Optional[torch.Generator] = None,
        sync_flow_time_across_ranks: bool = False,
    ) -> RectifiedFlowTrainingPair:
        """Create the analytic input and target for one differentiable model query."""
        _require_sample(target, "target")
        if noise is None:
            noise = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        _require_sample(noise, "noise")
        if noise.shape != target.shape or noise.device != target.device:
            raise ValueError("noise must match target shape and device")
        if flow_time is None:
            time = self.sample_flow_time(
                target.shape[0],
                device=target.device,
                generator=generator,
                sync_across_ranks=sync_flow_time_across_ranks,
            )
        else:
            time = _as_batch_time(flow_time, target)
            time = _distributed_broadcast_(time, sync_flow_time_across_ranks)
        return RectifiedFlowTrainingPair(
            noise=noise,
            flow_time=time,
            noisy_state=self.interpolate(noise, target, time),
            velocity_target=self.velocity_target(noise, target),
        )

    def time_grid(
        self, *, num_steps: Optional[int] = None, device: torch.device | str = "cpu"
    ) -> Tensor:
        steps = self.num_inference_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)

    def euler_sample(
        self,
        velocity_function: VelocityFunction,
        initial_noise: Tensor,
        *,
        num_steps: Optional[int] = None,
    ) -> EulerSampleResult:
        """Integrate from the supplied flow start at ``t=0`` to ``t=1``."""
        _require_sample(initial_noise, "initial_noise")
        grid = self.time_grid(num_steps=num_steps, device=initial_noise.device)
        state = initial_noise
        states = [state]
        batch_size = state.shape[0]
        for index in range(grid.numel() - 1):
            time = grid[index].expand(batch_size)
            next_time = grid[index + 1].expand(batch_size)
            velocity = velocity_function(state, time)
            if not isinstance(velocity, Tensor):
                raise TypeError("velocity_function must return a torch.Tensor")
            if velocity.shape != state.shape:
                raise ValueError(
                    f"velocity_function output must match the current state shape, got {tuple(velocity.shape)} versus {tuple(state.shape)}"
                )
            state = self.euler_step(state, velocity, time, next_time)
            states.append(state)
        return EulerSampleResult(
            final_state=state,
            states=torch.stack(states, dim=1),
            times=grid,
            num_model_evaluations=grid.numel() - 1,
        )

    sample = euler_sample


__all__ = ["EulerSampleResult", "RectifiedFlowScheduler", "RectifiedFlowTrainingPair"]
