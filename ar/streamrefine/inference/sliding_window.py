"""Synchronized continuous sliding-window refinement and tiled decoding."""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


class WindowPredictor(Protocol):
    def __call__(
        self,
        *,
        source_window: Any,
        history_windows: Sequence[Any],
        noise_window: Any,
        valid_mask: Any,
        global_coordinates: Any,
        refinement_step: int,
    ) -> Any: ...


@dataclass
class SlidingRefinementState:
    step: int
    latent: Any
    benefit_score: float
    stopped: bool
    forced_stop: bool
    window_score_variance: float
    seam_mse: float
    sampling_seconds: float
    window_count: int
    model_evaluations: int = 0


@dataclass
class SlidingRefinementResult:
    states: list[SlidingRefinementState] = field(default_factory=list)
    stop_step: int = 0

    @property
    def final_latent(self):
        if not self.states:
            raise RuntimeError("Sliding refinement produced no states")
        return self.states[-1].latent


def sliding_starts(length: int, window: int, overlap: float) -> list[int]:
    length, window = (int(length), int(window))
    overlap = float(overlap)
    if length <= 0 or window <= 0:
        raise ValueError("length and window must be positive")
    if window > length:
        raise ValueError(f"Window {window} is larger than full dimension {length}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    stride = max(1, int(round(window * (1.0 - overlap))))
    values = list(range(0, max(1, length - window + 1), stride))
    final = length - window
    if values[-1] != final:
        values.append(final)
    return values


def window_origins(
    full_shape_dhw: Sequence[int], window_shape_dhw: Sequence[int], overlap: float
) -> list[tuple[int, int, int]]:
    starts = [
        sliding_starts(full, window, overlap)
        for full, window in zip(full_shape_dhw, window_shape_dhw)
    ]
    return [(d, h, w) for d in starts[0] for h in starts[1] for w in starts[2]]


def gaussian_blend_weight(
    shape_dhw: Sequence[int],
    *,
    sigma_scale: float = 0.125,
    eps: float = 0.001,
    device: Any = None,
):
    import torch

    axes = []
    for length in (int(v) for v in shape_dhw):
        if length <= 0:
            raise ValueError("Gaussian shape must be positive")
        coord = torch.linspace(-1.0, 1.0, length, dtype=torch.float32, device=device)
        axes.append(torch.exp(-0.5 * (coord / float(sigma_scale)).square()))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (weight / weight.max().clamp_min(1e-12)).clamp_min(float(eps))


def _crop(value: Any, origin: Sequence[int], shape: Sequence[int], spatial_offset: int):
    d0, h0, w0 = (int(v) for v in origin)
    dd, hh, ww = (int(v) for v in shape)
    prefix = [slice(None)] * spatial_offset
    return value[
        tuple(prefix + [slice(d0, d0 + dd), slice(h0, h0 + hh), slice(w0, w0 + ww)])
    ]


def _parse_prediction(
    value: Any, *, require_score: bool = True
) -> tuple[Any, Any, int]:
    if isinstance(value, Mapping):
        latent = value.get("latent", value.get("state", value.get("sample")))
        score = value.get("benefit_score", value.get("score"))
        evaluations = int(value.get("model_evaluations", 0))
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        latent, score = value[:2]
        evaluations = int(value[2]) if len(value) > 2 else 0
    else:
        raise TypeError(
            "Window predictor must return a mapping or (latent, benefit_score[, evals])"
        )
    if latent is None or (require_score and score is None):
        raise ValueError(
            "Window predictor output lacks latent or required benefit score"
        )
    return (latent, score, evaluations)


def _parse_committed_score(value: Any) -> tuple[Any, int]:
    if isinstance(value, Mapping):
        score = value.get("benefit_score", value.get("score"))
        evaluations = int(value.get("model_evaluations", 0))
    elif isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("Committed-state scorer returned an empty sequence")
        score = value[0]
        evaluations = int(value[1]) if len(value) > 1 else 0
    else:
        score, evaluations = (value, 0)
    if score is None:
        raise ValueError("Committed-state scorer output lacks benefit_score")
    return (score, evaluations)


def _case_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def synchronized_sliding_refinement(
    *,
    source_latent: Any,
    valid_mask_soft: Any,
    global_coordinates: Any,
    predictor: WindowPredictor,
    controller: Callable[[float, int, int], bool] | None,
    k_max: int = 4,
    window_shape_dhw: Sequence[int] = (24, 12, 12),
    overlap: float = 0.5,
    seed: int = 3407,
    case_id: str = "case",
    noise_factory: Callable[[Any, int], Any] | None = None,
) -> SlidingRefinementResult:
    """Refine one full case; only blended global states are committed to history."""
    import torch

    if source_latent.ndim != 4:
        raise ValueError(
            f"source_latent must be [C,D,H,W], got {tuple(source_latent.shape)}"
        )
    full_shape = tuple((int(v) for v in source_latent.shape[-3:]))
    window_shape = tuple((int(v) for v in window_shape_dhw))
    if tuple(valid_mask_soft.shape) != full_shape:
        raise ValueError("valid_mask_soft must match source latent spatial shape")
    if tuple(global_coordinates.shape) != (*full_shape, 3):
        raise ValueError("global_coordinates must have shape [D,H,W,3]")
    if int(k_max) <= 0:
        raise ValueError("k_max must be positive")
    origins = window_origins(full_shape, window_shape, overlap)
    blend = gaussian_blend_weight(window_shape, device=source_latent.device)
    generator = torch.Generator(device=source_latent.device)
    generator.manual_seed(_case_seed(seed, case_id))
    history: list[Any] = []
    result = SlidingRefinementResult()
    committed_scorer = getattr(predictor, "score_committed", None)
    score_after_blend = callable(committed_scorer)
    for step in range(1, int(k_max) + 1):
        start_time = time.perf_counter()
        global_noise = (
            noise_factory(source_latent, step)
            if noise_factory is not None
            else torch.randn(
                source_latent.shape,
                generator=generator,
                device=source_latent.device,
                dtype=source_latent.dtype,
            )
        )
        value_sum = torch.zeros_like(source_latent, dtype=torch.float32)
        value_sq_sum = torch.zeros_like(source_latent, dtype=torch.float32)
        weight_sum = torch.zeros(
            full_shape, dtype=torch.float32, device=source_latent.device
        )
        score_values: list[torch.Tensor] = []
        score_weights: list[torch.Tensor] = []
        evaluations = 0
        evaluated_windows = 0
        evaluated_origins: list[tuple[int, int, int]] = []
        for origin in origins:
            source_window = _crop(source_latent, origin, window_shape, 1)
            history_windows = tuple(
                (_crop(state, origin, window_shape, 1) for state in history)
            )
            noise_window = _crop(global_noise, origin, window_shape, 1)
            mask_window = (
                _crop(valid_mask_soft, origin, window_shape, 0).float().clamp(0.0, 1.0)
            )
            if not bool((mask_window > 0).any()):
                continue
            coords_window = _crop(global_coordinates, origin, window_shape, 0)
            prediction, score, eval_count = _parse_prediction(
                predictor(
                    source_window=source_window,
                    history_windows=history_windows,
                    noise_window=noise_window,
                    valid_mask=mask_window,
                    global_coordinates=coords_window,
                    refinement_step=step,
                ),
                require_score=not score_after_blend,
            )
            prediction = torch.as_tensor(prediction, device=source_latent.device)
            if tuple(prediction.shape) == (1, *tuple(source_window.shape)):
                prediction = prediction[0]
            if tuple(prediction.shape) != tuple(source_window.shape):
                raise ValueError(
                    f"Predictor returned {tuple(prediction.shape)}, expected {tuple(source_window.shape)}"
                )
            if not torch.isfinite(prediction).all():
                raise ValueError("Window predictor returned NaN or Inf")
            local_weight = blend * mask_window
            d0, h0, w0 = origin
            dd, hh, ww = window_shape
            region = (slice(d0, d0 + dd), slice(h0, h0 + hh), slice(w0, w0 + ww))
            weighted = prediction.float() * local_weight.unsqueeze(0)
            value_sum[slice(None), *region].add_(weighted)
            value_sq_sum[slice(None), *region].add_(
                prediction.float().square() * local_weight.unsqueeze(0)
            )
            weight_sum[region].add_(local_weight)
            if not score_after_blend:
                score_tensor = torch.as_tensor(
                    score, dtype=torch.float32, device=source_latent.device
                ).reshape(-1)
                if score_tensor.numel() != 1 or not torch.isfinite(score_tensor).all():
                    raise ValueError(
                        "Each window must return one finite scalar benefit score"
                    )
                score_values.append(score_tensor[0])
                score_weights.append(mask_window.sum().clamp_min(1e-12))
            evaluations += eval_count
            evaluated_windows += 1
            evaluated_origins.append(origin)
        if not evaluated_windows:
            raise ValueError(
                "Full-volume valid_mask_soft contains no valid inference window"
            )
        covered = weight_sum > 0
        denominator = weight_sum.clamp_min(1e-12)
        blended = value_sum / denominator.unsqueeze(0)
        fallback = history[-1] if history else source_latent
        blended = torch.where(covered.unsqueeze(0), blended, fallback.float()).to(
            source_latent.dtype
        )
        second = value_sq_sum / denominator.unsqueeze(0)
        variance = (second - blended.float().square()).clamp_min(0.0)
        seam_mse = (
            float(variance[:, covered].mean().item()) if bool(covered.any()) else 0.0
        )
        if score_after_blend:
            assert callable(committed_scorer)
            for origin in evaluated_origins:
                source_window = _crop(source_latent, origin, window_shape, 1)
                history_windows = tuple(
                    (_crop(state, origin, window_shape, 1) for state in history)
                )
                committed_window = _crop(blended, origin, window_shape, 1)
                mask_window = (
                    _crop(valid_mask_soft, origin, window_shape, 0)
                    .float()
                    .clamp(0.0, 1.0)
                )
                coords_window = _crop(global_coordinates, origin, window_shape, 0)
                score, eval_count = _parse_committed_score(
                    committed_scorer(
                        source_window=source_window,
                        history_windows=history_windows,
                        committed_window=committed_window,
                        valid_mask=mask_window,
                        global_coordinates=coords_window,
                        refinement_step=step,
                    )
                )
                score_tensor = torch.as_tensor(
                    score, dtype=torch.float32, device=source_latent.device
                ).reshape(-1)
                if score_tensor.numel() != 1 or not torch.isfinite(score_tensor).all():
                    raise ValueError(
                        "Each committed window must return one finite scalar benefit score"
                    )
                score_values.append(score_tensor[0])
                score_weights.append(mask_window.sum().clamp_min(1e-12))
                evaluations += eval_count
        if not score_values:
            raise RuntimeError("No benefit scores were produced for valid windows")
        scores = torch.stack(score_values)
        score_weight = torch.stack(score_weights)
        volume_score_tensor = (
            scores * score_weight
        ).sum() / score_weight.sum().clamp_min(1e-12)
        volume_score = float(volume_score_tensor.item())
        score_variance = float(
            ((scores - volume_score_tensor).square() * score_weight)
            .sum()
            .div(score_weight.sum().clamp_min(1e-12))
            .item()
        )
        forced = step == int(k_max)
        stopped = forced or bool(
            controller(volume_score, step, int(k_max)) if controller else False
        )
        committed = blended.detach()
        history.append(committed)
        result.states.append(
            SlidingRefinementState(
                step=step,
                latent=committed,
                benefit_score=volume_score,
                stopped=stopped,
                forced_stop=forced,
                window_score_variance=score_variance,
                seam_mse=seam_mse,
                sampling_seconds=time.perf_counter() - start_time,
                window_count=evaluated_windows,
                model_evaluations=evaluations,
            )
        )
        if stopped:
            result.stop_step = step
            break
    if not result.stop_step:
        result.stop_step = len(result.states)
    return result


def tiled_decode_raw_latent(
    raw_latent: Any,
    *,
    decoder: Any,
    tile_shape_dhw: Sequence[int] = (8, 32, 32),
    overlap: float = 0.25,
    compression_dhw: Sequence[int] = (4, 8, 8),
    batch_size: int = 1,
    precision: str = "bf16",
    channel_mode: str = "average",
):
    """Decode continuous raw latents in overlapping tiles to `[1,D,H,W]`."""
    import torch

    if raw_latent.ndim != 4:
        raise ValueError("raw_latent must be [C,D,H,W]")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    full_shape = tuple((int(v) for v in raw_latent.shape[-3:]))
    tile = tuple(
        (min(int(size), full) for size, full in zip(tile_shape_dhw, full_shape))
    )
    compression = tuple((int(v) for v in compression_dhw))
    origins = window_origins(full_shape, tile, overlap)
    output_shape = tuple(
        (full * factor for full, factor in zip(full_shape, compression))
    )
    tile_output = tuple((size * factor for size, factor in zip(tile, compression)))
    output_sum = torch.zeros(output_shape, dtype=torch.float32)
    weight_sum = torch.zeros(output_shape, dtype=torch.float32)
    image_weight = gaussian_blend_weight(tile_output).cpu()
    for batch_start in range(0, len(origins), int(batch_size)):
        batch_origins = origins[batch_start : batch_start + int(batch_size)]
        tiles = torch.stack(
            [_crop(raw_latent, origin, tile, 1) for origin in batch_origins]
        )
        try:
            decoder_device = next(decoder.parameters()).device
        except (StopIteration, AttributeError):
            decoder_device = raw_latent.device
        decoded = decoder.decode_raw_mean(tiles.to(decoder_device), precision=precision)
        if tuple(decoded.shape[:2]) != (len(batch_origins), 3):
            raise ValueError(
                f"VidTok decoder returned unexpected shape {tuple(decoded.shape)}"
            )
        if tuple(decoded.shape[-3:]) != tile_output:
            raise ValueError(
                f"Decoded tile spatial shape {tuple(decoded.shape[-3:])} != expected {tile_output}"
            )
        gray = (
            decoded.mean(dim=1)
            if channel_mode == "average"
            else decoded[:, decoded.shape[1] // 2]
        )
        gray = gray.detach().float().cpu()
        del decoded, tiles
        for image, origin in zip(gray, batch_origins):
            out_origin = tuple((o * factor for o, factor in zip(origin, compression)))
            region = tuple(
                (
                    slice(start, start + size)
                    for start, size in zip(out_origin, tile_output)
                )
            )
            output_sum[region].add_(image * image_weight)
            weight_sum[region].add_(image_weight)
    if not bool((weight_sum > 0).all()):
        raise RuntimeError("Tiled VidTok decode left uncovered output voxels")
    return (output_sum / weight_sum).unsqueeze(0).contiguous()
