"""Global-coordinate three-axis rotary position embeddings."""

from __future__ import annotations
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs by ninety degrees."""
    if x.shape[-1] % 2:
        raise ValueError(f"rotary feature dimension must be even, got {x.shape[-1]}")
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def rotary_axis_dims(head_dim: int) -> tuple[int, int, int]:
    """Distribute complete feature pairs as evenly as possible over D/H/W."""
    head_dim = int(head_dim)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    num_pairs = head_dim // 2
    base, remainder = divmod(num_pairs, 3)
    pairs = [base + (1 if axis < remainder else 0) for axis in range(3)]
    return tuple((2 * count for count in pairs))


def _canonical_coordinates(
    coordinates: torch.Tensor, *, batch_size: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    if coordinates.ndim == 2 and coordinates.shape == (num_tokens, 3):
        coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
    elif coordinates.ndim == 3 and coordinates.shape[1:] == (num_tokens, 3):
        if coordinates.shape[0] == 1 and batch_size != 1:
            coordinates = coordinates.expand(batch_size, -1, -1)
        elif coordinates.shape[0] != batch_size:
            raise ValueError(
                f"coordinate batch {coordinates.shape[0]} does not match tensor batch {batch_size}"
            )
    else:
        raise ValueError(
            f"coordinates must be [{num_tokens},3] or [{batch_size},{num_tokens},3], got {tuple(coordinates.shape)}"
        )
    return coordinates.to(device=device, dtype=torch.float32)


def rope_cos_sin_3d(
    coordinates: torch.Tensor, head_dim: int, *, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[B,N,head_dim]`` cosine and sine tables for inspection/reuse."""
    head_dim = int(head_dim)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)
        squeeze_batch = True
    elif coordinates.ndim == 3:
        squeeze_batch = False
    else:
        raise ValueError(
            f"coordinates must be [N,3] or [B,N,3], got {tuple(coordinates.shape)}"
        )
    if coordinates.shape[-1] != 3:
        raise ValueError(
            f"coordinates must end in three axes, got {tuple(coordinates.shape)}"
        )
    coordinate_float = coordinates.to(dtype=torch.float32)
    cosine_parts: list[torch.Tensor] = []
    sine_parts: list[torch.Tensor] = []
    for axis, axis_dim in enumerate(rotary_axis_dims(head_dim)):
        if axis_dim == 0:
            continue
        inverse_frequency = 1.0 / float(base) ** (
            torch.arange(0, axis_dim, 2, device=coordinates.device, dtype=torch.float32)
            / axis_dim
        )
        phase = coordinate_float[..., axis : axis + 1] * inverse_frequency
        phase = torch.repeat_interleave(phase, 2, dim=-1)
        cosine_parts.append(phase.cos())
        sine_parts.append(phase.sin())
    cosine = (
        torch.cat(cosine_parts, dim=-1) if cosine_parts else coordinate_float[..., :0]
    )
    sine = torch.cat(sine_parts, dim=-1) if sine_parts else coordinate_float[..., :0]
    if cosine.shape[-1] < head_dim:
        remainder = head_dim - cosine.shape[-1]
        padding_shape = (*cosine.shape[:-1], remainder)
        cosine = torch.cat(
            [
                cosine,
                torch.ones(
                    padding_shape, device=coordinates.device, dtype=torch.float32
                ),
            ],
            dim=-1,
        )
        sine = torch.cat(
            [
                sine,
                torch.zeros(
                    padding_shape, device=coordinates.device, dtype=torch.float32
                ),
            ],
            dim=-1,
        )
    if squeeze_batch:
        return (cosine[0], sine[0])
    return (cosine, sine)


def apply_rope_3d(
    tensor: torch.Tensor, coordinates: torch.Tensor, *, base: float = 10000.0
) -> torch.Tensor:
    """Apply 3D RoPE to attention tensors shaped ``[B,H,N,head_dim]``."""
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    if tensor.ndim != 4:
        raise ValueError(f"tensor must be [B,H,N,D], got {tuple(tensor.shape)}")
    batch, _, num_tokens, head_dim = tensor.shape
    coords = _canonical_coordinates(
        coordinates, batch_size=batch, num_tokens=num_tokens, device=tensor.device
    )
    output_parts: list[torch.Tensor] = []
    offset = 0
    for axis, axis_dim in enumerate(rotary_axis_dims(head_dim)):
        if axis_dim == 0:
            continue
        current = tensor[..., offset : offset + axis_dim]
        inverse_frequency = 1.0 / float(base) ** (
            torch.arange(0, axis_dim, 2, device=tensor.device, dtype=torch.float32)
            / axis_dim
        )
        phase = coords[..., axis : axis + 1] * inverse_frequency
        phase = torch.repeat_interleave(phase, 2, dim=-1).unsqueeze(1)
        cosine = phase.cos().to(dtype=tensor.dtype)
        sine = phase.sin().to(dtype=tensor.dtype)
        output_parts.append(current * cosine + rotate_half(current) * sine)
        offset += axis_dim
    if offset < head_dim:
        output_parts.append(tensor[..., offset:])
    return torch.cat(output_parts, dim=-1)


class RotaryEmbedding3D(nn.Module):
    """Stateless module wrapper around :func:`apply_rope_3d`."""

    def __init__(self, base: float = 10000.0) -> None:
        super().__init__()
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")
        self.base = float(base)

    def forward(self, tensor: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return apply_rope_3d(tensor, coordinates, base=self.base)


__all__ = [
    "RotaryEmbedding3D",
    "apply_rope_3d",
    "rope_cos_sin_3d",
    "rotary_axis_dims",
    "rotate_half",
]
