"""Continuous 3D patchification and global-coordinate helpers.

The model-facing convention is ``[B, C, D, H, W]``.  Patches are flattened in
raster ``D -> H -> W`` order and each patch keeps channel-first voxel order.
This module deliberately contains no assumptions about a particular latent window.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import torch


def as_3tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    """Return a validated positive ``(D, H, W)`` tuple."""
    if isinstance(value, int):
        result = (int(value),) * 3
    else:
        result = tuple((int(item) for item in value))
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {result}")
    if any((item <= 0 for item in result)):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


@dataclass(frozen=True)
class PatchGrid3D:
    """Geometry shared by patchification, masks, and coordinate reduction."""

    spatial_shape_dhw: tuple[int, int, int]
    patch_size_dhw: tuple[int, int, int]

    def __init__(
        self,
        spatial_shape_dhw: Sequence[int],
        patch_size_dhw: int | Sequence[int] = (2, 1, 1),
    ) -> None:
        spatial_shape = as_3tuple(spatial_shape_dhw, "spatial_shape_dhw")
        patch_size = as_3tuple(patch_size_dhw, "patch_size_dhw")
        if any((size % patch for size, patch in zip(spatial_shape, patch_size))):
            raise ValueError(
                f"spatial shape {spatial_shape} must be divisible by patch size {patch_size}"
            )
        object.__setattr__(self, "spatial_shape_dhw", spatial_shape)
        object.__setattr__(self, "patch_size_dhw", patch_size)

    @property
    def grid_shape_dhw(self) -> tuple[int, int, int]:
        return tuple(
            (
                size // patch
                for size, patch in zip(self.spatial_shape_dhw, self.patch_size_dhw)
            )
        )

    @property
    def patch_volume(self) -> int:
        pd, ph, pw = self.patch_size_dhw
        return pd * ph * pw

    @property
    def num_patches(self) -> int:
        gd, gh, gw = self.grid_shape_dhw
        return gd * gh * gw


def patchify_3d(
    volume: torch.Tensor, patch_size_dhw: int | Sequence[int] = (2, 1, 1)
) -> torch.Tensor:
    """Convert ``[B,C,D,H,W]`` to ``[B,N,C*pd*ph*pw]`` exactly."""
    if volume.ndim != 5:
        raise ValueError(f"volume must be [B,C,D,H,W], got {tuple(volume.shape)}")
    batch, channels, depth, height, width = volume.shape
    geometry = PatchGrid3D((depth, height, width), patch_size_dhw)
    pd, ph, pw = geometry.patch_size_dhw
    gd, gh, gw = geometry.grid_shape_dhw
    patches = volume.reshape(batch, channels, gd, pd, gh, ph, gw, pw)
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return patches.reshape(
        batch, geometry.num_patches, channels * geometry.patch_volume
    )


def unpatchify_3d(
    patches: torch.Tensor,
    spatial_shape_dhw: Sequence[int],
    patch_size_dhw: int | Sequence[int] = (2, 1, 1),
    *,
    channels: int | None = None,
) -> torch.Tensor:
    """Invert :func:`patchify_3d` for a divisible spatial shape."""
    if patches.ndim != 3:
        raise ValueError(f"patches must be [B,N,P], got {tuple(patches.shape)}")
    geometry = PatchGrid3D(spatial_shape_dhw, patch_size_dhw)
    batch, num_patches, patch_dim = patches.shape
    if num_patches != geometry.num_patches:
        raise ValueError(
            f"expected {geometry.num_patches} patches for {geometry.spatial_shape_dhw}, got {num_patches}"
        )
    if channels is None:
        if patch_dim % geometry.patch_volume:
            raise ValueError(
                f"patch dimension {patch_dim} is not divisible by patch volume {geometry.patch_volume}"
            )
        channels = patch_dim // geometry.patch_volume
    channels = int(channels)
    if channels <= 0 or patch_dim != channels * geometry.patch_volume:
        raise ValueError(
            f"patch dimension {patch_dim} does not match channels={channels} and patch size {geometry.patch_size_dhw}"
        )
    pd, ph, pw = geometry.patch_size_dhw
    gd, gh, gw = geometry.grid_shape_dhw
    volume = patches.reshape(batch, gd, gh, gw, channels, pd, ph, pw)
    volume = volume.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    depth, height, width = geometry.spatial_shape_dhw
    return volume.reshape(batch, channels, depth, height, width)


def dense_coordinate_grid(
    spatial_shape_dhw: Sequence[int],
    *,
    origin_dhw: Sequence[float] | torch.Tensor = (0.0, 0.0, 0.0),
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build dense global latent coordinates in ``[...,D,H,W,3]`` form.

    A scalar origin produces ``[D,H,W,3]``.  Batched origins ``[B,3]`` produce
    ``[B,D,H,W,3]``.  Keeping this field explicit lets synchronized flips and
    rotations transform coordinates with source, target, and valid masks.
    """
    depth, height, width = as_3tuple(spatial_shape_dhw, "spatial_shape_dhw")
    origin = torch.as_tensor(origin_dhw, device=device, dtype=dtype)
    if origin.ndim == 1:
        if origin.shape != (3,):
            raise ValueError(
                f"origin_dhw must be [3] or [B,3], got {tuple(origin.shape)}"
            )
        batched = False
        origin = origin.unsqueeze(0)
    elif origin.ndim == 2 and origin.shape[-1] == 3:
        batched = True
    else:
        raise ValueError(f"origin_dhw must be [3] or [B,3], got {tuple(origin.shape)}")
    d = torch.arange(depth, device=origin.device, dtype=dtype)
    h = torch.arange(height, device=origin.device, dtype=dtype)
    w = torch.arange(width, device=origin.device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(d, h, w, indexing="ij"), dim=-1)
    grid = grid.unsqueeze(0) + origin[:, None, None, None, :]
    return grid if batched else grid[0]


def patchify_dense_coordinates(
    coordinates: torch.Tensor, patch_size_dhw: int | Sequence[int] = (2, 1, 1)
) -> torch.Tensor:
    """Reduce a transformed dense coordinate field to patch-center coordinates.

    Coordinates may be ``[D,H,W,3]`` or ``[B,D,H,W,3]``.  Averaging every
    patch is correct for ordinary grids and remains well defined after the same
    discrete spatial transform is applied to all paired tensors.
    """
    if coordinates.ndim == 4:
        if coordinates.shape[-1] != 3:
            raise ValueError(
                f"coordinates must end in three axes, got {tuple(coordinates.shape)}"
            )
        batched = False
        coordinates = coordinates.unsqueeze(0)
    elif coordinates.ndim == 5 and coordinates.shape[-1] == 3:
        batched = True
    else:
        raise ValueError(
            f"coordinates must be [D,H,W,3] or [B,D,H,W,3], got {tuple(coordinates.shape)}"
        )
    batch, depth, height, width, axes = coordinates.shape
    geometry = PatchGrid3D((depth, height, width), patch_size_dhw)
    pd, ph, pw = geometry.patch_size_dhw
    gd, gh, gw = geometry.grid_shape_dhw
    centers = coordinates.reshape(batch, gd, pd, gh, ph, gw, pw, axes)
    centers = centers.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    centers = centers.reshape(batch, geometry.num_patches, geometry.patch_volume, axes)
    centers = centers.to(dtype=torch.float32).mean(dim=2)
    return centers if batched else centers[0]


def patchify_valid_mask(
    valid_mask: torch.Tensor, patch_size_dhw: int | Sequence[int] = (2, 1, 1)
) -> torch.Tensor:
    """Return the valid fraction of each patch as ``[B,N]`` (or ``[N]``)."""
    if valid_mask.ndim == 3:
        batched = False
        valid_mask = valid_mask.unsqueeze(0)
    elif valid_mask.ndim == 4:
        batched = True
    elif valid_mask.ndim == 5 and valid_mask.shape[1] == 1:
        batched = True
        valid_mask = valid_mask[:, 0]
    else:
        raise ValueError(
            f"valid_mask must be [D,H,W], [B,D,H,W], or [B,1,D,H,W], got {tuple(valid_mask.shape)}"
        )
    batch, depth, height, width = valid_mask.shape
    geometry = PatchGrid3D((depth, height, width), patch_size_dhw)
    pd, ph, pw = geometry.patch_size_dhw
    gd, gh, gw = geometry.grid_shape_dhw
    fractions = valid_mask.reshape(batch, gd, pd, gh, ph, gw, pw)
    fractions = fractions.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    fractions = fractions.reshape(batch, geometry.num_patches, geometry.patch_volume)
    fractions = fractions.to(dtype=torch.float32).mean(dim=-1)
    return fractions if batched else fractions[0]


patchify = patchify_3d
unpatchify = unpatchify_3d
__all__ = [
    "PatchGrid3D",
    "as_3tuple",
    "dense_coordinate_grid",
    "patchify",
    "patchify_3d",
    "patchify_dense_coordinates",
    "patchify_valid_mask",
    "unpatchify",
    "unpatchify_3d",
]
