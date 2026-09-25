from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from torch.utils.data import Dataset

_BBOX_KEYS = ("h0", "h1", "w0", "w1", "d0", "d1")


def _validated_bbox(
    volume_chwd: Any, patch_meta: Mapping[str, Any]
) -> tuple[int, int, int, int, int, int]:
    if getattr(volume_chwd, "ndim", None) != 4:
        raise ValueError(
            f"Expected volume [C,H,W,D], got {getattr(volume_chwd, 'shape', None)!r}"
        )
    missing = [key for key in _BBOX_KEYS if key not in patch_meta]
    if missing:
        raise KeyError(f"Patch metadata is missing bbox fields: {missing}")
    h0, h1, w0, w1, d0, d1 = (int(patch_meta[key]) for key in _BBOX_KEYS)
    _, height, width, depth = (int(item) for item in volume_chwd.shape)
    if not (
        0 <= h0 < h1 <= height and 0 <= w0 < w1 <= width and (0 <= d0 < d1 <= depth)
    ):
        raise ValueError(
            f"Patch bbox {(h0, h1, w0, w1, d0, d1)} is outside volume shape {(height, width, depth)}."
        )
    return (h0, h1, w0, w1, d0, d1)


def crop_chwd(volume_chwd: Any, patch_meta: Mapping[str, Any]):
    """Crop a ``[C,H,W,D]`` tensor using an explicit HWD patch record."""
    h0, h1, w0, w1, d0, d1 = _validated_bbox(volume_chwd, patch_meta)
    return volume_chwd[:, h0:h1, w0:w1, d0:d1].contiguous()


class ModalityPatchDataset(Dataset):
    """In-memory patch dataset for batched offline VidTok encoding.

    The full preprocessed/padded volume is held once. Every item is cut from the
    same explicit, case-shared patch table; this class performs no file I/O,
    normalization, padding, or tokenizer calls.
    """

    def __init__(
        self, volume_chwd: Any, patch_table: Sequence[Mapping[str, Any]]
    ) -> None:
        if getattr(volume_chwd, "ndim", None) != 4:
            raise ValueError(
                f"Expected volume [C,H,W,D], got {getattr(volume_chwd, 'shape', None)!r}"
            )
        if int(volume_chwd.shape[0]) != 1:
            raise ValueError(
                f"Expected one grayscale channel, got shape {tuple(volume_chwd.shape)}"
            )
        if not patch_table:
            raise ValueError("patch_table must not be empty")
        self.volume_chwd = volume_chwd
        self.patch_table = [dict(row) for row in patch_table]
        seen: set[int] = set()
        for index, row in enumerate(self.patch_table):
            patch_id = int(row.get("patch_id", index))
            if patch_id in seen:
                raise ValueError(f"Duplicate patch_id={patch_id}")
            seen.add(patch_id)
            row["patch_id"] = patch_id
            _validated_bbox(self.volume_chwd, row)

    def __len__(self) -> int:
        return len(self.patch_table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patch_meta = dict(self.patch_table[index])
        return {
            "patch_id": int(patch_meta["patch_id"]),
            "crop": crop_chwd(self.volume_chwd, patch_meta),
            "patch_meta": patch_meta,
        }
