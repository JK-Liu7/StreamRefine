"""Full-volume dataset surface used by synchronized sliding inference."""

from __future__ import annotations
from typing import Any, Mapping


class FullVolumePairDataset:
    """Thin read-only wrapper that preserves target fields as supervision-only."""

    def __init__(self, persistent_dataset: Any) -> None:
        self.dataset = persistent_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample["target_is_supervision_only"] = True
        return sample


def build_full_volume_dataset(
    config: Mapping[str, Any], *, split: str = "val", include_images: bool = True
):
    from .persistent_pair_dataset import build_persistent_pair_dataset

    return FullVolumePairDataset(
        build_persistent_pair_dataset(
            config, split=split, include_images=include_images
        )
    )
