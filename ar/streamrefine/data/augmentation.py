"""Uncached aligned crop and grid augmentation for StreamRefine."""

from __future__ import annotations
import hashlib
import multiprocessing
from typing import Any, Mapping, Sequence


def _crop(
    value: Any, origin: Sequence[int], shape: Sequence[int], *, spatial_offset: int
):
    d0, h0, w0 = (int(v) for v in origin)
    dd, hh, ww = (int(v) for v in shape)
    prefix = [slice(None)] * spatial_offset
    return value[
        tuple(prefix + [slice(d0, d0 + dd), slice(h0, h0 + hh), slice(w0, w0 + ww)])
    ]


def _flip(value: Any, axis: int, *, spatial_offset: int):
    return value.flip(spatial_offset + axis)


def _rotate_hw(value: Any, k: int, *, spatial_offset: int):
    return value.rot90(k, dims=(spatial_offset + 1, spatial_offset + 2))


def apply_synchronised_grid_transform(
    sample: Mapping[str, Any],
    *,
    flips_dhw: Sequence[bool] = (False, False, False),
    rotate90_hw_k: int = 0,
) -> dict[str, Any]:
    """Move every spatial tensor together; coordinate vector values remain global."""
    result = dict(sample)
    channel_fields = (
        "source_latent_raw",
        "target_latent_raw",
        "source_latent_model",
        "target_latent_model",
        "source_latent",
        "target_latent",
        "preprocessed_source_image",
        "preprocessed_gt_image",
    )
    mask_fields = (
        "valid_mask_latent_soft",
        "valid_mask_latent",
        "valid_mask_latent_hard",
        "anatomy_mask_latent_hard",
        "pir_mask_latent_hard",
    )
    for axis, enabled in enumerate(flips_dhw):
        if not enabled:
            continue
        for field in channel_fields:
            if field in result:
                result[field] = _flip(
                    result[field], axis, spatial_offset=1
                ).contiguous()
        for field in mask_fields:
            if field in result:
                result[field] = _flip(
                    result[field], axis, spatial_offset=0
                ).contiguous()
        if "global_coords_dhw" in result:
            result["global_coords_dhw"] = _flip(
                result["global_coords_dhw"], axis, spatial_offset=0
            ).contiguous()
    k = int(rotate90_hw_k) % 4
    if k:
        for field in channel_fields:
            if field in result:
                result[field] = _rotate_hw(
                    result[field], k, spatial_offset=1
                ).contiguous()
        for field in mask_fields:
            if field in result:
                result[field] = _rotate_hw(
                    result[field], k, spatial_offset=0
                ).contiguous()
        if "global_coords_dhw" in result:
            result["global_coords_dhw"] = _rotate_hw(
                result["global_coords_dhw"], k, spatial_offset=0
            ).contiguous()
    return result


class RandomAlignedCropAugmentDataset:
    """Random wrapper deliberately placed outside MONAI PersistentDataset."""

    def __init__(
        self,
        dataset: Any,
        *,
        latent_crop_dhw: Sequence[int] = (24, 12, 12),
        image_crop_dhw: Sequence[int] = (96, 96, 96),
        min_valid_ratio: float = 0.85,
        max_tries: int = 100,
        augmentation: Mapping[str, Any] | None = None,
        seed: int = 3407,
        training: bool = True,
        require_pir_edges: bool = False,
    ) -> None:
        self.dataset = dataset
        self.latent_crop_dhw = tuple((int(v) for v in latent_crop_dhw))
        self.image_crop_dhw = tuple((int(v) for v in image_crop_dhw))
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_tries = int(max_tries)
        self.augmentation = dict(augmentation or {})
        self.seed = int(seed)
        self.training = bool(training)
        self.require_pir_edges = bool(require_pir_edges)
        self.epoch = 0
        self._shared_epoch = multiprocessing.Value("q", 0, lock=False)
        if any((v <= 0 for v in (*self.latent_crop_dhw, *self.image_crop_dhw))):
            raise ValueError("Crop sizes must be positive")
        ratios = tuple(
            (i / l for i, l in zip(self.image_crop_dhw, self.latent_crop_dhw))
        )
        if any((not value.is_integer() for value in ratios)):
            raise ValueError(
                "image_crop_dhw must be an integer multiple of latent_crop_dhw"
            )
        self.compression_dhw = tuple((int(value) for value in ratios))
        rotate_probability = float(
            self.augmentation.get("rotate90_hw_probability", 0.2)
        )
        if (
            self.training
            and bool(self.augmentation.get("enabled", True))
            and (rotate_probability > 0)
        ):
            if self.latent_crop_dhw[1] != self.latent_crop_dhw[2]:
                raise ValueError(
                    "90-degree H/W rotation requires a square latent H/W crop"
                )
            if self.image_crop_dhw[1] != self.image_crop_dhw[2]:
                raise ValueError(
                    "90-degree H/W rotation requires a square image H/W crop"
                )

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._shared_epoch.value = int(epoch)

    def _rng(self, index: int):
        import torch

        epoch = int(self._shared_epoch.value)
        payload = f"{self.seed}:{epoch}:{int(index)}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
            2**63 - 1
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from streamrefine.data.latent_window_sampler import (
            find_best_pir_edge_window,
            find_best_valid_window,
            mask_has_minimal_pir_edge,
            sample_valid_latent_window,
        )

        full = dict(self.dataset[index])
        generator = self._rng(index)
        if self.training:
            origin, ratio = sample_valid_latent_window(
                full["valid_mask_latent_hard"],
                window=self.latent_crop_dhw,
                min_valid_ratio=self.min_valid_ratio,
                max_tries=self.max_tries,
                generator=generator,
                required_pir_mask=full["pir_mask_latent_hard"]
                if self.require_pir_edges
                else None,
            )
        elif self.require_pir_edges:
            origin, ratio, _ = find_best_pir_edge_window(
                full["valid_mask_latent_hard"],
                full["pir_mask_latent_hard"],
                self.latent_crop_dhw,
                min_valid_ratio=self.min_valid_ratio,
            )
            d0, h0, w0 = origin
            wd, wh, ww = self.latent_crop_dhw
            selected_pir = full["pir_mask_latent_hard"][
                d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww
            ]
            if not mask_has_minimal_pir_edge(selected_pir):
                raise ValueError("Validation crop contains no valid anatomy PIR edge")
        else:
            origin, ratio = find_best_valid_window(
                full["valid_mask_latent_hard"], self.latent_crop_dhw
            )
        image_origin = tuple((o * c for o, c in zip(origin, self.compression_dhw)))
        crop: dict[str, Any] = dict(full)
        for field in (
            "source_latent_raw",
            "target_latent_raw",
            "source_latent_model",
            "target_latent_model",
            "source_latent",
            "target_latent",
        ):
            crop[field] = _crop(
                full[field], origin, self.latent_crop_dhw, spatial_offset=1
            ).contiguous()
        for field in (
            "valid_mask_latent_soft",
            "valid_mask_latent",
            "valid_mask_latent_hard",
            "anatomy_mask_latent_hard",
            "pir_mask_latent_hard",
        ):
            crop[field] = _crop(
                full[field], origin, self.latent_crop_dhw, spatial_offset=0
            ).contiguous()
        if not torch.equal(
            crop["pir_mask_latent_hard"],
            crop["valid_mask_latent_hard"] & crop["anatomy_mask_latent_hard"],
        ):
            raise RuntimeError("Cropped PIR mask is not valid & anatomy")
        crop["global_coords_dhw"] = _crop(
            full["global_coords_dhw"], origin, self.latent_crop_dhw, spatial_offset=0
        ).contiguous()
        for field in ("preprocessed_source_image", "preprocessed_gt_image"):
            if field in full:
                crop[field] = _crop(
                    full[field], image_origin, self.image_crop_dhw, spatial_offset=1
                ).contiguous()
        crop["crop_origin_dhw"] = torch.as_tensor(origin, dtype=torch.long)
        crop["crop_shape_dhw"] = torch.as_tensor(self.latent_crop_dhw, dtype=torch.long)
        crop["valid_ratio"] = torch.tensor(float(ratio), dtype=torch.float32)
        crop["valid_window_met_threshold"] = bool(ratio >= self.min_valid_ratio)
        if self.training and bool(self.augmentation.get("enabled", True)):
            probs = [
                float(self.augmentation.get("flip_probability_d", 0.5)),
                float(self.augmentation.get("flip_probability_h", 0.5)),
                float(self.augmentation.get("flip_probability_w", 0.5)),
            ]
            flips = [
                bool(torch.rand((), generator=generator).item() < probability)
                for probability in probs
            ]
            rotate = 0
            if torch.rand((), generator=generator).item() < float(
                self.augmentation.get("rotate90_hw_probability", 0.2)
            ):
                rotate = int(torch.randint(1, 4, (), generator=generator).item())
            crop = apply_synchronised_grid_transform(
                crop, flips_dhw=flips, rotate90_hw_k=rotate
            )
            if not torch.equal(
                crop["pir_mask_latent_hard"],
                crop["valid_mask_latent_hard"] & crop["anatomy_mask_latent_hard"],
            ):
                raise RuntimeError("Augmented PIR mask is not valid & anatomy")
            crop["augmentation_flips_dhw"] = torch.as_tensor(flips, dtype=torch.bool)
            crop["augmentation_rotate90_hw_k"] = rotate
        return crop


def training_requires_images(config):
    """The released training objective consumes standardized and raw latents."""
    return False


def build_training_dataset(config: Mapping[str, Any], *, split: str = "train"):
    from .persistent_pair_dataset import build_persistent_pair_dataset

    persistent = build_persistent_pair_dataset(
        config, split=split, include_images=training_requires_images(config)
    )
    return RandomAlignedCropAugmentDataset(
        persistent,
        latent_crop_dhw=config["latent"]["crop_size_dhw"],
        image_crop_dhw=config["latent"]["image_crop_size_dhw"],
        min_valid_ratio=float(config["latent"].get("min_valid_ratio", 0.85)),
        max_tries=int(config["latent"].get("max_crop_tries", 100)),
        augmentation=config.get("augmentation", {}),
        seed=int(config["train"]["seed"]),
        training=split == "train",
        require_pir_edges=str(config["benefit"].get("anatomy_metric", "pir")) == "pir"
        and str(config["method"].get("mode", "")) == "anatomy_aware",
    )
