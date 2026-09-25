"""Content-guarded corrections for visually audited BraTS header origins.

An origin-only affine difference does NOT establish voxel alignment. This is a
small allowlist, not a generic registration or resampling fallback. The audited
raw files remain untouched; only a cloned in-memory MetaTensor affine changes.
"""

from __future__ import annotations
import hashlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

POLICY = "verified_origin_only_v1"
VERIFIED_CASES = {
    "BraTS-MET-00232-000": {
        "reference_modality": "t1n",
        "corrected_modality": "t2w",
        "shape_hwd": [240, 240, 155],
        "reference_affine": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 239.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "original_affine": [
            [1.0, 0.0, 0.0, -120.0],
            [0.0, 1.0, 0.0, -129.0],
            [0.0, 0.0, 1.0, -68.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "voxel_sha256": {
            "t1n": "633efcee8d4509f46d1e4b9230d8dfc2e8ca34a1ea5137c8db04ee4bc4a5de54",
            "t1c": "b1667ace6ed8f8931847f8041bd1200a3ce9c1bfd3acb40032f26e5b63c9f325",
            "t2w": "a4dfa2889deebcc377ee2c388a94bee03f8cd2663452b64980ac98a7b4646a5e",
            "t2f": "a152030885e968cacf62f5b33f6df455e0b7980b55b9f8f2396aa9d0956466e0",
        },
        "evidence": "ar/diagnostics/BraTS-MET-00232-000-origin-audit; three planes, three slices per plane",
    }
}


def oriented_voxel_sha256(value: Any) -> str:
    """SHA256 of finite oriented HWD values as C-order little-endian float32.

    Ignore gzip encoding, paths and header serialization so copied/recompressed
    files are accepted, but altered intensities/voxel order are not.
    """
    import numpy as np
    import torch

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Verified BraTS correction requires [1,H,W,D] image data")
    array = np.asarray(tensor[0], dtype="<f4", order="C")
    if not np.isfinite(array).all():
        raise ValueError("Verified BraTS correction refuses nonfinite image values")
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


class ReconcileVerifiedBraTSAffined:
    """Correct only the exact audited case, header pattern and voxel contents."""

    def __init__(
        self,
        image_keys: Sequence[str],
        modality_by_key: Mapping[str, str],
        policy: str = POLICY,
    ) -> None:
        if policy not in {POLICY, "strict"}:
            raise ValueError(
                f"Unsupported brats_affine_policy={policy!r}; use {POLICY!r} or 'strict'"
            )
        self.image_keys = tuple(image_keys)
        self.modality_by_key = dict(modality_by_key)
        self.policy = policy

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch

        if self.policy == "strict":
            return data
        case_id = str(data.get("streamrefine_case_id", ""))
        spec = VERIFIED_CASES.get(case_id)
        if spec is None:
            return data
        keys_by_modality = {self.modality_by_key[key]: key for key in self.image_keys}
        reference_key = keys_by_modality.get(spec["reference_modality"])
        corrected_key = keys_by_modality.get(spec["corrected_modality"])
        if reference_key is None or corrected_key is None:
            return data
        reference_affine = getattr(data[reference_key], "affine", None)
        current_affine = getattr(data[corrected_key], "affine", None)
        if reference_affine is None or current_affine is None:
            raise ValueError(f"Missing affine for verified BraTS case {case_id}")
        reference_affine = torch.as_tensor(reference_affine, dtype=torch.float64)
        current_affine = torch.as_tensor(current_affine, dtype=torch.float64)
        if reference_affine.shape != (4, 4) or current_affine.shape != (4, 4):
            raise ValueError(f"Invalid affine shape for verified BraTS case {case_id}")
        if torch.allclose(reference_affine, current_affine, atol=0.001, rtol=1e-05):
            return data
        expected_reference = torch.tensor(spec["reference_affine"], dtype=torch.float64)
        expected_original = torch.tensor(spec["original_affine"], dtype=torch.float64)
        checked_hashes = {}
        for modality, key in keys_by_modality.items():
            image = data[key]
            if tuple(image.shape) != (1, *spec["shape_hwd"]):
                raise ValueError(
                    f"Unverified shape for {case_id}/{modality}: {tuple(image.shape)}"
                )
            affine = getattr(image, "affine", None)
            expected = expected_original if key == corrected_key else expected_reference
            if (
                affine is None
                or tuple(affine.shape) != (4, 4)
                or (
                    not torch.allclose(
                        torch.as_tensor(affine, dtype=torch.float64),
                        expected,
                        atol=1e-05,
                        rtol=0.0,
                    )
                )
            ):
                raise ValueError(
                    f"Unverified affine pattern for {case_id}/{modality}; refusing to overwrite a different origin, spacing or orientation."
                )
            digest = oriented_voxel_sha256(image)
            if digest != spec["voxel_sha256"].get(modality):
                raise ValueError(
                    f"Unverified voxel content for {case_id}/{modality}: SHA256={digest}. The local audited data and this volume differ; inspect alignment before adding an exception."
                )
            checked_hashes[modality] = digest
        corrected = data[corrected_key].clone()
        corrected.affine = reference_affine.clone()
        result = dict(data)
        result[corrected_key] = corrected
        result["streamrefine_affine_corrections"] = [
            *data.get("streamrefine_affine_corrections", []),
            {
                "case_id": case_id,
                "policy": POLICY,
                "modality": spec["corrected_modality"],
                "reference_modality": spec["reference_modality"],
                "original_oriented_affine": current_affine.tolist(),
                "corrected_oriented_affine": reference_affine.tolist(),
                "origin_delta_before_correction": (
                    current_affine[:3, 3] - reference_affine[:3, 3]
                ).tolist(),
                "verified_voxel_sha256": checked_hashes,
                "resampled": False,
                "evidence": spec["evidence"],
            },
        ]
        warnings.warn(
            f"[BraTS affine] {case_id}: corrected verified {spec['corrected_modality']} origin to {spec['reference_modality']} in memory; voxel values unchanged, no resampling.",
            RuntimeWarning,
            stacklevel=2,
        )
        return result
