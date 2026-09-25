from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
AR_ROOT = THIS_FILE.parents[1]
if str(AR_ROOT) not in sys.path:
    sys.path.insert(0, str(AR_ROOT))


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to inspect a .pt cache. Activate the StreamRefine training environment before running this command."
        ) from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a cache dictionary, got {type(value).__name__}")
    return value


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache(
    cache: dict[str, Any],
    expected_checkpoint_sha256: str | None,
    expected_config_sha256: str | None,
    expected_generation_contract_sha256: str | None,
) -> None:
    try:
        from streamrefine.tokenizer.cache_schema import validate_cache_dict
    except ImportError as exc:
        raise RuntimeError(
            "Could not import streamrefine.tokenizer.cache_schema. Run this tool from the StreamRefine ar checkout, or add that checkout to PYTHONPATH."
        ) from exc
    validate_cache_dict(
        cache,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_generation_contract_sha256=expected_generation_contract_sha256,
    )


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected a tensor, got {type(tensor).__name__}")
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    total = int(tensor.numel())
    result: dict[str, Any] = {
        "shape": [int(v) for v in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": total,
        "finite_count": finite_count,
        "finite_ratio": float(finite_count / total) if total else 0.0,
    }
    if finite_count:
        values = tensor[finite].float()
        result["min"] = float(values.min().item())
        result["max"] = float(values.max().item())
        result["mean"] = float(values.mean().item())
    else:
        result.update({"min": None, "max": None, "mean": None})
    return result


def summarize_cache(
    cache_path: Path,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_generation_contract_sha256: str | None = None,
) -> dict[str, Any]:
    cache_path = cache_path.expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")
    cache = _torch_load(cache_path)
    import torch

    _validate_cache(
        cache,
        expected_checkpoint_sha256,
        expected_config_sha256,
        expected_generation_contract_sha256,
    )
    latent = cache["latent_mu"]
    hard_mask = cache["valid_mask_latent"]
    soft_mask = cache["valid_mask_latent_soft"]
    if not torch.is_tensor(hard_mask) or not torch.is_tensor(soft_mask):
        raise TypeError("valid_mask_latent and valid_mask_latent_soft must be tensors")
    hard_ratio = (
        float(hard_mask.bool().float().mean().item()) if hard_mask.numel() else 0.0
    )
    soft_finite = torch.isfinite(soft_mask)
    soft_values = soft_mask[soft_finite].float()
    forbidden = sorted(
        (
            key
            for key in (
                "fsq_scalars",
                "packed_index",
                "fsq_levels",
                "latent_logvar",
                "sampled_z",
            )
            if key in cache
        )
    )
    return {
        "schema_valid": True,
        "cache_path": str(cache_path),
        "cache_file_sha256": _sha256_file(cache_path),
        "cache_version": cache.get("cache_version"),
        "dataset": cache.get("dataset"),
        "split": cache.get("split"),
        "case_id": cache.get("case_id"),
        "modality": cache.get("modality"),
        "posterior_mode": cache.get("posterior_mode"),
        "latent_layout": cache.get("latent_layout"),
        "latent": _tensor_summary(latent),
        "valid_mask": {
            "shape": [int(v) for v in hard_mask.shape],
            "dtype": str(hard_mask.dtype).removeprefix("torch."),
            "hard_valid_ratio": hard_ratio,
            "soft_valid_ratio": float(soft_values.mean().item())
            if soft_values.numel()
            else 0.0,
            "soft_min": float(soft_values.min().item())
            if soft_values.numel()
            else None,
            "soft_max": float(soft_values.max().item())
            if soft_values.numel()
            else None,
            "soft_finite_ratio": float(soft_finite.sum().item() / soft_mask.numel())
            if soft_mask.numel()
            else 0.0,
        },
        "original_shape_hwd": cache.get("original_shape_hwd"),
        "padded_shape_hwd": cache.get("padded_shape_hwd"),
        "latent_shape_dhw": cache.get("latent_shape_dhw"),
        "affine_corrections": cache.get("affine_corrections", []),
        "num_patches": len(cache.get("patch_index_table", [])),
        "generation_contract_sha256": cache.get("generation_contract_sha256"),
        "generation_contract": cache.get("generation_contract"),
        "input_snapshot_sha256": cache.get("input_snapshot_sha256"),
        "tokenizer": {
            "type": cache.get("tokenizer_type"),
            "causal": cache.get("tokenizer_causal"),
            "config": cache.get("tokenizer_config"),
            "config_sha256": cache.get("tokenizer_config_sha256"),
            "checkpoint": cache.get("tokenizer_checkpoint"),
            "checkpoint_sha256": cache.get("tokenizer_checkpoint_sha256"),
            "git_commit": cache.get("tokenizer_git_commit"),
        },
        "forbidden_fields_present": forbidden,
        "keys": sorted((str(key) for key in cache)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inspect one StreamRefine VidTok-KL posterior-mean cache."
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="Per-modality .pt cache to inspect."
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        "--expected_checkpoint_sha256",
        dest="expected_checkpoint_sha256",
        default=None,
        help="Optionally require this exact tokenizer checkpoint SHA256.",
    )
    parser.add_argument(
        "--expected-config-sha256",
        "--expected_config_sha256",
        dest="expected_config_sha256",
        default=None,
        help="Optionally require this exact tokenizer model-config SHA256.",
    )
    parser.add_argument(
        "--expected-generation-contract-sha256",
        "--expected_generation_contract_sha256",
        dest="expected_generation_contract_sha256",
        default=None,
        help="Optionally require this exact cache-generation contract SHA256.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the summary as formatted JSON."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_cache(
            args.cache,
            args.expected_checkpoint_sha256,
            args.expected_config_sha256,
            args.expected_generation_contract_sha256,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                rendered = value
            print(f"{key}: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
