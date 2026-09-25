from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
AR_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = AR_ROOT.parent
if str(AR_ROOT) not in sys.path:
    sys.path.insert(0, str(AR_ROOT))
DEFAULT_VIDTOK_ROOT = PROJECT_ROOT / "external" / "VidTok"
DEFAULT_VIDTOK_CONFIG = (
    DEFAULT_VIDTOK_ROOT / "configs" / "vidtok_kl_noncausal_488_16chn.yaml"
)
DEFAULT_VIDTOK_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "vidtok_kl_noncausal_488_16chn.ckpt"
)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for VidTok reconstruction. Activate the StreamRefine training environment first."
        ) from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a cache dictionary, got {type(value).__name__}")
    return value


def _validate_cache(cache: dict[str, Any]) -> None:
    try:
        from streamrefine.tokenizer.cache_schema import validate_cache_dict
    except ImportError as exc:
        raise RuntimeError(
            "Could not import streamrefine.tokenizer.cache_schema from the ar checkout."
        ) from exc
    validate_cache_dict(cache)


def _resolve_candidate(value: str | Path, bases: list[Path]) -> Path | None:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve() if path.exists() else None
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return None


def _select_model_path(
    *,
    cli_value: Path | None,
    metadata_value: Any,
    local_default: Path,
    label: str,
    cache_path: Path,
) -> Path:
    from streamrefine.paths import resolve_path_string

    if cli_value is not None:
        cli_value = Path(resolve_path_string(cli_value, base_dir=Path.cwd()))
    if metadata_value:
        metadata_value = resolve_path_string(str(metadata_value))
    local_default = Path(resolve_path_string(local_default))
    if cli_value is not None:
        selected = _resolve_candidate(cli_value, [Path.cwd()])
        if selected is None:
            raise FileNotFoundError(f"CLI {label} path does not exist: {cli_value}")
        return selected
    if metadata_value:
        selected = _resolve_candidate(
            str(metadata_value), [cache_path.parent, AR_ROOT, PROJECT_ROOT, Path.cwd()]
        )
        if selected is not None:
            return selected
        print(
            f"WARNING: cache metadata {label} path is unavailable: {metadata_value}; trying local default {local_default}",
            file=sys.stderr,
        )
    selected = local_default.expanduser().resolve()
    if not selected.exists():
        raise FileNotFoundError(
            f"No usable {label} path. Pass the explicit CLI override; local default is missing: {selected}"
        )
    return selected


def _padding_starts_hwd(cache: dict[str, Any]) -> tuple[int, int, int]:
    info = cache.get("padding_info") or {}
    raw = None
    if isinstance(info, dict):
        raw = info.get("pad_width_hwd", info.get("pad_width"))
        if raw is None and all((key in info for key in ("h", "w", "d"))):
            raw = [info["h"], info["w"], info["d"]]
    if raw is None:
        return (0, 0, 0)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"Unsupported padding_info pad width: {raw!r}")
    starts: list[int] = []
    for axis in raw:
        if isinstance(axis, dict):
            before = axis.get("before", axis.get("left", 0))
        elif isinstance(axis, (list, tuple)) and len(axis) == 2:
            before = axis[0]
        elif isinstance(axis, int):
            before = 0
        else:
            raise ValueError(f"Unsupported axis padding entry: {axis!r}")
        starts.append(int(before))
    return tuple(starts)


def _crop_to_original_hwd(volume_hwd: Any, cache: dict[str, Any]) -> Any:
    original = cache.get("original_shape_hwd")
    if not isinstance(original, (list, tuple)) or len(original) != 3:
        raise ValueError("Cache has no valid original_shape_hwd")
    original_hwd = tuple((int(value) for value in original))
    start_hwd = _padding_starts_hwd(cache)
    stops = tuple((start + size for start, size in zip(start_hwd, original_hwd)))
    decoded_hwd = tuple((int(value) for value in volume_hwd.shape))
    if len(decoded_hwd) != 3 or any(
        (stop > decoded for stop, decoded in zip(stops, decoded_hwd))
    ):
        raise ValueError(
            f"Decoded HWD shape {decoded_hwd} cannot be cropped at {start_hwd} to original HWD {original_hwd}"
        )
    h0, w0, d0 = start_hwd
    h1, w1, d1 = stops
    return volume_hwd[h0:h1, w0:w1, d0:d1].contiguous()


def _inverse_normalize_hwd(
    volume_hwd: Any, normalization_info: Any, *, requested: bool
) -> tuple[Any, dict[str, Any]]:
    import math
    import torch

    if not isinstance(normalization_info, dict):
        if requested:
            raise ValueError(
                "--inverse-normalization requires cache.normalization_info metadata"
            )
        return (
            volume_hwd,
            {
                "requested": False,
                "applied": False,
                "policy": None,
                "output_space": "tokenizer_normalized",
            },
        )
    policy = str(normalization_info.get("policy", "")).strip().lower()
    if not requested:
        return (
            volume_hwd,
            {
                "requested": False,
                "applied": False,
                "policy": policy or None,
                "output_space": "tokenizer_normalized",
            },
        )
    if not policy:
        raise ValueError(
            "Cannot inverse-normalize: cache.normalization_info has no recognized policy"
        )
    if policy == "pre_normalized_passthrough":
        return (
            volume_hwd,
            {
                "requested": True,
                "applied": False,
                "identity_passthrough": True,
                "policy": policy,
                "output_space": "pre_normalized_passthrough",
                "clipped_tails_recoverable": True,
            },
        )
    if policy == "hu_clip_to_minus1_1":
        input_range = normalization_info.get("input_clip")
        output_space = "clipped_hu"
    elif policy in {"percentile_to_minus1_1", "positive_percentile_to_minus1_1"}:
        input_range = normalization_info.get("actual_input_low_high")
        output_space = "clipped_input_intensity"
    else:
        raise ValueError(
            f"Unsupported normalization policy for inverse mapping: {policy!r}. Expected hu_clip_to_minus1_1, percentile_to_minus1_1, positive_percentile_to_minus1_1, or pre_normalized_passthrough."
        )
    output_range = normalization_info.get("output_range", [-1.0, 1.0])
    if not isinstance(input_range, (list, tuple)) or len(input_range) != 2:
        raise ValueError(
            f"normalization_info for policy={policy!r} lacks a two-value inverse range"
        )
    if not isinstance(output_range, (list, tuple)) or len(output_range) != 2:
        raise ValueError("normalization_info.output_range must contain two values")
    input_low, input_high = (float(input_range[0]), float(input_range[1]))
    output_low, output_high = (float(output_range[0]), float(output_range[1]))
    values = (input_low, input_high, output_low, output_high)
    if not all((math.isfinite(value) for value in values)):
        raise ValueError(f"Non-finite normalization range metadata: {values}")
    if input_high <= input_low or output_high <= output_low:
        raise ValueError(
            f"Invalid normalization ranges: input={input_range}, output={output_range}"
        )
    restored = (volume_hwd.float() - output_low) / (output_high - output_low)
    restored = restored * (input_high - input_low) + input_low
    if not bool(torch.isfinite(restored).all().item()):
        raise ValueError("Inverse normalization produced NaN/Inf")
    clipped = bool(normalization_info.get("clipped", True))
    return (
        restored,
        {
            "requested": True,
            "applied": True,
            "identity_passthrough": False,
            "policy": policy,
            "input_range": [input_low, input_high],
            "normalized_range": [output_low, output_high],
            "output_space": output_space,
            "clipped_tails_recoverable": not clipped,
        },
    )


def _cache_affine(cache: dict[str, Any]) -> Any:
    import torch

    affine = cache.get("affine")
    if affine is None and isinstance(cache.get("modality_metadata"), dict):
        affine = cache["modality_metadata"].get("affine")
    if affine is None:
        raise ValueError("Cache has no affine; refusing to write a geometry-less NIfTI")
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to save NIfTI output") from exc
    result = np.asarray(affine, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite 4x4 affine, got {result.shape}")
    return result


def _safe_stem(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    text = re.sub("[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or fallback


def _save_nifti(volume_hwd: Any, affine: Any, output_path: Path) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required to save NIfTI output. Install it in the StreamRefine environment, then rerun this command."
        ) from exc
    data = volume_hwd.detach().cpu().float().numpy()
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(output_path))


def _save_middle_png(volume_hwd: Any, output_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for --save-middle-png. Install pillow or omit that flag."
        ) from exc
    import torch

    middle = volume_hwd[:, :, int(volume_hwd.shape[2]) // 2].detach().cpu().float()
    finite = torch.isfinite(middle)
    if not bool(finite.all().item()):
        raise ValueError(
            "Cannot save PNG because the reconstructed middle slice contains NaN/Inf"
        )
    low = torch.quantile(middle, 0.01)
    high = torch.quantile(middle, 0.99)
    if float((high - low).abs().item()) < 1e-12:
        scaled = torch.zeros_like(middle)
    else:
        scaled = ((middle - low) / (high - low)).clamp(0, 1)
    array = (scaled * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array, mode="L").save(output_path)


def _rgb_to_gray_thw(decoded_rgb: Any, channel_mode: str) -> Any:
    if channel_mode == "middle":
        return decoded_rgb[:, int(decoded_rgb.shape[1]) // 2]
    if channel_mode == "average":
        return decoded_rgb.mean(dim=1)
    raise ValueError(f"Unsupported channel_mode: {channel_mode!r}")


def _decode_full_mean(
    encoder: Any,
    cache: dict[str, Any],
    *,
    device: str,
    precision: str,
    channel_mode: str,
) -> Any:
    import torch

    latent = cache["latent_mu"].float().unsqueeze(0).to(device)
    if tuple(latent.shape[:2]) != (1, 16):
        raise ValueError(
            f"Expected decoder input [1,16,D,H,W], got {tuple(latent.shape)}"
        )
    with torch.inference_mode():
        decoded_rgb = encoder.decode_raw_mean(latent, precision=precision)
    if not torch.is_tensor(decoded_rgb) or decoded_rgb.ndim != 5:
        raise TypeError("decode_raw_mean must return a tensor [B,3,T,H,W]")
    if tuple(decoded_rgb.shape[:2]) != (1, 3):
        raise ValueError(
            f"Expected decoded VidTok RGB [1,3,T,H,W], got {tuple(decoded_rgb.shape)}"
        )
    if not bool(torch.isfinite(decoded_rgb).all().item()):
        raise ValueError("VidTok decoder returned NaN/Inf")
    gray_thw = _rgb_to_gray_thw(decoded_rgb, channel_mode)[0]
    return gray_thw.permute(1, 2, 0).contiguous().cpu()


def _decode_tiled_mean(
    encoder: Any,
    cache: dict[str, Any],
    *,
    device: str,
    precision: str,
    channel_mode: str,
    batch_size: int,
) -> Any:
    """Memory-safe overlap decode using the cache's canonical patch table."""
    import torch
    import torch.nn.functional as functional
    from streamrefine.tokenizer.weighted_stitch import make_importance_map_dhw

    if int(batch_size) <= 0:
        raise ValueError("decode batch_size must be positive")
    latent = cache["latent_mu"].float()
    table = cache["patch_index_table"]
    if not isinstance(table, list) or not table:
        raise ValueError("Cache patch_index_table is empty")
    padded_h, padded_w, padded_d = (int(item) for item in cache["padded_shape_hwd"])
    patch_h, patch_w, patch_d = (int(item) for item in cache["patch_size_hwd"])
    latent_weight = make_importance_map_dhw(
        (patch_d // 4, patch_h // 8, patch_w // 8),
        eps=float(cache.get("importance_floor", 0.001)),
        mode=str(cache.get("importance_mode", "hann")),
        gaussian_sigma_scale=float(cache.get("gaussian_sigma_scale") or 0.125),
    )
    image_weight = functional.interpolate(
        latent_weight.unsqueeze(0),
        size=(patch_d, patch_h, patch_w),
        mode="trilinear",
        align_corners=False,
    )[0, 0].contiguous()
    image_weight = image_weight / image_weight.max().clamp_min(1e-12)
    image_sum = torch.zeros((padded_d, padded_h, padded_w), dtype=torch.float32)
    weight_sum = torch.zeros_like(image_sum)
    for batch_start in range(0, len(table), int(batch_size)):
        entries = table[batch_start : batch_start + int(batch_size)]
        tiles = [
            latent[
                :,
                int(entry["ld0"]) : int(entry["ld1"]),
                int(entry["lh0"]) : int(entry["lh1"]),
                int(entry["lw0"]) : int(entry["lw1"]),
            ]
            for entry in entries
        ]
        batch = torch.stack(tiles, dim=0).to(device)
        try:
            with torch.inference_mode():
                decoded_rgb = encoder.decode_raw_mean(batch, precision=precision)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory during tiled reconstruction; rerun with --decode-batch-size 1."
            ) from exc
        expected = (len(entries), 3, patch_d, patch_h, patch_w)
        if tuple(decoded_rgb.shape) != expected:
            raise ValueError(
                f"Tiled decoder expected {expected}, got {tuple(decoded_rgb.shape)}"
            )
        if not bool(torch.isfinite(decoded_rgb).all().item()):
            raise ValueError("VidTok tiled decoder returned NaN/Inf")
        gray_batch = _rgb_to_gray_thw(decoded_rgb, channel_mode).float().cpu()
        for gray, entry in zip(gray_batch, entries):
            d0, d1 = (int(entry["d0"]), int(entry["d1"]))
            h0, h1 = (int(entry["h0"]), int(entry["h1"]))
            w0, w1 = (int(entry["w0"]), int(entry["w1"]))
            image_sum[d0:d1, h0:h1, w0:w1].add_(gray * image_weight)
            weight_sum[d0:d1, h0:h1, w0:w1].add_(image_weight)
    if not bool((weight_sum > 0).all().item()):
        raise RuntimeError("Tiled reconstruction left uncovered output voxels")
    decoded_dhw = image_sum / weight_sum
    if not bool(torch.isfinite(decoded_dhw).all().item()):
        raise RuntimeError("Tiled reconstruction produced NaN/Inf")
    return decoded_dhw.permute(1, 2, 0).contiguous()


def reconstruct(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for VidTok reconstruction. Activate the StreamRefine training environment first."
        ) from exc
    try:
        from streamrefine.tokenizer.vidtok_kl_wrapper import load_vidtok_kl_mean_encoder
    except ImportError as exc:
        raise RuntimeError(
            "Could not import streamrefine.tokenizer.vidtok_kl_wrapper from the ar checkout."
        ) from exc
    from streamrefine.paths import resolve_path_string

    cache_path = (
        Path(resolve_path_string(args.cache, base_dir=Path.cwd()))
        .expanduser()
        .resolve()
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")
    cache = _torch_load(cache_path)
    _validate_cache(cache)
    vidtok_root = _select_model_path(
        cli_value=args.vidtok_root,
        metadata_value=cache.get("tokenizer_repo_root"),
        local_default=DEFAULT_VIDTOK_ROOT,
        label="VidTok repo root",
        cache_path=cache_path,
    )
    config_path = _select_model_path(
        cli_value=args.vidtok_config,
        metadata_value=cache.get("tokenizer_config"),
        local_default=DEFAULT_VIDTOK_CONFIG,
        label="VidTok model config",
        cache_path=cache_path,
    )
    config_sha256 = _sha256_file(config_path)
    expected_config_sha256 = (
        str(cache.get("tokenizer_config_sha256", "")).strip().lower()
    )
    if expected_config_sha256 and config_sha256.lower() != expected_config_sha256:
        raise ValueError(
            f"Selected VidTok model config does not match cache provenance: cache={expected_config_sha256}, selected={config_sha256}."
        )
    checkpoint_path = _select_model_path(
        cli_value=args.ckpt,
        metadata_value=cache.get("tokenizer_checkpoint"),
        local_default=DEFAULT_VIDTOK_CHECKPOINT,
        label="VidTok checkpoint",
        cache_path=cache_path,
    )
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    expected_sha256 = str(cache.get("tokenizer_checkpoint_sha256", "")).strip().lower()
    if checkpoint_sha256.lower() != expected_sha256:
        raise ValueError(
            f"Selected checkpoint does not match the cache tokenizer hash: cache={expected_sha256}, selected={checkpoint_sha256}. Use the exact checkpoint that generated latent_mu."
        )
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    encoder = load_vidtok_kl_mean_encoder(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        vidtok_root=vidtok_root,
        device=device,
        precision=args.precision,
        latent_channels=16,
        lightweight_loss=True,
    )
    if args.decode_mode == "tiled":
        padded_hwd = _decode_tiled_mean(
            encoder,
            cache,
            device=device,
            precision=args.precision,
            channel_mode=args.channel_mode,
            batch_size=args.decode_batch_size,
        )
    else:
        padded_hwd = _decode_full_mean(
            encoder,
            cache,
            device=device,
            precision=args.precision,
            channel_mode=args.channel_mode,
        )
    if args.clamp_decoder_range:
        padded_hwd = padded_hwd.clamp(-1.0, 1.0)
    volume_hwd = _crop_to_original_hwd(padded_hwd, cache)
    volume_hwd, inverse_summary = _inverse_normalize_hwd(
        volume_hwd,
        cache.get("normalization_info"),
        requested=args.inverse_normalization,
    )
    affine = _cache_affine(cache)
    output_dir = (
        Path(resolve_path_string(args.output_dir, base_dir=Path.cwd()))
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    case_stem = _safe_stem(cache.get("case_id"), cache_path.stem)
    modality_stem = _safe_stem(cache.get("modality"), "modality")
    nifti_name = args.nifti_name or f"{case_stem}_{modality_stem}_recon.nii.gz"
    if not nifti_name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("--nifti-name must end in .nii or .nii.gz")
    nifti_path = output_dir / nifti_name
    png_path = (
        output_dir / f"{case_stem}_{modality_stem}_middle.png"
        if args.save_middle_png
        else None
    )
    if nifti_path.exists() and (not args.overwrite):
        raise FileExistsError(
            f"NIfTI exists (pass --overwrite to replace it): {nifti_path}"
        )
    if png_path is not None and png_path.exists() and (not args.overwrite):
        raise FileExistsError(
            f"PNG exists (pass --overwrite to replace it): {png_path}"
        )
    _save_nifti(volume_hwd, affine, nifti_path)
    if png_path is not None:
        _save_middle_png(volume_hwd, png_path)
    return {
        "cache": str(cache_path),
        "nifti": str(nifti_path),
        "middle_png": str(png_path) if png_path else None,
        "decoded_padded_shape_hwd": [int(value) for value in padded_hwd.shape],
        "saved_original_shape_hwd": [int(value) for value in volume_hwd.shape],
        "channel_mode": args.channel_mode,
        "decode_mode": args.decode_mode,
        "decode_batch_size": int(args.decode_batch_size),
        "decoder_range_clamped": bool(args.clamp_decoder_range),
        "latent_stats_applied": False,
        "inverse_normalization": inverse_summary,
        "vidtok_root": str(vidtok_root),
        "vidtok_config": str(config_path),
        "vidtok_config_sha256": config_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "device": device,
        "precision": args.precision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode raw latent_mu from one StreamRefine VidTok-KL mean cache and write an affine-preserving NIfTI reconstruction."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, required=True
    )
    parser.add_argument(
        "--vidtok-root",
        "--vidtok_root",
        dest="vidtok_root",
        type=Path,
        default=None,
        help="CLI override; cache metadata/local default otherwise.",
    )
    parser.add_argument(
        "--vidtok-config",
        "--vidtok_config",
        dest="vidtok_config",
        type=Path,
        default=None,
        help="CLI override for the VidTok model YAML.",
    )
    parser.add_argument(
        "--ckpt",
        "--checkpoint",
        dest="ckpt",
        type=Path,
        default=None,
        help="CLI override for the VidTok checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:1.",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--decode-mode",
        "--decode_mode",
        dest="decode_mode",
        choices=("tiled", "full"),
        default="tiled",
        help="Tiled overlap decode is the memory-safe default; full is a small-volume reference path.",
    )
    parser.add_argument(
        "--decode-batch-size",
        "--decode_batch_size",
        dest="decode_batch_size",
        type=int,
        default=1,
        help="Number of latent tiles decoded together in tiled mode.",
    )
    parser.add_argument(
        "--channel-mode",
        "--channel_mode",
        dest="channel_mode",
        choices=("middle", "average"),
        default="average",
    )
    parser.add_argument("--nifti-name", "--nifti_name", dest="nifti_name", default=None)
    parser.add_argument(
        "--save-middle-png",
        "--save_middle_png",
        dest="save_middle_png",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing NIfTI/PNG outputs."
    )
    inverse_group = parser.add_mutually_exclusive_group()
    inverse_group.add_argument(
        "--inverse-normalization",
        "--inverse_normalization",
        dest="inverse_normalization",
        action="store_true",
        help="Map decoder [-1,1] values back to the cache-recorded clipped HU/percentile range.",
    )
    inverse_group.add_argument(
        "--no-inverse-normalization",
        "--no_inverse_normalization",
        dest="inverse_normalization",
        action="store_false",
        help="Keep tokenizer-normalized intensities (default).",
    )
    parser.add_argument(
        "--no-clamp-decoder-range",
        "--no_clamp_decoder_range",
        dest="clamp_decoder_range",
        action="store_false",
        help="Do not clamp VidTok decoder output to [-1,1] before saving.",
    )
    parser.set_defaults(clamp_decoder_range=True, inverse_normalization=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = reconstruct(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
