"""Synchronized full-volume sliding-window StreamRefine inference."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence

AR_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=AR_ROOT / "configs/streamrefine/base.yaml"
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=AR_ROOT / "configs/streamrefine/datasets/brats24.yaml",
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=AR_ROOT / "configs/streamrefine/methods/anatomy_aware.yaml",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--latent-stats", type=Path)
    parser.add_argument("--preprocessing-config", type=Path)
    parser.add_argument("--vidtok-config", type=Path)
    parser.add_argument("--vidtok-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve and print configuration without importing ML dependencies or touching data.",
    )
    return parser


def resolve_args_config(args: argparse.Namespace) -> dict:
    from streamrefine.config import resolve_config

    manifest_key = f"data.{args.split}_manifest"
    explicit = {
        manifest_key: None if args.manifest is None else str(args.manifest),
        "latent.stats_path": None
        if args.latent_stats is None
        else str(args.latent_stats),
        "data.preprocessing_config": None
        if args.preprocessing_config is None
        else str(args.preprocessing_config),
        "vidtok.config": None
        if args.vidtok_config is None
        else str(args.vidtok_config),
        "vidtok.checkpoint": None
        if args.vidtok_checkpoint is None
        else str(args.vidtok_checkpoint),
        "inference.output_dir": None
        if args.output_dir is None
        else str(args.output_dir),
    }
    return resolve_config(
        base_path=args.base_config,
        dataset_path=args.dataset_config,
        method_path=args.method_config,
        overrides=args.overrides,
        explicit=explicit,
        require_runtime_paths=not args.print_config,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.print_config and args.checkpoint is None:
        parser.error("--checkpoint is required for inference")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    config = resolve_args_config(args)
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    from streamrefine.inference.runner import run_inference
    from streamrefine.paths import resolve_path_string

    run_inference(
        config,
        checkpoint_path=Path(resolve_path_string(args.checkpoint, base_dir=Path.cwd())),
        split=args.split,
        max_cases=args.max_cases,
        skip_completed=not args.no_skip_completed,
    )


if __name__ == "__main__":
    main()
