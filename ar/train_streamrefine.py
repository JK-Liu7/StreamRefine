"""Train continuous-latent StreamRefine with automatic benefit calibration."""

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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--latent-stats", type=Path)
    parser.add_argument(
        "--latent-stats-manifest",
        type=Path,
        help="Full training modality-cache manifest for automatic statistics; defaults to cache_manifest_train.jsonl beside data.train_manifest.",
    )
    parser.add_argument(
        "--no-auto-latent-stats",
        action="store_true",
        help="Require existing latent statistics instead of generating them when missing.",
    )
    parser.add_argument("--preprocessing-config", type=Path)
    parser.add_argument("--vidtok-config", type=Path)
    parser.add_argument("--vidtok-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve and print configuration without importing ML dependencies or touching data.",
    )
    return parser


def resolve_args_config(args: argparse.Namespace) -> dict:
    from streamrefine.config import resolve_config

    explicit = {
        "project.output_dir": None if args.output_dir is None else str(args.output_dir),
        "data.train_manifest": None
        if args.train_manifest is None
        else str(args.train_manifest),
        "data.val_manifest": None
        if args.val_manifest is None
        else str(args.val_manifest),
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
        "train.auto_resume": False if args.no_auto_resume else None,
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
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    config = resolve_args_config(args)
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    from streamrefine.training.latent_stats import prepare_training_latent_stats

    prepare_training_latent_stats(
        config,
        manifest_path=args.latent_stats_manifest,
        enabled=not args.no_auto_latent_stats,
    )
    from streamrefine.training.trainer import run_training
    from streamrefine.paths import resolve_path_string

    run_training(
        config,
        explicit_resume=None
        if args.resume is None
        else Path(resolve_path_string(args.resume, base_dir=Path.cwd())),
        init_checkpoint=None
        if args.init_checkpoint is None
        else Path(resolve_path_string(args.init_checkpoint, base_dir=Path.cwd())),
    )


if __name__ == "__main__":
    main()
