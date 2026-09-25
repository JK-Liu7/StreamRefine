"""Check release syntax, configuration, imports, and optional file checksums."""

from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path
from streamrefine.config import resolve_config

ROOT = Path(__file__).resolve().parent.parent


def validate_release(check_manifest: bool = False) -> dict[str, int]:
    files = [
        p
        for p in sorted(ROOT.rglob("*.py"))
        if p.relative_to(ROOT).parts[0] in {"ar", "vidtok_rec"}
    ]
    for path in files:
        tree = ast.parse(
            path.read_text(encoding="utf-8"), filename=path.relative_to(ROOT).as_posix()
        )
        compile(tree, path.name, "exec")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("streamrefine.")
            ):
                candidate = ROOT / "ar" / Path(*node.module.split("."))
                if not candidate.with_suffix(".py").is_file() and (
                    not (candidate / "__init__.py").is_file()
                ):
                    raise ValueError(f"Missing dependency {node.module} in {path.name}")
    for dataset, base in (
        ("brats24", "base"),
        ("synthrad", "base_synthrad"),
        ("autopet", "base_autopet"),
    ):
        config = resolve_config(
            base_path=ROOT / f"ar/configs/streamrefine/{base}.yaml",
            dataset_path=ROOT / f"ar/configs/streamrefine/datasets/{dataset}.yaml",
            method_path=ROOT / "ar/configs/streamrefine/methods/anatomy_aware.yaml",
        )
        if (
            config["benefit"]["translation_metric"] != "latent_mae"
            or config["benefit"]["anatomy_metric"] != "pir"
        ):
            raise ValueError(f"Unexpected objective for {dataset}")
    checked = 0
    if check_manifest:
        for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError("Manifest path escapes release")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            checked += 1
    return {"python_files": len(files), "dataset_configs": 3, "checksums": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    print(validate_release(args.manifest))
