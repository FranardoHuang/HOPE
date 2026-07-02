#!/usr/bin/env python3
"""Prepare or verify the Agibot A3 ping-pong URDF asset used by Isaac training.

The source of truth is the tracked Agibot ping-pong URDF package under
``agi/URDF/A3T2.5-URDF-std-pingpang``. The generated Isaac asset lives under
``whole_body_tracking/assets/agibot_a3`` and rewrites mesh references from
``package://.../meshes/*.STL`` to ``../meshes/*.STL`` so Isaac Lab can load it
without ROS package resolution.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "agi" / "URDF" / "A3T2.5-URDF-std-pingpang"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "agibot_a3"
)
REQUIRED_PINGPONG_MESHES = {
    "right_hand_pingpang_Link.STL",
    "pingpang_red_Link.STL",
    "pingpang_black_Link.STL",
    "pingbang_ball_Link.STL",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Recreate the generated output root before copying.")
    parser.add_argument("--check", action="store_true", help="Only verify the already prepared output asset.")
    return parser.parse_args()


def _rewrite_mesh_ref(filename: str) -> str:
    if filename.startswith("package://") and "/meshes/" in filename:
        return "../meshes/" + filename.rsplit("/meshes/", 1)[1]
    return filename


def _source_paths(source_root: Path) -> tuple[Path, Path, Path]:
    urdf = source_root / "urdf" / "URDF-JOINT-LINK.urdf"
    meshes = source_root / "meshes"
    config = source_root / "config"
    return urdf, meshes, config


def prepare(source_root: Path, output_root: Path, force: bool) -> None:
    source_urdf, source_meshes, source_config = _source_paths(source_root)
    for path in (source_urdf, source_meshes):
        if not path.exists():
            raise FileNotFoundError(f"missing A3 source asset path: {path}")

    if output_root.exists() and force:
        shutil.rmtree(output_root)

    urdf_dir = output_root / "urdf"
    mesh_dir = output_root / "meshes"
    config_dir = output_root / "config"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    if source_config.exists():
        config_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_meshes, mesh_dir, dirs_exist_ok=True)
    if source_config.exists():
        shutil.copytree(source_config, config_dir, dirs_exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", _rewrite_mesh_ref(filename))
    tree.write(urdf_dir / "model.urdf", encoding="utf-8", xml_declaration=True)

    print(f"[prepare_a3_isaac_asset] prepared: {output_root}")


def _mesh_refs(model_urdf: Path) -> list[str]:
    tree = ET.parse(model_urdf)
    refs: list[str] = []
    for mesh in tree.getroot().findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            refs.append(filename)
    return refs


def check(output_root: Path) -> bool:
    model_urdf = output_root / "urdf" / "model.urdf"
    if not model_urdf.exists():
        print(f"[FAIL] missing prepared URDF: {model_urdf}")
        return False

    refs = _mesh_refs(model_urdf)
    failures: list[str] = []

    stale_package_refs = [ref for ref in refs if ref.startswith("package://") and "/meshes/" in ref]
    if stale_package_refs:
        failures.append("stale package:// mesh refs remain: " + ", ".join(sorted(set(stale_package_refs))[:8]))

    rel_mesh_refs = [ref for ref in refs if ref.startswith("../meshes/")]
    for ref in rel_mesh_refs:
        resolved = (model_urdf.parent / ref).resolve()
        if not resolved.exists():
            failures.append(f"missing mesh referenced by URDF: {ref}")

    ref_names = {Path(ref).name for ref in rel_mesh_refs}
    missing_required_refs = sorted(REQUIRED_PINGPONG_MESHES - ref_names)
    if missing_required_refs:
        failures.append("prepared URDF does not reference required ping-pong meshes: " + ", ".join(missing_required_refs))

    mesh_dir = output_root / "meshes"
    missing_required_files = sorted(name for name in REQUIRED_PINGPONG_MESHES if not (mesh_dir / name).exists())
    if missing_required_files:
        failures.append("prepared mesh directory is missing required ping-pong meshes: " + ", ".join(missing_required_files))

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return False

    print(
        "[prepare_a3_isaac_asset] OK: "
        f"{len(refs)} mesh refs, {len(rel_mesh_refs)} relative mesh refs, ping-pong meshes present"
    )
    return True


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not args.check:
        prepare(source_root, output_root, args.force)
    return 0 if check(output_root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
