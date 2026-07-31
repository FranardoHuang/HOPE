#!/usr/bin/env python3
"""Materialize exact AgiBot A3 collision-component OBBs for the table guard.

The Isaac A3 asset is prepared from the tracked vendor URDF by copying every
mesh byte and rewriting only its path.  This tool therefore reads that tracked
source directly, folds fixed-joint collision children into their runtime rigid
body, and emits one conservative local OBB per collision component.

The resulting artifact is data, not a hand-tuned safety margin.  Each component
OBB contains every vertex of the collision mesh from which it was derived.
Runtime may conservatively broaden the rotated OBB to a world AABB, but must
never shrink these materialized half axes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "agi" / "URDF" / "A3T2.5-URDF-std-pingpang"
)
DEFAULT_SOURCE_URDF = DEFAULT_SOURCE_ROOT / "urdf" / "URDF-JOINT-LINK.urdf"
DEFAULT_BODY_ORDER_SOURCE = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "agibot_a3"
    / "hope_env_cfg.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "configs"
    / "a3_table_collision_proxy_20260731"
    / "a3_table_collision_components.v1.json"
)
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "a3_table_collision_component_obb_v1"
PINNED_RUNTIME_USD_BUNDLE_TREE_SHA256 = (
    "716487dfdf02a5973f78263f0ae8a09e4680c04159e57dbe20796b7825dbeb4d"
)
PINNED_RUNTIME_USD_FILES = {
    ".asset_hash": (
        "3816a1a4bbca423e575650b6d6065f5141a7c840b02dd30c72d4278a225ed499",
        32,
    ),
    "config.yaml": (
        "3e35ad4c3ef7c21a10ce413be3ce28777bb83afee4b63fc245b30bd59a9818c2",
        1689,
    ),
    "configuration/model_base.usd": (
        "8e521141bfee4274b8a2369d382cdd8aac9bb1cfcae5bfa480666a1935a7fb42",
        21882690,
    ),
    "configuration/model_physics.usd": (
        "5b5fc00b96566be295a0cd4eb6b0cd276e360d9cca189057cef452ad0bfc7981",
        11164,
    ),
    "configuration/model_sensor.usd": (
        "c76c5bdd9e9b5434d72b45c9001858a9c80363656272011ed50d1419149ca60a",
        682,
    ),
    "model.usd": (
        "1b3fecd7685cd98ca80de226fbf89985b77b8a8cfc6a36f18fcc22e65080693c",
        1636,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument(
        "--body-order-source", type=Path, default=DEFAULT_BODY_ORDER_SOURCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runtime-usd-bundle-root",
        type=Path,
        required=True,
        help=(
            "Reviewed six-file Pod USD root.  Formal generation and --check "
            "both revalidate every byte and the canonical tree digest."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and require byte equality with the existing output.",
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _float_triplet(text: str | None, *, default: Sequence[float]) -> tuple[float, float, float]:
    if text is None:
        values = tuple(float(value) for value in default)
    else:
        values = tuple(float(value) for value in text.split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("expected one finite three-vector")
    return values


def _identity_rotation() -> tuple[tuple[float, float, float], ...]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rpy_rotation(
    rpy: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_vector(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        sum(float(row[index]) * float(vector[index]) for index in range(3))
        for row in matrix
    )


def _matrix_matrix(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(
                float(left[row][inner]) * float(right[inner][column])
                for inner in range(3)
            )
            for column in range(3)
        )
        for row in range(3)
    )


def _vector_add(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _compose(
    parent_rotation: Sequence[Sequence[float]],
    parent_translation: Sequence[float],
    child_rotation: Sequence[Sequence[float]],
    child_translation: Sequence[float],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[float, float, float],
]:
    return (
        _matrix_matrix(parent_rotation, child_rotation),
        _vector_add(
            _matrix_vector(parent_rotation, child_translation),
            parent_translation,
        ),
    )


def _origin_transform(
    element: ET.Element | None,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[float, float, float],
]:
    if element is None:
        return _identity_rotation(), (0.0, 0.0, 0.0)
    return (
        _rpy_rotation(
            _float_triplet(element.get("rpy"), default=(0.0, 0.0, 0.0))
        ),
        _float_triplet(element.get("xyz"), default=(0.0, 0.0, 0.0)),
    )


def _body_order(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name)
            and target.id == "TABLE_CONTACT_BODY_NAMES"
            for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        names = tuple(str(name) for name in value)
        if len(names) != 32 or len(set(names)) != 32:
            raise ValueError("TABLE_CONTACT_BODY_NAMES must be 32 unique names")
        return names
    raise ValueError("TABLE_CONTACT_BODY_NAMES assignment not found")


def _stl_vertices(path: Path) -> list[tuple[float, float, float]]:
    payload = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + triangle_count * 50 == len(payload):
            for triangle in range(triangle_count):
                base = 84 + triangle * 50 + 12
                for vertex in range(3):
                    values = struct.unpack_from(
                        "<fff", payload, base + vertex * 12
                    )
                    vertices.append(tuple(float(value) for value in values))
    if not vertices:
        text = payload.decode("ascii")
        for line in text.splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in fields[1:]))
    if not vertices or not all(
        math.isfinite(value) for vertex in vertices for value in vertex
    ):
        raise ValueError(f"STL has no finite vertices: {path}")
    return vertices


def _bounds(
    vertices: Iterable[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lower = [float("inf")] * 3
    upper = [float("-inf")] * 3
    count = 0
    for vertex in vertices:
        count += 1
        for axis in range(3):
            value = float(vertex[axis])
            lower[axis] = min(lower[axis], value)
            upper[axis] = max(upper[axis], value)
    if count == 0:
        raise ValueError("cannot bound an empty vertex set")
    center = tuple((lo + hi) * 0.5 for lo, hi in zip(lower, upper))
    half = tuple((hi - lo) * 0.5 for lo, hi in zip(lower, upper))
    if not all(value > 0.0 and math.isfinite(value) for value in half):
        raise ValueError("collision mesh must have positive finite 3-D bounds")
    return center, half


def _runtime_usd_binding(bundle_root: Path) -> dict[str, object]:
    entries = [
        {"path": path, "sha256": values[0], "size": values[1]}
        for path, values in sorted(PINNED_RUNTIME_USD_FILES.items())
    ]
    file_count = len(entries)
    total_file_bytes = sum(int(entry["size"]) for entry in entries)
    tree_sha256 = _sha256_bytes(_canonical_json_bytes(entries))
    if (
        file_count != 6
        or total_file_bytes != 21897893
        or tree_sha256 != PINNED_RUNTIME_USD_BUNDLE_TREE_SHA256
    ):
        raise ValueError("embedded A3 runtime USD bundle receipt is inconsistent")
    configured_root = bundle_root.expanduser()
    if configured_root.is_symlink():
        raise ValueError("runtime USD bundle root must not be a symlink")
    bundle_root = configured_root.resolve(strict=True)
    if not bundle_root.is_dir():
        raise ValueError("runtime USD bundle root must be one real directory")
    observed_paths = []
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"runtime USD bundle contains symlink: {path}")
        if path.is_file():
            observed_paths.append(path.relative_to(bundle_root).as_posix())
    if sorted(observed_paths) != [entry["path"] for entry in entries]:
        raise ValueError("runtime USD bundle file map differs from the six-file pin")
    for entry in entries:
        payload = (bundle_root / str(entry["path"])).read_bytes()
        if (
            len(payload) != entry["size"]
            or _sha256_bytes(payload) != entry["sha256"]
        ):
            raise ValueError(
                "runtime USD bundle file differs from pin: "
                f"{entry['path']}"
            )
    return {
        "bundle_tree_sha256": tree_sha256,
        "file_count": file_count,
        "files": entries,
        "symlinks_forbidden": True,
        "total_file_bytes": total_file_bytes,
    }


def _artifact(
    source_urdf: Path,
    body_order_source: Path,
    runtime_usd_bundle_root: Path,
) -> dict[str, object]:
    source_urdf = source_urdf.resolve()
    if REPO_ROOT.resolve() not in source_urdf.parents:
        raise ValueError("source URDF must remain inside the tracked repo")
    source_root = source_urdf.parents[1]
    mesh_root = source_root / "meshes"
    order = _body_order(body_order_source.resolve())
    order_set = set(order)
    root = ET.parse(source_urdf).getroot()

    parent: dict[
        str,
        tuple[
            str,
            str,
            tuple[tuple[float, float, float], ...],
            tuple[float, float, float],
        ],
    ] = {}
    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError("joint is missing parent/child")
        rotation, translation = _origin_transform(joint.find("origin"))
        child_name = str(child_element.get("link"))
        if child_name in parent:
            raise ValueError(f"duplicate URDF joint child: {child_name}")
        parent[child_name] = (
            str(parent_element.get("link")),
            str(joint.get("type")),
            rotation,
            translation,
        )

    def runtime_owner(
        link_name: str,
    ) -> tuple[
        str,
        tuple[tuple[float, float, float], ...],
        tuple[float, float, float],
    ]:
        owner = link_name
        rotation = _identity_rotation()
        translation = (0.0, 0.0, 0.0)
        seen: set[str] = set()
        while owner in parent and parent[owner][1] == "fixed":
            if owner in seen:
                raise ValueError("fixed-joint cycle in A3 URDF")
            seen.add(owner)
            parent_name, _, joint_rotation, joint_translation = parent[owner]
            rotation, translation = _compose(
                joint_rotation,
                joint_translation,
                rotation,
                translation,
            )
            owner = parent_name
        if owner not in order_set:
            raise ValueError(
                f"collision link {link_name!r} maps to unknown runtime body {owner!r}"
            )
        return owner, rotation, translation

    components: list[dict[str, object]] = []
    mesh_receipts: dict[str, str] = {}
    owner_counts = {name: 0 for name in order}
    seen_links: set[str] = set()
    for link in root.findall("link"):
        link_name = str(link.get("name"))
        if not link_name or link_name in seen_links:
            raise ValueError(f"duplicate or empty URDF link name: {link_name!r}")
        seen_links.add(link_name)
        owner, owner_from_link_rotation, owner_from_link_translation = (
            runtime_owner(link_name)
        )
        for collision_index, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(
                    f"non-mesh A3 collision is not materialized: {link_name}"
                )
            filename = str(mesh.get("filename"))
            marker = "/meshes/"
            if marker not in filename:
                raise ValueError(f"unexpected A3 collision mesh URI: {filename}")
            relative_mesh = filename.rsplit(marker, 1)[1]
            mesh_path = (mesh_root / relative_mesh).resolve()
            if not mesh_path.is_file() or mesh_root.resolve() not in mesh_path.parents:
                raise ValueError(
                    f"A3 collision mesh escapes or is missing: {relative_mesh}"
                )
            scale = _float_triplet(
                mesh.get("scale"), default=(1.0, 1.0, 1.0)
            )
            if not all(value > 0.0 for value in scale):
                raise ValueError("A3 collision mesh scale must be positive")
            mesh_center, mesh_half = _bounds(
                tuple(
                    float(vertex[axis]) * scale[axis] for axis in range(3)
                )
                for vertex in _stl_vertices(mesh_path)
            )
            link_from_mesh_rotation, link_from_mesh_translation = (
                _origin_transform(collision.find("origin"))
            )
            owner_from_mesh_rotation, owner_from_mesh_translation = _compose(
                owner_from_link_rotation,
                owner_from_link_translation,
                link_from_mesh_rotation,
                link_from_mesh_translation,
            )
            center_owner = _vector_add(
                _matrix_vector(owner_from_mesh_rotation, mesh_center),
                owner_from_mesh_translation,
            )
            # The outer axis dimension contains the three transformed half-axis
            # vectors.  Runtime rotates each vector by the live body quaternion
            # and sums absolute components.
            half_axes_owner = tuple(
                tuple(
                    float(owner_from_mesh_rotation[row][axis])
                    * float(mesh_half[axis])
                    for row in range(3)
                )
                for axis in range(3)
            )
            repo_relative_mesh = mesh_path.relative_to(REPO_ROOT).as_posix()
            mesh_sha = _sha256_bytes(mesh_path.read_bytes())
            mesh_receipts[repo_relative_mesh] = mesh_sha
            component_id = (
                f"{owner}:{link_name}:{collision_index}:{relative_mesh}"
            )
            components.append(
                {
                    "component_id": component_id,
                    "local_center_owner_m": list(center_owner),
                    "local_half_axes_owner_m": [
                        list(axis) for axis in half_axes_owner
                    ],
                    "mesh_path": repo_relative_mesh,
                    "mesh_sha256": mesh_sha,
                    "owner_body_name": owner,
                    "source_link_name": link_name,
                }
            )
            owner_counts[owner] += 1

    if any(count <= 0 for count in owner_counts.values()):
        missing = [name for name, count in owner_counts.items() if count <= 0]
        raise ValueError(f"runtime A3 bodies lack collision components: {missing}")
    components.sort(key=lambda row: str(row["component_id"]))
    content: dict[str, object] = {
        "artifact_type": ARTIFACT_TYPE,
        "body_order": list(order),
        "component_count": len(components),
        "components": components,
        "mesh_receipts": [
            {"path": path, "sha256": mesh_receipts[path]}
            for path in sorted(mesh_receipts)
        ],
        "runtime_usd_bundle": _runtime_usd_binding(
            runtime_usd_bundle_root
        ),
        "schema_version": SCHEMA_VERSION,
        "source_urdf": {
            "path": source_urdf.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_bytes(source_urdf.read_bytes()),
        },
    }
    content["content_sha256"] = _sha256_bytes(_canonical_json_bytes(content))
    return content


def main() -> int:
    args = _parse_args()
    document = _artifact(
        args.source_urdf,
        args.body_order_source,
        args.runtime_usd_bundle_root,
    )
    encoded = _canonical_json_bytes(document) + b"\n"
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_bytes() != encoded:
            print(f"[FAIL] collision proxy artifact differs: {output}")
            return 1
        print(
            "[materialize_a3_table_collision_proxy] OK: "
            f"{document['component_count']} components "
            f"sha256={_sha256_bytes(encoded)}"
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(
        "[materialize_a3_table_collision_proxy] wrote "
        f"{output} components={document['component_count']} "
        f"sha256={_sha256_bytes(encoded)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
