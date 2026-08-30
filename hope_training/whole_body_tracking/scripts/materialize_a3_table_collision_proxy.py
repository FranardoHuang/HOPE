#!/usr/bin/env python3
"""Materialize exact AgiBot A3 collision-mesh multi-OBBs for the table guard.

The Isaac A3 asset is prepared from the tracked vendor URDF by copying every
mesh byte and rewriting only its path.  This tool therefore reads that tracked
source directly, folds fixed-joint collision children into their runtime rigid
body, and emits one or more conservative local OBBs per collision component.

The resulting artifact is data, not a hand-tuned safety margin.  For a split
mesh, the complete backend convex hull is decomposed into a facet-to-interior
tetrahedron fan.  Every complete tetrahedron is assigned to exactly one proxy
OBB, and that OBB contains all four vertices; the OBB union therefore covers
the collision hull, not merely the STL surface.
Runtime may conservatively broaden the rotated OBB to a world AABB, but must
never shrink these materialized half axes.

The tool also refuses to launder an unverified USD cache.  ``--runtime-usd-
bundle-root`` used to be checked only against six hard-coded digests, which
proves the bundle was not edited and proves nothing about which robot it is a
cache of.  ``_plant_identity`` now re-derives IsaacLab's own ``.asset_hash``
from the bundle's converter configuration plus the exact URDF bytes measured
in this run, and carries that configuration into the artifact so any later
reader can redo the derivation without the Pod bundle in hand.
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
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "agi" / "URDF" / "A3P-P1-32dof-0807-OP3-pingpang"
)
DEFAULT_SOURCE_URDF = DEFAULT_SOURCE_ROOT / "urdf" / "model.urdf"
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
    / "a3_table_collision_proxy_a3p0807_20260808"
    / "a3_table_collision_components.v2.json"
)
DEFAULT_MUJOCO_MJCF = (
    REPO_ROOT
    / "agi"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
    / "a3_pingpong.xml"
)
PINNED_MUJOCO_MJCF_SHA256 = (
    "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a"
)
SCHEMA_VERSION = 2
ARTIFACT_TYPE = "a3_table_collision_component_multi_obb_v2"
PINNED_RUNTIME_USD_BUNDLE_TREE_SHA256 = (
    "365ba37edd5e5e1d4fac22f2cbb3ec871ead7bb49aeadb50161ef523a9ae6747"
)
PINNED_RUNTIME_USD_TOTAL_FILE_BYTES = 60519988
PINNED_RUNTIME_USD_FILES = {
    ".asset_hash": (
        "a78a2f8fb207cbf479cc1b308cf9d3c58e1a55eb7da9dbc2caf34be697e9c993",
        32,
    ),
    "config.yaml": (
        "f349c3f4d80a915f5ca3ce53d49785dfd7e6eeca2645dcd7b402d4d8a2288eb9",
        1685,
    ),
    "configuration/model_base.usd": (
        "108a4b45b96a8db8396d3a8feb995481c5db87efcde80066e6347ed494e658fc",
        60504873,
    ),
    "configuration/model_physics.usd": (
        "390cf66cc052ea697e88e9ef0131bf7e2eee96e70c35c0861e1ce33d363747f5",
        11078,
    ),
    "configuration/model_sensor.usd": (
        "4e16201f146db3240b8a0082ae14e3aca41255a75812c5331bf8f4e39701355c",
        687,
    ),
    "model.usd": (
        "13e5ecfe02238fbf1d20c13ed7177e18ed93d84bca8e0a592b6605f7fb85f351",
        1633,
    ),
}

##
# Plant identity for the USD bundle this artifact is allowed to describe.
#
# Until 2026-08-08 the block above was the whole story, and the whole story was
# not enough.  Six SHA-256 values prove "nobody edited these bytes".  They do
# not prove "these bytes are a conversion of the robot this artifact measures",
# and that gap was live: the pinned bundle was the retired 0409 robot's cache,
# this producer hashed a URDF two hundred lines later and never compared the
# two, and the resulting tracked artifact was believed by both engines.
#
# The three names below are compared, and then ``PINNED_ISAACLAB_ASSET_HASH``
# is RE-DERIVED: IsaacLab's own ``.asset_hash`` recipe, run offline over the
# bundle's converter configuration plus the exact URDF bytes this run measured
# geometry from.  Names can be doctored; that derivation cannot.  A bundle
# converted from any other robot fails it.
##
PLANT_RECEIPT_RELATIVE = "configs/a3p_p1_0807_model_set_v1.json"
PLANT_RECEIPT_MANIFEST_TYPE = "a3p_p1_0807_dual_engine_model_set_v1"
PLANT_ASSET_ROOT_NAME = "agibot_a3p_p1_0807_v1"
PINNED_SOURCE_URDF_SHA256 = (
    "15c83f5f3beea71350583143aef4d622d5219df65a0bed9a660a0edb7d388d09"
)
PINNED_ISAACLAB_ASSET_HASH = "676efde5febed3c0fde0f2ad59650cdf"
# isaaclab/sim/converters/asset_converter_base.py::_config_to_hash drops these
# three path keys before hashing the converter configuration.
ASSET_HASH_EXCLUDED_CONFIG_KEYS = ("asset_path", "usd_dir", "usd_file_name")
PLANT_IDENTITY_KIND = "a3_collision_proxy_plant_identity_v2"
# The hand+racket support mesh is non-convex.  Its old single AABB contained a
# large empty corner that crossed the 20 mm table keep-out even though the raw
# mesh was 72 mm clear. A deterministic two-way convex-hull tetra partition
# with PCA-oriented leaf boxes is the smallest split that removes that measured
# false positive with >5 mm reserve (26.49 mm before the outward pad in the
# frozen 2026-08-30 witness). Every other source component deliberately stays
# one box, keeping the runtime row count at 63.
MULTI_OBB_LEAF_COUNTS = {"right_hand_pingpang_link.stl": 2}
TRIANGLE_PARTITION_ALGORITHM = (
    "convex_hull_tetra_fan_recursive_centroid_median_pca_obb_v1"
)
PINNED_HULL_NUMPY_VERSION = "1.26.4"
PINNED_HULL_SCIPY_VERSION = "1.11.4"
PINNED_HULL_QHULL_OPTIONS = "Qt"
PCA_EIGENVALUE_TIE_RTOL = 1.0e-10
PROXY_OBB_OUTWARD_PAD_M = 1.0e-6
MUJOCO_COLLISION_SOURCE_GROUPS = {
    "right_wrist_yaw_collision": "right_wrist_yaw_link.stl",
    "right_hand_palm_collision": "right_hand_pingpang_link.stl",
    "right_hand_finger_collision": "right_hand_pingpang_link.stl",
    "right_hand_thumb_collision": "right_hand_pingpang_link.stl",
    "right_racket_collision": "right_hand_pingpang_link.stl",
    "right_racket_handle_collision": "right_hand_pingpang_link.stl",
}
# The 20 OmniPicker3 left-gripper collision links.  They enter the table guard
# for the first time with the 0807 plant, and a later "cleanup" that quietly
# drops them would silently re-open the volume they occupy.  Naming them here
# makes that deletion a refusal instead of a smaller number.
LEFT_GRIPPER_SOURCE_LINKS = (
    "left_base_link",
    "left_link1",
    "left_link10",
    "left_link11",
    "left_link11-1",
    "left_link13",
    "left_link14",
    "left_link14-1",
    "left_link15",
    "left_link17",
    "left_link18",
    "left_link2",
    "left_link3",
    "left_link4",
    "left_link4-1",
    "left_link6",
    "left_link7",
    "left_link7-1",
    "left_link8",
    "left_link9",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument(
        "--body-order-source", type=Path, default=DEFAULT_BODY_ORDER_SOURCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mujoco-mjcf",
        type=Path,
        default=DEFAULT_MUJOCO_MJCF,
        help=(
            "Tracked canonical A3 MJCF whose actual wrist collision meshes "
            "and analytic primitives must also fit inside the shared proxy."
        ),
    )
    parser.add_argument(
        "--runtime-usd-bundle-root",
        type=Path,
        required=True,
        help=(
            "Reviewed six-file Pod USD root.  Formal generation and --check "
            "both revalidate every byte, the canonical tree digest, and the "
            "IsaacLab derivation proof tying the cache to the measured URDF."
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


def _stl_triangles(
    path: Path,
) -> list[tuple[tuple[float, float, float], ...]]:
    payload = path.read_bytes()
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + triangle_count * 50 == len(payload):
            for triangle in range(triangle_count):
                base = 84 + triangle * 50 + 12
                vertices = []
                for vertex in range(3):
                    values = struct.unpack_from(
                        "<fff", payload, base + vertex * 12
                    )
                    vertices.append(tuple(float(value) for value in values))
                triangles.append(tuple(vertices))
    if not triangles:
        text = payload.decode("ascii")
        vertices = []
        for line in text.splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in fields[1:]))
        if len(vertices) % 3:
            raise ValueError(f"ASCII STL has incomplete triangle: {path}")
        triangles = [
            tuple(vertices[index : index + 3])
            for index in range(0, len(vertices), 3)
        ]
    if not triangles or not all(
        math.isfinite(value)
        for triangle in triangles
        for vertex in triangle
        for value in vertex
    ):
        raise ValueError(f"STL has no finite triangles: {path}")
    return triangles


def _mujoco_actual_wrist_collision_primitives(
    mjcf_path: Path,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Read the exact live MuJoCo wrist collision inventory.

    The canonical MuJoCo plant does not collide the visual hand/racket STL.
    It uses one optimized wrist mesh, three analytic hand primitives, one
    optimized racket-face mesh, and one analytic handle capsule.  These shapes
    are therefore part of the proxy's coverage authority, not optional
    diagnostics.
    """

    mjcf_path = mjcf_path.resolve(strict=True)
    if REPO_ROOT.resolve() not in mjcf_path.parents:
        raise ValueError("canonical MuJoCo MJCF must remain inside the repo")
    mjcf_bytes = mjcf_path.read_bytes()
    if _sha256_bytes(mjcf_bytes) != PINNED_MUJOCO_MJCF_SHA256:
        raise ValueError("canonical MuJoCo MJCF differs from the reviewed pin")
    root = ET.fromstring(mjcf_bytes)
    compiler = root.find("compiler")
    if compiler is None or compiler.get("convexhull") not in (None, "true"):
        raise ValueError("canonical MuJoCo mesh convex-hull semantics differ")
    meshdir = str(compiler.get("meshdir") or "")
    if not meshdir:
        raise ValueError("canonical MuJoCo MJCF has no meshdir")
    mesh_root = (mjcf_path.parent / meshdir).resolve(strict=True)
    if not mesh_root.is_dir() or mjcf_path.parent.resolve() not in mesh_root.parents:
        raise ValueError("canonical MuJoCo meshdir escapes its model root")
    mesh_assets = {}
    for row in root.findall("asset/mesh"):
        name = str(row.get("name") or "")
        if not name or name in mesh_assets:
            raise ValueError("canonical MuJoCo mesh names are malformed")
        mesh_assets[name] = row
    wrist_bodies = [
        row
        for row in root.iter("body")
        if row.get("name") == "right_wrist_yaw_Link"
    ]
    if len(wrist_bodies) != 1:
        raise ValueError("canonical MuJoCo wrist body inventory differs")
    collision_geoms = {
        str(row.get("name")): row
        for row in wrist_bodies[0].findall("geom")
        if row.get("class") == "collision"
    }
    if set(collision_geoms) != set(MUJOCO_COLLISION_SOURCE_GROUPS):
        raise ValueError("canonical MuJoCo wrist collision inventory differs")

    expected_kinds = {
        "right_wrist_yaw_collision": "mesh",
        "right_hand_palm_collision": "ellipsoid",
        "right_hand_finger_collision": "capsule",
        "right_hand_thumb_collision": "capsule",
        "right_racket_collision": "mesh",
        "right_racket_handle_collision": "capsule",
    }
    grouped = {
        source_name: []
        for source_name in sorted(set(MUJOCO_COLLISION_SOURCE_GROUPS.values()))
    }
    receipts = []
    for name in sorted(collision_geoms):
        row = collision_geoms[name]
        kind = str(row.get("type") or "")
        if kind != expected_kinds[name]:
            raise ValueError(f"canonical MuJoCo collider type differs: {name}")
        if any(row.get(field) is not None for field in ("quat", "euler", "axisangle")):
            raise ValueError(f"rotated MuJoCo target collider is unsupported: {name}")
        position = _float_triplet(
            row.get("pos"), default=(0.0, 0.0, 0.0)
        )
        primitive: dict[str, object] = {
            "kind": kind,
            "name": name,
        }
        receipt: dict[str, object] = {
            "kind": kind,
            "name": name,
            "position_owner_m": list(position),
            "source_component_mesh": MUJOCO_COLLISION_SOURCE_GROUPS[name],
        }
        if kind == "mesh":
            mesh_name = str(row.get("mesh") or "")
            mesh_asset = mesh_assets.get(mesh_name)
            if mesh_asset is None:
                raise ValueError(f"MuJoCo collider mesh asset is missing: {name}")
            filename = str(mesh_asset.get("file") or "")
            mesh_path = (mesh_root / filename).resolve(strict=True)
            if mesh_root not in mesh_path.parents or not mesh_path.is_file():
                raise ValueError(f"MuJoCo collider mesh escapes meshdir: {name}")
            scale = _float_triplet(
                mesh_asset.get("scale"), default=(1.0, 1.0, 1.0)
            )
            if not all(value > 0.0 for value in scale):
                raise ValueError("MuJoCo collision mesh scale must be positive")
            triangles = _stl_triangles(mesh_path)
            vertices = tuple(
                sorted(
                    {
                        tuple(
                            float(vertex[axis]) * scale[axis] + position[axis]
                            for axis in range(3)
                        )
                        for triangle in triangles
                        for vertex in triangle
                    }
                )
            )
            primitive["vertices"] = vertices
            receipt.update(
                {
                    "mesh_path": mesh_path.relative_to(REPO_ROOT).as_posix(),
                    "mesh_sha256": _sha256_bytes(mesh_path.read_bytes()),
                    "mesh_scale": list(scale),
                    "source_triangle_count": len(triangles),
                    "unique_vertex_count": len(vertices),
                    "vertices_owner_sha256": _sha256_bytes(
                        _canonical_json_bytes(vertices)
                    ),
                }
            )
        elif kind == "ellipsoid":
            radii = _float_triplet(row.get("size"), default=())
            if not all(value > 0.0 for value in radii):
                raise ValueError("MuJoCo ellipsoid radii must be positive")
            primitive.update({"center": position, "radii": radii})
            receipt["radii_m"] = list(radii)
        else:
            values = tuple(
                float(value) for value in str(row.get("fromto") or "").split()
            )
            size = tuple(
                float(value) for value in str(row.get("size") or "").split()
            )
            if (
                len(values) != 6
                or len(size) != 1
                or not all(math.isfinite(value) for value in (*values, *size))
                or size[0] <= 0.0
            ):
                raise ValueError("MuJoCo capsule geometry is malformed")
            start = values[:3]
            end = values[3:]
            primitive.update(
                {"end": end, "radius": size[0], "start": start}
            )
            receipt.update(
                {
                    "end_owner_m": list(end),
                    "radius_m": size[0],
                    "start_owner_m": list(start),
                }
            )
        grouped[MUJOCO_COLLISION_SOURCE_GROUPS[name]].append(primitive)
        receipts.append(receipt)
    binding = {
        "collision_semantics": "mesh_convex_hull_plus_analytic_primitives",
        "mjcf_path": mjcf_path.relative_to(REPO_ROOT).as_posix(),
        "mjcf_sha256": _sha256_bytes(mjcf_bytes),
        "target_colliders": receipts,
    }
    binding["content_sha256"] = _sha256_bytes(_canonical_json_bytes(binding))
    return grouped, binding


def _primitive_projection_interval(
    primitive: Mapping[str, object],
    unit_axes: Any,
) -> tuple[Any, Any]:
    """Return exact min/max support on each OBB axis for one Mu primitive."""

    import numpy as np

    kind = primitive["kind"]
    if kind == "mesh":
        projected = np.asarray(primitive["vertices"], dtype=np.float64) @ unit_axes.T
        return projected.min(axis=0), projected.max(axis=0)
    if kind == "ellipsoid":
        center = np.asarray(primitive["center"], dtype=np.float64)
        radii = np.asarray(primitive["radii"], dtype=np.float64)
        middle = unit_axes @ center
        support = np.sqrt(np.sum((unit_axes * radii) ** 2, axis=1))
        return middle - support, middle + support
    start = np.asarray(primitive["start"], dtype=np.float64)
    end = np.asarray(primitive["end"], dtype=np.float64)
    radius = float(primitive["radius"])
    first = unit_axes @ start
    second = unit_axes @ end
    return (
        np.minimum(first, second) - radius,
        np.maximum(first, second) + radius,
    )


def _convex_hull_tetrahedra(
    triangles: Sequence[Sequence[Sequence[float]]],
) -> tuple[list[tuple[tuple[float, float, float], ...]], dict[str, object]]:
    """Return a deterministic complete tetra fan for the backend convex hull.

    Isaac's ``convexHull`` mesh approximation and MuJoCo's mesh collision both
    consume the convex hull, not merely the STL surface.  Joining every
    canonical triangular hull facet to one strictly interior point partitions
    that full convex body into tetrahedra (up to shared zero-volume faces).
    """

    import numpy as np
    import scipy
    from scipy.spatial import ConvexHull

    if (
        np.__version__ != PINNED_HULL_NUMPY_VERSION
        or scipy.__version__ != PINNED_HULL_SCIPY_VERSION
    ):
        raise ValueError(
            "convex-hull materialization requires the pinned NumPy/SciPy "
            f"toolchain {PINNED_HULL_NUMPY_VERSION}/"
            f"{PINNED_HULL_SCIPY_VERSION}, got "
            f"{np.__version__}/{scipy.__version__}"
        )
    points = np.unique(
        np.asarray(
            [vertex for triangle in triangles for vertex in triangle],
            dtype=np.float64,
        ),
        axis=0,
    )
    hull = ConvexHull(points, qhull_options=PINNED_HULL_QHULL_OPTIONS)
    facets = tuple(
        sorted(tuple(sorted(int(index) for index in row)) for row in hull.simplices)
    )
    interior = points[np.sort(hull.vertices)].mean(axis=0)
    signed_interior = hull.equations[:, :3] @ interior + hull.equations[:, 3]
    if not bool(np.all(signed_interior < 0.0)):
        raise ValueError("convex-hull fan point is not strictly interior")
    tetrahedra = [
        tuple(
            tuple(float(value) for value in vertex)
            for vertex in (interior, *(points[index] for index in facet))
        )
        for facet in facets
    ]
    tetra_volume = 0.0
    for tetrahedron in tetrahedra:
        vertices = np.asarray(tetrahedron, dtype=np.float64)
        tetra_volume += abs(
            float(
                np.linalg.det(
                    np.stack(
                        (
                            vertices[1] - vertices[0],
                            vertices[2] - vertices[0],
                            vertices[3] - vertices[0],
                        ),
                        axis=1,
                    )
                )
            )
        ) / 6.0
    if not math.isclose(
        tetra_volume,
        float(hull.volume),
        rel_tol=1.0e-10,
        abs_tol=1.0e-15,
    ):
        raise ValueError("convex-hull tetra fan does not conserve hull volume")
    hull_vertices = tuple(
        tuple(float(value) for value in points[index])
        for index in sorted(int(index) for index in hull.vertices)
    )
    hull_digest_payload = {
        "facets": facets,
        "vertices_m": hull_vertices,
    }
    receipt = {
        "facet_count": len(facets),
        "facets_sha256": _sha256_bytes(_canonical_json_bytes(facets)),
        "hull_geometry_sha256": _sha256_bytes(
            _canonical_json_bytes(hull_digest_payload)
        ),
        "hull_vertex_count": int(len(hull.vertices)),
        "hull_vertices_sha256": _sha256_bytes(
            _canonical_json_bytes(hull_vertices)
        ),
        "hull_volume_m3": float(hull.volume),
        "interior_point_m": [float(value) for value in interior],
        "interior_strict_max_plane_value_m": float(np.max(signed_interior)),
        "numpy_version": np.__version__,
        "qhull_options": PINNED_HULL_QHULL_OPTIONS,
        "scipy_version": scipy.__version__,
        "tetra_count": len(tetrahedra),
        "tetra_fan_volume_m3": tetra_volume,
        "tetra_fan_volume_abs_error_m3": abs(
            tetra_volume - float(hull.volume)
        ),
        "tetrahedra_sha256": _sha256_bytes(
            _canonical_json_bytes(tetrahedra)
        ),
        "unique_source_vertex_count": int(points.shape[0]),
    }
    return tetrahedra, receipt


def _partition_simplices(
    simplices: Sequence[Sequence[Sequence[float]]], leaf_count: int
) -> list[tuple[int, ...]]:
    """Partition complete simplices by a deterministic spatial median tree."""

    if (
        isinstance(leaf_count, bool)
        or not isinstance(leaf_count, int)
        or leaf_count <= 0
        or leaf_count > len(simplices)
    ):
        raise ValueError("multi-OBB leaf count must fit the simplex count")
    centroids = tuple(
        tuple(
            sum(float(vertex[axis]) for vertex in simplex) / len(simplex)
            for axis in range(3)
        )
        for simplex in simplices
    )

    def split(indices: tuple[int, ...], leaves: int) -> list[tuple[int, ...]]:
        if leaves == 1:
            return [indices]
        if len(indices) < leaves:
            raise ValueError("simplex partition would create an empty leaf")
        spans = tuple(
            max(centroids[index][axis] for index in indices)
            - min(centroids[index][axis] for index in indices)
            for axis in range(3)
        )
        # ``max`` returns the first equal item, making X/Y/Z tie-breaking part
        # of the versioned algorithm rather than a platform accident.
        axis = max(range(3), key=lambda candidate: spans[candidate])
        ordered = tuple(
            sorted(indices, key=lambda index: (centroids[index][axis], index))
        )
        left_leaves = leaves // 2
        right_leaves = leaves - left_leaves
        midpoint = (len(ordered) * left_leaves) // leaves
        midpoint = max(left_leaves, min(midpoint, len(ordered) - right_leaves))
        return split(ordered[:midpoint], left_leaves) + split(
            ordered[midpoint:], right_leaves
        )

    leaves = split(tuple(range(len(simplices))), leaf_count)
    flattened = [index for leaf in leaves for index in leaf]
    if sorted(flattened) != list(range(len(simplices))):
        raise ValueError("simplex partition does not cover each simplex exactly once")
    return leaves


def _canonical_pca_basis(
    covariance: Any,
) -> tuple[Any, tuple[float, float, float], tuple[int, ...]]:
    """Return a platform-independent basis for a symmetric covariance.

    Eigenvector signs are arbitrary, and an eigensolver may return any basis
    inside a repeated-eigenvalue subspace.  We remove both freedoms: values
    are ordered descending with source-index tie breaking, near-equal values
    form one eigenspace, and that space is reconstructed by projecting the
    fixed X/Y/Z axes followed by deterministic Gram--Schmidt.  The first two
    axes then receive a largest-component-positive sign and the third is their
    right-handed cross product.
    """

    import numpy as np

    eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(covariance)
    order = tuple(
        sorted(
            range(3),
            key=lambda axis: (-float(eigenvalues_raw[axis]), axis),
        )
    )
    eigenvalues = tuple(float(eigenvalues_raw[axis]) for axis in order)
    scale = max(max(abs(value) for value in eigenvalues), 1.0e-30)
    groups: list[tuple[int, ...]] = []
    current = [0]
    for rank in range(1, 3):
        if abs(eigenvalues[rank] - eigenvalues[current[-1]]) <= (
            PCA_EIGENVALUE_TIE_RTOL * scale
        ):
            current.append(rank)
        else:
            groups.append(tuple(current))
            current = [rank]
    groups.append(tuple(current))

    ordered_vectors = eigenvectors_raw[:, list(order)]
    columns = []
    canonical_seeds = np.eye(3, dtype=np.float64)
    for group in groups:
        raw_space = ordered_vectors[:, list(group)]
        projector = raw_space @ raw_space.T
        group_columns = []
        for seed in canonical_seeds:
            candidate = projector @ seed
            for existing in group_columns:
                candidate -= existing * float(existing @ candidate)
            norm = float(np.linalg.norm(candidate))
            if norm > 1.0e-12:
                group_columns.append(candidate / norm)
            if len(group_columns) == len(group):
                break
        if len(group_columns) != len(group):
            raise ValueError("cannot canonicalize a PCA eigenspace")
        columns.extend(group_columns)
    basis = np.stack(columns, axis=1)
    for axis in range(2):
        pivot = int(np.argmax(np.abs(basis[:, axis])))
        if basis[pivot, axis] < 0.0:
            basis[:, axis] *= -1.0
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    basis[:, 2] /= np.linalg.norm(basis[:, 2])
    if not np.allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=1.0e-12):
        raise ValueError("canonical PCA basis is not orthonormal")
    if not math.isclose(float(np.linalg.det(basis)), 1.0, abs_tol=1.0e-12):
        raise ValueError("canonical PCA basis is not right handed")
    return basis, eigenvalues, tuple(len(group) for group in groups)


def _float32_max_abs_obb_coefficient(
    center: Any,
    half_axes: Any,
    vertices: Any,
) -> float:
    """Prove that the float32 geometry still contains every supplied vertex."""

    import numpy as np

    center_f32 = np.asarray(center, dtype=np.float32).astype(np.float64)
    axes_f32 = np.asarray(half_axes, dtype=np.float32).astype(np.float64)
    vertices_f64 = np.asarray(vertices, dtype=np.float64)
    coefficients = np.linalg.solve(
        axes_f32.T,
        (vertices_f64 - center_f32).T,
    ).T
    maximum = float(np.max(np.abs(coefficients)))
    if maximum > 1.0:
        raise ValueError("float32 proxy OBB shrinks below its coverage vertices")
    return maximum


def _partition_pca_obbs(
    simplices: Sequence[Sequence[Sequence[float]]],
    leaves: Sequence[Sequence[int]],
) -> tuple[
    list[
        tuple[
            tuple[float, float, float],
            tuple[tuple[float, float, float], ...],
        ]
    ],
    list[dict[str, object]],
]:
    """Fit a pinned PCA OBB around every complete-simplex leaf."""

    import numpy as np

    result = []
    receipts = []
    for leaf in leaves:
        vertices = np.unique(
            np.asarray(
                [
                    simplices[index][vertex]
                    for index in leaf
                    for vertex in range(len(simplices[index]))
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        mean = vertices.mean(axis=0)
        centered = vertices - mean
        covariance = centered.T @ centered / float(vertices.shape[0])
        basis, eigenvalues, tie_group_sizes = _canonical_pca_basis(covariance)
        projected = centered @ basis
        lower = projected.min(axis=0)
        upper = projected.max(axis=0)
        center = mean + basis @ ((lower + upper) * 0.5)
        half = (upper - lower) * 0.5 + PROXY_OBB_OUTWARD_PAD_M
        half_axes = np.stack(
            [basis[:, axis] * half[axis] for axis in range(3)], axis=0
        )
        for vertex in vertices:
            relative = vertex - center
            projection = np.abs(basis.T @ relative)
            if bool(np.any(projection > half)):
                raise ValueError("multi-OBB leaf failed full-simplex coverage")
        # Consumers use float32 on the GPU.  Validate the serialized geometry
        # in that precision as well; the 1 um outward pad must dominate all
        # center/basis conversion roundoff rather than relying on luck.
        max_abs_coefficient = _float32_max_abs_obb_coefficient(
            center, half_axes, vertices
        )
        result.append(
            (
                tuple(float(value) for value in center),
                tuple(
                    tuple(float(value) for value in axis)
                    for axis in half_axes
                ),
            )
        )
        receipts.append(
            {
                "basis_sha256": _sha256_bytes(
                    _canonical_json_bytes(
                        [[float(value) for value in row] for row in basis]
                    )
                ),
                "eigenvalue_tie_group_sizes": list(tie_group_sizes),
                "eigenvalues_m2": list(eigenvalues),
                "float32_max_abs_obb_coefficient": max_abs_coefficient,
                "unique_vertex_count": int(vertices.shape[0]),
            }
        )
    return result, receipts


def _expand_obbs_for_mujoco_primitives(
    obbs: Sequence[
        tuple[
            Sequence[float],
            Sequence[Sequence[float]],
        ]
    ],
    primitives: Sequence[Mapping[str, object]],
) -> tuple[
    list[
        tuple[
            tuple[float, float, float],
            tuple[tuple[float, float, float], ...],
        ]
    ],
    dict[str, object],
]:
    """Assign complete Mu colliders to leaves and expand by exact support.

    Assignment is exhaustive (the target has five primitives and two leaves),
    minimizes summed OBB volume, then breaks equal-volume ties by the
    lexicographic assignment tuple.  Every analytic primitive is assigned as a
    whole object; no surface sampling can create a false coverage proof.
    """

    import itertools
    import numpy as np

    if not obbs or len(obbs) > 8 or len(primitives) > 12:
        raise ValueError("MuJoCo target primitive assignment width is unsupported")
    base_frames = []
    for center_raw, axes_raw in obbs:
        center = np.asarray(center_raw, dtype=np.float64)
        axes = np.asarray(axes_raw, dtype=np.float64)
        half = np.linalg.norm(axes, axis=1)
        unit = axes / half[:, None]
        middle = unit @ center
        base_frames.append((unit, middle - half, middle + half))

    def fit(leaf_index: int, selected: Sequence[Mapping[str, object]]):
        unit, base_lower, base_upper = base_frames[leaf_index]
        lower = base_lower.copy()
        upper = base_upper.copy()
        for primitive in selected:
            primitive_lower, primitive_upper = _primitive_projection_interval(
                primitive, unit
            )
            lower = np.minimum(lower, primitive_lower)
            upper = np.maximum(upper, primitive_upper)
        middle = (lower + upper) * 0.5
        half = (upper - lower) * 0.5 + PROXY_OBB_OUTWARD_PAD_M
        center = unit.T @ middle
        axes = unit * half[:, None]
        return center, axes, float(np.prod(2.0 * half))

    best = None
    for assignment in itertools.product(
        range(len(obbs)), repeat=len(primitives)
    ):
        fitted = tuple(
            fit(
                leaf,
                [
                    primitive
                    for primitive, assigned in zip(primitives, assignment)
                    if assigned == leaf
                ],
            )
            for leaf in range(len(obbs))
        )
        candidate = (sum(row[2] for row in fitted), assignment, fitted)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("MuJoCo primitive assignment produced no candidate")
    _objective, assignment, fitted = best

    expanded = []
    max_interval_excess_f32 = float("-inf")
    for leaf, (center, axes, _volume) in enumerate(fitted):
        center_f32 = center.astype(np.float32).astype(np.float64)
        axes_f32 = axes.astype(np.float32).astype(np.float64)
        half_f32 = np.linalg.norm(axes_f32, axis=1)
        unit_f32 = axes_f32 / half_f32[:, None]
        middle_f32 = unit_f32 @ center_f32
        for primitive, assigned in zip(primitives, assignment):
            if assigned != leaf:
                continue
            lower, upper = _primitive_projection_interval(
                primitive, unit_f32
            )
            excess = max(
                float(np.max((middle_f32 - half_f32) - lower)),
                float(np.max(upper - (middle_f32 + half_f32))),
            )
            max_interval_excess_f32 = max(max_interval_excess_f32, excess)
            if excess > 0.0:
                raise ValueError(
                    "float32 proxy OBB does not contain its complete MuJoCo "
                    f"primitive: {primitive['name']}"
                )
        expanded.append(
            (
                tuple(float(value) for value in center),
                tuple(
                    tuple(float(value) for value in axis)
                    for axis in axes
                ),
            )
        )
    receipt = {
        "assignment_algorithm": (
            "exhaustive_min_sum_full_obb_volume_then_lexicographic_v1"
        ),
        "leaf_by_primitive": [
            {"leaf_index": assigned, "name": primitive["name"]}
            for primitive, assigned in zip(primitives, assignment)
        ],
        "max_float32_projection_interval_excess_m": (
            max_interval_excess_f32
        ),
        "objective_sum_full_obb_volume_m3": best[0],
    }
    receipt["content_sha256"] = _sha256_bytes(
        _canonical_json_bytes(receipt)
    )
    return expanded, receipt


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


def _recompute_isaaclab_asset_hash(
    config: Mapping[str, Any], urdf_path: Path
) -> str:
    """Redo IsaacLab's ``.asset_hash`` offline, without importing Isaac.

    Byte-compatible on purpose with
    ``isaaclab/sim/converters/asset_converter_base.py::_config_to_hash``: MD5
    over ``json.dumps`` of the converter configuration with the three path keys
    removed, then over the source asset file in 64 KiB chunks.  Reproducing it
    here is what lets this producer say "that USD cache came out of THIS URDF"
    while it still has the URDF open.
    """

    payload = dict(config)
    for key in ASSET_HASH_EXCLUDED_CONFIG_KEYS:
        payload.pop(key, None)
    digest = hashlib.md5()
    digest.update(json.dumps(payload).encode())
    with open(urdf_path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _plant_identity(
    source_urdf: Path,
    source_urdf_sha256: str,
    mesh_receipts: Mapping[str, str],
    bundle_root: Path,
) -> dict[str, object]:
    """Prove the pinned USD bundle is a conversion OF the URDF measured here.

    Everything before step 5 compares names and digests that a determined
    editor could restate.  Step 5 compares bytes to bytes through IsaacLab's
    own hash and is the reason this block is an identity proof rather than a
    second opinion about file integrity.  The converter configuration is
    carried into the artifact verbatim so a consumer with no access to the Pod
    bundle -- the MuJoCo lane, or a reviewer on a laptop -- can redo step 5
    from the repository alone.
    """

    import yaml  # local: the geometry half of this tool has no YAML dependency

    receipt_path = REPO_ROOT / PLANT_RECEIPT_RELATIVE
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))

    # 1. The checkout's own plant receipt, and the asset package it names.
    if receipt.get("manifest_type") != PLANT_RECEIPT_MANIFEST_TYPE:
        raise ValueError(
            "A3 plant receipt is not the reviewed dual-engine model set: "
            f"manifest_type={receipt.get('manifest_type')!r}"
        )
    isaac = receipt.get("isaac")
    if not isinstance(isaac, dict):
        raise ValueError("A3 plant receipt carries no isaac section")
    declared_asset = str(isaac.get("asset_path") or "")
    declared_urdf = str(isaac.get("urdf_path") or "")
    declared_sha = str(isaac.get("urdf_sha256") or "")
    if declared_asset.rsplit("/", 1)[-1] != PLANT_ASSET_ROOT_NAME:
        raise ValueError(
            "A3 plant receipt names a different asset package than the pin: "
            f"{declared_asset!r}"
        )
    if declared_sha != PINNED_SOURCE_URDF_SHA256:
        raise ValueError(
            "A3 plant moved without re-cutting the collision proxy: receipt "
            f"URDF sha256={declared_sha} but the pin is "
            f"{PINNED_SOURCE_URDF_SHA256}"
        )

    # 2. The URDF this run actually measured geometry from is that same URDF.
    if source_urdf_sha256 != PINNED_SOURCE_URDF_SHA256:
        raise ValueError(
            "collision proxy source URDF is not the pinned plant URDF: "
            f"{source_urdf_sha256} != {PINNED_SOURCE_URDF_SHA256}"
        )

    # 3. Every collision mesh used here is byte-identical to the same-named
    #    file in the receipt's asset closure.  ``.asset_hash`` covers the URDF
    #    text only, so without this the meshes would be unbound.
    closure = {
        str(row["path"]): str(row["sha256"])
        for row in isaac["closure"]["files"]
    }
    source_root = source_urdf.parents[1]
    checked = 0
    for repo_relative_mesh, mesh_sha in sorted(mesh_receipts.items()):
        relative = (
            (REPO_ROOT / repo_relative_mesh).relative_to(source_root).as_posix()
        )
        if closure.get(relative) != mesh_sha:
            raise ValueError(
                "collision mesh is absent from or differs inside the A3 plant "
                f"receipt closure: {relative}"
            )
        checked += 1

    # 4. What the converter itself recorded about the file it read.
    config_bytes = (bundle_root / "config.yaml").read_bytes()
    config_text = config_bytes.decode("ascii")
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError("A3 runtime USD config.yaml is not a mapping")
    recorded_asset_path = str(config.get("asset_path") or "")
    source_relative = f"{declared_asset}/{declared_urdf}"
    if not recorded_asset_path.endswith(f"/{source_relative}"):
        raise ValueError(
            "A3 runtime USD bundle was converted from a different robot: "
            f"config.yaml asset_path={recorded_asset_path} is not "
            f"{source_relative}"
        )

    # 5. The derivation proof.
    stored_asset_hash = (
        (bundle_root / ".asset_hash").read_text(encoding="ascii").strip()
    )
    recomputed = _recompute_isaaclab_asset_hash(config, source_urdf)
    if recomputed != stored_asset_hash:
        raise ValueError(
            "A3 runtime USD cache was not converted from the URDF this proxy "
            f"measures: IsaacLab asset hash recomputes to {recomputed} but the "
            f"bundle stores {stored_asset_hash}"
        )
    if stored_asset_hash != PINNED_ISAACLAB_ASSET_HASH:
        raise ValueError(
            "A3 runtime USD .asset_hash differs from the reviewed pin: "
            f"{stored_asset_hash} != {PINNED_ISAACLAB_ASSET_HASH}"
        )

    return {
        "compared": [
            "plant_receipt_manifest_type",
            "plant_receipt_asset_root_vs_pin",
            "plant_receipt_urdf_sha256_vs_pin",
            "measured_source_urdf_sha256_vs_pin",
            "measured_collision_meshes_vs_plant_receipt_closure",
            "bundle_config_asset_path_vs_plant_receipt",
            "bundle_isaaclab_asset_hash_vs_rederived_from_measured_urdf",
        ],
        "converter_config_asset_path": recorded_asset_path,
        "converter_config_sha256": _sha256_bytes(config_bytes),
        "converter_config_yaml": config_text,
        "isaaclab_asset_hash": stored_asset_hash,
        "isaaclab_asset_hash_excluded_config_keys": list(
            ASSET_HASH_EXCLUDED_CONFIG_KEYS
        ),
        "kind": PLANT_IDENTITY_KIND,
        "mesh_closure_files_checked": checked,
        "plant_asset_root_name": PLANT_ASSET_ROOT_NAME,
        "plant_receipt_manifest_type": PLANT_RECEIPT_MANIFEST_TYPE,
        "plant_receipt_path": PLANT_RECEIPT_RELATIVE,
        "plant_receipt_sha256": _sha256_bytes(receipt_bytes),
    }


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
        or total_file_bytes != PINNED_RUNTIME_USD_TOTAL_FILE_BYTES
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
    mujoco_mjcf: Path,
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
    mujoco_primitives_by_source, mujoco_collision_binding = (
        _mujoco_actual_wrist_collision_primitives(mujoco_mjcf)
    )
    consumed_mujoco_sources: set[str] = set()

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
    source_component_count = 0
    partition_receipts: list[dict[str, object]] = []
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
            triangles = [
                tuple(
                    tuple(
                        float(vertex[axis]) * scale[axis]
                        for axis in range(3)
                    )
                    for vertex in triangle
                )
                for triangle in _stl_triangles(mesh_path)
            ]
            leaf_count = MULTI_OBB_LEAF_COUNTS.get(relative_mesh.lower(), 1)
            source_vertices = tuple(
                sorted(
                    {
                        tuple(float(value) for value in vertex)
                        for triangle in triangles
                        for vertex in triangle
                    }
                )
            )
            source_vertices_sha256 = _sha256_bytes(
                _canonical_json_bytes(source_vertices)
            )
            mujoco_primitives = mujoco_primitives_by_source.get(
                relative_mesh.lower(), []
            )
            hull_receipt: dict[str, object] | None = None
            mujoco_cover_receipt: dict[str, object] | None = None
            if leaf_count == 1:
                mesh_center, mesh_half = _bounds(source_vertices)
                mesh_half = tuple(
                    float(value) + PROXY_OBB_OUTWARD_PAD_M
                    for value in mesh_half
                )
                leaf_obbs = [
                    (
                        mesh_center,
                        (
                            (mesh_half[0], 0.0, 0.0),
                            (0.0, mesh_half[1], 0.0),
                            (0.0, 0.0, mesh_half[2]),
                        ),
                    )
                ]
                # A convex box containing every source vertex contains the
                # Isaac convex-hull collider.  MuJoCo's canonical plant does
                # not use the same collision inventory; its exact mesh and
                # analytic primitives are incorporated below before this row
                # is allowed into the shared guard artifact.
                leaf_primitive_indices = [tuple(range(len(source_vertices)))]
                leaf_fit_receipts = [
                    {
                        "float32_max_abs_obb_coefficient": (
                            _float32_max_abs_obb_coefficient(
                                mesh_center,
                                leaf_obbs[0][1],
                                source_vertices,
                            )
                        ),
                        "unique_vertex_count": len(source_vertices),
                    }
                ]
                coverage_basis = "source_vertex_aabb_convex_hull_superset"
                coverage_primitive_kind = "source_vertex"
            else:
                tetrahedra, hull_receipt = _convex_hull_tetrahedra(triangles)
                leaf_primitive_indices = _partition_simplices(
                    tetrahedra, leaf_count
                )
                leaf_obbs, leaf_fit_receipts = _partition_pca_obbs(
                    tetrahedra, leaf_primitive_indices
                )
                coverage_basis = "complete_convex_hull_tetra_fan_pca_obb_union"
                coverage_primitive_kind = "convex_hull_tetrahedron"
            if mujoco_primitives:
                leaf_obbs, mujoco_cover_receipt = (
                    _expand_obbs_for_mujoco_primitives(
                        leaf_obbs,
                        mujoco_primitives,
                    )
                )
                consumed_mujoco_sources.add(relative_mesh.lower())
                coverage_basis += "_plus_complete_mujoco_actual_collision_cover"
            link_from_mesh_rotation, link_from_mesh_translation = (
                _origin_transform(collision.find("origin"))
            )
            owner_from_mesh_rotation, owner_from_mesh_translation = _compose(
                owner_from_link_rotation,
                owner_from_link_translation,
                link_from_mesh_rotation,
                link_from_mesh_translation,
            )
            repo_relative_mesh = mesh_path.relative_to(REPO_ROOT).as_posix()
            mesh_sha = _sha256_bytes(mesh_path.read_bytes())
            mesh_receipts[repo_relative_mesh] = mesh_sha
            source_component_id = (
                f"{owner}:{link_name}:{collision_index}:{relative_mesh}"
            )
            source_component_count += 1
            partition_sha256 = _sha256_bytes(
                _canonical_json_bytes(
                    [list(leaf) for leaf in leaf_primitive_indices]
                )
            )
            partition_receipt: dict[str, object] = {
                "coverage_basis": coverage_basis,
                "coverage_primitive_kind": coverage_primitive_kind,
                "leaf_count": leaf_count,
                "leaf_fit_receipts": leaf_fit_receipts,
                "leaf_primitive_counts": [
                    len(leaf) for leaf in leaf_primitive_indices
                ],
                "leaf_primitive_indices_sha256": [
                    _sha256_bytes(_canonical_json_bytes(list(leaf)))
                    for leaf in leaf_primitive_indices
                ],
                "mesh_path": repo_relative_mesh,
                "mesh_sha256": mesh_sha,
                "partition_sha256": partition_sha256,
                "source_component_id": source_component_id,
                "source_triangle_count": len(triangles),
                "source_vertex_count": len(source_vertices),
                "source_vertices_sha256": source_vertices_sha256,
            }
            if hull_receipt is not None:
                partition_receipt["convex_hull"] = hull_receipt
            if mujoco_cover_receipt is not None:
                partition_receipt["mujoco_actual_collision_cover"] = (
                    mujoco_cover_receipt
                )
            partition_receipts.append(partition_receipt)
            for leaf_index, (
                (mesh_center, mesh_half_axes),
                primitive_leaf,
            ) in enumerate(zip(leaf_obbs, leaf_primitive_indices)):
                center_owner = _vector_add(
                    _matrix_vector(owner_from_mesh_rotation, mesh_center),
                    owner_from_mesh_translation,
                )
                # The outer axis dimension contains the three transformed
                # half-axis vectors. Runtime rotates each vector by the live
                # body quaternion and sums absolute components.
                half_axes_owner = tuple(
                    _matrix_vector(owner_from_mesh_rotation, half_axis)
                    for half_axis in mesh_half_axes
                )
                component_id = source_component_id
                if leaf_count > 1:
                    component_id += f"#obb{leaf_index:04d}"
                components.append(
                    {
                        "component_id": component_id,
                        "coverage_basis": coverage_basis,
                        "coverage_primitive_count": len(primitive_leaf),
                        "coverage_primitive_indices_sha256": _sha256_bytes(
                            _canonical_json_bytes(list(primitive_leaf))
                        ),
                        "coverage_primitive_kind": coverage_primitive_kind,
                        "local_center_owner_m": list(center_owner),
                        "local_half_axes_owner_m": [
                            list(axis) for axis in half_axes_owner
                        ],
                        "mesh_path": repo_relative_mesh,
                        "mesh_sha256": mesh_sha,
                        "owner_body_name": owner,
                        "partition_sha256": partition_sha256,
                        "proxy_box_count": leaf_count,
                        "proxy_box_index": leaf_index,
                        "source_component_id": source_component_id,
                        "source_link_name": link_name,
                        "source_triangle_count": len(triangles),
                        "source_vertex_count": len(source_vertices),
                        "source_vertices_sha256": source_vertices_sha256,
                    }
                )
            owner_counts[owner] += 1

    if any(count <= 0 for count in owner_counts.values()):
        missing = [name for name, count in owner_counts.items() if count <= 0]
        raise ValueError(f"runtime A3 bodies lack collision components: {missing}")
    if consumed_mujoco_sources != set(mujoco_primitives_by_source):
        raise ValueError(
            "MuJoCo actual collision sources were not all bound into the proxy"
        )
    observed_gripper_links = sorted(
        {
            str(row["source_link_name"])
            for row in components
            if str(row["source_link_name"]) in set(LEFT_GRIPPER_SOURCE_LINKS)
        }
    )
    if tuple(observed_gripper_links) != LEFT_GRIPPER_SOURCE_LINKS:
        raise ValueError(
            "A3 left OmniPicker3 gripper collision links are not all "
            "materialized: missing "
            f"{sorted(set(LEFT_GRIPPER_SOURCE_LINKS) - set(observed_gripper_links))}"
        )
    components.sort(key=lambda row: str(row["component_id"]))
    source_urdf_sha256 = _sha256_bytes(source_urdf.read_bytes())
    bundle_root = runtime_usd_bundle_root.expanduser().resolve(strict=True)
    content: dict[str, object] = {
        "artifact_type": ARTIFACT_TYPE,
        "body_order": list(order),
        "component_count": len(components),
        "components": components,
        "decomposition": {
            "algorithm": TRIANGLE_PARTITION_ALGORITHM,
            "backend_collision_authority": (
                "component55 target refinement covers Isaac split meshes "
                "under convexHull plus exact canonical MuJoCo target mesh "
                "hulls and analytic wrist primitives; the other 61 source "
                "components retain the shared conservative source proxy"
            ),
            "coverage_contract": (
                "one-box rows contain every source vertex and therefore its "
                "convex hull; split rows partition the complete hull tetra "
                "fan exactly once and each PCA OBB contains every vertex of "
                "its complete tetrahedra; exact MuJoCo mesh vertices and "
                "analytic primitive support intervals are assigned whole to "
                "one leaf and revalidated after float32 conversion"
            ),
            "float32_coverage_validation_scope": (
                "mesh-frame source vertices and component55 MuJoCo target "
                "primitive projection intervals before owner-frame serialization"
            ),
            "hull_toolchain": {
                "numpy_version": PINNED_HULL_NUMPY_VERSION,
                "pca_eigenvalue_tie_rtol": PCA_EIGENVALUE_TIE_RTOL,
                "qhull_options": PINNED_HULL_QHULL_OPTIONS,
                "scipy_version": PINNED_HULL_SCIPY_VERSION,
            },
            "leaf_count_overrides": dict(sorted(MULTI_OBB_LEAF_COUNTS.items())),
            "outward_pad_m": PROXY_OBB_OUTWARD_PAD_M,
            "partition_receipts": sorted(
                partition_receipts, key=lambda row: str(row["source_component_id"])
            ),
        },
        "left_gripper_source_links": list(LEFT_GRIPPER_SOURCE_LINKS),
        "mesh_receipts": [
            {"path": path, "sha256": mesh_receipts[path]}
            for path in sorted(mesh_receipts)
        ],
        "plant_identity": _plant_identity(
            source_urdf, source_urdf_sha256, mesh_receipts, bundle_root
        ),
        "mujoco_actual_collision_binding": mujoco_collision_binding,
        "runtime_usd_bundle": _runtime_usd_binding(
            runtime_usd_bundle_root
        ),
        "schema_version": SCHEMA_VERSION,
        "source_component_count": source_component_count,
        "source_urdf": {
            "path": source_urdf.relative_to(REPO_ROOT).as_posix(),
            "sha256": source_urdf_sha256,
        },
    }
    content["content_sha256"] = _sha256_bytes(_canonical_json_bytes(content))
    return content


def main() -> int:
    args = _parse_args()
    document = _artifact(
        args.source_urdf,
        args.body_order_source,
        args.mujoco_mjcf,
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
