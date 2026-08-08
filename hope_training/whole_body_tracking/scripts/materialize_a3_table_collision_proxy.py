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
    / "a3_table_collision_components.v1.json"
)
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "a3_table_collision_component_obb_v1"
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
PLANT_IDENTITY_KIND = "a3_collision_proxy_plant_identity_v1"
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
        "left_gripper_source_links": list(LEFT_GRIPPER_SOURCE_LINKS),
        "mesh_receipts": [
            {"path": path, "sha256": mesh_receipts[path]}
            for path in sorted(mesh_receipts)
        ],
        "plant_identity": _plant_identity(
            source_urdf, source_urdf_sha256, mesh_receipts, bundle_root
        ),
        "runtime_usd_bundle": _runtime_usd_binding(
            runtime_usd_bundle_root
        ),
        "schema_version": SCHEMA_VERSION,
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
