#!/usr/bin/env python3
"""Build or verify the non-canonical A3-P1 0803 31-action Isaac candidate.

The vendor delivery is immutable and ignored under ``vendor_assets``.  This
tool derives a separate ignored asset under a versioned project-owned contract:
the non-policy left gripper is locked at its raw-URDF coordinate zero, while the
raw URDF (not the contradictory workbook) owns the mount frame.  Missing vendor
collision meshes are never fabricated; only the twenty absent gripper collision
elements are disabled and recorded.  The tool restores the established mixed-
case A3 body-name ABI and never changes the current ``assets/agibot_a3`` runtime
asset.

``v2`` adds one -- and only one -- declared correction to delivered geometry: the
0803 delivery breaks left/right mirror symmetry at ``right_elbow_joint`` in a way
no delivered mesh supports (see ``MIRROR_SYMMETRY_CORRECTION_CONTRACT``).  This is
a project-owned, evidence-backed, provisional deviation from raw, not a claim that
the vendor agrees.  ``v1`` and its Pod import receipt stay untouched on disk and in
``configs/a3_p1_0803_31d_v1.json``; its Pod evidence explicitly does not transfer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_INTAKE_MANIFEST = REPO_ROOT / "configs" / "a3_p1_0803_raw_intake_v1.json"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "vendor_assets" / "agibot" / "A3-P1-32dof-0803-BerkeleyPingpang-90deg"
)
ASSET_VERSION = "v2"
PREDECESSOR_VERSION = "v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / f"agibot_a3_p1_0803_31d_{ASSET_VERSION}"
)
ACTIVE_ASSET_ROOT = DEFAULT_OUTPUT_ROOT.parent / "agibot_a3"
DEFAULT_SUCCESSOR_MANIFEST = REPO_ROOT / "configs" / f"a3_p1_0803_31d_{ASSET_VERSION}.json"
PREDECESSOR_MANIFEST = REPO_ROOT / "configs" / f"a3_p1_0803_31d_{PREDECESSOR_VERSION}.json"
PREDECESSOR_OUTPUT_ROOT = (
    DEFAULT_OUTPUT_ROOT.parent / f"agibot_a3_p1_0803_31d_{PREDECESSOR_VERSION}"
)
RUNTIME_JOINT_ORDER = REPO_ROOT / "configs" / "a3_runtime_articulation_joint_order.txt"
RUNTIME_BODY_ORDER = REPO_ROOT / "configs" / "a3_runtime_body_order.txt"
GMR_JOINT_ORDER = REPO_ROOT / "configs" / "a3_gmr_dof_pos_joint_order.txt"
JOINT_BIJECTION = REPO_ROOT / "configs" / "a3_joint_order_bijection_v1.json"
ACTIVE_REFERENCE_URDF = (
    REPO_ROOT
    / "agi"
    / "URDF"
    / "A3T2.5-URDF-std-pingpang"
    / "urdf"
    / "URDF-JOINT-LINK.urdf"
)
ACTIVE_REFERENCE_MESHES = ACTIVE_REFERENCE_URDF.parents[1] / "meshes"

GRIPPER_MOUNT_JOINT = "left_OP3_joint"
RAW_DUPLICATE_LINK = "imu_in_pelvis_link"
EXPECTED_MOVABLE_JOINTS = 31
EXPECTED_FIXED_GRIPPER_JOINTS = 9
EXPECTED_GRIPPER_SUBTREE_MASS_KG = 0.76626209416
EXPECTED_NORMALIZED_UNIQUE_LINK_MASS_KG = 57.60001015416
EXPECTED_MISSING_GRIPPER_COLLISION_COUNT = 20
EXPECTED_MISSING_GRIPPER_COLLISIONS = (
    ("left_base_link", "../meshes/base_link_collision.stl"),
    ("left_link1", "../meshes/Link1_collision.stl"),
    ("left_link10", "../meshes/Link10_collision.stl"),
    ("left_link11", "../meshes/Link11_collision.stl"),
    ("left_link11-1", "../meshes/Link11-1_collision.stl"),
    ("left_link13", "../meshes/Link13_collision.stl"),
    ("left_link14", "../meshes/Link14_collision.stl"),
    ("left_link14-1", "../meshes/Link14-1_collision.stl"),
    ("left_link15", "../meshes/Link15_collision.stl"),
    ("left_link17", "../meshes/Link17_collision.stl"),
    ("left_link18", "../meshes/Link18_collision.stl"),
    ("left_link2", "../meshes/Link2_collision.stl"),
    ("left_link3", "../meshes/Link3_collision.stl"),
    ("left_link4", "../meshes/Link4_collision.stl"),
    ("left_link4-1", "../meshes/Link4-1_collision.stl"),
    ("left_link6", "../meshes/Link6_collision.stl"),
    ("left_link7", "../meshes/Link7_collision.stl"),
    ("left_link7-1", "../meshes/Link7-1_collision.stl"),
    ("left_link8", "../meshes/Link8_collision.stl"),
    ("left_link9", "../meshes/Link9_collision.stl"),
)
EXPECTED_MALFORMED_FIXED_AXES = {
    "torso_shell_joint": "0.0",
    "imu_in_torso_joint": "",
    "imu_in_pelvis_joint": "",
    "left_knee_shell_joint": "0.0",
    "right_knee_shell_joint": "0.0",
}
USD_SAFE_MESH_ALIASES = {
    "Link7-1.stl": "Link7_1.stl",
    "Link14-1.stl": "Link14_1.stl",
    "Link4-1.stl": "Link4_1.stl",
    "Link11-1.stl": "Link11_1.stl",
}
USD_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
V1_POD_VERIFIED_CLOSURE_SHA256 = "73a47e85fd96150c9b27e9601cae892e850b055c3cb9ddf0e77c504ac1188f08"
V1_POD_VERIFIED_URDF_SHA256 = "2f15df8a97004ee230098a89b0c6009bead9c75401b7a9c4bb738e6ff5622535"

# One declared, evidence-backed correction to delivered geometry.  Keyed by joint;
# every field is asserted against the raw delivery before the rewrite is applied,
# so a re-delivery that changes any of them fails loudly instead of being absorbed.
MIRROR_SYMMETRY_ORIGIN_CORRECTIONS = {
    "right_elbow_joint": {
        "raw_origin_xyz": "0.001 0 -0.1325",
        "raw_origin_rpy": "0 0 0",
        "corrected_origin_xyz": "0.01 0 -0.1325",
        "corrected_component": "x",
        "correction_m": 0.009,
        "mirror_reference_joint": "left_elbow_joint",
        "mirror_reference_origin_xyz": "0.01 0 -0.1325",
        "predecessor_origin_xyz": "0.00999999997356363 0 -0.132999999990091",
        "retained_delivered_component": (
            "z=-0.1325 is kept from the delivery; it is the vendor's genuine repair of a "
            "predecessor left/right z asymmetry (old right z=-0.133 vs left z=-0.1325)"
        ),
    },
}
MIRROR_SYMMETRY_CORRECTION_CONTRACT = {
    "schema_version": 1,
    "authority": "project_owned_geometry_correction_20260806",
    "status": "provisional_pending_vendor_confirmation",
    "claim_scope": (
        "project-owned repair of a delivery-internal inconsistency; NOT a vendor-confirmed "
        "value, NOT a claim that the delivered CAD is wrong at source, and NOT transferable "
        "to hardware without vendor reply"
    ),
    "vendor_question_open": True,
    "workbook_agrees_with_raw_urdf": True,
    "workbook_agreement_note": (
        "the delivered joint workbook also states x=0.001, so the defect is upstream of URDF "
        "emission and must still be reported even though the project patches it locally"
    ),
    "evidence": (
        "(1) no delivered part geometry moved: all 44 meshes shared with the predecessor are "
        "byte-identical by SHA-256, including right_shoulder_yaw_Link (the parent whose "
        "mounting face defines this origin) and right_elbow_Link, so a 9 mm shift of the "
        "elbow on the upper arm has no counterpart in the shipped CAD; "
        "(2) the delivery's own inertials say the two elbow parts are one mirrored part: "
        "left_elbow_link and right_elbow_link have identical mass (delta 0.000 g) and their "
        "centres of mass agree under the y-mirror to 0.0745 mm in x and 0.19 mm overall, so "
        "the 9 mm x asymmetry between their mount origins is ~90x the part-level mirror "
        "residual and ~120x it in the x component alone; "
        "(3) left_elbow_joint keeps x=0.01 in this same delivery and the predecessor plant "
        "carries x=0.01 on both arms, so the delivery introduces the asymmetry rather than "
        "correcting one; "
        "(4) corroborating, an axis-aligned-bounding-box mating check of right_elbow_Link "
        "against right_shoulder_yaw_Link reproduces the left arm's x overlap to 0.87 mm at "
        "x=0.01 but misses it by 8.13 mm at the delivered x=0.001. "
        "Note the parts are NOT identical at tessellation level (differing vertex counts), so "
        "(2) rests on the delivered inertial tensors, not on point-to-point mesh comparison"
    ),
    "not_corrected": (
        "right_hip_roll_joint x -0.0011 -> 0 and the five changed link inertials are accepted "
        "from the delivery unchanged; only the mirror-breaking elbow x is corrected"
    ),
}
EXPECTED_CORRECTED_VS_RAW_RACKET_DELTA_M = 0.009000000000000008
EXPECTED_CORRECTED_VS_PREDECESSOR_RACKET_DELTA_M = 0.0004999999900224823
EXPECTED_RAW_VS_PREDECESSOR_RACKET_DELTA_M = 0.009013878161711154
REQUIRED_RACKET_MESHES = {
    "right_hand_pingpang_Link.stl",
    "pingpang_red_Link.stl",
    "pingpang_black_Link.stl",
    "pingbang_ball_Link.stl",
}
EXPECTED_RACKET_MESH_SHA256 = {
    "right_hand_pingpang_Link.stl": "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd",
    "pingpang_red_Link.stl": "94182ec1c7c64db8c5ec7ce5f9aad44d427f433a6aae5cf23aa655e077633842",
    "pingpang_black_Link.stl": "5f0e772ea9ed81e5b70f5dfb4ded49f9d269c54c893249857209f85168361b1b",
    "pingbang_ball_Link.stl": "21c39c9f6112304776f4eadf7439193163a814b59391790df027ff5aa8249c93",
}
OFFICIAL_RACKET_SITE_XYZ_M = (0.21021, 0.032078, 0.032036)
OFFICIAL_RACKET_SITE_RPY_RAD = (0.0, 0.0, 0.0)
PROJECT_GRIPPER_LOCK_CONTRACT = {
    "schema_version": 1,
    "authority": "project_owned_training_projection_20260804",
    "raw_mount_frame_authority": "primary_urdf_selected_over_workbook",
    "locked_joint_position": "all_nine_gripper_movable_coordinates_q_equal_zero",
    "q0_claim_scope": "project_lock_pose_not_vendor_neutral_or_hardware_home",
    "gripper_policy_controlled": False,
    "missing_collision_policy": (
        "disable_only_collision_elements_whose_twenty_gripper_meshes_are_absent;"
        "never_substitute_visual_mesh_or_invent_geometry"
    ),
}


class AssetError(RuntimeError):
    """Raised when a source or derived asset violates the closed contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(payload.encode("utf-8"))


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(AssetError(f"non-finite JSON number: {value}")),
    )
    if not isinstance(data, dict):
        raise AssetError(f"expected JSON object: {path}")
    return data


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def closure(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise AssetError(f"asset closure root must be a real directory: {root}")
    entries = []
    paths = sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix())
    for path in paths:
        if path.is_symlink():
            raise AssetError(f"asset closure forbids symlink: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise AssetError(f"asset closure forbids non-regular file: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return {
        "algorithm": "canonical_relative_path_size_sha256_json_v1",
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "sha256": canonical_json_sha(entries),
        "files": entries,
    }


def verify_raw_closure(source_root: Path, intake: dict[str, Any]) -> None:
    observed = closure(source_root)
    expected = intake["closure"]
    for key in ("file_count", "total_bytes", "sha256"):
        if observed[key] != expected[key]:
            raise AssetError(
                f"raw intake {key} mismatch: expected {expected[key]!r}, observed {observed[key]!r}"
            )
    urdf_rel = intake["primary_urdf"]["path"]
    urdf = source_root / urdf_rel
    if not urdf.is_file():
        raise AssetError(f"missing raw primary URDF: {urdf}")
    if sha256_path(urdf) != intake["primary_urdf"]["sha256"]:
        raise AssetError("raw primary URDF SHA-256 does not match intake manifest")


def require_isolated_successor_root(output_root: Path, source_root: Optional[Path] = None) -> None:
    """Reject overlap with the current runtime pointer or immutable raw source."""

    resolved_output = output_root.resolve(strict=False)
    resolved_active = ACTIVE_ASSET_ROOT.resolve(strict=False)
    if resolved_output == resolved_active or resolved_active in resolved_output.parents:
        raise AssetError(
            "refusing to use the current runtime asset or a child path as successor output: "
            f"{output_root}"
        )
    resolved_predecessor = PREDECESSOR_OUTPUT_ROOT.resolve(strict=False)
    if resolved_output == resolved_predecessor or resolved_predecessor in resolved_output.parents:
        raise AssetError(
            "refusing to use the Pod-verified predecessor asset or a child path as successor "
            f"output: {output_root}"
        )
    if source_root is not None:
        resolved_source = source_root.resolve(strict=False)
        if (
            resolved_output == resolved_source
            or resolved_source in resolved_output.parents
            or resolved_output in resolved_source.parents
        ):
            raise AssetError(
                "refusing successor output that overlaps the immutable raw source: "
                f"output={output_root}, source={source_root}"
            )


def descendants_for_joint(root: ET.Element, mount_joint_name: str) -> tuple[list[str], list[str]]:
    joints = root.findall("joint")
    by_parent: dict[str, list[ET.Element]] = defaultdict(list)
    mount = None
    for joint in joints:
        if joint.find("parent") is None or joint.find("child") is None:
            raise AssetError(f"joint lacks parent/child: {joint.get('name')}")
        by_parent[joint.find("parent").get("link")].append(joint)
        if joint.get("name") == mount_joint_name:
            mount = joint
    if mount is None:
        raise AssetError(f"missing gripper mount joint: {mount_joint_name}")

    removed_joints: list[str] = []
    removed_links: list[str] = []
    stack = [mount]
    while stack:
        joint = stack.pop()
        child = joint.find("child").get("link")
        removed_joints.append(joint.get("name"))
        removed_links.append(child)
        stack.extend(reversed(by_parent.get(child, [])))
    return removed_joints, removed_links


def normalized_link_name(raw_name: str) -> str:
    if raw_name == "pelvis_link":
        return raw_name
    if raw_name.endswith("_link"):
        return raw_name[:-5] + "_Link"
    return raw_name


def element_fingerprint(element: ET.Element | None) -> Any:
    if element is None:
        return None
    return {
        "tag": element.tag,
        "attrib": dict(sorted(element.attrib.items())),
        "children": [element_fingerprint(child) for child in element],
    }


def link_mass_kg(root: ET.Element, names: set[str] | None = None) -> float:
    values = []
    for link in root.findall("link"):
        if names is not None and link.get("name") not in names:
            continue
        mass = link.find("inertial/mass")
        if mass is None:
            continue
        value = float(mass.get("value"))
        if not math.isfinite(value):
            raise AssetError(f"non-finite mass on {link.get('name')}")
        values.append(value)
    return math.fsum(values)


def retained_semantics(root: ET.Element, retained_link_raw_names: set[str], retained_joint_names: set[str]) -> dict[str, Any]:
    links = {}
    seen: set[str] = set()
    for link in reversed(root.findall("link")):
        name = link.get("name")
        if name not in retained_link_raw_names or name in seen:
            continue
        seen.add(name)
        links[name] = element_fingerprint(link.find("inertial"))
    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in retained_joint_names:
            continue
        joints[name] = {
            "type": joint.get("type"),
            "origin": element_fingerprint(joint.find("origin")),
            "axis": element_fingerprint(joint.find("axis")),
            "limit": element_fingerprint(joint.find("limit")),
        }
    return {"links": links, "joints": joints}


def parse_rgba(root: ET.Element) -> list[dict[str, str]]:
    invalid = []
    for element in root.findall(".//*[@rgba]"):
        rgba = element.get("rgba", "")
        try:
            values = [float(part) for part in rgba.split()]
        except ValueError:
            values = []
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            invalid.append({"tag": element.tag, "rgba": rgba})
    return invalid


def movable_joint_names(root: ET.Element) -> list[str]:
    return [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") not in {"fixed", "floating"}
    ]


def mesh_refs(root: ET.Element) -> list[str]:
    return [mesh.get("filename") for mesh in root.findall(".//mesh") if mesh.get("filename")]


def normalize_malformed_fixed_axes(root: ET.Element) -> list[dict[str, Any]]:
    """Remove importer-invalid axes from fixed joints, where axes have no kinematic meaning."""
    observed: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        axis = joint.find("axis")
        if axis is None:
            continue
        raw_xyz = axis.get("xyz", "")
        parts = raw_xyz.split()
        try:
            valid = len(parts) == 3 and all(math.isfinite(float(part)) for part in parts)
        except ValueError:
            valid = False
        if valid:
            continue
        if joint.get("type") != "fixed":
            raise AssetError(
                f"movable joint {joint.get('name')!r} has importer-invalid axis xyz={raw_xyz!r}"
            )
        name = joint.get("name")
        observed[name] = raw_xyz
        normalized.append(
            {
                "joint": name,
                "type": "fixed",
                "raw_axis_xyz": raw_xyz,
                "normalized_axis": None,
                "reason": "URDF fixed joints do not use an axis; omit importer-invalid non-3-vector data",
            }
        )
        joint.remove(axis)
    if observed != EXPECTED_MALFORMED_FIXED_AXES:
        raise AssetError(
            "unexpected malformed fixed-joint axis set: "
            f"expected {EXPECTED_MALFORMED_FIXED_AXES}, observed {observed}"
        )
    return normalized


def validate_importer_safe_axes_and_meshes(root: ET.Element) -> None:
    for joint in root.findall("joint"):
        axis = joint.find("axis")
        if axis is None:
            continue
        raw_xyz = axis.get("xyz", "")
        parts = raw_xyz.split()
        try:
            valid = len(parts) == 3 and all(math.isfinite(float(part)) for part in parts)
        except ValueError:
            valid = False
        if not valid:
            raise AssetError(f"importer-invalid axis remains on joint {joint.get('name')}: {raw_xyz!r}")
    invalid_mesh_basenames = sorted(
        {
            Path(ref).name
            for ref in mesh_refs(root)
            if USD_IDENTIFIER.fullmatch(Path(ref).stem) is None
        }
    )
    if invalid_mesh_basenames:
        raise AssetError(f"USD-unsafe retained mesh basenames: {invalid_mesh_basenames}")


def apply_mirror_symmetry_corrections(root: ET.Element) -> list[dict[str, Any]]:
    """Apply declared mirror-symmetry origin corrections, asserting every premise first.

    Each premise -- the raw value being corrected, the raw rpy, the mirror reference
    joint's value, which single component moves, and by how much -- is checked against
    the contract before anything is written.  A re-delivery that silently changes any
    of them fails here rather than being absorbed into the successor.
    """

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    applied = []
    for name, spec in sorted(MIRROR_SYMMETRY_ORIGIN_CORRECTIONS.items()):
        joint = joints.get(name)
        if joint is None:
            raise AssetError(f"mirror-symmetry correction target is absent: {name}")
        origin = joint.find("origin")
        if origin is None:
            raise AssetError(f"mirror-symmetry correction target has no origin: {name}")
        if origin.get("xyz") != spec["raw_origin_xyz"]:
            raise AssetError(
                f"mirror-symmetry correction premise drifted for {name}: delivered origin xyz "
                f"is {origin.get('xyz')!r}, contract expects {spec['raw_origin_xyz']!r}"
            )
        if origin.get("rpy") != spec["raw_origin_rpy"]:
            raise AssetError(
                f"mirror-symmetry correction premise drifted for {name}: delivered origin rpy "
                f"is {origin.get('rpy')!r}, contract expects {spec['raw_origin_rpy']!r}"
            )
        mirror_name = spec["mirror_reference_joint"]
        mirror = joints.get(mirror_name)
        mirror_origin = None if mirror is None else mirror.find("origin")
        if mirror_origin is None or mirror_origin.get("xyz") != spec["mirror_reference_origin_xyz"]:
            raise AssetError(
                f"mirror reference origin drifted for {mirror_name}: "
                f"{None if mirror_origin is None else mirror_origin.get('xyz')!r}, "
                f"contract expects {spec['mirror_reference_origin_xyz']!r}"
            )
        raw_values = [float(value) for value in spec["raw_origin_xyz"].split()]
        corrected_values = [float(value) for value in spec["corrected_origin_xyz"].split()]
        changed = [
            axis
            for axis, before, after in zip("xyz", raw_values, corrected_values)
            if before != after
        ]
        if changed != [spec["corrected_component"]]:
            raise AssetError(
                f"mirror-symmetry correction for {name} changes {changed}, contract declares "
                f"only {spec['corrected_component']!r}"
            )
        delta = math.dist(raw_values, corrected_values)
        if not math.isclose(delta, spec["correction_m"], rel_tol=0.0, abs_tol=1e-15):
            raise AssetError(
                f"mirror-symmetry correction magnitude for {name} is {delta}, contract "
                f"declares {spec['correction_m']}"
            )
        origin.set("xyz", spec["corrected_origin_xyz"])
        applied.append(
            {
                "joint": name,
                "delivered_origin_xyz": spec["raw_origin_xyz"],
                "corrected_origin_xyz": spec["corrected_origin_xyz"],
                "origin_rpy_unchanged": spec["raw_origin_rpy"],
                "corrected_component": spec["corrected_component"],
                "correction_m": delta,
                "mirror_reference_joint": mirror_name,
                "mirror_reference_origin_xyz": spec["mirror_reference_origin_xyz"],
                "predecessor_origin_xyz": spec["predecessor_origin_xyz"],
                "retained_delivered_component": spec["retained_delivered_component"],
                "axis_unchanged": True,
                "limit_unchanged": True,
            }
        )
    return applied


def normalize(raw_root: ET.Element, source_meshes: Path) -> tuple[ET.Element, dict[str, Any]]:
    root = copy.deepcopy(raw_root)
    gripper_joint_names, gripper_link_names = descendants_for_joint(root, GRIPPER_MOUNT_JOINT)
    gripper_joint_set = set(gripper_joint_names)
    gripper_link_set = set(gripper_link_names)

    fixed_movable = [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("name") in gripper_joint_set and joint.get("type") not in {"fixed", "floating"}
    ]
    if len(fixed_movable) != EXPECTED_FIXED_GRIPPER_JOINTS:
        raise AssetError(
            f"expected {EXPECTED_FIXED_GRIPPER_JOINTS} movable gripper joints, found {fixed_movable}"
        )
    raw_joints = {joint.get("name"): joint for joint in root.findall("joint")}
    for name in fixed_movable:
        limit = raw_joints[name].find("limit")
        if limit is None:
            raise AssetError(f"project-locked gripper joint has no limit: {name}")
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= 0.0 <= upper):
            raise AssetError(f"project q=0 lock is outside raw limits for {name}: [{lower}, {upper}]")
    raw_gripper_mass = link_mass_kg(root, gripper_link_set)
    if not math.isclose(
        raw_gripper_mass, EXPECTED_GRIPPER_SUBTREE_MASS_KG, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssetError(f"raw gripper subtree mass drifted: {raw_gripper_mass}")

    invalid_rgba_before = parse_rgba(root)
    if invalid_rgba_before != [{"tag": "color", "rgba": "nan nan nan nan"}]:
        raise AssetError(f"unexpected raw invalid-rgba set: {invalid_rgba_before}")
    removed_invalid_visuals = []
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            if parse_rgba(visual):
                removed_invalid_visuals.append({"link": link.get("name"), "reason": "nonfinite_rgba"})
                link.remove(visual)

    fixed_joint_originals = {}
    for joint in root.findall("joint"):
        if joint.get("name") not in fixed_movable:
            continue
        fixed_joint_originals[joint.get("name")] = {
            "type": joint.get("type"),
            "axis": element_fingerprint(joint.find("axis")),
            "limit": element_fingerprint(joint.find("limit")),
            "origin": element_fingerprint(joint.find("origin")),
        }
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)

    duplicate_count = Counter(link.get("name") for link in root.findall("link"))[RAW_DUPLICATE_LINK]
    if duplicate_count != 2:
        raise AssetError(f"expected two raw {RAW_DUPLICATE_LINK!r} definitions, found {duplicate_count}")
    duplicates = [link for link in root.findall("link") if link.get("name") == RAW_DUPLICATE_LINK]
    duplicate_sha = [canonical_json_sha(element_fingerprint(link)) for link in duplicates]
    # The last definition is the one that uses the delivered dedicated collision mesh.
    root.remove(duplicates[0])

    retained_link_raw_names = {link.get("name") for link in root.findall("link")}
    retained_joint_names = {joint.get("name") for joint in root.findall("joint")}
    preserved_body_joint_names = retained_joint_names - set(fixed_movable)
    raw_semantics = retained_semantics(raw_root, retained_link_raw_names, preserved_body_joint_names)

    link_name_map = {
        name: name if name in gripper_link_set else normalized_link_name(name)
        for name in sorted(retained_link_raw_names)
    }
    for link in root.findall("link"):
        link.set("name", link_name_map[link.get("name")])
    for joint in root.findall("joint"):
        for tag in ("parent", "child"):
            endpoint = joint.find(tag)
            raw_name = endpoint.get("link")
            if raw_name not in link_name_map:
                raise AssetError(f"retained joint {joint.get('name')} references removed link {raw_name}")
            endpoint.set("link", link_name_map[raw_name])

    root.set("name", f"A3-P1-0803-BerkeleyPingpang-31action-normalized-{ASSET_VERSION}")

    mirror_symmetry_corrections = apply_mirror_symmetry_corrections(root)
    malformed_fixed_axis_normalizations = normalize_malformed_fixed_axes(root)
    mesh_reference_rewrites, removed_missing_collisions = normalize_mesh_references(root, source_meshes)
    if len(removed_missing_collisions) != EXPECTED_MISSING_GRIPPER_COLLISION_COUNT:
        raise AssetError(
            "expected "
            f"{EXPECTED_MISSING_GRIPPER_COLLISION_COUNT} missing gripper collision refs, "
            f"found {len(removed_missing_collisions)}"
        )
    if any(item["link"] not in gripper_link_set for item in removed_missing_collisions):
        raise AssetError("normalization attempted to disable a missing collision outside the gripper")
    removed_pairs = tuple((item["link"], item["reference"]) for item in removed_missing_collisions)
    if removed_pairs != EXPECTED_MISSING_GRIPPER_COLLISIONS:
        raise AssetError(f"missing gripper collision inventory drifted: {removed_pairs}")

    output_semantics = {
        "links": {
            raw_name: element_fingerprint(
                next(link.find("inertial") for link in root.findall("link") if link.get("name") == normalized_name)
            )
            for raw_name, normalized_name in link_name_map.items()
        },
        "joints": {
            joint.get("name"): {
                "type": joint.get("type"),
                "origin": element_fingerprint(joint.find("origin")),
                "axis": element_fingerprint(joint.find("axis")),
                "limit": element_fingerprint(joint.find("limit")),
            }
            for joint in root.findall("joint")
            if joint.get("name") in preserved_body_joint_names
        },
    }
    for item in malformed_fixed_axis_normalizations:
        raw_semantics["joints"][item["joint"]]["axis"] = None
    # Fold in the declared corrections so the equality below still proves that nothing
    # ELSE moved: every other origin, every axis, every limit, and every retained inertial.
    for item in mirror_symmetry_corrections:
        expected_origin = raw_semantics["joints"][item["joint"]]["origin"]
        if expected_origin is None or expected_origin["attrib"].get("xyz") != item["delivered_origin_xyz"]:
            raise AssetError(
                f"declared correction does not match the raw semantics snapshot for {item['joint']}"
            )
        expected_origin["attrib"]["xyz"] = item["corrected_origin_xyz"]
    if raw_semantics != output_semantics:
        raise AssetError(
            "normalization changed retained body inertials or joint origin/axis/limit semantics "
            "beyond the declared mirror-symmetry corrections"
        )
    normalized_mass = link_mass_kg(root)
    if not math.isclose(
        normalized_mass, EXPECTED_NORMALIZED_UNIQUE_LINK_MASS_KG, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssetError(f"normalized unique-link mass drifted: {normalized_mass}")

    normalized_movable = movable_joint_names(root)
    if len(normalized_movable) != EXPECTED_MOVABLE_JOINTS:
        raise AssetError(f"normalized asset has {len(normalized_movable)} movable joints, expected 31")

    invalid_rgba_after = parse_rgba(root)
    if invalid_rgba_after:
        raise AssetError(f"normalized asset retains invalid rgba values: {invalid_rgba_after}")
    validate_importer_safe_axes_and_meshes(root)

    usd_safe_mesh_aliases = []
    for raw_name, normalized_name in sorted(USD_SAFE_MESH_ALIASES.items()):
        source = source_meshes / raw_name
        if not source.is_file():
            raise AssetError(f"missing source mesh for USD-safe alias: {source}")
        usd_safe_mesh_aliases.append(
            {
                "raw_basename": raw_name,
                "normalized_basename": normalized_name,
                "source_sha256": sha256_path(source),
                "bytes_unchanged": True,
            }
        )

    diff = {
        "robot_name": {
            "raw": raw_root.get("name"),
            "normalized": root.get("name"),
        },
        "fixed_gripper_subtree": {
            "root_joint": GRIPPER_MOUNT_JOINT,
            "joint_names": sorted(gripper_joint_names),
            "link_names": sorted(gripper_link_names),
            "converted_to_fixed_joint_names": fixed_movable,
            "converted_joint_originals": fixed_joint_originals,
            "fixed_pose": "each converted joint at its URDF q=0 origin",
            "lock_contract": PROJECT_GRIPPER_LOCK_CONTRACT,
            "all_q0_within_raw_limits": True,
            "retained_subtree_mass_kg": raw_gripper_mass,
            "reason": "left gripper is outside the established 31-action policy ABI; project-owned q=0 locking retains delivered mass, inertias, COMs, and joint origins without claiming vendor neutral or inventing coupling",
        },
        "removed_duplicate_link": {
            "name": RAW_DUPLICATE_LINK,
            "removed_occurrence": "first_document_occurrence",
            "retained_occurrence": "last_document_occurrence_with_dedicated_collision_mesh",
            "raw_occurrence_fingerprint_sha256": duplicate_sha,
        },
        "link_name_map": link_name_map,
        "mirror_symmetry_corrections": mirror_symmetry_corrections,
        "mirror_symmetry_correction_contract": MIRROR_SYMMETRY_CORRECTION_CONTRACT,
        "mesh_reference_policy": "rewrite to delivered case-exact basenames, replace USD-unsafe hyphens with deterministic underscore aliases, copy only referenced bytes, and explicitly disable only the twenty absent left-gripper collision elements",
        "mesh_reference_rewrites": mesh_reference_rewrites,
        "usd_safe_mesh_aliases": usd_safe_mesh_aliases,
        "malformed_fixed_axis_normalizations": malformed_fixed_axis_normalizations,
        "removed_missing_collision_elements": removed_missing_collisions,
        "removed_invalid_visual_elements": removed_invalid_visuals,
        "invalid_rgba_before": invalid_rgba_before,
        "invalid_rgba_after": invalid_rgba_after,
        "retained_semantics_sha256": canonical_json_sha(raw_semantics),
        "normalized_unique_link_mass_kg": normalized_mass,
    }
    return root, diff


def exact_mesh_source(source_meshes: Path, ref: str) -> Path:
    prefix = "../meshes/"
    if not ref.startswith(prefix) or Path(ref).name != ref[len(prefix) :]:
        raise AssetError(f"unsupported mesh reference outside delivered mesh root: {ref}")
    requested = Path(ref).name
    entries = {path.name: path for path in source_meshes.iterdir() if path.is_file()}
    if requested in entries:
        return entries[requested]
    raw_alias = next(
        (raw for raw, normalized in USD_SAFE_MESH_ALIASES.items() if normalized == requested), None
    )
    if raw_alias is not None:
        if raw_alias not in entries:
            raise AssetError(f"missing delivered source for USD-safe mesh alias: {raw_alias}")
        return entries[raw_alias]
    case_matches = sorted(name for name in entries if name.casefold() == requested.casefold())
    if len(case_matches) == 1:
        raise AssetError(f"mesh reference is not case-exact: {ref}; delivered match={case_matches[0]}")
    if case_matches:
        raise AssetError(f"ambiguous case-folded mesh reference: {ref}; matches={case_matches}")
    raise AssetError(f"missing retained mesh: {ref}")


def normalize_mesh_references(root: ET.Element, source_meshes: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    entries = {path.name: path for path in source_meshes.iterdir() if path.is_file()}
    by_casefold: dict[str, list[str]] = defaultdict(list)
    for name in entries:
        by_casefold[name.casefold()].append(name)
    rewrites: dict[tuple[str, str], None] = {}
    removed_missing: list[dict[str, str]] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        for collision in list(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            requested = Path(mesh.get("filename")).name
            if requested not in entries and not by_casefold.get(requested.casefold()):
                removed_missing.append({"link": link_name, "reference": mesh.get("filename")})
                link.remove(collision)
        for mesh in link.findall(".//mesh"):
            ref = mesh.get("filename")
            prefix = "../meshes/"
            if not ref.startswith(prefix) or Path(ref).name != ref[len(prefix) :]:
                raise AssetError(f"unsupported mesh reference outside delivered mesh root: {ref}")
            requested = Path(ref).name
            if requested in entries:
                actual = requested
            else:
                matches = sorted(by_casefold.get(requested.casefold(), []))
                if len(matches) != 1:
                    raise AssetError(f"unresolved retained mesh reference {ref}: matches={matches}")
                actual = matches[0]
            normalized = "../meshes/" + USD_SAFE_MESH_ALIASES.get(actual, actual)
            if normalized != ref:
                rewrites[(ref, normalized)] = None
                mesh.set("filename", normalized)
    return (
        [{"raw_reference": raw, "normalized_reference": normalized} for raw, normalized in sorted(rewrites)],
        sorted(removed_missing, key=lambda item: (item["link"], item["reference"])),
    )


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Deterministic equivalent of ElementTree.indent, including Python 3.8."""
    prefix = "\n" + level * "  "
    child_prefix = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in children:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_prefix
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = prefix
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = prefix


def write_urdf(root: ET.Element, path: Path) -> None:
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


def read_order(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def racket_contract(root: ET.Element, output_meshes: Path) -> dict[str, Any]:
    joint = next((joint for joint in root.findall("joint") if joint.get("name") == "pingpang_red_joint"), None)
    if joint is None:
        raise AssetError("normalized URDF lacks pingpang_red_joint")
    origin = joint.find("origin")
    parent = joint.find("parent")
    child = joint.find("child")
    mesh_sha = {}
    for name in sorted(REQUIRED_RACKET_MESHES):
        path = output_meshes / name
        if not path.is_file():
            raise AssetError(f"missing required racket mesh: {path}")
        mesh_sha[name] = sha256_path(path)
    if mesh_sha != EXPECTED_RACKET_MESH_SHA256:
        raise AssetError(f"right-racket mesh bytes drifted from the current/raw authority: {mesh_sha}")
    xyz = tuple(float(value) for value in origin.get("xyz").split())
    rpy = tuple(float(value) for value in origin.get("rpy").split())
    if xyz != OFFICIAL_RACKET_SITE_XYZ_M or rpy != OFFICIAL_RACKET_SITE_RPY_RAD:
        raise AssetError(f"official racket site local transform drifted: xyz={xyz}, rpy={rpy}")
    if parent.get("link") != "right_hand_pingpang_Link" or child.get("link") != "pingpang_red_Link":
        raise AssetError("official racket site parent/child link identity drifted")
    return {
        "parent_link": parent.get("link"),
        "child_link": child.get("link"),
        "origin_xyz": origin.get("xyz"),
        "origin_rpy": origin.get("rpy"),
        "mesh_sha256": mesh_sha,
        "local_contract_exact_current_and_raw": True,
        "official_paddle_center_control_point": True,
    }


def _origin_transform(origin: ET.Element | None) -> list[list[float]]:
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin is not None:
        xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
    cr, cp, cy = (math.cos(value) for value in rpy)
    sr, sp, sy = (math.sin(value) for value in rpy)
    # URDF fixed-axis roll-pitch-yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [
        rotation[0] + [xyz[0]],
        rotation[1] + [xyz[1]],
        rotation[2] + [xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul4(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [math.fsum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def zero_coordinate_link_frame(root: ET.Element, target_link: str) -> list[list[float]]:
    """Return root-to-link FK with every movable coordinate set to zero."""

    joints = []
    child_links = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise AssetError(f"joint lacks parent/child: {joint.get('name')}")
        child_links.add(child.get("link"))
        joints.append((parent.get("link"), child.get("link"), _origin_transform(joint.find("origin"))))
    root_links = [link.get("name") for link in root.findall("link") if link.get("name") not in child_links]
    if len(root_links) != 1:
        raise AssetError(f"expected one URDF root link, found {root_links}")
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    frames = {root_links[0]: identity}
    for _ in range(len(joints) + 1):
        for parent, child, transform in joints:
            if parent in frames and child not in frames:
                frames[child] = _matmul4(frames[parent], transform)
    if target_link not in frames:
        raise AssetError(f"target link is unreachable from URDF root: {target_link}")
    return frames[target_link]


def _frame_delta(left: list[list[float]], right: list[list[float]]) -> dict[str, float]:
    position_delta = math.sqrt(math.fsum((left[index][3] - right[index][3]) ** 2 for index in range(3)))
    rotation_max_abs = max(
        abs(left[row][column] - right[row][column]) for row in range(3) for column in range(3)
    )
    return {
        "position_norm_m": position_delta,
        "rotation_matrix_max_abs": rotation_max_abs,
    }


def racket_fk_lineage_contract(source_root: Path, normalized_root: ET.Element) -> dict[str, Any]:
    raw_root = ET.parse(source_root / "urdf" / "A3-P1-32dof-0803-BerkeleyPingpang-90deg.urdf").getroot()
    active_root = ET.parse(ACTIVE_REFERENCE_URDF).getroot()
    raw_links = {link.get("name"): link for link in raw_root.findall("link")}
    active_links = {link.get("name"): link for link in active_root.findall("link")}
    for raw_name in (
        "right_wrist_yaw_link",
        "right_hand_pingpang_link",
        "pingpang_red_link",
        "pingpang_black_link",
        "pingbang_ball_link",
    ):
        active_name = normalized_link_name(raw_name)
        if element_fingerprint(raw_links[raw_name].find("inertial")) != element_fingerprint(
            active_links[active_name].find("inertial")
        ):
            raise AssetError(f"raw/current right-hand or paddle inertial drifted: {raw_name}")
    raw_joints = {joint.get("name"): joint for joint in raw_root.findall("joint")}
    active_joints = {joint.get("name"): joint for joint in active_root.findall("joint")}
    for name in (
        "right_hand_pingpang_joint",
        "pingpang_red_joint",
        "pingpang_black_joint",
        "pingbang_ball_joint",
    ):
        raw_joint = raw_joints[name]
        active_joint = active_joints[name]
        raw_fixed = {
            "type": raw_joint.get("type"),
            "origin": element_fingerprint(raw_joint.find("origin")),
            "axis": element_fingerprint(raw_joint.find("axis")),
            "limit": element_fingerprint(raw_joint.find("limit")),
        }
        active_fixed = {
            "type": active_joint.get("type"),
            "origin": element_fingerprint(active_joint.find("origin")),
            "axis": element_fingerprint(active_joint.find("axis")),
            "limit": element_fingerprint(active_joint.find("limit")),
        }
        if raw_fixed != active_fixed:
            raise AssetError(f"raw/current right-racket fixed-joint semantics drifted: {name}")
    for root, expected_parent, expected_child, label in (
        (raw_root, "right_hand_pingpang_link", "pingpang_red_link", "raw"),
        (active_root, "right_hand_pingpang_Link", "pingpang_red_Link", "current"),
    ):
        joint = next(
            (candidate for candidate in root.findall("joint") if candidate.get("name") == "pingpang_red_joint"),
            None,
        )
        if joint is None:
            raise AssetError(f"{label} URDF lacks pingpang_red_joint")
        origin = joint.find("origin")
        xyz = tuple(float(value) for value in origin.get("xyz").split())
        rpy = tuple(float(value) for value in origin.get("rpy").split())
        if xyz != OFFICIAL_RACKET_SITE_XYZ_M or rpy != OFFICIAL_RACKET_SITE_RPY_RAD:
            raise AssetError(f"{label} official racket-site local transform drifted")
        if (
            joint.find("parent").get("link") != expected_parent
            or joint.find("child").get("link") != expected_child
        ):
            raise AssetError(f"{label} official racket-site parent/child drifted")
    normalized_frame = zero_coordinate_link_frame(normalized_root, "pingpang_red_Link")
    raw_frame = zero_coordinate_link_frame(raw_root, "pingpang_red_link")
    active_frame = zero_coordinate_link_frame(active_root, "pingpang_red_Link")
    normalized_raw = _frame_delta(normalized_frame, raw_frame)
    normalized_active = _frame_delta(normalized_frame, active_frame)
    raw_active = _frame_delta(raw_frame, active_frame)
    # The successor no longer reproduces the raw racket world site: the declared
    # mirror-symmetry elbow correction moves it, by exactly the correction, with no
    # rotation.  Anything else moving it is a bug.
    if not math.isclose(
        normalized_raw["position_norm_m"],
        EXPECTED_CORRECTED_VS_RAW_RACKET_DELTA_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or normalized_raw["rotation_matrix_max_abs"] > 1e-12:
        raise AssetError(
            "corrected/raw official racket-site q0 delta is not the declared mirror-symmetry "
            f"correction: {normalized_raw}"
        )
    if not math.isclose(
        raw_active["position_norm_m"],
        EXPECTED_RAW_VS_PREDECESSOR_RACKET_DELTA_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or raw_active["rotation_matrix_max_abs"] > 1e-12:
        raise AssetError(f"raw/current official racket-site q0 delta drifted: {raw_active}")
    expected_predecessor_delta = EXPECTED_CORRECTED_VS_PREDECESSOR_RACKET_DELTA_M
    if not math.isclose(
        normalized_active["position_norm_m"],
        expected_predecessor_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or normalized_active["rotation_matrix_max_abs"] > 1e-12:
        raise AssetError(
            "successor/current official racket-site q0 delta drifted: "
            f"{normalized_active}"
        )
    for raw_name, expected_sha in EXPECTED_RACKET_MESH_SHA256.items():
        active_name = raw_name[:-4] + ".STL"
        if sha256_path(ACTIVE_REFERENCE_MESHES / active_name) != expected_sha:
            raise AssetError(f"tracked active right-racket mesh drifted: {active_name}")
    return {
        "right_hand_and_paddle_link_inertials_exact_current": True,
        "right_racket_fixed_joint_semantics_exact_current": True,
        "normalized_preserves_raw_right_chain_and_site_for_all_common_q": False,
        "normalized_right_chain_deviates_from_raw_only_by_declared_corrections": True,
        "declared_correction_is_a_rigid_translation_at_every_q": True,
        "declared_correction_rationale": (
            "the correction is a constant translation in an upstream parent frame and the "
            "upstream rotation is orthonormal, so the racket-site displacement norm is the "
            "same at every joint configuration, and orientation is untouched"
        ),
        "normalized_vs_raw_q0": normalized_raw,
        "expected_normalized_vs_raw_q0_position_delta_m": (
            EXPECTED_CORRECTED_VS_RAW_RACKET_DELTA_M
        ),
        "raw_vs_current_q0": raw_active,
        "expected_raw_vs_current_q0_position_delta_m": (
            EXPECTED_RAW_VS_PREDECESSOR_RACKET_DELTA_M
        ),
        "successor_vs_current_q0": normalized_active,
        "successor_world_site_requires_new_motion_fk_revalidation": (
            normalized_active["position_norm_m"] > 1e-12
            or normalized_active["rotation_matrix_max_abs"] > 1e-12
        ),
        "expected_successor_vs_current_q0_position_delta_m": expected_predecessor_delta,
        "motion_bank_revalidation_note": (
            "the correction shrinks the successor-vs-current racket-site offset from "
            "9.013878 mm to 0.500000 mm, but 0.5 mm still exceeds the racket FK parity gate "
            "(1e-4 m) and the retarget's own full-phase p95 blade-centre precision, so the "
            "motion bank must still be re-audited on the successor before any training claim"
        ),
        "tracked_current_reference_urdf_path": relative_path(ACTIVE_REFERENCE_URDF),
        "tracked_current_reference_urdf_sha256": sha256_path(ACTIVE_REFERENCE_URDF),
    }


def build_manifest(
    source_root: Path,
    output_root: Path,
    intake_path: Path,
    normalized_root: ET.Element,
    diff: dict[str, Any],
) -> dict[str, Any]:
    observed_closure = closure(output_root)
    runtime_joint_order = read_order(RUNTIME_JOINT_ORDER)
    gmr_joint_order = read_order(GMR_JOINT_ORDER)
    movable = movable_joint_names(normalized_root)
    if set(movable) != set(runtime_joint_order):
        raise AssetError("normalized movable joint set does not equal the established runtime 31-action ABI")
    if movable != gmr_joint_order:
        raise AssetError("normalized URDF movable document order does not equal the established GMR order")
    normalized_links = {link.get("name") for link in normalized_root.findall("link")}
    runtime_bodies = read_order(RUNTIME_BODY_ORDER)
    missing_runtime_bodies = sorted(set(runtime_bodies) - normalized_links)
    if missing_runtime_bodies:
        raise AssetError(f"normalized asset misses runtime body ABI names: {missing_runtime_bodies}")

    # v1's Pod receipt is bound to v1's exact bytes.  v2 changes those bytes by design,
    # so this evaluates False and the receipt is withheld rather than inherited.
    pod_import_verified = (
        observed_closure["sha256"] == V1_POD_VERIFIED_CLOSURE_SHA256
        and sha256_path(output_root / "urdf" / "model.urdf") == V1_POD_VERIFIED_URDF_SHA256
    )
    pod_import_receipt = None
    if pod_import_verified:
        pod_import_receipt = {
            "evidence_level": "E2 Pod IsaacLab importer and finite PhysX diagnostic",
            "diagnostic_unauthorized": True,
            "pod": "Pod1 44d3379e8680",
            "observed_at_utc": "2026-08-03T20:13:01Z",
            "isaac_lab_commit": "21f7136325136ca3f6ca4e0a8125edffe5c24f7e",
            "converter_path": "/workspace/IsaacLab/scripts/tools/convert_urdf.py",
            "merge_fixed_joints": True,
            "generated_usd_sha256": "9cf108c9ddef258b30ce8cfe43230bb08254b7141fca755ed930da1113600ead",
            "articulation_joint_count": 31,
            "runtime_joint_order_exact": True,
            "articulation_body_count": 32,
            "runtime_body_order_exact": True,
            "runtime_body_missing": [],
            "runtime_body_extra": [],
            "finite_step_count": 20,
            "finite_steps_all_state_finite": True,
            "initial_q_within_imported_hard_limits": True,
            "qdes_within_imported_hard_limits": True,
            "max_abs_q_drift_rad": 0.05882056802511215,
            "min_root_z_m": 1.066352367401123,
            "formal_standing_hold_verified": False,
            "table_and_self_collision_verified": False,
            "current_runtime_pointer_changed": False,
            "pod_evidence_root": "/workspace/franco/runtime_assets/a3_p1_0803_31d_v1__closure_73a47e85fd96/evidence",
        }

    return {
        "schema_version": 2,
        "manifest_type": "agibot_a3_p1_0803_31action_normalized_asset_v1",
        "asset_id": f"a3_p1_0803_berkeley_pingpang_31action_normalized_{ASSET_VERSION}",
        "candidate_role": "future_primary_successor_candidate_not_current_runtime",
        "predecessor": {
            "asset_id": (
                "a3_p1_0803_berkeley_pingpang_31action_normalized_" f"{PREDECESSOR_VERSION}"
            ),
            "manifest_path": relative_path(PREDECESSOR_MANIFEST),
            "manifest_sha256": sha256_path(PREDECESSOR_MANIFEST),
            "asset_path": relative_path(PREDECESSOR_OUTPUT_ROOT),
            "relationship": "same raw delivery, plus the declared mirror-symmetry corrections",
            "pod_import_evidence_transfers": False,
            "pod_import_evidence_note": (
                "the predecessor Pod IsaacLab import receipt is bound to closure "
                f"{V1_POD_VERIFIED_CLOSURE_SHA256} and URDF {V1_POD_VERIFIED_URDF_SHA256}; "
                "this asset deliberately differs in bytes, so that receipt proves nothing "
                "about it and a fresh Pod import is required"
            ),
        },
        "status": (
            "pod_import_verified_short_step_diagnostic_standing_pending"
            if pod_import_verified
            else "host_static_candidate_pod_import_pending"
        ),
        "source": {
            "producer_path": relative_path(Path(__file__)),
            "producer_sha256": sha256_path(Path(__file__)),
            "raw_intake_manifest_path": relative_path(intake_path),
            "raw_intake_manifest_sha256": sha256_path(intake_path),
            "raw_asset_path": relative_path(source_root),
            "raw_asset_path_git_ignored": True,
        },
        "output": {
            "asset_path": relative_path(output_root),
            "asset_path_git_ignored": True,
            "urdf_path": "urdf/model.urdf",
            "urdf_sha256": sha256_path(output_root / "urdf" / "model.urdf"),
            "closure": observed_closure,
        },
        "normalization_diff": diff,
        "abi": {
            "policy_action_dim": EXPECTED_MOVABLE_JOINTS,
            "movable_joint_document_order": movable,
            "runtime_joint_order_path": relative_path(RUNTIME_JOINT_ORDER),
            "runtime_joint_order_sha256": sha256_path(RUNTIME_JOINT_ORDER),
            "runtime_joint_set_exact": True,
            "gmr_joint_order_path": relative_path(GMR_JOINT_ORDER),
            "gmr_joint_order_sha256": sha256_path(GMR_JOINT_ORDER),
            "urdf_movable_document_order_equals_gmr": True,
            "joint_bijection_path": relative_path(JOINT_BIJECTION),
            "joint_bijection_sha256": sha256_path(JOINT_BIJECTION),
            "runtime_body_order_path": relative_path(RUNTIME_BODY_ORDER),
            "runtime_body_order_sha256": sha256_path(RUNTIME_BODY_ORDER),
            "runtime_body_names_all_present": True,
        },
        "project_gripper_lock_contract": PROJECT_GRIPPER_LOCK_CONTRACT,
        "mirror_symmetry_correction_contract": MIRROR_SYMMETRY_CORRECTION_CONTRACT,
        "right_racket_contract": {
            **racket_contract(normalized_root, output_root / "meshes"),
            "fk_lineage": racket_fk_lineage_contract(source_root, normalized_root),
        },
        "pod_import_receipt": pod_import_receipt,
        "authorization": {
            "current_runtime_pointer_changed": False,
            "canonical_runtime": False,
            "materialization_authorized": True,
            "project_q0_gripper_lock_authorized": True,
            "missing_gripper_collision_elements_explicitly_disabled": True,
            "reproduces_delivered_joint_origins_exactly": False,
            "mirror_symmetry_correction_applied": True,
            "mirror_symmetry_correction_vendor_confirmed": False,
            "pod_isaac_import_verified": pod_import_verified,
            "racket_local_contract_verified": True,
            "standing_pose_verified": False,
            "racket_fk_parity_verified": False,
            "dynamics_parity_verified": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }


def prepare(source_root: Path, output_root: Path, intake_path: Path, manifest_out: Path) -> dict[str, Any]:
    require_isolated_successor_root(output_root, source_root)
    if output_root.exists():
        raise AssetError(f"refusing to overwrite versioned output root: {output_root}")
    intake = load_json(intake_path)
    verify_raw_closure(source_root, intake)
    raw_urdf = source_root / intake["primary_urdf"]["path"]
    raw_root = ET.parse(raw_urdf).getroot()
    normalized_root, diff = normalize(raw_root, source_root / "meshes")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=str(output_root.parent)))
    urdf_dir = staging_root / "urdf"
    mesh_dir = staging_root / "meshes"
    try:
        urdf_dir.mkdir(exist_ok=False)
        mesh_dir.mkdir(exist_ok=False)
        for ref in sorted(set(mesh_refs(normalized_root))):
            source = exact_mesh_source(source_root / "meshes", ref)
            destination = mesh_dir / Path(ref).name
            shutil.copyfile(source, destination)
            if sha256_path(source) != sha256_path(destination):
                raise AssetError(f"mesh alias copy changed bytes: {source} -> {destination}")
        write_urdf(normalized_root, urdf_dir / "model.urdf")
        manifest = build_manifest(source_root, staging_root, intake_path, normalized_root, diff)
        manifest["output"]["asset_path"] = relative_path(output_root)
        publish_manifest = not manifest_out.exists()
        if not publish_manifest and load_json(manifest_out) != manifest:
            raise AssetError("tracked versioned manifest does not match regenerated candidate")
        if output_root.exists():
            raise AssetError(f"versioned output appeared during build: {output_root}")
        staging_root.rename(output_root)
        if publish_manifest:
            manifest_out.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            fd, temp_name = tempfile.mkstemp(prefix=f".{manifest_out.name}.", dir=str(manifest_out.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temp_name, manifest_out)
            finally:
                Path(temp_name).unlink(missing_ok=True)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return manifest


def compare_closure(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    for key in ("file_count", "total_bytes", "sha256", "files"):
        if observed[key] != expected[key]:
            raise AssetError(f"output closure mismatch at {key}")


def check(source_root: Path, output_root: Path, intake_path: Path, manifest_path: Path) -> dict[str, Any]:
    require_isolated_successor_root(output_root, source_root)
    intake = load_json(intake_path)
    verify_raw_closure(source_root, intake)
    manifest = load_json(manifest_path)
    if manifest.get("manifest_type") != "agibot_a3_p1_0803_31action_normalized_asset_v1":
        raise AssetError("wrong normalized asset manifest type")
    if manifest["source"]["raw_intake_manifest_sha256"] != sha256_path(intake_path):
        raise AssetError("normalized manifest does not bind the current raw intake manifest")
    compare_closure(manifest["output"]["closure"], closure(output_root))

    urdf_path = output_root / manifest["output"]["urdf_path"]
    if sha256_path(urdf_path) != manifest["output"]["urdf_sha256"]:
        raise AssetError("normalized URDF SHA-256 mismatch")
    root = ET.parse(urdf_path).getroot()
    links = [link.get("name") for link in root.findall("link")]
    joints = [joint.get("name") for joint in root.findall("joint")]
    if len(links) != len(set(links)):
        raise AssetError("normalized URDF contains duplicate link names")
    if len(joints) != len(set(joints)):
        raise AssetError("normalized URDF contains duplicate joint names")
    gripper_joint_names, _ = descendants_for_joint(root, GRIPPER_MOUNT_JOINT)
    nonfixed_gripper = [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("name") in set(gripper_joint_names) and joint.get("type") != "fixed"
    ]
    if nonfixed_gripper:
        raise AssetError(f"normalized URDF retains movable left-gripper joints: {nonfixed_gripper}")
    if len(movable_joint_names(root)) != EXPECTED_MOVABLE_JOINTS:
        raise AssetError("normalized URDF is not 31 movable joints")
    if set(movable_joint_names(root)) != set(read_order(RUNTIME_JOINT_ORDER)):
        raise AssetError("normalized joint set drifted from runtime action ABI")
    if parse_rgba(root):
        raise AssetError("normalized URDF contains invalid rgba")
    validate_importer_safe_axes_and_meshes(root)

    refs = mesh_refs(root)
    for ref in refs:
        exact_mesh_source(output_root / "meshes", ref)
    if not REQUIRED_RACKET_MESHES.issubset({Path(ref).name for ref in refs}):
        raise AssetError("normalized URDF lacks required racket mesh references")

    source_root_xml = ET.parse(source_root / intake["primary_urdf"]["path"]).getroot()
    regenerated, regenerated_diff = normalize(source_root_xml, source_root / "meshes")
    if regenerated_diff != manifest["normalization_diff"]:
        raise AssetError("normalization diff does not reproduce from the bound raw source")
    with tempfile.TemporaryDirectory(prefix="a3_p1_0803_31d_check_") as temp_dir:
        regenerated_path = Path(temp_dir) / "model.urdf"
        write_urdf(regenerated, regenerated_path)
        if regenerated_path.read_bytes() != urdf_path.read_bytes():
            raise AssetError("normalized URDF bytes do not reproduce from the bound raw source")
    expected_manifest = build_manifest(
        source_root, output_root, intake_path, root, regenerated_diff
    )
    if expected_manifest != manifest:
        raise AssetError("normalized manifest does not fully reproduce from current producer/source/output")
    return {
        "status": "PASS",
        "movable_joint_count": len(movable_joint_names(root)),
        "link_count": len(links),
        "joint_count": len(joints),
        "mesh_reference_count": len(refs),
        "unique_mesh_count": len(set(refs)),
        "output_closure_sha256": manifest["output"]["closure"]["sha256"],
        "urdf_sha256": manifest["output"]["urdf_sha256"],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-intake-manifest", type=Path, default=RAW_INTAKE_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SUCCESSOR_MANIFEST)
    parser.add_argument("--check", action="store_true", help="Verify the existing ignored output and tracked manifest.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            report = check(
                args.source_root.resolve(),
                args.output_root.resolve(),
                args.raw_intake_manifest.resolve(),
                args.manifest.resolve(),
            )
        else:
            manifest = prepare(
                args.source_root.resolve(),
                args.output_root.resolve(),
                args.raw_intake_manifest.resolve(),
                args.manifest.resolve(),
            )
            report = {
                "status": "PREPARED",
                "output_root": args.output_root.as_posix(),
                "manifest": args.manifest.as_posix(),
                "closure_sha256": manifest["output"]["closure"]["sha256"],
            }
        print(json.dumps(report, sort_keys=True))
        return 0
    except (AssetError, FileNotFoundError, ET.ParseError, KeyError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
