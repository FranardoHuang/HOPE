#!/usr/bin/env python3
"""Build the 0807 A3P-P1 model set: one Isaac Lab asset and one MuJoCo MJCF, from one raw bundle.

Both engines are derived from the same immutable bundle assembled by
``scripts/intake_a3p_p1_0807_bundle.py`` so they cannot silently disagree, and every deviation from
delivered bytes is declared in the receipt rather than folded in silently.

Isaac output -- a normalised URDF + mesh closure:
  * the duplicate ``imu_in_pelvis_link`` is de-duplicated, keeping the copy that carries the
    delivered dedicated collision mesh;
  * the geometry-less NaN-rgba visual on ``left_base_footprint`` is dropped;
  * the four hyphenated gripper mesh basenames get deterministic USD-safe underscore aliases;
  * the nine non-policy gripper coordinates are locked at their URDF ``q=0`` and retyped fixed, so
    the 31-action ABI is preserved exactly;
  * every ``*_collision`` reference whose bytes equal its visual twin simply points at the visual
    file, so no identical mesh is stored twice; that includes the twenty absent gripper collisions,
    which resolve to their visual under the vendor's recorded "collision is the visual" confirmation
    -- nothing is fabricated and nothing is duplicated, and the equality is SHA-256 checked;
  * the mixed-case ``_Link`` body-name ABI is restored.

MuJoCo output -- a new versioned MJCF derived from the in-service ``a3_pingpong.xml``:
  * the incumbent file is never edited; its bytes are SHA-pinned by four consumers;
  * eight ``<body pos>`` values are re-pointed at the 0807 joint origins;
  * seven ``<inertial>`` blocks are re-derived from 0807 under the MJCF's own fixed-merge policy;
  * the retired ``left_hand_Link`` geometry is replaced by the twenty OP3 gripper meshes carried as
    geoms **of ``left_wrist_yaw_Link`` itself** at their ``q=0`` poses, so the 32-body / 31-actuator
    ABI, the keyframe width and the contact-exclusion set are all untouched;
  * every armature, damping, frictionloss, actuator, convex hull, site, sensor and racket constant
    is copied verbatim -- none of it is derivable from any URDF.

NOT VERIFIED HERE: this host has no ``mujoco``, so the emitted MJCF is validated structurally only.
It has never been compiled, never been loaded by ``mujoco_warp``, and carries no identity manifest.
The receipt says so; do not read "PREPARED" as "runs".
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "vendor_assets" / "agibot" / "A3P-P1-32dof-0807-OP3+pingpang"
BUNDLE_RECEIPT = REPO_ROOT / "configs" / "a3p_p1_0807_raw_intake_v1.json"

ISAAC_ASSET_ROOT = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "agibot_a3p_p1_0807_v1"
)
ACTIVE_ISAAC_ASSET_ROOT = ISAAC_ASSET_ROOT.parent / "agibot_a3"
# The tracked robot-description package, sitting beside the existing A3T2.5 one.  Same bytes as the
# ignored Isaac asset: this is the reviewable copy that survives a fresh clone.
TRACKED_PACKAGE_ROOT = REPO_ROOT / "agi" / "URDF" / "A3P-P1-32dof-0807-OP3-pingpang"

MJCF_MODEL_DIR = (
    REPO_ROOT
    / "agi"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
)
INCUMBENT_MJCF_DIR = MJCF_MODEL_DIR / "a3_pingpong"
INCUMBENT_MJCF = INCUMBENT_MJCF_DIR / "a3_pingpong.xml"
INCUMBENT_MJCF_SHA256 = "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a"
MJCF_OUTPUT_DIR = MJCF_MODEL_DIR / "a3p_pingpong_0807"
MJCF_OUTPUT_NAME = "a3p_pingpong_0807.xml"

MODEL_SET_RECEIPT = REPO_ROOT / "configs" / "a3p_p1_0807_model_set_v1.json"

RUNTIME_JOINT_ORDER = REPO_ROOT / "configs" / "a3_runtime_articulation_joint_order.txt"
RUNTIME_BODY_ORDER = REPO_ROOT / "configs" / "a3_runtime_body_order.txt"
GMR_JOINT_ORDER = REPO_ROOT / "configs" / "a3_gmr_dof_pos_joint_order.txt"

GRIPPER_MOUNT_JOINT = "left_OP3_joint"
GRIPPER_HOST_LINK = "left_wrist_yaw_link"
RAW_DUPLICATE_LINK = "imu_in_pelvis_link"
RETIRED_LEFT_HAND_LINK = "left_hand_Link"

EXPECTED_MOVABLE_JOINTS = 31
EXPECTED_GRIPPER_MOVABLE_JOINTS = 9
EXPECTED_GRIPPER_SUBTREE_MASS_KG = 0.76626209416
EXPECTED_NORMALIZED_UNIQUE_LINK_MASS_KG = 57.60001015416
EXPECTED_MATERIALISED_COLLISION_COUNT = 20
EXPECTED_MJCF_BODY_COUNT = 32
EXPECTED_MJCF_ACTUATOR_COUNT = 31
EXPECTED_MJCF_KEYFRAME_QPOS_WIDTH = 38

USD_SAFE_MESH_ALIASES = {
    "Link4-1.stl": "Link4_1.stl",
    "Link7-1.stl": "Link7_1.stl",
    "Link11-1.stl": "Link11_1.stl",
    "Link14-1.stl": "Link14_1.stl",
    "Link4-1_collision.stl": "Link4_1_collision.stl",
    "Link7-1_collision.stl": "Link7_1_collision.stl",
    "Link11-1_collision.stl": "Link11_1_collision.stl",
    "Link14-1_collision.stl": "Link14_1_collision.stl",
}

OFFICIAL_RACKET_SITE_XYZ_M = (0.21021, 0.032078, 0.032036)
OFFICIAL_RACKET_SITE_RPY_RAD = (0.0, 0.0, 0.0)
REQUIRED_MJCF_RACKET_NAMES = ("right_racket", "right_racket_collision", "right_racket_handle_collision")

# Straight from configs/a3p_p1_0807_raw_intake_v1.json; restated here so the substitution is
# refused rather than performed if the intake receipt ever stops carrying it.
GRIPPER_COLLISION_EQUALS_VISUAL_CONTRACT = {
    "schema_version": 1,
    "authority": "vendor_verbal_confirmation_20260807",
    "vendor_written_confirmation_on_file": False,
    "statement": "gripper collision geometry is identical to the gripper visual geometry",
    "applies_to": "the twenty left OmniPicker3 gripper links only",
    "method": "the collision element references the visual mesh file itself; SHA-256 equality checked",
    "supersedes": "the 0803 collision-disabled gripper contract",
    "not_covered": (
        "gripper joint coupling, neutral/home pose, and the eight placeholder finger limits remain "
        "unanswered; the nine gripper coordinates stay project-locked at q=0"
    ),
}

PROJECT_GRIPPER_LOCK_CONTRACT = {
    "schema_version": 2,
    "authority": "project_owned_training_projection_20260807",
    "locked_joint_position": "all_nine_gripper_movable_coordinates_q_equal_zero",
    "q0_claim_scope": "project_lock_pose_not_vendor_neutral_or_hardware_home",
    "gripper_policy_controlled": False,
    "isaac_representation": "nine joints retyped fixed; importer merge folds the subtree into left_wrist_yaw_Link",
    "mujoco_representation": (
        "no gripper bodies or joints are added; the subtree's mass/inertia is folded into "
        "left_wrist_yaw_Link and its twenty meshes are carried as geoms of that same body at their "
        "q=0 poses, so the 32-body and 31-actuator ABI is bit-preserved"
    ),
}

# MuJoCo's STL decoder refuses any mesh above this face count, so a delivered mesh that exceeds it
# cannot be carried verbatim into an MJCF no matter how much we would like to.
MUJOCO_MAX_MESH_FACES = 200000

# The only geometry deviations from the delivered 0807 URDF, and why each is taken.
MJCF_DECLARED_DEVIATIONS = {
    "oversize_mesh_convex_hull": (
        "MuJoCo's STL decoder hard-rejects meshes above "
        f"{MUJOCO_MAX_MESH_FACES} faces, and the delivered gripper base_link.stl carries 313,284. "
        "Only meshes over that engine limit are replaced, and only by their own convex hull -- "
        "which is what MuJoCo computes for collision anyway -- so the substitution is forced by the "
        "engine rather than chosen, and it is recorded per file with both face counts. The Isaac "
        "asset keeps the delivered mesh verbatim"
    ),
    "pelvis_imu_fold": (
        "the incumbent MJCF folds imu_in_torso_link into torso but omits imu_in_pelvis_link (0.02 kg) "
        "from pelvis; Isaac's merge_fixed_joints folds it, so the new MJCF folds it too -- this "
        "corrects a 20 g cross-engine parity gap that predates 0807"
    ),
}


class ModelSetError(RuntimeError):
    """Raised when an input or a derived artifact violates the declared contract."""


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def closure(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def read_order(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def normalized_link_name(raw_name: str) -> str:
    if raw_name == "pelvis_link":
        return raw_name
    if raw_name.endswith("_link"):
        return raw_name[:-5] + "_Link"
    return raw_name


def fmt(value: float) -> str:
    """Render a float the way MJCF does: short, round-trippable, no scientific noise."""
    if value == 0:
        return "0"
    text = repr(float(f"{value:.9g}"))
    return text[:-2] if text.endswith(".0") else text


def fmt_vec(values: Iterable[float]) -> str:
    return " ".join(fmt(float(v)) for v in values)


# --------------------------------------------------------------------------------------
# kinematics
# --------------------------------------------------------------------------------------


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    ca, sa = math.cos(roll), math.sin(roll)
    cb, sb = math.cos(pitch), math.sin(pitch)
    cc, sc = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cb * cc, sa * sb * cc - ca * sc, ca * sb * cc + sa * sc],
            [cb * sc, sa * sb * sc + ca * cc, ca * sb * sc - sa * cc],
            [-sb, sa * cb, ca * cb],
        ]
    )


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion order (w, x, y, z), sign-normalised to w >= 0."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ]
            )
        elif index == 1:
            s = math.sqrt(1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ]
            )
        else:
            s = math.sqrt(1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ]
            )
    quat = quat / np.linalg.norm(quat)
    return -quat if quat[0] < 0 else quat


class Urdf:
    """Read-only view over a URDF, with the transforms both engines need."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = ET.parse(path).getroot()
        self.links: dict[str, ET.Element] = {}
        for link in self.root.findall("link"):
            # Keep the LAST definition: for the duplicated imu link that is the one carrying the
            # delivered dedicated collision mesh.
            self.links[link.get("name")] = link
        self.joints = list(self.root.findall("joint"))
        self.by_name = {joint.get("name"): joint for joint in self.joints}
        self.children: dict[str, list[ET.Element]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.find("parent").get("link"), []).append(joint)

    def joint_transform(self, joint: ET.Element) -> tuple[np.ndarray, np.ndarray]:
        origin = joint.find("origin")
        if origin is None:
            return np.eye(3), np.zeros(3)
        xyz = origin.get("xyz")
        rpy = origin.get("rpy")
        translation = np.array([float(v) for v in xyz.split()]) if xyz else np.zeros(3)
        rotation = rpy_to_matrix(*[float(v) for v in rpy.split()]) if rpy else np.eye(3)
        return rotation, translation

    def inertial(self, name: str) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
        link = self.links.get(name)
        element = None if link is None else link.find("inertial")
        if element is None:
            return None
        origin = element.find("origin")
        attrib = element.find("inertia").attrib
        com = np.array([float(v) for v in origin.get("xyz").split()])
        rotation = (
            rpy_to_matrix(*[float(v) for v in origin.get("rpy").split()])
            if origin.get("rpy")
            else np.eye(3)
        )
        tensor = np.array(
            [
                [float(attrib["ixx"]), float(attrib["ixy"]), float(attrib["ixz"])],
                [float(attrib["ixy"]), float(attrib["iyy"]), float(attrib["iyz"])],
                [float(attrib["ixz"]), float(attrib["iyz"]), float(attrib["izz"])],
            ]
        )
        return float(element.find("mass").get("value")), com, rotation @ tensor @ rotation.T

    def welded_subtree(self, root_link: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Every link rigidly attached to ``root_link`` at q=0, with its pose in that frame.

        "Rigidly attached" means reached only through fixed joints or through the nine
        project-locked gripper coordinates; a real joint stops the walk.
        """
        found: list[tuple[str, np.ndarray, np.ndarray]] = []
        stack = [(root_link, np.eye(3), np.zeros(3))]
        while stack:
            name, rotation, translation = stack.pop()
            found.append((name, rotation, translation))
            for joint in self.children.get(name, []):
                locked = joint.get("type") == "fixed" or joint.get("name").startswith("left_joint")
                if not locked:
                    continue
                child_rotation, child_translation = self.joint_transform(joint)
                stack.append(
                    (
                        joint.find("child").get("link"),
                        rotation @ child_rotation,
                        translation + rotation @ child_translation,
                    )
                )
        return found

    def merged_inertial(self, root_link: str) -> Optional[dict[str, Any]]:
        """Fixed-merge inertial of ``root_link``, in MuJoCo's pos/quat/diaginertia form."""
        parts = []
        for name, rotation, translation in self.welded_subtree(root_link):
            value = self.inertial(name)
            if value is None:
                continue
            mass, com, tensor = value
            parts.append((mass, translation + rotation @ com, rotation @ tensor @ rotation.T))
        total = math.fsum(part[0] for part in parts)
        if total <= 0.0:
            return None
        com = sum(part[0] * part[1] for part in parts) / total
        tensor = np.zeros((3, 3))
        for mass, position, part_tensor in parts:
            offset = position - com
            tensor = tensor + part_tensor + mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] = -eigenvectors[:, 0]
        if float(eigenvalues.min()) <= 0.0:
            raise ModelSetError(f"non-positive principal inertia on {root_link}: {eigenvalues}")
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        residual = float(np.abs(reconstructed - tensor).max())
        if residual > 1e-9:
            raise ModelSetError(f"eigendecomposition did not reproduce {root_link}: {residual}")
        return {
            "mass": total,
            "pos": com,
            "quat": matrix_to_quat(eigenvectors),
            "diaginertia": eigenvalues,
            "member_links": [name for name, _, _ in self.welded_subtree(root_link)],
            "reconstruction_residual": residual,
        }

    def child_origin(self, child_link: str) -> Optional[np.ndarray]:
        for joint in self.joints:
            if joint.find("child").get("link") == child_link:
                origin = joint.find("origin")
                if origin is None or not origin.get("xyz"):
                    return np.zeros(3)
                return np.array([float(v) for v in origin.get("xyz").split()])
        return None


# --------------------------------------------------------------------------------------
# Isaac asset
# --------------------------------------------------------------------------------------


def indent_xml(element: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(element):
        if not (element.text or "").strip():
            element.text = pad + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (element.tail or "").strip():
        element.tail = pad


def build_isaac_asset(urdf: Urdf, bundle_meshes: Path, staging: Path) -> dict[str, Any]:
    root = copy.deepcopy(urdf.root)

    gripper_joints = []
    gripper_links = set()
    stack = [GRIPPER_MOUNT_JOINT]
    while stack:
        name = stack.pop()
        joint = urdf.by_name[name]
        child = joint.find("child").get("link")
        gripper_links.add(child)
        if name != GRIPPER_MOUNT_JOINT:
            gripper_joints.append(name)
        for nxt in urdf.children.get(child, []):
            stack.append(nxt.get("name"))

    locked = [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("name").startswith("left_joint") and joint.get("type") not in {"fixed", "floating"}
    ]
    if len(locked) != EXPECTED_GRIPPER_MOVABLE_JOINTS:
        raise ModelSetError(f"expected {EXPECTED_GRIPPER_MOVABLE_JOINTS} movable gripper joints, found {locked}")
    locked_originals = {}
    for joint in root.findall("joint"):
        if joint.get("name") not in locked:
            continue
        limit = joint.find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if not (lower <= 0.0 <= upper):
            raise ModelSetError(f"project q=0 lock outside delivered limits for {joint.get('name')}")
        locked_originals[joint.get("name")] = {
            "type": joint.get("type"),
            "limit": dict(sorted(limit.attrib.items())),
        }
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
            child_element = joint.find(tag)
            if child_element is not None:
                joint.remove(child_element)

    duplicates = [link for link in root.findall("link") if link.get("name") == RAW_DUPLICATE_LINK]
    if len(duplicates) != 2:
        raise ModelSetError(f"expected two {RAW_DUPLICATE_LINK!r} definitions, found {len(duplicates)}")
    root.remove(duplicates[0])

    removed_visuals = []
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            rgba = visual.find(".//*[@rgba]")
            has_nan = rgba is not None and "nan" in (rgba.get("rgba") or "").lower()
            if visual.find("geometry") is None or has_nan:
                removed_visuals.append({"link": link.get("name"), "reason": "no_geometry_or_nonfinite_rgba"})
                link.remove(visual)
    if len(removed_visuals) != 1:
        raise ModelSetError(f"expected exactly one malformed visual, removed {removed_visuals}")

    name_map = {}
    for link in root.findall("link"):
        raw = link.get("name")
        name_map[raw] = raw if raw in gripper_links else normalized_link_name(raw)
    for link in root.findall("link"):
        link.set("name", name_map[link.get("name")])
    for joint in root.findall("joint"):
        for tag in ("parent", "child"):
            endpoint = joint.find(tag)
            endpoint.set("link", name_map[endpoint.get("link")])
    root.set("name", "A3P-P1-0807-OP3-pingpang-31action-normalized-v1")

    on_disk = {path.name for path in bundle_meshes.iterdir() if path.is_file()}
    materialised = []
    rewrites = []
    deduplicated = []
    mesh_dir = staging / "meshes"
    mesh_dir.mkdir()
    for mesh in root.findall(".//mesh"):
        ref = mesh.get("filename")
        base = Path(ref).name
        source_base = base
        if base not in on_disk:
            if not base.endswith("_collision.stl"):
                raise ModelSetError(f"unresolvable non-collision mesh reference: {ref}")
            source_base = base.replace("_collision", "")
            if source_base not in on_disk:
                raise ModelSetError(f"materialisable collision {base} has no visual source")
        # The delivery ships every *_collision.stl as an exact byte copy of its visual twin, and
        # the twenty gripper collisions are materialised as copies by the vendor's own
        # confirmation.  Storing both halves of an identical pair twice would put ~42 MB of
        # duplicate bytes into git for no geometric content, so a byte-identical collision simply
        # references the visual file.  Only exact SHA-256 equality qualifies.
        visual_twin = base.replace("_collision", "") if base.endswith("_collision.stl") else None
        if (
            visual_twin
            and visual_twin in on_disk
            and sha256_path(bundle_meshes / source_base) == sha256_path(bundle_meshes / visual_twin)
        ):
            twin_alias = USD_SAFE_MESH_ALIASES.get(visual_twin, visual_twin)
            delivered = base in on_disk
            deduplicated.append(
                {
                    "collision_reference": base,
                    "points_at": twin_alias,
                    "sha256": sha256_path(bundle_meshes / visual_twin),
                    "byte_identical": True,
                    "kind": "delivered_collision_equal_to_visual" if delivered
                            else "gripper_collision_authorised_as_visual",
                }
            )
            base = visual_twin
            source_base = visual_twin
        alias = USD_SAFE_MESH_ALIASES.get(base, base)
        destination = mesh_dir / alias
        if not destination.exists():
            shutil.copyfile(bundle_meshes / source_base, destination)
            if sha256_path(destination) != sha256_path(bundle_meshes / source_base):
                raise ModelSetError(f"mesh copy changed bytes: {base}")
            if source_base != base:
                materialised.append(
                    {
                        # Both names are the POST-alias, on-disk ones: a hyphenated gripper mesh is
                        # stored under its underscore alias, so recording the delivered spelling
                        # here would make the receipt's own equality check unresolvable.
                        "collision_basename": alias,
                        "visual_basename": USD_SAFE_MESH_ALIASES.get(source_base, source_base),
                        "delivered_collision_reference": base,
                        "delivered_visual_basename": source_base,
                        "sha256": sha256_path(destination),
                        "byte_identical_to_visual": True,
                    }
                )
        if alias != base:
            rewrites.append({"raw": base, "normalized": alias})
        mesh.set("filename", f"../meshes/{alias}")

    gripper_rows = [r for r in deduplicated if r["kind"] == "gripper_collision_authorised_as_visual"]
    if len(gripper_rows) != EXPECTED_MATERIALISED_COLLISION_COUNT:
        raise ModelSetError(
            f"expected {EXPECTED_MATERIALISED_COLLISION_COUNT} gripper collision references, got {len(gripper_rows)}"
        )
    if materialised:
        raise ModelSetError("no collision mesh should be written as a separate copy under dedup")
    for row in deduplicated:
        target = mesh_dir / row["points_at"]
        if not target.is_file() or sha256_path(target) != row["sha256"]:
            raise ModelSetError(f"deduplicated collision target missing or drifted: {row['points_at']}")

    movable = [j.get("name") for j in root.findall("joint") if j.get("type") not in {"fixed", "floating"}]
    if len(movable) != EXPECTED_MOVABLE_JOINTS:
        raise ModelSetError(f"normalized asset has {len(movable)} movable joints, expected {EXPECTED_MOVABLE_JOINTS}")
    if set(movable) != set(read_order(RUNTIME_JOINT_ORDER)):
        raise ModelSetError("normalized movable joint set does not equal the runtime 31-action ABI")
    if movable != read_order(GMR_JOINT_ORDER):
        raise ModelSetError("normalized movable document order does not equal the GMR order")
    link_names = {link.get("name") for link in root.findall("link")}
    missing_bodies = sorted(set(read_order(RUNTIME_BODY_ORDER)) - link_names)
    if missing_bodies:
        raise ModelSetError(f"normalized asset misses runtime body ABI names: {missing_bodies}")

    masses = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            masses.setdefault(link.get("name"), float(inertial.find("mass").get("value")))
    total = math.fsum(masses.values())
    if abs(total - EXPECTED_NORMALIZED_UNIQUE_LINK_MASS_KG) > 1e-9:
        raise ModelSetError(f"normalized unique-link mass drifted: {total}")

    racket = next(j for j in root.findall("joint") if j.get("name") == "pingpang_red_joint")
    origin = racket.find("origin")
    if tuple(float(v) for v in origin.get("xyz").split()) != OFFICIAL_RACKET_SITE_XYZ_M:
        raise ModelSetError("official racket-site local transform drifted")
    if tuple(float(v) for v in origin.get("rpy").split()) != OFFICIAL_RACKET_SITE_RPY_RAD:
        raise ModelSetError("official racket-site rpy drifted")

    (staging / "urdf").mkdir()
    indent_xml(root)
    ET.ElementTree(root).write(staging / "urdf" / "model.urdf", encoding="utf-8", xml_declaration=True)

    return {
        "robot_name": root.get("name"),
        "link_count": len(root.findall("link")),
        "joint_count": len(root.findall("joint")),
        "movable_joint_count": len(movable),
        "movable_joint_document_order": movable,
        "unique_link_total_mass_kg": total,
        "removed_duplicate_link": {
            "name": RAW_DUPLICATE_LINK,
            "removed_occurrence": "first_document_occurrence",
            "retained_occurrence": "last_document_occurrence_with_dedicated_collision_mesh",
        },
        "removed_malformed_visual_elements": removed_visuals,
        "project_locked_gripper_joints": {
            "names": locked,
            "delivered_originals": locked_originals,
            "lock_contract": PROJECT_GRIPPER_LOCK_CONTRACT,
        },
        "materialised_gripper_collision_meshes": materialised,
        "collision_references_resolved_to_visual": deduplicated,
        "gripper_collision_reference_count": len(gripper_rows),
        "deduplication_note": (
            "every *_collision reference whose bytes equal its visual twin points at the single "
            "visual file instead of a second identical copy; the twenty gripper collisions do the "
            "same under the vendor confirmation, so no collision byte is ever fabricated or stored twice"
        ),
        "gripper_collision_contract": GRIPPER_COLLISION_EQUALS_VISUAL_CONTRACT,
        "usd_safe_mesh_aliases": rewrites,
        "link_name_map": name_map,
    }


# --------------------------------------------------------------------------------------
# MuJoCo MJCF
# --------------------------------------------------------------------------------------


def mjcf_identifier(basename: str) -> str:
    return Path(basename).stem.replace("-", "_")


def read_binary_stl(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ModelSetError(f"not a binary STL: {path}")
    count = int(np.frombuffer(payload[80:84], dtype="<u4")[0])
    if 84 + count * 50 != len(payload):
        raise ModelSetError(f"STL is not binary or is truncated: {path}")
    records = np.frombuffer(payload[84 : 84 + count * 50], dtype=np.uint8).reshape(count, 50)
    return np.frombuffer(records[:, 12:48].tobytes(), dtype="<f4").reshape(count, 3, 3).astype(np.float64)


def stl_face_count(path: Path) -> int:
    payload = path.read_bytes()
    if len(payload) < 84:
        return 0
    count = int(np.frombuffer(payload[80:84], dtype="<u4")[0])
    return count if 84 + count * 50 == len(payload) else 0


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    count = triangles.shape[0]
    out = bytearray(b"\0" * 80)
    out += int(count).to_bytes(4, "little")
    for triangle in triangles:
        edge_a = triangle[1] - triangle[0]
        edge_b = triangle[2] - triangle[0]
        normal = np.cross(edge_a, edge_b)
        length = float(np.linalg.norm(normal))
        normal = normal / length if length > 0 else np.zeros(3)
        out += np.asarray(normal, dtype="<f4").tobytes()
        out += np.asarray(triangle, dtype="<f4").tobytes()
        out += b"\0\0"
    path.write_bytes(bytes(out))


def convex_hull_stl(source: Path, destination: Path) -> dict[str, Any]:
    """Replace an over-limit mesh by its own convex hull, keeping the source frame."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ModelSetError(f"convex hull needs scipy, which is unavailable: {exc}")
    triangles = read_binary_stl(source)
    points = np.unique(triangles.reshape(-1, 3), axis=0)
    hull = ConvexHull(points)
    # ConvexHull simplices are not consistently wound; orient each outward from the centroid so the
    # emitted normals are not garbage.
    centroid = points[hull.vertices].mean(axis=0)
    faces = []
    for simplex in hull.simplices:
        tri = points[simplex]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.dot(normal, tri.mean(axis=0) - centroid) < 0:
            tri = tri[[0, 2, 1]]
        faces.append(tri)
    faces = np.asarray(faces)
    write_binary_stl(destination, faces)
    return {
        "source_basename": source.name,
        "source_faces": int(triangles.shape[0]),
        "hull_basename": destination.name,
        "hull_faces": int(faces.shape[0]),
        "hull_volume_m3": float(hull.volume),
        "reason": f"delivered mesh exceeds MuJoCo's {MUJOCO_MAX_MESH_FACES}-face decoder limit",
    }


def build_mjcf(urdf: Urdf, bundle_meshes: Path, staging: Path) -> dict[str, Any]:
    observed = sha256_path(INCUMBENT_MJCF)
    if observed != INCUMBENT_MJCF_SHA256:
        raise ModelSetError(
            "the in-service MJCF this derivation is based on has drifted: "
            f"{observed} != {INCUMBENT_MJCF_SHA256}"
        )
    tree = ET.parse(INCUMBENT_MJCF)
    root = tree.getroot()
    root.set("model", "A3P_P1_0807_OP3_pingpang")

    bodies = {body.get("name"): body for body in root.iter("body")}
    if len(bodies) != EXPECTED_MJCF_BODY_COUNT:
        raise ModelSetError(f"incumbent MJCF has {len(bodies)} bodies, expected {EXPECTED_MJCF_BODY_COUNT}")

    # 1. re-point every <body pos> at the delivered 0807 joint origin
    pos_updates = []
    for name, body in bodies.items():
        if body.get("pos") is None:
            continue
        origin = urdf.child_origin(name if name == "pelvis_link" else name[:-5] + "_link")
        if origin is None:
            continue
        before = np.array([float(v) for v in body.get("pos").split()])
        delta = float(np.linalg.norm(origin - before))
        if delta <= 0.0:
            continue
        body.set("pos", fmt_vec(origin))
        pos_updates.append(
            {
                "body": name,
                "incumbent_pos": fmt_vec(before),
                "delivered_pos": fmt_vec(origin),
                "delta_m": delta,
                "classification": "geometry_change" if delta > 1.0e-5 else "coordinate_rounding",
            }
        )

    # 2. re-derive every <inertial> under the MJCF's own fixed-merge policy, rewriting only those
    #    that actually moved so the diff stays readable and reviewable
    inertial_updates = []
    for name, body in bodies.items():
        element = body.find("inertial")
        if element is None:
            continue
        merged = urdf.merged_inertial(name if name == "pelvis_link" else name[:-5] + "_link")
        if merged is None:
            raise ModelSetError(f"0807 has no inertial for MJCF body {name}")
        before_mass = float(element.get("mass"))
        before_pos = np.array([float(v) for v in element.get("pos").split()])
        d_mass = abs(merged["mass"] - before_mass)
        d_pos = float(np.linalg.norm(merged["pos"] - before_pos))
        if d_mass <= 1.0e-5 and d_pos <= 1.0e-5:
            continue
        record = {
            "body": name,
            "incumbent_mass_kg": before_mass,
            "delivered_mass_kg": merged["mass"],
            "delta_mass_kg": merged["mass"] - before_mass,
            "delta_com_m": d_pos,
            "merged_link_count": len(merged["member_links"]),
            "eigendecomposition_residual": merged["reconstruction_residual"],
        }
        element.set("mass", fmt(merged["mass"]))
        element.set("pos", fmt_vec(merged["pos"]))
        element.set("quat", fmt_vec(merged["quat"]))
        element.set("diaginertia", fmt_vec(merged["diaginertia"]))
        if element.get("fullinertia") is not None:
            del element.attrib["fullinertia"]
        inertial_updates.append(record)

    # 3. retire the left_hand geometry and carry the OP3 gripper as geoms of its host body
    asset = root.find("asset")
    retired_assets = []
    for mesh in list(asset.findall("mesh")):
        if RETIRED_LEFT_HAND_LINK in (mesh.get("name") or "") or RETIRED_LEFT_HAND_LINK in (mesh.get("file") or ""):
            retired_assets.append({"name": mesh.get("name"), "file": mesh.get("file")})
            asset.remove(mesh)
    retired_names = {row["name"] for row in retired_assets}
    host = bodies["left_wrist_yaw_Link"]
    retired_geoms = []
    for geom in list(host.findall("geom")):
        if geom.get("mesh") in retired_names:
            retired_geoms.append({"name": geom.get("name"), "mesh": geom.get("mesh")})
            host.remove(geom)
    if not retired_assets or not retired_geoms:
        raise ModelSetError("expected to retire left_hand mesh assets and geoms, found none")

    mesh_dir = staging / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    gripper_geoms = []
    hull_substitutions: list[dict[str, Any]] = []
    for link_name, rotation, translation in sorted(
        urdf.welded_subtree(GRIPPER_HOST_LINK), key=lambda row: row[0]
    ):
        if link_name == GRIPPER_HOST_LINK:
            continue
        link = urdf.links[link_name]
        visual = None
        mesh_element = None
        for candidate in link.findall("visual"):
            found = candidate.find("geometry/mesh")
            if found is not None:
                visual, mesh_element = candidate, found
                break
        if mesh_element is None:
            continue
        basename = Path(mesh_element.get("filename")).name
        source = bundle_meshes / basename
        if not source.is_file():
            raise ModelSetError(f"gripper visual mesh missing from bundle: {basename}")
        alias = USD_SAFE_MESH_ALIASES.get(basename, basename)
        faces = stl_face_count(source)
        if faces > MUJOCO_MAX_MESH_FACES:
            alias = Path(alias).stem + "_convexhull.stl"
            destination = mesh_dir / alias
            if not destination.exists():
                hull_substitutions.append(convex_hull_stl(source, destination))
        else:
            destination = mesh_dir / alias
            if not destination.exists():
                shutil.copyfile(source, destination)
                if sha256_path(destination) != sha256_path(source):
                    raise ModelSetError(f"gripper mesh copy changed bytes: {basename}")
        identifier = mjcf_identifier(alias)
        offset = visual.find("origin")
        local_r, local_t = np.eye(3), np.zeros(3)
        if offset is not None:
            if offset.get("xyz"):
                local_t = np.array([float(v) for v in offset.get("xyz").split()])
            if offset.get("rpy"):
                local_r = rpy_to_matrix(*[float(v) for v in offset.get("rpy").split()])
        pose_r = rotation @ local_r
        pose_t = translation + rotation @ local_t
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"op3_{identifier}", "content_type": "model/stl", "file": alias},
        )
        common = {"type": "mesh", "mesh": f"op3_{identifier}", "pos": fmt_vec(pose_t), "quat": fmt_vec(matrix_to_quat(pose_r))}
        ET.SubElement(host, "geom", dict(common, rgba="1 1 1 1"))
        ET.SubElement(
            host,
            "geom",
            dict(common, **{"name": f"op3_{identifier}_collision", "class": "collision"}),
        )
        gripper_geoms.append(
            {
                "link": link_name,
                "mesh_basename": alias,
                "mesh_sha256": sha256_path(destination),
                "mesh_faces": stl_face_count(destination),
                "is_convex_hull_substitute": alias.endswith("_convexhull.stl"),
                "pos": fmt_vec(pose_t),
                "quat": fmt_vec(matrix_to_quat(pose_r)),
                "collision_is_visual_hull": True,
            }
        )
    if len(gripper_geoms) != EXPECTED_MATERIALISED_COLLISION_COUNT:
        raise ModelSetError(f"expected {EXPECTED_MATERIALISED_COLLISION_COUNT} gripper geoms, built {len(gripper_geoms)}")

    # 4. copy every remaining referenced mesh verbatim from the incumbent model
    for mesh in asset.findall("mesh"):
        rel = mesh.get("file")
        if (mesh.get("name") or "").startswith("op3_"):
            continue
        source = INCUMBENT_MJCF_DIR / "meshes" / rel
        if not source.is_file():
            raise ModelSetError(f"incumbent mesh missing: {source}")
        destination = mesh_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
            if sha256_path(destination) != sha256_path(source):
                raise ModelSetError(f"incumbent mesh copy changed bytes: {rel}")

    # 5. structural invariants that both lanes depend on
    joints = [j for j in root.iter("joint") if j.get("name")]
    motors = root.findall("actuator/motor")
    if len(joints) != EXPECTED_MOVABLE_JOINTS:
        raise ModelSetError(f"derived MJCF has {len(joints)} joints, expected {EXPECTED_MOVABLE_JOINTS}")
    if len(motors) != EXPECTED_MJCF_ACTUATOR_COUNT:
        raise ModelSetError(f"derived MJCF has {len(motors)} actuators, expected {EXPECTED_MJCF_ACTUATOR_COUNT}")
    if len(list(root.iter("body"))) != EXPECTED_MJCF_BODY_COUNT:
        raise ModelSetError("derived MJCF body count drifted")
    if len(root.findall(".//freejoint")) != 1:
        raise ModelSetError("derived MJCF must keep exactly one freejoint")
    key = root.find("keyframe/key")
    if key is None or len(key.get("qpos").split()) != EXPECTED_MJCF_KEYFRAME_QPOS_WIDTH:
        raise ModelSetError("derived MJCF keyframe qpos width drifted")
    names = {element.get("name") for element in root.iter() if element.get("name")}
    missing_racket = [name for name in REQUIRED_MJCF_RACKET_NAMES if name not in names]
    if missing_racket:
        raise ModelSetError(f"derived MJCF lost required racket names: {missing_racket}")
    site = root.find(".//site[@name='right_racket']")
    if tuple(float(v) for v in site.get("pos").split()) != OFFICIAL_RACKET_SITE_XYZ_M:
        raise ModelSetError("derived MJCF racket site moved")
    declared = {row["file"] for row in [{"file": m.get("file")} for m in asset.findall("mesh")]}
    on_disk = {p.relative_to(mesh_dir).as_posix() for p in mesh_dir.rglob("*") if p.is_file()}
    if declared != on_disk:
        raise ModelSetError(f"MJCF mesh closure mismatch: declared-only={declared-on_disk}, disk-only={on_disk-declared}")

    indent_xml(root)
    tree.write(staging / MJCF_OUTPUT_NAME, encoding="utf-8", xml_declaration=True)
    # The MJCF itself is small and belongs in review; the 92 meshes are byte copies of material
    # that is either already tracked under a3_pingpong/meshes or lives in ignored vendor_assets,
    # so duplicating ~39 MB into git would violate docs/ASSET_POLICY.md.  Regenerate with --check.
    (staging / ".gitignore").write_text("meshes/\n", encoding="utf-8")

    return {
        "mesh_directory_git_ignored": True,
        "mesh_directory_note": (
            "meshes/ is ignored and reproduced by this producer; the MJCF and this receipt are the "
            "tracked source of truth"
        ),
        "derived_from": {
            "path": relative_path(INCUMBENT_MJCF),
            "sha256": INCUMBENT_MJCF_SHA256,
            "edited_in_place": False,
        },
        "model_name": root.get("model"),
        "body_count": EXPECTED_MJCF_BODY_COUNT,
        "joint_count": len(joints),
        "actuator_count": len(motors),
        "keyframe_qpos_width": EXPECTED_MJCF_KEYFRAME_QPOS_WIDTH,
        "body_pos_updates": pos_updates,
        "inertial_updates": inertial_updates,
        "retired_left_hand_assets": retired_assets,
        "retired_left_hand_geoms": retired_geoms,
        "gripper_geoms_on_host_body": {
            "host_body": "left_wrist_yaw_Link",
            "count": len(gripper_geoms),
            "geoms": gripper_geoms,
            "contract": GRIPPER_COLLISION_EQUALS_VISUAL_CONTRACT,
            "total_faces": sum(row["mesh_faces"] for row in gripper_geoms),
            "oversize_mesh_hull_substitutions": hull_substitutions,
        },
        "declared_deviations": MJCF_DECLARED_DEVIATIONS,
        "preserved_verbatim": [
            "compiler/option/statistic/visual/default blocks",
            "all 31 joint armature, damping, frictionloss, range and actuatorfrcrange values",
            "all 31 <motor> actuators and their names",
            "the 33 hand-made convex hulls under meshes/collision_optimized/",
            "the racket face proxy mesh, its y-scale and the palm/finger primitive colliders",
            "all 6 sites, 155 sensors, 27 contact excludes, the keyframe and the scene floor/lights/camera",
        ],
        "verification_boundary": {
            "compiled_with_mujoco": False,
            "loaded_with_mujoco_warp": False,
            "identity_manifest_minted": False,
            "reason": "this host has no mujoco/mujoco_warp; structural validation only",
            "required_next": (
                "compile with mujoco.MjModel.from_xml_path on a MuJoCo-bearing pod, confirm "
                "nq=38/nv=37/nu=31/nbody=33 including world, then load via mjlab_lane with "
                "A3_PINGPONG_XML pointing here; mint an identity v3 only if the CPU evidence gates "
                "are to be re-entered"
            ),
        },
    }


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def require_isolated_outputs(isaac_root: Path, mjcf_dir: Path) -> None:
    for candidate, guard, label in (
        (isaac_root, ACTIVE_ISAAC_ASSET_ROOT, "current Isaac runtime asset"),
        (mjcf_dir, INCUMBENT_MJCF_DIR, "in-service MuJoCo model directory"),
    ):
        resolved = candidate.resolve(strict=False)
        protected = guard.resolve(strict=False)
        if resolved == protected or protected in resolved.parents:
            raise ModelSetError(f"refusing to write into the {label}: {candidate}")


def prepare(
    bundle_root: Path, isaac_root: Path, mjcf_dir: Path, receipt_out: Path,
    tracked_root: Path = TRACKED_PACKAGE_ROOT,
) -> dict[str, Any]:
    require_isolated_outputs(isaac_root, mjcf_dir)
    for existing in (isaac_root, mjcf_dir, tracked_root):
        if existing.exists():
            raise ModelSetError(f"refusing to overwrite versioned output: {existing}")

    intake = json.loads(BUNDLE_RECEIPT.read_text(encoding="utf-8"))
    observed_bundle = closure(bundle_root)
    if observed_bundle["sha256"] != intake["bundle"]["closure"]["sha256"]:
        raise ModelSetError("raw bundle closure does not match its intake receipt")
    confirmation = intake["unresolved_gripper_collision_references"]["vendor_confirmation"]
    if confirmation["statement"] != GRIPPER_COLLISION_EQUALS_VISUAL_CONTRACT["statement"]:
        raise ModelSetError("intake receipt no longer carries the gripper collision confirmation")

    urdf = Urdf(bundle_root / intake["bundle"]["urdf_path"])
    isaac_staging = Path(tempfile.mkdtemp(prefix=f".{isaac_root.name}.staging.", dir=str(isaac_root.parent)))
    mjcf_staging = Path(tempfile.mkdtemp(prefix=f".{mjcf_dir.name}.staging.", dir=str(mjcf_dir.parent)))
    try:
        isaac = build_isaac_asset(urdf, bundle_root / "meshes", isaac_staging)
        mjcf = build_mjcf(urdf, bundle_root / "meshes", mjcf_staging)
        isaac_closure = closure(isaac_staging)
        mjcf_closure = closure(mjcf_staging)

        receipt = {
            "schema_version": 1,
            "manifest_type": "a3p_p1_0807_dual_engine_model_set_v1",
            "asset_id": "a3p_p1_32dof_0807_op3_pingpang_v1",
            "candidate_role": "future_primary_successor_candidate_not_current_runtime",
            "producer": {"path": relative_path(Path(__file__)), "sha256": sha256_path(Path(__file__))},
            "raw_bundle": {
                "path": relative_path(bundle_root),
                "intake_manifest_path": relative_path(BUNDLE_RECEIPT),
                "intake_manifest_sha256": sha256_path(BUNDLE_RECEIPT),
                "closure_sha256": observed_bundle["sha256"],
            },
            "tracked_package": {
                "path": relative_path(tracked_root),
                "git_tracked": True,
                "identical_bytes_to_isaac_asset": True,
                "role": "reviewable robot-description package beside agi/URDF/A3T2.5-URDF-std-pingpang",
            },
            "isaac": {
                "asset_path": relative_path(isaac_root),
                "asset_path_git_ignored": True,
                "urdf_path": "urdf/model.urdf",
                "urdf_sha256": sha256_path(isaac_staging / "urdf" / "model.urdf"),
                "closure": isaac_closure,
                "normalization": isaac,
                "verification_boundary": {
                    "imported_with_isaac_lab": False,
                    "reason": "this host has no Isaac Lab; structural and ABI validation only",
                    "required_next": "run the Pod URDF->USD converter with merge_fixed_joints=True and confirm 31 joints / 32 bodies",
                },
            },
            "mujoco": {
                "model_dir": relative_path(mjcf_dir),
                "root_filename": MJCF_OUTPUT_NAME,
                "root_sha256": sha256_path(mjcf_staging / MJCF_OUTPUT_NAME),
                "closure": mjcf_closure,
                "gpu_lane_entry_point": (
                    "A3_PINGPONG_XML=<model_dir>/" + MJCF_OUTPUT_NAME + " -- mjlab_lane/a3_plant_env.py "
                    "default_xml() honours this env var and applies no hash pin"
                ),
                "derivation": mjcf,
            },
            "cross_engine": {
                "shared_raw_bundle": True,
                "shared_gripper_lock_contract": True,
                "isaac_body_fold": "merge_fixed_joints=True folds the gripper into left_wrist_yaw_Link",
                "mujoco_body_fold": "gripper inertia folded into left_wrist_yaw_Link by this producer",
                "parity_verified": False,
                "parity_note": (
                    "the two folds are constructed to agree, but no cross-engine parity run has been "
                    "executed; nothing here proves the engines agree at runtime"
                ),
            },
            "authorization": {
                "current_isaac_runtime_pointer_changed": False,
                "in_service_mjcf_edited": False,
                "materialization_authorized": True,
                "gripper_collision_equals_visual_authorized": True,
                "gripper_collision_vendor_written_confirmation": False,
                "isaac_import_verified": False,
                "mujoco_compile_verified": False,
                "mujoco_warp_load_verified": False,
                "mujoco_identity_v3_minted": False,
                "cross_engine_parity_verified": False,
                "motion_bank_revalidated": False,
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
        }
        publish = not receipt_out.exists()
        if not publish and json.loads(receipt_out.read_text(encoding="utf-8")) != receipt:
            raise ModelSetError("tracked model-set receipt does not match the regenerated outputs")
        shutil.copytree(isaac_staging, tracked_root)
        isaac_staging.rename(isaac_root)
        mjcf_staging.rename(mjcf_dir)
        if publish:
            receipt_out.parent.mkdir(parents=True, exist_ok=True)
            receipt_out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        for staging in (isaac_staging, mjcf_staging):
            if staging.exists():
                shutil.rmtree(staging)
        raise
    return receipt


def check(bundle_root: Path, isaac_root: Path, mjcf_dir: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("manifest_type") != "a3p_p1_0807_dual_engine_model_set_v1":
        raise ModelSetError("wrong model-set manifest type")
    for label, root, expected in (
        ("isaac", isaac_root, receipt["isaac"]["closure"]),
        ("mujoco", mjcf_dir, receipt["mujoco"]["closure"]),
    ):
        observed = closure(root)
        for key in ("file_count", "total_bytes", "sha256", "files"):
            if observed[key] != expected[key]:
                raise ModelSetError(f"{label} closure mismatch at {key}")
    if sha256_path(INCUMBENT_MJCF) != INCUMBENT_MJCF_SHA256:
        raise ModelSetError("the in-service MJCF has been modified")

    urdf_root = ET.parse(isaac_root / receipt["isaac"]["urdf_path"]).getroot()
    links = [link.get("name") for link in urdf_root.findall("link")]
    if len(links) != len(set(links)):
        raise ModelSetError("normalized URDF contains duplicate link names")
    movable = [j.get("name") for j in urdf_root.findall("joint") if j.get("type") not in {"fixed", "floating"}]
    if set(movable) != set(read_order(RUNTIME_JOINT_ORDER)):
        raise ModelSetError("normalized joint set drifted from the runtime action ABI")
    refs = {Path(m.get("filename")).name for m in urdf_root.iter("mesh")}
    on_disk = {p.name for p in (isaac_root / "meshes").iterdir() if p.is_file()}
    if refs - on_disk:
        raise ModelSetError(f"Isaac asset has unresolved mesh references: {sorted(refs - on_disk)}")
    for row in receipt["isaac"]["normalization"]["materialised_gripper_collision_meshes"]:
        collision = isaac_root / "meshes" / row["collision_basename"]
        visual = isaac_root / "meshes" / row["visual_basename"]
        if sha256_path(collision) != sha256_path(visual):
            raise ModelSetError(f"materialised collision is no longer its visual: {row['collision_basename']}")

    mjcf_root = ET.parse(mjcf_dir / receipt["mujoco"]["root_filename"]).getroot()
    declared = {m.get("file") for m in mjcf_root.findall("asset/mesh")}
    disk = {p.relative_to(mjcf_dir / "meshes").as_posix() for p in (mjcf_dir / "meshes").rglob("*") if p.is_file()}
    if declared != disk:
        raise ModelSetError("MJCF mesh closure drifted")
    if len(list(mjcf_root.iter("body"))) != EXPECTED_MJCF_BODY_COUNT:
        raise ModelSetError("MJCF body count drifted")
    if len(mjcf_root.findall("actuator/motor")) != EXPECTED_MJCF_ACTUATOR_COUNT:
        raise ModelSetError("MJCF actuator count drifted")
    return {
        "status": "PASS",
        "isaac_movable_joint_count": len(movable),
        "isaac_link_count": len(links),
        "isaac_mesh_reference_count": len(refs),
        "mujoco_body_count": EXPECTED_MJCF_BODY_COUNT,
        "mujoco_mesh_count": len(declared),
        "mujoco_compile_verified": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--isaac-root", type=Path, default=ISAAC_ASSET_ROOT)
    parser.add_argument("--mjcf-dir", type=Path, default=MJCF_OUTPUT_DIR)
    parser.add_argument("--receipt", type=Path, default=MODEL_SET_RECEIPT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            report = check(
                args.bundle_root.resolve(), args.isaac_root.resolve(),
                args.mjcf_dir.resolve(), args.receipt.resolve(),
            )
        else:
            receipt = prepare(
                args.bundle_root.resolve(), args.isaac_root.resolve(),
                args.mjcf_dir.resolve(), args.receipt.resolve(),
            )
            report = {
                "status": "PREPARED",
                "isaac_closure_sha256": receipt["isaac"]["closure"]["sha256"],
                "mujoco_root_sha256": receipt["mujoco"]["root_sha256"],
                "mujoco_compile_verified": False,
            }
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ModelSetError, FileNotFoundError, ET.ParseError, KeyError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
