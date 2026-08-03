#!/usr/bin/env python3
"""Build or verify the non-canonical A3-P1 0803 31-action Isaac candidate.

The vendor delivery is immutable and ignored under ``vendor_assets``.  This
tool derives a separate ignored asset by locking the unresolved left-gripper
subtree at its URDF zero pose, restoring the established mixed-case A3 body-name
ABI, and copying only meshes referenced by the retained robot.  It never changes the current
``assets/agibot_a3`` runtime asset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_INTAKE_MANIFEST = REPO_ROOT / "configs" / "a3_p1_0803_raw_intake_v1.json"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "vendor_assets" / "agibot" / "A3-P1-32dof-0803-BerkeleyPingpang-90deg"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "agibot_a3_p1_0803_31d_v1"
)
DEFAULT_SUCCESSOR_MANIFEST = REPO_ROOT / "configs" / "a3_p1_0803_31d_v1.json"
RUNTIME_JOINT_ORDER = REPO_ROOT / "configs" / "a3_runtime_articulation_joint_order.txt"
RUNTIME_BODY_ORDER = REPO_ROOT / "configs" / "a3_runtime_body_order.txt"
GMR_JOINT_ORDER = REPO_ROOT / "configs" / "a3_gmr_dof_pos_joint_order.txt"
JOINT_BIJECTION = REPO_ROOT / "configs" / "a3_joint_order_bijection_v1.json"

GRIPPER_MOUNT_JOINT = "left_OP3_joint"
RAW_DUPLICATE_LINK = "imu_in_pelvis_link"
EXPECTED_MOVABLE_JOINTS = 31
EXPECTED_FIXED_GRIPPER_JOINTS = 9
REQUIRED_RACKET_MESHES = {
    "right_hand_pingpang_Link.stl",
    "pingpang_red_Link.stl",
    "pingpang_black_Link.stl",
    "pingbang_ball_Link.stl",
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

    root.set("name", "A3-P1-0803-BerkeleyPingpang-31action-normalized-v1")

    mesh_reference_rewrites, removed_missing_collisions = normalize_mesh_references(root, source_meshes)
    if len(removed_missing_collisions) != 20:
        raise AssetError(
            f"expected 20 missing gripper collision refs, found {len(removed_missing_collisions)}"
        )

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
    if raw_semantics != output_semantics:
        raise AssetError("normalization changed retained body inertials or joint origin/axis/limit semantics")

    normalized_movable = movable_joint_names(root)
    if len(normalized_movable) != EXPECTED_MOVABLE_JOINTS:
        raise AssetError(f"normalized asset has {len(normalized_movable)} movable joints, expected 31")

    invalid_rgba_after = parse_rgba(root)
    if invalid_rgba_after:
        raise AssetError(f"normalized asset retains invalid rgba values: {invalid_rgba_after}")

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
            "reason": "left gripper is outside the established 31-action policy ABI; fixed links retain delivered mass, inertias, COMs, and joint origins without inventing coupling",
        },
        "removed_duplicate_link": {
            "name": RAW_DUPLICATE_LINK,
            "removed_occurrence": "first_document_occurrence",
            "retained_occurrence": "last_document_occurrence_with_dedicated_collision_mesh",
            "raw_occurrence_fingerprint_sha256": duplicate_sha,
        },
        "link_name_map": link_name_map,
        "mesh_reference_policy": "rewrite to delivered case-exact basenames and copy only referenced files",
        "mesh_reference_rewrites": mesh_reference_rewrites,
        "removed_missing_collision_elements": removed_missing_collisions,
        "removed_invalid_visual_elements": removed_invalid_visuals,
        "invalid_rgba_before": invalid_rgba_before,
        "invalid_rgba_after": invalid_rgba_after,
        "retained_semantics_sha256": canonical_json_sha(raw_semantics),
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
            normalized = "../meshes/" + actual
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
    return {
        "parent_link": parent.get("link"),
        "child_link": child.get("link"),
        "origin_xyz": origin.get("xyz"),
        "origin_rpy": origin.get("rpy"),
        "mesh_sha256": mesh_sha,
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

    return {
        "schema_version": 1,
        "manifest_type": "agibot_a3_p1_0803_31action_normalized_asset_v1",
        "asset_id": "a3_p1_0803_berkeley_pingpang_31action_normalized_v1",
        "status": "host_static_candidate_pod_import_pending",
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
        "right_racket_contract": racket_contract(normalized_root, output_root / "meshes"),
        "authorization": {
            "current_runtime_pointer_changed": False,
            "canonical_runtime": False,
            "pod_isaac_import_verified": False,
            "standing_pose_verified": False,
            "racket_fk_parity_verified": False,
            "dynamics_parity_verified": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }


def prepare(source_root: Path, output_root: Path, intake_path: Path, manifest_out: Path) -> dict[str, Any]:
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
            shutil.copyfile(source, mesh_dir / source.name)
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
