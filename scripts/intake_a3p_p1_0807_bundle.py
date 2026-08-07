#!/usr/bin/env python3
"""Assemble and receipt the 2026-08-07 A3P-P1 delivery as one immutable raw bundle.

The 0807 delivery arrived as two disjoint pieces that are individually unusable:

* a standalone ``A3P-P1-32dof-0807-OP3+pingpang_20260807_083135.urdf`` carrying **no** meshes, and
* an ``OmniPicker3-T1-0324-T1.5-close-ROS2`` package carrying only the 20 gripper visual meshes.

Every one of the 104 resolvable mesh references in the 0807 URDF resolves into the mesh folder of
the earlier ``A3-P1-32dof-0803-BerkeleyPingpang-90deg`` delivery, and the 20 OmniPicker3 ``.STL``
files are byte-identical (SHA-256, 20/20) to their 0803 counterparts, so the 0807 URDF is a
re-emission against unchanged part geometry.

This tool therefore builds a *project-assembled bundle* -- not a vendor closure -- and records the
exact origin package and SHA-256 of every single byte it places.  It never edits delivered bytes and
never fabricates geometry: the 20 gripper ``*_collision.stl`` references remain unresolved here and
are materialised downstream by the asset producer under the vendor's explicit "gripper collision is
the visual geometry" confirmation, so that the substitution stays visible in a receipt instead of
being laundered into what looks like a vendor delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor_assets" / "agibot"
BUNDLE_NAME = "A3P-P1-32dof-0807-OP3+pingpang"
DEFAULT_BUNDLE_ROOT = VENDOR_ROOT / BUNDLE_NAME
DEFAULT_RECEIPT = REPO_ROOT / "configs" / "a3p_p1_0807_raw_intake_v1.json"

PREDECESSOR_ROOT = VENDOR_ROOT / "A3-P1-32dof-0803-BerkeleyPingpang-90deg"
PREDECESSOR_INTAKE = REPO_ROOT / "configs" / "a3_p1_0803_raw_intake_v1.json"

DEFAULT_DOWNLOADS = Path(os.path.expanduser("~/Downloads"))
DELIVERED_URDF_NAME = "A3P-P1-32dof-0807-OP3+pingpang_20260807_083135.urdf"
DELIVERED_OP3_NAME = "OmniPicker3-T1-0324-T1.5-close-ROS2"

EXPECTED_MESH_REFERENCE_COUNT = 124
EXPECTED_RESOLVABLE_MESH_COUNT = 104
EXPECTED_UNRESOLVED_GRIPPER_COLLISION_COUNT = 20
EXPECTED_UNUSED_PREDECESSOR_MESHES = (
    "left_hand_Link.stl",
    "left_hand_Link_collision.stl",
    "right_hand_Link.stl",
    "right_hand_Link_collision.stl",
)
EXPECTED_ROBOT_NAME = "A3P-P1-32dof-0807-OP3+pingpang"
EXPECTED_LINK_ELEMENT_COUNT = 64
EXPECTED_UNIQUE_LINK_COUNT = 63
EXPECTED_DUPLICATE_LINK_NAMES = ("imu_in_pelvis_link",)
EXPECTED_MOVABLE_JOINT_COUNT = 40
EXPECTED_BODY_MOVABLE_JOINT_COUNT = 31
EXPECTED_TOTAL_UNIQUE_LINK_MASS_KG = 57.60001015416

# Recorded verbatim so the substitution is auditable rather than assumed.  Relayed by Franco on
# 2026-08-07 from Agibot; it is a vendor statement, not a project inference, and it is what makes
# materialising the 20 collision meshes downstream legitimate rather than fabrication.
VENDOR_GRIPPER_COLLISION_CONFIRMATION = {
    "schema_version": 1,
    "statement": "gripper collision geometry is identical to the gripper visual geometry",
    "received_utc_date": "2026-08-07",
    "channel": "relayed_by_project_owner_from_vendor",
    "written_evidence_on_file": False,
    "scope": "the twenty left OmniPicker3 gripper links only",
    "consequence": (
        "the twenty absent *_collision.stl references may be materialised downstream as byte copies "
        "of the corresponding visual mesh; this replaces the 0803 collision-disabled contract"
    ),
    "still_unconfirmed": (
        "gripper joint coupling model, neutral/home pose, and the eight placeholder finger limits "
        "(lower=-2 upper=2 effort=1 velocity=1) remain unanswered"
    ),
}

# Defects the 0807 re-emission fixed relative to 0803, and the ones it did not.  Asserted, not
# narrated: drift in either direction fails the intake.
EXPECTED_FIXED_SINCE_0803 = {
    "right_elbow_joint_origin_x": {
        "predecessor": "0.001 0 -0.1325",
        "delivered": "0.01 0 -0.1325",
        "note": "vendor adopted the project's mirror-symmetry diagnosis at source",
    },
    "illegal_axis_on_fixed_joints": {"predecessor_count": 5, "delivered_count": 0},
    "ankle_pitch_lateral_asymmetry": {
        "predecessor_left_y": "0.00150000000138195",
        "predecessor_right_y": "-0.001499999998404",
        "delivered_y": "0",
        "note": "1.5 mm per foot; a real geometry change, not rounding",
    },
}
EXPECTED_STILL_PRESENT_DEFECTS = (
    "duplicate_imu_in_pelvis_link_definition",
    "nan_rgba_geometryless_visual_on_left_base_footprint",
    "case_mismatched_mesh_references",
    "hyphenated_usd_unsafe_gripper_mesh_basenames",
    "no_gripper_coupling_authority",
    "no_joint_dynamics_or_actuator_data",
)


class IntakeError(RuntimeError):
    """Raised when the delivery or the assembled bundle violates the recorded contract."""


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


def mesh_references(urdf: Path) -> list[str]:
    text = urdf.read_text(encoding="utf-8")
    return re.findall(r'filename="[^"]*/([^/"]+)"', text)


def describe_urdf(urdf: Path) -> dict[str, Any]:
    root = ET.parse(urdf).getroot()
    if root.get("name") != EXPECTED_ROBOT_NAME:
        raise IntakeError(f"unexpected robot name: {root.get('name')!r}")
    link_names = [link.get("name") for link in root.findall("link")]
    duplicates = tuple(sorted(name for name, count in Counter(link_names).items() if count > 1))
    if len(link_names) != EXPECTED_LINK_ELEMENT_COUNT:
        raise IntakeError(f"expected {EXPECTED_LINK_ELEMENT_COUNT} link elements, found {len(link_names)}")
    if len(set(link_names)) != EXPECTED_UNIQUE_LINK_COUNT:
        raise IntakeError(f"expected {EXPECTED_UNIQUE_LINK_COUNT} unique links, found {len(set(link_names))}")
    if duplicates != EXPECTED_DUPLICATE_LINK_NAMES:
        raise IntakeError(f"duplicate-link inventory drifted: {duplicates}")

    joints = root.findall("joint")
    movable = [j.get("name") for j in joints if j.get("type") not in {"fixed", "floating"}]
    if len(movable) != EXPECTED_MOVABLE_JOINT_COUNT:
        raise IntakeError(f"expected {EXPECTED_MOVABLE_JOINT_COUNT} movable joints, found {len(movable)}")

    illegal_axes = [j.get("name") for j in joints if j.get("type") == "fixed" and j.find("axis") is not None]
    if illegal_axes:
        raise IntakeError(f"0807 was expected to have removed all fixed-joint axes, found {illegal_axes}")

    origins = {}
    for joint in joints:
        origin = joint.find("origin")
        origins[joint.get("name")] = None if origin is None else origin.get("xyz")
    fixed = EXPECTED_FIXED_SINCE_0803
    if origins.get("right_elbow_joint") != fixed["right_elbow_joint_origin_x"]["delivered"]:
        raise IntakeError(
            "0807 right_elbow_joint origin is not the vendor-corrected value: "
            f"{origins.get('right_elbow_joint')!r}"
        )
    for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        parts = (origins.get(name) or "").split()
        if len(parts) != 3 or float(parts[1]) != 0.0:
            raise IntakeError(f"0807 {name} lateral offset is not zero: {origins.get(name)!r}")

    masses = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        masses.setdefault(link.get("name"), float(inertial.find("mass").get("value")))
    total = sum(masses.values())
    if abs(total - EXPECTED_TOTAL_UNIQUE_LINK_MASS_KG) > 1e-9:
        raise IntakeError(f"0807 unique-link mass drifted: {total}")

    nan_rgba = [
        element.get("rgba")
        for element in root.findall(".//*[@rgba]")
        if "nan" in (element.get("rgba") or "").lower()
    ]

    return {
        "robot_name": root.get("name"),
        "link_element_count": len(link_names),
        "unique_link_name_count": len(set(link_names)),
        "duplicate_link_names": list(duplicates),
        "joint_count": len(joints),
        "movable_joint_count": len(movable),
        "body_movable_joint_count": EXPECTED_BODY_MOVABLE_JOINT_COUNT,
        "gripper_movable_joint_count": len(movable) - EXPECTED_BODY_MOVABLE_JOINT_COUNT,
        "mimic_joint_count": len(root.findall(".//mimic")),
        "transmission_count": len(root.findall("transmission")),
        "dynamics_element_count": len(root.findall(".//dynamics")),
        "fixed_joints_with_illegal_axis": illegal_axes,
        "nonfinite_rgba_values": nan_rgba,
        "unique_link_total_mass_kg": total,
    }


def resolve_meshes(urdf: Path, sources: list[tuple[str, Path]]) -> dict[str, Any]:
    refs = mesh_references(urdf)
    distinct = sorted(set(refs))
    if len(distinct) != EXPECTED_MESH_REFERENCE_COUNT:
        raise IntakeError(f"expected {EXPECTED_MESH_REFERENCE_COUNT} distinct mesh refs, found {len(distinct)}")

    indexes = []
    for label, root in sources:
        if not root.is_dir():
            raise IntakeError(f"mesh source is not a directory: {root}")
        indexes.append((label, root, {p.name.lower(): p for p in sorted(root.iterdir()) if p.is_file()}))

    resolved: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for ref in distinct:
        for label, _root, index in indexes:
            match = index.get(ref.lower())
            if match is not None:
                resolved[ref] = {
                    "source_package": label,
                    "source_basename": match.name,
                    "case_exact": match.name == ref,
                    "sha256": sha256_path(match),
                    "bytes": match.stat().st_size,
                    "_path": match,
                }
                break
        else:
            unresolved.append(ref)

    if len(resolved) != EXPECTED_RESOLVABLE_MESH_COUNT:
        raise IntakeError(f"expected {EXPECTED_RESOLVABLE_MESH_COUNT} resolvable meshes, found {len(resolved)}")
    if len(unresolved) != EXPECTED_UNRESOLVED_GRIPPER_COLLISION_COUNT:
        raise IntakeError(f"expected {EXPECTED_UNRESOLVED_GRIPPER_COLLISION_COUNT} unresolved refs, found {unresolved}")
    for ref in unresolved:
        if not ref.endswith("_collision.stl"):
            raise IntakeError(f"unresolved reference is not a gripper collision mesh: {ref}")
        visual = ref.replace("_collision", "")
        if visual not in resolved:
            raise IntakeError(f"unresolved collision {ref} has no delivered visual counterpart {visual}")
    return {"resolved": resolved, "unresolved": sorted(unresolved), "distinct_reference_count": len(distinct)}


def cross_check_omnipicker(op3_root: Path, resolved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The OmniPicker3 package must add no geometry: every mesh must already match byte-for-byte."""

    mesh_dir = op3_root / "meshes"
    if not mesh_dir.is_dir():
        raise IntakeError(f"OmniPicker3 package has no meshes directory: {mesh_dir}")
    by_lower = {ref.lower(): item for ref, item in resolved.items()}
    rows = []
    for path in sorted(mesh_dir.iterdir()):
        if not path.is_file():
            continue
        counterpart = by_lower.get(path.name.lower())
        if counterpart is None:
            raise IntakeError(f"OmniPicker3 ships a mesh the 0807 URDF never resolves: {path.name}")
        digest = sha256_path(path)
        if digest != counterpart["sha256"]:
            raise IntakeError(
                f"OmniPicker3 {path.name} differs from the delivered counterpart "
                f"{counterpart['source_basename']}: {digest} vs {counterpart['sha256']}"
            )
        rows.append({"basename": path.name, "sha256": digest, "matches_bundled_mesh": True})
    return {
        "mesh_count": len(rows),
        "all_byte_identical_to_bundled_meshes": True,
        "adds_new_geometry": False,
        "meshes": rows,
    }


def build(
    bundle_root: Path,
    receipt_out: Path,
    delivered_urdf: Path,
    op3_root: Path,
    predecessor_root: Path,
) -> dict[str, Any]:
    if bundle_root.exists():
        raise IntakeError(f"refusing to overwrite an existing raw bundle: {bundle_root}")
    if not delivered_urdf.is_file():
        raise IntakeError(f"delivered 0807 URDF not found: {delivered_urdf}")
    if not predecessor_root.is_dir():
        raise IntakeError(f"predecessor delivery not found (needed for meshes): {predecessor_root}")

    structure = describe_urdf(delivered_urdf)
    sources = [
        ("A3-P1-32dof-0803-BerkeleyPingpang-90deg", predecessor_root / "meshes"),
        (DELIVERED_OP3_NAME, op3_root / "meshes"),
    ]
    meshes = resolve_meshes(delivered_urdf, sources)
    op3 = cross_check_omnipicker(op3_root, meshes["resolved"])

    used = {item["source_basename"] for item in meshes["resolved"].values()}
    unused = tuple(sorted({p.name for p in (predecessor_root / "meshes").iterdir() if p.is_file()} - used))
    if unused != EXPECTED_UNUSED_PREDECESSOR_MESHES:
        raise IntakeError(f"predecessor unused-mesh inventory drifted: {unused}")

    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_root.name}.staging.", dir=str(bundle_root.parent)))
    try:
        (staging / "urdf").mkdir()
        (staging / "meshes").mkdir()
        shutil.copyfile(delivered_urdf, staging / "urdf" / delivered_urdf.name)
        if sha256_path(staging / "urdf" / delivered_urdf.name) != sha256_path(delivered_urdf):
            raise IntakeError("URDF copy changed bytes")
        for ref, item in sorted(meshes["resolved"].items()):
            # Land every mesh under the exact basename the 0807 URDF asks for, so the bundle is
            # self-consistent on a case-sensitive filesystem without any byte being altered.
            destination = staging / "meshes" / ref
            shutil.copyfile(item["_path"], destination)
            if sha256_path(destination) != item["sha256"]:
                raise IntakeError(f"mesh copy changed bytes: {ref}")
        shutil.copytree(op3_root, staging / "vendor_packages" / op3_root.name)

        observed = closure(staging)
        receipt = {
            "schema_version": 1,
            "manifest_type": "a3p_p1_0807_op3_pingpang_raw_intake_v1",
            "bundle_id": "a3p_p1_32dof_0807_op3_pingpang",
            "assembled_not_a_vendor_closure": True,
            "assembly_note": (
                "the vendor shipped a meshless URDF plus a gripper-only ROS2 package; every "
                "resolvable mesh reference is served by the 2026-08-03 delivery's meshes, "
                "byte-for-byte, and the OmniPicker3 package adds no new geometry"
            ),
            "producer": {
                "path": relative_path(Path(__file__)),
                "sha256": sha256_path(Path(__file__)),
            },
            "delivered_inputs": {
                "urdf": {
                    "delivered_basename": delivered_urdf.name,
                    "sha256": sha256_path(delivered_urdf),
                    "bytes": delivered_urdf.stat().st_size,
                },
                "omnipicker3_package": {
                    "delivered_name": op3_root.name,
                    "closure": closure(op3_root),
                },
            },
            "mesh_provenance": {
                "distinct_reference_count": meshes["distinct_reference_count"],
                "resolved_count": len(meshes["resolved"]),
                "source_package_histogram": dict(
                    Counter(item["source_package"] for item in meshes["resolved"].values())
                ),
                "case_exact_count": sum(1 for item in meshes["resolved"].values() if item["case_exact"]),
                "case_mismatched_count": sum(
                    1 for item in meshes["resolved"].values() if not item["case_exact"]
                ),
                "predecessor_intake_manifest_path": relative_path(PREDECESSOR_INTAKE),
                "predecessor_intake_manifest_sha256": sha256_path(PREDECESSOR_INTAKE),
                "predecessor_meshes_unused_by_0807": list(unused),
                "per_reference": {
                    ref: {k: v for k, v in item.items() if not k.startswith("_")}
                    for ref, item in sorted(meshes["resolved"].items())
                },
            },
            "unresolved_gripper_collision_references": {
                "count": len(meshes["unresolved"]),
                "references": meshes["unresolved"],
                "materialised_in_this_bundle": False,
                "downstream_policy": (
                    "the asset producer materialises each as a byte copy of its visual counterpart "
                    "under the recorded vendor confirmation; never fabricated here"
                ),
                "vendor_confirmation": VENDOR_GRIPPER_COLLISION_CONFIRMATION,
            },
            "omnipicker3_cross_check": op3,
            "structure": structure,
            "fixed_since_predecessor": EXPECTED_FIXED_SINCE_0803,
            "still_present_defects": list(EXPECTED_STILL_PRESENT_DEFECTS),
            "bundle": {
                "path": relative_path(bundle_root),
                "path_git_ignored": True,
                "urdf_path": f"urdf/{delivered_urdf.name}",
                "closure": observed,
            },
        }
        receipt["bundle"]["path"] = relative_path(bundle_root)
        publish = not receipt_out.exists()
        if not publish and json.loads(receipt_out.read_text(encoding="utf-8")) != receipt:
            raise IntakeError("tracked intake receipt does not match the regenerated bundle")
        staging.rename(bundle_root)
        if publish:
            receipt_out.parent.mkdir(parents=True, exist_ok=True)
            receipt_out.write_text(
                json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def check(bundle_root: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("manifest_type") != "a3p_p1_0807_op3_pingpang_raw_intake_v1":
        raise IntakeError("wrong intake manifest type")
    observed = closure(bundle_root)
    expected = receipt["bundle"]["closure"]
    for key in ("file_count", "total_bytes", "sha256", "files"):
        if observed[key] != expected[key]:
            raise IntakeError(f"bundle closure mismatch at {key}")
    urdf = bundle_root / receipt["bundle"]["urdf_path"]
    if sha256_path(urdf) != receipt["delivered_inputs"]["urdf"]["sha256"]:
        raise IntakeError("bundled URDF is not the delivered URDF")
    structure = describe_urdf(urdf)
    if structure != receipt["structure"]:
        raise IntakeError("bundled URDF structure does not reproduce the receipt")
    refs = sorted(set(mesh_references(urdf)))
    on_disk = {p.name for p in (bundle_root / "meshes").iterdir() if p.is_file()}
    missing = [ref for ref in refs if ref not in on_disk]
    if sorted(missing) != receipt["unresolved_gripper_collision_references"]["references"]:
        raise IntakeError(f"bundle unresolved-reference set drifted: {missing}")
    unused = sorted(on_disk - set(refs))
    if unused:
        raise IntakeError(f"bundle carries meshes the URDF never references: {unused}")
    return {
        "status": "PASS",
        "bundle_closure_sha256": observed["sha256"],
        "file_count": observed["file_count"],
        "mesh_reference_count": len(refs),
        "resolvable_case_exact_on_case_sensitive_filesystem": len(refs) - len(missing),
        "unresolved_gripper_collision_count": len(missing),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--delivered-urdf", type=Path, default=DEFAULT_DOWNLOADS / DELIVERED_URDF_NAME)
    parser.add_argument("--omnipicker3-root", type=Path, default=DEFAULT_DOWNLOADS / DELIVERED_OP3_NAME)
    parser.add_argument("--predecessor-root", type=Path, default=PREDECESSOR_ROOT)
    parser.add_argument("--check", action="store_true", help="Verify the existing bundle against the receipt.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            report = check(args.bundle_root.resolve(), args.receipt.resolve())
        else:
            receipt = build(
                args.bundle_root.resolve(),
                args.receipt.resolve(),
                args.delivered_urdf.expanduser().resolve(),
                args.omnipicker3_root.expanduser().resolve(),
                args.predecessor_root.resolve(),
            )
            report = {
                "status": "ASSEMBLED",
                "bundle_root": args.bundle_root.as_posix(),
                "receipt": args.receipt.as_posix(),
                "closure_sha256": receipt["bundle"]["closure"]["sha256"],
                "file_count": receipt["bundle"]["closure"]["file_count"],
            }
        print(json.dumps(report, sort_keys=True))
        return 0
    except (IntakeError, FileNotFoundError, ET.ParseError, KeyError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
