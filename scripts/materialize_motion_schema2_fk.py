#!/usr/bin/env python3
"""Exact B/C GMR-pickle -> schema-2 MuJoCo-FK materializer.

``static`` is dependency-light and validates the two independent preregistrations plus every
tracked source binding. ``inspect`` additionally reads the exact private pickle, exact donor ONNX
metadata and vendor MJCF without writing. ``consume`` is the only writing mode; it performs the
frozen 30 -> 50 Hz interpolation and MuJoCo forward kinematics, then publishes one NPZ and its
report into a new no-clobber directory.  This source does not authorize simulator, training or
hardware execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_ROOT = REPO_ROOT / "hope_training/whole_body_tracking/scripts"
for _path in (str(REPO_ROOT / "scripts"), str(HELPER_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLAN_STATUS = "preregistered_source_gate_pass_runtime_inspection_not_run"
SHARED_STATUS = "source_contract_complete_runtime_inspection_not_run"
RESULT_STATUS = "complete_exact_schema2_fk_materialization_certificate_blocked"
ASSET_IDS = ("franco_backhand_loop_b", "franco_backhand_loop_c")


class Schema2ContractError(ValueError):
    """A content, frame, order, runtime or publication contract is incomplete."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Schema2ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def json_loads_exact(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Schema2ContractError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Schema2ContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Schema2ContractError(f"{label} must be a JSON object")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json_loads_exact(path.read_bytes(), label)
    except OSError as exc:
        raise Schema2ContractError(f"cannot read {label} {path}: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise Schema2ContractError(f"{label} must be a lowercase SHA-256")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Schema2ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise Schema2ContractError(
            f"{label} keys changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def ensure_regular_no_symlink(path: Path, label: str) -> None:
    probe = path
    while probe != probe.parent:
        if probe.exists() and probe.is_symlink():
            raise Schema2ContractError(f"{label} contains symlink component {probe}")
        probe = probe.parent
    try:
        info = path.stat()
    except OSError as exc:
        raise Schema2ContractError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise Schema2ContractError(f"{label} is not a regular file: {path}")


def repo_path(raw: Any, label: str, *, repo_root: Path = REPO_ROOT) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        raise Schema2ContractError(f"{label} must be a non-empty repo-relative path")
    root = repo_root.resolve()
    lexical_path = Path(os.path.abspath(root / raw))
    ensure_regular_no_symlink(lexical_path, label)
    path = lexical_path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Schema2ContractError(f"{label} escapes repository: {raw}") from exc
    ensure_regular_no_symlink(path, label)
    return path


def verify_repo_binding(
    value: Any, label: str, *, repo_root: Path = REPO_ROOT, expected_path: str | None = None
) -> Path:
    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    if expected_path is not None and binding["path"] != expected_path:
        raise Schema2ContractError(f"{label}.path must equal {expected_path}")
    path = repo_path(binding["path"], f"{label}.path", repo_root=repo_root)
    expected_bytes = binding["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise Schema2ContractError(f"{label}.bytes must be a positive integer")
    if path.stat().st_size != expected_bytes:
        raise Schema2ContractError(
            f"{label} bytes {path.stat().st_size} != {expected_bytes}"
        )
    expected_sha = require_sha(binding["sha256"], f"{label}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise Schema2ContractError(f"{label} SHA {actual_sha} != {expected_sha}")
    return path


def validate_absolute_binding(value: Any, label: str, *, verify: bool) -> Path:
    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    raw = binding["path"]
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise Schema2ContractError(f"{label}.path must be absolute")
    expected_bytes = binding["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise Schema2ContractError(f"{label}.bytes must be a positive integer")
    require_sha(binding["sha256"], f"{label}.sha256")
    path = Path(raw)
    if verify:
        ensure_regular_no_symlink(path, label)
        if path.stat().st_size != expected_bytes:
            raise Schema2ContractError(
                f"{label} bytes {path.stat().st_size} != {expected_bytes}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != binding["sha256"]:
            raise Schema2ContractError(
                f"{label} SHA {actual_sha} != {binding['sha256']}"
            )
    return path


def canonical_names_sha256(names: Sequence[str]) -> str:
    payload = json.dumps(list(names), ensure_ascii=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def read_name_file(path: Path, *, expected_count: int, label: str) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(names) != expected_count or len(set(names)) != expected_count or any(not name for name in names):
        raise Schema2ContractError(
            f"{label} must contain {expected_count} unique non-empty names, got {len(names)}"
        )
    return names


def _resolve_inside(root: Path, candidate: Path, label: str) -> Path:
    lexical_path = Path(os.path.abspath(candidate))
    ensure_regular_no_symlink(lexical_path, label)
    resolved = lexical_path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise Schema2ContractError(f"{label} escapes model root: {candidate}") from exc
    return resolved


def derive_mjcf_closure(main_xml: Path, model_root: Path) -> dict[str, Any]:
    """Derive every recursively included XML and external file referenced by it."""

    root = model_root.resolve()
    main = _resolve_inside(root, main_xml, "canonical MJCF")
    queue = [main]
    xml_files: list[Path] = []
    external_files: list[Path] = []
    mesh_references = 0
    include_references = 0
    inherited_meshdir: Path | None = None
    while queue:
        xml_path = queue.pop(0)
        if xml_path in xml_files:
            raise Schema2ContractError(f"MJCF include cycle or duplicate include: {xml_path}")
        xml_files.append(xml_path)
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise Schema2ContractError(f"invalid MJCF XML {xml_path}: {exc}") from exc
        xml_root = tree.getroot()
        compiler = next(iter(xml_root.iter("compiler")), None)
        if compiler is not None and compiler.get("meshdir") is not None:
            current_meshdir = _resolve_directory_inside(
                root, xml_path.parent / compiler.get("meshdir", ""), "MJCF meshdir"
            )
            if inherited_meshdir is None:
                inherited_meshdir = current_meshdir
            elif current_meshdir != inherited_meshdir:
                raise Schema2ContractError("multiple MJCF meshdir values are not supported")
        for include in xml_root.iter("include"):
            include_references += 1
            raw = include.get("file")
            if not raw:
                raise Schema2ContractError("MJCF include lacks file")
            queue.append(_resolve_inside(root, xml_path.parent / raw, "MJCF include"))
        for node in xml_root.iter():
            raw = node.get("file")
            if not raw or node.tag == "include":
                continue
            if node.tag == "mesh":
                mesh_references += 1
                if inherited_meshdir is None:
                    base = xml_path.parent
                else:
                    base = inherited_meshdir
            else:
                base = xml_path.parent
            external_files.append(_resolve_inside(root, base / raw, f"MJCF {node.tag} file"))
    files = sorted(set(xml_files + external_files), key=lambda path: path.relative_to(root).as_posix())
    rows = []
    for path in files:
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "algorithm": "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": sha256_bytes(payload.encode("utf-8")),
        "xml_file_count": len(xml_files),
        "include_reference_count": include_references,
        "external_file_reference_count": len(external_files),
        "unique_external_file_count": len(set(external_files)),
        "mesh_reference_count": mesh_references,
    }


def _resolve_directory_inside(root: Path, candidate: Path, label: str) -> Path:
    lexical_path = Path(os.path.abspath(candidate))
    probe = lexical_path
    while probe != probe.parent:
        if probe.exists() and probe.is_symlink():
            raise Schema2ContractError(f"{label} contains symlink component {probe}")
        probe = probe.parent
    resolved = lexical_path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise Schema2ContractError(f"{label} escapes model root: {candidate}") from exc
    if not resolved.is_dir():
        raise Schema2ContractError(f"{label} is not a real directory: {resolved}")
    return resolved


def _import_helpers():
    materializer = importlib.import_module("materialize_motion_spatial_se2")
    joint_order = importlib.import_module("a3_joint_order_contract")
    kinematics = importlib.import_module("motion_kinematics_contract")
    converter = importlib.import_module("csv_to_npz_mujoco")
    hope_frame = importlib.import_module("hope_frame_utils")
    return materializer, joint_order, kinematics, converter, hope_frame


def validate_donor_metadata_snapshot(
    path: Path, joint_contract, joint_module, *, repo_root: Path
) -> dict[str, Any]:
    snapshot = read_json(path, "donor metadata snapshot")
    exact_keys(
        snapshot,
        {
            "schema_version", "contract_id", "status", "source_onnx_sha256",
            "source_evidence", "required_custom_metadata_map", "honesty_boundary",
        },
        "donor metadata snapshot",
    )
    if snapshot["schema_version"] != 1 or snapshot["contract_id"] != "a3-schema2-fk-donor-metadata-v1":
        raise Schema2ContractError("donor metadata snapshot identity changed")
    if snapshot["status"] != "expected_metadata_bound_to_exact_onnx_runtime_inspection_not_run":
        raise Schema2ContractError("donor metadata snapshot overclaims runtime extraction")
    evidence_path = verify_repo_binding(
        snapshot["source_evidence"], "donor source_evidence", repo_root=repo_root
    )
    evidence = read_json(evidence_path, "donor source evidence")
    expected_onnx_sha = require_sha(snapshot["source_onnx_sha256"], "source_onnx_sha256")
    if evidence.get("formal_input", {}).get("exported_onnx_sha256") != expected_onnx_sha:
        raise Schema2ContractError("donor evidence does not bind the expected ONNX SHA")
    metadata = exact_keys(
        snapshot["required_custom_metadata_map"],
        {"joint_names", "articulation_joint_names", "action_joint_ids"},
        "required_custom_metadata_map",
    )
    joint_module.validate_runtime_metadata(metadata, joint_contract)
    boundary = exact_keys(
        snapshot["honesty_boundary"],
        {
            "full_onnx_is_content_bound_by_sha256",
            "tracked_map_is_required_subset_not_full_custom_metadata_dump",
            "metadata_reextracted_from_exact_onnx_in_this_source_gate",
            "next_gate",
        },
        "donor honesty_boundary",
    )
    if boundary != {
        "full_onnx_is_content_bound_by_sha256": True,
        "tracked_map_is_required_subset_not_full_custom_metadata_dump": True,
        "metadata_reextracted_from_exact_onnx_in_this_source_gate": False,
        "next_gate": "read_only_runtime_inspection_must_hash_exact_onnx_and_compare_required_subset_before_fk",
    }:
        raise Schema2ContractError("donor honesty boundary changed")
    return snapshot


def validate_shared_document(shared: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    exact_keys(
        shared,
        {
            "schema_version", "contract_id", "status", "scope", "consumer",
            "restricted_pickle", "donor_metadata", "vendor_mjcf_closure",
            "joint_order", "runtime_body_order", "converter_helper", "kinematics",
            "temporal", "frame", "schema2_output", "authorization",
        },
        "shared runtime contract",
    )
    if shared["schema_version"] != 1 or shared["contract_id"] != "motion-backhand-loop-bc-schema2-fk-v1":
        raise Schema2ContractError("shared runtime contract identity changed")
    if shared["status"] != SHARED_STATUS:
        raise Schema2ContractError("shared runtime contract status changed")
    expected_scope = (
        "CPU-only exact B/C GMR-pickle to schema-2 materialization with vendor MuJoCo forward "
        "kinematics; source gate only until separate read-only runtime inspection"
    )
    if shared["scope"] != expected_scope:
        raise Schema2ContractError("shared scope changed or overclaims")

    consumer_path = verify_repo_binding(shared["consumer"], "consumer", repo_root=repo_root)
    if consumer_path != (repo_root / "scripts/materialize_motion_schema2_fk.py").resolve():
        raise Schema2ContractError("shared consumer must bind this script")
    materializer, joint_module, kinematics_module, converter_module, hope_frame_module = _import_helpers()

    restricted = exact_keys(
        shared["restricted_pickle"],
        {"loader", "entrypoint", "allowed_globals", "required_keys", "optional_keys", "unknown_keys"},
        "restricted_pickle",
    )
    loader_path = verify_repo_binding(
        restricted["loader"], "restricted pickle loader", repo_root=repo_root,
        expected_path="scripts/materialize_motion_spatial_se2.py",
    )
    if loader_path.resolve() != Path(materializer.__file__).resolve():
        raise Schema2ContractError("imported restricted loader differs from bound source")
    if restricted["entrypoint"] != "load_bound_pickle":
        raise Schema2ContractError("restricted pickle entrypoint changed")
    actual_globals = sorted(f"{module}.{name}" for module, name in materializer.RestrictedNumpyUnpickler._ALLOWED)
    if restricted["allowed_globals"] != actual_globals:
        raise Schema2ContractError("restricted pickle global allowlist drifted")
    if restricted["required_keys"] != sorted(materializer.BASE_PAYLOAD_KEYS):
        raise Schema2ContractError("restricted pickle required keys drifted")
    if restricted["optional_keys"] != sorted(materializer.OPTIONAL_WORLD_VECTOR_FIELDS):
        raise Schema2ContractError("restricted pickle optional keys drifted")
    if restricted["unknown_keys"] != "fail_closed":
        raise Schema2ContractError("unknown pickle fields must fail closed")

    joint_binding = exact_keys(
        shared["joint_order"], {"contract", "validator"}, "joint_order"
    )
    joint_path = verify_repo_binding(
        joint_binding["contract"], "joint_order.contract", repo_root=repo_root,
        expected_path="configs/a3_joint_order_bijection_v1.json",
    )
    joint_validator_path = verify_repo_binding(
        joint_binding["validator"], "joint_order.validator", repo_root=repo_root,
        expected_path="hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py",
    )
    if joint_validator_path.resolve() != Path(joint_module.__file__).resolve():
        raise Schema2ContractError("imported joint-order validator differs from bound source")
    joint_contract = joint_module.load_contract(
        joint_binding["contract"]["path"], repo_root=repo_root
    )

    donor = exact_keys(
        shared["donor_metadata"],
        {"snapshot", "source_onnx_sha256", "runtime_policy"},
        "donor_metadata",
    )
    donor_path = verify_repo_binding(
        donor["snapshot"], "donor metadata snapshot", repo_root=repo_root,
        expected_path="configs/a3_schema2_fk_donor_metadata_v1.json",
    )
    snapshot = validate_donor_metadata_snapshot(
        donor_path, joint_contract, joint_module, repo_root=repo_root
    )
    if donor["source_onnx_sha256"] != snapshot["source_onnx_sha256"]:
        raise Schema2ContractError("shared donor SHA differs from metadata snapshot")
    if donor["runtime_policy"] != (
        "operator_supplies_regular_non_symlink_ONNX; exact SHA then required metadata subset "
        "are re-extracted before model/FK use"
    ):
        raise Schema2ContractError("donor runtime inspection policy changed")

    mjcf = exact_keys(
        shared["vendor_mjcf_closure"],
        {"model_root", "canonical_mjcf", "derived_closure"},
        "vendor_mjcf_closure",
    )
    root_raw = mjcf["model_root"]
    if not isinstance(root_raw, str) or Path(root_raw).is_absolute():
        raise Schema2ContractError("model_root must be repository-relative")
    model_root = (repo_root / root_raw).resolve()
    try:
        model_root.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise Schema2ContractError("model_root escapes repository") from exc
    if not model_root.is_dir() or model_root.is_symlink():
        raise Schema2ContractError("model_root must be a real directory")
    main_xml = verify_repo_binding(
        mjcf["canonical_mjcf"], "canonical_mjcf", repo_root=repo_root,
        expected_path=(
            "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
            "a3_pingpong/a3_pingpong.xml"
        ),
    )
    derived = derive_mjcf_closure(main_xml, model_root)
    if mjcf["derived_closure"] != derived:
        raise Schema2ContractError(
            f"vendor MJCF include/external closure drifted: actual={derived}"
        )

    body = exact_keys(
        shared["runtime_body_order"], {"source", "count", "names_sha256"}, "runtime_body_order"
    )
    body_path = verify_repo_binding(
        body["source"], "runtime body order", repo_root=repo_root,
        expected_path="configs/a3_runtime_body_order.txt",
    )
    if body["count"] != 32:
        raise Schema2ContractError("runtime body count must be 32")
    body_names = read_name_file(body_path, expected_count=32, label="runtime body order")
    if canonical_names_sha256(body_names) != body["names_sha256"]:
        raise Schema2ContractError("runtime body canonical names SHA drifted")
    xml_body_names = tuple(
        node.get("name") for node in ET.parse(main_xml).getroot().iter("body")
    )
    if set(body_names) != set(xml_body_names) or len(xml_body_names) != len(body_names):
        raise Schema2ContractError("runtime body order is not a bijection over vendor MJCF bodies")
    xml_joint_names = tuple(
        node.get("name") for node in ET.parse(main_xml).getroot().iter("joint")
    )
    if (
        len(xml_joint_names) != len(joint_contract.target_names)
        or set(xml_joint_names) != set(joint_contract.target_names)
    ):
        raise Schema2ContractError("runtime joint order is not a bijection over vendor MJCF joints")

    converter_path = verify_repo_binding(
        shared["converter_helper"], "converter_helper", repo_root=repo_root,
        expected_path="hope_training/whole_body_tracking/scripts/csv_to_npz_mujoco.py",
    )
    if converter_path.resolve() != Path(converter_module.__file__).resolve():
        raise Schema2ContractError("imported converter helper differs from bound source")
    kinematics_path = verify_repo_binding(
        shared["kinematics"], "kinematics", repo_root=repo_root,
        expected_path="hope_training/whole_body_tracking/scripts/motion_kinematics_contract.py",
    )
    if kinematics_path.resolve() != Path(kinematics_module.__file__).resolve():
        raise Schema2ContractError("imported kinematics helper differs from bound source")
    if (
        kinematics_module.KINEMATICS_SCHEMA_VERSION != 2
        or kinematics_module.BODY_POS_POINT != "link_origin"
        or kinematics_module.BODY_LIN_VEL_POINT != "center_of_mass"
    ):
        raise Schema2ContractError("imported schema-2 point semantics drifted")

    if shared["temporal"] != {
        "input_fps": 30,
        "output_fps": 50,
        "output_frame_formula": "round(((input_frames-1)/30)*50)+1",
        "root_position_and_dof": "piecewise_linear_on_source_phase_clipped_to_last_frame",
        "root_quaternion": "shortest_path_slerp_xyzw_input_converted_to_wxyz_before_slerp",
        "joint_and_body_linear_velocity": "numpy_gradient_dt_1_over_50",
        "body_angular_velocity": "SO3_central_difference_dt_1_over_50_endpoint_repeat",
        "topp_or_time_warp": False,
    }:
        raise Schema2ContractError("30->50 Hz temporal contract changed")
    frame = exact_keys(
        shared["frame"],
        {
            "evidence", "hope_frame_helper", "input_already_in_HOPE_frame", "required_cli",
            "second_HOPE_rotation", "root_rotation_input", "root_rotation_mujoco_and_output",
        },
        "frame",
    )
    frame_evidence_path = verify_repo_binding(
        frame["evidence"], "frame.evidence", repo_root=repo_root,
        expected_path="configs/motion_video_gmr_frame_contract_results_20260711.json",
    )
    hope_frame_path = verify_repo_binding(
        frame["hope_frame_helper"], "frame.hope_frame_helper", repo_root=repo_root,
        expected_path="hope_training/whole_body_tracking/scripts/hope_frame_utils.py",
    )
    if hope_frame_path.resolve() != Path(hope_frame_module.__file__).resolve():
        raise Schema2ContractError("imported HOPE-frame helper differs from bound source")
    if {key: frame[key] for key in frame if key not in {"evidence", "hope_frame_helper"}} != {
        "input_already_in_HOPE_frame": True,
        "required_cli": ["--hope_frame", "off"],
        "second_HOPE_rotation": "forbidden",
        "root_rotation_input": "xyzw",
        "root_rotation_mujoco_and_output": "wxyz",
    }:
        raise Schema2ContractError("HOPE/root-frame contract changed")
    if shared["schema2_output"] != {
        "kinematics_schema_version": 2,
        "joint_pos_order": "runtime_articulation_joint_pos",
        "body_columns": "exact_runtime_body_order_file",
        "body_pos_w_point": "link_origin_from_MuJoCo_xpos",
        "body_quat_w": "link_frame_wxyz_from_MuJoCo_xquat",
        "body_lin_vel_w_point": "center_of_mass_gradient_from_MuJoCo_xipos",
        "body_ang_vel_w": "link_frame_angular_velocity_point_independent",
        "dtype": "float32_for_time_series",
        "finite_required": True,
    }:
        raise Schema2ContractError("schema-2 output semantics changed")
    if shared["authorization"] != {
        "runtime_inspection_authorized": True,
        "schema2_materialization_authorized_after_inspection_only": True,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }:
        raise Schema2ContractError("shared authorization changed")
    return {
        "joint_contract": joint_contract,
        "body_names": body_names,
        "donor_snapshot": snapshot,
        "frame_evidence": read_json(frame_evidence_path, "HOPE-frame evidence"),
        "main_xml": main_xml,
        "model_root": model_root,
        "closure": derived,
    }


def validate_plan(
    path: Path, expected_sha: str, *, repo_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    expected_sha = require_sha(expected_sha, "expected prereg SHA")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise Schema2ContractError(f"prereg SHA {actual_sha} != {expected_sha}")
    plan = read_json(path, "schema-2/FK preregistration")
    exact_keys(
        plan,
        {
            "schema_version", "plan_id", "status", "asset_id", "human_name", "scope",
            "shared_runtime", "source_result_ledger", "source_motion",
            "source_materialization_report", "source_structure", "output_contract",
            "authorization", "failure_policy", "next_gate",
        },
        "schema-2/FK preregistration",
    )
    if plan["schema_version"] != 1 or plan["status"] != PLAN_STATUS:
        raise Schema2ContractError("plan must remain source-gate-only preregistration")
    asset = plan["asset_id"]
    if asset not in ASSET_IDS:
        raise Schema2ContractError("plan asset must be frozen B or C")
    suffix = asset.rsplit("_", 1)[-1]
    expected_plan_id = f"motion-{asset.replace('_', '-')}-schema2-fk-20260714-v1"
    if plan["plan_id"] != expected_plan_id:
        raise Schema2ContractError("plan_id changed")
    if plan["human_name"] != f"Franco 反手拉候选 {suffix.upper()} exact schema-2/FK 实体化":
        raise Schema2ContractError("human_name changed")
    if plan["scope"] != (
        f"exact private SE(2) {suffix.upper()} GMR pickle to independent no-clobber schema-2 NPZ; "
        "source gate only, no simulator/training/hardware"
    ):
        raise Schema2ContractError("plan scope changed")
    shared_path = verify_repo_binding(
        plan["shared_runtime"], "shared_runtime", repo_root=repo_root,
        expected_path="configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json",
    )
    shared = read_json(shared_path, "shared schema-2/FK runtime contract")
    shared_evidence = validate_shared_document(shared, repo_root=repo_root)
    result_path = verify_repo_binding(
        plan["source_result_ledger"], "source_result_ledger", repo_root=repo_root,
        expected_path="configs/motion_backhand_loop_bc_se2_materialization_results_20260714.json",
    )
    result = read_json(result_path, "SE(2) result ledger")
    if result.get("status") != "complete_exact_pair_materialized_certificate_blocked":
        raise Schema2ContractError("SE(2) result ledger is not the accepted blocked pair")
    rows = [row for row in result.get("assets", []) if row.get("asset_id") == asset]
    if len(rows) != 1:
        raise Schema2ContractError(f"SE(2) result must contain exactly one row for {asset}")
    row = rows[0]
    if (
        row.get("report", {}).get("status")
        != "complete_exact_whole_motion_se2_materialization"
        or row.get("transform", {}).get("mirror") is not False
        or row.get("transform", {}).get("joint_edit") is not False
        or row.get("transform", {}).get("per_frame_edit") is not False
        or row.get("transform", {}).get("time_edit_or_topp") is not False
        or row.get("invariants", {}).get("no_mirror") is not True
        or row.get("invariants", {}).get("no_resample_or_topp") is not True
        or row.get("authorization", {}).get("schema2_materialized") is not False
    ):
        raise Schema2ContractError("SE(2) source row is not an exact unretimed blocked materialization")
    source = validate_absolute_binding(plan["source_motion"], "source_motion", verify=False)
    if plan["source_motion"] != row.get("output_motion"):
        raise Schema2ContractError("source_motion differs from accepted SE(2) result")
    report = validate_absolute_binding(
        plan["source_materialization_report"], "source_materialization_report", verify=False
    )
    # The result row additionally records a semantic status; the prereg binding carries exact file identity.
    row_report = {key: row["report"][key] for key in ("path", "bytes", "sha256")}
    if plan["source_materialization_report"] != row_report:
        raise Schema2ContractError("source report differs from accepted SE(2) result")
    structure = exact_keys(
        plan["source_structure"],
        {"frames", "input_fps", "expected_output_frames", "root_rotation", "already_in_HOPE_frame"},
        "source_structure",
    )
    expected_frames = 91 if asset.endswith("_b") else 98
    expected_output = int(round(((expected_frames - 1) / 30.0) * 50.0)) + 1
    if structure != {
        "frames": expected_frames,
        "input_fps": 30,
        "expected_output_frames": expected_output,
        "root_rotation": "xyzw",
        "already_in_HOPE_frame": True,
    }:
        raise Schema2ContractError("source/output frame contract changed")
    if row.get("structure", {}).get("frames") != expected_frames or row.get("structure", {}).get("fps") != 30:
        raise Schema2ContractError("source result structure differs from plan")
    frame_evidence = shared_evidence["frame_evidence"]
    if (
        frame_evidence.get("status") != "complete_verified_per_asset_hope_frame_and_not_mirrored"
        or frame_evidence.get("frame_contract", {}).get("status") != "verified"
        or frame_evidence.get("frame_contract", {}).get("gmr_world_to_hope_table_transform_verified") is not True
        or frame_evidence.get("frame_contract", {}).get("target_frame")
        != "HOPE env-local robot origin, +X forward, +Y left, +Z up"
    ):
        raise Schema2ContractError("bound frame evidence does not prove canonical HOPE axes")
    frame_rows = [
        candidate for candidate in frame_evidence.get("assets", [])
        if isinstance(candidate, dict) and candidate.get("asset_id") == asset
    ]
    if len(frame_rows) != 1:
        raise Schema2ContractError(f"frame evidence must contain exactly one {asset} row")
    frame_row = frame_rows[0]
    if (
        frame_row.get("frames") != expected_frames
        or frame_row.get("fps") != 30.0
        or frame_row.get("mirror_status") != "verified_not_mirrored"
        or frame_row.get("side_swap_required") is not False
        or frame_row.get("grounded_gmr", {}).get("sha256")
        != row.get("source_motion", {}).get("sha256")
    ):
        raise Schema2ContractError("frame-evidence row does not match the SE(2) input lineage")
    output = exact_keys(
        plan["output_contract"],
        {"output_root", "motion_filename", "report_filename", "output_root_must_not_exist", "no_clobber", "report_published_last"},
        "output_contract",
    )
    expected_root = (
        "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
        f"{asset}_{row['candidate_id'][:12]}"
    )
    if output != {
        "output_root": expected_root,
        "motion_filename": f"{asset}.{row['candidate_id'][:12]}.schema2_fk.npz",
        "report_filename": "schema2_fk_report.json",
        "output_root_must_not_exist": True,
        "no_clobber": True,
        "report_published_last": True,
    }:
        raise Schema2ContractError("output no-clobber contract changed")
    if plan["authorization"] != {
        "source_gate_pass": True,
        "runtime_inspection_run": False,
        "schema2_materialized": False,
        "l0_authorized": False,
        "vendor_l1_authorized": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }:
        raise Schema2ContractError("plan authorization changed")
    if plan["failure_policy"] != {
        "source_or_internal_failure": "stop_this_asset_no_fallback",
        "external_table_or_net_failure_after_schema2": "return_to_frozen_selector_only",
        "automatic_retry_or_fallback": False,
    }:
        raise Schema2ContractError("failure policy changed")
    if plan["next_gate"] != {
        "authorized": "read_only_runtime_inspection_of_exact_pickle_donor_and_MJCF_only",
        "after_inspection": "one_no_clobber_schema2_FK_materialization_then_L0_static_audit",
        "status": "runtime_inspection_not_run",
    }:
        raise Schema2ContractError("next gate changed or overclaims")
    return plan, actual_sha, {**shared_evidence, "shared": shared, "row": row, "source": source, "report": report}


def validate_pair(
    first: tuple[dict[str, Any], str, dict[str, Any]],
    second: tuple[dict[str, Any], str, dict[str, Any]],
) -> None:
    plans = [first[0], second[0]]
    if {plan["asset_id"] for plan in plans} != set(ASSET_IDS):
        raise Schema2ContractError("pair must contain exactly one B and one C plan")
    if plans[0]["shared_runtime"] != plans[1]["shared_runtime"]:
        raise Schema2ContractError("B/C plans must bind the same shared runtime")
    outputs = [Path(plan["output_contract"]["output_root"]) for plan in plans]
    if outputs[0] == outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        raise Schema2ContractError("B/C output namespaces overlap")
    sources = [plan["source_motion"]["sha256"] for plan in plans]
    if len(set(sources)) != 2:
        raise Schema2ContractError("B/C source motion identities must be distinct")


def inspect_runtime(
    plan: Mapping[str, Any], evidence: Mapping[str, Any], donor_path: Path
) -> dict[str, Any]:
    materializer, _joint_module, _kinematics, converter, _hope_frame = _import_helpers()
    source = validate_absolute_binding(plan["source_motion"], "source_motion", verify=True)
    validate_absolute_binding(
        plan["source_materialization_report"], "source_materialization_report", verify=True
    )
    payload = materializer.load_bound_pickle(source)
    structure = materializer.validate_payload(payload, frames=plan["source_structure"]["frames"])
    if structure["fps"] != 30 or structure["root_rotation_convention"] != "xyzw":
        raise Schema2ContractError("private GMR structure differs from exact plan")
    output_root = Path(plan["output_contract"]["output_root"])
    if output_root.exists():
        raise Schema2ContractError(f"output root already exists before inspection: {output_root}")
    materializer.ensure_no_symlink_components(output_root.parent, "output root parent")
    ensure_regular_no_symlink(donor_path, "donor ONNX")
    expected_donor_sha = evidence["shared"]["donor_metadata"]["source_onnx_sha256"]
    actual_donor_sha = sha256_file(donor_path)
    if actual_donor_sha != expected_donor_sha:
        raise Schema2ContractError(f"donor ONNX SHA {actual_donor_sha} != {expected_donor_sha}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise Schema2ContractError("onnxruntime is required for runtime inspection") from exc
    try:
        metadata = ort.InferenceSession(
            str(donor_path), providers=["CPUExecutionProvider"]
        ).get_modelmeta().custom_metadata_map
    except Exception as exc:
        raise Schema2ContractError(f"cannot inspect exact donor ONNX: {exc}") from exc
    required = evidence["donor_snapshot"]["required_custom_metadata_map"]
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise Schema2ContractError(f"donor metadata {key} differs from bound subset")
    try:
        fkm = converter.MjFK(
            str(evidence["main_xml"]), list(evidence["joint_contract"].target_names)
        )
    except Exception as exc:
        raise Schema2ContractError(f"cannot load exact vendor MJCF for FK: {exc}") from exc
    runtime_bodies = fkm.body_names()
    if set(runtime_bodies) != set(evidence["body_names"]) | {"world"}:
        raise Schema2ContractError("loaded MuJoCo body names differ from bound runtime domain")
    if any(name not in runtime_bodies for name in evidence["body_names"]):
        raise Schema2ContractError("loaded MuJoCo lacks a bound runtime body")
    return {
        "payload": payload,
        "structure": structure,
        "donor": {
            "path": str(donor_path),
            "bytes": donor_path.stat().st_size,
            "sha256": actual_donor_sha,
            "required_metadata_subset_exact": True,
        },
        "fkm": fkm,
    }


def resample_payload(payload: Mapping[str, Any], *, output_fps: int = 50):
    _materializer, _joint_module, _kinematics, converter, _hope_frame = _import_helpers()
    input_fps = int(payload["fps"])
    if input_fps != 30 or output_fps != 50:
        raise Schema2ContractError("only frozen 30->50 Hz resampling is supported")
    root_pos_in = np.asarray(payload["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
    dof_in = np.asarray(payload["dof_pos"], dtype=np.float32)
    n_in = root_pos_in.shape[0]
    duration = (n_in - 1) / float(input_fps)
    n_out = int(round(duration * output_fps)) + 1
    times = np.arange(n_out, dtype=np.float64) / output_fps
    phase = np.clip(times * input_fps, 0.0, n_in - 1)
    idx0 = np.floor(phase).astype(int)
    idx1 = np.minimum(idx0 + 1, n_in - 1)
    blend = (phase - idx0).astype(np.float32)[:, None]
    root_pos = root_pos_in[idx0] * (1.0 - blend) + root_pos_in[idx1] * blend
    dof_source = dof_in[idx0] * (1.0 - blend) + dof_in[idx1] * blend
    root_rot_wxyz_in = root_rot_xyzw[:, [3, 0, 1, 2]]
    root_rot = converter.quat_slerp(
        root_rot_wxyz_in[idx0], root_rot_wxyz_in[idx1], blend
    ).astype(np.float32)
    if not all(np.isfinite(array).all() for array in (root_pos, root_rot, dof_source)):
        raise Schema2ContractError("resampled source contains NaN/Inf")
    return root_pos, root_rot, dof_source


def build_schema2_arrays(
    plan: Mapping[str, Any], evidence: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    _materializer, joint_module, kinematics, converter, _hope_frame = _import_helpers()
    root_pos, root_rot, dof_source = resample_payload(runtime["payload"])
    joint_contract = evidence["joint_contract"]
    dof = joint_module.reorder_source_to_target(dof_source, joint_contract).astype(np.float32)
    dt = 1.0 / 50.0
    joint_vel = np.gradient(dof, dt, axis=0).astype(np.float32)
    fkm = runtime["fkm"]
    pos_all, quat_all, com_all = converter.fk_series_with_com(
        fkm, root_pos, root_rot, dof, list(joint_contract.target_names)
    )
    runtime_names = fkm.body_names()
    columns = [runtime_names.index(name) for name in evidence["body_names"]]
    body_pos = pos_all[:, columns].astype(np.float32)
    body_quat = quat_all[:, columns].astype(np.float32)
    body_com = com_all[:, columns].astype(np.float32)
    body_lin = np.gradient(body_com, dt, axis=0).astype(np.float32)
    body_ang = np.stack(
        [converter.so3_derivative(body_quat[:, index], dt) for index in range(body_quat.shape[1])],
        axis=1,
    ).astype(np.float32)
    arrays = {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": dof,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        **kinematics.metadata_arrays(body_names=evidence["body_names"]),
    }
    frames = plan["source_structure"]["expected_output_frames"]
    expected_shapes = {
        "joint_pos": (frames, 31), "joint_vel": (frames, 31),
        "body_pos_w": (frames, 32, 3), "body_quat_w": (frames, 32, 4),
        "body_lin_vel_w": (frames, 32, 3), "body_ang_vel_w": (frames, 32, 3),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape or arrays[key].dtype != np.float32:
            raise Schema2ContractError(f"{key} shape/dtype {arrays[key].shape}/{arrays[key].dtype} != {shape}/float32")
        if not np.isfinite(arrays[key]).all():
            raise Schema2ContractError(f"{key} contains NaN/Inf")
    if not np.allclose(np.linalg.norm(body_quat.astype(np.float64), axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise Schema2ContractError("FK body quaternion normalization failed")
    return arrays


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def consume(
    plan: Mapping[str, Any], plan_path: Path, plan_sha: str,
    evidence: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Path:
    materializer, _joint_module, kinematics, _converter, _hope_frame = _import_helpers()
    output = plan["output_contract"]
    output_root = Path(output["output_root"])
    if output_root.exists():
        raise Schema2ContractError(f"output root already exists: {output_root}")
    materializer.ensure_no_symlink_components(output_root.parent, "output root parent")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    materializer.ensure_no_symlink_components(output_root.parent, "output root parent")
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    try:
        staging.mkdir()
    except FileExistsError as exc:
        raise Schema2ContractError(f"staging already exists: {staging}") from exc
    motion_name = output["motion_filename"]
    report_name = output["report_filename"]
    try:
        arrays = build_schema2_arrays(plan, evidence, runtime)
        motion_path = staging / motion_name
        write_npz_exclusive(motion_path, arrays)
        with np.load(motion_path, allow_pickle=False) as reloaded:
            metadata = kinematics.read_metadata(reloaded)
            if not metadata.exact_motion_command_v2 or metadata.body_names != evidence["body_names"]:
                raise Schema2ContractError("serialized NPZ lacks exact schema-2 metadata")
            for key, expected in arrays.items():
                if key not in reloaded.files or not np.array_equal(reloaded[key], expected):
                    raise Schema2ContractError(f"serialized NPZ field changed: {key}")
        report = {
            "schema_version": 1,
            "status": RESULT_STATUS,
            "completed_utc": utc_now(),
            "scope": "exact schema-2 MuJoCo FK materialization only; no L0/L1, table/net, dynamics, simulator, training, formal-motion or hardware claim",
            "asset_id": plan["asset_id"],
            "preregistration": {"path": str(plan_path), "sha256": plan_sha},
            "shared_runtime": plan["shared_runtime"],
            "source_motion": plan["source_motion"],
            "source_materialization_report": plan["source_materialization_report"],
            "donor": runtime["donor"],
            "vendor_mjcf_closure": evidence["closure"],
            "output_motion": {
                "path": str(output_root / motion_name),
                "bytes": motion_path.stat().st_size,
                "sha256": sha256_file(motion_path),
            },
            "structure": {
                "input_frames": plan["source_structure"]["frames"],
                "input_fps": 30,
                "output_frames": plan["source_structure"]["expected_output_frames"],
                "output_fps": 50,
                "hope_frame": "off",
                "kinematics_schema_version": 2,
                "body_pos_point": "link_origin",
                "body_lin_vel_point": "center_of_mass",
                "joint_count": 31,
                "body_count": 32,
                "finite": True,
            },
            "authorization": {
                "schema2_materialized": True,
                "l0_authorized": True,
                "vendor_l1_authorized": False,
                "table_net_authorized": False,
                "dynamics_authorized": False,
                "simulator_authorized": False,
                "training_authorized": False,
                "formal_motion_authorized": False,
                "hardware_authorized": False,
            },
            "next_gate": "independent_L0_static_schema2_audit_then_vendor_L1_self_collision",
        }
        materializer._write_exclusive(staging / report_name, json_bytes(report))
        materializer.publish_report_last(staging, output_root, motion_name, report_name)
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / report_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--peer-prereg", type=Path, required=True)
    parser.add_argument("--expected-peer-prereg-sha256", required=True)
    parser.add_argument("--hope_frame", required=True, choices=("off",))
    parser.add_argument("--donor", type=Path)
    parser.add_argument("command", choices=("static", "inspect", "consume"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        primary = validate_plan(args.prereg.resolve(), args.expected_prereg_sha256)
        peer = validate_plan(args.peer_prereg.resolve(), args.expected_peer_prereg_sha256)
        validate_pair(primary, peer)
        plan, plan_sha, evidence = primary
        if args.command == "static":
            if args.donor is not None:
                raise Schema2ContractError("static source gate must not receive --donor")
            print(
                f"[schema2-fk] PASS static asset={plan['asset_id']} pair_exact=true "
                f"hope_frame=off prereg_sha256={plan_sha} runtime_inspection=false"
            )
            return 0
        if args.donor is None:
            raise Schema2ContractError("inspect/consume requires --donor")
        runtime = inspect_runtime(plan, evidence, args.donor.resolve())
        if args.command == "inspect":
            print(
                f"[schema2-fk] PASS inspect asset={plan['asset_id']} "
                f"frames={runtime['structure']['frames']} donor_exact=true no_write=true"
            )
            return 0
        report = consume(plan, args.prereg.resolve(), plan_sha, evidence, runtime)
        print(f"[schema2-fk] PASS consume report={report}")
        return 0
    except (Schema2ContractError, OSError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"[schema2-fk] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
