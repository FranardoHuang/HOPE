#!/usr/bin/env python3
"""Audit the canonical GMR -> HOPE +X/table frame and video mirror contract.

This is a CPU-only, offline evidence gate for the ten 2026-07-11 air swings.
It does not infer a table from the videos.  Instead it proves an explicit
counterfactual normalization used by HOPE training:

* preserve the already-audited GMR/MuJoCo ground plane (source/target z=0);
* map each clip's frame-0 A3 pelvis XY to the environment origin;
* yaw the frame-0 A3 pelvis heading to target +X with a proper rotation;
* bind the standard HOPE virtual-table pose relative to that robot origin.

Mirror evidence is deliberately separate.  ``witnesses`` decodes one bound
frame per source video and hashes an asymmetric background-text crop.  A human
review ledger may then attest that the Chinese glyphs are upright and
unreflected.  ``run`` accepts that review only when every crop SHA still
matches and the retargeted GMR is independently right-arm dominant.

The output can authorize the existing immutable *diagnostic* 64-question
phase/coverage screen.  It never supplies observed ball/contact truth, schema-2
motion, dynamics, TOPP approval, RL approval, or real-robot approval.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

LEFT_ARM = slice(5, 12)
RIGHT_ARM = slice(12, 19)


class FrameContractError(ValueError):
    """The requested frame/mirror evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameContractError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise FrameContractError(f"{label} must be a JSON object")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise FrameContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_number(value: Any, label: str, *, low: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrameContractError(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out) or (low is not None and out < low):
        raise FrameContractError(
            f"{label} must be finite" + (f" and >= {low}" if low is not None else "")
        )
    return out


def _verify_binding(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise FrameContractError(f"{label} must contain path/bytes/sha256")
    expected_sha = _require_sha(value.get("sha256"), f"{label}.sha256")
    expected_bytes = value.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise FrameContractError(f"{label}.bytes must be positive")
    path = Path(value["path"]).expanduser().resolve()
    if not path.is_file():
        raise FrameContractError(f"{label} is missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise FrameContractError(
            f"{label} bytes {path.stat().st_size} != {expected_bytes}: {path}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise FrameContractError(f"{label} sha256 {actual_sha} != {expected_sha}: {path}")
    return path


def _load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_sha(expected_sha256, "--expected-manifest-sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise FrameContractError(f"manifest sha256 {actual_sha} != {expected_sha256}")
    plan = _read_json(path, "frame-contract manifest")
    if plan.get("schema_version") != 1:
        raise FrameContractError("frame-contract manifest schema_version must be 1")
    if plan.get("plan_id") != "motion-video-gmr-frame-contract-20260711-v1":
        raise FrameContractError("unexpected frame-contract plan_id")
    if plan.get("cpu_only") is not True or plan.get("CUDA_VISIBLE_DEVICES") != "":
        raise FrameContractError("frame audit must be CPU-only with CUDA_VISIBLE_DEVICES empty")
    if plan.get("real_robot_commands_authorized") is not False:
        raise FrameContractError("real_robot_commands_authorized must be false")
    if plan.get("contact_phase_truth") is not None:
        raise FrameContractError("air-swing contact_phase_truth must remain null")
    if plan.get("input_mode") != "manifest_rows_only_no_directory_scan":
        raise FrameContractError("input_mode must forbid directory scanning")
    if plan.get("expected_asset_ids") != [
        "franco_forehand_block",
        "franco_backhand_block",
        "franco_forehand_loop",
        "franco_backhand_loop_a",
        "franco_backhand_loop_b",
        "franco_backhand_loop_c",
        "v6_forehand_block",
        "v6_backhand_block",
        "v7_forehand_block",
        "v7_backhand_block",
    ]:
        raise FrameContractError("expected_asset_ids changed")

    for name in (
        "intake_manifest",
        "canonical_grounding_result",
        "a3_mjcf",
        "hope_frame_interface",
        "racket_geometry_interface",
        "table_geometry_source",
    ):
        _verify_binding(plan.get("source_bindings", {}).get(name), f"source_bindings.{name}")

    tool = _verify_binding(plan.get("tool_contract", {}).get("audit"), "tool_contract.audit")
    if tool.name != Path(__file__).name:
        raise FrameContractError("tool_contract.audit basename mismatch")

    transform = plan.get("transform_contract")
    if not isinstance(transform, dict):
        raise FrameContractError("transform_contract must be an object")
    if transform.get("method") != "per_asset_frame0_pelvis_xy_origin_heading_to_plus_x_v1":
        raise FrameContractError("unexpected transform method")
    if transform.get("source_ground_plane") != "z=0 from bound canonical grounding result":
        raise FrameContractError("source ground-plane contract changed")
    if transform.get("target_axes") != {"x": "forward", "y": "left", "z": "up"}:
        raise FrameContractError("target axis contract changed")
    _require_number(transform.get("proper_rotation_tolerance"), "proper_rotation_tolerance", low=0)
    _require_number(transform.get("heading_tolerance_rad"), "heading_tolerance_rad", low=0)
    _require_number(transform.get("mapped_root_xy_tolerance_m"), "mapped_root_xy_tolerance_m", low=0)

    table = plan.get("target_table_contract")
    if not isinstance(table, dict) or table.get("pose_semantics") != (
        "canonical_counterfactual_HOPE_virtual_table_relative_to_robot_origin_not_capture_extrinsic"
    ):
        raise FrameContractError("target table semantics changed")
    expected_table = {
        "near_edge_x_m": 0.5,
        "surface_z_m": 0.76,
        "center_y_m": 0.0,
        "length_m": 2.74,
        "width_m": 1.525,
        "net_height_m": 0.1525,
    }
    for key, wanted in expected_table.items():
        got = _require_number(table.get(key), f"target_table_contract.{key}")
        if not math.isclose(got, wanted, abs_tol=1e-12, rel_tol=0.0):
            raise FrameContractError(f"target table {key}={got} != {wanted}")

    mirror = plan.get("mirror_contract")
    if not isinstance(mirror, dict):
        raise FrameContractError("mirror_contract must be an object")
    if mirror.get("witness_frame_rule") != "floor(frame_count/2)":
        raise FrameContractError("mirror witness frame rule changed")
    if mirror.get("review_assertion_required") != (
        "upright_unreflected_chinese_glyphs_in_bound_background_crop"
    ):
        raise FrameContractError("mirror review assertion changed")
    crop = mirror.get("crop_rgb_pixels")
    if crop != {"x": 1350, "y": 240, "width": 550, "height": 650}:
        raise FrameContractError("mirror witness crop changed")
    _require_number(mirror.get("minimum_right_left_arm_motion_energy_ratio"), "mirror ratio", low=1)
    if mirror.get("accepted_status") != "verified_not_mirrored":
        raise FrameContractError("accepted mirror status changed")

    git_source = plan.get("committed_task_source")
    if not isinstance(git_source, dict):
        raise FrameContractError("committed_task_source must be an object")
    _require_sha(git_source.get("content_sha256"), "committed_task_source.content_sha256")
    if not isinstance(git_source.get("commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", git_source["commit"]
    ):
        raise FrameContractError("committed_task_source.commit must be a full git SHA")
    if not isinstance(git_source.get("blob_oid"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", git_source["blob_oid"]
    ):
        raise FrameContractError("committed_task_source.blob_oid must be a full git OID")
    return plan


def _ast_number(source: str, class_name: str | None, variable: str) -> float:
    tree = ast.parse(source)
    nodes: list[ast.stmt]
    if class_name is None:
        nodes = tree.body
    else:
        matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        if len(matches) != 1:
            raise FrameContractError(f"cannot find exactly one class {class_name!r}")
        nodes = matches[0].body
    values: list[float] = []
    for node in nodes:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == variable and value is not None:
            try:
                materialized = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            if isinstance(materialized, (int, float)) and not isinstance(materialized, bool):
                values.append(float(materialized))
    if len(values) != 1:
        raise FrameContractError(
            f"cannot extract exactly one literal {class_name + '.' if class_name else ''}{variable}"
        )
    return values[0]


def _git_source(plan: dict[str, Any]) -> str:
    spec = plan["committed_task_source"]
    command = ["git", "show", f"{spec['commit']}:{spec['path']}"]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, check=False)
    if result.returncode != 0:
        raise FrameContractError(
            f"cannot load committed task source: {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    if len(result.stdout) != spec["bytes"]:
        raise FrameContractError("committed task source byte count changed")
    if hashlib.sha256(result.stdout).hexdigest() != spec["content_sha256"]:
        raise FrameContractError("committed task source SHA changed")
    oid = subprocess.run(
        ["git", "rev-parse", f"{spec['commit']}:{spec['path']}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if oid != spec["blob_oid"]:
        raise FrameContractError(f"committed task blob {oid} != {spec['blob_oid']}")
    return result.stdout.decode("utf-8")


def _validate_source_semantics(plan: dict[str, Any]) -> dict[str, Any]:
    bindings = plan["source_bindings"]
    mjcf_path = Path(bindings["a3_mjcf"]["path"]).expanduser().resolve()
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    floor = root.find("./worldbody/geom[@name='floor']")
    if floor is None or floor.attrib.get("type") != "plane":
        raise FrameContractError("bound A3 MJCF lacks the world floor plane")
    floor_pos = [float(x) for x in floor.attrib.get("pos", "0 0 0").split()]
    if len(floor_pos) != 3 or not np.allclose(floor_pos, [0, 0, 0], atol=0, rtol=0):
        raise FrameContractError(f"A3 floor must be at z=0, got {floor_pos}")
    site = root.find(".//site[@name='right_racket']")
    if site is None:
        raise FrameContractError("bound A3 MJCF lacks right_racket site")
    site_pos = [float(x) for x in site.attrib.get("pos", "").split()]
    if not np.allclose(site_pos, [0.21021, 0.032078, 0.032036], atol=1e-12, rtol=0):
        raise FrameContractError(f"right_racket site position changed: {site_pos}")
    parents = {child: parent for parent in root.iter() for child in parent}
    ancestor_names: list[str] = []
    node: ET.Element | None = site
    while node in parents:
        node = parents[node]
        if node.tag == "body" and "name" in node.attrib:
            ancestor_names.append(node.attrib["name"])
    if "right_wrist_yaw_Link" not in ancestor_names:
        raise FrameContractError("right_racket site is not under right_wrist_yaw_Link")
    left = root.find(".//body[@name='left_shoulder_pitch_Link']")
    right = root.find(".//body[@name='right_shoulder_pitch_Link']")
    if left is None or right is None:
        raise FrameContractError("A3 shoulder anchors are missing")
    left_y = float(left.attrib["pos"].split()[1])
    right_y = float(right.attrib["pos"].split()[1])
    if not left_y > 0 > right_y:
        raise FrameContractError("A3 +Y must point to the robot's anatomical left")

    geometry_source = Path(bindings["table_geometry_source"]["path"]).read_text(encoding="utf-8")
    length = _ast_number(geometry_source, None, "TABLE_LENGTH")
    width = _ast_number(geometry_source, None, "TABLE_WIDTH")
    net_height = _ast_number(geometry_source, None, "NET_HEIGHT")
    committed = _git_source(plan)
    near_x = _ast_number(committed, "RacketTargetCommandCfg", "vb_table_near_x")
    surface_z = _ast_number(committed, "RacketTargetCommandCfg", "vb_table_surface_z")
    expected = plan["target_table_contract"]
    observed = {
        "near_edge_x_m": near_x,
        "surface_z_m": surface_z,
        "center_y_m": 0.0,
        "length_m": length,
        "width_m": width,
        "net_height_m": net_height,
        "net_x_m": near_x + length / 2.0,
        "far_edge_x_m": near_x + length,
        "half_width_m": width / 2.0,
    }
    for key in ("near_edge_x_m", "surface_z_m", "center_y_m", "length_m", "width_m", "net_height_m"):
        if not math.isclose(observed[key], float(expected[key]), abs_tol=1e-12, rel_tol=0):
            raise FrameContractError(f"table source {key}={observed[key]} != plan {expected[key]}")
    return {
        "source_floor_z_m": floor_pos[2],
        "target_axes": {"x": "forward", "y": "left", "z": "up"},
        "a3_shoulder_y_m": {"left": left_y, "right": right_y},
        "right_racket_site_local_pos_m": site_pos,
        "right_racket_site_ancestors": ancestor_names,
        "right_racket_face_axis": "site local +Y from bound racket geometry interface",
        "target_table": observed,
        "capture_table_pose_observed": False,
        "table_pose_semantics": expected["pose_semantics"],
    }


def _asset_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    intake_path = Path(plan["source_bindings"]["intake_manifest"]["path"]).expanduser().resolve()
    ground_path = Path(plan["source_bindings"]["canonical_grounding_result"]["path"]).expanduser().resolve()
    intake = _read_json(intake_path, "intake manifest")
    ground = _read_json(ground_path, "grounding result")
    expected = plan["expected_asset_ids"]
    if intake.get("processing_order") != expected:
        raise FrameContractError("intake processing order disagrees with expected_asset_ids")
    intake_rows = intake.get("assets")
    ground_rows = ground.get("results")
    if not isinstance(intake_rows, list) or not isinstance(ground_rows, list):
        raise FrameContractError("source manifests lack asset/result rows")
    intake_by_id = {row.get("id"): row for row in intake_rows if isinstance(row, dict)}
    ground_by_id = {row.get("asset_id"): row for row in ground_rows if isinstance(row, dict)}
    if set(intake_by_id) != set(expected) or set(ground_by_id) != set(expected):
        raise FrameContractError("source manifests do not contain exactly the ten expected assets")
    return [intake_by_id[value] for value in expected], ground_by_id


def _verify_video(plan: dict[str, Any], row: dict[str, Any]) -> Path:
    root = Path(plan["private_roots"]["video_source_root"]).expanduser().resolve()
    rel = Path(row["source_relpath"])
    if rel.is_absolute() or ".." in rel.parts:
        raise FrameContractError(f"unsafe source video path {rel}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise FrameContractError(f"source video escapes root: {rel}") from None
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
        raise FrameContractError(f"video binding mismatch: {path}")
    return path


def _verify_gmr(plan: dict[str, Any], row: dict[str, Any]) -> Path:
    root = Path(plan["private_roots"]["grounded_gmr_root"]).expanduser().resolve()
    binding = row.get("output")
    if not isinstance(binding, dict):
        raise FrameContractError(f"ground row {row.get('asset_id')} lacks output binding")
    name = Path(str(binding.get("path"))).name
    if not name or name in {".", ".."}:
        raise FrameContractError("invalid grounded GMR basename")
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise FrameContractError(f"grounded GMR escapes root: {name}") from None
    if not path.is_file() or path.stat().st_size != binding["bytes"] or sha256_file(path) != binding["sha256"]:
        raise FrameContractError(f"grounded GMR binding mismatch: {path}")
    return path


def _ffmpeg_version(executable: str) -> str:
    result = subprocess.run([executable, "-version"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise FrameContractError(f"cannot execute ffmpeg: {executable}")
    return result.stdout.splitlines()[0]


def _decode_rgb_frame(
    executable: str, path: Path, frame: int, width: int, height: int
) -> np.ndarray:
    command = [
        executable,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"select=eq(n\\,{frame})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise FrameContractError(
            f"ffmpeg failed for {path}: {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    expected = int(width) * int(height) * 3
    if len(result.stdout) != expected:
        raise FrameContractError(
            f"decoded frame bytes {len(result.stdout)} != {expected} for {path} frame {frame}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(int(height), int(width), 3)


def collect_witnesses(plan: dict[str, Any]) -> dict[str, Any]:
    intake_rows, _ = _asset_rows(plan)
    ffmpeg = str(Path(plan["ffmpeg_executable"]).expanduser().resolve())
    version = _ffmpeg_version(ffmpeg)
    crop = plan["mirror_contract"]["crop_rgb_pixels"]
    x, y, width, height = (crop[k] for k in ("x", "y", "width", "height"))
    results: list[dict[str, Any]] = []
    for row in intake_rows:
        path = _verify_video(plan, row)
        media = row["media"]
        frame = int(media["frames"]) // 2
        rgb = _decode_rgb_frame(ffmpeg, path, frame, int(media["width"]), int(media["height"]))
        if not (0 <= x < x + width <= rgb.shape[1] and 0 <= y < y + height <= rgb.shape[0]):
            raise FrameContractError(f"mirror crop is out of bounds for {row['id']}")
        crop_rgb = np.ascontiguousarray(rgb[y : y + height, x : x + width])
        results.append(
            {
                "asset_id": row["id"],
                "source_sha256": row["sha256"],
                "frame_index": frame,
                "frame_count": int(media["frames"]),
                "decoded_rgb_shape": list(rgb.shape),
                "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "crop_rgb_pixels": dict(crop),
                "crop_rgb_shape": list(crop_rgb.shape),
                "crop_rgb_sha256": hashlib.sha256(crop_rgb.tobytes()).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "status": "decoded_witnesses_pending_human_orientation_review",
        "ffmpeg": {"executable": ffmpeg, "version_first_line": version},
        "review_instruction": plan["mirror_contract"]["review_instruction"],
        "assets": results,
        "witness_semantic_sha256": canonical_sha256(results),
    }


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise FrameContractError(f"root quaternion must be finite xyzw, got {q}")
    norm = float(np.linalg.norm(q))
    if not math.isclose(norm, 1.0, abs_tol=1e-5, rel_tol=0.0):
        raise FrameContractError(f"root quaternion norm {norm} is not one")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def derive_transform(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    *,
    proper_tolerance: float,
    heading_tolerance: float,
    root_xy_tolerance: float,
) -> dict[str, Any]:
    pos = np.asarray(root_pos, dtype=np.float64)
    rot = np.asarray(root_rot_xyzw, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or rot.shape != (pos.shape[0], 4) or len(pos) < 2:
        raise FrameContractError(f"invalid root pose shapes: {pos.shape}, {rot.shape}")
    if not np.isfinite(pos).all() or not np.isfinite(rot).all():
        raise FrameContractError("root pose contains NaN/Inf")
    root_rotation = _quat_xyzw_to_matrix(rot[0])
    forward = root_rotation[:, 0]
    horizontal = forward[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 0.5:
        raise FrameContractError(f"frame-0 pelvis forward is nearly vertical: {forward.tolist()}")
    yaw = float(math.atan2(horizontal[1], horizontal[0]))
    c, s = math.cos(-yaw), math.sin(-yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    translation = np.array(
        [
            -(rotation[0, 0] * pos[0, 0] + rotation[0, 1] * pos[0, 1]),
            -(rotation[1, 0] * pos[0, 0] + rotation[1, 1] * pos[0, 1]),
            0.0,
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    mapped_root = rotation @ pos[0] + translation
    mapped_forward = rotation @ forward
    mapped_yaw = float(math.atan2(mapped_forward[1], mapped_forward[0]))
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality > proper_tolerance or abs(determinant - 1.0) > proper_tolerance:
        raise FrameContractError("derived transform is not proper rigid")
    if abs(mapped_yaw) > heading_tolerance:
        raise FrameContractError(f"mapped heading yaw {mapped_yaw} exceeds tolerance")
    if float(np.linalg.norm(mapped_root[:2])) > root_xy_tolerance:
        raise FrameContractError(f"mapped root XY {mapped_root[:2]} exceeds tolerance")
    if not np.allclose(rotation @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=proper_tolerance, rtol=0):
        raise FrameContractError("derived transform does not preserve ground +Z")
    return {
        "matrix_4x4": np.round(matrix, 15).tolist(),
        "source_frame0_root_pos_m": pos[0].tolist(),
        "source_frame0_pelvis_forward": forward.tolist(),
        "source_frame0_heading_yaw_rad": yaw,
        "source_frame0_heading_yaw_deg": math.degrees(yaw),
        "mapped_frame0_root_pos_m": mapped_root.tolist(),
        "mapped_frame0_pelvis_forward": mapped_forward.tolist(),
        "mapped_frame0_heading_yaw_rad": mapped_yaw,
        "rotation_determinant": determinant,
        "rotation_orthogonality_max_error": orthogonality,
        "ground_z_preserved": True,
    }


def _load_gmr(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise FrameContractError(f"cannot load grounded GMR {path}: {exc}") from None
    if not isinstance(value, dict):
        raise FrameContractError(f"grounded GMR root must be a mapping: {path}")
    for key in ("fps", "root_pos", "root_rot", "dof_pos"):
        if key not in value:
            raise FrameContractError(f"grounded GMR lacks {key}: {path}")
    return value


def _validate_review(
    review_path: Path, expected_review_sha256: str, witnesses: dict[str, Any], expected_ids: list[str]
) -> dict[str, Any]:
    _require_sha(expected_review_sha256, "--expected-review-sha256")
    actual_sha = sha256_file(review_path)
    if actual_sha != expected_review_sha256:
        raise FrameContractError(f"mirror review sha256 {actual_sha} != {expected_review_sha256}")
    review = _read_json(review_path, "mirror review")
    if review.get("schema_version") != 1 or review.get("status") != "verified_not_mirrored":
        raise FrameContractError("mirror review must explicitly verify not-mirrored status")
    if review.get("witness_semantic_sha256") != witnesses["witness_semantic_sha256"]:
        raise FrameContractError("mirror review is not bound to this decoded witness set")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise FrameContractError("mirror review requires a named reviewer")
    rows = review.get("assets")
    if not isinstance(rows, list) or [row.get("asset_id") for row in rows] != expected_ids:
        raise FrameContractError("mirror review must list every expected asset in order")
    witnessed = {row["asset_id"]: row for row in witnesses["assets"]}
    for row in rows:
        asset_id = row["asset_id"]
        if row.get("crop_rgb_sha256") != witnessed[asset_id]["crop_rgb_sha256"]:
            raise FrameContractError(f"mirror review crop SHA mismatch for {asset_id}")
        if row.get("frame_index") != witnessed[asset_id]["frame_index"]:
            raise FrameContractError(f"mirror review frame mismatch for {asset_id}")
        if row.get("assertion") != "upright_unreflected_chinese_glyphs":
            raise FrameContractError(f"mirror review assertion missing for {asset_id}")
        if row.get("side_swap_required") is not False:
            raise FrameContractError(f"mirror review must not request a side swap for {asset_id}")
    if review.get("private_pixels_committed") is not False:
        raise FrameContractError("mirror witness pixels must remain private")
    return review


def run_audit(
    plan: dict[str, Any], review_path: Path, expected_review_sha256: str
) -> dict[str, Any]:
    source_semantics = _validate_source_semantics(plan)
    witnesses = collect_witnesses(plan)
    review = _validate_review(review_path, expected_review_sha256, witnesses, plan["expected_asset_ids"])
    _, ground_by_id = _asset_rows(plan)
    transform_contract = plan["transform_contract"]
    threshold = float(plan["mirror_contract"]["minimum_right_left_arm_motion_energy_ratio"])
    assets: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    for asset_id in plan["expected_asset_ids"]:
        ground_row = ground_by_id[asset_id]
        path = _verify_gmr(plan, ground_row)
        payload = _load_gmr(path)
        dof = np.asarray(payload["dof_pos"], dtype=np.float64)
        if dof.ndim != 2 or dof.shape[1] != 31 or len(dof) < 2 or not np.isfinite(dof).all():
            raise FrameContractError(f"{asset_id}: dof_pos must be finite (T,31)")
        left_energy = float(np.sum(np.diff(dof[:, LEFT_ARM], axis=0) ** 2))
        right_energy = float(np.sum(np.diff(dof[:, RIGHT_ARM], axis=0) ** 2))
        ratio = right_energy / max(left_energy, 1e-15)
        if ratio < threshold:
            raise FrameContractError(
                f"{asset_id}: right/left arm energy ratio {ratio:.6g} < {threshold}; mirror fails closed"
            )
        transform = derive_transform(
            np.asarray(payload["root_pos"]),
            np.asarray(payload["root_rot"]),
            proper_tolerance=float(transform_contract["proper_rotation_tolerance"]),
            heading_tolerance=float(transform_contract["heading_tolerance_rad"]),
            root_xy_tolerance=float(transform_contract["mapped_root_xy_tolerance_m"]),
        )
        witness = next(row for row in witnesses["assets"] if row["asset_id"] == asset_id)
        row = {
            "asset_id": asset_id,
            "grounded_gmr": {
                "bytes": ground_row["output"]["bytes"],
                "sha256": ground_row["output"]["sha256"],
            },
            "frames": int(np.asarray(payload["root_pos"]).shape[0]),
            "fps": float(np.asarray(payload["fps"]).reshape(-1)[0]),
            "mirror_witness": witness,
            "right_arm_motion_energy": right_energy,
            "left_arm_motion_energy": left_energy,
            "right_left_arm_motion_energy_ratio": ratio,
            "mirror_status": "verified_not_mirrored",
            "side_swap_required": False,
            "transform": transform,
        }
        assets.append(row)
        matrices.append({"asset_id": asset_id, "matrix_4x4": transform["matrix_4x4"]})
    return {
        "schema_version": 1,
        "status": "complete_verified_per_asset_hope_frame_and_not_mirrored",
        "scope": (
            "CPU-only per-asset proper-rigid GMR-world to canonical HOPE +X/virtual-table "
            "normalization plus content-addressed final-video mirror review; air-swing contact truth null"
        ),
        "cpu_only": True,
        "CUDA_VISIBLE_DEVICES": "",
        "real_robot_commands_authorized": False,
        "contact_phase_truth": None,
        "tool_contract": plan["tool_contract"],
        "source_bindings": plan["source_bindings"],
        "committed_task_source": plan["committed_task_source"],
        "source_semantics": source_semantics,
        "frame_contract": {
            "status": "verified",
            "transform_scope": "per_asset",
            "method": transform_contract["method"],
            "source_frame": "grounded canonical GMR/MuJoCo world, floor z=0",
            "target_frame": "HOPE env-local robot origin, +X forward, +Y left, +Z up",
            "gmr_world_to_hope_table_transform_verified": True,
            "per_asset_transform_semantic_sha256": canonical_sha256(matrices),
            "capture_table_pose_observed": False,
            "target_table_pose": source_semantics["target_table"],
            "target_table_pose_semantics": source_semantics["table_pose_semantics"],
        },
        "mirror_contract": {
            "status": "verified_not_mirrored",
            "review_path": str(review_path.resolve()),
            "review_bytes": review_path.stat().st_size,
            "review_sha256": expected_review_sha256,
            "review_semantics": review.get("review_semantics"),
            "witness_semantic_sha256": witnesses["witness_semantic_sha256"],
            "minimum_required_right_left_arm_motion_energy_ratio": threshold,
            "side_swap_required": False,
        },
        "assets": assets,
        "eligibility": {
            "immutable_64_question_returnability_phase_screen": True,
            "canonical_counterfactual_hope_table_coverage": "eligible_pending_bound_screen_runtime",
            "real_capture_returnability": None,
            "schema2_conversion": False,
            "topp": False,
            "rl_training": False,
            "real_robot": False,
        },
        "returnability_semantics": {
            "canonical_counterfactual_hope_table": (
                "eligible for the already-frozen diagnostic question paper after applying only the "
                "pre-result frame-0 pelvis transform"
            ),
            "real_capture_table": {
                "coverage": None,
                "status": "unsupported_no_table_ball_contact_or_camera_extrinsic_in_air_swing_video",
            },
        },
        "remaining_blockers": [
            "the videos are air swings: observed ball/contact phase remains null",
            "the table pose is the bound canonical counterfactual HOPE table, not a recovered capture extrinsic",
            "schema-2 conversion and L0/L1 audits have not run for these motions",
            "table/net swept-volume, dynamics and balance gates remain open",
            "TOPP and every post-retime safety/dynamics audit remain open",
        ],
    }


def _atomic_write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FrameContractError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FrameContractError(f"refusing concurrent overwrite: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.set_defaults(command="validate")
    witnesses = sub.add_parser("witnesses")
    witnesses.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mirror-review", required=True)
    run.add_argument("--expected-review-sha256", required=True)
    run.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        manifest_path = Path(args.manifest).expanduser().resolve()
        plan = _load_manifest(manifest_path, args.expected_manifest_sha256)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if args.command == "validate":
            _validate_source_semantics(plan)
            _asset_rows(plan)
            print(json.dumps({"status": "valid", "manifest_sha256": args.expected_manifest_sha256}))
            return 0
        if args.command == "witnesses":
            result = collect_witnesses(plan)
            output = Path(args.output).expanduser().resolve()
        else:
            result = run_audit(
                plan,
                Path(args.mirror_review).expanduser().resolve(),
                args.expected_review_sha256,
            )
            output = Path(args.output or plan["output_contract"]["result"]).expanduser().resolve()
        _atomic_write_new(output, result)
        print(json.dumps({"status": result["status"], "output": str(output), "sha256": sha256_file(output)}))
        return 0
    except (FrameContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
