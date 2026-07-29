#!/usr/bin/env python3
"""Materialize one exact, contact-only N=1 ActionBall training bundle.

This is the deliberately short path for the first ``bh_loop_c`` and
``bh_block`` experiments.  It keeps the reviewed incoming-ball profile from
the pinned four-action fivebind manifest, changes only its historically
absolute contact Z into a fully base-relative Z, and binds that row to:

* the exact tracked upper-body motion bytes;
* a freshly generated schema-v2 selected-face-centre prototype;
* freshly generated solver/physics profile pins from the final code tree; and
* the canonical N=1 counter-rally objective used for RL shaping.

The admission receipt proves only that the manifest hit time resolves to the
same interior motion frame and that the manifest task centre is within 3 cm
of the selected rubber face centre at that frame.  It explicitly makes no
landing, post-bounce, opponent-baseline, deployment, or hardware claim.

All output filenames are content addressed.  Every output is preflighted and
opened with exclusive creation; reruns never overwrite an existing byte.
The module depends only on the Python standard library plus NumPy and never
imports Torch, Isaac, or MuJoCo.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parents[2]
MDP_RELATIVE_DIR = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
SOURCE_MANIFEST_RELATIVE_PATH = Path(
    "configs/action_ball_n5_nomove_f10_20260728.json"
)
SOURCE_MANIFEST_SHA256 = (
    "0b640f5d7b2d35895d1a4696635e3a256f2a32341778ad586d828d677317e2b7"
)
SCOPE = "upper"
HOLDOUT_SAMPLES_PER_ACTION = 768
CENTER_ALIGNMENT_THRESHOLD_M = 0.03
TIMING_ABS_TOLERANCE_S = 1.0e-12
REFERENCE_SPEED_ABS_TOLERANCE_MPS = 1.0e-6
ROOT_STATIONARY_TOLERANCE_M = 1.0e-6
ROOT_YAW_TOLERANCE_RAD = 1.0e-6
WINDOW_HALF_FRAMES = 2
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

RACKET_WRIST_BODY = "right_wrist_yaw_Link"
ROOT_BODY = "pelvis_link"

SUPPORTED_ACTIONS = {
    "bh_loop_c": {
        "action_uid": 1722317591841513,
        "family": "backhand",
        "motion_path": (
            "motions/fivebind_n5_20260728/"
            "bh_loop_c_upper_fivebind.npz"
        ),
        "motion_sha256": (
            "c950a73e473cad84d0fafcd51c552ec4fef085580bbeaec0f4e96be2acd7e2fc"
        ),
        "reference_t_hit_s": 0.62,
        "reference_t_cycle_s": 1.4,
        "priority": 0,
    },
    "bh_block": {
        "action_uid": 1115176677418582,
        "family": "backhand",
        "motion_path": (
            "motions/fivebind_n5_20260728/"
            "bh_block_upper_fivebind.npz"
        ),
        "motion_sha256": (
            "0cd94aa47bf8feb59bbe7cc7a0306abb57ee7ec8ebcec6443a80bbdc58894309"
        ),
        "reference_t_hit_s": 0.48,
        "reference_t_cycle_s": 1.06,
        "priority": 2,
    },
}

CLAIMS = {
    "selector_executed": False,
    "action_identity_frozen_before_ball_sampling": True,
    "contact_alignment_claim": True,
    "landing_claim": False,
    "post_bounce_claim": False,
    "baseline_crossing_claim": False,
    "deployment_claim": False,
}


class N1ContactBundleError(ValueError):
    """Fail-closed configuration, identity, or admission error."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise N1ContactBundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise N1ContactBundleError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise N1ContactBundleError(f"cannot read {label}: {path}") from error
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise N1ContactBundleError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise N1ContactBundleError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise N1ContactBundleError(f"{label} must be one JSON object")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise N1ContactBundleError(f"{label} must be lowercase SHA-256")
    return value


def _require_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise N1ContactBundleError(f"{label} must be one plain number")
    result = float(value)
    if not math.isfinite(result):
        raise N1ContactBundleError(f"{label} must be finite")
    return result


def _repo_relative(path: Path, root: Path, *, label: str) -> str:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise N1ContactBundleError(
            f"{label} must remain inside repo root"
        ) from error
    value = relative.as_posix()
    if not value or PurePosixPath(value).is_absolute() or ".." in relative.parts:
        raise N1ContactBundleError(f"{label} is not repo-relative")
    return value


def _resolve_repo_file(
    root: Path,
    relative: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if type(relative) is not str:
        raise N1ContactBundleError(f"{label} path must be text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise N1ContactBundleError(f"{label} path must be repo-relative POSIX")
    candidate = root.joinpath(*pure.parts)
    try:
        metadata = candidate.stat()
    except OSError as error:
        raise N1ContactBundleError(f"{label} is missing: {relative}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise N1ContactBundleError(f"{label} is not a regular file")
    return candidate.resolve(strict=True), pure.as_posix()


def _load_module(name: str, path: Path) -> Any:
    unique = f"{name}_{_sha256_bytes(str(path.resolve()).encode())[:16]}"
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise N1ContactBundleError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(unique, None)
        raise
    return module


def _require_git_tracked(root: Path, relative: str) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ],
        capture_output=True,
        text=True,
    )
    tracked = tuple(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    if result.returncode != 0 or tracked != (relative,):
        raise N1ContactBundleError(
            f"motion must be one exact Git-tracked path: {relative}"
        )


def _quat_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape[-1] != 4 or not np.all(np.isfinite(q)):
        raise N1ContactBundleError("motion contains an invalid quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise N1ContactBundleError("motion contains a zero quaternion")
    q = q / norm
    w, x, y, z = (q[..., index] for index in range(4))
    result = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - w * z)
    result[..., 0, 2] = 2.0 * (x * z + w * y)
    result[..., 1, 0] = 2.0 * (x * y + w * z)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - w * x)
    result[..., 2, 0] = 2.0 * (x * z - w * y)
    result[..., 2, 1] = 2.0 * (y * z + w * x)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _yaw(quaternion_wxyz: np.ndarray) -> float:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return float(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _to_ready_b_yaw(
    points_w: np.ndarray,
    *,
    ready_root_w: np.ndarray,
    ready_yaw_rad: float,
) -> np.ndarray:
    points = np.asarray(points_w, dtype=np.float64)
    delta = points - np.asarray(ready_root_w, dtype=np.float64)
    cosine = math.cos(ready_yaw_rad)
    sine = math.sin(ready_yaw_rad)
    return np.stack(
        (
            cosine * delta[..., 0] + sine * delta[..., 1],
            -sine * delta[..., 0] + cosine * delta[..., 1],
            delta[..., 2],
        ),
        axis=-1,
    )


def _vector_to_ready_b_yaw(
    vectors_w: np.ndarray,
    *,
    ready_yaw_rad: float,
) -> np.ndarray:
    vectors = np.asarray(vectors_w, dtype=np.float64)
    cosine = math.cos(ready_yaw_rad)
    sine = math.sin(ready_yaw_rad)
    return np.stack(
        (
            cosine * vectors[..., 0] + sine * vectors[..., 1],
            -sine * vectors[..., 0] + cosine * vectors[..., 1],
            vectors[..., 2],
        ),
        axis=-1,
    )


def _from_ready_b_yaw(
    point_b: Sequence[float],
    *,
    ready_root_w: np.ndarray,
    ready_yaw_rad: float,
) -> np.ndarray:
    value = np.asarray(point_b, dtype=np.float64)
    cosine = math.cos(ready_yaw_rad)
    sine = math.sin(ready_yaw_rad)
    return np.asarray(
        (
            ready_root_w[0] + cosine * value[0] - sine * value[1],
            ready_root_w[1] + sine * value[0] + cosine * value[1],
            ready_root_w[2] + value[2],
        ),
        dtype=np.float64,
    )


def _runtime_site_velocity(points_w: np.ndarray, fps: float) -> np.ndarray:
    points = np.asarray(points_w, dtype=np.float64)
    count = points.shape[0]
    index = np.arange(count)
    upper = np.clip(index + WINDOW_HALF_FRAMES, 0, count - 1)
    lower = np.clip(index - WINDOW_HALF_FRAMES, 0, count - 1)
    return (
        (points[upper] - points[lower])
        / (2.0 * WINDOW_HALF_FRAMES / fps)
    )


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    cosine = float(
        np.dot(lhs, rhs)
        / max(np.linalg.norm(lhs) * np.linalg.norm(rhs), 1.0e-15)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _verify_profile_pins(
    *,
    repo_root: Path,
    path: Path,
    expected_sha256: str,
    geometry: Any,
    objective_sha256: str,
) -> dict[str, object]:
    expected = _require_sha256(
        expected_sha256, label="expected profile-pins SHA-256"
    )
    actual = _sha256_file(path)
    if actual != expected:
        raise N1ContactBundleError(
            f"profile-pins bytes changed: expected {expected}, got {actual}"
        )
    document = _read_json(path, label="profile pins")
    solver_payload = document.get("solver_payload")
    physics_payload = document.get("physics_payload")
    if type(solver_payload) is not dict or type(physics_payload) is not dict:
        raise N1ContactBundleError(
            "profile pins must contain solver_payload and physics_payload"
        )
    solver_sha = _require_sha256(
        document.get("solver_profile_sha256"),
        label="solver_profile_sha256",
    )
    physics_sha = _require_sha256(
        document.get("physics_profile_sha256"),
        label="physics_profile_sha256",
    )
    if _canonical_sha256(solver_payload) != solver_sha:
        raise N1ContactBundleError(
            "solver_profile_sha256 does not seal solver_payload"
        )
    if _canonical_sha256(physics_payload) != physics_sha:
        raise N1ContactBundleError(
            "physics_profile_sha256 does not seal physics_payload"
        )
    if solver_payload.get("physics_profile_sha256") != physics_sha:
        raise N1ContactBundleError(
            "solver payload does not bind the pinned physics profile"
        )
    contact_geometry = document.get("contact_geometry")
    expected_geometry = {
        "payload": geometry.GEOMETRY_SOURCE_PAYLOAD,
        "sha256": geometry.GEOMETRY_SOURCE_SHA256,
    }
    if contact_geometry != expected_geometry:
        raise N1ContactBundleError(
            "profile pins do not bind the live exact-face geometry payload"
        )
    if solver_payload.get("contact_geometry") != expected_geometry:
        raise N1ContactBundleError(
            "solver payload does not bind the live exact-face geometry"
        )
    counter = solver_payload.get("counter_rally")
    if (
        type(counter) is not dict
        or counter.get("mode") != "exact_n1_fixed_action_reverse_ray"
        or counter.get("objective_profile_sha256") != objective_sha256
        or counter.get("precheck_before_ordinary_solver") is not True
        or counter.get("selector_or_action_switching") is not False
        or SHA256_PATTERN.fullmatch(
            str(counter.get("venue_physics_sha256", ""))
        )
        is None
    ):
        raise N1ContactBundleError(
            "solver profile is not the exact canonical N=1 counter-rally "
            "solver profile"
        )
    source_hashes = document.get("solver_implementation_source_sha256")
    if type(source_hashes) is not dict:
        raise N1ContactBundleError(
            "profile pins lack implementation source hashes"
        )
    required_sources = {
        "hope_commands.py",
        "continuous_questions.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
        "counter_rally.py",
        "counter_rally_torch.py",
    }
    if set(source_hashes) != required_sources:
        raise N1ContactBundleError(
            "counter-rally profile pins must bind the exact seven "
            "implementation sources"
        )
    if solver_payload.get("implementation_source_sha256") != source_hashes:
        raise N1ContactBundleError(
            "solver payload implementation hashes differ from profile pins"
        )
    for filename, declared in source_hashes.items():
        if (
            type(filename) is not str
            or "/" in filename
            or "\\" in filename
        ):
            raise N1ContactBundleError(
                "profile-pins implementation filename is invalid"
            )
        source, _ = _resolve_repo_file(
            repo_root,
            (MDP_RELATIVE_DIR / filename).as_posix(),
            label=f"profile source {filename}",
        )
        if _sha256_file(source) != _require_sha256(
            declared, label=f"profile source {filename} SHA-256"
        ):
            raise N1ContactBundleError(
                f"profile pins are stale for implementation {filename}"
            )
    venue = physics_payload.get("venue_source")
    if type(venue) is not dict or set(venue) != {"path", "file_sha256"}:
        raise N1ContactBundleError(
            "physics profile venue_source keys mismatch"
        )
    venue_path, _ = _resolve_repo_file(
        repo_root, venue["path"], label="venue physics source"
    )
    if _sha256_file(venue_path) != _require_sha256(
        venue["file_sha256"], label="venue physics SHA-256"
    ):
        raise N1ContactBundleError(
            "physics profile venue bytes changed"
        )
    return {
        "path": _repo_relative(path, repo_root, label="profile pins"),
        "sha256": actual,
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "geometry_payload_sha256": geometry.GEOMETRY_SOURCE_SHA256,
    }


def _motion_state(
    *,
    motion_path: Path,
    action: Mapping[str, Any],
    geometry: Any,
) -> dict[str, Any]:
    try:
        with np.load(motion_path, allow_pickle=False) as archive:
            required = {
                "fps",
                "body_names",
                "body_pos_w",
                "body_quat_w",
                "body_ang_vel_w",
                "joint_vel",
                "kinematics_schema_version",
                "body_pos_point",
                "body_lin_vel_point",
            }
            missing = required.difference(archive.files)
            if missing:
                raise N1ContactBundleError(
                    f"motion lacks fields {sorted(missing)}"
                )
            fps_values = np.asarray(archive["fps"]).reshape(-1)
            if fps_values.size != 1:
                raise N1ContactBundleError("motion fps must be scalar")
            fps = float(fps_values[0])
            body_names = tuple(str(value) for value in archive["body_names"])
            positions = np.asarray(archive["body_pos_w"], dtype=np.float64)
            quaternions = np.asarray(
                archive["body_quat_w"], dtype=np.float64
            )
            angular_velocity = np.asarray(
                archive["body_ang_vel_w"], dtype=np.float64
            )
            joint_velocity = np.asarray(
                archive["joint_vel"], dtype=np.float64
            )
            schema = np.asarray(
                archive["kinematics_schema_version"]
            ).reshape(-1)
            position_point = str(
                np.asarray(archive["body_pos_point"]).reshape(-1)[0]
            )
            velocity_point = str(
                np.asarray(archive["body_lin_vel_point"]).reshape(-1)[0]
            )
    except (OSError, ValueError) as error:
        if isinstance(error, N1ContactBundleError):
            raise
        raise N1ContactBundleError(
            f"cannot load exact motion {motion_path}"
        ) from error
    if (
        not math.isfinite(fps)
        or abs(fps - 50.0) > 1.0e-12
        or schema.size != 1
        or int(schema[0]) != 2
        or position_point != "link_origin"
        or velocity_point != "center_of_mass"
    ):
        raise N1ContactBundleError(
            "motion kinematics schema/fps/point semantics mismatch"
        )
    if (
        positions.ndim != 3
        or positions.shape[-1] != 3
        or quaternions.shape != positions.shape[:-1] + (4,)
        or angular_velocity.shape != positions.shape
        or joint_velocity.ndim != 2
        or joint_velocity.shape[0] != positions.shape[0]
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(angular_velocity))
        or not np.all(np.isfinite(joint_velocity))
    ):
        raise N1ContactBundleError("motion array shapes or finiteness mismatch")
    if (
        body_names.count(RACKET_WRIST_BODY) != 1
        or body_names.count(ROOT_BODY) != 1
        or len(body_names) != positions.shape[1]
    ):
        raise N1ContactBundleError(
            "motion body-name identity/order is invalid"
        )
    wrist_index = body_names.index(RACKET_WRIST_BODY)
    root_index = body_names.index(ROOT_BODY)
    frame_count = positions.shape[0]
    strike_phase = _require_finite(
        action["strike_phase"], label="strike_phase"
    )
    contact_frame = round(strike_phase * (frame_count - 1))
    if not WINDOW_HALF_FRAMES <= contact_frame < (
        frame_count - WINDOW_HALF_FRAMES
    ):
        raise N1ContactBundleError(
            "contact frame is not interior to the exact alignment window"
        )
    motion_t_hit = contact_frame / fps
    motion_t_cycle = (frame_count - 1) / fps
    manifest_t_hit = _require_finite(
        action["reference_t_hit_s"], label="reference_t_hit_s"
    )
    manifest_t_cycle = _require_finite(
        action["reference_t_cycle_s"], label="reference_t_cycle_s"
    )
    hit_error = abs(manifest_t_hit - motion_t_hit)
    cycle_error = abs(manifest_t_cycle - motion_t_cycle)
    if (
        hit_error > TIMING_ABS_TOLERANCE_S
        or cycle_error > TIMING_ABS_TOLERANCE_S
    ):
        raise N1ContactBundleError(
            "manifest t_hit/t_cycle disagree with exact motion frames"
        )
    root_position = positions[:, root_index]
    ready_root = root_position[0]
    if (
        np.linalg.norm(root_position[contact_frame] - ready_root)
        > ROOT_STATIONARY_TOLERANCE_M
    ):
        raise N1ContactBundleError(
            "no_move teacher root translates before contact"
        )
    ready_yaw = _yaw(quaternions[0, root_index])
    contact_yaw = _yaw(quaternions[contact_frame, root_index])
    yaw_error = math.atan2(
        math.sin(contact_yaw - ready_yaw),
        math.cos(contact_yaw - ready_yaw),
    )
    if abs(yaw_error) > ROOT_YAW_TOLERANCE_RAD:
        raise N1ContactBundleError(
            "no_move teacher root yaw changes before contact"
        )
    rotations = _quat_to_rotation(quaternions[:, wrist_index])
    site_offset = np.asarray(
        geometry.RACKET_SITE_OFFSET_WRIST_M, dtype=np.float64
    )
    site_w = positions[:, wrist_index] + np.einsum(
        "tij,j->ti", rotations, site_offset
    )
    site_velocity_w = _runtime_site_velocity(site_w, fps)
    reference_site_speed = float(
        np.linalg.norm(site_velocity_w[contact_frame])
    )
    declared_site_speed = _require_finite(
        action["reference_racket_site_speed_mps"],
        label="reference_racket_site_speed_mps",
    )
    if (
        abs(reference_site_speed - declared_site_speed)
        > REFERENCE_SPEED_ABS_TOLERANCE_MPS
    ):
        raise N1ContactBundleError(
            "manifest reference racket-site speed differs from exact motion"
        )
    face_sign = int(action["mount_normal_sign"])
    face_offset_local = np.asarray(
        geometry.face_center_from_site_local(face_sign),
        dtype=np.float64,
    )
    face_offset_w = np.einsum(
        "tij,j->ti", rotations, face_offset_local
    )
    face_center_w = site_w + face_offset_w
    face_velocity_w = site_velocity_w + np.cross(
        angular_velocity[:, wrist_index], face_offset_w
    )
    face_velocity_b = _vector_to_ready_b_yaw(
        face_velocity_w, ready_yaw_rad=ready_yaw
    )
    face_speed = float(np.linalg.norm(face_velocity_b[contact_frame]))
    if face_speed <= 1.0e-9:
        raise N1ContactBundleError(
            "selected face centre is stationary at contact"
        )
    face_velocity_hat_b = face_velocity_b[contact_frame] / face_speed
    physical_normal_w = (
        float(face_sign) * rotations[contact_frame, :, 1]
    )
    physical_normal_b = _vector_to_ready_b_yaw(
        physical_normal_w, ready_yaw_rad=ready_yaw
    )
    physical_normal_b = physical_normal_b / np.linalg.norm(
        physical_normal_b
    )
    site_b = _to_ready_b_yaw(
        site_w, ready_root_w=ready_root, ready_yaw_rad=ready_yaw
    )
    face_center_b = _to_ready_b_yaw(
        face_center_w,
        ready_root_w=ready_root,
        ready_yaw_rad=ready_yaw,
    )
    lower = contact_frame - WINDOW_HALF_FRAMES
    upper = contact_frame + WINDOW_HALF_FRAMES
    window = slice(lower, upper + 1)
    direction_cone = max(
        _angle_deg(
            face_velocity_b[index],
            face_velocity_b[contact_frame],
        )
        for index in range(lower, upper + 1)
    )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "contact_frame": contact_frame,
        "motion_t_hit_s": motion_t_hit,
        "motion_t_cycle_s": motion_t_cycle,
        "t_hit_abs_error_s": hit_error,
        "t_cycle_abs_error_s": cycle_error,
        "ready_root_w_m": ready_root,
        "ready_yaw_rad": ready_yaw,
        "site_w_m": site_w[contact_frame],
        "site_b_yaw_m": site_b[contact_frame],
        "face_center_w_m": face_center_w[contact_frame],
        "face_center_b_yaw_m": face_center_b[contact_frame],
        "reference_site_speed_mps": reference_site_speed,
        "face_velocity_hat_b": face_velocity_hat_b,
        "face_speed_mps": face_speed,
        "face_direction_cone_deg": direction_cone,
        "physical_normal_b": physical_normal_b,
        "site_band_b_x": [
            float(np.min(site_b[window, 0])),
            float(np.max(site_b[window, 0])),
        ],
        "site_band_b_y": [
            float(np.min(site_b[window, 1])),
            float(np.max(site_b[window, 1])),
        ],
        "site_band_z_w": [
            float(np.min(site_w[window, 2])),
            float(np.max(site_w[window, 2])),
        ],
        "contact_window_frames": [lower, upper],
    }


def _selected_source_action(
    source: Mapping[str, Any], action_id: str
) -> dict[str, Any]:
    if source.get("schema_version") != 3:
        raise N1ContactBundleError("source manifest must use schema v3")
    if source.get("mobility_mode") != "no_move":
        raise N1ContactBundleError("source manifest must be no_move")
    actions = source.get("actions")
    if type(actions) is not list:
        raise N1ContactBundleError("source manifest actions must be a list")
    matches = [
        row
        for row in actions
        if type(row) is dict and row.get("action_id") == action_id
    ]
    if len(matches) != 1:
        raise N1ContactBundleError(
            f"source manifest must contain exactly one {action_id!r}"
        )
    action = deepcopy(matches[0])
    facts = SUPPORTED_ACTIONS[action_id]
    exact = {
        "action_uid": facts["action_uid"],
        "family": facts["family"],
        "motion_path": facts["motion_path"],
        "motion_sha256": facts["motion_sha256"],
        "reference_t_hit_s": facts["reference_t_hit_s"],
        "reference_t_cycle_s": facts["reference_t_cycle_s"],
    }
    for key, expected in exact.items():
        if action.get(key) != expected:
            raise N1ContactBundleError(
                f"source {action_id} {key} changed from pinned fact"
            )
    return action


def _correct_contact_z(
    action: dict[str, Any],
    *,
    ready_root_z_w_m: float,
) -> dict[str, Any]:
    corrected = deepcopy(action)
    profile = corrected.get("ball_profile")
    if type(profile) is not dict:
        raise N1ContactBundleError("source action lacks ball_profile")
    for key in (
        "contact_offset_center_b_yaw_m",
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
    ):
        vector = profile.get(key)
        if (
            type(vector) is not list
            or len(vector) != 3
            or any(
                not math.isfinite(float(value))
                for value in vector
                if type(value) in (int, float) and not isinstance(value, bool)
            )
            or any(
                isinstance(value, bool) or type(value) not in (int, float)
                for value in vector
            )
        ):
            raise N1ContactBundleError(
                f"ball_profile.{key} must contain three finite numbers"
            )
        vector[2] = float(vector[2]) - ready_root_z_w_m
    return corrected


def _prototype_document(
    *,
    action_id: str,
    action: Mapping[str, Any],
    state: Mapping[str, Any],
    source_manifest_pin: Mapping[str, str],
    profile_pin: Mapping[str, object],
    motion_pin: Mapping[str, str],
    geometry: Any,
    geometry_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    face_speed = float(state["face_speed_mps"])
    teacher_rate_min = float(action["teacher_rate_min"])
    teacher_rate_max = float(action["teacher_rate_max"])
    velocity_hat = np.asarray(state["face_velocity_hat_b"])
    normal = np.asarray(state["physical_normal_b"])
    row = {
        "motion_id": action_id,
        "scope": SCOPE,
        "family": action["family"],
        "clip_index": 0,
        "npz_sha256": motion_pin["sha256"],
        "frames": int(state["frame_count"]),
        "t_prepare_s": float(action["reference_t_hit_s"]),
        "t_prepare_min_s": (
            float(action["reference_t_hit_s"]) / teacher_rate_max
        ),
        "t_prepare_max_s": (
            float(action["reference_t_hit_s"]) / teacher_rate_min
        ),
        "band_b_x": list(state["site_band_b_x"]),
        "band_b_y": list(state["site_band_b_y"]),
        "band_z_w": list(state["site_band_z_w"]),
        "slack_b_xy_m": 0.15,
        "slack_z_w_m": 0.10,
        "p_contact_b": [
            float(value) for value in state["site_b_yaw_m"]
        ],
        "n_hat_b": [float(value) for value in normal],
        "face_sign": float(action["mount_normal_sign"]),
        "priority": int(SUPPORTED_ACTIONS[action_id]["priority"]),
        "enabled": True,
        # Prototype schema v2 resolves the exact integer frame, whereas the
        # preserved source manifest stores a rounded decimal that merely
        # resolves to that same frame.
        "strike_phase": (
            int(state["contact_frame"])
            / (int(state["frame_count"]) - 1)
        ),
        "contact_frame": int(state["contact_frame"]),
        "contact_window_frames": list(state["contact_window_frames"]),
        "racket_face_center_velocity_hat_b": [
            float(value) for value in velocity_hat
        ],
        "racket_face_center_elevation_deg": math.degrees(
            math.asin(max(-1.0, min(1.0, float(velocity_hat[2]))))
        ),
        "racket_face_center_window_dir_cone_deg": float(
            state["face_direction_cone_deg"]
        ),
        "racket_face_center_speed_nominal_mps": face_speed,
        "racket_face_center_speed_max_mps": (
            face_speed * teacher_rate_max
        ),
        "racket_face_center_speed_min_mps": (
            face_speed * teacher_rate_min
        ),
        "racket_face_center_v_star_cap_mps": (
            face_speed * teacher_rate_max
        ),
        "racket_face_center_v_dir_tol_deg": 10.0,
        "racket_face_center_cos_normal_velocity": float(
            np.dot(normal, velocity_hat)
        ),
    }
    scopes = {SCOPE: [row]}
    producer_sha = _sha256_file(Path(__file__).resolve())
    document = {
        "schema_version": 2,
        "prototype_set_id": (
            f"n1_{action_id}_upper_contact_"
            f"{motion_pin['sha256'][:12]}"
        ),
        "velocity_contract": {
            "direction_and_speed_point": "selected_rubber_face_center",
            "policy_control_point": "official_racket_site",
            "mapping": (
                "v_face_center=v_site+omega_world_cross_"
                "r_face_center_from_site_world"
            ),
            "site_velocity_authority": (
                "centered_position_fd_half_window_2_clamped_per_clip"
            ),
            "angular_velocity_authority": (
                "npz_body_ang_vel_w_at_right_wrist_yaw_Link"
            ),
            "direction_frame_authority": (
                "canonical_ready_root_yaw_at_frame_0"
            ),
            "geometry_source_sha256": geometry.GEOMETRY_SOURCE_SHA256,
        },
        "contact_rule": {
            "name": "pinned_manifest_reference_t_hit",
            "min_site_z_w_m": 0.88,
            "min_blade_vz_mps": -0.30,
            "note": (
                "Contact-only N1 diagnostic: the pinned manifest t_hit "
                "selects the exact motion frame; no return-flight claim."
            ),
        },
        "provenance": {
            "producer": Path(__file__).name,
            "producer_source_sha256": producer_sha,
            "source_manifest": dict(source_manifest_pin),
            "profile_pins": dict(profile_pin),
            "motion": dict(motion_pin),
            "geometry_source_path": _repo_relative(
                geometry_path, repo_root, label="geometry source"
            ),
            "geometry_source_file_sha256": _sha256_file(geometry_path),
            "contact_window_semantics": (
                "two_frames_before_through_two_frames_after_pinned_t_hit"
            ),
        },
        "scopes": scopes,
        "derived_sha256": _canonical_sha256(scopes),
    }
    return document


def _contact_receipt(
    *,
    action_id: str,
    action: Mapping[str, Any],
    source_manifest_pin: Mapping[str, str],
    profile_pin: Mapping[str, object],
    motion_pin: Mapping[str, str],
    geometry: Any,
    geometry_path: Path,
    repo_root: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    profile = action["ball_profile"]
    task_center_b = np.asarray(
        profile["contact_offset_center_b_yaw_m"], dtype=np.float64
    )
    task_center_w = _from_ready_b_yaw(
        task_center_b,
        ready_root_w=np.asarray(state["ready_root_w_m"]),
        ready_yaw_rad=float(state["ready_yaw_rad"]),
    )
    site_w = np.asarray(state["site_w_m"])
    face_w = np.asarray(state["face_center_w_m"])
    site_distance = float(np.linalg.norm(task_center_w - site_w))
    face_distance = float(np.linalg.norm(task_center_w - face_w))
    if face_distance > CENTER_ALIGNMENT_THRESHOLD_M:
        raise N1ContactBundleError(
            f"{action_id} task centre is {face_distance:.6f} m from the "
            "selected rubber face centre, above 0.03 m"
        )
    legacy_absolute_z = (
        float(task_center_b[2])
        + float(np.asarray(state["ready_root_w_m"])[2])
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "n1_contact_alignment_receipt_v1",
        "status": "PASS",
        "action_id": action_id,
        "action_uid": int(action["action_uid"]),
        "scope": SCOPE,
        "source_manifest": dict(source_manifest_pin),
        "profile_pins": dict(profile_pin),
        "motion": dict(motion_pin),
        "geometry": {
            "source_path": _repo_relative(
                geometry_path, repo_root, label="geometry source"
            ),
            "source_file_sha256": _sha256_file(geometry_path),
            "payload_sha256": geometry.GEOMETRY_SOURCE_SHA256,
            "kind": geometry.EXACT_FACE_CONTACT_KIND,
        },
        "timing": {
            "fps_hz": float(state["fps"]),
            "frame_count": int(state["frame_count"]),
            "contact_frame": int(state["contact_frame"]),
            "manifest_t_hit_s": float(action["reference_t_hit_s"]),
            "motion_t_hit_s": float(state["motion_t_hit_s"]),
            "manifest_t_cycle_s": float(action["reference_t_cycle_s"]),
            "motion_t_cycle_s": float(state["motion_t_cycle_s"]),
            "t_hit_abs_error_s": float(state["t_hit_abs_error_s"]),
            "t_cycle_abs_error_s": float(state["t_cycle_abs_error_s"]),
        },
        "frames": {
            "task_contact_frame": (
                "B_yaw_relative_to_actual_spawn_goal"
            ),
            "teacher_reference_frame": "B_yaw_at_frame0",
            "world_z_origin": "floor",
        },
        "alignment": {
            "threshold_m": CENTER_ALIGNMENT_THRESHOLD_M,
            "ready_root_z_w_m": float(
                np.asarray(state["ready_root_w_m"])[2]
            ),
            "legacy_absolute_contact_z_w_m": legacy_absolute_z,
            "corrected_contact_offset_z_b_yaw_m": float(
                task_center_b[2]
            ),
            "task_contact_offset_center_b_yaw_m": [
                float(value) for value in task_center_b
            ],
            "teacher_racket_site_b_yaw_m": [
                float(value) for value in state["site_b_yaw_m"]
            ],
            "teacher_selected_face_center_b_yaw_m": [
                float(value) for value in state["face_center_b_yaw_m"]
            ],
            "task_to_teacher_site_distance_m": site_distance,
            "task_to_teacher_face_center_distance_m": face_distance,
            "center_gate_point": "selected_rubber_face_center",
            "center_gate_distance_m": face_distance,
            "center_within_threshold": True,
        },
        "claims": dict(CLAIMS),
    }
    return receipt


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def materialize_n1_contact_bundle(
    *,
    repo_root: Path,
    action_id: str,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    profile_pins: Path,
    expected_profile_pins_sha256: str,
    output_dir: Path,
    require_git_tracked_motion: bool = True,
) -> dict[str, object]:
    """Validate all inputs, then exclusively create one content-addressed bundle."""

    root = Path(repo_root).resolve(strict=True)
    if action_id not in SUPPORTED_ACTIONS:
        raise N1ContactBundleError(
            f"unsupported action_id {action_id!r}"
        )
    if type(require_git_tracked_motion) is not bool:
        raise TypeError("require_git_tracked_motion must be bool")
    source_path = Path(source_manifest).resolve(strict=True)
    source_relative = _repo_relative(
        source_path, root, label="source manifest"
    )
    source_expected = _require_sha256(
        expected_source_manifest_sha256,
        label="expected source manifest SHA-256",
    )
    source_actual = _sha256_file(source_path)
    if source_actual != source_expected:
        raise N1ContactBundleError(
            f"source manifest bytes changed: expected {source_expected}, "
            f"got {source_actual}"
        )
    source_document = _read_json(
        source_path, label="source action manifest"
    )
    source_action = _selected_source_action(source_document, action_id)
    facts = SUPPORTED_ACTIONS[action_id]
    motion_path, motion_relative = _resolve_repo_file(
        root, facts["motion_path"], label=f"{action_id} upper motion"
    )
    motion_sha = _sha256_file(motion_path)
    if (
        motion_sha != facts["motion_sha256"]
        or motion_sha != source_action["motion_sha256"]
    ):
        raise N1ContactBundleError(
            f"{action_id} motion SHA differs from pinned exact bytes"
        )
    if require_git_tracked_motion:
        _require_git_tracked(root, motion_relative)
    geometry_path, _ = _resolve_repo_file(
        root,
        (MDP_RELATIVE_DIR / "racket_contact_geometry.py").as_posix(),
        label="exact-face geometry source",
    )
    geometry = _load_module(
        "n1_contact_geometry", geometry_path
    )
    manifest_module_path, _ = _resolve_repo_file(
        root,
        (MDP_RELATIVE_DIR / "action_ball_manifest.py").as_posix(),
        label="action-ball manifest contract",
    )
    manifest_module = _load_module(
        "n1_action_ball_manifest", manifest_module_path
    )
    profile_type = manifest_module._counter_rally_objective_profile_type()
    objective = profile_type()
    objective_mapping = dict(objective.to_mapping())
    objective_sha = objective.sha256
    profile_path = Path(profile_pins).resolve(strict=True)
    profile_pin = _verify_profile_pins(
        repo_root=root,
        path=profile_path,
        expected_sha256=expected_profile_pins_sha256,
        geometry=geometry,
        objective_sha256=objective_sha,
    )
    state = _motion_state(
        motion_path=motion_path,
        action=source_action,
        geometry=geometry,
    )
    corrected_action = _correct_contact_z(
        source_action,
        ready_root_z_w_m=float(
            np.asarray(state["ready_root_w_m"])[2]
        ),
    )
    # Only the three coordinate Z values may differ from the source row.
    expected_corrected = _correct_contact_z(
        source_action,
        ready_root_z_w_m=float(
            np.asarray(state["ready_root_w_m"])[2]
        ),
    )
    if corrected_action != expected_corrected:
        raise AssertionError("unexpected action-row mutation")
    source_manifest_pin = {
        "path": source_relative,
        "sha256": source_actual,
    }
    motion_pin = {
        "path": motion_relative,
        "sha256": motion_sha,
    }
    prototype = _prototype_document(
        action_id=action_id,
        action=corrected_action,
        state=state,
        source_manifest_pin=source_manifest_pin,
        profile_pin=profile_pin,
        motion_pin=motion_pin,
        geometry=geometry,
        geometry_path=geometry_path,
        repo_root=root,
    )
    prototype_bytes = _json_bytes(prototype)
    prototype_sha = _sha256_bytes(prototype_bytes)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve(strict=False)
    try:
        output_relative_dir = destination.relative_to(root).as_posix()
    except ValueError as error:
        raise N1ContactBundleError(
            "output_dir must remain inside repo root"
        ) from error
    if not output_relative_dir:
        raise N1ContactBundleError(
            "output_dir may not be the repository root"
        )
    prototype_name = (
        f"{action_id}.upper.prototype.v2.{prototype_sha[:12]}.json"
    )
    prototype_relative = (
        PurePosixPath(output_relative_dir) / prototype_name
    ).as_posix()
    manifest_document = deepcopy(source_document)
    manifest_document["manifest_id"] = (
        f"action_ball_n1_{action_id}_upper_contact_counter_rally_v1"
    )
    manifest_document["action_order"] = [action_id]
    manifest_document["actions"] = [corrected_action]
    manifest_document["prototype"] = {
        "path": prototype_relative,
        "sha256": prototype_sha,
        "scope": SCOPE,
    }
    manifest_document["solver_profile_sha256"] = profile_pin[
        "solver_profile_sha256"
    ]
    manifest_document["physics_profile_sha256"] = profile_pin[
        "physics_profile_sha256"
    ]
    holdout = deepcopy(manifest_document.get("holdout"))
    if type(holdout) is not dict:
        raise N1ContactBundleError("source manifest lacks holdout")
    holdout["samples_per_action"] = max(
        HOLDOUT_SAMPLES_PER_ACTION,
        int(holdout.get("samples_per_action", 0)),
    )
    holdout["split_id"] = (
        f"heldout_ball_{action_id}_counter_rally_n1_v1"
    )
    manifest_document["holdout"] = holdout
    manifest_document["counter_rally_objective"] = objective_mapping
    manifest_document["notes"] = (
        "Contact-only N=1 diagnostic derived from exact source manifest "
        f"SHA-256 {source_actual}. The selected incoming-ball profile is "
        "preserved except that contact center/min/max Z are converted from "
        "legacy absolute W-floor height to B_yaw relative to the actual "
        "no_move spawn/goal. Canonical counter-rally RL shaping is enabled; "
        "this artifact makes no teacher landing, post-bounce, baseline, "
        "deployment, or hardware claim."
    )
    validated_manifest = manifest_module.ActionBallManifest.from_mapping(
        manifest_document
    )
    manifest_bytes = manifest_module.canonical_manifest_bytes(
        validated_manifest
    )
    manifest_sha = _sha256_bytes(manifest_bytes)
    manifest_name = (
        f"{action_id}.manifest.v3.{manifest_sha[:12]}.json"
    )
    manifest_relative = (
        PurePosixPath(output_relative_dir) / manifest_name
    ).as_posix()
    receipt = _contact_receipt(
        action_id=action_id,
        action=corrected_action,
        source_manifest_pin=source_manifest_pin,
        profile_pin=profile_pin,
        motion_pin=motion_pin,
        geometry=geometry,
        geometry_path=geometry_path,
        repo_root=root,
        state=state,
    )
    receipt_bytes = _json_bytes(receipt)
    receipt_sha = _sha256_bytes(receipt_bytes)
    receipt_name = (
        f"{action_id}.contact_alignment.v1.{receipt_sha[:12]}.json"
    )
    receipt_relative = (
        PurePosixPath(output_relative_dir) / receipt_name
    ).as_posix()
    geometry_pin = {
        "source_path": _repo_relative(
            geometry_path, root, label="geometry source"
        ),
        "source_file_sha256": _sha256_file(geometry_path),
        "payload_sha256": geometry.GEOMETRY_SOURCE_SHA256,
        "kind": geometry.EXACT_FACE_CONTACT_KIND,
    }
    bundle = {
        "schema_version": 1,
        "artifact_type": "n1_contact_training_bundle_v1",
        "action_id": action_id,
        "action_uid": int(corrected_action["action_uid"]),
        "scope": SCOPE,
        "source_manifest": source_manifest_pin,
        "profile_pins": profile_pin,
        "motion": motion_pin,
        "prototype": {
            "path": prototype_relative,
            "sha256": prototype_sha,
            "schema_version": 2,
            "scope": SCOPE,
        },
        "manifest": {
            "path": manifest_relative,
            "sha256": manifest_sha,
            "schema_version": 3,
            "action_order": [action_id],
        },
        "contact_alignment": {
            "path": receipt_relative,
            "sha256": receipt_sha,
            "schema_version": 1,
            "status": "PASS",
        },
        "geometry": geometry_pin,
        "claims": dict(CLAIMS),
    }
    bundle_bytes = _json_bytes(bundle)
    bundle_sha = _sha256_bytes(bundle_bytes)
    bundle_name = (
        f"{action_id}.bundle.v1.{bundle_sha[:12]}.json"
    )
    bundle_relative = (
        PurePosixPath(output_relative_dir) / bundle_name
    ).as_posix()
    outputs = (
        (destination / prototype_name, prototype_bytes),
        (destination / manifest_name, manifest_bytes),
        (destination / receipt_name, receipt_bytes),
        (destination / bundle_name, bundle_bytes),
    )
    destination.mkdir(parents=True, exist_ok=True)
    collisions = [path for path, _ in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "no-clobber output already exists: "
            + ", ".join(str(path) for path in collisions)
        )
    for path, payload in outputs:
        _exclusive_write(path, payload)
    try:
        loaded = manifest_module.load_action_ball_manifest(
            destination / manifest_name,
            expected_sha256=manifest_sha,
            verify_referenced_assets=True,
            repo_root=root,
        )
        if (
            loaded.manifest.action_order != (action_id,)
            or loaded.manifest.prototype.sha256 != prototype_sha
            or loaded.manifest.counter_rally_objective.sha256
            != objective_sha
        ):
            raise N1ContactBundleError(
                "written manifest failed exact N=1/objective roundtrip"
            )
    except Exception as error:
        raise N1ContactBundleError(
            "written bundle failed strict referenced-asset verification; "
            "outputs are retained for forensic inspection and never "
            "overwritten"
        ) from error
    return {
        "status": "PASS",
        "action_id": action_id,
        "bundle_path": bundle_relative,
        "bundle_sha256": bundle_sha,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "prototype_path": prototype_relative,
        "prototype_sha256": prototype_sha,
        "contact_alignment_path": receipt_relative,
        "contact_alignment_sha256": receipt_sha,
        "solver_profile_sha256": profile_pin["solver_profile_sha256"],
        "physics_profile_sha256": profile_pin["physics_profile_sha256"],
        "counter_rally_objective_profile_sha256": objective_sha,
        "landing_claim": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT_DEFAULT),
        help="exact training repository root",
    )
    parser.add_argument(
        "--action-id",
        required=True,
        choices=tuple(sorted(SUPPORTED_ACTIONS)),
    )
    parser.add_argument(
        "--source-manifest",
        default=str(SOURCE_MANIFEST_RELATIVE_PATH),
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument("--profile-pins", required=True)
    parser.add_argument(
        "--expected-profile-pins-sha256", required=True
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new repo-internal directory/namespace for content-addressed outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    repo_root = Path(arguments.repo_root).resolve(strict=True)

    def under_root(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else repo_root / candidate

    result = materialize_n1_contact_bundle(
        repo_root=repo_root,
        action_id=arguments.action_id,
        source_manifest=under_root(arguments.source_manifest),
        expected_source_manifest_sha256=(
            arguments.expected_source_manifest_sha256
        ),
        profile_pins=under_root(arguments.profile_pins),
        expected_profile_pins_sha256=(
            arguments.expected_profile_pins_sha256
        ),
        output_dir=under_root(arguments.output_dir),
        require_git_tracked_motion=True,
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
