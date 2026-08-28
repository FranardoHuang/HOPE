#!/usr/bin/env python3
"""Verify the TABLE SAFETY ASSEMBLY in a really-constructed Isaac env, and price it.

人话:真开一个 Isaac 环境,量三件事——桌子在不在、在不在该在的位置、加了它每步慢多少。

Three checks, in increasing cost:

1. ``--cfg-only`` — no simulator.  The env CFG carries either the legacy top-only obstacle or the
   ActionBall five-part assembly (top, floor-to-slab robot keep-out, net and two posts).  Full
   ActionBall binds the exact 32-body articulation pose to materialized collision-component OBBs;
   it deliberately installs no pair-filtered ContactSensors.  Every pose/extent comes back to the
   shared ``table_tennis.table_frame`` derivation.
2. default — construct the env.  Read every SPAWNED prim's world transform and CollisionAPI back
   out of USD, so what is asserted is the thing PhysX actually has, not the thing the config asked
   for.  Also confirms the termination manager lists
   ``robot_hit_table`` as an active term (that is what makes it a named metrics channel:
   ``Live/Termination/robot_hit_table`` and ``termination_reason_robot_hit_table_count``).
3. ``--bench N`` — step-time with the table against step-time without it, same seed, same env
   count.  This is the runtime-cost number.

Usage (pod, inside the Isaac venv)::

    python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
        --task HOPE-PingPong-ActionBall-AgibotA3-v0 --num-envs 4096 --bench 200

The formal ``--receipt-out`` mode is intentionally stricter than the diagnostic
checks above.  It accepts one exact fresh-N5 ActionBall manifest, reopens every
motion as immutable bytes, executes the exact five-component × 32-body
pose-OBB/four-substep controls, sweeps every frame of every ordered motion, and
only then publishes ``isaac_action_ball_table_pose_obb_smoke_v4``.  There are no command
line switches that can self-report any PASS field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import action_ball_action_set_contract as action_set_contract


ACTION_BALL_TASK_ID = "HOPE-PingPong-ActionBall-AgibotA3-v0"
RETIRED_FAKE_ACTION_BALL_TASK_ID = (
    "Tracking-Flat-AgibotA3-Hope-ActionBall-v0"
)
FRESH_N5_ACTION_IDS = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
FRESH_N5_FORBIDDEN_ACTION_IDS = frozenset({"fh_loop", "fh_block_syn"})
FORMAL_RECEIPT_CLASS = "isaac_action_ball_table_pose_obb_smoke_v4"
NOMINAL_HOLD_ARTIFACT_KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
NOMINAL_HOLD_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS = (
    "support_centroid_anchored_world_z_rotation_to_teacher_root_yaw"
)
MEASURED_BIRTH_SHARED_LOWER_SEMANTICS = (
    "shared_seed_root_leg12_plus_teacher_frame0_nonleg19"
)
MEASURED_BIRTH_FULL_SEED_SEMANTICS = (
    "teacher_yaw_aligned_full_seed_plus_exact_teacher_reference"
)
MEASURED_BIRTH_HOLDABLE_FULL_SEED_SEMANTICS = (
    "teacher_yaw_aligned_seed_plus_contact_free_hold_projection_plus_exact_"
    "teacher_reference"
)
MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS = (
    "exact_measured_teacher_frame0_root_joint_physical_birth"
)
MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS = (
    "measured_frame0_direct_if_safe_else_lexicographic_whole_body_safe_ready"
)
_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS = {
    "left_sole_floor_slack_m": 1.0e-4,
    "right_sole_floor_slack_m": 1.0e-4,
    "left_contact_load_slack_n": 1.0e-1,
    "right_contact_load_slack_n": 1.0e-1,
    "support_margin_slack_m": 1.0e-3,
    "joint_position_slack_rad": 2.0e-2,
    "qdes_slack_rad": 2.0e-2,
    "torque_slack_nm": 2.0,
    "table_clearance_slack_m": 1.0e-2,
    "root_height_slack_m": 2.0e-2,
    "root_tilt_slack_rad": 2.0e-2,
    "collision_slack_m": 5.0e-3,
    "ground_lp_residual_slack": 5.0e-8,
}
_WHOLE_BODY_SAFETY_SLACK_SCALES = {
    "left_sole_floor_slack_m": 2.0e-3,
    "right_sole_floor_slack_m": 2.0e-3,
    "left_contact_load_slack_n": 100.0,
    "right_contact_load_slack_n": 100.0,
    "support_margin_slack_m": 2.0e-2,
    "joint_position_slack_rad": 0.2,
    "qdes_slack_rad": 0.2,
    "torque_slack_nm": 20.0,
    "table_clearance_slack_m": 5.0e-2,
    "root_height_slack_m": 0.2,
    "root_tilt_slack_rad": 0.2,
    "collision_slack_m": 2.0e-2,
    "ground_lp_residual_slack": 1.0e-6,
}
_WHOLE_BODY_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256 = (
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
)
_WHOLE_BODY_GROUND_LP_EQUALITY_RESIDUAL_TOLERANCE = 2.0e-7
_WHOLE_BODY_MEASURED_RACKET_AXIS_LOCAL = (
    1.0 / math.sqrt(2.0),
    0.0,
    1.0 / math.sqrt(2.0),
)
_A3_LEG_JOINT_NAMES = frozenset(
    {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    }
)
FORMAL_PRODUCER_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py"
)
MAX_ACTION_UID = (1 << 53) - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_MODULE_PREFIX = "whole_body_tracking"
_REQUIRED_RUNTIME_MODULES = (
    "whole_body_tracking.tasks.table_tennis.table_frame",
    "whole_body_tracking.tasks.tracking.config.agibot_a3",
    "whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg",
    "whole_body_tracking.tasks.tracking.mdp.action_ball_manifest",
    "whole_body_tracking.tasks.tracking.mdp.hope_actions",
    "whole_body_tracking.tasks.tracking.mdp.hope_commands",
    "whole_body_tracking.tasks.tracking.mdp.terminations",
)
_ACTION_BALL_SOLVER_SOURCE_NAMES = (
    "continuous_questions.py",
    "hope_commands.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
)
_ACTION_BALL_SOLVER_SOURCE_DIR = PurePosixPath(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
_RACKET_GEOMETRY_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "ball_target_point",
        "site_target_mapping",
        "face_velocity_mapping",
        "source_path",
        "source_sha256",
        "geometry_source_sha256",
    }
)


class TableSmokeReceiptError(RuntimeError):
    """Formal table-smoke evidence is incomplete, mutable, or malformed."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    repo_path: str
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int


@dataclass(frozen=True)
class _MotionInput:
    motion_id: str
    action_uid: int
    family: str
    strike_phase: float
    mount_normal_sign: int
    reference_t_cycle_s: float
    file: _FileSnapshot


@dataclass(frozen=True)
class _FormalInputs:
    repo_root: Path
    source: _FileSnapshot
    manifest: _FileSnapshot
    profile_pins: _FileSnapshot
    solver_profile_sha256: str
    physics_profile_sha256: str
    solver_sources: tuple[tuple[str, _FileSnapshot], ...]
    racket_geometry_contract: Mapping[str, Any]
    motions: tuple[_MotionInput, ...]
    action_set_contract: Mapping[str, Any]


@dataclass(frozen=True)
class _NominalHoldInput:
    artifact_path: Path
    artifact_sha256: str
    document: Mapping[str, Any]
    action_id: str
    joint_names: tuple[str, ...]
    motion_path: Path
    motion_sha256: str
    teacher_root_pos: tuple[float, ...]
    teacher_root_quat: tuple[float, ...]
    teacher_joint_pos: tuple[float, ...]
    teacher_physical_separated: bool
    physical_root_pos: tuple[float, ...]
    physical_root_quat: tuple[float, ...]
    physical_joint_pos: tuple[float, ...]
    hold_qdes: tuple[float, ...]
    hold_action: tuple[float, ...]
    expected_plant: Mapping[str, Any]


@dataclass(frozen=True)
class _RuntimeActionEvidence:
    motion_id: str
    action_uid: int
    motion_sha256: str
    frame_count: int
    physics_steps: int
    complete_cycle: bool
    table_contact_count: int
    fall_count: int
    hard_limit_count: int
    unsafe_count: int
    robot_body_contract_count: int = 32


@dataclass(frozen=True)
class _RuntimeEvidence:
    origin: object
    source_commit_sha: str
    isaac_version: str
    python_executable: str
    gpu_identity: Mapping[str, Any]
    physics_steps: int
    actions: tuple[_RuntimeActionEvidence, ...]
    pose_obb_guard_pass: bool
    full_action_ball_assembly: bool
    all_five_table_components_with_pose_obb: bool
    all_five_obstacles: bool
    all_four_substeps: bool
    positive_control_pass: bool
    negative_control_pass: bool
    zero_reset_leakage: bool


_ISAAC_RUNTIME_ORIGIN: object | None = None
_app_launcher = None
_app = None
gym = None
torch = None
parse_env_cfg = None
tt_frame = None
TABLE_COMPONENT_ROLES = ()
TABLE_CONTACT_BODY_NAMES = ()
TABLE_HIT_FORCE_THRESHOLD_N = float("nan")


def _parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=ACTION_BALL_TASK_ID)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--cfg-only", action="store_true",
                    help="stop after the cfg checks (the Kit app still launches — isaaclab "
                         "cannot be imported without omni.kit)")
    ap.add_argument("--bench", type=int, default=0, help="steps to time (one arm per process)")
    ap.add_argument(
        "--contact-smoke",
        action="store_true",
        help="ActionBall only: teleport a named robot rigid body into each table component for "
        "one chosen physics substep; prove all four pose-OBB samples, termination attribution, "
        "non-finite fail-safe, legal stance negative, and post-reset zero leakage",
    )
    ap.add_argument("--table-obstacle", choices=("on", "off"), default="on",
                    help="the arm this process measures; run twice and subtract")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--motion-file", default=None,
                    help="reference clip npz. Required to CONSTRUCT the env (train.py normally "
                         "pulls it from the registry); any canonical clip will do — this script "
                         "never steps a policy, it only needs the scene to exist.")
    ap.add_argument(
        "--action-set-profile",
        default=None,
        help="formal mode only: code-owned exact action-set profile ID",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="formal mode only: exact fresh-N5 ActionBall schema-v3 manifest",
    )
    ap.add_argument(
        "--profile-pins",
        default=None,
        help=(
            "formal mode only: exact ActionBall solver/physics profile-pins "
            "JSON consumed by the manifest"
        ),
    )
    ap.add_argument(
        "--profile-pins-sha256",
        default=None,
        help=(
            "formal mode only: preregistered SHA-256 of --profile-pins"
        ),
    )
    ap.add_argument(
        "--receipt-out",
        default=None,
        help="formal mode: exclusively publish an exact PASS receipt at this repository path",
    )
    ap.add_argument("--nominal-hold", default=None, metavar="DYNAMIC_READY_JSON",
                    help="run one A3 dynamic-ready hold diagnostic")
    ap.add_argument("--nominal-hold-sha256", default=None)
    ap.add_argument("--nominal-hold-receipt-out", default=None)
    ap.add_argument(
        "--duration-s", type=float, default=1.2,
        help="nominal-hold policy-step horizon in seconds (default: 1.2)",
    )
    ap.add_argument("--screenshot-dir", default=None,
                    help="fresh nominal-hold ready/step1/step10/final screenshot directory")
    return ap.parse_args(argv)


ARGS = None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TableSmokeReceiptError(
            f"JSON value is not finite/canonicalizable: {exc}"
        ) from exc


def _canonical_ascii_json_bytes(value: Any) -> bytes:
    """Match the dynamic-ready materializer's content-seal canonicalization."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TableSmokeReceiptError(
            f"JSON value is not finite/ASCII-canonicalizable: {exc}"
        ) from exc


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TableSmokeReceiptError(
                    f"{label} has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str):
        raise TableSmokeReceiptError(
            f"{label} contains forbidden JSON constant {token!r}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except TableSmokeReceiptError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TableSmokeReceiptError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TableSmokeReceiptError(f"{label} must contain one JSON object")
    _canonical_json_bytes(value)
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TableSmokeReceiptError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _derive_action_uid(
    action_id: str,
    family: str,
    motion_sha256: str,
) -> int:
    """Canonical planner-safe action identity (exact integer <= 2**53-1)."""

    payload = _canonical_json_bytes(
        {
            "action_id": action_id,
            "content_sha256": motion_sha256,
            "family": family,
        }
    )
    return 1 + (
        int.from_bytes(hashlib.sha256(payload).digest(), "big")
        % MAX_ACTION_UID
    )


def _assert_plain_components(path: Path, label: str) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    current = Path(lexical.parts[0])
    for part in lexical.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise TableSmokeReceiptError(
                f"cannot lstat {label} path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TableSmokeReceiptError(
                f"{label} contains symlink component {current}"
            )
    try:
        return lexical.resolve(strict=True)
    except OSError as exc:
        raise TableSmokeReceiptError(
            f"cannot resolve {label} {lexical}: {exc}"
        ) from exc


def _normalized_repo_path(
    value: str | Path,
    *,
    repo_root: Path,
    label: str,
) -> tuple[Path, str]:
    raw = str(value)
    if (
        not raw
        or raw.startswith("/")
        or raw.endswith("/")
        or "\\" in raw
        or "//" in raw
    ):
        raise TableSmokeReceiptError(
            f"{label} must be a normalized repository-relative path"
        )
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise TableSmokeReceiptError(f"{label} contains path traversal")
    root = _assert_plain_components(repo_root, "repository root")
    path = _assert_plain_components(root.joinpath(*pure.parts), label)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TableSmokeReceiptError(f"{label} escaped repository root") from exc
    return path, pure.as_posix()


def _read_snapshot(
    path: Path,
    *,
    repo_root: Path,
    label: str,
) -> _FileSnapshot:
    resolved = _assert_plain_components(path, label)
    try:
        relative = resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise TableSmokeReceiptError(
            f"{label} must resolve inside repository root"
        ) from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(resolved), flags)
    except OSError as exc:
        raise TableSmokeReceiptError(f"cannot open {label}: {exc}") from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise TableSmokeReceiptError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    except BaseException as exc:
        # ``SimulationApp.close`` can end Kit with status zero before Python's
        # ordinary unhandled-exception printer runs.  Emit the evidence first.
        import traceback

        print(
            "HOPE_TABLE_DIAGNOSTIC_STAGE="
            f"main_exception:{type(exc).__name__}:{exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        os.close(descriptor)
    path_stat = os.stat(resolved, follow_symlinks=False)
    if (
        descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
        or descriptor_stat.st_size != path_stat.st_size
        or len(payload) != descriptor_stat.st_size
    ):
        raise TableSmokeReceiptError(f"{label} changed during descriptor read")
    return _FileSnapshot(
        path=resolved,
        repo_path=relative,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        device=int(descriptor_stat.st_dev),
        inode=int(descriptor_stat.st_ino),
        size=int(descriptor_stat.st_size),
    )


def _finite_tuple(value, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise TableSmokeReceiptError(f"{label} must have {size} numbers")
    result = tuple(float(item) for item in value)
    if any(
        isinstance(item, bool)
        or type(item) not in (int, float)
        or not math.isfinite(number)
        for item, number in zip(value, result)
    ):
        raise TableSmokeReceiptError(f"{label} must be finite")
    return result


def _root_yaw_rad(quaternion_wxyz: Sequence[float]) -> float:
    w, x, y, z = _finite_tuple(
        list(quaternion_wxyz), 4, "root-yaw quaternion"
    )
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise TableSmokeReceiptError("root-yaw quaternion is degenerate")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _pinned_external_file(
    value: object, expected_sha256: object, label: str
) -> tuple[Path, str]:
    path = _assert_plain_components(Path(str(value)), label)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _require_sha256(expected_sha256, f"{label} SHA-256"):
        raise TableSmokeReceiptError(f"{label} SHA-256 mismatch")
    return path, digest


def _whole_body_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
    ):
        raise TableSmokeReceiptError(f"{label} must be one finite number")
    return float(value)


def _whole_body_named_numbers(
    value: object, *, expected: Mapping[str, float], label: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise TableSmokeReceiptError(f"{label} fields are incomplete or unknown")
    return {
        name: _whole_body_number(value[name], f"{label}.{name}")
        for name in expected
    }


def _whole_body_matrix(
    value: object, *, rows: int | None, columns: int, label: str
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or (rows is not None and len(value) != rows):
        expected_rows = "one or more" if rows is None else str(rows)
        raise TableSmokeReceiptError(
            f"{label} must have {expected_rows} rows of {columns} numbers"
        )
    result = tuple(
        _finite_tuple(row, columns, f"{label} row {index}")
        for index, row in enumerate(value)
    )
    if rows is None and not result:
        raise TableSmokeReceiptError(f"{label} must not be empty")
    return result


def _whole_body_state_sha256(
    joint_pos: Sequence[float],
    root_pos: Sequence[float],
    root_quat: Sequence[float],
) -> str:
    """Reproduce ``canonical_grounded_ready.state_digest`` without NumPy."""

    digest = hashlib.sha256()
    for label, values in (
        ("joint_pos", tuple(joint_pos)),
        ("root_pos_w", tuple(root_pos)),
        ("root_quat_wxyz", tuple(root_quat)),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"float64")
        digest.update(struct.pack("=q", len(values)))
        digest.update(struct.pack(f"={len(values)}d", *values))
    return digest.hexdigest()


def _whole_body_close(left: float, right: float, *, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=tolerance)


def _whole_body_vector_norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _whole_body_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _whole_body_mapping_equal(left: object, right: Mapping[str, Any]) -> bool:
    return isinstance(left, Mapping) and dict(left) == dict(right)


def _whole_body_cross(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2])
        - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0])
        - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1])
        - float(left[1]) * float(right[0]),
    )


def _whole_body_racket_fidelity(
    value: object,
    *,
    motion_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise TableSmokeReceiptError(
            "threshold-first whole-body racket-site fidelity is missing"
        )
    reference = value.get("reference_authority")
    if not isinstance(reference, Mapping):
        raise TableSmokeReceiptError(
            "threshold-first whole-body independent racket authority is missing"
        )
    mount_sign = reference.get("robot_mount_normal_sign")
    if isinstance(mount_sign, bool) or type(mount_sign) is not int:
        raise TableSmokeReceiptError(
            "threshold-first whole-body racket mount sign is invalid"
        )
    reference_position = _finite_tuple(
        reference.get("site_pos_w_m"), 3, "measured racket blade center"
    )
    measured_face = _finite_tuple(
        reference.get("signed_face_normal_w"),
        3,
        "measured racket signed face",
    )
    measured_long = _finite_tuple(
        reference.get("long_axis_w"), 3, "measured racket long axis"
    )
    measured_face_unit = _finite_tuple(
        reference.get("signed_face_normal_w_unit"),
        3,
        "unit measured racket signed face",
    )
    measured_long_unit = _finite_tuple(
        reference.get("long_axis_w_unit"),
        3,
        "unit measured racket long axis",
    )
    axis_local = _finite_tuple(
        reference.get("robot_butt_to_blade_axis_local"),
        3,
        "racket butt-to-blade local axis",
    )
    official_rotation = _whole_body_matrix(
        reference.get("official_site_rotation_w"),
        rows=3,
        columns=3,
        label="independent measured racket official-site rotation",
    )
    if (
        reference.get("authority")
        != "independent_schema_v4_measured_racket_channel"
        or reference.get("motion_sha256") != motion_sha256
        or reference.get("frame_index") != 0
        or reference.get("position_semantics") != "physical_blade_center"
        or reference.get("normal_semantics")
        != "signed_physical_hitting_face"
        or reference.get("long_axis_semantics")
        != "measured_paddle_butt_to_blade"
        or reference.get("robot_rigid_visual_mesh_sha256")
        != _WHOLE_BODY_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        or mount_sign not in (-1, 1)
        or any(
            not _whole_body_close(actual, expected, tolerance=1.0e-15)
            for actual, expected in zip(
                axis_local, _WHOLE_BODY_MEASURED_RACKET_AXIS_LOCAL
            )
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body measured racket lost schema-v4 authority"
        )
    face_norm = _whole_body_vector_norm(measured_face)
    long_norm = _whole_body_vector_norm(measured_long)
    if (
        abs(face_norm - 1.0) > 1.0e-3
        or abs(long_norm - 1.0) > 1.0e-3
        or abs(_whole_body_dot(measured_face, measured_long)) > 1.0e-3
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-12)
            for actual, expected in zip(
                measured_face_unit,
                (value / face_norm for value in measured_face),
            )
        )
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-12)
            for actual, expected in zip(
                measured_long_unit,
                (value / long_norm for value in measured_long),
            )
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body measured racket axes are invalid"
        )
    site_y = tuple(float(mount_sign) * value for value in measured_face_unit)
    projection = _whole_body_dot(site_y, measured_long_unit)
    site_y = tuple(
        value - projection * axis
        for value, axis in zip(site_y, measured_long_unit)
    )
    site_y_norm = _whole_body_vector_norm(site_y)
    if site_y_norm <= 1.0e-12:
        raise TableSmokeReceiptError(
            "threshold-first whole-body measured racket axes are degenerate"
        )
    site_y = tuple(value / site_y_norm for value in site_y)
    local_y = (0.0, 1.0, 0.0)
    local_third = _whole_body_cross(axis_local, local_y)
    world_third = _whole_body_cross(measured_long_unit, site_y)
    # R_world_local = B_world @ B_local.T.
    expected_rotation = tuple(
        tuple(
            measured_long_unit[row] * axis_local[column]
            + site_y[row] * local_y[column]
            + world_third[row] * local_third[column]
            for column in range(3)
        )
        for row in range(3)
    )
    if any(
        not _whole_body_close(actual, expected, tolerance=2.0e-6)
        for actual_row, expected_row in zip(
            official_rotation, expected_rotation
        )
        for actual, expected in zip(actual_row, expected_row)
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body official racket-site rotation drifted"
        )

    physical_position = _finite_tuple(
        value.get("physical_site_pos_w_m"), 3, "physical racket blade center"
    )
    physical_face = _finite_tuple(
        value.get("physical_signed_face_normal_w"),
        3,
        "physical racket signed face",
    )
    physical_long = _finite_tuple(
        value.get("physical_long_axis_w"), 3, "physical racket long axis"
    )
    position_delta = _finite_tuple(
        value.get("physical_minus_measured_position_w_m"),
        3,
        "physical-minus-measured racket center",
    )
    rotation_delta = _finite_tuple(
        value.get("physical_minus_measured_rotation_vector_rad"),
        3,
        "physical-minus-measured racket rotation",
    )
    position_error = _whole_body_number(
        value.get("position_error_m"), "racket position error"
    )
    orientation_error = _whole_body_number(
        value.get("orientation_error_rad"), "racket orientation error"
    )
    face_error = _whole_body_number(
        value.get("signed_face_error_rad"), "racket signed-face error"
    )
    long_error = _whole_body_number(
        value.get("long_axis_error_rad"), "racket long-axis error"
    )
    physical_face_norm = _whole_body_vector_norm(physical_face)
    physical_long_norm = _whole_body_vector_norm(physical_long)
    if physical_face_norm <= 1.0e-12 or physical_long_norm <= 1.0e-12:
        raise TableSmokeReceiptError(
            "threshold-first whole-body physical racket axes are degenerate"
        )
    expected_position_delta = tuple(
        actual - measured
        for actual, measured in zip(physical_position, reference_position)
    )
    expected_face_error = math.acos(
        max(
            -1.0,
            min(
                1.0,
                _whole_body_dot(physical_face, measured_face_unit)
                / physical_face_norm,
            ),
        )
    )
    expected_long_error = math.acos(
        max(
            -1.0,
            min(
                1.0,
                _whole_body_dot(physical_long, measured_long_unit)
                / physical_long_norm,
            ),
        )
    )
    if (
        value.get("site_name") != "right_racket"
        or value.get("site_semantics")
        != "official_mjcf_site_against_independent_schema_v4_measured_blade"
        or value.get("independent_measured_frame0_required") is not True
        or abs(physical_face_norm - 1.0) > 2.0e-6
        or abs(physical_long_norm - 1.0) > 2.0e-6
        or abs(_whole_body_dot(physical_face, physical_long)) > 2.0e-6
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-10)
            for actual, expected in zip(position_delta, expected_position_delta)
        )
        or not _whole_body_close(
            position_error,
            _whole_body_vector_norm(expected_position_delta),
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            orientation_error,
            _whole_body_vector_norm(rotation_delta),
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            face_error, expected_face_error, tolerance=2.0e-6
        )
        or not _whole_body_close(
            long_error, expected_long_error, tolerance=2.0e-6
        )
        or min(position_error, orientation_error, face_error, long_error) < 0.0
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body racket center/face/long fidelity is invalid"
        )
    return value, reference


def _whole_body_threshold_first_contract(
    document: Mapping[str, Any],
    *,
    joint_names: tuple[str, ...],
    motion_sha256: str,
    physical: Mapping[str, Any],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    bool,
]:
    """Accept only the materializer's seedless exact-frame0 short circuit."""

    teacher = document.get("teacher_reference")
    composition = document.get("physical_birth_composition")
    static = document.get("physical_birth_static_evidence")
    ready_source = document.get("ready_source")
    sources = document.get("sources")
    runtime = document.get("runtime_plant")
    hold = document.get("hold_candidate")
    if not all(
        isinstance(value, Mapping)
        for value in (
            teacher,
            composition,
            static,
            ready_source,
            sources,
            runtime,
            hold,
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body authority fields are incomplete"
        )
    assert isinstance(teacher, Mapping)
    assert isinstance(composition, Mapping)
    assert isinstance(static, Mapping)
    assert isinstance(ready_source, Mapping)
    assert isinstance(sources, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(hold, Mapping)

    physical_q = _finite_tuple(
        physical.get("joint_pos_rad"), 31, "whole-body ready q"
    )
    physical_root = _finite_tuple(
        physical.get("root_pos_w_m"), 3, "whole-body ready root position"
    )
    physical_quat = _finite_tuple(
        physical.get("root_quat_wxyz"), 4, "whole-body ready root quaternion"
    )
    physical_velocity = _finite_tuple(
        physical.get("joint_vel_radps"), 31, "whole-body ready velocity"
    )
    teacher_q = _finite_tuple(
        teacher.get("joint_pos_rad"), 31, "whole-body teacher frame-0 q"
    )
    teacher_root = _finite_tuple(
        teacher.get("root_pos_w_m"), 3, "whole-body teacher root position"
    )
    teacher_quat = _finite_tuple(
        teacher.get("root_quat_wxyz"), 4, "whole-body teacher root quaternion"
    )
    teacher_static_velocity = _finite_tuple(
        teacher.get("static_handoff_joint_vel_radps"),
        31,
        "whole-body teacher static-handoff velocity",
    )
    delta_q = _finite_tuple(
        composition.get("physical_minus_teacher_joint_pos_rad"),
        31,
        "whole-body physical-minus-teacher q",
    )
    delta_root = _finite_tuple(
        composition.get("physical_minus_teacher_root_pos_m"),
        3,
        "whole-body physical-minus-teacher root",
    )
    delta_rotation = _finite_tuple(
        composition.get("physical_minus_teacher_root_rotation_vector_rad"),
        3,
        "whole-body physical-minus-teacher root rotation",
    )
    recorded_teacher_quat = _finite_tuple(
        composition.get("teacher_root_quat_wxyz"),
        4,
        "whole-body composition teacher quaternion",
    )
    recorded_physical_quat = _finite_tuple(
        composition.get("physical_root_quat_wxyz"),
        4,
        "whole-body composition physical quaternion",
    )
    stored_physical_quat = _finite_tuple(
        composition.get("stored_physical_root_quat_wxyz"),
        4,
        "whole-body stored physical quaternion",
    )
    audit_quat = _finite_tuple(
        composition.get("mjcf_audit_root_quat_wxyz"),
        4,
        "whole-body MuJoCo audit quaternion",
    )
    stored_quaternion_norm = _whole_body_vector_norm(physical_quat)
    if stored_quaternion_norm <= 1.0e-12:
        raise TableSmokeReceiptError(
            "threshold-first whole-body stored quaternion is degenerate"
        )
    expected_audit_quat = tuple(
        value / stored_quaternion_norm for value in physical_quat
    )
    close_endpoint = lambda left, right: math.isclose(
        left, right, rel_tol=0.0, abs_tol=1.0e-12
    )
    by_name_delta = composition.get("physical_minus_teacher_joint_pos_by_name_rad")
    if (
        teacher.get("semantics") != "exact_motion_bytes_frame0_reference"
        or teacher.get("motion_sha256") != motion_sha256
        or teacher.get("frame_index") != 0
        or composition.get("semantics")
        != MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS
        or composition.get("teacher_reference_unchanged") is not True
        or composition.get("historical_physical_birth_seed_consumed") is not False
        or composition.get("selection_priority")
        != [
            "exact_measured_frame0_if_all_safety_gates_pass",
            "lexicographic_whole_body_safe_ready_only_if_frame0_unsafe",
        ]
        or composition.get("exact_measured_frame0_selected") is not True
        or composition.get("released_root_degrees_of_freedom")
        != ["z", "roll", "pitch"]
        or composition.get("released_joint_indices") != list(range(31))
        or tuple(composition.get("released_joint_names", ())) != joint_names
        or composition.get("changed_joint_mask") != [False] * 31
        or composition.get("changed_joint_indices") != []
        or composition.get("changed_joint_names") != []
        or composition.get("teacher_and_physical_birth_differ") is not False
        or composition.get("safety_weighted_against_tracking") is not False
        or composition.get("training_authorized") is not False
        or composition.get("deployment_authorized") is not False
        or composition.get("hardware_authorized") is not False
        or composition.get("required_live_table_gate")
        != NOMINAL_HOLD_RECEIPT_KIND
        or not isinstance(by_name_delta, Mapping)
        or set(by_name_delta) != set(joint_names)
        or any(
            _whole_body_number(
                by_name_delta[name],
                f"whole-body physical-minus-teacher q for {name}",
            )
            != 0.0
            for name in joint_names
        )
        or any(value != 0.0 for value in physical_velocity)
        or any(value != 0.0 for value in teacher_static_velocity)
        or teacher.get("static_handoff_velocity_semantics")
        != "constructed_zero_joint_velocity_endpoint_not_measured_motion_velocity"
        or physical_q != teacher_q
        or physical_root != teacher_root
        or physical_quat != teacher_quat
        or any(not close_endpoint(value, 0.0) for value in delta_q)
        or any(not close_endpoint(value, 0.0) for value in delta_root)
        or any(not close_endpoint(value, 0.0) for value in delta_rotation)
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(physical_q, teacher_q)
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(physical_root, teacher_root)
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(physical_quat, teacher_quat)
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(recorded_teacher_quat, teacher_quat)
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(recorded_physical_quat, physical_quat)
        )
        or stored_physical_quat != physical_quat
        or not math.isfinite(stored_quaternion_norm)
        or abs(stored_quaternion_norm - 1.0) > 2.0e-6
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(audit_quat, expected_audit_quat)
        )
        or sources.get("physical_birth_seed") is not None
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body exact-frame0 authority is invalid"
        )
    if (
        ready_source.get("kind") != "measured_retarget_l0_diagnostic"
        or ready_source.get("frame_index") != 0
        or ready_source.get("teacher_reference_unchanged") is not True
        or ready_source.get("teacher_and_physical_birth_same") is not True
        or ready_source.get("physical_birth_semantics")
        != MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS
        or ready_source.get("plant_template_action_binding_consumed") is not False
        or ready_source.get("plant_template_delay_overridden_to_zero") is not True
        or ready_source.get("isaac_live_plant_match_required") is not True
        or ready_source.get("diagnostic_unauthorized") is not True
        or ready_source.get("training_authorized") is not False
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body ready-source authority is invalid"
        )

    handoff_keys = {
        "schema_version",
        "kind",
        "selection_semantics",
        "state_sha256_semantics",
        "physical_ready_state_sha256",
        "teacher_frame0_state_sha256",
        "mjcf_audit_state_sha256",
        "stored_root_quaternion_norm",
        "mjcf_audit_root_quat_wxyz",
        "mjcf_audit_quaternion_semantics",
        "stored_teacher_and_physical_quaternion_unchanged",
        "endpoints_bitwise_equal",
        "physical_ready_joint_velocity_exact_zero",
        "teacher_static_endpoint_joint_velocity_exact_zero",
        "measured_motion_velocity_channels_consumed",
        "not_a_motion_velocity_continuity_claim",
        "certified_transition_s",
        "required_min_wait_s",
        "torque_speed_curve_required",
        "torque_speed_non_requirement_reason",
        "runtime_transition_reference_required",
        "required_followup_hold_gate",
        "required_followup_policy_steps",
        "required_followup_physics_steps",
        "diagnostic_unauthorized",
        "training_authorized",
    }
    top_handoff = document.get("frame0_handoff")
    composition_handoff = composition.get("frame0_handoff")
    static_handoff = static.get("frame0_handoff")
    if (
        not isinstance(top_handoff, Mapping)
        or not isinstance(composition_handoff, Mapping)
        or not isinstance(static_handoff, Mapping)
        or set(top_handoff) != handoff_keys
        or dict(top_handoff) != dict(composition_handoff)
        or dict(top_handoff) != dict(static_handoff)
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body frame0 handoff is missing or tampered"
        )
    physical_state_sha = _require_sha256(
        top_handoff.get("physical_ready_state_sha256"),
        "whole-body physical-ready state SHA-256",
    )
    teacher_state_sha = _require_sha256(
        top_handoff.get("teacher_frame0_state_sha256"),
        "whole-body teacher frame0 state SHA-256",
    )
    audit_state_sha = _require_sha256(
        top_handoff.get("mjcf_audit_state_sha256"),
        "whole-body MuJoCo audit state SHA-256",
    )
    handoff_audit_quat = _finite_tuple(
        top_handoff.get("mjcf_audit_root_quat_wxyz"),
        4,
        "whole-body handoff MuJoCo audit quaternion",
    )
    handoff_quaternion_norm = _whole_body_number(
        top_handoff.get("stored_root_quaternion_norm"),
        "whole-body handoff stored quaternion norm",
    )
    if (
        top_handoff.get("schema_version") != 1
        or top_handoff.get("kind")
        != "exact_frame0_zero_duration_handoff_v1"
        or top_handoff.get("selection_semantics")
        != "threshold_first_exact_frame0_direct"
        or top_handoff.get("state_sha256_semantics")
        != "float64_array_bytes_without_quaternion_normalization_v1"
        or physical_state_sha != teacher_state_sha
        or physical_state_sha
        != _whole_body_state_sha256(physical_q, physical_root, physical_quat)
        or audit_state_sha
        != _whole_body_state_sha256(physical_q, physical_root, audit_quat)
        or audit_state_sha
        != _whole_body_state_sha256(
            physical_q, physical_root, handoff_audit_quat
        )
        or not _whole_body_close(
            handoff_quaternion_norm,
            stored_quaternion_norm,
            tolerance=1.0e-15,
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(handoff_audit_quat, audit_quat)
        )
        or top_handoff.get("mjcf_audit_quaternion_semantics")
        != "stored_root_quat_unit_normalized_for_numerical_backend_only"
        or top_handoff.get(
            "stored_teacher_and_physical_quaternion_unchanged"
        )
        is not True
        or top_handoff.get("endpoints_bitwise_equal") is not True
        or top_handoff.get("physical_ready_joint_velocity_exact_zero")
        is not True
        or top_handoff.get(
            "teacher_static_endpoint_joint_velocity_exact_zero"
        )
        is not True
        or top_handoff.get("measured_motion_velocity_channels_consumed")
        is not False
        or top_handoff.get("not_a_motion_velocity_continuity_claim")
        is not True
        or top_handoff.get("certified_transition_s") != 0.0
        or top_handoff.get("required_min_wait_s") != 0.0
        or top_handoff.get("torque_speed_curve_required") is not False
        or top_handoff.get("torque_speed_non_requirement_reason")
        != (
            "identical_stored_configuration_and_constructed_zero_joint_"
            "velocity_endpoints"
        )
        or top_handoff.get("runtime_transition_reference_required") is not False
        or top_handoff.get("required_followup_hold_gate")
        != NOMINAL_HOLD_RECEIPT_KIND
        or top_handoff.get("required_followup_policy_steps") != 200
        or top_handoff.get("required_followup_physics_steps") != 800
        or top_handoff.get("diagnostic_unauthorized") is not True
        or top_handoff.get("training_authorized") is not False
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body zero-duration handoff is invalid"
        )
    required_gate = document.get("required_next_gate")
    if (
        not isinstance(required_gate, Mapping)
        or set(required_gate)
        != {
            "kind",
            "required_policy_steps",
            "required_physics_steps",
            "required_min_wait_s",
            "minimum_horizon_semantics",
            "zero_terminal_required",
        }
        or required_gate.get("kind") != NOMINAL_HOLD_RECEIPT_KIND
        or required_gate.get("required_policy_steps") != 200
        or required_gate.get("required_physics_steps") != 800
        or required_gate.get("required_min_wait_s") != 0.0
        or required_gate.get("minimum_horizon_semantics")
        != "validated_t_hit_plus_reaction_margin"
        or required_gate.get("zero_terminal_required")
        != [
            "joint_qdes_forbidden",
            "joint_actual_forbidden",
            "robot_hit_table",
            "base_fell_tilt",
            "base_too_low",
        ]
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body nominal-hold followup must be exact 200/800"
        )

    optimizer = composition.get("optimizer_report")
    if not isinstance(optimizer, Mapping):
        raise TableSmokeReceiptError(
            "threshold-first whole-body optimizer report is missing"
        )
    optimizer_thresholds = _whole_body_named_numbers(
        optimizer.get("direct_frame0_robust_minimum_slacks"),
        expected=_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS,
        label="optimizer direct-frame0 robust minimum slacks",
    )
    optimizer_scales = _whole_body_named_numbers(
        optimizer.get("slack_scales"),
        expected=_WHOLE_BODY_SAFETY_SLACK_SCALES,
        label="optimizer safety slack scales",
    )
    if (
        optimizer.get("algorithm")
        != "exact_measured_frame0_safety_short_circuit"
        or optimizer.get("global_optimum_claimed") is not False
        or optimizer.get("stage1_objective")
        != "prefer_exact_measured_frame0_when_all_safety_gates_pass"
        or optimizer.get("stage2_objective")
        != "not_run_exact_frame0_already_safe"
        or optimizer.get("safety_weighted_against_tracking") is not False
        or optimizer.get("exact_measured_frame0_selected") is not True
        or optimizer.get("stage1_runs") != []
        or optimizer.get("stage2_success") is not True
        or optimizer.get("stage2_status") != 0
        or optimizer.get("stage2_iterations") != 0
        or optimizer.get("stage2_accepted_steps") != 0
        or tuple(optimizer.get("movable_joint_names", ())) != joint_names
        or optimizer.get("root_degrees_of_freedom")
        != ["z", "roll", "pitch"]
        or optimizer.get("racket_reference_authority")
        != "caller_supplied_independent_measurement"
        or optimizer_thresholds
        != _WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS
        or optimizer_scales != _WHOLE_BODY_SAFETY_SLACK_SCALES
        or not _whole_body_mapping_equal(
            static.get("optimizer_report"), optimizer
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body artifact selected lexicographic fallback"
        )

    static_thresholds = _whole_body_named_numbers(
        static.get("direct_frame0_robust_minimum_slacks"),
        expected=_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS,
        label="fresh direct-frame0 robust minimum slacks",
    )
    expected_threshold_sha = hashlib.sha256(
        _canonical_json_bytes(_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS)
    ).hexdigest()
    safety_slacks = _whole_body_named_numbers(
        static.get("safety_slacks"),
        expected=_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS,
        label="fresh whole-body safety slacks",
    )
    normalized_slacks = _whole_body_named_numbers(
        static.get("normalized_safety_slacks"),
        expected=_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS,
        label="fresh whole-body normalized safety slacks",
    )
    required_final_gate = _whole_body_number(
        static.get("required_final_normalized_safety_gate"),
        "required final normalized safety gate",
    )
    composition_safety = _whole_body_named_numbers(
        composition.get("safety_slacks"),
        expected=_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS,
        label="composition safety slacks",
    )
    composition_normalized = _whole_body_named_numbers(
        composition.get("normalized_safety_slacks"),
        expected=_WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS,
        label="composition normalized safety slacks",
    )
    static_stored_quat = _finite_tuple(
        static.get("stored_root_quat_wxyz"),
        4,
        "fresh static stored root quaternion",
    )
    static_audit_quat = _finite_tuple(
        static.get("mjcf_audit_root_quat_wxyz"),
        4,
        "fresh static MuJoCo audit quaternion",
    )
    static_quaternion_norm = _whole_body_number(
        static.get("stored_root_quaternion_norm"),
        "fresh static stored quaternion norm",
    )
    if (
        static.get("authority")
        != "fresh_current_exact_mjcf_whole_body_lexicographic_search"
        or static.get("selected_hold_witness_authority")
        != "new_backend_new_solver_final_state_cache_miss"
        or static.get("exact_contact_lp_reused") is not False
        or static.get("all_safety_slacks_meet_original_and_locked_gate")
        is not True
        or static_thresholds
        != _WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS
        or static.get("direct_frame0_robust_gate_sha256")
        != expected_threshold_sha
        or static.get("fresh_direct_robust_gate_passed") is not True
        or static.get("geometry_passed") is not True
        or static.get("ground_dynamics_passed") is not True
        or static.get("stored_endpoint_state_sha256") != physical_state_sha
        or static.get("mjcf_audit_state_sha256") != audit_state_sha
        or static_stored_quat != physical_quat
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(static_audit_quat, audit_quat)
        )
        or not _whole_body_close(
            static_quaternion_norm,
            stored_quaternion_norm,
            tolerance=1.0e-15,
        )
        or safety_slacks != composition_safety
        or normalized_slacks != composition_normalized
        or any(
            safety_slacks[name] < minimum
            for name, minimum in _WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS.items()
        )
        or any(
            not _whole_body_close(
                normalized_slacks[name],
                safety_slacks[name] / _WHOLE_BODY_SAFETY_SLACK_SCALES[name],
                tolerance=2.0e-12,
            )
            for name in safety_slacks
        )
        or any(value < required_final_gate for value in normalized_slacks.values())
        or not _whole_body_close(
            _whole_body_number(
                composition.get("worst_normalized_safety_slack"),
                "worst normalized safety slack",
            ),
            min(normalized_slacks.values()),
            tolerance=2.0e-12,
        )
        or not _whole_body_close(
            _whole_body_number(
                composition.get("stage1_locked_worst_normalized_safety_slack"),
                "stage1 locked normalized safety slack",
            ),
            _whole_body_number(
                optimizer.get("stage1_locked_worst_normalized_slack"),
                "optimizer stage1 locked normalized safety slack",
            ),
            tolerance=2.0e-12,
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body fresh robust safety evidence is invalid"
        )

    fidelity, racket_reference = _whole_body_racket_fidelity(
        composition.get("racket_site_fidelity"), motion_sha256=motion_sha256
    )
    if (
        not _whole_body_mapping_equal(
            static.get("racket_site_fidelity"), fidelity
        )
        or not _whole_body_mapping_equal(
            static.get("independent_measured_racket_frame0"),
            racket_reference,
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body fresh racket evidence drifted"
        )

    evaluator_contract = composition.get("evaluator_contract")
    evaluator_contract_keys = {
        "executed_qdes_lower_rad",
        "executed_qdes_upper_rad",
        "exact_joint_position_lower_rad",
        "exact_joint_position_upper_rad",
        "table_near_x_m",
        "table_half_width_m",
        "table_surface_z_m",
        "minimum_table_clearance_m",
        "minimum_root_height_m",
        "maximum_root_tilt_rad",
        "collision_pair_authority",
    }
    if (
        not isinstance(evaluator_contract, Mapping)
        or set(evaluator_contract) != evaluator_contract_keys
        or not isinstance(
            evaluator_contract.get("collision_pair_authority"), Mapping
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body evaluator contract is incomplete"
        )
    evaluator_executed_lower = _finite_tuple(
        evaluator_contract.get("executed_qdes_lower_rad"),
        31,
        "whole-body evaluator executed qdes lower",
    )
    evaluator_executed_upper = _finite_tuple(
        evaluator_contract.get("executed_qdes_upper_rad"),
        31,
        "whole-body evaluator executed qdes upper",
    )
    evaluator_joint_lower = _finite_tuple(
        evaluator_contract.get("exact_joint_position_lower_rad"),
        31,
        "whole-body evaluator exact joint-position lower",
    )
    evaluator_joint_upper = _finite_tuple(
        evaluator_contract.get("exact_joint_position_upper_rad"),
        31,
        "whole-body evaluator exact joint-position upper",
    )
    evaluator_table = tuple(
        _whole_body_number(
            evaluator_contract.get(name), f"whole-body evaluator {name}"
        )
        for name in (
            "table_near_x_m",
            "table_half_width_m",
            "table_surface_z_m",
        )
    )
    evaluator_minimum_table_clearance = _whole_body_number(
        evaluator_contract.get("minimum_table_clearance_m"),
        "whole-body evaluator minimum table clearance",
    )
    evaluator_minimum_root_height = _whole_body_number(
        evaluator_contract.get("minimum_root_height_m"),
        "whole-body evaluator minimum root height",
    )
    evaluator_maximum_root_tilt = _whole_body_number(
        evaluator_contract.get("maximum_root_tilt_rad"),
        "whole-body evaluator maximum root tilt",
    )
    evaluator_collision_authority = evaluator_contract[
        "collision_pair_authority"
    ]

    model_source = sources.get("mujoco_model")
    if not isinstance(model_source, Mapping):
        raise TableSmokeReceiptError(
            "threshold-first whole-body exact MuJoCo source is missing"
        )
    _pinned_external_file(
        model_source.get("path"),
        model_source.get("sha256"),
        "whole-body exact MuJoCo model",
    )
    ground_model_binding = _require_sha256(
        model_source.get("ground_model_binding_sha256"),
        "whole-body ground-model binding SHA-256",
    )
    witness = static.get("evaluator_evidence")
    if not isinstance(witness, Mapping):
        raise TableSmokeReceiptError(
            "threshold-first whole-body fresh evaluator witness is missing"
        )
    solver_report = witness.get("solver_report")
    if not isinstance(solver_report, Mapping):
        raise TableSmokeReceiptError(
            "threshold-first whole-body fresh LP solver report is missing"
        )
    witness_q = _finite_tuple(
        witness.get("evaluated_joint_pos_rad"), 31, "fresh LP evaluated q"
    )
    witness_root = _finite_tuple(
        witness.get("evaluated_root_pos_w_m"), 3, "fresh LP evaluated root"
    )
    witness_quat = _finite_tuple(
        witness.get("evaluated_root_quat_wxyz"),
        4,
        "fresh LP evaluated root quaternion",
    )
    witness_sole_distances = _finite_tuple(
        witness.get("sole_minimum_distance_m"),
        2,
        "fresh exact sole-floor distances",
    )
    witness_joint_lower = _finite_tuple(
        witness.get("exact_joint_position_lower_rad"),
        31,
        "fresh exact joint-position lower",
    )
    witness_joint_upper = _finite_tuple(
        witness.get("exact_joint_position_upper_rad"),
        31,
        "fresh exact joint-position upper",
    )
    if (
        witness.get("exact_contact_lp_reused") is not True
        or witness.get("lp_feasible") is not True
        or witness.get("lp_error") is not None
        or witness.get("lp_objective")
        != "hold_minimax_normalized_available_torque"
        or witness.get("exact_state_lp_cache_hit") is not False
        or solver_report.get("exact_state_lp_cache_hit") is not False
        or solver_report.get("model_binding") != ground_model_binding
        or witness.get("evaluated_state_sha256") != audit_state_sha
        or witness.get("required_minimum_normal_force_per_contact_n") != 0.1
        or witness.get("required_minimum_normal_force_per_foot_n") != 1.0
        or witness_joint_lower != evaluator_joint_lower
        or witness_joint_upper != evaluator_joint_upper
        or any(
            not lower < position < upper
            for lower, position, upper in zip(
                witness_joint_lower, physical_q, witness_joint_upper
            )
        )
        or not _whole_body_close(
            safety_slacks["left_sole_floor_slack_m"],
            2.0e-3 - abs(witness_sole_distances[0]),
            tolerance=2.0e-12,
        )
        or not _whole_body_close(
            safety_slacks["right_sole_floor_slack_m"],
            2.0e-3 - abs(witness_sole_distances[1]),
            tolerance=2.0e-12,
        )
        or not _whole_body_close(
            safety_slacks["joint_position_slack_rad"],
            min(
                min(position - lower, upper - position)
                for lower, position, upper in zip(
                    witness_joint_lower, physical_q, witness_joint_upper
                )
            ),
            tolerance=2.0e-12,
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(witness_q, physical_q)
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(witness_root, physical_root)
        )
        or any(
            not close_endpoint(actual, expected)
            for actual, expected in zip(witness_quat, audit_quat)
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body fresh cache-miss 0.1N+1N LP authority is invalid"
        )

    rows_raw = witness.get("mujoco_row_for_runtime_joint")
    actuated_raw = witness.get("mujoco_actuated_dof_indices")
    if (
        not isinstance(rows_raw, list)
        or len(rows_raw) != 31
        or any(type(value) is not int for value in rows_raw)
        or sorted(rows_raw) != list(range(31))
        or not isinstance(actuated_raw, list)
        or len(actuated_raw) != 31
        or any(type(value) is not int for value in actuated_raw)
        or actuated_raw != list(range(6, 37))
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body LP actuator permutation is invalid"
        )
    rows = tuple(rows_raw)
    vector_names = (
        "model_tau_lower_mujoco_row_order_nm",
        "model_tau_upper_mujoco_row_order_nm",
        "runtime_tau_lower_runtime_order_nm",
        "runtime_tau_upper_runtime_order_nm",
        "runtime_tau_lower_mujoco_row_order_nm",
        "runtime_tau_upper_mujoco_row_order_nm",
        "effective_tau_lower_mujoco_row_order_nm",
        "effective_tau_upper_mujoco_row_order_nm",
    )
    witness_vectors = {
        name: _finite_tuple(witness.get(name), 31, f"fresh LP {name}")
        for name in vector_names
    }
    hold_vectors = {
        name: _finite_tuple(hold.get(name), 31, f"selected hold {name}")
        for name in vector_names
    }
    executed_lower = _finite_tuple(
        runtime.get("executed_qdes_lower_rad"), 31, "executed qdes lower"
    )
    executed_upper = _finite_tuple(
        runtime.get("executed_qdes_upper_rad"), 31, "executed qdes upper"
    )
    witness_executed_lower = _finite_tuple(
        witness.get("executed_qdes_lower_rad"),
        31,
        "fresh LP executed qdes lower",
    )
    witness_executed_upper = _finite_tuple(
        witness.get("executed_qdes_upper_rad"),
        31,
        "fresh LP executed qdes upper",
    )
    tau_model = _finite_tuple(
        witness.get("actuator_generalized_force_mujoco_row_order_nm"),
        31,
        "fresh LP model-order torque",
    )
    tau_runtime = _finite_tuple(
        witness.get("actuator_generalized_force_runtime_order_nm"),
        31,
        "fresh LP runtime-order torque",
    )
    qdes = _finite_tuple(
        witness.get("hold_qdes_joint_pos_rad"), 31, "fresh LP hold qdes"
    )
    selected_qdes = _finite_tuple(
        hold.get("hold_qdes_joint_pos_rad"), 31, "selected hold qdes"
    )
    kp = _finite_tuple(
        runtime.get("joint_stiffness"), 31, "whole-body joint stiffness"
    )
    effort = _finite_tuple(
        runtime.get("joint_effort_limits"), 31, "whole-body joint effort"
    )
    expected_runtime_lower = tuple(
        max(-limit, gain * (lower - position))
        for limit, gain, lower, position in zip(
            effort, kp, executed_lower, physical_q
        )
    )
    expected_runtime_upper = tuple(
        min(limit, gain * (upper - position))
        for limit, gain, upper, position in zip(
            effort, kp, executed_upper, physical_q
        )
    )
    runtime_lower = witness_vectors["runtime_tau_lower_runtime_order_nm"]
    runtime_upper = witness_vectors["runtime_tau_upper_runtime_order_nm"]
    runtime_lower_model = witness_vectors[
        "runtime_tau_lower_mujoco_row_order_nm"
    ]
    runtime_upper_model = witness_vectors[
        "runtime_tau_upper_mujoco_row_order_nm"
    ]
    model_lower = witness_vectors["model_tau_lower_mujoco_row_order_nm"]
    model_upper = witness_vectors["model_tau_upper_mujoco_row_order_nm"]
    effective_lower = witness_vectors[
        "effective_tau_lower_mujoco_row_order_nm"
    ]
    effective_upper = witness_vectors[
        "effective_tau_upper_mujoco_row_order_nm"
    ]
    mapped_runtime_lower = [0.0] * 31
    mapped_runtime_upper = [0.0] * 31
    for runtime_index, model_index in enumerate(rows):
        mapped_runtime_lower[model_index] = runtime_lower[runtime_index]
        mapped_runtime_upper[model_index] = runtime_upper[runtime_index]
    if (
        any(gain <= 0.0 for gain in kp)
        or witness_executed_lower != executed_lower
        or witness_executed_upper != executed_upper
        or evaluator_executed_lower != executed_lower
        or evaluator_executed_upper != executed_upper
        or witness_vectors != hold_vectors
        or qdes != selected_qdes
        or tau_runtime
        != tuple(tau_model[rows[index]] for index in range(31))
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-9)
            for actual, expected in zip(runtime_lower, expected_runtime_lower)
        )
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-9)
            for actual, expected in zip(runtime_upper, expected_runtime_upper)
        )
        or tuple(mapped_runtime_lower) != runtime_lower_model
        or tuple(mapped_runtime_upper) != runtime_upper_model
        or any(
            not _whole_body_close(
                effective_lower[index],
                max(model_lower[index], runtime_lower_model[index]),
                tolerance=2.0e-9,
            )
            or not _whole_body_close(
                effective_upper[index],
                min(model_upper[index], runtime_upper_model[index]),
                tolerance=2.0e-9,
            )
            or not effective_lower[index] <= tau_model[index] <= effective_upper[index]
            for index in range(31)
        )
        or any(
            not _whole_body_close(
                qdes[index],
                physical_q[index] + tau_runtime[index] / kp[index],
                tolerance=2.0e-10,
            )
            or not executed_lower[index] < qdes[index] < executed_upper[index]
            for index in range(31)
        )
        or hold.get("hold_qdes_mode") != "fresh_static_lp"
        or not isinstance(hold.get("selected_hold_authority"), Mapping)
        or hold["selected_hold_authority"].get("semantics")
        != "fresh_new_backend_whole_body_final_state_0p1n_static_lp"
        or hold["selected_hold_authority"].get(
            "source_physical_birth_seed_sha256"
        )
        is not None
        or hold["selected_hold_authority"].get("inherited_hold_claim") is not False
        or hold.get("lp_objective")
        != "hold_minimax_normalized_available_torque"
        or tuple(hold.get("mujoco_row_for_runtime_joint", ())) != rows
        or hold.get("mujoco_actuated_dof_indices") != actuated_raw
        or _finite_tuple(
            hold.get("actuator_generalized_force_mujoco_row_order_nm"),
            31,
            "selected hold model-order torque",
        )
        != tau_model
        or _finite_tuple(
            hold.get("actuator_generalized_force_runtime_order_nm"),
            31,
            "selected hold runtime-order torque",
        )
        != tau_runtime
        or hold.get("solver_report_role")
        != "selected_whole_body_final_state_single_witness"
        or not _whole_body_mapping_equal(
            hold.get("solver_report"), solver_report
        )
        or not isinstance(witness.get("actuator_limit_contract"), Mapping)
        or not _whole_body_mapping_equal(
            hold.get("actuator_limit_contract"),
            witness.get("actuator_limit_contract"),
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body single-witness qdes/torque evidence is invalid"
        )

    contact_normals = tuple(
        _whole_body_number(value, "fresh LP contact normal")
        for value in witness.get("normal_force_per_contact_n", ())
    )
    per_foot_normals = _finite_tuple(
        witness.get("normal_force_per_foot_n"),
        2,
        "fresh LP normal force per foot",
    )
    minimum_per_foot = _finite_tuple(
        witness.get("minimum_normal_force_per_contact_per_foot_n"),
        2,
        "fresh LP minimum contact normal per foot",
    )
    cop_margins = _finite_tuple(
        witness.get("cop_interior_margin_per_foot_m"),
        2,
        "fresh LP CoP interior margin",
    )
    contact_geometry = solver_report.get("contact_geometry")
    feet = (
        contact_geometry.get("feet")
        if isinstance(contact_geometry, Mapping)
        else None
    )
    if (
        len(contact_normals) < 6
        or any(value < 0.1 - 1.0e-7 for value in contact_normals)
        or not isinstance(feet, list)
        or len(feet) != 2
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body contact witness is invalid"
        )
    calculated_foot_normals: list[float] = []
    calculated_foot_minima: list[float] = []
    cursor = 0
    for foot_index, row in enumerate(feet):
        support_range = row.get("support_point_range") if isinstance(row, Mapping) else None
        if (
            not isinstance(support_range, list)
            or len(support_range) != 2
            or any(type(value) is not int for value in support_range)
            or support_range[0] != cursor
            or not cursor < support_range[1] <= len(contact_normals)
        ):
            raise TableSmokeReceiptError(
                f"threshold-first whole-body foot-{foot_index} support range is invalid"
            )
        values = contact_normals[support_range[0] : support_range[1]]
        calculated_foot_normals.append(sum(values))
        calculated_foot_minima.append(min(values))
        cursor = support_range[1]
    global_support_margin = _whole_body_number(
        witness.get("global_support_margin_m"), "fresh LP global support margin"
    )
    _whole_body_matrix(
        witness.get("support_hull_floor_xy_m"),
        rows=None,
        columns=2,
        label="fresh LP support hull",
    )
    expected_support_slack = min(
        global_support_margin - 5.0e-4,
        min(cop_margins) - 5.0e-4,
    )
    if (
        cursor != len(contact_normals)
        or any(value < 1.0 - 1.0e-7 for value in per_foot_normals)
        or any(value <= 0.0 for value in cop_margins)
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-8)
            for actual, expected in zip(
                per_foot_normals, calculated_foot_normals
            )
        )
        or any(
            not _whole_body_close(actual, expected, tolerance=2.0e-8)
            for actual, expected in zip(
                minimum_per_foot, calculated_foot_minima
            )
        )
        or tuple(solver_report.get("normal_force_per_contact_n", ()))
        != contact_normals
        or tuple(solver_report.get("normal_force_per_foot_n", ()))
        != per_foot_normals
        or tuple(solver_report.get("cop_interior_margin_per_foot_m", ()))
        != cop_margins
        or not _whole_body_close(
            safety_slacks["left_contact_load_slack_n"],
            min(minimum_per_foot[0] - 0.1, per_foot_normals[0] - 1.0),
            tolerance=2.0e-8,
        )
        or not _whole_body_close(
            safety_slacks["right_contact_load_slack_n"],
            min(minimum_per_foot[1] - 0.1, per_foot_normals[1] - 1.0),
            tolerance=2.0e-8,
        )
        or not _whole_body_close(
            safety_slacks["support_margin_slack_m"],
            expected_support_slack,
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            safety_slacks["qdes_slack_rad"],
            min(
                min(qdes[index] - executed_lower[index], executed_upper[index] - qdes[index])
                for index in range(31)
            ),
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            safety_slacks["torque_slack_nm"],
            min(
                min(tau_model[index] - effective_lower[index], effective_upper[index] - tau_model[index])
                for index in range(31)
            ),
            tolerance=2.0e-8,
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body contact/CoP/static margin evidence is invalid"
        )

    table_geometry = witness.get("table_geometry")
    root_limits = witness.get("root_limits")
    collision = witness.get("collision_clearance")
    if not all(
        isinstance(value, Mapping)
        for value in (table_geometry, root_limits, collision)
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body static geometry evidence is missing"
        )
    assert isinstance(table_geometry, Mapping)
    assert isinstance(root_limits, Mapping)
    assert isinstance(collision, Mapping)
    authority_rows = {
        "self_collision_geom_id_pairs": collision.get(
            "self_collision_geom_id_pairs"
        ),
        "unsupported_floor_robot_geom_ids": collision.get(
            "unsupported_floor_robot_geom_ids"
        ),
        "expected_foot_floor_geom_ids": collision.get(
            "expected_foot_floor_geom_ids"
        ),
        "floor_geom_id": collision.get("floor_geom_id"),
    }
    authority_sha = hashlib.sha256(
        _canonical_json_bytes(authority_rows)
    ).hexdigest()
    collision_authority_keys = {
        *authority_rows,
        "pair_authority_sha256",
        "enabled_self_pair_count",
        "unsupported_floor_pair_count",
        "required_clearance_m",
        "capped_clearance_m",
        "bisection_tolerance_m",
        "distance_semantics",
    }
    if set(evaluator_collision_authority) != collision_authority_keys:
        raise TableSmokeReceiptError(
            "threshold-first whole-body collision evaluator authority is invalid"
        )
    witness_collision_authority = {
        name: collision.get(name) for name in collision_authority_keys
    }
    realized_collision = _whole_body_number(
        collision.get("realized_capped_minimum_clearance_m"),
        "realized collision clearance",
    )
    table_clearance = _whole_body_number(
        witness.get("conservative_table_clearance_m"),
        "conservative table clearance",
    )
    table_contract_values = tuple(
        _whole_body_number(
            table_geometry.get(name), f"fresh LP table geometry {name}"
        )
        for name in ("near_x_m", "half_width_m", "surface_z_m")
    )
    equality_residual = _whole_body_number(
        witness.get("equality_residual"), "LP equality residual"
    )
    root_residual = _whole_body_number(
        witness.get("root_residual"), "LP root residual"
    )
    _w, x, y, _z = physical_quat
    quat_norm = math.sqrt(sum(value * value for value in physical_quat))
    root_tilt = math.acos(
        max(-1.0, min(1.0, 1.0 - 2.0 * ((x / quat_norm) ** 2 + (y / quat_norm) ** 2)))
    )
    if (
        table_geometry.get("required_clearance_m") != 1.0e-2
        or table_contract_values != evaluator_table
        or evaluator_minimum_table_clearance != 1.0e-2
        or table_geometry.get("semantics")
        != "collision_sphere_separation_from_overapproximated_near_side_table_prism"
        or root_limits.get("minimum_height_m") != 0.5
        or root_limits.get("maximum_tilt_rad") != 0.7
        or evaluator_minimum_root_height != 0.5
        or evaluator_maximum_root_tilt != 0.7
        or witness_collision_authority != dict(evaluator_collision_authority)
        or collision.get("pair_authority_sha256") != authority_sha
        or collision.get("enabled_self_pair_count")
        != len(authority_rows["self_collision_geom_id_pairs"] or ())
        or collision.get("unsupported_floor_pair_count")
        != len(authority_rows["unsupported_floor_robot_geom_ids"] or ())
        or collision.get("required_clearance_m") != 2.0e-3
        or collision.get("capped_clearance_m") != 2.0e-2
        or collision.get("bisection_tolerance_m") != 1.0e-4
        or collision.get("positive_unsaturated_conservative_deduction_m")
        != 1.0e-4
        or collision.get("unsupported_contacts") != []
        or collision.get("self_collision_pairs") != []
        or not _whole_body_close(
            safety_slacks["table_clearance_slack_m"],
            table_clearance - 1.0e-2,
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            safety_slacks["root_height_slack_m"],
            physical_root[2] - 0.5,
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            safety_slacks["root_tilt_slack_rad"],
            0.7 - root_tilt,
            tolerance=2.0e-10,
        )
        or not _whole_body_close(
            safety_slacks["collision_slack_m"],
            realized_collision - 2.0e-3,
            tolerance=2.0e-10,
        )
        or equality_residual < 0.0
        or root_residual < 0.0
        or not _whole_body_close(
            safety_slacks["ground_lp_residual_slack"],
            _WHOLE_BODY_GROUND_LP_EQUALITY_RESIDUAL_TOLERANCE
            - max(equality_residual, root_residual),
            tolerance=2.0e-15,
        )
    ):
        raise TableSmokeReceiptError(
            "threshold-first whole-body fresh static scene evidence is invalid"
        )
    return teacher_root, teacher_quat, teacher_q, False


def _nominal_teacher_physical_contract(
    document: Mapping[str, Any],
    *,
    joint_names: tuple[str, ...],
    motion_sha256: str,
    physical: Mapping[str, Any],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    bool,
]:
    """Validate teacher/physical split while preserving legacy same-frame input."""

    physical_q = _finite_tuple(
        physical.get("joint_pos_rad"), 31, "ready q"
    )
    physical_root = _finite_tuple(
        physical.get("root_pos_w_m"), 3, "root position"
    )
    physical_quat = _finite_tuple(
        physical.get("root_quat_wxyz"), 4, "root quaternion"
    )
    teacher = document.get("teacher_reference")
    composition = document.get("physical_birth_composition")
    if teacher is None and composition is None:
        return physical_root, physical_quat, physical_q, False
    if not isinstance(teacher, Mapping) or not isinstance(
        composition, Mapping
    ):
        raise TableSmokeReceiptError(
            "teacher reference and physical-birth composition must appear together"
        )
    ready_source = document.get("ready_source")
    static_evidence = document.get("physical_birth_static_evidence")
    sources = document.get("sources")
    seed_source = (
        sources.get("physical_birth_seed")
        if isinstance(sources, Mapping)
        else None
    )
    composition_semantics = composition.get("semantics")
    if (
        composition_semantics
        == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS
    ):
        return _whole_body_threshold_first_contract(
            document,
            joint_names=joint_names,
            motion_sha256=motion_sha256,
            physical=physical,
        )
    direct_frame0 = (
        composition_semantics == MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS
    )
    if direct_frame0:
        teacher_root = _finite_tuple(
            teacher.get("root_pos_w_m"), 3, "teacher root position"
        )
        teacher_quat = _finite_tuple(
            teacher.get("root_quat_wxyz"), 4, "teacher root quaternion"
        )
        teacher_q = _finite_tuple(
            teacher.get("joint_pos_rad"), 31, "teacher frame-0 q"
        )
        delta_q = _finite_tuple(
            composition.get("physical_minus_teacher_joint_pos_rad"),
            31,
            "physical-minus-teacher q",
        )
        delta_root = _finite_tuple(
            composition.get("physical_minus_teacher_root_pos_m"),
            3,
            "physical-minus-teacher root",
        )
        recorded_teacher_quat = _finite_tuple(
            composition.get("teacher_root_quat_wxyz"),
            4,
            "composition teacher quaternion",
        )
        recorded_physical_quat = _finite_tuple(
            composition.get("physical_root_quat_wxyz"),
            4,
            "composition physical quaternion",
        )
        backend_quat = composition.get("current_mjcf_audit_quaternion")
        close = lambda a, b: math.isclose(
            a, b, rel_tol=0.0, abs_tol=1.0e-12
        )
        if (
            teacher.get("semantics")
            != "exact_motion_bytes_frame0_reference"
            or teacher.get("motion_sha256") != motion_sha256
            or teacher.get("frame_index") != 0
            or composition.get("teacher_root_exactly_preserved") is not True
            or composition.get("teacher_all_joints_exactly_preserved") is not True
            or composition.get("teacher_and_physical_birth_differ") is not False
            or composition.get("historical_physical_birth_seed_consumed") is not False
            or composition.get("required_live_table_gate")
            != NOMINAL_HOLD_RECEIPT_KIND
            or not isinstance(backend_quat, Mapping)
            or backend_quat.get("semantics")
            != "unit_normalization_for_numerical_backend_only"
            or backend_quat.get(
                "stored_teacher_and_physical_quaternion_unchanged"
            )
            is not True
            or not isinstance(ready_source, Mapping)
            or ready_source.get("teacher_reference_unchanged") is not True
            or ready_source.get("teacher_and_physical_birth_same") is not True
            or ready_source.get("physical_birth_semantics")
            != MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS
            or not isinstance(static_evidence, Mapping)
            or static_evidence.get("geometry_passed") is not True
            or static_evidence.get("ground_dynamics_passed") is not True
            or seed_source is not None
            or any(not close(a, b) for a, b in zip(physical_q, teacher_q))
            or any(not close(a, b) for a, b in zip(physical_root, teacher_root))
            or any(not close(a, b) for a, b in zip(physical_quat, teacher_quat))
            or any(not close(value, 0.0) for value in delta_q)
            or any(not close(value, 0.0) for value in delta_root)
            or any(
                not close(a, b)
                for a, b in zip(recorded_teacher_quat, teacher_quat)
            )
            or any(
                not close(a, b)
                for a, b in zip(recorded_physical_quat, teacher_quat)
            )
        ):
            raise TableSmokeReceiptError(
                "direct measured frame0 physical-birth authority is invalid"
            )
        return teacher_root, teacher_quat, teacher_q, False
    if (
        teacher.get("semantics") != "exact_motion_bytes_frame0_reference"
        or teacher.get("motion_sha256") != motion_sha256
        or teacher.get("frame_index") != 0
        or composition_semantics
        not in (
            MEASURED_BIRTH_SHARED_LOWER_SEMANTICS,
            MEASURED_BIRTH_FULL_SEED_SEMANTICS,
            MEASURED_BIRTH_HOLDABLE_FULL_SEED_SEMANTICS,
        )
        or composition.get("teacher_and_physical_birth_differ") is not True
        or not isinstance(ready_source, Mapping)
        or ready_source.get("teacher_reference_unchanged") is not True
        or ready_source.get("teacher_and_physical_birth_same") is not False
        or ready_source.get("physical_birth_semantics")
        != composition_semantics
        or "original_motion_frame0_preserved" in ready_source
        or not isinstance(static_evidence, Mapping)
        or static_evidence.get("geometry_passed") is not True
        or static_evidence.get("ground_dynamics_passed") is not True
        or not isinstance(seed_source, Mapping)
        or seed_source.get("source_role") != "numerical_seed_only"
        or seed_source.get("inherited_model_identity") is not False
        or seed_source.get("inherited_hold_claim") is not False
        or seed_source.get("inherited_nominal_hold_claim") is not False
    ):
        raise TableSmokeReceiptError(
            "dynamic-ready teacher/physical-birth authority is invalid"
        )
    teacher_root = _finite_tuple(
        teacher.get("root_pos_w_m"), 3, "teacher root position"
    )
    teacher_quat = _finite_tuple(
        teacher.get("root_quat_wxyz"), 4, "teacher root quaternion"
    )
    teacher_q = _finite_tuple(
        teacher.get("joint_pos_rad"), 31, "teacher frame-0 q"
    )
    delta_q = _finite_tuple(
        composition.get("physical_minus_teacher_joint_pos_rad"),
        31,
        "physical-minus-teacher q",
    )
    delta_root = _finite_tuple(
        composition.get("physical_minus_teacher_root_pos_m"),
        3,
        "physical-minus-teacher root",
    )
    recorded_teacher_quat = _finite_tuple(
        composition.get("teacher_root_quat_wxyz"),
        4,
        "composition teacher quaternion",
    )
    recorded_physical_quat = _finite_tuple(
        composition.get("physical_root_quat_wxyz"),
        4,
        "composition physical quaternion",
    )
    alignment = composition.get("seed_world_yaw_alignment")
    if not isinstance(alignment, Mapping):
        raise TableSmokeReceiptError(
            "physical-birth seed yaw alignment is missing"
        )
    realized_alignment = alignment.get("realized_current_mjcf_fk")
    if (
        alignment.get("schema_version") != 1
        or alignment.get("semantics")
        != MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS
        or alignment.get("support_centroid_preserved") is not True
        or alignment.get("seed_tilt_preserved") is not True
        or alignment.get("teacher_yaw_exact") is not True
        or not isinstance(realized_alignment, Mapping)
        or realized_alignment.get("authority") != "current_exact_mjcf_fk"
        or realized_alignment.get("semantics")
        != MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS
        or realized_alignment.get("passed") is not True
    ):
        raise TableSmokeReceiptError(
            "physical-birth seed yaw alignment authority is invalid"
        )
    aligned_root_quat = _finite_tuple(
        alignment.get("aligned_root_quat_wxyz"),
        4,
        "aligned seed root quaternion",
    )
    seed_tilt = float(alignment.get("seed_root_tilt_rad", math.nan))
    aligned_tilt = float(alignment.get("aligned_root_tilt_rad", math.nan))
    recorded_yaw_error = float(
        alignment.get("aligned_minus_teacher_yaw_rad", math.nan)
    )
    realized_error_fields = (
        "maximum_foot_position_error_m",
        "maximum_foot_rotation_matrix_error",
        "support_centroid_xy_error_m",
        "maximum_foot_height_error_m",
    )
    realized_errors = tuple(
        float(realized_alignment.get(field, math.nan))
        for field in realized_error_fields
    )
    yaw_error = math.atan2(
        math.sin(_root_yaw_rad(physical_quat) - _root_yaw_rad(teacher_quat)),
        math.cos(_root_yaw_rad(physical_quat) - _root_yaw_rad(teacher_quat)),
    )
    if (
        not all(math.isfinite(value) for value in realized_errors)
        or any(value < 0.0 or value > 2.0e-10 for value in realized_errors)
        or not math.isfinite(seed_tilt)
        or not math.isfinite(aligned_tilt)
        or not math.isfinite(recorded_yaw_error)
        or abs(seed_tilt - aligned_tilt) > 1.0e-12
        or abs(recorded_yaw_error) > 1.0e-12
        or abs(yaw_error) > 1.0e-9
        or any(
            not math.isclose(
                aligned_root_quat[index], physical_quat[index],
                rel_tol=0.0, abs_tol=1.0e-12,
            )
            for index in range(4)
        )
    ):
        raise TableSmokeReceiptError(
            "physical birth is not an exact teacher-yaw-aligned seed"
        )
    leg_indices_raw = composition.get("leg_joint_indices")
    nonleg_indices_raw = composition.get("nonleg_joint_indices")
    leg_names = composition.get("leg_joint_names")
    nonleg_names = composition.get("nonleg_joint_names")
    if (
        not isinstance(leg_indices_raw, list)
        or any(type(value) is not int for value in leg_indices_raw)
        or not isinstance(nonleg_indices_raw, list)
        or any(type(value) is not int for value in nonleg_indices_raw)
    ):
        raise TableSmokeReceiptError("physical-birth joint mapping is invalid")
    leg_indices = tuple(leg_indices_raw)
    nonleg_indices = tuple(nonleg_indices_raw)
    expected_leg = tuple(
        index
        for index, name in enumerate(joint_names)
        if name in _A3_LEG_JOINT_NAMES
    )
    expected_nonleg = tuple(
        index for index in range(31) if index not in frozenset(expected_leg)
    )
    if (
        leg_indices != expected_leg
        or nonleg_indices != expected_nonleg
        or tuple(leg_names or ())
        != tuple(joint_names[index] for index in expected_leg)
        or tuple(nonleg_names or ())
        != tuple(joint_names[index] for index in expected_nonleg)
        or len(expected_leg) != 12
        or len(expected_nonleg) != 19
    ):
        raise TableSmokeReceiptError("physical-birth leg/nonleg mapping drifted")
    close = lambda a, b: math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-12)
    if composition_semantics == MEASURED_BIRTH_SHARED_LOWER_SEMANTICS:
        if (
            composition.get("teacher_nonleg_exactly_preserved") is not True
            or composition.get("seed_all_joints_exactly_preserved") is not None
        ):
            raise TableSmokeReceiptError(
                "shared-lower physical-birth composition is invalid"
            )
    elif composition_semantics == MEASURED_BIRTH_FULL_SEED_SEMANTICS:
        seed_joint_indices = composition.get("seed_joint_indices")
        seed_joint_names = composition.get("seed_joint_names")
        if (
            composition.get("teacher_nonleg_exactly_preserved") is not False
            or composition.get("seed_all_joints_exactly_preserved") is not True
            or seed_joint_indices != list(range(31))
            or tuple(seed_joint_names or ()) != tuple(joint_names)
        ):
            raise TableSmokeReceiptError(
                "full-seed physical-birth composition is invalid"
            )
    else:
        seed_delta = _finite_tuple(
            composition.get("physical_minus_seed_joint_pos_rad"),
            31,
            "projected physical-minus-seed q",
        )
        projection = composition.get("contact_free_hold_projection")
        changed_indices = (
            projection.get("changed_joint_indices")
            if isinstance(projection, Mapping)
            else None
        )
        changed_names = (
            projection.get("changed_joint_names")
            if isinstance(projection, Mapping)
            else None
        )
        if (
            composition.get("teacher_nonleg_exactly_preserved") is not False
            or composition.get("seed_all_joints_exactly_preserved") is not False
            or composition.get("seed_root_and_leg_joints_exactly_preserved")
            is not True
            or not isinstance(projection, Mapping)
            or projection.get("schema_version") != 1
            or projection.get("semantics")
            != "iterated_exact_bias_contact_free_pd_travel_projection"
            or projection.get("root_changed") is not False
            or projection.get("leg_joints_changed") is not False
            or projection.get("final_exact_ground_lp_feasible") is not True
            or float(
                projection.get("support_foot_pose_max_abs_delta", math.inf)
            )
            > 1.0e-12
            or not isinstance(changed_indices, list)
            or any(
                type(index) is not int or index not in expected_nonleg
                for index in changed_indices
            )
            or changed_names
            != [joint_names[index] for index in changed_indices]
            or any(
                not close(seed_delta[index], 0.0)
                for index in expected_leg
            )
            or any(
                (index in changed_indices)
                != (not close(seed_delta[index], 0.0))
                for index in range(31)
            )
        ):
            raise TableSmokeReceiptError(
                "contact-free projected physical-birth composition is invalid"
            )
    if (
        any(
            not close(physical_q[index], teacher_q[index] + delta_q[index])
            for index in range(31)
        )
        or (
            composition_semantics == MEASURED_BIRTH_SHARED_LOWER_SEMANTICS
            and any(
                not close(delta_q[index], 0.0) for index in expected_nonleg
            )
        )
        or any(
            not close(
                physical_root[index], teacher_root[index] + delta_root[index]
            )
            for index in range(3)
        )
        or any(
            not close(teacher_quat[index], recorded_teacher_quat[index])
            or not close(physical_quat[index], recorded_physical_quat[index])
            for index in range(4)
        )
        or not (
            any(not close(value, 0.0) for value in delta_q)
            or any(not close(value, 0.0) for value in delta_root)
            or any(
                not close(teacher_quat[index], physical_quat[index])
                for index in range(4)
            )
        )
    ):
        raise TableSmokeReceiptError(
            "dynamic-ready physical birth differs from recorded teacher delta"
        )
    return teacher_root, teacher_quat, teacher_q, True


def _load_nominal_hold_input(
    artifact_value: str | Path, *, expected_sha256: str
) -> _NominalHoldInput:
    artifact_path, artifact_sha = _pinned_external_file(
        artifact_value, expected_sha256, "dynamic-ready artifact"
    )
    payload = artifact_path.read_bytes()
    document = _strict_json_object(payload, "dynamic-ready artifact")
    if document.get("kind") != NOMINAL_HOLD_ARTIFACT_KIND:
        raise TableSmokeReceiptError("wrong dynamic-ready artifact kind")
    unsigned = dict(document)
    content_sha = _require_sha256(
        unsigned.pop("content_sha256", None), "artifact content SHA-256"
    )
    if (
        hashlib.sha256(_canonical_ascii_json_bytes(unsigned)).hexdigest()
        != content_sha
    ):
        raise TableSmokeReceiptError("dynamic-ready content SHA-256 mismatch")
    try:
        robot = document["robot"]
        physical = document["physical_ready"]
        runtime = document["runtime_plant"]
        hold = document["hold_candidate"]
        motion = document["sources"]["stable_motion"]
        action_id = document["action_id"]
    except (KeyError, TypeError) as exc:
        raise TableSmokeReceiptError("dynamic-ready core fields missing") from exc
    names = tuple(robot["joint_names"])
    authorization = document.get("authorization")
    if (
        document.get("schema_version") != 2
        or robot.get("family") != "AgiBot A3"
        or len(names) != 31
        or len(set(names)) != 31
        or not isinstance(action_id, str)
        or not action_id
        or not isinstance(authorization, Mapping)
        or any(
            authorization.get(key) is not False
            for key in (
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
                "isaac_nominal_hold_validated",
            )
        )
    ):
        raise TableSmokeReceiptError("dynamic-ready is not exact A3 N=1")
    motion_path, motion_sha = _pinned_external_file(
        motion["path"], motion["sha256"], "stable motion"
    )
    seed_source = document.get("sources", {}).get("physical_birth_seed")
    if seed_source is not None:
        if not isinstance(seed_source, Mapping):
            raise TableSmokeReceiptError(
                "dynamic-ready physical-birth seed source is invalid"
            )
        seed_path, _seed_sha = _pinned_external_file(
            seed_source.get("path"),
            seed_source.get("sha256"),
            "physical-birth numerical seed",
        )
        seed_document = _strict_json_object(
            seed_path.read_bytes(), "physical-birth numerical seed"
        )
        if seed_document.get("content_sha256") != seed_source.get(
            "content_sha256"
        ):
            raise TableSmokeReceiptError(
                "physical-birth numerical seed content identity mismatch"
            )
    vectors = {
        out: _finite_tuple(runtime[source], 31, source)
        for out, source in (
            ("joint_stiffness", "joint_stiffness"),
            ("joint_damping", "joint_damping"),
            ("joint_effort_limits", "joint_effort_limits"),
            ("joint_velocity_limits", "joint_velocity_limits"),
            ("joint_armature", "joint_armature"),
            ("default_joint_pos", "default_joint_pos_rad"),
            ("action_scale", "action_scale_rad"),
        )
    }
    limits = tuple(
        _finite_tuple(pair, 2, "q_des limit")
        for pair in runtime["qdes_joint_pos_limits"]
    )
    if len(limits) != 31:
        raise TableSmokeReceiptError("q_des limits must be [31,2]")
    delay = runtime["control_step_action_delay"]
    if (
        type(delay) is not dict
        or set(delay)
        != {
            "schema_version",
            "enabled",
            "semantic_unit",
            "sample_timing",
            "distribution",
            "min_steps",
            "max_steps",
            "shared_across_all_31_joints",
            "history_fill",
        }
        or delay["schema_version"] != 1
        or delay["semantic_unit"] != "policy_control_step"
        or delay["sample_timing"] != "once_per_episode_reset"
        or delay["distribution"] != "discrete_uniform_inclusive"
        or type(delay["enabled"]) is not bool
        or isinstance(delay["min_steps"], bool)
        or type(delay["min_steps"]) is not int
        or isinstance(delay["max_steps"], bool)
        or type(delay["max_steps"]) is not int
        or delay["min_steps"] < 0
        or delay["max_steps"] < delay["min_steps"]
        or delay["enabled"] != (delay["max_steps"] > 0)
        or delay["shared_across_all_31_joints"] is not True
        or delay["history_fill"]
        != "safe_default_or_action_specific_hold"
    ):
        raise TableSmokeReceiptError(
            "dynamic-ready action-delay contract is invalid"
        )
    expected_plant = {
        "joint_names": names,
        "articulation_joint_names": tuple(
            runtime["articulation_joint_names"]
        ),
        "action_joint_ids": tuple(runtime["action_joint_ids"]),
        **vectors,
        "qdes_joint_pos_limits": limits,
        "finite_projection_soft_envelope_inset_fraction": float(
            runtime["finite_projection_soft_envelope_inset_fraction"]
        ),
        "physics_step_dt_s": float(runtime["physics_step_dt_s"]),
        "policy_step_dt_s": float(runtime["policy_step_dt_s"]),
        "control_decimation": int(runtime["control_decimation"]),
        "control_step_action_delay": delay,
        **(
            {
                "physx_control_position_limits": (
                    _nominal_hold_physx_control_contract(
                        runtime["physx_control_position_limits"],
                        joint_names=names,
                    )
                )
            }
            if "physx_control_position_limits" in runtime
            else {}
        ),
    }
    frame0_semantics = document.get("physical_birth_composition", {}).get(
        "semantics"
    )
    if "physx_control_position_limits" not in expected_plant:
        if frame0_semantics == MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS:
            raise TableSmokeReceiptError(
                "direct measured frame0 hold requires exact Vendor PhysX H_ctrl"
            )
        if (
            frame0_semantics
            == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS
        ):
            raise TableSmokeReceiptError(
                "threshold-first whole-body frame0 hold requires exact Vendor PhysX H_ctrl"
            )
    else:
        control_limits = expected_plant["physx_control_position_limits"][
            "control_joint_pos_limits"
        ]
        if any(
            not control_lower <= qdes_lower < qdes_upper <= control_upper
            for (control_lower, control_upper), (qdes_lower, qdes_upper) in zip(
                control_limits, limits
            )
        ):
            raise TableSmokeReceiptError(
                "nominal hold qdes envelope must remain inside exact Vendor PhysX H_ctrl"
            )
    root_quat = _finite_tuple(
        physical["root_quat_wxyz"], 4, "root quaternion"
    )
    (
        teacher_root_pos,
        teacher_root_quat,
        teacher_joint_pos,
        teacher_physical_separated,
    ) = _nominal_teacher_physical_contract(
        document,
        joint_names=names,
        motion_sha256=motion_sha,
        physical=physical,
    )
    hold_qdes = _finite_tuple(
        hold["hold_qdes_joint_pos_rad"], 31, "hold q_des"
    )
    hold_action = _finite_tuple(
        hold["normalized_actor_action"], 31, "normalized hold action"
    )
    inset = expected_plant[
        "finite_projection_soft_envelope_inset_fraction"
    ]
    if (
        not math.isclose(
            sum(value * value for value in root_quat),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-6,
        )
        or any(
            not lo + inset * (hi - lo) < qdes < hi - inset * (hi - lo)
            for qdes, (lo, hi) in zip(hold_qdes, limits)
        )
        or any(
            not math.isclose(
                default + scale * action,
                qdes,
                rel_tol=0.0,
                abs_tol=2.0e-7,
            )
            for default, scale, action, qdes in zip(
                vectors["default_joint_pos"],
                vectors["action_scale"],
                hold_action,
                hold_qdes,
            )
        )
    ):
        raise TableSmokeReceiptError(
            "dynamic-ready quaternion/q_des envelope is invalid"
        )
    return _NominalHoldInput(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        document=document,
        action_id=action_id,
        joint_names=names,
        motion_path=motion_path,
        motion_sha256=motion_sha,
        teacher_root_pos=teacher_root_pos,
        teacher_root_quat=teacher_root_quat,
        teacher_joint_pos=teacher_joint_pos,
        teacher_physical_separated=teacher_physical_separated,
        physical_root_pos=_finite_tuple(
            physical["root_pos_w_m"], 3, "root position"
        ),
        physical_root_quat=root_quat,
        physical_joint_pos=_finite_tuple(
            physical["joint_pos_rad"], 31, "ready q"
        ),
        hold_qdes=hold_qdes,
        hold_action=hold_action,
        expected_plant=expected_plant,
    )


def _load_profile_contract(
    profile_pins_value: str | Path,
    *,
    expected_profile_pins_sha256: str,
    manifest_document: Mapping[str, Any],
    repo_root: Path,
) -> tuple[
    _FileSnapshot,
    str,
    str,
    tuple[tuple[str, _FileSnapshot], ...],
    Mapping[str, Any],
]:
    """Reopen the exact solver/physics/geometry bytes used by fresh N5.

    Merely copying the two payload digests out of the manifest would leave the
    Isaac receipt detached from the profile-pins file and implementation
    sources that produced those digests.  Formal mode therefore snapshots the
    preregistered profile file, recomputes both canonical payload digests, and
    reopens the exact five solver sources (including physical racket geometry).
    """

    expected_profile_sha = _require_sha256(
        expected_profile_pins_sha256,
        "--profile-pins-sha256",
    )
    profile_path, _ = _normalized_repo_path(
        profile_pins_value,
        repo_root=repo_root,
        label="ActionBall profile pins",
    )
    profile_snapshot = _read_snapshot(
        profile_path,
        repo_root=repo_root,
        label="ActionBall profile pins",
    )
    if profile_snapshot.sha256 != expected_profile_sha:
        raise TableSmokeReceiptError(
            "ActionBall profile-pins bytes differ from the preregistered "
            "SHA-256"
        )
    profile = _strict_json_object(
        profile_snapshot.payload, "ActionBall profile pins"
    )
    solver_payload = profile.get("solver_payload")
    physics_payload = profile.get("physics_payload")
    if not isinstance(solver_payload, Mapping) or not isinstance(
        physics_payload, Mapping
    ):
        raise TableSmokeReceiptError(
            "ActionBall profile pins omit solver_payload/physics_payload"
        )
    solver_profile_sha = hashlib.sha256(
        _canonical_json_bytes(solver_payload)
    ).hexdigest()
    physics_profile_sha = hashlib.sha256(
        _canonical_json_bytes(physics_payload)
    ).hexdigest()
    solver_contact_geometry = solver_payload.get("contact_geometry")
    profile_contact_geometry = profile.get("contact_geometry")
    if (
        not isinstance(solver_contact_geometry, Mapping)
        or set(solver_contact_geometry) != {"payload", "sha256"}
        or not isinstance(solver_contact_geometry.get("payload"), Mapping)
        or not isinstance(profile_contact_geometry, Mapping)
        or dict(profile_contact_geometry) != dict(solver_contact_geometry)
    ):
        raise TableSmokeReceiptError(
            "ActionBall profile pins omit the exact canonical contact "
            "geometry payload"
        )
    contact_geometry_sha = _require_sha256(
        solver_contact_geometry["sha256"],
        "solver_payload.contact_geometry.sha256",
    )
    if hashlib.sha256(
        _canonical_json_bytes(solver_contact_geometry["payload"])
    ).hexdigest() != contact_geometry_sha:
        raise TableSmokeReceiptError(
            "ActionBall canonical contact geometry payload seal is false"
        )
    manifest_solver_sha = _require_sha256(
        manifest_document.get("solver_profile_sha256"),
        "manifest.solver_profile_sha256",
    )
    manifest_physics_sha = _require_sha256(
        manifest_document.get("physics_profile_sha256"),
        "manifest.physics_profile_sha256",
    )
    if (
        profile.get("solver_profile_sha256") != solver_profile_sha
        or profile.get("physics_profile_sha256") != physics_profile_sha
        or manifest_solver_sha != solver_profile_sha
        or manifest_physics_sha != physics_profile_sha
    ):
        raise TableSmokeReceiptError(
            "manifest/profile solver or physics payload SHA does not close"
        )
    source_map = profile.get("solver_implementation_source_sha256")
    # Solver profile v3: the sealed payload binds the per-symbol semantic
    # surface, not the byte map.  The byte map stays in the document as the
    # provenance record and is still re-verified against the checkout below.
    payload_surface = solver_payload.get("semantic_surface")
    document_surface = profile.get("solver_semantic_surface")
    if (
        not isinstance(source_map, Mapping)
        or not isinstance(payload_surface, Mapping)
        or not isinstance(document_surface, Mapping)
        or payload_surface.get("sha256") != document_surface.get("sha256")
        or tuple(sorted(source_map))
        != tuple(sorted(_ACTION_BALL_SOLVER_SOURCE_NAMES))
    ):
        raise TableSmokeReceiptError(
            "ActionBall profile pins must bind the exact five solver sources "
            "and close their solver payload against the document's semantic "
            "surface"
        )
    solver_sources: list[tuple[str, _FileSnapshot]] = []
    for name in _ACTION_BALL_SOLVER_SOURCE_NAMES:
        expected_source_sha = _require_sha256(
            source_map[name], f"solver source {name!r} SHA-256"
        )
        source_path = repo_root.joinpath(
            *_ACTION_BALL_SOLVER_SOURCE_DIR.parts, name
        )
        source_snapshot = _read_snapshot(
            source_path,
            repo_root=repo_root,
            label=f"solver source {name!r}",
        )
        if source_snapshot.sha256 != expected_source_sha:
            raise TableSmokeReceiptError(
                f"solver source {name!r} bytes differ from profile pins"
            )
        solver_sources.append((name, source_snapshot))

    expected_geometry_path = (
        _ACTION_BALL_SOLVER_SOURCE_DIR / "racket_contact_geometry.py"
    ).as_posix()
    geometry_snapshot = next(
        snapshot
        for name, snapshot in solver_sources
        if name == "racket_contact_geometry.py"
    )
    if contact_geometry_sha == geometry_snapshot.sha256:
        # The canonical geometry-payload SHA and Python-source SHA are
        # deliberately distinct identities; accepting their accidental
        # substitution would erase the semantic-vs-byte distinction.
        raise TableSmokeReceiptError(
            "profile racket geometry source/payload identities do not close"
        )
    geometry = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": expected_geometry_path,
        "source_sha256": geometry_snapshot.sha256,
        "geometry_source_sha256": contact_geometry_sha,
    }
    return (
        profile_snapshot,
        solver_profile_sha,
        physics_profile_sha,
        tuple(solver_sources),
        dict(geometry),
    )


def _load_trusted_action_set(profile_id: str) -> dict[str, Any]:
    try:
        return action_set_contract.load_contract_from_source(
            (
                Path(__file__).resolve().with_name(
                    "action_ball_action_set_contract.py"
                )
            ).read_bytes(),
            profile_id,
        )
    except Exception as exc:
        raise TableSmokeReceiptError(
            f"trusted action-set contract is unavailable: {exc}"
        ) from exc


def _load_formal_inputs(
    manifest_value: str | Path,
    *,
    action_set_profile: str,
    profile_pins_value: str | Path,
    expected_profile_pins_sha256: str,
    repo_root: Path,
    source_path: Path | None = None,
) -> _FormalInputs:
    root = _assert_plain_components(repo_root, "repository root")
    source = _read_snapshot(
        Path(__file__) if source_path is None else source_path,
        repo_root=root,
        label="producer source",
    )
    trusted_action_set = _load_trusted_action_set(action_set_profile)
    manifest_path, manifest_relative = _normalized_repo_path(
        manifest_value, repo_root=root, label="ActionBall manifest"
    )
    manifest = _read_snapshot(
        manifest_path, repo_root=root, label="ActionBall manifest"
    )
    document = _strict_json_object(manifest.payload, "ActionBall manifest")
    forbidden_top = {
        "racket_geometry_contract",
        "physical_contact_contract",
    }
    if forbidden_top.intersection(document):
        raise TableSmokeReceiptError(
            "strict training manifest contains gate-only physical fields"
        )
    raw_manifest_actions = document.get("actions")
    if isinstance(raw_manifest_actions, list) and any(
        isinstance(row, Mapping)
        and {
            "physical_ball_launch",
            "physical_task_binding",
            "admission",
        }.intersection(row)
        for row in raw_manifest_actions
    ):
        raise TableSmokeReceiptError(
            "strict training manifest action contains gate-only physical fields"
        )
    if manifest_relative != trusted_action_set["manifest_path"]:
        raise TableSmokeReceiptError(
            "manifest path differs from the trusted action-set contract"
        )
    try:
        action_set_contract.verify_manifest_identity(
            trusted_action_set, document, manifest.payload
        )
    except Exception as exc:
        raise TableSmokeReceiptError(
            f"manifest differs from the trusted action-set contract: {exc}"
        ) from exc
    action_order = tuple(trusted_action_set["ordered_action_ids"])
    expected_n = int(trusted_action_set["expected_n"])
    raw_actions = document.get("actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != expected_n:
        raise TableSmokeReceiptError(
            "formal table smoke manifest must contain exact N actions"
        )
    (
        profile_pins,
        solver_profile_sha256,
        physics_profile_sha256,
        solver_sources,
        racket_geometry_contract,
    ) = _load_profile_contract(
        profile_pins_value,
        expected_profile_pins_sha256=expected_profile_pins_sha256,
        manifest_document=document,
        repo_root=root,
    )
    motions: list[_MotionInput] = []
    for index, (motion_id, raw) in enumerate(
        zip(action_order, raw_actions)
    ):
        if not isinstance(raw, Mapping) or raw.get("action_id") != motion_id:
            raise TableSmokeReceiptError(
                f"manifest actions[{index}] does not match {motion_id!r}"
            )
        family = raw.get("family")
        if family not in ("forehand", "backhand"):
            raise TableSmokeReceiptError(
                f"manifest action {motion_id!r} has invalid family"
            )
        strike_phase = raw.get("strike_phase")
        cycle_s = raw.get("reference_t_cycle_s")
        sign = raw.get("mount_normal_sign")
        if (
            isinstance(strike_phase, bool)
            or type(strike_phase) not in (int, float)
            or not math.isfinite(float(strike_phase))
            or not 0.0 < float(strike_phase) < 1.0
            or isinstance(cycle_s, bool)
            or type(cycle_s) not in (int, float)
            or not math.isfinite(float(cycle_s))
            or float(cycle_s) <= 0.0
            or type(sign) is not int
            or sign not in (-1, 1)
        ):
            raise TableSmokeReceiptError(
                f"manifest action {motion_id!r} timing/face fields are invalid"
            )
        motion_path, _ = _normalized_repo_path(
            raw.get("motion_path", ""),
            repo_root=root,
            label=f"manifest action {motion_id!r} motion_path",
        )
        snapshot = _read_snapshot(
            motion_path,
            repo_root=root,
            label=f"motion {motion_id!r}",
        )
        expected_sha = _require_sha256(
            raw.get("motion_sha256"),
            f"manifest action {motion_id!r} motion_sha256",
        )
        if snapshot.sha256 != expected_sha:
            raise TableSmokeReceiptError(
                f"motion {motion_id!r} bytes differ from manifest SHA-256"
            )
        action_uid = raw.get("action_uid")
        expected_uid = _derive_action_uid(
            motion_id, str(family), expected_sha
        )
        if (
            type(action_uid) is not int
            or not 1 <= action_uid <= MAX_ACTION_UID
            or action_uid != expected_uid
        ):
            raise TableSmokeReceiptError(
                f"manifest action {motion_id!r} action_uid is not its "
                f"canonical action identity (expected {expected_uid})"
            )
        motions.append(
            _MotionInput(
                motion_id=motion_id,
                action_uid=action_uid,
                family=str(family),
                strike_phase=float(strike_phase),
                mount_normal_sign=int(sign),
                reference_t_cycle_s=float(cycle_s),
                file=snapshot,
            )
        )
    if (
        len({row.file.path for row in motions}) != expected_n
        or len({row.file.sha256 for row in motions}) != expected_n
        or len({row.action_uid for row in motions}) != expected_n
    ):
        raise TableSmokeReceiptError(
            "formal table smoke requires N distinct action UIDs, motion paths "
            "and bytes"
        )
    return _FormalInputs(
        repo_root=root,
        source=source,
        manifest=manifest,
        profile_pins=profile_pins,
        solver_profile_sha256=solver_profile_sha256,
        physics_profile_sha256=physics_profile_sha256,
        solver_sources=solver_sources,
        racket_geometry_contract=racket_geometry_contract,
        motions=tuple(motions),
        action_set_contract=dict(trusted_action_set),
    )


def _repository_root_from_producer(
    source_path: Path | None = None,
) -> Path:
    """Resolve the repository root from the producer's fixed tracked path.

    ``Path.parents`` is easy to get subtly wrong here: this script is four
    directories below the repository root, not three.  Formal evidence must not
    accidentally treat ``hope_training/`` as the root because that would make
    every repository-relative manifest path and source binding refer to a
    different namespace than the admission consumer.
    """

    source = _assert_plain_components(
        Path(__file__) if source_path is None else source_path,
        "producer source",
    )
    parts = PurePosixPath(FORMAL_PRODUCER_REPO_PATH).parts
    if len(source.parts) < len(parts) or tuple(source.parts[-len(parts):]) != parts:
        raise TableSmokeReceiptError(
            "formal producer is not running from its exact tracked repository path "
            f"{FORMAL_PRODUCER_REPO_PATH!r}"
        )
    root = source
    for _ in parts:
        root = root.parent
    if source.relative_to(root).as_posix() != FORMAL_PRODUCER_REPO_PATH:
        raise TableSmokeReceiptError(
            "formal producer repository-root derivation is inconsistent"
        )
    return _assert_plain_components(root, "repository root")


def _assert_formal_inputs_unchanged(inputs: _FormalInputs) -> None:
    for label, snapshot in (
        ("producer source", inputs.source),
        ("ActionBall manifest", inputs.manifest),
        ("ActionBall profile pins", inputs.profile_pins),
        *(
            (f"solver source {name!r}", snapshot)
            for name, snapshot in inputs.solver_sources
        ),
        *(
            (f"motion {row.motion_id!r}", row.file)
            for row in inputs.motions
        ),
    ):
        current = _read_snapshot(
            snapshot.path, repo_root=inputs.repo_root, label=label
        )
        if (
            current.sha256 != snapshot.sha256
            or current.device != snapshot.device
            or current.inode != snapshot.inode
            or current.size != snapshot.size
        ):
            raise TableSmokeReceiptError(
                f"{label} inode or bytes changed during the PhysX run"
            )


def _committed_source_identity(inputs: _FormalInputs) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=inputs.repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        tracked_dirty = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=inputs.repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        )
        committed_source = subprocess.check_output(
            ["git", "show", f"HEAD:{inputs.source.repo_path}"],
            cwd=inputs.repo_root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TableSmokeReceiptError(
            f"cannot bind producer to an exact Git source commit: {exc}"
        ) from exc
    if _GIT_SHA_RE.fullmatch(commit) is None:
        raise TableSmokeReceiptError("producer Git HEAD is not one full SHA-1")
    if tracked_dirty.strip():
        raise TableSmokeReceiptError(
            "formal table smoke requires an exact clean checkout, including "
            "no non-ignored untracked files"
        )
    if committed_source != inputs.source.payload:
        raise TableSmokeReceiptError(
            "producer source bytes differ from the bound Git commit"
        )
    return commit


def _runtime_module_source_path(module_name: str, module: Any) -> Path:
    """Return one import's source path; reject bytecode/zip/dynamic origins."""

    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        raise TableSmokeReceiptError(
            f"runtime module {module_name!r} has no plain source file"
        )
    path = Path(raw)
    if path.suffix in (".pyc", ".pyo"):
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except (ValueError, NotImplementedError) as exc:
            raise TableSmokeReceiptError(
                f"runtime module {module_name!r} has no recoverable Python source"
            ) from exc
    if path.suffix != ".py":
        raise TableSmokeReceiptError(
            f"runtime module {module_name!r} is not loaded from tracked Python source"
        )
    return path


def _assert_runtime_source_closure(
    inputs: _FormalInputs,
    source_commit_sha: str,
    *,
    baseline: Mapping[str, _FileSnapshot] | None = None,
    required_modules: Sequence[str] = _REQUIRED_RUNTIME_MODULES,
) -> dict[str, _FileSnapshot]:
    """Bind every loaded project module to the same exact committed checkout.

    A clean checkout alone is insufficient when an editable install or a stale
    site-package copy wins import resolution.  Formal evidence therefore closes
    every currently loaded ``whole_body_tracking`` module over repository path,
    source bytes, and the exact Git commit, then repeats the closure after all
    dynamic imports and physics steps.
    """

    if _GIT_SHA_RE.fullmatch(source_commit_sha) is None:
        raise TableSmokeReceiptError(
            "runtime source closure received a malformed Git commit"
        )
    current: dict[str, _FileSnapshot] = {}
    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name == _RUNTIME_MODULE_PREFIX
            or module_name.startswith(_RUNTIME_MODULE_PREFIX + ".")
        ):
            continue
        if module is None:
            raise TableSmokeReceiptError(
                f"runtime module {module_name!r} is an unresolved import sentinel"
            )
        source_path = _runtime_module_source_path(module_name, module)
        snapshot = _read_snapshot(
            source_path,
            repo_root=inputs.repo_root,
            label=f"runtime module {module_name!r}",
        )
        try:
            committed = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{source_commit_sha}:{snapshot.repo_path}",
                ],
                cwd=inputs.repo_root,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TableSmokeReceiptError(
                f"runtime module {module_name!r} is not tracked by "
                f"source commit {source_commit_sha}: {exc}"
            ) from exc
        if committed != snapshot.payload:
            raise TableSmokeReceiptError(
                f"runtime module {module_name!r} bytes differ from "
                f"source commit {source_commit_sha}"
            )
        current[module_name] = snapshot

    missing = tuple(name for name in required_modules if name not in current)
    if missing:
        raise TableSmokeReceiptError(
            "formal runtime source closure is missing required module(s): "
            + ", ".join(missing)
        )
    if baseline is not None:
        for module_name, old in baseline.items():
            new = current.get(module_name)
            if (
                new is None
                or new.path != old.path
                or new.sha256 != old.sha256
                or new.device != old.device
                or new.inode != old.inode
                or new.size != old.size
            ):
                raise TableSmokeReceiptError(
                    f"runtime module {module_name!r} changed during the PhysX run"
                )
    return current


def _validate_cli_mode(args) -> None:
    if args.task == RETIRED_FAKE_ACTION_BALL_TASK_ID:
        raise TableSmokeReceiptError(
            f"retired fake task id {args.task!r} is forbidden; use "
            f"{ACTION_BALL_TASK_ID!r}"
        )
    formal = args.receipt_out is not None
    nominal_hold = args.nominal_hold is not None
    if formal and nominal_hold:
        raise TableSmokeReceiptError(
            "--receipt-out and --nominal-hold are mutually exclusive modes"
        )
    if formal:
        failures = []
        if args.task != ACTION_BALL_TASK_ID:
            failures.append(
                f"--task must equal {ACTION_BALL_TASK_ID!r}"
            )
        if not args.manifest:
            failures.append("--manifest is required")
        if not args.action_set_profile:
            failures.append("--action-set-profile is required")
        if not args.profile_pins:
            failures.append("--profile-pins is required")
        if not args.profile_pins_sha256:
            failures.append("--profile-pins-sha256 is required")
        elif _SHA256_RE.fullmatch(str(args.profile_pins_sha256)) is None:
            failures.append(
                "--profile-pins-sha256 must be one lowercase SHA-256"
            )
        if args.num_envs != 1:
            failures.append("--num-envs must equal 1")
        if args.device != "cuda:0":
            failures.append("--device must equal cuda:0")
        if args.cfg_only:
            failures.append("--cfg-only is forbidden")
        if args.bench:
            failures.append("--bench is forbidden")
        if args.table_obstacle != "on":
            failures.append("--table-obstacle must be on")
        if args.motion_file is not None:
            failures.append("--motion-file is manifest-owned in formal mode")
        if args.nominal_hold_sha256 is not None:
            failures.append("--nominal-hold-sha256 is forbidden")
        if args.nominal_hold_receipt_out is not None:
            failures.append("--nominal-hold-receipt-out is forbidden")
        if args.screenshot_dir is not None:
            failures.append("--screenshot-dir is forbidden")
        if failures:
            raise TableSmokeReceiptError(
                "invalid formal receipt mode: " + "; ".join(failures)
            )
    elif nominal_hold:
        formal_args = (
            args.manifest,
            args.action_set_profile,
            args.profile_pins,
            args.profile_pins_sha256,
        )
        invalid_shape = (
            args.task != ACTION_BALL_TASK_ID
            or args.num_envs != 1
            or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", str(args.device)) is None
            or args.cfg_only
            or args.bench
            or args.contact_smoke
            or args.table_obstacle != "on"
            or args.motion_file is not None
            or any(value is not None for value in formal_args)
        )
        if invalid_shape:
            raise TableSmokeReceiptError(
                "nominal hold requires one explicit cuda:N ActionBall table scene "
                "without formal, cfg, bench, contact-smoke or motion overrides"
            )
        if (
            not args.nominal_hold_sha256
            or _SHA256_RE.fullmatch(str(args.nominal_hold_sha256)) is None
            or not args.nominal_hold_receipt_out
            or not 0.0 < float(args.duration_s) <= 30.0
        ):
            raise TableSmokeReceiptError(
                "nominal hold requires artifact SHA/output and a finite duration"
            )
    elif any(
        value is not None
        for value in (
            args.manifest,
            args.action_set_profile,
            args.profile_pins,
            args.profile_pins_sha256,
        )
    ):
        raise TableSmokeReceiptError(
            "--action-set-profile/--manifest/--profile-pins are accepted only together with "
            "--receipt-out"
        )
    elif (
        args.nominal_hold_sha256 is not None
        or args.nominal_hold_receipt_out is not None
        or args.screenshot_dir is not None
    ):
        raise TableSmokeReceiptError(
            "--nominal-hold-sha256/--nominal-hold-receipt-out/"
            "--screenshot-dir require --nominal-hold"
        )


def _initialize_isaac_runtime(args) -> None:
    global _ISAAC_RUNTIME_ORIGIN, _app_launcher, _app
    global gym, torch, parse_env_cfg, tt_frame
    global TABLE_COMPONENT_ROLES
    global TABLE_CONTACT_BODY_NAMES, TABLE_HIT_FORCE_THRESHOLD_N

    from isaaclab.app import AppLauncher

    launcher_args = {"headless": True, "device": args.device}
    if args.screenshot_dir is not None:
        launcher_args["enable_cameras"] = True
    # Keep the launcher alive for the whole diagnostic, matching the shipped
    # train/play entrypoints.  The launcher owns Kit lifecycle callbacks in
    # addition to exposing the SimulationApp object.
    _app_launcher = AppLauncher(launcher_args)
    _app = _app_launcher.app
    import gymnasium as gym_module
    import torch as torch_module
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg as parse_env_cfg_function
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.table_tennis import table_frame as frame_module
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg

    gym = gym_module
    torch = torch_module
    parse_env_cfg = parse_env_cfg_function
    tt_frame = frame_module
    TABLE_COMPONENT_ROLES = tuple(frame_module.TABLE_ASSEMBLY_ROLES)
    TABLE_CONTACT_BODY_NAMES = tuple(hope_env_cfg.TABLE_CONTACT_BODY_NAMES)
    TABLE_HIT_FORCE_THRESHOLD_N = float(
        hope_env_cfg.TABLE_HIT_FORCE_THRESHOLD_N
    )
    _ISAAC_RUNTIME_ORIGIN = object()


def _gpu_identity() -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        raise TableSmokeReceiptError(
            "formal table smoke requires a live CUDA Isaac runtime"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or re.fullmatch(r"0|[1-9][0-9]*", visible) is None:
        raise TableSmokeReceiptError(
            "CUDA_VISIBLE_DEVICES must contain exactly one physical GPU index"
        )
    physical = int(visible)
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise TableSmokeReceiptError(
            "formal table smoke requires one visible GPU at logical cuda:0"
        )
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        raise TableSmokeReceiptError(
            f"NVML GPU identity verification failed: {exc}"
        ) from exc

    def text_value(value: Any) -> str:
        return (
            value.decode("utf-8")
            if isinstance(value, (bytes, bytearray))
            else str(value)
        )

    uuid_text = text_value(uuid)
    name_text = text_value(name)
    driver_text = text_value(driver)
    if (
        not uuid_text.startswith("GPU-")
        or not name_text
        or not driver_text
        or torch.cuda.get_device_name(0) != name_text
    ):
        raise TableSmokeReceiptError(
            "CUDA logical device and NVML physical GPU identity do not close"
        )
    return {
        "physical_index": physical,
        "logical_index": 0,
        "cuda_visible_devices": visible,
        "gpu_uuid": uuid_text,
        "gpu_name": name_text,
        "driver_version": driver_text,
        "nvml_verified": True,
    }


def _isaac_version_identity() -> str:
    versions: list[str] = []
    for distribution in ("isaaclab", "isaacsim"):
        try:
            versions.append(
                f"{distribution}={importlib.metadata.version(distribution)}"
            )
        except importlib.metadata.PackageNotFoundError:
            continue
    if not versions:
        try:
            import isaaclab

            value = str(getattr(isaaclab, "__version__", "") or "")
        except Exception as exc:
            raise TableSmokeReceiptError(
                f"cannot resolve Isaac runtime version: {exc}"
            ) from exc
        if value:
            versions.append(f"isaaclab={value}")
    if not versions:
        raise TableSmokeReceiptError("Isaac runtime version identity is empty")
    return ";".join(versions)


def _validate_runtime_evidence(
    evidence: _RuntimeEvidence,
    inputs: _FormalInputs,
) -> None:
    if (
        _ISAAC_RUNTIME_ORIGIN is None
        or not isinstance(evidence, _RuntimeEvidence)
        or evidence.origin is not _ISAAC_RUNTIME_ORIGIN
    ):
        raise TableSmokeReceiptError(
            "PASS receipt requires evidence created by this live Isaac runtime"
        )
    if _GIT_SHA_RE.fullmatch(evidence.source_commit_sha) is None:
        raise TableSmokeReceiptError("runtime source commit is malformed")
    if (
        not evidence.isaac_version
        or evidence.python_executable != sys.executable
        or type(evidence.physics_steps) is not int
        or evidence.physics_steps <= 0
    ):
        raise TableSmokeReceiptError("runtime identity/physics step count is invalid")
    expected_gpu_keys = {
        "physical_index",
        "logical_index",
        "cuda_visible_devices",
        "gpu_uuid",
        "gpu_name",
        "driver_version",
        "nvml_verified",
    }
    gpu = evidence.gpu_identity
    if (
        not isinstance(gpu, Mapping)
        or set(gpu) != expected_gpu_keys
        or type(gpu["physical_index"]) is not int
        or gpu["physical_index"] < 0
        or gpu["logical_index"] != 0
        or gpu["cuda_visible_devices"] != str(gpu["physical_index"])
        or not isinstance(gpu["gpu_uuid"], str)
        or not gpu["gpu_uuid"].startswith("GPU-")
        or not isinstance(gpu["gpu_name"], str)
        or not gpu["gpu_name"]
        or not isinstance(gpu["driver_version"], str)
        or not gpu["driver_version"]
        or gpu["nvml_verified"] is not True
    ):
        raise TableSmokeReceiptError("runtime GPU identity is not exact NVML truth")
    if any(
        value is not True
        for value in (
            evidence.pose_obb_guard_pass,
            evidence.full_action_ball_assembly,
            evidence.all_five_table_components_with_pose_obb,
            evidence.all_five_obstacles,
            evidence.all_four_substeps,
            evidence.positive_control_pass,
            evidence.negative_control_pass,
            evidence.zero_reset_leakage,
        )
    ):
        raise TableSmokeReceiptError(
            "runtime did not prove every required table-smoke control"
        )
    expected_n = int(inputs.action_set_contract["expected_n"])
    if len(evidence.actions) != expected_n:
        raise TableSmokeReceiptError(
            "runtime evidence must contain exactly N action cycles"
        )
    action_physics_steps = 0
    for expected, actual in zip(inputs.motions, evidence.actions):
        if (
            actual.motion_id != expected.motion_id
            or actual.action_uid != expected.action_uid
            or actual.motion_sha256 != expected.file.sha256
            or type(actual.frame_count) is not int
            or actual.frame_count < 3
            or type(actual.physics_steps) is not int
            or actual.physics_steps != 4 * actual.frame_count
            or actual.complete_cycle is not True
            or actual.robot_body_contract_count != 32
        ):
            raise TableSmokeReceiptError(
                f"runtime action {expected.motion_id!r} is incomplete or crossbound "
                "to different motion bytes"
            )
        for label, value in (
            ("table_contact_count", actual.table_contact_count),
            ("fall_count", actual.fall_count),
            ("hard_limit_count", actual.hard_limit_count),
            ("unsafe_count", actual.unsafe_count),
        ):
            if type(value) is not int or value != 0:
                raise TableSmokeReceiptError(
                    f"runtime action {expected.motion_id!r} {label} is not zero"
                )
        action_physics_steps += actual.physics_steps
    if evidence.physics_steps < action_physics_steps:
        raise TableSmokeReceiptError(
            "runtime physics_steps is smaller than the exact N action sweeps"
        )


def _build_formal_receipt(
    inputs: _FormalInputs,
    evidence: _RuntimeEvidence,
) -> dict[str, Any]:
    _validate_runtime_evidence(evidence, inputs)
    receipt: dict[str, Any] = {
        "schema_version": 4,
        "receipt_class": FORMAL_RECEIPT_CLASS,
        "verdict": "PASS",
        "task_id": ACTION_BALL_TASK_ID,
        "with_table": True,
        "scope": inputs.action_set_contract["scope"],
        "mobility_mode": inputs.action_set_contract["mobility_mode"],
        "ordered_action_ids": list(
            inputs.action_set_contract["ordered_action_ids"]
        ),
        "action_set_contract": dict(inputs.action_set_contract),
        "motion_sha256": [
            motion.file.sha256 for motion in inputs.motions
        ],
        "manifest": {
            "path": inputs.manifest.repo_path,
            "sha256": inputs.manifest.sha256,
        },
        "profile_contract": {
            "profile_pins": {
                "path": inputs.profile_pins.repo_path,
                "sha256": inputs.profile_pins.sha256,
            },
            "solver_profile_sha256": inputs.solver_profile_sha256,
            "physics_profile_sha256": inputs.physics_profile_sha256,
            "solver_implementation_sources": [
                {
                    "name": name,
                    "path": snapshot.repo_path,
                    "sha256": snapshot.sha256,
                }
                for name, snapshot in inputs.solver_sources
            ],
            "racket_geometry_contract": dict(
                inputs.racket_geometry_contract
            ),
        },
        "runtime_contract": {
            "source_commit_sha": evidence.source_commit_sha,
            "isaac_version": evidence.isaac_version,
            "python_executable": evidence.python_executable,
            "runtime_source": {
                "path": inputs.source.repo_path,
                "sha256": inputs.source.sha256,
            },
            "gpu_identity": dict(evidence.gpu_identity),
            "physics_steps": evidence.physics_steps,
            "pose_obb_guard_pass": evidence.pose_obb_guard_pass,
            "full_action_ball_assembly": evidence.full_action_ball_assembly,
            "all_five_table_components_with_pose_obb": (
                evidence.all_five_table_components_with_pose_obb
            ),
            "action_robot_body_contract_rows": (
                32 * len(evidence.actions)
            ),
            "all_five_obstacles": evidence.all_five_obstacles,
            "all_four_substeps": evidence.all_four_substeps,
            "positive_control_pass": evidence.positive_control_pass,
            "negative_control_pass": evidence.negative_control_pass,
            "zero_reset_leakage": evidence.zero_reset_leakage,
        },
        "actions": [
            {
                "motion_id": row.motion_id,
                "action_uid": row.action_uid,
                "scope": inputs.action_set_contract["scope"],
                "robot_body_contract_count": (
                    row.robot_body_contract_count
                ),
                "motion_sha256": row.motion_sha256,
                "complete_cycle": row.complete_cycle,
                "isaac_pose_obb_pass": (
                    row.table_contact_count == 0
                    and row.fall_count == 0
                    and row.hard_limit_count == 0
                    and row.unsafe_count == 0
                ),
                "table_contact_count": row.table_contact_count,
                "fall_count": row.fall_count,
                "hard_limit_count": row.hard_limit_count,
                "unsafe_count": row.unsafe_count,
                "verdict": "PASS",
            }
            for row in evidence.actions
        ],
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "training_authorization",
            "deployment_authorization",
            "hardware_authorization",
        ],
    }
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _validate_formal_receipt_document(receipt, inputs=inputs)
    return receipt


def _validate_formal_receipt_document(
    receipt: Mapping[str, Any],
    *,
    inputs: _FormalInputs | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "receipt_class",
        "verdict",
        "task_id",
        "with_table",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "action_set_contract",
        "motion_sha256",
        "manifest",
        "profile_contract",
        "runtime_contract",
        "actions",
        "authorization",
        "non_claims",
        "receipt_payload_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise TableSmokeReceiptError("formal receipt top-level keys are not exact")
    if (
        receipt["schema_version"] != 4
        or receipt["receipt_class"] != FORMAL_RECEIPT_CLASS
        or receipt["verdict"] != "PASS"
        or receipt["task_id"] != ACTION_BALL_TASK_ID
        or receipt["with_table"] is not True
    ):
        raise TableSmokeReceiptError("formal receipt identity is not exact")
    if inputs is not None and (
        receipt["scope"] != inputs.action_set_contract["scope"]
        or receipt["mobility_mode"]
        != inputs.action_set_contract["mobility_mode"]
        or receipt["ordered_action_ids"]
        != list(inputs.action_set_contract["ordered_action_ids"])
        or receipt["action_set_contract"]
        != dict(inputs.action_set_contract)
    ):
        raise TableSmokeReceiptError(
            "formal receipt action-set identity is not exact"
        )
    _require_sha256(
        receipt["receipt_payload_sha256"], "receipt_payload_sha256"
    )
    unsigned = dict(receipt)
    seal = unsigned.pop("receipt_payload_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != seal:
        raise TableSmokeReceiptError("formal receipt payload seal is false")
    manifest = receipt["manifest"]
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"path", "sha256"}
    ):
        raise TableSmokeReceiptError("formal receipt manifest binding is malformed")
    _require_sha256(manifest["sha256"], "manifest.sha256")
    profile_contract = receipt["profile_contract"]
    if (
        not isinstance(profile_contract, Mapping)
        or set(profile_contract)
        != {
            "profile_pins",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "solver_implementation_sources",
            "racket_geometry_contract",
        }
    ):
        raise TableSmokeReceiptError(
            "formal receipt profile_contract keys are not exact"
        )
    profile_pins = profile_contract["profile_pins"]
    if (
        not isinstance(profile_pins, Mapping)
        or set(profile_pins) != {"path", "sha256"}
        or not isinstance(profile_pins["path"], str)
        or not profile_pins["path"]
    ):
        raise TableSmokeReceiptError(
            "formal receipt profile-pins binding is malformed"
        )
    _require_sha256(profile_pins["sha256"], "profile_pins.sha256")
    _require_sha256(
        profile_contract["solver_profile_sha256"],
        "profile_contract.solver_profile_sha256",
    )
    _require_sha256(
        profile_contract["physics_profile_sha256"],
        "profile_contract.physics_profile_sha256",
    )
    source_rows = profile_contract["solver_implementation_sources"]
    if (
        not isinstance(source_rows, list)
        or [row.get("name") for row in source_rows if isinstance(row, Mapping)]
        != list(_ACTION_BALL_SOLVER_SOURCE_NAMES)
    ):
        raise TableSmokeReceiptError(
            "formal receipt solver source rows are missing or reordered"
        )
    for index, row in enumerate(source_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "path", "sha256"}
            or not isinstance(row["path"], str)
            or not row["path"]
        ):
            raise TableSmokeReceiptError(
                f"formal receipt solver source row {index} is malformed"
            )
        _require_sha256(
            row["sha256"],
            f"profile_contract.solver_implementation_sources[{index}]",
        )
    geometry = profile_contract["racket_geometry_contract"]
    if (
        not isinstance(geometry, Mapping)
        or frozenset(geometry) != _RACKET_GEOMETRY_CONTRACT_KEYS
    ):
        raise TableSmokeReceiptError(
            "formal receipt racket geometry binding is malformed"
        )
    for key in ("source_sha256", "geometry_source_sha256"):
        _require_sha256(
            geometry[key],
            f"profile_contract.racket_geometry_contract.{key}",
        )
    motions = receipt["motion_sha256"]
    if (
        not isinstance(motions, list)
        or (
            inputs is not None
            and len(motions)
            != int(inputs.action_set_contract["expected_n"])
        )
        or any(_SHA256_RE.fullmatch(str(value)) is None for value in motions)
    ):
        raise TableSmokeReceiptError("formal receipt motion SHA list is malformed")
    if inputs is not None and (
        manifest
        != {
            "path": inputs.manifest.repo_path,
            "sha256": inputs.manifest.sha256,
        }
        or profile_contract
        != {
            "profile_pins": {
                "path": inputs.profile_pins.repo_path,
                "sha256": inputs.profile_pins.sha256,
            },
            "solver_profile_sha256": inputs.solver_profile_sha256,
            "physics_profile_sha256": inputs.physics_profile_sha256,
            "solver_implementation_sources": [
                {
                    "name": name,
                    "path": snapshot.repo_path,
                    "sha256": snapshot.sha256,
                }
                for name, snapshot in inputs.solver_sources
            ],
            "racket_geometry_contract": dict(
                inputs.racket_geometry_contract
            ),
        }
        or motions != [row.file.sha256 for row in inputs.motions]
    ):
        raise TableSmokeReceiptError(
            "formal receipt input bytes differ from the runtime snapshots"
        )
    runtime = receipt["runtime_contract"]
    expected_runtime_keys = {
        "source_commit_sha",
        "isaac_version",
        "python_executable",
        "runtime_source",
        "gpu_identity",
        "physics_steps",
        "pose_obb_guard_pass",
        "full_action_ball_assembly",
        "all_five_table_components_with_pose_obb",
        "action_robot_body_contract_rows",
        "all_five_obstacles",
        "all_four_substeps",
        "positive_control_pass",
        "negative_control_pass",
        "zero_reset_leakage",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_runtime_keys:
        raise TableSmokeReceiptError(
            "formal receipt runtime_contract keys are not exact"
        )
    source = runtime["runtime_source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"path", "sha256"}
        or not isinstance(source["path"], str)
        or not source["path"]
    ):
        raise TableSmokeReceiptError("formal runtime source binding is malformed")
    _require_sha256(source["sha256"], "runtime_source.sha256")
    gpu = runtime["gpu_identity"]
    expected_gpu_keys = {
        "physical_index",
        "logical_index",
        "cuda_visible_devices",
        "gpu_uuid",
        "gpu_name",
        "driver_version",
        "nvml_verified",
    }
    if (
        not isinstance(gpu, Mapping)
        or set(gpu) != expected_gpu_keys
        or type(gpu["physical_index"]) is not int
        or gpu["physical_index"] < 0
        or gpu["logical_index"] != 0
        or gpu["cuda_visible_devices"] != str(gpu["physical_index"])
        or not isinstance(gpu["gpu_uuid"], str)
        or not gpu["gpu_uuid"].startswith("GPU-")
        or not isinstance(gpu["gpu_name"], str)
        or not gpu["gpu_name"]
        or not isinstance(gpu["driver_version"], str)
        or not gpu["driver_version"]
        or gpu["nvml_verified"] is not True
    ):
        raise TableSmokeReceiptError("formal runtime GPU identity is malformed")
    if (
        not isinstance(runtime["source_commit_sha"], str)
        or _GIT_SHA_RE.fullmatch(runtime["source_commit_sha"]) is None
        or not isinstance(runtime["isaac_version"], str)
        or not runtime["isaac_version"]
        or not isinstance(runtime["python_executable"], str)
        or not runtime["python_executable"]
        or type(runtime["physics_steps"]) is not int
        or runtime["physics_steps"] < 1
        or type(runtime["action_robot_body_contract_rows"]) is not int
        or (
            inputs is not None
            and runtime["action_robot_body_contract_rows"]
            != 32 * int(inputs.action_set_contract["expected_n"])
        )
        or any(
            runtime[key] is not True
            for key in (
                "pose_obb_guard_pass",
                "full_action_ball_assembly",
                "all_five_table_components_with_pose_obb",
                "all_five_obstacles",
                "all_four_substeps",
                "positive_control_pass",
                "negative_control_pass",
                "zero_reset_leakage",
            )
        )
    ):
        raise TableSmokeReceiptError(
            "formal runtime contract does not describe an exact stepped PASS"
        )
    action_keys = {
        "motion_id",
        "action_uid",
        "scope",
        "robot_body_contract_count",
        "motion_sha256",
        "complete_cycle",
        "isaac_pose_obb_pass",
        "table_contact_count",
        "fall_count",
        "hard_limit_count",
        "unsafe_count",
        "verdict",
    }
    actions = receipt["actions"]
    expected_n = (
        int(inputs.action_set_contract["expected_n"])
        if inputs is not None
        else len(actions) if isinstance(actions, list) else -1
    )
    if not isinstance(actions, list) or len(actions) != expected_n:
        raise TableSmokeReceiptError(
            "formal receipt actions must contain exact N rows"
        )
    expected_order = (
        tuple(inputs.action_set_contract["ordered_action_ids"])
        if inputs is not None
        else tuple(receipt["ordered_action_ids"])
    )
    for index, (motion_id, motion_sha, action_row) in enumerate(
        zip(expected_order, motions, actions)
    ):
        action_uid = (
            action_row.get("action_uid")
            if isinstance(action_row, Mapping)
            else None
        )
        if (
            type(action_uid) is not int
            or not 1 <= action_uid <= MAX_ACTION_UID
        ):
            raise TableSmokeReceiptError(
                f"formal receipt actions[{index}].action_uid is malformed"
            )
        if (
            not isinstance(action_row, Mapping)
            or set(action_row) != action_keys
            or action_row
            != {
                "motion_id": motion_id,
                "action_uid": (
                    inputs.motions[index].action_uid
                    if inputs is not None
                    else action_uid
                ),
                "scope": receipt["scope"],
                "robot_body_contract_count": 32,
                "motion_sha256": motion_sha,
                "complete_cycle": True,
                "isaac_pose_obb_pass": True,
                "table_contact_count": 0,
                "fall_count": 0,
                "hard_limit_count": 0,
                "unsafe_count": 0,
                "verdict": "PASS",
            }
        ):
            raise TableSmokeReceiptError(
                f"formal receipt actions[{index}] is partial or unsafe"
            )
    authorization = receipt["authorization"]
    if authorization != {
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }:
        raise TableSmokeReceiptError("formal receipt self-authorization is forbidden")
    non_claims = receipt["non_claims"]
    if (
        not isinstance(non_claims, list)
        or not {
            "training_authorization",
            "deployment_authorization",
            "hardware_authorization",
        }.issubset(set(non_claims))
        or any(not isinstance(value, str) or not value for value in non_claims)
    ):
        raise TableSmokeReceiptError("formal receipt non-claims are incomplete")


def _prepare_output_path(
    value: str | Path,
    *,
    repo_root: Path,
) -> tuple[Path, str]:
    raw = str(value)
    pure = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or raw.endswith("/")
        or "\\" in raw
        or "//" in raw
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise TableSmokeReceiptError(
            "receipt output must be a normalized repository-relative path"
        )
    parent = _assert_plain_components(
        repo_root.joinpath(*pure.parts[:-1]), "receipt output parent"
    )
    if not parent.is_dir():
        raise TableSmokeReceiptError("receipt output parent is not a directory")
    output = parent / pure.name
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise TableSmokeReceiptError(
            f"cannot inspect receipt output path: {exc}"
        ) from exc
    else:
        raise FileExistsError(
            f"refusing to overwrite formal table-smoke receipt {output}"
        )
    return output, pure.as_posix()


def _exclusive_publish_receipt(
    output: Path,
    receipt: Mapping[str, Any],
) -> str:
    _validate_formal_receipt_document(receipt)
    payload = _canonical_json_bytes(receipt)
    parent = _assert_plain_components(output.parent, "receipt output parent")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = -1
    descriptor = -1
    created = False
    try:
        parent_descriptor = os.open(str(parent), parent_flags)
        descriptor = os.open(
            output.name,
            flags,
            0o444,
            dir_fd=parent_descriptor,
        )
        created = True
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise TableSmokeReceiptError(
                    "exclusive receipt write made no progress"
                )
            offset += written
        # ``open(mode=0o444)`` is filtered by the process umask.  The receipt
        # contract is exact 0444, so normalize it explicitly on the already
        # O_EXCL-created descriptor before durable readback.
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        readback = b"".join(chunks)
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_path_stat = os.stat(parent, follow_symlinks=False)
        parent_descriptor_stat = os.fstat(parent_descriptor)
        reparsed = _strict_json_object(readback, "published table-smoke receipt")
        if (
            readback != payload
            or _canonical_json_bytes(reparsed) != payload
            or not stat.S_ISREG(path_stat.st_mode)
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
            or stat.S_IMODE(path_stat.st_mode) != 0o444
            or parent_descriptor_stat.st_dev != parent_path_stat.st_dev
            or parent_descriptor_stat.st_ino != parent_path_stat.st_ino
        ):
            raise TableSmokeReceiptError(
                "formal receipt durable descriptor readback/identity failed"
            )
        _validate_formal_receipt_document(reparsed)
        os.fsync(parent_descriptor)
        return hashlib.sha256(readback).hexdigest()
    except Exception:
        if created:
            try:
                os.unlink(output.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _fresh_nominal_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    parent = _assert_plain_components(path.parent, f"{label} parent")
    path = parent / path.name
    if not path.name or path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to reuse {label} {path}")
    return path


def _exclusive_write_bytes(path: Path, payload: bytes) -> str:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _nominal_hold_render_png(env) -> bytes:
    try:
        frame = env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        import imageio.v3 as imageio

        return bytes(imageio.imwrite("<bytes>", frame, extension=".png"))
    except Exception as exc:
        raise TableSmokeReceiptError(
            f"cannot capture Isaac rgb_array screenshot: {exc}"
        ) from exc


def _refresh_nominal_hold_derived_state(unwrapped) -> None:
    """Refresh render/body state after a direct reset or simulator write."""

    sim = getattr(unwrapped, "sim", None)
    if sim is not None and callable(getattr(sim, "forward", None)):
        sim.forward()
    scene = getattr(unwrapped, "scene", None)
    if scene is not None and callable(getattr(scene, "update", None)):
        scene.update(0.0)


def _publish_nominal_hold_screenshot(
    directory: Path,
    filename: str,
    label: str,
    policy_step: int,
    payload: bytes,
) -> dict[str, Any]:
    path = directory / filename
    return {
        "label": label,
        "policy_step": policy_step,
        "path": str(path),
        "sha256": _exclusive_write_bytes(path, payload),
    }


def _exclusive_publish_nominal_hold_receipt(
    output: Path, receipt: Mapping[str, Any]
) -> str:
    return _exclusive_write_bytes(output, _canonical_json_bytes(receipt))


TOL = 1e-6
_results: dict = {}


def _fail(msg):
    print(f"FAIL {msg}", file=sys.stderr)
    raise SystemExit(1)


def _close(got, want, label):
    if len(got) != len(want) or any(abs(float(a) - float(b)) > TOL for a, b in zip(got, want)):
        _fail(f"{label}: {list(map(float, got))} != {list(map(float, want))}")


_FULL_PRIMS = (
    "{ENV_REGEX_NS}/TableObstacle",
    "{ENV_REGEX_NS}/TableRobotKeepout",
    "{ENV_REGEX_NS}/TableNet",
    "{ENV_REGEX_NS}/TableNetPostLeft",
    "{ENV_REGEX_NS}/TableNetPostRight",
)
_FULL_ATTRS = (
    "table_obstacle",
    "table_robot_keepout",
    "table_net",
    "table_net_post_left",
    "table_net_post_right",
)
_LEGACY_TOP_ATTR = {
    "{ENV_REGEX_NS}/TableObstacle": "table_obstacle",
    "{ENV_REGEX_NS}/ShadowTable": "shadow_table",
    "{ENV_REGEX_NS}/PhysicalTable": "pb_table",
}


def _center_size(bounds):
    lo, hi = bounds
    return (
        tuple((float(a) + float(b)) / 2.0 for a, b in zip(lo, hi)),
        tuple(float(b) - float(a) for a, b in zip(lo, hi)),
    )


def _component_specs(env_cfg):
    """Return the exact canonical role/prim/asset/pose contract for this cfg."""

    rt = env_cfg.commands.racket_target
    near_x, surface_z = float(rt.vb_table_near_x), float(rt.vb_table_surface_z)
    full = bool(getattr(env_cfg, "table_robot_keepout", False))
    if full:
        if getattr(env_cfg.scene, "shadow_ball", None) is not None or getattr(
            env_cfg.scene, "pb_ball", None
        ) is not None:
            _fail("ActionBall robot keep-out coexists with a dynamic/shadow ball")
        bounds = tt_frame.table_assembly_aabbs_env(
            near_x, surface_z, keepout_floor_z=0.0, margin=0.0
        )
        specs = []
        for role, prim, attr, box in zip(
            tt_frame.TABLE_ASSEMBLY_ROLES,
            _FULL_PRIMS,
            _FULL_ATTRS,
            bounds,
        ):
            pos, size = _center_size(box)
            specs.append(
                {
                    "role": role,
                    "prim": prim,
                    "attr": attr,
                    "pos": pos,
                    "size": size,
                }
            )
        return True, specs

    prim = getattr(env_cfg, "table_obstacle_prim", "")
    attr = _LEGACY_TOP_ATTR.get(prim)
    if attr is None:
        _fail(f"unknown legacy table_obstacle_prim {prim!r}")
    return False, [
        {
            "role": "top",
            "prim": prim,
            "attr": attr,
            "pos": tt_frame.table_top_center_env(near_x, surface_z),
            "size": tt_frame.table_top_size(),
        }
    ]


def check_cfg(env_cfg):
    """The cfg carries exact colliders, guard wiring, termination and penalty."""

    rt = env_cfg.commands.racket_target
    near_x, surface_z = float(rt.vb_table_near_x), float(rt.vb_table_surface_z)
    full, specs = _component_specs(env_cfg)
    expected_prims = tuple(spec["prim"] for spec in specs)
    if getattr(env_cfg, "table_obstacle_prim", "") != expected_prims[0]:
        _fail("table_obstacle_prim is not the canonical top collider")
    if tuple(getattr(env_cfg, "table_obstacle_prims", ())) != expected_prims:
        _fail(
            "table_obstacle_prims is not canonical: "
            f"{tuple(getattr(env_cfg, 'table_obstacle_prims', ()))!r} != {expected_prims!r}"
        )

    component_rows = []
    for spec in specs:
        asset = getattr(env_cfg.scene, spec["attr"], None)
        if asset is None:
            _fail(f"scene.{spec['attr']} is missing for {spec['role']}")
        collision = getattr(getattr(asset, "spawn", None), "collision_props", None)
        if collision is None or not bool(collision.collision_enabled):
            _fail(f"scene.{spec['attr']} has collision DISABLED")
        rigid = getattr(getattr(asset, "spawn", None), "rigid_props", None)
        if full and (
            rigid is None
            or getattr(rigid, "kinematic_enabled", None) is not True
            or getattr(rigid, "disable_gravity", None) is not True
            or getattr(asset.spawn, "activate_contact_sensors", None)
            is not False
        ):
            _fail(
                f"scene.{spec['attr']} is not a gravity-free kinematic table "
                "collider with ContactSensor reporting disabled"
            )
        _close(asset.init_state.pos, spec["pos"], f"scene.{spec['attr']}.init_state.pos")
        _close(asset.spawn.size, spec["size"], f"scene.{spec['attr']}.spawn.size")
        component_rows.append(
            {
                "role": spec["role"],
                "prim": spec["prim"],
                "scene_attr": spec["attr"],
                "pos": [float(v) for v in spec["pos"]],
                "size": [float(v) for v in spec["size"]],
                "pose_guard_obstacle_kinematic": bool(full),
                "contact_reporter_requested": False,
            }
        )

    top = float(specs[0]["pos"][2]) + float(specs[0]["size"][2]) / 2.0
    if abs(top - surface_z) > TOL:
        _fail(f"table TOP face {top} != vb_table_surface_z {surface_z}")
    if full:
        keepout_top = float(specs[1]["pos"][2]) + float(specs[1]["size"][2]) / 2.0
        slab_bottom = float(specs[0]["pos"][2]) - float(specs[0]["size"][2]) / 2.0
        if abs(keepout_top - slab_bottom) > TOL:
            _fail("robot keep-out must end exactly at the top-slab underside")

    done = getattr(env_cfg.terminations, "robot_hit_table", None)
    if done is None:
        _fail("terminations.robot_hit_table is missing — the table is a decoration")
    for key, want in (("near_x", near_x), ("surface_z", surface_z)):
        if abs(float(done.params[key]) - want) > TOL:
            _fail(f"terminations.robot_hit_table.params.{key} "
                  f"{done.params[key]} != {want} (box would not match the collider)")
    if bool(done.params.get("full_table_assembly", False)) != full:
        _fail("robot_hit_table full_table_assembly does not match the scene")
    if bool(done.params.get("require_substep_latch", False)) != full:
        _fail("robot_hit_table substep-latch mode does not match the scene")
    if abs(
        float(done.params.get("force_threshold", float("nan")))
        - float(TABLE_HIT_FORCE_THRESHOLD_N)
    ) > 0.0:
        _fail(
            "robot_hit_table force threshold is not the reviewed no-touch "
            f"numerical-zero tolerance {TABLE_HIT_FORCE_THRESHOLD_N:g} N"
        )

    done_filtered = done.params.get("filtered_sensor_cfg")
    expected_compat_sensor_name = (
        "contact_forces"
        if full
        else "racket_table_contact"
    )
    if (
        done_filtered is None
        or done_filtered.name != expected_compat_sensor_name
    ):
        _fail(
            "terminations.robot_hit_table compatibility sensor must resolve "
            f"the installed {expected_compat_sensor_name!r} sensor"
        )
    pose_guard_cfg = None
    if full:
        configured_sensor_names = tuple(
            getattr(env_cfg, "table_pair_contact_sensor_names", ()) or ()
        )
        done_sensor_cfgs = done.params.get("full_table_filtered_sensor_cfgs")
        expected_body_names = tuple(
            done.params.get("expected_full_robot_body_names", ()) or ()
        )
        proxy_path = done.params.get("collision_proxy_artifact_path")
        proxy_sha256 = done.params.get(
            "collision_proxy_artifact_sha256"
        )
        if (
            not isinstance(done_sensor_cfgs, (tuple, list))
            or tuple(done_sensor_cfgs)
            or configured_sensor_names
        ):
            _fail(
                "ActionBall full pose guard must own zero pair-filtered sensors"
            )
        if (
            len(TABLE_CONTACT_BODY_NAMES) != 32
            or len(set(TABLE_CONTACT_BODY_NAMES)) != 32
            or expected_body_names != tuple(TABLE_CONTACT_BODY_NAMES)
            or tuple(
                done.params.get(
                    "expected_full_table_source_prim_paths", ()
                )
                or ()
            )
            != expected_prims
            or tuple(TABLE_COMPONENT_ROLES)
            != tuple(spec["role"] for spec in specs)
            or not isinstance(proxy_path, str)
            or not proxy_path
            or not isinstance(proxy_sha256, str)
            or _SHA256_RE.fullmatch(proxy_sha256) is None
        ):
            _fail(
                "ActionBall does not bind the exact five-component × 32-body "
                "pose-OBB artifact contract"
            )
        pose_guard_cfg = {
            "component_roles": list(TABLE_COMPONENT_ROLES),
            "source_prims": list(expected_prims),
            "robot_body_names": list(expected_body_names),
            "collision_proxy_artifact_path": proxy_path,
            "collision_proxy_artifact_sha256": proxy_sha256,
            "pair_filtered_sensor_count": 0,
        }
        action = env_cfg.actions.joint_pos
        if (
            not bool(getattr(action, "table_contact_substep_guard", False))
            or getattr(action, "table_contact_guard_termination_term", None)
            != "robot_hit_table"
            or int(getattr(action, "table_contact_guard_expected_decimation", -1)) != 4
            or int(getattr(env_cfg, "decimation", -1)) != 4
        ):
            _fail("ActionBall action term does not bind the exact four-substep table latch")
        filtered_cfg = None
    else:
        filtered_cfg = getattr(env_cfg.scene, "racket_table_contact", None)
        if filtered_cfg is None:
            _fail(
                "scene.racket_table_contact is missing — offset racket contacts can be missed"
            )
        if (
            filtered_cfg.prim_path
            != "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link"
        ):
            _fail(
                "scene.racket_table_contact.prim_path is not the A3 wrist"
            )
        if tuple(filtered_cfg.filter_prim_paths_expr) != expected_prims:
            _fail(
                "scene.racket_table_contact filters are not the exact canonical top"
            )

    rew = getattr(env_cfg.rewards, "table_hit_penalty", None)
    if rew is None:
        _fail("rewards.table_hit_penalty is missing")
    if rew.params.get("term_name") != "robot_hit_table":
        _fail(f"rewards.table_hit_penalty points at {rew.params.get('term_name')!r}")

    _results["cfg"] = {
        "mode": "full_action_ball" if full else "legacy_top",
        "table_obstacle_prim": expected_prims[0],
        "table_obstacle_prims": list(expected_prims),
        "components": component_rows,
        "surface_z": surface_z,
        "near_x": near_x,
        "termination_params": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                               for k, v in done.params.items()},
        "legacy_filtered_contact_sensor": (
            None
            if filtered_cfg is None
            else {
                "name": "racket_table_contact",
                "prim_path": filtered_cfg.prim_path,
                "filter_prim_paths_expr": list(
                    filtered_cfg.filter_prim_paths_expr
                ),
            }
        ),
        "full_table_pose_obb_guard": pose_guard_cfg,
        "force_threshold_n": float(
            done.params["force_threshold"]
        ),
        "table_hit_penalty_weight": float(rew.weight),
    }
    print(
        f"ok cfg: {len(specs)} collider(s) + "
        f"{0 if full else 1} pair-filter sensor(s) + "
        f"{'pose-OBB guard + ' if full else ''}"
        "termination + penalty all mutually consistent"
    )


def check_spawned(env, env_cfg):
    """Read every pose and CollisionAPI PhysX actually has, not only requested cfg."""

    print("HOPE_TABLE_DIAGNOSTIC_STAGE=spawn_check_begin", flush=True)
    from pxr import PhysxSchema, Usd, UsdGeom
    import isaacsim.core.utils.stage as stage_utils

    full, specs = _component_specs(env_cfg)
    stage = stage_utils.get_current_stage()
    origin = env.unwrapped.scene.env_origins[0].tolist()

    # ``CuboidCfg`` spawns an Xform at ``prim_path`` with the actual Cube geometry underneath, and
    # the collision API lands on the geometry prim, not on the Xform. Walk the subtree.
    from pxr import UsdPhysics

    spawned_components = []
    for spec in specs:
        print(
            "HOPE_TABLE_DIAGNOSTIC_STAGE="
            f"spawn_component_begin:{spec['role']}",
            flush=True,
        )
        prim_path = spec["prim"].replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            _fail(f"{prim_path} does not exist on the stage")
        xform = UsdGeom.Xformable(prim)
        m = xform.ComputeLocalToWorldTransform(0.0)
        t = m.ExtractTranslation()
        local = (
            float(t[0]) - origin[0],
            float(t[1]) - origin[1],
            float(t[2]) - origin[2],
        )
        _close(local, spec["pos"], f"{prim_path} world transform")
        # Do not call ``BBoxCache.ComputeWorldBound`` here.  In the shipped
        # Isaac Sim 4.5 runtime it can terminate Kit with status zero while
        # materializing an invisible kinematic Cuboid, before any contact
        # tensor is read.  ``check_cfg`` already verifies the authored Cuboid
        # size.  The spawned check therefore verifies the live transform,
        # CollisionAPI and RigidBodyAPI; the physical contact smoke below is
        # the stronger end-to-end proof that each authored volume exists.
        collider_paths = [
            str(descendant.GetPath())
            for descendant in Usd.PrimRange(prim)
            if descendant.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not collider_paths:
            _fail(
                f"{prim_path} subtree carries no UsdPhysics.CollisionAPI — PhysX will ignore it"
            )
        if full:
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                _fail(
                    f"{prim_path} pose-guard obstacle has no UsdPhysics.RigidBodyAPI"
                )
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rigid_api.GetKinematicEnabledAttr()
            if (
                not kinematic_attr
                or not kinematic_attr.HasAuthoredValue()
                or kinematic_attr.Get() is not True
            ):
                _fail(
                    f"{prim_path} pose-guard obstacle is not kinematic"
                )
        enabled = []
        for collider_path in collider_paths:
            api = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(collider_path))
            attr = api.GetCollisionEnabledAttr()
            enabled.append(
                bool(attr.Get()) if attr and attr.HasAuthoredValue() else True
            )
        if not all(enabled):
            _fail(f"{prim_path}: a collider has collisionEnabled=False")
        spawned_components.append(
            {
                "role": spec["role"],
                "prim_path": prim_path,
                "env_local_translation": list(local),
                "configured_size": list(spec["size"]),
                "collider_prims": collider_paths,
                "pose_guard_obstacle_kinematic": bool(full),
                "contact_report_api": bool(
                    prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                ),
            }
        )
        print(
            "HOPE_TABLE_DIAGNOSTIC_STAGE="
            f"spawn_component_done:{spec['role']}",
            flush=True,
        )

    print("HOPE_TABLE_DIAGNOSTIC_STAGE=spawn_visual_begin", flush=True)
    visual_attrs = (
        "table_obstacle_visual",
        "shadow_table_visual",
        "pb_table_visual",
    )
    configured_visuals = [
        (name, getattr(env_cfg.scene, name))
        for name in visual_attrs
        if getattr(env_cfg.scene, name, None) is not None
    ]
    if len(configured_visuals) != 1:
        _fail(
            "table assembly must have exactly one visual provider; got "
            f"{[name for name, _ in configured_visuals]!r}"
        )
    visual_name, visual_cfg = configured_visuals[0]
    visual_path = visual_cfg.prim_path.replace(
        "{ENV_REGEX_NS}", "/World/envs/env_0"
    )
    # SceneEntityCfg has already expanded ``{ENV_REGEX_NS}`` by this point on
    # some Isaac Lab versions, leaving the authored regex in the config while
    # the live stage contains the concrete env_0 prim.
    visual_path = re.sub(
        r"^/World/envs/env_\.\*/",
        "/World/envs/env_0/",
        visual_path,
    )
    visual_prim = stage.GetPrimAtPath(visual_path)
    if not visual_prim.IsValid():
        _fail(f"{visual_path} visual provider was not spawned")
    visual_colliders = [
        str(descendant.GetPath())
        for descendant in Usd.PrimRange(visual_prim)
        if descendant.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if visual_colliders:
        _fail(
            f"{visual_path} visual-only subtree unexpectedly carries colliders: "
            f"{visual_colliders!r}"
        )
    if full:
        for forbidden in ("/World/envs/env_0/ShadowTable", "/World/envs/env_0/PhysicalTable"):
            if stage.GetPrimAtPath(forbidden).IsValid():
                _fail(f"ActionBall full assembly has overlapping truth top {forbidden}")
    print("HOPE_TABLE_DIAGNOSTIC_STAGE=spawn_visual_done", flush=True)

    print("HOPE_TABLE_DIAGNOSTIC_STAGE=spawn_terms_begin", flush=True)
    active = tuple(env.unwrapped.termination_manager.active_terms)
    if "robot_hit_table" not in active:
        _fail(f"robot_hit_table is not an active termination; active={active}")
    print("HOPE_TABLE_DIAGNOSTIC_STAGE=spawn_terms_done", flush=True)
    runtime_sensor_rows = []
    pose_guard_row = None
    if full:
        robot_body_names = tuple(
            str(name) for name in env.unwrapped.scene["robot"].body_names
        )
        if (
            len(robot_body_names) != 32
            or len(set(robot_body_names)) != 32
            or set(robot_body_names) != set(TABLE_CONTACT_BODY_NAMES)
        ):
            _fail(
                "runtime articulation body names are not a bijection with the "
                "exact 32-body pose-guard contract"
            )
        if tuple(getattr(env_cfg, "table_pair_contact_sensor_names", ()) or ()):
            _fail("full pose guard unexpectedly owns pair-filtered sensors")
        action = env.unwrapped.action_manager.get_term("joint_pos")
        from whole_body_tracking.tasks.tracking.mdp.terminations import (
            _A3_COLLISION_PROXY_COMPONENT_COUNT as A3_TABLE_GUARD_COMPONENT_COUNT,
        )

        params = action._resolved_table_contact_params()
        prepared = getattr(
            action, "_table_contact_prepared_pose_guard", None
        )
        runtime_receipt = getattr(
            env.unwrapped.scene["robot"],
            "_hope_a3_runtime_usd_receipt",
            None,
        )
        if (
            params.get("full_table_assembly") is not True
            or tuple(params.get("full_table_filtered_sensor_cfgs", ()))
            or prepared is None
            or not isinstance(runtime_receipt, Mapping)
            or runtime_receipt.get("kind")
            != "a3_pose_guard_live_runtime_usd_v1"
            or _SHA256_RE.fullmatch(
                str(runtime_receipt.get("bundle_tree_sha256", ""))
            )
            is None
            or int(prepared._component_indices.shape[0])
            != A3_TABLE_GUARD_COMPONENT_COUNT
            or int(prepared._aabb_lo.shape[0]) != 5
        ):
            _fail(
                "spawned ActionBall pose guard is not fully prepared against "
                "the live USD/articulation/62-component/5-obstacle contract"
            )
        pose_guard_row = {
            "component_roles": list(TABLE_COMPONENT_ROLES),
            "robot_body_names": list(robot_body_names),
            "collision_component_count": int(
                prepared._component_indices.shape[0]
            ),
            "obstacle_count": int(prepared._aabb_lo.shape[0]),
            "pair_filtered_sensor_count": 0,
            "runtime_usd_receipt": dict(runtime_receipt),
        }
    else:
        filtered_sensor = env.unwrapped.scene.sensors.get(
            "racket_table_contact"
        )
        if filtered_sensor is None:
            _fail("spawned scene has no racket_table_contact sensor")
        matrix = getattr(filtered_sensor.data, "force_matrix_w", None)
        if (
            matrix is None
            or matrix.ndim != 4
            or matrix.shape[-1] != 3
            or int(matrix.shape[2]) != len(specs)
        ):
            _fail(
                "spawned racket_table_contact has no exact legacy "
                "[env, body, filter, 3] force_matrix_w; got "
                f"{None if matrix is None else tuple(matrix.shape)}"
            )
        runtime_sensor_rows.append(
            {
                "name": "racket_table_contact",
                "body_name": "right_wrist_yaw_Link",
                "force_matrix_shape": list(matrix.shape),
            }
        )
    rew_active = tuple(env.unwrapped.reward_manager.active_terms)
    _results["spawned"] = {
        "mode": "full_action_ball" if full else "legacy_top",
        "pose_guard_robot_body_names": (
            list(robot_body_names) if full else []
        ),
        "components": spawned_components,
        "visual_provider": {
            "scene_attr": visual_name,
            "prim_path": visual_path,
            "collider_prims": visual_colliders,
        },
        "full_table_pose_obb_guard": pose_guard_row,
        "legacy_filtered_contact_sensors": runtime_sensor_rows,
        "active_terminations": list(active),
        "table_hit_penalty_active": "table_hit_penalty" in rew_active,
        # These two names are the metrics channels the termination produces for free:
        # my_on_policy_runner logs Live/Termination/<term>, and the behavior ledger books
        # termination_reason_<term>_count from termination_manager.active_terms.
        "metric_channels": [
            "Live/Termination/robot_hit_table",
            "Live/racket_target/termination_reason_robot_hit_table_count",
        ],
    }
    print(
        f"ok spawned: {len(specs)} exact collider(s), one collider-free visual, "
        f"{len(runtime_sensor_rows)} legacy pair-filter sensor(s), "
        f"{'prepared pose-OBB guard, ' if full else ''}"
        "robot_hit_table active"
    )


def contact_smoke(env, env_cfg):
    """Inject exact pose-OBB overlaps at selected physics substeps.

    This is deliberately an opt-in destructive smoke on a one-environment throwaway process.  It
    moves the articulation root just before a chosen physics substep so a named rigid-body origin
    is inside one selected table component, then restores the safe pose after that substep has
    been sampled.  No pose tensors or termination masks are forged.
    """

    print("HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_begin", flush=True)
    unwrapped = env.unwrapped
    full, specs = _component_specs(env_cfg)
    if not full:
        _fail("--contact-smoke requires the ActionBall five-part table assembly")
    if int(unwrapped.num_envs) != 1:
        _fail("--contact-smoke requires --num-envs 1 so every reset/probe is isolated")
    if ARGS.bench:
        _fail("--contact-smoke and --bench must run in separate processes")

    print("HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_runtime_begin", flush=True)
    action = unwrapped.action_manager.get_term("joint_pos")
    robot = unwrapped.scene["robot"]
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    zero_action = torch.zeros(
        1, unwrapped.action_manager.total_action_dim, device=unwrapped.device
    )
    done_cfg = unwrapped.termination_manager.get_term_cfg("robot_hit_table")
    exact_cfgs = done_cfg.params.get("full_table_filtered_sensor_cfgs")
    expected_source_prims = tuple(
        spec["prim"] for spec in specs
    )
    if (
        not isinstance(exact_cfgs, (tuple, list))
        or tuple(exact_cfgs)
        or tuple(
            done_cfg.params.get(
                "expected_full_table_source_prim_paths", ()
            )
        )
        != expected_source_prims
        or tuple(
            done_cfg.params.get("expected_full_robot_body_names", ())
        )
        != tuple(TABLE_CONTACT_BODY_NAMES)
        or len(robot.body_names) != 32
        or set(robot.body_names) != set(TABLE_CONTACT_BODY_NAMES)
    ):
        _fail(
            "contact smoke runtime differs from the exact zero-sensor "
            "five-component × 32-body pose-OBB contract"
        )
    from whole_body_tracking.tasks.tracking.mdp.terminations import (
        _A3_COLLISION_PROXY_COMPONENT_COUNT as A3_TABLE_GUARD_COMPONENT_COUNT,
    )

    params = action._resolved_table_contact_params()
    prepared = getattr(action, "_table_contact_prepared_pose_guard", None)
    if (
        params.get("full_table_assembly") is not True
        or prepared is None
        or int(prepared._component_indices.shape[0])
            != A3_TABLE_GUARD_COMPONENT_COUNT
        or int(prepared._aabb_lo.shape[0]) != 5
    ):
        _fail(
            "contact smoke pose guard was not prepared against "
            f"{A3_TABLE_GUARD_COMPONENT_COUNT} collision components and five "
            "table boxes"
        )
    print("HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_runtime_done", flush=True)

    print("HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_ledger_begin", flush=True)
    command = unwrapped.command_manager.get_term("racket_target")
    ledger = command._ensure_exact_behavior_decision_counters()
    reason_key = "termination_reason_robot_hit_table_count"
    if reason_key not in ledger or "terminal_reset_count" not in ledger:
        _fail("behavior ledger has no raw robot_hit_table/terminal counters")
    print("HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_ledger_done", flush=True)

    role_to_spec = {spec["role"]: spec for spec in specs}
    top = role_to_spec["top"]
    # Pinned Isaac Lab allocates one force slot per filter expression.  Every table-source
    # matrix therefore exposes the exact ordered 32 A3 body slots explicitly, rather than
    # collapsing a Robot wildcard into one slot.
    # Runtime positive controls use rigid bodies with shipped collision
    # geometry; several intermediate A3 rigid links intentionally carry no
    # collider, so demanding a physical positive from every body is invalid.
    top_body_probes = (
        (
            "top_blade_wrist_s1",
            "top",
            "right_wrist_yaw_Link",
            1,
            top["pos"],
        ),
    )
    component_probes = (
        # These additional rows cover every non-top assembly component.
        # ``right_wrist_yaw_Link`` carries the fixed-merged blade and handle geoms.
        (
            "keepout_elbow_s3",
            "keepout",
            "right_elbow_Link",
            3,
            role_to_spec["keepout"]["pos"],
        ),
        (
            "net_blade_wrist_s4",
            "net",
            "right_wrist_yaw_Link",
            4,
            role_to_spec["net"]["pos"],
        ),
        (
            "post_left_ankle_s1",
            "post_left",
            "right_ankle_pitch_Link",
            1,
            role_to_spec["post_left"]["pos"],
        ),
        (
            "post_right_blade_wrist_s2",
            "post_right",
            "right_wrist_yaw_Link",
            2,
            role_to_spec["post_right"]["pos"],
        ),
    )
    probes = top_body_probes + component_probes
    print(
        f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_probes_ready:{len(probes)}",
        flush=True,
    )
    # Establish one clean state before the first pulse.  Every pulse below
    # already terminates and automatically resets its row, and its own
    # post-reset clean step proves that the next probe starts clean.  Calling
    # ``env.reset()`` again between probes would resample the generic motion
    # command and can create a different, unrelated table-contact pose.
    # Scene construction can leave the articulation at its pre-reset pose.
    # Perform exactly one explicit reset here; later
    # probes rely only on each pulse's automatic reset and verified clean step.
    env.reset()
    (
        _initial_obs,
        _initial_reward,
        _initial_terminated,
        _initial_truncated,
        _initial_extras,
    ) = env.step(zero_action)
    initial_table_reason = unwrapped.termination_manager.get_term(
        "robot_hit_table"
    )
    if bool(initial_table_reason[0].item()):
        _fail("contact smoke has no table-clean state before its first pulse")
    print(
        "HOPE_TABLE_DIAGNOSTIC_STAGE=contact_smoke_initial_clean_done",
        flush=True,
    )
    from whole_body_tracking.tasks.tracking.mdp.terminations import (
        geometric_table_contact_hit_mask,
    )

    selected_body_pos = torch.index_select(
        robot.data.body_pos_w,
        1,
        prepared._asset_body_indices,
    ).detach().clone()
    selected_body_quat = torch.index_select(
        robot.data.body_quat_w,
        1,
        prepared._asset_body_indices,
    ).detach().clone()
    selected_body_pos[0, 0, 0] = float("nan")
    nonfinite_hit = geometric_table_contact_hit_mask(
        selected_body_pos,
        selected_body_quat,
        unwrapped.scene.env_origins,
        prepared._component_indices,
        prepared._component_centers,
        prepared._component_half_axes,
        prepared._aabb_lo,
        prepared._aabb_hi,
        racket_body_index=prepared._racket_index,
        racket_blade_center_offset_wrist_m=prepared._blade_center,
        racket_blade_local_half_axes_m=(
            prepared._blade_local_half_axes
        ),
    )
    if not bool(nonfinite_hit[0].item()):
        _fail("pose-OBB guard did not fail safe on non-finite live pose")

    def run_probe(probe):
        (
            name,
            role,
            body_name,
            pulse_substep,
            target_local,
        ) = probe
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_begin:{name}",
            flush=True,
        )
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_clean_state_ready:{name}",
            flush=True,
        )
        if body_name not in robot.body_names:
            _fail(f"contact probe body {body_name!r} is absent from articulation")
        body_id = robot.body_names.index(body_name)
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_body_ready:{name}",
            flush=True,
        )
        safe_root_pose = torch.cat(
            (
                robot.data.root_pos_w[env_ids],
                robot.data.root_quat_w[env_ids],
            ),
            dim=-1,
        ).detach().clone()
        safe_joint_pos = robot.data.joint_pos[env_ids].detach().clone()
        safe_joint_vel = robot.data.joint_vel[env_ids].detach().clone()
        safe_body_pos = robot.data.body_pos_w[env_ids, body_id].detach().clone()
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_pose_read:{name}",
            flush=True,
        )
        target_world = torch.tensor(
            target_local, dtype=safe_body_pos.dtype, device=safe_body_pos.device
        ).view(1, 3) + unwrapped.scene.env_origins[env_ids]
        contact_root_pose = safe_root_pose.clone()
        contact_root_pose[:, :3] += target_world - safe_body_pos
        zero_root_velocity = torch.zeros(
            (1, 6), dtype=safe_root_pose.dtype, device=safe_root_pose.device
        )
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_pose_ready:{name}",
            flush=True,
        )

        original_apply = action.apply_actions
        original_sample = action._sample_table_contact_current
        apply_index = 0
        samples = []

        def move_root(pose):
            robot.write_root_pose_to_sim(pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim(zero_root_velocity, env_ids=env_ids)

        def sampled():
            hit = original_sample()
            samples.append(hit.detach().clone())
            return hit

        def applied():
            nonlocal apply_index
            # ``apply_actions`` samples the physics substep that just
            # completed before it dispatches the next command.  Install the
            # probe pose only after that readback, so it affects exactly the
            # upcoming substep rather than rewriting the pose being sampled.
            original_apply()
            if apply_index == pulse_substep - 1:
                move_root(contact_root_pose)
            else:
                # Keep every negative-control substep at the exact captured
                # table-clear root.  Letting the unactuated throwaway smoke
                # drift between substeps can create a second, unrelated OBB
                # overlap and falsely report that the injected pulse leaked.
                move_root(safe_root_pose)
            apply_index += 1

        reason_before = int(ledger[reason_key].item())
        terminal_before = int(ledger["terminal_reset_count"].item())
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_counters_read:{name}",
            flush=True,
        )
        action.apply_actions = applied
        action._sample_table_contact_current = sampled
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_hooks_ready:{name}",
            flush=True,
        )
        try:
            print(
                f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_step_begin:{name}",
                flush=True,
            )
            _obs, _reward, terminated, _truncated, _extras = env.step(zero_action)
            print(
                f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_step_done:{name}",
                flush=True,
            )
        finally:
            action.apply_actions = original_apply
            action._sample_table_contact_current = original_sample

        if apply_index != 4 or len(samples) != 4:
            _fail(
                f"{name}: expected four apply/four pose samples, got "
                f"{apply_index}/{len(samples)}"
            )
        hit_rows = [bool(sample[0].item()) for sample in samples]
        if hit_rows[pulse_substep - 1] is not True:
            _fail(f"{name}: selected substep {pulse_substep} did not report contact: {hit_rows}")
        if any(hit for index, hit in enumerate(hit_rows) if index != pulse_substep - 1):
            _fail(f"{name}: intended one-substep pulse leaked across frames: {hit_rows}")
        raw_reason = unwrapped.termination_manager.get_term("robot_hit_table")
        if not bool(terminated[0].item()) or not bool(raw_reason[0].item()):
            _fail(f"{name}: real contact did not terminate as robot_hit_table")
        if int(ledger[reason_key].item()) != reason_before + 1:
            _fail(f"{name}: robot_hit_table behavior reason was not booked exactly once")
        if int(ledger["terminal_reset_count"].item()) != terminal_before + 1:
            _fail(f"{name}: generic terminal event was not booked exactly once")

        print(
            "HOPE_TABLE_DIAGNOSTIC_POSE_OBB="
            + json.dumps(
                {
                    "name": name,
                    "hit_rows": hit_rows,
                    "selected_table_role": role,
                    "selected_robot_body": body_name,
                    "pulse_substep": pulse_substep,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # ``env.step`` already reset the selected terminal row.  Do not call
        # ``env.reset`` here.  The generic command reset may independently
        # resample a motion pose that touches the table, while ActionBall
        # training atomically reinstalls its certified dynamic-ready state.
        # Restore the exact table-clear snapshot that this probe started from,
        # then let the immediately following physics step test whether the
        # sticky table latch and stale report were cleared.
        robot.write_root_pose_to_sim(safe_root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(
            zero_root_velocity, env_ids=env_ids
        )
        robot.write_joint_state_to_sim(
            safe_joint_pos,
            safe_joint_vel,
            env_ids=env_ids,
        )
        reason_after_positive = int(ledger[reason_key].item())
        terminal_after_positive = int(ledger["terminal_reset_count"].item())
        (
            _obs,
            _reward,
            clean_terminated,
            clean_truncated,
            _extras,
        ) = env.step(zero_action)
        clean_raw_reason = unwrapped.termination_manager.get_term(
            "robot_hit_table"
        )
        print(
            "HOPE_TABLE_DIAGNOSTIC_POST_RESET="
            + json.dumps(
                {
                    "name": name,
                    "terminated": bool(clean_terminated[0].item()),
                    "truncated": bool(clean_truncated[0].item()),
                    "raw_robot_hit_table": bool(clean_raw_reason[0].item()),
                    "table_reason_delta": (
                        int(ledger[reason_key].item())
                        - reason_after_positive
                    ),
                    "terminal_delta": (
                        int(ledger["terminal_reset_count"].item())
                        - terminal_after_positive
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            bool(clean_raw_reason[0].item())
            or int(ledger[reason_key].item()) != reason_after_positive
        ):
            _fail(
                f"{name}: selected automatic reset leaked table-specific "
                "sticky/raw/reason evidence into its first clean step"
            )
        row = {
            "name": name,
            "role": role,
            "body": body_name,
            "pulse_substep": pulse_substep,
            "current_hit_by_substep": hit_rows,
            "raw_robot_hit_table": True,
            "generic_terminated": True,
            "post_reset_other_terminated": bool(
                clean_terminated[0].item()
                or clean_truncated[0].item()
                or int(ledger["terminal_reset_count"].item())
                != terminal_after_positive
            ),
            "selected_reset_zero_leak": True,
            "physics_steps": 8,
        }
        print(
            f"HOPE_TABLE_DIAGNOSTIC_STAGE=contact_probe_done:{name}",
            flush=True,
        )
        return row

    rows = [run_probe(probe) for probe in probes]

    # One additional clean step after the final pulse's automatic reset is the
    # cross-episode leakage control.  Do not introduce an unrelated generic
    # command resample with another explicit reset.
    _obs, _reward, terminated, truncated, _extras = env.step(zero_action)
    raw_reason = unwrapped.termination_manager.get_term("robot_hit_table")
    if bool(raw_reason[0].item()):
        _fail("zero-pulse control leaked robot_hit_table across reset")
    _results["contact_smoke"] = {
        "pose_obb_guard_pass": True,
        "probes": rows,
        "nonfinite_fail_safe": True,
        "legal_stance_negative": True,
        "zero_pulse_after_reset": True,
        "zero_pulse_other_terminated": bool(
            terminated[0].item() or truncated[0].item()
        ),
        "physics_steps": sum(int(row["physics_steps"]) for row in rows) + 4,
    }
    print(
        "ok pose-OBB smoke: representative racket/wrist, elbow and ankle "
        "components covered all five table boxes and all four substeps; "
        "legal stance stayed clear, non-finite pose failed safe, raw "
        "reason/generic terminal counted once, reset leakage zero"
    )
    return int(_results["contact_smoke"]["physics_steps"])


def _term_mask(unwrapped, name: str):
    try:
        value = unwrapped.termination_manager.get_term(name)
    except Exception as exc:
        _fail(f"formal motion sweep cannot read termination term {name!r}: {exc}")
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.bool
        or tuple(value.shape) != (int(unwrapped.num_envs),)
    ):
        _fail(
            f"termination term {name!r} is not one bool [num_envs] runtime mask"
        )
    return value


def _raw_action_for_joint_target(action, target):
    """Invert the live JointPositionAction affine map for one exact q_des."""

    if target.ndim != 2 or target.shape[0] != 1:
        _fail(f"formal q_des target has invalid shape {tuple(target.shape)}")
    try:
        scale = torch.as_tensor(
            action._scale, dtype=target.dtype, device=target.device
        )
        offset = torch.as_tensor(
            action._offset, dtype=target.dtype, device=target.device
        )
    except Exception as exc:
        _fail(f"cannot read live JointPositionAction affine map: {exc}")
    try:
        scale = torch.broadcast_to(scale, target.shape)
        offset = torch.broadcast_to(offset, target.shape)
    except RuntimeError as exc:
        _fail(
            "live JointPositionAction affine map does not broadcast to the "
            f"motion joint order: {exc}"
        )
    if not bool(torch.all(torch.isfinite(scale) & scale.ne(0.0)).item()):
        _fail("live JointPositionAction scale is non-finite or zero")
    raw = (target - offset) / scale
    if not bool(torch.all(torch.isfinite(raw)).item()):
        _fail("inverse-affine exact motion action is non-finite")
    return raw


def _assert_nominal_hold_motion(unwrapped, inputs: _NominalHoldInput) -> None:
    command = unwrapped.command_manager.get_term("motion")
    motion = command.motion
    if (
        tuple(Path(value).resolve() for value in command._motion_files)
        != (inputs.motion_path,)
        or tuple(command._motion_file_sha256) != (inputs.motion_sha256,)
        or int(motion.num_segments) != 1
        or int(motion.joint_pos.shape[1]) != 31
    ):
        _fail("live MotionCommand does not bind the exact dynamic-ready motion")
    expected_q = torch.tensor(
        inputs.teacher_joint_pos,
        dtype=motion.joint_pos.dtype,
        device=motion.joint_pos.device,
    )
    if not bool(torch.allclose(motion.joint_pos[0], expected_q, rtol=0.0, atol=1e-6)):
        _fail("live MotionCommand teacher q differs from exact teacher frame zero")
    expected_root_pos = torch.tensor(
        inputs.teacher_root_pos,
        dtype=motion.body_pos_w.dtype,
        device=motion.body_pos_w.device,
    )
    expected_root_quat = torch.tensor(
        inputs.teacher_root_quat,
        dtype=motion.body_quat_w.dtype,
        device=motion.body_quat_w.device,
    )
    if not bool(
        torch.allclose(
            motion.body_pos_w[0, 0], expected_root_pos, rtol=0.0, atol=1e-6
        )
        and torch.allclose(
            motion.body_quat_w[0, 0],
            expected_root_quat,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        _fail("live MotionCommand teacher root differs from exact teacher frame zero")


def _hold_root(robot) -> tuple[float, float]:
    pos = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0]
    if not bool(torch.all(torch.isfinite(torch.cat((pos, quat)))).item()):
        return float("nan"), float("nan")
    quat = quat / torch.linalg.vector_norm(quat).clamp(min=1e-12)
    upright = torch.clamp(
        1.0 - 2.0 * (quat[1].square() + quat[2].square()), -1.0, 1.0
    )
    return float(pos[2].item()), float(torch.acos(upright).item())


def _hold_feet(unwrapped) -> float | None:
    try:
        sensor = unwrapped.scene.sensors["contact_forces"]
        body_ids, _names = sensor.find_bodies([".*ankle_roll.*"])
        ids = [int(value) for value in body_ids]
        if len(ids) == 2:
            forces = torch.linalg.vector_norm(
                sensor.data.net_forces_w[0, ids, :], dim=-1
            )
            return float((forces > 10.0).to(torch.float32).mean().item())
    except Exception:
        pass
    return None


def _nominal_hold_paddle_center_w(unwrapped) -> object:
    """Return the live canonical racket site from wrist pose + cfg offset."""

    robot = unwrapped.scene["robot"]
    body_ids, body_names = robot.find_bodies(
        ["right_wrist_yaw_Link"], preserve_order=True
    )
    if len(body_ids) != 1 or tuple(body_names) != ("right_wrist_yaw_Link",):
        raise TableSmokeReceiptError(
            "nominal hold cannot resolve the canonical racket wrist body"
        )
    command = unwrapped.command_manager.get_term("racket_target")
    offset = torch.as_tensor(
        command.cfg.mount_offset,
        device=unwrapped.device,
        dtype=robot.data.body_pos_w.dtype,
    ).reshape(3)
    body_id = int(body_ids[0])
    position = robot.data.body_pos_w[0, body_id]
    quaternion = robot.data.body_quat_w[0, body_id]
    quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp(
        min=1.0e-12
    )
    vector = quaternion[1:]
    twice_cross = 2.0 * torch.linalg.cross(vector, offset, dim=0)
    rotated = (
        offset
        + quaternion[0] * twice_cross
        + torch.linalg.cross(vector, twice_cross, dim=0)
    )
    return position + rotated


def _nominal_hold_frame0_fidelity_target(
    unwrapped, inputs: _NominalHoldInput, motion_command: object
) -> dict[str, Any]:
    """Freeze exact frame-0 joint/root/measured-racket targets in world axes."""

    origin = unwrapped.scene.env_origins[0]
    root_pos = torch.as_tensor(
        inputs.teacher_root_pos,
        device=unwrapped.device,
        dtype=origin.dtype,
    ) + origin
    root_quat = torch.as_tensor(
        inputs.teacher_root_quat,
        device=unwrapped.device,
        dtype=origin.dtype,
    )
    joint_pos = torch.as_tensor(
        inputs.teacher_joint_pos,
        device=unwrapped.device,
        dtype=unwrapped.scene["robot"].data.joint_pos.dtype,
    )
    measured = getattr(
        motion_command.motion, "_measured_racket_site_pos_w", None
    )
    direct_frame0 = (
        inputs.document.get("physical_birth_composition", {}).get("semantics")
        == MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS
    )
    if direct_frame0 and (
        not bool(getattr(motion_command.motion, "measured_racket_available", False))
        or measured is None
        or tuple(measured.shape) != (int(motion_command.motion.time_step_total), 3)
    ):
        raise TableSmokeReceiptError(
            "direct measured frame0 hold has no exact measured racket-site channel"
        )
    if measured is None:
        paddle = _nominal_hold_paddle_center_w(unwrapped).detach().clone()
        paddle_source = "live_fk_after_exact_frame0_write"
    else:
        paddle = measured[0].to(device=unwrapped.device) + origin
        paddle_source = "motion_npz.measured_racket_site_pos_w[0]"
    return {
        "joint_pos": joint_pos,
        "root_pos_w": root_pos,
        "root_quat_wxyz": root_quat,
        "paddle_center_w": paddle,
        "paddle_reference_source": paddle_source,
    }


def _nominal_hold_frame0_fidelity_sample(
    unwrapped, target: Mapping[str, Any]
) -> dict[str, Any]:
    robot = unwrapped.scene["robot"]
    joint_error = robot.data.joint_pos[0] - target["joint_pos"]
    root_error = robot.data.root_pos_w[0] - target["root_pos_w"]
    actual_quat = robot.data.root_quat_w[0]
    target_quat = target["root_quat_wxyz"]
    actual_quat = actual_quat / torch.linalg.vector_norm(actual_quat).clamp(
        min=1.0e-12
    )
    target_quat = target_quat / torch.linalg.vector_norm(target_quat).clamp(
        min=1.0e-12
    )
    orientation_error = 2.0 * torch.acos(
        torch.clamp(torch.abs(torch.dot(actual_quat, target_quat)), 0.0, 1.0)
    )
    paddle_error = (
        _nominal_hold_paddle_center_w(unwrapped)
        - target["paddle_center_w"]
    )
    values = torch.cat(
        (joint_error, root_error, orientation_error.reshape(1), paddle_error)
    )
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise TableSmokeReceiptError(
            "frame0 fidelity telemetry became non-finite"
        )
    return {
        "joint_error_rad": [float(value) for value in joint_error.tolist()],
        "root_position_error_m": [float(value) for value in root_error.tolist()],
        "root_orientation_error_rad": float(orientation_error.item()),
        "paddle_center_error_m": [float(value) for value in paddle_error.tolist()],
    }


def _nominal_hold_frame0_fidelity_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
    paddle_reference_source: str,
) -> dict[str, Any]:
    if not samples:
        raise TableSmokeReceiptError("frame0 fidelity has no samples")
    per_joint_max = [
        max(abs(float(row["joint_error_rad"][index])) for row in samples)
        for index in range(31)
    ]
    joint_values = [
        float(value)
        for row in samples
        for value in row["joint_error_rad"]
    ]
    root_norms = [
        math.sqrt(sum(float(value) ** 2 for value in row["root_position_error_m"]))
        for row in samples
    ]
    paddle_norms = [
        math.sqrt(sum(float(value) ** 2 for value in row["paddle_center_error_m"]))
        for row in samples
    ]
    return {
        "schema_version": 1,
        "reference": "exact_teacher_frame0",
        "paddle_reference_source": paddle_reference_source,
        "sampling": "post_write_and_nonterminal_policy_step_endpoints",
        "sample_count": len(samples),
        "joint_order": list(joint_names),
        "maximum_absolute_joint_error_rad": max(per_joint_max),
        "rms_joint_error_rad": math.sqrt(
            sum(value * value for value in joint_values) / len(joint_values)
        ),
        "per_joint_maximum_absolute_error_rad": per_joint_max,
        "maximum_root_position_error_m": max(root_norms),
        "maximum_root_orientation_error_rad": max(
            float(row["root_orientation_error_rad"]) for row in samples
        ),
        "maximum_paddle_center_error_m": max(paddle_norms),
        "rms_paddle_center_error_m": math.sqrt(
            sum(value * value for value in paddle_norms) / len(paddle_norms)
        ),
        "initial": dict(samples[0]),
        "final_sample": dict(samples[-1]),
        "formal_thresholds_adopted": False,
    }


def _nominal_hold_delay_contract_matches(
    *, present: bool, actual: object, expected: Mapping[str, Any]
) -> bool:
    """Match the explicit candidate contract to runtime's disabled omission.

    ``runtime_execution_facts`` intentionally omits
    ``control_step_action_delay`` when the instantiated action term has exact
    zero delay.  Dynamic-ready artifacts keep an explicit disabled block so
    the candidate still pins that choice.  Absence is therefore the sole live
    representation of the explicit disabled block; an enabled or otherwise
    different live contract remains a hard mismatch.
    """

    if expected["enabled"] is False:
        return not present
    return present and actual == expected


def _nominal_hold_physx_control_contract(
    value: object, *, joint_names: Sequence[str]
) -> dict[str, Any]:
    """Validate the exact Vendor H_ctrl block carried by the hold artifact."""

    selected = (
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    )
    required = {
        "schema_version",
        "backend",
        "inset_fraction_per_side_hard_span",
        "selected_joint_names",
        "mechanical_joint_pos_limits",
        "control_joint_pos_limits",
        "unselected_joint_count",
        "unselected_limits_equal_mechanical",
        "articulation_mechanical_ledger_unchanged",
        "soft_qdes_ledger_unchanged",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TableSmokeReceiptError(
            "nominal hold Vendor PhysX H_ctrl fields are incomplete or unknown"
        )
    mechanical = tuple(
        _finite_tuple(row, 2, "H_mech row")
        for row in value["mechanical_joint_pos_limits"]
    )
    control = tuple(
        _finite_tuple(row, 2, "H_ctrl row")
        for row in value["control_joint_pos_limits"]
    )
    selected_indices = {
        index for index, name in enumerate(joint_names) if name in selected
    }
    if (
        value["schema_version"] != 1
        or value["backend"] != "physx_root_view_dof_limits"
        or type(value["inset_fraction_per_side_hard_span"]) is not float
        or value["inset_fraction_per_side_hard_span"] != 0.02
        or tuple(value["selected_joint_names"]) != selected
        or tuple(joint_names[index] for index in sorted(selected_indices))
        != selected
        or len(mechanical) != 31
        or len(control) != 31
        or value["unselected_joint_count"] != 27
        or value["unselected_limits_equal_mechanical"] is not True
        or value["articulation_mechanical_ledger_unchanged"] is not True
        or value["soft_qdes_ledger_unchanged"] is not True
    ):
        raise TableSmokeReceiptError(
            "nominal hold Vendor PhysX H_ctrl identity is invalid"
        )
    for index, (hard, constrained) in enumerate(zip(mechanical, control)):
        if index not in selected_indices:
            valid = constrained == hard
        else:
            span = hard[1] - hard[0]
            valid = (
                hard[0] < constrained[0] < constrained[1] < hard[1]
                and math.isclose(
                    constrained[0], hard[0] + 0.02 * span,
                    rel_tol=0.0, abs_tol=2.0e-7,
                )
                and math.isclose(
                    constrained[1], hard[1] - 0.02 * span,
                    rel_tol=0.0, abs_tol=2.0e-7,
                )
            )
        if hard[0] >= hard[1] or not valid:
            raise TableSmokeReceiptError(
                f"nominal hold Vendor PhysX H_ctrl row {index} is invalid"
            )
    return {
        **dict(value),
        "selected_joint_names": list(selected),
        "mechanical_joint_pos_limits": [list(row) for row in mechanical],
        "control_joint_pos_limits": [list(row) for row in control],
    }


def _nominal_hold_json_vector(
    value: object, *, expected: int, name: str
) -> list[float | None]:
    """Copy one runtime vector to JSON without allowing NaN/Inf to escape."""

    try:
        if hasattr(value, "detach"):
            raw = value.detach().to(device="cpu").reshape(-1).tolist()
        else:
            raw = list(value)
    except Exception as exc:
        raise TableSmokeReceiptError(f"cannot copy {name}") from exc
    if len(raw) != expected:
        raise TableSmokeReceiptError(
            f"{name} must contain {expected} values; got {len(raw)}"
        )
    result: list[float | None] = []
    for item in raw:
        number = float(item)
        result.append(number if math.isfinite(number) else None)
    return result


def _nominal_hold_joint_safety_summary(
    *,
    joint_names: Sequence[str],
    hard_lower: Sequence[float | None],
    hard_upper: Sequence[float | None],
    preterminal_q: Sequence[float | None],
    preterminal_qdot: Sequence[float | None],
    final_q: Sequence[float | None],
    final_qdot: Sequence[float | None],
    current_hard_edge: Sequence[bool],
    substep_actual_hard_edge: Sequence[bool],
    final_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exact per-joint nominal-hold attribution from copied values."""

    names = tuple(joint_names)
    size = len(names)
    if size != 31 or len(set(names)) != size:
        raise TableSmokeReceiptError(
            "nominal-hold safety telemetry requires 31 unique joints"
        )
    vectors = {
        "hard_lower_rad": list(hard_lower),
        "hard_upper_rad": list(hard_upper),
        "preterminal_joint_pos_rad": list(preterminal_q),
        "preterminal_joint_vel_radps": list(preterminal_qdot),
        "final_joint_pos_rad": list(final_q),
        "final_joint_vel_radps": list(final_qdot),
    }
    if any(len(value) != size for value in vectors.values()):
        raise TableSmokeReceiptError(
            "nominal-hold safety telemetry vector width drifted"
        )
    current = tuple(current_hard_edge)
    substep = tuple(substep_actual_hard_edge)
    if (
        len(current) != size
        or len(substep) != size
        or any(type(value) is not bool for value in (*current, *substep))
    ):
        raise TableSmokeReceiptError(
            "nominal-hold safety latch vectors must be 31 exact booleans"
        )
    gaps: list[float | None] = []
    for index, (lower, upper, position) in enumerate(
        zip(
            vectors["hard_lower_rad"],
            vectors["hard_upper_rad"],
            vectors["final_joint_pos_rad"],
        )
    ):
        if (
            lower is None
            or upper is None
            or not lower < upper
        ):
            raise TableSmokeReceiptError(
                f"nominal-hold hard bounds invalid at joint {index}"
            )
        gaps.append(
            None
            if position is None
            else min(position - lower, upper - position)
        )
    finite_gaps = [value for value in gaps if value is not None]
    minimum_gap = min(finite_gaps) if finite_gaps else None
    minimum_gap_joint = (
        names[gaps.index(minimum_gap)] if minimum_gap is not None else None
    )
    current_names = [
        names[index] for index, value in enumerate(current) if value
    ]
    substep_names = [
        names[index] for index, value in enumerate(substep) if value
    ]
    flagged = []
    for index, name in enumerate(names):
        if not (current[index] or substep[index]):
            continue
        flagged.append(
            {
                "joint_index": index,
                "joint_name": name,
                "current_actual_hard_edge": current[index],
                "substep_actual_hard_edge": substep[index],
                "preterminal_joint_pos_rad": vectors[
                    "preterminal_joint_pos_rad"
                ][index],
                "preterminal_joint_vel_radps": vectors[
                    "preterminal_joint_vel_radps"
                ][index],
                "final_joint_pos_rad": vectors["final_joint_pos_rad"][
                    index
                ],
                "final_joint_vel_radps": vectors[
                    "final_joint_vel_radps"
                ][index],
                "hard_lower_rad": vectors["hard_lower_rad"][index],
                "hard_upper_rad": vectors["hard_upper_rad"][index],
                "final_minimum_hard_gap_rad": gaps[index],
            }
        )
    return {
        "schema_version": 1,
        "complete": True,
        "joint_order": list(names),
        "hard_bounds_source": "robot.data.joint_pos_limits",
        **vectors,
        "final_minimum_hard_gap_rad": minimum_gap,
        "final_minimum_hard_gap_joint_name": minimum_gap_joint,
        "current_actual_hard_edge_joint_count": len(current_names),
        "current_actual_hard_edge_joint_names": current_names,
        "substep_actual_hard_edge_joint_count": len(substep_names),
        "substep_actual_hard_edge_joint_names": substep_names,
        "flagged_joint_rows": flagged,
        "final_source": dict(final_source),
    }


def _nominal_hold_live_joint_state(
    robot: object, *, joint_names: Sequence[str]
) -> dict[str, list[float | None]]:
    """Copy the one-env live articulation state and physical hard bounds."""

    size = len(tuple(joint_names))
    data = robot.data
    return {
        "joint_pos_rad": _nominal_hold_json_vector(
            data.joint_pos[0], expected=size, name="live joint position"
        ),
        "joint_vel_radps": _nominal_hold_json_vector(
            data.joint_vel[0], expected=size, name="live joint velocity"
        ),
        "hard_lower_rad": _nominal_hold_json_vector(
            data.joint_pos_limits[0, :, 0],
            expected=size,
            name="live hard lower bounds",
        ),
        "hard_upper_rad": _nominal_hold_json_vector(
            data.joint_pos_limits[0, :, 1],
            expected=size,
            name="live hard upper bounds",
        ),
    }


def _nominal_hold_terminal_joint_safety(
    action: object,
    *,
    joint_names: Sequence[str],
    hard_lower: Sequence[float | None],
    hard_upper: Sequence[float | None],
    preterminal: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    """Distill the exact terminal policy-step transcript retained across reset."""

    snapshot = action.joint_safety_ledger_snapshot()
    archives = snapshot.get("terminal_archives")
    if not isinstance(archives, tuple) or not archives:
        raise TableSmokeReceiptError(
            "terminal nominal hold has no joint-safety archive"
        )
    archive = max(archives, key=lambda row: int(row["archive_sequence"]))
    transcript = archive.get("transcript")
    if not isinstance(transcript, Mapping):
        raise TableSmokeReceiptError(
            "terminal joint-safety archive has no transcript"
        )
    record_count = transcript.get("record_count")
    if type(record_count) is not int or record_count <= 0:
        raise TableSmokeReceiptError(
            "terminal joint-safety transcript has no readback"
        )
    if transcript.get("complete") is not True:
        raise TableSmokeReceiptError(
            "terminal joint-safety transcript is incomplete"
        )
    final_index = record_count - 1
    q_records = transcript["q"]
    qdot_records = transcript["qdot"]
    actual_records = transcript["actual_hard_edge"]
    q = q_records[final_index]
    qdot = qdot_records[final_index]
    current = actual_records[final_index]
    substep = transcript["substep_actual_joint_latch"]
    record_kind = transcript.get("record_kind")
    timestamp = transcript.get("timestamp_s")
    summary = _nominal_hold_joint_safety_summary(
        joint_names=joint_names,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
        preterminal_q=preterminal["joint_pos_rad"],
        preterminal_qdot=preterminal["joint_vel_radps"],
        final_q=_nominal_hold_json_vector(
            q, expected=31, name="terminal joint position"
        ),
        final_qdot=_nominal_hold_json_vector(
            qdot, expected=31, name="terminal joint velocity"
        ),
        current_hard_edge=tuple(bool(value) for value in current.tolist()),
        substep_actual_hard_edge=tuple(
            bool(value) for value in substep.tolist()
        ),
        final_source={
            "kind": "joint_safety_terminal_archive",
            "archive_sequence": int(archive["archive_sequence"]),
            "policy_step_sequence": int(archive["policy_step_sequence"]),
            "transcript_complete": transcript.get("complete") is True,
            "record_count": record_count,
            "record_kind": (
                record_kind[final_index]
                if isinstance(record_kind, tuple)
                else None
            ),
            "timestamp_s": (
                float(timestamp[final_index])
                if isinstance(timestamp, tuple)
                else None
            ),
        },
    )
    q_rows = (
        q_records.detach().to(device="cpu").tolist()
        if hasattr(q_records, "detach")
        else q_records.tolist()
    )
    qdot_rows = (
        qdot_records.detach().to(device="cpu").tolist()
        if hasattr(qdot_records, "detach")
        else qdot_records.tolist()
    )
    actual_rows = (
        actual_records.detach().to(device="cpu").tolist()
        if hasattr(actual_records, "detach")
        else actual_records.tolist()
    )
    if not (
        len(q_rows) == len(qdot_rows) == len(actual_rows) == record_count
        and all(len(row) == 31 for row in q_rows)
        and all(len(row) == 31 for row in qdot_rows)
        and all(len(row) == 31 for row in actual_rows)
    ):
        raise TableSmokeReceiptError(
            "terminal joint-safety transcript matrix shape drifted"
        )
    trigger_rows = []
    for joint_index, joint_name in enumerate(joint_names):
        candidate_records = [
            index
            for index in range(record_count)
            if bool(actual_rows[index][joint_index])
        ]
        if not candidate_records:
            continue
        lower = float(hard_lower[joint_index])
        upper = float(hard_upper[joint_index])

        def signed_gap(record_index: int) -> float:
            position = float(q_rows[record_index][joint_index])
            if not math.isfinite(position):
                return float("-inf")
            return min(position - lower, upper - position)

        trigger_index = min(candidate_records, key=signed_gap)
        position = float(q_rows[trigger_index][joint_index])
        velocity = float(qdot_rows[trigger_index][joint_index])
        if not math.isfinite(position):
            side = "nonfinite_or_invalid"
            encoded_position = None
            gap = None
        elif position <= lower:
            side = "lower"
            encoded_position = position
            gap = position - lower
        elif position >= upper:
            side = "upper"
            encoded_position = position
            gap = upper - position
        else:
            side = "latched_but_final_record_inside"
            encoded_position = position
            gap = min(position - lower, upper - position)
        trigger_rows.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "side": side,
                "record_index": trigger_index,
                "record_kind": (
                    record_kind[trigger_index]
                    if isinstance(record_kind, tuple)
                    else None
                ),
                "timestamp_s": (
                    float(timestamp[trigger_index])
                    if isinstance(timestamp, tuple)
                    else None
                ),
                "joint_pos_rad": encoded_position,
                "joint_vel_radps": (
                    velocity if math.isfinite(velocity) else None
                ),
                "hard_lower_rad": lower,
                "hard_upper_rad": upper,
                "signed_hard_gap_rad": gap,
            }
        )
    summary["substep_trigger_joint_rows"] = trigger_rows
    return summary


def _nominal_hold_nonterminal_joint_safety(
    action: object,
    robot: object,
    *,
    joint_names: Sequence[str],
    preterminal: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    """Summarize a successful final live readback without a reset archive."""

    final = _nominal_hold_live_joint_state(robot, joint_names=joint_names)
    current = []
    for position, lower, upper in zip(
        final["joint_pos_rad"],
        final["hard_lower_rad"],
        final["hard_upper_rad"],
    ):
        current.append(
            position is None
            or lower is None
            or upper is None
            or position <= lower
            or position >= upper
        )
    substep_tensor = action.physics_substep_actual_hard_edge_joint_latch[0]
    return _nominal_hold_joint_safety_summary(
        joint_names=joint_names,
        hard_lower=final["hard_lower_rad"],
        hard_upper=final["hard_upper_rad"],
        preterminal_q=preterminal["joint_pos_rad"],
        preterminal_qdot=preterminal["joint_vel_radps"],
        final_q=final["joint_pos_rad"],
        final_qdot=final["joint_vel_radps"],
        current_hard_edge=tuple(current),
        substep_actual_hard_edge=tuple(
            bool(value) for value in substep_tensor.tolist()
        ),
        final_source={"kind": "live_post_policy_step_readback"},
    )


def nominal_hold_probe(
    env,
    env_cfg,
    inputs: _NominalHoldInput,
    *,
    duration_s: float,
    screenshot_dir: Path | None = None,
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    if int(unwrapped.num_envs) != 1:
        _fail("nominal hold requires exactly one environment")
    _assert_nominal_hold_motion(unwrapped, inputs)
    from whole_body_tracking.utils.training_contract import (
        runtime_execution_facts,
    )

    live_plant = runtime_execution_facts(unwrapped, None)
    mismatch = []
    for key, expected in inputs.expected_plant.items():
        actual = live_plant.get(key)
        if key in ("joint_names", "articulation_joint_names"):
            matched = tuple(actual or ()) == tuple(expected)
        elif key == "action_joint_ids":
            matched = tuple(actual or ()) == tuple(expected)
        elif key == "control_decimation":
            matched = actual == expected
        elif key == "control_step_action_delay":
            matched = _nominal_hold_delay_contract_matches(
                present=key in live_plant,
                actual=actual,
                expected=expected,
            )
        elif key == "physx_control_position_limits":
            matched = actual == expected
        else:
            try:
                got = torch.as_tensor(actual, dtype=torch.float64)
                want = torch.as_tensor(expected, dtype=torch.float64)
                matched = got.shape == want.shape and bool(
                    torch.allclose(got, want, rtol=0.0, atol=1.0e-6)
                )
            except Exception:
                matched = False
        if not matched:
            mismatch.append(key)
    if mismatch:
        _fail("dynamic-ready/live plant mismatch: " + ", ".join(mismatch))

    active = tuple(unwrapped.termination_manager.active_terms)
    safety = (
        "robot_hit_table",
        "base_fell_tilt",
        "base_too_low",
        "joint_qdes_forbidden",
        "joint_actual_forbidden",
    )
    reference_terms = ("anchor_pos", "anchor_ori", "ee_body_pos")
    if any(name not in active for name in safety) or any(
        name in active for name in reference_terms
    ):
        _fail("nominal-hold termination composition lost safety or kept reference envelopes")

    robot = unwrapped.scene["robot"]
    action = unwrapped.action_manager.get_term("joint_pos")
    motion_command = unwrapped.command_manager.get_term("motion")
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)

    if screenshot_dir is not None:
        os.mkdir(screenshot_dir, 0o755)
    screenshots = []
    last_png = None

    def save_frame(label: str, step: int, payload: bytes) -> None:
        assert screenshot_dir is not None
        screenshots.append(
            _publish_nominal_hold_screenshot(
                screenshot_dir,
                f"{len(screenshots):03d}_{label}_{step:04d}.png",
                label,
                step,
                payload,
            )
        )

    env.reset()
    _refresh_nominal_hold_derived_state(unwrapped)
    if screenshot_dir is not None:
        # The first headless rgb_array render primes RTX and is an all-black
        # frame on the Pod.  Discard it without stepping physics so the saved
        # raw reset image still represents the exact post-reset state.
        _nominal_hold_render_png(env)
        last_png = _nominal_hold_render_png(env)
        save_frame("raw_env_reset", 0, last_png)

    motion_command.clip_id[env_ids] = 0
    motion_command.time_steps[env_ids] = 0
    motion_command.time_steps_f[env_ids] = 0.0
    motion_command.speed_scale[env_ids] = 1.0
    root = robot.data.default_root_state[env_ids].detach().clone()
    root[:, :3] = (
        torch.tensor(inputs.physical_root_pos, device=root.device).view(1, 3)
        + unwrapped.scene.env_origins[env_ids]
    )
    root[:, 3:7] = torch.tensor(
        inputs.physical_root_quat, device=root.device
    ).view(1, 4)
    root[:, 7:13] = 0.0
    ready_q = torch.tensor(
        inputs.physical_joint_pos,
        device=unwrapped.device,
        dtype=robot.data.joint_pos.dtype,
    ).view(1, 31)
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        ready_q, torch.zeros_like(ready_q), env_ids=env_ids
    )
    unwrapped.scene.write_data_to_sim()
    _refresh_nominal_hold_derived_state(unwrapped)
    fidelity_target = _nominal_hold_frame0_fidelity_target(
        unwrapped, inputs, motion_command
    )
    fidelity_samples = [
        _nominal_hold_frame0_fidelity_sample(unwrapped, fidelity_target)
    ]
    hold_qdes = torch.tensor(
        inputs.hold_qdes, device=unwrapped.device, dtype=ready_q.dtype
    ).view(1, 31)
    raw_action = _raw_action_for_joint_target(action, hold_qdes)
    candidate_hold_action = torch.tensor(
        inputs.hold_action, device=unwrapped.device, dtype=ready_q.dtype
    ).view(1, 31)
    if not bool(
        torch.allclose(
            raw_action, candidate_hold_action, rtol=0.0, atol=2.0e-6
        )
    ):
        _fail("live action decoder disagrees with candidate normalized hold action")
    try:
        action.install_action_ball_dynamic_ready_state(
            env_ids,
            candidate_hold_action,
            hold_qdes,
            capture_rollback=False,
        )
    except Exception as exc:
        _fail(f"cannot install candidate hold qdes/action history: {exc}")

    if screenshot_dir is not None:
        last_png = _nominal_hold_render_png(env)
        save_frame("physical_ready_after_reset_write", 0, last_png)

    policy_dt = float(live_plant["policy_step_dt_s"])
    requested_steps = max(1, math.ceil(duration_s / policy_dt - 1e-12))
    root_samples = [_hold_root(robot)]
    foot_samples = []
    terminal_reasons = []
    terminated_value = False
    truncated_value = False
    completed = 0
    latest_preterminal = _nominal_hold_live_joint_state(
        robot, joint_names=inputs.joint_names
    )
    for step in range(1, requested_steps + 1):
        latest_preterminal = _nominal_hold_live_joint_state(
            robot, joint_names=inputs.joint_names
        )
        _obs, _reward, terminated, truncated, _extras = env.step(raw_action)
        completed = step
        terminated_value = bool(terminated[0].item())
        truncated_value = bool(truncated[0].item())
        terminal_reasons = [
            name
            for name in active
            if bool(_term_mask(unwrapped, name)[0].item())
        ]
        if terminated_value or truncated_value or terminal_reasons:
            if screenshot_dir is not None and last_png is not None:
                save_frame("preterminal", step - 1, last_png)
            break
        root_samples.append(_hold_root(robot))
        foot_samples.append(_hold_feet(unwrapped))
        fidelity_samples.append(
            _nominal_hold_frame0_fidelity_sample(unwrapped, fidelity_target)
        )
        if screenshot_dir is not None:
            last_png = _nominal_hold_render_png(env)
            if step in (1, 10):
                save_frame(f"after_step_{step}", step, last_png)

    if (
        screenshot_dir is not None
        and completed == requested_steps
        and not terminal_reasons
        and last_png is not None
    ):
        save_frame("final", completed, last_png)

    finite_roots = all(
        math.isfinite(z) and math.isfinite(tilt) for z, tilt in root_samples
    )
    known_feet = [value for value in foot_samples if value is not None]
    both_fraction = (
        None
        if not known_feet
        else sum(value >= 1.0 for value in known_feet) / len(known_feet)
    )
    try:
        if terminated_value or truncated_value or terminal_reasons:
            joint_safety = _nominal_hold_terminal_joint_safety(
                action,
                joint_names=inputs.joint_names,
                hard_lower=latest_preterminal["hard_lower_rad"],
                hard_upper=latest_preterminal["hard_upper_rad"],
                preterminal=latest_preterminal,
            )
        else:
            joint_safety = _nominal_hold_nonterminal_joint_safety(
                action,
                robot,
                joint_names=inputs.joint_names,
                preterminal=latest_preterminal,
            )
    except Exception as exc:
        joint_safety = {
            "schema_version": 1,
            "complete": False,
            "error": str(exc),
            "joint_order": list(inputs.joint_names),
        }
    passed = (
        completed == requested_steps
        and not terminated_value
        and not truncated_value
        and not terminal_reasons
        and finite_roots
        and joint_safety.get("complete") is True
        and joint_safety.get("current_actual_hard_edge_joint_count") == 0
        and joint_safety.get("substep_actual_hard_edge_joint_count") == 0
    )
    frame0_fidelity = _nominal_hold_frame0_fidelity_summary(
        fidelity_samples,
        joint_names=inputs.joint_names,
        paddle_reference_source=fidelity_target["paddle_reference_source"],
    )
    receipt = {
        "schema_version": 1,
        "kind": NOMINAL_HOLD_RECEIPT_KIND,
        "verdict": "PASS" if passed else "FAIL",
        "action_id": inputs.action_id,
        "artifact": {
            "path": str(inputs.artifact_path),
            "sha256": inputs.artifact_sha256,
            "content_sha256": inputs.document["content_sha256"],
        },
        "motion_sha256": inputs.motion_sha256,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": (
            inputs.teacher_physical_separated
        ),
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "plant_contract_match": True,
        "control_step_action_delay_runtime": (
            action.control_step_action_delay_runtime_receipt()
        ),
        "active_terminations": list(active),
        "requested_duration_s": duration_s,
        "completed_duration_s": completed * policy_dt,
        "completed_policy_steps": completed,
        "completed_physics_steps": completed * int(env_cfg.decimation),
        "terminal_reasons": terminal_reasons,
        "generic_terminated": terminated_value,
        "generic_truncated": truncated_value,
        "minimum_root_z_m": (
            min(z for z, _tilt in root_samples) if finite_roots else None
        ),
        "maximum_root_tilt_rad": (
            max(tilt for _z, tilt in root_samples) if finite_roots else None
        ),
        "both_feet_contact_fraction": both_fraction,
        "joint_safety_telemetry": joint_safety,
        "frame0_fidelity_telemetry": frame0_fidelity,
        "screenshots": screenshots,
    }
    receipt["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _results["nominal_hold"] = receipt
    return receipt


def sweep_formal_actions(
    env,
    env_cfg,
    inputs: _FormalInputs,
) -> tuple[tuple[_RuntimeActionEvidence, ...], int]:
    """Step every trusted action frame through the live four-substep path."""

    unwrapped = env.unwrapped
    if int(unwrapped.num_envs) != 1:
        _fail("formal motion sweep requires exactly one environment")
    if int(getattr(env_cfg, "decimation", -1)) != 4:
        _fail("formal motion sweep requires the ActionBall decimation of four")
    robot = unwrapped.scene["robot"]
    action = unwrapped.action_manager.get_term("joint_pos")
    motion_command = unwrapped.command_manager.get_term("motion")
    motion = motion_command.motion
    runtime_paths = tuple(
        Path(path).resolve() for path in motion_command._motion_files
    )
    expected_paths = tuple(row.file.path for row in inputs.motions)
    expected_sha = tuple(row.file.sha256 for row in inputs.motions)
    expected_n = int(inputs.action_set_contract["expected_n"])
    if (
        runtime_paths != expected_paths
        or tuple(motion_command._motion_file_sha256) != expected_sha
        or int(motion.num_segments) != expected_n
        or not bool(getattr(motion, "kinematics_contract_exact", False))
        or tuple(robot.body_names) != tuple(TABLE_CONTACT_BODY_NAMES)
    ):
        _fail(
            "live MotionCommand does not bind the exact ordered schema-2 "
            "trusted action-set bytes/body order"
        )
    segment_lengths = tuple(int(value) for value in motion.seg_len.tolist())
    if len(segment_lengths) != expected_n or any(
        value < 3 for value in segment_lengths
    ):
        _fail(f"formal motion segments are invalid: {segment_lengths!r}")
    if (
        int(unwrapped.action_manager.total_action_dim)
        != int(motion.joint_pos.shape[1])
    ):
        _fail(
            "formal motion joint columns do not equal the live action dimension"
        )
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    results: list[_RuntimeActionEvidence] = []
    total_physics_steps = 0
    for slot, (motion_input, segment_length) in enumerate(
        zip(inputs.motions, segment_lengths)
    ):
        env.reset()
        segment_start = int(motion.seg_start[slot].item())
        table_count = 0
        fall_count = 0
        hard_count = 0
        unsafe_count = 0
        stepped = 0
        for local_frame in range(segment_length):
            frame = segment_start + local_frame
            motion_command.clip_id[env_ids] = slot
            motion_command.time_steps[env_ids] = frame
            motion_command.time_steps_f[env_ids] = float(frame)
            motion_command.speed_scale[env_ids] = 1.0

            root_state = robot.data.default_root_state[env_ids].detach().clone()
            root_state[:, :3] = (
                motion._body_pos_w[frame, 0].view(1, 3)
                + unwrapped.scene.env_origins[env_ids]
            )
            root_state[:, 3:7] = motion._body_quat_w[frame, 0].view(1, 4)
            root_state[:, 7:10] = motion._body_lin_vel_w[frame, 0].view(1, 3)
            root_state[:, 10:13] = motion._body_ang_vel_w[frame, 0].view(
                1, 3
            )
            joint_pos = motion.joint_pos[frame].view(1, -1)
            joint_vel = motion.joint_vel[frame].view(1, -1)
            robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            robot.write_joint_state_to_sim(
                joint_pos, joint_vel, env_ids=env_ids
            )
            unwrapped.scene.write_data_to_sim()
            raw_action = _raw_action_for_joint_target(action, joint_pos)

            (
                _obs,
                _reward,
                terminated,
                truncated,
                _extras,
            ) = env.step(raw_action)
            stepped += 1
            total_physics_steps += 4
            table_hit = bool(
                _term_mask(unwrapped, "robot_hit_table")[0].item()
            )
            fell = any(
                bool(_term_mask(unwrapped, name)[0].item())
                for name in ("base_fell_tilt", "base_too_low")
            )
            hard = any(
                bool(_term_mask(unwrapped, name)[0].item())
                for name in (
                    "joint_qdes_forbidden",
                    "joint_actual_forbidden",
                )
            )
            generic = bool(terminated[0].item()) or bool(truncated[0].item())
            table_count += int(table_hit)
            fall_count += int(fell)
            hard_count += int(hard)
            unsafe_count += int(table_hit or fell or hard or generic)

            latch = getattr(action, "_table_contact_latch", None)
            if (
                not generic
                and (
                    latch is None
                    or latch.finalized is not True
                    or bool(latch.hit[0].item())
                )
            ):
                _fail(
                    f"{motion_input.motion_id} frame {local_frame}: "
                    "four-substep table latch was not finalized cleanly"
                )
            if generic:
                break
        complete = (
            stepped == segment_length
            and table_count == 0
            and fall_count == 0
            and hard_count == 0
            and unsafe_count == 0
        )
        row = _RuntimeActionEvidence(
            motion_id=motion_input.motion_id,
            action_uid=motion_input.action_uid,
            motion_sha256=motion_input.file.sha256,
            frame_count=segment_length,
            physics_steps=4 * stepped,
            complete_cycle=complete,
            table_contact_count=table_count,
            fall_count=fall_count,
            hard_limit_count=hard_count,
            unsafe_count=unsafe_count,
            robot_body_contract_count=32,
        )
        results.append(row)
        if not complete:
            _fail(
                f"{motion_input.motion_id}: formal cycle unsafe/incomplete "
                f"(frames={stepped}/{segment_length}, table={table_count}, "
                f"fall={fall_count}, hard={hard_count}, unsafe={unsafe_count})"
            )
    _results["formal_action_set_motion_sweep"] = {
        "ordered_action_ids": list(
            inputs.action_set_contract["ordered_action_ids"]
        ),
        "motions": [
            {
                "motion_id": row.motion_id,
                "motion_sha256": row.motion_sha256,
                "frame_count": row.frame_count,
                "physics_steps": row.physics_steps,
                "complete_cycle": row.complete_cycle,
                "table_contact_count": row.table_contact_count,
                "fall_count": row.fall_count,
                "hard_limit_count": row.hard_limit_count,
                "unsafe_count": row.unsafe_count,
            }
            for row in results
        ],
        "physics_steps": total_physics_steps,
    }
    print(
        "ok formal motion sweep: exact trusted action-set bytes completed every "
        "frame with zero table/fall/hard/generic unsafe events"
    )
    return tuple(results), total_physics_steps


# Historical test/import alias; the implementation is exact-N.
sweep_formal_n5_actions = sweep_formal_actions


def bench(env, steps):
    """Step time for THIS arm.  One arm per process, on purpose.

    Isaac Sim does not reliably build a second ``ManagerBasedRLEnv`` in one process — the second
    ``gym.make`` hangs after "Parsing configuration" — so this measures the env that is already
    up and the CALLER runs the script twice, once with ``--table-obstacle on`` and once with
    ``off``, and subtracts.  Trying to do both arms in one process is what the first version did
    and it deadlocked for 30 minutes.
    """
    act = torch.zeros(env.unwrapped.num_envs,
                      env.unwrapped.action_manager.total_action_dim,
                      device=env.unwrapped.device)
    # Short probes intentionally finish before the known early-policy
    # raw-hard onset; long probes retain the historical 20-step warm-up.
    warmup_steps = min(20, max(4, steps // 5))
    for _ in range(warmup_steps):  # PhysX broadphase + CUDA graphs settle
        env.step(act)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / steps
    _results["bench"] = {
        "table_obstacle": bool(ARGS.table_obstacle == "on"),
        "num_envs": int(env.unwrapped.num_envs),
        "warmup_steps": int(warmup_steps),
        "steps": int(steps),
        "seconds_per_step": per_step,
        "ms_per_step": per_step * 1e3,
    }
    print(f"ok bench (table={ARGS.table_obstacle}, {env.unwrapped.num_envs} envs, {steps} steps): "
          f"{per_step*1e3:.3f} ms/step")


def _validate_manifest_with_runtime_loader(inputs: _FormalInputs) -> None:
    from whole_body_tracking.tasks.tracking.mdp.action_ball_manifest import (
        load_action_ball_manifest,
    )

    loaded = load_action_ball_manifest(
        inputs.manifest.path,
        expected_sha256=inputs.manifest.sha256,
        verify_referenced_assets=True,
        repo_root=inputs.repo_root,
        require_formal_admission=False,
    )
    manifest = loaded.manifest
    assets = loaded.referenced_assets
    if (
        tuple(manifest.action_order)
        != tuple(inputs.action_set_contract["ordered_action_ids"])
        or assets is None
        or len(assets.motions)
        != int(inputs.action_set_contract["expected_n"])
    ):
        raise TableSmokeReceiptError(
            "production ActionBall loader did not resolve the trusted "
            "action-set assets"
        )
    for expected, action, verified in zip(
        inputs.motions, manifest.actions, assets.motions
    ):
        if (
            action.action_id != expected.motion_id
            or action.motion_sha256 != expected.file.sha256
            or verified.resolved_path != expected.file.path
            or verified.sha256 != expected.file.sha256
        ):
            raise TableSmokeReceiptError(
                f"production ActionBall loader motion binding drifted at "
                f"{expected.motion_id!r}"
            )


def _configure_nominal_hold_cfg(
    cfg,
    inputs: _NominalHoldInput,
    *,
    duration_s: float,
) -> None:
    """Make one deterministic hold scene without weakening physical safety."""

    cfg.commands.motion.motion_file = str(inputs.motion_path)
    cfg.commands.motion.canonical_ready_mode = False
    cfg.commands.motion.balanced_clip_sampling = False
    rt = cfg.commands.racket_target
    rt.target_mode = "reference_perturbed"
    rt.virtual_ball = False
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        setattr(cfg.terminations, name, None)
    # Preserve the startup nominal-q capture, but make it deterministic.
    events = cfg.events
    events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)
    for name in (
        "physics_material",
        "base_com",
        "randomize_link_mass",
        "randomize_pd_gains",
    ):
        setattr(events, name, None)
    delay = inputs.expected_plant["control_step_action_delay"]
    action_cfg = cfg.actions.joint_pos
    action_cfg.control_step_action_delay_min = delay["min_steps"]
    action_cfg.control_step_action_delay_max = delay["max_steps"]
    hctrl = inputs.expected_plant.get("physx_control_position_limits")
    if hctrl is not None:
        # Reproduce the Vendor V1/V2 control plant, not the base ActionBall
        # task's historical five-percent guard / zero H_ctrl defaults.
        action_cfg.pre_apply_guard_brake_mode = (
            "max_inward_until_nonoutward_v1"
        )
        action_cfg.pre_apply_guard_margin_rad = 0.0
        action_cfg.pre_apply_guard_margin_fraction = 0.06
        action_cfg.physx_control_position_limit_inset_fraction = float(
            hctrl["inset_fraction_per_side_hard_span"]
        )
    cfg.episode_length_s = max(
        5.0, float(duration_s) + 2.0 * float(cfg.sim.dt) * int(cfg.decimation)
    )


def _cfg(
    formal_inputs: _FormalInputs | None = None,
    nominal_hold_inputs: _NominalHoldInput | None = None,
):
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    if formal_inputs is not None and nominal_hold_inputs is not None:
        raise TableSmokeReceiptError(
            "formal table smoke and nominal hold cannot share one cfg"
        )
    if formal_inputs is not None:
        terrain = getattr(getattr(cfg, "scene", None), "terrain", None)
        if (
            terrain is None
            or getattr(terrain, "terrain_type", None) != "plane"
        ):
            raise TableSmokeReceiptError(
                "formal action-set table smoke requires the flat plane terrain "
                "used by the first launch"
            )
        cfg.commands.motion.motion_file = tuple(
            str(row.file.path) for row in formal_inputs.motions
        )
        cfg.commands.motion.canonical_ready_mode = False
        cfg.commands.motion.clip_family_per_clip = tuple(
            row.family for row in formal_inputs.motions
        )
        cfg.commands.motion.balanced_clip_sampling = True
        cfg.commands.motion.balanced_clip_sampling_seed = 0
        rt = cfg.commands.racket_target
        # The smoke is an evaluator-owned safety replay, not an ActionBall
        # training/evaluator-authority launch.  Keep the registered ActionBall
        # scene/termination/action leaf, but use the inert reference target
        # producer while exact manifest motions are replayed frame-by-frame.
        rt.target_mode = "reference_perturbed"
        rt.clip_names_per_clip = tuple(
            formal_inputs.action_set_contract["ordered_action_ids"]
        )
        rt.strike_phase_per_clip = tuple(
            row.strike_phase for row in formal_inputs.motions
        )
        rt.mount_normal_sign_per_clip = tuple(
            row.mount_normal_sign for row in formal_inputs.motions
        )
        rt.action_ball_manifest_path = ""
        rt.action_ball_manifest_sha256 = ""
        rt.action_ball_policy_contract_sha256 = ""
        rt.action_ball_diagnostic_unauthorized = False
        cfg.episode_length_s = max(
            60.0,
            max(row.reference_t_cycle_s for row in formal_inputs.motions)
            + 5.0,
        )
    elif nominal_hold_inputs is not None:
        _configure_nominal_hold_cfg(
            cfg,
            nominal_hold_inputs,
            duration_s=float(ARGS.duration_s),
        )
    elif ARGS.motion_file:
        cfg.commands.motion.motion_file = ARGS.motion_file
    if (
        formal_inputs is not None
        or nominal_hold_inputs is not None
        or ARGS.contact_smoke
        or ARGS.bench
    ):
        cfg.seed = 0
    if formal_inputs is not None or ARGS.contact_smoke or ARGS.bench:
        # A pose-OBB positive control must not depend on which random
        # reference frame happened to be sampled at reset.  Start from the
        # shipped A3 stand and hold the reference at frame zero; every probe
        # then moves the articulation into contact explicitly below.
        cfg.commands.motion.stand_start_prob = 1.0
        cfg.commands.motion.hold_steps_range = (100, 100)
        cfg.commands.motion.stand_start_min_hold = 100
    if ARGS.bench:
        # Price the steady-state table backend, not random-reference reset
        # storms or their evidence archive.  Physical hard-limit/fall/table
        # terms remain active.
        for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
            setattr(cfg.terminations, name, None)
    # This script constructs a scene and reads geometry back; it never trains and never reads a
    # reward. The virtual-ball command refuses to build without a solved question bank because an
    # UNBANKED landing reward is anti-correlated with returning the ball — a training concern that
    # does not apply here. Opting out explicitly (rather than silently picking a task variant that
    # dodges the check) keeps the thing under test the LIVE lineage's env class.
    rt = getattr(cfg.commands, "racket_target", None)
    if rt is not None and hasattr(rt, "allow_unbanked_landing_rewards"):
        rt.allow_unbanked_landing_rewards = True
    if ARGS.table_obstacle == "off":
        cfg.table_obstacle = False
        from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
            apply_table_obstacle,
        )

        apply_table_obstacle(cfg)   # removes collider + termination + penalty together
    return cfg


def main():
    global ARGS
    ARGS = _parse()
    try:
        _validate_cli_mode(ARGS)
    except TableSmokeReceiptError as exc:
        _fail(str(exc))

    formal_inputs = None
    nominal_hold_inputs = None
    output_path = None
    nominal_hold_output_path = None
    nominal_hold_screenshot_dir = None
    source_commit = None
    runtime_source_baseline = None
    repo_root = _repository_root_from_producer()
    if ARGS.receipt_out is not None:
        try:
            formal_inputs = _load_formal_inputs(
                ARGS.manifest,
                action_set_profile=ARGS.action_set_profile,
                profile_pins_value=ARGS.profile_pins,
                expected_profile_pins_sha256=(
                    ARGS.profile_pins_sha256
                ),
                repo_root=repo_root,
            )
            output_path, _ = _prepare_output_path(
                ARGS.receipt_out, repo_root=formal_inputs.repo_root
            )
            source_commit = _committed_source_identity(formal_inputs)
        except (TableSmokeReceiptError, FileExistsError) as exc:
            _fail(str(exc))
    elif ARGS.nominal_hold is not None:
        try:
            nominal_hold_inputs = _load_nominal_hold_input(
                ARGS.nominal_hold,
                expected_sha256=ARGS.nominal_hold_sha256,
            )
            nominal_hold_output_path = _fresh_nominal_path(
                ARGS.nominal_hold_receipt_out,
                "nominal-hold receipt",
            )
            if ARGS.screenshot_dir is not None:
                nominal_hold_screenshot_dir = _fresh_nominal_path(
                    ARGS.screenshot_dir,
                    "nominal-hold screenshot directory",
                )
        except (
            TableSmokeReceiptError,
            FileExistsError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            _fail(str(exc))

    try:
        _initialize_isaac_runtime(ARGS)
        if formal_inputs is not None:
            _validate_manifest_with_runtime_loader(formal_inputs)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _fail(f"cannot initialize exact Isaac runtime: {exc}")

    env_cfg = _cfg(formal_inputs, nominal_hold_inputs)
    if formal_inputs is not None:
        runtime_source_baseline = _assert_runtime_source_closure(
            formal_inputs, str(source_commit)
        )
    if ARGS.contact_smoke and ARGS.cfg_only:
        _fail("--contact-smoke requires a constructed env; remove --cfg-only")
    if ARGS.contact_smoke and ARGS.table_obstacle == "off":
        _fail("--contact-smoke requires --table-obstacle on")
    if ARGS.table_obstacle == "off":
        # The no-table control arm: assert the removal is COMPLETE, not partial.
        for attr, where in (("table_obstacle", env_cfg.scene),
                            ("racket_table_contact", env_cfg.scene),
                            ("robot_hit_table", env_cfg.terminations),
                            ("table_hit_penalty", env_cfg.rewards)):
            if getattr(where, attr, None) is not None:
                _fail(f"--table-obstacle off left {attr} behind")
        print("ok cfg: no-table control arm — collider, sensor, termination and penalty all removed")
        _results["cfg"] = {"table_obstacle": False}
    else:
        check_cfg(env_cfg)
    env = None
    exit_code = 0
    try:
        if not ARGS.cfg_only:
            print("HOPE_TABLE_DIAGNOSTIC_STAGE=gym_make_begin", flush=True)
            if nominal_hold_screenshot_dir is not None:
                env = gym.make(
                    ARGS.task, cfg=env_cfg, render_mode="rgb_array"
                )
            else:
                env = gym.make(ARGS.task, cfg=env_cfg)
            print("HOPE_TABLE_DIAGNOSTIC_STAGE=gym_make_done", flush=True)
            env.reset()
            print("HOPE_TABLE_DIAGNOSTIC_STAGE=initial_reset_done", flush=True)
            if (
                ARGS.table_obstacle != "off"
                and nominal_hold_inputs is None
            ):
                check_spawned(env, env_cfg)
                print(
                    "HOPE_TABLE_DIAGNOSTIC_STAGE=spawn_check_done",
                    flush=True,
                )
            elif ARGS.table_obstacle != "off":
                # The hold probe exercises the live table/fall/hard-limit
                # termination terms while stepping.  Its question is the A3
                # ready state, not the separate full force-matrix receipt.
                # Avoid materializing the five 32-column Robot-body-filter matrices merely to
                # inspect a reset pose; the dedicated table smoke owns that runtime proof.
                print(
                    "HOPE_TABLE_DIAGNOSTIC_STAGE="
                    "spawn_check_skipped_for_nominal_hold",
                    flush=True,
                )
            if nominal_hold_inputs is not None:
                print(
                    "HOPE_TABLE_DIAGNOSTIC_STAGE=nominal_hold_begin",
                    flush=True,
                )
                receipt = nominal_hold_probe(
                    env,
                    env_cfg,
                    nominal_hold_inputs,
                    duration_s=float(ARGS.duration_s),
                    screenshot_dir=nominal_hold_screenshot_dir,
                )
                print(
                    "HOPE_TABLE_DIAGNOSTIC_STAGE=nominal_hold_done",
                    flush=True,
                )
                assert nominal_hold_output_path is not None
                receipt_sha = _exclusive_publish_nominal_hold_receipt(
                    nominal_hold_output_path, receipt
                )
                _results["nominal_hold_receipt"] = {
                    "path": str(nominal_hold_output_path),
                    "sha256": receipt_sha,
                    "content_sha256": receipt["content_sha256"],
                    "verdict": receipt["verdict"],
                }
                print(
                    "HOPE_ISAAC_NOMINAL_HOLD_RECEIPT="
                    + json.dumps(
                        _results["nominal_hold_receipt"],
                        sort_keys=True,
                    )
                )
                if receipt["verdict"] != "PASS":
                    exit_code = 2
            elif formal_inputs is not None:
                action_rows, action_physics_steps = sweep_formal_n5_actions(
                    env, env_cfg, formal_inputs
                )
                contact_physics_steps = contact_smoke(env, env_cfg)
                contact = _results["contact_smoke"]
                # Isaac is already up at this point, so the guard module is
                # safe to import; read its pin instead of copying the number.
                from whole_body_tracking.tasks.tracking.mdp.terminations import (
                    _A3_COLLISION_PROXY_COMPONENT_COUNT
                    as A3_TABLE_GUARD_COMPONENT_COUNT,
                )

                probes = contact["probes"]
                cfg_result = _results["cfg"]
                spawned = _results["spawned"]
                pulse_substeps = {
                    int(row["pulse_substep"]) for row in probes
                }
                probed_roles = {str(row["role"]) for row in probes}
                runtime_evidence = _RuntimeEvidence(
                    origin=_ISAAC_RUNTIME_ORIGIN,
                    source_commit_sha=str(source_commit),
                    isaac_version=_isaac_version_identity(),
                    python_executable=sys.executable,
                    gpu_identity=_gpu_identity(),
                    physics_steps=(
                        action_physics_steps + contact_physics_steps
                    ),
                    actions=action_rows,
                    pose_obb_guard_pass=(
                        contact.get("pose_obb_guard_pass") is True
                        and contact.get("nonfinite_fail_safe") is True
                    ),
                    full_action_ball_assembly=(
                        cfg_result.get("mode") == "full_action_ball"
                        and spawned.get("mode") == "full_action_ball"
                    ),
                    all_five_table_components_with_pose_obb=(
                        cfg_result.get("full_table_pose_obb_guard", {}).get(
                            "pair_filtered_sensor_count"
                        )
                        == 0
                        and spawned.get(
                            "full_table_pose_obb_guard", {}
                        ).get("collision_component_count")
                        == A3_TABLE_GUARD_COMPONENT_COUNT
                        and spawned.get(
                            "full_table_pose_obb_guard", {}
                        ).get("obstacle_count")
                        == 5
                        and set(
                            spawned.get(
                                "pose_guard_robot_body_names", ()
                            )
                        )
                        == set(TABLE_CONTACT_BODY_NAMES)
                    ),
                    all_five_obstacles=(
                        len(cfg_result.get("components", ())) == 5
                        and len(spawned.get("components", ())) == 5
                        and set(tt_frame.TABLE_ASSEMBLY_ROLES).issubset(
                            probed_roles
                        )
                    ),
                    all_four_substeps=(pulse_substeps == {1, 2, 3, 4}),
                    positive_control_pass=(
                        bool(probes)
                        and all(
                            row.get("raw_robot_hit_table") is True
                            and row.get("generic_terminated") is True
                            for row in probes
                        )
                    ),
                    negative_control_pass=(
                        contact.get("legal_stance_negative") is True
                        and contact.get("zero_pulse_after_reset") is True
                    ),
                    zero_reset_leakage=(
                        bool(probes)
                        and all(
                            row.get("selected_reset_zero_leak") is True
                            for row in probes
                        )
                    ),
                )
                _assert_formal_inputs_unchanged(formal_inputs)
                final_runtime_sources = _assert_runtime_source_closure(
                    formal_inputs,
                    str(source_commit),
                    baseline=runtime_source_baseline,
                )
                if _committed_source_identity(formal_inputs) != source_commit:
                    _fail("producer Git source commit changed during PhysX run")
                _results["runtime_source_closure"] = {
                    "source_commit_sha": source_commit,
                    "module_count": len(final_runtime_sources),
                    "module_paths": sorted(
                        {
                            snapshot.repo_path
                            for snapshot in final_runtime_sources.values()
                        }
                    ),
                }
                receipt = _build_formal_receipt(
                    formal_inputs, runtime_evidence
                )
                assert output_path is not None
                receipt_sha = _exclusive_publish_receipt(
                    output_path, receipt
                )
                _results["formal_receipt"] = {
                    "path": output_path.relative_to(
                        formal_inputs.repo_root
                    ).as_posix(),
                    "sha256": receipt_sha,
                    "receipt_payload_sha256": receipt[
                        "receipt_payload_sha256"
                    ],
                }
                print(
                    "HOPE_ISAAC_TABLE_FILTERED_SMOKE_RECEIPT="
                    + json.dumps(
                        _results["formal_receipt"], sort_keys=True
                    )
                )
            elif ARGS.contact_smoke:
                contact_smoke(env, env_cfg)
            if ARGS.bench:
                bench(env, ARGS.bench)
            # Print the authoritative completion payload before closing Kit.
            # Isaac Sim 4.5 may terminate the process from ``SimulationApp.close``
            # with status zero, which would otherwise hide both successful
            # evidence and Python failures raised earlier in the diagnostic.
            print(
                "HOPE_TABLE_OBSTACLE_CHECK_JSON="
                + json.dumps(_results, sort_keys=True),
                flush=True,
            )
            print(
                "HOPE_TABLE_DIAGNOSTIC_STAGE=main_completed",
                flush=True,
            )
    finally:
        # Isaac Sim 4.5's ``SimulationApp.close()`` may hard-exit with status zero.  Calling it
        # while a SystemExit/exception or explicit nonzero verdict is active would counterfeit a
        # failed diagnostic as PASS at the process boundary.  On failure leave teardown to the OS;
        # successful runs still perform the normal ordered env/app close.
        failure_active = sys.exc_info()[0] is not None or exit_code != 0
        if not failure_active:
            if env is not None:
                env.close()
            if _app is not None:
                # Kit may terminate the process inside ``close``.  Flush the
                # already published success payload before transferring
                # teardown control to it.
                _flush_process_streams()
                _app.close()
    return exit_code


def _flush_process_streams() -> None:
    """Best-effort flush before an explicit process-boundary verdict."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:
            # A broken diagnostic stream must not replace the already selected
            # process verdict or let Kit's destructor rewrite it to zero.
            pass


def _entrypoint() -> None:
    """Publish one shell-visible verdict even if Kit owns process teardown."""

    try:
        exit_code = main()
    except BaseException as exc:
        try:
            traceback.print_exc()
        except BaseException:
            pass
        _flush_process_streams()
        if isinstance(exc, KeyboardInterrupt):
            failure_code = 130
        elif isinstance(exc, SystemExit) and isinstance(exc.code, int):
            # SystemExit(0) before main returns is not a completed diagnostic.
            failure_code = int(exc.code) if int(exc.code) != 0 else 1
        else:
            failure_code = 1
        os._exit(failure_code)

    _flush_process_streams()
    os._exit(int(exit_code))


if __name__ == "__main__":
    _entrypoint()
