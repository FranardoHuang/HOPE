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
NOMINAL_HOLD_ARTIFACT_KIND = "agibot_a3_action_dynamic_ready_candidate_v1"
NOMINAL_HOLD_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
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
    physical_root_pos: tuple[float, ...]
    physical_root_quat: tuple[float, ...]
    physical_joint_pos: tuple[float, ...]
    hold_qdes: tuple[float, ...]
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


def _pinned_external_file(
    value: object, expected_sha256: object, label: str
) -> tuple[Path, str]:
    path = _assert_plain_components(Path(str(value)), label)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _require_sha256(expected_sha256, f"{label} SHA-256"):
        raise TableSmokeReceiptError(f"{label} SHA-256 mismatch")
    return path, digest


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
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != content_sha:
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
    if (
        document.get("schema_version") != 1
        or robot.get("family") != "AgiBot A3"
        or len(names) != 31
        or len(set(names)) != 31
        or not isinstance(action_id, str)
        or not action_id
    ):
        raise TableSmokeReceiptError("dynamic-ready is not exact A3 N=1")
    motion_path, motion_sha = _pinned_external_file(
        motion["path"], motion["sha256"], "stable motion"
    )
    vectors = {
        out: _finite_tuple(runtime[source], 31, source)
        for out, source in (
            ("joint_stiffness", "joint_stiffness"),
            ("joint_damping", "joint_damping"),
            ("joint_effort_limits", "joint_effort_limits"),
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
    expected_plant = {
        "joint_names": names,
        **vectors,
        "qdes_joint_pos_limits": limits,
        "finite_projection_soft_envelope_inset_fraction": float(
            runtime["finite_projection_soft_envelope_inset_fraction"]
        ),
        "physics_step_dt_s": float(runtime["physics_step_dt_s"]),
        "policy_step_dt_s": float(runtime["policy_step_dt_s"]),
        "control_decimation": int(runtime["control_decimation"]),
    }
    root_quat = _finite_tuple(
        physical["root_quat_wxyz"], 4, "root quaternion"
    )
    hold_qdes = _finite_tuple(
        hold["hold_qdes_joint_pos_rad"], 31, "hold q_des"
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
        physical_root_pos=_finite_tuple(
            physical["root_pos_w_m"], 3, "root position"
        ),
        physical_root_quat=root_quat,
        physical_joint_pos=_finite_tuple(
            physical["joint_pos_rad"], 31, "ready q"
        ),
        hold_qdes=hold_qdes,
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
    payload_source_map = solver_payload.get(
        "implementation_source_sha256"
    )
    if (
        not isinstance(source_map, Mapping)
        or not isinstance(payload_source_map, Mapping)
        or dict(source_map) != dict(payload_source_map)
        or tuple(sorted(source_map))
        != tuple(sorted(_ACTION_BALL_SOLVER_SOURCE_NAMES))
    ):
        raise TableSmokeReceiptError(
            "ActionBall profile pins must bind the exact five solver sources"
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
            or int(prepared._component_indices.shape[0]) != 43
            or int(prepared._aabb_lo.shape[0]) != 5
        ):
            _fail(
                "spawned ActionBall pose guard is not fully prepared against "
                "the live USD/articulation/43-component/5-obstacle contract"
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
    params = action._resolved_table_contact_params()
    prepared = getattr(action, "_table_contact_prepared_pose_guard", None)
    if (
        params.get("full_table_assembly") is not True
        or prepared is None
        or int(prepared._component_indices.shape[0]) != 43
        or int(prepared._aabb_lo.shape[0]) != 5
    ):
        _fail(
            "contact smoke pose guard was not prepared against 43 collision "
            "components and five table boxes"
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
            if apply_index == pulse_substep - 1:
                move_root(contact_root_pose)
            original_apply()
            # Restore only after the preceding substep was sampled.  Substep four has no fifth
            # apply call; automatic terminal reset restores it after the DoneTerm samples it.
            if apply_index == pulse_substep:
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
        inputs.physical_joint_pos,
        dtype=motion.joint_pos.dtype,
        device=motion.joint_pos.device,
    )
    if not bool(torch.allclose(motion.joint_pos[0], expected_q, rtol=0.0, atol=1e-6)):
        _fail("dynamic-ready physical q is not motion frame zero")


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
        if key == "joint_names":
            matched = tuple(actual or ()) == tuple(expected)
        elif key == "control_decimation":
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
    hold_qdes = torch.tensor(
        inputs.hold_qdes, device=unwrapped.device, dtype=ready_q.dtype
    ).view(1, 31)
    raw_action = _raw_action_for_joint_target(action, hold_qdes)

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
    for step in range(1, requested_steps + 1):
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
    passed = (
        completed == requested_steps
        and not terminated_value
        and not truncated_value
        and not terminal_reasons
        and finite_roots
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
        "plant_contract_match": True,
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
    if formal_inputs is not None or ARGS.contact_smoke or ARGS.bench:
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
                        == 43
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
