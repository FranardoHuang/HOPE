#!/usr/bin/env python3
"""Materialize and validate the code-owned A3 vendor runtime authority.

The ActionBall dynamic-ready candidate is derived from a live schema-3
training contract.  Treating the contract SHA written inside an arbitrary
candidate as authority is circular: an operator could point a spec at the old
plant and repeat the old SHA.  This module closes that gap with one fixed,
tracked receipt that binds:

* the exact source commit used to construct the live runtime;
* the immutable vendor task profile and the robot actuator source;
* train.py, training_contract.py, and hope_actions.py;
* the stable-v2 bh_loop_c motion; and
* the real Pod-produced schema-3 training-contract bytes.

The producer is dependency-light and writes canonical JSON with O_EXCL.  The
validator is intentionally importable by launchers.  It reopens every source
blob from both the authority commit and the launch checkout's current HEAD,
so a later artifact-only commit is allowed while any scientific-source drift
is rejected before Kit starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
KIND = "agibot_a3_vendor_runtime_authority_v1"

RECEIPT_REPO_PATH = (
    "configs/a3_vendor_runtime_authority_20260731/"
    "bh_loop_c.vendor_runtime_authority.v1.json"
)
RUNTIME_CONTRACT_REPO_PATH = (
    "configs/a3_vendor_runtime_authority_20260731/"
    "bh_loop_c.shared_ready.training_contract.json"
)
VENDOR_TASK_REPO_PATH = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV1.yaml"
)
ACTION_BALL_TASK_REPO_PATH = (
    "hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBall.yaml"
)
HITTER_TASK_REPO_PATH = (
    "hope_training/whole_body_tracking/cfg/task/HOPEPingPongHitter.yaml"
)
ENV_BASE_REPO_PATH = "hope_training/whole_body_tracking/cfg/base/env_base.yaml"
SIM_BASE_REPO_PATH = "hope_training/whole_body_tracking/cfg/base/sim_base.yaml"
RANDOMIZATION_BASE_REPO_PATH = (
    "hope_training/whole_body_tracking/cfg/base/randomization_base.yaml"
)
ROBOT_SOURCE_REPO_PATH = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
ENV_CFG_SOURCE_REPO_PATH = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
TRAIN_SOURCE_REPO_PATH = "hope_training/whole_body_tracking/scripts/train.py"
TRAINING_CONTRACT_SOURCE_REPO_PATH = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py"
)
HOPE_ACTIONS_SOURCE_REPO_PATH = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_actions.py"
)
RUNNER_SOURCE_REPO_PATH = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/my_on_policy_runner.py"
)
STABLE_MOTION_REPO_PATH = (
    "assets/motions/fivebind_20260727/"
    "bh_loop_c_upper_stable_v2.npz"
)

SOURCE_PATHS = {
    "vendor_task_profile": VENDOR_TASK_REPO_PATH,
    "action_ball_task_profile": ACTION_BALL_TASK_REPO_PATH,
    "hitter_task_profile": HITTER_TASK_REPO_PATH,
    "environment_base_profile": ENV_BASE_REPO_PATH,
    "simulation_base_profile": SIM_BASE_REPO_PATH,
    "randomization_base_profile": RANDOMIZATION_BASE_REPO_PATH,
    "robot_actuator_source": ROBOT_SOURCE_REPO_PATH,
    "environment_config_source": ENV_CFG_SOURCE_REPO_PATH,
    "training_entrypoint": TRAIN_SOURCE_REPO_PATH,
    "training_contract_source": TRAINING_CONTRACT_SOURCE_REPO_PATH,
    "action_source": HOPE_ACTIONS_SOURCE_REPO_PATH,
    "runner_source": RUNNER_SOURCE_REPO_PATH,
    "stable_motion": STABLE_MOTION_REPO_PATH,
}
# Schema-3 serializes ``robot.data.joint_names`` and requires the joint-pos
# action term to resolve identity ids ``range(31)``.  This is therefore the
# actual A3 USD articulation/action order observed at runtime, not
# ``AGIBOT_A3_JOINT_NAMES`` (the latter is only the retargeted CSV/controller
# column order).  Keep the interleaved tree order exact and fail closed on a
# contract that merely carries the same joint-name set in logical groups.
RUNTIME_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "source_commit",
        "sources",
        "runtime_training_contract",
        "runtime_plant_identity",
        "verified_vendor_runtime",
        "authorization",
        "producer",
        "content_sha256",
    }
)
_SOURCE_KEYS = frozenset({"path", "sha256"})
_CONTRACT_KEYS = frozenset({"path", "sha256", "schema_version"})
_AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    }
)
_VENDOR_RUNTIME_KEYS = frozenset(
    {
        "action_id",
        "motion_sha256",
        "joint_count",
        "vendor_joint_values",
        "control_step_action_delay",
        "push_robot_event",
    }
)
_PRODUCER_KEYS = frozenset({"path", "sha256"})
_RUNTIME_PLANT_IDENTITY_KEYS = frozenset(
    {
        "joint_names",
        "articulation_joint_names",
        "action_joint_ids",
        "joint_stiffness",
        "joint_damping",
        "joint_effort_limits",
        "joint_velocity_limits",
        "joint_armature",
        "default_joint_pos_rad",
        "action_scale_rad",
        "qdes_joint_pos_limits",
        "physics_step_dt_s",
        "policy_step_dt_s",
        "control_decimation",
        "control_step_action_delay",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class VendorRuntimeAuthorityError(RuntimeError):
    """The vendor runtime authority cannot be produced or validated."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise VendorRuntimeAuthorityError(
            f"{name} must be 64 lowercase SHA-256 digits"
        )
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise VendorRuntimeAuthorityError(
            "authority document is not finite canonical JSON"
        ) from exc


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise VendorRuntimeAuthorityError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise VendorRuntimeAuthorityError(
                f"{name} contains non-finite float {token!r}"
            )
        return value

    def reject_constant(token: str):
        raise VendorRuntimeAuthorityError(
            f"{name} contains non-finite token {token!r}"
        )

    try:
        decoded = payload.decode("utf-8", "strict")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VendorRuntimeAuthorityError(
            f"{name} is not strict UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise VendorRuntimeAuthorityError(f"{name} must be one JSON object")
    return value


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> dict:
    if type(value) is not dict or frozenset(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise VendorRuntimeAuthorityError(
            f"{name} keys differ: expected={sorted(keys)!r}, actual={actual!r}"
        )
    return value


def _git(
    repo_root: Path, arguments: Sequence[str], *, text: bool = False
) -> bytes | str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=text,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise VendorRuntimeAuthorityError(
            f"git {' '.join(arguments)} failed: {str(detail).strip()}"
        ) from exc


def _resolve_commit(repo_root: Path, value: str) -> str:
    result = str(
        _git(repo_root, ["rev-parse", "--verify", f"{value}^{{commit}}"], text=True)
    ).strip()
    if _COMMIT_RE.fullmatch(result) is None:
        raise VendorRuntimeAuthorityError(
            f"source commit did not resolve to one full commit: {result!r}"
        )
    return result


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return bytes(_git(repo_root, ["show", f"{commit}:{relative}"]))


def _repo_file(repo_root: Path, relative: str, *, name: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise VendorRuntimeAuthorityError(
            f"{name} path is not normalized repo-relative POSIX"
        )
    requested = repo_root.joinpath(*pure.parts)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise VendorRuntimeAuthorityError(f"cannot resolve {name}: {exc}") from exc
    if requested != resolved or requested.is_symlink() or not resolved.is_file():
        raise VendorRuntimeAuthorityError(
            f"{name} must be one regular file without symlink components"
        )
    return resolved


def _expected_pin(
    repo_root: Path,
    source_commit: str,
    relative: str,
    expected_sha256: object,
    *,
    name: str,
) -> dict[str, str]:
    expected = _require_sha256(expected_sha256, name=f"expected {name} SHA-256")
    committed = _git_blob(repo_root, source_commit, relative)
    committed_sha = _sha256_bytes(committed)
    current = _repo_file(repo_root, relative, name=name)
    current_sha = _sha256_file(current)
    if committed_sha != expected or current_sha != expected:
        raise VendorRuntimeAuthorityError(
            f"{name} SHA mismatch: expected={expected}, "
            f"commit={committed_sha}, worktree={current_sha}"
        )
    return {"path": relative, "sha256": expected}


def _plain_vector(value: object, *, name: str, size: int) -> list[float]:
    if type(value) is not list or len(value) != size:
        raise VendorRuntimeAuthorityError(
            f"{name} must contain exactly {size} entries"
        )
    result = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise VendorRuntimeAuthorityError(
                f"{name} must contain plain finite numbers"
            )
        result.append(float(item))
    return result


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=2.0e-6)


def _plain_matrix(
    value: object, *, name: str, rows: int, columns: int
) -> list[list[float]]:
    if type(value) is not list or len(value) != rows:
        raise VendorRuntimeAuthorityError(
            f"{name} must contain exactly {rows} rows"
        )
    return [
        _plain_vector(row, name=f"{name}[{index}]", size=columns)
        for index, row in enumerate(value)
    ]


def _finite_positive_scalar(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise VendorRuntimeAuthorityError(f"{name} must be one positive finite number")
    return float(value)


def _canonical_runtime_plant_identity(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the exact live-plant subset used by the dynamic artifact.

    This is deliberately duplicated into the authority receipt.  A candidate
    cannot become authoritative merely by repeating the receipt's contract
    SHA: its host-readable ``runtime_plant`` must reproduce these values.
    """

    if contract.get("schema_version") != 3:
        raise VendorRuntimeAuthorityError(
            "runtime training contract must use schema_version=3"
        )
    names = contract.get("joint_names")
    articulation_names = contract.get("articulation_joint_names")
    action_joint_ids = contract.get("action_joint_ids")
    if (
        type(names) is not list
        or names != list(RUNTIME_JOINT_NAMES)
        or articulation_names != names
        or action_joint_ids != list(range(31))
    ):
        raise VendorRuntimeAuthorityError(
            "runtime contract does not bind the exact 31-joint action order"
        )

    vectors = {
        key: _plain_vector(contract.get(key), name=key, size=31)
        for key in (
            "joint_stiffness",
            "joint_damping",
            "joint_effort_limits",
            "joint_velocity_limits",
            "joint_armature",
        )
    }
    default_q = _plain_vector(
        contract.get("default_joint_pos"), name="default_joint_pos", size=31
    )
    action_scale = _plain_vector(
        contract.get("action_scale"), name="action_scale", size=31
    )
    qdes_limits = _plain_matrix(
        contract.get("qdes_joint_pos_limits"),
        name="qdes_joint_pos_limits",
        rows=31,
        columns=2,
    )
    if (
        any(value <= 0.0 for value in vectors["joint_stiffness"])
        or any(value < 0.0 for value in vectors["joint_damping"])
        or any(value <= 0.0 for value in vectors["joint_effort_limits"])
        or any(value <= 0.0 for value in vectors["joint_velocity_limits"])
        or any(value < 0.0 for value in vectors["joint_armature"])
        or any(value <= 0.0 for value in action_scale)
        or any(lower >= upper for lower, upper in qdes_limits)
    ):
        raise VendorRuntimeAuthorityError(
            "runtime contract plant vectors contain an invalid limit/value"
        )
    physics_dt = _finite_positive_scalar(
        contract.get("physics_step_dt_s"), name="physics_step_dt_s"
    )
    policy_dt = _finite_positive_scalar(
        contract.get("policy_step_dt_s"), name="policy_step_dt_s"
    )
    decimation = contract.get("control_decimation")
    if (
        isinstance(decimation, bool)
        or type(decimation) is not int
        or decimation <= 0
        or not math.isclose(
            policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise VendorRuntimeAuthorityError(
            "runtime contract policy/physics timing is inconsistent"
        )
    delay = contract.get("control_step_action_delay")
    expected_delay = {
        "schema_version": 1,
        "enabled": True,
        "semantic_unit": "policy_control_step",
        "sample_timing": "once_per_episode_reset",
        "distribution": "discrete_uniform_inclusive",
        "min_steps": 0,
        "max_steps": 2,
        "shared_across_all_31_joints": True,
        "history_fill": "safe_default_or_action_specific_hold",
    }
    if delay != expected_delay:
        raise VendorRuntimeAuthorityError(
            "runtime contract does not contain the exact vendor [0,2] "
            "control-step action delay"
        )
    return {
        "joint_names": list(names),
        "articulation_joint_names": list(articulation_names),
        "action_joint_ids": list(action_joint_ids),
        **vectors,
        "default_joint_pos_rad": default_q,
        "action_scale_rad": action_scale,
        "qdes_joint_pos_limits": qdes_limits,
        "physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "control_decimation": decimation,
        "control_step_action_delay": expected_delay,
    }


def _verified_vendor_runtime(
    contract: Mapping[str, Any], *, stable_motion_sha256: str
) -> dict[str, Any]:
    if contract.get("schema_version") != 3 or contract.get("target_mode") != "action_ball":
        raise VendorRuntimeAuthorityError(
            "runtime training contract must be schema-3 ActionBall"
        )
    joint_names = contract.get("joint_names")
    if (
        type(joint_names) is not list
        or joint_names != list(RUNTIME_JOINT_NAMES)
        or contract.get("articulation_joint_names") != joint_names
        or contract.get("action_joint_ids") != list(range(31))
    ):
        raise VendorRuntimeAuthorityError(
            "runtime contract does not bind the exact 31-joint articulation order"
        )
    fields = {
        "joint_stiffness": _plain_vector(
            contract.get("joint_stiffness"), name="joint_stiffness", size=31
        ),
        "joint_damping": _plain_vector(
            contract.get("joint_damping"), name="joint_damping", size=31
        ),
        "joint_effort_limits": _plain_vector(
            contract.get("joint_effort_limits"),
            name="joint_effort_limits",
            size=31,
        ),
        "joint_armature": _plain_vector(
            contract.get("joint_armature"), name="joint_armature", size=31
        ),
        "action_scale": _plain_vector(
            contract.get("action_scale"), name="action_scale", size=31
        ),
    }
    expected_by_group = {
        "waist_yaw_joint": {
            "joint_stiffness": 80.0,
            "joint_damping": 3.0,
            "joint_effort_limits": 220.0,
            "joint_armature": 0.066472,
            "action_scale": 0.6875,
        },
        "waist_pitch_joint": {
            "joint_stiffness": 50.0,
            "joint_damping": 2.0,
            "joint_effort_limits": 115.0,
            "joint_armature": 0.088220,
            "action_scale": 0.575,
        },
        "waist_roll_joint": {
            "joint_stiffness": 50.0,
            "joint_damping": 2.0,
            "joint_effort_limits": 46.0,
            "joint_armature": 0.014623,
            "action_scale": 0.23,
        },
        "head": {
            "joint_stiffness": 40.0,
            "joint_damping": 2.0,
            "joint_effort_limits": 6.0,
            "joint_armature": 0.0008100893338,
            "action_scale": 0.0375,
        },
        "hip_pitch_yaw": {
            "joint_stiffness": 80.0,
            "joint_damping": 3.0,
            "joint_effort_limits": 220.0,
            "joint_armature": 0.066472,
            "action_scale": 0.6875,
        },
        "hip_roll": {
            "joint_stiffness": 120.0,
            "joint_damping": 4.0,
            "joint_effort_limits": 220.0,
            "joint_armature": 0.066472,
            "action_scale": 0.4583333333333333,
        },
        "knee": {
            "joint_stiffness": 250.0,
            "joint_damping": 8.0,
            "joint_effort_limits": 320.0,
            "joint_armature": 0.120340,
            "action_scale": 0.32,
        },
        "ankle_pitch": {
            "joint_stiffness": 50.0,
            "joint_damping": 2.0,
            # Schema-3 records the instantiated float32 tensor.  This is the
            # exact binary32 representation of the vendor value 118.2, not a
            # widened tolerance or a different limit.
            "joint_effort_limits": 118.19999694824219,
            "joint_armature": 0.064449,
            "action_scale": 0.591,
        },
        "ankle_roll": {
            "joint_stiffness": 50.0,
            "joint_damping": 2.0,
            "joint_effort_limits": 54.75,
            "joint_armature": 0.020129,
            "action_scale": 0.27375,
        },
        "shoulder_pitch_roll": {
            "joint_stiffness": 40.0,
            "joint_damping": 3.0,
            "joint_effort_limits": 60.0,
            "joint_armature": 0.012085,
            "action_scale": 0.375,
        },
        "distal_arm": {
            "joint_stiffness": 30.0,
            "joint_damping": 2.0,
            "joint_effort_limits": 24.0,
            "joint_armature": 0.004968,
            "action_scale": 0.2,
        },
    }
    expected_by_joint: dict[str, dict[str, float]] = {}
    for joint in joint_names:
        if joint in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
            group = joint
        elif joint in ("head_yaw_joint", "head_pitch_joint"):
            group = "head"
        elif joint.endswith(("_hip_pitch_joint", "_hip_yaw_joint")):
            group = "hip_pitch_yaw"
        elif joint.endswith("_hip_roll_joint"):
            group = "hip_roll"
        elif joint.endswith("_knee_joint"):
            group = "knee"
        elif joint.endswith("_ankle_pitch_joint"):
            group = "ankle_pitch"
        elif joint.endswith("_ankle_roll_joint"):
            group = "ankle_roll"
        elif joint.endswith(("_shoulder_pitch_joint", "_shoulder_roll_joint")):
            group = "shoulder_pitch_roll"
        elif joint.endswith(
            (
                "_shoulder_yaw_joint",
                "_elbow_joint",
                "_wrist_roll_joint",
                "_wrist_pitch_joint",
                "_wrist_yaw_joint",
            )
        ):
            group = "distal_arm"
        else:
            raise VendorRuntimeAuthorityError(
                f"runtime contract has an unknown A3 joint {joint!r}"
            )
        expected_by_joint[joint] = expected_by_group[group]
    verified_joint_values: dict[str, dict[str, float]] = {}
    for joint, expected_fields in expected_by_joint.items():
        try:
            index = joint_names.index(joint)
        except ValueError as exc:
            raise VendorRuntimeAuthorityError(
                f"runtime contract lacks vendor joint {joint!r}"
            ) from exc
        actual_fields = {
            field: values[index] for field, values in fields.items()
        }
        for field, expected in expected_fields.items():
            if not _close(actual_fields[field], expected):
                raise VendorRuntimeAuthorityError(
                    f"runtime contract {joint}.{field}={actual_fields[field]!r} "
                    f"does not match vendor value {expected!r}"
                )
        verified_joint_values[joint] = actual_fields

    delay = contract.get("control_step_action_delay")
    expected_delay = {
        "schema_version": 1,
        "enabled": True,
        "semantic_unit": "policy_control_step",
        "sample_timing": "once_per_episode_reset",
        "distribution": "discrete_uniform_inclusive",
        "min_steps": 0,
        "max_steps": 2,
        "shared_across_all_31_joints": True,
        "history_fill": "safe_default_or_action_specific_hold",
    }
    if delay != expected_delay:
        raise VendorRuntimeAuthorityError(
            "runtime contract does not contain the exact vendor [0,2] "
            "control-step action delay"
        )

    push = contract.get("push_robot_event")
    expected_velocity = {
        "x": [-0.25, 0.25],
        "y": [-0.25, 0.25],
        "z": [-0.1, 0.1],
        "roll": [-0.26, 0.26],
        "pitch": [-0.26, 0.26],
        "yaw": [-0.39, 0.39],
    }
    expected_push = {
        "schema_version": 2,
        "enabled": True,
        "semantics": "symmetric_6d_velocity_delta",
        "func": "push_by_setting_velocity",
        "mode": "interval",
        "interval_range_s": [5.0, 15.0],
        "velocity_range": expected_velocity,
    }
    if push != expected_push:
        raise VendorRuntimeAuthorityError(
            "runtime contract does not contain the exact vendor-amplitude "
            "ActionBall 6-DoF push event"
        )

    try:
        action_ball = contract["action_ball_training"]
        preflight = action_ball["preflight"]
        bootstrap = action_ball["policy_bootstrap"]
        admission = action_ball["motion_admission"]
    except (KeyError, TypeError) as exc:
        raise VendorRuntimeAuthorityError(
            "runtime contract lacks its ActionBall bootstrap lineage"
        ) from exc
    if (
        type(preflight) is not dict
        or preflight.get("action_order") != ["bh_loop_c"]
        or type(bootstrap) is not dict
        or bootstrap.get("schema_version") != 1
        or bootstrap.get("kind") != "action_ball_shared_ready_actor_bootstrap_v1"
        or bootstrap.get("action_order") != ["bh_loop_c"]
        or type(admission) is not dict
        or admission.get("motion_file_sha256") != [stable_motion_sha256]
    ):
        raise VendorRuntimeAuthorityError(
            "runtime contract is not the exact bh_loop_c shared-ready bootstrap "
            "used to derive dynamic-ready"
        )
    return {
        "action_id": "bh_loop_c",
        "motion_sha256": stable_motion_sha256,
        "joint_count": 31,
        "vendor_joint_values": verified_joint_values,
        "control_step_action_delay": expected_delay,
        "push_robot_event": push,
    }


def _stable_contract_file(
    repo_root: Path, path_value: str | Path, expected_sha256: object
) -> tuple[Path, str, dict[str, Any]]:
    expected_path = repo_root.joinpath(
        *PurePosixPath(RUNTIME_CONTRACT_REPO_PATH).parts
    )
    requested = Path(path_value).expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise VendorRuntimeAuthorityError(
            f"cannot resolve runtime training contract: {exc}"
        ) from exc
    if (
        requested != resolved
        or resolved != expected_path
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise VendorRuntimeAuthorityError(
            "runtime training contract must use the fixed tracked path "
            f"{RUNTIME_CONTRACT_REPO_PATH}"
        )
    expected = _require_sha256(
        expected_sha256, name="expected runtime training contract SHA-256"
    )
    payload = resolved.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise VendorRuntimeAuthorityError(
            f"runtime training contract SHA mismatch: {actual} != {expected}"
        )
    return resolved, actual, _strict_json_bytes(
        payload, name="runtime training contract"
    )


def _write_exclusive(path_value: str | Path, payload: bytes) -> Path:
    requested = Path(path_value).expanduser().absolute()
    parent_input = requested.parent
    try:
        parent = parent_input.resolve(strict=True)
    except OSError as exc:
        raise VendorRuntimeAuthorityError(
            f"cannot resolve authority output parent: {exc}"
        ) from exc
    if parent_input != parent or not parent.is_dir() or not requested.name:
        raise VendorRuntimeAuthorityError(
            "authority output must have one leaf below an existing real directory"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_fd = os.open(parent, parent_flags)
    try:
        fd = os.open(requested.name, flags, 0o444, dir_fd=parent_fd)
        try:
            written = 0
            view = memoryview(payload)
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise OSError("exclusive write made no progress")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise VendorRuntimeAuthorityError(
            f"cannot publish no-clobber authority receipt: {exc}"
        ) from exc
    finally:
        os.close(parent_fd)
    return requested


def materialize_vendor_runtime_authority(
    *,
    repo_root: Path,
    source_commit: str,
    expected_vendor_task_sha256: str,
    expected_action_ball_task_sha256: str,
    expected_hitter_task_sha256: str,
    expected_env_base_sha256: str,
    expected_sim_base_sha256: str,
    expected_randomization_base_sha256: str,
    expected_robot_source_sha256: str,
    expected_env_cfg_source_sha256: str,
    expected_train_source_sha256: str,
    expected_training_contract_source_sha256: str,
    expected_hope_actions_source_sha256: str,
    expected_runner_source_sha256: str,
    expected_stable_motion_sha256: str,
    runtime_training_contract: Path,
    expected_runtime_training_contract_sha256: str,
    output: Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    commit = _resolve_commit(root, source_commit)
    head = _resolve_commit(root, "HEAD")
    if head != commit:
        raise VendorRuntimeAuthorityError(
            f"producer requires HEAD={commit}, got {head}"
        )
    output_expected = root.joinpath(*PurePosixPath(RECEIPT_REPO_PATH).parts)
    output_absolute = Path(output).expanduser().absolute()
    if output_absolute != output_expected:
        raise VendorRuntimeAuthorityError(
            f"authority output must use fixed path {RECEIPT_REPO_PATH}"
        )

    expected_by_name = {
        "vendor_task_profile": expected_vendor_task_sha256,
        "action_ball_task_profile": expected_action_ball_task_sha256,
        "hitter_task_profile": expected_hitter_task_sha256,
        "environment_base_profile": expected_env_base_sha256,
        "simulation_base_profile": expected_sim_base_sha256,
        "randomization_base_profile": expected_randomization_base_sha256,
        "robot_actuator_source": expected_robot_source_sha256,
        "environment_config_source": expected_env_cfg_source_sha256,
        "training_entrypoint": expected_train_source_sha256,
        "training_contract_source": expected_training_contract_source_sha256,
        "action_source": expected_hope_actions_source_sha256,
        "runner_source": expected_runner_source_sha256,
        "stable_motion": expected_stable_motion_sha256,
    }
    sources = {
        name: _expected_pin(
            root,
            commit,
            relative,
            expected_by_name[name],
            name=name.replace("_", " "),
        )
        for name, relative in SOURCE_PATHS.items()
    }
    _contract_path, contract_sha, contract = _stable_contract_file(
        root,
        runtime_training_contract,
        expected_runtime_training_contract_sha256,
    )
    verified = _verified_vendor_runtime(
        contract,
        stable_motion_sha256=sources["stable_motion"]["sha256"],
    )
    runtime_plant_identity = _canonical_runtime_plant_identity(contract)
    producer_path = Path(__file__).resolve(strict=True)
    try:
        producer_relative = producer_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise VendorRuntimeAuthorityError(
            "producer must run from the selected repository root"
        ) from exc
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source_commit": commit,
        "sources": sources,
        "runtime_training_contract": {
            "path": RUNTIME_CONTRACT_REPO_PATH,
            "sha256": contract_sha,
            "schema_version": 3,
        },
        "runtime_plant_identity": runtime_plant_identity,
        "verified_vendor_runtime": verified,
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "producer": {
            "path": producer_relative,
            "sha256": _sha256_file(producer_path),
        },
    }
    result["content_sha256"] = _sha256_bytes(_canonical_bytes(result))
    payload = _canonical_bytes(result) + b"\n"
    published = _write_exclusive(output_absolute, payload)
    reparsed = _strict_json_bytes(
        published.read_bytes(), name="published vendor runtime authority"
    )
    if reparsed != result:
        raise VendorRuntimeAuthorityError(
            "published authority receipt failed exact readback"
        )
    return result


def _validate_pin(
    pin_value: object,
    *,
    name: str,
    expected_path: str,
    repo_root: Path,
    source_commit: str,
    launch_commit: str,
) -> dict[str, str]:
    pin = _exact_keys(pin_value, _SOURCE_KEYS, name=name)
    if pin["path"] != expected_path:
        raise VendorRuntimeAuthorityError(
            f"{name}.path differs: {pin['path']!r} != {expected_path!r}"
        )
    digest = _require_sha256(pin["sha256"], name=f"{name}.sha256")
    source_sha = _sha256_bytes(_git_blob(repo_root, source_commit, expected_path))
    launch_sha = _sha256_bytes(_git_blob(repo_root, launch_commit, expected_path))
    worktree_sha = _sha256_file(_repo_file(repo_root, expected_path, name=name))
    if len({digest, source_sha, launch_sha, worktree_sha}) != 1:
        raise VendorRuntimeAuthorityError(
            f"{name} drifted across authority commit, launch commit, or worktree"
        )
    return {"path": expected_path, "sha256": digest}


def load_and_validate_vendor_runtime_authority(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_receipt_sha256: str | None = None,
    expected_runtime_training_contract_sha256: str | None = None,
    launch_commit: str = "HEAD",
    require_fixed_path: bool = True,
) -> dict[str, Any]:
    """Load the fixed receipt and close it over current launch-commit blobs.

    A vendor launcher should call this during ``plan`` and again immediately
    before delegating to Kit.  It should pass the runtime-contract SHA read
    from the candidate as ``expected_runtime_training_contract_sha256``.  An
    old candidate then fails even if the operator repeats its old SHA in the
    launch spec.
    """

    root = Path(repo_root).resolve(strict=True)
    requested = Path(receipt_path).expanduser().absolute()
    expected_path = root.joinpath(*PurePosixPath(RECEIPT_REPO_PATH).parts)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise VendorRuntimeAuthorityError(
            f"cannot resolve vendor runtime authority: {exc}"
        ) from exc
    if (
        requested != resolved
        or resolved.is_symlink()
        or not resolved.is_file()
        or (require_fixed_path and resolved != expected_path)
    ):
        raise VendorRuntimeAuthorityError(
            "vendor runtime authority must be the fixed regular receipt path"
        )
    payload = resolved.read_bytes()
    receipt_sha = _sha256_bytes(payload)
    if expected_receipt_sha256 is not None:
        expected_receipt = _require_sha256(
            expected_receipt_sha256, name="expected authority receipt SHA-256"
        )
        if receipt_sha != expected_receipt:
            raise VendorRuntimeAuthorityError(
                "vendor runtime authority receipt SHA differs from its code-owned pin"
            )
    receipt = _strict_json_bytes(payload, name="vendor runtime authority")
    if payload != _canonical_bytes(receipt) + b"\n":
        raise VendorRuntimeAuthorityError(
            "vendor runtime authority bytes are not canonical JSON plus newline"
        )
    _exact_keys(receipt, _TOP_LEVEL_KEYS, name="vendor runtime authority")
    if receipt["schema_version"] != SCHEMA_VERSION or receipt["kind"] != KIND:
        raise VendorRuntimeAuthorityError(
            "vendor runtime authority schema/kind is unsupported"
        )
    seal = _require_sha256(
        receipt["content_sha256"], name="authority content_sha256"
    )
    unsigned = dict(receipt)
    unsigned.pop("content_sha256")
    if _sha256_bytes(_canonical_bytes(unsigned)) != seal:
        raise VendorRuntimeAuthorityError(
            "vendor runtime authority content seal is false"
        )
    source_commit = receipt["source_commit"]
    if type(source_commit) is not str or _COMMIT_RE.fullmatch(source_commit) is None:
        raise VendorRuntimeAuthorityError("authority source_commit is malformed")
    source_commit = _resolve_commit(root, source_commit)
    current_commit = _resolve_commit(root, launch_commit)
    if require_fixed_path:
        try:
            committed_receipt = _git_blob(
                root, current_commit, RECEIPT_REPO_PATH
            )
        except VendorRuntimeAuthorityError as exc:
            raise VendorRuntimeAuthorityError(
                "vendor runtime authority is not an exact launch-commit blob"
            ) from exc
        if committed_receipt != payload:
            raise VendorRuntimeAuthorityError(
                "vendor runtime authority is not the exact launch-commit blob"
            )

    sources = receipt["sources"]
    if type(sources) is not dict or set(sources) != set(SOURCE_PATHS):
        raise VendorRuntimeAuthorityError(
            "authority source map is incomplete or has unknown roles"
        )
    validated_sources = {
        name: _validate_pin(
            sources[name],
            name=f"authority sources.{name}",
            expected_path=relative,
            repo_root=root,
            source_commit=source_commit,
            launch_commit=current_commit,
        )
        for name, relative in SOURCE_PATHS.items()
    }

    contract_pin = _exact_keys(
        receipt["runtime_training_contract"],
        _CONTRACT_KEYS,
        name="authority runtime_training_contract",
    )
    if (
        contract_pin["path"] != RUNTIME_CONTRACT_REPO_PATH
        or contract_pin["schema_version"] != 3
    ):
        raise VendorRuntimeAuthorityError(
            "authority runtime training contract path/schema differs"
        )
    contract_sha = _require_sha256(
        contract_pin["sha256"], name="authority runtime contract SHA-256"
    )
    if expected_runtime_training_contract_sha256 is not None:
        expected_contract = _require_sha256(
            expected_runtime_training_contract_sha256,
            name="candidate runtime training contract SHA-256",
        )
        if contract_sha != expected_contract:
            raise VendorRuntimeAuthorityError(
                "candidate runtime contract is not the code-owned vendor authority"
            )
    contract_path = _repo_file(
        root, RUNTIME_CONTRACT_REPO_PATH, name="authority runtime training contract"
    )
    launch_contract_sha = _sha256_bytes(
        _git_blob(root, current_commit, RUNTIME_CONTRACT_REPO_PATH)
    )
    if _sha256_file(contract_path) != contract_sha or launch_contract_sha != contract_sha:
        raise VendorRuntimeAuthorityError(
            "authority runtime contract differs from launch commit/worktree bytes"
        )
    contract = _strict_json_bytes(
        contract_path.read_bytes(), name="authority runtime training contract"
    )
    plant_identity = _canonical_runtime_plant_identity(contract)
    receipt_plant = _exact_keys(
        receipt["runtime_plant_identity"],
        _RUNTIME_PLANT_IDENTITY_KEYS,
        name="authority runtime_plant_identity",
    )
    if receipt_plant != plant_identity:
        raise VendorRuntimeAuthorityError(
            "authority runtime_plant_identity is not reproducible"
        )
    verified = _verified_vendor_runtime(
        contract,
        stable_motion_sha256=validated_sources["stable_motion"]["sha256"],
    )
    if receipt["verified_vendor_runtime"] != verified:
        raise VendorRuntimeAuthorityError(
            "authority verified_vendor_runtime is not reproducible"
        )
    authorization = _exact_keys(
        receipt["authorization"],
        _AUTHORIZATION_KEYS,
        name="authority authorization",
    )
    if any(value is not False for value in authorization.values()):
        raise VendorRuntimeAuthorityError(
            "vendor runtime authority may not self-authorize training/deployment/hardware"
        )
    producer = _exact_keys(
        receipt["producer"], _PRODUCER_KEYS, name="authority producer"
    )
    producer_path = producer["path"]
    producer_sha = _require_sha256(
        producer["sha256"], name="authority producer.sha256"
    )
    if (
        type(producer_path) is not str
        or _sha256_bytes(_git_blob(root, source_commit, producer_path))
        != producer_sha
        or _sha256_bytes(_git_blob(root, current_commit, producer_path))
        != producer_sha
        or _sha256_file(_repo_file(root, producer_path, name="authority producer"))
        != producer_sha
    ):
        raise VendorRuntimeAuthorityError(
            "authority producer drifted across source/launch/worktree bytes"
        )
    return {
        "receipt_path": RECEIPT_REPO_PATH,
        "receipt_sha256": receipt_sha,
        "source_commit": source_commit,
        "launch_commit": current_commit,
        "sources": validated_sources,
        "runtime_training_contract": {
            "path": RUNTIME_CONTRACT_REPO_PATH,
            "sha256": contract_sha,
            "schema_version": 3,
        },
        "runtime_plant_identity": plant_identity,
        "verified_vendor_runtime": verified,
        "authorization": dict(authorization),
    }


def _canonical_candidate_runtime_plant(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a schema-v2 dynamic candidate to the authority plant shape."""

    if (
        candidate.get("schema_version") != 2
        or candidate.get("kind")
        != "agibot_a3_action_dynamic_ready_candidate_v2"
    ):
        raise VendorRuntimeAuthorityError(
            "vendor launch requires schema-v2 dynamic-ready candidate"
        )
    robot = candidate.get("robot")
    runtime = candidate.get("runtime_plant")
    if type(robot) is not dict or type(runtime) is not dict:
        raise VendorRuntimeAuthorityError(
            "dynamic-ready candidate lacks robot/runtime_plant"
        )
    names = runtime.get("joint_names")
    if robot.get("family") != "AgiBot A3" or robot.get("joint_names") != names:
        raise VendorRuntimeAuthorityError(
            "dynamic-ready robot joint order differs from runtime_plant"
        )
    normalized_contract = {
        "schema_version": 3,
        "joint_names": names,
        "articulation_joint_names": runtime.get("articulation_joint_names"),
        "action_joint_ids": runtime.get("action_joint_ids"),
        "joint_stiffness": runtime.get("joint_stiffness"),
        "joint_damping": runtime.get("joint_damping"),
        "joint_effort_limits": runtime.get("joint_effort_limits"),
        "joint_velocity_limits": runtime.get("joint_velocity_limits"),
        "joint_armature": runtime.get("joint_armature"),
        "default_joint_pos": runtime.get("default_joint_pos_rad"),
        "action_scale": runtime.get("action_scale_rad"),
        "qdes_joint_pos_limits": runtime.get("qdes_joint_pos_limits"),
        "physics_step_dt_s": runtime.get("physics_step_dt_s"),
        "policy_step_dt_s": runtime.get("policy_step_dt_s"),
        "control_decimation": runtime.get("control_decimation"),
        "control_step_action_delay": runtime.get("control_step_action_delay"),
    }
    return _canonical_runtime_plant_identity(normalized_contract)


def validate_candidate_runtime_plant_against_vendor_authority(
    candidate: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a dynamic-ready candidate whose host-readable plant is stale."""

    expected = authority.get("runtime_plant_identity")
    if type(expected) is not dict:
        raise VendorRuntimeAuthorityError(
            "validated authority summary lacks runtime_plant_identity"
        )
    candidate_plant = _canonical_candidate_runtime_plant(candidate)
    sources = candidate.get("sources")
    stable_motion_pin = (
        sources.get("stable_motion") if type(sources) is dict else None
    )
    authoritative_sources = authority.get("sources")
    authoritative_motion = (
        authoritative_sources.get("stable_motion")
        if type(authoritative_sources) is dict
        else None
    )
    if (
        candidate.get("action_id") != "bh_loop_c"
        or type(stable_motion_pin) is not dict
        or type(authoritative_motion) is not dict
        or stable_motion_pin.get("sha256")
        != authoritative_motion.get("sha256")
    ):
        raise VendorRuntimeAuthorityError(
            "vendor authority currently admits only exact bh_loop_c stable motion"
        )
    if candidate_plant != expected:
        raise VendorRuntimeAuthorityError(
            "dynamic-ready runtime_plant differs from vendor authority"
        )
    runtime_pin = (
        sources.get("runtime_training_contract")
        if type(sources) is dict
        else None
    )
    authoritative_contract = authority.get("runtime_training_contract")
    if (
        type(runtime_pin) is not dict
        or type(authoritative_contract) is not dict
        or runtime_pin.get("sha256") != authoritative_contract.get("sha256")
    ):
        raise VendorRuntimeAuthorityError(
            "dynamic-ready runtime contract pin differs from vendor authority"
        )
    return candidate_plant


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-vendor-task-sha256", required=True)
    parser.add_argument("--expected-action-ball-task-sha256", required=True)
    parser.add_argument("--expected-hitter-task-sha256", required=True)
    parser.add_argument("--expected-env-base-sha256", required=True)
    parser.add_argument("--expected-sim-base-sha256", required=True)
    parser.add_argument("--expected-randomization-base-sha256", required=True)
    parser.add_argument("--expected-robot-source-sha256", required=True)
    parser.add_argument("--expected-env-cfg-source-sha256", required=True)
    parser.add_argument("--expected-train-source-sha256", required=True)
    parser.add_argument(
        "--expected-training-contract-source-sha256", required=True
    )
    parser.add_argument("--expected-hope-actions-source-sha256", required=True)
    parser.add_argument("--expected-runner-source-sha256", required=True)
    parser.add_argument("--expected-stable-motion-sha256", required=True)
    parser.add_argument("--runtime-training-contract", required=True)
    parser.add_argument(
        "--expected-runtime-training-contract-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_vendor_runtime_authority(
        repo_root=Path(args.repo_root),
        source_commit=args.source_commit,
        expected_vendor_task_sha256=args.expected_vendor_task_sha256,
        expected_action_ball_task_sha256=args.expected_action_ball_task_sha256,
        expected_hitter_task_sha256=args.expected_hitter_task_sha256,
        expected_env_base_sha256=args.expected_env_base_sha256,
        expected_sim_base_sha256=args.expected_sim_base_sha256,
        expected_randomization_base_sha256=(
            args.expected_randomization_base_sha256
        ),
        expected_robot_source_sha256=args.expected_robot_source_sha256,
        expected_env_cfg_source_sha256=args.expected_env_cfg_source_sha256,
        expected_train_source_sha256=args.expected_train_source_sha256,
        expected_training_contract_source_sha256=(
            args.expected_training_contract_source_sha256
        ),
        expected_hope_actions_source_sha256=args.expected_hope_actions_source_sha256,
        expected_runner_source_sha256=args.expected_runner_source_sha256,
        expected_stable_motion_sha256=args.expected_stable_motion_sha256,
        runtime_training_contract=Path(args.runtime_training_contract),
        expected_runtime_training_contract_sha256=(
            args.expected_runtime_training_contract_sha256
        ),
        output=Path(args.output),
    )
    print(
        json.dumps(
            {
                "output": RECEIPT_REPO_PATH,
                "content_sha256": result["content_sha256"],
                "runtime_training_contract_sha256": result[
                    "runtime_training_contract"
                ]["sha256"],
                "authorization": result["authorization"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
