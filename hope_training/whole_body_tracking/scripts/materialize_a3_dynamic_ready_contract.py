#!/usr/bin/env python3
"""Materialize one action-specific AgiBot A3 dynamic-ready hold candidate.

The artifact separates the exact motion frame-0 teacher reference, the
simulator physical birth, and the implicit-PD joint target used to hold that
birth.  It is a deterministic candidate, not an Isaac hold certificate and not
training authorization.  A downstream nominal-hold probe must validate it on
the exact Isaac/PhysX plant.

Two fail-closed ready-source branches are supported:

* ``stable_upper_v2`` preserves the historical stable-upper receipt and exact
  action-runtime binding path.
* ``measured_retarget_l0_diagnostic`` keeps the measured-retarget motion bytes
  and complete frame-0 teacher reference unchanged, but deliberately separates
  that reference from physical birth.  Seed-backed modes consume a
  content-pinned numerical ready without inheriting its historical model/hold
  claims.  The measured-frame0 projection mode instead preserves the exact
  root, every non-leg joint, and racket-site FK while solving only leg12 and an
  algorithmic common support-edge shift.  That leg-only path is still a failure
  baseline, but 2026-08-07 changed WHY (doc 5.6.7 sections eleven and twelve):
  its static geometry, ground LP and support margin now all pass -- what refuses
  it is the hold LP, because holding measured frame 0 needs -49.155 N*m at
  ``waist_pitch`` and this plant's position command tops out near -26 N*m even
  at the mechanical stop.  The refusal below names that.
  The whole-body mode releases root z/roll/pitch plus all 31 joints,
  first accepts exact measured frame 0 unchanged if all physical gates pass;
  otherwise it searches the fixed named robust-feasible set, then minimizes
  measured-frame0 root/joint/racket error inside that set.
  Every composition must pass current-
  MJCF static gates, a fresh ground LP, and the downstream exact Isaac
  nominal-hold gate.

Both branches only produce unauthorized candidates.  Neither branch becomes a
policy bootstrap until the exact downstream Isaac nominal-hold receipt passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
import tempfile
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import canonical_grounded_ready as grounded
import canonical_mujoco_path_adapter as path_adapter
import canonical_torque_path_topp as torque_topp
import audit_self_collision as self_collision_audit
import whole_body_safe_ready as whole_body_ready


SCHEMA_VERSION = 2
KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
LP_OBJECTIVE = torque_topp.GROUND_LP_OBJECTIVE_HOLD_MINIMAX
STABLE_UPPER_SOURCE_KIND = "stable_upper_v2"
MEASURED_RETARGET_SOURCE_KIND = "measured_retarget_l0_diagnostic"
MEASURED_BANK_RECEIPT_KIND = "chingmu73_measured_racket_schema_v4_repo_import"
MEASURED_MECHANICAL_AUDIT_KIND = "measured_racket_mechanical_admission_audit_v1"
_MJCF_XML_MODEL_NAME_BY_SHA256 = {
    "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a": (
        "A3T2.5_pingpong_0519"
    ),
    "7bbda723f339bdf252a20622afa7a7d53a6fca97464252c66c6e1a45199bcae1": (
        "A3P_P1_0807_OP3_pingpang"
    ),
}
PHYSICAL_BIRTH_SEED_KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS = (
    "support_centroid_anchored_world_z_rotation_to_teacher_root_yaw"
)
MEASURED_BIRTH_SHARED_LOWER_MODE = "shared_lower_teacher_nonleg"
MEASURED_BIRTH_FULL_SEED_MODE = "full_seed"
MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE = "full_seed_contact_free_hold_projection"
MEASURED_BIRTH_DIRECT_FRAME0_MODE = "direct_teacher_frame0"
MEASURED_BIRTH_PROJECTED_FRAME0_MODE = "projected_teacher_frame0_grounded"
MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE = (
    "whole_body_safe_teacher_frame0_grounded"
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
MEASURED_BIRTH_PROJECTED_FRAME0_SEMANTICS = (
    "exact_measured_teacher_frame0_root_nonleg_plus_grounded_leg12_projection"
)
MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS = (
    "measured_frame0_direct_if_safe_else_lexicographic_whole_body_safe_ready"
)
MEASURED_PROJECTED_FRAME0_RACKET_SITE = "right_racket"
FULL_SEED_QDES_FRESH_STATIC_LP = "fresh_static_lp"
FULL_SEED_QDES_SEED_TRANSPORT = "seed_transport"
CONTACT_FREE_PROJECTION_MINIMUM_TORQUE_SLACK_NM = 1.0e-2
WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N = 0.1
WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_FOOT_N = 1.0
WHOLE_BODY_REQUIRED_COLLISION_CLEARANCE_M = 2.0e-3
WHOLE_BODY_COLLISION_CLEARANCE_CAP_M = 2.0e-2
WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M = 1.0e-4
EXPECTED_MEASURED_RACKET_SCHEMA = 4
EXPECTED_MEASURED_RACKET_POSITION_SEMANTICS = "physical_blade_center"
EXPECTED_MEASURED_RACKET_NORMAL_SEMANTICS = "signed_physical_hitting_face"
EXPECTED_MEASURED_RACKET_LONG_AXIS_SEMANTICS = (
    "measured_paddle_butt_to_blade"
)
EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL = (
    1.0 / math.sqrt(2.0),
    0.0,
    1.0 / math.sqrt(2.0),
)
EXPECTED_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256 = (
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
)
EXPECTED_MEASURED_JOINT_ORDER_CONTRACT_ID = (
    "a3-gmr-dof-pos-to-runtime-articulation-v1"
)
EXPECTED_MEASURED_JOINT_ORDER_CONTRACT_SHA256 = (
    "b09987ff7a1bfa624b566cc8884d16672ba73c1acc3f92efb8a4faa99d314815"
)
MEASURED_DIAGNOSTIC_DELAY_CONTRACT = {
    "schema_version": 1,
    "enabled": False,
    "semantic_unit": "policy_control_step",
    "sample_timing": "once_per_episode_reset",
    "distribution": "discrete_uniform_inclusive",
    "min_steps": 0,
    "max_steps": 0,
    "shared_across_all_31_joints": True,
    "history_fill": "safe_default_or_action_specific_hold",
}
_PHYSX_CONTROL_POSITION_LIMIT_KEYS = frozenset(
    {
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
)
_PHYSX_CONTROL_POSITION_LIMIT_SELECTED_JOINT_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


class DynamicReadyMaterializationError(RuntimeError):
    """The requested dynamic-ready artifact cannot be produced exactly."""


# 人话:有一批关节,地面永远帮不上忙 —— 脚不在它们的子树里,所以脚底的支撑力
# 在这些关节上产生的力矩恒为零。腰、双臂、脖子都属于这一类。对这些关节,
# "撑住这个姿态需要多大力矩"根本没有解算自由度:它**就等于** `qfrc_bias`。
# 于是 hold LP 说"无解"时,可以不猜:先看这些关节的 `qfrc_bias` 落没落在
# 位置指令能产生的力矩区间里,落在外面就是**唯一且充分**的原因,并且能直接
# 报出差多少 N·m、需要多大的 q_des、以及边界是电机限幅还是 `kp × 行程`。
CONTACT_FREE_HOLD_TORQUE_SEMANTICS = (
    "floor_contacts_cannot_load_these_rows_so_tau_equals_qfrc_bias_exactly"
)


def contact_free_actuated_rows(model, actuated_dof_indices) -> np.ndarray:
    """Which actuated rows no floor contact can ever load, on this exact model.

    A DoF is contact-free when neither foot body sits in the sub-tree the DoF
    moves.  MuJoCo answers that directly: build the translational Jacobian of
    each foot body and keep the rows whose column is exactly zero.  Returned as
    a boolean mask aligned to ``actuated_dof_indices``.
    """

    import mujoco  # noqa: PLC0415

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rows = np.asarray(actuated_dof_indices, np.int64)
    loaded = np.zeros(int(model.nv), bool)
    for body_name in grounded.FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise DynamicReadyMaterializationError(
                f"exact MJCF has no foot body {body_name!r}"
            )
        jacp = np.zeros((3, int(model.nv)), np.float64)
        mujoco.mj_jacBodyCom(model, data, jacp, None, body_id)
        loaded |= np.any(jacp != 0.0, axis=0)
    return ~loaded[rows]


def contact_free_hold_torque_shortfall(
    *,
    joint_names: Sequence[str],
    contact_free: np.ndarray,
    required_nm: np.ndarray,
    tau_lower_nm: np.ndarray,
    tau_upper_nm: np.ndarray,
    kp: np.ndarray,
    ready_q_rad: np.ndarray,
    executed_qdes_lower_rad: np.ndarray,
    executed_qdes_upper_rad: np.ndarray,
    motor_effort_nm: np.ndarray,
) -> list[dict[str, Any]]:
    """Name every contact-free row whose required hold torque is unreachable.

    All arrays are in runtime joint order.  ``required_nm`` is ``qfrc_bias`` on
    the actuated rows at the candidate pose with zero velocity and zero
    acceleration; for a contact-free row that value IS the holding torque, so a
    row listed here is a sufficient, non-negotiable reason the static hold LP
    has no solution -- no other row can trade against it.
    """

    names = [str(name) for name in joint_names]
    size = len(names)
    arrays = {
        "contact_free": np.asarray(contact_free, bool),
        "required_nm": np.asarray(required_nm, np.float64),
        "tau_lower_nm": np.asarray(tau_lower_nm, np.float64),
        "tau_upper_nm": np.asarray(tau_upper_nm, np.float64),
        "kp": np.asarray(kp, np.float64),
        "ready_q_rad": np.asarray(ready_q_rad, np.float64),
        "executed_qdes_lower_rad": np.asarray(executed_qdes_lower_rad, np.float64),
        "executed_qdes_upper_rad": np.asarray(executed_qdes_upper_rad, np.float64),
        "motor_effort_nm": np.asarray(motor_effort_nm, np.float64),
    }
    for label, value in arrays.items():
        if value.shape != (size,):
            raise DynamicReadyMaterializationError(
                f"contact-free hold attribution {label} must have one entry per joint"
            )
    out: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        if not bool(arrays["contact_free"][index]):
            continue
        need = float(arrays["required_nm"][index])
        low = float(arrays["tau_lower_nm"][index])
        high = float(arrays["tau_upper_nm"][index])
        gain = float(arrays["kp"][index])
        q_now = float(arrays["ready_q_rad"][index])
        qdes_low = float(arrays["executed_qdes_lower_rad"][index])
        qdes_high = float(arrays["executed_qdes_upper_rad"][index])
        # 人话:先分清两件完全不同的事。
        #  (甲) 关节本身已经站在可发指令的 q_des 包络**之外** —— 这时连"零力矩"
        #       都发不出来,区间是空的,毛病在姿态/限位,不在增益。
        #  (乙) 关节在包络里,但 `kp × 还能走多远` 撑不出需要的力矩 —— 毛病在增益。
        # 混在一起报会把人引到错误的修法上,所以这里各有各的名字。
        outside = not (qdes_low <= q_now <= qdes_high)
        if not outside and low <= need <= high:
            continue
        record = {
            "joint": name,
            "required_hold_torque_nm": need,
            "reachable_torque_interval_nm": [low, high],
            "kp": gain,
            "motor_effort_limit_nm": float(arrays["motor_effort_nm"][index]),
            "pose_q_rad": q_now,
            "executed_qdes_interval_rad": [qdes_low, qdes_high],
            "semantics": CONTACT_FREE_HOLD_TORQUE_SEMANTICS,
        }
        if gain > 0.0:
            record["qdes_that_would_be_needed_rad"] = float(need / gain + q_now)
        if outside:
            record["binding_side"] = "pose_outside_executed_qdes_envelope"
            record["binding_authority"] = "pose_outside_executed_qdes_envelope"
            # 人话:这一档没有"差多少 N·m"这个数 —— 可达区间本身是空的。
            # 写 `None` 而不是 `NaN`,收据才能是合法 JSON(报告用 allow_nan=False)。
            record["shortfall_nm"] = None
            record["pose_outside_envelope_by_rad"] = (
                q_now - qdes_low if q_now < qdes_low else q_now - qdes_high
            )
        else:
            record["binding_side"] = "lower" if need < low else "upper"
            record["shortfall_nm"] = need - (low if need < low else high)
            edge = low if need < low else high
            record["binding_authority"] = (
                "motor_effort_limit"
                if abs(abs(edge) - float(arrays["motor_effort_nm"][index])) <= 1.0e-9
                else "kp_times_available_qdes_travel"
            )
        out.append(record)
    return out


def _contact_free_hold_refusal_text(records: Sequence[Mapping[str, Any]]) -> str:
    """One human-readable line per unreachable contact-free row."""

    parts = []
    for record in records:
        low, high = record["reachable_torque_interval_nm"]
        needed = record.get("qdes_that_would_be_needed_rad")
        qlo, qhi = record["executed_qdes_interval_rad"]
        if record["binding_side"] == "pose_outside_executed_qdes_envelope":
            parts.append(
                f"{record['joint']} sits at q={record['pose_q_rad']:+.4f} rad, which is "
                f"{abs(record['pose_outside_envelope_by_rad']):.4f} rad outside the "
                f"executed q_des envelope [{qlo:+.4f}, {qhi:+.4f}] rad, so no command "
                "reaches even zero torque there"
            )
            continue
        text = (
            f"{record['joint']} needs {record['required_hold_torque_nm']:+.3f} N*m "
            f"but a position command can only reach [{low:+.3f}, {high:+.3f}] N*m "
            f"(short {abs(record['shortfall_nm']):.3f} N*m, limited by "
            f"{record['binding_authority']}, kp={record['kp']:g}, "
            f"motor limit {record['motor_effort_limit_nm']:g} N*m"
        )
        if needed is not None:
            text += (
                f"; it would need q_des={needed:+.4f} rad and the executed "
                f"envelope is [{qlo:+.4f}, {qhi:+.4f}] rad"
            )
        parts.append(text + ")")
    return "; ".join(parts)


STATIC_HOLD_REFUSAL_PREFIX = (
    "no static double-support hold exists inside the executed qdes envelope"
)


def static_hold_required_generalized_force(model, qpos) -> np.ndarray:
    """``qfrc_bias`` at ``qpos`` with zero velocity and zero acceleration."""

    import mujoco  # noqa: PLC0415

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, np.float64)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return np.array(data.qfrc_bias, np.float64)


def _static_hold_refusal_message(
    *,
    backend,
    qpos,
    actuated: np.ndarray,
    model_row_for_runtime: np.ndarray,
    plant: Mapping[str, Any],
    ready_q: np.ndarray,
    executed_qdes_lower: np.ndarray,
    executed_qdes_upper: np.ndarray,
    hold_tau_lower_model: np.ndarray,
    hold_tau_upper_model: np.ndarray,
) -> str:
    """Turn an opaque infeasible hold LP into a named, numeric refusal."""

    try:
        contact_free_rows = contact_free_actuated_rows(backend.model, actuated)
        bias = static_hold_required_generalized_force(backend.model, qpos)
    except Exception as exc:  # noqa: BLE001
        return (
            f"{STATIC_HOLD_REFUSAL_PREFIX}; the contact-free attribution itself "
            f"could not be computed ({type(exc).__name__}: {exc})"
        )
    rows = np.asarray(model_row_for_runtime, np.int64)
    records = contact_free_hold_torque_shortfall(
        joint_names=list(plant["joint_names"]),
        contact_free=contact_free_rows[rows],
        required_nm=bias[np.asarray(actuated, np.int64)][rows],
        tau_lower_nm=np.asarray(hold_tau_lower_model, np.float64)[rows],
        tau_upper_nm=np.asarray(hold_tau_upper_model, np.float64)[rows],
        kp=np.asarray(plant["kp"], np.float64),
        ready_q_rad=np.asarray(ready_q, np.float64),
        executed_qdes_lower_rad=np.asarray(executed_qdes_lower, np.float64),
        executed_qdes_upper_rad=np.asarray(executed_qdes_upper, np.float64),
        motor_effort_nm=np.asarray(plant["effort"], np.float64),
    )
    if not records:
        return (
            f"{STATIC_HOLD_REFUSAL_PREFIX}; every contact-free row is inside its "
            "reachable torque interval, so the binding constraint is on the "
            "ground-loaded rows or the friction cone"
        )
    return (
        f"{STATIC_HOLD_REFUSAL_PREFIX}: "
        + _contact_free_hold_refusal_text(records)
    )


def _remove_pinned_snapshots(paths: list[Path]) -> None:
    for path in reversed(paths):
        try:
            path.unlink()
        except OSError:
            pass


class _PinnedFile:
    """Original display path plus a private snapshot of its hashed bytes."""

    __slots__ = (
        "source_path",
        "snapshot_path",
        "mjcf_snapshot_path",
        "_cleanup_paths",
        "_finalizer",
        "__weakref__",
    )

    def __init__(self, source_path: Path, snapshot_path: Path) -> None:
        self.source_path = source_path
        self.snapshot_path = snapshot_path
        self.mjcf_snapshot_path: Path | None = None
        self._cleanup_paths = [snapshot_path]
        self._finalizer = weakref.finalize(
            self, _remove_pinned_snapshots, self._cleanup_paths
        )

    def __fspath__(self) -> str:
        return str(self.source_path)

    def __str__(self) -> str:
        return str(self.source_path)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("write made no progress")
        written += count


def _pinned_snapshot_path(path: Path) -> Path:
    if isinstance(path, _PinnedFile):
        return path.snapshot_path
    return Path(path)


def _pinned_mjcf_path(path: Path) -> Path:
    """Return a same-directory MJCF snapshot so relative assets still resolve."""

    if not isinstance(path, _PinnedFile):
        return Path(path)
    if path.mjcf_snapshot_path is not None:
        return path.mjcf_snapshot_path
    descriptor = -1
    snapshot_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.source_path.name}.pinned-",
            suffix=path.source_path.suffix,
            dir=str(path.source_path.parent),
        )
        snapshot_path = Path(raw_path)
        with path.snapshot_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                _write_all(descriptor, block)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.close(descriptor)
        descriptor = -1
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except OSError:
                pass
        raise DynamicReadyMaterializationError(
            f"cannot create same-directory pinned MJCF snapshot: {exc}"
        ) from exc
    path.mjcf_snapshot_path = snapshot_path
    path._cleanup_paths.append(snapshot_path)
    return snapshot_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise DynamicReadyMaterializationError(
            f"{name} must be 64 lowercase SHA-256 digits"
        )
    return digest


def _pinned_file(
    path_value: str | Path, expected_sha256: object, *, name: str
) -> tuple[Path, str]:
    path_input = Path(path_value).expanduser().absolute()
    try:
        path = path_input.resolve(strict=True)
    except OSError as exc:
        raise DynamicReadyMaterializationError(
            f"cannot resolve {name}: {exc}"
        ) from exc
    if path_input != path or path_input.is_symlink() or not path.is_file():
        raise DynamicReadyMaterializationError(
            f"{name} must be one regular file without symlink components"
        )
    expected = _require_sha256(expected_sha256, name=f"expected {name}")
    source_descriptor = -1
    snapshot_descriptor = -1
    snapshot_path: Path | None = None
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        source_flags |= getattr(os, "O_CLOEXEC", 0)
        source_descriptor = os.open(path, source_flags)
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode):
            raise DynamicReadyMaterializationError(
                f"{name} must remain one regular file while it is pinned"
            )
        snapshot_descriptor, raw_snapshot_path = tempfile.mkstemp(
            prefix="a3-dynamic-ready-input-",
            suffix=path.suffix,
        )
        snapshot_path = Path(raw_snapshot_path)
        digest = hashlib.sha256()
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            _write_all(snapshot_descriptor, block)
        source_after = os.fstat(source_descriptor)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_size")
        if any(
            getattr(source_before, field) != getattr(source_after, field)
            for field in stable_fields
        ) or source_before.st_mtime_ns != source_after.st_mtime_ns:
            raise DynamicReadyMaterializationError(
                f"{name} changed while its bytes were being pinned"
            )
        actual = digest.hexdigest()
        if actual != expected:
            raise DynamicReadyMaterializationError(
                f"{name} SHA-256 mismatch: {actual} != {expected}"
            )
        os.fsync(snapshot_descriptor)
        os.fchmod(snapshot_descriptor, 0o400)
        os.close(snapshot_descriptor)
        snapshot_descriptor = -1
        os.close(source_descriptor)
        source_descriptor = -1
    except DynamicReadyMaterializationError:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except OSError:
                pass
        raise
    except OSError as exc:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except OSError:
                pass
        raise DynamicReadyMaterializationError(
            f"cannot pin {name}: {exc}"
        ) from exc
    return _PinnedFile(path, snapshot_path), actual


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            _pinned_snapshot_path(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicReadyMaterializationError(
            f"cannot read {name} JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DynamicReadyMaterializationError(f"{name} must be one JSON object")
    return payload


def _validate_stable_receipt(
    receipt: Mapping[str, Any],
    *,
    motion_sha256: str,
) -> None:
    if (
        receipt.get("schema_version") != 2
        or receipt.get("artifact_class")
        != "diagnostic_a3_stable_upper_motion_v2"
        or receipt.get("verdict")
        != "PASS_DIAGNOSTIC_A3_STABLE_UPPER_WAIST_REBASED_REBUILD"
    ):
        raise DynamicReadyMaterializationError(
            "stable receipt is not the exact A3 stable-upper-v2 artifact class"
        )
    robot = receipt.get("robot")
    authorization = receipt.get("authorization")
    outputs = receipt.get("outputs")
    if (
        not isinstance(robot, Mapping)
        or robot.get("family") != "AgiBot A3"
        or not isinstance(authorization, Mapping)
        or any(
            authorization.get(name) is not False
            for name in (
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            )
        )
        or not isinstance(outputs, Mapping)
        or outputs.get("motion_sha256") != motion_sha256
    ):
        raise DynamicReadyMaterializationError(
            "stable receipt robot, authorization, or output binding is invalid"
        )
    seal = _require_sha256(
        receipt.get("receipt_payload_sha256"),
        name="stable receipt payload seal",
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256", None)
    actual_seal = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if actual_seal != seal:
        raise DynamicReadyMaterializationError(
            "stable receipt payload seal does not match its canonical content"
        )


def _validate_diagnostic_plant_template(contract: Mapping[str, Any]) -> None:
    """Require a negatively authorized ActionBall contract as plant template.

    The measured teacher/physical-split branch deliberately does not consume the
    template's action/motion/ready binding.  Its only authority is the
    action-independent A3 plant extracted by :func:`_runtime_plant`; the exact
    Isaac nominal-hold probe must subsequently reproduce those values.
    """

    if contract.get("target_mode") != "action_ball":
        raise DynamicReadyMaterializationError(
            "diagnostic plant template is not an ActionBall contract"
        )
    training = contract.get("action_ball_training")
    authorization = (
        training.get("authorization") if isinstance(training, Mapping) else None
    )
    motion_admission = (
        training.get("motion_admission") if isinstance(training, Mapping) else None
    )
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("diagnostic_unauthorized") is not True
        or authorization.get("formal_evidence_prohibited") is not True
        or authorization.get("curriculum_promotion_prohibited") is not True
        or authorization.get("exact_export_prohibited") is not True
        or authorization.get("formal_judge_prohibited") is not True
        or not isinstance(motion_admission, Mapping)
        or motion_admission.get("diagnostic_unauthorized") is not True
        or motion_admission.get("training_authorized") is not False
    ):
        raise DynamicReadyMaterializationError(
            "measured teacher/physical-split plant template must remain diagnostic and "
            "training_authorized=false"
        )


def _load_measured_motion_identity(
    path: Path, *, expected_uid: str, expected_motion_sha256: str
) -> dict[str, Any]:
    """Validate the measured-retarget identity embedded in one exact NPZ."""

    try:
        with np.load(_pinned_snapshot_path(path), allow_pickle=False) as archive:
            required = {
                "joint_pos",
                "measured_racket_site_pos_w",
                "measured_racket_normal_w",
                "measured_racket_long_axis_w",
                "measured_racket_uid",
                "measured_racket_schema_version",
                "measured_racket_position_semantics",
                "measured_racket_normal_semantics",
                "measured_racket_long_axis_semantics",
                "measured_racket_robot_mount_normal_sign",
                "measured_racket_robot_butt_to_blade_axis_local",
                "measured_racket_robot_rigid_visual_mesh_sha256",
                "measured_racket_source_sha256",
                "measured_racket_retarget_receipt_sha256",
                "measured_racket_input_motion_sha256",
                "measured_racket_manifest_sha256",
                "measured_racket_catalog_sha256",
                "measured_racket_retarget_admitted",
                "measured_racket_joint_order_contract_id",
                "measured_racket_joint_order_contract_sha256",
            }
            missing = required - set(archive.files)
            if missing:
                raise DynamicReadyMaterializationError(
                    f"measured motion lacks identity fields {sorted(missing)}"
                )
            joint_pos = np.asarray(archive["joint_pos"])
            racket_position = np.asarray(
                archive["measured_racket_site_pos_w"], np.float64
            )
            racket_normal = np.asarray(
                archive["measured_racket_normal_w"], np.float64
            )
            racket_long_axis = np.asarray(
                archive["measured_racket_long_axis_w"], np.float64
            )

            def scalar(name: str) -> Any:
                value = np.asarray(archive[name]).reshape(-1)
                if value.size != 1:
                    raise DynamicReadyMaterializationError(
                        f"measured motion {name} must contain exactly one value"
                    )
                return value[0].item()

            def exact_integer(name: str) -> int:
                value = scalar(name)
                if (
                    isinstance(value, bool)
                    or type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) != float(int(value))
                ):
                    raise DynamicReadyMaterializationError(
                        f"measured motion {name} must be one exact integer"
                    )
                return int(value)

            uid = str(scalar("measured_racket_uid"))
            schema = exact_integer("measured_racket_schema_version")
            position_semantics = str(
                scalar("measured_racket_position_semantics")
            )
            normal_semantics = str(
                scalar("measured_racket_normal_semantics")
            )
            long_axis_semantics = str(
                scalar("measured_racket_long_axis_semantics")
            )
            mount_normal_sign = exact_integer(
                "measured_racket_robot_mount_normal_sign"
            )
            robot_axis_local = np.asarray(
                archive["measured_racket_robot_butt_to_blade_axis_local"],
                np.float64,
            ).reshape(-1)
            rigid_visual_mesh_sha256 = str(
                scalar("measured_racket_robot_rigid_visual_mesh_sha256")
            )
            measured_source_sha256 = str(
                scalar("measured_racket_source_sha256")
            )
            retarget_receipt_sha256 = str(
                scalar("measured_racket_retarget_receipt_sha256")
            )
            input_motion_sha256 = str(
                scalar("measured_racket_input_motion_sha256")
            )
            manifest_sha256 = str(scalar("measured_racket_manifest_sha256"))
            catalog_sha256 = str(scalar("measured_racket_catalog_sha256"))
            admitted = exact_integer("measured_racket_retarget_admitted")
            order_id = str(scalar("measured_racket_joint_order_contract_id"))
            order_sha = str(
                scalar("measured_racket_joint_order_contract_sha256")
            )
    except DynamicReadyMaterializationError:
        raise
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"cannot validate measured-retarget motion identity: {exc}"
        ) from exc
    if (
        uid != expected_uid
        or schema != EXPECTED_MEASURED_RACKET_SCHEMA
        or position_semantics != EXPECTED_MEASURED_RACKET_POSITION_SEMANTICS
        or normal_semantics != EXPECTED_MEASURED_RACKET_NORMAL_SEMANTICS
        or long_axis_semantics != EXPECTED_MEASURED_RACKET_LONG_AXIS_SEMANTICS
        or mount_normal_sign not in (-1, 1)
        or robot_axis_local.shape != (3,)
        or not np.array_equal(
            robot_axis_local,
            np.asarray(
                EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
                np.float64,
            ),
        )
        or rigid_visual_mesh_sha256
        != EXPECTED_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        or len(measured_source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in measured_source_sha256
        )
        or len(retarget_receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in retarget_receipt_sha256
        )
        or any(
            len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
            for digest in (
                input_motion_sha256,
                manifest_sha256,
                catalog_sha256,
            )
        )
        or admitted != 1
        or order_id != EXPECTED_MEASURED_JOINT_ORDER_CONTRACT_ID
        or order_sha != EXPECTED_MEASURED_JOINT_ORDER_CONTRACT_SHA256
        or joint_pos.ndim != 2
        or joint_pos.shape[0] < 2
        or joint_pos.shape[1] != 31
        or not np.all(np.isfinite(joint_pos))
        or racket_position.shape != (joint_pos.shape[0], 3)
        or racket_normal.shape != racket_position.shape
        or racket_long_axis.shape != racket_position.shape
        or not np.all(np.isfinite(racket_position))
        or not np.all(np.isfinite(racket_normal))
        or not np.all(np.isfinite(racket_long_axis))
        or float(
            np.max(np.abs(np.linalg.norm(racket_normal, axis=-1) - 1.0))
        )
        > 1.0e-3
        or float(
            np.max(np.abs(np.linalg.norm(racket_long_axis, axis=-1) - 1.0))
        )
        > 1.0e-3
        or float(
            np.max(
                np.abs(np.sum(racket_normal * racket_long_axis, axis=-1))
            )
        )
        > 1.0e-3
    ):
        raise DynamicReadyMaterializationError(
            "motion is not the exact admitted measured-retarget schema-v4 A3 clip"
        )
    return {
        "uid": uid,
        "motion_sha256": expected_motion_sha256,
        "frames": int(joint_pos.shape[0]),
        "measured_racket_schema_version": schema,
        "measured_racket_retarget_admitted": True,
        "joint_order_contract_id": order_id,
        "joint_order_contract_sha256": order_sha,
        "measured_racket_source_sha256": measured_source_sha256,
        "measured_racket_retarget_receipt_sha256": (
            retarget_receipt_sha256
        ),
        "measured_racket_input_motion_sha256": input_motion_sha256,
        "measured_racket_manifest_sha256": manifest_sha256,
        "measured_racket_catalog_sha256": catalog_sha256,
        "measured_racket_frame0": {
            "authority": "independent_schema_v4_measured_racket_channel",
            "motion_sha256": expected_motion_sha256,
            "frame_index": 0,
            "site_pos_w_m": racket_position[0].tolist(),
            "signed_face_normal_w": racket_normal[0].tolist(),
            "long_axis_w": racket_long_axis[0].tolist(),
            "position_semantics": position_semantics,
            "normal_semantics": normal_semantics,
            "long_axis_semantics": long_axis_semantics,
            "robot_mount_normal_sign": mount_normal_sign,
            "robot_butt_to_blade_axis_local": robot_axis_local.tolist(),
            "robot_rigid_visual_mesh_sha256": rigid_visual_mesh_sha256,
            "source_sha256": measured_source_sha256,
            "retarget_receipt_sha256": retarget_receipt_sha256,
            "input_motion_sha256": input_motion_sha256,
            "manifest_sha256": manifest_sha256,
            "catalog_sha256": catalog_sha256,
            "source_and_retarget_receipt_sha_semantics": (
                "opaque_labels_content_bound_by_exact_materialized_motion_sha"
            ),
        },
    }


def _validate_measured_retarget_l0_evidence(
    *,
    motion_path: Path,
    motion_sha256: str,
    measured_uid: str,
    bank_receipt: Mapping[str, Any],
    bank_receipt_sha256: str,
    mechanical_audit: Mapping[str, Any],
    allow_mechanical_unknown: bool,
) -> dict[str, Any]:
    """Cross-bind exact measured motion, bank receipt, and L0-style audit."""

    motion = _load_measured_motion_identity(
        motion_path,
        expected_uid=measured_uid,
        expected_motion_sha256=motion_sha256,
    )
    bank_authorization = bank_receipt.get("authorization")
    if (
        bank_receipt.get("schema_version") != 1
        or bank_receipt.get("kind") != MEASURED_BANK_RECEIPT_KIND
        or not isinstance(bank_authorization, Mapping)
        or bank_authorization.get("diagnostic_unauthorized") is not True
        or bank_authorization.get("training") is not False
        or bank_authorization.get("promotion") is not False
        or bank_authorization.get("deployment") is not False
        or bank_authorization.get("mechanical_admission") is not False
    ):
        raise DynamicReadyMaterializationError(
            "measured bank receipt is not the diagnostic-only schema-v4 import"
        )
    bank_authorities = bank_receipt.get("authorities")
    bank_manifest_authority = (
        bank_authorities.get("source_manifest")
        if isinstance(bank_authorities, Mapping)
        else None
    )
    bank_catalog_authority = (
        bank_authorities.get("signed_catalog")
        if isinstance(bank_authorities, Mapping)
        else None
    )
    bank_source_manifest = bank_receipt.get("source_manifest")
    if (
        not isinstance(bank_manifest_authority, Mapping)
        or not isinstance(bank_catalog_authority, Mapping)
        or not isinstance(bank_source_manifest, Mapping)
        or bank_manifest_authority.get("sha256")
        != motion["measured_racket_manifest_sha256"]
        or bank_source_manifest.get("sha256")
        != motion["measured_racket_manifest_sha256"]
        or bank_catalog_authority.get("sha256")
        != motion["measured_racket_catalog_sha256"]
    ):
        raise DynamicReadyMaterializationError(
            "measured bank manifest/catalog authorities do not bind the exact "
            "schema-v4 motion"
        )
    bank_rows = [
        row
        for row in bank_receipt.get("actions", ())
        if isinstance(row, Mapping) and row.get("uid") == measured_uid
    ]
    if (
        len(bank_rows) != 1
        or bank_rows[0].get("sha256") != motion_sha256
        or bank_rows[0].get("frames") != motion["frames"]
        or bank_rows[0].get("robot_mount_normal_sign")
        != motion["measured_racket_frame0"]["robot_mount_normal_sign"]
    ):
        raise DynamicReadyMaterializationError(
            "measured bank receipt does not bind the exact selected motion"
        )
    all_bank_rows = bank_receipt.get("actions")
    bank_denominators = bank_receipt.get("denominators")
    if (
        not isinstance(all_bank_rows, list)
        or not isinstance(bank_denominators, Mapping)
        or bank_denominators.get("materialized_npz") != len(all_bank_rows)
    ):
        raise DynamicReadyMaterializationError(
            "measured bank receipt has an incomplete action denominator"
        )
    bank_identity = {
        (row.get("uid"), row.get("sha256"))
        for row in all_bank_rows
        if isinstance(row, Mapping)
    }
    if len(bank_identity) != len(all_bank_rows):
        raise DynamicReadyMaterializationError(
            "measured bank receipt action identities are not unique and complete"
        )

    mechanical_authorization = mechanical_audit.get("authorization")
    mechanical_source = mechanical_audit.get("sources")
    bank_source = (
        mechanical_source.get("bank_import_receipt")
        if isinstance(mechanical_source, Mapping)
        else None
    )
    if (
        mechanical_audit.get("schema_version") != 1
        or mechanical_audit.get("kind") != MEASURED_MECHANICAL_AUDIT_KIND
        or mechanical_audit.get("diagnostic_unauthorized") is not True
        or not isinstance(mechanical_authorization, Mapping)
        or mechanical_authorization.get("training") is not False
        or mechanical_authorization.get("promotion") is not False
        or mechanical_authorization.get("deployment") is not False
        or mechanical_authorization.get("hardware") is not False
        or mechanical_authorization.get("mechanical_admission") is not False
        or not isinstance(bank_source, Mapping)
        or bank_source.get("sha256") != bank_receipt_sha256
        or bank_source.get("kind") != bank_receipt.get("kind")
    ):
        raise DynamicReadyMaterializationError(
            "mechanical audit is not the exact diagnostic-only measured-bank audit"
        )
    mechanical_rows = [
        row
        for row in mechanical_audit.get("actions", ())
        if isinstance(row, Mapping) and row.get("uid") == measured_uid
    ]
    if len(mechanical_rows) != 1 or mechanical_rows[0].get(
        "sha256"
    ) != motion_sha256:
        raise DynamicReadyMaterializationError(
            "mechanical audit does not bind the exact selected motion"
        )
    all_mechanical_rows = mechanical_audit.get("actions")
    mechanical_denominators = mechanical_audit.get("denominators")
    if (
        not isinstance(all_mechanical_rows, list)
        or not isinstance(mechanical_denominators, Mapping)
        or mechanical_denominators.get("actions_expected")
        != len(all_bank_rows)
        or mechanical_denominators.get("actions_audited")
        != len(all_bank_rows)
    ):
        raise DynamicReadyMaterializationError(
            "mechanical audit does not cover the exact measured bank"
        )
    mechanical_identity = {
        (row.get("uid"), row.get("sha256"))
        for row in all_mechanical_rows
        if isinstance(row, Mapping)
    }
    if mechanical_identity != bank_identity or len(all_mechanical_rows) != len(
        all_bank_rows
    ):
        raise DynamicReadyMaterializationError(
            "mechanical audit action identities differ from the exact measured bank"
        )
    selected = mechanical_rows[0]
    verdict = selected.get("mechanical_verdict")
    if selected.get("kinematic_limit_verdict") != "PASS" or verdict == "FAIL":
        raise DynamicReadyMaterializationError(
            "measured teacher source has an observed kinematic/mechanical failure"
        )
    if verdict not in ("PASS", "UNKNOWN"):
        raise DynamicReadyMaterializationError(
            "measured teacher mechanical verdict is invalid"
        )
    if verdict == "UNKNOWN" and allow_mechanical_unknown is not True:
        raise DynamicReadyMaterializationError(
            "mechanical verdict is UNKNOWN; pass "
            "--allow-mechanical-unknown-diagnostic explicitly"
        )
    return {
        **motion,
        "kinematic_limit_verdict": "PASS",
        "mechanical_verdict": verdict,
        "mechanical_admitted": selected.get("mechanical_admitted") is True,
        "unknown_explicitly_accepted_for_sim_diagnostic": verdict == "UNKNOWN",
        "teacher_reference_semantics": "exact_original_measured_motion_frame0",
        "physical_birth_authority": "separate_content_pinned_composition",
        "training_authorized": False,
        "diagnostic_unauthorized": True,
    }


def _plain_finite_vector(
    value: object, *, name: str, size: int
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise DynamicReadyMaterializationError(
            f"{name} must contain exactly {size} entries"
        )
    if any(
        isinstance(item, bool)
        or type(item) not in (int, float)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise DynamicReadyMaterializationError(
            f"{name} must contain plain finite numbers"
        )
    return np.asarray(value, np.float64)


def _plain_finite_matrix(
    value: object, *, name: str, rows: int, columns: int
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise DynamicReadyMaterializationError(
            f"{name} must contain exactly {rows} rows"
        )
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (rows, columns) or not np.all(np.isfinite(matrix)):
        raise DynamicReadyMaterializationError(
            f"{name} must be a finite [{rows},{columns}] matrix"
        )
    return matrix


def _physx_control_position_limits(
    value: object,
    *,
    joint_names: tuple[str, ...],
    qdes_limits: np.ndarray,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(
        _PHYSX_CONTROL_POSITION_LIMIT_KEYS
    ):
        raise DynamicReadyMaterializationError(
            "runtime physx_control_position_limits fields are incomplete or unknown"
        )
    expected_names = list(_PHYSX_CONTROL_POSITION_LIMIT_SELECTED_JOINT_NAMES)
    selected_indices = [
        index for index, name in enumerate(joint_names) if name in expected_names
    ]
    if (
        value["schema_version"] != 1
        or value["backend"] != "physx_root_view_dof_limits"
        or type(value["inset_fraction_per_side_hard_span"]) is not float
        or value["inset_fraction_per_side_hard_span"] != 0.02
        or value["selected_joint_names"] != expected_names
        or [joint_names[index] for index in selected_indices] != expected_names
        or type(value["unselected_joint_count"]) is not int
        or value["unselected_joint_count"] != 27
        or value["unselected_limits_equal_mechanical"] is not True
        or value["articulation_mechanical_ledger_unchanged"] is not True
        or value["soft_qdes_ledger_unchanged"] is not True
    ):
        raise DynamicReadyMaterializationError(
            "runtime PhysX H_ctrl identity or ledger proof is invalid"
        )
    mechanical = _plain_finite_matrix(
        value["mechanical_joint_pos_limits"],
        name="physx_control_position_limits.mechanical_joint_pos_limits",
        rows=31,
        columns=2,
    )
    control = _plain_finite_matrix(
        value["control_joint_pos_limits"],
        name="physx_control_position_limits.control_joint_pos_limits",
        rows=31,
        columns=2,
    )
    selected = set(selected_indices)
    for index in range(31):
        hard = mechanical[index]
        constrained = control[index]
        if hard[0] >= hard[1] or constrained[0] >= constrained[1]:
            raise DynamicReadyMaterializationError(
                "runtime PhysX H_ctrl/H_mech contains an empty row"
            )
        if index not in selected:
            if not np.array_equal(constrained, hard):
                raise DynamicReadyMaterializationError(
                    "runtime unselected H_ctrl must equal H_mech"
                )
        else:
            span = hard[1] - hard[0]
            if not (
                math.isclose(
                    constrained[0], hard[0] + 0.02 * span,
                    rel_tol=0.0, abs_tol=2.0e-7,
                )
                and math.isclose(
                    constrained[1], hard[1] - 0.02 * span,
                    rel_tol=0.0, abs_tol=2.0e-7,
                )
                and hard[0] < constrained[0] < constrained[1] < hard[1]
            ):
                raise DynamicReadyMaterializationError(
                    "runtime selected H_ctrl must be two percent per side inside H_mech"
                )
        if not (
            constrained[0]
            <= qdes_limits[index, 0]
            < qdes_limits[index, 1]
            <= constrained[1]
        ):
            raise DynamicReadyMaterializationError(
                "runtime qdes envelope must remain inside H_ctrl"
            )
    return {
        "schema_version": 1,
        "backend": "physx_root_view_dof_limits",
        "inset_fraction_per_side_hard_span": 0.02,
        "selected_joint_names": expected_names,
        "mechanical_joint_pos_limits": mechanical.tolist(),
        "control_joint_pos_limits": control.tolist(),
        "unselected_joint_count": 27,
        "unselected_limits_equal_mechanical": True,
        "articulation_mechanical_ledger_unchanged": True,
        "soft_qdes_ledger_unchanged": True,
    }


def _load_motion_frame0_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(_pinned_snapshot_path(path), allow_pickle=False) as archive:
            joint_pos = np.asarray(archive["joint_pos"], np.float64)
            body_pos = np.asarray(archive["body_pos_w"], np.float64)
            body_quat = np.asarray(archive["body_quat_w"], np.float64)
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"cannot load stable motion: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        joint_pos.ndim != 2
        or joint_pos.shape[0] < 2
        or joint_pos.shape[1] != 31
        or body_pos.ndim != 3
        or body_pos.shape[0] != joint_pos.shape[0]
        or body_pos.shape[1] < 1
        or body_pos.shape[2] != 3
        or body_quat.shape != (joint_pos.shape[0], body_pos.shape[1], 4)
        or not np.all(np.isfinite(joint_pos))
        or not np.all(np.isfinite(body_pos))
        or not np.all(np.isfinite(body_quat))
    ):
        raise DynamicReadyMaterializationError(
            "stable motion frame arrays are malformed or non-finite"
        )
    return joint_pos[0].copy(), body_pos[0, 0].copy(), body_quat[0, 0].copy()


def _load_motion_frame0(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_pos, root_pos, root_quat = _load_motion_frame0_arrays(path)
    root_quat_norm = float(np.linalg.norm(root_quat))
    if not math.isfinite(root_quat_norm) or root_quat_norm <= 1.0e-12:
        raise DynamicReadyMaterializationError(
            "stable motion frame-0 root quaternion is degenerate"
        )
    # Motion archives are float32, so a mathematically unit quaternion may be
    # a few 1e-8 away from unit length after promotion to float64.  Normalize
    # deterministically before the exact MuJoCo ready-state audit; do not make
    # a storage-rounding artifact look like a physical birth failure.
    root_quat /= root_quat_norm
    return joint_pos, root_pos, root_quat


def _load_motion_frame0_exact(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frame 0 without changing the stored measured quaternion values."""

    joint_pos, root_pos, root_quat = _load_motion_frame0_arrays(path)
    root_quat_norm = float(np.linalg.norm(root_quat))
    if (
        not math.isfinite(root_quat_norm)
        or abs(root_quat_norm - 1.0) > 2.0e-6
    ):
        raise DynamicReadyMaterializationError(
            "measured direct-frame0 root quaternion is not unit within "
            "float32 storage tolerance"
        )
    return joint_pos, root_pos, root_quat


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _stored_ready_state_sha256(
    joint_pos: Sequence[float],
    root_pos_w: Sequence[float],
    root_quat_wxyz: Sequence[float],
) -> str:
    """Hash emitted float64 state arrays without normalizing quaternion bytes."""

    digest = hashlib.sha256()
    for label, value, shape in (
        ("joint_pos", joint_pos, (31,)),
        ("root_pos_w", root_pos_w, (3,)),
        ("root_quat_wxyz", root_quat_wxyz, (4,)),
    ):
        array = np.ascontiguousarray(np.asarray(value, np.float64))
        if array.shape != shape or not np.isfinite(array).all():
            raise DynamicReadyMaterializationError(
                f"stored ready state has malformed {label}"
            )
        digest.update(label.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _load_physical_birth_seed(
    seed: Mapping[str, Any], *, joint_names: Sequence[str]
) -> dict[str, Any]:
    """Load only numerical root/leg values from an old dynamic-ready artifact.

    The source's model identity, LP result, nominal-hold history, and action
    identity have no authority in the newly composed candidate.  Its content
    seal and negative authorization are still required so the numerical seed
    cannot drift silently or import a promoted claim.
    """

    if (
        seed.get("schema_version") != SCHEMA_VERSION
        or seed.get("kind") != PHYSICAL_BIRTH_SEED_KIND
    ):
        raise DynamicReadyMaterializationError(
            "physical-birth seed kind/schema mismatch"
        )
    content_sha = _require_sha256(
        seed.get("content_sha256"), name="physical-birth seed content seal"
    )
    unsigned = dict(seed)
    unsigned.pop("content_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != content_sha:
        raise DynamicReadyMaterializationError(
            "physical-birth seed content seal mismatch"
        )
    authorization = seed.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(key) is not False
        for key in (
            "training_authorized",
            "deployment_authorized",
            "hardware_authorized",
            "isaac_nominal_hold_validated",
        )
    ):
        raise DynamicReadyMaterializationError(
            "physical-birth numerical seed must remain unauthorized"
        )
    robot = seed.get("robot")
    physical = seed.get("physical_ready")
    expected_names = tuple(str(name) for name in joint_names)
    if (
        not isinstance(robot, Mapping)
        or robot.get("family") != "AgiBot A3"
        or tuple(str(name) for name in robot.get("joint_names", ()))
        != expected_names
        or not isinstance(physical, Mapping)
    ):
        raise DynamicReadyMaterializationError(
            "physical-birth seed joint mapping drifted"
        )
    seed_q = _plain_finite_vector(
        physical.get("joint_pos_rad"), name="physical-birth seed joint_pos", size=31
    )
    seed_qd = _plain_finite_vector(
        physical.get("joint_vel_radps"),
        name="physical-birth seed joint_vel",
        size=31,
    )
    if not np.array_equal(seed_qd, np.zeros(31, np.float64)):
        raise DynamicReadyMaterializationError(
            "physical-birth seed joint velocity must be exact zero"
        )
    root_pos = _plain_finite_vector(
        physical.get("root_pos_w_m"), name="physical-birth seed root_pos", size=3
    )
    root_quat = _plain_finite_vector(
        physical.get("root_quat_wxyz"),
        name="physical-birth seed root_quat",
        size=4,
    )
    quat_norm = float(np.linalg.norm(root_quat))
    if not math.isfinite(quat_norm) or quat_norm <= 1.0e-12:
        raise DynamicReadyMaterializationError(
            "physical-birth seed root quaternion is degenerate"
        )
    root_quat /= quat_norm
    canonical_leg_names = frozenset(str(name) for name in grounded.LEG_JOINT_NAMES)
    leg_indices = np.asarray(
        [
            index
            for index, name in enumerate(expected_names)
            if name in canonical_leg_names
        ],
        dtype=np.int64,
    )
    if (
        leg_indices.shape != (12,)
        or canonical_leg_names
        != frozenset(expected_names[int(index)] for index in leg_indices)
    ):
        raise DynamicReadyMaterializationError(
            "canonical physical-birth leg mapping drifted from exact 12-D"
        )
    return {
        "source_action_id": str(seed.get("action_id")),
        "source_content_sha256": content_sha,
        "joint_pos_rad": seed_q,
        "root_pos_w_m": root_pos,
        "root_quat_wxyz": root_quat,
        "leg_joint_indices": leg_indices,
        "leg_joint_names": [expected_names[int(index)] for index in leg_indices],
    }


def _load_seed_hold_transport(
    seed: Mapping[str, Any], *, joint_names: Sequence[str]
) -> dict[str, Any]:
    """Load numerical qdes/action/tau without inheriting the old hold claim."""

    physical = seed.get("physical_ready")
    hold = seed.get("hold_candidate")
    plant = seed.get("runtime_plant")
    expected_names = tuple(str(name) for name in joint_names)
    if (
        not isinstance(physical, Mapping)
        or not isinstance(hold, Mapping)
        or not isinstance(plant, Mapping)
        or tuple(str(name) for name in plant.get("joint_names", ()))
        != expected_names
    ):
        raise DynamicReadyMaterializationError(
            "seed transport requires a complete same-order hold candidate"
        )
    q = _plain_finite_vector(
        physical.get("joint_pos_rad"), name="seed transport q", size=31
    )
    qdes = _plain_finite_vector(
        hold.get("hold_qdes_joint_pos_rad"),
        name="seed transport qdes",
        size=31,
    )
    action = _plain_finite_vector(
        hold.get("normalized_actor_action"),
        name="seed transport normalized action",
        size=31,
    )
    tau = _plain_finite_vector(
        hold.get("actuator_generalized_force_runtime_order_nm"),
        name="seed transport actuator force",
        size=31,
    )
    kp = _plain_finite_vector(
        plant.get("joint_stiffness"), name="seed transport kp", size=31
    )
    default_q = _plain_finite_vector(
        plant.get("default_joint_pos_rad"),
        name="seed transport default q",
        size=31,
    )
    scale = _plain_finite_vector(
        plant.get("action_scale_rad"),
        name="seed transport action scale",
        size=31,
    )
    lower = _plain_finite_vector(
        plant.get("executed_qdes_lower_rad"),
        name="seed transport qdes lower",
        size=31,
    )
    upper = _plain_finite_vector(
        plant.get("executed_qdes_upper_rad"),
        name="seed transport qdes upper",
        size=31,
    )
    if (
        np.any(kp <= 0.0)
        or np.any(scale <= 0.0)
        or np.any(lower >= upper)
        or np.any(qdes < lower)
        or np.any(qdes > upper)
        or not np.allclose(q + tau / kp, qdes, rtol=0.0, atol=2.0e-10)
        or not np.allclose(
            default_q + scale * action, qdes, rtol=0.0, atol=2.0e-10
        )
    ):
        raise DynamicReadyMaterializationError(
            "seed transport qdes/action/tau identity is invalid"
        )
    return {
        "q": q,
        "qdes": qdes,
        "normalized_action": action,
        "tau_runtime": tau,
        "kp": kp,
        "default_q": default_q,
        "action_scale": scale,
        "executed_qdes_lower": lower,
        "executed_qdes_upper": upper,
    }


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        np.float64,
    )


def _quat_rotation_wxyz(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, np.float64)
    norm = float(np.linalg.norm(quat))
    if quat.shape != (4,) or not np.all(np.isfinite(quat)) or norm <= 1.0e-12:
        raise DynamicReadyMaterializationError(
            "quaternion rotation requires one finite nonzero quaternion"
        )
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        np.float64,
    )


def _root_yaw_rad(quaternion_wxyz: Sequence[float]) -> float:
    quat = np.asarray(quaternion_wxyz, np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise DynamicReadyMaterializationError(
            "root yaw requires one finite quaternion"
        )
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise DynamicReadyMaterializationError("root yaw quaternion is degenerate")
    w, x, y, z = quat / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _align_seed_world_yaw_to_teacher(
    *,
    seed: Mapping[str, Any],
    teacher_root_quat: np.ndarray,
    seed_foot_positions_w: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Yaw-rotate the whole lower-body seed about its support centroid.

    A left world-Z rotation changes only heading.  Applying its matching SE(2)
    translation to the root keeps the midpoint of the two seed feet fixed, so
    the numerical stand is not silently moved relative to the table.  Leg
    angles remain byte-identical; the current MJCF static and ground-LP gates
    must still re-prove the resulting physical birth after the teacher upper
    body is overlaid.
    """

    root_pos = np.asarray(seed["root_pos_w_m"], np.float64)
    root_quat = np.asarray(seed["root_quat_wxyz"], np.float64)
    teacher_quat = np.asarray(teacher_root_quat, np.float64)
    feet = np.asarray(seed_foot_positions_w, np.float64)
    if (
        root_pos.shape != (3,)
        or root_quat.shape != (4,)
        or teacher_quat.shape != (4,)
        or feet.shape != (2, 3)
        or not np.all(np.isfinite(root_pos))
        or not np.all(np.isfinite(root_quat))
        or not np.all(np.isfinite(teacher_quat))
        or not np.all(np.isfinite(feet))
    ):
        raise DynamicReadyMaterializationError(
            "seed yaw alignment inputs are malformed"
        )
    seed_yaw = _root_yaw_rad(root_quat)
    teacher_yaw = _root_yaw_rad(teacher_quat)
    delta = math.atan2(
        math.sin(teacher_yaw - seed_yaw),
        math.cos(teacher_yaw - seed_yaw),
    )
    cosine = math.cos(delta)
    sine = math.sin(delta)
    rotation_z = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    support_pivot = feet[:, :2].mean(axis=0)
    aligned_root = root_pos.copy()
    aligned_root[:2] = support_pivot + rotation_z[:2, :2] @ (
        root_pos[:2] - support_pivot
    )
    yaw_quat = np.asarray(
        [math.cos(0.5 * delta), 0.0, 0.0, math.sin(0.5 * delta)],
        np.float64,
    )
    aligned_quat = _quat_multiply_wxyz(yaw_quat, root_quat)
    aligned_quat /= np.linalg.norm(aligned_quat)
    aligned_yaw = _root_yaw_rad(aligned_quat)
    yaw_error = math.atan2(
        math.sin(aligned_yaw - teacher_yaw),
        math.cos(aligned_yaw - teacher_yaw),
    )
    seed_rotation = _quat_rotation_wxyz(root_quat)
    aligned_rotation = _quat_rotation_wxyz(aligned_quat)
    seed_tilt = math.acos(float(np.clip(seed_rotation[2, 2], -1.0, 1.0)))
    aligned_tilt = math.acos(
        float(np.clip(aligned_rotation[2, 2], -1.0, 1.0))
    )
    expected_feet = (
        np.asarray([support_pivot[0], support_pivot[1], 0.0])
        + (rotation_z @ (
            feet
            - np.asarray([support_pivot[0], support_pivot[1], 0.0])
        ).T).T
    )
    if (
        abs(yaw_error) > 1.0e-12
        or abs(seed_tilt - aligned_tilt) > 1.0e-12
        or np.max(np.abs(expected_feet[:, 2] - feet[:, 2])) > 1.0e-12
        or np.max(
            np.abs(expected_feet[:, :2].mean(axis=0) - support_pivot)
        )
        > 1.0e-12
    ):
        raise DynamicReadyMaterializationError(
            "seed yaw alignment failed its rigid SE(2) invariants"
        )
    aligned = dict(seed)
    aligned["root_pos_w_m"] = aligned_root
    aligned["root_quat_wxyz"] = aligned_quat
    aligned["seed_world_yaw_alignment"] = {
        "schema_version": 1,
        "semantics": MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS,
        "seed_root_yaw_rad": seed_yaw,
        "teacher_root_yaw_rad": teacher_yaw,
        "applied_world_z_rotation_rad": delta,
        "aligned_root_yaw_rad": aligned_yaw,
        "aligned_minus_teacher_yaw_rad": yaw_error,
        "support_pivot_xy_w_m": support_pivot.tolist(),
        "seed_root_pos_w_m": root_pos.tolist(),
        "aligned_root_pos_w_m": aligned_root.tolist(),
        "seed_root_quat_wxyz": root_quat.tolist(),
        "aligned_root_quat_wxyz": aligned_quat.tolist(),
        "seed_root_tilt_rad": seed_tilt,
        "aligned_root_tilt_rad": aligned_tilt,
        "expected_aligned_seed_foot_positions_w_m": expected_feet.tolist(),
        "support_centroid_preserved": True,
        "seed_tilt_preserved": True,
        "teacher_yaw_exact": True,
    }
    return aligned


def _audit_realized_seed_yaw_alignment(
    *,
    backend: grounded.GroundedReadyBackend,
    seed_foot_poses: Sequence[grounded.FootPose],
    ready: grounded.ReadyState,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove FK realizes the recorded rigid world-Z seed transform."""

    if (
        alignment.get("semantics")
        != MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS
        or len(seed_foot_poses) != 2
    ):
        raise DynamicReadyMaterializationError(
            "seed yaw-alignment audit received the wrong contract"
        )
    delta = float(alignment["applied_world_z_rotation_rad"])
    pivot = np.asarray(alignment["support_pivot_xy_w_m"], np.float64)
    cosine = math.cos(delta)
    sine = math.sin(delta)
    rotation_z = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    pivot3 = np.asarray([pivot[0], pivot[1], 0.0], np.float64)
    expected_positions = np.stack(
        [
            pivot3
            + rotation_z
            @ (np.asarray(pose.position_w, np.float64) - pivot3)
            for pose in seed_foot_poses
        ],
        axis=0,
    )
    expected_rotations = np.stack(
        [
            rotation_z @ np.asarray(pose.rotation_w, np.float64)
            for pose in seed_foot_poses
        ],
        axis=0,
    )
    realized = backend.foot_poses(ready)
    realized_positions = np.stack(
        [np.asarray(pose.position_w, np.float64) for pose in realized],
        axis=0,
    )
    realized_rotations = np.stack(
        [np.asarray(pose.rotation_w, np.float64) for pose in realized],
        axis=0,
    )
    position_error = float(
        np.max(np.abs(realized_positions - expected_positions))
    )
    rotation_error = float(
        np.max(np.abs(realized_rotations - expected_rotations))
    )
    support_centroid_error = float(
        np.max(
            np.abs(
                realized_positions[:, :2].mean(axis=0)
                - np.stack(
                    [np.asarray(pose.position_w)[:2] for pose in seed_foot_poses]
                ).mean(axis=0)
            )
        )
    )
    foot_height_error = float(
        np.max(
            np.abs(
                realized_positions[:, 2]
                - np.asarray(
                    [pose.position_w[2] for pose in seed_foot_poses],
                    np.float64,
                )
            )
        )
    )
    tolerance = 2.0e-10
    if (
        position_error > tolerance
        or rotation_error > tolerance
        or support_centroid_error > tolerance
        or foot_height_error > tolerance
    ):
        raise DynamicReadyMaterializationError(
            "teacher-yaw alignment did not rigidly preserve the seed support"
        )
    return {
        "authority": "current_exact_mjcf_fk",
        "semantics": MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS,
        "passed": True,
        "absolute_tolerance": tolerance,
        "maximum_foot_position_error_m": position_error,
        "maximum_foot_rotation_matrix_error": rotation_error,
        "support_centroid_xy_error_m": support_centroid_error,
        "maximum_foot_height_error_m": foot_height_error,
        "expected_foot_positions_w_m": expected_positions.tolist(),
        "realized_foot_positions_w_m": realized_positions.tolist(),
    }


def _compose_measured_physical_birth(
    *,
    teacher_q: np.ndarray,
    teacher_root_pos: np.ndarray,
    teacher_root_quat: np.ndarray,
    seed: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compose seed root/legs with an otherwise exact teacher frame 0."""

    teacher_q = np.asarray(teacher_q, np.float64)
    teacher_root_pos = np.asarray(teacher_root_pos, np.float64)
    teacher_root_quat = np.asarray(teacher_root_quat, np.float64)
    if (
        teacher_q.shape != (31,)
        or teacher_root_pos.shape != (3,)
        or teacher_root_quat.shape != (4,)
        or not np.all(np.isfinite(teacher_q))
        or not np.all(np.isfinite(teacher_root_pos))
        or not np.all(np.isfinite(teacher_root_quat))
    ):
        raise DynamicReadyMaterializationError(
            "measured teacher frame 0 is malformed before birth composition"
        )
    leg_indices = np.asarray(seed["leg_joint_indices"], np.int64)
    seed_q = np.asarray(seed["joint_pos_rad"], np.float64)
    ready_q = teacher_q.copy()
    ready_q[leg_indices] = seed_q[leg_indices]
    nonleg_mask = np.ones(31, dtype=bool)
    nonleg_mask[leg_indices] = False
    if not np.array_equal(ready_q[nonleg_mask], teacher_q[nonleg_mask]):
        raise DynamicReadyMaterializationError(
            "physical birth changed a non-leg teacher joint"
        )
    root_pos = np.asarray(seed["root_pos_w_m"], np.float64).copy()
    root_quat = np.asarray(seed["root_quat_wxyz"], np.float64).copy()
    provenance = {
        "semantics": MEASURED_BIRTH_SHARED_LOWER_SEMANTICS,
        "leg_joint_indices": leg_indices.tolist(),
        "leg_joint_names": list(seed["leg_joint_names"]),
        "nonleg_joint_indices": np.flatnonzero(nonleg_mask).tolist(),
        "nonleg_joint_names": [
            grounded.RUNTIME_JOINT_NAMES[index]
            for index in np.flatnonzero(nonleg_mask)
        ],
        "teacher_nonleg_exactly_preserved": True,
        "physical_minus_teacher_joint_pos_rad": (ready_q - teacher_q).tolist(),
        "physical_minus_teacher_root_pos_m": (
            root_pos - teacher_root_pos
        ).tolist(),
        "physical_root_quat_wxyz": root_quat.tolist(),
        "teacher_root_quat_wxyz": teacher_root_quat.tolist(),
        "teacher_and_physical_birth_differ": bool(
            not np.array_equal(ready_q, teacher_q)
            or not np.array_equal(root_pos, teacher_root_pos)
            or not np.array_equal(root_quat, teacher_root_quat)
        ),
    }
    alignment = seed.get("seed_world_yaw_alignment")
    if not isinstance(alignment, Mapping):
        raise DynamicReadyMaterializationError(
            "measured physical birth requires teacher-yaw-aligned seed"
        )
    provenance["seed_world_yaw_alignment"] = dict(alignment)
    return ready_q, root_pos, root_quat, provenance


def _compose_measured_full_seed_physical_birth(
    *,
    teacher_q: np.ndarray,
    teacher_root_pos: np.ndarray,
    teacher_root_quat: np.ndarray,
    seed: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep the exact teacher as reference while birthing from all seed joints."""

    teacher_q = np.asarray(teacher_q, np.float64)
    teacher_root_pos = np.asarray(teacher_root_pos, np.float64)
    teacher_root_quat = np.asarray(teacher_root_quat, np.float64)
    seed_q = np.asarray(seed["joint_pos_rad"], np.float64)
    root_pos = np.asarray(seed["root_pos_w_m"], np.float64).copy()
    root_quat = np.asarray(seed["root_quat_wxyz"], np.float64).copy()
    if (
        teacher_q.shape != (31,)
        or teacher_root_pos.shape != (3,)
        or teacher_root_quat.shape != (4,)
        or seed_q.shape != (31,)
        or root_pos.shape != (3,)
        or root_quat.shape != (4,)
        or not np.all(np.isfinite(teacher_q))
        or not np.all(np.isfinite(teacher_root_pos))
        or not np.all(np.isfinite(teacher_root_quat))
        or not np.all(np.isfinite(seed_q))
        or not np.all(np.isfinite(root_pos))
        or not np.all(np.isfinite(root_quat))
    ):
        raise DynamicReadyMaterializationError(
            "measured teacher or full-seed physical birth is malformed"
        )
    alignment = seed.get("seed_world_yaw_alignment")
    if not isinstance(alignment, Mapping):
        raise DynamicReadyMaterializationError(
            "measured full-seed birth requires teacher-yaw-aligned seed"
        )
    leg_indices = np.asarray(seed["leg_joint_indices"], np.int64)
    nonleg_mask = np.ones(31, dtype=bool)
    nonleg_mask[leg_indices] = False
    nonleg_indices = np.flatnonzero(nonleg_mask)
    provenance = {
        "semantics": MEASURED_BIRTH_FULL_SEED_SEMANTICS,
        "leg_joint_indices": leg_indices.tolist(),
        "leg_joint_names": list(seed["leg_joint_names"]),
        "nonleg_joint_indices": nonleg_indices.tolist(),
        "nonleg_joint_names": [
            grounded.RUNTIME_JOINT_NAMES[index] for index in nonleg_indices
        ],
        "teacher_nonleg_exactly_preserved": False,
        "seed_all_joints_exactly_preserved": True,
        "seed_joint_indices": list(range(31)),
        "seed_joint_names": list(grounded.RUNTIME_JOINT_NAMES),
        "physical_minus_teacher_joint_pos_rad": (
            seed_q - teacher_q
        ).tolist(),
        "physical_minus_teacher_root_pos_m": (
            root_pos - teacher_root_pos
        ).tolist(),
        "physical_root_quat_wxyz": root_quat.tolist(),
        "teacher_root_quat_wxyz": teacher_root_quat.tolist(),
        "teacher_and_physical_birth_differ": bool(
            not np.array_equal(seed_q, teacher_q)
            or not np.array_equal(root_pos, teacher_root_pos)
            or not np.array_equal(root_quat, teacher_root_quat)
        ),
        "seed_world_yaw_alignment": dict(alignment),
    }
    return seed_q.copy(), root_pos, root_quat, provenance


def nearest_feasible_scalar_boundary(
    *,
    current: float,
    lower: float,
    upper: float,
    initial_step: float,
    slack_at: Any,
) -> tuple[float, dict[str, Any]]:
    """Find the nearest exact feasible scalar without assuming a derivative.

    ``slack_at(q) >= 0`` means feasible.  Both directions are bracketed because
    gravity torque can change faster than the PD torque envelope; the tempting
    ``shortfall / kp`` direction is therefore not generally correct.
    """

    if not lower < current < upper or initial_step <= 0.0:
        raise DynamicReadyMaterializationError(
            "scalar hold projection received an invalid search interval"
        )
    origin_slack = float(slack_at(current))
    if not math.isfinite(origin_slack) or origin_slack >= 0.0:
        raise DynamicReadyMaterializationError(
            "scalar hold projection requires one finite infeasible origin"
        )
    candidates: list[tuple[float, float, int, float]] = []
    for direction, limit in ((-1, lower), (1, upper)):
        infeasible = current
        step = initial_step
        evaluations = 1
        for _ in range(32):
            trial = current + direction * step
            trial = max(lower, trial) if direction < 0 else min(upper, trial)
            slack = float(slack_at(trial))
            evaluations += 1
            if not math.isfinite(slack):
                raise DynamicReadyMaterializationError(
                    "scalar hold projection encountered non-finite exact slack"
                )
            if slack >= 0.0:
                feasible = trial
                for _ in range(60):
                    midpoint = 0.5 * (infeasible + feasible)
                    if float(slack_at(midpoint)) >= 0.0:
                        feasible = midpoint
                    else:
                        infeasible = midpoint
                candidates.append(
                    (abs(feasible - current), feasible, evaluations + 60, slack)
                )
                break
            infeasible = trial
            if trial == limit:
                break
            step *= 2.0
    if not candidates:
        raise DynamicReadyMaterializationError(
            "no contact-free hold boundary exists inside the executed qdes envelope"
        )
    distance, value, evaluations, bracket_slack = min(candidates)
    return value, {
        "origin_slack_nm": origin_slack,
        "selected_delta_rad": value - current,
        "selected_boundary_rad": value,
        "absolute_delta_rad": distance,
        "exact_slack_evaluations": evaluations,
        "outer_feasible_bracket_slack_nm": bracket_slack,
    }


def _compose_measured_direct_frame0_physical_birth(
    *,
    teacher_q: np.ndarray,
    teacher_root_pos: np.ndarray,
    teacher_root_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Use the exact measured teacher frame 0 as a diagnostic physical birth.

    This deliberately skips every historical numerical ready seed.  It does not
    skip any downstream gate: the current exact-MJCF static audit, double-support
    ground LP, runtime qdes/effort envelope, and exact Isaac nominal-hold/table
    gate remain mandatory.  The returned vectors are copies so later solving
    cannot mutate the teacher reference in place.
    """

    teacher_q = np.asarray(teacher_q, np.float64)
    teacher_root_pos = np.asarray(teacher_root_pos, np.float64)
    teacher_root_quat = np.asarray(teacher_root_quat, np.float64)
    if (
        teacher_q.shape != (31,)
        or teacher_root_pos.shape != (3,)
        or teacher_root_quat.shape != (4,)
        or not np.all(np.isfinite(teacher_q))
        or not np.all(np.isfinite(teacher_root_pos))
        or not np.all(np.isfinite(teacher_root_quat))
    ):
        raise DynamicReadyMaterializationError(
            "measured direct-frame0 physical birth is malformed"
        )
    quaternion_norm = float(np.linalg.norm(teacher_root_quat))
    if (
        not math.isfinite(quaternion_norm)
        or abs(quaternion_norm - 1.0) > 2.0e-6
    ):
        raise DynamicReadyMaterializationError(
            "measured direct-frame0 root quaternion is not normalized"
        )
    provenance = {
        "semantics": MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS,
        "teacher_root_exactly_preserved": True,
        "teacher_all_joints_exactly_preserved": True,
        "physical_minus_teacher_joint_pos_rad": [0.0] * 31,
        "physical_minus_teacher_root_pos_m": [0.0, 0.0, 0.0],
        "physical_root_quat_wxyz": teacher_root_quat.tolist(),
        "teacher_root_quat_wxyz": teacher_root_quat.tolist(),
        "teacher_and_physical_birth_differ": False,
        "historical_physical_birth_seed_consumed": False,
        "required_live_table_gate": "isaac_action_ball_nominal_hold_v1",
    }
    return (
        teacher_q.copy(),
        teacher_root_pos.copy(),
        teacher_root_quat.copy(),
        provenance,
    )


def _exact_racket_site_pose(
    backend: grounded.MujocoGroundedReadyBackend,
    state: grounded.ReadyState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copied exact-MJCF racket-site position and rotation."""

    try:
        site_id = backend._name_id(
            backend._mujoco.mjtObj.mjOBJ_SITE,
            MEASURED_PROJECTED_FRAME0_RACKET_SITE,
        )
        backend._install(state)
        position = np.asarray(backend._data.site_xpos[site_id], np.float64).copy()
        rotation = np.asarray(
            backend._data.site_xmat[site_id], np.float64
        ).reshape(3, 3).copy()
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"cannot audit projected-frame0 racket site: {exc}"
        ) from exc
    if (
        position.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(rotation))
    ):
        raise DynamicReadyMaterializationError(
            "projected-frame0 racket site FK is malformed"
        )
    return position, rotation


def _whole_body_table_geometry(
    runtime_contract: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Resolve the bound ActionBall table prism without inventing dimensions."""

    try:
        objective = runtime_contract["action_ball_training"]["runtime"][
            "counter_rally"
        ]["objective_profile"]
        near_x = float(objective["table_near_x_env_m"])
        half_width = float(objective["table_half_width_m"])
        surface_z = float(objective["table_surface_z_env_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicReadyMaterializationError(
            "whole-body safe-ready requires the runtime-bound table geometry"
        ) from exc
    if not all(math.isfinite(value) for value in (near_x, half_width, surface_z)) or (
        half_width <= 0.0
    ):
        raise DynamicReadyMaterializationError(
            "runtime-bound table geometry is malformed"
        )
    return near_x, half_width, surface_z


def _independent_measured_racket_frame0_reference(
    value: Mapping[str, Any], *, expected_motion_sha256: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the official-site pose target from the independent v4 channel."""

    if not isinstance(value, Mapping):
        raise DynamicReadyMaterializationError(
            "whole-body safe-ready requires independent measured racket frame0"
        )
    raw_mount_sign = value.get("robot_mount_normal_sign")
    if (
        isinstance(raw_mount_sign, bool)
        or type(raw_mount_sign) not in (int, float)
        or not math.isfinite(float(raw_mount_sign))
        or float(raw_mount_sign) != float(int(raw_mount_sign))
    ):
        raise DynamicReadyMaterializationError(
            "independent measured racket mount sign must be one exact integer"
        )
    try:
        position = np.asarray(value["site_pos_w_m"], np.float64)
        signed_face = np.asarray(value["signed_face_normal_w"], np.float64)
        long_axis = np.asarray(value["long_axis_w"], np.float64)
        axis_local = np.asarray(
            value["robot_butt_to_blade_axis_local"], np.float64
        )
        mount_sign = int(raw_mount_sign)
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicReadyMaterializationError(
            "independent measured racket frame0 is malformed"
        ) from exc
    if (
        value.get("authority")
        != "independent_schema_v4_measured_racket_channel"
        or value.get("motion_sha256") != expected_motion_sha256
        or value.get("frame_index") != 0
        or value.get("position_semantics")
        != EXPECTED_MEASURED_RACKET_POSITION_SEMANTICS
        or value.get("normal_semantics")
        != EXPECTED_MEASURED_RACKET_NORMAL_SEMANTICS
        or value.get("long_axis_semantics")
        != EXPECTED_MEASURED_RACKET_LONG_AXIS_SEMANTICS
        or value.get("robot_rigid_visual_mesh_sha256")
        != EXPECTED_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        or position.shape != (3,)
        or signed_face.shape != (3,)
        or long_axis.shape != (3,)
        or axis_local.shape != (3,)
        or mount_sign not in (-1, 1)
        or not np.array_equal(
            axis_local,
            np.asarray(
                EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
                np.float64,
            ),
        )
        or not (
            np.isfinite(position).all()
            and np.isfinite(signed_face).all()
            and np.isfinite(long_axis).all()
        )
    ):
        raise DynamicReadyMaterializationError(
            "independent measured racket frame0 lost its schema-v4 authority"
        )
    face_norm = float(np.linalg.norm(signed_face))
    long_norm = float(np.linalg.norm(long_axis))
    if (
        abs(face_norm - 1.0) > 1.0e-3
        or abs(long_norm - 1.0) > 1.0e-3
        or abs(float(signed_face @ long_axis)) > 1.0e-3
    ):
        raise DynamicReadyMaterializationError(
            "independent measured racket frame0 axes are not orthonormal"
        )
    signed_face = signed_face / face_norm
    long_axis = long_axis / long_norm
    site_y = float(mount_sign) * signed_face
    site_y = site_y - float(site_y @ long_axis) * long_axis
    site_y_norm = float(np.linalg.norm(site_y))
    if site_y_norm <= 1.0e-12:
        raise DynamicReadyMaterializationError(
            "independent measured racket frame0 site axes are degenerate"
        )
    site_y = site_y / site_y_norm
    local_y = np.asarray([0.0, 1.0, 0.0], np.float64)
    local_third = np.cross(axis_local, local_y)
    world_third = np.cross(long_axis, site_y)
    local_basis = np.column_stack((axis_local, local_y, local_third))
    world_basis = np.column_stack((long_axis, site_y, world_third))
    rotation = world_basis @ local_basis.T
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 2.0e-6
        or np.linalg.det(rotation) < 1.0 - 2.0e-6
    ):
        raise DynamicReadyMaterializationError(
            "independent measured racket frame0 does not define a proper site rotation"
        )
    return position, rotation, {
        **dict(value),
        "signed_face_normal_w_unit": signed_face.tolist(),
        "long_axis_w_unit": long_axis.tolist(),
        "official_site_rotation_w": rotation.tolist(),
    }


def _conservative_robot_table_clearance_m(
    backend: grounded.MujocoGroundedReadyBackend,
    state: grounded.ReadyState,
    *,
    table_near_x_m: float,
    table_half_width_m: float,
    table_surface_z_m: float,
) -> float:
    """Conservative collision-sphere distance from the near-side table prism.

    The table occupies ``x>=near_x``, ``abs(y)<=half_width`` and
    ``z<=surface_z``.  This intentionally over-approximates the finite table:
    a positive result proves separation from top, edges and underside without
    relying on the vendor robot-only MJCF to contain an Isaac table body.
    """

    backend._install(state)
    model = backend.model
    data = backend._data
    clearances: list[float] = []
    for geom in range(int(model.ngeom)):
        if geom == backend._floor_geom or int(model.geom_bodyid[geom]) == 0:
            continue
        if int(model.geom_contype[geom]) == 0:
            continue
        center = np.asarray(data.geom_xpos[geom], np.float64)
        radius = float(model.geom_rbound[geom])
        if not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius < 0.0:
            raise DynamicReadyMaterializationError(
                "exact MJCF returned malformed robot collision bounds"
            )
        clearances.append(
            max(
                table_near_x_m - (float(center[0]) + radius),
                abs(float(center[1])) - (table_half_width_m + radius),
                (float(center[2]) - radius) - table_surface_z_m,
            )
        )
    if not clearances:
        raise DynamicReadyMaterializationError(
            "exact MJCF has no collidable robot geometry for the table gate"
        )
    return float(min(clearances))


def _whole_body_collision_pair_authority(
    backend: grounded.MujocoGroundedReadyBackend,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], dict[str, Any]]:
    """Freeze every enabled robot self pair and unsupported floor pair."""

    model = backend.model
    floor_geom = int(backend._floor_geom)
    robot_geoms = tuple(
        geom
        for geom in range(int(model.ngeom))
        if int(model.geom_bodyid[geom]) != 0
    )
    foot_geoms = frozenset(
        int(geom)
        for group in backend._foot_geom_sets
        for geom in group
    )
    self_pairs = tuple(
        (left, right)
        for offset, left in enumerate(robot_geoms)
        for right in robot_geoms[offset + 1 :]
        if self_collision_audit.geom_pair_enabled(model, left, right)
    )
    unsupported_floor_geoms = tuple(
        geom
        for geom in robot_geoms
        if geom not in foot_geoms
        and self_collision_audit.geom_pair_enabled(model, floor_geom, geom)
    )
    authority_rows = {
        "self_collision_geom_id_pairs": [list(pair) for pair in self_pairs],
        "unsupported_floor_robot_geom_ids": list(unsupported_floor_geoms),
        "expected_foot_floor_geom_ids": sorted(foot_geoms),
        "floor_geom_id": floor_geom,
    }
    authority_sha = hashlib.sha256(
        _canonical_json_bytes(authority_rows)
    ).hexdigest()
    return self_pairs, unsupported_floor_geoms, {
        **authority_rows,
        "pair_authority_sha256": authority_sha,
        "enabled_self_pair_count": len(self_pairs),
        "unsupported_floor_pair_count": len(unsupported_floor_geoms),
        "required_clearance_m": WHOLE_BODY_REQUIRED_COLLISION_CLEARANCE_M,
        "capped_clearance_m": WHOLE_BODY_COLLISION_CLEARANCE_CAP_M,
        "bisection_tolerance_m": (
            WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M
        ),
        "distance_semantics": (
            "mujoco_geomDistance_saturation_bisection_with_robot_pair_"
            "sphere_lower_bound_pruning"
        ),
    }


def _whole_body_collision_clearance_m(
    backend: grounded.MujocoGroundedReadyBackend,
    *,
    self_pairs: Sequence[tuple[int, int]],
    unsupported_floor_geoms: Sequence[int],
) -> tuple[float, float]:
    """Return a conservative capped signed clearance over the frozen pairs."""

    model = backend.model
    data = backend._data
    best = WHOLE_BODY_COLLISION_CLEARANCE_CAP_M
    raw_best = WHOLE_BODY_COLLISION_CLEARANCE_CAP_M
    floor_geom = int(backend._floor_geom)
    for geom in unsupported_floor_geoms:
        distance, saturated = self_collision_audit.geom_clearance(
            model,
            data,
            floor_geom,
            int(geom),
            distmax=best,
            tol=WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M,
        )
        raw_best = min(raw_best, float(distance))
        if distance < 0.0:
            return float(distance), raw_best
        conservative = (
            float(distance)
            if saturated
            else float(distance)
            - WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M
        )
        best = min(best, conservative)
    centers = np.asarray(data.geom_xpos, np.float64)
    radii = np.asarray(model.geom_rbound, np.float64)
    if (
        centers.shape != (int(model.ngeom), 3)
        or radii.shape != (int(model.ngeom),)
        or not np.isfinite(centers).all()
        or not np.isfinite(radii).all()
        or np.any(radii < 0.0)
    ):
        raise DynamicReadyMaterializationError(
            "exact MJCF returned malformed collision bounding spheres"
        )
    for left, right in self_pairs:
        sphere_lower_bound = float(
            np.linalg.norm(centers[left] - centers[right])
            - radii[left]
            - radii[right]
        )
        if sphere_lower_bound >= best:
            continue
        distance, saturated = self_collision_audit.geom_clearance(
            model,
            data,
            int(left),
            int(right),
            distmax=best,
            tol=WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M,
        )
        raw_best = min(raw_best, float(distance))
        if distance < 0.0:
            return float(distance), raw_best
        conservative = (
            float(distance)
            if saturated
            else float(distance)
            - WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M
        )
        best = min(best, conservative)
    if not math.isfinite(best):
        raise DynamicReadyMaterializationError(
            "exact MJCF collision clearance is non-finite"
        )
    return float(best), float(raw_best)


def _build_whole_body_safety_evaluator(
    *,
    backend: grounded.MujocoGroundedReadyBackend,
    identity: grounded.ExactModelIdentity,
    plant: Mapping[str, Any],
    hard_inner_lower: np.ndarray,
    hard_inner_upper: np.ndarray,
    runtime_contract: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Build one evaluator that reuses the exact static contact LP."""

    table_near_x, table_half_width, table_surface_z = (
        _whole_body_table_geometry(runtime_contract)
    )
    (
        collision_self_pairs,
        collision_unsupported_floor_geoms,
        collision_pair_authority,
    ) = _whole_body_collision_pair_authority(backend)
    qdes_limits = np.asarray(plant["qdes_limits"], np.float64)
    inset = float(plant["projection_inset"])
    span = qdes_limits[:, 1] - qdes_limits[:, 0]
    executed_lower = np.maximum(
        qdes_limits[:, 0] + inset * span,
        np.asarray(hard_inner_lower, np.float64),
    )
    executed_upper = np.minimum(
        qdes_limits[:, 1] - inset * span,
        np.asarray(hard_inner_upper, np.float64),
    )
    if np.any(executed_lower >= executed_upper):
        raise DynamicReadyMaterializationError(
            "whole-body evaluator has an empty executed qdes envelope"
        )
    contact_config = torque_topp.GroundContactConfig(
        expected_model_binding=identity.ground_model_binding_sha256,
        model_source_path=str(Path(identity.mjcf_path).expanduser().resolve()),
        expected_source_sha256=identity.mjcf_sha256,
        minimum_normal_force_per_contact_n=(
            WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
        ),
        minimum_normal_force_per_foot_n=(
            WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_FOOT_N
        ),
    )
    solver = torque_topp.MujocoGroundContactLPSolver(
        backend.model, contact_config
    )
    actuator_contract = torque_topp.direct_actuator_contract_from_mujoco(
        backend.model,
        support_mode="ground",
        contact_mode="double_support_floor",
        fixed_lp_solver="scipy.optimize.linprog:highs",
    )
    model_lower, model_upper, actuated, actuator_report = (
        torque_topp._resolve_grounded_actuator_limits(
            actuator_contract, int(backend.model.nv)
        )
    )
    model_row_for_runtime = (
        np.asarray(backend._binding.joint_dof_adrs, np.int64) - 6
    )
    if not np.array_equal(np.sort(model_row_for_runtime), np.arange(31)):
        raise DynamicReadyMaterializationError(
            "whole-body evaluator lost the runtime-to-model actuator permutation"
        )
    exact_lower = np.asarray(backend.position_lower, np.float64)
    exact_upper = np.asarray(backend.position_upper, np.float64)
    kp = np.asarray(plant["kp"], np.float64)
    effort = np.asarray(plant["effort"], np.float64)

    def evaluator(state: grounded.ReadyState) -> whole_body_ready.SafetyEvaluation:
        scene = backend.static_scene(
            state,
            contact_gap_tolerance_m=2.0e-3,
            penetration_tolerance_m=2.0e-3,
        )
        collision_clearance, raw_collision_clearance = (
            _whole_body_collision_clearance_m(
            backend,
            self_pairs=collision_self_pairs,
            unsupported_floor_geoms=collision_unsupported_floor_geoms,
            )
        )
        contact_list_collision = bool(
            scene.unsupported_contacts or scene.self_collision_pairs
        )
        contact_clearance_disagreed = bool(
            contact_list_collision != (collision_clearance <= 0.0)
        )
        if contact_list_collision:
            # ``static_scene`` deliberately uses a 2 mm contact tolerance,
            # while the signed-distance bisection reports geometric zero.  The
            # two booleans are therefore not equivalent objects.  Treat either
            # signal conservatively instead of aborting the search on a
            # same-writer equality check.
            collision_clearance = min(collision_clearance, 0.0)
        hull, _com_xy, global_support_margin = grounded._support_margin(scene)
        runtime_tau_lower = np.maximum(
            -effort, kp * (executed_lower - state.joint_pos)
        )
        runtime_tau_upper = np.minimum(
            effort, kp * (executed_upper - state.joint_pos)
        )
        runtime_tau_lower_model = np.empty(31, np.float64)
        runtime_tau_upper_model = np.empty(31, np.float64)
        runtime_tau_lower_model[model_row_for_runtime] = runtime_tau_lower
        runtime_tau_upper_model[model_row_for_runtime] = runtime_tau_upper
        hold_lower = np.maximum(model_lower, runtime_tau_lower_model)
        hold_upper = np.minimum(model_upper, runtime_tau_upper_model)
        solution = None
        lp_error = None
        if np.all(hold_lower < 0.0) and np.all(hold_upper > 0.0):
            try:
                solution = solver.solve(
                    backend._qpos(state),
                    np.zeros(int(backend.model.nv), np.float64),
                    np.zeros(int(backend.model.nv), np.float64),
                    actuated,
                    hold_lower,
                    hold_upper,
                    np.full(int(backend.model.nv), 1.0e6, np.float64),
                    path_tangent=np.zeros(int(backend.model.nv), np.float64),
                    lp_objective=LP_OBJECTIVE,
                )
            except Exception as exc:
                lp_error = f"{type(exc).__name__}: {exc}"
        feasible = bool(solution is not None and solution.feasible)
        tau_model = (
            np.asarray(solution.actuator_generalized_force, np.float64)
            if feasible
            else np.zeros(31, np.float64)
        )
        tau_runtime = tau_model[model_row_for_runtime]
        qdes = state.joint_pos + tau_runtime / kp
        report = dict(solution.report) if feasible else {}
        per_foot_normal = np.asarray(
            report.get("normal_force_per_foot_n", [-1.0, -1.0]), np.float64
        )
        cop_margin = np.asarray(
            report.get("cop_interior_margin_per_foot_m", [-1.0, -1.0]),
            np.float64,
        )
        normal_per_contact = np.asarray(
            report.get("normal_force_per_contact_n", ()), np.float64
        )
        feet_report = (
            report.get("contact_geometry", {}).get("feet", ())
            if isinstance(report.get("contact_geometry"), Mapping)
            else ()
        )
        minimum_normal_per_foot = np.full(2, -1.0, np.float64)
        if feasible and isinstance(feet_report, list) and len(feet_report) == 2:
            for foot, row in enumerate(feet_report):
                support_range = (
                    row.get("support_point_range")
                    if isinstance(row, Mapping)
                    else None
                )
                if (
                    isinstance(support_range, list)
                    and len(support_range) == 2
                    and all(type(value) is int for value in support_range)
                    and 0 <= support_range[0] < support_range[1]
                    <= len(normal_per_contact)
                ):
                    minimum_normal_per_foot[foot] = float(
                        np.min(
                            normal_per_contact[
                                support_range[0] : support_range[1]
                            ]
                        )
                    )
        qdes_slack = (
            float(np.min(np.minimum(qdes - executed_lower, executed_upper - qdes)))
            if feasible
            else -1.0
        )
        torque_slack = (
            float(np.min(np.minimum(tau_model - hold_lower, hold_upper - tau_model)))
            if feasible
            else -1.0
        )
        equality_residual = float(
            solution.equality_residual if feasible else 1.0
        )
        root_residual = float(solution.root_residual if feasible else 1.0)
        residual_tolerance = float(contact_config.equality_residual_tolerance)
        support_slack = -1.0
        if global_support_margin is not None and cop_margin.shape == (2,):
            support_slack = min(
                float(global_support_margin) - 5.0e-4,
                float(np.min(cop_margin)) - 5.0e-4,
            )
        root_rotation = whole_body_ready._quat_to_rotation(
            state.root_quat_wxyz
        )
        root_tilt = math.acos(
            float(np.clip(root_rotation[2, 2], -1.0, 1.0))
        )
        table_clearance = _conservative_robot_table_clearance_m(
            backend,
            state,
            table_near_x_m=table_near_x,
            table_half_width_m=table_half_width,
            table_surface_z_m=table_surface_z,
        )
        racket_position, racket_rotation = _exact_racket_site_pose(
            backend, state
        )
        slacks = {
            "left_sole_floor_slack_m": 2.0e-3
            - abs(float(scene.sole_minimum_distance_m[0])),
            "right_sole_floor_slack_m": 2.0e-3
            - abs(float(scene.sole_minimum_distance_m[1])),
            "left_contact_load_slack_n": (
                min(
                    float(minimum_normal_per_foot[0])
                    - WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N,
                    float(per_foot_normal[0])
                    - float(contact_config.minimum_normal_force_per_foot_n),
                )
                if minimum_normal_per_foot.shape == (2,)
                and per_foot_normal.shape == (2,)
                else -1.0
            ),
            "right_contact_load_slack_n": (
                min(
                    float(minimum_normal_per_foot[1])
                    - WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N,
                    float(per_foot_normal[1])
                    - float(contact_config.minimum_normal_force_per_foot_n),
                )
                if minimum_normal_per_foot.shape == (2,)
                and per_foot_normal.shape == (2,)
                else -1.0
            ),
            "support_margin_slack_m": support_slack,
            "joint_position_slack_rad": float(
                np.min(
                    np.minimum(
                        state.joint_pos - exact_lower,
                        exact_upper - state.joint_pos,
                    )
                )
            ),
            "qdes_slack_rad": qdes_slack,
            "torque_slack_nm": torque_slack,
            "table_clearance_slack_m": table_clearance - 1.0e-2,
            "root_height_slack_m": float(state.root_pos_w[2]) - 0.5,
            "root_tilt_slack_rad": 0.7 - root_tilt,
            "collision_slack_m": (
                collision_clearance
                - WHOLE_BODY_REQUIRED_COLLISION_CLEARANCE_M
            ),
            "ground_lp_residual_slack": residual_tolerance
            - max(equality_residual, root_residual),
        }
        return whole_body_ready.SafetyEvaluation(
            slacks=slacks,
            racket_position_w=racket_position,
            racket_rotation_w=racket_rotation,
            evidence={
                "exact_contact_lp_reused": True,
                "lp_feasible": feasible,
                "lp_error": lp_error,
                "lp_objective": LP_OBJECTIVE,
                "equality_residual": equality_residual,
                "root_residual": root_residual,
                "normal_force_per_foot_n": per_foot_normal.tolist(),
                "normal_force_per_contact_n": normal_per_contact.tolist(),
                "minimum_normal_force_per_contact_per_foot_n": (
                    minimum_normal_per_foot.tolist()
                ),
                "required_minimum_normal_force_per_contact_n": (
                    WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
                ),
                "required_minimum_normal_force_per_foot_n": float(
                    contact_config.minimum_normal_force_per_foot_n
                ),
                "cop_interior_margin_per_foot_m": cop_margin.tolist(),
                "global_support_margin_m": global_support_margin,
                "support_hull_floor_xy_m": hull.tolist(),
                "hold_qdes_joint_pos_rad": qdes.tolist(),
                "actuator_generalized_force_runtime_order_nm": tau_runtime.tolist(),
                "actuator_generalized_force_mujoco_row_order_nm": tau_model.tolist(),
                "mujoco_row_for_runtime_joint": model_row_for_runtime.tolist(),
                "mujoco_actuated_dof_indices": np.asarray(
                    actuated, np.int64
                ).tolist(),
                "executed_qdes_lower_rad": executed_lower.tolist(),
                "executed_qdes_upper_rad": executed_upper.tolist(),
                "model_tau_lower_mujoco_row_order_nm": model_lower.tolist(),
                "model_tau_upper_mujoco_row_order_nm": model_upper.tolist(),
                "runtime_tau_lower_runtime_order_nm": runtime_tau_lower.tolist(),
                "runtime_tau_upper_runtime_order_nm": runtime_tau_upper.tolist(),
                "runtime_tau_lower_mujoco_row_order_nm": (
                    runtime_tau_lower_model.tolist()
                ),
                "runtime_tau_upper_mujoco_row_order_nm": (
                    runtime_tau_upper_model.tolist()
                ),
                "effective_tau_lower_mujoco_row_order_nm": hold_lower.tolist(),
                "effective_tau_upper_mujoco_row_order_nm": hold_upper.tolist(),
                "actuator_limit_contract": actuator_report,
                "solver_report": report,
                "exact_state_lp_cache_hit": report.get(
                    "exact_state_lp_cache_hit"
                ),
                "evaluated_state_sha256": grounded.state_digest(state),
                "evaluated_joint_pos_rad": state.joint_pos.tolist(),
                "evaluated_root_pos_w_m": state.root_pos_w.tolist(),
                "evaluated_root_quat_wxyz": state.root_quat_wxyz.tolist(),
                "sole_minimum_distance_m": np.asarray(
                    scene.sole_minimum_distance_m, np.float64
                ).tolist(),
                "exact_joint_position_lower_rad": exact_lower.tolist(),
                "exact_joint_position_upper_rad": exact_upper.tolist(),
                "conservative_table_clearance_m": table_clearance,
                "table_geometry": {
                    "near_x_m": table_near_x,
                    "half_width_m": table_half_width,
                    "surface_z_m": table_surface_z,
                    "required_clearance_m": 1.0e-2,
                    "semantics": "collision_sphere_separation_from_overapproximated_near_side_table_prism",
                },
                "root_limits": {
                    "minimum_height_m": 0.5,
                    "maximum_tilt_rad": 0.7,
                },
                "collision_clearance": {
                    **collision_pair_authority,
                    "realized_capped_minimum_clearance_m": (
                        collision_clearance
                    ),
                    "raw_bisection_midpoint_or_saturated_cap_m": (
                        raw_collision_clearance
                    ),
                    "positive_unsaturated_conservative_deduction_m": (
                        WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M
                    ),
                    "realized_slack_m": (
                        collision_clearance
                        - WHOLE_BODY_REQUIRED_COLLISION_CLEARANCE_M
                    ),
                    "contact_list_signed_distance_disagreed": (
                        contact_clearance_disagreed
                    ),
                    "unsupported_contacts": [
                        dict(row) for row in scene.unsupported_contacts
                    ],
                    "self_collision_pairs": [
                        dict(row) for row in scene.self_collision_pairs
                    ],
                },
            },
        )

    return evaluator, {
        "exact_joint_position_lower_rad": exact_lower.tolist(),
        "exact_joint_position_upper_rad": exact_upper.tolist(),
        "executed_qdes_lower_rad": executed_lower.tolist(),
        "executed_qdes_upper_rad": executed_upper.tolist(),
        "table_near_x_m": table_near_x,
        "table_half_width_m": table_half_width,
        "table_surface_z_m": table_surface_z,
        "minimum_table_clearance_m": 1.0e-2,
        "minimum_root_height_m": 0.5,
        "maximum_root_tilt_rad": 0.7,
        "collision_pair_authority": collision_pair_authority,
    }


def _consume_whole_body_selected_hold_witness(
    *,
    static_birth_evidence: Mapping[str, Any],
    ready_q: np.ndarray,
    ready_root_pos: np.ndarray,
    ready_root_quat: np.ndarray,
    identity: grounded.ExactModelIdentity,
    kp: np.ndarray,
    model_row_for_runtime: np.ndarray,
    actuated: np.ndarray,
    expected_vectors: Mapping[str, np.ndarray],
) -> torque_topp.GroundContactLPSolution:
    """Validate and select the one fresh whole-body final-state LP witness.

    The whole-body optimizer uses many LP samples.  Its winner is re-audited
    with a newly loaded backend and a newly constructed solver.  This function
    binds that exact cache-miss result to the state and envelopes which will be
    written into the artifact; it deliberately never solves another LP.
    """

    if not isinstance(static_birth_evidence, Mapping):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold has no static evidence mapping"
        )
    witness = static_birth_evidence.get("evaluator_evidence")
    if not isinstance(witness, Mapping):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold has no final evaluator witness"
        )
    report = witness.get("solver_report")
    if not isinstance(report, Mapping):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold has no exact LP solver report"
        )
    if (
        static_birth_evidence.get("selected_hold_witness_authority")
        != "new_backend_new_solver_final_state_cache_miss"
        or witness.get("lp_feasible") is not True
        or witness.get("exact_state_lp_cache_hit") is not False
        or report.get("exact_state_lp_cache_hit") is not False
        or report.get("model_binding")
        != identity.ground_model_binding_sha256
        or witness.get("required_minimum_normal_force_per_contact_n")
        != WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
        or witness.get("required_minimum_normal_force_per_foot_n")
        != WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_FOOT_N
    ):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold witness lost its fresh 0.1N LP authority"
        )

    state_vectors = {
        "evaluated_joint_pos_rad": np.asarray(ready_q, np.float64),
        "evaluated_root_pos_w_m": np.asarray(ready_root_pos, np.float64),
        "evaluated_root_quat_wxyz": np.asarray(ready_root_quat, np.float64),
    }
    for name, expected in state_vectors.items():
        actual = np.asarray(witness.get(name, ()), np.float64)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise DynamicReadyMaterializationError(
                "whole-body selected hold witness state differs from physical_ready: "
                f"{name}"
            )
    witness_state = grounded.ReadyState(
        state_vectors["evaluated_joint_pos_rad"],
        state_vectors["evaluated_root_pos_w_m"],
        state_vectors["evaluated_root_quat_wxyz"],
    )
    if witness.get("evaluated_state_sha256") != grounded.state_digest(
        witness_state
    ):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold witness state digest is invalid"
        )

    required_vector_names = set(expected_vectors)
    if required_vector_names != {
        "executed_qdes_lower_rad",
        "executed_qdes_upper_rad",
        "model_tau_lower_mujoco_row_order_nm",
        "model_tau_upper_mujoco_row_order_nm",
        "runtime_tau_lower_runtime_order_nm",
        "runtime_tau_upper_runtime_order_nm",
        "runtime_tau_lower_mujoco_row_order_nm",
        "runtime_tau_upper_mujoco_row_order_nm",
        "effective_tau_lower_mujoco_row_order_nm",
        "effective_tau_upper_mujoco_row_order_nm",
    }:
        raise DynamicReadyMaterializationError(
            "internal whole-body selected hold vector schema is incomplete"
        )
    for name, expected_value in expected_vectors.items():
        expected = np.asarray(expected_value)
        actual = np.asarray(witness.get(name, ()), expected.dtype)
        if (
            actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.array_equal(actual, expected)
        ):
            raise DynamicReadyMaterializationError(
                "whole-body selected hold witness differs from current plant: "
                f"{name}"
            )
    witness_rows = np.asarray(
        witness.get("mujoco_row_for_runtime_joint", ()), np.int64
    )
    witness_actuated = np.asarray(
        witness.get("mujoco_actuated_dof_indices", ()), np.int64
    )
    if not np.array_equal(witness_rows, model_row_for_runtime) or not np.array_equal(
        witness_actuated, np.asarray(actuated, np.int64)
    ):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold witness actuator permutation changed"
        )

    tau_model = np.asarray(
        witness.get("actuator_generalized_force_mujoco_row_order_nm", ()),
        np.float64,
    )
    tau_runtime = np.asarray(
        witness.get("actuator_generalized_force_runtime_order_nm", ()),
        np.float64,
    )
    qdes = np.asarray(witness.get("hold_qdes_joint_pos_rad", ()), np.float64)
    if (
        tau_model.shape != (31,)
        or tau_runtime.shape != (31,)
        or qdes.shape != (31,)
        or not (
            np.isfinite(tau_model).all()
            and np.isfinite(tau_runtime).all()
            and np.isfinite(qdes).all()
        )
        or not np.array_equal(tau_runtime, tau_model[model_row_for_runtime])
        or not np.array_equal(
            qdes,
            np.asarray(ready_q, np.float64)
            + tau_runtime / np.asarray(kp, np.float64),
        )
    ):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold witness qdes/torque is internally inconsistent"
        )

    contact_normals = np.asarray(
        witness.get("normal_force_per_contact_n", ()), np.float64
    )
    report_normals = np.asarray(
        report.get("normal_force_per_contact_n", ()), np.float64
    )
    cop_margins = np.asarray(
        witness.get("cop_interior_margin_per_foot_m", ()), np.float64
    )
    report_cop_margins = np.asarray(
        report.get("cop_interior_margin_per_foot_m", ()), np.float64
    )
    per_foot_normals = np.asarray(
        witness.get("normal_force_per_foot_n", ()), np.float64
    )
    report_per_foot_normals = np.asarray(
        report.get("normal_force_per_foot_n", ()), np.float64
    )
    if (
        contact_normals.ndim != 1
        or contact_normals.size < 6
        or not np.isfinite(contact_normals).all()
        or np.any(
            contact_normals
            < WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N - 1.0e-7
        )
        or not np.array_equal(contact_normals, report_normals)
        or cop_margins.shape != (2,)
        or not np.isfinite(cop_margins).all()
        or np.any(cop_margins <= 0.0)
        or not np.array_equal(cop_margins, report_cop_margins)
        or per_foot_normals.shape != (2,)
        or not np.isfinite(per_foot_normals).all()
        or np.any(
            per_foot_normals
            < WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_FOOT_N - 1.0e-7
        )
        or not np.array_equal(per_foot_normals, report_per_foot_normals)
    ):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold witness lacks its required contact/CoP margin"
        )

    equality_residual = float(witness.get("equality_residual", math.inf))
    root_residual = float(witness.get("root_residual", math.inf))
    if not math.isfinite(equality_residual) or not math.isfinite(root_residual):
        raise DynamicReadyMaterializationError(
            "whole-body selected hold witness has non-finite LP residuals"
        )
    return torque_topp.GroundContactLPSolution(
        feasible=True,
        actuator_generalized_force=tau_model.copy(),
        point_force_floor=np.empty((0, 3), np.float64),
        equality_residual=equality_residual,
        root_residual=root_residual,
        report=dict(report),
    )


def _compose_measured_projected_frame0_physical_birth(
    *,
    teacher_q: np.ndarray,
    teacher_root_pos: np.ndarray,
    teacher_root_quat: np.ndarray,
    backend: grounded.MujocoGroundedReadyBackend,
    identity: grounded.ExactModelIdentity,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Ground measured frame 0 by changing leg12 and nothing upstream.

    The stored teacher quaternion remains byte-value exact.  A normalized copy
    is used only for the numerical MuJoCo backend, matching the direct-frame0
    branch.  Full projection and FK fidelity evidence is returned for sealing
    into the dynamic-ready candidate.
    """

    teacher_q = np.asarray(teacher_q, np.float64)
    teacher_root_pos = np.asarray(teacher_root_pos, np.float64)
    teacher_root_quat = np.asarray(teacher_root_quat, np.float64)
    if (
        teacher_q.shape != (31,)
        or teacher_root_pos.shape != (3,)
        or teacher_root_quat.shape != (4,)
        or not np.all(np.isfinite(teacher_q))
        or not np.all(np.isfinite(teacher_root_pos))
        or not np.all(np.isfinite(teacher_root_quat))
    ):
        raise DynamicReadyMaterializationError(
            "measured projected-frame0 physical birth is malformed"
        )
    quaternion_norm = float(np.linalg.norm(teacher_root_quat))
    if (
        not math.isfinite(quaternion_norm)
        or abs(quaternion_norm - 1.0) > 2.0e-6
    ):
        raise DynamicReadyMaterializationError(
            "measured projected-frame0 root quaternion is not normalized"
        )
    normalized_quat = teacher_root_quat / quaternion_norm
    teacher_state = grounded.ReadyState(
        teacher_q,
        teacher_root_pos,
        normalized_quat,
    )
    teacher_racket_position, teacher_racket_rotation = _exact_racket_site_pose(
        backend,
        teacher_state,
    )
    try:
        projected = grounded.solve_g1_support_edge_projection(
            teacher_state,
            backend=backend,
            expected_model_identity=identity,
            config=grounded.GroundedReadyConfig(),
            projection_config=grounded.SupportEdgeProjectionConfig(
                required_support_margin_m=5.0e-4,
                correction_guard_m=2.5e-4,
                maximum_iterations=8,
                maximum_common_shift_m=3.0e-2,
            ),
        )
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"measured frame0 grounded projection failed: {exc}"
        ) from exc
    if (
        getattr(projected, "geometry_passed", None) is not True
        or getattr(projected, "ground_dynamics_passed", None) is not True
        or getattr(projected, "candidate_id", None) != "G1S"
    ):
        raise DynamicReadyMaterializationError(
            "measured frame0 projection did not return a complete G1S static pass"
        )
    projected_q = np.asarray(projected.state.joint_pos, np.float64)
    leg_indices = grounded._joint_indices(
        backend.joint_names,
        grounded.LEG_JOINT_NAMES,
    )
    nonleg_indices = grounded._joint_indices(
        backend.joint_names,
        grounded.UPPER_JOINT_NAMES,
    )
    changed_indices = np.flatnonzero(projected_q != teacher_q)
    if (
        not np.array_equal(projected.state.root_pos_w, teacher_root_pos)
        or not np.array_equal(projected.state.root_quat_wxyz, normalized_quat)
        or not np.array_equal(
            projected_q[nonleg_indices], teacher_q[nonleg_indices]
        )
        or not set(int(index) for index in changed_indices).issubset(
            set(int(index) for index in leg_indices)
        )
    ):
        raise DynamicReadyMaterializationError(
            "measured frame0 projection changed root or a non-leg joint"
        )
    projected_racket_position, projected_racket_rotation = (
        _exact_racket_site_pose(backend, projected.state)
    )
    if not np.array_equal(
        projected_racket_position, teacher_racket_position
    ) or not np.array_equal(projected_racket_rotation, teacher_racket_rotation):
        raise DynamicReadyMaterializationError(
            "measured frame0 projection changed exact racket-site FK"
        )
    joint_delta = projected_q - teacher_q
    racket_position_delta = projected_racket_position - teacher_racket_position
    racket_rotation_delta = projected_racket_rotation - teacher_racket_rotation
    receipt = grounded._jsonable(projected.receipt)
    try:
        realized_support_margin_m = float(
            receipt["static_geometry"]["support"]["margin_m"]
        )
        receipt_gates = dict(receipt["gates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicReadyMaterializationError(
            "measured frame0 projection receipt lost static provenance"
        ) from exc
    if (
        not math.isfinite(realized_support_margin_m)
        or realized_support_margin_m < 5.0e-4
        or receipt_gates.get("static_ground_dynamics") != "PASS"
        or receipt_gates.get("support_margin") != "PASS"
    ):
        raise DynamicReadyMaterializationError(
            "measured frame0 projection receipt lost support or LP readiness"
        )
    provenance = {
        "semantics": MEASURED_BIRTH_PROJECTED_FRAME0_SEMANTICS,
        "teacher_root_exactly_preserved": True,
        "teacher_nonleg_exactly_preserved": True,
        "teacher_all_joints_exactly_preserved": not bool(len(changed_indices)),
        "historical_physical_birth_seed_consumed": False,
        "leg_joint_indices": leg_indices.tolist(),
        "leg_joint_names": list(grounded.LEG_JOINT_NAMES),
        "nonleg_joint_indices": nonleg_indices.tolist(),
        "nonleg_joint_names": list(grounded.UPPER_JOINT_NAMES),
        "changed_joint_indices": changed_indices.tolist(),
        "changed_joint_names": [
            str(backend.joint_names[int(index)]) for index in changed_indices
        ],
        "physical_minus_teacher_joint_pos_rad": joint_delta.tolist(),
        "physical_minus_teacher_joint_l2_rad": float(
            np.linalg.norm(joint_delta)
        ),
        "physical_minus_teacher_joint_linf_rad": float(
            np.max(np.abs(joint_delta))
        ),
        "physical_minus_teacher_leg12_rms_rad": float(
            np.linalg.norm(joint_delta[leg_indices]) / math.sqrt(12.0)
        ),
        "physical_minus_teacher_root_pos_m": [0.0, 0.0, 0.0],
        "physical_root_quat_wxyz": teacher_root_quat.tolist(),
        "teacher_root_quat_wxyz": teacher_root_quat.tolist(),
        "teacher_and_physical_birth_differ": bool(len(changed_indices)),
        "racket_site_fidelity": {
            "site_name": MEASURED_PROJECTED_FRAME0_RACKET_SITE,
            "authority": "current_exact_mjcf_fk",
            "teacher_position_w_m": teacher_racket_position.tolist(),
            "physical_position_w_m": projected_racket_position.tolist(),
            "physical_minus_teacher_position_w_m": (
                racket_position_delta.tolist()
            ),
            "position_bitwise_equal": True,
            "position_error_m": float(np.linalg.norm(racket_position_delta)),
            "teacher_rotation_w": teacher_racket_rotation.tolist(),
            "physical_rotation_w": projected_racket_rotation.tolist(),
            "maximum_rotation_matrix_error": float(
                np.max(np.abs(racket_rotation_delta))
            ),
            "rotation_bitwise_equal": True,
        },
        "grounded_projection_candidate_id": projected.candidate_id,
        "grounded_projection_receipt_sha256": projected.receipt_sha256,
        "required_live_table_gate": "isaac_action_ball_nominal_hold_v1",
    }
    static_evidence = {
        "authority": "fresh_current_exact_mjcf_grounded_projection",
        "grounded_ready_receipt_sha256": projected.receipt_sha256,
        "grounded_ready_receipt": receipt,
        "geometry_passed": True,
        "ground_dynamics_passed": True,
        "gates": receipt_gates,
        "required_support_margin_m": 5.0e-4,
        "realized_support_margin_m": realized_support_margin_m,
    }
    return (
        projected_q.copy(),
        teacher_root_pos.copy(),
        teacher_root_quat.copy(),
        provenance,
        static_evidence,
    )


def _compose_measured_whole_body_safe_frame0_physical_birth(
    *,
    teacher_q: np.ndarray,
    teacher_root_pos: np.ndarray,
    teacher_root_quat: np.ndarray,
    backend: grounded.MujocoGroundedReadyBackend,
    identity: grounded.ExactModelIdentity,
    plant: Mapping[str, Any],
    hard_inner_lower: np.ndarray,
    hard_inner_upper: np.ndarray,
    runtime_contract: Mapping[str, Any],
    measured_racket_frame0: Mapping[str, Any],
    motion_sha256: str,
    optimizer_initial_states: Sequence[grounded.ReadyState] = (),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Find a safe whole-body ready, then pull it toward measured frame 0."""

    stored_teacher_q = np.asarray(teacher_q, np.float64).copy()
    stored_teacher_root_pos = np.asarray(teacher_root_pos, np.float64).copy()
    stored_teacher_root_quat = np.asarray(teacher_root_quat, np.float64).copy()
    quaternion_norm = float(np.linalg.norm(stored_teacher_root_quat))
    if (
        stored_teacher_q.shape != (31,)
        or stored_teacher_root_pos.shape != (3,)
        or stored_teacher_root_quat.shape != (4,)
        or not np.isfinite(stored_teacher_q).all()
        or not np.isfinite(stored_teacher_root_pos).all()
        or not np.isfinite(stored_teacher_root_quat).all()
        or not math.isfinite(quaternion_norm)
        or abs(quaternion_norm - 1.0) > 2.0e-6
    ):
        raise DynamicReadyMaterializationError(
            "measured whole-body frame0 input is malformed"
    )
    measured = grounded.ReadyState(
        stored_teacher_q,
        stored_teacher_root_pos,
        stored_teacher_root_quat / quaternion_norm,
    )
    stored_endpoint_state_sha256 = _stored_ready_state_sha256(
        stored_teacher_q,
        stored_teacher_root_pos,
        stored_teacher_root_quat,
    )
    mjcf_audit_state_sha256 = grounded.state_digest(measured)
    (
        racket_reference_position,
        racket_reference_rotation,
        racket_reference_contract,
    ) = _independent_measured_racket_frame0_reference(
        measured_racket_frame0,
        expected_motion_sha256=motion_sha256,
    )
    evaluator, evaluator_contract = _build_whole_body_safety_evaluator(
        backend=backend,
        identity=identity,
        plant=plant,
        hard_inner_lower=hard_inner_lower,
        hard_inner_upper=hard_inner_upper,
        runtime_contract=runtime_contract,
    )
    initial_states: list[grounded.ReadyState] = list(optimizer_initial_states)
    # The compiled vendor key is merely a deterministic optimizer start.  It
    # imports no historical hold/authorization claim and is fully re-evaluated.
    try:
        initial_states.append(backend.vendor_key_state(0))
    except Exception:
        pass
    search_config = whole_body_ready.WholeBodySearchConfig()
    try:
        result = whole_body_ready.solve_measured_conditioned_whole_body_safe_ready(
            measured,
            evaluator=evaluator,
            racket_reference_position_w=racket_reference_position,
            racket_reference_rotation_w=racket_reference_rotation,
            position_lower=np.asarray(backend.position_lower, np.float64),
            position_upper=np.asarray(backend.position_upper, np.float64),
            initial_states=tuple(initial_states),
            config=search_config,
        )
    except Exception as exc:
        report = getattr(exc, "report", None)
        suffix = "" if report is None else f"; report={report}"
        raise DynamicReadyMaterializationError(
            f"measured-conditioned whole-body safe-ready failed: {exc}{suffix}"
        ) from exc

    # Do not seal the optimizer closure's cached LP result.  Load a new exact
    # backend and build a new LP solver after the winner is known, then make
    # this cache-miss evaluation the sole physical/hold witness consumed by
    # the artifact below.
    try:
        fresh_backend = grounded.MujocoGroundedReadyBackend.load(identity)
        fresh_evaluator, fresh_evaluator_contract = (
            _build_whole_body_safety_evaluator(
                backend=fresh_backend,
                identity=identity,
                plant=plant,
                hard_inner_lower=hard_inner_lower,
                hard_inner_upper=hard_inner_upper,
                runtime_contract=runtime_contract,
            )
        )
        fresh_final = fresh_evaluator(result.state)
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            "fresh whole-body winner re-audit failed on a new exact backend: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if fresh_evaluator_contract != evaluator_contract:
        raise DynamicReadyMaterializationError(
            "fresh whole-body evaluator contract differs from the search evaluator"
        )
    fresh_normalized_slacks = {
        name: float(
            fresh_final.slacks[name]
            / whole_body_ready.DEFAULT_SLACK_SCALES[name]
        )
        for name in whole_body_ready.REQUIRED_SAFETY_SLACK_NAMES
    }
    required_final_gate = max(
        search_config.positive_gate_normalized_slack,
        float(result.stage1_locked_worst_normalized_slack),
    )
    exact_measured_frame0_selected = bool(
        result.optimizer_report.get("exact_measured_frame0_selected", False)
    )
    fresh_direct_robust_gate_passed = all(
        fresh_final.slacks[name]
        >= whole_body_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[name]
        for name in whole_body_ready.REQUIRED_SAFETY_SLACK_NAMES
    )
    if (
        not math.isfinite(required_final_gate)
        or any(
            value <= search_config.positive_gate_normalized_slack
            or value < required_final_gate
            for value in fresh_normalized_slacks.values()
        )
        or not fresh_direct_robust_gate_passed
    ):
        raise DynamicReadyMaterializationError(
            "fresh whole-body winner violates the original or stage-1 safety gate"
        )
    fresh_witness = dict(fresh_final.evidence)
    expected_state_sha = grounded.state_digest(result.state)
    if (
        fresh_witness.get("lp_feasible") is not True
        or fresh_witness.get("exact_state_lp_cache_hit") is not False
        or fresh_witness.get("evaluated_state_sha256") != expected_state_sha
        or fresh_witness.get("required_minimum_normal_force_per_contact_n")
        != WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
    ):
        raise DynamicReadyMaterializationError(
            "fresh whole-body winner lacks one cache-miss 0.1N exact LP witness"
        )
    if exact_measured_frame0_selected and (
        not np.array_equal(result.state.joint_pos, measured.joint_pos)
        or not np.array_equal(result.state.root_pos_w, measured.root_pos_w)
        or not np.array_equal(
            result.state.root_quat_wxyz, measured.root_quat_wxyz
        )
    ):
        raise DynamicReadyMaterializationError(
            "exact-frame0 short circuit changed the teacher endpoint"
        )
    if not exact_measured_frame0_selected and (
        np.array_equal(result.state.joint_pos, measured.joint_pos)
        and np.array_equal(result.state.root_pos_w, measured.root_pos_w)
        and np.array_equal(result.state.root_quat_wxyz, measured.root_quat_wxyz)
    ):
        raise DynamicReadyMaterializationError(
            "learned-bridge fallback is bitwise equal to teacher frame0"
        )
    if exact_measured_frame0_selected:
        selected_q = stored_teacher_q
        selected_root_pos = stored_teacher_root_pos
        selected_root_quat = stored_teacher_root_quat
    else:
        selected_q = np.asarray(result.state.joint_pos, np.float64).copy()
        selected_root_pos = np.asarray(result.state.root_pos_w, np.float64).copy()
        selected_root_quat = np.asarray(
            result.state.root_quat_wxyz, np.float64
        ).copy()
    selected_quaternion_norm = float(np.linalg.norm(selected_root_quat))
    selected_stored_state_sha256 = _stored_ready_state_sha256(
        selected_q, selected_root_pos, selected_root_quat
    )
    transition_contract = {
        "schema_version": 1,
        "kind": (
            "exact_frame0_zero_duration_handoff_v1"
            if exact_measured_frame0_selected
            else "policy_learned_physical_birth_to_teacher_reference_v1"
        ),
        "selection_semantics": (
            "threshold_first_exact_frame0_direct"
            if exact_measured_frame0_selected
            else "deterministic_local_best_feasible_birth_then_policy_tracking"
        ),
        "state_sha256_semantics": (
            "float64_array_bytes_without_quaternion_normalization_v1"
        ),
        "physical_ready_state_sha256": selected_stored_state_sha256,
        "teacher_frame0_state_sha256": stored_endpoint_state_sha256,
        "mjcf_audit_state_sha256": grounded.state_digest(result.state),
        "stored_root_quaternion_norm": selected_quaternion_norm,
        "mjcf_audit_root_quat_wxyz": result.state.root_quat_wxyz.tolist(),
        "mjcf_audit_quaternion_semantics": (
            "stored_root_quat_unit_normalized_for_numerical_backend_only"
        ),
        "stored_teacher_and_physical_quaternion_unchanged": bool(
            exact_measured_frame0_selected
        ),
        "endpoints_bitwise_equal": bool(exact_measured_frame0_selected),
        "physical_ready_joint_velocity_exact_zero": True,
        "teacher_static_endpoint_joint_velocity_exact_zero": True,
        "measured_motion_velocity_channels_consumed": False,
        "not_a_motion_velocity_continuity_claim": True,
        "certified_transition_s": (
            0.0 if exact_measured_frame0_selected else None
        ),
        "required_min_wait_s": 0.0,
        "torque_speed_curve_required": False,
        "torque_speed_non_requirement_reason": (
            "identical_stored_configuration_and_constructed_zero_joint_velocity_endpoints"
            if exact_measured_frame0_selected
            else "no_scripted_transition_claim_policy_controls_the_bridge"
        ),
        "runtime_transition_reference_required": bool(
            not exact_measured_frame0_selected
        ),
        "required_followup_hold_gate": "isaac_action_ball_nominal_hold_v1",
        "required_followup_policy_steps": 60,
        "required_followup_physics_steps": 240,
        "diagnostic_unauthorized": True,
        "training_authorized": False,
    }
    changed_indices = [
        index for index, changed in enumerate(result.changed_joint_mask) if changed
    ]
    mount_sign = int(racket_reference_contract["robot_mount_normal_sign"])
    racket_axis_local = np.asarray(
        racket_reference_contract["robot_butt_to_blade_axis_local"],
        np.float64,
    )
    physical_racket_position = fresh_final.racket_position_w
    physical_racket_rotation = fresh_final.racket_rotation_w
    physical_signed_face = (
        physical_racket_rotation[:, 1] * float(mount_sign)
    )
    physical_long_axis = physical_racket_rotation @ racket_axis_local
    measured_signed_face = np.asarray(
        racket_reference_contract["signed_face_normal_w_unit"], np.float64
    )
    measured_long_axis = np.asarray(
        racket_reference_contract["long_axis_w_unit"], np.float64
    )
    racket_position_delta = physical_racket_position - racket_reference_position
    racket_rotation_delta = grounded._so3_log(
        racket_reference_rotation.T @ physical_racket_rotation
    )
    face_error_rad = math.acos(
        float(np.clip(physical_signed_face @ measured_signed_face, -1.0, 1.0))
    )
    long_axis_error_rad = math.acos(
        float(np.clip(physical_long_axis @ measured_long_axis, -1.0, 1.0))
    )
    provenance = {
        "semantics": MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS,
        "teacher_reference_unchanged": True,
        "historical_physical_birth_seed_consumed": False,
        "vendor_key_used_as_optimizer_start_only": bool(initial_states),
        "selection_priority": [
            "exact_measured_frame0_if_all_safety_gates_pass",
            "lexicographic_whole_body_safe_ready_only_if_frame0_unsafe",
        ],
        "exact_measured_frame0_selected": bool(
            exact_measured_frame0_selected
        ),
        "released_root_degrees_of_freedom": ["z", "roll", "pitch"],
        "released_joint_indices": list(range(31)),
        "released_joint_names": list(grounded.RUNTIME_JOINT_NAMES),
        "changed_joint_mask": list(result.changed_joint_mask),
        "changed_joint_indices": changed_indices,
        "changed_joint_names": [
            grounded.RUNTIME_JOINT_NAMES[index] for index in changed_indices
        ],
        "physical_minus_teacher_joint_pos_rad": list(result.joint_delta_rad),
        "physical_minus_teacher_joint_pos_by_name_rad": {
            name: float(result.joint_delta_rad[index])
            for index, name in enumerate(grounded.RUNTIME_JOINT_NAMES)
        },
        "physical_minus_teacher_root_pos_m": list(result.root_position_delta_m),
        "physical_minus_teacher_root_rotation_vector_rad": list(
            result.root_rotation_delta_rad
        ),
        "physical_root_quat_wxyz": selected_root_quat.tolist(),
        "stored_physical_root_quat_wxyz": selected_root_quat.tolist(),
        "mjcf_audit_root_quat_wxyz": result.state.root_quat_wxyz.tolist(),
        "teacher_root_quat_wxyz": stored_teacher_root_quat.tolist(),
        "teacher_and_physical_birth_differ": bool(
            not exact_measured_frame0_selected
        ),
        "racket_site_fidelity": {
            "site_name": MEASURED_PROJECTED_FRAME0_RACKET_SITE,
            "site_semantics": "official_mjcf_site_against_independent_schema_v4_measured_blade",
            "reference_authority": racket_reference_contract,
            "physical_site_pos_w_m": physical_racket_position.tolist(),
            "physical_signed_face_normal_w": physical_signed_face.tolist(),
            "physical_long_axis_w": physical_long_axis.tolist(),
            "physical_minus_measured_position_w_m": (
                racket_position_delta.tolist()
            ),
            "position_error_m": float(np.linalg.norm(racket_position_delta)),
            "physical_minus_measured_rotation_vector_rad": (
                racket_rotation_delta.tolist()
            ),
            "orientation_error_rad": float(np.linalg.norm(racket_rotation_delta)),
            "signed_face_error_rad": face_error_rad,
            "long_axis_error_rad": long_axis_error_rad,
            "independent_measured_frame0_required": True,
        },
        "safety_slacks": dict(fresh_final.slacks),
        "normalized_safety_slacks": dict(fresh_normalized_slacks),
        "worst_normalized_safety_slack": (
            min(fresh_normalized_slacks.values())
        ),
        "stage1_locked_worst_normalized_safety_slack": (
            result.stage1_locked_worst_normalized_slack
        ),
        "optimizer_report": dict(result.optimizer_report),
        "evaluator_contract": fresh_evaluator_contract,
        "safety_weighted_against_tracking": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "required_live_table_gate": "isaac_action_ball_nominal_hold_v1",
        "frame0_handoff": transition_contract,
    }
    static_evidence = {
        "authority": "fresh_current_exact_mjcf_whole_body_lexicographic_search",
        "selected_hold_witness_authority": (
            "new_backend_new_solver_final_state_cache_miss"
        ),
        "exact_contact_lp_reused": False,
        "all_safety_slacks_meet_original_and_locked_gate": all(
            value > search_config.positive_gate_normalized_slack
            and value >= required_final_gate
            for value in fresh_normalized_slacks.values()
        ) and fresh_direct_robust_gate_passed,
        "required_final_normalized_safety_gate": required_final_gate,
        "direct_frame0_robust_minimum_slacks": dict(
            whole_body_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS
        ),
        "direct_frame0_robust_gate_sha256": hashlib.sha256(
            _canonical_json_bytes(
                dict(whole_body_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS)
            )
        ).hexdigest(),
        "fresh_direct_robust_gate_passed": (
            fresh_direct_robust_gate_passed
            if exact_measured_frame0_selected
            else None
        ),
        "fresh_physical_birth_robust_gate_passed": (
            fresh_direct_robust_gate_passed
        ),
        "safety_slacks": dict(fresh_final.slacks),
        "normalized_safety_slacks": dict(fresh_normalized_slacks),
        "evaluator_evidence": fresh_witness,
        "stored_endpoint_state_sha256": selected_stored_state_sha256,
        "mjcf_audit_state_sha256": grounded.state_digest(result.state),
        "stored_root_quat_wxyz": selected_root_quat.tolist(),
        "mjcf_audit_root_quat_wxyz": result.state.root_quat_wxyz.tolist(),
        "stored_root_quaternion_norm": selected_quaternion_norm,
        "independent_measured_racket_frame0": racket_reference_contract,
        "racket_site_fidelity": provenance["racket_site_fidelity"],
        "frame0_handoff": transition_contract,
        "optimizer_report": dict(result.optimizer_report),
        "geometry_passed": True,
        "ground_dynamics_passed": True,
    }
    if (
        static_evidence[
            "all_safety_slacks_meet_original_and_locked_gate"
        ]
        is not True
    ):
        raise DynamicReadyMaterializationError(
            "whole-body solver returned a final safety slack below its gate"
        )
    return (
        selected_q,
        selected_root_pos,
        selected_root_quat,
        provenance,
        static_evidence,
    )


def _write_exclusive(path_value: str | Path, payload: bytes) -> Path:
    output = Path(path_value).expanduser().absolute()
    parent_input = output.parent
    try:
        parent = parent_input.resolve(strict=True)
    except OSError as exc:
        raise DynamicReadyMaterializationError(
            f"cannot resolve output parent: {exc}"
        ) from exc
    if parent_input != parent or not parent.is_dir() or not output.name:
        raise DynamicReadyMaterializationError(
            "output must have one concrete leaf under an existing real directory"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_descriptor = os.open(parent, parent_flags)
    descriptor = -1
    temporary_name: str | None = None
    try:
        for _attempt in range(128):
            candidate = f".a3-ready-{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    flags,
                    0o644,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise FileExistsError("cannot allocate a unique output temp file")
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            descriptor = -1
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)
    return output


def _exact_model_identity(
    receipt: Mapping[str, Any], *, mjcf_path: Path, mjcf_sha256: str
) -> grounded.ExactModelIdentity:
    try:
        exact = receipt["inputs"]["exact_model"]
    except (KeyError, TypeError) as exc:
        raise DynamicReadyMaterializationError(
            "stable receipt has no exact-model identity"
        ) from exc
    if not isinstance(exact, Mapping):
        raise DynamicReadyMaterializationError(
            "stable receipt exact-model identity must be an object"
        )
    if str(exact.get("mjcf_sha256")) != mjcf_sha256:
        raise DynamicReadyMaterializationError(
            "stable receipt and supplied MJCF have different SHA-256"
        )
    joint_order = tuple(str(value) for value in exact.get("joint_order", ()))
    if joint_order != grounded.RUNTIME_JOINT_NAMES:
        raise DynamicReadyMaterializationError(
            "stable receipt exact-model joint order is not the A3 runtime order"
        )
    pinned_mjcf_path = _pinned_mjcf_path(mjcf_path)
    return grounded.ExactModelIdentity(
        mjcf_path=str(pinned_mjcf_path),
        mjcf_sha256=mjcf_sha256,
        compiled_model_sha256=_require_sha256(
            exact.get("compiled_model_sha256"),
            name="compiled_model_sha256",
        ),
        path_model_binding_sha256=_require_sha256(
            exact.get("path_model_binding_sha256"),
            name="path_model_binding_sha256",
        ),
        ground_model_binding_sha256=_require_sha256(
            exact.get("ground_model_binding_sha256"),
            name="ground_model_binding_sha256",
        ),
        xml_model_name=str(exact.get("xml_model_name")),
    )


def _derive_exact_model_identity(
    *, mjcf_path: Path, mjcf_sha256: str
) -> grounded.ExactModelIdentity:
    """Derive and immediately re-verify exact model pins from a pinned MJCF.

    This is diagnostic materialization, not an external model certification.
    The resulting compiled/path/ground digests are content-sealed into the
    candidate and the ordinary backend reload verifies them before solving.
    """

    try:
        import mujoco

        pinned_mjcf_path = _pinned_mjcf_path(mjcf_path)
        expected_xml_model_name = _MJCF_XML_MODEL_NAME_BY_SHA256.get(mjcf_sha256)
        if expected_xml_model_name is None:
            raise DynamicReadyMaterializationError(
                "measured-branch MJCF generation is not registered"
            )
        model = mujoco.MjModel.from_xml_path(str(pinned_mjcf_path))
        compiled_sha = path_adapter.compiled_model_signature(model, mujoco)
        binding = path_adapter.bind_exact_mujoco_model(
            mujoco,
            model,
            mjcf_path=pinned_mjcf_path,
            expected_mjcf_sha256=mjcf_sha256,
            expected_compiled_model_sha256=compiled_sha,
            expected_xml_model_name=expected_xml_model_name,
        )
        ground_sha = torque_topp._mujoco_model_binding(model)
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"cannot derive exact measured-branch MuJoCo identity: {exc}"
        ) from exc
    return grounded.ExactModelIdentity(
        mjcf_path=str(pinned_mjcf_path),
        mjcf_sha256=mjcf_sha256,
        compiled_model_sha256=binding.compiled_model_sha256,
        path_model_binding_sha256=binding.model_binding_sha256,
        ground_model_binding_sha256=ground_sha,
        xml_model_name=binding.xml_model_name,
    )


def _audit_composed_physical_birth(
    *,
    ready: grounded.ReadyState,
    backend: grounded.MujocoGroundedReadyBackend,
    identity: grounded.ExactModelIdentity,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Require current-model static geometry and double-support dynamics."""

    try:
        targets = backend.foot_poses(ready)
        audit = grounded._audit_and_build_result(
            "measured-teacher-shared-lower-physical-birth",
            ready,
            targets,
            source=dict(source),
            backend=backend,
            expected_model_identity=identity,
            config=grounded.GroundedReadyConfig(),
        )
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"current-MJCF physical-birth static audit failed: {exc}"
        ) from exc
    gates = audit.receipt.get("gates")
    if (
        audit.geometry_passed is not True
        or audit.ground_dynamics_passed is not True
        or not isinstance(gates, Mapping)
    ):
        raise DynamicReadyMaterializationError(
            "composed physical birth failed current-MJCF static gates: "
            f"{dict(gates) if isinstance(gates, Mapping) else gates!r}"
        )
    return {
        "authority": "fresh_current_exact_mjcf_reaudit",
        "grounded_ready_receipt_sha256": audit.receipt_sha256,
        "geometry_passed": True,
        "ground_dynamics_passed": True,
        "gates": dict(gates),
    }


def _hard_inner_from_mechanical_limits(
    plant: Mapping[str, Any], *, margin_fraction: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the live policy bootstrap's physical-limit inner guard."""

    limits = np.asarray(
        plant["physx_control_position_limits"]["mechanical_joint_pos_limits"],
        dtype=np.float64,
    )
    if limits.shape != (31, 2) or not np.all(np.isfinite(limits)):
        raise DynamicReadyMaterializationError(
            "diagnostic plant template has malformed mechanical joint limits"
        )
    span = limits[:, 1] - limits[:, 0]
    if np.any(span <= 0.0):
        raise DynamicReadyMaterializationError(
            "diagnostic plant template has empty mechanical joint limits"
        )
    return (
        limits[:, 0] + margin_fraction * span,
        limits[:, 1] - margin_fraction * span,
    )


def _runtime_plant(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("schema_version") != 3:
        raise DynamicReadyMaterializationError(
            "runtime training contract schema_version must be 3"
        )
    names = tuple(str(value) for value in contract.get("joint_names", ()))
    articulation_names = tuple(
        str(value) for value in contract.get("articulation_joint_names", ())
    )
    action_joint_ids = contract.get("action_joint_ids")
    if (
        names != grounded.RUNTIME_JOINT_NAMES
        or articulation_names != names
        or action_joint_ids != list(range(len(names)))
    ):
        raise DynamicReadyMaterializationError(
            "runtime contract does not bind the exact full A3 action joint order"
        )
    count = len(names)
    kp = _plain_finite_vector(
        contract.get("joint_stiffness"), name="joint_stiffness", size=count
    )
    kd = _plain_finite_vector(
        contract.get("joint_damping"), name="joint_damping", size=count
    )
    effort = _plain_finite_vector(
        contract.get("joint_effort_limits"),
        name="joint_effort_limits",
        size=count,
    )
    velocity = _plain_finite_vector(
        contract.get("joint_velocity_limits"),
        name="joint_velocity_limits",
        size=count,
    )
    default_q = _plain_finite_vector(
        contract.get("default_joint_pos"),
        name="default_joint_pos",
        size=count,
    )
    action_scale = _plain_finite_vector(
        contract.get("action_scale"), name="action_scale", size=count
    )
    qdes_limits = _plain_finite_matrix(
        contract.get("qdes_joint_pos_limits"),
        name="qdes_joint_pos_limits",
        rows=count,
        columns=2,
    )
    physx_control_limits = _physx_control_position_limits(
        contract.get("physx_control_position_limits"),
        joint_names=names,
        qdes_limits=qdes_limits,
    )
    actuator_types = contract.get("joint_actuator_types")
    armature = _plain_finite_vector(
        contract.get("joint_armature"), name="joint_armature", size=count
    )
    friction = _plain_finite_vector(
        contract.get("joint_friction_coefficients"),
        name="joint_friction_coefficients",
        size=count,
    )
    if (
        np.any(kp <= 0.0)
        or np.any(kd < 0.0)
        or np.any(effort <= 0.0)
        or np.any(velocity <= 0.0)
        or np.any(action_scale <= 0.0)
        or np.any(armature < 0.0)
        or np.any(friction < 0.0)
        or np.any(qdes_limits[:, 0] >= qdes_limits[:, 1])
        or actuator_types != ["implicit"] * count
        or contract.get("action_use_default_offset") is not True
        or contract.get("joint_friction_backend") != "physx"
        or contract.get("joint_friction_semantics")
        != "load_dependent_spatial_force_coefficient"
        or contract.get("joint_friction_units") != "dimensionless"
        or contract.get("qdes_clamp") is not True
        or contract.get("finite_preclamp_qdes_projection_enabled") is not True
    ):
        raise DynamicReadyMaterializationError(
            "runtime contract has an invalid A3 implicit-PD/qdes contract"
        )
    inset = contract.get("finite_projection_soft_envelope_inset_fraction")
    if (
        isinstance(inset, bool)
        or type(inset) not in (int, float)
        or not math.isfinite(float(inset))
        or not 0.0 <= float(inset) < 0.5
    ):
        raise DynamicReadyMaterializationError(
            "runtime finite projection inset must lie in [0,0.5)"
        )
    physics_dt = float(contract.get("physics_step_dt_s", float("nan")))
    policy_dt = float(contract.get("policy_step_dt_s", float("nan")))
    decimation = contract.get("control_decimation")
    if (
        not math.isfinite(physics_dt)
        or physics_dt <= 0.0
        or not math.isfinite(policy_dt)
        or policy_dt <= 0.0
        or isinstance(decimation, bool)
        or type(decimation) is not int
        or decimation <= 0
        or not math.isclose(
            policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise DynamicReadyMaterializationError(
            "runtime physics/policy/decimation timing is inconsistent"
        )
    delay = contract.get("control_step_action_delay")
    if (
        type(delay) is not dict
        or delay.get("schema_version") != 1
        or delay.get("semantic_unit") != "policy_control_step"
        or delay.get("sample_timing") != "once_per_episode_reset"
        or delay.get("distribution") != "discrete_uniform_inclusive"
        or type(delay.get("enabled")) is not bool
        or isinstance(delay.get("min_steps"), bool)
        or type(delay.get("min_steps")) is not int
        or isinstance(delay.get("max_steps"), bool)
        or type(delay.get("max_steps")) is not int
        or delay["min_steps"] < 0
        or delay["max_steps"] < delay["min_steps"]
        or delay.get("shared_across_all_31_joints") is not True
        or delay.get("history_fill")
        != "safe_default_or_action_specific_hold"
        or delay["enabled"] != (delay["max_steps"] > 0)
    ):
        raise DynamicReadyMaterializationError(
            "runtime control-step action delay contract is invalid"
        )
    return {
        "joint_names": names,
        "kp": kp,
        "kd": kd,
        "effort": effort,
        "velocity": velocity,
        "default_q": default_q,
        "action_scale": action_scale,
        "qdes_limits": qdes_limits,
        "physx_control_position_limits": physx_control_limits,
        "projection_inset": float(inset),
        "physics_dt": physics_dt,
        "policy_dt": policy_dt,
        "decimation": decimation,
        "control_step_action_delay": dict(delay),
        "actuator_types": actuator_types,
        "armature": armature,
        "friction": friction,
        "friction_backend": contract.get("joint_friction_backend"),
        "friction_semantics": contract.get("joint_friction_semantics"),
        "friction_units": contract.get("joint_friction_units"),
    }


def _bind_action_runtime(
    contract: Mapping[str, Any],
    *,
    action_id: str,
    motion_sha256: str,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        training = contract["action_ball_training"]
        preflight = training["preflight"]
        bootstrap = training["policy_bootstrap"]
        admission = training["motion_admission"]
        action_runtime = training["runtime"]
    except (KeyError, TypeError) as exc:
        raise DynamicReadyMaterializationError(
            "runtime contract has no exact ActionBall N=1 binding"
        ) from exc
    if (
        not isinstance(preflight, Mapping)
        or not isinstance(bootstrap, Mapping)
        or not isinstance(admission, Mapping)
        or not isinstance(action_runtime, Mapping)
        or preflight.get("action_order") != [action_id]
        or bootstrap.get("action_order") != [action_id]
        or admission.get("motion_file_sha256") != [motion_sha256]
        or action_runtime.get("action_order") != [action_id]
        or bootstrap.get("joint_names") != list(grounded.RUNTIME_JOINT_NAMES)
    ):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall action order, motion, or joint binding drifted"
        )
    bindings = preflight.get("action_bindings")
    ready_source = bootstrap.get("ready_source")
    decoder = bootstrap.get("decoder")
    guard = bootstrap.get("hard_inner_guard")
    runtime_bindings = action_runtime.get("bindings")
    if (
        preflight.get("schema_version") != 1
        or bootstrap.get("schema_version") != 1
        or bootstrap.get("kind")
        != "action_ball_shared_ready_actor_bootstrap_v1"
        or bootstrap.get("action_count") != 1
        or admission.get("schema_version") != 1
        or not isinstance(bindings, list)
        or len(bindings) != 1
        or not isinstance(bindings[0], Mapping)
        or bindings[0].get("action_id") != action_id
        or bindings[0].get("action_slot") != 0
        or bindings[0].get("motion_sha256") != motion_sha256
        or not isinstance(ready_source, Mapping)
        or ready_source.get("motion_sha256_per_action") != [motion_sha256]
        or not isinstance(decoder, Mapping)
        or decoder.get("use_default_offset") is not True
        or not isinstance(runtime_bindings, list)
        or len(runtime_bindings) != 1
        or not isinstance(runtime_bindings[0], Mapping)
        or runtime_bindings[0].get("action_slot") != 0
        or runtime_bindings[0].get("motion_sha256") != motion_sha256
        or not isinstance(guard, Mapping)
        or guard.get("limit_source")
        != "articulation.data.joint_pos_limits"
        or guard.get("margin_fraction") != 0.02
        or guard.get("margin_rad") != 0.0
    ):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall N=1 bootstrap does not bind this action motion"
        )
    hard_lower = _plain_finite_vector(
        guard.get("hard_inner_lower"),
        name="policy_bootstrap.hard_inner_lower",
        size=size,
    )
    hard_upper = _plain_finite_vector(
        guard.get("hard_inner_upper"),
        name="policy_bootstrap.hard_inner_upper",
        size=size,
    )
    if np.any(hard_lower >= hard_upper):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall hard-inner guard is empty"
        )
    shared_ready = _plain_finite_vector(
        ready_source.get("shared_ready_joint_pos"),
        name="policy_bootstrap.shared_ready_joint_pos",
        size=size,
    )
    decoder_default = _plain_finite_vector(
        decoder.get("default_joint_pos"),
        name="policy_bootstrap.decoder.default_joint_pos",
        size=size,
    )
    decoder_scale = _plain_finite_vector(
        decoder.get("action_scale"),
        name="policy_bootstrap.decoder.action_scale",
        size=size,
    )
    return (
        hard_lower,
        hard_upper,
        shared_ready,
        decoder_default,
        decoder_scale,
    )


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_kind = getattr(args, "ready_source_kind", STABLE_UPPER_SOURCE_KIND)
    measured_birth_mode = getattr(
        args,
        "physical_birth_composition_mode",
        MEASURED_BIRTH_SHARED_LOWER_MODE,
    )
    robust_contact_normal_n = float(
        getattr(
            args,
            "full_seed_minimum_normal_force_per_support_vertex_n",
            0.0,
        )
    )
    full_seed_qdes_mode = getattr(
        args,
        "full_seed_hold_qdes_mode",
        FULL_SEED_QDES_FRESH_STATIC_LP,
    )
    if source_kind not in (
        STABLE_UPPER_SOURCE_KIND,
        MEASURED_RETARGET_SOURCE_KIND,
    ):
        raise DynamicReadyMaterializationError("unsupported ready-source kind")
    if measured_birth_mode not in (
        MEASURED_BIRTH_SHARED_LOWER_MODE,
        MEASURED_BIRTH_FULL_SEED_MODE,
        MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE,
        MEASURED_BIRTH_DIRECT_FRAME0_MODE,
        MEASURED_BIRTH_PROJECTED_FRAME0_MODE,
        MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE,
    ):
        raise DynamicReadyMaterializationError(
            "unsupported measured physical-birth composition mode"
        )
    if full_seed_qdes_mode not in (
        FULL_SEED_QDES_FRESH_STATIC_LP,
        FULL_SEED_QDES_SEED_TRANSPORT,
    ):
        raise DynamicReadyMaterializationError(
            "unsupported full-seed hold-qdes mode"
        )
    if (
        source_kind == STABLE_UPPER_SOURCE_KIND
        and measured_birth_mode != MEASURED_BIRTH_SHARED_LOWER_MODE
    ):
        raise DynamicReadyMaterializationError(
            "stable-upper branch cannot select a measured birth mode"
        )
    if (
        not math.isfinite(robust_contact_normal_n)
        or robust_contact_normal_n < 0.0
        or (
            robust_contact_normal_n > 0.0
            and (
                source_kind != MEASURED_RETARGET_SOURCE_KIND
                or measured_birth_mode
                not in (
                    MEASURED_BIRTH_FULL_SEED_MODE,
                    MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE,
                )
            )
        )
    ):
        raise DynamicReadyMaterializationError(
            "per-support-vertex normal reserve is a finite non-negative "
            "full-seed measured diagnostic option"
        )
    if (
        full_seed_qdes_mode == FULL_SEED_QDES_SEED_TRANSPORT
        and (
            source_kind != MEASURED_RETARGET_SOURCE_KIND
            or measured_birth_mode != MEASURED_BIRTH_FULL_SEED_MODE
            or robust_contact_normal_n != 0.0
        )
    ):
        raise DynamicReadyMaterializationError(
            "seed-transport qdes requires full-seed measured birth and cannot "
            "be combined with the independently infeasible contact reserve"
        )
    motion_path, motion_sha = _pinned_file(
        args.motion,
        args.expected_motion_sha256,
        name=(
            "stable motion"
            if source_kind == STABLE_UPPER_SOURCE_KIND
            else "measured motion"
        ),
    )
    runtime_path, runtime_sha = _pinned_file(
        args.runtime_contract,
        args.expected_runtime_contract_sha256,
        name="runtime training contract",
    )
    mjcf_path, mjcf_sha = _pinned_file(
        args.mjcf,
        args.expected_mjcf_sha256,
        name="A3 MJCF",
    )
    runtime_contract = _read_json(runtime_path, name="runtime training contract")
    seedless_frame0_birth = measured_birth_mode in (
        MEASURED_BIRTH_DIRECT_FRAME0_MODE,
        MEASURED_BIRTH_PROJECTED_FRAME0_MODE,
        MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE,
    )
    whole_body_optimizer_seed_requested = bool(
        measured_birth_mode == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE
        and getattr(args, "physical_birth_seed", None)
        and getattr(args, "expected_physical_birth_seed_sha256", None)
    )
    if measured_birth_mode == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE and bool(
        getattr(args, "physical_birth_seed", None)
    ) != bool(getattr(args, "expected_physical_birth_seed_sha256", None)):
        raise DynamicReadyMaterializationError(
            "whole-body optimizer start requires both seed path and exact SHA"
        )
    if source_kind == MEASURED_RETARGET_SOURCE_KIND and seedless_frame0_birth:
        teacher_q, teacher_root_pos, teacher_root_quat = (
            _load_motion_frame0_exact(motion_path)
        )
    else:
        teacher_q, teacher_root_pos, teacher_root_quat = _load_motion_frame0(
            motion_path
        )
    ready_q = teacher_q.copy()
    ready_root_pos = teacher_root_pos.copy()
    ready_root_quat = teacher_root_quat.copy()
    plant = _runtime_plant(runtime_contract)
    stable_receipt = None
    receipt_path = None
    receipt_sha = None
    bank_receipt_path = None
    bank_receipt_sha = None
    mechanical_audit_path = None
    mechanical_audit_sha = None
    physical_birth_seed_path = None
    physical_birth_seed_sha = None
    physical_birth_seed = None
    seed_hold_transport = None
    physical_birth_composition = None
    seed_foot_poses = None
    seed_yaw_alignment_evidence = None
    measured_evidence = None
    static_birth_evidence = None
    if source_kind == STABLE_UPPER_SOURCE_KIND:
        if not getattr(args, "stable_receipt", None) or not getattr(
            args, "expected_stable_receipt_sha256", None
        ):
            raise DynamicReadyMaterializationError(
                "stable-upper branch requires --stable-receipt and "
                "--expected-stable-receipt-sha256"
            )
        receipt_path, receipt_sha = _pinned_file(
            args.stable_receipt,
            args.expected_stable_receipt_sha256,
            name="stable receipt",
        )
        stable_receipt = _read_json(receipt_path, name="stable receipt")
        _validate_stable_receipt(stable_receipt, motion_sha256=motion_sha)
        if runtime_contract.get("target_mode") != "action_ball":
            raise DynamicReadyMaterializationError(
                "runtime contract is not an ActionBall contract"
            )
        (
            hard_inner_lower,
            hard_inner_upper,
            bootstrap_ready,
            bootstrap_default,
            bootstrap_scale,
        ) = _bind_action_runtime(
            runtime_contract,
            action_id=str(args.action_id),
            motion_sha256=motion_sha,
            size=len(plant["joint_names"]),
        )
        if (
            not np.array_equal(bootstrap_ready, ready_q)
            or not np.array_equal(bootstrap_default, plant["default_q"])
            or not np.array_equal(bootstrap_scale, plant["action_scale"])
        ):
            raise DynamicReadyMaterializationError(
                "runtime ActionBall bootstrap decoder or ready pose differs "
                "from the physical motion/runtime plant"
            )
        identity = _exact_model_identity(
            stable_receipt, mjcf_path=mjcf_path, mjcf_sha256=mjcf_sha
        )
        if (
            stable_receipt["robot"].get("exact_xml_model_name")
            != identity.xml_model_name
        ):
            raise DynamicReadyMaterializationError(
                "stable receipt robot model name differs from its exact-model identity"
            )
        try:
            ready_root_z = runtime_contract["action_ball_training"]["preflight"][
                "ready_root_z_by_slot_m"
            ]
        except (KeyError, TypeError) as exc:
            raise DynamicReadyMaterializationError(
                "runtime ActionBall preflight has no ready-root binding"
            ) from exc
        if (
            not isinstance(ready_root_z, list)
            or len(ready_root_z) != 1
            or not math.isclose(
                float(ready_root_z[0]),
                float(ready_root_pos[2]),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise DynamicReadyMaterializationError(
                "runtime ActionBall ready-root height differs from motion frame 0"
            )
    else:
        _validate_diagnostic_plant_template(runtime_contract)
        measured_uid = str(getattr(args, "measured_uid", "") or "")
        if not measured_uid:
            raise DynamicReadyMaterializationError(
                "measured teacher/physical-split branch requires --measured-uid"
            )
        if not getattr(args, "measured_bank_receipt", None) or not getattr(
            args, "expected_measured_bank_receipt_sha256", None
        ):
            raise DynamicReadyMaterializationError(
                "measured teacher/physical-split branch requires the exact "
                "bank receipt "
                "path and SHA"
            )
        if not getattr(args, "mechanical_audit", None) or not getattr(
            args, "expected_mechanical_audit_sha256", None
        ):
            raise DynamicReadyMaterializationError(
                "measured teacher/physical-split branch requires the exact mechanical "
                "audit path and SHA"
            )
        if (
            not seedless_frame0_birth
            and (
                not getattr(args, "physical_birth_seed", None)
                or not getattr(args, "expected_physical_birth_seed_sha256", None)
            )
        ):
            raise DynamicReadyMaterializationError(
                "measured branch requires --physical-birth-seed and its exact SHA"
            )
        bank_receipt_path, bank_receipt_sha = _pinned_file(
            args.measured_bank_receipt,
            args.expected_measured_bank_receipt_sha256,
            name="measured bank receipt",
        )
        mechanical_audit_path, mechanical_audit_sha = _pinned_file(
            args.mechanical_audit,
            args.expected_mechanical_audit_sha256,
            name="measured mechanical audit",
        )
        if not seedless_frame0_birth or whole_body_optimizer_seed_requested:
            physical_birth_seed_path, physical_birth_seed_sha = _pinned_file(
                args.physical_birth_seed,
                args.expected_physical_birth_seed_sha256,
                name="physical-birth numerical seed",
            )
        measured_evidence = _validate_measured_retarget_l0_evidence(
            motion_path=motion_path,
            motion_sha256=motion_sha,
            measured_uid=measured_uid,
            bank_receipt=_read_json(
                bank_receipt_path, name="measured bank receipt"
            ),
            bank_receipt_sha256=bank_receipt_sha,
            mechanical_audit=_read_json(
                mechanical_audit_path, name="measured mechanical audit"
            ),
            allow_mechanical_unknown=(
                getattr(args, "allow_mechanical_unknown_diagnostic", False)
                is True
            ),
        )
        if not seedless_frame0_birth or whole_body_optimizer_seed_requested:
            physical_birth_seed_document = _read_json(
                physical_birth_seed_path,
                name="physical-birth numerical seed",
            )
            physical_birth_seed = _load_physical_birth_seed(
                physical_birth_seed_document,
                joint_names=plant["joint_names"],
            )
            if full_seed_qdes_mode == FULL_SEED_QDES_SEED_TRANSPORT:
                seed_hold_transport = _load_seed_hold_transport(
                    physical_birth_seed_document,
                    joint_names=plant["joint_names"],
                )
        hard_inner_lower, hard_inner_upper = (
            _hard_inner_from_mechanical_limits(plant)
        )
        # Tonight's measured N1 diagnostic is delay-zero.  The template supplies
        # action-independent A3 plant values; delay is a code-owned task choice
        # and is re-proved against the live Isaac action term by nominal-hold.
        plant = dict(plant)
        plant["control_step_action_delay"] = dict(
            MEASURED_DIAGNOSTIC_DELAY_CONTRACT
        )
        identity = _derive_exact_model_identity(
            mjcf_path=mjcf_path, mjcf_sha256=mjcf_sha
        )
    backend = grounded.MujocoGroundedReadyBackend.load(identity)
    if source_kind == MEASURED_RETARGET_SOURCE_KIND:
        if measured_birth_mode == MEASURED_BIRTH_DIRECT_FRAME0_MODE:
            (
                ready_q,
                ready_root_pos,
                ready_root_quat,
                physical_birth_composition,
            ) = _compose_measured_direct_frame0_physical_birth(
                teacher_q=teacher_q,
                teacher_root_pos=teacher_root_pos,
                teacher_root_quat=teacher_root_quat,
            )
        elif measured_birth_mode == MEASURED_BIRTH_PROJECTED_FRAME0_MODE:
            (
                ready_q,
                ready_root_pos,
                ready_root_quat,
                physical_birth_composition,
                static_birth_evidence,
            ) = _compose_measured_projected_frame0_physical_birth(
                teacher_q=teacher_q,
                teacher_root_pos=teacher_root_pos,
                teacher_root_quat=teacher_root_quat,
                backend=backend,
                identity=identity,
            )
        elif measured_birth_mode == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE:
            optimizer_initial_states: tuple[grounded.ReadyState, ...] = ()
            if whole_body_optimizer_seed_requested:
                optimizer_seed = grounded.ReadyState(
                    np.asarray(
                        physical_birth_seed["joint_pos_rad"], np.float64
                    ),
                    np.asarray(
                        physical_birth_seed["root_pos_w_m"], np.float64
                    ),
                    np.asarray(
                        physical_birth_seed["root_quat_wxyz"], np.float64
                    ),
                )
                optimizer_seed_foot_poses = backend.foot_poses(optimizer_seed)
                aligned_optimizer_seed = _align_seed_world_yaw_to_teacher(
                    seed=physical_birth_seed,
                    teacher_root_quat=teacher_root_quat,
                    seed_foot_positions_w=[
                        pose.position_w for pose in optimizer_seed_foot_poses
                    ],
                )
                optimizer_initial_states = (
                    grounded.ReadyState(
                        np.asarray(
                            aligned_optimizer_seed["joint_pos_rad"], np.float64
                        ),
                        np.asarray(
                            aligned_optimizer_seed["root_pos_w_m"], np.float64
                        ),
                        np.asarray(
                            aligned_optimizer_seed["root_quat_wxyz"], np.float64
                        ),
                    ),
                )
            (
                ready_q,
                ready_root_pos,
                ready_root_quat,
                physical_birth_composition,
                static_birth_evidence,
            ) = _compose_measured_whole_body_safe_frame0_physical_birth(
                teacher_q=teacher_q,
                teacher_root_pos=teacher_root_pos,
                teacher_root_quat=teacher_root_quat,
                backend=backend,
                identity=identity,
                plant=plant,
                hard_inner_lower=hard_inner_lower,
                hard_inner_upper=hard_inner_upper,
                runtime_contract=runtime_contract,
                measured_racket_frame0=measured_evidence.get(
                    "measured_racket_frame0"
                ),
                motion_sha256=motion_sha,
                optimizer_initial_states=optimizer_initial_states,
            )
        else:
            seed_ready = grounded.ReadyState(
                np.asarray(physical_birth_seed["joint_pos_rad"], np.float64),
                np.asarray(physical_birth_seed["root_pos_w_m"], np.float64),
                np.asarray(physical_birth_seed["root_quat_wxyz"], np.float64),
            )
            seed_foot_poses = backend.foot_poses(seed_ready)
            physical_birth_seed = _align_seed_world_yaw_to_teacher(
                seed=physical_birth_seed,
                teacher_root_quat=teacher_root_quat,
                seed_foot_positions_w=[
                    pose.position_w for pose in seed_foot_poses
                ],
            )
            compose_birth = (
                _compose_measured_physical_birth
                if measured_birth_mode == MEASURED_BIRTH_SHARED_LOWER_MODE
                else _compose_measured_full_seed_physical_birth
            )
            (
                ready_q,
                ready_root_pos,
                ready_root_quat,
                physical_birth_composition,
            ) = compose_birth(
                teacher_q=teacher_q,
                teacher_root_pos=teacher_root_pos,
                teacher_root_quat=teacher_root_quat,
                seed=physical_birth_seed,
            )
    backend_ready_root_quat = ready_root_quat
    if seedless_frame0_birth:
        stored_quaternion_norm = float(np.linalg.norm(ready_root_quat))
        backend_ready_root_quat = ready_root_quat / stored_quaternion_norm
        physical_birth_composition["current_mjcf_audit_quaternion"] = {
            "semantics": "unit_normalization_for_numerical_backend_only",
            "stored_teacher_and_physical_quaternion_unchanged": (
                True
            ),
            "stored_quaternion_norm": stored_quaternion_norm,
            "backend_root_quat_wxyz": backend_ready_root_quat.tolist(),
        }
    ready = grounded.ReadyState(
        ready_q, ready_root_pos, backend_ready_root_quat
    )
    if (
        source_kind == MEASURED_RETARGET_SOURCE_KIND
        and measured_birth_mode == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE
    ):
        handoff = physical_birth_composition.get("frame0_handoff")
        stored_physical_state_sha = _stored_ready_state_sha256(
            ready_q, ready_root_pos, ready_root_quat
        )
        stored_teacher_state_sha = _stored_ready_state_sha256(
            teacher_q, teacher_root_pos, teacher_root_quat
        )
        exact_selected = physical_birth_composition.get(
            "exact_measured_frame0_selected"
        )
        if (
            not isinstance(handoff, Mapping)
            or type(exact_selected) is not bool
            or handoff.get("physical_ready_state_sha256")
            != stored_physical_state_sha
            or handoff.get("teacher_frame0_state_sha256")
            != stored_teacher_state_sha
            or handoff.get("mjcf_audit_state_sha256")
            != grounded.state_digest(ready)
            or not np.array_equal(
                np.asarray(handoff.get("mjcf_audit_root_quat_wxyz", ())),
                ready.root_quat_wxyz,
            )
            or (
                exact_selected
                and (
                    not np.array_equal(ready_q, teacher_q)
                    or not np.array_equal(ready_root_pos, teacher_root_pos)
                    or not np.array_equal(ready_root_quat, teacher_root_quat)
                    or handoff.get("endpoints_bitwise_equal") is not True
                    or handoff.get("kind")
                    != "exact_frame0_zero_duration_handoff_v1"
                )
            )
            or (
                not exact_selected
                and (
                    handoff.get("endpoints_bitwise_equal") is not False
                    or handoff.get("kind")
                    != "policy_learned_physical_birth_to_teacher_reference_v1"
                    or handoff.get("runtime_transition_reference_required")
                    is not True
                )
            )
        ):
            raise DynamicReadyMaterializationError(
                "whole-body learned bridge does not bind emitted physical and teacher states"
            )
    if source_kind == MEASURED_RETARGET_SOURCE_KIND:
        if not seedless_frame0_birth:
            seed_yaw_alignment_evidence = _audit_realized_seed_yaw_alignment(
                backend=backend,
                seed_foot_poses=seed_foot_poses,
                ready=ready,
                alignment=physical_birth_composition["seed_world_yaw_alignment"],
            )
            physical_birth_composition["seed_world_yaw_alignment"][
                "realized_current_mjcf_fk"
            ] = seed_yaw_alignment_evidence
        static_source = {
            "mode": physical_birth_composition["semantics"],
            "teacher_motion_sha256": motion_sha,
            "teacher_frame": 0,
        }
        if physical_birth_seed_sha is not None and not seedless_frame0_birth:
            static_source["physical_birth_seed_sha256"] = (
                physical_birth_seed_sha
            )
            static_source["seed_world_yaw_alignment"] = (
                physical_birth_composition["seed_world_yaw_alignment"]
            )
        elif whole_body_optimizer_seed_requested:
            static_source["optimizer_start_sha256"] = physical_birth_seed_sha
            static_source["optimizer_start_selected_state_inherited"] = False
        if static_birth_evidence is None:
            static_birth_evidence = _audit_composed_physical_birth(
                ready=ready,
                backend=backend,
                identity=identity,
                source=static_source,
            )
    qpos = backend._qpos(ready)

    whole_body_selected_hold = bool(
        source_kind == MEASURED_RETARGET_SOURCE_KIND
        and measured_birth_mode == MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE
    )
    selected_contact_normal_n = (
        WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
        if whole_body_selected_hold
        else robust_contact_normal_n
    )
    ground_config = torque_topp.GroundContactConfig(
        expected_model_binding=identity.ground_model_binding_sha256,
        model_source_path=str(_pinned_mjcf_path(mjcf_path)),
        expected_source_sha256=mjcf_sha,
        minimum_normal_force_per_contact_n=selected_contact_normal_n,
    )
    solver = (
        None
        if whole_body_selected_hold
        else torque_topp.MujocoGroundContactLPSolver(
            backend.model, ground_config
        )
    )
    actuator_contract = torque_topp.direct_actuator_contract_from_mujoco(
        backend.model,
        support_mode="ground",
        contact_mode="double_support_floor",
        fixed_lp_solver="scipy.optimize.linprog:highs",
    )
    (
        model_tau_lower,
        model_tau_upper,
        actuated,
        actuator_limit_report,
    ) = torque_topp._resolve_grounded_actuator_limits(
        actuator_contract, int(backend.model.nv)
    )
    model_row_for_runtime = (
        np.asarray(backend._binding.joint_dof_adrs, np.int64) - 6
    )
    expected_model_rows = np.arange(len(plant["joint_names"]), dtype=np.int64)
    if (
        model_row_for_runtime.shape != expected_model_rows.shape
        or not np.array_equal(np.sort(model_row_for_runtime), expected_model_rows)
        or not np.array_equal(
            np.asarray(actuated, np.int64),
            expected_model_rows + 6,
        )
    ):
        raise DynamicReadyMaterializationError(
            "exact A3 runtime-to-MuJoCo actuator rows are not one full permutation"
        )

    qdes_limits = plant["qdes_limits"]
    inset = plant["projection_inset"]
    span = qdes_limits[:, 1] - qdes_limits[:, 0]
    projected_soft_lower = qdes_limits[:, 0] + inset * span
    projected_soft_upper = qdes_limits[:, 1] - inset * span
    executed_qdes_lower = np.maximum(projected_soft_lower, hard_inner_lower)
    executed_qdes_upper = np.minimum(projected_soft_upper, hard_inner_upper)
    if np.any(executed_qdes_lower >= executed_qdes_upper):
        raise DynamicReadyMaterializationError(
            "runtime projected-soft and hard-inner qdes envelopes do not intersect"
        )
    if np.any(ready_q <= executed_qdes_lower) or np.any(
        ready_q >= executed_qdes_upper
    ):
        raise DynamicReadyMaterializationError(
            "physical ready lies outside the executed qdes envelope"
        )
    def hold_envelope(joint_pos: np.ndarray) -> tuple[np.ndarray, ...]:
        runtime_lower = np.maximum(
            -plant["effort"],
            plant["kp"] * (executed_qdes_lower - joint_pos),
        )
        runtime_upper = np.minimum(
            plant["effort"],
            plant["kp"] * (executed_qdes_upper - joint_pos),
        )
        runtime_lower_model = np.empty_like(runtime_lower)
        runtime_upper_model = np.empty_like(runtime_upper)
        runtime_lower_model[model_row_for_runtime] = runtime_lower
        runtime_upper_model[model_row_for_runtime] = runtime_upper
        return (
            runtime_lower,
            runtime_upper,
            runtime_lower_model,
            runtime_upper_model,
            np.maximum(model_tau_lower, runtime_lower_model),
            np.minimum(model_tau_upper, runtime_upper_model),
        )

    (
        runtime_tau_lower,
        runtime_tau_upper,
        runtime_tau_lower_model,
        runtime_tau_upper_model,
        hold_tau_lower_model,
        hold_tau_upper_model,
    ) = hold_envelope(ready_q)
    if np.any(hold_tau_lower_model >= 0.0) or np.any(
        hold_tau_upper_model <= 0.0
    ):
        raise DynamicReadyMaterializationError(
            "hold torque envelope must contain zero on both sides"
        )

    if whole_body_selected_hold:
        solution = _consume_whole_body_selected_hold_witness(
            static_birth_evidence=static_birth_evidence,
            ready_q=ready_q,
            ready_root_pos=ready_root_pos,
            ready_root_quat=ready.root_quat_wxyz,
            identity=identity,
            kp=plant["kp"],
            model_row_for_runtime=model_row_for_runtime,
            actuated=np.asarray(actuated, np.int64),
            expected_vectors={
                "executed_qdes_lower_rad": executed_qdes_lower,
                "executed_qdes_upper_rad": executed_qdes_upper,
                "model_tau_lower_mujoco_row_order_nm": model_tau_lower,
                "model_tau_upper_mujoco_row_order_nm": model_tau_upper,
                "runtime_tau_lower_runtime_order_nm": runtime_tau_lower,
                "runtime_tau_upper_runtime_order_nm": runtime_tau_upper,
                "runtime_tau_lower_mujoco_row_order_nm": (
                    runtime_tau_lower_model
                ),
                "runtime_tau_upper_mujoco_row_order_nm": (
                    runtime_tau_upper_model
                ),
                "effective_tau_lower_mujoco_row_order_nm": (
                    hold_tau_lower_model
                ),
                "effective_tau_upper_mujoco_row_order_nm": (
                    hold_tau_upper_model
                ),
            },
        )
    else:
        assert solver is not None
        project_contact_free = (
            source_kind == MEASURED_RETARGET_SOURCE_KIND
            and measured_birth_mode
            == MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE
        )
        projection_origin = ready_q.copy()
        projection_origin_feet = (
            backend.foot_poses(ready) if project_contact_free else None
        )
        projection_history: list[dict[str, Any]] = []
        for projection_iteration in range(9):
            solution = solver.solve(
                qpos,
                np.zeros(int(backend.model.nv), np.float64),
                np.zeros(int(backend.model.nv), np.float64),
                actuated,
                hold_tau_lower_model,
                hold_tau_upper_model,
                np.full(int(backend.model.nv), 1.0e6, np.float64),
                path_tangent=np.zeros(int(backend.model.nv), np.float64),
                lp_objective=LP_OBJECTIVE,
            )
            if solution.feasible or not project_contact_free:
                break
            if projection_iteration == 8:
                raise DynamicReadyMaterializationError(
                    "full-seed contact-free hold projection did not converge in 8 exact LP corrections"
                )
            contact_free_runtime = contact_free_actuated_rows(
                backend.model, np.asarray(actuated, np.int64)
            )[model_row_for_runtime]
            bias = static_hold_required_generalized_force(
                backend.model, qpos
            )
            records = contact_free_hold_torque_shortfall(
                joint_names=list(plant["joint_names"]),
                contact_free=contact_free_runtime,
                required_nm=bias[np.asarray(actuated, np.int64)][
                    model_row_for_runtime
                ],
                tau_lower_nm=hold_tau_lower_model[model_row_for_runtime],
                tau_upper_nm=hold_tau_upper_model[model_row_for_runtime],
                kp=plant["kp"],
                ready_q_rad=ready_q,
                executed_qdes_lower_rad=executed_qdes_lower,
                executed_qdes_upper_rad=executed_qdes_upper,
                motor_effort_nm=plant["effort"],
            )
            if not records:
                raise DynamicReadyMaterializationError(
                    "full-seed hold projection cannot repair a ground-loaded or friction-cone LP refusal"
                )
            record = records[0]
            joint_name = str(record["joint"])
            joint_index = list(plant["joint_names"]).index(joint_name)
            if (
                joint_index in physical_birth_seed["leg_joint_indices"]
                or record.get("binding_authority")
                != "kp_times_available_qdes_travel"
            ):
                raise DynamicReadyMaterializationError(
                    "contact-free projection refuses leg or non-PD-travel shortfalls"
                )
            binding_side = str(record["binding_side"])

            def exact_row_slack(candidate: float) -> float:
                trial_q = ready_q.copy()
                trial_q[joint_index] = candidate
                trial = grounded.ReadyState(
                    trial_q, ready_root_pos, backend_ready_root_quat
                )
                trial_qpos = backend._qpos(trial)
                trial_bias = static_hold_required_generalized_force(
                    backend.model, trial_qpos
                )
                trial_lower, trial_upper = hold_envelope(trial_q)[4:]
                required = trial_bias[np.asarray(actuated, np.int64)][
                    model_row_for_runtime
                ][joint_index]
                low = trial_lower[model_row_for_runtime][joint_index]
                high = trial_upper[model_row_for_runtime][joint_index]
                return float(
                    required - low
                    if binding_side == "lower"
                    else high - required
                ) - CONTACT_FREE_PROJECTION_MINIMUM_TORQUE_SLACK_NM

            projected, boundary = nearest_feasible_scalar_boundary(
                current=float(ready_q[joint_index]),
                lower=float(
                    np.nextafter(
                        executed_qdes_lower[joint_index],
                        executed_qdes_upper[joint_index],
                    )
                ),
                upper=float(
                    np.nextafter(
                        executed_qdes_upper[joint_index],
                        executed_qdes_lower[joint_index],
                    )
                ),
                initial_step=max(
                    abs(float(record["shortfall_nm"]))
                    / float(record["kp"]),
                    1.0e-5,
                ),
                slack_at=exact_row_slack,
            )
            ready_q = ready_q.copy()
            ready_q[joint_index] = projected
            applied = [
                {
                    "joint": joint_name,
                    "joint_index": joint_index,
                    "source_shortfall_nm": float(record["shortfall_nm"]),
                    **boundary,
                }
            ]
            projection_history.append(
                {
                    "iteration": projection_iteration,
                    "attributed_shortfalls": records,
                    "applied_corrections": applied,
                }
            )
            ready = grounded.ReadyState(
                ready_q, ready_root_pos, backend_ready_root_quat
            )
            qpos = backend._qpos(ready)
            (
                runtime_tau_lower,
                runtime_tau_upper,
                runtime_tau_lower_model,
                runtime_tau_upper_model,
                hold_tau_lower_model,
                hold_tau_upper_model,
            ) = hold_envelope(ready_q)
        if project_contact_free and solution.feasible:
            assert projection_origin_feet is not None
            projected_feet = backend.foot_poses(ready)
            foot_delta = max(
                max(
                    float(
                        np.max(np.abs(after.position_w - before.position_w))
                    ),
                    float(
                        np.max(np.abs(after.rotation_w - before.rotation_w))
                    ),
                )
                for before, after in zip(
                    projection_origin_feet, projected_feet, strict=True
                )
            )
            if foot_delta > 1.0e-12:
                raise DynamicReadyMaterializationError(
                    "contact-free hold projection changed a support-foot pose"
                )
            delta = ready_q - projection_origin
            changed = np.flatnonzero(delta != 0.0)
            physical_birth_composition.update(
                {
                    "semantics": (
                        MEASURED_BIRTH_HOLDABLE_FULL_SEED_SEMANTICS
                    ),
                    "seed_all_joints_exactly_preserved": False,
                    "seed_root_and_leg_joints_exactly_preserved": True,
                    "physical_minus_seed_joint_pos_rad": delta.tolist(),
                    "physical_minus_teacher_joint_pos_rad": (
                        ready_q - teacher_q
                    ).tolist(),
                    "contact_free_hold_projection": {
                        "schema_version": 1,
                        "semantics": (
                            "iterated_exact_bias_contact_free_pd_travel_projection"
                        ),
                        "root_changed": False,
                        "leg_joints_changed": False,
                        "support_foot_pose_max_abs_delta": foot_delta,
                        "changed_joint_indices": changed.tolist(),
                        "changed_joint_names": [
                            str(plant["joint_names"][index])
                            for index in changed
                        ],
                        "joint_delta_rad": delta.tolist(),
                        "maximum_abs_joint_delta_rad": float(
                            np.max(np.abs(delta))
                        ),
                        "l2_joint_delta_rad": float(np.linalg.norm(delta)),
                        "iterations": projection_history,
                        "final_exact_ground_lp_feasible": True,
                        "minimum_contact_free_torque_slack_nm": (
                            CONTACT_FREE_PROJECTION_MINIMUM_TORQUE_SLACK_NM
                        ),
                    },
                }
            )
            static_birth_evidence = _audit_composed_physical_birth(
                ready=ready,
                backend=backend,
                identity=identity,
                source={
                    "mode": physical_birth_composition["semantics"],
                    "teacher_motion_sha256": motion_sha,
                    "teacher_frame": 0,
                    "physical_birth_seed_sha256": physical_birth_seed_sha,
                    "seed_world_yaw_alignment": (
                        physical_birth_composition[
                            "seed_world_yaw_alignment"
                        ]
                    ),
                },
            )
    if not solution.feasible:
        # 人话:光说"没有解"没人能修。腰/臂/头这些关节地面根本使不上力,
        # 它们要多大力矩是唯一确定的,所以这里直接把"哪个关节、差多少 N·m、
        # 需要多大 q_des、卡的是电机还是 kp×行程"一并报出来。查不出来时才退回旧话。
        raise DynamicReadyMaterializationError(
            _static_hold_refusal_message(
                backend=backend,
                qpos=qpos,
                actuated=np.asarray(actuated, np.int64),
                model_row_for_runtime=model_row_for_runtime,
                plant=plant,
                ready_q=ready_q,
                executed_qdes_lower=executed_qdes_lower,
                executed_qdes_upper=executed_qdes_upper,
                hold_tau_lower_model=hold_tau_lower_model,
                hold_tau_upper_model=hold_tau_upper_model,
            )
        )
    contact_normals = np.asarray(
        solution.report.get("normal_force_per_contact_n", ()), np.float64
    )
    cop_interior_margins = np.asarray(
        solution.report.get("cop_interior_margin_per_foot_m", ()), np.float64
    )
    if selected_contact_normal_n > 0.0 and (
        contact_normals.ndim != 1
        or contact_normals.size < 6
        or not np.all(np.isfinite(contact_normals))
        or np.any(contact_normals < selected_contact_normal_n - 1.0e-7)
        or cop_interior_margins.shape != (2,)
        or not np.all(np.isfinite(cop_interior_margins))
        or np.any(cop_interior_margins <= 0.0)
    ):
        raise DynamicReadyMaterializationError(
            "selected hold lacks its required support-vertex/CoP margin"
        )
    fresh_tau_model = np.asarray(
        solution.actuator_generalized_force, np.float64
    )
    if fresh_tau_model.shape != (31,) or not np.all(
        np.isfinite(fresh_tau_model)
    ):
        raise DynamicReadyMaterializationError(
            "ground LP returned a malformed hold torque"
        )
    fresh_tau_runtime = fresh_tau_model[model_row_for_runtime]
    fresh_hold_qdes = ready_q + fresh_tau_runtime / plant["kp"]
    if full_seed_qdes_mode == FULL_SEED_QDES_SEED_TRANSPORT:
        source_vectors_match = (
            np.array_equal(seed_hold_transport["q"], ready_q)
            and np.array_equal(seed_hold_transport["kp"], plant["kp"])
            and np.array_equal(
                seed_hold_transport["default_q"], plant["default_q"]
            )
            and np.array_equal(
                seed_hold_transport["action_scale"], plant["action_scale"]
            )
            and np.array_equal(
                seed_hold_transport["executed_qdes_lower"],
                executed_qdes_lower,
            )
            and np.array_equal(
                seed_hold_transport["executed_qdes_upper"],
                executed_qdes_upper,
            )
        )
        if not source_vectors_match:
            raise DynamicReadyMaterializationError(
                "seed-transport hold does not match the current physical plant"
            )
        hold_qdes = np.asarray(seed_hold_transport["qdes"], np.float64)
        tau_runtime = plant["kp"] * (hold_qdes - ready_q)
        normalized_action = np.asarray(
            seed_hold_transport["normalized_action"], np.float64
        )
        if not np.allclose(
            tau_runtime,
            seed_hold_transport["tau_runtime"],
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise DynamicReadyMaterializationError(
                "seed-transport torque changed under the current plant"
            )
        tau_model = np.empty_like(fresh_tau_model)
        tau_model[model_row_for_runtime] = tau_runtime
    else:
        tau_model = fresh_tau_model
        tau_runtime = fresh_tau_runtime
        hold_qdes = fresh_hold_qdes
        normalized_action = (
            hold_qdes - plant["default_q"]
        ) / plant["action_scale"]
    tolerance = 1.0e-10
    if np.any(hold_qdes < executed_qdes_lower - tolerance) or np.any(
        hold_qdes > executed_qdes_upper + tolerance
    ):
        raise DynamicReadyMaterializationError(
            "derived hold qdes lies outside the executed qdes envelope"
        )
    if not np.all(np.isfinite(normalized_action)):
        raise DynamicReadyMaterializationError(
            "derived normalized hold action is non-finite"
        )
    if np.any(tau_model < hold_tau_lower_model - tolerance) or np.any(
        tau_model > hold_tau_upper_model + tolerance
    ):
        raise DynamicReadyMaterializationError(
            "selected hold torque lies outside the current exact envelope"
        )

    model_binding = solution.report.get("model_binding")
    if model_binding != identity.ground_model_binding_sha256:
        raise DynamicReadyMaterializationError(
            "ground LP result lost the exact model binding"
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "action_id": str(args.action_id),
        "robot": {
            "family": "AgiBot A3",
            "joint_names": list(plant["joint_names"]),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "ready_source": {
            "kind": source_kind,
            "frame_index": 0,
            "teacher_reference_unchanged": True,
            "teacher_and_physical_birth_same": (
                source_kind == STABLE_UPPER_SOURCE_KIND
                or (
                    isinstance(physical_birth_composition, Mapping)
                    and physical_birth_composition.get(
                        "teacher_and_physical_birth_differ"
                    )
                    is False
                )
            ),
            "physical_birth_semantics": (
                "motion_frame0"
                if source_kind == STABLE_UPPER_SOURCE_KIND
                else physical_birth_composition["semantics"]
            ),
            "plant_template_action_binding_consumed": (
                source_kind == STABLE_UPPER_SOURCE_KIND
            ),
            "plant_template_delay_overridden_to_zero": (
                source_kind == MEASURED_RETARGET_SOURCE_KIND
            ),
            "isaac_live_plant_match_required": True,
            "diagnostic_unauthorized": True,
            "training_authorized": False,
            "robust_contact_interior": {
                "enabled": selected_contact_normal_n > 0.0,
                "semantics": (
                    "every_exact_mujoco_support_vertex_has_positive_normal_force"
                ),
                "minimum_normal_force_per_support_vertex_n": (
                    selected_contact_normal_n
                ),
                "normal_force_per_support_vertex_n": contact_normals.tolist(),
                "cop_interior_margin_per_foot_m": (
                    cop_interior_margins.tolist()
                ),
            },
            **(
                {}
                if measured_evidence is None
                else {"measured_retarget_l0_evidence": measured_evidence}
            ),
        },
        "sources": {
            "stable_motion": {
                "path": str(motion_path),
                "sha256": motion_sha,
                "frame_index": 0,
            },
            **(
                {
                    "stable_receipt": {
                        "path": str(receipt_path),
                        "sha256": receipt_sha,
                    }
                }
                if source_kind == STABLE_UPPER_SOURCE_KIND
                else {
                    "measured_bank_receipt": {
                        "path": str(bank_receipt_path),
                        "sha256": bank_receipt_sha,
                    },
                    "measured_mechanical_audit": {
                        "path": str(mechanical_audit_path),
                        "sha256": mechanical_audit_sha,
                    },
                    **(
                        {
                            "physical_birth_optimizer_start": {
                                "path": str(physical_birth_seed_path),
                                "sha256": physical_birth_seed_sha,
                                "content_sha256": physical_birth_seed[
                                    "source_content_sha256"
                                ],
                                "source_action_id": physical_birth_seed[
                                    "source_action_id"
                                ],
                                "source_role": "optimizer_start_only",
                                "selected_state_inherited": False,
                                "inherited_model_identity": False,
                                "inherited_hold_claim": False,
                                "inherited_nominal_hold_claim": False,
                            }
                        }
                        if whole_body_optimizer_seed_requested
                        else (
                            {}
                            if seedless_frame0_birth
                            else {
                            "physical_birth_seed": {
                                "path": str(physical_birth_seed_path),
                                "sha256": physical_birth_seed_sha,
                                "content_sha256": physical_birth_seed[
                                    "source_content_sha256"
                                ],
                                "source_action_id": physical_birth_seed[
                                    "source_action_id"
                                ],
                                "source_role": "numerical_seed_only",
                                "consumed_fields": [
                                    "physical_ready.root_pos_w_m",
                                    "physical_ready.root_quat_wxyz",
                                    (
                                        "physical_ready.12_leg_joint_pos_rad"
                                        if measured_birth_mode
                                        == MEASURED_BIRTH_SHARED_LOWER_MODE
                                        else "physical_ready.31_joint_pos_rad"
                                    ),
                                ],
                                "inherited_model_identity": False,
                                "inherited_hold_claim": False,
                                "inherited_nominal_hold_claim": False,
                            }
                            }
                        )
                    ),
                }
            ),
            "runtime_training_contract": {
                "path": str(runtime_path),
                "sha256": runtime_sha,
            },
            "mujoco_model": {
                "path": str(mjcf_path),
                "sha256": mjcf_sha,
                "compiled_model_sha256": identity.compiled_model_sha256,
                "path_model_binding_sha256": (
                    identity.path_model_binding_sha256
                ),
                "ground_model_binding_sha256": (
                    identity.ground_model_binding_sha256
                ),
                "xml_model_name": identity.xml_model_name,
            },
        },
        "teacher_reference": {
            "semantics": (
                "exact_motion_bytes_frame0_reference"
                if source_kind == MEASURED_RETARGET_SOURCE_KIND
                else "exact_motion_bytes_frame0_reference_and_birth"
            ),
            "motion_sha256": motion_sha,
            "frame_index": 0,
            "root_pos_w_m": teacher_root_pos.tolist(),
            "root_quat_wxyz": teacher_root_quat.tolist(),
            "joint_pos_rad": teacher_q.tolist(),
            **(
                {
                    "static_handoff_joint_vel_radps": [0.0] * 31,
                    "static_handoff_velocity_semantics": (
                        "constructed_zero_joint_velocity_endpoint_not_"
                        "measured_motion_velocity"
                    ),
                }
                if whole_body_selected_hold
                else {}
            ),
        },
        **(
            {}
            if physical_birth_composition is None
            else {
                "physical_birth_composition": physical_birth_composition,
                "physical_birth_static_evidence": static_birth_evidence,
            }
        ),
        **(
            {
                "frame0_handoff": physical_birth_composition[
                    "frame0_handoff"
                ]
            }
            if whole_body_selected_hold
            else {}
        ),
        "physical_ready": {
            "root_pos_w_m": ready_root_pos.tolist(),
            "root_quat_wxyz": ready_root_quat.tolist(),
            "joint_pos_rad": ready_q.tolist(),
            "joint_vel_radps": [0.0] * 31,
        },
        "runtime_plant": {
            "joint_names": list(plant["joint_names"]),
            "articulation_joint_names": list(plant["joint_names"]),
            "action_joint_ids": list(range(31)),
            "joint_stiffness": plant["kp"].tolist(),
            "joint_damping": plant["kd"].tolist(),
            "joint_effort_limits": plant["effort"].tolist(),
            "joint_velocity_limits": plant["velocity"].tolist(),
            "joint_actuator_types": plant["actuator_types"],
            "joint_armature": plant["armature"].tolist(),
            "joint_friction_coefficients": plant["friction"].tolist(),
            "joint_friction_backend": plant["friction_backend"],
            "joint_friction_semantics": plant["friction_semantics"],
            "joint_friction_units": plant["friction_units"],
            "qdes_joint_pos_limits": qdes_limits.tolist(),
            "physx_control_position_limits": plant[
                "physx_control_position_limits"
            ],
            "finite_projection_soft_envelope_inset_fraction": inset,
            "projected_soft_qdes_lower_rad": projected_soft_lower.tolist(),
            "projected_soft_qdes_upper_rad": projected_soft_upper.tolist(),
            "hard_inner_qdes_lower_rad": hard_inner_lower.tolist(),
            "hard_inner_qdes_upper_rad": hard_inner_upper.tolist(),
            "executed_qdes_envelope_semantics": (
                "intersection(projected_soft_qdes,policy_bootstrap_hard_inner)"
            ),
            "executed_qdes_lower_rad": executed_qdes_lower.tolist(),
            "executed_qdes_upper_rad": executed_qdes_upper.tolist(),
            "default_joint_pos_rad": plant["default_q"].tolist(),
            "action_scale_rad": plant["action_scale"].tolist(),
            "physics_step_dt_s": plant["physics_dt"],
            "policy_step_dt_s": plant["policy_dt"],
            "control_decimation": plant["decimation"],
            "control_step_action_delay": plant[
                "control_step_action_delay"
            ],
        },
        "hold_candidate": {
            "semantics": (
                "tau_pd=kp*(qdes-physical_q) at zero joint velocity; "
                + (
                    (
                        "the new-backend cache-miss whole-body final-state LP "
                        "is the single selected witness"
                        if whole_body_selected_hold
                        else "current MuJoCo contact LP initializes the candidate"
                    )
                    if full_seed_qdes_mode == FULL_SEED_QDES_FRESH_STATIC_LP
                    else (
                        "content-pinned seed numerics are transported under "
                        "world-z yaw symmetry; current MuJoCo contact LP is "
                        "an unselected comparator"
                    )
                )
                + "; Isaac must validate it"
            ),
            "hold_qdes_mode": full_seed_qdes_mode,
            "selected_hold_authority": {
                "semantics": (
                    (
                        "fresh_new_backend_whole_body_final_state_0p1n_static_lp"
                        if whole_body_selected_hold
                        else "fresh_current_mjcf_static_lp"
                    )
                    if full_seed_qdes_mode == FULL_SEED_QDES_FRESH_STATIC_LP
                    else (
                        "world_z_yaw_symmetry_transport_of_content_pinned_"
                        "seed_numerics"
                    )
                ),
                "source_physical_birth_seed_sha256": (
                    None
                    if full_seed_qdes_mode
                    == FULL_SEED_QDES_FRESH_STATIC_LP
                    else physical_birth_seed_sha
                ),
                "inherited_hold_claim": False,
            },
            "lp_objective": LP_OBJECTIVE,
            "actuator_generalized_force_runtime_order_nm": (
                tau_runtime.tolist()
            ),
            "actuator_generalized_force_mujoco_row_order_nm": (
                tau_model.tolist()
            ),
            "hold_qdes_joint_pos_rad": hold_qdes.tolist(),
            "normalized_actor_action": normalized_action.tolist(),
            "mujoco_row_for_runtime_joint": model_row_for_runtime.tolist(),
            "mujoco_actuated_dof_indices": (
                np.asarray(actuated, np.int64).tolist()
            ),
            "model_tau_lower_mujoco_row_order_nm": model_tau_lower.tolist(),
            "model_tau_upper_mujoco_row_order_nm": model_tau_upper.tolist(),
            "runtime_tau_lower_runtime_order_nm": runtime_tau_lower.tolist(),
            "runtime_tau_upper_runtime_order_nm": runtime_tau_upper.tolist(),
            "runtime_tau_lower_mujoco_row_order_nm": (
                runtime_tau_lower_model.tolist()
            ),
            "runtime_tau_upper_mujoco_row_order_nm": (
                runtime_tau_upper_model.tolist()
            ),
            "effective_tau_lower_mujoco_row_order_nm": (
                hold_tau_lower_model.tolist()
            ),
            "effective_tau_upper_mujoco_row_order_nm": (
                hold_tau_upper_model.tolist()
            ),
            "actuator_limit_contract": actuator_limit_report,
            "solver_report": solution.report,
            "solver_report_role": (
                (
                    "selected_whole_body_final_state_single_witness"
                    if whole_body_selected_hold
                    else "selected_hold_solution"
                )
                if full_seed_qdes_mode
                == FULL_SEED_QDES_FRESH_STATIC_LP
                else "fresh_current_mjcf_comparator_not_selected"
            ),
            **(
                {}
                if full_seed_qdes_mode
                == FULL_SEED_QDES_FRESH_STATIC_LP
                else {
                    "fresh_static_lp_comparator": {
                        "actuator_generalized_force_runtime_order_nm": (
                            fresh_tau_runtime.tolist()
                        ),
                        "actuator_generalized_force_mujoco_row_order_nm": (
                            fresh_tau_model.tolist()
                        ),
                        "hold_qdes_joint_pos_rad": fresh_hold_qdes.tolist(),
                    }
                }
            ),
        },
        "required_next_gate": {
            "kind": "isaac_action_ball_nominal_hold_v1",
            "required_policy_steps": (
                60 if whole_body_selected_hold else None
            ),
            "required_physics_steps": (
                240 if whole_body_selected_hold else None
            ),
            "required_min_wait_s": (
                physical_birth_composition["frame0_handoff"][
                    "required_min_wait_s"
                ]
                if whole_body_selected_hold
                else None
            ),
            "minimum_horizon_semantics": "validated_t_hit_plus_reaction_margin",
            "zero_terminal_required": [
                "joint_qdes_forbidden",
                "joint_actual_forbidden",
                "robot_hit_table",
                "base_fell_tilt",
                "base_too_low",
            ],
        },
        "non_claims": [
            "not an Isaac or PhysX closed-loop hold certificate",
            "not a training policy bootstrap until the nominal hold gate passes",
            "not deployment or hardware authorization",
            "measured teacher validation does not claim mechanical admission",
            (
                "no historical numerical physical-birth seed is consumed"
                if seedless_frame0_birth
                else "historical numerical seed model/hold claims are not inherited"
            ),
        ],
        "producer": {
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
            "grounded_ready_tool_path": str(Path(grounded.__file__).resolve()),
            "grounded_ready_tool_sha256": _sha256_file(
                Path(grounded.__file__).resolve()
            ),
            "whole_body_safe_ready_tool_path": str(
                Path(whole_body_ready.__file__).resolve()
            ),
            "whole_body_safe_ready_tool_sha256": _sha256_file(
                Path(whole_body_ready.__file__).resolve()
            ),
            "mujoco_path_adapter_tool_path": str(
                Path(path_adapter.__file__).resolve()
            ),
            "mujoco_path_adapter_tool_sha256": _sha256_file(
                Path(path_adapter.__file__).resolve()
            ),
            "torque_lp_tool_path": str(Path(torque_topp.__file__).resolve()),
            "torque_lp_tool_sha256": _sha256_file(
                Path(torque_topp.__file__).resolve()
            ),
        },
    }
    unsigned = dict(result)
    content_sha = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    result["content_sha256"] = content_sha
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-id", required=True)
    parser.add_argument(
        "--ready-source-kind",
        choices=(STABLE_UPPER_SOURCE_KIND, MEASURED_RETARGET_SOURCE_KIND),
        default=STABLE_UPPER_SOURCE_KIND,
        help=(
            "stable_upper_v2 preserves the historical path; "
            "measured_retarget_l0_diagnostic separates the exact teacher frame0 "
            "from a content-pinned composed physical birth"
        ),
    )
    parser.add_argument("--motion", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--stable-receipt")
    parser.add_argument("--expected-stable-receipt-sha256")
    parser.add_argument("--measured-uid")
    parser.add_argument("--measured-bank-receipt")
    parser.add_argument("--expected-measured-bank-receipt-sha256")
    parser.add_argument("--mechanical-audit")
    parser.add_argument("--expected-mechanical-audit-sha256")
    parser.add_argument("--physical-birth-seed")
    parser.add_argument("--expected-physical-birth-seed-sha256")
    parser.add_argument(
        "--physical-birth-composition-mode",
        choices=(
            MEASURED_BIRTH_SHARED_LOWER_MODE,
            MEASURED_BIRTH_FULL_SEED_MODE,
            MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE,
            MEASURED_BIRTH_DIRECT_FRAME0_MODE,
            MEASURED_BIRTH_PROJECTED_FRAME0_MODE,
            MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE,
        ),
        default=MEASURED_BIRTH_SHARED_LOWER_MODE,
        help=(
            "measured branch only: overlay teacher non-leg joints on the seed, "
            "or preserve the full high-margin seed as physical birth while the "
            "measured motion remains the exact teacher, or project only named "
            "contact-free PD-travel hold shortfalls from that seed, or use exact teacher "
            "frame0 root/joints directly without consuming a historical seed, "
            "or ground frame0 by projecting only leg12 while preserving root, "
            "non-leg joints, and exact racket-site FK, or run the measured-"
            "conditioned lexical whole-body search that releases root z/roll/"
            "pitch and all 31 joints while locking safety before fidelity"
        ),
    )
    parser.add_argument(
        "--full-seed-minimum-normal-force-per-support-vertex-n",
        type=float,
        default=0.0,
        help=(
            "full-seed measured diagnostic only: require this positive normal "
            "force at every exact MuJoCo support vertex; zero preserves the "
            "legacy edge-CoP LP"
        ),
    )
    parser.add_argument(
        "--full-seed-hold-qdes-mode",
        choices=(
            FULL_SEED_QDES_FRESH_STATIC_LP,
            FULL_SEED_QDES_SEED_TRANSPORT,
        ),
        default=FULL_SEED_QDES_FRESH_STATIC_LP,
        help=(
            "full-seed measured diagnostic only: use the fresh current-MJCF "
            "static LP qdes, or transport exact content-pinned seed qdes/action/"
            "tau numerics under the applied world-z yaw symmetry; seed_transport "
            "does not inherit the historical hold claim"
        ),
    )
    parser.add_argument(
        "--allow-mechanical-unknown-diagnostic",
        action="store_true",
        help=(
            "allow a kinematically PASS but mechanically UNKNOWN measured clip "
            "for simulation-only diagnostic materialization"
        ),
    )
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--expected-mjcf-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _materialize(args)
    output = _write_exclusive(args.output, _pretty_json_bytes(result))
    print(
        json.dumps(
            {
                "output": str(output),
                "content_sha256": result["content_sha256"],
                "action_id": result["action_id"],
                "objective": LP_OBJECTIVE,
                "max_hold_utilization": result["hold_candidate"][
                "solver_report"
                ].get("optimum_max_normalized_available_hold_torque"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
