#!/usr/bin/env python3
"""Diagnostic A3 MuJoCo single-environment plant/action runner.

The module closes one narrow dependency that does not require the final
ActionBall N1 ABI: a schema-3 A3 plant contract, the five-solid table/net
scene, a 31-D normalized action, one episode-fixed whole-row delay, bounded
total-PD execution, and a byte-bound 100-control-tick tape.

It intentionally has no ball, reward, observation builder, policy or PPO.  A
successful receipt therefore proves only deterministic plant/action plumbing.
Every generated artifact says ``diagnostic_unauthorized=true`` and explicitly
denies training, promotion, deployment and hardware authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ACTION_DIM = 31
FIXED_TAPE_TICKS = 100
TAPE_KIND = "a3_mujoco_single_env_fixed_action_tape_v1"
RECEIPT_KIND = "a3_mujoco_single_env_fixed_tape_receipt_v1"
TRACE_KIND = "a3_mujoco_single_env_fixed_tape_trace_v1"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
TABLE_SCENE_PY = REPO_ROOT / "scripts/mujoco_table_scene.py"
JOINT_ORDER_CONTRACT = REPO_ROOT / "configs/a3_joint_order_bijection_v1.json"
JOINT_ORDER_CONTRACT_ID = "a3-gmr-dof-pos-to-runtime-articulation-v1"


class ContractError(RuntimeError):
    """The portable plant/action contract is missing, ambiguous or inconsistent."""


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"duplicate JSON key is forbidden: {key}")
        out[key] = value
    return out


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"payload is not finite canonical JSON: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value, raw


def _finite_vector(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != (ACTION_DIM,) or not np.isfinite(out).all():
        raise ContractError(f"{name} must be {ACTION_DIM} finite scalars")
    if positive and not np.all(out > 0.0):
        raise ContractError(f"{name} must be strictly positive")
    if nonnegative and not np.all(out >= 0.0):
        raise ContractError(f"{name} must be non-negative")
    out.setflags(write=False)
    return out


def _finite_limit_pairs(value: Any, name: str) -> np.ndarray:
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != (ACTION_DIM, 2) or not np.isfinite(out).all():
        raise ContractError(f"{name} must have shape ({ACTION_DIM}, 2) and be finite")
    if not np.all(out[:, 1] > out[:, 0]):
        raise ContractError(f"every {name} upper bound must exceed its lower bound")
    out.setflags(write=False)
    return out


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} cannot be bool")
    out = float(value)
    if not math.isfinite(out) or (positive and out <= 0.0):
        raise ContractError(f"{name} must be {'positive ' if positive else ''}finite")
    return out


def _int_scalar(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class PlantBinding:
    """The exact schema-3 fields consumed by the single-env runner."""

    source_path: str
    source_sha256: str
    binding_sha256: str
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    armature: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray
    qdes_limits: np.ndarray
    executed_qdes_limits: np.ndarray
    finite_projection_inset_fraction: float
    actuator_types: tuple[str, ...]
    physics_step_dt_s: float
    policy_step_dt_s: float
    control_decimation: int
    delay_min_steps: int
    delay_max_steps: int

    @staticmethod
    def from_mapping(
        payload: Mapping[str, Any], *, source_path: str, source_sha256: str
    ) -> "PlantBinding":
        if payload.get("schema_version") != 3:
            raise ContractError("training contract schema_version must equal 3")
        names_raw = payload.get("joint_names")
        if not isinstance(names_raw, list) or len(names_raw) != ACTION_DIM:
            raise ContractError(f"joint_names must contain exactly {ACTION_DIM} names")
        names = tuple(str(value) for value in names_raw)
        if any(not value for value in names) or len(set(names)) != ACTION_DIM:
            raise ContractError("joint_names must be non-empty and unique")
        articulation = tuple(str(value) for value in payload.get("articulation_joint_names", ()))
        if articulation != names:
            raise ContractError("action and articulation joint order must be identical")
        if payload.get("action_use_default_offset") is not True:
            raise ContractError("runner requires action_use_default_offset=true")
        if payload.get("qdes_clamp") is not True:
            raise ContractError("runner requires the schema-3 qdes clamp")

        default_q = _finite_vector(payload.get("default_joint_pos"), "default_joint_pos")
        scale = _finite_vector(payload.get("action_scale"), "action_scale", positive=True)
        kp = _finite_vector(payload.get("joint_stiffness"), "joint_stiffness", positive=True)
        kd = _finite_vector(payload.get("joint_damping"), "joint_damping", nonnegative=True)
        armature = _finite_vector(payload.get("joint_armature"), "joint_armature", nonnegative=True)
        effort = _finite_vector(
            payload.get("joint_effort_limits"), "joint_effort_limits", positive=True
        )
        velocity = _finite_vector(
            payload.get("joint_velocity_limits"), "joint_velocity_limits", positive=True
        )
        qdes = _finite_limit_pairs(payload.get("qdes_joint_pos_limits"), "qdes_joint_pos_limits")
        if np.any(default_q < qdes[:, 0]) or np.any(default_q > qdes[:, 1]):
            raise ContractError("default_joint_pos falls outside qdes_joint_pos_limits")
        if payload.get("finite_preclamp_qdes_projection_enabled") is not True:
            raise ContractError(
                "runner requires finite_preclamp_qdes_projection_enabled=true"
            )
        projection_inset = _finite_scalar(
            payload.get("finite_projection_soft_envelope_inset_fraction"),
            "finite_projection_soft_envelope_inset_fraction",
        )
        if not 0.0 <= projection_inset < 0.5:
            raise ContractError(
                "finite projection inset fraction must be in [0, 0.5)"
            )
        initialization = payload.get("action_ball_ppo_runner_recipe")
        if not isinstance(initialization, dict):
            raise ContractError("action_ball_ppo_runner_recipe must be an object")
        recipe = initialization.get("recipe")
        policy_initialization = (
            recipe.get("policy_initialization") if isinstance(recipe, dict) else None
        )
        hard_guard = (
            policy_initialization.get("hard_inner_guard")
            if isinstance(policy_initialization, dict)
            else None
        )
        if not isinstance(hard_guard, dict):
            raise ContractError("policy_initialization.hard_inner_guard is missing")
        hard_inner_lower = _finite_vector(
            hard_guard.get("hard_inner_lower"), "hard_inner_lower"
        )
        hard_inner_upper = _finite_vector(
            hard_guard.get("hard_inner_upper"), "hard_inner_upper"
        )
        if not np.all(hard_inner_upper > hard_inner_lower):
            raise ContractError("hard-inner qdes envelope is empty")
        soft_span = qdes[:, 1] - qdes[:, 0]
        executed_lower = np.maximum(
            qdes[:, 0] + projection_inset * soft_span, hard_inner_lower
        )
        executed_upper = np.minimum(
            qdes[:, 1] - projection_inset * soft_span, hard_inner_upper
        )
        if not np.all(executed_upper > executed_lower):
            raise ContractError("projected-soft and hard-inner qdes envelopes do not intersect")
        executed_qdes = np.stack((executed_lower, executed_upper), axis=1)
        executed_qdes.setflags(write=False)

        actuator_types_raw = payload.get("joint_actuator_types")
        if not isinstance(actuator_types_raw, list) or len(actuator_types_raw) != ACTION_DIM:
            raise ContractError("joint_actuator_types must contain 31 rows")
        actuator_types = tuple(str(value) for value in actuator_types_raw)
        if any(value != "implicit" for value in actuator_types):
            raise ContractError(
                "diagnostic total-PD core currently requires all 31 actuator types to be implicit"
            )

        physics_dt = _finite_scalar(
            payload.get("physics_step_dt_s"), "physics_step_dt_s", positive=True
        )
        policy_dt = _finite_scalar(
            payload.get("policy_step_dt_s"), "policy_step_dt_s", positive=True
        )
        decimation = _int_scalar(payload.get("control_decimation"), "control_decimation", minimum=1)
        if not math.isclose(
            policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ContractError("policy_step_dt_s != physics_step_dt_s * control_decimation")

        delay = payload.get("control_step_action_delay")
        if not isinstance(delay, dict):
            raise ContractError("control_step_action_delay must be an object")
        expected_delay = {
            "distribution": "discrete_uniform_inclusive",
            "sample_timing": "once_per_episode_reset",
            "semantic_unit": "policy_control_step",
            "shared_across_all_31_joints": True,
        }
        for key, expected in expected_delay.items():
            if delay.get(key) != expected:
                raise ContractError(
                    f"control_step_action_delay.{key} must equal {expected!r}"
                )
        delay_min = _int_scalar(delay.get("min_steps"), "delay.min_steps")
        delay_max = _int_scalar(delay.get("max_steps"), "delay.max_steps")
        if delay_max < delay_min:
            raise ContractError("delay max_steps cannot be smaller than min_steps")

        selected = {
            "schema_version": 1,
            "kind": "a3_mujoco_single_env_plant_binding_v1",
            "source_training_contract_sha256": source_sha256,
            "joint_names": list(names),
            "default_joint_pos": default_q.tolist(),
            "action_scale": scale.tolist(),
            "joint_stiffness": kp.tolist(),
            "joint_damping": kd.tolist(),
            "joint_armature": armature.tolist(),
            "joint_effort_limits": effort.tolist(),
            "joint_velocity_limits": velocity.tolist(),
            "qdes_joint_pos_limits": qdes.tolist(),
            "executed_qdes_limits": executed_qdes.tolist(),
            "finite_projection_soft_envelope_inset_fraction": projection_inset,
            "joint_actuator_types": list(actuator_types),
            "physics_step_dt_s": physics_dt,
            "policy_step_dt_s": policy_dt,
            "control_decimation": decimation,
            "delay_min_steps": delay_min,
            "delay_max_steps": delay_max,
            "action_semantics": (
                "qdes=project(default_joint_pos+action_scale*delayed_action,"
                "intersection(soft_5pct_inset,hard_inner_guard))"
            ),
            "effort_semantics": "tau=clip(kp*(qdes-q)-kd*qdot,+/-effort_limit)",
            "physx_dimensionless_joint_friction_in_mujoco": "not_applied_no_unit_conversion",
        }
        binding_sha = _sha256(_canonical_json_bytes(selected))
        return PlantBinding(
            source_path=source_path,
            source_sha256=source_sha256,
            binding_sha256=binding_sha,
            joint_names=names,
            default_joint_pos=default_q,
            action_scale=scale,
            stiffness=kp,
            damping=kd,
            armature=armature,
            effort_limits=effort,
            velocity_limits=velocity,
            qdes_limits=qdes,
            executed_qdes_limits=executed_qdes,
            finite_projection_inset_fraction=projection_inset,
            actuator_types=actuator_types,
            physics_step_dt_s=physics_dt,
            policy_step_dt_s=policy_dt,
            control_decimation=decimation,
            delay_min_steps=delay_min,
            delay_max_steps=delay_max,
        )

    def decode_action(self, delayed_action: Sequence[float]) -> tuple[np.ndarray, np.ndarray, int]:
        action = _finite_vector(delayed_action, "delayed_action")
        raw = self.default_joint_pos + self.action_scale * action
        applied = np.clip(
            raw,
            self.executed_qdes_limits[:, 0],
            self.executed_qdes_limits[:, 1],
        )
        clamp_count = int(np.count_nonzero(applied != raw))
        return raw, applied, clamp_count


def load_plant_binding(path: Path | str) -> PlantBinding:
    source = Path(path).expanduser().resolve()
    payload, raw = _load_strict_json(source)
    return PlantBinding.from_mapping(
        payload,
        source_path=str(source),
        source_sha256=_sha256(raw),
    )


def total_pd_effort(
    binding: PlantBinding,
    qdes: Sequence[float],
    q: Sequence[float],
    qd: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Independent-shaped oracle for the exact bounded total-PD operation."""

    qdes_vec = _finite_vector(qdes, "qdes")
    q_vec = _finite_vector(q, "q")
    qd_vec = _finite_vector(qd, "qd")
    raw = binding.stiffness * (qdes_vec - q_vec) - binding.damping * qd_vec
    applied = np.clip(raw, -binding.effort_limits, binding.effort_limits)
    return raw, applied, int(np.count_nonzero(applied != raw))


class ActionDelayLine:
    """Episode-fixed delay for complete 31-D normalized action rows."""

    def __init__(self, maximum_delay_steps: int):
        self.maximum_delay_steps = _int_scalar(
            maximum_delay_steps, "maximum_delay_steps"
        )
        self.delay_steps: int | None = None
        self._rows: list[np.ndarray] = []

    def reset(self, delay_steps: int, history_fill_action: Sequence[float]) -> None:
        delay = _int_scalar(delay_steps, "delay_steps")
        if delay > self.maximum_delay_steps:
            raise ContractError(
                f"delay_steps={delay} exceeds maximum={self.maximum_delay_steps}"
            )
        fill = _finite_vector(history_fill_action, "history_fill_action")
        self.delay_steps = delay
        self._rows = [fill.copy() for _ in range(self.maximum_delay_steps + 1)]

    def push(self, actor_action: Sequence[float]) -> np.ndarray:
        if self.delay_steps is None or not self._rows:
            raise ContractError("delay line must be reset before push")
        action = _finite_vector(actor_action, "actor_action")
        self._rows.append(action.copy())
        self._rows = self._rows[-(self.maximum_delay_steps + 1) :]
        return self._rows[-1 - self.delay_steps].copy()

    def state(self) -> np.ndarray:
        if self.delay_steps is None or not self._rows:
            raise ContractError("delay line has no episode state")
        return np.stack(self._rows, axis=0)


@dataclass(frozen=True)
class ResetState:
    mode: str
    key_name: str | None
    joint_pos: np.ndarray | None
    joint_vel: np.ndarray | None
    root_pos: np.ndarray | None
    root_quat_wxyz: np.ndarray | None
    root_lin_vel_w: np.ndarray | None
    root_ang_vel_w: np.ndarray | None
    root_lin_vel_point: str | None
    source_motion_path: str | None
    source_motion_sha256: str | None
    source_frame_index: int | None
    source_motion_uid: str | None
    source_joint_order_contract_id: str | None
    source_joint_order_contract_sha256: str | None

    @staticmethod
    def from_mapping(value: Any) -> "ResetState":
        if not isinstance(value, dict):
            raise ContractError("reset_state must be an object")
        mode = value.get("mode")
        if mode == "named_stand_root_executed_zero_action_q_zero_velocity":
            if set(value) != {"mode", "key_name"} or value.get("key_name") != "stand":
                raise ContractError("named-stand reset_state must bind only key_name='stand'")
            return ResetState(
                mode=mode,
                key_name="stand",
                joint_pos=None,
                joint_vel=None,
                root_pos=None,
                root_quat_wxyz=None,
                root_lin_vel_w=None,
                root_ang_vel_w=None,
                root_lin_vel_point=None,
                source_motion_path=None,
                source_motion_sha256=None,
                source_frame_index=None,
                source_motion_uid=None,
                source_joint_order_contract_id=None,
                source_joint_order_contract_sha256=None,
            )
        if mode != "teacher_frame":
            raise ContractError(f"unsupported reset_state mode: {mode!r}")
        expected = {
            "joint_pos",
            "joint_vel",
            "mode",
            "root_ang_vel_w",
            "root_lin_vel_point",
            "root_lin_vel_w",
            "root_pos",
            "root_quat_wxyz",
            "source_frame_index",
            "source_motion_path",
            "source_motion_sha256",
            "source_motion_uid",
            "source_joint_order_contract_id",
            "source_joint_order_contract_sha256",
        }
        if set(value) != expected:
            raise ContractError(
                f"teacher reset keys differ: missing={sorted(expected-set(value))}, "
                f"unknown={sorted(set(value)-expected)}"
            )
        source_sha = str(value.get("source_motion_sha256", ""))
        if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
            raise ContractError("teacher reset source_motion_sha256 is invalid")
        source_path = str(value.get("source_motion_path", ""))
        source_uid = str(value.get("source_motion_uid", ""))
        if not source_path or not source_uid:
            raise ContractError("teacher reset source path/UID cannot be empty")
        order_id = str(value.get("source_joint_order_contract_id", ""))
        order_sha = str(value.get("source_joint_order_contract_sha256", ""))
        if order_id != JOINT_ORDER_CONTRACT_ID:
            raise ContractError(f"unsupported teacher joint-order contract: {order_id!r}")
        if len(order_sha) != 64 or any(
            ch not in "0123456789abcdef" for ch in order_sha
        ):
            raise ContractError("teacher joint-order contract SHA-256 is invalid")
        frame = _int_scalar(value.get("source_frame_index"), "source_frame_index")
        root_point = value.get("root_lin_vel_point")
        if root_point not in ("center_of_mass", "link_origin"):
            raise ContractError("root_lin_vel_point must be center_of_mass|link_origin")

        def vector(raw: Any, name: str, size: int) -> np.ndarray:
            out = np.array(raw, dtype=np.float64, copy=True)
            if out.shape != (size,) or not np.isfinite(out).all():
                raise ContractError(f"{name} must contain {size} finite scalars")
            out.setflags(write=False)
            return out

        root_quat = vector(value.get("root_quat_wxyz"), "root_quat_wxyz", 4)
        quat_norm = float(np.linalg.norm(root_quat))
        if not math.isclose(quat_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
            raise ContractError("teacher reset root quaternion must be unit length")
        return ResetState(
            mode=mode,
            key_name=None,
            joint_pos=_finite_vector(value.get("joint_pos"), "reset joint_pos"),
            joint_vel=_finite_vector(value.get("joint_vel"), "reset joint_vel"),
            root_pos=vector(value.get("root_pos"), "root_pos", 3),
            root_quat_wxyz=root_quat,
            root_lin_vel_w=vector(value.get("root_lin_vel_w"), "root_lin_vel_w", 3),
            root_ang_vel_w=vector(value.get("root_ang_vel_w"), "root_ang_vel_w", 3),
            root_lin_vel_point=str(root_point),
            source_motion_path=source_path,
            source_motion_sha256=source_sha,
            source_frame_index=frame,
            source_motion_uid=source_uid,
            source_joint_order_contract_id=order_id,
            source_joint_order_contract_sha256=order_sha,
        )


@dataclass(frozen=True)
class FixedTape:
    source_path: str
    source_sha256: str
    joint_names: tuple[str, ...]
    plant_binding_sha256: str
    delay_steps: int
    history_fill_action: np.ndarray
    actions: np.ndarray
    reset_state: ResetState

    @staticmethod
    def from_mapping(
        payload: Mapping[str, Any], *, source_path: str, source_sha256: str,
        binding: PlantBinding
    ) -> "FixedTape":
        expected_keys = {
            "actions",
            "delay_steps",
            "history_fill_action",
            "joint_names",
            "kind",
            "plant_binding_sha256",
            "policy_hz",
            "reset_state",
            "schema_version",
            "ticks",
        }
        if set(payload) != expected_keys:
            raise ContractError(
                f"fixed tape keys differ: missing={sorted(expected_keys-set(payload))}, "
                f"unknown={sorted(set(payload)-expected_keys)}"
            )
        if payload.get("schema_version") != 1 or payload.get("kind") != TAPE_KIND:
            raise ContractError("fixed tape kind/schema mismatch")
        if payload.get("ticks") != FIXED_TAPE_TICKS:
            raise ContractError(f"fixed tape must contain exactly {FIXED_TAPE_TICKS} ticks")
        policy_hz = _finite_scalar(payload.get("policy_hz"), "policy_hz", positive=True)
        if not math.isclose(
            policy_hz, 1.0 / binding.policy_step_dt_s, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ContractError("fixed tape policy_hz disagrees with plant binding")
        names = tuple(str(value) for value in payload.get("joint_names", ()))
        if names != binding.joint_names:
            raise ContractError("fixed tape joint order disagrees with plant binding")
        if payload.get("plant_binding_sha256") != binding.binding_sha256:
            raise ContractError("fixed tape belongs to a different plant binding")
        delay = _int_scalar(payload.get("delay_steps"), "delay_steps")
        if delay < binding.delay_min_steps or delay > binding.delay_max_steps:
            raise ContractError("fixed tape delay is outside the training contract support")
        fill = _finite_vector(payload.get("history_fill_action"), "history_fill_action")
        try:
            actions = np.asarray(payload.get("actions"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"actions must have shape ({FIXED_TAPE_TICKS}, {ACTION_DIM}) and be finite"
            ) from exc
        if actions.shape != (FIXED_TAPE_TICKS, ACTION_DIM) or not np.isfinite(actions).all():
            raise ContractError(
                f"actions must have shape ({FIXED_TAPE_TICKS}, {ACTION_DIM}) and be finite"
            )
        actions.setflags(write=False)
        reset_state = ResetState.from_mapping(payload.get("reset_state"))
        return FixedTape(
            source_path=source_path,
            source_sha256=source_sha256,
            joint_names=names,
            plant_binding_sha256=binding.binding_sha256,
            delay_steps=delay,
            history_fill_action=fill,
            actions=actions,
            reset_state=reset_state,
        )


def load_fixed_tape(path: Path | str, binding: PlantBinding) -> FixedTape:
    source = Path(path).expanduser().resolve()
    payload, raw = _load_strict_json(source)
    tape = FixedTape.from_mapping(
        payload,
        source_path=str(source),
        source_sha256=_sha256(raw),
        binding=binding,
    )
    _validate_fixed_tape_reset_contract(tape, binding)
    return tape


def _teacher_frame_reset_payload(
    binding: PlantBinding, motion_path: Path | str, frame_index: int
) -> tuple[dict[str, Any], np.ndarray]:
    path = Path(motion_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read teacher motion {path}: {exc}") from exc
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as motion:
            required = {
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "body_names",
                "body_pos_point",
                "body_lin_vel_point",
                "measured_racket_uid",
                "measured_racket_joint_order_contract_id",
                "measured_racket_joint_order_contract_sha256",
            }
            missing = sorted(required - set(motion.files))
            if missing:
                raise ContractError(f"teacher motion is missing keys: {missing}")
            if str(np.asarray(motion["body_pos_point"]).item()) != "link_origin":
                raise ContractError("teacher root position must use link_origin semantics")
            root_lin_point = str(np.asarray(motion["body_lin_vel_point"]).item())
            if root_lin_point not in ("center_of_mass", "link_origin"):
                raise ContractError("teacher root linear-velocity point is unsupported")
            order_id = str(
                np.asarray(motion["measured_racket_joint_order_contract_id"]).item()
            )
            if order_id != JOINT_ORDER_CONTRACT_ID:
                raise ContractError(f"unsupported teacher joint-order contract: {order_id!r}")
            order_sha = str(
                np.asarray(motion["measured_racket_joint_order_contract_sha256"]).item()
            )
            try:
                expected_order_sha = _sha256(JOINT_ORDER_CONTRACT.read_bytes())
            except OSError as exc:
                raise ContractError(
                    f"cannot read joint-order contract {JOINT_ORDER_CONTRACT}: {exc}"
                ) from exc
            if order_sha != expected_order_sha:
                raise ContractError(
                    "teacher joint-order contract SHA-256 disagrees with the local "
                    f"contract: motion={order_sha!r}, local={expected_order_sha}"
                )
            body_names = tuple(str(value) for value in np.asarray(motion["body_names"]).tolist())
            if body_names.count("pelvis_link") != 1:
                raise ContractError("teacher motion must contain one pelvis_link body row")
            pelvis_index = body_names.index("pelvis_link")
            joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
            joint_vel = np.asarray(motion["joint_vel"], dtype=np.float64)
            body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
            body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
            body_lin = np.asarray(motion["body_lin_vel_w"], dtype=np.float64)
            body_ang = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)
            frame = _int_scalar(frame_index, "teacher frame_index")
            if (
                joint_pos.ndim != 2
                or joint_pos.shape[1] != ACTION_DIM
                or joint_vel.shape != joint_pos.shape
                or frame >= joint_pos.shape[0]
                or body_pos.shape[:2] != body_quat.shape[:2]
                or body_pos.shape[:2] != body_lin.shape[:2]
                or body_pos.shape[:2] != body_ang.shape[:2]
                or body_pos.shape[0] != joint_pos.shape[0]
                or body_pos.shape[1] != len(body_names)
            ):
                raise ContractError("teacher motion has inconsistent time/body/joint shapes")
            rows = (
                joint_pos[frame],
                joint_vel[frame],
                body_pos[frame, pelvis_index],
                body_quat[frame, pelvis_index],
                body_lin[frame, pelvis_index],
                body_ang[frame, pelvis_index],
            )
            if any(not np.isfinite(row).all() for row in rows):
                raise ContractError("teacher reset frame contains non-finite values")
            uid = str(np.asarray(motion["measured_racket_uid"]).item())
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot load teacher motion {path}: {exc}") from exc

    center_action = (joint_pos[frame] - binding.default_joint_pos) / binding.action_scale
    _raw_qdes, executed_qdes, _clamps = binding.decode_action(center_action)
    if not np.allclose(executed_qdes, joint_pos[frame], rtol=0.0, atol=2.0e-6):
        error = float(np.max(np.abs(executed_qdes - joint_pos[frame])))
        raise ContractError(
            f"teacher q0 is outside the executed qdes envelope: max_abs={error:.6g}"
        )
    payload = {
        "mode": "teacher_frame",
        "source_motion_path": str(path),
        "source_motion_sha256": _sha256(raw),
        "source_motion_uid": uid,
        "source_joint_order_contract_id": order_id,
        "source_joint_order_contract_sha256": order_sha,
        "source_frame_index": frame,
        "joint_pos": joint_pos[frame].tolist(),
        "joint_vel": joint_vel[frame].tolist(),
        "root_pos": body_pos[frame, pelvis_index].tolist(),
        "root_quat_wxyz": body_quat[frame, pelvis_index].tolist(),
        "root_lin_vel_w": body_lin[frame, pelvis_index].tolist(),
        "root_ang_vel_w": body_ang[frame, pelvis_index].tolist(),
        "root_lin_vel_point": root_lin_point,
    }
    return payload, center_action


def _validate_fixed_tape_reset_contract(
    tape: FixedTape, binding: PlantBinding
) -> None:
    """Fail closed when delayed history could pull the plant off its reset state."""

    _raw_qdes, fill_qdes, fill_clamps = binding.decode_action(
        tape.history_fill_action
    )
    if tape.reset_state.mode == "named_stand_root_executed_zero_action_q_zero_velocity":
        _unused_raw, zero_action_qdes, _zero_action_clamps = binding.decode_action(
            np.zeros(ACTION_DIM)
        )
        if (
            not np.array_equal(tape.history_fill_action, np.zeros(ACTION_DIM))
            or not np.allclose(
                fill_qdes, zero_action_qdes, rtol=0.0, atol=1.0e-12
            )
        ):
            raise ContractError(
                "named-stand tape history_fill_action must be exact zero/executed-qdes"
            )
        return

    reset = tape.reset_state
    assert reset.source_motion_path is not None
    assert reset.source_frame_index is not None
    expected_mapping, expected_center = _teacher_frame_reset_payload(
        binding, reset.source_motion_path, reset.source_frame_index
    )
    expected = ResetState.from_mapping(expected_mapping)
    scalar_fields = (
        "source_motion_path",
        "source_motion_sha256",
        "source_motion_uid",
        "source_frame_index",
        "source_joint_order_contract_id",
        "source_joint_order_contract_sha256",
        "root_lin_vel_point",
    )
    for field in scalar_fields:
        if getattr(reset, field) != getattr(expected, field):
            raise ContractError(f"teacher reset lineage disagrees with source: {field}")
    vector_fields = (
        "joint_pos",
        "joint_vel",
        "root_pos",
        "root_quat_wxyz",
        "root_lin_vel_w",
        "root_ang_vel_w",
    )
    for field in vector_fields:
        if not np.array_equal(getattr(reset, field), getattr(expected, field)):
            raise ContractError(f"teacher reset state disagrees with source: {field}")
    if fill_clamps != 0 or not np.allclose(
        fill_qdes, reset.joint_pos, rtol=0.0, atol=2.0e-6
    ):
        raise ContractError(
            "teacher history_fill_action does not decode exactly to reset joint_pos"
        )
    if not np.allclose(
        tape.history_fill_action, expected_center, rtol=0.0, atol=2.0e-12
    ):
        raise ContractError(
            "teacher history_fill_action differs from the source-derived action center"
        )


def build_probe_tape(
    binding: PlantBinding,
    *,
    delay_steps: int,
    teacher_motion: Path | str | None = None,
    teacher_frame_index: int = 0,
) -> dict[str, Any]:
    """Build a deterministic low-amplitude tape that excites every action column."""

    delay = _int_scalar(delay_steps, "delay_steps")
    if delay < binding.delay_min_steps or delay > binding.delay_max_steps:
        raise ContractError("requested delay is outside the training contract support")
    if teacher_motion is None:
        reset_state = {
            "mode": "named_stand_root_executed_zero_action_q_zero_velocity",
            "key_name": "stand",
        }
        center_action = np.zeros(ACTION_DIM, dtype=np.float64)
    else:
        reset_state, center_action = _teacher_frame_reset_payload(
            binding, teacher_motion, teacher_frame_index
        )
    tick = np.arange(FIXED_TAPE_TICKS, dtype=np.float64)[:, None]
    joint = np.arange(ACTION_DIM, dtype=np.float64)[None, :]
    frequency = 1.0 + np.mod(joint, 5.0)
    phase = 2.0 * math.pi * joint / ACTION_DIM
    actions = center_action[None, :] + 0.02 * np.sin(
        2.0 * math.pi * frequency * tick / FIXED_TAPE_TICKS + phase
    )
    return {
        "schema_version": 1,
        "kind": TAPE_KIND,
        "ticks": FIXED_TAPE_TICKS,
        "policy_hz": 1.0 / binding.policy_step_dt_s,
        "joint_names": list(binding.joint_names),
        "plant_binding_sha256": binding.binding_sha256,
        "delay_steps": delay,
        "history_fill_action": center_action.tolist(),
        "actions": actions.tolist(),
        "reset_state": reset_state,
    }


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o644)
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite existing output: {path}") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def write_fixed_tape(path: Path | str, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json_bytes(payload)
    _write_new_bytes(Path(path), encoded)
    return _sha256(encoded)


def _load_table_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location("_mujoco_native_table_scene", TABLE_SCENE_PY)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import table scene from {TABLE_SCENE_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _named_id(mujoco: Any, model: Any, kind: Any, name: str, label: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise ContractError(f"MuJoCo model is missing {label} {name!r}")
    return value


def _trace_content_sha256(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(metadata))
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name], dtype="<f8"))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(_canonical_json_bytes({"shape": list(value.shape), "dtype": "<f8"}))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _rotation_from_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    quat = np.array(quaternion, dtype=np.float64, copy=True)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ContractError("root quaternion must be four finite wxyz scalars")
    norm = float(np.linalg.norm(quat))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
        raise ContractError("root quaternion must be unit length")
    quat /= norm
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class MujocoSingleEnv:
    """One exact vendor A3 model plus the diagnostic plant/action contract."""

    def __init__(self, binding: PlantBinding, *, mjcf_path: Path | str = DEFAULT_MJCF):
        try:
            import mujoco
        except ImportError as exc:
            raise ContractError(
                "mujoco Python package is required for the real single-env runner"
            ) from exc
        self.mujoco = mujoco
        self.binding = binding
        self.mjcf_path = Path(mjcf_path).expanduser().resolve()
        scene_module = _load_table_scene_module()
        self.scene = scene_module.load_table_scene(
            mujoco,
            self.mjcf_path,
            collidable=True,
            action_ball_policy=True,
        )
        self.geometry_contract = scene_module.action_ball_policy_geometry_contract(
            self.scene.obstacle_rows
        )
        self.model = self.scene.model
        self.model.opt.timestep = binding.physics_step_dt_s

        joint_ids = np.asarray(
            [
                _named_id(mujoco, self.model, mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
                for name in binding.joint_names
            ],
            dtype=np.int64,
        )
        self.qpos_addr = np.asarray(self.model.jnt_qposadr[joint_ids], dtype=np.int64)
        self.dof_addr = np.asarray(self.model.jnt_dofadr[joint_ids], dtype=np.int64)
        self.actuator_ids = np.asarray(
            [
                _named_id(
                    mujoco,
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name + "_motor",
                    "actuator",
                )
                for name in binding.joint_names
            ],
            dtype=np.int64,
        )
        if len(set(self.qpos_addr.tolist())) != ACTION_DIM:
            raise ContractError("31 action joints do not map bijectively to MuJoCo qpos")
        if len(set(self.dof_addr.tolist())) != ACTION_DIM:
            raise ContractError("31 action joints do not map bijectively to MuJoCo dofs")
        if len(set(self.actuator_ids.tolist())) != ACTION_DIM:
            raise ContractError("31 action joints do not map bijectively to MuJoCo actuators")
        transmission_joint_ids = np.asarray(
            self.model.actuator_trnid[self.actuator_ids, 0], dtype=np.int64
        )
        if not np.array_equal(transmission_joint_ids, joint_ids):
            raise ContractError("actuator name order does not transmit to the same joint order")
        transmission_types = np.asarray(
            self.model.actuator_trntype[self.actuator_ids], dtype=np.int64
        )
        if not np.all(transmission_types == int(mujoco.mjtTrn.mjTRN_JOINT)):
            raise ContractError("all 31 actuators must use direct joint transmission")
        expected_gear = np.zeros((ACTION_DIM, 6), dtype=np.float64)
        expected_gear[:, 0] = 1.0
        if not np.array_equal(
            np.asarray(self.model.actuator_gear[self.actuator_ids], dtype=np.float64),
            expected_gear,
        ):
            raise ContractError("all 31 motor transmissions must have exact unit gear")

        # The training contract's friction coefficients are PhysX dimensionless
        # coefficients.  There is no non-zero conversion to MuJoCo frictionloss.
        self.model.dof_armature[self.dof_addr] = binding.armature
        self.model.dof_damping[self.dof_addr] = 0.0
        self.model.dof_frictionloss[self.dof_addr] = 0.0
        self.model.actuator_ctrlrange[self.actuator_ids, 0] = -binding.effort_limits
        self.model.actuator_ctrlrange[self.actuator_ids, 1] = binding.effort_limits
        self.data = mujoco.MjData(self.model)
        self.delay = ActionDelayLine(binding.delay_max_steps)

        obstacle_ids = set(int(value) for value in self.scene.obstacle_geom_ids.values())
        self._obstacle_geom_ids = obstacle_ids
        self._robot_body_mask = np.zeros(self.model.nbody, dtype=bool)
        pelvis = _named_id(
            mujoco, self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link", "body"
        )
        self._pelvis_body_id = pelvis
        pelvis_free_joints = np.flatnonzero(
            (np.asarray(self.model.jnt_type, dtype=np.int64) == int(mujoco.mjtJoint.mjJNT_FREE))
            & (np.asarray(self.model.jnt_bodyid, dtype=np.int64) == pelvis)
        )
        if pelvis_free_joints.shape != (1,):
            raise ContractError("pelvis_link must own exactly one free joint")
        pelvis_free_joint = int(pelvis_free_joints[0])
        if (
            int(self.model.jnt_qposadr[pelvis_free_joint]) != 0
            or int(self.model.jnt_dofadr[pelvis_free_joint]) != 0
        ):
            raise ContractError("pelvis free joint must start at qpos/dof address zero")
        for body_id in range(1, int(self.model.nbody)):
            cursor = body_id
            seen: set[int] = set()
            while cursor and cursor not in seen:
                if cursor == pelvis:
                    self._robot_body_mask[body_id] = True
                    break
                seen.add(cursor)
                cursor = int(self.model.body_parentid[cursor])
            if cursor in seen:
                raise ContractError("MuJoCo body-parent graph contains a cycle")
        self._robot_geom_mask = self._robot_body_mask[
            np.asarray(self.model.geom_bodyid, dtype=np.int64)
        ]

    def reset(
        self,
        *,
        reset_state: ResetState,
        delay_steps: int,
        history_fill_action: Sequence[float],
    ) -> None:
        mujoco = self.mujoco
        if reset_state.mode == "named_stand_root_executed_zero_action_q_zero_velocity":
            key_id = int(
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_KEY, str(reset_state.key_name)
                )
            )
            if key_id < 0:
                raise ContractError("vendor MJCF must contain named keyframe 'stand'")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            _raw, zero_action_qdes, _clamps = self.binding.decode_action(
                np.zeros(ACTION_DIM)
            )
            self.data.qpos[self.qpos_addr] = zero_action_qdes
            self.data.qvel[:] = 0.0
        elif reset_state.mode == "teacher_frame":
            mujoco.mj_resetData(self.model, self.data)
            assert reset_state.root_pos is not None
            assert reset_state.root_quat_wxyz is not None
            assert reset_state.root_lin_vel_w is not None
            assert reset_state.root_ang_vel_w is not None
            assert reset_state.joint_pos is not None
            assert reset_state.joint_vel is not None
            root_quat = reset_state.root_quat_wxyz / np.linalg.norm(
                reset_state.root_quat_wxyz
            )
            rotation = _rotation_from_wxyz(root_quat)
            root_origin_lin_vel = reset_state.root_lin_vel_w.copy()
            if reset_state.root_lin_vel_point == "center_of_mass":
                root_to_com_world = rotation @ np.asarray(
                    self.model.body_ipos[self._pelvis_body_id], dtype=np.float64
                )
                root_origin_lin_vel -= np.cross(
                    reset_state.root_ang_vel_w, root_to_com_world
                )
            self.data.qpos[0:3] = reset_state.root_pos
            self.data.qpos[3:7] = root_quat
            self.data.qpos[self.qpos_addr] = reset_state.joint_pos
            self.data.qvel[0:3] = root_origin_lin_vel
            self.data.qvel[3:6] = rotation.T @ reset_state.root_ang_vel_w
            self.data.qvel[self.dof_addr] = reset_state.joint_vel
        else:
            raise ContractError(f"unsupported reset mode: {reset_state.mode}")
        self.data.time = 0.0
        self.data.ctrl[:] = 0.0
        if self.data.act.size:
            self.data.act[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.delay.reset(delay_steps, history_fill_action)

    def _geom_name(self, geom_id: int) -> str:
        value = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
        )
        return str(value) if value else f"geom_{int(geom_id)}"

    def _contact_counts(
        self,
    ) -> tuple[int, int, float, float, str | None, str | None, str | None]:
        table_pairs = 0
        self_pairs = 0
        max_table_penetration = 0.0
        max_self_penetration = 0.0
        first_table_pair = None
        first_self_pair = None
        worst_self_pair = None
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (
                (g1 in self._obstacle_geom_ids and self._robot_geom_mask[g2])
                or (g2 in self._obstacle_geom_ids and self._robot_geom_mask[g1])
            ):
                table_pairs += 1
                if first_table_pair is None:
                    first_table_pair = f"{self._geom_name(g1)}~{self._geom_name(g2)}"
                max_table_penetration = max(
                    max_table_penetration, max(0.0, -float(contact.dist))
                )
            if self._robot_geom_mask[g1] and self._robot_geom_mask[g2]:
                self_pairs += 1
                pair = f"{self._geom_name(g1)}~{self._geom_name(g2)}"
                if first_self_pair is None:
                    first_self_pair = pair
                penetration = max(0.0, -float(contact.dist))
                if penetration >= max_self_penetration:
                    max_self_penetration = penetration
                    worst_self_pair = pair
        return (
            table_pairs,
            self_pairs,
            max_table_penetration,
            max_self_penetration,
            first_table_pair,
            first_self_pair,
            worst_self_pair,
        )

    def step(self, actor_action: Sequence[float]) -> dict[str, Any]:
        current_action = _finite_vector(actor_action, "actor_action")
        delayed_action = self.delay.push(current_action)
        qdes_raw, qdes, qdes_clamps = self.binding.decode_action(delayed_action)
        effort_clips = 0
        velocity_events = 0
        table_pairs = 0
        self_pairs = 0
        table_substeps = 0
        self_substeps = 0
        max_table_penetration = 0.0
        max_self_penetration = 0.0
        worst_self_pair = None
        max_velocity_ratio = 0.0
        first_table_pair = None
        first_self_pair = None
        tau = np.zeros(ACTION_DIM, dtype=np.float64)
        for _ in range(self.binding.control_decimation):
            q = np.asarray(self.data.qpos[self.qpos_addr], dtype=np.float64)
            qd = np.asarray(self.data.qvel[self.dof_addr], dtype=np.float64)
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise ContractError("non-finite MuJoCo state before physics substep")
            _raw_tau, tau, clip_count = total_pd_effort(
                self.binding, qdes, q, qd
            )
            effort_clips += clip_count
            self.data.ctrl[self.actuator_ids] = tau
            self.mujoco.mj_step(self.model, self.data)
            qd_post = np.asarray(self.data.qvel[self.dof_addr], dtype=np.float64)
            if not np.isfinite(qd_post).all():
                raise ContractError("non-finite MuJoCo velocity after physics substep")
            velocity_ratio = np.abs(qd_post) / self.binding.velocity_limits
            max_velocity_ratio = max(max_velocity_ratio, float(np.max(velocity_ratio)))
            velocity_events += int(np.count_nonzero(velocity_ratio > (1.0 + 1.0e-9)))
            (
                tc,
                sc,
                table_penetration,
                self_penetration,
                table_pair,
                self_pair,
                substep_worst_self_pair,
            ) = self._contact_counts()
            table_pairs += tc
            self_pairs += sc
            table_substeps += int(tc > 0)
            self_substeps += int(sc > 0)
            if first_table_pair is None and table_pair is not None:
                first_table_pair = table_pair
            if first_self_pair is None and self_pair is not None:
                first_self_pair = self_pair
            max_table_penetration = max(max_table_penetration, table_penetration)
            if substep_worst_self_pair is not None and (
                worst_self_pair is None or self_penetration > max_self_penetration
            ):
                max_self_penetration = self_penetration
                worst_self_pair = substep_worst_self_pair
        q = np.asarray(self.data.qpos[self.qpos_addr], dtype=np.float64).copy()
        qd = np.asarray(self.data.qvel[self.dof_addr], dtype=np.float64).copy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ContractError("non-finite MuJoCo state after control step")
        pelvis_rotation = np.asarray(
            self.data.xmat[self._pelvis_body_id], dtype=np.float64
        ).reshape(3, 3)
        return {
            "actor_action": current_action.copy(),
            "delayed_action": delayed_action,
            "qdes_raw": qdes_raw,
            "qdes": qdes,
            "q": q,
            "qd": qd,
            "tau": tau.copy(),
            "qdes_clamp_joint_events": qdes_clamps,
            "effort_clip_joint_events": effort_clips,
            "velocity_limit_joint_events": velocity_events,
            "table_contact_pairs": table_pairs,
            "self_contact_pairs": self_pairs,
            "table_contact_substeps": table_substeps,
            "self_contact_substeps": self_substeps,
            "max_table_penetration_m": max_table_penetration,
            "max_self_penetration_m": max_self_penetration,
            "worst_self_contact_pair": worst_self_pair,
            "max_joint_velocity_ratio": max_velocity_ratio,
            "pelvis_height_m": float(self.data.xpos[self._pelvis_body_id, 2]),
            "pelvis_up_world_z": float(pelvis_rotation[2, 2]),
            "first_table_contact_pair": first_table_pair,
            "first_self_contact_pair": first_self_pair,
        }

    def run_tape(self, tape: FixedTape) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if tape.plant_binding_sha256 != self.binding.binding_sha256:
            raise ContractError("tape and runner plant binding differ")
        self.reset(
            reset_state=tape.reset_state,
            delay_steps=tape.delay_steps,
            history_fill_action=tape.history_fill_action,
        )
        trace_rows: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "actor_action",
                "delayed_action",
                "qdes_raw",
                "qdes",
                "q",
                "qd",
                "tau",
            )
        }
        counters = {
            "policy_ticks": 0,
            "physics_substeps": 0,
            "qdes_clamp_joint_events": 0,
            "effort_clip_joint_events": 0,
            "velocity_limit_joint_events": 0,
            "table_contact_pairs": 0,
            "self_contact_pairs": 0,
            "table_contact_substeps": 0,
            "self_contact_substeps": 0,
        }
        max_table_penetration = 0.0
        max_self_penetration = 0.0
        worst_self_pair = None
        max_velocity_ratio = 0.0
        min_pelvis_height = math.inf
        min_pelvis_up_z = math.inf
        first_table_contact = None
        first_self_contact = None
        for tick, action in enumerate(tape.actions):
            row = self.step(action)
            for key in trace_rows:
                trace_rows[key].append(np.asarray(row[key], dtype=np.float64))
            counters["policy_ticks"] += 1
            counters["physics_substeps"] += self.binding.control_decimation
            for key in (
                "qdes_clamp_joint_events",
                "effort_clip_joint_events",
                "velocity_limit_joint_events",
                "table_contact_pairs",
                "self_contact_pairs",
                "table_contact_substeps",
                "self_contact_substeps",
            ):
                counters[key] += int(row[key])
            max_table_penetration = max(
                max_table_penetration, float(row["max_table_penetration_m"])
            )
            if row["worst_self_contact_pair"] is not None and (
                worst_self_pair is None
                or float(row["max_self_penetration_m"]) > max_self_penetration
            ):
                max_self_penetration = float(row["max_self_penetration_m"])
                worst_self_pair = row["worst_self_contact_pair"]
            max_velocity_ratio = max(
                max_velocity_ratio, float(row["max_joint_velocity_ratio"])
            )
            min_pelvis_height = min(min_pelvis_height, float(row["pelvis_height_m"]))
            min_pelvis_up_z = min(min_pelvis_up_z, float(row["pelvis_up_world_z"]))
            if first_table_contact is None and row["first_table_contact_pair"] is not None:
                first_table_contact = {
                    "policy_tick_zero_based": int(tick),
                    "pair": str(row["first_table_contact_pair"]),
                }
            if first_self_contact is None and row["first_self_contact_pair"] is not None:
                first_self_contact = {
                    "policy_tick_zero_based": int(tick),
                    "pair": str(row["first_self_contact_pair"]),
                }
        arrays = {
            key: np.stack(rows, axis=0) for key, rows in trace_rows.items()
        }
        if any(value.shape != (FIXED_TAPE_TICKS, ACTION_DIM) for value in arrays.values()):
            raise ContractError("trace is not exactly 100 ticks x 31 joints")
        metadata = {
            "schema_version": 1,
            "kind": TRACE_KIND,
            "plant_binding_sha256": self.binding.binding_sha256,
            "fixed_tape_sha256": tape.source_sha256,
            "ticks": FIXED_TAPE_TICKS,
            "action_dim": ACTION_DIM,
            "delay_steps": tape.delay_steps,
            "reset_mode": tape.reset_state.mode,
        }
        trace_sha = _trace_content_sha256(arrays, metadata)
        reasons = {
            "fixed_tape_complete": 1,
            "nonfinite_abort": 0,
            "table_contact_observed": int(counters["table_contact_pairs"] > 0),
            "self_contact_observed": int(counters["self_contact_pairs"] > 0),
            "joint_velocity_limit_observed": int(
                counters["velocity_limit_joint_events"] > 0
            ),
        }
        reset_lineage = {"mode": tape.reset_state.mode}
        if tape.reset_state.mode == "teacher_frame":
            reset_lineage.update(
                {
                    "source_motion_path": tape.reset_state.source_motion_path,
                    "source_motion_sha256": tape.reset_state.source_motion_sha256,
                    "source_motion_uid": tape.reset_state.source_motion_uid,
                    "source_frame_index": tape.reset_state.source_frame_index,
                    "source_joint_order_contract_id": (
                        tape.reset_state.source_joint_order_contract_id
                    ),
                    "source_joint_order_contract_sha256": (
                        tape.reset_state.source_joint_order_contract_sha256
                    ),
                    "root_lin_vel_point": tape.reset_state.root_lin_vel_point,
                }
            )
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "status": "DIAGNOSTIC_FIXED_TAPE_COMPLETE",
            "diagnostic_unauthorized": True,
            "authorization": {
                "canonical_training": False,
                "promotion": False,
                "deployment": False,
                "hardware_commands": False,
            },
            "scope": {
                "implemented": [
                    "vendor_a3_mjcf_in_memory_five_solid_table_net",
                    "schema3_31d_action_decoder",
                    "episode_fixed_whole_row_action_delay",
                    "total_pd_effort_clip",
                    "complete_mjdata_reset",
                    "teacher_frame0_reset",
                    "100_tick_fixed_tape",
                ],
                "not_implemented": [
                    "ball",
                    "reward",
                    "policy_observation",
                    "termination_manager",
                    "vecenv",
                    "ppo",
                    "checkpoint_export",
                ],
            },
            "lineage": {
                "training_contract_path": self.binding.source_path,
                "training_contract_sha256": self.binding.source_sha256,
                "plant_binding_sha256": self.binding.binding_sha256,
                "fixed_tape_path": tape.source_path,
                "fixed_tape_sha256": tape.source_sha256,
                "root_mjcf_path": str(self.mjcf_path),
                "root_mjcf_sha256": self.scene.canonical_xml_sha256,
                "augmented_scene_sha256": self.scene.augmented_xml_sha256,
                "table_geometry_sha256": self.geometry_contract["sha256"],
                "reset_state": reset_lineage,
                "trace_content_sha256": trace_sha,
            },
            "runtime": {
                "mujoco_version": str(getattr(self.mujoco, "__version__", "unknown")),
                "physics_step_dt_s": self.binding.physics_step_dt_s,
                "policy_step_dt_s": self.binding.policy_step_dt_s,
                "control_decimation": self.binding.control_decimation,
                "action_dim": ACTION_DIM,
                "delay_steps": tape.delay_steps,
                "delay_histogram_episode_count": {str(tape.delay_steps): 1},
                "reset_mode": tape.reset_state.mode,
                "history_fill_source": "fixed_tape_explicit_normalized_action",
                "passive_damping": "zeroed",
                "mujoco_frictionloss": "zeroed",
                "physx_dimensionless_joint_friction": "not_applied_no_unit_conversion",
            },
            "reasons": reasons,
            "counters": counters,
            "safety": {
                "max_joint_velocity_ratio": max_velocity_ratio,
                "max_table_penetration_m": max_table_penetration,
                "max_self_penetration_m": max_self_penetration,
                "worst_self_contact_pair": worst_self_pair,
                "min_pelvis_height_m": min_pelvis_height,
                "min_pelvis_up_world_z": min_pelvis_up_z,
                "table_contact_observed": bool(counters["table_contact_pairs"]),
                "self_contact_observed": bool(counters["self_contact_pairs"]),
                "first_table_contact": first_table_contact,
                "first_self_contact": first_self_contact,
                "diagnostic_no_contact_gate_passed": bool(
                    counters["table_contact_pairs"] == 0
                    and counters["self_contact_pairs"] == 0
                    and counters["velocity_limit_joint_events"] == 0
                ),
                "safe_for_hardware_claim": False,
            },
        }
        return arrays, receipt


def _write_trace(path: Path, arrays: Mapping[str, np.ndarray], receipt: Mapping[str, Any]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContractError(f"refusing to overwrite existing output: {path}")
    try:
        with path.open("xb") as stream:
            np.savez(
                stream,
                **arrays,
                metadata_json=np.asarray(
                    _canonical_json_bytes(
                        {
                            "kind": TRACE_KIND,
                            "lineage": receipt["lineage"],
                            "runtime": receipt["runtime"],
                        }
                    ).decode("utf-8")
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return _sha256(path.read_bytes())


def _cmd_make_tape(args: argparse.Namespace) -> int:
    binding = load_plant_binding(args.contract)
    payload = build_probe_tape(
        binding,
        delay_steps=args.delay,
        teacher_motion=args.teacher_motion,
        teacher_frame_index=args.teacher_frame,
    )
    digest = write_fixed_tape(args.tape, payload)
    print(
        json.dumps(
            {
                "status": "DIAGNOSTIC_FIXED_TAPE_WRITTEN",
                "diagnostic_unauthorized": True,
                "path": str(Path(args.tape).expanduser().resolve()),
                "sha256": digest,
                "plant_binding_sha256": binding.binding_sha256,
                "ticks": FIXED_TAPE_TICKS,
                "action_dim": ACTION_DIM,
                "delay_steps": args.delay,
                "reset_mode": payload["reset_state"]["mode"],
                "teacher_motion_sha256": payload["reset_state"].get(
                    "source_motion_sha256"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    binding = load_plant_binding(args.contract)
    tape = load_fixed_tape(args.tape, binding)
    runner = MujocoSingleEnv(binding, mjcf_path=args.mjcf)
    arrays, receipt = runner.run_tape(tape)
    trace_file_sha = _write_trace(Path(args.trace), arrays, receipt)
    receipt["lineage"]["trace_path"] = str(Path(args.trace).expanduser().resolve())
    receipt["lineage"]["trace_file_sha256"] = trace_file_sha
    encoded = _canonical_json_bytes(receipt)
    _write_new_bytes(Path(args.receipt), encoded)
    print(encoded.decode("utf-8"))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-tape", help="write one immutable 100-tick probe tape")
    make.add_argument("--contract", required=True, type=Path)
    make.add_argument("--tape", required=True, type=Path)
    make.add_argument("--delay", required=True, type=int)
    make.add_argument(
        "--teacher-motion",
        type=Path,
        help=(
            "measured-racket motion NPZ whose selected frame supplies root/q/dq reset; "
            "the probe action is centered on that frame's q"
        ),
    )
    make.add_argument("--teacher-frame", type=int, default=0)
    make.set_defaults(func=_cmd_make_tape)
    run = sub.add_parser("run", help="replay a fixed tape through one real MuJoCo A3 env")
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--tape", required=True, type=Path)
    run.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    run.add_argument("--trace", required=True, type=Path)
    run.add_argument("--receipt", required=True, type=Path)
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ContractError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
