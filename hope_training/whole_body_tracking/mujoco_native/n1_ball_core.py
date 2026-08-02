#!/usr/bin/env python3
"""Diagnostic MuJoCo ball-conditioned N1 single-environment core.

This closes the first policy-environment-shaped gap above ``single_env``:

* one externally SHA-bound manual probe or immutable-tape-derived question;
* one native free-joint ball in the five-solid ActionBall scene;
* deterministic robot + ball reset;
* purpose-grouped robot/ball/task/clock observations; and
* actual-contact edge latches for racket, table, net and floor.

It deliberately has no final flat ABI, reward, VecEnv, PPO, normalizer,
checkpoint or export.  A completed run proves deterministic N1 ball plumbing,
not learnability, contact fidelity, canonical training or deployment safety.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import physical_ball_scene
from . import single_env


QUESTION_KIND = "a3_mujoco_n1_physical_launch_probe_v1"
RECEIPT_KIND = "a3_mujoco_n1_ball_core_receipt_v1"
TRACE_KIND = "a3_mujoco_n1_ball_core_trace_v1"
FIXED_QUESTION_TAPE_PY = (
    single_env.REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "tasks/tracking/mdp/action_ball_fixed_question_tape.py"
)


class N1BallCoreError(RuntimeError):
    """The N1 question, runtime state or event ledger is invalid."""


def _reject_constant(value: str) -> None:
    raise N1BallCoreError(f"non-finite JSON constant is forbidden: {value}")


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise N1BallCoreError(f"duplicate JSON key is forbidden: {key}")
        out[key] = value
    return out


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise N1BallCoreError(f"payload is not finite canonical JSON: {exc}") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(single_env.REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise N1BallCoreError(
            f"authority source must be inside repository root: {path}"
        ) from exc


def _vector(value: Any, width: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (width,) or not np.isfinite(out).all():
        raise N1BallCoreError(f"{name} must be {width} finite scalars")
    return out.copy()


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise N1BallCoreError(f"{name} cannot be bool")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise N1BallCoreError(f"{name} must be positive finite")
    return out


@dataclass(frozen=True)
class N1Question:
    source_path: str
    source_sha256: str
    question_id: str
    scene_binding_sha256: str
    birth_position_w_m: np.ndarray
    birth_linear_velocity_w_mps: np.ndarray
    birth_spin_w_radps: np.ndarray
    landing_aim_xy_w_m: np.ndarray
    nominal_time_to_contact_s: float
    spin_valid: bool
    authority: Mapping[str, Any]


def build_question_payload(
    *,
    question_id: str,
    scene_binding_sha256: str,
    birth_position_w_m: Sequence[float],
    birth_linear_velocity_w_mps: Sequence[float],
    landing_aim_xy_w_m: Sequence[float],
    nominal_time_to_contact_s: float,
    birth_spin_w_radps: Sequence[float] = (0.0, 0.0, 0.0),
    spin_valid: bool = False,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(question_id, str) or not question_id.strip():
        raise N1BallCoreError("question_id must be a non-empty string")
    if (
        not isinstance(scene_binding_sha256, str)
        or len(scene_binding_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in scene_binding_sha256)
    ):
        raise N1BallCoreError("scene_binding_sha256 must be lowercase SHA-256")
    position = _vector(birth_position_w_m, 3, "birth_position_w_m")
    velocity = _vector(
        birth_linear_velocity_w_mps, 3, "birth_linear_velocity_w_mps"
    )
    spin = _vector(birth_spin_w_radps, 3, "birth_spin_w_radps")
    aim = _vector(landing_aim_xy_w_m, 2, "landing_aim_xy_w_m")
    ttc = _positive(nominal_time_to_contact_s, "nominal_time_to_contact_s")
    if type(spin_valid) is not bool:
        raise N1BallCoreError("spin_valid must be bool")
    if spin_valid:
        raise N1BallCoreError(
            "spin_valid=true is forbidden while native flight has no Magnus model"
        )
    if np.any(spin != 0.0):
        raise N1BallCoreError("spin-invalid N1 question must carry exact zero spin")
    if authority is None:
        authority = {
            "kind": "manual_native_gravity_engineering_probe",
            "immutable_n1_tape_bound": False,
            "incoming_question_parity": False,
        }
    authority = dict(authority)
    if authority.get("kind") not in {
        "manual_native_gravity_engineering_probe",
        "immutable_n1_tape_with_explicit_native_launch",
    }:
        raise N1BallCoreError("unsupported N1 question authority kind")
    payload = {
        "schema_version": 1,
        "kind": QUESTION_KIND,
        "question_id": question_id,
        "scene_binding_sha256": scene_binding_sha256,
        "birth": {
            "position_w_m": position.tolist(),
            "linear_velocity_w_mps": velocity.tolist(),
            "spin_w_radps": spin.tolist(),
        },
        "task": {
            "landing_aim_xy_w_m": aim.tolist(),
            "nominal_time_to_contact_s": ttc,
            "spin_valid": spin_valid,
        },
        "authority": authority,
        "semantics": {
            "policy_conditioning": "achieved_physical_ball_plus_landing_aim_plus_contact_clock",
            "desired_at_contact": "not_present_in_this_landing_only_core",
            "teacher": "provided_by_separate_robot_fixed_tape",
            "outcome": "actual_native_contact_events_only_no_reward",
        },
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
    }
    unsigned = _canonical_json_bytes(payload)
    payload["content_sha256"] = _sha256(unsigned)
    return payload


def _load_fixed_question_tape_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_mujoco_n1_fixed_question_tape_authority", FIXED_QUESTION_TAPE_PY
    )
    if spec is None or spec.loader is None:
        raise N1BallCoreError(
            f"cannot import immutable N1 tape authority from {FIXED_QUESTION_TAPE_PY}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_question_from_immutable_tape(
    *,
    immutable_tape_path: Path | str,
    expected_immutable_tape_sha256: str,
    target_recipe: str,
    scene_binding_sha256: str,
    physical_launch_position_w_m: Sequence[float],
    physical_launch_velocity_w_mps: Sequence[float],
) -> dict[str, Any]:
    """Bind authoritative task fields plus an explicit native-gravity launch.

    The current immutable tape describes the desired contact question, while
    its Isaac producer used venue flight with a possible table bounce.  This
    adapter therefore refuses to pretend a linear reverse ray is equivalent:
    the MuJoCo launch is explicit and the authority marks question parity false
    until a cross-engine launch producer is installed.
    """

    source = Path(immutable_tape_path).expanduser().resolve()
    module = _load_fixed_question_tape_module()
    try:
        tape = module.load_immutable_n1_tape(
            source, expected_file_sha256=expected_immutable_tape_sha256
        )
        question = tape.question_payload
        lineage = tape.target_lineage(target_recipe)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise N1BallCoreError(f"invalid immutable N1 tape authority: {exc}") from exc
    if question.get("incoming_spin_w_radps") != [0.0, 0.0, 0.0]:
        raise N1BallCoreError("this no-Magnus N1 core only accepts zero-spin tape rows")
    authority = {
        "kind": "immutable_n1_tape_with_explicit_native_launch",
        "immutable_n1_tape_bound": True,
        "incoming_question_parity": False,
        "why_not_parity": (
            "immutable tape flight may include venue aero/table bounce; explicit native launch "
            "is not claimed to reproduce its scheduled contact"
        ),
        "immutable_tape_repo_relative_path": _repo_relative(source),
        "immutable_tape_file_sha256": expected_immutable_tape_sha256,
        "immutable_tape_canonical_sha256": tape.canonical_sha256,
        "base_question_sha256": tape.question_sha256,
        "action_uid": int(question["action_uid"]),
        "motion_sha256": str(question["motion_sha256"]),
        "physics_sha256": str(question["physics_sha256"]),
        "profile_sha256": str(question["profile_sha256"]),
        "target_recipe": target_recipe,
        "target_producer_sha256": lineage["target_producer_sha256"],
        "target_column_sha256": lineage["target_column_sha256"],
        "launch_recipe": "explicit_native_gravity_probe_v1",
    }
    return build_question_payload(
        question_id=f"immutable_{tape.question_sha256[:12]}_{target_recipe}",
        scene_binding_sha256=scene_binding_sha256,
        birth_position_w_m=physical_launch_position_w_m,
        birth_linear_velocity_w_mps=physical_launch_velocity_w_mps,
        birth_spin_w_radps=(0.0, 0.0, 0.0),
        landing_aim_xy_w_m=question["landing_aim_w_xy_m"],
        nominal_time_to_contact_s=question["time_to_contact_s"],
        spin_valid=False,
        authority=authority,
    )


def write_question(path: Path | str, payload: Mapping[str, Any]) -> str:
    raw = _canonical_json_bytes(payload)
    single_env._write_new_bytes(Path(path).expanduser().resolve(), raw)
    return _sha256(raw)


def load_question(
    path: Path | str,
    *,
    expected_file_sha256: str,
    scene_binding_sha256: str,
) -> N1Question:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise N1BallCoreError(f"cannot read strict question {source}: {exc}") from exc
    if _sha256(raw) != expected_file_sha256:
        raise N1BallCoreError("question file SHA differs from external authority")
    if not isinstance(payload, dict):
        raise N1BallCoreError("question root must be an object")
    expected_top = {
        "schema_version",
        "kind",
        "question_id",
        "scene_binding_sha256",
        "birth",
        "task",
        "authority",
        "semantics",
        "diagnostic_unauthorized",
        "authorization",
        "content_sha256",
    }
    if set(payload) != expected_top:
        raise N1BallCoreError("question top-level keys differ from schema v1")
    if payload.get("schema_version") != 1 or payload.get("kind") != QUESTION_KIND:
        raise N1BallCoreError("question schema/kind mismatch")
    content_sha = payload.pop("content_sha256")
    recomputed = _sha256(_canonical_json_bytes(payload))
    payload["content_sha256"] = content_sha
    if content_sha != recomputed:
        raise N1BallCoreError("question content_sha256 mismatch")
    if payload.get("scene_binding_sha256") != scene_binding_sha256:
        raise N1BallCoreError("question binds a different physical-ball scene")
    if payload.get("diagnostic_unauthorized") is not True:
        raise N1BallCoreError("question must remain diagnostic_unauthorized")
    if payload.get("semantics") != {
        "policy_conditioning": "achieved_physical_ball_plus_landing_aim_plus_contact_clock",
        "desired_at_contact": "not_present_in_this_landing_only_core",
        "teacher": "provided_by_separate_robot_fixed_tape",
        "outcome": "actual_native_contact_events_only_no_reward",
    }:
        raise N1BallCoreError("question semantics differ from schema v1")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "training",
        "promotion",
        "deployment",
        "hardware",
    } or any(value is not False for value in authorization.values()):
        raise N1BallCoreError("question authorization must be exact all-false")
    birth = payload.get("birth")
    task = payload.get("task")
    authority = payload.get("authority")
    if not isinstance(authority, dict):
        raise N1BallCoreError("question authority must be an object")
    authority_kind = authority.get("kind")
    if authority_kind == "manual_native_gravity_engineering_probe":
        if authority != {
            "kind": "manual_native_gravity_engineering_probe",
            "immutable_n1_tape_bound": False,
            "incoming_question_parity": False,
        }:
            raise N1BallCoreError("manual question authority keys differ")
    elif authority_kind == "immutable_n1_tape_with_explicit_native_launch":
        required_authority = {
            "kind",
            "immutable_n1_tape_bound",
            "incoming_question_parity",
            "why_not_parity",
            "immutable_tape_repo_relative_path",
            "immutable_tape_file_sha256",
            "immutable_tape_canonical_sha256",
            "base_question_sha256",
            "action_uid",
            "motion_sha256",
            "physics_sha256",
            "profile_sha256",
            "target_recipe",
            "target_producer_sha256",
            "target_column_sha256",
            "launch_recipe",
        }
        if (
            set(authority) != required_authority
            or authority.get("immutable_n1_tape_bound") is not True
            or authority.get("incoming_question_parity") is not False
        ):
            raise N1BallCoreError("immutable question authority keys differ")
        tape_path = single_env.REPO_ROOT / str(
            authority["immutable_tape_repo_relative_path"]
        )
        module = _load_fixed_question_tape_module()
        try:
            tape = module.load_immutable_n1_tape(
                tape_path,
                expected_file_sha256=authority["immutable_tape_file_sha256"],
            )
            lineage = tape.target_lineage(authority["target_recipe"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise N1BallCoreError(
                f"immutable question authority cannot be revalidated: {exc}"
            ) from exc
        if (
            tape.canonical_sha256 != authority["immutable_tape_canonical_sha256"]
            or tape.question_sha256 != authority["base_question_sha256"]
            or lineage["target_producer_sha256"]
            != authority["target_producer_sha256"]
            or lineage["target_column_sha256"]
            != authority["target_column_sha256"]
        ):
            raise N1BallCoreError("immutable question authority lineage mismatch")
    else:
        raise N1BallCoreError("unsupported question authority")
    if not isinstance(birth, dict) or set(birth) != {
        "position_w_m",
        "linear_velocity_w_mps",
        "spin_w_radps",
    }:
        raise N1BallCoreError("question birth keys differ")
    if not isinstance(task, dict) or set(task) != {
        "landing_aim_xy_w_m",
        "nominal_time_to_contact_s",
        "spin_valid",
    }:
        raise N1BallCoreError("question task keys differ")
    # Reuse the constructor validation so build and consume cannot drift.
    build_question_payload(
        question_id=payload["question_id"],
        scene_binding_sha256=payload["scene_binding_sha256"],
        birth_position_w_m=birth["position_w_m"],
        birth_linear_velocity_w_mps=birth["linear_velocity_w_mps"],
        birth_spin_w_radps=birth["spin_w_radps"],
        landing_aim_xy_w_m=task["landing_aim_xy_w_m"],
        nominal_time_to_contact_s=task["nominal_time_to_contact_s"],
        spin_valid=task["spin_valid"],
        authority=authority,
    )
    return N1Question(
        source_path=str(source),
        source_sha256=_sha256(raw),
        question_id=payload["question_id"],
        scene_binding_sha256=scene_binding_sha256,
        birth_position_w_m=_vector(birth["position_w_m"], 3, "birth.position"),
        birth_linear_velocity_w_mps=_vector(
            birth["linear_velocity_w_mps"], 3, "birth.linear_velocity"
        ),
        birth_spin_w_radps=_vector(birth["spin_w_radps"], 3, "birth.spin"),
        landing_aim_xy_w_m=_vector(task["landing_aim_xy_w_m"], 2, "task.aim"),
        nominal_time_to_contact_s=_positive(
            task["nominal_time_to_contact_s"], "task.ttc"
        ),
        spin_valid=task["spin_valid"],
        authority=dict(authority),
    )


class MujocoN1BallCore:
    """One physical-ball scene around the existing exact plant/action core."""

    def __init__(
        self,
        binding: single_env.PlantBinding,
        *,
        mjcf_path: Path | str = single_env.DEFAULT_MJCF,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise N1BallCoreError("mujoco Python package is required") from exc
        self.mujoco = mujoco
        self.binding = binding
        self.ball_contract = physical_ball_scene.load_ball_contract(binding.source_path)
        if not math.isclose(
            self.ball_contract.physics_step_dt_s,
            binding.physics_step_dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise N1BallCoreError("ball and plant physics step differ")
        self.scene = physical_ball_scene.compile_physical_ball_scene(
            mujoco,
            mjcf_path=mjcf_path,
            ball_contract=self.ball_contract,
            strict_pair_filter=True,
            include_floor_pair=True,
        )
        scene_binding = self.scene.binding
        required_targets = {
            physical_ball_scene.RACKET_GEOM_NAME,
            physical_ball_scene.TABLE_GEOM_NAME,
            *physical_ball_scene.NET_GEOM_NAMES,
            physical_ball_scene.FLOOR_GEOM_NAME,
        }
        if (
            scene_binding.get("with_ball") is not True
            or scene_binding.get("strict_pair_filter") is not True
            or set(scene_binding.get("explicit_pair_targets", ())) != required_targets
            or scene_binding.get("robot_only_keepout_is_ball_surface") is not False
        ):
            raise N1BallCoreError(
                "N1 core requires strict ball racket/table/net/floor pairs and no keepout"
            )
        compiled = scene_binding.get("compiled_runtime")
        if not isinstance(compiled, dict) or not math.isclose(
            float(compiled.get("model_timestep_s", math.nan)),
            binding.physics_step_dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise N1BallCoreError("compiled physical-ball timestep differs from plant")
        self.plant = single_env.MujocoSingleEnv(
            binding,
            mjcf_path=mjcf_path,
            precompiled_scene=self.scene,
        )
        self.data = self.plant.data
        self.model = self.plant.model
        self._racket_geom_id = single_env._named_id(
            mujoco,
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            physical_ball_scene.RACKET_GEOM_NAME,
            "racket geom",
        )
        self._table_geom_id = self.scene.obstacle_geom_ids[
            physical_ball_scene.TABLE_GEOM_NAME
        ]
        self._net_geom_ids = {
            self.scene.obstacle_geom_ids[name]
            for name in physical_ball_scene.NET_GEOM_NAMES
        }
        self._floor_geom_id = single_env._named_id(
            mujoco,
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            physical_ball_scene.FLOOR_GEOM_NAME,
            "floor geom",
        )
        self.question: N1Question | None = None
        self.policy_tick = 0
        self._active_contact_labels: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._ambiguous_contact_substeps = 0
        self._racket_contact_edges = 0
        self._outgoing_state: dict[str, Any] | None = None
        self._contact_invalid_reasons: set[str] = set()

    @property
    def scene_binding_sha256(self) -> str:
        return str(self.scene.binding["binding_sha256"])

    def _contact_labels(self) -> set[str]:
        labels: set[str] = set()
        ball_id = self.scene.ball_geom_id
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if ball_id not in (g1, g2):
                continue
            other = g2 if g1 == ball_id else g1
            if other == self._racket_geom_id:
                labels.add("racket")
            elif other == self._table_geom_id:
                labels.add("table")
            elif other in self._net_geom_ids:
                labels.add("net")
            elif other == self._floor_geom_id:
                labels.add("floor")
            else:
                labels.add("unexpected")
        return labels

    def _observe_substep(self, _model: Any, _data: Any, substep_index: int) -> None:
        labels = self._contact_labels()
        if "unexpected" in labels:
            raise N1BallCoreError("physical ball touched an unexpected geom pair")
        if len(labels) > 1:
            self._ambiguous_contact_substeps += 1
            if "racket" in labels:
                self._contact_invalid_reasons.add("racket_contact_simultaneous_with_other")
        racket_was_active = "racket" in self._active_contact_labels
        racket_is_active = "racket" in labels
        if racket_is_active and not racket_was_active:
            self._racket_contact_edges += 1
            if self._racket_contact_edges > 1:
                self._contact_invalid_reasons.add("racket_recontact")
        if racket_was_active and not racket_is_active and self._outgoing_state is None:
            dof = self.scene.ball_dof_adr
            qpos = self.scene.ball_qpos_adr
            self._outgoing_state = {
                "time_s": float(self.data.time),
                "position_w_m": np.asarray(
                    self.data.qpos[qpos : qpos + 3], dtype=np.float64
                ).tolist(),
                "linear_velocity_w_mps": np.asarray(
                    self.data.qvel[dof : dof + 3], dtype=np.float64
                ).tolist(),
                "spin_w_radps": np.asarray(
                    self.data.qvel[dof + 3 : dof + 6], dtype=np.float64
                ).tolist(),
                "semantic": "first_contact_free_physics_substep_after_first_racket_contact",
            }
        for label in sorted(labels - self._active_contact_labels):
            self._events.append(
                {
                    "policy_tick": self.policy_tick,
                    "physics_substep": int(substep_index),
                    "time_s": float(self.data.time),
                    "event": label,
                }
            )
        self._active_contact_labels = labels

    def reset(
        self,
        *,
        robot_tape: single_env.FixedTape,
        question: N1Question,
    ) -> dict[str, np.ndarray]:
        if robot_tape.plant_binding_sha256 != self.binding.binding_sha256:
            raise N1BallCoreError("robot tape and N1 plant binding differ")
        if question.scene_binding_sha256 != self.scene_binding_sha256:
            raise N1BallCoreError("question and N1 scene binding differ")
        self.plant.reset(
            reset_state=robot_tape.reset_state,
            delay_steps=robot_tape.delay_steps,
            history_fill_action=robot_tape.history_fill_action,
        )
        qpos = self.scene.ball_qpos_adr
        dof = self.scene.ball_dof_adr
        self.data.qpos[qpos : qpos + 3] = question.birth_position_w_m
        self.data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof : dof + 3] = question.birth_linear_velocity_w_mps
        self.data.qvel[dof + 3 : dof + 6] = question.birth_spin_w_radps
        self.data.qacc_warmstart[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        initial = self._contact_labels()
        if initial:
            raise N1BallCoreError(
                f"question birth state starts in contact: {sorted(initial)}"
            )
        self.question = question
        self.policy_tick = 0
        self._active_contact_labels = set()
        self._events = []
        self._ambiguous_contact_substeps = 0
        self._racket_contact_edges = 0
        self._outgoing_state = None
        self._contact_invalid_reasons = set()
        return self.observation_groups()

    def observation_groups(self) -> dict[str, np.ndarray]:
        if self.question is None:
            raise N1BallCoreError("reset must install a question before observation")
        qpos = self.scene.ball_qpos_adr
        dof = self.scene.ball_dof_adr
        remaining = max(
            0.0,
            self.question.nominal_time_to_contact_s - float(self.data.time),
        )
        return {
            "robot_joint_pos": np.asarray(
                self.data.qpos[self.plant.qpos_addr], dtype=np.float64
            ).copy(),
            "robot_joint_vel": np.asarray(
                self.data.qvel[self.plant.dof_addr], dtype=np.float64
            ).copy(),
            "incoming_ball_position_w_m": np.asarray(
                self.data.qpos[qpos : qpos + 3], dtype=np.float64
            ).copy(),
            "incoming_ball_linear_velocity_w_mps": np.asarray(
                self.data.qvel[dof : dof + 3], dtype=np.float64
            ).copy(),
            "incoming_ball_spin_w_radps": np.asarray(
                self.data.qvel[dof + 3 : dof + 6], dtype=np.float64
            ).copy(),
            "landing_aim_xy_w_m": self.question.landing_aim_xy_w_m.copy(),
            "time_to_contact_s": np.asarray([remaining], dtype=np.float64),
            "validity": np.asarray(
                [1.0, float(self.question.spin_valid)], dtype=np.float64
            ),
        }

    def step(self, actor_action: Sequence[float]) -> dict[str, Any]:
        if self.question is None:
            raise N1BallCoreError("reset must be called before step")
        event_start = len(self._events)
        row = self.plant.step(
            actor_action,
            substep_observer=self._observe_substep,
        )
        observation = self.observation_groups()
        new_events = [dict(value) for value in self._events[event_start:]]
        self.policy_tick += 1
        return {
            "plant": row,
            "observation_groups": observation,
            "new_events": new_events,
        }

    def run_tape(
        self,
        *,
        robot_tape: single_env.FixedTape,
        question: N1Question,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.reset(robot_tape=robot_tape, question=question)
        traces: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "actor_action",
                "delayed_action",
                "q",
                "qd",
                "ball_position_w_m",
                "ball_linear_velocity_w_mps",
                "ball_spin_w_radps",
                "landing_aim_xy_w_m",
                "time_to_contact_s",
                "validity",
            )
        }
        counters = {
            "policy_ticks": 0,
            "physics_substeps": 0,
            "racket_contact_edges": 0,
            "table_contact_edges": 0,
            "net_contact_edges": 0,
            "floor_contact_edges": 0,
            "unexpected_contact_edges": 0,
            "ambiguous_simultaneous_contact_substeps": 0,
        }
        for action in robot_tape.actions:
            row = self.step(action)
            plant = row["plant"]
            obs = row["observation_groups"]
            traces["actor_action"].append(plant["actor_action"])
            traces["delayed_action"].append(plant["delayed_action"])
            traces["q"].append(plant["q"])
            traces["qd"].append(plant["qd"])
            traces["ball_position_w_m"].append(obs["incoming_ball_position_w_m"])
            traces["ball_linear_velocity_w_mps"].append(
                obs["incoming_ball_linear_velocity_w_mps"]
            )
            traces["ball_spin_w_radps"].append(obs["incoming_ball_spin_w_radps"])
            traces["landing_aim_xy_w_m"].append(obs["landing_aim_xy_w_m"])
            traces["time_to_contact_s"].append(obs["time_to_contact_s"])
            traces["validity"].append(obs["validity"])
            counters["policy_ticks"] += 1
            counters["physics_substeps"] += self.binding.control_decimation
            for event in row["new_events"]:
                counters[f"{event['event']}_contact_edges"] += 1
        counters["ambiguous_simultaneous_contact_substeps"] = (
            self._ambiguous_contact_substeps
        )
        arrays = {key: np.stack(value, axis=0) for key, value in traces.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise N1BallCoreError("N1 trace contains non-finite values")
        metadata = {
            "kind": TRACE_KIND,
            "policy_ticks": counters["policy_ticks"],
            "plant_binding_sha256": self.binding.binding_sha256,
            "scene_binding_sha256": self.scene_binding_sha256,
            "question_sha256": question.source_sha256,
            "robot_tape_sha256": robot_tape.source_sha256,
        }
        trace_sha = single_env._trace_content_sha256(arrays, metadata)
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "status": (
                "DIAGNOSTIC_MANUAL_NATIVE_BALL_PROBE_COMPLETE"
                if question.authority["kind"]
                == "manual_native_gravity_engineering_probe"
                else "DIAGNOSTIC_IMMUTABLE_QUESTION_EXPLICIT_LAUNCH_COMPLETE"
            ),
            "diagnostic_unauthorized": True,
            "authorization": {
                "training": False,
                "promotion": False,
                "deployment": False,
                "hardware": False,
            },
            "runtime": {
                "mujoco_version": str(getattr(self.mujoco, "__version__", "unknown")),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "model_options": {
                    "timestep_s": float(self.model.opt.timestep),
                    "integrator": int(self.model.opt.integrator),
                    "solver": int(self.model.opt.solver),
                    "iterations": int(self.model.opt.iterations),
                    "ls_iterations": int(getattr(self.model.opt, "ls_iterations", -1)),
                    "tolerance": float(self.model.opt.tolerance),
                    "disableflags": int(self.model.opt.disableflags),
                    "enableflags": int(self.model.opt.enableflags),
                },
            },
            "lineage": {
                "plant_contract_path": self.binding.source_path,
                "plant_contract_sha256": self.binding.source_sha256,
                "plant_binding_sha256": self.binding.binding_sha256,
                "canonical_mjcf_sha256": self.scene.canonical_xml_sha256,
                "physical_ball_scene": self.scene.binding,
                "robot_fixed_tape_path": robot_tape.source_path,
                "robot_fixed_tape_sha256": robot_tape.source_sha256,
                "question_path": question.source_path,
                "question_sha256": question.source_sha256,
                "question_id": question.question_id,
                "question_authority": dict(question.authority),
                "trace_content_sha256": trace_sha,
            },
            "counters": counters,
            "events": [dict(value) for value in self._events],
            "observation_contract": {
                "format": "purpose_grouped_not_final_flat_ABI",
                "ordered_groups": list(self.observation_groups()),
                "desired_at_contact_present": False,
                "teacher_source": "separate_robot_fixed_tape",
                "actual_outcome_privileged_only": True,
            },
            "actual_contact_eligibility": {
                "valid_actual_contact": (
                    self._racket_contact_edges == 1
                    and not self._contact_invalid_reasons
                ),
                "valid_achieved_outgoing_flight": (
                    self._racket_contact_edges == 1
                    and self._outgoing_state is not None
                    and not self._contact_invalid_reasons
                ),
                "racket_contact_edge_count": self._racket_contact_edges,
                "invalid_reasons": sorted(self._contact_invalid_reasons),
                "outgoing_state": self._outgoing_state,
                "reward_paid": False,
            },
            "known_limits": {
                "reward": "not_implemented",
                "vecenv": "not_implemented",
                "ppo": "not_implemented",
                "checkpoint": "not_implemented",
                "normalizer": "not_implemented",
                "aerodynamics_and_magnus": "not_implemented",
                "contact_calibration": "not_authorized_native_defaults",
            },
        }
        receipt["content_sha256"] = _sha256(_canonical_json_bytes(receipt))
        return arrays, receipt


def _triple(value: str) -> tuple[float, float, float]:
    try:
        out = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected x,y,z") from exc
    if len(out) != 3 or not all(math.isfinite(item) for item in out):
        raise argparse.ArgumentTypeError("expected three finite comma-separated scalars")
    return out


def _pair(value: str) -> tuple[float, float]:
    try:
        out = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected x,y") from exc
    if len(out) != 2 or not all(math.isfinite(item) for item in out):
        raise argparse.ArgumentTypeError("expected two finite comma-separated scalars")
    return out


def _write_trace(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    single_env._write_new_bytes(path, raw)
    return _sha256(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-question")
    make.add_argument("--contract", type=Path, required=True)
    make.add_argument("--mjcf", type=Path, default=single_env.DEFAULT_MJCF)
    make.add_argument("--question-id", default="n1_center_000")
    make.add_argument("--birth-position", type=_triple, required=True)
    make.add_argument("--birth-velocity", type=_triple, required=True)
    make.add_argument("--landing-aim", type=_pair, required=True)
    make.add_argument("--time-to-contact", type=float, required=True)
    make.add_argument("--out", type=Path, required=True)

    immutable = sub.add_parser("make-from-immutable")
    immutable.add_argument("--contract", type=Path, required=True)
    immutable.add_argument("--mjcf", type=Path, default=single_env.DEFAULT_MJCF)
    immutable.add_argument("--immutable-tape", type=Path, required=True)
    immutable.add_argument("--expected-immutable-tape-sha256", required=True)
    immutable.add_argument("--target-recipe", required=True)
    immutable.add_argument("--physical-launch-position", type=_triple, required=True)
    immutable.add_argument("--physical-launch-velocity", type=_triple, required=True)
    immutable.add_argument("--out", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--mjcf", type=Path, default=single_env.DEFAULT_MJCF)
    run.add_argument("--robot-tape", type=Path, required=True)
    run.add_argument("--expected-robot-tape-sha256", required=True)
    run.add_argument("--question", type=Path, required=True)
    run.add_argument("--expected-question-sha256", required=True)
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        binding = single_env.load_plant_binding(args.contract)
        if args.command in ("make-question", "make-from-immutable"):
            try:
                import mujoco
            except ImportError as exc:
                raise N1BallCoreError(
                    "make-question requires mujoco to bind the compiled scene"
                ) from exc
            ball_contract = physical_ball_scene.load_ball_contract(args.contract)
            scene = physical_ball_scene.compile_physical_ball_scene(
                mujoco,
                mjcf_path=args.mjcf,
                ball_contract=ball_contract,
                strict_pair_filter=True,
                include_floor_pair=True,
            )
            if args.command == "make-question":
                payload = build_question_payload(
                    question_id=args.question_id,
                    scene_binding_sha256=scene.binding["binding_sha256"],
                    birth_position_w_m=args.birth_position,
                    birth_linear_velocity_w_mps=args.birth_velocity,
                    landing_aim_xy_w_m=args.landing_aim,
                    nominal_time_to_contact_s=args.time_to_contact,
                )
            else:
                payload = build_question_from_immutable_tape(
                    immutable_tape_path=args.immutable_tape,
                    expected_immutable_tape_sha256=(
                        args.expected_immutable_tape_sha256
                    ),
                    target_recipe=args.target_recipe,
                    scene_binding_sha256=scene.binding["binding_sha256"],
                    physical_launch_position_w_m=args.physical_launch_position,
                    physical_launch_velocity_w_mps=args.physical_launch_velocity,
                )
            sha = write_question(args.out, payload)
            print(json.dumps({"question_sha256": sha, **payload}, indent=2))
            return 0
        core = MujocoN1BallCore(binding, mjcf_path=args.mjcf)
        if _sha256(args.robot_tape.expanduser().resolve().read_bytes()) != (
            args.expected_robot_tape_sha256
        ):
            raise N1BallCoreError("robot tape file SHA differs from external authority")
        robot_tape = single_env.load_fixed_tape(args.robot_tape, binding)
        question = load_question(
            args.question,
            expected_file_sha256=args.expected_question_sha256,
            scene_binding_sha256=core.scene_binding_sha256,
        )
        arrays, receipt = core.run_tape(robot_tape=robot_tape, question=question)
        trace_path = args.trace.expanduser().resolve()
        receipt_path = args.receipt.expanduser().resolve()
        trace_sha = _write_trace(trace_path, arrays)
        receipt["lineage"]["trace_file_sha256"] = trace_sha
        receipt.pop("content_sha256")
        receipt["content_sha256"] = _sha256(_canonical_json_bytes(receipt))
        single_env._write_new_bytes(receipt_path, _canonical_json_bytes(receipt))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (
        N1BallCoreError,
        physical_ball_scene.PhysicalBallSceneError,
        single_env.ContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[mujoco-n1-ball-core][ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
