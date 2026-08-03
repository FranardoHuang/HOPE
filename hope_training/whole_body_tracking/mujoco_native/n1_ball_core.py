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
import copy
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
from . import n1_reward_event_kernel
from . import observed_outcome_resolver
from . import selected_rubber_classifier
from . import single_env


QUESTION_KIND = "a3_mujoco_n1_physical_launch_probe_v1"
RECEIPT_KIND = "a3_mujoco_n1_ball_core_receipt_v2"
TRACE_KIND = "a3_mujoco_n1_ball_core_trace_v1"
PHASE_FIDELITY_REFERENCE_TAPE_KIND = (
    "a3_mujoco_phase_fidelity_reference_tape_v1"
)
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
    selected_rubber_action_lineage: Mapping[str, Any] | None


@dataclass(frozen=True)
class PhaseFidelityReferenceRow:
    motion_phase_context: str
    in_hold: bool
    reference_terminations_enabled: bool
    reference_anchor_pos_z_w_m: float
    reference_anchor_projected_gravity_b_z: float
    reference_ee_body_pos_z_w_m: tuple[float, ...]


@dataclass(frozen=True)
class PhaseFidelityReferenceTape:
    """Externally sealed post-control-step MotionCommand reference rows."""

    source_path: str
    source_sha256: str
    content_sha256: str
    sample_contract_sha256: str
    plant_binding_sha256: str
    scene_binding_sha256: str
    robot_tape_sha256: str
    anchor_body_name: str
    ee_body_order: tuple[str, ...]
    rows: tuple[PhaseFidelityReferenceRow, ...]
    authority_source_sha256: str


def _plain_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise N1BallCoreError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _phase_sample_contract_fields(
    sample_contract: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(sample_contract, Mapping):
        raise N1BallCoreError("phase sample contract must be a mapping")
    contract_sha = _plain_sha256(
        sample_contract.get("content_sha256"),
        "phase sample contract content_sha256",
    )
    unsigned = dict(sample_contract)
    unsigned.pop("content_sha256", None)
    if _sha256(_canonical_json_bytes(unsigned)) != contract_sha:
        raise N1BallCoreError("phase sample contract content seal differs")
    if sample_contract.get("kind") != "a3_mujoco_phase_fidelity_sample_contract_v1":
        raise N1BallCoreError("phase sample contract kind differs")
    contexts = sample_contract.get("motion_phase_contexts")
    body_order = sample_contract.get("ee_body_order")
    if (
        not isinstance(contexts, list)
        or set(contexts)
        != {"non_hold_swing_or_follow_through", "recovery_hold"}
        or len(contexts) != 2
    ):
        raise N1BallCoreError("phase sample contract contexts differ")
    if (
        not isinstance(body_order, list)
        or len(body_order) != 4
        or len(set(body_order)) != 4
        or any(not isinstance(name, str) or not name for name in body_order)
    ):
        raise N1BallCoreError("phase sample contract body order differs")
    return contract_sha, tuple(contexts), tuple(body_order)


def _phase_reference_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    contexts: tuple[str, ...],
    ee_body_order: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    expected_keys = {
        "motion_phase_context",
        "in_hold",
        "reference_terminations_enabled",
        "reference_anchor_pos_z_w_m",
        "reference_anchor_projected_gravity_b_z",
        "reference_ee_body_pos_z_w_m",
    }
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise N1BallCoreError("phase reference tape must contain at least one row")
    normalized: list[dict[str, Any]] = []
    frozen_gate: bool | None = None
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise N1BallCoreError(
                f"phase reference row {index} keys differ from schema"
            )
        context = raw["motion_phase_context"]
        in_hold = raw["in_hold"]
        enabled = raw["reference_terminations_enabled"]
        if context not in contexts or type(in_hold) is not bool or type(enabled) is not bool:
            raise N1BallCoreError(f"phase reference row {index} gates differ")
        if in_hold != (context == "recovery_hold"):
            raise N1BallCoreError(
                f"phase reference row {index} hold/context disagree"
            )
        if frozen_gate is None:
            frozen_gate = enabled
        elif enabled != frozen_gate:
            raise N1BallCoreError(
                "phase reference termination gate must be episode-frozen"
            )
        anchor_z = raw["reference_anchor_pos_z_w_m"]
        projected_z = raw["reference_anchor_projected_gravity_b_z"]
        if isinstance(anchor_z, bool) or isinstance(projected_z, bool):
            raise N1BallCoreError(f"phase reference row {index} scalars must be finite")
        anchor_z = float(anchor_z)
        projected_z = float(projected_z)
        if (
            not math.isfinite(anchor_z)
            or not math.isfinite(projected_z)
            or not -1.0 <= projected_z <= 1.0
        ):
            raise N1BallCoreError(
                f"phase reference row {index} scalars must be finite/physical"
            )
        ee_z = _vector(
            raw["reference_ee_body_pos_z_w_m"],
            len(ee_body_order),
            f"phase reference row {index} ee body z",
        )
        normalized.append(
            {
                "motion_phase_context": str(context),
                "in_hold": in_hold,
                "reference_terminations_enabled": enabled,
                "reference_anchor_pos_z_w_m": anchor_z,
                "reference_anchor_projected_gravity_b_z": projected_z,
                "reference_ee_body_pos_z_w_m": ee_z.tolist(),
            }
        )
    return tuple(normalized)


def build_phase_fidelity_reference_tape_payload(
    *,
    sample_contract: Mapping[str, Any],
    plant_binding_sha256: str,
    scene_binding_sha256: str,
    robot_tape_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    authority_source_sha256: str,
) -> dict[str, Any]:
    contract_sha, contexts, body_order = _phase_sample_contract_fields(
        sample_contract
    )
    normalized_rows = _phase_reference_rows(
        rows, contexts=contexts, ee_body_order=body_order
    )
    payload = {
        "schema_version": 1,
        "kind": PHASE_FIDELITY_REFERENCE_TAPE_KIND,
        "sample_contract_sha256": contract_sha,
        "plant_binding_sha256": _plain_sha256(
            plant_binding_sha256, "phase tape plant binding"
        ),
        "scene_binding_sha256": _plain_sha256(
            scene_binding_sha256, "phase tape scene binding"
        ),
        "robot_tape_sha256": _plain_sha256(
            robot_tape_sha256, "phase tape robot tape"
        ),
        "sample_timing": "post_control_step",
        "anchor_body_name": "pelvis_link",
        "ee_body_order": list(body_order),
        "authority": {
            "kind": "external_isaac_motion_command_phase_reference_v1",
            "source_artifact_sha256": _plain_sha256(
                authority_source_sha256, "phase tape authority source"
            ),
        },
        "rows": list(normalized_rows),
        "diagnostic_unauthorized": True,
    }
    payload["content_sha256"] = _sha256(_canonical_json_bytes(payload))
    return payload


def write_phase_fidelity_reference_tape(
    path: Path | str, payload: Mapping[str, Any]
) -> str:
    raw = _canonical_json_bytes(payload)
    single_env._write_new_bytes(Path(path).expanduser().resolve(), raw)
    return _sha256(raw)


def load_phase_fidelity_reference_tape(
    path: Path | str,
    *,
    expected_file_sha256: str,
    sample_contract: Mapping[str, Any],
) -> PhaseFidelityReferenceTape:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise N1BallCoreError(f"cannot read strict phase reference tape: {exc}") from exc
    if _sha256(raw) != _plain_sha256(
        expected_file_sha256, "phase tape expected file SHA"
    ):
        raise N1BallCoreError("phase reference tape file SHA differs from authority")
    expected_keys = {
        "schema_version",
        "kind",
        "sample_contract_sha256",
        "plant_binding_sha256",
        "scene_binding_sha256",
        "robot_tape_sha256",
        "sample_timing",
        "anchor_body_name",
        "ee_body_order",
        "authority",
        "rows",
        "diagnostic_unauthorized",
        "content_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise N1BallCoreError("phase reference tape top-level keys differ")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != PHASE_FIDELITY_REFERENCE_TAPE_KIND
        or payload.get("sample_timing") != "post_control_step"
        or payload.get("anchor_body_name") != "pelvis_link"
        or payload.get("diagnostic_unauthorized") is not True
    ):
        raise N1BallCoreError("phase reference tape schema semantics differ")
    content_sha = _plain_sha256(
        payload["content_sha256"], "phase tape content SHA"
    )
    unsigned = dict(payload)
    unsigned.pop("content_sha256")
    if _sha256(_canonical_json_bytes(unsigned)) != content_sha:
        raise N1BallCoreError("phase reference tape content seal differs")
    contract_sha, contexts, body_order = _phase_sample_contract_fields(
        sample_contract
    )
    if payload["sample_contract_sha256"] != contract_sha:
        raise N1BallCoreError("phase reference tape binds a different sample contract")
    if payload["ee_body_order"] != list(body_order):
        raise N1BallCoreError("phase reference tape body order differs")
    authority = payload.get("authority")
    if not isinstance(authority, dict) or set(authority) != {
        "kind",
        "source_artifact_sha256",
    } or authority.get("kind") != "external_isaac_motion_command_phase_reference_v1":
        raise N1BallCoreError("phase reference tape authority differs")
    authority_sha = _plain_sha256(
        authority["source_artifact_sha256"], "phase tape authority source"
    )
    normalized_rows = _phase_reference_rows(
        payload["rows"], contexts=contexts, ee_body_order=body_order
    )
    immutable_rows = tuple(
        PhaseFidelityReferenceRow(
            motion_phase_context=row["motion_phase_context"],
            in_hold=row["in_hold"],
            reference_terminations_enabled=row[
                "reference_terminations_enabled"
            ],
            reference_anchor_pos_z_w_m=row["reference_anchor_pos_z_w_m"],
            reference_anchor_projected_gravity_b_z=row[
                "reference_anchor_projected_gravity_b_z"
            ],
            reference_ee_body_pos_z_w_m=tuple(
                row["reference_ee_body_pos_z_w_m"]
            ),
        )
        for row in normalized_rows
    )
    return PhaseFidelityReferenceTape(
        source_path=str(source),
        source_sha256=_sha256(raw),
        content_sha256=content_sha,
        sample_contract_sha256=contract_sha,
        plant_binding_sha256=_plain_sha256(
            payload["plant_binding_sha256"], "phase tape plant binding"
        ),
        scene_binding_sha256=_plain_sha256(
            payload["scene_binding_sha256"], "phase tape scene binding"
        ),
        robot_tape_sha256=_plain_sha256(
            payload["robot_tape_sha256"], "phase tape robot tape"
        ),
        anchor_body_name="pelvis_link",
        ee_body_order=body_order,
        rows=immutable_rows,
        authority_source_sha256=authority_sha,
    )


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
        (
            "immutable_n1_tape_with_explicit_native_launch_and_"
            "selected_rubber_v2"
        ),
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
    action_manifest_path: Path | str,
    selected_rubber_classifier_binding: Mapping[str, Any],
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
    source_receipt = tape.source_receipt
    target = tape.targets[target_recipe]
    raw_sign = getattr(source_receipt, "mount_normal_sign", None)
    if (
        isinstance(raw_sign, bool)
        or not isinstance(raw_sign, (int, float))
        or not math.isfinite(float(raw_sign))
        or float(raw_sign) not in (-1.0, 1.0)
    ):
        raise N1BallCoreError(
            "immutable tape source receipt has no exact mount_normal_sign"
        )
    target_sign = target.runtime_target.get("mount_normal_sign")
    target_geometry_sha = target.runtime_target.get("geometry_source_sha256")
    if (
        isinstance(target_sign, bool)
        or not isinstance(target_sign, (int, float))
        or not math.isfinite(float(target_sign))
        or float(target_sign) != float(raw_sign)
        or target_geometry_sha != getattr(
            source_receipt, "geometry_source_sha256", None
        )
    ):
        raise N1BallCoreError(
            "immutable tape target and source receipt disagree on measured mount geometry"
        )
    try:
        selected_rubber_lineage = selected_rubber_classifier.bind_action_manifest(
            manifest_path=action_manifest_path,
            expected_manifest_sha256=str(source_receipt.manifest_sha256),
            action_uid=int(source_receipt.action_uid),
            motion_sha256=str(source_receipt.motion_sha256),
            mount_normal_sign=int(raw_sign),
            geometry_source_sha256=str(source_receipt.geometry_source_sha256),
            physics_sha256=str(source_receipt.physics_sha256),
            classifier_binding=selected_rubber_classifier_binding,
        )
    except selected_rubber_classifier.SelectedRubberClassifierError as exc:
        raise N1BallCoreError(
            f"immutable question cannot bind selected-rubber authority: {exc}"
        ) from exc
    authority = {
        "kind": (
            "immutable_n1_tape_with_explicit_native_launch_and_"
            "selected_rubber_v2"
        ),
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
        "selected_rubber_action_lineage": selected_rubber_lineage,
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
    selected_rubber_classifier_binding: Mapping[str, Any] | None = None,
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
    elif authority_kind in {
        "immutable_n1_tape_with_explicit_native_launch",
        (
            "immutable_n1_tape_with_explicit_native_launch_and_"
            "selected_rubber_v2"
        ),
    }:
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
        if authority_kind.endswith("selected_rubber_v2"):
            required_authority.add("selected_rubber_action_lineage")
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
    selected_rubber_action_lineage = None
    if authority_kind.endswith("selected_rubber_v2"):
        if selected_rubber_classifier_binding is None:
            raise N1BallCoreError(
                "selected-rubber question requires current classifier binding"
            )
        try:
            selected_rubber_action_lineage = (
                selected_rubber_classifier.validate_action_lineage(
                    authority["selected_rubber_action_lineage"],
                    classifier_binding=selected_rubber_classifier_binding,
                )
            )
        except selected_rubber_classifier.SelectedRubberClassifierError as exc:
            raise N1BallCoreError(
                f"selected-rubber question lineage is invalid: {exc}"
            ) from exc
        target = tape.targets[authority["target_recipe"]]
        source_receipt = tape.source_receipt
        if (
            selected_rubber_action_lineage["action_uid"]
            != int(source_receipt.action_uid)
            or selected_rubber_action_lineage["mount_normal_sign"]
            != int(source_receipt.mount_normal_sign)
            or selected_rubber_action_lineage["geometry_source_sha256"]
            != str(source_receipt.geometry_source_sha256)
            or target.runtime_target.get("mount_normal_sign")
            != source_receipt.mount_normal_sign
            or target.runtime_target.get("geometry_source_sha256")
            != source_receipt.geometry_source_sha256
        ):
            raise N1BallCoreError(
                "selected-rubber action lineage differs from immutable tape"
            )
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
        selected_rubber_action_lineage=(
            None
            if selected_rubber_action_lineage is None
            else dict(selected_rubber_action_lineage)
        ),
    )


class MujocoN1BallCore:
    """One physical-ball scene around the existing exact plant/action core."""

    def __init__(
        self,
        binding: single_env.PlantBinding,
        *,
        mjcf_path: Path | str = single_env.DEFAULT_MJCF,
        phase_fidelity_reference_tape: PhaseFidelityReferenceTape | None = None,
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
        try:
            self.selected_rubber_classifier_binding = (
                selected_rubber_classifier.build_classifier_binding(
                    scene_binding=scene_binding,
                    mjcf_path=mjcf_path,
                )
            )
        except selected_rubber_classifier.SelectedRubberClassifierError as exc:
            raise N1BallCoreError(
                f"selected-rubber classifier authority is invalid: {exc}"
            ) from exc
        try:
            self.observed_outcome_resolver_binding = (
                observed_outcome_resolver.build_resolver_binding(
                    scene_binding=scene_binding,
                    obstacle_rows=self.scene.obstacle_rows,
                    plant_binding_sha256=binding.binding_sha256,
                    policy_step_dt_s=binding.policy_step_dt_s,
                    control_decimation=binding.control_decimation,
                )
            )
            observed_outcome_resolver.validate_resolver_binding(
                self.observed_outcome_resolver_binding,
                expected_scene_binding=scene_binding,
                expected_obstacle_rows=self.scene.obstacle_rows,
                expected_plant_binding_sha256=binding.binding_sha256,
                expected_policy_step_dt_s=binding.policy_step_dt_s,
                expected_control_decimation=binding.control_decimation,
                expected_resolver_source_sha256=(
                    n1_reward_event_kernel.EXPECTED_OBSERVED_OUTCOME_RESOLVER_SOURCE_SHA256
                ),
            )
        except observed_outcome_resolver.ObservedOutcomeResolverError as exc:
            raise N1BallCoreError(
                f"observed-outcome resolver authority is invalid: {exc}"
            ) from exc
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
        self.phase_fidelity_reference_tape = phase_fidelity_reference_tape
        if phase_fidelity_reference_tape is not None:
            self.phase_fidelity_sample_contract_sha256 = (
                phase_fidelity_reference_tape.sample_contract_sha256
            )
            if phase_fidelity_reference_tape.anchor_body_name != "pelvis_link":
                raise N1BallCoreError("phase reference anchor must be pelvis_link")
            self._phase_ee_body_ids = tuple(
                single_env._named_id(
                    mujoco,
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                    f"phase-fidelity body {name}",
                )
                for name in phase_fidelity_reference_tape.ee_body_order
            )
            if len(set(self._phase_ee_body_ids)) != len(self._phase_ee_body_ids):
                raise N1BallCoreError("phase reference body order resolves duplicates")
        self._racket_geom_id = single_env._named_id(
            mujoco,
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            physical_ball_scene.RACKET_GEOM_NAME,
            "racket geom",
        )
        self._racket_site_id = single_env._named_id(
            mujoco,
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            selected_rubber_classifier.RACKET_SITE_NAME,
            "official racket site",
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
        self._first_racket_contact_stamp: dict[str, int] | None = None
        self._first_racket_contact_classification: dict[str, Any] | None = None
        self._selected_rubber_action_lineage: dict[str, Any] | None = None
        self._outgoing_state: dict[str, Any] | None = None
        self._observed_outcome_resolver: (
            observed_outcome_resolver.ObservedOutcomeResolver | None
        ) = None
        self._contact_invalid_reasons: set[str] = set()
        self.native_physical_event_contract_sha256 = (
            n1_reward_event_kernel.native_physical_event_facts_contract()[
                "content_sha256"
            ]
        )
        native_source_sha256 = _sha256(Path(__file__).read_bytes())
        if native_source_sha256 != (
            n1_reward_event_kernel.EXPECTED_N1_BALL_CORE_SOURCE_SHA256
        ):
            raise N1BallCoreError(
                "native N1 core source differs from external kernel authority"
            )
        self._native_physical_event_source_binding = (
            n1_reward_event_kernel.SourceBinding(
                source_id="mujoco_native/n1_ball_core.py",
                source_sha256=native_source_sha256,
                event_contract_sha256=(
                    self.native_physical_event_contract_sha256
                ),
            )
        )

    @property
    def scene_binding_sha256(self) -> str:
        return str(self.scene.binding["binding_sha256"])

    @property
    def native_physical_event_source_binding(
        self,
    ) -> n1_reward_event_kernel.SourceBinding:
        return self._native_physical_event_source_binding

    @property
    def observed_outcome_question_binding_sha256(self) -> str | None:
        resolver = self._observed_outcome_resolver
        if resolver is None:
            return None
        return str(resolver.question_binding["content_sha256"])

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
        outgoing_created = False
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
            if self._first_racket_contact_stamp is None:
                self._first_racket_contact_stamp = {
                    "policy_tick": self.policy_tick,
                    "physics_substep": int(substep_index),
                }
                if self._selected_rubber_action_lineage is not None:
                    try:
                        classification = (
                            selected_rubber_classifier.classify_observed_generic_blade_contact(
                                ball_center_w_m=np.asarray(
                                    self.data.xpos[self.scene.ball_body_id],
                                    dtype=np.float64,
                                ),
                                racket_site_position_w_m=np.asarray(
                                    self.data.site_xpos[self._racket_site_id],
                                    dtype=np.float64,
                                ),
                                racket_rotation_w_from_local=np.asarray(
                                    self.data.site_xmat[self._racket_site_id],
                                    dtype=np.float64,
                                ).reshape(3, 3),
                                action_lineage=self._selected_rubber_action_lineage,
                                classifier_binding=(
                                    self.selected_rubber_classifier_binding
                                ),
                                policy_tick=self.policy_tick,
                                physics_substep=int(substep_index),
                            )
                        )
                    except (
                        selected_rubber_classifier.SelectedRubberClassifierError
                    ) as exc:
                        raise N1BallCoreError(
                            f"selected-rubber contact classification failed: {exc}"
                        ) from exc
                    self._first_racket_contact_classification = classification
                    if classification["status"] == (
                        selected_rubber_classifier.STATUS_EDGE_RIM_AMBIGUOUS
                    ):
                        self._contact_invalid_reasons.add(
                            "racket_contact_edge_or_rim_ambiguous"
                        )
                    elif classification["status"] == (
                        selected_rubber_classifier.STATUS_BETWEEN_PLANES_AMBIGUOUS
                    ):
                        self._contact_invalid_reasons.add(
                            "racket_contact_between_outer_planes_ambiguous"
                        )
            if self._racket_contact_edges > 1:
                self._contact_invalid_reasons.add("racket_recontact")
        if racket_was_active and not racket_is_active and self._outgoing_state is None:
            dof = self.scene.ball_dof_adr
            qpos = self.scene.ball_qpos_adr
            self._outgoing_state = {
                "policy_tick": self.policy_tick,
                "physics_substep": int(substep_index),
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
            outgoing_created = True
        for label in sorted(labels - self._active_contact_labels):
            self._events.append(
                {
                    "policy_tick": self.policy_tick,
                    "physics_substep": int(substep_index),
                    "time_s": float(self.data.time),
                    "event": label,
                }
            )
        resolver = self._observed_outcome_resolver
        if resolver is not None:
            try:
                if outgoing_created:
                    resolver.arm(
                        self._outgoing_state,
                        active_contact_labels=labels,
                    )
                elif resolver.armed:
                    resolver.observe_substep(
                        policy_tick=self.policy_tick,
                        physics_substep=int(substep_index),
                        time_s=float(self.data.time),
                        ball_center_w_m=np.asarray(
                            self.data.xpos[self.scene.ball_body_id],
                            dtype=np.float64,
                        ),
                        active_contact_labels=labels,
                    )
            except observed_outcome_resolver.ObservedOutcomeResolverError as exc:
                raise N1BallCoreError(
                    f"observed native outcome resolution failed: {exc}"
                ) from exc
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
        if question.selected_rubber_action_lineage is None:
            selected_rubber_lineage = None
        else:
            try:
                selected_rubber_lineage = (
                    selected_rubber_classifier.validate_action_lineage(
                        question.selected_rubber_action_lineage,
                        classifier_binding=(
                            self.selected_rubber_classifier_binding
                        ),
                    )
                )
            except selected_rubber_classifier.SelectedRubberClassifierError as exc:
                raise N1BallCoreError(
                    f"question selected-rubber lineage differs from core: {exc}"
                ) from exc
        try:
            outcome_question_binding = observed_outcome_resolver.bind_question(
                resolver_binding=self.observed_outcome_resolver_binding,
                question_source_sha256=question.source_sha256,
                landing_aim_xy_w_m=question.landing_aim_xy_w_m,
                action_lineage_sha256=(
                    None
                    if selected_rubber_lineage is None
                    else selected_rubber_lineage["content_sha256"]
                ),
            )
            outcome_question_binding = (
                observed_outcome_resolver.validate_question_binding(
                    outcome_question_binding,
                    resolver_binding=self.observed_outcome_resolver_binding,
                    expected_question_source_sha256=question.source_sha256,
                    expected_landing_aim_xy_w_m=question.landing_aim_xy_w_m,
                    expected_action_lineage_sha256=(
                        None
                        if selected_rubber_lineage is None
                        else selected_rubber_lineage["content_sha256"]
                    ),
                )
            )
            outcome_resolver = observed_outcome_resolver.ObservedOutcomeResolver(
                resolver_binding=self.observed_outcome_resolver_binding,
                question_binding=outcome_question_binding,
            )
        except observed_outcome_resolver.ObservedOutcomeResolverError as exc:
            raise N1BallCoreError(
                f"question cannot bind observed-outcome resolver: {exc}"
            ) from exc
        phase_tape = self.phase_fidelity_reference_tape
        if phase_tape is not None:
            if phase_tape.plant_binding_sha256 != self.binding.binding_sha256:
                raise N1BallCoreError("phase reference tape and plant binding differ")
            if phase_tape.scene_binding_sha256 != self.scene_binding_sha256:
                raise N1BallCoreError("phase reference tape and scene binding differ")
            if phase_tape.robot_tape_sha256 != robot_tape.source_sha256:
                raise N1BallCoreError("phase reference tape and robot tape SHA differ")
            if len(phase_tape.rows) != int(robot_tape.actions.shape[0]):
                raise N1BallCoreError(
                    "phase reference tape row count differs from robot tape"
                )
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
        self._first_racket_contact_stamp = None
        self._first_racket_contact_classification = None
        self._selected_rubber_action_lineage = selected_rubber_lineage
        self._outgoing_state = None
        self._observed_outcome_resolver = outcome_resolver
        self._contact_invalid_reasons = set()
        return self.observation_groups()

    def native_physical_event_facts(self) -> dict[str, Any]:
        """Return cumulative physical facts without claiming reward eligibility."""

        source = self.native_physical_event_source_binding
        outcome_resolver = self._observed_outcome_resolver
        return {
            "schema_version": 4,
            "kind": n1_reward_event_kernel.NATIVE_PHYSICAL_EVENT_FACTS_KIND,
            "source": {
                "source_id": source.source_id,
                "source_sha256": source.source_sha256,
                "event_contract_sha256": source.event_contract_sha256,
            },
            "policy_tick": self.policy_tick,
            "racket_contact_edge_count_total": self._racket_contact_edges,
            "first_racket_contact_stamp": (
                None
                if self._first_racket_contact_stamp is None
                else dict(self._first_racket_contact_stamp)
            ),
            "outgoing_flight": (
                None
                if self._outgoing_state is None
                else copy.deepcopy(self._outgoing_state)
            ),
            "invalid_reasons": sorted(self._contact_invalid_reasons),
            "selected_rubber_authority_available": (
                self._selected_rubber_action_lineage is not None
            ),
            "selected_rubber_action_lineage": (
                None
                if self._selected_rubber_action_lineage is None
                else copy.deepcopy(self._selected_rubber_action_lineage)
            ),
            "first_racket_contact_classification": (
                None
                if self._first_racket_contact_classification is None
                else copy.deepcopy(self._first_racket_contact_classification)
            ),
            "observed_outcome_authority_available": outcome_resolver is not None,
            "observed_outcome_resolver_binding": (
                None
                if outcome_resolver is None
                else copy.deepcopy(self.observed_outcome_resolver_binding)
            ),
            "observed_outcome_question_binding": (
                None
                if outcome_resolver is None
                else copy.deepcopy(outcome_resolver.question_binding)
            ),
            "observed_outcome_snapshot": (
                None if outcome_resolver is None else outcome_resolver.snapshot()
            ),
        }

    def _selected_rubber_contact_receipt(
        self, *, question: N1Question
    ) -> dict[str, Any]:
        """Package first-contact face evidence without inferring a face.

        A generic blade edge is not evidence of either rubber face.  This
        receipt section therefore remains explicitly unknown and fail-closed
        unless the observed first edge carries a sealed classifier result.
        """

        outcome_binding = getattr(
            self, "observed_outcome_resolver_binding", None
        )
        landing_aim = getattr(question, "landing_aim_xy_w_m", None)
        facts = n1_reward_event_kernel.validate_native_physical_event_facts(
            self.native_physical_event_facts(),
            expected_source=self.native_physical_event_source_binding,
            expected_outcome_resolver_binding_sha256=(
                None
                if outcome_binding is None
                else outcome_binding["content_sha256"]
            ),
            expected_outcome_question_binding_sha256=(
                self.observed_outcome_question_binding_sha256
            ),
            expected_outcome_scene_binding_sha256=(
                None if outcome_binding is None else self.scene_binding_sha256
            ),
            expected_outcome_plant_binding_sha256=(
                None
                if outcome_binding is None
                else self.binding.binding_sha256
            ),
            expected_question_source_sha256=question.source_sha256,
            expected_question_landing_aim_xy_w_m=(
                None
                if landing_aim is None
                else tuple(float(value) for value in landing_aim)
            ),
        )
        try:
            classifier_binding = selected_rubber_classifier.validate_classifier_binding(
                self.selected_rubber_classifier_binding
            )
        except selected_rubber_classifier.SelectedRubberClassifierError as exc:
            raise N1BallCoreError(
                f"selected-rubber classifier binding is invalid at receipt time: {exc}"
            ) from exc

        backend_identity = {
            "mujoco_backend_version": classifier_binding["mujoco_backend_version"],
            "compiled_mesh_closure_members": classifier_binding[
                "compiled_mesh_closure_members"
            ],
        }
        receipt = {
            "schema_version": 1,
            "kind": "a3_mujoco_n1_selected_rubber_contact_receipt_v1",
            "generic_racket_contact_observed": (
                facts["racket_contact_edge_count_total"] > 0
            ),
            "racket_contact_edge_count_total": facts[
                "racket_contact_edge_count_total"
            ],
            "selected_rubber_authority_available": facts[
                "selected_rubber_authority_available"
            ],
            "classifier_binding_sha256": classifier_binding["content_sha256"],
            "classifier_source_sha256": classifier_binding["classifier_source_sha256"],
            "question_sha256": question.source_sha256,
            "scene_binding_sha256": classifier_binding["scene_binding_sha256"],
            "assembled_xml_sha256": classifier_binding["assembled_xml_sha256"],
            "backend_identity": backend_identity,
            "backend_identity_sha256": _sha256(
                _canonical_json_bytes(backend_identity)
            ),
            "first_racket_contact_stamp": facts["first_racket_contact_stamp"],
            "invalid_reasons": list(facts["invalid_reasons"]),
            "classification": None,
            "classification_content_sha256": None,
            "policy_tick": None,
            "physics_substep": None,
            "observed_face_sign": None,
            "selected_rubber": None,
            "tangential_distance_from_face_center_m": None,
            "safe_ball_center_tangential_radius_m": None,
        }
        if facts["racket_contact_edge_count_total"] == 0:
            receipt.update(
                {
                    "status": "unknown_no_generic_racket_contact",
                    "fail_closed": True,
                }
            )
            return receipt

        classification = facts["first_racket_contact_classification"]
        lineage = facts["selected_rubber_action_lineage"]
        if classification is None or lineage is None:
            receipt.update(
                {
                    "status": (
                        "unknown_generic_racket_contact_without_"
                        "selected_rubber_classification"
                    ),
                    "fail_closed": True,
                }
            )
            return receipt

        try:
            sealed = selected_rubber_classifier.validate_classification_seal(
                classification,
                action_lineage=lineage,
            )
        except selected_rubber_classifier.SelectedRubberClassifierError as exc:
            raise N1BallCoreError(
                f"selected-rubber contact classification seal is invalid: {exc}"
            ) from exc
        stamp = facts["first_racket_contact_stamp"]
        if stamp is None or (
            sealed["policy_tick"] != stamp["policy_tick"]
            or sealed["physics_substep"] != stamp["physics_substep"]
        ):
            raise N1BallCoreError(
                "selected-rubber classification stamp differs from first racket edge"
            )
        if sealed["classifier_binding_sha256"] != classifier_binding["content_sha256"]:
            raise N1BallCoreError(
                "selected-rubber classification differs from receipt classifier binding"
            )
        receipt.update(
            {
                "status": "classified_generic_racket_contact",
                # An ambiguous classifier result remains fail-closed: it names
                # no face even though the generic blade edge was observed.
                "fail_closed": sealed["selected_rubber"] is None,
                "classification": sealed,
                "classification_content_sha256": sealed["content_sha256"],
                "policy_tick": sealed["policy_tick"],
                "physics_substep": sealed["physics_substep"],
                "observed_face_sign": sealed["observed_face_sign"],
                "selected_rubber": sealed["selected_rubber"],
                "tangential_distance_from_face_center_m": sealed[
                    "tangential_distance_from_face_center_m"
                ],
                "safe_ball_center_tangential_radius_m": sealed[
                    "safe_ball_center_tangential_radius_m"
                ],
            }
        )
        return receipt

    def _phase_fidelity_sample(self) -> dict[str, Any]:
        tape = self.phase_fidelity_reference_tape
        if tape is None:
            raise N1BallCoreError("phase reference tape is not installed")
        if not 0 <= self.policy_tick < len(tape.rows):
            raise N1BallCoreError("phase reference tape is exhausted")
        reference = tape.rows[self.policy_tick]
        pelvis_id = self.plant._pelvis_body_id
        pelvis_rotation = np.asarray(
            self.data.xmat[pelvis_id], dtype=np.float64
        ).reshape(3, 3)
        robot_anchor_z = float(self.data.xpos[pelvis_id, 2])
        robot_projected_gravity_z = -float(pelvis_rotation[2, 2])
        robot_ee_z = np.asarray(
            self.data.xpos[np.asarray(self._phase_ee_body_ids, dtype=np.int64), 2],
            dtype=np.float64,
        )
        if (
            not math.isfinite(robot_anchor_z)
            or not math.isfinite(robot_projected_gravity_z)
            or robot_ee_z.shape != (len(tape.ee_body_order),)
            or not np.isfinite(robot_ee_z).all()
        ):
            raise N1BallCoreError("native phase-fidelity robot state is non-finite")
        reference_ee_z = np.asarray(
            reference.reference_ee_body_pos_z_w_m, dtype=np.float64
        )
        return {
            "schema_version": 1,
            "kind": "a3_mujoco_phase_fidelity_sample_v1",
            "motion_phase_context": reference.motion_phase_context,
            "in_hold": reference.in_hold,
            "reference_terminations_enabled": (
                reference.reference_terminations_enabled
            ),
            "anchor_pos_z_error_m": abs(
                reference.reference_anchor_pos_z_w_m - robot_anchor_z
            ),
            "anchor_projected_gravity_z_error_abs": abs(
                reference.reference_anchor_projected_gravity_b_z
                - robot_projected_gravity_z
            ),
            "ee_body_pos_z_error_m": np.abs(
                reference_ee_z - robot_ee_z
            ).tolist(),
        }

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
        phase_fidelity_sample = (
            None
            if self.phase_fidelity_reference_tape is None
            else self._phase_fidelity_sample()
        )
        self.policy_tick += 1
        result = {
            "plant": row,
            "observation_groups": observation,
            "new_events": new_events,
            "native_physical_event_facts": self.native_physical_event_facts(),
        }
        if phase_fidelity_sample is not None:
            result["phase_fidelity_sample"] = phase_fidelity_sample
        return result

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
            "schema_version": 2,
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
                "phase_fidelity_reference_tape": (
                    None
                    if self.phase_fidelity_reference_tape is None
                    else {
                        "path": self.phase_fidelity_reference_tape.source_path,
                        "file_sha256": (
                            self.phase_fidelity_reference_tape.source_sha256
                        ),
                        "content_sha256": (
                            self.phase_fidelity_reference_tape.content_sha256
                        ),
                        "authority_source_sha256": (
                            self.phase_fidelity_reference_tape.authority_source_sha256
                        ),
                        "sample_contract_sha256": (
                            self.phase_fidelity_reference_tape.sample_contract_sha256
                        ),
                    }
                ),
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
            "selected_rubber_contact_classification": (
                self._selected_rubber_contact_receipt(question=question)
            ),
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
    immutable.add_argument("--action-manifest", type=Path, required=True)
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
    run.add_argument("--phase-fidelity-reference-tape", type=Path)
    run.add_argument("--expected-phase-fidelity-reference-tape-sha256")
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
            classifier_binding = selected_rubber_classifier.build_classifier_binding(
                scene_binding=scene.binding,
                mjcf_path=args.mjcf,
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
                    action_manifest_path=args.action_manifest,
                    selected_rubber_classifier_binding=classifier_binding,
                    scene_binding_sha256=scene.binding["binding_sha256"],
                    physical_launch_position_w_m=args.physical_launch_position,
                    physical_launch_velocity_w_mps=args.physical_launch_velocity,
                )
            sha = write_question(args.out, payload)
            print(json.dumps({"question_sha256": sha, **payload}, indent=2))
            return 0
        if _sha256(args.robot_tape.expanduser().resolve().read_bytes()) != (
            args.expected_robot_tape_sha256
        ):
            raise N1BallCoreError("robot tape file SHA differs from external authority")
        robot_tape = single_env.load_fixed_tape(args.robot_tape, binding)
        if (args.phase_fidelity_reference_tape is None) != (
            args.expected_phase_fidelity_reference_tape_sha256 is None
        ):
            raise N1BallCoreError(
                "phase reference tape path and expected SHA must be supplied together"
            )
        phase_reference_tape = None
        if args.phase_fidelity_reference_tape is not None:
            from . import vec_env

            phase_reference_tape = load_phase_fidelity_reference_tape(
                args.phase_fidelity_reference_tape,
                expected_file_sha256=(
                    args.expected_phase_fidelity_reference_tape_sha256
                ),
                sample_contract=vec_env.phase_fidelity_sample_contract(),
            )
        core = MujocoN1BallCore(
            binding,
            mjcf_path=args.mjcf,
            phase_fidelity_reference_tape=phase_reference_tape,
        )
        question = load_question(
            args.question,
            expected_file_sha256=args.expected_question_sha256,
            scene_binding_sha256=core.scene_binding_sha256,
            selected_rubber_classifier_binding=(
                core.selected_rubber_classifier_binding
            ),
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
        selected_rubber_classifier.SelectedRubberClassifierError,
        single_env.ContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[mujoco-n1-ball-core][ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
