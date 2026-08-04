#!/usr/bin/env python3
"""Offline, no-clobber producer for the five fixed-question N1 target arms.

The producer consumes one exact measured-N1 prepared core.  It reconstructs the
production sampler identity for one explicit seed, draws the already-collapsed centre
question once, solves the historical LM arm once, constructs the closed-form analytic
and teacher-at-hit carriers, and writes one canonical tape containing all five recipes.
Nothing in this script imports Isaac Lab or runs inside reset.

Outputs are diagnostic-only and are written into a brand-new directory.  A retry never
overwrites or reuses a partial directory.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parents[2]
MDP_RELATIVE = PurePosixPath(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
TRAINING_CONTRACT_RELATIVE = PurePosixPath(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py"
)
ACTION_ID = "take_061_unit04_bh"
ACTION_UID = 5527597793770800
MEASURED_UID = "Take_061_unit04_BH"
MOTION_PATH = (
    "assets/motions/chingmu73_measured_v4_20260803/"
    "hope_Take_061_unit04_BH.npz"
)
MOTION_SHA256 = "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
RECIPES = (
    "current_lm",
    "analytic_full",
    "analytic_no_velocity",
    "teacher_pos_face_no_velocity",
    "outcome_dense_only",
)
VALIDITY = {
    "current_lm": (True, True, True),
    "analytic_full": (True, True, True),
    "analytic_no_velocity": (True, False, True),
    "teacher_pos_face_no_velocity": (True, False, True),
    "outcome_dense_only": (False, False, False),
}
POLICY_DT_S = 0.02
ANALYTIC_FLIGHT_TIME_S = 0.66
REVERSE_RETURN_MIN_COSINE = 0.5
BUILD_REPORT_KIND = "measured_action_ball_n1_fixed_tape_build_report_v1"
DYNAMIC_READY_V2_KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
DYNAMIC_READY_V2_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "action_id",
        "robot",
        "authorization",
        "ready_source",
        "sources",
        "teacher_reference",
        "physical_birth_composition",
        "physical_birth_static_evidence",
        "physical_ready",
        "runtime_plant",
        "hold_candidate",
        "required_next_gate",
        "non_claims",
        "producer",
        "content_sha256",
    }
)
TEACHER_REFERENCE_KEYS = frozenset(
    {
        "semantics",
        "motion_sha256",
        "frame_index",
        "root_pos_w_m",
        "root_quat_wxyz",
        "joint_pos_rad",
    }
)
NOMINAL_HOLD_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "verdict",
        "action_id",
        "artifact",
        "motion_sha256",
        "teacher_reference_unchanged",
        "teacher_physical_birth_separated",
        "candidate_physical_birth_written",
        "candidate_hold_qdes_and_delay_history_installed",
        "plant_contract_match",
        "control_step_action_delay_runtime",
        "active_terminations",
        "requested_duration_s",
        "completed_duration_s",
        "completed_policy_steps",
        "completed_physics_steps",
        "terminal_reasons",
        "generic_terminated",
        "generic_truncated",
        "minimum_root_z_m",
        "maximum_root_tilt_rad",
        "both_feet_contact_fraction",
        "joint_safety_telemetry",
        "screenshots",
        "content_sha256",
    }
)
TARGET_REPLACEMENT_FIELDS = frozenset(
    {
        "racket_site_target_w_m",
        "mount_normal_sign",
        "racket_normal_w",
        "reference_racket_quat_wxyz",
        "reference_racket_angular_velocity_w_radps",
        "racket_command_quat_wxyz",
        "racket_face_center_velocity_w_mps",
        "racket_site_velocity_w_mps",
        "racket_command_angular_velocity_w_radps",
        "geometry_source_sha256",
        "reference_t_hit_s",
        "reference_t_cycle_s",
        "reference_racket_site_speed_mps",
        "required_racket_site_speed_mps",
        "reaction_margin_s",
        "teacher_rate_min",
        "teacher_rate_max",
        "teacher_rate",
        "scaled_t_hit_s",
        "scaled_t_cycle_s",
        "pre_swing_wait_s",
        "solver_residual_m",
    }
)


class ProducerError(RuntimeError):
    """Input identity, solver admission, or publication contract failed."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _sealed_content_sha256(document: Mapping[str, Any], *, label: str) -> str:
    seal = document.get("content_sha256")
    if (
        type(seal) is not str
        or len(seal) != 64
        or seal != seal.lower()
        or any(character not in "0123456789abcdef" for character in seal)
    ):
        raise ProducerError("%s content SHA is malformed" % label)
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    if _canonical_sha256(unsigned) != seal:
        raise ProducerError("%s content SHA is not reproducible" % label)
    return seal


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProducerError("cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_mdp_modules(root: Path) -> dict[str, Any]:
    mdp = root.joinpath(*MDP_RELATIVE.parts)
    package_name = "_measured_n1_fixed_tape_mdp"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(mdp)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    modules = {}
    for name in (
        "racket_contact_geometry",
        "virtual_ball",
        "counter_rally",
        "action_ball_sampling",
        "action_ball_manifest",
        "action_ball_profile_adapter",
        "stroke_prototypes_torch",
        "continuous_questions",
        "strike_spec_analytic",
        "action_ball_runtime",
        "action_ball_fixed_question_tape",
    ):
        modules[name] = _load_module(
            "%s.%s" % (package_name, name), mdp / (name + ".py")
        )
    return modules


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ProducerError("%s contains duplicate key %r" % (label, key))
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProducerError("%s contains non-finite %s" % (label, token))
            ),
        )
    except ProducerError:
        raise
    except Exception as exc:
        raise ProducerError("cannot read %s: %s" % (label, exc)) from exc
    if type(value) is not dict:
        raise ProducerError("%s root must be an object" % label)
    return value


def _repo_path(root: Path, value: str, expected_sha: str, *, label: str) -> Path:
    path = (root / value).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProducerError("%s escapes repo root" % label) from exc
    if _sha256_file(path) != expected_sha:
        raise ProducerError("%s SHA differs" % label)
    return path


def _quat_rotate(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    q = np.asarray((x, y, z), dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return v + 2.0 * (w * np.cross(q, v) + np.cross(q, np.cross(q, v)))


def _yaw_from_quat(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _normalized(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
    row = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(row))
    if row.shape != (3,) or not np.isfinite(row).all() or norm <= 1.0e-12:
        raise ProducerError("%s must be one finite nonzero vector" % label)
    return tuple(float(component) for component in row / norm)


def _motion_state(motion_path: Path, strike_phase: float, geometry: Any) -> dict[str, Any]:
    with np.load(str(motion_path), allow_pickle=False) as motion:
        names = [str(name) for name in motion["body_names"]]
        try:
            wrist = names.index("right_wrist_yaw_Link")
            pelvis = names.index("pelvis_link")
        except ValueError as exc:
            raise ProducerError("motion lacks wrist/pelvis body") from exc
        body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
        body_ang = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        measured_site = np.asarray(motion["measured_racket_site_pos_w"], dtype=np.float64)
        measured_normal = np.asarray(motion["measured_racket_normal_w"], dtype=np.float64)
        sign = int(np.asarray(motion["measured_racket_robot_mount_normal_sign"]).reshape(-1)[0])
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    frames = int(body_pos.shape[0])
    strike = round(float(strike_phase) * (frames - 1))
    if fps != 50.0 or not 2 <= strike <= frames - 3 or sign != 1:
        raise ProducerError("selected measured motion frame/fps/sign identity differs")
    reference_quat = tuple(float(value) for value in body_quat[strike, wrist])
    reference_omega = tuple(float(value) for value in body_ang[strike, wrist])
    official_site = np.stack(
        [
            body_pos[index, wrist]
            + _quat_rotate(
                body_quat[index, wrist], geometry.RACKET_SITE_OFFSET_WRIST_M
            )
            for index in range(frames)
        ],
        axis=0,
    )
    site_velocity = tuple(
        float(value)
        for value in (
            (official_site[strike + 2] - official_site[strike - 2])
            / (4.0 / fps)
        )
    )
    physical_normal = _normalized(
        sign * _quat_rotate(body_quat[strike, wrist], (0.0, 1.0, 0.0)),
        label="official FK strike normal",
    )
    if (
        float(np.linalg.norm(official_site[strike] - measured_site[strike]))
        > 0.05
        or float(
            np.dot(
                np.asarray(physical_normal),
                np.asarray(_normalized(measured_normal[strike], label="measured normal")),
            )
        )
        < math.cos(math.radians(5.0))
    ):
        raise ProducerError("official FK and measured strike teacher differ")
    raw_normal = tuple(float(value) / sign for value in physical_normal)
    face_velocity = geometry.face_center_velocity_from_site(
        site_velocity,
        reference_omega,
        reference_quat,
        sign,
    )
    teacher_root_quat = np.asarray(body_quat[0, pelvis], dtype=np.float64)
    teacher_root_quat /= np.linalg.norm(teacher_root_quat)
    return {
        "frames": frames,
        "strike_frame": strike,
        "teacher_root_pos": tuple(float(value) for value in body_pos[0, pelvis]),
        "teacher_root_quat": tuple(float(value) for value in teacher_root_quat),
        "teacher_joint_pos": tuple(float(value) for value in joint_pos[0]),
        "reference_quat": reference_quat,
        "reference_omega": reference_omega,
        "teacher_site_position": tuple(float(value) for value in official_site[strike]),
        "teacher_site_velocity": site_velocity,
        "teacher_face_velocity": tuple(float(value) for value in face_velocity),
        "teacher_raw_normal": raw_normal,
    }


def _vector_matches(
    left: object,
    right: Sequence[float],
    *,
    expected: int,
    absolute_tolerance: float = 5.0e-7,
) -> bool:
    return (
        type(left) is list
        and len(left) == expected
        and all(type(value) in (int, float) and math.isfinite(float(value)) for value in left)
        and all(
            math.isclose(
                float(lhs), float(rhs), rel_tol=0.0, abs_tol=absolute_tolerance
            )
            for lhs, rhs in zip(left, right)
        )
    )


def _dynamic_ready_source(
    root: Path,
    *,
    prepared: Mapping[str, Any],
    core: Mapping[str, Any],
    motion_path: Path,
    motion_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the prepared core's exact physical birth and keep teacher frame0 separate."""

    if prepared.get("claims", {}).get("dynamic_ready_status") != "PASS":
        raise ProducerError("prepared core lacks dynamic-ready plus nominal-hold PASS")
    dynamic = core.get("dynamic_ready")
    if type(dynamic) is not dict or set(dynamic) != {
        "artifact",
        "nominal_hold_receipt",
    }:
        raise ProducerError("prepared core dynamic-ready pins differ")
    for name in ("artifact", "nominal_hold_receipt"):
        pin = dynamic[name]
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise ProducerError("prepared core dynamic-ready %s pin differs" % name)

    artifact_path = _repo_path(
        root,
        dynamic["artifact"]["path"],
        dynamic["artifact"]["sha256"],
        label="schema-v2 dynamic-ready artifact",
    )
    receipt_path = _repo_path(
        root,
        dynamic["nominal_hold_receipt"]["path"],
        dynamic["nominal_hold_receipt"]["sha256"],
        label="dynamic-ready nominal-hold receipt",
    )
    artifact = _strict_json(artifact_path, label="schema-v2 dynamic-ready artifact")
    receipt = _strict_json(receipt_path, label="dynamic-ready nominal-hold receipt")
    if set(artifact) != DYNAMIC_READY_V2_KEYS:
        raise ProducerError("schema-v2 dynamic-ready artifact keys differ")
    if set(receipt) != NOMINAL_HOLD_RECEIPT_KEYS:
        raise ProducerError("dynamic-ready nominal-hold receipt keys differ")
    artifact_content_sha = _sealed_content_sha256(
        artifact, label="schema-v2 dynamic-ready artifact"
    )
    receipt_content_sha = _sealed_content_sha256(
        receipt, label="dynamic-ready nominal-hold receipt"
    )

    training_contract_path = root.joinpath(*TRAINING_CONTRACT_RELATIVE.parts)
    module_suffix = hashlib.sha256(str(training_contract_path).encode("utf-8")).hexdigest()[:16]
    training_contract = _load_module(
        "_fixed_tape_training_contract_%s" % module_suffix,
        training_contract_path,
    )
    try:
        binding = training_contract.load_action_ball_dynamic_ready_runtime_binding(
            artifact_path=str(artifact_path),
            artifact_sha256=dynamic["artifact"]["sha256"],
            nominal_hold_receipt_path=str(receipt_path),
            nominal_hold_receipt_sha256=dynamic["nominal_hold_receipt"]["sha256"],
            action_order=[ACTION_ID],
            motion_paths=[str(motion_path)],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProducerError(
            "schema-v2 dynamic-ready runtime binding is invalid: %s" % exc
        ) from exc
    if (
        artifact["schema_version"] != 2
        or artifact["kind"] != DYNAMIC_READY_V2_KIND
        or artifact["action_id"] != ACTION_ID
        or binding.get("schema_version") != 2
        or binding.get("kind") != "action_ball_dynamic_ready_runtime_binding_v2"
        or binding.get("action_order") != [ACTION_ID]
        or binding.get("motion_sha256_per_action") != [MOTION_SHA256]
    ):
        raise ProducerError("schema-v2 dynamic-ready action/motion identity differs")
    rows = binding.get("rows")
    if type(rows) is not list or len(rows) != 1:
        raise ProducerError("schema-v2 dynamic-ready binding is not exact N1")
    row = rows[0]
    if "runtime_plant_identity" not in row:
        raise ProducerError("schema-v2 dynamic-ready runtime plant identity is absent")

    teacher = artifact["teacher_reference"]
    if type(teacher) is not dict or set(teacher) != TEACHER_REFERENCE_KEYS:
        raise ProducerError("schema-v2 dynamic-ready teacher reference keys differ")
    if (
        teacher["semantics"] != "exact_motion_bytes_frame0_reference"
        or teacher["motion_sha256"] != MOTION_SHA256
        or teacher["frame_index"] != 0
        or not _vector_matches(
            teacher["root_pos_w_m"], motion_state["teacher_root_pos"], expected=3
        )
        or not _vector_matches(
            teacher["root_quat_wxyz"], motion_state["teacher_root_quat"], expected=4
        )
        or not _vector_matches(
            teacher["joint_pos_rad"], motion_state["teacher_joint_pos"], expected=31
        )
    ):
        raise ProducerError("dynamic-ready teacher reference differs from motion frame0")

    physical = row["physical_ready"]
    composition = artifact["physical_birth_composition"]
    ready_source = artifact["ready_source"]
    receipt_artifact = receipt["artifact"]
    if (
        type(composition) is not dict
        or composition.get("semantics")
        != "teacher_yaw_aligned_full_seed_plus_exact_teacher_reference"
        or composition.get("teacher_and_physical_birth_differ") is not True
        or not _vector_matches(
            composition.get("teacher_root_quat_wxyz"),
            teacher["root_quat_wxyz"],
            expected=4,
        )
        or not _vector_matches(
            composition.get("physical_root_quat_wxyz"),
            physical["root_quat_wxyz"],
            expected=4,
        )
        or type(ready_source) is not dict
        or ready_source.get("teacher_and_physical_birth_same") is not False
        or ready_source.get("teacher_reference_unchanged") is not True
        or ready_source.get("physical_birth_semantics") != composition["semantics"]
        or receipt["schema_version"] != 1
        or receipt["kind"] != "isaac_action_ball_nominal_hold_v1"
        or receipt["verdict"] != "PASS"
        or receipt["action_id"] != ACTION_ID
        or receipt["motion_sha256"] != MOTION_SHA256
        or receipt["teacher_reference_unchanged"] is not True
        or receipt["teacher_physical_birth_separated"] is not True
        or receipt["candidate_physical_birth_written"] is not True
        or receipt["candidate_hold_qdes_and_delay_history_installed"] is not True
        or receipt["plant_contract_match"] is not True
        or receipt["terminal_reasons"] != []
        or receipt["generic_terminated"] is not False
        or receipt["generic_truncated"] is not False
        or type(receipt_artifact) is not dict
        or receipt_artifact.get("sha256") != dynamic["artifact"]["sha256"]
        or receipt_artifact.get("content_sha256") != artifact_content_sha
    ):
        raise ProducerError(
            "nominal-hold receipt does not prove physical/teacher separation"
        )
    if _vector_matches(
        physical["root_pos_w_m"], teacher["root_pos_w_m"], expected=3
    ) and _vector_matches(
        physical["root_quat_wxyz"], teacher["root_quat_wxyz"], expected=4
    ):
        raise ProducerError("physical birth silently aliases the teacher reference")

    physical_quat = tuple(float(value) for value in physical["root_quat_wxyz"])
    base_yaw = _yaw_from_quat(physical_quat)
    projected_base_quat = (
        math.cos(0.5 * base_yaw),
        0.0,
        0.0,
        math.sin(0.5 * base_yaw),
    )
    source_contract = {
        "schema_version": 1,
        "kind": "fixed_tape_dynamic_ready_source_v1",
        "action_id": ACTION_ID,
        "motion_sha256": MOTION_SHA256,
        "artifact": {
            "path": dynamic["artifact"]["path"],
            "file_sha256": dynamic["artifact"]["sha256"],
            "content_sha256": artifact_content_sha,
        },
        "nominal_hold_receipt": {
            "path": dynamic["nominal_hold_receipt"]["path"],
            "file_sha256": dynamic["nominal_hold_receipt"]["sha256"],
            "content_sha256": receipt_content_sha,
        },
        "runtime_binding_sha256": binding["binding_sha256"],
        "runtime_plant_identity_sha256": _canonical_sha256(
            row["runtime_plant_identity"]
        ),
        "teacher_reference": {
            "frame_index": 0,
            "root_pos_w_m": list(teacher["root_pos_w_m"]),
            "root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        },
        "physical_ready": {
            "root_pos_w_m": list(physical["root_pos_w_m"]),
            "root_quat_wxyz": list(physical_quat),
            "projected_base_yaw_rad": base_yaw,
            "projected_base_quat_wxyz": list(projected_base_quat),
        },
    }
    return {
        "ready_root_z": float(physical["root_pos_w_m"][2]),
        "contact_reference_root_z": float(teacher["root_pos_w_m"][2]),
        "base_yaw": base_yaw,
        "base_quat": projected_base_quat,
        "source_contract": source_contract,
    }


def _adapt_manifest_for_dynamic_ready(
    profile_adapter: Any,
    manifest: Any,
    dynamic_ready: Mapping[str, Any],
) -> Any:
    return profile_adapter.adapt_action_ball_manifest(
        manifest,
        ready_root_z_by_slot=(dynamic_ready["ready_root_z"],),
        contact_reference_root_z_by_slot=(
            dynamic_ready["contact_reference_root_z"],
        ),
    )


def _target_values(receipt: Any) -> dict[str, Any]:
    return {
        name: getattr(receipt, name)
        for name in TARGET_REPLACEMENT_FIELDS
    }


def _replace_target(
    base: Any,
    *,
    face_velocity: Sequence[float],
    raw_normal: Sequence[float],
    residual_m: float,
    geometry_module: Any,
    runtime: Any,
) -> Any:
    geometry = geometry_module.solve_exact_face_contact(
        ball_contact_w_m=base.ball_contact_w_m,
        racket_face_center_velocity_w_mps=face_velocity,
        solved_raw_a_normal_w=raw_normal,
        mount_normal_sign=base.mount_normal_sign,
        reference_racket_quat_wxyz=base.reference_racket_quat_wxyz,
        reference_racket_angular_velocity_w_radps=(
            base.reference_racket_angular_velocity_w_radps
        ),
        reference_racket_site_speed_mps=base.reference_racket_site_speed_mps,
        teacher_rate_min=base.teacher_rate_min,
        teacher_rate_max=base.teacher_rate_max,
    )
    timing = runtime.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=geometry.racket_site_velocity_w_mps,
        time_to_contact_s=base.time_to_contact_s,
        reference_t_hit_s=base.reference_t_hit_s,
        reference_t_cycle_s=base.reference_t_cycle_s,
        reference_racket_site_speed_mps=base.reference_racket_site_speed_mps,
        reaction_margin_s=base.reaction_margin_s,
        teacher_rate_min=base.teacher_rate_min,
        teacher_rate_max=base.teacher_rate_max,
    )
    return replace(
        base,
        racket_site_target_w_m=geometry.racket_site_target_w_m,
        mount_normal_sign=geometry.mount_normal_sign,
        racket_normal_w=tuple(float(value) for value in raw_normal),
        racket_command_quat_wxyz=geometry.racket_command_quat_wxyz,
        racket_face_center_velocity_w_mps=geometry.racket_face_center_velocity_w_mps,
        racket_site_velocity_w_mps=geometry.racket_site_velocity_w_mps,
        racket_command_angular_velocity_w_radps=(
            geometry.racket_command_angular_velocity_w_radps
        ),
        geometry_source_sha256=geometry.geometry_source_sha256,
        required_racket_site_speed_mps=timing.required_racket_site_speed_mps,
        teacher_rate=timing.teacher_rate,
        scaled_t_hit_s=timing.scaled_t_hit_s,
        scaled_t_cycle_s=timing.scaled_t_cycle_s,
        pre_swing_wait_s=timing.pre_swing_wait_s,
        solver_residual_m=float(residual_m),
    )


def _receipt_bytes(receipt: Any, runtime: Any) -> bytes:
    document = receipt.to_dict()
    if runtime.ActionBallTaskReceipt.from_dict(document) != receipt:
        raise ProducerError("task receipt failed exact runtime roundtrip")
    return _canonical_bytes(document)


def _outcome_metrics(receipt: Any, *, modules: Mapping[str, Any], prm: Any, planes: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    dtype = torch.float64
    incoming = torch.tensor([receipt.incoming_velocity_w_mps], dtype=dtype)
    spin = torch.tensor([receipt.incoming_spin_w_radps], dtype=dtype)
    face_velocity = torch.tensor([receipt.racket_face_center_velocity_w_mps], dtype=dtype)
    normal = torch.tensor([receipt.racket_normal_w], dtype=dtype)
    v_out, w_out = modules["virtual_ball"].predict_paddle_contact(
        incoming, face_velocity, normal, spin, prm
    )
    rollout = modules["virtual_ball"].coarse_landing(
        torch.tensor([receipt.ball_contact_w_m], dtype=dtype),
        v_out,
        w_out,
        prm,
        surface_z=float(planes["surface_z"]),
        net_x=float(planes["net_x"]),
        h=float(cfg["vb_rollout_h"]),
        n_steps=int(cfg["vb_rollout_steps"]),
    )
    incoming_xy = incoming[0, :2]
    outgoing_xy = v_out[0, :2]
    denominator = float(torch.linalg.norm(incoming_xy) * torch.linalg.norm(outgoing_xy))
    if denominator <= 1.0e-12:
        raise ProducerError("incoming/outgoing horizontal speed is zero")
    same_cosine = float(torch.dot(incoming_xy, outgoing_xy) / denominator)
    reverse_cosine = -same_cosine
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, same_cosine))))
    if not math.isfinite(reverse_cosine) or reverse_cosine < REVERSE_RETURN_MIN_COSINE:
        raise ProducerError(
            "target does not reverse the incoming XY direction: cosine=%.9g angle=%.9g"
            % (reverse_cosine, angle_deg)
        )
    return {
        "incoming_velocity_w_mps": [float(value) for value in incoming[0].tolist()],
        "desired_outgoing_velocity_w_mps": [float(value) for value in v_out[0].tolist()],
        "desired_outgoing_spin_w_radps": [float(value) for value in w_out[0].tolist()],
        "incoming_to_outgoing_xy_cosine": same_cosine,
        "reverse_return_xy_cosine": reverse_cosine,
        "incoming_to_outgoing_xy_angle_deg": angle_deg,
        "reverse_return_min_cosine": REVERSE_RETURN_MIN_COSINE,
        "reverse_return_gate_pass": True,
        "predicted_landing_w_xy_m": [float(value) for value in rollout["land_xy"][0].tolist()],
        "landing_error_m": float(
            torch.linalg.norm(
                rollout["land_xy"][0]
                - torch.tensor(receipt.landing_aim_w_xy_m, dtype=dtype)
            )
        ),
        "land_valid": bool(rollout["land_valid"][0]),
        "net_valid": bool(rollout["net_valid"][0]),
        "net_z_m": float(rollout["net_z"][0]),
    }


def _producer_contract(
    *,
    recipe: str,
    algorithm_id: str,
    parameters: Mapping[str, Any],
    source_sha256: Mapping[str, str],
    prepared_sha256: str,
    base_question_sha256: str,
    dynamic_ready_source: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "measured_action_ball_n1_target_producer_contract_v1",
        "recipe": recipe,
        "validity_mask": list(VALIDITY[recipe]),
        "algorithm_id": algorithm_id,
        "algorithm_parameters": dict(parameters),
        "implementation_source_sha256": dict(sorted(source_sha256.items())),
        "prepared_core_sha256": prepared_sha256,
        "base_question_sha256": base_question_sha256,
        "dynamic_ready_source": dict(dynamic_ready_source),
    }
    return {"payload": payload, "sha256": _canonical_sha256(payload)}


def produce(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    root = Path(args.repo_root).resolve(strict=True)
    materializer = _load_module(
        "_measured_n1_bundle_materializer_for_tape",
        SCRIPT_DIR / "materialize_measured_action_ball_n1_bundle.py",
    )
    prepared_pin, prepared, core = materializer._validate_prepared_core(
        root,
        Path(args.prepared_core_bundle),
        args.expected_prepared_core_bundle_sha256,
    )
    if (
        prepared["action_id"] != ACTION_ID
        or prepared["action_uid"] != ACTION_UID
        or prepared["measured_uid"] != MEASURED_UID
        or prepared["motion"] != {"path": MOTION_PATH, "sha256": MOTION_SHA256}
    ):
        raise ProducerError("prepared core is not the exact selected action")
    modules = _load_mdp_modules(root)
    runtime = modules["action_ball_runtime"]
    sampling = modules["action_ball_sampling"]
    manifest_module = modules["action_ball_manifest"]
    profile_adapter = modules["action_ball_profile_adapter"]
    geometry = modules["racket_contact_geometry"]
    motion_path = _repo_path(root, MOTION_PATH, MOTION_SHA256, label="selected motion")
    manifest_path = _repo_path(
        root, core["manifest"]["path"], core["manifest"]["sha256"], label="prepared N1 manifest"
    )
    prototype_path = _repo_path(
        root, core["prototype"]["path"], core["prototype"]["sha256"], label="prepared N1 prototype"
    )
    profile_path = _repo_path(
        root, core["profile_pins"]["path"], core["profile_pins"]["sha256"], label="prepared profile pins"
    )
    profile_pins = _strict_json(profile_path, label="prepared profile pins")
    loaded = manifest_module.load_action_ball_manifest(
        manifest_path,
        expected_sha256=core["manifest"]["sha256"],
        verify_referenced_assets=True,
        repo_root=root,
    )
    manifest = loaded.manifest
    if manifest.action_order != (ACTION_ID,) or manifest.mobility_mode != "no_move":
        raise ProducerError("prepared manifest is not exact fixed no_move N1")
    action = manifest.actions[0]
    state = _motion_state(motion_path, action.strike_phase, geometry)
    dynamic_ready = _dynamic_ready_source(
        root,
        prepared=prepared,
        core=core,
        motion_path=motion_path,
        motion_state=state,
    )
    adapted = _adapt_manifest_for_dynamic_ready(
        profile_adapter, manifest, dynamic_ready
    )
    profile = adapted.profiles[0]
    objective = profile.counter_rally_objective
    if objective is None:
        raise ProducerError("prepared profile lacks counter-rally objective")
    mixture = sampling.SamplingMixture()
    sampler = sampling.ActionBallSampler(
        adapted.profiles,
        seed=args.seed,
        sampling_mixture=mixture,
        contact_time_step_s=POLICY_DT_S,
        diagnostic_unauthorized=True,
        initial_center_single_question=True,
    )
    levels = sampling.DomainLevels()
    sampler_birth = sampler.reserve_birth(
        action_uid=ACTION_UID,
        domain_epoch=0,
        levels=levels,
        base_yaw_rad=dynamic_ready["base_yaw"],
    )
    sample = sampler.sample(
        birth=sampler_birth,
        action_uid=ACTION_UID,
        domain_epoch=0,
        levels=levels,
        base_yaw_rad=dynamic_ready["base_yaw"],
    )
    objective_module = sys.modules.get(type(objective).__module__)
    derive_counter_task = getattr(
        objective_module, "derive_counter_rally_task", None
    )
    if not callable(derive_counter_task):
        raise ProducerError("counter-rally objective module lacks task derivation")
    incoming_horizontal_norm = math.hypot(
        sample.incoming_direction_b_yaw[0],
        sample.incoming_direction_b_yaw[1],
    )
    if incoming_horizontal_norm <= 1.0e-12:
        raise ProducerError("counter-rally incoming horizontal direction is zero")
    incoming_direction_b_xy = (
        sample.incoming_direction_b_yaw[0] / incoming_horizontal_norm,
        sample.incoming_direction_b_yaw[1] / incoming_horizontal_norm,
    )
    counter_task = derive_counter_task(
        base_goal_env_xy_m=sample.base_goal_w_m[:2],
        base_yaw_env_rad=dynamic_ready["base_yaw"],
        contact_offset_b_yaw_m=sample.contact_offset_from_base_goal_b_yaw_m,
        incoming_direction_b_yaw=incoming_direction_b_xy,
        incoming_ball_speed_at_contact_mps=sample.incoming_speed_mps,
        landing_depth_env_x_m=sample.landing_aim_w_xy_m[0],
        profile=objective,
    )
    if any(
        abs(float(left) - float(right)) > 1.0e-9
        for left, right in zip(
            counter_task.landing_aim_env_xy_m,
            sample.landing_aim_w_xy_m,
        )
    ):
        raise ProducerError("sampler row and counter-rally task landing differ")
    counter_identity = runtime.CounterRallyTaskIdentity(
        objective_profile_sha256=objective.sha256,
        return_direction_env_xy=counter_task.return_direction_env_xy,
        target_baseline_speed_mps=counter_task.target_baseline_speed_mps,
    )
    runtime_levels = runtime.ActionDomainLevels(**levels.as_dict())
    domain_authority_sha = _canonical_sha256(
        {
            "kind": "measured_n1_offline_domain_authority_v1",
            "prepared_core_sha256": prepared_pin["sha256"],
            "seed": args.seed,
        }
    )
    claim = runtime.ActionDomainClaim(
        authority_contract_sha256=domain_authority_sha,
        arm_catalog_sha256=runtime.ARM_CATALOG_SHA256,
        action_uid=ACTION_UID,
        domain_epoch=0,
        domain_levels=runtime_levels,
        levels_sha256=runtime_levels.canonical_sha256,
        profile_sha256=profile.sha256,
        mobility_mode="no_move",
    )
    binding = runtime.ActionBinding(
        action_uid=ACTION_UID,
        action_slot=0,
        motion_path=MOTION_PATH,
        motion_sha256=MOTION_SHA256,
        profile_sha256=profile.sha256,
    )
    pins = runtime.RuntimePins(
        manifest_sha256=core["manifest"]["sha256"],
        sampler_sha256=sampler.sampler_contract_sha256,
        domain_authority_sha256=domain_authority_sha,
        physics_sha256=profile_pins["physics_profile_sha256"],
        solver_sha256=profile_pins["solver_profile_sha256"],
        counter_rally_objective_profile_sha256=objective.sha256,
    )
    runtime_mixture = runtime.ActionSamplingMixture.from_dict(mixture.as_dict())
    birth = runtime.ActionBirthReceipt(
        env_id=0,
        reset_generation=1,
        action_uid=ACTION_UID,
        action_slot=0,
        domain_epoch=0,
        domain_claim_sha256=claim.canonical_sha256,
        domain_authority_sha256=domain_authority_sha,
        domain_levels=runtime_levels,
        arm_catalog_sha256=runtime.ARM_CATALOG_SHA256,
        levels_sha256=runtime_levels.canonical_sha256,
        sampler_birth_sha256=sampler_birth.birth_id,
        sampler_birth_index=sampler_birth.birth_index,
        sampler_draw_start=sampler_birth.draw_start,
        sampler_draw_end=sampler_birth.draw_end,
        mobility_mode="no_move",
        base_yaw_rad=dynamic_ready["base_yaw"],
        base_quat_wxyz=dynamic_ready["base_quat"],
        base_spawn_w_m=sampler_birth.base_start_w_m,
        manifest_sha256=pins.manifest_sha256,
        sampler_sha256=pins.sampler_sha256,
        profile_sha256=profile.sha256,
        motion_sha256=MOTION_SHA256,
        physics_sha256=pins.physics_sha256,
        solver_sha256=pins.solver_sha256,
        registry_sha256=runtime._registry_sha256((binding,), pins, "no_move"),
        sampling_mixture=runtime_mixture,
        sampling_stratum=sampler_birth.sampling_stratum,
        sampling_levels=runtime.ActionDomainLevels(
            **sampler_birth.sampling_levels.as_dict()
        ),
        frontier_arm=sampler_birth.frontier_arm,
    )
    prototypes = modules["stroke_prototypes_torch"].load_stroke_prototype_tensors(
        str(prototype_path),
        scope="full",
        device="cpu",
        expected_sha256=core["prototype"]["sha256"],
        expected_motion_ids=(ACTION_ID,),
        expected_motion_sha256=(MOTION_SHA256,),
    )
    venue_path = _repo_path(
        root,
        profile_pins["physics_payload"]["venue_source"]["path"],
        profile_pins["physics_payload"]["venue_source"]["file_sha256"],
        label="venue physics",
    )
    prm = modules["virtual_ball"].load_venue_params(str(venue_path))
    cfg = profile_pins["cfg"]
    planes = profile_pins["planes"]
    solver_cfg = modules["continuous_questions"].ContinuousQuestionCfg(
        fixed_direction=True,
        n_iters=int(cfg["cq_n_iters"]),
        tol_m=float(cfg["cq_tol_m"]),
        speed_budget=float(cfg["cq_speed_budget"]),
    )
    solved = modules["continuous_questions"].solve_proposals(
        torch.tensor((0,), dtype=torch.long),
        torch.tensor((sample.contact_w_m,), dtype=torch.float32),
        torch.tensor((sample.incoming_velocity_w_mps,), dtype=torch.float32),
        torch.tensor((sample.spin_w_radps,), dtype=torch.float32),
        torch.tensor((sample.landing_aim_w_xy_m,), dtype=torch.float32),
        torch.tensor((state["teacher_raw_normal"],), dtype=torch.float32),
        protos=prototypes,
        base_quat=torch.tensor((dynamic_ready["base_quat"],), dtype=torch.float32),
        prm=prm,
        surface_z=float(planes["surface_z"]),
        net_x=float(planes["net_x"]),
        net_top_z=float(planes["net_top_z"]),
        cfg=solver_cfg,
        h=float(cfg["vb_rollout_h"]),
        n_steps=int(cfg["vb_rollout_steps"]),
    )
    if solved.ok.tolist() != [True] or solved.reason_counts:
        raise ProducerError("current LM rejected the fixed centre: %r" % solved.reason_counts)
    lm_geometry = geometry.solve_exact_face_contact(
        ball_contact_w_m=sample.contact_w_m,
        racket_face_center_velocity_w_mps=solved.v_racket[0].tolist(),
        solved_raw_a_normal_w=solved.n_racket[0].tolist(),
        mount_normal_sign=action.mount_normal_sign,
        reference_racket_quat_wxyz=state["reference_quat"],
        reference_racket_angular_velocity_w_radps=state["reference_omega"],
        reference_racket_site_speed_mps=action.reference_racket_site_speed_mps,
        teacher_rate_min=action.teacher_rate_min,
        teacher_rate_max=action.teacher_rate_max,
    )
    lm_timing = runtime.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=lm_geometry.racket_site_velocity_w_mps,
        time_to_contact_s=sample.time_to_contact_s,
        reference_t_hit_s=action.reference_t_hit_s,
        reference_t_cycle_s=action.reference_t_cycle_s,
        reference_racket_site_speed_mps=action.reference_racket_site_speed_mps,
        reaction_margin_s=action.reaction_margin_s,
        teacher_rate_min=action.teacher_rate_min,
        teacher_rate_max=action.teacher_rate_max,
    )
    goal = (
        float(sample.base_goal_w_m[0]),
        float(sample.base_goal_w_m[1]),
        float(birth.base_spawn_w_m[2]),
    )
    current = runtime.ActionBallTaskReceipt.from_birth(
        birth,
        sample_sha256=sample.sample_id,
        sample_index=sample.sample_index,
        sample_draw_start=sample.draw_start,
        sample_draw_end=sample.draw_end,
        swing_generation=0,
        base_goal_w_m=goal,
        base_spawn_latent_w_m=sample.base_spawn_latent_w_m,
        base_travel_latent_b_yaw_m=sample.base_travel_latent_b_yaw_m,
        contact_offset_from_base_goal_b_yaw_m=sample.contact_offset_from_base_goal_b_yaw_m,
        ball_contact_w_m=sample.contact_w_m,
        racket_site_target_w_m=lm_geometry.racket_site_target_w_m,
        time_to_contact_s=sample.time_to_contact_s,
        incoming_speed_mps=sample.incoming_speed_mps,
        incoming_direction_b_yaw=sample.incoming_direction_b_yaw,
        incoming_velocity_w_mps=sample.incoming_velocity_w_mps,
        spin_magnitude_radps=sample.spin_magnitude_radps,
        spin_direction_b_yaw=sample.spin_direction_b_yaw,
        incoming_spin_w_radps=sample.spin_w_radps,
        landing_aim_w_xy_m=sample.landing_aim_w_xy_m,
        mount_normal_sign=lm_geometry.mount_normal_sign,
        racket_normal_w=tuple(float(value) for value in solved.n_racket[0].tolist()),
        reference_racket_quat_wxyz=state["reference_quat"],
        reference_racket_angular_velocity_w_radps=state["reference_omega"],
        racket_command_quat_wxyz=lm_geometry.racket_command_quat_wxyz,
        racket_face_center_velocity_w_mps=lm_geometry.racket_face_center_velocity_w_mps,
        racket_site_velocity_w_mps=lm_geometry.racket_site_velocity_w_mps,
        racket_command_angular_velocity_w_radps=(
            lm_geometry.racket_command_angular_velocity_w_radps
        ),
        geometry_source_sha256=lm_geometry.geometry_source_sha256,
        reference_t_hit_s=action.reference_t_hit_s,
        reference_t_cycle_s=action.reference_t_cycle_s,
        reference_racket_site_speed_mps=action.reference_racket_site_speed_mps,
        required_racket_site_speed_mps=lm_timing.required_racket_site_speed_mps,
        reaction_margin_s=action.reaction_margin_s,
        teacher_rate_min=action.teacher_rate_min,
        teacher_rate_max=action.teacher_rate_max,
        teacher_rate=lm_timing.teacher_rate,
        scaled_t_hit_s=lm_timing.scaled_t_hit_s,
        scaled_t_cycle_s=lm_timing.scaled_t_cycle_s,
        pre_swing_wait_s=lm_timing.pre_swing_wait_s,
        solver_residual_m=float(solved.resid_m[0]),
        contact_time_step_s=sample.contact_time_step_s,
        time_to_contact_tick=sample.time_to_contact_tick,
        birth_index=sample.birth_index,
        birth_sampling_stratum=sample.birth_sampling_stratum,
        birth_sampling_levels=runtime.ActionDomainLevels(
            **sample.birth_sampling_levels.as_dict()
        ),
        birth_frontier_arm=sample.birth_frontier_arm,
        sampling_mixture=runtime_mixture,
        sampling_stratum=sample.sampling_stratum,
        sampling_levels=runtime.ActionDomainLevels(**sample.sampling_levels.as_dict()),
        frontier_arm=sample.frontier_arm,
        counter_rally_task=counter_identity,
    )
    analytic_result = modules["strike_spec_analytic"].solve_analytic(
        torch.tensor((sample.contact_w_m,), dtype=torch.float64),
        torch.tensor((sample.incoming_velocity_w_mps,), dtype=torch.float64),
        torch.tensor((sample.spin_w_radps,), dtype=torch.float64),
        torch.tensor((sample.landing_aim_w_xy_m,), dtype=torch.float64),
        prm,
        float(planes["surface_z"]),
        float(planes["net_x"]),
        t_flight=ANALYTIC_FLIGHT_TIME_S,
        ref_normal=torch.tensor(
            (state["teacher_raw_normal"],), dtype=torch.float64
        ),
        speed_budget=float(cfg["cq_speed_budget"]),
        net_top_z=float(planes["net_top_z"]),
    )
    if not bool(analytic_result["ok"][0]):
        reason_code = int(analytic_result["reason"][0])
        raise ProducerError(
            "analytic solver rejected fixed centre: %s"
            % modules["strike_spec_analytic"].REASONS[reason_code]
        )
    analytic = _replace_target(
        current,
        face_velocity=analytic_result["v_r"][0].tolist(),
        raw_normal=analytic_result["n"][0].tolist(),
        residual_m=0.0,
        geometry_module=geometry,
        runtime=runtime,
    )
    teacher = _replace_target(
        current,
        face_velocity=state["teacher_face_velocity"],
        raw_normal=state["teacher_raw_normal"],
        residual_m=0.0,
        geometry_module=geometry,
        runtime=runtime,
    )
    provisional = {"analytic": analytic, "teacher": teacher}
    provisional_outcome = {
        name: _outcome_metrics(
            receipt, modules=modules, prm=prm, planes=planes, cfg=cfg
        )
        for name, receipt in provisional.items()
    }
    analytic = replace(
        analytic,
        solver_residual_m=provisional_outcome["analytic"]["landing_error_m"],
    )
    teacher = replace(
        teacher,
        solver_residual_m=provisional_outcome["teacher"]["landing_error_m"],
    )
    targets = {
        "current_lm": current,
        "analytic_full": analytic,
        "analytic_no_velocity": analytic,
        "teacher_pos_face_no_velocity": teacher,
        "outcome_dense_only": current,
    }
    base_question_sha = modules["action_ball_fixed_question_tape"]._sha256_json(
        modules["action_ball_fixed_question_tape"]._question_payload(current)
    )
    source_files = {
        name: _sha256_file(root.joinpath(*MDP_RELATIVE.parts) / (name + ".py"))
        for name in (
            "continuous_questions",
            "strike_spec_torch",
            "strike_spec_analytic",
            "virtual_ball",
            "racket_contact_geometry",
            "action_ball_runtime",
            "action_ball_fixed_question_tape",
        )
    }
    # The orchestrator owns teacher finite-difference velocity, recipe mapping and
    # artifact assembly, so its own bytes are part of every producer lineage.
    source_files["fixed_tape_variant_producer"] = _sha256_file(Path(__file__).resolve())
    source_files["training_contract"] = _sha256_file(
        root.joinpath(*TRAINING_CONTRACT_RELATIVE.parts)
    )

    def implementation_sources(*names: str) -> dict[str, str]:
        return {
            name: source_files[name]
            for name in (
                *names,
                "training_contract",
                "fixed_tape_variant_producer",
            )
        }

    contracts = {
        "current_lm": _producer_contract(
            recipe="current_lm",
            algorithm_id="runtime_continuous_questions_lm",
            parameters={"n_iters": int(cfg["cq_n_iters"]), "tol_m": float(cfg["cq_tol_m"])},
            source_sha256=implementation_sources(
                "continuous_questions", "strike_spec_torch", "virtual_ball",
                "racket_contact_geometry",
            ),
            prepared_sha256=prepared_pin["sha256"],
            base_question_sha256=base_question_sha,
            dynamic_ready_source=dynamic_ready["source_contract"],
        ),
        "analytic_full": _producer_contract(
            recipe="analytic_full",
            algorithm_id="strike_spec_analytic_closed_form",
            parameters={
                "t_flight_s": ANALYTIC_FLIGHT_TIME_S,
                "pin": "normal",
                "reference_normal": "official_fk_teacher_raw_A",
                "n_outer": 4,
                "n_nodes": 4,
                "n_picard": 2,
                "horizon_s": 1.0,
                "envelope_rejects": False,
                "pin_tol_mps": 0.02,
                "speed_budget_mps": float(cfg["cq_speed_budget"]),
                "net_top_z_m": float(planes["net_top_z"]),
            },
            source_sha256=implementation_sources(
                "strike_spec_analytic", "virtual_ball", "racket_contact_geometry",
            ),
            prepared_sha256=prepared_pin["sha256"],
            base_question_sha256=base_question_sha,
            dynamic_ready_source=dynamic_ready["source_contract"],
        ),
        "analytic_no_velocity": _producer_contract(
            recipe="analytic_no_velocity",
            algorithm_id="strike_spec_analytic_closed_form_mask_velocity",
            parameters={
                "t_flight_s": ANALYTIC_FLIGHT_TIME_S,
                "pin": "normal",
                "reference_normal": "official_fk_teacher_raw_A",
                "n_outer": 4,
                "n_nodes": 4,
                "n_picard": 2,
                "horizon_s": 1.0,
                "envelope_rejects": False,
                "pin_tol_mps": 0.02,
                "speed_budget_mps": float(cfg["cq_speed_budget"]),
                "net_top_z_m": float(planes["net_top_z"]),
            },
            source_sha256=implementation_sources(
                "strike_spec_analytic", "virtual_ball", "racket_contact_geometry",
            ),
            prepared_sha256=prepared_pin["sha256"],
            base_question_sha256=base_question_sha,
            dynamic_ready_source=dynamic_ready["source_contract"],
        ),
        "teacher_pos_face_no_velocity": _producer_contract(
            recipe="teacher_pos_face_no_velocity",
            algorithm_id="official_fk_teacher_at_hit_position_face",
            parameters={
                "strike_frame": state["strike_frame"],
                "site_velocity_fd_half_window_frames": 2,
            },
            source_sha256={
                "motion": MOTION_SHA256,
                **implementation_sources("racket_contact_geometry"),
            },
            prepared_sha256=prepared_pin["sha256"],
            base_question_sha256=base_question_sha,
            dynamic_ready_source=dynamic_ready["source_contract"],
        ),
        "outcome_dense_only": _producer_contract(
            recipe="outcome_dense_only",
            algorithm_id="coherent_current_lm_carrier_mask_all_targets",
            parameters={},
            source_sha256=implementation_sources(
                "continuous_questions", "virtual_ball", "racket_contact_geometry",
            ),
            prepared_sha256=prepared_pin["sha256"],
            base_question_sha256=base_question_sha,
            dynamic_ready_source=dynamic_ready["source_contract"],
        ),
    }
    if len({row["sha256"] for row in contracts.values()}) != 5:
        raise ProducerError("five recipe producer contract SHAs must be distinct")
    tape = modules["action_ball_fixed_question_tape"].ImmutableN1QuestionTape.from_receipts(
        question_receipt=current,
        target_receipts=targets,
        target_producer_sha256={name: contracts[name]["sha256"] for name in RECIPES},
    )
    outcome = {
        name: _outcome_metrics(
            targets[name], modules=modules, prm=prm, planes=planes, cfg=cfg
        )
        for name in RECIPES
    }
    for recipe in ("current_lm", "analytic_full"):
        gate = outcome[recipe]
        if (
            gate["land_valid"] is not True
            or gate["net_valid"] is not True
            or gate["landing_error_m"] >= float(cfg["cq_tol_m"])
        ):
            raise ProducerError(
                "%s target failed achieved outgoing flight gate" % recipe
            )
    birth_x_lower_bound = modules["continuous_questions"].ball_birth_x_lower_bound_m(
        sample.contact_w_m[0],
        sample.incoming_velocity_w_mps[0],
        sample.time_to_contact_s,
    )
    birth_x_minimum = (
        float(planes["net_x"])
        + float(modules["continuous_questions"].BALL_BIRTH_NET_MARGIN_M)
    )
    episode_end_s = (
        current.pre_swing_wait_s + current.scaled_t_cycle_s + POLICY_DT_S
    )
    if birth_x_lower_bound < birth_x_minimum:
        raise ProducerError("fixed question ball birth is not beyond net")
    if episode_end_s > 10.0:
        raise ProducerError("fixed question cycle exceeds episode horizon")
    question_gates = {
        "ball_birth_x_lower_bound_m": birth_x_lower_bound,
        "ball_birth_minimum_x_m": birth_x_minimum,
        "ball_birth_beyond_net": True,
        "episode_end_s_including_close_margin": episode_end_s,
        "episode_length_s": 10.0,
        "cycle_within_episode_horizon": True,
        "counter_rally_task_identity_sha256": counter_identity.canonical_sha256,
    }
    base_bytes = _receipt_bytes(current, runtime)
    target_bytes = {name: _receipt_bytes(targets[name], runtime) for name in RECIPES}
    tape_bytes = _canonical_bytes(tape.to_dict())
    destination = Path(args.output_dir)
    destination = destination if destination.is_absolute() else root / destination
    destination = destination.resolve(strict=False)
    try:
        output_relative = destination.relative_to(root)
    except ValueError as exc:
        raise ProducerError("output directory must remain inside repo root") from exc
    if destination.exists():
        raise ProducerError("no-clobber output directory already exists: %s" % destination)
    base_name = "base_question.task_receipt.v5.%s.json" % hashlib.sha256(base_bytes).hexdigest()[:12]
    target_names = {
        name: "%s.target.task_receipt.v5.%s.json"
        % (name, hashlib.sha256(target_bytes[name]).hexdigest()[:12])
        for name in RECIPES
    }
    tape_name = "immutable_n1_tape.v1.%s.json" % hashlib.sha256(tape_bytes).hexdigest()[:12]
    def pin(name: str, raw: bytes) -> dict[str, str]:
        return {
            "path": (output_relative / name).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    source_identity = {
        "action_uid": current.action_uid,
        "action_slot": current.action_slot,
        "profile_sha256": current.profile_sha256,
        "motion_sha256": current.motion_sha256,
        "manifest_sha256": current.manifest_sha256,
        "sampler_sha256": current.sampler_sha256,
        "physics_sha256": current.physics_sha256,
        "solver_sha256": current.solver_sha256,
        "mobility_mode": current.mobility_mode,
        "counter_rally_objective_profile_sha256": objective.sha256,
    }
    common_contracts = {
        "incoming_ball": {
            "payload": {"kind": "fixed_sampler_question_v1", "seed": args.seed, "sample_sha256": current.sample_sha256},
        },
        "teacher_contact": {
            "payload": {"kind": "measured_teacher_contact_v1", "motion_sha256": MOTION_SHA256, "strike_frame": state["strike_frame"]},
        },
        "landing_spin_task": {
            "payload": {"kind": "fixed_manifest_landing_spin_task_v1", "manifest_sha256": current.manifest_sha256, "base_question_sha256": base_question_sha},
        },
        "dynamic_ready_birth": {
            "payload": dynamic_ready["source_contract"],
        },
    }
    for row in common_contracts.values():
        row["sha256"] = _canonical_sha256(row["payload"])
    target_rows = {}
    for name in RECIPES:
        lineage = tape.target_lineage(name)
        target_rows[name] = {
            "artifact": pin(target_names[name], target_bytes[name]),
            "task_receipt_canonical_sha256": targets[name].canonical_sha256,
            "base_question_sha256": base_question_sha,
            "sample_sha256": targets[name].sample_sha256,
            "target_producer_sha256": contracts[name]["sha256"],
            "target_column_sha256": lineage["target_column_sha256"],
            "validity_mask": list(VALIDITY[name]),
            "outgoing_gate": outcome[name],
        }
    report = {
        "schema_version": 1,
        "kind": BUILD_REPORT_KIND,
        "diagnostic_unauthorized": True,
        "sampler_seed": args.seed,
        "prepared_core": prepared_pin,
        "source_identity": source_identity,
        "base_question": {
            "artifact": pin(base_name, base_bytes),
            "task_receipt_canonical_sha256": current.canonical_sha256,
            "base_question_sha256": base_question_sha,
            "sample_sha256": current.sample_sha256,
            "hard_gates": question_gates,
        },
        "target_receipts": target_rows,
        "producer_contracts": {
            "common": common_contracts,
            "desired_contact": contracts,
        },
        "tape": {
            "artifact": pin(tape_name, tape_bytes),
            "canonical_sha256": tape.canonical_sha256,
            "base_question_sha256": tape.question_sha256,
            "row_count": 1,
            "question_shape": [1, modules["action_ball_fixed_question_tape"].QUESTION_WIDTH],
            "install_shape": [1, modules["action_ball_fixed_question_tape"].INSTALL_WIDTH],
            "observation_shape": [1, modules["action_ball_fixed_question_tape"].OBSERVATION_WIDTH],
            "all_target_lineage": {name: tape.target_lineage(name) for name in RECIPES},
        },
        "reset_semantics": {
            "selection": "constant_row_zero",
            "online_sampler_calls": 0,
            "online_lm_calls": 0,
            "physical_rng_draws": 0,
        },
        "diagnostic_admissibility": {
            "teacher_rate_max": action.teacher_rate_max,
            "source_teacher_rate_max": 1.0,
            "reason": (
                "fixed current-LM answer requires 1.0000373; diagnostic "
                "envelope is explicitly widened to 1.01"
            ),
        },
        "claims": {
            "diagnostic_unauthorized": True,
            "formal_evidence_prohibited": True,
            "promotion_prohibited": True,
            "export_prohibited": True,
            "deployment_prohibited": True,
            "hardware_prohibited": True,
        },
    }
    report_bytes = _canonical_bytes(report)
    report_name = "offline_n1_tape_build_report.v1.%s.json" % hashlib.sha256(report_bytes).hexdigest()[:12]
    outputs = [(base_name, base_bytes)]
    outputs.extend((target_names[name], target_bytes[name]) for name in RECIPES)
    outputs.extend(((tape_name, tape_bytes), (report_name, report_bytes)))
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, raw in outputs:
            with (destination / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
    except Exception:
        raise
    return {
        "status": "PASS_DIAGNOSTIC_ONLY",
        "diagnostic_unauthorized": True,
        "action_id": ACTION_ID,
        "action_uid": ACTION_UID,
        "sampler_seed": args.seed,
        "prepared_core": prepared_pin,
        "build_report": pin(report_name, report_bytes),
        "tape": pin(tape_name, tape_bytes),
        "base_question_sha256": base_question_sha,
        "target_producer_sha256": {name: contracts[name]["sha256"] for name in RECIPES},
        "online_reset_lm_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--prepared-core-bundle", required=True)
    parser.add_argument("--expected-prepared-core-bundle-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = produce(_parser().parse_args(argv))
    except (ProducerError, FileNotFoundError, ValueError, OSError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
