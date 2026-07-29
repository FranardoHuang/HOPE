#!/usr/bin/env python3
"""Formal learned-policy return gate for one trusted exact ActionBall bank.

This gate intentionally does *not* reuse ``mujoco_eval_onnx``'s virtual
landing scorer.  It drives the real MuJoCo robot with the exported actor and
the bound PD plant while the independently fitted ball referee owns the ball
flight, swept selected-face impulse, net crossing and first table landing.

The input teacher receipt is not a substitute for policy execution.  It is a
sealed prerequisite proving that the same action/task/solver/physics capsule
is physically solvable by the teacher.  This program then executes the frozen
policy on the positive center/support cases from that capsule.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mujoco_eval_onnx as legacy_mj  # noqa: E402
import mujoco_teacher_motion_fitted_ball_gate as fitted  # noqa: E402


SCHEMA_VERSION = 1
GATE_NAME = "mujoco_action_ball_policy_fitted_gate"
BASE_OBS_DIM = 177 + 4
ACTION_COUNT = 5  # Backward-compatible N5 test alias; formal code is dynamic.
OBS_DIM = BASE_OBS_DIM + ACTION_COUNT
POSITIVE_CASE_ROLES = (
    "center_positive_seed_0",
    "center_positive_seed_1",
    "support_positive",
)
FORMAL_DT_S = (0.001, 0.0005)
CONTROL_DT_S = 0.02
CONTACT_TIME_TOLERANCE_S = 0.060
LANDING_AIM_TOLERANCE_M = 0.10
STRIKE_POSITION_ERROR_TOLERANCE_M = legacy_mj.STRIKE_POS_THRESH
STRIKE_VELOCITY_ERROR_TOLERANCE_MPS = legacy_mj.STRIKE_VEL_THRESH
STRIKE_NORMAL_ERROR_TOLERANCE_RAD = math.radians(
    legacy_mj.STRIKE_NORMAL_THRESH_DEG
)
ROOT_Z_FALL_M = fitted.ROOT_Z_FALL_M
ROOT_TILT_FALL_RAD = fitted.ROOT_TILT_FALL_RAD
TABLE_CONTACT_FORCE_THRESHOLD_N = 1.0e-6
FIVE_SOLID_SWEEP_METHOD = (
    "linear_geom_center_segment_plus_rotation_invariant_mujoco_rbound_v1"
)


class PolicyGateError(RuntimeError):
    """Fail-closed input, contract or execution error."""


def _sha256_file(path: Path) -> str:
    return fitted.native_diag.sha256_file(path)


def _require_sha(value: str, label: str) -> str:
    try:
        return fitted.native_diag._require_sha(value, label)
    except Exception as exc:
        raise PolicyGateError(str(exc)) from exc


def _read_pinned(path: Path, expected_sha256: str, label: str) -> bytes:
    try:
        _resolved, payload = fitted.read_pinned_regular_file(
            path, _require_sha(expected_sha256, f"{label} SHA256"), label
        )
    except Exception as exc:
        raise PolicyGateError(str(exc)) from exc
    return payload


def _csv(md: Mapping[str, str], key: str, *, count: int) -> np.ndarray:
    raw = str(md.get(key, "")).strip()
    try:
        out = np.asarray([float(item) for item in raw.split(",")], np.float64)
    except ValueError as exc:
        raise PolicyGateError(f"invalid ONNX metadata {key}") from exc
    if out.shape != (count,) or not np.isfinite(out).all():
        raise PolicyGateError(
            f"ONNX metadata {key} must contain {count} finite values"
        )
    return out


def _csv_text(md: Mapping[str, str], key: str, *, count: int) -> Tuple[str, ...]:
    out = tuple(item.strip() for item in str(md.get(key, "")).split(","))
    if len(out) != count or any(not item for item in out):
        raise PolicyGateError(
            f"ONNX metadata {key} must contain {count} nonempty values"
        )
    return out


def _canonical_json_array_metadata(
    md: Mapping[str, str],
    key: str,
    *,
    count: int,
    item_type: type,
) -> tuple[Any, ...]:
    raw = str(md.get(key, "")).strip()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PolicyGateError(
            f"ONNX metadata {key} is not JSON"
        ) from exc
    if (
        type(value) is not list
        or len(value) != count
        or any(type(item) is not item_type for item in value)
        or raw
        != json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    ):
        raise PolicyGateError(
            f"ONNX metadata {key} is not an exact canonical JSON array"
        )
    return tuple(value)


def _actor_term_contract(
    action_count: int = ACTION_COUNT,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    # Importing the registry through the package requires Isaac Lab.  The exact
    # public contract is reproduced as ordered names/dims here and compared to
    # the ONNX metadata.  The 177-D prefix is legacy_mj's existing
    # HITTER_FOOTWORK contract; only the two explicit tails are new.
    names = tuple(legacy_mj.HITTER_FOOTWORK_OBS_NAMES) + (
        "racket_target_normal_cmd",
        "action_one_hot",
    )
    if type(action_count) is not int or action_count < 1:
        raise PolicyGateError("action count must be a positive integer")
    dims = tuple(legacy_mj.HITTER_FOOTWORK_OBS_DIMS) + (4, action_count)
    if sum(dims) != BASE_OBS_DIM + action_count:
        raise PolicyGateError("internal ActionBall actor layout is inconsistent")
    return names, dims


def validate_actor_metadata(
    md: Mapping[str, str],
    *,
    checkpoint_sha256: str,
    onnx_obs_dim: int,
    trusted_action_set: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the exact N-dependent ActionBall actor and plant binding."""

    action_count = int(trusted_action_set["expected_n"])
    obs_dim = int(trusted_action_set["actor_obs_width"])
    names, dims = _actor_term_contract(action_count)
    observed_names = tuple(
        item.strip() for item in str(md.get("observation_names", "")).split(",")
        if item.strip()
    )
    try:
        observed_dims = tuple(
            int(item) for item in str(md.get("actor_obs_term_dims", "")).split(",")
        )
    except ValueError as exc:
        raise PolicyGateError("invalid actor_obs_term_dims metadata") from exc
    expected = {
        "actor_obs_contract": str(
            trusted_action_set["actor_obs_contract"]
        ),
        "actor_obs_mode": "hitter_footwork",
        "actor_obs_total_dim": str(obs_dim),
        "training_contract_exact": "1",
        "training_contract_schema_version": "3",
        "action_ball_profile_id": str(trusted_action_set["profile_id"]),
        "action_ball_expected_n": str(action_count),
        "action_ball_scope": str(trusted_action_set["scope"]),
        "action_ball_mobility_mode": str(
            trusted_action_set["mobility_mode"]
        ),
        "action_ball_action_set_contract_sha256": str(
            trusted_action_set["contract_sha256"]
        ),
        "action_ball_manifest_sha256": str(
            trusted_action_set["manifest_sha256"]
        ),
        "action_ball_order_uid_digest_sha256": str(
            trusted_action_set["order_uid_digest_sha256"]
        ),
        "action_ball_action_set_contract_source_sha256": (
            fitted.native_diag.sha256_file(
                fitted.ACTION_SET_CONTRACT_SOURCE_PATH
            )
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": str(md.get(key, "")).strip()}
        for key, value in expected.items()
        if str(md.get(key, "")).strip() != value
    }
    if mismatches:
        raise PolicyGateError(f"ActionBall actor metadata mismatch: {mismatches}")
    if onnx_obs_dim != obs_dim:
        raise PolicyGateError(
            f"ONNX obs width {onnx_obs_dim} != trusted width {obs_dim}"
        )
    if _canonical_json_array_metadata(
        md,
        "action_ball_action_order",
        count=action_count,
        item_type=str,
    ) != tuple(trusted_action_set["ordered_action_ids"]):
        raise PolicyGateError(
            "ONNX action order differs from the trusted action-set contract"
        )
    try:
        metadata_uids = _canonical_json_array_metadata(
            md,
            "action_ball_ordered_action_uids",
            count=action_count,
            item_type=int,
        )
    except ValueError as exc:
        raise PolicyGateError(
            "ONNX action UID order is malformed"
        ) from exc
    if metadata_uids != tuple(trusted_action_set["ordered_action_uids"]):
        raise PolicyGateError(
            "ONNX action UID order differs from the trusted action-set contract"
        )
    try:
        metadata_schema = int(str(md.get("hope_metadata_schema_version", "0")))
    except ValueError as exc:
        raise PolicyGateError("invalid hope_metadata_schema_version") from exc
    if metadata_schema < 2:
        raise PolicyGateError(
            "formal ActionBall policy gate requires ONNX metadata schema >=2"
        )
    if observed_names != names or observed_dims != dims:
        raise PolicyGateError(
            "ONNX observation names/dims do not match the trusted action-set registry"
        )
    source_checkpoint = str(md.get("source_checkpoint_sha256", "")).strip().lower()
    if source_checkpoint != checkpoint_sha256:
        raise PolicyGateError(
            "ONNX source checkpoint does not match the pinned checkpoint bytes"
        )
    motion_sha = tuple(
        _require_sha(
            item.strip().lower(),
            f"motion_clip_sha256[{index}]",
        )
        for index, item in enumerate(
            str(md.get("motion_clip_sha256", "")).split(",")
        )
        if item.strip()
    )
    if len(motion_sha) != action_count:
        raise PolicyGateError(
            "ONNX must bind exactly N motion SHA256 values"
        )
    segment_lengths = tuple(
        int(float(item))
        for item in str(md.get("clip_seg_lengths", "")).split(",")
        if item.strip()
    )
    if len(segment_lengths) != action_count or any(
        item < 2 for item in segment_lengths
    ):
        raise PolicyGateError(
            "ONNX must bind N positive clip segment lengths"
        )
    physics_dt = float(str(md.get("physics_step_dt_s", "nan")))
    policy_dt = float(str(md.get("policy_step_dt_s", "nan")))
    decimation = int(str(md.get("control_decimation", "0")))
    if (
        not math.isfinite(physics_dt)
        or not math.isfinite(policy_dt)
        or physics_dt <= 0
        or abs(policy_dt - CONTROL_DT_S) > 1.0e-12
        or decimation <= 0
        or abs(physics_dt * decimation - policy_dt) > 1.0e-12
    ):
        raise PolicyGateError("ONNX policy/physics timestep contract is invalid")
    if str(md.get("action_use_default_offset", "")).strip() != "1":
        raise PolicyGateError(
            "formal ActionBall policy requires q_des=default_q+scale*action"
        )
    if str(md.get("qdes_clamp", "")).strip() != "1":
        raise PolicyGateError("formal ActionBall policy requires the training q_des clamp")
    if (
        str(md.get("joint_friction_backend", "")).strip() != "physx"
        or str(md.get("joint_friction_semantics", "")).strip()
        != "load_dependent_spatial_force_coefficient"
        or str(md.get("joint_friction_units", "")).strip() != "dimensionless"
    ):
        raise PolicyGateError("ONNX joint-friction provenance is not the exact PhysX contract")
    qdes_limits = _csv(md, "qdes_joint_pos_limits", count=62).reshape(31, 2)
    if np.any(qdes_limits[:, 0] > qdes_limits[:, 1]):
        raise PolicyGateError("qdes_joint_pos_limits contains lo > hi")
    return {
        "contract": trusted_action_set["actor_obs_contract"],
        "obs_dim": obs_dim,
        "action_set_contract": dict(trusted_action_set),
        "observation_names": list(names),
        "term_dims": list(dims),
        "source_checkpoint_sha256": source_checkpoint,
        "motion_clip_sha256": list(motion_sha),
        "clip_seg_lengths": list(segment_lengths),
        "training_physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "training_control_decimation": decimation,
        "qdes_joint_pos_limits": qdes_limits,
    }


def validate_teacher_receipt(
    receipt: Mapping[str, Any],
    manifest: fitted.PhysicalManifest,
) -> Dict[str, Any]:
    declared_payload_sha = str(receipt.get("receipt_payload_sha256", "")).lower()
    payload_without_sha = dict(receipt)
    payload_without_sha.pop("receipt_payload_sha256", None)
    actual_payload_sha = fitted.native_diag.sha256_bytes(
        fitted.native_diag.canonical_json_bytes(payload_without_sha)
    )
    if declared_payload_sha != actual_payload_sha:
        raise PolicyGateError("teacher fitted-ball receipt self-seal is invalid")
    if (
        receipt.get("gate") != "mujoco_teacher_motion_fitted_ball_gate"
        or receipt.get("status") != "PASS"
        or receipt.get("verdict") != "PASS"
        or receipt.get("analytic_return_scorer_executed") is not False
        or receipt.get("selector_executed") is not False
    ):
        raise PolicyGateError("teacher fitted-ball prerequisite is not a formal PASS")
    order = tuple(receipt.get("action_order") or ())
    if order != tuple(manifest.base.action_order):
        raise PolicyGateError("teacher receipt action order differs from the manifest")
    if receipt.get("action_set_contract") != dict(
        manifest.action_set_contract
    ):
        raise PolicyGateError(
            "teacher receipt action-set contract differs from the manifest"
        )
    rows = receipt.get("actions")
    if not isinstance(rows, list) or len(rows) != len(
        manifest.base.actions
    ):
        raise PolicyGateError(
            "teacher receipt must contain exactly N action rows"
        )
    by_action = {str(row.get("action_id")): row for row in rows if isinstance(row, dict)}
    if tuple(by_action) != tuple(manifest.base.action_order):
        raise PolicyGateError("teacher receipt action rows are missing, duplicated or reordered")
    controls_by_action: Dict[str, List[Dict[str, Any]]] = {}
    for action in manifest.base.actions:
        row = by_action[action.action_id]
        if (
            row.get("action_uid") != action.action_uid
            or row.get("motion_sha256") != action.motion_sha256
            or row.get("verdict") != "PASS"
        ):
            raise PolicyGateError(f"{action.action_id}: teacher identity/verdict mismatch")
        case_rows = (
            row.get("physical_task_binding", {}).get("cases")
            if isinstance(row.get("physical_task_binding"), dict)
            else None
        )
        if not isinstance(case_rows, list):
            raise PolicyGateError(f"{action.action_id}: teacher task cases missing")
        expected_cases = tuple(
            manifest.task_bindings[action.action_id].cases
        )
        if (
            len(case_rows) != len(expected_cases)
            or tuple(
                case.get("case_role")
                for case in case_rows
                if isinstance(case, dict)
            )
            != tuple(case.case_role for case in expected_cases)
        ):
            raise PolicyGateError(
                f"{action.action_id}: teacher task cases are missing, "
                "duplicated, extra, or reordered"
            )
        by_role = {str(case.get("case_role")): case for case in case_rows}
        controls: List[Dict[str, Any]] = []
        for case in expected_cases:
            observed = by_role.get(case.case_role)
            if (
                not isinstance(observed, dict)
                or observed.get("case_binding_sha256") != case.case_binding_sha256
                or observed.get("control_verdict") != "PASS"
            ):
                raise PolicyGateError(
                    f"{action.action_id}.{case.case_role}: teacher control is not bound PASS"
                )
            controls.append(
                {
                    "case_role": case.case_role,
                    "case_binding_sha256": case.case_binding_sha256,
                    "control_verdict": "PASS",
                }
            )
        controls_by_action[action.action_id] = controls
    return {
        "gate": receipt["gate"],
        "status": "PASS",
        "action_order": list(order),
        "receipt_payload_sha256": receipt.get("receipt_payload_sha256"),
        "controls_by_action": controls_by_action,
    }


class ActionBallPolicy:
    """Exact-N ONNX adapter; deliberately separate from the legacy scorer."""

    def __init__(
        self,
        *,
        onnx_path: Path,
        onnx_bytes: bytes,
        checkpoint_sha256: str,
        normalizer_path: Path,
        normalizer_sha256: str,
        trusted_action_set: Mapping[str, Any],
    ):
        self.action_set_contract = dict(trusted_action_set)
        self.action_count = int(trusted_action_set["expected_n"])
        self.obs_dim = int(trusted_action_set["actor_obs_width"])
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise PolicyGateError(f"onnxruntime unavailable: {exc}") from exc
        self.session = ort.InferenceSession(
            bytes(onnx_bytes), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if [item.name for item in inputs] != ["obs", "time_step"]:
            raise PolicyGateError("ONNX inputs must be exactly obs,time_step")
        if inputs[0].shape != [1, self.obs_dim] or inputs[1].shape != [1, 1]:
            raise PolicyGateError(
                f"ONNX input shapes must be [1,{self.obs_dim}] and [1,1]"
            )
        expected_outputs = (
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        if tuple(item.name for item in outputs) != expected_outputs:
            raise PolicyGateError("ONNX output order does not match the motion-policy contract")
        expected_shapes = (
            [1, 31],
            [1, 31],
            [1, 31],
            [1, 14, 3],
            [1, 14, 4],
            [1, 14, 3],
            [1, 14, 3],
        )
        if (
            any(item.type != "tensor(float)" for item in (*inputs, *outputs))
            or any(item.shape != shape for item, shape in zip(outputs, expected_shapes))
        ):
            raise PolicyGateError("ONNX tensor types/shapes differ from the exact policy ABI")
        self.out_names = expected_outputs
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.contract = validate_actor_metadata(
            self.metadata,
            checkpoint_sha256=checkpoint_sha256,
            onnx_obs_dim=self.obs_dim,
            trusted_action_set=trusted_action_set,
        )
        self.joint_names = _csv_text(self.metadata, "joint_names", count=31)
        self.body_names = _csv_text(self.metadata, "body_names", count=14)
        if tuple(self.body_names) != tuple(legacy_mj.TRACKED_BODIES):
            raise PolicyGateError("ONNX tracked-body order differs from the evaluator")
        self.default_q = _csv(self.metadata, "default_joint_pos", count=31)
        self.action_scale = _csv(self.metadata, "action_scale", count=31)
        self.kp = _csv(self.metadata, "joint_stiffness", count=31)
        self.kd = _csv(self.metadata, "joint_damping", count=31)
        self.qdes_joint_pos_limits = np.asarray(
            self.contract["qdes_joint_pos_limits"], np.float64
        )
        if (
            np.any(self.action_scale < 0.0)
            or np.any(self.kp < 0.0)
            or np.any(self.kd < 0.0)
        ):
            raise PolicyGateError("action scale and P-D gains must be non-negative")
        self.segment_lengths = tuple(self.contract["clip_seg_lengths"])
        self.segment_starts = tuple(
            sum(self.segment_lengths[:index])
            for index in range(self.action_count)
        )
        self.motion_sha = tuple(self.contract["motion_clip_sha256"])
        self.motion_hold_reference = str(
            self.metadata.get("motion_hold_reference", "")
        ).strip()
        if self.motion_hold_reference not in ("clip", "stand"):
            raise PolicyGateError(
                "formal ONNX must bind motion_hold_reference=clip|stand"
            )
        self.actuator_types = _csv_text(
            self.metadata, "joint_actuator_types", count=31
        )
        if any(item not in ("implicit", "explicit") for item in self.actuator_types):
            raise PolicyGateError("joint_actuator_types contains an unsupported value")
        self.joint_effort_limits = _csv(
            self.metadata, "joint_effort_limits", count=31
        )
        self.joint_armature = _csv(self.metadata, "joint_armature", count=31)
        self.joint_velocity_limits = _csv(
            self.metadata, "joint_velocity_limits", count=31
        )
        self.joint_friction_coefficients = _csv(
            self.metadata, "joint_friction_coefficients", count=31
        )
        if (
            np.any(self.joint_effort_limits <= 0)
            or np.any(self.joint_velocity_limits <= 0)
            or np.any(self.joint_armature < 0)
            or np.any(self.joint_friction_coefficients < 0)
        ):
            raise PolicyGateError(
                "effort/velocity limits must be positive and "
                "armature/friction must be non-negative"
            )
        normalizer_bytes = _read_pinned(
            normalizer_path, normalizer_sha256, "observation normalizer"
        )
        del normalizer_bytes  # np.load performs the structured validation below.
        if str(self.metadata.get("obs_norm_baked", "")).strip() == "1":
            raise PolicyGateError(
                "formal gate requires an unbaked actor plus its exact normalizer sidecar"
            )
        with np.load(normalizer_path, allow_pickle=False) as payload:
            required = {
                "mean",
                "std",
                "eps",
                "count",
                "source_checkpoint_sha256",
                "normalizer_state_sha256",
            }
            if not required.issubset(payload.files):
                raise PolicyGateError("normalizer sidecar lacks exact identity fields")
            self.obs_mean = np.asarray(payload["mean"], np.float64).reshape(-1)
            self.obs_std = np.asarray(payload["std"], np.float64).reshape(-1)
            self.obs_eps = float(np.asarray(payload["eps"]).item())
            count = int(np.asarray(payload["count"]).item())
            source = str(np.asarray(payload["source_checkpoint_sha256"]).item()).lower()
            state_sha = str(np.asarray(payload["normalizer_state_sha256"]).item()).lower()
        if (
            self.obs_mean.shape != (self.obs_dim,)
            or self.obs_std.shape != (self.obs_dim,)
            or not np.isfinite(self.obs_mean).all()
            or not np.isfinite(self.obs_std).all()
            or np.any(self.obs_std < 0)
            or self.obs_eps < 0
            or count <= 0
            or source != checkpoint_sha256
            or state_sha
            != legacy_mj.normalizer_state_sha256(
                self.obs_mean, self.obs_std, self.obs_eps, count
            )
            or str(self.metadata.get("obs_norm_sidecar_sha256", "")).strip().lower()
            != normalizer_sha256
        ):
            raise PolicyGateError("normalizer payload/ONNX/checkpoint binding mismatch")
        self.normalizer_contract = {
            "path": str(normalizer_path.resolve()),
            "sha256": normalizer_sha256,
            "count": count,
            "state_sha256": state_sha,
            "source_checkpoint_sha256": source,
        }

    def infer(self, obs: np.ndarray, time_step: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        obs = np.asarray(obs, np.float64)
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise PolicyGateError(
                f"actor observation must be finite shape ({self.obs_dim},)"
            )
        normalized = (obs - self.obs_mean) / (self.obs_std + self.obs_eps)
        outputs = self.session.run(
            None,
            {
                "obs": normalized[None].astype(np.float32),
                "time_step": np.asarray([[float(time_step)]], np.float32),
            },
        )
        rows = {name: outputs[index][0] for index, name in enumerate(self.out_names)}
        action = np.asarray(rows["actions"], np.float64)
        if action.shape != (31,) or not np.isfinite(action).all():
            raise PolicyGateError("actor emitted non-finite or wrong-shape action")
        return action, rows


class TaskCommand:
    """Duck-typed command object consumed by legacy_mj.build_obs' 177-D prefix."""

    def __init__(self, case: fitted.PhysicalTaskCase, action: Any):
        self.case = case
        self.racket_target_pos_w = np.asarray(case.racket_site_target_w_m, np.float64)
        self.racket_target_vel_w = np.asarray(case.racket_site_velocity_w_mps, np.float64)
        self.racket_target_normal_w = np.asarray(case.racket_normal_w, np.float64)
        self.base_target_pos_w = np.asarray(case.base_goal_w_m[:2], np.float64)
        self.time_to_strike = float(case.time_to_contact_s)
        self.swing_sign = float(action.mount_normal_sign)

    def racket_target_pos_b_rel_fk(
        self,
        base_pos_w: np.ndarray,
        base_quat_w: np.ndarray,
        racket_pos_w: np.ndarray,
    ) -> np.ndarray:
        # The 177-D ``hitter_footwork`` contract uses the yaw-heading frame,
        # not the fully tilted pelvis frame.  Roll/pitch must remain visible
        # through projected gravity instead of silently rotating the planner
        # target channels a second time.
        rotation = legacy_mj.mat_from_quat(legacy_mj.yaw_quat(base_quat_w))
        del base_pos_w
        return rotation.T @ (self.racket_target_pos_w - racket_pos_w)

    def base_target_pos_b(
        self, base_pos_w: np.ndarray, base_quat_w: np.ndarray
    ) -> np.ndarray:
        rotation = legacy_mj.mat_from_quat(legacy_mj.yaw_quat(base_quat_w))
        delta = np.zeros(3, np.float64)
        delta[:2] = self.base_target_pos_w - base_pos_w[:2]
        return (rotation.T @ delta)[:2]


def build_action_ball_obs(
    prefix_177: np.ndarray,
    *,
    target_normal_w: Sequence[float],
    action_slot: int,
    action_count: int = ACTION_COUNT,
) -> np.ndarray:
    prefix = np.asarray(prefix_177, np.float64)
    normal = np.asarray(target_normal_w, np.float64)
    if prefix.shape != (177,) or normal.shape != (3,) or not np.isfinite(normal).all():
        raise PolicyGateError("invalid ActionBall observation prefix or face normal")
    if type(action_count) is not int or action_count < 1:
        raise PolicyGateError("action_count must be a positive integer")
    if not 0 <= action_slot < action_count:
        raise PolicyGateError("action slot is outside the frozen action bank")
    one_hot = np.zeros(action_count, np.float64)
    one_hot[action_slot] = 1.0
    out = np.concatenate((prefix, normal, np.zeros(1), one_hot))
    if out.shape != (BASE_OBS_DIM + action_count,):
        raise PolicyGateError("internal ActionBall observation assembly mismatch")
    return out


def _collision_pair_enabled(model: Any, geom_a: int, geom_b: int) -> bool:
    return bool(
        (
            int(model.geom_contype[geom_a])
            & int(model.geom_conaffinity[geom_b])
        )
        or (
            int(model.geom_contype[geom_b])
            & int(model.geom_conaffinity[geom_a])
        )
    )


def _robot_safety_geom_ids(
    robot: legacy_mj.MujocoRobot,
) -> np.ndarray:
    """Every physics-enabled geom in the pelvis robot subtree, including feet."""

    obstacle_ids = tuple(
        int(
            robot.mj.mj_name2id(
                robot.model, robot.mj.mjtObj.mjOBJ_GEOM, name
            )
        )
        for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    )
    if any(geom_id < 0 for geom_id in obstacle_ids):
        raise PolicyGateError("compiled fitted scene lacks a five-solid obstacle")
    ids = [
        int(geom_id)
        for geom_id in np.flatnonzero(robot.robot_geom_mask)
        if any(
            _collision_pair_enabled(robot.model, int(geom_id), obstacle_id)
            for obstacle_id in obstacle_ids
        )
    ]
    if not ids:
        raise PolicyGateError(
            "five-solid safety guard found no physics-enabled robot geoms"
        )
    return np.asarray(ids, np.int64)


def _bind_embedded_table_sensor(
    robot: legacy_mj.MujocoRobot,
    geometry_contract: Mapping[str, Any],
) -> None:
    mujoco = robot.mj
    ids = []
    for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES:
        geom_id = int(
            mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        )
        if geom_id < 0:
            raise PolicyGateError(f"compiled fitted scene lacks obstacle geom {name}")
        ids.append(geom_id)
    racket_geom = int(
        mujoco.mj_name2id(
            robot.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            legacy_mj.RACKET_COLLISION_GEOM_NAME,
        )
    )
    if racket_geom < 0:
        raise PolicyGateError("compiled fitted scene lacks racket collision geom")
    robot.table_geom_mask[:] = False
    robot.table_geom_mask[np.asarray(ids, np.int64)] = True
    robot.racket_collision_geom = racket_geom
    robot.table_contact_contract = {
        "available": True,
        "force_threshold_n": TABLE_CONTACT_FORCE_THRESHOLD_N,
        "source": "action_ball_policy_five_solid_embedded_obstacle_geoms",
        "obstacles": list(
            fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
        ),
        "five_solid_geometry_sha256": geometry_contract["sha256"],
        "feet_included": True,
    }
    robot.action_ball_safety_geom_ids = _robot_safety_geom_ids(robot)


def _five_solid_contact_scan(
    robot: legacy_mj.MujocoRobot,
) -> Dict[str, Any]:
    """Read every resolved five-solid/robot pair, including foot pairs."""

    model, data, mj = robot.model, robot.data, robot.mj
    obstacle_by_id = {
        int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)): name
        for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    robot_ids = frozenset(
        int(value) for value in robot.action_ball_safety_geom_ids
    )
    per_obstacle = {
        name: 0
        for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    force6 = np.zeros(6, np.float64)
    count = 0
    max_force = 0.0
    max_penetration = 0.0
    worst_pair = ""
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in obstacle_by_id and g2 in robot_ids:
            obstacle_id, robot_id = g1, g2
        elif g2 in obstacle_by_id and g1 in robot_ids:
            obstacle_id, robot_id = g2, g1
        else:
            continue
        force6.fill(0.0)
        mj.mj_contactForce(model, data, contact_index, force6)
        force_n = float(np.linalg.norm(force6[:3]))
        penetration = max(0.0, -float(contact.dist))
        if not (math.isfinite(force_n) and math.isfinite(penetration)):
            raise PolicyGateError(
                "non-finite five-solid robot contact evidence"
            )
        # ActionBall is a no-touch task.  Match the training sensor's tiny
        # force threshold, while the penetration branch also rejects a
        # resolved overlap whose instantaneous solver force is numerically 0.
        if (
            force_n <= TABLE_CONTACT_FORCE_THRESHOLD_N
            and penetration <= 0.0
        ):
            continue
        obstacle_name = obstacle_by_id[obstacle_id]
        per_obstacle[obstacle_name] += 1
        count += 1
        if force_n >= max_force or penetration >= max_penetration:
            max_force = max(max_force, force_n)
            max_penetration = max(max_penetration, penetration)
            robot_name = (
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, robot_id)
                or f"geom{robot_id}"
            )
            worst_pair = f"{robot_name}~{obstacle_name}"
    return {
        "contact_count": count,
        "max_force_n": max_force,
        "max_penetration_m": max_penetration,
        "worst_pair": worst_pair,
        "per_obstacle": per_obstacle,
    }


def _segment_intersects_inflated_aabb(
    start: Sequence[float],
    end: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    inflation_m: float,
) -> bool:
    """Exact slab test for a line segment against an inflated AABB."""

    p0 = np.asarray(start, np.float64)
    p1 = np.asarray(end, np.float64)
    lower = np.asarray(lo, np.float64) - float(inflation_m)
    upper = np.asarray(hi, np.float64) + float(inflation_m)
    if (
        p0.shape != (3,)
        or p1.shape != (3,)
        or lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(
            np.concatenate((p0, p1, lower, upper))
        ).all()
        or not math.isfinite(float(inflation_m))
        or float(inflation_m) < 0.0
        or np.any(upper < lower)
    ):
        raise PolicyGateError("invalid swept-sphere/AABB input")
    delta = p1 - p0
    t_enter, t_exit = 0.0, 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-15:
            if p0[axis] < lower[axis] or p0[axis] > upper[axis]:
                return False
            continue
        inv = 1.0 / float(delta[axis])
        first = (float(lower[axis]) - float(p0[axis])) * inv
        second = (float(upper[axis]) - float(p0[axis])) * inv
        if first > second:
            first, second = second, first
        t_enter = max(t_enter, first)
        t_exit = min(t_exit, second)
        if t_enter > t_exit:
            return False
    return True


def _five_solid_swept_scan(
    robot: legacy_mj.MujocoRobot,
    centers_before: np.ndarray,
    obstacle_aabbs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Conservatively sweep every collision geom between physics states."""

    ids = np.asarray(robot.action_ball_safety_geom_ids, np.int64)
    centers_after = np.asarray(robot.data.geom_xpos[ids], np.float64)
    centers_before = np.asarray(centers_before, np.float64)
    if (
        centers_before.shape != centers_after.shape
        or centers_before.shape != (ids.size, 3)
        or not np.isfinite(centers_before).all()
        or not np.isfinite(centers_after).all()
    ):
        raise PolicyGateError("five-solid swept geom centers are invalid")
    per_obstacle = {
        name: 0
        for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    first_hit = None
    hit_count = 0
    for row_index, geom_id in enumerate(ids):
        radius = float(robot.model.geom_rbound[int(geom_id)])
        if not math.isfinite(radius) or radius < 0.0:
            raise PolicyGateError("robot geom has invalid MuJoCo rbound")
        for obstacle_name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES:
            lo, hi = obstacle_aabbs[obstacle_name]
            if not _segment_intersects_inflated_aabb(
                centers_before[row_index],
                centers_after[row_index],
                lo,
                hi,
                radius,
            ):
                continue
            hit_count += 1
            per_obstacle[obstacle_name] += 1
            if first_hit is None:
                geom_name = (
                    robot.mj.mj_id2name(
                        robot.model,
                        robot.mj.mjtObj.mjOBJ_GEOM,
                        int(geom_id),
                    )
                    or f"geom{int(geom_id)}"
                )
                first_hit = {
                    "robot_geom": geom_name,
                    "obstacle": obstacle_name,
                    "geom_rbound_m": radius,
                    "center_before_m": centers_before[row_index].tolist(),
                    "center_after_m": centers_after[row_index].tolist(),
                }
    return {
        "hit_count": hit_count,
        "per_obstacle": per_obstacle,
        "first_hit": first_hit,
        "method": FIVE_SOLID_SWEEP_METHOD,
    }


def _validate_compiled_five_solid_scene(
    robot: legacy_mj.MujocoRobot,
    geometry_rows: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    *,
    assembled_xml_sha256: str,
) -> Dict[str, Any]:
    """Bind exact geometry/filter bytes before any learned-policy rollout."""

    mj, model, data = robot.mj, robot.model, robot.data
    recomputed = fitted.table_scene.action_ball_policy_geometry_contract(
        geometry_rows
    )
    if recomputed != dict(geometry_contract):
        raise PolicyGateError("five-solid geometry contract drifted")
    mj.mj_forward(model, data)
    compiled_rows = []
    for row in fitted.table_scene.action_ball_policy_obstacle_rows(
        geometry_rows
    ):
        name = str(row["name"])
        geom_id = int(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        )
        if geom_id < 0:
            raise PolicyGateError(f"compiled policy scene lacks {name}")
        expected_center = np.asarray(
            row["center_mjcf_world_m"], np.float64
        )
        expected_half = 0.5 * np.asarray(
            row["full_extents_m"], np.float64
        )
        checks = (
            int(model.geom_type[geom_id])
            == int(mj.mjtGeom.mjGEOM_BOX),
            int(model.geom_bodyid[geom_id]) == 0,
            int(model.geom_contype[geom_id]) == 0,
            int(model.geom_conaffinity[geom_id]) == 7,
            np.allclose(
                data.geom_xpos[geom_id],
                expected_center,
                atol=1.0e-12,
                rtol=0.0,
            ),
            np.allclose(
                model.geom_size[geom_id, :3],
                expected_half,
                atol=1.0e-12,
                rtol=0.0,
            ),
            np.allclose(
                np.asarray(data.geom_xmat[geom_id]).reshape(3, 3),
                np.eye(3),
                atol=1.0e-12,
                rtol=0.0,
            ),
        )
        if not all(bool(value) for value in checks):
            raise PolicyGateError(
                f"compiled five-solid geometry/filter mismatch for {name}"
            )
        compiled_rows.append(
            {
                "name": name,
                "geom_id": geom_id,
                "contype": 0,
                "conaffinity": 7,
                "center_mjcf_world_m": expected_center.tolist(),
                "full_extents_m": (2.0 * expected_half).tolist(),
            }
        )
    ball_geom = int(
        mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, fitted.BALL_GEOM_NAME
        )
    )
    keepout_geom = int(
        mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_GEOM,
            fitted.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME,
        )
    )
    if min(ball_geom, keepout_geom) < 0:
        raise PolicyGateError("compiled policy scene lacks ball or keepout")
    if (
        int(model.geom_contype[ball_geom]) != 0
        or int(model.geom_conaffinity[ball_geom]) != 0
        or _collision_pair_enabled(model, ball_geom, keepout_geom)
    ):
        raise PolicyGateError(
            "robot-only keepout can affect the fitted ball"
        )
    analytic_ball_obstacles = fitted._obstacle_aabbs(geometry_rows)
    if (
        fitted.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
        in analytic_ball_obstacles
    ):
        raise PolicyGateError(
            "robot-only keepout entered the analytic ball referee"
        )
    robot_ids = np.asarray(robot.action_ball_safety_geom_ids, np.int64)
    if any(
        not _collision_pair_enabled(model, int(geom_id), keepout_geom)
        for geom_id in robot_ids
    ):
        raise PolicyGateError(
            "a guarded robot collision geom is filtered from the keepout"
        )
    return {
        "five_solid_geometry_sha256": geometry_contract["sha256"],
        "assembled_xml_sha256": _require_sha(
            assembled_xml_sha256, "assembled policy XML SHA256"
        ),
        "compiled_obstacles": compiled_rows,
        "robot_collision_geom_count": int(robot_ids.size),
        "robot_subject_includes_feet": True,
        "ball_keepout_native_pair_enabled": False,
        "ball_keepout_analytic_surface_enabled": False,
        "continuous_sweep_method": FIVE_SOLID_SWEEP_METHOD,
    }


@contextlib.contextmanager
def _materialized_fitted_xml(xml: bytes) -> Iterable[Path]:
    # Put the temporary root beside the canonical MJCF so every relative mesh
    # path has exactly the same resolution semantics as the trusted source.
    directory = fitted.CANONICAL_MJCF.resolve().parent
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".action_ball_policy_fitted.", suffix=".xml", dir=directory
    )
    path = Path(raw_path)
    try:
        os.write(descriptor, xml)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield path
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _model_robot(
    *,
    policy: ActionBallPolicy,
    xml: bytes,
    dt: float,
    geometry_contract: Mapping[str, Any],
) -> legacy_mj.MujocoRobot:
    with _materialized_fitted_xml(xml) as path:
        robot = legacy_mj.MujocoRobot(
            str(path),
            list(policy.joint_names),
            list(policy.body_names),
            dt,
            False,
            False,
            "implicit",
            kd_for_implicit=policy.kd,
            actuator_types=policy.actuator_types,
            joint_armature=policy.joint_armature,
            joint_velocity_limits=policy.joint_velocity_limits,
            joint_effort_limits=policy.joint_effort_limits,
            require_bound_plant_match=True,
            allow_velocity_limit_proxy=False,
            allow_effort_limit_proxy=False,
            fail_on_self_contact=False,
            with_table_obstacle=False,
        )
    _bind_embedded_table_sensor(robot, geometry_contract)
    return robot


def _reset_policy_state(
    robot: legacy_mj.MujocoRobot,
    clip: Any,
    case: fitted.PhysicalTaskCase,
) -> None:
    teacher = fitted.retimed_teacher_state(
        clip,
        world_time_s=0.0,
        pre_swing_wait_s=case.pre_swing_wait_s,
        teacher_rate=case.teacher_rate,
    )
    root_pos = np.asarray(teacher["root_pos"], np.float64).copy()
    root_pos[:2] = np.asarray(case.base_spawn_w_m[:2], np.float64)
    robot.reset_to_reference(
        root_pos,
        np.asarray(teacher["root_quat"], np.float64),
        np.asarray(teacher["root_lin_vel"], np.float64),
        np.asarray(teacher["root_ang_vel"], np.float64),
        clip.body_lin_vel_point,
        np.asarray(teacher["joint_pos"], np.float64),
        np.asarray(teacher["joint_vel"], np.float64),
    )


def _phase_step(
    *,
    world_time_s: float,
    case: fitted.PhysicalTaskCase,
    clip: Any,
    segment_start: int,
    segment_length: int,
) -> int:
    source_time = max(
        0.0, (float(world_time_s) - case.pre_swing_wait_s) * case.teacher_rate
    )
    local = min(segment_length - 1, int(math.floor(source_time * clip.fps + 1.0e-9)))
    return segment_start + local


def _grade_case(
    *,
    events: fitted.FittedEvents,
    case: fitted.PhysicalTaskCase,
    safety: Mapping[str, Any],
    imitation: Mapping[str, Any],
) -> Tuple[str, list[str]]:
    reasons: list[str] = []
    if events.paddle_impulse_count != 1 or events.paddle_contact is None:
        reasons.append("selected_face_contact_count_not_one")
    else:
        contact_time = float(events.paddle_contact["time_s"])
        if abs(contact_time - case.time_to_contact_s) > CONTACT_TIME_TOLERANCE_S:
            reasons.append("contact_not_at_bound_strike_window")
    if events.net_crossing is None:
        reasons.append("no_post_contact_net_crossing")
    if events.first_landing is None:
        reasons.append("no_first_table_landing")
    else:
        landing = np.asarray(events.first_landing["ball_center_xy_m"], np.float64)
        if np.linalg.norm(landing - case.landing_aim_w_xy_m) > LANDING_AIM_TOLERANCE_M:
            reasons.append("first_landing_misses_bound_task_aim")
    if events.return_table_bounces != 1:
        reasons.append("return_table_bounce_count_not_one")
    if events.event_order_violations:
        reasons.append("physical_event_order_violation")
    if events.ball_net_collision is not None:
        reasons.append("ball_hit_net")
    for key in (
        "table_contact_steps",
        "table_swept_guard_steps",
        "self_contact_steps",
        "hard_joint_limit_steps",
        "velocity_limit_steps",
        "fall_steps",
        "qdes_clamp_joint_commands",
    ):
        if int(safety.get(key, 0)) != 0:
            reasons.append(key)
    if not imitation.get("finite", False):
        reasons.append("teacher_imitation_metrics_nonfinite")
    strike = imitation.get("strike")
    if not isinstance(strike, Mapping):
        reasons.append("missing_bound_strike_frame_snapshot")
    else:
        for key, limit, reason in (
            (
                "racket_site_target_position_error_m",
                STRIKE_POSITION_ERROR_TOLERANCE_M,
                "bound_strike_position_error_exceeds_gate",
            ),
            (
                "racket_site_target_velocity_error_mps",
                STRIKE_VELOCITY_ERROR_TOLERANCE_MPS,
                "bound_strike_velocity_error_exceeds_gate",
            ),
            (
                "racket_face_normal_angle_error_rad",
                STRIKE_NORMAL_ERROR_TOLERANCE_RAD,
                "bound_strike_face_error_exceeds_gate",
            ),
        ):
            value = strike.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) > limit
            ):
                reasons.append(reason)
    return ("PASS" if not reasons else "FAIL"), reasons


def _run_case_dt(
    *,
    policy: ActionBallPolicy,
    robot: legacy_mj.MujocoRobot,
    action: Any,
    action_slot: int,
    clip: Any,
    case: fitted.PhysicalTaskCase,
    venue: fitted.VenueParams,
    profile: Mapping[str, Any],
    face_mesh: fitted.FaceMesh,
    obstacle_rows: Mapping[str, Any],
    capture_frames: bool,
    render_fps: int,
) -> Tuple[Dict[str, Any], list[np.ndarray]]:
    mj = robot.mj
    model, data = robot.model, robot.data
    _reset_policy_state(robot, clip, case)
    ball_joint = int(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, fitted.BALL_JOINT_NAME)
    )
    ball_body = int(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, fitted.BALL_BODY_NAME)
    )
    ball_geom = int(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, fitted.BALL_GEOM_NAME)
    )
    if min(ball_joint, ball_body, ball_geom) < 0:
        raise PolicyGateError("compiled model lacks the fitted ball")
    qadr = int(model.jnt_qposadr[ball_joint])
    dadr = int(model.jnt_dofadr[ball_joint])
    binding = fitted.motion_player.bind_model(mj, model)
    aabbs = fitted._obstacle_aabbs(obstacle_rows)
    safety_aabbs = (
        fitted.table_scene.action_ball_policy_obstacle_aabbs(
            obstacle_rows
        )
    )
    safety_geom_ids = np.asarray(
        robot.action_ball_safety_geom_ids, np.int64
    )
    command = TaskCommand(case, action)
    events = fitted.FittedEvents()
    safety = {
        "table_contact_steps": 0,
        "table_contact_events": 0,
        "table_contact_per_obstacle": {
            name: 0
            for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
        },
        "table_swept_guard_steps": 0,
        "table_swept_guard_hits": 0,
        "table_swept_guard_per_obstacle": {
            name: 0
            for name in fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
        },
        "table_swept_guard_first_hit": None,
        "table_swept_guard_method": FIVE_SOLID_SWEEP_METHOD,
        "self_contact_steps": 0,
        "hard_joint_limit_steps": 0,
        "velocity_limit_steps": 0,
        "fall_steps": 0,
        "qdes_clamp_joint_commands": 0,
        "qdes_total_joint_commands": 0,
    }
    dt = float(model.opt.timestep)
    decimation = int(round(CONTROL_DT_S / dt))
    if decimation <= 0 or abs(decimation * dt - CONTROL_DT_S) > 1.0e-12:
        raise PolicyGateError("formal MuJoCo dt does not divide the 20 ms policy period")
    total_time = max(
        case.pre_swing_wait_s + case.scaled_t_cycle_s,
        case.time_to_contact_s + fitted.FORMAL_POST_CONTACT_S,
    )
    frames: list[np.ndarray] = []
    renderer = None
    if capture_frames:
        try:
            renderer = mj.Renderer(model, height=720, width=960)
        except Exception:
            renderer = None
    render_stride = max(1, int(round(1.0 / (render_fps * dt))))
    next_render = 0
    last_action = np.zeros(31, np.float64)
    active = False
    returned = False
    control_steps = int(math.ceil(total_time / CONTROL_DT_S)) + 1
    joint_error_l2: list[float] = []
    body_error_l2: list[float] = []
    strike_snapshot: Optional[Dict[str, Any]] = None
    first_action: Optional[np.ndarray] = None
    contact_before: Optional[float] = None

    data.qpos[qadr : qadr + 3] = (0.0, 0.0, 100.0)
    data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(model, data)

    for control_index in range(control_steps):
        world_time = float(data.time)
        command.time_to_strike = case.time_to_contact_s - world_time
        time_step = _phase_step(
            world_time_s=world_time,
            case=case,
            clip=clip,
            segment_start=policy.segment_starts[action_slot],
            segment_length=policy.segment_lengths[action_slot],
        )
        # One inference obtains the actor action and the exact internal teacher
        # side outputs.  A zero observation is safe for the side outputs; the
        # real observation is then assembled and inferred once for control.
        _, refs = policy.infer(np.zeros(policy.obs_dim), time_step)
        if (
            policy.motion_hold_reference == "stand"
            and world_time < case.pre_swing_wait_s
        ):
            refs = legacy_mj.stand_hold_refs(refs, policy.default_q)
        prefix, *_ = legacy_mj.build_obs(
            refs,
            robot,
            command,
            last_action,
            policy.default_q,
            hitter=True,
        )
        obs = build_action_ball_obs(
            prefix,
            target_normal_w=case.racket_normal_w,
            action_slot=action_slot,
            action_count=policy.action_count,
        )
        action_out, refs = policy.infer(obs, time_step)
        if (
            policy.motion_hold_reference == "stand"
            and world_time < case.pre_swing_wait_s
        ):
            refs = legacy_mj.stand_hold_refs(refs, policy.default_q)
        if first_action is None:
            first_action = action_out.copy()
        last_action = action_out.copy()
        qdes_raw = policy.default_q + action_out * policy.action_scale
        qdes = np.clip(
            qdes_raw,
            policy.qdes_joint_pos_limits[:, 0],
            policy.qdes_joint_pos_limits[:, 1],
        )
        safety["qdes_clamp_joint_commands"] += int(np.count_nonzero(qdes != qdes_raw))
        safety["qdes_total_joint_commands"] += 31
        q_ref = np.asarray(refs["joint_pos"], np.float64)
        joint_error_l2.append(float(np.linalg.norm(robot.q_artic() - q_ref)))
        actual_body = robot.tracked_pos()
        ref_body = np.asarray(refs["body_pos_w"], np.float64)
        robot_anchor_pos = robot.body_pos(robot.torso_bid)
        robot_anchor_quat = robot.body_quat(robot.torso_bid)
        ref_anchor_pos = ref_body[legacy_mj.ANCHOR_TRACKED_IDX]
        ref_anchor_quat = np.asarray(
            refs["body_quat_w"][legacy_mj.ANCHOR_TRACKED_IDX], np.float64
        )
        ref_body_relative = legacy_mj.body_pos_relative_w(
            ref_body,
            ref_anchor_pos,
            ref_anchor_quat,
            robot_anchor_pos,
            robot_anchor_quat,
        )
        body_error_l2.append(
            float(np.sqrt(np.mean(np.square(actual_body - ref_body_relative))))
        )
        if strike_snapshot is None and world_time + 0.5 * CONTROL_DT_S >= case.time_to_contact_s:
            face = fitted._face_state(
                mj, model, data, binding, action.mount_normal_sign
            )
            normal_dot = float(
                np.clip(face.normal_w @ case.racket_normal_w, -1.0, 1.0)
            )
            strike_snapshot = {
                "time_s": world_time,
                "joint_l2_rad": joint_error_l2[-1],
                "tracked_body_rmse_m": body_error_l2[-1],
                "racket_site_position_w_m": robot.racket_pos().tolist(),
                "racket_site_velocity_w_mps": robot.racket_lin_vel_w().tolist(),
                "racket_site_target_position_error_m": float(
                    np.linalg.norm(
                        robot.racket_pos() - case.racket_site_target_w_m
                    )
                ),
                "racket_site_target_velocity_error_mps": float(
                    np.linalg.norm(
                        robot.racket_lin_vel_w()
                        - case.racket_site_velocity_w_mps
                    )
                ),
                "racket_face_normal_angle_error_rad": math.acos(normal_dot),
            }

        explicit = np.asarray(policy.actuator_types) == "explicit"
        for _substep in range(decimation):
            time_s = float(data.time)
            if not active and time_s + 0.5 * dt >= case.launch.activation_time_s:
                data.qpos[qadr : qadr + 3] = case.launch.position_w_m
                data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
                data.qvel[dadr : dadr + 3] = case.launch.velocity_w_mps
                fitted.set_ball_spin_world(
                    data, qadr, dadr, case.launch.spin_w_radps
                )
                active = True
                events.activation_time_s = time_s
                mj.mj_forward(model, data)
            face_before = fitted._face_state(
                mj, model, data, binding, action.mount_normal_sign
            )
            if active:
                p0 = np.asarray(data.qpos[qadr : qadr + 3], np.float64).copy()
                v0 = np.asarray(data.qvel[dadr : dadr + 3], np.float64).copy()
                w0 = fitted.ball_spin_world(data, qadr, dadr)
                aero = (
                    -venue.k_d * float(np.linalg.norm(v0)) * v0
                    + venue.k_m * np.cross(w0, v0)
                )
                data.xfrc_applied[ball_body, :3] = venue.ball_mass * aero
            else:
                p0 = v0 = w0 = None
                data.xfrc_applied[ball_body, :] = 0.0
            q = robot.q_artic()
            qd = robot.qd_artic()
            if not np.isfinite(qd).all():
                raise PolicyGateError("non-finite joint velocity before MuJoCo step")
            if np.any(np.abs(qd) > policy.joint_velocity_limits * (1.0 + 1.0e-9)):
                safety["velocity_limit_steps"] += 1
            proportional = policy.kp * (qdes - q)
            tau = proportional - policy.kd * qd
            tau = np.clip(tau, -policy.joint_effort_limits, policy.joint_effort_limits)
            # Both explicit and implicit training actuators apply the same
            # clipped total P-D law on this exact formal lane.
            if explicit.shape != tau.shape:
                raise PolicyGateError("actuator mask shape drift")
            data.ctrl[robot.act_id] = np.clip(tau, robot.ctrl_lo, robot.ctrl_hi)
            geom_centers_before = np.asarray(
                data.geom_xpos[safety_geom_ids], np.float64
            ).copy()
            mj.mj_step(model, data)
            face_after = fitted._face_state(
                mj, model, data, binding, action.mount_normal_sign
            )
            table = _five_solid_contact_scan(robot)
            if int(table["contact_count"]):
                safety["table_contact_steps"] += 1
                safety["table_contact_events"] += int(
                    table["contact_count"]
                )
                for name, count in table["per_obstacle"].items():
                    safety["table_contact_per_obstacle"][name] += int(
                        count
                    )
            swept = _five_solid_swept_scan(
                robot, geom_centers_before, safety_aabbs
            )
            if int(swept["hit_count"]):
                safety["table_swept_guard_steps"] += 1
                safety["table_swept_guard_hits"] += int(
                    swept["hit_count"]
                )
                for name, count in swept["per_obstacle"].items():
                    safety["table_swept_guard_per_obstacle"][name] += int(
                        count
                    )
                if safety["table_swept_guard_first_hit"] is None:
                    safety["table_swept_guard_first_hit"] = dict(
                        swept["first_hit"]
                    )
            self_count, _penetration, _pair = robot.self_contact_scan()
            if self_count:
                safety["self_contact_steps"] += 1
            q_after = robot.q_artic()
            limited = np.asarray(model.jnt_limited)[
                [
                    mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                    for name in policy.joint_names
                ]
            ].astype(bool)
            ranges = np.asarray(
                [
                    model.jnt_range[
                        mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                    ]
                    for name in policy.joint_names
                ]
            )
            if np.any(
                limited
                & ((q_after < ranges[:, 0] - 1.0e-7) | (q_after > ranges[:, 1] + 1.0e-7))
            ):
                safety["hard_joint_limit_steps"] += 1
            qd_after = robot.qd_artic()
            if np.any(
                np.abs(qd_after) > policy.joint_velocity_limits * (1.0 + 1.0e-9)
            ):
                safety["velocity_limit_steps"] += 1
            root_z = float(data.qpos[binding.root_qpos_adr + 2])
            root_tilt = fitted.native_diag._root_tilt_rad(
                data.qpos[binding.root_qpos_adr + 3 : binding.root_qpos_adr + 7]
            )
            if root_z < ROOT_Z_FALL_M or root_tilt > ROOT_TILT_FALL_RAD:
                safety["fall_steps"] += 1
            if active:
                p1 = np.asarray(data.qpos[qadr : qadr + 3], np.float64).copy()
                v1 = np.asarray(data.qvel[dadr : dadr + 3], np.float64).copy()
                w1 = fitted.ball_spin_world(data, qadr, dadr)
                p1, v1, w1, returned, _segments = (
                    fitted.process_surface_events_chronologically(
                        p0=p0,
                        p1=p1,
                        v0=v0,
                        v1=v1,
                        w0=w0,
                        w1=w1,
                        time_s=time_s,
                        dt=dt,
                        face_before=face_before,
                        face_after=face_after,
                        face_mesh=face_mesh,
                        action=action,
                        venue=venue,
                        profile=profile,
                        aabbs=aabbs,
                        events=events,
                        returned=returned,
                    )
                )
                data.qpos[qadr : qadr + 3] = p1
                data.qvel[dadr : dadr + 3] = v1
                fitted.set_ball_spin_world(data, qadr, dadr, w1)
                mj.mj_forward(model, data)
                if events.paddle_contact is not None and contact_before is None:
                    contact_before = float(events.paddle_contact["time_s"])
            if capture_frames and int(round(data.time / dt)) >= next_render:
                if renderer is not None:
                    try:
                        renderer.update_scene(data, camera="torso_follow")
                        frames.append(renderer.render().copy())
                    except Exception:
                        renderer = None
                next_render += render_stride
        if float(data.time) + 1.0e-12 >= total_time:
            break
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    imitation_values = np.asarray(joint_error_l2 + body_error_l2, np.float64)
    imitation = {
        "finite": bool(imitation_values.size and np.isfinite(imitation_values).all()),
        "joint_l2_rad_mean": float(np.mean(joint_error_l2)),
        "joint_l2_rad_max": float(np.max(joint_error_l2)),
        "tracked_body_rmse_m_mean": float(np.mean(body_error_l2)),
        "tracked_body_rmse_m_max": float(np.max(body_error_l2)),
        "strike": strike_snapshot,
    }
    verdict, reasons = _grade_case(
        events=events, case=case, safety=safety, imitation=imitation
    )
    landing = (
        None
        if events.first_landing is None
        else events.first_landing.get("ball_center_xy_m")
    )
    result = {
        "verdict": verdict,
        "failure_reasons": reasons,
        "dt_s": dt,
        "policy_control_dt_s": CONTROL_DT_S,
        "decimation": decimation,
        "attempted": True,
        "action_slot": action_slot,
        "case_id": case.case_id,
        "case_role": case.case_role,
        "case_binding_sha256": case.case_binding_sha256,
        "first_action": None if first_action is None else first_action.tolist(),
        "contact": events.paddle_contact,
        "contact_time_s": contact_before,
        "net_crossing": events.net_crossing,
        "first_landing_xy_m": landing,
        "return_table_bounces": events.return_table_bounces,
        "incoming_table_bounces": events.incoming_table_bounces,
        "ball_net_collision": events.ball_net_collision,
        "event_order_violations": list(events.event_order_violations),
        "safety": safety,
        "imitation": imitation,
        "selector_executed": False,
        "solver_executed_by_gate": False,
        "virtual_scorer_executed": False,
    }
    return result, frames


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--action-set-profile", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument(
        "--physical-gate-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--physical-gate-manifest-sha256", required=True
    )
    parser.add_argument(
        "--physical-gate-materialization-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--physical-gate-materialization-receipt-sha256",
        required=True,
    )
    parser.add_argument("--profile-pins", type=Path, required=True)
    parser.add_argument("--profile-pins-sha256", required=True)
    parser.add_argument("--launch-trust-root", type=Path, required=True)
    parser.add_argument("--launch-trust-root-sha256", required=True)
    parser.add_argument("--teacher-gate-receipt", type=Path, required=True)
    parser.add_argument("--teacher-gate-receipt-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-sha256", required=True)
    parser.add_argument("--obs-normalizer", type=Path, required=True)
    parser.add_argument("--obs-normalizer-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode()
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.out.exists() or args.render_dir.exists():
        print("[policy-fitted-gate][FATAL] refusing existing output", file=sys.stderr)
        return 2
    blockers: list[str] = []
    evidence: Dict[str, Any] = {}
    manifest = profile = venue = policy = None
    policy_obstacle_rows = None
    safety_geometry_contract = None
    teacher_controls: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_sha = _require_sha(args.checkpoint_sha256, "checkpoint SHA256")
    onnx_sha = _require_sha(args.onnx_sha256, "ONNX SHA256")
    normalizer_sha = _require_sha(args.obs_normalizer_sha256, "normalizer SHA256")
    try:
        policy_obstacle_rows = (
            fitted.table_scene.action_ball_policy_obstacle_geometry()
        )
        safety_geometry_contract = (
            fitted.table_scene.action_ball_policy_geometry_contract(
                policy_obstacle_rows
            )
        )
        evidence["five_solid_safety_geometry"] = {
            "five_solid_geometry_sha256": safety_geometry_contract[
                "sha256"
            ],
            "payload": safety_geometry_contract["payload"],
            "collision_filter_semantics": {
                "world_obstacle_contype": 0,
                "world_obstacle_conaffinity": 7,
                "robot_subject": (
                    "all physics-enabled pelvis-subtree collision geoms "
                    "including feet"
                ),
                "ball_contype": 0,
                "ball_conaffinity": 0,
                "under_table_keepout_role": "robot_only",
            },
            "continuous_sweep_method": FIVE_SOLID_SWEEP_METHOD,
        }
    except Exception as exc:
        blockers.append(f"five_solid_safety_geometry:{exc}")
    try:
        # Reuse the exact teacher preflight, including clean checkout, launch
        # trust root, manifest task/solver receipts and profile source hashes.
        teacher_args = argparse.Namespace(
            code_commit=args.code_commit,
            action_set_profile=args.action_set_profile,
            training_manifest=args.training_manifest,
            training_manifest_sha256=args.training_manifest_sha256,
            physical_gate_manifest=args.physical_gate_manifest,
            physical_gate_manifest_sha256=(
                args.physical_gate_manifest_sha256
            ),
            physical_gate_materialization_receipt=(
                args.physical_gate_materialization_receipt
            ),
            physical_gate_materialization_receipt_sha256=(
                args.physical_gate_materialization_receipt_sha256
            ),
            profile_pins=args.profile_pins,
            profile_pins_sha256=args.profile_pins_sha256,
            launch_trust_root=args.launch_trust_root,
            launch_trust_root_sha256=args.launch_trust_root_sha256,
            preflight_only=True,
            render_dir=args.render_dir,
            render_fps=args.render_fps,
        )
        inherited, teacher_evidence, manifest, profile, venue = fitted._preflight(
            teacher_args
        )
        blockers.extend(f"teacher_preflight:{item}" for item in inherited)
        evidence["teacher_preflight"] = teacher_evidence
    except Exception as exc:
        blockers.append(f"teacher_preflight:{exc}")
    try:
        receipt_raw = _read_pinned(
            args.teacher_gate_receipt,
            args.teacher_gate_receipt_sha256,
            "teacher fitted-ball gate receipt",
        )
        receipt = json.loads(receipt_raw)
        if manifest is None:
            raise PolicyGateError("manifest unavailable for teacher receipt validation")
        evidence["teacher_gate"] = validate_teacher_receipt(receipt, manifest)
        teacher_controls = dict(
            evidence["teacher_gate"]["controls_by_action"]
        )
    except Exception as exc:
        blockers.append(f"teacher_gate_receipt:{exc}")
    try:
        if manifest is None:
            raise PolicyGateError(
                "manifest unavailable for ONNX motion identity validation"
            )
        _read_pinned(args.checkpoint, checkpoint_sha, "checkpoint")
        onnx_bytes = _read_pinned(args.onnx, onnx_sha, "ONNX")
        policy = ActionBallPolicy(
            onnx_path=args.onnx,
            onnx_bytes=onnx_bytes,
            checkpoint_sha256=checkpoint_sha,
            normalizer_path=args.obs_normalizer,
            normalizer_sha256=normalizer_sha,
            trusted_action_set=manifest.action_set_contract,
        )
        evidence["actor"] = policy.contract
        evidence["normalizer"] = policy.normalizer_contract
        manifest_motion = tuple(action.motion_sha256 for action in manifest.base.actions)
        if policy.motion_sha != manifest_motion:
            raise PolicyGateError(
                "ONNX motion order/bytes differ from the trusted manifest"
            )
        # PhysX's load-dependent joint friction has no exact MuJoCo primitive.
        # A non-zero coefficient is an explicit formal blocker, never a hidden
        # plant approximation.
        if np.any(policy.joint_friction_coefficients != 0.0):
            blockers.append(
                "plant_parity:nonzero_physx_joint_friction_has_no_exact_mujoco_equivalent"
            )
    except Exception as exc:
        blockers.append(f"actor:{exc}")
    base = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "status": "BLOCKED" if blockers else "PREFLIGHT_PASS",
        "verdict": "BLOCKED" if blockers else "NOT_RUN",
        "formal_gate_executed": False,
        "expected_actions": (
            int(manifest.action_set_contract["expected_n"])
            if manifest is not None
            else None
        ),
        "action_set_contract": (
            dict(manifest.action_set_contract)
            if manifest is not None
            else None
        ),
        "expected_positive_case_roles": list(POSITIVE_CASE_ROLES),
        "selector_executed": False,
        "solver_executed_by_gate": False,
        "virtual_scorer_executed": False,
        "five_solid_safety_scene": (
            None
            if safety_geometry_contract is None
            else {
                "five_solid_geometry_sha256": (
                    safety_geometry_contract["sha256"]
                ),
                "geometry_payload": safety_geometry_contract["payload"],
                "obstacle_order": list(
                    fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
                ),
                "under_table_keepout_role": "robot_only",
                "ball_keepout_native_pair_enabled": False,
                "ball_keepout_analytic_surface_enabled": False,
                "contact_force_threshold_n": (
                    TABLE_CONTACT_FORCE_THRESHOLD_N
                ),
                "continuous_sweep_method": FIVE_SOLID_SWEEP_METHOD,
                "compiled_by_dt": {},
            }
        ),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
        },
        "onnx": {"path": str(args.onnx.resolve()), "sha256": onnx_sha},
        "preflight": {"blockers": blockers, "evidence": evidence},
        "actions": [],
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }
    if args.preflight_only or blockers:
        _write_no_clobber(args.out, base)
        return 3 if blockers else 0
    assert manifest is not None and profile is not None and venue is not None
    assert policy is not None
    assert policy_obstacle_rows is not None
    assert safety_geometry_contract is not None
    args.render_dir.mkdir(parents=True, exist_ok=False)
    actions_out = []
    try:
        from canonical_mujoco_identity import verify_exact_mujoco_identity

        verified_model = verify_exact_mujoco_identity(
            mjcf_path=fitted.CANONICAL_MJCF,
            expected_manifest_path=fitted.CANONICAL_IDENTITY_MANIFEST,
            trusted_expected_manifest_sha256=(
                fitted.CANONICAL_IDENTITY_MANIFEST_SHA256
            ),
        )
        import mujoco
        import onnxruntime

        base["runtime"] = {
            "python": sys.version,
            "mujoco_version": str(mujoco.__version__),
            "onnxruntime_version": str(onnxruntime.__version__),
            "base_mujoco_portable_identity_sha256": (
                verified_model.portable_identity_sha256
            ),
            "base_mujoco_verification_receipt_sha256": (
                verified_model.verification_receipt_sha256
            ),
        }
        canonical = fitted.CANONICAL_MJCF.read_bytes()
        obstacle_rows = policy_obstacle_rows
        models: Dict[float, legacy_mj.MujocoRobot] = {}
        compiled_safety_by_dt = {}
        for dt in FORMAL_DT_S:
            four_solid_xml, _scene = fitted.assemble_fitted_scene_xml(
                canonical, obstacle_rows, venue, dt
            )
            xml = fitted.table_scene.append_action_ball_policy_keepout_xml(
                four_solid_xml,
                obstacle_rows,
                collidable=True,
            )
            xml_sha = fitted.native_diag.sha256_bytes(xml)
            models[dt] = _model_robot(
                policy=policy,
                xml=xml,
                dt=dt,
                geometry_contract=safety_geometry_contract,
            )
            compiled_safety_by_dt[format(dt, ".4f")] = (
                _validate_compiled_five_solid_scene(
                    models[dt],
                    obstacle_rows,
                    safety_geometry_contract,
                    assembled_xml_sha256=xml_sha,
                )
            )
        base["five_solid_safety_scene"][
            "compiled_by_dt"
        ] = compiled_safety_by_dt
        video_slots = frozenset(
            fitted.formal_video_action_slots(len(manifest.base.actions))
        )
        for action_slot, action in enumerate(manifest.base.actions):
            video_required = action_slot in video_slots
            clip = fitted.load_motion_from_pinned_bytes(action)
            if clip.n_frames != policy.segment_lengths[action_slot]:
                raise PolicyGateError(
                    f"{action.action_id}: ONNX segment length "
                    f"{policy.segment_lengths[action_slot]} != pinned motion "
                    f"frames {clip.n_frames}"
                )
            face_mesh = fitted.load_binary_stl_face(action.mount_normal_sign)
            case_by_role = {
                case.case_role: case
                for case in manifest.task_bindings[action.action_id].cases
            }
            case_rows = []
            video_receipt = None
            for role in POSITIVE_CASE_ROLES:
                case = case_by_role.get(role)
                if case is None:
                    raise PolicyGateError(f"{action.action_id}: missing positive case {role}")
                dt_rows = {}
                for dt in FORMAL_DT_S:
                    result, frames = _run_case_dt(
                        policy=policy,
                        robot=models[dt],
                        action=action,
                        action_slot=action_slot,
                        clip=clip,
                        case=case,
                        venue=venue,
                        profile=profile,
                        face_mesh=face_mesh,
                        obstacle_rows=obstacle_rows,
                        capture_frames=(
                            video_required
                            and
                            role == "center_positive_seed_0" and dt == FORMAL_DT_S[0]
                        ),
                        render_fps=args.render_fps,
                    )
                    dt_rows[format(dt, ".4f")] = result
                    if frames:
                        contact = result.get("contact") or {}
                        safety = result.get("safety") or {}
                        overlay_lines = (
                            f"policy action={action.action_id}",
                            (
                                "t_hit={:.4f}s t_cycle={:.4f}s".format(
                                    float(case.scaled_t_hit_s),
                                    float(case.scaled_t_cycle_s),
                                )
                            ),
                            (
                                "contact_u_n={:.4f} m/s".format(
                                    float(
                                        contact.get(
                                            "relative_normal_speed_mps",
                                            float("nan"),
                                        )
                                    )
                                )
                            ),
                            (
                                "landing={} table_steps={} net_collision={}".format(
                                    result.get("first_landing_xy_m"),
                                    safety.get("table_contact_steps"),
                                    bool(result.get("ball_net_collision")),
                                )
                            ),
                        )
                        path = (
                            args.render_dir
                            / f"{action.action_id}_policy_fitted_ball.mp4"
                        )
                        video_receipt = fitted.native_diag._render_video(
                            fitted.overlay_video_frames(
                                frames, overlay_lines
                            ),
                            path,
                            args.render_fps,
                        )
                        video_receipt["overlay"] = {
                            "burned_in": True,
                            "lines": list(overlay_lines),
                        }
                case_rows.append(
                    {
                        "case_id": case.case_id,
                        "case_role": role,
                        "case_binding_sha256": case.case_binding_sha256,
                        "dt_results": dt_rows,
                        "verdict": (
                            "PASS"
                            if all(row["verdict"] == "PASS" for row in dt_rows.values())
                            else "FAIL"
                        ),
                    }
                )
            reasons = [
                f"{row['case_role']}:policy_physical_return_failed"
                for row in case_rows
                if row["verdict"] != "PASS"
            ]
            if not video_required:
                video_receipt = {
                    "status": "NOT_SAMPLED_NUMERIC_GATE_COMPLETE",
                    "path": None,
                }
            if (
                video_required
                and (
                    not isinstance(video_receipt, dict)
                    or video_receipt.get("status") != "WRITTEN"
                )
            ):
                reasons.append("required_policy_video_not_written")
            actions_out.append(
                {
                    "action_id": action.action_id,
                    "action_uid": action.action_uid,
                    "action_slot": action_slot,
                    "motion_sha256": action.motion_sha256,
                    "task_binding": {
                        "cases_sha256": manifest.task_bindings[
                            action.action_id
                        ].cases_sha256,
                        "solver_execution_receipt_sha256": manifest.task_bindings[
                            action.action_id
                        ].solver_execution_receipt_sha256,
                    },
                    "positive_cases": case_rows,
                    "teacher_prerequisite_controls": {
                        "positive": [
                            row
                            for row in teacher_controls[action.action_id]
                            if row["case_role"] in POSITIVE_CASE_ROLES
                        ],
                        "negative": [
                            row
                            for row in teacher_controls[action.action_id]
                            if row["case_role"] not in POSITIVE_CASE_ROLES
                        ],
                    },
                    "video": video_receipt,
                    "verdict": "PASS" if not reasons else "FAIL",
                    "failure_reasons": reasons,
                }
            )
        overall = all(row["verdict"] == "PASS" for row in actions_out)
        verified_model.assert_model_unchanged()
        base.update(
            {
                "status": "PASS" if overall else "FAIL",
                "verdict": "PASS" if overall else "FAIL",
                "formal_gate_executed": True,
                "action_order": list(manifest.base.action_order),
                "video_sampled_action_slots": sorted(video_slots),
                "actions": actions_out,
            }
        )
        # Re-hash every external policy input after execution.
        post_hashes = {
            "checkpoint": _sha256_file(args.checkpoint),
            "onnx": _sha256_file(args.onnx),
            "normalizer": _sha256_file(args.obs_normalizer),
            "training_manifest": _sha256_file(args.training_manifest),
            "physical_gate_manifest": _sha256_file(
                args.physical_gate_manifest
            ),
            "physical_gate_materialization_receipt": _sha256_file(
                args.physical_gate_materialization_receipt
            ),
            "profile_pins": _sha256_file(args.profile_pins),
            "teacher_gate_receipt": _sha256_file(args.teacher_gate_receipt),
        }
        expected_hashes = {
            "checkpoint": checkpoint_sha,
            "onnx": onnx_sha,
            "normalizer": normalizer_sha,
            "training_manifest": args.training_manifest_sha256,
            "physical_gate_manifest": args.physical_gate_manifest_sha256,
            "physical_gate_materialization_receipt": (
                args.physical_gate_materialization_receipt_sha256
            ),
            "profile_pins": args.profile_pins_sha256,
            "teacher_gate_receipt": args.teacher_gate_receipt_sha256,
        }
        if post_hashes != expected_hashes:
            raise PolicyGateError("pinned input bytes changed during policy execution")
        base["checkout_post_runtime"] = fitted.validate_clean_checkout(
            args.code_commit
        )
        base["post_runtime_input_sha256"] = post_hashes
        _write_no_clobber(args.out, base)
        return 0 if overall else 3
    except Exception as exc:
        base.update(
            {
                "status": "BLOCKED",
                "verdict": "BLOCKED",
                "formal_gate_executed": False,
                "actions": actions_out,
                "runtime_blocker": str(exc),
            }
        )
        _write_no_clobber(args.out, base)
        print(f"[policy-fitted-gate][FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
