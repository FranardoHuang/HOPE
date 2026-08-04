#!/usr/bin/env python3
"""Deterministic lexicographic search for a measured-conditioned safe ready.

This module is deliberately plant-agnostic.  The caller owns the exact-model
contact LP and returns named *physical slacks* for every sampled state.  Exact
measured frame 0 is preferred and returned unchanged when it already clears
every gate.  Only when that direct state is unsafe does the fallback search
maximize the worst normalized safety slack and then minimize measured-frame-0
root/joint/racket error while constraining every safety slack to remain at the
stage-1 optimum (up to one explicit numerical lock tolerance).

Consequently no weighted tracking objective can buy its way through a safety
gate.  This is an unauthorized host diagnostic, not an Isaac hold or hardware
certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import canonical_grounded_ready as grounded


REQUIRED_SAFETY_SLACK_NAMES = (
    "left_sole_floor_slack_m",
    "right_sole_floor_slack_m",
    "left_contact_load_slack_n",
    "right_contact_load_slack_n",
    "support_margin_slack_m",
    "joint_position_slack_rad",
    "qdes_slack_rad",
    "torque_slack_nm",
    "table_clearance_slack_m",
    "root_height_slack_m",
    "root_tilt_slack_rad",
    "collision_slack_m",
    "ground_lp_residual_slack",
)

DEFAULT_SLACK_SCALES = MappingProxyType(
    {
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
)

# Threshold-first direct-frame0 admission is deliberately stronger than the
# tiny numerical interior gate used by the fallback optimizer.  Each entry is
# a physical reserve beyond the evaluator's already-conservative base limit.
DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS = MappingProxyType(
    {
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
)


class WholeBodySafeReadyError(RuntimeError):
    """The lexical safe-ready search cannot produce an authoritative result."""

    def __init__(self, message: str, *, code: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.report = None if report is None else dict(report)


@dataclass(frozen=True)
class SafetyEvaluation:
    """One exact evaluator result at a sampled state."""

    slacks: Mapping[str, float]
    racket_position_w: np.ndarray
    racket_rotation_w: np.ndarray
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        names = tuple(sorted(str(name) for name in self.slacks))
        expected = tuple(sorted(REQUIRED_SAFETY_SLACK_NAMES))
        if names != expected:
            raise WholeBodySafeReadyError(
                f"safety evaluator fields differ: {names} != {expected}",
                code="SAFETY_SLACK_SCHEMA_MISMATCH",
            )
        slacks = {str(name): float(self.slacks[name]) for name in REQUIRED_SAFETY_SLACK_NAMES}
        if not all(math.isfinite(value) for value in slacks.values()):
            raise WholeBodySafeReadyError(
                "safety evaluator returned NaN/Inf slack",
                code="NONFINITE_SAFETY_SLACK",
            )
        position = np.array(self.racket_position_w, np.float64, copy=True)
        rotation = np.array(self.racket_rotation_w, np.float64, copy=True)
        if position.shape != (3,) or rotation.shape != (3, 3) or not (
            np.isfinite(position).all() and np.isfinite(rotation).all()
        ):
            raise WholeBodySafeReadyError(
                "safety evaluator returned malformed racket pose",
                code="INVALID_RACKET_POSE",
            )
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1.0e-7 or np.linalg.det(rotation) < 1.0 - 1.0e-7:
            raise WholeBodySafeReadyError(
                "safety evaluator racket rotation is not proper orthonormal",
                code="INVALID_RACKET_POSE",
            )
        position.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "slacks", MappingProxyType(slacks))
        object.__setattr__(self, "racket_position_w", position)
        object.__setattr__(self, "racket_rotation_w", rotation)
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class WholeBodySearchConfig:
    """Explicit bounds and lexical numerical tolerances."""

    movable_joint_names: tuple[str, ...] = grounded.RUNTIME_JOINT_NAMES
    root_z_delta_bound_m: float = 0.15
    root_roll_delta_bound_rad: float = 0.35
    root_pitch_delta_bound_rad: float = 0.35
    joint_delta_bound_rad: float = 0.60
    positive_gate_normalized_slack: float = 1.0e-6
    stage1_lock_tolerance_normalized: float = 5.0e-5
    stage1_max_iterations: int = 180
    stage2_max_iterations: int = 240
    optimizer_ftol: float = 1.0e-9
    root_position_weight: float = 25.0
    root_rotation_weight: float = 4.0
    joint_weight: float = 1.0
    racket_position_weight: float = 25.0
    racket_rotation_weight: float = 4.0

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.movable_joint_names)
        if not names or len(names) != len(set(names)) or not set(names).issubset(
            grounded.RUNTIME_JOINT_NAMES
        ):
            raise ValueError("movable_joint_names must be a non-empty unique runtime subset")
        required = set(grounded.LEG_JOINT_NAMES) | {
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        }
        if not required.issubset(names):
            raise ValueError("whole-body search must release waist3 and leg12")
        object.__setattr__(self, "movable_joint_names", names)
        for name in (
            "root_z_delta_bound_m",
            "root_roll_delta_bound_rad",
            "root_pitch_delta_bound_rad",
            "joint_delta_bound_rad",
            "positive_gate_normalized_slack",
            "stage1_lock_tolerance_normalized",
            "optimizer_ftol",
            "root_position_weight",
            "root_rotation_weight",
            "joint_weight",
            "racket_position_weight",
            "racket_rotation_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.stage1_max_iterations < 1 or self.stage2_max_iterations < 1:
            raise ValueError("optimizer iteration limits must be positive")


@dataclass(frozen=True)
class WholeBodySafeReadyResult:
    state: grounded.ReadyState
    safety_slacks: Mapping[str, float]
    normalized_safety_slacks: Mapping[str, float]
    worst_normalized_safety_slack: float
    stage1_locked_worst_normalized_slack: float
    changed_joint_mask: tuple[bool, ...]
    joint_delta_rad: tuple[float, ...]
    root_position_delta_m: tuple[float, ...]
    root_rotation_delta_rad: tuple[float, ...]
    racket_position_delta_m: tuple[float, ...]
    racket_rotation_delta_rad: tuple[float, ...]
    evaluator_evidence: Mapping[str, Any]
    optimizer_report: Mapping[str, Any]
    training_authorized: bool = False
    deployment_authorized: bool = False
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        if self.training_authorized or self.deployment_authorized or self.hardware_authorized:
            raise WholeBodySafeReadyError(
                "whole-body host search cannot authorize runtime use",
                code="AUTHORIZATION_FORBIDDEN",
            )


def solve_measured_conditioned_whole_body_safe_ready(
    measured_state: grounded.ReadyState,
    *,
    evaluator: Callable[[grounded.ReadyState], SafetyEvaluation],
    racket_reference_position_w: np.ndarray,
    racket_reference_rotation_w: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
    initial_states: Sequence[grounded.ReadyState] = (),
    slack_scales: Mapping[str, float] = DEFAULT_SLACK_SCALES,
    config: WholeBodySearchConfig | None = None,
) -> WholeBodySafeReadyResult:
    """Run the two-stage lexical search and freshly re-evaluate its winner."""

    cfg = WholeBodySearchConfig() if config is None else config
    lower = np.asarray(position_lower, np.float64)
    upper = np.asarray(position_upper, np.float64)
    if lower.shape != (31,) or upper.shape != (31,) or not (
        np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower < upper)
    ):
        raise WholeBodySafeReadyError("invalid exact joint limits", code="INVALID_JOINT_LIMITS")
    scales = {name: float(slack_scales[name]) for name in REQUIRED_SAFETY_SLACK_NAMES}
    if set(scales) != set(REQUIRED_SAFETY_SLACK_NAMES) or not all(
        math.isfinite(value) and value > 0.0 for value in scales.values()
    ):
        raise WholeBodySafeReadyError("invalid safety slack scales", code="INVALID_SLACK_SCALES")

    movable = grounded._joint_indices(grounded.RUNTIME_JOINT_NAMES, cfg.movable_joint_names)
    measured_rotation = _quat_to_rotation(measured_state.root_quat_wxyz)
    measured_roll, measured_pitch, measured_yaw = _rotation_to_rpy(measured_rotation)
    variable_lower = np.concatenate(
        (
            np.asarray(
                [
                    measured_state.root_pos_w[2] - cfg.root_z_delta_bound_m,
                    measured_roll - cfg.root_roll_delta_bound_rad,
                    measured_pitch - cfg.root_pitch_delta_bound_rad,
                ],
                np.float64,
            ),
            np.maximum(lower[movable], measured_state.joint_pos[movable] - cfg.joint_delta_bound_rad),
        )
    )
    variable_upper = np.concatenate(
        (
            np.asarray(
                [
                    measured_state.root_pos_w[2] + cfg.root_z_delta_bound_m,
                    measured_roll + cfg.root_roll_delta_bound_rad,
                    measured_pitch + cfg.root_pitch_delta_bound_rad,
                ],
                np.float64,
            ),
            np.minimum(upper[movable], measured_state.joint_pos[movable] + cfg.joint_delta_bound_rad),
        )
    )
    if np.any(variable_lower >= variable_upper):
        raise WholeBodySafeReadyError("whole-body variable box is empty", code="EMPTY_SEARCH_BOX")

    def encode(state: grounded.ReadyState) -> np.ndarray:
        roll, pitch, _yaw = _rotation_to_rpy(_quat_to_rotation(state.root_quat_wxyz))
        raw = np.concatenate(
            (
                np.asarray([state.root_pos_w[2], roll, pitch], np.float64),
                np.asarray(state.joint_pos[movable], np.float64),
            )
        )
        return np.clip(raw, variable_lower, variable_upper)

    def decode(value: np.ndarray) -> grounded.ReadyState:
        row = np.clip(np.asarray(value, np.float64), variable_lower, variable_upper)
        q = np.asarray(measured_state.joint_pos, np.float64).copy()
        q[movable] = row[3:]
        root = np.asarray(measured_state.root_pos_w, np.float64).copy()
        root[2] = row[0]
        quat = _rotation_to_quat(_rpy_to_rotation(row[1], row[2], measured_yaw))
        return grounded.ReadyState(q, root, quat)

    cache: dict[bytes, tuple[grounded.ReadyState, SafetyEvaluation, np.ndarray]] = {}
    evaluation_count = 0

    def evaluate(value: np.ndarray) -> tuple[grounded.ReadyState, SafetyEvaluation, np.ndarray]:
        nonlocal evaluation_count
        clipped = np.ascontiguousarray(np.clip(value, variable_lower, variable_upper), np.float64)
        key = clipped.tobytes()
        if key not in cache:
            state = decode(clipped)
            try:
                row = evaluator(state)
            except WholeBodySafeReadyError:
                raise
            except Exception as exc:
                raise WholeBodySafeReadyError(
                    f"exact safety evaluator failed: {type(exc).__name__}: {exc}",
                    code="SAFETY_EVALUATOR_FAILED",
                ) from exc
            normalized = np.asarray(
                [row.slacks[name] / scales[name] for name in REQUIRED_SAFETY_SLACK_NAMES],
                np.float64,
            )
            cache[key] = (state, row, normalized)
            evaluation_count += 1
        return cache[key]

    def worst(value: np.ndarray) -> float:
        return float(np.min(evaluate(value)[2]))

    racket_reference_position = np.asarray(
        racket_reference_position_w, np.float64
    )
    racket_reference_rotation = np.asarray(
        racket_reference_rotation_w, np.float64
    )
    if (
        racket_reference_position.shape != (3,)
        or racket_reference_rotation.shape != (3, 3)
        or not np.isfinite(racket_reference_position).all()
        or not np.isfinite(racket_reference_rotation).all()
        or np.max(
            np.abs(
                racket_reference_rotation.T
                @ racket_reference_rotation
                - np.eye(3)
            )
        )
        > 1.0e-7
        or np.linalg.det(racket_reference_rotation) < 1.0 - 1.0e-7
    ):
        raise WholeBodySafeReadyError(
            "caller-supplied racket reference is malformed",
            code="INVALID_RACKET_REFERENCE",
        )
    racket_reference_authority = "caller_supplied_independent_measurement"

    starts = [encode(measured_state)] + [encode(state) for state in initial_states]
    unique_starts: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        key = np.ascontiguousarray(start, np.float64).tobytes()
        if key not in seen:
            unique_starts.append(start)
            seen.add(key)

    # Exact measured frame 0 is the preferred ready.  If it already clears
    # every physical gate, do not optimize it away merely to buy more margin;
    # that would create an unnecessary transition before teacher reveal.
    measured_direct_eval = evaluator(measured_state)
    measured_direct_normalized = {
        name: float(measured_direct_eval.slacks[name] / scales[name])
        for name in REQUIRED_SAFETY_SLACK_NAMES
    }
    measured_direct_worst = min(measured_direct_normalized.values())
    measured_direct_robust = all(
        measured_direct_eval.slacks[name]
        >= DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[name]
        for name in REQUIRED_SAFETY_SLACK_NAMES
    )
    if (
        measured_direct_worst > cfg.positive_gate_normalized_slack
        and measured_direct_robust
    ):
        final_eval = evaluator(measured_state)  # mandatory fresh re-audit
        final_normalized = {
            name: float(final_eval.slacks[name] / scales[name])
            for name in REQUIRED_SAFETY_SLACK_NAMES
        }
        final_worst = min(final_normalized.values())
        if (
            final_worst <= cfg.positive_gate_normalized_slack
            or any(
                final_eval.slacks[name]
                < DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[name]
                for name in REQUIRED_SAFETY_SLACK_NAMES
            )
        ):
            raise WholeBodySafeReadyError(
                "fresh exact measured frame0 re-audit lost its safety gate",
                code="FINAL_REAUDIT_FAILED",
            )
        racket_position_delta = (
            final_eval.racket_position_w - racket_reference_position
        )
        racket_rotation_delta = grounded._so3_log(
            racket_reference_rotation.T @ final_eval.racket_rotation_w
        )
        direct_tracking_objective = float(
            cfg.racket_position_weight
            * (racket_position_delta @ racket_position_delta)
            + cfg.racket_rotation_weight
            * (racket_rotation_delta @ racket_rotation_delta)
        )
        return WholeBodySafeReadyResult(
            state=measured_state,
            safety_slacks=MappingProxyType(dict(final_eval.slacks)),
            normalized_safety_slacks=MappingProxyType(final_normalized),
            worst_normalized_safety_slack=float(final_worst),
            stage1_locked_worst_normalized_slack=float(
                measured_direct_worst
            ),
            changed_joint_mask=(False,) * 31,
            joint_delta_rad=(0.0,) * 31,
            root_position_delta_m=(0.0, 0.0, 0.0),
            root_rotation_delta_rad=(0.0, 0.0, 0.0),
            racket_position_delta_m=tuple(
                float(value) for value in racket_position_delta
            ),
            racket_rotation_delta_rad=tuple(
                float(value) for value in racket_rotation_delta
            ),
            evaluator_evidence=MappingProxyType(dict(final_eval.evidence)),
            optimizer_report=MappingProxyType(
                {
                    "algorithm": "exact_measured_frame0_safety_short_circuit",
                    "global_optimum_claimed": False,
                    "stage1_objective": (
                        "prefer_exact_measured_frame0_when_all_safety_gates_pass"
                    ),
                    "stage2_objective": "not_run_exact_frame0_already_safe",
                    "safety_weighted_against_tracking": False,
                    "exact_measured_frame0_selected": True,
                    "direct_frame0_robust_minimum_slacks": dict(
                        DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS
                    ),
                    "stage1_runs": [],
                    "stage1_worst_normalized_slack": float(
                        measured_direct_worst
                    ),
                    "stage1_lock_tolerance_normalized": (
                        cfg.stage1_lock_tolerance_normalized
                    ),
                    "stage1_locked_worst_normalized_slack": float(
                        measured_direct_worst
                    ),
                    "stage2_success": True,
                    "stage2_status": 0,
                    "stage2_message": "not run; exact measured frame0 is safe",
                    "stage2_iterations": 0,
                    "stage2_accepted_steps": 0,
                    "stage2_objective_value": direct_tracking_objective,
                    "evaluation_count": int(evaluation_count + 3),
                    "movable_joint_names": list(cfg.movable_joint_names),
                    "root_degrees_of_freedom": ["z", "roll", "pitch"],
                    "slack_scales": scales,
                    "racket_reference_authority": (
                        racket_reference_authority
                    ),
                }
            ),
        )

    def coordinate_search(
        start: np.ndarray,
        *,
        objective: Callable[[np.ndarray], float],
        feasible: Callable[[np.ndarray], bool],
        maximum_sweeps: int,
    ) -> dict[str, Any]:
        """Bounded deterministic pattern search with no optional dependency.

        The routine is intentionally local and its report says so.  Exact LP
        evaluation dominates runtime, so one reproducible coordinate stencil
        is preferable to an optimizer whose library/version changes sampling.
        """

        value = np.clip(np.asarray(start, np.float64), variable_lower, variable_upper)
        score = float(objective(value))
        steps = 0.25 * (variable_upper - variable_lower)
        sweeps = 0
        accepted = 0
        while sweeps < maximum_sweeps and float(np.max(steps)) > cfg.optimizer_ftol:
            sweeps += 1
            improved = False
            for index in range(len(value)):
                choices: list[tuple[float, np.ndarray]] = []
                for sign in (-1.0, 1.0):
                    trial = value.copy()
                    trial[index] = float(
                        np.clip(
                            value[index] + sign * steps[index],
                            variable_lower[index],
                            variable_upper[index],
                        )
                    )
                    if trial[index] == value[index] or not feasible(trial):
                        continue
                    choices.append((float(objective(trial)), trial))
                if choices:
                    trial_score, trial = min(
                        choices,
                        key=lambda row: (row[0], row[1].tobytes()),
                    )
                    if trial_score < score - cfg.optimizer_ftol:
                        value = trial
                        score = trial_score
                        accepted += 1
                        improved = True
            if not improved:
                steps *= 0.5
        return {
            "x": value,
            "objective": score,
            "success": bool(feasible(value)),
            "status": 0 if feasible(value) else 1,
            "message": (
                "deterministic coordinate stencil converged"
                if feasible(value)
                else "deterministic coordinate stencil has no feasible point"
            ),
            "iterations": sweeps,
            "accepted_steps": accepted,
        }

    stage1_rows: list[dict[str, Any]] = []
    stage1_candidates: list[np.ndarray] = list(unique_starts)
    for start_index, start in enumerate(unique_starts):
        result = coordinate_search(
            start,
            objective=lambda value: -worst(value),
            feasible=lambda _value: True,
            maximum_sweeps=cfg.stage1_max_iterations,
        )
        stage1_candidates.append(np.asarray(result["x"], np.float64))
        stage1_rows.append(
            {
                "start_index": start_index,
                "success": bool(result["success"]),
                "status": int(result["status"]),
                "message": str(result["message"]),
                "iterations": int(result["iterations"]),
                "accepted_steps": int(result["accepted_steps"]),
                "worst_normalized_slack": worst(result["x"]),
            }
        )
    stage1_value = max(stage1_candidates, key=worst)
    stage1_worst = worst(stage1_value)
    if stage1_worst <= cfg.positive_gate_normalized_slack:
        state, row, normalized = evaluate(stage1_value)
        raise WholeBodySafeReadyError(
            "whole-body stage 1 found no strictly positive all-gate safety state",
            code="NO_POSITIVE_SAFETY_INTERIOR",
            report={
                "best_state_sha256": grounded.state_digest(state),
                "best_slacks": dict(row.slacks),
                "best_normalized_slacks": dict(zip(REQUIRED_SAFETY_SLACK_NAMES, normalized.tolist())),
                "worst_normalized_slack": stage1_worst,
                "stage1_runs": stage1_rows,
                "evaluation_count": evaluation_count,
            },
        )
    # The numerical lock may relax the stage-1 optimum, but it must never
    # relax the caller's original admission gate.  Otherwise a thin stage-1
    # interior (smaller than the lock tolerance) lets the tracking objective
    # purchase a final state that the original gate would reject.
    locked = max(
        cfg.positive_gate_normalized_slack,
        stage1_worst - cfg.stage1_lock_tolerance_normalized,
    )

    def secondary(value: np.ndarray) -> float:
        state, row, _normalized = evaluate(value)
        joint_delta = state.joint_pos - measured_state.joint_pos
        root_delta = state.root_pos_w - measured_state.root_pos_w
        root_rotation_delta = grounded._so3_log(
            measured_rotation.T @ _quat_to_rotation(state.root_quat_wxyz)
        )
        racket_position_delta = row.racket_position_w - racket_reference_position
        racket_rotation_delta = grounded._so3_log(
            racket_reference_rotation.T @ row.racket_rotation_w
        )
        return float(
            cfg.root_position_weight * (root_delta @ root_delta)
            + cfg.root_rotation_weight * (root_rotation_delta @ root_rotation_delta)
            + cfg.joint_weight * (joint_delta @ joint_delta)
            + cfg.racket_position_weight * (racket_position_delta @ racket_position_delta)
            + cfg.racket_rotation_weight * (racket_rotation_delta @ racket_rotation_delta)
        )

    def locked_feasible(value: np.ndarray) -> bool:
        normalized = evaluate(value)[2]
        return bool(
            np.all(normalized >= locked)
            and np.all(normalized > cfg.positive_gate_normalized_slack)
        )

    stage2 = coordinate_search(
        stage1_value,
        objective=secondary,
        feasible=locked_feasible,
        maximum_sweeps=cfg.stage2_max_iterations,
    )
    stage2_candidates = [stage1_value, np.asarray(stage2["x"], np.float64)]
    feasible_stage2 = [
        value for value in stage2_candidates if locked_feasible(value)
    ]
    if not feasible_stage2:
        raise WholeBodySafeReadyError(
            "stage 2 lost the locked safety interior",
            code="STAGE2_SAFETY_REGRESSION",
        )
    winner = min(feasible_stage2, key=secondary)
    final_state = decode(winner)
    final_eval = evaluator(final_state)  # mandatory fresh exact re-audit
    final_normalized = {
        name: float(final_eval.slacks[name] / scales[name])
        for name in REQUIRED_SAFETY_SLACK_NAMES
    }
    final_worst = min(final_normalized.values())
    if (
        final_worst <= cfg.positive_gate_normalized_slack
        or final_worst < locked
    ):
        raise WholeBodySafeReadyError(
            "fresh final evaluator did not preserve the original and locked safety gates",
            code="FINAL_REAUDIT_FAILED",
        )

    q_delta = final_state.joint_pos - measured_state.joint_pos
    root_delta = final_state.root_pos_w - measured_state.root_pos_w
    root_rotation_delta = grounded._so3_log(
        measured_rotation.T @ _quat_to_rotation(final_state.root_quat_wxyz)
    )
    racket_position_delta = final_eval.racket_position_w - racket_reference_position
    racket_rotation_delta = grounded._so3_log(
        racket_reference_rotation.T @ final_eval.racket_rotation_w
    )
    changed = tuple(bool(value != 0.0) for value in q_delta)
    return WholeBodySafeReadyResult(
        state=final_state,
        safety_slacks=MappingProxyType(dict(final_eval.slacks)),
        normalized_safety_slacks=MappingProxyType(final_normalized),
        worst_normalized_safety_slack=float(final_worst),
        stage1_locked_worst_normalized_slack=float(locked),
        changed_joint_mask=changed,
        joint_delta_rad=tuple(float(value) for value in q_delta),
        root_position_delta_m=tuple(float(value) for value in root_delta),
        root_rotation_delta_rad=tuple(float(value) for value in root_rotation_delta),
        racket_position_delta_m=tuple(float(value) for value in racket_position_delta),
        racket_rotation_delta_rad=tuple(float(value) for value in racket_rotation_delta),
        evaluator_evidence=MappingProxyType(dict(final_eval.evidence)),
        optimizer_report=MappingProxyType(
            {
                "algorithm": "two_stage_deterministic_coordinate_local_lexicographic",
                "global_optimum_claimed": False,
                "stage1_objective": "maximize_min_normalized_physical_safety_slack",
                "stage2_objective": "minimize_weighted_root_31q_racket_error",
                "racket_reference_authority": racket_reference_authority,
                "safety_weighted_against_tracking": False,
                "stage1_runs": stage1_rows,
                "stage1_worst_normalized_slack": float(stage1_worst),
                "stage1_lock_tolerance_normalized": cfg.stage1_lock_tolerance_normalized,
                "stage1_locked_worst_normalized_slack": float(locked),
                "stage2_success": bool(stage2["success"]),
                "stage2_status": int(stage2["status"]),
                "stage2_message": str(stage2["message"]),
                "stage2_iterations": int(stage2["iterations"]),
                "stage2_accepted_steps": int(stage2["accepted_steps"]),
                "stage2_objective_value": float(secondary(winner)),
                "evaluation_count": int(evaluation_count),
                "movable_joint_names": list(cfg.movable_joint_names),
                "root_degrees_of_freedom": ["z", "roll", "pitch"],
                "slack_scales": scales,
            }
        ),
    )


def _quat_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        np.float64,
    )


def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quat = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale],
            np.float64,
        )
    else:
        pivot = int(np.argmax(np.diag(matrix)))
        following = (pivot + 1) % 3
        remaining = (pivot + 2) % 3
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[pivot, pivot] - matrix[following, following] - matrix[remaining, remaining]))
        quat = np.empty(4, np.float64)
        quat[pivot + 1] = 0.25 * scale
        quat[0] = (matrix[remaining, following] - matrix[following, remaining]) / scale
        quat[following + 1] = (matrix[following, pivot] + matrix[pivot, following]) / scale
        quat[remaining + 1] = (matrix[remaining, pivot] + matrix[pivot, remaining]) / scale
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        np.float64,
    )


def _rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, np.float64)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1.0e-8:
        raise WholeBodySafeReadyError(
            "root rotation is at the roll/pitch/yaw singularity",
            code="ROOT_RPY_SINGULAR",
        )
    roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return roll, pitch, yaw
