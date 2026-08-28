#!/usr/bin/env python3
"""Deterministic constrained search for a measured-conditioned safe ready.

This module is deliberately plant-agnostic.  The caller owns the exact-model
contact LP and returns named *physical slacks* for every sampled state.  Exact
measured frame 0 is preferred and returned unchanged when it already clears
every gate.  Only when that direct state is unsafe does the fallback search
find the fixed, named robust feasible set and then minimize measured-frame-0
root/joint/racket error inside it.  Safety is a constraint, not an objective:
extra distance from the teacher cannot be justified by unused margin.

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
    fallback_minimum_slacks: Mapping[str, float] | None = None

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
        minimums = self.fallback_minimum_slacks
        if minimums is not None:
            copied = {str(name): float(value) for name, value in minimums.items()}
            if set(copied) != set(REQUIRED_SAFETY_SLACK_NAMES) or not all(
                math.isfinite(value) for value in copied.values()
            ):
                raise ValueError(
                    "fallback_minimum_slacks must cover every named safety slack"
                )
            object.__setattr__(
                self, "fallback_minimum_slacks", MappingProxyType(copied)
            )


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
    """Return the best robust-feasible state found by fixed local starts."""

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
    # A historical/shared ready may be supplied only as a deterministic search
    # start.  Do not clip that known feasible start out of the search box: doing
    # so silently turns "optimizer start only" into an unusable label.  Exact
    # joint limits and the evaluator's named physical constraints remain the
    # authority for every selected state.
    for initial_state in initial_states:
        initial_roll, initial_pitch, _initial_yaw = _rotation_to_rpy(
            _quat_to_rotation(initial_state.root_quat_wxyz)
        )
        initial_values = np.concatenate(
            (
                np.asarray(
                    [initial_state.root_pos_w[2], initial_roll, initial_pitch],
                    np.float64,
                ),
                np.asarray(initial_state.joint_pos[movable], np.float64),
            )
        )
        variable_lower = np.minimum(variable_lower, initial_values)
        variable_upper = np.maximum(variable_upper, initial_values)
    variable_lower[3:] = np.maximum(variable_lower[3:], lower[movable])
    variable_upper[3:] = np.minimum(variable_upper[3:], upper[movable])
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

    # The fallback is allowed to differ from the teacher only because the
    # teacher is physically infeasible.  Therefore it must meet the same named
    # reserve that would admit teacher frame 0; accepting a merely-positive
    # numerical interior here would turn the word "robust" into an unchecked
    # label and leave exploration noise to discover the missing margin during
    # training.
    fallback_minimum_slacks = (
        dict(DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS)
        if cfg.fallback_minimum_slacks is None
        else dict(cfg.fallback_minimum_slacks)
    )
    required_normalized = np.asarray(
        [
            fallback_minimum_slacks[name] / scales[name]
            for name in REQUIRED_SAFETY_SLACK_NAMES
        ],
        np.float64,
    )

    def stage1_feasibility_restoration_key(
        value: np.ndarray,
    ) -> tuple[float, ...]:
        """Return a bounded per-gate key used only to navigate infeasible states.

        An infeasible contact LP reports several deliberately conservative
        sentinels.  In particular, a residual slack near ``-1`` becomes about
        ``-1e6`` after its physical ``1e-6`` normalization.  Maximizing only
        the raw minimum then makes every coordinate trial look identical until
        the discontinuous LP suddenly becomes feasible; pattern search cannot
        take the smooth sole/contact-precondition steps needed to reach it.

        This key is a standard feasibility-restoration merit: first reduce the
        number of uncleared gates, then their bounded dimensionless deficits.
        Clipping is deliberately confined to *navigation*.  Stage 1 still
        selects its winner by the uncapped worst normalized slack below, and
        both the original positive gate and the fresh final evaluator remain
        authoritative.
        """

        normalized = evaluate(value)[2]
        unsafe_count = float(np.count_nonzero(normalized < required_normalized))
        deficits = np.maximum(required_normalized - normalized, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            bounded_deficits = 1.0 - 1.0 / (1.0 + deficits)
        bounded_deficits = np.nan_to_num(
            bounded_deficits,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        clipped_margins = np.clip(
            normalized - required_normalized, -1.0, 1.0
        )
        return (
            unsafe_count,
            float(bounded_deficits @ bounded_deficits),
            float(np.sum(bounded_deficits)),
            -float(np.sum(clipped_margins)),
            -float(np.min(clipped_margins)),
        )

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
        ranking: Callable[[np.ndarray], tuple[float, ...]] | None = None,
    ) -> dict[str, Any]:
        """Bounded deterministic pattern search with no optional dependency.

        The routine is intentionally local and its report says so.  Exact LP
        evaluation dominates runtime, so one reproducible coordinate stencil
        is preferable to an optimizer whose library/version changes sampling.
        """

        def ranking_key(
            candidate: np.ndarray, candidate_score: float
        ) -> tuple[float, ...]:
            raw = (
                (candidate_score,)
                if ranking is None
                else tuple(float(item) for item in ranking(candidate))
            )
            if not raw or not all(math.isfinite(item) for item in raw):
                raise WholeBodySafeReadyError(
                    "coordinate-search ranking is empty or nonfinite",
                    code="INVALID_SEARCH_RANKING",
                )
            return raw

        def improves(
            candidate: tuple[float, ...], incumbent: tuple[float, ...]
        ) -> bool:
            if len(candidate) != len(incumbent):
                raise WholeBodySafeReadyError(
                    "coordinate-search ranking cardinality changed",
                    code="INVALID_SEARCH_RANKING",
                )
            for candidate_item, incumbent_item in zip(candidate, incumbent):
                if candidate_item < incumbent_item - cfg.optimizer_ftol:
                    return True
                if candidate_item > incumbent_item + cfg.optimizer_ftol:
                    return False
            return False

        value = np.clip(np.asarray(start, np.float64), variable_lower, variable_upper)
        score = float(objective(value))
        score_key = ranking_key(value, score)
        steps = 0.25 * (variable_upper - variable_lower)
        sweeps = 0
        accepted = 0
        while sweeps < maximum_sweeps and float(np.max(steps)) > cfg.optimizer_ftol:
            sweeps += 1
            improved = False
            for index in range(len(value)):
                choices: list[
                    tuple[tuple[float, ...], float, np.ndarray]
                ] = []
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
                    trial_score = float(objective(trial))
                    choices.append(
                        (
                            ranking_key(trial, trial_score),
                            trial_score,
                            trial,
                        )
                    )
                if choices:
                    trial_key, trial_score, trial = choices[0]
                    for candidate_key, candidate_score, candidate in choices[1:]:
                        candidate_better = improves(candidate_key, trial_key)
                        trial_better = improves(trial_key, candidate_key)
                        if candidate_better or (
                            not trial_better
                            and candidate.tobytes() < trial.tobytes()
                        ):
                            trial_key = candidate_key
                            trial_score = candidate_score
                            trial = candidate
                    if improves(trial_key, score_key):
                        value = trial
                        score = trial_score
                        score_key = trial_key
                        accepted += 1
                        improved = True
            if not improved:
                steps *= 0.5
        return {
            "x": value,
            "objective": score,
            "ranking": score_key,
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

    def robust_feasible(value: np.ndarray) -> bool:
        return bool(np.all(evaluate(value)[2] >= required_normalized))

    stage1_rows: list[dict[str, Any]] = []
    stage1_candidates: list[np.ndarray] = list(unique_starts)
    for start_index, start in enumerate(unique_starts):
        result = coordinate_search(
            start,
            objective=lambda value: -worst(value),
            feasible=lambda _value: True,
            maximum_sweeps=cfg.stage1_max_iterations,
            ranking=stage1_feasibility_restoration_key,
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
                "feasibility_restoration_key": list(result["ranking"]),
            }
        )
    # Search navigation may temporarily trade one unsafe margin for progress
    # on another.  Candidate selection considers every exact evaluation, not
    # merely navigation endpoints.
    stage1_candidates.extend(
        np.frombuffer(key, dtype=np.float64).copy() for key in cache
    )
    feasible_stage1 = [
        value for value in stage1_candidates if robust_feasible(value)
    ]
    if not feasible_stage1:
        stage1_value = max(stage1_candidates, key=worst)
        state, row, normalized = evaluate(stage1_value)
        raise WholeBodySafeReadyError(
            "whole-body stage 1 found no state meeting the named robust constraints",
            code="NO_ROBUST_FEASIBLE_STATE",
            report={
                "best_state_sha256": grounded.state_digest(state),
                "best_slacks": dict(row.slacks),
                "best_normalized_slacks": dict(zip(REQUIRED_SAFETY_SLACK_NAMES, normalized.tolist())),
                "worst_normalized_slack": float(np.min(normalized)),
                "required_normalized_slacks": dict(
                    zip(
                        REQUIRED_SAFETY_SLACK_NAMES,
                        required_normalized.tolist(),
                    )
                ),
                "stage1_runs": stage1_rows,
                "stage1_navigation_objective": (
                    "minimize_count_and_bounded_dimensionless_deficit_of_"
                    "uncleared_physical_gates"
                ),
                "evaluation_count": evaluation_count,
            },
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

    stage1_value = min(feasible_stage1, key=secondary)
    stage1_worst = worst(stage1_value)
    locked = float(np.min(required_normalized))

    def locked_feasible(value: np.ndarray) -> bool:
        return robust_feasible(value)

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
    if any(
        final_eval.slacks[name] < fallback_minimum_slacks[name]
        for name in REQUIRED_SAFETY_SLACK_NAMES
    ):
        raise WholeBodySafeReadyError(
            "fresh final evaluator did not preserve the named robust constraints",
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
                "algorithm": "deterministic_coordinate_local_robust_constrained_bridge",
                "global_optimum_claimed": False,
                "stage1_objective": "find_named_robust_constraint_feasible_set",
                "stage1_navigation_objective": (
                    "minimize_count_and_bounded_dimensionless_deficit_of_"
                    "uncleared_physical_gates"
                ),
                "stage2_objective": (
                    "minimize_weighted_root_31q_racket_error_inside_"
                    "named_robust_constraints"
                ),
                "racket_reference_authority": racket_reference_authority,
                "safety_weighted_against_tracking": False,
                "stage1_runs": stage1_rows,
                "stage1_worst_normalized_slack": float(stage1_worst),
                "stage1_lock_tolerance_normalized": cfg.stage1_lock_tolerance_normalized,
                "stage1_locked_worst_normalized_slack": float(locked),
                "fallback_minimum_slacks": dict(fallback_minimum_slacks),
                "required_normalized_slacks": dict(
                    zip(
                        REQUIRED_SAFETY_SLACK_NAMES,
                        required_normalized.tolist(),
                    )
                ),
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
