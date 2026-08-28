"""Host-only tests for measured-conditioned constrained whole-body ready search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_grounded_ready as grounded  # noqa: E402
import whole_body_safe_ready as safe_ready  # noqa: E402


def _state(*, root_z: float = 0.9, waist_yaw: float = 0.0) -> grounded.ReadyState:
    q = np.zeros(31, np.float64)
    q[grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")] = waist_yaw
    return grounded.ReadyState(
        q,
        np.asarray([0.0, 0.0, root_z], np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0], np.float64),
    )


def _lexical_evaluator(state: grounded.ReadyState) -> safe_ready.SafetyEvaluation:
    waist = float(
        state.joint_pos[
            grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")
        ]
    )
    z_delta = float(state.root_pos_w[2] - 0.9)
    # The teacher (waist=0, z_delta=0) fails two gates.  The max-min safety
    # point is waist=z_delta=0.10, with 0.05 raw slack on every limiting side.
    rows = {
        name: 0.20 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
    }
    rows["left_contact_load_slack_n"] = waist - 0.05
    rows["right_contact_load_slack_n"] = 0.15 - waist
    rows["root_height_slack_m"] = z_delta - 0.05
    rows["root_tilt_slack_rad"] = 0.15 - z_delta
    rotation = grounded._rotation_z(0.5 * waist)
    return safe_ready.SafetyEvaluation(
        slacks=rows,
        racket_position_w=np.asarray([0.8 + waist, -0.2, 1.1 + z_delta]),
        racket_rotation_w=rotation,
        evidence={"exact_contact_lp_reused": True},
    )


def _independent_racket_reference() -> dict[str, np.ndarray]:
    return {
        "racket_reference_position_w": np.asarray(
            [0.8, -0.2, 1.1], np.float64
        ),
        "racket_reference_rotation_w": np.eye(3),
    }


def _synthetic_positive_minima(value: float = 1.0e-6) -> dict[str, float]:
    """Give dimensionless fixtures an explicit feasible set.

    Production uses the named physical-unit robust reserve.  These toy
    evaluators intentionally use one synthetic unit for every field and must
    not accidentally redefine that production contract.
    """

    return {
        name: float(value) for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
    }


def test_search_minimizes_frame0_gap_inside_fixed_feasible_set() -> None:
    measured = _state()
    safe_seed = _state(root_z=1.0, waist_yaw=0.1)
    result = safe_ready.solve_measured_conditioned_whole_body_safe_ready(
        measured,
        evaluator=_lexical_evaluator,
        **_independent_racket_reference(),
        position_lower=np.full(31, -2.0),
        position_upper=np.full(31, 2.0),
        initial_states=(safe_seed,),
        slack_scales={
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        },
        config=safe_ready.WholeBodySearchConfig(
            stage1_max_iterations=100,
            stage2_max_iterations=120,
            stage1_lock_tolerance_normalized=1.0e-4,
            fallback_minimum_slacks=_synthetic_positive_minima(),
        ),
    )

    waist_index = grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")
    assert result.safety_slacks["left_contact_load_slack_n"] > 1.0e-6
    assert result.safety_slacks["right_contact_load_slack_n"] > 1.0e-6
    assert result.safety_slacks["root_height_slack_m"] > 1.0e-6
    assert result.safety_slacks["root_tilt_slack_rad"] > 1.0e-6
    assert result.worst_normalized_safety_slack >= (
        result.stage1_locked_worst_normalized_slack - 1.0e-9
    )
    assert 0.05 < result.state.joint_pos[waist_index] < 0.06
    assert 0.95 < result.state.root_pos_w[2] < 0.96
    assert result.changed_joint_mask[waist_index] is True
    assert sum(result.changed_joint_mask) == 1
    assert result.racket_position_delta_m != (0.0, 0.0, 0.0)
    assert result.racket_rotation_delta_rad != (0.0, 0.0, 0.0)
    assert result.optimizer_report["safety_weighted_against_tracking"] is False
    assert result.optimizer_report["global_optimum_claimed"] is False
    assert result.evaluator_evidence["exact_contact_lp_reused"] is True
    assert result.training_authorized is False
    assert result.deployment_authorized is False
    assert result.hardware_authorized is False


def test_exact_measured_frame0_is_preferred_when_all_gates_already_pass() -> None:
    waist_index = grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")

    def already_safe(state: grounded.ReadyState) -> safe_ready.SafetyEvaluation:
        waist = float(state.joint_pos[waist_index])
        return safe_ready.SafetyEvaluation(
            slacks={
                name: (
                    safe_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[name]
                    + 0.1
                    + abs(waist)
                )
                for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
            },
            racket_position_w=np.asarray([0.8 + waist, -0.2, 1.1]),
            racket_rotation_w=np.eye(3),
            evidence={"exact_contact_lp_reused": True},
        )

    measured = _state(waist_yaw=0.0)
    result = safe_ready.solve_measured_conditioned_whole_body_safe_ready(
        measured,
        evaluator=already_safe,
        racket_reference_position_w=np.asarray([0.7, -0.2, 1.1]),
        racket_reference_rotation_w=grounded._rotation_z(0.1),
        position_lower=np.full(31, -2.0),
        position_upper=np.full(31, 2.0),
        initial_states=(_state(waist_yaw=0.4),),
        slack_scales={
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        },
    )

    np.testing.assert_array_equal(result.state.joint_pos, measured.joint_pos)
    np.testing.assert_array_equal(result.state.root_pos_w, measured.root_pos_w)
    assert result.changed_joint_mask == (False,) * 31
    assert result.optimizer_report["exact_measured_frame0_selected"] is True
    assert result.optimizer_report["algorithm"] == (
        "exact_measured_frame0_safety_short_circuit"
    )
    assert result.optimizer_report["racket_reference_authority"] == (
        "caller_supplied_independent_measurement"
    )
    assert result.racket_position_delta_m == pytest.approx((0.1, 0.0, 0.0))
    assert np.linalg.norm(result.racket_rotation_delta_rad) > 0.09
    assert result.optimizer_report["stage2_objective_value"] > 0.0


def test_positive_but_nonrobust_measured_frame0_does_not_short_circuit() -> None:
    waist_index = grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")

    def thin_margin(state: grounded.ReadyState) -> safe_ready.SafetyEvaluation:
        waist = float(state.joint_pos[waist_index])
        return safe_ready.SafetyEvaluation(
            slacks={
                name: max(
                    2.0e-6,
                    0.5 * safe_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[name],
                )
                + waist
                for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
            },
            racket_position_w=np.asarray([0.8 + waist, -0.2, 1.1]),
            racket_rotation_w=np.eye(3),
            evidence={"exact_contact_lp_reused": True},
        )

    measured = _state(waist_yaw=0.0)
    result = safe_ready.solve_measured_conditioned_whole_body_safe_ready(
        measured,
        evaluator=thin_margin,
        **_independent_racket_reference(),
        position_lower=np.full(31, -0.2),
        position_upper=np.full(31, 0.2),
        initial_states=(_state(waist_yaw=0.2),),
        slack_scales={
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        },
        config=safe_ready.WholeBodySearchConfig(
            stage1_max_iterations=40,
            stage2_max_iterations=40,
            fallback_minimum_slacks={
                name: value + 0.19
                for name, value in thin_margin(measured).slacks.items()
            },
        ),
    )

    assert min(thin_margin(measured).slacks.values()) > 1.0e-6
    assert any(
        thin_margin(measured).slacks[name]
        < safe_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[name]
        for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
    )
    assert result.optimizer_report.get("exact_measured_frame0_selected") is not True
    assert result.state.joint_pos[waist_index] > 0.19


def test_search_fails_closed_without_positive_all_gate_interior() -> None:
    def infeasible(state: grounded.ReadyState) -> safe_ready.SafetyEvaluation:
        row = _lexical_evaluator(state)
        slacks = dict(row.slacks)
        slacks["table_clearance_slack_m"] = -0.01
        return safe_ready.SafetyEvaluation(
            slacks=slacks,
            racket_position_w=row.racket_position_w,
            racket_rotation_w=row.racket_rotation_w,
            evidence=row.evidence,
        )

    with pytest.raises(
        safe_ready.WholeBodySafeReadyError,
        match="no state meeting the named robust constraints",
    ) as caught:
        safe_ready.solve_measured_conditioned_whole_body_safe_ready(
            _state(),
            evaluator=infeasible,
            **_independent_racket_reference(),
            position_lower=np.full(31, -2.0),
            position_upper=np.full(31, 2.0),
            initial_states=(_state(root_z=1.0, waist_yaw=0.1),),
            slack_scales={
                name: 1.0
                for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
            },
            config=safe_ready.WholeBodySearchConfig(
                stage1_max_iterations=30,
                stage2_max_iterations=30,
            ),
        )
    assert caught.value.code == "NO_ROBUST_FEASIBLE_STATE"
    assert caught.value.report["best_slacks"]["table_clearance_slack_m"] < 0.0


def test_stage1_restores_feasibility_across_ground_lp_sentinel_plateau() -> None:
    """LP sentinels must not suppress smooth pre-contact search progress.

    This models the Pod failure: while either sole is outside the contact
    tolerance the LP returns conservative ``-1`` sentinels, including a
    ground residual which normalizes to roughly ``-1e6``.  Reaching the valid
    LP region requires two separate coordinate moves, neither of which alone
    improves that raw worst value.
    """

    waist_index = grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")
    measured = _state(root_z=0.9, waist_yaw=0.0)

    def discontinuous_lp(
        state: grounded.ReadyState,
    ) -> safe_ready.SafetyEvaluation:
        waist = float(state.joint_pos[waist_index])
        root_delta = float(state.root_pos_w[2] - 0.9)
        left_sole = waist - 0.05
        right_sole = root_delta - 0.05
        lp_feasible = left_sole > 0.0 and right_sole > 0.0
        slacks = {
            name: 0.2 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        }
        slacks["left_sole_floor_slack_m"] = left_sole
        slacks["right_sole_floor_slack_m"] = right_sole
        if not lp_feasible:
            slacks.update(
                {
                    "left_contact_load_slack_n": -1.0,
                    "right_contact_load_slack_n": -1.0,
                    "support_margin_slack_m": -1.0,
                    "qdes_slack_rad": -1.0,
                    "torque_slack_nm": -1.0,
                    "ground_lp_residual_slack": 2.0e-7 - 1.0,
                }
            )
        return safe_ready.SafetyEvaluation(
            slacks=slacks,
            racket_position_w=np.asarray(
                [0.8 + waist, -0.2, 1.1 + root_delta], np.float64
            ),
            racket_rotation_w=np.eye(3),
            evidence={"lp_feasible": lp_feasible},
        )

    measured_row = discontinuous_lp(measured)
    measured_ground_normalized = (
        measured_row.slacks["ground_lp_residual_slack"]
        / safe_ready.DEFAULT_SLACK_SCALES["ground_lp_residual_slack"]
    )
    assert measured_ground_normalized == pytest.approx(-999999.8)

    result = safe_ready.solve_measured_conditioned_whole_body_safe_ready(
        measured,
        evaluator=discontinuous_lp,
        **_independent_racket_reference(),
        position_lower=np.full(31, -0.2),
        position_upper=np.full(31, 0.2),
        config=safe_ready.WholeBodySearchConfig(
            stage1_max_iterations=40,
            stage2_max_iterations=40,
            fallback_minimum_slacks=_synthetic_positive_minima(),
        ),
    )

    assert sum(
        row["accepted_steps"]
        for row in result.optimizer_report["stage1_runs"]
    ) > 0
    assert result.evaluator_evidence["lp_feasible"] is True
    assert result.safety_slacks["left_sole_floor_slack_m"] > 0.0
    assert result.safety_slacks["right_sole_floor_slack_m"] > 0.0
    assert all(value > 0.0 for value in result.safety_slacks.values())
    assert all(
        len(row["feasibility_restoration_key"]) == 5
        for row in result.optimizer_report["stage1_runs"]
    )
    assert all(
        value
        > safe_ready.WholeBodySearchConfig().positive_gate_normalized_slack
        for value in result.normalized_safety_slacks.values()
    )


def test_stage2_lock_cannot_relax_below_original_positive_gate() -> None:
    """Regression for a stage-1 interior thinner than the lock tolerance.

    Before the original gate was part of the lock, stage 1 found ``x=5.05e-5``
    while stage 2 was allowed to track toward ``x=0`` down to roughly
    ``5e-7``.  That state satisfied the relaxed lock but violated the caller's
    ``1e-6`` admission gate.
    """

    waist_index = grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")

    def thin_interior(state: grounded.ReadyState) -> safe_ready.SafetyEvaluation:
        x = float(state.joint_pos[waist_index])
        slacks = {
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        }
        slacks["left_contact_load_slack_n"] = x
        slacks["right_contact_load_slack_n"] = 1.01e-4 - x
        return safe_ready.SafetyEvaluation(
            slacks=slacks,
            racket_position_w=np.asarray([0.8, -0.2, 1.1], np.float64),
            racket_rotation_w=np.eye(3),
            evidence={"selected_witness": True},
        )

    measured = _state(waist_yaw=0.0)
    safe_seed = _state(waist_yaw=5.05e-5)
    gate = 1.0e-6
    result = safe_ready.solve_measured_conditioned_whole_body_safe_ready(
        measured,
        evaluator=thin_interior,
        **_independent_racket_reference(),
        position_lower=np.full(31, -2.0),
        position_upper=np.full(31, 2.0),
        initial_states=(safe_seed,),
        slack_scales={
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        },
        config=safe_ready.WholeBodySearchConfig(
            positive_gate_normalized_slack=gate,
            stage1_lock_tolerance_normalized=5.0e-5,
            stage1_max_iterations=30,
            stage2_max_iterations=60,
            joint_weight=1.0e12,
            fallback_minimum_slacks=_synthetic_positive_minima(
                np.nextafter(gate, np.inf)
            ),
        ),
    )

    assert result.optimizer_report["stage1_worst_normalized_slack"] > gate
    assert result.stage1_locked_worst_normalized_slack > gate
    assert result.worst_normalized_safety_slack > gate
    assert all(value > gate for value in result.normalized_safety_slacks.values())


def test_stage2_rejects_exact_original_gate_and_keeps_safe_fallback() -> None:
    waist_index = grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")
    gate = 1.0e-6
    half_width = 0.3 / (2**10)

    def equality_at_teacher(
        state: grounded.ReadyState,
    ) -> safe_ready.SafetyEvaluation:
        x = float(state.joint_pos[waist_index])
        slacks = {
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        }
        slacks["left_contact_load_slack_n"] = gate + x
        slacks["right_contact_load_slack_n"] = gate + 2.0 * half_width - x
        return safe_ready.SafetyEvaluation(
            slacks=slacks,
            racket_position_w=np.asarray([0.8, -0.2, 1.1], np.float64),
            racket_rotation_w=np.eye(3),
            evidence={"selected_witness": True},
        )

    result = safe_ready.solve_measured_conditioned_whole_body_safe_ready(
        _state(waist_yaw=0.0),
        evaluator=equality_at_teacher,
        **_independent_racket_reference(),
        position_lower=np.full(31, -2.0),
        position_upper=np.full(31, 2.0),
        initial_states=(_state(waist_yaw=half_width),),
        slack_scales={
            name: 1.0 for name in safe_ready.REQUIRED_SAFETY_SLACK_NAMES
        },
        config=safe_ready.WholeBodySearchConfig(
            positive_gate_normalized_slack=gate,
            stage1_lock_tolerance_normalized=5.0e-4,
            stage1_max_iterations=30,
            stage2_max_iterations=60,
            joint_weight=1.0e12,
            fallback_minimum_slacks=_synthetic_positive_minima(
                np.nextafter(gate, np.inf)
            ),
        ),
    )

    assert result.stage1_locked_worst_normalized_slack > gate
    assert result.worst_normalized_safety_slack > gate
    assert result.state.joint_pos[waist_index] > 0.0


def test_default_search_scope_releases_root_waist_legs_and_all_31_joints() -> None:
    config = safe_ready.WholeBodySearchConfig()
    assert config.movable_joint_names == grounded.RUNTIME_JOINT_NAMES
    assert set(grounded.LEG_JOINT_NAMES).issubset(config.movable_joint_names)
    assert {
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    }.issubset(config.movable_joint_names)
    assert set(safe_ready.REQUIRED_SAFETY_SLACK_NAMES) == {
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
    }
