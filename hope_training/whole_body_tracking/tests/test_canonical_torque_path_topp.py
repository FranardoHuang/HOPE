"""CPU-only tests for strict torque-aware scalar path retiming."""
from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "canonical_torque_path_topp", _SCRIPTS / "canonical_torque_path_topp.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["canonical_torque_path_topp"] = _MOD
_SPEC.loader.exec_module(_MOD)

DirectActuatorContract = _MOD.DirectActuatorContract
GroundContactConfig = _MOD.GroundContactConfig
GroundContactLPSolution = _MOD.GroundContactLPSolution
GROUND_LP_OBJECTIVE_FEASIBILITY = _MOD.GROUND_LP_OBJECTIVE_FEASIBILITY
GROUND_LP_OBJECTIVE_HOLD_MINIMAX = _MOD.GROUND_LP_OBJECTIVE_HOLD_MINIMAX
IncompleteTorqueCertificate = _MOD.IncompleteTorqueCertificate
MujocoGroundContactLPSolver = _MOD.MujocoGroundContactLPSolver
MujocoSmoothInverseDynamics = _MOD.MujocoSmoothInverseDynamics
TangentPathGrid = _MOD.TangentPathGrid
TorqueRetimeError = _MOD.TorqueRetimeError
_mujoco_model_binding = _MOD._mujoco_model_binding
_resolve_grounded_actuator_limits = _MOD._resolve_grounded_actuator_limits
_solve_ground_contact_force_lp = _MOD._solve_ground_contact_force_lp
direct_actuator_contract_from_mujoco = _MOD.direct_actuator_contract_from_mujoco
identify_dynamics_slice = _MOD.identify_dynamics_slice
resolve_generalized_force_limits = _MOD.resolve_generalized_force_limits
retime_torque_path = _MOD.retime_torque_path
validate_mujoco_tangent_path_grid = _MOD.validate_mujoco_tangent_path_grid

_SCIPY_VERSION_PARTS = tuple(
    int(part) for part in scipy.__version__.split(".")[:2]
)
_SCIPY_HIGHS_TESTABLE = _SCIPY_VERSION_PARTS >= (1, 9)


def _linear_path(samples: int = 21, *, nv: int = 1) -> TangentPathGrid:
    s = np.linspace(0.0, 1.0, samples)
    midpoint = 0.5 * (s[:-1] + s[1:])
    slope = np.linspace(1.0, 0.6, nv)
    return TangentPathGrid(
        path_position=s,
        qpos=s[:, None] * slope[None, :],
        q_s=np.tile(slope, (samples, 1)),
        q_ss=np.zeros((samples, nv)),
        midpoint_qpos=midpoint[:, None] * slope[None, :],
        midpoint_q_s=np.tile(slope, (samples - 1, 1)),
        midpoint_q_ss=np.zeros((samples - 1, nv)),
    )


def _contract(
    *,
    effort: float = 4.0,
    frictionloss: float = 0.0,
    free_dof_count: int = 0,
    support_mode: str = "fixed_base",
) -> DirectActuatorContract:
    return DirectActuatorContract(
        dof_to_actuator=np.array([0]),
        actuator_gear=np.array([1.0]),
        actuator_force_lower=np.array([-effort]),
        actuator_force_upper=np.array([effort]),
        actuator_control_lower=np.array([-np.inf]),
        actuator_control_upper=np.array([np.inf]),
        joint_force_lower=np.array([-np.inf]),
        joint_force_upper=np.array([np.inf]),
        frictionloss=np.array([frictionloss]),
        free_dof_count=free_dof_count,
        support_mode=support_mode,
    )


def _one_dof_dynamics(
    *, mass: float = 1.0, damping: float = 0.0, gravity: float = 0.0
):
    def inverse(_qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray) -> np.ndarray:
        return mass * qacc + damping * qvel + gravity

    return inverse


def test_identification_keeps_mass_gravity_damping_and_curvature_terms_separate():
    # q_s=1.5, q_ss=.3:
    # A=m*q_s=3
    # c_sqrt=d*q_s=.75
    # c_x=m*q_ss=.6
    # c0=gravity=1.2
    identified = identify_dynamics_slice(
        np.array([0.4]),
        np.array([1.5]),
        np.array([0.3]),
        _one_dof_dynamics(mass=2.0, damping=0.5, gravity=1.2),
    )
    assert identified.A == pytest.approx([3.0])
    assert identified.c0 == pytest.approx([1.2])
    assert identified.c_sqrt == pytest.approx([0.75])
    assert identified.c_x == pytest.approx([0.6])
    assert identified.generalized_force(2.25, -0.4) == pytest.approx(
        [-1.2 + 1.2 + 1.125 + 1.35]
    )


def test_negative_gear_force_control_joint_caps_and_friction_all_intersect():
    contract = DirectActuatorContract(
        dof_to_actuator=np.array([0]),
        actuator_gear=np.array([-2.0]),
        actuator_force_lower=np.array([-2.0]),
        actuator_force_upper=np.array([3.0]),
        actuator_control_lower=np.array([-1.0]),
        actuator_control_upper=np.array([2.5]),
        joint_force_lower=np.array([-4.0]),
        joint_force_upper=np.array([1.5]),
        frictionloss=np.array([0.25]),
    )
    lower, upper, report = resolve_generalized_force_limits(contract)
    # actuator intersection [-1, 2.5], negative gear -> [-5, 2],
    # joint intersection -> [-4, 1.5], full friction margin -> [-3.75, 1.25].
    assert lower == pytest.approx([-3.75])
    assert upper == pytest.approx([1.25])
    assert report["actuator_gear"] == [-2.0]
    assert report["frictionloss_full_margin"] == [0.25]


def test_analytic_one_dof_gravity_retime_is_rest_to_rest_and_effort_bounded():
    result = retime_torque_path(
        _linear_path(31),
        _one_dof_dynamics(mass=1.0, gravity=1.0),
        _contract(effort=5.0),
        np.array([10.0]),
        search_samples=65,
    )
    # Available positive/negative scalar acceleration is +4 / -6.  The exact
    # bang-bang duration over unit distance is sqrt(2D/a+)+sqrt(2D/a-) after
    # solving the switch distance, equivalently sqrt(2*(1/a+ + 1/|a-|)).
    assert result.time_s[-1] == pytest.approx(np.sqrt(2.0 * (1.0 / 4.0 + 1.0 / 6.0)))
    assert result.path_speed[0] == 0.0
    assert result.path_speed[-1] == 0.0
    assert np.max(result.path_acceleration) <= 4.0 + 2e-8
    assert np.min(result.path_acceleration) >= -6.0 - 2e-8
    assert result.report["uniform_actuator_torque_enforced"] is False
    assert "scalar path acceleration" in result.report["uniform_torque_explanation"]
    assert (
        result.report["status"]
        == "CANDIDATE_FIXED_BASE_THREE_POINT_COLLOCATION"
    )


def test_viscous_damping_is_sqrt_x_and_changes_the_time_law():
    no_damping = retime_torque_path(
        _linear_path(31),
        _one_dof_dynamics(mass=1.0, gravity=1.0),
        _contract(effort=5.0),
        np.array([10.0]),
        search_samples=65,
    )
    damping = retime_torque_path(
        _linear_path(31),
        _one_dof_dynamics(mass=1.0, damping=0.8, gravity=1.0),
        _contract(effort=5.0),
        np.array([10.0]),
        search_samples=65,
    )
    assert damping.time_s[-1] > no_damping.time_s[-1]
    assert damping.report["max_identification_probe_residual"] < 1e-10


def test_dry_friction_is_full_worst_case_margin_not_free_assistance():
    baseline = retime_torque_path(
        _linear_path(31),
        _one_dof_dynamics(),
        _contract(effort=4.0, frictionloss=0.0),
        np.array([10.0]),
        search_samples=65,
    )
    reserved = retime_torque_path(
        _linear_path(31),
        _one_dof_dynamics(),
        _contract(effort=4.0, frictionloss=0.75),
        np.array([10.0]),
        search_samples=65,
    )
    assert reserved.time_s[-1] > baseline.time_s[-1]
    assert (
        reserved.report["frictionloss_policy"]
        == "disabled_in_inverse_dynamics_and_full_absolute_value_deducted_from_both_limits"
    )
    assert reserved.report["force_contract"]["generalized_force_after_friction"] == [
        [-3.25, 3.25]
    ]


def test_raw_sign_dependent_friction_inside_inverse_dynamics_is_rejected_nonaffine():
    def wrong_inverse(_qpos, qvel, qacc):
        # At qvel=0 this implementation changes branch with acceleration, as a
        # naive dry-friction inverse often does.  It must not be linearised into
        # a fake A*u+b certificate.
        direction = np.where(
            np.abs(qvel) > 1e-12, np.sign(qvel), np.sign(qacc)
        )
        return qacc + 0.4 * direction

    with pytest.raises(TorqueRetimeError, match="not affine"):
        identify_dynamics_slice(
            np.array([0.0]), np.array([1.0]), np.array([0.0]), wrong_inverse
        )


def test_generic_nonlinear_acceleration_dependence_is_rejected():
    def nonlinear(_qpos, _qvel, qacc):
        return qacc + 0.05 * qacc * qacc

    with pytest.raises(TorqueRetimeError, match="not affine"):
        identify_dynamics_slice(
            np.array([0.0]), np.array([1.0]), np.array([0.0]), nonlinear
        )


def test_unprobed_speed_nonlinearity_is_caught_by_exact_achieved_state_validation():
    roots = (0.0, 0.25, 1.0, 2.25, 3.1, 4.0)

    def adversarial(_qpos, qvel, qacc):
        x = qvel * qvel
        hidden = np.ones_like(x)
        for root in roots:
            hidden *= x - root
        # Every identification probe sees exactly zero hidden force, while a
        # fitted-only final validator would miss the large force elsewhere.
        return qacc + 100.0 * hidden

    with pytest.raises(TorqueRetimeError, match="final actuator-force validation"):
        retime_torque_path(
            _linear_path(21),
            adversarial,
            _contract(effort=2.0),
            np.array([20.0]),
            search_samples=65,
        )


def test_explicit_model_binding_mismatch_fails_before_dynamics_claim():
    contract = _contract(effort=4.0)
    contract = DirectActuatorContract(
        **{
            **contract.__dict__,
            "model_binding": "a" * 64,
        }
    )
    inverse = _one_dof_dynamics()
    inverse.model_binding = "b" * 64
    with pytest.raises(TorqueRetimeError, match="different model bindings"):
        retime_torque_path(
            _linear_path(11), inverse, contract, np.array([5.0])
        )


@pytest.mark.parametrize(
    ("support_mode", "expected_missing"),
    [
        ("floating_no_contact", "ground contact mode"),
        ("ground", "fixed CPU LP solver"),
    ],
)
def test_floating_root_never_gets_a_joint_row_torque_certificate(
    support_mode: str, expected_missing: str
):
    contract = _contract(
        effort=5.0, free_dof_count=1, support_mode=support_mode
    )
    with pytest.raises(IncompleteTorqueCertificate) as raised:
        retime_torque_path(
            _linear_path(11),
            _one_dof_dynamics(),
            contract,
            np.array([5.0]),
        )
    assert raised.value.report["status"] == "INCOMPLETE_FAIL_CLOSED"
    assert expected_missing in raised.value.report["missing"]


def test_no_early_brake_window_is_a_real_constrained_comparison():
    path = _linear_path(21)
    ordinary = retime_torque_path(
        path,
        _one_dof_dynamics(),
        _contract(effort=2.0),
        np.array([20.0]),
        window=(0.4, 0.8),
        no_early_brake=False,
        search_samples=65,
    )
    protected = retime_torque_path(
        path,
        _one_dof_dynamics(),
        _contract(effort=2.0),
        np.array([20.0]),
        window=(0.4, 0.8),
        no_early_brake=True,
        search_samples=65,
    )
    cell_midpoint = 0.5 * (
        protected.path_position[:-1] + protected.path_position[1:]
    )
    inside = (cell_midpoint >= 0.4) & (cell_midpoint <= 0.8)
    assert np.any(ordinary.path_acceleration[inside] < -1e-5)
    assert np.all(protected.path_acceleration[inside] >= -2e-9)
    assert protected.time_s[-1] > ordinary.time_s[-1]
    assert (
        protected.report["window_policy"]
        == "u_nonnegative_from_path_start_through_window_end"
    )


def test_no_early_brake_without_followthrough_fails_instead_of_faking_green():
    with pytest.raises(TorqueRetimeError, match="followthrough|infeasible|transition"):
        retime_torque_path(
            _linear_path(21),
            _one_dof_dynamics(),
            _contract(effort=2.0),
            np.array([20.0]),
            window=(0.5, 1.0),
            no_early_brake=True,
            search_samples=65,
        )


def _two_point_ground_map() -> tuple[np.ndarray, np.ndarray]:
    """Two floor points at y=+/-0.1 for an 8-DoF toy floating plant."""

    mapping = np.zeros((8, 6), np.float64)
    for point, position in enumerate(
        (np.array([0.0, -0.1, 0.0]), np.array([0.0, 0.1, 0.0]))
    ):
        skew = np.array(
            [
                [0.0, -position[2], position[1]],
                [position[2], 0.0, -position[0]],
                [-position[1], position[0], 0.0],
            ]
        )
        mapping[:3, 3 * point : 3 * point + 3] = np.eye(3)
        mapping[3:6, 3 * point : 3 * point + 3] = skew
    return mapping, np.array([0, 1], np.int64)


@pytest.mark.skipif(
    not _SCIPY_HIGHS_TESTABLE,
    reason="old local SciPy HiGHS can deadlock; exact test runs in A3 CPU env",
)
def test_ground_lp_balances_free_root_and_actuator_rows_with_both_feet_loaded():
    force_map, feet = _two_point_ground_map()
    demand = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, -1.0])
    solution = _solve_ground_contact_force_lp(
        demand,
        force_map,
        feet,
        np.array([6, 7]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        friction_coefficient=0.5,
        minimum_normal_force_per_foot_n=1.0,
        residual_tolerance=1e-9,
        solver_name="scipy.optimize.linprog:highs",
    )
    assert solution.feasible
    assert solution.actuator_generalized_force == pytest.approx([1.0, -1.0])
    assert np.sum(solution.point_force_floor[:, 2]) == pytest.approx(100.0)
    assert solution.point_force_floor[:, 2] == pytest.approx([50.0, 50.0])
    assert solution.root_residual < 1e-8


@pytest.mark.skipif(
    not _SCIPY_HIGHS_TESTABLE,
    reason="old local SciPy HiGHS can deadlock; exact test runs in A3 CPU env",
)
def test_ground_lp_default_remains_the_explicit_feasibility_solve():
    force_map, feet = _two_point_ground_map()
    arguments = (
        np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, -1.0]),
        force_map,
        feet,
        np.array([6, 7]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
    )
    keywords = {
        "friction_coefficient": 0.5,
        "minimum_normal_force_per_foot_n": 1.0,
        "residual_tolerance": 1e-9,
        "solver_name": "scipy.optimize.linprog:highs",
    }
    implicit = _solve_ground_contact_force_lp(*arguments, **keywords)
    explicit = _solve_ground_contact_force_lp(
        *arguments,
        **keywords,
        lp_objective=GROUND_LP_OBJECTIVE_FEASIBILITY,
    )
    assert implicit.feasible
    assert explicit.feasible
    assert implicit.actuator_generalized_force == pytest.approx(
        explicit.actuator_generalized_force
    )
    assert implicit.point_force_floor == pytest.approx(explicit.point_force_floor)
    assert implicit.report == explicit.report
    assert implicit.report["variables"] == 8
    assert "lp_objective" not in implicit.report


def _load_sharing_ground_map() -> tuple[np.ndarray, np.ndarray]:
    """Toy floating plant whose foot-load split can unload one actuator."""

    mapping = np.zeros((7, 6), np.float64)
    mapping[:3, :3] = np.eye(3)
    mapping[:3, 3:] = np.eye(3)
    mapping[6, 2] = 1.0
    mapping[6, 5] = -1.0
    return mapping, np.array([0, 1], np.int64)


@pytest.mark.skipif(
    not _SCIPY_HIGHS_TESTABLE,
    reason="old local SciPy HiGHS can deadlock; exact test runs in A3 CPU env",
)
def test_ground_lp_hold_minimax_is_bounded_and_not_worse_than_feasibility():
    force_map, feet = _load_sharing_ground_map()
    arguments = (
        np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 120.0]),
        force_map,
        feet,
        np.array([6]),
        np.array([-50.0]),
        np.array([150.0]),
    )
    keywords = {
        "friction_coefficient": 0.5,
        "minimum_normal_force_per_foot_n": 10.0,
        "residual_tolerance": 1e-9,
        "solver_name": "scipy.optimize.linprog:highs",
    }
    feasibility = _solve_ground_contact_force_lp(*arguments, **keywords)
    minimax = _solve_ground_contact_force_lp(
        *arguments,
        **keywords,
        lp_objective=GROUND_LP_OBJECTIVE_HOLD_MINIMAX,
    )
    assert feasibility.feasible
    assert minimax.feasible
    assert minimax.equality_residual < 1e-8
    assert minimax.root_residual < 1e-8
    # The asymmetric executable interval has center 50, but physical-ready
    # hold torque is zero.  The contact constraints make 40 the closest
    # attainable torque to zero; a center-based objective would choose 50.
    assert minimax.actuator_generalized_force == pytest.approx([40.0], abs=1e-8)
    assert minimax.point_force_floor[:, 2] == pytest.approx(
        [90.0, 10.0], abs=1e-8
    )
    assert np.all(minimax.actuator_generalized_force >= -50.0 - 1e-8)
    assert np.all(minimax.actuator_generalized_force <= 150.0 + 1e-8)
    assert minimax.report["max_inequality_violation"] < 1e-8
    assert minimax.report["bound_violation"] < 1e-8
    feasibility_hold_ratio = float(
        np.max(
            np.maximum(
                np.maximum(feasibility.actuator_generalized_force, 0.0)
                / np.array([150.0]),
                np.maximum(-feasibility.actuator_generalized_force, 0.0)
                / np.array([50.0]),
            )
        )
    )
    assert (
        minimax.report["max_normalized_available_hold_torque"]
        <= feasibility_hold_ratio + 1e-10
    )
    assert minimax.report["minimax_objective_value"] == pytest.approx(
        40.0 / 150.0, abs=1e-9
    )
    assert minimax.report["objective_mode"] == GROUND_LP_OBJECTIVE_HOLD_MINIMAX
    assert minimax.report["objective_effort_lower"] == pytest.approx([-50.0])
    assert minimax.report["objective_effort_upper"] == pytest.approx([150.0])
    assert minimax.report["negative_available_hold_torque"] == pytest.approx(
        [50.0]
    )
    assert minimax.report["positive_available_hold_torque"] == pytest.approx(
        [150.0]
    )
    assert minimax.report[
        "optimum_max_normalized_available_hold_torque"
    ] == pytest.approx(
        minimax.report["minimax_objective_value"]
    )
    assert minimax.report[
        "max_normalized_available_hold_torque"
    ] == pytest.approx(minimax.report["minimax_objective_value"], abs=1e-9)
    with pytest.raises(TorqueRetimeError, match="strictly contain zero"):
        _solve_ground_contact_force_lp(
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            np.array([0.0]),
            arguments[5],
            **keywords,
            lp_objective=GROUND_LP_OBJECTIVE_HOLD_MINIMAX,
        )


@pytest.mark.skipif(
    not _SCIPY_HIGHS_TESTABLE,
    reason="old local SciPy HiGHS can deadlock; exact test runs in A3 CPU env",
)
def test_ground_lp_uses_inscribed_l1_friction_pyramid_not_unsafe_square():
    force_map, feet = _two_point_ground_map()
    # Each component is below mu*sum(fn)=50 N, so the unsafe independent
    # square bounds would pass.  The diagonal vector violates the conservative
    # L1 pyramid because |40|+|40| > 50.
    demand = np.array([40.0, 40.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    solution = _solve_ground_contact_force_lp(
        demand,
        force_map,
        feet,
        np.array([6, 7]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        friction_coefficient=0.5,
        minimum_normal_force_per_foot_n=1.0,
        residual_tolerance=1e-9,
        solver_name="scipy.optimize.linprog:highs",
    )
    assert not solution.feasible
    assert solution.report["highs_status"] == 2


@pytest.mark.skipif(
    not _SCIPY_HIGHS_TESTABLE,
    reason="old local SciPy optimize import can deadlock; exact test runs in A3 CPU env",
)
def test_ground_lp_numerical_solver_failure_is_incomplete_not_physical_false(
    monkeypatch,
):
    import scipy.optimize

    class Failed:
        status = 4
        success = False
        message = "synthetic numerical failure"
        nit = 3
        x = None

    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: Failed())
    force_map, feet = _two_point_ground_map()
    with pytest.raises(IncompleteTorqueCertificate) as raised:
        _solve_ground_contact_force_lp(
            np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            force_map,
            feet,
            np.array([6, 7]),
            np.array([-2.0, -2.0]),
            np.array([2.0, 2.0]),
            friction_coefficient=0.5,
            minimum_normal_force_per_foot_n=1.0,
            residual_tolerance=1e-9,
            solver_name="scipy.optimize.linprog:highs",
        )
    assert raised.value.report["status"] == "INCOMPLETE_FAIL_CLOSED"
    assert raised.value.report["details"]["highs_status"] == 4


def _toy_grounded_contract(binding: str) -> DirectActuatorContract:
    return DirectActuatorContract(
        dof_to_actuator=np.array([-1, -1, -1, -1, -1, -1, 0]),
        actuator_gear=np.array([1.0]),
        actuator_force_lower=np.array([-4.0]),
        actuator_force_upper=np.array([4.0]),
        actuator_control_lower=np.array([-np.inf]),
        actuator_control_upper=np.array([np.inf]),
        joint_force_lower=np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -4.0]
        ),
        joint_force_upper=np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 4.0]
        ),
        frictionloss=np.zeros(7),
        free_dof_count=6,
        support_mode="ground",
        contact_mode="double_support_floor",
        fixed_lp_solver="scipy.optimize.linprog:highs",
        model_binding=binding,
    )


class _AlwaysFeasibleGroundSolver:
    def __init__(self, binding: str):
        self.model_binding = binding
        self.calls = 0

    def validate_path_grid(self, path):
        return {
            "status": "PASS_SAMPLED_MUJOCO_FINITE_DIFFERENCE",
            "test_double": True,
        }

    def solve(
        self,
        qpos,
        qvel,
        qacc,
        actuated_dof_indices,
        effort_lower,
        effort_upper,
        velocity_limits,
        *,
        expected_generalized_force,
        path_tangent,
    ):
        self.calls += 1
        assert np.asarray(path_tangent).shape == (7,)
        assert np.asarray(actuated_dof_indices).tolist() == [6]
        return GroundContactLPSolution(
            feasible=True,
            actuator_generalized_force=np.array([0.0]),
            point_force_floor=np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]]),
            equality_residual=1e-12,
            root_residual=1e-12,
            report={
                "solver": "scipy.optimize.linprog:highs",
                "solver_library": "scipy",
                "scipy_version": "test-double",
                "friction_pyramid": "abs(ft1)+abs(ft2)<=mu*fn",
                "friction_coefficient": 0.5,
                "minimum_normal_force_per_foot_n": 1.0,
                "model_binding": self.model_binding,
                "inverse_dynamics": "test-double",
                "normal_force_per_foot_n": [50.0, 50.0],
                "contact_geometry": {
                    "test_double": True,
                    "friction_used": 0.5,
                    "feet": [
                        {
                            "foot_index": 0,
                            "support_vertices": 4,
                            "support_vertex_floor_distance_m": [0.0],
                            "minimum_floor_distance_m": 0.0,
                        },
                        {
                            "foot_index": 1,
                            "support_vertices": 4,
                            "support_vertex_floor_distance_m": [0.0],
                            "minimum_floor_distance_m": 0.0,
                        },
                    ],
                },
                "contact_kinematics": {
                    "max_abs_Jp_qs_m_per_path_unit": 0.0,
                    "max_abs_contact_velocity_m_s": 0.0,
                    "max_abs_contact_acceleration_m_s2": 0.0,
                },
            },
        )


def test_grounded_retimer_uses_contact_lp_and_labels_only_sampled_certificate():
    binding = "a" * 64

    def inverse(_qpos, qvel, qacc):
        return qacc + 0.05 * qvel

    inverse.model_binding = binding
    solver = _AlwaysFeasibleGroundSolver(binding)
    result = retime_torque_path(
        _linear_path(11, nv=7),
        inverse,
        _toy_grounded_contract(binding),
        np.full(7, 5.0),
        grounded_contact_solver=solver,
        search_samples=17,
    )
    assert solver.calls > 0
    assert result.report["status"] == (
        "CANDIDATE_GROUNDED_DOUBLE_SUPPORT_SAMPLED_LP"
    )
    assert result.report["plant_binding_status"] == "BOUND_MATCH_GROUNDED_TRIPLE"
    assert result.report["contact_certificate"] == (
        "SAMPLED_LEFT_MIDPOINT_RIGHT_EACH_CELL_ONLY"
    )
    assert result.report["continuous_cell_certificate"] is False
    assert (
        result.report["grounded_contact_lp"]["claim_boundary"]
        == "sampled path feasibility; not continuous global TOPP-RA"
    )


def test_grounded_retimer_requires_exact_shared_model_binding():
    binding = "a" * 64

    def inverse(_qpos, _qvel, qacc):
        return qacc

    inverse.model_binding = binding
    solver = _AlwaysFeasibleGroundSolver("b" * 64)
    with pytest.raises(IncompleteTorqueCertificate, match="model binding"):
        retime_torque_path(
            _linear_path(11, nv=7),
            inverse,
            _toy_grounded_contract(binding),
            np.full(7, 5.0),
            grounded_contact_solver=solver,
            search_samples=17,
        )


@pytest.mark.skipif(
    not os.environ.get("A3_MJCF_PATH"),
    reason="set A3_MJCF_PATH for exact vendor-MJCF CPU integration",
)
def test_exact_a3_static_double_support_lp_integration():
    mujoco = pytest.importorskip("mujoco")
    model_path = Path(os.environ["A3_MJCF_PATH"]).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    binding = _mujoco_model_binding(model)
    source_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with pytest.raises(IncompleteTorqueCertificate, match="model binding"):
        MujocoGroundContactLPSolver(
            model,
            GroundContactConfig(
                expected_model_binding="0" * 64,
                model_source_path=str(model_path),
                expected_source_sha256=source_sha256,
            ),
        )
    with pytest.raises(IncompleteTorqueCertificate, match="source SHA"):
        MujocoGroundContactLPSolver(
            model,
            GroundContactConfig(
                expected_model_binding=binding,
                model_source_path=str(model_path),
                expected_source_sha256="0" * 64,
            ),
        )
    solver = MujocoGroundContactLPSolver(
        model,
        GroundContactConfig(
            expected_model_binding=binding,
            model_source_path=str(model_path),
            expected_source_sha256=source_sha256,
        ),
    )
    inverse = MujocoSmoothInverseDynamics(model)
    contract = direct_actuator_contract_from_mujoco(
        model,
        support_mode="ground",
        contact_mode="double_support_floor",
        fixed_lp_solver="scipy.optimize.linprog:highs",
    )
    lower, upper, actuated, _ = _resolve_grounded_actuator_limits(
        contract, int(model.nv)
    )

    qpos = np.asarray(model.qpos0, np.float64).copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    lowest = np.inf
    for geom_name in (
        "left_ankle_roll_collision",
        "right_ankle_roll_collision",
    ):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[start : start + count], float)
        rotation = np.asarray(data.geom_xmat[geom], float).reshape(3, 3)
        world = vertices @ rotation.T + np.asarray(data.geom_xpos[geom], float)
        lowest = min(lowest, float(np.min(world[:, 2])))
    qpos[2] -= lowest + 5.0e-4
    yaw_s = np.array([0.0, 0.5, 1.0])
    yaw_qpos = np.tile(qpos, (3, 1))
    yaw_mid_qpos = np.tile(qpos, (2, 1))
    yaw_rate = 0.1
    for row, angle in enumerate(yaw_s * yaw_rate):
        yaw_qpos[row, 3:7] = [
            np.cos(0.5 * angle),
            0.0,
            0.0,
            np.sin(0.5 * angle),
        ]
    for row, angle in enumerate(np.array([0.25, 0.75]) * yaw_rate):
        yaw_mid_qpos[row, 3:7] = [
            np.cos(0.5 * angle),
            0.0,
            0.0,
            np.sin(0.5 * angle),
        ]
    yaw_q_s = np.zeros((3, int(model.nv)))
    yaw_mid_q_s = np.zeros((2, int(model.nv)))
    yaw_q_s[:, 5] = yaw_rate
    yaw_mid_q_s[:, 5] = yaw_rate
    yaw_path = TangentPathGrid(
        path_position=yaw_s,
        qpos=yaw_qpos,
        q_s=yaw_q_s,
        q_ss=np.zeros_like(yaw_q_s),
        midpoint_qpos=yaw_mid_qpos,
        midpoint_q_s=yaw_mid_q_s,
        midpoint_q_ss=np.zeros_like(yaw_mid_q_s),
    )
    derivative_report = solver.validate_path_grid(yaw_path)
    assert derivative_report["status"] == (
        "PASS_SAMPLED_MUJOCO_FINITE_DIFFERENCE"
    )
    bad_mid_q_s = yaw_mid_q_s.copy()
    bad_mid_q_s[:, 3] = 0.2
    with pytest.raises(IncompleteTorqueCertificate, match="q_s/q_ss"):
        solver.validate_path_grid(
            TangentPathGrid(
                path_position=yaw_s,
                qpos=yaw_qpos,
                q_s=yaw_q_s,
                q_ss=np.zeros_like(yaw_q_s),
                midpoint_qpos=yaw_mid_qpos,
                midpoint_q_s=bad_mid_q_s,
                midpoint_q_ss=np.zeros_like(yaw_mid_q_s),
            )
        )
    qvel = np.zeros(int(model.nv))
    qacc = np.zeros(int(model.nv))
    demand = inverse(qpos, qvel, qacc)
    slipping_tangent = np.zeros(int(model.nv))
    slipping_tangent[0] = 1.0
    slipping = solver.solve(
        qpos,
        qvel,
        qacc,
        actuated,
        lower,
        upper,
        np.full(int(model.nv), 100.0),
        expected_generalized_force=demand,
        path_tangent=slipping_tangent,
    )
    assert not slipping.feasible
    assert slipping.report["status"] == "CONTACT_KINEMATICS_INFEASIBLE"
    with pytest.raises(IncompleteTorqueCertificate, match="actuator rows"):
        solver.solve(
            qpos,
            qvel,
            qacc,
            np.arange(31),
            lower,
            upper,
            np.full(int(model.nv), 100.0),
            expected_generalized_force=demand,
            path_tangent=np.zeros(int(model.nv)),
        )
    solution = solver.solve(
        qpos,
        qvel,
        qacc,
        actuated,
        lower,
        upper,
        np.full(int(model.nv), 100.0),
        expected_generalized_force=demand,
        path_tangent=np.zeros(int(model.nv)),
    )
    assert solution.feasible
    assert solution.root_residual < 1e-4
    assert min(solution.report["normal_force_per_foot_n"]) >= 1.0
    assert solution.report["model_binding"] == binding
