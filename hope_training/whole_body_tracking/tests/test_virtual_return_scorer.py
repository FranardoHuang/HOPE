"""Simulator-free parity/regression tests for the Phase-1 hit/return scorer."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "hope_training" / "whole_body_tracking" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mujoco_eval_onnx as mujoco_eval  # noqa: E402
from venue_ball_sampler import VenueStrike  # noqa: E402
from virtual_return_scorer import (  # noqa: E402
    VirtualReturnScorer,
    VirtualReturnSpec,
    flight_accel,
    load_venue_params,
)


def _scorer() -> VirtualReturnScorer:
    return VirtualReturnScorer(
        load_venue_params(str(ROOT / "configs" / "ball_physics_venue.yaml")),
        VirtualReturnSpec(
            table_surface_z=0.76,
            net_x=1.87,
            far_x=3.24,
            half_width=0.7625,
            net_height=0.1525,
        ),
    )


# This vector is frozen from the audited Isaac Torch equations (float64): orient-normal + fitted
# contact, then virtual_ball.coarse_landing(h=.01, n_steps=100, surface_z=.76+R, net_x=1.87).
GOLDEN = dict(
    racket_pos=np.array([1.337724, -0.086193, 1.189590]),
    racket_vel=np.array([1.060350, -0.417343, 1.438771]),
    racket_normal=np.array([1.0, 0.074477, 0.209362]),
    ball_vel=np.array([-1.362673, -0.006527, -0.695381]),
    ball_spin=np.zeros(3),
)


def _strike() -> VenueStrike:
    return VenueStrike(
        clip=0,
        ball_pos_w=GOLDEN["racket_pos"].copy(),
        ball_vel_w=GOLDEN["ball_vel"].copy(),
        ball_spin_w=GOLDEN["ball_spin"].copy(),
        intended_landing_xy=np.array([2.45, -0.06]),
        target_pos_w=GOLDEN["racket_pos"].copy(),
        target_vel_w=GOLDEN["racket_vel"].copy(),
        target_normal_w=GOLDEN["racket_normal"].copy(),
        spec_speed=float(np.linalg.norm(GOLDEN["racket_vel"])),
        spec_iters=0,
        tries=1,
    )


def test_golden_contact_surface_net_and_opponent_outcome():
    """All four score stages match one frozen Isaac-semantics input."""

    outcome = _scorer().score(**GOLDEN, pos_err=0.01)

    assert outcome.contacted
    np.testing.assert_allclose(
        outcome.outgoing_vel,
        [3.01330766, 0.01191966, 1.06675185],
        rtol=0.0,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        outcome.outgoing_spin,
        [-9.32575035, 61.93688568, 22.51066054],
        rtol=0.0,
        atol=5e-9,
    )
    assert outcome.landing_valid
    np.testing.assert_allclose(
        outcome.landing_xy,
        [2.44686384, -0.06118294],
        rtol=0.0,
        atol=5e-9,
    )
    assert abs(outcome.flight_time - 0.3999043515448479) < 1e-12
    assert outcome.net_crossed
    assert abs(outcome.net_z - 1.2048415361232327) < 1e-12
    assert outcome.net_clear
    assert outcome.on_opponent
    assert outcome.landed_ok


def test_contact_gate_keeps_strict_isaac_thresholds():
    scorer = _scorer()
    common = dict(
        ball_vel=[-2.0, 0.0, -0.5],
        ball_spin=[0.0, 0.0, 0.0],
        racket_pos=[0.9, 0.0, 1.0],
        racket_vel=[0.5, 0.0, 1.0],
        racket_normal=[1.0, 0.0, 0.2],
    )
    assert scorer.score(**common, pos_err=0.095 - 1e-9).contacted
    assert not scorer.score(**common, pos_err=0.095).contacted

    common["racket_normal"] = [1.0, 0.0, 0.0]
    common["racket_vel"] = [0.300001, 0.0, 0.0]
    assert scorer.score(**common, pos_err=0.01).contacted
    common["racket_vel"] = [0.3, 0.0, 0.0]
    assert not scorer.score(**common, pos_err=0.01).contacted


@pytest.mark.parametrize(
    ("racket_normal", "expected_net", "expected_opponent"),
    [
        ([1.0, 0.0, 0.0], False, True),   # lands across the net but below net+R
        ([1.0, 0.0, 0.4], True, False),   # clears net but lands beyond the far edge
    ],
)
def test_net_and_opponent_gates_are_independent(racket_normal, expected_net, expected_opponent):
    scorer = _scorer()
    assert scorer.contact_plane_z == 0.78
    outcome = scorer.score(
        ball_vel=[-2.0, 0.0, -0.5],
        ball_spin=[0.0, 0.0, 0.0],
        racket_pos=[0.9, 0.0, 1.0],
        racket_vel=[3.0, 0.0, 1.0],
        racket_normal=racket_normal,
        pos_err=0.01,
    )
    assert outcome.contacted
    assert outcome.net_clear is expected_net
    assert outcome.on_opponent is expected_opponent
    assert not outcome.landed_ok


def _legacy_euler_bare_surface_landing(scorer: VirtualReturnScorer, position, velocity, spin):
    """The removed MuJoCo scorer path: 1 ms Euler, z relative to bare table surface."""

    p = np.asarray(position, np.float64).copy() - np.array([0.0, 0.0, 0.76])
    v = np.asarray(velocity, np.float64).copy()
    omega = np.asarray(spin, np.float64)
    dt = 0.001
    for _ in range(2000):
        accel = flight_accel(v, omega, scorer.params)
        p_new = p + v * dt + 0.5 * accel * dt**2
        v_new = v + accel * dt
        if p[2] > 0.0 and p_new[2] <= 0.0:
            fraction = p[2] / (p[2] - p_new[2])
            return p[:2] + fraction * (p_new[:2] - p[:2])
        p, v = p_new, v_new
    raise AssertionError("legacy reference never crossed its bare z=0 table plane")


def test_regression_old_euler_bare_surface_was_about_16mm_wrong():
    """Lock the bug reproduction: this is large enough to flip edge/net-depth decisions."""

    scorer = _scorer()
    outcome = scorer.score(**GOLDEN, pos_err=0.01)
    old_xy = _legacy_euler_bare_surface_landing(
        scorer, GOLDEN["racket_pos"], outcome.outgoing_vel, outcome.outgoing_spin
    )
    error_m = float(np.linalg.norm(old_xy - outcome.landing_xy))
    assert abs(error_m - 0.01619599888040801) < 2e-8
    assert error_m > 0.01


def test_mujoco_production_path_delegates_to_authoritative_scorer():
    """The evaluator, not the physics-hash-bound sampler, owns both production score calls."""

    sentinel = object()

    class RecordingScorer:
        def score(self, **kwargs):
            self.kwargs = kwargs
            return sentinel

    scorer = RecordingScorer()
    strike = _strike()
    got = mujoco_eval.score_virtual_return(
        scorer,
        strike,
        GOLDEN["racket_pos"],
        GOLDEN["racket_vel"],
        GOLDEN["racket_normal"],
        pos_err=0.01,
    )
    assert got is sentinel
    assert scorer.kwargs["ball_vel"] is strike.ball_vel_w
    assert scorer.kwargs["ball_spin"] is strike.ball_spin_w
    assert scorer.kwargs["racket_pos"] is GOLDEN["racket_pos"]
    assert scorer.kwargs["racket_vel"] is GOLDEN["racket_vel"]
    assert scorer.kwargs["racket_normal"] is GOLDEN["racket_normal"]
    assert scorer.kwargs["pos_err"] == 0.01
    assert scorer.kwargs["intended_landing_xy"] is strike.intended_landing_xy

    production_source = inspect.getsource(mujoco_eval.run_rollout)
    assert production_source.count("score_virtual_return(") == 2
    assert "venue_sampler.score_return" not in production_source


def test_mujoco_production_adapter_has_direct_result_parity():
    """Actual and counterfactual adapters preserve every authoritative score field exactly."""

    strike = _strike()
    scorer = _scorer()
    got = mujoco_eval.score_virtual_return(
        scorer,
        strike,
        GOLDEN["racket_pos"],
        GOLDEN["racket_vel"],
        GOLDEN["racket_normal"],
        pos_err=0.01,
    )
    direct = _scorer().score(
        ball_vel=GOLDEN["ball_vel"],
        ball_spin=GOLDEN["ball_spin"],
        racket_pos=GOLDEN["racket_pos"],
        racket_vel=GOLDEN["racket_vel"],
        racket_normal=GOLDEN["racket_normal"],
        pos_err=0.01,
        intended_landing_xy=strike.intended_landing_xy,
    )
    assert got.contacted == direct.contacted
    assert got.landing_valid == direct.landing_valid
    assert got.net_crossed == direct.net_crossed
    assert got.net_clear == direct.net_clear
    assert got.on_opponent == direct.on_opponent
    assert got.landed_ok == direct.landed_ok
    np.testing.assert_array_equal(got.landing_xy, direct.landing_xy)
    np.testing.assert_array_equal(got.outgoing_vel, direct.outgoing_vel)
    np.testing.assert_array_equal(got.outgoing_spin, direct.outgoing_spin)
    assert got.approach_speed == direct.approach_speed
    assert got.flight_time == direct.flight_time
    assert got.net_z == direct.net_z
    assert got.land_err == direct.land_err


def test_production_scorer_contract_binds_source_config_and_score_spec():
    sampler_geometry = SimpleNamespace(
        table_surface_z=0.76,
        net_x=1.87,
        far_x=3.24,
        half_w=0.7625,
        table=SimpleNamespace(net_height=0.1525),
    )
    scorer, contract = mujoco_eval.build_virtual_return_scorer(ROOT, sampler_geometry)

    body = dict(contract)
    recorded_sha = body.pop("sha256")
    assert mujoco_eval.canonical_contract_sha256(body) == recorded_sha
    assert contract["source"]["repo_relative_path"].endswith("virtual_return_scorer.py")
    assert contract["physics_config"]["repo_relative_path"] == "configs/ball_physics_venue.yaml"
    assert contract["physics_config"]["sha256"] == mujoco_eval.sha256_file(
        ROOT / "configs" / "ball_physics_venue.yaml"
    )
    assert contract["score_spec"]["rollout_h"] == 0.01
    assert contract["score_spec"]["rollout_steps"] == 100
    assert scorer.contact_plane_z == 0.78

    robot = SimpleNamespace(
        model=SimpleNamespace(
            dof_damping=np.zeros(1),
            dof_frictionloss=np.zeros(1),
            dof_armature=np.zeros(1),
            opt=SimpleNamespace(integrator=0),
        ),
        vadr=np.array([0]),
        ctrl_lo=np.array([-1.0]),
        ctrl_hi=np.array([1.0]),
        soft_jnt_lo=np.array([-1.0]),
        soft_jnt_hi=np.array([1.0]),
    )
    policy = SimpleNamespace(
        obs_dim=1,
        joint_names=("joint",),
        default_q=np.zeros(1),
        action_scale=np.ones(1),
        kp=np.ones(1),
        kd=np.ones(1),
    )
    execution = mujoco_eval.build_evaluation_execution_contract(
        robot=robot,
        policy=policy,
        mjcf_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        ready_state_contract={"mode": "test", "sha256": "c" * 64},
        sim_dt=0.005,
        decimation=4,
        pd_mode="implicit",
        passive_damping_mode="zero",
        frictionloss_mode="zero",
        qdes_clamp=True,
        one_question_reset=True,
        virtual_return_scorer_contract=contract,
    )
    assert execution["virtual_return_scorer_contract"] == contract
    assert execution["sha256"] != recorded_sha


def test_audited_training_callsite_still_uses_ball_center_plane():
    """No-Isaac drift guard for the read-only training implementation audited by this change."""

    mdp = ROOT / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking" / \
        "whole_body_tracking" / "tasks" / "tracking" / "mdp"
    virtual_ball = (mdp / "virtual_ball.py").read_text(encoding="utf-8")
    hope_commands = (mdp / "hope_commands.py").read_text(encoding="utf-8")
    assert "_EPS = 1e-12" in virtual_ball
    assert "surface_z=float(self.cfg.vb_table_surface_z) + prm.ball_radius" in hope_commands
    assert 'land["net_z"] > self._vb_net_top_z + self._vb_ball_r' in hope_commands
    assert "(lx > self._vb_net_x) & (lx <= self._vb_far_x)" in hope_commands


def test_direct_torch_parity_when_torch_is_available():
    """Run the same golden row through the actual Isaac module in an Isaac-capable venv."""

    torch = pytest.importorskip("torch")
    module_path = ROOT / "hope_training" / "whole_body_tracking" / "source" / \
        "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp" / \
        "virtual_ball.py"
    module_spec = importlib.util.spec_from_file_location("scorer_parity_virtual_ball", module_path)
    virtual_ball = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = virtual_ball
    module_spec.loader.exec_module(virtual_ball)
    params = virtual_ball.load_venue_params(str(ROOT / "configs" / "ball_physics_venue.yaml"))

    def tensor(name):
        return torch.as_tensor(GOLDEN[name], dtype=torch.float64).unsqueeze(0)

    v_plus, spin_plus = virtual_ball.predict_paddle_contact(
        tensor("ball_vel"), tensor("racket_vel"), tensor("racket_normal"),
        tensor("ball_spin"), params,
    )
    landing = virtual_ball.coarse_landing(
        tensor("racket_pos"), v_plus, spin_plus, params,
        surface_z=0.76 + params.ball_radius, net_x=1.87, h=0.01, n_steps=100,
    )
    scorer_out = _scorer().score(**GOLDEN, pos_err=0.01)
    np.testing.assert_allclose(v_plus[0].numpy(), scorer_out.outgoing_vel, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(spin_plus[0].numpy(), scorer_out.outgoing_spin, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(landing["land_xy"][0].numpy(), scorer_out.landing_xy,
                               rtol=0.0, atol=1e-12)
    assert bool(landing["land_valid"][0]) == scorer_out.landing_valid
    assert bool(landing["net_valid"][0]) == scorer_out.net_crossed
    assert abs(float(landing["net_z"][0]) - scorer_out.net_z) < 1e-12
    torch_on_opponent = bool(
        landing["land_valid"][0]
        and landing["land_xy"][0, 0] > 1.87
        and landing["land_xy"][0, 0] <= 3.24
        and landing["land_xy"][0, 1].abs() <= 0.7625
    )
    torch_net_clear = bool(landing["net_valid"][0] and landing["net_z"][0] > 0.9325)
    assert torch_on_opponent == scorer_out.on_opponent
    assert torch_net_clear == scorer_out.net_clear
