"""Dependency-light physical counter-rally truth regressions.

These tests drive the real ``PhysicalBallManager`` trajectory detector with CPU tensors and a
minimal scene double.  They do not call the analytic counter-rally rollout or consume its success
boolean: the evidence under test is the engine-state segment, fitted code-authoritative table
bounce, and interpolated opponent-baseline state.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PB = _load(MDP / "physical_ball.py", "physical_ball_counter_truth_tested")


class _Ball:
    def __init__(self):
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(1, 3),
            root_lin_vel_w=torch.zeros(1, 3),
            root_ang_vel_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.data.root_pos_w[env_ids] = pose[:, :3]

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        self.data.root_lin_vel_w[env_ids] = velocity[:, :3]
        self.data.root_ang_vel_w[env_ids] = velocity[:, 3:]

    def set_external_force_and_torque(self, *_args):
        pass

    def write_data_to_sim(self):
        pass


def _manager():
    ball = _Ball()

    class Scene:
        env_origins = torch.zeros(1, 3)

        def __getitem__(self, key):
            assert key == "pb_ball"
            return ball

    class Sim:
        def __init__(self):
            self.callback = None

        def add_physics_callback(self, _name, callback):
            self.callback = callback

    command = SimpleNamespace(
        device="cpu",
        num_envs=1,
        metrics={},
        cfg=SimpleNamespace(
            vb_table_near_x=0.5,
            vb_table_surface_z=0.76,
            exact_success_decay=1.0,
            physical_ball_impulse=True,
            physical_ball_substep=1,
            question_bank="",
            vb_target_x=2.555,
            vb_target_y=0.0,
        ),
        _action_ball_enabled=False,
    )
    env = SimpleNamespace(scene=Scene(), sim=Sim(), step_dt=0.02)
    manager = PB.PhysicalBallManager(command, env)
    manager.begin_external_exam_attempt(
        torch.tensor([0]),
        torch.tensor([17]),
        require_counter_rally=True,
    )
    manager._truth_served[:] = True
    manager._truth_exact_seen[:] = True
    return manager, ball


def _complete_contact(
    manager,
    *,
    landing_xy=(2.5, 0.0),
    net_crossed=True,
    net_clear=True,
):
    manager._impulse_done[:] = True
    manager._landed[:] = True
    manager._ret_land_xy[:] = torch.tensor([landing_xy])
    manager._net_crossed[:] = bool(net_crossed)
    manager._net_z[:] = (
        manager._net_clear_z + 0.1
        if net_clear
        else manager._net_clear_z - 0.1
    )


def test_opponent_baseline_crossing_interpolates_engine_state_and_fails_closed():
    prev_position = torch.tensor(
        [[3.20, 0.10, 0.90], [3.20, 0.00, 0.90], [3.20, 0.00, 0.90]]
    )
    new_position = torch.tensor(
        [[3.28, 0.18, 0.94], [3.10, 0.00, 0.90], [float("nan"), 0.0, 0.9]]
    )
    prev_velocity = torch.tensor(
        [[2.0, 0.2, 0.4], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    new_velocity = torch.tensor(
        [[3.0, 0.4, 0.8], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    crossed, yz, velocity, fraction = PB.opponent_baseline_crossing(
        prev_position,
        new_position,
        prev_velocity,
        new_velocity,
        3.24,
    )
    assert crossed.tolist() == [True, False, False]
    assert float(fraction[0]) == pytest.approx(0.5)
    assert yz[0].tolist() == pytest.approx([0.14, 0.92])
    assert velocity[0].tolist() == pytest.approx([2.5, 0.3, 0.6])
    assert torch.equal(yz[1:], torch.zeros_like(yz[1:]))
    assert torch.equal(velocity[1:], torch.zeros_like(velocity[1:]))


def test_detected_legal_bounce_then_baseline_publishes_physical_velocity():
    manager, ball = _manager()
    manager._mode[:] = PB._MODE_RETURN
    manager._impulse_done[:] = True
    manager._active_host = 1
    manager._prev_valid[:] = True
    manager._prev_pos_env[:] = torch.tensor([[2.40, 0.00, 0.82]])
    manager._prev_vel_w[:] = torch.tensor([[3.00, 0.00, -1.80]])
    manager._net_crossed[:] = True
    manager._net_z[:] = manager._net_clear_z + 0.10
    ball.data.root_pos_w[:] = torch.tensor([[2.60, 0.02, 0.74]])
    ball.data.root_lin_vel_w[:] = torch.tensor([[2.80, 0.20, -2.00]])
    ball.data.root_ang_vel_w[:] = torch.tensor([[0.0, 8.0, 0.0]])

    manager._detect_bounce_and_landing()
    assert bool(manager._landed[0])
    assert bool(manager._counter_first_opponent_bounce[0])
    assert not bool(manager._counter_baseline_crossed[0])

    # Ordinary single-return truth is already complete at first landing.  Counter-rally truth
    # deliberately remains unavailable until the independent physical baseline segment exists.
    ordinary = manager.cross_engine_physical_truth(
        0, expected_attempt_token=17, final=True
    )
    assert ordinary["available"] is True and ordinary["returned"] is True
    assert "counter_rally" not in ordinary
    pending = manager.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert pending["available"] is False
    assert pending["reason"] == "physical_trajectory_incomplete"

    previous_position = manager._prev_pos_env.clone()
    previous_velocity = manager._prev_vel_w.clone()
    ball.data.root_pos_w[:] = torch.tensor([[3.32, 0.08, 1.02]])
    ball.data.root_lin_vel_w[:] = torch.tensor([[2.40, 0.30, 0.55]])
    expected = PB.opponent_baseline_crossing(
        previous_position,
        ball.data.root_pos_w,
        previous_velocity,
        ball.data.root_lin_vel_w,
        manager._near_x + manager._table_len,
    )
    assert bool(expected[0][0])

    manager._detect_bounce_and_landing()
    manager._publish_cross_engine_truth(torch.tensor([0]))
    truth = manager.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert truth["available"] is True
    assert truth["returned"] is True  # ordinary semantics are unchanged
    counter = truth["counter_rally"]
    assert counter["available"] is True
    assert counter["structural_crossing_complete"] is True
    assert counter["reason"] is None
    assert counter["first_opponent_bounce"] is True
    assert counter["opponent_baseline_crossed"] is True
    assert counter["baseline_cross_env_yz_m"] == pytest.approx(
        expected[1][0].tolist()
    )
    expected_velocity = expected[2][0]
    assert counter["baseline_velocity_world_mps"] == pytest.approx(
        expected_velocity.tolist()
    )
    expected_horizontal_speed = float(
        torch.linalg.norm(expected_velocity[:2])
    )
    assert counter["baseline_horizontal_speed_mps"] == pytest.approx(
        expected_horizontal_speed
    )
    assert counter["baseline_speed_mps"] == pytest.approx(
        float(torch.linalg.norm(expected_velocity))
    )
    assert counter["baseline_horizontal_direction_env_xy"] == pytest.approx(
        (expected_velocity[:2] / expected_horizontal_speed).tolist()
    )


@pytest.mark.parametrize(
    "landing_xy,net_crossed,net_clear,reason",
    [
        ((2.50, 0.00), False, False, "net_not_crossed"),
        ((2.50, 0.00), True, False, "net_not_clear"),
        ((2.50, 0.90), True, True, "first_landing_outside_table"),
        ((1.60, 0.00), True, True, "first_landing_own_half"),
    ],
)
def test_counter_truth_distinguishes_net_and_table_side_failures(
    landing_xy, net_crossed, net_clear, reason
):
    manager, _ = _manager()
    _complete_contact(
        manager,
        landing_xy=landing_xy,
        net_crossed=net_crossed,
        net_clear=net_clear,
    )
    truth = manager.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert truth["available"] is True
    counter = truth["counter_rally"]
    assert counter["available"] is True
    assert counter["structural_crossing_complete"] is False
    assert counter["reason"] == reason


def test_counter_truth_separates_terminal_baseline_miss_from_invalid_physics():
    baseline_miss, _ = _manager()
    _complete_contact(baseline_miss)
    baseline_miss._counter_first_opponent_bounce[:] = True
    baseline_miss._counter_terminal[:] = True
    baseline_miss._counter_second_surface_before_baseline[:] = True
    miss = baseline_miss.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert miss["available"] is True
    assert miss["counter_rally"]["reason"] == (
        "opponent_baseline_not_crossed"
    )
    assert miss["counter_rally"]["second_surface_before_baseline"] is True

    invalid, ball = _manager()
    invalid._mode[:] = PB._MODE_RETURN
    invalid._impulse_done[:] = True
    invalid._active_host = 1
    invalid._prev_valid[:] = True
    invalid._prev_pos_env[:] = torch.tensor([[2.2, 0.0, 1.0]])
    invalid._prev_vel_w[:] = torch.tensor([[2.0, 0.0, -1.0]])
    ball.data.root_pos_w[:] = torch.tensor([[float("nan"), 0.0, 0.9]])
    ball.data.root_lin_vel_w[:] = torch.tensor([[2.0, 0.0, -1.0]])
    invalid._detect_bounce_and_landing()
    assert int(invalid._mode[0]) == PB._MODE_FAILED
    assert bool(invalid._counter_physics_invalid[0])
    failed = invalid.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert failed["available"] is False
    assert failed["reason"] == "physical_simulation_invalid"
    assert failed["counter_rally"]["physics_valid"] is False

    invalid_quaternion, quat_ball = _manager()
    invalid_quaternion._mode[:] = PB._MODE_RETURN
    invalid_quaternion._impulse_done[:] = True
    quat_ball.data.root_pos_w[:] = torch.tensor([[2.2, 0.0, 1.0]])
    quat_ball.data.root_lin_vel_w[:] = torch.tensor([[2.0, 0.0, -1.0]])
    quat_ball.data.root_quat_w[:] = torch.tensor(
        [[float("nan"), 0.0, 0.0, 0.0]]
    )
    invalid_quaternion._on_physics_step(0.005)
    assert int(invalid_quaternion._mode[0]) == PB._MODE_FAILED
    assert bool(invalid_quaternion._counter_physics_invalid[0])
    quat_failed = invalid_quaternion.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert quat_failed["available"] is False
    assert quat_failed["reason"] == "physical_simulation_invalid"


def test_counter_truth_must_be_armed_at_attempt_begin():
    manager, _ = _manager()
    manager._truth_counter_required[:] = False
    manager._counter_required_host = 0
    result = manager.cross_engine_physical_truth(
        0,
        expected_attempt_token=17,
        final=True,
        require_counter_rally=True,
    )
    assert result["available"] is False
    assert result["reason"] == "counter_rally_physical_truth_not_armed"
