"""Dependency-light Phase-B contract, contact, publication, and flag-off regressions.

No Isaac import is performed.  The final test is a marked/skip-gated verifier for a scorecard
produced later by the documented clean-detached Isaac run; it makes no runtime claim by itself.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MDP = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


A = _load(ROOT / "scripts/isaac_bank_exam_adapter.py", "phase_b_adapter_tested")
PB = _load(MDP / "physical_ball.py", "phase_b_physical_ball_tested")
VB = _load(MDP / "virtual_ball.py", "phase_b_virtual_ball_tested")


def test_racket_impulse_delegates_bit_exactly_and_disc_scan_catches_tunneling():
    torch.manual_seed(7)
    count = 64
    v_ball = (torch.rand(count, 3) - 0.5) * 10.0
    v_blade = (torch.rand(count, 3) - 0.5) * 6.0
    normal = torch.nn.functional.normalize(torch.randn(count, 3), dim=-1)
    spin = (torch.rand(count, 3) - 0.5) * 100.0
    params = VB.load_venue_params()
    got = PB.racket_impulse(v_ball, v_blade, normal, spin, params)
    expected = VB.predict_paddle_contact(v_ball, v_blade, normal, spin, params)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])

    blade_pos = torch.tensor([[0.95, 0.0, 1.0]])
    blade_vel = torch.tensor([[2.0, 0.0, 0.0]])
    face = torch.tensor([[1.0, 0.0, 0.0]])
    hit, _ = PB.blade_disc_contact(
        torch.tensor([[0.96, 0.0, 1.0]]),
        torch.tensor([[-3.0, 0.0, 0.0]]),
        blade_pos,
        blade_vel,
        face,
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
    )
    assert hit.tolist() == [True]
    tunnel, signed = PB.blade_disc_contact(
        torch.tensor([[0.89, 0.0, 1.0]]),
        torch.tensor([[5.0, 0.0, 0.0]]),
        blade_pos,
        blade_vel,
        face,
        torch.tensor([0.06]),
        torch.ones(1, dtype=torch.bool),
    )
    assert tunnel.tolist() == [True] and float(signed[0]) < 0.0
    miss, _ = PB.blade_disc_contact(
        torch.tensor([[0.89, 0.10, 1.0]]),
        torch.tensor([[5.0, 0.0, 0.0]]),
        blade_pos,
        blade_vel,
        face,
        torch.tensor([0.06]),
        torch.ones(1, dtype=torch.bool),
    )
    assert miss.tolist() == [False]


def test_action_ball_selected_face_contact_is_one_sided_and_radius_aware():
    face_center = torch.zeros(1, 3)
    face_velocity = torch.zeros(1, 3)
    selected_normal = torch.tensor([[1.0, 0.0, 0.0]])
    previous_face_center = face_center.clone()
    previous_normal = selected_normal.clone()

    tangent_hit, tangent_gap = PB.selected_face_disc_contact(
        torch.tensor([[0.02, 0.0, 0.0]]),
        torch.tensor([[-3.0, 0.0, 0.0]]),
        face_center,
        face_velocity,
        selected_normal,
        torch.tensor([[0.03, 0.0, 0.0]]),
        previous_face_center,
        previous_normal,
        torch.zeros(1, dtype=torch.bool),
        ball_radius=0.02,
    )
    assert tangent_hit.tolist() == [True]
    assert torch.allclose(tangent_gap, torch.zeros(1))

    opposite_face, _ = PB.selected_face_disc_contact(
        torch.tensor([[-0.02, 0.0, 0.0]]),
        torch.tensor([[3.0, 0.0, 0.0]]),
        face_center,
        face_velocity,
        selected_normal,
        torch.tensor([[-0.03, 0.0, 0.0]]),
        previous_face_center,
        previous_normal,
        torch.zeros(1, dtype=torch.bool),
        ball_radius=0.02,
    )
    assert opposite_face.tolist() == [False]

    tunneled, gap = PB.selected_face_disc_contact(
        torch.tensor([[-0.01, 0.0, 0.0]]),
        torch.tensor([[-8.0, 0.0, 0.0]]),
        face_center,
        face_velocity,
        selected_normal,
        torch.tensor([[0.06, 0.0, 0.0]]),
        previous_face_center,
        previous_normal,
        torch.ones(1, dtype=torch.bool),
        ball_radius=0.02,
    )
    assert tunneled.tolist() == [True]
    assert float(gap[0]) < 0.0

    reverse_crossing, _ = PB.selected_face_disc_contact(
        torch.tensor([[0.03, 0.0, 0.0]]),
        torch.tensor([[8.0, 0.0, 0.0]]),
        face_center,
        face_velocity,
        selected_normal,
        torch.tensor([[0.01, 0.0, 0.0]]),
        previous_face_center,
        previous_normal,
        torch.ones(1, dtype=torch.bool),
        ball_radius=0.02,
    )
    assert reverse_crossing.tolist() == [False]


@pytest.mark.parametrize("face_sign", [1.0, -1.0])
def test_action_ball_selected_face_swept_crossing_handles_fast_both_faces(
    face_sign,
):
    normal = torch.tensor([[0.0, face_sign, 0.0]])
    face = torch.zeros(1, 3)
    previous_ball = torch.tensor([[0.0, 0.03 * face_sign, 0.0]])
    current_ball = torch.tensor([[0.0, -0.02 * face_sign, 0.0]])

    hit, gap = PB.selected_face_disc_contact(
        current_ball,
        torch.tensor([[0.0, -10.0 * face_sign, 0.0]]),
        face,
        torch.zeros(1, 3),
        normal,
        previous_ball,
        face,
        normal,
        torch.ones(1, dtype=torch.bool),
        ball_radius=0.02,
        pad=0.003,
    )
    assert hit.tolist() == [True]
    assert float(gap[0]) < -0.03

    reverse, _ = PB.selected_face_disc_contact(
        previous_ball,
        torch.tensor([[0.0, 10.0 * face_sign, 0.0]]),
        face,
        torch.zeros(1, 3),
        normal,
        current_ball,
        face,
        normal,
        torch.ones(1, dtype=torch.bool),
        ball_radius=0.02,
        pad=0.003,
    )
    assert reverse.tolist() == [False]


def test_action_ball_selected_face_swept_radius_uses_crossing_not_endpoint():
    normal = torch.tensor([[0.0, 1.0, 0.0]])
    face = torch.zeros(1, 3)
    velocity = torch.tensor([[0.0, -10.0, 0.0]])
    valid = torch.ones(1, dtype=torch.bool)

    # The current endpoint is outside the 7.5 cm disc, but the segment
    # intersects the selected-face tangent plane at x=4 cm.
    inside_at_crossing, _ = PB.selected_face_disc_contact(
        torch.tensor([[0.08, 0.01, 0.0]]),
        velocity,
        face,
        torch.zeros(1, 3),
        normal,
        torch.tensor([[0.00, 0.03, 0.0]]),
        face,
        normal,
        valid,
        racket_radius=0.075,
        ball_radius=0.02,
    )
    assert inside_at_crossing.tolist() == [True]

    # The endpoint has moved inside, but the actual tangent-plane crossing
    # occurred at x=11.5 cm and must not create a false hit.
    outside_at_crossing, _ = PB.selected_face_disc_contact(
        torch.tensor([[0.07, 0.01, 0.0]]),
        velocity,
        face,
        torch.zeros(1, 3),
        normal,
        torch.tensor([[0.16, 0.03, 0.0]]),
        face,
        normal,
        valid,
        racket_radius=0.075,
        ball_radius=0.02,
    )
    assert outside_at_crossing.tolist() == [False]


def _truth_manager(*, callback: bool = True):
    manager = object.__new__(PB.PhysicalBallManager)
    manager._cmd = SimpleNamespace(num_envs=1)
    manager.device = "cpu"
    manager._impulse_on = True
    manager._cb_active = callback
    manager._substep = 1
    manager._prm = SimpleNamespace(ball_radius=0.02)
    manager._near_x = 0.5
    manager._table_len = 2.74
    manager._half_w = 1.525 / 2.0
    manager._net_x_env = 1.87
    manager._net_clear_z = 0.9325
    manager._truth_started = torch.ones(1, dtype=torch.bool)
    manager._truth_exam_active = torch.ones(1, dtype=torch.bool)
    manager._truth_attempt_token = torch.tensor([7], dtype=torch.long)
    manager._truth_served = torch.ones(1, dtype=torch.bool)
    manager._truth_exact_seen = torch.ones(1, dtype=torch.bool)
    manager._truth_published = torch.zeros(1, dtype=torch.bool)
    manager._truth_published_served = torch.zeros(1, dtype=torch.bool)
    manager._truth_published_exact_seen = torch.zeros(1, dtype=torch.bool)
    manager._truth_counter_required = torch.zeros(1, dtype=torch.bool)
    manager._truth_published_counter_required = torch.zeros(
        1, dtype=torch.bool
    )
    manager._truth_available = torch.zeros(1, dtype=torch.bool)
    manager._truth_contacted = torch.zeros(1, dtype=torch.bool)
    manager._truth_net_clear = torch.zeros(1, dtype=torch.bool)
    manager._truth_landed_ok = torch.zeros(1, dtype=torch.bool)
    manager._truth_returned = torch.zeros(1, dtype=torch.bool)
    manager._truth_landing_xy = torch.zeros(1, 2)
    manager._truth_landed = torch.zeros(1, dtype=torch.bool)
    manager._truth_net_crossed = torch.zeros(1, dtype=torch.bool)
    manager._truth_counter_first_opponent_bounce = torch.zeros(
        1, dtype=torch.bool
    )
    manager._truth_counter_baseline_crossed = torch.zeros(
        1, dtype=torch.bool
    )
    manager._truth_counter_baseline_yz = torch.zeros(1, 2)
    manager._truth_counter_baseline_velocity_w = torch.zeros(1, 3)
    manager._truth_counter_terminal = torch.zeros(1, dtype=torch.bool)
    manager._truth_counter_physics_invalid = torch.zeros(
        1, dtype=torch.bool
    )
    manager._truth_counter_second_surface_before_baseline = torch.zeros(
        1, dtype=torch.bool
    )
    manager._impulse_done = torch.ones(1, dtype=torch.bool)
    manager._landed = torch.ones(1, dtype=torch.bool)
    manager._net_crossed = torch.ones(1, dtype=torch.bool)
    manager._net_z = torch.tensor([1.05])
    manager._ret_land_xy = torch.tensor([[2.55, 0.03]])
    manager._hit_new = torch.zeros(1, dtype=torch.bool)
    manager._ret_land_new = torch.zeros(1, dtype=torch.bool)
    manager._ret_bounce_new = torch.zeros(1, dtype=torch.bool)
    manager._pred_valid = torch.zeros(1, dtype=torch.bool)
    manager._prev_dn_valid = torch.zeros(1, dtype=torch.bool)
    manager._prev_vel_w = torch.zeros(1, 3)
    manager._counter_first_opponent_bounce = torch.zeros(
        1, dtype=torch.bool
    )
    manager._counter_baseline_crossed = torch.zeros(1, dtype=torch.bool)
    manager._counter_baseline_yz = torch.zeros(1, 2)
    manager._counter_baseline_velocity_w = torch.zeros(1, 3)
    manager._counter_terminal = torch.zeros(1, dtype=torch.bool)
    manager._counter_physics_invalid = torch.zeros(1, dtype=torch.bool)
    manager._counter_second_surface_before_baseline = torch.zeros(
        1, dtype=torch.bool
    )
    manager._mode = torch.tensor([PB._MODE_PARKED], dtype=torch.long)
    manager._counter_required_host = 0
    return manager


def test_physical_truth_publication_is_explicit_and_callback_failure_is_nonformal():
    manager = _truth_manager()
    assert manager.cross_engine_truth_metadata == {
        "available": True,
        "capability": A.PHASE_B_FULL_CAPABILITY,
        "physics_callback_active": True,
        "racket_impulse_enabled": True,
        "aero_substep": 1,
        "contact_authority": "code_driven_blade_disc_and_venue_paddle_impulse",
        "post_contact_rollout": "physx_gravity_plus_deterministic_venue_aero_and_code_table_bounce",
        "collision_authority": "code_only_ball_collider_disabled",
        "racket_contact_radius_m": 0.075,
        "ball_radius_m": 0.02,
        "reason": "full physics-substep Phase-B truth instrument active",
    }
    manager._publish_cross_engine_truth(torch.tensor([0]))
    truth = manager.cross_engine_physical_truth(0, expected_attempt_token=7, final=True)
    assert truth["available"] is True
    assert truth["contacted"] is True
    assert truth["net_clear"] is True
    assert truth["landed_ok"] is True
    assert truth["returned"] is True
    assert truth["landing_xy_env_m"] == pytest.approx([2.55, 0.03])
    # A repeated empty resample before evaluator control returns must not overwrite the held hit.
    manager._impulse_done.zero_()
    manager._landed.zero_()
    manager._net_crossed.zero_()
    manager._ret_land_xy.zero_()
    manager._publish_cross_engine_truth(torch.tensor([0]))
    assert manager.cross_engine_physical_truth(
        0, expected_attempt_token=7, final=True
    )["returned"] is True

    degraded = _truth_manager(callback=False)
    metadata = degraded.cross_engine_truth_metadata
    assert metadata["available"] is False
    assert metadata["capability"] == "phase_b_degraded_control_rate_no_physics_callback"
    assert degraded.cross_engine_physical_truth(
        0, expected_attempt_token=7, final=True
    )["available"] is False

    pre_attempt = _truth_manager()
    pre_attempt._truth_exam_active[:] = False
    pre_attempt._truth_published[:] = False
    pre_attempt._publish_cross_engine_truth(torch.tensor([0]))
    assert not bool(pre_attempt._truth_published[0])

    # The evaluator-owned begin seam clears reset-time/old-generation truth and binds a token.
    generation = _truth_manager()
    generation._truth_exam_active[:] = False
    generation._truth_published[:] = True
    generation._truth_attempt_token[:] = 3
    generation.begin_external_exam_attempt(torch.tensor([0]), torch.tensor([11]))
    assert not bool(generation._truth_published[0])
    assert int(generation._truth_attempt_token[0]) == 11
    not_served = generation.cross_engine_physical_truth(
        0, expected_attempt_token=11, final=True
    )
    assert not_served["available"] is False and "never served" in not_served["reason"]
    generation._truth_served[:] = True
    pre_exact = generation.cross_engine_physical_truth(
        0, expected_attempt_token=11, final=True
    )
    assert pre_exact["available"] is False and "before its exact" in pre_exact["reason"]
    generation._truth_exact_seen[:] = True
    policy_miss = generation.cross_engine_physical_truth(
        0, expected_attempt_token=11, final=True
    )
    assert policy_miss["available"] is True and policy_miss["contacted"] is False
    mismatch = generation.cross_engine_physical_truth(
        0, expected_attempt_token=12, final=True
    )
    assert mismatch["available"] is False and "generation mismatch" in mismatch["reason"]


def _question_document(physical):
    return A.strike_state_instrumentation_document(
        observation_phase="exact_strike",
        base_root_state_env=np.zeros(13),
        racket_pos_env=np.zeros(3),
        racket_lin_vel_world=np.zeros(3),
        racket_face_normal_signed_pre_orient_world=np.array([1.0, 0.0, 0.0]),
        racket_face_normal_raw_plus_y_world=np.array([1.0, 0.0, 0.0]),
        analytic_face_normal_oriented_world=np.array([1.0, 0.0, 0.0]),
        target_racket_pos_env=np.zeros(3),
        target_racket_lin_vel_world=np.zeros(3),
        target_face_normal_world=np.array([1.0, 0.0, 0.0]),
        incoming_ball_lin_vel_world=np.array([-3.0, 0.0, 0.0]),
        incoming_ball_spin_world=np.zeros(3),
        analytic_available=True,
        analytic_capture_gate=True,
        analytic_net_clear=True,
        analytic_on_opponent=True,
        analytic_landing_valid=True,
        analytic_landing_xy_env=np.array([2.5, 0.0]),
        physical_truth=physical,
    )


def test_scorecard_truth_replacement_changes_only_truth_and_rebinds_hash():
    pending = {
        "available": False,
        "capability": A.PHASE_B_FULL_CAPABILITY,
        "reason": "pending",
    }
    before = _question_document(pending)
    truth = {
        "available": True,
        "capability": A.PHASE_B_FULL_CAPABILITY,
        "contacted": False,
        "net_clear": False,
        "landed_ok": False,
        "returned": False,
        "landing_xy_env_m": None,
        "attempt_token": 0,
        "served": True,
        "exact_seen": True,
        "contact_authority": "code_driven_blade_disc_and_venue_paddle_impulse",
        "post_contact_rollout": "physx_gravity_plus_deterministic_venue_aero_and_code_table_bounce",
    }
    after = A.replace_instrumentation_physical_truth(before, truth)
    assert after["physical_truth"] == truth
    assert after["sha256"] != before["sha256"]
    before_without = dict(before)
    after_without = dict(after)
    before_without.pop("physical_truth")
    after_without.pop("physical_truth")
    before_without.pop("sha256")
    after_without.pop("sha256")
    assert after_without == before_without
    bad = dict(truth, returned=True)
    with pytest.raises(A.IsaacBankExamError, match="returned must equal"):
        A.replace_instrumentation_physical_truth(before, bad)


def test_contract_hash_source_binding_and_profile_are_fail_closed(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    target = {
        "checkpoint_sha256": "1" * 64,
        "training_contract_sha256": "2" * 64,
        "exam_bank_sha256": "3" * 64,
        "schedule_file_sha256": "5" * 64,
        "schedule_sha256": "4" * 64,
        "question_id_order_sha256": "6" * 64,
        "schedule_k": 100,
        "per_clip_quota": 50,
        "schedule_seed": 0,
        "noise_scale": 0.0,
        "fresh_lineage": True,
        "evaluation_contract_exact": True,
        "run_name": "unit_run",
        "checkpoint_iteration": 2000,
        "attempts_per_side": 50,
        "plant_cell": "SZ_zero_friction_protocol_exact",
    }
    contract = {
        "schema": A.PHASE_B_CONTRACT_SCHEMA,
        "schema_version": 1,
        "contract_id": "unit-phase-b",
        "auto_start": False,
        "runtime_validation_required": True,
        "threshold_changes_allowed": False,
        "legacy_virtual_score_changes_allowed": False,
        "real_robot_authorized": False,
        "phase_b_profile": {
            "physical_ball": True,
            "physical_ball_impulse": True,
            "physical_ball_substep": 1,
            "virtual_ball": True,
            "event_timing_mode": "disabled",
            "required_capability": A.PHASE_B_FULL_CAPABILITY,
            "collision_authority": "code_only_ball_collider_disabled",
            "physics_callback_required": True,
            "contact_authority": "code_driven_blade_disc_and_venue_paddle_impulse",
            "racket_contact_radius_m": 0.075,
            "ball_radius_m": 0.02,
            "impulse": "virtual_ball.predict_paddle_contact_bit_exact_delegation",
            "post_contact_rollout": "physx_gravity_plus_deterministic_venue_aero_and_code_table_bounce",
        },
        "target": target,
        "sources": {
            "unit": {"path": "source.py", "sha256": A.sha256_file(source)},
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    loaded = A.load_and_validate_phase_b_contract(
        path, expected_sha256=A.sha256_file(path), repository_root=tmp_path
    )
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(
                virtual_ball=False,
                physical_ball=False,
                vb_capture_radius=0.095,
                vb_min_approach_speed=0.3,
            )
        ),
        physical_ball=False,
    )
    profile = A.apply_phase_b_eval_profile(cfg, loaded)
    racket = cfg.commands.racket_target
    assert racket.virtual_ball is True and racket.physical_ball is True
    assert racket.physical_ball_impulse is True and racket.physical_ball_substep == 1
    assert racket.vb_capture_radius == 0.095 and racket.vb_min_approach_speed == 0.3
    assert profile["legacy_virtual_thresholds_changed"] is False
    A.validate_phase_b_runtime_target(
        loaded,
        checkpoint_sha256="1" * 64,
        training_contract_sha256="2" * 64,
        exam_bank_sha256="3" * 64,
        schedule_file_sha256="5" * 64,
        schedule_sha256="4" * 64,
        question_id_order_sha256="6" * 64,
        schedule_k=100,
        per_clip_quota=50,
        attempts_per_side=50,
        schedule_seed=0,
        noise_scale=0.0,
        fresh_lineage=True,
        evaluation_contract_exact=True,
    )
    with pytest.raises(A.IsaacBankExamError, match="frozen paper"):
        A.validate_phase_b_runtime_target(
            loaded,
            checkpoint_sha256="f" * 64,
            training_contract_sha256="2" * 64,
            exam_bank_sha256="3" * 64,
            schedule_file_sha256="5" * 64,
            schedule_sha256="4" * 64,
            question_id_order_sha256="6" * 64,
            schedule_k=100,
            per_clip_quota=50,
            attempts_per_side=50,
            schedule_seed=0,
            noise_scale=0.0,
            fresh_lineage=True,
            evaluation_contract_exact=True,
        )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(A.IsaacBankExamError, match="source hash mismatch"):
        A.load_and_validate_phase_b_contract(
            path, expected_sha256=A.sha256_file(path), repository_root=tmp_path
        )


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


def test_phase_b_disabled_keeps_phase_a_manager_lane_inert():
    ball = _Ball()

    class Scene:
        env_origins = torch.zeros(1, 3)

        def __getitem__(self, key):
            assert key == "pb_ball"
            return ball

    class Sim:
        def add_physics_callback(self, _name, _callback):
            raise RuntimeError("dependency-light harness")

    normal = torch.tensor([[1.0, 0.0, 0.0]])
    command = SimpleNamespace(
        device="cpu",
        num_envs=1,
        metrics={},
        cfg=SimpleNamespace(
            vb_table_near_x=0.5,
            vb_table_surface_z=0.76,
            exact_success_decay=1.0,
            physical_ball_impulse=False,
            physical_ball_substep=1,
            question_bank="",
            vb_target_x=2.555,
            vb_target_y=0.0,
        ),
        time_to_strike=torch.tensor([0.3]),
        racket_target_pos_w=torch.tensor([[0.42, -0.15, 2.5]]),
        vb_vel_in_w=torch.tensor([[-3.5, 0.0, -0.6]]),
        vb_spin_in_w=torch.zeros(1, 3),
        racket_pos_w=torch.tensor([[0.95, 0.0, 1.0]]),
        racket_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        racket_lin_vel_w=torch.zeros(1, 3),
        racket_normal_raw_w=normal,
        racket_normal_w=normal,
        vb_fired=torch.zeros(1, dtype=torch.bool),
        vb_landing_xy=torch.zeros(1, 2),
        vb_landing_valid=torch.zeros(1, dtype=torch.bool),
    )
    command._racket_fk = lambda: (
        command.racket_pos_w.clone(),
        command.racket_quat_w.clone(),
        command.racket_lin_vel_w.clone(),
        command.racket_normal_raw_w.clone(),
        command.racket_normal_w.clone(),
    )
    manager = PB.PhysicalBallManager(command, SimpleNamespace(scene=Scene(), sim=Sim(), step_dt=0.02))
    assert "pb_hit_count" not in command.metrics
    manager.on_resample(torch.tensor([0]))
    manager.update(torch.zeros(1, dtype=torch.bool))
    ball.data.root_pos_w[0] = torch.tensor([0.96, 0.0, 1.0])
    ball.data.root_lin_vel_w[0] = torch.tensor([-3.0, 0.0, -0.5])
    before = ball.data.root_lin_vel_w.clone()
    command.time_to_strike[:] = 0.28
    manager.update(torch.zeros(1, dtype=torch.bool))
    assert torch.equal(ball.data.root_lin_vel_w, before)
    assert manager.cross_engine_truth_capability == "incoming_flight_only_no_paddle_contact_phase_a"


@pytest.mark.skipif(
    not os.environ.get("HOPE_PHASE_B_ISAAC_SCORECARD"),
    reason="set HOPE_PHASE_B_ISAAC_SCORECARD after the documented clean-detached Isaac run",
)
def test_simulator_dependent_phase_b_scorecard_has_complete_physical_truth():
    """Marked simulator-dependent acceptance check; skipped until an artifact is provided."""

    scorecard = json.loads(Path(os.environ["HOPE_PHASE_B_ISAAC_SCORECARD"]).read_text())
    binding = scorecard["physical_truth_phase_b_contract"]
    assert binding["runtime_validated"] is True
    attempts = scorecard["attempts"]
    assert len(attempts) == 100
    assert all(not row["censored"] for row in attempts)
    for row in attempts:
        truth = row["instrumentation"]["physical_truth"]
        assert truth["available"] is True
        assert truth["capability"] == A.PHASE_B_FULL_CAPABILITY
        assert all(isinstance(truth[key], bool) for key in (
            "contacted", "net_clear", "landed_ok", "returned", "served", "exact_seen"
        ))
        assert truth["served"] is True and truth["exact_seen"] is True
        assert truth["attempt_token"] == row["schedule_index"]
