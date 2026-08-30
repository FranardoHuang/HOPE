"""Focused construction tests for the opt-in frozen-teacher diagnostic."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types

import numpy as np
import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_full_mdp_teacher_replay as replay
import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env
import mujoco_gpu_ac_full_mdp_wait_rsl3 as runner
import mujoco_gpu_ac_table_keepout as keepout


def test_teacher_replay_wait_bridge_and_affine_decoder_are_exact():
    hold = torch.tensor([[1.0, 2.0]])
    frame0 = torch.tensor([[5.0, 10.0]])
    previous = hold.clone()
    commands = []
    for valid, frozen in ((False, 4), (True, 3), (True, 2), (True, 1), (True, 0)):
        previous = replay.frozen_teacher_qdes(
            torch=torch,
            task_valid=torch.tensor([valid]),
            hold_qdes=hold,
            previous_qdes=previous,
            teacher_qdes=frame0,
            frozen_steps=torch.tensor([frozen]),
            bridge=wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge,
        )
        commands.append(previous.clone())
    assert torch.equal(commands[0], hold)
    assert torch.equal(commands[-1], frame0)
    offset = torch.tensor([0.5, -1.0])
    scale = torch.tensor([0.5, 2.0])
    action = replay.decode_teacher_qdes_to_action(
        torch=torch, qdes=frame0, action_offset=offset, action_scale=scale
    )
    torch.testing.assert_close(offset + scale * action[0], frame0[0])


def test_direct_frame0_mode_holds_then_selects_one_handoff_and_natural_teacher():
    valid = torch.tensor([True, True])
    teacher_frame = torch.tensor([0, 0], dtype=torch.long)
    frozen = torch.tensor([1, 0], dtype=torch.long)
    applied = torch.tensor([False, False])
    due = replay.direct_frame0_handoff_due(
        torch=torch,
        task_valid=valid,
        teacher_frame=teacher_frame,
        frozen_steps=frozen,
        already_applied=applied,
    )
    assert due.tolist() == [False, True]
    hold = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    teacher = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    requested = replay.direct_frame0_teacher_qdes(
        torch=torch,
        task_valid=valid,
        hold_qdes=hold,
        teacher_qdes=teacher,
        frozen_steps=frozen,
    )
    assert requested.tolist() == [[1.0, 2.0], [7.0, 8.0]]
    applied[1] = True
    assert not bool(replay.direct_frame0_handoff_due(
        torch=torch,
        task_valid=valid,
        teacher_frame=teacher_frame,
        frozen_steps=frozen,
        already_applied=applied,
    ).any())


def _direct_frame0_install_rig():
    env = object.__new__(wait_env.FullMdpInitialWaitVecEnv)
    env._torch = torch
    env.device = torch.device("cpu")
    env.full_a_mode = True
    env.num_envs = 1
    env.num_actions = 31
    env.step_dt = 0.02
    env.root_qadr = 0
    env.root_vadr = 0
    env._q_slice = slice(7, 38)
    env._v_slice = slice(6, 37)
    env.q_adr_act = torch.arange(7, 38, dtype=torch.long)
    env.v_adr_act = torch.arange(6, 37, dtype=torch.long)
    env.b_q = 38
    env.b_v = 37
    qpos = torch.zeros((1, 45), dtype=torch.float32)
    qvel = torch.zeros((1, 43), dtype=torch.float32)
    qpos[:, 2] = 1.1
    qpos[:, 3] = 1.0
    qpos[:, 38:45] = torch.tensor(
        [[4.0, 5.0, 5.9979399, 1.0, 0.0, 0.0, 0.0]]
    )
    # This is the real returned actor boundary after a waiting row was parked
    # and then advanced by 20 x 1 ms semi-implicit Euler gravity substeps.
    qvel[:, 39] = -0.1962
    data = types.SimpleNamespace(
        qpos=qpos,
        qvel=qvel,
        qacc_warmstart=torch.cat((
            torch.ones((1, 37), dtype=torch.float32),
            torch.full((1, 6), 3.0, dtype=torch.float32),
        ), dim=1),
        ctrl=torch.ones((1, 31), dtype=torch.float32),
        act=torch.empty((1, 0), dtype=torch.float32),
    )
    forward_calls = []
    env.sim = types.SimpleNamespace(
        data=data, forward=lambda: forward_calls.append("forward")
    )
    env.mj_model = types.SimpleNamespace(na=0)
    env.env = types.SimpleNamespace(
        scene=types.SimpleNamespace(env_origins=torch.tensor([[0.4, -0.2, 0.0]]))
    )
    env._full_a_park_position_scene = torch.tensor([3.6, 5.2, 6.0])
    env._full_a_park_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    body_pos = torch.zeros((2, len(wait_env.FULLMDP_TRACKED_BODY_NAMES), 3))
    body_pos[0, 0] = torch.tensor([0.1, 0.2, 0.9])
    body_quat = torch.zeros((2, len(wait_env.FULLMDP_TRACKED_BODY_NAMES), 4))
    body_quat[..., 0] = 1.0
    joint_q0 = torch.linspace(-0.3, 0.3, 31).reshape(1, 31)
    env._full_a_teacher = types.SimpleNamespace(
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        joint_pos=torch.cat((joint_q0, joint_q0 + 0.01), dim=0),
    )
    env.action_offset = torch.zeros(31)
    env.act_scale = torch.ones(31)
    env.jnt_lo = torch.full((31,), -1.0)
    env.jnt_hi = torch.full((31,), 1.0)
    env._cap_ok = False
    env.actions = torch.full((1, 31), 7.0)
    env.last_actions = torch.full((1, 31), 8.0)
    env.action_nonfinite_buf = torch.ones(1, dtype=torch.bool)
    env._qdes_previous_executable = torch.full((1, 31), 9.0)
    env._qdes_previous_executable_valid = torch.zeros(1, dtype=torch.bool)
    env._qdes_guard_terminal = torch.ones(1, dtype=torch.bool)
    env._qdes_guard_intervention = torch.ones(1, dtype=torch.bool)
    env._actual_hard_edge_latch = torch.ones(1, dtype=torch.bool)
    env._qdes_reward_processed = torch.full((1, 31), 10.0)
    env._qdes_reward_pre_clamp = torch.full((1, 31), 11.0)
    env._qdes_reward_nominal_projected = torch.full((1, 31), 12.0)
    env._qdes_reward_operand_valid = torch.ones(1, dtype=torch.bool)
    env._controller_trace_latest = {"stale": True}
    env._refresh_aligned_teacher_body_pose = lambda: forward_calls.append(
        "refresh_teacher"
    )
    env._compute_obs = lambda: forward_calls.append("compute_obs")
    env._diagnostic_direct_frame0_table_state = lambda: (
        torch.tensor([False]),
        torch.tensor([False]),
    )
    env._qpos_act = lambda: data.qpos[:, env._q_slice]
    env._epoch_task_f32 = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    env._full_a_launch_state_f32 = torch.arange(
        13, dtype=torch.float32
    ).reshape(1, 13)
    env._full_a_action_slot = torch.tensor([0], dtype=torch.long)
    env._full_a_action_uid = torch.tensor([17], dtype=torch.long)
    env._full_a_mount_normal_sign = torch.tensor([1], dtype=torch.int8)
    env._full_a_teacher_rate = torch.tensor([1.0])
    env._full_a_scaled_t_hit_s = torch.tensor([0.4])
    env._full_a_scaled_t_cycle_s = torch.tensor([1.0])
    env._full_a_pre_swing_wait_s = torch.tensor([0.9])
    env._epoch_phase = torch.tensor(
        [wait_env.FULL_A_PHASE_REVEAL_COMMITTED], dtype=torch.long
    )
    env._epoch_task_valid = torch.tensor([True])
    env._epoch_selected = torch.tensor([True])
    env._epoch_launch_succeeded = torch.tensor([False])
    env._epoch_clock_ticks = torch.tensor([[48, 140, 130, 190, 198]])
    env.reset_generation = torch.tensor([3], dtype=torch.long)
    env.episode_length_buf = torch.tensor([93], dtype=torch.long)
    env.ball_age_buf = torch.tensor([1], dtype=torch.long)
    env._full_a_physical_present = torch.tensor([False])
    env._full_a_owner_valid_bits = torch.tensor([[1, 2, 3, 4]])
    env._full_a_owner_fault_bits = torch.tensor([[0, 0, 0, 0]])
    env._full_a_owner_source_step = torch.tensor([[1, 2, 3, 4]])
    env.common_step_counter = 93
    env._full_a_teacher_frame = torch.tensor([0], dtype=torch.long)
    env._full_a_motion_phase_code = torch.tensor(
        [wait_env.FULL_A_MOTION_PREPARE_PHASE_INDEX], dtype=torch.long
    )
    return env, joint_q0, forward_calls


def test_direct_frame0_install_is_atomic_one_shot_and_preserves_shot_state():
    env, joint_q0, forward_calls = _direct_frame0_install_rig()
    ids = torch.tensor([0], dtype=torch.long)
    ball_before = torch.cat((
        env.sim.data.qpos[:, env.b_q:env.b_q + 7],
        env.sim.data.qvel[:, env.b_v:env.b_v + 6],
    ), dim=1).clone()
    task_before = env._epoch_task_f32.clone()
    lifecycle_before = env._epoch_clock_ticks.clone()
    env.enable_diagnostic_direct_frame0_playback()
    receipt = env.install_diagnostic_direct_frame0_playback(ids)
    torch.testing.assert_close(env._qpos_act(), joint_q0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        env.sim.data.qpos[:, :3], torch.tensor([[0.5, 0.0, 0.9]]),
        rtol=0.0, atol=0.0,
    )
    assert torch.equal(env.sim.data.qvel[:, :37], torch.zeros((1, 37)))
    assert torch.equal(
        env.sim.data.qacc_warmstart[:, :37], torch.zeros((1, 37))
    )
    assert torch.equal(
        env.sim.data.qacc_warmstart[:, 37:], torch.full((1, 6), 3.0)
    )
    assert torch.equal(env.sim.data.ctrl, torch.zeros((1, 31)))
    assert torch.equal(env.actions, joint_q0)
    assert torch.equal(env.last_actions, joint_q0)
    assert torch.equal(env._qdes_previous_executable, joint_q0)
    assert env._controller_trace_latest is None
    assert torch.equal(torch.cat((
        env.sim.data.qpos[:, env.b_q:env.b_q + 7],
        env.sim.data.qvel[:, env.b_v:env.b_v + 6],
    ), dim=1), ball_before)
    assert torch.equal(env._epoch_task_f32, task_before)
    assert torch.equal(env._epoch_clock_ticks, lifecycle_before)
    assert forward_calls == ["forward", "refresh_teacher", "compute_obs"]
    assert all(bool(receipt[name][0]) for name in (
        "applied", "ball_unchanged", "task_unchanged",
        "lifecycle_unchanged", "frame0_pose_exact",
        "robot_velocity_zero", "robot_qacc_warmstart_zero", "ctrl_zero",
        "controller_history_exact", "teacher_cache_refreshed",
        "actuator_state_absent",
    ))
    assert float(receipt["joint_q0_error_max_after_rad"][0]) == 0.0
    assert not bool(receipt["installed_frame0_table_keepout"][0])
    assert not bool(
        receipt["installed_frame0_backend_resolved_table_contact"][0]
    )
    with pytest.raises(RuntimeError, match="boundary differs"):
        env.install_diagnostic_direct_frame0_playback(ids)


def test_direct_frame0_actor_boundary_reports_real_park_drift_without_gating_it():
    env, _joint_q0, _forward_calls = _direct_frame0_install_rig()
    ids = torch.tensor([0], dtype=torch.long)
    env.enable_diagnostic_direct_frame0_playback()

    report = env._diagnostic_direct_frame0_boundary_report(ids)

    assert all(report["checks"].values())
    assert report["schema"] == "action_ball_direct_frame0_boundary_v2"
    assert report["actual"]["ball_age"] == 1
    assert report["actual"]["ball_position_max_abs_delta_from_park_m"] \
        == pytest.approx(0.0020601, abs=2.0e-7)
    assert report["actual"]["ball_linear_velocity_max_abs_mps"] \
        == pytest.approx(0.1962, abs=2.0e-7)


def test_direct_frame0_boundary_failure_names_predicate_and_measured_delta():
    env, _joint_q0, forward_calls = _direct_frame0_install_rig()
    ids = torch.tensor([0], dtype=torch.long)
    env.sim.data.qpos[0, env.b_q + 3 : env.b_q + 7] = 0.0
    env.enable_diagnostic_direct_frame0_playback()

    with pytest.raises(RuntimeError) as error:
        env.install_diagnostic_direct_frame0_playback(ids)

    prefix = "direct frame-zero diagnostic boundary differs: "
    assert str(error.value).startswith(prefix)
    report = json.loads(str(error.value)[len(prefix):])
    assert not report["checks"]["ball_quaternion_nonzero"]
    assert report["checks"]["ball_position_finite"]
    assert report["checks"]["ball_velocity_finite"]
    assert report["actual"]["ball_quaternion_norm"] == 0.0
    assert forward_calls == []

    nonfinite, _joint_q0, nonfinite_calls = _direct_frame0_install_rig()
    nonfinite.sim.data.qvel[0, nonfinite.b_v] = float("nan")
    nonfinite.enable_diagnostic_direct_frame0_playback()
    with pytest.raises(RuntimeError) as nonfinite_error:
        nonfinite.install_diagnostic_direct_frame0_playback(ids)
    nonfinite_report = json.loads(
        str(nonfinite_error.value)[len(prefix):]
    )
    assert not nonfinite_report["checks"]["ball_velocity_finite"]
    assert nonfinite_report["actual"][
        "ball_linear_velocity_max_abs_mps"
    ] is None
    assert nonfinite_calls == []


def test_direct_frame0_production_reparks_before_advancing_waiting_rows():
    prepare = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step
    )
    step = inspect.getsource(wait_env.FullMdpInitialWaitVecEnv._step_full_a)
    runner_source = inspect.getsource(runner.run_teacher_replay)

    assert prepare.index("self._full_a_park_rows(") < prepare.index(
        "return scheduled_due, launch, missed_launch"
    )
    assert step.index("self._full_a_prepare_step()") < step.index(
        "self._advance_plant(actions)"
    )
    assert runner_source.index("install_diagnostic_direct_frame0_playback") \
        < runner_source.index("env.step(action)")


def test_direct_frame0_install_rejects_an_early_or_live_ball_boundary():
    ids = torch.tensor([0], dtype=torch.long)
    early, _joint_q0, early_calls = _direct_frame0_install_rig()
    early.common_step_counter -= 1
    early.enable_diagnostic_direct_frame0_playback()
    with pytest.raises(RuntimeError, match="boundary differs"):
        early.install_diagnostic_direct_frame0_playback(ids)
    assert early_calls == []

    live, _joint_q0, live_calls = _direct_frame0_install_rig()
    live._epoch_launch_succeeded[0] = True
    live._full_a_physical_present[0] = True
    live.enable_diagnostic_direct_frame0_playback()
    with pytest.raises(RuntimeError, match="boundary differs"):
        live.install_diagnostic_direct_frame0_playback(ids)
    assert live_calls == []


def test_direct_frame0_invalid_target_fails_before_mutating_the_plant():
    env, _joint_q0, forward_calls = _direct_frame0_install_rig()
    ids = torch.tensor([0], dtype=torch.long)
    qpos_before = env.sim.data.qpos.clone()
    qvel_before = env.sim.data.qvel.clone()
    env.act_scale[4] = 0.0
    env.enable_diagnostic_direct_frame0_playback()
    with pytest.raises(RuntimeError, match="target is invalid"):
        env.install_diagnostic_direct_frame0_playback(ids)
    assert torch.equal(env.sim.data.qpos, qpos_before)
    assert torch.equal(env.sim.data.qvel, qvel_before)
    assert forward_calls == []


def test_teacher_replay_frozen_counter_and_hash_are_content_exact():
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10]),
        pre_swing_wait_s=torch.tensor([0.10]),
        step_dt=0.02,
    )
    assert frozen.tolist() == [3]
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    expected = hashlib.sha256(
        np.asarray([[1.0, 2.0]], dtype="<f4").tobytes()
    ).hexdigest()
    assert replay.tensor_f32_sha256(value) == expected


@pytest.mark.parametrize(
    ("dtype", "encoded_seconds", "expected"),
    (
        (torch.float32, 0.10, [3, 3, 4]),
        (torch.float64, 0.10, [3, 3, 4]),
        (torch.float32, 0.06, [1, 1, 2]),
        (torch.float64, 0.06, [1, 1, 2]),
    ),
)
def test_teacher_replay_frozen_counter_only_snaps_dtype_roundoff_at_tick_boundary(
    dtype, encoded_seconds, expected
):
    encoded_boundary = torch.tensor([encoded_seconds], dtype=dtype)
    below_boundary = torch.nextafter(
        encoded_boundary, torch.full_like(encoded_boundary, -torch.inf)
    )
    above_boundary = torch.nextafter(
        encoded_boundary, torch.full_like(encoded_boundary, torch.inf)
    )
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10, 10, 10]),
        pre_swing_wait_s=torch.cat(
            (below_boundary, encoded_boundary, above_boundary)
        ),
        step_dt=0.02,
    )
    # The encoded schedule and its lower neighbour both end at the intended
    # integer tick.  One representable float above it is a real positive margin
    # and must keep the following tick instead of being swallowed by boundary
    # normalization.
    assert frozen.tolist() == expected


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_teacher_replay_frozen_counter_strictly_ceils_non_boundary_wait(dtype):
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10]),
        pre_swing_wait_s=torch.tensor([0.061], dtype=dtype),
        step_dt=0.02,
    )
    assert frozen.tolist() == [2]


def test_teacher_replay_decimal_timebase_encodes_three_fiftieths_as_point_zero_six():
    numerator, denominator = replay._decimal_policy_timebase_ratio(0.02)
    assert (numerator, denominator) == (1, 50)
    literal_boundary = torch.tensor([0.06], dtype=torch.float64)
    upper_neighbour = torch.nextafter(
        literal_boundary, torch.full_like(literal_boundary, torch.inf)
    )
    canonical_boundary = (
        torch.tensor([3.0], dtype=torch.float64)
        * float(numerator)
        / float(denominator)
    )
    assert torch.equal(canonical_boundary, literal_boundary)
    assert not torch.equal(canonical_boundary, upper_neighbour)


def test_pre_step_snapshot_drives_request_while_post_step_is_independent():
    env = types.SimpleNamespace(
        _epoch_task_valid=torch.tensor([True]),
        _full_a_teacher_frame=torch.tensor([4]),
        _full_a_motion_phase_code=torch.tensor([1]),
        _full_a_teacher_joint_pos=torch.tensor([[2.0, 4.0]]),
        _epoch_clock_ticks=torch.tensor([[10, 11, 12, 13, 14]]),
        _full_a_pre_swing_wait_s=torch.tensor([0.0]),
    )
    pre = replay.capture_teacher_replay_pre_step(env)
    env._epoch_task_valid.fill_(False)
    env._full_a_teacher_frame.fill_(9)
    env._full_a_motion_phase_code.fill_(4)
    env._full_a_teacher_joint_pos.fill_(99.0)
    requested = replay.frozen_teacher_qdes(
        torch=torch,
        task_valid=pre.task_valid,
        hold_qdes=torch.zeros((1, 2)),
        previous_qdes=torch.zeros((1, 2)),
        teacher_qdes=pre.teacher_qdes,
        frozen_steps=torch.zeros(1, dtype=torch.long),
        bridge=wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge,
    )
    assert pre.task_valid.tolist() == [True]
    assert pre.teacher_frame.tolist() == [4]
    assert pre.motion_phase.tolist() == [1]
    assert requested.tolist() == [[2.0, 4.0]]
    assert env._epoch_task_valid.tolist() == [False]
    assert env._full_a_teacher_frame.tolist() == [9]


def test_contact_patch_boundary_does_not_alias_final_forward_to_substep():
    substep = replay.contact_capture_boundary(
        transition_start_step=17,
        capture_boundary="physics_substep_poststate",
        physics_substep_index=19,
        decimation=20,
    )
    final = replay.contact_capture_boundary(
        transition_start_step=17,
        capture_boundary="post_forward_final",
        physics_substep_index=None,
        decimation=20,
    )
    assert substep["completed_physics_substeps"] == 19
    assert final["completed_physics_substeps"] == 20
    assert substep["physics_substep_index"] == 19
    assert final["physics_substep_index"] is None
    with pytest.raises(ValueError, match="physics substep"):
        replay.contact_capture_boundary(
            transition_start_step=17,
            capture_boundary="physics_substep_poststate",
            physics_substep_index=0,
            decimation=20,
        )
    with pytest.raises(ValueError, match="final-forward"):
        replay.contact_capture_boundary(
            transition_start_step=17,
            capture_boundary="post_forward_final",
            physics_substep_index=20,
            decimation=20,
        )


def test_different_shot_generation_is_not_a_valid_patch_key():
    question = torch.tensor([1.0, 2.0])
    question_sha = replay.tensor_f32_sha256(question)
    with pytest.raises(RuntimeError, match="different shot"):
        replay.validate_contact_patch_shot(
            contact_patch={
                "present": True,
                "reset_generation": 4,
                "question_f32_sha256": question_sha,
            },
            question_reset_generation=3,
            question_f32_sha256=question_sha,
        )


def test_contact_patch_is_absent_until_explicitly_enabled():
    bare = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="not enabled"):
        wait_env.FullMdpInitialWaitVecEnv.diagnostic_first_generic_contact_patch(
            bare
        )
    source = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts
    )
    tree = ast.parse(textwrap.dedent(source))
    getattr_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "_diagnostic_contact_patch_consumer"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value is None
    ]
    assert len(getattr_calls) == 1
    guarded_calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "patch_consumer"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.IsNot)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        ):
            continue
        guarded_calls.extend(
            child
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "patch_consumer"
        )
    assert len(guarded_calls) == 1


def test_contact_patch_bridge_distinguishes_torch_proxy_warp_and_device_duck():
    calls = []

    class FakeWarpArray:
        pass

    class FakeWarp:
        array = FakeWarpArray

        @staticmethod
        def to_torch(value):
            calls.append(value)
            return torch.tensor([3.0])

    class FakeTorchArray:
        def __init__(self, tensor):
            self.tensor = tensor
            self.device = tensor.device
            self.indexes = []

        def __getitem__(self, index):
            self.indexes.append(index)
            return self.tensor[index]

    direct = torch.tensor([1.0])
    proxy_view = torch.tensor([2.0])
    proxy = FakeTorchArray(proxy_view)
    warp_value = FakeWarpArray()
    bridge = wait_env.FullMdpInitialWaitVecEnv._diagnostic_contact_field_torch_view

    assert bridge(
        torch=torch,
        warp=FakeWarp,
        torch_array_type=FakeTorchArray,
        value=direct,
    ) is direct
    bridged_proxy = bridge(
        torch=torch,
        warp=FakeWarp,
        torch_array_type=FakeTorchArray,
        value=proxy,
    )
    assert torch.equal(bridged_proxy, proxy_view)
    assert bridged_proxy.data_ptr() == proxy_view.data_ptr()
    assert proxy.indexes == [Ellipsis]
    bridged_warp = bridge(
        torch=torch,
        warp=FakeWarp,
        torch_array_type=FakeTorchArray,
        value=warp_value,
    )
    assert torch.equal(bridged_warp, torch.tensor([3.0]))
    assert calls == [warp_value]

    device_only = types.SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(TypeError, match="no supported Torch view"):
        bridge(
            torch=torch,
            warp=FakeWarp,
            torch_array_type=FakeTorchArray,
            value=device_only,
        )


def _table_attribution_rig():
    return types.SimpleNamespace(
        _torch=torch,
        decimation=20,
        _diagnostic_contact_patch_transition_start_step=41,
        _diagnostic_first_table_terminal_source=None,
        _diagnostic_first_resolved_substep=None,
        _diagnostic_table_tick_first_positive=None,
        _diagnostic_table_tick_keepout=False,
        _diagnostic_table_tick_final_resolved=False,
        _diagnostic_table_tick_resolved_any_substep=False,
        _diagnostic_table_attribution_consumer=True,
    )


def _keepout_witness(*, component_id="torso_box"):
    return {
        "schema": "action_ball_keepout_first_witness_v1",
        "selection": "first_production_order_overlap",
        "reason": "sat_overlap",
        "component_index": 7,
        "component_id": component_id,
        "component_kind": "body_proxy",
        "owner_body_index": 9,
        "owner_body_name": "torso_Link",
        "table_index": 0,
        "table_role": "top",
        "sat_signed_margin_m": -0.003,
        "root_position_env_m": [0.0, 0.0, 1.0],
        "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "owner_position_env_m": [0.2, 0.0, 1.1],
        "owner_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "plant_identity": {
            "root_mjcf_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "portable_identity_sha256": "c" * 64,
            "verification_receipt_sha256": "d" * 64,
            "owner_local_frame_sha256": "e" * 64,
        },
        "collision_artifact_sha256": "f" * 64,
        "collision_content_sha256": "0" * 64,
    }


def test_keepout_witness_winner_is_component_major_and_unknown_fails_closed():
    exact = torch.zeros((63, 5), dtype=torch.bool)
    exact[12, 0] = True
    exact[3, 4] = True
    assert keepout._host_test_first_overlap_index(exact) == (3, 4)
    with pytest.raises(RuntimeError, match="no SAT witness"):
        keepout._host_test_first_overlap_index(torch.zeros_like(exact))


def test_table_attribution_freezes_first_positive_before_reset_once_only():
    env = _table_attribution_rig()
    capture = wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution
    finalize = wait_env.FullMdpInitialWaitVecEnv.diagnostic_table_attribution_tick
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=1,
        capture_boundary="physics_substep_poststate",
    )
    capture(
        env,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=2,
        capture_boundary="physics_substep_poststate",
        keepout_witness=_keepout_witness(),
    )
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([True]),
        resolved_is_final=False,
        substep_index=3,
        capture_boundary="physics_substep_poststate",
    )
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([False]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
    )
    result = finalize(
        env,
        torch.tensor([
            wait_env.FULLMDP_TERMINATION_BITS["robot_hit_table"]
        ]),
    )
    assert result["keepout_source"] is True
    assert result["backend_resolved_table_contact"] is False
    assert result["resolved_any_substep"] is True
    assert result["first_resolved_substep"]["physics_substep_index"] == 3
    assert result["first_table_terminal_source"] == {
        "transition_start_step": 41,
        "capture_boundary": "physics_substep_poststate",
        "physics_substep_index": 2,
        "completed_physics_substeps": 2,
        "keepout_source": True,
        "backend_resolved_table_contact": False,
        "keepout_witness": _keepout_witness(),
    }

    wait_env.FullMdpInitialWaitVecEnv._begin_diagnostic_table_attribution_tick(env)
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([True]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
    )
    second = finalize(
        env,
        torch.tensor([
            wait_env.FULLMDP_TERMINATION_BITS["robot_hit_table"]
        ]),
    )
    assert second["first_table_terminal_source"] == result[
        "first_table_terminal_source"
    ]


@pytest.mark.parametrize(
    ("keepout", "resolved", "terminal"),
    ((False, False, True), (True, False, False), (False, True, False)),
)
def test_table_attribution_unknown_or_unmatched_source_fails_closed(
    keepout, resolved, terminal
):
    env = _table_attribution_rig()
    wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution(
        env,
        keepout=torch.tensor([keepout]),
        resolved=torch.tensor([resolved]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
        keepout_witness=_keepout_witness() if keepout else None,
    )
    bits = (
        wait_env.FULLMDP_TERMINATION_BITS["robot_hit_table"] if terminal else 0
    )
    with pytest.raises(RuntimeError, match="source is unknown"):
        wait_env.FullMdpInitialWaitVecEnv.diagnostic_table_attribution_tick(
            env, torch.tensor([bits])
        )


def test_keepout_witness_is_frozen_once_and_unknown_fails_closed():
    env = _table_attribution_rig()
    capture = wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution
    capture(
        env,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=4,
        capture_boundary="physics_substep_poststate",
        keepout_witness=_keepout_witness(component_id="first"),
    )
    capture(
        env,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=5,
        capture_boundary="physics_substep_poststate",
        keepout_witness=_keepout_witness(component_id="second"),
    )
    assert env._diagnostic_table_tick_first_positive["keepout_witness"][
        "component_id"
    ] == "first"

    unknown = _keepout_witness()
    unknown["table_role"] = "mystery"
    fresh = _table_attribution_rig()
    with pytest.raises(RuntimeError, match="witness is unknown"):
        capture(
            fresh,
            keepout=torch.tensor([True]),
            resolved=torch.tensor([False]),
            resolved_is_final=False,
            substep_index=1,
            capture_boundary="physics_substep_poststate",
            keepout_witness=unknown,
        )

    invalid_pose = _keepout_witness()
    invalid_pose["owner_position_env_m"][1] = float("nan")
    fresh = _table_attribution_rig()
    with pytest.raises(RuntimeError, match="witness is unknown"):
        capture(
            fresh,
            keepout=torch.tensor([True]),
            resolved=torch.tensor([False]),
            resolved_is_final=False,
            substep_index=1,
            capture_boundary="physics_substep_poststate",
            keepout_witness=invalid_pose,
        )


def test_keepout_witness_labels_never_reread_the_artifact_at_enable_time():
    source = inspect.getsource(
        wait_env.DeviceExactTableKeepout.enable_diagnostic_first_positive_witness
    )
    assert "read_text" not in source
    assert "COLLISION_PROXY_ARTIFACT" not in source


def test_substep_resolved_contact_can_separate_before_canonical_final_forward():
    env = _table_attribution_rig()
    capture = wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([True]),
        resolved_is_final=False,
        substep_index=7,
        capture_boundary="physics_substep_poststate",
    )
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([False]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
    )
    result = wait_env.FullMdpInitialWaitVecEnv.diagnostic_table_attribution_tick(
        env, torch.tensor([0])
    )
    assert result["resolved_any_substep"] is True
    assert result["backend_resolved_table_contact"] is False
    assert result["keepout_source"] is False
    assert result["first_table_terminal_source"] is None
    assert result["first_resolved_substep"] == {
        "transition_start_step": 41,
        "capture_boundary": "physics_substep_poststate",
        "physics_substep_index": 7,
        "completed_physics_substeps": 7,
        "backend_resolved_table_contact": True,
    }


def test_teacher_replay_cli_is_n1_zero_ppo_and_reuses_live_owner():
    signature = inspect.signature(runner.main)
    assert signature.parameters["diagnostic_teacher_replay"].default is False
    assert signature.parameters["diagnostic_teacher_replay_steps"].default == 180
    assert (
        signature.parameters[
            "diagnostic_teacher_replay_handoff_mode"
        ].default
        == replay.TEACHER_REPLAY_HANDOFF_SPLIT_READY_BRIDGE
    )
    source = inspect.getsource(runner._run_teacher_replay)
    assert "env.step(action)" in source
    assert "ppo_update_calls\": 0" in source
    assert "enable_diagnostic_first_generic_contact_patch" in source
    assert "_epoch_task_f32" in source
    assert "_full_a_launch_state_f32" in source
    assert "install_diagnostic_direct_frame0_playback" in source
    with pytest.raises(
        ValueError, match="requires diagnostic teacher replay"
    ):
        runner.main(
            diagnostic_teacher_replay_handoff_mode=(
                replay.TEACHER_REPLAY_HANDOFF_DIRECT_FRAME0
            )
        )
