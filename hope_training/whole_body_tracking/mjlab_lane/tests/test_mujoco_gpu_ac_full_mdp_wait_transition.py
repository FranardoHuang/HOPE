"""Direct contracts for the real MuJoCo FullMDP WAIT transition.

The host tests exercise the same reward and termination helpers used by the
production ``step`` method with fixed tensors.  The opt-in GPU tests cross the
real ``A3ReadyBallVecEnv._advance_plant`` callpoint; no legacy M04 owner,
receipt, or synthetic VecEnv is accepted as runtime evidence.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import ast
import inspect
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env
import mujoco_full_mdp_update_ledger as update_ledger


P = wait_env.observation_contract
RUN_GPU_DIRECT = os.environ.get("ACTIONBALL_RUN_MUJOCO_GPU_DIRECT") == "1"


def test_regularization_projection_operands_are_the_same_shared_guard_result():
    plant_source = inspect.getsource(wait_env.A3ReadyBallVecEnv._advance_plant)
    assert (
        "self._qdes_reward_processed = q_des" in plant_source
        and "self._qdes_reward_nominal_projected = guard.nominal_projected_qdes"
        in plant_source
        and "self._qdes_reward_projection_span = guard.nominal_projection_span"
        in plant_source
    )
    assert "_advance_plant" not in wait_env.FullMdpInitialWaitVecEnv.__dict__
    reward_source = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_regularization_reward_terms
    )
    assert "self._qdes_reward_processed" in reward_source
    assert "self._qdes_reward_nominal_projected" in reward_source
    assert "self._qdes_reward_projection_span" in reward_source
    assert "torch.clamp" not in reward_source


def test_step_refreshes_derived_state_before_termination_and_reward():
    source = Path(wait_env.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FullMdpInitialWaitVecEnv"
    )
    step = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )

    def first_line(attr):
        return min(
            node.lineno
            for node in ast.walk(step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        )

    assert first_line("_advance_plant") < first_line("forward")
    assert first_line("forward") < first_line(
        "_latch_post_forward_resolved_table_contacts"
    )
    assert first_line("_latch_post_forward_resolved_table_contacts") < first_line(
        "_latch_post_forward_table_keepout"
    )
    assert first_line("_latch_post_forward_table_keepout") < first_line("_state")
    assert first_line("_state") < first_line("_fullmdp_termination")
    assert first_line("_fullmdp_termination") < first_line("_fullmdp_reward")


def test_post_forward_resolved_table_contact_is_latched_without_counter_replay():
    env = SimpleNamespace(
        _torch=torch,
        _con_geom=torch.tensor([[5, 2], [7, 8], [2, 1]], dtype=torch.int64),
        _con_idx=torch.arange(3),
        _nacon=torch.tensor([2], dtype=torch.int64),
        _con_world=torch.tensor([0, 1, 0], dtype=torch.int64),
        _ball_gid=0,
        _geom_class=torch.zeros(9, dtype=torch.int8),
        _table_gid=5,
        _robot_table_ok=True,
        _is_robot_geom=torch.tensor(
            [False, False, True, False, False, False, False, False, False]
        ),
        _cur_robot_table=torch.zeros(2),
        _contact_ball_racket_by_world=torch.zeros(2),
        _contact_ball_table_by_world=torch.zeros(2),
        _contact_robot_table_by_world=torch.zeros(2),
        _contact_zero_total=torch.zeros((), dtype=torch.long),
        device=torch.device("cpu"),
        num_envs=2,
    )
    wait_env.FullMdpInitialWaitVecEnv._latch_post_forward_resolved_table_contacts(env)
    assert torch.equal(env._cur_robot_table, torch.tensor([1.0, 0.0]))


def test_contact_census_preserves_base_counts_and_returns_compact_world_facts():
    geom_class = torch.zeros(7, dtype=torch.int8)
    geom_class[1] = 1
    geom_class[5] = 2
    robot_geom = torch.zeros(7, dtype=torch.bool)
    robot_geom[6] = True
    env = SimpleNamespace(
        _torch=torch,
        device=torch.device("cpu"),
        num_envs=2,
        _con_geom=torch.tensor(
            [[2, 1], [5, 2], [5, 6], [2, 1]], dtype=torch.long
        ),
        _con_idx=torch.arange(4),
        _nacon=torch.tensor([3], dtype=torch.long),
        _con_world=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        _ball_gid=2,
        _geom_class=geom_class,
        _robot_table_ok=True,
        _table_gid=5,
        _is_robot_geom=robot_geom,
        _contact_ball_racket_by_world=torch.zeros(2),
        _contact_ball_table_by_world=torch.zeros(2),
        _contact_robot_table_by_world=torch.zeros(2),
        _contact_zero_total=torch.zeros((), dtype=torch.long),
        _cur_touched=torch.zeros(2),
        _cur_robot_table=torch.zeros(2),
        _acc={
            "contact_ball_racket_substeps": torch.zeros(()),
            "contact_ball_table_substeps": torch.zeros(()),
            "contact_robot_table_substeps": torch.zeros(()),
        },
    )
    env._contact_census = MethodType(
        wait_env.A3ReadyBallVecEnv._contact_census, env
    )

    census = wait_env.A3ReadyBallVecEnv._probe_contacts(env)

    assert tuple(census.ball_racket_by_world.shape) == (2,)
    assert tuple(census.ball_table_by_world.shape) == (2,)
    assert tuple(census.robot_table_by_world.shape) == (2,)
    assert torch.equal(census.ball_racket_by_world, torch.tensor([1.0, 0.0]))
    assert torch.equal(census.ball_table_by_world, torch.tensor([1.0, 0.0]))
    assert torch.equal(census.robot_table_by_world, torch.tensor([0.0, 1.0]))
    assert int(census.ball_racket_total) == 1
    assert int(census.ball_table_total) == 1
    assert int(census.robot_table_total) == 1
    assert env._acc["contact_ball_racket_substeps"] == 1
    assert env._acc["contact_ball_table_substeps"] == 1
    assert env._acc["contact_robot_table_substeps"] == 1
    assert torch.equal(env._cur_touched, torch.tensor([1.0, 0.0]))
    assert torch.equal(env._cur_robot_table, torch.tensor([0.0, 1.0]))


def test_full_a_contact_consumers_do_not_rescan_the_contact_buffer():
    source = ast.parse(Path(wait_env.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )

    def method(name):
        return next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    forbidden_contact_arrays = {"_con_geom", "_con_idx", "_nacon", "_con_world"}
    for name in (
        "_full_a_latch_ball_contacts",
        "_latch_post_forward_resolved_table_contacts",
    ):
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr in forbidden_contact_arrays
            for node in ast.walk(method(name))
        )

    full_step = method("_step_full_a")
    census_calls = [
        node
        for node in ast.walk(full_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_contact_census"
    ]
    assert len(census_calls) == 1
    for consumer in (
        "_latch_post_forward_resolved_table_contacts",
        "_full_a_latch_ball_contacts",
    ):
        call = next(
            node
            for node in ast.walk(full_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == consumer
        )
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "final_contact_census"


def _fixed_reward_env():
    num_envs = 2
    num_bodies = 3
    dtype = torch.float64
    teacher_pos = torch.zeros((num_envs, num_bodies, 3), dtype=dtype)
    teacher_quat = torch.zeros((num_envs, num_bodies, 4), dtype=dtype)
    teacher_quat[..., 0] = 1.0
    teacher_lin = torch.zeros((num_envs, num_bodies, 3), dtype=dtype)
    teacher_ang = torch.zeros_like(teacher_lin)

    body_pos = teacher_pos.clone()
    body_pos[0, 1] = torch.tensor([0.12, -0.04, 0.03], dtype=dtype)
    body_pos[1] = torch.tensor(
        [[0.01, 0.02, 0.03], [-0.04, 0.05, 0.06], [0.07, 0.08, -0.09]],
        dtype=dtype,
    )
    body_quat = teacher_quat.clone()
    for row, angles in enumerate(((0.10, 0.20, 0.30), (0.15, 0.25, 0.35))):
        half = 0.5 * torch.tensor(angles, dtype=dtype)
        body_quat[row, :, 0] = torch.cos(half)
        body_quat[row, :, 1] = torch.sin(half)
    body_lin = torch.tensor(
        [
            [[0.10, 0.00, 0.00], [0.00, 0.20, 0.00], [0.00, 0.00, 0.30]],
            [[-0.20, 0.10, 0.00], [0.10, 0.00, 0.20], [0.00, -0.10, 0.10]],
        ],
        dtype=dtype,
    )
    body_ang = torch.tensor(
        [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[-0.09, 0.08, -0.07], [0.06, -0.05, 0.04], [-0.03, 0.02, -0.01]],
        ],
        dtype=dtype,
    )
    xipos = body_pos + torch.tensor([0.11, -0.07, 0.05], dtype=dtype)
    subtree_com = torch.tensor(
        [
            [[-0.20, 0.30, 0.10], [0.40, -0.10, 0.20], [0.0, 0.0, 0.0]],
            [[0.10, 0.25, -0.15], [-0.30, 0.20, 0.05], [0.0, 0.0, 0.0]],
        ],
        dtype=dtype,
    )
    body_root_ids = torch.tensor([0, 0, 1])
    subtree_root = subtree_com[:, body_root_ids]
    cvel_linear = body_lin + torch.cross(
        xipos - subtree_root, body_ang, dim=-1
    )
    cvel = torch.cat((body_ang, cvel_linear), dim=-1)
    env = SimpleNamespace(
        _torch=torch,
        sim=SimpleNamespace(
            data=SimpleNamespace(
                xpos=body_pos,
                xquat=body_quat,
                xipos=xipos,
                subtree_com=subtree_com,
                cvel=cvel,
            )
        ),
        _fullmdp_body_ids=torch.arange(num_bodies),
        _fullmdp_body_root_ids=body_root_ids,
        _fullmdp_anchor_index=1,
        _fullmdp_upper_non_wrist_body_indices=torch.tensor([0, 1]),
        _teacher_body_pos=teacher_pos,
        _teacher_body_quat=teacher_quat,
        _teacher_body_lin_vel=teacher_lin,
        _teacher_body_ang_vel=teacher_ang,
        _teacher_racket_site_pos_w=torch.zeros((num_envs, 3), dtype=dtype),
        _teacher_racket_site_lin_vel_w=torch.zeros(
            (num_envs, 3), dtype=dtype
        ),
        _teacher_racket_signed_normal_w=torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=dtype
        ).repeat(num_envs, 1),
        _teacher_racket_long_axis_w=torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=dtype
        ).repeat(num_envs, 1),
        _fullmdp_dense_weights=torch.tensor(
            [0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=dtype
        ),
        _fullmdp_paddle_weights=torch.tensor(
            [
                spec.manager_weight
                for spec in wait_env.FULLMDP_PADDLE_REWARD_SPECS
            ],
            dtype=dtype,
        ),
        _fullmdp_paddle_playback_scales=torch.tensor(
            [
                spec.scale_during_playback
                for spec in wait_env.FULLMDP_PADDLE_REWARD_SPECS
            ],
            dtype=dtype,
        ),
        _fullmdp_paddle_precision_stds=torch.tensor(
            [spec.std for spec in wait_env.FULLMDP_PADDLE_REWARD_SPECS],
            dtype=dtype,
        ),
        _fullmdp_paddle_coarse_stds=torch.tensor(
            [spec.coarse_std for spec in wait_env.FULLMDP_PADDLE_REWARD_SPECS],
            dtype=dtype,
        ),
        _fullmdp_mount_normal_sign=torch.ones(num_envs, dtype=dtype),
        _fullmdp_regularization_reward_terms=lambda: torch.zeros(
            (num_envs, len(wait_env.reward_contract.REGULARIZATION_SPECS)),
            dtype=dtype,
        ),
        _epoch_task_valid=torch.zeros(num_envs, dtype=torch.bool),
        _epoch_phase=torch.zeros(num_envs, dtype=torch.long),
        _full_a_motion_phase_code=torch.full(
            (num_envs,),
            wait_env.FULL_A_MOTION_PREPARE_PHASE_INDEX,
            dtype=torch.long,
        ),
        _full_a_outcome_code=torch.zeros(num_envs, dtype=torch.long),
        _full_a_lifecycle_reward_weights=torch.tensor(
            wait_env.portable_reward.LIFECYCLE_WEIGHTS, dtype=dtype
        ),
        _quat_error_sq=wait_env.FullMdpInitialWaitVecEnv._quat_error_sq,
        _aligned_teacher_body_pose=(
            wait_env.FullMdpInitialWaitVecEnv._aligned_teacher_body_pose
        ),
        _apply_teacher_yaw_alignment=(
            wait_env.FullMdpInitialWaitVecEnv._apply_teacher_yaw_alignment
        ),
        _teacher_yaw_alignment=(
            wait_env.FullMdpInitialWaitVecEnv._teacher_yaw_alignment
        ),
        _quat_apply_wxyz=(
            wait_env.FullMdpInitialWaitVecEnv._quat_apply_wxyz
        ),
        _body_com_velocities_from_cvel=(
            wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_from_cvel
        ),
        _full_a_racket_kinematics=lambda: (
            torch.zeros((num_envs, 3), dtype=dtype),
            torch.zeros((num_envs, 3), dtype=dtype),
            torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype).repeat(
                num_envs, 1
            ),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).repeat(
                num_envs, 1
            ),
        ),
        env=SimpleNamespace(
            scene=SimpleNamespace(
                env_origins=torch.zeros((num_envs, 3), dtype=dtype)
            )
        ),
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.02,
    )
    env._body_com_velocities_w = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_w, env
    )
    env._refresh_aligned_teacher_body_pose = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._refresh_aligned_teacher_body_pose,
        env,
    )
    env._refresh_aligned_teacher_body_pose()
    env._full_a_racket_kinematics = lambda: (
        env._aligned_teacher_racket_site_pos_w - env.env.scene.env_origins,
        env._aligned_teacher_racket_site_lin_vel_w,
        env._aligned_teacher_racket_signed_normal_w,
        env._aligned_teacher_racket_long_axis_w,
    )
    return env


def _test_quat_mul(left, right):
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _test_quat_apply(quaternion, vector):
    q_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(q_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.cross(
        q_vector, twice_cross, dim=-1
    )


def _independent_aligned_teacher(env, pos, quat):
    anchor = env._fullmdp_anchor_index
    teacher_anchor_pos = env._teacher_body_pos[:, anchor]
    teacher_anchor_quat = env._teacher_body_quat[:, anchor]
    inverse = teacher_anchor_quat * teacher_anchor_quat.new_tensor(
        [1.0, -1.0, -1.0, -1.0]
    )
    delta = _test_quat_mul(quat[:, anchor], inverse)
    yaw = torch.atan2(
        2.0 * (delta[:, 0] * delta[:, 3] + delta[:, 1] * delta[:, 2]),
        1.0 - 2.0 * (delta[:, 2].square() + delta[:, 3].square()),
    )
    yaw_quat = torch.zeros_like(delta)
    yaw_quat[:, 0] = torch.cos(0.5 * yaw)
    yaw_quat[:, 3] = torch.sin(0.5 * yaw)
    expanded = yaw_quat[:, None, :].expand_as(env._teacher_body_quat)
    aligned_anchor = pos[:, anchor].clone()
    aligned_anchor[:, 2] = teacher_anchor_pos[:, 2]
    aligned_pos = aligned_anchor[:, None, :] + _test_quat_apply(
        expanded, env._teacher_body_pos - teacher_anchor_pos[:, None, :]
    )
    return aligned_pos, _test_quat_mul(expanded, env._teacher_body_quat)


def test_host_reward28_matches_common_and_measured_paddle_formulas():
    env = _fixed_reward_env()
    env._fullmdp_regularization_reward_terms = lambda: torch.zeros((2, 4))
    assert env._fullmdp_paddle_weights.tolist() == [1.0, 1.0, 1.0, 0.5]
    assert env._fullmdp_paddle_playback_scales.tolist() == [4.0] * 4
    reward, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)

    data = env.sim.data
    ids = env._fullmdp_body_ids
    pos = data.xpos[:, ids]
    quat = data.xquat[:, ids]
    lin, ang = wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_w(env)
    anchor = env._fullmdp_anchor_index
    non_wrist = env._fullmdp_upper_non_wrist_body_indices
    aligned_pos = env._aligned_teacher_body_pos
    aligned_quat = env._aligned_teacher_body_quat
    quat_dot = torch.abs(torch.sum(aligned_quat * quat, dim=-1)).clamp(0.0, 1.0)
    quat_error_sq = torch.square(2.0 * torch.acos(quat_dot))
    expected_raw = torch.stack(
        (
            torch.exp(
                -torch.square(env._teacher_body_pos[:, anchor] - pos[:, anchor])
                .sum(dim=-1)
                / 0.3**2
            ),
            torch.exp(-quat_error_sq[:, anchor] / 0.4**2),
            torch.exp(
                -torch.square(aligned_pos[:, non_wrist] - pos[:, non_wrist])
                .sum(dim=-1)
                .mean(dim=-1)
                / 0.3**2
            ),
            0.5
            * (
                torch.exp(
                    -quat_error_sq[:, non_wrist].mean(dim=-1) / 0.4**2
                )
                + torch.exp(
                    -quat_error_sq[:, non_wrist].mean(dim=-1)
                    / wait_env.FULL_A_BODY_ORIENTATION_COARSE_STD_RAD**2
                )
            ),
            torch.exp(
                -torch.square(
                    env._teacher_body_lin_vel[:, non_wrist]
                    - lin[:, non_wrist]
                )
                .sum(dim=-1)
                .mean(dim=-1)
                / 1.0**2
            ),
            torch.exp(
                -torch.square(
                    env._teacher_body_ang_vel[:, non_wrist]
                    - ang[:, non_wrist]
                )
                .sum(dim=-1)
                .mean(dim=-1)
                / 3.14**2
            ),
        ),
        dim=1,
    )
    weights = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=terms.dtype)
    expected_dense = expected_raw * weights * env.step_dt
    expected_paddle = env._fullmdp_paddle_weights * env.step_dt

    assert tuple(terms.shape) == (2, wait_env.reward_contract.REWARD_TERM_COUNT)
    assert torch.count_nonzero(terms[:, :14]) == 0
    assert torch.allclose(terms[:, 14:20], expected_dense, rtol=0.0, atol=1.0e-15)
    assert torch.equal(terms[:, 20:24], expected_paddle.repeat(2, 1))
    assert torch.count_nonzero(terms[:, 24:]) == 0
    assert torch.allclose(
        reward,
        expected_dense.sum(dim=1) + expected_paddle.sum(),
        rtol=0.0,
        atol=1.0e-15,
    )

    shared_racket_kinematics = env._full_a_racket_kinematics()
    shared_tracked_kinematics = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_tracked_body_kinematics(env)
    )
    original_racket_kinematics = env._full_a_racket_kinematics
    original_body_velocities = env._body_com_velocities_w
    env._full_a_racket_kinematics = lambda: pytest.fail(
        "precomputed official-site tuple was recomputed"
    )
    env._body_com_velocities_w = lambda: pytest.fail(
        "precomputed tracked-body tuple was recomputed"
    )
    cached_reward, cached_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(
            env, shared_racket_kinematics, shared_tracked_kinematics
        )
    )
    assert torch.equal(cached_reward, reward)
    assert torch.equal(cached_terms, terms)
    env._full_a_racket_kinematics = original_racket_kinematics
    env._body_com_velocities_w = original_body_velocities

    # Counterexample: this buffer is live Reward28 authority, not a private
    # command-ramp scratch tensor.  Replacing one exact teacher body with a
    # midpoint changes the body-position reward immediately.
    exact_body_term = terms[:, 16].clone()
    env._teacher_body_pos[:, 0, 0] += 0.25
    env._refresh_aligned_teacher_body_pose()
    _drifted_reward, drifted_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    )
    assert not torch.equal(drifted_terms[:, 16], exact_body_term)


def test_reward28_paddle_composite_channels_have_fixed_analytic_precision_value():
    env = _fixed_reward_env()
    target_pos = (
        env._aligned_teacher_racket_site_pos_w - env.env.scene.env_origins
    )
    target_velocity = env._aligned_teacher_racket_site_lin_vel_w
    target_pos = target_pos + torch.tensor(
        [0.075, 0.0, 0.0], dtype=torch.float64
    )
    target_velocity = target_velocity + torch.tensor(
        [0.50, 0.0, 0.0], dtype=torch.float64
    )
    face_angle = math.radians(15.0)
    target_normal = torch.tensor(
        [[math.sin(face_angle), math.cos(face_angle), 0.0]],
        dtype=torch.float64,
    ).repeat(env.num_envs, 1)
    long_angle = math.radians(10.0)
    target_long_axis = torch.tensor(
        [[math.cos(long_angle), 0.0, math.sin(long_angle)]],
        dtype=torch.float64,
    ).repeat(env.num_envs, 1)
    calls = 0

    def achieved_once():
        nonlocal calls
        calls += 1
        return target_pos, target_velocity, target_normal, target_long_axis

    env._full_a_racket_kinematics = achieved_once
    _reward, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)

    assert calls == 1
    expected_kernel = 0.5 * math.exp(-1.0) + 0.5 / (1.0 + 0.25**2)
    torch.testing.assert_close(
        terms[:, 20:24],
        (expected_kernel * env._fullmdp_paddle_weights * env.step_dt).repeat(
            env.num_envs, 1
        ),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_reward28_scales_only_motion_owned_playback_rows():
    env = _fixed_reward_env()
    env.full_a_mode = True
    env._full_a_motion_phase_code.copy_(
        torch.tensor(
            [
                wait_env.FULL_A_MOTION_SWING_PHASE_INDEX,
                wait_env.FULL_A_MOTION_PREPARE_PHASE_INDEX,
            ],
            dtype=torch.long,
        )
    )
    env._full_a_owner_valid_bits = torch.zeros(
        (env.num_envs, wait_env.portable_reward.OWNER_COUNT), dtype=torch.long
    )
    env._full_a_owner_fault_bits = torch.zeros_like(
        env._full_a_owner_valid_bits
    )
    env._full_a_owner_fact_f32 = torch.zeros(
        (
            env.num_envs,
            wait_env.portable_reward.OWNER_COUNT,
            wait_env.portable_reward.OWNER_FACT_F32_WIDTH,
        ),
        dtype=torch.float64,
    )
    env._full_a_selected_contact_event = torch.zeros(
        env.num_envs, dtype=torch.bool
    )
    env._full_a_r03_present = torch.zeros(env.num_envs, dtype=torch.bool)
    env._full_a_r06_payment_event = torch.zeros(env.num_envs, dtype=torch.bool)

    reward, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    baseline = env._fullmdp_paddle_weights * env.step_dt
    torch.testing.assert_close(
        terms[0, 20:24], baseline * 4.0, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        terms[1, 20:24], baseline, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        reward, terms.sum(dim=1), rtol=0.0, atol=0.0
    )


def test_reward28_generic_body_terms_exclude_only_the_held_wrist():
    env = _fixed_reward_env()
    _reward, baseline = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)

    # Body ordinal two stands in for the configured held wrist.  Perturb every
    # generic body channel without changing the actual official-site tuple.
    env.sim.data.xpos[:, 2] += 100.0
    env.sim.data.xquat[:, 2] = torch.tensor(
        [0.0, 1.0, 0.0, 0.0], dtype=torch.float64
    )
    env.sim.data.cvel[:, 2] += 100.0
    _reward, wrist_only = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    assert torch.equal(wrist_only[:, 14:20], baseline[:, 14:20])

    env.sim.data.xpos[:, 0, 0] += 1.0
    env._refresh_aligned_teacher_body_pose()
    _reward, non_wrist_changed = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(
        env
    )
    assert not torch.equal(non_wrist_changed[:, 16], baseline[:, 16])


def test_reward28_ready_face_is_raw_y_but_selected_face_uses_mount_sign():
    env = _fixed_reward_env()
    env.full_a_mode = True
    env._full_a_owner_valid_bits = torch.zeros(
        (env.num_envs, wait_env.portable_reward.OWNER_COUNT), dtype=torch.long
    )
    env._full_a_owner_fault_bits = torch.zeros_like(
        env._full_a_owner_valid_bits
    )
    env._full_a_owner_fact_f32 = torch.zeros(
        (
            env.num_envs,
            wait_env.portable_reward.OWNER_COUNT,
            wait_env.portable_reward.OWNER_FACT_F32_WIDTH,
        ),
        dtype=torch.float64,
    )
    env._full_a_selected_contact_event = torch.zeros(
        env.num_envs, dtype=torch.bool
    )
    env._full_a_r03_present = torch.zeros(env.num_envs, dtype=torch.bool)
    env._full_a_r06_payment_event = torch.zeros(
        env.num_envs, dtype=torch.bool
    )
    env._fullmdp_mount_normal_sign.fill_(-1.0)
    env._epoch_task_valid.zero_()
    env._aligned_teacher_racket_signed_normal_w[:] = torch.tensor(
        [0.0, 1.0, 0.0], dtype=torch.float64
    )
    actual_pos = (
        env._aligned_teacher_racket_site_pos_w - env.env.scene.env_origins
    ).clone()
    actual_velocity = env._aligned_teacher_racket_site_lin_vel_w.clone()
    actual_raw_normal = torch.tensor(
        [[0.0, 1.0, 0.0]], dtype=torch.float64
    ).repeat(env.num_envs, 1)
    actual_long_axis = env._aligned_teacher_racket_long_axis_w.clone()
    env._full_a_racket_kinematics = lambda: (
        actual_pos,
        actual_velocity,
        actual_raw_normal,
        actual_long_axis,
    )
    _reward, ready_terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    assert ready_terms[:, 22].equal(
        (env._fullmdp_paddle_weights[2] * env.step_dt).expand(env.num_envs)
    )

    env._epoch_task_valid.fill_(True)
    env._aligned_teacher_racket_signed_normal_w.neg_()
    _reward, selected_terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(
        env
    )
    assert torch.equal(selected_terms[:, 22], ready_terms[:, 22])


def test_relative_body_rewards_use_prior_anchor_cache_then_align_next_tick():
    env = _fixed_reward_env()
    dtype = env._teacher_body_pos.dtype
    teacher_pos = torch.tensor(
        [
            [[0.2, -0.4, 0.6], [0.0, 0.0, 0.8], [-0.3, 0.5, 1.0]],
            [[0.2, -0.4, 0.6], [0.0, 0.0, 0.8], [-0.3, 0.5, 1.0]],
        ],
        dtype=dtype,
    )
    teacher_quat = torch.zeros((2, 3, 4), dtype=dtype)
    teacher_quat[..., 0] = 1.0
    env._teacher_body_pos = teacher_pos
    env._teacher_body_quat = teacher_quat
    env._aligned_teacher_body_pos = teacher_pos.clone()
    env._aligned_teacher_body_quat = teacher_quat.clone()
    paddle_offset = torch.tensor(
        [[0.25, -0.10, 0.15], [0.25, -0.10, 0.15]], dtype=dtype
    )
    env._teacher_racket_site_pos_w = teacher_pos[:, 1] + paddle_offset
    env._teacher_racket_site_lin_vel_w = torch.tensor(
        [[0.6, -0.2, 0.1], [0.6, -0.2, 0.1]], dtype=dtype
    )
    env._teacher_racket_signed_normal_w = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    env._teacher_racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype
    )
    env.sim.data.xpos[0].copy_(teacher_pos[0] + torch.tensor([0.4, -0.3, 0.0]))
    env.sim.data.xquat[0].copy_(teacher_quat[0])

    yaw = torch.tensor(torch.pi / 2.0, dtype=dtype)
    yaw_quat = torch.tensor(
        [torch.cos(yaw / 2.0), 0.0, 0.0, torch.sin(yaw / 2.0)], dtype=dtype
    )
    anchor = env._fullmdp_anchor_index
    offsets = teacher_pos[1] - teacher_pos[1, anchor]
    env.sim.data.xpos[1].copy_(
        teacher_pos[1, anchor]
        + _test_quat_apply(yaw_quat.expand_as(teacher_quat[1]), offsets)
    )
    env.sim.data.xquat[1].copy_(yaw_quat.expand_as(teacher_quat[1]))

    _, first_terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    first_raw = first_terms[:, 14:20] / (
        env._fullmdp_dense_weights * env.step_dt
    )
    assert first_raw[0, 0] < 1.0
    assert first_raw[0, 1] == 1.0
    assert first_raw[0, 2] < 1.0
    assert first_raw[0, 3] == 1.0
    assert first_raw[1, 0] == 1.0
    assert first_raw[1, 1] < 1.0
    assert first_raw[1, 2] < 1.0
    assert first_raw[1, 3] < 1.0

    env._refresh_aligned_teacher_body_pose()
    expected_pos, expected_quat = _independent_aligned_teacher(
        env, env.sim.data.xpos, env.sim.data.xquat
    )
    assert torch.equal(env._aligned_teacher_body_pos, expected_pos)
    assert torch.equal(env._aligned_teacher_body_quat, expected_quat)
    row_yaw = torch.stack(
        (torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype), yaw_quat)
    )
    expected_anchor = env.sim.data.xpos[:, anchor].clone()
    expected_anchor[:, 2] = teacher_pos[:, anchor, 2]
    torch.testing.assert_close(
        env._aligned_teacher_racket_site_pos_w,
        expected_anchor + _test_quat_apply(row_yaw, paddle_offset),
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        env._aligned_teacher_racket_site_lin_vel_w,
        _test_quat_apply(row_yaw, env._teacher_racket_site_lin_vel_w),
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        env._aligned_teacher_racket_signed_normal_w,
        _test_quat_apply(row_yaw, env._teacher_racket_signed_normal_w),
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        env._aligned_teacher_racket_long_axis_w,
        _test_quat_apply(row_yaw, env._teacher_racket_long_axis_w),
        rtol=0.0,
        atol=1.0e-15,
    )
    _, second_terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    second_raw = second_terms[:, 14:20] / (
        env._fullmdp_dense_weights * env.step_dt
    )
    assert second_raw[0, 0] < 1.0
    assert second_raw[0, 1] == 1.0
    assert torch.equal(second_raw[0, 2:4], torch.ones(2, dtype=dtype))
    assert second_raw[1, 0] == 1.0
    assert second_raw[1, 1] < 1.0
    assert torch.equal(second_raw[1, 2:4], torch.ones(2, dtype=dtype))


def test_full_a_teacher_target_does_not_reanchor_to_live_torso():
    env = _fixed_reward_env()
    env.full_a_mode = True
    env._refresh_aligned_teacher_body_pose()
    before = (
        env._aligned_teacher_body_pos.clone(),
        env._aligned_teacher_body_quat.clone(),
        env._aligned_teacher_racket_site_pos_w.clone(),
        env._aligned_teacher_racket_site_lin_vel_w.clone(),
        env._aligned_teacher_racket_signed_normal_w.clone(),
        env._aligned_teacher_racket_long_axis_w.clone(),
    )
    env.sim.data.xpos.add_(torch.tensor([3.0, -2.0, 0.5]))
    env.sim.data.xquat.zero_()
    env.sim.data.xquat[..., 3] = 1.0
    env._refresh_aligned_teacher_body_pose()
    after = (
        env._aligned_teacher_body_pos,
        env._aligned_teacher_body_quat,
        env._aligned_teacher_racket_site_pos_w,
        env._aligned_teacher_racket_site_lin_vel_w,
        env._aligned_teacher_racket_signed_normal_w,
        env._aligned_teacher_racket_long_axis_w,
    )
    for expected, actual in zip(before, after):
        assert torch.equal(actual, expected)


def test_reward28_two_transition_moving_anchor_uses_prior_alignment_cache():
    env = _fixed_reward_env()
    dtype = env._teacher_body_pos.dtype
    anchor = env._fullmdp_anchor_index
    non_wrist = env._fullmdp_upper_non_wrist_body_indices
    identity = torch.zeros((2, 3, 4), dtype=dtype)
    identity[..., 0] = 1.0
    teacher7 = torch.tensor(
        [
            [[-0.35, 0.20, 0.85], [0.00, 0.00, 1.00], [0.30, -0.25, 1.15]],
            [[-0.25, 0.15, 0.90], [0.05, -0.05, 1.05], [0.35, -0.20, 1.20]],
        ],
        dtype=dtype,
    )

    def rigid_pose(teacher_pos, yaw, xy_shift):
        half = 0.5 * yaw
        yaw_quat = torch.stack(
            (
                torch.cos(half),
                torch.zeros_like(half),
                torch.zeros_like(half),
                torch.sin(half),
            ),
            dim=1,
        )
        expanded = yaw_quat[:, None, :].expand_as(identity)
        live_anchor = teacher_pos[:, anchor].clone()
        live_anchor[:, :2] += xy_shift
        live_pos = live_anchor[:, None] + _test_quat_apply(
            expanded, teacher_pos - teacher_pos[:, anchor, None]
        )
        live_quat = _test_quat_mul(expanded, identity)
        return live_pos, live_quat

    def configured_position_terms(aligned, actual, raw_teacher):
        relative_error = torch.sum(
            torch.square(aligned[:, non_wrist] - actual[:, non_wrist]), dim=-1
        ).mean(-1)
        relative = (
            torch.exp(
                -relative_error
                / wait_env.FULLMDP_DENSE_REWARD_SPECS[2].std**2
            )
            * env._fullmdp_dense_weights[2]
            * env.step_dt
        )
        anchor_error = torch.sum(
            torch.square(raw_teacher[:, anchor] - actual[:, anchor]), dim=-1
        )
        global_anchor = (
            torch.exp(
                -anchor_error
                / wait_env.FULLMDP_DENSE_REWARD_SPECS[0].std**2
            )
            * env._fullmdp_dense_weights[0]
            * env.step_dt
        )
        return relative, global_anchor

    env._teacher_body_pos = teacher7.clone()
    env._teacher_body_quat = identity.clone()
    env._teacher_racket_site_pos_w = teacher7[:, anchor] + torch.tensor(
        [0.22, -0.08, 0.12], dtype=dtype
    )
    env._teacher_racket_site_lin_vel_w.zero_()
    x7, q7 = rigid_pose(
        teacher7,
        torch.tensor([0.40, -0.30], dtype=dtype),
        torch.tensor([[0.35, -0.25], [-0.20, 0.30]], dtype=dtype),
    )
    env.sim.data.xpos.copy_(x7)
    env.sim.data.xquat.copy_(q7)
    env._refresh_aligned_teacher_body_pose()
    aligned7 = env._aligned_teacher_body_pos.clone()
    torch.testing.assert_close(aligned7, x7, rtol=0.0, atol=2.0e-15)

    # Transition 7->8: post-physics X8 is scored against the A7 cache that was
    # published with the action, not a counterfactual realignment at X8.
    x8, q8 = rigid_pose(
        teacher7,
        torch.tensor([0.75, -0.55], dtype=dtype),
        torch.tensor([[0.52, -0.10], [-0.05, 0.48]], dtype=dtype),
    )
    env.sim.data.xpos.copy_(x8)
    env.sim.data.xquat.copy_(q8)
    _reward8, terms8 = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    expected_relative8, expected_anchor8 = configured_position_terms(
        aligned7, x8, teacher7
    )
    torch.testing.assert_close(
        terms8[:, 16], expected_relative8, rtol=0.0, atol=2.0e-15
    )
    torch.testing.assert_close(
        terms8[:, 14], expected_anchor8, rtol=0.0, atol=2.0e-15
    )
    counterfactual7, _ = _independent_aligned_teacher(env, x8, q8)
    assert not torch.allclose(
        terms8[:, 16],
        configured_position_terms(counterfactual7, x8, teacher7)[0],
    )

    # The observation boundary advances both the measured teacher and its one
    # alignment cache.  T8 changes the anchor and the relative body geometry.
    teacher8 = teacher7 + torch.tensor(
        [
            [[0.10, -0.02, 0.03], [0.04, 0.03, 0.08], [-0.06, 0.08, -0.02]],
            [[-0.08, 0.05, 0.02], [0.02, -0.04, 0.06], [0.07, -0.03, 0.01]],
        ],
        dtype=dtype,
    )
    env._teacher_body_pos = teacher8
    env._teacher_racket_site_pos_w = teacher8[:, anchor] + torch.tensor(
        [0.18, -0.04, 0.16], dtype=dtype
    )
    env._refresh_aligned_teacher_body_pose()
    aligned8 = env._aligned_teacher_body_pos.clone()
    independent8, independent_quat8 = _independent_aligned_teacher(env, x8, q8)
    torch.testing.assert_close(aligned8, independent8, rtol=0.0, atol=2.0e-15)
    torch.testing.assert_close(
        env._aligned_teacher_body_quat,
        independent_quat8,
        rtol=0.0,
        atol=2.0e-15,
    )

    # Transition 8->9 repeats the same chronology with a new nonzero XY+yaw
    # anchor motion and the genuinely different T8 reference.
    x9, q9 = rigid_pose(
        teacher8,
        torch.tensor([-0.20, 0.25], dtype=dtype),
        torch.tensor([[0.15, 0.42], [-0.38, 0.12]], dtype=dtype),
    )
    env.sim.data.xpos.copy_(x9)
    env.sim.data.xquat.copy_(q9)
    _reward9, terms9 = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    expected_relative9, expected_anchor9 = configured_position_terms(
        aligned8, x9, teacher8
    )
    torch.testing.assert_close(
        terms9[:, 16], expected_relative9, rtol=0.0, atol=2.0e-15
    )
    torch.testing.assert_close(
        terms9[:, 14], expected_anchor9, rtol=0.0, atol=2.0e-15
    )
    counterfactual8, _ = _independent_aligned_teacher(env, x9, q9)
    assert not torch.allclose(
        terms9[:, 16],
        configured_position_terms(counterfactual8, x9, teacher8)[0],
    )


def test_body_com_velocity_transform_matches_native_jacobian_oracle():
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <inertial pos="0.10 0 0" mass="2" diaginertia=".10 .11 .12"/>
              <body name="child" pos="0.70 0.20 0.10">
                <joint name="hinge" type="hinge" axis="0 0 1"/>
                <inertial pos="0.25 -0.15 0.05" mass="1"
                          diaginertia=".05 .06 .07"/>
                <geom type="sphere" size=".05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qvel[:] = np.asarray([0.4, -0.2, 0.1, 0.3, 0.5, -0.4, 1.2])
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "child")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    expected_lin = jacp @ data.qvel
    expected_ang = jacr @ data.qvel

    cvel = torch.as_tensor(data.cvel[body_id]).reshape(1, 1, 6)
    xipos = torch.as_tensor(data.xipos[body_id]).reshape(1, 1, 3)
    root = int(model.body_rootid[body_id])
    subtree_root = torch.as_tensor(data.subtree_com[root]).reshape(1, 1, 3)
    actual_lin, actual_ang = (
        wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_from_cvel(
            torch, cvel, xipos, subtree_root
        )
    )
    assert np.allclose(actual_lin.numpy()[0, 0], expected_lin, rtol=0.0, atol=1e-12)
    assert np.allclose(actual_ang.numpy()[0, 0], expected_ang, rtol=0.0, atol=1e-12)
    assert not np.allclose(data.cvel[body_id, 3:], expected_lin, rtol=0.0, atol=1e-8)


def test_host_termination_bootstraps_only_pure_time_limit_rows():
    num_envs = 7
    env = SimpleNamespace(
        _torch=torch,
        episode_length_buf=torch.tensor([0, 10, 10, 0, 0, 0, 0]),
        max_episode_length=10,
        _cur_robot_table=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        _cur_table_keepout=torch.tensor([False, False, False, False, False, False, True]),
        _qdes_guard_terminal=torch.tensor(
            [False, False, False, False, True, False, False]
        ),
        _actual_hard_edge_latch=torch.zeros(7, dtype=torch.bool),
        _qdes_guard_intervention=torch.zeros(7, dtype=torch.bool),
    )
    proj_g = torch.tensor(
        [[0.0, 0.0, -1.0]] * 2
        + [[0.0, 0.0, 0.0]]
        + [[0.0, 0.0, -1.0]] * 4
    )
    base_pos = torch.ones((num_envs, 3))
    base_pos[3, 2] = 0.49
    qdes = torch.zeros((num_envs, 31))
    qdes[4, 0] = float("nan")

    terminated, truncated, bits, resolved_table = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_termination(
            env, {"proj_g": proj_g, "base_pos": base_pos}, qdes
        )
    )
    assert torch.equal(
        terminated, torch.tensor([False, False, True, True, True, True, True])
    )
    assert torch.equal(
        truncated, torch.tensor([False, True, False, False, False, False, False])
    )
    assert torch.equal(bits, torch.tensor([0, 1, 2, 4, 8, 16, 16]))
    assert torch.equal(
        resolved_table,
        torch.tensor([False, False, False, False, False, True, False]),
    )


def test_step_delegates_raw_action_to_single_owner_and_preserves_nonfinite_evidence():
    captured = {}
    previous = torch.tensor([[-0.25, 0.30], [0.20, -0.40]])

    def advance(actions):
        captured["raw_actions"] = actions.clone()
        requested = env.q_ready.unsqueeze(0) + env.act_scale * actions
        finite = torch.isfinite(requested)
        captured["safe_actions"] = torch.where(finite, actions, previous)
        return {}, torch.zeros(2), requested

    def terminate(_st, requested_qdes):
        captured["requested_qdes"] = requested_qdes.clone()
        false = torch.zeros(2, dtype=torch.bool)
        return false, false.clone(), torch.zeros(2, dtype=torch.long), false.clone()

    env = SimpleNamespace(
        _torch=torch,
        device=torch.device("cpu"),
        q_ready=torch.zeros(2),
        act_scale=torch.ones(2),
        actions=previous,
        _advance_plant=advance,
        _latch_post_forward_resolved_table_contacts=lambda _census=None: None,
        _latch_post_forward_table_keepout=lambda: None,
        _state=lambda: {},
        _fullmdp_termination=terminate,
        _fullmdp_reward=lambda: (
            torch.zeros(2),
            torch.zeros(2, wait_env.reward_contract.REWARD_TERM_COUNT),
        ),
        last_terminal_bits=torch.zeros(2, dtype=torch.long),
        reset_generation=torch.ones(2, dtype=torch.long),
        _all_env_ids=torch.arange(2),
        _serve=lambda _ids: None,
        sim=SimpleNamespace(forward=lambda: None),
        _cap_ok=False,
        _refresh_aligned_teacher_body_pose=lambda: None,
        _compute_obs=lambda: None,
        get_observations=lambda: {},
    )
    incoming = torch.tensor([[float("nan"), 1.0], [0.5, float("inf")]])
    wait_env.FullMdpInitialWaitVecEnv.step(env, incoming)

    torch.testing.assert_close(captured["raw_actions"], incoming, equal_nan=True)
    assert torch.equal(
        captured["safe_actions"],
        torch.tensor([[-0.25, 1.0], [0.5, -0.40]]),
    )
    assert torch.isnan(captured["requested_qdes"][0, 0])
    assert torch.isinf(captured["requested_qdes"][1, 1])


def _host_full_a_lifecycle_env():
    n = 2
    qpos = torch.zeros((n, 20))
    qpos[:, 3] = 1.0
    qvel = torch.zeros((n, 12))

    def centre_question_stub(**kwargs):
        count = int(kwargs["base_position_scene"].shape[0])
        task = torch.zeros((count, P.TASK_F32_WIDTH))
        task[:, 0] = 0.04
        task[:, 1] = 1.0
        task[:, 2] = 0.02
        task[:, 3] = 0.04
        task[:, 4] = 0.02
        task[:, 5:8] = torch.tensor([0.0, -0.76, 1.05])
        task[:, 8:11] = torch.tensor([-3.3, 0.0, -0.05])
        task[:, 11:14] = torch.tensor([0.0, 1.0, 0.0])
        task[:, 14:17] = task[:, 5:8]
        task[:, 17:20] = task[:, 8:11]
        task[:, 20] = 1.0
        task[:, 32:35] = torch.tensor([0.1, -0.76, 1.1])
        task[:, 35] = 1.0
        task[:, 39:42] = torch.tensor([-3.3, 0.0, 0.4])
        return {
            "task_f32": task,
            "launch_state_f32": task[:, 32:45],
            "ttc_ticks": torch.full((count,), 2, dtype=torch.long),
            "launch_horizon_ticks": torch.ones(count, dtype=torch.long),
            "teacher_rate": torch.ones(count),
            "scaled_t_hit_s": torch.full((count,), 0.02),
            "scaled_t_cycle_s": torch.full((count,), 0.04),
            "pre_swing_wait_s": torch.full((count,), 0.02),
            "teacher_source_to_task_yaw_wxyz": torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            ).expand(count, 4),
            "teacher_source_to_task_translation_scene": torch.zeros(
                count, 3
            ),
        }

    env = SimpleNamespace(
        _torch=torch,
        num_envs=n,
        device=torch.device("cpu"),
        q_ready=torch.zeros(31),
        action_offset=torch.zeros(31),
        qpos_init=torch.zeros(20),
        qvel_init=torch.zeros(12),
        serve_pos_lo=torch.tensor([1.95, -0.85, 0.64]),
        serve_pos_hi=torch.tensor([2.05, -0.70, 0.72]),
        serve_vel_lo=torch.tensor([-3.5, -0.05, 0.35]),
        serve_vel_hi=torch.tensor([-3.1, 0.05, 0.55]),
        hope_to_scene=torch.tensor([0.0, 0.0, 0.76]),
        env=SimpleNamespace(
            scene=SimpleNamespace(env_origins=torch.zeros((n, 3)))
        ),
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=qpos,
                qvel=qvel,
                xpos=qpos[:, 7:10].unsqueeze(1),
                site_xpos=torch.tensor(
                    [[[0.0, -0.76, 1.05]], [[0.1, -0.76, 1.05]]]
                ),
                site_xmat=torch.eye(3).reshape(1, 1, 3, 3).repeat(n, 1, 1, 1),
                cvel=torch.zeros((n, 1, 6)),
                subtree_com=torch.zeros((n, 1, 3)),
            )
        ),
        racket_sid=0,
        _fullmdp_ball_body_id=0,
        _fullmdp_racket_body_id=0,
        _fullmdp_racket_root_id=0,
        _fullmdp_racket_long_axis_local=torch.tensor(
            wait_env.racket_contact_geometry.RACKET_BUTT_TO_BLADE_AXIS_LOCAL
        ),
        _fullmdp_mount_normal_sign=torch.ones(n),
        root_qadr=0,
        b_q=7,
        b_v=0,
        ball_age_buf=torch.zeros(n, dtype=torch.long),
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        common_step_counter=0,
        step_dt=0.02,
        full_a_mode=True,
        _fullmdp_initialized=True,
        _all_env_ids=torch.arange(n),
        _epoch_phase=torch.zeros(n, dtype=torch.long),
        _epoch_task_valid=torch.zeros(n, dtype=torch.bool),
        _epoch_selected=torch.zeros(n, dtype=torch.bool),
        _epoch_launch_succeeded=torch.zeros(n, dtype=torch.bool),
        reset_generation=torch.ones(n, dtype=torch.long),
        _cur_touched=torch.zeros(n),
        _cur_robot_table=torch.zeros(n),
        _cur_table_keepout=torch.zeros(n, dtype=torch.bool),
        _con_geom=torch.zeros((1, 2), dtype=torch.long),
        _con_idx=torch.arange(1),
        _nacon=torch.zeros(1, dtype=torch.long),
        _con_world=torch.zeros(1, dtype=torch.long),
        _ball_gid=0,
        _geom_class=torch.zeros(2, dtype=torch.int8),
        _robot_table_ok=False,
        _table_gid=-1,
        _is_robot_geom=torch.zeros(2, dtype=torch.bool),
        _contact_ball_racket_by_world=torch.zeros(n),
        _contact_ball_table_by_world=torch.zeros(n),
        _contact_robot_table_by_world=torch.zeros(n),
        _contact_zero_total=torch.zeros((), dtype=torch.long),
        cfg=SimpleNamespace(
            ball_dead_z_hope=-0.35,
            ball_dead_x_lo_hope=-1.2,
            ball_dead_x_hi_hope=3.4,
        ),
        _full_a_catalog=SimpleNamespace(
            fresh_action=SimpleNamespace(
                action_slot=0,
                action_uid=5527597793770800,
                mount_normal_sign=1,
                base_spawn_center_w_xy_m=(-0.19223234, 0.28527881),
            )
        ),
        _full_a_cadence=SimpleNamespace(
            first_reveal_tick=2,
            cadence_ticks=293,
            episode_horizon_ticks=1500,
            # Tests-only shifted schedule: preserve the production count of
            # four opportunities while making focused lifecycle tests cheap.
            reference_due_ticks=(2, 295, 588, 881),
        ),
        _full_a_teacher=SimpleNamespace(contact_reference_root_z_scene=0.9),
        _full_a_question_builder=centre_question_stub,
        _full_a_landing_target_xy=torch.tensor([2.55, 0.0]),
        _full_a_target_positive_x=True,
        _full_a_table_bounds=(0.50, 3.24, -0.7625, 0.7625),
        _full_a_net_x=1.87,
        _full_a_net_clear_z=0.94,
        _full_a_landing_plane_z=0.78,
        _full_a_placement_broad_sigma=wait_env.FULL_A_PLACEMENT_BROAD_SIGMA_M,
        _full_a_placement_narrow_sigma=0.04,
        _full_a_recovery_joint_limit_ok=lambda: torch.ones(n, dtype=torch.bool),
        _actual_hard_edge_latch=torch.zeros(n, dtype=torch.bool),
        _qdes_guard_intervention=torch.zeros(n, dtype=torch.bool),
    )
    env._clear_lifecycle = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._clear_lifecycle, env
    )
    for name in (
        "_full_a_reveal_rows",
        "_full_a_launch_rows",
        "_full_a_park_rows",
        "_full_a_prepare_step",
        "_full_a_settle_reveal",
        "_full_a_reset_cadence_rows",
        "_full_a_racket_kinematics",
        "_full_a_publish_r03_fact",
        "_full_a_publish_physical_fact",
        "_full_a_settle_outcome",
        "_full_a_recovery_clock",
        "_full_a_completed_action_epoch",
    ):
        setattr(
            env,
            name,
            MethodType(getattr(wait_env.FullMdpInitialWaitVecEnv, name), env),
        )
    wait_env.FullMdpInitialWaitVecEnv._initialize_full_a_state(env)
    wait_env.FullMdpInitialWaitVecEnv._install_full_a_physical_spawn(
        env, env._all_env_ids
    )
    # Step-order host tapes replace both tracked-state consumers.  Thread one
    # explicit ephemeral object through those replacements without inventing
    # fake body tensors in this lifecycle-only fixture.
    env._fullmdp_tracked_body_kinematics = lambda: object()
    # Most focused lifecycle tests start at a deliberately admitted D05 due
    # boundary.  Cadence-specific tests reset these fields to the production
    # 295-tick balance prefix explicitly.
    env._full_a_next_reveal_tick.fill_(2)
    env.episode_length_buf.fill_(1)
    return env


def _prepare_and_settle(env, dones=None):
    """Advance the two-stage reveal contract for lifecycle setup only."""

    scheduled_due, launch, missed = env._full_a_prepare_step()
    # Production settles after the integrated transition, whose plant owner
    # has advanced this boundary exactly once.
    env.common_step_counter += 1
    if dones is None:
        dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reveal, due, deferred, terminal_overlap = env._full_a_settle_reveal(
        scheduled_due, dones
    )
    assert not terminal_overlap.any()
    return reveal, launch, due, deferred, missed


@pytest.mark.parametrize("mutation", (
    "missing", "wrong_type", "wrong_dtype", "wrong_device",
    "nonfinite", "invalid_positive", "invalid_integer",
))
def test_full_a_builder_failure_is_bitwise_zero_write(mutation):
    env = _host_full_a_lifecycle_env()
    fields = (
        "_full_a_scheduled_ordinal", "_full_a_next_reveal_tick",
        "_epoch_phase", "_epoch_clock_ticks", "_epoch_task_f32",
    )
    before = {name: getattr(env, name).clone() for name in fields}
    scheduled_due, _launch, _missed = env._full_a_prepare_step()
    env.common_step_counter += 1
    original = env._full_a_question_builder
    def fail(**kwargs):
        question = original(**kwargs)
        if mutation == "missing":
            del question["launch_state_f32"]
        elif mutation == "wrong_type":
            question["teacher_rate"] = question["teacher_rate"].tolist()
        elif mutation == "wrong_dtype":
            question["teacher_rate"] = question["teacher_rate"].double()
        elif mutation == "wrong_device":
            if not torch.cuda.is_available():
                pytest.skip("CUDA required for cross-device staged payload")
            question["teacher_rate"] = question["teacher_rate"].cuda()
        elif mutation == "nonfinite":
            question["teacher_rate"] = question["teacher_rate"].clone()
            question["teacher_rate"][0] = torch.nan
        elif mutation == "invalid_positive":
            question["teacher_rate"] = torch.zeros_like(question["teacher_rate"])
        else:
            question["ttc_ticks"] = question["ttc_ticks"].float()
        return question
    env._full_a_question_builder = fail
    with pytest.raises(RuntimeError, match="portable centre"):
        env._full_a_settle_reveal(scheduled_due, torch.zeros(env.num_envs, dtype=torch.bool))
    for name in fields:
        assert torch.equal(getattr(env, name), before[name]), name


def test_full_a_racket_kinematics_materializes_one_official_site_tuple():
    env = _host_full_a_lifecycle_env()
    dtype = env.sim.data.site_xpos.dtype
    origins = torch.tensor(
        [[6.0, -12.0, 0.0], [-6.0, 12.0, 0.0]], dtype=dtype
    )
    scene_position = torch.tensor(
        [[0.3, -0.5, 1.1], [-0.2, 0.4, 0.9]], dtype=dtype
    )
    env.env.scene.env_origins = origins
    env.sim.data.site_xpos[:, 0] = scene_position + origins
    rotations = torch.stack(
        (
            torch.eye(3, dtype=dtype),
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=dtype,
            ),
        )
    )
    env.sim.data.site_xmat[:, 0] = rotations
    env.sim.data.subtree_com[:, 0] = origins + torch.tensor(
        [[0.1, -0.2, 0.7], [0.2, 0.1, 0.8]], dtype=dtype
    )
    angular = torch.tensor(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]], dtype=dtype
    )
    linear_at_subtree = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=dtype
    )
    env.sim.data.cvel[:, 0] = torch.cat((angular, linear_at_subtree), dim=1)

    position, velocity, raw_normal, long_axis = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_racket_kinematics(env)
    )
    expected_velocity = linear_at_subtree + torch.cross(
        angular,
        env.sim.data.site_xpos[:, 0] - env.sim.data.subtree_com[:, 0],
        dim=1,
    )

    torch.testing.assert_close(position, scene_position, rtol=0.0, atol=5.0e-7)
    assert torch.equal(velocity, expected_velocity)
    assert torch.equal(raw_normal, rotations[:, :, 1])
    torch.testing.assert_close(
        long_axis,
        torch.matmul(rotations, env._fullmdp_racket_long_axis_local),
        rtol=0.0,
        atol=1.0e-7,
    )


def _legacy_finish_recovery_reference(env, terminated, truncated):
    """V4 completion semantics retained as a healthy-tape parity oracle."""

    torch_module = env._torch
    outcome_settled = env._epoch_phase.eq(wait_env.FULL_A_PHASE_OUTCOME_SETTLED)
    invalid_outcome = outcome_settled & env._full_a_outcome_code.eq(
        wait_env.FULL_A_OUTCOME_INVALID
    )
    recovery = outcome_settled & ~invalid_outcome
    age = int(env.common_step_counter) - env._epoch_clock_ticks[:, 3]
    expected_cells = (
        wait_env.FULL_A_RECOVERY_END_AGE_TICK
        - wait_env.FULL_A_RECOVERY_START_AGE_TICK
        + 1
    )
    complete_window = (
        env._full_a_recovery_expected_count.eq(expected_cells)
        & env._full_a_recovery_eligible_count.eq(expected_cells)
        & env._full_a_recovery_last_age.eq(
            wait_env.FULL_A_RECOVERY_END_AGE_TICK
        )
        & ~env._full_a_recovery_sticky_fault
    )
    completion_due = (
        recovery
        & age.ge(wait_env.FULL_A_RECOVERY_END_AGE_TICK)
        & ~terminated
        & ~truncated
    )
    if bool((completion_due & ~complete_window).any()):
        raise RuntimeError("legacy oracle received a fault tape")
    terminal, success, failure, timeout = wait_env.portable_outcome.recovery_status(
        torch=torch_module,
        recovering=recovery,
        age=age,
        terminated=terminated,
        truncated=truncated,
        ready_seen=env._full_a_recovery_ready_seen,
        end_age=wait_env.FULL_A_RECOVERY_END_AGE_TICK,
    )
    terminal |= invalid_outcome
    failure |= invalid_outcome
    retired = terminal & ~terminated & ~truncated
    env._epoch_phase.copy_(
        torch_module.where(
            retired,
            torch_module.full_like(
                env._epoch_phase, wait_env.FULL_A_PHASE_RETIRED
            ),
            env._epoch_phase,
        )
    )
    return terminal, success, failure, timeout


def _fixed_prepare_step_tape(num_envs):
    """Build a CPU-only mixed-phase tape for old/new prepare-step parity."""

    row = torch.arange(num_envs, dtype=torch.long)
    phase_codes = torch.tensor(
        [
            wait_env.FULL_A_PHASE_IDLE,
            wait_env.FULL_A_PHASE_REVEAL_COMMITTED,
            wait_env.FULL_A_PHASE_REVEAL_COMMITTED,
            wait_env.FULL_A_PHASE_OUTCOME_SETTLED,
            wait_env.FULL_A_PHASE_RETIRED,
            wait_env.FULL_A_PHASE_LAUNCH_SETTLED,
        ],
        dtype=torch.long,
    )
    phase = phase_codes[row.remainder(len(phase_codes))].clone()
    clock = torch.full((num_envs, 5), -1, dtype=torch.long)
    clock[:, 2] = 9
    clock[row.remainder(len(phase_codes)).eq(1), 2] = 7
    clock[row.remainder(len(phase_codes)).eq(2), 2] = 6
    phase_kind = row.remainder(len(phase_codes))
    due = phase_kind.eq(0) | phase_kind.eq(4) | phase_kind.eq(5)
    env = SimpleNamespace(
        _torch=torch,
        episode_length_buf=torch.full((num_envs,), 7, dtype=torch.long),
        _full_a_cadence=SimpleNamespace(
            reference_due_ticks=(8, 301, 594, 887),
            cadence_ticks=293,
            episode_horizon_ticks=1500,
        ),
        _full_a_scheduled_ordinal=torch.full(
            (num_envs,), -1, dtype=torch.long
        ),
        _full_a_next_reveal_tick=torch.where(
            due,
            torch.full((num_envs,), 8, dtype=torch.long),
            torch.full((num_envs,), 9, dtype=torch.long),
        ),
        _epoch_phase=phase,
        _epoch_clock_ticks=clock,
        common_step_counter=7,
        _scene_marker=row.to(torch.float64),
        _event_counter=torch.zeros(num_envs, dtype=torch.long),
        _fault_bits=row.remainder(7).clone(),
        _park_count=torch.zeros(num_envs, dtype=torch.long),
        _row_trace=[[] for _ in range(num_envs)],
    )

    def record(ids, event):
        for index in ids.tolist():
            env._row_trace[index].append(event)

    def clear(ids):
        record(ids, "clear")
        env._epoch_phase[ids] = wait_env.FULL_A_PHASE_IDLE
        env._epoch_clock_ticks[ids] = -1
        env._fault_bits[ids] = 0

    def reveal(ids):
        record(ids, "reveal")
        env._epoch_phase[ids] = wait_env.FULL_A_PHASE_REVEAL_COMMITTED
        env._epoch_clock_ticks[ids, 2] = env.common_step_counter + 1
        env._event_counter[ids] += 1

    def launch(ids):
        record(ids, "launch")
        env._epoch_phase[ids] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
        env._scene_marker[ids] = 1000.0 + ids.to(env._scene_marker.dtype)
        env._event_counter[ids] += 1

    def park(ids):
        record(ids, "park")
        env._scene_marker[ids] = -1000.0 - ids.to(env._scene_marker.dtype)
        env._park_count[ids] += 1

    env._clear_lifecycle = clear
    env._full_a_reveal_rows = reveal
    env._full_a_launch_rows = launch
    env._full_a_park_rows = park
    return env


def test_prepare_then_settle_fixed_tape_freezes_start_state_and_commits_survivors():
    env = _fixed_prepare_step_tape(12)
    before = {
        name: getattr(env, name).clone()
        for name in (
            "_full_a_scheduled_ordinal",
            "_full_a_next_reveal_tick",
            "_epoch_phase",
            "_epoch_clock_ticks",
            "_fault_bits",
        )
    }
    scheduled_due, launch, missed = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    )
    kind = torch.arange(12).remainder(6)
    assert torch.equal(scheduled_due, kind.eq(0) | kind.eq(4) | kind.eq(5))
    assert torch.equal(launch, kind.eq(1))
    assert torch.equal(missed, kind.eq(2))
    for name in ("_full_a_scheduled_ordinal", "_full_a_next_reveal_tick", "_fault_bits"):
        assert torch.equal(getattr(env, name), before[name]), name
    assert torch.equal(
        env._epoch_phase[scheduled_due], before["_epoch_phase"][scheduled_due]
    )
    assert torch.equal(
        env._epoch_clock_ticks[scheduled_due],
        before["_epoch_clock_ticks"][scheduled_due],
    )
    assert not any("clear" in trace or "reveal" in trace for trace in env._row_trace)

    dones = torch.zeros(12, dtype=torch.bool)
    dones[0] = True
    # Row 5 was busy when due froze, naturally retires during the transition,
    # and must ACCEPT.  Row 11 stays busy and must DEFER.
    env._epoch_phase[5] = wait_env.FULL_A_PHASE_RETIRED
    final_reveal, final_due, final_deferred, terminal_overlap = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_settle_reveal(
            env, scheduled_due, dones
        )
    )
    assert not final_due[0] and not final_reveal[0] and not final_deferred[0]
    assert torch.equal(terminal_overlap, scheduled_due & dones)
    assert final_reveal[5]
    assert final_deferred[11]
    assert torch.equal(final_due, scheduled_due & ~dones)
    assert torch.equal(final_reveal | final_deferred, final_due)
    assert env._full_a_scheduled_ordinal[0] == -1
    assert env._full_a_next_reveal_tick[0] == 8
    assert env._row_trace[0] == ["park"]
    for index in final_reveal.nonzero(as_tuple=False).squeeze(-1).tolist():
        assert env._row_trace[index][-2:] == ["clear", "reveal"]


def test_two_stage_reveal_hot_paths_have_minimal_dynamic_shape_selections():
    source = ast.parse(Path(wait_env.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_full_a_prepare_step", "_full_a_settle_reveal"}
    }
    prepare_calls = [
        node
        for node in ast.walk(methods["_full_a_prepare_step"])
        if isinstance(node, ast.Call)
    ]
    settle_calls = [
        node
        for node in ast.walk(methods["_full_a_settle_reveal"])
        if isinstance(node, ast.Call)
    ]
    assert sum(
        isinstance(node.func, ast.Attribute) and node.func.attr == "nonzero"
        for node in prepare_calls
    ) == 2
    assert sum(
        isinstance(node.func, ast.Attribute) and node.func.attr == "nonzero"
        for node in settle_calls
    ) == 1
    assert sum(
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "_full_a_park_rows"
        for node in prepare_calls
    ) == 1


def test_park_constants_are_construction_cached_but_origins_remain_live(
    monkeypatch,
):
    env = _host_full_a_lifecycle_env()
    ids = torch.arange(env.num_envs)
    position_ptr = env._full_a_park_position_scene.data_ptr()
    quaternion_ptr = env._full_a_park_quaternion.data_ptr()
    expected_offset = (
        torch.tensor(wait_env.WAIT_BALL_PARK_HOPE, dtype=env.qpos_init.dtype)
        + env.hope_to_scene
    )
    expected_quaternion = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=env.qpos_init.dtype
    )

    def forbidden_constructor(*_args, **_kwargs):
        raise AssertionError("park hot path reconstructed an immutable tensor")

    for origins in (
        torch.tensor([[4.0, -3.0, 0.0], [-2.0, 5.0, 1.0]]),
        torch.tensor([[-7.0, 8.0, 2.0], [9.0, -6.0, -1.0]]),
    ):
        env.env.scene.env_origins.copy_(origins)
        env.sim.data.qpos[:, env.b_q : env.b_q + 7].fill_(99.0)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "tensor", forbidden_constructor)
            wait_env.FullMdpInitialWaitVecEnv._full_a_park_rows(env, ids)
        assert torch.equal(
            env.sim.data.qpos[:, env.b_q : env.b_q + 3],
            origins + expected_offset,
        )
        assert torch.equal(
            env.sim.data.qpos[:, env.b_q + 3 : env.b_q + 7],
            expected_quaternion.expand(env.num_envs, 4),
        )
        env._clear_lifecycle(ids)

    assert env._full_a_park_position_scene.data_ptr() == position_ptr
    assert env._full_a_park_quaternion.data_ptr() == quaternion_ptr
    assert env._full_a_park_position_scene.dtype == env.qpos_init.dtype
    assert env._full_a_park_quaternion.dtype == env.qpos_init.dtype
    assert env._full_a_park_position_scene.device == env.device
    assert env._full_a_park_quaternion.device == env.device


def test_park_hot_path_has_no_fixed_tensor_constructor():
    source = ast.parse(Path(wait_env.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_full_a_park_rows"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"tensor", "as_tensor", "new_tensor"}
        for node in ast.walk(method)
    )


def test_full_a_reset_serve_reuses_cache_and_matches_zero_origin_legacy_tape(
    monkeypatch,
):
    env = _host_full_a_lifecycle_env()
    ids = torch.tensor([1], dtype=torch.long)
    env.sim.data.qpos.copy_(
        torch.arange(env.sim.data.qpos.numel(), dtype=env.qpos_init.dtype).reshape_as(
            env.sim.data.qpos
        )
    )
    env.sim.data.qvel.copy_(
        -torch.arange(
            1,
            env.sim.data.qvel.numel() + 1,
            dtype=env.qvel_init.dtype,
        ).reshape_as(env.sim.data.qvel)
    )
    env.sim.data.qpos[ids, env.b_q : env.b_q + 7] = float("nan")
    env.sim.data.qvel[ids, env.b_v : env.b_v + 6] = float("inf")
    env.ball_age_buf.copy_(torch.tensor([17, -91], dtype=torch.long))

    # Reset must not consume or rewrite any chronology, event, or reason owner.
    env.common_step_counter = 37
    env.episode_length_buf.copy_(torch.tensor([11, 23], dtype=torch.long))
    env.reset_generation.copy_(torch.tensor([5, 8], dtype=torch.long))
    env._epoch_phase.copy_(
        torch.tensor(
            [wait_env.FULL_A_PHASE_LAUNCH_SETTLED, wait_env.FULL_A_PHASE_RETIRED],
            dtype=torch.long,
        )
    )
    env._full_a_selected_contact_event.copy_(torch.tensor([True, False]))
    env._full_a_owner_fault_bits[0, 2] = 9
    env.last_terminal_bits = torch.tensor([4, 16], dtype=torch.long)

    expected_qpos = env.sim.data.qpos.clone()
    expected_qvel = env.sim.data.qvel.clone()
    expected_ball_age = env.ball_age_buf.clone()
    legacy_park_scene = (
        torch.tensor(wait_env.WAIT_BALL_PARK_HOPE, dtype=env.qpos_init.dtype)
        + env.hope_to_scene
    )
    legacy_park_quaternion = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=env.qpos_init.dtype
    )
    expected_qpos[ids, env.b_q : env.b_q + 3] = legacy_park_scene
    expected_qpos[ids, env.b_q + 3 : env.b_q + 7] = legacy_park_quaternion
    expected_qvel[ids, env.b_v : env.b_v + 6] = 0.0
    expected_ball_age[ids] = 0
    untouched = {
        name: getattr(env, name).clone()
        for name in (
            "episode_length_buf",
            "reset_generation",
            "_epoch_phase",
            "_full_a_selected_contact_event",
            "_full_a_owner_fault_bits",
            "last_terminal_bits",
        )
    }
    rng_before = torch.random.get_rng_state().clone()
    position_ptr = env._full_a_park_position_scene.data_ptr()
    quaternion_ptr = env._full_a_park_quaternion.data_ptr()

    def forbidden_constructor(*_args, **_kwargs):
        raise AssertionError("Full-A reset reconstructed an immutable tensor")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", forbidden_constructor)
        patch.setattr(torch, "as_tensor", forbidden_constructor)
        wait_env.FullMdpInitialWaitVecEnv._serve(env, ids)

    assert torch.equal(env.sim.data.qpos, expected_qpos)
    assert torch.equal(env.sim.data.qvel, expected_qvel)
    assert torch.equal(env.ball_age_buf, expected_ball_age)
    assert env.common_step_counter == 37
    for name, value in untouched.items():
        assert torch.equal(getattr(env, name), value), name
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert env._full_a_park_position_scene.data_ptr() == position_ptr
    assert env._full_a_park_quaternion.data_ptr() == quaternion_ptr


def test_full_a_reset_serve_applies_distinct_live_origins_and_preserves_scene_view():
    env = _host_full_a_lifecycle_env()
    ids = torch.arange(env.num_envs)
    origins = torch.tensor(
        [[8.0, -5.0, 1.25], [-3.0, 7.0, -0.50]],
        dtype=env.qpos_init.dtype,
    )
    env.env.scene.env_origins.copy_(origins)
    env.sim.data.qpos[:, env.b_q : env.b_q + 7].fill_(99.0)
    env.sim.data.qvel[:, env.b_v : env.b_v + 6].fill_(-99.0)
    env.ball_age_buf.fill_(43)

    wait_env.FullMdpInitialWaitVecEnv._serve(env, ids)

    ball_position_w = env.sim.data.qpos[:, env.b_q : env.b_q + 3]
    assert torch.equal(
        ball_position_w,
        origins + env._full_a_park_position_scene,
    )
    # Full-A observations normalize world positions by these same live origins.
    assert torch.equal(
        ball_position_w - origins,
        env._full_a_park_position_scene.expand(env.num_envs, 3),
    )
    assert torch.equal(
        env.sim.data.qpos[:, env.b_q + 3 : env.b_q + 7],
        env._full_a_park_quaternion.expand(env.num_envs, 4),
    )
    assert torch.count_nonzero(
        env.sim.data.qvel[:, env.b_v : env.b_v + 6]
    ) == 0
    assert torch.count_nonzero(env.ball_age_buf) == 0


def test_legacy_wait_serve_keeps_pre_full_a_fallback_coordinates():
    env = _host_full_a_lifecycle_env()
    env.full_a_mode = False
    env._fullmdp_initialized = False
    del env._full_a_park_position_scene
    del env._full_a_park_quaternion
    ids = torch.arange(env.num_envs)
    env.env.scene.env_origins.copy_(
        torch.tensor([[9.0, 8.0, 7.0], [-6.0, -5.0, -4.0]])
    )
    expected_park = (
        torch.tensor(wait_env.WAIT_BALL_PARK_HOPE, dtype=env.qpos_init.dtype)
        + env.hope_to_scene
    )

    wait_env.FullMdpInitialWaitVecEnv._serve(env, ids)

    assert torch.equal(
        env.sim.data.qpos[:, env.b_q : env.b_q + 3],
        expected_park.expand(env.num_envs, 3),
    )


def test_completed_action_epoch_cannot_be_spliced_across_env_rows():
    env = _host_full_a_lifecycle_env()
    retired = torch.ones(2, dtype=torch.bool)
    r03 = wait_env.portable_reward.R03_PRESENT | wait_env.portable_reward.R03_PHYSICALLY_VALID
    r06 = (
        wait_env.portable_reward.R06_PRESENT
        | wait_env.portable_reward.R06_POLICY_ELIGIBLE
        | wait_env.portable_reward.R06_SOURCE_VALID
    )
    cells = wait_env.FULL_A_RECOVERY_END_AGE_TICK - wait_env.FULL_A_RECOVERY_START_AGE_TICK + 1

    # Row zero owns launch/contact/R03; row one owns R06/R07.  Marginals are
    # complete, but no action epoch is complete on one authoritative row.
    env._epoch_launch_succeeded[:] = torch.tensor([True, False])
    env._full_a_selected_racket_contact[:] = torch.tensor([True, False])
    env._full_a_owner_valid_bits[:, 1] = torch.tensor([r03, 0])
    env._full_a_owner_valid_bits[:, 2] = torch.tensor([0, r06])
    env._full_a_recovery_expected_count[:] = torch.tensor([0, cells])
    env._full_a_recovery_eligible_count[:] = torch.tensor([0, cells])
    env._full_a_recovery_last_age[:] = torch.tensor(
        [-1, wait_env.FULL_A_RECOVERY_END_AGE_TICK]
    )
    complete = env._full_a_completed_action_epoch(retired)
    assert not complete.any()

    env._full_a_owner_valid_bits[0, 2] = r06
    env._full_a_recovery_expected_count[0] = cells
    env._full_a_recovery_eligible_count[0] = cells
    env._full_a_recovery_last_age[0] = wait_env.FULL_A_RECOVERY_END_AGE_TICK
    assert torch.equal(
        env._full_a_completed_action_epoch(retired), torch.tensor([True, False])
    )
    env._full_a_owner_fault_bits[0, 2] = 1
    assert not env._full_a_completed_action_epoch(retired).any()


def test_host_full_a_reveal_launch_r03_one_shot_and_selected_clear_are_rowwise():
    env = _host_full_a_lifecycle_env()
    reveal, launch, due, deferred, _missed_launch = _prepare_and_settle(env)

    assert reveal.all()
    assert not launch.any()
    assert due.all() and not deferred.any()

    assert torch.equal(
        env._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_REVEAL_COMMITTED),
    )
    assert env._epoch_task_valid.all() and env._epoch_selected.all()
    assert torch.isfinite(env._epoch_task_f32).all()
    assert torch.count_nonzero(env._epoch_task_f32[:, :32]) > 0
    assert torch.equal(env._epoch_task_f32[:, 32:], env._full_a_launch_state_f32)
    assert torch.equal(
        env._epoch_clock_ticks,
        torch.tensor([[1, 3, 2, 4, 294], [1, 3, 2, 4, 294]]),
    )

    env.common_step_counter = 1
    scheduled_due, launch, _missed_launch = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    )
    assert not scheduled_due.any()
    assert not launch.any()
    env.common_step_counter = 2
    scheduled_due, launch, _missed_launch = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    )
    assert not scheduled_due.any()
    assert launch.all()
    assert torch.equal(
        env._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_LAUNCH_SETTLED),
    )
    assert env._epoch_launch_succeeded.all()
    assert torch.equal(
        env.sim.data.qpos[:, env.b_q : env.b_q + 7],
        env._full_a_launch_state_f32[:, :7],
    )
    assert torch.equal(
        env.sim.data.qvel[:, env.b_v : env.b_v + 6],
        env._full_a_launch_state_f32[:, 7:13],
    )
    present, physically_valid = env._full_a_publish_r03_fact()
    assert not present.any() and not physically_valid.any()
    env.common_step_counter = 3
    # Row one loses the physical-launch phase exactly at strike time.  It must
    # not publish on this tick or catch up after its phase is repaired.
    env._epoch_phase[1] = wait_env.FULL_A_PHASE_REVEAL_COMMITTED
    shared_racket_kinematics = env._full_a_racket_kinematics()
    original_racket_kinematics = env._full_a_racket_kinematics
    env._full_a_racket_kinematics = lambda: pytest.fail(
        "precomputed R03 official-site tuple was recomputed"
    )
    present, physically_valid = env._full_a_publish_r03_fact(
        shared_racket_kinematics
    )
    env._full_a_racket_kinematics = original_racket_kinematics
    assert torch.equal(present, torch.tensor([True, False]))
    assert torch.equal(physically_valid, torch.tensor([True, False]))
    assert torch.equal(env._full_a_r03_present, present)
    assert torch.equal(
        env._full_a_owner_valid_bits[:, 1],
        torch.tensor([3, 0], dtype=torch.long),
    )
    assert torch.equal(
        env._full_a_r03_fact_f32[0, 15:18],
        env.sim.data.site_xpos[0, 0],
    )
    assert torch.equal(
        env._full_a_r03_fact_f32[:, 21:24],
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    captured_fact = env._full_a_r03_fact_f32.clone()
    captured_valid = env._full_a_owner_valid_bits[:, 1].clone()
    captured_source = env._full_a_owner_source_step[:, 1].clone()
    assert torch.equal(captured_source, torch.tensor([3, -1]))
    assert not env._full_a_r03_armed.any()
    assert torch.equal(
        env._full_a_r03_expected_source_step, torch.tensor([-1, -1])
    )

    env._epoch_phase[1] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env.sim.data.site_xpos[:, 0] += 100.0
    env.common_step_counter = 4
    present, physically_valid = env._full_a_publish_r03_fact()
    assert not present.any() and not physically_valid.any()
    assert not env._full_a_r03_present.any()
    assert env._epoch_task_valid.all()
    assert torch.equal(env._full_a_r03_fact_f32, captured_fact)
    assert torch.equal(env._full_a_owner_valid_bits[:, 1], captured_valid)
    assert torch.equal(env._full_a_owner_source_step[:, 1], captured_source)

    env.sim.data.qpos[:, env.b_q : env.b_q + 3] += torch.tensor(
        [[-0.06, 0.01, -0.02], [-0.04, -0.01, 0.03]]
    )
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_physical_fact(env)
    assert env._full_a_physical_present.all()
    assert torch.equal(
        env._full_a_physical_fact_f32[:, :3],
        env.sim.data.qpos[:, env.b_q : env.b_q + 3]
        - env.env.scene.env_origins,
    )
    assert torch.count_nonzero(env._full_a_physical_fact_f32[:, 10:]) == 0

    env._full_a_selected_racket_contact[:] = True
    env._full_a_previous_ball_center[:] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    env._full_a_previous_ball_center_valid[:] = True
    env._full_a_net_crossed[:] = True
    env._full_a_net_clear[:] = True
    env._full_a_landing_crossing_present[:] = True
    env._full_a_landing_crossing_xy[:] = torch.tensor([[2.4, 0.1], [2.6, -0.1]])
    env._full_a_landing_on_table[:] = True
    env._full_a_landing_opponent_bound[:] = True
    env._full_a_landing_on_opponent[:] = True
    env._full_a_recovery_ready_streak[:] = torch.tensor([1, 2])
    env._full_a_recovery_ready_seen[:] = torch.tensor([False, True])
    env._full_a_recovery_expected_count[:] = torch.tensor([3, 4])
    env._full_a_recovery_eligible_count[:] = torch.tensor([2, 4])
    env._full_a_recovery_last_age[:] = torch.tensor([12, 15])
    env._full_a_recovery_sticky_fault[:] = torch.tensor([True, False])
    env._full_a_fact_integrity_fault_bits[:] = torch.tensor([
        wait_env.FULL_A_FACT_INTEGRITY_R03_NONFINITE,
        wait_env.FULL_A_FACT_INTEGRITY_R07_NONFINITE,
    ])
    env._full_a_teacher_source_to_task_yaw_wxyz[:] = torch.tensor(
        [[0.8, 0.0, 0.0, 0.6], [0.6, 0.0, 0.0, -0.8]]
    )
    env._full_a_teacher_source_to_task_translation_scene[:] = torch.tensor(
        [[0.4, -0.2, 0.0], [-0.7, 0.3, 0.0]]
    )
    incident_bits = env._full_a_fact_integrity_fault_bits.clone()
    peer = {
        name: getattr(env, name)[1].clone()
        for name in (
            "_epoch_phase",
            "_epoch_task_valid",
            "_epoch_selected",
            "_epoch_launch_succeeded",
            "_epoch_task_f32",
            "_epoch_clock_ticks",
            "_full_a_launch_state_f32",
            "_full_a_teacher_source_to_task_yaw_wxyz",
            "_full_a_teacher_source_to_task_translation_scene",
            "_full_a_owner_valid_bits",
            "_full_a_owner_fault_bits",
            "_full_a_owner_source_step",
            "_full_a_owner_fact_f32",
            "_full_a_physical_present",
            "_full_a_r03_present",
            "_full_a_r03_physically_valid",
            "_full_a_r03_armed",
            "_full_a_r03_expected_source_step",
            "_full_a_racket_contact",
            "_full_a_ball_table_contact",
            "_full_a_selected_racket_contact",
            "_full_a_contact_center",
            "_full_a_previous_ball_center",
            "_full_a_previous_ball_center_valid",
            "_full_a_net_crossed",
            "_full_a_net_clear",
            "_full_a_landing_crossing_present",
            "_full_a_landing_crossing_xy",
            "_full_a_landing_on_table",
            "_full_a_landing_opponent_bound",
            "_full_a_landing_on_opponent",
            "_full_a_outcome_code",
            "_full_a_recovery_ready_streak",
            "_full_a_recovery_ready_seen",
            "_full_a_recovery_expected_count",
            "_full_a_recovery_eligible_count",
            "_full_a_recovery_last_age",
            "_full_a_recovery_sticky_fault",
            "_full_a_contact_classification_status",
            "_full_a_generic_contact_event",
            "_full_a_selected_contact_event",
            "_full_a_opposite_contact_event",
            "_full_a_edge_contact_event",
            "_full_a_between_contact_event",
            "_full_a_invalid_contact_event",
        )
    }
    env._clear_lifecycle(torch.tensor([0]))
    assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_IDLE
    assert not env._epoch_task_valid[0]
    assert not env._full_a_selected_racket_contact[0]
    assert not env._full_a_previous_ball_center_valid[0]
    assert not env._full_a_recovery_ready_seen[0]
    assert env._full_a_recovery_expected_count[0] == 0
    assert env._full_a_recovery_eligible_count[0] == 0
    assert env._full_a_recovery_last_age[0] == -1
    assert not env._full_a_recovery_sticky_fault[0]
    assert env._full_a_owner_valid_bits[0, 1] == 0
    assert env._full_a_owner_source_step[0, 1] == -1
    assert torch.count_nonzero(env._full_a_r03_fact_f32[0]) == 0
    assert not env._full_a_r03_present[0]
    assert not env._full_a_r03_physically_valid[0]
    assert not env._full_a_r03_armed[0]
    assert env._full_a_r03_expected_source_step[0] == -1
    assert torch.equal(
        env._full_a_teacher_source_to_task_yaw_wxyz[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    assert torch.count_nonzero(
        env._full_a_teacher_source_to_task_translation_scene[0]
    ) == 0
    for name, expected in peer.items():
        assert torch.equal(getattr(env, name)[1], expected), name
    # Lifecycle replacement/reset happens before extras are assembled; it may
    # not wash away the transition's already-observed producer incident.
    assert torch.equal(env._full_a_fact_integrity_fault_bits, incident_bits)
    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    assert not env._full_a_fact_integrity_fault_bits.any()


def test_full_a_task_close_uses_real_slot_motion_duration_and_global_cadence():
    env = _host_full_a_lifecycle_env()
    row = wait_env.portable_catalog.load_portable_action_center_table().fresh_action
    env._full_a_cadence = wait_env.portable_catalog.derive_portable_fresh_cadence(
        wait_env.portable_catalog.load_portable_action_center_table()
    )
    geometry = wait_env.racket_contact_geometry
    questions = []

    def reverse(contact, velocity, spin, tts, _params, **_kwargs):
        return contact - velocity * tts[:, None], velocity.clone(), tts.clone()

    def question(**kwargs):
        result = wait_env.portable_question.build_center_question(
            torch=torch,
            row=row,
            base_position_scene=kwargs["base_position_scene"],
            base_quat_wxyz=kwargs["base_quat_wxyz"],
            contact_reference_root_z_scene=kwargs[
                "contact_reference_root_z_scene"
            ],
            step_dt=kwargs["step_dt"],
            table_surface_z_scene=kwargs["table_surface_z_scene"],
            back_integrate=reverse,
            venue_params=SimpleNamespace(ball_radius=geometry.BALL_RADIUS_M),
            geometry=geometry,
            serve_horizon_s=0.6,
            backint_h=0.005,
            plane_margin=0.005,
        )
        questions.append(result)
        return result

    env._full_a_catalog = SimpleNamespace(fresh_action=row)
    env._full_a_question_builder = question
    env.common_step_counter = 1000
    # The cadence scheduler is episode-relative and can differ rowwise after a
    # masked reset.  Epoch clocks remain monotonic common-step boundaries.
    env.episode_length_buf[:] = torch.tensor([294, 1])
    env._full_a_next_reveal_tick[:] = torch.tensor([480, 295])
    env._full_a_reveal_rows(torch.arange(2))

    assert torch.equal(
        env._full_a_teacher_rate, questions[-1]["teacher_rate"]
    )
    assert torch.equal(
        env._epoch_clock_ticks[:, 3], torch.full((2,), 1101)
    )
    old_contact_plus_one_second = env._epoch_clock_ticks[:, 1] + 50
    assert torch.equal(
        old_contact_plus_one_second - env._epoch_clock_ticks[:, 3],
        torch.full((2,), 41),
    )
    assert torch.equal(
        env._epoch_clock_ticks[:, 4], torch.full((2,), 1185)
    )
    assert not torch.equal(
        env._epoch_clock_ticks[:, 4], env._epoch_clock_ticks[:, 3] + 1
    )

    # On the sixth ACCEPT the episode scheduler has already parked itself at
    # horizon+1. The accepted shot still owns one 185-tick cadence boundary.
    env.common_step_counter = 1220
    env.episode_length_buf[0] = 1035
    env._full_a_next_reveal_tick[0] = 1501
    env._full_a_reveal_rows(torch.tensor([0]))
    assert env._epoch_clock_ticks[0, 4] == 1405
    assert env._epoch_clock_ticks[0, 4] != 1501


def test_full_a_masked_reset_cadence_maps_to_global_epoch_clocks_and_r06_first():
    env = _host_full_a_lifecycle_env()
    ids = torch.arange(2)
    env._clear_lifecycle(ids)
    env._full_a_cadence = SimpleNamespace(
        first_reveal_tick=295,
        cadence_ticks=293,
        episode_horizon_ticks=1500,
        reference_due_ticks=(295, 588, 881, 1174),
    )
    env.common_step_counter = 1000
    env.episode_length_buf[:] = torch.tensor([17, 0])
    env._full_a_next_reveal_tick[:] = torch.tensor([999, 999])
    env._full_a_scheduled_ordinal.zero_()

    # A masked reset restarts only row one's episode-relative cadence.  The
    # accepted question clocks still enter the monotonic common-step domain.
    env._full_a_reset_cadence_rows(torch.tensor([1]))
    env.episode_length_buf[1] = 294
    scheduled_due, launch, missed = env._full_a_prepare_step()
    assert torch.equal(scheduled_due, torch.tensor([False, True]))
    assert not launch.any() and not missed.any()
    env.episode_length_buf += 1
    env.common_step_counter += 1
    reveal, due, deferred, overlap = env._full_a_settle_reveal(
        scheduled_due, torch.zeros(2, dtype=torch.bool)
    )
    assert torch.equal(reveal, torch.tensor([False, True]))
    assert torch.equal(due, reveal)
    assert not deferred.any() and not overlap.any()
    assert env.episode_length_buf[1] == 295
    assert env._epoch_clock_ticks[1].tolist() == [1001, 1003, 1002, 1004, 1294]

    # R06 settles at the exact global task-close boundary even though the row's
    # episode clock is only 295.  It cannot retire before the R07 join.
    env._epoch_phase[1] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env.common_step_counter = int(env._epoch_clock_ticks[1, 3])
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()
    settled, outcome = env._full_a_settle_outcome({"ball_pos": ball_pos})
    assert torch.equal(settled, torch.tensor([False, True]))
    present, source_valid, _common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env, settled, outcome
        )
    )
    assert torch.equal(present, torch.tensor([False, True]))
    assert torch.equal(source_valid, present)
    early = wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
        env,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    assert not early[0].any()

    task_close = int(env._epoch_clock_ticks[1, 3])
    for age in range(
        wait_env.FULL_A_RECOVERY_START_AGE_TICK,
        wait_env.FULL_A_RECOVERY_END_AGE_TICK + 1,
    ):
        env.common_step_counter = task_close + age
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        r07_present, r07_valid = (
            wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(
                env, torch.zeros((2, 13)), torch.ones(2, dtype=torch.bool)
            )
        )
        assert torch.equal(r07_present, torch.tensor([False, True]))
        assert torch.equal(r07_valid, r07_present)

    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            env,
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert torch.equal(terminal, torch.tensor([False, True]))
    assert torch.equal(success, terminal)
    assert not failure.any() and not timeout.any() and not completion_fault.any()
    assert env._epoch_phase[1] == wait_env.FULL_A_PHASE_RETIRED


def test_full_a_r07_first_late_r06_retires_and_accepts_same_due_tick():
    env = _host_full_a_lifecycle_env()
    ids = torch.arange(2)
    env._clear_lifecycle(ids)
    env._epoch_task_valid[0] = True
    env._epoch_selected[0] = True
    env._epoch_phase[0] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_clock_ticks[0] = torch.tensor([0, 2, 1, 0, 100])
    env._full_a_selected_racket_contact[0] = True

    for age in range(
        wait_env.FULL_A_RECOVERY_START_AGE_TICK,
        wait_env.FULL_A_RECOVERY_END_AGE_TICK + 1,
    ):
        env.common_step_counter = age
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        present, valid = (
            wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(
                env, torch.zeros((2, 13)), torch.ones(2, dtype=torch.bool)
            )
        )
        assert torch.equal(present, torch.tensor([True, False]))
        assert torch.equal(valid, present)

    pre_r06 = wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
        env,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    assert not pre_r06[0].any()
    assert env._full_a_recovery_expected_count[0] == 68
    assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_LAUNCH_SETTLED

    # The fixed Motion-owned R07 tape can be complete before physical R06.
    # Its fact is present and valid, but term 13 remains exactly unpaid until
    # the existing clean OUTCOME_SETTLED authority opens payment.
    assert torch.bitwise_and(
        env._full_a_owner_valid_bits[0, 3],
        wait_env.portable_reward.R07_PRESENT
        | wait_env.portable_reward.R07_NUMERICALLY_VALID,
    ).ne(0)
    dense_env = _fixed_reward_env()
    dense_env.full_a_mode = True
    dense_env._epoch_task_valid.copy_(env._epoch_task_valid)
    dense_env._epoch_phase.copy_(env._epoch_phase)
    dense_env._full_a_outcome_code.copy_(env._full_a_outcome_code)
    dense_env._full_a_owner_valid_bits = env._full_a_owner_valid_bits
    dense_env._full_a_owner_fault_bits = env._full_a_owner_fault_bits
    dense_env._full_a_owner_fact_f32 = env._full_a_owner_fact_f32
    dense_env._full_a_lifecycle_reward_weights = torch.tensor(
        wait_env.portable_reward.LIFECYCLE_WEIGHTS,
        dtype=dense_env._full_a_owner_fact_f32.dtype,
    )
    dense_env._full_a_selected_contact_event = torch.zeros(
        2, dtype=torch.bool
    )
    dense_env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r06_payment_event = torch.zeros(2, dtype=torch.bool)
    _total, pre_r06_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(dense_env)
    )
    assert torch.count_nonzero(pre_r06_terms[:, 13]) == 0

    # The selected/no-crossing R06 boundary and the next per-episode cadence due
    # coincide.  Production must settle R06, join/retire, then ACCEPT the frozen
    # due without a reset or an intervening tick.
    env.common_step_counter = 99
    env.episode_length_buf[:] = torch.tensor([294, 0])
    env._full_a_next_reveal_tick[:] = torch.tensor([295, 999])
    env._full_a_scheduled_ordinal.fill_(-1)
    scheduled_due, launch, missed = env._full_a_prepare_step()
    assert torch.equal(scheduled_due, torch.tensor([True, False]))
    assert not launch.any() and not missed.any()
    # Mirror the real _advance_plant transition before all settlement.  Clock4
    # names the post-physics boundary, not the policy-side prepare boundary.
    env.episode_length_buf += 1
    env.common_step_counter += 1
    assert env.common_step_counter == 100
    assert env.episode_length_buf[0] == 295
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()
    settled, outcome = env._full_a_settle_outcome({"ball_pos": ball_pos})
    assert torch.equal(settled, torch.tensor([True, False]))
    r06_present, r06_valid, _common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env, settled, outcome
        )
    )
    assert torch.equal(r06_present, torch.tensor([True, False]))
    assert torch.equal(r06_valid, r06_present)
    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            env,
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert torch.equal(terminal, torch.tensor([True, False]))
    assert torch.equal(success, terminal)
    assert not failure.any() and not timeout.any() and not completion_fault.any()
    assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_RETIRED

    reveal, due, deferred, overlap = env._full_a_settle_reveal(
        scheduled_due, torch.zeros(2, dtype=torch.bool)
    )
    assert torch.equal(reveal, torch.tensor([True, False]))
    assert torch.equal(due, reveal)
    assert not deferred.any() and not overlap.any()
    assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_REVEAL_COMMITTED
    assert env._epoch_clock_ticks[0, 0] == 100
    assert env._epoch_clock_ticks[0, 4] == 393
    assert env._full_a_scheduled_ordinal[0] == 0


@pytest.mark.parametrize("source", ("site_xpos", "cvel"))
def test_host_full_a_r03_nonfinite_is_neutral_named_and_peer_exact(source):
    faulty = _host_full_a_lifecycle_env()
    clean = _host_full_a_lifecycle_env()
    for env in (faulty, clean):
        _prepare_and_settle(env)
        env.common_step_counter = 2
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
        env.common_step_counter = 3
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    if source == "site_xpos":
        faulty.sim.data.site_xpos[0, 0, 0] = float("nan")
    else:
        faulty.sim.data.cvel[0, 0, 3] = float("nan")

    bad_present, bad_valid = faulty._full_a_publish_r03_fact()
    clean_present, clean_valid = clean._full_a_publish_r03_fact()
    assert torch.equal(bad_present, torch.tensor([False, True]))
    assert torch.equal(bad_valid, torch.tensor([False, True]))
    assert clean_present.all() and clean_valid.all()
    assert torch.equal(
        faulty._full_a_fact_integrity_fault_bits,
        torch.tensor([
            wait_env.FULL_A_FACT_INTEGRITY_R03_NONFINITE, 0
        ]),
    )
    assert torch.equal(
        faulty._full_a_owner_fault_bits[:, 1], torch.tensor([1, 0])
    )
    assert torch.equal(
        faulty._full_a_owner_fact_f32[1, 1],
        clean._full_a_owner_fact_f32[1, 1],
    )
    terms = wait_env.portable_reward.lifecycle_reward14(
        valid_bits=faulty._full_a_owner_valid_bits,
        fact_f32=faulty._full_a_owner_fact_f32,
        owner_fault_bits=faulty._full_a_owner_fault_bits,
        step_dt=faulty.step_dt,
    )
    assert torch.isfinite(terms).all()
    assert torch.count_nonzero(terms[0, :10]) == 0
    assert torch.count_nonzero(terms[1, :10]) > 0


def test_host_full_a_retired_shot_accepts_next_due_without_true_reset():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env._epoch_phase[0] = wait_env.FULL_A_PHASE_RETIRED
    env.actions = torch.arange(62, dtype=torch.float32).reshape(2, 31)
    env.last_actions = -env.actions.clone()
    env.episode_length_buf = torch.tensor([294, 37])
    env.reset_generation = torch.tensor([4, 8])
    robot_qpos = env.sim.data.qpos[:, : env.b_q].clone()
    actions = env.actions.clone()
    last_actions = env.last_actions.clone()
    reveal, _launch, due, deferred, _missed_launch = _prepare_and_settle(env)
    assert torch.equal(due, torch.tensor([True, False]))
    assert torch.equal(reveal, torch.tensor([True, False]))
    assert not deferred.any()
    assert env._epoch_task_valid.all()
    assert torch.equal(env.sim.data.qpos[:, : env.b_q], robot_qpos)
    assert torch.equal(env.actions, actions)
    assert torch.equal(env.last_actions, last_actions)
    assert torch.equal(env.reset_generation, torch.tensor([4, 8]))


def test_host_full_a_post_transition_retirement_accepts_while_busy_peer_defers():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_task_valid[:] = True
    generation_before = env.reset_generation.clone()
    busy_tasks = env._epoch_task_f32.clone()

    scheduled_due, launch, _missed_launch = env._full_a_prepare_step()

    assert scheduled_due.tolist() == [True, True]
    assert not launch.any()
    # Availability is decided after recovery.  The first row naturally
    # retires on this transition, while the second remains busy.
    env._epoch_phase[0] = wait_env.FULL_A_PHASE_RETIRED
    reveal, due, deferred, terminal_overlap = env._full_a_settle_reveal(
        scheduled_due, torch.zeros(2, dtype=torch.bool)
    )
    assert reveal.tolist() == [True, False]
    assert due.tolist() == [True, True]
    assert deferred.tolist() == [False, True]
    assert not terminal_overlap.any()
    assert env._epoch_phase.tolist() == [
        wait_env.FULL_A_PHASE_REVEAL_COMMITTED,
        wait_env.FULL_A_PHASE_LAUNCH_SETTLED,
    ]
    assert env._epoch_task_valid.all()
    assert torch.equal(env._epoch_task_f32[1], busy_tasks[1])
    assert torch.equal(env.reset_generation, generation_before)
    assert env._full_a_scheduled_ordinal.tolist() == [0, 0]
    assert env._full_a_next_reveal_tick.tolist() == [295, 295]

    env.episode_length_buf.fill_(2)
    env.common_step_counter = 1
    scheduled_due, _launch, _missed_launch = env._full_a_prepare_step()
    assert not scheduled_due.any()


def test_host_full_a_busy_without_retirement_defers_without_overwrite():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[1] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_task_valid[1] = True
    busy_task = env._epoch_task_f32[1].clone()

    scheduled_due, launch, _missed_launch = env._full_a_prepare_step()
    assert not launch.any()
    reveal, due, deferred, terminal_overlap = env._full_a_settle_reveal(
        scheduled_due, torch.zeros(2, dtype=torch.bool)
    )
    assert reveal.tolist() == [True, False]
    assert due.tolist() == [True, True]
    assert deferred.tolist() == [False, True]
    assert not terminal_overlap.any()
    assert env._epoch_phase[1] == wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    assert env._epoch_task_valid[1]
    assert torch.equal(env._epoch_task_f32[1], busy_task)
    assert env._full_a_scheduled_ordinal.tolist() == [0, 0]
    assert env._full_a_next_reveal_tick.tolist() == [295, 295]


def test_host_full_a_scheduled_ordinal_caps_episode_fit_opportunities():
    env = _host_full_a_lifecycle_env()
    env._full_a_cadence = SimpleNamespace(
        first_reveal_tick=2,
        cadence_ticks=3,
        episode_horizon_ticks=10,
        reference_due_ticks=(2, 5),
    )
    env._full_a_next_reveal_tick.fill_(2)
    env._full_a_scheduled_ordinal.fill_(-1)
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_LAUNCH_SETTLED)

    for episode_length in (1, 4):
        env.episode_length_buf.fill_(episode_length)
        scheduled_due, _launch, _missed_launch = env._full_a_prepare_step()
        reveal, due, deferred, terminal_overlap = env._full_a_settle_reveal(
            scheduled_due, torch.zeros(2, dtype=torch.bool)
        )
        assert due.all() and deferred.all() and not reveal.any()
        assert not terminal_overlap.any()

    assert env._full_a_scheduled_ordinal.eq(1).all()
    assert env._full_a_next_reveal_tick.eq(11).all()
    env.episode_length_buf.fill_(7)
    scheduled_due, _launch, _missed_launch = env._full_a_prepare_step()
    assert not scheduled_due.any()


def test_host_full_a_fourth_due_exhausts_schedule_but_keeps_shot_boundary():
    env = _host_full_a_lifecycle_env()
    env._full_a_cadence = SimpleNamespace(
        first_reveal_tick=295,
        cadence_ticks=293,
        episode_horizon_ticks=1500,
        reference_due_ticks=(295, 588, 881, 1174),
    )
    ids = torch.arange(env.num_envs)
    env._clear_lifecycle(ids)
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_RETIRED)
    env._full_a_scheduled_ordinal.fill_(2)
    env._full_a_next_reveal_tick.fill_(1174)
    env.episode_length_buf.fill_(1173)
    env.common_step_counter = 1173

    scheduled_due, _launch, missed = env._full_a_prepare_step()
    # Mirror the real integrated transition before the returned-observation
    # settlement boundary.
    env.episode_length_buf += 1
    env.common_step_counter += 1
    reveal, due, deferred, overlap = env._full_a_settle_reveal(
        scheduled_due, torch.zeros(env.num_envs, dtype=torch.bool)
    )

    assert due.all() and reveal.all()
    assert not deferred.any() and not overlap.any() and not missed.any()
    assert env.episode_length_buf.eq(1174).all()
    assert env._full_a_scheduled_ordinal.eq(3).all()
    assert env._full_a_next_reveal_tick.eq(1501).all()
    # This is the accepted fourth shot's internal settlement boundary, not a
    # fifth curriculum opportunity.
    assert env._epoch_clock_ticks[:, 4].eq(1467).all()

    env.episode_length_buf.fill_(1466)
    scheduled_due, _launch, _missed_launch = env._full_a_prepare_step()
    assert not scheduled_due.any()


def test_host_full_a_step_reveals_after_balance_prefix_without_r07_admission():
    env = _host_full_a_lifecycle_env()
    ids = torch.arange(2)
    env._clear_lifecycle(ids)
    wait_env.FullMdpInitialWaitVecEnv._full_a_reset_cadence_rows(env, ids)
    env._full_a_next_reveal_tick.fill_(295)
    env._full_a_cadence_ready_streak.zero_()
    env.episode_length_buf.fill_(293)
    env.common_step_counter = 0
    env._full_a_begin_control_step = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step, env
    )
    env._full_a_update_cadence_readiness = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_update_cadence_readiness, env
    )

    def advance(actions):
        env.episode_length_buf += 1
        env.common_step_counter += 1
        return {}, torch.zeros(2), torch.zeros_like(actions)

    env._advance_plant = advance
    env.sim.forward = lambda: None
    env._latch_post_forward_resolved_table_contacts = lambda _census=None: None
    env._latch_post_forward_table_keepout = lambda: None
    env._full_a_latch_ball_contacts = lambda _census=None: None
    env._cap_ok = False
    env._state = lambda: {}
    env._full_a_teacher_frame = torch.zeros(2, dtype=torch.long)
    env._full_a_teacher_joint_pos = torch.zeros((2, 31))
    env._aligned_teacher_body_pos = torch.zeros((2, 1, 3))
    reward_teacher = []
    obs_teacher = []
    tracked_materializations = []
    recovery_tracked_inputs = []
    racket_materializations = []
    r03_racket_inputs = []
    reward_racket_inputs = []
    fixed_racket_kinematics = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    fixed_terms = torch.zeros((2, wait_env.reward_contract.REWARD_TERM_COUNT))
    fixed_terms[:, 20] = fixed_racket_kinematics[0][:, 0]
    fixed_terms[:, 21] = fixed_racket_kinematics[1][:, 0]

    def racket_kinematics():
        racket_materializations.append(1)
        return fixed_racket_kinematics

    env._full_a_racket_kinematics = racket_kinematics
    fixed_tracked_kinematics = object()

    def tracked_body_kinematics():
        tracked_materializations.append(1)
        return fixed_tracked_kinematics

    env._fullmdp_tracked_body_kinematics = tracked_body_kinematics
    active_teacher_step = {"enabled": False}

    def update_teacher():
        if active_teacher_step["enabled"]:
            env._full_a_teacher_frame += 1
            env._full_a_teacher_joint_pos += 100.0
            env._aligned_teacher_body_pos += 1000.0
            return
        env._full_a_teacher_frame.zero_()
        env._full_a_teacher_joint_pos.copy_(
            env._epoch_task_valid[:, None].to(torch.float32).expand(-1, 31)
            * 100.0
        )
        env._aligned_teacher_body_pos.copy_(
            env._epoch_task_valid[:, None, None]
            .to(torch.float32)
            .expand(-1, 1, 3)
            * 1000.0
        )

    terminal_mask = torch.zeros(2, dtype=torch.bool)
    env._full_a_update_teacher = update_teacher
    env._fullmdp_termination = lambda _state, _qdes: (
        terminal_mask.clone(),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
    )
    def recovery_errors(tracked_kinematics=None):
        recovery_tracked_inputs.append(tracked_kinematics)
        return torch.zeros((2, 13))

    env._full_a_recovery_component_errors = recovery_errors
    env._full_a_recovery_joint_limit_ok = lambda: torch.ones(2, dtype=torch.bool)
    env._full_a_publish_physical_fact = lambda: None
    def publish_r03(racket_kinematics):
        r03_racket_inputs.append(racket_kinematics)
        return (
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )

    env._full_a_publish_r03_fact = publish_r03
    env._full_a_settle_outcome = lambda _state: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
    )
    env._full_a_publish_r06_fact = lambda _event, _outcome: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_publish_r07_fact = lambda *_args: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    def reward28(
        racket_kinematics,
        tracked_kinematics=None,
        *,
        return_paddle_error=False,
    ):
        reward_racket_inputs.append(racket_kinematics)
        assert tracked_kinematics is fixed_tracked_kinematics
        reward_teacher.append(
            (
                env._full_a_teacher_frame.clone(),
                env._full_a_teacher_joint_pos.clone(),
                env._epoch_task_valid.clone(),
                env._aligned_teacher_body_pos.clone(),
            )
        )
        result = (fixed_terms.sum(1), fixed_terms.clone())
        if return_paddle_error:
            return result + (torch.zeros((2, 4)),)
        return result

    env._fullmdp_reward = reward28
    env._full_a_finish_recovery = lambda _terminated, _truncated: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    def compute_obs():
        obs_teacher.append(
            (
                env._full_a_teacher_frame.clone(),
                env._full_a_teacher_joint_pos.clone(),
                env._epoch_task_valid.clone(),
                env._aligned_teacher_body_pos.clone(),
            )
        )

    env._compute_obs = compute_obs
    env.get_observations = lambda: {}

    first = wait_env.FullMdpInitialWaitVecEnv._step_full_a(
        env, torch.zeros((2, 31))
    )
    assert torch.equal(first[1], fixed_terms.sum(1))
    assert not first[3]["full_a_reveal_due_event"].any()
    # One sample cannot satisfy the legacy two-transition R07 dwell.  The
    # next due still reveals because R07 is telemetry, not admission.
    assert env._full_a_cadence_ready_streak.eq(1).all()
    second = wait_env.FullMdpInitialWaitVecEnv._step_full_a(
        env, torch.zeros((2, 31))
    )
    assert torch.equal(second[1], fixed_terms.sum(1))
    assert second[3]["full_a_reveal_due_event"].all()
    assert second[3]["full_a_reveal_event"].all()
    assert not second[3]["full_a_reveal_deferred_event"].any()
    assert env.episode_length_buf.eq(295).all()
    # The due transition's action and reward use the same hidden frame-0
    # ready teacher.  Only after settle installs the task does the returned
    # observation publish the selected motion's distinct frame-0 teacher.
    assert torch.equal(reward_teacher[1][0], torch.zeros(2, dtype=torch.long))
    assert torch.count_nonzero(reward_teacher[1][1]) == 0
    assert not reward_teacher[1][2].any()
    assert torch.equal(obs_teacher[1][0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(obs_teacher[1][1], torch.full((2, 31), 100.0))
    assert obs_teacher[1][2].all()

    # A due row that terminates inside the transition never becomes a public
    # due/reveal/defer event and never installs a task.  Its true reset restores
    # the initial cadence, while the surviving peer ACCEPTs independently.
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_RETIRED)
    env._full_a_scheduled_ordinal.fill_(0)
    env._full_a_next_reveal_tick.fill_(588)
    env.episode_length_buf.fill_(587)
    terminal_mask.copy_(torch.tensor([True, False]))
    env.last_terminal_bits = torch.zeros(2, dtype=torch.long)

    def reset_idx(reset_ids):
        env.episode_length_buf[reset_ids] = 0

    env._reset_idx = reset_idx
    env._full_a_prime_cadence_readiness = lambda _ids: None
    third = wait_env.FullMdpInitialWaitVecEnv._step_full_a(
        env, torch.zeros((2, 31))
    )
    assert torch.equal(third[1], fixed_terms.sum(1))
    assert len(racket_materializations) == 3
    assert len(tracked_materializations) == 3
    assert len(recovery_tracked_inputs) == 3
    assert all(
        value is fixed_tracked_kinematics
        for value in recovery_tracked_inputs
    )
    assert all(value is fixed_racket_kinematics for value in r03_racket_inputs)
    assert all(
        value is fixed_racket_kinematics for value in reward_racket_inputs
    )
    third_extras = third[3]
    assert third_extras["full_a_scheduled_due_event"].all()
    assert third_extras["full_a_due_terminal_overlap_event"].tolist() == [
        True,
        False,
    ]
    assert not third_extras["full_a_reveal_due_event"][0]
    assert not third_extras["full_a_reveal_event"][0]
    assert not third_extras["full_a_reveal_deferred_event"][0]
    assert third_extras["full_a_reveal_due_event"][1]
    assert third_extras["full_a_reveal_event"][1]
    assert not third_extras["full_a_reveal_deferred_event"][1]
    assert env._full_a_scheduled_ordinal.tolist() == [-1, 1]
    assert env._full_a_next_reveal_tick.tolist() == [
        env._full_a_cadence.first_reveal_tick,
        881,
    ]
    assert not env._epoch_task_valid[0] and env._epoch_task_valid[1]

    # Active mimic is the chronology-sensitive case: reward must use the exact
    # cached teacher t that produced the action, while the returned observation
    # alone advances to t+1.  This kills a pre-reward teacher update even when
    # a hidden ready teacher happens to be numerically stationary.
    terminal_mask.zero_()
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_LAUNCH_SETTLED)
    env._epoch_task_valid.fill_(True)
    env._full_a_next_reveal_tick.fill_(1000)
    env.episode_length_buf.zero_()
    env._full_a_teacher_frame.copy_(torch.tensor([7, 11]))
    env._full_a_teacher_joint_pos.copy_(
        torch.tensor([70.0, 110.0])[:, None].expand(-1, 31)
    )
    env._aligned_teacher_body_pos.copy_(
        torch.tensor([700.0, 1100.0])[:, None, None].expand(-1, 1, 3)
    )
    active_teacher_step["enabled"] = True
    wait_env.FullMdpInitialWaitVecEnv._step_full_a(
        env, torch.zeros((2, 31))
    )
    assert torch.equal(reward_teacher[-1][0], torch.tensor([7, 11]))
    assert torch.equal(
        reward_teacher[-1][1],
        torch.tensor([70.0, 110.0])[:, None].expand(-1, 31),
    )
    assert torch.equal(
        reward_teacher[-1][3],
        torch.tensor([700.0, 1100.0])[:, None, None].expand(-1, 1, 3),
    )
    assert torch.equal(obs_teacher[-1][0], torch.tensor([8, 12]))
    assert torch.equal(
        obs_teacher[-1][1],
        torch.tensor([170.0, 210.0])[:, None].expand(-1, 31),
    )
    assert torch.equal(
        obs_teacher[-1][3],
        torch.tensor([1700.0, 2100.0])[:, None, None].expand(-1, 1, 3),
    )


def test_host_full_a_r07_not_ready_does_not_defer_curriculum_exposure():
    env = _host_full_a_lifecycle_env()
    ids = torch.arange(2)
    env._clear_lifecycle(ids)
    wait_env.FullMdpInitialWaitVecEnv._full_a_reset_cadence_rows(env, ids)
    env._full_a_next_reveal_tick.fill_(295)
    env.episode_length_buf.fill_(294)

    reveal, _launch, due, deferred, _missed_launch = _prepare_and_settle(env)
    assert due.all() and reveal.all() and not deferred.any()
    assert env._full_a_next_reveal_tick.eq(588).all()


def test_host_full_a_step_freezes_landing_crossing_on_shot_retirement():
    env = _host_full_a_lifecycle_env()
    env._epoch_selected[:] = True
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    env._full_a_outcome_code[:] = wait_env.FULL_A_OUTCOME_OUT
    env._full_a_begin_control_step = lambda: None
    env._advance_plant = lambda actions: (
        {},
        torch.zeros(2),
        torch.zeros_like(actions),
    )
    env.sim.forward = lambda: None
    env._latch_post_forward_resolved_table_contacts = lambda _census=None: None
    env._latch_post_forward_table_keepout = lambda: None
    env._full_a_latch_ball_contacts = lambda _census=None: None
    env._cap_ok = False
    env._state = lambda: {}
    env._full_a_update_teacher = lambda: None
    env._fullmdp_termination = lambda _state, _qdes: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_publish_physical_fact = lambda: None
    env._full_a_recovery_component_errors = (
        lambda _tracked=None: torch.zeros((2, 13))
    )
    env._full_a_recovery_joint_limit_ok = lambda: torch.ones(2, dtype=torch.bool)
    env._full_a_update_cadence_readiness = lambda *_args: None
    env._full_a_publish_r03_fact = lambda _racket_kinematics=None: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )

    def settle(_state):
        env._full_a_landing_crossing_present[:] = True
        return (
            torch.ones(2, dtype=torch.bool),
            torch.full((2,), wait_env.FULL_A_OUTCOME_OUT, dtype=torch.long),
        )

    env._full_a_settle_outcome = settle
    env._full_a_publish_r06_fact = lambda _event, _outcome: (
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_publish_r07_fact = lambda *_args: (
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    env._fullmdp_reward = (
        lambda _racket_kinematics=None, _tracked=None,
        return_paddle_error=False: (
            torch.zeros(2),
            torch.zeros((2, wait_env.reward_contract.REWARD_TERM_COUNT)),
            *(() if not return_paddle_error else (torch.zeros((2, 4)),)),
        )
    )
    def finish(_terminated, _truncated):
        env._epoch_phase[:] = wait_env.FULL_A_PHASE_RETIRED
        return (
            torch.ones(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )

    env._full_a_finish_recovery = finish
    env._compute_obs = lambda: None
    env.get_observations = lambda: {}

    generation_before = env.reset_generation.clone()
    _obs, _reward, dones, extras = (
        wait_env.FullMdpInitialWaitVecEnv._step_full_a(
            env, torch.zeros((2, 31))
        )
    )

    assert not bool(dones.any())
    assert extras["full_a_landing_crossing_event"].all()
    assert extras["full_a_shot_retired_event"].all()
    assert extras["full_a_scheduled_due_event"].all()
    assert extras["full_a_reveal_due_event"].all()
    assert extras["full_a_reveal_event"].all()
    assert not extras["full_a_due_terminal_overlap_event"].any()
    assert extras["full_a_r07_present_event"].all()
    assert extras["full_a_r07_eligible_event"].all()
    assert extras["full_a_phase_before_reset"].eq(
        wait_env.FULL_A_PHASE_REVEAL_COMMITTED
    ).all()
    assert not extras["full_a_selected_reset_event"].any()
    assert torch.equal(env.reset_generation, generation_before)
    assert env._full_a_scheduled_ordinal.eq(0).all()
    # Durable old-shot facts were cloned before ACCEPT cleared the row.
    assert not env._full_a_landing_crossing_present.any()


def test_host_full_a_physical_fact_is_scene_local_for_distinct_env_origins():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_task_valid[:] = True
    env._epoch_launch_succeeded[:] = True
    env._full_a_racket_contact[:] = True
    env._full_a_selected_contact_event[:] = True
    env.env.scene.env_origins[:] = torch.tensor(
        [[0.0, 0.0, 0.0], [6.0, -2.0, 0.0]]
    )
    local_center = torch.tensor([1.25, -0.25, 1.0])
    local_contact = torch.tensor([0.75, -0.5, 1.0])
    env.sim.data.qpos[:, env.b_q : env.b_q + 3] = (
        env.env.scene.env_origins + local_center
    )
    env._full_a_contact_center[:] = env.env.scene.env_origins + local_contact

    env._full_a_publish_physical_fact()

    assert torch.equal(
        env._full_a_physical_fact_f32[:, :3], local_center.expand(2, 3)
    )
    assert torch.equal(
        env._full_a_physical_fact_f32[:, 3:6], local_contact.expand(2, 3)
    )
    assert torch.equal(
        env._full_a_physical_fact_f32[:, 6:9], local_contact.expand(2, 3)
    )


def test_host_full_a_physical_fact_freezes_after_outcome_settlement():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_task_valid[:] = True
    env._epoch_launch_succeeded[:] = True
    env._full_a_selected_contact_event[:] = True
    env.sim.data.qpos[:, env.b_q : env.b_q + 3] = torch.tensor(
        [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]
    )
    env._full_a_contact_center[:] = env.sim.data.qpos[
        :, env.b_q : env.b_q + 3
    ]
    env._full_a_publish_physical_fact()
    retained_fact = env._full_a_physical_fact_f32.clone()
    retained_ordinal = env._full_a_observation_ordinal.clone()
    retained_source = env._full_a_physical_source_step.clone()

    env._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    env.sim.data.qpos[:, env.b_q : env.b_q + 3] = -100.0
    env._full_a_selected_contact_event.zero_()
    env._full_a_publish_physical_fact()

    assert torch.equal(env._full_a_physical_fact_f32, retained_fact)
    assert torch.equal(env._full_a_observation_ordinal, retained_ordinal)
    assert torch.equal(env._full_a_physical_source_step, retained_source)
    assert not env._full_a_physical_present.any()


def _portable_center_row(**updates):
    values = {
        "mount_normal_sign": 1,
        "base_spawn_center_w_xy_m": (0.2, -0.1),
        "base_travel_center_b_yaw_xy_m": (0.1, 0.0),
        "contact_offset_center_b_yaw_m": (0.7, -0.5, 0.1),
        "incoming_direction_center_b_yaw": (-1.0, 0.0, 0.0),
        "incoming_speed_center_mps": 3.0,
        "spin_direction_center_b_yaw": (0.0, 1.0, 0.0),
        "spin_magnitude_center_radps": 10.0,
        "reference_base_root_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "reference_racket_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "reference_racket_site_position_w_m": (0.3, -0.6, 1.0),
        "reference_reach_offset_xy_m": (0.3, -0.5),
        "reference_racket_site_velocity_w_mps": (3.0, 0.0, 0.0),
        "reference_racket_angular_velocity_w_radps": (0.0, 0.0, 2.0),
        "reference_racket_site_speed_mps": 3.0,
        "reference_t_hit_s": 1.04,
        "reference_t_cycle_s": 1.9,
        "reaction_margin_s": 0.1,
        "teacher_rate_min": 0.6,
        "teacher_rate_max": 1.0,
        "time_to_contact_center_s": 1.94,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_portable_center_question_uses_action_row_yaw_and_two_stage_reverse_integration():
    calls = []

    def truncated_reverse(contact, velocity, spin, tts, _params, **_kwargs):
        calls.append(tts.clone())
        effective = torch.where(tts > 0.5, torch.full_like(tts, 0.4), tts)
        return contact - velocity * effective[:, None], velocity.clone(), effective

    geometry = SimpleNamespace(
        BALL_RADIUS_M=0.02,
        face_normal_local=lambda sign: (0.0, float(sign), 0.0),
        ball_center_from_site_local=lambda sign: (
            0.001,
            0.02 if sign == 1 else -0.033,
            0.002,
        ),
        face_center_from_site_local=lambda sign: (
            0.001,
            0.0 if sign == 1 else -0.013,
            0.002,
        ),
        canonical_teacher_rate_from_site_speed=lambda required, reference, lo, hi: required / reference,
    )
    base = torch.tensor([[0.0, 0.0, 0.9], [1.0, 2.0, 0.9]])
    yaw_90 = 2.0 ** -0.5
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [yaw_90, 0.0, 0.0, yaw_90]]
    )
    result = wait_env.portable_question.build_center_question(
        torch=torch,
        row=_portable_center_row(),
        base_position_scene=base,
        base_quat_wxyz=quat,
        contact_reference_root_z_scene=0.9,
        step_dt=0.02,
        table_surface_z_scene=0.76,
        back_integrate=truncated_reverse,
        venue_params=SimpleNamespace(ball_radius=0.02),
        geometry=geometry,
        serve_horizon_s=0.6,
        backint_h=0.005,
        plane_margin=0.005,
    )

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.full((2,), 0.6))
    torch.testing.assert_close(calls[1], torch.full((2,), 0.4))
    assert torch.equal(result["ttc_ticks"], torch.full((2,), 97))
    assert torch.equal(result["launch_horizon_ticks"], torch.full((2,), 20))
    task = result["task_f32"]
    torch.testing.assert_close(task[:, 0], torch.full((2,), 1.94))
    torch.testing.assert_close(task[:, 4], torch.full((2,), 0.9))
    # At the strike tick, exact measured-teacher mimic plus base-goal tracking
    # places the official site at task[5:8], and the ball centre is exactly the
    # selected-face geometry offset away.  This is the curriculum hand-off,
    # not a random rollout success proxy.
    torch.testing.assert_close(task[0, 5:8], torch.tensor([0.4, -0.5, 1.0]))
    torch.testing.assert_close(task[0, 14:17], torch.tensor([0.401, -0.48, 1.002]))
    torch.testing.assert_close(task[1, 5:8], torch.tensor([1.5, 2.4, 1.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(task[1, 14:17], torch.tensor([1.48, 2.401, 1.002]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(task[0, 24:26], torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(task[1, 24:26], torch.tensor([1.0, 2.1]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(task[0, 26:29], torch.tensor([-3.0, 0.0, 0.0]))
    torch.testing.assert_close(
        task[1, 26:29],
        torch.tensor([0.0, -3.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    # Launch is reverse-integrated from the action question.  It is neither
    # the legacy serve-range midpoint nor a forward gravity shortcut.
    torch.testing.assert_close(task[0, 32:35], torch.tensor([1.601, -0.48, 1.002]))
    assert not torch.equal(task[0, 32:35], torch.tensor([2.0, -0.775, 1.44]))


def test_portable_center_question_changes_when_action_center_changes():
    def reverse(contact, velocity, spin, tts, _params, **_kwargs):
        return contact - velocity * tts[:, None], velocity.clone(), tts.clone()

    kwargs = {
        "torch": torch,
        "base_position_scene": torch.tensor([[0.0, 0.0, 0.9]]),
        "base_quat_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "contact_reference_root_z_scene": 0.9,
        "step_dt": 0.02,
        "table_surface_z_scene": 0.76,
        "back_integrate": reverse,
        "venue_params": SimpleNamespace(ball_radius=0.02),
        "geometry": SimpleNamespace(
            BALL_RADIUS_M=0.02,
            face_normal_local=lambda sign: (0.0, float(sign), 0.0),
            ball_center_from_site_local=lambda sign: (
                0.001,
                0.02 if sign == 1 else -0.033,
                0.002,
            ),
            face_center_from_site_local=lambda sign: (
                0.001,
                0.0 if sign == 1 else -0.013,
                0.002,
            ),
            canonical_teacher_rate_from_site_speed=lambda required, reference, lo, hi: required / reference,
        ),
        "serve_horizon_s": 0.6,
        "backint_h": 0.005,
        "plane_margin": 0.005,
    }
    first = wait_env.portable_question.build_center_question(
        row=_portable_center_row(), **kwargs
    )
    second = wait_env.portable_question.build_center_question(
        row=_portable_center_row(
            contact_offset_center_b_yaw_m=(0.8, -0.5, 0.1),
            incoming_direction_center_b_yaw=(-0.8, 0.6, 0.0),
        ),
        **kwargs,
    )
    # The former independent contact offset is intentionally no longer a
    # second authority.  Incoming direction still changes the reverse launch,
    # while teacher/site geometry uniquely owns contact position.
    torch.testing.assert_close(
        first["task_f32"][:, 14:17], second["task_f32"][:, 14:17]
    )
    assert not torch.equal(
        first["task_f32"][:, 26:29], second["task_f32"][:, 26:29]
    )
    assert not torch.equal(
        first["launch_state_f32"][:, :3],
        second["launch_state_f32"][:, :3],
    )


def test_fresh_action_exact_mimic_closes_the_selected_face_contact_geometry():
    row = wait_env.portable_catalog.load_portable_action_center_table().fresh_action
    geometry = wait_env.racket_contact_geometry
    root_z = float(row.reference_racket_site_position_w_m[2]) - 0.1

    def reverse(contact, velocity, spin, tts, _params, **_kwargs):
        return contact - velocity * tts[:, None], velocity.clone(), tts.clone()

    result = wait_env.portable_question.build_center_question(
        torch=torch,
        row=row,
        base_position_scene=torch.tensor(
            [[*row.base_spawn_center_w_xy_m, root_z]], dtype=torch.float64
        ),
        base_quat_wxyz=torch.tensor(
            [row.reference_base_root_quat_wxyz], dtype=torch.float64
        ),
        contact_reference_root_z_scene=root_z,
        step_dt=0.02,
        table_surface_z_scene=0.76,
        back_integrate=reverse,
        venue_params=SimpleNamespace(ball_radius=geometry.BALL_RADIUS_M),
        geometry=geometry,
        serve_horizon_s=0.6,
        backint_h=0.005,
        plane_margin=0.005,
    )
    task = result["task_f32"][0]
    site = task[5:8]
    contact = task[14:17]
    racket_quat = task[20:24]
    ball_local = torch.tensor(
        [geometry.ball_center_from_site_local(row.mount_normal_sign)],
        dtype=task.dtype,
    )
    reconstructed = site + wait_env.portable_question._quat_apply_wxyz(
        torch, racket_quat[None, :], ball_local
    )[0]
    torch.testing.assert_close(contact, reconstructed, rtol=0.0, atol=1.0e-12)

    # With live yaw equal to the measured reference yaw, perfect base-goal and
    # teacher tracking recover the catalog's measured strike-site displacement.
    expected_site = torch.tensor(
        [
            task[24] + row.reference_reach_offset_xy_m[0],
            task[25] + row.reference_reach_offset_xy_m[1],
            row.reference_racket_site_position_w_m[2],
        ],
        dtype=task.dtype,
    )
    torch.testing.assert_close(site, expected_site, rtol=0.0, atol=2.0e-8)

    face = torch.tensor(
        geometry.face_center_from_site_local(row.mount_normal_sign),
        dtype=task.dtype,
    )
    normal = torch.tensor(
        geometry.face_normal_local(row.mount_normal_sign), dtype=task.dtype
    )
    separation = ball_local[0] - face
    torch.testing.assert_close(
        torch.dot(separation, normal),
        torch.tensor(geometry.BALL_RADIUS_M, dtype=task.dtype),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        separation - torch.dot(separation, normal) * normal,
        torch.zeros(3, dtype=task.dtype),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        task[4] + task[2], task[0], rtol=0.0, atol=1.0e-12
    )
    task_yaw = result["teacher_source_to_task_yaw_wxyz"]
    task_translation = result["teacher_source_to_task_translation_scene"]
    source_root = torch.tensor(
        [
            [
                row.reference_racket_site_position_w_m[0]
                - row.reference_reach_offset_xy_m[0],
                row.reference_racket_site_position_w_m[1]
                - row.reference_reach_offset_xy_m[1],
                root_z,
            ]
        ],
        dtype=task.dtype,
    )
    source_site = torch.tensor(
        [row.reference_racket_site_position_w_m], dtype=task.dtype
    )
    source_velocity = torch.tensor(
        [row.reference_racket_site_velocity_w_mps], dtype=task.dtype
    )
    source_quat = torch.tensor(
        [row.reference_racket_quat_wxyz], dtype=task.dtype
    )
    source_normal = wait_env.portable_question._quat_apply_wxyz(
        torch,
        source_quat,
        torch.tensor([[0.0, 1.0, 0.0]], dtype=task.dtype),
    )
    torch.testing.assert_close(
        wait_env.portable_question._quat_apply_wxyz(
            torch, task_yaw, source_root
        )[0]
        + task_translation[0],
        torch.stack((task[24], task[25], task.new_tensor(root_z))),
        rtol=0.0,
        atol=2.0e-8,
    )
    torch.testing.assert_close(
        wait_env.portable_question._quat_apply_wxyz(
            torch, task_yaw, source_site
        )[0]
        + task_translation[0],
        task[5:8],
        rtol=0.0,
        atol=2.0e-8,
    )
    torch.testing.assert_close(
        wait_env.portable_question._quat_apply_wxyz(
            torch, task_yaw, source_velocity
        )[0],
        task[8:11],
        rtol=0.0,
        atol=2.0e-8,
    )
    torch.testing.assert_close(
        wait_env.portable_question._quat_apply_wxyz(
            torch, task_yaw, source_normal
        )[0],
        task[11:14],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_negative_mount_perfect_mimic_keeps_raw_normal_and_signed_ball_offset():
    row = _portable_center_row(mount_normal_sign=-1)
    geometry = wait_env.racket_contact_geometry

    def reverse(contact, velocity, spin, tts, _params, **_kwargs):
        return contact - velocity * tts[:, None], velocity.clone(), tts.clone()

    result = wait_env.portable_question.build_center_question(
        torch=torch,
        row=row,
        base_position_scene=torch.tensor(
            [[0.0, 0.0, 0.9]], dtype=torch.float64
        ),
        base_quat_wxyz=torch.tensor(
            [row.reference_base_root_quat_wxyz], dtype=torch.float64
        ),
        contact_reference_root_z_scene=0.9,
        step_dt=0.02,
        table_surface_z_scene=0.76,
        back_integrate=reverse,
        venue_params=SimpleNamespace(ball_radius=geometry.BALL_RADIUS_M),
        geometry=geometry,
        serve_horizon_s=0.6,
        backint_h=0.005,
        plane_margin=0.005,
    )
    task = result["task_f32"][0]
    raw_normal = torch.tensor([0.0, 1.0, 0.0], dtype=task.dtype)
    selected_normal = torch.tensor(
        geometry.face_normal_local(-1), dtype=task.dtype
    )

    # Observation V3's task tail subtracts the live raw A/+Y normal from
    # task[11:14]; its separate motion residual uses the physical signed face.
    # A perfect negative-face mimic must therefore carry exactly zero actor
    # residual instead of the old sign-flipped two-unit residual.
    torch.testing.assert_close(
        task[11:14] - raw_normal,
        torch.zeros(3, dtype=task.dtype),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.dot(task[11:14], selected_normal) == -1.0

    # R03 packs task[11:14] as target and raw +Y as achieved normal.
    fact = torch.zeros((1, 4, 32), dtype=task.dtype)
    fact[0, 1, 0:15] = task[5:20]
    fact[0, 1, 15:24] = task[5:14]
    normal_error = torch.acos(
        torch.sum(fact[0, 1, 6:9] * fact[0, 1, 21:24]).clamp(-1.0, 1.0)
    )
    assert normal_error == 0.0
    valid = torch.zeros((1, 4), dtype=torch.int64)
    valid[0, 1] = (
        wait_env.portable_reward.R03_PRESENT
        | wait_env.portable_reward.R03_PHYSICALLY_VALID
    )
    terms = wait_env.portable_reward.lifecycle_reward14(
        valid_bits=valid,
        fact_f32=fact,
        owner_fault_bits=torch.zeros_like(valid),
        step_dt=0.02,
    )
    torch.testing.assert_close(
        terms[0, torch.tensor([2, 5, 8])],
        torch.full((3,), 0.02, dtype=task.dtype),
        rtol=0.0,
        atol=0.0,
    )

    # Only physical face selection remains signed: the ball centre is exactly
    # one radius outside the negative rubber, and task contact uses that point.
    face_local = torch.tensor(
        geometry.face_center_from_site_local(-1), dtype=task.dtype
    )
    ball_local = torch.tensor(
        geometry.ball_center_from_site_local(-1), dtype=task.dtype
    )
    separation = ball_local - face_local
    torch.testing.assert_close(
        torch.dot(separation, selected_normal),
        torch.tensor(geometry.BALL_RADIUS_M, dtype=task.dtype),
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        separation
        - torch.dot(separation, selected_normal) * selected_normal,
        torch.zeros(3, dtype=task.dtype),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert ball_local[1] < face_local[1] < 0.0
    reconstructed = task[5:8] + wait_env.portable_question._quat_apply_wxyz(
        torch, task[20:24][None, :], ball_local[None, :]
    )[0]
    torch.testing.assert_close(
        task[14:17], reconstructed, rtol=0.0, atol=1.0e-15
    )


def test_full_a_reveal_packs_env_local_question_and_launch_restores_world_origin():
    env = _host_full_a_lifecycle_env()
    origins = torch.tensor([[6.0, -12.0, 0.0], [-6.0, 12.0, 0.0]])
    local_base = torch.tensor([[0.1, -0.2, 0.9], [0.3, -0.4, 0.9]])
    env.env.scene.env_origins = origins
    env.sim.data.qpos[:, :3] = origins + local_base
    frozen_base = env._full_a_frozen_root_position_scene.clone()
    captured = {}

    def question(**kwargs):
        captured["base"] = kwargs["base_position_scene"].clone()
        task = torch.zeros((2, P.TASK_F32_WIDTH))
        launch = torch.zeros((2, 13))
        launch[:, :3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        launch[:, 3] = 1.0
        return {
            "task_f32": task,
            "launch_state_f32": launch,
            "ttc_ticks": torch.full((2,), 4, dtype=torch.long),
            "launch_horizon_ticks": torch.full((2,), 2, dtype=torch.long),
            "teacher_rate": torch.ones(2),
            "scaled_t_hit_s": torch.ones(2),
            "scaled_t_cycle_s": torch.full((2,), 2.0),
            "pre_swing_wait_s": torch.ones(2),
            "teacher_source_to_task_yaw_wxyz": torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            ).expand(2, 4),
            "teacher_source_to_task_translation_scene": torch.zeros(2, 3),
        }

    env._full_a_question_builder = question
    ids = torch.arange(2)
    env.common_step_counter = 1
    wait_env.FullMdpInitialWaitVecEnv._full_a_reveal_rows(env, ids)
    env._full_a_next_reveal_tick[:] = 295
    torch.testing.assert_close(captured["base"], frozen_base)
    assert not torch.equal(captured["base"], local_base)
    env.sim.data.qpos[:, env.b_q : env.b_q + 3] = -100.0
    env.sim.data.qvel[:, env.b_v : env.b_v + 6] = 10.0
    env.common_step_counter = 1
    _scheduled_due, launch, _missed_launch = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    )
    assert not launch.any()
    torch.testing.assert_close(
        env.sim.data.qpos[:, env.b_q : env.b_q + 3],
        origins
        + torch.tensor(wait_env.WAIT_BALL_PARK_HOPE)
        + env.hope_to_scene,
    )
    assert torch.count_nonzero(
        env.sim.data.qvel[:, env.b_v : env.b_v + 6]
    ) == 0
    env.common_step_counter = 3
    _scheduled_due, launch, _missed_launch = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    )
    assert launch.all()
    torch.testing.assert_close(
        env.sim.data.qpos[:, env.b_q : env.b_q + 3],
        origins + env._full_a_launch_state_f32[:, :3],
    )


def test_launch_tick_is_post_transition_boundary_and_fixed_tape_is_h_times_dt():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_REVEAL_COMMITTED)
    env._epoch_clock_ticks[:, 2] = 5
    env._full_a_launch_state_f32.zero_()
    env._full_a_launch_state_f32[:, 3] = 1.0
    env._full_a_launch_state_f32[:, 7] = 1.0
    env._full_a_next_reveal_tick.fill_(100)
    env.episode_length_buf.zero_()

    env.common_step_counter = 4
    _scheduled_due, launch, _missed_launch = env._full_a_prepare_step()
    assert not launch.any()
    env.common_step_counter = 5
    _scheduled_due, launch, _missed_launch = env._full_a_prepare_step()
    assert launch.all()

    dt, horizon, contact_tick = 0.02, 20, 25
    for boundary in range(6, contact_tick + 1):
        env.sim.data.qpos[:, env.b_q] += (
            env.sim.data.qvel[:, env.b_v] * dt
        )
        env.common_step_counter = boundary
    torch.testing.assert_close(
        env.sim.data.qpos[:, env.b_q],
        torch.full((2,), 0.40),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert horizon == contact_tick - 5
    assert not torch.isclose(env.sim.data.qpos[0, env.b_q], torch.tensor(0.42))


def test_missed_launch_tick_is_returned_as_named_chronology_fault_not_catchup():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_REVEAL_COMMITTED)
    env._epoch_clock_ticks[:, 2] = 4
    env._full_a_next_reveal_tick.fill_(100)
    env.episode_length_buf.zero_()
    env.common_step_counter = 5

    _scheduled_due, launch, missed_launch = env._full_a_prepare_step()
    assert missed_launch.all()
    assert not launch.any()
    assert "_assert_async" not in inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step
    )


def test_portable_motion_teacher_waits_then_hits_the_catalog_strike_frame():
    frames = 96
    joint = torch.arange(frames, dtype=torch.float32).reshape(frames, 1).repeat(1, 31)
    body_pos = (
        torch.arange(frames, dtype=torch.float32)
        .reshape(frames, 1, 1)
        .repeat(1, 2, 3)
    )
    body_quat = torch.zeros((frames, 2, 4))
    body_quat[..., 0] = 1.0
    teacher = wait_env.portable_question.PortableMotionTeacher(
        fps=50.0,
        strike_frame=52,
        contact_reference_root_z_scene=0.9,
        joint_pos=joint,
        joint_vel=torch.ones_like(joint),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=torch.ones_like(body_pos),
        body_ang_vel_w=torch.ones_like(body_pos) * 2.0,
        measured_racket_site_pos_w=body_pos[:, 0],
        measured_racket_site_lin_vel_w=torch.ones((frames, 3)) * 50.0,
        measured_racket_normal_w=torch.tensor([[0.0, 1.0, 0.0]]).repeat(
            frames, 1
        ),
        measured_racket_long_axis_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(
            frames, 1
        ),
    )
    sampled = wait_env.portable_question.sample_motion_teacher(
        torch,
        teacher,
        elapsed_s=torch.tensor([0.5, 1.94]),
        teacher_rate=torch.ones(2),
        pre_swing_wait_s=torch.full((2,), 0.9),
    )
    assert torch.equal(sampled["frame"], torch.tensor([0, teacher.strike_frame]))
    assert torch.count_nonzero(sampled["joint_vel"][0]) == 0
    assert torch.equal(sampled["joint_pos"][1], joint[teacher.strike_frame])
    assert torch.equal(sampled["body_lin_vel_w"][1], torch.ones((2, 3)))
    assert torch.count_nonzero(sampled["measured_racket_site_lin_vel_w"][0]) == 0
    assert torch.equal(
        sampled["measured_racket_site_lin_vel_w"][1],
        torch.full((3,), 50.0),
    )


def test_portable_teacher_loads_schema_v4_and_derives_site_velocity(tmp_path):
    np = pytest.importorskip("numpy")
    frames = 3
    motion_file = tmp_path / "motion.npz"
    joint = np.zeros((frames, 31), dtype=np.float32)
    body_pos = np.zeros((frames, 1, 3), dtype=np.float32)
    body_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    measured_pos = np.asarray(
        [[0.00, 0.0, 1.0], [0.02, 0.0, 1.0], [0.06, 0.0, 1.0]],
        dtype=np.float32,
    )
    contract_sha = hashlib.sha256(
        wait_env.portable_question._JOINT_ORDER_CONTRACT.read_bytes()
    ).hexdigest()
    np.savez(
        motion_file,
        fps=np.asarray([50.0], dtype=np.float64),
        joint_pos=joint,
        joint_vel=joint,
        body_names=np.asarray(["pelvis_link"]),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_pos,
        body_ang_vel_w=body_pos,
        kinematics_schema_version=np.asarray([2], dtype=np.int64),
        body_pos_point=np.asarray(["link_origin"]),
        body_lin_vel_point=np.asarray(["center_of_mass"]),
        measured_racket_site_pos_w=measured_pos,
        measured_racket_normal_w=np.tile(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (frames, 1)
        ),
        measured_racket_long_axis_w=np.tile(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
        ),
        measured_racket_schema_version=np.asarray([4], dtype=np.int64),
        measured_racket_position_semantics=np.asarray(["physical_blade_center"]),
        measured_racket_normal_semantics=np.asarray(
            ["signed_physical_hitting_face"]
        ),
        measured_racket_long_axis_semantics=np.asarray(
            ["measured_paddle_butt_to_blade"]
        ),
        measured_racket_retarget_admitted=np.asarray([1], dtype=np.int64),
        measured_racket_robot_mount_normal_sign=np.asarray([1], dtype=np.int64),
        measured_racket_joint_order_contract_id=np.asarray(
            ["a3-gmr-dof-pos-to-runtime-articulation-v1"]
        ),
        measured_racket_joint_order_contract_sha256=np.asarray([contract_sha]),
    )
    row = SimpleNamespace(
        motion_file=str(motion_file),
        motion_sha256=hashlib.sha256(motion_file.read_bytes()).hexdigest(),
        mount_normal_sign=1,
        strike_phase=0.5,
        reference_t_hit_s=0.02,
        reference_t_cycle_s=0.04,
    )
    teacher = wait_env.portable_question.load_portable_motion_teacher(
        row=row,
        tracked_body_names=("pelvis_link",),
        torch=torch,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        teacher.measured_racket_site_lin_vel_w[:, 0],
        torch.tensor([1.0, 1.5, 2.0]),
    )
    sampled = wait_env.portable_question.sample_motion_teacher(
        torch,
        teacher,
        elapsed_s=torch.tensor([0.01]),
        teacher_rate=torch.tensor([2.0]),
        pre_swing_wait_s=torch.tensor([0.0]),
    )
    assert sampled["frame"].item() == 1
    torch.testing.assert_close(
        sampled["measured_racket_site_lin_vel_w"][0],
        torch.tensor([3.0, 0.0, 0.0]),
    )


def test_diagnostic_qdes_bridge_is_continuous_finite_and_exact_at_endpoints():
    previous = torch.zeros((1, 2))
    frame0 = torch.tensor([[2.0, 4.0]])
    sent = []
    for frozen in (4, 3, 2, 1, 0):
        previous = (
            wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge(
                torch=torch,
                previous_qdes=previous,
                frame0_qdes=frame0,
                frozen_steps=torch.tensor([frozen], dtype=torch.long),
            )
        )
        sent.append(previous.clone())

    torch.testing.assert_close(
        torch.cat(sent, dim=0),
        torch.tensor(
            [[0.4, 0.8], [0.8, 1.6], [1.2, 2.4], [1.6, 3.2], [2.0, 4.0]]
        ),
    )
    assert torch.equal(sent[-1], frame0)
    assert torch.isfinite(torch.cat(sent)).all()

    with pytest.raises(RuntimeError):
        wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge(
            torch=torch,
            previous_qdes=previous,
            frame0_qdes=frame0,
            frozen_steps=torch.tensor([-1], dtype=torch.long),
        )


def test_full_a_environment_consumes_the_measured_teacher_clock():
    env = _host_full_a_lifecycle_env()
    env.q_ready = torch.full((31,), -7.0)
    origins = torch.tensor([[6.0, -12.0, 0.0], [-6.0, 12.0, 0.0]])
    env.env.scene.env_origins = origins
    frames = 4
    joint = (
        torch.arange(frames, dtype=torch.float32)
        .reshape(frames, 1)
        .repeat(1, 31)
    )
    body_pos = (
        torch.arange(frames, dtype=torch.float32)
        .reshape(frames, 1, 1)
        .repeat(1, 1, 3)
    )
    body_quat = torch.zeros((frames, 1, 4))
    body_quat[..., 0] = 1.0
    env._full_a_teacher = wait_env.portable_question.PortableMotionTeacher(
        fps=50.0,
        strike_frame=1,
        contact_reference_root_z_scene=0.9,
        joint_pos=joint,
        joint_vel=torch.ones_like(joint),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=torch.ones_like(body_pos),
        body_ang_vel_w=torch.ones_like(body_pos) * 2.0,
        measured_racket_site_pos_w=body_pos[:, 0],
        measured_racket_site_lin_vel_w=torch.ones((frames, 3)),
        measured_racket_normal_w=torch.tensor([[0.0, 1.0, 0.0]]).repeat(
            frames, 1
        ),
        measured_racket_long_axis_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(
            frames, 1
        ),
    )
    env._ready_teacher_body_pos = torch.tensor(
        [[[0.25, -0.10, 1.05]], [[-0.20, 0.15, 1.10]]]
    ) + origins[:, None, :]
    env._ready_teacher_body_quat = torch.zeros((2, 1, 4))
    env._ready_teacher_body_quat[..., 1] = 1.0
    env._ready_teacher_body_lin_vel = torch.zeros((2, 1, 3))
    env._ready_teacher_body_ang_vel = torch.zeros((2, 1, 3))
    env._teacher_body_pos = env._ready_teacher_body_pos.clone()
    env._teacher_body_quat = env._ready_teacher_body_quat.clone()
    env._teacher_body_lin_vel = env._ready_teacher_body_lin_vel.clone()
    env._teacher_body_ang_vel = env._ready_teacher_body_ang_vel.clone()
    env._ready_teacher_racket_site_pos_w = (
        env._ready_teacher_body_pos[:, 0] + torch.tensor([0.2, 0.0, 0.0])
    )
    env._ready_teacher_racket_signed_normal_w = torch.tensor(
        [[0.0, 1.0, 0.0]]
    ).repeat(2, 1)
    env._ready_teacher_racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0]]
    ).repeat(2, 1)
    env._teacher_racket_site_pos_w = (
        env._ready_teacher_racket_site_pos_w.clone()
    )
    env._teacher_racket_site_lin_vel_w = torch.zeros((2, 3))
    env._teacher_racket_signed_normal_w = (
        env._ready_teacher_racket_signed_normal_w.clone()
    )
    env._teacher_racket_long_axis_w = (
        env._ready_teacher_racket_long_axis_w.clone()
    )
    env._refresh_aligned_teacher_body_pose = lambda: None

    # Before curriculum reveal, body and joint teachers describe the same
    # reset-ready hold.  Action frame zero is intentionally different.
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(env._teacher_body_pos, env._ready_teacher_body_pos)
    assert torch.equal(env._teacher_body_quat, env._ready_teacher_body_quat)
    assert torch.equal(
        env._full_a_teacher_joint_pos,
        env.q_ready.unsqueeze(0).repeat(2, 1),
    )
    assert torch.equal(
        env._teacher_racket_site_pos_w,
        env._ready_teacher_racket_site_pos_w,
    )
    assert torch.count_nonzero(env._teacher_racket_site_lin_vel_w) == 0
    assert torch.equal(
        env._teacher_racket_signed_normal_w,
        env._ready_teacher_racket_signed_normal_w,
    )

    _prepare_and_settle(env)

    env.common_step_counter = 3
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)

    assert torch.equal(env._full_a_teacher_frame, torch.ones(2, dtype=torch.long))
    assert torch.equal(env._full_a_teacher_joint_pos, joint[1].repeat(2, 1))
    assert torch.equal(env._full_a_teacher_joint_vel, torch.ones((2, 31)))
    assert torch.equal(
        env._teacher_body_pos,
        body_pos[1].repeat(2, 1, 1) + origins[:, None, :],
    )
    assert torch.equal(
        env._teacher_racket_site_pos_w,
        body_pos[1, 0].repeat(2, 1) + origins,
    )
    assert torch.equal(
        env._teacher_racket_site_lin_vel_w, torch.ones((2, 3))
    )
    assert torch.equal(
        env._teacher_racket_signed_normal_w,
        torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1),
    )
    assert torch.equal(
        env._teacher_racket_long_axis_w,
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
    )
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.ones(2, dtype=torch.long),
    )

    env.common_step_counter = 4
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.full(
            (2,), wait_env.FULL_A_MOTION_FOLLOW_PHASE_INDEX, dtype=torch.long
        ),
    )
    # The stub declares a 40 ms active clip, so the exact close boundary is
    # its measured frame two.  It must not fall back to reset-ready until the
    # following control tick.
    assert torch.equal(env._full_a_teacher_frame, torch.full((2,), 2))
    assert torch.equal(env._full_a_teacher_joint_pos, joint[2].repeat(2, 1))
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) > 0
    assert bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).all()
    )

    env.common_step_counter = 5
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.full(
            (2,), wait_env.READY_HOLD_PHASE_INDEX, dtype=torch.long
        ),
    )
    assert torch.equal(
        env._full_a_teacher_joint_pos,
        joint[0].repeat(2, 1),
    )
    assert torch.equal(
        env._teacher_body_pos,
        body_pos[0].repeat(2, 1, 1) + origins[:, None, :],
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0
    assert torch.count_nonzero(env._teacher_body_lin_vel) == 0
    assert torch.count_nonzero(env._teacher_body_ang_vel) == 0
    assert torch.equal(
        env._teacher_racket_site_pos_w,
        body_pos[0, 0].repeat(2, 1) + origins,
    )
    assert torch.count_nonzero(env._teacher_racket_site_lin_vel_w) == 0
    assert not bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).any()
    )


def test_full_a_teacher_holds_coherent_frame0_then_plays_and_retires_rowwise():
    env = _host_full_a_lifecycle_env()
    env.q_ready = torch.linspace(-2.0, 1.0, 31)
    origins = torch.tensor([[6.0, -12.0, 0.0], [-6.0, 12.0, 0.0]])
    env.env.scene.env_origins = origins
    frames = 96
    joint = (
        torch.linspace(3.0, 4.0, 31).unsqueeze(0)
        + torch.arange(frames, dtype=torch.float32)[:, None] * 0.01
    )
    joint_vel = torch.ones_like(joint) * 0.25
    body_pos = torch.zeros((frames, 1, 3))
    body_pos[:, 0, 0] = torch.arange(frames, dtype=torch.float32) * 0.01
    body_pos[:, 0, 1] = -0.4
    body_pos[:, 0, 2] = 0.9
    body_quat = torch.zeros((frames, 1, 4))
    body_quat[..., 0] = 2.0**-0.5
    body_quat[..., 3] = 2.0**-0.5
    body_vel = torch.ones_like(body_pos) * 0.5
    env._full_a_teacher = wait_env.portable_question.PortableMotionTeacher(
        fps=50.0,
        strike_frame=52,
        contact_reference_root_z_scene=0.9,
        joint_pos=joint,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_vel,
        body_ang_vel_w=body_vel * 2.0,
        measured_racket_site_pos_w=body_pos[:, 0],
        measured_racket_site_lin_vel_w=torch.tensor(
            [[0.5, 0.0, 0.0]]
        ).repeat(frames, 1),
        measured_racket_normal_w=torch.tensor([[0.0, 1.0, 0.0]]).repeat(
            frames, 1
        ),
        measured_racket_long_axis_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(
            frames, 1
        ),
    )
    ready_pos_local = torch.tensor(
        [[[0.2, -0.1, 1.1]], [[-0.3, 0.2, 1.05]]]
    )
    env._ready_teacher_body_pos = ready_pos_local + origins[:, None, :]
    env._ready_teacher_body_quat = torch.zeros((2, 1, 4))
    env._ready_teacher_body_quat[..., 0] = 1.0
    env._ready_teacher_body_lin_vel = torch.zeros((2, 1, 3))
    env._ready_teacher_body_ang_vel = torch.zeros((2, 1, 3))
    env._teacher_body_pos = env._ready_teacher_body_pos.clone()
    env._teacher_body_quat = env._ready_teacher_body_quat.clone()
    env._teacher_body_lin_vel = env._ready_teacher_body_lin_vel.clone()
    env._teacher_body_ang_vel = env._ready_teacher_body_ang_vel.clone()
    env._ready_teacher_racket_site_pos_w = (
        env._ready_teacher_body_pos[:, 0] + torch.tensor([0.2, 0.0, 0.0])
    )
    env._ready_teacher_racket_signed_normal_w = torch.tensor(
        [[0.0, 1.0, 0.0]]
    ).repeat(2, 1)
    env._ready_teacher_racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0]]
    ).repeat(2, 1)
    env._teacher_racket_site_pos_w = (
        env._ready_teacher_racket_site_pos_w.clone()
    )
    env._teacher_racket_site_lin_vel_w = torch.zeros((2, 3))
    env._teacher_racket_signed_normal_w = (
        env._ready_teacher_racket_signed_normal_w.clone()
    )
    env._teacher_racket_long_axis_w = (
        env._ready_teacher_racket_long_axis_w.clone()
    )
    env._refresh_aligned_teacher_body_pose = lambda: None

    def question(**kwargs):
        count = kwargs["base_position_scene"].shape[0]
        task = torch.zeros((count, P.TASK_F32_WIDTH))
        launch = torch.zeros((count, 13))
        launch[:, 3] = 1.0
        return {
            "task_f32": task,
            "launch_state_f32": launch,
            "ttc_ticks": torch.full((count,), 97, dtype=torch.long),
            "launch_horizon_ticks": torch.full((count,), 20, dtype=torch.long),
            "teacher_rate": torch.ones(count),
            "scaled_t_hit_s": torch.full((count,), 1.04),
            "scaled_t_cycle_s": torch.full((count,), 1.90),
            "pre_swing_wait_s": torch.full((count,), 0.90),
            "teacher_source_to_task_yaw_wxyz": torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            ).expand(count, 4),
            "teacher_source_to_task_translation_scene": torch.zeros(
                count, 3
            ),
        }

    env._full_a_question_builder = question
    physical_root_before = env.sim.data.qpos[:, :7].clone()
    _prepare_and_settle(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)

    # Reveal atomically switches both joint and body teachers to the same
    # stationary frame-zero preparation target.  The clip clock stays frozen;
    # physical qpos still carries continuously from the learned ready pose.
    assert torch.max(torch.abs(joint[0] - env.q_ready)) > 3.0
    assert torch.equal(env.sim.data.qpos[:, :7], physical_root_before)
    assert not hasattr(env, "_full_a_ready_bridge_alpha")
    assert torch.equal(
        env._full_a_teacher_joint_pos,
        joint[0].repeat(2, 1),
    )
    assert torch.equal(
        env._teacher_body_pos, body_pos[0].repeat(2, 1, 1) + origins[:, None, :]
    )
    assert torch.equal(
        env._teacher_body_quat, body_quat[0].repeat(2, 1, 1)
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0
    assert torch.count_nonzero(env._teacher_body_lin_vel) == 0
    assert torch.count_nonzero(env._teacher_body_ang_vel) == 0
    assert torch.equal(
        env._teacher_racket_site_pos_w,
        body_pos[0, 0].repeat(2, 1) + origins,
    )
    assert torch.count_nonzero(env._teacher_racket_site_lin_vel_w) == 0
    assert torch.equal(env._full_a_teacher_frame, torch.zeros(2, dtype=torch.long))
    assert torch.equal(env._full_a_motion_phase_code, torch.zeros(2, dtype=torch.long))

    # The entire preparation window keeps the coherent frame-zero target.
    env.common_step_counter = 22
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_teacher_joint_pos,
        joint[0].repeat(2, 1),
    )
    assert torch.equal(
        env._teacher_body_pos, body_pos[0].repeat(2, 1, 1) + origins[:, None, :]
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0
    assert torch.equal(env._full_a_motion_phase_code, torch.zeros(2, dtype=torch.long))

    # The exact end of the pre-wait is still stationary frame-zero HOLD.
    env.common_step_counter = 46
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_teacher_joint_pos,
        joint[0].repeat(2, 1),
    )
    assert torch.equal(
        env._teacher_body_pos, body_pos[0].repeat(2, 1, 1) + origins[:, None, :]
    )
    assert torch.equal(env._full_a_teacher_frame, torch.zeros(2, dtype=torch.long))
    assert torch.equal(env._full_a_motion_phase_code, torch.zeros(2, dtype=torch.long))
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0
    assert not bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).any()
    )

    # A half-rate row still rounds to frame zero on the first positive clock
    # tick.  It remains PREPARE with zero teacher velocity and stays out of
    # the playback-only reward denominator until the measured pose advances.
    env._full_a_teacher_rate[0] = 0.5
    env.common_step_counter = 47
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(env._full_a_teacher_frame, torch.tensor([0, 1]))
    assert torch.equal(
        env._full_a_teacher_joint_pos,
        torch.stack((joint[0], joint[1])),
    )
    assert torch.equal(
        env._full_a_teacher_joint_vel,
        torch.stack((torch.zeros_like(joint_vel[0]), joint_vel[1])),
    )
    assert torch.equal(
        env._teacher_racket_site_pos_w,
        torch.stack((body_pos[0, 0], body_pos[1, 0])) + origins,
    )
    assert torch.equal(
        env._teacher_racket_site_lin_vel_w,
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    assert torch.equal(
        env._full_a_motion_phase_code, torch.tensor([0, 1], dtype=torch.long)
    )
    assert torch.equal(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(env),
        torch.tensor([False, True]),
    )

    env.common_step_counter = 48
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(env._full_a_teacher_frame, torch.tensor([1, 2]))
    assert torch.equal(env._full_a_motion_phase_code, torch.ones(2, dtype=torch.long))
    assert bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).all()
    )

    env._full_a_teacher_rate.fill_(1.0)
    env.common_step_counter = 98
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_teacher_frame,
        torch.full((2,), env._full_a_teacher.strike_frame, dtype=torch.long),
    )
    torch.testing.assert_close(
        env._full_a_teacher_joint_pos,
        joint[env._full_a_teacher.strike_frame].repeat(2, 1),
    )
    assert torch.equal(env._full_a_motion_phase_code, torch.ones(2, dtype=torch.long))

    # Physical outcome is not a Motion writer.  An outcome that arrives before
    # the accepted clip suffix ends must leave measured follow-through open.
    env.common_step_counter = 120
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.full((2,), wait_env.FULL_A_MOTION_FOLLOW_PHASE_INDEX),
    )
    assert bool(
        (env._full_a_teacher_frame > env._full_a_teacher.strike_frame).all()
    )
    torch.testing.assert_close(
        env._full_a_teacher_joint_pos,
        joint[env._full_a_teacher_frame],
    )
    torch.testing.assert_close(
        env._teacher_body_pos,
        body_pos[env._full_a_teacher_frame] + origins[:, None, :],
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) > 0
    assert torch.count_nonzero(env._teacher_body_lin_vel) > 0
    assert torch.count_nonzero(env._teacher_body_ang_vel) > 0
    torch.testing.assert_close(
        env._teacher_racket_site_pos_w,
        body_pos[env._full_a_teacher_frame, 0] + origins,
    )
    assert torch.count_nonzero(env._teacher_racket_site_lin_vel_w) > 0
    assert bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).all()
    )

    # Due-before-add semantics retain the final measured boundary at the exact
    # integer task-close tick.  The following tick atomically returns every
    # measured target to frame zero with zero velocities.
    env.common_step_counter = 141
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.full(
            (2,), wait_env.FULL_A_MOTION_FOLLOW_PHASE_INDEX, dtype=torch.long
        ),
    )
    assert torch.equal(
        env._full_a_teacher_frame,
        torch.full((2,), frames - 1, dtype=torch.long),
    )
    assert torch.equal(
        env._full_a_teacher_joint_pos, joint[-1].repeat(2, 1)
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) > 0
    assert torch.count_nonzero(env._teacher_racket_site_lin_vel_w) > 0
    assert bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).all()
    )

    env.common_step_counter = 142
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.full(
            (2,), wait_env.READY_HOLD_PHASE_INDEX, dtype=torch.long
        ),
    )
    assert torch.equal(env._full_a_teacher_joint_pos, joint[0].repeat(2, 1))
    assert torch.equal(
        env._teacher_body_pos,
        body_pos[0].repeat(2, 1, 1) + origins[:, None, :],
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0
    assert torch.count_nonzero(env._teacher_body_lin_vel) == 0
    assert torch.count_nonzero(env._teacher_body_ang_vel) == 0
    assert torch.equal(
        env._teacher_racket_site_pos_w,
        body_pos[0, 0].repeat(2, 1) + origins,
    )
    assert torch.count_nonzero(env._teacher_racket_site_lin_vel_w) == 0
    assert not bool(
        wait_env.FullMdpInitialWaitVecEnv._full_a_paddle_prior_playback_mask(
            env
        ).any()
    )

    # RETIRE is nonterminating and retains the closed row until the next real
    # ACCEPT; it never increments true-reset generation or touches the peer.
    peer_joint = env._full_a_teacher_joint_pos[1].clone()
    peer_body = env._teacher_body_pos[1].clone()
    generation = env.reset_generation.clone()
    env._epoch_phase[0] = wait_env.FULL_A_PHASE_RETIRED
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    torch.testing.assert_close(env._full_a_teacher_joint_pos[0], joint[0])
    torch.testing.assert_close(
        env._teacher_body_pos[0], body_pos[0] + origins[0]
    )
    torch.testing.assert_close(env._full_a_teacher_joint_pos[1], peer_joint)
    torch.testing.assert_close(env._teacher_body_pos[1], peer_body)
    env.episode_length_buf[0] = 294
    env._full_a_next_reveal_tick[0] = 295
    reveal, _launch, due, deferred, _missed_launch = _prepare_and_settle(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert due[0] and reveal[0] and not deferred[0]
    assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_REVEAL_COMMITTED
    assert torch.equal(env.sim.data.qpos[:, :7], physical_root_before)
    assert torch.equal(env._full_a_teacher_joint_pos[0], joint[0])
    assert torch.count_nonzero(env._full_a_teacher_joint_vel[0]) == 0
    torch.testing.assert_close(env._full_a_teacher_joint_pos[1], peer_joint)
    assert torch.equal(env.reset_generation, generation)


def test_semantic_v3_drops_raw_task_owner_fault_and_reward_ledger_columns():
    actor = {name for name, _width in P.ACTOR_LAYOUT_V3}
    critic = {name for name, _width in P.CRITIC_EXTENSION_LAYOUT_V3}
    assert not actor.intersection(
        {"epoch_task_f32", "epoch_clock_remaining_s", "epoch_selected", "epoch_launch_succeeded"}
    )
    assert not critic.intersection(
        {
            "physical_r03_r06_r07_fact_present",
            "physical_r03_r06_r07_fact_age_s",
            "physical_r03_r06_r07_fact_f32",
            "physical_r03_r06_r07_fault_present",
            "reward_due",
            "reward_paid",
        }
    )


def test_full_a_constructor_observation_stays_wait_until_buffers_are_installed():
    env = _host_full_a_lifecycle_env()
    env._fullmdp_initialized = False
    env.actions = torch.zeros((2, 31))
    env._qpos_act = lambda: torch.zeros((2, 31))
    env._qvel_act = lambda: torch.zeros((2, 31))
    env._con_geom = torch.tensor([[0, 0]], dtype=torch.long)
    env._con_idx = torch.tensor([0], dtype=torch.long)
    env._nacon = torch.tensor([0], dtype=torch.long)
    env._ball_gid = 2
    del env._full_a_motion_phase_code
    del env._full_a_teacher_joint_pos
    del env._full_a_teacher_joint_vel

    wait_env.FullMdpInitialWaitVecEnv._compute_obs(
        env,
        st={
            "proj_g": torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
            "base_ang_b": torch.zeros((2, 3)),
        },
    )

    offsets = {}
    start = 0
    for name, width in P.ACTOR_LAYOUT_V1:
        offsets[name] = slice(start, start + width)
        start += width
    phase = env._obs_buf[:, offsets["motion_phase_one_hot"]]
    assert torch.equal(
        phase[:, wait_env.READY_HOLD_PHASE_INDEX], torch.ones(2)
    )
    assert torch.count_nonzero(env._obs_buf[:, offsets["teacher_joint_pos_rel"]]) == 0
    assert torch.count_nonzero(env._obs_buf[:, offsets["teacher_joint_vel_rel"]]) == 0

    env._con_geom = torch.tensor([[env._ball_gid, 0]], dtype=torch.long)
    env._nacon = torch.tensor([1], dtype=torch.long)
    with pytest.raises(RuntimeError, match="initial-WAIT ball is in contact"):
        wait_env.FullMdpInitialWaitVecEnv._compute_obs(
            env,
            st={
                "proj_g": torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
                "base_ang_b": torch.zeros((2, 3)),
            },
        )


def test_host_full_a_outcome_uses_shared_contact_and_crossing_horizon_clocks():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._epoch_clock_ticks[:, 3] = torch.tensor([3, 3])
    env._epoch_clock_ticks[:, 4] = torch.tensor([4, 5])
    env._full_a_selected_racket_contact[1] = True
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()
    ball_pos[:, 0] = -2.0

    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )
    assert not settled.any()
    assert torch.count_nonzero(outcome) == 0

    env.common_step_counter = 3
    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )
    assert torch.equal(settled, torch.tensor([True, False]))
    assert torch.equal(
        outcome,
        torch.tensor(
            [
                wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED,
                wait_env.FULL_A_OUTCOME_NONE,
            ]
        ),
    )
    assert env._epoch_clock_ticks[0, 3] == 3

    env.common_step_counter = 5
    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )
    assert torch.equal(settled, torch.tensor([False, True]))
    assert torch.equal(
        outcome,
        torch.tensor(
            [
                wait_env.FULL_A_OUTCOME_NONE,
                wait_env.FULL_A_OUTCOME_OUT,
            ]
        ),
    )
    assert env._epoch_clock_ticks[1, 3] == 3
    assert torch.equal(
        env._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_OUTCOME_SETTLED),
    )


def test_host_full_a_outcome_does_not_catch_up_missed_exact_boundaries():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._epoch_clock_ticks[:, 3] = 3
    env._epoch_clock_ticks[:, 4] = 3
    env._full_a_selected_racket_contact[1] = True
    env.common_step_counter = 4
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()

    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )

    assert not settled.any()
    assert torch.count_nonzero(outcome) == 0
    assert env._epoch_phase.eq(wait_env.FULL_A_PHASE_LAUNCH_SETTLED).all()


def test_host_full_a_invalid_contact_settles_r06_fault_with_finite_ball():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._full_a_invalid_contact_event[0] = True
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()

    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )
    assert torch.equal(settled, torch.tensor([True, False]))
    assert torch.equal(
        outcome,
        torch.tensor(
            [wait_env.FULL_A_OUTCOME_INVALID, wait_env.FULL_A_OUTCOME_NONE]
        ),
    )
    present, source_valid, common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env, settled, outcome
        )
    )
    assert torch.equal(present, torch.tensor([True, False]))
    assert not source_valid.any() and not common.any()
    assert env._full_a_owner_fault_bits[0, 2] == 1
    assert torch.equal(
        env._full_a_fact_integrity_fault_bits,
        torch.tensor([
            wait_env.FULL_A_FACT_INTEGRITY_R06_SOURCE_INVALID, 0
        ]),
    )
    assert torch.count_nonzero(env._full_a_owner_fact_f32[0, 2]) == 0


@pytest.mark.parametrize("invalid_source", ("mount_sign", "site_xmat"))
def test_host_full_a_step_settles_invalid_r06_without_early_recovery_retire(
    invalid_source,
):
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_IDLE
    env._epoch_phase[0] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_selected[0] = True
    env._epoch_task_valid[0] = True
    env._epoch_launch_succeeded[0] = True

    # Drive the real contact classifier with a finite but invalid pose input.
    # Both corrupt sources must retain one named incident for the ledger.
    if invalid_source == "mount_sign":
        env._full_a_mount_normal_sign[0] = 0
    else:
        env.sim.data.site_xmat[0, 0, 0, 0] = 2.0
    env._ball_gid = 0
    env._con_geom = torch.tensor([[0, 1]], dtype=torch.long)
    env._con_idx = torch.tensor([0], dtype=torch.long)
    env._nacon = torch.tensor([1], dtype=torch.long)
    env._con_world = torch.tensor([0], dtype=torch.long)
    env._geom_class = torch.tensor([0, 1], dtype=torch.int8)

    env._full_a_begin_control_step = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step, env
    )
    env._full_a_prepare_step = lambda: tuple(
        torch.zeros(2, dtype=torch.bool) for _ in range(3)
    )
    env._advance_plant = lambda actions: (
        {}, torch.zeros(2), torch.zeros_like(actions)
    )
    env.sim.forward = lambda: None
    env._latch_post_forward_resolved_table_contacts = lambda _census=None: None
    env._latch_post_forward_table_keepout = lambda: None
    env._full_a_latch_ball_contacts = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts, env
    )
    env._cap_ok = False
    env._state = lambda: {
        "ball_pos": env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()
    }
    env._full_a_update_teacher = lambda: None
    env._fullmdp_termination = lambda _state, _qdes: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_recovery_component_errors = (
        lambda _tracked=None: torch.zeros((2, 13))
    )
    env._full_a_update_cadence_readiness = lambda *_args: None
    env._full_a_publish_r03_fact = lambda _racket_kinematics=None: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_publish_r06_fact = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact, env
    )
    env._full_a_publish_r07_fact = lambda *_args: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    env._fullmdp_reward = (
        lambda _racket_kinematics=None, _tracked=None,
        return_paddle_error=False: (
            torch.zeros(2),
            torch.zeros((2, wait_env.reward_contract.REWARD_TERM_COUNT)),
            *(() if not return_paddle_error else (torch.zeros((2, 4)),)),
        )
    )
    env._full_a_finish_recovery = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery, env
    )
    env._compute_obs = lambda: None
    env.get_observations = lambda: {}

    generation_before = env.reset_generation.clone()
    _obs, _reward, dones, extras = (
        wait_env.FullMdpInitialWaitVecEnv._step_full_a(
            env, torch.zeros((2, 31))
        )
    )

    assert torch.equal(dones, torch.zeros(2, dtype=torch.long))
    assert torch.equal(env.reset_generation, generation_before)
    assert torch.equal(
        extras["full_a_invalid_contact_event"],
        torch.tensor([True, False]),
    )
    assert torch.equal(
        extras["full_a_flight_terminal_event"],
        torch.tensor([True, False]),
    )
    assert extras["full_a_outcome_code"].tolist() == [
        wait_env.FULL_A_OUTCOME_INVALID,
        wait_env.FULL_A_OUTCOME_NONE,
    ]
    assert torch.equal(
        extras["full_a_recovery_failure_event"],
        torch.tensor([False, False]),
    )
    assert torch.equal(
        extras["full_a_shot_retired_event"],
        torch.tensor([False, False]),
    )
    assert torch.equal(
        extras["full_a_fact_integrity_fault_bits"],
        torch.tensor([
            wait_env.FULL_A_FACT_INTEGRITY_R06_SOURCE_INVALID, 0
        ]),
    )
    assert torch.isfinite(extras["reward_terms"]).all()
    assert torch.count_nonzero(extras["reward_terms"][0, 11:13]) == 0
    assert not extras["full_a_selected_reset_event"].any()
    assert extras["full_a_phase_before_reset"].tolist() == [
        wait_env.FULL_A_PHASE_OUTCOME_SETTLED,
        wait_env.FULL_A_PHASE_IDLE,
    ]


def test_host_full_a_r06_distinguishes_eligible_valid_fault_and_no_contact():
    env = _host_full_a_lifecycle_env()
    settled = torch.ones(2, dtype=torch.bool)
    env._full_a_selected_racket_contact[:] = True
    env._full_a_landing_crossing_present[:] = True
    env._full_a_landing_crossing_xy[:] = env._full_a_landing_target_xy
    env._full_a_landing_opponent_bound[:] = True
    env._full_a_landing_on_opponent[:] = True
    env._full_a_net_crossed[:] = True
    env._full_a_net_clear[:] = True
    outcome = torch.tensor(
        [
            wait_env.FULL_A_OUTCOME_LEGAL_LANDING,
            wait_env.FULL_A_OUTCOME_INVALID,
        ]
    )

    present, eligible, common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env, settled, outcome
        )
    )
    reward = wait_env.portable_reward
    assert present.all()
    assert torch.equal(eligible, torch.tensor([True, False]))
    assert torch.equal(common, torch.tensor([True, False]))
    assert torch.equal(
        env._full_a_owner_valid_bits[:, 2],
        torch.tensor(
            [
                reward.R06_PRESENT
                | reward.R06_POLICY_ELIGIBLE
                | reward.R06_SOURCE_VALID,
                reward.R06_PRESENT,
            ]
        ),
    )
    assert torch.equal(env._full_a_owner_fault_bits[:, 2], torch.tensor([0, 1]))
    terms = reward.lifecycle_reward14(
        valid_bits=env._full_a_owner_valid_bits,
        fact_f32=env._full_a_owner_fact_f32,
        owner_fault_bits=env._full_a_owner_fault_bits,
        step_dt=env.step_dt,
    )
    torch.testing.assert_close(
        terms[0, 11:13],
        torch.tensor([20.0, 1.0]) * env.step_dt,
    )
    assert torch.count_nonzero(terms[1, 11:13]) == 0

    retained_bits = env._full_a_owner_valid_bits[:, 2].clone()
    retained_fact = env._full_a_owner_fact_f32[:, 2].clone()

    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    assert torch.equal(env._full_a_owner_valid_bits[:, 2], retained_bits)
    assert torch.equal(env._full_a_owner_fact_f32[:, 2], retained_fact)
    assert not env._full_a_r06_payment_event.any()
    dense_env = _fixed_reward_env()
    dense_env.full_a_mode = True
    dense_env._full_a_owner_valid_bits = env._full_a_owner_valid_bits
    dense_env._full_a_owner_fault_bits = env._full_a_owner_fault_bits
    dense_env._full_a_owner_fact_f32 = env._full_a_owner_fact_f32
    dense_env._full_a_lifecycle_reward_weights = torch.tensor(
        wait_env.portable_reward.LIFECYCLE_WEIGHTS,
        dtype=dense_env._full_a_owner_fact_f32.dtype,
    )
    dense_env._full_a_selected_contact_event = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r06_payment_event = env._full_a_r06_payment_event
    _total, retained_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(dense_env)
    )
    assert torch.count_nonzero(retained_terms[:, 11:13]) == 0

    env._full_a_selected_racket_contact[:] = False
    outcome = torch.tensor(
        [
            wait_env.FULL_A_OUTCOME_LEGAL_LANDING,
            wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED,
        ]
    )
    present, eligible, common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env, settled, outcome
        )
    )
    assert present.all() and eligible.all() and not common.any()
    assert torch.equal(
        env._full_a_owner_valid_bits[:, 2],
        torch.full(
            (2,),
            reward.R06_PRESENT
            | reward.R06_POLICY_ELIGIBLE
            | reward.R06_SOURCE_VALID,
            dtype=torch.long,
        ),
    )
    assert torch.count_nonzero(env._full_a_owner_fault_bits[:, 2]) == 0
    assert torch.equal(
        env._full_a_owner_fact_f32[:, 2, 2], torch.ones(2)
    )
    terms = reward.lifecycle_reward14(
        valid_bits=env._full_a_owner_valid_bits,
        fact_f32=env._full_a_owner_fact_f32,
        owner_fault_bits=env._full_a_owner_fault_bits,
        step_dt=env.step_dt,
    )
    assert torch.count_nonzero(terms[:, 11:13]) == 0


def test_host_full_a_r07_support_uses_10n_normal_force_and_faults_nonfinite():
    env = SimpleNamespace(
        _torch=torch,
        num_envs=2,
        device=torch.device("cpu"),
        qpos_init=torch.zeros(1),
        _con_geom=torch.tensor([[0, 1], [0, 2], [0, 1], [0, 2]]),
        _con_idx=torch.arange(4),
        _nacon=torch.tensor([4]),
        _con_world=torch.tensor([0, 0, 1, 1]),
        _full_a_floor_geom_id=0,
        _full_a_foot_geom_class=torch.tensor([0, 1, 2], dtype=torch.int8),
        _full_a_contact_normal_force=lambda: torch.tensor(
            [9.999, 10.0, 12.0, float("nan")]
        ),
        _body_com_velocities_w=lambda: (
            torch.zeros((2, 14, 3)),
            torch.zeros((2, 14, 3)),
        ),
    )
    support, slip, valid = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_recovery_foot_support(env)
    )
    assert torch.equal(
        support, torch.tensor([[False, True], [True, False]])
    )
    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.count_nonzero(slip) == 0


def test_host_full_a_r07_ready_rejects_joint_outside_isaac_soft_limit():
    lower = torch.full((31,), -1.0)
    upper = torch.full((31,), 1.0)
    joint = torch.zeros((2, 31))
    joint[1, 0] = 0.91
    env = SimpleNamespace(
        _torch=torch,
        jnt_lo=lower,
        jnt_hi=upper,
        _qpos_act=lambda: joint,
    )
    hard_safety = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_recovery_joint_limit_ok(env)
    )
    assert torch.equal(hard_safety, torch.tensor([True, False]))

    eligible, valid, ready, facts = wait_env.portable_outcome.r07_rows(
        torch=torch,
        expected=torch.ones(2, dtype=torch.bool),
        age=torch.full((2,), wait_env.FULL_A_RECOVERY_START_AGE_TICK),
        errors=torch.zeros((2, 13)),
        hard_safety_ok=hard_safety,
        scales=torch.ones((1, 13)),
        ready_tolerances=torch.ones((1, 13)),
        weight=wait_env.portable_outcome.RECOVERY_REWARD_WEIGHT,
    )
    assert eligible.all() and valid.all()
    assert torch.equal(ready, torch.tensor([True, False]))
    assert torch.equal(facts[:, 0], torch.full((2,), 0.7))


def test_host_full_a_r07_nonunit_or_zero_quaternion_faults_row():
    env = _host_full_a_lifecycle_env()
    env._fullmdp_body_ids = torch.arange(14)
    env.sim.data.xpos = torch.zeros((2, 14, 3))
    env.sim.data.xquat = torch.zeros((2, 14, 4))
    env.sim.data.xquat[0, :, 0] = 1.0
    env._full_a_teacher = SimpleNamespace(
        joint_pos=torch.zeros((1, 31)),
        body_pos_w=torch.zeros((1, 14, 3)),
        body_quat_w=torch.nn.functional.pad(
            torch.ones((1, 14, 1)), (0, 3)
        ),
    )
    env._qpos_act = lambda: torch.zeros((2, 31))
    env._qvel_act = lambda: torch.zeros((2, 31))
    env._body_com_velocities_w = lambda: (
        torch.zeros((2, 14, 3)),
        torch.zeros((2, 14, 3)),
    )
    env._full_a_recovery_foot_support = lambda _body_lin_vel: (
        torch.ones((2, 2), dtype=torch.bool),
        torch.zeros((2, 2, 2)),
        torch.ones(2, dtype=torch.bool),
    )
    env._quat_error_sq = wait_env.FullMdpInitialWaitVecEnv._quat_error_sq

    errors = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_recovery_component_errors(env)
    )
    assert torch.isfinite(errors[0]).all()
    assert torch.isnan(errors[1]).all()
    tracked = (
        env.sim.data.xpos[:, env._fullmdp_body_ids],
        env.sim.data.xquat[:, env._fullmdp_body_ids],
        torch.zeros((2, 14, 3)),
        torch.zeros((2, 14, 3)),
    )
    env._body_com_velocities_w = lambda: pytest.fail(
        "explicit recovery tracked-body tuple fell back to a second gather"
    )
    explicit = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_recovery_component_errors(
            env, tracked
        )
    )
    torch.testing.assert_close(
        explicit, errors, rtol=0.0, atol=0.0, equal_nan=True
    )


def test_host_full_a_recovery_uses_the_accepted_task_transform():
    env = _host_full_a_lifecycle_env()
    env._fullmdp_body_ids = torch.arange(14)
    source_pos = torch.zeros((1, 14, 3))
    source_pos[0, :, 0] = torch.linspace(-0.4, 0.5, 14)
    source_pos[0, :, 1] = torch.linspace(0.3, -0.2, 14)
    source_pos[0, :, 2] = torch.linspace(0.7, 1.3, 14)
    source_quat = torch.zeros((1, 14, 4))
    source_quat[..., 0] = 1.0
    env._full_a_teacher = SimpleNamespace(
        joint_pos=torch.zeros((1, 31)),
        body_pos_w=source_pos,
        body_quat_w=source_quat,
    )
    yaw = torch.tensor([0.4, -0.7])
    task_yaw = torch.zeros((2, 4))
    task_yaw[:, 0] = torch.cos(0.5 * yaw)
    task_yaw[:, 3] = torch.sin(0.5 * yaw)
    translation = torch.tensor([[0.8, -0.3, 0.1], [-0.6, 0.9, -0.05]])
    env._full_a_teacher_source_to_task_yaw_wxyz.copy_(task_yaw)
    env._full_a_teacher_source_to_task_translation_scene.copy_(translation)
    expanded_yaw = task_yaw[:, None, :].expand(2, 14, 4)
    actual_pos = _test_quat_apply(
        expanded_yaw, source_pos.expand(2, -1, -1)
    ) + translation[:, None, :]
    actual_quat = _test_quat_mul(
        expanded_yaw, source_quat.expand(2, -1, -1)
    )
    zeros = torch.zeros((2, 14, 3))
    env._qpos_act = lambda: torch.zeros((2, 31))
    env._qvel_act = lambda: torch.zeros((2, 31))
    env._full_a_recovery_foot_support = lambda _body_lin_vel: (
        torch.ones((2, 2), dtype=torch.bool),
        torch.zeros((2, 2, 2)),
        torch.ones(2, dtype=torch.bool),
    )
    errors = wait_env.FullMdpInitialWaitVecEnv._full_a_recovery_component_errors(
        env, (actual_pos, actual_quat, zeros, zeros)
    )
    torch.testing.assert_close(
        errors, torch.zeros_like(errors), rtol=0.0, atol=2.0e-6
    )


def test_host_full_a_r07_membership_does_not_add_a_clock_order_gate():
    env = _host_full_a_lifecycle_env()
    env._epoch_task_valid[:] = torch.tensor([True, False])
    # Deliberately malformed ordering is diagnostic input, not a second
    # membership authority.  The accepted-task bit plus clock3 age alone owns
    # the R07 row.
    env._epoch_clock_ticks[:, 0] = torch.tensor([100, 0])
    env._epoch_clock_ticks[:, 3] = torch.tensor([50, 0])
    env.common_step_counter = 60

    present, eligible = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(
            env,
            torch.zeros((2, 13)),
            torch.ones(2, dtype=torch.bool),
        )
    )

    assert torch.equal(present, torch.tensor([True, False]))
    assert torch.equal(eligible, torch.tensor([True, False]))


def test_host_full_a_invalid_outcome_completes_r07_unpaid_then_retires_failure():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    env._epoch_task_valid.fill_(True)
    env._full_a_outcome_code[:] = torch.tensor(
        [wait_env.FULL_A_OUTCOME_INVALID, wait_env.FULL_A_OUTCOME_OUT]
    )
    env._epoch_clock_ticks[:, 0] = 0
    env._epoch_clock_ticks[:, 3] = 0
    env._full_a_owner_source_step[:, 2] = 0
    env.common_step_counter = wait_env.FULL_A_RECOVERY_START_AGE_TICK
    env._full_a_recovery_component_errors = (
        lambda _tracked=None: torch.zeros((2, 13))
    )

    present, source_valid, _common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env,
            torch.ones(2, dtype=torch.bool),
            env._full_a_outcome_code,
        )
    )
    assert present.all()
    assert torch.equal(source_valid, torch.tensor([False, True]))
    assert torch.equal(env._full_a_owner_fault_bits[:, 2], torch.tensor([1, 0]))

    r07_present, r07_valid = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    )
    assert r07_present.all() and r07_valid.all()
    early = wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
        env,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    assert not early[0].any() and not early[2].any()
    assert env._epoch_phase.eq(wait_env.FULL_A_PHASE_OUTCOME_SETTLED).all()

    # The fixed 68-cell producer tape is physical-outcome independent.  Reward
    # payment is narrower: an invalid R06 row cannot earn the R07 term.
    dense_env = _fixed_reward_env()
    dense_env.full_a_mode = True
    dense_env._epoch_phase.copy_(env._epoch_phase)
    dense_env._full_a_outcome_code.copy_(env._full_a_outcome_code)
    dense_env._full_a_owner_valid_bits = env._full_a_owner_valid_bits
    dense_env._full_a_owner_fault_bits = env._full_a_owner_fault_bits
    dense_env._full_a_owner_fact_f32 = env._full_a_owner_fact_f32
    dense_env._full_a_lifecycle_reward_weights = torch.tensor(
        wait_env.portable_reward.LIFECYCLE_WEIGHTS,
        dtype=dense_env._full_a_owner_fact_f32.dtype,
    )
    dense_env._full_a_selected_contact_event = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r06_payment_event = torch.zeros(2, dtype=torch.bool)
    _total, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(dense_env)
    torch.testing.assert_close(
        terms[:, 13],
        torch.tensor([0.0, 0.7 * dense_env.step_dt], dtype=terms.dtype),
    )

    for age in range(
        wait_env.FULL_A_RECOVERY_START_AGE_TICK + 1,
        wait_env.FULL_A_RECOVERY_END_AGE_TICK + 1,
    ):
        env.common_step_counter = age
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        r07_present, r07_valid = (
            wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
        )
        assert r07_present.all() and r07_valid.all()

    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            env,
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert terminal.all()
    assert not completion_fault.any()
    assert torch.equal(success, torch.tensor([False, True]))
    assert not timeout.any()
    assert torch.equal(failure, torch.tensor([True, False]))
    assert env._epoch_phase.eq(wait_env.FULL_A_PHASE_RETIRED).all()

    env.common_step_counter += 1
    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    r07_present, r07_valid = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    )
    assert not r07_present.any() and not r07_valid.any()


def test_host_full_a_r07_window_success_failure_timeout_and_reward28_conservation():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    env._epoch_task_valid.fill_(True)
    env._epoch_clock_ticks[:, 0] = 0
    env._epoch_clock_ticks[:, 3] = 0
    env._full_a_owner_source_step[:, 2] = 0
    errors = torch.zeros((2, 13))
    errors[1, 0] = float("nan")
    env._full_a_recovery_component_errors = lambda: errors
    env.common_step_counter = wait_env.FULL_A_RECOVERY_START_AGE_TICK - 1
    present, eligible = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    )
    assert not present.any() and not eligible.any()
    assert torch.count_nonzero(env._full_a_owner_valid_bits[:, 3]) == 0
    assert torch.count_nonzero(env._full_a_owner_fact_f32[:, 3]) == 0
    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    env.common_step_counter = wait_env.FULL_A_RECOVERY_START_AGE_TICK

    present, eligible = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    )
    reward = wait_env.portable_reward
    assert present.all()
    assert torch.equal(eligible, torch.tensor([True, False]))
    assert torch.equal(
        env._full_a_owner_valid_bits[:, 3],
        torch.tensor(
            [
                reward.R07_PRESENT | reward.R07_NUMERICALLY_VALID,
                reward.R07_PRESENT,
            ]
        ),
    )
    assert torch.equal(env._full_a_owner_fault_bits[:, 3], torch.tensor([0, 1]))
    assert torch.equal(
        env._full_a_fact_integrity_fault_bits,
        torch.tensor([
            0, wait_env.FULL_A_FACT_INTEGRITY_R07_NONFINITE
        ]),
    )
    assert torch.equal(
        env._full_a_owner_source_step[:, 3], torch.tensor([10, -1])
    )
    lifecycle = reward.lifecycle_reward14(
        valid_bits=env._full_a_owner_valid_bits,
        fact_f32=env._full_a_owner_fact_f32,
        owner_fault_bits=env._full_a_owner_fault_bits,
        step_dt=env.step_dt,
    )
    torch.testing.assert_close(
        lifecycle[:, 13], torch.tensor([0.7 * env.step_dt, 0.0])
    )

    dense_env = _fixed_reward_env()
    dense_env.full_a_mode = True
    dense_env._epoch_phase.fill_(wait_env.FULL_A_PHASE_OUTCOME_SETTLED)
    dense_env._full_a_outcome_code.fill_(wait_env.FULL_A_OUTCOME_OUT)
    dense_env._full_a_owner_valid_bits = env._full_a_owner_valid_bits
    dense_env._full_a_owner_fault_bits = env._full_a_owner_fault_bits
    dense_env._full_a_owner_fact_f32 = env._full_a_owner_fact_f32
    dense_env._full_a_lifecycle_reward_weights = torch.tensor(
        wait_env.portable_reward.LIFECYCLE_WEIGHTS,
        dtype=dense_env._full_a_owner_fact_f32.dtype,
    )
    dense_env._full_a_selected_contact_event = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r06_payment_event = torch.zeros(2, dtype=torch.bool)
    total, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(dense_env)
    torch.testing.assert_close(terms[:, :14], lifecycle.to(terms.dtype))
    torch.testing.assert_close(total, terms.sum(dim=1))

    errors.zero_()
    for age in range(
        wait_env.FULL_A_RECOVERY_START_AGE_TICK + 1,
        wait_env.FULL_A_RECOVERY_END_AGE_TICK + 1,
    ):
        env.common_step_counter = age
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        _, clean_eligible = (
            wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
        )
        assert torch.equal(clean_eligible, torch.tensor([True, False]))
        assert env._full_a_owner_fault_bits[1, 3] == 1
        assert env._full_a_owner_source_step[1, 3] == -1
        assert torch.count_nonzero(env._full_a_owner_fact_f32[1, 3]) == 0
    assert torch.equal(
        env._full_a_recovery_ready_seen, torch.tensor([True, False])
    )
    assert env._full_a_recovery_sticky_fault[1]
    # Sticky reward suppression persists, but the packed ingress counts the
    # producer incident once rather than relabelling every later clean cell.
    assert not env._full_a_fact_integrity_fault_bits.any()
    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            env,
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert torch.equal(terminal, torch.tensor([True, False]))
    assert torch.equal(success, torch.tensor([True, False]))
    assert not failure.any() and not timeout.any()
    assert torch.equal(completion_fault, torch.tensor([False, True]))
    assert torch.equal(
        env._epoch_phase,
        torch.tensor(
            [
                wait_env.FULL_A_PHASE_RETIRED,
                wait_env.FULL_A_PHASE_OUTCOME_SETTLED,
            ]
        ),
    )

    clean = _host_full_a_lifecycle_env()
    clean._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    clean._epoch_task_valid.fill_(True)
    clean._epoch_clock_ticks[:, 0] = 0
    clean._epoch_clock_ticks[:, 3] = 0
    clean._full_a_owner_source_step[:, 2] = 0
    clean_errors = torch.zeros((2, 13))
    clean_errors[1] = 1.0
    clean._full_a_recovery_component_errors = lambda: clean_errors
    for age in range(
        wait_env.FULL_A_RECOVERY_START_AGE_TICK,
        wait_env.FULL_A_RECOVERY_END_AGE_TICK + 1,
    ):
        clean.common_step_counter = age
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(clean)
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(clean)
    assert torch.equal(
        clean._full_a_recovery_expected_count, torch.full((2,), 68)
    )
    assert torch.equal(
        clean._full_a_recovery_eligible_count, torch.full((2,), 68)
    )
    assert not clean._full_a_recovery_sticky_fault.any()
    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            clean,
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert terminal.all()
    assert not completion_fault.any()
    assert torch.equal(success, torch.tensor([True, False]))
    assert not failure.any()
    assert torch.equal(timeout, torch.tensor([False, True]))
    assert torch.equal(
        clean._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_RETIRED),
    )

    safety = _host_full_a_lifecycle_env()
    safety._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    safety._epoch_task_valid.fill_(True)
    safety._epoch_clock_ticks[:, 0] = 0
    safety._epoch_clock_ticks[:, 3] = 0
    safety._full_a_owner_source_step[:, 2] = 0
    safety.common_step_counter = 20
    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            safety,
            torch.tensor([True, False]),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert torch.equal(terminal, torch.tensor([True, False]))
    assert not completion_fault.any()
    assert not success.any() and not timeout.any()
    assert torch.equal(failure, torch.tensor([True, False]))
    assert torch.equal(
        safety._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_OUTCOME_SETTLED),
    )


def test_finish_recovery_healthy_completion_matches_v4_bytes():
    legacy = _host_full_a_lifecycle_env()
    current = _host_full_a_lifecycle_env()
    expected_cells = (
        wait_env.FULL_A_RECOVERY_END_AGE_TICK
        - wait_env.FULL_A_RECOVERY_START_AGE_TICK
        + 1
    )
    for env in (legacy, current):
        env._epoch_phase.fill_(wait_env.FULL_A_PHASE_OUTCOME_SETTLED)
        env._full_a_outcome_code.fill_(wait_env.FULL_A_OUTCOME_OUT)
        env._epoch_clock_ticks[:, 3] = 0
        env._full_a_owner_source_step[:, 2] = 0
        env._full_a_recovery_expected_count.fill_(expected_cells)
        env._full_a_recovery_eligible_count.fill_(expected_cells)
        env._full_a_recovery_last_age.fill_(
            wait_env.FULL_A_RECOVERY_END_AGE_TICK
        )
        env._full_a_recovery_sticky_fault.zero_()
        env._full_a_recovery_ready_seen.copy_(
            torch.tensor([True, False], dtype=torch.bool)
        )
        env.common_step_counter = wait_env.FULL_A_RECOVERY_END_AGE_TICK
    terminated = torch.zeros(2, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)

    expected = _legacy_finish_recovery_reference(
        legacy, terminated, truncated
    )
    actual = wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
        current, terminated, truncated
    )

    def tensor_bytes(value):
        return value.detach().contiguous().numpy().tobytes()

    for expected_value, actual_value in zip(expected, actual[:4]):
        assert expected_value.dtype == actual_value.dtype
        assert tuple(expected_value.shape) == tuple(actual_value.shape)
        assert tensor_bytes(expected_value) == tensor_bytes(actual_value)
    assert tensor_bytes(legacy._epoch_phase) == tensor_bytes(current._epoch_phase)
    assert tensor_bytes(legacy._full_a_recovery_sticky_fault) == tensor_bytes(
        current._full_a_recovery_sticky_fault
    )
    assert not actual[4].any()


def test_finish_recovery_fault_path_has_no_step_host_sync(monkeypatch):
    source = ast.parse(Path(wait_env.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_full_a_finish_recovery"
    )
    assert not any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "bool")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"any", "item", "cpu", "_assert_async"}
            )
        )
        for node in ast.walk(method)
    )

    env = _host_full_a_lifecycle_env()
    expected_cells = (
        wait_env.FULL_A_RECOVERY_END_AGE_TICK
        - wait_env.FULL_A_RECOVERY_START_AGE_TICK
        + 1
    )
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_OUTCOME_SETTLED)
    env._epoch_task_valid.fill_(True)
    env._epoch_clock_ticks[:, 0] = 0
    env._full_a_outcome_code.fill_(wait_env.FULL_A_OUTCOME_OUT)
    env._epoch_clock_ticks[:, 3] = 0
    env._full_a_owner_source_step[:, 2] = 0
    env._full_a_recovery_expected_count.copy_(
        torch.tensor([expected_cells - 1, expected_cells])
    )
    env._full_a_recovery_eligible_count.fill_(expected_cells)
    env._full_a_recovery_last_age.fill_(wait_env.FULL_A_RECOVERY_END_AGE_TICK)
    env._full_a_recovery_sticky_fault.zero_()
    env._full_a_recovery_ready_seen.zero_()
    env.common_step_counter = wait_env.FULL_A_RECOVERY_END_AGE_TICK

    def forbidden_host_read(*_args, **_kwargs):
        raise AssertionError("completion fault synchronized in the control step")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "any", forbidden_host_read)
        patch.setattr(torch.Tensor, "item", forbidden_host_read)
        patch.setattr(torch.Tensor, "cpu", forbidden_host_read)
        patch.setattr(torch.Tensor, "__bool__", forbidden_host_read)
        terminal, success, failure, timeout, completion_fault = (
            wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
                env,
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
            )
        )

    assert torch.equal(terminal, torch.tensor([False, True]))
    assert not success.any() and not failure.any()
    assert torch.equal(timeout, torch.tensor([False, True]))
    assert torch.equal(completion_fault, torch.tensor([True, False]))
    assert torch.equal(
        env._full_a_recovery_sticky_fault, torch.tensor([True, False])
    )


def _completion_fault_step_result(*, compound_invalid_contact=False):
    """Return one named fault row and one healthy timeout from the real step."""

    env = _host_full_a_lifecycle_env()
    expected_cells = (
        wait_env.FULL_A_RECOVERY_END_AGE_TICK
        - wait_env.FULL_A_RECOVERY_START_AGE_TICK
        + 1
    )
    env._epoch_phase.fill_(wait_env.FULL_A_PHASE_OUTCOME_SETTLED)
    env._epoch_task_valid.fill_(True)
    env._epoch_clock_ticks[:, 0] = 0
    env._full_a_outcome_code.fill_(wait_env.FULL_A_OUTCOME_OUT)
    env._epoch_clock_ticks[:, 3] = 0
    env._full_a_owner_source_step[:, 2] = 0
    env._full_a_recovery_expected_count.copy_(
        torch.tensor([expected_cells - 1, expected_cells])
    )
    env._full_a_recovery_eligible_count.fill_(expected_cells)
    env._full_a_recovery_last_age.fill_(wait_env.FULL_A_RECOVERY_END_AGE_TICK)
    env._full_a_recovery_sticky_fault.zero_()
    env._full_a_recovery_ready_seen.zero_()
    env.common_step_counter = wait_env.FULL_A_RECOVERY_END_AGE_TICK
    def begin_control_step(self):
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(self)
        if compound_invalid_contact:
            self._full_a_generic_contact_event[0] = True
            self._full_a_invalid_contact_event[0] = True
            self._full_a_contact_classification_status[0] = 5

    env._full_a_begin_control_step = MethodType(begin_control_step, env)
    env._full_a_prepare_step = lambda: tuple(
        torch.zeros(2, dtype=torch.bool) for _ in range(3)
    )
    env._advance_plant = lambda actions: (
        {}, torch.zeros(2), torch.zeros_like(actions)
    )
    env.sim.forward = lambda: None
    env._latch_post_forward_resolved_table_contacts = lambda _census=None: None
    env._latch_post_forward_table_keepout = lambda: None
    env._full_a_latch_ball_contacts = lambda _census=None: None
    env._cap_ok = False
    env._state = lambda: {}
    env._full_a_update_teacher = lambda: None
    env._fullmdp_termination = lambda _state, _qdes: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_recovery_component_errors = (
        lambda _tracked=None: torch.zeros((2, 13))
    )
    env._full_a_update_cadence_readiness = lambda *_args: None
    env._full_a_publish_physical_fact = lambda: None
    env._full_a_publish_r03_fact = lambda _racket_kinematics=None: (
        torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    )
    env._full_a_settle_outcome = lambda _state: (
        torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long)
    )
    env._full_a_publish_r06_fact = lambda _event, _outcome: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    env._full_a_publish_r07_fact = lambda *_args: (
        torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    )
    env._fullmdp_reward = (
        lambda _racket_kinematics=None, _tracked=None,
        return_paddle_error=False: (
            torch.zeros(2),
            torch.zeros((2, wait_env.reward_contract.REWARD_TERM_COUNT)),
            *(() if not return_paddle_error else (torch.zeros((2, 4)),)),
        )
    )
    env._full_a_finish_recovery = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery, env
    )
    env._compute_obs = lambda: None
    env.get_observations = lambda: {}
    result = wait_env.FullMdpInitialWaitVecEnv._step_full_a(
        env, torch.zeros((2, 31))
    )
    return env, result


def test_completion_fault_is_rejected_once_at_pre_optimizer_boundary(
    monkeypatch,
):
    env, result = _completion_fault_step_result(
        compound_invalid_contact=True
    )
    extras = result[3]
    assert torch.equal(
        extras["full_a_recovery_completion_fault_event"],
        torch.tensor([True, False]),
    )
    assert extras["full_a_invalid_contact_event"][0]
    assert not extras["full_a_recovery_failure_event"].any()
    assert torch.equal(
        extras["full_a_recovery_timeout_event"], torch.tensor([False, True])
    )
    assert torch.equal(
        extras["full_a_shot_retired_event"], torch.tensor([False, True])
    )
    assert not extras["full_a_completed_action_epoch_event"].any()
    assert extras["full_a_phase_before_reset"].tolist() == [
        wait_env.FULL_A_PHASE_OUTCOME_SETTLED,
        wait_env.FULL_A_PHASE_RETIRED,
    ]
    assert not result[2].any()

    fake_contract = SimpleNamespace(
        clone_plant_model_identity=lambda value: dict(value)
    )
    monkeypatch.setattr(update_ledger, "_plant_model_is_exact", lambda _value: True)
    monkeypatch.setattr(
        update_ledger, "_plant_contract_module", lambda: fake_contract
    )
    action_uid = int(env._full_a_catalog.fresh_action.action_uid)
    ledger = update_ledger.FullMdpUpdateLedger(
        torch_module=torch,
        num_envs=2,
        num_steps_per_env=1,
        device=torch.device("cpu"),
        termination_bits=dict(wait_env.FULLMDP_TERMINATION_BITS),
        action_slot=0,
        action_uid=action_uid,
        mount_normal_sign=1,
        family="backhand",
        initial_reset_generation=torch.ones(2, dtype=torch.long),
        run_identity={
            "source_commit": "0" * 40,
            "run_namespace": "completion-fault-boundary-v5",
            "runtime_stack": copy.deepcopy(update_ledger.EXACT_RUNTIME_STACK),
            "plant_model": {},
        },
    )
    storage = {
        name: torch.ones((1, 2, width), dtype=torch.float32)
        for name, width in update_ledger.STORAGE_FLOAT_WIDTHS
    }
    storage_dones = torch.zeros((1, 2, 1), dtype=torch.uint8)
    policy_std = torch.full((2, 31), 0.02)
    real_cpu = torch.Tensor.cpu
    cpu_calls = []

    def counted_cpu(tensor, *args, **kwargs):
        cpu_calls.append(tuple(tensor.shape))
        return real_cpu(tensor, *args, **kwargs)

    optimizer_calls = []
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", counted_cpu)
        ledger.ingest(result)
        assert cpu_calls == []
        with pytest.raises(
            RuntimeError, match="recovery_completion_fault_rows"
        ):
            ledger.prepare(
                0,
                environment_steps=1,
                storage_step=1,
                storage_tensors=storage,
                storage_dones=storage_dones,
                policy_std=policy_std,
            )
            optimizer_calls.append("optimizer")
    assert len(cpu_calls) == 1
    assert optimizer_calls == []


def test_host_full_a_r07_window_rejects_skipped_ages():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_OUTCOME_SETTLED
    env._epoch_task_valid.fill_(True)
    env._epoch_clock_ticks[:, 0] = 0
    env._epoch_clock_ticks[:, 3] = 0
    env._full_a_owner_source_step[:, 2] = 0
    env._full_a_recovery_component_errors = (
        lambda _tracked=None: torch.zeros((2, 13))
    )
    for age in (
        wait_env.FULL_A_RECOVERY_START_AGE_TICK,
        wait_env.FULL_A_RECOVERY_START_AGE_TICK + 2,
    ):
        env.common_step_counter = age
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    assert torch.equal(
        env._full_a_fact_integrity_fault_bits,
        torch.full(
            (2,), wait_env.FULL_A_FACT_INTEGRITY_R07_SEQUENCE,
            dtype=torch.long,
        ),
    )
    assert env._full_a_recovery_sticky_fault.all()
    terminal, success, failure, timeout, completion_fault = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_finish_recovery(
            env,
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    )
    assert not terminal.any()
    assert not success.any() and not failure.any() and not timeout.any()
    assert not completion_fault.any()
    assert env._epoch_phase.eq(wait_env.FULL_A_PHASE_OUTCOME_SETTLED).all()


def test_host_full_a_r07_window_is_anchored_to_deadline_not_early_landing():
    env = _host_full_a_lifecycle_env()
    env._epoch_phase[:] = wait_env.FULL_A_PHASE_LAUNCH_SETTLED
    env._epoch_task_valid.fill_(True)
    env._epoch_clock_ticks[:, 0] = 0
    env._epoch_clock_ticks[:, 3] = 50
    env._epoch_clock_ticks[:, 4] = 51
    env._full_a_selected_racket_contact[:] = True
    env._full_a_landing_crossing_present[:] = True
    env._full_a_landing_crossing_xy[:] = env._full_a_landing_target_xy
    env._full_a_landing_opponent_bound[:] = True
    env._full_a_landing_on_opponent[:] = True
    env._full_a_net_crossed[:] = True
    env._full_a_net_clear[:] = True
    env.common_step_counter = 45
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()

    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )
    assert settled.all()
    assert torch.equal(
        outcome,
        torch.full((2,), wait_env.FULL_A_OUTCOME_LEGAL_LANDING),
    )
    assert torch.equal(env._epoch_clock_ticks[:, 3], torch.full((2,), 50))
    env._full_a_recovery_component_errors = (
        lambda _tracked=None: torch.zeros((2, 13))
    )

    env.common_step_counter = 59
    present, eligible = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    )
    assert not present.any() and not eligible.any()
    env.common_step_counter = 60
    present, eligible = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r07_fact(env)
    )
    assert present.all() and eligible.all()


def test_portable_reward14_uses_independent_r03_error_and_live_contact_bits():
    reward = wait_env.portable_reward
    valid = torch.zeros((2, 4), dtype=torch.long)
    faults = torch.zeros_like(valid)
    facts = torch.zeros((2, 4, 32))
    valid[:, 1] = reward.R03_PRESENT | reward.R03_PHYSICALLY_VALID
    facts[:, 1, 6:9] = torch.tensor([0.0, 1.0, 0.0])
    facts[:, 1, 21:24] = torch.tensor([0.0, 1.0, 0.0])
    facts[:, 1, 9:12] = torch.tensor([0.0, 0.0, 1.0])
    facts[:, 1, 0:3] = facts[:, 1, 9:12]
    facts[:, 1, 15:18] = facts[:, 1, 9:12]
    # Row one changes only the achieved FK; the target writer is untouched.
    facts[1, 1, 15] += 0.2
    valid[0, 0] = reward.PHYSICAL_PRESENT | reward.PHYSICAL_SELECTED_CONTACT
    valid[1, 0] = reward.PHYSICAL_PRESENT

    terms = reward.lifecycle_reward14(
        valid_bits=valid,
        fact_f32=facts,
        owner_fault_bits=faults,
        step_dt=0.02,
    )
    assert terms.shape == (2, 14)
    assert torch.isfinite(terms).all()
    assert torch.isclose(terms[0, 0], torch.tensor(0.02))
    assert torch.isclose(terms[1, 0], torch.exp(torch.tensor(-1.0)) * 0.02)
    assert torch.isclose(terms[0, 10], torch.tensor(0.02))
    assert terms[1, 10] == 0.0
    assert torch.count_nonzero(terms[:, 11:14]) == 0


def test_cached_reward14_matches_fallback_bitwise_for_all_terms_and_fact_states():
    reward = wait_env.portable_reward
    valid = torch.zeros((6, 4), dtype=torch.long)
    faults = torch.zeros_like(valid)
    facts = torch.zeros((6, 4, 32), dtype=torch.float64)
    valid[:, 0] = reward.PHYSICAL_PRESENT | reward.PHYSICAL_SELECTED_CONTACT
    valid[:, 1] = reward.R03_PRESENT | reward.R03_PHYSICALLY_VALID
    valid[:, 2] = (
        reward.R06_PRESENT | reward.R06_POLICY_ELIGIBLE | reward.R06_SOURCE_VALID
    )
    valid[:, 3] = reward.R07_PRESENT | reward.R07_NUMERICALLY_VALID
    facts[:, 1, 6:9] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    facts[:, 1, 21:24] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    facts[:, 2, 0:3] = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)
    facts[:, 3, 0] = 0.7
    facts[:, 3, 2:4] = 1.0

    # Invalid-bit, owner-fault, and nonfinite R03/R06/R07 rows share one tape.
    valid[1].zero_()
    faults[2].fill_(1)
    facts[3, 1, 15] = float("nan")
    facts[4, 2, 0] = float("inf")
    facts[4, 2, 1] = float("nan")
    facts[5, 3, 0] = float("nan")

    fallback = reward.lifecycle_reward14(
        valid_bits=valid, fact_f32=facts, owner_fault_bits=faults, step_dt=0.02,
    )
    cached = reward.lifecycle_reward14(
        valid_bits=valid,
        fact_f32=facts,
        owner_fault_bits=faults,
        step_dt=0.02,
        weights=torch.tensor(reward.LIFECYCLE_WEIGHTS, dtype=torch.float64),
    )
    assert torch.equal(cached, fallback)
    assert torch.isfinite(cached).all()
    assert torch.count_nonzero(cached[0]) == 14
    assert torch.count_nonzero(cached[1]) == 0
    assert torch.count_nonzero(cached[2]) == 0


def test_reward28_cached_lifecycle_weights_avoid_step_host_ops(monkeypatch):
    env = _fixed_reward_env()
    env.full_a_mode = True
    env._full_a_owner_valid_bits = torch.zeros((2, 4), dtype=torch.long)
    env._full_a_owner_valid_bits[:, 0] = (
        wait_env.portable_reward.PHYSICAL_PRESENT
        | wait_env.portable_reward.PHYSICAL_SELECTED_CONTACT
    )
    env._full_a_owner_fault_bits = torch.zeros_like(
        env._full_a_owner_valid_bits
    )
    env._full_a_owner_fact_f32 = torch.zeros((2, 4, 32), dtype=torch.float64)
    env._full_a_selected_contact_event = torch.ones(2, dtype=torch.bool)
    env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    env._full_a_r06_payment_event = torch.zeros(2, dtype=torch.bool)

    expected_reward, expected_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    )
    cached_pointer = env._full_a_lifecycle_reward_weights.data_ptr()

    def forbidden_host_or_constructor(*_args, **_kwargs):
        raise AssertionError("Reward28 hot path performed a host read or tensor construction")

    with monkeypatch.context() as patch:
        patch.setattr(wait_env.portable_reward.torch, "tensor", forbidden_host_or_constructor)
        patch.setattr(torch.Tensor, "new_tensor", forbidden_host_or_constructor)
        patch.setattr(torch.Tensor, "__bool__", forbidden_host_or_constructor)
        patch.setattr(torch.Tensor, "item", forbidden_host_or_constructor)
        patch.setattr(torch.Tensor, "cpu", forbidden_host_or_constructor)
        actual_reward, actual_terms = (
            wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
        )

    assert env._full_a_lifecycle_reward_weights.data_ptr() == cached_pointer
    assert torch.equal(actual_terms, expected_terms)
    assert torch.equal(actual_reward, expected_reward)
    assert torch.equal(
        actual_terms[:, 10], torch.full((2,), env.step_dt, dtype=torch.float64)
    )


def test_reward28_explicitly_passes_the_construction_cached_weights():
    source = ast.parse(Path(wait_env.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fullmdp_reward"
    )
    call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lifecycle_reward14"
    )
    keyword = next(value for value in call.keywords if value.arg == "weights")
    assert isinstance(keyword.value, ast.Attribute)
    assert keyword.value.attr == "_full_a_lifecycle_reward_weights"


def test_reward28_pays_r03_once_while_retaining_the_sticky_fact():
    env = _fixed_reward_env()
    env.full_a_mode = True
    env._full_a_owner_valid_bits = torch.zeros((2, 4), dtype=torch.long)
    env._full_a_owner_fault_bits = torch.zeros_like(
        env._full_a_owner_valid_bits
    )
    env._full_a_owner_fact_f32 = torch.zeros((2, 4, 32), dtype=torch.float64)
    r03_bits = (
        wait_env.portable_reward.R03_PRESENT
        | wait_env.portable_reward.R03_PHYSICALLY_VALID
    )
    env._full_a_owner_valid_bits[:, 1] = r03_bits
    r03 = env._full_a_owner_fact_f32[:, 1]
    r03[:, 6:9] = torch.tensor([0.0, 1.0, 0.0])
    r03[:, 21:24] = torch.tensor([0.0, 1.0, 0.0])
    env._full_a_selected_contact_event = torch.zeros(2, dtype=torch.bool)
    env._full_a_r06_payment_event = torch.zeros(2, dtype=torch.bool)
    env._full_a_r03_present = torch.ones(2, dtype=torch.bool)

    _reward_n, terms_n = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    )
    sticky_bits = env._full_a_owner_valid_bits[:, 1].clone()
    sticky_fact = env._full_a_owner_fact_f32[:, 1].clone()
    assert torch.count_nonzero(terms_n[:, :10]) > 0

    env._full_a_r03_present.zero_()
    _reward_n1, terms_n1 = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    )
    assert torch.count_nonzero(terms_n1[:, :10]) == 0
    assert torch.equal(env._full_a_owner_valid_bits[:, 1], sticky_bits)
    assert torch.equal(env._full_a_owner_fact_f32[:, 1], sticky_fact)


def test_reward28_nonfinite_is_left_for_pre_optimizer_ledger_without_step_sync():
    source = ast.parse(Path(wait_env.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fullmdp_reward"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
        for node in ast.walk(method)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_assert_async"
        for node in ast.walk(method)
    )

    env = _fixed_reward_env()
    env._teacher_body_pos[0, env._fullmdp_anchor_index, 0] = float("nan")
    reward, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(env)
    assert not torch.isfinite(reward[0])
    assert not torch.isfinite(terms[0]).all()


def test_host_live_generic_contact_classifies_selected_and_opposite_once():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._con_geom = torch.tensor([[2, 1], [2, 1]], dtype=torch.long)
    env._con_idx = torch.arange(2)
    env._nacon = torch.tensor([2], dtype=torch.long)
    env._con_world = torch.tensor([0, 1], dtype=torch.long)
    env._ball_gid = 2
    env._geom_class = torch.tensor([0, 1, 0], dtype=torch.int8)
    center_x, center_z = wait_env.racket_contact_geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M
    env.sim.data.qpos[0, env.b_q : env.b_q + 3] = (
        env.sim.data.site_xpos[0, 0]
        + torch.tensor([center_x, 0.020, center_z])
    )
    env.sim.data.qpos[1, env.b_q : env.b_q + 3] = (
        env.sim.data.site_xpos[1, 0]
        + torch.tensor([center_x, -0.033208, center_z])
    )
    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_physical_fact(env)

    assert torch.equal(env._full_a_generic_contact_event, torch.tensor([True, True]))
    assert torch.equal(env._full_a_selected_contact_event, torch.tensor([True, False]))
    assert torch.equal(env._full_a_opposite_contact_event, torch.tensor([False, True]))
    assert torch.equal(
        env._full_a_contact_classification_status,
        torch.tensor(
            [
                wait_env.racket_contact_geometry.OBSERVED_RUBBER_STATUS_SELECTED,
                wait_env.racket_contact_geometry.OBSERVED_RUBBER_STATUS_OPPOSITE,
            ],
            dtype=torch.int8,
        ),
    )
    assert torch.equal(
        env._full_a_owner_valid_bits[:, 0],
        torch.tensor(
            [
                wait_env.portable_reward.PHYSICAL_PRESENT
                | wait_env.portable_reward.PHYSICAL_SELECTED_CONTACT,
                wait_env.portable_reward.PHYSICAL_PRESENT,
            ]
        ),
    )
    terms = wait_env.portable_reward.lifecycle_reward14(
        valid_bits=env._full_a_owner_valid_bits,
        fact_f32=env._full_a_owner_fact_f32,
        owner_fault_bits=env._full_a_owner_fault_bits,
        step_dt=env.step_dt,
    )
    assert terms[0, 10] == env.step_dt
    assert terms[1, 10] == 0.0

    source_step = env._full_a_physical_source_step.clone()
    contact_fact = env._full_a_physical_fact_f32[:, 3:9].clone()
    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_physical_fact(env)
    assert torch.equal(env._full_a_physical_source_step, source_step)
    assert torch.equal(env._full_a_physical_fact_f32[:, 3:9], contact_fact)
    assert torch.bitwise_and(
        env._full_a_owner_valid_bits[:, 0],
        wait_env.portable_reward.PHYSICAL_SELECTED_CONTACT,
    ).ne(0)[0]

    dense_env = _fixed_reward_env()
    dense_env.full_a_mode = True
    dense_env._full_a_owner_valid_bits = env._full_a_owner_valid_bits
    dense_env._full_a_owner_fault_bits = env._full_a_owner_fault_bits
    dense_env._full_a_owner_fact_f32 = env._full_a_owner_fact_f32
    dense_env._full_a_lifecycle_reward_weights = torch.tensor(
        wait_env.portable_reward.LIFECYCLE_WEIGHTS,
        dtype=dense_env._full_a_owner_fact_f32.dtype,
    )
    dense_env._full_a_selected_contact_event = env._full_a_selected_contact_event
    dense_env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r06_payment_event = env._full_a_r06_payment_event
    _total, repeated_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(dense_env)
    )
    assert torch.count_nonzero(repeated_terms[:, 10]) == 0


def test_host_live_generic_contact_uses_same_forward_ball_pose_at_safe_edge():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._con_geom = torch.tensor([[2, 1]], dtype=torch.long)
    env._con_idx = torch.tensor([0])
    env._nacon = torch.tensor([1], dtype=torch.long)
    env._con_world = torch.tensor([0], dtype=torch.long)
    env._ball_gid = 2
    env._geom_class = torch.tensor([0, 1, 0], dtype=torch.int8)

    same_forward_radius_m = 0.0435
    post_integrated_radius_m = 0.0455
    safe_radius_m = (
        wait_env.racket_contact_geometry.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M
    )
    assert same_forward_radius_m < safe_radius_m < post_integrated_radius_m
    center_x, center_z = (
        wait_env.racket_contact_geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M
    )
    site = env.sim.data.site_xpos[0, env.racket_sid]
    same_forward_center = site + torch.tensor(
        [center_x + same_forward_radius_m, 0.020, center_z]
    )
    post_integrated_center = site + torch.tensor(
        [center_x + post_integrated_radius_m, 0.020, center_z]
    )
    # Break the host fixture's normal qpos/xpos view alias to reproduce
    # MJWarp's post-step state: qpos has integrated ahead while derived body
    # and site poses still describe the contact census' forward snapshot.
    env.sim.data.xpos = env.sim.data.xpos.clone()
    env.sim.data.xpos[0, env._fullmdp_ball_body_id] = same_forward_center
    env.sim.data.qpos[0, env.b_q : env.b_q + 3] = post_integrated_center

    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts(env)

    assert bool(env._full_a_selected_contact_event[0])
    assert not bool(env._full_a_edge_contact_event[0])
    assert env._full_a_contact_classification_status[0] == (
        wait_env.racket_contact_geometry.OBSERVED_RUBBER_STATUS_SELECTED
    )
    torch.testing.assert_close(
        env._full_a_contact_center[0], same_forward_center, rtol=0.0, atol=0.0
    )


def test_host_full_a_second_net_crossing_cannot_upgrade_first_low_crossing():
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._full_a_selected_racket_contact[:] = True
    env._full_a_previous_ball_center_valid[:] = True
    env._full_a_previous_ball_center[:] = torch.tensor(
        [[1.70, 0.0, 1.00], [1.70, 0.0, 1.00]]
    )
    env._con_geom = torch.tensor([[0, 0]], dtype=torch.long)
    env._con_idx = torch.tensor([0])
    env._nacon = torch.tensor([0])
    env._con_world = torch.tensor([0])
    env._ball_gid = 2
    env._geom_class = torch.zeros(3, dtype=torch.int8)

    def observe_at(position):
        env.sim.data.qpos[:, env.b_q : env.b_q + 3] = torch.tensor(position)
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts(env)

    observe_at([[2.00, 0.0, 0.88], [2.00, 0.0, 0.88]])
    assert env._full_a_net_crossed.all() and not env._full_a_net_clear.any()
    observe_at([[1.70, 0.0, 1.10], [1.70, 0.0, 1.10]])
    observe_at([[2.00, 0.0, 1.10], [2.00, 0.0, 1.10]])
    assert env._full_a_net_crossed.all()
    assert not env._full_a_net_clear.any()

    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts(env)
    assert not bool(env._full_a_generic_contact_event.any())
    assert not bool(env._full_a_selected_contact_event.any())


def test_mu_ball_filtered_net_low_crossing_can_land_but_cannot_score():
    import mujoco
    import a3_court_env as court_env

    # Compile the production court builder with its normal explicit pairs.  The
    # net remains visible/robot-collidable geometry, but neither masks nor an
    # explicit pair may let its unmeasured response perturb the ball trajectory.
    spec = mujoco.MjSpec()
    for name in (
        court_env.FLOOR_GEOM,
        *court_env.RACKET_GEOMS,
    ):
        geom = spec.worldbody.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.05, 0.05, 0.05]
        geom.pos = [-10.0, 0.0, 0.0]
    court_env.make_court_spec_fn(
        (2.0, -0.76, 0.70), "elliptic", add_pairs=True
    )(spec)
    model = spec.compile()
    net_gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "court_net"
    )
    ball_gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, court_env.BALL_GEOM
    )
    assert net_gid >= 0 and ball_gid >= 0
    mask_contact = bool(
        int(model.geom_contype[ball_gid])
        & int(model.geom_conaffinity[net_gid])
    ) or bool(
        int(model.geom_contype[net_gid])
        & int(model.geom_conaffinity[ball_gid])
    )
    assert not mask_contact
    floor_gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, court_env.FLOOR_GEOM
    )
    assert bool(
        int(model.geom_contype[floor_gid])
        & int(model.geom_conaffinity[net_gid])
    ) or bool(
        int(model.geom_contype[net_gid])
        & int(model.geom_conaffinity[floor_gid])
    )
    assert not any(
        {
            int(model.pair_geom1[pair_id]),
            int(model.pair_geom2[pair_id]),
        }
        == {ball_gid, net_gid}
        for pair_id in range(model.npair)
    )

    # Fixed tape: both rows cross the net and descend through the exact same
    # target landing.  Row 0 clears at z=0.96 m; row 1 intersects the net band
    # at z=0.93 m (< the 0.94 m centre-clearance plane).  A physical collider
    # would alter row 1 before landing; under the single adjudication contract
    # it may continue, but its low first crossing must remain OUT and earn zero.
    env = _host_full_a_lifecycle_env()
    _prepare_and_settle(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._full_a_selected_racket_contact[:] = True
    env._full_a_previous_ball_center_valid[:] = True
    env._full_a_previous_ball_center[:] = torch.tensor(
        [[1.70, 0.0, 0.96], [1.70, 0.0, 0.93]]
    )
    empty_census = SimpleNamespace(
        ball_racket_by_world=torch.zeros(2),
        ball_table_by_world=torch.zeros(2),
    )

    def observe_at(points):
        env.sim.data.xpos[:, env._fullmdp_ball_body_id] = torch.tensor(points)
        wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
        wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts(
            env, empty_census
        )

    observe_at([[2.00, 0.0, 0.96], [2.00, 0.0, 0.93]])
    assert env._full_a_net_crossed.all()
    assert torch.equal(env._full_a_net_clear, torch.tensor([True, False]))
    observe_at([[2.40, 0.0, 0.90], [2.40, 0.0, 0.90]])
    observe_at([[2.65, 0.0, 0.70], [2.65, 0.0, 0.70]])
    assert env._full_a_landing_on_opponent.all()
    torch.testing.assert_close(
        env._full_a_landing_crossing_xy,
        env._full_a_landing_target_xy.expand(2, 2),
        rtol=0.0,
        atol=1.0e-6,
    )

    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": env.sim.data.xpos[:, env._fullmdp_ball_body_id]}
    )
    assert settled.all()
    assert outcome.tolist() == [
        wait_env.FULL_A_OUTCOME_LEGAL_LANDING,
        wait_env.FULL_A_OUTCOME_OUT,
    ]
    present, eligible, common = (
        wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r06_fact(
            env, settled, outcome
        )
    )
    assert present.all() and eligible.all()
    assert torch.equal(common, torch.tensor([True, False]))

    dense_env = _fixed_reward_env()
    dense_env.full_a_mode = True
    dense_env._epoch_phase.copy_(env._epoch_phase)
    dense_env._full_a_outcome_code.copy_(env._full_a_outcome_code)
    dense_env._full_a_owner_valid_bits = env._full_a_owner_valid_bits
    dense_env._full_a_owner_fault_bits = env._full_a_owner_fault_bits
    dense_env._full_a_owner_fact_f32 = env._full_a_owner_fact_f32
    dense_env._full_a_lifecycle_reward_weights = torch.tensor(
        wait_env.portable_reward.LIFECYCLE_WEIGHTS,
        dtype=dense_env._full_a_owner_fact_f32.dtype,
    )
    dense_env._full_a_selected_contact_event = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r03_present = torch.zeros(2, dtype=torch.bool)
    dense_env._full_a_r06_payment_event = env._full_a_r06_payment_event
    _total, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward(dense_env)
    torch.testing.assert_close(
        terms[0, 11:13],
        torch.tensor([20.0, 1.0], dtype=terms.dtype) * dense_env.step_dt,
    )
    assert torch.count_nonzero(terms[1, 11:13]) == 0


def _assert_step_surface(result, *, num_envs: int):
    observations, reward, dones, extras = result
    assert tuple(observations["policy"].shape) == (num_envs, P.ACTOR_WIDTH_V1)
    assert tuple(observations["critic"].shape) == (num_envs, P.CRITIC_WIDTH_V1)
    assert tuple(reward.shape) == (num_envs,)
    assert tuple(dones.shape) == (num_envs,)
    assert dones.dtype == torch.long
    assert set(extras) == {
        "time_outs",
        "termination_bits",
        "backend_resolved_table_contact",
        "reward_terms",
        "reset_generation",
    }
    assert tuple(extras["time_outs"].shape) == (num_envs,)
    assert extras["time_outs"].dtype == torch.bool
    assert tuple(extras["termination_bits"].shape) == (num_envs,)
    assert extras["termination_bits"].dtype == torch.int64
    assert tuple(extras["backend_resolved_table_contact"].shape) == (num_envs,)
    assert extras["backend_resolved_table_contact"].dtype == torch.bool
    assert tuple(extras["reward_terms"].shape) == (
        num_envs,
        wait_env.reward_contract.REWARD_TERM_COUNT,
    )
    assert tuple(extras["reset_generation"].shape) == (num_envs,)
    assert extras["reset_generation"].dtype == torch.int64
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(extras["reward_terms"]).all()
    assert torch.allclose(reward, extras["reward_terms"].sum(dim=1))
    return observations, reward, dones, extras


def _assert_full_a_step_surface(result, *, num_envs: int):
    observations, reward, dones, extras = result
    assert tuple(observations["policy"].shape) == (num_envs, P.ACTOR_WIDTH_V3)
    assert tuple(observations["critic"].shape) == (num_envs, P.CRITIC_WIDTH_V3)
    assert tuple(reward.shape) == (num_envs,)
    assert tuple(dones.shape) == (num_envs,)
    assert dones.dtype == torch.long
    assert set(extras) == {
        "time_outs",
        "termination_bits",
        "backend_resolved_table_contact",
        "reward_terms",
        "full_a_paddle_prior_playback",
        "reset_generation",
        "full_a_phase_before_reset",
        "full_a_outcome_code",
        "full_a_fact_integrity_fault_bits",
        "full_a_racket_contact",
        "full_a_ball_table_contact",
        "full_a_physical_current_center",
        "full_a_scheduled_due_event",
        "full_a_due_terminal_overlap_event",
        "full_a_reveal_event",
        "full_a_reveal_due_event",
        "full_a_reveal_deferred_event",
        "full_a_launch_event",
        "full_a_missed_launch_event",
        "full_a_flight_terminal_event",
        "full_a_shot_retired_event",
        "full_a_completed_action_epoch_event",
        "full_a_selected_reset_event",
        "full_a_racket_contact_event",
        "full_a_action_slot",
        "full_a_action_uid",
        "full_a_mount_normal_sign",
        "full_a_contact_classification_status",
        "full_a_selected_contact_event",
        "full_a_opposite_contact_event",
        "full_a_edge_contact_event",
        "full_a_between_contact_event",
        "full_a_invalid_contact_event",
        "full_a_actual_hard_edge_event",
        "full_a_qdes_guard_intervention_event",
        "full_a_r03_present_event",
        "full_a_r03_physically_valid_event",
        "full_a_landing_crossing_event",
        "full_a_landing_on_opponent",
        "full_a_landing_opponent_bound",
        "full_a_r06_present_event",
        "full_a_r06_eligible_event",
        "full_a_r06_common_event",
        "full_a_r07_present_event",
        "full_a_r07_eligible_event",
        "full_a_recovery_success_event",
        "full_a_recovery_failure_event",
        "full_a_recovery_timeout_event",
        "full_a_recovery_completion_fault_event",
    }
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(extras["reward_terms"]).all()
    assert torch.equal(extras["full_a_action_slot"], torch.zeros(num_envs, dtype=torch.long, device=extras["full_a_action_slot"].device))
    assert torch.equal(extras["full_a_action_uid"], torch.full((num_envs,), 5527597793770800, dtype=torch.long, device=extras["full_a_action_uid"].device))
    assert torch.equal(extras["full_a_mount_normal_sign"], torch.ones(num_envs, dtype=torch.int8, device=extras["full_a_mount_normal_sign"].device))
    assert tuple(extras["full_a_fact_integrity_fault_bits"].shape) == (num_envs,)
    assert tuple(extras["full_a_paddle_prior_playback"].shape) == (num_envs,)
    assert extras["full_a_paddle_prior_playback"].dtype == torch.bool
    assert extras["full_a_fact_integrity_fault_bits"].dtype == torch.int64
    assert torch.allclose(reward, extras["reward_terms"].sum(dim=1))
    return observations, reward, dones, extras


def _gpu_env(*, num_envs: int, full_a_mode: bool = False):
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("mjlab")
    pytest.importorskip("tensordict")
    task = wait_env.TaskCfg(
        episode_length_s=30.0 if full_a_mode else 3.0,
        action_scale_mode="vendor",
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    ready_path = os.environ.get("ACTIONBALL_READY_POSE")
    return wait_env.FullMdpInitialWaitVecEnv(
        sim_cfg=wait_env.SimCfg(nworld=num_envs),
        task_cfg=task,
        device=os.environ.get("ACTIONBALL_MUJOCO_DEVICE", "cuda:0"),
        ready_pose_path=Path(ready_path) if ready_path else None,
        full_a_mode=full_a_mode,
    )


def _jump_real_full_a_clock_to_transition_before_first_due(env) -> None:
    """Skip only the expensive balance-prefix clock in direct GPU tests.

    Production runs every transition from reset.  These tests own live MuJoCo
    callpoints at the real tick-295 boundary, so they advance both authoritative
    control counters together and never publish the synthetic prefix as
    business evidence.
    """

    first_due = int(env._full_a_cadence.first_reveal_tick)
    assert first_due == 295
    assert int(env.common_step_counter) == 0
    assert bool(env.episode_length_buf.eq(0).all())
    env.common_step_counter = first_due - 2
    env.episode_length_buf.fill_(first_due - 2)


def _v3_layout_slice(layout, name, *, offset=0):
    """Resolve one shared ABI field for direct GPU value checks."""

    start = int(offset)
    for field, width in layout:
        end = start + int(width)
        if field == name:
            return slice(start, end)
        start = end
    raise AssertionError(name)


def _assert_real_full_a_returned_observation(
    env,
    observations,
    *,
    expected_action,
    expected_task_visible,
    expected_live_ball,
):
    """Check one real returned V3 boundary against its live numeric owners."""

    policy = observations["policy"]
    critic = observations["critic"]
    immediate = env.get_observations()
    assert torch.equal(policy, immediate["policy"])
    assert torch.equal(critic, immediate["critic"])
    assert torch.equal(critic[:, : P.ACTOR_WIDTH_V3], policy)

    def actor(name):
        return policy[:, _v3_layout_slice(P.ACTOR_LAYOUT_V3, name)]

    def privileged(name):
        return critic[
            :,
            _v3_layout_slice(
                P.CRITIC_EXTENSION_LAYOUT_V3,
                name,
                offset=P.ACTOR_WIDTH_V3,
            ),
        ]

    torch.testing.assert_close(
        actor("last_action"), expected_action, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actor("teacher_joint_pos_rel"),
        env._full_a_teacher_joint_pos - env.action_offset.unsqueeze(0),
        rtol=0.0,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        actor("teacher_joint_vel"),
        env._full_a_teacher_joint_vel * 0.05,
        rtol=0.0,
        atol=2.0e-6,
    )

    anchor = env._fullmdp_anchor_index
    live_anchor_pos = env.sim.data.xpos[:, env._fullmdp_anchor_body_id]
    live_anchor_quat = env.sim.data.xquat[:, env._fullmdp_anchor_body_id]
    anchor_pos, anchor_ori = P.relative_pose_6d(
        live_anchor_pos,
        live_anchor_quat,
        env._teacher_body_pos[:, anchor],
        env._teacher_body_quat[:, anchor],
    )
    torch.testing.assert_close(
        actor("motion_anchor_pos_b"),
        anchor_pos * (10.0 / 3.0),
        rtol=0.0,
        atol=3.0e-6,
    )
    torch.testing.assert_close(
        actor("motion_anchor_ori_b6"),
        anchor_ori,
        rtol=0.0,
        atol=3.0e-6,
    )
    expected_motion_phase = torch.nn.functional.one_hot(
        env._full_a_motion_phase_code, num_classes=5
    ).to(dtype=policy.dtype)
    assert torch.equal(actor("motion_phase_one_hot"), expected_motion_phase)

    task_visible = env._epoch_task_valid & env._full_a_motion_phase_code.lt(
        wait_env.RECOVER_HIDDEN_PHASE_INDEX
    )
    assert bool(task_visible[0]) is bool(expected_task_visible)
    assert torch.equal(
        actor("task_valid")[:, 0], task_visible.to(dtype=policy.dtype)
    )

    st = env._state()
    heading_xy = P.heading_xy_from_quat_wxyz(st["base_quat"])

    def heading(value):
        return P.rotate_world_to_heading_xy(heading_xy, value)

    racket_pos, racket_vel, racket_raw_normal, racket_long_axis = (
        env._full_a_racket_kinematics()
    )
    racket_face_sign = torch.where(
        env._epoch_task_valid,
        env._fullmdp_mount_normal_sign,
        torch.ones_like(env._fullmdp_mount_normal_sign),
    )
    racket_signed_normal = racket_raw_normal * racket_face_sign[:, None]
    expected_motion_reference = torch.cat(
        (
            heading(
                env._aligned_teacher_racket_site_pos_w
                - env.env.scene.env_origins
                - racket_pos
            )
            * 5.0,
            heading(
                env._aligned_teacher_racket_site_lin_vel_w - racket_vel
            ),
            heading(
                env._aligned_teacher_racket_signed_normal_w
                - racket_signed_normal
            )
            * 2.0,
            heading(
                env._aligned_teacher_racket_long_axis_w - racket_long_axis
            )
            * 2.0,
        ),
        dim=1,
    )
    actual_motion_reference = torch.cat(
        (
            actor("motion_racket_pos_error_heading"),
            actor("motion_racket_vel_error_heading"),
            actor("motion_racket_signed_normal_error_heading"),
            actor("motion_racket_long_axis_error_heading"),
        ),
        dim=1,
    )
    torch.testing.assert_close(
        actual_motion_reference,
        expected_motion_reference,
        rtol=0.0,
        atol=3.0e-6,
    )
    task = env._epoch_task_f32
    root_scene = st["base_pos"] - env.env.scene.env_origins
    base_goal = torch.cat(
        (
            task[:, 24:26] - root_scene[:, :2],
            torch.zeros_like(root_scene[:, :1]),
        ),
        dim=1,
    )
    task_mask = task_visible[:, None]

    def visible_heading(value):
        rotated = heading(value)
        return torch.where(task_mask, rotated, torch.zeros_like(rotated))

    expected_task_reference = torch.cat(
        (
            visible_heading(task[:, 5:8] - racket_pos) * 5.0,
            visible_heading(task[:, 8:11] - racket_vel),
            visible_heading(task[:, 11:14] - racket_raw_normal) * 2.0,
            visible_heading(base_goal)[:, :2] * 5.0,
        ),
        dim=1,
    )
    actual_task_reference = torch.cat(
        (
            actor("racket_target_pos_error_heading"),
            actor("racket_target_vel_error_heading"),
            actor("racket_target_normal_error_heading"),
            actor("base_goal_error_heading_xy"),
        ),
        dim=1,
    )
    torch.testing.assert_close(
        actual_task_reference,
        expected_task_reference,
        rtol=0.0,
        atol=3.0e-6,
    )

    live_ball = (
        env._epoch_task_valid
        & env._epoch_launch_succeeded
        & env._epoch_phase.eq(wait_env.FULL_A_PHASE_LAUNCH_SETTLED)
    )
    assert bool(live_ball[0]) is bool(expected_live_ball)
    if expected_live_ball:
        ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3]
        ball_quat = env.sim.data.qpos[:, env.b_q + 3 : env.b_q + 7]
        ball_lin = env.sim.data.qvel[:, env.b_v : env.b_v + 3]
        ball_ang = P.quat_rotate_wxyz(
            ball_quat, env.sim.data.qvel[:, env.b_v + 3 : env.b_v + 6]
        )
        torch.testing.assert_close(
            privileged("live_ball_center_rel_root_heading"),
            heading(ball_pos - st["base_pos"]),
            rtol=0.0,
            atol=3.0e-6,
        )
        torch.testing.assert_close(
            privileged("live_ball_lin_vel_heading"),
            heading(ball_lin) * 0.1,
            rtol=0.0,
            atol=3.0e-6,
        )
        torch.testing.assert_close(
            privileged("live_ball_ang_vel_heading"),
            heading(ball_ang) / 60.0,
            rtol=0.0,
            atol=3.0e-6,
        )
        expected_latches = torch.stack(
            (
                env._full_a_selected_racket_contact,
                env._full_a_net_crossed,
                env._full_a_net_clear,
            ),
            dim=1,
        ).to(dtype=critic.dtype)
        actual_latches = torch.cat(
            (
                privileged("selected_rubber_contact_latched"),
                privileged("net_crossed_latched"),
                privileged("net_clear_latched"),
            ),
            dim=1,
        )
        assert torch.equal(actual_latches, expected_latches)
    else:
        assert torch.count_nonzero(critic[:, 204:216]) == 0

    recovery_observable = (
        env._epoch_task_valid
        & env._epoch_phase.eq(wait_env.FULL_A_PHASE_OUTCOME_SETTLED)
        & env._full_a_outcome_code.ne(wait_env.FULL_A_OUTCOME_INVALID)
    )
    expected_support = torch.where(
        recovery_observable[:, None],
        env._full_a_foot_supported_lr,
        torch.zeros_like(env._full_a_foot_supported_lr),
    ).to(dtype=critic.dtype)
    expected_dwell = torch.where(
        recovery_observable,
        env._full_a_cadence_ready_streak.clamp(0, 2).to(dtype=critic.dtype),
        torch.zeros_like(env._full_a_cadence_ready_streak, dtype=critic.dtype),
    )[:, None] / 2.0
    assert torch.equal(privileged("foot_supported_lr"), expected_support)
    assert torch.equal(
        privileged("cadence_ready_dwell_fraction"), expected_dwell
    )


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_n1_zero_action_step_and_forced_timeout_reset():
    env = _gpu_env(num_envs=1)
    try:
        zero = torch.zeros((1, 31), device=env.device)
        env.reset()
        probe_action = zero.clone()
        probe_action[:, 0] = 0.2
        env._advance_plant(probe_action)
        before_forward = env.sim.data.cvel[:, env._fullmdp_body_ids].clone()
        env.sim.forward()
        after_forward = env.sim.data.cvel[:, env._fullmdp_body_ids].clone()
        assert torch.any(before_forward.ne(after_forward))
        body_lin, body_ang = env._body_com_velocities_w()
        assert torch.isfinite(body_lin).all()
        assert torch.isfinite(body_ang).all()
        env.reset()
        before_step = int(env.common_step_counter)
        _, _, dones0, extras0 = _assert_step_surface(env.step(zero), num_envs=1)
        assert int(env.common_step_counter) == before_step + 1
        assert torch.equal(dones0, torch.zeros_like(dones0))
        assert not bool(extras0["time_outs"].any())
        assert torch.count_nonzero(extras0["termination_bits"]) == 0

        generation0 = extras0["reset_generation"].clone()
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, dones1, extras1 = _assert_step_surface(env.step(zero), num_envs=1)
        assert torch.equal(dones1, torch.ones_like(dones1))
        assert bool(extras1["time_outs"].all())
        assert torch.equal(
            extras1["termination_bits"],
            torch.full_like(extras1["termination_bits"], 1),
        )
        assert torch.equal(extras1["reset_generation"], generation0 + 1)
        assert torch.count_nonzero(env.episode_length_buf) == 0
        assert torch.count_nonzero(env.actions) == 0
        assert torch.count_nonzero(env.last_actions) == 0

        expected_park = torch.tensor(
            wait_env.WAIT_BALL_PARK_HOPE,
            dtype=env.qpos_init.dtype,
            device=env.device,
        ) + env.hope_to_scene
        assert torch.equal(
            env.sim.data.qpos[0, env.b_q : env.b_q + 3], expected_park
        )
        assert torch.count_nonzero(
            env.sim.data.qvel[0, env.b_v : env.b_v + 6]
        ) == 0
    finally:
        env.close()


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_n2_timeout_reset_preserves_every_peer_reset_owned_row_exactly():
    try:
        env = _gpu_env(num_envs=2)
    except ValueError as exc:
        if "requires nworld=1" in str(exc):
            pytest.skip("current WAIT slice is intentionally restricted to N=1")
        raise
    try:
        zero = torch.zeros((2, 31), device=env.device)
        _, _, _, extras0 = _assert_step_surface(env.step(zero), num_envs=2)
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.episode_length_buf[1] = 0
        peer_generation = extras0["reset_generation"][1].clone()

        original_reset_idx = env._reset_idx
        reset_checked = {"value": False}
        plant_names = (
            "qpos",
            "qvel",
            "ctrl",
            "qacc",
            "qacc_warmstart",
            "act",
            "time",
            "qfrc_applied",
            "xfrc_applied",
            "eq_active",
            "mocap_pos",
            "mocap_quat",
        )
        buffer_names = (
            "actions",
            "last_actions",
            "episode_length_buf",
            "ball_age_buf",
            "_epoch_phase",
            "_epoch_task_valid",
            "_epoch_selected",
            "_epoch_launch_succeeded",
            "last_terminal_bits",
            "reset_generation",
            "_cur_ret",
            "_cur_min_d",
            "_cur_touched",
            "_cur_robot_table",
            "_cur_table_keepout",
        )

        def checked_reset(_self, ids):
            assert torch.equal(ids, torch.tensor([0], device=env.device))
            plant_before = {
                name: getattr(env.sim.data, name)[1].clone()
                for name in plant_names
                if torch.is_tensor(getattr(env.sim.data, name, None))
                and getattr(env.sim.data, name).ndim >= 1
                and getattr(env.sim.data, name).shape[0] == env.num_envs
            }
            buffers_before = {
                name: getattr(env, name)[1].clone() for name in buffer_names
            }
            original_reset_idx(ids)
            for name, expected in plant_before.items():
                assert torch.equal(getattr(env.sim.data, name)[1], expected), name
            for name, expected in buffers_before.items():
                assert torch.equal(getattr(env, name)[1], expected), name
            reset_checked["value"] = True

        env._reset_idx = MethodType(checked_reset, env)

        _, _, dones, extras = _assert_step_surface(env.step(zero), num_envs=2)
        assert reset_checked["value"]
        assert torch.equal(dones, torch.tensor([1, 0], device=env.device))
        assert torch.equal(extras["time_outs"], torch.tensor([True, False], device=env.device))
        assert torch.equal(extras["termination_bits"], torch.tensor([1, 0], device=env.device))
        assert extras["reset_generation"][0] == extras0["reset_generation"][0] + 1
        assert extras["reset_generation"][1] == peer_generation
    finally:
        env.close()


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_full_a_n1_zero_action_survival_opportunity_accepts_without_next_tick_retry():
    env = _gpu_env(num_envs=1, full_a_mode=True)
    try:
        zero = torch.zeros((1, 31), device=env.device)
        env.reset()
        _jump_real_full_a_clock_to_transition_before_first_due(env)
        generation = env.reset_generation.clone()
        _, _, dones0, tick1 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(dones0.any())
        assert not bool(tick1["full_a_reveal_due_event"][0])
        assert not bool(tick1["full_a_reveal_event"][0])
        # The balance prefix itself earns the first learnable question.  This
        # live-physics check deliberately does not pin R07 readiness: that
        # telemetry may improve without changing curriculum admission.  The
        # controlled host A/B in this file owns the no-R07-admission assertion.

        _, _, dones0, extras0 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert bool(extras0["full_a_reveal_due_event"][0])
        accepted = bool(extras0["full_a_reveal_event"][0])
        deferred = bool(extras0["full_a_reveal_deferred_event"][0])
        assert accepted
        assert not deferred
        assert not bool(extras0["full_a_launch_event"][0])
        assert (
            extras0["full_a_phase_before_reset"][0]
            == wait_env.FULL_A_PHASE_REVEAL_COMMITTED
        )
        if not bool(dones0[0]):
            assert bool(env._epoch_task_valid[0])
            assert bool(env._epoch_selected[0])
            assert torch.equal(env.reset_generation, generation)
            assert int(env._full_a_next_reveal_tick[0]) == (
                int(env._full_a_cadence.first_reveal_tick)
                + int(env._full_a_cadence.cadence_ticks)
            )
        assert not bool(extras0["full_a_r03_present_event"][0])
        assert torch.count_nonzero(extras0["reward_terms"][:, :14]) == 0

        # ACCEPT consumes this frozen opportunity and cannot slide it to the
        # next tick.  Busy rows exercise the rowwise DEFER path separately.
        _, _, dones1, extras1 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(extras1["full_a_reveal_due_event"][0])
        assert not bool(extras1["full_a_reveal_event"][0])
        assert not bool(extras1["full_a_reveal_deferred_event"][0])
        if not bool(dones0[0]) and not bool(dones1[0]):
            assert torch.equal(env.reset_generation, generation)

    finally:
        env.close()


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_full_a_n1_returned_observation_tracks_motion_reference_lifecycle():
    """Prove the real returned 215/231 boundary across all teacher stages."""

    env = _gpu_env(num_envs=1, full_a_mode=True)
    try:
        zero = torch.zeros((1, 31), device=env.device)
        reset_observations, _reset_extras = env.reset()
        _assert_real_full_a_returned_observation(
            env,
            reset_observations,
            expected_action=zero,
            expected_task_visible=False,
            expected_live_ball=False,
        )
        assert not bool(env._epoch_task_valid[0])
        assert env._full_a_motion_phase_code[0] == wait_env.READY_HOLD_PHASE_INDEX
        assert torch.count_nonzero(env._full_a_teacher_joint_pos - env.q_ready) == 0
        assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0

        _jump_real_full_a_clock_to_transition_before_first_due(env)
        ready, _, ready_done, ready_extras = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(ready_done[0])
        assert not bool(ready_extras["full_a_reveal_event"][0])
        _assert_real_full_a_returned_observation(
            env,
            ready,
            expected_action=zero,
            expected_task_visible=False,
            expected_live_ball=False,
        )

        accepted, _, accepted_done, accepted_extras = (
            _assert_full_a_step_surface(env.step(zero), num_envs=1)
        )
        assert not bool(accepted_done[0])
        assert bool(accepted_extras["full_a_reveal_event"][0])
        assert env._epoch_clock_ticks[0, 0] == int(env.common_step_counter)
        assert env._full_a_motion_phase_code[0] == (
            wait_env.FULL_A_MOTION_PREPARE_PHASE_INDEX
        )
        assert env._full_a_teacher_frame[0] == 0
        torch.testing.assert_close(
            env._full_a_teacher_joint_pos[0],
            env._full_a_teacher.joint_pos[0],
            rtol=0.0,
            atol=2.0e-6,
        )
        assert torch.count_nonzero(env._full_a_teacher_joint_vel[0]) == 0
        _assert_real_full_a_returned_observation(
            env,
            accepted,
            expected_action=zero,
            expected_task_visible=True,
            expected_live_ball=False,
        )

        def jump_common_to(boundary):
            boundary = int(boundary)
            delta = boundary - int(env.common_step_counter)
            assert delta >= 0
            env.common_step_counter = boundary
            env.episode_length_buf += delta

        reveal_tick = int(env._epoch_clock_ticks[0, 0])
        pre_wait_s = float(env._full_a_pre_swing_wait_s[0].item())
        first_playback_tick = reveal_tick + math.floor(
            pre_wait_s / float(env.step_dt) + 1.0e-12
        ) + 1
        launch_tick = int(env._epoch_clock_ticks[0, 2])
        assert reveal_tick < first_playback_tick <= launch_tick
        jump_common_to(first_playback_tick - 1)
        playback_action = zero.clone()
        playback_action[:, 0] = 0.01
        playback, _, playback_done, _playback_extras = (
            _assert_full_a_step_surface(
                env.step(playback_action), num_envs=1
            )
        )
        assert not bool(playback_done[0])
        assert env._full_a_motion_phase_code[0] == (
            wait_env.FULL_A_MOTION_SWING_PHASE_INDEX
        )
        elapsed_s = torch.full(
            (1,),
            (int(env.common_step_counter) - reveal_tick) * float(env.step_dt),
            dtype=env.qpos_init.dtype,
            device=env.device,
        )
        expected_playback = wait_env.portable_question.sample_motion_teacher(
            torch,
            env._full_a_teacher,
            elapsed_s,
            env._full_a_teacher_rate,
            env._full_a_pre_swing_wait_s,
        )
        # Leaving HOLD is time-owned.  At the first positive active-time
        # boundary the rounded measured sampler may still legitimately own
        # frame zero; the returned observation must follow that exact owner.
        assert torch.equal(
            env._full_a_teacher_frame, expected_playback["frame"]
        )
        _assert_real_full_a_returned_observation(
            env,
            playback,
            expected_action=playback_action,
            expected_task_visible=True,
            expected_live_ball=False,
        )

        # Hit the exact launch authority rather than jumping over it.  The
        # returned boundary is the first integrated live-ball history row.
        jump_common_to(launch_tick)
        launched, _, launched_done, launched_extras = (
            _assert_full_a_step_surface(env.step(zero), num_envs=1)
        )
        assert not bool(launched_done[0])
        assert bool(launched_extras["full_a_launch_event"][0])
        _assert_real_full_a_returned_observation(
            env,
            launched,
            expected_action=zero,
            expected_task_visible=True,
            expected_live_ball=True,
        )

        task_close = int(env._epoch_clock_ticks[0, 3])
        assert int(env.common_step_counter) < task_close
        jump_common_to(task_close - 1)
        closed, _, closed_done, closed_extras = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(closed_done[0])
        assert bool(closed_extras["full_a_flight_terminal_event"][0])
        assert bool(closed_extras["full_a_r06_present_event"][0])
        # Motion close is inclusive: this exact returned boundary still owns
        # the final measured suffix, even though physical R06 has settled.
        assert env._full_a_motion_phase_code[0] == (
            wait_env.FULL_A_MOTION_FOLLOW_PHASE_INDEX
        )
        _assert_real_full_a_returned_observation(
            env,
            closed,
            expected_action=zero,
            expected_task_visible=True,
            expected_live_ball=False,
        )

        recovery_action = zero.clone()
        recovery_action[:, 1] = -0.01
        recovery, _, recovery_done, recovery_extras = (
            _assert_full_a_step_surface(
                env.step(recovery_action), num_envs=1
            )
        )
        assert not bool(recovery_done[0])
        assert not bool(recovery_extras["full_a_shot_retired_event"][0])
        assert env._full_a_motion_phase_code[0] == wait_env.READY_HOLD_PHASE_INDEX
        assert env._full_a_teacher_frame[0] == 0
        torch.testing.assert_close(
            env._full_a_teacher_joint_pos[0],
            env._full_a_teacher.joint_pos[0],
            rtol=0.0,
            atol=2.0e-6,
        )
        assert torch.count_nonzero(env._full_a_teacher_joint_vel[0]) == 0
        _assert_real_full_a_returned_observation(
            env,
            recovery,
            expected_action=recovery_action,
            expected_task_visible=False,
            expected_live_ball=False,
        )

        # Ages 10..77 are the fixed Motion-owned recovery tape.  Run its real
        # producer/step path (only the N/A ages 2..9 are skipped) and prove the
        # final returned boundary is the nonterminating retired observation.
        jump_common_to(task_close + wait_env.FULL_A_RECOVERY_START_AGE_TICK - 1)
        retired = None
        retired_extras = None
        for age in range(
            wait_env.FULL_A_RECOVERY_START_AGE_TICK,
            wait_env.FULL_A_RECOVERY_END_AGE_TICK + 1,
        ):
            retired, _, dones, retired_extras = _assert_full_a_step_surface(
                env.step(zero), num_envs=1
            )
            assert int(env.common_step_counter) == task_close + age
            assert not bool(dones[0])
            assert torch.equal(
                retired["policy"], env.get_observations()["policy"]
            )
            assert torch.equal(
                retired["critic"][:, : P.ACTOR_WIDTH_V3],
                retired["policy"],
            )
        assert retired is not None and retired_extras is not None
        assert bool(retired_extras["full_a_shot_retired_event"][0])
        assert not bool(retired_extras["full_a_completed_action_epoch_event"][0])
        assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_RETIRED
        _assert_real_full_a_returned_observation(
            env,
            retired,
            expected_action=zero,
            expected_task_visible=False,
            expected_live_ball=False,
        )
    finally:
        env.close()


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_full_a_n1_launch_reports_live_selected_rubber_contact():
    env = _gpu_env(num_envs=1, full_a_mode=True)
    try:
        zero = torch.zeros((1, 31), device=env.device)
        env.reset()
        _jump_real_full_a_clock_to_transition_before_first_due(env)
        _, _, _, tick1 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(tick1["full_a_reveal_due_event"][0])
        # This test owns the downstream live-contact callpoint.  Survival to
        # the real frozen tick-295 opportunity exposes the question directly;
        # no tests-only readiness override may hide that production contract.
        _, _, _, reveal = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert bool(reveal["full_a_reveal_due_event"][0])
        assert bool(reveal["full_a_reveal_event"][0])
        assert not bool(reveal["full_a_launch_event"][0])

        racket_ids = env._geom_class.eq(1).nonzero(as_tuple=False).squeeze(-1)
        assert int(racket_ids.numel()) >= 1
        racket_gid = int(racket_ids[0].item())
        # A mesh geom's frame origin is not guaranteed to lie inside its
        # collision volume.  The production measured-racket site is the
        # authoritative live blade point and is independently resolved from
        # the actual model; placing the ball there must create a real pair.
        site = env.sim.data.site_xpos[0, env.racket_sid].clone()
        rotation = env.sim.data.site_xmat[0, env.racket_sid].reshape(3, 3)
        local_selected_center = torch.tensor(
            [
                wait_env.racket_contact_geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M[0],
                wait_env.racket_contact_geometry.BALL_RADIUS_M - 0.001,
                wait_env.racket_contact_geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M[1],
            ],
            dtype=site.dtype,
            device=site.device,
        )
        # Exact tangency is not a resolved-contact guarantee: the pinned ball
        # radius is about 20 mm and MuJoCo may legitimately emit no pair at
        # zero penetration.  Keep the centre on the selected side while
        # penetrating the measured outer face by an explicit 1 mm.
        racket_center = site + rotation @ local_selected_center
        env._full_a_launch_state_f32[0, :3] = (
            racket_center - env.env.scene.env_origins[0]
        )
        env._full_a_launch_state_f32[0, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device
        )
        env._full_a_launch_state_f32[0, 7:13] = 0.0

        # Prove the constructed overlap reaches MuJoCo-Warp's own contact
        # array before asking the production step/latch path to report it.
        env.sim.data.qpos[0, env.b_q : env.b_q + 3] = racket_center
        env.sim.data.qpos[0, env.b_q + 3 : env.b_q + 7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device
        )
        env.sim.data.qvel[0, env.b_v : env.b_v + 6] = 0.0
        env.sim.forward()
        geom = env._con_geom
        valid = env._con_idx < env._nacon[0]
        live_pair = valid & (
            (geom[:, 0].eq(env._ball_gid) & geom[:, 1].eq(racket_gid))
            | (geom[:, 1].eq(env._ball_gid) & geom[:, 0].eq(racket_gid))
        )
        assert bool(live_pair.any())

        # One real physics substep retains the forced overlap while still
        # crossing the production launch -> contact -> fact publication path.
        env.decimation = 1
        env._epoch_clock_ticks[0, 2] = int(env.common_step_counter)
        _, _, dones, contact = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(dones[0])
        assert bool(contact["full_a_launch_event"][0])
        assert bool(contact["full_a_racket_contact_event"][0])
        assert bool(contact["full_a_selected_contact_event"][0])
        assert not bool(contact["full_a_opposite_contact_event"][0])
        assert contact["full_a_contact_classification_status"][0] == (
            wait_env.racket_contact_geometry.OBSERVED_RUBBER_STATUS_SELECTED
        )
        torch.testing.assert_close(
            contact["reward_terms"][0, 10],
            contact["reward_terms"].new_tensor(env.step_dt),
            rtol=0.0,
            atol=1.0e-7,
        )
        assert bool(contact["full_a_racket_contact"][0])
        assert torch.count_nonzero(env._full_a_physical_fact_f32[0, 3:6]) > 0
        assert torch.equal(
            env._full_a_physical_fact_f32[0, 3:6],
            env._full_a_contact_center[0],
        )
    finally:
        env.close()


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_full_a_n2_selected_outcome_preserves_peer_rows():
    env = _gpu_env(num_envs=2, full_a_mode=True)
    try:
        zero = torch.zeros((2, 31), device=env.device)
        env.reset()
        _jump_real_full_a_clock_to_transition_before_first_due(env)
        _, _, _, tick1 = _assert_full_a_step_surface(
            env.step(zero), num_envs=2
        )
        assert not bool(tick1["full_a_reveal_due_event"].any())
        # Survival to the frozen opportunity exposes both rows directly.
        # Keep this peer/outcome test on the production path without a
        # tests-only R07 readiness override.
        _, _, _, reveal = _assert_full_a_step_surface(
            env.step(zero), num_envs=2
        )
        assert bool(reveal["full_a_reveal_event"].all())
        env._epoch_clock_ticks[:, 2] = int(env.common_step_counter)
        _, _, _, before = _assert_full_a_step_surface(env.step(zero), num_envs=2)
        peer_generation = before["reset_generation"][1].clone()
        generation_before = env.reset_generation.clone()
        peer_task = env._epoch_task_f32[1].clone()
        outcome_boundary = int(env.common_step_counter) + 1
        env._epoch_clock_ticks[0, 3] = outcome_boundary
        env._epoch_clock_ticks[1, 3] += 100
        _, _, outcome_dones, outcome_extras = _assert_full_a_step_surface(
            env.step(zero), num_envs=2
        )
        assert int(env.common_step_counter) == outcome_boundary
        assert torch.equal(
            outcome_dones, torch.tensor([0, 0], device=env.device)
        )
        assert (
            outcome_extras["full_a_outcome_code"][0]
            == wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED
        )
        assert (
            outcome_extras["full_a_phase_before_reset"][0]
            == wait_env.FULL_A_PHASE_OUTCOME_SETTLED
        )
        assert bool(outcome_extras["full_a_flight_terminal_event"][0])
        assert bool(outcome_extras["full_a_r06_present_event"][0])
        assert bool(outcome_extras["full_a_r06_eligible_event"][0])
        assert not bool(outcome_extras["full_a_flight_terminal_event"][1])
        assert not bool(outcome_extras["full_a_r06_present_event"][1])
        assert not bool(outcome_extras["full_a_r06_eligible_event"][1])
        assert not bool(outcome_extras["full_a_selected_reset_event"].any())
        assert not bool(outcome_extras["full_a_shot_retired_event"].any())
        assert torch.equal(outcome_extras["reset_generation"], generation_before)
        assert outcome_extras["reset_generation"][1] == peer_generation
        assert torch.equal(env._epoch_task_f32[1], peer_task)
    finally:
        env.close()
