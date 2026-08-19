"""Direct contracts for the real MuJoCo FullMDP WAIT transition.

The host tests exercise the same reward and termination helpers used by the
production ``step`` method with fixed tensors.  The opt-in GPU tests cross the
real ``A3ReadyBallVecEnv._advance_plant`` callpoint; no legacy M04 owner,
receipt, or synthetic VecEnv is accepted as runtime evidence.
"""

from __future__ import annotations

import os
from pathlib import Path
import ast
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env


P = wait_env.observation_contract
RUN_GPU_DIRECT = os.environ.get("ACTIONBALL_RUN_MUJOCO_GPU_DIRECT") == "1"


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
    assert first_line("_fullmdp_termination") < first_line("_fullmdp_reward20")


def test_post_forward_resolved_table_contact_is_latched_without_counter_replay():
    env = SimpleNamespace(
        _torch=torch,
        _con_geom=torch.tensor([[5, 2], [7, 8], [2, 1]], dtype=torch.int64),
        _con_idx=torch.arange(3),
        _nacon=torch.tensor([2], dtype=torch.int64),
        _con_world=torch.tensor([0, 1, 0], dtype=torch.int64),
        _table_gid=5,
        _is_robot_geom=torch.tensor(
            [False, False, True, False, False, False, False, False, False]
        ),
        _cur_robot_table=torch.zeros(2),
        num_envs=2,
    )
    wait_env.FullMdpInitialWaitVecEnv._latch_post_forward_resolved_table_contacts(env)
    assert torch.equal(env._cur_robot_table, torch.tensor([1.0, 0.0]))


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
        _teacher_body_pos=teacher_pos,
        _teacher_body_quat=teacher_quat,
        _teacher_body_lin_vel=teacher_lin,
        _teacher_body_ang_vel=teacher_ang,
        _fullmdp_dense_weights=torch.tensor(
            [0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=dtype
        ),
        _quat_error_sq=wait_env.FullMdpInitialWaitVecEnv._quat_error_sq,
        _aligned_teacher_body_pose=(
            wait_env.FullMdpInitialWaitVecEnv._aligned_teacher_body_pose
        ),
        _body_com_velocities_from_cvel=(
            wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_from_cvel
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


def test_host_reward20_matches_six_dense_formulas_and_weight_times_step_dt():
    env = _fixed_reward_env()
    reward, terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward20(env)

    data = env.sim.data
    ids = env._fullmdp_body_ids
    pos = data.xpos[:, ids]
    quat = data.xquat[:, ids]
    lin, ang = wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_w(env)
    anchor = env._fullmdp_anchor_index
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
                -torch.square(aligned_pos - pos)
                .sum(dim=-1)
                .mean(dim=-1)
                / 0.3**2
            ),
            torch.exp(-quat_error_sq.mean(dim=-1) / 0.4**2),
            torch.exp(
                -torch.square(env._teacher_body_lin_vel - lin)
                .sum(dim=-1)
                .mean(dim=-1)
                / 1.0**2
            ),
            torch.exp(
                -torch.square(env._teacher_body_ang_vel - ang)
                .sum(dim=-1)
                .mean(dim=-1)
                / 3.14**2
            ),
        ),
        dim=1,
    )
    weights = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=terms.dtype)
    expected_dense = expected_raw * weights * env.step_dt

    assert tuple(terms.shape) == (2, 20)
    assert torch.count_nonzero(terms[:, :14]) == 0
    assert torch.allclose(terms[:, 14:], expected_dense, rtol=0.0, atol=1.0e-15)
    assert torch.allclose(reward, expected_dense.sum(dim=1), rtol=0.0, atol=1.0e-15)

    # Counterexample: this buffer is live Reward20 authority, not a private
    # command-ramp scratch tensor.  Replacing one exact teacher body with a
    # midpoint changes the body-position reward immediately.
    exact_body_term = terms[:, 16].clone()
    env._teacher_body_pos[:, 0, 0] += 0.25
    env._refresh_aligned_teacher_body_pose()
    _drifted_reward, drifted_terms = (
        wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward20(env)
    )
    assert not torch.equal(drifted_terms[:, 16], exact_body_term)


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

    _, first_terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward20(env)
    first_raw = first_terms[:, 14:] / (
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
    _, second_terms = wait_env.FullMdpInitialWaitVecEnv._fullmdp_reward20(env)
    second_raw = second_terms[:, 14:] / (
        env._fullmdp_dense_weights * env.step_dt
    )
    assert second_raw[0, 0] < 1.0
    assert second_raw[0, 1] == 1.0
    assert torch.equal(second_raw[0, 2:4], torch.ones(2, dtype=dtype))
    assert second_raw[1, 0] == 1.0
    assert second_raw[1, 1] < 1.0
    assert torch.equal(second_raw[1, 2:4], torch.ones(2, dtype=dtype))


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


def test_host_termination_bits_keep_timeout_independent_of_physical_terminal():
    num_envs = 7
    env = SimpleNamespace(
        _torch=torch,
        episode_length_buf=torch.tensor([0, 10, 10, 0, 0, 0, 0]),
        max_episode_length=10,
        _cur_robot_table=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        _cur_table_keepout=torch.tensor([False, False, False, False, False, False, True]),
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
        truncated, torch.tensor([False, True, True, False, False, False, False])
    )
    assert torch.equal(bits, torch.tensor([0, 1, 3, 4, 8, 0, 16]))
    assert torch.equal(
        resolved_table,
        torch.tensor([False, False, False, False, False, True, False]),
    )


def test_step_falls_back_per_joint_but_preserves_raw_nonfinite_qdes_evidence():
    captured = {}
    previous = torch.tensor([[-0.25, 0.30], [0.20, -0.40]])

    def advance(actions):
        captured["safe_actions"] = actions.clone()
        return {}, torch.zeros(2), torch.zeros_like(actions)

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
        _latch_post_forward_resolved_table_contacts=lambda: None,
        _latch_post_forward_table_keepout=lambda: None,
        _state=lambda: {},
        _fullmdp_termination=terminate,
        _fullmdp_reward20=lambda: (torch.zeros(2), torch.zeros(2, 20)),
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
        task = torch.zeros((n, P.TASK_F32_WIDTH))
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
            "ttc_ticks": torch.full((n,), 2, dtype=torch.long),
            "launch_horizon_ticks": torch.ones(n, dtype=torch.long),
            "teacher_rate": torch.ones(n),
            "scaled_t_hit_s": torch.full((n,), 0.02),
            "scaled_t_cycle_s": torch.full((n,), 0.04),
            "pre_swing_wait_s": torch.full((n,), 0.02),
        }

    env = SimpleNamespace(
        _torch=torch,
        num_envs=n,
        device=torch.device("cpu"),
        q_ready=torch.zeros(31),
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
                site_xpos=torch.tensor(
                    [[[0.0, -0.76, 1.05]], [[0.1, -0.76, 1.05]]]
                ),
                site_xmat=torch.eye(3).reshape(1, 1, 3, 3).repeat(n, 1, 1, 1),
                cvel=torch.zeros((n, 1, 6)),
                subtree_com=torch.zeros((n, 1, 3)),
            )
        ),
        racket_sid=0,
        _fullmdp_racket_body_id=0,
        _fullmdp_racket_root_id=0,
        root_qadr=0,
        b_q=7,
        b_v=0,
        ball_age_buf=torch.zeros(n, dtype=torch.long),
        common_step_counter=0,
        step_dt=0.02,
        full_a_mode=True,
        _fullmdp_initialized=True,
        _all_env_ids=torch.arange(n),
        _epoch_phase=torch.zeros(n, dtype=torch.long),
        _epoch_task_valid=torch.zeros(n, dtype=torch.bool),
        _epoch_selected=torch.zeros(n, dtype=torch.bool),
        _epoch_launch_succeeded=torch.zeros(n, dtype=torch.bool),
        _cur_touched=torch.zeros(n),
        _cur_robot_table=torch.zeros(n),
        _cur_table_keepout=torch.zeros(n, dtype=torch.bool),
        cfg=SimpleNamespace(
            ball_dead_z_hope=-0.35,
            ball_dead_x_lo_hope=-1.2,
            ball_dead_x_hi_hope=3.4,
        ),
        _full_a_catalog=SimpleNamespace(
            fresh_action=SimpleNamespace(
                action_slot=0,
                action_uid=6907688916670928,
                mount_normal_sign=1,
            )
        ),
        _full_a_teacher=SimpleNamespace(contact_reference_root_z_scene=0.9),
        _full_a_question_builder=centre_question_stub,
    )
    env._clear_lifecycle = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._clear_lifecycle, env
    )
    for name in (
        "_full_a_reveal_rows",
        "_full_a_launch_rows",
        "_full_a_park_rows",
        "_full_a_prepare_step",
        "_full_a_racket_kinematics",
        "_full_a_publish_r03_fact",
        "_full_a_publish_physical_fact",
        "_full_a_settle_outcome",
    ):
        setattr(
            env,
            name,
            MethodType(getattr(wait_env.FullMdpInitialWaitVecEnv, name), env),
        )
    wait_env.FullMdpInitialWaitVecEnv._initialize_full_a_state(env)
    return env


def test_host_full_a_reveal_launch_physical_fact_and_selected_clear_are_rowwise():
    env = _host_full_a_lifecycle_env()
    reveal, launch = wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)

    assert reveal.all()
    assert not launch.any()

    assert torch.equal(
        env._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_REVEAL_COMMITTED),
    )
    assert env._epoch_task_valid.all() and env._epoch_selected.all()
    assert torch.isfinite(env._epoch_task_f32).all()
    assert torch.count_nonzero(env._epoch_task_f32[:, :32]) > 0
    assert torch.equal(env._epoch_task_f32[:, 32:], env._full_a_launch_state_f32)
    assert torch.equal(env._epoch_clock_ticks[:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(env._epoch_clock_ticks[:, 2], torch.ones(2, dtype=torch.long))

    env.common_step_counter = 1
    reveal, launch = wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    assert not reveal.any()
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
    env.common_step_counter = 2
    present, physically_valid = env._full_a_publish_r03_fact()
    assert present.all() and physically_valid.all()
    assert torch.equal(
        env._full_a_owner_valid_bits[:, 1],
        torch.full((2,), 3, dtype=torch.long),
    )
    assert torch.equal(
        env._full_a_r03_fact_f32[:, 15:18],
        env.sim.data.site_xpos[:, 0],
    )
    assert torch.equal(
        env._full_a_r03_fact_f32[:, 21:24],
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    )

    env.sim.data.qpos[:, env.b_q : env.b_q + 3] += torch.tensor(
        [[-0.06, 0.01, -0.02], [-0.04, -0.01, 0.03]]
    )
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_physical_fact(env)
    assert env._full_a_physical_present.all()
    assert torch.equal(
        env._full_a_physical_fact_f32[:, :3],
        env.sim.data.qpos[:, env.b_q : env.b_q + 3],
    )
    assert torch.count_nonzero(env._full_a_physical_fact_f32[:, 10:]) == 0

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
            "_full_a_contact_center",
            "_full_a_outcome_code",
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
    for name, expected in peer.items():
        assert torch.equal(getattr(env, name)[1], expected), name


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
    torch.testing.assert_close(task[0, 14:17], torch.tensor([1.0, -0.6, 1.0]))
    torch.testing.assert_close(task[1, 14:17], torch.tensor([0.7, 0.7, 1.0]))
    torch.testing.assert_close(task[0, 24:26], torch.tensor([0.3, -0.1]))
    torch.testing.assert_close(task[1, 24:26], torch.tensor([0.2, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(task[0, 26:29], torch.tensor([-3.0, 0.0, 0.0]))
    torch.testing.assert_close(
        task[1, 26:29],
        torch.tensor([0.0, -3.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    # Launch is reverse-integrated from the action question.  It is neither
    # the legacy serve-range midpoint nor a forward gravity shortcut.
    torch.testing.assert_close(task[0, 32:35], torch.tensor([2.2, -0.6, 1.0]))
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
    assert not torch.equal(
        first["task_f32"][:, 14:17], second["task_f32"][:, 14:17]
    )
    assert not torch.equal(
        first["task_f32"][:, 26:29], second["task_f32"][:, 26:29]
    )
    assert not torch.equal(
        first["launch_state_f32"][:, :3],
        second["launch_state_f32"][:, :3],
    )


def test_full_a_reveal_packs_env_local_question_and_launch_restores_world_origin():
    env = _host_full_a_lifecycle_env()
    origins = torch.tensor([[6.0, -12.0, 0.0], [-6.0, 12.0, 0.0]])
    local_base = torch.tensor([[0.1, -0.2, 0.9], [0.3, -0.4, 0.9]])
    env.env.scene.env_origins = origins
    env.sim.data.qpos[:, :3] = origins + local_base
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
        }

    env._full_a_question_builder = question
    ids = torch.arange(2)
    wait_env.FullMdpInitialWaitVecEnv._full_a_reveal_rows(env, ids)
    torch.testing.assert_close(captured["base"], local_base)
    env.sim.data.qpos[:, env.b_q : env.b_q + 3] = -100.0
    env.sim.data.qvel[:, env.b_v : env.b_v + 6] = 10.0
    env.common_step_counter = 1
    _reveal, launch = wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
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
    env.common_step_counter = 2
    _reveal, launch = wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    assert launch.all()
    torch.testing.assert_close(
        env.sim.data.qpos[:, env.b_q : env.b_q + 3],
        origins + env._full_a_launch_state_f32[:, :3],
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

    with pytest.raises(RuntimeError, match="frozen steps are negative"):
        wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge(
            torch=torch,
            previous_qdes=previous,
            frame0_qdes=frame0,
            frozen_steps=torch.tensor([-1], dtype=torch.long),
        )


def test_full_a_environment_consumes_the_measured_teacher_clock():
    env = _host_full_a_lifecycle_env()
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
    )
    env._ready_teacher_body_pos = torch.zeros((2, 1, 3))
    env._ready_teacher_body_quat = torch.zeros((2, 1, 4))
    env._ready_teacher_body_quat[..., 0] = 1.0
    env._ready_teacher_body_lin_vel = torch.zeros((2, 1, 3))
    env._ready_teacher_body_ang_vel = torch.zeros((2, 1, 3))
    env._teacher_body_pos = env._ready_teacher_body_pos.clone()
    env._teacher_body_quat = env._ready_teacher_body_quat.clone()
    env._teacher_body_lin_vel = env._ready_teacher_body_lin_vel.clone()
    env._teacher_body_ang_vel = env._ready_teacher_body_ang_vel.clone()
    env._refresh_aligned_teacher_body_pose = lambda: None
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)

    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)

    assert torch.equal(env._full_a_teacher_frame, torch.ones(2, dtype=torch.long))
    assert torch.equal(env._full_a_teacher_joint_pos, joint[1].repeat(2, 1))
    assert torch.equal(env._full_a_teacher_joint_vel, torch.ones((2, 31)))
    assert torch.equal(
        env._teacher_body_pos,
        body_pos[1].repeat(2, 1, 1) + origins[:, None, :],
    )
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.ones(2, dtype=torch.long),
    )

    env.common_step_counter = 4
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_motion_phase_code,
        torch.full((2,), 3, dtype=torch.long),
    )
    assert torch.equal(env._full_a_teacher_joint_pos, env.q_ready.repeat(2, 1))
    assert torch.equal(env._teacher_body_pos, env._ready_teacher_body_pos)


def test_full_a_reveal_teacher_is_atomic_frame0_then_strike_and_resets_rowwise():
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
        }

    env._full_a_question_builder = question
    physical_root_before = env.sim.data.qpos[:, :7].clone()
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)

    # The physical plant remains on its separately admitted ready reset.  The
    # Motion observation/reward teacher is a different channel and switches
    # atomically to exact measured frame zero at public reveal, just as Isaac
    # does; the diagnostic q_des bridge must never leak into these buffers.
    assert torch.max(torch.abs(joint[0] - env.q_ready)) > 3.0
    assert torch.equal(env.sim.data.qpos[:, :7], physical_root_before)
    assert not hasattr(env, "_full_a_ready_bridge_alpha")
    assert torch.equal(
        env._full_a_teacher_joint_pos, joint[0].repeat(2, 1)
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
    assert torch.equal(env._full_a_teacher_frame, torch.zeros(2, dtype=torch.long))
    assert torch.equal(env._full_a_motion_phase_code, torch.zeros(2, dtype=torch.long))

    # There is no halfway Motion target during the frozen prepare clock: both
    # actor teacher observation and Reward20 body teacher remain exact frame0.
    env.common_step_counter = 22
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_teacher_joint_pos, joint[0].repeat(2, 1)
    )
    assert torch.equal(
        env._teacher_body_pos, body_pos[0].repeat(2, 1, 1) + origins[:, None, :]
    )
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0
    assert torch.equal(env._full_a_motion_phase_code, torch.zeros(2, dtype=torch.long))

    # The exact end of the pre-wait is still frame0/prepare.  Only a later
    # rounded frame transition starts measured playback.
    env.common_step_counter = 45
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert torch.equal(
        env._full_a_teacher_joint_pos, joint[0].repeat(2, 1)
    )
    assert torch.equal(
        env._teacher_body_pos, body_pos[0].repeat(2, 1, 1) + origins[:, None, :]
    )
    assert torch.equal(env._full_a_teacher_frame, torch.zeros(2, dtype=torch.long))
    assert torch.equal(env._full_a_motion_phase_code, torch.zeros(2, dtype=torch.long))
    assert torch.count_nonzero(env._full_a_teacher_joint_vel) == 0

    env.common_step_counter = 97
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

    # A selected reset returns hidden references to the physical ready tuple.
    # The peer remains on its own strike target; the next public reveal again
    # switches only that row atomically to exact frame0.
    peer_joint = env._full_a_teacher_joint_pos[1].clone()
    peer_body = env._teacher_body_pos[1].clone()
    env._clear_lifecycle(torch.tensor([0]))
    torch.testing.assert_close(env._full_a_teacher_joint_pos[0], env.q_ready)
    torch.testing.assert_close(
        env._teacher_body_pos[0], env._ready_teacher_body_pos[0]
    )
    torch.testing.assert_close(env._full_a_teacher_joint_pos[1], peer_joint)
    torch.testing.assert_close(env._teacher_body_pos[1], peer_body)
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_update_teacher(env)
    assert env._epoch_task_valid[0]
    assert torch.equal(env.sim.data.qpos[:, :7], physical_root_before)
    assert torch.equal(env._full_a_teacher_joint_pos[0], joint[0])
    assert torch.count_nonzero(env._full_a_teacher_joint_vel[0]) == 0
    torch.testing.assert_close(env._full_a_teacher_joint_pos[1], peer_joint)


def test_host_full_a_packs_task_clocks_and_real_physical_r03_facts():
    env = _host_full_a_lifecycle_env()
    env.q_ready = torch.zeros(31)
    env.actions = torch.zeros((2, 31))
    env._qpos_act = lambda: torch.zeros((2, 31))
    env._qvel_act = lambda: torch.zeros((2, 31))
    env._con_geom = torch.tensor([[0, 0]], dtype=torch.long)
    env._con_idx = torch.tensor([0], dtype=torch.long)
    env._nacon = torch.tensor([0], dtype=torch.long)
    env._ball_gid = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env.common_step_counter = 2
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_r03_fact(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_physical_fact(env)
    exact_teacher_joint = torch.arange(31, dtype=torch.float32).repeat(2, 1)
    exact_teacher_vel = -exact_teacher_joint
    env._full_a_teacher_joint_pos.copy_(exact_teacher_joint)
    env._full_a_teacher_joint_vel.copy_(exact_teacher_vel)

    wait_env.FullMdpInitialWaitVecEnv._compute_obs(
        env,
        st={
            "proj_g": torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
            "base_ang_b": torch.zeros((2, 3)),
        },
    )
    actor_offset = dict()
    start = 0
    for name, width in P.ACTOR_LAYOUT_V1:
        actor_offset[name] = slice(start, start + width)
        start += width
    critic_offset = dict()
    start = P.ACTOR_WIDTH_V1
    for name, width in P.CRITIC_EXTENSION_LAYOUT_V1:
        critic_offset[name] = slice(start, start + width)
        start += width

    assert torch.equal(
        env._obs_buf[:, actor_offset["epoch_task_f32"]], env._epoch_task_f32
    )
    # These actor fields consume the Motion teacher buffers directly.  A q_des
    # bridge value inserted there would therefore become a cross-backend
    # observation-contract drift, not an internal control implementation.
    assert torch.equal(
        env._obs_buf[:, actor_offset["teacher_joint_pos_rel"]],
        exact_teacher_joint,
    )
    assert torch.equal(
        env._obs_buf[:, actor_offset["teacher_joint_vel_rel"]],
        exact_teacher_vel,
    )
    expected_clocks = (
        env._epoch_clock_ticks - env.common_step_counter
    ).float() * env.step_dt
    assert torch.equal(
        env._obs_buf[:, actor_offset["epoch_clock_remaining_s"]], expected_clocks
    )
    present = env._critic_obs_buf[
        :, critic_offset["physical_r03_r06_r07_fact_present"]
    ]
    assert torch.equal(present, torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(2, 1))
    facts = env._critic_obs_buf[
        :, critic_offset["physical_r03_r06_r07_fact_f32"]
    ].reshape(2, 4, 32)
    assert torch.equal(facts[:, 0], env._full_a_physical_fact_f32)
    assert torch.equal(facts[:, 1], env._full_a_r03_fact_f32)
    assert torch.count_nonzero(facts[:, 2:]) == 0
    assert torch.count_nonzero(
        env._critic_obs_buf[:, critic_offset["reward_due"]]
    ) == 0
    assert torch.count_nonzero(
        env._critic_obs_buf[:, critic_offset["reward_paid"]]
    ) == 0


def test_host_full_a_outcome_uses_live_contact_or_bounded_flight_only():
    env = _host_full_a_lifecycle_env()
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env.common_step_counter = 1
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env._epoch_clock_ticks[1, 3] = 1
    ball_pos = env.sim.data.qpos[:, env.b_q : env.b_q + 3].clone()
    ball_pos[0, 0] = -2.0

    settled, outcome = wait_env.FullMdpInitialWaitVecEnv._full_a_settle_outcome(
        env, {"ball_pos": ball_pos}
    )
    assert settled.all()
    assert torch.equal(
        outcome,
        torch.tensor(
            [
                wait_env.FULL_A_OUTCOME_BALL_DEAD,
                wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED,
            ]
        ),
    )
    assert torch.equal(
        env._epoch_phase,
        torch.full((2,), wait_env.FULL_A_PHASE_OUTCOME_SETTLED),
    )


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


def test_host_live_generic_contact_classifies_selected_and_opposite_once():
    env = _host_full_a_lifecycle_env()
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    env.common_step_counter = 1
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

    wait_env.FullMdpInitialWaitVecEnv._full_a_begin_control_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts(env)
    assert not bool(env._full_a_generic_contact_event.any())
    assert not bool(env._full_a_selected_contact_event.any())


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
    assert tuple(extras["reward_terms"].shape) == (num_envs, 20)
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
        "full_a_phase_before_reset",
        "full_a_outcome_code",
        "full_a_racket_contact",
        "full_a_ball_table_contact",
        "full_a_physical_current_center",
        "full_a_reveal_event",
        "full_a_launch_event",
        "full_a_flight_terminal_event",
        "full_a_selected_reset_event",
        "full_a_racket_contact_eligible_event",
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
        "full_a_r03_present_event",
        "full_a_r03_physically_valid_event",
    }
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
    assert torch.count_nonzero(extras["reward_terms"][:, 11:14]) == 0
    assert torch.equal(extras["full_a_action_slot"], torch.zeros(num_envs, dtype=torch.long, device=extras["full_a_action_slot"].device))
    assert torch.equal(extras["full_a_action_uid"], torch.full((num_envs,), 6907688916670928, dtype=torch.long, device=extras["full_a_action_uid"].device))
    assert torch.equal(extras["full_a_mount_normal_sign"], torch.ones(num_envs, dtype=torch.int8, device=extras["full_a_mount_normal_sign"].device))
    assert torch.allclose(reward, extras["reward_terms"].sum(dim=1))
    return observations, reward, dones, extras


def _gpu_env(*, num_envs: int, full_a_mode: bool = False):
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("mjlab")
    pytest.importorskip("tensordict")
    task = wait_env.TaskCfg(
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
def test_real_full_a_n1_reveals_launches_flies_and_settles_one_shot():
    env = _gpu_env(num_envs=1, full_a_mode=True)
    try:
        zero = torch.zeros((1, 31), device=env.device)
        env.reset()
        generation = env.reset_generation.clone()
        obs0, _, dones0, extras0 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(dones0.any())
        assert extras0["full_a_phase_before_reset"][0] == wait_env.FULL_A_PHASE_REVEAL_COMMITTED
        assert not bool(extras0["full_a_r03_present_event"][0])
        task_start = 0
        for name, width in P.ACTOR_LAYOUT_V1:
            if name == "epoch_task_f32":
                break
            task_start += width
        assert torch.count_nonzero(obs0["policy"][0, task_start : task_start + 45]) > 0

        launch_center = env._full_a_launch_state_f32[0, :3].clone()
        # Close this exact shot at the launch transition.  Waiting for the
        # natural one-second horizon lets an independent posture/table
        # termination win the race and makes the test mistake an ordinary
        # physical terminal (outcome=NONE) for a broken Full-A settlement.
        # The forced deadline still crosses one real launch/physics step and
        # therefore proves movement, outcome publication, and selected reset.
        env._epoch_clock_ticks[0, 2] = int(env.common_step_counter)
        env._epoch_clock_ticks[0, 3] = int(env.common_step_counter)
        _, _, dones1, extras1 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        centers = [extras1["full_a_physical_current_center"][0].clone()]
        terminal = bool(dones1[0])
        last = extras1
        assert terminal
        assert any(not torch.equal(center, launch_center) for center in centers)
        assert last["full_a_outcome_code"][0] in (
            wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED,
            wait_env.FULL_A_OUTCOME_BALL_DEAD,
        )
        assert last["full_a_phase_before_reset"][0] == wait_env.FULL_A_PHASE_OUTCOME_SETTLED
        assert env.reset_generation[0] == generation[0] + 1
        assert torch.count_nonzero(last["reward_terms"][:, 10:14]) == 0
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
        _, _, _, reveal = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
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
        assert bool(contact["full_a_racket_contact_eligible_event"][0])
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
def test_real_full_a_n2_selected_outcome_reset_preserves_peer_rows():
    env = _gpu_env(num_envs=2, full_a_mode=True)
    try:
        zero = torch.zeros((2, 31), device=env.device)
        env.reset()
        _assert_full_a_step_surface(env.step(zero), num_envs=2)
        env._epoch_clock_ticks[:, 2] = int(env.common_step_counter)
        _, _, _, before = _assert_full_a_step_surface(env.step(zero), num_envs=2)
        peer_generation = before["reset_generation"][1].clone()
        env._epoch_clock_ticks[0, 3] = int(env.common_step_counter)
        env._epoch_clock_ticks[1, 3] += 100

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
            "_epoch_task_f32",
            "_epoch_clock_ticks",
            "_full_a_launch_state_f32",
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
            "_full_a_contact_center",
            "_full_a_outcome_code",
            "last_terminal_bits",
            "reset_generation",
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
        _, _, dones, extras = _assert_full_a_step_surface(
            env.step(zero), num_envs=2
        )
        assert reset_checked["value"]
        assert torch.equal(dones, torch.tensor([1, 0], device=env.device))
        assert extras["full_a_outcome_code"][0] == wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED
        assert extras["full_a_outcome_code"][1] == wait_env.FULL_A_OUTCOME_NONE
        assert extras["reset_generation"][1] == peer_generation
    finally:
        env.close()
