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
    qvel = torch.zeros((n, 12))
    env = SimpleNamespace(
        _torch=torch,
        num_envs=n,
        device=torch.device("cpu"),
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
            )
        ),
        racket_sid=0,
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
    )
    env._clear_lifecycle = MethodType(
        wait_env.FullMdpInitialWaitVecEnv._clear_lifecycle, env
    )
    for name in (
        "_full_a_reveal_rows",
        "_full_a_launch_rows",
        "_full_a_prepare_step",
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
            "_full_a_physical_fact_f32",
        )
    }
    env._clear_lifecycle(torch.tensor([0]))
    assert env._epoch_phase[0] == wait_env.FULL_A_PHASE_IDLE
    assert not env._epoch_task_valid[0]
    for name, expected in peer.items():
        assert torch.equal(getattr(env, name)[1], expected), name


def test_host_full_a_packs_task_clocks_and_only_real_physical_fact():
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
    env.common_step_counter = 1
    wait_env.FullMdpInitialWaitVecEnv._full_a_prepare_step(env)
    wait_env.FullMdpInitialWaitVecEnv._full_a_publish_physical_fact(env)

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
    expected_clocks = (
        env._epoch_clock_ticks - env.common_step_counter
    ).float() * env.step_dt
    assert torch.equal(
        env._obs_buf[:, actor_offset["epoch_clock_remaining_s"]], expected_clocks
    )
    present = env._critic_obs_buf[
        :, critic_offset["physical_r03_r06_r07_fact_present"]
    ]
    assert torch.equal(present, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    facts = env._critic_obs_buf[
        :, critic_offset["physical_r03_r06_r07_fact_f32"]
    ].reshape(2, 4, 32)
    assert torch.equal(facts[:, 0], env._full_a_physical_fact_f32)
    assert torch.count_nonzero(facts[:, 1:]) == 0
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
        "full_a_selected_contact",
        "full_a_ball_table_contact",
        "full_a_physical_current_center",
        "full_a_reveal_event",
        "full_a_launch_event",
        "full_a_flight_terminal_event",
        "full_a_selected_reset_event",
        "full_a_contact_eligible_event",
        "full_a_selected_contact_event",
    }
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
    assert torch.count_nonzero(extras["reward_terms"][:, :14]) == 0
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
        task_start = 0
        for name, width in P.ACTOR_LAYOUT_V1:
            if name == "epoch_task_f32":
                break
            task_start += width
        assert torch.count_nonzero(obs0["policy"][0, task_start : task_start + 45]) > 0

        launch_center = env._full_a_launch_state_f32[0, :3].clone()
        _, _, dones1, extras1 = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        centers = [extras1["full_a_physical_current_center"][0].clone()]
        terminal = bool(dones1[0])
        last = extras1
        for _ in range(80):
            if terminal:
                break
            _, _, dones, last = _assert_full_a_step_surface(
                env.step(zero), num_envs=1
            )
            centers.append(last["full_a_physical_current_center"][0].clone())
            terminal = bool(dones[0])
        assert terminal
        assert any(not torch.equal(center, launch_center) for center in centers)
        assert last["full_a_outcome_code"][0] in (
            wait_env.FULL_A_OUTCOME_FLIGHT_EXPIRED,
            wait_env.FULL_A_OUTCOME_BALL_DEAD,
        )
        assert last["full_a_phase_before_reset"][0] == wait_env.FULL_A_PHASE_OUTCOME_SETTLED
        assert env.reset_generation[0] == generation[0] + 1
        assert torch.count_nonzero(last["reward_terms"][:, :14]) == 0
    finally:
        env.close()


@pytest.mark.skipif(
    not RUN_GPU_DIRECT,
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_full_a_n1_launch_reports_only_live_racket_contact():
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
        racket_center = env.sim.data.geom_xpos[0, racket_gid].clone()
        env._full_a_launch_state_f32[0, :3] = racket_center
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
        _, _, dones, contact = _assert_full_a_step_surface(
            env.step(zero), num_envs=1
        )
        assert not bool(dones[0])
        assert bool(contact["full_a_contact_eligible_event"][0])
        assert bool(contact["full_a_selected_contact_event"][0])
        assert bool(contact["full_a_selected_contact"][0])
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
            "_full_a_physical_fact_f32",
            "_full_a_physical_present",
            "_full_a_selected_contact",
            "_full_a_ball_table_contact",
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
