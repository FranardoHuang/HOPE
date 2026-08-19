"""Focused counterexamples for the one MuJoCo action/order boundary."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).parents[1] / "a3_train_ppo.py"
WAIT_SOURCE = SOURCE.parent / "mujoco_gpu_ac_full_mdp_initial_wait_env.py"
REPO = Path(__file__).parents[4]
CONTRACT_PATH = REPO / "configs/a3_joint_order_bijection_v1.json"
READY_ARTIFACT = (
    REPO
    / "configs/action_ball_n1_measured_20260803"
    / "evidence_holdpass_robust20n_20260803"
    / "take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json"
)
sys.path.insert(0, str(SOURCE.parent))
import a3_train_ppo as train  # noqa: E402
import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env  # noqa: E402


def _method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "A3ReadyBallVecEnv"
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _contains_name(node, name):
    return any(
        isinstance(item, ast.Name) and item.id == name for item in ast.walk(node)
    )


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _order(path):
    return [
        row.strip()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]


def _source_and_target():
    contract = _contract()
    return (
        contract,
        _order(REPO / contract["source_order"]["path"]),
        _order(REPO / contract["target_order"]["path"]),
    )


def _fake_model_and_mujoco(monkeypatch, actuator_joint_names, prefix="robot/"):
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_JOINT=1),
        mj_id2name=lambda _model, _kind, joint_id: (
            prefix + actuator_joint_names[joint_id]
        ),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    return SimpleNamespace(
        nu=len(actuator_joint_names),
        actuator_trnid=np.column_stack(
            (
                np.arange(len(actuator_joint_names), dtype=np.int64),
                np.zeros(len(actuator_joint_names), dtype=np.int64),
            )
        ),
    )


class _FakeSim:
    def __init__(self, n, width):
        self.data = SimpleNamespace(
            qpos=torch.zeros(n, width),
            qvel=torch.zeros(n, width),
            ctrl=torch.zeros(n, width),
        )
        self.step_calls = 0
        self.reset_calls = []

    def step(self):
        self.step_calls += 1

    def reset(self, ids):
        self.reset_calls.append(ids.detach().clone())


def _plant_env(n, width, actuator_from_runtime=None):
    env = train.A3ReadyBallVecEnv.__new__(train.A3ReadyBallVecEnv)
    env._torch = torch
    env.device = torch.device("cpu")
    env.num_envs = n
    env.num_actions = width
    env.q_ready = torch.zeros(width)
    env.action_offset = torch.zeros(width)
    env.act_scale = torch.ones(width)
    env.jnt_lo = torch.full((width,), -10.0)
    env.jnt_hi = torch.full((width,), 10.0)
    env.actions = torch.zeros(n, width)
    env.last_actions = torch.zeros_like(env.actions)
    env.action_nonfinite_buf = torch.zeros(n, dtype=torch.bool)
    env.kp = torch.ones(width)
    env.kd = torch.zeros(width)
    env.tau_lo = torch.full((width,), -100.0)
    env.tau_hi = torch.full((width,), 100.0)
    env.tau_scale = torch.full((width,), 100.0)
    env.actuator_from_runtime = torch.as_tensor(
        np.arange(width) if actuator_from_runtime is None
        else actuator_from_runtime,
        dtype=torch.long,
    )
    env.decimation = 1
    env.sim = _FakeSim(n, width)
    env._qpos_act = lambda: env.sim.data.qpos
    env._qvel_act = lambda: env.sim.data.qvel
    env._after_physics_substep = lambda _index: None
    env._contact_ok = False
    env._cap_ok = False
    env._state = lambda: {"sentinel": True}
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env.ball_age_buf = torch.zeros(n, dtype=torch.long)
    env.common_step_counter = 0
    return env


def test_runtime_decoder_has_one_unclipped_ctrl_scatter_owner():
    advance = _method("_advance_plant")
    assignments = {
        _dotted(node.targets[0]): node.value
        for node in advance.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    assert not _contains_name(advance, "action_clip")
    assert _contains_name(assignments["pre_clamp_qdes"], "incoming")
    assert any(
        isinstance(node, ast.Attribute)
        and _dotted(node) == "self.action_offset"
        for node in ast.walk(assignments["pre_clamp_qdes"])
    )
    assert not any(
        isinstance(node, ast.Attribute) and _dotted(node) == "self.q_ready"
        for node in ast.walk(assignments["pre_clamp_qdes"])
    )
    safe = assignments["safe_actions"]
    assert isinstance(safe, ast.Call) and _dotted(safe.func) == "torch.where"
    assert [_dotted(arg) for arg in safe.args] == [
        "finite_qdes",
        "incoming",
        "self.actions",
    ]
    assert _contains_name(assignments["q_des"], "safe_actions")
    assert not _contains_name(assignments["q_des"], "incoming")
    assert any(
        isinstance(node, ast.Attribute)
        and _dotted(node) == "self.action_offset"
        for node in ast.walk(assignments["q_des"])
    )
    ctrl_writes = [
        node
        for node in ast.walk(advance)
        if isinstance(node, (ast.Assign, ast.AugAssign))
        and any(
            isinstance(item, ast.Attribute) and item.attr == "ctrl"
            for item in ast.walk(
                node.targets[0] if isinstance(node, ast.Assign) else node.target
            )
        )
    ]
    assert len(ctrl_writes) == 1
    assert any(
        isinstance(node, ast.Attribute)
        and _dotted(node) == "self.actuator_from_runtime"
        for node in ast.walk(ctrl_writes[0])
    )
    assert any(
        isinstance(node, ast.Attribute)
        and _dotted(node) == "self.action_nonfinite_buf"
        for node in ast.walk(_method("_terminate"))
    )


def test_wait_and_full_a_delegate_raw_actions_to_the_base_decoder():
    tree = ast.parse(WAIT_SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FullMdpInitialWaitVecEnv"
    )
    for method_name in ("step", "_step_full_a"):
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and _dotted(node.func) == "self._advance_plant"
        ]
        assert len(calls) == 1
        assert len(calls[0].args) == 1
        assert isinstance(calls[0].args[0], ast.Name)
        assert calls[0].args[0].id == "actions"
        assert not _contains_name(method, "safe_actions")
        assert not _contains_name(method, "action_clip")


def test_contract_and_all_31_one_hot_name_columns_are_exact(monkeypatch):
    contract, source, target = _source_and_target()
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        train.ACTION_JOINT_ORDER_CONTRACT_SHA256
    )
    model = _fake_model_and_mujoco(monkeypatch, source)
    runtime_from_actuator, actuator_from_runtime, live_target = (
        train._runtime_action_wiring(model, "robot/")
    )
    np.testing.assert_array_equal(
        runtime_from_actuator, contract["target_from_source_indices"]
    )
    np.testing.assert_array_equal(
        actuator_from_runtime, contract["source_from_target_indices"]
    )
    assert not runtime_from_actuator.flags.writeable
    assert not actuator_from_runtime.flags.writeable
    assert tuple(target) == live_target

    for runtime_index, actuator_index in enumerate(runtime_from_actuator):
        assert target[runtime_index] == source[int(actuator_index)]
        runtime_tau = np.zeros(31)
        runtime_tau[runtime_index] = 1.0
        actuator_tau = runtime_tau[actuator_from_runtime]
        assert np.flatnonzero(actuator_tau).tolist() == [int(actuator_index)]


def test_runtime_action_wiring_rejects_wrong_contract_bytes(
    monkeypatch, tmp_path
):
    _contract_value, source, _target = _source_and_target()
    altered = tmp_path / "a3_joint_order_bijection_v1.json"
    altered.write_bytes(CONTRACT_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(train, "_ACTION_JOINT_ORDER_CONTRACT", altered)
    model = _fake_model_and_mujoco(monkeypatch, source)
    with pytest.raises(RuntimeError, match="contract bytes differ"):
        train._runtime_action_wiring(model, "robot/")


def test_runtime_action_wiring_rejects_wrong_live_actuator_order(monkeypatch):
    _contract_value, source, _target = _source_and_target()
    wrong = list(source)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    model = _fake_model_and_mujoco(monkeypatch, wrong)
    with pytest.raises(RuntimeError, match="actuator order differs"):
        train._runtime_action_wiring(model, "robot/")


def test_runtime_action_offset_is_the_pinned_runtime_plant_field():
    _contract_value, _source, target = _source_and_target()
    payload = READY_ARTIFACT.read_bytes()
    offset, digest = train._runtime_action_offset(payload, target)
    document = json.loads(payload)

    np.testing.assert_array_equal(
        offset,
        np.asarray(document["runtime_plant"]["default_joint_pos_rad"], np.float32),
    )
    assert digest == train.ACTION_OFFSET_FLOAT32_SHA256
    assert hashlib.sha256(offset.astype("<f4").tobytes()).hexdigest() == digest
    ready = np.asarray(document["physical_ready"]["joint_pos_rad"], np.float32)
    assert np.count_nonzero(ready != offset) == 14
    assert np.max(np.abs(ready - offset)) > 1.5


@pytest.mark.parametrize("mutation", ("missing", "wrong_value", "wrong_order"))
def test_runtime_action_offset_rejects_missing_or_wrong_authority(mutation):
    _contract_value, _source, target = _source_and_target()
    document = json.loads(READY_ARTIFACT.read_bytes())
    if mutation == "missing":
        del document["runtime_plant"]["default_joint_pos_rad"]
        match = "lacks runtime_plant.default_joint_pos_rad"
    elif mutation == "wrong_value":
        document["runtime_plant"]["default_joint_pos_rad"][0] += 0.01
        match = "runtime action offset differs"
    else:
        names = document["runtime_plant"]["joint_names"]
        names[0], names[1] = names[1], names[0]
        match = "action-offset ABI differs"
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    with pytest.raises(RuntimeError, match=match):
        train._runtime_action_offset(payload, target)


def test_action_contract_identity_is_one_read_only_copy():
    base = _plant_env(1, 31)
    env = wait_env.FullMdpInitialWaitVecEnv.__new__(
        wait_env.FullMdpInitialWaitVecEnv
    )
    vars(env).update(vars(base))
    env._action_offset_sha256 = train.ACTION_OFFSET_FLOAT32_SHA256
    expected = {
        "action_joint_order_contract_id": train.ACTION_JOINT_ORDER_CONTRACT_ID,
        "action_joint_order_contract_sha256": (
            train.ACTION_JOINT_ORDER_CONTRACT_SHA256
        ),
        "action_offset_source": "runtime_plant.default_joint_pos_rad",
        "action_offset_sha256": train.ACTION_OFFSET_FLOAT32_SHA256,
        "full_a_reset_joint_source": "runtime_plant.default_joint_pos_rad",
        "full_a_reset_root_source": "AGIBOT_A3_CFG.init_state.pos/rot",
        "full_a_policy_bootstrap": "a3_default_stand_zero_head_v1",
        "raw_action_clip": None,
        "executable_qdes_guard": "mujoco_hard_range_only_divergent_declared",
        "transfer_authority": False,
        "matched_cross_backend_authority": False,
    }
    env.full_a_mode = True
    identity = env.action_contract_identity
    assert identity == expected
    identity["transfer_authority"] = True
    assert env.action_contract_identity == expected


def test_raw_zero_uses_action_offset_while_reset_stays_physical_ready():
    _contract_value, _source, target = _source_and_target()
    document = json.loads(READY_ARTIFACT.read_bytes())
    offset, _digest = train._runtime_action_offset(
        READY_ARTIFACT.read_bytes(), target
    )
    ready = torch.tensor(document["physical_ready"]["joint_pos_rad"])
    env = _plant_env(1, 31)
    env.q_ready = ready.clone()
    env.action_offset = torch.from_numpy(offset.copy())
    env.qpos_init = ready.clone()
    env.qvel_init = torch.zeros(31)
    env.root_qadr = 0
    env.cfg = SimpleNamespace(
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    env._serve = lambda _ids: None
    env._cur_ret = torch.ones(1)
    env._cur_min_d = torch.zeros(1)

    train.A3ReadyBallVecEnv._reset_idx(env, torch.tensor([0]))
    torch.testing.assert_close(env.sim.data.qpos[0], ready)
    _state, _tau_sq, requested = train.A3ReadyBallVecEnv._advance_plant(
        env, torch.zeros(1, 31)
    )

    torch.testing.assert_close(requested[0], env.action_offset)
    assert not torch.equal(requested[0], env.q_ready)


def test_full_a_reset_uses_default_joint_and_root_birth_not_take061():
    env = wait_env.FullMdpInitialWaitVecEnv.__new__(
        wait_env.FullMdpInitialWaitVecEnv
    )
    env._torch = torch
    env.device = torch.device("cpu")
    env.num_envs, env.num_actions = 2, 31
    env.full_a_mode = True
    env.root_qadr = 0
    env.q_adr_act = torch.arange(7, 38, dtype=torch.long)
    env.v_adr_act = torch.arange(6, 37, dtype=torch.long)
    env.qpos_init = torch.full((40,), 9.0)
    env.qpos_init[3] = 1.0
    env.qvel_init = torch.zeros(40)
    env.q_ready = torch.full((31,), 3.0)
    env.action_offset = torch.linspace(-0.4, 0.4, 31)
    env.env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.zeros((2, 3)))
    )
    env.jnt_lo = torch.full((31,), -10.0)
    env.jnt_hi = torch.full((31,), 10.0)
    env.cfg = SimpleNamespace(
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    env.sim = _FakeSim(2, 40)
    env._serve = lambda _ids: None
    env.episode_length_buf = torch.full((2,), 5, dtype=torch.long)
    env.actions = torch.ones((2, 31))
    env.last_actions = -torch.ones((2, 31))
    env.action_nonfinite_buf = torch.ones(2, dtype=torch.bool)
    env._cur_ret = torch.ones(2)
    env._cur_min_d = torch.zeros(2)
    peer_qpos = env.sim.data.qpos[1].clone()

    env._reset_idx(torch.tensor([0]))

    torch.testing.assert_close(
        env.sim.data.qpos[0, :3],
        torch.tensor(wait_env.FULL_A_DEFAULT_ROOT_POS),
    )
    torch.testing.assert_close(
        env.sim.data.qpos[0, 3:7],
        torch.tensor(wait_env.FULL_A_DEFAULT_ROOT_QUAT_WXYZ),
    )
    torch.testing.assert_close(
        env.sim.data.qpos[0, env.q_adr_act], env.action_offset
    )
    assert not torch.equal(env.sim.data.qpos[0, env.q_adr_act], env.q_ready)
    assert not env.actions[0].any() and not env.last_actions[0].any()
    assert torch.equal(env.sim.data.qpos[1], peer_qpos)


def test_real_unbound_advance_plant_scatter_matches_all_31_names():
    contract, _source, _target = _source_and_target()
    actuator_from_runtime = np.asarray(contract["source_from_target_indices"])
    env = _plant_env(31, 31, actuator_from_runtime)
    incoming = torch.eye(31)

    state, _tau_sq, requested = train.A3ReadyBallVecEnv._advance_plant(
        env, incoming
    )

    torch.testing.assert_close(requested, incoming)
    torch.testing.assert_close(env.actions, incoming)
    torch.testing.assert_close(env.last_actions, torch.zeros_like(incoming))
    torch.testing.assert_close(
        env.sim.data.ctrl, incoming[:, torch.as_tensor(actuator_from_runtime)]
    )
    assert state == {"sentinel": True}
    assert env.sim.step_calls == 1


def test_nonfinite_fallback_and_selected_reset_leave_peer_unchanged():
    env = _plant_env(2, 2)
    env.q_ready = torch.tensor([0.0, 0.2])
    env.action_offset = torch.tensor([-0.4, 0.5])
    env.act_scale = torch.tensor([0.25, 0.5])
    env.jnt_lo = torch.tensor([-1.0, -1.0])
    env.jnt_hi = -env.jnt_lo
    previous = torch.tensor([[3.0, -2.0], [1.0, -1.0]])
    env.actions = previous.clone()
    incoming = torch.tensor(
        [[20.0, -20.0], [float("nan"), float("inf")]]
    )

    _state, _tau_sq, requested = train.A3ReadyBallVecEnv._advance_plant(
        env, incoming
    )

    torch.testing.assert_close(
        env.sim.data.ctrl, torch.tensor([[1.0, -1.0], [-0.15, 0.0]])
    )
    assert torch.isfinite(env.sim.data.ctrl).all()
    assert not torch.isfinite(requested[1]).all()
    assert env.action_nonfinite_buf.tolist() == [False, True]
    torch.testing.assert_close(env.actions[0], incoming[0])
    torch.testing.assert_close(env.actions[1], previous[1])
    torch.testing.assert_close(env.last_actions, previous)

    env.cfg = SimpleNamespace(
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    env.qpos_init = torch.zeros(2)
    env.qvel_init = torch.zeros(2)
    env.root_qadr = 0
    env._serve = lambda _ids: None
    env._cur_ret = torch.ones(2)
    env._cur_min_d = torch.zeros(2)
    env.action_nonfinite_buf[0] = True
    peer_action = env.actions[0].clone()
    peer_last_action = env.last_actions[0].clone()

    train.A3ReadyBallVecEnv._reset_idx(env, torch.tensor([1]))

    assert env.action_nonfinite_buf.tolist() == [True, False]
    assert not env.actions[1].any() and not env.last_actions[1].any()
    torch.testing.assert_close(env.actions[0], peer_action)
    torch.testing.assert_close(env.last_actions[0], peer_last_action)
    assert len(env.sim.reset_calls) == 1
    assert env.sim.reset_calls[0].tolist() == [1]
