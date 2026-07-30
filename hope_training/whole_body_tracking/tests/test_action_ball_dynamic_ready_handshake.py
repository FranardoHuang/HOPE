"""Focused action-specific dynamic-ready runtime handshake tests.

These tests use the existing IsaacLab CPU stubs and exercise the shipped Motion
and action-term methods.  They pin three reset invariants:

* ordered motion bytes and exact frame-0 physical state close the runtime
  binding before any training reset;
* a true reset installs normalized actor history and physical hold q_des
  coherently while raw policy-history validity remains false;
* simulator and action-manager/term state share one rollback transaction.

Run only in the Pod Isaac environment:

    python -m pytest \
      hope_training/whole_body_tracking/tests/test_action_ball_dynamic_ready_handshake.py -q
"""

from __future__ import annotations

from copy import deepcopy
import types

import pytest
import torch

from test_reward_flags_mdp import commands_mod as C
from test_reward_flags_mdp import hope_actions_mod as A


_JOINTS = 31
_SHA_A = "1" * 64
_SHA_B = "2" * 64
_PIN_SHA = "3" * 64


def _action_term(num_envs: int = 3):
    names = tuple(f"joint_{index}" for index in range(_JOINTS))
    soft = torch.stack(
        (
            torch.full((num_envs, _JOINTS), -2.0),
            torch.full((num_envs, _JOINTS), 2.0),
        ),
        dim=-1,
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=torch.zeros(num_envs, _JOINTS),
            soft_joint_pos_limits=soft,
        )
    )
    cfg = types.SimpleNamespace(
        asset_name="robot",
        scale=1.0,
        use_default_offset=False,
        clamp=True,
    )
    env = types.SimpleNamespace(
        scene={"robot": asset},
        num_envs=num_envs,
        device="cpu",
    )
    action = A.ClampedJointPositionAction(cfg, env)
    manager = types.SimpleNamespace(
        _action=torch.zeros(num_envs, _JOINTS),
        _prev_action=torch.zeros(num_envs, _JOINTS),
    )
    manager.get_term = lambda name: action if name == "joint_pos" else None
    env.action_manager = manager
    return action, manager


def test_action_install_and_restore_keep_actor_qdes_contract():
    action, manager = _action_term()
    ids = torch.tensor([0, 2])
    normalized = torch.stack(
        (torch.linspace(-0.5, 0.5, _JOINTS), torch.linspace(0.5, -0.5, _JOINTS))
    )
    hold_qdes = normalized * 0.25

    rollback = action.install_action_ball_dynamic_ready_state(
        ids, normalized, hold_qdes
    )

    assert torch.equal(manager._action[ids], normalized)
    assert torch.equal(manager._prev_action[ids], normalized)
    assert torch.equal(action.raw_actions[ids], normalized)
    assert torch.equal(action.prev_raw_actions[ids], normalized)
    assert torch.equal(action.prev_prev_raw_actions[ids], normalized)
    assert torch.equal(action.processed_actions[ids], hold_qdes)
    assert torch.equal(action.previous_processed_qdes[ids], hold_qdes)
    assert torch.equal(action.pre_clamp_qdes[ids], hold_qdes)
    assert torch.equal(action.nominal_projected_qdes[ids], hold_qdes)
    assert action._processed_qdes_valid[ids].tolist() == [True, True]
    assert action.previous_processed_qdes_valid[ids].tolist() == [True, True]
    assert action.pre_clamp_qdes_valid[ids].tolist() == [True, True]
    assert action.nominal_projected_qdes_valid[ids].tolist() == [True, True]
    assert action._raw_actions_valid[ids].tolist() == [False, False]
    assert action.raw_action_history_valid[ids].tolist() == [False, False]
    assert manager._action[1].abs().sum().item() == 0.0
    assert action.processed_actions[1].abs().sum().item() == 0.0

    action.restore_action_ball_dynamic_ready_state(ids, rollback)
    assert manager._action.abs().sum().item() == 0.0
    assert manager._prev_action.abs().sum().item() == 0.0
    assert action.raw_actions.abs().sum().item() == 0.0
    assert action.processed_actions.abs().sum().item() == 0.0
    assert not bool(action._processed_qdes_valid.any())
    assert not bool(action.previous_processed_qdes_valid.any())


def test_action_install_without_rollback_skips_snapshot_and_still_installs():
    action, manager = _action_term()
    ids = torch.tensor([0, 2])
    normalized = torch.stack(
        (torch.linspace(-0.5, 0.5, _JOINTS), torch.linspace(0.5, -0.5, _JOINTS))
    )
    hold_qdes = normalized * 0.25

    def forbidden_snapshot(self, env_ids):
        raise AssertionError("diagnostic install captured rollback state")

    action.snapshot_action_ball_dynamic_ready_state = types.MethodType(
        forbidden_snapshot, action
    )

    rollback = action.install_action_ball_dynamic_ready_state(
        ids,
        normalized,
        hold_qdes,
        capture_rollback=False,
    )

    assert rollback is None
    assert torch.equal(manager._action[ids], normalized)
    assert torch.equal(manager._prev_action[ids], normalized)
    assert torch.equal(action.raw_actions[ids], normalized)
    assert torch.equal(action.prev_raw_actions[ids], normalized)
    assert torch.equal(action.prev_prev_raw_actions[ids], normalized)
    assert torch.equal(action.processed_actions[ids], hold_qdes)
    assert torch.equal(action.previous_processed_qdes[ids], hold_qdes)
    assert torch.equal(action.pre_clamp_qdes[ids], hold_qdes)
    assert torch.equal(action.nominal_projected_qdes[ids], hold_qdes)
    assert action._processed_qdes_valid[ids].tolist() == [True, True]
    assert action.previous_processed_qdes_valid[ids].tolist() == [True, True]
    assert action.pre_clamp_qdes_valid[ids].tolist() == [True, True]
    assert action.nominal_projected_qdes_valid[ids].tolist() == [True, True]
    assert action._raw_actions_valid[ids].tolist() == [False, False]
    assert action.raw_action_history_valid[ids].tolist() == [False, False]


class _FakeRobot:
    def __init__(self, num_envs: int):
        self.data = types.SimpleNamespace(
            root_state_w=torch.zeros(num_envs, 13),
            joint_pos=torch.zeros(num_envs, _JOINTS),
            joint_vel=torch.zeros(num_envs, _JOINTS),
        )

    def write_root_state_to_sim(self, root_state, *, env_ids):
        self.data.root_state_w[env_ids] = root_state

    def write_joint_state_to_sim(
        self, joint_pos, joint_vel, *, env_ids
    ):
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel


class _FakeDynamicAction:
    def __init__(self):
        self.installed = None
        self.restored = None
        self.capture_rollback = None

    def install_action_ball_dynamic_ready_state(
        self,
        env_ids,
        normalized_action,
        hold_qdes,
        *,
        capture_rollback=True,
    ):
        self.capture_rollback = capture_rollback
        self.installed = (
            env_ids.clone(),
            normalized_action.clone(),
            hold_qdes.clone(),
        )
        if not capture_rollback:
            return None
        return {"previous": torch.tensor([17])}

    def restore_action_ball_dynamic_ready_state(self, env_ids, state):
        self.restored = (env_ids.clone(), state)


def test_motion_write_and_later_commit_rollback_include_action_state():
    num_envs = 3
    command = C.MotionCommand.__new__(C.MotionCommand)
    command._env = types.SimpleNamespace(
        scene=types.SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    )
    command.robot = _FakeRobot(num_envs)
    command.clip_id = torch.tensor([0, 1, 0])
    command.motion = types.SimpleNamespace(
        seg_start=torch.tensor([0, 1]),
        body_pos_w=torch.tensor(
            [[[0.0, 0.0, 1.0]], [[0.1, 0.0, 1.1]]]
        ),
        body_quat_w=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
        ),
        joint_pos=torch.stack(
            (torch.zeros(_JOINTS), torch.full((_JOINTS,), 0.2))
        ),
    )
    dynamic_action = _FakeDynamicAction()
    # Production configuration sets this seal before the first true reset;
    # this focused write/rollback rig bypasses that constructor path.
    command._action_ball_dynamic_ready_binding_sha256 = _PIN_SHA
    command._action_ball_dynamic_ready_action_term = dynamic_action
    command._action_ball_dynamic_ready_normalized_actor_action = torch.stack(
        (torch.full((_JOINTS,), -0.1), torch.full((_JOINTS,), 0.1))
    )
    command._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = torch.stack(
        (torch.full((_JOINTS,), -0.2), torch.full((_JOINTS,), 0.2))
    )
    ids = torch.tensor([0, 1])
    spawn = torch.tensor([[0.5, -0.2, 1.0], [0.6, 0.3, 1.1]])
    yaw_quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )

    rollback = command._write_canonical_ready_state(
        ids,
        action_ball_base_spawn_w_m=spawn,
        action_ball_base_quat_wxyz=yaw_quat,
    )
    assert set(rollback) == {
        "root_state",
        "joint_pos",
        "joint_vel",
        "action_state",
    }
    assert torch.equal(command.robot.data.root_state_w[ids, :3], spawn)
    assert torch.equal(
        dynamic_action.installed[1],
        command._action_ball_dynamic_ready_normalized_actor_action[
            command.clip_id[ids]
        ],
    )
    assert dynamic_action.capture_rollback is True

    command._restore_action_ball_sim_state(ids, rollback)
    assert command.robot.data.root_state_w[ids].abs().sum().item() == 0.0
    assert command.robot.data.joint_pos[ids].abs().sum().item() == 0.0
    assert dynamic_action.restored[1]["previous"].item() == 17


class _SnapshotForbiddenRows:
    def __getitem__(self, key):
        raise AssertionError("diagnostic ready write read simulator rollback rows")


class _DiagnosticFakeRobot:
    def __init__(self):
        self.data = types.SimpleNamespace(
            root_state_w=_SnapshotForbiddenRows(),
            joint_pos=_SnapshotForbiddenRows(),
            joint_vel=_SnapshotForbiddenRows(),
        )
        self.root_state = None
        self.joint_pos = None
        self.joint_vel = None

    def write_root_state_to_sim(self, root_state, *, env_ids):
        self.root_state = root_state.clone()

    def write_joint_state_to_sim(
        self, joint_pos, joint_vel, *, env_ids
    ):
        self.joint_pos = joint_pos.clone()
        self.joint_vel = joint_vel.clone()


def test_diagnostic_motion_write_skips_all_rollback_snapshots():
    num_envs = 3
    command = C.MotionCommand.__new__(C.MotionCommand)
    command._env = types.SimpleNamespace(
        scene=types.SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    )
    command.robot = _DiagnosticFakeRobot()
    command.clip_id = torch.tensor([0, 1, 0])
    command.motion = types.SimpleNamespace(
        seg_start=torch.tensor([0, 1]),
        body_pos_w=torch.tensor(
            [[[0.0, 0.0, 1.0]], [[0.1, 0.0, 1.1]]]
        ),
        body_quat_w=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
        ),
        joint_pos=torch.stack(
            (torch.zeros(_JOINTS), torch.full((_JOINTS,), 0.2))
        ),
    )
    command._action_ball_birth_broker = types.SimpleNamespace(
        diagnostic_fast_path=True
    )
    dynamic_action = _FakeDynamicAction()
    command._action_ball_dynamic_ready_binding_sha256 = _PIN_SHA
    command._action_ball_dynamic_ready_action_term = dynamic_action
    command._action_ball_dynamic_ready_normalized_actor_action = torch.stack(
        (torch.full((_JOINTS,), -0.1), torch.full((_JOINTS,), 0.1))
    )
    command._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = torch.stack(
        (torch.full((_JOINTS,), -0.2), torch.full((_JOINTS,), 0.2))
    )
    ids = torch.tensor([0, 1])
    spawn = torch.tensor([[0.5, -0.2, 1.0], [0.6, 0.3, 1.1]])
    yaw_quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )

    rollback = command._write_canonical_ready_state(
        ids,
        action_ball_base_spawn_w_m=spawn,
        action_ball_base_quat_wxyz=yaw_quat,
    )

    assert rollback is None
    assert torch.equal(command.robot.root_state[:, :3], spawn)
    assert torch.count_nonzero(command.robot.root_state[:, 7:]).item() == 0
    assert torch.count_nonzero(command.robot.joint_vel).item() == 0
    assert dynamic_action.capture_rollback is False
    assert torch.equal(
        dynamic_action.installed[1],
        command._action_ball_dynamic_ready_normalized_actor_action[
            command.clip_id[ids]
        ],
    )


def _binding_harness():
    command = C.MotionCommand.__new__(C.MotionCommand)
    command.cfg = types.SimpleNamespace()
    command.canonical_ready_mode = True
    command._motion_file_sha256 = (_SHA_A, _SHA_B)
    starts = torch.tensor([0, 2])
    joint_pos = torch.stack(
        (
            torch.zeros(_JOINTS),
            torch.full((_JOINTS,), 0.1),
            torch.full((_JOINTS,), 0.2),
            torch.full((_JOINTS,), 0.3),
        )
    )
    joint_vel = torch.zeros_like(joint_pos)
    body_pos = torch.tensor(
        [
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0]],
            [[0.1, 0.0, 1.1]],
            [[0.1, 0.0, 1.1]],
        ]
    )
    body_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]]] * 4
    )
    command.motion = types.SimpleNamespace(
        num_segments=2,
        seg_start=starts,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
    )
    action = types.SimpleNamespace(
        processed_actions=torch.zeros(4, _JOINTS),
        install_action_ball_dynamic_ready_state=lambda *args: None,
        restore_action_ball_dynamic_ready_state=lambda *args: None,
    )
    manager = types.SimpleNamespace(
        get_term=lambda name: action if name == "joint_pos" else None
    )
    # CommandManager is constructed before ActionManager in real Isaac Lab.
    # Keep the decoder unavailable during pre-scene binding validation.
    command._env = types.SimpleNamespace()
    command._deferred_test_action_manager = manager
    rows = []
    for slot, action_id in enumerate(("loop", "block")):
        frame = int(starts[slot])
        rows.append(
            {
                "action_id": action_id,
                "physical_ready": {
                    "root_pos_w_m": body_pos[frame, 0].tolist(),
                    "root_quat_wxyz": body_quat[frame, 0].tolist(),
                    "joint_pos_rad": joint_pos[frame].tolist(),
                    "joint_vel_radps": joint_vel[frame].tolist(),
                },
                "hold_qdes_joint_pos_rad": joint_pos[frame].tolist(),
                "normalized_actor_action": joint_pos[frame].tolist(),
                "artifact": {
                    "path": f"/artifact/{action_id}.json",
                    "sha256": _PIN_SHA,
                    "content_sha256": _PIN_SHA,
                },
                "nominal_hold_receipt": {
                    "path": f"/receipt/{action_id}.json",
                    "sha256": _PIN_SHA,
                    "content_sha256": _PIN_SHA,
                },
            }
        )
    unsigned = {
        "schema_version": 1,
        "kind": "action_ball_dynamic_ready_runtime_binding_v1",
        "action_order": ["loop", "block"],
        "motion_sha256_per_action": [_SHA_A, _SHA_B],
        "rows": rows,
    }
    binding = dict(unsigned)
    binding["binding_sha256"] = (
        C.MotionCommand._action_ball_dynamic_ready_sha256(unsigned)
    )
    command.cfg.action_ball_dynamic_ready = binding
    return command


def test_binding_closes_ordered_motion_and_exact_physical_frame_zero():
    command = _binding_harness()
    command._configure_action_ball_dynamic_ready()
    assert command._action_ball_dynamic_ready_action_order == (
        "loop",
        "block",
    )
    assert command._action_ball_dynamic_ready_hold_qdes_joint_pos_rad.shape == (
        2,
        _JOINTS,
    )
    assert command._action_ball_dynamic_ready_action_term is None
    command._env.action_manager = command._deferred_test_action_manager
    bound = command._bind_action_ball_dynamic_ready_action_term()
    assert bound is command._deferred_test_action_manager.get_term("joint_pos")

    bad = _binding_harness()
    tampered = deepcopy(bad.cfg.action_ball_dynamic_ready)
    tampered["rows"][1]["physical_ready"]["joint_pos_rad"][7] += 0.1
    unsigned = dict(tampered)
    del unsigned["binding_sha256"]
    tampered["binding_sha256"] = (
        C.MotionCommand._action_ball_dynamic_ready_sha256(unsigned)
    )
    bad.cfg.action_ball_dynamic_ready = tampered
    with pytest.raises(ValueError, match="physical frame-0 mismatch"):
        bad._configure_action_ball_dynamic_ready()


def test_missing_action_manager_fails_before_true_reset_state_write():
    command = _binding_harness()
    command._configure_action_ball_dynamic_ready()
    command._env.scene = types.SimpleNamespace(
        env_origins=torch.zeros(1, 3)
    )
    command.robot = _FakeRobot(1)
    command.clip_id = torch.zeros(1, dtype=torch.long)
    original_root = command.robot.data.root_state_w.clone()
    original_joint = command.robot.data.joint_pos.clone()

    with pytest.raises(
        RuntimeError, match="ActionManager.get_term before its first true reset"
    ):
        command._write_canonical_ready_state(
            torch.tensor([0]),
            action_ball_base_spawn_w_m=torch.tensor([[0.0, 0.0, 1.0]]),
            action_ball_base_quat_wxyz=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
        )
    assert torch.equal(command.robot.data.root_state_w, original_root)
    assert torch.equal(command.robot.data.joint_pos, original_joint)
