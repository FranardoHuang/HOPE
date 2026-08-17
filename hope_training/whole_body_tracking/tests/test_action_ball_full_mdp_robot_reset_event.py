"""Direct plant-state tests for the fresh full-MDP robot reset event."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest


torch = pytest.importorskip("torch")


def _load_events_without_starting_kit():
    """Load the pure reset function without importing the task package/Kit."""

    class _ArticulationType:
        pass

    class _SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    def _unused_randomizer(*_args, **_kwargs):
        raise AssertionError("unrelated randomizer was called")

    modules = {
        name: ModuleType(name)
        for name in (
            "isaaclab",
            "isaaclab.utils",
            "isaaclab.utils.math",
            "isaaclab.assets",
            "isaaclab.envs",
            "isaaclab.envs.mdp",
            "isaaclab.envs.mdp.events",
            "isaaclab.managers",
        )
    }
    for name in (
        "isaaclab",
        "isaaclab.utils",
        "isaaclab.envs",
        "isaaclab.envs.mdp",
    ):
        modules[name].__path__ = []
    modules["isaaclab.utils.math"].sample_uniform = _unused_randomizer
    modules["isaaclab.assets"].Articulation = _ArticulationType
    modules["isaaclab.envs.mdp.events"]._randomize_prop_by_op = _unused_randomizer
    modules["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfg

    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "events.py"
    )
    module_name = "_test_action_ball_full_mdp_robot_reset_events_under_test"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    prior = {name: sys.modules.get(name) for name in modules}
    prior_under_test = sys.modules.get(module_name)
    try:
        sys.modules.update(modules)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        for name, value in prior.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if prior_under_test is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior_under_test
    return module


E = _load_events_without_starting_kit()


class _Articulation:
    def __init__(self):
        self.device = torch.device("cpu")
        self.data = SimpleNamespace(
            default_root_state=torch.tensor(
                [
                    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0,
                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0,
                     0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                    [7.0, 8.0, 9.0, 0.0, 0.0, 1.0, 0.0,
                     1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                ],
                dtype=torch.float32,
            ),
            default_joint_pos=torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [1.1, 1.2, 1.3, 1.4],
                    [2.1, 2.2, 2.3, 2.4],
                ],
                dtype=torch.float32,
            ),
            # Deliberately nonzero: the reset contract requires zero velocity,
            # not merely a copy of whatever a fixture calls the default.
            default_joint_vel=torch.tensor(
                [
                    [3.1, 3.2, 3.3, 3.4],
                    [4.1, 4.2, 4.3, 4.4],
                    [5.1, 5.2, 5.3, 5.4],
                ],
                dtype=torch.float32,
            ),
        )
        self.root_state = torch.arange(39, dtype=torch.float32).reshape(3, 13)
        self.joint_pos = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.joint_vel = -self.joint_pos.clone()
        self.root_writes = []
        self.joint_writes = []

    def write_root_state_to_sim(self, value, *, env_ids):
        self.root_writes.append((value.detach().clone(), env_ids))
        self.root_state[env_ids] = value

    def write_joint_state_to_sim(self, position, velocity, *, env_ids):
        self.joint_writes.append(
            (position.detach().clone(), velocity.detach().clone(), env_ids)
        )
        self.joint_pos[env_ids] = position
        self.joint_vel[env_ids] = velocity


class _Scene:
    def __init__(self, robot):
        self.robot = robot
        self.env_origins = torch.tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
            dtype=torch.float32,
        )

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


def _raw_bytes(value):
    return value.detach().contiguous().view(torch.uint8).clone()


def test_fresh_robot_reset_writes_only_selected_materialized_defaults_without_rng(
    monkeypatch,
):
    def forbid_rng(*_args, **_kwargs):
        raise AssertionError("deterministic full-MDP robot reset sampled RNG")

    monkeypatch.setattr(E.math_utils, "sample_uniform", forbid_rng)
    monkeypatch.setattr(E, "_randomize_prop_by_op", forbid_rng)
    for name in ("rand", "rand_like", "randn", "randn_like", "randint"):
        monkeypatch.setattr(torch, name, forbid_rng)

    robot = _Articulation()
    env = SimpleNamespace(scene=_Scene(robot))
    selected = torch.tensor([2, 0], dtype=torch.int64)
    peer_root_before = _raw_bytes(robot.root_state[1])
    peer_joint_pos_before = _raw_bytes(robot.joint_pos[1])
    peer_joint_vel_before = _raw_bytes(robot.joint_vel[1])

    E.reset_action_ball_full_mdp_robot_to_default(env, selected)

    expected_root = robot.data.default_root_state[selected].clone()
    expected_root[:, :3] += env.scene.env_origins[selected]
    expected_root[:, 7:] = 0.0
    expected_joint_pos = robot.data.default_joint_pos[selected]
    expected_joint_vel = torch.zeros_like(robot.data.default_joint_vel[selected])

    assert len(robot.root_writes) == 1
    assert len(robot.joint_writes) == 1
    root_value, root_ids = robot.root_writes[0]
    joint_pos_value, joint_vel_value, joint_ids = robot.joint_writes[0]
    assert root_ids is selected
    assert joint_ids is selected
    assert root_value.shape == (2, 13)
    assert joint_pos_value.shape == joint_vel_value.shape == (2, 4)
    assert torch.equal(root_value, expected_root)
    assert torch.equal(joint_pos_value, expected_joint_pos)
    assert torch.equal(joint_vel_value, expected_joint_vel)
    assert torch.equal(robot.root_state[selected], expected_root)
    assert torch.equal(robot.joint_pos[selected], expected_joint_pos)
    assert torch.equal(robot.joint_vel[selected], expected_joint_vel)

    # The unselected peer never appears in either setter call and retains its
    # exact bit pattern in all three live plant tensors.
    assert torch.equal(_raw_bytes(robot.root_state[1]), peer_root_before)
    assert torch.equal(_raw_bytes(robot.joint_pos[1]), peer_joint_pos_before)
    assert torch.equal(_raw_bytes(robot.joint_vel[1]), peer_joint_vel_before)


def test_fresh_robot_reset_rejects_implicit_all_env_selection():
    robot = _Articulation()
    env = SimpleNamespace(scene=_Scene(robot))

    with pytest.raises(ValueError, match="selected int64 env_ids"):
        E.reset_action_ball_full_mdp_robot_to_default(env, None)

    assert robot.root_writes == []
    assert robot.joint_writes == []
