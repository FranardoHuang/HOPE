"""Dependency-light red-team tests for the direct-COM Isaac runtime candidate."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
SCRIPTS = ROOT / "scripts"


def _load_runtime_modules():
    package_name = "lateral_isaac_runtime_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MDP)]
    sys.modules[package_name] = package
    loaded = {}
    for short_name in ("lateral_perturbation", "isaac_lateral_perturbation"):
        name = f"{package_name}.{short_name}"
        spec = importlib.util.spec_from_file_location(name, MDP / f"{short_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[short_name] = module
    artifact_spec = importlib.util.spec_from_file_location(
        "lateral_probe_artifacts_test_module", SCRIPTS / "lateral_probe_artifacts.py"
    )
    artifact_module = importlib.util.module_from_spec(artifact_spec)
    sys.modules[artifact_spec.name] = artifact_module
    assert artifact_spec.loader is not None
    artifact_spec.loader.exec_module(artifact_module)
    return loaded["lateral_perturbation"], loaded["isaac_lateral_perturbation"], artifact_module


L, IL, ART = _load_runtime_modules()


def _yaw_quat(yaw_rad: float) -> torch.Tensor:
    return torch.tensor(
        [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)],
        dtype=torch.float32,
    )


class _RootView:
    def __init__(self, robot: "_Robot", masses: torch.Tensor):
        self.robot = robot
        self.masses = masses
        self.coms = torch.zeros(robot.num_instances, robot.num_bodies, 7)
        self.coms[..., 6] = 1.0
        self.apply_calls: list[dict[str, object]] = []
        self.apply_exception: BaseException | None = None
        self.apply_return: object | None = None
        self.on_apply = None

    def get_masses(self):
        return self.masses.clone()

    def get_coms(self):
        return self.coms.clone()

    def apply_forces_and_torques_at_position(
        self, *, force_data, torque_data, position_data, indices, is_global
    ):
        self.apply_calls.append(
            {
                "force_data": force_data.clone(),
                "torque_data": torque_data.clone(),
                "position_data": position_data.clone() if position_data is not None else None,
                "indices": indices,
                "is_global": is_global,
            }
        )
        if self.on_apply is not None:
            self.on_apply()
        if self.apply_exception is not None:
            raise self.apply_exception
        return self.apply_return


class _RobotData:
    def __init__(self, num_envs: int, num_bodies: int):
        self.body_pos_w = torch.zeros(num_envs, num_bodies, 3)
        self.body_quat_w = torch.zeros(num_envs, num_bodies, 4)
        self.body_quat_w[..., 0] = 1.0
        self.com_pos_b = torch.zeros(num_envs, num_bodies, 3)


class _Robot:
    def __init__(self, num_envs: int = 2):
        self.body_names = ["pelvis", "torso_link", "right_arm"]
        self.num_instances = num_envs
        self.num_bodies = len(self.body_names)
        self._external_force_b = torch.zeros(num_envs, self.num_bodies, 3)
        self._external_torque_b = torch.zeros_like(self._external_force_b)
        self.has_external_wrench = False
        self._ALL_INDICES = torch.arange(num_envs, dtype=torch.long)
        self.data = _RobotData(num_envs, self.num_bodies)
        masses = torch.tensor([[10.0, 20.0, 3.0]]).repeat(num_envs, 1)
        self.root_physx_view = _RootView(self, masses)
        self.reset_competing_writer = False

    def reset(self, reset_mask: torch.Tensor):
        self._external_force_b[reset_mask] = 0.0
        self._external_torque_b[reset_mask] = 0.0
        if self.reset_competing_writer:
            self._external_force_b[~reset_mask, 2, 0] = 5.0


class _MotionAsset:
    time_step_total = 20


class _MotionTerm:
    def __init__(self, num_envs: int):
        self.in_hold = torch.ones(num_envs, dtype=torch.bool)
        self.hold_counter = torch.full((num_envs,), 10, dtype=torch.long)
        self.time_steps = torch.zeros(num_envs, dtype=torch.long)
        self.motion = _MotionAsset()
        self._multiseg = False
        self.event_timing_enabled = False


class _TargetTerm:
    def __init__(self, num_envs: int):
        self.strike_window = torch.zeros(num_envs, dtype=torch.bool)
        self.pre_strike = torch.ones(num_envs, dtype=torch.bool)


class _CommandManager:
    def __init__(self, motion: _MotionTerm, target: _TargetTerm):
        self._terms = {"motion": motion, "racket_target": target}

    def get_term(self, name: str):
        return self._terms[name]


class _Scene:
    def __init__(self, robot: _Robot):
        self.robot = robot
        self.write_count = 0
        self.competing_writer_at: int | None = None
        self.raise_at: int | None = None
        self.return_at: int | None = None
        self.refuse_restore = False

    def __delattr__(self, name: str) -> None:
        if name == "write_data_to_sim" and getattr(self, "refuse_restore", False):
            raise RuntimeError("synthetic scene hook restore failure")
        super().__delattr__(name)

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def write_data_to_sim(self):
        self.write_count += 1
        if self.competing_writer_at == self.write_count:
            self.robot._external_torque_b[0, 2, 1] = 7.0
        if self.raise_at == self.write_count:
            raise RuntimeError("synthetic scene write failure")
        if self.return_at == self.write_count:
            return object()
        return None


class _Cfg:
    decimation = 4


class _FakeEnv:
    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = 0.02
        self.cfg = _Cfg()
        self.robot = _Robot(num_envs)
        self.scene = _Scene(self.robot)
        self.motion = _MotionTerm(num_envs)
        self.target = _TargetTerm(num_envs)
        self.command_manager = _CommandManager(self.motion, self.target)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
        self.reset_next = torch.zeros(num_envs, dtype=torch.bool)
        self.yaws_per_substep = [0.0, math.pi / 2, math.pi, -math.pi / 2]
        self.wrong_output = False

    def step(self, action):
        for index, yaw in enumerate(self.yaws_per_substep):
            self.robot.data.body_quat_w[:, 1] = _yaw_quat(yaw)
            self.robot.data.body_pos_w[:, 1, 0] = float(index)
            self.scene.write_data_to_sim()
        self.episode_length_buf += 1
        terminated = self.reset_next.clone()
        truncated = torch.zeros_like(terminated)
        self.reset_buf = terminated | truncated
        if torch.any(self.reset_buf):
            self.robot.reset(self.reset_buf)
            self.episode_length_buf[self.reset_buf] = 0
            self.scene.write_data_to_sim()
        self.reset_next.zero_()
        if self.wrong_output:
            return object()
        return (action, object(), terminated, truncated, {"sentinel": object()})


def _cfg(*, duration_steps: int = 2):
    return L.LateralPerturbationConfig(
        policy_dt_s=0.02,
        opportunity_interval_steps=duration_steps,
        pulse_duration_steps=duration_steps,
        selection_probability=1.0,
        normalized_impulse_min_mps=0.04,
        normalized_impulse_max_mps=0.04,
        seed=1,
    )


def _dispatch_first_step(robot: _Robot):
    cfg = _cfg()
    scheduler = L.LateralPulseScheduler(robot.num_instances, cfg, require_application_ack=True)
    result = scheduler.step(
        step_token=0,
        episode_indices=torch.zeros(robot.num_instances, dtype=torch.long),
        episode_steps=torch.zeros(robot.num_instances, dtype=torch.long),
        recovery_hold_eligible=torch.ones(robot.num_instances, dtype=torch.bool),
        strike_window=torch.zeros(robot.num_instances, dtype=torch.bool),
        safe_window_remaining_steps=torch.full(
            (robot.num_instances,), cfg.pulse_duration_steps, dtype=torch.long
        ),
    )
    adapter = IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: None)
    ledger = L.dispatch_lateral_wrench_fail_closed(
        scheduler=scheduler,
        result=result,
        total_mass_kg=adapter.read_actual_total_mass_kg(),
        adapter=adapter,
    )
    return scheduler, result, adapter, ledger


def _reshape_call(call: dict[str, object], robot: _Robot, name: str) -> torch.Tensor:
    value = call[name]
    assert isinstance(value, torch.Tensor)
    return value.reshape(robot.num_instances, robot.num_bodies, 3)


def test_adapter_commit_is_private_and_builtin_buffer_remains_unowned():
    robot = _Robot()
    _, result, adapter, ledger = _dispatch_first_step(robot)
    assert ledger.actual_total_mass_kg.tolist() == [33.0, 33.0]
    assert torch.all(ledger.commanded_world_force_y_N.abs() == 33.0)
    assert torch.all(result.active_force_mask)
    assert robot.root_physx_view.apply_calls == []
    assert torch.equal(robot._external_force_b, torch.zeros_like(robot._external_force_b))
    assert torch.equal(robot._external_torque_b, torch.zeros_like(robot._external_torque_b))
    assert robot.has_external_wrench is False
    assert adapter.application_backend_token is robot.root_physx_view


def test_nonzero_com_offset_is_rotated_and_submitted_explicitly_in_world():
    robot = _Robot()
    robot.data.body_pos_w[:, 1] = torch.tensor([1.0, 2.0, 3.0])
    robot.data.body_quat_w[:, 1] = _yaw_quat(math.pi / 2)
    robot.data.com_pos_b[:, 1] = torch.tensor([0.1, 0.0, 0.0])
    _, _, adapter, _ = _dispatch_first_step(robot)
    receipt = adapter.apply_before_sim_substep(policy_step_token=0, physics_substep_index=0)
    call = robot.root_physx_view.apply_calls[-1]
    positions = _reshape_call(call, robot, "position_data")
    forces = _reshape_call(call, robot, "force_data")
    torques = _reshape_call(call, robot, "torque_data")
    expected_com = torch.tensor([[1.0, 2.1, 3.0], [1.0, 2.1, 3.0]])
    assert call["position_data"] is not None
    assert call["is_global"] is True
    assert call["indices"] is robot._ALL_INDICES
    assert torch.allclose(positions[:, 1], expected_com, atol=1e-6)
    assert torch.allclose(receipt.torso_com_position_w, expected_com, atol=1e-6)
    assert torch.allclose(receipt.torso_local_com_position_b, torch.tensor([[0.1, 0, 0]]).repeat(2, 1))
    assert torch.all(forces[:, 1, 0] == 0.0)
    assert torch.all(forces[:, 1, 2] == 0.0)
    assert torch.all(forces[:, 1, 1].abs() == 33.0)
    assert torch.equal(forces[:, 0], torch.zeros(2, 3))
    assert torch.equal(forces[:, 2], torch.zeros(2, 3))
    assert torch.equal(torques, torch.zeros_like(torques))


def test_derived_world_com_overflow_is_terminal_before_direct_setter():
    robot = _Robot()
    largest = torch.finfo(robot.data.body_pos_w.dtype).max
    robot.data.body_pos_w[:, 1, 0] = largest
    robot.data.com_pos_b[:, 1, 0] = largest
    _, _, adapter, _ = _dispatch_first_step(robot)

    with pytest.raises(RuntimeError, match="derived WORLD torso COM positions must be finite"):
        adapter.apply_before_sim_substep(policy_step_token=0, physics_substep_index=0)
    assert adapter.dirty_unknown is True
    assert adapter.terminal is True
    assert robot.root_physx_view.apply_calls == []


def test_adapter_mass_mismatch_is_side_effect_free():
    robot = _Robot()
    adapter = IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: None)
    force = torch.zeros(2, 1, 3)
    force[:, 0, 1] = 10.0
    with pytest.raises(RuntimeError, match="post-randomization PhysX mass"):
        adapter.preflight_world_wrench_at_body_com(
            step_token=0,
            total_mass_kg=torch.ones(2),
            force_w=force,
            torque_w=torch.zeros_like(force),
            preflight_token=object(),
        )
    assert robot.root_physx_view.apply_calls == []
    assert robot.has_external_wrench is False


@pytest.mark.parametrize("owner_kind", ["force", "torque", "flag"])
def test_adapter_rejects_preexisting_external_wrench_owner(owner_kind: str):
    robot = _Robot()
    if owner_kind == "force":
        robot._external_force_b[0, 0, 0] = 1.0
    elif owner_kind == "torque":
        robot._external_torque_b[1, 2, 0] = 1.0
    else:
        robot.has_external_wrench = True
    with pytest.raises(RuntimeError, match="refuses"):
        IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: None)


def test_substep_competing_non_torso_writer_is_terminal_and_cannot_continue():
    env = _FakeEnv()
    env.scene.competing_writer_at = 1
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    with pytest.raises(RuntimeError, match="competing/non-torso torque writer"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.dirty_unknown is True
    assert hook.terminal_zero_submit_succeeded is False
    with pytest.raises(RuntimeError, match="DIRTY/UNKNOWN"):
        hook.step(torch.zeros(env.num_envs, 1))


def test_direct_setter_same_tick_competing_writer_is_detected_and_terminal():
    env = _FakeEnv()

    def inject_competing_writer():
        env.robot._external_force_b[0, 2, 0] = 11.0

    env.robot.root_physx_view.on_apply = inject_competing_writer
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env, _cfg(), enabled=True, synchronize=lambda d: None
    )
    with pytest.raises(RuntimeError, match="competing/non-torso force writer"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.dirty_unknown is True
    assert hook.terminal_zero_submit_succeeded is False
    with pytest.raises(RuntimeError, match="DIRTY/UNKNOWN"):
        hook.step(torch.zeros(env.num_envs, 1))


def test_reset_non_torso_writer_is_detected_before_adapter_clear():
    env = _FakeEnv()
    env.robot.reset_competing_writer = True
    env.reset_next[0] = True
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    with pytest.raises(RuntimeError, match="competing/non-torso force writer"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.dirty_unknown is True
    with pytest.raises(RuntimeError, match="DIRTY/UNKNOWN"):
        hook.step(torch.zeros(env.num_envs, 1))


def test_scene_write_exception_gets_terminal_direct_zero_overwrite():
    env = _FakeEnv()
    env.scene.raise_at = 1
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    with pytest.raises(RuntimeError, match="synthetic scene write failure"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.dirty_unknown is True
    assert hook.terminal_zero_submit_succeeded is True
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
    assert torch.equal(last["torque_data"], torch.zeros_like(last["torque_data"]))


def test_wrong_scene_write_return_type_gets_terminal_zero_overwrite():
    env = _FakeEnv()
    env.scene.return_at = 1
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    with pytest.raises(RuntimeError, match="scene.write_data_to_sim returned a non-None"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.terminal_zero_submit_succeeded is True


def test_wrong_environment_step_return_type_gets_terminal_zero_overwrite():
    env = _FakeEnv()
    env.wrong_output = True
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    with pytest.raises(RuntimeError, match="terminated/truncated"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.dirty_unknown is True
    assert hook.terminal_zero_submit_succeeded is True
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))


def test_scene_hook_restore_failure_is_raised_and_zeroed_after_last_step():
    env = _FakeEnv()
    env.scene.refuse_restore = True
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env, _cfg(), enabled=True, synchronize=lambda d: None
    )
    with pytest.raises(RuntimeError, match="failed to restore scene.write_data_to_sim"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert hook.dirty_unknown is True
    assert hook.terminal_zero_submit_succeeded is True
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
    assert torch.equal(last["torque_data"], torch.zeros_like(last["torque_data"]))
    with pytest.raises(RuntimeError, match="DIRTY/UNKNOWN"):
        hook.step(torch.zeros(env.num_envs, 1))


def test_direct_setter_exception_and_wrong_return_are_terminal():
    for failure_mode in ("raise", "return"):
        env = _FakeEnv()
        if failure_mode == "raise":
            env.robot.root_physx_view.apply_exception = RuntimeError("setter failed")
            expected = "setter failed"
        else:
            env.robot.root_physx_view.apply_return = object()
            expected = "non-None"
        hook = IL.IsaacLateralPerturbationRuntimeHook(
            env, _cfg(), enabled=True, synchronize=lambda d: None
        )
        with pytest.raises(RuntimeError, match=expected):
            hook.step(torch.zeros(env.num_envs, 1))
        assert hook.dirty_unknown is True
        assert hook.terminal_zero_submit_succeeded is False
        last = env.robot.root_physx_view.apply_calls[-1]
        assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
        assert torch.equal(last["torque_data"], torch.zeros_like(last["torque_data"]))
        with pytest.raises(RuntimeError, match="DIRTY/UNKNOWN"):
            hook.step(torch.zeros(env.num_envs, 1))


def test_default_off_delegates_without_reading_or_mutating_env():
    sentinel = (object(), object(), object())

    class MinimalEnv:
        calls = 0

        def step(self, action):
            self.calls += 1
            return sentinel

    env = MinimalEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=False)
    assert hook.step(object()) is sentinel
    assert env.calls == 1
    assert hook.receipts() == ()
    assert hook.consume_counters() == {}


def test_runtime_submits_explicit_current_com_on_every_physics_substep():
    env = _FakeEnv()
    env.robot.data.com_pos_b[:, 1, 0] = 0.2
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env, _cfg(), enabled=True, synchronize=lambda device: None
    )
    action = torch.zeros(env.num_envs, 1)
    output = hook.step(action)
    assert output[0] is action
    row = hook.receipts()[0]
    assert len(row.physics_substeps) == env.cfg.decimation
    assert len(env.robot.root_physx_view.apply_calls) == env.cfg.decimation
    for index, (yaw, substep) in enumerate(zip(env.yaws_per_substep, row.physics_substeps)):
        expected = torch.tensor([float(index), 0.0, 0.0]) + IL._quat_rotate_wxyz(
            _yaw_quat(yaw), torch.tensor([0.2, 0.0, 0.0])
        )
        assert torch.allclose(substep.torso_com_position_w, expected.repeat(env.num_envs, 1), atol=1e-6)
        assert substep.direct_physx_call_completed_synchronously
        assert substep.scene_write_completed_synchronously
        assert substep.private_command_readback_exact
        assert substep.built_in_wrench_buffers_zero_exact
        assert not substep.solver_execution_readback_available
    assert torch.equal(env.robot._external_force_b, torch.zeros_like(env.robot._external_force_b))
    assert env.robot.has_external_wrench is False


def test_strike_interruption_submits_full_zero_on_next_policy_step():
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    action = torch.zeros(env.num_envs, 1)
    hook.step(action)
    env.motion.in_hold.zero_()
    env.motion.hold_counter.zero_()
    env.target.pre_strike.zero_()
    env.target.strike_window.fill_(True)
    hook.step(action)
    first, second = hook.receipts()
    assert torch.all(first.scheduler_step.active_force_mask)
    assert torch.all(second.scheduler_step.interrupted_for_strike_mask)
    assert all(torch.all(row.submitted_force_w_full == 0.0) for row in second.physics_substeps)
    assert all(torch.all(row.submitted_torque_w_full == 0.0) for row in second.physics_substeps)


def test_reset_clear_submits_zero_and_reconciles_next_episode():
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    action = torch.zeros(env.num_envs, 1)
    env.reset_next[0] = True
    hook.step(action)
    first = hook.receipts()[0]
    assert first.reset_after_step.tolist() == [True, False]
    assert first.reset_scene_write_observed is True
    assert first.reset_live_wrench_zero_exact is True
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
    hook.step(action)
    second = hook.receipts()[1]
    assert second.episode_indices.tolist() == [1, 0]
    assert second.episode_steps.tolist() == [0, 1]
    assert second.scheduler_step.interrupted_for_reset_mask.tolist() == [True, False]


def test_runtime_refuses_event_driven_window_inference_without_live_command():
    env = _FakeEnv()
    env.motion.event_timing_enabled = True
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    with pytest.raises(RuntimeError, match="event-driven T1"):
        hook.step(torch.zeros(env.num_envs, 1))
    assert env.robot.root_physx_view.apply_calls == []


def test_next_step_validation_failure_zeros_the_previous_live_command():
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda d: None)
    action = torch.zeros(env.num_envs, 1)
    hook.step(action)
    env.motion.event_timing_enabled = True
    with pytest.raises(RuntimeError, match="event-driven T1"):
        hook.step(action)
    assert hook.terminal_zero_submit_succeeded is True
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))


def test_clean_rollout_termination_zeroes_and_blocks_future_steps():
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env, _cfg(), enabled=True, synchronize=lambda d: None
    )
    action = torch.zeros(env.num_envs, 1)
    hook.step(action)
    assert hook.terminate_lateral_wrench_noexcept() is True
    assert hook.terminate_lateral_wrench_noexcept() is True
    assert hook.terminal is True
    assert hook.dirty_unknown is False
    assert hook.terminal_zero_submit_succeeded is True
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
    assert torch.equal(last["torque_data"], torch.zeros_like(last["torque_data"]))
    with pytest.raises(RuntimeError, match="terminal"):
        hook.step(action)


def test_clean_rollout_terminal_zero_failure_blocks_publication(tmp_path: Path):
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env, _cfg(), enabled=True, synchronize=lambda d: None
    )
    hook.step(torch.zeros(env.num_envs, 1))
    env.robot.root_physx_view.apply_return = object()

    assert hook.terminate_lateral_wrench_noexcept() is False
    assert hook.terminal is True
    assert hook.dirty_unknown is True
    assert hook.terminal_zero_submit_succeeded is False
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
    assert torch.equal(last["torque_data"], torch.zeros_like(last["torque_data"]))

    target = tmp_path / "must-not-exist.json"
    if hook.terminate_lateral_wrench_noexcept() and not hook.dirty_unknown:
        target.write_bytes(b"forbidden success\n")
    assert not target.exists()


def test_output_publication_failure_occurs_only_after_clean_terminal_zero(tmp_path: Path):
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env, _cfg(), enabled=True, synchronize=lambda d: None
    )
    hook.step(torch.zeros(env.num_envs, 1))
    assert hook.terminate_lateral_wrench_noexcept() is True

    target = tmp_path / "receipt.json"
    guard = ART.StableOutputDirectory.open(str(target))
    try:
        target.write_text("competitor\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="clobber"):
            guard.write_no_clobber(b"{}\n")
    finally:
        guard.close()

    assert hook.dirty_unknown is False
    last = env.robot.root_physx_view.apply_calls[-1]
    assert torch.equal(last["force_data"], torch.zeros_like(last["force_data"]))
    assert torch.equal(last["torque_data"], torch.zeros_like(last["torque_data"]))


def test_motion_path_swap_is_detected_even_with_identical_bytes(tmp_path: Path):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"same bytes")
    handle = ART.StableInputFile.open(str(motion), label="motion")
    try:
        old = tmp_path / "old-motion.npz"
        motion.rename(old)
        motion.write_bytes(b"same bytes")
        with pytest.raises(RuntimeError, match="identity changed"):
            handle.verify_path_unchanged()
    finally:
        handle.close()


def test_motion_runtime_path_reads_the_stable_inode_after_public_path_swap(tmp_path: Path):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"reviewed motion")
    handle = ART.StableInputFile.open(str(motion), label="motion")
    try:
        runtime_path = Path(handle.runtime_path())
        motion.unlink()
        motion.write_bytes(b"attacker motion")
        assert runtime_path.read_bytes() == b"reviewed motion"
        with pytest.raises(RuntimeError, match="identity changed"):
            handle.verify_path_unchanged()
    finally:
        handle.close()


def test_output_parent_symlink_swap_is_detected_before_openat(tmp_path: Path):
    parent = tmp_path / "receipt-parent"
    attacker = tmp_path / "attacker"
    parent.mkdir()
    attacker.mkdir()
    target = parent / "receipt.json"
    guard = ART.StableOutputDirectory.open(str(target))
    try:
        stable_parent = tmp_path / "stable-parent"
        parent.rename(stable_parent)
        os.symlink(attacker, parent)
        with pytest.raises(RuntimeError, match="symlink"):
            guard.write_no_clobber(b"{}\n")
        assert not (attacker / "receipt.json").exists()
        assert not (stable_parent / "receipt.json").exists()
    finally:
        guard.close()


def test_stable_output_dirfd_writes_once_and_never_clobbers(tmp_path: Path):
    target = tmp_path / "receipt.json"
    guard = ART.StableOutputDirectory.open(str(target))
    try:
        guard.write_no_clobber(b"first\n")
        assert target.read_bytes() == b"first\n"
        with pytest.raises(RuntimeError, match="clobber"):
            guard.write_no_clobber(b"second\n")
        assert target.read_bytes() == b"first\n"
    finally:
        guard.close()


def test_contract_pins_explicit_com_and_keeps_launch_boundary_closed():
    backend = IL.isaac_lateral_backend_contract()
    transform = IL.isaac_lateral_transform_contract()
    assert backend["isaaclab_tag"] == "v2.1.0"
    assert backend["isaaclab_commit"] == "21f7136325136ca3f6ca4e0a8125edffe5c24f7e"
    assert backend["position_data"] == "explicit_current_torso_com_world"
    assert backend["position_data_none_forbidden"] is True
    assert backend["is_global"] is True
    assert backend["solver_execution_readback_available"] is False
    assert transform["refresh"] == "before_every_physics_substep"
    assert transform["local_com"] == "ArticulationData.com_pos_b_device_normalized"
    assert len(IL.isaac_lateral_backend_identity_sha256()) == 64
    assert len(IL.isaac_lateral_transform_identity_sha256()) == 64
    assert IL.IsaacLab21LateralWrenchAdapter.commit_failure_is_terminal is True
    assert not hasattr(IL.IsaacLab21LateralWrenchAdapter, "commit_is_atomic_and_noexcept")
