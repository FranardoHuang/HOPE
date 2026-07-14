"""Dependency-light tests for the Isaac lateral-perturbation runtime candidate."""

from __future__ import annotations

import importlib.util
import math
import sys
import torch
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp"


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
    return loaded["lateral_perturbation"], loaded["isaac_lateral_perturbation"]


L, IL = _load_runtime_modules()


def _yaw_quat(yaw_rad: float) -> torch.Tensor:
    return torch.tensor(
        [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)],
        dtype=torch.float32,
    )


class _RootView:
    def __init__(self, masses: torch.Tensor):
        self.masses = masses

    def get_masses(self):
        return self.masses.clone()


class _RobotData:
    def __init__(self, num_envs: int, num_bodies: int):
        self.body_quat_w = torch.zeros(num_envs, num_bodies, 4)
        self.body_quat_w[..., 0] = 1.0


class _Robot:
    def __init__(self, num_envs: int = 2):
        self.body_names = ["pelvis", "torso_link", "right_arm"]
        self.num_instances = num_envs
        self.num_bodies = len(self.body_names)
        self._external_force_b = torch.zeros(num_envs, self.num_bodies, 3)
        self._external_torque_b = torch.zeros_like(self._external_force_b)
        self.has_external_wrench = False
        self.data = _RobotData(num_envs, self.num_bodies)
        masses = torch.tensor([[10.0, 20.0, 3.0]]).repeat(num_envs, 1)
        self.root_physx_view = _RootView(masses)

    def reset(self, reset_mask: torch.Tensor):
        self._external_force_b[reset_mask] = 0.0
        self._external_torque_b[reset_mask] = 0.0


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

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def write_data_to_sim(self):
        self.write_count += 1


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

    def step(self, action):
        for yaw in self.yaws_per_substep:
            self.robot.data.body_quat_w[:, 1, :] = _yaw_quat(yaw)
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
        return (action, object(), terminated, truncated, {"sentinel": object()})


def _cfg(*, duration_steps: int = 2):
    return L.LateralPerturbationConfig(
        policy_dt_s=0.02,
        opportunity_interval_steps=duration_steps,
        pulse_duration_steps=duration_steps,
        selection_probability=1.0,
        normalized_impulse_min_mps=0.04,
        normalized_impulse_max_mps=0.04,
        # Seed 1 gives phase_offset=0 for env 0/1 at the two-step interval, so the
        # dependency-light lifecycle tests deterministically start a pulse on step zero.
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
        safe_window_remaining_steps=torch.full((robot.num_instances,), cfg.pulse_duration_steps, dtype=torch.long),
    )
    sync_calls = []
    adapter = IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: sync_calls.append(device))
    mass = adapter.read_actual_total_mass_kg()
    ledger = L.dispatch_lateral_wrench_fail_closed(
        scheduler=scheduler,
        result=result,
        total_mass_kg=mass,
        adapter=adapter,
    )
    return scheduler, result, adapter, ledger, sync_calls


def test_adapter_dispatch_binds_current_mass_and_overwrites_only_torso():
    robot = _Robot()
    robot.data.body_quat_w[:, 1, :] = _yaw_quat(math.pi / 2)
    _, result, adapter, ledger, sync_calls = _dispatch_first_step(robot)

    assert ledger.actual_total_mass_kg.tolist() == [33.0, 33.0]
    assert torch.all(ledger.commanded_world_force_y_N.abs() == 33.0)
    assert torch.all(result.active_force_mask)
    # WORLD +Y at torso yaw +90deg maps to BODY +X; every other body is exactly zero.
    torso_force = robot._external_force_b[:, 1, :]
    assert torch.allclose(torso_force.abs(), torch.tensor([[33.0, 0.0, 0.0]]).repeat(2, 1))
    assert torch.equal(robot._external_force_b[:, 0], torch.zeros(2, 3))
    assert torch.equal(robot._external_force_b[:, 2], torch.zeros(2, 3))
    assert torch.equal(robot._external_torque_b, torch.zeros_like(robot._external_torque_b))
    assert robot.has_external_wrench is True
    # Ownership guard + successful commit/readback.
    assert sync_calls == [torch.device("cpu"), torch.device("cpu")]
    assert adapter.application_backend_token is robot._external_force_b


def test_adapter_mass_mismatch_is_side_effect_free():
    robot = _Robot()
    adapter = IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: None)
    force = torch.zeros(2, 1, 3)
    force[:, 0, 1] = 10.0
    torque = torch.zeros_like(force)
    before_force = robot._external_force_b.clone()
    before_torque = robot._external_torque_b.clone()
    with pytest.raises(RuntimeError, match="post-randomization PhysX mass"):
        adapter.preflight_world_wrench_at_body_com(
            step_token=0,
            total_mass_kg=torch.ones(2),
            force_w=force,
            torque_w=torque,
            preflight_token=object(),
        )
    assert torch.equal(robot._external_force_b, before_force)
    assert torch.equal(robot._external_torque_b, before_torque)
    assert robot.has_external_wrench is False


def test_adapter_rejects_preexisting_external_wrench_owner():
    robot = _Robot()
    robot._external_force_b[0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="refuses to steal"):
        IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: None)

    robot = _Robot()
    robot.has_external_wrench = True
    with pytest.raises(RuntimeError, match="existing external-wrench owner"):
        IL.IsaacLab21LateralWrenchAdapter(robot, synchronize=lambda device: None)


def test_adapter_rejects_another_writer_between_policy_steps():
    robot = _Robot()
    _, _, adapter, _, _ = _dispatch_first_step(robot)
    robot._external_force_b[0, 0, 0] = 1.0
    force = torch.zeros(2, 1, 3)
    torque = torch.zeros_like(force)
    with pytest.raises(RuntimeError, match="another producer changed"):
        adapter.preflight_world_wrench_at_body_com(
            step_token=1,
            total_mass_kg=adapter.read_actual_total_mass_kg(),
            force_w=force,
            torque_w=torque,
            preflight_token=object(),
        )
    assert adapter.dirty_unknown is True


def test_adapter_fails_closed_if_live_torque_buffer_identity_changes():
    robot = _Robot()
    _, _, adapter, _, _ = _dispatch_first_step(robot)
    robot._external_torque_b = robot._external_torque_b.clone()
    with pytest.raises(RuntimeError, match="torque-buffer identity changed"):
        adapter.refresh_before_sim_substep(policy_step_token=0, physics_substep_index=0)
    assert adapter.dirty_unknown is True


def test_scene_write_readback_covers_non_torso_rows():
    robot = _Robot()
    _, _, adapter, _, _ = _dispatch_first_step(robot)
    row = adapter.refresh_before_sim_substep(policy_step_token=0, physics_substep_index=0)
    robot._external_force_b[0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="full-articulation force buffer"):
        adapter.confirm_scene_write_completed(row)
    assert adapter.dirty_unknown is True


def test_default_off_delegates_without_reading_or_mutating_env():
    sentinel = (object(), object(), object())

    class MinimalEnv:
        def __init__(self):
            self.calls = 0

        def step(self, action):
            self.calls += 1
            return sentinel

    env = MinimalEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=False)
    result = hook.step(object())
    assert result is sentinel
    assert env.calls == 1
    assert hook.receipts() == ()
    assert hook.consume_counters() == {}


def test_runtime_refreshes_world_force_each_substep_and_syncs_scene_write():
    env = _FakeEnv()
    sync_calls = []
    hook = IL.IsaacLateralPerturbationRuntimeHook(
        env,
        _cfg(),
        enabled=True,
        synchronize=lambda device: sync_calls.append(device),
    )
    action = torch.zeros(env.num_envs, 1)
    output = hook.step(action)
    assert output[0] is action
    rows = hook.receipts()
    assert len(rows) == 1
    row = rows[0]
    assert len(row.physics_substeps) == env.cfg.decimation
    assert all(r.scene_write_completed_synchronously for r in row.physics_substeps)
    assert all(r.buffer_readback_exact for r in row.physics_substeps)
    assert all(not r.solver_execution_readback_available for r in row.physics_substeps)
    # Ownership guard + initial commit + (write + post-scene sync) for every substep.
    assert len(sync_calls) == 2 + 2 * env.cfg.decimation

    # The body-frame command changes with yaw, while rotating each receipt back to WORLD recovers
    # the exact frozen WORLD-Y command.
    for yaw, substep in zip(env.yaws_per_substep, row.physics_substeps):
        quat = _yaw_quat(yaw).repeat(env.num_envs, 1)
        recovered_w = IL._quat_rotate_wxyz(quat, substep.written_force_b[:, 0])
        assert torch.allclose(recovered_w, substep.commanded_force_w[:, 0], atol=1e-5)


def test_strike_interruption_commits_full_zero_on_next_policy_step():
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda device: None)
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
    assert not torch.any(second.scheduler_step.active_force_mask)
    assert torch.equal(env.robot._external_force_b, torch.zeros_like(env.robot._external_force_b))
    assert torch.equal(env.robot._external_torque_b, torch.zeros_like(env.robot._external_torque_b))


def test_reset_zero_write_and_next_step_impulse_reconciliation():
    env = _FakeEnv()
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda device: None)
    action = torch.zeros(env.num_envs, 1)
    env.reset_next[0] = True
    hook.step(action)
    first = hook.receipts()[0]
    assert first.reset_after_step.tolist() == [True, False]
    assert first.reset_scene_write_observed is True
    assert first.reset_torso_buffer_zero_exact is True
    # The reset-only scene write covers the whole vectorized scene.  Non-reset rows must also be
    # cleared so they cannot receive an extra force outside the four configured physics substeps.
    assert torch.equal(env.robot._external_force_b, torch.zeros_like(env.robot._external_force_b))
    assert torch.equal(env.robot._external_torque_b, torch.zeros_like(env.robot._external_torque_b))

    # The next scheduler tick sees an exact episode-index transition and books the unfinished
    # pulse as a reset interruption for env 0.  Env 1 continues normally.
    hook.step(action)
    second = hook.receipts()[1]
    assert second.episode_indices.tolist() == [1, 0]
    assert second.episode_steps.tolist() == [0, 1]
    assert second.scheduler_step.interrupted_for_reset_mask.tolist() == [True, False]
    assert second.scheduler_step.active_force_mask.tolist() == [False, True]
    assert torch.equal(env.robot._external_force_b[0], torch.zeros_like(env.robot._external_force_b[0]))
    assert torch.any(env.robot._external_force_b[1] != 0.0)
    sampled = second.scheduler_step.reset_interrupted_sampled_impulse_y_mps
    commanded = second.scheduler_step.reset_interrupted_commanded_impulse_y_mps
    applied = second.scheduler_step.reset_interrupted_applied_impulse_y_mps
    assert torch.equal(
        sampled,
        commanded + second.scheduler_step.reset_abandoned_uncommanded_impulse_y_mps,
    )
    assert torch.equal(
        commanded,
        applied + second.scheduler_step.reset_abandoned_unapplied_impulse_y_mps,
    )


def test_runtime_refuses_event_driven_window_inference():
    env = _FakeEnv()
    env.motion.event_timing_enabled = True
    hook = IL.IsaacLateralPerturbationRuntimeHook(env, _cfg(), enabled=True, synchronize=lambda device: None)
    with pytest.raises(RuntimeError, match="event-driven T1"):
        hook.step(torch.zeros(env.num_envs, 1))


def test_contracts_pin_isaaclab21_and_do_not_claim_solver_readback():
    backend = IL.isaac_lateral_backend_contract()
    transform = IL.isaac_lateral_transform_contract()
    assert backend["isaaclab_tag"] == "v2.1.0"
    assert backend["isaaclab_commit"] == "21f7136325136ca3f6ca4e0a8125edffe5c24f7e"
    assert backend["solver_execution_readback_available"] is False
    assert transform["refresh"] == "before_every_physics_substep"
    assert len(IL.isaac_lateral_backend_identity_sha256()) == 64
    assert len(IL.isaac_lateral_transform_identity_sha256()) == 64
    assert IL.IsaacLab21LateralWrenchAdapter.commit_failure_is_terminal is True
    assert not hasattr(IL.IsaacLab21LateralWrenchAdapter, "commit_is_atomic_and_noexcept")
