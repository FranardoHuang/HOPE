"""Focused env/lean-owner chronology and fail-stop regression tests.

The production module imports IsaacLab. These tests substitute only the base
environment so they can exercise local callpoints without Kit. Formal owner
source/DAG/SHA admission was retired; this file tests the exact lean owner and
runtime facts that can actually fail.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
MODULE_PATH = (
    SOURCE
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "full_mdp_env.py"
)
for path in (str(SOURCE), str(MDP)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_subject():
    subject_name = "_action_ball_full_mdp_env_test_subject"
    if subject_name in sys.modules:
        return sys.modules[subject_name]

    class FakeManagerBasedRLEnv:
        base_constructions = 0

        def __init__(self, *args, **kwargs):
            type(self).base_constructions += 1
            raise AssertionError("focused tests must not construct a Kit env")

        def close(self):
            return None

    class FakeManagerBasedRLEnvCfg:
        pass

    stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab.envs.common": types.ModuleType("isaaclab.envs.common"),
        "isaaclab.envs.manager_based_rl_env": types.ModuleType(
            "isaaclab.envs.manager_based_rl_env"
        ),
        "isaaclab.envs.manager_based_rl_env_cfg": types.ModuleType(
            "isaaclab.envs.manager_based_rl_env_cfg"
        ),
    }
    stubs["isaaclab"].__path__ = []
    stubs["isaaclab.envs"].__path__ = []
    stubs["isaaclab.envs.common"].VecEnvStepReturn = tuple
    stubs["isaaclab.envs.manager_based_rl_env"].ManagerBasedRLEnv = (
        FakeManagerBasedRLEnv
    )
    stubs["isaaclab.envs.manager_based_rl_env_cfg"].ManagerBasedRLEnvCfg = (
        FakeManagerBasedRLEnvCfg
    )

    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(subject_name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[subject_name] = module
        spec.loader.exec_module(module)
        module._TEST_FAKE_BASE = FakeManagerBasedRLEnv
        return module
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


M = _load_subject()


def _lean_modules():
    qualified = (
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_runtime"
    )
    runtime = (
        sys.modules[qualified]
        if qualified in sys.modules
        else importlib.import_module("action_ball_full_mdp_lean_runtime")
    )
    reward_name = (
        runtime.__package__ + ".action_ball_full_mdp_lean_rewards"
        if runtime.__package__
        else "action_ball_full_mdp_lean_rewards"
    )
    rewards = importlib.import_module(reward_name)
    return runtime.epoch_v1, rewards, runtime


EPOCH, REWARDS, LEAN = _lean_modules()
CANONICAL_ENV_MODULE = "whole_body_tracking.tasks.tracking.full_mdp_env"


@pytest.fixture(autouse=True)
def _exact_stamp_namespace(monkeypatch):
    """Let the real lean owner see the source-loaded env stamp identities."""

    monkeypatch.setitem(sys.modules, CANONICAL_ENV_MODULE, M)
    monkeypatch.setattr(
        M.FullMdpPrePhysicsSubstepStamp,
        "__module__",
        CANONICAL_ENV_MODULE,
    )
    monkeypatch.setattr(
        M.FullMdpPhysicsSubstepStamp,
        "__module__",
        CANONICAL_ENV_MODULE,
    )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_name(node: ast.AST) -> str:
    return _dotted_name(node.func) if isinstance(node, ast.Call) else ""


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _one_call_line(function: ast.AST, name: str) -> int:
    matches = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def test_local_step_has_the_real_lean_callpoints_in_causal_order():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    step = _class_method(
        tree, "ActionBallFullMdpManagerBasedRLEnv", "step"
    )
    assert _one_call_line(step, "self._before_policy_step") < _one_call_line(
        step, "self.action_manager.process_action"
    )

    loop = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.For)
        and any(
            isinstance(item, ast.Call) and _call_name(item) == "range"
            for item in ast.walk(node.iter)
        )
    )
    loop_calls = (
        "self.action_manager.apply_action",
        "before_physics",
        "self.scene.write_data_to_sim",
        "self.sim.step",
        "self.recorder_manager.record_post_physics_decimation_step",
        "self.sim.render",
        "self.scene.update",
        "self._publish_post_physics_substep",
    )
    loop_lines = tuple(_one_call_line(loop, name) for name in loop_calls)
    assert loop_lines == tuple(sorted(loop_lines))

    post_calls = (
        "self.termination_manager.compute",
        "self.reward_manager.compute",
        "components.reward_graph.close_milestone_actual_reward",
        "components.epoch_owner.milestone.add_step_return",
        "self._after_reward_close",
        "self._reset_idx",
        "self.command_manager.compute",
        "after_command",
    )
    post_lines = tuple(_one_call_line(step, name) for name in post_calls)
    assert post_lines == tuple(sorted(post_lines))
    calls = {
        _call_name(node)
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
    }
    assert not any(name.endswith("add_physics_callback") for name in calls)

    forbidden_suffixes = (
        ".item",
        ".cpu",
        ".numpy",
        ".tolist",
        ".synchronize",
        ".get_rng_state",
        ".manual_seed",
        ".seed",
        ".rand",
        ".randn",
        ".randint",
    )
    for method_name in (
        "_protected_manager_state",
        "_assert_protected_manager_state_unchanged",
        "_publish_post_physics_substep",
        "step",
    ):
        method = _class_method(
            tree, "ActionBallFullMdpManagerBasedRLEnv", method_name
        )
        method_calls = {
            _call_name(node)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
        }
        assert not any(
            name.endswith(suffix)
            for name in method_calls
            for suffix in forbidden_suffixes
        )


def test_missing_lean_owner_factory_and_extension_mode_fail_pre_base(
    monkeypatch,
):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    before = M._TEST_FAKE_BASE.base_constructions
    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError,
        match="requires one post-physics owner factory",
    ):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg=None,
            full_mdp_runtime_owner_factory=None,
        )
    assert M._TEST_FAKE_BASE.base_constructions == before
    assert vars(env) == {}

    monkeypatch.setattr(
        M.builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", True, raising=False
    )
    with pytest.raises(M.FullMdpUnsupportedRuntimeError, match="extension-mode"):
        M._require_standalone_simulation_app()
    monkeypatch.setattr(M.builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", False)
    M._require_standalone_simulation_app()


class _PartialCloseSim:
    def __init__(self, trace, *, fail_on=None, reenter_on=None):
        self.trace = trace
        self.fail_on = fail_on
        self.reenter_on = reenter_on
        self.reenter_env = None

    def _record(self, name):
        self.trace.append(name)
        if self.reenter_on == name:
            self.reenter_env.close()
        if self.fail_on == name:
            raise RuntimeError(f"injected {name} failure")

    def stop(self):
        self._record("stop")

    def clear(self):
        self._record("clear")

    def clear_all_callbacks(self):
        self._record("clear_all_callbacks")

    def clear_instance(self):
        self._record("clear_instance")


def _partial_close_env(trace, *, with_sim=True):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    env._action_ball_full_mdp_base_construction_state = "entered"
    env.command_manager = object()
    env.reward_manager = object()
    # Deliberately omit termination_manager: partial construction has no
    # obligation to fabricate every upstream manager merely to close.
    env.curriculum_manager = object()
    env.action_manager = object()
    env.event_manager = object()
    env.scene = object()
    env._window = object()
    if with_sim:
        env.sim = _PartialCloseSim(trace)
    return env


def test_partial_close_guard_blocks_manager_delete_reentry_and_keeps_order(
    monkeypatch,
):
    trace = []
    env = _partial_close_env(trace)
    env._action_ball_full_mdp_live_physx_shutdown = lambda: trace.append(
        "physx_shutdown"
    )

    def reenter_and_fail_reward_delete(subject, name):
        trace.append(f"del:{name}")
        if name == "reward_manager":
            trace.append("manager_reenter")
            subject.close()
            raise RuntimeError("injected reward-manager delete failure")
        object.__delattr__(subject, name)

    monkeypatch.setattr(
        M.ActionBallFullMdpManagerBasedRLEnv,
        "__delattr__",
        reenter_and_fail_reward_delete,
    )
    with pytest.raises(RuntimeError, match="injected reward-manager"):
        env.close()

    assert trace == [
        "physx_shutdown",
        "del:command_manager",
        "del:reward_manager",
        "manager_reenter",
        "del:curriculum_manager",
        "del:action_manager",
        "del:event_manager",
        "del:scene",
        "clear_all_callbacks",
        "clear_instance",
    ]
    assert env._window is None
    assert env._is_closed is True
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_complete"
    )
    assert "_action_ball_full_mdp_partial_close_in_progress" not in vars(env)
    trace_after_first_close = list(trace)
    env.close()
    assert trace == trace_after_first_close


def test_failed_live_physx_shutdown_does_not_block_partial_sim_cleanup():
    trace = []
    env = _partial_close_env(trace)

    def fail_shutdown():
        trace.append("physx_shutdown")
        raise RuntimeError("injected live PhysX shutdown failure")

    env._action_ball_full_mdp_live_physx_shutdown = fail_shutdown
    with pytest.raises(RuntimeError, match="live PhysX shutdown"):
        env.close()
    assert trace[-2:] == ["clear_all_callbacks", "clear_instance"]
    assert trace.count("clear_all_callbacks") == 1
    assert trace.count("clear_instance") == 1
    assert env._is_closed is True
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_complete"
    )


def test_partial_close_guard_blocks_sim_callback_reentry():
    trace = []
    env = _partial_close_env(trace)
    env.sim = _PartialCloseSim(trace, reenter_on="clear_all_callbacks")
    env.sim.reenter_env = env

    env.close()
    assert trace == ["clear_all_callbacks", "clear_instance"]
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_complete"
    )
    assert env._is_closed is True
    assert "_action_ball_full_mdp_partial_close_in_progress" not in vars(env)


def test_partial_in_memory_stage_cleanup_is_ordered_and_best_effort(
    monkeypatch,
):
    trace = []
    env = _partial_close_env(trace)
    env.cfg = types.SimpleNamespace(
        sim=types.SimpleNamespace(create_stage_in_memory=True)
    )
    env.sim = _PartialCloseSim(trace, fail_on="stop")

    interface = types.SimpleNamespace(
        detach_stage=lambda: trace.append("detach_stage")
    )
    base_module = types.ModuleType("isaaclab.envs.manager_based_env")
    base_module.get_isaac_sim_version = lambda: types.SimpleNamespace(major=5)
    base_module.omni = types.SimpleNamespace(
        physx=types.SimpleNamespace(
            get_physx_simulation_interface=lambda: interface
        )
    )
    monkeypatch.setitem(
        sys.modules, "isaaclab.envs.manager_based_env", base_module
    )

    with pytest.raises(RuntimeError, match="injected stop failure"):
        env.close()
    assert trace == [
        "detach_stage",
        "stop",
        "clear",
        "clear_all_callbacks",
        "clear_instance",
    ]
    assert env._is_closed is False
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_failed"
    )
    assert env._action_ball_full_mdp_partial_close_failed_terminal_steps == (
        "stop",
    )
    trace_after_failure = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="cold process exit"):
        env.close()
    assert trace == trace_after_failure
    assert env._is_closed is False


def test_partial_close_terminal_failure_is_durable_fail_stop():
    trace = []
    env = _partial_close_env(trace)
    env.sim = _PartialCloseSim(trace, fail_on="clear_instance")
    env._action_ball_full_mdp_live_physx_shutdown = lambda: trace.append(
        "physx_shutdown"
    )

    with pytest.raises(RuntimeError, match="clear_instance failure"):
        env.close()
    assert trace == [
        "physx_shutdown",
        "clear_all_callbacks",
        "clear_instance",
    ]
    assert env._is_closed is False
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_failed"
    )
    assert env._action_ball_full_mdp_partial_close_failed_terminal_steps == (
        "clear_instance",
    )

    env._action_ball_full_mdp_live_physx_shutdown = lambda: trace.append(
        "late_shutdown"
    )
    trace_after_failure = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="cold process exit"):
        env.close()
    assert trace == trace_after_failure
    assert "_action_ball_full_mdp_live_physx_shutdown" in vars(env)
    assert env._is_closed is False


def test_constructor_preserves_original_error_identity_across_cleanup_baseexception(
    monkeypatch,
):
    trace = []
    original_error = RuntimeError("original base construction failure")

    def fail_cleanup():
        raise KeyboardInterrupt("cleanup interrupt")

    def fail_base_construction(env, *args, **kwargs):
        env.sim = _PartialCloseSim(trace)
        env._action_ball_full_mdp_live_physx_shutdown = fail_cleanup
        raise original_error

    monkeypatch.setattr(M._TEST_FAKE_BASE, "__init__", fail_base_construction)
    monkeypatch.setattr(M, "_require_owner_factory", lambda factory: object())
    for name in (
        "_require_single_action_lean_cfg",
        "_require_standalone_simulation_app",
        "_assert_runtime_uses_pinned_upstream_step",
        "_assert_runtime_uses_pinned_manager_based_env",
        "_assert_runtime_uses_pinned_local_step",
        "_assert_runtime_instance_uses_pinned_local_step",
    ):
        monkeypatch.setattr(M, name, lambda *args, **kwargs: None)

    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    with pytest.raises(RuntimeError) as caught:
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg=object(),
            full_mdp_runtime_owner_factory=object(),
        )
    assert caught.value is original_error
    assert trace == ["clear_all_callbacks", "clear_instance"]
    assert env._is_closed is True
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_complete"
    )


def test_partial_close_without_sim_is_safe_and_idempotent():
    trace = []
    env = _partial_close_env(trace, with_sim=False)
    env.close()
    env.close()
    assert trace == []
    assert env._is_closed is True
    assert env._action_ball_full_mdp_base_construction_state == (
        "partial_close_complete"
    )


def _component_values():
    return [object() for _ in range(12)]


def _components_from_values(values):
    return M.FullMdpLeanRuntimeComponents(
        epoch_owner=values[0],
        device_r05_owner=values[1],
        motion_owner=values[2],
        racket_owner=values[3],
        physical_owner=values[4],
        r03_owner=values[5],
        r06_owner=values[6],
        r07_owner=values[7],
        r07_plant_fact_adapter=values[8],
        reward_graph=values[9],
        lean_runtime_owner=values[10],
        observation_source=values[11],
    )


def test_lean_component_registry_rejects_missing_or_aliased_roles():
    values = _component_values()
    components = _components_from_values(values)
    assert components.lean_runtime_owner is values[10]

    missing = list(values)
    missing[6] = None
    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError, match="missing role"
    ):
        _components_from_values(missing)

    aliased = list(values)
    aliased[7] = aliased[6]
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="aliases"):
        _components_from_values(aliased)


class _R05:
    def __init__(self, trace):
        self.trace = trace

    def advance_action_ball_full_mdp_rows(self):
        self.trace.append(("r05",))
        return None


class _Motion:
    def __init__(self, trace, count=2):
        self.trace = trace
        self.time_left = torch.ones(count)
        self.command_counter = torch.zeros(count, dtype=torch.long)

    def install_action_ball_continuous_r07_ready_projection(self, projection):
        assert projection is not None
        self.trace.append(("motion_ready",))
        return None


class _Racket:
    def __init__(self, trace, count=2):
        self.trace = trace
        self.time_left = torch.ones(count)
        self.command_counter = torch.zeros(count, dtype=torch.long)

    def arm_action_ball_full_mdp_epoch_strike_fact(self):
        self.trace.append(("racket_arm",))
        return None

    def publish_action_ball_full_mdp_epoch_strike_fact(self, *, source_step):
        self.trace.append(("racket_publish", source_step))
        return None


class _Physical:
    def __init__(self, env, trace):
        self.env = env
        self.trace = trace
        self.fail_publish = False
        self.mutate_clock = False
        self.transport_work = True
        self.keyed_epoch_work = True

    def launch_action_epoch(self):
        self.trace.append(("physical_launch",))
        return None

    def refresh_action_epoch_host_activity(self, *, next_control_step):
        self.trace.append(("physical_refresh", next_control_step))
        return None

    def action_epoch_host_activity_verdict(self, *, control_step):
        return type("Activity", (), {
            "control_step": control_step,
            "transport_work": self.transport_work,
            "keyed_epoch_work": self.keyed_epoch_work,
        })()

    def publish_action_epoch_post_physics(self, stamp):
        self.trace.append(("physical_publish", stamp.exact_tuple()))
        if self.mutate_clock:
            self.env.common_step_counter += 1
        if self.fail_publish:
            raise ValueError("physical publish counterexample")
        return None


class _R06:
    def __init__(self, trace):
        self.trace = trace

    def close_action_ball_full_mdp_epoch_reward_rows(self):
        self.trace.append(("r06_close",))
        return None


class _R07:
    def __init__(self, trace):
        self.trace = trace
        self.ready = object()

    def publish_epoch_reward_facts(self, *, current_source_step):
        self.trace.append(
            ("r07_publish", tuple(current_source_step.shape))
        )
        return None

    def refresh_epoch_readiness_without_keyed_facts(
        self, *, current_source_step
    ):
        self.trace.append(
            ("r07_readiness_only", tuple(current_source_step.shape))
        )
        return None

    def motion_ready_projection(self):
        self.trace.append(("r07_ready",))
        return self.ready


def _install_exact_lean_graph(env, trace):
    lease = getattr(env, "_action_ball_full_mdp_runtime_lease", object())
    env._action_ball_full_mdp_runtime_lease = lease
    env._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
    epoch = EPOCH.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.ones(2, dtype=torch.int64),
    )
    graph = REWARDS.LeanActionEpochRewardGraph(epoch_owner=epoch)
    graph.configure_milestone_configured_income(
        {
            name: types.SimpleNamespace(weight=1.0)
            for name in REWARDS.MANAGER_NAMES
        },
        getattr(env, "step_dt", 0.02),
    )
    r05 = _R05(trace)
    motion = _Motion(trace)
    racket = _Racket(trace)
    physical = _Physical(env, trace)
    r06 = _R06(trace)
    r07 = _R07(trace)
    r03 = object()
    owner = LEAN.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=lease,
        epoch_owner=epoch,
        reward_graph=graph,
        r05_runtime=r05,
        motion=motion,
        racket=racket,
        physical_ball=physical,
        r06_landing_outcome=r06,
        r03_strike_fact=r03,
        r07_recovery=r07,
    )
    components = M.FullMdpLeanRuntimeComponents(
        epoch_owner=epoch,
        device_r05_owner=r05,
        motion_owner=motion,
        racket_owner=racket,
        physical_owner=physical,
        r03_owner=r03,
        r06_owner=r06,
        r07_owner=r07,
        r07_plant_fact_adapter=object(),
        reward_graph=graph,
        lean_runtime_owner=owner,
        observation_source=object(),
    )
    env._action_ball_full_mdp_components = components
    env._action_ball_full_mdp_lean_reward_graph = graph
    return owner, components, physical


def test_exact_lean_owner_binding_and_lease_identity(monkeypatch):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    env.step_dt = 0.02
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    owner, components, _physical = _install_exact_lean_graph(env, [])
    lease = env._action_ball_full_mdp_runtime_lease
    monkeypatch.setattr(
        M, "FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE", LEAN.__name__
    )

    binding = env._validate_lean_owner_install(owner, expected_lease=lease)
    assert binding.owner_type is LEAN.ActionBallFullMdpLeanRuntimeOwner
    assert binding.publish_function is vars(binding.owner_type)[
        "publish_post_physics_substep"
    ]
    assert env.action_ball_full_mdp_runtime_lease is lease
    assert components.lean_runtime_owner is owner

    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="binding or authorization"
    ):
        env._validate_lean_owner_install(owner, expected_lease=object())
    env._action_ball_full_mdp_runtime_lease = object()
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="lease identity changed"
    ):
        _ = env.action_ball_full_mdp_runtime_lease


def test_stamp_and_dispatch_enforce_exact_integer_chronology():
    stamp = M.FullMdpPhysicsSubstepStamp(
        control_step=7,
        physics_substep=2,
        physics_substeps_per_control=4,
        sim_step=43,
        event_phase=M.FullMdpPhysicsEventPhase.POST_SCENE_UPDATE,
    )
    assert stamp.exact_tuple() == (7, 2, 4, 43, 1)
    with pytest.raises(AttributeError):
        object.__setattr__(stamp, "sim_step", 44)

    dispatch = M._ControlStepDispatch(
        control_step=1, decimation=2, sim_step_before=10
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="plain integers"
    ):
        dispatch.prepare(physics_substep=True, sim_step=11)
    first = dispatch.prepare(physics_substep=0, sim_step=11)
    dispatch.commit(first)
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="final physics substep"
    ):
        dispatch.finish()
    final = dispatch.prepare(physics_substep=1, sim_step=12)
    dispatch.commit(final)
    dispatch.finish()
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="skipped, duplicated or reordered",
    ):
        dispatch.prepare(physics_substep=1, sim_step=12)


class _ActionManager:
    def __init__(self, trace):
        self.trace = trace

    def process_action(self, action):
        self.trace.append(("process", tuple(action.shape)))

    def apply_action(self):
        self.trace.append(("apply",))


class _Scene:
    def __init__(self, trace):
        self.trace = trace

    def write_data_to_sim(self):
        self.trace.append(("write",))

    def update(self, *, dt):
        self.trace.append(("update", dt))


class _Simulation:
    def __init__(self, trace, *, rendering):
        self.trace = trace
        self.rendering = rendering

    def has_gui(self):
        return self.rendering

    def has_rtx_sensors(self):
        return False

    def step(self, *, render):
        assert render is False
        self.trace.append(("sim",))

    def render(self):
        self.trace.append(("render",))


class _RecorderManager:
    active_terms = ()

    def __init__(self, trace):
        self.trace = trace

    def record_pre_step(self):
        self.trace.append(("pre_record",))

    def record_post_physics_decimation_step(self):
        self.trace.append(("post_physics_record",))


class _TerminationManager:
    _NAMES = tuple(name for name, _bit in M._FULL_MDP_TERMINATION_REASON_BITS)

    def __init__(self, trace, count):
        self.trace = trace
        self._term_dones = torch.zeros(
            (count, len(self._NAMES)), dtype=torch.bool
        )
        self._terminated = torch.zeros(count, dtype=torch.bool)
        self._time_outs = torch.zeros(count, dtype=torch.bool)

    @property
    def active_terms(self):
        return list(self._NAMES)

    @property
    def terminated(self):
        return self._terminated

    @property
    def time_outs(self):
        return self._time_outs

    def compute(self):
        self.trace.append(("termination",))
        return self._terminated | self._time_outs


class _RewardManager:
    def __init__(self, trace, graph, count, step_dt):
        self.trace = trace
        self.graph = graph
        self.count = count
        self.step_dt = step_dt
        self.fail = False

    def compute(self, *, dt):
        assert dt == self.step_dt
        self.trace.append(("reward", dt))
        if self.fail:
            raise ValueError("reward failure counterexample")
        actual = torch.zeros(self.count, dtype=torch.float32)
        for ordinal in range(REWARDS.LIFECYCLE_PAYMENT_COUNT):
            value = self.graph.pay(
                ordinal, scale=1.0 if ordinal < 10 else None
            )
            actual.add_(value * self.step_dt)
        for offset in range(len(REWARDS.COMMON_DENSE_NAMES)):
            ordinal = REWARDS.LIFECYCLE_PAYMENT_COUNT + offset
            value = self.graph.record_common_dense(
                ordinal, torch.zeros(self.count, dtype=torch.float32)
            )
            actual.add_(value * self.step_dt)
        return actual


class _ObservationManager:
    def __init__(self, trace, count):
        self.trace = trace
        self.count = count

    def compute(self, **kwargs):
        self.trace.append(("observation", tuple(sorted(kwargs.items()))))
        return {"policy": torch.zeros(self.count, 1)}


class _CommandManager:
    def __init__(self, trace, motion, racket):
        self.trace = trace
        self.terms = {"motion": motion, "racket_target": racket}

    @property
    def active_terms(self):
        return list(self.terms)

    def get_term(self, name):
        return self.terms[name]

    def compute(self, *, dt):
        self.trace.append(("command", dt))


class _EventManager:
    available_modes = ()

    def apply(self, **kwargs):
        raise AssertionError("no interval event is configured")


def _fake_env(
    *, rendering=False, decimation=3, transport_work=True, keyed_epoch_work=True
):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    count = 2
    env.device = "cpu"
    env.num_envs = count
    env.cfg = types.SimpleNamespace(
        decimation=decimation,
        sim=types.SimpleNamespace(render_interval=1),
        num_rerenders_on_reset=0,
    )
    env.physics_dt = 0.005
    env.step_dt = 0.02
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(count, dtype=torch.long)
    env._full_mdp_active_dispatch = None
    env._full_mdp_last_after_reward_close_control_step = 0
    env._full_mdp_post_physics_poison = None
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    owner, components, physical = _install_exact_lean_graph(env, trace)
    env.action_manager = _ActionManager(trace)
    env.scene = _Scene(trace)
    env.sim = _Simulation(trace, rendering=rendering)
    env.recorder_manager = _RecorderManager(trace)
    env.termination_manager = _TerminationManager(trace, count)
    env.reward_manager = _RewardManager(
        trace, components.reward_graph, count, env.step_dt
    )
    env.observation_manager = _ObservationManager(trace, count)
    env.command_manager = _CommandManager(
        trace, components.motion_owner, components.racket_owner
    )
    env.event_manager = _EventManager()
    env.extras = {}
    env._full_mdp_runtime_owner = owner
    env._full_mdp_before_policy_step = owner.before_policy_step
    env._full_mdp_before_physics_substep = owner.before_physics_substep
    env._full_mdp_post_physics_publish = owner.publish_post_physics_substep
    env._full_mdp_after_reward_close = owner.after_reward_close
    env._full_mdp_after_command_compute_before_observation = (
        owner.after_command_compute_before_observation
    )
    env._full_mdp_selected_true_reset = owner.selected_true_reset
    physical.transport_work = transport_work
    physical.keyed_epoch_work = keyed_epoch_work
    owner.after_command_compute_before_observation(0)
    trace.clear()
    return env, owner, physical, trace


def _without_render(trace):
    return [row for row in trace if row[0] != "render"]


def test_render_path_preserves_exact_lean_physics_parity_without_rng_use():
    rng_before = torch.random.get_rng_state().clone()
    plain, _plain_owner, _plain_physical, plain_trace = _fake_env(
        rendering=False
    )
    rendered, _rendered_owner, _rendered_physical, rendered_trace = _fake_env(
        rendering=True
    )
    action = torch.zeros(2, 4)
    plain_result = plain.step(action)
    rendered_result = rendered.step(action)
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert _without_render(rendered_trace) == plain_trace
    stamps = [
        row[1] for row in plain_trace if row[0] == "physical_publish"
    ]
    assert stamps == [
        (1, 0, 3, 1, 1),
        (1, 1, 3, 2, 1),
        (1, 2, 3, 3, 1),
    ]
    assert [row[0] for row in rendered_trace].count("render") == 3
    assert plain.common_step_counter == rendered.common_step_counter == 1
    assert plain._sim_step_counter == rendered._sim_step_counter == 3
    assert plain.episode_length_buf.tolist() == [1, 1]
    assert rendered.episode_length_buf.tolist() == [1, 1]
    assert len(plain_result) == len(rendered_result) == 5


def test_global_idle_without_key_skips_r03_and_all_motion_ready_work(
    monkeypatch,
):
    def reject_projection(_self):
        raise AssertionError("idle no-key projected R07 readiness to Motion")

    def reject_install(_self, _projection):
        raise AssertionError("idle no-key installed R07 readiness into Motion")

    monkeypatch.setattr(_R07, "motion_ready_projection", reject_projection)
    monkeypatch.setattr(
        _Motion,
        "install_action_ball_continuous_r07_ready_projection",
        reject_install,
    )
    env, _owner, _physical, trace = _fake_env(
        decimation=2,
        transport_work=False,
        keyed_epoch_work=False,
    )
    env.step(torch.zeros(2, 4))
    names = [row[0] for row in trace]
    assert "racket_arm" not in names
    assert "racket_publish" not in names
    assert "r07_publish" not in names
    assert names.count("r07_readiness_only") == 1
    assert "r07_ready" not in names
    assert "motion_ready" not in names
    assert names.count("physical_publish") == 2


def test_transport_idle_keyed_epoch_keeps_full_r07_publication():
    env, _owner, _physical, trace = _fake_env(
        decimation=2,
        transport_work=False,
        keyed_epoch_work=True,
    )
    env.step(torch.zeros(2, 4))
    names = [row[0] for row in trace]
    assert "racket_arm" not in names
    assert "racket_publish" not in names
    assert names.count("r07_publish") == 1
    assert "r07_readiness_only" not in names
    assert names.count("r07_ready") == 1
    assert names.count("motion_ready") == 1


def test_transport_work_without_key_keeps_r03_but_skips_r07_keyed_writes():
    env, _owner, _physical, trace = _fake_env(
        decimation=2,
        transport_work=True,
        keyed_epoch_work=False,
    )
    env.step(torch.zeros(2, 4))
    names = [row[0] for row in trace]
    assert names.count("racket_publish") == 1
    assert "r07_publish" not in names
    assert names.count("r07_readiness_only") == 1
    assert "r07_ready" not in names
    assert "motion_ready" not in names


@pytest.mark.parametrize(
    "failure, expected",
    [
        ("mutate_clock", "mutated a protected manager clock"),
        ("publish", "post-physics owner failed"),
        ("reward", "reward failure counterexample"),
    ],
)
def test_partial_step_failure_is_sticky_before_any_retry(failure, expected):
    env, owner, physical, trace = _fake_env(decimation=2)
    if failure == "mutate_clock":
        physical.mutate_clock = True
    elif failure == "publish":
        physical.fail_publish = True
    else:
        env.reward_manager.fail = True

    with pytest.raises(Exception, match=expected):
        env.step(torch.zeros(2, 4))
    assert env._full_mdp_post_physics_poison is not None
    assert not any(row[0] == "command" for row in trace)
    trace_before_retry = list(trace)
    with pytest.raises(
        M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"
    ):
        env.step(torch.zeros(2, 4))
    assert trace == trace_before_retry
    if failure == "publish":
        assert owner.poisoned is True


def _reset_event_env():
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    env.device = "cpu"
    env.num_envs = 2
    env.step_dt = 0.02
    env.cfg = types.SimpleNamespace(decimation=4)
    env.common_step_counter = 17
    env._sim_step_counter = 68
    env.episode_length_buf = torch.tensor([5, 9], dtype=torch.int64)
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    env._action_ball_full_mdp_active_reset_record = None
    env._action_ball_full_mdp_reset_callpoint_authority = None
    env._action_ball_full_mdp_reset_generation = torch.tensor(
        [4, torch.iinfo(torch.int64).max], dtype=torch.int64
    )
    env.termination_manager = _TerminationManager([], 2)
    env.termination_manager._term_dones[1, 1] = True
    owner, components, _physical = _install_exact_lean_graph(env, [])
    env._full_mdp_runtime_owner = owner
    return env, components


def test_selected_reset_projects_real_mask_clock_reason_and_overflow_once():
    env, components = _reset_event_env()
    env_ids = torch.tensor([1], dtype=torch.int64)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    event = env._mint_action_ball_full_mdp_selected_reset_event(env_ids)
    projection = env._project_action_ball_full_mdp_lean_selected_reset_event(
        event
    )

    assert type(components) is M.FullMdpLeanRuntimeComponents
    assert projection.selected_env_index.tolist() == [1]
    assert projection.selected_mask.tolist() == [False, True]
    assert projection.generation_before.tolist() == [4, 2**63 - 1]
    assert projection.generation_after.tolist() == [4, 2**63 - 1]
    assert projection.generation_overflow_fault.tolist() == [False, True]
    assert projection.terminal_reset_facts_i64[:, :2].tolist() == [
        [-1, -1],
        [17, 9],
    ]
    assert projection.terminal_reset_facts_i64[0, 2].item() == 0
    assert projection.terminal_reset_facts_i64[1, 2].item() != 0
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="stale, foreign, or replayed"
    ):
        env._project_action_ball_full_mdp_lean_selected_reset_event(event)


def test_selected_reset_requires_the_exact_authorized_tensor_identity():
    env, _components = _reset_event_env()
    authorized = torch.tensor([0], dtype=torch.int64)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        authorized, source="step_nonzero"
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="callpoint authority"
    ):
        env._mint_action_ball_full_mdp_selected_reset_event(
            authorized.clone()
        )


def test_hot_dispatch_does_not_rescan_owner_python_bindings():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpManagerBasedRLEnv"
    )
    assert all(
        not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_assert_owner_binding_current"
        )
        for node in owner.body
    )
    for method in owner.body:
        if not isinstance(method, ast.FunctionDef) or method.name not in {
            "step",
            "_before_policy_step",
            "_publish_post_physics_substep",
            "_after_reward_close",
            "_reset_idx",
        }:
            continue
        assert not any(
            isinstance(node, ast.Call)
            and _call_name(node) == "self._assert_owner_binding_current"
            for node in ast.walk(method)
        )
