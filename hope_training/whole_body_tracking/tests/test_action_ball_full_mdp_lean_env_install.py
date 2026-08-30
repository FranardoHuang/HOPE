"""Focused tests for the one-shot diagnostic lean environment install."""

from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from full_mdp_env_canonical_harness import (
    load_canonical_full_mdp_env,
    probe_canonical_full_mdp_env_subprocess,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "full_mdp_env.py"
)


def _test_namespace_snapshot() -> dict[str, object]:
    return {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "whole_body_tracking"
        or name.startswith("whole_body_tracking.")
        or name == "isaaclab"
        or name.startswith("isaaclab.")
    }


def _clear_test_namespace() -> None:
    for name in tuple(sys.modules):
        if (
            name == "whole_body_tracking"
            or name.startswith("whole_body_tracking.")
            or name == "isaaclab"
            or name.startswith("isaaclab.")
        ):
            sys.modules.pop(name, None)


_PRIOR_TEST_NAMESPACE = _test_namespace_snapshot()
M = load_canonical_full_mdp_env(MODULE_PATH, retain_namespace=True)
_CANONICAL_TEST_NAMESPACE = _test_namespace_snapshot()
_clear_test_namespace()
sys.modules.update(_PRIOR_TEST_NAMESPACE)


def test_cold_import_uses_launcher_canonical_namespace_without_ambient_kit():
    probe_canonical_full_mdp_env_subprocess(MODULE_PATH)


class _ObservationSource:
    def __init__(self, env, runtime_owner, epoch_owner):
        self._env = env
        self._runtime_owner = runtime_owner
        self._epoch_owner = epoch_owner

    def observe(self, group):
        current = getattr(self._epoch_owner, "current", None)
        if callable(current):
            return current().reset_generation.clone()
        return group


_OBSERVATION_MODULE = types.ModuleType(
    "whole_body_tracking.tasks.tracking.mdp."
    "action_ball_full_mdp_lean_observation_cfg"
)
_OBSERVATION_MODULE.LeanActionEpochObservationSource = _ObservationSource


class _RewardGraph:
    def __init__(self, epoch_owner):
        self.epoch_owner = epoch_owner
        self.calls = []

    def configure_milestone_configured_income(self, manager_cfg, step_dt):
        self.milestone_config = (manager_cfg, step_dt)

    def pay(self, ordinal, *, scale=None):
        self.calls.append((ordinal, scale))
        return torch.tensor([float(ordinal)], dtype=torch.float32)


_REWARD_MODULE = types.ModuleType(
    "whole_body_tracking.tasks.tracking.mdp."
    "action_ball_full_mdp_lean_rewards"
)
_REWARD_MODULE.LIFECYCLE_PAYMENT_COUNT = 14
_REWARD_MODULE.LeanActionEpochRewardGraph = _RewardGraph


def _seal_env_reward_hot_path(env, graph):
    return types.SimpleNamespace(
        graph=graph,
        graph_type=_RewardGraph,
        dispatcher=type(env)._action_ball_full_mdp_lean_reward_term,
        lifecycle_payment_count=14,
        paddle_first_ordinal=20,
        regularization_first_ordinal=24,
    )


_REWARD_MODULE.seal_env_reward_hot_path = _seal_env_reward_hot_path


@pytest.fixture(autouse=True)
def _install_lean_env_lazy_import_stubs(monkeypatch):
    """Scope the canonical graph and synthetic lazy imports to each test."""

    for name, module in _CANONICAL_TEST_NAMESPACE.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules, _OBSERVATION_MODULE.__name__, _OBSERVATION_MODULE
    )
    monkeypatch.setitem(sys.modules, _REWARD_MODULE.__name__, _REWARD_MODULE)


def test_cold_local_binding_covers_install_getter_hooks_reset_and_teardown():
    assert not hasattr(M, "PINNED_LOCAL_FULL_MDP_STEP_AST_SHA256")
    assert not hasattr(M, "PINNED_LOCAL_FULL_MDP_CLASS_AST_SHA256")
    assert not hasattr(M, "_validate_pinned_local_step_source_bytes")
    direct = {
        name: (function, code, defaults)
        for name, function, code, defaults
        in M._PINNED_LOCAL_FULL_MDP_DIRECT_METHODS
    }
    properties = {
        name: (descriptor, getter, code, defaults)
        for name, descriptor, getter, code, defaults
        in M._PINNED_LOCAL_FULL_MDP_PROPERTY_GETTERS
    }
    assert {
        "__init__",
        "step",
        "load_managers",
        "install_action_ball_full_mdp_lean_runtime_graph",
        "action_ball_full_mdp_ppo_drain_owner",
        "action_ball_full_mdp_lean_runtime_owner",
        "action_ball_full_mdp_lean_reward_graph",
        "_action_ball_full_mdp_lean_observe_term",
        "_action_ball_full_mdp_lean_reward_term",
        "_before_policy_step",
        "_publish_post_physics_substep",
        "_after_reward_close",
        "_reset_idx",
        "reset",
        "close",
    } <= direct.keys()
    retired_direct_surface = {
        "install_action_ball_full_mdp_reset_genesis",
        "install_action_ball_full_mdp_runtime_components",
        "bind_action_ball_full_mdp_selected_reset_authority",
        "action_ball_full_mdp_num_envs",
        "action_ball_full_mdp_device",
        "action_ball_full_mdp_device_r05_owner",
        "action_ball_full_mdp_motion_owner",
        "action_ball_full_mdp_racket_owner",
        "action_ball_full_mdp_physical_owner",
        "action_ball_full_mdp_r03_owner",
        "action_ball_full_mdp_r06_owner",
        "action_ball_full_mdp_r07_owner",
    }
    assert retired_direct_surface.isdisjoint(direct)
    for name in retired_direct_surface:
        assert not hasattr(M.ActionBallFullMdpManagerBasedRLEnv, name)
    assert {
        "action_ball_full_mdp_runtime_lease",
        "full_mdp_runtime_owner",
        "action_ball_r10_checkpoint_adapter",
        "action_ball_r10_cold_restore_capsule",
    } <= properties.keys()
    assert M._PINNED_LOCAL_FULL_MDP_MODULE is M
    assert M._PINNED_LOCAL_FULL_MDP_ENV_CLASS is (
        M.ActionBallFullMdpManagerBasedRLEnv
    )
    for function, code, defaults in direct.values():
        assert type(function) is types.FunctionType
        assert type(code) is types.CodeType
        assert function.__code__ is code
        assert M._cold_local_plain_callable_defaults(function) == defaults
        assert function.__globals__ is vars(M)
    for descriptor, getter, code, defaults in properties.values():
        assert type(descriptor) is property
        assert type(getter) is types.FunctionType
        assert type(code) is types.CodeType
        assert getter.__code__ is code
        assert M._cold_local_plain_callable_defaults(getter) == defaults
        assert getter.__globals__ is vars(M)
    M._assert_runtime_uses_pinned_local_step()


def test_cold_local_checks_precede_constructor_or_install_mutation():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpManagerBasedRLEnv"
    )
    constructor = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    call_lines = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in {
            "_assert_runtime_uses_pinned_local_step",
            "_assert_runtime_instance_uses_pinned_local_step",
            "super().__init__",
        }
    }
    lease_line = next(
        node.lineno
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and ast.unparse(target)
            == "self._action_ball_full_mdp_runtime_lease"
            for target in node.targets
        )
    )
    assert call_lines["_assert_runtime_uses_pinned_local_step"] < lease_line
    assert (
        call_lines["_assert_runtime_instance_uses_pinned_local_step"]
        < lease_line
        < call_lines["super().__init__"]
    )


def test_prephysics_hook_is_between_apply_action_and_scene_write():
    tree = ast.parse(MODULE_PATH.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpManagerBasedRLEnv"
    )
    step = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    calls = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in {
            "self.action_manager.apply_action",
            "before_physics",
            "self.scene.write_data_to_sim",
        }
    }
    assert calls["self.action_manager.apply_action"] < calls[
        "before_physics"
    ] < calls["self.scene.write_data_to_sim"]


def test_reward_actual_close_is_between_manager_compute_and_runtime_close_before_reset():
    tree = ast.parse(MODULE_PATH.read_text())
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpManagerBasedRLEnv"
    )
    step = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    names = {
        "self.reward_manager.compute",
        "components.reward_graph.close_milestone_actual_reward",
        "components.epoch_owner.milestone.add_step_return",
        "self._after_reward_close",
        "self._reset_idx",
    }
    calls = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(step)
        if isinstance(node, ast.Call) and ast.unparse(node.func) in names
    }
    assert set(calls) == names
    assert (
        calls["self.reward_manager.compute"]
        < calls["components.reward_graph.close_milestone_actual_reward"]
        < calls["components.epoch_owner.milestone.add_step_return"]
        < calls["self._after_reward_close"]
        < calls["self._reset_idx"]
    )


def _env():
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    env._action_ball_full_mdp_runtime_lease = lease
    env._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
    env._action_ball_full_mdp_manager_construction_state = "command_manager_ready"
    env.cfg = types.SimpleNamespace(
        rewards=object(), observations=object(), terminations=object()
    )
    env.step_dt = 0.02
    return env, lease


class _ControllerHistoryOwner:
    def action_ball_full_mdp_restore_physical_birth_controller_history(
        self, _env_ids
    ):
        return None


def _components(env):
    values = [object() for _ in range(11)]
    values[2] = _ControllerHistoryOwner()
    source = _ObservationSource(env, values[10], values[0])
    graph = _RewardGraph(values[0])
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
        reward_graph=graph,
        lean_runtime_owner=values[10],
        observation_source=source,
    )


def _manager_cfgs():
    return (
        {name: object() for name in M.reward_contract.MANAGER_NAMES},
        {"policy": object(), "critic": object()},
        {
            name: object()
            for name in (
                "time_out",
                "base_fell_tilt",
                "base_too_low",
                "joint_qdes_forbidden",
                "robot_hit_table",
            )
        },
    )


class _LivePhysxPort:
    def shutdown_action_epoch_live_physx_fact_owner(self):
        pass


def _live_physx_shutdown():
    return _LivePhysxPort().shutdown_action_epoch_live_physx_fact_owner


def _observation_source(env, components):
    assert components.observation_source._env is env
    return components.observation_source


def test_lean_graph_installs_genesis_and_same_top_together():
    env, lease = _env()
    components = _components(env)
    authority, receipt = object(), object()
    rewards, observations, terminations = _manager_cfgs()

    env.install_action_ball_full_mdp_lean_runtime_graph(
        lease,
        genesis_authority=authority,
        genesis_receipt=receipt,
        components=components,
        reward_manager_cfg=rewards,
        observation_source=_observation_source(env, components),
        observation_manager_cfg=observations,
        termination_manager_cfg=terminations,
        live_physx_shutdown=_live_physx_shutdown(),
    )

    assert env._action_ball_full_mdp_components is components
    assert components.reward_graph.milestone_config == (rewards, 0.02)
    genesis = env._action_ball_full_mdp_reset_genesis_install
    assert genesis.authority is authority
    assert genesis.receipt is receipt
    assert env._action_ball_full_mdp_lean_reward_graph is components.reward_graph
    assert env._action_ball_full_mdp_reward_hot_path.graph is components.reward_graph
    source = env._action_ball_full_mdp_lean_observation_source
    assert source._env is env
    assert source._runtime_owner is components.lean_runtime_owner
    assert env.cfg.rewards is rewards
    assert env.cfg.observations is observations
    assert env.cfg.terminations is terminations
    assert callable(env._action_ball_full_mdp_live_physx_shutdown)
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    assert env.action_ball_full_mdp_lean_reward_graph(lease) is components.reward_graph
    assert env._action_ball_full_mdp_lean_observe_term(group="policy") == "policy"
    value = env._action_ball_full_mdp_lean_reward_term(
        ordinal=0, scale=0.2
    )
    assert torch.equal(value, torch.tensor([0.0]))
    assert components.reward_graph.calls == [(0, 0.2)]


def test_lean_reward_term_rejects_non_exact_ordinal_before_graph_pay():
    env, lease = _env()
    components = _components(env)
    rewards, observations, terminations = _manager_cfgs()
    env.install_action_ball_full_mdp_lean_runtime_graph(
        lease,
        genesis_authority=object(),
        genesis_receipt=object(),
        components=components,
        reward_manager_cfg=rewards,
        observation_source=_observation_source(env, components),
        observation_manager_cfg=observations,
        termination_manager_cfg=terminations,
        live_physx_shutdown=_live_physx_shutdown(),
    )
    env._action_ball_full_mdp_manager_construction_state = "sealed"

    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="ordinal must be one exact int"
    ):
        env._action_ball_full_mdp_lean_reward_term(
            ordinal=True, scale=0.2
        )
    assert components.reward_graph.calls == []


def test_lean_reward_term_uses_construction_bound_graph_after_publication_drift():
    env, lease = _env()
    components = _components(env)
    rewards, observations, terminations = _manager_cfgs()
    env.install_action_ball_full_mdp_lean_runtime_graph(
        lease,
        genesis_authority=object(),
        genesis_receipt=object(),
        components=components,
        reward_manager_cfg=rewards,
        observation_source=_observation_source(env, components),
        observation_manager_cfg=observations,
        termination_manager_cfg=terminations,
        live_physx_shutdown=_live_physx_shutdown(),
    )
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    foreign = _RewardGraph(components.epoch_owner)
    env._action_ball_full_mdp_lean_reward_graph = foreign

    value = env._action_ball_full_mdp_lean_reward_term(
        ordinal=0, scale=0.2
    )
    assert torch.equal(value, torch.tensor([0.0]))
    assert foreign.calls == []
    assert components.reward_graph.calls == [(0, 0.2)]


def test_lean_graph_foreign_lease_has_no_partial_install():
    env, _lease = _env()
    rewards, observations, terminations = _manager_cfgs()
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="foreign lease"):
        env.install_action_ball_full_mdp_lean_runtime_graph(
            object(),
            genesis_authority=object(),
            genesis_receipt=object(),
            components=_components(env),
            reward_manager_cfg=rewards,
            observation_source=object(),
            observation_manager_cfg=observations,
            termination_manager_cfg=terminations,
            live_physx_shutdown=_live_physx_shutdown(),
        )
    assert not hasattr(env, "_action_ball_full_mdp_components")
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")


@pytest.mark.parametrize(
    "failed_name", ("rewards", "observations", "terminations")
)
def test_lean_graph_setter_failure_restores_all_three_cfgs_and_publications(
    failed_name,
):
    """A mutating setter is a legal counterexample to naive sequential install."""

    env, lease = _env()
    components = _components(env)
    authority, receipt = object(), object()
    rewards, observations, terminations = _manager_cfgs()
    old = {
        "rewards": env.cfg.rewards,
        "observations": env.cfg.observations,
        "terminations": env.cfg.terminations,
    }

    class _MutateThenRaiseCfg:
        def __init__(self):
            self._values = dict(old)
            self.failed_once = False

        def __getattr__(self, name):
            if name in self._values:
                return self._values[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name in {"_values", "failed_once"}:
                object.__setattr__(self, name, value)
                return
            if name not in self._values:
                raise AttributeError(name)
            self._values[name] = value
            if name == failed_name and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("injected mutating setter failure")

    env.cfg = _MutateThenRaiseCfg()
    with pytest.raises(RuntimeError, match="mutating setter"):
        env.install_action_ball_full_mdp_lean_runtime_graph(
            lease,
            genesis_authority=authority,
            genesis_receipt=receipt,
            components=components,
            reward_manager_cfg=rewards,
            observation_source=_observation_source(env, components),
            observation_manager_cfg=observations,
            termination_manager_cfg=terminations,
            live_physx_shutdown=_live_physx_shutdown(),
        )

    assert env.cfg.rewards is old["rewards"]
    assert env.cfg.observations is old["observations"]
    assert env.cfg.terminations is old["terminations"]
    assert not hasattr(env, "_action_ball_full_mdp_components")
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_lean_reward_graph")
    assert not hasattr(env, "_action_ball_full_mdp_lean_observation_source")
    assert not hasattr(env, "_action_ball_full_mdp_live_physx_shutdown")
    assert not hasattr(env, "_action_ball_full_mdp_reward_hot_path")


@pytest.mark.parametrize("prior_state", ("complete", "cold_discarded"))
def test_lean_graph_is_one_shot_after_atomic_install(prior_state):
    env, lease = _env()
    components = _components(env)
    rewards, observations, terminations = _manager_cfgs()
    env.install_action_ball_full_mdp_lean_runtime_graph(
        lease,
        genesis_authority=object(),
        genesis_receipt=object(),
        components=components,
        reward_manager_cfg=rewards,
        observation_source=_observation_source(env, components),
        observation_manager_cfg=observations,
        termination_manager_cfg=terminations,
        live_physx_shutdown=_live_physx_shutdown(),
    )
    if prior_state == "cold_discarded":
        env._poison_action_ball_full_mdp_construction_installs()

    replacement = _components(env)
    replacement_rewards, replacement_observations, replacement_terminations = (
        _manager_cfgs()
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="cannot replace a partial or complete install",
    ):
        env.install_action_ball_full_mdp_lean_runtime_graph(
            lease,
            genesis_authority=object(),
            genesis_receipt=object(),
            components=replacement,
            reward_manager_cfg=replacement_rewards,
            observation_source=_observation_source(env, replacement),
            observation_manager_cfg=replacement_observations,
            termination_manager_cfg=replacement_terminations,
            live_physx_shutdown=_live_physx_shutdown(),
        )

    if prior_state == "complete":
        assert env._action_ball_full_mdp_components is components
        assert env._action_ball_full_mdp_lean_reward_graph is components.reward_graph
        assert env.cfg.rewards is rewards
        assert env.cfg.observations is observations
        assert env.cfg.terminations is terminations
    else:
        assert env._action_ball_full_mdp_components is M._ABSENT
        assert env._action_ball_full_mdp_reset_genesis_install is M._ABSENT
        assert env._action_ball_full_mdp_lean_reward_graph is M._ABSENT
        assert env._action_ball_full_mdp_lean_observation_source is M._ABSENT


def test_lean_registry_rejects_role_aliasing():
    shared = object()
    source = object()
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="aliases"):
        M.FullMdpLeanRuntimeComponents(
            epoch_owner=shared,
            device_r05_owner=shared,
            motion_owner=object(),
            racket_owner=object(),
            physical_owner=object(),
            r03_owner=object(),
            r06_owner=object(),
            r07_owner=object(),
            r07_plant_fact_adapter=object(),
            reward_graph=object(),
            lean_runtime_owner=object(),
            observation_source=source,
        )


def test_post_install_manager_failure_poison_drops_all_three_manager_refs():
    env, lease = _env()
    rewards, observations, terminations = _manager_cfgs()
    components = _components(env)
    env.install_action_ball_full_mdp_lean_runtime_graph(
        lease,
        genesis_authority=object(),
        genesis_receipt=object(),
        components=components,
        reward_manager_cfg=rewards,
        observation_source=_observation_source(env, components),
        observation_manager_cfg=observations,
        termination_manager_cfg=terminations,
        live_physx_shutdown=_live_physx_shutdown(),
    )
    env.observation_manager = object()
    env.termination_manager = object()
    env.reward_manager = object()

    env._poison_action_ball_full_mdp_construction_installs()

    assert env._action_ball_full_mdp_components is M._ABSENT
    assert env._action_ball_full_mdp_reset_genesis_install is M._ABSENT
    assert env._action_ball_full_mdp_lean_reward_graph is M._ABSENT
    assert env._action_ball_full_mdp_lean_observation_source is M._ABSENT
    assert callable(env._action_ball_full_mdp_live_physx_shutdown)
    assert env.observation_manager is M._ABSENT
    assert env.termination_manager is M._ABSENT
    assert env.reward_manager is M._ABSENT


class _ResetManager:
    def __init__(self, trace, label):
        self._trace = trace
        self._label = label

    def reset(self, env_ids):
        self._trace.append((self._label, tuple(env_ids.tolist())))
        return {}

    def compute(self, *, env_ids):
        self._trace.append((self._label + "_compute", tuple(env_ids.tolist())))


class _ResetEventManager(_ResetManager):
    available_modes = ()


class _TerminalResetManager(_ResetManager):
    _NAMES = (
        "time_out",
        "base_fell_tilt",
        "base_too_low",
        "joint_qdes_forbidden",
        "robot_hit_table",
    )

    def __init__(self, trace, label, *, device):
        super().__init__(trace, label)
        self._term_dones = torch.zeros(
            (2, len(self._NAMES)), dtype=torch.bool, device=device
        )
        self._term_dones[0, self._NAMES.index("joint_qdes_forbidden")] = True
        self._term_dones[0, self._NAMES.index("robot_hit_table")] = True

    @property
    def active_terms(self):
        return list(self._NAMES)

    def get_term(self, name):
        return self._term_dones[:, self._NAMES.index(name)]


class _SelectedResetLeafPort:
    """One leaf-local exact-method port; the real Lean owner coordinates."""

    def __init__(self, name, trace):
        self.name = name
        self.trace = trace
        self._owned = set()

    def _mint(self, operation):
        self.trace.append((self.name, operation))
        token = object()
        self._owned.add(token)
        return token

    def _require(self, token):
        assert token in self._owned
        return token

    def action_ball_full_mdp_restore_physical_birth_controller_history(
        self, _env_ids
    ):
        return None

    def prepare_action_ball_continuous_motion_selected_reset(self, _prepared):
        return self._mint("prepare")

    def arm_prevalidated_action_ball_continuous_motion_selected_reset(
        self, token
    ):
        self._require(token)
        return self._mint("arm")

    def commit_prevalidated_action_ball_continuous_motion_selected_reset(
        self, token
    ):
        self._require(token)
        return self._mint("commit")

    def stage_action_ball_continuous_racket_selected_reset(self, _prepared):
        return self._mint("stage")

    def finalize_action_ball_continuous_racket_selected_reset(self, token):
        self._require(token)
        return self._mint("finalize")

    def commit_prevalidated_action_ball_continuous_racket_selected_reset(
        self, token
    ):
        self._require(token)
        return self._mint("commit")

    def prepare_selected_reset(self, _prepared):
        return self._mint("prepare")

    def arm_prevalidated_selected_reset(self, token, physical):
        self._require(token)
        assert physical is not None
        return self._mint("arm")

    def commit_prevalidated_selected_reset(self, token, physical):
        self._require(token)
        assert physical is not None
        return self._mint("commit")

    def stage_selected_true_reset(self, r06_prepared):
        assert r06_prepared is not None
        return self._mint("stage")

    def finalize_selected_true_reset(self, token):
        self._require(token)
        return self._mint("finalize")

    def prearm_selected_true_reset(self, token, r06_armed):
        self._require(token)
        assert r06_armed is not None
        return self._mint("prearm")

    def commit_prevalidated_selected_true_reset(self, token):
        self._require(token)
        return self._mint("commit")

    def require_owned_selected_reset_commit(
        self, token, *, expected_prepared_true_reset=None
    ):
        del expected_prepared_true_reset
        return self._require(token)

    def acknowledge_r06_selected_reset_commit(self, physical, r06):
        self._require(physical)
        assert r06 is not None
        self.trace.append((self.name, "ack_r06"))

    def complete_action_ball_continuous_motion_selected_reset_after_r05(
        self, token, _receipt
    ):
        self._require(token)
        return self._mint("complete")

    def complete_action_ball_continuous_racket_selected_reset_after_r05(
        self, token, _receipt
    ):
        self._require(token)
        return self._mint("complete")

    def complete_selected_reset_after_r05(self, token, _receipt):
        self._require(token)
        return self._mint("complete")

    def complete_selected_true_reset_after_r05(
        self, physical, r06, _receipt
    ):
        self._require(physical)
        assert r06 is not None
        return self._mint("complete")

    def consume_owned_selected_reset_completion(
        self, token, *, expected_prepared_true_reset=None
    ):
        del expected_prepared_true_reset
        self._require(token)
        self.trace.append((self.name, "consume"))

    def abort_prevalidated_action_ball_continuous_motion_selected_reset(
        self, token
    ):
        self._require(token)
        self.trace.append((self.name, "abort"))

    def abort_prevalidated_action_ball_continuous_racket_selected_reset(
        self, token
    ):
        self._require(token)
        self.trace.append((self.name, "abort"))

    def abort_selected_reset(self, token):
        self._require(token)
        self.trace.append((self.name, "abort"))

    def abort_selected_true_reset(self, token):
        self._require(token)
        self.trace.append((self.name, "abort"))

    def poison_global_reveal_epoch(self, _reason):
        self.trace.append((self.name, "poison"))

    def poison_selected_reset(self, _reason):
        self.trace.append((self.name, "poison"))


def _real_selected_reset_env(device, monkeypatch):
    """Build the real env callpoint and real lean top around thin leaves."""

    import test_action_ball_continuous_runtime_transaction_device as dtest

    lean_runtime = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_runtime"
    )
    epoch_module = lean_runtime.epoch_v1
    selected_reset_module = epoch_module.selected_reset
    selected_runtime_name = (
        selected_reset_module.__package__ + "."
        if selected_reset_module.__package__
        else ""
    ) + "action_ball_full_mdp_lean_runtime"
    # Other focused files may preload the same production sources through
    # their short module names.  SelectedReset resolves its owner class from
    # its own package namespace, so bind both possible import spellings to the
    # exact runtime that supplied this epoch instead of borrowing collection
    # order as identity.
    for runtime_name in {
        selected_runtime_name,
        "action_ball_full_mdp_lean_runtime",
        (
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_runtime"
        ),
    }:
        monkeypatch.setitem(sys.modules, runtime_name, lean_runtime)
    reward_leaf = "action_ball_full_mdp_lean_rewards"
    reward_name = lean_runtime.__package__ + "." + reward_leaf
    package_module = sys.modules[lean_runtime.__package__]
    rewards_module = vars(package_module).get(reward_leaf)
    if rewards_module is _REWARD_MODULE or not hasattr(
        rewards_module, "epoch_v1"
    ):
        if sys.modules.get(reward_name) is _REWARD_MODULE:
            monkeypatch.delitem(sys.modules, reward_name)
        if vars(package_module).get(reward_leaf) is _REWARD_MODULE:
            monkeypatch.delattr(package_module, reward_leaf)
        rewards_module = importlib.import_module(reward_name)
    else:
        monkeypatch.setitem(sys.modules, reward_name, rewards_module)
    assert rewards_module.epoch_v1 is epoch_module
    assert lean_runtime.device_r05 is dtest.r05
    monkeypatch.setattr(
        lean_runtime.device_r05,
        "_require_action_epoch_module",
        lambda: epoch_module,
    )
    assert dtest.r05 is lean_runtime.device_r05
    assert sys.modules["action_ball_continuous_runtime_transaction_device"] is (
        lean_runtime.device_r05
    )

    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    poisoned = []
    env.num_envs = 2
    env.device = torch.device(device)
    env.step_dt = 0.02
    env.cfg = types.SimpleNamespace(decimation=1)
    env._sim_step_counter = 0
    env.common_step_counter = 53
    env.episode_length_buf = torch.tensor(
        [31, 47], dtype=torch.int64, device=device
    )
    env._action_ball_full_mdp_runtime_lease = object()
    env._action_ball_full_mdp_active_reset_record = None
    env._action_ball_full_mdp_reset_generation = torch.ones(
        2, dtype=torch.int64, device=device
    )
    env._action_ball_full_mdp_reset_callpoint_authority = None
    env._action_ball_full_mdp_lean_genesis_reset_pending = False
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    env._full_mdp_post_physics_poison = None
    env._full_mdp_active_dispatch = None

    epoch_owner = epoch_module.ActionEpochOwner(
        num_envs=2,
        device=device,
        initial_reset_generation=torch.ones(
            2, dtype=torch.int64, device=device
        ),
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=device),
        reset_generation=torch.ones(2, dtype=torch.int64, device=device),
    )
    profile = dtest._ProfileAuthority(device)
    genesis = dtest._Genesis(device, 2)
    cadence = dtest._Cadence(device, 2)
    question = dtest._Question(device)
    d05_owner = lean_runtime.device_r05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=12345,
        num_envs=2,
        journal_capacity=64,
        max_reveal_epochs_per_drain=64,
        genesis_authority=genesis,
        genesis_receipt=genesis.receipt,
        cadence_authority=cadence,
        question_authority=question,
        reveal_boundary_authority=None,
        child_completion_authorities=(),
        true_reset_authority=None,
        diagnostic_epoch_owner=epoch_owner,
    )
    epoch_owner.bind_motion_cadence_owner(cadence)
    epoch_owner.bind_d05_accept_writers(
        motion_write=d05_owner._commit_action_epoch_motion_write,
        racket_write=d05_owner._commit_action_epoch_racket_write,
        r05_write=d05_owner._commit_action_epoch_r05_write,
    )
    leaves = {
        name: _SelectedResetLeafPort(name, trace)
        for name in ("motion", "racket", "physical", "r06")
    }
    reward_graph = rewards_module.LeanActionEpochRewardGraph(
        epoch_owner=epoch_owner
    )
    env._action_ball_full_mdp_lean_reward_graph = reward_graph
    r03_owner = object()
    r07_owner = object()
    runtime_owner = lean_runtime.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=env._action_ball_full_mdp_runtime_lease,
        epoch_owner=epoch_owner,
        reward_graph=reward_graph,
        r05_runtime=d05_owner,
        motion=leaves["motion"],
        racket=leaves["racket"],
        physical_ball=leaves["physical"],
        r06_landing_outcome=leaves["r06"],
        r03_strike_fact=r03_owner,
        r07_recovery=r07_owner,
    )
    env._full_mdp_runtime_owner = runtime_owner
    epoch_owner.bind_selected_reset_owner(runtime_owner)
    d05_owner.bind_true_reset_authority(runtime_owner)
    first = runtime_owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=2
    )
    runtime_owner.mark_optimizer_returned(first, update_index=0)
    summary = runtime_owner.prepare_post_update_summary(first, update_index=0)
    runtime_owner.acknowledge_post_update(first, summary, update_index=0)

    observation_source = _ObservationSource(
        env, runtime_owner, epoch_owner
    )
    components = M.FullMdpLeanRuntimeComponents(
        epoch_owner=epoch_owner,
        device_r05_owner=d05_owner,
        motion_owner=leaves["motion"],
        racket_owner=leaves["racket"],
        physical_owner=leaves["physical"],
        r03_owner=r03_owner,
        r06_owner=leaves["r06"],
        r07_owner=r07_owner,
        r07_plant_fact_adapter=object(),
        reward_graph=reward_graph,
        lean_runtime_owner=runtime_owner,
        observation_source=observation_source,
    )
    env._action_ball_full_mdp_components = components
    env._action_ball_full_mdp_lean_observation_source = observation_source
    env._full_mdp_selected_true_reset = runtime_owner.selected_true_reset
    env._test_terminal_reset_facts_seen = []
    original_prepare_selected_reset = epoch_owner.prepare_selected_true_reset

    def capture_prepare_selected_reset(**kwargs):
        env._test_terminal_reset_facts_seen.append(
            kwargs["terminal_reset_facts_i64"].detach().clone()
        )
        return original_prepare_selected_reset(**kwargs)

    epoch_owner.prepare_selected_true_reset = capture_prepare_selected_reset

    def poison(*, reason, exact_stamp):
        poisoned.append({"reason": reason, "exact_stamp": exact_stamp})
        if env._full_mdp_post_physics_poison is None:
            env._full_mdp_post_physics_poison = (reason, exact_stamp)

    env._poison = poison
    env.curriculum_manager = _ResetManager(trace, "curriculum")
    env.scene = types.SimpleNamespace(
        reset=lambda ids: trace.append(
            ("scene", tuple(ids.detach().cpu().tolist()))
        )
    )
    env.event_manager = _ResetEventManager(trace, "event")
    env.observation_manager = _ResetManager(trace, "observation_reset")
    env.action_manager = _ResetManager(trace, "action_reset")
    env.reward_manager = _ResetManager(trace, "reward_reset")
    env.recorder_manager = _ResetManager(trace, "recorder_reset")
    env.termination_manager = _TerminalResetManager(
        trace, "termination_reset", device=device
    )
    env.reset_terminated = torch.tensor(
        [True, False], dtype=torch.bool, device=device
    )
    env.reset_time_outs = torch.zeros(2, dtype=torch.bool, device=device)
    env._reset_non_action_ball_commands = lambda ids: {
        "thin_native_ids": ids.detach().clone()
    }
    env.extras = {}
    return (
        env,
        runtime_owner,
        epoch_owner,
        d05_owner,
        leaves,
        trace,
        poisoned,
    )


def _peer_raw_bytes(epoch_owner, d05_owner):
    rows = []
    record = epoch_owner._publication.current

    def collect(value):
        if type(value) is torch.Tensor and value.ndim >= 1 and value.shape[0] == 2:
            rows.append(
                value[1]
                .detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
        elif hasattr(value, "__dict__"):
            for nested in vars(value).values():
                collect(nested)

    collect(record)
    for value in d05_owner._publication.live.values():
        collect(value)
    return tuple(rows)


def _milestone_semantic_payload(epoch_owner):
    leaf = epoch_owner.milestone
    module = sys.modules[type(leaf).__module__]
    i64, f64 = (
        value.detach().cpu().contiguous() for value in leaf.pack_views()
    )
    names = tuple(str(index) for index in range(module.REWARD_TERM_COUNT))
    return module.decode_host_window(i64, f64).as_json(names)


def _selected_env_devices():
    values = ["cpu"]
    if torch.cuda.is_available():
        values.append("cuda:0")
    return values


def _cuda_selected_reset_mutation_subprocess_main(
    case: str, receipt_path: str
) -> None:
    """Exercise a rejected independent join in an isolated CUDA process."""

    monkeypatch = pytest.MonkeyPatch()
    try:
        (
            env,
            owner,
            epoch_owner,
            d05_owner,
            leaves,
            trace,
            _poisoned,
        ) = _real_selected_reset_env(torch.device("cuda:0"), monkeypatch)
        env_ids = torch.tensor([0], dtype=torch.int64, device="cuda:0")
        if case == "d05":
            d05_owner._reset_generation[0] += 1
        elif case == "epoch":
            epoch_owner._reset_generation[0] += 1
        elif case == "writer":
            # In-range but non-monotonic: env construction is defined, while
            # Device-R05 independently latches its selection writer fault.
            env_ids = torch.tensor([1, 0], dtype=torch.int64, device="cuda:0")
        elif case == "overflow":
            maximum = torch.iinfo(torch.int64).max
            env._action_ball_full_mdp_reset_generation.fill_(maximum)
            epoch_owner._reset_generation.fill_(maximum)
            d05_owner._reset_generation.fill_(maximum)
        else:
            raise AssertionError("unknown selected-reset mutation case")

        env_generation_before = (
            env._action_ball_full_mdp_reset_generation.detach().clone()
        )
        epoch_generation_before = epoch_owner._reset_generation.detach().clone()
        d05_generation_before = d05_owner._reset_generation.detach().clone()
        episode_before = env.episode_length_buf.detach().clone()
        epoch_head_before = epoch_owner.commit_head
        peer_before = _peer_raw_bytes(epoch_owner, d05_owner)
        milestone_before = tuple(
            value.detach().clone() for value in epoch_owner.milestone.pack_views()
        )
        leaf_owned_before = {
            name: len(leaf._owned) for name, leaf in leaves.items()
        }
        trace.clear()
        env._authorize_action_ball_full_mdp_reset_callpoint(
            env_ids, source="step_nonzero"
        )
        failure = None
        try:
            env._reset_idx(env_ids)
        except M.FullMdpPostPhysicsProtocolError as exc:
            failure = type(exc).__name__
        if failure is None:
            raise AssertionError("mutated CUDA reset unexpectedly committed")

        assert owner.poisoned is True
        assert torch.equal(
            env._action_ball_full_mdp_reset_generation,
            env_generation_before,
        )
        assert torch.equal(epoch_owner._reset_generation, epoch_generation_before)
        assert torch.equal(d05_owner._reset_generation, d05_generation_before)
        assert torch.equal(env.episode_length_buf, episode_before)
        assert epoch_owner.commit_head == epoch_head_before
        assert _peer_raw_bytes(epoch_owner, d05_owner) == peer_before
        milestone_after = tuple(
            value.detach().clone() for value in epoch_owner.milestone.pack_views()
        )
        assert all(
            torch.equal(before, after)
            for before, after in zip(milestone_before, milestone_after)
        )
        assert {
            name: len(leaf._owned) for name, leaf in leaves.items()
        } == leaf_owned_before
        leaf_trace = [
            item for item in trace if item and item[0] in leaves
        ]
        assert leaf_trace == []
        payload = {
            "case": case,
            "cuda_device": torch.cuda.get_device_name(0),
            "failure": failure,
            "leaf_commit_events": 0,
            "leaf_after_image_unchanged": True,
            "epoch_commit_head_unchanged": True,
            "env_generation_unchanged": True,
        }
        Path(receipt_path).write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        monkeypatch.undo()
    raise SystemExit(23)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ("d05", "epoch", "writer", "overflow"))
def test_cuda_independent_reset_join_rejects_before_every_leaf(
    case, tmp_path
):
    receipt = tmp_path / (case + ".json")
    env = os.environ.copy()
    tests = str(ROOT / "tests")
    env["PYTHONPATH"] = tests + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import test_action_ball_full_mdp_lean_env_install as t; "
        "t._cuda_selected_reset_mutation_subprocess_main"
        "(__import__('sys').argv[1], __import__('sys').argv[2])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, case, str(receipt)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload == {
        "case": case,
        "cuda_device": torch.cuda.get_device_name(0),
        "failure": "FullMdpPostPhysicsProtocolError",
        "leaf_commit_events": 0,
        "leaf_after_image_unchanged": True,
        "epoch_commit_head_unchanged": True,
        "env_generation_unchanged": True,
    }


def test_selected_reset_fixture_ignores_stale_focused_epoch_preloader(
    monkeypatch,
):
    import test_action_ball_continuous_runtime_transaction_device  # noqa: F401

    canonical_lean = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_runtime"
    )

    stale_epoch = types.ModuleType("action_ball_full_mdp_epoch")
    stale_epoch.ActionEpochOwner = type("StaleActionEpochOwner", (), {})
    stale_test = types.ModuleType("test_action_ball_full_mdp_lean_runtime")
    stale_test.E = stale_epoch
    monkeypatch.setitem(sys.modules, stale_epoch.__name__, stale_epoch)
    monkeypatch.setitem(sys.modules, stale_test.__name__, stale_test)

    _env_value, owner, epoch_owner, d05_owner, *_rest = (
        _real_selected_reset_env(torch.device("cpu"), monkeypatch)
    )
    assert type(epoch_owner) is canonical_lean.epoch_v1.ActionEpochOwner
    assert owner.epoch_owner is epoch_owner
    assert d05_owner._diagnostic_epoch_owner is epoch_owner


def test_genesis_reset_computes_command_once_before_first_observation():
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    env.num_envs = 2
    env.device = "cpu"
    env.step_dt = 0.02
    env.cfg = types.SimpleNamespace(decimation=1)
    env._sim_step_counter = 0
    env.episode_length_buf = torch.ones(2, dtype=torch.int64)
    env._action_ball_full_mdp_components = _components(env)
    env._action_ball_full_mdp_lean_genesis_reset_pending = True
    env._action_ball_full_mdp_active_reset_record = None
    env._action_ball_full_mdp_reset_generation = torch.ones(
        2, dtype=torch.int64
    )
    env._action_ball_full_mdp_reset_callpoint_authority = None
    env._full_mdp_selected_true_reset = lambda *_args: trace.append(
        ("selected_reset",)
    )
    env._poison = lambda **_kwargs: None
    env.curriculum_manager = _ResetManager(trace, "curriculum")
    env.scene = types.SimpleNamespace(
        reset=lambda ids: trace.append(("scene", tuple(ids.tolist())))
    )
    env.event_manager = _ResetManager(trace, "event")
    env.event_manager.available_modes = ()
    env.observation_manager = _ResetManager(trace, "observation_reset")
    env.action_manager = _ResetManager(trace, "action_reset")
    env.reward_manager = _ResetManager(trace, "reward_reset")
    env.recorder_manager = _ResetManager(trace, "recorder_reset")
    env.termination_manager = _ResetManager(trace, "termination_reset")
    env._reset_non_action_ball_commands = lambda ids: {}
    env.extras = {}
    env.command_manager = types.SimpleNamespace(
        compute=lambda *, dt: trace.append(("command", dt))
    )
    env._full_mdp_after_command_compute_before_observation = (
        lambda step: trace.append(("after_command", step))
    )

    env_ids = torch.arange(2, dtype=torch.int64)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="reset_all_arange"
    )
    env._reset_idx(env_ids)

    assert ("selected_reset",) not in trace
    assert trace.count(("command", 0.02)) == 1
    assert trace.count(("after_command", 0)) == 1
    assert trace.index(("recorder_reset", (0, 1))) < trace.index(
        ("command", 0.02)
    ) < trace.index(("after_command", 0))
    assert env._action_ball_full_mdp_lean_genesis_reset_pending is False
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation,
        torch.ones(2, dtype=torch.int64),
    )


@pytest.mark.parametrize("device", _selected_env_devices())
def test_real_env_non_genesis_selected_reset_reaches_next_observation(
    device, monkeypatch
):
    (
        env,
        owner,
        epoch_owner,
        d05_owner,
        _leaves,
        trace,
        poisoned,
    ) = _real_selected_reset_env(torch.device(device), monkeypatch)
    # Preserve two awkward peer payloads through the actual environment
    # callpoint: float32 negative zero and a noncanonical quiet-NaN payload.
    task_bits = epoch_owner._publication.current.task.task_f32.view(
        torch.int32
    )
    task_bits[1, 0, 0] = -2147483648
    task_bits[1, 0, 1] = 2143294004
    epoch_owner.milestone.add_step_return(
        torch.tensor([2.0, 5.0], device=device)
    )
    peer_before = _peer_raw_bytes(epoch_owner, d05_owner)
    trace.clear()

    env_ids = torch.tensor([0], dtype=torch.int64, device=device)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    env._reset_idx(env_ids)

    assert poisoned == []
    assert owner.poisoned is False
    assert env._action_ball_full_mdp_reset_generation.tolist() == [2, 1]
    assert epoch_owner.current().reset_generation.tolist() == [2, 1]
    assert d05_owner.reset_generation.tolist() == [2, 1]
    assert len(env._test_terminal_reset_facts_seen) == 1
    assert env._test_terminal_reset_facts_seen[0].tolist() == [
        [53, 31, 24],
        [-1, -1, 0],
    ]
    assert env.episode_length_buf.tolist() == [0, 47]
    milestone = _milestone_semantic_payload(epoch_owner)
    assert milestone["episodes"] == {
        "completed": 1,
        "length_sum": 31,
        "reason_time_out": 0,
        "reason_base_fell_tilt": 0,
        "reason_base_too_low": 0,
        "reason_joint_qdes_forbidden": 1,
        "reason_robot_hit_table": 1,
        "return_sum": 2.0,
        "return_sum_sq": 4.0,
    }
    assert epoch_owner.milestone.open_episode_return.tolist() == [0.0, 5.0]
    assert _peer_raw_bytes(epoch_owner, d05_owner) == peer_before
    assert d05_owner._true_reset_receipts == {}
    assert d05_owner._journal_tail == d05_owner._journal_head == 1
    native = [
        item
        for item in trace
        if item[0]
        in {
            "curriculum_compute",
            "scene",
            "observation_reset",
            "action_reset",
            "reward_reset",
            "curriculum",
            "event",
            "termination_reset",
            "recorder_reset",
        }
    ]
    assert native == [
        ("curriculum_compute", (0,)),
        ("scene", (0,)),
        ("observation_reset", (0,)),
        ("action_reset", (0,)),
        ("reward_reset", (0,)),
        ("curriculum", (0,)),
        ("event", (0,)),
        ("termination_reset", (0,)),
        ("recorder_reset", (0,)),
    ]
    next_observation = env._action_ball_full_mdp_lean_observe_term(
        group="policy"
    )
    assert next_observation.tolist() == [2, 1]


@pytest.mark.parametrize("device", _selected_env_devices())
def test_native_reset_failure_does_not_preclose_episode_carry(device, monkeypatch):
    env, owner, epoch_owner, *_rest = _real_selected_reset_env(
        torch.device(device), monkeypatch
    )
    epoch_owner.milestone.add_step_return(
        torch.tensor([2.0, 5.0], device=device)
    )

    def fail_native_reset(_env_ids):
        raise RuntimeError("synthetic native reward reset failure")

    env.reward_manager.reset = fail_native_reset
    env_ids = torch.tensor([0], dtype=torch.int64, device=device)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="selected reset failed; environment is poisoned",
    ):
        env._reset_idx(env_ids)

    assert owner.poisoned is False
    milestone = _milestone_semantic_payload(epoch_owner)
    assert all(value == 0 for value in milestone["episodes"].values())
    assert epoch_owner.milestone.open_episode_return.tolist() == [2.0, 5.0]


@pytest.mark.parametrize("device", _selected_env_devices())
def test_real_env_selected_reset_failure_precedes_native_and_blocks_retry(
    device, monkeypatch
):
    (
        env,
        owner,
        epoch_owner,
        d05_owner,
        _leaves,
        trace,
        poisoned,
    ) = _real_selected_reset_env(torch.device(device), monkeypatch)
    d05_owner._mutation_version.fill_(torch.iinfo(torch.int64).max)
    generation_before = (
        env._action_ball_full_mdp_reset_generation.detach().clone()
    )
    peer_before = _peer_raw_bytes(epoch_owner, d05_owner)
    trace.clear()
    env_ids = torch.tensor([0], dtype=torch.int64, device=device)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )

    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="selected reset failed; environment is poisoned",
    ):
        env._reset_idx(env_ids)

    assert owner.poisoned is True
    assert poisoned
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation, generation_before
    )
    assert env.episode_length_buf.tolist() == [31, 47]
    assert _peer_raw_bytes(epoch_owner, d05_owner) == peer_before
    assert not any(
        item[0]
        in {
            "curriculum_compute",
            "scene",
            "observation_reset",
            "action_reset",
            "reward_reset",
            "curriculum",
            "event",
            "termination_reset",
            "recorder_reset",
        }
        for item in trace
    )

    assert env._action_ball_full_mdp_reset_callpoint_authority is None
    retry_trace_before = tuple(trace)
    with pytest.raises(
        M.FullMdpPostPhysicsPoisonedError,
        match="cold reconstruction is required",
    ):
        env._assert_step_may_start()
    assert env._action_ball_full_mdp_reset_callpoint_authority is None
    assert tuple(trace) == retry_trace_before
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation, generation_before
    )
    assert env.episode_length_buf.tolist() == [31, 47]
