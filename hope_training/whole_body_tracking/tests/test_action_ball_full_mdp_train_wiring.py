"""Focused, dependency-light tests for the fresh full-MDP train callsites."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
TRAIN_PATH = SCRIPTS / "train.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_TEMP_IMPORT_STUBS = []
if "hydra" not in sys.modules and importlib.util.find_spec("hydra") is None:
    hydra_stub = types.ModuleType("hydra")

    def _main(**_kwargs):
        return lambda function: function

    hydra_stub.main = _main
    sys.modules["hydra"] = hydra_stub
    _TEMP_IMPORT_STUBS.append("hydra")
if (
    "omegaconf" not in sys.modules
    and importlib.util.find_spec("omegaconf") is None
):
    omega_stub = types.ModuleType("omegaconf")

    class _ListConfig(list):
        pass

    class _OmegaConf:
        pass

    omega_stub.ListConfig = _ListConfig
    omega_stub.OmegaConf = _OmegaConf
    sys.modules["omegaconf"] = omega_stub
    _TEMP_IMPORT_STUBS.append("omegaconf")

import train as train_mod  # noqa: E402

for _stub_name in _TEMP_IMPORT_STUBS:
    sys.modules.pop(_stub_name, None)


class _Registry:
    def __init__(self, entry_point=None):
        self.entry_point = (
            train_mod._ACTION_BALL_FULL_MDP_GYM_ENTRY_POINT
            if entry_point is None
            else entry_point
        )
        self.spec_calls = []

    def spec(self, task_id):
        self.spec_calls.append(task_id)
        return types.SimpleNamespace(entry_point=self.entry_point)


class _LeanOwner:
    @classmethod
    def create_from_env(cls, env, lease):
        return (cls, env, lease)


class _Epoch:
    pass


def _lean_module():
    epoch_v1 = types.SimpleNamespace(ActionEpochOwner=_Epoch)
    return types.SimpleNamespace(
        ActionBallFullMdpLeanRuntimeOwner=_LeanOwner,
        DIAGNOSTIC_DEPENDENCY_KIND=(
            "action_ball_epoch_runtime_dependencies_v1"
        ),
        DIAGNOSTIC_UNAUTHORIZED=True,
        LAUNCH_AUTHORIZED=False,
        epoch_v1=epoch_v1,
    )


def _resolve(monkeypatch, lean_module=None, **overrides):
    lean_module = _lean_module() if lean_module is None else lean_module
    real_import = train_mod.importlib.import_module

    def import_module(name, package=None):
        if name.endswith("action_ball_full_mdp_lean_runtime"):
            return lean_module
        return real_import(name, package)

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    kwargs = {
        "requested": True,
        "task_id": "HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0",
        "gym_registry": _Registry(),
        "num_envs": 2,
        "checkpoint_path": None,
        "checkpoint_tolerant": False,
    }
    kwargs.update(overrides)
    return train_mod._resolve_action_ball_full_mdp_pre_gym_binding(**kwargs)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ({}, False),
        ({"action_ball_full_mdp_runtime": False}, False),
        ({"action_ball_full_mdp_runtime": True}, True),
    ],
)
def test_exact_task_flag_selects_the_actual_fresh_branch(task, expected):
    assert train_mod._action_ball_full_mdp_runtime_requested(task) is expected


@pytest.mark.parametrize(
    "value", [None, 0, 1, "true", "false", [], object()]
)
def test_fresh_task_flag_rejects_non_boolean_values(value):
    with pytest.raises(train_mod._OverrideError, match="exact explicit boolean"):
        train_mod._action_ball_full_mdp_runtime_requested(
            {"action_ball_full_mdp_runtime": value}
        )


def test_fresh_flag_routes_around_the_obsolete_successor_partial_probe():
    env_cfg = types.SimpleNamespace(
        _action_ball_strike_fact_successor_receipt={}
    )
    train_mod._finalize_action_ball_strike_fact_successor_cfg(
        env_cfg,
        {
            "action_ball_full_mdp_runtime": True,
            "action_ball_strike_fact_successor": False,
        },
        [],
    )


def test_fresh_and_obsolete_successor_flags_are_mutually_exclusive():
    with pytest.raises(train_mod._OverrideError, match="mutually exclusive"):
        train_mod._finalize_action_ball_strike_fact_successor_cfg(
            types.SimpleNamespace(),
            {
                "action_ball_full_mdp_runtime": True,
                "action_ball_strike_fact_successor": True,
            },
            [],
        )


def test_legacy_pre_gym_resolution_is_a_strict_noop():
    class _BombRegistry:
        def spec(self, _task_id):
            raise AssertionError("legacy path inspected Gym registration")

    assert (
        train_mod._resolve_action_ball_full_mdp_pre_gym_binding(
            requested=False,
            task_id="legacy",
            gym_registry=_BombRegistry(),
            num_envs=0,
            checkpoint_path=object(),
            checkpoint_tolerant=False,
        )
        is None
    )


def test_fresh_resume_holds_before_gym_or_environment_reset():
    class _BombRegistry:
        def spec(self, _task_id):
            raise AssertionError("resume HOLD reached Gym allocation")

    with pytest.raises(RuntimeError, match="checkpoint resume remains HOLD"):
        train_mod._resolve_action_ball_full_mdp_pre_gym_binding(
            requested=True,
            task_id="HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0",
            gym_registry=_BombRegistry(),
            num_envs=2,
            checkpoint_path="/externally/pinned/r10.chk",
            checkpoint_tolerant=False,
        )


def test_fresh_task_requires_the_dedicated_gym_entry_point(monkeypatch):
    registry = _Registry(entry_point="isaaclab.envs:ManagerBasedRLEnv")
    with pytest.raises(RuntimeError, match="dedicated fresh Gym entry point"):
        _resolve(monkeypatch, gym_registry=registry)
    assert registry.spec_calls == [
        "HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0"
    ]


@pytest.mark.parametrize("family", ("A", "C"))
def test_pre_gym_accepts_only_the_two_code_owned_family_tasks(
    monkeypatch, family
):
    binding = _resolve(
        monkeypatch,
        task_id=f"HOPE-PingPong-ActionBall-FullMdp{family}-AgibotA3-v0",
    )
    assert binding.owner_type is _LeanOwner

    with pytest.raises(RuntimeError, match="exact code-owned A/C task id"):
        _resolve(
            monkeypatch,
            task_id="HOPE-PingPong-ActionBall-FullMdpForeign-AgibotA3-v0",
        )


def _fresh_reward_config_module():
    class _A:
        action_ball_full_mdp_family_role = "A"

    class _C:
        action_ball_full_mdp_family_role = "C"

    return types.SimpleNamespace(
        HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg=_A,
        HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg=_C,
    )


@pytest.mark.parametrize("family", ("A", "C"))
def test_fresh_reward_graph_never_calls_legacy_pack_or_mutates_terms(
    monkeypatch, family
):
    module = _fresh_reward_config_module()
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda _name: module,
    )
    monkeypatch.setattr(
        train_mod,
        "_expand_reward_pack",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fresh graph entered legacy reward-pack expander")
        ),
    )
    cfg_type = getattr(
        module,
        f"HOPEPingPongActionBallFullMdp{family}AgibotA3EnvCfg",
    )
    env_cfg = cfg_type()
    env_cfg.rewards = types.SimpleNamespace(marker=object())
    marker = env_cfg.rewards.marker
    applied = []
    task = {
        "gym_task": (
            f"HOPE-PingPong-ActionBall-FullMdp{family}-AgibotA3-v0"
        ),
        "action_ball_full_mdp_family_role": family,
        "action_ball_full_mdp_runtime": True,
    }
    assert (
        train_mod._resolve_reward_override_mapping(
            env_cfg, task, None, applied
        )
        is None
    )
    assert env_cfg.rewards.marker is marker
    assert applied == [
        "rewards=fresh full-MDP numeric graph "
        f"(family={family}; legacy reward_pack expansion not applicable)"
    ]


def test_legacy_reward_graph_still_uses_exact_pack_expander(monkeypatch):
    module = _fresh_reward_config_module()
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda _name: module,
    )
    calls = []
    result = object()

    def expand(env_cfg, task, rw, applied):
        calls.append((env_cfg, task, rw, applied))
        return result

    monkeypatch.setattr(train_mod, "_expand_reward_pack", expand)
    env_cfg = types.SimpleNamespace()
    task = {"gym_task": "HOPE-PingPong-ActionBall-A211Learnability-v0"}
    rw = {"reward_pack": "v2"}
    applied = []
    assert (
        train_mod._resolve_reward_override_mapping(
            env_cfg, task, rw, applied
        )
        is result
    )
    assert calls == [(env_cfg, task, rw, applied)]


def test_foreign_task_cannot_borrow_exact_fresh_reward_config(monkeypatch):
    module = _fresh_reward_config_module()
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda _name: module,
    )
    env_cfg = module.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    with pytest.raises(train_mod._OverrideError, match="foreign task/config"):
        train_mod._resolve_reward_override_mapping(
            env_cfg,
            {
                "gym_task": "HOPE-PingPong-ActionBall-FullMdpForeign-v0",
                "action_ball_full_mdp_family_role": "A",
                "action_ball_full_mdp_runtime": True,
            },
            None,
            [],
        )


@pytest.mark.parametrize("family", ("A", "C"))
def test_fresh_reward_template_metadata_never_enters_legacy_receipts(
    monkeypatch, family
):
    module = _fresh_reward_config_module()
    legacy = types.SimpleNamespace(
        build_reward_backend_compatibility_receipt=lambda _cfg: (
            _ for _ in ()
        ).throw(AssertionError("fresh metadata entered backend enumerator")),
        build_effective_reward_receipt=lambda _cfg, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("fresh metadata entered effective enumerator")),
    )

    def import_module(name):
        if name.endswith("hope_env_cfg"):
            return module
        if name.endswith("effective_reward_recipe"):
            return legacy
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    cfg_type = getattr(
        module,
        f"HOPEPingPongActionBallFullMdp{family}AgibotA3EnvCfg",
    )
    env_cfg = cfg_type()
    metadata = types.SimpleNamespace(
        schema_version=1,
        kind="template",
        status="awaiting_numeric_authority",
        terms=tuple(object() for _ in range(14)),
    )
    env_cfg.rewards = metadata
    task = {
        "gym_task": (
            f"HOPE-PingPong-ActionBall-FullMdp{family}-AgibotA3-v0"
        ),
        "action_ball_full_mdp_family_role": family,
        "action_ball_full_mdp_runtime": True,
    }
    assert train_mod._build_reward_backend_compatibility_receipt_for_training(
        env_cfg, task
    ) == (family, None)
    assert (
        train_mod._build_effective_reward_receipt_for_training(
            env_cfg, types.SimpleNamespace(), task=task
        )
        is None
    )
    assert env_cfg.rewards is metadata
    assert metadata.schema_version == 1
    assert len(metadata.terms) == 14


def test_legacy_reward_receipts_still_use_the_exact_enumerators(monkeypatch):
    config = _fresh_reward_config_module()
    backend = object()
    effective = object()
    calls = []
    legacy = types.SimpleNamespace(
        build_reward_backend_compatibility_receipt=lambda cfg: (
            calls.append(("backend", cfg)) or backend
        ),
        build_effective_reward_receipt=lambda cfg, **kwargs: (
            calls.append(("effective", cfg, kwargs)) or effective
        ),
    )

    def import_module(name):
        if name.endswith("hope_env_cfg"):
            return config
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        legacy,
    )
    env_cfg = types.SimpleNamespace()
    task = {"gym_task": "HOPE-PingPong-ActionBall-A211Learnability-v0"}
    assert train_mod._build_reward_backend_compatibility_receipt_for_training(
        env_cfg, task
    ) == (None, backend)
    assert (
        train_mod._build_effective_reward_receipt_for_training(
            env_cfg, types.SimpleNamespace(), task=task
        )
        is effective
    )
    assert calls == [
        ("backend", env_cfg),
        ("effective", env_cfg, {}),
    ]


def test_foreign_task_cannot_skip_either_legacy_reward_receipt(monkeypatch):
    module = _fresh_reward_config_module()
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda _name: module,
    )
    env_cfg = module.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    foreign = {
        "gym_task": "HOPE-PingPong-ActionBall-FullMdpForeign-v0",
        "action_ball_full_mdp_family_role": "A",
        "action_ball_full_mdp_runtime": True,
    }
    for build in (
        lambda: train_mod._build_reward_backend_compatibility_receipt_for_training(
            env_cfg, foreign
        ),
        lambda: train_mod._build_effective_reward_receipt_for_training(
            env_cfg, types.SimpleNamespace(), task=foreign
        ),
    ):
        with pytest.raises(train_mod._OverrideError, match="foreign task/config"):
            build()


def _fresh_motion_catalog_fixture(monkeypatch, *, family="A"):
    config_module = _fresh_reward_config_module()

    class _Table:
        pass

    table = _Table()
    table.action_order = tuple(f"action_{index:02d}" for index in range(73))
    table.motion_files = tuple(f"/code/motion_{index:02d}.npz" for index in range(73))
    table.motion_sha256 = tuple(f"{index + 1:064x}" for index in range(73))
    table.manifest_file_sha256 = "a" * 64
    table.manifest_canonical_sha256 = "b" * 64
    kind = "action_ball_full_mdp_code_owned_diagnostic_catalog_v1"
    cfg_type = getattr(
        config_module,
        f"HOPEPingPongActionBallFullMdp{family}AgibotA3EnvCfg",
    )
    env_cfg = cfg_type()
    motion = types.SimpleNamespace(
        motion_file=table.motion_files,
        action_ball_full_mdp_diagnostic_catalog=kind,
    )
    racket = types.SimpleNamespace()
    env_cfg.commands = types.SimpleNamespace(motion=motion, racket_target=racket)
    calls = []

    def require_bindings(actual_motion, actual_racket):
        calls.append((actual_motion, actual_racket))
        assert actual_motion is motion
        assert actual_racket is racket
        return table

    commands_module = types.SimpleNamespace(
        ActionBallFullMdpDiagnosticCatalogTable=_Table,
        ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND=kind,
        require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings=(
            require_bindings
        ),
    )

    def import_module(name):
        if name.endswith("hope_env_cfg"):
            return config_module
        if name.endswith("mdp.commands"):
            return commands_module
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    monkeypatch.setattr(
        train_mod, "_ORIGINAL_TRAINING_ARGV", ("python", "train.py")
    )
    task = {
        "gym_task": (
            f"HOPE-PingPong-ActionBall-FullMdp{family}-AgibotA3-v0"
        ),
        "action_ball_full_mdp_family_role": family,
        "action_ball_full_mdp_runtime": True,
        "registry_name": None,
        "registry_name_2": None,
    }
    cfg = {
        "motion_file": None,
        "motion_file_2": None,
        "registry_name": None,
        "registry_name_2": None,
    }
    return cfg, env_cfg, task, table, calls


@pytest.mark.parametrize("family", ("A", "C"))
def test_fresh_motion_route_preserves_code_owned_order_and_skips_legacy(
    monkeypatch, family
):
    cfg, env_cfg, task, table, calls = _fresh_motion_catalog_fixture(
        monkeypatch, family=family
    )
    monkeypatch.setattr(
        train_mod,
        "resolve_motion_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fresh catalog entered legacy motion resolver")
        ),
    )
    original = env_cfg.commands.motion.motion_file
    files, registries, status = train_mod._resolve_motion_sources_for_training(
        cfg, env_cfg, task
    )
    assert files is original is table.motion_files
    assert registries == ()
    assert calls == [
        (env_cfg.commands.motion, env_cfg.commands.racket_target)
    ]
    assert status == {
        "family": family,
        "kind": "action_ball_full_mdp_code_owned_diagnostic_catalog_v1",
        "action_count": 73,
        "manifest_file_sha256": "a" * 64,
        "manifest_canonical_sha256": "b" * 64,
        "first_action": "action_00",
        "last_action": "action_72",
        "diagnostic_unauthorized": True,
    }


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("cfg", "motion_file", ["/caller/motion.npz"]),
        ("cfg", "motion_file_2", "/caller/second.npz"),
        ("cfg", "registry_name", "caller/registry"),
        ("cfg", "registry_name_2", "caller/registry2"),
        ("task", "motion_file", "/task/motion.npz"),
        ("task", "motion_file_2", "/task/second.npz"),
        ("task", "registry_name", "task/registry"),
        ("task", "registry_name_2", "task/registry2"),
    ],
)
def test_fresh_motion_route_rejects_every_configured_override(
    monkeypatch, owner, field, value
):
    cfg, env_cfg, task, _table, calls = _fresh_motion_catalog_fixture(
        monkeypatch
    )
    if owner == "cfg":
        cfg[field] = value
    else:
        task[field] = value
    with pytest.raises(
        train_mod._OverrideError, match="forbids caller/task motion"
    ):
        train_mod._resolve_motion_sources_for_training(cfg, env_cfg, task)
    assert calls == []


@pytest.mark.parametrize(
    "arg",
    [
        "motion_file=null",
        "+motion_file_2=/caller/second.npz",
        "registry_name=caller/registry",
        "task.registry_name_2=null",
    ],
)
def test_fresh_motion_route_rejects_explicit_cli_override_even_if_null(
    monkeypatch, arg
):
    cfg, env_cfg, task, _table, calls = _fresh_motion_catalog_fixture(
        monkeypatch
    )
    monkeypatch.setattr(
        train_mod,
        "_ORIGINAL_TRAINING_ARGV",
        ("python", "train.py", arg),
    )
    with pytest.raises(train_mod._OverrideError, match="argv="):
        train_mod._resolve_motion_sources_for_training(cfg, env_cfg, task)
    assert calls == []


def test_legacy_motion_route_is_unchanged(monkeypatch):
    config_module = _fresh_reward_config_module()
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda name: config_module
        if name.endswith("hope_env_cfg")
        else (_ for _ in ()).throw(AssertionError(f"unexpected import {name}")),
    )
    cfg = types.SimpleNamespace()
    env_cfg = types.SimpleNamespace()
    task = {"gym_task": "legacy"}
    expected = (["legacy.npz"], ["legacy/registry"])
    calls = []

    def resolve(actual):
        calls.append(actual)
        return expected

    monkeypatch.setattr(train_mod, "resolve_motion_sources", resolve)
    assert train_mod._resolve_motion_sources_for_training(
        cfg, env_cfg, task
    ) == (*expected, None)
    assert calls == [cfg]


def test_only_direct_lean_owner_remains_constructible(monkeypatch):
    retired_owner = (
        HERE.parent
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_full_mdp_runtime_owner.py"
    )
    assert not retired_owner.exists()
    binding = _resolve(monkeypatch)
    assert binding.owner_type is _LeanOwner


def test_diagnostic_pre_gym_source_does_not_read_post_gym_flags():
    source = inspect.getsource(
        train_mod._resolve_action_ball_full_mdp_pre_gym_binding
    )
    for forbidden in (
        "action_ball_full_mdp_runtime_owner",
        "launch_authorized =",
        "diagnostic_operational",
        "runtime_integrated",
        "post_physics_integrated",
        "selected_reset_integrated",
        "ppo_drain_bindings_integrated",
        "dependency_sources_frozen",
    ):
        assert forbidden not in source


def test_pre_gym_binding_uses_only_code_owned_lean_factory(monkeypatch):
    binding = _resolve(monkeypatch)
    assert type(binding) is train_mod._ActionBallFullMdpPreGymBinding
    assert binding.owner_type is _LeanOwner
    assert binding.owner_factory.__self__ is _LeanOwner
    assert binding.owner_factory.__func__ is _LeanOwner.create_from_env.__func__
    assert binding.dependency_kind == "action_ball_epoch_runtime_dependencies_v1"
    assert binding.epoch_owner_type is not None
    assert binding.gym_entry_point == train_mod._ACTION_BALL_FULL_MDP_GYM_ENTRY_POINT
    with pytest.raises(FrozenInstanceError):
        binding.owner_type = object


def test_full_mdp_typed_ppo_v2_replaces_only_the_fresh_algo_mapping():
    legacy = {
        "name": "ppo",
        "runner": {"num_steps_per_env": 24, "max_iterations": 25_000},
        "policy": {"init_noise_std": 1.0},
        "algorithm": {"lam": 0.95, "num_mini_batches": 4},
    }
    recipe = train_mod._apply_action_ball_full_mdp_ppo_recipe(
        legacy, requested=True
    )
    assert legacy["name"] == "ppo"
    assert legacy["runner"] == {
        "num_steps_per_env": 48,
        "max_iterations": 12_500,
        "save_interval": 500,
        "empirical_normalization": False,
    }
    assert legacy["policy"] == recipe.policy()
    assert legacy["algorithm"] == recipe.algorithm()
    assert legacy["algorithm"]["lam"] == 0.98
    assert legacy["algorithm"]["num_mini_batches"] == 8

    class _AgentCfg:
        def to_dict(self):
            return {
                **legacy["runner"],
                "policy": legacy["policy"],
                "algorithm": legacy["algorithm"],
            }

    serialized = train_mod._task_first_agent_recipe(_AgentCfg())
    assert serialized["recipe"] == recipe.learning_recipe()
    assert serialized["sha256"] == recipe.learning_recipe_sha256()


def test_full_mdp_ppo_cli_preflight_rejects_competing_schedule_before_kit(
    monkeypatch,
):
    cfg = {
        "task": {"action_ball_full_mdp_runtime": True},
        "max_iterations": 12_500,
    }
    monkeypatch.setattr(
        train_mod,
        "_ORIGINAL_TRAINING_ARGV",
        ("python", "train.py", "task=FullMdpA"),
    )
    assert train_mod._preflight_action_ball_full_mdp_ppo_cli(cfg) is None

    cfg["max_iterations"] = 5
    with pytest.raises(train_mod._OverrideError, match="code-owned at 12500"):
        train_mod._preflight_action_ball_full_mdp_ppo_cli(cfg)

    cfg["max_iterations"] = 12_500.0
    with pytest.raises(train_mod._OverrideError, match="exact integer"):
        train_mod._preflight_action_ball_full_mdp_ppo_cli(cfg)

    cfg["max_iterations"] = 12_500
    monkeypatch.setattr(
        train_mod,
        "_ORIGINAL_TRAINING_ARGV",
        (
            "python",
            "train.py",
            "task=FullMdpA",
            "algo.runner.max_iterations=5",
        ),
    )
    with pytest.raises(train_mod._OverrideError, match="nested Hydra PPO"):
        train_mod._preflight_action_ball_full_mdp_ppo_cli(cfg)


def test_legacy_ppo_cli_preflight_does_not_claim_the_typed_recipe(monkeypatch):
    monkeypatch.setattr(
        train_mod,
        "_ORIGINAL_TRAINING_ARGV",
        ("python", "train.py", "algo.runner.max_iterations=5"),
    )
    assert train_mod._preflight_action_ball_full_mdp_ppo_cli(
        {"task": {"action_ball_full_mdp_runtime": False}, "max_iterations": 5}
    ) is None


def test_legacy_algo_does_not_load_or_apply_the_full_mdp_recipe(monkeypatch):
    algo = {"runner": {"num_steps_per_env": 24}}
    monkeypatch.setattr(
        train_mod,
        "_action_ball_full_mdp_ppo_recipe_module",
        lambda: (_ for _ in ()).throw(AssertionError("legacy loaded recipe")),
    )
    assert (
        train_mod._apply_action_ball_full_mdp_ppo_recipe(
            algo, requested=False
        )
        is None
    )
    assert algo == {"runner": {"num_steps_per_env": 24}}


def test_full_mdp_hard_contract_reuses_existing_agent_recipe_serializer():
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert '"action_ball_full_mdp_ppo_runner_recipe"' in source
    assert "_task_first_agent_recipe(agent_cfg)" in source


@pytest.mark.parametrize("num_envs", [1, 2, 64, 4096])
def test_single_action_lean_binding_accepts_every_positive_n(
    monkeypatch, num_envs
):
    binding = _resolve(monkeypatch, num_envs=num_envs)
    assert binding.owner_type is _LeanOwner


@pytest.mark.parametrize(
    "raw_num_envs",
    [True, False, 0, -1, 1.0, "2", None],
)
def test_fresh_num_envs_rejects_coercible_or_nonpositive_raw_values(
    raw_num_envs,
):
    with pytest.raises(RuntimeError, match="before any cast"):
        train_mod._resolve_training_num_envs(
            raw_num_envs,
            action_ball_full_mdp_requested=True,
        )


def test_fresh_num_envs_rejects_numpy_integer_before_cast():
    numpy = pytest.importorskip("numpy")
    with pytest.raises(RuntimeError, match="before any cast"):
        train_mod._resolve_training_num_envs(
            numpy.int64(64),
            action_ball_full_mdp_requested=True,
        )


@pytest.mark.parametrize("num_envs", [1, 2, 64, 4096])
def test_fresh_num_envs_preserves_exact_python_int_identity(num_envs):
    assert (
        train_mod._resolve_training_num_envs(
            num_envs,
            action_ball_full_mdp_requested=True,
        )
        is num_envs
    )


def test_caller_cannot_select_or_self_assert_the_diagnostic_mode(monkeypatch):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _resolve(
            monkeypatch,
            run_mode="single_action_lean",
        )


def test_pre_gym_rejects_a_lean_module_that_claims_launch_authority(monkeypatch):
    lean = _lean_module()
    lean.LAUNCH_AUTHORIZED = True
    with pytest.raises(RuntimeError, match="lean runtime owner API differs"):
        _resolve(monkeypatch, lean_module=lean)


def test_pre_gym_binding_rejects_a_static_factory_shim(monkeypatch):
    class _StaticOwner:
        create_from_env = staticmethod(lambda env, lease: (env, lease))

    lean = _lean_module()
    lean.ActionBallFullMdpLeanRuntimeOwner = _StaticOwner
    real_import = train_mod.importlib.import_module

    def import_module(name):
        if name.endswith("action_ball_full_mdp_lean_runtime"):
            return lean
        return real_import(name)

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="exact bound classmethod"):
        train_mod._resolve_action_ball_full_mdp_pre_gym_binding(
            requested=True,
            task_id="HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0",
            gym_registry=_Registry(),
            num_envs=2,
            checkpoint_path=None,
            checkpoint_tolerant=False,
        )


def _installed_runtime(binding):
    owner_type = binding.owner_type
    adapter = None
    env = types.SimpleNamespace()
    lease = object()
    owner = object.__new__(owner_type)
    owner.full_mdp_runtime_env = env
    owner.full_mdp_runtime_lease = lease
    owner.diagnostic_dependency_kind = binding.dependency_kind
    owner.diagnostic_unauthorized = True
    owner.launch_authorized = False
    lean_module = _lean_module()
    epoch_module = lean_module.epoch_v1
    assert binding.epoch_owner_type is epoch_module.ActionEpochOwner
    owner._epoch = epoch_module.ActionEpochOwner()
    owner._epoch.num_envs = 2
    owner.epoch_owner = owner._epoch
    owner._test_lean_module = lean_module
    owner._ppo_drain = owner
    owner.action_ball_r10_checkpoint_adapter = adapter
    env.full_mdp_runtime_owner = owner
    env.action_ball_full_mdp_runtime_lease = lease
    env.action_ball_full_mdp_ppo_drain_owner = types.MethodType(
        lambda runtime, supplied_lease: owner._ppo_drain
        if runtime is env and supplied_lease is lease
        else None,
        env,
    )
    env.action_ball_r10_checkpoint_adapter = adapter
    env.action_ball_r10_cold_restore_capsule = None
    env.num_envs = 2
    return env, owner, adapter


def test_runtime_and_runner_inputs_preserve_one_owner_and_adapter_identity(
    monkeypatch,
):
    binding = _resolve(monkeypatch)
    env, owner, adapter = _installed_runtime(binding)
    real_import = train_mod.importlib.import_module
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda name: owner._test_lean_module
        if name.endswith("action_ball_full_mdp_lean_runtime")
        else real_import(name),
    )
    resolved = train_mod._resolve_action_ball_full_mdp_runtime_binding(
        binding, env
    )
    assert resolved.runtime_owner is owner
    assert resolved.ppo_drain_owner is owner._ppo_drain
    assert resolved.epoch_owner is owner.epoch_owner
    assert resolved.checkpoint_adapter is adapter is None
    assert resolved.cold_restore_capsule is None
    contract = train_mod._action_ball_full_mdp_training_contract(
        binding,
        resolved.runtime_owner,
        resolved.checkpoint_adapter,
        resolved.cold_restore_capsule,
    )
    assert contract == {
        "schema_version": 1,
        "kind": "action_ball_full_mdp_train_wiring_v1",
        "gym_entry_point": train_mod._ACTION_BALL_FULL_MDP_GYM_ENTRY_POINT,
        "target_mode": "action_ball_full_mdp",
        "actor_obs_mode": "action_ball_full_mdp",
        "runtime_owner_type": f"{_LeanOwner.__module__}.{_LeanOwner.__qualname__}",
        "r10_checkpoint_adapter_bound": False,
        "cold_restore": False,
        "launch_authorized": False,
        "diagnostic_unauthorized": True,
        "joint_safety_evidence_mode": (
            "diagnostic_compact_two_phase_update_v1"
        ),
        "formal_evidence_prohibited": True,
        "curriculum_promotion_prohibited": True,
        "exact_export_prohibited": True,
        "deployment_prohibited": True,
        "run_mode": "single_action_lean",
        "no_save": True,
        "diagnostic_operational": True,
        "runtime_dependency_kind": "action_ball_epoch_runtime_dependencies_v1",
    }


def _installed_lean_actor_observation(monkeypatch, binding):
    assert binding.owner_type is _LeanOwner
    env, owner, _adapter = _installed_runtime(binding)
    epoch_owner = owner.epoch_owner
    epoch_owner.device = "cpu"
    parts = {
        "r05_runtime": object(),
        "motion": object(),
        "racket": object(),
        "physical_ball": object(),
        "r06_landing_outcome": object(),
        "r03_strike_fact": object(),
        "r07_recovery": object(),
    }
    owner.component_identities = tuple(parts.items())

    class _Source:
        def __init__(self):
            self._env = env
            self._runtime_owner = owner
            self._epoch_owner = epoch_owner
            self._num_envs = 2
            self._device = epoch_owner.device
            self._components = {
                "motion_owner": parts["motion"],
                "racket_owner": parts["racket"],
                "physical_owner": parts["physical_ball"],
                "r03_owner": parts["r03_strike_fact"],
                "r06_owner": parts["r06_landing_outcome"],
                "r07_owner": parts["r07_recovery"],
            }

        @property
        def group_widths(self):
            return {"policy": 229, "critic": 399}

    def _term(_env, *, group):
        return (_env, group)

    actor_layout = (("actor_payload", 229),)
    critic_extension = (("critic_extension", 170),)
    lean_observations = types.SimpleNamespace(
        LeanActionEpochObservationSource=_Source,
        _term=_term,
        MANAGER_GROUP_ORDER=("policy", "critic"),
        ACTOR_LAYOUT_V1=actor_layout,
        CRITIC_EXTENSION_LAYOUT_V1=critic_extension,
        ACTOR_WIDTH_V1=229,
        CRITIC_WIDTH_V1=399,
        DIAGNOSTIC_UNAUTHORIZED=True,
        LAUNCH_AUTHORIZED=False,
    )

    class _ActorTerm:
        def __init__(self, name, dim, deploy_source, description):
            self.name = name
            self.dim = dim
            self.deploy_source = deploy_source
            self.description = description

    class _ActorContract:
        def __init__(self, *, name, obs_mode, total_dim, terms):
            self.name = name
            self.obs_mode = obs_mode
            self.total_dim = total_dim
            self.terms = terms

    actor_contracts = types.SimpleNamespace(
        ActorObservationContract=_ActorContract,
        ActorObservationTerm=_ActorTerm,
    )
    source = _Source()
    env._action_ball_full_mdp_lean_observation_source = source
    env._action_ball_full_mdp_components = types.SimpleNamespace(
        observation_source=source,
        lean_runtime_owner=owner,
        epoch_owner=epoch_owner,
    )
    env.observation_manager = types.SimpleNamespace(
        active_terms={"policy": ["action_epoch"], "critic": ["action_epoch"]},
        group_obs_term_dim={"policy": [(229,)], "critic": [(399,)]},
        group_obs_dim={"policy": (229,), "critic": (399,)},
        group_obs_concatenate={"policy": True, "critic": True},
        _group_obs_term_cfgs={
            "policy": [
                types.SimpleNamespace(
                    func=_term,
                    params={"group": "policy"},
                    noise=None,
                    history_length=0,
                )
            ],
            "critic": [
                types.SimpleNamespace(
                    func=_term,
                    params={"group": "critic"},
                    noise=None,
                    history_length=0,
                )
            ],
        },
    )
    runtime_binding = train_mod._ActionBallFullMdpRuntimeBinding(
        runtime_owner=owner,
        runtime_lease=env.action_ball_full_mdp_runtime_lease,
        ppo_drain_owner=owner,
        epoch_owner=epoch_owner,
        checkpoint_adapter=None,
        cold_restore_capsule=None,
    )
    def diagnostic_facts(supplied_env, actor_contract):
        assert actor_contract is None
        manager = supplied_env.observation_manager
        policy = manager._group_obs_term_cfgs["policy"][0]
        critic = manager._group_obs_term_cfgs["critic"][0]
        if (
            supplied_env is not env
            or manager.active_terms
            != {"policy": ["action_epoch"], "critic": ["action_epoch"]}
            or manager.group_obs_term_dim
            != {"policy": [(229,)], "critic": [(399,)]}
            or manager.group_obs_dim
            != {"policy": (229,), "critic": (399,)}
            or policy.params != {"group": "policy"}
            or critic.params != {"group": "critic"}
            or supplied_env._action_ball_full_mdp_lean_observation_source
            is not source
            or supplied_env._action_ball_full_mdp_components.observation_source
            is not source
            or source._runtime_owner is not owner
            or source._epoch_owner is not epoch_owner
            or source._components["r06_owner"]
            is not dict(owner.component_identities)["r06_landing_outcome"]
            or source._components["motion_owner"]
            is not dict(owner.component_identities)["motion"]
        ):
            raise RuntimeError("foreign diagnostic observation")
        return {
            "actor_obs_contract": "action_ball_full_mdp_action_epoch_v1",
            "actor_obs_mode": "action_ball_full_mdp",
            "actor_obs_total_dim": 229,
            "actor_obs_term_names": ["action_epoch"],
            "actor_obs_term_dims": [229],
            "critic_obs_contract": (
                "action_ball_full_mdp_action_epoch_critic_v1"
            ),
            "critic_obs_total_dim": 399,
            "critic_obs_term_names": ["action_epoch"],
            "critic_obs_term_dims": [399],
            "fresh_full_mdp_observation_kind": (
                "action_ball_full_mdp_action_epoch_observation_v1"
            ),
            "fresh_full_mdp_diagnostic_unauthorized": True,
            "fresh_full_mdp_launch_authorized": False,
            "fresh_full_mdp_no_capacity_receipt_or_sha_authority": True,
        }

    lean_observations.installed_observation_facts = (
        lambda supplied_env: diagnostic_facts(
            supplied_env, actor_contract=None
        )
    )
    real_import = train_mod.importlib.import_module

    def import_module(name):
        if name.endswith("action_ball_full_mdp_lean_observation_cfg"):
            return lean_observations
        if name.endswith("actor_observation_contract"):
            return actor_contracts
        return real_import(name)

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    return types.SimpleNamespace(
        env=env,
        owner=owner,
        epoch_owner=epoch_owner,
        source=source,
        source_type=_Source,
        parts=parts,
        runtime_binding=runtime_binding,
        lean_observations=lean_observations,
        actor_contract_type=_ActorContract,
    )


def test_diagnostic_actor_contract_uses_only_installed_action_epoch_source(
    monkeypatch,
):
    binding = _resolve(monkeypatch)
    state = _installed_lean_actor_observation(monkeypatch, binding)
    contract = (
        train_mod._resolve_action_ball_full_mdp_lean_actor_observation_contract(
            binding, state.runtime_binding, state.env
        )
    )
    assert type(contract) is state.actor_contract_type
    assert contract.name == "action_ball_full_mdp_action_epoch_v1"
    assert contract.obs_mode == "action_ball_full_mdp"
    assert contract.total_dim == 229
    assert tuple((term.name, term.dim) for term in contract.terms) == (
        ("action_epoch", 229),
    )
    assert not hasattr(state.parts["r06_landing_outcome"], "capacity_authority")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.env.observation_manager.active_terms["policy"].append(
            "foreign"
        ),
        lambda state: state.env.observation_manager.group_obs_term_dim.__setitem__(
            "policy", [(228,)]
        ),
        lambda state: state.env.observation_manager.group_obs_dim.__setitem__(
            "critic", (398,)
        ),
        lambda state: state.env.observation_manager._group_obs_term_cfgs[
            "critic"
        ][0].params.__setitem__("source", state.source_type()),
        lambda state: setattr(state.source, "_runtime_owner", object()),
        lambda state: setattr(state.source, "_epoch_owner", object()),
        lambda state: state.source._components.__setitem__(
            "r06_owner", object()
        ),
        lambda state: setattr(
            state.owner,
            "component_identities",
            tuple(
                (name, object() if name == "motion" else value)
                for name, value in state.owner.component_identities
            ),
        ),
    ],
)
def test_diagnostic_actor_contract_rejects_foreign_manager_or_source(
    monkeypatch, mutation
):
    binding = _resolve(monkeypatch)
    state = _installed_lean_actor_observation(monkeypatch, binding)
    mutation(state)
    with pytest.raises(RuntimeError):
        train_mod._resolve_action_ball_full_mdp_lean_actor_observation_contract(
            binding, state.runtime_binding, state.env
        )


def _installed_lean_motion_body_order(monkeypatch):
    class _MotionCfg:
        pass

    class _Indexes:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return list(self.values)

    class _Motion:
        pass

    class _Components:
        pass

    class _Env:
        pass

    env = _Env()
    lease = object()
    owner = object.__new__(_LeanOwner)
    epoch = object()
    cfg = _MotionCfg()
    cfg.class_type = _Motion
    cfg.asset_name = "robot"
    cfg.body_names = ["root", "arm", "wrist"]
    robot = types.SimpleNamespace(
        body_names=["root", "leg", "arm", "wrist"]
    )
    motion = _Motion()
    motion.cfg = cfg
    motion.robot = robot
    motion.body_indexes = _Indexes([0, 2, 3])
    components = _Components()
    components.epoch_owner = epoch
    components.device_r05_owner = object()
    components.motion_owner = motion
    components.racket_owner = object()
    components.physical_owner = object()
    components.r06_owner = object()
    components.r03_owner = object()
    components.r07_owner = object()
    components.lean_runtime_owner = owner
    owner.full_mdp_runtime_env = env
    owner.full_mdp_runtime_lease = lease
    owner.epoch_owner = epoch
    owner.diagnostic_unauthorized = True
    owner.launch_authorized = False
    owner.component_identities = (
        ("r05_runtime", components.device_r05_owner),
        ("motion", motion),
        ("racket", components.racket_owner),
        ("physical_ball", components.physical_owner),
        ("r06_landing_outcome", components.r06_owner),
        ("r03_strike_fact", components.r03_owner),
        ("r07_recovery", components.r07_owner),
    )
    env.full_mdp_runtime_owner = owner
    env.action_ball_full_mdp_runtime_lease = lease
    env._action_ball_full_mdp_components = components
    env.command_manager = types.SimpleNamespace(
        get_term=lambda name: motion if name == "motion" else None
    )
    env.cfg = types.SimpleNamespace(
        commands=types.SimpleNamespace(motion=cfg)
    )
    env.scene = {"robot": robot}

    modules = {
        "action_ball_full_mdp_lean_runtime": types.SimpleNamespace(
            ActionBallFullMdpLeanRuntimeOwner=_LeanOwner
        ),
        "full_mdp_env": types.SimpleNamespace(
            FullMdpLeanRuntimeComponents=_Components
        ),
        "mdp.commands": types.SimpleNamespace(
            MotionCommand=_Motion, MotionCommandCfg=_MotionCfg
        ),
    }
    real_import = train_mod.importlib.import_module

    def import_module(name):
        for suffix, module in modules.items():
            if name.endswith(suffix):
                return module
        return real_import(name)

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    return types.SimpleNamespace(
        env=env,
        owner=owner,
        motion=motion,
        components=components,
        motion_cfg_type=_MotionCfg,
        motion_type=_Motion,
    )


def test_diagnostic_motion_body_order_uses_exact_installed_motion(monkeypatch):
    state = _installed_lean_motion_body_order(monkeypatch)
    assert not hasattr(state.env, "action_ball_full_mdp_motion_owner")
    assert train_mod._diagnostic_full_mdp_motion_body_names_contract(
        state.env
    ) == {
        "schema_version": 1,
        "kind": "action_ball_full_mdp_diagnostic_motion_body_order_v1",
        "ordered_body_names": ["root", "arm", "wrist"],
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "no_reward_term_receipt_or_sha_authority": True,
    }


def test_diagnostic_motion_body_order_accepts_manager_cfg_copy(monkeypatch):
    state = _installed_lean_motion_body_order(monkeypatch)
    copied = state.motion_cfg_type()
    copied.class_type = state.motion_type
    copied.asset_name = state.motion.cfg.asset_name
    copied.body_names = list(state.motion.cfg.body_names)
    state.env.cfg.commands.motion = copied
    contract = train_mod._diagnostic_full_mdp_motion_body_names_contract(
        state.env
    )
    assert contract["ordered_body_names"] == ["root", "arm", "wrist"]


@pytest.mark.parametrize("mutation", ("body_names", "order", "asset", "class_type"))
def test_diagnostic_motion_body_order_rejects_foreign_cfg_copy(
    monkeypatch, mutation
):
    state = _installed_lean_motion_body_order(monkeypatch)
    copied = state.motion_cfg_type()
    copied.class_type = state.motion_type
    copied.asset_name = state.motion.cfg.asset_name
    copied.body_names = list(state.motion.cfg.body_names)
    state.env.cfg.commands.motion = copied
    if mutation == "body_names":
        copied.body_names[-1] = "foreign"
    elif mutation == "order":
        copied.body_names[1:] = reversed(copied.body_names[1:])
    elif mutation == "asset":
        copied.asset_name = "foreign_robot"
    else:
        copied.class_type = object
    with pytest.raises(RuntimeError):
        train_mod._diagnostic_full_mdp_motion_body_names_contract(state.env)


@pytest.mark.parametrize("mutation", ("foreign", "reordered", "missing"))
def test_diagnostic_motion_body_order_rejects_bad_component_join(
    monkeypatch, mutation
):
    state = _installed_lean_motion_body_order(monkeypatch)
    if mutation == "foreign":
        state.components.motion_owner = object()
    elif mutation == "reordered":
        state.motion.cfg.body_names[:] = ["root", "wrist", "arm"]
    else:
        state.owner.component_identities = tuple(
            item for item in state.owner.component_identities if item[0] != "motion"
        )
    with pytest.raises(RuntimeError):
        train_mod._diagnostic_full_mdp_motion_body_names_contract(state.env)


@pytest.mark.parametrize("bad_symbol", (None, object()))
def test_diagnostic_motion_body_order_rejects_missing_or_nonclass_cfg_symbol(
    monkeypatch, bad_symbol
):
    state = _installed_lean_motion_body_order(monkeypatch)
    real_import = train_mod.importlib.import_module

    def import_module(name):
        module = real_import(name)
        if name.endswith("mdp.commands"):
            return types.SimpleNamespace(
                MotionCommand=type(state.motion),
                MotionCommandCfg=bad_symbol,
            )
        return module

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError):
        train_mod._diagnostic_full_mdp_motion_body_names_contract(state.env)


def _lean_reward20_status(monkeypatch):
    names = tuple(f"manager_{index}" for index in range(20))
    module = types.SimpleNamespace(MANAGER_NAMES=names, MANAGER_TERM_COUNT=20)
    real_import = train_mod.importlib.import_module
    monkeypatch.setattr(
        train_mod.importlib,
        "import_module",
        lambda name: module
        if name.endswith("action_ball_full_mdp_lean_rewards")
        else real_import(name),
    )
    return {
        "kind": "action_ball_epoch_lean_reward_graph_v1",
        "ordered_manager_names": list(names),
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "no_receipt_or_sha_authority": True,
    }


def test_diagnostic_racket_guidance_is_covered_by_exact_reward20(monkeypatch):
    status = _lean_reward20_status(monkeypatch)
    contract = train_mod._diagnostic_full_mdp_racket_guidance_contract(status)
    assert contract == {
        "schema_version": 2,
        "kind": "action_ball_full_mdp_racket_guidance_in_lean_reward20_v2",
        "covered_by_reward_graph_kind": status["kind"],
        "ordered_manager_names": status["ordered_manager_names"],
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
    }


@pytest.mark.parametrize("mutation", ("reordered", "missing"))
def test_diagnostic_racket_guidance_rejects_inexact_reward20_manager_order(
    monkeypatch, mutation
):
    status = _lean_reward20_status(monkeypatch)
    if mutation == "reordered":
        status["ordered_manager_names"][:2] = reversed(
            status["ordered_manager_names"][:2]
        )
    else:
        status["ordered_manager_names"].pop()
    with pytest.raises(RuntimeError):
        train_mod._diagnostic_full_mdp_racket_guidance_contract(status)


def _installed_lean_reward_graph(monkeypatch, binding):
    assert binding.owner_type is _LeanOwner

    class _LeanGraph:
        def __init__(self, epoch_owner):
            self.epoch_owner = epoch_owner
            self.num_envs = 2
            self.device = epoch_owner.device
            self._poisoned = False
            self._cycle_open = False

        @property
        def poisoned(self):
            return self._poisoned

        @property
        def cycle_open(self):
            return self._cycle_open

    manager_names = tuple(f"manager_{i}" for i in range(20))
    consumer_owners = ("r03",) * 10 + ("physical", "r06", "r06", "r07")
    consumers = tuple(
        f"{owner}:{name}"
        for owner, name in zip(consumer_owners, manager_names[:14])
    )
    calls = []

    class _LeanRewardEnv:
        def action_ball_full_mdp_lean_reward_graph(self, lease):
            calls.append(lease)
            if lease is not self.action_ball_full_mdp_runtime_lease:
                raise RuntimeError("foreign lease")
            return self._installed_graph

    env = _LeanRewardEnv()
    lease = object()
    owner = object.__new__(binding.owner_type)
    epoch_owner = binding.epoch_owner_type()
    epoch_owner.device = "cpu"
    epoch_owner.num_envs = 2
    owner.full_mdp_runtime_env = env
    owner.full_mdp_runtime_lease = lease
    owner.epoch_owner = epoch_owner
    owner.diagnostic_unauthorized = True
    owner.launch_authorized = False
    graph = _LeanGraph(epoch_owner)
    env.action_ball_full_mdp_runtime_lease = lease
    env.num_envs = 2
    env.full_mdp_runtime_owner = owner
    env.reward_manager = types.SimpleNamespace(
        active_terms=list(manager_names)
    )
    env._installed_graph = graph

    lean_rewards = types.SimpleNamespace(
        LeanActionEpochRewardGraph=_LeanGraph,
        GRAPH_ATTR="action_ball_full_mdp_lean_reward_graph",
        MANAGER_NAMES=manager_names,
        MANAGER_TERM_COUNT=20,
        ORDERED_CONSUMERS=consumers,
        LIFECYCLE_PAYMENT_COUNT=14,
        DIAGNOSTIC_N2_REWARD_PROFILE_KIND=(
            "action_ball_full_mdp_diagnostic_n2_reward_profile_v2"
        ),
        DIAGNOSTIC_UNAUTHORIZED=True,
        LAUNCH_AUTHORIZED=False,
    )
    real_import = train_mod.importlib.import_module

    def import_module(name):
        if name.endswith("action_ball_full_mdp_lean_rewards"):
            return lean_rewards
        if name.endswith("action_ball_full_mdp_runtime_factory"):
            raise AssertionError(
                "diagnostic lean graph imported the legacy factory validator"
            )
        return real_import(name)

    monkeypatch.setattr(train_mod.importlib, "import_module", import_module)
    return types.SimpleNamespace(
        env=env,
        owner=owner,
        epoch_owner=epoch_owner,
        graph=graph,
        graph_type=_LeanGraph,
        manager_names=manager_names,
        consumers=consumers,
        lean_rewards=lean_rewards,
        calls=calls,
    )


def test_diagnostic_post_gym_uses_only_exact_lean_graph_getter(monkeypatch):
    binding = _resolve(monkeypatch)
    state = _installed_lean_reward_graph(monkeypatch, binding)
    status = train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
        binding, state.env
    )
    assert state.calls == [state.env.action_ball_full_mdp_runtime_lease]
    assert status == {
        "schema_version": 1,
        "kind": "action_ball_epoch_lean_reward_graph_v1",
        "profile_kind": (
            "action_ball_full_mdp_diagnostic_n2_reward_profile_v2"
        ),
        "ordered_manager_names": list(state.manager_names),
        "ordered_payment_consumers": list(state.consumers),
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "no_receipt_or_sha_authority": True,
    }
    assert all("sha256" not in name for name in status)


def test_diagnostic_real_lean_reward_module_crosses_exact_20_term_abi(
    monkeypatch,
):
    try:
        importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_rewards"
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "whole_body_tracking",
            "isaaclab",
            "isaaclab_tasks",
        }:
            raise
        pytest.skip(f"real lean Reward dependency unavailable: {exc.name}")
    binding = _resolve(monkeypatch)
    with pytest.raises(RuntimeError, match="lacks the runtime lease"):
        train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
            binding, types.SimpleNamespace()
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda module: setattr(module, "MANAGER_TERM_COUNT", 14),
        lambda module: setattr(module, "LIFECYCLE_PAYMENT_COUNT", 20),
        lambda module: setattr(module, "MANAGER_NAMES", module.MANAGER_NAMES[:-1]),
        lambda module: setattr(
            module,
            "ORDERED_CONSUMERS",
            module.ORDERED_CONSUMERS[:-1] + ("reward:foreign",),
        ),
        lambda module: setattr(
            module,
            "DIAGNOSTIC_N2_REWARD_PROFILE_KIND",
            "action_ball_full_mdp_diagnostic_n2_reward_profile_v1",
        ),
    ),
)
def test_diagnostic_lean_reward_module_rejects_stale_14_term_abi(
    monkeypatch, mutation
):
    binding = _resolve(monkeypatch)
    state = _installed_lean_reward_graph(monkeypatch, binding)
    mutation(state.lean_rewards)
    with pytest.raises(RuntimeError, match="Reward module ABI differs"):
        train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
            binding, state.env
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: setattr(state.env, "_installed_graph", object()),
        lambda state: setattr(
            state.graph, "epoch_owner", state.graph_type(state.epoch_owner)
        ),
        lambda state: setattr(
            state.env.reward_manager,
            "active_terms",
            list(reversed(state.manager_names)),
        ),
        lambda state: setattr(
            state.owner, "full_mdp_runtime_env", object()
        ),
        lambda state: setattr(
            state.owner, "full_mdp_runtime_lease", object()
        ),
        lambda state: setattr(
            state.owner, "diagnostic_unauthorized", False
        ),
        lambda state: setattr(state.owner, "launch_authorized", True),
        lambda state: setattr(state.graph, "_poisoned", True),
        lambda state: setattr(state.graph, "_cycle_open", True),
    ],
)
def test_diagnostic_foreign_lean_reward_graph_fails_closed(
    monkeypatch, mutation
):
    binding = _resolve(monkeypatch)
    state = _installed_lean_reward_graph(monkeypatch, binding)
    mutation(state)
    with pytest.raises(RuntimeError):
        train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
            binding, state.env
        )


def test_diagnostic_rejects_instance_shadow_of_exact_lean_getter(monkeypatch):
    binding = _resolve(monkeypatch)
    state = _installed_lean_reward_graph(monkeypatch, binding)
    state.env.action_ball_full_mdp_lean_reward_graph = (
        lambda _lease: state.graph
    )
    with pytest.raises(RuntimeError, match="exact lease-bound"):
        train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
            binding, state.env
        )


def test_legacy_env_rejects_partial_fresh_installed_reward_graph():
    assert (
        train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
            None, types.SimpleNamespace()
        )
        is None
    )
    with pytest.raises(RuntimeError, match="legacy env acquired a partial"):
        train_mod._resolve_action_ball_full_mdp_installed_reward_graph(
            None,
            types.SimpleNamespace(
                action_ball_full_mdp_lean_reward_graph=lambda _lease: object()
            ),
        )


def test_runtime_binding_reads_each_env_authority_exactly_once(monkeypatch):
    binding = _resolve(monkeypatch)
    env, owner, adapter = _installed_runtime(binding)

    class _SingleReadRuntime:
        num_envs = 2

        def __init__(self):
            self.counts = {}

        def _read(self, name, value):
            self.counts[name] = self.counts.get(name, 0) + 1
            if self.counts[name] != 1:
                raise AssertionError(f"{name} was read more than once")
            return value

        @property
        def full_mdp_runtime_owner(self):
            return self._read("owner", owner)

        @property
        def action_ball_r10_checkpoint_adapter(self):
            return self._read("adapter", adapter)

        @property
        def action_ball_full_mdp_runtime_lease(self):
            return self._read("lease", env.action_ball_full_mdp_runtime_lease)

        @property
        def action_ball_r10_cold_restore_capsule(self):
            return self._read("capsule", None)

        def action_ball_full_mdp_ppo_drain_owner(self, lease):
            assert lease is env.action_ball_full_mdp_runtime_lease
            return self._read("drain", owner._ppo_drain)

    runtime = _SingleReadRuntime()
    owner.full_mdp_runtime_env = runtime
    owner.full_mdp_runtime_lease = env.action_ball_full_mdp_runtime_lease
    resolved = train_mod._resolve_action_ball_full_mdp_runtime_binding(
        binding, runtime
    )
    assert resolved.runtime_owner is owner
    assert resolved.checkpoint_adapter is None
    assert runtime.counts == {
        "owner": 1,
        "lease": 1,
        "adapter": 1,
        "capsule": 1,
        "drain": 1,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda env, _owner, _adapter: setattr(
            env, "full_mdp_runtime_owner", object()
        ),
        lambda env, owner, _adapter: setattr(
            owner, "diagnostic_dependency_kind", "foreign_epoch_kind"
        ),
        lambda env, owner, _adapter: setattr(
            owner, "action_ball_r10_checkpoint_adapter", object()
        ),
        lambda env, _owner, _adapter: setattr(
            env, "action_ball_r10_checkpoint_adapter", object()
        ),
        lambda env, _owner, _adapter: setattr(
            env, "action_ball_r10_cold_restore_capsule", object()
        ),
        lambda env, _owner, _adapter: setattr(
            env,
            "action_ball_full_mdp_ppo_drain_owner",
            types.MethodType(lambda _runtime, _lease: object(), env),
        ),
    ],
)
def test_partial_foreign_or_postconstructed_runtime_binding_fails_closed(
    monkeypatch, mutation
):
    binding = _resolve(monkeypatch)
    env, owner, adapter = _installed_runtime(binding)
    mutation(env, owner, adapter)
    with pytest.raises(RuntimeError):
        train_mod._resolve_action_ball_full_mdp_runtime_binding(binding, env)


def test_diagnostic_runtime_binding_rejects_foreign_epoch_identity(monkeypatch):
    binding = _resolve(monkeypatch)
    env, owner, _adapter = _installed_runtime(binding)
    owner.epoch_owner = object()
    with pytest.raises(RuntimeError, match="owner/epoch/drain join differs"):
        train_mod._resolve_action_ball_full_mdp_runtime_binding(binding, env)


def test_legacy_contract_remains_absent_and_rejects_partial_objects():
    assert (
        train_mod._action_ball_full_mdp_training_contract(
            None, None, None, None
        )
        is None
    )
    with pytest.raises(RuntimeError, match="legacy training acquired partial"):
        train_mod._action_ball_full_mdp_training_contract(
            None, object(), None, None
        )


def _runtime_modes(*, target_mode, obs_mode):
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            obs_mode=obs_mode,
            commands=types.SimpleNamespace(
                racket_target=types.SimpleNamespace(target_mode=target_mode)
            ),
        )
    )


def test_fresh_binding_requires_the_disjoint_full_mdp_runtime_modes(
    monkeypatch,
):
    binding = _resolve(monkeypatch)
    train_mod._validate_action_ball_full_mdp_runtime_modes(
        binding,
        _runtime_modes(
            target_mode="action_ball_full_mdp",
            obs_mode="action_ball_full_mdp",
        ),
    )
    for target_mode, obs_mode in (
        ("action_ball", "action_ball_a211"),
        ("action_ball_full_mdp", "action_ball_a211"),
        ("action_ball", "action_ball_full_mdp"),
    ):
        with pytest.raises(RuntimeError, match="legacy ActionBall modes"):
            train_mod._validate_action_ball_full_mdp_runtime_modes(
                binding,
                _runtime_modes(
                    target_mode=target_mode,
                    obs_mode=obs_mode,
                ),
            )


def test_full_mdp_modes_cannot_enter_the_legacy_train_branch():
    for target_mode, obs_mode in (
        ("action_ball_full_mdp", "legacy"),
        ("legacy", "action_ball_full_mdp"),
    ):
        with pytest.raises(RuntimeError, match="requires task"):
            train_mod._validate_action_ball_full_mdp_runtime_modes(
                None,
                _runtime_modes(
                    target_mode=target_mode,
                    obs_mode=obs_mode,
                ),
            )


def test_hard_contract_requires_the_same_exact_full_mdp_modes(monkeypatch):
    binding = _resolve(monkeypatch)
    env, owner, adapter = _installed_runtime(binding)
    fresh = train_mod._action_ball_full_mdp_training_contract(
        binding, owner, adapter, None
    )
    hard_contract = {
        "target_mode": "action_ball_full_mdp",
        "actor_obs_mode": "action_ball_full_mdp",
        "fresh_full_mdp_installed_reward_graph": {},
    }
    train_mod._finalize_action_ball_full_mdp_hard_contract(
        hard_contract, fresh
    )
    assert hard_contract["action_ball_full_mdp_runtime"] is fresh

    for target_mode, actor_obs_mode in (
        ("action_ball", "action_ball_a211"),
        ("action_ball_full_mdp", "action_ball_a211"),
        ("action_ball", "action_ball_full_mdp"),
    ):
        with pytest.raises(RuntimeError, match="legacy target/actor mode"):
            train_mod._finalize_action_ball_full_mdp_hard_contract(
                {
                    "target_mode": target_mode,
                    "actor_obs_mode": actor_obs_mode,
                },
                fresh,
            )


def test_legacy_hard_contract_rejects_either_full_mdp_mode():
    for target_mode, actor_obs_mode in (
        ("action_ball_full_mdp", "legacy"),
        ("legacy", "action_ball_full_mdp"),
    ):
        with pytest.raises(RuntimeError, match="legacy hard contract"):
            train_mod._finalize_action_ball_full_mdp_hard_contract(
                {
                    "target_mode": target_mode,
                    "actor_obs_mode": actor_obs_mode,
                },
                None,
            )


def test_run_source_has_real_callsites_exact_runner_selection_and_no_config_factory():
    source = inspect.getsource(train_mod._run_with_environment_close_owner)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    gym_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gym"
        and node.func.attr == "make"
    ]
    assert len(gym_calls) == 1
    gym_call = gym_calls[0]
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "full_mdp_gym_kwargs"
        for keyword in gym_call.keywords
    )

    runner_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "runner_type"
    ]
    assert len(runner_calls) == 1
    runner_type_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "runner_type"
    ]
    assert len(runner_type_assignments) == 1
    runner_choice = runner_type_assignments[0].value
    assert isinstance(runner_choice, ast.IfExp)
    assert ast.unparse(runner_choice.body) == "ActionBallFullMdpRsl3Runner"
    assert ast.unparse(runner_choice.orelse) == "OnPolicyRunner"
    runner_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in runner_calls[0].keywords
    }
    assert runner_keywords["action_ball_full_mdp_runtime_owner"] == (
        "action_ball_full_mdp_runtime_owner"
    )
    assert runner_keywords["action_ball_r10_checkpoint_adapter"] == (
        "action_ball_r10_checkpoint_adapter"
    )
    assert runner_keywords["action_ball_r10_cold_restore_capsule"] == (
        "action_ball_r10_cold_restore_capsule"
    )
    assert runner_keywords["action_ball_full_mdp_run_mode"] == (
        "None if action_ball_full_mdp_pre_gym_binding is None else "
        "_ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE"
    )
    assert "_get(cfg, \"full_mdp_runtime_owner_factory\")" not in source
    assert "_get(cfg.task, \"full_mdp_runtime_owner_factory\")" not in source

    resolve_index = source.index(
        "_resolve_action_ball_full_mdp_pre_gym_binding("
    )
    parse_index = source.index("parse_env_cfg(task_id")
    gym_index = source.index("env = gym.make(")
    runtime_binding_index = source.index(
        "_resolve_action_ball_full_mdp_runtime_binding("
    )
    contract_write_index = source.index("os.makedirs(os.path.dirname(contract_path)")
    contract_index = source.index(
        "_finalize_action_ball_full_mdp_hard_contract("
    )
    runner_index = source.index("runner = runner_type(")
    assert (
        resolve_index
        < parse_index
        < gym_index
        < runtime_binding_index
        < contract_index
        < contract_write_index
        < runner_index
    )
    assert (
        "action_ball_full_mdp_training = "
        "action_ball_full_mdp_contract is not None"
    ) in source
    # One shared learn call now owns both the lateral and ordinary paths; the
    # duplicated second learn branch disappeared with close-once ownership.
    assert source.count("or action_ball_full_mdp_training") == 2
    compact_source = source.replace("\n", "").replace(" ", "")
    legacy_mutation_guard = (
        "fresh_full_mdp_reward_familyisNoneandnot_configured_items("
        "_get(cfg,\"motion_file\"),_get(cfg,\"motion_file_2\"))"
    )
    assert legacy_mutation_guard in compact_source
    assert source.index("_resolve_motion_sources_for_training(") < source.index(
        "env = gym.make("
    )
    assert (
        "action_ball_full_mdp_enabled=("
        "action_ball_full_mdp_pre_gym_bindingisnotNone)"
        in compact_source
    )


def test_launcher_publishes_failure_status_to_kit_before_cleanup():
    events = []

    class _App:
        def post_quit(self, code):
            events.append(("post_quit", code))

    class _SimulationApp:
        app = _App()

        def close(self):
            events.append(("close",))

    train_mod._close_simulation_app(_SimulationApp(), failed=True)
    assert events == [("post_quit", 1), ("close",)]


def test_launcher_normal_cleanup_does_not_publish_a_failure_status():
    events = []

    class _App:
        def post_quit(self, code):
            events.append(("post_quit", code))

    class _SimulationApp:
        app = _App()

        def close(self):
            events.append(("close",))

    train_mod._close_simulation_app(_SimulationApp(), failed=False)
    assert events == [("close",)]


class _LifecycleRawEnv:
    def __init__(self, events, close_error=None):
        self.events = events
        self.close_error = close_error
        self.close_count = 0

    def close(self):
        self.close_count += 1
        self.events.append("raw_close")
        if self.close_error is not None:
            raise self.close_error


class _LifecycleRootWrapper:
    def __init__(self, owner, events):
        self.owner = owner
        self.events = events

    def close(self):
        self.events.append("root_wrapper_close")
        self.owner._close_raw_once()


class _LifecycleOuterWrapper:
    def __init__(self, inner, events):
        self.inner = inner
        self.events = events

    def close(self):
        self.events.append("outer_wrapper_close")
        self.inner.close()


def _register_lifecycle_raw(owner, events, *, close_error=None):
    raw = _LifecycleRawEnv(events, close_error=close_error)
    events.append("gym_make_returned")
    owner.register_raw(raw)
    events.append("raw_registered")
    root = owner.adopt(_LifecycleRootWrapper(owner, events))
    return raw, root


@pytest.mark.parametrize(
    ("stage", "outer_wrapper"),
    [
        ("wrapper_constructor_reset", False),
        ("hard_contract", False),
        ("runner_constructor", True),
    ],
)
def test_pre_learn_failures_close_registered_raw_once(
    monkeypatch, stage, outer_wrapper
):
    events = []
    primary_error = RuntimeError(f"{stage} failed")
    observed = {}

    def injected_run(_cfg, owner):
        raw, root = _register_lifecycle_raw(owner, events)
        observed["raw"] = raw
        if outer_wrapper:
            owner.adopt(_LifecycleOuterWrapper(root, events))
        events.append(stage)
        raise primary_error

    monkeypatch.setattr(
        train_mod, "_run_with_environment_close_owner", injected_run
    )

    with pytest.raises(RuntimeError, match=stage) as caught:
        train_mod._run(object())

    assert caught.value is primary_error
    assert observed["raw"].close_count == 1
    assert events[-(4 if outer_wrapper else 3):] == [
        stage,
        *(["outer_wrapper_close"] if outer_wrapper else []),
        "root_wrapper_close",
        "raw_close",
    ]


def test_learn_failure_remains_primary_when_raw_close_also_fails(
    monkeypatch, capsys
):
    events = []
    learn_error = RuntimeError("learn failed")
    close_error = OSError("raw close failed")
    observed = {}

    def injected_run(_cfg, owner):
        raw, root = _register_lifecycle_raw(
            owner, events, close_error=close_error
        )
        observed["raw"] = raw
        owner.adopt(_LifecycleOuterWrapper(root, events))
        events.extend(("runner_constructed", "learn"))
        raise learn_error

    monkeypatch.setattr(
        train_mod, "_run_with_environment_close_owner", injected_run
    )

    with pytest.raises(RuntimeError, match="learn failed") as caught:
        train_mod._run(object())

    assert caught.value is learn_error
    assert observed["raw"].close_count == 1
    assert events[-5:] == [
        "runner_constructed",
        "learn",
        "outer_wrapper_close",
        "root_wrapper_close",
        "raw_close",
    ]
    assert "environment close failed while preserving the primary run failure" in (
        capsys.readouterr().err
    )


def test_normal_run_closes_outer_then_raw_exactly_once(monkeypatch):
    events = []
    observed = {}
    lateral_close_count = []

    def injected_run(_cfg, owner):
        raw, root = _register_lifecycle_raw(owner, events)
        observed["raw"] = raw
        def close_lateral_runtime():
            lateral_close_count.append(1)
            events.append("lateral_runtime_close")
            root.close()

        lateral_close_once = owner.add_backstop(close_lateral_runtime)

        def close_outer_wrapper():
            events.append("outer_wrapper_close")
            lateral_close_once()

        owner.adopt(types.SimpleNamespace(close=close_outer_wrapper))
        events.extend(("runner_constructed", "learn_complete"))
        return "completed"

    monkeypatch.setattr(
        train_mod, "_run_with_environment_close_owner", injected_run
    )

    assert train_mod._run(object()) == "completed"
    assert observed["raw"].close_count == 1
    assert len(lateral_close_count) == 1
    assert events[-6:] == [
        "runner_constructed",
        "learn_complete",
        "outer_wrapper_close",
        "lateral_runtime_close",
        "root_wrapper_close",
        "raw_close",
    ]
