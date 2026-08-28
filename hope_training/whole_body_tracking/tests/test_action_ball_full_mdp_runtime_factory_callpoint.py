"""Exact construction-callpoint tests for the fresh full-MDP runtime graph.

These tests prove the IsaacLab manager order and fail-stop construction seam.
They do not replace any production producer or authorize launch.
"""

from __future__ import annotations

import ast
import inspect
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from full_mdp_env_canonical_harness import load_canonical_full_mdp_env


_EXACT_SPLIT_ASSET = (
    "/workspace/franco/runtime_assets/"
    "a3p0807_split_rubber_diagnostic_v3/model.usd"
)


@pytest.fixture(autouse=True)
def _bind_exact_split_asset_environment(monkeypatch):
    """Isolate non-asset factory tests from the external Pod asset tree.

    Asset byte/reconstruction behavior is covered by
    ``test_action_ball_full_mdp_cfg_registration.py``.  The dedicated factory
    call-order test below replaces this seam with a rejecting implementation.
    """

    monkeypatch.setenv("HOPE_AGIBOT_A3_USD_PATH", _EXACT_SPLIT_ASSET)
    # Collection must restore any canonical package generation that another
    # test already owns.  Bind this file's subject only for one factory test;
    # monkeypatch then restores the prior generation at teardown.
    monkeypatch.setitem(sys.modules, M.__name__, M)
    split_asset_name = (
        "whole_body_tracking.tasks.tracking.config.agibot_a3."
        "action_ball_full_mdp_split_asset"
    )
    split_asset = sys.modules.get(split_asset_name)
    if split_asset is None:
        source = (
            TRACKING_ROOT
            / "config"
            / "agibot_a3"
            / "action_ball_full_mdp_split_asset.py"
        )
        spec = importlib.util.spec_from_file_location(split_asset_name, source)
        assert spec is not None and spec.loader is not None
        split_asset = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, split_asset_name, split_asset)
        spec.loader.exec_module(split_asset)

    def accept_external_asset_seam():
        return _EXACT_SPLIT_ASSET

    accept_external_asset_seam.__module__ = split_asset.__name__
    monkeypatch.setattr(
        split_asset,
        "require_action_ball_full_mdp_split_asset",
        accept_external_asset_seam,
    )


ROOT = Path(__file__).resolve().parents[1]
ENV_MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "full_mdp_env.py"
)
FACTORY_MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_full_mdp_runtime_factory.py"
)
SOURCE_ROOT = ROOT / "source" / "whole_body_tracking"
TRACKING_ROOT = SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking"
MDP_ROOT = TRACKING_ROOT / "mdp"
def _load_env_subject():
    return load_canonical_full_mdp_env(
        ENV_MODULE_PATH, retain_namespace=False
    )


def _load_constructing_env_subject(name: str):
    """Load over a base whose __init__ really dispatches load_managers."""

    class ConstructingManagerBasedRLEnv:
        def __init__(self, cfg, render_mode=None, **kwargs):
            del render_mode, kwargs
            self.cfg = cfg
            self.num_envs = 2
            self.device = cfg.device
            self.step_dt = 0.02
            self.common_step_counter = 0
            self.episode_length_buf = torch.zeros(
                cfg.scene.num_envs,
                dtype=torch.long,
                device=cfg.sim.device,
            )
            self.event_manager = _EventManager(cfg.trace)
            self._configure_gym_env_spaces = lambda: cfg.trace.append("spaces")
            self.load_managers()
            if cfg.swap_lease_before_super_return:
                self._action_ball_full_mdp_runtime_lease = object()

        def close(self):
            self.cfg.trace.append("close")

    del name
    module = load_canonical_full_mdp_env(
        ENV_MODULE_PATH, retain_namespace=False
    )
    module.ManagerBasedRLEnv.__init__ = ConstructingManagerBasedRLEnv.__init__
    module.ManagerBasedRLEnv.close = ConstructingManagerBasedRLEnv.close
    return module


def _load_factory_subject(name: str):
    spec = importlib.util.spec_from_file_location(name, FACTORY_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_single_action_lean_cfg_modules(
    monkeypatch,
    *,
    cfg,
    family="A",
    entry_point=None,
    factory_flags=(False, False, True),
    canary_flags=(2, False, True, False, False),
    owner_constructor=None,
):
    config_module = types.ModuleType(M.FULL_MDP_CONFIG_MODULE)
    type_name = (
        "HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg"
        if family == "A"
        else "HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg"
    )
    setattr(config_module, type_name, type(cfg))
    other_name = (
        "HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg"
        if family == "A"
        else "HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg"
    )
    setattr(config_module, other_name, type(f"Foreign{family}", (), {}))
    monkeypatch.setitem(sys.modules, M.FULL_MDP_CONFIG_MODULE, config_module)

    task_id = (
        "HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0"
        if family == "A"
        else "HOPE-PingPong-ActionBall-FullMdpC-AgibotA3-v0"
    )
    gym = types.ModuleType("gymnasium")
    gym.spec = lambda observed: types.SimpleNamespace(
        entry_point=(
            M.FULL_MDP_GYM_ENTRY_POINT
            if entry_point is None
            else entry_point
        ),
        kwargs={"env_cfg_entry_point": type(cfg)},
    ) if observed == task_id else (_ for _ in ()).throw(KeyError(observed))
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    factory = types.ModuleType(M.FULL_MDP_RUNTIME_FACTORY_MODULE)
    (
        factory.RUNTIME_INTEGRATED,
        factory.LAUNCH_AUTHORIZED,
        factory.DIAGNOSTIC_UNAUTHORIZED,
    ) = factory_flags
    exec(
        "def construct_action_ball_full_mdp_runtime_graph(env):\n"
        "    raise RuntimeError('causal builder reached')\n",
        factory.__dict__,
    )
    monkeypatch.setitem(sys.modules, M.FULL_MDP_RUNTIME_FACTORY_MODULE, factory)

    owner = types.ModuleType(M.FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE)
    owner.__dict__["factory_calls"] = []
    owner.__dict__["owner_constructor"] = owner_constructor
    exec(
        "class ActionBallFullMdpLeanRuntimeOwner:\n"
        "    @classmethod\n"
        "    def create_from_env(cls, env, lease):\n"
        "        factory_calls.append((env, lease))\n"
        "        if owner_constructor is None:\n"
        "            return None\n"
        "        return owner_constructor(env, lease)\n",
        owner.__dict__,
    )
    monkeypatch.setitem(
        sys.modules, M.FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE, owner
    )
    factory.test_owner_module = owner
    factory.test_owner_factory = (
        owner.ActionBallFullMdpLeanRuntimeOwner.create_from_env
    )

    canary = types.ModuleType("action_ball_full_mdp_canary_target_profile")
    (
        canary.CANARY_NUM_ENVS,
        canary.CANARY_SAVE_CHECKPOINTS,
        canary.DIAGNOSTIC_UNAUTHORIZED,
        canary.FORMAL_PROFILE,
        canary.FORMAL_LAUNCH_AUTHORIZED,
    ) = canary_flags
    monkeypatch.setitem(
        sys.modules, "action_ball_full_mdp_canary_target_profile", canary
    )
    return factory


def _single_action_lean_cfg(*, family="A"):
    cfg_type = type(f"ExactDiagnostic{family}Cfg", (), {})
    cfg = cfg_type()
    cfg.scene = types.SimpleNamespace(num_envs=2)
    cfg.sim = types.SimpleNamespace(device="cpu")
    cfg.commands = types.SimpleNamespace(
        racket_target=types.SimpleNamespace(
            target_mode="action_ball_full_mdp",
            action_ball_diagnostic_unauthorized=True,
        )
    )
    cfg.action_ball_full_mdp_family_role = family
    cfg.obs_mode = "action_ball_full_mdp"
    cfg.action_ball_full_mdp_scene_capacity = 2
    cfg.action_ball_full_mdp_capacity_receipt_sha256 = ""
    cfg.action_ball_full_mdp_runtime_construction_status = "HOLD"
    cfg.checkpoint_path = None
    cfg.checkpoint_tolerant = False
    cfg.observations = object()
    return cfg


M = _load_env_subject()


class _Cfg:
    commands = object()
    terminations = object()
    rewards = object()
    observations = object()
    curriculum = object()


class _ConstructingCfg(_Cfg):
    def __init__(self, trace, *, device, swap_lease_before_super_return):
        self.trace = trace
        self.device = device
        self.scene = types.SimpleNamespace(num_envs=2)
        self.sim = types.SimpleNamespace(device=device)
        self.commands = types.SimpleNamespace(
            racket_target=types.SimpleNamespace(
                target_mode="action_ball_full_mdp",
                action_ball_diagnostic_unauthorized=True,
            )
        )
        self.action_ball_full_mdp_family_role = "A"
        self.obs_mode = "action_ball_full_mdp"
        self.action_ball_full_mdp_scene_capacity = 2
        self.action_ball_full_mdp_capacity_receipt_sha256 = ""
        self.action_ball_full_mdp_runtime_construction_status = "HOLD"
        self.checkpoint_path = None
        self.checkpoint_tolerant = False
        self.swap_lease_before_super_return = swap_lease_before_super_return


class _EventManager:
    available_modes = ("startup",)

    def __init__(self, trace):
        self.trace = trace

    def apply(self, *, mode):
        self.trace.append(("startup", mode))


class _GenesisAuthority:
    def __init__(self, subject=M):
        self.subject = subject
        self.receipt = object()
        self.world_reset_identity = object()

    def require_owned_full_mdp_reset_genesis(
        self, receipt, *, device, num_envs
    ):
        assert receipt is self.receipt
        return self.subject.FullMdpResetGenesisProjection(
            world_reset_identity=self.world_reset_identity,
            reset_generations=torch.ones(
                num_envs, dtype=torch.int64, device=device
            ),
        )


class _Scene(types.SimpleNamespace):
    def __getitem__(self, name):
        return getattr(self, name)


class _SceneAsset:
    def __init__(self, root):
        self.data = types.SimpleNamespace(root_state_w=root)

    def write_root_pose_to_sim(self, value, env_ids=None):
        assert env_ids is not None
        self.data.root_state_w[env_ids, :7] = value

    def write_root_velocity_to_sim(self, value, env_ids=None):
        assert env_ids is not None
        self.data.root_state_w[env_ids, 7:] = value


class _LeanRewardGraph:
    def __init__(self):
        self.configured = None

    def configure_milestone_configured_income(self, manager_cfg, step_dt):
        assert type(manager_cfg) is dict
        assert tuple(manager_cfg) == M.reward_contract.MANAGER_NAMES
        self.configured = (manager_cfg, step_dt)


class _LivePhysxSubscriber:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown_action_epoch_live_physx_fact_owner(self):
        self.shutdown_calls += 1


def _install_lean_observation_module(monkeypatch):
    name = (
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_observation_cfg"
    )
    module = types.ModuleType(name)
    exec(
        "class LeanActionEpochObservationSource:\n"
        "    def __init__(self, env, runtime_owner, epoch_owner):\n"
        "        self._env = env\n"
        "        self._runtime_owner = runtime_owner\n"
        "        self._epoch_owner = epoch_owner\n"
        "    def observe(self, group):\n"
        "        raise AssertionError('cold observation source must not run')\n",
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, name, module)
    return module.LeanActionEpochObservationSource


def _atomic_lean_graph_inputs(
    env,
    motion,
    racket,
    *,
    subject,
    observation_source_type,
    lean_owner=None,
):
    epoch_owner = object()
    runtime_owner = object() if lean_owner is None else lean_owner
    reward_graph = _LeanRewardGraph()
    observation_source = observation_source_type(
        env, runtime_owner, epoch_owner
    )
    components = subject.FullMdpLeanRuntimeComponents(
        epoch_owner=epoch_owner,
        device_r05_owner=object(),
        motion_owner=motion,
        racket_owner=racket,
        physical_owner=object(),
        r03_owner=object(),
        r06_owner=object(),
        r07_owner=object(),
        r07_plant_fact_adapter=object(),
        reward_graph=reward_graph,
        lean_runtime_owner=runtime_owner,
        observation_source=observation_source,
    )
    subscriber = _LivePhysxSubscriber()
    return {
        "components": components,
        "reward_manager_cfg": {
            name: object() for name in subject.reward_contract.MANAGER_NAMES
        },
        "observation_source": observation_source,
        "observation_manager_cfg": {
            "policy": object(),
            "critic": object(),
        },
        "termination_manager_cfg": {
            "time_out": object(),
            "base_fell_tilt": object(),
            "base_too_low": object(),
            "joint_qdes_forbidden": object(),
            "robot_hit_table": object(),
        },
        "live_physx_shutdown": (
            subscriber.shutdown_action_epoch_live_physx_fact_owner
        ),
        "subscriber": subscriber,
    }


def _manager_modules(trace, *, motion, racket, subject=M):
    managers = types.ModuleType("isaaclab.managers")
    base_module = types.ModuleType("isaaclab.envs.manager_based_env")

    class CommandManager:
        def __init__(self, cfg, env):
            assert cfg is env.cfg.commands
            trace.append("command")
            self._terms = {"motion": motion, "racket_target": racket}

        def get_term(self, name):
            return self._terms[name]

    class TerminationManager:
        def __init__(self, cfg, env):
            assert cfg is env.cfg.terminations
            trace.append("termination")

    class RewardManager:
        def __init__(self, cfg, env):
            assert cfg is env.cfg.rewards
            trace.append("reward")

    class CurriculumManager:
        def __init__(self, cfg, env):
            assert cfg is env.cfg.curriculum
            trace.append("curriculum")

    class ManagerBasedEnv:
        @staticmethod
        def load_managers(env):
            assert type(env._action_ball_full_mdp_components) is subject.FullMdpLeanRuntimeComponents
            assert type(env._action_ball_full_mdp_reset_genesis_install) is subject._FullMdpResetGenesisInstall
            trace.append("recorder")
            env.recorder_manager = object()
            trace.append("action")
            env.action_manager = object()
            trace.append("observation")
            env.observation_manager = object()

    managers.CommandManager = CommandManager
    managers.TerminationManager = TerminationManager
    managers.RewardManager = RewardManager
    managers.CurriculumManager = CurriculumManager
    base_module.ManagerBasedEnv = ManagerBasedEnv
    return managers, base_module


def _env(trace, *, device_name="cpu"):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    env.cfg = _Cfg()
    env.cfg.scene = types.SimpleNamespace(num_envs=2)
    env.cfg.sim = types.SimpleNamespace(device=device_name)
    env.event_manager = _EventManager(trace)
    env._action_ball_full_mdp_manager_construction_state = "armed"
    env._action_ball_full_mdp_runtime_graph_builder_invocations = 0
    lease = object()
    env._action_ball_full_mdp_runtime_lease = lease
    env._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
    env._configure_gym_env_spaces = lambda: trace.append("spaces")
    env.num_envs = 2
    env.device = device_name
    env.step_dt = 0.02
    env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long, device=device_name)
    return env


def _install_fake_isaac_modules(
    monkeypatch, trace, *, motion, racket, subject=M
):
    managers, base_module = _manager_modules(
        trace, motion=motion, racket=racket, subject=subject
    )
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(
        sys.modules, "isaaclab.envs.manager_based_env", base_module
    )


def _install_success_builder(
    monkeypatch, trace, *, motion, racket, subject=M, lean_owner=None
):
    module = types.ModuleType(subject.FULL_MDP_RUNTIME_FACTORY_MODULE)
    authority = _GenesisAuthority(subject)
    observation_source_type = _install_lean_observation_module(monkeypatch)

    def make_install_inputs(env):
        return _atomic_lean_graph_inputs(
            env,
            motion,
            racket,
            subject=subject,
            observation_source_type=observation_source_type,
            lean_owner=lean_owner,
        )

    module.__dict__.update(
        {
            "trace": trace,
            "authority": authority,
            "receipt": authority.receipt,
            "make_install_inputs": make_install_inputs,
            "seen_lease": None,
            "seen_install_inputs": None,
        }
    )
    exec(
        "def construct_action_ball_full_mdp_runtime_graph(env):\n"
        "    trace.append('builder')\n"
        "    global seen_lease, seen_install_inputs\n"
        "    seen_lease = env.action_ball_full_mdp_construction_lease()\n"
        "    seen_install_inputs = make_install_inputs(env)\n"
        "    env.install_action_ball_full_mdp_lean_runtime_graph(\n"
        "        seen_lease,\n"
        "        genesis_authority=authority,\n"
        "        genesis_receipt=receipt,\n"
        "        **{key: value for key, value in seen_install_inputs.items() if key != 'subscriber'},\n"
        "    )\n",
        module.__dict__,
    )
    monkeypatch.setitem(
        sys.modules, subject.FULL_MDP_RUNTIME_FACTORY_MODULE, module
    )
    return module


class _InitOwner:
    def __init__(self, env, lease):
        self._env = env
        self._lease = lease

    @property
    def full_mdp_runtime_env(self):
        return self._env

    @property
    def full_mdp_runtime_lease(self):
        return self._lease

    def before_policy_step(self, control_step, action):
        del control_step, action

    def before_physics_substep(self, stamp):
        del stamp

    def publish_post_physics_substep(self, stamp):
        del stamp

    def after_reward_close(self, control_step):
        assert type(control_step) is int
        return None

    def after_command_compute_before_observation(self, control_step):
        assert type(control_step) is int
        return None

    def selected_true_reset(self, event):
        return event


def _init_owner_binding(subject, owner):
    owner_type = type(owner)
    return subject._LeanOwnerExecutableBinding(
        owner_type=owner_type,
        before_policy_step_function=vars(owner_type)["before_policy_step"],
        before_physics_substep_function=vars(owner_type)[
            "before_physics_substep"
        ],
        publish_function=vars(owner_type)["publish_post_physics_substep"],
        after_reward_close_function=vars(owner_type)["after_reward_close"],
        after_command_compute_before_observation_function=vars(owner_type)[
            "after_command_compute_before_observation"
        ],
        selected_true_reset_function=vars(owner_type)["selected_true_reset"],
    )


def _prepare_real_init_subject(
    monkeypatch, *, name, trace, device_name, swap_lease
):
    subject = _load_constructing_env_subject(name)
    motion, racket = object(), object()
    _install_fake_isaac_modules(
        monkeypatch,
        trace,
        motion=motion,
        racket=racket,
        subject=subject,
    )
    cfg = _ConstructingCfg(
        trace,
        device=device_name,
        swap_lease_before_super_return=swap_lease,
    )
    diagnostic = _install_single_action_lean_cfg_modules(
        monkeypatch,
        cfg=cfg,
        owner_constructor=_InitOwner,
    )
    builder = _install_success_builder(
        monkeypatch,
        trace,
        motion=motion,
        racket=racket,
        subject=subject,
    )
    builder.RUNTIME_INTEGRATED = False
    builder.LAUNCH_AUTHORIZED = False
    builder.DIAGNOSTIC_UNAUTHORIZED = True
    builder.test_owner_module = diagnostic.test_owner_module
    builder.test_owner_factory = diagnostic.test_owner_factory
    monkeypatch.setitem(
        sys.modules,
        subject.FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE,
        diagnostic.test_owner_module,
    )
    for name in (
        "_require_standalone_simulation_app",
        "_assert_runtime_uses_pinned_upstream_step",
        "_assert_runtime_uses_pinned_manager_based_env",
        "_assert_runtime_uses_pinned_local_step",
    ):
        monkeypatch.setattr(subject, name, lambda: None)
    monkeypatch.setattr(
        subject.ActionBallFullMdpManagerBasedRLEnv,
        "_validate_lean_owner_install",
        lambda self, owner, **kwargs: _init_owner_binding(subject, owner),
    )
    return subject, builder, cfg


def test_builder_runs_exactly_once_after_command_before_observation(monkeypatch):
    trace = []
    motion, racket = object(), object()
    env = _env(trace)
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    builder_module = _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )
    minted_lease = env._action_ball_full_mdp_runtime_lease

    env.load_managers()

    assert trace == [
        "command",
        "builder",
        "recorder",
        "action",
        "observation",
        "termination",
        "reward",
        "curriculum",
        "spaces",
        ("startup", "startup"),
    ]
    assert env._action_ball_full_mdp_runtime_graph_builder_invocations == 1
    assert builder_module.seen_lease is minted_lease
    install_inputs = builder_module.seen_install_inputs
    assert type(install_inputs["components"]) is M.FullMdpLeanRuntimeComponents
    assert tuple(install_inputs["reward_manager_cfg"]) == (
        M.reward_contract.MANAGER_NAMES
    )
    assert install_inputs["components"].reward_graph.configured == (
        install_inputs["reward_manager_cfg"],
        env.step_dt,
    )
    assert env._action_ball_full_mdp_manager_construction_state == (
        "base_managers_complete"
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="replayed or previously failed",
    ):
        env.load_managers()
    assert trace.count("command") == trace.count("builder") == 1


@pytest.mark.parametrize("family", ("A", "C"))
@pytest.mark.parametrize("num_envs", (1, 2, 64))
def test_exact_single_action_lean_cfg_reaches_unique_causal_builder(
    monkeypatch, family, num_envs
):
    cfg = _single_action_lean_cfg(family=family)
    cfg.scene.num_envs = num_envs
    diagnostic_modules = _install_single_action_lean_cfg_modules(
        monkeypatch, cfg=cfg, family=family
    )

    M._require_single_action_lean_cfg(
        cfg,
        owner_factory=diagnostic_modules.test_owner_factory,
    )
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    motion, racket = object(), object()
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )

    def fake_base_init(candidate, cfg, render_mode=None, **kwargs):
        del render_mode, kwargs
        candidate.cfg = cfg
        candidate.num_envs = num_envs
        candidate.device = "cpu"
        candidate.common_step_counter = 0
        candidate.episode_length_buf = torch.zeros(
            cfg.scene.num_envs,
            dtype=torch.long,
            device=cfg.sim.device,
        )
        candidate.event_manager = _EventManager(trace)
        candidate._configure_gym_env_spaces = lambda: trace.append("spaces")
        candidate.load_managers()

    monkeypatch.setattr(M.ManagerBasedRLEnv, "__init__", fake_base_init)
    monkeypatch.setattr(M, "_require_standalone_simulation_app", lambda: None)
    monkeypatch.setattr(M, "_assert_runtime_uses_pinned_upstream_step", lambda: None)
    monkeypatch.setattr(M, "_assert_runtime_uses_pinned_manager_based_env", lambda: None)
    monkeypatch.setattr(M, "_assert_runtime_uses_pinned_local_step", lambda: None)

    with pytest.raises(RuntimeError, match="causal builder reached"):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg,
            full_mdp_runtime_owner_factory=(
                diagnostic_modules.test_owner_factory
            ),
        )
    assert env._action_ball_full_mdp_manager_construction_state == "failed"
    assert env._action_ball_full_mdp_runtime_graph_builder_invocations == 1
    assert trace == ["command"]


def test_single_action_lean_rejects_foreign_owner_factory_before_super(
    monkeypatch,
):
    cfg = _single_action_lean_cfg()
    diagnostic_modules = _install_single_action_lean_cfg_modules(
        monkeypatch, cfg=cfg
    )
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    super_calls = []
    monkeypatch.setattr(
        M.ManagerBasedRLEnv,
        "__init__",
        lambda *args, **kwargs: super_calls.append((args, kwargs)),
    )

    def wrapper(*args):
        return diagnostic_modules.test_owner_factory(*args)

    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError,
        match="foreign, wrapped or caller-selected",
    ):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg,
            full_mdp_runtime_owner_factory=wrapper,
        )
    assert super_calls == []
    assert not hasattr(env, "_action_ball_full_mdp_runtime_lease")


def test_builder_owner_factory_mismatch_cold_discards_lean_graph(
    monkeypatch,
):
    trace = []
    cfg = _single_action_lean_cfg()
    cfg.trace = trace
    cfg.device = "cpu"
    cfg.swap_lease_before_super_return = False
    cfg.terminations = object()
    cfg.rewards = object()
    cfg.curriculum = object()
    subject = _load_constructing_env_subject(
        "_full_mdp_lean_builder_mismatch_subject"
    )
    motion, racket = object(), object()
    _install_fake_isaac_modules(
        monkeypatch,
        trace,
        motion=motion,
        racket=racket,
        subject=subject,
    )
    builder = _install_success_builder(
        monkeypatch,
        trace,
        motion=motion,
        racket=racket,
        subject=subject,
    )
    diagnostic_modules = _install_single_action_lean_cfg_modules(
        monkeypatch, cfg=cfg
    )
    builder.RUNTIME_INTEGRATED = False
    builder.LAUNCH_AUTHORIZED = False
    builder.DIAGNOSTIC_UNAUTHORIZED = True
    monkeypatch.setitem(
        sys.modules,
        subject.FULL_MDP_RUNTIME_FACTORY_MODULE,
        builder,
    )
    owner_module = diagnostic_modules.test_owner_module
    monkeypatch.setitem(
        sys.modules,
        subject.FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE,
        owner_module,
    )
    for name in (
        "_require_standalone_simulation_app",
        "_assert_runtime_uses_pinned_upstream_step",
        "_assert_runtime_uses_pinned_manager_based_env",
        "_assert_runtime_uses_pinned_local_step",
    ):
        monkeypatch.setattr(subject, name, lambda: None)
    env = object.__new__(subject.ActionBallFullMdpManagerBasedRLEnv)
    owner_factory = (
        owner_module.ActionBallFullMdpLeanRuntimeOwner.create_from_env
    )

    with pytest.raises(
        subject.FullMdpPostPhysicsProtocolError,
        match="foreign, duplicated, or not factory-installed",
    ):
        subject.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg,
            full_mdp_runtime_owner_factory=owner_factory,
        )
    assert trace.count("builder") == 1
    assert trace.count("close") == 1
    assert owner_module.factory_calls == [
        (env, env._action_ball_full_mdp_runtime_lease)
    ]
    assert env._action_ball_full_mdp_manager_construction_state == "failed"
    assert env._action_ball_full_mdp_runtime_graph_builder_invocations == 1
    assert env._action_ball_full_mdp_components is subject._ABSENT
    assert env._action_ball_full_mdp_reset_genesis_install is subject._ABSENT
    assert getattr(env, "_full_mdp_runtime_owner", subject._ABSENT) is subject._ABSENT
    with pytest.raises(
        subject.FullMdpPostPhysicsProtocolError,
        match="runtime lease is not sealed",
    ):
        _ = env.action_ball_full_mdp_runtime_lease
    with pytest.raises(
        subject.FullMdpPostPhysicsProtocolError,
        match="replayed or previously failed",
    ):
        env.load_managers()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("foreign_type", "exact registered A/C"),
        ("family", "family_role_rewritten"),
        ("num_envs", "num_envs_must_be_positive_exact_int"),
        ("diagnostic", "diagnostic_unauthorized_differs"),
        ("checkpoint", "checkpoint_resume_present"),
        ("registration", "gym_registration_differs"),
        ("factory_flag", "factory_authorization_flags_differ"),
        ("canary_save", "canary_no_save_or_authorization_flags_differ"),
    ),
)
def test_single_action_lean_rejects_foreign_or_save_authority(
    monkeypatch, mutation, match
):
    cfg = _single_action_lean_cfg()
    exact_cfg = cfg
    entry_point = None
    factory_flags = (False, False, True)
    canary_flags = (2, False, True, False, False)
    if mutation == "foreign_type":
        exact_cfg = _single_action_lean_cfg()
    elif mutation == "family":
        cfg.action_ball_full_mdp_family_role = "C"
    elif mutation == "num_envs":
        cfg.scene.num_envs = 0
    elif mutation == "diagnostic":
        cfg.commands.racket_target.action_ball_diagnostic_unauthorized = False
    elif mutation == "checkpoint":
        cfg.checkpoint_path = "/tmp/forbidden.pt"
    elif mutation == "registration":
        entry_point = "isaaclab.envs:ManagerBasedRLEnv"
    elif mutation == "factory_flag":
        factory_flags = (True, False, True)
    elif mutation == "canary_save":
        canary_flags = (2, True, True, False, False)
    _install_single_action_lean_cfg_modules(
        monkeypatch,
        cfg=exact_cfg,
        entry_point=entry_point,
        factory_flags=factory_flags,
        canary_flags=canary_flags,
    )

    with pytest.raises(M.FullMdpPostPhysicsOwnerMissingError, match=match):
        M._require_single_action_lean_cfg(
            cfg,
            owner_factory=(
                sys.modules[M.FULL_MDP_RUNTIME_FACTORY_MODULE]
                .test_owner_factory
            ),
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_real_super_return_seals_the_same_pre_manager_lease(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trace = []
    subject, builder, cfg = _prepare_real_init_subject(
        monkeypatch,
        name=f"_full_mdp_real_init_{device_name.replace(':', '_')}",
        trace=trace,
        device_name=device_name,
        swap_lease=False,
    )

    env = subject.ActionBallFullMdpManagerBasedRLEnv(
        cfg,
        full_mdp_runtime_owner_factory=builder.test_owner_factory,
    )

    assert trace[:3] == ["command", "builder", "recorder"]
    assert builder.seen_lease is env.action_ball_full_mdp_runtime_lease
    assert env.full_mdp_runtime_owner.full_mdp_runtime_lease is builder.seen_lease
    assert env._action_ball_full_mdp_manager_construction_state == "sealed"
    assert env.device == device_name


def test_real_super_return_lease_swap_cold_discards_whole_env(monkeypatch):
    trace = []
    subject, builder, cfg = _prepare_real_init_subject(
        monkeypatch,
        name="_full_mdp_real_init_swap",
        trace=trace,
        device_name="cpu",
        swap_lease=True,
    )

    env = object.__new__(subject.ActionBallFullMdpManagerBasedRLEnv)
    with pytest.raises(
        subject.FullMdpPostPhysicsProtocolError, match="identity changed"
    ):
        subject.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg,
            full_mdp_runtime_owner_factory=builder.test_owner_factory,
        )

    assert env._action_ball_full_mdp_manager_construction_state == "failed"
    assert env._action_ball_full_mdp_components is subject._ABSENT
    assert env._action_ball_full_mdp_reset_genesis_install is subject._ABSENT
    assert trace[-1] == "close"


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_super_return_seals_the_exact_pre_manager_lease(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trace = []
    motion, racket = object(), object()
    env = _env(trace, device_name=device_name)
    minted_lease = env._action_ball_full_mdp_runtime_lease
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )

    env.load_managers()
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="not sealed"):
        _ = env.action_ball_full_mdp_runtime_lease
    env._seal_action_ball_full_mdp_after_base_construction(minted_lease)

    assert env.action_ball_full_mdp_runtime_lease is minted_lease
    assert env._action_ball_full_mdp_manager_construction_state == "sealed"
    assert env.num_envs == 2
    assert env.device == device_name


def test_atomic_lean_install_rejects_foreign_lease_before_publish(monkeypatch):
    trace = []
    motion, racket = object(), object()
    env = _env(trace)
    env._action_ball_full_mdp_manager_construction_state = (
        "command_manager_ready"
    )
    module = _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )
    install_inputs = module.make_install_inputs(env)

    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="foreign lease"):
        env.install_action_ball_full_mdp_lean_runtime_graph(
            object(),
            genesis_authority=module.authority,
            genesis_receipt=module.receipt,
            **{
                key: value
                for key, value in install_inputs.items()
                if key != "subscriber"
            },
        )
    assert not hasattr(env, "_action_ball_full_mdp_components")
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")


def test_super_return_rejects_lease_swap_without_sealing(monkeypatch):
    trace = []
    motion, racket = object(), object()
    env = _env(trace)
    minted_lease = env._action_ball_full_mdp_runtime_lease
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )
    env.load_managers()
    env._action_ball_full_mdp_runtime_lease = object()

    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="identity changed"):
        env._seal_action_ball_full_mdp_after_base_construction(minted_lease)
    assert env._action_ball_full_mdp_manager_construction_state == (
        "base_managers_complete"
    )


def test_builder_lease_swap_is_sticky_and_cannot_publish_partial_graph(
    monkeypatch,
):
    trace = []
    motion, racket = object(), object()
    env = _env(trace)
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    module = _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )
    exec(
        "def construct_action_ball_full_mdp_runtime_graph(env):\n"
        "    trace.append('builder')\n"
        "    lease = env.action_ball_full_mdp_construction_lease()\n"
        "    env._action_ball_full_mdp_runtime_lease = object()\n"
        "    install_inputs = make_install_inputs(env)\n"
        "    env.install_action_ball_full_mdp_lean_runtime_graph(\n"
        "        lease,\n"
        "        genesis_authority=authority,\n"
        "        genesis_receipt=receipt,\n"
        "        **{key: value for key, value in install_inputs.items() if key != 'subscriber'},\n"
        "    )\n",
        module.__dict__,
    )

    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="identity changed"):
        env.load_managers()
    assert trace == ["command", "builder"]
    assert env._action_ball_full_mdp_manager_construction_state == "failed"
    assert not hasattr(env, "_action_ball_full_mdp_components")
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="outside"):
        env.action_ball_full_mdp_construction_lease()


def test_builder_failure_after_install_is_cold_discard_not_partial_authority(
    monkeypatch,
):
    trace = []
    motion, racket = object(), object()
    env = _env(trace)
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    module = _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )
    exec(
        "def construct_action_ball_full_mdp_runtime_graph(env):\n"
        "    trace.append('builder')\n"
        "    lease = env.action_ball_full_mdp_construction_lease()\n"
        "    install_inputs = make_install_inputs(env)\n"
        "    env.install_action_ball_full_mdp_lean_runtime_graph(\n"
        "        lease,\n"
        "        genesis_authority=authority,\n"
        "        genesis_receipt=receipt,\n"
        "        **{key: value for key, value in install_inputs.items() if key != 'subscriber'},\n"
        "    )\n"
        "    raise ValueError('builder counterexample')\n",
        module.__dict__,
    )

    with pytest.raises(ValueError, match="counterexample"):
        env.load_managers()
    assert env._action_ball_full_mdp_manager_construction_state == "failed"
    assert not hasattr(env, "observation_manager")
    assert env._action_ball_full_mdp_components is M._ABSENT
    assert env._action_ball_full_mdp_reset_genesis_install is M._ABSENT
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="not sealed"):
        _ = env.action_ball_full_mdp_runtime_lease
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="previously failed"):
        env.load_managers()


def test_missing_production_node_is_sticky_hold_before_observation(monkeypatch):
    trace = []
    probe, motion, racket, _scene = _exact_single_action_lean_factory_env()
    motion._action_ball_continuous_motion_profile = None
    env = _env(trace)
    env.cfg = probe.cfg
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    factory = _load_factory_subject(M.FULL_MDP_RUNTIME_FACTORY_MODULE)
    monkeypatch.setitem(
        sys.modules, M.FULL_MDP_RUNTIME_FACTORY_MODULE, factory
    )

    with pytest.raises(
        factory.ActionBallFullMdpRuntimeFactoryHold,
        match="before ObservationManager",
    ) as caught:
        env.load_managers()

    assert trace == ["command"]
    assert not hasattr(env, "action_manager")
    assert not hasattr(env, "observation_manager")
    assert env._action_ball_full_mdp_manager_construction_state == "failed"
    failure = env._action_ball_full_mdp_manager_construction_failure
    assert type(failure) is M.FullMdpManagerConstructionFailureSnapshot
    assert failure.phase == "manager_graph_construction"
    assert failure.message == str(caught.value)
    assert failure.exception_type.endswith("ActionBallFullMdpRuntimeFactoryHold")
    assert not hasattr(failure, "__traceback__")
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="previously failed",
    ):
        env.load_managers()
    assert trace == ["command"]


def test_atomic_builder_cannot_omit_genesis(monkeypatch):
    trace = []
    motion, racket = object(), object()
    env = _env(trace)
    _install_fake_isaac_modules(
        monkeypatch, trace, motion=motion, racket=racket
    )
    module = _install_success_builder(
        monkeypatch, trace, motion=motion, racket=racket
    )
    exec(
        "def construct_action_ball_full_mdp_runtime_graph(env):\n"
        "    trace.append('builder')\n"
        "    lease = env.action_ball_full_mdp_construction_lease()\n"
        "    install_inputs = make_install_inputs(env)\n"
        "    env.install_action_ball_full_mdp_lean_runtime_graph(\n"
        "        lease,\n"
        "        genesis_authority=None,\n"
        "        genesis_receipt=receipt,\n"
        "        **{key: value for key, value in install_inputs.items() if key != 'subscriber'},\n"
        "    )\n",
        module.__dict__,
    )

    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError,
        match="lacks its independent reset genesis",
    ):
        env.load_managers()
    assert trace == ["command", "builder"]
    assert not hasattr(env, "observation_manager")
    assert env._action_ball_full_mdp_manager_construction_state == "failed"


def _exact_single_action_lean_factory_env(*, num_envs=2, device_name="cpu"):
    """Build the exact nominal seam without constructing an Isaac simulator."""

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from test_reward_flags_mdp import _PKG, _load

    sys.modules[_PKG].__path__ = [str(MDP_ROOT)]
    commands = _load(f"{_PKG}.commands", "commands.py")
    hope_commands = _load(f"{_PKG}.hope_commands", "hope_commands.py")
    _load(f"{_PKG}.continuous_questions", "continuous_questions.py")
    for package_name, package_path in (
        ("whole_body_tracking", SOURCE_ROOT / "whole_body_tracking"),
        ("whole_body_tracking.tasks", SOURCE_ROOT / "whole_body_tracking" / "tasks"),
        ("whole_body_tracking.tasks.tracking", TRACKING_ROOT),
        ("whole_body_tracking.tasks.tracking.mdp", MDP_ROOT),
    ):
        package = sys.modules.setdefault(
            package_name, types.ModuleType(package_name)
        )
        package.__path__ = [str(package_path)]

    config_name = (
        "whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg"
    )
    config = sys.modules.get(config_name)
    if config is None:
        config = types.ModuleType(config_name)
        config.__file__ = str(
            TRACKING_ROOT / "config" / "agibot_a3" / "hope_env_cfg.py"
        )
        sys.modules[config_name] = config
        exec(
            "class HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg:\n"
            "    pass\n"
            "class HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg:\n"
            "    pass\n"
            "def action_ball_full_mdp_family_role(env_cfg):\n"
            "    roles = {\n"
            "        HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg: 'A',\n"
            "        HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg: 'C',\n"
            "    }\n"
            "    resolved = roles.get(type(env_cfg))\n"
            "    if resolved is None:\n"
            "        raise RuntimeError('exact registered EnvCfg type')\n"
            "    if env_cfg.action_ball_full_mdp_family_role != resolved:\n"
            "        raise RuntimeError('role was rewritten')\n"
            "    return resolved\n",
            config.__dict__,
        )

    table_root = SOURCE_ROOT / "whole_body_tracking" / "tasks" / "table_tennis"
    table_package_name = "whole_body_tracking.tasks.table_tennis"
    table_package = sys.modules.setdefault(
        table_package_name, types.ModuleType(table_package_name)
    )
    table_package.__path__ = [str(table_root)]
    for short_name in ("geometry", "table_frame"):
        dotted = f"{table_package_name}.{short_name}"
        if dotted not in sys.modules:
            module_spec = importlib.util.spec_from_file_location(
                dotted, table_root / f"{short_name}.py"
            )
            assert module_spec is not None and module_spec.loader is not None
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[dotted] = module
            module_spec.loader.exec_module(module)
            setattr(table_package, short_name, module)
    scene_name = (
        "whole_body_tracking.tasks.tracking.config.agibot_a3."
        "action_ball_full_mdp_ball_scene"
    )
    scene = sys.modules.get(scene_name)
    if scene is None:
        scene_path = (
            SOURCE_ROOT
            / "whole_body_tracking"
            / "tasks"
            / "tracking"
            / "config"
            / "agibot_a3"
            / "action_ball_full_mdp_ball_scene.py"
        )
        for package_name, package_path in (
            (
                "whole_body_tracking.tasks.tracking.config",
                scene_path.parents[1],
            ),
            (
                "whole_body_tracking.tasks.tracking.config.agibot_a3",
                scene_path.parent,
            ),
        ):
            package = sys.modules.setdefault(
                package_name, types.ModuleType(package_name)
            )
            package.__path__ = [str(package_path)]
        spec = importlib.util.spec_from_file_location(scene_name, scene_path)
        assert spec is not None and spec.loader is not None
        scene = importlib.util.module_from_spec(spec)
        sys.modules[scene_name] = scene
        spec.loader.exec_module(scene)

    cadence = __import__("action_ball_motion_cadence_device")
    _parent, _parent_receipt, motion_profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    torch = pytest.importorskip("torch")
    device = torch.device(device_name)
    # Use the real Motion constructor harness so the exact code-owned catalog
    # and MotionLoader are retained by production initialization.  Manually
    # assigning ``_action_ball_action_uids`` or a synthetic catalog would make
    # the parent-schedule check self-authenticating and conceal the same cold
    # order bug this factory test is meant to catch.
    import test_action_ball_continuous_motion_bridge as motion_bridge
    import test_action_ball_motion_genesis_cadence_activation as motion_genesis

    motion, _env_ids = motion_bridge._configure_unbound_command(
        num_envs=num_envs,
        profile=motion_profile,
    )
    motion_genesis._move_command(motion, device)
    assert type(motion) is commands.MotionCommand
    motion._canonical_diagnostic_unauthorized = True
    racket = object.__new__(hope_commands.RacketTargetCommand)
    # Mirror the real Racket CommandTerm's cold state.  The recurring
    # constructor must call the production no-argument cold initializer,
    # which resolves this ``None`` through the already-complete
    # CommandManager.  Supplying ``motion`` here would bypass that real
    # producer and turn the focused test into a hand-filled fixture.
    racket._motion_term = None
    racket_cfg = hope_commands.RacketTargetCommandCfg()
    racket_cfg.target_mode = "action_ball_full_mdp"
    racket_cfg.action_ball_diagnostic_unauthorized = True
    live_racket_cfg = hope_commands.RacketTargetCommandCfg()
    live_racket_cfg.target_mode = "action_ball_full_mdp"
    live_racket_cfg.action_ball_diagnostic_unauthorized = True
    motion_cfg = object.__new__(commands.MotionCommandCfg)
    motion_cfg.action_ball_continuous_motion_cadence = dict(motion_profile)
    scene_spec = scene.ActionBallFullMdpDiagnosticBallSceneSpec(
        schema_version=scene.SCHEMA_VERSION,
        kind=scene.DIAGNOSTIC_SCENE_SPEC_KIND,
        capacity_authority_kind=(
            "action_ball_full_mdp_code_owned_diagnostic_n2_capacity_v1"
        ),
        formal_capacity_receipt_sha256=None,
        flight_capacity=2,
        scene_entity_names=(
            "action_ball_flight_ball_000",
            "action_ball_flight_ball_001",
        ),
        prim_paths=(
            "{ENV_REGEX_NS}/ActionBallFlightBall_000",
            "{ENV_REGEX_NS}/ActionBallFlightBall_001",
        ),
        ball_radius_m=0.02,
        ball_mass_kg=0.0034,
        park_position_env_m=scene.PARK_POSITION_ENV_M,
        collision_enabled=True,
        gravity_enabled=True,
    )
    cfg = config.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    cfg.commands = types.SimpleNamespace(
        motion=motion_cfg,
        racket_target=racket_cfg,
    )
    cfg.action_ball_full_mdp_ball_scene_spec = scene_spec
    cfg.action_ball_full_mdp_scene_capacity = 2
    cfg.action_ball_full_mdp_scene_capacity_authority_kind = (
        scene_spec.capacity_authority_kind
    )
    cfg.action_ball_full_mdp_capacity_receipt_sha256 = ""
    cfg.action_ball_full_mdp_scene_spec_sha256 = scene_spec.canonical_sha256
    cfg.action_ball_full_mdp_runtime_construction_status = "HOLD"
    cfg.action_ball_full_mdp_family_role = "A"
    cfg.scene = types.SimpleNamespace(
        num_envs=num_envs,
        robot=types.SimpleNamespace(
            spawn=types.SimpleNamespace(usd_path=_EXACT_SPLIT_ASSET)
        )
    )
    cfg.sim = types.SimpleNamespace(device=device_name)
    cfg.checkpoint_path = None
    cfg.checkpoint_tolerant = False
    env_origins = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    live_scene = _Scene(
        env_origins=env_origins,
        cfg=types.SimpleNamespace(replicate_physics=True),
        env_prim_paths=tuple(
            f"/World/envs/env_{index}" for index in range(num_envs)
        ),
    )
    for name in scene_spec.scene_entity_names:
        root = torch.zeros(
            (num_envs, 13), dtype=torch.float32, device=device
        )
        root[:, :3] = env_origins + torch.tensor(
            scene.PARK_POSITION_ENV_M,
            dtype=torch.float32,
            device=device,
        )
        root[:, 3] = 1.0
        setattr(live_scene, name, _SceneAsset(root))
    env = types.SimpleNamespace(
        cfg=cfg,
        scene=live_scene,
        _action_ball_full_mdp_manager_construction_state=(
            "command_manager_ready"
        ),
        command_manager=types.SimpleNamespace(
            get_term=lambda name: {"motion": motion, "racket_target": racket}[
                name
            ]
        ),
        num_envs=num_envs,
        device=device_name,
        step_dt=0.02,
        max_episode_length=500,
    )
    motion._env = env
    # Match real CommandManager construction: the live term owns a distinct
    # configclass instance rather than the EnvCfg template object.
    racket.cfg = live_racket_cfg
    racket._env = env
    strike = _load(
        f"{_PKG}.action_ball_strike_fact_device",
        "action_ball_strike_fact_device.py",
    )
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_strike_fact_device_coordinator = (
        strike.ActionBallStrikeFactDeviceCoordinator(
            num_envs=num_envs,
            device=device,
            observation_projection_mode=(
                strike.OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP
            ),
        )
    )
    return env, motion, racket, scene


def _load_mdp_test_module(name: str, filename: str):
    from test_reward_flags_mdp import _load

    return _load(name, filename)


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_factory_cold_failure_retires_genesis_without_partial_env_install(
    monkeypatch, device_name
):
    torch = pytest.importorskip("torch")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    issued = []
    real_issue = genesis.issue_action_ball_full_mdp_reset_genesis

    def capture_issue(*, num_envs, device):
        value = real_issue(num_envs=num_envs, device=device)
        issued.append(value)
        return value

    monkeypatch.setattr(
        genesis, "issue_action_ball_full_mdp_reset_genesis", capture_issue
    )
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_reward_hold_subject"
    )
    assert factory.NUMERIC_REWARD_FACTORY_HOLD_REASONS == (
        "constructed_runtime_reward_graph_producer_absent",
        "real_four_shot_unit_income_phase_support_producer_absent",
        "launcher_owned_finite_candidate_set_producer_absent",
        "fourteen_live_reward_consumers_not_factory_bound",
    )
    profile = __import__("action_ball_device_profile_authority")
    constructed_profiles = []
    real_construct_profile = profile.construct_device_profile_authority

    def capture_profile(spec, *, device, expected_support_size):
        value = real_construct_profile(
            spec,
            device=device,
            expected_support_size=expected_support_size,
        )
        constructed_profiles.append((spec, value))
        return value

    monkeypatch.setattr(
        profile, "construct_device_profile_authority", capture_profile
    )
    landing = _load_mdp_test_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_landing_outcome_device",
        "action_ball_landing_outcome_device.py",
    )
    recovery = __import__("action_ball_continuous_recovery_device")
    constructed_r06 = []
    r07_calls = []
    real_construct_r06 = landing.construct_diagnostic_n2_no_save_r06
    real_construct_r07 = (
        recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner
    )

    def capture_r06(**kwargs):
        value = real_construct_r06(**kwargs)
        constructed_r06.append((kwargs, value))
        return value

    def capture_r07(
        *,
        env,
        motion_owner,
        action_epoch_owner,
        motion_parent_authority,
        motion_parent_receipt,
    ):
        kwargs = {
            "env": env,
            "motion_owner": motion_owner,
            "action_epoch_owner": action_epoch_owner,
            "motion_parent_authority": motion_parent_authority,
            "motion_parent_receipt": motion_parent_receipt,
        }
        r07_calls.append(kwargs)
        return real_construct_r07(**kwargs)

    monkeypatch.setattr(
        landing, "construct_diagnostic_n2_no_save_r06", capture_r06
    )
    monkeypatch.setattr(
        recovery,
        "construct_action_ball_full_mdp_diagnostic_n2_recovery_owner",
        capture_r07,
    )
    env, motion, racket, _scene = _exact_single_action_lean_factory_env(
        device_name=device_name
    )
    live_r03 = racket.action_ball_full_mdp_r03_owner()
    env_before = dict(vars(env))
    with pytest.raises(factory.ActionBallFullMdpRuntimeFactoryHold) as caught:
        factory.construct_action_ball_full_mdp_runtime_graph(env)
    message = str(caught.value)
    assert message
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")
    assert len(issued) == 1
    assert len(constructed_profiles) == 1
    # This existing cold-failure fixture intentionally stops at its first real
    # producer HOLD, currently the schema-2 Racket FK source before R07.
    assert r07_calls == []
    assert racket.action_ball_full_mdp_r03_owner() is live_r03
    _spec, (profile_owner, profile_receipt) = constructed_profiles[0]
    assert type(profile_owner) is profile.DeviceProfileAuthorityOwner
    assert type(profile_receipt) is profile.DeviceProfileReceipt
    projection = profile_owner.require_owned_r05_profile(profile_receipt)
    assert projection.targets_xy_m.shape == (3, 2)
    assert projection.targets_xy_m.device == torch.device(device_name)
    assert dict(vars(env)) == env_before
    # Physical consumed only its third exact projection.  The issuer retires
    # its retained record at the downstream HOLD, so the unissued env/R05
    # consumers cannot leak across repeated failed env construction.
    for projector in (
        issued[0].authority.require_owned_full_mdp_reset_genesis,
        issued[0].authority.require_owned_r05_genesis,
    ):
        with pytest.raises(
            genesis.ActionBallFullMdpResetGenesisError,
            match="foreign or unregistered",
        ):
            projector(
                issued[0].receipt,
                device=torch.device(device_name),
                num_envs=2,
            )
    assert not any(
        reason in message for reason in factory.NUMERIC_REWARD_FACTORY_HOLD_REASONS
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_factory_builds_exact_scene_port_without_fabricated_contact_capture(
    monkeypatch, device_name
):
    torch = pytest.importorskip("torch")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_physical_scene_subject"
    )
    env, _motion, _racket, scene = _exact_single_action_lean_factory_env(
        device_name=device_name
    )
    env_before = dict(vars(env))

    seed = factory._construct_offside_seed(env)
    profile = factory._construct_offside_device_profile(seed)
    question = factory._construct_offside_question_inputs(profile)
    physical = factory._construct_offside_physical_scene_inputs(question)

    capacity = __import__("action_ball_full_mdp_diagnostic_capacity")
    assert type(physical.scene_port) is scene.IsaacLabPhysicalFlightScenePort
    assert (
        type(physical.diagnostic_capacity_binding)
        is capacity.DiagnosticN2CapacityBinding
    )
    assert not hasattr(
        physical.diagnostic_capacity_binding, "capacity_receipt_sha256"
    )
    assert physical.scene_port.read_state_env().shape == (2, 2, 13)
    physical_module = sys.modules[
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_physical_flight_device"
    ]
    assert (
        type(physical.physical_owner)
        is physical_module.ActionBallPhysicalFlightDeviceOwner
    )
    physical_question = __import__("action_ball_physical_question_device")
    assert (
        type(physical.physical_question_core)
        is physical_question.PhysicalQuestionNumericCore
    )
    assert physical.physical_owner.scene_snapshot().state_env_f32.shape == (
        2,
        2,
        13,
    )
    assert not hasattr(physical, "producer_absence_capture")
    assert dict(vars(env)) == env_before
    # Cold port/owner construction leaves both atomic-install consumers live
    # until the caller either completes the graph or explicitly retires the
    # failed bundle.
    env_projection = seed.reset_genesis_authority.require_owned_full_mdp_reset_genesis(
        seed.reset_genesis_receipt,
        device=torch.device(device_name),
        num_envs=2,
    )
    r05_projection = seed.reset_genesis_authority.require_owned_r05_genesis(
        seed.reset_genesis_receipt,
        device=torch.device(device_name),
        num_envs=2,
    )
    epoch_projection = (
        seed.reset_genesis_authority.require_owned_action_epoch_genesis(
            seed.reset_genesis_receipt,
            device=torch.device(device_name),
            num_envs=2,
        )
    )
    assert (
        env_projection.world_reset_identity
        is r05_projection.world_reset_identity
        is epoch_projection.world_reset_identity
    )
    assert (
        env_projection.world_reset_identity
        is physical.physical_owner._genesis_world_reset_identity
    )
    assert env_projection.reset_generations.shape == (2,)
    assert r05_projection.reset_generations.shape == (2,)
    assert epoch_projection.reset_generations.shape == (2,)
    if torch.device(device_name).type == "cpu":
        assert torch.equal(
            env_projection.reset_generations,
            torch.ones_like(env_projection.reset_generations),
        )
        assert torch.equal(
            r05_projection.reset_generations,
            torch.ones_like(r05_projection.reset_generations),
        )
        assert torch.equal(
            epoch_projection.reset_generations,
            torch.ones_like(epoch_projection.reset_generations),
        )


def _construct_exact_factory_physical_for_r06(factory):
    """Build the real cold factory prefix consumed by canonical R06."""

    env, _motion, _racket, _scene = _exact_single_action_lean_factory_env()
    seed = factory._construct_offside_seed(env)
    profile = factory._construct_offside_device_profile(seed)
    question = factory._construct_offside_question_inputs(profile)
    physical = factory._construct_offside_physical_scene_inputs(question)
    return env, seed, physical


def test_factory_physical_constructs_canonical_r06_without_legacy_uid_plane():
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_canonical_r06_subject"
    )
    env, _seed, physical = _construct_exact_factory_physical_for_r06(factory)
    landing = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_landing_outcome_device"
    )

    assert not hasattr(
        physical.physical_owner, "_action_epoch_flight_action_uid"
    )
    assert landing._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS == {}
    r06_owner = landing.construct_diagnostic_n2_no_save_r06(
        env=env,
        physical_owner=physical.physical_owner,
        diagnostic_n2_capacity_binding=physical.diagnostic_capacity_binding,
    )

    assert type(r06_owner) is landing.ActionBallLandingOutcomeDeviceCoordinator
    assert landing._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS == {}
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")


def test_canonical_r06_rejects_overlapping_live_scene_asset_storage_before_install():
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_r06_scene_overlap_subject"
    )
    env, seed, physical = _construct_exact_factory_physical_for_r06(factory)
    landing = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_landing_outcome_device"
    )
    first_asset, second_asset = physical.scene_port.assets
    first_root = first_asset.data.root_state_w
    second_asset.data.root_state_w = first_root

    assert env.num_envs == physical.physical_owner.num_envs == 2
    assert physical.scene_port.num_envs == 2
    assert first_root.shape == second_asset.data.root_state_w.shape == (2, 13)
    assert landing._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS == {}
    with pytest.raises(
        landing.LandingOutcomeDeviceError,
        match="scene asset 1 root storage overlaps asset 0",
    ):
        landing.construct_diagnostic_n2_no_save_r06(
            env=env,
            physical_owner=physical.physical_owner,
            diagnostic_n2_capacity_binding=(
                physical.diagnostic_capacity_binding
            ),
        )

    assert landing._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS == {}
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")
    # Physical already consumed its one-shot genesis projection.  This
    # counterexample proves failure-before-R06-registry/env-install only; it
    # deliberately makes no rollback claim about the constructed Physical.
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    with pytest.raises(
        genesis.ActionBallFullMdpResetGenesisError,
        match="already projected to physical",
    ):
        seed.reset_genesis_authority.require_owned_physical_genesis(
            seed.reset_genesis_receipt,
            device=seed.device,
            num_envs=2,
        )


def test_factory_binds_r06_to_diagnostic_device_r05_reset_owner():
    """Lock the live reset dependency; AppLauncher supplies semantic evidence."""

    tree = ast.parse(FACTORY_MODULE_PATH.read_text(encoding="utf-8"))
    construct = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_construct_offside_lean_runtime"
    )
    methods = tuple(
        node.func.attr
        for node in ast.walk(construct)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.value is not None
        and ast.unparse(node.func.value) == "inputs.r06_owner"
        and "device_r05_reset_owner" in node.func.attr
    )
    assert methods == ("bind_diagnostic_n2_device_r05_reset_owner",)


def test_failure_after_physical_genesis_consumption_retires_bundle_without_rollback(
    monkeypatch,
):
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_physical_burn_subject"
    )
    env, _motion, _racket, _scene = _exact_single_action_lean_factory_env()
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    physical = __import__(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_physical_flight_device",
        fromlist=("ActionBallPhysicalFlightDeviceOwner",),
    )
    seed = factory._construct_offside_seed(env)
    profile = factory._construct_offside_device_profile(seed)
    question = factory._construct_offside_question_inputs(profile)
    real_snapshot = physical.ActionBallPhysicalFlightDeviceOwner.scene_snapshot
    discard_calls = []
    retire_calls = []
    real_retire = (
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis
    )

    def fail_after_constructor(self):
        real_snapshot(self)
        raise RuntimeError("post-constructor probe failed")

    def forbid_fake_discard(**kwargs):
        discard_calls.append(kwargs)
        raise AssertionError("projected genesis must not be discarded")

    def capture_retire(**kwargs):
        retire_calls.append(kwargs)
        return real_retire(**kwargs)

    monkeypatch.setattr(
        physical.ActionBallPhysicalFlightDeviceOwner,
        "scene_snapshot",
        fail_after_constructor,
    )
    monkeypatch.setattr(
        genesis,
        "discard_unpublished_action_ball_full_mdp_reset_genesis",
        forbid_fake_discard,
    )
    monkeypatch.setattr(
        genesis,
        "retire_failed_unpublished_action_ball_full_mdp_reset_genesis",
        capture_retire,
    )

    with pytest.raises(
        factory.ActionBallFullMdpRuntimeFactoryHold,
        match="remaining unpublished genesis consumers were retired",
    ):
        factory._construct_offside_physical_scene_inputs(question)

    assert discard_calls == []
    assert len(retire_calls) == 1
    # The third projection was really consumed; no self-receipt/zero-call gate
    # stands in for that irreversible fact.
    with pytest.raises(
        genesis.ActionBallFullMdpResetGenesisError,
        match="foreign or unregistered",
    ):
        seed.reset_genesis_authority.require_owned_physical_genesis(
            seed.reset_genesis_receipt,
            device=seed.device,
            num_envs=2,
        )
    for projector in (
        seed.reset_genesis_authority.require_owned_full_mdp_reset_genesis,
        seed.reset_genesis_authority.require_owned_r05_genesis,
    ):
        with pytest.raises(
            genesis.ActionBallFullMdpResetGenesisError,
            match="foreign or unregistered",
        ):
            projector(
                seed.reset_genesis_receipt,
                device=seed.device,
                num_envs=2,
            )


@pytest.mark.parametrize(
    "mutation,expected",
    (
        (
            "foreign_env_cfg",
            "env_cfg_is_not_exact_registered_full_mdp_leaf",
        ),
        ("rewritten_env_role", "live_env_cfg_family_authority_rejected"),
        ("cfg_env_n_differ", "cfg_env_num_envs_differ"),
        ("formal_scene", "scene_is_not_exact_diagnostic_n2_spec"),
        (
            "diagnostic_false",
            "racket_cfg.action_ball_diagnostic_unauthorized_is_not_true",
        ),
        (
            "live_diagnostic_false",
            "live_racket_cfg.action_ball_diagnostic_unauthorized_is_not_true",
        ),
        ("foreign_racket_cfg", "live_racket_cfg_exact_type_differs"),
    ),
)
def test_factory_rejects_noncanary_modes_before_genesis_issue(
    monkeypatch, mutation, expected
):
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    issued = []
    real_issue = genesis.issue_action_ball_full_mdp_reset_genesis

    def capture_issue(*, num_envs, device):
        issued.append((num_envs, device))
        return real_issue(num_envs=num_envs, device=device)

    monkeypatch.setattr(
        genesis, "issue_action_ball_full_mdp_reset_genesis", capture_issue
    )
    factory = _load_factory_subject(
        f"_action_ball_full_mdp_runtime_factory_preissue_{mutation}_subject"
    )
    env, motion, racket, scene = _exact_single_action_lean_factory_env()
    if mutation == "foreign_env_cfg":
        foreign = types.SimpleNamespace(**vars(env.cfg))
        env.cfg = foreign
        motion._env = env
        racket._env = env
    elif mutation == "rewritten_env_role":
        env.cfg.action_ball_full_mdp_family_role = "C"
    elif mutation == "cfg_env_n_differ":
        env.num_envs = 3
    elif mutation == "formal_scene":
        env.cfg.action_ball_full_mdp_ball_scene_spec = (
            scene.ActionBallFullMdpBallSceneSpec(
                schema_version=scene.SCHEMA_VERSION,
                kind=scene.SCENE_SPEC_KIND,
                contract_source_sha256=scene.CONTRACT_SOURCE_SHA256,
                capacity_receipt_sha256="0" * 64,
                flight_capacity=2,
                scene_entity_names=(
                    "action_ball_flight_ball_000",
                    "action_ball_flight_ball_001",
                ),
                prim_paths=(
                    "{ENV_REGEX_NS}/ActionBallFlightBall_000",
                    "{ENV_REGEX_NS}/ActionBallFlightBall_001",
                ),
                ball_radius_m=0.02,
                ball_mass_kg=0.0034,
                park_position_env_m=scene.PARK_POSITION_ENV_M,
                collision_enabled=True,
                gravity_enabled=True,
            )
        )
    elif mutation == "diagnostic_false":
        env.cfg.commands.racket_target.action_ball_diagnostic_unauthorized = False
    elif mutation == "live_diagnostic_false":
        racket.cfg.action_ball_diagnostic_unauthorized = False
    else:
        racket.cfg = object()

    with pytest.raises(
        factory.ActionBallFullMdpRuntimeFactoryHold,
        match=expected,
    ) as caught:
        factory.construct_action_ball_full_mdp_runtime_graph(env)

    assert "before reset-genesis issue" in str(caught.value)
    assert issued == []
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")


def test_factory_calls_v3_asset_consumer_before_genesis_or_owner_construction(
    monkeypatch,
):
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_asset_consumer_subject"
    )
    env, _motion, _racket, _scene = _exact_single_action_lean_factory_env()
    split_asset = importlib.import_module(
        "whole_body_tracking.tasks.tracking.config.agibot_a3."
        "action_ball_full_mdp_split_asset"
    )
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    trace = []

    def reject_asset():
        trace.append("asset_recompute")
        raise split_asset.ActionBallFullMdpSplitAssetError(
            "injected reconstructed-source mismatch"
        )

    reject_asset.__module__ = split_asset.__name__

    monkeypatch.setattr(
        split_asset,
        "require_action_ball_full_mdp_split_asset",
        reject_asset,
    )
    monkeypatch.setattr(
        genesis,
        "issue_action_ball_full_mdp_reset_genesis",
        lambda **_kwargs: trace.append("genesis"),
    )
    with pytest.raises(
        factory.ActionBallFullMdpRuntimeFactoryHold,
        match="failed independent reconstruction before reset genesis",
    ):
        factory.construct_action_ball_full_mdp_runtime_graph(env)

    assert trace == ["asset_recompute"]
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")


def test_foreign_live_r03_retires_consumed_physical_genesis_without_install():
    torch = pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_foreign_r03_subject"
    )
    env, _motion, racket, _scene = _exact_single_action_lean_factory_env()
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    seed = factory._construct_offside_seed(env)
    profile = factory._construct_offside_device_profile(seed)
    question = factory._construct_offside_question_inputs(profile)
    physical = factory._construct_offside_physical_scene_inputs(question)
    # This test owns only the R03 identity failure boundary.  The production
    # recurring constructor requires a fully initialized CommandManager term,
    # which this focused scene fixture deliberately does not fabricate.
    recurring = factory._OffsideRecurringQuestionInputs(
        physical=physical,
        recurring_question_bundle=object(),
    )
    racket._action_ball_strike_fact_device_coordinator = object()

    with pytest.raises(
        factory.ActionBallFullMdpRuntimeFactoryHold,
        match="remaining unpublished genesis consumers were retired",
    ):
        factory._construct_offside_r03_r06(env, recurring)

    for projector in (
        seed.reset_genesis_authority.require_owned_full_mdp_reset_genesis,
        seed.reset_genesis_authority.require_owned_r05_genesis,
        seed.reset_genesis_authority.require_owned_action_epoch_genesis,
    ):
        with pytest.raises(
            genesis.ActionBallFullMdpResetGenesisError,
            match="foreign or unregistered",
        ):
            projector(
                seed.reset_genesis_receipt,
                device=torch.device("cpu"),
                num_envs=2,
            )
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")


def test_factory_names_the_exact_live_motion_owner_missing_diagnostic_marker(
    monkeypatch,
):
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    issued = []
    monkeypatch.setattr(
        genesis,
        "issue_action_ball_full_mdp_reset_genesis",
        lambda **kwargs: issued.append(kwargs),
    )
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_exact_motion_marker"
    )
    env, motion, racket, _scene = _exact_single_action_lean_factory_env()
    motion._canonical_diagnostic_unauthorized = False

    with pytest.raises(factory.ActionBallFullMdpRuntimeFactoryHold) as caught:
        factory.construct_action_ball_full_mdp_runtime_graph(env)

    message = str(caught.value)
    assert (
        "motion_owner._canonical_diagnostic_unauthorized_is_not_true"
        in message
    )
    assert "before reset-genesis issue" in message
    assert issued == []


def test_reset_genesis_is_now_one_concrete_production_implementation():
    implementations = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.name.startswith("._"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "require_owned_full_mdp_reset_genesis"
            ):
                implementations.append(path)
    assert implementations == [
        SOURCE_ROOT / "action_ball_full_mdp_reset_genesis.py"
    ]


def test_factory_constructs_real_genesis_offside_without_any_env_install(
    monkeypatch,
):
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_offside_genesis_subject"
    )
    issued = []
    real_issue = genesis.issue_action_ball_full_mdp_reset_genesis

    def capture_issue(*, num_envs, device):
        value = real_issue(num_envs=num_envs, device=device)
        issued.append(value)
        return value

    monkeypatch.setattr(
        genesis, "issue_action_ball_full_mdp_reset_genesis", capture_issue
    )
    cadence = __import__("action_ball_motion_cadence_device")
    _parent, _receipt, motion_profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    motion = types.SimpleNamespace(
        _action_ball_continuous_motion_profile=motion_profile
    )
    racket = types.SimpleNamespace(cfg=object())
    env = types.SimpleNamespace(
        _action_ball_full_mdp_manager_construction_state="command_manager_ready",
        cfg=types.SimpleNamespace(
            commands=types.SimpleNamespace(racket_target=object())
        ),
        command_manager=types.SimpleNamespace(
            get_term=lambda name: {"motion": motion, "racket_target": racket}[name]
        ),
        num_envs=2,
        device="cpu",
    )

    seed = factory._construct_offside_seed(env)

    assert len(issued) == 1
    assert seed.motion_owner is motion
    assert seed.racket_owner is racket
    assert seed.reset_genesis_authority is issued[0].authority
    assert seed.reset_genesis_receipt is issued[0].receipt
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")
    issue = issued[0]
    device = __import__("torch").device("cpu")
    env_projection = issue.authority.require_owned_full_mdp_reset_genesis(
        issue.receipt, device=device, num_envs=2
    )
    r05_projection = issue.authority.require_owned_r05_genesis(
        issue.receipt, device=device, num_envs=2
    )
    assert env_projection.world_reset_identity is r05_projection.world_reset_identity


def test_factory_cold_binds_exact_motion_parent_before_first_command_compute(
):
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_motion_parent_subject"
    )
    env, motion, _racket, _scene = _exact_single_action_lean_factory_env()
    seed = factory._construct_offside_seed(env)
    assert motion._action_ball_continuous_parent_authority_binding is None

    # This is the real consumer-side precondition reached by the first Motion
    # command compute.  Before the factory bind, a legal live Motion owner is
    # rejected even though it retained the correct profile bytes.
    with pytest.raises(
        RuntimeError,
        match="requires external C01/C02 authority binding",
    ):
        motion._require_action_ball_continuous_parent_authorities()

    factory._require_precommand_motion_cadence(seed)

    motion._require_action_ball_continuous_parent_authorities()
    assert motion._action_ball_continuous_parent_authority_binding is not None
    assert motion._action_ball_continuous_schedule_projection is not None


def test_motion_parent_bind_failure_discards_unpublished_genesis(monkeypatch):
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_motion_parent_failure_subject"
    )
    env, _motion, _racket, _scene = _exact_single_action_lean_factory_env()
    seed = factory._construct_offside_seed(env)
    cadence = __import__("action_ball_motion_cadence_device")
    genesis = __import__("action_ball_full_mdp_reset_genesis")

    def fail_bind(self, motion_owner, receipt):
        del self, motion_owner, receipt
        raise RuntimeError("injected exact parent bind failure")

    monkeypatch.setattr(
        cadence.DiagnosticMotionParentScheduleAuthority,
        "bind_exact_parent_schedule",
        fail_bind,
    )
    with pytest.raises(
        factory.ActionBallFullMdpRuntimeFactoryHold,
        match="parent-schedule cold binding failed",
    ):
        factory._require_precommand_motion_cadence(seed)

    for projector in (
        seed.reset_genesis_authority.require_owned_full_mdp_reset_genesis,
        seed.reset_genesis_authority.require_owned_r05_genesis,
        seed.reset_genesis_authority.require_owned_physical_genesis,
    ):
        with pytest.raises(
            genesis.ActionBallFullMdpResetGenesisError,
            match="foreign or unregistered",
        ):
            projector(
                seed.reset_genesis_receipt,
                device=seed.device,
                num_envs=2,
            )


def test_factory_has_one_exact_motion_r07_cold_binding_callpoint():
    tree = ast.parse(FACTORY_MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_construct_offside_r07"
    )
    source = ast.unparse(function)
    constructor_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        == "construct_action_ball_full_mdp_diagnostic_n2_recovery_owner"
    ]
    assert len(constructor_calls) == 1
    assert constructor_calls[0].args == []
    assert tuple(keyword.arg for keyword in constructor_calls[0].keywords) == (
        "env",
        "motion_owner",
        "action_epoch_owner",
        "motion_parent_authority",
        "motion_parent_receipt",
    )
    assert tuple(
        ast.unparse(keyword.value) for keyword in constructor_calls[0].keywords
    ) == (
        "env",
        "seed.motion_owner",
        "epoch_owner",
        "seed.motion_parent_authority",
        "seed.motion_parent_receipt",
    )
    assert source.count(
        "'bind_action_ball_continuous_r07_ready_projection'"
    ) == 2
    assert source.count("'require_owned_motion_ready_projection'") == 2
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "motion_bind"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    assert ast.unparse(calls[0].args[0]) == "bundle"
    assert tuple(keyword.arg for keyword in calls[0].keywords) == (
        "require_owned_ready_projection",
    )
    assert ast.unparse(calls[0].keywords[0].value) == "ready_validator"

    returns = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
    ]
    assert len(returns) == 1
    assert ast.unparse(returns[0].value) == "(bundle, bundle.plant_fact_adapter)"


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_factory_r07_consumes_the_exact_seed_parent_identity(
    monkeypatch, device_name
):
    torch = pytest.importorskip("torch")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from test_action_ball_continuous_recovery_live_facts import (
        _subject as exact_r07_subject,
    )

    genesis = __import__("action_ball_full_mdp_reset_genesis")
    recovery = __import__("action_ball_continuous_recovery_device")
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_r07_exact_parent_"
        + device_name.replace(":", "_")
    )
    device = torch.device(device_name)
    env, motion, _robot, _sensor = exact_r07_subject(
        monkeypatch,
        device=device,
    )
    epoch_owner = motion._diagnostic_test_epoch_owner
    issue = genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2,
        device=device,
    )
    seed = types.SimpleNamespace(
        motion_owner=motion,
        motion_parent_authority=(
            motion._diagnostic_test_motion_parent_authority
        ),
        motion_parent_receipt=motion._diagnostic_test_motion_parent_receipt,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
    )
    inputs = types.SimpleNamespace(
        recurring=types.SimpleNamespace(
            physical=types.SimpleNamespace(
                question=types.SimpleNamespace(
                    profile=types.SimpleNamespace(seed=seed)
                )
            )
        )
    )
    calls = []
    real_construct_r07 = (
        recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner
    )

    def capture_r07(**kwargs):
        calls.append(dict(kwargs))
        return real_construct_r07(**kwargs)

    monkeypatch.setattr(
        recovery,
        "construct_action_ball_full_mdp_diagnostic_n2_recovery_owner",
        capture_r07,
    )
    env_before = dict(vars(env))
    bundle, adapter = factory._construct_offside_r07(
        env,
        inputs,
        epoch_owner,
    )

    assert len(calls) == 1
    call = calls[0]
    assert tuple(call) == (
        "env",
        "motion_owner",
        "action_epoch_owner",
        "motion_parent_authority",
        "motion_parent_receipt",
    )
    assert call["env"] is env
    assert call["motion_owner"] is motion
    assert call["action_epoch_owner"] is epoch_owner
    assert call["motion_parent_authority"] is seed.motion_parent_authority
    assert call["motion_parent_receipt"] is seed.motion_parent_receipt
    assert type(bundle) is recovery.DiagnosticN2ContinuousRecoveryBundle
    assert bundle.motion_owner is motion
    assert bundle.action_epoch_owner is epoch_owner
    assert bundle.motion_parent_authority is seed.motion_parent_authority
    assert bundle.motion_parent_receipt is seed.motion_parent_receipt
    assert adapter is bundle.plant_fact_adapter
    assert motion._action_ball_continuous_r07_ready_owner is bundle
    assert (
        motion._action_ball_continuous_r07_ready_validator.__self__
        is bundle
    )
    parent_identity = seed.motion_parent_authority.project_bound_action_identity(
        seed.motion_parent_receipt,
        motion_owner=motion,
    )
    assert parent_identity.authority is seed.motion_parent_authority
    assert parent_identity.motion_owner is motion
    assert parent_identity.action_uid == parent_identity.action_uids[0]
    assert dict(vars(env)) == env_before
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")
    genesis.discard_unpublished_action_ball_full_mdp_reset_genesis(
        authority=issue.authority,
        receipt=issue.receipt,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "omit_parent_pair",
        "swap_parent_pair",
        "foreign_authority",
        "foreign_receipt",
        "same_value_foreign_pair",
    ),
)
def test_factory_r07_parent_identity_counterexamples_retire_before_publish(
    monkeypatch, mutation
):
    torch = pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    genesis = __import__("action_ball_full_mdp_reset_genesis")
    cadence = __import__("action_ball_motion_cadence_device")
    recovery = __import__("action_ball_continuous_recovery_device")
    from test_action_ball_continuous_recovery_live_facts import (
        _subject as exact_r07_subject,
    )

    wrapper_calls = []
    equal_profile_counterexamples = []
    real_construct_r07 = (
        recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner
    )

    def mutate_r07_parent(
        *,
        env,
        motion_owner,
        action_epoch_owner,
        motion_parent_authority,
        motion_parent_receipt,
    ):
        wrapper_calls.append(
            (
                env,
                motion_owner,
                action_epoch_owner,
                motion_parent_authority,
                motion_parent_receipt,
            )
        )
        foreign_authority, foreign_receipt, foreign_profile = (
            cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
        )
        retained_profile = motion_parent_authority.require_owned_motion_profile(
            motion_parent_receipt
        )
        equal_profile_counterexamples.append(
            foreign_profile == retained_profile
        )
        if mutation == "omit_parent_pair":
            return real_construct_r07(
                env=env,
                motion_owner=motion_owner,
                action_epoch_owner=action_epoch_owner,
            )
        if mutation == "swap_parent_pair":
            candidate_authority = motion_parent_receipt
            candidate_receipt = motion_parent_authority
        elif mutation == "foreign_authority":
            candidate_authority = foreign_authority
            candidate_receipt = motion_parent_receipt
        elif mutation == "foreign_receipt":
            candidate_authority = motion_parent_authority
            candidate_receipt = foreign_receipt
        else:
            # Both foreign objects expose the same profile values as the real
            # producer.  Value equality is deliberately insufficient: this
            # authority was never bound to the live Motion owner.
            candidate_authority = foreign_authority
            candidate_receipt = foreign_receipt
        return real_construct_r07(
            env=env,
            motion_owner=motion_owner,
            action_epoch_owner=action_epoch_owner,
            motion_parent_authority=candidate_authority,
            motion_parent_receipt=candidate_receipt,
        )

    monkeypatch.setattr(
        recovery,
        "construct_action_ball_full_mdp_diagnostic_n2_recovery_owner",
        mutate_r07_parent,
    )
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_r07_parent_mutation_"
        + mutation
    )
    env, motion, _robot, _sensor = exact_r07_subject(
        monkeypatch,
        device=torch.device("cpu"),
    )
    epoch_owner = motion._diagnostic_test_epoch_owner
    issue = genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2,
        device=torch.device("cpu"),
    )
    seed = types.SimpleNamespace(
        motion_owner=motion,
        motion_parent_authority=(
            motion._diagnostic_test_motion_parent_authority
        ),
        motion_parent_receipt=motion._diagnostic_test_motion_parent_receipt,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
    )
    inputs = types.SimpleNamespace(
        recurring=types.SimpleNamespace(
            physical=types.SimpleNamespace(
                question=types.SimpleNamespace(
                    profile=types.SimpleNamespace(seed=seed)
                )
            )
        )
    )
    env_before = dict(vars(env))
    with pytest.raises(factory.ActionBallFullMdpRuntimeFactoryHold) as caught:
        factory._construct_offside_r07(env, inputs, epoch_owner)

    assert "exact R07 diagnostic constructor" in str(caught.value)
    assert len(wrapper_calls) == 1
    assert equal_profile_counterexamples == [True]
    assert dict(vars(env)) == env_before
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")
    for projector in (
        issue.authority.require_owned_full_mdp_reset_genesis,
        issue.authority.require_owned_r05_genesis,
    ):
        with pytest.raises(
            genesis.ActionBallFullMdpResetGenesisError,
            match="foreign or unregistered",
        ):
            projector(
                issue.receipt,
                device=torch.device("cpu"),
                num_envs=2,
            )


def test_motion_r07_cold_bind_rejects_foreign_owner_validator_pair():
    pytest.importorskip("torch")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    _factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_motion_r07_counterexample"
    )
    _env, motion, _racket, _scene = _exact_single_action_lean_factory_env()

    class _Owner:
        def require_owned_motion_ready_projection(
            self, projection, *, owner_kind
        ):
            del projection, owner_kind

    owner = _Owner()
    foreign = _Owner()
    with pytest.raises(TypeError, match="owner-bound validator"):
        motion.bind_action_ball_continuous_r07_ready_projection(
            owner,
            require_owned_ready_projection=(
                foreign.require_owned_motion_ready_projection
            ),
        )

    assert motion._action_ball_continuous_r07_ready_owner is None
    assert motion._action_ball_continuous_r07_ready_validator is None


def _load_pod_lean_reward_stack():
    pytest.importorskip("torch")
    pytest.importorskip("isaaclab.managers")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    try:
        factory = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_runtime_factory"
        )
        rewards = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_rewards"
        )
        epoch = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
        )
    except ModuleNotFoundError as exc:
        if exc.name is not None and (
            exc.name.startswith("omni")
            or exc.name.startswith("isaaclab")
        ):
            pytest.skip("full Isaac/Kit Reward stack is unavailable")
        raise
    return factory, rewards, epoch


def _diagnostic_lean_reward_bundle(*, device_name="cpu"):
    factory, rewards, epoch = _load_pod_lean_reward_stack()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    owner = epoch.ActionEpochOwner(
        num_envs=2,
        device=device,
        shot_slot_capacity=1,
        initial_reset_generation=torch.ones(
            2, dtype=torch.int64, device=device
        ),
    )
    bundle = rewards.materialize_diagnostic_n2_reward_manager_cfg(
        epoch_owner=owner
    )
    return factory, rewards, epoch, owner, bundle


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_code_owned_lean_reward_bundle_materializes_exact_graph_and_cfg(
    device_name,
):
    _factory, rewards, epoch, owner, bundle = _diagnostic_lean_reward_bundle(
        device_name=device_name
    )
    assert type(bundle) is rewards.DiagnosticN2RewardManagerBundle
    assert type(bundle.graph) is rewards.LeanActionEpochRewardGraph
    assert bundle.graph.epoch_owner is owner
    assert len(bundle.manager_cfg) == rewards.MANAGER_TERM_COUNT
    assert tuple(bundle.manager_cfg) == rewards.MANAGER_NAMES
    assert tuple(term.func for term in bundle.manager_cfg.values()) == (
        rewards.REWARD_TERM_CALLABLES
    )
    assert bundle.profile_kind == rewards.DIAGNOSTIC_N2_REWARD_PROFILE_KIND
    assert bundle.diagnostic_unauthorized is True
    assert owner.device.type == torch.device(device_name).type
    assert owner.shot_slot_capacity == 1
    assert epoch.REWARD_CONSUMER_ORDER == rewards.ORDERED_CONSUMERS


def test_fresh_lean_materializer_has_no_caller_numeric_or_receipt_seam():
    _factory, rewards, _epoch = _load_pod_lean_reward_stack()
    signature = inspect.signature(
        rewards.materialize_diagnostic_n2_reward_manager_cfg
    )
    assert tuple(signature.parameters) == ("epoch_owner",)
    assert signature.parameters["epoch_owner"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        rewards.materialize_diagnostic_n2_reward_manager_cfg(
            epoch_owner=object(),
            weights={},
        )


def test_lean_materializer_rejects_foreign_epoch_before_manager_install():
    _factory, rewards, _epoch = _load_pod_lean_reward_stack()
    with pytest.raises(
        rewards.LeanRewardConstructionHold,
        match="exact ActionEpochOwner",
    ):
        rewards.materialize_diagnostic_n2_reward_manager_cfg(
            epoch_owner=object()
        )


def test_mutating_one_bundle_does_not_rewrite_code_owned_next_bundle():
    _factory, rewards, _epoch, _owner, first = _diagnostic_lean_reward_bundle()
    first.manager_cfg["common_on_table_outcome"].weight = 1.0
    _factory, rewards, _epoch, _owner, second = _diagnostic_lean_reward_bundle()
    assert second.manager_cfg["common_on_table_outcome"].weight == 20.0


def test_live_physx_subscriber_is_last_offside_step_before_atomic_env_publish():
    tree = ast.parse(FACTORY_MODULE_PATH.read_text(encoding="utf-8"))
    install = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_install_lean_runtime_graph"
    )
    calls = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(install)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in {
            "env_module.FullMdpLeanRuntimeComponents",
            "install_live_physx",
            "env.install_action_ball_full_mdp_lean_runtime_graph",
        }
    }
    assert (
        calls["env_module.FullMdpLeanRuntimeComponents"]
        < calls["install_live_physx"]
        < calls["env.install_action_ball_full_mdp_lean_runtime_graph"]
    )
    construct = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_construct_offside_lean_runtime"
    )
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "install_live_physx"
        for node in ast.walk(construct)
    )


def test_atomic_env_publish_failure_calls_real_physx_unsubscribe_boundary(
    monkeypatch,
):
    factory = _load_factory_subject(
        "_action_ball_full_mdp_runtime_factory_unsubscribe_subject"
    )
    shutdown_calls = []

    class _ScenePort:
        def __init__(self):
            self.installed = False

        def install_action_epoch_live_physx_fact_owner(self, *, stage):
            assert stage is live_stage
            self.installed = True

        def shutdown_action_epoch_live_physx_fact_owner(self):
            if self.installed:
                shutdown_calls.append("shutdown")
                self.installed = False

    scene_port = _ScenePort()
    seed = types.SimpleNamespace(
        motion_owner=object(),
        racket_owner=object(),
        reset_genesis_authority=object(),
        reset_genesis_receipt=object(),
    )
    physical = types.SimpleNamespace(
        scene_port=scene_port,
        physical_owner=object(),
        question=types.SimpleNamespace(
            profile=types.SimpleNamespace(seed=seed)
        ),
    )
    cold = types.SimpleNamespace(
        recurring=types.SimpleNamespace(physical=physical),
        r03_owner=object(),
        r06_owner=object(),
    )
    graph = factory._OffsideLeanRuntimeInputs(
        cold=cold,
        r07_owner=object(),
        r07_plant_fact_adapter=object(),
        epoch_owner=object(),
        device_r05_owner=object(),
        reward_graph=object(),
        reward_manager_cfg={
            name: object() for name in factory.reward_contract.MANAGER_NAMES
        },
        observation_source=object(),
        observation_manager_cfg={"policy": object(), "critic": object()},
        termination_manager_cfg={
            name: object()
            for name in (
                "time_out",
                "base_fell_tilt",
                "base_too_low",
                "joint_qdes_forbidden",
                "robot_hit_table",
            )
        },
        lean_runtime_owner=object(),
    )
    env_module = types.SimpleNamespace(
        FullMdpLeanRuntimeComponents=lambda **kwargs: types.SimpleNamespace(
            **kwargs
        )
    )
    live_stage = object()
    omni_usd = types.SimpleNamespace(
        get_context=lambda: types.SimpleNamespace(
            get_stage=lambda: live_stage
        )
    )
    real_import = factory.importlib.import_module

    def import_module(name):
        if name == "whole_body_tracking.tasks.tracking.full_mdp_env":
            return env_module
        if name == "omni.usd":
            return omni_usd
        return real_import(name)

    monkeypatch.setattr(factory.importlib, "import_module", import_module)
    env = types.SimpleNamespace(
        action_ball_full_mdp_construction_lease=lambda: object(),
        install_action_ball_full_mdp_lean_runtime_graph=(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected env publication failure")
            )
        ),
    )
    with pytest.raises(RuntimeError, match="publication failure"):
        factory._install_lean_runtime_graph(env, graph)
    assert shutdown_calls == ["shutdown"]


def _pinned_isaaclab_source() -> Path:
    candidate = Path(
        "/workspace/IsaacLab-8320e0be/source/isaaclab/isaaclab/envs/"
        "manager_based_rl_env.py"
    )
    if not candidate.is_file():
        pytest.skip("pinned build_2 IsaacLab source is unavailable")
    return candidate


def _pinned_manager_based_env_source() -> Path:
    candidate = Path(
        "/workspace/IsaacLab-8320e0be/source/isaaclab/isaaclab/envs/"
        "manager_based_env.py"
    )
    if not candidate.is_file():
        pytest.skip("pinned build_2 ManagerBasedEnv source is unavailable")
    return candidate


def test_exact_upstream_load_managers_pin_and_order_counterexample():
    source = _pinned_isaaclab_source().read_bytes()
    M._validate_pinned_upstream_source_bytes(source)
    text = source.decode("utf-8")

    tree = ast.parse(text)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRLEnv"
    )
    mutated = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_managers"
    )
    command = M._statement_call_index(mutated.body, "CommandManager")
    parent = M._statement_call_index(mutated.body, "load_managers")
    mutated.body[command], mutated.body[parent] = (
        mutated.body[parent],
        mutated.body[command],
    )
    ast.fix_missing_locations(tree)
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="manager construction order differs",
    ):
        M._assert_pinned_upstream_manager_order(ast.unparse(tree))


def test_exact_parent_manager_loader_pin_and_order():
    source = _pinned_manager_based_env_source().read_bytes()
    M._validate_pinned_manager_based_env_source_bytes(source)

    tree = ast.parse(source.decode("utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedEnv"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_managers"
    )
    action = M._statement_call_index(method.body, "ActionManager")
    observation = M._statement_call_index(method.body, "ObservationManager")
    method.body[action], method.body[observation] = (
        method.body[observation],
        method.body[action],
    )
    ast.fix_missing_locations(tree)
    mutated = ast.unparse(tree).encode("utf-8")
    monkey_file_sha = M.hashlib.sha256(mutated).hexdigest()
    original_file_sha = M.PINNED_MANAGER_BASED_ENV_FILE_SHA256
    try:
        M.PINNED_MANAGER_BASED_ENV_FILE_SHA256 = monkey_file_sha
        with pytest.raises(
            M.FullMdpUpstreamSourceDriftError,
            match="manager order differs",
        ):
            M._validate_pinned_manager_based_env_source_bytes(mutated)
    finally:
        M.PINNED_MANAGER_BASED_ENV_FILE_SHA256 = original_file_sha
