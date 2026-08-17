"""Narrow structural tests for the fresh full-MDP env/top callpoints.

These strict fakes prove sequencing, exact row selection and fail-stop
behavior only.  They do not make the currently unfrozen production owner or
its dependency inventory runtime-ready.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


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


def _load_subject():
    name = "_action_ball_full_mdp_runtime_callpoints_subject"
    if name in sys.modules:
        return sys.modules[name]

    class FakeManagerBasedRLEnv:
        base_constructions = 0

        def __init__(self, *args, **kwargs):
            type(self).base_constructions += 1
            raise AssertionError("focused tests do not construct a Kit env")

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
    previous = {key: sys.modules.get(key) for key in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module._TEST_FAKE_BASE = FakeManagerBasedRLEnv
        return module
    finally:
        for key, prior in previous.items():
            if prior is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prior


M = _load_subject()


class _Owner:
    def __init__(self, env, lease, trace):
        self._env = env
        self._lease = lease
        self._dag = "a" * 64
        self.trace = trace
        self.fail_at = None
        self.reset_ids = []
        self.reset_facts = []
        self.reset_events = []
        self.reset_receipts = {}
        self.fail_validator = False
        self.after_reward_return = None
        self.action_ball_r10_checkpoint_adapter = object()

    @property
    def full_mdp_runtime_dependency_dag_sha256(self):
        return self._dag

    @property
    def full_mdp_runtime_env(self):
        return self._env

    @property
    def full_mdp_runtime_lease(self):
        return self._lease

    def before_policy_step(self, control_step, action):
        self.trace.append(("before", control_step, action))
        if self.fail_at == "before":
            raise ValueError("before counterexample")
        return None

    def publish_post_physics_substep(self, stamp):
        assert self.trace[-1][0] == "scene_update"
        self.trace.append(("postphysics", stamp.exact_tuple()))
        if self.fail_at == "postphysics":
            raise ValueError("postphysics counterexample")
        return None

    def after_reward_close(self, control_step):
        assert type(control_step) is int
        self.trace.append(("after_reward_close", control_step))
        if self.fail_at == "after_reward":
            raise ValueError("after-Reward counterexample")
        return self.after_reward_return

    def selected_true_reset(self, event):
        self.trace.append(("top_reset", event))
        self.reset_events.append(event)
        if self.fail_at == "reset":
            raise ValueError("reset counterexample")
        projection = self._env.project_action_ball_full_mdp_selected_reset_event(
            event,
            expected_top=self,
            device=self._env.device,
            num_envs=self._env.num_envs,
            live_reset_ledger_identity=self.live_reset_ledger_identity,
            live_reset_generation=self.live_reset_generation.clone(),
        )
        self.reset_ids.append(projection.selected_env_index.clone())
        self.reset_facts.append(
            projection.terminal_reset_facts_i64.clone()
        )
        self.live_reset_generation.copy_(projection.generation_after)
        receipt = object()
        self.reset_receipts[receipt] = event
        return receipt

    def require_owned_selected_true_reset_receipt(self, receipt, expected_event):
        self.trace.append(("top_reset_validate", receipt, expected_event))
        if self.fail_validator or self.reset_receipts.get(receipt) is not expected_event:
            raise ValueError("reset receipt counterexample")
        del self.reset_receipts[receipt]
        return receipt


class _GenesisAuthority:
    def __init__(self, num_envs, device):
        self.receipt = object()
        self.world_reset_identity = object()
        self.generations = torch.full(
            (num_envs,), 7, dtype=torch.int64, device=device
        )

    def require_owned_full_mdp_reset_genesis(self, receipt, *, device, num_envs):
        assert receipt is self.receipt
        assert device == self.generations.device
        assert num_envs == self.generations.shape[0]
        return M.FullMdpResetGenesisProjection(
            world_reset_identity=self.world_reset_identity,
            reset_generations=self.generations.clone(),
        )


class _ActionManager:
    def __init__(self, trace):
        self.trace = trace

    def process_action(self, action):
        self.trace.append(("action_process", action))

    def apply_action(self):
        self.trace.append(("action_apply",))

    def reset(self, env_ids):
        self.trace.append(("action_reset", env_ids.clone()))
        return {}


class _Scene:
    def __init__(self, trace):
        self.trace = trace

    def write_data_to_sim(self):
        self.trace.append(("scene_write",))

    def update(self, *, dt):
        self.trace.append(("scene_update", dt))

    def reset(self, env_ids):
        self.trace.append(("scene_reset", env_ids.clone()))


class _Sim:
    def __init__(self, trace):
        self.trace = trace

    def has_gui(self):
        return False

    def has_rtx_sensors(self):
        return False

    def step(self, *, render):
        assert render is False
        self.trace.append(("sim_step",))

    def render(self):
        raise AssertionError("focused fake has no renderer")

    def forward(self):
        self.trace.append(("sim_forward",))


class _Recorder:
    active_terms = ()

    def __init__(self, trace):
        self.trace = trace

    def record_pre_step(self):
        self.trace.append(("record_pre_step",))

    def record_post_physics_decimation_step(self):
        self.trace.append(("record_post_physics",))

    def record_pre_reset(self, env_ids):
        self.trace.append(("record_pre_reset", env_ids.clone()))

    def record_post_reset(self, env_ids):
        self.trace.append(("record_post_reset", env_ids.clone()))

    def reset(self, env_ids):
        self.trace.append(("recorder_reset", env_ids.clone()))
        return {}


class _Termination:
    _NAMES = (
        "time_out",
        "base_fell_tilt",
        "base_too_low",
        "joint_qdes_forbidden",
        "robot_hit_table",
    )

    def __init__(self, trace, mask, reason_masks=None):
        self.trace = trace
        if reason_masks is None:
            reason_masks = {"base_fell_tilt": mask}
        by_name = {
            name: torch.as_tensor(
                reason_masks.get(name, torch.zeros_like(mask)),
                dtype=torch.bool,
            ).clone()
            for name in self._NAMES
        }
        self._term_dones = torch.stack(
            tuple(by_name[name] for name in self._NAMES), dim=1
        )
        self.time_outs = by_name["time_out"].clone()
        self.terminated = torch.zeros_like(mask)
        for name in self._NAMES[1:]:
            self.terminated |= by_name[name]

    @property
    def active_terms(self):
        return list(self._NAMES)

    def compute(self):
        self.trace.append(("termination",))
        return self.terminated | self.time_outs

    def get_term(self, name):
        return self._term_dones[:, self._NAMES.index(name)]

    def reset(self, env_ids):
        self.trace.append(("termination_reset", env_ids.clone()))
        return {}


class _PlainManager:
    def __init__(self, trace, name, *, count=2, compute_value=None):
        self.trace = trace
        self.name = name
        self.count = count
        self.compute_value = compute_value

    def compute(self, **kwargs):
        self.trace.append((self.name + "_compute", kwargs))
        if self.compute_value is not None:
            return self.compute_value
        return torch.zeros(self.count)

    def reset(self, env_ids):
        self.trace.append((self.name + "_reset", env_ids.clone()))
        return {}


class _CommandTerm:
    def __init__(self, trace, name):
        self.trace = trace
        self.name = name
        self.time_left = torch.ones(2)
        self.command_counter = torch.zeros(2, dtype=torch.long)
        self.reset_calls = []

    def reset(self, *, env_ids):
        self.reset_calls.append(env_ids.clone())
        self.trace.append((self.name + "_legacy_reset", env_ids.clone()))
        return {"count": 1}


class _Commands:
    def __init__(self, trace, terms):
        self.trace = trace
        self.terms = terms
        self.reset_calls = 0

    @property
    def active_terms(self):
        return list(self.terms)

    def get_term(self, name):
        return self.terms[name]

    def compute(self, *, dt):
        self.trace.append(("command_compute", dt))

    def reset(self, env_ids):
        self.reset_calls += 1
        raise AssertionError("legacy whole CommandManager.reset is forbidden")


class _Events:
    available_modes = ()

    def __init__(self, trace):
        self.trace = trace

    def reset(self, env_ids):
        self.trace.append(("event_reset", env_ids.clone()))
        return {}

    def apply(self, **kwargs):
        raise AssertionError("no event mode is configured")


def _binding(owner):
    owner_type = type(owner)
    members = M._loaded_class_executable_members(
        owner_type,
        module_object=sys.modules[owner_type.__module__],
        strict_plain_functions=True,
    )
    return M._ConcreteOwnerExecutableBinding(
        module_name=owner_type.__module__,
        qualname=owner_type.__qualname__,
        module_object=sys.modules[owner_type.__module__],
        owner_type=owner_type,
        direct_executable_sha256=M._live_owner_direct_executable_sha256(
            owner_type
        ),
        executable_members=members,
        publish_function=vars(owner_type)["publish_post_physics_substep"],
        before_policy_step_function=vars(owner_type)["before_policy_step"],
        after_reward_close_function=vars(owner_type)["after_reward_close"],
        selected_true_reset_function=vars(owner_type)["selected_true_reset"],
        selected_true_reset_receipt_validator_function=vars(owner_type)[
            "require_owned_selected_true_reset_receipt"
        ],
    )


def _env(
    *, reset_mask=(False, False), decimation=2, termination_reason_masks=None
):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    env.device = "cpu"
    env.num_envs = 2
    env.cfg = types.SimpleNamespace(
        decimation=decimation,
        sim=types.SimpleNamespace(render_interval=1),
        num_rerenders_on_reset=0,
    )
    env.physics_dt = 0.005
    env.step_dt = 0.01
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.episode_length_buf = torch.tensor([9, 19], dtype=torch.long)
    env.action_manager = _ActionManager(trace)
    env.scene = _Scene(trace)
    env.sim = _Sim(trace)
    env.recorder_manager = _Recorder(trace)
    env.termination_manager = _Termination(
        trace,
        torch.tensor(reset_mask, dtype=torch.bool),
        reason_masks=termination_reason_masks,
    )
    env.reward_manager = _PlainManager(trace, "reward")
    env.observation_manager = _PlainManager(
        trace,
        "observation",
        compute_value={"policy": torch.zeros(2, 1)},
    )
    env.curriculum_manager = _PlainManager(trace, "curriculum")
    env.event_manager = _Events(trace)
    motion = _CommandTerm(trace, "motion")
    racket = _CommandTerm(trace, "racket")
    wind = _CommandTerm(trace, "wind")
    env.command_manager = _Commands(
        trace,
        {"motion": motion, "racket_target": racket, "wind": wind},
    )
    env.extras = {}
    env._full_mdp_active_dispatch = None
    env._full_mdp_last_after_reward_close_control_step = 0
    env._full_mdp_post_physics_poison = None
    env._action_ball_full_mdp_reset_callpoint_authority = None
    env._full_mdp_runtime_expected_dependency_dag = "a" * 64
    lease = object()
    env._action_ball_full_mdp_runtime_lease = lease
    env._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    env._action_ball_full_mdp_num_envs = 2
    env._action_ball_full_mdp_device = "cpu"
    env._action_ball_full_mdp_components = M.FullMdpRuntimeComponents(
        r05_owner=object(),
        device_r05_owner=object(),
        motion_owner=motion,
        racket_owner=racket,
        r06_owner=object(),
        physical_owner=object(),
        r03_owner=object(),
        r07_owner=object(),
        ppo_drain_owner=object(),
    )
    genesis = _GenesisAuthority(env.num_envs, torch.device(env.device))
    env._action_ball_full_mdp_reset_genesis_install = M._FullMdpResetGenesisInstall(
        authority=genesis,
        receipt=genesis.receipt,
    )
    env._capture_action_ball_full_mdp_reset_genesis()
    owner = _Owner(env, lease, trace)
    owner.live_reset_ledger_identity = object()
    owner.live_reset_generation = genesis.generations.clone()
    env.bind_action_ball_full_mdp_selected_reset_authority(
        lease,
        expected_top=owner,
        result_validator=owner.require_owned_selected_true_reset_receipt,
        live_reset_ledger_identity=owner.live_reset_ledger_identity,
        world_reset_identity=genesis.world_reset_identity,
    )
    executable_binding = _binding(owner)
    env._full_mdp_runtime_owner = owner
    env._full_mdp_runtime_owner_type = type(owner)
    env._full_mdp_runtime_executable_binding = executable_binding
    env._full_mdp_before_policy_step = owner.before_policy_step
    env._full_mdp_post_physics_publish = owner.publish_post_physics_substep
    env._full_mdp_after_reward_close = owner.after_reward_close
    env._full_mdp_selected_true_reset = owner.selected_true_reset
    return env, owner, trace, motion, racket, wind


def _names(trace):
    return [row[0] for row in trace]


@pytest.mark.parametrize("bad_value", [0, torch.iinfo(torch.int64).max])
def test_reset_genesis_rejects_noncontinuing_foreign_authority(bad_value):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    env.num_envs = 2
    env.device = "cpu"
    authority = _GenesisAuthority(env.num_envs, torch.device(env.device))
    authority.generations.fill_(bad_value)
    env._action_ball_full_mdp_reset_genesis_install = M._FullMdpResetGenesisInstall(
        authority=authority,
        receipt=authority.receipt,
    )
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="continuation"):
        env._capture_action_ball_full_mdp_reset_genesis()
    assert not hasattr(env, "_action_ball_full_mdp_reset_generation")


def test_top_callpoint_sequence_and_no_legacy_duplicate():
    env, owner, trace, motion, racket, wind = _env(decimation=2)
    action = torch.zeros(2, 4)
    result = env.step(action)

    names = _names(trace)
    assert names.count("before") == 1
    assert names.count("postphysics") == 2
    assert names.index("before") < names.index("action_process")
    updates = [index for index, name in enumerate(names) if name == "scene_update"]
    simulations = [index for index, name in enumerate(names) if name == "sim_step"]
    recordings = [
        index
        for index, name in enumerate(names)
        if name == "record_post_physics"
    ]
    publications = [
        index for index, name in enumerate(names) if name == "postphysics"
    ]
    assert all(
        simulation < recording < update
        for simulation, recording, update in zip(
            simulations, recordings, updates
        )
    )
    assert all(publication == update + 1 for update, publication in zip(updates, publications))
    assert publications[-1] < names.index("termination") < names.index("reward_compute")
    assert names.index("reward_compute") + 1 == names.index("after_reward_close")
    assert names.index("after_reward_close") < names.index("command_compute")
    assert names.count("after_reward_close") == 1
    assert names.count("top_reset") == 0
    final_observation = next(
        row for row in reversed(trace) if row[0] == "observation_compute"
    )
    assert final_observation[1] == {"update_history": True}
    assert motion.reset_calls == racket.reset_calls == wind.reset_calls == []
    assert len(result) == 5
    assert owner.trace[0][2] is trace[names.index("action_process")][1]


def test_n2_selected_reset_changes_only_selected_native_row():
    env, owner, trace, motion, racket, wind = _env(
        reset_mask=(False, True), decimation=1
    )
    env.step(torch.zeros(2, 4))

    assert len(owner.reset_ids) == 1
    assert torch.equal(owner.reset_ids[0], torch.tensor([1]))
    assert type(owner.reset_events[0]) is M.FullMdpSelectedResetEvent
    assert not hasattr(owner.reset_events[0], "env_ids")
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation,
        torch.tensor([7, 8]),
    )
    assert torch.equal(owner.live_reset_generation, torch.tensor([7, 8]))
    assert torch.equal(
        owner.reset_facts[0],
        torch.tensor([[-1, -1, 0], [1, 20, 2]], dtype=torch.int64),
    )
    assert env.episode_length_buf.tolist() == [10, 0]
    assert motion.reset_calls == []
    assert racket.reset_calls == []
    assert len(wind.reset_calls) == 1
    assert torch.equal(wind.reset_calls[0], torch.tensor([1]))
    assert env.command_manager.reset_calls == 0
    names = _names(trace)
    assert names.index("top_reset") < names.index("curriculum_compute")
    assert names.index("top_reset_validate") < names.index("curriculum_compute")
    assert names.index("top_reset") < names.index("scene_reset")
    assert names.index("wind_legacy_reset") < names.index("event_reset")
    assert names.count("scene_write") == 1
    assert names.count("sim_forward") == 0
    final_observation = next(
        row for row in reversed(trace) if row[0] == "observation_compute"
    )
    assert final_observation[1] == {"update_history": True}


def test_selected_reset_uses_exact_num_rerenders_without_write_or_forward():
    env, _owner, trace, *_ = _env(
        reset_mask=(False, True), decimation=1
    )
    env.cfg.num_rerenders_on_reset = 2
    env.sim.has_rtx_sensors = lambda: True
    env.sim.render = lambda: trace.append(("render",))

    env.step(torch.zeros(2, 4))

    names = _names(trace)
    assert names.count("render") == 3
    assert names.count("scene_write") == 1
    assert names.count("sim_forward") == 0
    scene_reset = names.index("scene_reset")
    post_reset = names.index("record_post_reset")
    reset_renders = [
        index
        for index, name in enumerate(names)
        if name == "render" and scene_reset < index < post_reset
    ]
    assert len(reset_renders) == 2


def test_selected_reset_facts_preserve_overlapping_fixed_reasons_and_clocks():
    env, owner, *_ = _env(
        reset_mask=(False, True),
        decimation=1,
        termination_reason_masks={
            "joint_qdes_forbidden": torch.tensor([False, True]),
            "robot_hit_table": torch.tensor([False, True]),
        },
    )
    env.step(torch.zeros(2, 4))
    assert torch.equal(
        owner.reset_facts[0],
        torch.tensor([[-1, -1, 0], [1, 20, 24]], dtype=torch.int64),
    )


def test_missing_owner_first_operation_fails_before_action_or_sim():
    env, _owner, trace, *_ = _env(decimation=1)
    del env._full_mdp_runtime_owner
    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError, match="first operation"
    ):
        env.step(torch.zeros(2, 4))
    assert trace == []
    assert env._full_mdp_post_physics_poison is not None


@pytest.mark.parametrize(
    "failure", ("before", "postphysics", "after_reward", "reset")
)
def test_top_exception_poisons_and_has_zero_duplicate(failure):
    reset_mask = (False, True) if failure == "reset" else (False, False)
    env, owner, trace, motion, racket, wind = _env(
        reset_mask=reset_mask, decimation=1
    )
    owner.fail_at = failure
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="failed"):
        env.step(torch.zeros(2, 4))
    count_before = _names(trace).count(
        {
            "before": "before",
            "postphysics": "postphysics",
            "after_reward": "after_reward_close",
            "reset": "top_reset",
        }[
            failure
        ]
    )
    assert count_before == 1
    before_retry = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"):
        env.step(torch.zeros(2, 4))
    assert trace == before_retry
    assert motion.reset_calls == racket.reset_calls == wind.reset_calls == []


def test_after_reward_non_none_return_is_sticky_and_blocks_reset():
    env, owner, trace, *_ = _env(decimation=1)
    owner.after_reward_return = object()
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="must return None"
    ):
        env.step(torch.zeros(2, 4))
    assert _names(trace).count("after_reward_close") == 1
    before_reset = list(trace)
    with pytest.raises(
        M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"
    ):
        env.reset()
    assert trace == before_reset


@pytest.mark.parametrize("mutation", ("missing", "foreign"))
def test_after_reward_bound_callable_drift_rejected_before_step(mutation):
    env, _owner, trace, *_ = _env(decimation=1)
    if mutation == "missing":
        del env._full_mdp_after_reward_close
    else:
        foreign_env, foreign_owner, *_ = _env(decimation=1)
        assert foreign_env is not env
        env._full_mdp_after_reward_close = foreign_owner.after_reward_close
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="binding changed"
    ):
        env.step(torch.zeros(2, 4))
    assert trace == []
    assert env._full_mdp_post_physics_poison is not None


def test_after_reward_replay_is_rejected_without_second_owner_call():
    env, _owner, trace, *_ = _env(decimation=1)
    env._after_reward_close(control_step=1)
    assert _names(trace).count("after_reward_close") == 1
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="duplicated, or replayed"
    ):
        env._after_reward_close(control_step=1)
    assert _names(trace).count("after_reward_close") == 1
    assert env._full_mdp_post_physics_poison is not None


def test_production_descriptor_install_rejects_missing_after_reward_callpoint():
    class MissingAfterRewardOwner:
        @property
        def full_mdp_runtime_dependency_dag_sha256(self):
            return "a" * 64

        @property
        def full_mdp_runtime_env(self):
            return None

        @property
        def full_mdp_runtime_lease(self):
            return None

        def before_policy_step(self, control_step, action):
            return None

        def publish_post_physics_substep(self, stamp):
            return None

        def selected_true_reset(self, event):
            return None

        def require_owned_selected_true_reset_receipt(
            self, receipt, expected_event
        ):
            return receipt

    owner = MissingAfterRewardOwner()
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="after_reward_close"
    ):
        M._require_owner_api_descriptors(owner, type(owner))


def test_reset_receipt_failure_keeps_env_generation_and_native_state_unchanged():
    env, owner, trace, motion, racket, wind = _env(
        reset_mask=(False, True), decimation=1
    )
    owner.fail_validator = True
    generation_before = env._action_ball_full_mdp_reset_generation.clone()
    episode_before = env.episode_length_buf.clone()
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="failed"):
        env.step(torch.zeros(2, 4))
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation, generation_before
    )
    assert torch.equal(env.episode_length_buf, episode_before + 1)
    assert motion.reset_calls == racket.reset_calls == wind.reset_calls == []
    assert "scene_reset" not in _names(trace)


def test_native_failure_after_global_receipt_is_sticky_poison():
    env, owner, trace, *_ = _env(reset_mask=(False, True), decimation=1)
    generation_before = env._action_ball_full_mdp_reset_generation.clone()

    def fail_scene_reset(_env_ids):
        trace.append(("scene_reset_failure",))
        raise ValueError("native reset counterexample")

    env.scene.reset = fail_scene_reset
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="failed"):
        env.step(torch.zeros(2, 4))
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation,
        generation_before + torch.tensor([0, 1]),
    )
    before_retry = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"):
        env.step(torch.zeros(2, 4))
    assert trace == before_retry


def test_reset_event_replay_foreign_top_and_live_generation_are_rejected():
    env, owner, *_ = _env(decimation=1)
    env.common_step_counter = 1
    env_ids = torch.tensor([1], dtype=torch.long)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    event = env._mint_action_ball_full_mdp_selected_reset_event(env_ids)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="stale, foreign"):
        env.project_action_ball_full_mdp_selected_reset_event(
            M.FullMdpSelectedResetEvent(),
            expected_top=owner,
            device=env.device,
            num_envs=env.num_envs,
            live_reset_ledger_identity=owner.live_reset_ledger_identity,
            live_reset_generation=owner.live_reset_generation,
        )
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="stale, foreign"):
        env.project_action_ball_full_mdp_selected_reset_event(
            event,
            expected_top=object(),
            device=env.device,
            num_envs=env.num_envs,
            live_reset_ledger_identity=owner.live_reset_ledger_identity,
            live_reset_generation=owner.live_reset_generation,
        )
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="stale, foreign"):
        env.project_action_ball_full_mdp_selected_reset_event(
            event,
            expected_top=owner,
            device=env.device,
            num_envs=env.num_envs,
            live_reset_ledger_identity=object(),
            live_reset_generation=owner.live_reset_generation.clone(),
        )
    projection = env.project_action_ball_full_mdp_selected_reset_event(
        event,
        expected_top=owner,
        device=env.device,
        num_envs=env.num_envs,
        live_reset_ledger_identity=owner.live_reset_ledger_identity,
        live_reset_generation=owner.live_reset_generation,
    )
    projection.selected_env_index.fill_(0)
    projection.generation_after.fill_(999)
    record = env._action_ball_full_mdp_active_reset_record
    assert torch.equal(record.selected_env_index, torch.tensor([1]))
    assert torch.equal(record.generation_after, torch.tensor([7, 8]))
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="stale, foreign"):
        env.project_action_ball_full_mdp_selected_reset_event(
            event,
            expected_top=owner,
            device=env.device,
            num_envs=env.num_envs,
            live_reset_ledger_identity=owner.live_reset_ledger_identity,
            live_reset_generation=owner.live_reset_generation,
        )


def test_genesis_and_top_reset_authority_are_one_shot_and_identity_bound():
    env, owner, *_ = _env(decimation=1)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="cannot be rebound"):
        env.bind_action_ball_full_mdp_selected_reset_authority(
            env.action_ball_full_mdp_runtime_lease,
            expected_top=owner,
            result_validator=owner.require_owned_selected_true_reset_receipt,
            live_reset_ledger_identity=owner.live_reset_ledger_identity,
            world_reset_identity=env._action_ball_full_mdp_world_reset_identity,
        )
    late = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    late._action_ball_full_mdp_runtime_lease = lease
    late._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
    late._action_ball_full_mdp_manager_construction_state = "sealed"
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="builder phase"):
        late.install_action_ball_full_mdp_reset_genesis(
            lease, object(), object()
        )


def test_selected_reset_rejects_any_direct_unowned_ids():
    env, *_ = _env(decimation=1)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="callpoint authority"):
        env._mint_action_ball_full_mdp_selected_reset_event(
            torch.tensor([], dtype=torch.long)
        )
    env, *_ = _env(decimation=1)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="callpoint authority"):
        env._mint_action_ball_full_mdp_selected_reset_event(
            torch.tensor([1, 1], dtype=torch.long)
        )


def test_selected_reset_callpoint_authority_is_exact_identity_and_single_use():
    env, *_ = _env(decimation=1)
    env.common_step_counter = 1
    env_ids = torch.tensor([1], dtype=torch.long)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="callpoint authority"):
        env._mint_action_ball_full_mdp_selected_reset_event(env_ids.clone())
    event = env._mint_action_ball_full_mdp_selected_reset_event(env_ids)
    assert type(event) is M.FullMdpSelectedResetEvent
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="callpoint authority"):
        env._mint_action_ball_full_mdp_selected_reset_event(env_ids)


def test_manual_partial_reset_is_rejected_before_base_reset():
    env, *_ = _env(decimation=1)
    with pytest.raises(M.FullMdpUnsupportedRuntimeError, match="partial reset"):
        env.reset(env_ids=torch.tensor([1], dtype=torch.long))
    assert env._action_ball_full_mdp_reset_callpoint_authority is None


def test_selected_reset_preserves_unselected_generation_bytes_exactly():
    env, owner, *_ = _env(decimation=1)
    env.common_step_counter = 1
    before = env._action_ball_full_mdp_reset_generation.clone()
    env_ids = torch.tensor([1], dtype=torch.long)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    event = env._mint_action_ball_full_mdp_selected_reset_event(env_ids)
    receipt = owner.selected_true_reset(event)
    env._consume_action_ball_full_mdp_selected_reset_result(event, receipt)
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation[:1], before[:1]
    )
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation[1:], before[1:] + 1
    )


def test_selected_reset_event_never_wraps_int64_generation():
    env, owner, *_ = _env(decimation=1)
    env.common_step_counter = 1
    maximum = torch.iinfo(torch.int64).max
    env._action_ball_full_mdp_reset_generation.copy_(
        torch.tensor([maximum, 7], dtype=torch.int64)
    )
    owner.live_reset_generation.copy_(env._action_ball_full_mdp_reset_generation)
    env_ids = torch.tensor([0], dtype=torch.long)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    event = env._mint_action_ball_full_mdp_selected_reset_event(env_ids)
    record = env._action_ball_full_mdp_active_reset_record
    assert torch.equal(
        record.generation_after,
        torch.tensor([maximum, 7], dtype=torch.int64),
    )
    assert record.generation_overflow_fault.tolist() == [True, False]
    # This fixture deliberately proves only the env writer after-image.  The
    # independent top/Device-R05 packed preflight owns the blocking verdict;
    # a fake owner receipt must not be called safety evidence here.
    assert record.event is event
    assert not record.projected
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation,
        torch.tensor([maximum, 7], dtype=torch.int64),
    )


def test_reset_seam_source_has_no_host_observation_api():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpManagerBasedRLEnv"
    )
    reset_names = {
        "_mint_action_ball_full_mdp_selected_reset_event",
        "project_action_ball_full_mdp_selected_reset_event",
        "_consume_action_ball_full_mdp_selected_reset_result",
        "_reset_idx",
    }
    forbidden = {"item", "cpu", "numpy", "tolist", "synchronize"}
    for method in owner_class.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        if method.name not in reset_names:
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden


def test_dormant_cold_restore_holds_before_base_or_factory_side_effect():
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    calls = []
    before = M._TEST_FAKE_BASE.base_constructions

    def factory(*args):
        calls.append(args)
        raise AssertionError("dormant mode cannot invoke factory")

    with pytest.raises(M.FullMdpUnsupportedRuntimeError, match="remains HOLD"):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg=None,
            full_mdp_runtime_owner_factory=factory,
            full_mdp_runtime_owner_expected_dependency_dag_sha256="a" * 64,
            full_mdp_cold_restore_dormant=True,
        )
    assert calls == []
    assert M._TEST_FAKE_BASE.base_constructions == before
    assert not any(
        hasattr(env, name)
        for name in (
            "sim",
            "scene",
            "observation_manager",
            "action_manager",
            "_full_mdp_runtime_owner",
        )
    )


def test_component_getters_are_exact_lease_protected_identities():
    env, owner, _trace, motion, racket, _wind = _env(decimation=1)
    lease = env.action_ball_full_mdp_runtime_lease
    components = env._action_ball_full_mdp_components
    assert env.full_mdp_runtime_owner is owner
    assert env.action_ball_full_mdp_num_envs(lease) == 2
    assert env.action_ball_full_mdp_device(lease) == "cpu"
    assert env.action_ball_full_mdp_r05_owner(lease) is components.r05_owner
    assert (
        env.action_ball_full_mdp_device_r05_owner(lease)
        is components.device_r05_owner
    )
    assert env.action_ball_full_mdp_motion_owner(lease) is motion
    assert env.action_ball_full_mdp_racket_owner(lease) is racket
    assert env.action_ball_full_mdp_r06_owner(lease) is components.r06_owner
    assert (
        env.action_ball_full_mdp_physical_owner(lease)
        is components.physical_owner
    )
    assert env.action_ball_full_mdp_r03_owner(lease) is components.r03_owner
    assert env.action_ball_full_mdp_r07_owner(lease) is components.r07_owner
    assert (
        env.action_ball_full_mdp_ppo_drain_owner(lease)
        is components.ppo_drain_owner
    )
    assert env.action_ball_r10_checkpoint_adapter is (
        owner.action_ball_r10_checkpoint_adapter
    )
    assert env.action_ball_r10_cold_restore_capsule is None
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="foreign lease"):
        env.action_ball_full_mdp_motion_owner(object())


@pytest.mark.parametrize("missing_role", ("r03_owner", "r07_owner"))
def test_component_registry_rejects_missing_global_drain_owner(missing_role):
    owners = {
        "r05_owner": object(),
        "device_r05_owner": object(),
        "motion_owner": object(),
        "racket_owner": object(),
        "r06_owner": object(),
        "physical_owner": object(),
        "r03_owner": object(),
        "r07_owner": object(),
        "ppo_drain_owner": object(),
    }
    owners[missing_role] = None
    with pytest.raises(M.FullMdpPostPhysicsOwnerMissingError, match="missing owner"):
        M.FullMdpRuntimeComponents(**owners)


@pytest.mark.parametrize(
    ("left_role", "right_role"),
    (
        ("r05_owner", "device_r05_owner"),
        ("r03_owner", "r07_owner"),
        ("r07_owner", "ppo_drain_owner"),
    ),
)
def test_component_registry_rejects_cross_role_alias(left_role, right_role):
    owners = {
        "r05_owner": object(),
        "device_r05_owner": object(),
        "motion_owner": object(),
        "racket_owner": object(),
        "r06_owner": object(),
        "physical_owner": object(),
        "r03_owner": object(),
        "r07_owner": object(),
        "ppo_drain_owner": object(),
    }
    owners[right_role] = owners[left_role]
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="aliases"):
        M.FullMdpRuntimeComponents(**owners)


@pytest.mark.parametrize(
    "getter_name",
    (
        "action_ball_full_mdp_r03_owner",
        "action_ball_full_mdp_r07_owner",
    ),
)
def test_global_drain_owner_getters_reject_foreign_lease(getter_name):
    env, *_ = _env(decimation=1)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="foreign lease"):
        getattr(env, getter_name)(object())


def test_structural_fixture_does_not_claim_runtime_go():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Construction intentionally\nremains HOLD" in source
    assert M.PINNED_FULL_MDP_OWNER_MODULE is None
    assert M.PINNED_FULL_MDP_OWNER_DEPENDENCY_DAG_SHA256 is None
