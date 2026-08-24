"""Narrow behavioral tests for the fresh full-MDP env/lean-top callpoints."""

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
        "whole_body_tracking": types.ModuleType("whole_body_tracking"),
        "whole_body_tracking.tasks": types.ModuleType(
            "whole_body_tracking.tasks"
        ),
        "whole_body_tracking.tasks.tracking": types.ModuleType(
            "whole_body_tracking.tasks.tracking"
        ),
        "whole_body_tracking.tasks.tracking.mdp": types.ModuleType(
            "whole_body_tracking.tasks.tracking.mdp"
        ),
        (
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_reward_contract"
        ): types.ModuleType("action_ball_full_mdp_reward_contract"),
    }
    stubs["isaaclab"].__path__ = []
    stubs["isaaclab.envs"].__path__ = []
    stubs["whole_body_tracking"].__path__ = []
    stubs["whole_body_tracking.tasks"].__path__ = []
    stubs["whole_body_tracking.tasks.tracking"].__path__ = []
    stubs["whole_body_tracking.tasks.tracking.mdp"].__path__ = []
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


class _Milestone:
    def __init__(self, trace):
        self.trace = trace

    def add_step_return(self, reward):
        self.trace.append(("milestone_add_step_return", reward))

    def close_episodes(self, selected_mask, episode_tick, reason_bits):
        self.trace.append(
            (
                "milestone_close_episodes",
                selected_mask.clone(),
                episode_tick.clone(),
                reason_bits.clone(),
            )
        )


class _EpochOwner:
    def __init__(self, trace):
        self.milestone = _Milestone(trace)


class _RewardGraph:
    def __init__(self, epoch_owner, trace):
        self.epoch_owner = epoch_owner
        self.trace = trace

    def close_milestone_actual_reward(self, reward):
        self.trace.append(("reward_actual_close", reward))


class _LeanOwner:
    def __init__(self, env, lease, trace, *, epoch_owner):
        self._env = env
        self._lease = lease
        self.epoch_owner = epoch_owner
        self.trace = trace
        self.fail_at = None
        self.reset_ids = []
        self.reset_facts = []
        self.reset_events = []
        self.after_reward_return = None
        self.selected_reset_return = None
        self.action_ball_r10_checkpoint_adapter = None

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

    def before_physics_substep(self, stamp):
        assert self.trace[-1][0] == "action_apply"
        self.trace.append(("prephysics", tuple(stamp)))
        if self.fail_at == "prephysics":
            raise ValueError("prephysics counterexample")
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

    def after_command_compute_before_observation(self, control_step):
        assert type(control_step) is int
        self.trace.append(("after_command", control_step))
        if self.fail_at == "after_command":
            raise ValueError("after-command counterexample")
        return None

    def selected_true_reset(self, event, projection):
        self.trace.append(("top_reset", event, projection))
        self.reset_events.append(event)
        if self.fail_at == "reset":
            raise ValueError("reset counterexample")
        self.reset_ids.append(projection.selected_env_index.clone())
        self.reset_facts.append(
            projection.terminal_reset_facts_i64.clone()
        )
        return self.selected_reset_return


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
    def __init__(self, trace, name, *, count=2):
        self.trace = trace
        self.name = name
        self.time_left = torch.ones(count)
        self.command_counter = torch.zeros(count, dtype=torch.long)
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


def _env(
    *,
    reset_mask=(False, False),
    decimation=2,
    termination_reason_masks=None,
    episode_lengths=None,
):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    reset_mask_tensor = torch.as_tensor(reset_mask, dtype=torch.bool)
    assert reset_mask_tensor.ndim == 1
    num_envs = int(reset_mask_tensor.shape[0])
    env.device = "cpu"
    env.num_envs = num_envs
    env.cfg = types.SimpleNamespace(
        decimation=decimation,
        sim=types.SimpleNamespace(render_interval=1),
        num_rerenders_on_reset=0,
    )
    env.physics_dt = 0.005
    env.step_dt = 0.01
    env.common_step_counter = 0
    env._sim_step_counter = 0
    if episode_lengths is None:
        episode_lengths = (
            (9, 19) if num_envs == 2 else tuple(range(num_envs))
        )
    env.episode_length_buf = torch.as_tensor(
        episode_lengths, dtype=torch.long
    ).clone()
    assert env.episode_length_buf.shape == (num_envs,)
    env.action_manager = _ActionManager(trace)
    env.scene = _Scene(trace)
    env.sim = _Sim(trace)
    env.recorder_manager = _Recorder(trace)
    env.termination_manager = _Termination(
        trace,
        reset_mask_tensor,
        reason_masks=termination_reason_masks,
    )
    env.reset_terminated = env.termination_manager.terminated.clone()
    env.reset_time_outs = torch.logical_and(
        env.termination_manager.time_outs,
        ~env.reset_terminated,
    )
    env.reward_manager = _PlainManager(trace, "reward", count=num_envs)
    env.observation_manager = _PlainManager(
        trace,
        "observation",
        count=num_envs,
        compute_value={"policy": torch.zeros(num_envs, 1)},
    )
    env.curriculum_manager = _PlainManager(
        trace, "curriculum", count=num_envs
    )
    env.event_manager = _Events(trace)
    motion = _CommandTerm(trace, "motion", count=num_envs)
    racket = _CommandTerm(trace, "racket", count=num_envs)
    wind = _CommandTerm(trace, "wind", count=num_envs)
    env.command_manager = _Commands(
        trace,
        {"motion": motion, "racket_target": racket, "wind": wind},
    )
    env.extras = {}
    env._full_mdp_active_dispatch = None
    env._full_mdp_last_after_reward_close_control_step = 0
    env._full_mdp_post_physics_poison = None
    env._action_ball_full_mdp_reset_callpoint_authority = None
    lease = object()
    env._action_ball_full_mdp_runtime_lease = lease
    env._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
    env._action_ball_full_mdp_manager_construction_state = "sealed"
    epoch_owner = _EpochOwner(trace)
    reward_graph = _RewardGraph(epoch_owner, trace)
    owner = _LeanOwner(
        env,
        lease,
        trace,
        epoch_owner=epoch_owner,
    )
    env._action_ball_full_mdp_components = M.FullMdpLeanRuntimeComponents(
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
        lean_runtime_owner=owner,
        observation_source=object(),
    )
    env._action_ball_full_mdp_lean_reward_graph = reward_graph
    genesis = _GenesisAuthority(env.num_envs, torch.device(env.device))
    env._action_ball_full_mdp_reset_genesis_install = M._FullMdpResetGenesisInstall(
        authority=genesis,
        receipt=genesis.receipt,
    )
    env._capture_action_ball_full_mdp_reset_genesis()
    # The focused fixture models a live env after its one canonical genesis
    # reset.  Subsequent reset selections therefore exercise the exact lean
    # selected-reset path rather than the construction-only genesis shortcut.
    env._action_ball_full_mdp_lean_genesis_reset_pending = False
    env._full_mdp_runtime_owner = owner
    env._full_mdp_before_policy_step = owner.before_policy_step
    env._full_mdp_before_physics_substep = owner.before_physics_substep
    env._full_mdp_post_physics_publish = owner.publish_post_physics_substep
    env._full_mdp_after_reward_close = owner.after_reward_close
    env._full_mdp_after_command_compute_before_observation = (
        owner.after_command_compute_before_observation
    )
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
    assert names.count("prephysics") == 2
    assert names.count("postphysics") == 2
    assert names.index("before") < names.index("action_process")
    applications = [
        index for index, name in enumerate(names) if name == "action_apply"
    ]
    prephysics = [
        index for index, name in enumerate(names) if name == "prephysics"
    ]
    writes = [index for index, name in enumerate(names) if name == "scene_write"]
    assert all(
        application < prephysics_call < write
        for application, prephysics_call, write in zip(
            applications, prephysics, writes
        )
    )
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
    assert (
        names.index("reward_compute")
        < names.index("reward_actual_close")
        < names.index("milestone_add_step_return")
        < names.index("after_reward_close")
    )
    assert names.index("after_reward_close") < names.index("command_compute")
    assert names.index("command_compute") < names.index("after_command")
    assert names.count("after_reward_close") == 1
    assert names.count("after_command") == 1
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
    assert names.index("top_reset") < names.index("scene_reset")
    assert names.index("termination_reset") < names.index(
        "milestone_close_episodes"
    )
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


@pytest.mark.parametrize(
    ("plant_reason_masks", "expected_reason_bits"),
    (
        ({"base_fell_tilt": torch.tensor([False, True])}, 2),
        ({"joint_qdes_forbidden": torch.tensor([False, True])}, 8),
        ({"robot_hit_table": torch.tensor([False, True])}, 16),
        (
            {
                "joint_qdes_forbidden": torch.tensor([False, True]),
                "robot_hit_table": torch.tensor([False, True]),
            },
            24,
        ),
    ),
)
def test_horizon_overlap_with_plant_terminal_is_never_rsl_timeout(
    plant_reason_masks, expected_reason_bits
):
    # Row 1 starts at max_episode_length - 1 and reaches the horizon on this
    # transition.  A plant terminal on the same row must own the learning done;
    # row 0 is the healthy peer and must remain unselected with its generation.
    reason_masks = {
        "time_out": torch.tensor([False, True]),
        **plant_reason_masks,
    }
    env, owner, *_ = _env(
        reset_mask=(False, True),
        decimation=1,
        termination_reason_masks=reason_masks,
    )
    env.episode_length_buf.copy_(torch.tensor([9, 19], dtype=torch.int64))
    result = env.step(torch.zeros(2, 4))

    assert result[2].tolist() == [False, True]
    assert result[3].tolist() == [False, False]
    assert env.termination_manager.time_outs.tolist() == [False, True]
    assert torch.equal(
        owner.reset_facts[0],
        torch.tensor(
            [[-1, -1, 0], [1, 20, expected_reason_bits]],
            dtype=torch.int64,
        ),
    )
    assert env._action_ball_full_mdp_reset_generation.tolist() == [7, 8]
    assert env.episode_length_buf.tolist() == [10, 0]


def test_pure_horizon_transition_remains_rsl_timeout_and_reason_owner():
    env, owner, *_ = _env(
        reset_mask=(False, True),
        decimation=1,
        termination_reason_masks={
            "time_out": torch.tensor([False, True]),
        },
    )
    env.episode_length_buf.copy_(torch.tensor([9, 19], dtype=torch.int64))
    result = env.step(torch.zeros(2, 4))

    assert result[2].tolist() == [False, False]
    assert result[3].tolist() == [False, True]
    assert env.termination_manager.time_outs.tolist() == [False, True]
    assert torch.equal(
        owner.reset_facts[0],
        torch.tensor([[-1, -1, 0], [1, 20, 1]], dtype=torch.int64),
    )
    assert env._action_ball_full_mdp_reset_generation.tolist() == [7, 8]
    assert env.episode_length_buf.tolist() == [10, 0]


@pytest.mark.parametrize(
    ("reason_name", "expected_reason_bits", "expected_terminated", "expected_timeout"),
    (
        ("time_out", 1, False, True),
        ("base_fell_tilt", 2, True, False),
        ("robot_hit_table", 16, True, False),
    ),
)
def test_due_transition_294_to_295_journals_exact_terminal_tick(
    reason_name,
    expected_reason_bits,
    expected_terminated,
    expected_timeout,
):
    # Row 1 terminates on the exact transition that advances its episode clock
    # from 294 to the first public cadence tick.  The top env must freeze 295 in
    # ResetTelemetry before native reset clears the row; row 0 is an async peer.
    env, owner, *_ = _env(
        reset_mask=(False, True),
        decimation=1,
        episode_lengths=(41, 294),
        termination_reason_masks={
            reason_name: torch.tensor([False, True]),
        },
    )

    result = env.step(torch.zeros(2, 4))

    assert result[2].tolist() == [False, expected_terminated]
    assert result[3].tolist() == [False, expected_timeout]
    assert torch.equal(
        owner.reset_facts[0],
        torch.tensor(
            [[-1, -1, 0], [1, 295, expected_reason_bits]],
            dtype=torch.int64,
        ),
    )
    assert env.episode_length_buf.tolist() == [42, 0]


def test_mixed_batch_partitions_timeout_plant_and_reset_facts_rowwise():
    # Exercise all three terminal classes in one manager result.  Row 0 is a
    # pure horizon, row 1 reaches the horizon and has a plant terminal on the
    # same transition, and row 2 has only the plant terminal.  The retained
    # manager timeout is raw telemetry; the returned timeout and terminal
    # reason ledger are the disjoint learning/reset ownership partition.
    env, owner, *_ = _env(
        reset_mask=(True, True, True),
        decimation=1,
        episode_lengths=(19, 19, 5),
        termination_reason_masks={
            "time_out": torch.tensor([True, True, False]),
            "base_fell_tilt": torch.tensor([False, True, True]),
        },
    )

    result = env.step(torch.zeros(3, 4))

    assert env.termination_manager.time_outs.tolist() == [True, True, False]
    assert result[2].tolist() == [False, True, True]
    assert result[3].tolist() == [True, False, False]
    assert torch.equal(owner.reset_ids[0], torch.tensor([0, 1, 2]))
    assert torch.equal(
        owner.reset_facts[0],
        torch.tensor(
            [[1, 20, 1], [1, 20, 2], [1, 6, 2]],
            dtype=torch.int64,
        ),
    )
    reason_bits = owner.reset_facts[0][:, 2]
    assert torch.all(reason_bits > 0)
    assert torch.all(torch.bitwise_and(reason_bits, reason_bits - 1) == 0)
    assert env._action_ball_full_mdp_reset_generation.tolist() == [8, 8, 8]
    assert env.episode_length_buf.tolist() == [0, 0, 0]


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


def test_selected_reset_non_none_keeps_generation_and_native_state_unchanged():
    env, owner, trace, motion, racket, wind = _env(
        reset_mask=(False, True), decimation=1
    )
    owner.selected_reset_return = object()
    generation_before = env._action_ball_full_mdp_reset_generation.clone()
    episode_before = env.episode_length_buf.clone()
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="must return None"):
        env.step(torch.zeros(2, 4))
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation, generation_before
    )
    assert torch.equal(env.episode_length_buf, episode_before + 1)
    assert motion.reset_calls == racket.reset_calls == wind.reset_calls == []
    assert "scene_reset" not in _names(trace)


def test_native_failure_after_lean_reset_settlement_is_sticky_poison():
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


def test_lean_reset_projection_rejects_foreign_and_replayed_events():
    env, _owner, *_ = _env(decimation=1)
    env.common_step_counter = 1
    env_ids = torch.tensor([1], dtype=torch.long)
    env._authorize_action_ball_full_mdp_reset_callpoint(
        env_ids, source="step_nonzero"
    )
    event = env._mint_action_ball_full_mdp_selected_reset_event(env_ids)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="stale, foreign"):
        env._project_action_ball_full_mdp_lean_selected_reset_event(
            M.FullMdpSelectedResetEvent()
        )
    projection = env._project_action_ball_full_mdp_lean_selected_reset_event(
        event
    )
    projection.selected_env_index.fill_(0)
    projection.generation_after.fill_(999)
    record = env._action_ball_full_mdp_active_reset_record
    assert torch.equal(record.selected_env_index, torch.tensor([1]))
    assert torch.equal(record.generation_after, torch.tensor([7, 8]))
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="stale, foreign"):
        env._project_action_ball_full_mdp_lean_selected_reset_event(
            event
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
    env._reset_idx(env_ids)
    assert len(owner.reset_ids) == 1
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation[:1], before[:1]
    )
    assert torch.equal(
        env._action_ball_full_mdp_reset_generation[1:], before[1:] + 1
    )


def test_selected_reset_event_never_wraps_int64_generation():
    env, _owner, *_ = _env(decimation=1)
    env.common_step_counter = 1
    maximum = torch.iinfo(torch.int64).max
    env._action_ball_full_mdp_reset_generation.copy_(
        torch.tensor([maximum, 7], dtype=torch.int64)
    )
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
    # This fixture proves only the env writer after-image.  The independent
    # lean top/Device-R05 packed preflight owns the blocking verdict.
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
        "_project_action_ball_full_mdp_lean_selected_reset_event",
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
    env, owner, *_ = _env(decimation=1)
    lease = env.action_ball_full_mdp_runtime_lease
    components = env._action_ball_full_mdp_components
    assert env.full_mdp_runtime_owner is owner
    assert (
        env.action_ball_full_mdp_ppo_drain_owner(lease)
        is components.lean_runtime_owner
    )
    assert (
        env.action_ball_full_mdp_lean_runtime_owner(lease)
        is components.lean_runtime_owner
    )
    assert (
        env.action_ball_full_mdp_lean_reward_graph(lease)
        is components.reward_graph
    )
    assert env.action_ball_r10_checkpoint_adapter is (
        owner.action_ball_r10_checkpoint_adapter
    )
    assert env.action_ball_r10_cold_restore_capsule is None


def _lean_component_roles():
    return {
        "epoch_owner": object(),
        "device_r05_owner": object(),
        "motion_owner": object(),
        "racket_owner": object(),
        "physical_owner": object(),
        "r03_owner": object(),
        "r06_owner": object(),
        "r07_owner": object(),
        "r07_plant_fact_adapter": object(),
        "reward_graph": object(),
        "lean_runtime_owner": object(),
        "observation_source": object(),
    }


@pytest.mark.parametrize("missing_role", ("epoch_owner", "r07_owner"))
def test_lean_component_registry_rejects_missing_role(missing_role):
    owners = _lean_component_roles()
    owners[missing_role] = None
    with pytest.raises(M.FullMdpPostPhysicsOwnerMissingError, match="missing role"):
        M.FullMdpLeanRuntimeComponents(**owners)


@pytest.mark.parametrize(
    ("left_role", "right_role"),
    (
        ("epoch_owner", "device_r05_owner"),
        ("r03_owner", "r07_owner"),
        ("r07_owner", "lean_runtime_owner"),
    ),
)
def test_lean_component_registry_rejects_cross_role_alias(
    left_role, right_role
):
    owners = _lean_component_roles()
    owners[right_role] = owners[left_role]
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="aliases"):
        M.FullMdpLeanRuntimeComponents(**owners)


@pytest.mark.parametrize(
    "getter_name",
    (
        "action_ball_full_mdp_ppo_drain_owner",
        "action_ball_full_mdp_lean_runtime_owner",
        "action_ball_full_mdp_lean_reward_graph",
    ),
)
def test_live_lean_getters_reject_foreign_lease(getter_name):
    env, *_ = _env(decimation=1)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match="foreign lease"):
        getattr(env, getter_name)(object())
