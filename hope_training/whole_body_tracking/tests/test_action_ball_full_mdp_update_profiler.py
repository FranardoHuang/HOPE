from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "source/whole_body_tracking/whole_body_tracking/utils"
SOURCE = UTILS / "action_ball_full_mdp_update_profiler.py"
ADAPTER_SOURCE = UTILS / "action_ball_full_mdp_rsl3_adapter.py"


def _load_source():
    spec = importlib.util.spec_from_file_location(
        "action_ball_full_mdp_update_profiler_under_test", SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _load_source()


class _Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1_000_000
        return self.now


class _ActionManager:
    def process_action(self, _action):
        return None

    def apply_action(self):
        return None


class _Scene:
    def write_data_to_sim(self):
        return None

    def update(self, _dt):
        return None


class _Simulation:
    def step(self, *, render=False):
        return render


class _ComputeManager:
    def compute(self, *args, **kwargs):
        return args, kwargs


class _EventManager:
    def apply(self, *args, **kwargs):
        return args, kwargs


class _Recorder:
    def record_pre_step(self):
        return None

    def record_post_physics_decimation_step(self):
        return None

    def record_post_step(self):
        return None

    def record_pre_reset(self, _env_ids):
        return None

    def record_post_reset(self, _env_ids):
        return None


class _Physical:
    def __init__(self, epoch):
        self._action_epoch_owner = epoch

    def capture_post_physics_facts(self, _stamp=None):
        return None

    def publish_action_epoch_post_physics(self, _stamp=None):
        return self.capture_post_physics_facts(_stamp)


class _R06:
    def __init__(self, epoch):
        self._action_ball_full_mdp_epoch_owner = epoch

    def _close_action_ball_full_mdp_epoch_reward_rows_impl(self):
        return None

    def close_action_ball_full_mdp_epoch_reward_rows(self):
        return self._close_action_ball_full_mdp_epoch_reward_rows_impl()


class _R07Owner:
    def action_epoch_reward_view(self, *args, **kwargs):
        return args, kwargs

    def _publish_action_epoch_motion_readiness(self, *args, **kwargs):
        return args, kwargs

    def stamp_action_epoch_idle_observation(self, *args, **kwargs):
        return args, kwargs


class _Epoch:
    def snapshot_idle_observation_chronology(self, *args, **kwargs):
        return args, kwargs

    def settle_d05_transaction(self, *args, **kwargs):
        return args, kwargs

    def _milestone_after_business_write(self, *args, **kwargs):
        return args, kwargs

    def prepare_drain(self):
        return (0, 1)

    def materialize_drain(self, *, start, end):
        return start, end

    def acknowledge_drain(self, *, start, end):
        return start, end


class _PlantAdapter:
    def read(self):
        return None


class _Motion:
    def project_action_ball_full_mdp_recovery_ready_reference(self, *args, **kwargs):
        return args, kwargs

    def install_action_ball_continuous_r07_ready_projection(self, _projection):
        return None

    def publish_action_ball_full_mdp_post_d05_observation(self):
        return None


class _PhysicalQuestionCore:
    __slots__ = ()

    def issue_horizon_for_test(self, batch):
        return batch

    def project_horizon_for_test(self, receipt):
        return receipt

    def finalize_exact_ticks_for_test(self, receipt, **kwargs):
        return receipt, kwargs


class _QuestionBundle:
    def __init__(self):
        self._physical_owner = _PhysicalQuestionCore()

    def compose_r05_candidate_bank_inside_prepare(self, _context):
        return None


class _R05:
    def __init__(self, epoch, motion):
        self._diagnostic_epoch_owner = epoch
        self._diagnostic_motion_owner = motion
        self._question_bundle = _QuestionBundle()
        self._internal_question_compose = (
            self._question_bundle.compose_r05_candidate_bank_inside_prepare
        )
        self._num_envs = 4
        self._prepared_records = {}

    def advance_action_ball_full_mdp_rows(self):
        return None

    def _prepare_many_impl(self, *args, **kwargs):
        return getattr(self, "_prepared_token", (args, kwargs))

    def _preview_impl(self, *args, **kwargs):
        return args, kwargs

    def _build_row_transaction(self, *args, **kwargs):
        return args, kwargs


class _R07Bundle:
    def __init__(self, motion):
        self.owner = _R07Owner()
        self.owner._diagnostic_n2_bundle = self
        self.plant_fact_adapter = _PlantAdapter()
        self.action_epoch_owner = _Epoch()
        self.motion_owner = motion

    def stamp_epoch_idle_observation_without_keyed_facts(self, *args, **kwargs):
        return args, kwargs

    def publish_epoch_reward_facts(self, *args, **kwargs):
        return args, kwargs

    def motion_ready_projection(self):
        return None


class _RuntimeOwner:
    def __init__(self):
        self._motion = _Motion()
        self._r07_recovery = _R07Bundle(self._motion)
        self._physical_ball = _Physical(self.epoch_owner)
        self._r06_landing_outcome = _R06(self.epoch_owner)
        self._r05_runtime = _R05(self.epoch_owner, self._motion)

    @property
    def epoch_owner(self):
        return self._r07_recovery.action_epoch_owner

    @property
    def component_identities(self):
        return (
            ("r05_runtime", self._r05_runtime),
            ("motion", self._motion),
            ("physical_ball", self._physical_ball),
            ("r06_landing_outcome", self._r06_landing_outcome),
            ("r07_recovery", self._r07_recovery),
        )


class _ExactEnv:
    def __init__(self):
        self._full_mdp_runtime_owner = _RuntimeOwner()
        self.action_manager = _ActionManager()
        self.scene = _Scene()
        self.sim = _Simulation()
        self.termination_manager = _ComputeManager()
        self.reward_manager = _ComputeManager()
        self.command_manager = _ComputeManager()
        self.event_manager = _EventManager()
        self.observation_manager = _ComputeManager()
        self.recorder_manager = _Recorder()

    def _before_policy_step(self):
        return None

    def _assert_step_may_start(self):
        return None

    def _manager_clock_token(self):
        return (1, 1)

    def _assert_manager_clock_token_unchanged(self, _token):
        return None

    def _publish_post_physics_substep(self):
        return None

    def _after_reward_close(self):
        return None

    def _reset_idx(self, env_ids):
        self.recorder_manager.record_pre_reset(env_ids)
        self.recorder_manager.record_post_reset(env_ids)

    def step(self, action):
        self.recorder_manager.record_pre_step()
        self._before_policy_step()
        self._assert_step_may_start()
        manager_clock = self._manager_clock_token()
        self.action_manager.process_action(action)
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self._publish_post_physics_substep()
        self.recorder_manager.record_post_physics_decimation_step()
        self.scene.update(0.005)
        self._assert_manager_clock_token_unchanged(manager_clock)
        self.termination_manager.compute()
        self.reward_manager.compute(0.02)
        self._after_reward_close()
        self._reset_idx((0, 3))
        self.command_manager.compute(0.02)
        # The real FullMDP task has reset events but no interval event.  The
        # after-command owner must therefore close at the following mandatory
        # observation callpoint rather than waiting for EventManager.apply.
        self.observation_manager.compute()
        self.recorder_manager.record_post_step()
        return "done"


def _install_exact_module(monkeypatch):
    module_name = "whole_body_tracking.tasks.tracking.full_mdp_env"
    fake = types.ModuleType(module_name)
    fake.ActionBallFullMdpManagerBasedRLEnv = _ExactEnv
    monkeypatch.setitem(sys.modules, module_name, fake)


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((None, 0), ("", 0), ("0", 0), ("1", 1), ("50", 50)),
)
def test_profile_update_request_is_canonical_and_bounded(raw, expected):
    environ = {} if raw is None else {P.PROFILE_ENV_VAR: raw}
    assert P.parse_full_mdp_profile_updates(environ) == expected


@pytest.mark.parametrize("raw", ("00", "01", "+1", "1.0", "51", "-1", "x"))
def test_profile_update_request_rejects_ambiguous_values(raw):
    with pytest.raises(RuntimeError, match="HOPE_ACTION_BALL"):
        P.parse_full_mdp_profile_updates({P.PROFILE_ENV_VAR: raw})


def test_exact_profiler_counts_real_callpoints_and_auto_restores(monkeypatch):
    _install_exact_module(monkeypatch)
    env = _ExactEnv()
    lines = []
    profiler = P.install_full_mdp_update_profiler(
        env,
        requested_updates=1,
        clock_ns=_Clock(),
        emit_line=lines.append,
    )

    assert env.step(object()) == "done"
    r07 = env._full_mdp_runtime_owner._r07_recovery

    def idle_snapshot():
        r07.action_epoch_owner.snapshot_idle_observation_chronology(owner=r07)
        r07.owner.stamp_action_epoch_idle_observation()
        return "profiled"

    assert env._full_mdp_runtime_owner._full_mdp_profile_runtime_call(
        "r07_idle_stamp", idle_snapshot
    ) == "profiled"
    for segment in (
        "physical_epoch_refresh",
        "r06_postphysics_settle",
        "r06_postphysics_retire",
        "physical_scene_retire_prepare",
        "physical_scene_retire_apply",
        "r06_epoch_facts_publish",
    ):
        assert env._full_mdp_runtime_owner._full_mdp_profile_runtime_call(
            segment, lambda: segment
        ) == segment
    runtime = env._full_mdp_runtime_owner
    assert vars(type(runtime._physical_ball))[
        "publish_action_epoch_post_physics"
    ] is runtime._physical_ball.publish_action_epoch_post_physics.__func__
    assert vars(type(runtime._r06_landing_outcome))[
        "close_action_ball_full_mdp_epoch_reward_rows"
    ] is (
        runtime._r06_landing_outcome
        .close_action_ball_full_mdp_epoch_reward_rows.__func__
    )
    runtime._physical_ball.publish_action_epoch_post_physics()
    runtime._r06_landing_outcome.close_action_ball_full_mdp_epoch_reward_rows()
    runtime.epoch_owner._milestone_after_business_write("owner", "method")
    start, end = runtime.epoch_owner.prepare_drain()
    runtime.epoch_owner.materialize_drain(start=start, end=end)
    runtime.epoch_owner.acknowledge_drain(start=start, end=end)
    payload = profiler.emit_update(
        update=7,
        collection_time_s=22.0,
        learning_time_s=1.25,
        expected_env_step_calls=1,
    )

    assert payload["update"] == 7
    assert payload["schema_version"] == 3
    assert payload["observed_env_step_calls"] == 1
    assert payload["rollout_call_count_exact"] is True
    assert payload["auto_close_after_emit"] is True
    assert payload["speed_evidence_eligible"] is False
    assert set(payload) == {
        "event",
        "schema_version",
        "update",
        "profile_update_ordinal",
        "requested_profile_updates",
        "clock",
        "inclusive_nested_spans",
        "speed_evidence_eligible",
        "collection_ms",
        "learning_ms",
        "expected_env_step_calls",
        "observed_env_step_calls",
        "rollout_call_count_exact",
        "auto_close_after_emit",
        "segments",
    }
    assert payload["segments"]["env_step_total"]["calls"] == 1
    assert payload["segments"]["sim_step"]["calls"] == 1
    assert payload["segments"]["step_may_start_assert"]["calls"] == 1
    settlement_gap = payload["segments"]["after_command_to_observation_gap"]
    assert settlement_gap["calls"] == 1
    assert settlement_gap["inclusive_host_wall_ms"] > 0.0
    assert payload["segments"]["event_apply"]["calls"] == 0
    assert payload["segments"]["d05_total"]["calls"] == 0
    assert payload["segments"]["d05_question_compose"]["calls"] == 0
    assert payload["segments"]["r07_idle_epoch_snapshot"]["calls"] == 1
    assert payload["segments"]["r07_idle_support_read"]["calls"] == 0
    assert payload["segments"]["r07_idle_state_store"]["calls"] == 1
    for segment in (
        "physical_active_flight_capture",
        "physical_epoch_refresh",
        "r06_postphysics_settle",
        "r06_postphysics_retire",
        "physical_scene_retire_prepare",
        "physical_scene_retire_apply",
        "r06_epoch_facts_publish",
        "r06_reward_close_impl",
        "epoch_milestone_write",
        "ppo_drain_prepare",
        "ppo_drain_materialize_d2h_decode",
        "ppo_drain_ack",
    ):
        assert payload["segments"][segment]["calls"] == 1
    selected_reset = payload["segments"]["selected_reset_total"]
    assert selected_reset["calls"] == 1
    assert selected_reset["env_count"] == 2
    assert selected_reset["inclusive_host_wall_ms"] > 0.0
    assert selected_reset["ms_per_call"] > 0.0
    assert payload["segments"]["recorder_callbacks"]["calls"] == 5
    assert profiler.closed
    assert "step" not in env.__dict__
    assert "compute" not in env.reward_manager.__dict__
    assert "snapshot_idle_observation_chronology" not in (
        r07.action_epoch_owner.__dict__
    )
    assert "stamp_action_epoch_idle_observation" not in r07.owner.__dict__
    assert "capture_post_physics_facts" not in runtime._physical_ball.__dict__
    assert (
        "_close_action_ball_full_mdp_epoch_reward_rows_impl"
        not in runtime._r06_landing_outcome.__dict__
    )
    assert "_milestone_after_business_write" not in runtime.epoch_owner.__dict__
    assert "materialize_drain" not in runtime.epoch_owner.__dict__
    assert "_full_mdp_profile_runtime_call" not in (
        env._full_mdp_runtime_owner.__dict__
    )
    assert len(lines) == 1
    observed = json.loads(lines[0].removeprefix(P.PROFILE_JSON_PREFIX))
    assert observed == payload


def test_profiler_preserves_exact_r05_and_motion_method_identity(monkeypatch):
    _install_exact_module(monkeypatch)
    env = _ExactEnv()
    runtime = env._full_mdp_runtime_owner
    r05 = runtime._r05_runtime
    motion = runtime._motion
    profiler = P.install_full_mdp_update_profiler(
        env,
        requested_updates=1,
        clock_ns=_Clock(),
        emit_line=lambda _line: None,
    )

    assert vars(type(r05))["advance_action_ball_full_mdp_rows"] is (
        r05.advance_action_ball_full_mdp_rows.__func__
    )
    assert vars(type(motion))[
        "publish_action_ball_full_mdp_post_d05_observation"
    ] is motion.publish_action_ball_full_mdp_post_d05_observation.__func__
    assert runtime._full_mdp_profile_runtime_call(
        "d05_total", r05.advance_action_ball_full_mdp_rows
    ) is None
    assert profiler._segments["d05_total"]["calls"] == 1
    assert "_prepare_many_impl" in r05.__dict__
    assert "_internal_question_compose" in r05.__dict__

    class _Attempts:
        ndim = 1
        shape = (4,)

        def __getitem__(self, active):
            assert active is active_mask
            return self

        @staticmethod
        def bincount(*, minlength):
            assert minlength == 4

            class _Histogram:
                @staticmethod
                def detach():
                    return _Histogram()

                @staticmethod
                def cpu():
                    return _Histogram()

                @staticmethod
                def tolist():
                    return [0, 2, 1, 1]

            return _Histogram()

    token = object()
    active_mask = types.SimpleNamespace(ndim=1, shape=(4,))
    r05._prepared_token = token
    r05._prepared_records[token] = types.SimpleNamespace(
        rounds_attempted=_Attempts(), rng_advance_mask=active_mask
    )
    assert r05._prepare_many_impl() is token
    assert profiler._segments["d05_round_1_attempted_rows"]["env_count"] == 4
    assert profiler._segments["d05_round_2_attempted_rows"]["env_count"] == 2
    assert profiler._segments["d05_round_3_attempted_rows"]["env_count"] == 1

    profiler.close()
    assert "advance_action_ball_full_mdp_rows" not in r05.__dict__
    assert "publish_action_ball_full_mdp_post_d05_observation" not in (
        motion.__dict__
    )


def test_partial_install_failure_restores_every_bound_method(monkeypatch):
    _install_exact_module(monkeypatch)
    env = _ExactEnv()
    env.observation_manager = object()
    profiler = P.FullMdpUpdateProfiler(requested_updates=1)
    with pytest.raises(RuntimeError, match="cannot bind"):
        profiler.install(env)
    assert profiler.closed
    assert "step" not in env.__dict__
    assert "compute" not in env.reward_manager.__dict__


def test_install_rejects_instance_replaced_leaf_and_restores_runtime_hook(
    monkeypatch,
):
    _install_exact_module(monkeypatch)
    env = _ExactEnv()
    runtime = env._full_mdp_runtime_owner
    runtime._physical_ball.capture_post_physics_facts = lambda _stamp=None: None
    profiler = P.FullMdpUpdateProfiler(requested_updates=1)

    with pytest.raises(RuntimeError, match="Physical/R06 identities"):
        profiler.install(env)

    assert profiler.closed
    assert "_full_mdp_profile_runtime_call" not in runtime.__dict__


def test_runner_wires_bounded_profiler_only_through_exact_opt_in():
    source = ADAPTER_SOURCE.read_text()
    assert "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES" in source
    assert "parse_full_mdp_profile_updates(os.environ)" in source
    assert "install_full_mdp_update_profiler" in source
    assert "expected_env_step_calls=int(self.num_steps_per_env)" in source
    assert "if profiler.closed:" in source
