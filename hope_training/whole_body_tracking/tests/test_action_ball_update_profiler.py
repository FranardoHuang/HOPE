from __future__ import annotations

import builtins
import json
import types

import pytest

from whole_body_tracking.utils import action_ball_update_profiler as P
import test_exact_resume_state as ER


class _Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1_000_000
        return self.now


class _Provider:
    def provide_many(self, requests):
        return tuple(requests)


class _Solver:
    def solve_many(self, requests):
        return tuple(requests)


class _Broker:
    def __init__(self, provider):
        self.provider = provider
        self.diagnostic_fast_path = True

    def reserve_many_true_reset(self, requests, **_kwargs):
        return self.provider.provide_many(requests)

    def consume_many_true_reset(self, requests, **_kwargs):
        return tuple(requests)


class _Pool:
    def __init__(self, solver):
        self.solver = solver

    def request_many(self, requests):
        return self.solver.solve_many(requests)


class _Motion:
    def __init__(self, broker):
        self.broker = broker
        self._resampling_from_wrap = False

    def _resample_command(self, env_ids):
        transaction = self._reserve_action_ball_true_reset(env_ids)
        self._write_canonical_ready_state(env_ids)
        self._commit_action_ball_true_reset(env_ids, transaction)

    def _reserve_action_ball_true_reset(self, env_ids):
        self.broker.reserve_many_true_reset(tuple(env_ids))
        return {"count": len(env_ids)}

    def _write_canonical_ready_state(self, _env_ids):
        return None

    def _commit_action_ball_true_reset(self, _env_ids, _transaction):
        return None


class _Racket:
    def __init__(self, provider, broker, pool, solver):
        self._resample_is_wrap = False
        self._action_ball_runtime_initialized = True
        self._action_ball_birth_provider = provider
        self._action_ball_broker = broker
        self._action_ball_pool = pool
        self._action_ball_pool_solver = solver

    def _ensure_action_ball_runtime_initialized(self):
        return None

    def _sample_targets_action_ball(self, *_args, **_kwargs):
        return None

    def _resample_command(self, env_ids):
        self._ensure_action_ball_runtime_initialized()
        self._action_ball_retire_previous_births(env_ids)
        self._action_ball_broker.consume_many_true_reset(tuple(env_ids))
        self._action_ball_pool.request_many(tuple(env_ids))
        self._action_ball_commit_install(ids=env_ids)

    def _action_ball_retire_previous_births(self, _env_ids):
        return None

    def _action_ball_commit_install(self, **_kwargs):
        return None


def _rig():
    provider = _Provider()
    solver = _Solver()
    broker = _Broker(provider)
    pool = _Pool(solver)
    motion = _Motion(broker)
    racket = _Racket(provider, broker, pool, solver)
    terms = {"motion": motion, "racket": racket}
    manager = types.SimpleNamespace(
        active_terms=("motion", "racket"),
        get_term=terms.__getitem__,
    )
    env = types.SimpleNamespace(command_manager=manager)
    return env, motion, racket


def test_default_request_is_false_and_installs_nothing():
    env, motion, racket = _rig()
    motion_before = dict(motion.__dict__)
    racket_before = dict(racket.__dict__)

    assert P.parse_action_ball_update_profile_request({}) is False
    assert (
        P.parse_action_ball_update_profile_request(
            {P.PROFILE_ENV_VAR: "0"}
        )
        is False
    )
    assert motion.__dict__ == motion_before
    assert racket.__dict__ == racket_before
    assert env.command_manager.active_terms == ("motion", "racket")


def test_request_rejects_ambiguous_value_and_formal_mode():
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        P.parse_action_ball_update_profile_request(
            {P.PROFILE_ENV_VAR: "true"}
        )
    env, motion, racket = _rig()
    with pytest.raises(RuntimeError, match="formal profiling is fail-closed"):
        P.install_diagnostic_action_ball_update_profiler(
            env, diagnostic_fast_path=False
        )
    assert "_resample_command" not in motion.__dict__
    assert "_resample_command" not in racket.__dict__


def test_installer_rejects_runtime_broker_not_in_diagnostic_mode():
    env, motion, racket = _rig()
    racket._action_ball_broker.diagnostic_fast_path = False
    with pytest.raises(RuntimeError, match="non-diagnostic runtime broker"):
        P.install_diagnostic_action_ball_update_profiler(
            env, diagnostic_fast_path=True
        )
    assert "_resample_command" not in motion.__dict__
    assert "_resample_command" not in racket.__dict__


def test_diagnostic_profile_emits_stable_update_schema_and_resets():
    env, motion, racket = _rig()
    lines = []
    profiler = P.install_diagnostic_action_ball_update_profiler(
        env,
        diagnostic_fast_path=True,
        clock_ns=_Clock(),
        emit_line=lines.append,
    )
    motion._resample_command((0, 1, 2))
    racket._resample_command((0, 1, 2))
    motion._resampling_from_wrap = True
    racket._resample_is_wrap = True
    motion._resample_command((3, 4))
    racket._resample_command((3, 4))

    payload = profiler.emit_update(
        update=7,
        collection_time_s=0.100,
        learning_time_s=0.010,
    )
    assert len(lines) == 1
    assert lines[0].startswith(P.PROFILE_JSON_PREFIX)
    assert json.loads(lines[0][len(P.PROFILE_JSON_PREFIX) :]) == payload
    assert payload["schema_version"] == 1
    assert payload["update"] == 7
    assert payload["clock"] == "host_perf_counter_ns_no_cuda_sync"
    assert payload["reset_env_count"] == 3
    assert payload["wrap_env_count"] == 2
    assert payload["reset_ms_per_env"] is not None
    assert payload["wrap_ms_per_env"] is not None
    assert payload["segments"]["motion_true_reset_total"][
        "env_count"
    ] == 3
    assert payload["segments"]["racket_true_reset_total"][
        "env_count"
    ] == 3
    assert payload["segments"]["provider_provide_many"]["calls"] == 2
    assert payload["segments"]["solver_solve_many"]["calls"] == 2
    assert payload["unattributed"]["may_include_async_gpu_work"] is True
    assert payload["unattributed"]["nonnegative"] is True
    assert payload["unattributed"]["timing_scope_mismatch"] is False

    empty = profiler.emit_update(
        update=8,
        collection_time_s=0.001,
        learning_time_s=0.001,
    )
    assert empty["reset_env_count"] == 0
    assert empty["wrap_env_count"] == 0
    assert all(
        row["calls"] == 0 for row in empty["segments"].values()
    )

    profiler.close()
    assert "_resample_command" not in motion.__dict__
    assert "_resample_command" not in racket.__dict__


def test_active_span_closes_and_wrappers_unwind_after_exception():
    env, motion, _racket = _rig()
    lines = []
    profiler = P.install_diagnostic_action_ball_update_profiler(
        env,
        diagnostic_fast_path=True,
        clock_ns=_Clock(),
        emit_line=lines.append,
    )

    def fail(_requests, **_kwargs):
        raise RuntimeError("synthetic reset failure")

    motion.broker.reserve_many_true_reset = fail
    with pytest.raises(RuntimeError, match="synthetic reset failure"):
        motion._resample_command((0,))
    payload = profiler.emit_update(
        update=0,
        collection_time_s=0.010,
        learning_time_s=0.001,
    )
    assert payload["segments"]["motion_true_reset_total"]["calls"] == 1
    assert payload["segments"]["motion_reserve"]["calls"] == 1
    assert payload["reset_env_count"] == 1
    profiler.close()


@pytest.fixture()
def profile_runner_module(monkeypatch):
    return ER._load_runner_module(
        monkeypatch, ER._load_contract_module()
    )


class _ResetStop(RuntimeError):
    pass


class _ResetProbe:
    def __init__(self, *, stop=False):
        self.reset_calls = 0
        self._stop = stop

    def reset(self):
        self.reset_calls += 1
        if self._stop:
            raise _ResetStop("reset boundary reached")
        return None, {}


def _preflight_runner(
    runner_module,
    *,
    diagnostic,
    disable_logs,
    rank,
    stop_on_reset=False,
):
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.env = _ResetProbe(stop=stop_on_reset)
    runner.disable_logs = disable_logs
    runner.rank = rank
    runner.current_learning_iteration = 0
    runner._exact_resume_roundtrip_pending = True
    runner._action_ball_resume_reset_pending = True
    runner._service_action_ball_frozen_evaluation = lambda _step: False
    runner._action_ball_diagnostic_unauthorized = lambda: diagnostic
    return runner


def test_runner_formal_profile_rejects_before_resume_reset(
    profile_runner_module, monkeypatch
):
    monkeypatch.setenv(P.PROFILE_ENV_VAR, "1")
    runner = _preflight_runner(
        profile_runner_module,
        diagnostic=False,
        disable_logs=False,
        rank=0,
    )
    with pytest.raises(RuntimeError, match="formal profiling is fail-closed"):
        runner.learn(1)
    assert runner.env.reset_calls == 0
    assert runner._exact_resume_roundtrip_pending is True


def test_runner_disabled_logs_profile_rejects_before_resume_reset(
    profile_runner_module, monkeypatch
):
    monkeypatch.setenv(P.PROFILE_ENV_VAR, "1")
    runner = _preflight_runner(
        profile_runner_module,
        diagnostic=True,
        disable_logs=True,
        rank=0,
    )
    with pytest.raises(RuntimeError, match="disable_logs=False"):
        runner.learn(1)
    assert runner.env.reset_calls == 0
    assert runner._exact_resume_roundtrip_pending is True


def test_runner_nonprimary_profile_rejects_before_resume_reset(
    profile_runner_module, monkeypatch
):
    monkeypatch.setenv(P.PROFILE_ENV_VAR, "1")
    runner = _preflight_runner(
        profile_runner_module,
        diagnostic=True,
        disable_logs=False,
        rank=1,
    )
    with pytest.raises(RuntimeError, match="primary runner rank 0"):
        runner.learn(1)
    assert runner.env.reset_calls == 0
    assert runner._exact_resume_roundtrip_pending is True


def test_runner_default_unset_does_not_import_profiler(
    profile_runner_module, monkeypatch
):
    monkeypatch.delenv(P.PROFILE_ENV_VAR, raising=False)
    imported = []
    original_import = builtins.__import__

    def observe_import(name, *args, **kwargs):
        imported.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe_import)
    runner = _preflight_runner(
        profile_runner_module,
        diagnostic=True,
        disable_logs=False,
        rank=0,
        stop_on_reset=True,
    )
    with pytest.raises(_ResetStop, match="reset boundary reached"):
        runner.learn(1)
    assert runner.env.reset_calls == 1
    assert (
        "whole_body_tracking.utils.action_ball_update_profiler"
        not in imported
    )
