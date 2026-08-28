"""Bounded host-wall attribution for the real FullMDP rollout path.

The legacy ActionBall profiler binds the retired diagnostic broker.  This
profiler instead wraps the exact manager/env callpoints consumed by
``ActionBallFullMdpRsl3Runner``.  It is opt-in, introduces no CUDA
synchronization, and removes every wrapper automatically after a bounded
number of PPO updates so the same process can continue as the long run.

Durations are inclusive host wall spans.  Nested rows deliberately overlap;
they locate waits and Python work but do not claim GPU-kernel attribution.
"""

from __future__ import annotations

import functools
import importlib
import json
import math
import time
from collections.abc import Callable, Mapping


PROFILE_ENV_VAR = "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES"
PROFILE_JSON_PREFIX = "HOPE_ACTION_BALL_FULL_MDP_PROFILE_JSON="
PROFILE_SCHEMA_VERSION = 2
MAX_PROFILE_UPDATES = 50

_SEGMENT_NAMES = (
    "env_step_total",
    "before_policy_step",
    "step_may_start_assert",
    "protected_state_capture",
    "protected_state_assert",
    "action_process",
    "action_apply",
    "scene_write_data_to_sim",
    "sim_step",
    "scene_update",
    "post_physics_publish",
    "physical_epoch_postphysics",
    "physical_active_flight_capture",
    "r07_idle_stamp",
    "r07_idle_epoch_snapshot",
    "r07_idle_support_read",
    "r07_idle_state_store",
    "r07_keyed_publish",
    "r07_motion_projection",
    "motion_ready_install",
    "termination_compute",
    "reward_compute",
    "after_reward_close",
    "r06_reward_close_impl",
    "epoch_milestone_write",
    "selected_reset_total",
    "command_compute",
    "after_command_to_observation_gap",
    "d05_total",
    "d05_prepare",
    "d05_question_compose",
    "d05_rk4_horizon_discovery",
    "d05_physical_horizon_projection",
    "d05_rk4_exact_finalize",
    "d05_preview",
    "d05_build_transaction",
    "d05_epoch_settle",
    "d05_round_1_attempted_rows",
    "d05_round_2_attempted_rows",
    "d05_round_3_attempted_rows",
    "event_apply",
    "observation_compute",
    "recorder_callbacks",
    "ppo_drain_prepare",
    "ppo_drain_materialize_d2h_decode",
    "ppo_drain_ack",
)


def parse_full_mdp_profile_updates(environ: Mapping[str, str]) -> int:
    """Return the exact bounded request; absent/``0`` keeps the path inert."""

    raw = environ.get(PROFILE_ENV_VAR)
    if raw is None or raw == "" or raw == "0":
        return 0
    if not raw.isascii() or not raw.isdecimal() or raw.startswith("0"):
        raise RuntimeError(
            f"{PROFILE_ENV_VAR} must be 0 or a canonical positive integer"
        )
    updates = int(raw)
    if updates < 1 or updates > MAX_PROFILE_UPDATES:
        raise RuntimeError(
            f"{PROFILE_ENV_VAR} must be between 1 and {MAX_PROFILE_UPDATES}"
        )
    return updates


def _batch_size(value: object) -> int:
    try:
        size = len(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError("profiled reset batch has no length") from exc
    if type(size) is not int or size < 0:
        raise RuntimeError("profiled reset batch size differs")
    return size


def _is_exact_bound_method(owner: object, method_name: str) -> bool:
    """Reject inherited, instance-replaced, or foreign leaf callpoints."""

    function = vars(type(owner)).get(method_name)
    bound = getattr(owner, method_name, None)
    return bool(
        callable(function)
        and callable(bound)
        and getattr(bound, "__self__", None) is owner
        and getattr(bound, "__func__", None) is function
    )


class FullMdpUpdateProfiler:
    """Install reversible wrappers around one exact FullMDP environment."""

    def __init__(
        self,
        *,
        requested_updates: int,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        emit_line: Callable[[str], None] = print,
    ) -> None:
        if (
            type(requested_updates) is not int
            or requested_updates < 1
            or requested_updates > MAX_PROFILE_UPDATES
        ):
            raise RuntimeError("FullMDP profiler update budget differs")
        self._requested_updates = requested_updates
        self._clock_ns = clock_ns
        self._emit_line = emit_line
        self._wrapped: list[tuple[object, str, bool, object]] = []
        self._wrapped_keys: set[tuple[int, str]] = set()
        self._runtime_owner: object | None = None
        self._closed = False
        self._emitted_updates = 0
        self._marks: dict[str, int] = {}
        self._reset_update_counters()

    @property
    def closed(self) -> bool:
        return self._closed

    def _reset_update_counters(self) -> None:
        self._segments = {
            name: {"ns": 0, "calls": 0, "env_count": 0}
            for name in _SEGMENT_NAMES
        }

    def _record(self, name: str, elapsed_ns: int, env_count: int) -> None:
        row = self._segments[name]
        row["ns"] += int(elapsed_ns)
        row["calls"] += 1
        row["env_count"] += int(env_count)

    def _wrap_method(
        self,
        owner: object,
        method_name: str,
        *,
        segment_name: str,
        env_count_arg: int | None = None,
        mark_end: str | None = None,
        gap_from_mark: str | None = None,
        gap_segment_name: str | None = None,
        result_env_counts: Callable[[object], Mapping[str, int]] | None = None,
    ) -> None:
        if segment_name not in self._segments:
            raise RuntimeError("FullMDP profiler segment name differs")
        key = (id(owner), method_name)
        if key in self._wrapped_keys:
            raise RuntimeError("FullMDP profiler duplicate callpoint")
        if (gap_from_mark is None) != (gap_segment_name is None):
            raise RuntimeError("FullMDP profiler gap binding differs")
        if gap_segment_name is not None and gap_segment_name not in self._segments:
            raise RuntimeError("FullMDP profiler gap segment differs")
        original = getattr(owner, method_name, None)
        namespace = getattr(owner, "__dict__", None)
        if not callable(original) or not isinstance(namespace, dict):
            raise RuntimeError(
                "FullMDP profiler cannot bind "
                f"{type(owner).__module__}.{type(owner).__qualname__}."
                f"{method_name}"
            )
        had_instance_attr = method_name in namespace
        previous_instance_attr = namespace.get(method_name)

        @functools.wraps(original)
        def profiled(*args, **kwargs):
            env_count = (
                0
                if env_count_arg is None
                else _batch_size(args[env_count_arg])
            )
            started_ns = self._clock_ns()
            if gap_from_mark is not None:
                prior_ns = self._marks.pop(gap_from_mark, None)
                if prior_ns is not None:
                    self._record(
                        gap_segment_name,
                        started_ns - prior_ns,
                        0,
                    )
            succeeded = False
            try:
                result = original(*args, **kwargs)
                succeeded = True
                if result_env_counts is not None:
                    counts = result_env_counts(result)
                    if type(counts) is not dict:
                        raise RuntimeError(
                            "FullMDP profiler result counters differ"
                        )
                    for count_name, count in counts.items():
                        if (
                            count_name not in self._segments
                            or type(count) is not int
                            or count < 0
                        ):
                            raise RuntimeError(
                                "FullMDP profiler result counter differs"
                            )
                        self._record(count_name, 0, count)
                return result
            finally:
                finished_ns = self._clock_ns()
                self._record(
                    segment_name,
                    finished_ns - started_ns,
                    env_count,
                )
                if succeeded and mark_end is not None:
                    self._marks[mark_end] = finished_ns

        setattr(owner, method_name, profiled)
        self._wrapped.append(
            (owner, method_name, had_instance_attr, previous_instance_attr)
        )
        self._wrapped_keys.add(key)

    def profile_runtime_call(
        self,
        segment_name: str,
        operation: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Measure one already-authenticated runtime call without replacing it."""

        if self._closed or segment_name not in self._segments:
            raise RuntimeError("FullMDP profiler runtime segment differs")
        if not callable(operation):
            raise RuntimeError("FullMDP profiler runtime operation differs")
        started_ns = self._clock_ns()
        try:
            return operation(*args, **kwargs)
        finally:
            self._record(segment_name, self._clock_ns() - started_ns, 0)

    def install(self, env: object) -> None:
        if self._closed or self._wrapped:
            raise RuntimeError("FullMDP profiler can be installed exactly once")
        module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.full_mdp_env"
        )
        expected_type = getattr(
            module, "ActionBallFullMdpManagerBasedRLEnv", None
        )
        unwrapped = getattr(env, "unwrapped", env)
        if not isinstance(expected_type, type) or type(unwrapped) is not expected_type:
            raise RuntimeError("FullMDP profiler requires the exact live env")
        if not hasattr(unwrapped, "_full_mdp_runtime_owner"):
            raise RuntimeError("FullMDP profiler requires the installed top owner")
        runtime_owner = unwrapped._full_mdp_runtime_owner
        runtime_namespace = getattr(runtime_owner, "__dict__", None)
        if (
            not isinstance(runtime_namespace, dict)
            or "_full_mdp_profile_runtime_call" in runtime_namespace
        ):
            raise RuntimeError("FullMDP runtime profiler binding differs")
        setattr(
            runtime_owner,
            "_full_mdp_profile_runtime_call",
            self.profile_runtime_call,
        )
        self._runtime_owner = runtime_owner
        try:
            components = dict(runtime_owner.component_identities)
            r05 = components.get("r05_runtime")
            motion = components.get("motion")
            physical = components.get("physical_ball")
            r06 = components.get("r06_landing_outcome")
            r07 = components.get("r07_recovery")
            epoch = runtime_owner.epoch_owner
            r07_epoch = getattr(r07, "action_epoch_owner", None)
            r07_owner = getattr(r07, "owner", None)
            if (
                r07_epoch is not epoch
                or getattr(r07_owner, "_diagnostic_n2_bundle", None) is not r07
            ):
                raise RuntimeError("FullMDP profiler R07 identities differ")
            if (
                physical is None
                or r06 is None
                or getattr(physical, "_action_epoch_owner", None) is not epoch
                or getattr(r06, "_action_ball_full_mdp_epoch_owner", None)
                is not epoch
                or not _is_exact_bound_method(
                    physical, "capture_post_physics_facts"
                )
                or not _is_exact_bound_method(
                    r06, "_close_action_ball_full_mdp_epoch_reward_rows_impl"
                )
                or not _is_exact_bound_method(
                    epoch, "_milestone_after_business_write"
                )
                or any(
                    not _is_exact_bound_method(epoch, method)
                    for method in (
                        "prepare_drain",
                        "materialize_drain",
                        "acknowledge_drain",
                    )
                )
            ):
                raise RuntimeError("FullMDP profiler Physical/R06 identities differ")
            question_compose = getattr(r05, "_internal_question_compose", None)
            question_bundle = getattr(question_compose, "__self__", None)
            question_core = getattr(question_bundle, "_physical_owner", None)
            compose_function = getattr(question_compose, "__func__", None)
            if (
                r05 is None
                or motion is None
                or getattr(r05, "_diagnostic_epoch_owner", None)
                is not epoch
                or getattr(r05, "_diagnostic_motion_owner", None) is not motion
                or question_bundle is None
                or not _is_exact_bound_method(
                    question_bundle,
                    "compose_r05_candidate_bank_inside_prepare",
                )
                or compose_function
                is not vars(type(question_bundle)).get(
                    "compose_r05_candidate_bank_inside_prepare"
                )
                or question_core is None
                or any(
                    not _is_exact_bound_method(question_core, method)
                    for method in (
                        "issue_horizon_for_test",
                        "project_horizon_for_test",
                        "finalize_exact_ticks_for_test",
                    )
                )
            ):
                raise RuntimeError("FullMDP profiler D05 identities differ")
        except (AttributeError, TypeError, ValueError) as exc:
            self.close()
            raise RuntimeError("FullMDP profiler component identities differ") from exc
        except BaseException:
            self.close()
            raise
        bindings = (
            (unwrapped, "step", "env_step_total", None),
            (unwrapped, "_before_policy_step", "before_policy_step", None),
            (unwrapped, "_assert_step_may_start", "step_may_start_assert", None),
            (unwrapped, "_protected_manager_state", "protected_state_capture", None),
            (
                unwrapped,
                "_assert_protected_manager_state_unchanged",
                "protected_state_assert",
                None,
            ),
            (
                unwrapped,
                "_publish_post_physics_substep",
                "post_physics_publish",
                None,
            ),
            (unwrapped, "_after_reward_close", "after_reward_close", None),
            (unwrapped, "_reset_idx", "selected_reset_total", 0),
            (unwrapped.action_manager, "process_action", "action_process", None),
            (unwrapped.action_manager, "apply_action", "action_apply", None),
            (
                unwrapped.scene,
                "write_data_to_sim",
                "scene_write_data_to_sim",
                None,
            ),
            (unwrapped.sim, "step", "sim_step", None),
            (unwrapped.scene, "update", "scene_update", None),
            (
                unwrapped.termination_manager,
                "compute",
                "termination_compute",
                None,
            ),
            (unwrapped.reward_manager, "compute", "reward_compute", None),
            (unwrapped.command_manager, "compute", "command_compute", None),
            (
                physical,
                "capture_post_physics_facts",
                "physical_active_flight_capture",
                None,
            ),
            (
                r06,
                "_close_action_ball_full_mdp_epoch_reward_rows_impl",
                "r06_reward_close_impl",
                None,
            ),
            (
                epoch,
                "_milestone_after_business_write",
                "epoch_milestone_write",
                None,
            ),
            (r05, "_prepare_many_impl", "d05_prepare", None),
            (r05, "_internal_question_compose", "d05_question_compose", None),
            (
                question_core,
                "issue_horizon_for_test",
                "d05_rk4_horizon_discovery",
                None,
            ),
            (
                question_core,
                "project_horizon_for_test",
                "d05_physical_horizon_projection",
                None,
            ),
            (
                question_core,
                "finalize_exact_ticks_for_test",
                "d05_rk4_exact_finalize",
                None,
            ),
            (r05, "_preview_impl", "d05_preview", None),
            (r05, "_build_row_transaction", "d05_build_transaction", None),
            (
                runtime_owner.epoch_owner,
                "settle_d05_transaction",
                "d05_epoch_settle",
                None,
            ),
            (unwrapped.event_manager, "apply", "event_apply", None),
            (
                unwrapped.observation_manager,
                "compute",
                "observation_compute",
                None,
            ),
            (
                r07_epoch,
                "snapshot_idle_observation_chronology",
                "r07_idle_epoch_snapshot",
                None,
            ),
            (
                r07_owner,
                "stamp_action_epoch_idle_observation",
                "r07_idle_state_store",
                None,
            ),
            (epoch, "prepare_drain", "ppo_drain_prepare", None),
            (
                epoch,
                "materialize_drain",
                "ppo_drain_materialize_d2h_decode",
                None,
            ),
            (epoch, "acknowledge_drain", "ppo_drain_ack", None),
        )

        def d05_round_attempt_counts(token: object) -> dict[str, int]:
            records = getattr(r05, "_prepared_records", None)
            record = records.get(token) if isinstance(records, dict) else None
            attempts = getattr(record, "rounds_attempted", None)
            active = getattr(record, "rng_advance_mask", None)
            if (
                attempts is None
                or getattr(attempts, "ndim", None) != 1
                or getattr(attempts, "shape", (None,))[0] != getattr(
                    r05, "_num_envs", None
                )
                or active is None
                or getattr(active, "ndim", None) != 1
                or getattr(active, "shape", None) != getattr(
                    attempts, "shape", None
                )
            ):
                raise RuntimeError("FullMDP profiler D05 attempts differ")
            histogram = (
                attempts[active].bincount(minlength=4).detach().cpu().tolist()
            )
            if (
                type(histogram) is not list
                or len(histogram) < 4
                or any(type(value) is not int or value < 0 for value in histogram)
            ):
                raise RuntimeError("FullMDP profiler D05 histogram differs")
            return {
                f"d05_round_{round_index}_attempted_rows": sum(
                    histogram[round_index:]
                )
                for round_index in range(1, 4)
            }
        recorder_methods = (
            "record_pre_step",
            "record_post_physics_decimation_step",
            "record_post_step",
            "record_pre_reset",
            "record_post_reset",
        )
        try:
            for owner, method, segment, env_count_arg in bindings:
                kwargs = {}
                if owner is unwrapped.command_manager and method == "compute":
                    kwargs["mark_end"] = "command_compute_end"
                elif owner is unwrapped.observation_manager and method == "compute":
                    kwargs.update(
                        gap_from_mark="command_compute_end",
                        gap_segment_name="after_command_to_observation_gap",
                    )
                elif owner is r05 and method == "_prepare_many_impl":
                    kwargs["result_env_counts"] = d05_round_attempt_counts
                self._wrap_method(
                    owner,
                    method,
                    segment_name=segment,
                    env_count_arg=env_count_arg,
                    **kwargs,
                )
            for method in recorder_methods:
                self._wrap_method(
                    unwrapped.recorder_manager,
                    method,
                    segment_name="recorder_callbacks",
                )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _milliseconds(seconds: object, *, label: str) -> float:
        if (
            isinstance(seconds, bool)
            or type(seconds) not in (int, float)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.0
        ):
            raise RuntimeError(f"FullMDP profiler invalid {label}")
        return float(seconds) * 1000.0

    def emit_update(
        self,
        *,
        update: int,
        collection_time_s: object,
        learning_time_s: object,
        expected_env_step_calls: int,
    ) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("closed FullMDP profiler cannot emit")
        if type(update) is not int or update < 0:
            raise RuntimeError("FullMDP profiler update differs")
        if type(expected_env_step_calls) is not int or expected_env_step_calls < 1:
            raise RuntimeError("FullMDP profiler rollout length differs")
        collection_ms = self._milliseconds(
            collection_time_s, label="collection_time"
        )
        learning_ms = self._milliseconds(
            learning_time_s, label="learning_time"
        )
        rows: dict[str, dict[str, object]] = {}
        for name in _SEGMENT_NAMES:
            raw = self._segments[name]
            elapsed_ms = float(raw["ns"]) / 1.0e6
            calls = int(raw["calls"])
            rows[name] = {
                "calls": calls,
                "env_count": int(raw["env_count"]),
                "inclusive_host_wall_ms": round(elapsed_ms, 6),
                "ms_per_call": (
                    None if calls == 0 else round(elapsed_ms / calls, 6)
                ),
            }
        env_step_calls = int(self._segments["env_step_total"]["calls"])
        self._emitted_updates += 1
        close_after_emit = self._emitted_updates == self._requested_updates
        payload: dict[str, object] = {
            "event": "action_ball_full_mdp_update_profile",
            "schema_version": PROFILE_SCHEMA_VERSION,
            "update": update,
            "profile_update_ordinal": self._emitted_updates,
            "requested_profile_updates": self._requested_updates,
            "clock": "host_perf_counter_ns_no_cuda_sync",
            "inclusive_nested_spans": True,
            "speed_evidence_eligible": False,
            "collection_ms": round(collection_ms, 6),
            "learning_ms": round(learning_ms, 6),
            "expected_env_step_calls": expected_env_step_calls,
            "observed_env_step_calls": env_step_calls,
            "rollout_call_count_exact": env_step_calls
            == expected_env_step_calls,
            "auto_close_after_emit": close_after_emit,
            "segments": rows,
        }
        line = PROFILE_JSON_PREFIX + json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._emit_line(line)
        self._reset_update_counters()
        self._marks.clear()
        if close_after_emit:
            self.close()
        return payload

    def close(self) -> None:
        if self._closed:
            return
        for owner, method, had_attr, previous in reversed(self._wrapped):
            if had_attr:
                setattr(owner, method, previous)
            else:
                delattr(owner, method)
        self._wrapped.clear()
        self._wrapped_keys.clear()
        if self._runtime_owner is not None:
            namespace = getattr(self._runtime_owner, "__dict__", None)
            if isinstance(namespace, dict):
                namespace.pop("_full_mdp_profile_runtime_call", None)
            self._runtime_owner = None
        self._marks.clear()
        self._closed = True


def install_full_mdp_update_profiler(
    env: object,
    *,
    requested_updates: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    emit_line: Callable[[str], None] = print,
) -> FullMdpUpdateProfiler:
    profiler = FullMdpUpdateProfiler(
        requested_updates=requested_updates,
        clock_ns=clock_ns,
        emit_line=emit_line,
    )
    profiler.install(env)
    return profiler
