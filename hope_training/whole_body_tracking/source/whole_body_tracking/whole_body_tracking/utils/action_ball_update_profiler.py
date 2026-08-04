"""Opt-in host-wall profiler for diagnostic ActionBall reset work.

The profiler deliberately installs instance-level wrappers only when the
diagnostic runner opts in.  A normal run does not import this module, install a
wrapper, allocate a CUDA event, or synchronize the device.

The spans use ``perf_counter_ns`` and therefore measure Python host wall time,
including waits at synchronization points that already exist in the wrapped
code.  They do not pretend to attribute asynchronous GPU kernels: no new CUDA
synchronization is introduced for profiling.
"""

from __future__ import annotations

import functools
import json
import math
import time
from typing import Callable, Mapping, Optional


PROFILE_ENV_VAR = "HOPE_ACTION_BALL_UPDATE_PROFILE"
PROFILE_JSON_PREFIX = "HOPE_ACTION_BALL_UPDATE_PROFILE_JSON="
PROFILE_SCHEMA_VERSION = 2

_SEGMENT_NAMES = (
    "motion_true_reset_total",
    "motion_wrap_total",
    "motion_reserve",
    "motion_state_write",
    "motion_commit",
    "racket_runtime_ensure",
    "racket_true_reset_total",
    "racket_wrap_total",
    "racket_retire",
    "broker_reserve",
    "provider_provide_many",
    "broker_consume",
    "pool_request_many",
    "solver_solve_many",
    "racket_install",
)


def parse_action_ball_update_profile_request(
    environ: Mapping[str, str],
) -> bool:
    """Parse the exact opt-in without accepting typo-shaped modes."""

    value = environ.get(PROFILE_ENV_VAR)
    if value is None or value == "" or value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(
        f"{PROFILE_ENV_VAR} must be exactly 0 or 1 when set"
    )


def _batch_size(value: object) -> int:
    """Return a reset batch size without forcing a device-to-host transfer."""

    try:
        size = len(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError(
            "profiled ActionBall reset batch has no length"
        ) from exc
    if type(size) is not int or size < 0:
        raise RuntimeError(
            "profiled ActionBall reset batch has an invalid length"
        )
    return size


class ActionBallUpdateProfiler:
    """Install reversible diagnostic-only wrappers and emit one update row."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        emit_line: Callable[[str], None] = print,
    ) -> None:
        self._clock_ns = clock_ns
        self._emit_line = emit_line
        self._wrapped = []
        self._wrapped_keys = set()
        self._closed = False
        self._components_installed = False
        self._reset_update_counters()

    def _reset_update_counters(self) -> None:
        self._segments = {
            name: {"ns": 0, "calls": 0, "env_count": 0}
            for name in _SEGMENT_NAMES
        }
        self._reset_env_count = 0
        self._wrap_env_count = 0

    @staticmethod
    def _wrap_segment_for(owner: object, *, prefix: str) -> str:
        return (
            f"{prefix}_wrap_total"
            if bool(getattr(owner, "_resampling_from_wrap", False))
            or bool(getattr(owner, "_resample_is_wrap", False))
            else f"{prefix}_true_reset_total"
        )

    def _record(
        self, name: str, *, elapsed_ns: int, env_count: int = 0
    ) -> None:
        row = self._segments[name]
        row["ns"] += int(elapsed_ns)
        row["calls"] += 1
        row["env_count"] += int(env_count)

    def _wrap_method(
        self,
        owner: object,
        method_name: str,
        *,
        segment_name: Optional[str] = None,
        segment_resolver: Optional[Callable[[], str]] = None,
        env_count_arg: Optional[int] = None,
        count_reset_kind: Optional[str] = None,
        after_success: Optional[Callable[[], None]] = None,
    ) -> None:
        key = (id(owner), method_name)
        if key in self._wrapped_keys:
            return
        original = getattr(owner, method_name, None)
        if not callable(original):
            raise RuntimeError(
                "ActionBall update profiler cannot bind callable "
                f"{type(owner).__name__}.{method_name}"
            )
        if (segment_name is None) == (segment_resolver is None):
            raise RuntimeError(
                "profile wrapper requires exactly one segment selector"
            )
        namespace = getattr(owner, "__dict__", None)
        if not isinstance(namespace, dict):
            raise RuntimeError(
                "ActionBall update profiler requires mutable instance methods"
            )
        had_instance_attr = method_name in namespace
        previous_instance_attr = namespace.get(method_name)

        @functools.wraps(original)
        def profiled(*args, **kwargs):
            name = (
                segment_name
                if segment_name is not None
                else segment_resolver()
            )
            env_count = (
                0
                if env_count_arg is None
                else _batch_size(args[env_count_arg])
            )
            started_ns = self._clock_ns()
            succeeded = False
            try:
                result = original(*args, **kwargs)
                succeeded = True
                return result
            finally:
                elapsed_ns = self._clock_ns() - started_ns
                self._record(
                    name, elapsed_ns=elapsed_ns, env_count=env_count
                )
                if count_reset_kind == "from_segment" and name.endswith(
                    "_true_reset_total"
                ):
                    self._reset_env_count += env_count
                elif count_reset_kind == "from_segment" and name.endswith(
                    "_wrap_total"
                ):
                    self._wrap_env_count += env_count
                if succeeded and after_success is not None:
                    after_success()

        setattr(owner, method_name, profiled)
        self._wrapped.append(
            (
                owner,
                method_name,
                had_instance_attr,
                previous_instance_attr,
            )
        )
        self._wrapped_keys.add(key)

    def _install_runtime_components(self, racket: object) -> None:
        if self._components_installed:
            return
        if not bool(
            getattr(racket, "_action_ball_runtime_initialized", False)
        ):
            return
        provider = getattr(racket, "_action_ball_birth_provider", None)
        broker = getattr(racket, "_action_ball_broker", None)
        pool = getattr(racket, "_action_ball_pool", None)
        solver = getattr(racket, "_action_ball_pool_solver", None)
        if any(value is None for value in (provider, broker, pool, solver)):
            raise RuntimeError(
                "diagnostic ActionBall runtime initialized without "
                "provider/broker/pool/solver components"
            )
        if getattr(broker, "diagnostic_fast_path", None) is not True:
            raise RuntimeError(
                "ActionBall update profiler refuses a non-diagnostic "
                "runtime broker"
            )
        self._wrap_method(
            provider,
            "provide_many",
            segment_name="provider_provide_many",
        )
        self._wrap_method(
            broker,
            "reserve_many_true_reset",
            segment_name="broker_reserve",
        )
        self._wrap_method(
            broker,
            "consume_many_true_reset",
            segment_name="broker_consume",
        )
        self._wrap_method(
            pool,
            "request_many",
            segment_name="pool_request_many",
        )
        self._wrap_method(
            solver,
            "solve_many",
            segment_name="solver_solve_many",
        )
        self._components_installed = True

    def install(self, env: object) -> None:
        """Bind one Motion/Racket pair; partially installed wrappers unwind."""

        if self._closed or self._wrapped:
            raise RuntimeError(
                "ActionBall update profiler can be installed exactly once"
            )
        unwrapped = getattr(env, "unwrapped", env)
        manager = getattr(unwrapped, "command_manager", None)
        getter = None if manager is None else getattr(manager, "get_term", None)
        if not callable(getter):
            raise RuntimeError(
                "ActionBall update profiler requires command_manager.get_term"
            )
        names = tuple(getattr(manager, "active_terms", ()))
        terms = tuple(getter(name) for name in names)
        motions = tuple(
            term
            for term in terms
            if callable(
                getattr(term, "_reserve_action_ball_true_reset", None)
            )
            and callable(
                getattr(term, "_commit_action_ball_true_reset", None)
            )
        )
        rackets = tuple(
            term
            for term in terms
            if callable(
                getattr(term, "_sample_targets_action_ball", None)
            )
            and callable(
                getattr(
                    term,
                    "_ensure_action_ball_runtime_initialized",
                    None,
                )
            )
        )
        if len(motions) != 1 or len(rackets) != 1:
            raise RuntimeError(
                "ActionBall update profiler requires exactly one Motion "
                f"and one Racket term; got motion={len(motions)}, "
                f"racket={len(rackets)}"
            )
        motion = motions[0]
        racket = rackets[0]
        try:
            self._wrap_method(
                motion,
                "_resample_command",
                segment_resolver=lambda: self._wrap_segment_for(
                    motion, prefix="motion"
                ),
                env_count_arg=0,
                count_reset_kind="from_segment",
            )
            self._wrap_method(
                motion,
                "_reserve_action_ball_true_reset",
                segment_name="motion_reserve",
            )
            self._wrap_method(
                motion,
                "_write_canonical_ready_state",
                segment_name="motion_state_write",
            )
            self._wrap_method(
                motion,
                "_commit_action_ball_true_reset",
                segment_name="motion_commit",
            )
            self._wrap_method(
                racket,
                "_resample_command",
                segment_resolver=lambda: self._wrap_segment_for(
                    racket, prefix="racket"
                ),
                env_count_arg=0,
                count_reset_kind=None,
            )
            self._wrap_method(
                racket,
                "_action_ball_retire_previous_births",
                segment_name="racket_retire",
            )
            self._wrap_method(
                racket,
                "_action_ball_commit_install",
                segment_name="racket_install",
            )
            # Runtime construction is lazy.  This wrapper adds no timing row;
            # it merely attaches component spans before the first task issue.
            self._wrap_method(
                racket,
                "_ensure_action_ball_runtime_initialized",
                segment_name="racket_runtime_ensure",
                after_success=lambda: self._install_runtime_components(
                    racket
                ),
            )
            self._install_runtime_components(racket)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _seconds_to_ms(value: object, *, name: str) -> float:
        if (
            isinstance(value, bool)
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(
                f"ActionBall update profiler got invalid {name}"
            )
        return float(value) * 1000.0

    @staticmethod
    def _rounded(value: float) -> float:
        return round(float(value), 6)

    def emit_update(
        self,
        *,
        update: int,
        collection_time_s: object,
        learning_time_s: object,
        reset_reason_counters: Optional[Mapping[str, int]] = None,
    ) -> dict:
        if self._closed:
            raise RuntimeError(
                "closed ActionBall update profiler cannot emit"
            )
        if type(update) is not int or update < 0:
            raise RuntimeError(
                "ActionBall update profiler requires non-negative update"
            )
        collection_ms = self._seconds_to_ms(
            collection_time_s, name="collection_time"
        )
        learning_ms = self._seconds_to_ms(
            learning_time_s, name="learning_time"
        )
        reason_counters = {}
        if reset_reason_counters is not None:
            if not isinstance(reset_reason_counters, Mapping):
                raise RuntimeError(
                    "ActionBall update profiler reset reasons must be a mapping"
                )
            for raw_name, raw_count in reset_reason_counters.items():
                if type(raw_name) is not str or not raw_name:
                    raise RuntimeError(
                        "ActionBall update profiler reset reason name is invalid"
                    )
                if type(raw_count) is not int or raw_count < 0:
                    raise RuntimeError(
                        "ActionBall update profiler reset reason count is invalid"
                    )
                reason_counters[raw_name] = raw_count
        segment_rows = {}
        for name in _SEGMENT_NAMES:
            raw = self._segments[name]
            elapsed_ms = float(raw["ns"]) / 1.0e6
            calls = int(raw["calls"])
            segment_rows[name] = {
                "calls": calls,
                "env_count": int(raw["env_count"]),
                "ms": self._rounded(elapsed_ms),
                "ms_per_call": (
                    None
                    if calls == 0
                    else self._rounded(elapsed_ms / calls)
                ),
            }
        reset_ms = sum(
            float(self._segments[name]["ns"]) / 1.0e6
            for name in (
                "motion_true_reset_total",
                "racket_true_reset_total",
            )
        )
        wrap_ms = sum(
            float(self._segments[name]["ns"]) / 1.0e6
            for name in ("motion_wrap_total", "racket_wrap_total")
        )
        profiled_reset_ms = reset_ms + wrap_ms
        unattributed_collection_ms = collection_ms - profiled_reset_ms
        payload = {
            "event": "action_ball_update_profile",
            "schema_version": PROFILE_SCHEMA_VERSION,
            "update": update,
            "clock": "host_perf_counter_ns_no_cuda_sync",
            "measurement_mode": "profile_on_attribution_only",
            "profile_overhead_present": True,
            "speed_evidence_eligible": False,
            "gpu_kernel_attribution": {
                "claimed": False,
                "reason": (
                    "host wall spans do not synchronize or delimit "
                    "asynchronous GPU kernels"
                ),
            },
            "collection_ms": self._rounded(collection_ms),
            "learning_ms": self._rounded(learning_ms),
            "total_ms": self._rounded(collection_ms + learning_ms),
            "update_wall_ms": self._rounded(
                collection_ms + learning_ms
            ),
            "reset_env_count": int(self._reset_env_count),
            "wrap_env_count": int(self._wrap_env_count),
            "reset_ms_per_env": (
                None
                if self._reset_env_count == 0
                else self._rounded(reset_ms / self._reset_env_count)
            ),
            "wrap_ms_per_env": (
                None
                if self._wrap_env_count == 0
                else self._rounded(wrap_ms / self._wrap_env_count)
            ),
            "profiled_reset_ms": self._rounded(profiled_reset_ms),
            "reset_strata": {
                "true_reset_env_count": int(self._reset_env_count),
                "wrap_env_count": int(self._wrap_env_count),
                "exact_behavior_counters": dict(
                    sorted(reason_counters.items())
                ),
                "reason_counter_source": (
                    "same-update exact behavior ledger"
                ),
            },
            "unattributed": {
                "collection_ms": self._rounded(
                    unattributed_collection_ms
                ),
                "may_include_async_gpu_work": True,
                "nonnegative": unattributed_collection_ms >= 0.0,
                "timing_scope_mismatch": (
                    unattributed_collection_ms < 0.0
                ),
            },
            "segments": segment_rows,
        }
        self._emit_line(
            PROFILE_JSON_PREFIX
            + json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        self._reset_update_counters()
        return payload

    def close(self) -> None:
        if self._closed:
            return
        for (
            owner,
            method_name,
            had_instance_attr,
            previous_instance_attr,
        ) in reversed(self._wrapped):
            if had_instance_attr:
                setattr(owner, method_name, previous_instance_attr)
            else:
                delattr(owner, method_name)
        self._wrapped.clear()
        self._wrapped_keys.clear()
        self._closed = True


def install_diagnostic_action_ball_update_profiler(
    env: object,
    *,
    diagnostic_fast_path: bool,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    emit_line: Callable[[str], None] = print,
) -> ActionBallUpdateProfiler:
    """Construct only for diagnostic ActionBall; formal use is fail-closed."""

    if type(diagnostic_fast_path) is not bool:
        raise TypeError("diagnostic_fast_path must be an exact bool")
    if not diagnostic_fast_path:
        raise RuntimeError(
            f"{PROFILE_ENV_VAR}=1 is allowed only for diagnostic "
            "ActionBall; formal profiling is fail-closed"
        )
    profiler = ActionBallUpdateProfiler(
        clock_ns=clock_ns, emit_line=emit_line
    )
    profiler.install(env)
    return profiler
