"""Hydra training entry for HOPE Agibot A3 WBC (106B-Final-Project style).

Pick the task/algo YAML on the command line and override any field:

    python scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
        registry_name=<entity>/wandb-registry-motions/hope_forehand

    python scripts/train.py task=TrackingFlat algo=ppo num_envs=2048 max_iterations=20000 \
        registry_name=<org>/wandb-registry-motions/hope_forehand

Tune by editing cfg/task/*.yaml (env / reward / racket / DR) and cfg/algo/ppo.yaml (PPO). This
script reuses BeyondMimic's training mechanics (Isaac Lab + rsl_rl). Local video-generated `.npz`
motions are first-class inputs; the WandB motion registry is an optional sharing/publishing layer.
The legacy `scripts/rsl_rl/train.py --task=... --registry_name=...` still works too.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

import hydra
from omegaconf import ListConfig, OmegaConf


def _capture_original_training_argv() -> tuple[str, ...]:
    """Capture the kernel argv before Hydra and Kit rewrite ``sys.argv``."""

    try:
        parts = pathlib.Path("/proc/self/cmdline").read_bytes().split(b"\0")
        argv = tuple(part.decode("utf-8", "strict") for part in parts if part)
    except (OSError, UnicodeDecodeError):
        argv = ()
    if argv:
        return argv
    # Non-Linux source tests never publish a queue binding.  Keep ordinary
    # training portable while the formal Linux callback still rechecks /proc.
    return (
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        *sys.argv[1:],
    )


_ORIGINAL_TRAINING_ARGV = _capture_original_training_argv()


def _lean_queue_binding_requested(cfg) -> bool:
    """Return whether this is a queue launch, rejecting a half-bound request."""

    claim_path = _get(cfg, "training_queue_claim_path")
    binding_path = _get(cfg, "training_run_binding_path")
    if claim_path is None and binding_path is None:
        return False
    if claim_path is None or binding_path is None:
        raise RuntimeError(
            "training_queue_claim_path and training_run_binding_path must be supplied together"
        )
    return True


def _emit_lean_queue_phase(cfg, phase: str, **fields) -> None:
    """Emit machine-readable boot telemetry only for the lean queue."""

    if not _lean_queue_binding_requested(cfg):
        return
    payload = {
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    print(
        "[train.py] LEAN_QUEUE_PHASE "
        + json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def dump_pickle(filename: str, data):
    """Compatibility helper for IsaacLab builds that no longer expose dump_pickle."""
    import os
    import pickle

    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# --------------------------------------------------------------------------- #
# Task YAML -> Isaac Lab env cfg overrides (only keys present in the YAML are applied).
# --------------------------------------------------------------------------- #
def _get(node, key, default=None):
    try:
        return node.get(key, default)
    except Exception:
        return default


def _explicitly_null(node, key) -> bool:
    """True only when ``key`` is PRESENT and its value is null — not when it is absent.

    ``_get`` collapses "absent" and "written as null" into the same None, which makes "clear this
    knob" inexpressible for any setting whose cfg default is non-None.
    """
    try:
        if key not in node:
            return False
    except Exception:
        return False
    return _get(node, key, _MISSING) is None


_MISSING = object()


def _publish_lean_queue_binding_if_requested(cfg, log_dir: str) -> None:
    """Publish the trainer-owned RSL directory binding for queue launches only."""

    if not _lean_queue_binding_requested(cfg):
        return
    claim_path = _get(cfg, "training_queue_claim_path")
    binding_path = _get(cfg, "training_run_binding_path")
    claim_digest = _get(cfg, "training_launch_claim_sha256")
    if claim_digest is None:
        raise RuntimeError(
            "lean queue binding requires training_launch_claim_sha256"
        )
    from lean_queue_runtime import publish_run_binding

    binding = publish_run_binding(
        claim_path=str(claim_path),
        binding_path=str(binding_path),
        log_dir=log_dir,
        claim_digest=str(claim_digest),
        actual_argv=_ORIGINAL_TRAINING_ARGV,
    )
    print(
        "[train.py] LEAN_QUEUE_RUN_BOUND: "
        f"path={binding_path} sha256={binding['content_sha256']} log={log_dir}",
        flush=True,
    )
    _emit_lean_queue_phase(cfg, "log_dir_bound", rsl_log_dir=log_dir)


_KIT_CARB_TASKING_THREAD_SETTING = "/plugins/carb.tasking.plugin/threadCount"
_KIT_TBB_THREAD_SETTING = "/plugins/omni.tbb.globalcontrol/maxThreadCount"
_KIT_USE_OMNI_JOB_SETTING = "/plugins/carb.tasking.plugin/useOmniJob"


def _resolve_kit_thread_caps(cfg):
    """Return optional, exact Kit argv for a paired runtime thread cap."""

    carb_count = _get(cfg, "kit_carb_tasking_thread_count")
    tbb_count = _get(cfg, "kit_tbb_thread_count")
    if carb_count is None and tbb_count is None:
        return None, None, None
    if carb_count is None or tbb_count is None:
        raise ValueError(
            "kit_carb_tasking_thread_count and kit_tbb_thread_count must be supplied together"
        )
    for name, value in (
        ("kit_carb_tasking_thread_count", carb_count),
        ("kit_tbb_thread_count", tbb_count),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer (bool is not accepted), got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value!r}")
    kit_args = (
        f"--{_KIT_CARB_TASKING_THREAD_SETTING}={carb_count} "
        f"--{_KIT_TBB_THREAD_SETTING}={tbb_count}"
    )
    return kit_args, carb_count, tbb_count


def _verify_kit_thread_caps(settings, expected_carb_count: int, expected_tbb_count: int) -> None:
    """Fail closed unless Kit applied both caps and disabled the OmniJob backend."""

    observed_carb_count = settings.get(_KIT_CARB_TASKING_THREAD_SETTING)
    observed_tbb_count = settings.get(_KIT_TBB_THREAD_SETTING)
    use_omni_job = settings.get(_KIT_USE_OMNI_JOB_SETTING)
    if type(observed_carb_count) is not int or observed_carb_count != expected_carb_count:
        raise RuntimeError(
            "Kit carb.tasking thread cap mismatch: "
            f"expected={expected_carb_count!r} observed={observed_carb_count!r}"
        )
    if type(observed_tbb_count) is not int or observed_tbb_count != expected_tbb_count:
        raise RuntimeError(
            "Kit omni.tbb thread cap mismatch: "
            f"expected={expected_tbb_count!r} observed={observed_tbb_count!r}"
        )
    if type(use_omni_job) is not bool or use_omni_job is not False:
        raise RuntimeError(
            "Kit carb.tasking useOmniJob must be exactly false, "
            f"observed={use_omni_job!r}"
        )
    print(
        "[train.py] KIT_THREAD_CAP_OK: "
        f"carb.tasking={observed_carb_count} omni.tbb={observed_tbb_count} "
        "useOmniJob=false",
        flush=True,
    )


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


def _as_explicit_bool(x, name: str) -> bool:
    """Parse a safety/plant switch without turning a typo into an implicit ``False``."""
    if isinstance(x, bool):
        return x
    value = str(x).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise _OverrideError(f"{name} must be an explicit boolean, got {x!r}")


_LATERAL_TRAINING_SPEC_ATTR = "_hope_lateral_perturbation_training_spec_v1"


def _apply_lateral_perturbation_task_override(env_cfg, task, applied) -> None:
    """Translate the narrow, default-off Hydra surface for the frozen L0/L1 pair.

    An absent section and an explicit ``enabled=false`` do not attach any attribute to the
    environment config, preserving the historical runtime and hard-contract bytes.  L0 is the
    matched zero-impulse scheduler cell, so selecting a cell while disabled is a configuration
    error rather than a silent no-op.
    """

    node = _get(task, "lateral_perturbation")
    if node is None:
        return
    try:
        node.keys()
    except Exception as exc:
        raise _OverrideError(
            f"task.lateral_perturbation must be a mapping, got {node!r}"
        ) from exc
    _check_unknown_keys(
        node,
        ("enabled", "cell", "seed"),
        "task.lateral_perturbation",
    )
    enabled_raw = _get(node, "enabled")
    enabled = (
        False
        if enabled_raw is None
        else _as_explicit_bool(enabled_raw, "task.lateral_perturbation.enabled")
    )
    cell = _get(node, "cell")
    seed = _get(node, "seed")
    if not enabled:
        if cell is not None or seed is not None:
            raise _OverrideError(
                "task.lateral_perturbation.cell/seed require enabled=true; use enabled=true, "
                "cell=L0 for the matched zero-impulse control"
            )
        applied.append("lateral_perturbation.enabled=False (historical no-hook path)")
        return
    if type(cell) is not str or cell not in ("L0", "L1"):
        raise _OverrideError(
            "task.lateral_perturbation.cell must be exactly 'L0' or 'L1'"
        )
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise _OverrideError(
            "task.lateral_perturbation.seed must be an exact uint32 integer "
            "(bool/coercion forbidden)"
        )
    commands = getattr(env_cfg, "commands", None)
    _require(
        commands is not None
        and getattr(commands, "motion", None) is not None
        and getattr(commands, "racket_target", None) is not None,
        "commands.motion + commands.racket_target "
        "(task.lateral_perturbation.enabled=true)",
    )
    if hasattr(env_cfg, _LATERAL_TRAINING_SPEC_ATTR):
        raise _OverrideError(
            "composed env cfg already owns the lateral trainer spec attribute; refusing a "
            "competing activation writer"
        )
    setattr(
        env_cfg,
        _LATERAL_TRAINING_SPEC_ATTR,
        {"schema_version": 1, "cell": cell, "seed": seed},
    )
    applied.append(
        "lateral_perturbation="
        f"(enabled=True,cell={cell},seed={seed},recovery_hold_only,frozen_L1_envelope)"
    )


# YAML keys under `push:` (Wave-P random base push; PACE/BeyondMimic-style shove, default OFF =
# the HITTER-aligned no-push recipe every running matrix cell trains with).  Same fail-loud
# contract as _RACKET_KEYS/_MOTION_KEYS: every key must be whitelisted here AND consumed below.
_PUSH_KEYS = ("enable", "interval_range_s", "vel_xy_mps", "ang_vel_radps", "ang_axes")


def _apply_push_robot_task_override(env_cfg, task, applied) -> None:
    """Translate the default-off ``task.push.*`` surface into the interval push event.

    人话:训练时每隔几秒随机"推机器人一把"(直接改底座线速度/角速度),练抗扰平衡。
    An absent ``task.push`` section (all currently running matrix cells) is a byte-for-byte
    no-op: ``events.push_robot`` stays ``None``.  ``enable=false`` may not carry dormant
    amplitudes/axes; ``enable=true`` requires the COMPLETE recipe (interval + both amplitudes +
    axes) so every arm states its push explicitly.  Amplitude/axis consistency is validated by
    ``training_contract.push_robot_event_block`` — the single assembly source shared with the
    env-cfg flag path (hope_env_cfg.apply_push_robot_event) and the schema-3 validator.
    """

    node = _get(task, "push")
    if node is None:
        return
    try:
        node.keys()
    except Exception as exc:
        raise _OverrideError(f"task.push must be a mapping, got {node!r}") from exc
    _check_unknown_keys(node, _PUSH_KEYS, "task.push")
    enable_raw = _get(node, "enable")
    if enable_raw is None:
        raise _OverrideError("task.push must explicitly set enable=true|false")
    enable = _as_explicit_bool(enable_raw, "task.push.enable")
    if not enable:
        dormant = sorted(
            key for key in _PUSH_KEYS if key != "enable" and _get(node, key) is not None
        )
        if dormant:
            raise _OverrideError(
                f"task.push.enable=false may not carry dormant push fields {dormant} — "
                "a disabled push with a loaded amplitude/axis is a config error; delete "
                "them or set enable=true"
            )
        applied.append("push.enable=False (historical no-push path)")
        return
    missing = sorted(key for key in _PUSH_KEYS if _get(node, key) is None)
    if missing:
        raise _OverrideError(
            f"task.push.enable=true requires the complete push recipe; missing {missing}"
        )
    raw_interval = _get(node, "interval_range_s")
    try:
        interval_items = list(raw_interval)
    except TypeError as exc:
        raise _OverrideError(
            f"task.push.interval_range_s must be a [lo, hi] pair of seconds, got {raw_interval!r}"
        ) from exc
    if len(interval_items) != 2:
        raise _OverrideError(
            f"task.push.interval_range_s must be a [lo, hi] pair of seconds, got {raw_interval!r}"
        )
    interval = tuple(
        _as_exact_float(value, f"task.push.interval_range_s[{index}]")
        for index, value in enumerate(interval_items)
    )
    vel_xy = _as_exact_float(_get(node, "vel_xy_mps"), "task.push.vel_xy_mps")
    ang_vel = _as_exact_float(_get(node, "ang_vel_radps"), "task.push.ang_vel_radps")
    ang_axes = str(_get(node, "ang_axes"))
    from whole_body_tracking.utils.training_contract import push_robot_event_block

    try:
        block = push_robot_event_block(
            enable=True,
            interval_range_s=interval,
            vel_xy_mps=vel_xy,
            ang_vel_radps=ang_vel,
            ang_axes=ang_axes,
        )
    except ValueError as exc:
        raise _OverrideError(f"task.push: {exc}") from exc
    _require(
        hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"),
        "events.push_robot (task.push.enable=true)",
    )
    from isaaclab.managers import EventTermCfg as _EventTerm

    from whole_body_tracking.tasks.tracking import mdp as _mdp

    env_cfg.events.push_robot = _EventTerm(
        func=_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        ),
        params={
            "velocity_range": {
                axis: (float(rng[0]), float(rng[1]))
                for axis, rng in block["velocity_range"].items()
            }
        },
    )
    push_flags = getattr(env_cfg, "push", None)
    if push_flags is not None:  # keep the descriptive HOPE cfg flag group honest
        push_flags.enable = True
        push_flags.interval_range_s = (
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        )
        push_flags.vel_xy_mps = float(vel_xy)
        push_flags.ang_vel_radps = float(ang_vel)
        push_flags.ang_axes = ang_axes
    applied.append(
        "events.push_robot=interval "
        f"{float(block['interval_range_s'][0])}-{float(block['interval_range_s'][1])}s "
        f"vxy=±{float(vel_xy)}m/s ang=±{float(ang_vel)}rad/s axes={ang_axes} "
        "(Wave-P random base push)"
    )


def _push_robot_event_contract(env_cfg) -> dict | None:
    """Bind the (post-override) random base-push event into the hard contract.

    人话:合同照抄实际生效的 push 事件;没开(= push_robot None,所有在跑矩阵格)就不写
    这个块,合同字节与历史逐位相同。An unrecognized push term shape (non-interval mode,
    asymmetric box, z push, unequal x/y, alien axis set) or a half-wired flag/term pair is
    REFUSED rather than silently escaping the contract.
    """

    events = getattr(env_cfg, "events", None)
    term = None if events is None else getattr(events, "push_robot", None)
    push_flags = getattr(env_cfg, "push", None)
    if term is None:
        if push_flags is not None and bool(getattr(push_flags, "enable", False)):
            raise RuntimeError(
                "push.enable=true but events.push_robot is None (half-wired push)"
            )
        return None
    if push_flags is not None and not bool(getattr(push_flags, "enable", False)):
        raise RuntimeError(
            "events.push_robot is active but push.enable=false (half-wired push)"
        )
    func = getattr(term, "func", None)
    func_name = func if isinstance(func, str) else getattr(func, "__name__", None)
    if func_name != "push_by_setting_velocity":
        raise RuntimeError(
            f"push_robot event func must be push_by_setting_velocity, got {func_name!r}"
        )
    if getattr(term, "mode", None) != "interval":
        raise RuntimeError(
            f"push_robot event mode must be 'interval', got {getattr(term, 'mode', None)!r}"
        )
    interval = tuple(float(value) for value in term.interval_range_s)
    params = getattr(term, "params", None)
    velocity_range = params.get("velocity_range") if isinstance(params, dict) else None
    if not hasattr(velocity_range, "items"):
        raise RuntimeError("push_robot event params must carry a velocity_range mapping")
    axes_present = {str(axis) for axis in velocity_range}

    def _symmetric_amplitude(axis):
        rng = velocity_range[axis]
        lo, hi = float(rng[0]), float(rng[1])
        if not (hi >= 0.0 and lo == -hi):
            raise RuntimeError(
                f"push_robot velocity_range.{axis} must be a symmetric ±v pair, got {rng!r}"
            )
        return hi

    if not {"x", "y"} <= axes_present:
        raise RuntimeError("push_robot velocity_range must push x and y")
    vel_x = _symmetric_amplitude("x")
    vel_y = _symmetric_amplitude("y")
    if vel_x != vel_y:
        raise RuntimeError(
            "push_robot x/y amplitudes must match (one shared vel_xy_mps)"
        )
    angular_axes = axes_present - {"x", "y"}
    if angular_axes == set():
        ang_axes, ang_vel = "none", 0.0
    elif angular_axes == {"yaw"}:
        ang_axes, ang_vel = "yaw", _symmetric_amplitude("yaw")
    elif angular_axes == {"roll", "pitch", "yaw"}:
        amplitudes = {
            axis: _symmetric_amplitude(axis) for axis in ("roll", "pitch", "yaw")
        }
        if len(set(amplitudes.values())) != 1:
            raise RuntimeError(
                "push_robot roll/pitch/yaw amplitudes must match (one shared ang_vel_radps)"
            )
        ang_axes, ang_vel = "rpy", amplitudes["yaw"]
    else:
        raise RuntimeError(
            f"push_robot velocity_range axes {sorted(axes_present)} do not match the "
            "none|yaw|rpy Wave-P recipe"
        )
    from whole_body_tracking.utils.training_contract import push_robot_event_block

    block = push_robot_event_block(
        enable=True,
        interval_range_s=interval,
        vel_xy_mps=vel_x,
        ang_vel_radps=ang_vel,
        ang_axes=ang_axes,
    )
    canonical = {
        str(axis): [float(rng[0]), float(rng[1])]
        for axis, rng in velocity_range.items()
    }
    if canonical != block["velocity_range"]:
        raise RuntimeError(
            "push_robot velocity_range disagrees with the canonical Wave-P assembly"
        )
    return block


# YAML keys under `force_push:` (F-axis interval FORCE push; matched-impulse companion of the
# velocity push, default OFF = no force push, byte-identical to every running matrix cell).
# Same fail-loud contract as _PUSH_KEYS: every key must be whitelisted here AND consumed below.
_FORCE_PUSH_KEYS = ("enable", "interval_range_s", "force_n", "duration_s")


def _force_push_control_dt_s(env_cfg) -> float:
    """Read the control step length (sim.dt x decimation) off the composed env cfg, fail-closed."""

    sim = getattr(env_cfg, "sim", None)
    dt = None if sim is None else getattr(sim, "dt", None)
    decimation = getattr(env_cfg, "decimation", None)
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(float(dt))
        or float(dt) <= 0.0
        or isinstance(decimation, bool)
        or type(decimation) is not int
        or decimation < 1
    ):
        raise _OverrideError(
            "task.force_push requires a composed env cfg with finite sim.dt > 0 and "
            f"integer decimation >= 1, got dt={dt!r} decimation={decimation!r}"
        )
    return float(dt) * int(decimation)


def _apply_force_push_task_override(env_cfg, task, applied) -> None:
    """Translate the default-off ``task.force_push.*`` surface into the force-push event pair.

    人话:训练时每隔几秒朝水平随机方向对 pelvis_link 施加持续 ``duration_s`` 秒的恒力(推底座
    练抗扰),与速度推档位按同冲量对表(Δv_equiv = F·Δt/m,运行时算好写进合同)。An absent
    ``task.force_push`` section (all currently running matrix cells) is a byte-for-byte no-op:
    ``events.force_push`` and ``events.force_push_sweep`` both stay ``None``.  ``enable=false``
    may not carry dormant fields; ``enable=true`` requires the COMPLETE recipe (interval +
    force_n + duration_s) so every arm states its push explicitly.  Consistency is validated by
    ``training_contract.force_push_event_block`` — the single assembly source shared with the
    env-cfg flag path (hope_env_cfg.apply_force_push_event) and the schema-3 validator.  The
    sweeper term is wired HERE together with the force term: Isaac interval events are not
    called per step, so expiry needs its own high-frequency term or the force never clears.
    """

    node = _get(task, "force_push")
    if node is None:
        return
    try:
        node.keys()
    except Exception as exc:
        raise _OverrideError(f"task.force_push must be a mapping, got {node!r}") from exc
    _check_unknown_keys(node, _FORCE_PUSH_KEYS, "task.force_push")
    enable_raw = _get(node, "enable")
    if enable_raw is None:
        raise _OverrideError("task.force_push must explicitly set enable=true|false")
    enable = _as_explicit_bool(enable_raw, "task.force_push.enable")
    if not enable:
        dormant = sorted(
            key
            for key in _FORCE_PUSH_KEYS
            if key != "enable" and _get(node, key) is not None
        )
        if dormant:
            raise _OverrideError(
                f"task.force_push.enable=false may not carry dormant force-push fields "
                f"{dormant} — a disabled push with a loaded force/duration is a config "
                "error; delete them or set enable=true"
            )
        applied.append("force_push.enable=False (historical no-force-push path)")
        return
    missing = sorted(key for key in _FORCE_PUSH_KEYS if _get(node, key) is None)
    if missing:
        raise _OverrideError(
            f"task.force_push.enable=true requires the complete force-push recipe; "
            f"missing {missing}"
        )
    raw_interval = _get(node, "interval_range_s")
    try:
        interval_items = list(raw_interval)
    except TypeError as exc:
        raise _OverrideError(
            "task.force_push.interval_range_s must be a [lo, hi] pair of seconds, "
            f"got {raw_interval!r}"
        ) from exc
    if len(interval_items) != 2:
        raise _OverrideError(
            "task.force_push.interval_range_s must be a [lo, hi] pair of seconds, "
            f"got {raw_interval!r}"
        )
    interval = tuple(
        _as_exact_float(value, f"task.force_push.interval_range_s[{index}]")
        for index, value in enumerate(interval_items)
    )
    force_n = _as_exact_float(_get(node, "force_n"), "task.force_push.force_n")
    duration_s = _as_exact_float(_get(node, "duration_s"), "task.force_push.duration_s")
    control_dt_s = _force_push_control_dt_s(env_cfg)
    from whole_body_tracking.utils.training_contract import force_push_event_block

    try:
        block = force_push_event_block(
            enable=True,
            interval_range_s=interval,
            force_n=force_n,
            duration_s=duration_s,
            control_dt_s=control_dt_s,
        )
    except ValueError as exc:
        raise _OverrideError(f"task.force_push: {exc}") from exc
    _require(
        hasattr(env_cfg, "events")
        and hasattr(env_cfg.events, "force_push")
        and hasattr(env_cfg.events, "force_push_sweep"),
        "events.force_push + events.force_push_sweep (task.force_push.enable=true)",
    )
    from isaaclab.managers import EventTermCfg as _EventTerm

    from whole_body_tracking.tasks.tracking import mdp as _mdp

    env_cfg.events.force_push = _EventTerm(
        func=_mdp.push_by_applying_wrench,
        mode="interval",
        interval_range_s=(
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        ),
        params={
            "force_n": float(block["force_n"]),
            "duration_steps": int(block["duration_steps"]),
            "body_name": str(block["body_name"]),
        },
    )
    env_cfg.events.force_push_sweep = _EventTerm(
        func=_mdp.sweep_expired_force_pushes,
        mode="interval",
        interval_range_s=(control_dt_s, control_dt_s),
        params={},
    )
    force_push_flags = getattr(env_cfg, "force_push", None)
    if force_push_flags is not None:  # keep the descriptive HOPE cfg flag group honest
        force_push_flags.enable = True
        force_push_flags.interval_range_s = (
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        )
        force_push_flags.force_n = float(force_n)
        force_push_flags.duration_s = float(duration_s)
    applied.append(
        "events.force_push=interval "
        f"{float(block['interval_range_s'][0])}-{float(block['interval_range_s'][1])}s "
        f"F={float(force_n)}N dur={float(duration_s)}s "
        f"({int(block['duration_steps'])} steps @ pelvis_link origin) "
        "+ per-control-step expiry sweep (F-axis force push)"
    )


def _force_push_event_contract(env_cfg, env) -> dict | None:
    """Bind the (post-override) F-axis force-push event pair into the hard contract.

    人话:合同照抄实际生效的力推事件,并记录运行时真实读到的机器人总质量与换算出的
    Δv_equiv = force_n × duration_s / m_robot,供与速度推档位(p02/p035/p05/p08)对表。
    质量读的是 articulation 的初始质量表(data.default_mass,USD 名义值;逐 env 的
    randomize_link_mass ±10% 让每个 env 的真实 Δv 也散 ±10%,合同记名义值)。没开
    (= 两个事件都 None,所有在跑矩阵格)就不写这个块,合同字节与历史逐位相同。半接线
    (旗标开着但事件没挂 / 施力事件在但清扫事件缺 —— 力永远清不掉)与走样的事件形状一律
    REFUSED,绝不静默漏出合同。
    """

    events = getattr(env_cfg, "events", None)
    term = None if events is None else getattr(events, "force_push", None)
    sweep = None if events is None else getattr(events, "force_push_sweep", None)
    flags = getattr(env_cfg, "force_push", None)
    if term is None:
        if flags is not None and bool(getattr(flags, "enable", False)):
            raise RuntimeError(
                "force_push.enable=true but events.force_push is None (half-wired force push)"
            )
        if sweep is not None:
            raise RuntimeError(
                "events.force_push_sweep is active without events.force_push "
                "(half-wired force push)"
            )
        return None
    if flags is None:
        raise RuntimeError(
            "events.force_push is active but the descriptive force_push flag group is "
            "missing (half-wired force push)"
        )
    if not bool(getattr(flags, "enable", False)):
        raise RuntimeError(
            "events.force_push is active but force_push.enable=false (half-wired force push)"
        )
    if sweep is None:
        raise RuntimeError(
            "events.force_push without events.force_push_sweep — expired forces would "
            "never clear (half-wired force push)"
        )
    func = getattr(term, "func", None)
    func_name = func if isinstance(func, str) else getattr(func, "__name__", None)
    if func_name != "push_by_applying_wrench":
        raise RuntimeError(
            f"force_push event func must be push_by_applying_wrench, got {func_name!r}"
        )
    if getattr(term, "mode", None) != "interval":
        raise RuntimeError(
            f"force_push event mode must be 'interval', got {getattr(term, 'mode', None)!r}"
        )
    sweep_func = getattr(sweep, "func", None)
    sweep_name = (
        sweep_func if isinstance(sweep_func, str) else getattr(sweep_func, "__name__", None)
    )
    if sweep_name != "sweep_expired_force_pushes":
        raise RuntimeError(
            f"force_push sweep func must be sweep_expired_force_pushes, got {sweep_name!r}"
        )
    if getattr(sweep, "mode", None) != "interval":
        raise RuntimeError("force_push sweep mode must be 'interval'")
    params = getattr(term, "params", None)
    if not isinstance(params, dict) or set(params) != {
        "force_n", "duration_steps", "body_name",
    }:
        raise RuntimeError(
            "force_push event params must be exactly {force_n, duration_steps, body_name}"
        )
    control_dt_s = _force_push_control_dt_s(env_cfg)
    sweep_interval = tuple(float(v) for v in sweep.interval_range_s)
    if sweep_interval != (control_dt_s, control_dt_s):
        raise RuntimeError(
            "force_push sweep interval must be exactly (control_dt, control_dt) so expiry "
            f"runs every control step, got {sweep_interval!r} vs dt={control_dt_s!r}"
        )
    interval = tuple(float(value) for value in term.interval_range_s)
    duration_s = float(getattr(flags, "duration_s"))
    from whole_body_tracking.utils.training_contract import (
        bind_force_push_runtime_mass,
        force_push_event_block,
    )

    block = force_push_event_block(
        enable=True,
        interval_range_s=interval,
        force_n=float(params["force_n"]),
        duration_s=duration_s,
        control_dt_s=control_dt_s,
    )
    if (
        str(params["body_name"]) != block["body_name"]
        or int(params["duration_steps"]) != block["duration_steps"]
        or float(getattr(flags, "force_n")) != block["force_n"]
        or tuple(float(v) for v in getattr(flags, "interval_range_s")) != tuple(
            block["interval_range_s"]
        )
    ):
        raise RuntimeError(
            "force_push event term disagrees with the descriptive flag group / canonical "
            "assembly (body_name, duration_steps, force_n or interval drifted)"
        )
    robot = env.scene["robot"]
    masses = getattr(getattr(robot, "data", None), "default_mass", None)
    if masses is None:
        raise RuntimeError(
            "force_push contract requires articulation data.default_mass to record the "
            "runtime robot mass"
        )
    try:
        robot_mass_kg = float(masses[0].sum())
    except Exception as exc:
        raise RuntimeError(
            "force_push contract could not sum articulation data.default_mass"
        ) from exc
    return bind_force_push_runtime_mass(block, robot_mass_kg=robot_mass_kg)


def _resolve_lateral_training_runtime(env):
    """Return ``(cfg, hard_contract)`` for an enabled cell, else ``None``."""

    spec = getattr(env.cfg, _LATERAL_TRAINING_SPEC_ATTR, None)
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {"schema_version", "cell", "seed"}:
        raise RuntimeError("lateral trainer runtime spec has an invalid key set")
    if spec["schema_version"] != 1:
        raise RuntimeError("lateral trainer runtime spec schema is not v1")
    from whole_body_tracking.tasks.tracking.mdp.lateral_perturbation import (
        frozen_lateral_training_config,
    )
    from whole_body_tracking.tasks.tracking.mdp.isaac_lateral_perturbation import (
        isaac_lateral_event_term_manifest,
        isaac_lateral_training_hard_contract,
    )

    cfg = frozen_lateral_training_config(
        cell=spec["cell"], seed=spec["seed"], policy_dt_s=float(env.step_dt)
    )
    event_term_manifest = isaac_lateral_event_term_manifest(
        getattr(env, "event_manager", None)
    )
    return cfg, isaac_lateral_training_hard_contract(
        cell=spec["cell"],
        cfg=cfg,
        event_term_manifest=event_term_manifest,
    )


def _as_exact_int(x, name: str) -> int:
    if type(x) is not int:
        raise _OverrideError(f"{name} must be an exact integer (bool/coercion forbidden), got {x!r}")
    return x


def _as_exact_float(x, name: str) -> float:
    if type(x) is not float or not math.isfinite(x):
        raise _OverrideError(f"{name} must be an exact finite float, got {x!r}")
    return x


def _load_task_first_manifest_from_racket_cfg(racket_cfg):
    """Load the byte-pinned task-first manifest from the composed command cfg."""

    path = str(getattr(racket_cfg, "task_first_manifest_path", "") or "").strip()
    expected_sha256 = str(
        getattr(racket_cfg, "task_first_manifest_sha256", "") or ""
    ).strip()
    if not path:
        raise _OverrideError(
            "[train.py] racket.target_mode=task_first requires "
            "racket.task_first_manifest_path"
        )
    if not expected_sha256:
        raise _OverrideError(
            "[train.py] racket.target_mode=task_first requires "
            "racket.task_first_manifest_sha256"
        )
    from whole_body_tracking.tasks.tracking.mdp.task_first_manifest import (
        load_task_first_manifest,
    )

    try:
        loaded = load_task_first_manifest(
            path,
            expected_sha256=expected_sha256,
            require_training_authorized=True,
        )
    except (OSError, ValueError) as exc:
        raise _OverrideError(
            f"[train.py] invalid task-first manifest {path!r}: {exc}"
        ) from exc
    return loaded


def _task_first_manifest_contract(racket_cfg, motion_cfg, env_cfg) -> dict:
    """Return the immutable task/action/curriculum identity for checkpoints."""

    # Do not cache ``LoadedTaskFirstManifest`` on an Isaac configclass.  It
    # contains pathlib.Path and dataclass objects; Isaac's standard class-to-dict
    # dump includes single-underscore attributes and would emit a Python-specific
    # YAML object that cannot be read back with ``yaml.full_load``.  Reloading this
    # small, byte-pinned JSON also rechecks that it did not change between gates.
    loaded = _load_task_first_manifest_from_racket_cfg(racket_cfg)
    from whole_body_tracking.tasks.tracking.mdp.task_first_manifest import (
        build_task_first_training_contract,
    )

    return build_task_first_training_contract(
        loaded,
        racket_cfg=racket_cfg,
        motion_cfg=motion_cfg,
        env_cfg=env_cfg,
    )


def _action_ball_repo_root(motion_cfg) -> pathlib.Path:
    """Return the trusted root used to resolve action-ball referenced assets."""

    configured = str(
        getattr(motion_cfg, "canonical_registry_repo_root", "") or ""
    ).strip()
    if configured and not pathlib.Path(configured).is_absolute():
        raise _OverrideError(
            "[train.py] action-ball canonical_registry_repo_root must be "
            "absolute when explicitly configured; relative paths would make "
            "the process CWD part of motion authority"
        )
    root = (
        pathlib.Path(configured)
        if configured
        else pathlib.Path(__file__).resolve().parents[3]
    )
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _OverrideError(
            f"[train.py] action-ball repo root does not resolve: {root!s}"
        ) from exc
    if not resolved.is_dir():
        raise _OverrideError(
            f"[train.py] action-ball repo root is not a directory: {resolved!s}"
        )
    return resolved


def _load_action_ball_manifest_from_cfg(racket_cfg, motion_cfg):
    """Load exact manifest bytes and re-open every referenced action asset.

    This is a byte-identity preflight only.  Metadata is intentionally unable
    to mint formal motion admission; ``_build_training_hard_contract`` requires
    the instantiated MotionCommand's separate code-rooted receipt.
    """

    path = str(
        getattr(racket_cfg, "action_ball_manifest_path", "") or ""
    ).strip()
    expected_sha256 = str(
        getattr(racket_cfg, "action_ball_manifest_sha256", "") or ""
    ).strip()
    if not path:
        raise _OverrideError(
            "[train.py] racket.target_mode=action_ball requires "
            "racket.action_ball_manifest_path"
        )
    if not expected_sha256:
        raise _OverrideError(
            "[train.py] racket.target_mode=action_ball requires "
            "racket.action_ball_manifest_sha256"
        )
    from whole_body_tracking.tasks.tracking.mdp.action_ball_manifest import (
        load_action_ball_manifest,
    )

    try:
        return load_action_ball_manifest(
            path,
            expected_sha256=expected_sha256,
            verify_referenced_assets=True,
            repo_root=_action_ball_repo_root(motion_cfg),
            require_formal_admission=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise _OverrideError(
            f"[train.py] invalid action-ball manifest {path!r}: {exc}"
        ) from exc


def _canonical_contract_sha256(value) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _action_ball_ready_root_z_by_slot(loaded, motion_cfg) -> tuple[float, ...]:
    """Read the exact float32 ready-root Z that ``MotionLoader`` will expose.

    Runtime selects the first configured tracked body from each schema-2
    motion and converts ``body_pos_w`` to torch.float32.  Preflight must derive
    the same values from the already byte-verified action assets; otherwise
    the adapter/profile SHAs are guaranteed to drift as soon as ready Z is
    nonzero.
    """

    assets = loaded.referenced_assets
    if assets is None:
        raise _OverrideError(
            "[train.py] action-ball ready-root preflight requires verified "
            "referenced motion assets"
        )
    selected_body_names = tuple(
        str(value)
        for value in (getattr(motion_cfg, "body_names", ()) or ())
    )
    if not selected_body_names or not selected_body_names[0]:
        raise _OverrideError(
            "[train.py] action-ball ready-root preflight requires a "
            "non-empty motion.body_names table"
        )
    ready_body_name = selected_body_names[0]

    import io
    import numpy as np

    ready_root_z = []
    for slot, (action, asset) in enumerate(
        zip(loaded.manifest.actions, assets.motions)
    ):
        try:
            raw = asset.resolved_path.read_bytes()
        except OSError as exc:
            raise _OverrideError(
                "[train.py] action-ball ready-root motion cannot be read: "
                f"slot={slot} action={action.action_id!r}"
            ) from exc
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != asset.sha256:
            raise _OverrideError(
                "[train.py] action-ball ready-root motion bytes changed "
                "after manifest verification: "
                f"slot={slot} action={action.action_id!r} "
                f"verified={asset.sha256} actual={actual_sha256}"
            )
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                if "body_names" not in data.files:
                    raise ValueError(
                        "schema-2 body_names metadata is required"
                    )
                names_raw = np.asarray(data["body_names"])
                if names_raw.ndim != 1:
                    raise ValueError("body_names must be one-dimensional")
                body_names = []
                for value in names_raw.tolist():
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    body_names.append(str(value))
                if (
                    not body_names
                    or len(body_names) != len(set(body_names))
                    or any(not name for name in body_names)
                ):
                    raise ValueError(
                        "body_names must contain unique non-empty names"
                    )
                try:
                    ready_body_index = body_names.index(ready_body_name)
                except ValueError as exc:
                    raise ValueError(
                        f"tracked ready body {ready_body_name!r} is absent"
                    ) from exc
                body_pos_w = np.asarray(data["body_pos_w"])
                if (
                    body_pos_w.ndim != 3
                    or body_pos_w.shape[0] < 1
                    or body_pos_w.shape[1] != len(body_names)
                    or body_pos_w.shape[2] != 3
                ):
                    raise ValueError(
                        "body_pos_w must have exact (T,body_names,3) shape"
                    )
                # MotionLoader materializes this array as torch.float32 before
                # runtime reads frame zero.  Match that rounding exactly.
                ready_z = float(
                    np.float32(body_pos_w[0, ready_body_index, 2])
                )
                if not math.isfinite(ready_z):
                    raise ValueError("ready-root Z is non-finite")
        except (KeyError, UnicodeDecodeError, ValueError) as exc:
            raise _OverrideError(
                "[train.py] action-ball ready-root motion contract is "
                f"invalid at slot={slot} action={action.action_id!r}: {exc}"
            ) from exc
        ready_root_z.append(ready_z)
    if len(ready_root_z) != len(loaded.manifest.actions):
        raise _OverrideError(
            "[train.py] action-ball ready-root motion count disagrees with "
            "the manifest"
        )
    return tuple(ready_root_z)


def _action_ball_preflight_contract(
    racket_cfg,
    motion_cfg,
    *,
    policy_dt_s: float,
) -> dict:
    """Build the independent launch-side identity expected from runtime."""

    loaded = _load_action_ball_manifest_from_cfg(racket_cfg, motion_cfg)
    if loaded.referenced_assets is None:
        raise _OverrideError(
            "[train.py] action-ball preflight did not verify referenced assets"
        )
    from whole_body_tracking.tasks.tracking.mdp.action_ball_profile_adapter import (
        adapt_action_ball_manifest,
        build_curriculum_config,
    )
    from whole_body_tracking.tasks.tracking.mdp.action_ball_sampling import (
        ARM_CATALOG_SHA256,
        ActionBallSampler,
        SamplingMixture,
    )

    ready_root_z_by_slot = _action_ball_ready_root_z_by_slot(
        loaded,
        motion_cfg,
    )
    adapted = adapt_action_ball_manifest(
        loaded.manifest,
        ready_root_z_by_slot=ready_root_z_by_slot,
    )
    seed = getattr(racket_cfg, "action_ball_seed", None)
    if type(seed) is not int or not 0 <= seed < (1 << 63):
        raise _OverrideError(
            "[train.py] action-ball requires racket.action_ball_seed "
            "as a plain integer in [0,2**63)"
        )
    if (
        isinstance(policy_dt_s, bool)
        or type(policy_dt_s) not in (int, float)
        or not math.isfinite(float(policy_dt_s))
        or float(policy_dt_s) <= 0.0
    ):
        raise _OverrideError(
            "[train.py] action-ball preflight requires the exact finite "
            "positive policy control step"
        )
    sampler = ActionBallSampler(
        adapted.profiles,
        seed=seed,
        sampling_mixture=SamplingMixture(),
        contact_time_step_s=float(policy_dt_s),
    )
    curriculum_config = build_curriculum_config(loaded.manifest).as_dict()
    profile_adapter = adapted.to_contract()
    try:
        manifest_relative_path = (
            loaded.source_path.resolve(strict=True)
            .relative_to(loaded.referenced_assets.repo_root)
            .as_posix()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise _OverrideError(
            "[train.py] action-ball manifest must be a regular tracked/input "
            "file below the trusted repository root"
        ) from exc
    action_bindings = [
        {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "action_slot": index,
            "family": action.family,
            "motion_path": action.motion_path,
            "motion_sha256": action.motion_sha256,
            "sampling_profile_sha256": adapted.profile_sha256[index],
            "strike_phase": action.strike_phase,
            "mount_normal_sign": action.mount_normal_sign,
        }
        for index, action in enumerate(loaded.manifest.actions)
    ]
    contract = {
        "schema_version": 1,
        "manifest": {
            "path": manifest_relative_path,
            "file_sha256": loaded.file_sha256,
            "canonical_sha256": loaded.canonical_sha256,
            "manifest_id": loaded.manifest.manifest_id,
        },
        "mobility_mode": loaded.manifest.mobility_mode,
        "action_order": list(loaded.manifest.action_order),
        "action_uids": [
            action.action_uid for action in loaded.manifest.actions
        ],
        "ready_root_z_by_slot_m": list(ready_root_z_by_slot),
        "action_bindings": action_bindings,
        "prototype": {
            "path": loaded.manifest.prototype.path,
            "scope": loaded.manifest.prototype.scope,
            "sha256": loaded.manifest.prototype.sha256,
        },
        "profile_adapter": {
            "contract": profile_adapter,
            "sha256": adapted.contract_sha256,
        },
        "sampler": {
            "contract_sha256": sampler.sampler_contract_sha256,
            "arm_catalog_sha256": ARM_CATALOG_SHA256,
            "seed": seed,
            "pool_refill_rows": getattr(
                racket_cfg, "action_ball_pool_refill_rows", None
            ),
        },
        "solver_profile_sha256": loaded.manifest.solver_profile_sha256,
        "physics_profile_sha256": loaded.manifest.physics_profile_sha256,
        "curriculum": {
            "config": curriculum_config,
            "config_sha256": _canonical_contract_sha256(curriculum_config),
        },
        "holdout": loaded.manifest.holdout.to_mapping(),
        "fixed_direction": bool(
            getattr(racket_cfg, "action_ball_fixed_direction", False)
        ),
        "initial_episode_length_randomization": False,
        "policy_contract_sha256": str(
            getattr(
                racket_cfg, "action_ball_policy_contract_sha256", ""
            )
            or ""
        ),
        "evaluator_launch": {
            "path": str(
                getattr(
                    racket_cfg,
                    "action_ball_evaluator_launch_receipt_path",
                    "",
                )
                or ""
            ),
            "file_sha256": str(
                getattr(
                    racket_cfg,
                    "action_ball_evaluator_launch_receipt_file_sha256",
                    "",
                )
                or ""
            ),
        },
        "sidecar_launch": {
            "path": str(
                getattr(
                    racket_cfg,
                    "action_ball_sidecar_launch_receipt_path",
                    "",
                )
                or ""
            ),
            "file_sha256": str(
                getattr(
                    racket_cfg,
                    "action_ball_sidecar_launch_receipt_file_sha256",
                    "",
                )
                or ""
            ),
        },
        "drain_reset_launch": {
            "path": str(
                getattr(
                    racket_cfg,
                    "action_ball_drain_reset_launch_receipt_path",
                    "",
                )
                or ""
            ),
            "file_sha256": str(
                getattr(
                    racket_cfg,
                    "action_ball_drain_reset_launch_receipt_file_sha256",
                    "",
                )
                or ""
            ),
        },
        "evaluation_inbox": {
            "root": str(
                getattr(
                    racket_cfg,
                    "action_ball_evaluation_inbox_root",
                    "",
                )
                or ""
            ),
            "owner_id": str(
                getattr(
                    racket_cfg,
                    "action_ball_evaluation_owner_id",
                    "",
                )
                or ""
            ),
            "run_id": str(
                getattr(
                    racket_cfg,
                    "action_ball_evaluation_run_id",
                    "",
                )
                or ""
            ),
            "interval_updates": getattr(
                racket_cfg,
                "action_ball_frozen_eval_interval_updates",
                None,
            ),
        },
    }
    contract["sha256"] = _canonical_contract_sha256(contract)
    return contract


def _task_first_require_zero(value, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _OverrideError(
            f"[train.py] task-first requires {name}=0, got {value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric != 0.0:
        raise _OverrideError(
            f"[train.py] task-first requires {name}=0, got {value!r}"
        )


def _task_first_reward_terms(rewards):
    """Yield composed reward terms without depending on Isaac configclass."""

    if isinstance(rewards, dict):
        names = set(rewards)
        getter = rewards.get
    else:
        names = set(vars(rewards))
        for cls in type(rewards).__mro__:
            names.update(getattr(cls, "__annotations__", ()))
        getter = lambda name: getattr(rewards, name, None)
    for name in sorted(str(value) for value in names if not str(value).startswith("_")):
        yield name, getter(name)


def _task_first_reward_func_identity(term) -> str:
    if isinstance(term, dict):
        func = term.get("func")
    else:
        func = getattr(term, "func", None)
    if isinstance(func, str):
        return func.lower()
    return (
        f"{getattr(func, '__module__', '')}."
        f"{getattr(func, '__qualname__', getattr(func, '__name__', ''))}"
    ).lower()


def _task_first_term_func(term):
    if isinstance(term, dict):
        return term.get("func")
    return getattr(term, "func", None)


def _task_first_term_time_out(term):
    if isinstance(term, dict):
        return term.get("time_out", False)
    return getattr(term, "time_out", False)


def _task_first_require_authoritative_func(
    term, expected_name: str, term_name: str
) -> None:
    """Reject name-substring lookalikes while keeping host fixtures lightweight."""

    actual = _task_first_term_func(term)
    if isinstance(actual, str):
        if actual != expected_name:
            raise _OverrideError(
                f"[train.py] task-first terminations.{term_name} must name "
                f"exact function {expected_name!r}, got {actual!r}"
            )
        return
    try:
        from whole_body_tracking.tasks.tracking import mdp

        expected = getattr(mdp, expected_name)
    except (ImportError, AttributeError) as exc:
        raise _OverrideError(
            f"[train.py] cannot resolve authoritative termination {expected_name!r}"
        ) from exc
    if actual is not expected:
        raise _OverrideError(
            f"[train.py] task-first terminations.{term_name} must use the "
            f"authoritative callable object mdp.{expected_name}"
        )


def _validate_task_first_reward_semantics(env_cfg) -> None:
    """Reject reward terms whose truth source does not exist in task-first."""

    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        raise _OverrideError(
            "[train.py] task-first requires a composed rewards cfg for outcome "
            "and absolute-anchor validation"
        )
    forbidden = []
    for name, term in _task_first_reward_terms(rewards):
        if term is None:
            continue
        lowered = name.lower()
        identity = _task_first_reward_func_identity(term)
        semantic = f"{lowered} {identity}"
        ball_outcome = (
            lowered.startswith("virtual_")
            or lowered == "strike_capture_bonus"
            or "incoming_ball" in semantic
            or "pass_net" in semantic
            or "virtual_return" in semantic
            or "ball_outcome" in semantic
            or "analytic_outcome" in semantic
            or "return_outcome" in semantic
            or ("landing" in semantic and not lowered.startswith("foot_"))
        )
        absolute_anchor_xy = (
            "motion_global_anchor_pos" in semantic
            or "motion_global_anchor_position" in semantic
            or (
                "anchor" in semantic
                and "absolute" in semantic
                and ("_xy" in semantic or "position" in semantic)
            )
            or (
                "world_xy" in semantic
                and ("imitation" in semantic or "anchor" in semantic)
            )
        )
        if not (ball_outcome or absolute_anchor_xy):
            continue
        weight = term.get("weight") if isinstance(term, dict) else getattr(term, "weight", None)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
        ):
            raise _OverrideError(
                "[train.py] task-first forbidden reward term "
                f"{name!r} must be None or carry an explicit finite zero weight; "
                f"got weight={weight!r}"
            )
        if float(weight) != 0.0:
            reason = (
                "incoming-ball/landing/net/analytic outcome"
                if ball_outcome
                else "absolute world-XY anchor imitation"
            )
            forbidden.append(f"{name}(weight={float(weight)!r},{reason})")
    if forbidden:
        raise _OverrideError(
            "[train.py] task-first has no ball outcome truth and moves the station "
            "independently of the reference world XY; disable these reward terms "
            "(set None or exact weight 0 so RewardManager will not execute them): "
            + ", ".join(forbidden)
        )


def _task_first_term_params(term):
    if isinstance(term, dict):
        return term.get("params")
    return getattr(term, "params", None)


def _task_first_scene_entity_name(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name", "") or "")
    return str(getattr(value, "name", "") or "")


def _task_first_scene_entity_body_names(value) -> tuple[str, ...]:
    if isinstance(value, dict):
        names = value.get("body_names")
    else:
        names = getattr(value, "body_names", None)
    if names is None:
        return ()
    if isinstance(names, str):
        return (names,)
    try:
        return tuple(str(name) for name in names)
    except TypeError:
        return ()


def _validate_task_first_safety_semantics(env_cfg) -> None:
    """Require the three unsafe truth channels used by curriculum promotion."""

    if type(getattr(env_cfg, "table_obstacle", None)) is not bool or not env_cfg.table_obstacle:
        raise _OverrideError(
            "[train.py] task-first requires table_obstacle=true; otherwise "
            "table unsafe evidence is structurally pinned to zero"
        )
    table_prim = str(getattr(env_cfg, "table_obstacle_prim", "") or "").strip()
    if not table_prim:
        raise _OverrideError(
            "[train.py] task-first requires a concrete table_obstacle_prim"
        )
    scene = getattr(env_cfg, "scene", None)
    filtered_sensor = (
        None if scene is None else getattr(scene, "racket_table_contact", None)
    )
    if filtered_sensor is None:
        raise _OverrideError(
            "[train.py] task-first requires the filtered racket_table_contact "
            "sensor bound to the table collider"
        )
    if str(getattr(filtered_sensor, "prim_path", "") or "") != (
        "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link"
    ):
        raise _OverrideError(
            "[train.py] task-first racket_table_contact must watch the exact "
            "right_wrist_yaw_Link that carries the merged paddle collider"
        )
    filter_prims = tuple(
        str(value)
        for value in (
            getattr(filtered_sensor, "filter_prim_paths_expr", ()) or ()
        )
    )
    racket_cfg = getattr(
        getattr(env_cfg, "commands", None), "racket_target", None
    )
    action_ball = (
        str(getattr(racket_cfg, "target_mode", "") or "") == "action_ball"
    )
    configured_table_prims = tuple(
        str(value)
        for value in (getattr(env_cfg, "table_obstacle_prims", ()) or ())
    )
    if action_ball:
        expected_filter_prims = (
            "{ENV_REGEX_NS}/TableObstacle",
            "{ENV_REGEX_NS}/TableRobotKeepout",
            "{ENV_REGEX_NS}/TableNet",
            "{ENV_REGEX_NS}/TableNetPostLeft",
            "{ENV_REGEX_NS}/TableNetPostRight",
        )
    else:
        expected_filter_prims = (table_prim,)
    if (
        configured_table_prims != expected_filter_prims
        or filter_prims != expected_filter_prims
    ):
        raise _OverrideError(
            "[train.py] task-first racket_table_contact must filter exactly "
            "the configured ordered table assembly; "
            f"expected={expected_filter_prims!r} "
            f"env={configured_table_prims!r} sensor={filter_prims!r}"
        )
    table_asset = None if scene is None else getattr(scene, "table_obstacle", None)
    if (
        table_prim != "{ENV_REGEX_NS}/TableObstacle"
        or table_asset is None
        or str(getattr(table_asset, "prim_path", "") or "") != table_prim
    ):
        raise _OverrideError(
            "[train.py] task-first requires the exact solid "
            "{ENV_REGEX_NS}/TableObstacle asset"
        )
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        raise _OverrideError("[train.py] task-first requires terminations")
    missing = [
        name
        for name in ("base_fell_tilt", "base_too_low", "robot_hit_table")
        if getattr(terminations, name, None) is None
    ]
    if missing:
        raise _OverrideError(
            "[train.py] task-first requires all unsafe termination terms active: "
            + ", ".join(missing)
        )
    for term_name, expected_func, threshold_name, expected_threshold in (
        ("base_fell_tilt", "bad_orientation", "limit_angle", 0.7),
        (
            "base_too_low",
            "root_height_below_minimum",
            "minimum_height",
            0.5,
        ),
    ):
        term = getattr(terminations, term_name)
        _task_first_require_authoritative_func(
            term, expected_func, term_name
        )
        if _task_first_term_time_out(term) is not False:
            raise _OverrideError(
                f"[train.py] task-first terminations.{term_name}.time_out "
                "must be false so the event is counted as unsafe"
            )
        guard_params = _task_first_term_params(term)
        if (
            not isinstance(guard_params, dict)
            or set(guard_params) != {threshold_name}
            or isinstance(guard_params.get(threshold_name), bool)
            or not isinstance(
                guard_params.get(threshold_name), (int, float)
            )
            or not math.isfinite(float(guard_params[threshold_name]))
            or float(guard_params[threshold_name]) != expected_threshold
        ):
            raise _OverrideError(
                f"[train.py] task-first terminations.{term_name} requires "
                f"{threshold_name}={expected_threshold}"
            )
    table_term = getattr(terminations, "robot_hit_table")
    _task_first_require_authoritative_func(
        table_term, "robot_hit_table", "robot_hit_table"
    )
    if _task_first_term_time_out(table_term) is not False:
        raise _OverrideError(
            "[train.py] task-first terminations.robot_hit_table.time_out "
            "must be false so table strikes are counted as unsafe"
        )
    params = _task_first_term_params(table_term)
    expected_table_param_keys = {
        "sensor_cfg",
        "filtered_sensor_cfg",
        "all_body_filtered_sensor_cfgs",
        "expected_full_table_filter_prim_paths",
        "asset_cfg",
        "near_x",
        "surface_z",
        "force_threshold",
        "margin",
        "full_table_assembly",
        "keepout_floor_z",
        "action_name",
        "require_substep_latch",
    }
    if (
        not isinstance(params, dict)
        or set(params) != expected_table_param_keys
    ):
        raise _OverrideError(
            "[train.py] task-first robot_hit_table term requires exactly "
            f"{sorted(expected_table_param_keys)!r}"
        )
    if _task_first_scene_entity_name(
        params.get("filtered_sensor_cfg")
    ) != "racket_table_contact":
        raise _OverrideError(
            "[train.py] task-first robot_hit_table must consume the filtered "
            "racket_table_contact sensor"
        )
    exact_filtered_cfgs = params.get("all_body_filtered_sensor_cfgs")
    if not isinstance(exact_filtered_cfgs, (tuple, list)):
        raise _OverrideError(
            "[train.py] task-first robot_hit_table requires an ordered "
            "all_body_filtered_sensor_cfgs sequence"
        )
    exact_filtered_names = tuple(
        _task_first_scene_entity_name(value)
        for value in exact_filtered_cfgs
    )
    expected_filter_binding = params.get(
        "expected_full_table_filter_prim_paths"
    )
    if not isinstance(expected_filter_binding, (tuple, list)):
        raise _OverrideError(
            "[train.py] task-first robot_hit_table requires an ordered "
            "expected_full_table_filter_prim_paths sequence"
        )
    expected_filter_binding = tuple(
        str(value) for value in expected_filter_binding
    )
    broad_regex = (
        r"^(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$).+$"
    )
    if (
        _task_first_scene_entity_name(params.get("sensor_cfg"))
        != "contact_forces"
        or _task_first_scene_entity_name(params.get("asset_cfg")) != "robot"
        or _task_first_scene_entity_body_names(params.get("sensor_cfg"))
        != (broad_regex,)
        or _task_first_scene_entity_body_names(params.get("asset_cfg"))
        != (broad_regex,)
    ):
        raise _OverrideError(
            "[train.py] task-first robot_hit_table broad channel must align "
            "the exact non-foot contact_forces and robot body selections"
        )
    for name in ("near_x", "surface_z", "force_threshold", "margin"):
        value = params.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise _OverrideError(
                f"[train.py] task-first robot_hit_table.{name} must be finite"
            )
    expected_near_x = getattr(racket_cfg, "vb_table_near_x", None)
    expected_surface_z = getattr(racket_cfg, "vb_table_surface_z", None)
    if (
        isinstance(expected_near_x, bool)
        or not isinstance(expected_near_x, (int, float))
        or isinstance(expected_surface_z, bool)
        or not isinstance(expected_surface_z, (int, float))
        or float(params["near_x"]) != float(expected_near_x)
        or float(params["surface_z"]) != float(expected_surface_z)
    ):
        raise _OverrideError(
            "[train.py] task-first robot_hit_table near_x/surface_z must "
            "exactly match the live racket table geometry"
        )
    if (
        float(params["force_threshold"]) != 1.0e-6
        or float(params["margin"]) != 0.02
        or params["full_table_assembly"] is not action_ball
        or isinstance(params["keepout_floor_z"], bool)
        or not isinstance(params["keepout_floor_z"], (int, float))
        or not math.isfinite(float(params["keepout_floor_z"]))
        or float(params["keepout_floor_z"]) != 0.0
        or params["action_name"] != "joint_pos"
        or params["require_substep_latch"] is not action_ball
    ):
        raise _OverrideError(
            "[train.py] task-first robot_hit_table requires the reviewed "
            "force/margin/assembly/substep-latch contract"
        )
    expected_center = (
        float(expected_near_x) + 1.37,
        0.0,
        float(expected_surface_z) - 0.025,
    )
    init_state = getattr(table_asset, "init_state", None)
    spawn = getattr(table_asset, "spawn", None)
    collision_props = getattr(spawn, "collision_props", None)
    try:
        actual_center = tuple(
            float(value) for value in getattr(init_state, "pos")
        )
        actual_size = tuple(float(value) for value in getattr(spawn, "size"))
    except (TypeError, ValueError):
        actual_center = ()
        actual_size = ()
    if (
        actual_center != expected_center
        or actual_size != (2.74, 1.525, 0.05)
        or getattr(collision_props, "collision_enabled", None) is not True
    ):
        raise _OverrideError(
            "[train.py] task-first table collider pose/size/collision flag "
            "does not match the reviewed live table geometry"
        )
    if action_ball:
        from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
            TABLE_ALL_BODY_CONTACT_SENSOR_NAMES as _table_sensor_names,
            TABLE_CONTACT_BODY_NAMES as _table_body_names,
        )

        if getattr(env_cfg, "table_robot_keepout", None) is not True:
            raise _OverrideError(
                "[train.py] action-ball requires the conservative robot-only "
                "under-table keepout"
            )
        if int(getattr(env_cfg, "decimation", -1)) != 4:
            raise _OverrideError(
                "[train.py] action-ball table-contact latch requires decimation=4"
            )
        if (
            tuple(
                getattr(env_cfg, "table_pair_contact_sensor_names", ()) or ()
            )
            != tuple(_table_sensor_names)
            or exact_filtered_names != tuple(_table_sensor_names)
            or expected_filter_binding != expected_filter_prims
            or len(_table_sensor_names) != len(_table_body_names)
            or len(_table_body_names) != 32
        ):
            raise _OverrideError(
                "[train.py] action-ball requires the exact ordered 32-body "
                "pair-filter sensor table and five-part filter binding"
            )
        for index, (sensor_name, body_name) in enumerate(
            zip(_table_sensor_names, _table_body_names)
        ):
            sensor_cfg = (
                None if scene is None else getattr(scene, sensor_name, None)
            )
            if sensor_cfg is None:
                raise _OverrideError(
                    "[train.py] action-ball exact table-contact sensor is "
                    f"missing at index {index}: {sensor_name!r}"
                )
            if (
                str(getattr(sensor_cfg, "prim_path", "") or "")
                != f"{{ENV_REGEX_NS}}/Robot/{body_name}"
                or tuple(
                    str(value)
                    for value in (
                        getattr(
                            sensor_cfg,
                            "filter_prim_paths_expr",
                            (),
                        )
                        or ()
                    )
                )
                != expected_filter_prims
                or float(getattr(sensor_cfg, "update_period", math.nan))
                != 0.0
            ):
                raise _OverrideError(
                    "[train.py] action-ball exact table-contact sensor "
                    f"{sensor_name!r} does not bind body {body_name!r}, the "
                    "five-part table assembly, and every-physics-step updates"
                )
        action_cfg = getattr(
            getattr(env_cfg, "actions", None), "joint_pos", None
        )
        if (
            action_cfg is None
            or getattr(action_cfg, "table_contact_substep_guard", None)
            is not True
            or str(
                getattr(
                    action_cfg,
                    "table_contact_guard_termination_term",
                    "",
                )
                or ""
            )
            != "robot_hit_table"
            or int(
                getattr(
                    action_cfg,
                    "table_contact_guard_expected_decimation",
                    -1,
                )
            )
            != 4
        ):
            raise _OverrideError(
                "[train.py] action-ball requires the reviewed four-substep "
                "sticky table-contact action guard"
            )

        from whole_body_tracking.tasks.table_tennis import geometry as _tt_geom
        from whole_body_tracking.tasks.table_tennis import table_frame as _tt_frame
        from whole_body_tracking.tasks.table_tennis import (
            table_tennis_env_cfg as _tt_cfg,
        )

        underside_z = float(expected_surface_z) - float(
            _tt_geom.TABLE_THICKNESS
        )
        top_center = _tt_frame.table_top_center_env(
            float(expected_near_x),
            float(expected_surface_z),
        )
        expected_assets = (
            (
                "table_robot_keepout",
                expected_filter_prims[1],
                (top_center[0], top_center[1], underside_z / 2.0),
                (
                    float(_tt_geom.TABLE_LENGTH),
                    float(_tt_geom.TABLE_WIDTH),
                    underside_z,
                ),
            ),
            (
                "table_net",
                expected_filter_prims[2],
                _tt_frame.net_center_env(
                    float(expected_near_x),
                    float(expected_surface_z),
                ),
                tuple(float(value) for value in _tt_geom.net_size()),
            ),
            (
                "table_net_post_left",
                expected_filter_prims[3],
                _tt_frame.net_post_center_env(
                    float(expected_near_x),
                    float(expected_surface_z),
                    left=True,
                    post_height=_tt_cfg.NET_POST_HEIGHT,
                ),
                tuple(float(value) for value in _tt_cfg.net_post_size()),
            ),
            (
                "table_net_post_right",
                expected_filter_prims[4],
                _tt_frame.net_post_center_env(
                    float(expected_near_x),
                    float(expected_surface_z),
                    left=False,
                    post_height=_tt_cfg.NET_POST_HEIGHT,
                ),
                tuple(float(value) for value in _tt_cfg.net_post_size()),
            ),
        )
        for attr, expected_prim, expected_pos, expected_size in expected_assets:
            asset = getattr(scene, attr, None)
            init_state = None if asset is None else getattr(asset, "init_state", None)
            spawn = None if asset is None else getattr(asset, "spawn", None)
            collision_props = (
                None if spawn is None else getattr(spawn, "collision_props", None)
            )
            try:
                actual_pos = tuple(
                    float(value) for value in getattr(init_state, "pos")
                )
                actual_size = tuple(
                    float(value) for value in getattr(spawn, "size")
                )
            except (TypeError, ValueError):
                actual_pos = ()
                actual_size = ()
            if (
                asset is None
                or str(getattr(asset, "prim_path", "") or "")
                != expected_prim
                or actual_pos != tuple(float(value) for value in expected_pos)
                or actual_size != expected_size
                or getattr(collision_props, "collision_enabled", None)
                is not True
            ):
                raise _OverrideError(
                    "[train.py] action-ball table assembly asset does not "
                    f"match the reviewed contract: {attr}"
                )


def _finalize_task_first_training_cfg(env_cfg, task, applied) -> None:
    """Fail closed and append the two task-first actor terms in canonical order.

    This runs after every task override but before ``gym.make``.  In particular,
    the generic face-observation branch deliberately defers its append in
    task-first mode so this function can prove the Hitter-footwork prefix is
    untouched and then append ``face(4), action(N)`` as one atomic layout.
    """

    commands = getattr(env_cfg, "commands", None)
    racket_cfg = None if commands is None else getattr(commands, "racket_target", None)
    if racket_cfg is None or str(getattr(racket_cfg, "target_mode", "")) != "task_first":
        return

    motion_cfg = getattr(commands, "motion", None)
    if motion_cfg is None:
        raise _OverrideError(
            "[train.py] task-first requires commands.motion"
        )
    racket_node = _get(task, "racket")
    raw_clip_names = _get(racket_node, "clip_names")
    if raw_clip_names is None:
        raise _OverrideError(
            "[train.py] task-first requires an explicit non-empty racket.clip_names list"
        )
    clip_names = _resolve_clip_names(racket_node)
    action_count = len(clip_names)
    if action_count < 1:
        raise _OverrideError("[train.py] task-first requires at least one action")

    loaded = _load_task_first_manifest_from_racket_cfg(racket_cfg)
    if tuple(loaded.manifest.action_order) != tuple(clip_names):
        raise _OverrideError(
            "[train.py] task-first racket.clip_names must exactly equal the manifest "
            f"action_order: clip_names={tuple(clip_names)!r} "
            f"manifest={tuple(loaded.manifest.action_order)!r}"
        )

    expected_actor_contract = f"task_first_n{action_count}"
    configured_actor_contract = _get(task, "actor_obs_contract")
    if str(configured_actor_contract or "") != expected_actor_contract:
        raise _OverrideError(
            "[train.py] task-first actor_obs_contract must be exactly "
            f"{expected_actor_contract!r}; got {configured_actor_contract!r}. "
            "hitter_footwork omits the demanded-face and per-action identity tail."
        )
    if str(getattr(env_cfg, "obs_mode", "")) != "hitter_footwork":
        raise _OverrideError(
            "[train.py] task-first requires obs_mode='hitter_footwork', got "
            f"{getattr(env_cfg, 'obs_mode', None)!r}"
        )
    if not bool(getattr(racket_cfg, "face_command", False)):
        raise _OverrideError("[train.py] task-first requires racket.face_command=true")
    if str(getattr(racket_cfg, "face_command_pairing", "")) != "shared_plus_y":
        raise _OverrideError(
            "[train.py] task-first requires "
            "racket.face_command_pairing='shared_plus_y'"
        )
    if not bool(getattr(env_cfg, "face_command_obs", False)):
        raise _OverrideError("[train.py] task-first requires racket.face_command_obs=true")
    if bool(getattr(env_cfg, "station_obs", False)):
        raise _OverrideError(
            "[train.py] task-first uses Hitter's native base_target_pos_b channel; "
            "the legacy station_obs tail must be disabled"
        )

    _task_first_require_zero(
        getattr(racket_cfg, "achieved_target_mix_prob", None),
        "racket.achieved_target_mix_prob",
    )
    _task_first_require_zero(
        getattr(racket_cfg, "midswing_resample_prob", None),
        "racket.midswing_resample_prob",
    )
    for attr, label in (
        ("target_delay_steps", "racket.target_delay_steps"),
        ("target_jitter_pos_per_s", "racket.target_jitter_pos_per_s"),
        ("target_jitter_vel_per_s", "racket.target_jitter_vel_per_s"),
        ("target_noise_white", "racket.target_noise_white"),
        ("target_noise_ar1_sigma", "racket.target_noise_ar1_sigma"),
        ("target_dropout_prob", "racket.target_dropout_prob"),
        ("target_post_strike_dropout_s", "racket.target_post_strike_dropout_s"),
        ("target_bias_per_swing", "racket.target_bias_per_swing"),
    ):
        _task_first_require_zero(getattr(racket_cfg, attr, None), label)
    if str(getattr(racket_cfg, "target_delay_tts_mode", "")) != "live":
        raise _OverrideError(
            "[train.py] task-first requires racket.target_delay_tts_mode='live'"
        )

    artifact_fields = {
        name: str(getattr(racket_cfg, name, "") or "").strip()
        for name in ("question_bank", "cq_anchor_bank", "exam_bank")
    }
    explicit_cq_keys = []
    try:
        racket_keys = list(racket_node.keys())
    except Exception:
        racket_keys = []
    for raw_key in racket_keys:
        key = str(raw_key)
        value = _get(racket_node, raw_key)
        if key.startswith("cq_") and value not in (None, ""):
            explicit_cq_keys.append(key)
    if (
        any(artifact_fields.values())
        or explicit_cq_keys
        or bool(getattr(racket_cfg, "question_bank_allow_legacy", False))
    ):
        raise _OverrideError(
            "[train.py] task-first owns task production and requires question/CQ "
            "configuration empty, got "
            f"artifacts={artifact_fields} explicit_cq_keys={sorted(explicit_cq_keys)} "
            "question_bank_allow_legacy="
            f"{bool(getattr(racket_cfg, 'question_bank_allow_legacy', False))}"
        )
    enabled_ball_paths = [
        name
        for name in (
            "virtual_ball",
            "vb_metrics_only",
            "shadow_ball",
            "shadow_table",
            "physical_ball",
        )
        if bool(getattr(racket_cfg, name, False))
    ]
    if bool(getattr(env_cfg, "physical_ball", False)):
        enabled_ball_paths.append("env.physical_ball")
    if enabled_ball_paths:
        raise _OverrideError(
            "[train.py] task-first executor training is ball-free; disable "
            + ", ".join(enabled_ball_paths)
        )
    _validate_task_first_safety_semantics(env_cfg)
    _validate_task_first_reward_semantics(env_cfg)

    if bool(getattr(racket_cfg, "planner_revision_enabled", False)) or bool(
        getattr(motion_cfg, "planner_revision_enabled", False)
    ):
        raise _OverrideError(
            "[train.py] task-first requires planner_revision disabled"
        )
    if bool(getattr(motion_cfg, "balanced_clip_sampling", False)) is not True:
        raise _OverrideError(
            "[train.py] task-first requires motion.balanced_clip_sampling=true"
        )
    balanced_seed = getattr(motion_cfg, "balanced_clip_sampling_seed", None)
    if (
        type(balanced_seed) is not int
        or not 0 <= balanced_seed < (1 << 63)
    ):
        raise _OverrideError(
            "[train.py] task-first requires "
            "motion.balanced_clip_sampling_seed in [0,2**63)"
        )
    _task_first_require_zero(
        getattr(motion_cfg, "clip_switch_prob", None),
        "motion.clip_switch_prob",
    )
    speed_range = tuple(
        float(value)
        for value in (getattr(motion_cfg, "speed_scale_range", ()) or ())
    )
    if speed_range != (1.0, 1.0):
        raise _OverrideError(
            "[train.py] task-first requires motion.speed_scale_range=[1.0,1.0], "
            f"got {speed_range!r}"
        )
    speed_per_clip = getattr(motion_cfg, "speed_scale_per_clip", None)
    if speed_per_clip is not None and any(
        float(value) != 1.0 for value in speed_per_clip
    ):
        raise _OverrideError(
            "[train.py] task-first requires motion.speed_scale_per_clip absent or all 1.0"
        )
    if str(getattr(motion_cfg, "event_timing_mode", "")) != "disabled":
        raise _OverrideError(
            "[train.py] task-first requires motion.event_timing_mode='disabled'"
        )
    base_threshold = getattr(
        racket_cfg, "task_first_base_success_thresh_m", None
    )
    if (
        isinstance(base_threshold, bool)
        or not isinstance(base_threshold, (int, float))
        or not math.isfinite(float(base_threshold))
        or float(base_threshold) <= 0.0
        or float(base_threshold) > 0.10
    ):
        raise _OverrideError(
            "[train.py] task-first requires finite "
            "0 < racket.task_first_base_success_thresh_m <= 0.10"
        )
    for attr, maximum in (
        ("strike_success_pos_thresh", 0.075),
        ("strike_success_vel_thresh", 0.5),
        ("strike_success_normal_thresh_deg", 15.0),
    ):
        value = getattr(racket_cfg, attr, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or float(value) > maximum
        ):
            raise _OverrideError(
                f"[train.py] task-first requires 0 < racket.{attr} <= {maximum}"
            )
    if getattr(racket_cfg, "clean_reference_strike_velocity", None) is not True:
        raise _OverrideError(
            "[train.py] task-first requires "
            "racket.clean_reference_strike_velocity=true"
        )
    clean_window = getattr(racket_cfg, "clean_strike_vel_window", None)
    if (
        type(clean_window) is not int
        or clean_window < 1
        or clean_window > 10
    ):
        raise _OverrideError(
            "[train.py] task-first requires integer "
            "racket.clean_strike_vel_window in [1,10]"
        )
    if str(getattr(racket_cfg, "racket_body_name", "") or "") != (
        "pingpang_red_Link"
    ):
        raise _OverrideError(
            "[train.py] task-first requires "
            "racket.racket_body_name='pingpang_red_Link'"
        )
    if str(getattr(racket_cfg, "wrist_body_name", "") or "") != (
        "right_wrist_yaw_Link"
    ):
        raise _OverrideError(
            "[train.py] task-first requires "
            "racket.wrist_body_name='right_wrist_yaw_Link'"
        )
    for attr, expected in (
        ("mount_offset", (0.21021, 0.032078, 0.032036)),
        ("mount_quat", (1.0, 0.0, 0.0, 0.0)),
    ):
        value = getattr(racket_cfg, attr, None)
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, (list, tuple))
            or len(value) != len(expected)
            or any(
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
                for component in value
            )
            or tuple(float(component) for component in value) != expected
        ):
            raise _OverrideError(
                f"[train.py] task-first racket.{attr} must equal the reviewed "
                f"physical paddle-site transform {expected!r}"
            )
    mount_axis = getattr(racket_cfg, "mount_normal_axis", None)
    if type(mount_axis) is not int or mount_axis != 1:
        raise _OverrideError(
            "[train.py] task-first racket.mount_normal_axis must be 1 "
            "(physical paddle-local +Y)"
        )

    policy = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy is None:
        raise _OverrideError("[train.py] task-first requires observations.policy")
    occupied = [
        name
        for name in (
            "racket_target_normal_cmd",
            "action_one_hot",
            "station_anchor_err_b",
        )
        if getattr(policy, name, None) is not None
    ]
    if occupied:
        raise _OverrideError(
            "[train.py] task-first requires a clean Hitter-footwork policy prefix; "
            f"term(s) already attached: {occupied}"
        )

    from isaaclab.managers import ObservationTermCfg as _ObsTerm

    from whole_body_tracking.tasks.tracking import mdp as _mdp

    policy.racket_target_normal_cmd = _ObsTerm(
        func=_mdp.racket_target_normal_cmd,
        params={"command_name": "racket_target"},
    )
    policy.action_one_hot = _ObsTerm(
        func=_mdp.action_one_hot,
        params={
            "command_name": "racket_target",
            "expected_actions": action_count,
        },
    )
    applied.append(
        "observations.policy task-first tail="
        f"racket_target_normal_cmd(+4),action_one_hot(+{action_count}) "
        f"(actor_obs_contract={expected_actor_contract})"
    )


def _finalize_action_ball_training_cfg(env_cfg, task, applied) -> None:
    """Fail closed on the action -> ball -> solved-task training recipe."""

    commands = getattr(env_cfg, "commands", None)
    racket_cfg = (
        None if commands is None else getattr(commands, "racket_target", None)
    )
    if (
        racket_cfg is None
        or str(getattr(racket_cfg, "target_mode", "")) != "action_ball"
    ):
        return
    motion_cfg = getattr(commands, "motion", None)
    if motion_cfg is None:
        raise _OverrideError(
            "[train.py] action-ball requires commands.motion"
        )
    racket_node = _get(task, "racket")
    if _get(racket_node, "clip_names") is None:
        raise _OverrideError(
            "[train.py] action-ball requires an explicit non-empty "
            "racket.clip_names list"
        )
    clip_names = _resolve_clip_names(racket_node)
    if not clip_names:
        raise _OverrideError(
            "[train.py] action-ball requires at least one action"
        )

    loaded = _load_action_ball_manifest_from_cfg(racket_cfg, motion_cfg)
    manifest = loaded.manifest
    if tuple(manifest.action_order) != tuple(clip_names):
        raise _OverrideError(
            "[train.py] action-ball racket.clip_names must exactly equal the "
            "manifest action_order: "
            f"clip_names={tuple(clip_names)!r} "
            f"manifest={tuple(manifest.action_order)!r}"
        )
    manifest_scope = str(manifest.prototype.scope)
    if manifest_scope not in ("upper", "full"):
        raise _OverrideError(
            "[train.py] action-ball manifest prototype scope must be "
            "'upper' or 'full'"
        )
    reward_node = _get(task, "rewards")
    configured_full_body = _get(reward_node, "full_body_mimic")
    if configured_full_body is None:
        raise _OverrideError(
            "[train.py] action-ball requires an explicit launcher-owned "
            "rewards.full_body_mimic value"
        )
    configured_full_body = _as_explicit_bool(
        configured_full_body,
        "task.rewards.full_body_mimic",
    )
    expected_full_body = manifest_scope == "full"
    if configured_full_body is not expected_full_body:
        raise _OverrideError(
            "[train.py] action-ball full-body imitation must be derived from "
            "the exact manifest prototype scope: "
            f"scope={manifest_scope!r}, "
            f"full_body_mimic={configured_full_body!r}"
        )
    action_count = len(clip_names)
    if action_count > 1024:
        raise _OverrideError(
            "[train.py] action-ball supports at most 1024 actions because "
            "the dynamic actor observation contracts are bounded"
        )
    configured_actor_contract = str(
        _get(task, "actor_obs_contract") or ""
    )
    legacy_actor_contract = f"action_ball_n{action_count}"
    table_pose_actor_contract = (
        f"action_ball_table_pose_n{action_count}"
    )
    table_pose_twist_actor_contract = (
        f"action_ball_table_pose_twist_n{action_count}"
    )
    table_pose_twist_heading_task_actor_contract = (
        "action_ball_table_pose_twist_heading_task_n"
        f"{action_count}"
    )
    teacher_start_actor_contract = (
        "action_ball_table_pose_twist_heading_task_teacher_start_n"
        f"{action_count}"
    )
    if configured_actor_contract not in (
        legacy_actor_contract,
        table_pose_actor_contract,
        table_pose_twist_actor_contract,
        table_pose_twist_heading_task_actor_contract,
        teacher_start_actor_contract,
    ):
        raise _OverrideError(
            "[train.py] action-ball actor_obs_contract must match the exact "
            "action count: expected "
            f"{teacher_start_actor_contract!r} "
            f"(preferred), or compatibility contracts "
            f"{table_pose_twist_heading_task_actor_contract!r}/"
            f"{table_pose_twist_actor_contract!r}/"
            f"{table_pose_actor_contract!r}/{legacy_actor_contract!r}; got "
            f"{configured_actor_contract!r}"
        )
    include_table_pose = (
        configured_actor_contract
        in (
            table_pose_actor_contract,
            table_pose_twist_actor_contract,
            table_pose_twist_heading_task_actor_contract,
            teacher_start_actor_contract,
        )
    )
    include_base_twist = (
        configured_actor_contract
        in (
            table_pose_twist_actor_contract,
            table_pose_twist_heading_task_actor_contract,
            teacher_start_actor_contract,
        )
    )
    include_heading_task = (
        configured_actor_contract in (
            table_pose_twist_heading_task_actor_contract,
            teacher_start_actor_contract,
        )
    )
    include_teacher_start = (
        configured_actor_contract == teacher_start_actor_contract
    )
    expected_actor_contract = configured_actor_contract
    if str(getattr(env_cfg, "obs_mode", "")) != "hitter_footwork":
        raise _OverrideError(
            "[train.py] action-ball requires obs_mode='hitter_footwork'"
        )
    if getattr(racket_cfg, "face_command", None) is not True:
        raise _OverrideError(
            "[train.py] action-ball requires racket.face_command=true"
        )
    if str(getattr(racket_cfg, "face_command_pairing", "")) != "shared_plus_y":
        raise _OverrideError(
            "[train.py] action-ball requires "
            "racket.face_command_pairing='shared_plus_y'"
        )
    if getattr(env_cfg, "face_command_obs", None) is not True:
        raise _OverrideError(
            "[train.py] action-ball requires racket.face_command_obs=true"
        )
    if bool(getattr(env_cfg, "station_obs", False)):
        raise _OverrideError(
            "[train.py] action-ball uses Hitter's native base target channel; "
            "the legacy station_obs tail must be disabled"
        )

    for attr, label in (
        ("achieved_target_mix_prob", "racket.achieved_target_mix_prob"),
        ("midswing_resample_prob", "racket.midswing_resample_prob"),
        ("target_delay_steps", "racket.target_delay_steps"),
        ("target_jitter_pos_per_s", "racket.target_jitter_pos_per_s"),
        ("target_jitter_vel_per_s", "racket.target_jitter_vel_per_s"),
        ("target_noise_white", "racket.target_noise_white"),
        ("target_noise_ar1_sigma", "racket.target_noise_ar1_sigma"),
        ("target_dropout_prob", "racket.target_dropout_prob"),
        (
            "target_post_strike_dropout_s",
            "racket.target_post_strike_dropout_s",
        ),
        ("target_bias_per_swing", "racket.target_bias_per_swing"),
    ):
        try:
            value = float(getattr(racket_cfg, attr, None))
        except (TypeError, ValueError) as exc:
            raise _OverrideError(
                f"[train.py] action-ball requires {label}=0"
            ) from exc
        if not math.isfinite(value) or value != 0.0:
            raise _OverrideError(
                f"[train.py] action-ball requires {label}=0, got {value!r}"
            )
    if str(getattr(racket_cfg, "target_delay_tts_mode", "")) != "live":
        raise _OverrideError(
            "[train.py] action-ball requires "
            "racket.target_delay_tts_mode='live'"
        )

    artifact_fields = {
        name: str(getattr(racket_cfg, name, "") or "").strip()
        for name in ("question_bank", "cq_anchor_bank", "exam_bank")
    }
    # The action-ball solver intentionally reuses a narrow set of reviewed
    # continuous-question numerical kernels.  Their executable knobs may be
    # explicitly pinned by YAML and are authenticated by the solver payload.
    # Legacy CQ producers/distributions/buffers remain forbidden.
    action_ball_solver_cq_keys = {
        "cq_overdraw",
        "cq_n_iters",
        "cq_tol_m",
        "cq_speed_budget",
        "cq_max_redraw_rounds",
    }
    explicit_legacy_cq_keys = []
    try:
        racket_keys = list(racket_node.keys())
    except Exception:
        racket_keys = []
    for raw_key in racket_keys:
        key = str(raw_key)
        if (
            key.startswith("cq_")
            and key not in action_ball_solver_cq_keys
            and _get(racket_node, raw_key) not in (None, "")
        ):
            explicit_legacy_cq_keys.append(key)
    if (
        any(artifact_fields.values())
        or explicit_legacy_cq_keys
        or bool(getattr(racket_cfg, "question_bank_allow_legacy", False))
    ):
        raise _OverrideError(
            "[train.py] action-ball owns ball/task production and requires "
            "question/legacy-CQ producer configuration empty, got "
            f"artifacts={artifact_fields} "
            "explicit_legacy_cq_keys="
            f"{sorted(explicit_legacy_cq_keys)}"
        )

    # The analytic ball is the authoritative sampled truth.  PhysX/shadow
    # instruments are permitted only when their duplicate cfg switches agree;
    # their runtime constructors remain responsible for rejecting an
    # unintegrated lifecycle rather than this preflight pretending parity.
    if getattr(racket_cfg, "virtual_ball", None) is not True:
        raise _OverrideError(
            "[train.py] action-ball requires racket.virtual_ball=true"
        )
    racket_physical_raw = getattr(racket_cfg, "physical_ball", False)
    env_physical_raw = getattr(env_cfg, "physical_ball", False)
    if type(racket_physical_raw) is not bool or type(env_physical_raw) is not bool:
        raise _OverrideError(
            "[train.py] action-ball physical-ball truth switches must be "
            "explicit booleans"
        )
    racket_physical = racket_physical_raw
    env_physical = env_physical_raw
    if racket_physical != env_physical:
        raise _OverrideError(
            "[train.py] action-ball physical-ball truth switches disagree: "
            f"racket={racket_physical} env={env_physical}"
        )
    shadow_ball = bool(getattr(racket_cfg, "shadow_ball", False))
    shadow_table = bool(getattr(racket_cfg, "shadow_table", False))
    if shadow_table and not shadow_ball:
        raise _OverrideError(
            "[train.py] action-ball shadow_table=true requires shadow_ball=true"
        )

    _validate_task_first_safety_semantics(env_cfg)
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        raise _OverrideError(
            "[train.py] action-ball requires a composed rewards cfg"
        )
    lower_body_names = (
        "pelvis_link",
        "left_hip_roll_Link",
        "left_knee_Link",
        "left_ankle_roll_Link",
        "right_hip_roll_Link",
        "right_knee_Link",
        "right_ankle_roll_Link",
    )
    for term_name in (
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ):
        term = getattr(rewards_cfg, term_name, None)
        body_names = (
            None if term is None else getattr(term, "params", {}).get("body_names")
        )
        if not isinstance(body_names, (list, tuple)):
            raise _OverrideError(
                "[train.py] action-ball scope validation requires "
                f"rewards.{term_name}.params.body_names"
            )
        body_names = tuple(str(name) for name in body_names)
        if expected_full_body:
            if (
                body_names[: len(lower_body_names)] != lower_body_names
                or any(body_names.count(name) != 1 for name in lower_body_names)
            ):
                raise _OverrideError(
                    "[train.py] full-scope ActionBall must include the exact "
                    "pelvis+six-leg prefix in every body-imitation term: "
                    f"term={term_name!r}, body_names={body_names!r}"
                )
        elif any(name in body_names for name in lower_body_names):
            raise _OverrideError(
                "[train.py] upper-scope ActionBall must not imitate lower-body "
                f"links: term={term_name!r}, body_names={body_names!r}"
            )
    applied.append(
        "action-ball imitation scope="
        f"{manifest_scope} (full_body_mimic={expected_full_body})"
    )
    if getattr(rewards_cfg, "motion_global_anchor_pos", None) is not None:
        raise _OverrideError(
            "[train.py] action-ball base-spawn variation requires "
            "rewards.motion_global_anchor_pos=None"
        )
    if bool(getattr(racket_cfg, "planner_revision_enabled", False)) or bool(
        getattr(motion_cfg, "planner_revision_enabled", False)
    ):
        raise _OverrideError(
            "[train.py] action-ball requires planner_revision disabled"
        )
    if getattr(motion_cfg, "canonical_ready_mode", None) is not True:
        raise _OverrideError(
            "[train.py] action-ball requires motion.canonical_ready_mode=true"
        )
    diagnostic_unauthorized = getattr(
        racket_cfg, "action_ball_diagnostic_unauthorized", False
    )
    if type(diagnostic_unauthorized) is not bool:
        raise _OverrideError(
            "[train.py] racket.action_ball_diagnostic_unauthorized must be "
            "an exact boolean"
        )
    if (
        not diagnostic_unauthorized
        and configured_actor_contract
        != table_pose_twist_heading_task_actor_contract
    ):
        raise _OverrideError(
            "[train.py] formal ActionBall requires the frame-consistent "
            f"{table_pose_twist_heading_task_actor_contract!r} actor "
            "observation contract; legacy layouts are diagnostic/read-only "
            "compatibility only"
        )
    if not diagnostic_unauthorized:
        for attr in (
            "canonical_registry_path",
            "canonical_promotion_certificate_path",
        ):
            if not str(getattr(motion_cfg, attr, "") or "").strip():
                raise _OverrideError(
                    f"[train.py] action-ball requires motion.{attr}"
                )
        for attr in (
            "canonical_registry_sha256",
            "canonical_registry_alignment_sha256",
            "canonical_ready_sha256",
            "canonical_ready_fk_sha256",
        ):
            digest = str(getattr(motion_cfg, attr, "") or "")
            if (
                len(digest) != 64
                or digest != digest.lower()
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise _OverrideError(
                    f"[train.py] action-ball motion.{attr} must be exactly "
                    "64 lowercase hexadecimal characters"
                )
    if getattr(motion_cfg, "wrap_teleport", None) is not False:
        raise _OverrideError(
            "[train.py] action-ball requires motion.wrap_teleport=false"
        )
    if getattr(motion_cfg, "balanced_clip_sampling", None) is not True:
        raise _OverrideError(
            "[train.py] action-ball requires "
            "motion.balanced_clip_sampling=true"
        )
    balanced_seed = getattr(motion_cfg, "balanced_clip_sampling_seed", None)
    if (
        type(balanced_seed) is not int
        or not 0 <= balanced_seed < (1 << 63)
    ):
        raise _OverrideError(
            "[train.py] action-ball requires "
            "motion.balanced_clip_sampling_seed in [0,2**63)"
        )
    try:
        clip_switch_prob = float(
            getattr(motion_cfg, "clip_switch_prob", None)
        )
    except (TypeError, ValueError) as exc:
        raise _OverrideError(
            "[train.py] action-ball requires motion.clip_switch_prob=0"
        ) from exc
    if not math.isfinite(clip_switch_prob) or clip_switch_prob != 0.0:
        raise _OverrideError(
            "[train.py] action-ball requires motion.clip_switch_prob=0"
        )
    speed_range = tuple(
        float(value)
        for value in (getattr(motion_cfg, "speed_scale_range", ()) or ())
    )
    if speed_range != (1.0, 1.0):
        raise _OverrideError(
            "[train.py] action-ball requires native motion speed "
            "motion.speed_scale_range=[1.0,1.0]"
        )
    speed_per_clip = getattr(motion_cfg, "speed_scale_per_clip", None)
    if speed_per_clip is not None and (
        len(speed_per_clip) != action_count
        or any(float(value) != 1.0 for value in speed_per_clip)
    ):
        raise _OverrideError(
            "[train.py] action-ball motion.speed_scale_per_clip must be "
            "absent or contain exactly one 1.0 per action"
        )
    if str(getattr(motion_cfg, "event_timing_mode", "")) != "disabled":
        raise _OverrideError(
            "[train.py] action-ball requires "
            "motion.event_timing_mode='disabled'"
        )

    joint_range = tuple(
        float(value)
        for value in (
            getattr(motion_cfg, "joint_position_range", ()) or ()
        )
    )
    if joint_range != (0.0, 0.0):
        raise _OverrideError(
            "[train.py] action-ball canonical-ready entry requires "
            "motion.joint_position_range=[0,0]"
        )
    for attr in ("pose_range", "velocity_range"):
        ranges = getattr(motion_cfg, attr, None)
        try:
            values = list(ranges.values())
        except Exception as exc:
            raise _OverrideError(
                f"[train.py] action-ball motion.{attr} must be a mapping"
            ) from exc
        for value in values:
            try:
                pair = tuple(float(component) for component in value)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    f"[train.py] action-ball motion.{attr} ranges must be [0,0]"
                ) from exc
            if pair != (0.0, 0.0):
                raise _OverrideError(
                    f"[train.py] action-ball canonical-ready entry requires "
                    f"all motion.{attr} ranges to equal [0,0]"
                )

    manifest_phases = tuple(
        float(action.strike_phase) for action in manifest.actions
    )
    configured_phases = tuple(
        float(value)
        for value in (
            getattr(racket_cfg, "strike_phase_per_clip", ()) or ()
        )
    )
    if configured_phases and configured_phases != manifest_phases:
        raise _OverrideError(
            "[train.py] action-ball strike_phase_per_clip disagrees with "
            "the manifest"
        )
    manifest_signs = tuple(
        int(action.mount_normal_sign) for action in manifest.actions
    )
    configured_signs = tuple(
        int(value)
        for value in (
            getattr(racket_cfg, "mount_normal_sign_per_clip", ()) or ()
        )
    )
    if configured_signs and configured_signs != manifest_signs:
        raise _OverrideError(
            "[train.py] action-ball mount_normal_sign_per_clip disagrees "
            "with the manifest"
        )
    racket_cfg.clip_names_per_clip = tuple(clip_names)
    racket_cfg.strike_phase_per_clip = manifest_phases
    racket_cfg.mount_normal_sign_per_clip = manifest_signs
    if hasattr(motion_cfg, "clip_family_per_clip"):
        manifest_families = tuple(
            action.family for action in manifest.actions
        )
        configured_families = tuple(
            str(value)
            for value in (
                getattr(motion_cfg, "clip_family_per_clip", ()) or ()
            )
        )
        if configured_families and configured_families != manifest_families:
            raise _OverrideError(
                "[train.py] action-ball motion.clip_family_per_clip "
                "disagrees with the manifest"
            )
        motion_cfg.clip_family_per_clip = manifest_families

    for attr, expected in (
        ("mount_offset", (0.21021, 0.032078, 0.032036)),
        ("mount_quat", (1.0, 0.0, 0.0, 0.0)),
    ):
        value = getattr(racket_cfg, attr, None)
        try:
            converted = tuple(float(component) for component in value)
        except (TypeError, ValueError):
            converted = ()
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, (list, tuple))
            or converted != expected
        ):
            raise _OverrideError(
                f"[train.py] action-ball racket.{attr} must equal the "
                f"reviewed paddle-site transform {expected!r}"
            )
    if getattr(racket_cfg, "mount_normal_axis", None) != 1:
        raise _OverrideError(
            "[train.py] action-ball racket.mount_normal_axis must be 1"
        )
    if str(getattr(racket_cfg, "racket_body_name", "")) != "pingpang_red_Link":
        raise _OverrideError(
            "[train.py] action-ball requires "
            "racket.racket_body_name='pingpang_red_Link'"
        )
    if str(getattr(racket_cfg, "wrist_body_name", "")) != "right_wrist_yaw_Link":
        raise _OverrideError(
            "[train.py] action-ball requires "
            "racket.wrist_body_name='right_wrist_yaw_Link'"
        )
    if getattr(racket_cfg, "action_ball_fixed_direction", None) is not True:
        raise _OverrideError(
            "[train.py] action-ball requires "
            "racket.action_ball_fixed_direction=true"
        )
    policy_sha = str(
        getattr(racket_cfg, "action_ball_policy_contract_sha256", "")
        or ""
    )
    if (
        len(policy_sha) != 64
        or policy_sha != policy_sha.lower()
        or any(character not in "0123456789abcdef" for character in policy_sha)
    ):
        raise _OverrideError(
            "[train.py] action-ball requires a 64-lowercase-hex "
            "racket.action_ball_policy_contract_sha256"
        )
    refill = getattr(racket_cfg, "action_ball_pool_refill_rows", None)
    if type(refill) is not int or refill <= 0:
        raise _OverrideError(
            "[train.py] action-ball requires a positive plain integer "
            "racket.action_ball_pool_refill_rows"
        )

    # Load/adapter construction is also a cheap pre-gym validation of every
    # profile and curriculum field.  It deliberately does not sample.
    preflight = _action_ball_preflight_contract(
        racket_cfg,
        motion_cfg,
        policy_dt_s=(
            float(env_cfg.sim.dt) * int(env_cfg.decimation)
        ),
    )
    if diagnostic_unauthorized:
        # Franco 2026-07-28 approved bypass: no trust-set or certificate
        # chain is consulted; the applied receipt below and every runtime
        # artifact carry the diagnostic_unauthorized brand instead, and the
        # formal/export paths reject that brand fail-loud.
        print(
            "[train.py] WARN action-ball DIAGNOSTIC UNAUTHORIZED: skipping "
            "canonical motion admission and evaluator launch receipt "
            "validation; this run cannot authorize promotion or export",
            flush=True,
        )
        motion_admission = {
            "diagnostic_unauthorized": True,
            "certificate_sha256": "0" * 64,
        }
        evaluator_launch = {
            "diagnostic_unauthorized": True,
            "launch_receipt_canonical_sha256": "0" * 64,
        }
    else:
        try:
            motion_admission = _validate_action_ball_static_motion_admission(
                preflight,
                motion_cfg=motion_cfg,
            )
        except RuntimeError as exc:
            raise _OverrideError(
                "[train.py] action-ball canonical motion admission is not "
                f"authorized before Gym construction: {exc}"
            ) from exc
        evaluator_launch = _load_action_ball_evaluator_launch_from_cfg(
            racket_cfg,
            motion_cfg,
            preflight=preflight,
        )
    policy = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy is None:
        raise _OverrideError(
            "[train.py] action-ball requires observations.policy"
        )
    occupied = [
        name
        for name in (
            "base_position_table",
            "base_orientation_table_6d",
            "base_lin_vel_heading",
            "racket_target_normal_cmd",
            "racket_target_normal_cmd_heading",
            "racket_target_vel_heading",
            "action_one_hot",
            "time_to_teacher_start_s",
            "station_anchor_err_b",
        )
        if getattr(policy, name, None) is not None
    ]
    if occupied:
        raise _OverrideError(
            "[train.py] action-ball requires a clean Hitter-footwork policy "
            f"prefix; term(s) already attached: {occupied}"
        )
    from isaaclab.managers import ObservationTermCfg as _ObsTerm
    from whole_body_tracking.tasks.tracking import mdp as _mdp

    if include_heading_task:
        # The Hitter-footwork base config owns this slot under the historical
        # ``racket_target_vel_w`` key.  ActionBall must version both the
        # semantics and the term name without moving the slot: ObservationManager
        # concatenates config ``__dict__`` insertion order.  Keep the legacy key
        # as explicit None and insert its frame-correct successor immediately
        # after it, then fail the runtime contract if configclass ordering ever
        # changes underneath us.
        policy_items = list(vars(policy).items())
        legacy_matches = [
            index
            for index, (name, _value) in enumerate(policy_items)
            if name == "racket_target_vel_w"
        ]
        if len(legacy_matches) != 1:
            raise _OverrideError(
                "[train.py] frame-consistent ActionBall expected exactly one "
                "racket_target_vel_w config slot before replacement"
            )
        legacy_value = policy_items[legacy_matches[0]][1]
        if legacy_value is None:
            raise _OverrideError(
                "[train.py] frame-consistent ActionBall cannot replace a "
                "disabled racket_target_vel_w config slot"
            )
        rebuilt_policy_items = []
        for name, value in policy_items:
            if name != "racket_target_vel_w":
                rebuilt_policy_items.append((name, value))
                continue
            rebuilt_policy_items.append(("racket_target_vel_w", None))
            rebuilt_policy_items.append(
                (
                    "racket_target_vel_heading",
                    _ObsTerm(
                        func=_mdp.racket_target_vel_heading,
                        params={"command_name": "racket_target"},
                    ),
                )
            )
        policy.__dict__.clear()
        policy.__dict__.update(rebuilt_policy_items)

    if include_table_pose:
        policy.base_position_table = _ObsTerm(
            func=_mdp.base_position_table,
            params={"command_name": "racket_target"},
        )
        policy.base_orientation_table_6d = _ObsTerm(
            func=_mdp.base_orientation_table_6d,
            params={"command_name": "racket_target"},
        )
    if include_base_twist:
        policy.base_lin_vel_heading = _ObsTerm(
            func=_mdp.base_lin_vel_heading,
            params={"command_name": "racket_target"},
        )
    normal_term_name = (
        "racket_target_normal_cmd_heading"
        if include_heading_task
        else "racket_target_normal_cmd"
    )
    setattr(
        policy,
        normal_term_name,
        _ObsTerm(
            func=(
                _mdp.racket_target_normal_cmd_heading
                if include_heading_task
                else _mdp.racket_target_normal_cmd
            ),
            params={"command_name": "racket_target"},
        ),
    )
    if include_teacher_start:
        # Keep the categorical action identity as the final N columns.  This
        # is a fresh observation contract; old 194-D checkpoints are not
        # silently warm-started under the shifted one-hot offsets.
        policy.time_to_teacher_start_s = _ObsTerm(
            func=_mdp.time_to_teacher_start_s,
            params={"command_name": "racket_target"},
        )
    policy.action_one_hot = _ObsTerm(
        func=_mdp.action_one_hot,
        params={
            "command_name": "racket_target",
            "expected_actions": action_count,
        },
    )
    applied.append(
        "observations.policy action-ball tail="
        + (
            "base_position_table(+3),base_orientation_table_6d(+6),"
            if include_table_pose
            else ""
        )
        + ("base_lin_vel_heading(+3)," if include_base_twist else "")
        + (
            "racket_target_normal_cmd_heading(+4),"
            if include_heading_task
            else "racket_target_normal_cmd(+4),"
        )
        + (
            "time_to_teacher_start_s(+1),"
            if include_teacher_start
            else ""
        )
        + f"action_one_hot(+{action_count}) "
        + f"(actor_obs_contract={expected_actor_contract}; "
        f"preflight_sha256={preflight['sha256']}; "
        "motion_admission_certificate_sha256="
        f"{motion_admission['certificate_sha256']}; "
        "evaluator_launch_sha256="
        f"{evaluator_launch['launch_receipt_canonical_sha256']}"
        + (
            "; diagnostic_unauthorized=true"
            if diagnostic_unauthorized
            else ""
        )
        + ")"
    )


def _validate_task_first_motion_sources(env_cfg, motion_files) -> None:
    """Bind the runtime motion files to the manifest's ordered action revisions."""

    racket_cfg = getattr(getattr(env_cfg, "commands", None), "racket_target", None)
    if racket_cfg is None or str(getattr(racket_cfg, "target_mode", "")) != "task_first":
        return
    loaded = _load_task_first_manifest_from_racket_cfg(racket_cfg)
    paths = [str(path) for path in motion_files]
    if len(paths) != len(loaded.manifest.actions):
        raise _OverrideError(
            "[train.py] task-first motion source count does not match the manifest: "
            f"files={len(paths)} actions={len(loaded.manifest.actions)}"
        )
    for index, (path, action) in enumerate(zip(paths, loaded.manifest.actions)):
        actual_sha256 = _sha256_file(str(pathlib.Path(path).resolve()))
        if actual_sha256 != action.motion_sha256:
            raise _OverrideError(
                "[train.py] task-first motion revision mismatch at local action slot "
                f"{index} ({action.action_id!r}): manifest={action.motion_sha256} "
                f"runtime={actual_sha256} path={path!r}"
            )


def _validate_action_ball_motion_sources(env_cfg, motion_files) -> None:
    """Bind resolved local motion bytes to manifest order and exact paths."""

    commands = getattr(env_cfg, "commands", None)
    racket_cfg = (
        None if commands is None else getattr(commands, "racket_target", None)
    )
    if (
        racket_cfg is None
        or str(getattr(racket_cfg, "target_mode", "")) != "action_ball"
    ):
        return
    motion_cfg = getattr(commands, "motion", None)
    loaded = _load_action_ball_manifest_from_cfg(racket_cfg, motion_cfg)
    if loaded.referenced_assets is None:
        raise _OverrideError(
            "[train.py] action-ball motion validation lacks referenced assets"
        )
    paths = [pathlib.Path(str(path)).resolve() for path in motion_files]
    expected_assets = loaded.referenced_assets.motions
    if len(paths) != len(expected_assets):
        raise _OverrideError(
            "[train.py] action-ball motion source count does not match the "
            f"manifest: files={len(paths)} actions={len(expected_assets)}"
        )
    for index, (path, expected, action) in enumerate(
        zip(paths, expected_assets, loaded.manifest.actions)
    ):
        if path != expected.resolved_path:
            raise _OverrideError(
                "[train.py] action-ball motion path mismatch at slot "
                f"{index} ({action.action_id!r}): manifest={expected.resolved_path!s} "
                f"runtime={path!s}"
            )
        actual_sha256 = _sha256_file(str(path))
        if actual_sha256 != action.motion_sha256:
            raise _OverrideError(
                "[train.py] action-ball motion revision mismatch at slot "
                f"{index} ({action.action_id!r}): "
                f"manifest={action.motion_sha256} runtime={actual_sha256}"
            )


def _build_effective_reward_receipt_for_training(
    env_cfg,
    root_cfg,
    *,
    require_expected_sha256: bool = False,
):
    """Hash the post-override reward terms and optionally enforce a root pin."""

    from whole_body_tracking.utils.effective_reward_recipe import (
        build_effective_reward_receipt,
    )

    expected = _get(root_cfg, "expected_effective_reward_recipe_sha256")
    if expected in (None, ""):
        receipt = build_effective_reward_receipt(env_cfg)
        if require_expected_sha256:
            raise _OverrideError(
                "[train.py] formal task-first requires "
                "expected_effective_reward_recipe_sha256. The fully composed "
                f"candidate SHA-256 is {receipt['sha256']}; review/preregister "
                "that receipt, then relaunch with the exact pin."
            )
        return receipt
    if (
        type(expected) is not str
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise _OverrideError(
            "[train.py] expected_effective_reward_recipe_sha256 must be "
            "exactly 64 lowercase hexadecimal characters"
        )
    return build_effective_reward_receipt(
        env_cfg,
        expected_sha256=expected,
    )


def _write_effective_reward_receipt(
    path: str | pathlib.Path, receipt: dict, hard_contract: dict
) -> None:
    """Persist and read back the effective reward receipt, failing on drift."""

    from whole_body_tracking.utils.effective_reward_recipe import (
        canonical_effective_reward_recipe_json,
    )

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with path.open(encoding="utf-8") as stream:
        written = json.load(stream)
    if written != receipt:
        raise RuntimeError(
            "effective_reward_recipe.json changed during write/readback"
        )
    payload = {
        "schema_version": written.get("schema_version"),
        "terms": written.get("terms"),
    }
    digest = hashlib.sha256(
        canonical_effective_reward_recipe_json(payload).encode("utf-8")
    ).hexdigest()
    if written.get("sha256") != digest:
        raise RuntimeError(
            "effective_reward_recipe.json SHA-256 does not match its canonical payload"
        )
    if hard_contract.get("effective_reward_recipe") != written:
        raise RuntimeError(
            "effective_reward_recipe.json disagrees with the embedded hard contract"
        )


def _write_reward_backend_compatibility_receipt(
    path: str | pathlib.Path,
    receipt: dict,
    effective_reward_receipt: dict,
) -> None:
    """Persist the backend decision without adding disabled terms to Reward."""

    from whole_body_tracking.utils.effective_reward_recipe import (
        canonical_reward_backend_compatibility_json,
    )

    path = pathlib.Path(path)
    expected_keys = {
        "schema_version",
        "kind",
        "effective_reward_recipe_sha256",
        "decisions",
        "sha256",
    }
    if set(receipt) != expected_keys:
        raise RuntimeError(
            "reward backend compatibility receipt has invalid keys"
        )
    if (
        receipt["effective_reward_recipe_sha256"]
        != effective_reward_receipt.get("sha256")
    ):
        raise RuntimeError(
            "reward backend compatibility receipt does not bind the effective "
            "Reward recipe"
        )
    payload = {
        key: receipt[key]
        for key in (
            "schema_version",
            "kind",
            "effective_reward_recipe_sha256",
            "decisions",
        )
    }
    digest = hashlib.sha256(
        canonical_reward_backend_compatibility_json(payload).encode("utf-8")
    ).hexdigest()
    if receipt["sha256"] != digest:
        raise RuntimeError(
            "reward backend compatibility receipt SHA-256 does not match "
            "its canonical payload"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with path.open(encoding="utf-8") as stream:
        written = json.load(stream)
    if written != receipt:
        raise RuntimeError(
            "reward_backend_compatibility.json changed during write/readback"
        )


def _reward_backend_compatibility_log_line(decision: dict) -> str:
    """Render a decision without claiming that its requested weight ran."""

    return (
        "reward backend compatibility: "
        f"{decision['name']} requested_weight="
        f"{decision['requested_weight']} effective_weight="
        f"{decision['effective_weight']} status={decision['status']} "
        f"reason_code={decision['reason_code']}"
    )


def _as_face_command_pairing(value) -> str:
    pairing = str(value)
    allowed = ("shared_plus_y", "legacy_signed_vs_A")
    if pairing not in allowed:
        raise _OverrideError(
            f"task.racket.face_command_pairing must be one of {allowed}, got {pairing!r}"
        )
    return pairing


def _as_target_delay_tts_mode(value) -> str:
    mode = str(value)
    allowed = ("live", "source_timestamp_compensated", "uncompensated")
    if mode not in allowed:
        raise _OverrideError(
            f"task.racket.target_delay_tts_mode must be one of {allowed}, got {mode!r}"
        )
    return mode


def _as_yaw_range(value):
    if len(value) != 2:
        raise _OverrideError("stand_start_yaw_range must contain exactly [lo, hi]")
    lo, hi = (float(value[0]), float(value[1]))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise _OverrideError(
            f"stand_start_yaw_range must be finite with lo <= hi, got {(lo, hi)}"
        )
    if abs(lo) > math.pi or abs(hi) > math.pi:
        raise _OverrideError(
            f"stand_start_yaw_range must stay within [-pi, pi], got {(lo, hi)}"
        )
    return (lo, hi)


def _is_noneish(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null")
    return False


def _configured_items(primary, secondary=None) -> list:
    items = []
    if not _is_noneish(primary):
        if isinstance(primary, (list, tuple, ListConfig)):
            items.extend(primary)
        else:
            items.append(primary)
    if not _is_noneish(secondary):
        if isinstance(secondary, (list, tuple, ListConfig)):
            items.extend(secondary)
        else:
            items.append(secondary)
    return [item for item in items if not _is_noneish(item)]


def _contract_value(value):
    """Convert config values to stable JSON primitives without importing Isaac/Torch helpers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, ListConfig)):
        return [_contract_value(v) for v in value]
    try:
        return [_contract_value(v) for v in value]
    except TypeError:
        return str(value)


def _require_zero_joint_friction_contract(contract: dict) -> None:
    """Fail closed unless the instantiated runtime plant has 31 exact zero coefficients."""
    names = contract.get("joint_names")
    friction = contract.get("joint_friction_coefficients")
    if not isinstance(names, list) or not isinstance(friction, list):
        raise RuntimeError(
            "zero-joint-friction runtime check requires joint_names and "
            "joint_friction_coefficients lists"
        )
    if len(names) != 31 or len(friction) != len(names):
        raise RuntimeError(
            "zero-joint-friction runtime check requires exactly 31 aligned joints, got "
            f"names={len(names)} friction={len(friction)}"
        )
    nonzero = [(name, value) for name, value in zip(names, friction) if float(value) != 0.0]
    if nonzero:
        raise RuntimeError(
            "zero-joint-friction was requested but the instantiated PhysX plant contains "
            f"non-zero coefficients: {nonzero}"
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MOTION_IMITATION_BODY_TERMS = (
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)


def _motion_imitation_body_names_contract(env_cfg) -> dict[str, list[str] | None]:
    """Return the post-override body mask that makes a checkpoint scientifically identifiable.

    Reward weights remain curriculum-mutable, but changing which robot bodies a motion-imitation
    term observes changes the experiment itself.  Each checkpoint therefore binds the four body
    lists after every Hydra override (including free-wrist and non-striking-arm flags).  Generic
    tracking tasks whose term has no explicit body list record ``null`` rather than guessing a
    runtime default.
    """

    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        raise RuntimeError("training hard contract requires env_cfg.rewards")
    contract: dict[str, list[str] | None] = {}
    for term_name in _MOTION_IMITATION_BODY_TERMS:
        if not hasattr(rewards, term_name):
            raise RuntimeError(
                f"training hard contract requires rewards.{term_name}"
            )
        term = getattr(rewards, term_name)
        if term is None:
            contract[term_name] = None
            continue
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            raise RuntimeError(
                f"training hard contract requires rewards.{term_name}.params mapping"
            )
        raw = params.get("body_names")
        if raw is None:
            contract[term_name] = None
            continue
        if isinstance(raw, (str, bytes)):
            raise RuntimeError(
                f"training hard contract requires rewards.{term_name}.body_names list"
            )
        names = [str(value) for value in raw]
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise RuntimeError(
                f"training hard contract requires rewards.{term_name}.body_names "
                "to be non-empty, unique names"
            )
        contract[term_name] = names
    return contract


def _racket_guidance_reward_contract(env_cfg, *, racket_task: bool) -> dict | None:
    """Bind the post-override guidance terms that define a guidance ablation.

    Most reward weights remain curriculum-mutable.  These terms are different: a signed-face
    mechanism comparison uses their exact post-Hydra values as its causal identity.  Binding both
    historical guidance terms and the conditional fixed-budget term prevents a copied checkpoint
    from being relabelled as another arm after it leaves the launch directory.
    """

    if not racket_task:
        return None
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        raise RuntimeError("racket guidance hard contract requires env_cfg.rewards")

    def term_contract(name: str, bound_name: str) -> dict:
        term = getattr(rewards, name, None)
        if term is None:
            raise RuntimeError(f"racket guidance hard contract requires rewards.{name}")
        weight = getattr(term, "weight", None)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RuntimeError(f"rewards.{name}.weight must be a finite number")
        weight = float(weight)
        if not math.isfinite(weight) or weight > 0.0:
            raise RuntimeError(f"rewards.{name}.weight must be finite and <= 0")
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            raise RuntimeError(f"rewards.{name}.params must be a mapping")
        if params.get("command_name") != "racket_target":
            raise RuntimeError(
                f"rewards.{name}.command_name must be exactly 'racket_target'"
            )
        bound = params.get(bound_name)
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise RuntimeError(f"rewards.{name}.{bound_name} must be a finite number")
        bound = float(bound)
        upper = math.pi if bound_name == "theta_max" else math.inf
        if not math.isfinite(bound) or bound <= 0.0 or bound > upper:
            range_text = "(0, pi]" if bound_name == "theta_max" else "> 0"
            raise RuntimeError(f"rewards.{name}.{bound_name} must be finite and {range_text}")
        return {
            "weight": weight,
            "command_name": "racket_target",
            bound_name: bound,
        }

    def conditional_face_contract() -> dict:
        name = "racket_face_conditional_guidance"
        term = getattr(rewards, name, None)
        if term is None:
            raise RuntimeError(f"racket guidance hard contract requires rewards.{name}")
        weight = getattr(term, "weight", None)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RuntimeError(f"rewards.{name}.weight must be a finite number")
        weight = float(weight)
        if not math.isfinite(weight) or weight > 0.0:
            raise RuntimeError(f"rewards.{name}.weight must be finite and <= 0")
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            raise RuntimeError(f"rewards.{name}.params must be a mapping")
        if params.get("command_name") != "racket_target":
            raise RuntimeError(
                f"rewards.{name}.command_name must be exactly 'racket_target'"
            )
        names = ("theta_free", "theta_max", "pos_full", "pos_zero", "vel_full", "vel_zero")
        values: dict[str, float] = {}
        for field in names:
            value = params.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"rewards.{name}.{field} must be a finite number")
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(f"rewards.{name}.{field} must be a finite number")
            values[field] = value
        if not 0.0 <= values["theta_free"] < values["theta_max"] <= math.pi:
            raise RuntimeError(
                f"rewards.{name} requires 0 <= theta_free < theta_max <= pi"
            )
        if not 0.0 < values["pos_full"] < values["pos_zero"]:
            raise RuntimeError(f"rewards.{name} requires 0 < pos_full < pos_zero")
        if not 0.0 < values["vel_full"] < values["vel_zero"]:
            raise RuntimeError(f"rewards.{name} requires 0 < vel_full < vel_zero")
        return {"weight": weight, "command_name": "racket_target", **values}

    return {
        "position": term_contract("racket_guidance", "d_max"),
        "signed_face": term_contract("racket_face_guidance", "theta_max"),
        "conditional_signed_face": conditional_face_contract(),
    }


def _joint_velocity_limit_hinge_reward_contract(env_cfg, runtime_facts: dict) -> dict | None:
    """Bind the post-Hydra qdot-limit hinge and its runtime-ordered denominator.

    The complete 31-element name/limit vectors already live in ``runtime_facts`` and therefore in
    the same hard contract.  This subsection binds the reward identity and refuses any ambiguity
    between articulation order, action order, and the velocity-limit vector it normalizes by.
    Generic tasks without this source-gated term retain ``None`` for backward compatibility.
    """
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        raise RuntimeError("training hard contract requires env_cfg.rewards")
    term = getattr(rewards, "joint_velocity_limit_hinge", None)
    if term is None:
        return None

    weight = getattr(term, "weight", None)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.weight must be a finite number"
        )
    weight = float(weight)
    if not math.isfinite(weight) or weight > 0.0:
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.weight must be finite and <= 0"
        )

    params = getattr(term, "params", None)
    if not isinstance(params, dict):
        raise RuntimeError("rewards.joint_velocity_limit_hinge.params must be a mapping")
    margin = params.get("margin")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.margin must be finite and in (0, 1)"
        )
    margin = float(margin)
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.margin must be finite and in (0, 1)"
        )
    expected_count = params.get("expected_joint_count")
    if type(expected_count) is not int or expected_count != 31:
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.expected_joint_count must be exactly 31"
        )
    asset_cfg = params.get("asset_cfg")
    if getattr(asset_cfg, "name", None) != "robot":
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.asset_cfg must select the robot articulation"
        )
    raw_joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        selected_joint_ids = list(range(31))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        try:
            selected_joint_ids = [int(value) for value in raw_joint_ids]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "rewards.joint_velocity_limit_hinge.asset_cfg must select identity 31-joint order"
            ) from exc
    if selected_joint_ids != list(range(31)):
        raise RuntimeError(
            "rewards.joint_velocity_limit_hinge.asset_cfg must select identity 31-joint order"
        )

    articulation_names = runtime_facts.get("articulation_joint_names")
    joint_names = runtime_facts.get("joint_names")
    limits = runtime_facts.get("joint_velocity_limits")
    if (
        not isinstance(articulation_names, list)
        or not isinstance(joint_names, list)
        or len(joint_names) != 31
        or len(set(joint_names)) != 31
        or joint_names != articulation_names
    ):
        raise RuntimeError(
            "joint_velocity_limit_hinge requires identity 31-joint runtime articulation order"
        )
    if not isinstance(limits, list) or len(limits) != 31:
        raise RuntimeError(
            "joint_velocity_limit_hinge requires 31 runtime joint_velocity_limits"
        )
    try:
        limits = [float(value) for value in limits]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "joint_velocity_limit_hinge limits must be finite and positive"
        ) from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise RuntimeError(
            "joint_velocity_limit_hinge limits must be finite and positive"
        )

    return {
        "schema_version": 1,
        "enabled": weight < 0.0,
        "weight": weight,
        "margin": margin,
        "asset_name": "robot",
        "joint_count": 31,
        "joint_order": "runtime_articulation_identity",
        "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
        # 2026-07-25 SUM 裁定:与 validator 逐字节一致;旧 mean 串 sidecar fail-loud。
        "formula": "sum(relu(abs(qd)/joint_velocity_limits-margin)^2)",
    }


def _processed_qdes_slew_hinge_reward_contract(
    env_cfg, runtime_facts: dict
) -> dict | None:
    """Conditionally bind the deploy-space recovery slew arm.

    The default-off declaration must not change any historical/default canonical hard-contract
    bytes.  Therefore the subsection is absent until either the real term is enabled or train.py
    has raised its zero-valued probe for an explicitly configured control/treatment arm.
    """

    rewards = getattr(env_cfg, "rewards", None)
    term = None if rewards is None else getattr(
        rewards, "processed_qdes_slew_hinge", None
    )
    probe = None if rewards is None else getattr(
        rewards, "processed_qdes_slew_hinge_probe", None
    )
    if term is None and probe is None:
        return None
    if term is None or probe is None:
        raise RuntimeError(
            "processed_qdes_slew_hinge and its activation probe must be declared together"
        )

    raw_weight = getattr(term, "weight", None)
    raw_probe_weight = getattr(probe, "weight", None)
    for label, value in (("weight", raw_weight), ("probe weight", raw_probe_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"rewards.processed_qdes_slew_hinge {label} must be finite and non-positive/zero-valued"
            )
        if not math.isfinite(float(value)):
            raise RuntimeError(
                f"rewards.processed_qdes_slew_hinge {label} must be finite and non-positive/zero-valued"
            )
    weight = float(raw_weight)
    probe_weight = float(raw_probe_weight)
    if weight > 0.0 or probe_weight not in (0.0, 1.0):
        raise RuntimeError(
            "rewards.processed_qdes_slew_hinge weight must be <= 0 and probe weight must be 0 or 1"
        )
    if weight == 0.0 and probe_weight == 0.0:
        return None

    try:
        control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "processed_qdes_slew_hinge requires env_cfg.sim.dt * decimation == 0.02 s"
        ) from exc
    if not math.isfinite(control_dt) or not math.isclose(
        control_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires env_cfg.sim.dt * decimation == 0.02 s"
        )

    params = getattr(term, "params", None)
    probe_params = getattr(probe, "params", None)
    if not isinstance(params, dict) or not isinstance(probe_params, dict):
        raise RuntimeError(
            "rewards.processed_qdes_slew_hinge params must be mappings"
        )
    if params != probe_params:
        raise RuntimeError(
            "processed_qdes_slew_hinge probe params must exactly match the real term"
        )
    if params.get("action_name") != "joint_pos":
        raise RuntimeError(
            "processed_qdes_slew_hinge action_name must be exactly 'joint_pos'"
        )
    if params.get("command_name") != "racket_target":
        raise RuntimeError(
            "processed_qdes_slew_hinge command_name must be exactly 'racket_target'"
        )
    raw_margin = params.get("margin")
    raw_start = params.get("recovery_start_s")
    raw_end = params.get("recovery_end_s")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (raw_margin, raw_start, raw_end)
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge margin/window must be finite with 0 < margin < 1 and 0 <= start < end"
        )
    margin, start, end = map(float, (raw_margin, raw_start, raw_end))
    if (
        not all(math.isfinite(value) for value in (margin, start, end))
        or not 0.0 < margin < 1.0
        or start < 0.0
        or start >= end
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge margin/window must be finite with 0 < margin < 1 and 0 <= start < end"
        )

    runtime_names = runtime_facts.get("joint_names")
    articulation_names = runtime_facts.get("articulation_joint_names")
    limits = runtime_facts.get("joint_velocity_limits")
    waist_leg_names = {
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    }
    if (
        not isinstance(runtime_names, list)
        or len(runtime_names) != 31
        or runtime_names != articulation_names
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires identity 31-joint A3 runtime order"
        )
    selected_names = [name for name in runtime_names if name in waist_leg_names]
    if len(selected_names) != 15 or set(selected_names) != waist_leg_names:
        raise RuntimeError(
            "processed_qdes_slew_hinge requires the exact 15 A3 waist/leg joints"
        )
    if not isinstance(limits, list) or len(limits) != 31:
        raise RuntimeError(
            "processed_qdes_slew_hinge requires 31 runtime joint velocity limits"
        )
    selected_limits = [float(limits[runtime_names.index(name)]) for name in selected_names]
    if any(not math.isfinite(value) or value <= 0.0 for value in selected_limits):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires finite positive waist/leg velocity limits"
        )

    return {
        "schema_version": 1,
        "enabled": weight < 0.0,
        "weight": weight,
        "margin": margin,
        "recovery_start_s": start,
        "recovery_end_s": end,
        "action_name": "joint_pos",
        "command_name": "racket_target",
        "joint_count": 15,
        "joint_names": selected_names,
        "joint_order": "runtime_articulation_subsequence",
        "control_dt_s": control_dt,
        "control_dt_source": "env_cfg.sim.dt_times_decimation",
            "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
            "age_source": "per_env_exact_strike_control_tick_latch",
            "formula": (
            # 2026-07-25 SUM 裁定:与 validator 逐字节一致;旧 mean 串 sidecar fail-loud。
            "sum(1-exp(-square(relu(abs(delta_processed_qdes)/(joint_velocity_limits*0.02)-margin)/(1-margin))))"
        ),
        "gate": "same_attempt_post_strike_age_s_inclusive",
    }


_QDES_LIMIT_BARRIER_FORMULA = (
    # 2026-07-25 站姿豁免:m_eff 按关节收窄到默认站姿之外(0.005 = STANCE_EPS),
    # 旧一刀切 margin_frac 公式的 sidecar 与新数学不可静默互续,合同串随数学一起换。
    "sum(1-exp(-square(relu(m_eff-min(qdes-lo,hi-qdes)/(hi-lo))/m_eff)));"
    "m_eff=min(margin_frac,min(default_q-lo,hi-default_q)/(hi-lo)-0.005)"
)

_SOFT_LIMIT_BARRIER_V2_FORMULA = (
    "sum(where(u>0,penalty_floor+(1-penalty_floor)*"
    "(1-exp(-shape_rate*clamp(u,0,1)))/(1-exp(-shape_rate)),0));"
    "u=relu(m_eff-min(q-lo,hi-q)/(hi-lo))/m_eff;"
    "m_eff=min(margin_frac,min(default_q-lo,hi-default_q)/(hi-lo)-stance_eps);"
    "require_all(m_eff>margin_floor)"
)
_SOFT_LIMIT_BARRIER_V2_SHAPE_RATE = 4.0
_SOFT_LIMIT_BARRIER_V2_STANCE_EPS = 0.005
_SOFT_LIMIT_BARRIER_V2_MARGIN_FLOOR = 0.005


def _authoritative_mdp_reward_callable(name: str):
    """Resolve one Reward callable from the imported production MDP namespace."""

    try:
        from whole_body_tracking.tasks.tracking import mdp

        expected = getattr(mdp, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"cannot resolve authoritative Reward callable mdp.{name}"
        ) from exc
    if not callable(expected):
        raise RuntimeError(f"authoritative Reward mdp.{name} is not callable")
    return expected


def _require_authoritative_reward_callable(term, *, term_name: str, expected: str) -> str:
    """Prove the composed RewardTerm owns the exact reviewed function object."""

    actual = getattr(term, "func", None)
    authoritative = _authoritative_mdp_reward_callable(expected)
    if not callable(actual) or actual is not authoritative:
        raise RuntimeError(
            f"rewards.{term_name}.func must be the authoritative callable "
            f"object mdp.{expected}"
        )
    return f"whole_body_tracking.tasks.tracking.mdp.{expected}"


def _finite_soft_limit_v2_param(params: dict, name: str) -> float:
    value = params.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"soft-limit barrier v2 {name} must be finite")
    return float(value)


def _qdes_limit_barrier_reward_contract(env_cfg, runtime_facts: dict) -> dict | None:
    """Conditionally bind the Wave-Q all-joint q_des position-limit barrier.

    Jiayi V14 全关节 top-k qdes barrier 思想,去 top-k 重做(Franco 指示).  The default-off
    declaration must not change historical hard-contract bytes: the subsection is absent until
    the real term is enabled or train.py has raised the zero-valued probe for an explicitly
    configured control/treatment arm.
    """

    rewards = getattr(env_cfg, "rewards", None)
    term = None if rewards is None else getattr(rewards, "qdes_limit_barrier", None)
    probe = None if rewards is None else getattr(
        rewards, "qdes_limit_barrier_probe", None
    )
    if term is None and probe is None:
        return None
    if term is None or probe is None:
        raise RuntimeError(
            "qdes_limit_barrier and its activation probe must be declared together"
        )

    raw_weight = getattr(term, "weight", None)
    raw_probe_weight = getattr(probe, "weight", None)
    for label, value in (("weight", raw_weight), ("probe weight", raw_probe_weight)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                f"rewards.qdes_limit_barrier {label} must be finite and non-positive/zero-valued"
            )
    weight = float(raw_weight)
    probe_weight = float(raw_probe_weight)
    if weight > 0.0 or probe_weight not in (0.0, 1.0):
        raise RuntimeError(
            "rewards.qdes_limit_barrier weight must be <= 0 and probe weight must be 0 or 1"
        )
    if weight == 0.0 and probe_weight == 0.0:
        return None
    if probe_weight != 1.0:
        raise RuntimeError(
            "explicit qdes_limit_barrier requires its weight-independent probe"
        )

    params = getattr(term, "params", None)
    probe_params = getattr(probe, "params", None)
    if not isinstance(params, dict) or not isinstance(probe_params, dict):
        raise RuntimeError("rewards.qdes_limit_barrier params must be mappings")
    if params != probe_params:
        raise RuntimeError(
            "qdes_limit_barrier probe params must exactly match the real term"
        )
    if params.get("action_name") != "joint_pos":
        raise RuntimeError(
            "qdes_limit_barrier action_name must be exactly 'joint_pos'"
        )
    raw_margin = params.get("margin_frac")
    if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)):
        raise RuntimeError(
            "qdes_limit_barrier margin_frac must be finite and in (0, 0.5)"
        )
    margin_frac = float(raw_margin)
    if not math.isfinite(margin_frac) or not 0.0 < margin_frac < 0.5:
        raise RuntimeError(
            "qdes_limit_barrier margin_frac must be finite and in (0, 0.5)"
        )

    runtime_names = runtime_facts.get("joint_names")
    articulation_names = runtime_facts.get("articulation_joint_names")
    if (
        not isinstance(runtime_names, list)
        or len(runtime_names) != 31
        or len(set(runtime_names)) != 31
        or runtime_names != articulation_names
    ):
        raise RuntimeError(
            "qdes_limit_barrier requires identity 31-joint A3 runtime order"
        )
    limits = runtime_facts.get("qdes_joint_pos_limits")
    if not isinstance(limits, list) or len(limits) != 31:
        raise RuntimeError(
            "qdes_limit_barrier requires 31 runtime qdes_joint_pos_limits pairs"
        )
    for pair in limits:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise RuntimeError(
                "qdes_limit_barrier requires finite [lo, hi] joint position limits with lo < hi"
            )
        lo, hi = float(pair[0]), float(pair[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise RuntimeError(
                "qdes_limit_barrier requires finite [lo, hi] joint position limits with lo < hi"
            )

    # The legacy term intentionally remains schema 1 for exact-resume compatibility.  Fresh
    # ActionBall opts in by composing the reviewed v2 callable and its mandatory floor parameter.
    # Detect v2 from both the callable and parameter surface: a v2 callable with a missing floor
    # must fail closed rather than being silently serialized as v1.
    actual_func = getattr(term, "func", None)
    actual_name = getattr(actual_func, "__name__", "")
    v2_requested = actual_name == "qdes_limit_barrier_v2" or "penalty_floor" in params
    if v2_requested:
        term_callable = _require_authoritative_reward_callable(
            term,
            term_name="qdes_limit_barrier",
            expected="qdes_limit_barrier_v2",
        )
        probe_callable = _require_authoritative_reward_callable(
            probe,
            term_name="qdes_limit_barrier_probe",
            expected="qdes_limit_barrier_v2_probe",
        )
        if set(params) != {"action_name", "margin_frac", "penalty_floor"}:
            raise RuntimeError(
                "qdes_limit_barrier_v2 params must be exactly "
                "action_name, margin_frac, penalty_floor"
            )
        penalty_floor = _finite_soft_limit_v2_param(params, "penalty_floor")
        if not 0.0 < penalty_floor < 1.0:
            raise RuntimeError(
                "qdes_limit_barrier_v2 penalty_floor must be in (0, 1)"
            )
        return {
            "schema_version": 2,
            "enabled": weight < 0.0,
            "probe_enabled": True,
            "term_name": "qdes_limit_barrier",
            "probe_term_name": "qdes_limit_barrier_probe",
            "term_callable": term_callable,
            "probe_callable": probe_callable,
            "activation_ledger": "weight_independent_control_step_counters",
            "weight": weight,
            "margin_frac": margin_frac,
            "penalty_floor": penalty_floor,
            "shape_rate": _SOFT_LIMIT_BARRIER_V2_SHAPE_RATE,
            "stance_eps": _SOFT_LIMIT_BARRIER_V2_STANCE_EPS,
            "margin_floor": _SOFT_LIMIT_BARRIER_V2_MARGIN_FLOOR,
            "action_name": "joint_pos",
            "joint_count": 31,
            "joint_order": "runtime_articulation_identity",
            "position_source": "joint_pos.processed_actions",
            "position_limit_source": "articulation.data.soft_joint_pos_limits",
            "default_stance_source": "articulation.data.default_joint_pos",
            "formula": _SOFT_LIMIT_BARRIER_V2_FORMULA,
            "aggregation": "sum_all_31_joints",
            "per_joint_cap": 1.0,
            "gate": "dense_every_control_step",
        }

    return {
        "schema_version": 1,
        "enabled": weight < 0.0,
        "probe_enabled": True,
        "activation_ledger": "weight_independent_control_step_counters",
        "weight": weight,
        "margin_frac": margin_frac,
        "action_name": "joint_pos",
        "joint_count": 31,
        "joint_order": "runtime_articulation_identity",
        "position_limit_source": "articulation.data.soft_joint_pos_limits",
        "formula": _QDES_LIMIT_BARRIER_FORMULA,
        "gate": "dense_every_control_step",
    }


def _actual_joint_limit_barrier_reward_contract(
    env_cfg,
    runtime_facts: dict,
    *,
    qdes_contract: dict | None,
) -> dict | None:
    """Bind the independent actual-q v2 safety objective.

    This block exists only beside a q_des schema-2 block.  The two channels must use the same
    dose and kernel parameters so a config override cannot silently weaken one side.
    """

    if qdes_contract is None or qdes_contract.get("schema_version") != 2:
        return None
    rewards = getattr(env_cfg, "rewards", None)
    term = None if rewards is None else getattr(rewards, "joint_limit", None)
    probe = None if rewards is None else getattr(
        rewards, "actual_joint_limit_barrier_probe", None
    )
    if term is None or probe is None:
        raise RuntimeError(
            "soft-limit barrier v2 requires joint_limit and "
            "actual_joint_limit_barrier_probe together"
        )
    weight = getattr(term, "weight", None)
    probe_weight = getattr(probe, "weight", None)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) > 0.0
        or isinstance(probe_weight, bool)
        or not isinstance(probe_weight, (int, float))
        or float(probe_weight) != 1.0
    ):
        raise RuntimeError(
            "actual-q soft-limit barrier v2 requires a finite non-positive "
            "weight and weight-independent probe"
        )
    params = getattr(term, "params", None)
    probe_params = getattr(probe, "params", None)
    if not isinstance(params, dict) or not isinstance(probe_params, dict):
        raise RuntimeError("actual-q soft-limit barrier v2 params must be mappings")
    if params != probe_params:
        raise RuntimeError(
            "actual-q soft-limit barrier v2 probe params must exactly match the real term"
        )
    if set(params) != {
        "asset_cfg",
        "margin_frac",
        "penalty_floor",
        "expected_joint_count",
    }:
        raise RuntimeError(
            "actual-q soft-limit barrier v2 params must be exactly asset_cfg, "
            "margin_frac, penalty_floor, expected_joint_count"
        )
    asset_cfg = params["asset_cfg"]
    if getattr(asset_cfg, "name", None) != "robot":
        raise RuntimeError(
            "actual-q soft-limit barrier v2 asset_cfg must select robot"
        )
    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if joint_ids != slice(None):
        raise RuntimeError(
            "actual-q soft-limit barrier v2 asset_cfg must select identity 31 joints"
        )
    if type(params["expected_joint_count"]) is not int or params["expected_joint_count"] != 31:
        raise RuntimeError(
            "actual-q soft-limit barrier v2 expected_joint_count must be 31"
        )
    margin_frac = _finite_soft_limit_v2_param(params, "margin_frac")
    penalty_floor = _finite_soft_limit_v2_param(params, "penalty_floor")
    if not 0.0 < margin_frac < 0.5 or not 0.0 < penalty_floor < 1.0:
        raise RuntimeError(
            "actual-q soft-limit barrier v2 margin_frac/penalty_floor are invalid"
        )
    term_callable = _require_authoritative_reward_callable(
        term,
        term_name="joint_limit",
        expected="actual_joint_limit_barrier_v2",
    )
    probe_callable = _require_authoritative_reward_callable(
        probe,
        term_name="actual_joint_limit_barrier_probe",
        expected="actual_joint_limit_barrier_v2_probe",
    )
    block = {
        "schema_version": 2,
        "enabled": float(weight) < 0.0,
        "probe_enabled": True,
        "term_name": "joint_limit",
        "probe_term_name": "actual_joint_limit_barrier_probe",
        "term_callable": term_callable,
        "probe_callable": probe_callable,
        "activation_ledger": "weight_independent_control_step_counters",
        "weight": float(weight),
        "margin_frac": margin_frac,
        "penalty_floor": penalty_floor,
        "shape_rate": _SOFT_LIMIT_BARRIER_V2_SHAPE_RATE,
        "stance_eps": _SOFT_LIMIT_BARRIER_V2_STANCE_EPS,
        "margin_floor": _SOFT_LIMIT_BARRIER_V2_MARGIN_FLOOR,
        "asset_name": "robot",
        "joint_count": 31,
        "joint_order": "runtime_articulation_identity",
        "position_source": "articulation.data.joint_pos",
        "position_limit_source": "articulation.data.soft_joint_pos_limits",
        "default_stance_source": "articulation.data.default_joint_pos",
        "formula": _SOFT_LIMIT_BARRIER_V2_FORMULA,
        "aggregation": "sum_all_31_joints",
        "per_joint_cap": 1.0,
        "gate": "dense_every_control_step",
    }
    for key in ("weight", "margin_frac", "penalty_floor"):
        if block[key] != qdes_contract[key]:
            raise RuntimeError(
                f"qdes/actual soft-limit barrier v2 {key} must match exactly"
            )
    return block


_A3_LOWER_BODY_RUNTIME_JOINT_ORDER = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
_A3_LOWER_BODY_LEG_JOINTS = frozenset(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER[-12:])


def _lower_body_runtime_contract_names(
    runtime_facts: dict,
    reference_joint_width: int | None = None,
    *,
    require_motion_reference: bool,
) -> tuple[list[str], list[str]]:
    names = runtime_facts.get("joint_names")
    articulation = runtime_facts.get("articulation_joint_names")
    # The live Isaac articulation enumerates joints breadth-first (left_hip_pitch,
    # right_hip_pitch, waist_yaw, ...), not in the deploy-runtime order, so this
    # contract requires identity with the articulation and the exact A3 name set,
    # and selects target joints by name — the same discipline the proven
    # processed_qdes_slew_hinge contract uses.
    if (
        not isinstance(names, list)
        or len(names) != len(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER)
        or len(set(names)) != len(names)
        or set(names) != set(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER)
        or names != articulation
    ):
        raise RuntimeError(
            "lower-body reward contracts require the exact 31-joint A3 runtime order"
        )
    if require_motion_reference and (
        type(reference_joint_width) is not int or reference_joint_width != 31
    ):
        raise RuntimeError(
            "lower-body pose imitation requires a 31-column motion reference"
        )
    # The contract records the canonical deploy-order leg list regardless of the
    # live articulation enumeration; the runtime selects indices by name.
    legs = [
        name
        for name in _A3_LOWER_BODY_RUNTIME_JOINT_ORDER
        if name in _A3_LOWER_BODY_LEG_JOINTS
    ]
    if len(legs) != 12:
        raise RuntimeError("lower-body reward contracts require the exact 12-leg joint set")
    return names, legs


def _lower_body_reward_number(value, *, name: str, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"rewards.{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"rewards.{name} must be a finite number")
    if positive and value <= 0.0:
        raise RuntimeError(f"rewards.{name} must be finite and > 0")
    if nonnegative and value < 0.0:
        raise RuntimeError(f"rewards.{name} must be finite and >= 0")
    return value


def _lower_body_pose_imitation_reward_contract(
    env_cfg, runtime_facts: dict, *, reference_joint_width: int = 31
) -> dict | None:
    rewards = getattr(env_cfg, "rewards", None)
    term = None if rewards is None else getattr(rewards, "lower_body_pose_imitation", None)
    probe = None if rewards is None else getattr(
        rewards, "lower_body_pose_imitation_probe", None
    )
    if term is None and probe is None:
        return None
    if term is None or probe is None:
        raise RuntimeError("lower_body_pose_imitation and probe must be declared together")
    weight = _lower_body_reward_number(
        getattr(term, "weight", None),
        name="lower_body_pose_imitation.weight",
        nonnegative=True,
    )
    probe_weight = _lower_body_reward_number(
        getattr(probe, "weight", None), name="lower_body_pose_imitation_probe.weight"
    )
    if probe_weight not in (0.0, 1.0):
        raise RuntimeError("lower_body_pose_imitation probe weight must be 0 or 1")
    if weight == 0.0 and probe_weight == 0.0:
        return None
    if probe_weight != 1.0:
        raise RuntimeError(
            "explicit lower_body_pose_imitation requires its weight-independent probe"
        )
    params = getattr(term, "params", None)
    probe_params = getattr(probe, "params", None)
    if not isinstance(params, dict) or params != probe_params:
        raise RuntimeError("lower_body_pose_imitation probe params must match the reward")
    if (
        params.get("racket_command_name") != "racket_target"
        or params.get("motion_command_name") != "motion"
    ):
        raise RuntimeError("lower_body_pose_imitation requires racket_target/motion commands")
    std = _lower_body_reward_number(
        params.get("std"), name="lower_body_pose_imitation.std", positive=True
    )
    pre = _lower_body_reward_number(
        params.get("support_pre_s"),
        name="lower_body_pose_imitation.support_pre_s",
        nonnegative=True,
    )
    post = _lower_body_reward_number(
        params.get("support_post_s"),
        name="lower_body_pose_imitation.support_post_s",
        nonnegative=True,
    )
    _, legs = _lower_body_runtime_contract_names(
        runtime_facts,
        reference_joint_width,
        require_motion_reference=True,
    )
    return {
        "schema_version": 1,
        "enabled": weight > 0.0,
        "probe_enabled": True,
        "activation_ledger": "weight_independent_control_step_counters",
        "weight": weight,
        "std_rad": std,
        "support_pre_s": pre,
        "support_post_s": post,
        "racket_command_name": "racket_target",
        "motion_command_name": "motion",
        "joint_count": 12,
        "joint_names": legs,
        "joint_order": "canonical_deploy_order_selected_by_name",
        "reference_joint_order": "motion_command_runtime_articulation_identity",
        "formula": "exp(-mean(square(q_leg-qref_leg))/square(std_rad))",
        "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
        "success_conditioned": False,
    }


def _lower_body_stability_bundle_reward_contract(
    env_cfg, runtime_facts: dict
) -> dict | None:
    rewards = getattr(env_cfg, "rewards", None)
    term = None if rewards is None else getattr(rewards, "lower_body_stability_bundle", None)
    probe = None if rewards is None else getattr(
        rewards, "lower_body_stability_bundle_probe", None
    )
    if term is None and probe is None:
        return None
    if term is None or probe is None:
        raise RuntimeError("lower_body_stability_bundle and probe must be declared together")
    weight = _lower_body_reward_number(
        getattr(term, "weight", None), name="lower_body_stability_bundle.weight"
    )
    probe_weight = _lower_body_reward_number(
        getattr(probe, "weight", None), name="lower_body_stability_bundle_probe.weight"
    )
    if weight > 0.0:
        raise RuntimeError("lower_body_stability_bundle weight must be finite and <= 0")
    if probe_weight not in (0.0, 1.0):
        raise RuntimeError("lower_body_stability_bundle probe weight must be 0 or 1")
    if weight == 0.0 and probe_weight == 0.0:
        return None
    if probe_weight != 1.0:
        raise RuntimeError(
            "explicit lower_body_stability_bundle requires its weight-independent probe"
        )
    params = getattr(term, "params", None)
    probe_params = getattr(probe, "params", None)
    if not isinstance(params, dict) or params != probe_params:
        raise RuntimeError("lower_body_stability_bundle probe params must match the reward")
    if (
        params.get("racket_command_name") != "racket_target"
        or params.get("motion_command_name") != "motion"
    ):
        raise RuntimeError("lower_body_stability_bundle requires racket_target/motion commands")
    numeric_specs = (
        ("min_stance_width_m", True, False),
        ("stance_scale_m", True, False),
        ("leg_velocity_margin_radps", False, True),
        ("leg_velocity_scale_radps", True, False),
        ("support_pre_s", False, True),
        ("support_post_s", False, True),
    )
    values = {
        name: _lower_body_reward_number(
            params.get(name),
            name=f"lower_body_stability_bundle.{name}",
            positive=positive,
            nonnegative=nonnegative,
        )
        for name, positive, nonnegative in numeric_specs
    }
    _, legs = _lower_body_runtime_contract_names(
        runtime_facts,
        require_motion_reference=False,
    )
    articulation_bodies = runtime_facts.get("articulation_body_names")
    required_foot_bodies = ["left_ankle_roll_Link", "right_ankle_roll_Link"]
    if (
        not isinstance(articulation_bodies, list)
        or len(set(articulation_bodies)) != len(articulation_bodies)
        or any(name not in articulation_bodies for name in required_foot_bodies)
    ):
        raise RuntimeError(
            "lower_body_stability_bundle requires exact left/right A3 ankle-roll bodies"
        )
    return {
        "schema_version": 1,
        "enabled": weight < 0.0,
        "probe_enabled": True,
        "activation_ledger": "weight_independent_control_step_counters",
        "weight": weight,
        **values,
        "racket_command_name": "racket_target",
        "motion_command_name": "motion",
        "leg_joint_count": 12,
        "leg_joint_names": legs,
        "foot_body_names": required_foot_bodies,
        "joint_order": "canonical_deploy_order_selected_by_name",
        "stance_width_frame": "base_yaw_lateral_signed_left_minus_right",
        "components": [
            "stance_width_lower_hinge",
            "twelve_leg_realized_qdot_tail",
        ],
        "formula": "mean(bounded_stance_tail,bounded_leg_qdot_tail)",
        "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
        "success_conditioned": False,
        "uses_motion_reference": False,
        "duplicates_slip_or_upright": False,
    }


# S1 post-swing settle debt (Jiayi V13 post-swing debts idea, clean main-side redo; the numeric
# margins/scales are this repo's own, not the unmerged branch's unvalidated numbers).
_POST_SWING_SETTLE_NUMERIC_SPECS = (
    ("base_lin_margin_mps", False, True),
    ("base_lin_scale_mps", True, False),
    ("base_ang_margin_radps", False, True),
    ("base_ang_scale_radps", True, False),
    ("tilt_margin_rad", False, True),
    ("tilt_scale_rad", True, False),
    ("nominal_root_z_m", True, False),
    ("root_height_deadband_m", False, True),
    ("root_height_scale_m", True, False),
    ("foot_slip_margin_mps", False, True),
    ("foot_slip_scale_mps", True, False),
    ("recovery_start_s", False, True),
    ("recovery_end_s", True, False),
)
_POST_SWING_SETTLE_COMPONENTS = [
    "base_quiet_lin",
    "base_quiet_ang",
    "tilt_debt",
    "root_height_debt",
    "settle_foot_slip",
]


def _post_swing_settle_debt_reward_contract(env_cfg, runtime_facts: dict) -> dict | None:
    rewards = getattr(env_cfg, "rewards", None)
    term = None if rewards is None else getattr(rewards, "post_swing_settle_debt", None)
    probe = None if rewards is None else getattr(
        rewards, "post_swing_settle_debt_probe", None
    )
    if term is None and probe is None:
        return None
    if term is None or probe is None:
        raise RuntimeError("post_swing_settle_debt and probe must be declared together")
    weight = _lower_body_reward_number(
        getattr(term, "weight", None), name="post_swing_settle_debt.weight"
    )
    probe_weight = _lower_body_reward_number(
        getattr(probe, "weight", None), name="post_swing_settle_debt_probe.weight"
    )
    if weight > 0.0:
        raise RuntimeError("post_swing_settle_debt weight must be finite and <= 0")
    if probe_weight not in (0.0, 1.0):
        raise RuntimeError("post_swing_settle_debt probe weight must be 0 or 1")
    if weight == 0.0 and probe_weight == 0.0:
        return None
    if probe_weight != 1.0:
        raise RuntimeError(
            "explicit post_swing_settle_debt requires its weight-independent probe"
        )
    params = getattr(term, "params", None)
    probe_params = getattr(probe, "params", None)
    if not isinstance(params, dict) or params != probe_params:
        raise RuntimeError("post_swing_settle_debt probe params must match the reward")
    if (
        params.get("racket_command_name") != "racket_target"
        or params.get("motion_command_name") != "motion"
    ):
        raise RuntimeError("post_swing_settle_debt requires racket_target/motion commands")
    values = {
        name: _lower_body_reward_number(
            params.get(name),
            name=f"post_swing_settle_debt.{name}",
            positive=positive,
            nonnegative=nonnegative,
        )
        for name, positive, nonnegative in _POST_SWING_SETTLE_NUMERIC_SPECS
    }
    if values["recovery_start_s"] >= values["recovery_end_s"]:
        raise RuntimeError(
            "post_swing_settle_debt recovery window must satisfy 0 <= start < end"
        )
    articulation_bodies = runtime_facts.get("articulation_body_names")
    required_foot_bodies = ["left_ankle_roll_Link", "right_ankle_roll_Link"]
    if (
        not isinstance(articulation_bodies, list)
        or len(set(articulation_bodies)) != len(articulation_bodies)
        or any(name not in articulation_bodies for name in required_foot_bodies)
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires exact left/right A3 ankle-roll bodies"
        )
    return {
        "schema_version": 1,
        "enabled": weight < 0.0,
        "probe_enabled": True,
        "activation_ledger": "weight_independent_control_step_counters",
        "weight": weight,
        **values,
        "racket_command_name": "racket_target",
        "motion_command_name": "motion",
        "foot_body_names": required_foot_bodies,
        "components": list(_POST_SWING_SETTLE_COMPONENTS),
        "formula": "mean(5x(1-exp(-square(relu(x-margin)/scale))))",
        "gate": "same_attempt_post_strike_age_s_inclusive",
        "age_source": "per_env_exact_strike_control_tick_latch",
        "success_conditioned": False,
        "uses_motion_reference": False,
    }


def _task_first_agent_recipe(agent_cfg) -> dict:
    """Return the canonical learning recipe required for an exact resume.

    Checkpoint tensors and optimizer moments are not sufficient to reproduce
    the next PPO update: rollout length, GAE/PPO coefficients, minibatching,
    normalization, and policy architecture also participate.  Operational
    fields such as total budget, save cadence, log destination, and device are
    intentionally excluded.
    """

    if agent_cfg is None or not callable(getattr(agent_cfg, "to_dict", None)):
        raise RuntimeError(
            "task-first exact resume requires an RSL runner cfg with to_dict()"
        )
    raw = agent_cfg.to_dict()
    if not isinstance(raw, dict):
        raise RuntimeError("RSL runner cfg to_dict() must return a mapping")

    def canonical(value, *, path: str):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise RuntimeError(f"{path} must not contain non-finite numbers")
            return value
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise RuntimeError(f"{path} must use string mapping keys")
            return {
                key: canonical(value[key], path=f"{path}.{key}")
                for key in sorted(value)
            }
        if isinstance(value, (list, tuple, ListConfig)):
            return [
                canonical(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        raise RuntimeError(
            f"{path} contains unsupported config value {type(value).__name__}"
        )

    steps = raw.get("num_steps_per_env")
    if type(steps) is not int or steps <= 0:
        raise RuntimeError(
            "task-first PPO num_steps_per_env must be a positive plain integer"
        )
    empirical = raw.get("empirical_normalization")
    if type(empirical) is not bool:
        raise RuntimeError(
            "task-first empirical_normalization must be a plain bool"
        )
    policy = canonical(raw.get("policy"), path="agent.policy")
    algorithm = canonical(raw.get("algorithm"), path="agent.algorithm")
    if not isinstance(policy, dict) or not policy:
        raise RuntimeError("task-first policy recipe must be a non-empty mapping")
    if not isinstance(algorithm, dict) or not algorithm:
        raise RuntimeError(
            "task-first algorithm recipe must be a non-empty mapping"
        )
    for optional_stateful_feature in ("rnd_cfg", "rnd", "symmetry_cfg"):
        if algorithm.get(optional_stateful_feature) not in (None, False, {}):
            raise RuntimeError(
                "task-first exact resume does not yet validate state for "
                f"algorithm.{optional_stateful_feature}"
            )

    recipe = {
        "schema_version": 1,
        "runner": {
            "num_steps_per_env": steps,
            "empirical_normalization": empirical,
        },
        "policy": policy,
        "algorithm": algorithm,
    }
    encoded = json.dumps(
        recipe,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "recipe": recipe,
    }


def _action_ball_exact_dict(value, expected_keys, *, name: str) -> dict:
    """Return one exact JSON object or fail on missing/extra contract fields."""

    if type(value) is not dict:
        raise RuntimeError(f"{name} must be a plain mapping")
    actual = set(value)
    expected = set(expected_keys)
    if actual != expected:
        raise RuntimeError(
            f"{name} has invalid keys: "
            f"missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return value


def _action_ball_training_authorization_contract(
    diagnostic_unauthorized: bool,
) -> dict:
    """Bind the diagnostic bypass to explicit negative downstream rights."""

    if type(diagnostic_unauthorized) is not bool:
        raise RuntimeError(
            "action-ball diagnostic authorization flag must be an exact bool"
        )
    return {
        "diagnostic_unauthorized": diagnostic_unauthorized,
        "formal_evidence_prohibited": diagnostic_unauthorized,
        "curriculum_promotion_prohibited": diagnostic_unauthorized,
        "exact_export_prohibited": diagnostic_unauthorized,
        "formal_judge_prohibited": diagnostic_unauthorized,
    }


def _validate_action_ball_training_authorization(
    action_ball_contract,
) -> bool:
    """Return the diagnostic brand after cross-view fail-closed validation."""

    if type(action_ball_contract) is not dict:
        raise RuntimeError("action-ball training contract must be a mapping")
    authorization = _action_ball_exact_dict(
        action_ball_contract.get("authorization"),
        (
            "diagnostic_unauthorized",
            "formal_evidence_prohibited",
            "curriculum_promotion_prohibited",
            "exact_export_prohibited",
            "formal_judge_prohibited",
        ),
        name="action-ball training authorization",
    )
    diagnostic = authorization["diagnostic_unauthorized"]
    expected = _action_ball_training_authorization_contract(diagnostic)
    if authorization != expected:
        raise RuntimeError(
            "action-ball training authorization contains contradictory "
            "diagnostic/formal rights"
        )
    runtime_contract = action_ball_contract.get("runtime")
    motion_admission = action_ball_contract.get("motion_admission")
    if type(runtime_contract) is not dict or type(motion_admission) is not dict:
        raise RuntimeError(
            "action-ball training authorization requires runtime and motion "
            "admission mappings"
        )
    runtime_diagnostic = (
        runtime_contract.get("diagnostic_unauthorized") is True
    )
    motion_diagnostic = (
        motion_admission.get("diagnostic_unauthorized") is True
    )
    if runtime_diagnostic != diagnostic or motion_diagnostic != diagnostic:
        raise RuntimeError(
            "action-ball diagnostic authorization disagrees across the "
            "training/runtime/motion-admission contracts"
        )
    evaluator_authority = runtime_contract.get("evaluator_authority")
    if type(evaluator_authority) is not dict:
        raise RuntimeError(
            "action-ball training authorization requires an evaluator "
            "authority mapping"
        )
    evaluator_diagnostic = (
        evaluator_authority.get("diagnostic_unauthorized") is True
    )
    evaluator_formal = (
        evaluator_authority.get("formal_authority_available") is True
    )
    if (
        evaluator_diagnostic != diagnostic
        or evaluator_formal == diagnostic
    ):
        raise RuntimeError(
            "action-ball evaluator authority disagrees with the live "
            "diagnostic/formal authorization"
        )
    if diagnostic:
        _action_ball_exact_dict(
            evaluator_authority,
            (
                "diagnostic_unauthorized",
                "formal_authority_available",
                "formal_launch_requires_code_pinned_receipt",
                "runtime_or_manifest_may_self_authorize",
                "authority_binding",
                "authority_state_owner_sha256",
            ),
            name="diagnostic action-ball evaluator authority",
        )
        if (
            evaluator_authority[
                "formal_launch_requires_code_pinned_receipt"
            ]
            is not True
            or evaluator_authority[
                "runtime_or_manifest_may_self_authorize"
            ]
            is not False
        ):
            raise RuntimeError(
                "diagnostic action-ball evaluator authority must remain "
                "code-pinned and may not self-authorize"
            )
        if motion_admission.get("training_authorized") is not False:
            raise RuntimeError(
                "diagnostic action-ball motion admission must explicitly "
                "set training_authorized=false"
            )
    elif motion_admission.get("authorization_purpose") != "training":
        raise RuntimeError(
            "formal action-ball motion admission must be training-authorized"
        )
    return diagnostic


def _action_ball_contract_lineage_exact(
    *,
    source_lineage_exact: bool,
    motion_kinematics_exact: bool,
    diagnostic_unauthorized: bool,
) -> bool:
    """Compute formal lineage without allowing a diagnostic run to self-upgrade."""

    values = {
        "source_lineage_exact": source_lineage_exact,
        "motion_kinematics_exact": motion_kinematics_exact,
        "diagnostic_unauthorized": diagnostic_unauthorized,
    }
    for name, value in values.items():
        if type(value) is not bool:
            raise RuntimeError(
                f"action-ball lineage {name} must be an exact bool"
            )
    return (
        source_lineage_exact
        and motion_kinematics_exact
        and not diagnostic_unauthorized
    )


def _action_ball_agent_recipe(
    agent_cfg, policy_bootstrap: dict | None = None
) -> dict:
    """Extend the exact PPO recipe with the action-ball first-reset rule."""

    base = _task_first_agent_recipe(agent_cfg)
    recipe = {
        "schema_version": (
            2 if policy_bootstrap is not None else base["recipe"]["schema_version"]
        ),
        "runner": {
            **base["recipe"]["runner"],
            # A randomized initial episode length can close an attempt before
            # one complete native action cycle and corrupt the first C/F
            # denominator.  Action-ball always starts at a true reset.
            "init_at_random_ep_len": False,
        },
        "policy": base["recipe"]["policy"],
        "algorithm": base["recipe"]["algorithm"],
        **(
            {}
            if policy_bootstrap is None
            else {"policy_initialization": policy_bootstrap}
        ),
    }
    encoded = json.dumps(
        recipe,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "recipe": recipe,
    }


def _resolve_action_ball_shared_ready_bootstrap_request(
    cfg, *, action_ball_launch_requested: bool
) -> tuple[bool, str | None]:
    """Resolve the legacy shared-ready switch and common recipe output path."""

    raw = _get(cfg, "action_ball_shared_ready_bootstrap")
    requested = False if raw is None else _as_explicit_bool(
        raw, "action_ball_shared_ready_bootstrap"
    )
    output = _get(cfg, "action_ball_policy_recipe_output_path")
    if output is not None:
        if type(output) is not str or not output.strip():
            raise RuntimeError(
                "action_ball_policy_recipe_output_path must be null or a "
                "non-empty absolute path"
            )
        output = output.strip()
        if not os.path.isabs(output):
            raise RuntimeError(
                "action_ball_policy_recipe_output_path must be absolute"
            )
    if requested and not action_ball_launch_requested:
        raise RuntimeError(
            "action_ball_shared_ready_bootstrap is ActionBall-only"
        )
    return requested, output


def _resolve_action_ball_dynamic_ready_bootstrap_request(
    cfg, *, action_ball_launch_requested: bool
) -> tuple[bool, dict[str, str] | None]:
    """Resolve the exact N=1 dynamic-ready candidate and hold-receipt pins."""

    raw = _get(cfg, "action_ball_dynamic_ready_bootstrap")
    requested = False if raw is None else _as_explicit_bool(
        raw, "action_ball_dynamic_ready_bootstrap"
    )
    field_names = (
        "action_ball_dynamic_ready_artifact_path",
        "action_ball_dynamic_ready_artifact_sha256",
        "action_ball_dynamic_ready_nominal_receipt_path",
        "action_ball_dynamic_ready_nominal_receipt_sha256",
    )
    values = {name: _get(cfg, name) for name in field_names}
    present = {name: value is not None for name, value in values.items()}
    if not requested:
        if any(present.values()):
            raise RuntimeError(
                "dynamic-ready artifact/receipt pins require "
                "action_ball_dynamic_ready_bootstrap=true"
            )
        return False, None
    if not action_ball_launch_requested:
        raise RuntimeError(
            "action_ball_dynamic_ready_bootstrap is ActionBall-only"
        )
    if not all(present.values()):
        missing = sorted(name for name, is_present in present.items() if not is_present)
        raise RuntimeError(
            "dynamic-ready bootstrap requires all artifact/receipt pins; "
            f"missing={missing}"
        )
    normalized = {}
    for name, value in values.items():
        if type(value) is not str or not value.strip():
            raise RuntimeError(
                f"{name} must be one non-empty string when dynamic-ready is enabled"
            )
        normalized[name] = value.strip()
    return True, normalized


def _materialize_action_ball_policy_recipe(
    output_path: str,
    *,
    policy_recipe: dict,
    policy_bootstrap: dict,
) -> dict:
    """Write one no-clobber bootstrap-bound PPO recipe artifact."""

    path = pathlib.Path(output_path)
    parent = path.parent
    if not parent.is_dir():
        raise RuntimeError(
            "action-ball policy recipe output parent must already exist"
        )
    document = {
        "schema_version": 1,
        "kind": (
            "action_ball_shared_ready_policy_recipe_materialization_v1"
        ),
        "action_count": int(policy_bootstrap["action_count"]),
        "action_order": list(policy_bootstrap["action_order"]),
        "policy_contract_sha256": str(policy_recipe["sha256"]),
        "action_ball_ppo_runner_recipe": policy_recipe,
        "policy_bootstrap": policy_bootstrap,
    }
    encoded = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(
            "action-ball policy recipe output must be a fresh no-clobber file"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return document


def _action_ball_policy_bootstrap_contract(
    env, actor_contract, agent_cfg, *, dynamic_ready_binding=None
) -> dict:
    """Build the fresh-policy shared-ready or N1 dynamic-ready contract."""

    import torch

    from whole_body_tracking.utils.training_contract import (
        ACTION_BALL_POLICY_BOOTSTRAP_KIND,
        action_ball_shared_ready_sha256,
        runtime_execution_facts,
        validate_action_ball_dynamic_ready_runtime_binding,
        validate_action_ball_policy_bootstrap,
    )

    racket_cmd = env.command_manager.get_term("racket_target")
    racket_cfg = racket_cmd.cfg
    if str(getattr(racket_cfg, "target_mode", "")) != "action_ball":
        raise RuntimeError(
            "shared-ready policy bootstrap is only valid for action-ball"
        )
    motion_cmd = env.command_manager.get_term("motion")
    motion = motion_cmd.motion
    action_order = tuple(
        str(value)
        for value in (
            getattr(racket_cfg, "clip_names_per_clip", ()) or ()
        )
    )
    action_count = int(getattr(motion, "num_segments", 0))
    if dynamic_ready_binding is not None:
        try:
            dynamic_ready_binding = (
                validate_action_ball_dynamic_ready_runtime_binding(
                    dynamic_ready_binding, expected_action_count=action_count
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "dynamic-ready actor bootstrap received an invalid runtime binding"
            ) from exc
    if (
        action_count
        not in ((1,) if dynamic_ready_binding is not None else (1, 5))
        or len(action_order) != action_count
        or len(set(action_order)) != action_count
    ):
        raise RuntimeError(
            "shared-ready actor bootstrap supports exact N=1/N=5, while "
            "dynamic-ready bootstrap currently supports exact N=1; every "
            "loaded motion must have one unique action id"
        )
    if (
        dynamic_ready_binding is not None
        and dynamic_ready_binding["action_order"] != list(action_order)
    ):
        raise RuntimeError(
            "dynamic-ready runtime binding action order disagrees with the "
            "loaded ActionBall command order"
        )

    starts = getattr(motion, "seg_start", None)
    joint_pos = getattr(motion, "joint_pos", None)
    if (
        not torch.is_tensor(starts)
        or starts.ndim != 1
        or tuple(starts.shape) != (action_count,)
        or starts.dtype != torch.long
        or not torch.is_tensor(joint_pos)
        or joint_pos.ndim != 2
        or int(joint_pos.shape[1]) != 31
    ):
        raise RuntimeError(
            "shared-ready actor bootstrap cannot identify the exact 31-joint "
            "MotionLoader segment starts"
        )
    ready_per_action = joint_pos.index_select(
        0, starts.to(device=joint_pos.device)
    )
    if dynamic_ready_binding is None:
        shared_ready = ready_per_action[0]
        for slot in range(1, action_count):
            if not torch.equal(ready_per_action[slot], shared_ready):
                mismatch = ready_per_action[slot] != shared_ready
                mismatch_count = int(torch.count_nonzero(mismatch).item())
                max_abs = float(
                    torch.max(
                        torch.abs(
                            ready_per_action[slot].to(dtype=torch.float64)
                            - shared_ready.to(dtype=torch.float64)
                        )
                    ).item()
                )
                raise RuntimeError(
                    "N=5 constant actor bias requires one exact shared ready "
                    f"joint pose; slot={slot} mismatches={mismatch_count} "
                    f"max_abs={max_abs:.9g}. Use action-conditioned bootstrap."
                )
    else:
        physical_ready_values = dynamic_ready_binding["rows"][0][
            "physical_ready"
        ]["joint_pos_rad"]
        physical_ready_tensor = torch.tensor(
            physical_ready_values,
            device=ready_per_action.device,
            dtype=ready_per_action.dtype,
        )
        if not torch.equal(ready_per_action[0], physical_ready_tensor):
            mismatch = ready_per_action[0] != physical_ready_tensor
            raise RuntimeError(
                "dynamic-ready physical source is not the exact loaded motion "
                "frame0; mismatches="
                f"{int(torch.count_nonzero(mismatch).item())}"
            )
        shared_ready = physical_ready_tensor

    runtime_facts = runtime_execution_facts(env, actor_contract)
    joint_names = list(runtime_facts["joint_names"])
    default_q = list(runtime_facts["default_joint_pos"])
    action_scale = list(runtime_facts["action_scale"])
    if len(joint_names) != 31:
        raise RuntimeError(
            "shared-ready actor bootstrap is bound to the 31-joint A3 action order"
        )
    physical_ready_q = [
        float(value)
        for value in shared_ready.detach().cpu().to(dtype=torch.float64).tolist()
    ]
    if dynamic_ready_binding is None:
        target_q = list(physical_ready_q)
        normalized_bias = [
            (ready - default) / scale
            for ready, default, scale in zip(
                target_q, default_q, action_scale
            )
        ]
    else:
        target_q = list(
            dynamic_ready_binding["rows"][0][
                "hold_qdes_joint_pos_rad"
            ]
        )
        normalized_bias = list(
            dynamic_ready_binding["rows"][0]["normalized_actor_action"]
        )
        for index, (default, scale, normalized, target) in enumerate(
            zip(default_q, action_scale, normalized_bias, target_q)
        ):
            if not math.isclose(
                default + scale * normalized,
                target,
                rel_tol=0.0,
                abs_tol=2.0e-7,
            ):
                raise RuntimeError(
                    "dynamic-ready normalized actor action does not decode "
                    f"through the live action decoder at joint {index}"
                )
    startup_event = getattr(
        getattr(env.cfg, "events", None), "add_joint_default_pos", None
    )
    startup_params = (
        None if startup_event is None else getattr(startup_event, "params", None)
    )
    startup_func = (
        None if startup_event is None else getattr(startup_event, "func", None)
    )
    try:
        startup_range = tuple(
            float(value)
            for value in startup_params["pos_distribution_params"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "shared-ready actor bootstrap requires the explicit startup "
            "joint-default calibration range"
        ) from exc
    if (
        len(startup_range) != 2
        or not all(math.isfinite(value) for value in startup_range)
        or startup_range[0] > startup_range[1]
        or startup_params.get("operation") != "add"
        or startup_params.get("distribution", "uniform") != "uniform"
        or getattr(startup_func, "__name__", "")
        != "randomize_joint_default_pos"
    ):
        raise RuntimeError(
            "shared-ready actor bootstrap requires uniform additive startup "
            "joint-default calibration randomization"
        )
    startup_delta_lower = [startup_range[0]] * 31
    startup_delta_upper = [startup_range[1]] * 31

    motion_cfg = motion_cmd.cfg
    motion_files = _configured_items(
        getattr(motion_cfg, "motion_file", None)
    )
    if len(motion_files) != action_count:
        raise RuntimeError(
            "shared-ready actor bootstrap requires one motion file per action"
        )
    motion_sha256 = []
    for index, path in enumerate(motion_files):
        absolute = pathlib.Path(str(path)).expanduser().resolve()
        if not absolute.is_file():
            raise RuntimeError(
                "shared-ready actor bootstrap motion file is missing: "
                f"slot={index} path={absolute}"
            )
        motion_sha256.append(_sha256_file(str(absolute)))
    if (
        dynamic_ready_binding is not None
        and motion_sha256
        != dynamic_ready_binding["motion_sha256_per_action"]
    ):
        raise RuntimeError(
            "dynamic-ready runtime binding motion bytes disagree with the "
            "MotionCommand inputs"
        )

    agent = agent_cfg.to_dict()
    try:
        init_noise_std = float(agent["policy"]["init_noise_std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "shared-ready actor bootstrap requires an explicit PPO init_noise_std"
        ) from exc
    if not math.isfinite(init_noise_std) or init_noise_std != 0.02:
        raise RuntimeError(
            "shared-ready actor bootstrap requires "
            f"algo.policy.init_noise_std=0.02, got {init_noise_std!r}"
        )

    termination = getattr(
        getattr(env.cfg, "terminations", None),
        "joint_qdes_forbidden",
        None,
    )
    params = None if termination is None else getattr(termination, "params", None)
    if (
        not isinstance(params, dict)
        or params.get("action_name") != "joint_pos"
        or params.get("limit_source") != "joint_pos_limits"
        or float(params.get("margin_rad", float("nan"))) != 0.0
        or float(params.get("margin_fraction", float("nan"))) != 0.02
    ):
        raise RuntimeError(
            "shared-ready actor bootstrap requires the existing exact "
            "joint_qdes_forbidden two-percent physical-limit guard"
        )
    hard_limits = env.scene["robot"].data.joint_pos_limits
    if not torch.is_tensor(hard_limits):
        raise RuntimeError(
            "shared-ready actor bootstrap requires articulation hard joint limits"
        )
    if hard_limits.ndim == 3:
        hard_limits = hard_limits[0]
    if tuple(hard_limits.shape) != (31, 2):
        raise RuntimeError(
            "shared-ready actor bootstrap hard joint limits must be [31,2]"
        )
    hard_limits = hard_limits.detach().cpu().to(dtype=torch.float64)
    hard_lower = [float(value) for value in hard_limits[:, 0].tolist()]
    hard_upper = [float(value) for value in hard_limits[:, 1].tolist()]
    hard_inner_lower = [
        lower + 0.02 * (upper - lower)
        for lower, upper in zip(hard_lower, hard_upper)
    ]
    hard_inner_upper = [
        upper - 0.02 * (upper - lower)
        for lower, upper in zip(hard_lower, hard_upper)
    ]
    if dynamic_ready_binding is None:
        ready_source = {
            "semantics": (
                "motion.joint_pos[motion.seg_start[action_slot]]"
            ),
            "canonical_ready_sha256": str(
                getattr(motion_cfg, "canonical_ready_sha256", "") or ""
            ),
            "canonical_ready_fk_sha256": str(
                getattr(motion_cfg, "canonical_ready_fk_sha256", "") or ""
            ),
            "motion_sha256_per_action": motion_sha256,
            "shared_ready_joint_pos": physical_ready_q,
            "shared_ready_joint_pos_sha256": (
                action_ball_shared_ready_sha256(
                    action_order=list(action_order),
                    joint_names=joint_names,
                    shared_ready_joint_pos=physical_ready_q,
                )
            ),
        }
    else:
        ready_source = {
            "semantics": (
                "action_ball_dynamic_ready.rows[action_slot].physical_ready"
            ),
            "motion_sha256_per_action": motion_sha256,
            "physical_ready": dict(
                dynamic_ready_binding["rows"][0]["physical_ready"]
            ),
            "identity": dynamic_ready_binding,
        }
    decoder = {
        "semantics": "q_des=default_joint_pos+action_scale*action",
        "use_default_offset": True,
        "default_joint_pos": default_q,
        "action_scale": action_scale,
        "normalized_bias": normalized_bias,
        "startup_offset_delta_source": (
            "events.add_joint_default_pos.uniform_add"
        ),
        "startup_offset_delta_lower": startup_delta_lower,
        "startup_offset_delta_upper": startup_delta_upper,
    }
    if dynamic_ready_binding is not None:
        decoder["target_joint_pos"] = target_q
    contract = {
        "schema_version": 1 if dynamic_ready_binding is None else 2,
        "kind": ACTION_BALL_POLICY_BOOTSTRAP_KIND,
        "action_count": action_count,
        "action_order": list(action_order),
        "joint_names": joint_names,
        "ready_source": ready_source,
        "decoder": decoder,
        "initialization": {
            "fresh_only": True,
            "resume_overwrite_prohibited": True,
            "output_layer_weight": "zeros",
            "output_layer_bias": "decoder.normalized_bias",
            "init_noise_std": init_noise_std,
            "sigma_envelope": 4.0,
        },
        "hard_inner_guard": {
            "limit_source": "articulation.data.joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
            "hard_lower": hard_lower,
            "hard_upper": hard_upper,
            "hard_inner_lower": hard_inner_lower,
            "hard_inner_upper": hard_inner_upper,
        },
    }
    try:
        validate_action_ball_policy_bootstrap(
            contract, expected_action_count=action_count
        )
    except ValueError as exc:
        raise RuntimeError(
            "ActionBall actor bootstrap failed the 4-sigma hard-inner gate"
        ) from exc
    return contract


def _apply_action_ball_fresh_policy_bootstrap(
    runner, policy_bootstrap: dict, *, checkpoint_path
) -> bool:
    """Apply the reviewed initialization once, and never across a resume."""

    import torch

    from whole_body_tracking.utils.training_contract import (
        validate_action_ball_policy_bootstrap,
    )

    try:
        contract = validate_action_ball_policy_bootstrap(policy_bootstrap)
    except ValueError as exc:
        raise RuntimeError(
            "refusing to apply an invalid ActionBall policy bootstrap"
        ) from exc
    if checkpoint_path is not None:
        return False
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    actor = getattr(policy, "actor", None)
    if not isinstance(actor, torch.nn.Sequential):
        raise RuntimeError(
            "ActionBall bootstrap requires the reviewed RSL actor Sequential"
        )
    children = list(actor.children())
    if not children or not isinstance(children[-1], torch.nn.Linear):
        raise RuntimeError(
            "ActionBall bootstrap cannot safely identify the actor output Linear"
        )
    output = children[-1]
    bias_values = contract["decoder"]["normalized_bias"]
    if (
        output.out_features != len(bias_values)
        or output.bias is None
        or tuple(output.weight.shape)[0] != len(bias_values)
    ):
        raise RuntimeError(
            "ActionBall bootstrap actor output does not match the 31-joint contract"
        )
    expected_bias = torch.tensor(
        bias_values, device=output.bias.device, dtype=output.bias.dtype
    )
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(expected_bias)
    if (
        int(torch.count_nonzero(output.weight).item()) != 0
        or not torch.equal(output.bias, expected_bias)
    ):
        raise RuntimeError(
            "ActionBall actor output bootstrap did not apply exactly"
        )
    actual_std = getattr(policy, "std", None)
    if not torch.is_tensor(actual_std):
        raise RuntimeError(
            "ActionBall bootstrap cannot verify the RSL policy exploration std"
        )
    expected_std = torch.full_like(
        actual_std,
        float(contract["initialization"]["init_noise_std"]),
    )
    if not torch.allclose(
        actual_std, expected_std, rtol=0.0, atol=1.0e-8
    ):
        raise RuntimeError(
            "ActionBall runtime policy std disagrees with the bootstrap contract"
        )
    return True


def _action_ball_sha256(value, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _action_ball_json_equal(actual, expected) -> bool:
    """Type-sensitive equality for canonical JSON contract values."""

    try:
        actual_encoded = json.dumps(
            actual,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        expected_encoded = json.dumps(
            expected,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "action-ball hard contract must contain only finite JSON data"
        ) from exc
    return actual_encoded == expected_encoded


def _action_ball_assert_json_equal(actual, expected, *, name: str) -> None:
    if not _action_ball_json_equal(actual, expected):
        raise RuntimeError(
            f"{name} disagrees with the independently composed launch contract: "
            f"runtime={actual!r}, expected={expected!r}"
        )


def _action_ball_content_receipt(value, *, name: str) -> dict:
    """Verify one executable payload receipt instead of trusting its label."""

    receipt = _action_ball_exact_dict(
        value, ("payload", "sha256"), name=name
    )
    if type(receipt["payload"]) is not dict:
        raise RuntimeError(f"{name}.payload must be a plain mapping")
    digest = _action_ball_sha256(
        receipt["sha256"], name=f"{name}.sha256"
    )
    actual = _canonical_contract_sha256(receipt["payload"])
    if digest != actual:
        raise RuntimeError(
            f"{name}.sha256 does not authenticate its payload: "
            f"declared={digest}, actual={actual}"
        )
    return receipt


def _action_ball_repo_relative_source(
    path_value,
    *,
    repo_root: pathlib.Path,
    name: str,
    reject_symlinks: bool = False,
) -> pathlib.Path:
    """Resolve one canonical POSIX source path below the trusted repository."""

    if type(path_value) is not str or not path_value:
        raise RuntimeError(f"{name} must be a non-empty string")
    pure = pathlib.PurePosixPath(path_value)
    if (
        pure.is_absolute()
        or path_value != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in path_value
    ):
        raise RuntimeError(
            f"{name} must be a normalized repository-relative POSIX path"
        )
    try:
        root = pathlib.Path(repo_root).resolve(strict=True)
        unresolved = root.joinpath(*pure.parts)
        if reject_symlinks:
            component = root
            for part in pure.parts:
                component = component / part
                if component.is_symlink():
                    raise RuntimeError(
                        f"{name} must not traverse a symbolic link"
                    )
        source = unresolved.resolve(strict=True)
        source.relative_to(root)
    except RuntimeError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"{name} does not resolve below the trusted repository root"
        ) from exc
    if not source.is_file():
        raise RuntimeError(f"{name} must resolve to a regular file")
    return source


def _load_action_ball_canonical_registry_module(
    repo_root: pathlib.Path,
):
    """Execute the exact repository canonical-registry verifier.

    Formal motion authority is code-rooted.  Reusing a caller-populated import
    alias would let configuration or a test double replace the code-owned
    promotion-certificate trust set, so this loader always executes the
    repository bytes under a private alias.
    """

    source = _action_ball_repo_relative_source(
        (
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_registry.py"
        ),
        repo_root=repo_root,
        name="action-ball canonical registry verifier source",
        reject_symlinks=True,
    )
    expected_source = pathlib.Path(__file__).resolve(strict=True).with_name(
        "canonical_motion_registry.py"
    )
    if source != expected_source:
        raise RuntimeError(
            "action-ball canonical registry verifier is not the exact "
            "training-checkout source"
        )
    module_name = "_hope_action_ball_pre_gym_canonical_motion_registry"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot create the action-ball canonical registry verifier spec"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    try:
        module_source = pathlib.Path(module.__file__).resolve(strict=True)
    except (AttributeError, OSError) as exc:
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            "action-ball canonical registry verifier has no exact source"
        ) from exc
    if module_source != source:
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            "action-ball canonical registry verifier executed wrong bytes"
        )
    return module


def _validate_action_ball_static_motion_admission(
    preflight: dict,
    *,
    motion_cfg,
) -> dict:
    """Re-prove generic-v2-capable motion admission before constructing Gym.

    The manifest supplies ordered identity only.  Training authority comes
    from an independently pinned canonical registry plus exact promotion
    certificate bytes whose digest is present in the verifier's code-owned
    trust set.  The canonical loader accepts legacy schema 1 and arbitrary-N
    schema 2; this layer deliberately does not impose a five-action limit.
    """

    repo_root = _action_ball_repo_root(motion_cfg)
    registry_path_text = str(
        getattr(motion_cfg, "canonical_registry_path", "") or ""
    ).strip()
    certificate_path_text = str(
        getattr(
            motion_cfg, "canonical_promotion_certificate_path", ""
        )
        or ""
    ).strip()
    registry_path = _action_ball_repo_relative_source(
        registry_path_text,
        repo_root=repo_root,
        name="action-ball motion.canonical_registry_path",
        reject_symlinks=True,
    )
    certificate_path = _action_ball_repo_relative_source(
        certificate_path_text,
        repo_root=repo_root,
        name=(
            "action-ball motion.canonical_promotion_certificate_path"
        ),
        reject_symlinks=True,
    )
    expected_registry_sha256 = _action_ball_sha256(
        getattr(motion_cfg, "canonical_registry_sha256", None),
        name="action-ball motion.canonical_registry_sha256",
    )
    expected_alignment_sha256 = _action_ball_sha256(
        getattr(
            motion_cfg, "canonical_registry_alignment_sha256", None
        ),
        name="action-ball motion.canonical_registry_alignment_sha256",
    )
    expected_ready_sha256 = _action_ball_sha256(
        getattr(motion_cfg, "canonical_ready_sha256", None),
        name="action-ball motion.canonical_ready_sha256",
    )
    expected_ready_fk_sha256 = _action_ball_sha256(
        getattr(motion_cfg, "canonical_ready_fk_sha256", None),
        name="action-ball motion.canonical_ready_fk_sha256",
    )

    module = _load_action_ball_canonical_registry_module(repo_root)
    admission_module = getattr(module, "motion_admission", None)
    trusted = getattr(
        admission_module,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        None,
    )
    if (
        type(trusted) is not frozenset
        or any(
            type(value) is not str
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
            for value in trusted
        )
    ):
        raise RuntimeError(
            "canonical motion promotion-certificate code trust set has an "
            "invalid shape"
        )
    if not trusted:
        raise RuntimeError(
            "canonical motion promotion-certificate code trust set is empty; "
            "diagnostic registry inspection cannot authorize formal training"
        )
    certificate_sha256 = _sha256_file(str(certificate_path))
    if certificate_sha256 not in trusted:
        raise RuntimeError(
            "configured canonical promotion certificate raw-byte SHA-256 is "
            "absent from the code trust set"
        )
    actual_registry_sha256 = _sha256_file(str(registry_path))
    if actual_registry_sha256 != expected_registry_sha256:
        raise RuntimeError(
            "configured canonical registry bytes disagree with the exact "
            "registry pin: "
            f"configured={expected_registry_sha256}, "
            f"actual={actual_registry_sha256}"
        )

    loader = getattr(module, "load_training_adopted_registry", None)
    if not callable(loader):
        raise RuntimeError(
            "canonical registry verifier lacks "
            "load_training_adopted_registry()"
        )
    try:
        registry, tables = loader(
            str(registry_path),
            str(certificate_path),
            repo_root=str(repo_root),
            expected_registry_sha256=expected_registry_sha256,
            expected_alignment_sha256=expected_alignment_sha256,
            expected_canonical_ready_sha256=expected_ready_sha256,
            expected_canonical_ready_fk_sha256=expected_ready_fk_sha256,
            expected_promotion_certificate_sha256=certificate_sha256,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(
            "canonical registry/certificate/evidence closure failed strict "
            f"training adoption: {exc}"
        ) from exc

    registry_type = getattr(module, "CanonicalMotionBankRegistry", None)
    tables_type = getattr(module, "CanonicalRuntimeTables", None)
    if (
        not isinstance(registry_type, type)
        or type(registry) is not registry_type
        or not isinstance(tables_type, type)
        or type(tables) is not tables_type
    ):
        raise RuntimeError(
            "canonical registry loader returned an untrusted result type"
        )
    supported_schema_versions = (
        getattr(module, "REGISTRY_SCHEMA_VERSION", None),
        getattr(module, "GENERIC_REGISTRY_SCHEMA_VERSION", None),
    )
    if (
        type(registry.schema_version) is not int
        or registry.schema_version not in supported_schema_versions
    ):
        raise RuntimeError(
            "canonical registry returned an unsupported schema version"
        )
    if (
        pathlib.Path(registry.path).resolve(strict=True) != registry_path
        or pathlib.Path(registry.repo_root).resolve(strict=True) != repo_root
        or registry.registry_sha256 != expected_registry_sha256
        or registry.registry_digest_pinned is not True
    ):
        raise RuntimeError(
            "canonical registry result lost its exact path/root/byte pin"
        )
    if (
        tables.registry_sha256 != expected_registry_sha256
        or tables.alignment_sha256 != expected_alignment_sha256
        or tables.canonical_ready_sha256 != expected_ready_sha256
        or tables.canonical_ready_fk_sha256 != expected_ready_fk_sha256
        or tables.authorization_purpose != "training"
    ):
        raise RuntimeError(
            "canonical runtime tables disagree with the four pinned training "
            "identities"
        )

    action_bindings = preflight.get("action_bindings")
    if type(action_bindings) is not list or not action_bindings:
        raise RuntimeError(
            "action-ball preflight action_bindings must be a non-empty list"
        )
    expected_action_order = tuple(preflight.get("action_order", ()))
    if (
        not expected_action_order
        or len(expected_action_order) != len(action_bindings)
        or tuple(registry.motion_ids) != expected_action_order
        or tuple(tables.motion_ids) != expected_action_order
    ):
        raise RuntimeError(
            "canonical registry ordered motion_ids disagree with the "
            "action-ball manifest action_order"
        )
    if (
        registry.scope != preflight.get("prototype", {}).get("scope")
        or tables.scope != registry.scope
        or tables.bank_id != registry.bank_id
    ):
        raise RuntimeError(
            "canonical bank scope/identity disagrees with the manifest "
            "prototype"
        )
    if (
        type(registry.entries) is not tuple
        or len(registry.entries) != len(action_bindings)
    ):
        raise RuntimeError(
            "canonical registry row count disagrees with the action manifest"
        )

    motion_rows = []
    resolved_motion_files = []
    for index, (binding, entry) in enumerate(
        zip(action_bindings, registry.entries)
    ):
        binding = _action_ball_exact_dict(
            binding,
            (
                "action_id",
                "action_uid",
                "action_slot",
                "family",
                "motion_path",
                "motion_sha256",
                "sampling_profile_sha256",
                "strike_phase",
                "mount_normal_sign",
            ),
            name=f"action-ball preflight action_bindings[{index}]",
        )
        expected_row = {
            "motion_id": binding["action_id"],
            "motion_path": binding["motion_path"],
            "motion_sha256": binding["motion_sha256"],
            "family": binding["family"],
            "strike_phase": float(binding["strike_phase"]),
            "mount_normal_sign": float(binding["mount_normal_sign"]),
        }
        observed_row = {
            "motion_id": entry.motion_id,
            "motion_path": entry.npz_path_text,
            "motion_sha256": entry.npz_sha256,
            "family": entry.family,
            "strike_phase": float(entry.strike_phase),
            "mount_normal_sign": float(entry.mount_normal_sign),
        }
        _action_ball_assert_json_equal(
            observed_row,
            expected_row,
            name=f"canonical motion registry row[{index}]",
        )
        resolved_motion_files.append(
            str(
                _action_ball_repo_relative_source(
                    binding["motion_path"],
                    repo_root=repo_root,
                    name=f"action-ball manifest motion[{index}]",
                    reject_symlinks=True,
                )
            )
        )
        motion_rows.append(
            {
                "motion_id": binding["action_id"],
                "action_uid": binding["action_uid"],
                "action_slot": binding["action_slot"],
                "motion_path": binding["motion_path"],
                "motion_sha256": binding["motion_sha256"],
                "profile_sha256": binding[
                    "sampling_profile_sha256"
                ],
            }
        )
    if tuple(tables.motion_file) != tuple(resolved_motion_files):
        raise RuntimeError(
            "canonical runtime motion_file table disagrees with manifest "
            "paths"
        )
    if (
        tuple(tables.clip_family_per_clip)
        != tuple(row["family"] for row in action_bindings)
        or tuple(float(value) for value in tables.strike_phase_per_clip)
        != tuple(float(row["strike_phase"]) for row in action_bindings)
        or tuple(float(value) for value in tables.mount_normal_sign_per_clip)
        != tuple(
            float(row["mount_normal_sign"]) for row in action_bindings
        )
    ):
        raise RuntimeError(
            "canonical runtime family/phase/face tables disagree with the "
            "manifest"
        )

    binding_factory = getattr(module, "bank_promotion_binding", None)
    binding_hasher = getattr(admission_module, "_binding_sha256", None)
    if not callable(binding_factory) or not callable(binding_hasher):
        raise RuntimeError(
            "canonical admission verifier lacks the exact promotion-binding "
            "derivation"
        )
    try:
        promotion_binding = binding_factory(
            registry, authorization_purpose="training"
        )
        promotion_binding_sha256 = _action_ball_sha256(
            binding_hasher(promotion_binding),
            name="canonical promotion binding SHA-256",
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(
            "canonical promotion binding could not be independently derived"
        ) from exc

    try:
        ready_path = (
            pathlib.Path(registry.canonical_ready_path)
            .resolve(strict=True)
            .relative_to(repo_root)
            .as_posix()
        )
        ready_fk_path = (
            pathlib.Path(registry.canonical_ready_fk_path)
            .resolve(strict=True)
            .relative_to(repo_root)
            .as_posix()
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "canonical ready assets do not resolve below the trusted root"
        ) from exc
    return {
        "registry_schema_version": registry.schema_version,
        "bank_id": registry.bank_id,
        "scope": registry.scope,
        "registry_path": registry_path_text,
        "registry_sha256": expected_registry_sha256,
        "alignment_sha256": expected_alignment_sha256,
        "canonical_ready_path": ready_path,
        "canonical_ready_sha256": expected_ready_sha256,
        "canonical_ready_fk_path": ready_fk_path,
        "canonical_ready_fk_sha256": expected_ready_fk_sha256,
        "certificate_path": certificate_path_text,
        "certificate_sha256": certificate_sha256,
        "promotion_binding_sha256": promotion_binding_sha256,
        "motion_rows": motion_rows,
    }


def _validate_action_ball_motion_admission_receipt(
    receipt,
    *,
    preflight: dict,
    motion_cfg,
    expected_runtime_contract_sha256: str,
    expected_broker_state_schema_version: int,
    expected_broker_registry_sha256: str,
    expected_provider_state_owner_sha256: str,
) -> dict:
    """Independently validate MotionCommand's opaque runtime admission receipt."""

    repo_root = _action_ball_repo_root(motion_cfg)
    static = _validate_action_ball_static_motion_admission(
        preflight,
        motion_cfg=motion_cfg,
    )
    row = _action_ball_exact_dict(
        receipt,
        (
            "schema_version",
            "kind",
            "authorization_purpose",
            "trusted_repo_root",
            "opaque_capability",
            "canonical_bank",
            "runtime_binding",
            "implementation_sources",
            "canonical_sha256",
        ),
        name="action-ball opaque motion admission receipt",
    )
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise RuntimeError(
            "action-ball opaque motion admission schema_version must be 1"
        )
    if (
        row["kind"]
        != "whole_body_tracking.MotionCommand.action_ball_motion_admission"
        or row["authorization_purpose"] != "training"
    ):
        raise RuntimeError(
            "action-ball opaque motion admission kind/purpose drifted"
        )
    if row["trusted_repo_root"] != str(repo_root):
        raise RuntimeError(
            "action-ball opaque motion admission trusted_repo_root drifted"
        )
    declared = _action_ball_sha256(
        row["canonical_sha256"],
        name="action-ball opaque motion admission canonical_sha256",
    )
    unsigned = dict(row)
    del unsigned["canonical_sha256"]
    computed = _canonical_contract_sha256(unsigned)
    if declared != computed:
        raise RuntimeError(
            "action-ball opaque motion admission canonical_sha256 mismatch: "
            f"declared={declared}, actual={computed}"
        )

    opaque = _action_ball_exact_dict(
        row["opaque_capability"],
        (
            "capability_type",
            "purpose",
            "promotion_binding_sha256",
            "certificate_path",
            "certificate_sha256",
        ),
        name="action-ball opaque motion capability",
    )
    expected_opaque = {
        "capability_type": "TrustedMotionAdmission",
        "purpose": "training",
        "promotion_binding_sha256": static[
            "promotion_binding_sha256"
        ],
        "certificate_path": static["certificate_path"],
        "certificate_sha256": static["certificate_sha256"],
    }
    _action_ball_assert_json_equal(
        opaque,
        expected_opaque,
        name="action-ball opaque motion capability",
    )

    bank = _action_ball_exact_dict(
        row["canonical_bank"],
        (
            "bank_id",
            "scope",
            "registry_path",
            "registry_sha256",
            "alignment_sha256",
            "canonical_ready_path",
            "canonical_ready_sha256",
            "canonical_ready_fk_path",
            "canonical_ready_fk_sha256",
            "motion_rows",
        ),
        name="action-ball opaque canonical bank",
    )
    expected_bank = {
        "bank_id": static["bank_id"],
        "scope": static["scope"],
        "registry_path": static["registry_path"],
        "registry_sha256": static["registry_sha256"],
        "alignment_sha256": static["alignment_sha256"],
        "canonical_ready_path": static["canonical_ready_path"],
        "canonical_ready_sha256": static["canonical_ready_sha256"],
        "canonical_ready_fk_path": static["canonical_ready_fk_path"],
        "canonical_ready_fk_sha256": static[
            "canonical_ready_fk_sha256"
        ],
        "motion_rows": static["motion_rows"],
    }
    _action_ball_assert_json_equal(
        bank,
        expected_bank,
        name="action-ball opaque canonical bank",
    )

    runtime_binding = _action_ball_exact_dict(
        row["runtime_binding"],
        (
            "runtime_contract_sha256",
            "broker_state_schema_version",
            "broker_registry_sha256",
            "provider_state_owner_sha256",
            "ordered_action_uids",
            "manifest_rows_are_identity_only",
        ),
        name="action-ball opaque motion runtime binding",
    )
    expected_runtime_sha256 = _action_ball_sha256(
        expected_runtime_contract_sha256,
        name="action-ball opaque expected runtime contract SHA-256",
    )
    expected_registry_sha256 = _action_ball_sha256(
        expected_broker_registry_sha256,
        name="action-ball opaque expected broker registry SHA-256",
    )
    expected_owner_sha256 = _action_ball_sha256(
        expected_provider_state_owner_sha256,
        name="action-ball opaque expected provider owner SHA-256",
    )
    if (
        type(expected_broker_state_schema_version) is not int
        or expected_broker_state_schema_version <= 0
    ):
        raise RuntimeError(
            "action-ball expected broker state schema must be a positive "
            "plain integer"
        )
    expected_runtime_binding = {
        "runtime_contract_sha256": expected_runtime_sha256,
        "broker_state_schema_version": (
            expected_broker_state_schema_version
        ),
        "broker_registry_sha256": expected_registry_sha256,
        "provider_state_owner_sha256": expected_owner_sha256,
        "ordered_action_uids": list(preflight["action_uids"]),
        "manifest_rows_are_identity_only": True,
    }
    _action_ball_assert_json_equal(
        runtime_binding,
        expected_runtime_binding,
        name="action-ball opaque motion runtime binding",
    )

    expected_source_paths = {
        "commands": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/commands.py"
        ),
        "action_ball_runtime": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_runtime.py"
        ),
        "canonical_motion_registry": (
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_registry.py"
        ),
        "canonical_motion_admission": (
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_admission.py"
        ),
    }
    sources = _action_ball_exact_dict(
        row["implementation_sources"],
        tuple(expected_source_paths),
        name="action-ball opaque motion implementation sources",
    )
    for source_name, expected_path in expected_source_paths.items():
        source_receipt = _action_ball_exact_dict(
            sources[source_name],
            ("path", "sha256"),
            name=(
                "action-ball opaque motion implementation source "
                f"{source_name}"
            ),
        )
        if source_receipt["path"] != expected_path:
            raise RuntimeError(
                "action-ball opaque motion implementation source "
                f"{source_name} resolved to an unexpected path"
            )
        source = _action_ball_repo_relative_source(
            source_receipt["path"],
            repo_root=repo_root,
            name=(
                "action-ball opaque motion implementation source "
                f"{source_name}.path"
            ),
            reject_symlinks=True,
        )
        declared_source_sha256 = _action_ball_sha256(
            source_receipt["sha256"],
            name=(
                "action-ball opaque motion implementation source "
                f"{source_name}.sha256"
            ),
        )
        actual_source_sha256 = _sha256_file(str(source))
        if declared_source_sha256 != actual_source_sha256:
            raise RuntimeError(
                "action-ball opaque motion implementation source "
                f"{source_name} bytes drifted: "
                f"declared={declared_source_sha256}, "
                f"actual={actual_source_sha256}"
            )
    return row


def _validate_action_ball_mdp_source_map(
    value, *, expected_names, repo_root: pathlib.Path, name: str
) -> dict:
    """Re-hash one exact map of executable MDP module sources."""

    expected_names = tuple(expected_names)
    if (
        not expected_names
        or len(expected_names) != len(set(expected_names))
        or any(
            type(source_name) is not str
            or not source_name
            or pathlib.PurePosixPath(source_name).name != source_name
            for source_name in expected_names
        )
    ):
        raise RuntimeError(f"{name} expected source-name schema is invalid")
    source_map = _action_ball_exact_dict(
        value, expected_names, name=name
    )
    mdp_prefix = pathlib.PurePosixPath(
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    for source_name in expected_names:
        declared = _action_ball_sha256(
            source_map[source_name],
            name=f"{name}.{source_name}",
        )
        source = _action_ball_repo_relative_source(
            (mdp_prefix / source_name).as_posix(),
            repo_root=repo_root,
            name=f"{name}.{source_name}.path",
        )
        actual = _sha256_file(str(source))
        if declared != actual:
            raise RuntimeError(
                f"{name}.{source_name} source bytes drifted: "
                f"declared={declared}, actual={actual}"
            )
    return source_map


def _validate_action_ball_reference_guard_contract(value, *, racket_cfg) -> dict:
    """Bind the active reference-envelope behavior and its counter schema."""

    from whole_body_tracking.tasks.tracking.mdp.action_ball_reference_guard import (
        REFERENCE_GUARD_CONTRACT_PAYLOAD,
        REFERENCE_GUARD_CONTRACT_SHA256,
        validate_reference_guard_mode,
    )

    row = _action_ball_exact_dict(
        value,
        ("mode", "contract_payload", "contract_sha256"),
        name="action-ball reference guard",
    )
    mode = validate_reference_guard_mode(row["mode"])
    configured_mode = validate_reference_guard_mode(
        getattr(racket_cfg, "reference_guard_mode", None)
    )
    if mode != configured_mode:
        raise RuntimeError(
            "action-ball reference-guard runtime/config mode mismatch: "
            f"runtime={mode!r}, configured={configured_mode!r}"
        )
    _action_ball_assert_json_equal(
        row["contract_payload"],
        REFERENCE_GUARD_CONTRACT_PAYLOAD,
        name="action-ball reference-guard contract payload",
    )
    declared = _action_ball_sha256(
        row["contract_sha256"],
        name="action-ball reference-guard contract SHA-256",
    )
    if declared != REFERENCE_GUARD_CONTRACT_SHA256:
        raise RuntimeError(
            "action-ball reference-guard contract SHA-256 drifted: "
            f"runtime={declared}, executable={REFERENCE_GUARD_CONTRACT_SHA256}"
        )
    return row


def _validate_action_ball_evaluator_launch_receipt(
    receipt,
    *,
    declared_launch_sha256,
    preflight: dict,
    solver_sha256: str,
    repo_root: pathlib.Path,
    attempt_source,
) -> dict:
    """Verify the code-pinned V4 inbox evaluator launch authority.

    Formal ActionBall has no legacy evaluator fallback.  The exact receipt
    must construct the V4 authority over the append-only inbox source before
    ``gym.make``; a schema-3 receipt therefore fails at this boundary.
    """

    row = _action_ball_exact_dict(
        receipt,
        (
            "schema_version",
            "kind",
            "authority_contract_sha256",
            "curriculum_contract_sha256",
            "profile_order",
            "arm_catalog_sha256",
            "scheduler_contract_sha256",
            "sampler_sha256",
            "solver_sha256",
            "policy_contract_sha256",
            "attempt_source_contract_sha256",
            "attempt_source_path",
            "attempt_source_sha256",
            "window_contract",
        ),
        name="action-ball frozen evaluator launch receipt",
    )

    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_curriculum as curriculum_module,
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_evaluation as evaluator_module,
    )

    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != evaluator_module.V4_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "action-ball frozen evaluator launch schema_version must equal "
            f"V4 ({evaluator_module.V4_SCHEMA_VERSION})"
        )
    if row["kind"] != "action_ball_frozen_evaluator_v4_launch":
        raise RuntimeError(
            "action-ball formal evaluator must be the V4 inbox launch"
        )

    authority_sha256 = _action_ball_sha256(
        row["authority_contract_sha256"],
        name="action-ball evaluator authority_contract_sha256",
    )
    if (
        authority_sha256
        != evaluator_module.FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
    ):
        raise RuntimeError(
            "action-ball V4 evaluator authority contract disagrees with code"
        )

    expected_profiles = [
        {
            "action_uid": binding["action_uid"],
            "profile_sha256": binding["sampling_profile_sha256"],
            "mobility": preflight["mobility_mode"],
        }
        for binding in preflight["action_bindings"]
    ]
    if type(row["profile_order"]) is not list:
        raise RuntimeError(
            "action-ball evaluator profile_order must be a plain list"
        )
    for index, profile in enumerate(row["profile_order"]):
        _action_ball_exact_dict(
            profile,
            ("action_uid", "profile_sha256", "mobility"),
            name=f"action-ball evaluator profile_order[{index}]",
        )
    _action_ball_assert_json_equal(
        row["profile_order"],
        expected_profiles,
        name="action-ball evaluator profile_order",
    )

    expected_pins = {
        "curriculum_contract_sha256": preflight["profile_adapter"]["sha256"],
        "arm_catalog_sha256": curriculum_module.ARM_CATALOG_SHA256,
        "scheduler_contract_sha256": (
            curriculum_module.ArmSchedulerConfig().contract_sha256
        ),
        "sampler_sha256": preflight["sampler"]["contract_sha256"],
        "solver_sha256": solver_sha256,
        "policy_contract_sha256": preflight["policy_contract_sha256"],
    }
    for field, expected in expected_pins.items():
        actual = _action_ball_sha256(
            row[field], name=f"action-ball evaluator {field}"
        )
        if actual != expected:
            raise RuntimeError(
                f"action-ball evaluator {field} disagrees with the "
                f"validated launch identity: runtime={actual}, expected={expected}"
            )
    _action_ball_sha256(
        row["attempt_source_contract_sha256"],
        name="action-ball evaluator attempt_source_contract_sha256",
    )
    declared_source_sha256 = _action_ball_sha256(
        row["attempt_source_sha256"],
        name="action-ball evaluator attempt_source_sha256",
    )
    source = _action_ball_repo_relative_source(
        row["attempt_source_path"],
        repo_root=repo_root,
        name="action-ball evaluator attempt_source_path",
        reject_symlinks=True,
    )
    actual_source_sha256 = _sha256_file(str(source))
    if declared_source_sha256 != actual_source_sha256:
        raise RuntimeError(
            "action-ball evaluator attempt source bytes drifted: "
            f"declared={declared_source_sha256}, actual={actual_source_sha256}"
        )

    try:
        normalized = evaluator_module.launch_receipt_document_v4(
            curriculum_contract_sha256=expected_pins[
                "curriculum_contract_sha256"
            ],
            profile_order=tuple(
                curriculum_module.ActionProfileKey(**profile)
                for profile in expected_profiles
            ),
            arm_catalog_sha256=expected_pins[
                "arm_catalog_sha256"
            ],
            scheduler_contract_sha256=expected_pins[
                "scheduler_contract_sha256"
            ],
            sampler_sha256=expected_pins["sampler_sha256"],
            solver_sha256=expected_pins["solver_sha256"],
            policy_contract_sha256=expected_pins[
                "policy_contract_sha256"
            ],
            attempt_source_contract_sha256=row[
                "attempt_source_contract_sha256"
            ],
            attempt_source_path=row["attempt_source_path"],
            attempt_source_sha256=declared_source_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "action-ball V4 evaluator receipt cannot be normalized by code"
        ) from exc
    _action_ball_assert_json_equal(
        row,
        normalized,
        name="action-ball canonical V4 evaluator launch receipt",
    )

    computed_launch_sha256 = _canonical_contract_sha256(row)
    declared = _action_ball_sha256(
        declared_launch_sha256,
        name="action-ball evaluator launch receipt SHA",
    )
    if declared != computed_launch_sha256:
        raise RuntimeError(
            "action-ball evaluator launch receipt SHA mismatch: "
            f"declared={declared}, actual={computed_launch_sha256}"
        )
    trusted = (
        evaluator_module
        .TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256
    )
    if (
        type(trusted) is not frozenset
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in trusted
        )
    ):
        raise RuntimeError(
            "action-ball evaluator code trust set has an invalid shape"
        )
    if declared not in trusted:
        raise RuntimeError(
            "action-ball evaluator launch receipt is not code-pinned; "
            "runtime, manifest, and checkpoint self-authorization are forbidden"
        )

    try:
        authority = (
            evaluator_module.FrozenEvaluatorV4Authority
            .from_trusted_launch_receipt(
                row,
                attempt_source=attempt_source,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "action-ball evaluator launch receipt failed executable "
            "code-rooted authority construction"
        ) from exc
    binding = authority.binding_document()
    expected_binding = {
        "schema_version": evaluator_module.V4_SCHEMA_VERSION,
        "authority_contract_sha256": authority_sha256,
        "launch_receipt_sha256": declared,
        **expected_pins,
        "profile_order": expected_profiles,
        "attempt_source_contract_sha256": row[
            "attempt_source_contract_sha256"
        ],
        "attempt_source_path": row["attempt_source_path"],
        "attempt_source_sha256": declared_source_sha256,
        "source_state_owner_sha256": (
            attempt_source.state_owner_sha256
        ),
        "state_owner_sha256": authority.state_owner_sha256,
    }
    _action_ball_assert_json_equal(
        binding,
        expected_binding,
        name="action-ball evaluator authority binding",
    )
    return {
        "launch_receipt": row,
        "launch_receipt_canonical_sha256": declared,
        "authority_binding": binding,
        "authority_state_owner_sha256": authority.state_owner_sha256,
        "_authority": authority,
    }


def _load_action_ball_evaluator_launch_from_cfg(
    racket_cfg, motion_cfg, *, preflight: dict
) -> dict:
    """Load the complete V4 evaluator/sidecar/drain graph before Gym."""

    relative_path = str(
        getattr(
            racket_cfg,
            "action_ball_evaluator_launch_receipt_path",
            "",
        )
        or ""
    ).strip()
    expected_file_sha256 = str(
        getattr(
            racket_cfg,
            "action_ball_evaluator_launch_receipt_file_sha256",
            "",
        )
        or ""
    ).strip()
    if not relative_path:
        raise _OverrideError(
            "[train.py] formal action-ball training requires "
            "racket.action_ball_evaluator_launch_receipt_path; dependency-"
            "light diagnostic evaluator APIs remain available but may not learn"
        )
    try:
        expected_file_sha256 = _action_ball_sha256(
            expected_file_sha256,
            name="action-ball evaluator launch receipt file SHA",
        )
        repo_root = _action_ball_repo_root(motion_cfg)
        source = _action_ball_repo_relative_source(
            relative_path,
            repo_root=repo_root,
            name="action-ball evaluator launch receipt path",
            reject_symlinks=True,
        )
        raw = source.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise _OverrideError(
            f"[train.py] invalid action-ball evaluator launch receipt: {exc}"
        ) from exc

    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_evaluation_inbox as inbox_protocol,
    )

    inbox_root = pathlib.Path(
        str(
            getattr(
                racket_cfg,
                "action_ball_evaluation_inbox_root",
                "",
            )
            or ""
        ).strip()
    )
    owner_id = str(
        getattr(racket_cfg, "action_ball_evaluation_owner_id", "") or ""
    ).strip()
    run_id = str(
        getattr(racket_cfg, "action_ball_evaluation_run_id", "") or ""
    ).strip()
    interval = getattr(
        racket_cfg, "action_ball_frozen_eval_interval_updates", None
    )
    if (
        not inbox_root.is_absolute()
        or type(interval) is not int
        or interval < 1
    ):
        raise _OverrideError(
            "[train.py] formal action-ball requires an absolute evaluation "
            "inbox root and a positive plain-integer evaluation interval"
        )
    try:
        inbox = inbox_protocol.EvaluationInbox(inbox_root)
        attempt_source = inbox_protocol.FrozenSidecarInboxAttemptSource(
            inbox=inbox,
            owner_id=owner_id,
            run_id=run_id,
        )
    except Exception as exc:
        raise _OverrideError(
            "[train.py] invalid formal action-ball evaluation inbox "
            f"identity: {exc}"
        ) from exc
    actual_file_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise _OverrideError(
            "[train.py] action-ball evaluator launch receipt file SHA "
            f"mismatch: expected={expected_file_sha256}, "
            f"actual={actual_file_sha256}"
        )

    def no_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number {token!r}")
        return value

    def reject_constant(token):
        raise ValueError(f"non-finite JSON constant {token!r}")

    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise _OverrideError(
            "[train.py] action-ball evaluator launch receipt must be "
            f"strict UTF-8 JSON without duplicate keys: {exc}"
        ) from exc
    canonical_sha256 = _canonical_contract_sha256(document)
    try:
        verified = _validate_action_ball_evaluator_launch_receipt(
            document,
            declared_launch_sha256=canonical_sha256,
            preflight=preflight,
            solver_sha256=preflight["solver_profile_sha256"],
            repo_root=repo_root,
            attempt_source=attempt_source,
        )
    except (KeyError, RuntimeError) as exc:
        raise _OverrideError(
            f"[train.py] action-ball evaluator launch is not authorized: {exc}"
        ) from exc
    def load_companion(path_field, sha_field, label):
        companion_relative = str(
            getattr(racket_cfg, path_field, "") or ""
        ).strip()
        companion_sha = str(
            getattr(racket_cfg, sha_field, "") or ""
        ).strip()
        if not companion_relative:
            raise _OverrideError(
                f"[train.py] formal action-ball requires racket.{path_field}"
            )
        try:
            companion_sha = _action_ball_sha256(
                companion_sha, name=f"{label} file SHA"
            )
            companion_path = _action_ball_repo_relative_source(
                companion_relative,
                repo_root=repo_root,
                name=f"{label} path",
                reject_symlinks=True,
            )
            companion_raw = companion_path.read_bytes()
        except (OSError, RuntimeError) as exc:
            raise _OverrideError(
                f"[train.py] invalid {label}: {exc}"
            ) from exc
        observed = hashlib.sha256(companion_raw).hexdigest()
        if observed != companion_sha:
            raise _OverrideError(
                f"[train.py] {label} file SHA mismatch: "
                f"expected={companion_sha}, actual={observed}"
            )
        try:
            companion = json.loads(
                companion_raw.decode("utf-8"),
                object_pairs_hook=no_duplicate_keys,
                parse_float=finite_float,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise _OverrideError(
                f"[train.py] {label} must be strict UTF-8 JSON: {exc}"
            ) from exc
        return {
            "path": companion_relative,
            "file_sha256": observed,
            "document": companion,
        }

    sidecar = load_companion(
        "action_ball_sidecar_launch_receipt_path",
        "action_ball_sidecar_launch_receipt_file_sha256",
        "action-ball frozen-evaluation sidecar launch receipt",
    )
    sidecar_code_relative = (
        "hope_training/whole_body_tracking/scripts/"
        "action_ball_frozen_eval_sidecar.py"
    )
    try:
        sidecar_code = _action_ball_repo_relative_source(
            sidecar_code_relative,
            repo_root=repo_root,
            name="action-ball frozen-evaluation sidecar code",
            reject_symlinks=True,
        )
        sidecar_code_sha = _sha256_file(str(sidecar_code))
        sidecar_content = sidecar["document"]["content"]
        inbox_protocol.validate_sidecar_launch_document(
            sidecar["document"],
            actual_sidecar_code_sha256=sidecar_code_sha,
            backend_contract_sha256=(
                inbox_protocol.FORMAL_ISAAC_BACKEND_CONTRACT_SHA256
            ),
            require_trust=True,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise _OverrideError(
            "[train.py] action-ball sidecar launch is not authorized: "
            f"{exc}"
        ) from exc
    consumer_code_relative = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    )
    try:
        consumer_code = _action_ball_repo_relative_source(
            consumer_code_relative,
            repo_root=repo_root,
            name="action-ball evaluation consumer code",
            reject_symlinks=True,
        )
        coordinator = (
            inbox_protocol.FrozenEvaluationInboxCoordinator(
                inbox=inbox,
                owner_id=owner_id,
                run_id=run_id,
                sidecar_launch_sha256=sidecar["document"][
                    "content_sha256"
                ],
                consumer_code_sha256=_sha256_file(
                    str(consumer_code)
                ),
                evaluator_authority=verified["_authority"],
            )
        )
    except Exception as exc:
        raise _OverrideError(
            "[train.py] action-ball evaluation coordinator cannot be "
            f"constructed: {exc}"
        ) from exc

    drain = load_companion(
        "action_ball_drain_reset_launch_receipt_path",
        "action_ball_drain_reset_launch_receipt_file_sha256",
        "action-ball drain/reset launch receipt",
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_curriculum as curriculum_module,
    )

    class _PreGymDrainSource:
        def binding_document(self):
            row = drain["document"]
            return {
                field: row[field]
                for field in (
                    "runtime_source_contract_sha256",
                    "runtime_source_path",
                    "runtime_source_sha256",
                    "broker_contract_sha256",
                    "attempt_pool_contract_sha256",
                    "task_receipt_pool_contract_sha256",
                    "env_reset_contract_sha256",
                )
            }

    try:
        drain_authority = (
            curriculum_module.DrainResetAuthority
            .from_trusted_launch_receipt(
                drain["document"],
                runtime_source=_PreGymDrainSource(),
            )
        )
        drain_authority.assert_binding(
            curriculum_contract_sha256=preflight[
                "profile_adapter"
            ]["sha256"],
            profile_order=tuple(
                curriculum_module.ActionProfileKey(
                    action_uid=binding["action_uid"],
                    profile_sha256=binding[
                        "sampling_profile_sha256"
                    ],
                    mobility=preflight["mobility_mode"],
                )
                for binding in preflight["action_bindings"]
            ),
            arm_catalog_sha256=curriculum_module.ARM_CATALOG_SHA256,
            scheduler_contract_sha256=(
                curriculum_module.ArmSchedulerConfig()
                .contract_sha256
            ),
            sampler_sha256=preflight["sampler"][
                "contract_sha256"
            ],
            solver_sha256=preflight["solver_profile_sha256"],
            policy_contract_sha256=preflight[
                "policy_contract_sha256"
            ],
        )
    except Exception as exc:
        raise _OverrideError(
            "[train.py] action-ball drain/reset launch is not authorized: "
            f"{exc}"
        ) from exc

    return {
        "schema_version": 4,
        "path": relative_path,
        "file_sha256": actual_file_sha256,
        "attempt_source_state_owner_sha256": (
            attempt_source.state_owner_sha256
        ),
        "coordinator_state_owner_sha256": (
            coordinator.state_owner_sha256
        ),
        "inbox_root": str(inbox_root),
        "inbox_owner_id": owner_id,
        "inbox_run_id": run_id,
        "sidecar_launch": sidecar,
        "sidecar_launch_receipt_path": sidecar["path"],
        "sidecar_launch_receipt_file_sha256": sidecar[
            "file_sha256"
        ],
        "sidecar_launch_receipt_content_sha256": sidecar[
            "document"
        ]["content_sha256"],
        "sidecar_code_path": sidecar_code_relative,
        "sidecar_code_sha256": sidecar_code_sha,
        "drain_reset_launch": {
            **drain,
            "launch_receipt_canonical_sha256": (
                curriculum_module._canonical_sha256(
                    drain["document"]
                )
            ),
            "runtime_source_binding": (
                _PreGymDrainSource().binding_document()
            ),
            "authority_state_owner_sha256": (
                drain_authority.state_owner_sha256
            ),
        },
        **{
            key: value
            for key, value in verified.items()
            if not key.startswith("_")
        },
    }


def _validate_action_ball_visible_motion_identity(
    value, *, motion_cfg, action_count: int
) -> dict:
    """Validate supplemental canonical-bank facts.

    This mapping is useful checkpoint identity, but is deliberately not motion
    authorization.  Formal authority is accepted only from MotionCommand's
    separate opaque admission receipt.
    """

    keys = (
        "canonical_registry_sha256",
        "canonical_registry_alignment_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "canonical_source_manifest_sha256_per_clip",
        "canonical_build_manifest_sha256_per_clip",
        "canonical_applicability_manifest_sha256_per_clip",
        "canonical_evidence_level_per_clip",
        "canonical_evidence_manifest_sha256_per_clip",
        "canonical_training_config_sha256_per_clip",
        "canonical_adoption_manifest_sha256_per_clip",
    )
    identity = _action_ball_exact_dict(
        value, keys, name="action-ball runtime motion_admission identity"
    )
    for field in keys[:4]:
        actual = _action_ball_sha256(
            identity[field],
            name=f"action-ball runtime motion_admission.{field}",
        )
        expected = str(getattr(motion_cfg, field, "") or "")
        if actual != expected:
            raise RuntimeError(
                "action-ball runtime motion admission identity disagrees "
                f"with motion.{field}: runtime={actual}, expected={expected}"
            )
    sha_columns = (
        "canonical_source_manifest_sha256_per_clip",
        "canonical_build_manifest_sha256_per_clip",
        "canonical_applicability_manifest_sha256_per_clip",
        "canonical_evidence_manifest_sha256_per_clip",
        "canonical_training_config_sha256_per_clip",
        "canonical_adoption_manifest_sha256_per_clip",
    )
    for field in sha_columns:
        values = identity[field]
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, (list, tuple))
            or len(values) != action_count
        ):
            raise RuntimeError(
                f"action-ball runtime motion_admission.{field} must contain "
                f"exactly {action_count} entries"
            )
        for index, digest in enumerate(values):
            _action_ball_sha256(
                digest,
                name=(
                    "action-ball runtime motion_admission."
                    f"{field}[{index}]"
                ),
            )
    evidence = identity["canonical_evidence_level_per_clip"]
    if (
        isinstance(evidence, (str, bytes))
        or not isinstance(evidence, (list, tuple))
        or len(evidence) != action_count
        or any(type(level) is not str or not level for level in evidence)
    ):
        raise RuntimeError(
            "action-ball runtime motion_admission."
            "canonical_evidence_level_per_clip must contain one non-empty "
            "string per action"
        )
    return identity


def _validate_action_ball_runtime_hard_contract(
    runtime_contract,
    *,
    preflight: dict,
    racket_cfg,
    motion_cfg,
    expected_runtime_contract_sha256: str,
) -> dict:
    """Cross-check runtime truth against an independently built preflight."""

    preflight = _action_ball_exact_dict(
        preflight,
        (
            "schema_version",
            "manifest",
            "mobility_mode",
            "action_order",
            "action_uids",
            "ready_root_z_by_slot_m",
            "action_bindings",
            "prototype",
            "profile_adapter",
            "sampler",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "curriculum",
            "holdout",
            "fixed_direction",
            "initial_episode_length_randomization",
            "policy_contract_sha256",
            "evaluator_launch",
            "sha256",
        ),
        name="action-ball launch preflight",
    )
    if type(preflight["schema_version"]) is not int or preflight[
        "schema_version"
    ] != 1:
        raise RuntimeError("action-ball launch preflight schema_version must be 1")
    if preflight["fixed_direction"] is not True:
        raise RuntimeError(
            "action-ball launch preflight fixed_direction must be true"
        )
    if preflight["initial_episode_length_randomization"] is not False:
        raise RuntimeError(
            "action-ball launch preflight must disable initial episode "
            "length randomization"
        )
    ready_root_z = preflight["ready_root_z_by_slot_m"]
    if (
        type(ready_root_z) is not list
        or len(ready_root_z) != len(preflight["action_order"])
        or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            for value in ready_root_z
        )
    ):
        raise RuntimeError(
            "action-ball launch preflight ready_root_z_by_slot_m must "
            "contain one finite number per action"
        )
    sampler_preflight = _action_ball_exact_dict(
        preflight["sampler"],
        (
            "contract_sha256",
            "arm_catalog_sha256",
            "seed",
            "pool_refill_rows",
        ),
        name="action-ball launch preflight sampler",
    )
    for field in ("contract_sha256", "arm_catalog_sha256"):
        _action_ball_sha256(
            sampler_preflight[field],
            name=f"action-ball launch preflight sampler.{field}",
        )
    from whole_body_tracking.tasks.tracking.mdp.action_ball_sampling import (
        ARM_CATALOG_SHA256,
    )

    if sampler_preflight["arm_catalog_sha256"] != ARM_CATALOG_SHA256:
        raise RuntimeError(
            "action-ball launch preflight arm catalog disagrees with the "
            "executable sampler"
        )
    evaluator_launch_preflight = _action_ball_exact_dict(
        preflight["evaluator_launch"],
        ("path", "file_sha256"),
        name="action-ball launch preflight evaluator_launch",
    )
    if (
        type(evaluator_launch_preflight["path"]) is not str
        or not evaluator_launch_preflight["path"]
    ):
        raise RuntimeError(
            "action-ball launch preflight evaluator_launch.path must be "
            "non-empty"
        )
    _action_ball_sha256(
        evaluator_launch_preflight["file_sha256"],
        name="action-ball launch preflight evaluator_launch.file_sha256",
    )
    preflight_declared = _action_ball_sha256(
        preflight["sha256"], name="action-ball launch preflight sha256"
    )
    preflight_unsigned = dict(preflight)
    del preflight_unsigned["sha256"]
    preflight_computed = _canonical_contract_sha256(preflight_unsigned)
    if preflight_declared != preflight_computed:
        raise RuntimeError(
            "action-ball launch preflight SHA mismatch: "
            f"declared={preflight_declared}, actual={preflight_computed}"
        )

    contract = _action_ball_exact_dict(
        runtime_contract,
        (
            "schema_version",
            "kind",
            "manifest",
            "mobility_mode",
            "action_order",
            "action_uids",
            "bindings",
            "prototype",
            "profiles",
            "sampling",
            "timing",
            "reference_guard",
            "solver",
            "physics",
            "domain_authority",
            "mutable_state_owner",
            "curriculum",
            "evaluator_authority",
            "runtime",
            "motion_admission",
            "canonical_sha256",
        ),
        name="action-ball runtime hard contract",
    )
    if type(contract["schema_version"]) is not int or contract[
        "schema_version"
    ] != 1:
        raise RuntimeError(
            "action-ball runtime hard contract schema_version must be 1"
        )
    if (
        contract["kind"]
        != "whole_body_tracking.RacketTargetCommand.action_ball_hard_contract"
    ):
        raise RuntimeError("action-ball runtime hard contract kind drifted")
    declared = _action_ball_sha256(
        contract["canonical_sha256"],
        name="action-ball runtime hard contract canonical_sha256",
    )
    unsigned = dict(contract)
    del unsigned["canonical_sha256"]
    computed = _canonical_contract_sha256(unsigned)
    if declared != computed:
        raise RuntimeError(
            "action-ball runtime hard contract canonical_sha256 mismatch: "
            f"declared={declared}, actual={computed}"
        )
    _validate_action_ball_reference_guard_contract(
        contract["reference_guard"],
        racket_cfg=racket_cfg,
    )

    _action_ball_assert_json_equal(
        contract["manifest"],
        preflight["manifest"],
        name="action-ball runtime manifest",
    )
    _action_ball_assert_json_equal(
        contract["mobility_mode"],
        preflight["mobility_mode"],
        name="action-ball runtime mobility_mode",
    )
    _action_ball_assert_json_equal(
        contract["action_order"],
        preflight["action_order"],
        name="action-ball runtime action_order",
    )
    _action_ball_assert_json_equal(
        contract["action_uids"],
        preflight["action_uids"],
        name="action-ball runtime action_uids",
    )
    expected_bindings = [
        {
            "action_uid": row["action_uid"],
            "action_slot": row["action_slot"],
            "motion_path": row["motion_path"],
            "motion_sha256": row["motion_sha256"],
            "profile_sha256": row["sampling_profile_sha256"],
        }
        for row in preflight["action_bindings"]
    ]
    if type(contract["bindings"]) is not list:
        raise RuntimeError("action-ball runtime bindings must be a list")
    for index, row in enumerate(contract["bindings"]):
        _action_ball_exact_dict(
            row,
            (
                "action_uid",
                "action_slot",
                "motion_path",
                "motion_sha256",
                "profile_sha256",
            ),
            name=f"action-ball runtime bindings[{index}]",
        )
    _action_ball_assert_json_equal(
        contract["bindings"],
        expected_bindings,
        name="action-ball runtime bindings",
    )
    _action_ball_assert_json_equal(
        contract["prototype"],
        preflight["prototype"],
        name="action-ball runtime prototype",
    )
    expected_profiles = {
        "adapter_contract_sha256": preflight["profile_adapter"]["sha256"],
        "profile_sha256": [
            row["sampling_profile_sha256"]
            for row in preflight["action_bindings"]
        ],
        "arm_catalog_sha256": preflight["sampler"][
            "arm_catalog_sha256"
        ],
        "sampler_contract_sha256": preflight["sampler"][
            "contract_sha256"
        ],
    }
    _action_ball_assert_json_equal(
        contract["profiles"],
        expected_profiles,
        name="action-ball runtime profiles",
    )

    expected_sampling = {
        "action_ball_seed": preflight["sampler"]["seed"],
        "pool_refill_rows": preflight["sampler"]["pool_refill_rows"],
        "balanced_clip_sampling": bool(
            getattr(motion_cfg, "balanced_clip_sampling", False)
        ),
        "balanced_clip_sampling_seed": getattr(
            motion_cfg, "balanced_clip_sampling_seed", None
        ),
        "external_overdraw_multiplier": float(
            getattr(racket_cfg, "cq_overdraw")
        ),
        "maximum_external_proposal_rounds": int(
            getattr(racket_cfg, "cq_max_redraw_rounds")
        ),
    }
    _action_ball_assert_json_equal(
        contract["sampling"],
        expected_sampling,
        name="action-ball runtime sampling",
    )

    timing = _action_ball_exact_dict(
        contract["timing"],
        (
            "authority",
            "policy_dt_s",
            "attempt_close_margin_s",
            "episode_length_s",
            "time_to_strike_source",
            "legacy_motion_time_owners",
        ),
        name="action-ball runtime timing",
    )
    for field in (
        "policy_dt_s",
        "attempt_close_margin_s",
        "episode_length_s",
    ):
        if (
            type(timing[field]) not in (int, float)
            or not math.isfinite(float(timing[field]))
            or float(timing[field]) <= 0.0
        ):
            raise RuntimeError(
                f"action-ball runtime timing.{field} must be finite and "
                "positive"
            )
    from whole_body_tracking.tasks.tracking.mdp.action_ball_runtime import (
        TASK_RECEIPT_TIMING_AUTHORITY,
    )

    if (
        timing["authority"] != TASK_RECEIPT_TIMING_AUTHORITY
        or timing["time_to_strike_source"]
        != "MotionCommand.action_ball_time_to_contact_remaining_s"
        or float(timing["policy_dt_s"])
        != float(timing["attempt_close_margin_s"])
        or float(timing["episode_length_s"])
        < float(timing["policy_dt_s"])
    ):
        raise RuntimeError(
            "action-ball runtime timing authority/dt/horizon contract drifted"
        )
    legacy_motion_time_owners = _action_ball_exact_dict(
        timing["legacy_motion_time_owners"],
        (
            "hold_steps_range",
            "stand_start_min_hold",
            "post_swing_min_hold",
            "stagger_initial_clock",
            "speed_scale_range",
            "speed_scale_per_clip",
            "planner_revision_enabled",
        ),
        name="action-ball runtime timing.legacy_motion_time_owners",
    )
    speed_per_clip = getattr(motion_cfg, "speed_scale_per_clip", None)
    expected_legacy_motion_time_owners = {
        "hold_steps_range": [
            int(value)
            for value in (
                getattr(motion_cfg, "hold_steps_range", ()) or ()
            )
        ],
        "stand_start_min_hold": int(
            getattr(motion_cfg, "stand_start_min_hold")
        ),
        "post_swing_min_hold": int(
            getattr(motion_cfg, "post_swing_min_hold")
        ),
        "stagger_initial_clock": bool(
            getattr(motion_cfg, "stagger_initial_clock")
        ),
        "speed_scale_range": [
            float(value)
            for value in (
                getattr(motion_cfg, "speed_scale_range", ()) or ()
            )
        ],
        "speed_scale_per_clip": (
            None
            if speed_per_clip is None
            else [float(value) for value in speed_per_clip]
        ),
        "planner_revision_enabled": bool(
            getattr(motion_cfg, "planner_revision_enabled", False)
        ),
    }
    _action_ball_assert_json_equal(
        legacy_motion_time_owners,
        expected_legacy_motion_time_owners,
        name="action-ball runtime legacy motion timing owners",
    )

    physics = _action_ball_content_receipt(
        contract["physics"], name="action-ball runtime physics"
    )
    solver = _action_ball_content_receipt(
        contract["solver"], name="action-ball runtime solver"
    )
    if physics["sha256"] != preflight["physics_profile_sha256"]:
        raise RuntimeError(
            "action-ball runtime physics payload does not match the "
            "manifest-pinned physics profile"
        )
    if solver["sha256"] != preflight["solver_profile_sha256"]:
        raise RuntimeError(
            "action-ball runtime solver payload does not match the "
            "manifest-pinned solver profile"
        )
    if (
        solver["payload"].get("physics_profile_sha256")
        != physics["sha256"]
    ):
        raise RuntimeError(
            "action-ball runtime solver payload is not bound to the "
            "validated physics payload"
        )

    repo_root = _action_ball_repo_root(motion_cfg)
    domain_authority = _action_ball_content_receipt(
        contract["domain_authority"],
        name="action-ball runtime domain_authority",
    )
    domain_payload = _action_ball_exact_dict(
        domain_authority["payload"],
        (
            "schema_version",
            "kind",
            "implementation_source_sha256",
            "manifest_sha256",
            "adapter_contract_sha256",
            "action_uids",
            "profile_sha256",
            "mobility_mode",
            "curriculum_config",
            "policy_contract_sha256",
            "schedule",
        ),
        name="action-ball runtime domain_authority.payload",
    )
    if (
        type(domain_payload["schema_version"]) is not int
        or domain_payload["schema_version"] < 1
        or domain_payload["kind"]
        != "whole_body_tracking.action_ball.domain_claim_authority"
    ):
        raise RuntimeError(
            "action-ball runtime domain authority schema/kind drifted"
        )
    domain_source_map = _validate_action_ball_mdp_source_map(
        domain_payload["implementation_source_sha256"],
        expected_names=(
            "hope_commands.py",
            "action_ball_curriculum.py",
            "action_ball_runtime.py",
        ),
        repo_root=repo_root,
        name="action-ball runtime domain authority sources",
    )
    expected_domain_payload = {
        "schema_version": domain_payload["schema_version"],
        "kind": "whole_body_tracking.action_ball.domain_claim_authority",
        "implementation_source_sha256": domain_source_map,
        "manifest_sha256": preflight["manifest"]["file_sha256"],
        "adapter_contract_sha256": preflight["profile_adapter"]["sha256"],
        "action_uids": preflight["action_uids"],
        "profile_sha256": [
            row["sampling_profile_sha256"]
            for row in preflight["action_bindings"]
        ],
        "mobility_mode": preflight["mobility_mode"],
        "curriculum_config": preflight["curriculum"]["config"],
        "policy_contract_sha256": preflight["policy_contract_sha256"],
        "schedule": {
            "claim_barrier": "true_reset_only",
            "domain_source": (
                "frozen_ActionBallCurriculum.expected_domains"
            ),
            "selection": "per_action_round_robin",
            "training_selector": False,
            "live_rollout_updates_curriculum": False,
        },
    }
    _action_ball_assert_json_equal(
        domain_payload,
        expected_domain_payload,
        name="action-ball runtime domain authority payload",
    )

    mutable_state_owner = _action_ball_exact_dict(
        contract["mutable_state_owner"],
        (
            "schema_version",
            "state_owner_sha256",
            "protocol_views",
            "checkpoint_state_is_mutable",
            "mutable_state_sha256_is_not_a_hard_contract_pin",
        ),
        name="action-ball runtime mutable_state_owner",
    )
    if (
        type(mutable_state_owner["schema_version"]) is not int
        or mutable_state_owner["schema_version"] < 1
    ):
        raise RuntimeError(
            "action-ball runtime mutable-state schema_version must be a "
            "positive integer"
        )
    expected_state_owner_sha256 = _canonical_contract_sha256(
        {
            "schema_version": mutable_state_owner["schema_version"],
            "kind": (
                "whole_body_tracking.RacketTargetCommand."
                "action_ball_mutable_state_owner"
            ),
            "action_uids": preflight["action_uids"],
            "sampler_contract_sha256": preflight["sampler"][
                "contract_sha256"
            ],
            "domain_authority_contract_sha256": domain_authority["sha256"],
            "solver_contract_sha256": solver["sha256"],
        }
    )
    expected_mutable_state_owner = {
        "schema_version": mutable_state_owner["schema_version"],
        "state_owner_sha256": expected_state_owner_sha256,
        "protocol_views": [
            "domain_claim_authority",
            "birth_provider",
            "task_solver",
        ],
        "checkpoint_state_is_mutable": True,
        "mutable_state_sha256_is_not_a_hard_contract_pin": True,
    }
    _action_ball_assert_json_equal(
        mutable_state_owner,
        expected_mutable_state_owner,
        name="action-ball runtime mutable-state owner",
    )

    expected_curriculum = {
        "config": preflight["curriculum"]["config"],
        "policy_contract_sha256": preflight["policy_contract_sha256"],
        "frozen_checkpoint_evidence_required": True,
        "live_rollout_advances_curriculum": False,
    }
    _action_ball_assert_json_equal(
        contract["curriculum"],
        expected_curriculum,
        name="action-ball runtime curriculum",
    )

    evaluator_authority = _action_ball_exact_dict(
        contract["evaluator_authority"],
        (
            "authority_contract_sha256",
            "trusted_launch_receipt_sha256",
            "evaluator_launch_receipt_path",
            "evaluator_launch_receipt_file_sha256",
            "evaluator_launch_receipt",
            "launch_receipt_canonical_sha256",
            "authority_binding",
            "authority_state_owner_sha256",
            "attempt_source_state_owner_sha256",
            "coordinator_state_owner_sha256",
            "inbox_root",
            "inbox_owner_id",
            "inbox_run_id",
            "sidecar_launch_receipt_path",
            "sidecar_launch_receipt_file_sha256",
            "sidecar_launch_receipt_content_sha256",
            "sidecar_code_path",
            "sidecar_code_sha256",
            "drain_reset",
            "evaluation_interval_updates",
            "formal_authority_available",
            "formal_launch_requires_code_pinned_receipt",
            "runtime_or_manifest_may_self_authorize",
        ),
        name="action-ball runtime evaluator_authority",
    )
    verified_evaluator_launch = _load_action_ball_evaluator_launch_from_cfg(
        racket_cfg,
        motion_cfg,
        preflight=preflight,
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_evaluation as evaluator_module,
    )

    expected_evaluator_authority = {
        "authority_contract_sha256": (
            evaluator_module.FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
        ),
        "trusted_launch_receipt_sha256": sorted(
            evaluator_module
            .TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256
        ),
        "evaluator_launch_receipt_path": verified_evaluator_launch["path"],
        "evaluator_launch_receipt_file_sha256": (
            verified_evaluator_launch["file_sha256"]
        ),
        "evaluator_launch_receipt": verified_evaluator_launch[
            "launch_receipt"
        ],
        "launch_receipt_canonical_sha256": verified_evaluator_launch[
            "launch_receipt_canonical_sha256"
        ],
        "authority_binding": verified_evaluator_launch["authority_binding"],
        "authority_state_owner_sha256": verified_evaluator_launch[
            "authority_state_owner_sha256"
        ],
        "attempt_source_state_owner_sha256": (
            verified_evaluator_launch[
                "attempt_source_state_owner_sha256"
            ]
        ),
        "coordinator_state_owner_sha256": (
            verified_evaluator_launch[
                "coordinator_state_owner_sha256"
            ]
        ),
        "inbox_root": verified_evaluator_launch["inbox_root"],
        "inbox_owner_id": verified_evaluator_launch["inbox_owner_id"],
        "inbox_run_id": verified_evaluator_launch["inbox_run_id"],
        "sidecar_launch_receipt_path": (
            verified_evaluator_launch[
                "sidecar_launch_receipt_path"
            ]
        ),
        "sidecar_launch_receipt_file_sha256": (
            verified_evaluator_launch[
                "sidecar_launch_receipt_file_sha256"
            ]
        ),
        "sidecar_launch_receipt_content_sha256": (
            verified_evaluator_launch[
                "sidecar_launch_receipt_content_sha256"
            ]
        ),
        "sidecar_code_path": verified_evaluator_launch[
            "sidecar_code_path"
        ],
        "sidecar_code_sha256": verified_evaluator_launch[
            "sidecar_code_sha256"
        ],
        "drain_reset": verified_evaluator_launch[
            "drain_reset_launch"
        ],
        "evaluation_interval_updates": int(
            racket_cfg.action_ball_frozen_eval_interval_updates
        ),
        "formal_authority_available": True,
        "formal_launch_requires_code_pinned_receipt": True,
        "runtime_or_manifest_may_self_authorize": False,
    }
    _action_ball_assert_json_equal(
        evaluator_authority,
        expected_evaluator_authority,
        name="action-ball runtime evaluator authority",
    )

    expected_runtime_sha = _action_ball_sha256(
        expected_runtime_contract_sha256,
        name="action-ball executable runtime contract SHA",
    )
    expected_registry = _canonical_contract_sha256(
        {
            "runtime_contract_sha256": expected_runtime_sha,
            "pins": {
                "manifest_sha256": preflight["manifest"]["file_sha256"],
                "sampler_sha256": preflight["sampler"]["contract_sha256"],
                "domain_authority_sha256": domain_authority["sha256"],
                "physics_sha256": physics["sha256"],
                "solver_sha256": solver["sha256"],
            },
            "mobility_mode": preflight["mobility_mode"],
            "bindings": expected_bindings,
        }
    )
    runtime_source_map = _validate_action_ball_mdp_source_map(
        _action_ball_exact_dict(
            contract["runtime"],
            (
                "runtime_contract_sha256",
                "registry_sha256",
                "implementation_source_sha256",
                "fixed_direction",
                "wrap_teleport",
            ),
            name="action-ball runtime protocol identity",
        )["implementation_source_sha256"],
        expected_names=(
            "hope_commands.py",
            "action_ball_curriculum.py",
            "action_ball_evaluation.py",
            "action_ball_manifest.py",
            "action_ball_profile_adapter.py",
            "action_ball_reference_guard.py",
            "action_ball_runtime.py",
            "action_ball_sampling.py",
            "continuous_questions.py",
            "racket_contact_geometry.py",
            "stroke_adapt_torch.py",
            "virtual_ball.py",
        ),
        repo_root=repo_root,
        name="action-ball runtime implementation sources",
    )
    expected_runtime = {
        "runtime_contract_sha256": expected_runtime_sha,
        "registry_sha256": expected_registry,
        "implementation_source_sha256": runtime_source_map,
        "fixed_direction": True,
        "wrap_teleport": False,
    }
    _action_ball_assert_json_equal(
        contract["runtime"],
        expected_runtime,
        name="action-ball runtime protocol identity",
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_runtime as runtime_module,
    )

    _validate_action_ball_motion_admission_receipt(
        contract["motion_admission"],
        preflight=preflight,
        motion_cfg=motion_cfg,
        expected_runtime_contract_sha256=expected_runtime_sha,
        expected_broker_state_schema_version=(
            runtime_module.BROKER_STATE_SCHEMA_VERSION
        ),
        expected_broker_registry_sha256=expected_registry,
        expected_provider_state_owner_sha256=mutable_state_owner[
            "state_owner_sha256"
        ],
    )
    return contract


def _validate_action_ball_policy_recipe(
    preflight: dict, agent_cfg, policy_bootstrap: dict | None = None
) -> dict:
    """Bind the manifest policy identity to the exact composed PPO recipe."""

    recipe = _action_ball_agent_recipe(
        agent_cfg, policy_bootstrap=policy_bootstrap
    )
    configured = _action_ball_sha256(
        preflight["policy_contract_sha256"],
        name="action-ball policy contract SHA",
    )
    if configured != recipe["sha256"]:
        raise RuntimeError(
            "action-ball policy contract SHA does not match the exact "
            "post-compose PPO runner/policy/algorithm recipe: "
            f"configured={configured}, actual={recipe['sha256']}"
        )
    return recipe


_ACTION_BALL_FINITE_QDES_PROJECTION_FACT = (
    "finite_preclamp_qdes_projection_enabled"
)


def _require_action_ball_finite_qdes_projection_fact(
    runtime_facts: dict, *, action_ball_enabled: bool
) -> None:
    """Keep constrained-action semantics inside the immutable run identity."""

    if type(action_ball_enabled) is not bool:
        raise RuntimeError("action_ball_enabled must be an exact boolean")
    value_present = _ACTION_BALL_FINITE_QDES_PROJECTION_FACT in runtime_facts
    value = runtime_facts.get(_ACTION_BALL_FINITE_QDES_PROJECTION_FACT)
    if action_ball_enabled:
        if value is not True:
            raise RuntimeError(
                "ActionBall requires the instantiated finite pre-clamp q_des "
                "projection runtime fact to be exact true"
            )
    elif value_present:
        raise RuntimeError(
            "finite pre-clamp q_des projection is ActionBall-only"
        )


def _build_training_hard_contract(
    env,
    actor_contract,
    effective_reward_receipt=None,
    agent_cfg=None,
    action_set_identity=None,
    action_ball_policy_bootstrap=None,
) -> dict:
    """Immutable actor/task facts that must match across a checkpoint resume.

    The complete post-override effective reward recipe is content-addressed here
    so a nominal reward-pack label cannot hide the weights/params that actually
    reached Isaac.  Geometry, command meaning, clip identity, action
    processing, and every field that can move a strike/reveal/deadline or
    actor-visible target in time are also immutable.
    """
    from whole_body_tracking.utils.training_contract import (
        TRAINING_CONTRACT_SCHEMA_VERSION,
        runtime_execution_facts,
        validate_action_ball_action_set_runtime_identity,
    )

    from whole_body_tracking.utils.effective_reward_recipe import (
        build_effective_reward_receipt,
    )

    env_cfg = env.cfg
    runtime_reward_receipt = build_effective_reward_receipt(env_cfg)
    if (
        effective_reward_receipt is not None
        and effective_reward_receipt != runtime_reward_receipt
    ):
        raise RuntimeError(
            "effective reward recipe changed between pre-gym composition and "
            "runtime hard-contract capture"
        )
    effective_reward_receipt = runtime_reward_receipt
    motion_cmd = env.command_manager.get_term("motion")
    motion = motion_cmd.cfg
    try:
        racket_cmd = env.command_manager.get_term("racket_target")
    except KeyError:
        racket_cmd = None
    racket = None if racket_cmd is None else racket_cmd.cfg
    task_first_contract = None
    action_ball_contract = None
    action_ball_ppo_recipe = None
    if racket is not None and str(getattr(racket, "target_mode", "")) == "task_first":
        command_manager = getattr(env, "command_manager", None)
        active_names = tuple(
            str(name)
            for name in getattr(command_manager, "active_terms", ())
        )
        if (
            command_manager is None
            or len(active_names) != len(set(active_names))
            or not {"motion", "racket_target"}.issubset(set(active_names))
        ):
            raise RuntimeError(
                "task-first requires unique active motion and racket_target "
                f"command terms; got {active_names!r}"
            )
        non_explicit = []
        for term_name in active_names:
            term = command_manager.get_term(term_name)
            getter = getattr(term, "exact_resume_state_dict", None)
            loader = getattr(term, "load_exact_resume_state_dict", None)
            if not callable(getter) or not callable(loader):
                non_explicit.append(term_name)
        if non_explicit:
            raise RuntimeError(
                "task-first requires explicit exact-resume hooks on every "
                "active command term; missing "
                + ", ".join(non_explicit)
            )
        termination_manager = getattr(
            getattr(racket_cmd, "_env", env), "termination_manager", None
        )
        active_terminations = set(
            str(name)
            for name in getattr(
                termination_manager, "active_terms", ()
            )
        )
        required_terminations = {
            "base_fell_tilt",
            "base_too_low",
            "robot_hit_table",
        }
        if (
            termination_manager is None
            or not required_terminations.issubset(active_terminations)
        ):
            raise RuntimeError(
                "task-first instantiated termination manager is missing "
                "unsafe truth channels: "
                f"{sorted(required_terminations - active_terminations)}"
            )
        task_first_contract = _task_first_manifest_contract(
            racket, motion_cmd.cfg, env_cfg
        )
        runtime_contract_fn = getattr(
            racket_cmd, "task_first_hard_contract", None
        )
        if not callable(runtime_contract_fn):
            raise RuntimeError(
                "runtime task-first command is missing the mandatory "
                "task_first_hard_contract() implementation"
            )
        runtime_task_first = runtime_contract_fn()
        if runtime_task_first != task_first_contract:
            raise RuntimeError(
                "runtime task-first command hard contract disagrees with "
                "the byte-pinned manifest/config contract"
            )
        racket_mode = str(getattr(racket_cmd, "_racket_mode", ""))
        wrist_index = getattr(racket_cmd, "_wrist_body_index", None)
        racket_index = getattr(racket_cmd, "_racket_body_index", None)
        if (
            racket_mode != "wrist_offset"
            or type(wrist_index) is not int
            or wrist_index < 0
            or type(racket_index) is not int
            or racket_index != -1
        ):
            raise RuntimeError(
                "task-first v1 requires the reviewed wrist_offset physical "
                "paddle-site resolution; the instantiated asset resolved "
                f"mode={racket_mode!r}, wrist_index={wrist_index!r}, "
                f"racket_index={racket_index!r}"
            )
        task_first_contract["resolved_racket_kinematics"] = {
            "mode": racket_mode,
            "source_body_name": str(racket.wrist_body_name),
            "source_body_index": wrist_index,
            "racket_body_index": racket_index,
        }
    if (
        racket is not None
        and str(getattr(racket, "target_mode", "")) == "action_ball"
    ):
        command_manager = getattr(env, "command_manager", None)
        active_names = tuple(
            str(name)
            for name in getattr(command_manager, "active_terms", ())
        )
        if (
            command_manager is None
            or len(active_names) != len(set(active_names))
            or not {"motion", "racket_target"}.issubset(set(active_names))
        ):
            raise RuntimeError(
                "action-ball requires unique active motion and racket_target "
                f"command terms; got {active_names!r}"
            )
        runtime_contract_fn = getattr(
            racket_cmd, "action_ball_hard_contract", None
        )
        if not callable(runtime_contract_fn):
            raise RuntimeError(
                "runtime action-ball command is missing the mandatory "
                "action_ball_hard_contract() implementation"
            )
        # This call performs the deferred cross-command binding now that
        # Isaac has attached the completed CommandManager.  That binding
        # installs the exact-resume hooks checked immediately below.
        runtime_action_ball = runtime_contract_fn()
        if runtime_action_ball is None:
            raise RuntimeError(
                "runtime action-ball command returned no hard contract"
            )
        if type(runtime_action_ball) is not dict:
            raise RuntimeError(
                "runtime action-ball hard contract must be a plain mapping"
            )
        _validate_action_ball_reference_guard_contract(
            runtime_action_ball.get("reference_guard"),
            racket_cfg=racket,
        )
        non_explicit = []
        for term_name in active_names:
            term = command_manager.get_term(term_name)
            getter = getattr(term, "exact_resume_state_dict", None)
            loader = getattr(term, "load_exact_resume_state_dict", None)
            if not callable(getter) or not callable(loader):
                non_explicit.append(term_name)
        if non_explicit:
            raise RuntimeError(
                "action-ball requires explicit exact-resume hooks on every "
                "active command term; missing "
                + ", ".join(non_explicit)
            )
        termination_manager = getattr(
            getattr(racket_cmd, "_env", env), "termination_manager", None
        )
        active_terminations = set(
            str(name)
            for name in getattr(termination_manager, "active_terms", ())
        )
        required_terminations = {
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "base_fell_tilt",
            "base_too_low",
            "robot_hit_table",
            "joint_qdes_forbidden",
            "joint_actual_forbidden",
        }
        if (
            termination_manager is None
            or not required_terminations.issubset(active_terminations)
        ):
            raise RuntimeError(
                "action-ball instantiated termination manager is missing "
                "unsafe truth channels: "
                f"{sorted(required_terminations - active_terminations)}"
            )

        preflight = _action_ball_preflight_contract(
            racket,
            motion_cmd.cfg,
            policy_dt_s=float(env.step_dt),
        )
        from whole_body_tracking.tasks.tracking.mdp.action_ball_runtime import (
            RUNTIME_CONTRACT_SHA256,
        )

        if getattr(
            racket, "action_ball_diagnostic_unauthorized", False
        ) is True:
            # Franco 2026-07-28 diagnostic bypass: the formal cross-check
            # cannot bind without an admission/receipt chain; require the
            # brand on the runtime hard contract instead so the bypass can
            # never be silent.
            if runtime_action_ball.get("diagnostic_unauthorized") is not True:
                raise RuntimeError(
                    "diagnostic action-ball run produced an unbranded "
                    "runtime hard contract"
                )
            print(
                "[train.py] WARN diagnostic_unauthorized runtime hard "
                "contract accepted without the formal cross-check",
                flush=True,
            )
        else:
            runtime_action_ball = _validate_action_ball_runtime_hard_contract(
                runtime_action_ball,
                preflight=preflight,
                racket_cfg=racket,
                motion_cfg=motion_cmd.cfg,
                expected_runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
            )

        admission_fn = getattr(
            motion_cmd, "action_ball_motion_admission_hard_contract", None
        )
        if not callable(admission_fn):
            raise RuntimeError(
                "action-ball requires MotionCommand's code-rooted opaque "
                "action_ball_motion_admission_hard_contract() receipt"
            )
        try:
            motion_admission_receipt = admission_fn()
        except Exception as exc:
            raise RuntimeError(
                "action-ball MotionCommand admission receipt failed closed"
            ) from exc
        if type(motion_admission_receipt) is not dict:
            raise RuntimeError(
                "action-ball MotionCommand admission receipt must be a "
                "non-optional plain mapping"
            )
        _action_ball_assert_json_equal(
            runtime_action_ball.get("motion_admission"),
            motion_admission_receipt,
            name=(
                "action-ball runtime/reopened motion admission receipt"
            ),
        )

        action_ball_ppo_recipe = _validate_action_ball_policy_recipe(
            preflight,
            agent_cfg,
            policy_bootstrap=action_ball_policy_bootstrap,
        )

        diagnostic_action_ball = (
            getattr(
                racket,
                "action_ball_diagnostic_unauthorized",
                False,
            )
            is True
        )
        if diagnostic_action_ball:
            if action_set_identity is not None:
                raise RuntimeError(
                    "diagnostic ActionBall training cannot consume a formal "
                    "action-set launch identity"
                )
        else:
            if action_set_identity is None:
                raise RuntimeError(
                    "formal ActionBall training requires a verified "
                    "launch-claim action-set identity"
                )
            try:
                action_set_identity = (
                    validate_action_ball_action_set_runtime_identity(
                        action_set_identity,
                        actor_obs_contract=getattr(
                            actor_contract, "name", None
                        ),
                        actor_obs_width=getattr(
                            actor_contract, "total_dim", None
                        ),
                        manifest_path=preflight["manifest"]["path"],
                        manifest_sha256=preflight["manifest"][
                            "file_sha256"
                        ],
                        scope=preflight["prototype"]["scope"],
                        mobility_mode=preflight["mobility_mode"],
                        ordered_action_ids=preflight["action_order"],
                        ordered_action_uids=preflight["action_uids"],
                        experiment_name=getattr(
                            agent_cfg, "experiment_name", None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "formal ActionBall action-set launch identity disagrees "
                    "with the instantiated manifest/runtime actor"
                ) from exc

        racket_mode = str(getattr(racket_cmd, "_racket_mode", ""))
        wrist_index = getattr(racket_cmd, "_wrist_body_index", None)
        racket_index = getattr(racket_cmd, "_racket_body_index", None)
        if (
            racket_mode != "wrist_offset"
            or type(wrist_index) is not int
            or wrist_index < 0
            or type(racket_index) is not int
            or racket_index != -1
        ):
            raise RuntimeError(
                "action-ball v1 requires the reviewed wrist_offset physical "
                "paddle-site resolution; the instantiated asset resolved "
                f"mode={racket_mode!r}, wrist_index={wrist_index!r}, "
                f"racket_index={racket_index!r}"
            )
        action_ball_contract = {
            "schema_version": 1,
            "preflight": preflight,
            "runtime": runtime_action_ball,
            "motion_admission": motion_admission_receipt,
            **(
                {}
                if action_ball_policy_bootstrap is None
                else {"policy_bootstrap": action_ball_policy_bootstrap}
            ),
            **(
                {}
                if action_set_identity is None
                else {"action_set_identity": action_set_identity}
            ),
            "authorization": (
                _action_ball_training_authorization_contract(
                    getattr(
                        racket,
                        "action_ball_diagnostic_unauthorized",
                        False,
                    )
                )
            ),
            "resolved_racket_kinematics": {
                "mode": racket_mode,
                "source_body_name": str(racket.wrist_body_name),
                "source_body_index": wrist_index,
                "racket_body_index": racket_index,
            },
            "effective_reward_recipe_sha256": effective_reward_receipt[
                "sha256"
            ],
        }
        _validate_action_ball_training_authorization(action_ball_contract)
    elif action_set_identity is not None:
        raise RuntimeError(
            "an action-set launch identity may only be consumed by "
            "target_mode=action_ball"
        )
    runtime_facts = runtime_execution_facts(env, actor_contract)
    _require_action_ball_finite_qdes_projection_fact(
        runtime_facts,
        action_ball_enabled=action_ball_contract is not None,
    )
    lateral_training = _resolve_lateral_training_runtime(env)
    processed_qdes_slew_contract = _processed_qdes_slew_hinge_reward_contract(
        env_cfg, runtime_facts
    )
    qdes_limit_barrier_contract = _qdes_limit_barrier_reward_contract(
        env_cfg, runtime_facts
    )
    actual_joint_limit_barrier_contract = (
        _actual_joint_limit_barrier_reward_contract(
            env_cfg,
            runtime_facts,
            qdes_contract=qdes_limit_barrier_contract,
        )
    )
    loaded_joint_reference = getattr(getattr(motion_cmd, "motion", None), "joint_pos", None)
    loaded_joint_reference_shape = tuple(getattr(loaded_joint_reference, "shape", ()))
    reference_joint_width = (
        int(loaded_joint_reference_shape[1])
        if len(loaded_joint_reference_shape) == 2
        else -1
    )
    lower_body_pose_contract = _lower_body_pose_imitation_reward_contract(
        env_cfg, runtime_facts, reference_joint_width=reference_joint_width
    )
    lower_body_bundle_contract = _lower_body_stability_bundle_reward_contract(
        env_cfg, runtime_facts
    )
    if (lower_body_pose_contract is None) != (lower_body_bundle_contract is None):
        raise RuntimeError(
            "explicit Wave-B cells require both B1 and B2 contract blocks"
        )
    if (
        lower_body_pose_contract is not None
        and lower_body_bundle_contract is not None
        and lower_body_pose_contract["enabled"]
        and lower_body_bundle_contract["enabled"]
    ):
        raise RuntimeError(
            "Wave-B B1 pose imitation and B2 stability bundle are mutually exclusive"
        )
    post_swing_settle_contract = _post_swing_settle_debt_reward_contract(
        env_cfg, runtime_facts
    )
    push_robot_contract = _push_robot_event_contract(env_cfg)
    force_push_contract = _force_push_event_contract(env_cfg, env)
    ground_plant_contract = _ground_plant_contract(env_cfg)
    if (
        post_swing_settle_contract is not None
        and post_swing_settle_contract["enabled"]
        and (
            (lower_body_pose_contract is not None and lower_body_pose_contract["enabled"])
            or (
                lower_body_bundle_contract is not None
                and lower_body_bundle_contract["enabled"]
            )
        )
    ):
        raise RuntimeError(
            "S1 post_swing_settle_debt and the Wave-B lower-body mechanisms are mutually exclusive"
        )
    motion_files = motion.motion_file
    if not isinstance(motion_files, (list, tuple, ListConfig)):
        motion_files = [motion_files]
    segment_lengths = runtime_facts["motion_segment_lengths"]
    if len(motion_files) != len(segment_lengths):
        raise RuntimeError(
            "loaded motion file count does not match runtime segment count: "
            f"files={len(motion_files)} segments={segment_lengths}"
        )

    def attr(obj, name, default=None):
        return _contract_value(getattr(obj, name, default))

    clips = []
    for index, (path, segment_length) in enumerate(zip(motion_files, segment_lengths)):
        absolute = str(pathlib.Path(path).resolve())
        kinematics = runtime_facts["motion_kinematics_contracts"][index]
        clip_fps = runtime_facts["motion_clip_fps"][index]
        clips.append({
            "index": index,
            "basename": pathlib.Path(path).name,
            "sha256": _sha256_file(absolute),
            "segment_length": int(segment_length),
            "fps": float(clip_fps),
            "kinematics": kinematics,
        })
    question_bank = None
    bank_path_cfg = str(getattr(racket, "question_bank", "") or "").strip()
    if bank_path_cfg:
        import numpy as np

        validated_bank = getattr(racket_cmd, "_question_bank", None)
        bank_path = str(
            pathlib.Path(getattr(validated_bank, "source_path", bank_path_cfg)).resolve()
        )
        if not pathlib.Path(bank_path).is_file():
            raise RuntimeError(f"training question bank does not exist: {bank_path}")
        with np.load(bank_path) as bank_npz:
            meta = None
            if "meta_json" in bank_npz:
                try:
                    meta = json.loads(
                        bytes(np.asarray(bank_npz["meta_json"], dtype=np.uint8)).decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    meta = None
        allow_legacy = bool(getattr(racket, "question_bank_allow_legacy", False))
        if meta is None or int(meta.get("schema_version", 0)) != 3:
            if not allow_legacy:
                raise RuntimeError(
                    "current training configuration requires a schema-v3 question bank"
                )
            question_bank = {
                "sha256": _sha256_file(bank_path),
                "schema_version": "legacy",
                "split": "unknown",
                "source_family_sha256": None,
                "exact": False,
            }
        else:
            if meta.get("split") != "train":
                raise RuntimeError(
                    f"training hard contract requires a train-split bank, got {meta.get('split')!r}"
                )
            family_sha = str(meta.get("source_family_sha256", "")).strip().lower()
            if len(family_sha) != 64 or any(
                ch not in "0123456789abcdef" for ch in family_sha
            ):
                raise RuntimeError("schema-v3 train bank has invalid source-family SHA")
            question_bank = {
                "sha256": _sha256_file(bank_path),
                "schema_version": 3,
                "split": "train",
                "source_family_sha256": family_sha,
                "exact": True,
            }
    # CONTINUOUS producer: a continuous arm has an empty question_bank, so without this block the
    # checkpoint would record NOTHING about how its targets were produced — no cfg hash, no solver
    # version, no physics contract. That regresses reproducibility below the PRE-bank baseline.
    continuous_questions = None
    if hasattr(racket_cmd, "_cq_hard_contract"):
        continuous_questions = racket_cmd._cq_hard_contract()
    if question_bank is not None and continuous_questions is not None:
        raise RuntimeError(
            "a run cannot have both a training question bank and a continuous producer"
        )
    post_swing_replay = motion_cmd.post_swing_replay_hard_contract()
    planner_runtime_contract = motion_cmd.planner_revision_hard_contract()
    planner_training_contract = motion_cmd.planner_revision_training_hard_contract()
    if (planner_runtime_contract is None) != (planner_training_contract is None):
        raise RuntimeError(
            "planner runtime and training hard contracts must be enabled atomically"
        )
    from whole_body_tracking.tasks.tracking.mdp.post_swing_teacher import (
        training_contract_extension,
    )
    return {
        "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "effective_reward_recipe": effective_reward_receipt,
        **runtime_facts,
        "target_mode": attr(racket, "target_mode"),
        "normal_mode": attr(racket, "normal_mode"),
        "racket_pos_range_per_clip": attr(racket, "racket_pos_range_per_clip"),
        "racket_vel_range_per_clip": attr(racket, "racket_vel_range_per_clip"),
        # N-stroke addressing + the per-clip incoming-ball regime. Recorded so a run's per-clip
        # metric bucket names and per-clip ball boxes are reconstructable from the contract alone.
        "clip_names_per_clip": attr(racket, "clip_names_per_clip"),
        "vb_vel_range_per_clip": attr(racket, "vb_vel_range_per_clip"),
        "vb_spin_abs_max_per_clip": attr(racket, "vb_spin_abs_max_per_clip"),
        "base_target_x_range": attr(racket, "base_target_x_range"),
        "base_target_y_range": attr(racket, "base_target_y_range"),
        "mount_normal_axis": attr(racket, "mount_normal_axis"),
        "mount_normal_sign": attr(racket, "mount_normal_sign"),
        "mount_normal_sign_per_clip": attr(racket, "mount_normal_sign_per_clip"),
        "face_command_enabled": bool(getattr(racket, "face_command", False)),
        "face_command_pairing": attr(racket, "face_command_pairing", "shared_plus_y"),
        "racket_control_point": (
            "pingpang_red_Link_origin_v1" if racket is not None else None
        ),
        "racket_control_point_offset_wrist_m": attr(racket, "mount_offset"),
        "strike_phase_per_clip": attr(racket, "strike_phase_per_clip"),
        "racket_strike_phase": attr(racket, "strike_phase"),
        "racket_strike_window_s": attr(racket, "strike_window_s"),
        "racket_strike_window_pos_s": attr(racket, "strike_window_pos_s"),
        "racket_strike_window_wide_s": attr(racket, "strike_window_wide_s"),
        "racket_midswing_resample_prob": attr(racket, "midswing_resample_prob"),
        "racket_midswing_resample_tts_floor": attr(
            racket, "midswing_resample_tts_floor"
        ),
        "racket_target_delay_steps": attr(racket, "target_delay_steps"),
        "racket_target_delay_tts_mode": attr(
            racket, "target_delay_tts_mode", "live"
        ),
        "racket_target_jitter_pos_per_s": attr(racket, "target_jitter_pos_per_s"),
        "racket_target_jitter_vel_per_s": attr(racket, "target_jitter_vel_per_s"),
        "racket_target_noise_white": attr(racket, "target_noise_white"),
        "racket_target_noise_ar1_sigma": attr(racket, "target_noise_ar1_sigma"),
        "racket_target_noise_ar1_rho": attr(racket, "target_noise_ar1_rho"),
        "racket_target_dropout_prob": attr(racket, "target_dropout_prob"),
        "racket_target_post_strike_dropout_s": attr(
            racket, "target_post_strike_dropout_s"
        ),
        "racket_target_bias_per_swing": attr(racket, "target_bias_per_swing"),
        "episode_length_s": float(getattr(env_cfg, "episode_length_s")),
        "motion_wrap_teleport": bool(getattr(motion, "wrap_teleport", False)),
        "motion_hold_steps_range": attr(motion, "hold_steps_range"),
        "motion_hold_reference": "stand",
        "motion_stand_start_prob": attr(motion, "stand_start_prob"),
        "motion_stand_start_min_hold": attr(motion, "stand_start_min_hold"),
        "motion_stand_start_yaw_range": attr(motion, "stand_start_yaw_range"),
        "motion_speed_scale_range": attr(motion, "speed_scale_range"),
        "motion_speed_scale_per_clip": attr(motion, "speed_scale_per_clip"),
        # 每 clip 的 forehand/backhand 家族表(spdmix v2 硬绑定一)。只在显式配置时写进合同——缺席
        # (现役所有在跑臂)不落键,合同字节与历史完全一致,老 checkpoint 的 resume 对账
        # (_contract_diff)不受影响(照 training_contract_extension 的"receipt-free 字节兼容"先例)。
        **(
            {}
            if getattr(motion, "clip_family_per_clip", None) is None
            else {"motion_clip_family_per_clip": attr(motion, "clip_family_per_clip")}
        ),
        "motion_post_swing_start_prob": attr(motion, "post_swing_start_prob"),
        "motion_post_swing_buffer_size": attr(motion, "post_swing_buffer_size"),
        "motion_post_swing_min_fill": attr(motion, "post_swing_min_fill"),
        "motion_post_swing_min_hold": attr(motion, "post_swing_min_hold"),
        # Keep every receipt-free historical/default contract byte-compatible.  The new nested
        # identity exists only when an external teacher artifact can actually affect reset state.
        **training_contract_extension(post_swing_replay),
        "motion_clip_switch_prob": attr(motion, "clip_switch_prob"),
        "motion_rsi_skip_settle_frames": attr(motion, "rsi_skip_settle_frames"),
        "motion_stagger_initial_clock": attr(motion, "stagger_initial_clock"),
        "motion_stagger_hold_max_steps": attr(motion, "stagger_hold_max_steps"),
        "motion_adaptive_kernel_size": attr(motion, "adaptive_kernel_size"),
        "motion_adaptive_lambda": attr(motion, "adaptive_lambda"),
        "motion_adaptive_uniform_ratio": attr(motion, "adaptive_uniform_ratio"),
        "motion_adaptive_alpha": attr(motion, "adaptive_alpha"),
        "motion_event_timing": motion_cmd.event_timing_hard_contract(),
        **(
            {}
            if planner_runtime_contract is None
            else {
                "planner_task_revision": planner_runtime_contract,
                "planner_task_revision_training": {
                    "initial_tts_sampling_semantics": (
                        "explicit_weighted_mixture_over_initial_tts_range_s"
                    ),
                    "initial_tts_mixture": planner_training_contract[
                        "initial_tts_mixture"
                    ],
                    "initial_feasibility_gate": (
                        "normalized_phase_rate_and_acceleration_envelope_only"
                    ),
                    "dynamics_certified_action_tau_min_bound": False,
                    "timing_exam_semantics": {
                        "0.5_s": "required_baseline_gate",
                        "below_0.5_s": "stress_diagnostic_not_support_floor",
                    },
                    "position_std_m": attr(
                        racket, "planner_revision_position_std_m"
                    ),
                    "velocity_std_mps": attr(
                        racket, "planner_revision_velocity_std_mps"
                    ),
                    "normal_std_rad": attr(
                        racket, "planner_revision_normal_std_rad"
                    ),
                    "tts_std_s": attr(racket, "planner_revision_tts_std_s"),
                    "truth_fields_immutable": [
                        "question_bank_row",
                        "physical_ball",
                        "reward_target",
                        "critic_target",
                    ],
                    "actor_revision_fields": [
                        "target_position",
                        "target_velocity",
                        "signed_target_normal",
                        "time_to_strike",
                    ],
                },
            }
        ),
        "motion_allow_legacy_link_origin_velocity": bool(
            getattr(motion, "allow_legacy_link_origin_velocity", False)
        ),
        "motion_rsi_hold_root_stand_z": bool(getattr(motion, "rsi_hold_root_stand_z", False)),
        # Unlike weights/stds, a body mask changes which robot subsystem the teacher constrains.
        # Bind the exact post-override lists so A0/A1 checkpoints remain distinguishable even if
        # copied away from their outer launch directories.
        "motion_imitation_body_names": _motion_imitation_body_names_contract(env_cfg),
        "racket_guidance_reward": _racket_guidance_reward_contract(
            env_cfg, racket_task=racket_cmd is not None
        ),
        "joint_velocity_limit_hinge_reward": (
            _joint_velocity_limit_hinge_reward_contract(env_cfg, runtime_facts)
        ),
        **(
            {}
            if processed_qdes_slew_contract is None
            else {
                "processed_qdes_slew_hinge_reward": processed_qdes_slew_contract
            }
        ),
        **(
            {}
            if qdes_limit_barrier_contract is None
            else {"qdes_limit_barrier_reward": qdes_limit_barrier_contract}
        ),
        **(
            {}
            if actual_joint_limit_barrier_contract is None
            else {
                "actual_joint_limit_barrier_reward": (
                    actual_joint_limit_barrier_contract
                )
            }
        ),
        **(
            {}
            if lower_body_pose_contract is None
            else {"lower_body_pose_imitation_reward": lower_body_pose_contract}
        ),
        **(
            {}
            if lower_body_bundle_contract is None
            else {"lower_body_stability_bundle_reward": lower_body_bundle_contract}
        ),
        **(
            {}
            if post_swing_settle_contract is None
            else {"post_swing_settle_debt_reward": post_swing_settle_contract}
        ),
        **(
            {}
            if push_robot_contract is None
            else {"push_robot_event": push_robot_contract}
        ),
        **(
            {}
            if force_push_contract is None
            else {"force_push_event": force_push_contract}
        ),
        **(
            {}
            if lateral_training is None
            else {"lateral_perturbation": lateral_training[1]}
        ),
        # 地面/地形 plant 指纹(2026-07-22):默认平地配方 = 不落键(历史 checkpoint 逐字节
        # 兼容);任何摩擦/地形改动 = 落键,resume 对账把它当另一套 plant 拒绝静默续训。
        **(
            {}
            if ground_plant_contract is None
            else {"ground_plant": ground_plant_contract}
        ),
        "motion_clips": clips,
        "question_bank": question_bank,
        # Absent (not None) for every historical bank/no-bank run, so old checkpoints stay
        # byte-identical under the resume contract diff.
        **({} if continuous_questions is None
           else {"continuous_questions": continuous_questions}),
        **(
            {}
            if task_first_contract is None
            else {
                "task_first_training": task_first_contract,
                "task_first_ppo_runner_recipe": _task_first_agent_recipe(
                    agent_cfg
                ),
            }
        ),
        **(
            {}
            if action_ball_contract is None
            else {
                "action_ball_training": action_ball_contract,
                "action_ball_ppo_runner_recipe": action_ball_ppo_recipe,
            }
        ),
    }


def _contract_diff(expected, actual, prefix="") -> list[str]:
    """Human-readable recursive diff for fail-loud resume errors."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                lines.append(f"{child}: checkpoint=<missing> current={actual[key]!r}")
            elif key not in actual:
                lines.append(f"{child}: checkpoint={expected[key]!r} current=<missing>")
            else:
                lines.extend(_contract_diff(expected[key], actual[key], child))
        return lines
    if expected != actual:
        return [f"{prefix}: checkpoint={expected!r} current={actual!r}"]
    return []


def _normalize_registry_name(name) -> str:
    reg = str(name)
    if ":" not in reg:
        reg += ":latest"
    return reg


def _motion_clip_name_from_path(value) -> str | None:
    items = _configured_items(value)
    if not items:
        return None
    parts = [p for p in str(items[0]).replace("\\", "/").split("/") if p]
    if not parts:
        return None
    if parts[-1] == "motion.npz" and len(parts) >= 2:
        return parts[-2].split(":")[0] or None
    return pathlib.PurePath(parts[-1]).stem or None


def _resolve_local_motion_files(primary, secondary=None, cwd: pathlib.Path | None = None) -> list[str]:
    files = []
    base = pathlib.Path.cwd() if cwd is None else pathlib.Path(cwd)
    for value in _configured_items(primary, secondary):
        path = pathlib.Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"[train.py] motion_file not found: {path}")
        files.append(str(path))
    return files


def _download_registry_motion_files(primary, secondary=None) -> tuple[list[str], list[str]]:
    registries = [_normalize_registry_name(value) for value in _configured_items(primary, secondary)]
    if not registries:
        raise RuntimeError(
            "[train.py] No reference motion configured. Pass motion_file=/path/to/motion.npz "
            "(and optional motion_file_2=/path/to/backhand.npz) for the local path, or pass "
            "registry_name=<org>/wandb-registry-motions/<name>."
        )
    # Import lazily and AFTER the guard: the no-WandB local path must never require wandb, and a
    # missing-motion misconfiguration should raise the guidance error above, not ModuleNotFoundError.
    import wandb

    api = wandb.Api()
    motion_files = []
    for reg in registries:
        art = api.artifact(reg)
        # Provenance: record exactly which artifact version/digest the run trains on (the registry
        # alias is mutable, e.g. ':latest' can move between runs).
        print(f"[train.py] motion clip: {reg} -> {art.source_qualified_name} (digest {art.digest[:12]})", flush=True)
        motion_files.append(str(pathlib.Path(art.download()) / "motion.npz"))
    return motion_files, registries


def resolve_motion_sources(cfg, *, cwd: pathlib.Path | None = None) -> tuple[list[str], list[str]]:
    """Resolve local or registry motion sources for train/play.

    Local files are intentionally all-or-nothing: once ``motion_file`` is set, training does not touch the
    registry. Use ``motion_file_2`` (or a list-valued ``motion_file``) for the unified forehand/backhand
    local workflow.
    """
    local_files = _resolve_local_motion_files(
        _get(cfg, "motion_file"),
        _get(cfg, "motion_file_2"),
        cwd=cwd,
    )
    if local_files:
        return local_files, []

    task = _get(cfg, "task")
    registry_name = _get(cfg, "registry_name") if _get(cfg, "registry_name") is not None else _get(task, "registry_name")
    reg2 = _get(cfg, "registry_name_2") if _get(cfg, "registry_name_2") is not None else _get(task, "registry_name_2")
    return _download_registry_motion_files(registry_name, reg2)


class _OverrideError(AttributeError):
    """Raised when the task YAML asks to override an attribute the composed env cfg does not have."""


def _require(cond, target):
    # The YAML explicitly set a value, but the target attribute is missing on the composed env cfg.
    # That is NEVER a benign no-op: either a STALE/shadowed whole_body_tracking was imported (so the
    # cfg classes differ from the working tree) or the Hydra base groups failed to compose. Fail loud
    # instead of silently dropping the override (the old behaviour that hid the std/curriculum edits).
    if not cond:
        raise _OverrideError(
            f"[train.py] task YAML overrides '{target}' but the composed env cfg has no such attribute. "
            f"Check the '[train.py] env cfg source:' line above — if it points into site-packages rather "
            f"than your working tree, a stale install is shadowing the source (fix PYTHONPATH ordering / "
            f"reinstall editable). Otherwise the Hydra base-group composition for this task failed."
        )


def _set_attr(obj, attr, val, cast, applied, where):
    if val is None:
        return  # key absent from YAML -> keep the code default (documented contract)
    _require(hasattr(obj, attr), f"{where}.{attr}")
    setattr(obj, attr, cast(val))
    applied.append(f"{where}.{attr}={cast(val)!r}")


def _set_range(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    rng = (float(val[0]), float(val[1]))
    setattr(obj, attr, rng)
    applied.append(f"{where}.{attr}={rng}")


def _set_vec3(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    vec = (float(val[0]), float(val[1]), float(val[2]))
    setattr(obj, attr, vec)
    applied.append(f"{where}.{attr}={vec}")


def _set_reward(rewards, name, weight, std, applied):
    if weight is None and std is None:
        return  # this reward term is not overridden by the YAML -> keep code defaults
    _require(hasattr(rewards, name), f"rewards.{name}")
    term = getattr(rewards, name)
    if weight is not None:
        term.weight = float(weight)
        applied.append(f"rewards.{name}.weight={float(weight)}")
    if std is not None:
        _require("std" in term.params, f"rewards.{name}.params['std']")
        std_value = float(std)
        if not math.isfinite(std_value) or std_value <= 0.0:
            raise _OverrideError(f"rewards.{name}.std must be finite and > 0, got {std!r}")
        term.params["std"] = std_value
        applied.append(f"rewards.{name}.params.std={std_value}")


def _check_unknown_keys(node, known, where):
    # _require guards one direction (YAML sets a key, env cfg lacks the attribute); this guards the
    # other: a key present under the node that no _set_attr/_set_range call below ever reads would be
    # a SILENT no-op (this is exactly how r3_P2_product's task.racket.target_noise_white=0.0019 /
    # target_noise_ar1_sigma=0.0052 / vb_spin_mode=minimize CLI overrides got dropped on 2026-07-03).
    if node is None:
        return
    try:
        present = list(node.keys())
    except Exception:
        return
    unknown = sorted(str(k) for k in present if str(k) not in known)
    if unknown:
        raise _OverrideError(
            f"[train.py] {where} sets key(s) {unknown} that the override translation layer does not "
            f"consume — they would be silently ignored. Add each to the whitelist AND a "
            f"_set_attr/_set_range call in _apply_task_overrides, or remove it from the YAML/CLI."
        )


_PLANNER_REVISION_KEYS = (
    "enabled",
    "profile",
    "initial_tts_range_s",
    "initial_tts_mixture",
    "position_std_m",
    "velocity_std_mps",
    "normal_std_rad",
    "tts_std_s",
)


# YAML keys under `racket:` that target the RacketTargetCommandCfg (used to decide whether the task
# actually requested racket overrides before requiring the command to exist).
_RACKET_KEYS = (
    "strike_phase", "strike_phase_by_motion", "strike_window_s", "strike_success_pos_thresh",
    # 1c split strike windows (position tight / normal+velocity wide; defaults None = single window)
    "strike_window_pos_s", "strike_window_wide_s",
    "pos_x_range", "pos_y_range", "pos_z_range", "racket_pos_y_abs_range", "pos_range_per_clip",
    "vel_x_range", "vel_y_range", "vel_z_range", "vel_range_per_clip",
    # N-stroke addressing: the ORDERED per-clip key list every per-clip block above is keyed by
    # (absent = the legacy two family names). Lifts the two-stroke ceiling.
    "clip_names",
    # per-clip INCOMING-ball regime (a block gets fast balls, a loop slow ones, same run)
    "vb_vel_range_per_clip", "vb_spin_abs_max_per_clip",
    # Escape hatch for the "commanded return velocity must point at the opponent" construction gate.
    "allow_non_forward_target_velocity",
    "base_target_x_range", "base_target_y_range",
    "normal_mode", "forehand_on_negative_y", "mount_normal_axis", "mount_normal_sign",
    # 每 clip 击球面符号(正反手各用拍子固定的一面;空/缺省=标量 mount_normal_sign,现役行为不变)
    "mount_normal_sign_per_clip",
    "target_mode", "ref_perturb_pos", "ref_perturb_vel", "ref_perturb_normal",
    # Task-first: exact manifest bytes + same-attempt base success gate.
    "task_first_manifest_path", "task_first_manifest_sha256",
    "task_first_base_success_thresh_m",
    # Action-ball: exact action/ball manifest, immutable policy recipe, and
    # deterministic per-action sampler/pool identity.
    "action_ball_manifest_path", "action_ball_manifest_sha256",
    "action_ball_policy_contract_sha256", "action_ball_seed",
    "action_ball_pool_refill_rows", "action_ball_fixed_direction",
    "action_ball_evaluator_launch_receipt_path",
    "action_ball_evaluator_launch_receipt_file_sha256",
    "action_ball_sidecar_launch_receipt_path",
    "action_ball_sidecar_launch_receipt_file_sha256",
    "action_ball_drain_reset_launch_receipt_path",
    "action_ball_drain_reset_launch_receipt_file_sha256",
    "action_ball_evaluation_inbox_root",
    "action_ball_evaluation_owner_id",
    "action_ball_evaluation_run_id",
    "action_ball_frozen_eval_interval_updates",
    "action_ball_diagnostic_unauthorized",
    "reference_guard_mode",
    "virtual_ball",
    "ref_perturb_curriculum_steps", "ref_perturb_curriculum_start", "ref_perturb_success_gated",
    "ref_perturb_advance_threshold", "ref_perturb_advance_rate", "ref_vel_scale", "ref_vel_scale_by_motion",
    "debug_reward_logging",
    "clean_reference_strike_velocity", "clean_strike_vel_window",
    "adaptive_sigma", "sigma_update_every", "sigma_ema_scale",
    "sigma_pos_min", "sigma_pos_max", "sigma_vel_min", "sigma_vel_max",
    # 拍面 sigma 第三通道(A1 2026-07-25):racket_normal 的核宽也跟着 exact-strike 面角
    # 误差自适应收紧(必须搭在 adaptive_sigma 上,单开在 hope_commands 构造期 fail-loud)。
    "adaptive_sigma_normal",
    # HER-style achieved-target replay (mixture sampling from previously-achieved strike states).
    "achieved_target_mix_prob", "achieved_buffer_size", "achieved_min_fill",
    "achieved_jitter_pos", "achieved_jitter_vel", "achieved_clamp_inflate",
    # A1 target latency & time-variance (actor-visible delay, SMASH tts-decaying jitter,
    # mid-swing target refinement). Defaults OFF; byte-identical baseline.
    "target_delay_steps", "target_delay_tts_mode",
    "target_jitter_pos_per_s", "target_jitter_vel_per_s",
    "midswing_resample_prob", "midswing_resample_tts_floor",
    # A1v2 calibrated mocap-degradation channels (white/AR1 noise, frame dropout, per-swing bias).
    "target_noise_white", "target_noise_ar1_sigma", "target_noise_ar1_rho",
    "target_dropout_prob", "target_post_strike_dropout_s", "target_bias_per_swing",
    # Tier-1 virtual ball: incoming-ball sampling boxes + outgoing-spin objective.
    "vb_spin_mode", "vb_spin_min_sigma", "vb_spin_abs_max",
    "vb_vel_x_range", "vb_vel_y_range", "vb_vel_z_range",
    # metrics-only virtual ball (in-training 上台率/击球率 curves without vb rewards)
    "vb_metrics_only",
    # metric-sync fix (2026-07-09): keep the OLD mixed-ledger rally curves alive as *_legacy for
    # one new/old-comparison transition period. 人话:旧算法上台率对照曲线还要不要发。
    "rally_legacy_metrics",
    # translated below but previously missing from this whitelist
    "strike_phase_per_clip", "base_couple_blend", "base_couple_max_offset",
    # HITTER separate-commands base/racket coupling ("blend" | "reference_reach"), 2026-07-05
    "base_couple_mode",

    # Stage-1 question bank (fixed contact point, inverse-solved face+velocity targets) + the
    # face-command reward re-anchor / +4 actor obs channel (normal + rho placeholder, 175->179).
    "question_bank", "question_bank_allow_legacy", "face_command", "face_command_pairing",
    "face_command_obs",
    # CONTINUOUS question production (owner: 离散题库是考试用的,训练必须连续采样). ONE switch,
    # racket.target_mode: solved — every cq_* below is either derived or fail-closed, so there is
    # no second thing to remember. cq_anchor_bank is the validated CONTRACT ANCHOR (never trained
    # on); exam_bank is what judge.sh reads so no operator has to pass --exam-bank.
    "cq_vel_range_per_clip", "cq_aim_xy", "cq_spin_abs_max", "cq_buffer_rows", "cq_overdraw",
    "cq_n_iters", "cq_tol_m", "cq_speed_budget", "cq_max_redraw_rounds", "cq_max_face_deg",
    "cq_exam_holdout", "cq_seed", "cq_accept_buckets", "cq_max_exhausted_frac",
    "cq_abort_exhausted_frac", "cq_min_accept_rate", "cq_closed_loop_rows",
    "cq_closed_loop_max_err_m", "cq_anchor_bank", "exam_bank",
    # R10c 站位锚观测(+2 actor 尾维,179->181;franco 2026-07-09"planner 的 p_base 应该加进去")
    "station_obs", "station_anchor_offset_xy",
    # SHADOW physical ball + table (flag-gated, METRICS-ONLY engine-vs-analytic landing
    # cross-check; requires the virtual-ball task variant). shadow_ball.py.
    "shadow_ball", "shadow_table",
)

# YAML keys under `motion:` that target the MotionCommandCfg swing-entry structure
# (Phase-A multi-swing machinery: no-teleport wrap, stand-entry resets, pre-swing hold,
# A8 post-swing initial-state buffer).
_MOTION_KEYS = (
    "wrap_teleport", "stand_start_prob", "hold_steps_range", "stand_start_min_hold",
    "stand_start_yaw_range",
    "post_swing_start_prob", "post_swing_buffer_size", "post_swing_min_fill", "post_swing_min_hold",
    "post_swing_teacher_receipt", "post_swing_teacher_receipt_sha256",
    "post_swing_teacher_retry_authorization",
    "post_swing_teacher_retry_authorization_sha256",
    "post_swing_teacher_root_linear_velocity_limit_mps",
    "post_swing_teacher_root_angular_velocity_limit_radps",
    "post_swing_require_ready_at_init",
    "post_swing_fail_fast_first_reset",
    "post_swing_first_reset_min_adopted_count",
    "post_swing_first_reset_min_adopted_fraction",
    "post_swing_first_reset_selection_tolerance",
    "post_swing_first_reset_require_readback",
    "post_swing_capture_output_dir", "post_swing_capture_target_count",
    # deploy-parity mid-swing clip switch (018467a added the yaml key + MotionCommandCfg field but not
    # this whitelist/translation, so every run of the task yaml raised in _check_unknown_keys).
    "clip_switch_prob",
    # T1 immutable post-strike event timing (blocked preregistration; code path is fail-closed and
    # remains disabled unless a materialized schedule and its exact byte SHA are both supplied).
    "event_timing_mode", "event_timing_schedule", "event_timing_schedule_sha256",
    "event_timing_repeat",
    # P2.4/R14 per-swing reference playback speed range (retiming).
    "speed_scale_range",
    # 2026-07-08 backhand-fix ablation: fixed per-clip reference playback speed (e.g. [1.0, 0.8]).
    "speed_scale_per_clip",
    # spdmix v2 硬绑定一 (2026-07-22): per-clip forehand/backhand family labels so a 6-clip
    # speed-variant list stops misreading forehand variants as backhands. Absent = legacy 2-clip
    # derivation, byte-identical.
    "clip_family_per_clip",
    # R-c RSI birth fixes (reward_staged_design 2026-07-08 §⑥): (i) skip the clip's first N
    # IK-cold-start frames at every swing entry; (ii) held-RSI births get the default-STAND root
    # height (the stand joints were already used; the crouch root z left the feet 0.29 m under
    # the floor -> PhysX depenetration kick).
    "rsi_skip_settle_frames", "rsi_hold_root_stand_z",
    # 防同步 stagger (metric-sync fix 2026-07-09, default OFF): one-shot random offsets on the
    # first hold + episode clock so a same-instant 4096-env cohort stops timing out / swinging in
    # one synchronized wave (the EMA-metric oscillation disease). 人话:把所有 env 的节拍随机错开。
    "stagger_initial_clock", "stagger_hold_max_steps",
    # Diagnostic-only opt-in for old motion npz files whose body velocity lives at link origin.
    # Absent/false keeps the exact schema-v2 motion contract path unchanged.
    "allow_legacy_link_origin_velocity",
    # Arbitrary-N task-first allocation.  The sampler state is checkpointed by
    # MotionCommand's exact-resume hook; absent keeps historical RNG untouched.
    "balanced_clip_sampling", "balanced_clip_sampling_seed",
    # Formal action-ball entry: every true reset starts from the code-admitted
    # canonical ready state, with no random pose/velocity/joint perturbation.
    "canonical_ready_mode",
    "canonical_registry_path", "canonical_registry_repo_root",
    "canonical_registry_sha256", "canonical_registry_alignment_sha256",
    "canonical_ready_sha256", "canonical_ready_fk_sha256",
    "canonical_promotion_certificate_path",
    "joint_position_range", "pose_range", "velocity_range",
)

# YAML keys under `rewards:` consumed by the rewards block of _apply_task_overrides below.
# Same fail-loud contract as _RACKET_KEYS/_MOTION_KEYS (2026-07-09): before this whitelist an
# unknown/misspelled task.rewards key (e.g. a CLI override typo) was SILENTLY ignored — the run
# started and trained on the wrong reward config. Add each new key here AND a translation below.
_REWARD_KEYS = (
    "racket_position_weight", "racket_position_std", "racket_position_static",
    "racket_velocity_weight", "racket_velocity_std",
    "racket_normal_weight", "racket_normal_std",
    "base_position_weight", "base_position_std",
    "hold_ready_weight", "hold_ready_std", "hold_ready_reach", "hold_ready_reach_mode",
    "post_strike_brake_weight", "post_strike_brake_std",
    "hold_heading_weight", "hold_heading_std", "foot_orientation_hold_gate",
    "base_decel_weight", "base_decel_std", "base_decel_v_gain", "base_decel_v_max",
    "joint_velocity_limit_hinge_weight", "joint_velocity_limit_hinge_margin",
    "processed_qdes_slew_hinge_weight", "processed_qdes_slew_hinge_margin",
    "processed_qdes_slew_hinge_recovery_start_s",
    "processed_qdes_slew_hinge_recovery_end_s",
    # Wave-Q qbar all-joint q_des position-limit barrier (Jiayi V14 idea, top-k removed).
    # Any explicit key requires the weight; an explicit zero weight is a measured control.
    "qdes_limit_barrier_weight", "qdes_limit_barrier_margin_frac",
    # mjlab-ported foot-contact shaping(默认全关):落地冲击(first-contact 法向峰值力超阈
    # 有界惩罚)+ 摆动相抬脚高度(|脚高-目标高| x 水平速度)。参数键不带 weight 一律拒收,
    # 显式 weight=0 是对照。人话:一个罚"落地砸太重",一个罚"腾空脚贴地扫"。
    "foot_soft_landing_weight", "foot_soft_landing_force_threshold_n",
    "foot_clearance_weight", "foot_clearance_target_m",
    # 触地脚水平蹭滑(foot_slip_sq,默认 -1.0)与拖脚(foot_drag,默认 -0.5)的剂量键
    # (2026-07-22 penlight 减负臂需要;此前这两项是源码常开、CLI 够不着的软惩罚)。
    "foot_slip_sq_weight", "foot_drag_weight",
    # Wave-B mutually-exclusive lower-body diagnostics. Explicit zero-valued controls still
    # activate their measurement probes and hard-contract identity.
    "lower_body_pose_imitation_weight", "lower_body_pose_imitation_std",
    "lower_body_pose_imitation_support_pre_s",
    "lower_body_pose_imitation_support_post_s",
    # 击球窗内下肢模仿衰减系数(默认 1.0 = 不衰减;上半身 motion_scale_in_window 的下肢版,
    # 同一个 WIDE strike window)。人话:触球那一瞬让下肢模仿小声点,别把击球奖励淹了。
    # 走 Wave-B 信封:配了它就必须显式给出 B1/B2 两个 weight(param-without-weight 拒收)。
    "lower_body_imitation_scale_in_window",
    "lower_body_stability_bundle_weight",
    "lower_body_stability_min_stance_width_m",
    "lower_body_stability_stance_scale_m",
    "lower_body_stability_leg_velocity_margin_radps",
    "lower_body_stability_leg_velocity_scale_radps",
    "lower_body_stability_support_pre_s",
    "lower_body_stability_support_post_s",
    # S1 post-swing settle-debt bundle (Jiayi V13 post-swing debts idea, clean main-side redo).
    # Any explicit S1 key raises the measurement probe; a parameter key without the weight is
    # refused so every S1 cell states its weight explicitly.
    "post_swing_settle_debt_weight",
    "post_swing_settle_base_lin_margin_mps",
    "post_swing_settle_base_lin_scale_mps",
    "post_swing_settle_base_ang_margin_radps",
    "post_swing_settle_base_ang_scale_radps",
    "post_swing_settle_tilt_margin_rad",
    "post_swing_settle_tilt_scale_rad",
    "post_swing_settle_nominal_root_z_m",
    "post_swing_settle_root_height_deadband_m",
    "post_swing_settle_root_height_scale_m",
    "post_swing_settle_foot_slip_margin_mps",
    "post_swing_settle_foot_slip_scale_mps",
    "post_swing_settle_recovery_start_s",
    "post_swing_settle_recovery_end_s",
    # R16 / V1 wrist-mimic surgery (orientation 2026-07-04; linear velocity 2026-07-08 §③).
    "free_wrist_ori_mimic", "free_wrist_vel_mimic",
    # A0/A1 non-striking-arm imitation ablation (2026-07-14).  This deliberately has one
    # narrow meaning: remove the three LEFT-arm links from all four body-imitation terms while
    # retaining torso + the complete right (racket) arm.  It does not touch any safety term.
    "free_non_striking_arm_mimic",
    # 全身模仿开关(Franco 2026-07-25 裁定:下半身也应全局模仿,可开关)。true = 把
    # pelvis + 六个腿部 link 加回四个 motion_body_* 名单(恢复 BeyondMimic 原始 13-body
    # 集),与 free_wrist_*/free_non_striking 的摘除可叠加(腿加回,被摘的照旧不学)。
    "full_body_mimic",
    "joint_torques_weight",
    # per-term overrides of the six imitation terms + the global/in-window scales
    "motion_global_anchor_pos_weight", "motion_global_anchor_pos_std",
    "motion_global_anchor_ori_weight", "motion_global_anchor_ori_std",
    "motion_body_pos_weight", "motion_body_pos_std",
    "motion_body_ori_weight", "motion_body_ori_std",
    "motion_body_lin_vel_weight", "motion_body_lin_vel_std",
    "motion_body_ang_vel_weight", "motion_body_ang_vel_std",
    "motion_scale", "motion_scale_in_window",
    # penalties / regularization。action_acc = mjlab 档①动作二阶平滑(action_rate 罚"步子
    # 迈多大",它罚"方向掉头多猛";weight-only 键,finite 且 <= 0,显式 0 = 对照;剂量别抄
    # 一阶惯用值,见 hope_env_cfg 注释)。
    "action_rate_weight", "action_acc_weight", "joint_limit_weight", "undesired_contacts_weight",
    "pre_strike_foot_slip_weight", "prestrike_waist_twist_weight",
    "arm_torque_saturation_weight", "prestrike_upright_weight", "foot_orientation_weight",
    # proximity power-gate for the face/velocity channels (reward_staged_design §② C2a)
    "face_gate_by_pos", "face_gate_radius",
    # constant guidance penalty toward the racket target (reward_staged_design §② B2)
    "racket_guidance_weight", "racket_face_guidance_weight", "racket_face_guidance_theta_max",
    "racket_face_conditional_guidance_weight",
    # v2 奖励包一键选择器(reward_redesign_20260725 §3;缺席 = v1 现状,逐字节不变;唯一合法值
    # "v2")。人话:一个键把 v2 蓝图整套换装展开;包先展开、显式同名键后写后赢,见
    # _expand_reward_pack。
    "reward_pack",
    # 冻结数保护开关(07-27;默认 false = 只响亮记账+WARNING,行为逐字节不变)。true =
    # 任何显式键压过 _REWARD_PACK_V2_CALIBRATED 里的标定冻结数一律 fail-loud。人话:
    # prereg 冻结臂打开它,"这条臂跑的就是冻结表"这句话才不可能再变成假话。
    "reward_pack_strict",
    # landing 延付消融 flag(07-26 Franco:默认关;>0 = 大奖延付该秒数、同 attempt 存活才发)。
    "virtual_landing_settle_delay_s",
    # scale 消融键(07-26 pod1 队列):臂级覆写上台大奖权重与底薪比例(显式键压过包值)。
    "virtual_landing_weight", "virtual_landing_base_frac",
    # 摔死罚消融键(07-26 death09 臂;配方审计发现包 direct 写死 -1800 无 CLI 面)。
    "death_penalty_weight",
    # 撞桌罚消融键(07-27 上桌障碍物;与 death_penalty_weight 同形,只认 robot_hit_table)。
    "table_hit_penalty_weight",
)

# jiayi 的 YAML-null 删参修复(8ee2e82a,搬进 main 血统)。人话:task YAML 层层继承时,子任务
# 想把父层 EnvCfg __post_init__ 传下来的某个 reward 参数"删掉"(典型:后继任务复用同名 reward
# 项但换了函数/签名——RallyV13 的全关节 barrier 就是这么撞上的——旧参数塞给新函数直接炸),
# 唯一的 YAML 写法是 `some_key: null`;可 _apply_task_overrides 的覆盖层对 None 一律当"没写"
# 跳过,null 就永远删不掉。这张表列出每个"一键对一 params 项"的映射(YAML 键 -> (reward 项名,
# params 键));进 rewards 块时先按表把显式 null 的参数 pop 掉并记进 applied,再走正常覆盖。
# 不在表里的键 null 仍等价于"没写":weight/机制开关不是 params 项,face_gate_radius /
# motion_scale_in_window 这类一键写多个 term 的参数只由覆盖层自己写入,EnvCfg 血统里根本不带,
# 不存在"继承了删不掉"的问题。
_REWARD_NULL_REMOVABLE_PARAMS = {
    # _set_reward 家族:<term>_std 都写进 term.params["std"]
    "racket_position_std": ("racket_position", "std"),
    "racket_velocity_std": ("racket_velocity", "std"),
    "racket_normal_std": ("racket_normal", "std"),
    "base_position_std": ("base_position", "std"),
    "hold_ready_std": ("hold_ready", "std"),
    "post_strike_brake_std": ("post_strike_brake", "std"),
    "hold_heading_std": ("hold_heading", "std"),
    "base_decel_std": ("base_decel", "std"),
    "motion_global_anchor_pos_std": ("motion_global_anchor_pos", "std"),
    "motion_global_anchor_ori_std": ("motion_global_anchor_ori", "std"),
    "motion_body_pos_std": ("motion_body_pos", "std"),
    "motion_body_ori_std": ("motion_body_ori", "std"),
    "motion_body_lin_vel_std": ("motion_body_lin_vel", "std"),
    "motion_body_ang_vel_std": ("motion_body_ang_vel", "std"),
    # 逐项 params 键(与下方各 ad-hoc setter 一一对应)
    "hold_ready_reach": ("hold_ready", "reach"),
    "hold_ready_reach_mode": ("hold_ready", "reach_mode"),
    "foot_orientation_hold_gate": ("foot_orientation", "hold_gate"),
    "base_decel_v_gain": ("base_decel", "v_gain"),
    "base_decel_v_max": ("base_decel", "v_max"),
    "joint_velocity_limit_hinge_margin": ("joint_velocity_limit_hinge", "margin"),
    "processed_qdes_slew_hinge_margin": ("processed_qdes_slew_hinge", "margin"),
    "processed_qdes_slew_hinge_recovery_start_s": (
        "processed_qdes_slew_hinge", "recovery_start_s"),
    "processed_qdes_slew_hinge_recovery_end_s": (
        "processed_qdes_slew_hinge", "recovery_end_s"),
    "qdes_limit_barrier_margin_frac": ("qdes_limit_barrier", "margin_frac"),
    "foot_soft_landing_force_threshold_n": ("foot_soft_landing", "force_threshold_n"),
    "foot_clearance_target_m": ("foot_clearance", "target_m"),
    # lower_body_imitation_scale_in_window 不进此表:它和 motion_scale_in_window 一样只由
    # 覆盖层写进 params,EnvCfg 血统里根本不带,不存在"继承了删不掉"的问题(见上方注释)。
    "lower_body_pose_imitation_std": ("lower_body_pose_imitation", "std"),
    "lower_body_pose_imitation_support_pre_s": (
        "lower_body_pose_imitation", "support_pre_s"),
    "lower_body_pose_imitation_support_post_s": (
        "lower_body_pose_imitation", "support_post_s"),
    "lower_body_stability_min_stance_width_m": (
        "lower_body_stability_bundle", "min_stance_width_m"),
    "lower_body_stability_stance_scale_m": (
        "lower_body_stability_bundle", "stance_scale_m"),
    "lower_body_stability_leg_velocity_margin_radps": (
        "lower_body_stability_bundle", "leg_velocity_margin_radps"),
    "lower_body_stability_leg_velocity_scale_radps": (
        "lower_body_stability_bundle", "leg_velocity_scale_radps"),
    "lower_body_stability_support_pre_s": (
        "lower_body_stability_bundle", "support_pre_s"),
    "lower_body_stability_support_post_s": (
        "lower_body_stability_bundle", "support_post_s"),
    "racket_face_guidance_theta_max": ("racket_face_guidance", "theta_max"),
}
# S1 settle-debt 的 13 个数值参数同样一键对一 params 项,直接从规格表生成,防手抄漂移。
_REWARD_NULL_REMOVABLE_PARAMS.update({
    f"post_swing_settle_{_name}": ("post_swing_settle_debt", _name)
    for _name, _positive, _nonnegative in _POST_SWING_SETTLE_NUMERIC_SPECS
})
# 建表自检(import 时就炸):null-删参表里的每个键必须同时在 _REWARD_KEYS 白名单里,否则
# _check_unknown_keys 会先把它拒掉,表项就成了永远走不到的死代码。
_NULL_TABLE_STRAYS = sorted(set(_REWARD_NULL_REMOVABLE_PARAMS) - set(_REWARD_KEYS))
if _NULL_TABLE_STRAYS:
    raise RuntimeError(
        "[train.py] _REWARD_NULL_REMOVABLE_PARAMS keys missing from _REWARD_KEYS: "
        f"{_NULL_TABLE_STRAYS}"
    )
del _NULL_TABLE_STRAYS


def _apply_reward_param_null_removals(rewards, rw, applied):
    """task.rewards 里显式写 null 的参数键 -> 从对应 term.params 里删掉(记进 applied)。

    人话:null 的意思是"确保这个继承来的参数不存在",所以是幂等的——term 在这个 cfg 血统里
    本来就没有 / params 里本来就没这个键,都算已达成,静默跳过(与 jiayi 8ee2e82a 语义一致)。
    probe 项不用单独删:每个激活块都会 probe.params.clear() + update(term.params) 整体跟随。
    """
    for yaml_name, (term_name, param_name) in _REWARD_NULL_REMOVABLE_PARAMS.items():
        try:
            explicitly_null = yaml_name in rw and _get(rw, yaml_name) is None
        except (TypeError, AttributeError):
            # rw 不是 mapping(异常形态)——这里不裁决,交给覆盖层其余检查 fail-loud。
            explicitly_null = False
        if not explicitly_null:
            continue
        term = getattr(rewards, term_name, None)
        params = getattr(term, "params", None)
        if params is None or param_name not in params:
            continue  # 删除是幂等的:本来就不存在 = 已达成
        params.pop(param_name)
        applied.append(
            f"rewards.{term_name}.params.{param_name}=<removed by YAML null>"
        )


# --------------------------------------------------------------------------------------------- #
# v2 奖励包(docs/research/reward_redesign_20260725.md §3 蓝图)。task.rewards.reward_pack 是
# "一键成套"选择器。2026-07-25 Franco 裁定【默认翻转】:缺席 = 按 "v2" 展开(applied 记
# defaulted 标记);显式 "v1" = 兜底 flag,不展开、逐字节保留 legacy 现状(只记一条 v1 标记);
# 别的值 fail-loud。老配方(配了 motion_scale_in_window / adaptive_sigma=false 的)在默认
# 路径上会响亮失败——这是有意的:legacy 配方必须显式声明 reward_pack=v1 才能渲染。
# v2 的人话:
#   * 模仿全身全程全额 —— full_body_mimic=true(pelvis+6 腿回名单),窗内不再给模仿打折
#     (motion_scale_in_window 在 v2 里被废除,与包同时显式配它属于矛盾配方,直接 raise);
#   * 平衡从"税"改"工资" —— 收入型 upright_exp(+1.0)替代税型 upright(flat_orientation_l2
#     清零);
#   * 窗内站稳四件套(strike_upright/ang_vel/foot_vel/vbob)删掉,换 PACE 单条
#     hit_unstable_support(-10.0,击球窗内单脚/无支撑记罚);
#   * 反作弊小税清零交给全身模仿 —— foot_orientation / prestrike_upright /
#     prestrike_waist_twist / arm_overreach / hold_ready 全部 0;
#   * foot slip/drag 降到 mjlab 档位 —— foot_slip_sq -0.1、foot_drag 0;
#   * 拍面 sigma 第三通道 —— racket.adaptive_sigma_normal=true(必须搭在 adaptive_sigma 上,
#     半配组合在翻译层就拒;包【默认生效】且用户没显式表态 adaptive_sigma 时,默认包代为
#     置 true 保持自洽——显式 v2 配方仍要求自己声明);
#   * 击球三通道名义权重 —— racket_position/velocity/normal = 60/45/35(§3.5 L3;名义值,
#     probe 校准后冻结 prereg);
#   * 击中层 —— strike_capture_bonus 850(L2 one-shot,绑 vb_fired capture 门)、
#     racket_strike_success 30(L3.5 三核乘法加强层);同为名义值,probe 校准后冻结 prereg。
# 展开顺序:包【先】展开,再走正常覆写层 —— 显式 task.rewards.* / task.racket.* 同名键
# "后写后赢"(包只填用户没写的键,直接写 cfg 的项今天没有 CLI 键,不存在冲突面)。
# --------------------------------------------------------------------------------------------- #
_REWARD_PACK_ALLOWED = ("v1", "v2")
# 走【现有键控覆写层】的部分:注入 task.rewards 同名键(必须都在 _REWARD_KEYS 白名单里,
# import 时自检),用户显式写了同名键 -> 包不填,用户赢。
_REWARD_PACK_V2_KEYED = (
    ("full_body_mimic", True),           # 全身模仿:pelvis+6 腿回四个 motion_body_* 名单
    ("hold_ready_weight", 0.0),          # hold 工资清零(hold 场景现被 planner 禁;rally 课程回归时再全额)
    ("foot_orientation_weight", 0.0),    # 脚姿税下岗,交给全身模仿
    ("prestrike_upright_weight", 0.0),   # 挥前站正税下岗
    ("prestrike_waist_twist_weight", 0.0),  # 挥前拧腰税下岗
    ("foot_slip_sq_weight", -0.1),       # 触地脚蹭滑降到 mjlab 档位(v1 源码 -1.0)
    ("foot_drag_weight", 0.0),           # 拖脚税下岗(v1 源码 -0.5)
    ("foot_soft_landing_weight", -0.003),  # 落地冲击罚(蓝图 §2.4 档位;07-26 配方审计补漏——
                                           # 此前包漏设,三条在跑科学臂落地冲击罚=0)
    ("action_acc_weight", -0.05),        # 二阶平滑(mjlab 1/4 档;clamp 36.0 由 direct-params 落)
    # L3 击球三通道(redesign §3.5:名义值,probe 校准后冻结 prereg)。用户显式键照旧赢——
    # 但这三条是【标定过的冻结数】,压过它们必须响亮记账,见 _REWARD_PACK_V2_CALIBRATED。
    ("racket_position_weight", 393.4),   # 触点尖峰位置核(v4rg probe 冻结 07-26;k_eff 口径)
    ("racket_velocity_weight", 295.1),   # 拍速核(v4rg 冻结)
    ("racket_normal_weight", 229.5),     # 拍面核(v4rg 冻结)
)
# ------------------------------------------------------------------------------------------- #
# 静默 no-op 防线(2026-07-27)。事故:上面三条冻结质量权重【从未在任何一条臂上生效】——
# 每一个 cfg/task/*.yaml 都显式写了这三键(VirtualBall 4.0/0.5/0.5、Hitter 谱系 14/10/5),
# 而包的规则是"显式键赢,包不碰",于是包在质量三通道上是死码。旧记账行
#   "rewards.racket_position_weight explicitly set — user override wins"
# 【不带数值】,所以三位作者读了 applied 日志仍没看出压过的是 4.0 vs 393.4(98×):
#   - docs/experiments/2026-07/EXP-V2-REWARD-FREEZE-20260726.md:8   把对照臂写成"冻结表全默认"
#   - 同文件 §0.11 把正手 face 死区归因于"位置核(393)"——在跑的其实是 4.0
#   - docs/research/reward_v2_explained_20260725.md 把冻结值当默认路径生效值
# 修法(不改任何在跑臂的行为,默认路径逐字节不变):
#   ① 记账行带上【双值+倍率】,压过冻结数这件事在 applied 日志里可读、可 grep、可断言;
#   ② 偏离超容差时【额外打 WARNING】——按发射工序纪律 WARN 必进摘要;
#   ③ 想让冻结表不可被静默压过的臂,显式 rewards.reward_pack_strict=true → 直接 fail-loud。
# 为什么不学 action_rate_weight 的"剥离+记账":剥离=让 393.4 生效=把每一条新发射的臂
# 质量权重悄悄乘 ~100,比现在的缺陷更危险;为什么不无条件 raise:任务 yaml 全都带这三键,
# 无条件 raise 会炸掉每一次 default-v2 boot(7263464b 已经踩过一次这个坑)。
# key -> (冻结值, 人话)
_REWARD_PACK_V2_CALIBRATED = {
    "racket_position_weight": (393.4, "质量核·拍位(v4rg probe 冻结)"),
    "racket_velocity_weight": (295.1, "质量核·拍速(v4rg probe 冻结)"),
    "racket_normal_weight": (229.5, "质量核·拍面(v4rg probe 冻结)"),
}
# 相对偏差超过这个比例才算"压过冻结数"(同值/浮点噪声不报警)。
_REWARD_PACK_CALIBRATED_TOL = 1e-6
# 今天没有 CLI 键的项:直接写 cfg 对象上的 weight(与 free_non_striking_arm_mimic 等
# direct-cfg 改动同款,逐条记 applied,标 reward_pack=v2)。
_REWARD_PACK_V2_DIRECT = (
    ("strike_upright", 0.0),             # 窗内站稳四件套下岗……
    ("strike_ang_vel", 0.0),
    ("strike_foot_vel", 0.0),
    ("strike_vbob", 0.0),
    ("hit_unstable_support", -10.0),     # ……换 PACE 单条(B1 已在 cfg 声明 weight=0 待命)
    ("upright", 0.0),                    # 税型站正(flat_orientation_l2)下岗……
    ("upright_exp", 1.0),                # ……换收入型站正(B1 已声明 weight=0 待命)
    ("arm_overreach", 0.0),              # 伸臂过远税下岗,交给全身模仿
    # 击中/结果层(Franco 2026-07-25 v2.1 裁定:代理全删、分开学、上台扛大奖)。
    # 三核乘积(strike_success)是"结果"的人造 AND 代理,capture_bonus 是它的二值化——
    # 而落点核才是物理正确的 AND(落点=拍位x拍速x拍面经球物理联合决定,梯度经解析
    # 接触模型自动分摊回三通道)。有物理组合项,两个代理都是重复,删。
    ("racket_strike_success", 0.0),      # v1 谱系默认 5.0 -> v2 删(人造 AND 代理)
    # strike_capture_bonus 不进包:cfg 默认 0 即"不存在"(capture 门保留原职=上台组闸门)
    # v2.2(Franco 07-25):上台组只留 landing 一项——"过网+落台"是先决条件(gate)
    # 而非单独给钱的项;pass_net 的过网高塑形随之下岗(先决条件由 gate 表达)。
    ("virtual_pass_net", 0.0),
    ("virtual_landing", 1648.8),         # 唯一每拍大奖(v4rg 冻结:18.46×46.3/(0.6×0.864))
                                         # 阶梯 1:3:7.5 锚实测模仿收入(Franco 07-26 终裁);换动作谱系必须重 probe 重定
    ("virtual_spin", 0.0),               # 弧圈类动作自带旋转,minimize 先验打架动作身份;
                                         # 遥测保留;落点预测本就旋转感知(RK4 含 Magnus)
    # 值封顶平滑(fresh 自杀区间的解,冻结表档位):无封顶 action_rate_l2 归零,换封顶版。
    ("action_rate_l2", 0.0),
    ("action_rate_clamped", -0.2),
    # 统一灾难价(07-28 Franco 终裁):fall/table/hard-qdes/hard-actual 都只经 generic
    # termination 收一次。−3600×policy_dt(0.02 s)=−72，严格高于满分上台约 +33，
    # 关死 reset/death 套利；具名原因只分账，不叠加第二份罚。
    ("death_penalty", -3600.0),
)
# 包里的【可选】项:term 不存在时【跳过并记账】,不 fail-loud。
# 与上面 DIRECT 的区别就是这一条,理由也只有一条:DIRECT 的 fail-loud 是在说"这个 cfg 血统根
# 本不长 v2 要动的项",而可选项的缺席是一个【合法配置】——桌子被 task.table_obstacle=false
# 关掉时 table_hit_penalty 会被 apply_table_obstacle 一并撤走,这不是配错,是无桌对照臂。
_REWARD_PACK_V2_OPTIONAL = (
    # 桌碰仍是独立 hard-unsafe termination/counter，但不再叠加 reason-specific reward。
    # generic death 已给 −72；这里固定 0，防同一 terminal transition 被收两次。
    ("table_hit_penalty", 0.0),
)
# v2.2 direct-params:landing 换 legal_base 语义(v1 climb 字节等价保留在函数默认)。
# 延付(settle_delay_s)07-26 Franco 裁决:默认关、降级为消融 flag——generic death
# + stand_start 已双重关死刷分回路,延付的边际价值待消融检验(臂级用
# rewards.virtual_landing_settle_delay_s 显式开)。
_REWARD_PACK_V2_LANDING_PARAMS = {"mode": "legal_base", "base_frac": 0.6, "settle_delay_s": 0.0}
# 建表自检(import 时就炸):键控注入表的键必须都在 _REWARD_KEYS 白名单里,否则
# _check_unknown_keys 会把注入后的 mapping 当异常配置拒掉(虽然注入发生在检查之后,
# 表键漂移仍属于契约漂移,响亮报)。
_PACK_TABLE_STRAYS = sorted(
    {key for key, _ in _REWARD_PACK_V2_KEYED} - set(_REWARD_KEYS)
)
if _PACK_TABLE_STRAYS:
    raise RuntimeError(
        "[train.py] _REWARD_PACK_V2_KEYED keys missing from _REWARD_KEYS: "
        f"{_PACK_TABLE_STRAYS}"
    )
del _PACK_TABLE_STRAYS


def _calibrated_override_marker(key, override, *, strict):
    """显式键压过 reward_pack=v2 时的记账行(冻结数额外带双值+WARNING/fail-loud)。

    返回一条 applied 记账字符串。对 ``_REWARD_PACK_V2_CALIBRATED`` 里的【标定冻结数】:

    * 记账行带上 ``<显式值> over frozen <包值> (ratio Nx)`` —— 旧行不带数值,正是三处
      记录把"4.0 在跑"误写成"393.4 在跑"的原因;带了数值就能被日志审计和测试断言抓到。
    * 偏离超容差时另打一条 ``[train.py] WARNING:`` 到 stdout(发射工序:WARN 必进摘要)。
    * ``rewards.reward_pack_strict=true`` 时直接 fail-loud —— prereg 冻结臂用它保证
      "冻结表在跑"这句话不可能再变成假话。

    非冻结键(布尔/清零类)保持原语义与原措辞,一个字都不变。
    """
    calibrated = _REWARD_PACK_V2_CALIBRATED.get(key)
    if calibrated is None:
        return (
            f"rewards.{key} explicitly set — user override wins (reward_pack=v2 keeps hands off)"
        )
    frozen, human = calibrated
    try:
        ratio = float(override) / float(frozen) if float(frozen) else float("inf")
    except (TypeError, ValueError):
        ratio = float("nan")
    differs = not (ratio == ratio) or abs(ratio - 1.0) > _REWARD_PACK_CALIBRATED_TOL
    detail = (
        f"rewards.{key}={override!r} explicitly set — user override wins over reward_pack=v2 "
        f"FROZEN {frozen!r} (ratio {ratio:.4g}x; {human})"
    )
    if not differs:
        return detail + " [same value]"
    if strict:
        raise _OverrideError(
            f"[train.py] rewards.{key}={override!r} would silently defeat the calibrated "
            f"reward_pack=v2 frozen value {frozen!r} (ratio {ratio:.4g}x; {human}), and "
            "rewards.reward_pack_strict=true forbids that. Either drop the key from the task "
            "yaml / CLI so the frozen table applies, or set reward_pack_strict=false (default) "
            "to keep the override with a loud WARNING."
        )
    print(
        f"[train.py] WARNING: reward_pack=v2 FROZEN {key}={frozen!r} is DEFEATED by an explicit "
        f"task.rewards value {override!r} ({ratio:.4g}x). {human}. Every cfg/task/*.yaml declares "
        "this key, so the frozen quality table does NOT run unless you delete it there or pass a "
        "deliberate CLI value; do not describe this run as 'frozen table defaults'. Set "
        "rewards.reward_pack_strict=true to make this a hard error.",
        flush=True,
    )
    return detail


def _expand_reward_pack(env_cfg, task, rw, applied):
    """task.rewards.reward_pack 展开("v2" 展开成套;"v1" 兜底不展开)。返回 rewards 覆写 mapping。

    2026-07-25 Franco 裁定默认翻转:键缺席 = 默认按 "v2" 展开(applied 记 defaulted 标记);
    显式 "v1" = legacy 兜底,原 mapping 原样返回、逐字节不变(只记一条 v1 标记)。

    人话:包不是绕开覆写层的后门 —— 有 CLI 键的项注入进 task.rewards mapping,由下面的
    【现有翻译层】落地(_set_attr/_set_reward 照常记 applied、照常做类型/契约校验);没有
    CLI 键的项直接写 cfg 对象并逐条记 applied。每条改动的 applied 标记都带 reward_pack=v2,
    事后翻 run 记录能一眼看出哪些值是包写的、哪些是用户显式压过包的。
    """
    pack = _get(rw, "reward_pack")
    defaulted = pack is None
    if defaulted:
        pack = "v2"  # 2026-07-25 Franco 裁定:缺席 = v2;要 legacy 基线请显式 reward_pack=v1
    if not isinstance(pack, str) or pack not in _REWARD_PACK_ALLOWED:
        raise _OverrideError(
            f"[train.py] task.rewards.reward_pack must be one of {_REWARD_PACK_ALLOWED} "
            f"(absent = v2 default, 2026-07-25 Franco ruling; v1 = explicit legacy baseline), "
            f"got {pack!r} — unknown pack values never fall back silently."
        )
    if pack == "v1":
        applied.append("rewards.reward_pack=v1 (legacy baseline)")
        return rw  # 显式 v1 兜底:不展开,legacy 行为逐字节不变
    # v2 的定义就是"窗内不给模仿打折":与生效的 v2 包(含默认)同时显式配
    # motion_scale_in_window 属于矛盾配方,fail-loud 而不是静默裁决谁赢(显式 null 按覆写层
    # 惯例等价于"没写",不算冲突)。默认翻转后老配方会在这里响亮失败——有意为之。
    if _get(rw, "motion_scale_in_window") is not None:
        raise _OverrideError(
            "[train.py] rewards.reward_pack=v2 "
            + ("(defaulted, 2026-07-25 Franco ruling) " if defaulted else "")
            + "abolishes the in-window imitation discount, but rewards.motion_scale_in_window "
            "is also explicitly set — contradictory recipe. Legacy recipes must declare "
            "reward_pack=v1 to keep motion_scale_in_window; a v2 recipe must drop the key."
        )
    if defaulted:
        applied.append(
            "rewards.reward_pack defaulted to v2 (2026-07-25 Franco ruling; set "
            "reward_pack=v1 for legacy baseline)"
        )
    else:
        applied.append(
            "rewards.reward_pack=v2 (reward_redesign_20260725 §3; pack expands FIRST, explicit "
            "keys win)"
        )
    # 注入:把 rw 物化成普通 dict(dict / OmegaConf 节点都支持 keys()+get()),只填用户
    # 没写的键;之后整个 rewards 覆写层读的就是这份合并视图。默认路径上 rewards 节点可能
    # 根本不存在(rw is None),从空表起步。
    merged = {} if rw is None else {str(key): _get(rw, key) for key in rw.keys()}
    strict = _get(rw, "reward_pack_strict")
    strict = False if strict is None else _as_explicit_bool(strict, "task.rewards.reward_pack_strict")
    if strict:
        applied.append("rewards.reward_pack_strict=True (frozen pack values may not be overridden)")
    for key, value in _REWARD_PACK_V2_KEYED:
        override = merged.get(key)
        if override is not None:
            applied.append(
                _calibrated_override_marker(key, override, strict=strict)
            )
            continue
        merged[key] = value
        applied.append(f"rewards.{key}={value!r} (reward_pack=v2)")
    # direct-cfg 项:今天没有 CLI 键,直接写 weight;term 缺失/为 None 说明这个 cfg 血统
    # 根本不长 v2 要动的项(例如非 DeployParity 谱系),fail-loud 而不是静默半套换装。
    R = getattr(env_cfg, "rewards", None)
    _require(R is not None, "rewards (reward_pack=v2)")
    for name, weight in _REWARD_PACK_V2_DIRECT:
        if float(weight) == 0.0 and getattr(R, name, None) is None:
            # 退役零标记:该谱系的 cfg 类根本不长这项(如 action_ball 无 virtual_pass_net)。
            # 零权重的缺席不改变收入结构,记账跳过;非零项缺席仍在下方 fail-loud。
            applied.append(
                f"rewards.{name} ABSENT retired-zero -> skipped (reward_pack=v2)"
            )
            continue
        _require(
            hasattr(R, name) and getattr(R, name) is not None,
            f"rewards.{name} (reward_pack=v2)",
        )
        getattr(R, name).weight = float(weight)
        applied.append(f"rewards.{name}.weight={float(weight)} (reward_pack=v2)")
    for name, weight in _REWARD_PACK_V2_OPTIONAL:
        if getattr(R, name, None) is None:
            # 记账不是可选的:哪怕跳过也要在 applied log 里留一行,否则"这臂没有撞桌罚"就只能
            # 靠猜。
            applied.append(f"rewards.{name} ABSENT -> skipped (reward_pack=v2 optional)")
            continue
        getattr(R, name).weight = float(weight)
        applied.append(f"rewards.{name}.weight={float(weight)} (reward_pack=v2 optional)")
    # v2.2:landing 切 legal_base 语义(过网+落台=先决条件,门内底薪+中心核梯度)。
    _landing = getattr(R, "virtual_landing")
    _require(
        isinstance(_landing.params, dict) and "command_name" in _landing.params,
        "rewards.virtual_landing.params (reward_pack=v2 landing mode swap)",
    )
    _landing.params.update(_REWARD_PACK_V2_LANDING_PARAMS)
    applied.append(
        f"rewards.virtual_landing.params+={_REWARD_PACK_V2_LANDING_PARAMS} (reward_pack=v2)"
    )
    # v2 值封顶:action_acc 的 clamp 参数(weight 走 KEYED 的 action_acc_weight)。
    _acc = getattr(R, "action_acc_l2", None)
    _require(_acc is not None, "rewards.action_acc_l2 (reward_pack=v2 value clamp)")
    _acc.params["value_clamp"] = 36.0
    applied.append("rewards.action_acc_l2.params.value_clamp=36.0 (reward_pack=v2)")
    # v2 用封顶版平滑:任务 YAML 谱系基线普遍带 action_rate_weight(如 DeployParity -0.10),
    # 包在此【剥离】该键并记账(防双计费;不 raise——基线键不是用户矛盾配方,legacy 语义
    # 要保留请显式 reward_pack=v1)。剥离后 DIRECT 的 action_rate_l2=0 + action_rate_clamped
    # 生效,封顶平滑单一计费。
    if _get(rw, "action_rate_weight") is not None:
        dropped = merged.pop("action_rate_weight", None)
        applied.append(
            f"rewards.action_rate_weight={dropped!r} dropped (reward_pack=v2 uses "
            "value-clamped action_rate_clamped; declare reward_pack=v1 for legacy)"
        )
    # sigma 体系(07-26 Franco 裁决:adaptive sigma 在新体系退役)——早期采集不靠质量核
    # (模仿+站正+progress+landing 是收入主链),晚期精度由 capture/legal 门与落点核管;
    # σ 静态钉在验收档(cfg 默认 0.075/0.5/0.262,与 k_eff 校准口径一致,k_eff 就是在
    # σ=验收档状态下实测的)。机制代码保留:显式 task.racket.adaptive_sigma[_normal]=true
    # 仍可开(消融 flag;SMASH 的反向证据在别人的栈里,值得空槽时 A/B 一次)。包不再
    # 触碰 racket cfg。
    # motion 侧(07-26 Franco:"stand_start_prob=1.0 作为 default"):防"挥拍中段 RSI
    # 空降→借参考动量刷分"是 v2 大奖结构的配套件,进包。用户显式 task.motion.* 键在
    # 后面的 motion 翻译层覆写(后写后赢);canonical_ready_mode 谱系天然 frame-0 起步,
    # 本设置无害。post_swing_start_prob 一并归零,避免与 stand=1.0 组成 sum>1 的抽签面。
    _require(
        hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"),
        "commands.motion (reward_pack=v2 -> stand starts)",
    )
    M = env_cfg.commands.motion
    mo = _get(task, "motion")
    for attr, target in (("stand_start_prob", 1.0), ("post_swing_start_prob", 0.0)):
        if _get(mo, attr) is not None:
            applied.append(
                f"motion.{attr} explicitly set — user override wins (reward_pack=v2 keeps hands off)"
            )
            continue
        _require(hasattr(M, attr), f"commands.motion.{attr} (reward_pack=v2)")
        setattr(M, attr, float(target))
        applied.append(f"motion.{attr}={float(target)} (reward_pack=v2 stand starts)")
    return merged


# YAML keys under `terminations:` (R-b envelope-termination softening, reward_staged_design §⑥;
# R9 lower-body-free ablation anchor_pos_off / ee_upper_only, franco 拍板 2026-07-08).
_TERMINATION_KEYS = ("envelope_as_penalty", "envelope_penalty_weight",
                     "anchor_pos_off", "ee_upper_only")


def _registry_clip_name(cfg):
    """Motion clip name used to key per-motion settings (e.g. ``strike_phase_by_motion``).

    Resolution order (most explicit wins): CLI ``registry_name`` -> explicit ``motion_file`` (its parent
    dir, the artifact folder such as ``hope_backhand:v0``) -> the task default ``registry_name``. Any
    registry path prefix and ``:version`` suffix are stripped, so the result is e.g. ``hope_forehand`` /
    ``hope_backhand``. Returns ``None`` when nothing is set. Shared by train/play/probe so all three pick
    the same per-clip strike phase.
    """
    reg = _get(cfg, "registry_name")  # CLI override (train/play): wins over the forehand default
    if not _is_noneish(reg):
        return str(reg).split("/")[-1].split(":")[0] or None
    mf = _motion_clip_name_from_path(_get(cfg, "motion_file"))
    if mf is not None:
        return mf
    task = _get(cfg, "task")
    reg = _get(task, "registry_name") if task is not None else None  # task default (forehand): last
    if not _is_noneish(reg):
        return str(reg).split("/")[-1].split(":")[0] or None
    return None


def _resolve_strike_phase(rk, clip_name):
    """Select the strike phase for the trained clip (paddle-contact frame is PER-CLIP).

    A single global ``strike_phase`` cannot serve both swings (the racket-tip speed peak lands at a
    different fraction in each clip, e.g. forehand ~0.46 vs backhand ~0.59), so ``strike_phase_by_motion``
    maps a motion-name substring (the registry clip name) to its contact phase; the most-specific
    (longest) matching key wins. Falls back to the scalar ``strike_phase`` when nothing matches or the
    clip is unknown. Returns ``(phase_or_None, note_or_None)``; ``note`` records which mapping fired.
    """
    by_motion = _get(rk, "strike_phase_by_motion")
    if by_motion is not None and clip_name:
        cn = str(clip_name).lower()
        matches = [(str(k).lower(), v) for k, v in by_motion.items()
                   if str(k).lower() in cn or cn in str(k).lower()]
        if matches:
            matches.sort(key=lambda kv: len(kv[0]), reverse=True)  # longest key = most specific
            k, v = matches[0]
            return float(v), f"racket_target.strike_phase<-by_motion[{k}]={float(v)} (clip={clip_name})"
    sp = _get(rk, "strike_phase")
    return (None if sp is None else float(sp)), None


# Legacy YAML clip names: the two FAMILIES, which for a 2-clip run are also the two clips.
# The system is no longer限于两个动作: ``racket.clip_names`` declares an ORDERED per-clip key list
# (e.g. [fh_loop, bh_loop_c, s0_highpress, bh_block, fh_block_syn]) and every per-clip YAML block
# below is keyed by those names, in that order. Absent -> the legacy two names, byte-identical.
_DEFAULT_CLIP_NAMES = ("forehand", "backhand")

#: The physical-validity guards that must EXIST in the checkout a run is launched from.
#: 人话:这三道闸是"目标点不能在桌面以下 / 命令速度必须朝对面 / 参考击球帧必须能把自己的球打
#: 回去"。一个旧 checkout 照样 import 得动、照样跑,只是三道闸一道都不在——训练曲线不会告诉你。
#: 所以发射时对着**真正被 import 的那个模块**核对,而不是对着操作员看的那个仓库。
_REQUIRED_PHYSICAL_GUARDS = (
    ("_assert_contact_clears_table",
     "commanded contact point below the table surface + ball radius"),
    ("_assert_target_velocity_points_forward",
     "commanded return velocity that can point away from the opponent"),
    ("_assert_reference_strike_can_return_its_own_regime",
     "a bound strike frame that cannot return any ball from that clip's own regime"),
)


def _assert_physical_validity_guards_present(racket_cfg):
    """Fail closed when the imported command module predates the physical-validity guards."""
    import inspect as _inspect

    module = sys.modules.get(type(racket_cfg).__module__)
    if module is None:
        raise _OverrideError(
            f"cannot resolve the module of {type(racket_cfg).__name__} to verify the "
            f"physical-validity guards are present in this checkout")
    command_cls = getattr(module, "RacketTargetCommand", None)
    if command_cls is None:
        raise _OverrideError(
            f"{getattr(module, '__file__', module.__name__)} has no RacketTargetCommand — this "
            f"checkout cannot carry the physical-validity guards")
    missing = [(name, what) for name, what in _REQUIRED_PHYSICAL_GUARDS
               if not hasattr(command_cls, name)]
    if missing:
        # Report the MODULE's file: that is the checkout artifact the operator has to update.
        where = getattr(module, "__file__", None)
        if not where:
            try:
                where = _inspect.getsourcefile(command_cls) or "<unknown>"
            except TypeError:
                where = "<unknown>"
        detail = "; ".join(f"{name} (guards against: {what})" for name, what in missing)
        raise _OverrideError(
            f"launch source check: the checkout being trained from ({where}) is missing "
            f"{len(missing)} of {len(_REQUIRED_PHYSICAL_GUARDS)} physical-validity guard(s): "
            f"{detail}. It would import and train cleanly with those checks silently absent. "
            f"Update the checkout on this machine before launching")


def _physical_validity_guards_required(racket_cfg) -> bool:
    """Whether the composed command can execute one of the guarded physical constructions.

    This is deliberately a launch-level decision, not a side effect of translating every
    unrelated ``task.racket`` override.  The three guarded constructions run for
    reference/solved/formal action modes, for a non-empty bank, or for the per-clip boxes whose
    command constructor invokes the table-clearance/forward-velocity checks.
    """

    if racket_cfg is None:
        return False
    target_mode = str(getattr(racket_cfg, "target_mode", "") or "")
    if target_mode in {
        "reference_perturbed",
        "solved",
        "task_first",
        "action_ball",
    }:
        return True
    if str(getattr(racket_cfg, "question_bank", "") or "").strip():
        return True
    if str(getattr(racket_cfg, "base_couple_mode", "") or "") == "reference_reach":
        return True
    if tuple(getattr(racket_cfg, "clip_names_per_clip", ()) or ()):
        return True
    return any(
        getattr(racket_cfg, name, None) is not None
        for name in (
            "racket_pos_range_per_clip",
            "racket_vel_range_per_clip",
            "vb_vel_range_per_clip",
        )
    )


def _resolve_clip_names(rk):
    """The ORDERED per-clip key list every per-clip YAML block is addressed by.

    人话:以前"每 clip 一行"的框表只认 forehand/backhand 两个名字,所以五个动作根本写不出来,
    而且同一族的两个 clip 会撞成一行。现在在 racket 下写一行::

        clip_names: [fh_loop, bh_loop_c, s0_highpress, bh_block, fh_block_syn]

    顺序 = motion_file 的 clip 顺序 = clip_id。不写就还是老的两个名字,现役 YAML 逐字节不变。
    """
    raw = _get(rk, "clip_names")
    if raw is None:
        return _DEFAULT_CLIP_NAMES
    names = tuple(str(n).strip().lower() for n in raw)
    if not names:
        raise _OverrideError("racket.clip_names: empty list — give one name per loaded clip")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise _OverrideError(
            f"racket.clip_names: duplicate name(s) {dupes} — clip names index clip_id, so each "
            f"must be unique (two clips of the same family need two DIFFERENT names, which is "
            f"exactly the collision this list exists to remove)")
    return names


def _resolve_range_per_clip(rk, key, names):
    """Shared parser for the per-clip 3-axis box blocks, keyed by ``names`` in order.

    YAML (each axis a [lo, hi] list)::

        vel_range_per_clip:
          fh_loop:   {x: [1.5, 3.5], y: [-1.0, 1.0], z: [0.0, 1.5]}
          bh_block:  {x: [1.2, 2.4], y: [-1.0, 1.0], z: [0.0, 1.2]}
          ...

    Returns a tuple indexed by clip_id of ``((xlo,xhi),(ylo,yhi),(zlo,zhi))``, or ``None`` when the
    key is absent (-> keep the shared scalar box; backward compatible).

    FAIL-CLOSED on length: the block must name EVERY clip in ``names`` exactly once. The old parser
    built ``tuple(by_id[i] for i in range(len(by_id)))``, so a block that named only some clips
    produced a SHORT table that then silently rode the two-row family expansion, or KeyError'd on
    an unrelated index.
    """
    block = _get(rk, key)
    if block is None:
        return None
    index = {n: i for i, n in enumerate(names)}
    by_id = {}
    for name in block:
        cid = index.get(str(name).strip().lower())
        if cid is None:
            raise _OverrideError(
                f"racket.{key}: unknown clip name {name!r} — expected one of {list(names)} "
                f"(set racket.clip_names to declare the per-clip key order)")
        axes = _get(block, name)

        def _r(ax):
            v = _get(axes, ax)
            if v is None:
                raise _OverrideError(f"racket.{key}[{name}]: missing '{ax}' [lo,hi] range")
            return (float(v[0]), float(v[1]))

        by_id[cid] = (_r("x"), _r("y"), _r("z"))
    missing = [n for i, n in enumerate(names) if i not in by_id]
    if missing:
        raise _OverrideError(
            f"racket.{key} has {len(by_id)} row(s) but racket.clip_names declares "
            f"{len(names)} clip(s); missing {missing}. Give one row per clip — a short table "
            f"would silently be re-used for the clips it does not mention")
    return tuple(by_id[i] for i in range(len(names)))


def _resolve_vel_range_per_clip(rk, names=_DEFAULT_CLIP_NAMES):
    """PER-CLIP racket target-VELOCITY boxes. See :func:`_resolve_range_per_clip`.

    Each clip has its own natural strike speed, so a shared box overshoots the slower ones.
    """
    return _resolve_range_per_clip(rk, "vel_range_per_clip", names)


def _resolve_pos_range_per_clip(rk, names=_DEFAULT_CLIP_NAMES):
    """PER-CLIP racket target-POSITION boxes (ADDED to the env origin; y is SIGNED).

    Lets each clip's target track its own reference strike point (e.g. a backhand that sits higher
    and further forward at its strike phase, which a shared z<=1.05 box makes unreachable).
    """
    return _resolve_range_per_clip(rk, "pos_range_per_clip", names)


def _resolve_vb_vel_range_per_clip(rk, names=_DEFAULT_CLIP_NAMES):
    """PER-CLIP INCOMING-BALL velocity boxes — the per-clip ball regime.

    人话:这是"给挡球喂快球、给拉球喂慢球"的开关。以前来球速度只有一个全局框
    (``vb_vel_x/y/z_range``),所有 clip 共用,于是一个动作的最佳来球速度落在框外就永远得低分,
    而这恰恰是 bh_loop_c 在自己速度下峰值 5.12 m/s(框是 2.0-4.6)只有 0.150 分的机制。

    Shaped exactly like ``vel_range_per_clip``; absent -> the shared ``vb_vel_*_range`` box,
    byte-identical to today.
    """
    return _resolve_range_per_clip(rk, "vb_vel_range_per_clip", names)


def _resolve_vb_spin_abs_max_per_clip(rk, names=_DEFAULT_CLIP_NAMES):
    """PER-CLIP incoming-ball |spin| ceiling (rad/s), keyed like the boxes above.

    A block answering a heavy topspin loop and a loop answering a float serve are different
    regimes; absent -> the scalar ``vb_spin_abs_max`` for every clip, byte-identical.
    """
    block = _get(rk, "vb_spin_abs_max_per_clip")
    if block is None:
        return None
    if isinstance(block, (list, tuple, ListConfig)):
        vals = [float(v) for v in block]
        if len(vals) != len(names):
            raise _OverrideError(
                f"racket.vb_spin_abs_max_per_clip has {len(vals)} entries but racket.clip_names "
                f"declares {len(names)} clip(s) — give one per clip, in clip order")
        return tuple(vals)
    index = {n: i for i, n in enumerate(names)}
    by_id = {}
    for name in block:
        cid = index.get(str(name).strip().lower())
        if cid is None:
            raise _OverrideError(
                f"racket.vb_spin_abs_max_per_clip: unknown clip name {name!r} — expected one of "
                f"{list(names)}")
        by_id[cid] = float(_get(block, name))
    missing = [n for i, n in enumerate(names) if i not in by_id]
    if missing:
        raise _OverrideError(
            f"racket.vb_spin_abs_max_per_clip is missing {missing} — give one value per clip")
    return tuple(by_id[i] for i in range(len(names)))


def _resolve_ref_vel_scale(rk, clip_name):
    """Select the reference racket-velocity scale for the trained clip (PER-CLIP, like strike_phase).

    ``ref_vel_scale`` <1.0 trains a slower-than-reference hit. It was tuned to TAME the violent forehand
    (~6 m/s tip) — but the backhand is already a gentle swing (~3.3 m/s tip / ~1.8 m/s at the mount), so
    down-scaling it shrinks the velocity TARGET into the body-jitter floor AND pits the imitation prior
    (wants full speed) against the velocity goal (wants 0.6x). So the scale must be per-clip:
    ``ref_vel_scale_by_motion`` maps a motion-name substring to its scale (longest match wins); falls back
    to the scalar ``ref_vel_scale``. Returns ``(scale_or_None, note_or_None)``.
    """
    by_motion = _get(rk, "ref_vel_scale_by_motion")
    if by_motion is not None and clip_name:
        cn = str(clip_name).lower()
        matches = [(str(k).lower(), v) for k, v in by_motion.items()
                   if str(k).lower() in cn or cn in str(k).lower()]
        if matches:
            matches.sort(key=lambda kv: len(kv[0]), reverse=True)  # longest key = most specific
            k, v = matches[0]
            return float(v), f"racket_target.ref_vel_scale<-by_motion[{k}]={float(v)} (clip={clip_name})"
    rv = _get(rk, "ref_vel_scale")
    return (None if rv is None else float(rv)), None


# task.plant whitelist. zero_joint_friction is the legacy schema-v3 zero-friction opt-in; the
# 2026-07-22 ground/terrain keys follow the same precedent: default ABSENT = byte-identical
# current recipe (plane terrain, ground material 1.0/1.0, robot-body material randomization
# (0.3,1.6)/(0.3,1.2)), any explicit value enters the schema-3 ground_plant contract block.
_PLANT_KEYS = (
    "zero_joint_friction",
    "ground_static_friction",
    "ground_dynamic_friction",
    "robot_material_static_friction_range",
    "robot_material_dynamic_friction_range",
    "robot_material_make_consistent",
    "terrain_rough_height_range",
)
_GROUND_PLANT_TASK_KEYS = _PLANT_KEYS[1:]


def _attach_rough_ground_patch(env_cfg, height_range):
    """Per-env 零均值凹凸地垫的挂载 seam(真身在包内 terrain_patch;lazy import:Kit 起来后
    才 import 得动;host 测试 monkeypatch 这个名字)。

    人话:2026-07-29 抬脚地形修复。旧的 ``terrain_type="generator"`` 全局地形会把
    ``scene.env_origins`` 换成地形 tile 原点,而克隆出来的静态桌子还钉在 GridCloner 网格上
    ——机器人被传送到没有自己桌子的地方;而且 env_spacing=2.5 m 下邻居的桌子足迹和本 env 的
    机器人活动区在空间上重叠,一张共享地面 mesh 根本做不到"我这里凹凸、你桌下平整"。改成
    每个 env 自己的静态凹凸垫(shadow-table 同款 ENV_REGEX_NS+碰撞过滤先例):凹凸只铺机器人
    一侧,高度以 0 为平均(±(hi-lo)/2),桌子足迹强制平在 z=0,桌面/动作库/虚拟球标定全不动。
    """
    from whole_body_tracking.tasks.tracking import terrain_patch

    return terrain_patch.attach_rough_ground_patch(env_cfg, height_range)


def _apply_ground_plant_task_override(env_cfg, plant, applied):
    """task.plant 地面摩擦 / 机器人材质随机化范围 / 随机凹凸地形(默认全缺席 = 字节等价)。

    fail-loud 信封与 qbar/foot 先例一致:值先全部验完才动 cfg;显式 null 的
    terrain_rough_height_range 等价于缺席(= 平地现状)。谱系保护不靠这里——靠 schema-3
    ground_plant 合同块 + resume 时的 _contract_diff(平地 checkpoint 上粗糙地会被拒)。
    """
    raws = {key: _get(plant, key) for key in _GROUND_PLANT_TASK_KEYS}
    if all(value is None for value in raws.values()):
        return

    def _plant_number(raw, label):
        if isinstance(raw, bool):
            raise _OverrideError(f"task.plant.{label} must be a finite number >= 0")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise _OverrideError(
                f"task.plant.{label} must be a finite number >= 0"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise _OverrideError(f"task.plant.{label} must be a finite number >= 0")
        return value

    def _plant_range(raw, label):
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, ListConfig)):
            raise _OverrideError(f"task.plant.{label} must be a [lo, hi] pair")
        items = list(raw)
        if len(items) != 2:
            raise _OverrideError(f"task.plant.{label} must be a [lo, hi] pair")
        lo = _plant_number(items[0], f"{label}[0]")
        hi = _plant_number(items[1], f"{label}[1]")
        if lo > hi:
            raise _OverrideError(f"task.plant.{label} must satisfy 0 <= lo <= hi")
        return lo, hi

    scene = getattr(env_cfg, "scene", None)
    terrain = None if scene is None else getattr(scene, "terrain", None)

    # 1) 地面材质摩擦(scene.terrain.physics_material;现状 1.0/1.0)。动摩擦不得大于静摩擦
    # ——用"显式值或现值"做交叉检查,单边覆盖也逃不过。
    ground_static_raw = raws["ground_static_friction"]
    ground_dynamic_raw = raws["ground_dynamic_friction"]
    material = None if terrain is None else getattr(terrain, "physics_material", None)
    if ground_static_raw is not None or ground_dynamic_raw is not None:
        _require(
            material is not None
            and hasattr(material, "static_friction")
            and hasattr(material, "dynamic_friction"),
            "scene.terrain.physics_material (task.plant ground friction)",
        )
        static_value = (
            _plant_number(ground_static_raw, "ground_static_friction")
            if ground_static_raw is not None
            else float(material.static_friction)
        )
        dynamic_value = (
            _plant_number(ground_dynamic_raw, "ground_dynamic_friction")
            if ground_dynamic_raw is not None
            else float(material.dynamic_friction)
        )
        if dynamic_value > static_value:
            raise _OverrideError(
                "task.plant ground dynamic friction must not exceed static friction "
                f"(effective static={static_value}, dynamic={dynamic_value})"
            )
        if ground_static_raw is not None:
            material.static_friction = static_value
            applied.append(
                f"scene.terrain.physics_material.static_friction={static_value}"
            )
        if ground_dynamic_raw is not None:
            material.dynamic_friction = dynamic_value
            applied.append(
                f"scene.terrain.physics_material.dynamic_friction={dynamic_value}"
            )

    # 2) 机器人 body 材质随机化范围(events.physics_material;现状 (0.3,1.6)/(0.3,1.2) ——
    # 下界 0.3 等效"脚地很滑",grip 臂就是要把它抬高)。
    for key, param in (
        ("robot_material_static_friction_range", "static_friction_range"),
        ("robot_material_dynamic_friction_range", "dynamic_friction_range"),
    ):
        raw = raws[key]
        if raw is None:
            continue
        lo, hi = _plant_range(raw, key)
        events = getattr(env_cfg, "events", None)
        term = None if events is None else getattr(events, "physics_material", None)
        params = None if term is None else getattr(term, "params", None)
        _require(
            isinstance(params, dict) and param in params,
            f"events.physics_material.params['{param}'] (task.plant.{key})",
        )
        params[param] = (lo, hi)
        applied.append(f"events.physics_material.params.{param}=({lo}, {hi})")

    # 2.5) 静/动摩擦物理一致性(2026-07-29)。人话:isaaclab 的材质随机化默认静、动独立采样,
    # 约 1/3 的桶会采到 动>静 的非物理组合;显式 true 让每个桶 dynamic=min(static, dynamic)。
    # false 的唯一拼写是缺席/null(= 现状独立采样,字节等价),写 false 也当没写。
    mc_raw = raws["robot_material_make_consistent"]
    if mc_raw is not None:
        if not isinstance(mc_raw, bool):
            raise _OverrideError(
                "task.plant.robot_material_make_consistent must be a bool"
            )
        if mc_raw:
            events = getattr(env_cfg, "events", None)
            term = None if events is None else getattr(events, "physics_material", None)
            params = None if term is None else getattr(term, "params", None)
            _require(
                isinstance(params, dict),
                "events.physics_material.params (task.plant.robot_material_make_consistent)",
            )
            params["make_consistent"] = True
            applied.append(
                "events.physics_material.params.make_consistent=True "
                "(逐桶 dynamic=min(static, dynamic),动摩擦不再超过静摩擦)"
            )

    # 3) 随机凹凸地垫(显式 null = 平地现状,零改动)。只允许从 plane 起点切过去——起点不是
    # plane 说明配置血统不对,拒绝。带宽 [lo, hi] 会被居中到 0(±(hi-lo)/2,5 mm 量化),所以
    # 太窄的带会量化成死平垫,同样拒绝。挂载细节见 terrain_patch.attach_rough_ground_patch。
    rough_raw = raws["terrain_rough_height_range"]
    if rough_raw is not None:
        lo, hi = _plant_range(rough_raw, "terrain_rough_height_range")
        if hi <= 0.0:
            raise _OverrideError(
                "task.plant.terrain_rough_height_range requires hi > 0 meters "
                "(explicit flat ground is spelled null/absent)"
            )
        if hi > 0.5:
            raise _OverrideError(
                "task.plant.terrain_rough_height_range hi > 0.5 m is not a plausible "
                "arena floor"
            )
        if (hi - lo) < 0.01 - 1e-12:
            raise _OverrideError(
                "task.plant.terrain_rough_height_range band (hi - lo) must be >= 0.01 m: "
                "heights are re-centred to ±(hi-lo)/2 about z=0 and quantized at 5 mm, a "
                f"narrower band builds a dead-flat pad (got [{lo}, {hi}])"
            )
        if (hi - lo) > 0.15 + 1e-12:
            raise _OverrideError(
                "task.plant.terrain_rough_height_range band (hi - lo) must be <= 0.15 m: "
                "beyond that the height-field slope wall correction pulls below-zero "
                f"vertices onto the flat table boundary column (got [{lo}, {hi}])"
            )
        _band_ratio = ((hi - lo) / 2.0) / 0.005
        if abs(_band_ratio - round(_band_ratio)) > 1e-6:
            raise _OverrideError(
                "task.plant.terrain_rough_height_range band (hi - lo) must be a multiple "
                "of 0.01 m (heights quantize to 5 mm levels; a non-multiple band would "
                f"silently build a different amplitude than authored; got [{lo}, {hi}])"
            )
        _require(
            terrain is not None and getattr(terrain, "terrain_type", None) == "plane",
            "scene.terrain.terrain_type=='plane' (task.plant.terrain_rough_height_range)",
        )
        for line in _attach_rough_ground_patch(env_cfg, (lo, hi)):
            applied.append(line)


def _ground_plant_contract(env_cfg) -> dict | None:
    """POST-OVERRIDE ground/terrain plant identity for the schema-3 hard contract.

    读的是实例化 env cfg 的实际状态(不是 YAML 回声):地形类型 + 地面材质摩擦 + 机器人材质
    随机化范围。与历史字节默认完全相等 -> None(合同不落键,老 checkpoint resume 逐字节兼容);
    任何偏离 -> 完整 ground_plant 块(rough/滑地 checkpoint 与平地谱系互相拒绝静默续训)。
    按文件路径加载合同模块而不是 import whole_body_tracking 包(包 __init__ 连带注册 Isaac
    任务,host-only 测试 import 不动;照 checkpoint_normalization_preflight 先例)。
    """
    tc = getattr(_ground_plant_contract, "_tc_module", None)
    if tc is None:
        _tc_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
        )
        _tc_spec = importlib.util.spec_from_file_location(
            "hope_training_contract_by_path", _tc_path
        )
        tc = importlib.util.module_from_spec(_tc_spec)
        _tc_spec.loader.exec_module(tc)
        _ground_plant_contract._tc_module = tc
    GROUND_PLANT_TERRAIN_PLANE = tc.GROUND_PLANT_TERRAIN_PLANE
    GROUND_PLANT_TERRAIN_ROUGH = tc.GROUND_PLANT_TERRAIN_ROUGH
    ground_plant_block = tc.ground_plant_block

    scene = getattr(env_cfg, "scene", None)
    terrain = None if scene is None else getattr(scene, "terrain", None)
    patch = None if scene is None else getattr(scene, "rough_ground_patch", None)
    if terrain is not None and patch is None:
        material = getattr(terrain, "physics_material", None)
        if material is None:
            raise RuntimeError(
                "ground-plant contract requires scene.terrain.physics_material"
            )
        terrain_type_raw = getattr(terrain, "terrain_type", None)
        if terrain_type_raw != "plane":
            raise RuntimeError(
                f"ground-plant contract cannot fingerprint terrain_type={terrain_type_raw!r}"
            )
        terrain_type = GROUND_PLANT_TERRAIN_PLANE
        height = None
    elif terrain is None and patch is not None:
        # 2026-07-29 抬脚地形:per-env 零均值凹凸垫替代 TerrainImporter(plane importer 被
        # attach_rough_ground_patch 摘掉,env origins 回落克隆网格)。指纹从垫子的 spawn cfg
        # 读回 AUTHORED 带宽和地面材质。
        spawn = getattr(patch, "spawn", None)
        band = None if spawn is None else getattr(spawn, "height_range_m", None)
        try:
            height = [float(band[0]), float(band[1])]
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                "ground-plant contract requires rough_ground_patch.spawn"
                ".height_range_m=[lo, hi]"
            ) from exc
        material = getattr(spawn, "physics_material", None)
        if material is None:
            raise RuntimeError(
                "ground-plant contract requires rough_ground_patch.spawn.physics_material"
            )
        terrain_type = GROUND_PLANT_TERRAIN_ROUGH
    else:
        raise RuntimeError(
            "ground-plant contract requires exactly one of scene.terrain (plane recipe) "
            "or scene.rough_ground_patch (zero-mean rough pad)"
        )
    events = getattr(env_cfg, "events", None)
    event_term = None if events is None else getattr(events, "physics_material", None)
    params = None if event_term is None else getattr(event_term, "params", None)
    if not isinstance(params, dict):
        raise RuntimeError(
            "ground-plant contract requires events.physics_material.params"
        )
    ranges = {}
    for param in ("static_friction_range", "dynamic_friction_range"):
        raw = params.get(param)
        try:
            ranges[param] = [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                f"ground-plant contract requires events.physics_material.params"
                f"['{param}']=[lo, hi]"
            ) from exc
    make_consistent = params.get("make_consistent", False)
    if not isinstance(make_consistent, bool):
        raise RuntimeError(
            "ground-plant contract requires events.physics_material.params"
            "['make_consistent'] to be a bool when present"
        )
    return ground_plant_block(
        ground_static_friction=float(material.static_friction),
        ground_dynamic_friction=float(material.dynamic_friction),
        robot_material_static_friction_range=ranges["static_friction_range"],
        robot_material_dynamic_friction_range=ranges["dynamic_friction_range"],
        robot_material_make_consistent=make_consistent,
        terrain_type=terrain_type,
        terrain_rough_height_range_m=height,
    )


def _venue_profile_module():
    """按文件路径加载 utils/venue_profile.py(Wave-1 场地档案严格加载器)。

    人话:不 import whole_body_tracking 包 —— 包 __init__ 连带注册 Isaac 任务,host-only
    单测 import 不动;照 _ground_plant_contract 的 training_contract 先例按路径加载并缓存。
    loader 本身纯标准库,host/pod/部署机都加载得动。
    """
    mod = getattr(_venue_profile_module, "_module", None)
    if mod is None:
        _vp_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "source/whole_body_tracking/whole_body_tracking/utils/venue_profile.py"
        )
        _vp_spec = importlib.util.spec_from_file_location(
            "hope_venue_profile_by_path", _vp_path
        )
        mod = importlib.util.module_from_spec(_vp_spec)
        _vp_spec.loader.exec_module(mod)
        _venue_profile_module._module = mod
    return mod


# venue profile 的 physics section -> events 旋钮映射:(profile 键, 对应 task.plant 用户键)。
# 用户键为 None 表示今天没有 CLI 通道(restitution/link mass),档案值直接落地。
_VENUE_PROFILE_PHYSICS_MATERIAL_KEYS = (
    ("static_friction_range", "robot_material_static_friction_range"),
    ("dynamic_friction_range", "robot_material_dynamic_friction_range"),
    ("restitution_range", None),
)


def _apply_venue_profile_task_override(env_cfg, task, applied):
    """task.venue_profile=<裸名或 .json 路径> —— 场地/环境档案一键展开(顶层白名单键)。

    人话:换场地、换动捕、换机器人时,"环境长什么样"的标定(动捕噪声/链路延迟丢帧/材质与
    质量随机化区间)集中放在 configs/venue_profiles/<name>.json 一份档案里,训练用
    ``task.venue_profile=<name>`` 一键导入,不再散抄进各个 yaml。加载与校验全部走 Wave-1
    的严格 loader(utils/venue_profile.py,未知档案/坏 schema 当场 raise)。

    展开顺序 = 档案先、显式键后(用户赢):

    * mocap_noise + transport -> 这里【不】直接写 cfg,而是返回 task.racket 同名键注入表,
      由下方 racket 覆写块的现有翻译层落地(_set_attr 记 applied、tts 非 live 换 actor 观测
      func 等副作用一个不少);用户显式写了同名 racket 键 -> 不注入。
    * physics -> events.physics_material / events.randomize_link_mass 的 params,这里直接写
      (无翻译层副作用);摩擦区间若被 task.plant.robot_material_* 显式配了 -> 用户赢不写
      (plant 覆写块在本函数之后跑,双保险)。

    每条 applied 标记都带 ``venue_profile=<name>@<sha256 前 8 位>``,run 记录可逐字节对账。
    返回 ``(racket 注入 dict, tag)`` 或 ``None``(键缺席 = 逐字节 no-op)。
    """
    raw = _get(task, "venue_profile")
    if raw is None:
        return None
    vp = _venue_profile_module()
    profile, meta = vp.load_venue_profile(raw)
    tag = f"venue_profile={meta['name']}@{meta['sha256'][:8]}"
    applied.append(f"{tag} loaded ({meta['path']})")

    plant = _get(task, "plant")
    events = getattr(env_cfg, "events", None)
    material = None if events is None else getattr(events, "physics_material", None)
    material_params = None if material is None else getattr(material, "params", None)
    physics = profile["physics"]
    for profile_key, plant_key in _VENUE_PROFILE_PHYSICS_MATERIAL_KEYS:
        if plant_key is not None and _get(plant, plant_key) is not None:
            applied.append(
                f"events.physics_material.params.{profile_key}: task.plant.{plant_key} "
                f"explicitly set — user override wins ({tag})"
            )
            continue
        _require(
            isinstance(material_params, dict) and profile_key in material_params,
            f"events.physics_material.params['{profile_key}'] (task.venue_profile)",
        )
        value = tuple(physics[profile_key])
        material_params[profile_key] = value
        applied.append(f"events.physics_material.params.{profile_key}={value} ({tag})")
    link_mass = None if events is None else getattr(events, "randomize_link_mass", None)
    link_mass_params = None if link_mass is None else getattr(link_mass, "params", None)
    _require(
        isinstance(link_mass_params, dict) and "mass_distribution_params" in link_mass_params,
        "events.randomize_link_mass.params['mass_distribution_params'] (task.venue_profile)",
    )
    mass_value = tuple(physics["mass_distribution_params"])
    link_mass_params["mass_distribution_params"] = mass_value
    applied.append(
        f"events.randomize_link_mass.params.mass_distribution_params={mass_value} ({tag})"
    )

    inject = {}
    inject.update(profile["mocap_noise"])
    inject.update(profile["transport"])
    return inject, tag


def _inject_venue_racket_keys(rk, inject, tag, applied):
    """把 venue profile 的 racket 同名键注入 task.racket mapping(用户显式键赢,不覆盖)。

    人话:返回一份普通 dict 合并视图给 racket 翻译层消费 —— 注入的键和用户键走同一条
    校验/记账/副作用路径,档案值不可能绕开翻译层留下半配状态。
    """
    merged = {} if rk is None else {str(key): _get(rk, key) for key in rk.keys()}
    for key, value in inject.items():
        if merged.get(key) is not None:
            applied.append(
                f"task.racket.{key} explicitly set — user override wins ({tag})"
            )
            continue
        merged[key] = value
        applied.append(f"task.racket.{key}={value!r} ({tag})")
    return merged


def _apply_task_overrides(env_cfg, task, clip_name=None):
    """Apply cfg/task/<name>.yaml overrides (incl. the composed base/ groups) onto the env cfg.

    Returns the list of applied "attr=value" strings (logged by the caller). Keys absent from the
    YAML are left at the code default; keys present whose target attribute is missing RAISE (so a
    stale/shadowed cfg or a broken Hydra composition can never silently swallow an override).
    """
    applied = []

    # 顶层 venue_profile 键(场地档案;缺席 = 逐字节 no-op)。必须在 plant/racket 覆写块
    # 之前展开:档案先写、显式键后写后赢。racket 注入表在下方 racket 块入口合并。
    _venue_racket = _apply_venue_profile_task_override(env_cfg, task, applied)

    # Plant contract control. The checked-in A3 actuator config intentionally preserves the
    # historical, uncalibrated PhysX joint-friction coefficients for legacy checkpoint lineage.
    # A fresh schema-v3 run can opt into the only currently cross-engine-exact setting: every
    # actuator friction coefficient is zero. This must happen before ``gym.make`` so the saved
    # env config and runtime training contract both record the actual plant. False/absent is a
    # byte-for-byte no-op; old checkpoints are never rewritten or silently laundered.
    plant = _get(task, "plant")
    if plant is not None:
        try:
            plant.keys()
        except Exception as exc:
            raise _OverrideError(
                f"[train.py] task.plant must be a mapping, got {plant!r}"
            ) from exc
    _check_unknown_keys(plant, _PLANT_KEYS, "task.plant")
    zero_joint_friction = _get(plant, "zero_joint_friction")
    if zero_joint_friction is not None and _as_explicit_bool(
        zero_joint_friction, "task.plant.zero_joint_friction"
    ):
        scene = getattr(env_cfg, "scene", None)
        robot = None if scene is None else getattr(scene, "robot", None)
        actuators = None if robot is None else getattr(robot, "actuators", None)
        _require(
            isinstance(actuators, dict) and bool(actuators),
            "scene.robot.actuators (task.plant.zero_joint_friction=true)",
        )
        missing = [name for name, actuator in actuators.items() if not hasattr(actuator, "friction")]
        _require(
            not missing,
            "every scene.robot actuator exposes friction; missing=" + ",".join(missing),
        )
        for actuator in actuators.values():
            actuator.friction = 0.0
        applied.append(
            "scene.robot.actuators[*].friction=0.0 "
            "(explicit schema-v3 zero-friction plant control)"
        )

    # Recovery/hold-only random WORLD-Y torso-force training.  This deliberately is not an
    # EventManager term: the reviewed adapter must own every substep write, prove a complete zero
    # overwrite when inactive, and fail closed on any competing writer.
    _apply_lateral_perturbation_task_override(env_cfg, task, applied)

    # Wave-P random base push (task.push.*; default OFF/absent = events.push_robot stays None,
    # the HITTER-aligned recipe every running matrix cell trains with). 人话:每隔几秒随机推
    # 机器人一把练抗扰平衡;论文依据 PACE(±0.2 m/s 每 5–15 s)与 BeyondMimic(±0.5 m/s +
    # rpy 角速度每 1–3 s)。
    _apply_push_robot_task_override(env_cfg, task, applied)

    # F-axis interval FORCE push (task.force_push.*; default OFF/absent = events.force_push +
    # events.force_push_sweep both stay None). 人话:每隔几秒朝水平随机方向对 pelvis_link 施加
    # 持续 duration_s 的恒力,与速度推同冲量可比(Δv_equiv = F·Δt/m 运行时记进合同)。
    _apply_force_push_task_override(env_cfg, task, applied)

    # env base (num_envs is applied earlier via parse_env_cfg). Read every value through _get so the
    # logic works on both OmegaConf nodes (runtime) and plain dicts (unit tests).
    env = _get(task, "env")
    if env is not None:
        es = _get(env, "env_spacing")
        if es is not None:
            env_cfg.scene.env_spacing = float(es)
            applied.append(f"scene.env_spacing={float(es)}")
        els = _get(env, "episode_length_s")
        if els is not None:
            env_cfg.episode_length_s = float(els)
            applied.append(f"episode_length_s={float(els)}")

    # Ground/terrain plant keys (2026-07-22, chatter-ground-foot wave; 2026-07-29 抬脚地形修复).
    # 人话:地面材质摩擦、机器人 body 材质随机化范围、随机凹凸地垫——默认全缺席 = 现状平地
    # 配方逐字节不变;任何显式值都会让 schema-3 合同长出 ground_plant 块,平地谱系 checkpoint
    # 对着新 plant 续训会被 _contract_diff 拒绝(rough 臂必须 fresh-from-random)。故意排在
    # env-base 块之后:凹凸垫的兜底地板要按 POST-OVERRIDE 的 env_spacing 算尺寸。
    _apply_ground_plant_task_override(env_cfg, plant, applied)

    # sim base (control frequency = 1 / (dt * decimation))
    sim = _get(task, "sim")
    if sim is not None:
        dt = _get(sim, "dt")
        if dt is not None:
            env_cfg.sim.dt = float(dt)
            applied.append(f"sim.dt={float(dt)}")
        dec = _get(sim, "decimation")
        if dec is not None:
            env_cfg.decimation = int(dec)
            env_cfg.sim.render_interval = env_cfg.decimation  # keep render in step with decimation
            applied.append(f"decimation={int(dec)}")

    # One top-level block configures both command terms.  There is intentionally no supported
    # per-term YAML seam: a clock-only or target-only activation would train a protocol that cannot
    # exist in the runner.  Enabled blocks require every field, including an exact complete profile;
    # disabled/absent keeps the historical source and hard-contract bytes unchanged.
    planner_revision = _get(task, "planner_revision")
    _check_unknown_keys(
        planner_revision, _PLANNER_REVISION_KEYS, "task.planner_revision"
    )
    if planner_revision is not None:
        enabled_raw = _get(planner_revision, "enabled")
        if enabled_raw is None:
            raise _OverrideError(
                "task.planner_revision must explicitly set enabled=true|false"
            )
        enabled = _as_explicit_bool(
            enabled_raw, "task.planner_revision.enabled"
        )
        present = {
            key for key in _PLANNER_REVISION_KEYS if _get(planner_revision, key) is not None
        }
        if not enabled:
            extras = sorted(present - {"enabled"})
            if extras:
                raise _OverrideError(
                    "disabled task.planner_revision may not carry dormant fields: "
                    f"{extras}"
                )
        else:
            missing = sorted(set(_PLANNER_REVISION_KEYS) - present)
            if missing:
                raise _OverrideError(
                    "enabled task.planner_revision is incomplete; missing "
                    f"{missing}"
                )
            _require(
                hasattr(env_cfg.commands, "motion")
                and hasattr(env_cfg.commands, "racket_target"),
                "commands.motion + commands.racket_target (task.planner_revision)",
            )
            from whole_body_tracking.tasks.tracking.mdp.planner_revision import (
                InitialTtsMixture,
                PhaseGovernorProfile,
            )

            raw_profile = _get(planner_revision, "profile")
            try:
                profile = PhaseGovernorProfile.from_mapping(dict(raw_profile))
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    f"task.planner_revision.profile is invalid: {exc}"
                ) from exc
            initial_tts = tuple(
                float(value)
                for value in _get(planner_revision, "initial_tts_range_s")
            )
            if (
                len(initial_tts) != 2
                or not profile.min_tts_s <= initial_tts[0] < initial_tts[1] <= profile.max_tts_s
            ):
                raise _OverrideError(
                    "task.planner_revision.initial_tts_range_s must be a non-degenerate ordered "
                    "pair inside "
                    "the profile TTS envelope"
                )
            raw_initial_tts_mixture = _get(
                planner_revision, "initial_tts_mixture"
            )
            try:
                initial_tts_mixture = InitialTtsMixture.from_mapping(
                    dict(raw_initial_tts_mixture)
                )
                initial_tts_mixture.validate_support(
                    lo_s=initial_tts[0], hi_s=initial_tts[1]
                )
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    f"task.planner_revision.initial_tts_mixture is invalid: {exc}"
                ) from exc
            initial_tts_mixture_doc = initial_tts_mixture.document()
            profile_doc = profile.document()
            motion_cfg = env_cfg.commands.motion
            racket_cfg = env_cfg.commands.racket_target
            for command_cfg, label in (
                (motion_cfg, "commands.motion"),
                (racket_cfg, "commands.racket_target"),
            ):
                _require(
                    hasattr(command_cfg, "planner_revision_enabled")
                    and hasattr(command_cfg, "planner_revision_profile")
                    and hasattr(command_cfg, "planner_revision_initial_tts_range_s"),
                    f"{label}.planner_revision_*",
                )
                command_cfg.planner_revision_enabled = True
                command_cfg.planner_revision_profile = profile_doc
                command_cfg.planner_revision_initial_tts_range_s = initial_tts
                _require(
                    hasattr(command_cfg, "planner_revision_initial_tts_mixture"),
                    f"{label}.planner_revision_initial_tts_mixture",
                )
                command_cfg.planner_revision_initial_tts_mixture = (
                    initial_tts_mixture_doc
                )
            # initial_tts is the sole preparation/deadline clock.  Leaving the legacy hold clocks
            # active would consume the same deadline twice and create impossible late releases.
            motion_cfg.hold_steps_range = (0, 0)
            motion_cfg.stand_start_min_hold = 0
            motion_cfg.post_swing_min_hold = 0
            for source_key, target_attr in (
                ("position_std_m", "planner_revision_position_std_m"),
                ("velocity_std_mps", "planner_revision_velocity_std_mps"),
                ("normal_std_rad", "planner_revision_normal_std_rad"),
                ("tts_std_s", "planner_revision_tts_std_s"),
            ):
                _require(hasattr(racket_cfg, target_attr), f"commands.racket_target.{target_attr}")
                value = _as_exact_float(
                    _get(planner_revision, source_key),
                    f"task.planner_revision.{source_key}",
                )
                if value < 0.0:
                    raise _OverrideError(
                        f"task.planner_revision.{source_key} must be non-negative"
                    )
                setattr(racket_cfg, target_attr, value)
            from whole_body_tracking.tasks.tracking import mdp as _mdp

            _require(
                getattr(env_cfg.observations.policy, "time_to_strike", None) is not None,
                "observations.policy.time_to_strike (task.planner_revision)",
            )
            env_cfg.observations.policy.time_to_strike.func = _mdp.actor_time_to_strike
            applied.append(
                "planner_task_revision=enabled(same physical ball; atomic target/TTS; "
                f"phase_governor_sha={profile.profile_sha256},initial_tts={initial_tts},"
                f"tts_mixture={initial_tts_mixture_doc})"
            )

    # motion command (swing-entry structure): no-teleport wrap / stand-entry resets / pre-swing hold
    mt = _get(task, "motion")
    _check_unknown_keys(mt, _MOTION_KEYS, "task.motion")
    if mt is not None:
        provided = [k for k in _MOTION_KEYS if _get(mt, k) is not None]
        if provided:
            _require(hasattr(env_cfg.commands, "motion"),
                     f"commands.motion (task YAML sets motion keys {provided})")
            M = env_cfg.commands.motion
            _set_attr(M, "wrap_teleport", _get(mt, "wrap_teleport"), _as_bool, applied, "commands.motion")
            _set_attr(M, "stand_start_prob", _get(mt, "stand_start_prob"), float, applied, "commands.motion")
            _set_attr(M, "hold_steps_range", _get(mt, "hold_steps_range"),
                      lambda v: tuple(int(x) for x in v), applied, "commands.motion")
            _set_attr(M, "stand_start_min_hold", _get(mt, "stand_start_min_hold"), int, applied, "commands.motion")
            _set_attr(M, "stand_start_yaw_range", _get(mt, "stand_start_yaw_range"),
                      _as_yaw_range, applied, "commands.motion")
            _set_attr(M, "post_swing_start_prob", _get(mt, "post_swing_start_prob"), float, applied, "commands.motion")
            _set_attr(M, "post_swing_buffer_size", _get(mt, "post_swing_buffer_size"), int, applied, "commands.motion")
            _set_attr(M, "post_swing_min_fill", _get(mt, "post_swing_min_fill"), int, applied, "commands.motion")
            _set_attr(M, "post_swing_min_hold", _get(mt, "post_swing_min_hold"), int, applied, "commands.motion")
            _set_attr(M, "post_swing_teacher_receipt", _get(mt, "post_swing_teacher_receipt"), str, applied, "commands.motion")
            _set_attr(M, "post_swing_teacher_receipt_sha256", _get(mt, "post_swing_teacher_receipt_sha256"), str, applied, "commands.motion")
            _set_attr(M, "post_swing_teacher_retry_authorization", _get(mt, "post_swing_teacher_retry_authorization"), str, applied, "commands.motion")
            _set_attr(M, "post_swing_teacher_retry_authorization_sha256", _get(mt, "post_swing_teacher_retry_authorization_sha256"), str, applied, "commands.motion")
            _set_attr(M, "post_swing_teacher_root_linear_velocity_limit_mps", _get(mt, "post_swing_teacher_root_linear_velocity_limit_mps"), lambda value: _as_exact_float(value, "task.motion.post_swing_teacher_root_linear_velocity_limit_mps"), applied, "commands.motion")
            _set_attr(M, "post_swing_teacher_root_angular_velocity_limit_radps", _get(mt, "post_swing_teacher_root_angular_velocity_limit_radps"), lambda value: _as_exact_float(value, "task.motion.post_swing_teacher_root_angular_velocity_limit_radps"), applied, "commands.motion")
            _set_attr(M, "post_swing_require_ready_at_init", _get(mt, "post_swing_require_ready_at_init"), lambda value: _as_explicit_bool(value, "task.motion.post_swing_require_ready_at_init"), applied, "commands.motion")
            _set_attr(M, "post_swing_fail_fast_first_reset", _get(mt, "post_swing_fail_fast_first_reset"), lambda value: _as_explicit_bool(value, "task.motion.post_swing_fail_fast_first_reset"), applied, "commands.motion")
            _set_attr(M, "post_swing_first_reset_min_adopted_count", _get(mt, "post_swing_first_reset_min_adopted_count"), lambda value: _as_exact_int(value, "task.motion.post_swing_first_reset_min_adopted_count"), applied, "commands.motion")
            _set_attr(M, "post_swing_first_reset_min_adopted_fraction", _get(mt, "post_swing_first_reset_min_adopted_fraction"), lambda value: _as_exact_float(value, "task.motion.post_swing_first_reset_min_adopted_fraction"), applied, "commands.motion")
            _set_attr(M, "post_swing_first_reset_selection_tolerance", _get(mt, "post_swing_first_reset_selection_tolerance"), lambda value: _as_exact_float(value, "task.motion.post_swing_first_reset_selection_tolerance"), applied, "commands.motion")
            _set_attr(M, "post_swing_first_reset_require_readback", _get(mt, "post_swing_first_reset_require_readback"), lambda value: _as_explicit_bool(value, "task.motion.post_swing_first_reset_require_readback"), applied, "commands.motion")
            _set_attr(M, "post_swing_capture_output_dir", _get(mt, "post_swing_capture_output_dir"), str, applied, "commands.motion")
            _set_attr(M, "post_swing_capture_target_count", _get(mt, "post_swing_capture_target_count"), lambda value: _as_exact_int(value, "task.motion.post_swing_capture_target_count"), applied, "commands.motion")
            _set_attr(M, "clip_switch_prob", _get(mt, "clip_switch_prob"), float, applied, "commands.motion")
            _set_attr(M, "event_timing_mode", _get(mt, "event_timing_mode"), str, applied, "commands.motion")
            _set_attr(M, "event_timing_schedule", _get(mt, "event_timing_schedule"), str, applied, "commands.motion")
            _set_attr(M, "event_timing_schedule_sha256", _get(mt, "event_timing_schedule_sha256"), str, applied, "commands.motion")
            _set_attr(M, "event_timing_repeat", _get(mt, "event_timing_repeat"), _as_bool, applied, "commands.motion")
            _set_attr(M, "speed_scale_range", _get(mt, "speed_scale_range"),
                      lambda v: tuple(float(x) for x in v), applied, "commands.motion")
            # Backhand-fix ablation (2026-07-08): fixed per-clip reference playback speed.
            _set_attr(M, "speed_scale_per_clip", _get(mt, "speed_scale_per_clip"),
                      lambda v: tuple(float(x) for x in v), applied, "commands.motion")
            # spdmix v2 硬绑定一 (2026-07-22): 每 clip 的 forehand/backhand 家族表。人话:6-clip
            # 变速列表里,正手 1.0/1.2 变体不再被"clips==0 才是正手"误判成反手。缺席 = 旧 2-clip
            # 推导,逐字节不变;值/长度/两族齐全在 MotionCommand 开机时整表校验(fail-loud)。
            _set_attr(M, "clip_family_per_clip", _get(mt, "clip_family_per_clip"),
                      lambda v: tuple(str(x) for x in v), applied, "commands.motion")
            _set_attr(
                M,
                "balanced_clip_sampling",
                _get(mt, "balanced_clip_sampling"),
                lambda value: _as_explicit_bool(
                    value, "task.motion.balanced_clip_sampling"
                ),
                applied,
                "commands.motion",
            )
            _set_attr(
                M,
                "balanced_clip_sampling_seed",
                _get(mt, "balanced_clip_sampling_seed"),
                lambda value: _as_exact_int(
                    value, "task.motion.balanced_clip_sampling_seed"
                ),
                applied,
                "commands.motion",
            )
            _set_attr(
                M,
                "canonical_ready_mode",
                _get(mt, "canonical_ready_mode"),
                lambda value: _as_explicit_bool(
                    value, "task.motion.canonical_ready_mode"
                ),
                applied,
                "commands.motion",
            )
            for _canonical_string_field in (
                "canonical_registry_path",
                "canonical_registry_repo_root",
                "canonical_registry_sha256",
                "canonical_registry_alignment_sha256",
                "canonical_ready_sha256",
                "canonical_ready_fk_sha256",
                "canonical_promotion_certificate_path",
            ):
                _set_attr(
                    M,
                    _canonical_string_field,
                    _get(mt, _canonical_string_field),
                    str,
                    applied,
                    "commands.motion",
                )
            _set_attr(
                M,
                "joint_position_range",
                _get(mt, "joint_position_range"),
                lambda value: tuple(float(component) for component in value),
                applied,
                "commands.motion",
            )
            for _canonical_mapping_field in ("pose_range", "velocity_range"):
                _set_attr(
                    M,
                    _canonical_mapping_field,
                    _get(mt, _canonical_mapping_field),
                    lambda value: {
                        str(key): tuple(
                            float(component) for component in pair
                        )
                        for key, pair in value.items()
                    },
                    applied,
                    "commands.motion",
                )
            # R-c(i): every swing entry (RSI reset AND wrap) starts the reference N frames past the
            # clip start — the v5 clips carry a 3-4 frame IK cold-start transient at frame 0 (GMR
            # warm-up bug); N=6 is the design's stopgap until the source fix lands. Default 0 = off.
            # 人话:出生别传送到 IK 瞬态帧上,参考从第 N 帧起播。
            _set_attr(M, "rsi_skip_settle_frames", _get(mt, "rsi_skip_settle_frames"), int, applied, "commands.motion")
            # R-c(ii): held-RSI births (hold_counter>0: stand joints, frozen reference) get the
            # DEFAULT-STAND root height instead of the reference frame-0 crouch z — stand joints at
            # crouch height put the feet ~0.29 m under the floor and PhysX kicks the robot out.
            # 人话:站姿关节配站姿身高,脚不再穿地被物理引擎弹飞。
            _set_attr(M, "rsi_hold_root_stand_z", _get(mt, "rsi_hold_root_stand_z"), _as_bool, applied, "commands.motion")
            # 防同步 stagger (metric-sync fix 2026-07-09; default OFF = byte-identical): one-shot
            # random offsets on each env's first hold clock + the episode clock, so a same-instant
            # 4096-env cohort stops swinging/timing-out in one synchronized wave (EMA 指标同步振荡病).
            # 人话:开了它,所有 env 的节拍被随机错开,摔率/完成率/上台率曲线不再集体振荡。
            _set_attr(M, "stagger_initial_clock", _get(mt, "stagger_initial_clock"), _as_bool, applied, "commands.motion")
            _set_attr(M, "stagger_hold_max_steps", _get(mt, "stagger_hold_max_steps"), int, applied, "commands.motion")
            _set_attr(
                M,
                "allow_legacy_link_origin_velocity",
                _get(mt, "allow_legacy_link_origin_velocity"),
                lambda value: _as_explicit_bool(
                    value, "task.motion.allow_legacy_link_origin_velocity"
                ),
                applied,
                "commands.motion",
            )
            if bool(getattr(M, "allow_legacy_link_origin_velocity", False)):
                applied.append(
                    "[diagnostic] legacy link-origin motion velocity explicitly allowed; "
                    "motion_kinematics_exact=false and every descendant remains inexact"
                )

    rw = _get(task, "rewards")
    _check_unknown_keys(rw, _REWARD_KEYS, "task.rewards")
    # v2 奖励包:先展开(注入用户没写的键 + direct-cfg 改动),再走下面的正常覆写层,
    # 显式同名键后写后赢。2026-07-25 默认翻转:缺席 = 按 v2 展开;显式 reward_pack=v1 =
    # legacy 兜底,原 mapping 原样返回,逐字节不变。
    rw = _expand_reward_pack(env_cfg, task, rw, applied)
    if rw is not None:
        R = env_cfg.rewards
        # YAML 显式 null 先删参(jiayi 8ee2e82a 的修复;人话注释见 _REWARD_NULL_REMOVABLE_PARAMS)。
        # 必须在下面各 set/probe 逻辑之前跑:probe.params.update(term.params) 复制到的才是删完的状态。
        _apply_reward_param_null_removals(R, rw, applied)
        _set_reward(R, "racket_position", _get(rw, "racket_position_weight"), _get(rw, "racket_position_std"), applied)
        # Ablation B: swap the racket-position term to the no-swing-through (static strike-point) variant.
        if _as_bool(_get(rw, "racket_position_static", False)) and _get(rw, "racket_position_static") is not None:
            from whole_body_tracking.tasks.tracking import mdp as _mdp
            _require(hasattr(R, "racket_position"), "rewards.racket_position")
            R.racket_position.func = _mdp.racket_position_tracking_static_exp
            applied.append("rewards.racket_position.func=racket_position_tracking_static_exp")
        _set_reward(R, "racket_velocity", _get(rw, "racket_velocity_weight"), _get(rw, "racket_velocity_std"), applied)
        _set_reward(R, "racket_normal", _get(rw, "racket_normal_weight"), _get(rw, "racket_normal_std"), applied)
        _set_reward(R, "base_position", _get(rw, "base_position_weight"), _get(rw, "base_position_std"), applied)
        # Between-swing recovery: positive ready-stance reward during the pre-swing hold (deploy-parity).
        _set_reward(R, "hold_ready", _get(rw, "hold_ready_weight"), _get(rw, "hold_ready_std"), applied)
        _hr_reach = _get(rw, "hold_ready_reach")
        if _hr_reach is not None:
            _require(hasattr(R, "hold_ready"), "rewards.hold_ready")
            R.hold_ready.params["reach"] = float(_hr_reach)
            applied.append(f"rewards.hold_ready.params.reach={float(_hr_reach)}")
        # FOOTWORK V2 (2026-07-05): gate mode for the hold_ready reach gate — "racket" (legacy
        # blade->target distance; arm-gameable, not station-selective) or "station" (planar
        # base->commanded-station error; required for the HITTER footwork task). See hold_ready().
        _hr_mode = _get(rw, "hold_ready_reach_mode")
        if _hr_mode is not None:
            _require(hasattr(R, "hold_ready"), "rewards.hold_ready")
            _hr_mode = str(_hr_mode)
            if _hr_mode not in ("racket", "station"):
                raise ValueError(
                    f"[train.py] rewards.hold_ready_reach_mode must be 'racket' or 'station', got '{_hr_mode}'"
                )
            R.hold_ready.params["reach_mode"] = _hr_mode
            applied.append(f"rewards.hold_ready.params.reach_mode={_hr_mode}")
        # CONTINUOUS RALLY (2026-07-07): positive braking kernel through the follow-through
        # ((~pre_strike) & (~strike_window)) — arrests the walk-and-strike lunge momentum between
        # swings (deploy Gate-2.5 P7 drift fall). Default weight 0.0 = OFF (plain HitterPure).
        _set_reward(R, "post_strike_brake", _get(rw, "post_strike_brake_weight"), _get(rw, "post_strike_brake_std"), applied)
        # Recovery from yawed holds. This is inert unless the task enables both a yawed
        # stand-start distribution and a positive hold_heading weight.
        _set_reward(R, "hold_heading", _get(rw, "hold_heading_weight"), _get(rw, "hold_heading_std"), applied)
        # P2.4 PACE-style smooth deceleration (flag-gated, default weight 0.0 = OFF): pseudo base-speed
        # command proportional to the remaining planar racket->target error. REWARD-side only (the
        # frozen 175-D actor obs contract is untouched).
        _set_reward(R, "base_decel", _get(rw, "base_decel_weight"), _get(rw, "base_decel_std"), applied)
        for _pk, _yk in (("v_gain", "base_decel_v_gain"), ("v_max", "base_decel_v_max")):
            _bd = _get(rw, _yk)
            if _bd is not None:
                _require(hasattr(R, "base_decel"), "rewards.base_decel")
                R.base_decel.params[_pk] = float(_bd)
                applied.append(f"rewards.base_decel.params.{_pk}={float(_bd)}")
        _base_decel_activation_requested = any(
            _get(rw, key) is not None
            for key in (
                "base_decel_weight",
                "base_decel_std",
                "base_decel_v_gain",
                "base_decel_v_max",
            )
        )
        if _base_decel_activation_requested:
            _require(
                hasattr(R, "base_decel") and R.base_decel is not None,
                "rewards.base_decel (base_decel activation observer)",
            )
            _require(
                hasattr(R, "base_decel_activation_probe")
                and R.base_decel_activation_probe is not None,
                "rewards.base_decel_activation_probe",
            )
            _base_decel_params = R.base_decel.params
            for _required_param in ("v_gain", "v_max", "std"):
                _require(
                    _required_param in _base_decel_params,
                    f"rewards.base_decel.params.{_required_param}",
                )
            _base_decel_probe = R.base_decel_activation_probe
            _base_decel_probe.weight = 1.0
            _base_decel_probe.params.update(
                {
                    key: float(_base_decel_params[key])
                    for key in ("v_gain", "v_max", "std")
                }
            )
            applied.append(
                "rewards.base_decel_activation_probe="
                f"({float(_base_decel_params['v_gain'])},"
                f"{float(_base_decel_params['v_max'])},"
                f"{float(_base_decel_params['std'])},weight=1.0)"
            )
        # D6 normalized qdot-limit hinge (default OFF): this is a realized joint-speed penalty
        # against the actual articulation limits, not an action-rate proxy.  Its sign and margin
        # are causal arm identity, so both are validated here and bound into the hard contract.
        _qdot_weight = _get(rw, "joint_velocity_limit_hinge_weight")
        _qdot_margin = _get(rw, "joint_velocity_limit_hinge_margin")
        if _qdot_weight is not None or _qdot_margin is not None:
            _require(
                hasattr(R, "joint_velocity_limit_hinge"),
                "rewards.joint_velocity_limit_hinge",
            )
            _require(
                hasattr(R, "joint_velocity_limit_hinge_probe")
                and R.joint_velocity_limit_hinge_probe is not None,
                "rewards.joint_velocity_limit_hinge_probe",
            )
            _qdot_term = R.joint_velocity_limit_hinge
            if _qdot_weight is not None:
                if isinstance(_qdot_weight, bool):
                    raise _OverrideError(
                        "rewards.joint_velocity_limit_hinge.weight must be finite and <= 0"
                    )
                try:
                    _qdot_weight_value = float(_qdot_weight)
                except (TypeError, ValueError) as exc:
                    raise _OverrideError(
                        "rewards.joint_velocity_limit_hinge.weight must be finite and <= 0"
                    ) from exc
                if not math.isfinite(_qdot_weight_value) or _qdot_weight_value > 0.0:
                    raise _OverrideError(
                        "rewards.joint_velocity_limit_hinge.weight must be finite and <= 0"
                    )
                _qdot_term.weight = _qdot_weight_value
                applied.append(
                    f"rewards.joint_velocity_limit_hinge.weight={_qdot_weight_value}"
                )
            if _qdot_margin is not None:
                if isinstance(_qdot_margin, bool):
                    raise _OverrideError(
                        "rewards.joint_velocity_limit_hinge.margin must be finite and in (0, 1)"
                    )
                try:
                    _qdot_margin_value = float(_qdot_margin)
                except (TypeError, ValueError) as exc:
                    raise _OverrideError(
                        "rewards.joint_velocity_limit_hinge.margin must be finite and in (0, 1)"
                    ) from exc
                if not math.isfinite(_qdot_margin_value) or not 0.0 < _qdot_margin_value < 1.0:
                    raise _OverrideError(
                        "rewards.joint_velocity_limit_hinge.margin must be finite and in (0, 1)"
                    )
                _require(
                    "margin" in _qdot_term.params,
                    "rewards.joint_velocity_limit_hinge.params['margin']",
                )
                _qdot_term.params["margin"] = _qdot_margin_value
                applied.append(
                    "rewards.joint_velocity_limit_hinge.params.margin="
                    f"{_qdot_margin_value}"
                )
            _qdot_probe = R.joint_velocity_limit_hinge_probe
            _qdot_probe.weight = 1.0
            _qdot_probe.params.update(
                {
                    "asset_cfg": _qdot_term.params["asset_cfg"],
                    "margin": float(_qdot_term.params["margin"]),
                    "expected_joint_count": int(
                        _qdot_term.params["expected_joint_count"]
                    ),
                }
            )
            applied.append(
                "rewards.joint_velocity_limit_hinge_probe="
                f"(margin={float(_qdot_term.params['margin'])},weight=1.0)"
            )
        # Processed-q_des recovery slew (default OFF).  This is not a duplicate action_rate_l2:
        # the term reads the deploy-space target after offset/scale/clamp, restricts itself to the
        # 15 waist/leg joints, and opens only for the same attempt's post-contact recovery clock.
        _slew_weight = _get(rw, "processed_qdes_slew_hinge_weight")
        _slew_margin = _get(rw, "processed_qdes_slew_hinge_margin")
        _slew_start = _get(rw, "processed_qdes_slew_hinge_recovery_start_s")
        _slew_end = _get(rw, "processed_qdes_slew_hinge_recovery_end_s")
        _slew_requested = any(
            value is not None
            for value in (_slew_weight, _slew_margin, _slew_start, _slew_end)
        )
        if _slew_requested:
            _require(
                hasattr(R, "processed_qdes_slew_hinge")
                and R.processed_qdes_slew_hinge is not None,
                "rewards.processed_qdes_slew_hinge",
            )
            _require(
                hasattr(R, "processed_qdes_slew_hinge_probe")
                and R.processed_qdes_slew_hinge_probe is not None,
                "rewards.processed_qdes_slew_hinge_probe",
            )
            _slew_term = R.processed_qdes_slew_hinge
            _slew_probe = R.processed_qdes_slew_hinge_probe
            if _slew_weight is not None:
                if isinstance(_slew_weight, bool):
                    raise _OverrideError(
                        "rewards.processed_qdes_slew_hinge.weight must be finite and <= 0"
                    )
                try:
                    _slew_weight_value = float(_slew_weight)
                except (TypeError, ValueError) as exc:
                    raise _OverrideError(
                        "rewards.processed_qdes_slew_hinge.weight must be finite and <= 0"
                    ) from exc
                if not math.isfinite(_slew_weight_value) or _slew_weight_value > 0.0:
                    raise _OverrideError(
                        "rewards.processed_qdes_slew_hinge.weight must be finite and <= 0"
                    )
                _slew_term.weight = _slew_weight_value
                applied.append(
                    f"rewards.processed_qdes_slew_hinge.weight={_slew_weight_value}"
                )

            _slew_param_overrides = (
                ("margin", _slew_margin),
                ("recovery_start_s", _slew_start),
                ("recovery_end_s", _slew_end),
            )
            for _param_name, _raw_value in _slew_param_overrides:
                if _raw_value is None:
                    continue
                if isinstance(_raw_value, bool):
                    raise _OverrideError(
                        "rewards.processed_qdes_slew_hinge margin/window must be finite with "
                        "0 < margin < 1 and 0 <= start < end"
                    )
                try:
                    _param_value = float(_raw_value)
                except (TypeError, ValueError) as exc:
                    raise _OverrideError(
                        "rewards.processed_qdes_slew_hinge margin/window must be finite with "
                        "0 < margin < 1 and 0 <= start < end"
                    ) from exc
                if not math.isfinite(_param_value):
                    raise _OverrideError(
                        "rewards.processed_qdes_slew_hinge margin/window must be finite with "
                        "0 < margin < 1 and 0 <= start < end"
                    )
                _require(
                    _param_name in _slew_term.params,
                    f"rewards.processed_qdes_slew_hinge.params['{_param_name}']",
                )
                _slew_term.params[_param_name] = _param_value
                applied.append(
                    f"rewards.processed_qdes_slew_hinge.params.{_param_name}={_param_value}"
                )
            _resolved_margin = float(_slew_term.params["margin"])
            _resolved_start = float(_slew_term.params["recovery_start_s"])
            _resolved_end = float(_slew_term.params["recovery_end_s"])
            if (
                not 0.0 < _resolved_margin < 1.0
                or _resolved_start < 0.0
                or _resolved_start >= _resolved_end
            ):
                raise _OverrideError(
                    "rewards.processed_qdes_slew_hinge margin/window must be finite with "
                    "0 < margin < 1 and 0 <= start < end"
                )
            if _slew_term.params.get("action_name") != "joint_pos":
                raise _OverrideError(
                    "rewards.processed_qdes_slew_hinge.action_name must be exactly 'joint_pos'"
                )
            if _slew_term.params.get("command_name") != "racket_target":
                raise _OverrideError(
                    "rewards.processed_qdes_slew_hinge.command_name must be exactly 'racket_target'"
                )
            _slew_probe.weight = 1.0
            _slew_probe.params.clear()
            _slew_probe.params.update(_slew_term.params)
            applied.append(
                "rewards.processed_qdes_slew_hinge_probe="
                f"(margin={_resolved_margin},recovery={_resolved_start}..{_resolved_end},weight=1.0)"
            )

        # Wave-Q qbar: all-joint q_des position-limit barrier (default OFF).  Jiayi V14's
        # 全关节 top-k qdes barrier idea with the top-k removed (Franco 2026-07-21): all 31
        # deploy-space targets pay inside the margin band next to their position limits, on
        # every control step (dense, no phase gate).  人话:目标角贴近限位就罚,全身关节全程
        # 盯着;任何 qbar 键都必须明说 weight,配了就必带零权重探针记账。
        _qbar_weight_raw = _get(rw, "qdes_limit_barrier_weight")
        _qbar_margin_raw = _get(rw, "qdes_limit_barrier_margin_frac")
        _qbar_requested = (
            _qbar_weight_raw is not None or _qbar_margin_raw is not None
        )
        if _qbar_requested:
            if _qbar_weight_raw is None:
                raise _OverrideError(
                    "qdes_limit_barrier_margin_frac requires "
                    "qdes_limit_barrier_weight explicitly"
                )
            _require(
                hasattr(R, "qdes_limit_barrier")
                and R.qdes_limit_barrier is not None,
                "rewards.qdes_limit_barrier",
            )
            _require(
                hasattr(R, "qdes_limit_barrier_probe")
                and R.qdes_limit_barrier_probe is not None,
                "rewards.qdes_limit_barrier_probe",
            )
            _qbar_term = R.qdes_limit_barrier
            _qbar_probe = R.qdes_limit_barrier_probe
            _require(
                isinstance(_qbar_term.params, dict)
                and isinstance(_qbar_probe.params, dict),
                "rewards.qdes_limit_barrier.params",
            )
            if isinstance(_qbar_weight_raw, bool):
                raise _OverrideError(
                    "rewards.qdes_limit_barrier.weight must be finite and <= 0"
                )
            try:
                _qbar_weight = float(_qbar_weight_raw)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    "rewards.qdes_limit_barrier.weight must be finite and <= 0"
                ) from exc
            if not math.isfinite(_qbar_weight) or _qbar_weight > 0.0:
                raise _OverrideError(
                    "rewards.qdes_limit_barrier.weight must be finite and <= 0"
                )
            _qbar_margin = None
            if _qbar_margin_raw is not None:
                if isinstance(_qbar_margin_raw, bool):
                    raise _OverrideError(
                        "rewards.qdes_limit_barrier.margin_frac must be finite "
                        "and in (0, 0.5)"
                    )
                try:
                    _qbar_margin = float(_qbar_margin_raw)
                except (TypeError, ValueError) as exc:
                    raise _OverrideError(
                        "rewards.qdes_limit_barrier.margin_frac must be finite "
                        "and in (0, 0.5)"
                    ) from exc
                if not math.isfinite(_qbar_margin) or not 0.0 < _qbar_margin < 0.5:
                    raise _OverrideError(
                        "rewards.qdes_limit_barrier.margin_frac must be finite "
                        "and in (0, 0.5)"
                    )
                _require(
                    "margin_frac" in _qbar_term.params,
                    "rewards.qdes_limit_barrier.params['margin_frac']",
                )
            if _qbar_term.params.get("action_name") != "joint_pos":
                raise _OverrideError(
                    "rewards.qdes_limit_barrier.action_name must be exactly 'joint_pos'"
                )
            _qbar_term.weight = _qbar_weight
            applied.append(f"rewards.qdes_limit_barrier.weight={_qbar_weight}")
            if _qbar_margin is not None:
                _qbar_term.params["margin_frac"] = _qbar_margin
                applied.append(
                    f"rewards.qdes_limit_barrier.params.margin_frac={_qbar_margin}"
                )
            _qbar_probe.weight = 1.0
            _qbar_probe.params.clear()
            _qbar_probe.params.update(_qbar_term.params)
            applied.append(
                "rewards.qdes_limit_barrier_probe="
                f"(margin_frac={float(_qbar_term.params['margin_frac'])},weight=1.0)"
            )

        # mjlab-ported foot-contact shaping (default OFF).  Same fail-loud envelope as the qbar
        # precedent: a parameter key without its weight is refused (weight-less 参数会静默不生效,
        # 必须明说), the weight must be a finite penalty (<= 0; explicit 0 = measured control),
        # and nothing mutates until every value validates.  人话:foot_soft_landing 罚落地砸太
        # 重(法向冲击超阈,有界),foot_clearance 罚腾空脚又低又快地扫(给"允许跨步"臂)。
        def _foot_term_overrides(term_name, weight_key, param_name, param_key,
                                 param_label, param_positive_msg):
            _weight_raw = _get(rw, weight_key)
            _param_raw = _get(rw, param_key)
            if _weight_raw is None and _param_raw is None:
                return
            if _weight_raw is None:
                raise _OverrideError(f"{param_key} requires {weight_key} explicitly")
            _require(
                hasattr(R, term_name) and getattr(R, term_name) is not None,
                f"rewards.{term_name}",
            )
            _term = getattr(R, term_name)
            _require(isinstance(_term.params, dict), f"rewards.{term_name}.params")
            if isinstance(_weight_raw, bool):
                raise _OverrideError(
                    f"rewards.{term_name}.weight must be finite and <= 0"
                )
            try:
                _weight = float(_weight_raw)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    f"rewards.{term_name}.weight must be finite and <= 0"
                ) from exc
            if not math.isfinite(_weight) or _weight > 0.0:
                raise _OverrideError(
                    f"rewards.{term_name}.weight must be finite and <= 0"
                )
            _param = None
            if _param_raw is not None:
                if isinstance(_param_raw, bool):
                    raise _OverrideError(param_positive_msg)
                try:
                    _param = float(_param_raw)
                except (TypeError, ValueError) as exc:
                    raise _OverrideError(param_positive_msg) from exc
                if not math.isfinite(_param) or _param <= 0.0:
                    raise _OverrideError(param_positive_msg)
                _require(
                    param_name in _term.params,
                    f"rewards.{term_name}.params['{param_name}']",
                )
            _term.weight = _weight
            applied.append(f"rewards.{term_name}.weight={_weight}")
            if _param is not None:
                _term.params[param_name] = _param
                applied.append(
                    f"rewards.{term_name}.params.{param_name}={_param} ({param_label})"
                )

        _foot_term_overrides(
            "foot_soft_landing",
            "foot_soft_landing_weight",
            "force_threshold_n",
            "foot_soft_landing_force_threshold_n",
            "落地法向冲击力阈值,N",
            "rewards.foot_soft_landing.force_threshold_n must be finite and > 0 (newton)",
        )
        _foot_term_overrides(
            "foot_clearance",
            "foot_clearance_weight",
            "target_m",
            "foot_clearance_target_m",
            "摆动相 ankle_roll 原点目标高度,m",
            "rewards.foot_clearance.target_m must be finite and > 0 (meters)",
        )

        # Wave B: mutually-exclusive lower-body hypotheses with weight-independent probes.  B1
        # is a positive bounded pose kernel on the exact twelve v4rg leg joints.  B2 is one
        # reference-free negative bounded bundle (stance collapse + realized leg-qdot tail).
        # Existing foot_orientation/upright/slip terms are deliberately untouched.
        def _lower_body_number(raw, label, *, positive=False, nonnegative=False):
            if isinstance(raw, bool):
                raise _OverrideError(f"rewards.{label} must be a finite number")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    f"rewards.{label} must be a finite number"
                ) from exc
            if not math.isfinite(value):
                raise _OverrideError(f"rewards.{label} must be a finite number")
            if positive and value <= 0.0:
                raise _OverrideError(f"rewards.{label} must be finite and > 0")
            if nonnegative and value < 0.0:
                raise _OverrideError(f"rewards.{label} must be finite and >= 0")
            return value

        _pose_fields = (
            ("std", "lower_body_pose_imitation_std", True, False),
            ("support_pre_s", "lower_body_pose_imitation_support_pre_s", False, True),
            ("support_post_s", "lower_body_pose_imitation_support_post_s", False, True),
        )
        _pose_weight_raw = _get(rw, "lower_body_pose_imitation_weight")
        # 击球窗内下肢模仿衰减(V2 motion_scale_in_window 的下肢版)。它和其他 pose 参数键
        # 走同一个 Wave-B 信封(配了就必须显式给两个 weight),但 EnvCfg 血统里不带这个
        # params 项——和 motion_scale_in_window 一样只由覆盖层写入,函数默认 1.0 = 字节等价。
        _pose_scale_in_window_raw = _get(rw, "lower_body_imitation_scale_in_window")
        _pose_requested = (
            _pose_weight_raw is not None
            or _pose_scale_in_window_raw is not None
            or any(_get(rw, key) is not None for _, key, _, _ in _pose_fields)
        )
        _bundle_fields = (
            ("min_stance_width_m", "lower_body_stability_min_stance_width_m", True, False),
            ("stance_scale_m", "lower_body_stability_stance_scale_m", True, False),
            (
                "leg_velocity_margin_radps",
                "lower_body_stability_leg_velocity_margin_radps",
                False,
                True,
            ),
            (
                "leg_velocity_scale_radps",
                "lower_body_stability_leg_velocity_scale_radps",
                True,
                False,
            ),
            ("support_pre_s", "lower_body_stability_support_pre_s", False, True),
            ("support_post_s", "lower_body_stability_support_post_s", False, True),
        )
        _bundle_weight_raw = _get(rw, "lower_body_stability_bundle_weight")
        _bundle_requested = _bundle_weight_raw is not None or any(
            _get(rw, key) is not None for _, key, _, _ in _bundle_fields
        )
        # Validate the paired cell envelope before mutating either reward cfg; every valid
        # B0/B1/B2 cell binds both probes.
        if (_pose_requested or _bundle_requested) and (
            _pose_weight_raw is None or _bundle_weight_raw is None
        ):
            raise _OverrideError(
                "Wave-B B0/B1/B2 requires both lower_body_pose_imitation_weight "
                "and lower_body_stability_bundle_weight explicitly"
            )
        _pose_term = _pose_probe = None
        _bundle_term = _bundle_probe = None
        _pose_weight = _bundle_weight = None
        _pose_param_values = {}
        _bundle_param_values = {}
        if _pose_requested:
            _require(
                hasattr(R, "lower_body_pose_imitation")
                and R.lower_body_pose_imitation is not None,
                "rewards.lower_body_pose_imitation",
            )
            _require(
                hasattr(R, "lower_body_pose_imitation_probe")
                and R.lower_body_pose_imitation_probe is not None,
                "rewards.lower_body_pose_imitation_probe",
            )
            _pose_term = R.lower_body_pose_imitation
            _pose_probe = R.lower_body_pose_imitation_probe
            _require(
                isinstance(_pose_term.params, dict) and isinstance(_pose_probe.params, dict),
                "rewards.lower_body_pose_imitation.params",
            )
            _pose_weight = _lower_body_number(
                _pose_weight_raw,
                "lower_body_pose_imitation_weight",
                nonnegative=True,
            )
            for _param, _key, _positive, _nonnegative in _pose_fields:
                _raw = _get(rw, _key)
                if _raw is None:
                    continue
                _require(
                    _param in _pose_term.params,
                    f"rewards.lower_body_pose_imitation.params['{_param}']",
                )
                _pose_param_values[_param] = _lower_body_number(
                    _raw, _key, positive=_positive, nonnegative=_nonnegative
                )
            if _pose_scale_in_window_raw is not None:
                # 覆盖层专属参数(血统不带,故不做 "已在 params 里" 的检查);0 = 窗内全静音,
                # 1 = 不衰减。>=0 即合法,和 motion_scale_in_window 同宽容度。
                _pose_param_values["scale_in_window"] = _lower_body_number(
                    _pose_scale_in_window_raw,
                    "lower_body_imitation_scale_in_window",
                    nonnegative=True,
                )
            if (
                _pose_term.params.get("racket_command_name") != "racket_target"
                or _pose_term.params.get("motion_command_name") != "motion"
            ):
                raise _OverrideError(
                    "rewards.lower_body_pose_imitation requires racket_target/motion commands"
                )
        if _bundle_requested:
            _require(
                hasattr(R, "lower_body_stability_bundle")
                and R.lower_body_stability_bundle is not None,
                "rewards.lower_body_stability_bundle",
            )
            _require(
                hasattr(R, "lower_body_stability_bundle_probe")
                and R.lower_body_stability_bundle_probe is not None,
                "rewards.lower_body_stability_bundle_probe",
            )
            _bundle_term = R.lower_body_stability_bundle
            _bundle_probe = R.lower_body_stability_bundle_probe
            _require(
                isinstance(_bundle_term.params, dict) and isinstance(_bundle_probe.params, dict),
                "rewards.lower_body_stability_bundle.params",
            )
            _bundle_weight = _lower_body_number(
                _bundle_weight_raw, "lower_body_stability_bundle_weight"
            )
            if _bundle_weight > 0.0:
                raise _OverrideError(
                    "rewards.lower_body_stability_bundle_weight must be finite and <= 0"
                )
            for _param, _key, _positive, _nonnegative in _bundle_fields:
                _raw = _get(rw, _key)
                if _raw is None:
                    continue
                _require(
                    _param in _bundle_term.params,
                    f"rewards.lower_body_stability_bundle.params['{_param}']",
                )
                _bundle_param_values[_param] = _lower_body_number(
                    _raw, _key, positive=_positive, nonnegative=_nonnegative
                )
            if (
                _bundle_term.params.get("racket_command_name") != "racket_target"
                or _bundle_term.params.get("motion_command_name") != "motion"
            ):
                raise _OverrideError(
                    "rewards.lower_body_stability_bundle requires racket_target/motion commands"
                )
        if (
            _pose_weight is not None
            and _bundle_weight is not None
            and _pose_weight > 0.0
            and _bundle_weight < 0.0
        ):
            raise _OverrideError(
                "Wave-B B1 pose imitation and B2 stability bundle are mutually exclusive"
            )
        if _pose_requested:
            _pose_term.weight = _pose_weight
            applied.append(
                f"rewards.lower_body_pose_imitation.weight={_pose_weight}"
            )
            for _param, _value in _pose_param_values.items():
                _pose_term.params[_param] = _value
                applied.append(
                    f"rewards.lower_body_pose_imitation.params.{_param}={_value}"
                )
            _pose_probe.weight = 1.0
            _pose_probe.params.clear()
            _pose_probe.params.update(_pose_term.params)
            applied.append("rewards.lower_body_pose_imitation_probe.weight=1.0")

        if _bundle_requested:
            _bundle_term.weight = _bundle_weight
            applied.append(
                f"rewards.lower_body_stability_bundle.weight={_bundle_weight}"
            )
            for _param, _value in _bundle_param_values.items():
                _bundle_term.params[_param] = _value
                applied.append(
                    f"rewards.lower_body_stability_bundle.params.{_param}={_value}"
                )
            _bundle_probe.weight = 1.0
            _bundle_probe.params.clear()
            _bundle_probe.params.update(_bundle_term.params)
            applied.append("rewards.lower_body_stability_bundle_probe.weight=1.0")

        if (
            hasattr(R, "lower_body_pose_imitation")
            and R.lower_body_pose_imitation is not None
            and hasattr(R, "lower_body_stability_bundle")
            and R.lower_body_stability_bundle is not None
            and float(R.lower_body_pose_imitation.weight) > 0.0
            and float(R.lower_body_stability_bundle.weight) < 0.0
        ):
            raise _OverrideError(
                "Wave-B B1 pose imitation and B2 stability bundle are mutually exclusive"
            )

        # S1 post-swing settle debt (Jiayi V13 post-swing debts idea, clean main-side redo).
        # Same fail-loud contract as the Wave-B pair: any explicit S1 key requires the weight,
        # nothing mutates until every value validates, and the probe follows the reward params.
        _settle_fields = tuple(
            (name, f"post_swing_settle_{name}", positive, nonnegative)
            for name, positive, nonnegative in _POST_SWING_SETTLE_NUMERIC_SPECS
        )
        _settle_weight_raw = _get(rw, "post_swing_settle_debt_weight")
        _settle_requested = _settle_weight_raw is not None or any(
            _get(rw, key) is not None for _, key, _, _ in _settle_fields
        )
        if _settle_requested:
            if _settle_weight_raw is None:
                raise _OverrideError(
                    "S1 requires post_swing_settle_debt_weight explicitly alongside any "
                    "post_swing_settle_* parameter"
                )
            _require(
                hasattr(R, "post_swing_settle_debt")
                and R.post_swing_settle_debt is not None,
                "rewards.post_swing_settle_debt",
            )
            _require(
                hasattr(R, "post_swing_settle_debt_probe")
                and R.post_swing_settle_debt_probe is not None,
                "rewards.post_swing_settle_debt_probe",
            )
            _settle_term = R.post_swing_settle_debt
            _settle_probe = R.post_swing_settle_debt_probe
            _require(
                isinstance(_settle_term.params, dict)
                and isinstance(_settle_probe.params, dict),
                "rewards.post_swing_settle_debt.params",
            )
            _settle_weight = _lower_body_number(
                _settle_weight_raw, "post_swing_settle_debt_weight"
            )
            if _settle_weight > 0.0:
                raise _OverrideError(
                    "rewards.post_swing_settle_debt_weight must be finite and <= 0"
                )
            _settle_param_values = {}
            for _param, _key, _positive, _nonnegative in _settle_fields:
                _raw = _get(rw, _key)
                if _raw is None:
                    continue
                _require(
                    _param in _settle_term.params,
                    f"rewards.post_swing_settle_debt.params['{_param}']",
                )
                _settle_param_values[_param] = _lower_body_number(
                    _raw, _key, positive=_positive, nonnegative=_nonnegative
                )
            if (
                _settle_term.params.get("racket_command_name") != "racket_target"
                or _settle_term.params.get("motion_command_name") != "motion"
            ):
                raise _OverrideError(
                    "rewards.post_swing_settle_debt requires racket_target/motion commands"
                )
            _settle_start = _settle_param_values.get(
                "recovery_start_s", _settle_term.params.get("recovery_start_s")
            )
            _settle_end = _settle_param_values.get(
                "recovery_end_s", _settle_term.params.get("recovery_end_s")
            )
            if (
                not isinstance(_settle_start, (int, float))
                or not isinstance(_settle_end, (int, float))
                or isinstance(_settle_start, bool)
                or isinstance(_settle_end, bool)
                or not float(_settle_start) < float(_settle_end)
            ):
                raise _OverrideError(
                    "rewards.post_swing_settle_debt recovery window must satisfy 0 <= start < end"
                )
            _settle_term.weight = _settle_weight
            applied.append(f"rewards.post_swing_settle_debt.weight={_settle_weight}")
            for _param, _value in _settle_param_values.items():
                _settle_term.params[_param] = _value
                applied.append(
                    f"rewards.post_swing_settle_debt.params.{_param}={_value}"
                )
            _settle_probe.weight = 1.0
            _settle_probe.params.clear()
            _settle_probe.params.update(_settle_term.params)
            applied.append("rewards.post_swing_settle_debt_probe.weight=1.0")

        # Matrix S-axis: S1/S2/S3 are mutually exclusive treatments — a cell enables at most one.
        if (
            hasattr(R, "post_swing_settle_debt")
            and R.post_swing_settle_debt is not None
            and float(R.post_swing_settle_debt.weight) < 0.0
            and (
                (
                    hasattr(R, "lower_body_pose_imitation")
                    and R.lower_body_pose_imitation is not None
                    and float(R.lower_body_pose_imitation.weight) > 0.0
                )
                or (
                    hasattr(R, "lower_body_stability_bundle")
                    and R.lower_body_stability_bundle is not None
                    and float(R.lower_body_stability_bundle.weight) < 0.0
                )
            )
        ):
            raise _OverrideError(
                "S1 post_swing_settle_debt and the Wave-B lower-body mechanisms are mutually exclusive"
            )
        _foot_hold_gate = _get(rw, "foot_orientation_hold_gate")
        if _foot_hold_gate is not None:
            _require(hasattr(R, "foot_orientation"), "rewards.foot_orientation")
            R.foot_orientation.params["hold_gate"] = _as_bool(_foot_hold_gate)
            applied.append(
                f"rewards.foot_orientation.params.hold_gate={_as_bool(_foot_hold_gate)}"
            )
        # R16 (franco 2026-07-04): free the racket wrist from ORIENTATION mimic. Config-level only —
        # drop the racket-mount link from the body lists of the two orientation-imitation terms;
        # position / linear-velocity mimic keep the swing path, and the face orientation is then
        # shaped by the racket_normal reward alone (commanded normal at contract v3).
        if _get(rw, "free_wrist_ori_mimic") is not None and _as_bool(_get(rw, "free_wrist_ori_mimic")):
            _WRIST = "right_wrist_yaw_Link"
            for _tn in ("motion_body_ori", "motion_body_ang_vel"):
                _require(hasattr(R, _tn), f"rewards.{_tn}")
                _term = getattr(R, _tn)
                _names = [b for b in _term.params["body_names"] if b != _WRIST]
                _require(len(_names) < len(_term.params["body_names"]),
                         f"rewards.{_tn}.params.body_names contains {_WRIST}")
                _term.params["body_names"] = _names
                applied.append(f"rewards.{_tn}.body_names-={_WRIST}")
        # V1 (reward_staged_design 2026-07-08 §③): free the racket wrist from the LINEAR-VELOCITY
        # mimic — the "second master" fight free_wrist_ori_mimic did NOT touch (the question-bank
        # answer velocity sits a median 34° outside the teacher's swing-velocity cone, so
        # motion_body_lin_vel pulls the wrist toward the teacher while racket_velocity pulls it
        # toward the answer). Config-level only, same drop-the-link pattern as the ori flag above:
        # the wrist's linear velocity is then shaped by racket_velocity alone; position/ori mimic
        # keep the swing path. Default OFF = body list untouched, byte-identical baseline.
        if _get(rw, "free_wrist_vel_mimic") is not None and _as_bool(_get(rw, "free_wrist_vel_mimic")):
            _WRIST = "right_wrist_yaw_Link"
            for _tn in ("motion_body_lin_vel",):
                _require(hasattr(R, _tn), f"rewards.{_tn}")
                _term = getattr(R, _tn)
                _require(_term is not None and "body_names" in _term.params,
                         f"rewards.{_tn}.params['body_names'] (free_wrist_vel_mimic needs an explicit body list)")
                _names = [b for b in _term.params["body_names"] if b != _WRIST]
                _require(len(_names) < len(_term.params["body_names"]),
                         f"rewards.{_tn}.params.body_names contains {_WRIST}")
                _term.params["body_names"] = _names
                applied.append(f"rewards.{_tn}.body_names-={_WRIST}")
            _require(hasattr(env_cfg.commands, "motion"),
                     "commands.motion (free_wrist_vel_mimic activation ledger)")
            env_cfg.commands.motion.v1_free_wrist_vel_mimic_activation = True
        # A0/A1 (Franco 2026-07-13): test whether the non-racket LEFT arm should remain free to
        # regulate balance instead of copying the teacher.  Fail closed on the exact current
        # upper-body contract; broad regexes or best-effort subtraction could silently release a
        # torso/right-arm body after a future asset rename.  A1 changes ONLY the four imitation
        # body lists.  Joint/action limits, torque/contact/self-collision rewards, all
        # terminations, and every other reward remain untouched.
        _free_non_striking = _get(rw, "free_non_striking_arm_mimic")
        if _free_non_striking is not None and _as_explicit_bool(
            _free_non_striking, "task.rewards.free_non_striking_arm_mimic"
        ):
            _EXPECTED_UPPER = (
                "torso_Link",
                "left_shoulder_roll_Link",
                "left_elbow_Link",
                "left_wrist_yaw_Link",
                "right_shoulder_roll_Link",
                "right_elbow_Link",
                "right_wrist_yaw_Link",
            )
            _EXPECTED_WITHOUT_RACKET_WRIST = tuple(
                name for name in _EXPECTED_UPPER if name != "right_wrist_yaw_Link"
            )
            _LEFT_NON_STRIKING = {
                "left_shoulder_roll_Link",
                "left_elbow_Link",
                "left_wrist_yaw_Link",
            }
            _A1_BODY_TERMS = (
                "motion_body_pos",
                "motion_body_ori",
                "motion_body_lin_vel",
                "motion_body_ang_vel",
            )
            for _tn in _A1_BODY_TERMS:
                _require(hasattr(R, _tn), f"rewards.{_tn}")
                _term = getattr(R, _tn)
                _require(
                    _term is not None and isinstance(_term.params.get("body_names"), (list, tuple)),
                    f"rewards.{_tn}.params['body_names'] (non-striking-arm ablation needs an explicit body list)",
                )
                _before = tuple(str(name) for name in _term.params["body_names"])
                # The current Phase-1 recipe already opts out of racket-wrist ORIENTATION mimic
                # through free_wrist_ori_mimic=true.  Accept exactly that reviewed six-body
                # variant as well as the seven-body base, then preserve whichever right-arm list
                # A0 supplied.  No other omission/reordering is accepted.
                _allowed_before = {_EXPECTED_UPPER}
                if _tn in ("motion_body_ori", "motion_body_ang_vel"):
                    _allowed_before.add(_EXPECTED_WITHOUT_RACKET_WRIST)
                if _tn == "motion_body_lin_vel":
                    # V1 is separately reviewed and may be composed explicitly in a future pair.
                    _allowed_before.add(_EXPECTED_WITHOUT_RACKET_WRIST)
                _require(
                    _before in _allowed_before,
                    f"rewards.{_tn}.params.body_names equals an exact reviewed A3 upper contract",
                )
                _after = [name for name in _before if name not in _LEFT_NON_STRIKING]
                _require(
                    tuple(name for name in _after if name == "torso_Link" or name.startswith("right_"))
                    == tuple(
                        name
                        for name in _before
                        if name == "torso_Link" or name.startswith("right_")
                    ),
                    f"rewards.{_tn}.params.body_names preserves the exact A0 torso/right-arm list",
                )
                _term.params["body_names"] = _after
                applied.append(
                    f"rewards.{_tn}.body_names={_after} "
                    "(left non-striking arm imitation removed)"
                )
        # 全身模仿(Franco 2026-07-25):下半身回到全局模仿名单,flag 可开关。把 pelvis +
        # 六个腿部 link 加回四个 motion_body_* 名单——恢复 BeyondMimic 原始 13-body 集。
        # 顺序在 wrist/left-arm 摘除之后:被摘的照旧不学,腿加回。fail-loud 面:名单必须
        # 是已审的上身合同变体(7-body 基线或摘 wrist/左臂后的 4/5/6-body),且不得已含
        # 任何下身 link。注意核是 body-均值:加 7 件 = 同权重下核变难、模仿收益变低
        # (F5 反向),预算表按人数换算。
        _full_body = _get(rw, "full_body_mimic")
        if _full_body is not None and _as_explicit_bool(
            _full_body, "task.rewards.full_body_mimic"
        ):
            _LOWER_BODY = (
                "pelvis_link",
                "left_hip_roll_Link",
                "left_knee_Link",
                "left_ankle_roll_Link",
                "right_hip_roll_Link",
                "right_knee_Link",
                "right_ankle_roll_Link",
            )
            _UPPER_CONTRACT_CORE = (
                "torso_Link",
                "left_shoulder_roll_Link",
                "left_elbow_Link",
                "left_wrist_yaw_Link",
                "right_shoulder_roll_Link",
                "right_elbow_Link",
                "right_wrist_yaw_Link",
            )
            _FB_BODY_TERMS = (
                "motion_body_pos",
                "motion_body_ori",
                "motion_body_lin_vel",
                "motion_body_ang_vel",
            )
            for _tn in _FB_BODY_TERMS:
                _require(hasattr(R, _tn), f"rewards.{_tn}")
                _term = getattr(R, _tn)
                _require(
                    _term is not None and isinstance(_term.params.get("body_names"), (list, tuple)),
                    f"rewards.{_tn}.params['body_names'] (full-body mimic needs an explicit body list)",
                )
                _before = tuple(str(name) for name in _term.params["body_names"])
                _require(
                    all(name not in _LOWER_BODY for name in _before),
                    f"rewards.{_tn}.params.body_names must not already contain lower-body links",
                )
                # 下限 3 = 同时摘 wrist + 左臂后的最小合同(torso + right_shoulder + right_elbow)。
                _require(
                    all(name in _UPPER_CONTRACT_CORE for name in _before)
                    and len(set(_before)) == len(_before)
                    and len(_before) >= 3
                    and "torso_Link" in _before,
                    f"rewards.{_tn}.params.body_names equals a reviewed A3 upper contract variant",
                )
                _after = list(_LOWER_BODY) + list(_before)
                _term.params["body_names"] = _after
                applied.append(
                    f"rewards.{_tn}.body_names={_after} "
                    "(full-body mimic: pelvis + 6 leg links restored)"
                )
        _vlw = _get(rw, "virtual_landing_weight")
        if _vlw is not None:
            _vlw_f = float(_vlw)
            _require(
                math.isfinite(_vlw_f) and _vlw_f >= 0.0,
                "rewards.virtual_landing_weight (finite, >= 0)",
            )
            _require(
                hasattr(R, "virtual_landing") and R.virtual_landing is not None,
                "rewards.virtual_landing (weight override)",
            )
            R.virtual_landing.weight = _vlw_f
            applied.append(f"rewards.virtual_landing.weight={_vlw_f}")
        _vlb = _get(rw, "virtual_landing_base_frac")
        if _vlb is not None:
            _vlb_f = float(_vlb)
            _require(
                math.isfinite(_vlb_f) and 0.0 < _vlb_f < 1.0,
                "rewards.virtual_landing_base_frac (in (0, 1))",
            )
            R.virtual_landing.params["base_frac"] = _vlb_f
            applied.append(f"rewards.virtual_landing.params.base_frac={_vlb_f}")
        _sds = _get(rw, "virtual_landing_settle_delay_s")
        if _sds is not None:
            _sds_f = float(_sds)
            _require(
                math.isfinite(_sds_f) and _sds_f >= 0.0,
                "rewards.virtual_landing_settle_delay_s (finite, >= 0)",
            )
            _require(
                hasattr(R, "virtual_landing") and R.virtual_landing is not None,
                "rewards.virtual_landing (settle_delay override)",
            )
            R.virtual_landing.params["settle_delay_s"] = _sds_f
            applied.append(f"rewards.virtual_landing.params.settle_delay_s={_sds_f}")
        _dpw = _get(rw, "death_penalty_weight")
        if _dpw is not None:
            _dpw_f = float(_dpw)
            # 摔死罚只许 <=0(0=消融关闭);包 direct 写 -1800 在前,用户键在此压包
            _require(
                math.isfinite(_dpw_f) and _dpw_f <= 0.0,
                "rewards.death_penalty_weight (finite, <= 0)",
            )
            _require(
                hasattr(R, "death_penalty") and R.death_penalty is not None,
                "rewards.death_penalty (weight override)",
            )
            R.death_penalty.weight = _dpw_f
            applied.append(f"rewards.death_penalty.weight={_dpw_f}")

        # 撞桌罚(07-27)。与 death_penalty_weight 同形:只许 <=0,0 = 消融关闭。
        # 该项由 apply_table_obstacle 随桌子装上;桌子被关掉时它不存在,这时给非零权重是
        # 配置错误(罚一个永远不会发生的事),直接报错而不是静默忽略。
        _thw = _get(rw, "table_hit_penalty_weight")
        if _thw is not None:
            _thw_f = float(_thw)
            _require(
                math.isfinite(_thw_f) and _thw_f <= 0.0,
                "rewards.table_hit_penalty_weight (finite, <= 0)",
            )
            _require(
                hasattr(R, "table_hit_penalty") and R.table_hit_penalty is not None,
                "rewards.table_hit_penalty (weight override; needs task.table_obstacle=true)",
            )
            R.table_hit_penalty.weight = _thw_f
            applied.append(f"rewards.table_hit_penalty.weight={_thw_f}")
        jt = _get(rw, "joint_torques_weight")
        if jt is not None:
            _require(hasattr(R, "joint_torques"), "rewards.joint_torques")
            R.joint_torques.weight = float(jt)
            applied.append(f"rewards.joint_torques.weight={float(jt)}")

        # --- motion imitation prior (the 6 motion_* terms; base weights sum ~5.0) ---------------
        # `motion_scale` multiplies all six at once — the main lever to demote imitation to a soft
        # prior so the racket goal can dominate. Per-term weight/std overrides are also accepted
        # (e.g. motion_body_pos_weight / motion_body_pos_std) and are applied BEFORE the scale.
        _MOTION_TERMS = (
            "motion_global_anchor_pos", "motion_global_anchor_ori",
            "motion_body_pos", "motion_body_ori",
            "motion_body_lin_vel", "motion_body_ang_vel",
        )
        for _t in _MOTION_TERMS:
            _set_reward(R, _t, _get(rw, f"{_t}_weight"), _get(rw, f"{_t}_std"), applied)
        ms = _get(rw, "motion_scale")
        if ms is not None:
            ms = float(ms)
            _scaled = []
            for _t in _MOTION_TERMS:
                _require(hasattr(R, _t), f"rewards.{_t}")
                _term = getattr(R, _t)
                if _term is None:
                    continue  # term REMOVED in this cfg lineage (e.g. footwork cfg sets
                              # motion_global_anchor_pos = None) — nothing to scale
                _term.weight *= ms
                _scaled.append(_t)
            _require(len(_scaled) > 0, "rewards.motion_scale (all six motion terms are None)")
            applied.append(f"rewards.motion_scale={ms} (x{len(_scaled)} motion weights: "
                           + ",".join(_scaled) + ")")
        # V2 (reward_staged_design 2026-07-08 §③): IN-WINDOW imitation yield — inside the strike
        # window every motion_* mimic term is multiplied by k (0 = teacher fully silent, 0.25 =
        # quarter voice); outside the window imitation pays in full. 人话:触球窗内老师闭嘴(或小
        # 声),听题目的。A RewTerm weight is a constant, so the gating happens INSIDE the reward
        # funcs via the window_scale/window_command_name params they all now accept; the mask is
        # the racket command's WIDE strike window (== the legacy strike_window unless the 1c
        # split-window flags set racket.strike_window_wide_s), so V2 composes with V3/1c.
        # Default (key absent) = params untouched = byte-identical baseline.
        msw = _get(rw, "motion_scale_in_window")
        if msw is not None:
            msw = float(msw)
            _require(hasattr(env_cfg.commands, "racket_target"),
                     "commands.racket_target (rewards.motion_scale_in_window needs the strike window)")
            _gated = []
            for _t in _MOTION_TERMS:
                _require(hasattr(R, _t), f"rewards.{_t}")
                _term = getattr(R, _t)
                if _term is None:
                    continue  # term REMOVED in this cfg lineage (e.g. footwork cfg) — nothing to gate
                _term.params["window_scale"] = msw
                _term.params["window_command_name"] = "racket_target"
                _gated.append(_t)
            _require(len(_gated) > 0, "rewards.motion_scale_in_window (all six motion terms are None)")
            _require(hasattr(env_cfg.commands, "motion"),
                     "commands.motion (motion_scale_in_window activation ledger)")
            env_cfg.commands.motion.v2_motion_scale_in_window_activation = msw
            applied.append(f"rewards.motion_scale_in_window={msw} (x{len(_gated)} motion terms inside "
                           "the strike window: " + ",".join(_gated) + ")")
        # Proximity power-gate (reward_staged_design 2026-07-08 §② C2 case a): racket_normal and
        # racket_velocity are additionally multiplied by sigmoid((r_gate - pos_err)/0.05) — the
        # face/velocity channels only power on when the paddle can physically reach the target.
        # 人话:拍子够得着球,才开始付拍面/拍速的钱;够不着时也不吃它们的梯度噪声。
        # racket_position (the reach gradient) and racket_strike_success (already multiplicative)
        # are NOT gated. Both keys must be set together — a half-configured gate fails loud.
        _fg = _get(rw, "face_gate_by_pos")
        _fg_r = _get(rw, "face_gate_radius")
        if _fg is not None and _as_bool(_fg):
            if _fg_r is None:
                raise _OverrideError(
                    "[train.py] rewards.face_gate_by_pos=true requires rewards.face_gate_radius "
                    "(meters; design candidates 0.15 = 2x strike_success_pos_thresh, or 0.095 = "
                    "the vb capture gate).")
            for _tn in ("racket_velocity", "racket_normal"):
                _require(hasattr(R, _tn), f"rewards.{_tn}")
                getattr(R, _tn).params["pos_gate_radius"] = float(_fg_r)
            applied.append(f"rewards.face_gate_by_pos=true (racket_velocity+racket_normal x "
                           f"sigmoid(({float(_fg_r)}-pos_err)/0.05))")
        elif _fg_r is not None:
            raise _OverrideError(
                "[train.py] rewards.face_gate_radius is set but face_gate_by_pos is not enabled — "
                "the radius would be silently ignored. Set rewards.face_gate_by_pos=true or drop it.")
        # Constant guidance penalty (reward_staged_design 2026-07-08 §② B2): -w * min(dist, 0.5)
        # every pre-strike + in-window step (dist = ||racket_FK - target||). 人话:挥不到球也天天
        # 有"往哪挥"的工资单——小而恒,exp 核远处饿死的解药。The func returns a POSITIVE clamped
        # magnitude, so the weight must be <= 0; 0.0 (the cfg default) keeps the term skipped.
        _gw = _get(rw, "racket_guidance_weight")
        if _gw is not None:
            _gw = float(_gw)
            if _gw > 0.0:
                raise _OverrideError(
                    f"[train.py] rewards.racket_guidance_weight must be <= 0 (penalty; the term "
                    f"returns +min(dist, d_max)), got {_gw}")
            _require(hasattr(R, "racket_guidance"), "rewards.racket_guidance")
            R.racket_guidance.weight = _gw
            applied.append(f"rewards.racket_guidance.weight={_gw}")
        # Face-angle guidance penalty (2026-07-10): -w * min(angle, theta_max) — the face-channel
        # twin of racket_guidance (exp 拍面核死区解药;翻面修复后 33-53° 残差全在 exp 零梯度带)。
        # POSITIVE radians from the func, so the weight must be <= 0; 0.0 (cfg default) = skipped.
        _fgw = _get(rw, "racket_face_guidance_weight")
        if _fgw is not None:
            _fgw = float(_fgw)
            if _fgw > 0.0:
                raise _OverrideError(
                    f"[train.py] rewards.racket_face_guidance_weight must be <= 0 (penalty; the "
                    f"term returns +min(angle, theta_max) radians), got {_fgw}")
            _require(hasattr(R, "racket_face_guidance"), "rewards.racket_face_guidance")
            R.racket_face_guidance.weight = _fgw
            applied.append(f"rewards.racket_face_guidance.weight={_fgw}")
        # theta_max passthrough: the cfg default pi/2 zeroes the gradient exactly where a >90°
        # dead-zone start needs it (G1 swingsyn bh enters at ~95°) — rescues must pin pi.
        _fgt = _get(rw, "racket_face_guidance_theta_max")
        if _fgt is not None:
            _fgt = float(_fgt)
            if not 0.0 < _fgt <= 3.1415926535897932:
                raise _OverrideError(
                    f"[train.py] rewards.racket_face_guidance_theta_max must be in (0, pi] "
                    f"radians (pi = no clamp), got {_fgt}")
            _require(hasattr(R, "racket_face_guidance"), "rewards.racket_face_guidance")
            R.racket_face_guidance.params["theta_max"] = _fgt
            applied.append(f"rewards.racket_face_guidance.params.theta_max={_fgt}")
        # Conditional fixed-budget face guidance (2026-07-14): the function returns [0,1] and is
        # identically zero outside the strike window.  Inside it, unready states keep the fixed maximum
        # cost and readiness converts that cost into signed-face error, so deliberately leaving the
        # gate cannot evade a negative penalty.  Therefore |weight| is the maximum per-window-step
        # budget.  All thresholds are source-frozen and hard-bound; this flag selects the mechanism.
        _cfgw_raw = _get(rw, "racket_face_conditional_guidance_weight")
        if _cfgw_raw is not None:
            if isinstance(_cfgw_raw, bool):
                raise _OverrideError(
                    "[train.py] rewards.racket_face_conditional_guidance_weight must be a finite "
                    "number <= 0, not a boolean"
                )
            try:
                _cfgw = float(_cfgw_raw)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    "[train.py] rewards.racket_face_conditional_guidance_weight must be a finite "
                    f"number <= 0, got {_cfgw_raw!r}"
                ) from exc
            if not math.isfinite(_cfgw) or _cfgw > 0.0:
                raise _OverrideError(
                    "[train.py] rewards.racket_face_conditional_guidance_weight must be a finite "
                    f"number <= 0, got {_cfgw_raw!r}"
                )
            _require(
                hasattr(R, "racket_face_conditional_guidance"),
                "rewards.racket_face_conditional_guidance",
            )
            R.racket_face_conditional_guidance.weight = _cfgw
            applied.append(f"rewards.racket_face_conditional_guidance.weight={_cfgw}")

        # --- penalties / regularization (negative weights: energy + smoothness + safety) --------
        _action_rate_weight = _get(rw, "action_rate_weight")
        if _action_rate_weight is not None:
            if isinstance(_action_rate_weight, bool):
                raise _OverrideError(
                    "rewards.action_rate_l2.weight must be finite and <= 0"
                )
            try:
                _action_rate_weight_value = float(_action_rate_weight)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    "rewards.action_rate_l2.weight must be finite and <= 0"
                ) from exc
            if (
                not math.isfinite(_action_rate_weight_value)
                or _action_rate_weight_value > 0.0
            ):
                raise _OverrideError(
                    "rewards.action_rate_l2.weight must be finite and <= 0"
                )
            _require(
                hasattr(R, "action_rate_l2") and R.action_rate_l2 is not None,
                "rewards.action_rate_l2",
            )
            R.action_rate_l2.weight = _action_rate_weight_value
            applied.append(
                f"rewards.action_rate_l2.weight={_action_rate_weight_value}"
            )
        # mjlab 档①第三项:action_acc_l2 动作二阶差分罚(默认 weight=0 关断,字节等价)。
        # 校验拼法照 action_rate:bool 拒收,必须 finite 且 <= 0(显式 0 = 对照臂)。
        _action_acc_weight = _get(rw, "action_acc_weight")
        if _action_acc_weight is not None:
            if isinstance(_action_acc_weight, bool):
                raise _OverrideError(
                    "rewards.action_acc_l2.weight must be finite and <= 0"
                )
            try:
                _action_acc_weight_value = float(_action_acc_weight)
            except (TypeError, ValueError) as exc:
                raise _OverrideError(
                    "rewards.action_acc_l2.weight must be finite and <= 0"
                ) from exc
            if (
                not math.isfinite(_action_acc_weight_value)
                or _action_acc_weight_value > 0.0
            ):
                raise _OverrideError(
                    "rewards.action_acc_l2.weight must be finite and <= 0"
                )
            _require(
                hasattr(R, "action_acc_l2") and R.action_acc_l2 is not None,
                "rewards.action_acc_l2",
            )
            R.action_acc_l2.weight = _action_acc_weight_value
            applied.append(
                f"rewards.action_acc_l2.weight={_action_acc_weight_value}"
            )
        for _name, _key in (
            ("joint_limit", "joint_limit_weight"),
            ("undesired_contacts", "undesired_contacts_weight"),
            ("pre_strike_foot_slip", "pre_strike_foot_slip_weight"),
            # 触地脚蹭滑/拖脚(2026-07-22):此前源码常开 -1.0/-0.5 且 CLI 够不着,
            # penlight 减负臂与将来消融都需要显式剂量。
            ("foot_slip_sq", "foot_slip_sq_weight"),
            ("foot_drag", "foot_drag_weight"),
            ("prestrike_waist_twist", "prestrike_waist_twist_weight"),
            # sim2real fine-tune (explicit-PD): torque-saturation penalty + pre-strike upright shaping.
            ("arm_torque_saturation", "arm_torque_saturation_weight"),
            ("prestrike_upright", "prestrike_upright_weight"),
            # Foot discipline (jiayi hold-fall stack, 2026-07-05): hip yaw/roll + ankle roll held to
            # the reference footwork. Cfg default 0.0 (merge-audit flag-off); jiayi lineages pin it.
            ("foot_orientation", "foot_orientation_weight"),
        ):
            _w = _get(rw, _key)
            if _w is not None:
                _require(hasattr(R, _name), f"rewards.{_name}")
                getattr(R, _name).weight = float(_w)
                if _name != "arm_torque_saturation":
                    applied.append(f"rewards.{_name}.weight={float(_w)}")

    # actions: deploy-faithful action-processing switches (train==deploy parity knobs).
    ac = _get(task, "actions")
    _check_unknown_keys(ac, ("qdes_clamp",), "task.actions")
    if ac is not None:
        _qc = _get(ac, "qdes_clamp")
        if _qc is not None:
            _require(hasattr(env_cfg.actions, "joint_pos") and hasattr(env_cfg.actions.joint_pos, "clamp"),
                     "actions.joint_pos.clamp")
            env_cfg.actions.joint_pos.clamp = _as_bool(_qc)
            applied.append(f"actions.joint_pos.clamp={_as_bool(_qc)}")

    rk = _get(task, "racket")
    _check_unknown_keys(rk, _RACKET_KEYS, "task.racket")
    if _venue_racket is not None:
        # venue profile 的 mocap/transport 键并入 task.racket 视图(用户显式键赢),走下面
        # 的现有翻译层落地 —— 记账/校验/副作用与手写 yaml 完全一致。
        rk = _inject_venue_racket_keys(rk, _venue_racket[0], _venue_racket[1], applied)
    if rk is not None:
        # Only require the racket_target command when the YAML actually sets racket keys, so tasks
        # without a racket objective (e.g. TrackingFlat, which has no `racket:` block) never trip this.
        provided = [k for k in _RACKET_KEYS if _get(rk, k) is not None]
        if provided:
            _require(hasattr(env_cfg.commands, "racket_target"),
                     f"commands.racket_target (task YAML sets racket keys {provided})")
            C = env_cfg.commands.racket_target
            # strike_phase is PER-MOTION: the racket-tip contact frame differs per clip, so a single
            # global value is wrong when the trained motion changes (forehand 0.46 vs backhand 0.59).
            # `strike_phase_by_motion` (clip-name substring -> phase) wins when it matches `clip_name`;
            # `strike_phase` is the fallback. See _resolve_strike_phase / _registry_clip_name.
            _sp_val, _sp_note = _resolve_strike_phase(rk, clip_name)
            _set_attr(C, "strike_phase", _sp_val, float, applied, "racket_target")
            if _sp_note is not None:
                applied.append(_sp_note)
            _set_attr(C, "strike_window_s", _get(rk, "strike_window_s"), float, applied, "racket_target")
            # 1c split strike windows (reward_staged_design 2026-07-08 §② C1): racket_position keeps
            # a TIGHT half-window (contact must be precise, SMASH position window 0.02 s) while
            # racket_normal/racket_velocity get a WIDE one (±0.1 s). None (default) = both windows
            # fall back to strike_window_s = the legacy single window, byte-identical.
            # 人话:触点要准(紧窗),挥向挥速给余量(宽窗)。
            _set_attr(C, "strike_window_pos_s", _get(rk, "strike_window_pos_s"), float, applied, "racket_target")
            _set_attr(C, "strike_window_wide_s", _get(rk, "strike_window_wide_s"), float, applied, "racket_target")
            _set_attr(C, "strike_success_pos_thresh", _get(rk, "strike_success_pos_thresh"), float, applied, "racket_target")
            # P2.3 adaptive tracking sigma (coarse-to-fine reward kernel widths)
            _set_attr(C, "adaptive_sigma", _get(rk, "adaptive_sigma"), _as_bool, applied, "racket_target")
            # 拍面 sigma 第三通道(A1):必须搭在 adaptive_sigma 上,单开由 hope_commands
            # 构造期 fail-loud;reward_pack=v2 也会写它(显式键在此处后写后赢)。
            _set_attr(
                C,
                "adaptive_sigma_normal",
                _get(rk, "adaptive_sigma_normal"),
                lambda value: _as_explicit_bool(value, "task.racket.adaptive_sigma_normal"),
                applied,
                "racket_target",
            )
            _set_attr(C, "sigma_update_every", _get(rk, "sigma_update_every"), int, applied, "racket_target")
            _set_attr(C, "sigma_ema_scale", _get(rk, "sigma_ema_scale"), float, applied, "racket_target")
            _set_attr(C, "sigma_pos_min", _get(rk, "sigma_pos_min"), float, applied, "racket_target")
            _set_attr(C, "sigma_pos_max", _get(rk, "sigma_pos_max"), float, applied, "racket_target")
            _set_attr(C, "sigma_vel_min", _get(rk, "sigma_vel_min"), float, applied, "racket_target")
            _set_attr(C, "sigma_vel_max", _get(rk, "sigma_vel_max"), float, applied, "racket_target")
            _set_range(C, "racket_pos_x_range", _get(rk, "pos_x_range"), applied, "racket_target")
            _set_range(C, "racket_pos_y_range", _get(rk, "pos_y_range"), applied, "racket_target")
            _set_range(C, "racket_pos_z_range", _get(rk, "pos_z_range"), applied, "racket_target")
            # Unified multi-clip: per-clip strike phase (aligned with the clip order) + per-clip |y| region.
            _spc = _get(rk, "strike_phase_per_clip")
            if _spc is not None:
                C.strike_phase_per_clip = tuple(float(x) for x in _spc)
                applied.append(f"racket_target.strike_phase_per_clip={C.strike_phase_per_clip}")
            _set_range(C, "racket_pos_y_abs_range", _get(rk, "racket_pos_y_abs_range"), applied, "racket_target")
            _set_range(C, "racket_vel_x_range", _get(rk, "vel_x_range"), applied, "racket_target")
            _set_range(C, "racket_vel_y_range", _get(rk, "vel_y_range"), applied, "racket_target")
            _set_range(C, "racket_vel_z_range", _get(rk, "vel_z_range"), applied, "racket_target")
            # Optional PER-CLIP velocity boxes (unified policy): forehand=clip 0, backhand=clip 1. Absent ->
            # keep the shared vel_*_range above (backward compatible). The slower backhand needs a lower box.
            _clip_names = _resolve_clip_names(rk)
            if _clip_names != _DEFAULT_CLIP_NAMES:
                _require(hasattr(C, "clip_names_per_clip"), "racket_target.clip_names_per_clip")
                C.clip_names_per_clip = tuple(_clip_names)
                applied.append(f"racket_target.clip_names_per_clip={C.clip_names_per_clip}")
            _vpc = _resolve_vel_range_per_clip(rk, _clip_names)
            if _vpc is not None:
                _require(hasattr(C, "racket_vel_range_per_clip"), "racket_target.racket_vel_range_per_clip")
                C.racket_vel_range_per_clip = _vpc
                applied.append(f"racket_target.racket_vel_range_per_clip={_vpc}")
            elif _explicitly_null(rk, "vel_range_per_clip"):
                # A banked/solved run has the velocity SOLVED, so this box is dead and
                # _assert_solved_target_recipe_is_coherent refuses to launch while it is set. Its
                # own error says "set it to None" — writing ``vel_range_per_clip: null`` is how a
                # run says that, and without this branch the instruction is unfollowable because
                # the box comes from a shipped cfg default, not from the yaml.
                _require(hasattr(C, "racket_vel_range_per_clip"), "racket_target.racket_vel_range_per_clip")
                C.racket_vel_range_per_clip = None
                applied.append("racket_target.racket_vel_range_per_clip=None (explicit null)")
            _set_attr(C, "allow_non_forward_target_velocity",
                      _get(rk, "allow_non_forward_target_velocity"), _as_bool, applied, "racket_target")
            # Optional PER-CLIP position boxes (unified policy): forehand=clip 0, backhand=clip 1. Absent ->
            # keep the shared pos_*_range + |y|-sign box above (backward compatible). Lets each clip's target
            # track its own reference strike point (e.g. backhand z~1.2 at strike_phase 0.50).
            _ppc = _resolve_pos_range_per_clip(rk, _clip_names)
            if _ppc is not None:
                _require(hasattr(C, "racket_pos_range_per_clip"), "racket_target.racket_pos_range_per_clip")
                C.racket_pos_range_per_clip = _ppc
                applied.append(f"racket_target.racket_pos_range_per_clip={_ppc}")
            # PER-CLIP INCOMING-BALL regime (TASK B): a block gets fast balls, a loop slow ones, in
            # the SAME run. Absent -> the shared vb_vel_*_range box, byte-identical.
            _vbpc = _resolve_vb_vel_range_per_clip(rk, _clip_names)
            if _vbpc is not None:
                _require(hasattr(C, "vb_vel_range_per_clip"), "racket_target.vb_vel_range_per_clip")
                C.vb_vel_range_per_clip = _vbpc
                applied.append(f"racket_target.vb_vel_range_per_clip={_vbpc}")
            _vbspc = _resolve_vb_spin_abs_max_per_clip(rk, _clip_names)
            if _vbspc is not None:
                _require(hasattr(C, "vb_spin_abs_max_per_clip"),
                         "racket_target.vb_spin_abs_max_per_clip")
                C.vb_spin_abs_max_per_clip = _vbspc
                applied.append(f"racket_target.vb_spin_abs_max_per_clip={_vbspc}")
            _set_range(C, "base_target_x_range", _get(rk, "base_target_x_range"), applied, "racket_target")
            _set_range(C, "base_target_y_range", _get(rk, "base_target_y_range"), applied, "racket_target")
            # weak base->racket coupling (uniform mode): fraction of the racket Y offset + clamp (meters)
            _set_attr(C, "base_couple_blend", _get(rk, "base_couple_blend"), float, applied, "racket_target")
            _set_attr(C, "base_couple_max_offset", _get(rk, "base_couple_max_offset"), float, applied, "racket_target")
            # HITTER base-station derivation: "blend" (legacy) | "reference_reach" (base = racket − ref reach)
            _set_attr(C, "base_couple_mode", _get(rk, "base_couple_mode"), str, applied, "racket_target")
            _set_attr(C, "normal_mode", _get(rk, "normal_mode"), str, applied, "racket_target")
            _set_attr(C, "forehand_on_negative_y", _get(rk, "forehand_on_negative_y"), _as_bool, applied, "racket_target")
            _set_attr(C, "mount_normal_axis", _get(rk, "mount_normal_axis"), int, applied, "racket_target")
            _set_attr(C, "mount_normal_sign", _get(rk, "mount_normal_sign"), float, applied, "racket_target")
            # 每 clip 击球面符号(正手一面、反手另一面,franco"哪面超前就是哪面";顺序 = motion_file
            # clip 顺序,同 strike_phase_per_clip)。缺省/空 -> 用上面的标量符号,现役行为逐位不变;
            # 表长和 clip 数不匹配由 RacketTargetCommand._mount_signs_cfg 在环境侧当场报错。
            _mnspc = _get(rk, "mount_normal_sign_per_clip")
            if _mnspc is not None:
                C.mount_normal_sign_per_clip = tuple(float(x) for x in _mnspc)
                applied.append(f"racket_target.mount_normal_sign_per_clip={C.mount_normal_sign_per_clip}")
            # reference_perturbed target sampling (rank 5): couple targets to the reference swing.
            _set_attr(C, "target_mode", _get(rk, "target_mode"), str, applied, "racket_target")
            _set_attr(
                C,
                "task_first_manifest_path",
                _get(rk, "task_first_manifest_path"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "task_first_manifest_sha256",
                _get(rk, "task_first_manifest_sha256"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "task_first_base_success_thresh_m",
                _get(rk, "task_first_base_success_thresh_m"),
                float,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_manifest_path",
                _get(rk, "action_ball_manifest_path"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_manifest_sha256",
                _get(rk, "action_ball_manifest_sha256"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_policy_contract_sha256",
                _get(rk, "action_ball_policy_contract_sha256"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_diagnostic_unauthorized",
                _get(rk, "action_ball_diagnostic_unauthorized"),
                lambda value: _as_explicit_bool(
                    value,
                    "task.racket.action_ball_diagnostic_unauthorized",
                ),
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "reference_guard_mode",
                _get(rk, "reference_guard_mode"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "virtual_ball",
                _get(rk, "virtual_ball"),
                lambda value: _as_explicit_bool(
                    value,
                    "task.racket.virtual_ball",
                ),
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_evaluator_launch_receipt_path",
                _get(rk, "action_ball_evaluator_launch_receipt_path"),
                str,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_evaluator_launch_receipt_file_sha256",
                _get(
                    rk,
                    "action_ball_evaluator_launch_receipt_file_sha256",
                ),
                str,
                applied,
                "racket_target",
            )
            for _field in (
                "action_ball_sidecar_launch_receipt_path",
                "action_ball_sidecar_launch_receipt_file_sha256",
                "action_ball_drain_reset_launch_receipt_path",
                "action_ball_drain_reset_launch_receipt_file_sha256",
                "action_ball_evaluation_inbox_root",
                "action_ball_evaluation_owner_id",
                "action_ball_evaluation_run_id",
            ):
                _set_attr(
                    C,
                    _field,
                    _get(rk, _field),
                    str,
                    applied,
                    "racket_target",
                )
            _set_attr(
                C,
                "action_ball_frozen_eval_interval_updates",
                _get(rk, "action_ball_frozen_eval_interval_updates"),
                lambda value: _as_exact_int(
                    value,
                    "task.racket.action_ball_frozen_eval_interval_updates",
                ),
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_seed",
                _get(rk, "action_ball_seed"),
                lambda value: _as_exact_int(
                    value, "task.racket.action_ball_seed"
                ),
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_pool_refill_rows",
                _get(rk, "action_ball_pool_refill_rows"),
                lambda value: _as_exact_int(
                    value, "task.racket.action_ball_pool_refill_rows"
                ),
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "action_ball_fixed_direction",
                _get(rk, "action_ball_fixed_direction"),
                lambda value: _as_explicit_bool(
                    value, "task.racket.action_ball_fixed_direction"
                ),
                applied,
                "racket_target",
            )
            _set_vec3(C, "ref_perturb_pos", _get(rk, "ref_perturb_pos"), applied, "racket_target")
            _set_vec3(C, "ref_perturb_vel", _get(rk, "ref_perturb_vel"), applied, "racket_target")
            _set_attr(C, "ref_perturb_normal", _get(rk, "ref_perturb_normal"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_steps", _get(rk, "ref_perturb_curriculum_steps"), int, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_start", _get(rk, "ref_perturb_curriculum_start"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_success_gated", _get(rk, "ref_perturb_success_gated"), _as_bool, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_threshold", _get(rk, "ref_perturb_advance_threshold"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_rate", _get(rk, "ref_perturb_advance_rate"), float, applied, "racket_target")
            # Stage slow->fast hitting: scale the reference racket-velocity target (<1.0 trains slower hits).
            # PER-CLIP: ref_vel_scale_by_motion wins for the trained clip, else the scalar ref_vel_scale.
            _rv_val, _rv_note = _resolve_ref_vel_scale(rk, clip_name)
            _set_attr(C, "ref_vel_scale", _rv_val, float, applied, "racket_target")
            if _rv_note is not None:
                applied.append(_rv_note)
            # Debug logging (sign verification + raw/gated reward kernels). Off for production runs.
            _set_attr(C, "debug_reward_logging", _get(rk, "debug_reward_logging"), _as_bool, applied, "racket_target")
            # Clean reference strike velocity (denoise the FD'd target velocity at the racket tip).
            _set_attr(C, "clean_reference_strike_velocity", _get(rk, "clean_reference_strike_velocity"),
                      _as_bool, applied, "racket_target")
            _set_attr(C, "clean_strike_vel_window", _get(rk, "clean_strike_vel_window"), int, applied, "racket_target")
            # HER-style achieved-target replay: with prob achieved_target_mix_prob the next swing's target
            # is a jittered previously-ACHIEVED strike state (per-clip ring buffer) instead of a box sample.
            _set_attr(C, "achieved_target_mix_prob", _get(rk, "achieved_target_mix_prob"), float, applied, "racket_target")
            _set_attr(C, "achieved_buffer_size", _get(rk, "achieved_buffer_size"), int, applied, "racket_target")
            _set_attr(C, "achieved_min_fill", _get(rk, "achieved_min_fill"), int, applied, "racket_target")
            _set_attr(C, "achieved_jitter_pos", _get(rk, "achieved_jitter_pos"), float, applied, "racket_target")
            _set_attr(C, "achieved_jitter_vel", _get(rk, "achieved_jitter_vel"), float, applied, "racket_target")
            _set_attr(C, "achieved_clamp_inflate", _get(rk, "achieved_clamp_inflate"), float, applied, "racket_target")
            # A1 target latency & time-variance: the ACTOR-visible target arrives late
            # (target_delay_steps), noisy (SMASH-style tts-decaying jitter), and is refined
            # mid-swing (midswing_resample_*), matching the real mocap->planner->runner loop.
            # Rewards/critic keep the live target. All default OFF (byte-identical baseline).
            _set_attr(C, "target_delay_steps", _get(rk, "target_delay_steps"), int, applied, "racket_target")
            _tts_mode_requested = _get(rk, "target_delay_tts_mode")
            _set_attr(
                C,
                "target_delay_tts_mode",
                _tts_mode_requested,
                _as_target_delay_tts_mode,
                applied,
                "racket_target",
            )
            if _tts_mode_requested is not None and C.target_delay_tts_mode != "live":
                # The shipped observation cfg intentionally shares the live TTS callable between
                # policy and critic.  Only the explicit atomic-tuple arm swaps the POLICY source;
                # the critic keeps mdp.time_to_strike and rewards/gates read cmd.time_to_strike.
                from whole_body_tracking.tasks.tracking import mdp as _mdp

                _require(
                    hasattr(env_cfg, "observations")
                    and hasattr(env_cfg.observations, "policy")
                    and getattr(env_cfg.observations.policy, "time_to_strike", None) is not None,
                    "observations.policy.time_to_strike (task.racket.target_delay_tts_mode)",
                )
                env_cfg.observations.policy.time_to_strike.func = _mdp.actor_time_to_strike
                applied.append(
                    "observations.policy.time_to_strike.func=actor_time_to_strike "
                    f"(atomic planner tuple mode={C.target_delay_tts_mode}; critic remains live)"
                )
            _set_attr(C, "target_jitter_pos_per_s", _get(rk, "target_jitter_pos_per_s"), float, applied, "racket_target")
            _set_attr(C, "target_jitter_vel_per_s", _get(rk, "target_jitter_vel_per_s"), float, applied, "racket_target")
            _set_attr(C, "midswing_resample_prob", _get(rk, "midswing_resample_prob"), float, applied, "racket_target")
            _set_attr(C, "midswing_resample_tts_floor", _get(rk, "midswing_resample_tts_floor"), float, applied, "racket_target")
            # A1v2 calibrated mocap-degradation channels — same actor-only scope as the delay/jitter
            # group above (venue fits documented in the task YAML: white 0.0019, ar1 0.0052).
            _set_attr(C, "target_noise_white", _get(rk, "target_noise_white"), float, applied, "racket_target")
            _set_attr(C, "target_noise_ar1_sigma", _get(rk, "target_noise_ar1_sigma"), float, applied, "racket_target")
            _set_attr(C, "target_noise_ar1_rho", _get(rk, "target_noise_ar1_rho"), float, applied, "racket_target")
            _set_attr(C, "target_dropout_prob", _get(rk, "target_dropout_prob"), float, applied, "racket_target")
            _set_attr(C, "target_post_strike_dropout_s", _get(rk, "target_post_strike_dropout_s"), float, applied, "racket_target")
            _set_attr(C, "target_bias_per_swing", _get(rk, "target_bias_per_swing"), float, applied, "racket_target")
            # Tier-1 virtual ball: incoming-ball sampling boxes + outgoing-spin objective. The reward
            # side reads vb_spin_mode with a default-else branch, so an unknown mode would silently
            # train topspin — validate the value here instead.
            _set_attr(C, "vb_spin_mode", _get(rk, "vb_spin_mode"), str, applied, "racket_target")
            if getattr(C, "vb_spin_mode", "topspin") not in ("topspin", "minimize"):
                raise _OverrideError(
                    f"[train.py] racket.vb_spin_mode must be 'topspin' or 'minimize', "
                    f"got {C.vb_spin_mode!r}")
            _set_attr(C, "vb_spin_min_sigma", _get(rk, "vb_spin_min_sigma"), float, applied, "racket_target")
            _set_attr(C, "vb_spin_abs_max", _get(rk, "vb_spin_abs_max"), float, applied, "racket_target")
            _set_range(C, "vb_vel_x_range", _get(rk, "vb_vel_x_range"), applied, "racket_target")
            _set_range(C, "vb_vel_y_range", _get(rk, "vb_vel_y_range"), applied, "racket_target")
            _set_range(C, "vb_vel_z_range", _get(rk, "vb_vel_z_range"), applied, "racket_target")
            # Metrics-only virtual ball (franco 2026-07-06): in-training 上台率/击球率 curves on
            # tasks whose rewards have no virtual_* terms (DeployParity/Hitter). Metrics only.
            _set_attr(C, "vb_metrics_only", _get(rk, "vb_metrics_only"), _as_bool, applied, "racket_target")
            # metric-sync fix (2026-07-09): transition-period *_legacy rally curves on/off.
            # 人话:旧算法上台率对照曲线还要不要发(新算法曲线不受此开关影响)。
            _set_attr(C, "rally_legacy_metrics", _get(rk, "rally_legacy_metrics"), _as_bool, applied, "racket_target")

            # Stage-1 question bank + face-command channel (defaults OFF). question_bank = bank npz
            # path (gen_stage1_questions.py); face_command re-anchors the racket_normal reward onto
            # the demanded normal (target_normal_cmd).
            _set_attr(C, "question_bank", _get(rk, "question_bank"), str, applied, "racket_target")
            # --- CONTINUOUS questions (racket.target_mode: solved) ---------------------------
            _set_attr(C, "cq_anchor_bank", _get(rk, "cq_anchor_bank"), str, applied, "racket_target")
            _set_attr(C, "exam_bank", _get(rk, "exam_bank"), str, applied, "racket_target")
            _set_attr(C, "cq_spin_abs_max", _get(rk, "cq_spin_abs_max"), float, applied, "racket_target")
            _set_attr(C, "cq_buffer_rows", _get(rk, "cq_buffer_rows"), int, applied, "racket_target")
            _set_attr(C, "cq_overdraw", _get(rk, "cq_overdraw"), float, applied, "racket_target")
            _set_attr(C, "cq_n_iters", _get(rk, "cq_n_iters"), int, applied, "racket_target")
            _set_attr(C, "cq_tol_m", _get(rk, "cq_tol_m"), float, applied, "racket_target")
            _set_attr(C, "cq_speed_budget", _get(rk, "cq_speed_budget"), float, applied, "racket_target")
            _set_attr(C, "cq_max_redraw_rounds", _get(rk, "cq_max_redraw_rounds"), int, applied, "racket_target")
            _set_attr(C, "cq_max_face_deg", _get(rk, "cq_max_face_deg"), float, applied, "racket_target")
            _set_attr(C, "cq_exam_holdout", _get(rk, "cq_exam_holdout"), _as_bool, applied, "racket_target")
            _set_attr(C, "cq_seed", _get(rk, "cq_seed"), int, applied, "racket_target")
            _set_attr(C, "cq_accept_buckets", _get(rk, "cq_accept_buckets"), int, applied, "racket_target")
            _set_attr(C, "cq_max_exhausted_frac", _get(rk, "cq_max_exhausted_frac"), float, applied, "racket_target")
            _set_attr(C, "cq_abort_exhausted_frac", _get(rk, "cq_abort_exhausted_frac"), float, applied, "racket_target")
            _set_attr(C, "cq_min_accept_rate", _get(rk, "cq_min_accept_rate"), float, applied, "racket_target")
            _set_attr(C, "cq_closed_loop_rows", _get(rk, "cq_closed_loop_rows"), int, applied, "racket_target")
            _set_attr(C, "cq_closed_loop_max_err_m", _get(rk, "cq_closed_loop_max_err_m"), float,
                      applied, "racket_target")
            _cqv = _get(rk, "cq_vel_range_per_clip")
            if _cqv is not None:
                # 每 clip 一行 ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi)) —— 这就是这条臂**声明的**
                # 来球分布,没有共享兜底(静默兜底会让每条臂的真任务和它的 yaml 不是一回事)。
                C.cq_vel_range_per_clip = tuple(
                    tuple((float(lo), float(hi)) for (lo, hi) in clip_rng) for clip_rng in _cqv)
                applied.append(f"racket_target.cq_vel_range_per_clip={C.cq_vel_range_per_clip}")
            _cqa = _get(rk, "cq_aim_xy")
            if _cqa is not None:
                C.cq_aim_xy = tuple(float(v) for v in _cqa)
                applied.append(f"racket_target.cq_aim_xy={C.cq_aim_xy}")
            _set_attr(C, "question_bank_allow_legacy", _get(rk, "question_bank_allow_legacy"),
                      _as_bool, applied, "racket_target")
            _set_attr(C, "face_command", _get(rk, "face_command"), _as_bool, applied, "racket_target")
            _set_attr(
                C,
                "face_command_pairing",
                _get(rk, "face_command_pairing"),
                _as_face_command_pairing,
                applied,
                "racket_target",
            )
            if bool(getattr(C, "face_command", False)):
                pairing = _as_face_command_pairing(
                    getattr(C, "face_command_pairing", "shared_plus_y")
                )
                if pairing == "shared_plus_y":
                    applied.append(
                        "[face] face_command kernel frame=+Y(A/bank); "
                        "mount sign applies to metric/ref channels only"
                    )
                else:
                    applied.append(
                        "[diagnostic] face_command pairing=legacy_signed_vs_A; "
                        "signed measured normal is graded against the A-frame bank target"
                    )
            # The question-bank answer is an ABSOLUTE physics target.  This remains compatible with
            # motion retiming because _apply_question_bank_targets runs after generic target sampling
            # and overwrites the speed-scaled provisional velocity with the unscaled bank answer.
            # Keep only the real semantic conflicts: HitterPure target ownership and mid-swing bank
            # row redraw without rescheduling the paired incoming ball.
            if str(getattr(C, "question_bank", "") or ""):
                _target_mode = str(getattr(C, "target_mode", "uniform"))
                if _target_mode == "hitter_pure":
                    raise _OverrideError(
                        "[train.py] racket.question_bank is incompatible with "
                        "racket.target_mode=hitter_pure: the bank owns a fixed contact point and "
                        "atomic incoming-ball/answer row, while HitterPure owns station-relative "
                        "target sampling. Use uniform/reference_perturbed or drop the bank.")
                _ms_prob = float(getattr(C, "midswing_resample_prob", 0.0))
                if _ms_prob > 0.0:
                    raise _OverrideError(
                        "[train.py] racket.question_bank is incompatible with "
                        f"racket.midswing_resample_prob={_ms_prob}: a redraw would change the "
                        "question without rescheduling the same physical/shadow ball. Set it to 0.")
            # Same rule, keyed on "is the target SOLVED" rather than "is there an npz path" — a
            # continuous arm re-enters the same solve seam mid-swing and would swap the question
            # while the ball is still flying to the old one.
            if str(getattr(C, "target_mode", "")) == "solved":
                _ms_prob = float(getattr(C, "midswing_resample_prob", 0.0))
                if _ms_prob > 0.0:
                    raise _OverrideError(
                        "[train.py] racket.target_mode=solved (continuous questions) is "
                        f"incompatible with racket.midswing_resample_prob={_ms_prob}: the "
                        "mid-swing path re-enters the solve seam but never reschedules the "
                        "physical/shadow ball. Set it to 0.")
                if str(getattr(C, "question_bank", "") or "").strip():
                    raise _OverrideError(
                        "[train.py] racket.target_mode=solved produces the target from a "
                        "continuous draw; racket.question_bank must be empty. To keep a bank as "
                        "the validated CONTRACT ANCHOR (never trained on) use "
                        "racket.cq_anchor_bank.")
                # 时间律,提前到解析期。人话:同一条规则本来只有第一次 env reset 才炸 —— 也就是
                # 整个 Isaac Sim 起完(实测 3 分 09 秒)才告诉你 yaml 写错了一行。规则不变,只是
                # 让它在还没花钱之前就响。环境侧那条留着,它是最后一道。
                _M = env_cfg.commands.motion
                _ssr = tuple(float(x) for x in (getattr(_M, "speed_scale_range", (1.0, 1.0))
                                                or (1.0, 1.0)))
                if _ssr != (1.0, 1.0):
                    raise _OverrideError(
                        f"[train.py] racket.target_mode=solved with motion.speed_scale_range="
                        f"{_ssr}: the retiming scale no longer shapes the TARGET (it is solved at "
                        f"the ball) but it still divides the actor's time-to-strike, so a "
                        f"non-unity range is a silent inconsistency between command and clock. "
                        f"Set task.motion.speed_scale_range=[1.0, 1.0]. NOTE this is also why a "
                        f"continuous-vs-bank A/B cannot be run at matched speed_scale_range: the "
                        f"live bank arms run [0.6, 1.0] only because the same guard is dead for "
                        f"them (MotionLoader carries no cfg).")
                _sspc = getattr(_M, "speed_scale_per_clip", None)
                if _sspc is not None and any(float(s) != 1.0 for s in _sspc):
                    raise _OverrideError(
                        f"[train.py] racket.target_mode=solved with motion.speed_scale_per_clip="
                        f"{tuple(float(s) for s in _sspc)} — same inconsistency as "
                        f"speed_scale_range; set every entry to 1.0.")
            # face_command_obs (+4 actor dims: demanded normal (3) + zero-filled rho placeholder (1),
            # the contract-day 175 -> 179 layout): the obs groups were finalized in __post_init__
            # BEFORE overrides run, so setting env_cfg.face_command_obs here would be a silent
            # no-op — attach the ObsTerm directly (same term/tail position as the cfg switch).
            # The enabling experiment must update/remove actor_obs_contract in its YAML:
            # validate_actor_observation_contract stays a loud error on the frozen 175-D value.
            _fc_obs = _get(rk, "face_command_obs")
            if _fc_obs is not None and _as_bool(_fc_obs):
                if str(getattr(C, "target_mode", "")) in (
                    "task_first",
                    "action_ball",
                ):
                    # Formal arbitrary-N modes append face(+4) and action(+N)
                    # atomically onto the native Hitter-footwork prefix. Defer
                    # both terms so a pre-attached/wrongly ordered tail is
                    # rejected rather than silently reused.
                    if hasattr(env_cfg, "face_command_obs"):
                        env_cfg.face_command_obs = True
                    applied.append(
                        "racket.face_command_obs=True "
                        "(formal face/action tail deferred for atomic append)"
                    )
                else:
                    # 尾部顺序守卫:站位通道必须在拍面通道之后(179 前缀不变才有纯尾部扩列热启)。
                    # 若站位已先挂上(如 cfg 旗标开了 station_obs、拍面却走 YAML 覆盖),此时再挂
                    # 拍面会得到 175+站位2+拍面4 的错序布局——loud error,别让它静默开训。
                    if getattr(env_cfg.observations.policy, "station_anchor_err_b", None) is not None:
                        raise _OverrideError(
                            "[train.py] racket.face_command_obs enabling AFTER station_anchor_err_b is "
                            "already attached would put the station channel BEFORE the face channel "
                            "(layout != 179+station tail). Enable both via the same path (racket.station_obs "
                            "+ racket.face_command_obs in the task YAML/CLI), not mixed cfg-flag/override.")
                    from isaaclab.managers import ObservationTermCfg as _ObsTerm

                    from whole_body_tracking.tasks.tracking import mdp as _mdp

                    env_cfg.observations.policy.racket_target_normal_cmd = _ObsTerm(
                        func=_mdp.racket_target_normal_cmd, params={"command_name": "racket_target"})
                    if hasattr(env_cfg, "face_command_obs"):
                        env_cfg.face_command_obs = True  # keep the descriptive cfg field honest
                    applied.append(
                        "observations.policy.racket_target_normal_cmd(+4D face-command obs, 175->179)")
            # R10c station_obs (+2 actor 尾维:世界系站位锚误差 = 出生点常数 − 当前 base XY,旋进
            # base 系;franco 2026-07-09"就算不需要移动,它也是一个锚")。同 face_command_obs 时序:
            # __post_init__ 已跑完,这里直接挂 ObsTerm(同名同尾部位置)。要求拍面通道已开——
            # 单开=177 维且与 Hitter 177(站位在第 167 列)布局不同,评估器按维数认契约会静默
            # 错位;R10c 的合法形状只有 181。锚点可用 racket.station_anchor_offset_xy 挪离出生点。
            _st_obs = _get(rk, "station_obs")
            if _st_obs is not None and _as_bool(_st_obs):
                if getattr(env_cfg.observations.policy, "racket_target_normal_cmd", None) is None:
                    raise _OverrideError(
                        "[train.py] racket.station_obs=true requires the face channel already attached "
                        "(racket.face_command_obs=true or env cfg face_command_obs): station alone would "
                        "be an ambiguous 177-D layout (Hitter's 177 has the station at column 167, not "
                        "the tail) — the evaluator resolves contracts by dim and would silently misread. "
                        "R10c's only legal shape is 181 = 179 + station tail.")
                from isaaclab.managers import ObservationTermCfg as _ObsTerm

                from whole_body_tracking.tasks.tracking import mdp as _mdp

                env_cfg.observations.policy.station_anchor_err_b = _ObsTerm(
                    func=_mdp.station_anchor_err_b, params={"command_name": "racket_target"})
                if hasattr(env_cfg, "station_obs"):
                    env_cfg.station_obs = True  # keep the descriptive cfg field honest
                applied.append(
                    "observations.policy.station_anchor_err_b(+2D station-anchor obs, 179->181)")
            _set_range(C, "station_anchor_offset_xy", _get(rk, "station_anchor_offset_xy"),
                       applied, "racket_target")
            # SHADOW physical ball + table (METRICS-ONLY): a real PhysX ball flies each question
            # in, is struck via the same venue contact model, and lands under engine integration —
            # an online engine-vs-analytic cross-check of the vb landing prediction. The scene
            # entities must be attached HERE because __post_init__ already ran before overrides
            # (the exact face_command_obs timing problem above); attach_shadow_ball_scene is
            # idempotent so cfg-flag and YAML/CLI paths compose. Requires virtual_ball=True
            # (RacketTargetCommand.__init__ raises loudly otherwise).
            _set_attr(C, "shadow_ball", _get(rk, "shadow_ball"), _as_bool, applied, "racket_target")
            _set_attr(C, "shadow_table", _get(rk, "shadow_table"), _as_bool, applied, "racket_target")
            if getattr(C, "shadow_table", False) and not getattr(C, "shadow_ball", False):
                raise _OverrideError(
                    "[train.py] racket.shadow_table=true requires racket.shadow_ball=true "
                    "(the table exists only for the shadow ball to land on).")
            if getattr(C, "shadow_ball", False):
                from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
                    attach_shadow_ball_scene as _attach_shadow,
                )

                _attach_shadow(env_cfg, shadow_table=bool(getattr(C, "shadow_table", False)))
                applied.append(
                    f"scene.shadow_ball attached (metrics-only; table={bool(C.shadow_table)})")

    # PHYSICAL ball + table truth instrument (Phase A) — TOP-LEVEL task key (task.physical_ball),
    # mirroring the env-cfg field HOPEPingPongAgibotA3EnvCfg.physical_ball. Each swing's question
    # incoming ball is realized physically (reverse-integrated venue launch, aero-wrench flight,
    # CODE-DRIVEN fitted table bounce, robot pass-through — racket impulse = Phase B); METRICS-ONLY,
    # rewards/obs untouched. __post_init__ already ran before overrides (the face_command_obs
    # timing), so the scene must be attached HERE; attach_physical_ball_scene is idempotent so the
    # cfg-flag and YAML/CLI paths compose. Requires the virtual-ball task variant
    # (RacketTargetCommand.__init__ raises loudly otherwise). Consumed in this same commit
    # (018467a whitelist rule): this block is the translation; there is no top-level unknown-key
    # scan, so this comment is the whitelist.
    _pb = _get(task, "physical_ball")
    if _pb is not None and _as_bool(_pb):
        from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
            attach_physical_ball_scene as _attach_physical,
        )

        _require(hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "racket_target"),
                 "commands.racket_target (task.physical_ball)")
        env_cfg.commands.racket_target.physical_ball = True
        if hasattr(env_cfg, "physical_ball"):
            env_cfg.physical_ball = True  # keep the descriptive env-cfg field honest
        _attach_physical(env_cfg)
        applied.append("scene.pb_ball+pb_table attached (Phase A truth instrument; metrics-only)")

    # TABLE OBSTACLE — TOP-LEVEL task key (task.table_obstacle), mirroring the env-cfg field
    # HOPEPingPongAgibotA3EnvCfg.table_obstacle. DEFAULT ON in the cfg, so this key exists to turn
    # it OFF for a deliberate no-table ablation (and to make "this arm trained without a table"
    # a thing the applied log states rather than something you infer from the date).
    # 人话:桌子默认在。这个键是给"故意不要桌子"的对照臂用的。
    # __post_init__ already ran, so flipping the field here is not enough — apply_table_obstacle
    # re-runs the whole install/remove (scene collider + termination + penalty) and is idempotent,
    # so the cfg-flag and YAML/CLI paths compose. Consumed in this same block; there is no
    # top-level unknown-key scan, so this comment is the whitelist (task.physical_ball precedent).
    _tob = _get(task, "table_obstacle")
    if _tob is not None:
        from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
            apply_table_obstacle as _apply_table,
        )

        _require(hasattr(env_cfg, "table_obstacle"),
                 "env_cfg.table_obstacle (task.table_obstacle)")
        env_cfg.table_obstacle = _as_bool(_tob)
        _apply_table(env_cfg)
        applied.append(
            f"task.table_obstacle={env_cfg.table_obstacle} "
            f"(collider={env_cfg.table_obstacle_prim or 'NONE'}; "
            f"terminations.robot_hit_table="
            f"{'on' if getattr(env_cfg.terminations, 'robot_hit_table', None) is not None else 'off'})"
        )

    # R-a actor leg-reference masking (reward_staged_design 2026-07-08 §⑥; HITTER critic-only
    # reference structure) — TOP-LEVEL task key (task.actor_leg_ref_mask), mirroring the
    # task.physical_ball precedent (this comment is the whitelist). The ACTOR's 62-D motion
    # command keeps its exact layout but the 24 leg dims (12 leg-joint pos + 12 leg-joint vel)
    # are fed the DEFAULT STAND pose + zero velocity; the critic's command term is untouched
    # (privileged). Obs dim unchanged = zero contract cost. 人话:actor 眼里腿参考=站姿常数,
    # critic 照旧全看。The leg-joint indices are derived at RUNTIME via robot.find_joints and
    # printed to the launch log (never hardcoded — a wrong index table would be a
    # policy-killing experiment); see mdp.generated_commands_actor_leg_masked.
    _alm = _get(task, "actor_leg_ref_mask")
    if _alm is not None and _as_bool(_alm):
        from whole_body_tracking.tasks.tracking import mdp as _mdp

        _require(
            hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy")
            and getattr(env_cfg.observations.policy, "command", None) is not None,
            "observations.policy.command (task.actor_leg_ref_mask)")
        env_cfg.observations.policy.command.func = _mdp.generated_commands_actor_leg_masked
        applied.append("observations.policy.command.func=generated_commands_actor_leg_masked "
                       "(R-a: actor leg ref dims -> default stand + zero vel; critic untouched)")

    # R-b envelope-termination softening (reward_staged_design 2026-07-08 §⑥ + R-b细则): the two
    # tracking-ENVELOPE terminations (anchor_pos / ee_body_pos, both z>0.25 m vs the reference)
    # stop ENDING the episode and become a per-step penalty (rewards.tracking_envelope, weight =
    # terminations.envelope_penalty_weight, default -1.0 => -1.0/s => -0.02/step @50 Hz). The
    # ABSOLUTE terminations (base_fell_tilt 0.7 rad / base_too_low 0.5 m) and anchor_ori stay.
    # 人话:跟丢参考不再判死,改成站在违规区里每秒扣钱;真摔倒照样判死。
    # Accounting migration (design's一票否决项): terminated now only fires on the absolute terms,
    # so pre/post_strike_fall_rate narrow to REAL falls automatically; the envelope violations get
    # their own counters — tracking_loss_rate (rising edges / swing starts, per-clip too) and
    # envelope_violated_frac — enabled via racket_target.track_envelope_violation. Cross-arm
    # comparison: old-arm falls ≈ new-arm (falls + tracking_loss).
    tm = _get(task, "terminations")
    _check_unknown_keys(tm, _TERMINATION_KEYS, "task.terminations")
    if tm is not None:
        _eap = _get(tm, "envelope_as_penalty")
        _epw = _get(tm, "envelope_penalty_weight")
        if _eap is not None and _as_bool(_eap):
            T = env_cfg.terminations
            for _tn in ("anchor_pos", "ee_body_pos"):
                _require(hasattr(T, _tn), f"terminations.{_tn}")
                setattr(T, _tn, None)  # configclass None = term removed (footwork-cfg precedent)
            _w = -1.0 if _epw is None else float(_epw)
            if _w >= 0.0:
                raise _OverrideError(
                    f"[train.py] terminations.envelope_penalty_weight must be < 0 (per-step "
                    f"penalty replacing the removed terminations), got {_w}")
            _require(hasattr(env_cfg.rewards, "tracking_envelope"), "rewards.tracking_envelope")
            env_cfg.rewards.tracking_envelope.weight = _w
            env_cfg.rewards.tracking_envelope.params["ignore_hold"] = True
            _require(hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "racket_target")
                     and hasattr(env_cfg.commands.racket_target, "track_envelope_violation"),
                     "commands.racket_target.track_envelope_violation (envelope accounting)")
            env_cfg.commands.racket_target.track_envelope_violation = True
            applied.append("terminations.anchor_pos=None terminations.ee_body_pos=None "
                           "(envelope_as_penalty: envelope no longer terminates)")
            applied.append(f"rewards.tracking_envelope.weight={_w} "
                           "ignore_hold=True (+tracking_loss_rate/envelope_violated_frac accounting)")
        elif _epw is not None:
            raise _OverrideError(
                "[train.py] terminations.envelope_penalty_weight is set but envelope_as_penalty "
                "is not enabled — it would be silently ignored. Set "
                "terminations.envelope_as_penalty=true or drop the weight.")

        # R9 lower-body-free ablation (franco 拍板 2026-07-08): 蹲深该由任务决定,不该抄舞谱。
        # The imitation reward already only watches the upper body; these two flags cut the last
        # two LOWER-BODY reference leashes at the TERMINATION layer (删缰绳, not a penalty swap —
        # that would be envelope_as_penalty's business and the two are mutually exclusive).
        # The ABSOLUTE safety terms (base_fell_tilt 0.7 rad / base_too_low 0.5 m) and anchor_ori
        # are untouched: a real fall/sink still ends the episode.
        _eap_on = _eap is not None and _as_bool(_eap)
        # ① terminations.anchor_pos_off — REMOVE the torso-z leash termination outright
        #    (anchor_pos = bad_anchor_pos_z_only, |z - ref z| > 0.25 m judged死).
        #    人话:躯干高度不再跟舞谱对表,想蹲多深蹲多深;真摔倒/坐地照样判死。
        _apo = _get(tm, "anchor_pos_off")
        if _apo is not None and _as_bool(_apo):
            if _eap_on:
                raise _OverrideError(
                    "[train.py] terminations.anchor_pos_off conflicts with envelope_as_penalty: "
                    "the penalty swap keeps charging for the very anchor-z deviation this switch "
                    "frees. Enable one or the other, not both.")
            T = env_cfg.terminations
            _require(hasattr(T, "anchor_pos"), "terminations.anchor_pos")
            T.anchor_pos = None  # configclass None = term removed (envelope_as_penalty precedent)
            applied.append("terminations.anchor_pos=None (anchor_pos_off: torso-z reference leash "
                           "removed; base_fell_tilt/base_too_low/anchor_ori absolutes stay)")
        # ② terminations.ee_upper_only — bad_motion_body_pos_z_only keeps judging the WRISTS
        #    against the reference z (挥拍 execution 还有锚) but drops the ANKLES from its body
        #    list (脚自由:步法/下蹲不再因为脚离开舞谱高度带被判死).
        _euo = _get(tm, "ee_upper_only")
        if _euo is not None and _as_bool(_euo):
            if _eap_on:
                raise _OverrideError(
                    "[train.py] terminations.ee_upper_only conflicts with envelope_as_penalty: "
                    "envelope_as_penalty already removed the ee_body_pos termination this flag "
                    "narrows. Enable one or the other, not both.")
            T = env_cfg.terminations
            _term = getattr(T, "ee_body_pos", None)
            _require(_term is not None, "terminations.ee_body_pos")
            _params = getattr(_term, "params", None)
            _require(isinstance(_params, dict)
                     and isinstance(_params.get("body_names"), (list, tuple)),
                     "terminations.ee_body_pos.params['body_names'] (explicit body list)")
            _names = [str(n) for n in _params["body_names"]]
            _kept = [n for n in _names if "wrist" in n.lower()]
            _dropped = [n for n in _names if "wrist" not in n.lower()]
            if not _kept or any("ankle" not in n.lower() for n in _dropped):
                raise _OverrideError(
                    f"[train.py] terminations.ee_upper_only expects the ee_body_pos body list to "
                    f"be wrists+ankles, got {_names} — refusing to guess which bodies to free.")
            _params["body_names"] = _kept
            applied.append(f"terminations.ee_body_pos.body_names={_kept} "
                           f"(ee_upper_only: ankles freed, wrist z still tracks the reference)")

    # Domain randomization: behaviour preserved exactly unless an N=1
    # diagnostic launch explicitly selects the stable-ready plant.  This is a
    # learning prerequisite, not a robustness claim: the current 4096-env
    # probes show a shared waist-roll raw-hard failure before PPO while the
    # nominal hold and both teacher clips have ample limit margin.  Keep the
    # already recipe-bound ±0.01 joint-default offset and historical material
    # randomization, but remove the three plant axes that directly change the
    # weak waist equilibrium (torso CoM, link mass and PD gains).
    dr = _get(task, "domain_rand")
    if dr is not None and hasattr(env_cfg, "events"):
        E = env_cfg.events
        mr = _get(dr, "link_mass_range")
        pr = _get(dr, "pd_gain_range")
        stable_ready_plant_raw = _get(dr, "stable_ready_plant")
        stable_ready_plant = (
            False
            if stable_ready_plant_raw is None
            else _as_explicit_bool(
                stable_ready_plant_raw,
                "task.domain_rand.stable_ready_plant",
            )
        )
        if stable_ready_plant:
            racket_cfg = getattr(
                getattr(env_cfg, "commands", None), "racket_target", None
            )
            stable_ready_actions = tuple(
                getattr(racket_cfg, "clip_names_per_clip", ()) or ()
            )
            if (
                getattr(racket_cfg, "target_mode", None) != "action_ball"
                or getattr(
                    racket_cfg,
                    "action_ball_diagnostic_unauthorized",
                    None,
                )
                is not True
                or len(stable_ready_actions) != 1
            ):
                raise _OverrideError(
                    "[train.py] task.domain_rand.stable_ready_plant=true is "
                    "restricted to exact N=1 diagnostic ActionBall launches; "
                    f"target_mode={getattr(racket_cfg, 'target_mode', None)!r} "
                    "diagnostic_unauthorized="
                    f"{getattr(racket_cfg, 'action_ball_diagnostic_unauthorized', None)!r} "
                    f"clip_names_per_clip={stable_ready_actions!r}"
                )
            required_events = (
                "base_com",
                "randomize_link_mass",
                "randomize_pd_gains",
            )
            missing_events = [
                name for name in required_events if not hasattr(E, name)
            ]
            if missing_events:
                raise _OverrideError(
                    "[train.py] stable-ready ActionBall plant cannot disable "
                    f"the required DR axes; missing event slots={missing_events}"
                )
            E.base_com = None
            E.randomize_link_mass = None
            E.randomize_pd_gains = None
            applied.append(
                "task.domain_rand.stable_ready_plant=true "
                "(torso CoM/link-mass/PD randomization disabled; historical "
                "robot-material and recipe-bound joint-default offset retained)"
            )
        else:
            if mr is not None and hasattr(E, "randomize_link_mass"):
                E.randomize_link_mass.params["mass_distribution_params"] = (
                    float(mr[0]),
                    float(mr[1]),
                )
                applied.append(
                    "events.randomize_link_mass.mass_distribution_params="
                    f"({float(mr[0])}, {float(mr[1])})"
                )
        if not stable_ready_plant and hasattr(E, "randomize_pd_gains"):
            if pr is None:
                E.randomize_pd_gains = None  # disable
                applied.append("events.randomize_pd_gains=None(disabled)")
            else:
                E.randomize_pd_gains.params["stiffness_distribution_params"] = (float(pr[0]), float(pr[1]))
                E.randomize_pd_gains.params["damping_distribution_params"] = (float(pr[0]), float(pr[1]))
                applied.append(f"events.randomize_pd_gains=({float(pr[0])}, {float(pr[1])})")

    # These must be the final command/observation mutations: each formal mode
    # validates the fully composed state and appends its canonical actor tail
    # atomically.  The mode guards make the two calls mutually exclusive.
    _finalize_task_first_training_cfg(env_cfg, task, applied)
    _finalize_action_ball_training_cfg(env_cfg, task, applied)

    return applied


# --------------------------------------------------------------------------- #
# Training (runs after the simulator is launched).
# --------------------------------------------------------------------------- #
def _run(cfg):
    import os
    from datetime import datetime

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking  # noqa: F401
    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.tasks.tracking.actor_observation_contract import (
        infer_actor_observation_contract,
        validate_actor_observation_contract,
    )
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs
    from whole_body_tracking.utils.training_contract import (
        checkpoint_contract_lineage_exact,
        load_action_ball_dynamic_ready_runtime_binding,
        load_action_ball_action_set_identity_from_launch_claim,
        require_checkpoint_contract_binding,
        validate_schema3_contract,
        validate_schema3_contract_structure,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Provenance: confirm we imported the WORKING TREE, not a stale install. If this path points into
    # site-packages instead of .../source/whole_body_tracking, a shadow copy is overriding your edits
    # (fix PYTHONPATH ordering in setup_train_env.sh / reinstall editable) and the YAML edits below are
    # being applied onto the wrong cfg classes.
    print(f"[train.py] whole_body_tracking imported from: {whole_body_tracking.__file__}", flush=True)

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) env cfg (gym registry) + task YAML overrides
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _cfg_mod = sys.modules.get(type(env_cfg).__module__)
    print(f"[train.py] env cfg source: {type(env_cfg).__name__} <- {getattr(_cfg_mod, '__file__', '?')}", flush=True)
    applied = _apply_task_overrides(env_cfg, cfg.task, _registry_clip_name(cfg))
    from whole_body_tracking.utils.effective_reward_recipe import (
        build_reward_backend_compatibility_receipt,
    )

    reward_backend_compatibility_receipt = (
        build_reward_backend_compatibility_receipt(env_cfg)
    )
    for decision in reward_backend_compatibility_receipt["decisions"]:
        print(
            "[train.py] "
            + _reward_backend_compatibility_log_line(decision),
            flush=True,
        )
    _launch_racket_cfg = getattr(
        getattr(env_cfg, "commands", None), "racket_target", None
    )
    if _physical_validity_guards_required(_launch_racket_cfg):
        # LAUNCH-SOURCE GATE: prove the checkout we are about to train FROM actually contains
        # the physical-validity guards. A pod checkout that predates them imports cleanly and
        # silently skips all three, so "the guard exists" must be asserted against the module
        # that was really imported, not against the repo the operator is looking at.
        _assert_physical_validity_guards_present(_launch_racket_cfg)
    plant_cfg = _get(cfg.task, "plant")
    zero_joint_friction_raw = _get(plant_cfg, "zero_joint_friction")
    zero_joint_friction_requested = (
        False
        if zero_joint_friction_raw is None
        else _as_explicit_bool(
            zero_joint_friction_raw, "task.plant.zero_joint_friction"
        )
    )
    print(f"[train.py] applied {len(applied)} task override(s) from cfg/task/{_get(cfg.task, 'name', task_id)}.yaml:", flush=True)
    for _a in applied:
        print(f"[train.py]     {_a}", flush=True)
    if not applied:
        print("[train.py] WARNING: 0 task overrides applied -> the run is using CODE DEFAULTS, not the "
              "YAML (the rewards/racket/env blocks did not compose, or all keys were absent).", flush=True)
    # Human-readable confirmation of the strike-training knobs, straight from the post-override cfg, so
    # you can read the actual runtime values off the launch log without opening logs/.../params/env.yaml.
    R = env_cfg.rewards
    if hasattr(R, "racket_position"):
        print("[train.py] racket reward std (post-override): "
              f"pos={R.racket_position.params.get('std')} vel={R.racket_velocity.params.get('std')} "
              f"normal={R.racket_normal.params.get('std')}", flush=True)
    if hasattr(env_cfg.commands, "racket_target"):
        _C = env_cfg.commands.racket_target
        print("[train.py] racket target (post-override): "
              f"target_mode={_C.target_mode} ref_perturb_curriculum_start={_C.ref_perturb_curriculum_start} "
              f"strike_window_s={_C.strike_window_s}", flush=True)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)

    # 2) PPO runner cfg from cfg.algo
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    # Task-level algo override (merge-audit 2026-07-06): a task YAML may pin ITS lineage's
    # algorithm deviations (e.g. Hitter entropy_coef 0.015) without touching the global
    # cfg/algo/ppo.yaml that every other lineage trains through. Whitelisted + fail-loud,
    # same contract as _apply_task_overrides.
    _task_algo = _get(cfg.task, "algo")
    _check_unknown_keys(_task_algo, ("entropy_coef",), "task.algo")
    if _task_algo is not None:
        _ec = _get(_task_algo, "entropy_coef")
        if _ec is not None:
            algo["algorithm"]["entropy_coef"] = float(_ec)
            print(f"[train.py] task.algo override: algorithm.entropy_coef={float(_ec)}", flush=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)
    _diag_racket_cfg = getattr(
        getattr(env_cfg, "commands", None), "racket_target", None
    )
    if getattr(
        _diag_racket_cfg, "action_ball_diagnostic_unauthorized", False
    ) is True:
        # Franco 2026-07-28: the run name itself carries the brand so no
        # logger/dashboard can present a bypassed run as formal evidence.
        _diag_suffix = "DIAGNOSTIC_UNAUTHORIZED"
        if _diag_suffix not in str(agent_cfg.run_name or ""):
            agent_cfg.run_name = (
                f"{agent_cfg.run_name}-{_diag_suffix}"
                if agent_cfg.run_name
                else _diag_suffix
            )
        print(
            "[train.py] WARN action-ball DIAGNOSTIC UNAUTHORIZED run: "
            f"run_name={agent_cfg.run_name}",
            flush=True,
        )
    if cfg.logger is not None:
        agent_cfg.logger = str(cfg.logger)
    if agent_cfg.logger in {"wandb", "neptune"} and cfg.log_project_name:
        agent_cfg.wandb_project = str(cfg.log_project_name)
        agent_cfg.neptune_project = str(cfg.log_project_name)

    # 3) reference motion clip(s), LOCAL-FIRST: motion_file=/motion_file_2= (or a local .npz path passed
    #    as registry_name/registry_name_2) skips WandB entirely (the documented no-WandB path — see
    #    run_training.md); otherwise the WandB registry is used.
    #    ONE clip = single-action policy. N clips = one arbitrary-N policy:
    #    MotionLoader concatenates them and clip_id selects the reference action
    #    each env imitates. Order must match every per-clip command table (and,
    #    for task-first, the manifest action_order).
    def _local_motion(name):
        """If ``name`` is a local motion.npz (or a dir containing one), return that path, else None."""
        p = pathlib.Path(str(name).split(":")[0])  # tolerate a :version suffix
        if p.is_file() and p.suffix == ".npz":
            return str(p)
        if (p / "motion.npz").is_file():
            return str(p / "motion.npz")
        return None

    if not _configured_items(_get(cfg, "motion_file"), _get(cfg, "motion_file_2")):
        # Back-compat: local paths passed as registry_name/registry_name_2 become motion_file, so
        # resolve_motion_sources below stays the single source of truth for local-vs-registry.
        _reg_candidates = _configured_items(
            _get(cfg, "registry_name") if _get(cfg, "registry_name") is not None else _get(cfg.task, "registry_name"),
            _get(cfg, "registry_name_2")
            if _get(cfg, "registry_name_2") is not None
            else _get(cfg.task, "registry_name_2"),
        )
        _local_hits = [_local_motion(r) for r in _reg_candidates]
        if _local_hits and all(h is not None for h in _local_hits):
            cfg.motion_file = _local_hits
        elif any(h is not None for h in _local_hits):
            # Local clips are all-or-nothing (see resolve_motion_sources): fail loud instead of
            # letting wandb.Api().artifact(<local path>) throw a cryptic HTTP error below.
            raise RuntimeError(
                f"[train.py] Mixed motion sources in registry_name/registry_name_2: {_reg_candidates}. "
                "Some values are local .npz paths and some are registry refs. Pass ALL clips locally "
                "via motion_file=/motion_file_2= (or make every registry_name a local path), or "
                "publish the local clip to the registry."
            )
    motion_files, motion_registries = resolve_motion_sources(cfg)
    for i, mf in enumerate(motion_files):
        src = motion_registries[i] if i < len(motion_registries) else "LOCAL (no registry)"
        print(f"[train.py] motion clip {i}: {mf}  [{src}]", flush=True)
    if len(motion_files) > 1:
        print(
            "[train.py] UNIFIED multi-clip policy: "
            f"{len(motion_files)} ordered action clip(s); local slots="
            f"{list(range(len(motion_files)))}",
            flush=True,
        )
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    _validate_task_first_motion_sources(env_cfg, motion_files)
    _validate_action_ball_motion_sources(env_cfg, motion_files)

    # 4) logging dir (same layout as scripts/rsl_rl/train.py so export/eval are unchanged)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Task: {task_id} | experiment: {agent_cfg.experiment_name} | log: {log_dir}")
    # Queue provenance is operational, not part of the scientific hard contract.
    # Publish the exact RSL directory before environment construction; a bad or
    # pre-existing binding therefore fails before any learning/checkpoint write.
    training_launch_claim_sha256 = _get(cfg, "training_launch_claim_sha256")
    training_launch_claim_path = _get(
        cfg, "training_launch_claim_path"
    )
    if (training_launch_claim_sha256 is None) != (
        training_launch_claim_path is None
    ):
        raise RuntimeError(
            "training_launch_claim_path and "
            "training_launch_claim_sha256 must be supplied together"
    )
    action_set_identity = None
    action_ball_launch_requested = (
        str(getattr(_launch_racket_cfg, "target_mode", ""))
        == "action_ball"
    )
    diagnostic_launch = False
    if training_launch_claim_path is not None and action_ball_launch_requested:
        try:
            action_set_identity = (
                load_action_ball_action_set_identity_from_launch_claim(
                    str(training_launch_claim_path),
                    expected_claim_sha256=str(
                        training_launch_claim_sha256
                    ),
                    actual_argv=_ORIGINAL_TRAINING_ARGV,
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "formal ActionBall launch claim did not yield a verified "
                "code-owned action-set identity"
            ) from exc
        print(
            "[train.py] action-set launch identity verified: "
            f"profile={action_set_identity['profile_id']} "
            f"N={action_set_identity['expected_n']} "
            f"order_digest="
            f"{action_set_identity['order_uid_digest_sha256']}",
            flush=True,
        )
    if action_ball_launch_requested:
        diagnostic_launch = (
            getattr(
                _launch_racket_cfg,
                "action_ball_diagnostic_unauthorized",
                False,
            )
            is True
        )
        if diagnostic_launch and action_set_identity is not None:
            raise RuntimeError(
                "diagnostic ActionBall training cannot consume a formal "
                "launch-claim action-set identity"
            )
        if not diagnostic_launch and action_set_identity is None:
            raise RuntimeError(
                "formal ActionBall training requires "
                "training_launch_claim_path/SHA and a verified code-owned "
                "action-set identity before scene construction"
            )
    (
        action_ball_shared_ready_bootstrap_requested,
        action_ball_policy_recipe_output_path,
    ) = _resolve_action_ball_shared_ready_bootstrap_request(
        cfg, action_ball_launch_requested=action_ball_launch_requested
    )
    (
        action_ball_dynamic_ready_bootstrap_requested,
        action_ball_dynamic_ready_pins,
    ) = _resolve_action_ball_dynamic_ready_bootstrap_request(
        cfg, action_ball_launch_requested=action_ball_launch_requested
    )
    if (
        action_ball_shared_ready_bootstrap_requested
        and action_ball_dynamic_ready_bootstrap_requested
    ):
        raise RuntimeError(
            "shared-ready and dynamic-ready actor bootstraps are mutually exclusive"
        )
    if (
        action_ball_policy_recipe_output_path is not None
        and not (
            action_ball_shared_ready_bootstrap_requested
            or action_ball_dynamic_ready_bootstrap_requested
        )
    ):
        raise RuntimeError(
            "action_ball_policy_recipe_output_path requires exactly one "
            "shared-ready or dynamic-ready bootstrap"
        )
    action_ball_dynamic_ready_binding = None
    if action_ball_dynamic_ready_bootstrap_requested:
        if _get(cfg, "checkpoint_path") is not None:
            raise RuntimeError(
                "dynamic-ready bootstrap is fresh-only and cannot alter a resumed run"
            )
        action_order = tuple(
            str(value)
            for value in (
                getattr(_launch_racket_cfg, "clip_names_per_clip", ()) or ()
            )
        )
        try:
            action_ball_dynamic_ready_binding = (
                load_action_ball_dynamic_ready_runtime_binding(
                    artifact_path=action_ball_dynamic_ready_pins[
                        "action_ball_dynamic_ready_artifact_path"
                    ],
                    artifact_sha256=action_ball_dynamic_ready_pins[
                        "action_ball_dynamic_ready_artifact_sha256"
                    ],
                    nominal_hold_receipt_path=(
                        action_ball_dynamic_ready_pins[
                            "action_ball_dynamic_ready_nominal_receipt_path"
                        ]
                    ),
                    nominal_hold_receipt_sha256=(
                        action_ball_dynamic_ready_pins[
                            "action_ball_dynamic_ready_nominal_receipt_sha256"
                        ]
                    ),
                    action_order=list(action_order),
                    motion_paths=list(motion_files),
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "dynamic-ready candidate/nominal-hold pins failed strict "
                "pre-scene validation"
            ) from exc
        env_cfg.commands.motion.action_ball_dynamic_ready = (
            action_ball_dynamic_ready_binding
        )
        print(
            "[train.py] ActionBall dynamic-ready pre-scene binding verified: "
            f"action={action_order[0]} binding_sha256="
            f"{action_ball_dynamic_ready_binding['binding_sha256']}",
            flush=True,
        )
    if action_ball_policy_recipe_output_path is not None:
        if (
            not diagnostic_launch
            or num_envs != 1
            or _get(cfg, "checkpoint_path") is not None
        ):
            raise RuntimeError(
                "action-ball policy recipe materialization requires a fresh "
                "diagnostic ActionBall run with exactly one environment"
            )
    _publish_lean_queue_binding_if_requested(cfg, log_dir)

    # 5) build env, wrap, run
    # Reward provenance is captured from the fully composed env cfg (including
    # every task/Hydra override), never from the nominal reward-pack label.
    # An optional root-level expected SHA makes a preregistered recipe fail
    # before the expensive scene import.
    effective_reward_receipt = _build_effective_reward_receipt_for_training(
        env_cfg,
        cfg,
        require_expected_sha256=(
            str(
                getattr(
                    getattr(env_cfg.commands, "racket_target", None),
                    "target_mode",
                    "",
                )
            )
            in ("task_first", "action_ball")
        ),
    )
    print(
        "[train.py] effective reward recipe SHA-256: "
        f"{effective_reward_receipt['sha256']}",
        flush=True,
    )
    if (
        reward_backend_compatibility_receipt[
            "effective_reward_recipe_sha256"
        ]
        != effective_reward_receipt["sha256"]
    ):
        raise RuntimeError(
            "Reward configuration changed after backend compatibility "
            "resolution"
        )
    render_mode = "rgb_array" if cfg.video else None
    _emit_lean_queue_phase(cfg, "scene_import_start")
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    runtime_env = env.unwrapped

    def _scene_has(name: str) -> bool:
        try:
            runtime_env.scene[name]
        except KeyError:
            return False
        return True

    runtime_racket = getattr(
        getattr(runtime_env.cfg, "commands", None), "racket_target", None
    )
    _emit_lean_queue_phase(
        cfg,
        "scene_import_done",
        actual_num_envs=int(runtime_env.num_envs),
        physical_ball_enabled=bool(
            getattr(runtime_racket, "physical_ball", False)
        ),
        physical_scene_entities={
            name: _scene_has(name)
            for name in ("pb_ball", "pb_table", "pb_table_visual")
        },
    )
    expected_contract = _get(cfg.task, "actor_obs_contract")
    actor_contract = None
    if expected_contract is not None:
        actor_contract = validate_actor_observation_contract(env.unwrapped, str(expected_contract))
        print(
            "[train.py] actor observation contract validated: "
            f"{actor_contract.name} ({actor_contract.total_dim}D, obs_mode={actor_contract.obs_mode})",
            flush=True,
        )
    else:
        actor_contract = infer_actor_observation_contract(env.unwrapped)

    action_ball_policy_bootstrap = None
    if (
        action_ball_shared_ready_bootstrap_requested
        or action_ball_dynamic_ready_bootstrap_requested
    ):
        action_ball_policy_bootstrap = (
            _action_ball_policy_bootstrap_contract(
                env.unwrapped,
                actor_contract,
                agent_cfg,
                dynamic_ready_binding=action_ball_dynamic_ready_binding,
            )
        )
        ready_identity = (
            action_ball_policy_bootstrap["ready_source"][
                "shared_ready_joint_pos_sha256"
            ]
            if action_ball_policy_bootstrap["schema_version"] == 1
            else action_ball_policy_bootstrap["ready_source"]["identity"][
                "binding_sha256"
            ]
        )
        print(
            "[train.py] ActionBall policy bootstrap validated: "
            f"schema={action_ball_policy_bootstrap['schema_version']} "
            f"N={action_ball_policy_bootstrap['action_count']} "
            "noise_std="
            f"{action_ball_policy_bootstrap['initialization']['init_noise_std']} "
            f"ready_identity_sha256={ready_identity}",
            flush=True,
        )
    if action_ball_policy_recipe_output_path is not None:
        materialized_recipe = _action_ball_agent_recipe(
            agent_cfg, policy_bootstrap=action_ball_policy_bootstrap
        )
        env.close()
        document = _materialize_action_ball_policy_recipe(
            action_ball_policy_recipe_output_path,
            policy_recipe=materialized_recipe,
            policy_bootstrap=action_ball_policy_bootstrap,
        )
        print(
            "[train.py] ACTION_BALL_POLICY_RECIPE_MATERIALIZED "
            + json.dumps(
                {
                    "output_path": action_ball_policy_recipe_output_path,
                    "policy_contract_sha256": document[
                        "policy_contract_sha256"
                    ],
                    "action_order": document["action_order"],
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        return

    hard_contract = _build_training_hard_contract(
        env.unwrapped,
        actor_contract,
        effective_reward_receipt=effective_reward_receipt,
        agent_cfg=agent_cfg,
        action_set_identity=action_set_identity,
        action_ball_policy_bootstrap=action_ball_policy_bootstrap,
    )
    task_first_training = "task_first_training" in hard_contract
    action_ball_training = "action_ball_training" in hard_contract
    if task_first_training and action_ball_training:
        raise RuntimeError(
            "training hard contract cannot enable task-first and action-ball "
            "simultaneously"
        )
    action_ball_diagnostic_unauthorized = (
        _validate_action_ball_training_authorization(
            hard_contract["action_ball_training"]
        )
        if action_ball_training
        else False
    )
    strict_exact_training = task_first_training or action_ball_training
    strict_training_label = (
        "action-ball" if action_ball_training else "task-first"
    )
    try:
        validate_schema3_contract_structure(hard_contract)
    except ValueError as exc:
        raise RuntimeError(
            "new training hard contract failed schema-3 structural validation"
        ) from exc
    lateral_training_runtime = None
    lateral_training = _resolve_lateral_training_runtime(env.unwrapped)
    if lateral_training is not None:
        lateral_cfg, lateral_hard_contract = lateral_training
        from whole_body_tracking.tasks.tracking.mdp.isaac_lateral_perturbation import (
            IsaacLateralPerturbationTrainingRuntime,
        )

        lateral_training_runtime = IsaacLateralPerturbationTrainingRuntime(
            env,
            lateral_cfg,
            cell=str(
                getattr(
                    env.unwrapped.cfg, _LATERAL_TRAINING_SPEC_ATTR
                )["cell"]
            ),
        )
        if lateral_training_runtime.hard_contract != lateral_hard_contract:
            raise RuntimeError(
                "lateral trainer runtime and checkpoint hard contract disagree"
            )
        if hard_contract.get("lateral_perturbation") != lateral_hard_contract:
            raise RuntimeError(
                "training_contract.json omitted or changed the enabled lateral trainer identity"
            )
        print(
            "[train.py] LATERAL_TRAINER_READY: "
            f"cell={lateral_hard_contract['cell']} seed={lateral_hard_contract['seed']} "
            f"body={lateral_hard_contract['body_name']} "
            f"frame={lateral_hard_contract['force_frame']} "
            f"impulse=[{lateral_hard_contract['normalized_impulse_min_mps']},"
            f"{lateral_hard_contract['normalized_impulse_max_mps']}]m/s",
            flush=True,
        )
    if zero_joint_friction_requested:
        _require_zero_joint_friction_contract(hard_contract)
        print(
            "[train.py] ZERO_FRICTION_RUNTIME_OK: 31/31 instantiated PhysX joint "
            "friction coefficients are exactly 0.0",
            flush=True,
        )
    contract_path = os.path.join(log_dir, "params", "training_contract.json")
    os.makedirs(os.path.dirname(contract_path), exist_ok=True)
    with open(contract_path, "w", encoding="utf-8") as stream:
        json.dump(hard_contract, stream, indent=2, sort_keys=True)
        stream.write("\n")
    reward_receipt_path = os.path.join(
        log_dir, "params", "effective_reward_recipe.json"
    )
    _write_effective_reward_receipt(
        reward_receipt_path, effective_reward_receipt, hard_contract
    )
    backend_compatibility_path = os.path.join(
        log_dir, "params", "reward_backend_compatibility.json"
    )
    _write_reward_backend_compatibility_receipt(
        backend_compatibility_path,
        reward_backend_compatibility_receipt,
        effective_reward_receipt,
    )
    hard_contract_sha256 = _sha256_file(contract_path)
    print(f"[train.py] hard training contract: {contract_path}", flush=True)
    print(
        f"[train.py] effective reward receipt: {reward_receipt_path}",
        flush=True,
    )
    print(
        "[train.py] reward backend compatibility receipt: "
        f"{backend_compatibility_path}",
        flush=True,
    )
    _emit_lean_queue_phase(
        cfg,
        "hard_contract_written",
        path=contract_path,
        schema_version=int(hard_contract["schema_version"]),
        sha256=hard_contract_sha256,
    )
    if lateral_training_runtime is not None:
        class _LateralTrainingGymWrapper(gym.Wrapper):
            def __init__(self, wrapped_env, runtime):
                super().__init__(wrapped_env)
                self._lateral_runtime = runtime

            def step(self, action):
                return self._lateral_runtime.step(action)

            def reset(self, *args, **kwargs):
                return self._lateral_runtime.reset(*args, **kwargs)

            def close(self):
                return self._lateral_runtime.close()

        env = _LateralTrainingGymWrapper(env, lateral_training_runtime)
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    # Only hand the runner registry refs for wandb lineage (use_artifact) when the clips actually came
    # from the registry; local runs pass None (a local motion path would crash wandb.run.use_artifact).
    # resolve_motion_sources already returned normalized 'collection:alias' refs (a bare collection name
    # is an HTTP 400). List-valued: the runner records ALL used clips, not just clip 0.
    runner_registry_name = motion_registries if motion_registries else None
    # Optional operational provenance supplied by a fail-closed launcher.  It is embedded only in
    # checkpoint infos and deliberately excluded from training_contract.json (and its scientific
    # contract SHA).  Plain training commands remain compatible: absent means no claim is written.
    # A checkpoint may claim formal schema-3 provenance only when it is fresh from this contract or
    # resumes an already exact-bound schema-3 lineage. A legacy/mismatched warm-start remains useful,
    # but every descendant checkpoint is permanently marked exact-ineligible; merely saving it beside
    # a new JSON must never launder historical execution semantics.
    ckpt = getattr(cfg, "checkpoint_path", None)
    source_checkpoint = None
    motion_kinematics_exact = bool(hard_contract["motion_kinematics_exact"])
    if strict_exact_training and not motion_kinematics_exact:
        raise RuntimeError(
            f"[train.py] {strict_training_label} requires exact schema-2 "
            "motion kinematics "
            "for every action; a velocity-center curriculum cannot use "
            "ambiguous COM/link-origin motion semantics"
        )
    contract_lineage_exact = _action_ball_contract_lineage_exact(
        source_lineage_exact=ckpt is None,
        motion_kinematics_exact=motion_kinematics_exact,
        diagnostic_unauthorized=action_ball_diagnostic_unauthorized,
    )
    if action_ball_diagnostic_unauthorized:
        print(
            "[train.py] WARN diagnostic_unauthorized forces "
            "training_contract_lineage_exact=0; formal evidence, curriculum "
            "promotion, exact export, and formal judge are prohibited",
            flush=True,
        )
    if not motion_kinematics_exact:
        print(
            "[train.py] WARNING: one or more motion clips lack declared schema-2 COM-velocity/body-order "
            "semantics; checkpoints from this run are formal-ineligible until the clips are "
            "migrated/re-exported (see scripts/migrate_motion_kinematics.py).",
            flush=True,
        )
    if ckpt is not None:
        ckpt = os.path.abspath(str(ckpt))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"[train.py] checkpoint_path does not exist: {ckpt}")
        if strict_exact_training and bool(
            getattr(cfg, "checkpoint_tolerant", False)
        ):
            raise RuntimeError(
                f"[train.py] {strict_training_label} forbids "
                "checkpoint_tolerant actor-only "
                "warm starts; optimizer and exact curriculum state must resume "
                "together, or the run must start a fresh lineage"
            )
        source_checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        if strict_exact_training and (
            not isinstance(source_checkpoint, dict)
            or "optimizer_state_dict" not in source_checkpoint
        ):
            raise RuntimeError(
                f"[train.py] {strict_training_label} resume requires "
                "optimizer_state_dict; "
                "actor-only or fresh-optimizer warm starts lose the exact PPO "
                "and curriculum lineage"
            )
        prior_contract_path = os.path.join(os.path.dirname(ckpt), "params", "training_contract.json")
        allow_missing = bool(getattr(cfg, "checkpoint_allow_missing_contract", False))
        allow_mismatch = bool(getattr(cfg, "checkpoint_allow_contract_mismatch", False))
        if not os.path.isfile(prior_contract_path):
            message = (
                f"[train.py] checkpoint has no hard training contract: {prior_contract_path}. "
                "Tensor shapes cannot detect a changed HitterPure strike plane/box/face convention."
            )
            if strict_exact_training:
                raise RuntimeError(
                    message
                    + f" {strict_training_label} resumes never allow an "
                    "unbound warm-start; "
                    "start a fresh lineage instead."
                )
            if getattr(actor_contract, "name", None) == "hitter_pure" and not allow_missing:
                raise RuntimeError(
                    message + " For a deliberately audited legacy warm-start, pass "
                    "checkpoint_allow_missing_contract=true; the override is recorded in stdout."
                )
            print(message + " Continuing only because this is non-HitterPure or explicitly allowed.",
                  flush=True)
        else:
            with open(prior_contract_path, encoding="utf-8") as stream:
                prior_contract = json.load(stream)
            diffs = _contract_diff(prior_contract, hard_contract)
            if diffs and (strict_exact_training or not allow_mismatch):
                raise RuntimeError(
                    "[train.py] checkpoint hard-contract mismatch; refusing a contaminated resume:\n  - "
                    + "\n  - ".join(diffs)
                    + (
                        f"\n{strict_training_label} does not permit "
                        "representation-transfer overrides; start a fresh "
                        "lineage."
                        if strict_exact_training
                        else "\nIf this is an intentional representation transfer rather than a "
                        "resume, pass checkpoint_allow_contract_mismatch=true and use a new run name."
                    )
                )
            if diffs:
                print("[train.py] WARNING: explicit hard-contract mismatch override:\n  - "
                      + "\n  - ".join(diffs), flush=True)
            else:
                print(f"[train.py] checkpoint hard contract MATCH: {prior_contract_path}", flush=True)
                try:
                    if action_ball_diagnostic_unauthorized:
                        validate_schema3_contract_structure(prior_contract)
                    else:
                        validate_schema3_contract(prior_contract)
                    require_checkpoint_contract_binding(
                        source_checkpoint,
                        schema=int(prior_contract["schema_version"]),
                        sha256=_sha256_file(prior_contract_path),
                        require_lineage_exact=(
                            not action_ball_diagnostic_unauthorized
                        ),
                    )
                    observed_lineage_exact = (
                        checkpoint_contract_lineage_exact(source_checkpoint)
                    )
                    expected_lineage_exact = (
                        not action_ball_diagnostic_unauthorized
                    )
                    if observed_lineage_exact is not expected_lineage_exact:
                        raise ValueError(
                            "checkpoint lineage does not match the live "
                            "diagnostic/formal authorization"
                        )
                except ValueError as exc:
                    if strict_exact_training:
                        raise RuntimeError(
                            f"[train.py] {strict_training_label} source "
                            "checkpoint is not "
                            "exact-bound to its schema-3 training contract"
                        ) from exc
                    print(
                        "[train.py] WARNING: source checkpoint contract is not exact-bound; "
                        f"descendants remain formal-ineligible: {exc}",
                        flush=True,
                    )
                else:
                    contract_lineage_exact = (
                        _action_ball_contract_lineage_exact(
                            source_lineage_exact=True,
                            motion_kinematics_exact=motion_kinematics_exact,
                            diagnostic_unauthorized=(
                                action_ball_diagnostic_unauthorized
                            ),
                        )
                    )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        registry_name=runner_registry_name,
        training_contract_schema_version=int(hard_contract["schema_version"]),
        training_contract_sha256=hard_contract_sha256,
        training_contract_lineage_exact=contract_lineage_exact,
        training_launch_claim_sha256=training_launch_claim_sha256,
        require_exact_resume_state=strict_exact_training,
    )
    if action_ball_policy_bootstrap is not None:
        bootstrap_applied = _apply_action_ball_fresh_policy_bootstrap(
            runner,
            action_ball_policy_bootstrap,
            checkpoint_path=ckpt,
        )
        print(
            "[train.py] ActionBall policy bootstrap: "
            f"{'APPLIED_FRESH' if bootstrap_applied else 'SKIPPED_RESUME'}",
            flush=True,
        )
    if strict_exact_training and bool(getattr(runner, "is_distributed", False)):
        raise RuntimeError(
            f"[train.py] {strict_training_label} is single-process only. "
            "Curriculum "
            "evidence has no distributed all-reduce contract yet, so multi-GPU "
            "ranks could promote different action domains."
        )
    runner.add_git_repo_to_log(__file__)

    # Params/runtime lineage must be durable and bound before a strict resume
    # load can consume checkpoint bytes.  Keep the historical load behavior in
    # one closure, invoked below after the ActionBall bootstrap receipt is
    # minted and attached to the runner.
    def _load_requested_checkpoint():
        if ckpt is None:
            return
        if bool(getattr(cfg, "checkpoint_tolerant", False)):
            # Warm-start ACROSS critic-layout changes (e.g. the 318-D pre-merge lineage into the
            # 316-D merged model, or deploy-parity ckpts into VirtualBall's critic): actor + std
            # (+ obs normalizer if shapes agree) load strictly by name; the critic re-initializes
            # and re-learns — PPO tolerates this warm-start (fresh value function, ~hundreds of
            # iterations of value lag). Deliberate resume stays STRICT without this flag.
            from whole_body_tracking.utils.ckpt_compat import load_actor_tolerant

            load_actor_tolerant(runner, ckpt)
            print(f"[train.py] TOLERANT warm-start from {ckpt} (actor loaded; critic fresh if "
                  f"layout changed — deliberate warm-start semantics)", flush=True)
        else:
            # Warm-start checkpoints (e.g. make_hitter_warmstart.py) deliberately drop the
            # optimizer state because parameter shapes changed; a fresh optimizer is correct there.
            has_optimizer = "optimizer_state_dict" in source_checkpoint
            runner.load(ckpt, load_optimizer=has_optimizer)
            print(f"[train.py] RESUMED from checkpoint: {ckpt} (continuing at iteration "
                  f"{getattr(runner, 'current_learning_iteration', '?')}, "
                  f"optimizer={'resumed' if has_optimizer else 'FRESH — no optimizer_state_dict in ckpt'})",
                  flush=True)

    params_dir = os.path.join(log_dir, "params")
    env_yaml_path = os.path.join(params_dir, "env.yaml")
    agent_yaml_path = os.path.join(params_dir, "agent.yaml")
    env_pickle_path = os.path.join(params_dir, "env.pkl")
    agent_pickle_path = os.path.join(params_dir, "agent.pkl")
    dump_yaml(env_yaml_path, env_cfg)
    dump_yaml(agent_yaml_path, agent_cfg)
    dump_pickle(env_pickle_path, env_cfg)
    dump_pickle(agent_pickle_path, agent_cfg)
    if action_ball_training and not action_ball_diagnostic_unauthorized:
        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_frozen_eval_identity as frozen_eval_identity,
            action_ball_runtime_bootstrap as runtime_bootstrap,
        )

        if training_launch_claim_sha256 is None:
            raise RuntimeError(
                "formal ActionBall frozen evaluation requires the exact "
                "training launch-claim SHA"
            )
        if training_launch_claim_path is None:
            raise RuntimeError(
                "formal ActionBall runtime bootstrap requires the exact "
                "training launch-claim path"
            )
        runtime_repo_root = _action_ball_repo_root(
            env_cfg.commands.motion
        )
        identity_document = (
            frozen_eval_identity.build_runtime_identity_document(
                repo_root=runtime_repo_root,
                task_id=task_id,
                training_launch_claim_sha256=(
                    training_launch_claim_sha256
                ),
                training_contract_path=contract_path,
                environment_config_pickle_path=env_pickle_path,
                agent_config_pickle_path=agent_pickle_path,
            )
        )
        identity_path = os.path.join(
            params_dir, "action_ball_frozen_eval_runtime.json"
        )
        identity_bytes = (
            frozen_eval_identity.canonical_document_bytes(
                identity_document
            )
        )
        try:
            descriptor = os.open(
                identity_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "ActionBall frozen-evaluation runtime identity namespace "
                f"is already spent: {identity_path}"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(identity_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        print(
            "[train.py] ActionBall frozen-evaluation runtime identity: "
            f"{identity_path} "
            f"sha256={hashlib.sha256(identity_bytes).hexdigest()}",
            flush=True,
        )
        runtime_bootstrap.durably_sync_runtime_inputs(
            contract_path,
            env_pickle_path,
            agent_pickle_path,
            identity_path,
        )
        bootstrap_document = (
            runtime_bootstrap.build_runtime_bootstrap_receipt_document(
                repo_root=runtime_repo_root,
                task_id=task_id,
                training_launch_claim_sha256=(
                    training_launch_claim_sha256
                ),
                launch_claim_path=training_launch_claim_path,
                training_contract_path=contract_path,
                environment_config_pickle_path=env_pickle_path,
                agent_config_pickle_path=agent_pickle_path,
                runtime_identity_path=identity_path,
            )
        )
        bootstrap_path = os.path.join(
            params_dir,
            runtime_bootstrap.RECEIPT_FILENAME,
        )
        bootstrap_publication = (
            runtime_bootstrap.publish_runtime_bootstrap_receipt(
                output_path=bootstrap_path,
                document=bootstrap_document,
            )
        )
        bind_runtime_bootstrap = getattr(
            runner,
            "bind_runtime_bootstrap_receipt",
            None,
        )
        if not callable(bind_runtime_bootstrap):
            raise RuntimeError(
                "formal ActionBall runner lacks runtime-bootstrap binding"
            )
        bind_runtime_bootstrap(
            content_sha256=bootstrap_publication[
                "content_sha256"
            ],
            artifact_receipt=bootstrap_publication[
                "artifact_receipt"
            ],
        )
        print(
            "[train.py] ActionBall runtime bootstrap receipt: "
            f"{bootstrap_path} "
            "content_sha256="
            f"{bootstrap_publication['content_sha256']} "
            "file_sha256="
            f"{bootstrap_publication['artifact_receipt']['sha256']}",
            flush=True,
        )

    _load_requested_checkpoint()

    if lateral_training_runtime is None:
        # Preserve the historical default-off control flow exactly.
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=not action_ball_training,
        )
        env.close()
    else:
        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=not action_ball_training,
            )
        finally:
            # A clean terminal full-batch zero overwrite is part of the enabled run contract.
            # Close the outer RSL/Gym wrappers for their own bookkeeping, then call the runtime
            # owner directly as an idempotent backstop in case an upstream wrapper failed to
            # forward ``close``.
            try:
                env.close()
            finally:
                lateral_training_runtime.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    _emit_lean_queue_phase(cfg, "hydra_resolved")
    kit_args, kit_carb_count, kit_tbb_count = _resolve_kit_thread_caps(cfg)

    # 热启动/续训的"归一化 2x2 真值表"预检,必须在 Kit 启动前跑:checkpoint 里有没有
    # obs_norm_state_dict,和本次解析出的 algo.runner.empirical_normalization 不一致时,老路径
    # 要等 Kit 起完、_run 里 runner.load/ckpt_compat 才炸出费解的 KeyError,白烧几分钟 GPU 启动
    # 费;这里在 CPU 上先加载先炸,并给出单条 CLI 修复命令。main 没有 exact-resume 一类"无视
    # 本次 CLI runner 配置"的路径 —— strict resume 和 checkpoint_tolerant 热启动最终都吃 _run 里
    # runner_kwargs 解析的同一份配置 —— 所以所有带 checkpoint_path 的启动一律预检,不设豁免。
    if _get(cfg, "checkpoint_path") is not None:
        # 刻意按文件路径加载模块而不是 import whole_body_tracking.utils.…:包 __init__.py 会连带
        # 注册 Isaac 任务,而此刻 Kit 还没起。
        _preflight_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "source/whole_body_tracking/whole_body_tracking/utils"
            / "checkpoint_normalization_preflight.py"
        )
        _preflight_spec = importlib.util.spec_from_file_location(
            "checkpoint_normalization_preflight", _preflight_path
        )
        _preflight = importlib.util.module_from_spec(_preflight_spec)
        _preflight_spec.loader.exec_module(_preflight)
        _runner_node = _get(_get(cfg, "algo"), "runner")
        _empirical = _get(_runner_node, "empirical_normalization")
        if _empirical is None:
            # 与 _run 里 runner_kwargs 的 bool(r["empirical_normalization"]) 同源同语义:缺 key
            # 现在就报错,不带默认值猜 —— 猜错了预检就成了假保证。
            raise RuntimeError(
                "[train.py] algo.runner.empirical_normalization is missing from the resolved "
                "config; cannot preflight the checkpoint normalization truth table."
            )
        resolved_checkpoint = _preflight.preflight_checkpoint_normalization(
            _get(cfg, "checkpoint_path"),
            empirical_normalization=_as_explicit_bool(
                _empirical, "algo.runner.empirical_normalization"
            ),
        )
        if resolved_checkpoint is not None:
            # 目录形式的 checkpoint_path 重绑到刚通过预检的精确 model_N.pt:防止并发训练继续往
            # 同一目录写更新的 checkpoint,导致 Kit 起完后 _run 真正加载的字节和预检看过的不一致
            # (竞态);顺带让 _run 里"必须是文件"的检查对目录输入也成立。
            cfg.checkpoint_path = str(resolved_checkpoint)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so the kit app does not try to
    # parse Hydra's `task=...`/`algo=...` overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher_kwargs = {
        "headless": bool(cfg.headless),
        "device": str(cfg.device),
        "enable_cameras": bool(cfg.video),
    }
    if kit_args is not None:
        app_launcher_kwargs["kit_args"] = kit_args
    app_launcher = AppLauncher(**app_launcher_kwargs)
    simulation_app = app_launcher.app
    _emit_lean_queue_phase(cfg, "app_started")
    # Print the traceback BEFORE closing the app: Isaac's simulation_app.close() hard-exits the
    # process (os._exit), which otherwise swallows any exception from _run and makes a real failure
    # look like a clean "exit 0" with the log truncated at startup.
    failed = False
    try:
        if kit_args is not None:
            import carb

            _verify_kit_thread_caps(
                carb.settings.get_settings(), kit_carb_count, kit_tbb_count
            )
        _run(cfg)
    except Exception:
        import traceback
        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        failed = True
    finally:
        try:
            import wandb

            if getattr(wandb, "run", None) is not None:
                wandb.finish()
        except Exception as exc:
            print(f"[train.py] WARNING: wandb.finish() failed: {exc}", flush=True)
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
