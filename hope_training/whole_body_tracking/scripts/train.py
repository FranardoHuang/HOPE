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
import json
import math
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
        "formula": "mean(relu(abs(qd)/joint_velocity_limits-margin)^2)",
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
            "mean(1-exp(-square(relu(abs(delta_processed_qdes)/(joint_velocity_limits*0.02)-margin)/(1-margin))))"
        ),
        "gate": "same_attempt_post_strike_age_s_inclusive",
    }


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
    legs = [name for name in names if name in _A3_LOWER_BODY_LEG_JOINTS]
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
        "joint_order": "runtime_articulation_subsequence",
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
        "joint_order": "runtime_articulation_subsequence",
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


def _build_training_hard_contract(env, actor_contract) -> dict:
    """Immutable actor/task facts that must match across a checkpoint resume.

    Reward weights, termination thresholds and optimizer settings are normally absent because
    they are curriculum-mutable.  Narrow exceptions bind the post-override racket-guidance pair
    and qdot-limit hinge: those values are causal identities of their registered ablations.
    Geometry, command meaning, clip identity, action processing, and every field that can move a
    strike/reveal/deadline or actor-visible target in time are also immutable.
    """
    from whole_body_tracking.utils.training_contract import (
        TRAINING_CONTRACT_SCHEMA_VERSION,
        runtime_execution_facts,
    )

    env_cfg = env.cfg
    motion_cmd = env.command_manager.get_term("motion")
    motion = motion_cmd.cfg
    try:
        racket_cmd = env.command_manager.get_term("racket_target")
    except KeyError:
        racket_cmd = None
    racket = None if racket_cmd is None else racket_cmd.cfg
    runtime_facts = runtime_execution_facts(env, actor_contract)
    lateral_training = _resolve_lateral_training_runtime(env)
    processed_qdes_slew_contract = _processed_qdes_slew_hinge_reward_contract(
        env_cfg, runtime_facts
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
        **runtime_facts,
        "target_mode": attr(racket, "target_mode"),
        "normal_mode": attr(racket, "normal_mode"),
        "racket_pos_range_per_clip": attr(racket, "racket_pos_range_per_clip"),
        "racket_vel_range_per_clip": attr(racket, "racket_vel_range_per_clip"),
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
            if lateral_training is None
            else {"lateral_perturbation": lateral_training[1]}
        ),
        "motion_clips": clips,
        "question_bank": question_bank,
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
    "base_target_x_range", "base_target_y_range",
    "normal_mode", "forehand_on_negative_y", "mount_normal_axis", "mount_normal_sign",
    # 每 clip 击球面符号(正反手各用拍子固定的一面;空/缺省=标量 mount_normal_sign,现役行为不变)
    "mount_normal_sign_per_clip",
    "target_mode", "ref_perturb_pos", "ref_perturb_vel", "ref_perturb_normal",
    "ref_perturb_curriculum_steps", "ref_perturb_curriculum_start", "ref_perturb_success_gated",
    "ref_perturb_advance_threshold", "ref_perturb_advance_rate", "ref_vel_scale", "ref_vel_scale_by_motion",
    "debug_reward_logging",
    "clean_reference_strike_velocity", "clean_strike_vel_window",
    "adaptive_sigma", "sigma_update_every", "sigma_ema_scale",
    "sigma_pos_min", "sigma_pos_max", "sigma_vel_min", "sigma_vel_max",
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
    # Wave-B mutually-exclusive lower-body diagnostics. Explicit zero-valued controls still
    # activate their measurement probes and hard-contract identity.
    "lower_body_pose_imitation_weight", "lower_body_pose_imitation_std",
    "lower_body_pose_imitation_support_pre_s",
    "lower_body_pose_imitation_support_post_s",
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
    "joint_torques_weight",
    # per-term overrides of the six imitation terms + the global/in-window scales
    "motion_global_anchor_pos_weight", "motion_global_anchor_pos_std",
    "motion_global_anchor_ori_weight", "motion_global_anchor_ori_std",
    "motion_body_pos_weight", "motion_body_pos_std",
    "motion_body_ori_weight", "motion_body_ori_std",
    "motion_body_lin_vel_weight", "motion_body_lin_vel_std",
    "motion_body_ang_vel_weight", "motion_body_ang_vel_std",
    "motion_scale", "motion_scale_in_window",
    # penalties / regularization
    "action_rate_weight", "joint_limit_weight", "undesired_contacts_weight",
    "pre_strike_foot_slip_weight", "prestrike_waist_twist_weight",
    "arm_torque_saturation_weight", "prestrike_upright_weight", "foot_orientation_weight",
    # proximity power-gate for the face/velocity channels (reward_staged_design §② C2a)
    "face_gate_by_pos", "face_gate_radius",
    # constant guidance penalty toward the racket target (reward_staged_design §② B2)
    "racket_guidance_weight", "racket_face_guidance_weight", "racket_face_guidance_theta_max",
    "racket_face_conditional_guidance_weight",
)

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


# YAML clip-name -> clip_id index. MUST match RacketTargetCommand._clip_names (0=forehand, 1=backhand).
_CLIP_NAME_TO_ID = {"forehand": 0, "backhand": 1}


def _resolve_vel_range_per_clip(rk):
    """Build the optional PER-CLIP racket target-velocity tuple from the YAML ``vel_range_per_clip`` block.

    YAML (readable, keyed by swing name; each axis a [lo, hi] list)::

        vel_range_per_clip:
          forehand: {x: [1.5, 3.5], y: [-1.0, 1.0], z: [0.0, 1.5]}
          backhand: {x: [1.2, 2.4], y: [-1.0, 1.0], z: [0.0, 1.2]}

    Returns a tuple indexed by clip_id (0=forehand, 1=backhand) of ``((xlo,xhi),(ylo,yhi),(zlo,zhi))``,
    or ``None`` when the key is absent (-> keep the shared ``vel_*_range`` box; backward compatible).
    Forehand and backhand reference clips have different natural strike speeds, so a shared box overshoots
    the slower backhand. Mirrors the per-clip ``strike_phase_per_clip`` / ``ref_vel_scale_by_motion`` style.
    """
    block = _get(rk, "vel_range_per_clip")
    if block is None:
        return None
    by_id = {}
    for name in block:
        cid = _CLIP_NAME_TO_ID.get(str(name).lower())
        if cid is None:
            raise _OverrideError(
                f"racket.vel_range_per_clip: unknown clip name {name!r} (expected forehand/backhand)")
        axes = _get(block, name)

        def _r(ax):
            v = _get(axes, ax)
            if v is None:
                raise _OverrideError(f"racket.vel_range_per_clip[{name}]: missing '{ax}' [lo,hi] range")
            return (float(v[0]), float(v[1]))

        by_id[cid] = (_r("x"), _r("y"), _r("z"))
    return tuple(by_id[i] for i in range(len(by_id)))


def _resolve_pos_range_per_clip(rk):
    """Build the optional PER-CLIP racket target-POSITION tuple from the YAML ``pos_range_per_clip`` block.

    YAML (readable, keyed by swing name; each axis a [lo, hi] list, ADDED to the env origin; y is SIGNED)::

        pos_range_per_clip:
          forehand: {x: [0.50, 0.62], y: [-0.45, -0.20], z: [0.72, 0.98]}
          backhand: {x: [0.50, 0.62], y: [ 0.20,  0.45], z: [1.05, 1.30]}

    Returns a tuple indexed by clip_id (0=forehand, 1=backhand) of ``((xlo,xhi),(ylo,yhi),(zlo,zhi))``,
    or ``None`` when the key is absent (-> keep the shared ``pos_*_range`` box; backward compatible).
    Mirrors ``vel_range_per_clip``: lets each clip's target track its own reference strike point (e.g. the
    backhand sits higher/forward at strike_phase 0.50, so a shared z<=1.05 box makes it unreachable).
    """
    block = _get(rk, "pos_range_per_clip")
    if block is None:
        return None
    by_id = {}
    for name in block:
        cid = _CLIP_NAME_TO_ID.get(str(name).lower())
        if cid is None:
            raise _OverrideError(
                f"racket.pos_range_per_clip: unknown clip name {name!r} (expected forehand/backhand)")
        axes = _get(block, name)

        def _r(ax):
            v = _get(axes, ax)
            if v is None:
                raise _OverrideError(f"racket.pos_range_per_clip[{name}]: missing '{ax}' [lo,hi] range")
            return (float(v[0]), float(v[1]))

        by_id[cid] = (_r("x"), _r("y"), _r("z"))
    return tuple(by_id[i] for i in range(len(by_id)))


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


def _apply_task_overrides(env_cfg, task, clip_name=None):
    """Apply cfg/task/<name>.yaml overrides (incl. the composed base/ groups) onto the env cfg.

    Returns the list of applied "attr=value" strings (logged by the caller). Keys absent from the
    YAML are left at the code default; keys present whose target attribute is missing RAISE (so a
    stale/shadowed cfg or a broken Hydra composition can never silently swallow an override).
    """
    applied = []

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
    _check_unknown_keys(plant, ("zero_joint_friction",), "task.plant")
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
    if rw is not None:
        R = env_cfg.rewards
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
        _pose_requested = _pose_weight_raw is not None or any(
            _get(rw, key) is not None for _, key, _, _ in _pose_fields
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
        for _name, _key in (
            ("joint_limit", "joint_limit_weight"),
            ("undesired_contacts", "undesired_contacts_weight"),
            ("pre_strike_foot_slip", "pre_strike_foot_slip_weight"),
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
            _vpc = _resolve_vel_range_per_clip(rk)
            if _vpc is not None:
                _require(hasattr(C, "racket_vel_range_per_clip"), "racket_target.racket_vel_range_per_clip")
                C.racket_vel_range_per_clip = _vpc
                applied.append(f"racket_target.racket_vel_range_per_clip={_vpc}")
            # Optional PER-CLIP position boxes (unified policy): forehand=clip 0, backhand=clip 1. Absent ->
            # keep the shared pos_*_range + |y|-sign box above (backward compatible). Lets each clip's target
            # track its own reference strike point (e.g. backhand z~1.2 at strike_phase 0.50).
            _ppc = _resolve_pos_range_per_clip(rk)
            if _ppc is not None:
                _require(hasattr(C, "racket_pos_range_per_clip"), "racket_target.racket_pos_range_per_clip")
                C.racket_pos_range_per_clip = _ppc
                applied.append(f"racket_target.racket_pos_range_per_clip={_ppc}")
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
            # face_command_obs (+4 actor dims: demanded normal (3) + zero-filled rho placeholder (1),
            # the contract-day 175 -> 179 layout): the obs groups were finalized in __post_init__
            # BEFORE overrides run, so setting env_cfg.face_command_obs here would be a silent
            # no-op — attach the ObsTerm directly (same term/tail position as the cfg switch).
            # The enabling experiment must update/remove actor_obs_contract in its YAML:
            # validate_actor_observation_contract stays a loud error on the frozen 175-D value.
            _fc_obs = _get(rk, "face_command_obs")
            if _fc_obs is not None and _as_bool(_fc_obs):
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

    # Domain randomization: behaviour preserved exactly (the pd_gain "absent/null -> disable" semantics
    # are intentional). Only logging is added; the hasattr guards stay so DR stays optional per task.
    dr = _get(task, "domain_rand")
    if dr is not None and hasattr(env_cfg, "events"):
        E = env_cfg.events
        mr = _get(dr, "link_mass_range")
        if mr is not None and hasattr(E, "randomize_link_mass"):
            E.randomize_link_mass.params["mass_distribution_params"] = (float(mr[0]), float(mr[1]))
            applied.append(f"events.randomize_link_mass.mass_distribution_params=({float(mr[0])}, {float(mr[1])})")
        if hasattr(E, "randomize_pd_gains"):
            pr = _get(dr, "pd_gain_range")
            if pr is None:
                E.randomize_pd_gains = None  # disable
                applied.append("events.randomize_pd_gains=None(disabled)")
            else:
                E.randomize_pd_gains.params["stiffness_distribution_params"] = (float(pr[0]), float(pr[1]))
                E.randomize_pd_gains.params["damping_distribution_params"] = (float(pr[0]), float(pr[1]))
                applied.append(f"events.randomize_pd_gains=({float(pr[0])}, {float(pr[1])})")

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
    if cfg.logger is not None:
        agent_cfg.logger = str(cfg.logger)
    if agent_cfg.logger in {"wandb", "neptune"} and cfg.log_project_name:
        agent_cfg.wandb_project = str(cfg.log_project_name)
        agent_cfg.neptune_project = str(cfg.log_project_name)

    # 3) reference motion clip(s), LOCAL-FIRST: motion_file=/motion_file_2= (or a local .npz path passed
    #    as registry_name/registry_name_2) skips WandB entirely (the documented no-WandB path — see
    #    run_training.md); otherwise the WandB registry is used.
    #    ONE clip = single-swing-type policy. TWO clips (forehand + backhand) = unified HITTER policy:
    #    MotionLoader concatenates them and clip_id selects which swing each env imitates. Order matters:
    #    clip 0 = forehand, clip 1 = backhand; it must match racket.strike_phase_per_clip.
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
        print(f"[train.py] UNIFIED multi-clip policy: clip0=forehand  clip1=backhand", flush=True)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

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
    _publish_lean_queue_binding_if_requested(cfg, log_dir)

    # 5) build env, wrap, run
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

    hard_contract = _build_training_hard_contract(env.unwrapped, actor_contract)
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
    hard_contract_sha256 = _sha256_file(contract_path)
    print(f"[train.py] hard training contract: {contract_path}", flush=True)
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
    contract_lineage_exact = ckpt is None and motion_kinematics_exact
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
        source_checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        prior_contract_path = os.path.join(os.path.dirname(ckpt), "params", "training_contract.json")
        allow_missing = bool(getattr(cfg, "checkpoint_allow_missing_contract", False))
        allow_mismatch = bool(getattr(cfg, "checkpoint_allow_contract_mismatch", False))
        if not os.path.isfile(prior_contract_path):
            message = (
                f"[train.py] checkpoint has no hard training contract: {prior_contract_path}. "
                "Tensor shapes cannot detect a changed HitterPure strike plane/box/face convention."
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
            if diffs and not allow_mismatch:
                raise RuntimeError(
                    "[train.py] checkpoint hard-contract mismatch; refusing a contaminated resume:\n  - "
                    + "\n  - ".join(diffs)
                    + "\nIf this is an intentional representation transfer rather than a resume, pass "
                      "checkpoint_allow_contract_mismatch=true and use a new run name."
                )
            if diffs:
                print("[train.py] WARNING: explicit hard-contract mismatch override:\n  - "
                      + "\n  - ".join(diffs), flush=True)
            else:
                print(f"[train.py] checkpoint hard contract MATCH: {prior_contract_path}", flush=True)
                try:
                    validate_schema3_contract(prior_contract)
                    require_checkpoint_contract_binding(
                        source_checkpoint,
                        schema=int(prior_contract["schema_version"]),
                        sha256=_sha256_file(prior_contract_path),
                    )
                except ValueError as exc:
                    print(
                        "[train.py] WARNING: source checkpoint contract is not exact-bound; "
                        f"descendants remain formal-ineligible: {exc}",
                        flush=True,
                    )
                else:
                    contract_lineage_exact = motion_kinematics_exact

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
    )
    runner.add_git_repo_to_log(__file__)

    # Resume / curriculum hand-off: load weights+optimizer from a prior checkpoint and CONTINUE (the
    # iteration counter resumes from the checkpoint). Config changes in the task YAML (e.g. a tighter
    # racket_velocity_std) take effect immediately on the loaded policy — no fresh restart needed.
    if ckpt is not None:
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

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if lateral_training_runtime is None:
        # Preserve the historical default-off control flow exactly.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        env.close()
    else:
        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
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
