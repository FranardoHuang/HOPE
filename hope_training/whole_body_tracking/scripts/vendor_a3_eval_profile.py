"""Fail-closed evaluation profiles for the vendor-authoritative A3 task.

The Agibot training authority has two evaluation semantics that must not be
collapsed into one:

``vendor_play_v1``
    Mirrors the vendor's Play table.  Startup plant randomization and interval
    pushes are disabled.  Policy observation corruption remains enabled and
    the episode-sampled actuator delay remains as trained because neither is
    listed among the disabled Play terms.

``deterministic_ranking_v1``
    Starts from vendor Play, then removes the remaining stochastic evaluation
    axes used by this repository's deterministic checkpoint ranking: policy
    observation corruption, actuator delay and physical reset-state noise.

Both profiles operate only on the exact immutable vendor task name.  The
helpers deliberately use duck typing so CPU-only source tests do not import
Isaac Lab or start Kit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


VENDOR_TASK_NAME = "HOPEPingPongActionBallA3VendorV1"
VENDOR_PLAY_PROFILE = "vendor_play_v1"
DETERMINISTIC_RANKING_PROFILE = "deterministic_ranking_v1"

_TRAIN_ONLY_EVENT_NAMES = (
    "physics_material",
    "add_joint_default_pos",
    "base_com",
    "randomize_link_mass",
    "randomize_pd_gains",
    "push_robot",
    "force_push",
    "force_push_sweep",
    "combined_push",
    "combined_push_sweep",
)
_REQUIRED_VENDOR_EVENTS = (
    "physics_material",
    "add_joint_default_pos",
    "base_com",
    "randomize_link_mass",
    "randomize_pd_gains",
    "push_robot",
)
_ROOT_RESET_EVENT_NAMES = ("reset_base", "reset_robot")


class VendorA3EvalProfileError(RuntimeError):
    """The vendor task cannot be made to match the requested eval profile."""


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    try:
        return getattr(node, key)
    except AttributeError:
        try:
            return node.get(key, default)
        except (AttributeError, TypeError):
            return default


def _zero_pair(value: Any, *, label: str) -> tuple[float, float]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise VendorA3EvalProfileError(f"{label} must be a two-element range") from exc
    if len(values) != 2:
        raise VendorA3EvalProfileError(f"{label} must be a two-element range")
    return (0.0, 0.0)


def _zero_axis_mapping(value: Any, *, label: str) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping):
        raise VendorA3EvalProfileError(f"{label} must be a mapping of axis ranges")
    return {str(axis): _zero_pair(bounds, label=f"{label}.{axis}") for axis, bounds in value.items()}


def _replace_term_params(term: Any, replacements: Mapping[str, Any], *, label: str) -> None:
    params = getattr(term, "params", None)
    if not isinstance(params, Mapping):
        raise VendorA3EvalProfileError(f"{label}.params must be a mapping")
    updated = dict(params)
    for key, value in replacements.items():
        if key not in updated:
            raise VendorA3EvalProfileError(f"{label}.params has no {key!r}")
        updated[key] = value
    term.params = updated


def _make_reset_event_nominal(term: Any, *, name: str) -> dict[str, Any]:
    """Remove state noise without deleting the reset writer itself."""

    params = getattr(term, "params", None)
    if not isinstance(params, Mapping):
        raise VendorA3EvalProfileError(f"events.{name}.params must be a mapping")
    replacements: dict[str, Any] = {}
    if "pose_range" in params:
        replacements["pose_range"] = _zero_axis_mapping(
            params["pose_range"], label=f"events.{name}.params.pose_range"
        )
    if "velocity_range" in params:
        velocity = params["velocity_range"]
        replacements["velocity_range"] = (
            _zero_axis_mapping(
                velocity, label=f"events.{name}.params.velocity_range"
            )
            if isinstance(velocity, Mapping)
            else _zero_pair(
                velocity, label=f"events.{name}.params.velocity_range"
            )
        )
    if name == "reset_robot_joints" and "position_range" in params:
        func_name = str(getattr(getattr(term, "func", None), "__name__", ""))
        # Isaac Lab's reset_joints_by_scale uses 1.0 as the nominal scale;
        # offset-style reset terms use a zero additive range.
        replacements["position_range"] = (
            (1.0, 1.0) if func_name == "reset_joints_by_scale" else (0.0, 0.0)
        )
    if not replacements:
        raise VendorA3EvalProfileError(
            f"events.{name} exposes no recognized reset-state ranges"
        )
    _replace_term_params(term, replacements, label=f"events.{name}")
    return replacements


def _zero_motion_reset_ranges(env_cfg: Any) -> list[str]:
    motion = _get(_get(env_cfg, "commands"), "motion")
    if motion is None:
        raise VendorA3EvalProfileError("vendor task has no commands.motion reset contract")
    changed: list[str] = []
    for name in ("pose_range", "velocity_range"):
        value = _get(motion, name)
        if value is None:
            raise VendorA3EvalProfileError(f"commands.motion.{name} is missing")
        setattr(
            motion,
            name,
            _zero_axis_mapping(value, label=f"commands.motion.{name}"),
        )
        changed.append(f"commands.motion.{name}")
    for name in ("joint_position_range", "stand_start_yaw_range"):
        value = _get(motion, name)
        if value is None:
            raise VendorA3EvalProfileError(f"commands.motion.{name} is missing")
        setattr(motion, name, _zero_pair(value, label=f"commands.motion.{name}"))
        changed.append(f"commands.motion.{name}")
    return changed


def apply_vendor_a3_eval_profile(
    env_cfg: Any,
    task: Any,
    *,
    profile: str,
) -> dict[str, Any] | None:
    """Apply one explicit vendor evaluation profile before ``gym.make``.

    Non-vendor tasks return ``None`` and remain byte-for-byte untouched.  The
    exact vendor task fails closed when its expected DR/action/observation
    surfaces are absent, preventing a renamed or partially composed task from
    silently claiming vendor Play or deterministic ranking semantics.
    """

    task_name = str(_get(task, "name", ""))
    if task_name != VENDOR_TASK_NAME:
        return None
    if profile not in (VENDOR_PLAY_PROFILE, DETERMINISTIC_RANKING_PROFILE):
        raise VendorA3EvalProfileError(f"unsupported vendor eval profile {profile!r}")

    events = _get(env_cfg, "events")
    if events is None:
        raise VendorA3EvalProfileError("vendor task has no events configuration")
    missing = [name for name in _REQUIRED_VENDOR_EVENTS if not hasattr(events, name)]
    if missing:
        raise VendorA3EvalProfileError(
            f"vendor task is missing required event slots: {missing}"
        )
    disabled_events: list[str] = []
    for name in _TRAIN_ONLY_EVENT_NAMES:
        if not hasattr(events, name):
            continue
        if getattr(events, name) is not None:
            setattr(events, name, None)
            disabled_events.append(name)

    actions = _get(_get(env_cfg, "actions"), "joint_pos")
    if actions is None:
        raise VendorA3EvalProfileError("vendor task has no actions.joint_pos")
    for name in (
        "control_step_action_delay_min",
        "control_step_action_delay_max",
    ):
        if not hasattr(actions, name):
            raise VendorA3EvalProfileError(f"actions.joint_pos.{name} is missing")

    policy = _get(_get(env_cfg, "observations"), "policy")
    if policy is None or not hasattr(policy, "enable_corruption"):
        raise VendorA3EvalProfileError(
            "vendor task has no observations.policy.enable_corruption"
        )

    # The vendor Play table retains root reset but explicitly removes the
    # reset-time joint offset.  Keep the reset writer and make only its ranges
    # nominal; deleting the event would leave stale joint state across episodes.
    nominalized: dict[str, Any] = {}
    reset_joints = getattr(events, "reset_robot_joints", None)
    if reset_joints is not None:
        nominalized["reset_robot_joints"] = _make_reset_event_nominal(
            reset_joints, name="reset_robot_joints"
        )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hope.vendor_a3_eval_profile",
        "task_name": task_name,
        "profile": profile,
        "disabled_train_only_events": sorted(disabled_events),
        "observation_corruption_enabled": bool(policy.enable_corruption),
        "control_step_action_delay": [
            int(actions.control_step_action_delay_min),
            int(actions.control_step_action_delay_max),
        ],
        "root_reset_semantics": "vendor_play_retained",
        "joint_reset_semantics": "vendor_play_nominal_if_present",
        "nominalized_reset_events": nominalized,
        "zeroed_motion_reset_ranges": [],
    }

    if profile == VENDOR_PLAY_PROFILE:
        return receipt

    policy.enable_corruption = False
    actions.control_step_action_delay_min = 0
    actions.control_step_action_delay_max = 0
    for name in _ROOT_RESET_EVENT_NAMES:
        term = getattr(events, name, None)
        if term is not None:
            nominalized[name] = _make_reset_event_nominal(term, name=name)
    zeroed_motion = _zero_motion_reset_ranges(env_cfg)
    receipt.update(
        {
            "observation_corruption_enabled": False,
            "control_step_action_delay": [0, 0],
            "root_reset_semantics": "nominal_deterministic",
            "nominalized_reset_events": nominalized,
            "zeroed_motion_reset_ranges": zeroed_motion,
        }
    )
    return receipt
