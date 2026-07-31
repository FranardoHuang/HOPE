from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


_A3_COLLISION_PROXY_SOURCE_URDF_SHA256 = (
    "0d83529cf808e2e68036f8168bd8b7a1c9a97d9c536eb9a14981ea4105d6b9ae"
)
_A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256 = (
    "716487dfdf02a5973f78263f0ae8a09e4680c04159e57dbe20796b7825dbeb4d"
)
_A3_COLLISION_PROXY_RUNTIME_USD_FILES = (
    (
        ".asset_hash",
        "3816a1a4bbca423e575650b6d6065f5141a7c840b02dd30c72d4278a225ed499",
        32,
    ),
    (
        "config.yaml",
        "3e35ad4c3ef7c21a10ce413be3ce28777bb83afee4b63fc245b30bd59a9818c2",
        1689,
    ),
    (
        "configuration/model_base.usd",
        "8e521141bfee4274b8a2369d382cdd8aac9bb1cfcae5bfa480666a1935a7fb42",
        21882690,
    ),
    (
        "configuration/model_physics.usd",
        "5b5fc00b96566be295a0cd4eb6b0cd276e360d9cca189057cef452ad0bfc7981",
        11164,
    ),
    (
        "configuration/model_sensor.usd",
        "c76c5bdd9e9b5434d72b45c9001858a9c80363656272011ed50d1419149ca60a",
        682,
    ),
    (
        "model.usd",
        "1b3fecd7685cd98ca80de226fbf89985b77b8a8cfc6a36f18fcc22e65080693c",
        1636,
    ),
)


@lru_cache(maxsize=8)
def _verify_loaded_runtime_usd_bundle(
    model_usd_path: str,
) -> str:
    """Bind the guard to the exact six-file USD tree loaded by the articulation.

    The launcher already validates this ignored runtime asset before Kit starts.
    The pose guard repeats the check once at construction against
    ``asset.cfg.spawn.usd_path`` so the collision artifact cannot accidentally
    describe one USD while the live articulation uses another.  This function
    is cached by the resolved absolute ``model.usd`` path and never runs in the
    physics-step hot path.
    """

    if not isinstance(model_usd_path, str) or not model_usd_path:
        raise RuntimeError(
            "robot_hit_table pose guard requires the live articulation model.usd path"
        )
    configured = Path(model_usd_path).expanduser()
    if not configured.is_absolute() or configured.name != "model.usd":
        raise RuntimeError(
            "robot_hit_table live USD must be one absolute model.usd path"
        )
    try:
        model_path = configured.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "robot_hit_table live articulation USD cannot be resolved"
        ) from exc
    if (
        model_path != configured
        or configured.is_symlink()
        or not model_path.is_file()
    ):
        raise RuntimeError(
            "robot_hit_table live model.usd must be one real regular file"
        )
    bundle_root = model_path.parent
    if bundle_root.is_symlink() or bundle_root.resolve(strict=True) != bundle_root:
        raise RuntimeError(
            "robot_hit_table live USD bundle root must be one real directory"
        )
    expected_paths = {
        path for path, _sha256, _size in _A3_COLLISION_PROXY_RUNTIME_USD_FILES
    }
    observed_paths = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                "robot_hit_table live USD bundle must not contain symlinks"
            )
        if path.is_file():
            observed_paths.add(path.relative_to(bundle_root).as_posix())
    if observed_paths != expected_paths:
        raise RuntimeError(
            "robot_hit_table live USD bundle differs from the exact six-file pin"
        )
    entries = []
    for relative, expected_sha256, expected_size in (
        _A3_COLLISION_PROXY_RUNTIME_USD_FILES
    ):
        path = bundle_root / relative
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError(
                "robot_hit_table live USD bundle file differs from pin: "
                f"{relative}"
            )
        entries.append(
            {
                "path": relative,
                "sha256": actual_sha256,
                "size": len(payload),
            }
        )
    canonical_entries = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    tree_sha256 = hashlib.sha256(canonical_entries).hexdigest()
    if tree_sha256 != _A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256:
        raise RuntimeError(
            "robot_hit_table live USD bundle tree SHA differs from collision proxy"
        )
    return tree_sha256


def _validate_forbidden_zone_margins(
    margin_rad: float,
    margin_fraction: float,
) -> tuple[float, float]:
    """Validate the two explicit per-side joint-position safety margins.

    ``margin_rad`` is an absolute angular inset in radians for the A3's revolute joints.
    ``margin_fraction`` is an additional inset equal to that fraction of each joint's complete
    ``upper - lower`` travel, also applied independently at both ends.  They add; neither has a
    hidden default.  For example, ``margin_rad=0.02, margin_fraction=0.05`` makes the allowed open
    interval ``(lower + 0.02 + 0.05*travel, upper - 0.02 - 0.05*travel)``.
    """

    if (
        isinstance(margin_rad, bool)
        or isinstance(margin_fraction, bool)
        or not isinstance(margin_rad, Real)
        or not isinstance(margin_fraction, Real)
    ):
        raise ValueError(
            "joint forbidden-zone margins must be finite numeric values, not booleans"
        )
    absolute = float(margin_rad)
    fraction = float(margin_fraction)
    if not math.isfinite(absolute) or absolute < 0.0:
        raise ValueError(
            "joint forbidden-zone margin_rad must be finite and >= 0 radians"
        )
    if not math.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise ValueError(
            "joint forbidden-zone margin_fraction must be finite and in [0, 0.5)"
        )
    return absolute, fraction


def joint_position_forbidden_zone_per_joint(
    joint_position: torch.Tensor,
    joint_position_limits: torch.Tensor,
    *,
    margin_rad: float,
    margin_fraction: float,
) -> torch.Tensor:
    """Pure tensor kernel: classify forbidden joint-position targets/states.

    Parameters are deliberately explicit:

    * ``joint_position`` must be ``[num_envs, num_joints]`` in articulation order.
    * ``joint_position_limits`` must be the matching runtime envelope
      ``[num_envs, num_joints, 2]`` with lower/upper in the last axis.  A caller may pass
      ``soft_joint_pos_limits`` to use the same deploy envelope as the HOPE q_des clamp, or
      ``joint_pos_limits`` to use the articulation's hard envelope; this kernel never guesses.
    * ``margin_rad`` is the absolute per-side inset in radians.
    * ``margin_fraction`` is an additional per-side inset as a fraction of full joint travel.

    The remaining allowed interval is OPEN.  Reaching either exact inner edge is forbidden; the
    safety term is not allowed to normalize contact with a limit.  Non-finite positions or bounds,
    reversed/zero-width bounds, and margins that consume the interval all fail safe as forbidden.
    Shape/device/dtype disagreement raises rather than silently broadcasting or reordering joints.
    """

    absolute, fraction = _validate_forbidden_zone_margins(
        margin_rad, margin_fraction
    )
    if not torch.is_tensor(joint_position) or joint_position.ndim != 2:
        raise RuntimeError(
            "joint forbidden-zone position must be a tensor shaped [num_envs, num_joints]"
        )
    if not torch.is_tensor(joint_position_limits) or joint_position_limits.ndim != 3:
        raise RuntimeError(
            "joint forbidden-zone limits must be a tensor shaped [num_envs, num_joints, 2]"
        )
    expected_limits_shape = tuple(joint_position.shape) + (2,)
    if tuple(joint_position_limits.shape) != expected_limits_shape:
        raise RuntimeError(
            "joint forbidden-zone limits must exactly match position order/shape: "
            f"position={tuple(joint_position.shape)} limits={tuple(joint_position_limits.shape)}"
        )
    if not torch.is_floating_point(joint_position) or not torch.is_floating_point(
        joint_position_limits
    ):
        raise RuntimeError("joint forbidden-zone position and limits must be floating tensors")
    if (
        joint_position.device != joint_position_limits.device
        or joint_position.dtype != joint_position_limits.dtype
    ):
        raise RuntimeError(
            "joint forbidden-zone position and limits must have identical device and dtype"
        )

    lower = joint_position_limits[..., 0]
    upper = joint_position_limits[..., 1]
    travel = upper - lower
    inset = absolute + fraction * travel
    inner_lower = lower + inset
    inner_upper = upper - inset

    finite = (
        torch.isfinite(joint_position)
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & torch.isfinite(inset)
    )
    valid_interval = (travel > 0.0) & (inner_lower < inner_upper)
    return (
        ~finite
        | ~valid_interval
        | (joint_position <= inner_lower)
        | (joint_position >= inner_upper)
    )


def joint_position_forbidden_zone_mask(
    joint_position: torch.Tensor,
    joint_position_limits: torch.Tensor,
    *,
    margin_rad: float,
    margin_fraction: float,
) -> torch.Tensor:
    """Reduce :func:`joint_position_forbidden_zone_per_joint` to one bit per environment."""

    return torch.any(
        joint_position_forbidden_zone_per_joint(
            joint_position,
            joint_position_limits,
            margin_rad=margin_rad,
            margin_fraction=margin_fraction,
        ),
        dim=1,
    )


def _identity_joint_ids(raw_ids: object, joint_count: int, context: str) -> list[int]:
    """Resolve a joint selection and require the complete articulation identity order."""

    if isinstance(raw_ids, slice):
        joint_ids = list(range(joint_count))[raw_ids]
    else:
        if torch.is_tensor(raw_ids):
            if (
                raw_ids.ndim != 1
                or raw_ids.dtype == torch.bool
                or torch.is_floating_point(raw_ids)
            ):
                raise RuntimeError(
                    f"{context} requires one-dimensional integer joint_ids"
                )
            raw_ids = raw_ids.tolist()
        try:
            selected = list(raw_ids)  # type: ignore[arg-type]
        except TypeError as exc:
            raise RuntimeError(
                f"{context} requires complete identity-order joint_ids"
            ) from exc
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in selected
        ):
            raise RuntimeError(f"{context} requires integer joint_ids")
        joint_ids = [int(value) for value in selected]
    if joint_ids != list(range(joint_count)):
        raise RuntimeError(
            f"{context} requires complete articulation identity order; got joint_ids={joint_ids}"
        )
    return joint_ids


def _runtime_joint_names(asset: Articulation, context: str) -> list[str]:
    """Read and validate the articulation's authoritative runtime joint order."""

    data = asset.data
    names = list(getattr(data, "joint_names", getattr(asset, "joint_names", ())))
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError(f"{context} requires non-empty runtime articulation joint names")
    if len(set(names)) != len(names):
        raise RuntimeError(f"{context} requires unique runtime articulation joint names")
    return names


def _runtime_joint_position_limits(
    data: object,
    *,
    limit_source: str,
    expected_shape: tuple[int, int],
    context: str,
) -> torch.Tensor:
    """Resolve an explicitly named runtime limit envelope without broadcasting."""

    allowed = ("soft_joint_pos_limits", "joint_pos_limits")
    if limit_source not in allowed:
        raise ValueError(
            f"{context} limit_source must be one of {allowed}; got {limit_source!r}"
        )
    limits = getattr(data, limit_source, None)
    if not torch.is_tensor(limits):
        raise RuntimeError(f"{context} requires runtime articulation {limit_source}")
    if tuple(limits.shape) != expected_shape + (2,):
        raise RuntimeError(
            f"{context} requires {limit_source} shaped "
            f"[num_envs, num_joints, 2]={expected_shape + (2,)}; got {tuple(limits.shape)}"
        )
    return limits


def pre_clamp_qdes_forbidden_zone(
    env: ManagerBasedRLEnv,
    action_name: str,
    limit_source: str,
    margin_rad: float,
    margin_fraction: float,
) -> torch.Tensor:
    """Terminate a valid row whose affine q_des request is a hard-safety event.

    This reads :class:`ClampedJointPositionAction`'s current *pre-clamp* deploy-space target, so
    the deploy clamp cannot hide an extreme request.  Legacy mode terminates a finite forbidden
    request.  In the explicit ActionBall projection mode, a finite request is constrained to the
    target envelope and shaped by projection distance instead; only NaN/Inf remains owned by this
    q_des term.  Predicted crossings still activate the action term's finite brake target without
    resetting the episode, while realized or substep hard-edge events remain terminal through
    :func:`actual_joint_position_forbidden_zone`.  Invalid rows immediately after reset return
    ``False`` until the first real action is processed.  ``limit_source`` and both margins are
    required.
    """

    context = "pre_clamp_qdes_forbidden_zone"
    action = env.action_manager.get_term(action_name)
    post_step_readback = getattr(
        action, "finalize_joint_safety_post_step_readback", None
    )
    if not callable(post_step_readback):
        raise RuntimeError(
            f"{context} requires the joint action's post-step safety readback hook"
        )
    post_step_readback()
    qdes = getattr(action, "pre_clamp_qdes", None)
    valid = getattr(action, "pre_clamp_qdes_valid", None)
    asset = getattr(action, "_asset", None)
    if asset is None or getattr(asset, "data", None) is None:
        raise RuntimeError(f"{context} requires the action term's runtime articulation")
    names = _runtime_joint_names(asset, context)
    action_names = list(getattr(action, "_joint_names", ()))
    if action_names != names:
        raise RuntimeError(
            f"{context} requires action/articulation identity joint-name order"
        )
    _identity_joint_ids(getattr(action, "_joint_ids", None), len(names), context)
    expected_shape = (int(env.num_envs), len(names))
    if not torch.is_tensor(qdes) or tuple(qdes.shape) != expected_shape:
        raise RuntimeError(
            f"{context} requires pre_clamp_qdes shaped {expected_shape}"
        )
    if (
        not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != (expected_shape[0],)
        or valid.device != qdes.device
    ):
        raise RuntimeError(
            f"{context} requires a same-device bool validity mask shaped "
            f"[num_envs]={expected_shape[0]}"
        )
    limits = _runtime_joint_position_limits(
        asset.data,
        limit_source=limit_source,
        expected_shape=expected_shape,
        context=context,
    )
    violation = joint_position_forbidden_zone_mask(
        qdes,
        limits,
        margin_rad=margin_rad,
        margin_fraction=margin_fraction,
    )
    pre_apply_latch = getattr(action, "pre_apply_joint_safety_latch", None)
    if (
        not torch.is_tensor(pre_apply_latch)
        or pre_apply_latch.dtype != torch.bool
        or tuple(pre_apply_latch.shape) != (expected_shape[0],)
        or pre_apply_latch.device != qdes.device
    ):
        raise RuntimeError(
            f"{context} requires a same-device bool pre-apply safety latch shaped "
            f"[num_envs]={expected_shape[0]}"
        )
    finite_projection_enabled = getattr(
        action, "finite_preclamp_qdes_projection_enabled", False
    )
    if type(finite_projection_enabled) is not bool:
        raise RuntimeError(
            f"{context} requires an exact boolean finite-projection mode"
        )
    if finite_projection_enabled:
        # In the ActionBall constrained-action mode, a finite affine request is never allowed to
        # reach the drive outside the already-validated target envelope.  Treating that proposal as
        # terminal would throw away the transition that carries its projection-distance penalty and
        # recreate the one-step reset wall.  A predicted crossing already selects the finite brake
        # target in the action term; resetting here defeats that recovery path.  Non-finite policy
        # output is still terminal.  Realized/substep hard-edge events are independently owned by
        # actual_joint_position_forbidden_zone, including sticky evidence after a safe bounce.
        nonfinite_request = torch.any(~torch.isfinite(qdes), dim=1)
        return valid & nonfinite_request
    return valid & (violation | pre_apply_latch)


def actual_joint_position_forbidden_zone(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    limit_source: str,
    margin_rad: float,
    margin_fraction: float,
) -> torch.Tensor:
    """Terminate only at the physical hard edge; retain the inset band as diagnostic evidence.

    The continuous actual-q barrier owns recoverable proximity to a limit.  Turning the
    two-percent inner band into an immediate reset starves PPO of the transition that can teach
    recovery, and conflates a soft constraint with a mechanical hard-edge violation.  This term
    still requires the inset margins so diagnostic runs can report the exact joint/side occupancy,
    but the Done bit is reserved for non-finite/invalid state or a current/substep raw hard edge.
    """

    context = "actual_joint_position_forbidden_zone"
    action = env.action_manager.get_term("joint_pos")
    post_step_readback = getattr(
        action, "finalize_joint_safety_post_step_readback", None
    )
    if not callable(post_step_readback):
        raise RuntimeError(
            f"{context} requires the joint action's post-step safety readback hook"
        )
    post_step_readback()
    asset: Articulation = env.scene[asset_cfg.name]
    names = _runtime_joint_names(asset, context)
    _identity_joint_ids(
        getattr(asset_cfg, "joint_ids", None), len(names), context
    )
    expected_shape = (int(env.num_envs), len(names))
    joint_pos = getattr(asset.data, "joint_pos", None)
    if not torch.is_tensor(joint_pos) or tuple(joint_pos.shape) != expected_shape:
        raise RuntimeError(
            f"{context} requires runtime joint_pos shaped {expected_shape}"
        )
    limits = _runtime_joint_position_limits(
        asset.data,
        limit_source=limit_source,
        expected_shape=expected_shape,
        context=context,
    )
    current_violation_per_joint = joint_position_forbidden_zone_per_joint(
        joint_pos,
        limits,
        margin_rad=margin_rad,
        margin_fraction=margin_fraction,
    )
    current_violation = torch.any(current_violation_per_joint, dim=1)
    substep_actual_latch = getattr(
        action, "physics_substep_actual_hard_edge_latch", None
    )
    if (
        not torch.is_tensor(substep_actual_latch)
        or substep_actual_latch.dtype != torch.bool
        or tuple(substep_actual_latch.shape) != (expected_shape[0],)
        or substep_actual_latch.device != joint_pos.device
    ):
        raise RuntimeError(
            f"{context} requires a same-device bool substep actual-hard-edge latch "
            f"shaped [num_envs]={expected_shape[0]}"
        )
    lower = limits[..., 0]
    upper = limits[..., 1]
    hard_comparable = (
        torch.isfinite(joint_pos)
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & upper.gt(lower)
    )
    current_hard_per_joint = (
        ~hard_comparable
        | joint_pos.le(lower)
        | joint_pos.ge(upper)
    )
    hard_terminal = (
        torch.any(current_hard_per_joint, dim=1) | substep_actual_latch
    )
    observed_event = current_violation | hard_terminal
    diagnostic_enabled = getattr(
        action, "actual_joint_forbidden_diagnostic_enabled", False
    )
    if type(diagnostic_enabled) is not bool:
        raise RuntimeError(
            f"{context} requires an exact boolean diagnostic-enabled flag"
        )
    if diagnostic_enabled:
        diagnostic_recorder = getattr(
            action, "record_actual_joint_forbidden_diagnostic", None
        )
        if not callable(diagnostic_recorder):
            raise RuntimeError(
                f"{context} diagnostic mode requires an attribution recorder"
            )
        absolute, fraction = _validate_forbidden_zone_margins(
            margin_rad, margin_fraction
        )
        travel = upper - lower
        inset = absolute + fraction * travel
        inner_lower = lower + inset
        inner_upper = upper - inset
        finite = (
            torch.isfinite(joint_pos)
            & torch.isfinite(lower)
            & torch.isfinite(upper)
            & torch.isfinite(inset)
        )
        valid_interval = (travel > 0.0) & (inner_lower < inner_upper)
        comparable = finite & valid_interval
        diagnostic_recorder(
            current_lower=(
                current_violation_per_joint
                & comparable
                & joint_pos.le(inner_lower)
            ),
            current_upper=(
                current_violation_per_joint
                & comparable
                & joint_pos.ge(inner_upper)
            ),
            current_nonfinite_or_invalid=(
                current_violation_per_joint & ~comparable
            ),
            observed_event=observed_event,
            hard_terminal=hard_terminal,
            episode_age=env.episode_length_buf,
        )
    return hard_terminal


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def align_body_ids(
    sensor_names: list[str], asset_names: list[str],
    sensor_ids: list[int], asset_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Pair a contact-sensor body selection with an articulation body selection BY NAME.

    A ``ContactSensor``'s body order comes from the PhysX rigid-body view built over the matched
    prims; an ``Articulation``'s body order comes from the articulation's own link order.  They
    are not guaranteed to agree, and if they silently disagree the termination would read one
    body's contact force against another body's position — a bug that produces plausible-looking
    numbers forever.  So the pairing is done on names and the intersection is returned in a fixed
    (sensor) order.  Bodies the articulation does not expose are dropped, not guessed.
    """
    want = {sensor_names[i]: i for i in sensor_ids}
    have = {asset_names[i]: i for i in asset_ids}
    common = [n for n in (sensor_names[i] for i in sensor_ids) if n in have]
    if not common:
        raise RuntimeError(
            "robot_hit_table: sensor and articulation body selections do not overlap; "
            f"sensor={sorted(want)} asset={sorted(have)}"
        )
    return [want[n] for n in common], [have[n] for n in common]


def align_body_ids_in_expected_order(
    sensor_names: list[str],
    asset_names: list[str],
    sensor_ids: list[int],
    asset_ids: list[int],
    expected_names: tuple[str, ...] | list[str],
) -> tuple[list[int], list[int]]:
    """Align two complete selections in one explicit, backend-independent order.

    PhysX is free to enumerate a ``ContactSensor`` view differently from the
    articulation.  The table guard's collision-component owners and racket
    index are defined in the reviewed A3 order, so merely proving the two
    runtime views agree with each other is insufficient: both streams must be
    reordered to that explicit contract before the tensor kernel consumes them.
    """

    expected = tuple(str(name) for name in expected_names)
    if not expected or len(set(expected)) != len(expected):
        raise RuntimeError(
            "robot_hit_table expected body order must be non-empty and unique"
        )
    selected_sensor_names = [sensor_names[index] for index in sensor_ids]
    selected_asset_names = [asset_names[index] for index in asset_ids]
    if (
        len(set(selected_sensor_names)) != len(selected_sensor_names)
        or len(set(selected_asset_names)) != len(selected_asset_names)
        or set(selected_sensor_names) != set(expected)
        or set(selected_asset_names) != set(expected)
    ):
        raise RuntimeError(
            "robot_hit_table runtime body selections must exactly cover the "
            "reviewed 32-body A3 set"
        )
    sensor_by_name = {
        sensor_names[index]: index for index in sensor_ids
    }
    asset_by_name = {
        asset_names[index]: index for index in asset_ids
    }
    return (
        [sensor_by_name[name] for name in expected],
        [asset_by_name[name] for name in expected],
    )


def _aligned_body_ids(sensor, asset, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """``align_body_ids`` memoized on the sensor (the selections are fixed for a run)."""
    key = f"_hope_table_hit_ids__{sensor_cfg.name}__{asset_cfg.name}"
    cached = getattr(sensor, key, None)
    if cached is not None:
        return cached
    s_ids = sensor_cfg.body_ids
    a_ids = asset_cfg.body_ids
    s_ids = list(range(len(sensor.body_names))) if not isinstance(s_ids, list) else list(s_ids)
    a_ids = list(range(len(asset.body_names))) if not isinstance(a_ids, list) else list(a_ids)
    pair = align_body_ids(list(sensor.body_names), list(asset.body_names), s_ids, a_ids)
    setattr(sensor, key, pair)
    return pair


def _aligned_body_ids_in_expected_order(
    sensor,
    asset,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    expected_names: tuple[str, ...] | list[str],
):
    """Memoize explicit-order alignment for the ActionBall full-table guard."""

    key = (
        f"_hope_table_hit_expected_ids__{sensor_cfg.name}__"
        f"{asset_cfg.name}"
    )
    expected = tuple(str(name) for name in expected_names)
    cached = getattr(sensor, key, None)
    if cached is not None:
        cached_expected, pair = cached
        if cached_expected != expected:
            raise RuntimeError(
                "robot_hit_table expected body order changed during one run"
            )
        return pair
    sensor_ids = sensor_cfg.body_ids
    asset_ids = asset_cfg.body_ids
    sensor_ids = (
        list(sensor_ids)
        if isinstance(sensor_ids, (list, tuple))
        else list(range(len(sensor.body_names)))
    )
    asset_ids = (
        list(asset_ids)
        if isinstance(asset_ids, (list, tuple))
        else list(range(len(asset.body_names)))
    )
    pair = align_body_ids_in_expected_order(
        list(sensor.body_names),
        list(asset.body_names),
        sensor_ids,
        asset_ids,
        expected,
    )
    setattr(sensor, key, (expected, pair))
    return pair


def _asset_body_ids_in_expected_order(
    asset,
    asset_cfg: SceneEntityCfg,
    expected_names: tuple[str, ...] | list[str],
) -> list[int]:
    """Memoize one articulation selection in explicit reviewed A3 order."""

    key = f"_hope_table_pose_expected_ids__{asset_cfg.name}"
    expected = tuple(str(name) for name in expected_names)
    cached = getattr(asset, key, None)
    if cached is not None:
        cached_expected, body_ids = cached
        if cached_expected != expected:
            raise RuntimeError(
                "robot_hit_table expected body order changed during one run"
            )
        return body_ids
    live_names = list(asset.body_names)
    if (
        len(live_names) != len(expected)
        or len(set(live_names)) != len(live_names)
        or set(live_names) != set(expected)
    ):
        raise RuntimeError(
            "robot_hit_table live articulation body_names must be an exact "
            "name-bijective copy of the reviewed 32-body A3 set"
        )
    selected_ids = asset_cfg.body_ids
    selected_ids = (
        list(selected_ids)
        if isinstance(selected_ids, (list, tuple))
        else list(range(len(asset.body_names)))
    )
    selected_names = [asset.body_names[index] for index in selected_ids]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or len(selected_names) != len(live_names)
        or len(set(selected_names)) != len(selected_names)
        or set(selected_names) != set(expected)
    ):
        raise RuntimeError(
            "robot_hit_table articulation selection must exactly cover the "
            "reviewed 32-body A3 set"
        )
    by_name = {
        asset.body_names[index]: index for index in selected_ids
    }
    body_ids = [by_name[name] for name in expected]
    setattr(asset, key, (expected, body_ids))
    return body_ids


def table_hit_mask(
    body_pos_w: torch.Tensor,
    contact_force_w: torch.Tensor,
    env_origins: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Pure tensor kernel behind :func:`robot_hit_table`.  No env, no Isaac — so it is testable.

    ``body_pos_w`` (E, B, 3) and ``contact_force_w`` (E, B, 3) are WORLD-frame; ``env_origins``
    (E, 3) converts them to the env-local frame the table box is expressed in.  ``aabb_lo`` /
    ``aabb_hi`` may be one ``[3]`` box or an assembly ``[obstacles, 3]``.  A body counts as a
    strike when it is BOTH pushing (|f| > threshold) AND geometrically inside any box.
    """

    p_local = body_pos_w - env_origins[:, None, :]
    if aabb_lo.ndim == 1:
        inside = torch.all(
            (p_local >= aabb_lo) & (p_local <= aabb_hi), dim=-1
        )
    elif aabb_lo.ndim == 2:
        inside_per_obstacle = torch.all(
            (p_local[:, :, None, :] >= aabb_lo[None, None, :, :])
            & (p_local[:, :, None, :] <= aabb_hi[None, None, :, :]),
            dim=-1,
        )
        inside = torch.any(inside_per_obstacle, dim=-1)
    else:
        raise ValueError("table-hit AABBs must be shaped [3] or [obstacles, 3]")
    safe_force = torch.nan_to_num(
        contact_force_w,
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("-inf"),
    )
    pushing = torch.norm(safe_force, dim=-1) > float(force_threshold)
    return torch.any(inside & pushing, dim=-1)


def _quat_rotate_wxyz(
    quaternion_wxyz: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors by WXYZ quaternions without importing a second geometry stack."""

    q_vec = quaternion_wxyz[..., 1:4]
    q_w = quaternion_wxyz[..., 0:1]
    twice_cross = 2.0 * torch.cross(q_vec, vector, dim=-1)
    return (
        vector
        + q_w * twice_cross
        + torch.cross(q_vec, twice_cross, dim=-1)
    )


def _strict_json_object(pairs):
    """Reject duplicate JSON keys instead of accepting the last spelling."""

    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_collision_proxy_artifact_path(raw_path: str) -> Path:
    """Resolve one tracked repo-relative artifact without silently picking a shadow copy."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(
            "robot_hit_table requires a non-empty collision proxy artifact path"
        )
    configured = Path(raw_path).expanduser()
    candidates = [configured] if configured.is_absolute() else [
        Path.cwd() / configured,
        *(
            parent / configured
            for parent in Path(__file__).resolve().parents
        ),
    ]
    matches = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if not matches:
        raise RuntimeError(
            "robot_hit_table collision proxy artifact does not exist: "
            f"{raw_path!r}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            "robot_hit_table collision proxy artifact path is ambiguous: "
            f"{raw_path!r} -> {[str(path) for path in matches]}"
        )
    return matches[0]


@lru_cache(maxsize=16)
def _load_table_collision_proxy_artifact(
    raw_path: str,
    expected_file_sha256: str,
    expected_body_names: tuple[str, ...],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[tuple[float, float, float], ...], ...],
]:
    """Load and fail-closed validate the run-static A3 collision-component OBBs."""

    if not _is_lower_sha256(expected_file_sha256):
        raise RuntimeError(
            "robot_hit_table collision proxy SHA must be one lowercase sha256"
        )
    if (
        len(expected_body_names) != 32
        or len(set(expected_body_names)) != 32
        or any(not isinstance(name, str) or not name for name in expected_body_names)
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy requires the exact ordered "
            "32-body A3 contract"
        )
    artifact_path = _resolve_collision_proxy_artifact_path(raw_path)
    payload = artifact_path.read_bytes()
    actual_file_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise RuntimeError(
            "robot_hit_table collision proxy artifact SHA mismatch: "
            f"expected={expected_file_sha256} actual={actual_file_sha256}"
        )
    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "robot_hit_table collision proxy artifact is not strict ASCII JSON"
        ) from exc
    if not isinstance(document, dict):
        raise RuntimeError(
            "robot_hit_table collision proxy artifact must be one JSON object"
        )
    if (
        document.get("schema_version") != 1
        or document.get("artifact_type")
        != "a3_table_collision_component_obb_v1"
        or tuple(document.get("body_order", ())) != expected_body_names
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy schema/body order does not match "
            "the exact A3 contract"
        )
    source_urdf = document.get("source_urdf")
    runtime_usd = document.get("runtime_usd_bundle")
    expected_runtime_files = [
        {"path": path, "sha256": sha256, "size": size}
        for path, sha256, size in _A3_COLLISION_PROXY_RUNTIME_USD_FILES
    ]
    if (
        not isinstance(source_urdf, dict)
        or source_urdf.get("sha256")
        != _A3_COLLISION_PROXY_SOURCE_URDF_SHA256
        or not isinstance(runtime_usd, dict)
        or runtime_usd.get("bundle_tree_sha256")
        != _A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256
        or runtime_usd.get("file_count") != 6
        or runtime_usd.get("total_file_bytes") != 21897893
        or runtime_usd.get("symlinks_forbidden") is not True
        or runtime_usd.get("files") != expected_runtime_files
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy does not bind the reviewed vendor "
            "URDF and exact six-file Pod runtime USD bundle"
        )
    content_sha256 = document.get("content_sha256")
    if not _is_lower_sha256(content_sha256):
        raise RuntimeError(
            "robot_hit_table collision proxy content SHA is malformed"
        )
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    canonical_unsigned = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    if hashlib.sha256(canonical_unsigned).hexdigest() != content_sha256:
        raise RuntimeError(
            "robot_hit_table collision proxy content SHA mismatch"
        )
    components = document.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 43
        or document.get("component_count") != len(components)
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy component count is malformed"
        )

    body_index = {
        body_name: index
        for index, body_name in enumerate(expected_body_names)
    }
    component_ids = []
    owner_indices = []
    centers = []
    half_axes = []
    owner_coverage = set()
    for component in components:
        if not isinstance(component, dict):
            raise RuntimeError(
                "robot_hit_table collision proxy component is malformed"
            )
        component_id = component.get("component_id")
        owner_name = component.get("owner_body_name")
        center = component.get("local_center_owner_m")
        axes = component.get("local_half_axes_owner_m")
        mesh_sha = component.get("mesh_sha256")
        if (
            not isinstance(component_id, str)
            or not component_id
            or owner_name not in body_index
            or not _is_lower_sha256(mesh_sha)
            or not isinstance(center, list)
            or len(center) != 3
            or not isinstance(axes, list)
            or len(axes) != 3
            or any(not isinstance(axis, list) or len(axis) != 3 for axis in axes)
        ):
            raise RuntimeError(
                "robot_hit_table collision proxy component metadata is malformed"
            )
        try:
            center_values = tuple(float(value) for value in center)
            axis_values = tuple(
                tuple(float(value) for value in axis) for axis in axes
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "robot_hit_table collision proxy geometry is not numeric"
            ) from exc
        if (
            not all(math.isfinite(value) for value in center_values)
            or not all(
                math.isfinite(value)
                for axis in axis_values
                for value in axis
            )
            or any(
                sum(value * value for value in axis) <= 0.0
                for axis in axis_values
            )
        ):
            raise RuntimeError(
                "robot_hit_table collision proxy geometry must be finite "
                "with three positive half axes"
            )
        component_ids.append(component_id)
        owner_indices.append(body_index[owner_name])
        centers.append(center_values)
        half_axes.append(axis_values)
        owner_coverage.add(owner_name)
    if (
        component_ids != sorted(component_ids)
        or len(set(component_ids)) != len(component_ids)
        or owner_coverage != set(expected_body_names)
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy components must be unique, "
            "canonically ordered, and cover every A3 rigid body"
        )
    return tuple(owner_indices), tuple(centers), tuple(half_axes)


def _squared_distance_to_aabbs(
    point_xyz: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
) -> torch.Tensor:
    """Squared point-to-AABB distance without an ``[..., boxes, xyz]`` temporary.

    ``point_xyz`` is ``[..., 3]`` and the boxes are ``[boxes, 3]``.  The dense
    formulation materializes three ``[..., boxes, 3]`` intermediates.  Iterating
    over the fixed three spatial axes preserves the same arithmetic order while
    keeping every work buffer at ``[..., boxes]`` rather than
    ``[..., boxes, 3]``.  Inputs are never mutated.
    """

    distance_sq = None
    for axis in range(3):
        coordinate = point_xyz[..., axis].unsqueeze(-1)
        outside_axis = torch.maximum(
            aabb_lo[:, axis] - coordinate,
            coordinate - aabb_hi[:, axis],
        )
        outside_axis.clamp_min_(0.0)
        outside_axis.square_()
        if distance_sq is None:
            distance_sq = outside_axis
        else:
            distance_sq.add_(outside_axis)
    assert distance_sq is not None
    return distance_sq


def geometric_table_contact_hit_mask(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
    component_body_indices: torch.Tensor,
    component_local_center_m: torch.Tensor,
    component_local_half_axes_m: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
    *,
    racket_body_index: int,
    racket_blade_center_offset_wrist_m: torch.Tensor,
    racket_blade_local_half_axes_m: torch.Tensor,
) -> torch.Tensor:
    """Conservative ActionBall robot/table overlap from live articulation pose.

    Every A3 collision component uses its materialized owner-frame OBB.  Each live OBB is
    conservatively broadened to a world AABB before comparison with the five table AABBs: this may
    reject a near-corner brush but cannot miss an overlap.  The merged wrist/racket body
    additionally keeps the reviewed blade OBB as an independent channel.  Full ActionBall
    deliberately treats this as a pose keep-out rather than reading the expensive whole-body
    ``ContactSensor`` every physics substep; the physical kinematic table colliders remain
    installed separately.  Non-finite runtime pose data fails safe per environment.  Component
    geometry, AABBs and blade geometry are run-static tensors validated and cached before this hot
    kernel is entered.
    """

    if (
        body_pos_w.ndim != 3
        or body_pos_w.shape[-1] != 3
        or body_quat_w.shape != (*body_pos_w.shape[:-1], 4)
        or env_origins.shape != (body_pos_w.shape[0], 3)
        or component_body_indices.ndim != 1
        or component_local_center_m.shape
        != (component_body_indices.shape[0], 3)
        or component_local_half_axes_m.shape
        != (component_body_indices.shape[0], 3, 3)
        or component_body_indices.shape[0] <= 0
        or aabb_lo.ndim != 2
        or aabb_lo.shape[-1] != 3
        or aabb_hi.shape != aabb_lo.shape
        or racket_blade_center_offset_wrist_m.shape != (3,)
        or racket_blade_local_half_axes_m.shape != (3, 3)
    ):
        raise RuntimeError(
            "geometric table contact requires body pose [E,B,*], origins [E,3], "
            "component owners [C], component centers [C,3], component half axes "
            "[C,3,3], assembly AABBs [O,3], a racket offset [3], and cached "
            "racket half-axes [3,3]"
        )
    tensors = (
        body_quat_w,
        env_origins,
        component_local_center_m,
        component_local_half_axes_m,
        aabb_lo,
        aabb_hi,
        racket_blade_center_offset_wrist_m,
        racket_blade_local_half_axes_m,
    )
    if (
        not torch.is_floating_point(body_pos_w)
        or component_body_indices.dtype != torch.long
        or component_body_indices.device != body_pos_w.device
        or any(
            not torch.is_floating_point(value)
            or value.device != body_pos_w.device
            or value.dtype != body_pos_w.dtype
            for value in tensors
        )
    ):
        raise RuntimeError(
            "geometric table contact geometry must share one floating dtype/device "
            "and component owners must be same-device int64"
        )
    if (
        isinstance(racket_body_index, bool)
        or not isinstance(racket_body_index, Integral)
        or not 0 <= int(racket_body_index) < body_pos_w.shape[1]
    ):
        raise RuntimeError("racket_body_index is outside the selected A3 body order")
    return _geometric_table_contact_hit_mask_unchecked(
        body_pos_w,
        body_quat_w,
        env_origins,
        component_body_indices,
        component_local_center_m,
        component_local_half_axes_m,
        aabb_lo,
        aabb_hi,
        racket_body_index=int(racket_body_index),
        racket_blade_center_offset_wrist_m=(
            racket_blade_center_offset_wrist_m
        ),
        racket_blade_local_half_axes_m=(
            racket_blade_local_half_axes_m
        ),
    )


def _geometric_table_contact_hit_mask_unchecked(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
    component_body_indices: torch.Tensor,
    component_local_center_m: torch.Tensor,
    component_local_half_axes_m: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
    *,
    racket_body_index: int,
    racket_blade_center_offset_wrist_m: torch.Tensor,
    racket_blade_local_half_axes_m: torch.Tensor,
) -> torch.Tensor:
    """Tensor-only pose guard used after one-time construction validation."""

    p_local = body_pos_w - env_origins[:, None, :]

    body_quat_norm_sq = torch.sum(
        body_quat_w * body_quat_w, dim=-1, keepdim=True
    )
    safe_body_quat = body_quat_w / torch.sqrt(
        torch.clamp(
            body_quat_norm_sq,
            min=torch.finfo(body_pos_w.dtype).tiny,
        )
    )
    component_quat = torch.index_select(
        safe_body_quat, 1, component_body_indices
    )
    component_owner_pos = torch.index_select(
        p_local, 1, component_body_indices
    )
    component_center = component_owner_pos + _quat_rotate_wxyz(
        component_quat,
        component_local_center_m.unsqueeze(0).expand(
            body_pos_w.shape[0], -1, -1
        ),
    )
    component_world_aabb_half = torch.zeros_like(component_center)
    for local_axis in range(3):
        rotated_axis = _quat_rotate_wxyz(
            component_quat,
            component_local_half_axes_m[:, local_axis, :]
            .unsqueeze(0)
            .expand(body_pos_w.shape[0], -1, -1),
        )
        component_world_aabb_half.add_(torch.abs(rotated_axis))
    # Cover float64-artifact -> runtime-dtype conversion and quaternion arithmetic
    # without recreating the centimetre-scale false positives of the retired
    # uniform spheres.  One micrometre is below both PhysX contact offsets and the
    # table guard's configured margin.
    component_world_aabb_half.add_(1.0e-6)
    component_lo = component_center - component_world_aabb_half
    component_hi = component_center + component_world_aabb_half

    component_overlap = torch.ones(
        (
            body_pos_w.shape[0],
            component_body_indices.shape[0],
            aabb_lo.shape[0],
        ),
        device=body_pos_w.device,
        dtype=torch.bool,
    )
    for axis in range(3):
        component_overlap.logical_and_(
            (
                component_hi[..., axis, None]
                >= aabb_lo[None, None, :, axis]
            )
            & (
                component_lo[..., axis, None]
                <= aabb_hi[None, None, :, axis]
            )
        )
    body_hit = torch.any(component_overlap, dim=(1, 2))

    safe_quat = safe_body_quat[:, racket_body_index, :]
    blade_offset_w = _quat_rotate_wxyz(
        safe_quat, racket_blade_center_offset_wrist_m.expand_as(p_local[:, 0])
    )
    blade_center_local = (
        p_local[:, racket_body_index, :] + blade_offset_w
    )
    local_half_axes = racket_blade_local_half_axes_m.unsqueeze(0).expand(
        body_pos_w.shape[0], -1, -1
    )
    blade_quat = safe_quat[:, None, :].expand(-1, 3, -1)
    rotated_half_axes = _quat_rotate_wxyz(blade_quat, local_half_axes)
    blade_world_aabb_half = torch.sum(
        torch.abs(rotated_half_axes), dim=1
    )
    blade_lo = blade_center_local - blade_world_aabb_half
    blade_hi = blade_center_local + blade_world_aabb_half
    blade_overlap = torch.any(
        torch.all(
            (blade_hi[:, None, :] >= aabb_lo[None, :, :])
            & (blade_lo[:, None, :] <= aabb_hi[None, :, :]),
            dim=-1,
        ),
        dim=1,
    )
    racket_hit = blade_overlap

    invalid_runtime = (
        ~torch.isfinite(body_pos_w).all(dim=(1, 2))
        | ~torch.isfinite(body_quat_w).all(dim=(1, 2))
        | ~torch.isfinite(env_origins).all(dim=1)
        | ~(body_quat_norm_sq[..., 0] > 0.0).all(dim=1)
    )
    return body_hit | racket_hit | invalid_runtime


def filtered_contact_hit_mask(
    force_matrix_w: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Reduce a filtered contact-force matrix to one table-hit bit per environment.

    ``ContactSensorData.force_matrix_w`` is shaped ``[env, sensor body, filter expression, xyz]``
    in the pinned Isaac Lab implementation.  Legacy uses the right wrist as source and the table
    top as its one filter.  Full ActionBall does not call this helper: it intentionally avoids the
    pinned backend's broken/expensive many-filter matrix and uses
    :func:`geometric_table_contact_hit_mask`.

    Non-finite force data fails safe: it becomes an infinite force and ends the affected episode
    instead of silently turning a broken contact stream into ``False``.
    """
    safe_force = torch.nan_to_num(
        force_matrix_w, nan=float("inf"), posinf=float("inf"), neginf=float("-inf")
    )
    pushing = torch.norm(safe_force, dim=-1) > float(force_threshold)
    return torch.any(pushing.flatten(start_dim=1), dim=1)


class _PreparedRobotTablePoseGuard:
    """Run-static ActionBall pose guard with a tensor-only sampling call."""

    __slots__ = (
        "_asset",
        "_asset_body_indices",
        "_env_origins",
        "_component_indices",
        "_component_centers",
        "_component_half_axes",
        "_aabb_lo",
        "_aabb_hi",
        "_racket_index",
        "_blade_center",
        "_blade_local_half_axes",
        "runtime_usd_receipt",
    )

    def __init__(
        self,
        *,
        asset,
        asset_body_indices: torch.Tensor,
        env_origins: torch.Tensor,
        component_indices: torch.Tensor,
        component_centers: torch.Tensor,
        component_half_axes: torch.Tensor,
        aabb_lo: torch.Tensor,
        aabb_hi: torch.Tensor,
        racket_index: int,
        blade_center: torch.Tensor,
        blade_local_half_axes: torch.Tensor,
        runtime_usd_receipt: dict[str, object],
    ) -> None:
        self._asset = asset
        self._asset_body_indices = asset_body_indices
        self._env_origins = env_origins
        self._component_indices = component_indices
        self._component_centers = component_centers
        self._component_half_axes = component_half_axes
        self._aabb_lo = aabb_lo
        self._aabb_hi = aabb_hi
        self._racket_index = int(racket_index)
        self._blade_center = blade_center
        self._blade_local_half_axes = blade_local_half_axes
        self.runtime_usd_receipt = runtime_usd_receipt

    def __call__(self) -> torch.Tensor:
        # All names, paths, shapes, dtypes and static tensors were verified by
        # ``prepare_robot_table_pose_guard``.  Per physics substep this path
        # performs only tensor selection and the pose-overlap kernel.
        body_pos = torch.index_select(
            self._asset.data.body_pos_w, 1, self._asset_body_indices
        )
        body_quat = torch.index_select(
            self._asset.data.body_quat_w, 1, self._asset_body_indices
        )
        return _geometric_table_contact_hit_mask_unchecked(
            body_pos,
            body_quat,
            self._env_origins,
            self._component_indices,
            self._component_centers,
            self._component_half_axes,
            self._aabb_lo,
            self._aabb_hi,
            racket_body_index=self._racket_index,
            racket_blade_center_offset_wrist_m=self._blade_center,
            racket_blade_local_half_axes_m=self._blade_local_half_axes,
        )


def _live_articulation_model_usd_path(env, asset) -> str:
    """Resolve the actual configured articulation USD and reject split identity."""

    candidates = []
    asset_spawn = getattr(getattr(asset, "cfg", None), "spawn", None)
    asset_path = getattr(asset_spawn, "usd_path", None)
    if isinstance(asset_path, str) and asset_path:
        candidates.append(asset_path)
    scene_cfg = getattr(getattr(env, "cfg", None), "scene", None)
    robot_cfg = getattr(scene_cfg, "robot", None)
    scene_spawn = getattr(robot_cfg, "spawn", None)
    scene_path = getattr(scene_spawn, "usd_path", None)
    if isinstance(scene_path, str) and scene_path:
        candidates.append(scene_path)
    if not candidates:
        raise RuntimeError(
            "robot_hit_table cannot resolve the live articulation model.usd path"
        )
    resolved = []
    for candidate in candidates:
        try:
            resolved.append(str(Path(candidate).expanduser().resolve(strict=True)))
        except OSError as exc:
            raise RuntimeError(
                "robot_hit_table live articulation USD cannot be resolved"
            ) from exc
    if len(set(resolved)) != 1:
        raise RuntimeError(
            "robot_hit_table asset and scene configs disagree on live USD identity"
        )
    environment_path = os.environ.get("HOPE_AGIBOT_A3_USD_PATH")
    if environment_path:
        try:
            environment_resolved = str(
                Path(environment_path).expanduser().resolve(strict=True)
            )
        except OSError as exc:
            raise RuntimeError(
                "robot_hit_table HOPE_AGIBOT_A3_USD_PATH cannot be resolved"
            ) from exc
        if environment_resolved != resolved[0]:
            raise RuntimeError(
                "robot_hit_table live USD differs from launch environment pin"
            )
    return resolved[0]


def prepare_robot_table_pose_guard(
    env: ManagerBasedRLEnv,
    *,
    asset_cfg: SceneEntityCfg,
    near_x: float,
    surface_z: float,
    full_table_filtered_sensor_cfgs: tuple[SceneEntityCfg, ...]
    | list[SceneEntityCfg] = (),
    expected_full_table_source_prim_paths: tuple[str, ...]
    | list[str] = (),
    expected_full_robot_body_names: tuple[str, ...] | list[str] = (),
    margin: float = 0.02,
    keepout_floor_z: float = 0.0,
    collision_proxy_artifact_path: str = "",
    collision_proxy_artifact_sha256: str = "",
    racket_body_name: str = "right_wrist_yaw_Link",
    racket_blade_center_offset_wrist_m: tuple[float, float, float]
    | list[float] = (0.206194, 0.025474, 0.028020),
    racket_blade_half_extents_m: tuple[float, float, float]
    | list[float] = (0.082, 0.008, 0.082),
) -> _PreparedRobotTablePoseGuard:
    """Validate and materialize every static full-table guard input once."""

    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame

    if tuple(full_table_filtered_sensor_cfgs):
        raise RuntimeError(
            "robot_hit_table full assembly must not install or read "
            "pair-filtered contact sensors"
        )
    asset: Articulation = env.scene[asset_cfg.name]
    expected_names = tuple(expected_full_robot_body_names)
    if len(expected_names) != 32 or len(set(expected_names)) != 32:
        raise RuntimeError(
            "robot_hit_table full assembly requires one unique 32-body A3 contract"
        )
    source_paths = tuple(expected_full_table_source_prim_paths)
    if (
        len(source_paths) != 5
        or len(set(source_paths)) != 5
        or any(not isinstance(path, str) or not path for path in source_paths)
    ):
        raise RuntimeError(
            "robot_hit_table full assembly requires five unique table geometry source paths"
        )
    asset_ids = _asset_body_ids_in_expected_order(
        asset, asset_cfg, expected_names
    )
    body_pos_all = getattr(asset.data, "body_pos_w", None)
    body_quat_all = getattr(asset.data, "body_quat_w", None)
    env_origins = getattr(env.scene, "env_origins", None)
    if (
        body_pos_all is None
        or body_pos_all.ndim != 3
        or body_pos_all.shape[-1] != 3
        or body_quat_all is None
        or body_quat_all.ndim != 3
        or body_quat_all.shape[-1] != 4
        or body_pos_all.shape[:2] != body_quat_all.shape[:2]
        or body_pos_all.shape[0] != int(env.num_envs)
        or not torch.is_floating_point(body_pos_all)
        or not torch.is_floating_point(body_quat_all)
        or body_pos_all.device != body_quat_all.device
        or body_pos_all.dtype != body_quat_all.dtype
        or not torch.is_tensor(env_origins)
        or env_origins.shape != (int(env.num_envs), 3)
        or env_origins.device != body_pos_all.device
        or env_origins.dtype != body_pos_all.dtype
    ):
        raise RuntimeError(
            "robot_hit_table full assembly requires same-device/dtype "
            "body pose [env,body,*] and env_origins [env,3]"
        )
    if (
        not isinstance(racket_body_name, str)
        or racket_body_name not in expected_names
    ):
        raise RuntimeError(
            "robot_hit_table racket body is absent from the exact A3 body order"
        )
    live_model_usd_path = _live_articulation_model_usd_path(env, asset)
    runtime_tree_sha256 = _verify_loaded_runtime_usd_bundle(
        live_model_usd_path
    )
    cache_key = (
        float(near_x),
        float(surface_z),
        float(keepout_floor_z),
        float(margin),
        str(collision_proxy_artifact_path),
        str(collision_proxy_artifact_sha256),
        expected_names,
        source_paths,
        str(racket_body_name),
        tuple(float(value) for value in racket_blade_center_offset_wrist_m),
        tuple(float(value) for value in racket_blade_half_extents_m),
        live_model_usd_path,
        runtime_tree_sha256,
        str(body_pos_all.device),
        str(body_pos_all.dtype),
    )
    cached = getattr(asset, "_hope_table_geometric_guard_cache", None)
    if cached is not None:
        if cached[0] != cache_key:
            raise RuntimeError(
                "robot_hit_table pose guard contract changed during one run"
            )
        return cached[1]

    (
        component_owner_indices,
        component_center_values,
        component_half_axes_values,
    ) = _load_table_collision_proxy_artifact(
        collision_proxy_artifact_path,
        collision_proxy_artifact_sha256,
        expected_names,
    )
    blade_center_values = tuple(
        float(value) for value in racket_blade_center_offset_wrist_m
    )
    blade_half_values = tuple(
        float(value) for value in racket_blade_half_extents_m
    )
    if (
        len(blade_center_values) != 3
        or len(blade_half_values) != 3
        or not all(math.isfinite(value) for value in blade_center_values)
        or not all(
            math.isfinite(value) and value > 0.0
            for value in blade_half_values
        )
    ):
        raise RuntimeError(
            "robot_hit_table racket blade geometry must be finite with positive half extents"
        )
    boxes = tt_frame.table_assembly_aabbs_env(
        near_x,
        surface_z,
        keepout_floor_z=keepout_floor_z,
        margin=margin,
    )
    if (
        len(boxes) != 5
        or any(
            len(box) != 2
            or len(box[0]) != 3
            or len(box[1]) != 3
            or any(
                not math.isfinite(float(value))
                for value in (*box[0], *box[1])
            )
            or any(
                float(upper) < float(lower)
                for lower, upper in zip(box[0], box[1])
            )
            for box in boxes
        )
    ):
        raise RuntimeError(
            "robot_hit_table table-assembly AABBs must be five finite ordered boxes"
        )
    device = body_pos_all.device
    dtype = body_pos_all.dtype
    prepared = _PreparedRobotTablePoseGuard(
        asset=asset,
        asset_body_indices=torch.tensor(
            asset_ids, device=device, dtype=torch.long
        ),
        env_origins=env_origins,
        component_indices=torch.tensor(
            component_owner_indices, device=device, dtype=torch.long
        ),
        component_centers=torch.tensor(
            component_center_values, device=device, dtype=dtype
        ),
        component_half_axes=torch.tensor(
            component_half_axes_values, device=device, dtype=dtype
        ),
        aabb_lo=torch.tensor(
            [box[0] for box in boxes], device=device, dtype=dtype
        ),
        aabb_hi=torch.tensor(
            [box[1] for box in boxes], device=device, dtype=dtype
        ),
        racket_index=expected_names.index(racket_body_name),
        blade_center=torch.tensor(
            blade_center_values, device=device, dtype=dtype
        ),
        blade_local_half_axes=torch.diag(
            torch.tensor(blade_half_values, device=device, dtype=dtype)
        ),
        runtime_usd_receipt={
            "kind": "a3_pose_guard_live_runtime_usd_v1",
            "model_usd_path": live_model_usd_path,
            "bundle_tree_sha256": runtime_tree_sha256,
        },
    )
    setattr(
        asset,
        "_hope_a3_runtime_usd_receipt",
        dict(prepared.runtime_usd_receipt),
    )
    setattr(asset, "_hope_table_geometric_guard_cache", (cache_key, prepared))
    return prepared


def sample_robot_table_contact_current(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    filtered_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    near_x: float,
    surface_z: float,
    full_table_filtered_sensor_cfgs: tuple[SceneEntityCfg, ...]
    | list[SceneEntityCfg] = (),
    expected_full_table_source_prim_paths: tuple[str, ...]
    | list[str] = (),
    expected_full_robot_body_names: tuple[str, ...] | list[str] = (),
    force_threshold: float = 1.0,
    margin: float = 0.02,
    full_table_assembly: bool = False,
    keepout_floor_z: float = 0.0,
    collision_proxy_artifact_path: str = "",
    collision_proxy_artifact_sha256: str = "",
    racket_body_name: str = "right_wrist_yaw_Link",
    racket_blade_center_offset_wrist_m: tuple[float, float, float]
    | list[float] = (0.206194, 0.025474, 0.028020),
    racket_blade_half_extents_m: tuple[float, float, float]
    | list[float] = (0.082, 0.008, 0.082),
) -> torch.Tensor:
    """Sample current table contact once.

    Legacy top-only mode uses broad-origin attribution plus an exact wrist pair.  Full ActionBall
    mode uses live articulation pose and conservative table geometry without touching a
    ``ContactSensor``.
    """

    if full_table_assembly:
        prepared = prepare_robot_table_pose_guard(
            env,
            asset_cfg=asset_cfg,
            near_x=near_x,
            surface_z=surface_z,
            full_table_filtered_sensor_cfgs=(
                full_table_filtered_sensor_cfgs
            ),
            expected_full_table_source_prim_paths=(
                expected_full_table_source_prim_paths
            ),
            expected_full_robot_body_names=(
                expected_full_robot_body_names
            ),
            margin=margin,
            keepout_floor_z=keepout_floor_z,
            collision_proxy_artifact_path=collision_proxy_artifact_path,
            collision_proxy_artifact_sha256=(
                collision_proxy_artifact_sha256
            ),
            racket_body_name=racket_body_name,
            racket_blade_center_offset_wrist_m=(
                racket_blade_center_offset_wrist_m
            ),
            racket_blade_half_extents_m=racket_blade_half_extents_m,
        )
        return prepared()

    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame

    sensor = env.scene.sensors[sensor_cfg.name]
    forces = getattr(sensor.data, "net_forces_w", None)
    if forces is None or forces.ndim != 3:
        raise RuntimeError(
            "robot_hit_table requires sensor net_forces_w shaped [env, body, 3]; got "
            f"{None if forces is None else tuple(forces.shape)}"
        )
    asset: Articulation = env.scene[asset_cfg.name]
    sensor_ids, asset_ids = _aligned_body_ids(sensor, asset, sensor_cfg, asset_cfg)

    f = forces[:, sensor_ids, :]
    p = asset.data.body_pos_w[:, asset_ids, :]

    dev, dt = f.device, f.dtype
    aabb_key = (
        float(near_x),
        float(surface_z),
        float(force_threshold),
        float(margin),
        str(dev),
        str(dt),
    )
    cached_aabb = getattr(sensor, "_hope_table_hit_aabb_cache", None)
    if cached_aabb is None or cached_aabb[0] != aabb_key:
        lo, hi = tt_frame.table_top_aabb_env(
            near_x, surface_z, margin=margin
        )
        cached_aabb = (
            aabb_key,
            torch.tensor(lo, device=dev, dtype=dt),
            torch.tensor(hi, device=dev, dtype=dt),
        )
        setattr(sensor, "_hope_table_hit_aabb_cache", cached_aabb)
    lo_t, hi_t = cached_aabb[1], cached_aabb[2]
    broad_hit = table_hit_mask(p, f, env.scene.env_origins, lo_t, hi_t, force_threshold)

    try:
        filtered_sensor = env.scene.sensors[filtered_sensor_cfg.name]
    except KeyError as exc:
        raise RuntimeError(
            "robot_hit_table requires the filtered wrist-vs-table contact sensor "
            f"{filtered_sensor_cfg.name!r}"
        ) from exc
    force_matrix = getattr(filtered_sensor.data, "force_matrix_w", None)
    expected_filter_count = 1
    if (
        force_matrix is None
        or force_matrix.ndim != 4
        or force_matrix.shape[0] != broad_hit.shape[0]
        or force_matrix.shape[1] < 1
        or force_matrix.shape[2] != expected_filter_count
        or force_matrix.shape[3] != 3
    ):
        raise RuntimeError(
            "robot_hit_table requires filtered force_matrix_w shaped "
            f"[env, body, {expected_filter_count}, 3]; got "
            f"{None if force_matrix is None else tuple(force_matrix.shape)}"
        )
    filtered_hit = filtered_contact_hit_mask(force_matrix, force_threshold)
    return broad_hit | filtered_hit


def robot_hit_table(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    filtered_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    near_x: float,
    surface_z: float,
    full_table_filtered_sensor_cfgs: tuple[SceneEntityCfg, ...]
    | list[SceneEntityCfg] = (),
    expected_full_table_source_prim_paths: tuple[str, ...]
    | list[str] = (),
    expected_full_robot_body_names: tuple[str, ...] | list[str] = (),
    force_threshold: float = 1.0,
    margin: float = 0.02,
    full_table_assembly: bool = False,
    keepout_floor_z: float = 0.0,
    collision_proxy_artifact_path: str = "",
    collision_proxy_artifact_sha256: str = "",
    racket_body_name: str = "right_wrist_yaw_Link",
    racket_blade_center_offset_wrist_m: tuple[float, float, float]
    | list[float] = (0.206194, 0.025474, 0.028020),
    racket_blade_half_extents_m: tuple[float, float, float]
    | list[float] = (0.082, 0.008, 0.082),
    action_name: str = "joint_pos",
    require_substep_latch: bool = False,
) -> torch.Tensor:
    """The robot violated the table assembly guard.  Terminal, exactly like falling over.

    Legacy top-only mode keeps the broad non-foot/body-origin channel plus one exact wrist/racket
    pair channel.  ActionBall instead applies a pose-only keep-out with materialized
    collision-component OBBs plus a live racket-blade OBB and does not read a ``ContactSensor``.
    A full-assembly positive is therefore conservative keep-out evidence, not proof of resolved
    physical contact.
    ActionBall also requires the action term's
    policy-step latch: apply calls 2..4 sample physics substeps 1..3 and this DoneTerm finalizes
    substep 4, so a transient contact in any of the four substeps remains terminal.

    ``full_table_assembly`` includes a floor-to-slab-underside conservative robot keep-out, the
    real top slab, regulation net and two post proxies.  The keep-out is not a model of individual
    table legs and is prohibited in physical/shadow-ball scenes.
    """

    if require_substep_latch:
        action_manager = getattr(env, "action_manager", None)
        get_term = getattr(action_manager, "get_term", None)
        if not callable(get_term):
            raise RuntimeError(
                "robot_hit_table requires the action manager for substep latching"
            )
        action = get_term(action_name)
        finalize = getattr(
            action, "finalize_table_contact_substep_readback", None
        )
        if not callable(finalize):
            raise RuntimeError(
                "robot_hit_table requires an enabled table-contact substep action guard"
            )
        result = finalize()
        if (
            not torch.is_tensor(result)
            or result.dtype != torch.bool
            or tuple(result.shape) != (int(env.num_envs),)
        ):
            raise RuntimeError(
                "table-contact substep guard returned a malformed terminal mask"
            )
        return result

    return sample_robot_table_contact_current(
        env,
        sensor_cfg=sensor_cfg,
        filtered_sensor_cfg=filtered_sensor_cfg,
        full_table_filtered_sensor_cfgs=full_table_filtered_sensor_cfgs,
        expected_full_table_source_prim_paths=(
            expected_full_table_source_prim_paths
        ),
        expected_full_robot_body_names=expected_full_robot_body_names,
        asset_cfg=asset_cfg,
        near_x=near_x,
        surface_z=surface_z,
        force_threshold=force_threshold,
        margin=margin,
        full_table_assembly=full_table_assembly,
        keepout_floor_z=keepout_floor_z,
        collision_proxy_artifact_path=collision_proxy_artifact_path,
        collision_proxy_artifact_sha256=collision_proxy_artifact_sha256,
        racket_body_name=racket_body_name,
        racket_blade_center_offset_wrist_m=(
            racket_blade_center_offset_wrist_m
        ),
        racket_blade_half_extents_m=racket_blade_half_extents_m,
    )
