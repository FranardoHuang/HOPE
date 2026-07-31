from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


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
    articulation.  The table guard's radii and racket index are defined in the
    reviewed A3 order, so merely proving the two runtime views agree with each
    other is insufficient: both streams must be reordered to that explicit
    contract before the tensor kernel consumes them.
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


def geometric_table_contact_hit_mask(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    contact_force_w: torch.Tensor,
    env_origins: torch.Tensor,
    body_proxy_radius_m: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
    *,
    racket_body_index: int,
    racket_blade_center_offset_wrist_m: torch.Tensor,
    racket_blade_half_extents_m: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Attribute one unfiltered whole-body force stream to conservative table geometry.

    Ordinary A3 links use one conservative sphere around the live rigid-body origin.  The merged
    wrist/racket rigid body additionally uses the live wrist quaternion and the shipped blade
    collision OBB.  OBB-vs-AABB is reduced through the OBB's conservative world AABB: this may
    reject a near-corner brush but cannot miss a blade/table overlap.  Contact force is still
    required, so legal free-space poses are not terminated merely for entering an inflated broad
    phase.  Non-finite runtime pose/force data fails safe per environment.
    """

    if (
        body_pos_w.ndim != 3
        or body_pos_w.shape[-1] != 3
        or body_quat_w.shape != (*body_pos_w.shape[:-1], 4)
        or contact_force_w.shape != body_pos_w.shape
        or env_origins.shape != (body_pos_w.shape[0], 3)
        or body_proxy_radius_m.shape != (body_pos_w.shape[1],)
        or aabb_lo.ndim != 2
        or aabb_lo.shape[-1] != 3
        or aabb_hi.shape != aabb_lo.shape
        or racket_blade_center_offset_wrist_m.shape != (3,)
        or racket_blade_half_extents_m.shape != (3,)
    ):
        raise RuntimeError(
            "geometric table contact requires body pose/force [E,B,*], origins [E,3], "
            "radii [B], assembly AABBs [O,3], and racket vectors [3]"
        )
    tensors = (
        body_quat_w,
        contact_force_w,
        env_origins,
        body_proxy_radius_m,
        aabb_lo,
        aabb_hi,
        racket_blade_center_offset_wrist_m,
        racket_blade_half_extents_m,
    )
    if (
        not torch.is_floating_point(body_pos_w)
        or any(
            not torch.is_floating_point(value)
            or value.device != body_pos_w.device
            or value.dtype != body_pos_w.dtype
            for value in tensors
        )
    ):
        raise RuntimeError(
            "geometric table contact tensors must share one floating dtype/device"
        )
    if (
        isinstance(racket_body_index, bool)
        or not isinstance(racket_body_index, Integral)
        or not 0 <= int(racket_body_index) < body_pos_w.shape[1]
    ):
        raise RuntimeError("racket_body_index is outside the selected A3 body order")
    if isinstance(force_threshold, bool) or not isinstance(
        force_threshold, Real
    ):
        raise RuntimeError("table contact force threshold must be numeric")

    racket_body_index = int(racket_body_index)
    p_local = body_pos_w - env_origins[:, None, :]
    # Squared distance from each sphere centre to each table-assembly AABB.
    below = aabb_lo[None, None, :, :] - p_local[:, :, None, :]
    above = p_local[:, :, None, :] - aabb_hi[None, None, :, :]
    outside = torch.maximum(
        torch.maximum(below, above), torch.zeros_like(below)
    )
    sphere_overlap = torch.any(
        torch.sum(outside * outside, dim=-1)
        <= body_proxy_radius_m[None, :, None]
        * body_proxy_radius_m[None, :, None],
        dim=2,
    )

    safe_force = torch.nan_to_num(
        contact_force_w,
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("-inf"),
    )
    pushing = torch.linalg.vector_norm(safe_force, dim=-1) > float(
        force_threshold
    )
    body_hit = torch.any(sphere_overlap & pushing, dim=1)

    wrist_quat = body_quat_w[:, racket_body_index, :]
    quat_norm_sq = torch.sum(wrist_quat * wrist_quat, dim=-1, keepdim=True)
    safe_quat = wrist_quat / torch.sqrt(
        torch.clamp(quat_norm_sq, min=torch.finfo(body_pos_w.dtype).tiny)
    )
    blade_offset_w = _quat_rotate_wxyz(
        safe_quat, racket_blade_center_offset_wrist_m.expand_as(p_local[:, 0])
    )
    blade_center_local = (
        p_local[:, racket_body_index, :] + blade_offset_w
    )
    local_half_axes = torch.diag(
        racket_blade_half_extents_m
    ).unsqueeze(0).expand(body_pos_w.shape[0], -1, -1)
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
    racket_hit = blade_overlap & pushing[:, racket_body_index]

    invalid_runtime = (
        ~torch.isfinite(body_pos_w).all(dim=(1, 2))
        | ~torch.isfinite(body_quat_w).all(dim=(1, 2))
        | ~torch.isfinite(contact_force_w).all(dim=(1, 2))
        | ~torch.isfinite(env_origins).all(dim=1)
        | ~(quat_norm_sq[:, 0] > 0.0)
    )
    invalid_static = (
        ~torch.isfinite(body_proxy_radius_m).all()
        | ~torch.isfinite(aabb_lo).all()
        | ~torch.isfinite(aabb_hi).all()
        | ~torch.isfinite(racket_blade_center_offset_wrist_m).all()
        | ~torch.isfinite(racket_blade_half_extents_m).all()
        | torch.any(body_proxy_radius_m < 0.0)
        | torch.any(aabb_hi < aabb_lo)
        | torch.any(racket_blade_half_extents_m <= 0.0)
    )
    return body_hit | racket_hit | invalid_runtime | invalid_static


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
    body_proxy_radius_m: float = 0.18,
    foot_proxy_radius_m: float = 0.10,
    wrist_proxy_radius_m: float = 0.08,
    foot_body_names: tuple[str, ...] | list[str] = (),
    racket_body_name: str = "right_wrist_yaw_Link",
    racket_blade_center_offset_wrist_m: tuple[float, float, float]
    | list[float] = (0.206194, 0.025474, 0.028020),
    racket_blade_half_extents_m: tuple[float, float, float]
    | list[float] = (0.082, 0.008, 0.082),
) -> torch.Tensor:
    """Sample current table contact once.

    Legacy top-only mode uses broad-origin attribution plus an exact wrist pair.  Full ActionBall
    mode uses the existing whole-body unfiltered force stream and conservative table geometry.
    """

    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame

    if full_table_assembly:
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = getattr(sensor.data, "net_forces_w", None)
        if (
            forces is None
            or forces.ndim != 3
            or forces.shape[-1] != 3
            or not torch.is_floating_point(forces)
        ):
            raise RuntimeError(
                "robot_hit_table full assembly requires whole-body net_forces_w "
                "shaped [env, body, 3]"
            )
        asset: Articulation = env.scene[asset_cfg.name]
        expected_names = tuple(expected_full_robot_body_names)
        if len(expected_names) != 32 or len(set(expected_names)) != 32:
            raise RuntimeError(
                "robot_hit_table full assembly requires one unique 32-body "
                "A3 contract"
            )
        sensor_ids, asset_ids = _aligned_body_ids_in_expected_order(
            sensor,
            asset,
            sensor_cfg,
            asset_cfg,
            expected_names,
        )
        selected_sensor_names = tuple(
            str(sensor.body_names[index]) for index in sensor_ids
        )
        selected_asset_names = tuple(
            str(asset.body_names[index]) for index in asset_ids
        )
        if (
            selected_sensor_names != expected_names
            or selected_asset_names != expected_names
        ):
            raise RuntimeError(
                "robot_hit_table full assembly requires the exact ordered "
                "32-body A3 unfiltered-force contract"
            )
        source_paths = tuple(expected_full_table_source_prim_paths)
        if (
            len(source_paths) != 5
            or len(set(source_paths)) != 5
            or any(not isinstance(path, str) or not path for path in source_paths)
        ):
            raise RuntimeError(
                "robot_hit_table full assembly requires five unique table "
                "geometry source paths"
            )
        body_pos = asset.data.body_pos_w[:, asset_ids, :]
        body_quat = getattr(asset.data, "body_quat_w", None)
        if (
            body_quat is None
            or body_quat.ndim != 3
            or body_quat.shape[-1] != 4
        ):
            raise RuntimeError(
                "robot_hit_table full assembly requires body_quat_w "
                "shaped [env, body, 4]"
            )
        body_quat = body_quat[:, asset_ids, :]
        selected_forces = forces[:, sensor_ids, :]
        if (
            body_pos.shape[:2] != selected_forces.shape[:2]
            or body_quat.shape[:2] != selected_forces.shape[:2]
            or body_pos.device != selected_forces.device
            or body_quat.device != selected_forces.device
            or body_pos.dtype != selected_forces.dtype
            or body_quat.dtype != selected_forces.dtype
            or body_pos.shape[0] != int(env.num_envs)
        ):
            raise RuntimeError(
                "robot_hit_table body pose and whole-body force streams disagree"
            )
        if (
            not isinstance(racket_body_name, str)
            or racket_body_name not in expected_names
        ):
            raise RuntimeError(
                "robot_hit_table racket body is absent from the exact A3 body order"
            )
        foot_names = tuple(foot_body_names)
        if (
            len(foot_names) != 2
            or len(set(foot_names)) != 2
            or any(name not in expected_names for name in foot_names)
        ):
            raise RuntimeError(
                "robot_hit_table requires the exact two A3 support-foot body names"
            )

        cache_key = (
            float(near_x),
            float(surface_z),
            float(keepout_floor_z),
            float(margin),
            float(body_proxy_radius_m),
            float(foot_proxy_radius_m),
            float(wrist_proxy_radius_m),
            foot_names,
            str(racket_body_name),
            tuple(float(value) for value in racket_blade_center_offset_wrist_m),
            tuple(float(value) for value in racket_blade_half_extents_m),
            str(selected_forces.device),
            str(selected_forces.dtype),
        )
        cached = getattr(sensor, "_hope_table_geometric_guard_cache", None)
        if cached is None or cached[0] != cache_key:
            if not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    float(body_proxy_radius_m),
                    float(foot_proxy_radius_m),
                    float(wrist_proxy_radius_m),
                )
            ):
                raise RuntimeError(
                    "robot_hit_table body proxy radii must be finite and non-negative"
                )
            boxes = tt_frame.table_assembly_aabbs_env(
                near_x,
                surface_z,
                keepout_floor_z=keepout_floor_z,
                margin=margin,
            )
            lo_t = torch.tensor(
                [box[0] for box in boxes],
                device=selected_forces.device,
                dtype=selected_forces.dtype,
            )
            hi_t = torch.tensor(
                [box[1] for box in boxes],
                device=selected_forces.device,
                dtype=selected_forces.dtype,
            )
            radii = torch.full(
                (len(expected_names),),
                float(body_proxy_radius_m),
                device=selected_forces.device,
                dtype=selected_forces.dtype,
            )
            for foot_name in foot_names:
                radii[expected_names.index(foot_name)] = float(
                    foot_proxy_radius_m
                )
            racket_index = expected_names.index(racket_body_name)
            radii[racket_index] = float(wrist_proxy_radius_m)
            blade_center = torch.tensor(
                tuple(float(v) for v in racket_blade_center_offset_wrist_m),
                device=selected_forces.device,
                dtype=selected_forces.dtype,
            )
            blade_half = torch.tensor(
                tuple(float(v) for v in racket_blade_half_extents_m),
                device=selected_forces.device,
                dtype=selected_forces.dtype,
            )
            cached = (
                cache_key,
                radii,
                lo_t,
                hi_t,
                racket_index,
                blade_center,
                blade_half,
            )
            setattr(sensor, "_hope_table_geometric_guard_cache", cached)
        (
            _,
            radii,
            lo_t,
            hi_t,
            racket_index,
            blade_center,
            blade_half,
        ) = cached
        return geometric_table_contact_hit_mask(
            body_pos,
            body_quat,
            selected_forces,
            env.scene.env_origins,
            radii,
            lo_t,
            hi_t,
            racket_body_index=racket_index,
            racket_blade_center_offset_wrist_m=blade_center,
            racket_blade_half_extents_m=blade_half,
            force_threshold=force_threshold,
        )

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
    body_proxy_radius_m: float = 0.18,
    foot_proxy_radius_m: float = 0.10,
    wrist_proxy_radius_m: float = 0.08,
    foot_body_names: tuple[str, ...] | list[str] = (),
    racket_body_name: str = "right_wrist_yaw_Link",
    racket_blade_center_offset_wrist_m: tuple[float, float, float]
    | list[float] = (0.206194, 0.025474, 0.028020),
    racket_blade_half_extents_m: tuple[float, float, float]
    | list[float] = (0.082, 0.008, 0.082),
    action_name: str = "joint_pos",
    require_substep_latch: bool = False,
) -> torch.Tensor:
    """The robot struck the table assembly.  Terminal, exactly like falling over.

    Legacy top-only mode keeps the broad non-foot/body-origin channel plus one exact wrist/racket
    pair channel.  ActionBall instead reads the existing whole-body unfiltered force stream and
    attributes it with conservative link spheres plus a live racket-blade OBB.  ActionBall also
    requires the action term's
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
        body_proxy_radius_m=body_proxy_radius_m,
        foot_proxy_radius_m=foot_proxy_radius_m,
        wrist_proxy_radius_m=wrist_proxy_radius_m,
        foot_body_names=foot_body_names,
        racket_body_name=racket_body_name,
        racket_blade_center_offset_wrist_m=(
            racket_blade_center_offset_wrist_m
        ),
        racket_blade_half_extents_m=racket_blade_half_extents_m,
    )
