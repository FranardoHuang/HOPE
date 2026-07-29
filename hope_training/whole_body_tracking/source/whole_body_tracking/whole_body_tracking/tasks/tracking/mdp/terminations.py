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
    """Terminate when realized articulation q reaches the explicitly inset limit envelope."""

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
    terminal = current_violation | substep_actual_latch
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
        lower = limits[..., 0]
        upper = limits[..., 1]
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
            terminal=terminal,
            episode_age=env.episode_length_buf,
        )
    return terminal


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


def filtered_contact_hit_mask(
    force_matrix_w: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Reduce a filtered contact-force matrix to one table-hit bit per environment.

    ``ContactSensorData.force_matrix_w`` is shaped ``[env, sensor body, filter body, xyz]``.
    The dedicated table sensor has one body (the right wrist, which also carries the merged
    racket collision geometry) and either one legacy top filter or the exact five-part ActionBall
    assembly.  Reducing both dimensions yields one current-frame bit per environment.

    Non-finite force data fails safe: it becomes an infinite force and ends the affected episode
    instead of silently turning a broken contact stream into ``False``.
    """
    safe_force = torch.nan_to_num(
        force_matrix_w, nan=float("inf"), posinf=float("inf"), neginf=float("-inf")
    )
    pushing = torch.norm(safe_force, dim=-1) > float(force_threshold)
    return torch.any(pushing.flatten(start_dim=1), dim=1)


def all_body_filtered_contact_hit_mask(
    force_matrices_w: list[torch.Tensor] | tuple[torch.Tensor, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Reduce one exact table-pair force matrix per robot rigid body.

    Isaac Lab's filtered contact reporter is one-sensor-body-to-many-filter-bodies only.  Stacking
    several robot bodies into one filtered ``ContactSensor`` is therefore not a valid substitute
    for this list: every matrix here must come from its own one-body sensor.  The caller validates
    the runtime body identity and exact five-filter shape before this pure tensor reduction.
    """

    if not isinstance(force_matrices_w, (list, tuple)) or not force_matrices_w:
        raise RuntimeError(
            "robot_hit_table requires a non-empty exact per-body force-matrix list"
        )
    hits = [filtered_contact_hit_mask(matrix, force_threshold) for matrix in force_matrices_w]
    reference = hits[0]
    for index, hit in enumerate(hits[1:], start=1):
        if (
            hit.dtype != torch.bool
            or tuple(hit.shape) != tuple(reference.shape)
            or hit.device != reference.device
        ):
            raise RuntimeError(
                "robot_hit_table per-body force matrices do not share one env/device "
                f"contract at index {index}"
            )
    return torch.any(torch.stack(hits, dim=1), dim=1)


def _sample_all_body_filtered_table_contact(
    env: ManagerBasedRLEnv,
    *,
    filtered_sensor_cfg: SceneEntityCfg,
    all_body_filtered_sensor_cfgs: tuple[SceneEntityCfg, ...] | list[SceneEntityCfg],
    asset_cfg: SceneEntityCfg,
    force_threshold: float,
    expected_filter_prim_paths: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """Read exact robot-body/table-assembly pair forces without body-origin attribution."""

    if (
        not isinstance(expected_filter_prim_paths, (tuple, list))
        or not expected_filter_prim_paths
        or any(
            not isinstance(path, str) or not path
            for path in expected_filter_prim_paths
        )
        or len(set(expected_filter_prim_paths))
        != len(expected_filter_prim_paths)
    ):
        raise RuntimeError(
            "robot_hit_table full assembly requires non-empty unique expected "
            "filter prim paths"
        )
    expected_filter_prim_paths = tuple(expected_filter_prim_paths)
    expected_filter_count = len(expected_filter_prim_paths)
    scene_env_regex_ns = getattr(env.scene, "env_regex_ns", None)
    if scene_env_regex_ns is not None and (
        not isinstance(scene_env_regex_ns, str) or not scene_env_regex_ns
    ):
        raise RuntimeError(
            "robot_hit_table scene env_regex_ns must be a non-empty string"
        )
    runtime_expected_filter_prim_paths = tuple(
        path.format(ENV_REGEX_NS=scene_env_regex_ns)
        if scene_env_regex_ns is not None
        else path
        for path in expected_filter_prim_paths
    )
    if not isinstance(all_body_filtered_sensor_cfgs, (tuple, list)):
        raise RuntimeError(
            "robot_hit_table full assembly requires ordered per-body filtered sensor configs"
        )
    cfgs = tuple(all_body_filtered_sensor_cfgs)
    asset: Articulation = env.scene[asset_cfg.name]
    expected_body_names = tuple(str(name) for name in asset.body_names)
    if not expected_body_names or len(set(expected_body_names)) != len(
        expected_body_names
    ):
        raise RuntimeError(
            "robot_hit_table requires a non-empty unique articulation body-name table"
        )
    if len(cfgs) != len(expected_body_names):
        raise RuntimeError(
            "robot_hit_table requires exactly one filtered sensor per articulation "
            f"rigid body; got {len(cfgs)} sensors for {len(expected_body_names)} bodies"
        )

    cfg_names = tuple(getattr(cfg, "name", None) for cfg in cfgs)
    if (
        any(not isinstance(name, str) or not name for name in cfg_names)
        or len(set(cfg_names)) != len(cfg_names)
    ):
        raise RuntimeError(
            "robot_hit_table per-body filtered sensor names must be non-empty and unique"
        )
    if getattr(filtered_sensor_cfg, "name", None) not in cfg_names:
        raise RuntimeError(
            "robot_hit_table wrist/racket filtered sensor is missing from the "
            "all-body exact sensor set"
        )
    wrist_sensor_index = cfg_names.index(filtered_sensor_cfg.name)
    if expected_body_names[wrist_sensor_index] != "right_wrist_yaw_Link":
        raise RuntimeError(
            "robot_hit_table filtered_sensor_cfg does not identify the exact "
            "right_wrist_yaw_Link/racket sensor"
        )

    force_matrices: list[torch.Tensor] = []
    observed_body_names: list[str] = []
    expected_env_count: int | None = None
    expected_device = None
    for index, cfg in enumerate(cfgs):
        try:
            sensor = env.scene.sensors[cfg.name]
        except KeyError as exc:
            raise RuntimeError(
                "robot_hit_table is missing exact per-body filtered sensor "
                f"{cfg.name!r}"
            ) from exc
        body_names = tuple(str(name) for name in getattr(sensor, "body_names", ()))
        expected_body_name = expected_body_names[index]
        if body_names != (expected_body_name,):
            raise RuntimeError(
                "robot_hit_table exact filtered sensors must each resolve to one "
                f"expected rigid body; {cfg.name!r} expected "
                f"{expected_body_name!r}, resolved {body_names!r}"
            )
        observed_body_names.append(body_names[0])

        runtime_sensor_cfg = getattr(sensor, "cfg", None)
        expected_sensor_prim_template = (
            f"{{ENV_REGEX_NS}}/Robot/{expected_body_name}"
        )
        expected_sensor_prim = (
            expected_sensor_prim_template.format(
                ENV_REGEX_NS=scene_env_regex_ns
            )
            if scene_env_regex_ns is not None
            else expected_sensor_prim_template
        )
        runtime_filter_paths = tuple(
            getattr(runtime_sensor_cfg, "filter_prim_paths_expr", ()) or ()
        )
        if (
            getattr(runtime_sensor_cfg, "prim_path", None)
            != expected_sensor_prim
            or runtime_filter_paths != runtime_expected_filter_prim_paths
        ):
            raise RuntimeError(
                "robot_hit_table exact per-body sensor source/filter binding drift: "
                f"{cfg.name!r}"
            )

        matrix = getattr(sensor.data, "force_matrix_w", None)
        if (
            matrix is None
            or matrix.ndim != 4
            or matrix.shape[1] != 1
            or matrix.shape[2] != expected_filter_count
            or matrix.shape[3] != 3
        ):
            raise RuntimeError(
                "robot_hit_table requires every exact per-body force_matrix_w shaped "
                f"[env, 1, {expected_filter_count}, 3]; sensor {cfg.name!r} got "
                f"{None if matrix is None else tuple(matrix.shape)}"
            )
        if expected_env_count is None:
            expected_env_count = int(matrix.shape[0])
            expected_device = matrix.device
        elif (
            int(matrix.shape[0]) != expected_env_count
            or matrix.device != expected_device
        ):
            raise RuntimeError(
                "robot_hit_table exact per-body force matrices disagree on env/device "
                f"at sensor index {index}"
            )
        force_matrices.append(matrix)

    if tuple(observed_body_names) != expected_body_names:
        raise RuntimeError(
            "robot_hit_table exact per-body sensor coverage/order does not equal the "
            "runtime articulation body table"
        )
    return all_body_filtered_contact_hit_mask(force_matrices, force_threshold)


def sample_robot_table_contact_current(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    filtered_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    near_x: float,
    surface_z: float,
    all_body_filtered_sensor_cfgs: tuple[SceneEntityCfg, ...]
    | list[SceneEntityCfg] = (),
    expected_full_table_filter_prim_paths: tuple[str, ...]
    | list[str] = (),
    force_threshold: float = 1.0,
    margin: float = 0.02,
    full_table_assembly: bool = False,
    keepout_floor_z: float = 0.0,
) -> torch.Tensor:
    """Sample current table contact once.

    Legacy top-only mode uses broad-origin attribution plus an exact wrist pair.  Full ActionBall
    mode uses only exact one-body pair-filter sensors over the five-part assembly.
    """

    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame

    if full_table_assembly:
        # Exact pair identity replaces the legacy body-origin attribution.  It catches a contact
        # point on an elbow/forearm/racket mesh even when that rigid body's origin is outside every
        # obstacle AABB, and it includes feet because a foot/table contact is not sanctioned.
        return _sample_all_body_filtered_table_contact(
            env,
            filtered_sensor_cfg=filtered_sensor_cfg,
            all_body_filtered_sensor_cfgs=all_body_filtered_sensor_cfgs,
            asset_cfg=asset_cfg,
            force_threshold=force_threshold,
            expected_filter_prim_paths=expected_full_table_filter_prim_paths,
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
    all_body_filtered_sensor_cfgs: tuple[SceneEntityCfg, ...]
    | list[SceneEntityCfg] = (),
    expected_full_table_filter_prim_paths: tuple[str, ...]
    | list[str] = (),
    force_threshold: float = 1.0,
    margin: float = 0.02,
    full_table_assembly: bool = False,
    keepout_floor_z: float = 0.0,
    action_name: str = "joint_pos",
    require_substep_latch: bool = False,
) -> torch.Tensor:
    """The robot struck the table assembly.  Terminal, exactly like falling over.

    Legacy top-only mode keeps the broad non-foot/body-origin channel plus one exact wrist/racket
    pair channel.  ActionBall instead reads one one-body ``force_matrix_w`` sensor for every
    articulation rigid body, each filtered against the installed table, keep-out, net and posts;
    body origins never determine pair identity.  ActionBall also requires the action term's
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
        all_body_filtered_sensor_cfgs=all_body_filtered_sensor_cfgs,
        expected_full_table_filter_prim_paths=(
            expected_full_table_filter_prim_paths
        ),
        asset_cfg=asset_cfg,
        near_x=near_x,
        surface_z=surface_z,
        force_threshold=force_threshold,
        margin=margin,
        full_table_assembly=full_table_assembly,
        keepout_floor_z=keepout_floor_z,
    )
