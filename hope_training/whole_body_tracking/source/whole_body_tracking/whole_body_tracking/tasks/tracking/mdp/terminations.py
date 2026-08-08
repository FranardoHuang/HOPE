from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


# 2026-08-08: re-cut onto the 0807 A3P-P1 plant, and upgraded from byte
# integrity to an identity proof.
#
# Until today these were the retired 0409 robot's digests -- the third copy of
# them in the repository -- and they were the reason ``oracle32`` refused the
# moment the launcher started spawning the 0807 plant.  That refusal was
# correct.  But re-cutting six digests only ever restores the old, weak claim:
# "this USD cache was not edited".  It cannot say "this cache is a cache of the
# robot whose collision volumes this guard is testing", and any robot's USD
# satisfies it equally well.
#
# ``_rederive_isaaclab_asset_hash`` below closes that gap the same way the
# launcher does: IsaacLab's own ``.asset_hash`` is recomputed from the live
# bundle's converter configuration plus the bytes of the tracked plant URDF the
# collision proxy was materialized from, and must equal the digest the bundle
# stores.  Names can be doctored; a cache converted from a different robot
# cannot survive that.  Do not re-cut one half without the other.
_A3_COLLISION_PROXY_SOURCE_URDF_RELATIVE = (
    "agi/URDF/A3P-P1-32dof-0807-OP3-pingpang/urdf/model.urdf"
)
_A3_COLLISION_PROXY_SOURCE_URDF_SHA256 = (
    "15c83f5f3beea71350583143aef4d622d5219df65a0bed9a660a0edb7d388d09"
)
_A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH = "676efde5febed3c0fde0f2ad59650cdf"
# isaaclab/sim/converters/asset_converter_base.py::_config_to_hash drops these
# three path keys before hashing the converter configuration.
_A3_COLLISION_PROXY_ASSET_HASH_EXCLUDED_CONFIG_KEYS = (
    "asset_path",
    "usd_dir",
    "usd_file_name",
)
_A3_COLLISION_PROXY_PLANT_IDENTITY_KIND = "a3_collision_proxy_plant_identity_v1"
_A3_COLLISION_PROXY_PLANT_ASSET_ROOT_NAME = "agibot_a3p_p1_0807_v1"
_A3_COLLISION_PROXY_COMPONENT_COUNT = 62
# The 20 OmniPicker3 left-gripper collision links.  The 0409 plant carried one
# coarse ``left_hand_link`` placeholder box here; the 0807 plant carries the
# real gripper, and its volume enters this guard for the first time.  Naming
# the links makes a later "cleanup" that drops them a refusal rather than a
# quietly smaller component count.
_A3_COLLISION_PROXY_LEFT_GRIPPER_SOURCE_LINKS = (
    "left_base_link",
    "left_link1",
    "left_link10",
    "left_link11",
    "left_link11-1",
    "left_link13",
    "left_link14",
    "left_link14-1",
    "left_link15",
    "left_link17",
    "left_link18",
    "left_link2",
    "left_link3",
    "left_link4",
    "left_link4-1",
    "left_link6",
    "left_link7",
    "left_link7-1",
    "left_link8",
    "left_link9",
)
_A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256 = (
    "365ba37edd5e5e1d4fac22f2cbb3ec871ead7bb49aeadb50161ef523a9ae6747"
)
_A3_COLLISION_PROXY_RUNTIME_USD_TOTAL_FILE_BYTES = 60519988
_A3_COLLISION_PROXY_RUNTIME_USD_FILES = (
    (
        ".asset_hash",
        "a78a2f8fb207cbf479cc1b308cf9d3c58e1a55eb7da9dbc2caf34be697e9c993",
        32,
    ),
    (
        "config.yaml",
        "f349c3f4d80a915f5ca3ce53d49785dfd7e6eeca2645dcd7b402d4d8a2288eb9",
        1685,
    ),
    (
        "configuration/model_base.usd",
        "108a4b45b96a8db8396d3a8feb995481c5db87efcde80066e6347ed494e658fc",
        60504873,
    ),
    (
        "configuration/model_physics.usd",
        "390cf66cc052ea697e88e9ef0131bf7e2eee96e70c35c0861e1ce33d363747f5",
        11078,
    ),
    (
        "configuration/model_sensor.usd",
        "4e16201f146db3240b8a0082ae14e3aca41255a75812c5331bf8f4e39701355c",
        687,
    ),
    (
        "model.usd",
        "13e5ecfe02238fbf1d20c13ed7177e18ed93d84bca8e0a592b6605f7fb85f351",
        1633,
    ),
)
_TABLE_GUARD_OBSTACLE_ROLES = (
    "top",
    "keepout",
    "net",
    "post_left",
    "post_right",
)


def action_ball_diagnostic_single_stroke_complete(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """End the scoped measured-N1 episode after one complete non-looping stroke.

    The term is installed only by the diagnostic split-ready task preflight and
    is marked as a timeout, so completion triggers a true reset without being
    counted as a failure/death.  Formal canonical clips retain their existing
    within-episode wrap semantics and never install this term.
    """

    command = env.command_manager.get_term(command_name)
    enabled = getattr(
        command, "action_ball_diagnostic_split_ready_teacher", False
    )
    if enabled is not True:
        raise RuntimeError(
            "single-stroke completion termination requires the scoped "
            "split-ready measured diagnostic"
        )
    complete = command.action_ball_single_stroke_complete
    if (
        not torch.is_tensor(complete)
        or complete.dtype != torch.bool
        or complete.shape != (env.num_envs,)
    ):
        raise RuntimeError(
            "single-stroke completion latch must be bool[num_envs]"
        )
    return complete


def _resolve_repo_relative_file(relative: str, *, name: str) -> Path:
    """Resolve one tracked repo-relative file without picking a shadow copy."""

    configured = Path(relative)
    candidates = [
        Path.cwd() / configured,
        *(parent / configured for parent in Path(__file__).resolve().parents),
    ]
    matches: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if len(matches) != 1:
        raise RuntimeError(
            f"robot_hit_table {name} must resolve to exactly one tracked file: "
            f"{relative!r} -> {[str(path) for path in matches]}"
        )
    return matches[0]


def _rederive_isaaclab_asset_hash(config: dict, urdf_path: Path) -> str:
    """Redo IsaacLab's ``.asset_hash`` offline, without importing Kit.

    Byte-compatible on purpose with
    ``isaaclab/sim/converters/asset_converter_base.py::_config_to_hash``: MD5
    over ``json.dumps`` of the converter configuration with the three path keys
    removed, then over the source asset file in 64 KiB chunks.
    """

    payload = dict(config)
    for key in _A3_COLLISION_PROXY_ASSET_HASH_EXCLUDED_CONFIG_KEYS:
        payload.pop(key, None)
    digest = hashlib.md5()
    digest.update(json.dumps(payload).encode())
    with open(urdf_path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_live_bundle_is_a_cache_of_this_plant(bundle_root: Path) -> str:
    """Prove the live USD bundle was converted from the proxied plant URDF.

    The six-file digest pin above answers "are these the reviewed bytes".  This
    answers the question that actually matters to a collision proxy: "are these
    bytes a cache of the robot whose collision volumes I am about to test?".
    The answer is a derivation, not a name match -- the converter configuration
    the bundle stores, plus the tracked URDF the proxy was materialized from,
    have to reproduce the digest the converter itself wrote.
    """

    import yaml

    urdf_path = _resolve_repo_relative_file(
        _A3_COLLISION_PROXY_SOURCE_URDF_RELATIVE, name="plant URDF"
    )
    observed_urdf_sha256 = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if observed_urdf_sha256 != _A3_COLLISION_PROXY_SOURCE_URDF_SHA256:
        raise RuntimeError(
            "robot_hit_table plant URDF differs from the collision proxy pin: "
            f"{observed_urdf_sha256} != {_A3_COLLISION_PROXY_SOURCE_URDF_SHA256}"
        )
    try:
        config = yaml.safe_load(
            (bundle_root / "config.yaml").read_text(encoding="ascii")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(
            "robot_hit_table live USD bundle config.yaml cannot be read"
        ) from exc
    if not isinstance(config, dict):
        raise RuntimeError(
            "robot_hit_table live USD bundle config.yaml is not a mapping"
        )
    recorded_asset_path = config.get("asset_path")
    if (
        not isinstance(recorded_asset_path, str)
        or f"/{_A3_COLLISION_PROXY_PLANT_ASSET_ROOT_NAME}/"
        not in recorded_asset_path
    ):
        raise RuntimeError(
            "robot_hit_table live USD bundle was converted from a different "
            f"asset package: config.yaml asset_path={recorded_asset_path!r}"
        )
    stored = (bundle_root / ".asset_hash").read_text(encoding="ascii").strip()
    rederived = _rederive_isaaclab_asset_hash(config, urdf_path)
    if rederived != stored:
        raise RuntimeError(
            "robot_hit_table live USD bundle is not a cache of the proxied "
            f"plant: IsaacLab asset hash over "
            f"{_A3_COLLISION_PROXY_SOURCE_URDF_RELATIVE} recomputes to "
            f"{rederived} but the bundle stores {stored}"
        )
    if stored != _A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH:
        raise RuntimeError(
            "robot_hit_table live USD bundle .asset_hash differs from the "
            f"reviewed pin: {stored} != "
            f"{_A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH}"
        )
    return stored


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

    Two claims are made, and they are not the same claim.  Every byte of the
    six-file tree matches its pin -- integrity.  And the tree re-derives to
    IsaacLab's stored ``.asset_hash`` from the tracked plant URDF this proxy
    measures -- identity.  Integrity alone was what let a retired robot's cache
    pass for months.
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
    _verify_live_bundle_is_a_cache_of_this_plant(bundle_root)
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
    terminate: bool = True,
) -> torch.Tensor:
    """Terminate only at the physical hard edge; retain the inset band as diagnostic evidence.

    The continuous actual-q barrier owns recoverable proximity to a limit.  Turning the
    two-percent inner band into an immediate reset starves PPO of the transition that can teach
    recovery, and conflates a soft constraint with a mechanical hard-edge violation.  This term
    still requires the inset margins so diagnostic runs can report the exact joint/side occupancy,
    but the Done bit is reserved for non-finite/invalid state or a current/substep raw hard edge.

    ``terminate=False`` keeps every measurement and latch above but stops the hard edge from
    shortening the episode, matching the vendor-aligned ``build_1`` structure where the same
    quantity is a ``DoneTerm`` that always returns False and feeds checkpoint NO-GO evidence
    instead of PPO resets.  A binary reset on every violation is exactly the CaT
    (arXiv:2403.18765) ablation whose return is identically zero; on this branch it was measured
    as 7/7 episodes killed at ticks 69--88, all before the nominal strike tick, with a teacher
    whose own minimum limit margin is 0.116 rad (16.6% of travel).  Evidence is therefore
    mandatory in this mode: the diagnostic recorder must be enabled so a non-zero hard-edge count
    still blocks promotion and deployment.
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
    if type(terminate) is not bool:
        raise RuntimeError(f"{context} requires an exact boolean terminate flag")
    if not terminate and not diagnostic_enabled:
        raise RuntimeError(
            f"{context} telemetry mode (terminate=False) requires the attribution "
            "recorder: dropping the reset without latching the hard-edge evidence "
            "would train a policy that cannot be promoted or deployed"
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
    if not terminate:
        # The recorder above already latched the true hard-edge event, so the evidence and the
        # promotion NO-GO survive; only the Done bit is withheld.
        return torch.zeros_like(hard_terminal)
    return hard_terminal


# [已删除 2026-08-06 过期结构清理] bad_anchor_pos(全 3D 版):零调用点,而且**不可达**——
# 终止项是按字符串解析的,scripts/termination_contract.py 的 FUNCTION_IDENTITIES 是穷举
# allow-list("an unlisted active term is a hard reject by design"),里面只有下面这个
# _z_only 版。两个只差 norm(3D) / abs(z) 的同名族函数并排放着,谁去调阈值都可能调错那份。
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


# [已删除 2026-08-06 过期结构清理] bad_motion_body_pos(全 3D 版):同上,零调用点且不在
# termination_contract.FUNCTION_IDENTITIES 的穷举 allow-list 里,任何 frozen 合同都到不了它。
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


# [已删除 2026-08-06 过期结构清理] _aligned_body_ids_in_expected_order(43 行):零调用点。
# 它是上面 _aligned_body_ids(现役,唯一调用点在 robot_hit_table 的 :2280 附近)的
# "显式顺序"孪生兄弟——同一个 sensor 上的 memo、同一句 "expected body order changed
# during one run"、只差调 align_body_ids_in_expected_order 而不是 align_body_ids。
# robot_hit_table 正是眼下 32/32 终止的那条路径,并排放两份几乎一样的 body-id 对齐 memo
# 是明确的踩坑点:改一份、另一份留在原地,而且没有任何东西会报警。
# 注意:纯函数 align_body_ids_in_expected_order(:719)保留,但它现在**只剩测试调用点**
# (test_table_obstacle_termination.py),生产侧无消费方;将来真要做显式顺序守卫时从那里接。


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
        or source_urdf.get("path")
        != _A3_COLLISION_PROXY_SOURCE_URDF_RELATIVE
        or source_urdf.get("sha256")
        != _A3_COLLISION_PROXY_SOURCE_URDF_SHA256
        or not isinstance(runtime_usd, dict)
        or runtime_usd.get("bundle_tree_sha256")
        != _A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256
        or runtime_usd.get("file_count") != 6
        or runtime_usd.get("total_file_bytes")
        != _A3_COLLISION_PROXY_RUNTIME_USD_TOTAL_FILE_BYTES
        or runtime_usd.get("symlinks_forbidden") is not True
        or runtime_usd.get("files") != expected_runtime_files
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy does not bind the reviewed vendor "
            "URDF and exact six-file Pod runtime USD bundle"
        )
    # The artifact must carry its own derivation proof, and that proof must name
    # the same plant the live bundle re-derives to.  Without this the artifact
    # would only be pinned to a USD tree by digest, which is exactly the claim
    # that failed to notice a retired robot.
    plant_identity = document.get("plant_identity")
    if (
        not isinstance(plant_identity, dict)
        or plant_identity.get("kind")
        != _A3_COLLISION_PROXY_PLANT_IDENTITY_KIND
        or plant_identity.get("plant_asset_root_name")
        != _A3_COLLISION_PROXY_PLANT_ASSET_ROOT_NAME
        or plant_identity.get("isaaclab_asset_hash")
        != _A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH
        or plant_identity.get("isaaclab_asset_hash_excluded_config_keys")
        != list(_A3_COLLISION_PROXY_ASSET_HASH_EXCLUDED_CONFIG_KEYS)
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy carries no derivation proof "
            "tying it to the reviewed A3 plant"
        )
    converter_config = plant_identity.get("converter_config_yaml")
    if (
        not isinstance(converter_config, str)
        or hashlib.sha256(converter_config.encode("ascii")).hexdigest()
        != plant_identity.get("converter_config_sha256")
        or plant_identity.get("converter_config_sha256")
        != dict(
            (path, sha256)
            for path, sha256, _size in _A3_COLLISION_PROXY_RUNTIME_USD_FILES
        )["config.yaml"]
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy converter configuration is not "
            "the pinned bundle's config.yaml"
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
        or len(components) != _A3_COLLISION_PROXY_COMPONENT_COUNT
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
    source_links = set()
    for component in components:
        if not isinstance(component, dict):
            raise RuntimeError(
                "robot_hit_table collision proxy component is malformed"
            )
        component_id = component.get("component_id")
        owner_name = component.get("owner_body_name")
        source_links.add(component.get("source_link_name"))
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
    missing_gripper = sorted(
        set(_A3_COLLISION_PROXY_LEFT_GRIPPER_SOURCE_LINKS) - source_links
    )
    if missing_gripper or document.get("left_gripper_source_links") != list(
        _A3_COLLISION_PROXY_LEFT_GRIPPER_SOURCE_LINKS
    ):
        raise RuntimeError(
            "robot_hit_table collision proxy omits left OmniPicker3 gripper "
            f"collision links: {missing_gripper}"
        )
    return tuple(owner_indices), tuple(centers), tuple(half_axes)


@lru_cache(maxsize=16)
def _load_table_collision_proxy_attribution_labels(
    raw_path: str,
    expected_file_sha256: str,
    expected_body_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read diagnostic labels only after the geometry artifact passed its pin."""

    owners, _centers, _axes = _load_table_collision_proxy_artifact(
        raw_path, expected_file_sha256, expected_body_names
    )
    artifact_path = _resolve_collision_proxy_artifact_path(raw_path)
    document = json.loads(
        artifact_path.read_text(encoding="ascii"),
        object_pairs_hook=_strict_json_object,
    )
    components = document["components"]
    component_ids = tuple(component["component_id"] for component in components)
    owner_names = tuple(expected_body_names[index] for index in owners)
    if (
        len(component_ids) != len(owners)
        or len(set(component_ids)) != len(component_ids)
        or tuple(sorted(component_ids)) != component_ids
    ):
        raise RuntimeError(
            "robot_hit_table attribution labels differ from pinned component order"
        )
    return component_ids, owner_names


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


class TableGuardAttribution(NamedTuple):
    """Per-pair decomposition of the exact full ActionBall table guard.

    ``legacy_mask`` retains its established action-latch field name, but is
    now the exact OBB-vs-AABB terminal verdict. Conservative world-AABB
    overlap is a prefilter and diagnostic only. The pair tensors retain every
    component/obstacle pair so a caller can deterministically latch the first
    positive substep without guessing which body or table part fired.
    """

    legacy_mask: torch.Tensor
    component_conservative_overlap: torch.Tensor
    component_exact_overlap: torch.Tensor
    blade_conservative_overlap: torch.Tensor
    blade_exact_overlap: torch.Tensor
    nonfinite: torch.Tensor


def _obb_aabb_sat_overlap(
    obb_center: torch.Tensor,
    obb_half_axes: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
    *,
    broad_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact OBB-vs-AABB overlap by the 15-axis separating-axis test.

    ``obb_center`` is ``[E,N,3]`` and ``obb_half_axes`` is ``[E,N,3,3]``;
    each of the last-but-one rows is one rotated half-axis vector, not merely
    an extent.  Results are ``[E,N,O]`` for the ``O`` axis-aligned boxes.
    Degenerate cross-product axes impose no constraint, as required by SAT.
    ``broad_phase``, when supplied, is the conservative world-AABB prefilter
    with the same ``[E,N,O]`` shape. SAT is evaluated only for its positive
    pairs; all other pairs are exactly false. No world-AABB approximation is
    used in the final verdict.
    """
    shape = (
        obb_center.shape[0],
        obb_center.shape[1],
        aabb_lo.shape[0],
    )
    if broad_phase is None:
        broad_phase = torch.ones(
            shape, dtype=torch.bool, device=obb_center.device
        )
    elif (
        tuple(broad_phase.shape) != shape
        or broad_phase.dtype != torch.bool
        or broad_phase.device != obb_center.device
    ):
        raise RuntimeError(
            "OBB-vs-AABB SAT broad phase must be same-device bool [env,obb,box]"
        )

    result = torch.zeros_like(broad_phase)
    candidate = torch.nonzero(broad_phase, as_tuple=False)
    env_index, obb_index, box_index = candidate.unbind(dim=1)
    pair_center = obb_center[env_index, obb_index]
    pair_half_axes = obb_half_axes[env_index, obb_index]
    box_center = 0.5 * (aabb_lo + aabb_hi)
    box_half = 0.5 * (aabb_hi - aabb_lo)
    delta = box_center[box_index] - pair_center

    axis_norm = torch.linalg.vector_norm(pair_half_axes, dim=-1)
    safe_norm = torch.clamp(
        axis_norm, min=torch.finfo(pair_center.dtype).tiny
    )
    obb_unit_axes = pair_half_axes / safe_norm[..., None]
    overlap = torch.ones(
        (candidate.shape[0],), dtype=torch.bool, device=obb_center.device
    )

    def apply_axis(axis: torch.Tensor) -> None:
        # axis: [candidate,3]. Projection radii may use an unnormalised axis;
        # every term then carries the same scale and degenerate cross axes
        # reduce to the tautology 0 <= 0.
        separation = torch.abs(torch.sum(delta * axis, dim=-1))
        obb_radius = torch.sum(
            torch.abs(
                torch.sum(pair_half_axes * axis[:, None, :], dim=-1)
            ),
            dim=-1,
        )
        box_radius = torch.sum(
            box_half[box_index] * torch.abs(axis),
            dim=-1,
        )
        overlap.logical_and_(separation <= obb_radius + box_radius)

    world_axes = torch.eye(
        3, dtype=obb_center.dtype, device=obb_center.device
    )
    for world_axis in range(3):
        apply_axis(world_axes[world_axis].expand_as(pair_center))
    for obb_axis in range(3):
        axis = obb_unit_axes[:, obb_axis, :]
        apply_axis(axis)
        for world_axis in range(3):
            apply_axis(
                torch.cross(
                    axis,
                    world_axes[world_axis].expand_as(pair_center),
                    dim=-1,
                )
            )
    result[env_index, obb_index, box_index] = overlap
    return result


def _geometric_table_contact_attribution_unchecked(
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
    terminal_mask: torch.Tensor,
) -> TableGuardAttribution:
    """Compute broad-prefilter and exact-terminal evidence together."""

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
    component_world_half_axes = torch.stack(
        tuple(
            _quat_rotate_wxyz(
                component_quat,
                component_local_half_axes_m[:, local_axis, :]
                .unsqueeze(0)
                .expand(body_pos_w.shape[0], -1, -1),
            )
            for local_axis in range(3)
        ),
        dim=2,
    )
    # Match the terminal kernel's three in-place additions exactly; this is a
    # parity assertion, so even a reduction-order change at a touching edge is
    # not acceptable diagnostic behavior.
    component_world_aabb_half = torch.zeros_like(component_center)
    for local_axis in range(3):
        component_world_aabb_half.add_(
            torch.abs(component_world_half_axes[:, :, local_axis, :])
        )
    component_world_aabb_half.add_(1.0e-6)
    component_lo = component_center - component_world_aabb_half
    component_hi = component_center + component_world_aabb_half
    component_broad = torch.all(
        (component_hi[:, :, None, :] >= aabb_lo[None, None, :, :])
        & (component_lo[:, :, None, :] <= aabb_hi[None, None, :, :]),
        dim=-1,
    )
    component_exact = _obb_aabb_sat_overlap(
        component_center,
        component_world_half_axes,
        aabb_lo,
        aabb_hi,
        broad_phase=component_broad,
    )

    safe_racket_quat = safe_body_quat[:, racket_body_index, :]
    blade_center = p_local[:, racket_body_index, :] + _quat_rotate_wxyz(
        safe_racket_quat,
        racket_blade_center_offset_wrist_m.expand_as(p_local[:, 0]),
    )
    blade_world_half_axes = torch.stack(
        tuple(
            _quat_rotate_wxyz(
                safe_racket_quat,
                racket_blade_local_half_axes_m[local_axis]
                .unsqueeze(0)
                .expand(body_pos_w.shape[0], -1),
            )
            for local_axis in range(3)
        ),
        dim=1,
    )
    blade_world_aabb_half = torch.sum(
        torch.abs(blade_world_half_axes), dim=1
    )
    blade_lo = blade_center - blade_world_aabb_half
    blade_hi = blade_center + blade_world_aabb_half
    blade_broad = torch.all(
        (blade_hi[:, None, :] >= aabb_lo[None, :, :])
        & (blade_lo[:, None, :] <= aabb_hi[None, :, :]),
        dim=-1,
    )
    blade_exact = _obb_aabb_sat_overlap(
        blade_center[:, None, :],
        blade_world_half_axes[:, None, :, :],
        aabb_lo,
        aabb_hi,
        broad_phase=blade_broad[:, None, :],
    )[:, 0, :]

    nonfinite = (
        ~torch.isfinite(body_pos_w).all(dim=(1, 2))
        | ~torch.isfinite(body_quat_w).all(dim=(1, 2))
        | ~torch.isfinite(env_origins).all(dim=1)
        | ~(body_quat_norm_sq[..., 0] > 0.0).all(dim=1)
    )
    valid = ~nonfinite
    component_broad &= valid[:, None, None]
    blade_broad &= valid[:, None]
    component_exact &= valid[:, None, None]
    blade_exact &= valid[:, None]
    exact_terminal_union = (
        torch.any(component_exact, dim=(1, 2))
        | torch.any(blade_exact, dim=1)
        | nonfinite
    )
    if body_pos_w.device.type == "cpu":
        if not torch.equal(exact_terminal_union, terminal_mask):
            raise RuntimeError(
                "table-guard attribution exact phase differs from terminal mask"
            )
    else:
        torch._assert_async(torch.all(exact_terminal_union == terminal_mask))
    return TableGuardAttribution(
        legacy_mask=terminal_mask,
        component_conservative_overlap=component_broad,
        component_exact_overlap=component_exact,
        blade_conservative_overlap=blade_broad,
        blade_exact_overlap=blade_exact,
        nonfinite=nonfinite,
    )


def geometric_table_contact_attribution(
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
) -> TableGuardAttribution:
    """Return exact terminal evidence plus conservative prefilter evidence."""

    terminal_mask = geometric_table_contact_hit_mask(
        body_pos_w,
        body_quat_w,
        env_origins,
        component_body_indices,
        component_local_center_m,
        component_local_half_axes_m,
        aabb_lo,
        aabb_hi,
        racket_body_index=racket_body_index,
        racket_blade_center_offset_wrist_m=(
            racket_blade_center_offset_wrist_m
        ),
        racket_blade_local_half_axes_m=racket_blade_local_half_axes_m,
    )
    return _geometric_table_contact_attribution_unchecked(
        body_pos_w,
        body_quat_w,
        env_origins,
        component_body_indices,
        component_local_center_m,
        component_local_half_axes_m,
        aabb_lo,
        aabb_hi,
        racket_body_index=racket_body_index,
        racket_blade_center_offset_wrist_m=(
            racket_blade_center_offset_wrist_m
        ),
        racket_blade_local_half_axes_m=racket_blade_local_half_axes_m,
        terminal_mask=terminal_mask,
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
    component_world_half_axes = torch.stack(
        tuple(
            _quat_rotate_wxyz(
                component_quat,
                component_local_half_axes_m[:, local_axis, :]
                .unsqueeze(0)
                .expand(body_pos_w.shape[0], -1, -1),
            )
            for local_axis in range(3)
        ),
        dim=2,
    )
    component_world_aabb_half = torch.zeros_like(component_center)
    for local_axis in range(3):
        component_world_aabb_half.add_(
            torch.abs(component_world_half_axes[:, :, local_axis, :])
        )
    # Cover float64-artifact -> runtime-dtype conversion and quaternion arithmetic
    # without recreating the centimetre-scale false positives of the retired
    # uniform spheres.  One micrometre is below both PhysX contact offsets and the
    # table guard's configured margin.
    component_world_aabb_half.add_(1.0e-6)
    component_lo = component_center - component_world_aabb_half
    component_hi = component_center + component_world_aabb_half

    component_broad = torch.ones(
        (
            body_pos_w.shape[0],
            component_body_indices.shape[0],
            aabb_lo.shape[0],
        ),
        device=body_pos_w.device,
        dtype=torch.bool,
    )
    for axis in range(3):
        component_broad.logical_and_(
            (
                component_hi[..., axis, None]
                >= aabb_lo[None, None, :, axis]
            )
            & (
                component_lo[..., axis, None]
                <= aabb_hi[None, None, :, axis]
            )
        )
    component_exact = _obb_aabb_sat_overlap(
        component_center,
        component_world_half_axes,
        aabb_lo,
        aabb_hi,
        broad_phase=component_broad,
    )
    body_hit = torch.any(component_exact, dim=(1, 2))

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
    blade_world_half_axes = _quat_rotate_wxyz(blade_quat, local_half_axes)
    blade_world_aabb_half = torch.sum(
        torch.abs(blade_world_half_axes), dim=1
    )
    blade_lo = blade_center_local - blade_world_aabb_half
    blade_hi = blade_center_local + blade_world_aabb_half
    blade_broad = torch.all(
        (blade_hi[:, None, :] >= aabb_lo[None, :, :])
        & (blade_lo[:, None, :] <= aabb_hi[None, :, :]),
        dim=-1,
    )
    blade_exact = _obb_aabb_sat_overlap(
        blade_center_local[:, None, :],
        blade_world_half_axes[:, None, :, :],
        aabb_lo,
        aabb_hi,
        broad_phase=blade_broad[:, None, :],
    )[:, 0, :]
    racket_hit = torch.any(blade_exact, dim=1)

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
        "_attribution_enabled",
        "_component_ids",
        "_component_owner_names",
        "_obstacle_roles",
        "_attribution_command",
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
        attribution_enabled: bool = False,
        component_ids: tuple[str, ...] = (),
        component_owner_names: tuple[str, ...] = (),
        obstacle_roles: tuple[str, ...] = (),
        attribution_command=None,
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
        self._attribution_enabled = bool(attribution_enabled)
        self._component_ids = tuple(component_ids)
        self._component_owner_names = tuple(component_owner_names)
        self._obstacle_roles = tuple(obstacle_roles)
        self._attribution_command = attribution_command
        self.runtime_usd_receipt = runtime_usd_receipt

    def __call__(self) -> torch.Tensor:
        # All names, paths, shapes, dtypes and static tensors were verified by
        # ``prepare_robot_table_pose_guard``.  Per physics substep this path
        # performs only tensor selection, a broad AABB prefilter and exact SAT.
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

    @property
    def attribution_enabled(self) -> bool:
        """Whether diagnostic SAT attribution was explicitly configured."""

        return self._attribution_enabled

    def sample_with_attribution(self) -> TableGuardAttribution:
        """Sample the exact terminal verdict plus its per-pair evidence."""

        if not self._attribution_enabled:
            raise RuntimeError(
                "table-guard attribution was not enabled for this prepared guard"
            )
        body_pos = torch.index_select(
            self._asset.data.body_pos_w, 1, self._asset_body_indices
        )
        body_quat = torch.index_select(
            self._asset.data.body_quat_w, 1, self._asset_body_indices
        )
        terminal_mask = _geometric_table_contact_hit_mask_unchecked(
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
        return _geometric_table_contact_attribution_unchecked(
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
            terminal_mask=terminal_mask,
        )

    def record_first_hits(
        self,
        first_hit_mask: torch.Tensor,
        attribution: TableGuardAttribution,
    ) -> None:
        """Forward a substep latch's newly-positive rows to the command ledger."""

        if not self._attribution_enabled or self._attribution_command is None:
            raise RuntimeError(
                "table-guard attribution recorder is not configured"
            )
        recorder = getattr(
            self._attribution_command,
            "record_table_guard_first_hits",
            None,
        )
        if not callable(recorder):
            raise RuntimeError(
                "racket-target command lacks table-guard attribution recorder"
            )
        recorder(first_hit_mask, attribution)


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
    attribution_diagnostic: bool = False,
    attribution_command_name: str = "racket_target",
) -> _PreparedRobotTablePoseGuard:
    """Validate and materialize every static full-table guard input once."""

    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame

    if type(attribution_diagnostic) is not bool:
        raise RuntimeError(
            "table-guard attribution_diagnostic must be one explicit boolean"
        )
    if (
        not isinstance(attribution_command_name, str)
        or not attribution_command_name
    ):
        raise RuntimeError(
            "table-guard attribution command name must be one non-empty string"
        )

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
        bool(attribution_diagnostic),
        str(attribution_command_name),
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
    component_ids: tuple[str, ...] = ()
    component_owner_names: tuple[str, ...] = ()
    attribution_command = None
    if attribution_diagnostic:
        command_manager = getattr(env, "command_manager", None)
        get_term = getattr(command_manager, "get_term", None)
        if not callable(get_term):
            raise RuntimeError(
                "table-guard attribution requires the command manager"
            )
        attribution_command = get_term(attribution_command_name)
        configure = getattr(
            attribution_command,
            "configure_table_guard_attribution",
            None,
        )
        if not callable(configure):
            raise RuntimeError(
                "racket-target command lacks table-guard attribution configuration"
            )
        (
            component_ids,
            component_owner_names,
        ) = _load_table_collision_proxy_attribution_labels(
            collision_proxy_artifact_path,
            collision_proxy_artifact_sha256,
            expected_names,
        )
        configure(
            component_ids=component_ids,
            component_owner_names=component_owner_names,
            obstacle_roles=_TABLE_GUARD_OBSTACLE_ROLES,
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
        attribution_enabled=bool(attribution_diagnostic),
        component_ids=component_ids,
        component_owner_names=component_owner_names,
        obstacle_roles=(
            _TABLE_GUARD_OBSTACLE_ROLES if attribution_diagnostic else ()
        ),
        attribution_command=attribution_command,
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
    attribution_diagnostic: bool = False,
    attribution_command_name: str = "racket_target",
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
            attribution_diagnostic=attribution_diagnostic,
            attribution_command_name=attribution_command_name,
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
    attribution_diagnostic: bool = False,
    attribution_command_name: str = "racket_target",
) -> torch.Tensor:
    """The robot violated the table assembly guard.  Terminal, exactly like falling over.

    Legacy top-only mode keeps the broad non-foot/body-origin channel plus one exact wrist/racket
    pair channel. ActionBall first uses conservative world-AABB overlap to select candidate pairs,
    then terminates only on exact OBB-vs-table-AABB SAT overlap of its materialized
    collision-component OBBs or live racket-blade OBB; it does not read a ``ContactSensor``.
    A full-assembly positive is exact for this guard geometry, not proof of resolved physical
    contact.
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
        attribution_diagnostic=attribution_diagnostic,
        attribution_command_name=attribution_command_name,
    )
