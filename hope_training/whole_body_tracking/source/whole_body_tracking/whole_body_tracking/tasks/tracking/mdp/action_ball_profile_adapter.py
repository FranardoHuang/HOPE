"""Strict bridge from schema-v3 manifests to schema-v3 sampler profiles.

The manifest intentionally stores base spawn and travel as two-dimensional
domains: canonical-ready owns base z and this curriculum must never sample it.
The sampler uses three-dimensional vectors for uniform transform math.  This
adapter is the sole conversion point: base-spawn center/min/max receive the
selected clip's exact canonical-ready root z, while every z width and every
base-travel z remains exactly ``0.0``.

Mobility is a manifest-level run identity.  It is copied into every immutable
``SamplingProfile`` and is not accepted as an adapter or sampler override.
Consequently, no-move and move runs may share byte-identical latent travel
parameters while still receiving different manifest, profile, and adapter
contract digests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import json
from typing import Dict, Tuple

if __package__:
    from .action_ball_manifest import (
        ActionBallAction,
        ActionBallManifest,
        canonical_manifest_sha256,
    )
    from .action_ball_sampling import SamplingProfile
    from .action_ball_curriculum import BallCurriculumConfig
else:  # pragma: no cover - exercised by host-only spec loaders
    from action_ball_manifest import (
        ActionBallAction,
        ActionBallManifest,
        canonical_manifest_sha256,
    )
    from action_ball_sampling import SamplingProfile
    from action_ball_curriculum import BallCurriculumConfig


ADAPTER_CONTRACT_SCHEMA_VERSION = 2


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _xy_with_explicit_zero_z(
    value: Tuple[float, float],
    *,
    name: str,
    z: float = 0.0,
) -> Tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a validated length-2 tuple")
    x, y = value
    if type(x) is not float or type(y) is not float:
        raise ValueError(f"{name} must contain validated floats")
    if type(z) is not float or not math.isfinite(z):
        raise ValueError(f"{name} injected z must be a finite float")
    return (x, y, z)


def _validated_manifest(
    manifest: ActionBallManifest,
) -> ActionBallManifest:
    if not isinstance(manifest, ActionBallManifest):
        raise TypeError("manifest must be an ActionBallManifest")
    # Frozen dataclasses can still be constructed directly.  Round-tripping
    # through the strict schema prevents the adapter becoming a bypass around
    # the loader's exact-key, unit-vector, interval, and bool-number checks.
    validated = ActionBallManifest.from_mapping(manifest.to_mapping())
    if validated != manifest:
        raise ValueError(
            "manifest does not round-trip through its strict schema"
        )
    return validated


def _build_profile(
    manifest: ActionBallManifest,
    action: ActionBallAction,
    *,
    ready_root_z: float = 0.0,
) -> SamplingProfile:
    ball = action.ball_profile
    aim = manifest.landing_aim
    profile = SamplingProfile(
        action_uid=action.action_uid,
        contact_offset_center_b_yaw_m=(
            ball.contact_offset_center_b_yaw_m
        ),
        contact_offset_std_lower_initial_m=(
            ball.contact_offset_std_lower_initial_m
        ),
        contact_offset_std_lower_max_m=(
            ball.contact_offset_std_lower_max_m
        ),
        contact_offset_std_upper_initial_m=(
            ball.contact_offset_std_upper_initial_m
        ),
        contact_offset_std_upper_max_m=(
            ball.contact_offset_std_upper_max_m
        ),
        contact_offset_min_b_yaw_m=ball.contact_offset_min_b_yaw_m,
        contact_offset_max_b_yaw_m=ball.contact_offset_max_b_yaw_m,
        time_to_contact_center_s=ball.time_to_contact_center_s,
        time_to_contact_std_lower_initial_s=(
            ball.time_to_contact_std_lower_initial_s
        ),
        time_to_contact_std_lower_max_s=(
            ball.time_to_contact_std_lower_max_s
        ),
        time_to_contact_std_upper_initial_s=(
            ball.time_to_contact_std_upper_initial_s
        ),
        time_to_contact_std_upper_max_s=(
            ball.time_to_contact_std_upper_max_s
        ),
        time_to_contact_min_s=ball.time_to_contact_min_s,
        time_to_contact_max_s=ball.time_to_contact_max_s,
        incoming_direction_center_b_yaw=(
            ball.incoming_direction_center_b_yaw
        ),
        incoming_direction_tangent_u_b_yaw=(
            ball.incoming_direction_tangent_u_b_yaw
        ),
        incoming_direction_tangent_v_b_yaw=(
            ball.incoming_direction_tangent_v_b_yaw
        ),
        incoming_direction_tangent_u_neg_initial_deg=(
            ball.incoming_direction_tangent_u_neg_initial_deg
        ),
        incoming_direction_tangent_u_neg_max_deg=(
            ball.incoming_direction_tangent_u_neg_max_deg
        ),
        incoming_direction_tangent_u_pos_initial_deg=(
            ball.incoming_direction_tangent_u_pos_initial_deg
        ),
        incoming_direction_tangent_u_pos_max_deg=(
            ball.incoming_direction_tangent_u_pos_max_deg
        ),
        incoming_direction_tangent_v_neg_initial_deg=(
            ball.incoming_direction_tangent_v_neg_initial_deg
        ),
        incoming_direction_tangent_v_neg_max_deg=(
            ball.incoming_direction_tangent_v_neg_max_deg
        ),
        incoming_direction_tangent_v_pos_initial_deg=(
            ball.incoming_direction_tangent_v_pos_initial_deg
        ),
        incoming_direction_tangent_v_pos_max_deg=(
            ball.incoming_direction_tangent_v_pos_max_deg
        ),
        incoming_inbound_axis_b_yaw=ball.incoming_inbound_axis_b_yaw,
        incoming_inbound_min_cosine=ball.incoming_inbound_min_cosine,
        incoming_speed_center_mps=ball.incoming_speed_center_mps,
        incoming_speed_std_lower_initial_mps=(
            ball.incoming_speed_std_lower_initial_mps
        ),
        incoming_speed_std_lower_max_mps=(
            ball.incoming_speed_std_lower_max_mps
        ),
        incoming_speed_std_upper_initial_mps=(
            ball.incoming_speed_std_upper_initial_mps
        ),
        incoming_speed_std_upper_max_mps=(
            ball.incoming_speed_std_upper_max_mps
        ),
        incoming_speed_min_mps=ball.incoming_speed_min_mps,
        incoming_speed_max_mps=ball.incoming_speed_max_mps,
        spin_direction_center_b_yaw=ball.spin_direction_center_b_yaw,
        spin_direction_tangent_u_b_yaw=(
            ball.spin_direction_tangent_u_b_yaw
        ),
        spin_direction_tangent_v_b_yaw=(
            ball.spin_direction_tangent_v_b_yaw
        ),
        spin_direction_tangent_u_neg_initial_deg=(
            ball.spin_direction_tangent_u_neg_initial_deg
        ),
        spin_direction_tangent_u_neg_max_deg=(
            ball.spin_direction_tangent_u_neg_max_deg
        ),
        spin_direction_tangent_u_pos_initial_deg=(
            ball.spin_direction_tangent_u_pos_initial_deg
        ),
        spin_direction_tangent_u_pos_max_deg=(
            ball.spin_direction_tangent_u_pos_max_deg
        ),
        spin_direction_tangent_v_neg_initial_deg=(
            ball.spin_direction_tangent_v_neg_initial_deg
        ),
        spin_direction_tangent_v_neg_max_deg=(
            ball.spin_direction_tangent_v_neg_max_deg
        ),
        spin_direction_tangent_v_pos_initial_deg=(
            ball.spin_direction_tangent_v_pos_initial_deg
        ),
        spin_direction_tangent_v_pos_max_deg=(
            ball.spin_direction_tangent_v_pos_max_deg
        ),
        spin_magnitude_center_radps=ball.spin_magnitude_center_radps,
        spin_magnitude_std_lower_initial_radps=(
            ball.spin_magnitude_std_lower_initial_radps
        ),
        spin_magnitude_std_lower_max_radps=(
            ball.spin_magnitude_std_lower_max_radps
        ),
        spin_magnitude_std_upper_initial_radps=(
            ball.spin_magnitude_std_upper_initial_radps
        ),
        spin_magnitude_std_upper_max_radps=(
            ball.spin_magnitude_std_upper_max_radps
        ),
        spin_magnitude_min_radps=ball.spin_magnitude_min_radps,
        spin_magnitude_max_radps=ball.spin_magnitude_max_radps,
        base_spawn_center_w_m=_xy_with_explicit_zero_z(
            ball.base_spawn_center_w_xy_m,
            name="base_spawn_center_w_xy_m",
            z=ready_root_z,
        ),
        base_spawn_std_lower_initial_m=_xy_with_explicit_zero_z(
            ball.base_spawn_std_lower_initial_m,
            name="base_spawn_std_lower_initial_m",
        ),
        base_spawn_std_lower_max_m=_xy_with_explicit_zero_z(
            ball.base_spawn_std_lower_max_m,
            name="base_spawn_std_lower_max_m",
        ),
        base_spawn_std_upper_initial_m=_xy_with_explicit_zero_z(
            ball.base_spawn_std_upper_initial_m,
            name="base_spawn_std_upper_initial_m",
        ),
        base_spawn_std_upper_max_m=_xy_with_explicit_zero_z(
            ball.base_spawn_std_upper_max_m,
            name="base_spawn_std_upper_max_m",
        ),
        base_spawn_min_w_m=_xy_with_explicit_zero_z(
            ball.base_spawn_min_w_xy_m,
            name="base_spawn_min_w_xy_m",
            z=ready_root_z,
        ),
        base_spawn_max_w_m=_xy_with_explicit_zero_z(
            ball.base_spawn_max_w_xy_m,
            name="base_spawn_max_w_xy_m",
            z=ready_root_z,
        ),
        base_travel_center_b_yaw_m=_xy_with_explicit_zero_z(
            ball.base_travel_center_b_yaw_xy_m,
            name="base_travel_center_b_yaw_xy_m",
        ),
        base_travel_std_lower_initial_m=_xy_with_explicit_zero_z(
            ball.base_travel_std_lower_initial_m,
            name="base_travel_std_lower_initial_m",
        ),
        base_travel_std_lower_max_m=_xy_with_explicit_zero_z(
            ball.base_travel_std_lower_max_m,
            name="base_travel_std_lower_max_m",
        ),
        base_travel_std_upper_initial_m=_xy_with_explicit_zero_z(
            ball.base_travel_std_upper_initial_m,
            name="base_travel_std_upper_initial_m",
        ),
        base_travel_std_upper_max_m=_xy_with_explicit_zero_z(
            ball.base_travel_std_upper_max_m,
            name="base_travel_std_upper_max_m",
        ),
        base_travel_min_b_yaw_m=_xy_with_explicit_zero_z(
            ball.base_travel_min_b_yaw_xy_m,
            name="base_travel_min_b_yaw_xy_m",
        ),
        base_travel_max_b_yaw_m=_xy_with_explicit_zero_z(
            ball.base_travel_max_b_yaw_xy_m,
            name="base_travel_max_b_yaw_xy_m",
        ),
        landing_aim_center_w_xy_m=aim.center_w_xy_m,
        landing_aim_std_lower_initial_m=aim.std_lower_initial_m,
        landing_aim_std_lower_max_m=aim.std_lower_max_m,
        landing_aim_std_upper_initial_m=aim.std_upper_initial_m,
        landing_aim_std_upper_max_m=aim.std_upper_max_m,
        landing_aim_min_w_xy_m=aim.min_w_xy_m,
        landing_aim_max_w_xy_m=aim.max_w_xy_m,
        reference_t_hit_s=action.reference_t_hit_s,
        reference_t_cycle_s=action.reference_t_cycle_s,
        reference_racket_site_speed_mps=(
            action.reference_racket_site_speed_mps
        ),
        reaction_margin_s=action.reaction_margin_s,
        teacher_rate_min=action.teacher_rate_min,
        teacher_rate_max=action.teacher_rate_max,
        mobility_mode=manifest.mobility_mode,
        counter_rally_objective=manifest.counter_rally_objective,
    )
    for name in (
        "base_spawn_std_lower_initial_m",
        "base_spawn_std_lower_max_m",
        "base_spawn_std_upper_initial_m",
        "base_spawn_std_upper_max_m",
        "base_travel_center_b_yaw_m",
        "base_travel_std_lower_initial_m",
        "base_travel_std_lower_max_m",
        "base_travel_std_upper_initial_m",
        "base_travel_std_upper_max_m",
        "base_travel_min_b_yaw_m",
        "base_travel_max_b_yaw_m",
    ):
        if getattr(profile, name)[2] != 0.0:
            raise AssertionError(f"{name} implicit z must be exactly zero")
    for name in (
        "base_spawn_center_w_m",
        "base_spawn_min_w_m",
        "base_spawn_max_w_m",
    ):
        if getattr(profile, name)[2] != ready_root_z:
            raise AssertionError(
                f"{name} z must equal the selected canonical-ready root z"
            )
    return profile


@dataclass(frozen=True)
class AdaptedSamplingProfiles:
    """Manifest-ordered sampler profiles plus a checkpoint-safe binding."""

    manifest_canonical_sha256: str
    mobility_mode: str
    action_order: Tuple[str, ...]
    action_uids: Tuple[int, ...]
    profiles: Tuple[SamplingProfile, ...]

    def __post_init__(self) -> None:
        if (
            type(self.manifest_canonical_sha256) is not str
            or len(self.manifest_canonical_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.manifest_canonical_sha256
            )
        ):
            raise ValueError(
                "manifest_canonical_sha256 must be 64 lowercase hex"
            )
        if self.mobility_mode not in ("no_move", "move"):
            raise ValueError("mobility_mode must be 'no_move' or 'move'")
        if type(self.action_order) is not tuple:
            raise TypeError("action_order must be a tuple")
        if type(self.action_uids) is not tuple:
            raise TypeError("action_uids must be a tuple")
        if type(self.profiles) is not tuple:
            raise TypeError("profiles must be a tuple")
        count = len(self.action_order)
        if count == 0:
            raise ValueError("adapted profile bundle must be non-empty")
        if len(self.action_uids) != count or len(self.profiles) != count:
            raise ValueError("adapted profile bundle columns must align")
        if any(
            type(action_id) is not str or not action_id
            for action_id in self.action_order
        ):
            raise ValueError("action_order must contain non-empty strings")
        if len(set(self.action_order)) != count:
            raise ValueError("action_order must not contain duplicates")
        if any(
            type(action_uid) is not int or action_uid < 1
            for action_uid in self.action_uids
        ):
            raise ValueError("action_uids must contain positive plain integers")
        if len(set(self.action_uids)) != count:
            raise ValueError("action_uids must not contain duplicates")
        if any(
            not isinstance(profile, SamplingProfile)
            for profile in self.profiles
        ):
            raise TypeError("profiles must contain SamplingProfile values")
        if tuple(profile.action_uid for profile in self.profiles) != (
            self.action_uids
        ):
            raise ValueError(
                "adapted profiles must preserve ordered action_uids"
            )
        if any(
            profile.mobility_mode != self.mobility_mode
            for profile in self.profiles
        ):
            raise ValueError(
                "every profile must use the manifest mobility_mode"
            )

    @property
    def profile_sha256(self) -> Tuple[str, ...]:
        return tuple(profile.sha256 for profile in self.profiles)

    def to_contract(self) -> Dict[str, object]:
        return {
            "schema_version": ADAPTER_CONTRACT_SCHEMA_VERSION,
            "manifest_canonical_sha256": self.manifest_canonical_sha256,
            "mobility_mode": self.mobility_mode,
            "profiles": [
                {
                    "action_id": action_id,
                    "action_uid": action_uid,
                    "sampling_profile_sha256": profile_sha256,
                }
                for action_id, action_uid, profile_sha256 in zip(
                    self.action_order,
                    self.action_uids,
                    self.profile_sha256,
                )
            ],
        }

    @property
    def contract_sha256(self) -> str:
        return _canonical_sha256(self.to_contract())

    def profile_for_action_id(self, action_id: str) -> SamplingProfile:
        if type(action_id) is not str:
            raise TypeError("action_id must be a string")
        try:
            index = self.action_order.index(action_id)
        except ValueError as error:
            raise ValueError(f"unknown action_id {action_id!r}") from error
        return self.profiles[index]


def adapt_action_ball_manifest(
    manifest: ActionBallManifest,
    *,
    ready_root_z_by_slot: Tuple[float, ...] = (),
) -> AdaptedSamplingProfiles:
    """Strictly convert one validated manifest in exact action order.

    ``ready_root_z_by_slot``: runtime/preflight 共用的逐动作 canonical-ready
    root Z（base_spawn 的常数 z，非课程轴）。空元组只保留给不绑定真实
    motion bytes 的离线 profile 工具，z 取 0.0。
    """

    validated = _validated_manifest(manifest)
    if ready_root_z_by_slot and len(ready_root_z_by_slot) != len(validated.actions):
        raise ValueError(
            "ready_root_z_by_slot must have exactly one z per manifest action"
        )
    profiles = tuple(
        _build_profile(
            validated,
            action,
            ready_root_z=(
                float(ready_root_z_by_slot[slot])
                if ready_root_z_by_slot
                else 0.0
            ),
        )
        for slot, action in enumerate(validated.actions)
    )
    return AdaptedSamplingProfiles(
        manifest_canonical_sha256=canonical_manifest_sha256(validated),
        mobility_mode=validated.mobility_mode,
        action_order=validated.action_order,
        action_uids=tuple(
            action.action_uid for action in validated.actions
        ),
        profiles=profiles,
    )


def build_sampling_profiles(
    manifest: ActionBallManifest,
) -> Tuple[SamplingProfile, ...]:
    """Return manifest-ordered profiles ready for ``ActionBallSampler``."""

    return adapt_action_ball_manifest(manifest).profiles


def build_sampling_profile(
    manifest: ActionBallManifest,
    *,
    action_id: str,
) -> SamplingProfile:
    """Return one named profile without allowing a mobility override."""

    return adapt_action_ball_manifest(manifest).profile_for_action_id(
        action_id
    )


def build_curriculum_config(
    manifest: ActionBallManifest,
) -> BallCurriculumConfig:
    """Build the exact controller config without a runtime field guess.

    Manifest schema v3 intentionally uses the same names and denominator
    semantics as ``BallCurriculumConfig``.  Constructing it explicitly here
    makes future schema drift fail as a test/import boundary rather than
    silently defaulting install/start/close integrity gates.
    """

    validated = _validated_manifest(manifest)
    declared = validated.curriculum
    config = BallCurriculumConfig(
        min_proposals=declared.min_proposals,
        min_safe_closed=declared.min_safe_closed,
        target_failure_rate=declared.target_failure_rate,
        failure_band_half_width=declared.failure_band_half_width,
        min_solver_admit_rate=declared.min_solver_admit_rate,
        min_install_rate=declared.min_install_rate,
        min_start_rate=declared.min_start_rate,
        min_close_rate=declared.min_close_rate,
        max_other_unsafe_rate=declared.max_other_unsafe_rate,
        confidence_z=declared.confidence_z,
        max_center_failures=declared.max_center_failures,
        objective_inactive_arms=(
            ()
            if validated.counter_rally_objective is None
            else tuple(
                validated.counter_rally_objective.inactive_curriculum_arms
            )
        ),
    )
    expected_mapping = declared.to_mapping()
    if validated.counter_rally_objective is not None:
        expected_mapping = {
            **expected_mapping,
            "objective_inactive_arms": list(
                validated.counter_rally_objective.inactive_curriculum_arms
            ),
        }
    if config.as_dict() != expected_mapping:
        raise AssertionError(
            "manifest/controller curriculum config mapping drifted"
        )
    return config
