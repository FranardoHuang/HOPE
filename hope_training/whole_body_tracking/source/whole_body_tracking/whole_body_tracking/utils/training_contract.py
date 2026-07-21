"""Immutable training/export execution-contract helpers.

This module deliberately has no Isaac, Torch, Hydra, or ONNX imports.  It is shared by the
training entry point and both export paths, and its duck-typed runtime extractor is covered by
dependency-light tests.  Schema 3 is the first schema that binds the policy's execution values
(joint/action order, decoder, nominal PD envelope, q-des limits, timing, body/reference order and
the exact actor layout) rather than only task-level configuration.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping


TRAINING_CONTRACT_SCHEMA_VERSION = 3
ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH = 1
ACTOR_LEG_REF_MASK_PROVENANCE_KEY = "actor_leg_ref_mask_provenance_epoch"
ACTOR_LEG_REF_MASK_PROVENANCE_BINDING_KEY = "actor_leg_ref_mask_provenance_sha256"
CHECKPOINT_CONTRACT_SCHEMA_KEY = "training_contract_schema_version"
CHECKPOINT_CONTRACT_SHA_KEY = "training_contract_sha256"
CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY = "training_contract_lineage_exact"
CHECKPOINT_LAUNCH_CLAIM_SHA_KEY = "training_launch_claim_sha256"
SCHEMA3_TASK_KEYS = (
    "racket_control_point",
    "racket_control_point_offset_wrist_m",
)

PLANNER_TASK_REVISION_KEY = "planner_task_revision"
PLANNER_TASK_REVISION_TRAINING_KEY = "planner_task_revision_training"
_PLANNER_TASK_REVISION_KEYS = frozenset(
    {"enabled", "revision_schema_version", "governor", "initial_tts_range_s"}
)
_PLANNER_GOVERNOR_KEYS = frozenset(
    {"contract_version", "schema_version", "profile_sha256", "profile"}
)
_PLANNER_GOVERNOR_PROFILE_KEYS = frozenset(
    {
        "policy_dt_s",
        "min_tts_s",
        "max_tts_s",
        "max_phase_rate_per_s",
        "max_phase_acceleration_per_s2",
        "max_deadline_revision_delta_s",
        "max_position_revision_delta_m",
        "max_velocity_revision_delta_mps",
        "max_normal_revision_delta_rad",
        "normal_unit_tolerance",
        "early_deadline_tolerance_s",
        "contract_version",
        "schema_version",
    }
)
_PLANNER_TASK_REVISION_TRAINING_KEYS = frozenset(
    {
        "initial_tts_sampling_semantics",
        "initial_tts_mixture",
        "initial_feasibility_gate",
        "dynamics_certified_action_tau_min_bound",
        "timing_exam_semantics",
        "position_std_m",
        "velocity_std_mps",
        "normal_std_rad",
        "tts_std_s",
        "truth_fields_immutable",
        "actor_revision_fields",
    }
)
_PLANNER_INITIAL_TTS_MIXTURE_KEYS = frozenset(
    {"contract_version", "components"}
)
_PLANNER_INITIAL_TTS_COMPONENT_KEYS = frozenset(
    {"name", "range_s", "weight"}
)

# Isaac Lab 2.1 passes ``ImplicitActuatorCfg.friction`` to PhysX as a dimensionless,
# load-dependent joint-friction coefficient.  It is *not* MuJoCo ``frictionloss`` (a constant
# Coulomb torque in N m).  Keep these strings in the immutable contract so a consumer cannot
# silently copy the same-looking numbers between physics backends and call that exact parity.
JOINT_FRICTION_BACKEND = "physx"
JOINT_FRICTION_SEMANTICS = "load_dependent_spatial_force_coefficient"
JOINT_FRICTION_UNITS = "dimensionless"
MOTION_BODY_LIN_VEL_POINTS = ("center_of_mass", "link_origin")

RUNTIME_EXECUTION_KEYS = (
    "articulation_joint_names",
    "action_joint_ids",
    "joint_names",
    "default_joint_pos",
    "action_scale",
    "joint_stiffness",
    "joint_damping",
    "joint_effort_limits",
    "joint_actuator_types",
    "joint_armature",
    "joint_friction_coefficients",
    "joint_velocity_limits",
    "joint_friction_backend",
    "joint_friction_semantics",
    "joint_friction_units",
    "qdes_joint_pos_limits",
    "action_use_default_offset",
    "qdes_clamp",
    "physics_step_dt_s",
    "policy_step_dt_s",
    "control_decimation",
    "actor_obs_contract",
    "actor_obs_mode",
    "actor_obs_total_dim",
    "actor_obs_term_names",
    "actor_obs_term_dims",
    "observation_history_lengths",
    "articulation_body_names",
    "body_names",
    "body_indices",
    "anchor_body_name",
    "anchor_body_index",
    "motion_segment_lengths",
    "motion_clip_fps",
    "motion_kinematics_contracts",
    "motion_kinematics_exact",
)


def _require_exact_mapping_keys(value: object, expected: frozenset[str], *, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    return value


def _planner_finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _canonical_planner_json(value: Mapping, *, trailing_newline: bool) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("planner task revision contract is not canonical JSON data") from exc
    return encoded + ("\n" if trailing_newline else "")


def _validate_planner_initial_tts_mixture(
    value: object, *, support_lo: float, support_hi: float
) -> None:
    """Validate the complete training-only mixture without leaking it into ONNX metadata."""

    mixture = _require_exact_mapping_keys(
        value,
        _PLANNER_INITIAL_TTS_MIXTURE_KEYS,
        name="planner_task_revision_training.initial_tts_mixture",
    )
    if mixture["contract_version"] != "initial_tts_mixture_v1":
        raise ValueError("planner initial-TTS mixture contract version is unsupported")
    components = mixture["components"]
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("planner initial-TTS mixture components must be a non-empty list")
    names: set[str] = set()
    total_weight = 0.0
    component_lows: list[float] = []
    component_highs: list[float] = []
    has_exact_0p5 = False
    has_sub_0p5_stress = False
    for index, raw_component in enumerate(components):
        component = _require_exact_mapping_keys(
            raw_component,
            _PLANNER_INITIAL_TTS_COMPONENT_KEYS,
            name=f"planner initial-TTS mixture component {index}",
        )
        name = component["name"]
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError(
                f"planner initial-TTS mixture component {index} name is invalid"
            )
        if name in names:
            raise ValueError("planner initial-TTS mixture component names must be unique")
        names.add(name)
        interval = component["range_s"]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(
                f"planner initial-TTS mixture component {index} range_s must contain two values"
            )
        lo = _planner_finite_number(
            interval[0], name=f"planner initial-TTS component {index} lo_s", minimum=0.0
        )
        hi = _planner_finite_number(
            interval[1], name=f"planner initial-TTS component {index} hi_s", minimum=lo
        )
        weight = _planner_finite_number(
            component["weight"],
            name=f"planner initial-TTS component {index} weight",
            strictly_positive=True,
        )
        total_weight += weight
        component_lows.append(lo)
        component_highs.append(hi)
        has_exact_0p5 |= lo == 0.5 and hi == 0.5
        has_sub_0p5_stress |= lo < 0.5
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("planner initial-TTS mixture weights must sum to 1")
    if min(component_lows) != support_lo or max(component_highs) != support_hi:
        raise ValueError(
            "planner initial-TTS mixture support must equal initial_tts_range_s"
        )
    if not has_exact_0p5:
        raise ValueError("planner initial-TTS mixture requires an exact 0.5 s point mass")
    if not has_sub_0p5_stress:
        raise ValueError("planner initial-TTS mixture requires a sub-0.5 s stress stratum")


def planner_task_revision_metadata(contract: Mapping) -> str | None:
    """Return the sole C++ metadata JSON for a checkpoint-bound revision profile.

    Absence of both revision keys is the byte-compatible legacy/OFF spelling and returns ``None``.
    Any partial spelling fails closed.  The returned JSON is reconstructed only from the immutable
    schema-3 sidecar; callers must never fill it from the current export environment or an ONNX
    donor.  ``clip_seg_lengths`` and ``clip_strike_phases`` remain separate established metadata
    keys, but are validated here because the C++ phase governor maps normalized phase to the strike
    frame derived from that exact pair.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    revision_present = PLANNER_TASK_REVISION_KEY in contract
    training_present = PLANNER_TASK_REVISION_TRAINING_KEY in contract
    if not revision_present and not training_present:
        return None
    if revision_present != training_present:
        raise ValueError(
            "schema-3 planner task revision is half-configured: planner_task_revision and "
            "planner_task_revision_training must either both be present or both be absent"
        )
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 3:
        raise ValueError("planner task revision metadata requires schema-3 training contract")

    revision = _require_exact_mapping_keys(
        contract[PLANNER_TASK_REVISION_KEY],
        _PLANNER_TASK_REVISION_KEYS,
        name="planner_task_revision",
    )
    if revision["enabled"] is not True:
        raise ValueError("planner_task_revision.enabled must be true when the block is present")
    if (
        type(revision["revision_schema_version"]) is not int
        or revision["revision_schema_version"] != 1
    ):
        raise ValueError("planner_task_revision.revision_schema_version must be integer 1")

    governor = _require_exact_mapping_keys(
        revision["governor"],
        _PLANNER_GOVERNOR_KEYS,
        name="planner_task_revision.governor",
    )
    profile = _require_exact_mapping_keys(
        governor["profile"],
        _PLANNER_GOVERNOR_PROFILE_KEYS,
        name="planner_task_revision.governor.profile",
    )
    if governor["contract_version"] != "phase_governor_v1" or profile[
        "contract_version"
    ] != "phase_governor_v1":
        raise ValueError("planner task revision requires phase_governor_v1")
    if (
        type(governor["schema_version"]) is not int
        or governor["schema_version"] != 1
        or type(profile["schema_version"]) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("planner task revision governor schema_version must be integer 1")

    profile_sha = governor["profile_sha256"]
    if (
        type(profile_sha) is not str
        or len(profile_sha) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha)
    ):
        raise ValueError("planner task revision profile_sha256 must be 64 lowercase hex characters")
    canonical_profile = _canonical_planner_json(profile, trailing_newline=True)
    calculated_profile_sha = hashlib.sha256(canonical_profile.encode("utf-8")).hexdigest()
    if profile_sha != calculated_profile_sha:
        raise ValueError("planner task revision profile_sha256 does not bind the canonical profile")

    policy_dt = _planner_finite_number(
        profile["policy_dt_s"], name="planner profile policy_dt_s", strictly_positive=True
    )
    min_tts = _planner_finite_number(
        profile["min_tts_s"], name="planner profile min_tts_s", strictly_positive=True
    )
    max_tts = _planner_finite_number(
        profile["max_tts_s"], name="planner profile max_tts_s", strictly_positive=True
    )
    if max_tts <= min_tts:
        raise ValueError("planner profile max_tts_s must be greater than min_tts_s")
    _planner_finite_number(
        profile["max_phase_rate_per_s"],
        name="planner profile max_phase_rate_per_s",
        strictly_positive=True,
    )
    _planner_finite_number(
        profile["max_phase_acceleration_per_s2"],
        name="planner profile max_phase_acceleration_per_s2",
        strictly_positive=True,
    )
    for key in (
        "max_deadline_revision_delta_s",
        "max_position_revision_delta_m",
        "max_velocity_revision_delta_mps",
        "early_deadline_tolerance_s",
    ):
        _planner_finite_number(
            profile[key], name=f"planner profile {key}", minimum=0.0
        )
    max_normal_delta = _planner_finite_number(
        profile["max_normal_revision_delta_rad"],
        name="planner profile max_normal_revision_delta_rad",
        minimum=0.0,
    )
    if max_normal_delta > math.pi:
        raise ValueError("planner profile max_normal_revision_delta_rad must be <= pi")
    normal_tolerance = _planner_finite_number(
        profile["normal_unit_tolerance"],
        name="planner profile normal_unit_tolerance",
        minimum=0.0,
    )
    if normal_tolerance >= 1.0:
        raise ValueError("planner profile normal_unit_tolerance must be < 1")

    raw_initial_tts = revision["initial_tts_range_s"]
    if not isinstance(raw_initial_tts, (list, tuple)) or len(raw_initial_tts) != 2:
        raise ValueError("planner_task_revision.initial_tts_range_s must contain two values")
    initial_lo = _planner_finite_number(
        raw_initial_tts[0], name="planner initial_tts_range_s[0]"
    )
    initial_hi = _planner_finite_number(
        raw_initial_tts[1], name="planner initial_tts_range_s[1]"
    )
    if initial_lo < min_tts or initial_hi > max_tts or initial_hi <= initial_lo:
        raise ValueError(
            "planner_task_revision.initial_tts_range_s must be strictly ordered inside the "
            "governor TTS envelope"
        )
    contract_policy_dt = _planner_finite_number(
        contract.get("policy_step_dt_s"),
        name="schema-3 policy_step_dt_s",
        strictly_positive=True,
    )
    if policy_dt != contract_policy_dt:
        raise ValueError(
            "planner profile policy_dt_s disagrees with schema-3 policy_step_dt_s"
        )

    segment_lengths = contract.get("motion_segment_lengths")
    strike_phases = contract.get("strike_phase_per_clip")
    if (
        not isinstance(segment_lengths, (list, tuple))
        or not segment_lengths
        or not isinstance(strike_phases, (list, tuple))
        or len(strike_phases) != len(segment_lengths)
    ):
        raise ValueError(
            "planner task revision requires one checkpoint-bound strike phase per motion segment"
        )
    for index, (raw_length, raw_phase) in enumerate(
        zip(segment_lengths, strike_phases, strict=True)
    ):
        if isinstance(raw_length, bool) or type(raw_length) is not int or raw_length <= 0:
            raise ValueError(f"planner clip {index} segment length must be a positive integer")
        phase = _planner_finite_number(
            raw_phase, name=f"planner clip {index} strike phase"
        )
        if phase < 0.0 or phase > 1.0:
            raise ValueError(f"planner clip {index} strike phase must be in [0, 1]")

    training = _require_exact_mapping_keys(
        contract[PLANNER_TASK_REVISION_TRAINING_KEY],
        _PLANNER_TASK_REVISION_TRAINING_KEYS,
        name="planner_task_revision_training",
    )
    if training["initial_tts_sampling_semantics"] != (
        "explicit_weighted_mixture_over_initial_tts_range_s"
    ):
        raise ValueError("planner revision initial TTS sampling semantics are unsupported")
    _validate_planner_initial_tts_mixture(
        training["initial_tts_mixture"],
        support_lo=initial_lo,
        support_hi=initial_hi,
    )
    if training["initial_feasibility_gate"] != (
        "normalized_phase_rate_and_acceleration_envelope_only"
    ):
        raise ValueError("planner revision feasibility-gate semantics are unsupported")
    if type(training["dynamics_certified_action_tau_min_bound"]) is not bool:
        raise ValueError("planner revision dynamics certification flag must be boolean")
    if training["timing_exam_semantics"] != {
        "0.5_s": "required_baseline_gate",
        "below_0.5_s": "stress_diagnostic_not_support_floor",
    }:
        raise ValueError("planner revision timing-exam semantics are unsupported")
    for key in ("position_std_m", "velocity_std_mps", "normal_std_rad", "tts_std_s"):
        _planner_finite_number(
            training[key], name=f"planner_task_revision_training.{key}", minimum=0.0
        )
    if training["truth_fields_immutable"] != [
        "question_bank_row",
        "physical_ball",
        "reward_target",
        "critic_target",
    ]:
        raise ValueError("planner revision immutable truth field list is invalid")
    if training["actor_revision_fields"] != [
        "target_position",
        "target_velocity",
        "signed_target_normal",
        "time_to_strike",
    ]:
        raise ValueError("planner revision actor field list is invalid")

    # nlohmann::json uses a sorted object map and compact dump for the C++ parser.  This exact
    # spelling keeps exporter outputs deterministic and makes the embedded profile SHA replayable.
    return _canonical_planner_json(revision, trailing_newline=False)


def bind_planner_task_revision_metadata(
    metadata: MutableMapping[str, str], contract: Mapping | None
) -> str | None:
    """Replace any donor/runtime claim with the checkpoint-side revision contract.

    This is intentionally destructive before validation: callers write artifacts only after all
    checks pass, so a failed contract cannot leave a stale donor claim available for later code to
    copy.  ``None`` or an OFF/legacy contract always removes the metadata key.
    """

    metadata.pop(PLANNER_TASK_REVISION_KEY, None)
    if contract is None:
        return None
    encoded = planner_task_revision_metadata(contract)
    if encoded is not None:
        metadata[PLANNER_TASK_REVISION_KEY] = encoded
    return encoded


def validate_training_launch_claim_sha256(value: str) -> str:
    """Validate the immutable launcher claim embedded in checkpoint ``infos``.

    Launch claims are operational provenance rather than part of the scientific hard contract.
    Their spelling stays deliberately strict so whitespace or case normalization cannot make two
    distinct atomic claims look identical after the process starts.
    """

    if type(value) is not str or len(value) != 64 or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise ValueError("training_launch_claim_sha256 must be 64 lowercase hex characters")
    return value


def actor_leg_ref_mask_provenance_required(contract: Mapping) -> bool:
    """Return whether the actor has the 62-D command term whose leg semantics may be masked."""

    names = contract.get("actor_obs_term_names")
    dims = contract.get("actor_obs_term_dims")
    if not isinstance(names, (list, tuple)) or not isinstance(dims, (list, tuple)):
        return False
    return any(
        str(name) == "command" and type(dim) is int and dim == 62
        for name, dim in zip(names, dims)
    )


def require_actor_leg_ref_mask_provenance(contract: Mapping) -> None:
    """Reject command-bearing contracts from before explicit mask/unmasked provenance existed."""

    if not actor_leg_ref_mask_provenance_required(contract):
        if contract.get("actor_leg_ref_mask") is True:
            raise ValueError("actor_leg_ref_mask is invalid without a 62-D command term")
        return
    epoch = contract.get(ACTOR_LEG_REF_MASK_PROVENANCE_KEY)
    if isinstance(epoch, bool) or type(epoch) is not int or epoch != ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH:
        raise ValueError(
            "command-bearing training contract lacks actor_leg_ref_mask_provenance_epoch=1; "
            "pre-epoch masked and unmasked checkpoints are indistinguishable"
        )


def actor_leg_ref_mask_provenance_payload(
    *, training_contract_sha256: str, source_checkpoint_sha256: str, masked: bool
) -> str:
    """Build the stable payload binding mask semantics to one contract/checkpoint pair."""

    contract_sha = str(training_contract_sha256).strip()
    checkpoint_sha = str(source_checkpoint_sha256).strip()
    for name, value in (
        ("training_contract_sha256", contract_sha),
        ("source_checkpoint_sha256", checkpoint_sha),
    ):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{name} must be 64 lowercase hex characters")
    if type(masked) is not bool:
        raise ValueError("actor_leg_ref_mask provenance masked value must be boolean")
    return (
        f"actor_leg_ref_mask_provenance_epoch={ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH}\n"
        f"actor_leg_ref_mask={1 if masked else 0}\n"
        f"training_contract_sha256={contract_sha}\n"
        f"source_checkpoint_sha256={checkpoint_sha}\n"
    )


def actor_leg_ref_mask_provenance_sha256(
    *, training_contract_sha256: str, source_checkpoint_sha256: str, masked: bool
) -> str:
    payload = actor_leg_ref_mask_provenance_payload(
        training_contract_sha256=training_contract_sha256,
        source_checkpoint_sha256=source_checkpoint_sha256,
        masked=masked,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def bind_actor_leg_ref_mask_metadata(
    metadata: MutableMapping[str, str], contract: Mapping | None
) -> None:
    """Replace donor mask provenance with the checkpoint contract's authoritative value.

    The key is intentionally only-when-true, but absence means unmasked only beside epoch 1.
    Clearing first is essential for standalone exports, where a donor may otherwise contaminate
    a checkpoint or launder ambiguous pre-epoch provenance.
    """

    metadata.pop("actor_leg_ref_mask", None)
    metadata.pop(ACTOR_LEG_REF_MASK_PROVENANCE_KEY, None)
    metadata.pop(ACTOR_LEG_REF_MASK_PROVENANCE_BINDING_KEY, None)
    if contract is None or not actor_leg_ref_mask_provenance_required(contract):
        return
    require_actor_leg_ref_mask_provenance(contract)
    masked = contract.get("actor_leg_ref_mask") is True
    metadata[ACTOR_LEG_REF_MASK_PROVENANCE_KEY] = str(ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH)
    metadata[ACTOR_LEG_REF_MASK_PROVENANCE_BINDING_KEY] = (
        actor_leg_ref_mask_provenance_sha256(
            training_contract_sha256=metadata.get("training_contract_sha256", ""),
            source_checkpoint_sha256=metadata.get("source_checkpoint_sha256", ""),
            masked=masked,
        )
    )
    if masked:
        metadata["actor_leg_ref_mask"] = "1"
        # Old consumers ignore unknown mask metadata. Keep the established exactness bit false so
        # they also reject this unsupported observation semantics instead of publishing it.
        metadata["training_contract_exact"] = "0"


def _canonical_actor_leg_ref_mask_callables():
    """Return the two exact command functions allowed to mint epoch-1 provenance."""

    from isaaclab.envs.mdp import generated_commands
    from whole_body_tracking.tasks.tracking.mdp.hope_observations import (
        generated_commands_actor_leg_masked,
    )

    return (
        (generated_commands, False),
        (generated_commands_actor_leg_masked, True),
    )


def _classify_actor_leg_ref_mask_callable(func) -> bool | None:
    for canonical, masked in _canonical_actor_leg_ref_mask_callables():
        if func is canonical:
            return masked
    return None


def resolve_motion_body_lin_vel_points(kinematics_contracts) -> tuple[str, ...]:
    """Resolve the runtime linear-velocity point independently for every motion clip.

    Declared COM/link-origin points are authoritative even for an inexact diagnostic contract.
    The one historical null spelling ``legacy_unbound_assumed_com`` is known to have been loaded
    through Isaac's COM-velocity channel, so it resolves to COM while its separate ``exact=False``
    provenance remains unchanged.  No other null or unknown spelling is safe to guess.
    """

    if not isinstance(kinematics_contracts, (list, tuple)) or not kinematics_contracts:
        raise ValueError("motion_kinematics_contracts must be a non-empty array")
    resolved = []
    for index, item in enumerate(kinematics_contracts):
        if not isinstance(item, Mapping):
            raise ValueError(f"motion kinematics clip {index} must be an object")
        point = item.get("body_lin_vel_point")
        if point in MOTION_BODY_LIN_VEL_POINTS:
            resolved.append(str(point))
            continue
        status = item.get("status")
        if point is None and status == "legacy_unbound_assumed_com":
            resolved.append("center_of_mass")
            continue
        if point is None:
            raise ValueError(
                f"motion kinematics clip {index} has unresolved null body_lin_vel_point "
                f"for status {status!r}"
            )
        raise ValueError(
            f"motion kinematics clip {index} has unknown body_lin_vel_point {point!r}"
        )
    return tuple(resolved)


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _flat_floats(value, *, name: str, expected: int | None = None) -> list[float]:
    raw = _tolist(value)
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise RuntimeError(f"{name} must be a vector or a single nominal row, got {len(raw)} rows")
        raw = raw[0]
    out = [float(item) for item in raw]
    if expected is not None and len(out) != expected:
        raise RuntimeError(f"{name} has {len(out)} values, expected {expected}")
    if any(not math.isfinite(item) for item in out):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return out


def _nominal_row(value, *, name: str, expected: int) -> list[float]:
    raw = _tolist(value)
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return _flat_floats(raw, name=name, expected=expected)


def _joint_ids(value, count: int) -> list[int]:
    if isinstance(value, slice):
        if value != slice(None):
            return list(range(count))[value]
        return list(range(count))
    return [int(item) for item in _tolist(value)]


def _joint_actuator_types(robot, count: int) -> list[str]:
    """Resolve the instantiated actuator integration model for every articulation joint."""
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping) or not actuators:
        raise RuntimeError("robot.actuators is unavailable; actuator integration cannot be proven")
    result: list[str | None] = [None] * count
    for group_name, actuator in actuators.items():
        if not hasattr(actuator, "joint_indices") or not hasattr(actuator, "is_implicit_model"):
            raise RuntimeError(
                f"actuator group {group_name!r} lacks joint_indices/is_implicit_model"
            )
        kind = "implicit" if bool(actuator.is_implicit_model) else "explicit"
        for joint_id in _joint_ids(actuator.joint_indices, count):
            if not (0 <= joint_id < count):
                raise RuntimeError(
                    f"actuator group {group_name!r} has out-of-range joint id {joint_id}"
                )
            if result[joint_id] is not None:
                raise RuntimeError(
                    f"joint {joint_id} belongs to multiple actuator groups"
                )
            result[joint_id] = kind
    missing = [index for index, value in enumerate(result) if value is None]
    if missing:
        raise RuntimeError(f"actuator integration is unresolved for joints {missing}")
    return [str(value) for value in result]


def _policy_layout(env) -> tuple[list[str], list[int], int]:
    manager = env.observation_manager
    names = [str(name) for name in manager.active_terms["policy"]]
    raw_dims = manager.group_obs_term_dim["policy"]
    dims = []
    for dim in raw_dims:
        if isinstance(dim, (tuple, list)):
            if len(dim) != 1:
                raise RuntimeError(f"policy observation term has non-flat dimension {dim!r}")
            dim = dim[0]
        dims.append(int(dim))
    total = manager.group_obs_dim["policy"]
    if isinstance(total, (tuple, list)):
        if len(total) != 1:
            raise RuntimeError(f"policy observation group has non-flat dimension {total!r}")
        total = total[0]
    total = int(total)
    if len(names) != len(dims) or sum(dims) != total:
        raise RuntimeError(
            f"invalid policy layout: names={len(names)} dims={dims} total={total}"
        )
    return names, dims, total


def _observation_history_lengths(env, names: list[str]) -> list[int]:
    group_cfg = env.observation_manager.cfg.policy
    if group_cfg.history_length is not None:
        return [int(group_cfg.history_length)] * len(names)
    cfg_by_name = group_cfg.to_dict()
    out = []
    for name in names:
        raw = cfg_by_name[name]["history_length"]
        value = 0 if raw is None else int(raw)
        out.append(1 if value == 0 else value)
    return out


def runtime_execution_facts(
    env, actor_contract, *, allow_legacy_actor_leg_ref_mask_ambiguity: bool = False
) -> dict:
    """Extract schema-3 execution facts from the instantiated, startup-initialized environment."""
    robot = env.scene["robot"]
    data = robot.data
    articulation_names = list(
        getattr(data, "joint_names", getattr(robot, "joint_names", ()))
    )
    if not articulation_names or len(set(articulation_names)) != len(articulation_names):
        raise RuntimeError("robot articulation joint names are empty or non-unique")
    n = len(articulation_names)

    action = env.action_manager.get_term("joint_pos")
    ids = _joint_ids(getattr(action, "_joint_ids", slice(None)), n)
    identity = list(range(n))
    if ids != identity:
        raise RuntimeError(
            "schema-3 ONNX requires identity action/articulation order because actions and baked "
            f"reference joints share one joint_names contract; got action_joint_ids={ids}"
        )
    joint_names = [articulation_names[index] for index in ids]
    action_cfg = getattr(action, "cfg", getattr(env.cfg.actions, "joint_pos", None))
    use_default_offset = bool(getattr(action_cfg, "use_default_offset", False))
    if not use_default_offset:
        raise RuntimeError(
            "schema-3 deploy decoder requires JointPositionAction use_default_offset=True"
        )

    if not hasattr(data, "default_joint_pos_nominal"):
        raise RuntimeError(
            "robot.data.default_joint_pos_nominal is missing; the startup nominal-pose capture "
            "must run before writing a schema-3 contract"
        )
    default_q = _flat_floats(
        data.default_joint_pos_nominal, name="default_joint_pos_nominal", expected=n
    )
    action_scale = _nominal_row(action._scale, name="action_scale", expected=n)
    kp = _nominal_row(data.default_joint_stiffness, name="joint_stiffness", expected=n)
    kd = _nominal_row(data.default_joint_damping, name="joint_damping", expected=n)
    effort = _nominal_row(data.joint_effort_limits, name="joint_effort_limits", expected=n)
    actuator_types = _joint_actuator_types(robot, n)
    armature = _nominal_row(
        data.default_joint_armature, name="joint_armature", expected=n
    )
    friction = _nominal_row(
        data.default_joint_friction_coeff,
        name="joint_friction_coefficients",
        expected=n,
    )
    velocity_limits = _nominal_row(
        data.joint_vel_limits, name="joint_velocity_limits", expected=n
    )
    if any(value <= 0.0 for value in action_scale):
        raise RuntimeError("action_scale must be finite and positive")
    if any(value <= 0.0 for value in kp) or any(value <= 0.0 for value in kd):
        raise RuntimeError("nominal joint stiffness/damping must be finite and positive")
    if any(value <= 0.0 for value in effort):
        raise RuntimeError("joint_effort_limits must be finite and positive")
    if any(value < 0.0 for value in armature):
        raise RuntimeError("joint_armature must be finite and non-negative")
    if any(value < 0.0 for value in friction):
        raise RuntimeError("joint_friction_coefficients must be finite and non-negative")
    if any(value <= 0.0 for value in velocity_limits):
        raise RuntimeError("joint_velocity_limits must be finite and positive")

    limits_raw = _tolist(data.soft_joint_pos_limits)
    if (
        limits_raw
        and isinstance(limits_raw[0], (list, tuple))
        and limits_raw[0]
        and isinstance(limits_raw[0][0], (list, tuple))
    ):
        # Isaac stores [num_envs, num_joints, 2]; select the nominal first environment.
        limits_raw = limits_raw[0]
    if len(limits_raw) != n:
        raise RuntimeError(f"soft_joint_pos_limits has {len(limits_raw)} joints, expected {n}")
    limits = []
    for index, pair in enumerate(limits_raw):
        if len(pair) != 2:
            raise RuntimeError(f"soft_joint_pos_limits[{index}] is not [lo, hi]")
        lo, hi = float(pair[0]), float(pair[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise RuntimeError(f"invalid soft q-des limits for {joint_names[index]}: {(lo, hi)}")
        limits.append([lo, hi])

    physics_dt = float(env.physics_dt)
    policy_dt = float(env.step_dt)
    decimation = int(env.cfg.decimation)
    if not (math.isfinite(physics_dt) and physics_dt > 0.0):
        raise RuntimeError(f"invalid physics dt {physics_dt!r}")
    if not (math.isfinite(policy_dt) and policy_dt > 0.0) or decimation <= 0:
        raise RuntimeError(f"invalid policy dt/decimation {policy_dt!r}/{decimation!r}")
    if not math.isclose(policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"policy dt {policy_dt:.17g} != physics dt {physics_dt:.17g} * {decimation}"
        )

    obs_names, obs_dims, obs_total = _policy_layout(env)
    history = _observation_history_lengths(env, obs_names)
    if actor_contract is not None:
        expected_layout = [(term.name, int(term.dim)) for term in actor_contract.terms]
        if list(zip(obs_names, obs_dims)) != expected_layout or obs_total != int(
            actor_contract.total_dim
        ):
            raise RuntimeError("actor contract object does not match the instantiated policy layout")

    motion = env.command_manager.get_term("motion")
    body_ids = [int(item) for item in _tolist(motion.body_indexes)]
    robot_body_names = list(getattr(robot, "body_names", ()))
    if (
        not robot_body_names
        or any(not str(name) for name in robot_body_names)
        or len(set(robot_body_names)) != len(robot_body_names)
    ):
        raise RuntimeError("robot body_names are unavailable or non-unique")
    if any(index < 0 or index >= len(robot_body_names) for index in body_ids):
        raise RuntimeError(f"resolved motion body indices are out of range: {body_ids}")
    body_names = [robot_body_names[index] for index in body_ids]
    configured_bodies = [str(name) for name in motion.cfg.body_names]
    if body_names != configured_bodies:
        raise RuntimeError(
            f"resolved motion body order {body_names} != configured order {configured_bodies}"
        )
    anchor_name = str(motion.cfg.anchor_body_name)
    if anchor_name not in body_names:
        raise RuntimeError(f"motion anchor {anchor_name!r} is absent from resolved body order")
    anchor_index = body_names.index(anchor_name)
    segment_lengths = [int(value) for value in _tolist(motion.motion.seg_len)]
    if not segment_lengths or any(value <= 0 for value in segment_lengths):
        raise RuntimeError(f"invalid motion segment lengths {segment_lengths}")
    clip_fps = [float(value) for value in motion.motion.per_clip_fps]
    if len(clip_fps) != len(segment_lengths):
        raise RuntimeError(
            "motion fps count does not match segments: "
            f"{len(clip_fps)} vs {len(segment_lengths)}"
        )
    policy_hz = 1.0 / policy_dt
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or not math.isclose(value, policy_hz, rel_tol=0.0, abs_tol=1e-9)
        for value in clip_fps
    ):
        raise RuntimeError(
            f"motion clip fps {clip_fps} must all equal policy rate {policy_hz:.12g} Hz"
        )
    kinematics_contracts = [dict(item) for item in motion.motion.kinematics_contracts]
    if len(kinematics_contracts) != len(segment_lengths):
        raise RuntimeError(
            "motion kinematics-contract count does not match segments: "
            f"{len(kinematics_contracts)} vs {len(segment_lengths)}"
        )
    try:
        resolve_motion_body_lin_vel_points(kinematics_contracts)
    except ValueError as exc:
        raise RuntimeError(f"unresolved motion velocity-point contract: {exc}") from exc
    kinematics_exact = bool(motion.motion.kinematics_contract_exact)
    if kinematics_exact != all(bool(item.get("exact", False)) for item in kinematics_contracts):
        raise RuntimeError("motion kinematics exact flag disagrees with per-clip contracts")

    qdes_clamp = bool(
        getattr(action, "_clamp_enabled", getattr(env.cfg.actions.joint_pos, "clamp", False))
    )
    # R-a masking leaves the 62-D layout unchanged, so both the true-only mask bit and an always
    # present provenance epoch are needed: only epoch=1 + absent mask proves unmasked. Detection
    # unwraps only structurally empty partials and accepts only the two canonical callables.  Bound
    # args/kwargs can change command selection or semantics and may not be discarded as provenance.
    _cmd_term = getattr(
        getattr(getattr(env.cfg, "observations", None), "policy", None), "command", None
    )
    _cmd_func = getattr(_cmd_term, "func", None)
    # Only the exact built-in partial type is transparent.  ``functools.partial`` is subclassable,
    # and a subclass may override ``__call__`` while keeping canonical ``.func`` plus empty
    # ``args``/``keywords``.  Treating such an object as a plain partial would mint provenance for
    # behavior that is not the canonical callable.  Apply the exact-type rule at every unwrap layer.
    while type(_cmd_func) is functools.partial:
        if _cmd_func.args or _cmd_func.keywords:
            _cmd_func = None
            break
        _cmd_func = _cmd_func.func
    facts = {
        "articulation_joint_names": articulation_names,
        "action_joint_ids": ids,
        "joint_names": joint_names,
        "default_joint_pos": default_q,
        "action_scale": action_scale,
        "joint_stiffness": kp,
        "joint_damping": kd,
        "joint_effort_limits": effort,
        "joint_actuator_types": actuator_types,
        "joint_armature": armature,
        "joint_friction_coefficients": friction,
        "joint_velocity_limits": velocity_limits,
        "joint_friction_backend": JOINT_FRICTION_BACKEND,
        "joint_friction_semantics": JOINT_FRICTION_SEMANTICS,
        "joint_friction_units": JOINT_FRICTION_UNITS,
        "qdes_joint_pos_limits": limits,
        "action_use_default_offset": use_default_offset,
        "qdes_clamp": qdes_clamp,
        "physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "control_decimation": decimation,
        "actor_obs_contract": getattr(actor_contract, "name", None),
        "actor_obs_mode": getattr(actor_contract, "obs_mode", None),
        "actor_obs_total_dim": obs_total,
        "actor_obs_term_names": obs_names,
        "actor_obs_term_dims": obs_dims,
        "observation_history_lengths": history,
        "articulation_body_names": robot_body_names,
        "body_names": body_names,
        "body_indices": body_ids,
        "anchor_body_name": anchor_name,
        "anchor_body_index": anchor_index,
        "motion_segment_lengths": segment_lengths,
        "motion_clip_fps": clip_fps,
        "motion_kinematics_contracts": kinematics_contracts,
        "motion_kinematics_exact": kinematics_exact,
    }
    if any(name == "command" and dim == 62 for name, dim in zip(obs_names, obs_dims)):
        actor_leg_ref_mask = _classify_actor_leg_ref_mask_callable(_cmd_func)
        if actor_leg_ref_mask is None and not allow_legacy_actor_leg_ref_mask_ambiguity:
            raise RuntimeError(
                "62-D actor command func is not one of the two canonical epoch-1 callables; "
                "wrappers, partial subclasses, partials with bound args/kwargs, and copied "
                "provenance attributes are not authoritative"
            )
        if actor_leg_ref_mask is not None:
            facts[ACTOR_LEG_REF_MASK_PROVENANCE_KEY] = ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH
        if actor_leg_ref_mask is True:
            facts["actor_leg_ref_mask"] = True
    return facts


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
_A3_LOWER_BODY_LEGS = tuple(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER[-12:])


def _wave_finite(value: object, *, name: str, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"schema-3 {name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"schema-3 {name} must be a finite number")
    if positive and value <= 0.0:
        raise ValueError(f"schema-3 {name} must be finite and > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"schema-3 {name} must be finite and >= 0")
    return value


def _validate_lower_body_wave_contracts(
    contract: Mapping,
    joint_names: list[str],
    articulation_joint_names: list[str],
) -> None:
    pose = contract.get("lower_body_pose_imitation_reward")
    bundle = contract.get("lower_body_stability_bundle_reward")
    if (pose is None) != (bundle is None):
        raise ValueError("schema-3 explicit Wave-B cells require both B1 and B2 blocks")
    if pose is not None:
        pose = _require_exact_mapping_keys(
            pose,
            frozenset(
                {
                    "schema_version", "enabled", "probe_enabled", "activation_ledger",
                    "weight", "std_rad", "support_pre_s", "support_post_s",
                    "racket_command_name", "motion_command_name", "joint_count",
                    "joint_names", "joint_order", "reference_joint_order", "formula",
                    "gate", "success_conditioned",
                }
            ),
            name="schema-3 lower_body_pose_imitation_reward",
        )
        if type(pose["schema_version"]) is not int or pose["schema_version"] != 1:
            raise ValueError("schema-3 lower_body_pose_imitation_reward schema_version must be 1")
        if (
            not isinstance(pose["enabled"], bool)
            or pose["probe_enabled"] is not True
            or pose["success_conditioned"] is not False
        ):
            raise ValueError("schema-3 lower_body_pose_imitation_reward flags are invalid")
        weight = _wave_finite(
            pose["weight"], name="lower_body_pose_imitation_reward.weight", nonnegative=True
        )
        _wave_finite(pose["std_rad"], name="lower_body_pose_imitation_reward.std_rad", positive=True)
        _wave_finite(
            pose["support_pre_s"],
            name="lower_body_pose_imitation_reward.support_pre_s",
            nonnegative=True,
        )
        _wave_finite(
            pose["support_post_s"],
            name="lower_body_pose_imitation_reward.support_post_s",
            nonnegative=True,
        )
        if pose["enabled"] != (weight > 0.0):
            raise ValueError("schema-3 lower_body_pose_imitation_reward enabled disagrees with weight")
        expected = {
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "activation_ledger": "weight_independent_control_step_counters",
            "joint_count": 12,
            "joint_names": list(_A3_LOWER_BODY_LEGS),
            "joint_order": "canonical_deploy_order_selected_by_name",
            "reference_joint_order": "motion_command_runtime_articulation_identity",
            "formula": "exp(-mean(square(q_leg-qref_leg))/square(std_rad))",
            "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
        }
        for key, value in expected.items():
            if pose[key] != value:
                raise ValueError(
                    f"schema-3 lower_body_pose_imitation_reward {key} must be {value!r}"
                )

    if bundle is not None:
        bundle = _require_exact_mapping_keys(
            bundle,
            frozenset(
                {
                    "schema_version", "enabled", "probe_enabled", "activation_ledger",
                    "weight", "min_stance_width_m", "stance_scale_m",
                    "leg_velocity_margin_radps", "leg_velocity_scale_radps",
                    "support_pre_s", "support_post_s", "racket_command_name",
                    "motion_command_name", "leg_joint_count", "leg_joint_names",
                    "foot_body_names", "joint_order", "stance_width_frame", "components",
                    "formula", "gate", "success_conditioned", "uses_motion_reference",
                    "duplicates_slip_or_upright",
                }
            ),
            name="schema-3 lower_body_stability_bundle_reward",
        )
        if type(bundle["schema_version"]) is not int or bundle["schema_version"] != 1:
            raise ValueError("schema-3 lower_body_stability_bundle_reward schema_version must be 1")
        if (
            not isinstance(bundle["enabled"], bool)
            or bundle["probe_enabled"] is not True
            or bundle["success_conditioned"] is not False
            or bundle["uses_motion_reference"] is not False
            or bundle["duplicates_slip_or_upright"] is not False
        ):
            raise ValueError("schema-3 lower_body_stability_bundle_reward flags are invalid")
        weight = _wave_finite(bundle["weight"], name="lower_body_stability_bundle_reward.weight")
        if weight > 0.0 or bundle["enabled"] != (weight < 0.0):
            raise ValueError("schema-3 lower_body_stability_bundle_reward weight/enabled is invalid")
        for name in (
            "min_stance_width_m", "stance_scale_m", "leg_velocity_scale_radps",
        ):
            _wave_finite(bundle[name], name=f"lower_body_stability_bundle_reward.{name}", positive=True)
        for name in ("leg_velocity_margin_radps", "support_pre_s", "support_post_s"):
            _wave_finite(
                bundle[name], name=f"lower_body_stability_bundle_reward.{name}", nonnegative=True
            )
        expected = {
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "activation_ledger": "weight_independent_control_step_counters",
            "leg_joint_count": 12,
            "leg_joint_names": list(_A3_LOWER_BODY_LEGS),
            "foot_body_names": ["left_ankle_roll_Link", "right_ankle_roll_Link"],
            "joint_order": "canonical_deploy_order_selected_by_name",
            "stance_width_frame": "base_yaw_lateral_signed_left_minus_right",
            "components": [
                "stance_width_lower_hinge",
                "twelve_leg_realized_qdot_tail",
            ],
            "formula": "mean(bounded_stance_tail,bounded_leg_qdot_tail)",
            "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
        }
        for key, value in expected.items():
            if bundle[key] != value:
                raise ValueError(
                    f"schema-3 lower_body_stability_bundle_reward {key} must be {value!r}"
                )
        articulation_bodies = contract.get("articulation_body_names")
        if (
            not isinstance(articulation_bodies, (list, tuple))
            or any(name not in articulation_bodies for name in bundle["foot_body_names"])
        ):
            raise ValueError(
                "schema-3 lower_body_stability_bundle_reward foot bodies are absent from articulation"
            )

    # The live articulation enumerates the same 31 joints breadth-first; require
    # the exact A3 name set and runtime==articulation identity, not one fixed order.
    if (pose is not None or bundle is not None) and (
        not isinstance(joint_names, (list, tuple))
        or len(joint_names) != len(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER)
        or len(set(joint_names)) != len(joint_names)
        or set(joint_names) != set(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER)
        or list(articulation_joint_names or []) != list(joint_names)
    ):
        raise ValueError(
            "schema-3 lower-body rewards require exact A3 runtime/articulation joint identity"
        )
    if pose is not None and bundle is not None and pose["enabled"] and bundle["enabled"]:
        raise ValueError("schema-3 Wave-B B1 and B2 rewards are mutually exclusive")


# --------------------------------------------------------------------------------------------- #
# Wave-P random base push (PACE/BeyondMimic-style shove; default OFF = HITTER no-push recipe).
# --------------------------------------------------------------------------------------------- #
PUSH_ROBOT_EVENT_KEY = "push_robot_event"
PUSH_ROBOT_EVENT_FUNC = "push_by_setting_velocity"
PUSH_ROBOT_EVENT_MODE = "interval"
PUSH_ROBOT_ANG_AXES = ("none", "yaw", "rpy")
_PUSH_ROBOT_EVENT_KEYS = frozenset(
    {
        "schema_version", "enabled", "func", "mode", "interval_range_s",
        "vel_xy_mps", "ang_vel_radps", "ang_axes", "velocity_range",
    }
)


def push_robot_event_block(
    *, enable, interval_range_s, vel_xy_mps, ang_vel_radps, ang_axes
):
    """Translate the push flag group into the canonical ``push_robot_event`` contract block.

    人话:把"要不要推、隔几秒推一次、水平推多快、角速度踢多快、踢哪些转轴"翻译成 push_robot
    事件的对称速度区间表 + 合同块。``enable=False`` 返回 ``None``(= 不推,合同不写这个块,
    所有历史/在跑配置逐位不变),但此时任何非零幅度都是配置错误(fail-closed:关着的开关
    不许挂着上膛的参数)。This is the single validation/assembly source shared by the env cfg
    flag path, the train.py ``task.push`` override, and the schema-3 contract validator.
    """

    if not isinstance(enable, bool):
        raise ValueError("push_robot_event enable must be an explicit boolean")
    if not enable:
        for name, value in (
            ("vel_xy_mps", vel_xy_mps),
            ("ang_vel_radps", ang_vel_radps),
        ):
            if value is not None and (isinstance(value, bool) or float(value) != 0.0):
                raise ValueError(
                    f"push_robot_event disabled but {name}={value!r} is nonzero — "
                    "delete the dormant amplitude or set enable=true"
                )
        if ang_axes not in (None, "none"):
            raise ValueError(
                f"push_robot_event disabled but ang_axes={ang_axes!r} selects push axes — "
                "delete it or set enable=true"
            )
        return None
    try:
        interval_items = list(interval_range_s)
    except TypeError as exc:
        raise ValueError(
            "push_robot_event interval_range_s must be a [lo, hi] pair of seconds"
        ) from exc
    if len(interval_items) != 2:
        raise ValueError(
            "push_robot_event interval_range_s must be a [lo, hi] pair of seconds"
        )
    interval_lo = _wave_finite(
        interval_items[0], name="push_robot_event.interval_range_s[0]", positive=True
    )
    interval_hi = _wave_finite(
        interval_items[1], name="push_robot_event.interval_range_s[1]", positive=True
    )
    if interval_lo > interval_hi:
        raise ValueError(
            "push_robot_event interval_range_s must satisfy 0 < lo <= hi"
        )
    vel = _wave_finite(vel_xy_mps, name="push_robot_event.vel_xy_mps", nonnegative=True)
    ang = _wave_finite(
        ang_vel_radps, name="push_robot_event.ang_vel_radps", nonnegative=True
    )
    if ang_axes not in PUSH_ROBOT_ANG_AXES:
        raise ValueError(
            f"push_robot_event ang_axes must be one of {PUSH_ROBOT_ANG_AXES}, got {ang_axes!r}"
        )
    if ang_axes == "none" and ang != 0.0:
        raise ValueError(
            "push_robot_event ang_axes='none' requires ang_vel_radps=0 "
            "(pick ang_axes='yaw'|'rpy' to push angular velocity)"
        )
    if ang_axes != "none" and ang == 0.0:
        raise ValueError(
            f"push_robot_event ang_axes={ang_axes!r} requires ang_vel_radps > 0 "
            "(a zero angular push must spell ang_axes='none')"
        )
    if vel == 0.0 and ang == 0.0:
        raise ValueError(
            "push_robot_event enabled but every amplitude is zero — "
            "an enabled push must push something (or set enable=false)"
        )
    velocity_range = {"x": [-vel, vel], "y": [-vel, vel]}
    if ang_axes == "yaw":
        velocity_range["yaw"] = [-ang, ang]
    elif ang_axes == "rpy":
        velocity_range["roll"] = [-ang, ang]
        velocity_range["pitch"] = [-ang, ang]
        velocity_range["yaw"] = [-ang, ang]
    return {
        "schema_version": 1,
        "enabled": True,
        "func": PUSH_ROBOT_EVENT_FUNC,
        "mode": PUSH_ROBOT_EVENT_MODE,
        "interval_range_s": [interval_lo, interval_hi],
        "vel_xy_mps": vel,
        "ang_vel_radps": ang,
        "ang_axes": ang_axes,
        "velocity_range": velocity_range,
    }


def _validate_push_robot_event_contract(contract: Mapping) -> None:
    """Wave-P random base-push block (task.push; PACE/BeyondMimic push, HITTER default = none).

    Absent block = push disabled (every historical/no-push run, byte-identical contract).  A
    present block is always an ENABLED push and must be internally consistent: its stored
    velocity box must equal the canonical re-assembly from its own amplitudes/axes, so a
    hand-edited or drifted sidecar cannot smuggle a different push recipe past a resume.
    """

    block = contract.get(PUSH_ROBOT_EVENT_KEY)
    if block is None:
        if PUSH_ROBOT_EVENT_KEY in contract:
            raise ValueError(
                "schema-3 push_robot_event must be omitted when disabled, not null"
            )
        return
    block = _require_exact_mapping_keys(
        block, _PUSH_ROBOT_EVENT_KEYS, name="schema-3 push_robot_event"
    )
    if type(block["schema_version"]) is not int or block["schema_version"] != 1:
        raise ValueError("schema-3 push_robot_event schema_version must be integer 1")
    if block["enabled"] is not True:
        raise ValueError(
            "schema-3 push_robot_event enabled must be true "
            "(a disabled push is spelled by omitting the block)"
        )
    if block["func"] != PUSH_ROBOT_EVENT_FUNC:
        raise ValueError(
            f"schema-3 push_robot_event func must be {PUSH_ROBOT_EVENT_FUNC!r}"
        )
    if block["mode"] != PUSH_ROBOT_EVENT_MODE:
        raise ValueError(
            f"schema-3 push_robot_event mode must be {PUSH_ROBOT_EVENT_MODE!r}"
        )
    try:
        expected = push_robot_event_block(
            enable=True,
            interval_range_s=block["interval_range_s"],
            vel_xy_mps=block["vel_xy_mps"],
            ang_vel_radps=block["ang_vel_radps"],
            ang_axes=block["ang_axes"],
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 push_robot_event is invalid: {exc}") from exc
    stored_range = block["velocity_range"]
    if not isinstance(stored_range, Mapping):
        raise ValueError("schema-3 push_robot_event velocity_range must be an object")
    normalized_range = {}
    for axis, rng in stored_range.items():
        if (
            not isinstance(rng, (list, tuple))
            or len(rng) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in rng)
        ):
            raise ValueError(
                f"schema-3 push_robot_event velocity_range.{axis} must be a [lo, hi] pair"
            )
        normalized_range[str(axis)] = [float(rng[0]), float(rng[1])]
    stored_interval = [float(v) for v in block["interval_range_s"]]
    if (
        normalized_range != expected["velocity_range"]
        or stored_interval != expected["interval_range_s"]
    ):
        raise ValueError(
            "schema-3 push_robot_event is internally inconsistent: the stored "
            "velocity_range/interval does not equal the canonical assembly from "
            "vel_xy_mps/ang_vel_radps/ang_axes"
        )


# --------------------------------------------------------------------------------------------- #
# F-axis interval FORCE push (matched-impulse companion of the Wave-P velocity push; default OFF).
# --------------------------------------------------------------------------------------------- #
FORCE_PUSH_EVENT_KEY = "force_push_event"
FORCE_PUSH_EVENT_FUNC = "push_by_applying_wrench"
FORCE_PUSH_SWEEP_FUNC = "sweep_expired_force_pushes"
FORCE_PUSH_EVENT_MODE = "interval"
FORCE_PUSH_BODY_NAME = "pelvis_link"
# 施力点语义显式化(Yikang V9 教训:把 link 原点标成 COM):PhysX 在 positions=None 时把外力
# 施加在 link 原点,合同必须诚实写 "pelvis_link_origin",不许标 COM。
FORCE_PUSH_APPLICATION_POINT = "pelvis_link_origin"
_FORCE_PUSH_ASSEMBLY_KEYS = frozenset(
    {
        "schema_version", "enabled", "func", "mode", "interval_range_s",
        "force_n", "duration_s", "duration_steps", "control_dt_s",
        "body_name", "application_point",
    }
)
_FORCE_PUSH_EVENT_KEYS = _FORCE_PUSH_ASSEMBLY_KEYS | {
    "robot_mass_kg", "delta_v_equiv_mps",
}


def force_push_event_block(
    *, enable, interval_range_s, force_n, duration_s, control_dt_s
):
    """Translate the force-push flag group into the canonical assembly block (no runtime mass yet).

    人话:把"要不要力推、隔几秒推一次、推多少牛、持续几秒"翻译成 force_push 事件的合同装配块。
    ``enable=False`` 返回 ``None``(= 不推,合同不写这个块,历史/在跑配置逐字节不变),但此时
    非零 force_n 是配置错误(fail-closed:关着的开关不许挂上膛的力)。``duration_s`` 必须是
    控制步长的整数倍(恒力持续整数个控制步,0.30 s = 15 步 @ 50 Hz),否则 fail-loud。This is
    the single validation/assembly source shared by the env cfg flag path
    (hope_env_cfg.apply_force_push_event), the train.py ``task.force_push`` override, and the
    schema-3 contract validator. 运行时质量与 Δv_equiv 由 :func:`bind_force_push_runtime_mass`
    在拿到真实 articulation 质量后追加。
    """

    if not isinstance(enable, bool):
        raise ValueError("force_push_event enable must be an explicit boolean")
    if not enable:
        if force_n is not None and (isinstance(force_n, bool) or float(force_n) != 0.0):
            raise ValueError(
                f"force_push_event disabled but force_n={force_n!r} is nonzero — "
                "delete the dormant force or set enable=true"
            )
        return None
    try:
        interval_items = list(interval_range_s)
    except TypeError as exc:
        raise ValueError(
            "force_push_event interval_range_s must be a [lo, hi] pair of seconds"
        ) from exc
    if len(interval_items) != 2:
        raise ValueError(
            "force_push_event interval_range_s must be a [lo, hi] pair of seconds"
        )
    interval_lo = _wave_finite(
        interval_items[0], name="force_push_event.interval_range_s[0]", positive=True
    )
    interval_hi = _wave_finite(
        interval_items[1], name="force_push_event.interval_range_s[1]", positive=True
    )
    if interval_lo > interval_hi:
        raise ValueError(
            "force_push_event interval_range_s must satisfy 0 < lo <= hi"
        )
    force = _wave_finite(force_n, name="force_push_event.force_n", positive=True)
    duration = _wave_finite(
        duration_s, name="force_push_event.duration_s", positive=True
    )
    control_dt = _wave_finite(
        control_dt_s, name="force_push_event.control_dt_s", positive=True
    )
    duration_steps = int(round(duration / control_dt))
    if duration_steps < 1 or abs(duration_steps * control_dt - duration) > 1e-9:
        raise ValueError(
            "force_push_event duration_s must be a positive whole number of control "
            f"steps (duration_s={duration!r} is not an integer multiple of "
            f"control_dt_s={control_dt!r})"
        )
    return {
        "schema_version": 1,
        "enabled": True,
        "func": FORCE_PUSH_EVENT_FUNC,
        "mode": FORCE_PUSH_EVENT_MODE,
        "interval_range_s": [interval_lo, interval_hi],
        "force_n": force,
        "duration_s": duration,
        "duration_steps": duration_steps,
        "control_dt_s": control_dt,
        "body_name": FORCE_PUSH_BODY_NAME,
        "application_point": FORCE_PUSH_APPLICATION_POINT,
    }


def bind_force_push_runtime_mass(block, *, robot_mass_kg):
    """Append the runtime articulation mass + matched-impulse Δv to an assembly block.

    人话:合同必须记运行时真实读到的机器人总质量与换算出的 Δv_equiv = force_n × duration_s /
    m_robot,供与速度推档位(p02/p035/p05/p08)对表。装配块形状不对、质量非正,一律 raise。
    """

    if not isinstance(block, Mapping) or set(block) != _FORCE_PUSH_ASSEMBLY_KEYS:
        raise ValueError(
            "force_push_event runtime binding requires the exact canonical assembly block"
        )
    if block.get("enabled") is not True:
        raise ValueError("force_push_event runtime binding requires an enabled block")
    mass = _wave_finite(
        robot_mass_kg, name="force_push_event.robot_mass_kg", positive=True
    )
    delta_v = float(block["force_n"]) * float(block["duration_s"]) / mass
    return {**block, "robot_mass_kg": mass, "delta_v_equiv_mps": delta_v}


def _validate_force_push_event_contract(contract: Mapping) -> None:
    """F-axis interval force-push block (task.force_push; matched-impulse vs the velocity push).

    Absent block = force push disabled (every historical/no-push run, byte-identical contract).
    A present block is always an ENABLED push and must be internally consistent: interval /
    duration_steps must equal the canonical re-assembly from its own fields, and the recorded
    Δv_equiv must equal force_n x duration_s / robot_mass_kg recomputed — a hand-edited sidecar
    cannot smuggle a different impulse past a resume. ``application_point`` 必须是
    ``pelvis_link_origin``(Yikang V9 反例:link 原点被标成 COM)。
    """

    block = contract.get(FORCE_PUSH_EVENT_KEY)
    if block is None:
        if FORCE_PUSH_EVENT_KEY in contract:
            raise ValueError(
                "schema-3 force_push_event must be omitted when disabled, not null"
            )
        return
    block = _require_exact_mapping_keys(
        block, _FORCE_PUSH_EVENT_KEYS, name="schema-3 force_push_event"
    )
    if type(block["schema_version"]) is not int or block["schema_version"] != 1:
        raise ValueError("schema-3 force_push_event schema_version must be integer 1")
    if block["enabled"] is not True:
        raise ValueError(
            "schema-3 force_push_event enabled must be true "
            "(a disabled force push is spelled by omitting the block)"
        )
    if block["func"] != FORCE_PUSH_EVENT_FUNC:
        raise ValueError(
            f"schema-3 force_push_event func must be {FORCE_PUSH_EVENT_FUNC!r}"
        )
    if block["mode"] != FORCE_PUSH_EVENT_MODE:
        raise ValueError(
            f"schema-3 force_push_event mode must be {FORCE_PUSH_EVENT_MODE!r}"
        )
    if block["body_name"] != FORCE_PUSH_BODY_NAME:
        raise ValueError(
            f"schema-3 force_push_event body_name must be {FORCE_PUSH_BODY_NAME!r}"
        )
    if block["application_point"] != FORCE_PUSH_APPLICATION_POINT:
        raise ValueError(
            "schema-3 force_push_event application_point must be "
            f"{FORCE_PUSH_APPLICATION_POINT!r} — the wrench lands on the pelvis LINK "
            "ORIGIN, and labelling it as the COM is exactly the Yikang V9 mistake"
        )
    try:
        expected = force_push_event_block(
            enable=True,
            interval_range_s=block["interval_range_s"],
            force_n=block["force_n"],
            duration_s=block["duration_s"],
            control_dt_s=block["control_dt_s"],
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 force_push_event is invalid: {exc}") from exc
    stored_interval = [float(v) for v in block["interval_range_s"]]
    if (
        stored_interval != expected["interval_range_s"]
        or type(block["duration_steps"]) is not int
        or block["duration_steps"] != expected["duration_steps"]
    ):
        raise ValueError(
            "schema-3 force_push_event is internally inconsistent: the stored "
            "interval/duration_steps does not equal the canonical assembly from "
            "duration_s/control_dt_s"
        )
    try:
        expected_full = bind_force_push_runtime_mass(
            expected, robot_mass_kg=block["robot_mass_kg"]
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 force_push_event is invalid: {exc}") from exc
    if (
        isinstance(block["delta_v_equiv_mps"], bool)
        or not isinstance(block["delta_v_equiv_mps"], (int, float))
        or float(block["delta_v_equiv_mps"]) != expected_full["delta_v_equiv_mps"]
    ):
        raise ValueError(
            "schema-3 force_push_event is internally inconsistent: delta_v_equiv_mps "
            "must equal force_n * duration_s / robot_mass_kg recomputed"
        )


def _validate_post_swing_settle_debt_contract(contract: Mapping) -> None:
    """S1 post-swing settle debt block (Jiayi V13 post-swing debts idea, clean main-side redo).

    Absent block = mechanism untouched (legacy/default runs).  A present block must bind the
    exact five-debt formula, the shared same-attempt recovery-window gate, and every margin/scale
    the run trained with; it is mutually exclusive with an enabled Wave-B lower-body mechanism.
    """

    settle = contract.get("post_swing_settle_debt_reward")
    if settle is None:
        return
    settle = _require_exact_mapping_keys(
        settle,
        frozenset(
            {
                "schema_version", "enabled", "probe_enabled", "activation_ledger",
                "weight", "base_lin_margin_mps", "base_lin_scale_mps",
                "base_ang_margin_radps", "base_ang_scale_radps",
                "tilt_margin_rad", "tilt_scale_rad",
                "nominal_root_z_m", "root_height_deadband_m", "root_height_scale_m",
                "foot_slip_margin_mps", "foot_slip_scale_mps",
                "recovery_start_s", "recovery_end_s",
                "racket_command_name", "motion_command_name", "foot_body_names",
                "components", "formula", "gate", "age_source",
                "success_conditioned", "uses_motion_reference",
            }
        ),
        name="schema-3 post_swing_settle_debt_reward",
    )
    if type(settle["schema_version"]) is not int or settle["schema_version"] != 1:
        raise ValueError("schema-3 post_swing_settle_debt_reward schema_version must be 1")
    if (
        not isinstance(settle["enabled"], bool)
        or settle["probe_enabled"] is not True
        or settle["success_conditioned"] is not False
        or settle["uses_motion_reference"] is not False
    ):
        raise ValueError("schema-3 post_swing_settle_debt_reward flags are invalid")
    weight = _wave_finite(settle["weight"], name="post_swing_settle_debt_reward.weight")
    if weight > 0.0 or settle["enabled"] != (weight < 0.0):
        raise ValueError("schema-3 post_swing_settle_debt_reward weight/enabled is invalid")
    for name in (
        "base_lin_scale_mps", "base_ang_scale_radps", "tilt_scale_rad",
        "nominal_root_z_m", "root_height_scale_m", "foot_slip_scale_mps",
    ):
        _wave_finite(settle[name], name=f"post_swing_settle_debt_reward.{name}", positive=True)
    for name in (
        "base_lin_margin_mps", "base_ang_margin_radps", "tilt_margin_rad",
        "root_height_deadband_m", "foot_slip_margin_mps", "recovery_start_s",
    ):
        _wave_finite(
            settle[name], name=f"post_swing_settle_debt_reward.{name}", nonnegative=True
        )
    end = _wave_finite(
        settle["recovery_end_s"],
        name="post_swing_settle_debt_reward.recovery_end_s",
        positive=True,
    )
    if float(settle["recovery_start_s"]) >= end:
        raise ValueError(
            "schema-3 post_swing_settle_debt_reward recovery window must satisfy 0 <= start < end"
        )
    expected = {
        "racket_command_name": "racket_target",
        "motion_command_name": "motion",
        "activation_ledger": "weight_independent_control_step_counters",
        "foot_body_names": ["left_ankle_roll_Link", "right_ankle_roll_Link"],
        "components": [
            "base_quiet_lin",
            "base_quiet_ang",
            "tilt_debt",
            "root_height_debt",
            "settle_foot_slip",
        ],
        "formula": "mean(5x(1-exp(-square(relu(x-margin)/scale))))",
        "gate": "same_attempt_post_strike_age_s_inclusive",
        "age_source": "per_env_exact_strike_control_tick_latch",
    }
    for key, value in expected.items():
        if settle[key] != value:
            raise ValueError(
                f"schema-3 post_swing_settle_debt_reward {key} must be {value!r}"
            )
    articulation_bodies = contract.get("articulation_body_names")
    if (
        not isinstance(articulation_bodies, (list, tuple))
        or any(name not in articulation_bodies for name in settle["foot_body_names"])
    ):
        raise ValueError(
            "schema-3 post_swing_settle_debt_reward foot bodies are absent from articulation"
        )
    if settle["enabled"]:
        for other_key in (
            "lower_body_pose_imitation_reward",
            "lower_body_stability_bundle_reward",
        ):
            other = contract.get(other_key)
            if isinstance(other, Mapping) and other.get("enabled") is True:
                raise ValueError(
                    "schema-3 S1 post_swing_settle_debt and Wave-B lower-body rewards are "
                    "mutually exclusive"
                )


def _validate_qdes_limit_barrier_contract(contract: Mapping) -> None:
    """Wave-Q all-joint q_des position-limit barrier block (Jiayi V14 idea, top-k removed).

    Absent block = mechanism untouched (legacy/default runs).  A present block must bind the
    exact dense all-31-joint formula, the margin fraction, and the deploy-parity position-limit
    source; no top-k and no joint subset are representable.
    """

    barrier = contract.get("qdes_limit_barrier_reward")
    if barrier is None:
        return
    barrier = _require_exact_mapping_keys(
        barrier,
        frozenset(
            {
                "schema_version", "enabled", "probe_enabled", "activation_ledger",
                "weight", "margin_frac", "action_name", "joint_count",
                "joint_order", "position_limit_source", "formula", "gate",
            }
        ),
        name="schema-3 qdes_limit_barrier_reward",
    )
    if type(barrier["schema_version"]) is not int or barrier["schema_version"] != 1:
        raise ValueError("schema-3 qdes_limit_barrier_reward schema_version must be 1")
    if not isinstance(barrier["enabled"], bool) or barrier["probe_enabled"] is not True:
        raise ValueError("schema-3 qdes_limit_barrier_reward flags are invalid")
    weight = _wave_finite(barrier["weight"], name="qdes_limit_barrier_reward.weight")
    if weight > 0.0 or barrier["enabled"] != (weight < 0.0):
        raise ValueError("schema-3 qdes_limit_barrier_reward weight/enabled is invalid")
    margin_frac = _wave_finite(
        barrier["margin_frac"], name="qdes_limit_barrier_reward.margin_frac"
    )
    if not 0.0 < margin_frac < 0.5:
        raise ValueError(
            "schema-3 qdes_limit_barrier_reward margin_frac must be in (0, 0.5)"
        )
    joint_names = contract.get("joint_names")
    articulation_names = contract.get("articulation_joint_names")
    if (
        type(barrier["joint_count"]) is not int
        or barrier["joint_count"] != 31
        or not isinstance(joint_names, (list, tuple))
        or len(joint_names) != 31
        or not isinstance(articulation_names, (list, tuple))
        or [str(value) for value in articulation_names]
        != [str(value) for value in joint_names]
    ):
        raise ValueError(
            "schema-3 qdes_limit_barrier_reward requires identity 31-joint order"
        )
    expected = {
        "action_name": "joint_pos",
        "activation_ledger": "weight_independent_control_step_counters",
        "joint_order": "runtime_articulation_identity",
        "position_limit_source": "articulation.data.soft_joint_pos_limits",
        "formula": (
            "sum(1-exp(-square(relu(margin_frac-min(qdes-lo,hi-qdes)/(hi-lo))/margin_frac)))"
        ),
        "gate": "dense_every_control_step",
    }
    for key, value in expected.items():
        if barrier[key] != value:
            raise ValueError(
                f"schema-3 qdes_limit_barrier_reward {key} must be exactly {value!r}"
            )


def validate_schema3_contract_structure(contract: Mapping) -> None:
    """Validate a schema-3 sidecar without promoting it to a formal-exact lineage.

    Schema 3 binds the instantiated execution contract even for deliberately diagnostic runs
    (for example, a causal continuation on an untagged legacy motion).  Those sidecars still need
    complete, internally consistent runtime facts and an adjacent checkpoint hash binding; the
    narrower :func:`validate_schema3_contract` adds the formal schema-2 motion requirement.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    try:
        schema = int(contract.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid training-contract schema version") from exc
    if schema != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported formal training-contract schema {schema}; expected "
            f"{TRAINING_CONTRACT_SCHEMA_VERSION}"
        )
    missing = [key for key in (*RUNTIME_EXECUTION_KEYS, *SCHEMA3_TASK_KEYS) if key not in contract]
    if missing:
        raise ValueError("schema-3 training contract missing execution facts: " + ", ".join(missing))
    # Optional only as a complete pair.  The OFF/legacy spelling is total absence; an enabled
    # block must bind the exact C++ governor JSON plus the clip layout that defines strike frames.
    planner_task_revision_metadata(contract)
    actor_names_raw = contract["actor_obs_term_names"]
    actor_dims_raw = contract["actor_obs_term_dims"]
    actor_history_raw = contract["observation_history_lengths"]
    if (
        not isinstance(actor_names_raw, (list, tuple))
        or not isinstance(actor_dims_raw, (list, tuple))
        or not isinstance(actor_history_raw, (list, tuple))
        or not actor_names_raw
        or len(actor_names_raw) != len(actor_dims_raw)
        or len(actor_names_raw) != len(actor_history_raw)
    ):
        raise ValueError(
            "schema-3 actor observation names/dims/history must be non-empty equal-length arrays"
        )
    if any(type(name) is not str or not name.strip() for name in actor_names_raw):
        raise ValueError("schema-3 actor observation names must be non-empty strings")
    if len(set(actor_names_raw)) != len(actor_names_raw):
        raise ValueError("schema-3 actor observation names must be unique")
    if any(type(dim) is not int or dim <= 0 for dim in actor_dims_raw):
        raise ValueError("schema-3 actor observation dims must be positive integers")
    if any(type(value) is not int or value <= 0 for value in actor_history_raw):
        raise ValueError("schema-3 actor observation history lengths must be positive integers")
    actor_total = contract["actor_obs_total_dim"]
    if type(actor_total) is not int or actor_total != sum(actor_dims_raw):
        raise ValueError("schema-3 actor_obs_total_dim must equal the sum of term dims")
    for key in ("face_command_enabled", "motion_allow_legacy_link_origin_velocity"):
        if key in contract and not isinstance(contract[key], bool):
            raise ValueError(f"schema-3 {key} must be boolean when present")
    if ACTOR_LEG_REF_MASK_PROVENANCE_KEY in contract:
        epoch = contract[ACTOR_LEG_REF_MASK_PROVENANCE_KEY]
        if (
            isinstance(epoch, bool)
            or type(epoch) is not int
            or epoch != ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH
        ):
            raise ValueError("schema-3 actor_leg_ref_mask_provenance_epoch must be integer 1")
    if "actor_leg_ref_mask" in contract and contract["actor_leg_ref_mask"] is not True:
        raise ValueError("schema-3 actor_leg_ref_mask must be true when present")
    require_actor_leg_ref_mask_provenance(contract)
    if "face_command_pairing" in contract and contract["face_command_pairing"] not in (
        "shared_plus_y",
        "legacy_signed_vs_A",
    ):
        raise ValueError("schema-3 face_command_pairing is invalid")
    if contract.get("actor_obs_contract") == "deploy_parity_face179":
        if contract.get("face_command_enabled") is not True:
            raise ValueError("formal face179 schema-3 contract requires face_command_enabled=true")
        if contract.get("face_command_pairing") != "shared_plus_y":
            raise ValueError("formal face179 schema-3 contract requires shared_plus_y")
        try:
            raw_face_signs = contract["mount_normal_sign_per_clip"]
            if any(isinstance(value, bool) for value in raw_face_signs):
                raise ValueError
            face_signs = tuple(float(value) for value in raw_face_signs)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
            ) from exc
        if face_signs != (1.0, -1.0):
            raise ValueError(
                "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
            )

    joint_names = contract["joint_names"]
    if not isinstance(joint_names, (list, tuple)) or not joint_names:
        raise ValueError("schema-3 joint_names must be a non-empty array")
    n = len(joint_names)
    if len(set(str(value) for value in joint_names)) != n:
        raise ValueError("schema-3 joint_names must be unique")
    raw_actuator_types = contract["joint_actuator_types"]
    actuator_types = (
        list(raw_actuator_types)
        if isinstance(raw_actuator_types, (list, tuple))
        else []
    )
    if len(actuator_types) != n or any(
        value not in ("implicit", "explicit") for value in actuator_types
    ):
        raise ValueError(
            "schema-3 joint_actuator_types must contain one implicit|explicit value per joint"
        )

    def finite_vector(key: str, *, positive: bool) -> None:
        value = contract[key]
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise ValueError(f"schema-3 {key} must contain one value per joint")
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"schema-3 {key} must be numeric") from exc
        if any(
            not math.isfinite(item) or (item <= 0.0 if positive else item < 0.0)
            for item in numbers
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"schema-3 {key} must be finite and {qualifier}")

    finite_vector("joint_stiffness", positive=True)
    finite_vector("joint_damping", positive=True)
    finite_vector("joint_effort_limits", positive=True)
    finite_vector("joint_armature", positive=False)
    finite_vector("joint_friction_coefficients", positive=False)
    finite_vector("joint_velocity_limits", positive=True)

    qdot_hinge = contract.get("joint_velocity_limit_hinge_reward")
    if qdot_hinge is not None:
        if not isinstance(qdot_hinge, Mapping):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward must be an object or null"
            )
        if type(qdot_hinge.get("schema_version")) is not int or qdot_hinge.get(
            "schema_version"
        ) != 1:
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward schema_version must be integer 1"
            )
        enabled = qdot_hinge.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward enabled must be boolean"
            )
        raw_weight = qdot_hinge.get("weight")
        raw_margin = qdot_hinge.get("margin")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward weight must be finite and <= 0"
            )
        if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward margin must be finite and in (0, 1)"
            )
        weight = float(raw_weight)
        margin = float(raw_margin)
        if not math.isfinite(weight) or weight > 0.0:
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward weight must be finite and <= 0"
            )
        if not math.isfinite(margin) or not 0.0 < margin < 1.0:
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward margin must be finite and in (0, 1)"
            )
        if enabled != (weight < 0.0):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward enabled disagrees with weight"
            )
        articulation_joint_names = contract["articulation_joint_names"]
        if (
            n != 31
            or type(qdot_hinge.get("joint_count")) is not int
            or qdot_hinge.get("joint_count") != 31
            or not isinstance(articulation_joint_names, (list, tuple))
            or [str(value) for value in articulation_joint_names]
            != [str(value) for value in joint_names]
        ):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward requires identity 31-joint order"
            )
        expected_fixed = {
            "asset_name": "robot",
            "joint_order": "runtime_articulation_identity",
            "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
            "formula": "mean(relu(abs(qd)/joint_velocity_limits-margin)^2)",
        }
        for key, expected in expected_fixed.items():
            if qdot_hinge.get(key) != expected:
                raise ValueError(
                    "schema-3 joint_velocity_limit_hinge_reward "
                    f"{key} must be exactly {expected!r}"
                )

    qdes_slew = contract.get("processed_qdes_slew_hinge_reward")
    if qdes_slew is not None:
        if not isinstance(qdes_slew, Mapping):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward must be an object"
            )
        if type(qdes_slew.get("schema_version")) is not int or qdes_slew.get(
            "schema_version"
        ) != 1:
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward schema_version must be integer 1"
            )
        enabled = qdes_slew.get("enabled")
        raw_weight = qdes_slew.get("weight")
        raw_margin = qdes_slew.get("margin")
        raw_start = qdes_slew.get("recovery_start_s")
        raw_end = qdes_slew.get("recovery_end_s")
        if not isinstance(enabled, bool) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (raw_weight, raw_margin, raw_start, raw_end)
        ):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward has invalid enabled/weight/margin/window"
            )
        weight, margin, start, end = map(
            float, (raw_weight, raw_margin, raw_start, raw_end)
        )
        if (
            not all(math.isfinite(value) for value in (weight, margin, start, end))
            or weight > 0.0
            or not 0.0 < margin < 1.0
            or start < 0.0
            or start >= end
            or enabled != (weight < 0.0)
        ):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward has invalid enabled/weight/margin/window"
            )
        selected_names = qdes_slew.get("joint_names")
        expected_names = {
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        }
        if (
            type(qdes_slew.get("joint_count")) is not int
            or qdes_slew.get("joint_count") != 15
            or not isinstance(selected_names, (list, tuple))
            or len(selected_names) != 15
            or set(selected_names) != expected_names
            or [name for name in joint_names if name in expected_names]
            != list(selected_names)
        ):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward requires the runtime-ordered 15 A3 waist/leg joints"
            )
        expected_fixed = {
            "action_name": "joint_pos",
            "command_name": "racket_target",
            "joint_order": "runtime_articulation_subsequence",
            "control_dt_s": 0.02,
            "control_dt_source": "env_cfg.sim.dt_times_decimation",
            "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
            "age_source": "per_env_exact_strike_control_tick_latch",
            "formula": (
                "mean(1-exp(-square(relu(abs(delta_processed_qdes)/(joint_velocity_limits*0.02)-margin)/(1-margin))))"
            ),
            "gate": "same_attempt_post_strike_age_s_inclusive",
        }
        for key, expected in expected_fixed.items():
            if qdes_slew.get(key) != expected:
                raise ValueError(
                    "schema-3 processed_qdes_slew_hinge_reward "
                    f"{key} must be exactly {expected!r}"
                )
    _validate_lower_body_wave_contracts(
        contract,
        [str(value) for value in joint_names],
        [str(value) for value in contract["articulation_joint_names"]],
    )
    _validate_post_swing_settle_debt_contract(contract)
    _validate_qdes_limit_barrier_contract(contract)
    _validate_push_robot_event_contract(contract)
    _validate_force_push_event_contract(contract)
    if contract["joint_friction_backend"] != JOINT_FRICTION_BACKEND:
        raise ValueError("schema-3 joint_friction_backend must be physx")
    if contract["joint_friction_semantics"] != JOINT_FRICTION_SEMANTICS:
        raise ValueError(
            "schema-3 joint_friction_semantics does not describe Isaac/PhysX joint friction"
        )
    if contract["joint_friction_units"] != JOINT_FRICTION_UNITS:
        raise ValueError("schema-3 joint_friction_units must be dimensionless")

    articulation_body_names_raw = contract["articulation_body_names"]
    if not isinstance(articulation_body_names_raw, (list, tuple)):
        raise ValueError("schema-3 articulation_body_names must be non-empty and unique")
    articulation_body_names = [str(value) for value in articulation_body_names_raw]
    if (
        not articulation_body_names
        or any(not value for value in articulation_body_names)
        or len(set(articulation_body_names)) != len(articulation_body_names)
    ):
        raise ValueError("schema-3 articulation_body_names must be non-empty and unique")
    selected_body_names_raw = contract["body_names"]
    selected_body_indices = contract["body_indices"]
    if (
        not isinstance(selected_body_names_raw, (list, tuple))
        or not isinstance(selected_body_indices, (list, tuple))
        or len(selected_body_names_raw) != len(selected_body_indices)
        or not selected_body_names_raw
    ):
        raise ValueError("schema-3 selected body names/indices are malformed")
    selected_body_names = [str(value) for value in selected_body_names_raw]
    if any(not value for value in selected_body_names) or len(set(selected_body_names)) != len(
        selected_body_names
    ):
        raise ValueError("schema-3 selected body names must be non-empty and unique")
    try:
        parsed_body_indices = []
        for raw_index in selected_body_indices:
            index = int(raw_index)
            if isinstance(raw_index, bool) or float(raw_index) != float(index):
                raise ValueError
            parsed_body_indices.append(index)
        if any(
            index < 0 or index >= len(articulation_body_names)
            for index in parsed_body_indices
        ):
            raise IndexError
        resolved_selected = [
            articulation_body_names[index] for index in parsed_body_indices
        ]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("schema-3 body_indices are outside articulation_body_names") from exc
    if resolved_selected != selected_body_names:
        raise ValueError("schema-3 selected body names do not match articulation body indices")

    segment_lengths = contract["motion_segment_lengths"]
    clip_fps = contract["motion_clip_fps"]
    kinematics = contract["motion_kinematics_contracts"]
    if not isinstance(segment_lengths, (list, tuple)) or not segment_lengths:
        raise ValueError("schema-3 motion_segment_lengths must be positive")
    try:
        parsed_segment_lengths = [int(value) for value in segment_lengths]
    except (TypeError, ValueError) as exc:
        raise ValueError("schema-3 motion_segment_lengths must be positive") from exc
    if any(value <= 0 for value in parsed_segment_lengths):
        raise ValueError("schema-3 motion_segment_lengths must be positive")
    if (
        not isinstance(clip_fps, (list, tuple))
        or len(clip_fps) != len(segment_lengths)
        or not isinstance(kinematics, (list, tuple))
        or len(kinematics) != len(segment_lengths)
    ):
        raise ValueError("schema-3 motion fps/kinematics counts must match segments")
    try:
        policy_hz = 1.0 / float(contract["policy_step_dt_s"])
        parsed_fps = [float(value) for value in clip_fps]
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("schema-3 motion_clip_fps/policy_step_dt_s are invalid") from exc
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or not math.isclose(value, policy_hz, rel_tol=0.0, abs_tol=1e-9)
        for value in parsed_fps
    ):
        raise ValueError("schema-3 every motion clip fps must equal the policy rate")
    expected_body_order = articulation_body_names
    clip_exact_flags = []
    for index, item in enumerate(kinematics):
        if not isinstance(item, Mapping):
            raise ValueError(f"schema-3 motion kinematics clip {index} must be an object")
        missing_item = [
            key
            for key in (
                "schema_version",
                "body_pos_point",
                "body_lin_vel_point",
                "body_names",
                "exact",
            )
            if key not in item
        ]
        if missing_item:
            raise ValueError(
                f"schema-3 motion kinematics clip {index} is missing "
                + ", ".join(missing_item)
            )
        if not isinstance(item["exact"], bool):
            raise ValueError(f"schema-3 motion kinematics clip {index} exact must be boolean")
        clip_exact = item["exact"] is True
        clip_exact_flags.append(clip_exact)
        try:
            item_schema = (
                None if item.get("schema_version") is None else int(item["schema_version"])
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has invalid schema_version"
            )
        if item_schema not in (None, 1, 2):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has unsupported schema_version "
                f"{item_schema!r}"
            )
        pos_point = item.get("body_pos_point")
        vel_point = item.get("body_lin_vel_point")
        if pos_point not in (None, "link_origin") or vel_point not in (
            None,
            "link_origin",
            "center_of_mass",
        ):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has invalid point semantics"
            )
        raw_body_names = item.get("body_names")
        if raw_body_names is not None:
            if not isinstance(raw_body_names, (list, tuple)):
                raise ValueError(
                    f"schema-3 motion kinematics clip {index} body_names must be an array or null"
                )
            item_body_names = [str(value) for value in raw_body_names]
            if (
                not item_body_names
                or any(not value for value in item_body_names)
                or len(set(item_body_names)) != len(item_body_names)
                or item_body_names != expected_body_order
            ):
                raise ValueError(
                    f"schema-3 motion kinematics clip {index} body_names do not match the "
                    "runtime articulation"
                )
        status = item.get("status")
        if (not clip_exact) and (not isinstance(status, str) or not status.strip()):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} status must be non-empty"
            )
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} status must be non-empty when present"
            )
        if clip_exact and (
            item_schema != 2
            or pos_point != "link_origin"
            or vel_point != "center_of_mass"
            or raw_body_names is None
        ):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} claims exact without an exact "
                "schema-2 body order"
            )
        if not clip_exact and item_schema == 2:
            raise ValueError(
                f"schema-3 motion kinematics clip {index} is schema-2 but marked inexact"
            )
    resolve_motion_body_lin_vel_points(kinematics)
    motion_exact = contract["motion_kinematics_exact"]
    if not isinstance(motion_exact, bool) or motion_exact != all(clip_exact_flags):
        raise ValueError(
            "schema-3 motion_kinematics_exact disagrees with the per-clip contracts"
        )


def validate_schema3_contract(contract: Mapping) -> None:
    """Validate the formal-exact subset of the schema-3 execution contract."""

    validate_schema3_contract_structure(contract)
    if contract["motion_kinematics_exact"] is not True:
        raise ValueError("schema-3 formal lineage requires motion_kinematics_exact=true")


def checkpoint_claims_contract(checkpoint: Mapping) -> bool:
    """Return whether checkpoint infos claim any adjacent training-contract binding."""

    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    return isinstance(infos, Mapping) and any(
        key in infos
        for key in (
            CHECKPOINT_CONTRACT_SCHEMA_KEY,
            CHECKPOINT_CONTRACT_SHA_KEY,
            CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY,
        )
    )


def checkpoint_contract_binding(checkpoint: Mapping) -> tuple[int | None, str | None]:
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    if not isinstance(infos, Mapping):
        return None, None
    schema_raw = infos.get(CHECKPOINT_CONTRACT_SCHEMA_KEY)
    digest_raw = infos.get(CHECKPOINT_CONTRACT_SHA_KEY)
    try:
        schema = None if schema_raw is None else int(schema_raw)
    except (TypeError, ValueError):
        return None, None
    digest = None if digest_raw is None else str(digest_raw).strip().lower()
    return schema, digest


def checkpoint_contract_lineage_exact(checkpoint: Mapping) -> bool:
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    value = infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY) if isinstance(infos, Mapping) else None
    return value in (True, 1, "1")


def require_checkpoint_contract_binding(
    checkpoint: Mapping, *, schema: int, sha256: str, require_lineage_exact: bool = True
) -> None:
    bound_schema, bound_sha = checkpoint_contract_binding(checkpoint)
    expected_sha = str(sha256).strip().lower()
    if bound_schema != schema or bound_sha != expected_sha:
        raise ValueError(
            "checkpoint is not bound to the adjacent training contract: "
            f"checkpoint schema/sha={bound_schema}/{bound_sha}, file={schema}/{expected_sha}"
        )
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise ValueError("training-contract SHA256 is malformed")
    if require_lineage_exact and not checkpoint_contract_lineage_exact(checkpoint):
        infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
        lineage_exact = (
            infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY)
            if isinstance(infos, Mapping)
            else None
        )
        raise ValueError(
            "checkpoint contract binding is not exact-lineage eligible "
            f"({CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY}={lineage_exact!r})"
        )
