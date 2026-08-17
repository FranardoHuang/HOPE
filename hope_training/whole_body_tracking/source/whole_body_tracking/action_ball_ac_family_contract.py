#!/usr/bin/env python3
"""Fail-closed family contract for fresh matched ActionBall A/C.

C07 names the superseded schema-v2 245/353 pre-contact-mask experiment.  C10
is its common-pre/contact, post-contact-placement replacement.  The C07 layout
and SHA remain below only so receipts/checkpoints can be
identified and rejected; they are not a target ABI and cannot be loaded.

The replacement deliberately leaves actor/critic widths unfrozen.  A and C
must share every pre/contact observation value, normalizer operation, provider,
desired-at-contact fact, and all ten one-shot strike/contact payments.  Their
only family treatment is an after-the-fact placement-guidance gain paid from
the *original shot's* full-key delayed mailbox on a selected-rubber-contact
cohort.  Common on-table success, scorer, denominator, and curriculum do not
change.  No constructed runtime has passed this contract, therefore launch
readiness remains false.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import hmac
import inspect
import json
import math
from numbers import Real
from pathlib import Path
import re
import secrets
import struct
import types
from typing import Callable, Mapping, Optional, Sequence, Tuple

import action_ball_continuous_successor as c03_successor
import action_ball_landing_outcome_mailbox as c05_mailbox
import action_ball_landing_placement as c04_landing
import action_ball_placement_success as c06_success


SCHEMA_VERSION = 2
C10_SCHEMA_VERSION = 1
FAMILIES = ("A", "C")
BACKENDS = ("isaac", "mujoco")
GUIDANCE_ALLOWED_DELTA_PATHS = (
    "treatment.guide_enabled",
    "treatment.guide_reward_weight",
    "treatment_witness.actor_treated_guide_sha256",
    "treatment_witness.critic_treated_guide_sha256",
    "treatment_witness.guide_reward_sha256",
    "treatment_witness.guide_valid_count",
    "treatment_witness.guide_reward_all_positive_zero",
    "treatment_witness.treated_equals_normalized",
    "treatment_witness.treated_all_positive_zero",
)
IDENTITY_ONLY_FIELDS = (
    "live_job_id",
    "run_name",
    "experiment_name",
    "namespace",
    "output_dir",
    "artifact_id",
)
IDENTITY_ALLOWED_DELTA_PATHS = tuple(
    "identity.%s" % name for name in IDENTITY_ONLY_FIELDS
)
PAIR_ALLOWED_DELTA_PATHS = (
    "family",
) + IDENTITY_ALLOWED_DELTA_PATHS + GUIDANCE_ALLOWED_DELTA_PATHS

ACTOR_WIDTH = 245
CRITIC_WIDTH = 353
ACTOR_GUIDE_SLICE = (208, 217)
CRITIC_GUIDE_SLICE = (316, 325)
DESIRED_AT_CONTACT_WIDTH = 9
DESIRED_AT_CONTACT_COMPONENTS = ("position", "velocity", "face")
DESIRED_AT_CONTACT_COMPONENT_WIDTHS = (3, 3, 3)
DESIRED_AT_CONTACT_SOURCE_SEMANTICS = (
    "desired_at_contact_position_velocity_face_v1"
)
DISABLED_GUIDANCE_FILL = "positive_zero"
DISABLED_GUIDANCE_MASK_STAGE = "final_after_normalization"
FIXED_TAPE_PHASES = (
    "hidden_wait",
    "swing_pre_contact",
    "post_strike_suffix",
    "recovery_bridge",
    "ready_hold",
)
FIXED_TAPE_CONTEXTS = (
    "initial_observation",
    "transition_next_observation",
    "terminal_bootstrap_observation",
    "checkpoint_export_observation",
)
FIXED_TAPE_POINTS = tuple(
    "%s/%s" % (phase, context)
    for phase in FIXED_TAPE_PHASES
    for context in FIXED_TAPE_CONTEXTS
)
FLOAT32_BYTES = 4
ACTOR_GUIDE_VALID_INDEX = 231
CRITIC_GUIDE_VALID_INDEX = 339
ACTOR_TASK_VALID_INDEX = 229
CRITIC_TASK_VALID_INDEX = 337
ACTOR_BALL_VALID_INDEX = 230
CRITIC_BALL_VALID_INDEX = 338
ACTOR_PHASE_SLICE = (224, 229)
CRITIC_PHASE_SLICE = (332, 337)
ACTOR_TARGET_SLICE = (206, 208)
CRITIC_TARGET_SLICE = (314, 316)
ACTOR_CURRENT_OUTCOME_PENDING_INDEX = 234
CRITIC_CURRENT_OUTCOME_PENDING_INDEX = 342
ACTOR_NET_CROSSED_INDEX = 235
CRITIC_NET_CROSSED_INDEX = 343
ACTOR_NET_CLEAR_INDEX = 236
CRITIC_NET_CLEAR_INDEX = 344
ACTOR_PREVIOUS_SHOT_VALID_INDEX = 237
CRITIC_PREVIOUS_SHOT_VALID_INDEX = 345
_TASK_VALID_BY_PHASE = (0, 1, 1, 0, 0)
_BALL_VALID_BY_PHASE = (0, 1, 1, 0, 0)
_RAW_GUIDE_VALID_BY_PHASE = (0, 1, 0, 0, 0)
_OUTCOME_PENDING_BY_PHASE = (0, 0, 1, 0, 0)
_NET_CROSSED_BY_PHASE = (0, 0, 1, 0, 0)
_NET_CLEAR_BY_PHASE = (0, 0, 1, 0, 0)
_PREVIOUS_SHOT_VALID_BY_PHASE = (0, 0, 0, 1, 1)

TYPED_SCHEMA_READY = False
RESOLVED_OBJECT_BUILDER_READY = False
FIXED_TAPE_SCHEMA_READY = False
FIXED_TAPE_VALUE_WITNESS_READY = False
CALLABLE_NONINTERFERENCE_READY = False
RUNTIME_TEMPORAL_WITNESS_READY = False
NORMALIZER_REEXECUTION_READY = False
CHECKPOINT_RUNTIME_EXTRACTOR_READY = False
CONSTRUCTED_ENV_EXTRACTOR_READY = False
LAUNCH_GATE_READY = False

# C10 replacement state.  Widths are intentionally not guessed before a real
# constructed-runtime projection exists.
C10_CONTRACT_STATUS = "ACTIVE_SCHEMA_WIDTH_UNFROZEN_NO_LAUNCH"
C10_ACTOR_WIDTH = None
C10_CRITIC_WIDTH = None
C10_TYPED_FAMILY_SCHEMA_READY = True
C10_CONSTRUCTED_RUNTIME_READY = False
C10_FIXED_TAPE_VALUE_READY = False
C10_CALLABLE_NONINTERFERENCE_READY = False
C10_PHYSICS_STAMP_RUNTIME_READY = False
C10_SLOT_CAPACITY_RUNTIME_READY = False
C10_MIDSEQUENCE_CHECKPOINT_RUNTIME_READY = False
C10_LAUNCH_GATE_READY = False
COMPATIBILITY_245_353_LAYOUT_EXPORTS_ONLY = True
C10_EVIDENCE_LEVEL = "typed_schema_candidate_no_constructed_capability_v1"
C10_CONTRACT_AUTHORITY_KIND = (
    "action_ball_common_precontact_postcontact_placement_parent_authority_c10_v1"
)
C10_RUNTIME_TARGET_AUTHORITY_SOURCE_SHA256 = hashlib.sha256(
    Path(c03_successor.__file__).read_bytes()
).hexdigest()
C10_LANDING_PLACEMENT_AUTHORITY_SOURCE_SHA256 = (
    "3e2e056336a8c021c20bd255c474487cb2346a3dcdfcca8a1b1a608dd90636e2"
)
if hashlib.sha256(Path(c04_landing.__file__).read_bytes()).hexdigest() != (
    C10_LANDING_PLACEMENT_AUTHORITY_SOURCE_SHA256
):
    raise RuntimeError("C10 canonical C04 landing-placement source SHA drifted")
_C10_CANONICAL_C04_SCORER = c04_landing.score_landing_placement
C10_LANDING_PLACEMENT_TORCH_AUTHORITY_SOURCE_SHA256 = (
    "a1e8c41089ff7373b2befdf6b6e7719ba315e7987c445633af179cee3d237c4d"
)
C10_LANDING_PLACEMENT_TORCH_AUTHORITY_TEST_SHA256 = (
    "ef9b93c0283d439447092b08eba9a8700bc3ba1ae8389c754d3fb7c37398c86e"
)
C10_LANDING_OUTCOME_MAILBOX_AUTHORITY_SOURCE_SHA256 = hashlib.sha256(
    Path(c05_mailbox.__file__).read_bytes()
).hexdigest()
C10_PLACEMENT_SUCCESS_AUTHORITY_SOURCE_SHA256 = hashlib.sha256(
    Path(c06_success.__file__).read_bytes()
).hexdigest()

_COMMON_COMPONENT_NAMES = (
    "question_provider",
    "common_snapshot",
    "common_observation_pack",
    "normalizer",
    "guide_projection",
    "common_reward",
    "guide_reward",
    "exact_contact",
    "landing_outcome",
    "recovery_termination",
    "plant_step",
    "ppo_step",
)
_COMMON_INPUT_SCHEMAS = {
    "question_provider": "action_ball_common_question_v1",
    "common_snapshot": "action_ball_frozen_common_snapshot_input_v1",
    "common_observation_pack": "action_ball_frozen_common_snapshot_v1",
    "normalizer": "action_ball_masked_observation_pair_v1",
    "guide_projection": "action_ball_normalized_observation_and_guide_switch_v1",
    "common_reward": "action_ball_common_reward_facts_v1",
    "guide_reward": "action_ball_desired_contact_facts_and_weight_v1",
    "exact_contact": "action_ball_full_key_strike_fact_v1",
    "landing_outcome": "action_ball_full_key_landing_fact_v1",
    "recovery_termination": "action_ball_common_lifecycle_snapshot_v1",
    "plant_step": "action_ball_common_plant_step_v1",
    "ppo_step": "action_ball_common_rollout_batch_v1",
}
_FORBIDDEN_RUNTIME_IDENTIFIERS = frozenset(
    {
        "family",
        "family_axis",
        "family_id",
        "family_name",
        "obs_mode",
        "actor_contract",
        "run_name",
        "treatment",
        "variant",
        "experiment_name",
        "namespace",
        "output_dir",
        "artifact_id",
    }
).union(IDENTITY_ONLY_FIELDS)
_FORBIDDEN_DYNAMIC_INTROSPECTION = frozenset(
    {
        "eval",
        "exec",
        "getattr",
        "globals",
        "hasattr",
        "import_module",
        "locals",
        "setattr",
        "vars",
        "__dict__",
        "__getattribute__",
    }
)
_SAFE_GLOBAL_MODULE_ROOTS = frozenset({"math", "numpy", "operator", "torch"})
_BUILD_TOKEN = object()
_BUILD_AUTH_KEY = secrets.token_bytes(32)


class ACFamilyContractError(ValueError):
    """The supplied runtime objects do not establish the matched contract."""


@dataclass(frozen=True)
class FieldSpec:
    """One portable, ordered observation field."""

    name: str
    width: int
    source_semantics: str
    frame: str
    units: str
    validity: str
    normalizer: str

    def to_mapping(self, start: int) -> dict[str, object]:
        return {
            "frame": self.frame,
            "name": self.name,
            "normalizer": self.normalizer,
            "slice": [start, start + self.width],
            "source_semantics": self.source_semantics,
            "units": self.units,
            "validity": self.validity,
            "width": self.width,
        }


def _field(
    name: str,
    width: int,
    source: str,
    frame: str,
    units: str,
    validity: str = "always",
    normalizer: str = "masked_welford_v1",
) -> FieldSpec:
    return FieldSpec(name, width, source, frame, units, validity, normalizer)


_ACTOR_PREFIX = (
    _field("actual_base_pose_lin_vel_world", 12, "actual_base_fused_v2", "canonical_hope_world", "m3+orientation6d6+mps3"),
    _field("base_ang_vel_body", 3, "pelvis_imu_gyro_v1", "pelvis_body", "radps3"),
    _field("joint_pos", 31, "actual_joint_pos_minus_default_v1", "a3_actor_joint_order", "rad31"),
    _field("joint_vel", 31, "actual_joint_velocity_v1", "a3_actor_joint_order", "radps31"),
    _field("actions", 31, "previous_normalized_action_v1", "a3_actor_joint_order", "unitless31"),
    _field("racket_site_achieved_now_heading", 9, "official_racket_site_actual_v1", "current_actual_base_yaw_heading", "relative_m3+mps3+signed_normal3"),
    _field("teacher_joint_pos", 31, "teacher_joint_pos_minus_default_v1", "a3_actor_joint_order", "rad31"),
    _field("teacher_joint_vel", 31, "teacher_joint_velocity_v1", "a3_actor_joint_order", "radps31"),
    _field("racket_site_teacher_now_heading", 9, "measured_teacher_racket_now_v1", "current_actual_base_yaw_heading", "relative_m3+mps3+signed_normal3"),
    _field("racket_site_teacher_at_reference_hit_heading", 9, "measured_teacher_racket_hit_v1", "current_actual_base_yaw_heading", "relative_m3+mps3+signed_normal3"),
)

_CRITIC_PREFIX = (
    _field("command", 62, "teacher_joint_command_v1", "a3_actor_joint_order", "rad31+radps31"),
    _field("motion_anchor_pos_b", 3, "motion_anchor_position_v1", "motion_anchor_body", "m3"),
    _field("motion_anchor_ori_b", 6, "motion_anchor_orientation_v1", "motion_anchor_body", "orientation6d6"),
    _field("body_pos", 42, "privileged_body_position_v1", "motion_anchor_body", "m42"),
    _field("body_ori", 84, "privileged_body_orientation_v1", "motion_anchor_body", "orientation6d84"),
    _field("base_lin_vel", 3, "privileged_base_linear_velocity_v1", "pelvis_body", "mps3"),
    _field("base_ang_vel", 3, "privileged_base_angular_velocity_v1", "pelvis_body", "radps3"),
    _field("joint_pos", 31, "actual_joint_pos_minus_default_v1", "a3_actor_joint_order", "rad31"),
    _field("joint_vel", 31, "actual_joint_velocity_v1", "a3_actor_joint_order", "radps31"),
    _field("actions", 31, "previous_normalized_action_v1", "a3_actor_joint_order", "unitless31"),
    _field("racket_site_teacher_at_reference_hit_heading", 9, "measured_teacher_racket_hit_v1", "current_actual_base_yaw_heading", "relative_m3+mps3+signed_normal3"),
)

_COMMON_SUFFIX = (
    _field("shot_ball_position_heading", 3, "causal_live_shot_ball_state_v1", "current_actual_base_yaw_heading", "relative_m3", "ball_state_valid"),
    _field("shot_ball_velocity_heading", 3, "causal_live_shot_ball_state_v1", "current_actual_base_yaw_heading", "mps3", "ball_state_valid"),
    _field("shot_ball_spin_heading", 3, "causal_live_shot_ball_state_v1", "current_actual_base_yaw_heading", "radps3", "ball_state_valid"),
    _field("landing_target_xy_table", 2, "full_task_receipt_target_xy_v1", "canonical_table_xy", "m2", "task_valid"),
    _field("guide_desired_contact_position_heading", 3, DESIRED_AT_CONTACT_SOURCE_SEMANTICS, "current_actual_base_yaw_heading", "relative_m3", "guide_valid"),
    _field("guide_desired_contact_velocity_heading", 3, DESIRED_AT_CONTACT_SOURCE_SEMANTICS, "current_actual_base_yaw_heading", "mps3", "guide_valid"),
    _field("guide_desired_contact_face_heading", 3, DESIRED_AT_CONTACT_SOURCE_SEMANTICS, "current_actual_base_yaw_heading", "signed_normal3", "guide_valid"),
    _field("desired_base_xy_world", 2, "full_task_receipt_base_station_v1", "canonical_hope_world", "m2", "task_valid"),
    _field("time_to_contact", 1, "scheduled_contact_clock_v1", "task_clock", "signed_s1", "task_valid"),
    _field("time_to_teacher_start", 1, "scheduled_teacher_start_clock_v1", "teacher_clock", "signed_s1", "task_valid"),
    _field("phase_elapsed", 1, "public_lifecycle_phase_clock_v1", "phase_clock", "s1"),
    _field("time_to_next_reveal", 1, "immutable_next_reveal_clock_v1", "sequence_clock", "signed_s1"),
    _field("ball_estimate_age", 1, "causal_ball_estimate_age_v1", "sensor_clock", "s1", "ball_state_valid"),
    _field("public_phase_one_hot", 5, "public_lifecycle_phase_v1", "lifecycle_state", "binary5", "always", "passthrough_binary_v1"),
    _field("task_valid", 1, "atomic_public_task_valid_v1", "lifecycle_state", "binary1", "always", "passthrough_binary_v1"),
    _field("ball_state_valid", 1, "causal_ball_state_valid_v1", "lifecycle_state", "binary1", "always", "passthrough_binary_v1"),
    _field("guide_valid", 1, "resolved_guide_treatment_valid_v1", "treatment_state", "binary1", "always", "passthrough_binary_v1"),
    _field("current_strike_reached", 1, "full_key_strike_latch_v1", "shot_history", "binary1", "task_valid", "passthrough_binary_v1"),
    _field("current_selected_contact", 1, "selected_rubber_contact_latch_v1", "shot_history", "binary1", "task_valid", "passthrough_binary_v1"),
    _field("current_outcome_pending", 1, "landing_mailbox_open_latch_v1", "shot_history", "binary1", "task_valid", "passthrough_binary_v1"),
    _field("current_ball_net_crossed", 1, "landing_mailbox_net_crossed_latch_v1", "shot_history", "binary1", "current_outcome_pending", "passthrough_binary_v1"),
    _field("current_ball_net_clear", 1, "landing_mailbox_net_clear_latch_v1", "shot_history", "binary1", "current_outcome_pending", "passthrough_binary_v1"),
    _field("previous_shot_valid", 1, "previous_paid_shot_summary_v1", "shot_history", "binary1", "always", "passthrough_binary_v1"),
    _field("previous_selected_contact", 1, "previous_paid_shot_summary_v1", "shot_history", "binary1", "previous_shot_valid", "passthrough_binary_v1"),
    _field("previous_first_crossing_valid", 1, "previous_paid_shot_summary_v1", "shot_history", "binary1", "previous_shot_valid", "passthrough_binary_v1"),
    _field("previous_on_table", 1, "previous_paid_shot_summary_v1", "shot_history", "binary1", "previous_shot_valid", "passthrough_binary_v1"),
    _field("previous_target_error", 1, "previous_paid_shot_summary_v1", "shot_history", "m1", "previous_first_crossing_valid"),
    _field("previous_target_xy_table", 2, "previous_paid_shot_summary_v1", "canonical_table_xy", "m2", "previous_shot_valid"),
    _field("shot_index", 1, "carry_chain_shot_index_v1", "shot_history", "count1", "task_valid"),
)

ACTOR_LAYOUT = _ACTOR_PREFIX + _COMMON_SUFFIX
CRITIC_LAYOUT = _CRITIC_PREFIX + _COMMON_SUFFIX


def _layout_mapping(layout: Sequence[FieldSpec]) -> list[dict[str, object]]:
    offset = 0
    result = []
    for spec in layout:
        result.append(spec.to_mapping(offset))
        offset += spec.width
    return result


def _layout_width(layout: Sequence[FieldSpec]) -> int:
    return sum(spec.width for spec in layout)


if _layout_width(ACTOR_LAYOUT) != ACTOR_WIDTH:
    raise RuntimeError("typed actor ABI width is not 245")
if _layout_width(CRITIC_LAYOUT) != CRITIC_WIDTH:
    raise RuntimeError("typed critic ABI width is not 353")

PORTABLE_ABI_MAPPING = {
    "actor": _layout_mapping(ACTOR_LAYOUT),
    "actor_guide_slice": list(ACTOR_GUIDE_SLICE),
    "actor_width": ACTOR_WIDTH,
    "critic": _layout_mapping(CRITIC_LAYOUT),
    "critic_guide_slice": list(CRITIC_GUIDE_SLICE),
    "critic_width": CRITIC_WIDTH,
    "disabled_guide_fill": DISABLED_GUIDANCE_FILL,
    "disabled_guide_mask_stage": DISABLED_GUIDANCE_MASK_STAGE,
    "causal_scope": "pre_contact_guide_ablation_not_strict_landing_only_v1",
    "checkpoint_policy": "fresh_common_initial_state_no_cross_family_resume_v1",
    "fixed_tape_contexts": list(FIXED_TAPE_CONTEXTS),
    "fixed_tape_phases": list(FIXED_TAPE_PHASES),
    "fixed_tape_points": list(FIXED_TAPE_POINTS),
    "kind": "action_ball_matched_ac_245_353_portable_abi_v2",
    "next_reveal_policy": "immutable_public_remaining_clock_v1",
    "normalizer_policy": "shared_raw_input_then_final_guide_mask_v1",
    "outcome_policy": "settle_and_pay_before_next_reveal_v1",
    "schema_version": SCHEMA_VERSION,
}


def canonical_sha256(value: object) -> str:
    normalized = _canonicalize(value, path="$", ancestors=set())
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedCallableInput:
    """Actual callable plus actual bound inputs and scanned dependency closure."""

    function: Callable[..., object]
    bound_params: Mapping[str, object]
    input_schema_id: str
    dependency_functions: Tuple[Callable[..., object], ...] = ()


@dataclass(frozen=True)
class ResolvedObservationTerm:
    """One resolved manager term; order and width come from the live manager."""

    name: str
    width: int
    producer: ResolvedCallableInput


@dataclass(frozen=True)
class ResolvedObservationGroup:
    terms: Tuple[ResolvedObservationTerm, ...]


@dataclass(frozen=True)
class CommonRuntimeCallables:
    question_provider: ResolvedCallableInput
    common_snapshot: ResolvedCallableInput
    common_observation_pack: ResolvedCallableInput
    normalizer: ResolvedCallableInput
    guide_projection: ResolvedCallableInput
    common_reward: ResolvedCallableInput
    guide_reward: ResolvedCallableInput
    exact_contact: ResolvedCallableInput
    landing_outcome: ResolvedCallableInput
    recovery_termination: ResolvedCallableInput
    plant_step: ResolvedCallableInput
    ppo_step: ResolvedCallableInput


@dataclass(frozen=True)
class FixedTapeBytes:
    """Bytes captured from a real same-action A/C tape.

    Every tuple is ordered exactly as :data:`FIXED_TAPE_POINTS`: five public
    phases crossed with initial, next-observation, terminal-bootstrap, and
    checkpoint-export assembly paths.  Float tensors use finite little-endian
    float32 bytes and exact ABI widths.  ``*_normalized_pre_treatment`` is the
    neutral shared normalizer output; ``*_final`` is what the policy/value
    network actually receives after the sole guide mask.
    """

    tape_bytes: bytes
    actor_raw_pre_treatment: Tuple[bytes, ...]
    critic_raw_pre_treatment: Tuple[bytes, ...]
    actor_normalized_pre_treatment: Tuple[bytes, ...]
    critic_normalized_pre_treatment: Tuple[bytes, ...]
    actor_final: Tuple[bytes, ...]
    critic_final: Tuple[bytes, ...]
    common_reward: Tuple[bytes, ...]
    guide_reward: Tuple[bytes, ...]
    termination: Tuple[bytes, ...]
    lifecycle_phase: Tuple[bytes, ...]
    call_input_trace: Tuple[bytes, ...]
    outcome_atomicity: "OutcomeAtomicityWitness"


@dataclass(frozen=True)
class OutcomeAtomicityWitness:
    """Required single-slot seam for OPEN -> settle/pay/close -> next OPEN."""

    previous_reveal_tick: int
    open_observation_tick: int
    settlement_tick: int
    payment_tick: int
    close_tick: int
    closed_observation_tick: int
    next_reveal_tick: int
    next_open_tick: int
    observed_mailbox_states: Tuple[str, ...]
    ordered_events: Tuple[str, ...]


@dataclass(frozen=True)
class FreshNormalizerState:
    """Closed fresh Welford state; live extraction/re-execution is still pending."""

    actor_sample_count: int
    critic_sample_count: int
    actor_mean_f32: bytes
    actor_m2_f32: bytes
    critic_mean_f32: bytes
    critic_m2_f32: bytes
    epsilon: float
    clip: float


@dataclass(frozen=True)
class FreshCheckpointState:
    """Exact fresh initialization bytes; resume parents are forbidden for A/C."""

    policy_state_bytes: bytes
    optimizer_state_bytes: bytes
    rng_state_bytes: bytes
    rollout_state_bytes: bytes
    parent_checkpoint_sha256: Optional[str]
    resume_requested: bool


@dataclass(frozen=True)
class ResolvedRuntimeInputs:
    family: str
    backend: str
    backend_binding_bytes: bytes
    initial_normalizer_state: FreshNormalizerState
    fresh_checkpoint_state: FreshCheckpointState
    fixed_inter_shot_cadence_ticks: int
    policy_dt_seconds: float
    identity: Mapping[str, str]
    actor: ResolvedObservationGroup
    critic: ResolvedObservationGroup
    common: CommonRuntimeCallables
    fixed_tape: FixedTapeBytes
    guide_enabled: bool
    guide_reward_weight: float


@dataclass(frozen=True)
class ACFamilyPairValidation:
    schema_version: int
    a_projection_sha256: str
    c_projection_sha256: str
    common_runtime_sha256: str
    allowed_delta_paths: Tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "allowed_delta_paths": list(self.allowed_delta_paths),
            "a_projection_sha256": self.a_projection_sha256,
            "c_projection_sha256": self.c_projection_sha256,
            "common_runtime_sha256": self.common_runtime_sha256,
            "schema_version": self.schema_version,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True)
class ResolvedACProjection:
    """Builder-owned projection.  Direct construction is rejected by validation."""

    _payload_json: bytes
    _auth_tag: bytes
    _token: object

    def to_mapping(self) -> dict[str, object]:
        value = json.loads(self._payload_json.decode("ascii"))
        if not isinstance(value, dict):
            raise ACFamilyContractError("builder-owned projection payload is not a mapping")
        return value

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._payload_json).hexdigest()


def build_resolved_runtime_projection(
    inputs: ResolvedRuntimeInputs,
) -> ResolvedACProjection:
    """Reject the superseded 245/353 pre-contact-mask projection.

    Kept as a named tombstone so an old caller fails loudly instead of silently
    creating a schema-v2 projection under the C10 family definition.
    """

    raise ACFamilyContractError(
        "schema-v2 245/353 pre-contact guide-mask ABI is SUPERSEDED and "
        "checkpoint-incompatible; use build_c10_family_projection with "
        "unfrozen widths and post-contact placement treatment"
    )


def validate_ac_family_pair(
    a_projection: ResolvedACProjection,
    c_projection: ResolvedACProjection,
) -> ACFamilyPairValidation:
    """Reject schema-v2 projections after the C10 contract replacement.

    The legacy name remains only to turn stale integrations into an explicit
    incompatibility rather than a false matched-pair PASS.
    """

    raise ACFamilyContractError(
        "schema-v2 245/353 pair validation is SUPERSEDED; use "
        "validate_c10_family_pair"
    )


def build_c07_family_projection(*args: object, **kwargs: object) -> None:
    """Named tombstone for the superseded 245/353 C07 family design."""

    del args, kwargs
    raise ACFamilyContractError(
        "C07 245/353 pre-contact guide-mask family contract is SUPERSEDED; "
        "fresh replacement authority is C10 and remains launch-HOLD"
    )


def validate_c07_family_pair(*args: object, **kwargs: object) -> None:
    """Named tombstone for stale C07 matched-pair callers."""

    del args, kwargs
    raise ACFamilyContractError(
        "C07 pair admission is SUPERSEDED; use validate_c10_family_pair"
    )


def projection_artifact(projection: ResolvedACProjection) -> dict[str, object]:
    """Serialize a builder-owned projection for audit, never for re-admission."""

    payload = _owned_projection(projection, label="artifact")
    return {
        "canonical_sha256": canonical_sha256(payload),
        "projection": payload,
    }


def _resolve_observation_group(
    group: ResolvedObservationGroup,
    *,
    expected: Sequence[FieldSpec],
    role: str,
    identity: Mapping[str, str],
) -> dict[str, object]:
    if type(group) is not ResolvedObservationGroup or type(group.terms) is not tuple:
        raise ACFamilyContractError("%s must be a ResolvedObservationGroup" % role)
    if len(group.terms) != len(expected):
        raise ACFamilyContractError("%s resolved term count differs" % role)
    bindings = []
    offset = 0
    for index, (term, spec) in enumerate(zip(group.terms, expected)):
        if type(term) is not ResolvedObservationTerm:
            raise ACFamilyContractError("%s term %d has wrong type" % (role, index))
        if term.name != spec.name or type(term.width) is not int or term.width != spec.width:
            raise ACFamilyContractError(
                "%s term %d name/width differs: expected=(%s,%d) actual=(%r,%r)"
                % (role, index, spec.name, spec.width, term.name, term.width)
            )
        binding = _resolve_callable(
            term.producer,
            identity=identity,
            path="%s.terms[%d].producer" % (role, index),
        )
        bindings.append(
            {
                "binding": binding,
                "portable": spec.to_mapping(offset),
            }
        )
        offset += spec.width
    expected_width = ACTOR_WIDTH if role == "actor" else CRITIC_WIDTH
    if offset != expected_width:
        raise ACFamilyContractError("%s resolved width differs" % role)
    return {
        "ordered_terms": bindings,
        "resolved_order_sha256": canonical_sha256(
            [(term.name, term.width) for term in group.terms]
        ),
        "width": offset,
    }


def _resolve_common_callables(
    value: CommonRuntimeCallables,
    *,
    identity: Mapping[str, str],
) -> dict[str, object]:
    if type(value) is not CommonRuntimeCallables:
        raise ACFamilyContractError("common must be CommonRuntimeCallables")
    actual_names = tuple(field.name for field in dataclass_fields(value))
    if actual_names != _COMMON_COMPONENT_NAMES:
        raise ACFamilyContractError("common callable schema differs")
    result = {}
    for name in _COMMON_COMPONENT_NAMES:
        item = getattr(value, name)
        if item.input_schema_id != _COMMON_INPUT_SCHEMAS[name]:
            raise ACFamilyContractError(
                "common.%s input schema differs" % name
            )
        result[name] = _resolve_callable(
            item,
            identity=identity,
            path="common.%s" % name,
        )
    return result


def _resolve_callable(
    value: ResolvedCallableInput,
    *,
    identity: Mapping[str, str],
    path: str,
) -> dict[str, object]:
    if type(value) is not ResolvedCallableInput:
        raise ACFamilyContractError("%s must be ResolvedCallableInput" % path)
    function = value.function
    if not isinstance(function, types.FunctionType) or function.__name__ == "<lambda>":
        raise ACFamilyContractError(
            "%s must resolve to a plain named Python function" % path
        )
    input_schema = _nonempty_text(value.input_schema_id, path=path + ".input_schema_id")
    _reject_forbidden_text(input_schema, path=path + ".input_schema_id")
    bound = _canonicalize(value.bound_params, path=path + ".bound_params", ancestors=set())
    if not isinstance(bound, Mapping):
        raise ACFamilyContractError("%s.bound_params must be a mapping" % path)
    _reject_runtime_identity(bound, identity=identity, path=path + ".bound_params")
    dependencies = _function_dependency_closure(
        function,
        explicit=value.dependency_functions,
        identity=identity,
        path=path,
    )
    sources = []
    global_inputs = []
    for dependency in dependencies:
        sources.append(_scanned_function_source(dependency, path=path))
        global_inputs.append(
            _callable_static_input_receipt(
                dependency,
                identity=identity,
                path=path,
            )
        )
    signature = inspect.signature(function)
    for parameter in signature.parameters.values():
        if _normalized_token(parameter.name) in _FORBIDDEN_RUNTIME_IDENTIFIERS:
            raise ACFamilyContractError(
                "%s callable signature contains forbidden identity input %r"
                % (path, parameter.name)
            )
    return {
        "bound_params_sha256": canonical_sha256(bound),
        "dependency_closure_sha256": canonical_sha256(sources),
        "static_input_closure_sha256": canonical_sha256(global_inputs),
        "input_schema_id": input_schema,
        "signature": str(signature),
        "symbol": "%s.%s" % (function.__module__, function.__qualname__),
    }


def _callable_static_input_receipt(
    function: types.FunctionType,
    *,
    identity: Mapping[str, str],
    path: str,
) -> dict[str, object]:
    """Hash defaults and referenced primitive globals, not source alone."""

    defaults = _canonicalize(
        {
            "keyword": function.__kwdefaults__ or {},
            "positional": function.__defaults__ or (),
        },
        path=path + ".callable_defaults",
        ancestors=set(),
    )
    _reject_runtime_identity(
        defaults,
        identity=identity,
        path=path + ".callable_defaults",
    )
    primitives = {}
    for global_name in sorted(set(function.__code__.co_names)):
        if global_name not in function.__globals__:
            continue
        global_value = function.__globals__[global_name]
        if global_value is None or type(global_value) in (bool, int, float, str):
            primitive = _canonicalize(
                global_value,
                path=path + ".global_inputs.%s" % global_name,
                ancestors=set(),
            )
            _reject_runtime_identity(
                {global_name: primitive},
                identity=identity,
                path=path + ".global_inputs.%s" % global_name,
            )
            primitives[global_name] = primitive
    return {
        "defaults_sha256": canonical_sha256(defaults),
        "primitive_globals_sha256": canonical_sha256(primitives),
        "symbol": "%s.%s" % (function.__module__, function.__qualname__),
    }


def _function_dependency_closure(
    function: types.FunctionType,
    *,
    explicit: Tuple[Callable[..., object], ...],
    identity: Mapping[str, str],
    path: str,
) -> Tuple[types.FunctionType, ...]:
    """Resolve plain-function globals instead of trusting a declared closure."""

    if type(explicit) is not tuple:
        raise ACFamilyContractError("%s dependency_functions must be an exact tuple" % path)
    pending = [function]
    for dependency in explicit:
        if not isinstance(dependency, types.FunctionType):
            raise ACFamilyContractError(
                "%s dependency closure must contain plain functions" % path
            )
        pending.append(dependency)
    resolved = []
    seen = set()
    while pending:
        dependency = pending.pop(0)
        marker = id(dependency)
        if marker in seen:
            continue
        seen.add(marker)
        if dependency.__closure__:
            raise ACFamilyContractError(
                "%s callable closure cells are forbidden; bind typed values explicitly"
                % path
            )
        defaults = {
            "positional": dependency.__defaults__ or (),
            "keyword": dependency.__kwdefaults__ or {},
        }
        defaults_value = _canonicalize(
            defaults,
            path=path + ".callable_defaults",
            ancestors=set(),
        )
        _reject_runtime_identity(
            defaults_value,
            identity=identity,
            path=path + ".callable_defaults",
        )
        for global_name in dependency.__code__.co_names:
            if _normalized_token(global_name) in _FORBIDDEN_RUNTIME_IDENTIFIERS:
                raise ACFamilyContractError(
                    "%s callable reads forbidden global %r" % (path, global_name)
                )
            global_value = dependency.__globals__.get(global_name)
            if isinstance(global_value, types.FunctionType):
                pending.append(global_value)
            elif isinstance(global_value, types.ModuleType):
                module_root = global_value.__name__.split(".", 1)[0]
                if module_root not in _SAFE_GLOBAL_MODULE_ROOTS:
                    raise ACFamilyContractError(
                        "%s callable reads unsupported global module %r"
                        % (path, global_value.__name__)
                    )
            elif global_value is None or type(global_value) in (bool, int, float, str):
                primitive = _canonicalize(
                    {global_name: global_value},
                    path=path + ".global_inputs",
                    ancestors=set(),
                )
                _reject_runtime_identity(
                    primitive,
                    identity=identity,
                    path=path + ".global_inputs",
                )
            elif global_name in dependency.__globals__:
                raise ACFamilyContractError(
                    "%s callable reads unsupported global object %r"
                    % (path, global_name)
                )
        resolved.append(dependency)
    return tuple(sorted(resolved, key=lambda item: (item.__module__, item.__qualname__)))


def _scanned_function_source(
    function: Callable[..., object], *, path: str
) -> dict[str, object]:
    try:
        source_path = inspect.getsourcefile(function)
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise ACFamilyContractError(
            "%s callable source is unavailable" % path
        ) from exc
    if source_path is None:
        raise ACFamilyContractError("%s callable source path is unavailable" % path)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ACFamilyContractError("%s callable source cannot be parsed" % path) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            raise ACFamilyContractError(
                "%s callable source uses dynamic string construction" % path
            )
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and _ast_contains_string_literal(node)
            and _ast_static_string(node) is None
        ):
            raise ACFamilyContractError(
                "%s callable source uses non-constant string-key construction" % path
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("format", "format_map", "join")
        ):
            raise ACFamilyContractError(
                "%s callable source uses dynamic string-key construction" % path
            )
        token = None
        if isinstance(node, ast.Name):
            token = node.id
        elif isinstance(node, ast.Attribute):
            token = node.attr
        elif isinstance(node, ast.keyword) and node.arg is not None:
            token = node.arg
        elif isinstance(node, ast.Constant) and type(node.value) is str:
            if _normalized_token(node.value) in _FORBIDDEN_RUNTIME_IDENTIFIERS:
                raise ACFamilyContractError(
                    "%s callable source contains forbidden runtime identity key %r"
                    % (path, node.value)
                )
        static_text = _ast_static_string(node)
        if (
            static_text is not None
            and _normalized_token(static_text) in _FORBIDDEN_RUNTIME_IDENTIFIERS
        ):
            raise ACFamilyContractError(
                "%s callable source constructs forbidden runtime identity %r"
                % (path, static_text)
            )
        if token is not None and _normalized_token(token) in _FORBIDDEN_RUNTIME_IDENTIFIERS:
            raise ACFamilyContractError(
                "%s callable source reads forbidden runtime identity %r"
                % (path, token)
            )
        if token is not None and _normalized_token(token) in _FORBIDDEN_DYNAMIC_INTROSPECTION:
            raise ACFamilyContractError(
                "%s callable source uses forbidden dynamic introspection %r"
                % (path, token)
            )
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if type(node.slice.value) is str and _normalized_token(node.slice.value) in _FORBIDDEN_RUNTIME_IDENTIFIERS:
                raise ACFamilyContractError(
                    "%s callable source indexes forbidden runtime identity %r"
                    % (path, node.slice.value)
                )
    file_bytes = Path(source_path).read_bytes()
    return {
        "file_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "function_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "symbol": "%s.%s" % (function.__module__, function.__qualname__),
    }


def _ast_static_string(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _ast_static_string(node.left)
        right = _ast_static_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _ast_contains_string_literal(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant) and type(child.value) is str
        for child in ast.walk(node)
    )


def _fresh_normalizer_witness(value: FreshNormalizerState) -> dict[str, object]:
    if type(value) is not FreshNormalizerState:
        raise ACFamilyContractError(
            "initial_normalizer_state must be FreshNormalizerState"
        )
    for name in ("actor_sample_count", "critic_sample_count"):
        count = getattr(value, name)
        if type(count) is not int or count != 0:
            raise ACFamilyContractError("fresh normalizer %s must be exact zero" % name)
    blocks = {}
    for name, width in (
        ("actor_mean_f32", ACTOR_WIDTH),
        ("actor_m2_f32", ACTOR_WIDTH),
        ("critic_mean_f32", CRITIC_WIDTH),
        ("critic_m2_f32", CRITIC_WIDTH),
    ):
        block = _nonempty_bytes(getattr(value, name), path="normalizer.%s" % name)
        if len(block) != width * FLOAT32_BYTES:
            raise ACFamilyContractError(
                "normalizer.%s must contain exactly %d float32 scalars"
                % (name, width)
            )
        if not _all_positive_zero(block):
            raise ACFamilyContractError(
                "fresh normalizer %s must contain positive-zero bytes" % name
            )
        blocks[name + "_sha256"] = hashlib.sha256(block).hexdigest()
    epsilon = _finite_scalar(value.epsilon, path="normalizer.epsilon")
    clip = _finite_scalar(value.clip, path="normalizer.clip")
    if epsilon <= 0.0 or clip <= 0.0:
        raise ACFamilyContractError("normalizer epsilon/clip must be positive")
    payload = {
        "actor_sample_count": 0,
        "critic_sample_count": 0,
        "algorithm": "fresh_masked_welford_identity_v1",
        "clip": clip,
        "epsilon": epsilon,
        **blocks,
    }
    return {
        **payload,
        "canonical_sha256": canonical_sha256(payload),
    }


def _fresh_checkpoint_witness(value: FreshCheckpointState) -> dict[str, object]:
    if type(value) is not FreshCheckpointState:
        raise ACFamilyContractError(
            "fresh_checkpoint_state must be FreshCheckpointState"
        )
    if value.parent_checkpoint_sha256 is not None:
        raise ACFamilyContractError("fresh A/C lineage cannot have a checkpoint parent")
    if type(value.resume_requested) is not bool or value.resume_requested:
        raise ACFamilyContractError("fresh A/C lineage cannot request resume")
    payload = {
        "kind": "fresh_common_initial_policy_optimizer_rng_rollout_v1",
        "parent_checkpoint_sha256": None,
        "resume_requested": False,
    }
    for name in (
        "policy_state_bytes",
        "optimizer_state_bytes",
        "rng_state_bytes",
        "rollout_state_bytes",
    ):
        block = _nonempty_bytes(getattr(value, name), path="checkpoint.%s" % name)
        payload[name.replace("_bytes", "_sha256")] = hashlib.sha256(block).hexdigest()
    return {
        **payload,
        "canonical_sha256": canonical_sha256(payload),
    }


def _fixed_tape_witness(
    value: FixedTapeBytes,
    *,
    family: str,
    cadence_ticks: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(value) is not FixedTapeBytes:
        raise ACFamilyContractError("fixed_tape must be FixedTapeBytes")
    tape = _nonempty_bytes(value.tape_bytes, path="fixed_tape.tape_bytes")
    actor_raw = _point_float32_blocks(
        value.actor_raw_pre_treatment,
        scalar_count=ACTOR_WIDTH,
        path="fixed_tape.actor_raw_pre_treatment",
    )
    critic_raw = _point_float32_blocks(
        value.critic_raw_pre_treatment,
        scalar_count=CRITIC_WIDTH,
        path="fixed_tape.critic_raw_pre_treatment",
    )
    actor_normalized = _point_float32_blocks(
        value.actor_normalized_pre_treatment,
        scalar_count=ACTOR_WIDTH,
        path="fixed_tape.actor_normalized_pre_treatment",
    )
    critic_normalized = _point_float32_blocks(
        value.critic_normalized_pre_treatment,
        scalar_count=CRITIC_WIDTH,
        path="fixed_tape.critic_normalized_pre_treatment",
    )
    actor_final = _point_float32_blocks(
        value.actor_final,
        scalar_count=ACTOR_WIDTH,
        path="fixed_tape.actor_final",
    )
    critic_final = _point_float32_blocks(
        value.critic_final,
        scalar_count=CRITIC_WIDTH,
        path="fixed_tape.critic_final",
    )
    common_reward = _point_float32_blocks(
        value.common_reward,
        scalar_count=1,
        path="fixed_tape.common_reward",
    )
    guide_reward = _point_float32_blocks(
        value.guide_reward,
        scalar_count=1,
        path="fixed_tape.guide_reward",
    )
    termination = _point_binary_blocks(
        value.termination,
        path="fixed_tape.termination",
    )
    lifecycle_phase = _point_phase_blocks(value.lifecycle_phase)
    call_input_trace = _point_exact_blocks(
        value.call_input_trace,
        byte_count=hashlib.sha256().digest_size,
        path="fixed_tape.call_input_trace",
    )
    outcome_atomicity = _outcome_atomicity_witness(
        value.outcome_atomicity,
        cadence_ticks=cadence_ticks,
    )
    if actor_normalized != actor_raw or critic_normalized != critic_raw:
        raise ACFamilyContractError(
            "fresh zero-count normalizer must be identity on the fixed tape"
        )
    raw_guide = []
    actor_treated = []
    critic_treated = []
    raw_guide_valid = []
    final_guide_valid = []
    valid_normalized_nonzero = False
    guide_reward_nonzero = False
    for point_index in range(len(FIXED_TAPE_POINTS)):
        phase_index = point_index // len(FIXED_TAPE_CONTEXTS)
        _validate_raw_point(
            actor_raw[point_index],
            critic_raw[point_index],
            actor_normalized[point_index],
            critic_normalized[point_index],
            phase_index=phase_index,
            point=FIXED_TAPE_POINTS[point_index],
        )
        expected_actor_final = _expected_final_vector(
            actor_normalized[point_index],
            family=family,
            guide_slice=ACTOR_GUIDE_SLICE,
            guide_valid_index=ACTOR_GUIDE_VALID_INDEX,
        )
        expected_critic_final = _expected_final_vector(
            critic_normalized[point_index],
            family=family,
            guide_slice=CRITIC_GUIDE_SLICE,
            guide_valid_index=CRITIC_GUIDE_VALID_INDEX,
        )
        if actor_final[point_index] != expected_actor_final:
            raise ACFamilyContractError(
                "fixed_tape.actor_final[%d] changes values outside the resolved guide treatment"
                % point_index
            )
        if critic_final[point_index] != expected_critic_final:
            raise ACFamilyContractError(
                "fixed_tape.critic_final[%d] changes values outside the resolved guide treatment"
                % point_index
            )
        actor_raw_guide = _float32_slice(actor_raw[point_index], ACTOR_GUIDE_SLICE)
        critic_raw_guide = _float32_slice(critic_raw[point_index], CRITIC_GUIDE_SLICE)
        if actor_raw_guide != critic_raw_guide:
            raise ACFamilyContractError("actor/critic raw guide sidebands differ")
        raw_guide.append(actor_raw_guide)
        actor_treated.append(_float32_slice(actor_final[point_index], ACTOR_GUIDE_SLICE))
        critic_treated.append(_float32_slice(critic_final[point_index], CRITIC_GUIDE_SLICE))
        raw_valid = _float32_slice(
            actor_raw[point_index],
            (ACTOR_GUIDE_VALID_INDEX, ACTOR_GUIDE_VALID_INDEX + 1),
        )
        raw_guide_valid.append(raw_valid)
        final_valid = _float32_slice(
            actor_final[point_index],
            (ACTOR_GUIDE_VALID_INDEX, ACTOR_GUIDE_VALID_INDEX + 1),
        )
        final_guide_valid.append(final_valid)
        if phase_index == 1:
            normalized_guide = _float32_slice(
                actor_normalized[point_index], ACTOR_GUIDE_SLICE
            )
            valid_normalized_nonzero |= not _all_positive_zero(normalized_guide)
            guide_reward_nonzero |= not _all_positive_zero(guide_reward[point_index])

    if not valid_normalized_nonzero:
        raise ACFamilyContractError(
            "fixed tape never witnesses a nonzero normalized guide in swing_pre_contact"
        )
    if family == "A":
        if not guide_reward_nonzero:
            raise ACFamilyContractError(
                "A fixed tape never witnesses a nonzero guide reward"
            )
        for point_index, reward in enumerate(guide_reward):
            phase_index = point_index // len(FIXED_TAPE_CONTEXTS)
            if phase_index != 1 and not _all_positive_zero(reward):
                raise ACFamilyContractError(
                    "A guide reward must be zero outside swing_pre_contact"
                )
    elif any(not _all_positive_zero(block) for block in guide_reward):
        raise ACFamilyContractError("C guide reward must be positive-zero at every tape point")
    for phase_index in range(len(FIXED_TAPE_PHASES)):
        start = phase_index * len(FIXED_TAPE_CONTEXTS)
        stop = start + len(FIXED_TAPE_CONTEXTS)
        for name, blocks in (
            ("actor_raw_pre_treatment", actor_raw),
            ("critic_raw_pre_treatment", critic_raw),
            ("actor_normalized_pre_treatment", actor_normalized),
            ("critic_normalized_pre_treatment", critic_normalized),
            ("actor_final", actor_final),
            ("critic_final", critic_final),
            ("common_reward", common_reward),
            ("guide_reward", guide_reward),
        ):
            if len(frozenset(blocks[start:stop])) != 1:
                raise ACFamilyContractError(
                    "fixed_tape.%s differs across assembly paths for phase %s"
                    % (name, FIXED_TAPE_PHASES[phase_index])
                )

    common = {
        "actor_normalized_pre_treatment_sha256": _point_bytes_sha(actor_normalized),
        "actor_raw_pre_treatment_sha256": _point_bytes_sha(actor_raw),
        "call_input_trace_sha256": _point_bytes_sha(call_input_trace),
        "common_reward_sha256": _point_bytes_sha(common_reward),
        "critic_normalized_pre_treatment_sha256": _point_bytes_sha(critic_normalized),
        "critic_raw_pre_treatment_sha256": _point_bytes_sha(critic_raw),
        "lifecycle_phase_sha256": _point_bytes_sha(lifecycle_phase),
        "outcome_atomicity": outcome_atomicity,
        "phases": list(FIXED_TAPE_PHASES),
        "contexts": list(FIXED_TAPE_CONTEXTS),
        "points": list(FIXED_TAPE_POINTS),
        "raw_guide_sha256": _point_bytes_sha(tuple(raw_guide)),
        "raw_guide_valid_sha256": _point_bytes_sha(tuple(raw_guide_valid)),
        "tape_sha256": hashlib.sha256(tape).hexdigest(),
        "termination_sha256": _point_bytes_sha(termination),
    }
    valid_count = sum(not _all_positive_zero(block) for block in final_guide_valid)
    treatment = {
        "actor_treated_guide_sha256": _point_bytes_sha(tuple(actor_treated)),
        "critic_treated_guide_sha256": _point_bytes_sha(tuple(critic_treated)),
        "guide_reward_all_positive_zero": all(
            _all_positive_zero(block) for block in guide_reward
        ),
        "guide_reward_sha256": _point_bytes_sha(guide_reward),
        "guide_valid_count": valid_count,
        "guide_valid_size": len(final_guide_valid),
        "treated_all_positive_zero": all(
            _all_positive_zero(block)
            for block in tuple(actor_treated) + tuple(critic_treated)
        ),
        "treated_equals_normalized": family == "A",
    }
    return common, treatment


def _outcome_atomicity_witness(
    value: OutcomeAtomicityWitness,
    *,
    cadence_ticks: int,
) -> dict[str, object]:
    if type(value) is not OutcomeAtomicityWitness:
        raise ACFamilyContractError(
            "fixed_tape.outcome_atomicity must be OutcomeAtomicityWitness"
        )
    tick_names = (
        "previous_reveal_tick",
        "open_observation_tick",
        "settlement_tick",
        "payment_tick",
        "close_tick",
        "closed_observation_tick",
        "next_reveal_tick",
        "next_open_tick",
    )
    ticks = {}
    for name in tick_names:
        tick = getattr(value, name)
        if type(tick) is not int or tick < 0:
            raise ACFamilyContractError(
                "outcome atomicity %s must be a nonnegative exact int" % name
            )
        ticks[name] = tick
    if value.observed_mailbox_states != ("OPEN", "CLOSED", "OPEN"):
        raise ACFamilyContractError(
            "observation boundaries must expose OPEN, CLOSED, then next OPEN"
        )
    expected_events = (
        "OPEN_OBSERVATION",
        "SETTLE",
        "PAY",
        "CLOSE",
        "CLOSED_OBSERVATION",
        "NEXT_REVEAL",
        "NEXT_OPEN",
    )
    if value.ordered_events != expected_events:
        raise ACFamilyContractError(
            "mailbox events must prove close-before-next-open ordering"
        )
    if ticks["next_reveal_tick"] != ticks["previous_reveal_tick"] + cadence_ticks:
        raise ACFamilyContractError("next reveal does not follow frozen cadence")
    if not (
        ticks["previous_reveal_tick"]
        <= ticks["open_observation_tick"]
        < ticks["settlement_tick"]
    ):
        raise ACFamilyContractError("outcome open/settlement ordering differs")
    if not (
        ticks["settlement_tick"]
        == ticks["payment_tick"]
        == ticks["close_tick"]
    ):
        raise ACFamilyContractError(
            "settlement, payment, and close must be atomic within one environment tick"
        )
    if not (
        ticks["close_tick"]
        <= ticks["closed_observation_tick"]
        < ticks["next_reveal_tick"]
    ):
        raise ACFamilyContractError(
            "closed observation must precede the immutable next reveal"
        )
    if ticks["next_open_tick"] != ticks["next_reveal_tick"]:
        raise ACFamilyContractError("next mailbox open must use the reveal tick")
    if ticks["close_tick"] > ticks["next_open_tick"]:
        raise ACFamilyContractError("mailbox reuse occurs before close")
    payload = {
        **ticks,
        "ordered_events": list(value.ordered_events),
        "observed_mailbox_states": list(value.observed_mailbox_states),
        "settled_unpaid_observation_forbidden": True,
    }
    return {
        **payload,
        "canonical_sha256": canonical_sha256(payload),
    }


def _validate_raw_point(
    actor_raw: bytes,
    critic_raw: bytes,
    actor_normalized: bytes,
    critic_normalized: bytes,
    *,
    phase_index: int,
    point: str,
) -> None:
    actor_suffix = _float32_slice(actor_raw, (197, ACTOR_WIDTH))
    critic_suffix = _float32_slice(critic_raw, (305, CRITIC_WIDTH))
    if actor_suffix != critic_suffix:
        raise ACFamilyContractError("%s actor/critic common raw suffix differs" % point)
    checks = (
        (ACTOR_PHASE_SLICE, CRITIC_PHASE_SLICE, _one_hot_float32(phase_index, 5), "phase"),
        ((ACTOR_TASK_VALID_INDEX, ACTOR_TASK_VALID_INDEX + 1), (CRITIC_TASK_VALID_INDEX, CRITIC_TASK_VALID_INDEX + 1), _binary_float32(_TASK_VALID_BY_PHASE[phase_index]), "task_valid"),
        ((ACTOR_BALL_VALID_INDEX, ACTOR_BALL_VALID_INDEX + 1), (CRITIC_BALL_VALID_INDEX, CRITIC_BALL_VALID_INDEX + 1), _binary_float32(_BALL_VALID_BY_PHASE[phase_index]), "ball_valid"),
        ((ACTOR_GUIDE_VALID_INDEX, ACTOR_GUIDE_VALID_INDEX + 1), (CRITIC_GUIDE_VALID_INDEX, CRITIC_GUIDE_VALID_INDEX + 1), _binary_float32(_RAW_GUIDE_VALID_BY_PHASE[phase_index]), "raw_guide_valid"),
        ((ACTOR_CURRENT_OUTCOME_PENDING_INDEX, ACTOR_CURRENT_OUTCOME_PENDING_INDEX + 1), (CRITIC_CURRENT_OUTCOME_PENDING_INDEX, CRITIC_CURRENT_OUTCOME_PENDING_INDEX + 1), _binary_float32(_OUTCOME_PENDING_BY_PHASE[phase_index]), "outcome_pending"),
        ((ACTOR_NET_CROSSED_INDEX, ACTOR_NET_CROSSED_INDEX + 1), (CRITIC_NET_CROSSED_INDEX, CRITIC_NET_CROSSED_INDEX + 1), _binary_float32(_NET_CROSSED_BY_PHASE[phase_index]), "net_crossed"),
        ((ACTOR_NET_CLEAR_INDEX, ACTOR_NET_CLEAR_INDEX + 1), (CRITIC_NET_CLEAR_INDEX, CRITIC_NET_CLEAR_INDEX + 1), _binary_float32(_NET_CLEAR_BY_PHASE[phase_index]), "net_clear"),
        ((ACTOR_PREVIOUS_SHOT_VALID_INDEX, ACTOR_PREVIOUS_SHOT_VALID_INDEX + 1), (CRITIC_PREVIOUS_SHOT_VALID_INDEX, CRITIC_PREVIOUS_SHOT_VALID_INDEX + 1), _binary_float32(_PREVIOUS_SHOT_VALID_BY_PHASE[phase_index]), "previous_shot_valid"),
    )
    for actor_slice, critic_slice, expected, label in checks:
        if _float32_slice(actor_raw, actor_slice) != expected:
            raise ACFamilyContractError("%s actor raw %s has wrong causal value" % (point, label))
        if _float32_slice(critic_raw, critic_slice) != expected:
            raise ACFamilyContractError("%s critic raw %s has wrong causal value" % (point, label))
        if _float32_slice(actor_normalized, actor_slice) != expected:
            raise ACFamilyContractError("%s actor normalized %s is not passthrough" % (point, label))
        if _float32_slice(critic_normalized, critic_slice) != expected:
            raise ACFamilyContractError("%s critic normalized %s is not passthrough" % (point, label))
    _validate_validity_masks(
        actor_raw,
        actor_normalized,
        layout=ACTOR_LAYOUT,
        point=point,
        role="actor",
    )
    _validate_validity_masks(
        critic_raw,
        critic_normalized,
        layout=CRITIC_LAYOUT,
        point=point,
        role="critic",
    )
    if not _TASK_VALID_BY_PHASE[phase_index]:
        for vector, target_slice, label in (
            (actor_raw, ACTOR_TARGET_SLICE, "actor raw"),
            (critic_raw, CRITIC_TARGET_SLICE, "critic raw"),
            (actor_normalized, ACTOR_TARGET_SLICE, "actor normalized"),
            (critic_normalized, CRITIC_TARGET_SLICE, "critic normalized"),
        ):
            if not _all_positive_zero(_float32_slice(vector, target_slice)):
                raise ACFamilyContractError(
                    "%s reveals %s target before/after active task" % (point, label)
                )
    if not _BALL_VALID_BY_PHASE[phase_index]:
        for vector, ball_slice, label in (
            (actor_raw, (197, 206), "actor raw"),
            (critic_raw, (305, 314), "critic raw"),
            (actor_normalized, (197, 206), "actor normalized"),
            (critic_normalized, (305, 314), "critic normalized"),
        ):
            if not _all_positive_zero(_float32_slice(vector, ball_slice)):
                raise ACFamilyContractError(
                    "%s reveals %s ball while invalid" % (point, label)
                )
    if not _RAW_GUIDE_VALID_BY_PHASE[phase_index]:
        for vector, guide_slice, label in (
            (actor_raw, ACTOR_GUIDE_SLICE, "actor raw"),
            (critic_raw, CRITIC_GUIDE_SLICE, "critic raw"),
            (actor_normalized, ACTOR_GUIDE_SLICE, "actor normalized"),
            (critic_normalized, CRITIC_GUIDE_SLICE, "critic normalized"),
        ):
            if not _all_positive_zero(_float32_slice(vector, guide_slice)):
                raise ACFamilyContractError(
                    "%s reveals %s guide outside pre-contact" % (point, label)
                )


def _validate_validity_masks(
    raw: bytes,
    normalized: bytes,
    *,
    layout: Sequence[FieldSpec],
    point: str,
    role: str,
) -> None:
    offsets = {}
    offset = 0
    for spec in layout:
        offsets[spec.name] = (offset, offset + spec.width)
        offset += spec.width
    for spec in layout:
        if spec.validity == "always":
            continue
        validity_slice = offsets.get(spec.validity)
        if validity_slice is None or validity_slice[1] - validity_slice[0] != 1:
            raise ACFamilyContractError(
                "%s validity source %r is not a scalar ABI field"
                % (spec.name, spec.validity)
            )
        raw_validity = _float32_slice(raw, validity_slice)
        if raw_validity not in (_binary_float32(0), _binary_float32(1)):
            raise ACFamilyContractError(
                "%s %s validity %s is not binary" % (point, role, spec.validity)
            )
        if _float32_slice(normalized, validity_slice) != raw_validity:
            raise ACFamilyContractError(
                "%s %s validity %s is not passthrough"
                % (point, role, spec.validity)
            )
        if raw_validity == _binary_float32(0):
            field_slice = offsets[spec.name]
            if not _all_positive_zero(_float32_slice(raw, field_slice)):
                raise ACFamilyContractError(
                    "%s %s invalid field %s is not positive-zero"
                    % (point, role, spec.name)
                )
            if not _all_positive_zero(_float32_slice(normalized, field_slice)):
                raise ACFamilyContractError(
                    "%s %s normalized invalid field %s is not positive-zero"
                    % (point, role, spec.name)
                )


def _expected_final_vector(
    normalized: bytes,
    *,
    family: str,
    guide_slice: Tuple[int, int],
    guide_valid_index: int,
) -> bytes:
    if family == "A":
        return normalized
    output = bytearray(normalized)
    output[guide_slice[0] * FLOAT32_BYTES : guide_slice[1] * FLOAT32_BYTES] = bytes(
        (guide_slice[1] - guide_slice[0]) * FLOAT32_BYTES
    )
    output[guide_valid_index * FLOAT32_BYTES : (guide_valid_index + 1) * FLOAT32_BYTES] = bytes(FLOAT32_BYTES)
    return bytes(output)


def _point_float32_blocks(
    value: Sequence[bytes],
    *,
    scalar_count: int,
    path: str,
) -> Tuple[bytes, ...]:
    result = _point_exact_blocks(
        value,
        byte_count=scalar_count * FLOAT32_BYTES,
        path=path,
    )
    for index, block in enumerate(result):
        if any(not math.isfinite(item[0]) for item in struct.iter_unpack("<f", block)):
            raise ACFamilyContractError("%s[%d] contains nonfinite float32" % (path, index))
    return result


def _point_binary_blocks(value: Sequence[bytes], *, path: str) -> Tuple[bytes, ...]:
    result = _point_exact_blocks(value, byte_count=1, path=path)
    if any(block not in (b"\x00", b"\x01") for block in result):
        raise ACFamilyContractError("%s must contain exact uint8 boolean rows" % path)
    return result


def _point_phase_blocks(value: Sequence[bytes]) -> Tuple[bytes, ...]:
    result = _point_exact_blocks(value, byte_count=1, path="fixed_tape.lifecycle_phase")
    for point_index, block in enumerate(result):
        expected = point_index // len(FIXED_TAPE_CONTEXTS)
        if block != bytes((expected,)):
            raise ACFamilyContractError(
                "fixed_tape.lifecycle_phase[%d] does not match ordered phase" % point_index
            )
    return result


def _point_exact_blocks(
    value: Sequence[bytes],
    *,
    byte_count: int,
    path: str,
) -> Tuple[bytes, ...]:
    if type(value) is not tuple or len(value) != len(FIXED_TAPE_POINTS):
        raise ACFamilyContractError(
            "%s must contain exactly %d ordered fixed-tape points"
            % (path, len(FIXED_TAPE_POINTS))
        )
    result = []
    for index, item in enumerate(value):
        block = _nonempty_bytes(item, path="%s[%d]" % (path, index))
        if len(block) != byte_count:
            raise ACFamilyContractError(
                "%s[%d] must contain exactly %d bytes" % (path, index, byte_count)
            )
        result.append(block)
    return tuple(result)


def _point_bytes_sha(value: Sequence[bytes]) -> str:
    return hashlib.sha256(_length_prefixed(value)).hexdigest()


def _float32_slice(value: bytes, scalar_slice: Tuple[int, int]) -> bytes:
    return value[
        scalar_slice[0] * FLOAT32_BYTES : scalar_slice[1] * FLOAT32_BYTES
    ]


def _binary_float32(value: int) -> bytes:
    return struct.pack("<f", float(value))


def _one_hot_float32(index: int, width: int) -> bytes:
    return b"".join(_binary_float32(position == index) for position in range(width))


def _all_positive_zero(value: bytes) -> bool:
    return value == bytes(len(value))


def _length_prefixed(values: Sequence[bytes]) -> bytes:
    output = bytearray()
    for value in values:
        output.extend(len(value).to_bytes(8, byteorder="big", signed=False))
        output.extend(value)
    return bytes(output)


def _treatment(*, family: str, enabled: object, weight: object) -> dict[str, object]:
    if type(enabled) is not bool:
        raise ACFamilyContractError("guide_enabled must be an exact bool")
    resolved_weight = _finite_scalar(weight, path="guide_reward_weight")
    if family == "A":
        if not enabled or resolved_weight <= 0.0:
            raise ACFamilyContractError("family A guide must be enabled with positive weight")
    elif enabled or resolved_weight != 0.0:
        raise ACFamilyContractError("family C guide must be disabled with zero weight")
    return {
        "guide_enabled": enabled,
        "guide_reward_weight": resolved_weight,
    }


def _owned_projection(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not ResolvedACProjection or value._token is not _BUILD_TOKEN:
        raise ACFamilyContractError(
            "%s projection must come from build_resolved_runtime_projection" % label
        )
    expected_auth_tag = hmac.new(
        _BUILD_AUTH_KEY,
        value._payload_json,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(value._auth_tag, expected_auth_tag):
        raise ACFamilyContractError(
            "%s projection builder authentication failed" % label
        )
    payload = value.to_mapping()
    expected = frozenset(
        {
            "backend",
            "backend_binding_sha256",
            "common_runtime",
            "family",
            "identity",
            "kind",
            "schema_version",
            "treatment",
            "treatment_witness",
        }
    )
    if frozenset(payload) != expected:
        raise ACFamilyContractError("builder-owned projection schema drifted")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ACFamilyContractError("builder-owned projection version drifted")
    if payload["kind"] != "action_ball_matched_ac_resolved_runtime_projection_v2":
        raise ACFamilyContractError("builder-owned projection kind drifted")
    if payload["common_runtime"]["portable_abi_sha256"] != PORTABLE_ABI_SHA256:  # type: ignore[index]
        raise ACFamilyContractError("builder-owned portable ABI drifted")
    return payload


def _identity(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ACFamilyContractError("identity must be a mapping")
    unknown = frozenset(value) - frozenset(IDENTITY_ONLY_FIELDS)
    if unknown:
        raise ACFamilyContractError("identity has unknown fields %r" % sorted(unknown))
    result = {}
    for key in sorted(value):
        result[key] = _nonempty_text(value[key], path="identity.%s" % key)
    return result


def _reject_runtime_identity(
    value: object,
    *,
    identity: Mapping[str, str],
    path: str,
) -> None:
    identity_values = frozenset(identity.values())
    for item_path, key, item in _walk(value, path=path):
        if key is not None and _normalized_token(key) in _FORBIDDEN_RUNTIME_IDENTIFIERS:
            raise ACFamilyContractError(
                "%s contains forbidden runtime identity input" % item_path
            )
        if type(item) is str and item in identity_values:
            raise ACFamilyContractError("%s aliases an identity-only value" % item_path)
        if type(item) is str and item in FAMILIES:
            raise ACFamilyContractError(
                "%s contains a family label as a callable input" % item_path
            )


def _reject_forbidden_text(value: str, *, path: str) -> None:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    for forbidden in _FORBIDDEN_RUNTIME_IDENTIFIERS:
        if re.search(r"(?:^|_)%s(?:_|$)" % re.escape(forbidden), normalized):
            raise ACFamilyContractError(
                "%s contains a forbidden identity token" % path
            )


def _walk(value: object, *, path: str, key: Optional[str] = None):
    yield path, key, value
    if isinstance(value, Mapping):
        for child_key in sorted(value):
            yield from _walk(
                value[child_key],
                path=_path_key(path, child_key),
                key=child_key,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path="%s[%d]" % (path, index))


def _canonicalize(value: object, *, path: str, ancestors: set[int]) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ACFamilyContractError("%s must be finite" % path)
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ACFamilyContractError("%s contains a cycle" % path)
        ancestors.add(identity)
        try:
            result = {}
            if any(type(key) is not str for key in value):
                raise ACFamilyContractError("%s contains a non-string key" % path)
            for key in sorted(value):
                result[key] = _canonicalize(
                    value[key], path=_path_key(path, key), ancestors=ancestors
                )
            return result
        finally:
            ancestors.remove(identity)
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in ancestors:
            raise ACFamilyContractError("%s contains a cycle" % path)
        ancestors.add(identity)
        try:
            return [
                _canonicalize(item, path="%s[%d]" % (path, index), ancestors=ancestors)
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)
    raise ACFamilyContractError(
        "%s contains unsupported type %s" % (path, type(value).__name__)
    )


def _first_difference(left: object, right: object, *, path: str):
    if type(left) is not type(right):
        return path, "type %s != %s" % (type(left).__name__, type(right).__name__)
    if isinstance(left, Mapping):
        if frozenset(left) != frozenset(right):
            return path, "mapping keys differ"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], path=_path_key(path, key))
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path, "list length differs"
        for index, (a, c) in enumerate(zip(left, right)):
            difference = _first_difference(a, c, path="%s[%d]" % (path, index))
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, "A=%r C=%r" % (left, right)
    return None


def _family(value: object) -> str:
    return _enum_text(value, FAMILIES, path="inputs.family")


def _enum_text(value: object, allowed: Sequence[str], *, path: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ACFamilyContractError("%s must be one of %r" % (path, tuple(allowed)))
    return value


def _nonempty_text(value: object, *, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ACFamilyContractError("%s must be a non-empty string" % path)
    return value


def _nonempty_bytes(value: object, *, path: str) -> bytes:
    if type(value) is not bytes or not value:
        raise ACFamilyContractError("%s must be non-empty exact bytes" % path)
    return value


def _finite_scalar(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ACFamilyContractError("%s must be a finite scalar" % path)
    result = float(value)
    if not math.isfinite(result):
        raise ACFamilyContractError("%s must be a finite scalar" % path)
    return 0.0 if result == 0.0 else result


def _canonical_float32(value: object, *, path: str) -> float:
    finite = _finite_scalar(value, path=path)
    try:
        result = struct.unpack("!f", struct.pack("!f", finite))[0]
    except OverflowError as exc:
        raise ACFamilyContractError("%s is outside float32 range" % path) from exc
    if not math.isfinite(result):
        raise ACFamilyContractError("%s is outside finite float32 range" % path)
    return 0.0 if result == 0.0 else float(result)


def _positive_exact_int(value: object, *, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ACFamilyContractError("%s must be a positive exact int" % path)
    return value


def _normalized_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _path_key(path: str, key: str) -> str:
    return "%s.%s" % (path, key) if key.isidentifier() else "%s[%r]" % (path, key)


PORTABLE_ABI_SHA256 = canonical_sha256(PORTABLE_ABI_MAPPING)

# Schema-v2 remains content-addressed evidence, never an active ABI parent.
SUPERSEDED_PORTABLE_ABI_SHA256 = PORTABLE_ABI_SHA256
SUPERSEDED_ACTOR_WIDTH = ACTOR_WIDTH
SUPERSEDED_CRITIC_WIDTH = CRITIC_WIDTH
SUPERSEDED_C07_PORTABLE_ABI_SHA256 = PORTABLE_ABI_SHA256
SUPERSEDED_C07_ACTOR_WIDTH = ACTOR_WIDTH
SUPERSEDED_C07_CRITIC_WIDTH = CRITIC_WIDTH
_EXPECTED_SUPERSEDED_PORTABLE_ABI_SHA256 = (
    "506078dae8eb6db1c02e6c48b25fcc3ed40c2ce3bec4711fb634a2df00e17382"
)
if SUPERSEDED_PORTABLE_ABI_SHA256 != _EXPECTED_SUPERSEDED_PORTABLE_ABI_SHA256:
    raise RuntimeError("superseded 245/353 evidence SHA drifted")


# ---------------------------------------------------------------------------
# C10 replacement: common pre/contact core, post-contact placement treatment.
# ---------------------------------------------------------------------------


def _load_strike_fact_consumer_authority() -> tuple[Tuple[str, ...], str, str]:
    """Read the device schema-2 tuple without copying its order here."""

    source_path = (
        Path(__file__).resolve().parent
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_strike_fact_device.py"
    )
    source_bytes = source_path.read_bytes()
    tree = ast.parse(source_bytes.decode("utf-8"), filename=str(source_path))
    values: dict[str, object] = {}

    def resolve(node: ast.AST) -> object:
        if isinstance(node, ast.Constant) and type(node.value) in (str, int):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(resolve(item) for item in node.elts)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = resolve(node.left)
            right = resolve(node.right)
            if type(left) is tuple and type(right) is tuple:
                return left + right
        raise RuntimeError("unsupported strike-fact authority expression")

    wanted = {
        "SCHEMA_VERSION",
        "GUIDE_CONSUMERS",
        "PADDLE_CENTER_PROXIMITY_CONSUMER",
        "COMMON_CONSUMERS",
        "STRIKE_FACT_CONSUMERS",
    }
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in wanted
        ):
            values[statement.targets[0].id] = resolve(statement.value)
    if values.get("SCHEMA_VERSION") != 2:
        raise RuntimeError("strike-fact device authority schema is not frozen v2")
    consumers = values.get("STRIKE_FACT_CONSUMERS")
    if (
        type(consumers) is not tuple
        or len(consumers) != 10
        or any(type(item) is not str or not item for item in consumers)
        or len(set(consumers)) != len(consumers)
    ):
        raise RuntimeError("strike-fact device ordered consumer authority is invalid")
    return (
        consumers,
        hashlib.sha256(source_bytes).hexdigest(),
        canonical_sha256(list(consumers)),
    )


(
    C10_CONTACT_REWARD_TERM_NAMES,
    C10_STRIKE_FACT_AUTHORITY_SOURCE_SHA256,
    C10_STRIKE_FACT_CONSUMER_ORDER_SHA256,
) = _load_strike_fact_consumer_authority()
C10_COMMON_CALLABLE_INPUT_SCHEMAS = {
    "observation_provider": "action_ball_common_observation_candidate_unfrozen_v1",
    "desired_at_contact_provider": "action_ball_common_desired_at_contact_p_v_face_v1",
    "normalizer": "action_ball_common_observation_normalizer_identity_transform_v1",
    "question_provider": "action_ball_common_question_full_receipt_v1",
    "selected_rubber_contact": "action_ball_common_selected_rubber_full_shot_key_v1",
    "on_table_scorer": "action_ball_common_on_table_success_v1",
    "success_denominator": "action_ball_common_on_table_denominator_v1",
    "curriculum": "action_ball_common_question_curriculum_v1",
    "common_on_table_outcome_consumer": (
        "action_ball_c06_common_on_table_paid_mailbox_consumer_v1"
    ),
    "post_contact_placement_raw_consumer": (
        "action_ball_c06_postcontact_raw_paid_mailbox_consumer_v1"
    ),
    "placement_mailbox_reader": (
        "action_ball_original_shot_full_key_delayed_mailbox_reader_v1"
    ),
    "placement_canonical_scorer": (
        "action_ball_c04_landing_profile_task_identity_facts_v1"
    ),
    "plant_step": "action_ball_common_plant_step_v1",
    "ppo_step": "action_ball_common_rollout_batch_v1",
}
C10_CONTACT_REWARD_INPUT_SCHEMA = (
    "action_ball_same_transition_full_key_one_shot_contact_reward_v1"
)
C10_OBSERVATION_WIDTH_STATUS = "UNFROZEN_UNTIL_CONSTRUCTED_RUNTIME"
C10_PRECONTACT_NORMALIZER_SEMANTICS = (
    "same_order_same_values_no_family_mask_or_transform_v1"
)
C10_CONTACT_PAYMENT_SEMANTICS = (
    "same_transition_full_shot_key_one_shot_post_physics_v1"
)
C10_PLACEMENT_SOURCE_SEMANTICS = (
    "original_shot_full_key_delayed_mailbox_c04_profile_task_facts_score_v1"
)
C10_PLACEMENT_ELIGIBILITY = (
    "original_shot_c05_full_key_selected_rubber_c04_score_bounded_late_mailbox_v1"
)
C10_ON_TABLE_SUCCESS_SEMANTICS = "common_valid_opponent_table_first_landing_v1"
C10_ALLOWED_DELTA_PATHS = (
    "family",
) + IDENTITY_ALLOWED_DELTA_PATHS + (
    "treatment.post_contact_placement_treatment_gain",
    "treatment_witness.guidance_payment_sha256",
    "treatment_witness.guidance_payment_all_zero",
    "treatment_witness.guidance_payment_nonzero_count",
    "treatment_witness.post_contact_placement_treatment_gain",
)
_C10_FORBIDDEN_PLACEMENT_TOKENS = frozenset(
    {
        "command",
        "command_metric",
        "command_manager",
        "command_name",
        "command_term",
        "current_command",
        "current_metric",
        "current_target",
        "current_target_xy",
        "get_command",
        "get_term",
        "metric",
        "metrics",
        "racket_target",
        "reward_manager",
        "target",
        "target_xy",
        "variant",
    }
)
_C10_PLACEMENT_MAILBOX_READER_BOUND_PARAMS = {
    "mailbox": "landing_outcome_full_shot_key"
}
C10_PLACEMENT_CONSUMER_NAMES = (
    "common_on_table_outcome",
    "post_contact_placement_raw",
)
C10_EVENT_PHASE_CONTACT = 1
C10_EVENT_PHASE_NET = 2
C10_EVENT_PHASE_LANDING = 3
C10_EVENT_PHASE_INBOUND = 0
C10_MIDSEQUENCE_STATES = (
    "INBOUND",
    "OPEN",
    "SETTLED_UNPAID",
    "PARTIALLY_PAID",
    "PAID",
)
C10_SLOT_ADMISSION_DECISIONS = (
    "ADMITTED",
    "REJECTED_PRE_REVEAL_CAPACITY",
)
_C10_PLACEMENT_CONSUMER_BOUND_PARAMS = {
    "common_on_table_outcome_consumer": {
        "consumer": C10_PLACEMENT_CONSUMER_NAMES[0],
        "mailbox": "c05_paid_original_shot",
    },
    "post_contact_placement_raw_consumer": {
        "consumer": C10_PLACEMENT_CONSUMER_NAMES[1],
        "mailbox": "c05_paid_original_shot",
    },
}
_C10_BUILD_TOKEN = object()
_C10_BUILD_AUTH_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class C10DesiredAtContactFact:
    """One typed, task-keyed desired position/velocity/face tape point."""

    point_index: int
    task_sha256: str
    position_world_m: Tuple[float, float, float]
    velocity_world_mps: Tuple[float, float, float]
    face_normal_world: Tuple[float, float, float]
    valid: bool


@dataclass(frozen=True)
class C10PhysicsStamp:
    """External R06 event ordering key: int64/int32/uint8."""

    control_step: int
    physics_substep: int
    event_phase: int


@dataclass(frozen=True)
class C10FlightSlotOwner:
    """One full-key INBOUND/OPEN owner of a physical flight slot."""

    shot_key: c05_mailbox.LandingOutcomeShotKey
    slot: int
    state: str


@dataclass(frozen=True)
class C10FlightSlotCapacityWitness:
    """One admitted or pre-reveal capacity-rejected successor decision."""

    open_shot_key: c05_mailbox.LandingOutcomeShotKey
    next_shot_key: c05_mailbox.LandingOutcomeShotKey
    occupied_owners: Tuple[C10FlightSlotOwner, ...]
    assigned_next_slot: Optional[int]
    decision: str


@dataclass(frozen=True)
class C10MidsequenceStateSnapshot:
    """One externally pinned R06/R10 restore-capability state."""

    state: str
    full_shot_key: c05_mailbox.LandingOutcomeShotKey
    physics_stamp: C10PhysicsStamp
    flight_slot: Optional[int]
    mailbox_slot: Optional[int]
    physical_retired: bool
    ball_generation: int
    flight_horizon_steps: int
    original_target_x_m: float
    original_target_y_m: float
    profile_sha256: str
    common_outcome_consumed: bool
    placement_treatment_consumed: bool


@dataclass(frozen=True)
class C10UnfrozenObservationContract:
    """Complete A/C-common observation candidate without declaring a width."""

    actor_width: Optional[int]
    critic_width: Optional[int]
    width_status: str
    actor_layout_candidate_bytes: bytes
    critic_layout_candidate_bytes: bytes
    observation_provider: ResolvedCallableInput
    desired_at_contact_provider: ResolvedCallableInput
    normalizer: ResolvedCallableInput
    normalizer_semantics: str
    normalizer_state_bytes: bytes
    desired_at_contact_fact_tape: Tuple[C10DesiredAtContactFact, ...]
    desired_at_contact_validity_semantics: str
    actor_normalized_precontact_tape: Tuple[bytes, ...]
    actor_policy_input_precontact_tape: Tuple[bytes, ...]
    critic_normalized_precontact_tape: Tuple[bytes, ...]
    critic_policy_input_precontact_tape: Tuple[bytes, ...]
    precontact_family_mask_applied: bool
    precontact_treatment_slots: Tuple[str, ...]


@dataclass(frozen=True)
class C10ContactRewardTerm:
    """One of the ten exact-common same-transition strike payments."""

    name: str
    reward: ResolvedCallableInput
    weight: float
    one_shot: bool
    same_transition: bool
    full_shot_keyed: bool
    payment_semantics: str


@dataclass(frozen=True)
class C10PlacementSource:
    """Common mailbox source bound to the canonical C04 scorer and profile."""

    mailbox_reader: ResolvedCallableInput
    canonical_scorer: ResolvedCallableInput
    canonical_profile: c04_landing.LandingPlacementProfile
    source_semantics: str
    eligibility_semantics: str
    mailbox_kind: str
    selected_rubber_contact_required: bool
    after_the_fact_required: bool


@dataclass(frozen=True)
class C10CommonRuntime:
    """Everything that must be byte/semantic-equal between fresh A and C."""

    backend_binding_bytes: bytes
    common_recipe_bytes: bytes
    post_dt_budget_receipt_bytes: bytes
    continuous_target_profile_sha256: str
    continuous_target_selection_authority_sha256: str
    continuous_target_runtime_dtype: str
    flight_horizon_steps: int
    flight_slot_capacity: int
    mailbox_slot_capacity: int
    observation: C10UnfrozenObservationContract
    question_provider: ResolvedCallableInput
    selected_rubber_contact: ResolvedCallableInput
    contact_rewards: Tuple[C10ContactRewardTerm, ...]
    on_table_scorer: ResolvedCallableInput
    on_table_success_weight: float
    on_table_success_semantics: str
    success_denominator: ResolvedCallableInput
    curriculum: ResolvedCallableInput
    common_on_table_outcome_consumer: ResolvedCallableInput
    post_contact_placement_raw_consumer: ResolvedCallableInput
    placement_source: C10PlacementSource
    post_contact_placement_consumer_scheduled: bool
    post_contact_placement_manager_weight: float
    plant_step: ResolvedCallableInput
    ppo_step: ResolvedCallableInput
    recovery_contract_bytes: bytes


@dataclass(frozen=True)
class C10PlacementTapePoint:
    """One delayed mailbox settlement used to derive the treatment payment."""

    shot_key: c05_mailbox.LandingOutcomeShotKey
    mailbox_shot_key: c05_mailbox.LandingOutcomeShotKey
    desired_fact_index: int
    selected_rubber_contact: bool
    strike_fact_tick: int
    selected_contact_stamp: Optional[C10PhysicsStamp]
    net_crossing_stamp: Optional[C10PhysicsStamp]
    landing_stamp: Optional[C10PhysicsStamp]
    physical_slot: int
    mailbox_slot: int
    physical_retired_at_settlement: bool
    ball_generation: int
    settlement_tick: int
    payment_tick: int
    next_target_reveal_tick: int
    next_precontact_observation_tick: int
    next_strike_fact_tick: int
    target_selection: c03_successor.TargetSelectionReceipt
    task_identity: c04_landing.LandingPlacementTaskIdentity
    facts: c04_landing.LandingPlacementFacts
    score: c04_landing.LandingPlacementScore
    next_shot_key: c05_mailbox.LandingOutcomeShotKey
    next_target_selection: c03_successor.TargetSelectionReceipt
    next_task_identity: c04_landing.LandingPlacementTaskIdentity


@dataclass(frozen=True)
class C10FixedTape:
    """Common values plus original-shot placement mailbox witnesses."""

    tape_bytes: bytes
    paired_replay_id: str
    common_call_input_trace: Tuple[bytes, ...]
    no_contact_strike_shot_key: c05_mailbox.LandingOutcomeShotKey
    no_contact_desired_fact_index: int
    no_contact_strike_fact_tick: int
    no_contact_strike_payment_tick: int
    no_contact_reward_fire_counts: Tuple[int, ...]
    no_contact_reward_payments: Tuple[float, ...]
    no_contact_selected_milestone_payment: float
    no_contact_outcome_payment: float
    on_table_success_values: Tuple[bool, ...]
    on_table_reward_values: Tuple[float, ...]
    common_on_table_consumer_fire_counts: Tuple[int, ...]
    placement_treatment_consumer_fire_counts: Tuple[int, ...]
    placement_points: Tuple[C10PlacementTapePoint, ...]
    landing_outcome_mailbox_checkpoint: Mapping[str, object]
    landing_outcome_mailbox_checkpoint_sha256: str
    flight_slot_capacity_witnesses: Tuple[C10FlightSlotCapacityWitness, ...]
    midsequence_states: Tuple[C10MidsequenceStateSnapshot, ...]
    midsequence_checkpoint_bytes: bytes
    midsequence_external_root_sha256: str
    restore_invokes_env_reset: bool


@dataclass(frozen=True)
class C10CheckpointLineage:
    """Only fresh initialization is admissible while widths are unfrozen."""

    initial_state_bytes: bytes
    optimizer_state_bytes: bytes
    normalizer_state_bytes: bytes
    rng_state_bytes: bytes
    resume_requested: bool
    parent_checkpoint_sha256: Optional[str]
    parent_abi_sha256: Optional[str]


@dataclass(frozen=True)
class C10ResolvedRuntimeInputs:
    family: str
    backend: str
    identity: Mapping[str, str]
    common: C10CommonRuntime
    fixed_tape: C10FixedTape
    checkpoint: C10CheckpointLineage
    post_contact_placement_treatment_gain: float


@dataclass(frozen=True)
class C10ResolvedProjection:
    """Builder-owned family projection; serialized artifacts cannot re-enter."""

    _payload_json: bytes
    _auth_tag: bytes
    _token: object
    _source_inputs: C10ResolvedRuntimeInputs

    def to_mapping(self) -> dict[str, object]:
        value = json.loads(self._payload_json.decode("ascii"))
        if not isinstance(value, dict):
            raise ACFamilyContractError("C10 projection payload is not a mapping")
        return value

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._payload_json).hexdigest()


@dataclass(frozen=True)
class C10FamilyPairValidation:
    schema_version: int
    evidence_level: str
    launch_gate_ready: bool
    a_projection_sha256: str
    c_projection_sha256: str
    common_runtime_sha256: str
    allowed_delta_paths: Tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "a_projection_sha256": self.a_projection_sha256,
            "allowed_delta_paths": list(self.allowed_delta_paths),
            "c_projection_sha256": self.c_projection_sha256,
            "common_runtime_sha256": self.common_runtime_sha256,
            "evidence_level": self.evidence_level,
            "launch_gate_ready": self.launch_gate_ready,
            "schema_version": self.schema_version,
        }


def build_c10_family_projection(
    inputs: C10ResolvedRuntimeInputs,
) -> C10ResolvedProjection:
    """Build a non-launch C10 family projection from typed common evidence."""

    if type(inputs) is not C10ResolvedRuntimeInputs:
        raise ACFamilyContractError("inputs must be C10ResolvedRuntimeInputs")
    family = _family(inputs.family)
    backend = _enum_text(inputs.backend, BACKENDS, path="inputs.backend")
    identity = _identity(inputs.identity)
    common = _resolve_c10_common(inputs.common, identity=identity)
    fixed_tape, placement_points = _resolve_c10_fixed_tape(
        inputs.fixed_tape,
        identity=identity,
        on_table_success_weight=float(common["on_table_success_weight"]),
        placement_source=common["placement_source"],
        desired_at_contact_facts=common["observation"][
            "desired_at_contact_facts"
        ],
        continuous_target_profile_sha256=common[
            "continuous_target_profile_sha256"
        ],
        continuous_target_selection_authority_sha256=common[
            "continuous_target_selection_authority_sha256"
        ],
        flight_horizon_steps=int(common["flight_horizon_steps"]),
        flight_slot_capacity=int(common["flight_slot_capacity"]),
        mailbox_slot_capacity=int(common["mailbox_slot_capacity"]),
    )
    checkpoint = _resolve_c10_checkpoint(inputs.checkpoint)
    if (
        common["observation"]["normalizer_state_sha256"]
        != checkpoint["normalizer_state_sha256"]
    ):
        raise ACFamilyContractError(
            "common observation normalizer state must equal fresh checkpoint "
            "normalizer state"
        )
    treatment = _c10_treatment(
        family=family,
        gain=inputs.post_contact_placement_treatment_gain,
    )
    treatment_witness = _c10_treatment_witness(
        family=family,
        manager_weight=float(common["post_contact_placement_manager_weight"]),
        gain=treatment["post_contact_placement_treatment_gain"],
        points=placement_points,
    )
    payload = {
        "abi_state": {
            "actor_width": C10_ACTOR_WIDTH,
            "critic_width": C10_CRITIC_WIDTH,
            "status": C10_OBSERVATION_WIDTH_STATUS,
            "superseded_checkpoint_compatible": False,
            "superseded_portable_abi_sha256": SUPERSEDED_PORTABLE_ABI_SHA256,
        },
        "backend": backend,
        "checkpoint_lineage": checkpoint,
        "contract_authority_kind": C10_CONTRACT_AUTHORITY_KIND,
        "contract_authority_sha256": C10_CONTRACT_AUTHORITY_SHA256,
        "common_runtime": {
            "contract": common,
            "fixed_tape": fixed_tape,
        },
        "family": family,
        "identity": identity,
        "evidence_level": C10_EVIDENCE_LEVEL,
        "kind": "action_ball_matched_ac_post_contact_placement_c10_v1",
        "launch_gate_ready": False,
        "schema_version": C10_SCHEMA_VERSION,
        "treatment": treatment,
        "treatment_witness": treatment_witness,
    }
    normalized = _canonicalize(payload, path="$", ancestors=set())
    payload_json = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    auth_tag = hmac.new(_C10_BUILD_AUTH_KEY, payload_json, hashlib.sha256).digest()
    return C10ResolvedProjection(
        _payload_json=payload_json,
        _auth_tag=auth_tag,
        _token=_C10_BUILD_TOKEN,
        _source_inputs=inputs,
    )


def validate_c10_family_pair(
    a_projection: C10ResolvedProjection,
    c_projection: C10ResolvedProjection,
) -> C10FamilyPairValidation:
    """Admit only exact-common A/C projections with the one placement axis."""

    a = _owned_c10_projection(a_projection, label="A")
    c = _owned_c10_projection(c_projection, label="C")
    if a["family"] != "A" or c["family"] != "C":
        raise ACFamilyContractError("C10 pair must be ordered family A then C")
    for key in (
        "abi_state",
        "backend",
        "checkpoint_lineage",
        "common_runtime",
        "contract_authority_kind",
        "contract_authority_sha256",
    ):
        difference = _first_difference(a[key], c[key], path="$.%s" % key)
        if difference is not None:
            path, detail = difference
            raise ACFamilyContractError(
                "%s differs outside post-contact placement treatment: %s"
                % (path, detail)
            )
    return C10FamilyPairValidation(
        schema_version=C10_SCHEMA_VERSION,
        evidence_level=C10_EVIDENCE_LEVEL,
        launch_gate_ready=False,
        a_projection_sha256=canonical_sha256(a),
        c_projection_sha256=canonical_sha256(c),
        common_runtime_sha256=canonical_sha256(a["common_runtime"]),
        allowed_delta_paths=C10_ALLOWED_DELTA_PATHS,
    )


def c10_projection_artifact(
    projection: C10ResolvedProjection,
) -> dict[str, object]:
    payload = _owned_c10_projection(projection, label="artifact")
    return {
        "canonical_sha256": canonical_sha256(payload),
        "projection": payload,
    }


def _resolve_c10_common(
    value: C10CommonRuntime,
    *,
    identity: Mapping[str, str],
) -> dict[str, object]:
    if type(value) is not C10CommonRuntime:
        raise ACFamilyContractError("common must be C10CommonRuntime")
    observation = _resolve_c10_observation(value.observation, identity=identity)
    callables = {}
    for name in (
        "question_provider",
        "selected_rubber_contact",
        "on_table_scorer",
        "success_denominator",
        "curriculum",
        "common_on_table_outcome_consumer",
        "post_contact_placement_raw_consumer",
        "plant_step",
        "ppo_step",
    ):
        callables[name] = _resolve_c10_callable(
            getattr(value, name),
            expected_schema=C10_COMMON_CALLABLE_INPUT_SCHEMAS[name],
            identity=identity,
            path="common.%s" % name,
        )
    for name, expected_bound in _C10_PLACEMENT_CONSUMER_BOUND_PARAMS.items():
        supplied = getattr(value, name)
        resolved_bound = _canonicalize(
            supplied.bound_params,
            path="common.%s.bound_params" % name,
            ancestors=set(),
        )
        if resolved_bound != expected_bound:
            raise ACFamilyContractError(
                "%s must bind its distinct original-shot C05 consumer domain" % name
            )
        _reject_c10_current_state_placement_inputs(
            supplied,
            identity=identity,
            path="common.%s" % name,
        )
    contact_rewards = _resolve_c10_contact_rewards(
        value.contact_rewards,
        identity=identity,
    )
    placement_source = _resolve_c10_placement_source(
        value.placement_source,
        identity=identity,
    )
    on_table_weight = _finite_scalar(
        value.on_table_success_weight,
        path="common.on_table_success_weight",
    )
    if on_table_weight <= 0.0:
        raise ACFamilyContractError("common on-table success weight must be positive")
    if value.on_table_success_semantics != C10_ON_TABLE_SUCCESS_SEMANTICS:
        raise ACFamilyContractError("common on-table success semantics differ")
    if type(value.post_contact_placement_consumer_scheduled) is not bool:
        raise ACFamilyContractError(
            "post-contact placement consumer scheduled must be exact bool"
        )
    if not value.post_contact_placement_consumer_scheduled:
        raise ACFamilyContractError(
            "both A and C must schedule the post-contact placement consumer"
        )
    placement_manager_weight = _finite_scalar(
        value.post_contact_placement_manager_weight,
        path="common.post_contact_placement_manager_weight",
    )
    if placement_manager_weight <= 0.0:
        raise ACFamilyContractError(
            "post-contact placement manager weight must be common and positive"
        )
    target_profile_sha = _sha256_text(
        value.continuous_target_profile_sha256,
        path="common.continuous_target_profile_sha256",
    )
    target_selection_authority_sha = _sha256_text(
        value.continuous_target_selection_authority_sha256,
        path="common.continuous_target_selection_authority_sha256",
    )
    if value.continuous_target_runtime_dtype != c03_successor.RUNTIME_TARGET_DTYPE:
        raise ACFamilyContractError(
            "continuous target runtime dtype must use canonical float32 authority"
        )
    flight_horizon_steps = _positive_exact_int(
        value.flight_horizon_steps,
        path="common.flight_horizon_steps",
    )
    flight_slot_capacity = _positive_exact_int(
        value.flight_slot_capacity,
        path="common.flight_slot_capacity",
    )
    mailbox_slot_capacity = _positive_exact_int(
        value.mailbox_slot_capacity,
        path="common.mailbox_slot_capacity",
    )
    return {
        "backend_binding_sha256": hashlib.sha256(
            _nonempty_bytes(
                value.backend_binding_bytes,
                path="common.backend_binding_bytes",
            )
        ).hexdigest(),
        "callables": callables,
        "common_recipe_sha256": hashlib.sha256(
            _nonempty_bytes(value.common_recipe_bytes, path="common.common_recipe_bytes")
        ).hexdigest(),
        "contact_rewards": contact_rewards,
        "contact_reward_order_authority_sha256": (
            C10_STRIKE_FACT_CONSUMER_ORDER_SHA256
        ),
        "contact_reward_source_sha256": C10_STRIKE_FACT_AUTHORITY_SOURCE_SHA256,
        "continuous_target_profile_sha256": target_profile_sha,
        "continuous_target_runtime_dtype": value.continuous_target_runtime_dtype,
        "continuous_target_selection_authority_sha256": (
            target_selection_authority_sha
        ),
        "flight_horizon_steps": flight_horizon_steps,
        "continuous_target_semantics_source_sha256": (
            C10_RUNTIME_TARGET_AUTHORITY_SOURCE_SHA256
        ),
        "landing_outcome_mailbox_authority_source_sha256": (
            C10_LANDING_OUTCOME_MAILBOX_AUTHORITY_SOURCE_SHA256
        ),
        "placement_success_authority_source_sha256": (
            C10_PLACEMENT_SUCCESS_AUTHORITY_SOURCE_SHA256
        ),
        "observation": observation,
        "on_table_success_semantics": value.on_table_success_semantics,
        "on_table_success_weight": on_table_weight,
        "placement_source": placement_source,
        "post_contact_placement_consumer_scheduled": True,
        "post_contact_placement_manager_weight": placement_manager_weight,
        "flight_slot_capacity": flight_slot_capacity,
        "mailbox_slot_capacity": mailbox_slot_capacity,
        "post_dt_budget_receipt_sha256": hashlib.sha256(
            _nonempty_bytes(
                value.post_dt_budget_receipt_bytes,
                path="common.post_dt_budget_receipt_bytes",
            )
        ).hexdigest(),
        "recovery_contract_sha256": hashlib.sha256(
            _nonempty_bytes(
                value.recovery_contract_bytes,
                path="common.recovery_contract_bytes",
            )
        ).hexdigest(),
    }


def _resolve_c10_observation(
    value: C10UnfrozenObservationContract,
    *,
    identity: Mapping[str, str],
) -> dict[str, object]:
    if type(value) is not C10UnfrozenObservationContract:
        raise ACFamilyContractError(
            "common.observation must be C10UnfrozenObservationContract"
        )
    if value.actor_width is not None or value.critic_width is not None:
        raise ACFamilyContractError(
            "C10 actor/critic widths must remain explicit None until constructed runtime"
        )
    if value.width_status != C10_OBSERVATION_WIDTH_STATUS:
        raise ACFamilyContractError("C10 observation width status differs")
    if value.normalizer_semantics != C10_PRECONTACT_NORMALIZER_SEMANTICS:
        raise ACFamilyContractError("pre/contact normalizer semantics differ")
    if type(value.precontact_family_mask_applied) is not bool:
        raise ACFamilyContractError("precontact_family_mask_applied must be exact bool")
    if value.precontact_family_mask_applied:
        raise ACFamilyContractError(
            "family-aware pre-contact zero mask/transform is forbidden"
        )
    if type(value.precontact_treatment_slots) is not tuple:
        raise ACFamilyContractError("precontact_treatment_slots must be exact tuple")
    if value.precontact_treatment_slots:
        raise ACFamilyContractError(
            "pre/contact observation cannot contain family treatment slots or validity"
        )
    tapes = {}
    for prefix in ("actor", "critic"):
        normalized = _nonempty_bytes_tuple(
            getattr(value, "%s_normalized_precontact_tape" % prefix),
            path="common.observation.%s_normalized_precontact_tape" % prefix,
        )
        policy_input = _nonempty_bytes_tuple(
            getattr(value, "%s_policy_input_precontact_tape" % prefix),
            path="common.observation.%s_policy_input_precontact_tape" % prefix,
        )
        if normalized != policy_input:
            raise ACFamilyContractError(
                "%s pre-contact policy input must equal normalized common values; "
                "C zero-mask is forbidden" % prefix
            )
        tapes[prefix] = {
            "normalized_sha256": _point_bytes_sha(normalized),
            "policy_input_sha256": _point_bytes_sha(policy_input),
            "point_count": len(normalized),
        }
    if (
        type(value.desired_at_contact_fact_tape) is not tuple
        or not value.desired_at_contact_fact_tape
    ):
        raise ACFamilyContractError(
            "desired-at-contact fact tape must be a non-empty exact tuple"
        )
    desired_facts = []
    for index, fact in enumerate(value.desired_at_contact_fact_tape):
        if type(fact) is not C10DesiredAtContactFact:
            raise ACFamilyContractError(
                "desired-at-contact fact %d must be C10DesiredAtContactFact" % index
            )
        if type(fact.point_index) is not int or fact.point_index != index:
            raise ACFamilyContractError(
                "desired-at-contact facts must have contiguous ordered point_index"
            )
        if type(fact.valid) is not bool:
            raise ACFamilyContractError(
                "desired-at-contact fact validity must be exact bool"
            )
        position = _finite_tuple(
            fact.position_world_m,
            path="desired_at_contact_fact_tape[%d].position_world_m" % index,
            expected_size=3,
        )
        velocity = _finite_tuple(
            fact.velocity_world_mps,
            path="desired_at_contact_fact_tape[%d].velocity_world_mps" % index,
            expected_size=3,
        )
        face = _finite_tuple(
            fact.face_normal_world,
            path="desired_at_contact_fact_tape[%d].face_normal_world" % index,
            expected_size=3,
        )
        if fact.valid:
            if not math.isclose(
                math.sqrt(sum(component * component for component in face)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ACFamilyContractError(
                    "valid desired-at-contact face must be a unit world-frame normal"
                )
        elif any(component != 0.0 for component in position + velocity + face):
            raise ACFamilyContractError(
                "invalid desired-at-contact p/v/face must use canonical positive zero"
            )
        desired_facts.append(
            {
                "face_normal_world": list(face),
                "point_index": index,
                "position_world_m": list(position),
                "task_sha256": _sha256_text(
                    fact.task_sha256,
                    path="desired_at_contact_fact_tape[%d].task_sha256" % index,
                ),
                "valid": fact.valid,
                "velocity_world_mps": list(velocity),
            }
        )
    if not any(fact["valid"] is True for fact in desired_facts):
        raise ACFamilyContractError(
            "desired-at-contact tape must witness at least one valid p/v/face fact"
        )
    desired_point_count = len(desired_facts)
    if any(
        tape["point_count"] != desired_point_count for tape in tapes.values()
    ):
        raise ACFamilyContractError(
            "desired-at-contact p/v/face facts must align one-for-one with actor "
            "and critic pre-contact tape points"
        )
    if value.desired_at_contact_validity_semantics != (
        "common_task_fact_no_treatment_mask_v1"
    ):
        raise ACFamilyContractError(
            "desired-at-contact validity cannot be family/treatment-aware"
        )
    return {
        "actor_layout_candidate_sha256": hashlib.sha256(
            _nonempty_bytes(
                value.actor_layout_candidate_bytes,
                path="common.observation.actor_layout_candidate_bytes",
            )
        ).hexdigest(),
        "actor_width": None,
        "critic_layout_candidate_sha256": hashlib.sha256(
            _nonempty_bytes(
                value.critic_layout_candidate_bytes,
                path="common.observation.critic_layout_candidate_bytes",
            )
        ).hexdigest(),
        "critic_width": None,
        "desired_at_contact_provider": _resolve_c10_callable(
            value.desired_at_contact_provider,
            expected_schema=C10_COMMON_CALLABLE_INPUT_SCHEMAS[
                "desired_at_contact_provider"
            ],
            identity=identity,
            path="common.observation.desired_at_contact_provider",
        ),
        "desired_at_contact_components": [
            {"name": name, "width": width}
            for name, width in zip(
                DESIRED_AT_CONTACT_COMPONENTS,
                DESIRED_AT_CONTACT_COMPONENT_WIDTHS,
            )
        ],
        "desired_at_contact_fact_sha256": canonical_sha256(desired_facts),
        "desired_at_contact_facts": desired_facts,
        "desired_at_contact_point_count": desired_point_count,
        "desired_at_contact_validity_semantics": (
            value.desired_at_contact_validity_semantics
        ),
        "desired_at_contact_validity_sha256": canonical_sha256(
            [fact["valid"] for fact in desired_facts]
        ),
        "normalizer": _resolve_c10_callable(
            value.normalizer,
            expected_schema=C10_COMMON_CALLABLE_INPUT_SCHEMAS["normalizer"],
            identity=identity,
            path="common.observation.normalizer",
        ),
        "normalizer_semantics": value.normalizer_semantics,
        "normalizer_state_sha256": hashlib.sha256(
            _nonempty_bytes(
                value.normalizer_state_bytes,
                path="common.observation.normalizer_state_bytes",
            )
        ).hexdigest(),
        "observation_provider": _resolve_c10_callable(
            value.observation_provider,
            expected_schema=C10_COMMON_CALLABLE_INPUT_SCHEMAS[
                "observation_provider"
            ],
            identity=identity,
            path="common.observation.observation_provider",
        ),
        "precontact_family_mask_applied": False,
        "precontact_treatment_slots": [],
        "precontact_value_tape": tapes,
        "width_status": value.width_status,
    }


def _resolve_c10_contact_rewards(
    value: Tuple[C10ContactRewardTerm, ...],
    *,
    identity: Mapping[str, str],
) -> list[dict[str, object]]:
    if type(value) is not tuple or len(value) != len(C10_CONTACT_REWARD_TERM_NAMES):
        raise ACFamilyContractError("C10 requires exactly ten ordered contact rewards")
    output = []
    for index, (term, expected_name) in enumerate(
        zip(value, C10_CONTACT_REWARD_TERM_NAMES)
    ):
        if type(term) is not C10ContactRewardTerm or term.name != expected_name:
            raise ACFamilyContractError(
                "contact reward %d must be %s" % (index, expected_name)
            )
        weight = _finite_scalar(term.weight, path="contact_rewards[%d].weight" % index)
        if weight <= 0.0:
            raise ACFamilyContractError(
                "all ten common contact reward weights must be positive"
            )
        if (
            type(term.one_shot) is not bool
            or type(term.same_transition) is not bool
            or type(term.full_shot_keyed) is not bool
            or not term.one_shot
            or not term.same_transition
            or not term.full_shot_keyed
        ):
            raise ACFamilyContractError(
                "contact reward %s must be one-shot, same-transition, full-keyed"
                % expected_name
            )
        if term.payment_semantics != C10_CONTACT_PAYMENT_SEMANTICS:
            raise ACFamilyContractError(
                "contact reward %s payment semantics differ" % expected_name
            )
        output.append(
            {
                "name": expected_name,
                "payment_semantics": term.payment_semantics,
                "reward": _resolve_c10_callable(
                    term.reward,
                    expected_schema=C10_CONTACT_REWARD_INPUT_SCHEMA,
                    identity=identity,
                    path="common.contact_rewards[%d]" % index,
                ),
                "weight": weight,
            }
        )
    return output


def _resolve_c10_placement_source(
    value: C10PlacementSource,
    *,
    identity: Mapping[str, str],
) -> dict[str, object]:
    if type(value) is not C10PlacementSource:
        raise ACFamilyContractError("placement_source must be C10PlacementSource")
    if value.source_semantics != C10_PLACEMENT_SOURCE_SEMANTICS:
        raise ACFamilyContractError("placement source must use original-shot mailbox")
    if value.eligibility_semantics != C10_PLACEMENT_ELIGIBILITY:
        raise ACFamilyContractError("placement eligibility semantics differ")
    if value.mailbox_kind != "action_ball_landing_outcome_mailbox_full_shot_key_v1":
        raise ACFamilyContractError("placement mailbox kind differs")
    if (
        type(value.selected_rubber_contact_required) is not bool
        or not value.selected_rubber_contact_required
        or type(value.after_the_fact_required) is not bool
        or not value.after_the_fact_required
    ):
        raise ACFamilyContractError(
            "placement eligibility requires after-the-fact selected-rubber contact"
        )
    mailbox_reader = _resolve_c10_callable(
        value.mailbox_reader,
        expected_schema=C10_COMMON_CALLABLE_INPUT_SCHEMAS[
            "placement_mailbox_reader"
        ],
        identity=identity,
        path="common.placement_source.mailbox_reader",
    )
    resolved_bound = _canonicalize(
        value.mailbox_reader.bound_params,
        path="common.placement_source.mailbox_reader.bound_params",
        ancestors=set(),
    )
    if resolved_bound != _C10_PLACEMENT_MAILBOX_READER_BOUND_PARAMS:
        raise ACFamilyContractError(
            "placement mailbox reader bound params must select only the original-shot "
            "full-key landing mailbox"
        )
    _reject_c10_current_state_placement_inputs(
        value.mailbox_reader,
        identity=identity,
        path="common.placement_source.mailbox_reader",
    )
    if type(value.canonical_profile) is not c04_landing.LandingPlacementProfile:
        raise ACFamilyContractError(
            "placement profile must be canonical C04 LandingPlacementProfile"
        )
    profile = value.canonical_profile
    if type(value.canonical_scorer) is not ResolvedCallableInput:
        raise ACFamilyContractError(
            "placement canonical scorer must be ResolvedCallableInput"
        )
    if (
        value.canonical_scorer.input_schema_id
        != C10_COMMON_CALLABLE_INPUT_SCHEMAS["placement_canonical_scorer"]
    ):
        raise ACFamilyContractError("placement canonical scorer input schema differs")
    if value.canonical_scorer.function is not _C10_CANONICAL_C04_SCORER:
        raise ACFamilyContractError(
            "placement scorer must be the canonical C04 score_landing_placement"
        )
    if value.canonical_scorer.dependency_functions:
        raise ACFamilyContractError(
            "canonical C04 scorer cannot accept caller-declared dependencies"
        )
    scorer_bound = _canonicalize(
        value.canonical_scorer.bound_params,
        path="common.placement_source.canonical_scorer.bound_params",
        ancestors=set(),
    )
    if scorer_bound != {"profile_sha256": profile.canonical_sha256}:
        raise ACFamilyContractError(
            "canonical C04 scorer must bind the exact profile SHA"
        )
    canonical_scorer = {
        "authority_source_sha256": C10_LANDING_PLACEMENT_AUTHORITY_SOURCE_SHA256,
        "bound_params_sha256": canonical_sha256(scorer_bound),
        "input_schema_id": value.canonical_scorer.input_schema_id,
        "signature": str(inspect.signature(value.canonical_scorer.function)),
        "symbol": "%s.%s"
        % (
            value.canonical_scorer.function.__module__,
            value.canonical_scorer.function.__qualname__,
        ),
    }
    return {
        "after_the_fact_required": True,
        "canonical_profile": profile.to_mapping(),
        "canonical_profile_sha256": profile.canonical_sha256,
        "canonical_scorer": canonical_scorer,
        "canonical_scorer_kind": c04_landing.SCORE_KIND,
        "eligibility_semantics": value.eligibility_semantics,
        "mailbox_kind": value.mailbox_kind,
        "mailbox_reader": mailbox_reader,
        "score_authority": (
            "canonical_c04_profile_task_facts_score_reexecution_v1"
        ),
        "selected_rubber_contact_required": True,
        "source_semantics": value.source_semantics,
    }


def _resolve_c10_shot_key(
    value: c05_mailbox.LandingOutcomeShotKey,
    *,
    path: str,
) -> dict[str, object]:
    """Resolve only the public C05 14-field key through its sealed factory."""

    if type(value) is not c05_mailbox.LandingOutcomeShotKey:
        raise ACFamilyContractError(
            "%s must be exact C05 LandingOutcomeShotKey" % path
        )
    try:
        round_tripped = c05_mailbox.LandingOutcomeShotKey.from_mapping(
            value.to_mapping()
        )
    except (TypeError, ValueError, c05_mailbox.LandingOutcomeMailboxError) as exc:
        raise ACFamilyContractError(
            "%s is not a sealed C05 full shot key" % path
        ) from exc
    if round_tripped != value:
        raise ACFamilyContractError("%s C05 key round-trip differs" % path)
    return round_tripped.full_key_dict()


def _c10_task_ref_from_shot_key(
    shot_key: Mapping[str, object],
    *,
    path: str,
) -> c03_successor.ContinuousActionTaskReceiptRef:
    try:
        return c03_successor.ContinuousActionTaskReceiptRef.from_runtime_mapping(
            {
                name: shot_key[name]
                for name in c03_successor.RUNTIME_TASK_REF_FIELDS
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ACFamilyContractError(
            "%s C05 runtime fields do not form the canonical task ref" % path
        ) from exc


def _resolve_c10_optional_physics_stamp(
    value: Optional[C10PhysicsStamp],
    *,
    expected_phase: int,
    path: str,
) -> Optional[dict[str, int]]:
    if value is None:
        return None
    if type(value) is not C10PhysicsStamp:
        raise ACFamilyContractError("%s must be C10PhysicsStamp or None" % path)
    if (
        type(value.control_step) is not int
        or not 0 <= value.control_step <= (1 << 63) - 1
    ):
        raise ACFamilyContractError("%s.control_step must be nonnegative int64" % path)
    if (
        type(value.physics_substep) is not int
        or not 0 <= value.physics_substep <= (1 << 31) - 1
    ):
        raise ACFamilyContractError(
            "%s.physics_substep must be nonnegative int32" % path
        )
    if (
        type(value.event_phase) is not int
        or not 0 <= value.event_phase <= 255
        or value.event_phase != expected_phase
    ):
        raise ACFamilyContractError(
            "%s.event_phase must equal its fixed uint8 phase" % path
        )
    return {
        "control_step": value.control_step,
        "event_phase": value.event_phase,
        "physics_substep": value.physics_substep,
    }


def _physics_stamp_key(value: Mapping[str, int]) -> Tuple[int, int, int]:
    return (
        value["control_step"],
        value["physics_substep"],
        value["event_phase"],
    )


def _raw_c10_midsequence_snapshot_mapping(
    value: C10MidsequenceStateSnapshot,
    *,
    path: str,
) -> dict[str, object]:
    if type(value) is not C10MidsequenceStateSnapshot:
        raise ACFamilyContractError(
            "%s must be C10MidsequenceStateSnapshot" % path
        )
    if value.state not in C10_MIDSEQUENCE_STATES:
        raise ACFamilyContractError("%s.state is not in the closed enum" % path)
    shot_key = _resolve_c10_shot_key(value.full_shot_key, path=path + ".full_shot_key")
    stamp = _resolve_c10_optional_physics_stamp(
        value.physics_stamp,
        expected_phase=value.physics_stamp.event_phase,
        path=path + ".physics_stamp",
    )
    assert stamp is not None
    if value.physics_stamp.event_phase not in (
        C10_EVENT_PHASE_INBOUND,
        C10_EVENT_PHASE_CONTACT,
        C10_EVENT_PHASE_NET,
        C10_EVENT_PHASE_LANDING,
    ):
        raise ACFamilyContractError("midsequence PhysicsStamp phase is unknown")
    for name in ("flight_slot", "mailbox_slot"):
        slot = getattr(value, name)
        if slot is not None and (type(slot) is not int or slot < 0):
            raise ACFamilyContractError(
                "midsequence %s must be nonnegative int or None" % name
            )
    if type(value.physical_retired) is not bool:
        raise ACFamilyContractError("midsequence physical_retired must be exact bool")
    if type(value.ball_generation) is not int or value.ball_generation < 0:
        raise ACFamilyContractError("midsequence ball_generation must be nonnegative int")
    horizon = _positive_exact_int(
        value.flight_horizon_steps,
        path=path + ".flight_horizon_steps",
    )
    if type(value.common_outcome_consumed) is not bool or type(
        value.placement_treatment_consumed
    ) is not bool:
        raise ACFamilyContractError(
            "midsequence dual-consumer bits must be exact bools"
        )
    return {
        "ball_generation": value.ball_generation,
        "common_outcome_consumed": value.common_outcome_consumed,
        "flight_horizon_steps": horizon,
        "full_shot_key": shot_key,
        "original_target_xy_float32": [
            _canonical_float32(
                value.original_target_x_m,
                path=path + ".original_target_x_m",
            ),
            _canonical_float32(
                value.original_target_y_m,
                path=path + ".original_target_y_m",
            ),
        ],
        "flight_slot": value.flight_slot,
        "mailbox_slot": value.mailbox_slot,
        "physical_retired": value.physical_retired,
        "physics_stamp": stamp,
        "placement_treatment_consumed": value.placement_treatment_consumed,
        "profile_sha256": _sha256_text(
            value.profile_sha256,
            path=path + ".profile_sha256",
        ),
        "state": value.state,
    }


def c10_midsequence_checkpoint_candidate_root(
    states: Tuple[C10MidsequenceStateSnapshot, ...],
    checkpoint_bytes: bytes,
) -> str:
    if type(states) is not tuple:
        raise ACFamilyContractError("midsequence states must be an exact tuple")
    mappings = tuple(
        _raw_c10_midsequence_snapshot_mapping(
            state,
            path="midsequence_states[%d]" % index,
        )
        for index, state in enumerate(states)
    )
    checkpoint_sha = hashlib.sha256(
        _nonempty_bytes(
            checkpoint_bytes,
            path="midsequence_checkpoint_bytes",
        )
    ).hexdigest()
    return canonical_sha256(
        {
            "checkpoint_state_sha256": checkpoint_sha,
            "kind": "action_ball_c10_midsequence_external_root_v1",
            "states": list(mappings),
        }
    )


def _resolve_c10_slot_capacity_witnesses(
    value: Tuple[C10FlightSlotCapacityWitness, ...],
    *,
    flight_slot_capacity: int,
    points: Tuple[Mapping[str, object], ...],
) -> list[dict[str, object]]:
    if type(value) is not tuple or len(value) != 2:
        raise ACFamilyContractError(
            "slot capacity tape must contain one admitted and one rejected witness"
        )
    resolved = []
    for index, witness in enumerate(value):
        if type(witness) is not C10FlightSlotCapacityWitness:
            raise ACFamilyContractError(
                "flight_slot_capacity_witnesses[%d] has wrong type" % index
            )
        open_key = _resolve_c10_shot_key(
            witness.open_shot_key,
            path="flight_slot_capacity_witnesses[%d].open_shot_key" % index,
        )
        next_key = _resolve_c10_shot_key(
            witness.next_shot_key,
            path="flight_slot_capacity_witnesses[%d].next_shot_key" % index,
        )
        if open_key != points[0]["shot_key"] or next_key != points[0]["next_shot_key"]:
            raise ACFamilyContractError(
                "slot capacity witness must bind the late original and successor keys"
            )
        if type(witness.occupied_owners) is not tuple:
            raise ACFamilyContractError("occupied_owners must be an exact tuple")
        owners = []
        for owner_index, owner in enumerate(witness.occupied_owners):
            if type(owner) is not C10FlightSlotOwner:
                raise ACFamilyContractError(
                    "occupied flight owner %d has wrong typed schema" % owner_index
                )
            owner_key = _resolve_c10_shot_key(
                owner.shot_key,
                path=(
                    "flight_slot_capacity_witnesses[%d].occupied_owners[%d].shot_key"
                    % (index, owner_index)
                ),
            )
            if (
                type(owner.slot) is not int
                or not 0 <= owner.slot < flight_slot_capacity
            ):
                raise ACFamilyContractError(
                    "occupied flight owner slot must lie inside flight capacity"
                )
            if owner.state not in ("INBOUND", "OPEN"):
                raise ACFamilyContractError(
                    "flight slot owner state must be INBOUND or OPEN"
                )
            owners.append(
                {
                    "shot_key": owner_key,
                    "shot_key_sha256": c05_mailbox.LandingOutcomeShotKey.coerce(
                        owner_key
                    ).canonical_sha256,
                    "slot": owner.slot,
                    "state": owner.state,
                }
            )
        owner_slots = tuple(owner["slot"] for owner in owners)
        owner_key_shas = tuple(owner["shot_key_sha256"] for owner in owners)
        if (
            len(set(owner_slots)) != len(owner_slots)
            or tuple(sorted(owner_slots)) != owner_slots
            or len(set(owner_key_shas)) != len(owner_key_shas)
        ):
            raise ACFamilyContractError(
                "occupied flight owners must have unique keys and sorted unique slots"
            )
        old_open_owners = tuple(
            owner
            for owner in owners
            if owner["shot_key"] == open_key
            and owner["slot"] == points[0]["physical_slot"]
            and owner["state"] == "OPEN"
        )
        if len(old_open_owners) != 1:
            raise ACFamilyContractError(
                "capacity witness must include the old shot as its exact OPEN owner"
            )
        if any(owner["shot_key"] == next_key for owner in owners):
            raise ACFamilyContractError(
                "successor key cannot own a flight slot before admission"
            )
        for owner in owners:
            for key_name in (
                "env_id",
                "reset_generation",
                "action_uid",
                "action_slot",
                "birth_sha256",
                "run_id",
                "carry_chain_id",
                "source_sha256",
                "config_sha256",
            ):
                if owner["shot_key"][key_name] != open_key[key_name]:
                    raise ACFamilyContractError(
                        "occupied flight owner must share the exact C05 carry cohort"
                    )
        occupied = owner_slots
        if witness.decision not in C10_SLOT_ADMISSION_DECISIONS:
            raise ACFamilyContractError("slot admission decision is unknown")
        assigned = witness.assigned_next_slot
        if witness.decision == "ADMITTED":
            if (
                type(assigned) is not int
                or not 0 <= assigned < flight_slot_capacity
                or assigned in occupied
                or assigned != points[1]["physical_slot"]
            ):
                raise ACFamilyContractError(
                    "admitted successor must receive an available physical slot"
                )
        elif assigned is not None or occupied != tuple(range(flight_slot_capacity)):
            raise ACFamilyContractError(
                "capacity exhaustion must reject before reveal without a slot"
            )
        resolved.append(
            {
                "assigned_next_slot": assigned,
                "decision": witness.decision,
                "next_shot_key": next_key,
                "occupied_owners": owners,
                "occupied_slots": list(occupied),
                "open_shot_key": open_key,
            }
        )
    if tuple(item["decision"] for item in resolved) != C10_SLOT_ADMISSION_DECISIONS:
        raise ACFamilyContractError(
            "slot capacity witnesses must be ordered admitted then pre-reveal rejected"
        )
    if not points[0]["next_target_reveal_tick"] < points[0]["settlement_tick"]:
        raise ACFamilyContractError(
            "slot witness must retain an OPEN old ball across the next reveal"
        )
    return resolved


def _resolve_c10_midsequence_states(
    value: Tuple[C10MidsequenceStateSnapshot, ...],
    *,
    checkpoint_bytes: bytes,
    external_root_sha256: str,
    restore_invokes_env_reset: object,
    flight_horizon_steps: int,
    flight_slot_capacity: int,
    mailbox_slot_capacity: int,
    points: Tuple[Mapping[str, object], ...],
    canonical_profile_sha256: str,
) -> dict[str, object]:
    if (
        type(value) is not tuple
        or any(type(item) is not C10MidsequenceStateSnapshot for item in value)
        or tuple(item.state for item in value) != C10_MIDSEQUENCE_STATES
    ):
        raise ACFamilyContractError(
            "midsequence capability must cover the exact ordered state set"
        )
    mappings = tuple(
        _raw_c10_midsequence_snapshot_mapping(
            state,
            path="midsequence_states[%d]" % index,
        )
        for index, state in enumerate(value)
    )
    expected_bits = (
        (False, False),
        (False, False),
        (False, False),
        (True, False),
        (True, True),
    )
    for index, (mapping, bits) in enumerate(zip(mappings, expected_bits)):
        if mapping["full_shot_key"] != points[0]["shot_key"]:
            raise ACFamilyContractError(
                "midsequence state must retain the original full 14-field key"
            )
        if (
            mapping["ball_generation"] != points[0]["ball_generation"]
            or mapping["flight_horizon_steps"] != flight_horizon_steps
            or mapping["profile_sha256"] != canonical_profile_sha256
            or mapping["original_target_xy_float32"]
            != [
                _canonical_float32(
                    points[0]["original_target_xy"][0],
                    path="points[0].original_target_x",
                ),
                _canonical_float32(
                    points[0]["original_target_xy"][1],
                    path="points[0].original_target_y",
                ),
            ]
        ):
            raise ACFamilyContractError(
                "midsequence state lost slot/ball/horizon/original target/profile"
            )
        if index < 2:
            if (
                mapping["flight_slot"] != points[0]["physical_slot"]
                or mapping["flight_slot"] >= flight_slot_capacity
                or mapping["mailbox_slot"] is not None
                or mapping["physical_retired"] is not False
            ):
                raise ACFamilyContractError(
                    "INBOUND/OPEN may occupy only a live flight slot"
                )
        elif (
            mapping["flight_slot"] is not None
            or mapping["mailbox_slot"] != points[0]["mailbox_slot"]
            or mapping["mailbox_slot"] >= mailbox_slot_capacity
            or mapping["physical_retired"] is not True
        ):
            raise ACFamilyContractError(
                "settled/unpaid/paid owner must retire flight and live only in mailbox"
            )
        actual_bits = (
            mapping["common_outcome_consumed"],
            mapping["placement_treatment_consumed"],
        )
        if actual_bits != bits:
            raise ACFamilyContractError(
                "midsequence dual-consumer bits differ at state %d" % index
            )
    stamp_keys = tuple(_physics_stamp_key(item["physics_stamp"]) for item in mappings)
    if any(left >= right for left, right in zip(stamp_keys, stamp_keys[1:])):
        raise ACFamilyContractError(
            "midsequence PhysicsStamps must be strictly lexicographic"
        )
    if type(restore_invokes_env_reset) is not bool or restore_invokes_env_reset:
        raise ACFamilyContractError(
            "midsequence restore must preserve state without env.reset"
        )
    expected_root = c10_midsequence_checkpoint_candidate_root(
        value,
        checkpoint_bytes,
    )
    supplied_root = _sha256_text(
        external_root_sha256,
        path="midsequence_external_root_sha256",
    )
    if supplied_root != expected_root:
        raise ACFamilyContractError(
            "midsequence checkpoint differs from the external root pin"
        )
    return {
        "checkpoint_state_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "external_root_sha256": supplied_root,
        "restore_invokes_env_reset": False,
        "states": list(mappings),
    }


def _resolve_c10_fixed_tape(
    value: C10FixedTape,
    *,
    identity: Mapping[str, str],
    on_table_success_weight: float,
    placement_source: Mapping[str, object],
    desired_at_contact_facts: Sequence[Mapping[str, object]],
    continuous_target_profile_sha256: str,
    continuous_target_selection_authority_sha256: str,
    flight_horizon_steps: int,
    flight_slot_capacity: int,
    mailbox_slot_capacity: int,
) -> tuple[dict[str, object], Tuple[dict[str, object], ...]]:
    if type(value) is not C10FixedTape:
        raise ACFamilyContractError("fixed_tape must be C10FixedTape")
    checkpoint_sha = _sha256_text(
        value.landing_outcome_mailbox_checkpoint_sha256,
        path="fixed_tape.landing_outcome_mailbox_checkpoint_sha256",
    )
    try:
        mailbox = c05_mailbox.LandingOutcomeMailbox.from_checkpoint(
            value.landing_outcome_mailbox_checkpoint,
            expected_checkpoint_sha256=checkpoint_sha,
        )
        canonical_mailbox_checkpoint = mailbox.to_checkpoint()
    except (TypeError, ValueError, c05_mailbox.LandingOutcomeMailboxError) as exc:
        raise ACFamilyContractError(
            "fixed tape must bind a canonical bounded C05 mailbox checkpoint"
        ) from exc
    if canonical_mailbox_checkpoint["canonical_sha256"] != checkpoint_sha:
        raise ACFamilyContractError("C05 mailbox checkpoint authority SHA differs")
    if canonical_mailbox_checkpoint["capacity"] != mailbox_slot_capacity:
        raise ACFamilyContractError(
            "C05 checkpoint capacity must equal the separate mailbox slot capacity"
        )

    contact_payments = _finite_tuple(
        value.no_contact_reward_payments,
        path="fixed_tape.no_contact_reward_payments",
        expected_size=len(C10_CONTACT_REWARD_TERM_NAMES),
    )
    if any(item <= 0.0 for item in contact_payments):
        raise ACFamilyContractError(
            "fixed tape must witness all ten positive common strike-shaping "
            "payments on a legal no-contact fact"
        )
    for name in ("no_contact_strike_fact_tick", "no_contact_strike_payment_tick"):
        tick = getattr(value, name)
        if type(tick) is not int or tick < 0:
            raise ACFamilyContractError(
                "fixed_tape.%s must be nonnegative exact int" % name
            )
    if value.no_contact_strike_fact_tick != value.no_contact_strike_payment_tick:
        raise ACFamilyContractError(
            "all ten strike rewards must fire once on the same post-physics transition"
        )
    if (
        type(value.no_contact_reward_fire_counts) is not tuple
        or len(value.no_contact_reward_fire_counts)
        != len(C10_CONTACT_REWARD_TERM_NAMES)
        or any(
            type(item) is not int or item != 1
            for item in value.no_contact_reward_fire_counts
        )
    ):
        raise ACFamilyContractError(
            "no_contact_reward_fire_counts must witness each full-key strike "
            "payment exactly once"
        )
    if (
        type(value.on_table_success_values) is not tuple
        or not value.on_table_success_values
        or any(type(item) is not bool for item in value.on_table_success_values)
    ):
        raise ACFamilyContractError(
            "on_table_success_values must be a non-empty exact bool tuple"
        )
    on_table_rewards = _finite_tuple(
        value.on_table_reward_values,
        path="fixed_tape.on_table_reward_values",
        expected_size=len(value.on_table_success_values),
    )
    dual_consumer_counts = {}
    for name in (
        "common_on_table_consumer_fire_counts",
        "placement_treatment_consumer_fire_counts",
    ):
        counts = getattr(value, name)
        if (
            type(counts) is not tuple
            or len(counts) != len(value.on_table_success_values)
            or any(type(item) is not int or item != 1 for item in counts)
        ):
            raise ACFamilyContractError(
                "%s must record exactly one consumption per C05 paid row" % name
            )
        dual_consumer_counts[name] = counts
    traces = _nonempty_bytes_tuple(
        value.common_call_input_trace,
        path="fixed_tape.common_call_input_trace",
    )
    if type(value.placement_points) is not tuple or len(value.placement_points) < 2:
        raise ACFamilyContractError(
            "placement_points must contain at least two continuous shots"
        )
    points = tuple(
        _resolve_c10_placement_point(
            point,
            index=index,
            mailbox=mailbox,
            placement_source=placement_source,
            desired_at_contact_facts=desired_at_contact_facts,
            continuous_target_profile_sha256=continuous_target_profile_sha256,
            continuous_target_selection_authority_sha256=(
                continuous_target_selection_authority_sha256
            ),
            flight_horizon_steps=flight_horizon_steps,
            flight_slot_capacity=flight_slot_capacity,
            mailbox_slot_capacity=mailbox_slot_capacity,
        )
        for index, point in enumerate(value.placement_points)
    )

    point_key_shas = tuple(point["c05_full_shot_key_sha256"] for point in points)
    if len(set(point_key_shas)) != len(point_key_shas):
        raise ACFamilyContractError(
            "each C05 full shot key may own exactly one placement payment"
        )
    checkpoint_entries = canonical_mailbox_checkpoint["entries"]
    if not isinstance(checkpoint_entries, list) or len(checkpoint_entries) != len(points):
        raise ACFamilyContractError(
            "C05 checkpoint rows must equal the fixed placement cohort exactly"
        )
    try:
        checkpoint_key_shas = tuple(
            c05_mailbox.LandingOutcomeShotKey.from_mapping(entry["task_key"])
            .canonical_sha256
            for entry in checkpoint_entries
        )
    except (KeyError, TypeError, ValueError, c05_mailbox.LandingOutcomeMailboxError) as exc:
        raise ACFamilyContractError("C05 checkpoint row key is invalid") from exc
    if set(checkpoint_key_shas) != set(point_key_shas):
        raise ACFamilyContractError(
            "C05 bounded mailbox contains hidden or missing placement rows"
        )
    payment_ids = tuple(point["mailbox_payment_idempotency_sha256"] for point in points)
    if len(set(payment_ids)) != len(payment_ids):
        raise ACFamilyContractError(
            "C05 mailbox payment idempotency keys must be unique per full shot key"
        )
    replay_keys = tuple(
        [point["shot_key"] for point in points]
        + [points[-1]["next_shot_key"]]
    )
    replay_run_ids = {key["run_id"] for key in replay_keys}
    replay_carry_chain_ids = {key["carry_chain_id"] for key in replay_keys}
    if len(replay_run_ids) != 1 or len(replay_carry_chain_ids) != 1:
        raise ACFamilyContractError(
            "fixed tape C05 keys must share one paired-replay source identity"
        )
    if "live_job_id" not in identity:
        raise ACFamilyContractError(
            "C10 identity must contain a typed live_job_id"
        )
    paired_replay_id = _nonempty_text(
        value.paired_replay_id,
        path="fixed_tape.paired_replay_id",
    )
    replay_run_id = next(iter(replay_run_ids))
    replay_carry_chain_id = next(iter(replay_carry_chain_ids))
    if paired_replay_id != replay_run_id:
        raise ACFamilyContractError(
            "paired_replay_id must equal the shared C05 replay-source run_id"
        )
    if identity["live_job_id"] in (paired_replay_id, replay_carry_chain_id):
        raise ACFamilyContractError(
            "paired replay identity must differ from the required live_job_id"
        )
    paired_replay_source_sha256 = canonical_sha256(
        {
            "carry_chain_id": replay_carry_chain_id,
            "kind": "action_ball_c10_shared_paired_replay_source_v1",
            "ordered_full_key_sha256": [
                c05_mailbox.LandingOutcomeShotKey.coerce(key).canonical_sha256
                for key in replay_keys
            ],
            "run_id": paired_replay_id,
            "tape_sha256": hashlib.sha256(
                _nonempty_bytes(value.tape_bytes, path="fixed_tape.tape_bytes")
            ).hexdigest(),
        }
    )
    flight_slot_capacity_witnesses = _resolve_c10_slot_capacity_witnesses(
        value.flight_slot_capacity_witnesses,
        flight_slot_capacity=flight_slot_capacity,
        points=points,
    )
    midsequence_capability = _resolve_c10_midsequence_states(
        value.midsequence_states,
        checkpoint_bytes=value.midsequence_checkpoint_bytes,
        external_root_sha256=value.midsequence_external_root_sha256,
        restore_invokes_env_reset=value.restore_invokes_env_reset,
        flight_horizon_steps=flight_horizon_steps,
        flight_slot_capacity=flight_slot_capacity,
        mailbox_slot_capacity=mailbox_slot_capacity,
        points=points,
        canonical_profile_sha256=str(
            placement_source["canonical_profile_sha256"]
        ),
    )

    for index, (point, successor) in enumerate(zip(points, points[1:])):
        if (
            point["next_shot_key"] != successor["shot_key"]
            or point["next_task_identity_sha256"]
            != successor["canonical_c04_task_identity_sha256"]
            or point["next_target_selection_sha256"]
            != successor["canonical_float32_target_selection_sha256"]
        ):
            raise ACFamilyContractError(
                "continuous point %d successor receipts must equal point %d current "
                "shot/task/float32 target" % (index, index + 1)
            )
        if point["next_strike_fact_tick"] != successor["strike_fact_tick"]:
            raise ACFamilyContractError(
                "continuous successor strike-fact tick must chain to the next point"
            )

    no_contact_strike_shot_key = _resolve_c10_shot_key(
        value.no_contact_strike_shot_key,
        path="fixed_tape.no_contact_strike_shot_key",
    )
    if (
        type(value.no_contact_desired_fact_index) is not int
        or value.no_contact_desired_fact_index < 0
        or value.no_contact_desired_fact_index >= len(desired_at_contact_facts)
    ):
        raise ACFamilyContractError(
            "no-contact strike desired_fact_index must reference the common typed tape"
        )
    no_contact_desired_fact = desired_at_contact_facts[
        value.no_contact_desired_fact_index
    ]
    if no_contact_strike_shot_key["task_sha256"] != no_contact_desired_fact[
        "task_sha256"
    ]:
        raise ACFamilyContractError(
            "no-contact strike full shot key must match its desired p/v/face fact"
        )
    if no_contact_desired_fact["valid"] is not True:
        raise ACFamilyContractError(
            "positive no-contact strike shaping requires a valid desired p/v/face fact"
        )
    matching_points = tuple(
        point for point in points if point["shot_key"] == no_contact_strike_shot_key
    )
    if len(matching_points) != 1:
        raise ACFamilyContractError(
            "no-contact strike witness must bind exactly one full-key mailbox cohort"
        )
    no_contact_point = matching_points[0]
    if no_contact_point["selected_rubber_contact"]:
        raise ACFamilyContractError(
            "strike shaping must include a legal no-contact fact independent of "
            "selected-contact milestone"
        )
    if (
        value.no_contact_strike_fact_tick != no_contact_point["strike_fact_tick"]
        or value.no_contact_strike_payment_tick
        != no_contact_point["strike_fact_tick"]
    ):
        raise ACFamilyContractError(
            "all ten no-contact strike payments must bind the exact-fact transition tick"
        )
    selected_milestone = _finite_scalar(
        value.no_contact_selected_milestone_payment,
        path="fixed_tape.no_contact_selected_milestone_payment",
    )
    outcome_payment = _finite_scalar(
        value.no_contact_outcome_payment,
        path="fixed_tape.no_contact_outcome_payment",
    )
    if selected_milestone != 0.0 or outcome_payment != 0.0:
        raise ACFamilyContractError(
            "legal no-contact strike shaping must keep selected-contact milestone "
            "and outcome payment at zero"
        )
    if len(points) != len(value.on_table_success_values):
        raise ACFamilyContractError(
            "on-table truth/payment tape must align with placement mailbox points"
        )
    for index, (point, success, reward) in enumerate(
        zip(points, value.on_table_success_values, on_table_rewards)
    ):
        if point["on_table_success"] is not success:
            raise ACFamilyContractError(
                "on-table truth at point %d disagrees with original-shot mailbox" % index
            )
        expected_reward = on_table_success_weight if success else 0.0
        if reward != expected_reward:
            raise ACFamilyContractError(
                "on-table payment at point %d must use the common scorer/weight" % index
            )
    if not any(
        point["eligible"] and point["placement_score"] > 0.0 for point in points
    ):
        raise ACFamilyContractError(
            "fixed tape must witness nonzero eligible delayed placement score"
        )
    mapping = {
        "common_call_input_trace_sha256": _point_bytes_sha(traces),
        "landing_outcome_mailbox_capacity": canonical_mailbox_checkpoint["capacity"],
        "landing_outcome_mailbox_checkpoint_sha256": checkpoint_sha,
        "landing_outcome_mailbox_entry_count": len(checkpoint_entries),
        "midsequence_checkpoint_capability": midsequence_capability,
        "flight_slot_capacity_witnesses": flight_slot_capacity_witnesses,
        "placement_dual_consumer_names": list(C10_PLACEMENT_CONSUMER_NAMES),
        "common_on_table_consumer_fire_counts": list(
            dual_consumer_counts["common_on_table_consumer_fire_counts"]
        ),
        "placement_treatment_consumer_fire_counts": list(
            dual_consumer_counts["placement_treatment_consumer_fire_counts"]
        ),
        "common_on_table_consumer_ledger_keys": [
            canonical_sha256(
                {
                    "consumer": C10_PLACEMENT_CONSUMER_NAMES[0],
                    "mailbox_payment_idempotency_sha256": payment_id,
                }
            )
            for payment_id in payment_ids
        ],
        "placement_treatment_consumer_ledger_keys": [
            canonical_sha256(
                {
                    "consumer": C10_PLACEMENT_CONSUMER_NAMES[1],
                    "mailbox_payment_idempotency_sha256": payment_id,
                }
            )
            for payment_id in payment_ids
        ],
        "paired_replay_source_sha256": paired_replay_source_sha256,
        "paired_replay_id": paired_replay_id,
        "ordered_strike_reward_names": list(C10_CONTACT_REWARD_TERM_NAMES),
        "ordered_strike_reward_source_sha256": C10_STRIKE_FACT_AUTHORITY_SOURCE_SHA256,
        "ordered_strike_reward_order_sha256": C10_STRIKE_FACT_CONSUMER_ORDER_SHA256,
        "no_contact_desired_fact_index": value.no_contact_desired_fact_index,
        "no_contact_strike_fact_tick": value.no_contact_strike_fact_tick,
        "no_contact_strike_payment_tick": value.no_contact_strike_payment_tick,
        "no_contact_strike_payment_ledger_keys": [
            "%s:%s" % (
                c05_mailbox.LandingOutcomeShotKey.coerce(
                    no_contact_strike_shot_key
                ).canonical_sha256,
                name,
            )
            for name in C10_CONTACT_REWARD_TERM_NAMES
        ],
        "no_contact_reward_fire_counts": list(value.no_contact_reward_fire_counts),
        "no_contact_reward_payments": list(contact_payments),
        "no_contact_selected_milestone_payment": selected_milestone,
        "no_contact_outcome_payment": outcome_payment,
        "no_contact_strike_shot_key": no_contact_strike_shot_key,
        "on_table_reward_values": list(on_table_rewards),
        "on_table_success_values": list(value.on_table_success_values),
        "placement_points": list(points),
        "tape_sha256": hashlib.sha256(
            _nonempty_bytes(value.tape_bytes, path="fixed_tape.tape_bytes")
        ).hexdigest(),
    }
    return mapping, points


def _resolve_c10_placement_point(
    value: C10PlacementTapePoint,
    *,
    index: int,
    mailbox: c05_mailbox.LandingOutcomeMailbox,
    placement_source: Mapping[str, object],
    desired_at_contact_facts: Sequence[Mapping[str, object]],
    continuous_target_profile_sha256: str,
    continuous_target_selection_authority_sha256: str,
    flight_horizon_steps: int,
    flight_slot_capacity: int,
    mailbox_slot_capacity: int,
) -> dict[str, object]:
    if type(value) is not C10PlacementTapePoint:
        raise ACFamilyContractError(
            "placement_points[%d] must be C10PlacementTapePoint" % index
        )
    shot_key = _resolve_c10_shot_key(value.shot_key, path="shot_key")
    mailbox_key = _resolve_c10_shot_key(
        value.mailbox_shot_key,
        path="mailbox_shot_key",
    )
    if shot_key != mailbox_key:
        raise ACFamilyContractError(
            "placement point %d mailbox does not belong to original C05 full shot key"
            % index
        )
    if type(value.selected_rubber_contact) is not bool:
        raise ACFamilyContractError("selected_rubber_contact must be exact bool")
    if (
        type(value.physical_slot) is not int
        or not 0 <= value.physical_slot < flight_slot_capacity
    ):
        raise ACFamilyContractError(
            "placement physical_slot must lie inside the flight slot capacity"
        )
    if (
        type(value.mailbox_slot) is not int
        or not 0 <= value.mailbox_slot < mailbox_slot_capacity
    ):
        raise ACFamilyContractError(
            "placement mailbox_slot must lie inside the separate mailbox capacity"
        )
    if (
        type(value.physical_retired_at_settlement) is not bool
        or not value.physical_retired_at_settlement
    ):
        raise ACFamilyContractError(
            "settlement must atomically retire the physical flight slot"
        )
    if type(value.ball_generation) is not int or value.ball_generation < 0:
        raise ACFamilyContractError("placement ball_generation must be nonnegative int")
    if (
        type(value.desired_fact_index) is not int
        or value.desired_fact_index < 0
        or value.desired_fact_index >= len(desired_at_contact_facts)
    ):
        raise ACFamilyContractError(
            "placement point %d desired_fact_index must reference the common typed tape"
            % index
        )
    desired_fact = desired_at_contact_facts[value.desired_fact_index]
    if value.selected_rubber_contact and desired_fact["valid"] is not True:
        raise ACFamilyContractError(
            "selected contact must reference a valid desired p/v/face fact"
        )
    for name in (
        "strike_fact_tick",
        "settlement_tick",
        "payment_tick",
        "next_target_reveal_tick",
        "next_precontact_observation_tick",
        "next_strike_fact_tick",
    ):
        tick = getattr(value, name)
        if type(tick) is not int or tick < 0:
            raise ACFamilyContractError(
                "placement point %d %s must be nonnegative exact int" % (index, name)
            )
    if not value.strike_fact_tick < value.settlement_tick <= value.payment_tick:
        raise ACFamilyContractError(
            "placement point %d must settle and pay after its original strike fact"
            % index
        )
    if not (
        value.strike_fact_tick
        < value.next_target_reveal_tick
        <= value.next_precontact_observation_tick
        < value.next_strike_fact_tick
    ):
        raise ACFamilyContractError(
            "next target must reveal by a pre-contact observation and strictly before "
            "the successor strike fact"
        )
    if type(value.task_identity) is not c04_landing.LandingPlacementTaskIdentity:
        raise ACFamilyContractError(
            "placement task identity must be canonical C04 typed identity"
        )
    if type(value.facts) is not c04_landing.LandingPlacementFacts:
        raise ACFamilyContractError("placement facts must be canonical C04 typed facts")
    if type(value.score) is not c04_landing.LandingPlacementScore:
        raise ACFamilyContractError("placement score must be canonical C04 typed score")
    profile = c04_landing.LandingPlacementProfile.from_mapping(
        placement_source["canonical_profile"]
    )
    try:
        recomputed = _C10_CANONICAL_C04_SCORER(
            profile,
            value.task_identity,
            value.facts,
        )
    except (TypeError, ValueError, c04_landing.LandingPlacementIdentityError) as exc:
        raise ACFamilyContractError(
            "canonical C04 placement profile/task/facts chain is invalid"
        ) from exc
    if recomputed.to_mapping() != value.score.to_mapping():
        raise ACFamilyContractError(
            "placement score must byte-equal canonical C04 scorer reexecution"
        )
    if shot_key["task_sha256"] != value.task_identity.task_receipt_sha256:
        raise ACFamilyContractError(
            "C05 full shot key must bind the canonical C04 task receipt"
        )
    current_ref = _c10_task_ref_from_shot_key(shot_key, path="shot_key")
    if type(value.target_selection) is not c03_successor.TargetSelectionReceipt:
        raise ACFamilyContractError(
            "current target must use canonical float32 TargetSelectionReceipt"
        )
    if (
        value.target_selection.profile_sha256 != continuous_target_profile_sha256
        or value.target_selection.selection_authority_sha256
        != continuous_target_selection_authority_sha256
        or value.target_selection.runtime_dtype != c03_successor.RUNTIME_TARGET_DTYPE
        or value.target_selection.target_generation != shot_key["swing_generation"]
        or value.target_selection.task_ref_sha256 != current_ref.canonical_sha256
    ):
        raise ACFamilyContractError(
            "current float32 target receipt profile/authority/generation/task binding differs"
        )
    current_runtime_target = (
        _canonical_float32(
            value.task_identity.target_x_m,
            path="placement.task_identity.target_x_m",
        ),
        _canonical_float32(
            value.task_identity.target_y_m,
            path="placement.task_identity.target_y_m",
        ),
    )
    if value.target_selection.runtime_target_xy_m != current_runtime_target:
        raise ACFamilyContractError(
            "C04 task target must equal canonical runtime float32 target bytes"
        )
    if shot_key["task_sha256"] != desired_fact["task_sha256"]:
        raise ACFamilyContractError(
            "placement/strike full shot key task must match its desired p/v/face fact"
        )
    if value.selected_rubber_contact is not value.facts.contact_valid:
        raise ACFamilyContractError(
            "selected-rubber contact must equal canonical C04 contact authority"
        )
    if value.ball_generation != shot_key["swing_generation"]:
        raise ACFamilyContractError(
            "physical ball_generation must equal the original shot generation"
        )
    contact_stamp = _resolve_c10_optional_physics_stamp(
        value.selected_contact_stamp,
        expected_phase=C10_EVENT_PHASE_CONTACT,
        path="placement.selected_contact_stamp",
    )
    net_stamp = _resolve_c10_optional_physics_stamp(
        value.net_crossing_stamp,
        expected_phase=C10_EVENT_PHASE_NET,
        path="placement.net_crossing_stamp",
    )
    landing_stamp = _resolve_c10_optional_physics_stamp(
        value.landing_stamp,
        expected_phase=C10_EVENT_PHASE_LANDING,
        path="placement.landing_stamp",
    )
    if value.selected_rubber_contact:
        if contact_stamp is None or contact_stamp["control_step"] != value.strike_fact_tick:
            raise ACFamilyContractError(
                "selected contact must bind its exact CONTACT PhysicsStamp"
            )
        if value.facts.first_plane_crossing_valid:
            if net_stamp is None or landing_stamp is None:
                raise ACFamilyContractError(
                    "scored landing requires CONTACT, NET and LANDING PhysicsStamps"
                )
            if not (
                _physics_stamp_key(contact_stamp)
                < _physics_stamp_key(net_stamp)
                < _physics_stamp_key(landing_stamp)
            ):
                raise ACFamilyContractError(
                    "PhysicsStamp order must be CONTACT < NET < LANDING"
                )
            if landing_stamp["control_step"] > value.settlement_tick:
                raise ACFamilyContractError(
                    "LANDING PhysicsStamp cannot occur after mailbox settlement"
                )
    elif any(item is not None for item in (contact_stamp, net_stamp, landing_stamp)):
        raise ACFamilyContractError(
            "no-contact horizon row cannot forge CONTACT/NET/LANDING PhysicsStamps"
        )

    next_shot_key = _resolve_c10_shot_key(value.next_shot_key, path="next_shot_key")
    if type(value.next_task_identity) is not c04_landing.LandingPlacementTaskIdentity:
        raise ACFamilyContractError(
            "next task identity must be canonical C04 typed identity"
        )
    if next_shot_key["task_sha256"] != value.next_task_identity.task_receipt_sha256:
        raise ACFamilyContractError(
            "next C05 full shot key must bind the next C04 task receipt"
        )
    next_ref = _c10_task_ref_from_shot_key(next_shot_key, path="next_shot_key")
    if type(value.next_target_selection) is not c03_successor.TargetSelectionReceipt:
        raise ACFamilyContractError(
            "next target must use canonical float32 TargetSelectionReceipt"
        )
    if (
        value.next_target_selection.profile_sha256 != continuous_target_profile_sha256
        or value.next_target_selection.selection_authority_sha256
        != continuous_target_selection_authority_sha256
        or value.next_target_selection.runtime_dtype
        != c03_successor.RUNTIME_TARGET_DTYPE
        or value.next_target_selection.target_generation
        != next_shot_key["swing_generation"]
        or value.next_target_selection.task_ref_sha256 != next_ref.canonical_sha256
    ):
        raise ACFamilyContractError(
            "next float32 target receipt profile/authority/generation/task binding differs"
        )
    next_runtime_target = (
        _canonical_float32(
            value.next_task_identity.target_x_m,
            path="placement.next_task_identity.target_x_m",
        ),
        _canonical_float32(
            value.next_task_identity.target_y_m,
            path="placement.next_task_identity.target_y_m",
        ),
    )
    if value.next_target_selection.runtime_target_xy_m != next_runtime_target:
        raise ACFamilyContractError(
            "next C04 task target must equal canonical runtime float32 target bytes"
        )
    for key_name in (
        "env_id",
        "reset_generation",
        "action_uid",
        "action_slot",
        "birth_sha256",
        "run_id",
        "carry_chain_id",
        "source_sha256",
        "config_sha256",
    ):
        if next_shot_key[key_name] != shot_key[key_name]:
            raise ACFamilyContractError(
                "continuous successor must keep the same C05 carry cohort"
            )
    if (
        next_shot_key["swing_generation"] != shot_key["swing_generation"] + 1
        or next_shot_key["shot_index"] != shot_key["shot_index"] + 1
    ):
        raise ACFamilyContractError(
            "continuous successor swing_generation and shot_index must increment once"
        )
    if (
        next_shot_key["sample_sha256"] == shot_key["sample_sha256"]
        or next_shot_key["task_sha256"] == shot_key["task_sha256"]
        or next_shot_key["receipt_content_sha256"]
        == shot_key["receipt_content_sha256"]
        or next_runtime_target == current_runtime_target
    ):
        raise ACFamilyContractError(
            "continuous successor must reveal a newly sampled, different target task"
        )
    if value.next_task_identity.profile_sha256 != profile.canonical_sha256:
        raise ACFamilyContractError(
            "continuous successor must use the same canonical C04 profile"
        )

    try:
        mailbox_view = mailbox.inspect(
            task_key=value.shot_key,
            profile=profile,
            task_identity=value.task_identity,
            source_step=value.strike_fact_tick,
        )
    except (TypeError, ValueError, c05_mailbox.LandingOutcomeMailboxError) as exc:
        raise ACFamilyContractError(
            "placement point is not owned by its original C05 mailbox row"
        ) from exc
    if (
        mailbox_view.state != c05_mailbox.PAID
        or mailbox_view.task_key != value.shot_key
        or mailbox_view.settlement_step != value.settlement_tick
        or mailbox_view.payment_step != value.payment_tick
        or mailbox_view.facts != value.facts
        or mailbox_view.score != value.score
        or mailbox_view.payment_idempotency_sha256 is None
    ):
        raise ACFamilyContractError(
            "C05 mailbox must hold one paid canonical C04 outcome for the original key"
        )
    if mailbox_view.flight_horizon_step - value.strike_fact_tick != flight_horizon_steps:
        raise ACFamilyContractError(
            "C05 row flight horizon must equal the common profile horizon"
        )
    try:
        c06_shot = c06_success.PlacementShotIdentity.from_mailbox_key(
            value.shot_key
        )
        if c06_shot.to_mailbox_key() != value.shot_key:
            raise ValueError("C06 key round-trip differs")
        c05_payment = c05_mailbox.LandingOutcomePayment(
            task_key=value.shot_key,
            profile_sha256=profile.canonical_sha256,
            task_identity_sha256=value.task_identity.canonical_sha256,
            target_x_m=value.task_identity.target_x_m,
            target_y_m=value.task_identity.target_y_m,
            source_step=value.strike_fact_tick,
            settlement_step=value.settlement_tick,
            payment_step=value.payment_tick,
            idempotency_sha256=mailbox_view.payment_idempotency_sha256,
            score=value.score,
        )
        c06_consumer = c06_success.PlacementRewardConsumerReceipt.from_mailbox_payment(
            shot=c06_shot,
            payment=c05_payment,
        )
    except (
        TypeError,
        ValueError,
        c05_mailbox.LandingOutcomeMailboxError,
        c06_success.PlacementSuccessContractError,
    ) as exc:
        raise ACFamilyContractError(
            "placement row must round-trip through the canonical C06 consumer"
        ) from exc

    valid_landing = recomputed.reason in ("scored_on_table", "scored_off_table")
    return {
        "c05_full_shot_key_sha256": value.shot_key.canonical_sha256,
        "canonical_c04_facts_sha256": value.facts.canonical_sha256,
        "canonical_c04_score_sha256": value.score.canonical_sha256,
        "canonical_c04_task_identity_sha256": value.task_identity.canonical_sha256,
        "canonical_float32_target_selection_sha256": (
            value.target_selection.canonical_sha256
        ),
        "canonical_c06_consumer_receipt_sha256": c06_consumer.canonical_sha256,
        "canonical_c06_shot_identity_sha256": c06_shot.canonical_sha256,
        "desired_fact_index": value.desired_fact_index,
        "ball_generation": value.ball_generation,
        "eligible": value.selected_rubber_contact and recomputed.total > 0.0,
        "first_landing_xy": [
            value.facts.first_plane_crossing_x_m,
            value.facts.first_plane_crossing_y_m,
        ],
        "mailbox_payment_idempotency_sha256": (
            mailbox_view.payment_idempotency_sha256
        ),
        "mailbox_slot": value.mailbox_slot,
        "landing_stamp": landing_stamp,
        "net_crossing_stamp": net_stamp,
        "next_precontact_observation_tick": value.next_precontact_observation_tick,
        "next_shot_key": next_shot_key,
        "next_strike_fact_tick": value.next_strike_fact_tick,
        "next_target_reveal_tick": value.next_target_reveal_tick,
        "next_task_identity_sha256": value.next_task_identity.canonical_sha256,
        "next_target_selection_sha256": (
            value.next_target_selection.canonical_sha256
        ),
        "on_table_success": recomputed.on_opponent_table,
        "original_target_xy": [
            value.task_identity.target_x_m,
            value.task_identity.target_y_m,
        ],
        "payment_tick": value.payment_tick,
        "physical_slot": value.physical_slot,
        "physical_retired_at_settlement": True,
        "placement_score": recomputed.total,
        "selected_rubber_contact": value.selected_rubber_contact,
        "selected_contact_stamp": contact_stamp,
        "settlement_tick": value.settlement_tick,
        "shot_key": shot_key,
        "strike_fact_tick": value.strike_fact_tick,
        "valid_landing": valid_landing,
    }


def _resolve_c10_checkpoint(value: C10CheckpointLineage) -> dict[str, object]:
    if type(value) is not C10CheckpointLineage:
        raise ACFamilyContractError("checkpoint must be C10CheckpointLineage")
    if value.parent_abi_sha256 == SUPERSEDED_PORTABLE_ABI_SHA256:
        raise ACFamilyContractError(
            "schema-v2 245/353 checkpoint is incompatible with C10 replacement"
        )
    if type(value.resume_requested) is not bool:
        raise ACFamilyContractError("resume_requested must be exact bool")
    if (
        value.resume_requested
        or value.parent_checkpoint_sha256 is not None
        or value.parent_abi_sha256 is not None
    ):
        raise ACFamilyContractError(
            "C10 widths are unfrozen; only fresh no-parent initialization is admissible"
        )
    blocks = {}
    for name in (
        "initial_state_bytes",
        "optimizer_state_bytes",
        "normalizer_state_bytes",
        "rng_state_bytes",
    ):
        blocks[name.replace("_bytes", "_sha256")] = hashlib.sha256(
            _nonempty_bytes(getattr(value, name), path="checkpoint.%s" % name)
        ).hexdigest()
    return {
        **blocks,
        "parent_abi_sha256": None,
        "parent_checkpoint_sha256": None,
        "resume_requested": False,
        "superseded_checkpoint_compatible": False,
    }


def _resolve_c10_callable(
    value: ResolvedCallableInput,
    *,
    expected_schema: str,
    identity: Mapping[str, str],
    path: str,
) -> dict[str, object]:
    if type(value) is not ResolvedCallableInput:
        raise ACFamilyContractError("%s must be ResolvedCallableInput" % path)
    if value.input_schema_id != expected_schema:
        raise ACFamilyContractError("%s input schema differs" % path)
    return _resolve_callable(value, identity=identity, path=path)


def _reject_c10_current_state_placement_inputs(
    value: ResolvedCallableInput,
    *,
    identity: Mapping[str, str],
    path: str,
) -> None:
    bound = _canonicalize(value.bound_params, path=path + ".bound_params", ancestors=set())
    _reject_runtime_identity(bound, identity=identity, path=path + ".bound_params")
    for item_path, key, item in _walk(bound, path=path + ".bound_params"):
        candidates = []
        if key is not None:
            candidates.append(key)
        if type(item) is str:
            candidates.append(item)
        for candidate in candidates:
            if _c10_forbidden_placement_token(candidate):
                raise ACFamilyContractError(
                    "%s reads current target/command metric instead of original-shot mailbox"
                    % item_path
                )
    dependencies = _function_dependency_closure(
        value.function,
        explicit=value.dependency_functions,
        identity=identity,
        path=path,
    )
    root_parameters = tuple(inspect.signature(value.function).parameters.values())
    if (
        len(root_parameters) != 1
        or root_parameters[0].name != "mailbox_entry"
        or root_parameters[0].kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ):
        raise ACFamilyContractError(
            "%s placement mailbox reader must accept exactly one original-shot mailbox_entry"
            % path
        )
    for dependency in dependencies:
        signature = inspect.signature(dependency)
        for parameter in signature.parameters.values():
            if _c10_forbidden_placement_token(parameter.name):
                raise ACFamilyContractError(
                    "%s placement callable signature reads current target/command metric"
                    % path
                )
        try:
            source = inspect.getsource(dependency)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError) as exc:
            raise ACFamilyContractError(
                "%s placement callable source is unavailable" % path
            ) from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                raise ACFamilyContractError(
                    "%s placement callable uses dynamic string-key construction" % path
                )
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add)
                and _ast_contains_string_literal(node)
                and _ast_static_string(node) is None
            ):
                raise ACFamilyContractError(
                    "%s placement callable uses dynamic string-key construction" % path
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("format", "format_map", "join")
            ):
                raise ACFamilyContractError(
                    "%s placement callable uses dynamic string-key construction" % path
                )
            token = None
            if isinstance(node, ast.Name):
                token = node.id
            elif isinstance(node, ast.Attribute):
                token = node.attr
            elif isinstance(node, ast.keyword) and node.arg is not None:
                token = node.arg
            elif isinstance(node, ast.Constant) and type(node.value) is str:
                token = node.value
            if token is not None and _c10_forbidden_placement_token(token):
                raise ACFamilyContractError(
                    "%s placement callable source reads current target/command metric"
                    % path
                )
            static_text = _ast_static_string(node)
            if (
                static_text is not None
                and _c10_forbidden_placement_token(static_text)
            ):
                raise ACFamilyContractError(
                    "%s placement callable constructs current target/command metric key"
                    % path
                )


def _c10_forbidden_placement_token(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if normalized in _C10_FORBIDDEN_PLACEMENT_TOKENS:
        return True
    return (
        normalized.startswith("current_target")
        or normalized.startswith("current_command")
        or normalized.endswith("command_metric")
    )


def _c10_treatment(*, family: str, gain: object) -> dict[str, object]:
    resolved_gain = _finite_scalar(
        gain,
        path="post_contact_placement_treatment_gain",
    )
    expected_gain = 1.0 if family == "A" else 0.0
    if resolved_gain != expected_gain:
        raise ACFamilyContractError(
            "post-contact placement treatment gain must be A=1 and C=0"
        )
    return {
        "post_contact_placement_treatment_gain": resolved_gain,
        "scope": "post_contact_original_c05_mailbox_payment_gain_only_v1",
    }


def _c10_treatment_witness(
    *,
    family: str,
    manager_weight: float,
    gain: float,
    points: Tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    payments = []
    for point in points:
        eligible = point["eligible"] is True
        score = float(point["placement_score"])
        payments.append(manager_weight * gain * score if eligible else 0.0)
    if family == "A" and not any(item > 0.0 for item in payments):
        raise ACFamilyContractError(
            "family A tape must witness nonzero post-contact placement payment"
        )
    if family == "C" and any(item != 0.0 for item in payments):
        raise ACFamilyContractError("family C placement payment must remain zero")
    return {
        "eligible_original_shot_count": sum(
            point["eligible"] is True for point in points
        ),
        "guidance_payment_all_zero": all(item == 0.0 for item in payments),
        "guidance_payment_nonzero_count": sum(item != 0.0 for item in payments),
        "guidance_payment_sha256": canonical_sha256(payments),
        "payment_stage": "after_original_shot_mailbox_settlement_v1",
        "post_contact_placement_consumer_scheduled": True,
        "post_contact_placement_manager_weight": manager_weight,
        "post_contact_placement_treatment_gain": gain,
        "selected_rubber_contact_required": True,
    }


def _owned_c10_projection(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not C10ResolvedProjection or value._token is not _C10_BUILD_TOKEN:
        raise ACFamilyContractError(
            "%s projection must come from build_c10_family_projection" % label
        )
    expected_auth_tag = hmac.new(
        _C10_BUILD_AUTH_KEY,
        value._payload_json,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(value._auth_tag, expected_auth_tag):
        raise ACFamilyContractError("%s C10 builder authentication failed" % label)
    rebuilt = build_c10_family_projection(value._source_inputs)
    if not hmac.compare_digest(rebuilt._payload_json, value._payload_json):
        raise ACFamilyContractError(
            "%s C10 projection differs from full typed-input revalidation" % label
        )
    payload = value.to_mapping()
    expected = frozenset(
        {
            "abi_state",
            "backend",
            "checkpoint_lineage",
            "common_runtime",
            "contract_authority_kind",
            "contract_authority_sha256",
            "evidence_level",
            "family",
            "identity",
            "kind",
            "launch_gate_ready",
            "schema_version",
            "treatment",
            "treatment_witness",
        }
    )
    if frozenset(payload) != expected:
        raise ACFamilyContractError("C10 projection schema drifted")
    if payload["schema_version"] != C10_SCHEMA_VERSION:
        raise ACFamilyContractError("C10 projection version drifted")
    if payload["kind"] != "action_ball_matched_ac_post_contact_placement_c10_v1":
        raise ACFamilyContractError("C10 projection kind drifted")
    if payload["launch_gate_ready"] is not False:
        raise ACFamilyContractError("C10 projection cannot claim launch readiness")
    if payload["evidence_level"] != C10_EVIDENCE_LEVEL:
        raise ACFamilyContractError("C10 projection evidence level drifted")
    if (
        payload["contract_authority_kind"] != C10_CONTRACT_AUTHORITY_KIND
        or payload["contract_authority_sha256"] != C10_CONTRACT_AUTHORITY_SHA256
    ):
        raise ACFamilyContractError("C10 parent contract authority drifted")
    return payload


def _nonempty_bytes_tuple(value: object, *, path: str) -> Tuple[bytes, ...]:
    if type(value) is not tuple or not value:
        raise ACFamilyContractError("%s must be a non-empty exact tuple" % path)
    return tuple(
        _nonempty_bytes(item, path="%s[%d]" % (path, index))
        for index, item in enumerate(value)
    )


def _finite_tuple(
    value: object,
    *,
    path: str,
    expected_size: int,
) -> Tuple[float, ...]:
    if type(value) is not tuple or len(value) != expected_size:
        raise ACFamilyContractError(
            "%s must contain exactly %d values" % (path, expected_size)
        )
    return tuple(
        _finite_scalar(item, path="%s[%d]" % (path, index))
        for index, item in enumerate(value)
    )


def _sha256_text(value: object, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ACFamilyContractError("%s must be lowercase SHA-256 hex" % path)
    return value


C10_FAMILY_CONTRACT_MAPPING = {
    "actor_width": None,
    "allowed_family_axis": (
        "common_positive_manager_weight_family_treatment_gain_a1_c0_v1"
    ),
    "common_ordered_strike_reward_terms": list(C10_CONTACT_REWARD_TERM_NAMES),
    "common_ordered_strike_reward_order_sha256": (
        C10_STRIKE_FACT_CONSUMER_ORDER_SHA256
    ),
    "common_ordered_strike_reward_source_sha256": (
        C10_STRIKE_FACT_AUTHORITY_SOURCE_SHA256
    ),
    "continuous_target_runtime_dtype": c03_successor.RUNTIME_TARGET_DTYPE,
    "continuous_target_semantics_source_sha256": (
        C10_RUNTIME_TARGET_AUTHORITY_SOURCE_SHA256
    ),
    "landing_placement_authority_source_sha256": (
        C10_LANDING_PLACEMENT_AUTHORITY_SOURCE_SHA256
    ),
    "landing_placement_torch_authority_source_sha256": (
        C10_LANDING_PLACEMENT_TORCH_AUTHORITY_SOURCE_SHA256
    ),
    "landing_placement_torch_authority_test_sha256": (
        C10_LANDING_PLACEMENT_TORCH_AUTHORITY_TEST_SHA256
    ),
    "landing_outcome_mailbox_authority_source_sha256": (
        C10_LANDING_OUTCOME_MAILBOX_AUTHORITY_SOURCE_SHA256
    ),
    "placement_success_authority_source_sha256": (
        C10_PLACEMENT_SUCCESS_AUTHORITY_SOURCE_SHA256
    ),
    "placement_dual_consumers": list(C10_PLACEMENT_CONSUMER_NAMES),
    "placement_profile_parameters_status": "UNFROZEN_UNTIL_R06_BUDGET_RECEIPT",
    "reward_manager_weights_status": "UNFROZEN_UNTIL_POST_DT_BUDGET_RECEIPT",
    "physics_stamp_schema": {
        "control_step": "int64",
        "physics_substep": "int32",
        "event_phase": "uint8",
        "ordered_phases": ["CONTACT", "NET", "LANDING"],
    },
    "physics_stamp_runtime_ready": False,
    "capacity_profile_fields": [
        "flight_horizon_steps",
        "flight_slot_capacity",
        "mailbox_slot_capacity",
    ],
    "slot_transfer_semantics": (
        "flight_inbound_open_atomic_settle_copy_to_mailbox_"
        "and_retire_physical_v1"
    ),
    "flight_slot_ownership_states": ["INBOUND", "OPEN"],
    "mailbox_slot_ownership_states": [
        "SETTLED_UNPAID",
        "PARTIALLY_PAID",
        "PAID",
    ],
    "slot_capacity_runtime_ready": False,
    "midsequence_checkpoint_runtime_ready": False,
    "midsequence_state_set": list(C10_MIDSEQUENCE_STATES),
    "fixed_tape_c05_identity_scope": (
        "shared_paired_replay_id_distinct_from_required_live_job_id_v2"
    ),
    "critic_width": None,
    "common_semantic_roles": [
        "actor_critic_observation_fields_and_values",
        "observation_provider",
        "desired_at_contact_position_velocity_face",
        "normalizer_semantics_and_state",
        "question_provider_and_full_receipt",
        "selected_rubber_contact_full_shot_key_independent_milestone",
        "ten_same_transition_one_shot_strike_payments_on_hit_and_miss",
        "on_table_success_scorer_weight_denominator",
        "question_curriculum",
        "original_shot_bounded_c05_mailbox_canonical_c04_profile_task_facts_score",
        "late_settlement_payment_keeps_original_key_across_successor_reveal",
        "common_positive_placement_manager_weight_and_scheduled_consumer",
        "c06_common_outcome_and_raw_placement_dual_consumers",
        "plant_ppo_recovery",
    ],
    "constructed_runtime_required_receipts": [
        "contract_authority_sha256",
        "post_dt_budget_receipt_sha256",
    ],
    "constructed_runtime_self_report_sufficient": False,
    "kind": C10_CONTRACT_AUTHORITY_KIND,
    "legacy_211_319_admissible": False,
    "observation_width_status": C10_OBSERVATION_WIDTH_STATUS,
    "evidence_level": C10_EVIDENCE_LEVEL,
    "on_table_success_semantics": C10_ON_TABLE_SUCCESS_SEMANTICS,
    "placement_eligibility": C10_PLACEMENT_ELIGIBILITY,
    "placement_source_semantics": C10_PLACEMENT_SOURCE_SEMANTICS,
    "precontact_normalizer_semantics": C10_PRECONTACT_NORMALIZER_SEMANTICS,
    "schema_version": C10_SCHEMA_VERSION,
    "sealed_content_addressed_parent": True,
    "superseded_checkpoint_compatible": False,
    "superseded_portable_abi_sha256": SUPERSEDED_PORTABLE_ABI_SHA256,
}
C10_FAMILY_CONTRACT_SHA256 = canonical_sha256(C10_FAMILY_CONTRACT_MAPPING)
C10_CONTRACT_AUTHORITY_PAYLOAD = C10_FAMILY_CONTRACT_MAPPING
C10_CONTRACT_AUTHORITY_SHA256 = C10_FAMILY_CONTRACT_SHA256


__all__ = (
    "ACFamilyContractError",
    "ACFamilyPairValidation",
    "ACTOR_GUIDE_SLICE",
    "ACTOR_GUIDE_VALID_INDEX",
    "ACTOR_LAYOUT",
    "ACTOR_NET_CLEAR_INDEX",
    "ACTOR_NET_CROSSED_INDEX",
    "ACTOR_WIDTH",
    "BACKENDS",
    "C10_ACTOR_WIDTH",
    "C10_ALLOWED_DELTA_PATHS",
    "C10_CALLABLE_NONINTERFERENCE_READY",
    "C10_COMMON_CALLABLE_INPUT_SCHEMAS",
    "C10_CONSTRUCTED_RUNTIME_READY",
    "C10_CONTRACT_AUTHORITY_KIND",
    "C10_CONTRACT_AUTHORITY_PAYLOAD",
    "C10_CONTRACT_AUTHORITY_SHA256",
    "C10_CONTACT_PAYMENT_SEMANTICS",
    "C10_CONTACT_REWARD_INPUT_SCHEMA",
    "C10_CONTACT_REWARD_TERM_NAMES",
    "C10_CONTRACT_STATUS",
    "C10_CRITIC_WIDTH",
    "C10_EVIDENCE_LEVEL",
    "C10CheckpointLineage",
    "C10ContactRewardTerm",
    "C10DesiredAtContactFact",
    "C10FlightSlotOwner",
    "C10PhysicsStamp",
    "C10FlightSlotCapacityWitness",
    "C10MidsequenceStateSnapshot",
    "C10FamilyPairValidation",
    "C10FixedTape",
    "C10PlacementSource",
    "C10PlacementTapePoint",
    "C10ResolvedProjection",
    "C10ResolvedRuntimeInputs",
    "C10UnfrozenObservationContract",
    "C10CommonRuntime",
    "C10_FAMILY_CONTRACT_MAPPING",
    "C10_FAMILY_CONTRACT_SHA256",
    "C10_FIXED_TAPE_VALUE_READY",
    "C10_LAUNCH_GATE_READY",
    "C10_LANDING_OUTCOME_MAILBOX_AUTHORITY_SOURCE_SHA256",
    "C10_LANDING_PLACEMENT_AUTHORITY_SOURCE_SHA256",
    "C10_LANDING_PLACEMENT_TORCH_AUTHORITY_SOURCE_SHA256",
    "C10_LANDING_PLACEMENT_TORCH_AUTHORITY_TEST_SHA256",
    "C10_PLACEMENT_CONSUMER_NAMES",
    "C10_PLACEMENT_SUCCESS_AUTHORITY_SOURCE_SHA256",
    "C10_OBSERVATION_WIDTH_STATUS",
    "C10_ON_TABLE_SUCCESS_SEMANTICS",
    "C10_PLACEMENT_ELIGIBILITY",
    "C10_PLACEMENT_SOURCE_SEMANTICS",
    "C10_PRECONTACT_NORMALIZER_SEMANTICS",
    "C10_EVENT_PHASE_CONTACT",
    "C10_EVENT_PHASE_INBOUND",
    "C10_EVENT_PHASE_LANDING",
    "C10_EVENT_PHASE_NET",
    "C10_MIDSEQUENCE_STATES",
    "C10_PHYSICS_STAMP_RUNTIME_READY",
    "C10_SLOT_CAPACITY_RUNTIME_READY",
    "C10_MIDSEQUENCE_CHECKPOINT_RUNTIME_READY",
    "C10_RUNTIME_TARGET_AUTHORITY_SOURCE_SHA256",
    "C10_SCHEMA_VERSION",
    "C10_STRIKE_FACT_AUTHORITY_SOURCE_SHA256",
    "C10_STRIKE_FACT_CONSUMER_ORDER_SHA256",
    "C10_TYPED_FAMILY_SCHEMA_READY",
    "CALLABLE_NONINTERFERENCE_READY",
    "CHECKPOINT_RUNTIME_EXTRACTOR_READY",
    "COMPATIBILITY_245_353_LAYOUT_EXPORTS_ONLY",
    "CONSTRUCTED_ENV_EXTRACTOR_READY",
    "CRITIC_GUIDE_SLICE",
    "CRITIC_GUIDE_VALID_INDEX",
    "CRITIC_LAYOUT",
    "CRITIC_NET_CLEAR_INDEX",
    "CRITIC_NET_CROSSED_INDEX",
    "CRITIC_WIDTH",
    "CommonRuntimeCallables",
    "DESIRED_AT_CONTACT_COMPONENTS",
    "DESIRED_AT_CONTACT_COMPONENT_WIDTHS",
    "DESIRED_AT_CONTACT_SOURCE_SEMANTICS",
    "DISABLED_GUIDANCE_FILL",
    "DISABLED_GUIDANCE_MASK_STAGE",
    "FAMILIES",
    "FIXED_TAPE_CONTEXTS",
    "FIXED_TAPE_PHASES",
    "FIXED_TAPE_POINTS",
    "FIXED_TAPE_SCHEMA_READY",
    "FIXED_TAPE_VALUE_WITNESS_READY",
    "FixedTapeBytes",
    "FreshCheckpointState",
    "FreshNormalizerState",
    "GUIDANCE_ALLOWED_DELTA_PATHS",
    "IDENTITY_ONLY_FIELDS",
    "IDENTITY_ALLOWED_DELTA_PATHS",
    "LAUNCH_GATE_READY",
    "NORMALIZER_REEXECUTION_READY",
    "OutcomeAtomicityWitness",
    "PAIR_ALLOWED_DELTA_PATHS",
    "PORTABLE_ABI_MAPPING",
    "PORTABLE_ABI_SHA256",
    "RESOLVED_OBJECT_BUILDER_READY",
    "RUNTIME_TEMPORAL_WITNESS_READY",
    "ResolvedACProjection",
    "ResolvedCallableInput",
    "ResolvedObservationGroup",
    "ResolvedObservationTerm",
    "ResolvedRuntimeInputs",
    "SCHEMA_VERSION",
    "SUPERSEDED_ACTOR_WIDTH",
    "SUPERSEDED_C07_ACTOR_WIDTH",
    "SUPERSEDED_C07_CRITIC_WIDTH",
    "SUPERSEDED_C07_PORTABLE_ABI_SHA256",
    "SUPERSEDED_CRITIC_WIDTH",
    "SUPERSEDED_PORTABLE_ABI_SHA256",
    "TYPED_SCHEMA_READY",
    "build_c10_family_projection",
    "build_resolved_runtime_projection",
    "c10_projection_artifact",
    "c10_midsequence_checkpoint_candidate_root",
    "canonical_sha256",
    "projection_artifact",
    "validate_c10_family_pair",
    "validate_ac_family_pair",
)
