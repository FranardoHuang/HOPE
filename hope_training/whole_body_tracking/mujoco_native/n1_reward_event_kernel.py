"""Fail-closed, pure N1 reward/event eligibility kernel.

This module is deliberately *not* a MuJoCo callback, a trajectory predictor,
or a PPO reward implementation.  It turns already-observed, source-bound
facts into the four denominator/payout gates required by the native-N1
readiness contract:

``motion mimic -> A contact target -> actual selected-rubber hit -> achieved
outgoing flight -> predicted outcome -> observed outcome``.

In particular, it never infers a contact from a target-window match and never
infers an outcome from a desired target.  Callers must provide their own
physics-event and flight/outcome authorities.  Missing or contradictory facts
raise :class:`N1RewardEventKernelError`; they are not treated as a miss.
"""

from __future__ import annotations

import math
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from . import observed_outcome_resolver
from . import selected_rubber_classifier


N1_REWARD_EVENT_KERNEL_KIND = "a3_mujoco_n1_pure_reward_event_kernel_v1"
NATIVE_PHYSICAL_EVENT_FACTS_KIND = "a3_mujoco_n1_physical_event_facts_v4"
NATIVE_PHYSICAL_EVENT_FACTS_CONTRACT_KIND = (
    "a3_mujoco_n1_physical_event_facts_contract_v4"
)
NATIVE_CONTACT_INVALID_REASONS = (
    "racket_contact_between_outer_planes_ambiguous",
    "racket_contact_edge_or_rim_ambiguous",
    "racket_contact_simultaneous_with_other",
    "racket_recontact",
)
EXPECTED_OBSERVED_OUTCOME_RESOLVER_SOURCE_SHA256 = (
    "c1b40201ab965650f68f903ff8684769b6a7b97ddad9c9018c27b4e8088af575"
)
EXPECTED_N1_BALL_CORE_SOURCE_SHA256 = (
    "49da56a1f5c795777f3dc5f2291a72b2f3a85edd09f5b2b0004a87f1336ed32d"
)
EXPECTED_PHYSICAL_BALL_SCENE_SOURCE_SHA256 = (
    "8b78f8ba80a60e06e3cbb67701400e21114bbee4cc82bf6a72ffce76fb8e9b01"
)
EXPECTED_TABLE_SCENE_SOURCE_SHA256 = (
    "db382094674ee5e290f980164b01ad10ece676c45be6496525e4609399213ee2"
)


class N1RewardEventKernelError(ValueError):
    """The caller supplied incomplete, unordered, or non-finite event facts."""


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SourceBinding:
    """Opaque identity of the producer whose facts this kernel may consume.

    The kernel intentionally does not open files or import the producer: the
    integration layer supplies the expected binding from its pinned receipt and
    must match it exactly on every call.
    """

    source_id: str
    source_sha256: str
    event_contract_sha256: str


@dataclass(frozen=True, order=True)
class EventStamp:
    """A physics-event position; order is ``(policy_tick, physics_substep)``."""

    policy_tick: int
    physics_substep: int


@dataclass(frozen=True)
class ContactEvidence:
    """Observed ball/racket contact, not a target-window proxy."""

    occurred: bool
    stamp: Optional[EventStamp]
    selected_rubber: bool


@dataclass(frozen=True)
class OutgoingFlightEvidence:
    """First contact-free ball state after a valid actual contact."""

    valid: bool
    stamp: Optional[EventStamp]
    position_w_m: Optional[Vector3]
    linear_velocity_w_mps: Optional[Vector3]
    spin_w_radps: Optional[Vector3]


@dataclass(frozen=True)
class PredictedOutcomeEvidence:
    """A predictor result evaluated from the achieved outgoing flight only."""

    evaluated: bool
    predicted_net_clear: Optional[bool]
    predicted_legal_landing: Optional[bool]


@dataclass(frozen=True)
class ObservedOutcomeEvidence:
    """Native physical outcome resolved after the outgoing flight."""

    resolved: bool
    stamp: Optional[EventStamp]
    observed_net_clear: Optional[bool]
    observed_legal_landing: Optional[bool]


@dataclass(frozen=True)
class SwingClosureEvidence:
    """The per-swing close event used for the hit denominator."""

    closed: bool
    stamp: Optional[EventStamp]
    timeout: bool


@dataclass(frozen=True)
class N1RewardEventInput:
    """All facts needed for one N1 swing; all booleans are explicit evidence."""

    source: SourceBinding
    motion_mimic_eligible: bool
    target_valid: bool
    strike_window: bool
    actual_contact: ContactEvidence
    outgoing_flight: OutgoingFlightEvidence
    predicted_outcome: PredictedOutcomeEvidence
    observed_outcome: ObservedOutcomeEvidence
    swing_closure: SwingClosureEvidence


@dataclass(frozen=True)
class N1RewardEligibility:
    """Boolean increments/gates; this kernel assigns no reward magnitudes."""

    motion_mimic_denominator: bool
    contact_target_denominator: bool
    closed_swing_denominator: bool
    actual_contact_numerator: bool
    achieved_outgoing_flight_denominator: bool
    predicted_outcome_denominator: bool
    predicted_net_clear_numerator: bool
    predicted_legal_landing_numerator: bool
    observed_outcome_denominator: bool
    observed_net_clear_numerator: bool
    observed_legal_landing_numerator: bool
    unresolved_achieved_flight: bool
    motion_mimic_pay_eligible: bool
    contact_target_pay_eligible: bool
    actual_contact_pay_eligible: bool
    predicted_outcome_pay_eligible: bool
    observed_outcome_pay_eligible: bool


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def native_physical_event_facts_contract() -> dict[str, Any]:
    """Return the strict core-to-VecEnv ABI for observed physical facts.

    Selected-rubber identity is conditional on an exact per-question action
    lineage.  The ABI still cannot by itself authorize reward payment.
    """

    payload = {
        "schema_version": 4,
        "kind": NATIVE_PHYSICAL_EVENT_FACTS_CONTRACT_KIND,
        "sample_kind": NATIVE_PHYSICAL_EVENT_FACTS_KIND,
        "sample_keys": [
            "schema_version",
            "kind",
            "source",
            "policy_tick",
            "racket_contact_edge_count_total",
            "first_racket_contact_stamp",
            "outgoing_flight",
            "invalid_reasons",
            "selected_rubber_authority_available",
            "selected_rubber_action_lineage",
            "first_racket_contact_classification",
            "observed_outcome_authority_available",
            "observed_outcome_resolver_binding",
            "observed_outcome_question_binding",
            "observed_outcome_snapshot",
        ],
        "source_keys": [
            "source_id",
            "source_sha256",
            "event_contract_sha256",
        ],
        "outgoing_flight_keys": [
            "policy_tick",
            "physics_substep",
            "time_s",
            "position_w_m",
            "linear_velocity_w_mps",
            "spin_w_radps",
            "semantic",
        ],
        "invalid_reasons": list(NATIVE_CONTACT_INVALID_REASONS),
        "selected_rubber_classifier_available": True,
        "selected_rubber_authority_semantics": (
            "per_question_exact_action_manifest_mount_scene_backend_lineage"
        ),
        "selected_rubber_classifier_kind": (
            selected_rubber_classifier.CLASSIFICATION_KIND
        ),
        "selected_rubber_classification_statuses": list(
            selected_rubber_classifier.CLASSIFICATION_STATUSES
        ),
        "generic_blade_contact_receipt_preserved": True,
        "observed_outcome_resolver_available": True,
        "observed_outcome_resolver_kind": (
            observed_outcome_resolver.SNAPSHOT_KIND
        ),
        "observed_outcome_semantics": (
            "source_bound_native_substep_net_crossing_and_first_table_landing;"
            "no_prediction_and_no_reward"
        ),
        "reward_authorized": False,
    }
    payload["content_sha256"] = _sha256_json(payload)
    return payload


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise N1RewardEventKernelError("%s must be bool" % name)
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise N1RewardEventKernelError("%s must be lowercase SHA-256" % name)
    return value


def _validate_source(source: SourceBinding, name: str) -> None:
    if type(source) is not SourceBinding:
        raise N1RewardEventKernelError("%s must be SourceBinding" % name)
    if type(source.source_id) is not str or not source.source_id.strip():
        raise N1RewardEventKernelError("%s.source_id must be a non-empty string" % name)
    _require_sha256(source.source_sha256, "%s.source_sha256" % name)
    _require_sha256(source.event_contract_sha256, "%s.event_contract_sha256" % name)


def _validate_stamp(stamp: Optional[EventStamp], name: str, required: bool) -> None:
    if stamp is None:
        if required:
            raise N1RewardEventKernelError("%s is required" % name)
        return
    if type(stamp) is not EventStamp:
        raise N1RewardEventKernelError("%s must be EventStamp or None" % name)
    if not _is_plain_int(stamp.policy_tick) or stamp.policy_tick < 0:
        raise N1RewardEventKernelError("%s.policy_tick must be a non-negative plain int" % name)
    if not _is_plain_int(stamp.physics_substep) or stamp.physics_substep < 0:
        raise N1RewardEventKernelError(
            "%s.physics_substep must be a non-negative plain int" % name
        )


def _validate_vector(value: Optional[Vector3], name: str, required: bool) -> None:
    if value is None:
        if required:
            raise N1RewardEventKernelError("%s is required" % name)
        return
    if type(value) is not tuple or len(value) != 3:
        raise N1RewardEventKernelError("%s must be a length-3 tuple" % name)
    for index, scalar in enumerate(value):
        if isinstance(scalar, bool):
            raise N1RewardEventKernelError("%s[%d] cannot be bool" % (name, index))
        try:
            finite = math.isfinite(float(scalar))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise N1RewardEventKernelError("%s[%d] must be finite" % (name, index))


def _source_mapping(source: SourceBinding) -> dict[str, str]:
    _validate_source(source, "source")
    return {
        "source_id": source.source_id,
        "source_sha256": source.source_sha256,
        "event_contract_sha256": source.event_contract_sha256,
    }


def _stamp_from_mapping(value: object, name: str) -> EventStamp | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "policy_tick",
        "physics_substep",
    }:
        raise N1RewardEventKernelError("%s keys differ from EventStamp" % name)
    stamp = EventStamp(
        policy_tick=value["policy_tick"],
        physics_substep=value["physics_substep"],
    )
    _validate_stamp(stamp, name, True)
    return stamp


def validate_native_physical_event_facts(
    sample: Mapping[str, Any],
    *,
    expected_source: SourceBinding,
    expected_outcome_resolver_binding_sha256: str | None = None,
    expected_outcome_question_binding_sha256: str | None = None,
    expected_outcome_scene_binding_sha256: str | None = None,
    expected_outcome_plant_binding_sha256: str | None = None,
    expected_question_source_sha256: str | None = None,
    expected_question_landing_aim_xy_w_m: tuple[float, float] | None = None,
    expected_resolver_source_sha256: str = (
        EXPECTED_OBSERVED_OUTCOME_RESOLVER_SOURCE_SHA256
    ),
) -> dict[str, Any]:
    """Validate and canonicalize one cumulative native physical-event sample."""

    contract = native_physical_event_facts_contract()
    if not isinstance(sample, Mapping) or set(sample) != set(
        contract["sample_keys"]
    ):
        raise N1RewardEventKernelError(
            "native physical event fact keys differ from exact ABI"
        )
    if (
        sample["schema_version"] != 4
        or sample["kind"] != NATIVE_PHYSICAL_EVENT_FACTS_KIND
    ):
        raise N1RewardEventKernelError(
            "native physical event fact kind/schema differs"
        )
    expected_source_mapping = _source_mapping(expected_source)
    if sample["source"] != expected_source_mapping:
        raise N1RewardEventKernelError(
            "native physical event source binding differs from authority"
        )
    policy_tick = sample["policy_tick"]
    edge_count = sample["racket_contact_edge_count_total"]
    if not _is_plain_int(policy_tick) or policy_tick < 0:
        raise N1RewardEventKernelError("policy_tick must be non-negative plain int")
    if not _is_plain_int(edge_count) or edge_count < 0:
        raise N1RewardEventKernelError(
            "racket_contact_edge_count_total must be non-negative plain int"
        )
    first_stamp = _stamp_from_mapping(
        sample["first_racket_contact_stamp"], "first_racket_contact_stamp"
    )
    if (edge_count == 0) != (first_stamp is None):
        raise N1RewardEventKernelError(
            "racket contact edge count and first stamp disagree"
        )
    if first_stamp is not None and first_stamp.policy_tick > policy_tick:
        raise N1RewardEventKernelError("first racket contact stamp is in the future")
    invalid_reasons = sample["invalid_reasons"]
    if (
        type(invalid_reasons) is not list
        or invalid_reasons != sorted(set(invalid_reasons))
        or any(value not in NATIVE_CONTACT_INVALID_REASONS for value in invalid_reasons)
    ):
        raise N1RewardEventKernelError(
            "native physical event invalid reasons are not canonical"
        )
    authority_available = _require_bool(
        sample["selected_rubber_authority_available"],
        "selected_rubber_authority_available",
    )
    lineage = sample["selected_rubber_action_lineage"]
    classification = sample["first_racket_contact_classification"]
    if authority_available:
        try:
            lineage = selected_rubber_classifier.validate_action_lineage_seal(
                lineage
            )
        except selected_rubber_classifier.SelectedRubberClassifierError as exc:
            raise N1RewardEventKernelError(
                "selected-rubber action lineage is invalid"
            ) from exc
        if (edge_count == 0) != (classification is None):
            raise N1RewardEventKernelError(
                "selected-rubber classification and generic contact count disagree"
            )
        if classification is not None:
            try:
                classification = (
                    selected_rubber_classifier.validate_classification_seal(
                        classification, action_lineage=lineage
                    )
                )
            except selected_rubber_classifier.SelectedRubberClassifierError as exc:
                raise N1RewardEventKernelError(
                    "selected-rubber contact classification is invalid"
                ) from exc
            classification_stamp = EventStamp(
                classification["policy_tick"],
                classification["physics_substep"],
            )
            if classification_stamp != first_stamp:
                raise N1RewardEventKernelError(
                    "selected-rubber classification stamp differs from first contact"
                )
            expected_ambiguity_reason = {
                selected_rubber_classifier.STATUS_EDGE_RIM_AMBIGUOUS: (
                    "racket_contact_edge_or_rim_ambiguous"
                ),
                selected_rubber_classifier.STATUS_BETWEEN_PLANES_AMBIGUOUS: (
                    "racket_contact_between_outer_planes_ambiguous"
                ),
            }.get(classification["status"])
            if (expected_ambiguity_reason is not None) != (
                expected_ambiguity_reason in invalid_reasons
            ):
                raise N1RewardEventKernelError(
                    "selected-rubber ambiguity status/reason disagree"
                )
    elif lineage is not None or classification is not None:
        raise N1RewardEventKernelError(
            "unavailable selected-rubber authority cannot carry lineage/classification"
        )
    outgoing = sample["outgoing_flight"]
    if outgoing is not None:
        if not isinstance(outgoing, Mapping) or set(outgoing) != set(
            contract["outgoing_flight_keys"]
        ):
            raise N1RewardEventKernelError("native outgoing flight keys differ")
        outgoing_stamp = EventStamp(
            outgoing["policy_tick"], outgoing["physics_substep"]
        )
        _validate_stamp(outgoing_stamp, "outgoing_flight.stamp", True)
        if first_stamp is None:
            raise N1RewardEventKernelError(
                "native outgoing flight requires an observed racket contact"
            )
        _require_strictly_after(
            outgoing_stamp, first_stamp, "native outgoing flight"
        )
        if outgoing_stamp.policy_tick > policy_tick:
            raise N1RewardEventKernelError("native outgoing flight is in the future")
        time_s = outgoing["time_s"]
        if isinstance(time_s, bool):
            raise N1RewardEventKernelError("outgoing_flight.time_s must be finite")
        try:
            finite_time = math.isfinite(float(time_s)) and float(time_s) >= 0.0
        except (TypeError, ValueError):
            finite_time = False
        if not finite_time:
            raise N1RewardEventKernelError(
                "outgoing_flight.time_s must be finite and non-negative"
            )
        for field in (
            "position_w_m",
            "linear_velocity_w_mps",
            "spin_w_radps",
        ):
            value = outgoing[field]
            if type(value) is not list:
                raise N1RewardEventKernelError(
                    "outgoing_flight.%s must be a JSON vector" % field
                )
            _validate_vector(tuple(value), "outgoing_flight.%s" % field, True)
        if outgoing["semantic"] != (
            "first_contact_free_physics_substep_after_first_racket_contact"
        ):
            raise N1RewardEventKernelError("native outgoing flight semantic differs")
    outcome_authority_available = _require_bool(
        sample["observed_outcome_authority_available"],
        "observed_outcome_authority_available",
    )
    outcome_binding = sample["observed_outcome_resolver_binding"]
    outcome_question = sample["observed_outcome_question_binding"]
    outcome_snapshot = sample["observed_outcome_snapshot"]
    if outcome_authority_available:
        if (
            expected_outcome_resolver_binding_sha256 is None
            or expected_outcome_question_binding_sha256 is None
            or expected_outcome_scene_binding_sha256 is None
            or expected_outcome_plant_binding_sha256 is None
            or expected_question_source_sha256 is None
            or expected_question_landing_aim_xy_w_m is None
        ):
            raise N1RewardEventKernelError(
                "observed-outcome authority requires external resolver/question parents"
            )
        expected_resolver_sha = _require_sha256(
            expected_outcome_resolver_binding_sha256,
            "expected observed-outcome resolver binding SHA",
        )
        expected_question_sha = _require_sha256(
            expected_outcome_question_binding_sha256,
            "expected observed-outcome question binding SHA",
        )
        expected_scene_sha = _require_sha256(
            expected_outcome_scene_binding_sha256,
            "expected observed-outcome scene binding SHA",
        )
        expected_plant_sha = _require_sha256(
            expected_outcome_plant_binding_sha256,
            "expected observed-outcome plant binding SHA",
        )
        expected_question_source_sha = _require_sha256(
            expected_question_source_sha256,
            "expected observed-outcome question source SHA",
        )
        expected_resolver_source_sha = _require_sha256(
            expected_resolver_source_sha256,
            "expected observed-outcome resolver source SHA",
        )
        expected_aim = tuple(expected_question_landing_aim_xy_w_m)
        if len(expected_aim) != 2 or any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in expected_aim
        ):
            raise N1RewardEventKernelError(
                "expected observed-outcome landing aim must be two finite values"
            )
        try:
            outcome_binding = (
                observed_outcome_resolver.validate_resolver_binding_seal(
                outcome_binding
                )
            )
            outcome_question = (
                observed_outcome_resolver.validate_question_binding_seal(
                    outcome_question
                )
            )
            outcome_snapshot = observed_outcome_resolver.validate_snapshot(
                outcome_snapshot,
                question_binding=outcome_question,
                resolver_binding=outcome_binding,
                expected_question_binding_sha256=expected_question_sha,
                expected_resolver_binding_sha256=expected_resolver_sha,
            )
        except observed_outcome_resolver.ObservedOutcomeResolverError as exc:
            raise N1RewardEventKernelError(
                "observed-outcome authority or snapshot is invalid"
            ) from exc
        if (
            outcome_binding["content_sha256"] != expected_resolver_sha
            or outcome_question["content_sha256"] != expected_question_sha
            or outcome_binding["scene_binding_sha256"] != expected_scene_sha
            or outcome_binding["plant_binding_sha256"] != expected_plant_sha
            or outcome_binding["resolver_source_sha256"]
            != expected_resolver_source_sha
            or outcome_binding["physical_ball_scene_source_sha256"]
            != EXPECTED_PHYSICAL_BALL_SCENE_SOURCE_SHA256
            or outcome_binding["table_scene_source_sha256"]
            != EXPECTED_TABLE_SCENE_SOURCE_SHA256
            or outcome_question["question_source_sha256"]
            != expected_question_source_sha
            or outcome_question["landing_aim_xy_w_m"]
            != [float(value) for value in expected_aim]
        ):
            raise N1RewardEventKernelError(
                "observed-outcome authority differs from external parent"
            )
        expected_lineage_sha = (
            None if lineage is None else lineage["content_sha256"]
        )
        if outcome_question["action_lineage_sha256"] != expected_lineage_sha:
            raise N1RewardEventKernelError(
                "observed-outcome question and selected-rubber lineage differ"
            )
        if lineage is not None and (
            outcome_binding["scene_binding_sha256"]
            != lineage["scene_binding_sha256"]
            or outcome_binding["mujoco_backend_version"]
            != lineage["mujoco_backend_version"]
        ):
            raise N1RewardEventKernelError(
                "observed-outcome scene/backend and selected-rubber lineage differ"
            )
        if outcome_snapshot["armed"] is not (outgoing is not None):
            raise N1RewardEventKernelError(
                "observed-outcome arm state and native outgoing flight disagree"
            )
        if outgoing is not None:
            resolver_outgoing = outcome_snapshot["outgoing_sample"]
            if (
                resolver_outgoing["stamp"]
                != {
                    "policy_tick": outgoing["policy_tick"],
                    "physics_substep": outgoing["physics_substep"],
                }
                or resolver_outgoing["time_s"] != float(outgoing["time_s"])
                or resolver_outgoing["ball_center_w_m"]
                != [float(value) for value in outgoing["position_w_m"]]
            ):
                raise N1RewardEventKernelError(
                    "observed-outcome resolver seed differs from outgoing flight"
                )
        last_stamp = outcome_snapshot["last_sample_stamp"]
        if last_stamp is not None and last_stamp["policy_tick"] > policy_tick:
            raise N1RewardEventKernelError(
                "observed-outcome resolver snapshot is in the future"
            )
        if outcome_snapshot["armed"]:
            expected_last_stamp = {
                "policy_tick": policy_tick - 1,
                "physics_substep": outcome_binding["control_decimation"] - 1,
            }
            if policy_tick < 1 or last_stamp != expected_last_stamp:
                raise N1RewardEventKernelError(
                    "observed-outcome transcript does not reach native fact cutoff"
                )
    elif any(
        value is not None
        for value in (outcome_binding, outcome_question, outcome_snapshot)
    ):
        raise N1RewardEventKernelError(
            "unavailable observed-outcome authority cannot carry evidence"
        )
    return deepcopy(dict(sample))


def contact_evidence_from_native_facts(
    sample: Mapping[str, Any],
    *,
    expected_source: SourceBinding,
    expected_outcome_resolver_binding_sha256: str | None = None,
    expected_outcome_question_binding_sha256: str | None = None,
    expected_outcome_scene_binding_sha256: str | None = None,
    expected_outcome_plant_binding_sha256: str | None = None,
    expected_question_source_sha256: str | None = None,
    expected_question_landing_aim_xy_w_m: tuple[float, float] | None = None,
) -> ContactEvidence:
    """Extract actual selected-rubber contact evidence from validated facts.

    A generic-blade hit on the opposite face or in an ambiguous edge/rim cell
    remains an observed contact with ``selected_rubber=False``.  A caller may
    count it as a miss, but cannot unlock outgoing-flight reward eligibility.
    """

    canonical = validate_native_physical_event_facts(
        sample,
        expected_source=expected_source,
        expected_outcome_resolver_binding_sha256=(
            expected_outcome_resolver_binding_sha256
        ),
        expected_outcome_question_binding_sha256=(
            expected_outcome_question_binding_sha256
        ),
        expected_outcome_scene_binding_sha256=(
            expected_outcome_scene_binding_sha256
        ),
        expected_outcome_plant_binding_sha256=(
            expected_outcome_plant_binding_sha256
        ),
        expected_question_source_sha256=expected_question_source_sha256,
        expected_question_landing_aim_xy_w_m=(
            expected_question_landing_aim_xy_w_m
        ),
    )
    if not canonical["selected_rubber_authority_available"]:
        raise N1RewardEventKernelError(
            "native facts have no selected-rubber authority"
        )
    stamp = _stamp_from_mapping(
        canonical["first_racket_contact_stamp"], "first_racket_contact_stamp"
    )
    if stamp is None:
        return ContactEvidence(occurred=False, stamp=None, selected_rubber=False)
    classification = canonical["first_racket_contact_classification"]
    return ContactEvidence(
        occurred=True,
        stamp=stamp,
        selected_rubber=(
            classification["status"]
            == selected_rubber_classifier.STATUS_SELECTED
            and canonical["racket_contact_edge_count_total"] == 1
            and not canonical["invalid_reasons"]
        ),
    )


def outgoing_flight_evidence_from_native_facts(
    sample: Mapping[str, Any],
    *,
    expected_source: SourceBinding,
    expected_outcome_resolver_binding_sha256: str | None = None,
    expected_outcome_question_binding_sha256: str | None = None,
    expected_outcome_scene_binding_sha256: str | None = None,
    expected_outcome_plant_binding_sha256: str | None = None,
    expected_question_source_sha256: str | None = None,
    expected_question_landing_aim_xy_w_m: tuple[float, float] | None = None,
) -> OutgoingFlightEvidence:
    """Extract achieved outgoing flight only after a valid selected hit."""

    canonical = validate_native_physical_event_facts(
        sample,
        expected_source=expected_source,
        expected_outcome_resolver_binding_sha256=(
            expected_outcome_resolver_binding_sha256
        ),
        expected_outcome_question_binding_sha256=(
            expected_outcome_question_binding_sha256
        ),
        expected_outcome_scene_binding_sha256=(
            expected_outcome_scene_binding_sha256
        ),
        expected_outcome_plant_binding_sha256=(
            expected_outcome_plant_binding_sha256
        ),
        expected_question_source_sha256=expected_question_source_sha256,
        expected_question_landing_aim_xy_w_m=(
            expected_question_landing_aim_xy_w_m
        ),
    )
    contact = contact_evidence_from_native_facts(
        canonical,
        expected_source=expected_source,
        expected_outcome_resolver_binding_sha256=(
            expected_outcome_resolver_binding_sha256
        ),
        expected_outcome_question_binding_sha256=(
            expected_outcome_question_binding_sha256
        ),
        expected_outcome_scene_binding_sha256=(
            expected_outcome_scene_binding_sha256
        ),
        expected_outcome_plant_binding_sha256=(
            expected_outcome_plant_binding_sha256
        ),
        expected_question_source_sha256=expected_question_source_sha256,
        expected_question_landing_aim_xy_w_m=(
            expected_question_landing_aim_xy_w_m
        ),
    )
    outgoing = canonical["outgoing_flight"]
    if (
        not contact.occurred
        or not contact.selected_rubber
        or outgoing is None
        or canonical["invalid_reasons"]
    ):
        return OutgoingFlightEvidence(False, None, None, None, None)
    return OutgoingFlightEvidence(
        valid=True,
        stamp=EventStamp(outgoing["policy_tick"], outgoing["physics_substep"]),
        position_w_m=tuple(float(value) for value in outgoing["position_w_m"]),
        linear_velocity_w_mps=tuple(
            float(value) for value in outgoing["linear_velocity_w_mps"]
        ),
        spin_w_radps=tuple(float(value) for value in outgoing["spin_w_radps"]),
    )


def observed_outcome_evidence_from_native_facts(
    sample: Mapping[str, Any],
    *,
    expected_source: SourceBinding,
    expected_outcome_resolver_binding_sha256: str,
    expected_outcome_question_binding_sha256: str,
    expected_outcome_scene_binding_sha256: str,
    expected_outcome_plant_binding_sha256: str,
    expected_question_source_sha256: str,
    expected_question_landing_aim_xy_w_m: tuple[float, float],
) -> ObservedOutcomeEvidence:
    """Consume sealed native net/landing facts without predicting an outcome."""

    canonical = validate_native_physical_event_facts(
        sample,
        expected_source=expected_source,
        expected_outcome_resolver_binding_sha256=(
            expected_outcome_resolver_binding_sha256
        ),
        expected_outcome_question_binding_sha256=(
            expected_outcome_question_binding_sha256
        ),
        expected_outcome_scene_binding_sha256=(
            expected_outcome_scene_binding_sha256
        ),
        expected_outcome_plant_binding_sha256=(
            expected_outcome_plant_binding_sha256
        ),
        expected_question_source_sha256=expected_question_source_sha256,
        expected_question_landing_aim_xy_w_m=(
            expected_question_landing_aim_xy_w_m
        ),
    )
    if not canonical["observed_outcome_authority_available"]:
        raise N1RewardEventKernelError(
            "native facts have no observed-outcome authority"
        )
    flight = outgoing_flight_evidence_from_native_facts(
        canonical,
        expected_source=expected_source,
        expected_outcome_resolver_binding_sha256=(
            expected_outcome_resolver_binding_sha256
        ),
        expected_outcome_question_binding_sha256=(
            expected_outcome_question_binding_sha256
        ),
        expected_outcome_scene_binding_sha256=(
            expected_outcome_scene_binding_sha256
        ),
        expected_outcome_plant_binding_sha256=(
            expected_outcome_plant_binding_sha256
        ),
        expected_question_source_sha256=expected_question_source_sha256,
        expected_question_landing_aim_xy_w_m=(
            expected_question_landing_aim_xy_w_m
        ),
    )
    snapshot = canonical["observed_outcome_snapshot"]
    if not flight.valid or not snapshot["outcome_resolved"]:
        return ObservedOutcomeEvidence(False, None, None, None)
    stamp = _stamp_from_mapping(snapshot["outcome_stamp"], "observed outcome stamp")
    if stamp is None:
        raise N1RewardEventKernelError("resolved observed outcome has no stamp")
    return ObservedOutcomeEvidence(
        resolved=True,
        stamp=stamp,
        observed_net_clear=snapshot["observed_net_clear"],
        observed_legal_landing=snapshot["observed_legal_landing"],
    )


def _require_strictly_after(later: EventStamp, earlier: EventStamp, name: str) -> None:
    if not later > earlier:
        raise N1RewardEventKernelError("%s must occur strictly after its prerequisite" % name)


def _validate_input(sample: N1RewardEventInput, expected_source: SourceBinding) -> None:
    if type(sample) is not N1RewardEventInput:
        raise N1RewardEventKernelError("sample must be N1RewardEventInput")
    _validate_source(expected_source, "expected_source")
    _validate_source(sample.source, "sample.source")
    if sample.source != expected_source:
        raise N1RewardEventKernelError("sample source binding does not match expected authority")
    _require_bool(sample.motion_mimic_eligible, "motion_mimic_eligible")
    _require_bool(sample.target_valid, "target_valid")
    _require_bool(sample.strike_window, "strike_window")

    contact = sample.actual_contact
    if type(contact) is not ContactEvidence:
        raise N1RewardEventKernelError("actual_contact must be ContactEvidence")
    contact_occurred = _require_bool(contact.occurred, "actual_contact.occurred")
    _require_bool(contact.selected_rubber, "actual_contact.selected_rubber")
    _validate_stamp(contact.stamp, "actual_contact.stamp", contact_occurred)
    if not contact_occurred and (contact.stamp is not None or contact.selected_rubber):
        raise N1RewardEventKernelError("absent actual contact cannot carry contact facts")

    flight = sample.outgoing_flight
    if type(flight) is not OutgoingFlightEvidence:
        raise N1RewardEventKernelError("outgoing_flight must be OutgoingFlightEvidence")
    flight_valid = _require_bool(flight.valid, "outgoing_flight.valid")
    _validate_stamp(flight.stamp, "outgoing_flight.stamp", flight_valid)
    _validate_vector(flight.position_w_m, "outgoing_flight.position_w_m", flight_valid)
    _validate_vector(
        flight.linear_velocity_w_mps, "outgoing_flight.linear_velocity_w_mps", flight_valid
    )
    _validate_vector(flight.spin_w_radps, "outgoing_flight.spin_w_radps", flight_valid)
    if not flight_valid and any(
        value is not None
        for value in (
            flight.stamp,
            flight.position_w_m,
            flight.linear_velocity_w_mps,
            flight.spin_w_radps,
        )
    ):
        raise N1RewardEventKernelError("invalid outgoing flight cannot carry flight facts")
    valid_actual_contact = contact_occurred and contact.selected_rubber
    if flight_valid:
        if not valid_actual_contact:
            raise N1RewardEventKernelError(
                "valid outgoing flight requires an actual selected-rubber contact"
            )
        _require_strictly_after(flight.stamp, contact.stamp, "outgoing flight")

    predicted = sample.predicted_outcome
    if type(predicted) is not PredictedOutcomeEvidence:
        raise N1RewardEventKernelError("predicted_outcome must be PredictedOutcomeEvidence")
    predicted_evaluated = _require_bool(predicted.evaluated, "predicted_outcome.evaluated")
    if predicted_evaluated:
        if not flight_valid:
            raise N1RewardEventKernelError(
                "predicted outcome requires a valid achieved outgoing flight"
            )
        _require_bool(predicted.predicted_net_clear, "predicted_outcome.predicted_net_clear")
        _require_bool(
            predicted.predicted_legal_landing,
            "predicted_outcome.predicted_legal_landing",
        )
        if predicted.predicted_legal_landing and not predicted.predicted_net_clear:
            raise N1RewardEventKernelError("predicted legal landing requires predicted net clear")
    elif (
        predicted.predicted_net_clear is not None
        or predicted.predicted_legal_landing is not None
    ):
        raise N1RewardEventKernelError("unevaluated predicted outcome cannot carry result facts")

    observed = sample.observed_outcome
    if type(observed) is not ObservedOutcomeEvidence:
        raise N1RewardEventKernelError("observed_outcome must be ObservedOutcomeEvidence")
    observed_resolved = _require_bool(observed.resolved, "observed_outcome.resolved")
    _validate_stamp(observed.stamp, "observed_outcome.stamp", observed_resolved)
    if observed_resolved:
        if not flight_valid:
            raise N1RewardEventKernelError(
                "observed outcome requires a valid achieved outgoing flight"
            )
        _require_strictly_after(observed.stamp, flight.stamp, "observed outcome")
        _require_bool(observed.observed_net_clear, "observed_outcome.observed_net_clear")
        _require_bool(
            observed.observed_legal_landing,
            "observed_outcome.observed_legal_landing",
        )
        if observed.observed_legal_landing and not observed.observed_net_clear:
            raise N1RewardEventKernelError("observed legal landing requires observed net clear")
    elif (
        observed.stamp is not None
        or observed.observed_net_clear is not None
        or observed.observed_legal_landing is not None
    ):
        raise N1RewardEventKernelError("unresolved observed outcome cannot carry result facts")

    closure = sample.swing_closure
    if type(closure) is not SwingClosureEvidence:
        raise N1RewardEventKernelError("swing_closure must be SwingClosureEvidence")
    closed = _require_bool(closure.closed, "swing_closure.closed")
    timeout = _require_bool(closure.timeout, "swing_closure.timeout")
    _validate_stamp(closure.stamp, "swing_closure.stamp", closed)
    if not closed and (closure.stamp is not None or timeout):
        raise N1RewardEventKernelError("open swing cannot have closure facts")
    if closed:
        if contact_occurred:
            _require_strictly_after(closure.stamp, contact.stamp, "swing closure")
        if flight_valid:
            _require_strictly_after(closure.stamp, flight.stamp, "swing closure")
        if observed_resolved:
            _require_strictly_after(closure.stamp, observed.stamp, "swing closure")


def evaluate_n1_reward_event(
    sample: N1RewardEventInput, *, expected_source: SourceBinding
) -> N1RewardEligibility:
    """Return pure eligibility/count increments for one source-bound N1 swing.

    No input object is mutated.  A caller may aggregate the returned booleans
    into per-action/per-side denominators, but must retain zero cells as zero
    rather than treating them as successful reward events.
    """

    _validate_input(sample, expected_source)
    contact = sample.actual_contact
    flight = sample.outgoing_flight
    predicted = sample.predicted_outcome
    observed = sample.observed_outcome
    closure = sample.swing_closure
    valid_actual_contact = contact.occurred and contact.selected_rubber
    valid_flight = valid_actual_contact and flight.valid
    predicted_eligible = valid_flight and predicted.evaluated
    observed_eligible = valid_flight and observed.resolved
    observed_legal = observed_eligible and bool(observed.observed_legal_landing)

    return N1RewardEligibility(
        motion_mimic_denominator=sample.motion_mimic_eligible,
        contact_target_denominator=sample.target_valid and sample.strike_window,
        closed_swing_denominator=closure.closed,
        actual_contact_numerator=closure.closed and valid_actual_contact,
        achieved_outgoing_flight_denominator=valid_flight,
        predicted_outcome_denominator=predicted_eligible,
        predicted_net_clear_numerator=(
            predicted_eligible and bool(predicted.predicted_net_clear)
        ),
        predicted_legal_landing_numerator=(
            predicted_eligible and bool(predicted.predicted_legal_landing)
        ),
        observed_outcome_denominator=observed_eligible,
        observed_net_clear_numerator=(
            observed_eligible and bool(observed.observed_net_clear)
        ),
        observed_legal_landing_numerator=observed_legal,
        unresolved_achieved_flight=valid_flight and closure.closed and not observed.resolved,
        motion_mimic_pay_eligible=sample.motion_mimic_eligible,
        contact_target_pay_eligible=sample.target_valid and sample.strike_window,
        actual_contact_pay_eligible=valid_actual_contact,
        predicted_outcome_pay_eligible=predicted_eligible,
        observed_outcome_pay_eligible=observed_legal,
    )
