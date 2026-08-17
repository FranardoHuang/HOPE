"""Fail-closed, engine-neutral schema for the fresh ActionBall reward budget.

The original revision deliberately has no READY construction path.  It
freezes only facts that are already authoritative: the policy clock, ordered
common strike consumers, same-transition task-reference eligibility, the
explicit eight-to-fourteen-field delayed-outcome identity bridge, symbolic
reward hierarchy, and the C04 landing-placement scorer identity.  Every
scientific price or support-dependent quantity remains an explicit ``UNSET``
value.  That schema-1 API remains byte-compatible.

Schema 2 adds a separate finite-candidate numeric materializer.  It has no
embedded candidate, weight, kernel scale, or placement-profile default.  A
caller must supply three independently sealed inputs: a constructed resolved
graph, an exact four-shot unit-income/phase-support tape, and an explicit
finite candidate set.  A separate launcher-owned trust-root mapping must pin
the source closure and all three receipt hashes; self-sealed input cycles are
not authority.  Arithmetic uses exact rational numbers, the actual payment
tick, one policy-dt multiplication, and the frozen PPO gamma.  Every shot must
independently pass cadence, flight/mailbox capacity, A/C common parity, and
reward-hierarchy gates.  Only one feasible candidate at the unique minimum
explicit lexicographic priority can produce a budget authority.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import weakref

import action_ball_landing_placement as c04


SCHEMA_VERSION = 1
KIND = "action_ball_fresh_reward_budget_v1"
BLOCKED_STATUS = "BLOCKED_UNRESOLVED_NUMERIC_BUDGET"
READY_STATUS = "READY"
UNSET_STATE = "UNSET"

NUMERIC_SCHEMA_VERSION = 2
NUMERIC_GRAPH_KIND = "action_ball_fresh_reward_resolved_graph_v2"
NUMERIC_TAPE_KIND = "action_ball_fresh_reward_four_shot_tape_v2"
NUMERIC_CANDIDATE_SET_KIND = "action_ball_fresh_reward_candidate_set_v2"
NUMERIC_MATERIALIZATION_KIND = "action_ball_fresh_reward_materialization_v2"
NUMERIC_AUTHORITY_KIND = "action_ball_fresh_reward_authority_v2"
NUMERIC_BLOCKED_STATUS = "BLOCKED_NO_UNIQUE_FEASIBLE_NUMERIC_CANDIDATE"
CONSTRUCTED_EVIDENCE_SCOPE = "constructed_runtime"
EXPLICIT_CANDIDATE_SCOPE = "explicit_finite_candidates"
LEXICOGRAPHIC_SELECTION_SEMANTICS = (
    "unique_minimum_explicit_integer_priority_without_candidate_id_tiebreak_v1"
)
NUMERIC_AUTHORITY_SCOPE = (
    "reward_budget_only_not_training_launch_authority_v1"
)
TRUSTED_INPUT_ROOT_KIND = "launcher_owned_fresh_budget_input_roots_v1"
NUMERIC_RUNTIME_INTEGRATED = False
NUMERIC_LAUNCH_AUTHORIZED = False
NUMERIC_DIAGNOSTIC_UNAUTHORIZED = True
NUMERIC_PRODUCTION_HOLD_REASONS = (
    "constructed_runtime_reward_graph_producer_absent",
    "real_four_shot_unit_income_phase_support_producer_absent",
    "launcher_owned_finite_candidate_set_producer_absent",
    "fourteen_live_reward_consumers_not_factory_bound",
)
RECOVERY_AGE_START = 10
RECOVERY_AGE_END = 77
RECOVERY_AGE_COUNT = RECOVERY_AGE_END - RECOVERY_AGE_START + 1
MIN_DEADLINE_TO_NEXT_REVEAL_TICKS = RECOVERY_AGE_END + 1

POLICY_DT = {"numerator": 1, "denominator": 50}
PPO_GAMMA = {"numerator": 99, "denominator": 100}
DT_APPLICATION_SEMANTICS = (
    "reward_manager_multiplies_raw_times_manager_weight_by_policy_dt_"
    "exactly_once_per_term_evaluation_v1"
)

# R03 schema-2 preserves the historical nine consumer indices and appends the
# common causal paddle-centre shaper.  Semantic aliases make the fine/coarse/
# precision meaning explicit without silently accepting a reordered set.
ORDERED_CONTACT_TERMS = (
    ("racket_position", "desired_contact_position_fine"),
    ("racket_velocity", "desired_contact_velocity_fine"),
    ("racket_normal", "desired_contact_face_fine"),
    ("racket_position_coarse", "desired_contact_position_coarse"),
    ("racket_velocity_coarse", "desired_contact_velocity_coarse"),
    ("racket_normal_coarse", "desired_contact_face_coarse"),
    ("racket_position_precision", "desired_contact_position_precision"),
    ("racket_velocity_precision", "desired_contact_velocity_precision"),
    ("racket_normal_precision", "desired_contact_face_precision"),
    ("paddle_center_proximity", "paddle_center_distance_one_shot"),
)
ORDERED_CONTACT_TERM_NAMES = tuple(item[0] for item in ORDERED_CONTACT_TERMS)
CONTACT_TERM_COUNT = len(ORDERED_CONTACT_TERMS)
NUMERIC_STRIKE_KERNEL_KINDS = {
    "racket_position": "gaussian_v1",
    "racket_velocity": "gaussian_v1",
    "racket_normal": "gaussian_v1",
    "racket_position_coarse": "cauchy_v1",
    "racket_velocity_coarse": "cauchy_v1",
    "racket_normal_coarse": "cauchy_v1",
    "racket_position_precision": "gaussian_v1",
    "racket_velocity_precision": "gaussian_v1",
    "racket_normal_precision": "gaussian_v1",
    "paddle_center_proximity": "cauchy_v1",
}
PROHIBITED_FIXTURE_PLACEMENT_PROFILE = {
    "alpha_broad": Fraction(2, 5),
    "sigma_broad_m": Fraction(1, 2),
    "sigma_narrow_m": Fraction(1, 10),
}

# The device strike fact uses this exact eight-field runtime task reference.
# It is deliberately *not* called a full shot key: delayed outcome ownership
# adds six successor-lineage fields and is a distinct fourteen-field ABI.
ACTION_TASK_RECEIPT_REF_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
)
LANDING_OUTCOME_SUCCESSOR_FIELDS = (
    "run_id",
    "carry_chain_id",
    "shot_index",
    "source_sha256",
    "config_sha256",
    "receipt_content_sha256",
)
LANDING_OUTCOME_SHOT_KEY_FIELDS = (
    *ACTION_TASK_RECEIPT_REF_FIELDS,
    *LANDING_OUTCOME_SUCCESSOR_FIELDS,
)

C04_LOGICAL_SOURCE_PATH = (
    "source/whole_body_tracking/action_ball_landing_placement.py"
)
C04_SOURCE_SHA256 = (
    "3e2e056336a8c021c20bd255c474487cb2346a3dcdfcca8a1b1a608dd90636e2"
)

LEGACY_WINDOW_MANAGER_WEIGHTS = {
    "racket_position": "4.6",
    "racket_velocity": "0.575",
    "racket_normal": "0.575",
    "racket_position_coarse": "11.5",
    "racket_velocity_coarse": "11.5",
    "racket_normal_coarse": "5.75",
    "racket_position_precision": "0.575",
    "racket_velocity_precision": "0.2875",
    "racket_normal_precision": "0.575",
}
LEGACY_PROXIMITY_MANAGER_WEIGHT = "240"
LEGACY_LANDING_MANAGER_WEIGHT = "700"

UNRESOLVED_REASON_CODES = (
    "resolved_manager_graph_unbound",
    "fresh_phase_support_unfrozen",
    "landing_outcome_key_derivation_unbound",
    "fresh_strike_kernel_profile_unfrozen",
    "benchmark_error_tape_unfrozen",
    "common_motion_budget_unfrozen",
    "ten_post_dt_vector_unfrozen",
    "selected_contact_post_dt_unfrozen",
    "common_on_table_post_dt_unfrozen",
    "recovery_budget_unfrozen",
    "placement_profile_unfrozen",
    "placement_post_dt_unfrozen",
    "ready_authority_constructor_not_installed",
)


class FreshRewardBudgetError(ValueError):
    """A structural drift or attempted early authorization."""


class FreshRewardNumericProductionHold(FreshRewardBudgetError):
    """Production numeric authority is unavailable without real producers."""


def _canonical_json_bytes(value: object) -> bytes:
    """Encode canonical JSON without collapsing booleans into integers."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    """Hash strict repository-style canonical JSON."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _type_exact_equal(left: object, right: object) -> bool:
    """Compare recursively without Python's ``True == 1`` coercion.

    The receipt is a JSON value tree.  Requiring the exact container/scalar
    type at every node also prevents a tuple, mapping subclass, or integer-like
    value from being treated as the frozen JSON ABI by ordinary equality.
    """

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if len(left) != len(right):
            return False
        unmatched = list(left.items())
        for right_key, right_value in right.items():
            for index, (left_key, left_value) in enumerate(unmatched):
                if _type_exact_equal(left_key, right_key):
                    if not _type_exact_equal(left_value, right_value):
                        return False
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    if type(left) in (list, tuple):
        if len(left) != len(right):
            return False
        return all(
            _type_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _frozen_exact_equal(left: object, right: object) -> bool:
    """Require recursive type identity and identical canonical bytes."""

    if not _type_exact_equal(left, right):
        return False
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unset(reason: str) -> dict[str, str]:
    if reason not in UNRESOLVED_REASON_CODES:
        raise FreshRewardBudgetError(f"unknown UNSET reason {reason!r}")
    return {"state": UNSET_STATE, "reason": reason}


def _is_unset(value: object, reason: str | None = None) -> bool:
    if not isinstance(value, Mapping):
        return False
    if frozenset(value) != {"state", "reason"} or value.get("state") != UNSET_STATE:
        return False
    return reason is None or value.get("reason") == reason


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise FreshRewardBudgetError("ratio must use exact integers and positive denominator")
    value = Fraction(numerator, denominator)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _as_fraction(value: object) -> Fraction | None:
    """Read an exact candidate number without accepting binary JSON floats."""

    if isinstance(value, bool) or isinstance(value, float):
        return None
    if type(value) is int:
        return Fraction(value, 1)
    if type(value) is str:
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, OverflowError, ValueError, ZeroDivisionError):
            return None
    if isinstance(value, Mapping) and frozenset(value) == {
        "numerator",
        "denominator",
    }:
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if (
            type(numerator) is int
            and type(denominator) is int
            and denominator != 0
        ):
            return Fraction(numerator, denominator)
    return None


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_float(key) or _contains_float(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(item) for item in value)
    return False


def _c04_scorer_identity() -> dict[str, object]:
    return {
        "logical_source_path": C04_LOGICAL_SOURCE_PATH,
        "source_sha256": C04_SOURCE_SHA256,
        "profile_kind": c04.PROFILE_KIND,
        "score_kind": c04.SCORE_KIND,
        "contact_source_semantics": c04.SELECTED_RUBBER_CONTACT_AUTHORITY,
        "broad_kernel": c04.CAUCHY_DEFINITION,
        "narrow_kernel": c04.GAUSSIAN_DEFINITION,
        "table_gate": _ratio(1, 1),
        "opponent_bound_off_table_failure_gate": _ratio(1, 2),
        "invalid_gate": _ratio(0, 1),
        "success_semantics": "valid_opponent_table_first_landing_only_v1",
    }


def _contact_term(index: int, runtime_name: str, semantic_role: str) -> dict[str, object]:
    return {
        "index": index,
        "runtime_name": runtime_name,
        "semantic_role": semantic_role,
        "family_scope": "common_a_c",
        "included_in_ten": True,
        "kernel_profile": _unset("fresh_strike_kernel_profile_unfrozen"),
        "raw_upper_bound": _unset("fresh_strike_kernel_profile_unfrozen"),
        "manager_weight": _unset("ten_post_dt_vector_unfrozen"),
        "post_dt_peak": _unset("ten_post_dt_vector_unfrozen"),
        "dt_applications_per_evaluation": 1,
        "max_payments_per_action_task_receipt_ref": 1,
        "payment_semantics": (
            "same_transition_action_task_receipt_ref_one_shot_"
            "post_physics_pre_reward_v1"
        ),
    }


def _blocked_payload() -> dict[str, object]:
    contact_terms = [
        _contact_term(index, runtime_name, semantic_role)
        for index, (runtime_name, semantic_role) in enumerate(ORDERED_CONTACT_TERMS)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": BLOCKED_STATUS,
        "launch_authorized": False,
        "authority_sha256": None,
        "policy_clock": {
            "policy_dt": dict(POLICY_DT),
            "ppo_gamma": dict(PPO_GAMMA),
            "dt_application_semantics": DT_APPLICATION_SEMANTICS,
        },
        "source_authorities": {
            "common_parent_sha256": _unset("resolved_manager_graph_unbound"),
            "resolved_manager_graph_sha256": _unset(
                "resolved_manager_graph_unbound"
            ),
            "phase_support_contract_sha256": _unset(
                "fresh_phase_support_unfrozen"
            ),
            "benchmark_error_tape_sha256": _unset(
                "benchmark_error_tape_unfrozen"
            ),
            "landing_outcome_key_derivation_sha256": _unset(
                "landing_outcome_key_derivation_unbound"
            ),
            "c04_landing_placement_scorer": _c04_scorer_identity(),
        },
        "contact_bundle": {
            "ordered": True,
            "count": CONTACT_TERM_COUNT,
            "terms": contact_terms,
            "eligibility": {
                "publisher_phase": "post_physics_pre_reward",
                "same_source_and_payment_transition": True,
                "scheduled_admitted_task_valid_shot_required": True,
                "selected_rubber_contact_required": False,
                "selected_rubber_contact_is_separate_common_event": True,
                "family_selector_forbidden": True,
                "action_task_receipt_ref_required": True,
                "action_task_receipt_ref_fields": list(
                    ACTION_TASK_RECEIPT_REF_FIELDS
                ),
                "previous_or_next_transition_payment_forbidden": True,
            },
        },
        "identity_bridge": {
            "runtime_task_ref_kind": "action_task_receipt_ref_8_field_v1",
            "runtime_task_ref_fields": list(ACTION_TASK_RECEIPT_REF_FIELDS),
            "landing_outcome_key_kind": "landing_outcome_shot_key_14_field_v1",
            "landing_outcome_key_fields": list(
                LANDING_OUTCOME_SHOT_KEY_FIELDS
            ),
            "runtime_prefix_must_equal_task_ref": True,
            "successor_suffix_fields": list(LANDING_OUTCOME_SUCCESSOR_FIELDS),
            "derivation_semantics": (
                "landing_outcome_key_equals_exact_action_task_receipt_ref_prefix_"
                "plus_reveal_transaction_successor_lineage_v1"
            ),
            "derivation_authority_sha256": _unset(
                "landing_outcome_key_derivation_unbound"
            ),
            "eight_field_task_ref_cannot_substitute_for_outcome_key": True,
            "canonical_outcome_key_collision_forbidden": True,
        },
        "common_events": {
            "selected_rubber_contact": {
                "included_in_ten": False,
                "family_scope": "common_a_c",
                "raw_semantics": "physical_selected_rubber_contact_binary_v1",
                "action_task_receipt_ref_required": True,
                "landing_outcome_shot_key_required": True,
                "manager_weight": _unset("selected_contact_post_dt_unfrozen"),
                "post_dt_peak": _unset("selected_contact_post_dt_unfrozen"),
                "max_payments_per_landing_outcome_shot_key": 1,
                "dt_applications_per_evaluation": 1,
            },
            "on_table_success": {
                "included_in_ten": False,
                "family_scope": "common_a_c",
                "raw_semantics": "valid_opponent_table_first_landing_binary_v1",
                "source": "landing_outcome_shot_key_14_field_delayed_mailbox",
                "landing_outcome_shot_key_required": True,
                "selected_rubber_contact_required": True,
                "off_table_is_failure": True,
                "manager_weight": _unset("common_on_table_post_dt_unfrozen"),
                "post_dt_peak": _unset("common_on_table_post_dt_unfrozen"),
                "max_payments_per_landing_outcome_shot_key": 1,
                "dt_applications_per_evaluation": 1,
            },
        },
        "motion_and_recovery": {
            "common_motion_recipe_sha256": _unset(
                "common_motion_budget_unfrozen"
            ),
            "discounted_motion_cap": _unset("common_motion_budget_unfrozen"),
            "recovery_term_set": _unset("recovery_budget_unfrozen"),
            "recovery_eligibility_support": _unset(
                "fresh_phase_support_unfrozen"
            ),
            "recovery_post_dt_budget": _unset("recovery_budget_unfrozen"),
            "nonzero_recovery_required_from_rollout_zero": True,
        },
        "placement_treatment": {
            "included_in_ten": False,
            "source": "landing_outcome_shot_key_14_field_delayed_mailbox",
            "landing_outcome_shot_key_required": True,
            "scorer_identity": _c04_scorer_identity(),
            "profile": {
                "alpha_broad": _unset("placement_profile_unfrozen"),
                "sigma_broad_m": _unset("placement_profile_unfrozen"),
                "sigma_narrow_m": _unset("placement_profile_unfrozen"),
            },
            "common_manager_term": {
                "scheduled_for_a": True,
                "scheduled_for_c": True,
                "manager_weight": _unset("placement_post_dt_unfrozen"),
                "post_dt_peak_if_treatment_gain_one": _unset(
                    "placement_post_dt_unfrozen"
                ),
                "resolved_manager_weight_must_be_positive": True,
                "dt_applications_per_evaluation": 1,
            },
            "treatment_gain": {
                "a": _ratio(1, 1),
                "c": _ratio(0, 1),
            },
            "payment_record_required": {"a": True, "c": True},
            "c_zero_payment_record_required": True,
            "max_payments_per_landing_outcome_shot_key": 1,
            "consume_once_transition": "settled_unpaid_to_paid",
            "repeated_peek_payment_forbidden": True,
            "only_family_delta": "identity_bound_treatment_gain",
            "may_satisfy_common_hierarchy": False,
        },
        "symbolic_budget": {
            "post_dt_term": "raw * manager_weight * (1/50)",
            "discounted_group": "sum_t((99/100)^(t-t0) * post_dt_term_t)",
            "ten_peak_per_shot": "sum_i(b10_i)",
            "a_placement_peak_per_shot": "b_place",
            "a_off_table_placement_upper": "(1/2) * b_place",
            "c_placement_peak_per_shot": "0",
            "hierarchy": [
                {
                    "name": "no_contact_strike_guidance_dominates_motion",
                    "stratum": "scheduled_admitted_task_valid_no_contact_shot",
                    "inequality": "B_motion_cap_discounted < B10_reference_discounted",
                    "contact_bonus_may_satisfy": False,
                    "placement_may_satisfy": False,
                },
                {
                    "name": "common_table_outcome_dominates_total_strike",
                    "stratum": "selected_contact_and_settled_outcome_witness",
                    "inequality": (
                        "B10_peak_discounted + B_selected_contact_peak_discounted "
                        "< B_common_on_table_peak_discounted"
                    ),
                    "must_pass_for_family_c_without_placement": True,
                    "placement_may_satisfy": False,
                },
                {
                    "name": "auxiliary_income_does_not_invert_main_hierarchy",
                    "stratum": "realized_common_typical_and_p95",
                    "inequality": (
                        "balance_recovery_regularization_typical_p95 "
                        "must_not_invert_motion_strike_table_order"
                    ),
                    "numeric_threshold": _unset("recovery_budget_unfrozen"),
                },
            ],
        },
        "forbidden": {
            "legacy_3_11_tick_manager_weights": dict(
                LEGACY_WINDOW_MANAGER_WEIGHTS
            ),
            "legacy_proximity_manager_weight": LEGACY_PROXIMITY_MANAGER_WEIGHT,
            "legacy_landing_manager_weight": LEGACY_LANDING_MANAGER_WEIGHT,
            "placement_broad_kernel": "gaussian",
            "outcome_objective_tokens": [
                "baseline_direction",
                "baseline_speed",
                "counter_rally_total",
                "independent_pass_net_reward",
                "legal_return_base_reward",
            ],
        },
        "unresolved": list(UNRESOLVED_REASON_CODES),
    }


def build_blocked_receipt() -> dict[str, object]:
    """Return the only receipt this revision is authorized to construct."""

    payload = _blocked_payload()
    result = dict(payload)
    result["design_sha256"] = canonical_sha256(payload)
    return result


@dataclass(frozen=True)
class FreshRewardBudgetAudit:
    """Structural audit result; unresolved science remains separately visible."""

    structurally_valid: bool
    status: str
    launch_authorized: bool
    design_sha256: str | None
    authority_sha256: None
    structural_blockers: tuple[str, ...]
    unresolved: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "structurally_valid": self.structurally_valid,
            "status": self.status,
            "launch_authorized": self.launch_authorized,
            "design_sha256": self.design_sha256,
            "authority_sha256": None,
            "structural_blockers": list(self.structural_blockers),
            "unresolved": list(self.unresolved),
        }


def _append_once(blockers: list[str], reason: str) -> None:
    if reason not in blockers:
        blockers.append(reason)


def _resealed_payload(value: Mapping[str, object]) -> tuple[dict[str, object], object]:
    declared = value.get("design_sha256")
    payload = {key: item for key, item in value.items() if key != "design_sha256"}
    return payload, declared


def _term_mapping(value: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return {}
    output: dict[str, Mapping[str, object]] = {}
    for item in value:
        if isinstance(item, Mapping) and type(item.get("runtime_name")) is str:
            output[str(item["runtime_name"])] = item
    return output


def audit_receipt(value: object) -> FreshRewardBudgetAudit:
    """Audit a blocked design without ever authorizing it."""

    blockers: list[str] = []
    if not isinstance(value, Mapping):
        return FreshRewardBudgetAudit(
            structurally_valid=False,
            status="INVALID",
            launch_authorized=False,
            design_sha256=None,
            authority_sha256=None,
            structural_blockers=("receipt_must_be_mapping",),
            unresolved=UNRESOLVED_REASON_CODES,
        )
    if type(value) is not dict:
        _append_once(blockers, "receipt_must_be_plain_json_object")

    expected = build_blocked_receipt()
    payload, declared_design_sha = _resealed_payload(value)
    expected_keys = frozenset(expected)
    if frozenset(value) != expected_keys:
        _append_once(blockers, "receipt_top_level_keys_drift")
    if (
        type(declared_design_sha) is not str
        or len(declared_design_sha) != 64
        or canonical_sha256(payload) != declared_design_sha
    ):
        _append_once(blockers, "design_sha256_mismatch")
    if _contains_float(value):
        _append_once(blockers, "binary_float_encoding_forbidden")

    if type(value.get("schema_version")) is not int or value.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        _append_once(blockers, "schema_version_must_equal_one")

    status = value.get("status")
    if status != BLOCKED_STATUS:
        if status == READY_STATUS:
            _append_once(blockers, "ready_authority_constructor_not_installed")
        else:
            _append_once(blockers, "status_drift")
    if value.get("launch_authorized") is not False:
        _append_once(blockers, "launch_authorization_forbidden")
    if value.get("authority_sha256") is not None:
        _append_once(blockers, "authority_sha256_forbidden_while_unresolved")

    policy_clock = value.get("policy_clock")
    if not isinstance(policy_clock, Mapping):
        _append_once(blockers, "policy_clock_missing")
    else:
        if not _frozen_exact_equal(policy_clock.get("policy_dt"), POLICY_DT):
            _append_once(blockers, "policy_dt_must_equal_1_over_50")
        if not _frozen_exact_equal(policy_clock.get("ppo_gamma"), PPO_GAMMA):
            _append_once(blockers, "ppo_gamma_must_equal_99_over_100")
        if policy_clock.get("dt_application_semantics") != DT_APPLICATION_SEMANTICS:
            _append_once(blockers, "dt_application_semantics_drift")

    contact_bundle = value.get("contact_bundle")
    terms: object = None
    if not isinstance(contact_bundle, Mapping):
        _append_once(blockers, "contact_bundle_missing")
    else:
        terms = contact_bundle.get("terms")
        names = []
        if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes, bytearray)):
            names = [
                item.get("runtime_name") if isinstance(item, Mapping) else None
                for item in terms
            ]
        if tuple(names) != ORDERED_CONTACT_TERM_NAMES:
            _append_once(blockers, "ordered_ten_drift")
        if type(contact_bundle.get("count")) is not int or contact_bundle.get(
            "count"
        ) != CONTACT_TERM_COUNT:
            _append_once(blockers, "contact_term_count_must_equal_ten")
        eligibility = contact_bundle.get("eligibility")
        if not isinstance(eligibility, Mapping):
            _append_once(blockers, "contact_eligibility_missing")
        else:
            if eligibility.get("selected_rubber_contact_required") is not False:
                _append_once(blockers, "contact_gated_ten_forbidden")
            if eligibility.get("family_selector_forbidden") is not True:
                _append_once(blockers, "family_selector_for_ten_forbidden")
            if eligibility.get("same_source_and_payment_transition") is not True:
                _append_once(blockers, "same_transition_payment_required")
            if tuple(
                eligibility.get("action_task_receipt_ref_fields", ())
            ) != ACTION_TASK_RECEIPT_REF_FIELDS:
                _append_once(blockers, "action_task_receipt_ref_fields_drift")

    identity_bridge = value.get("identity_bridge")
    if not isinstance(identity_bridge, Mapping):
        _append_once(blockers, "identity_bridge_missing")
    else:
        runtime_fields = tuple(identity_bridge.get("runtime_task_ref_fields", ()))
        outcome_fields = tuple(identity_bridge.get("landing_outcome_key_fields", ()))
        if runtime_fields != ACTION_TASK_RECEIPT_REF_FIELDS:
            _append_once(blockers, "action_task_receipt_ref_fields_drift")
        if outcome_fields != LANDING_OUTCOME_SHOT_KEY_FIELDS:
            _append_once(blockers, "landing_outcome_key_fields_drift")
        if outcome_fields == runtime_fields or len(outcome_fields) != 14:
            _append_once(
                blockers, "eight_field_task_ref_cannot_substitute_outcome_key"
            )
        if tuple(identity_bridge.get("successor_suffix_fields", ())) != (
            LANDING_OUTCOME_SUCCESSOR_FIELDS
        ):
            _append_once(blockers, "landing_outcome_successor_fields_drift")
        if identity_bridge.get("runtime_prefix_must_equal_task_ref") is not True:
            _append_once(blockers, "outcome_key_runtime_prefix_binding_required")
        if identity_bridge.get(
            "canonical_outcome_key_collision_forbidden"
        ) is not True:
            _append_once(blockers, "outcome_key_collision_gate_required")
        if not _is_unset(
            identity_bridge.get("derivation_authority_sha256"),
            "landing_outcome_key_derivation_unbound",
        ):
            _append_once(blockers, "outcome_key_derivation_must_remain_unset")

    term_by_name = _term_mapping(terms)
    if isinstance(terms, Sequence) and not isinstance(
        terms, (str, bytes, bytearray)
    ):
        for expected_index, term in enumerate(terms):
            if (
                not isinstance(term, Mapping)
                or type(term.get("index")) is not int
                or term.get("index") != expected_index
            ):
                _append_once(blockers, "contact_term_index_drift")
                break
    legacy_match = True
    for name, legacy_value in LEGACY_WINDOW_MANAGER_WEIGHTS.items():
        term = term_by_name.get(name)
        if term is None:
            legacy_match = False
            continue
        actual = _as_fraction(term.get("manager_weight"))
        if actual != _as_fraction(legacy_value):
            legacy_match = False
        if not _is_unset(term.get("manager_weight"), "ten_post_dt_vector_unfrozen"):
            _append_once(blockers, "ten_numeric_budget_must_remain_unset")
        if not _is_unset(term.get("post_dt_peak"), "ten_post_dt_vector_unfrozen"):
            _append_once(blockers, "ten_numeric_budget_must_remain_unset")
        if (
            type(term.get("dt_applications_per_evaluation")) is not int
            or term.get("dt_applications_per_evaluation") != 1
        ):
            _append_once(blockers, "dt_application_count_must_equal_one")
        if (
            type(term.get("max_payments_per_action_task_receipt_ref")) is not int
            or term.get("max_payments_per_action_task_receipt_ref") != 1
        ):
            _append_once(blockers, "one_shot_payment_count_must_equal_one")
        if term.get("family_scope") != "common_a_c":
            _append_once(blockers, "ten_must_be_common_a_c")
        if term.get("included_in_ten") is not True:
            _append_once(blockers, "ordered_ten_membership_drift")
    if legacy_match:
        _append_once(blockers, "legacy_3_11_tick_weight_table_forbidden")

    proximity = term_by_name.get("paddle_center_proximity")
    if proximity is not None:
        if _as_fraction(proximity.get("manager_weight")) == _as_fraction(
            LEGACY_PROXIMITY_MANAGER_WEIGHT
        ):
            _append_once(blockers, "legacy_proximity_weight_240_forbidden")
        if not _is_unset(
            proximity.get("manager_weight"), "ten_post_dt_vector_unfrozen"
        ) or not _is_unset(
            proximity.get("post_dt_peak"), "ten_post_dt_vector_unfrozen"
        ):
            _append_once(blockers, "ten_numeric_budget_must_remain_unset")
        if (
            type(proximity.get("dt_applications_per_evaluation")) is not int
            or proximity.get("dt_applications_per_evaluation") != 1
        ):
            _append_once(blockers, "dt_application_count_must_equal_one")
        if proximity.get("family_scope") != "common_a_c":
            _append_once(blockers, "ten_must_be_common_a_c")

    source_authorities = value.get("source_authorities")
    scorer_identity = None
    if isinstance(source_authorities, Mapping):
        scorer_identity = source_authorities.get("c04_landing_placement_scorer")
        if not _is_unset(
            source_authorities.get("resolved_manager_graph_sha256"),
            "resolved_manager_graph_unbound",
        ):
            _append_once(
                blockers, "resolved_manager_graph_sha256_must_remain_unset"
            )
        if not _is_unset(
            source_authorities.get("phase_support_contract_sha256"),
            "fresh_phase_support_unfrozen",
        ):
            _append_once(
                blockers, "phase_support_contract_sha256_must_remain_unset"
            )
        if not _is_unset(
            source_authorities.get("benchmark_error_tape_sha256"),
            "benchmark_error_tape_unfrozen",
        ):
            _append_once(
                blockers, "benchmark_error_tape_sha256_must_remain_unset"
            )
    else:
        _append_once(blockers, "source_authorities_missing")
    if not _frozen_exact_equal(scorer_identity, _c04_scorer_identity()):
        _append_once(blockers, "c04_scorer_identity_drift")
        if isinstance(scorer_identity, Mapping) and scorer_identity.get(
            "broad_kernel"
        ) != c04.CAUCHY_DEFINITION:
            _append_once(blockers, "placement_broad_kernel_must_be_cauchy")
    actual_c04_sha = _file_sha256(Path(c04.__file__).resolve())
    if actual_c04_sha != C04_SOURCE_SHA256:
        _append_once(blockers, "c04_source_sha256_drift")

    common_events = value.get("common_events")
    if isinstance(common_events, Mapping):
        for event in common_events.values():
            if not isinstance(event, Mapping):
                continue
            if (
                type(event.get("dt_applications_per_evaluation")) is not int
                or event.get("dt_applications_per_evaluation") != 1
            ):
                _append_once(blockers, "dt_application_count_must_equal_one")
            if (
                type(event.get("max_payments_per_landing_outcome_shot_key"))
                is not int
                or event.get("max_payments_per_landing_outcome_shot_key") != 1
            ):
                _append_once(blockers, "one_shot_payment_count_must_equal_one")
        success = common_events.get("on_table_success")
        if isinstance(success, Mapping):
            if _as_fraction(success.get("manager_weight")) == _as_fraction(
                LEGACY_LANDING_MANAGER_WEIGHT
            ):
                _append_once(blockers, "legacy_landing_weight_700_forbidden")
            if not _is_unset(
                success.get("manager_weight"), "common_on_table_post_dt_unfrozen"
            ):
                _append_once(blockers, "common_on_table_budget_must_remain_unset")

    placement = value.get("placement_treatment")
    if not isinstance(placement, Mapping):
        _append_once(blockers, "placement_treatment_missing")
    else:
        if placement.get("included_in_ten") is not False:
            _append_once(blockers, "placement_inside_ten_forbidden")
        if not _frozen_exact_equal(
            placement.get("scorer_identity"), _c04_scorer_identity()
        ):
            _append_once(blockers, "c04_scorer_identity_drift")
        common_manager = placement.get("common_manager_term")
        if not isinstance(common_manager, Mapping):
            _append_once(blockers, "placement_common_manager_term_missing")
        else:
            if common_manager.get("scheduled_for_a") is not True:
                _append_once(blockers, "a_placement_consumer_schedule_required")
            if common_manager.get("scheduled_for_c") is not True:
                _append_once(blockers, "c_placement_consumer_skip_forbidden")
            manager_weight = common_manager.get("manager_weight")
            if _as_fraction(manager_weight) == Fraction(0, 1):
                _append_once(
                    blockers, "placement_common_manager_weight_zero_forbidden"
                )
            if not _is_unset(
                manager_weight, "placement_post_dt_unfrozen"
            ) or not _is_unset(
                common_manager.get("post_dt_peak_if_treatment_gain_one"),
                "placement_post_dt_unfrozen",
            ):
                _append_once(
                    blockers, "placement_common_budget_must_remain_unset"
                )
            if (
                type(common_manager.get("dt_applications_per_evaluation"))
                is not int
                or common_manager.get("dt_applications_per_evaluation") != 1
            ):
                _append_once(blockers, "dt_application_count_must_equal_one")
        gains = placement.get("treatment_gain")
        if not isinstance(gains, Mapping):
            _append_once(blockers, "placement_treatment_gain_missing")
        else:
            if _as_fraction(gains.get("a")) != Fraction(1, 1):
                _append_once(blockers, "a_placement_treatment_gain_must_equal_one")
            if _as_fraction(gains.get("c")) != Fraction(0, 1):
                _append_once(blockers, "c_placement_treatment_gain_must_equal_zero")
        payment_record = placement.get("payment_record_required")
        if not _frozen_exact_equal(payment_record, {"a": True, "c": True}):
            _append_once(blockers, "placement_payment_record_required_for_both_families")
        if placement.get("c_zero_payment_record_required") is not True:
            _append_once(blockers, "c_zero_payment_record_required")
        if (
            type(placement.get("max_payments_per_landing_outcome_shot_key"))
            is not int
            or placement.get("max_payments_per_landing_outcome_shot_key") != 1
        ):
            _append_once(blockers, "placement_one_shot_payment_count_must_equal_one")
        if placement.get("consume_once_transition") != "settled_unpaid_to_paid":
            _append_once(blockers, "placement_consume_once_transition_required")
        if placement.get("repeated_peek_payment_forbidden") is not True:
            _append_once(blockers, "placement_repeated_peek_payment_forbidden")

    motion = value.get("motion_and_recovery")
    if isinstance(motion, Mapping):
        recovery = motion.get("recovery_post_dt_budget")
        if _as_fraction(recovery) == Fraction(0, 1):
            _append_once(blockers, "recovery_zero_forbidden")
        if not _is_unset(recovery, "recovery_budget_unfrozen"):
            _append_once(blockers, "recovery_budget_must_remain_unset")

    if not _frozen_exact_equal(
        value.get("unresolved"), list(UNRESOLVED_REASON_CODES)
    ):
        _append_once(blockers, "unresolved_reason_set_drift")

    # Exact equality makes every other frozen symbolic/schema field fail closed,
    # while the targeted checks above keep important counterexamples legible.
    if not _frozen_exact_equal(value, expected):
        _append_once(blockers, "frozen_blocked_schema_drift")

    return FreshRewardBudgetAudit(
        structurally_valid=not blockers,
        status=str(status) if type(status) is str else "INVALID",
        launch_authorized=False,
        design_sha256=(
            declared_design_sha if type(declared_design_sha) is str else None
        ),
        authority_sha256=None,
        structural_blockers=tuple(blockers),
        unresolved=UNRESOLVED_REASON_CODES,
    )


def validate_blocked_receipt(value: object) -> FreshRewardBudgetAudit:
    """Require exact frozen blocked semantics and return its unresolved audit."""

    audit = audit_receipt(value)
    if not audit.structurally_valid:
        raise FreshRewardBudgetError(
            "fresh reward budget receipt invalid: "
            + ",".join(audit.structural_blockers)
        )
    return audit


def materialize_authority_sha256(value: object) -> str:
    """Refuse authority construction until a later reviewed numeric revision."""

    audit = validate_blocked_receipt(value)
    raise FreshRewardBudgetError(
        "fresh reward budget authority unavailable: "
        + ",".join(audit.unresolved)
    )


_NUMERIC_GRAPH_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "evidence_scope",
        "constructed",
        "source_closure_sha256",
        "common_parent_sha256",
        "resolved_manager_graph_sha256",
        "phase_support_contract_sha256",
        "landing_outcome_key_derivation_sha256",
        "c04_scorer_source_sha256",
        "policy_clock",
        "term_groups",
        "resolved_terms",
        "family_contract",
        "receipt_sha256",
    )
)
_NUMERIC_TAPE_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "evidence_scope",
        "constructed",
        "source_closure_sha256",
        "resolved_graph_receipt_sha256",
        "phase_support_contract_sha256",
        "fixed_tape_source_sha256",
        "action_chronology_root_sha256",
        "question_chronology_root_sha256",
        "teacher_chronology_root_sha256",
        "source_chronology_root_sha256",
        "payment_chronology_root_sha256",
        "diagnostic_unauthorized",
        "policy_clock",
        "shot_count",
        "flight_horizon_ticks",
        "flight_horizon_witness_sha256",
        "mailbox_horizon_ticks",
        "mailbox_horizon_witness_sha256",
        "tail_closure_tick",
        "flight_capacity",
        "mailbox_capacity",
        "observed_max_open_flights",
        "observed_max_open_mailboxes",
        "shots",
        "receipt_sha256",
    )
)
_NUMERIC_CANDIDATE_SET_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "evidence_scope",
        "source_closure_sha256",
        "resolved_graph_receipt_sha256",
        "fixed_tape_receipt_sha256",
        "selection_semantics",
        "lexicographic_priority_width",
        "candidates",
        "receipt_sha256",
    )
)
_TRUSTED_INPUT_ROOT_KEYS = frozenset(
    (
        "kind",
        "source_closure_sha256",
        "action_chronology_root_sha256",
        "question_chronology_root_sha256",
        "teacher_chronology_root_sha256",
        "source_chronology_root_sha256",
        "payment_chronology_root_sha256",
        "resolved_graph_receipt_sha256",
        "four_shot_tape_receipt_sha256",
        "candidate_set_receipt_sha256",
    )
)
_TERM_GROUP_KEYS = frozenset(
    (
        "motion",
        "ordered_ten",
        "recovery",
        "auxiliary",
        "selected_contact",
        "on_table",
        "placement",
    )
)
_FAMILY_CONTRACT = {
    "common_term_family_scope": "common_a_c",
    "common_terms_scheduled_for_a": True,
    "common_terms_scheduled_for_c": True,
    "placement_scheduled_for_a": True,
    "placement_scheduled_for_c": True,
    "placement_treatment_gain": {
        "a": {"numerator": 1, "denominator": 1},
        "c": {"numerator": 0, "denominator": 1},
    },
    "only_family_delta": "post_contact_placement_gain_v1",
    "dt_applications_per_evaluation": 1,
    "max_payments_per_shot": 1,
    "recovery_age_start": RECOVERY_AGE_START,
    "recovery_age_end": RECOVERY_AGE_END,
}
_SHOT_KEYS = frozenset(
    (
        "shot_index",
        "task_ref_sha256",
        "outcome_key_sha256",
        "target_float32_sha256",
        "reveal_tick",
        "deadline_tick",
        "next_boundary_tick",
        "phase_support",
        "ac_common_parity",
        "ordinary_outcome_telemetry",
        "candidate_unit_income",
    )
)
_PHASE_SUPPORT_KEYS = frozenset(
    (
        "motion_full_suffix_ticks",
        "motion_suffix_end_tick",
        "strike_source_tick",
        "strike_payment_tick",
        "selected_contact_source_tick",
        "selected_contact_payment_tick",
        "outcome_source_tick",
        "on_table_payment_tick",
        "placement_payment_tick",
        "recovery_age_ticks",
        "auxiliary_support_ticks",
    )
)
_AC_PARITY_KEYS = frozenset(
    (
        "a_common_facts_sha256",
        "c_common_facts_sha256",
        "a_common_raw_sha256",
        "c_common_raw_sha256",
        "a_common_payment_sha256",
        "c_common_payment_sha256",
        "a_placement_raw_sha256",
        "c_placement_raw_sha256",
        "a_placement_source_tick",
        "c_placement_source_tick",
        "a_placement_payment_tick",
        "c_placement_payment_tick",
        "a_treatment_gain",
        "c_treatment_gain",
    )
)
_ORDINARY_OUTCOME_TELEMETRY_KEYS = frozenset(
    (
        "scheduled_task_eligible",
        "eligibility_receipt_sha256",
        "selected_contact_raw",
        "on_table_raw",
        "placement_raw",
    )
)
_CANDIDATE_KEYS = frozenset(
    (
        "candidate_id",
        "lexicographic_priority",
        "manager_weights",
        "strike_kernel_profiles",
        "placement_profile",
    )
)
_MANAGER_WEIGHT_KEYS = frozenset(
    (
        "motion",
        "ordered_ten",
        "selected_contact",
        "on_table",
        "recovery",
        "auxiliary",
        "placement",
    )
)
_KERNEL_PROFILE_KEYS = frozenset(
    ("kernel_kind", "scale", "profile_sha256")
)
_RESOLVED_TERM_KEYS = frozenset(
    (
        "index",
        "group",
        "runtime_name",
        "a_callable_source_sha256",
        "c_callable_source_sha256",
        "a_params_sha256",
        "c_params_sha256",
        "a_cadence_semantics",
        "c_cadence_semantics",
        "a_scheduled",
        "c_scheduled",
        "a_family_scope",
        "c_family_scope",
        "manager_weight_source",
    )
)
_PLACEMENT_PROFILE_KEYS = frozenset(
    ("alpha_broad", "sigma_broad_m", "sigma_narrow_m", "profile_sha256")
)
_UNIT_INCOME_KEYS = frozenset(
    (
        "candidate_id",
        "raw_profile_sha256",
        "motion",
        "ten_reference",
        "ten_peak",
        "selected_contact_peak",
        "on_table_peak",
        "recovery",
        "auxiliary_typical",
        "auxiliary_p95",
        "auxiliary_negative",
        "placement_broad_reference",
        "placement_narrow_reference",
        "placement_peak",
    )
)
_COMMON_UNIT_INCOME_KEYS = (
    "motion",
    "selected_contact_peak",
    "on_table_peak",
    "recovery",
    "auxiliary_typical",
    "auxiliary_p95",
    "auxiliary_negative",
)
_STRIKE_PROFILE_UNIT_INCOME_KEYS = ("ten_reference", "ten_peak")
_PLACEMENT_PROFILE_UNIT_INCOME_KEYS = (
    "placement_broad_reference",
    "placement_narrow_reference",
    "placement_peak",
)
_EVENT_KEYS = frozenset(("source_tick", "payment_tick", "raw"))
_MATERIALIZATION_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "status",
        "budget_authority_ready",
        "launch_authorized",
        "authority_scope",
        "authority_sha256",
        "authority_payload",
        "input_receipts",
        "policy_clock",
        "global_evidence_blockers",
        "candidate_evaluations",
        "selection",
        "selected_numeric_parameters",
        "materialization_sha256",
    )
)

_OWNER_CONSTRUCTION_KEY = object()


class NumericRewardAuthorityReceipt:
    """Opaque same-process identity for one owner-issued numeric budget.

    The portable materialization remains useful diagnostic evidence, but its
    content SHA is not a capability.  Production consumers must hold this
    exact registry-backed object and ask its issuing owner for a clone of the
    validated payload.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("numeric reward authority receipts are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("numeric reward authority receipts are immutable")

    def __copy__(self):
        raise TypeError("numeric reward authority receipts cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("numeric reward authority receipts cannot be copied")

    def __reduce__(self):
        raise TypeError("numeric reward authority receipts cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("numeric reward authority receipts cannot be serialized")


@dataclass(frozen=True)
class _NumericAuthorityRecord:
    materialization_sha256: str
    authority_sha256: str
    authority_payload: Mapping[str, object]


class NumericRewardAuthorityOwner:
    """Registry owner that converts verified evidence into one-shot authority."""

    __slots__ = ("_pending", "_consumed")

    def __init__(self, construction_key: object) -> None:
        if construction_key is not _OWNER_CONSTRUCTION_KEY:
            raise TypeError("numeric reward authority owners require a factory")
        self._pending: dict[NumericRewardAuthorityReceipt, _NumericAuthorityRecord] = {}
        self._consumed: weakref.WeakSet[NumericRewardAuthorityReceipt] = weakref.WeakSet()

    def __copy__(self):
        raise TypeError("numeric reward authority owners cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("numeric reward authority owners cannot be copied")

    def __reduce__(self):
        raise TypeError("numeric reward authority owners cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("numeric reward authority owners cannot be serialized")

    def issue_for_diagnostic(
        self,
        resolved_graph_receipt: object,
        four_shot_tape_receipt: object,
        candidate_set_receipt: object,
        *,
        trusted_input_roots: object,
    ) -> NumericRewardAuthorityReceipt:
        """Issue from recomputed evidence without granting launch authority."""

        materialization = materialize_numeric_authority(
            resolved_graph_receipt,
            four_shot_tape_receipt,
            candidate_set_receipt,
            trusted_input_roots=trusted_input_roots,
        )
        if materialization["status"] != READY_STATUS:
            raise FreshRewardBudgetError(
                "numeric authority cannot be issued from BLOCKED materialization"
            )
        validated = validate_numeric_materialization_receipt(
            materialization,
            resolved_graph_receipt=resolved_graph_receipt,
            four_shot_tape_receipt=four_shot_tape_receipt,
            candidate_set_receipt=candidate_set_receipt,
            trusted_input_roots=trusted_input_roots,
        )
        receipt = object.__new__(NumericRewardAuthorityReceipt)
        record = _NumericAuthorityRecord(
            materialization_sha256=str(validated["materialization_sha256"]),
            authority_sha256=str(validated["authority_sha256"]),
            authority_payload=deepcopy(validated["authority_payload"]),
        )
        self._pending[receipt] = record
        return receipt

    def consume(
        self, receipt: NumericRewardAuthorityReceipt
    ) -> dict[str, object]:
        """Consume one exact owner-issued receipt and return a detached payload."""

        if type(receipt) is not NumericRewardAuthorityReceipt:
            raise FreshRewardBudgetError("numeric reward receipt type differs")
        if receipt in self._consumed:
            raise FreshRewardBudgetError("numeric reward receipt was already consumed")
        record = self._pending.pop(receipt, None)
        if record is None:
            raise FreshRewardBudgetError(
                "numeric reward receipt is foreign or not owner-issued"
            )
        self._consumed.add(receipt)
        return {
            "kind": NUMERIC_AUTHORITY_KIND,
            "scope": NUMERIC_AUTHORITY_SCOPE,
            "diagnostic_unauthorized": NUMERIC_DIAGNOSTIC_UNAUTHORIZED,
            "runtime_integrated": NUMERIC_RUNTIME_INTEGRATED,
            "launch_authorized": NUMERIC_LAUNCH_AUTHORIZED,
            "materialization_sha256": record.materialization_sha256,
            "authority_sha256": record.authority_sha256,
            "authority_payload": deepcopy(record.authority_payload),
        }


def make_diagnostic_numeric_reward_authority_owner() -> NumericRewardAuthorityOwner:
    """Create the explicit diagnostic owner; never a production factory."""

    return NumericRewardAuthorityOwner(_OWNER_CONSTRUCTION_KEY)


def construct_production_numeric_reward_authority_owner(
    *,
    constructed_runtime_reward_graph_producer: object,
    real_four_shot_unit_income_phase_support_producer: object,
    launcher_owned_finite_candidate_set_producer: object,
    fourteen_live_reward_consumer_graph: object,
) -> NumericRewardAuthorityOwner:
    """Keep production HOLD until all named real upstream/downstream owners exist."""

    del (
        constructed_runtime_reward_graph_producer,
        real_four_shot_unit_income_phase_support_producer,
        launcher_owned_finite_candidate_set_producer,
        fourteen_live_reward_consumer_graph,
    )
    raise FreshRewardNumericProductionHold(
        "numeric reward production owner remains HOLD: "
        + ",".join(NUMERIC_PRODUCTION_HOLD_REASONS)
    )


@dataclass(frozen=True)
class _NumericCandidate:
    candidate_id: str
    priority: tuple[int, ...]
    motion_weights: Mapping[str, Fraction]
    ten_weights: Mapping[str, Fraction]
    selected_contact_weight: Fraction
    on_table_weight: Fraction
    recovery_weights: Mapping[str, Fraction]
    auxiliary_weights: Mapping[str, Fraction]
    placement_weight: Fraction
    raw_profile_sha256: str
    strike_numeric_profile_sha256: str
    placement_numeric_profile_sha256: str
    normalized_parameters: Mapping[str, object]
    admissibility_reasons: tuple[str, ...]


def _require_plain_json(value: object, *, label: str) -> None:
    """Reject floats, subclasses, non-string keys, and non-JSON containers."""

    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        raise FreshRewardBudgetError(f"{label} contains a binary float")
    if type(value) is list:
        for index, item in enumerate(value):
            _require_plain_json(item, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise FreshRewardBudgetError(
                    f"{label} contains a non-string mapping key"
                )
            _require_plain_json(item, label=f"{label}.{key}")
        return
    raise FreshRewardBudgetError(
        f"{label} must be a plain canonical JSON tree, got {type(value).__name__}"
    )


def _exact_keys(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise FreshRewardBudgetError(f"{label} must be a plain JSON object")
    actual = frozenset(value)
    if actual != expected:
        raise FreshRewardBudgetError(
            f"{label} keys differ: missing={sorted(expected - actual)!r} "
            f"unknown={sorted(actual - expected)!r}"
        )
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreshRewardBudgetError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise FreshRewardBudgetError(f"{label} must be a non-empty string")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise FreshRewardBudgetError(f"{label} must be an exact integer{suffix}")
    return value


def _require_fraction(
    value: object,
    *,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> Fraction:
    result = _as_fraction(value)
    if result is None:
        raise FreshRewardBudgetError(
            f"{label} must be an exact integer, decimal string, or ratio"
        )
    if positive and result <= 0:
        raise FreshRewardBudgetError(f"{label} must be > 0")
    if nonnegative and result < 0:
        raise FreshRewardBudgetError(f"{label} must be >= 0")
    return result


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _sealed_numeric_receipt(
    value: object,
    *,
    expected_kind: str,
    expected_keys: frozenset[str],
    label: str,
) -> tuple[dict[str, object], str]:
    _require_plain_json(value, label=label)
    receipt = _exact_keys(value, expected_keys, label=label)
    if receipt["schema_version"] != NUMERIC_SCHEMA_VERSION or type(
        receipt["schema_version"]
    ) is not int:
        raise FreshRewardBudgetError(f"{label}.schema_version must equal 2")
    if receipt["kind"] != expected_kind:
        raise FreshRewardBudgetError(f"{label}.kind differs")
    declared = _require_sha256(
        receipt["receipt_sha256"], label=f"{label}.receipt_sha256"
    )
    payload = {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    }
    if canonical_sha256(payload) != declared:
        raise FreshRewardBudgetError(f"{label}.receipt_sha256 mismatch")
    return receipt, declared


def _validate_trusted_input_roots(
    value: object,
    *,
    graph: Mapping[str, object],
    tape: Mapping[str, object],
    candidate_set: Mapping[str, object],
) -> dict[str, object]:
    """Validate launcher-owned roots separately from self-sealed receipts."""

    _require_plain_json(value, label="trusted input roots")
    roots = _exact_keys(
        value, _TRUSTED_INPUT_ROOT_KEYS, label="trusted input roots"
    )
    if roots["kind"] != TRUSTED_INPUT_ROOT_KIND:
        raise FreshRewardBudgetError("trusted input roots kind differs")
    expected = {
        "source_closure_sha256": graph["source_closure_sha256"],
        "action_chronology_root_sha256": tape[
            "action_chronology_root_sha256"
        ],
        "question_chronology_root_sha256": tape[
            "question_chronology_root_sha256"
        ],
        "teacher_chronology_root_sha256": tape[
            "teacher_chronology_root_sha256"
        ],
        "source_chronology_root_sha256": tape[
            "source_chronology_root_sha256"
        ],
        "payment_chronology_root_sha256": tape[
            "payment_chronology_root_sha256"
        ],
        "resolved_graph_receipt_sha256": graph["receipt_sha256"],
        "four_shot_tape_receipt_sha256": tape["receipt_sha256"],
        "candidate_set_receipt_sha256": candidate_set["receipt_sha256"],
    }
    for field, actual in expected.items():
        trusted = _require_sha256(
            roots[field], label=f"trusted input roots.{field}"
        )
        if trusted != actual:
            raise FreshRewardBudgetError(
                f"trusted input roots {field} differs"
            )
    return roots


def _validate_policy_clock(value: object, *, label: str) -> None:
    expected = {
        "policy_dt": dict(POLICY_DT),
        "ppo_gamma": dict(PPO_GAMMA),
        "dt_application_semantics": DT_APPLICATION_SEMANTICS,
    }
    if not _frozen_exact_equal(value, expected):
        raise FreshRewardBudgetError(
            f"{label} must freeze dt=1/50, gamma=99/100, and dt-once semantics"
        )


def _term_name_list(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not list or (not value and not allow_empty):
        raise FreshRewardBudgetError(f"{label} must be a non-empty ordered list")
    names = tuple(_require_text(item, label=f"{label}[]") for item in value)
    if len(set(names)) != len(names):
        raise FreshRewardBudgetError(f"{label} contains duplicate term names")
    return names


def validate_numeric_resolved_graph_receipt(
    value: object,
) -> dict[str, object]:
    """Validate the constructed common Reward graph required by schema 2."""

    receipt, _declared = _sealed_numeric_receipt(
        value,
        expected_kind=NUMERIC_GRAPH_KIND,
        expected_keys=_NUMERIC_GRAPH_KEYS,
        label="resolved graph receipt",
    )
    if receipt["evidence_scope"] != CONSTRUCTED_EVIDENCE_SCOPE:
        raise FreshRewardBudgetError(
            "resolved graph receipt must be constructed-runtime evidence"
        )
    if receipt["constructed"] is not True:
        raise FreshRewardBudgetError("resolved graph receipt constructed must be true")
    for field in (
        "source_closure_sha256",
        "common_parent_sha256",
        "resolved_manager_graph_sha256",
        "phase_support_contract_sha256",
        "landing_outcome_key_derivation_sha256",
    ):
        _require_sha256(receipt[field], label=f"resolved graph receipt.{field}")
    if receipt["c04_scorer_source_sha256"] != C04_SOURCE_SHA256:
        raise FreshRewardBudgetError("resolved graph C04 scorer source SHA differs")
    if _file_sha256(Path(c04.__file__).resolve()) != C04_SOURCE_SHA256:
        raise FreshRewardBudgetError(
            "local C04 scorer source SHA differs from frozen authority"
        )
    _validate_policy_clock(
        receipt["policy_clock"], label="resolved graph receipt.policy_clock"
    )

    groups = _exact_keys(
        receipt["term_groups"], _TERM_GROUP_KEYS, label="term_groups"
    )
    motion = _term_name_list(groups["motion"], label="term_groups.motion")
    ordered_ten = _term_name_list(
        groups["ordered_ten"], label="term_groups.ordered_ten"
    )
    recovery = _term_name_list(
        groups["recovery"], label="term_groups.recovery"
    )
    auxiliary = _term_name_list(
        groups["auxiliary"], label="term_groups.auxiliary"
    )
    if ordered_ten != ORDERED_CONTACT_TERM_NAMES:
        raise FreshRewardBudgetError("resolved graph ordered-ten ABI differs")
    scalar_names = tuple(
        _require_text(groups[key], label=f"term_groups.{key}")
        for key in ("selected_contact", "on_table", "placement")
    )
    all_names = motion + ordered_ten + recovery + auxiliary + scalar_names
    if len(set(all_names)) != len(all_names):
        raise FreshRewardBudgetError("resolved graph term groups overlap")
    expected_resolved = (
        tuple(("motion", name) for name in motion)
        + tuple(("ordered_ten", name) for name in ordered_ten)
        + (("selected_contact", scalar_names[0]),)
        + (("on_table", scalar_names[1]),)
        + tuple(("recovery", name) for name in recovery)
        + tuple(("auxiliary", name) for name in auxiliary)
        + (("placement", scalar_names[2]),)
    )
    resolved_terms = receipt["resolved_terms"]
    if type(resolved_terms) is not list or len(resolved_terms) != len(
        expected_resolved
    ):
        raise FreshRewardBudgetError("resolved graph term closure length differs")
    for index, ((expected_group, expected_name), term_value) in enumerate(
        zip(expected_resolved, resolved_terms)
    ):
        term = _exact_keys(
            term_value, _RESOLVED_TERM_KEYS, label=f"resolved_terms[{index}]"
        )
        if term["index"] != index or type(term["index"]) is not int:
            raise FreshRewardBudgetError("resolved graph term index differs")
        if term["group"] != expected_group or term["runtime_name"] != expected_name:
            raise FreshRewardBudgetError("resolved graph term order/group differs")
        for field in (
            "a_callable_source_sha256",
            "c_callable_source_sha256",
            "a_params_sha256",
            "c_params_sha256",
        ):
            _require_sha256(term[field], label=f"resolved_terms[{index}].{field}")
        if term["a_callable_source_sha256"] != term["c_callable_source_sha256"]:
            raise FreshRewardBudgetError("resolved graph A/C callable source differs")
        if term["a_params_sha256"] != term["c_params_sha256"]:
            raise FreshRewardBudgetError("resolved graph A/C term params differ")
        if term["a_cadence_semantics"] != term["c_cadence_semantics"]:
            raise FreshRewardBudgetError("resolved graph A/C cadence differs")
        _require_text(
            term["a_cadence_semantics"],
            label=f"resolved_terms[{index}].a_cadence_semantics",
        )
        if term["a_scheduled"] is not True or term["c_scheduled"] is not True:
            raise FreshRewardBudgetError("resolved graph A/C term must be scheduled")
        if term["a_family_scope"] != "common_a_c" or term[
            "c_family_scope"
        ] != "common_a_c":
            raise FreshRewardBudgetError("resolved graph term family scope differs")
        if term["manager_weight_source"] != "explicit_candidate_set_v2":
            raise FreshRewardBudgetError(
                "resolved graph manager weight source must be candidate-bound"
            )
    if not _frozen_exact_equal(receipt["family_contract"], _FAMILY_CONTRACT):
        raise FreshRewardBudgetError("resolved graph A/C family contract differs")
    return receipt


def validate_numeric_four_shot_tape_receipt(
    value: object,
) -> dict[str, object]:
    """Validate the sealed shape of one constructed four-shot numeric tape."""

    receipt, _declared = _sealed_numeric_receipt(
        value,
        expected_kind=NUMERIC_TAPE_KIND,
        expected_keys=_NUMERIC_TAPE_KEYS,
        label="four-shot tape receipt",
    )
    if receipt["evidence_scope"] != CONSTRUCTED_EVIDENCE_SCOPE:
        raise FreshRewardBudgetError(
            "four-shot tape receipt must be constructed-runtime evidence"
        )
    if receipt["constructed"] is not True:
        raise FreshRewardBudgetError("four-shot tape constructed must be true")
    if receipt["diagnostic_unauthorized"] is not True:
        raise FreshRewardBudgetError(
            "four-shot tape must explicitly remain diagnostic_unauthorized"
        )
    for field in (
        "source_closure_sha256",
        "resolved_graph_receipt_sha256",
        "phase_support_contract_sha256",
        "fixed_tape_source_sha256",
        "action_chronology_root_sha256",
        "question_chronology_root_sha256",
        "teacher_chronology_root_sha256",
        "source_chronology_root_sha256",
        "payment_chronology_root_sha256",
    ):
        _require_sha256(receipt[field], label=f"four-shot tape receipt.{field}")
    _validate_policy_clock(
        receipt["policy_clock"], label="four-shot tape receipt.policy_clock"
    )
    if type(receipt["shot_count"]) is not int or receipt["shot_count"] != 4:
        raise FreshRewardBudgetError("numeric materializer requires exactly four shots")
    if type(receipt["shots"]) is not list or len(receipt["shots"]) != 4:
        raise FreshRewardBudgetError("four-shot tape must contain exactly four rows")
    for field in (
        "flight_horizon_ticks",
        "mailbox_horizon_ticks",
        "flight_capacity",
        "mailbox_capacity",
    ):
        _require_int(receipt[field], label=field, minimum=1)
    _require_int(receipt["tail_closure_tick"], label="tail_closure_tick", minimum=1)
    for field in (
        "flight_horizon_witness_sha256",
        "mailbox_horizon_witness_sha256",
    ):
        _require_sha256(receipt[field], label=field)
    for field in ("observed_max_open_flights", "observed_max_open_mailboxes"):
        _require_int(receipt[field], label=field, minimum=0)
    for index, shot_value in enumerate(receipt["shots"]):
        shot = _exact_keys(
            shot_value, _SHOT_KEYS, label=f"four-shot tape.shots[{index}]"
        )
        _require_int(shot["shot_index"], label=f"shots[{index}].shot_index", minimum=0)
        for field in ("task_ref_sha256", "outcome_key_sha256", "target_float32_sha256"):
            _require_sha256(shot[field], label=f"shots[{index}].{field}")
        for field in ("reveal_tick", "deadline_tick", "next_boundary_tick"):
            _require_int(shot[field], label=f"shots[{index}].{field}", minimum=0)
        phase = _exact_keys(
            shot["phase_support"],
            _PHASE_SUPPORT_KEYS,
            label=f"shots[{index}].phase_support",
        )
        for field in (
            "motion_suffix_end_tick",
            "strike_source_tick",
            "strike_payment_tick",
            "selected_contact_source_tick",
            "selected_contact_payment_tick",
            "outcome_source_tick",
            "on_table_payment_tick",
            "placement_payment_tick",
        ):
            _require_int(phase[field], label=f"shots[{index}].phase_support.{field}", minimum=0)
        if type(phase["motion_full_suffix_ticks"]) is not list:
            raise FreshRewardBudgetError(
                f"shots[{index}].motion_full_suffix_ticks must be a list"
            )
        for tick in phase["motion_full_suffix_ticks"]:
            _require_int(tick, label="motion_full_suffix_tick", minimum=0)
        if type(phase["recovery_age_ticks"]) is not list:
            raise FreshRewardBudgetError(
                f"shots[{index}].recovery_age_ticks must be a list"
            )
        for age in phase["recovery_age_ticks"]:
            _require_int(age, label="recovery_age_tick", minimum=0)
        if type(phase["auxiliary_support_ticks"]) is not list:
            raise FreshRewardBudgetError(
                f"shots[{index}].auxiliary_support_ticks must be a list"
            )
        for tick in phase["auxiliary_support_ticks"]:
            _require_int(tick, label="auxiliary_support_tick", minimum=0)
        parity = _exact_keys(
            shot["ac_common_parity"],
            _AC_PARITY_KEYS,
            label=f"shots[{index}].ac_common_parity",
        )
        for field in (
            "a_common_facts_sha256",
            "c_common_facts_sha256",
            "a_common_raw_sha256",
            "c_common_raw_sha256",
            "a_common_payment_sha256",
            "c_common_payment_sha256",
            "a_placement_raw_sha256",
            "c_placement_raw_sha256",
        ):
            _require_sha256(parity[field], label=f"shots[{index}].{field}")
        for field in (
            "a_placement_source_tick",
            "c_placement_source_tick",
            "a_placement_payment_tick",
            "c_placement_payment_tick",
        ):
            _require_int(parity[field], label=f"shots[{index}].{field}", minimum=0)
        _require_fraction(parity["a_treatment_gain"], label="a_treatment_gain")
        _require_fraction(parity["c_treatment_gain"], label="c_treatment_gain")
        telemetry = _exact_keys(
            shot["ordinary_outcome_telemetry"],
            _ORDINARY_OUTCOME_TELEMETRY_KEYS,
            label=f"shots[{index}].ordinary_outcome_telemetry",
        )
        if telemetry["scheduled_task_eligible"] is not True:
            raise FreshRewardBudgetError(
                "ordinary outcome telemetry requires an eligible task receipt"
            )
        _require_sha256(
            telemetry["eligibility_receipt_sha256"],
            label="ordinary outcome telemetry eligibility receipt",
        )
        for field in ("selected_contact_raw", "on_table_raw", "placement_raw"):
            raw = _require_fraction(
                telemetry[field], label=f"ordinary outcome telemetry.{field}"
            )
            if raw < 0 or raw > 1:
                raise FreshRewardBudgetError(
                    f"ordinary outcome telemetry.{field} must be in [0,1]"
                )
        if type(shot["candidate_unit_income"]) is not list:
            raise FreshRewardBudgetError(
                f"shots[{index}].candidate_unit_income must be a list"
            )
    return receipt


def _weight_map(
    value: object,
    names: Sequence[str],
    *,
    label: str,
    positive: bool,
) -> tuple[dict[str, Fraction], dict[str, object]]:
    expected = frozenset(names)
    mapping = _exact_keys(value, expected, label=label)
    parsed: dict[str, Fraction] = {}
    normalized: dict[str, object] = {}
    for name in names:
        number = _require_fraction(
            mapping[name], label=f"{label}.{name}", positive=positive
        )
        parsed[name] = number
        normalized[name] = _fraction_payload(number)
    return parsed, normalized


def _parse_numeric_candidates(
    value: object,
    *,
    graph: Mapping[str, object],
    tape: Mapping[str, object],
) -> tuple[dict[str, object], tuple[_NumericCandidate, ...]]:
    receipt, _declared = _sealed_numeric_receipt(
        value,
        expected_kind=NUMERIC_CANDIDATE_SET_KIND,
        expected_keys=_NUMERIC_CANDIDATE_SET_KEYS,
        label="candidate-set receipt",
    )
    if receipt["evidence_scope"] != EXPLICIT_CANDIDATE_SCOPE:
        raise FreshRewardBudgetError(
            "candidate-set receipt must declare explicit finite candidates"
        )
    if receipt["source_closure_sha256"] != graph["source_closure_sha256"]:
        raise FreshRewardBudgetError("candidate-set source closure differs")
    if receipt["resolved_graph_receipt_sha256"] != graph["receipt_sha256"]:
        raise FreshRewardBudgetError("candidate-set graph receipt binding differs")
    if receipt["fixed_tape_receipt_sha256"] != tape["receipt_sha256"]:
        raise FreshRewardBudgetError("candidate-set fixed-tape binding differs")
    if receipt["selection_semantics"] != LEXICOGRAPHIC_SELECTION_SEMANTICS:
        raise FreshRewardBudgetError("candidate-set selection semantics differ")
    priority_width = _require_int(
        receipt["lexicographic_priority_width"],
        label="lexicographic_priority_width",
        minimum=1,
    )
    candidates_value = receipt["candidates"]
    if type(candidates_value) is not list or not candidates_value:
        raise FreshRewardBudgetError("candidate-set candidates must be non-empty")

    groups = graph["term_groups"]
    motion_names = tuple(groups["motion"])
    ten_names = tuple(groups["ordered_ten"])
    recovery_names = tuple(groups["recovery"])
    auxiliary_names = tuple(groups["auxiliary"])
    output: list[_NumericCandidate] = []
    seen_ids: set[str] = set()
    for index, candidate_value in enumerate(candidates_value):
        label = f"candidate-set.candidates[{index}]"
        candidate = _exact_keys(candidate_value, _CANDIDATE_KEYS, label=label)
        candidate_id = _require_text(
            candidate["candidate_id"], label=f"{label}.candidate_id"
        )
        if candidate_id in seen_ids:
            raise FreshRewardBudgetError("candidate ids must be unique")
        seen_ids.add(candidate_id)
        priority_value = candidate["lexicographic_priority"]
        if type(priority_value) is not list or len(priority_value) != priority_width:
            raise FreshRewardBudgetError(
                f"{label}.lexicographic_priority width differs"
            )
        priority = tuple(
            _require_int(item, label=f"{label}.lexicographic_priority[]", minimum=0)
            for item in priority_value
        )

        manager = _exact_keys(
            candidate["manager_weights"],
            _MANAGER_WEIGHT_KEYS,
            label=f"{label}.manager_weights",
        )
        motion_weights, normalized_motion = _weight_map(
            manager["motion"],
            motion_names,
            label=f"{label}.manager_weights.motion",
            positive=True,
        )
        ten_weights, normalized_ten = _weight_map(
            manager["ordered_ten"],
            ten_names,
            label=f"{label}.manager_weights.ordered_ten",
            positive=True,
        )
        recovery_weights, normalized_recovery = _weight_map(
            manager["recovery"],
            recovery_names,
            label=f"{label}.manager_weights.recovery",
            positive=True,
        )
        auxiliary_weights, normalized_auxiliary = _weight_map(
            manager["auxiliary"],
            auxiliary_names,
            label=f"{label}.manager_weights.auxiliary",
            positive=False,
        )
        selected_contact_weight = _require_fraction(
            manager["selected_contact"],
            label=f"{label}.manager_weights.selected_contact",
            positive=True,
        )
        on_table_weight = _require_fraction(
            manager["on_table"],
            label=f"{label}.manager_weights.on_table",
            positive=True,
        )
        placement_weight = _require_fraction(
            manager["placement"],
            label=f"{label}.manager_weights.placement",
            positive=True,
        )

        kernel_profiles = _exact_keys(
            candidate["strike_kernel_profiles"],
            frozenset(ten_names),
            label=f"{label}.strike_kernel_profiles",
        )
        normalized_kernels: dict[str, object] = {}
        for name in ten_names:
            profile = _exact_keys(
                kernel_profiles[name],
                _KERNEL_PROFILE_KEYS,
                label=f"{label}.strike_kernel_profiles.{name}",
            )
            kind = _require_text(
                profile["kernel_kind"],
                label=f"{label}.strike_kernel_profiles.{name}.kernel_kind",
            )
            if kind != NUMERIC_STRIKE_KERNEL_KINDS[name]:
                raise FreshRewardBudgetError(
                    f"{label}.strike_kernel_profiles.{name}.kernel_kind differs"
                )
            scale = _require_fraction(
                profile["scale"],
                label=f"{label}.strike_kernel_profiles.{name}.scale",
                positive=True,
            )
            profile_sha256 = _require_sha256(
                profile["profile_sha256"],
                label=f"{label}.strike_kernel_profiles.{name}.profile_sha256",
            )
            normalized_kernels[name] = {
                "kernel_kind": kind,
                "scale": _fraction_payload(scale),
                "profile_sha256": profile_sha256,
            }

        placement = _exact_keys(
            candidate["placement_profile"],
            _PLACEMENT_PROFILE_KEYS,
            label=f"{label}.placement_profile",
        )
        alpha = _require_fraction(
            placement["alpha_broad"],
            label=f"{label}.placement_profile.alpha_broad",
            positive=True,
        )
        sigma_broad = _require_fraction(
            placement["sigma_broad_m"],
            label=f"{label}.placement_profile.sigma_broad_m",
            positive=True,
        )
        sigma_narrow = _require_fraction(
            placement["sigma_narrow_m"],
            label=f"{label}.placement_profile.sigma_narrow_m",
            positive=True,
        )
        if alpha >= 1:
            raise FreshRewardBudgetError(
                f"{label}.placement_profile.alpha_broad must be < 1"
            )
        if sigma_broad <= sigma_narrow:
            raise FreshRewardBudgetError(
                f"{label} placement sigmas must satisfy broad > narrow"
            )
        placement_sha256 = _require_sha256(
            placement["profile_sha256"],
            label=f"{label}.placement_profile.profile_sha256",
        )
        normalized_placement = {
            "alpha_broad": _fraction_payload(alpha),
            "sigma_broad_m": _fraction_payload(sigma_broad),
            "sigma_narrow_m": _fraction_payload(sigma_narrow),
            "profile_sha256": placement_sha256,
        }
        strike_numeric_profile_sha256 = canonical_sha256(
            {
                name: {
                    "kernel_kind": normalized_kernels[name]["kernel_kind"],
                    "scale": normalized_kernels[name]["scale"],
                }
                for name in ten_names
            }
        )
        placement_numeric_profile_sha256 = canonical_sha256(
            {
                "alpha_broad": normalized_placement["alpha_broad"],
                "sigma_broad_m": normalized_placement["sigma_broad_m"],
                "sigma_narrow_m": normalized_placement["sigma_narrow_m"],
            }
        )
        raw_profile = {
            "strike_kernel_profiles": normalized_kernels,
            "placement_profile": normalized_placement,
        }
        raw_profile_sha256 = canonical_sha256(raw_profile)
        admissibility_reasons: list[str] = []
        if all(
            ten_weights[name] == Fraction(legacy_value)
            for name, legacy_value in LEGACY_WINDOW_MANAGER_WEIGHTS.items()
        ):
            _append_reason(
                admissibility_reasons,
                "legacy_3_11_tick_weight_table_forbidden",
            )
        if ten_weights["paddle_center_proximity"] == Fraction(
            LEGACY_PROXIMITY_MANAGER_WEIGHT
        ):
            _append_reason(
                admissibility_reasons,
                "legacy_proximity_weight_240_forbidden",
            )
        if on_table_weight == Fraction(LEGACY_LANDING_MANAGER_WEIGHT):
            _append_reason(
                admissibility_reasons,
                "legacy_landing_weight_700_forbidden",
            )
        if (
            alpha == PROHIBITED_FIXTURE_PLACEMENT_PROFILE["alpha_broad"]
            and sigma_broad
            == PROHIBITED_FIXTURE_PLACEMENT_PROFILE["sigma_broad_m"]
            and sigma_narrow
            == PROHIBITED_FIXTURE_PLACEMENT_PROFILE["sigma_narrow_m"]
        ):
            _append_reason(
                admissibility_reasons,
                "fixture_placement_profile_0p4_0p5_0p1_forbidden",
            )
        normalized_parameters = {
            "candidate_id": candidate_id,
            "lexicographic_priority": list(priority),
            "manager_weights": {
                "motion": normalized_motion,
                "ordered_ten": normalized_ten,
                "selected_contact": _fraction_payload(selected_contact_weight),
                "on_table": _fraction_payload(on_table_weight),
                "recovery": normalized_recovery,
                "auxiliary": normalized_auxiliary,
                "placement": _fraction_payload(placement_weight),
            },
            "strike_kernel_profiles": normalized_kernels,
            "placement_profile": normalized_placement,
            "treatment_gain": deepcopy(_FAMILY_CONTRACT["placement_treatment_gain"]),
            "raw_profile_sha256": raw_profile_sha256,
        }
        output.append(
            _NumericCandidate(
                candidate_id=candidate_id,
                priority=priority,
                motion_weights=motion_weights,
                ten_weights=ten_weights,
                selected_contact_weight=selected_contact_weight,
                on_table_weight=on_table_weight,
                recovery_weights=recovery_weights,
                auxiliary_weights=auxiliary_weights,
                placement_weight=placement_weight,
                raw_profile_sha256=raw_profile_sha256,
                strike_numeric_profile_sha256=(
                    strike_numeric_profile_sha256
                ),
                placement_numeric_profile_sha256=(
                    placement_numeric_profile_sha256
                ),
                normalized_parameters=normalized_parameters,
                admissibility_reasons=tuple(admissibility_reasons),
            )
        )
    return receipt, tuple(output)


def validate_numeric_candidate_set_receipt(
    value: object,
    *,
    resolved_graph_receipt: object,
    four_shot_tape_receipt: object,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Validate explicit candidates and return their ordered identifiers."""

    graph = validate_numeric_resolved_graph_receipt(resolved_graph_receipt)
    tape = validate_numeric_four_shot_tape_receipt(four_shot_tape_receipt)
    receipt, candidates = _parse_numeric_candidates(value, graph=graph, tape=tape)
    return receipt, tuple(candidate.candidate_id for candidate in candidates)


def _event_rows(value: object, *, label: str) -> tuple[tuple[int, int, Fraction], ...]:
    if type(value) is not list or not value:
        raise FreshRewardBudgetError(f"{label} must be a non-empty event list")
    output: list[tuple[int, int, Fraction]] = []
    for index, row_value in enumerate(value):
        row = _exact_keys(row_value, _EVENT_KEYS, label=f"{label}[{index}]")
        source_tick = _require_int(
            row["source_tick"], label=f"{label}[{index}].source_tick", minimum=0
        )
        payment_tick = _require_int(
            row["payment_tick"], label=f"{label}[{index}].payment_tick", minimum=0
        )
        raw = _require_fraction(row["raw"], label=f"{label}[{index}].raw")
        output.append((source_tick, payment_tick, raw))
    return tuple(output)


def _event_map(
    value: object,
    names: Sequence[str],
    *,
    label: str,
) -> dict[str, tuple[tuple[int, int, Fraction], ...]]:
    mapping = _exact_keys(value, frozenset(names), label=label)
    return {
        name: _event_rows(mapping[name], label=f"{label}.{name}")
        for name in names
    }


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _max_inclusive_overlap(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    points = sorted({tick for interval in intervals for tick in interval})
    return max(
        sum(1 for start, end in intervals if start <= point <= end)
        for point in points
    )


def _global_numeric_evidence_blockers(
    graph: Mapping[str, object],
    tape: Mapping[str, object],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if tape["source_closure_sha256"] != graph["source_closure_sha256"]:
        _append_reason(reasons, "source_closure_mismatch")
    if tape["resolved_graph_receipt_sha256"] != graph["receipt_sha256"]:
        _append_reason(reasons, "resolved_graph_receipt_binding_mismatch")
    if (
        tape["phase_support_contract_sha256"]
        != graph["phase_support_contract_sha256"]
    ):
        _append_reason(reasons, "phase_support_contract_mismatch")
    if not _frozen_exact_equal(tape["policy_clock"], graph["policy_clock"]):
        _append_reason(reasons, "policy_clock_mismatch")

    flight_intervals: list[tuple[int, int]] = []
    mailbox_intervals: list[tuple[int, int]] = []
    task_refs: set[str] = set()
    outcome_keys: set[str] = set()
    targets: list[str] = []
    shots = tape["shots"]
    horizon = int(tape["flight_horizon_ticks"])
    mailbox_horizon = int(tape["mailbox_horizon_ticks"])
    first_reveal = int(shots[0]["reveal_tick"])
    cadence = int(shots[1]["reveal_tick"]) - first_reveal
    deadline_offset = int(shots[0]["deadline_tick"]) - first_reveal
    if cadence <= 0:
        _append_reason(reasons, "frozen_cadence_must_be_positive")
    if deadline_offset <= 0:
        _append_reason(reasons, "frozen_deadline_offset_must_be_positive")
    if cadence - deadline_offset < MIN_DEADLINE_TO_NEXT_REVEAL_TICKS:
        _append_reason(reasons, "frozen_cadence_c_minus_d_below_78")
    for index, shot in enumerate(shots):
        prefix = f"shot_{index}"
        if shot["shot_index"] != index:
            _append_reason(reasons, f"{prefix}_index_mismatch")
        reveal = int(shot["reveal_tick"])
        deadline = int(shot["deadline_tick"])
        boundary = int(shot["next_boundary_tick"])
        phase = shot["phase_support"]
        if reveal != first_reveal + index * cadence:
            _append_reason(reasons, f"{prefix}_frozen_cadence_reveal_drift")
        if deadline != reveal + deadline_offset:
            _append_reason(reasons, f"{prefix}_frozen_deadline_offset_drift")
        if boundary != reveal + cadence:
            _append_reason(reasons, f"{prefix}_frozen_cadence_boundary_drift")
        if not reveal < deadline < boundary:
            _append_reason(reasons, f"{prefix}_reveal_deadline_boundary_order")
        if boundary - deadline < MIN_DEADLINE_TO_NEXT_REVEAL_TICKS:
            _append_reason(reasons, f"{prefix}_cadence_c_minus_d_below_78")
        if index < len(shots) - 1 and boundary != shots[index + 1]["reveal_tick"]:
            _append_reason(reasons, f"{prefix}_next_reveal_binding_mismatch")

        motion_ticks = phase["motion_full_suffix_ticks"]
        motion_end = int(phase["motion_suffix_end_tick"])
        expected_motion_ticks = list(range(reveal, motion_end + 1))
        if motion_ticks != expected_motion_ticks:
            _append_reason(reasons, f"{prefix}_motion_full_suffix_support_incomplete")
        auxiliary_ticks = phase["auxiliary_support_ticks"]
        if (
            not auxiliary_ticks
            or auxiliary_ticks != sorted(set(auxiliary_ticks))
            or any(tick not in motion_ticks for tick in auxiliary_ticks)
        ):
            _append_reason(
                reasons, f"{prefix}_auxiliary_support_not_in_motion_suffix"
            )
        if motion_end < deadline:
            _append_reason(reasons, f"{prefix}_motion_suffix_ends_before_deadline")
        if motion_end >= deadline + RECOVERY_AGE_START:
            _append_reason(
                reasons, f"{prefix}_motion_suffix_overlaps_recovery_window"
            )
        if motion_end >= boundary:
            _append_reason(reasons, f"{prefix}_motion_suffix_overlaps_next_reveal")
        if phase["recovery_age_ticks"] != list(
            range(RECOVERY_AGE_START, RECOVERY_AGE_END + 1)
        ):
            _append_reason(reasons, f"{prefix}_recovery_age_support_drift")

        strike_source = int(phase["strike_source_tick"])
        strike_payment = int(phase["strike_payment_tick"])
        if not reveal <= strike_source <= deadline:
            _append_reason(reasons, f"{prefix}_strike_source_outside_shot")
        if strike_source != strike_payment:
            _append_reason(reasons, f"{prefix}_strike_not_same_transition")
        contact_source = int(phase["selected_contact_source_tick"])
        contact_payment = int(phase["selected_contact_payment_tick"])
        if contact_source != contact_payment:
            _append_reason(reasons, f"{prefix}_contact_not_same_transition")
        if not reveal <= contact_source <= deadline:
            _append_reason(reasons, f"{prefix}_contact_source_outside_shot")

        outcome_source = int(phase["outcome_source_tick"])
        on_table_payment = int(phase["on_table_payment_tick"])
        placement_payment = int(phase["placement_payment_tick"])
        if outcome_source < reveal:
            _append_reason(reasons, f"{prefix}_outcome_before_reveal")
        if outcome_source < strike_source:
            _append_reason(reasons, f"{prefix}_outcome_before_strike")
        if outcome_source < contact_source:
            _append_reason(reasons, f"{prefix}_outcome_before_contact")
        if outcome_source - reveal > horizon:
            _append_reason(reasons, f"{prefix}_flight_horizon_exceeded")
        if on_table_payment < outcome_source or placement_payment < outcome_source:
            _append_reason(reasons, f"{prefix}_outcome_paid_before_source")
        if (
            max(on_table_payment, placement_payment) - outcome_source
            > mailbox_horizon
        ):
            _append_reason(reasons, f"{prefix}_mailbox_horizon_exceeded")
        if on_table_payment != placement_payment:
            _append_reason(reasons, f"{prefix}_outcome_consumer_payment_tick_mismatch")
        flight_intervals.append((reveal, outcome_source))
        mailbox_intervals.append(
            (outcome_source, max(on_table_payment, placement_payment))
        )

        parity = shot["ac_common_parity"]
        for role in ("facts", "raw", "payment"):
            if parity[f"a_common_{role}_sha256"] != parity[
                f"c_common_{role}_sha256"
            ]:
                _append_reason(reasons, f"{prefix}_ac_common_{role}_mismatch")
        if parity["a_placement_raw_sha256"] != parity["c_placement_raw_sha256"]:
            _append_reason(reasons, f"{prefix}_ac_placement_raw_mismatch")
        if parity["a_placement_source_tick"] != parity["c_placement_source_tick"]:
            _append_reason(reasons, f"{prefix}_ac_placement_source_tick_mismatch")
        if parity["a_placement_payment_tick"] != parity[
            "c_placement_payment_tick"
        ]:
            _append_reason(reasons, f"{prefix}_ac_placement_payment_tick_mismatch")
        if parity["a_placement_source_tick"] != outcome_source:
            _append_reason(reasons, f"{prefix}_placement_source_phase_mismatch")
        if parity["a_placement_payment_tick"] != placement_payment:
            _append_reason(reasons, f"{prefix}_placement_payment_phase_mismatch")
        if _as_fraction(parity["a_treatment_gain"]) != Fraction(1, 1):
            _append_reason(reasons, f"{prefix}_a_treatment_gain_not_one")
        if _as_fraction(parity["c_treatment_gain"]) != Fraction(0, 1):
            _append_reason(reasons, f"{prefix}_c_treatment_gain_not_zero")

        task_ref = str(shot["task_ref_sha256"])
        outcome_key = str(shot["outcome_key_sha256"])
        if task_ref in task_refs:
            _append_reason(reasons, f"{prefix}_task_ref_reused")
        if outcome_key in outcome_keys:
            _append_reason(reasons, f"{prefix}_outcome_key_reused")
        task_refs.add(task_ref)
        outcome_keys.add(outcome_key)
        targets.append(str(shot["target_float32_sha256"]))

    for index in range(len(targets) - 1):
        if targets[index] == targets[index + 1]:
            _append_reason(reasons, f"shot_{index}_adjacent_target_not_distinct")

    max_flights = _max_inclusive_overlap(flight_intervals)
    max_mailboxes = _max_inclusive_overlap(mailbox_intervals)
    if tape["observed_max_open_flights"] != max_flights:
        _append_reason(reasons, "observed_max_open_flights_mismatch")
    if tape["observed_max_open_mailboxes"] != max_mailboxes:
        _append_reason(reasons, "observed_max_open_mailboxes_mismatch")
    if max_flights > tape["flight_capacity"]:
        _append_reason(reasons, "flight_capacity_k_insufficient")
    if max_mailboxes > tape["mailbox_capacity"]:
        _append_reason(reasons, "mailbox_capacity_k_insufficient")
    if cadence > 0:
        required_flight_capacity = horizon // cadence + 1
        required_mailbox_capacity = mailbox_horizon // cadence + 1
        if tape["flight_capacity"] < required_flight_capacity:
            _append_reason(
                reasons, "flight_capacity_k_below_horizon_worst_case"
            )
        if tape["mailbox_capacity"] < required_mailbox_capacity:
            _append_reason(
                reasons, "mailbox_capacity_k_below_horizon_worst_case"
            )
    minimum_tail_closure = (
        int(shots[-1]["reveal_tick"]) + horizon + mailbox_horizon
    )
    if tape["tail_closure_tick"] < minimum_tail_closure:
        _append_reason(reasons, "tail_closure_does_not_cover_q3_horizons")
    if any(
        max(interval) > tape["tail_closure_tick"]
        for interval in (*flight_intervals, *mailbox_intervals)
    ):
        _append_reason(reasons, "observed_interval_exceeds_tail_closure")
    return tuple(reasons)


def _validate_unit_matrix_shape(
    shot: Mapping[str, object],
    candidate: _NumericCandidate,
    graph: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    rows = shot["candidate_unit_income"]
    by_id: dict[str, dict[str, object]] = {}
    for row_index, row_value in enumerate(rows):
        row = _exact_keys(
            row_value,
            _UNIT_INCOME_KEYS,
            label=f"candidate_unit_income[{row_index}]",
        )
        candidate_id = _require_text(
            row["candidate_id"], label="candidate_unit_income.candidate_id"
        )
        if candidate_id in by_id:
            raise FreshRewardBudgetError(
                "candidate_unit_income contains duplicate candidate ids"
            )
        _require_sha256(
            row["raw_profile_sha256"],
            label="candidate_unit_income.raw_profile_sha256",
        )
        by_id[candidate_id] = row
    if candidate.candidate_id not in by_id:
        raise FreshRewardBudgetError(
            f"candidate_unit_income missing {candidate.candidate_id!r}"
        )
    row = by_id[candidate.candidate_id]
    groups = graph["term_groups"]
    parsed = {
        "motion": _event_map(
            row["motion"], groups["motion"], label="unit_income.motion"
        ),
        "ten_reference": _event_map(
            row["ten_reference"],
            groups["ordered_ten"],
            label="unit_income.ten_reference",
        ),
        "ten_peak": _event_map(
            row["ten_peak"],
            groups["ordered_ten"],
            label="unit_income.ten_peak",
        ),
        "selected_contact_peak": _event_rows(
            row["selected_contact_peak"], label="unit_income.selected_contact_peak"
        ),
        "on_table_peak": _event_rows(
            row["on_table_peak"], label="unit_income.on_table_peak"
        ),
        "recovery": _event_map(
            row["recovery"], groups["recovery"], label="unit_income.recovery"
        ),
        "auxiliary_typical": _event_map(
            row["auxiliary_typical"],
            groups["auxiliary"],
            label="unit_income.auxiliary_typical",
        ),
        "auxiliary_p95": _event_map(
            row["auxiliary_p95"],
            groups["auxiliary"],
            label="unit_income.auxiliary_p95",
        ),
        "auxiliary_negative": _event_map(
            row["auxiliary_negative"],
            groups["auxiliary"],
            label="unit_income.auxiliary_negative",
        ),
        "placement_broad_reference": _event_rows(
            row["placement_broad_reference"],
            label="unit_income.placement_broad_reference",
        ),
        "placement_narrow_reference": _event_rows(
            row["placement_narrow_reference"],
            label="unit_income.placement_narrow_reference",
        ),
        "placement_peak": _event_rows(
            row["placement_peak"], label="unit_income.placement_peak"
        ),
    }
    return row, parsed


def _cross_candidate_unit_income_blockers(
    shot: Mapping[str, object],
    candidates: Sequence[_NumericCandidate],
) -> tuple[str, ...]:
    """Reject candidate-id-dependent raw evidence on one fixed tape."""

    rows: dict[str, Mapping[str, object]] = {
        str(row["candidate_id"]): row for row in shot["candidate_unit_income"]
    }
    reasons: list[str] = []
    common_reference: object | None = None
    strike_by_profile: dict[str, object] = {}
    placement_by_profile: dict[str, object] = {}
    full_by_profile: dict[str, object] = {}
    for candidate in candidates:
        row = rows[candidate.candidate_id]
        common = {key: row[key] for key in _COMMON_UNIT_INCOME_KEYS}
        strike = {
            key: row[key] for key in _STRIKE_PROFILE_UNIT_INCOME_KEYS
        }
        placement = {
            key: row[key] for key in _PLACEMENT_PROFILE_UNIT_INCOME_KEYS
        }
        full = {
            key: row[key]
            for key in _UNIT_INCOME_KEYS
            if key not in ("candidate_id", "raw_profile_sha256")
        }
        if common_reference is None:
            common_reference = common
        elif not _frozen_exact_equal(common, common_reference):
            _append_reason(
                reasons, "common_unit_income_differs_across_candidates"
            )
        for digest, payload, seen, label in (
            (
                candidate.strike_numeric_profile_sha256,
                strike,
                strike_by_profile,
                "strike_numeric_profile_unit_income_differs",
            ),
            (
                candidate.placement_numeric_profile_sha256,
                placement,
                placement_by_profile,
                "placement_numeric_profile_unit_income_differs",
            ),
            (
                candidate.raw_profile_sha256,
                full,
                full_by_profile,
                "identical_raw_profile_unit_income_differs",
            ),
        ):
            if digest in seen and not _frozen_exact_equal(payload, seen[digest]):
                _append_reason(reasons, label)
            else:
                seen[digest] = payload
    return tuple(reasons)


def _rows_have_ticks(
    rows: Sequence[tuple[int, int, Fraction]],
    ticks: Sequence[int],
) -> bool:
    return sorted((source, payment) for source, payment, _raw in rows) == [
        (tick, tick) for tick in ticks
    ]


def _raws_in_unit_interval(
    rows: Sequence[tuple[int, int, Fraction]],
) -> bool:
    return all(Fraction(0, 1) <= raw <= Fraction(1, 1) for _s, _p, raw in rows)


def _raws_in_positive_unit_interval(
    rows: Sequence[tuple[int, int, Fraction]],
) -> bool:
    return all(Fraction(0, 1) < raw <= Fraction(1, 1) for _s, _p, raw in rows)


def _discounted_rows(
    rows: Sequence[tuple[int, int, Fraction]],
    *,
    weight: Fraction,
    reveal_tick: int,
) -> Fraction:
    dt = Fraction(POLICY_DT["numerator"], POLICY_DT["denominator"])
    gamma = Fraction(PPO_GAMMA["numerator"], PPO_GAMMA["denominator"])
    result = Fraction(0, 1)
    for _source_tick, payment_tick, raw in rows:
        if payment_tick < reveal_tick:
            raise FreshRewardBudgetError("unit-income payment precedes shot reveal")
        result += gamma ** (payment_tick - reveal_tick) * raw * weight * dt
    return result


def _discounted_map(
    rows: Mapping[str, Sequence[tuple[int, int, Fraction]]],
    weights: Mapping[str, Fraction],
    *,
    reveal_tick: int,
) -> Fraction:
    return sum(
        (
            _discounted_rows(
                rows[name], weight=weights[name], reveal_tick=reveal_tick
            )
            for name in weights
        ),
        Fraction(0, 1),
    )


def _evaluate_candidate_shot(
    shot: Mapping[str, object],
    candidate: _NumericCandidate,
    graph: Mapping[str, object],
) -> dict[str, object]:
    reasons: list[str] = []
    row, unit = _validate_unit_matrix_shape(shot, candidate, graph)
    shot_index = int(shot["shot_index"])
    phase = shot["phase_support"]
    reveal = int(shot["reveal_tick"])
    deadline = int(shot["deadline_tick"])
    boundary = int(shot["next_boundary_tick"])

    if row["raw_profile_sha256"] != candidate.raw_profile_sha256:
        _append_reason(reasons, "raw_profile_binding_mismatch")

    motion_ticks = tuple(int(tick) for tick in phase["motion_full_suffix_ticks"])
    for name, rows in unit["motion"].items():
        if not _rows_have_ticks(rows, motion_ticks):
            _append_reason(reasons, f"motion_full_suffix_rows_drift:{name}")
        if any(raw < 0 for _source, _payment, raw in rows):
            _append_reason(reasons, f"motion_raw_negative:{name}")

    strike_ticks = (int(phase["strike_source_tick"]),)
    for name in graph["term_groups"]["ordered_ten"]:
        reference_rows = unit["ten_reference"][name]
        peak_rows = unit["ten_peak"][name]
        if not _rows_have_ticks(reference_rows, strike_ticks):
            _append_reason(reasons, f"ten_reference_tick_drift:{name}")
        if not _rows_have_ticks(peak_rows, strike_ticks):
            _append_reason(reasons, f"ten_peak_tick_drift:{name}")
        if not _raws_in_unit_interval(reference_rows):
            _append_reason(reasons, f"ten_reference_raw_out_of_range:{name}")
        if (
            NUMERIC_STRIKE_KERNEL_KINDS[name] == "cauchy_v1"
            and not _raws_in_positive_unit_interval(reference_rows)
        ):
            _append_reason(
                reasons, f"ten_reference_cauchy_tail_must_be_positive:{name}"
            )
        if not _raws_in_unit_interval(peak_rows):
            _append_reason(reasons, f"ten_peak_raw_out_of_range:{name}")
        if len(peak_rows) != 1 or peak_rows[0][2] != Fraction(1, 1):
            _append_reason(reasons, f"ten_peak_must_equal_one:{name}")
        if (
            len(reference_rows) == 1
            and len(peak_rows) == 1
            and reference_rows[0][2] > peak_rows[0][2]
        ):
            _append_reason(reasons, f"ten_reference_exceeds_peak:{name}")

    contact_rows = unit["selected_contact_peak"]
    if not _rows_have_ticks(
        contact_rows, (int(phase["selected_contact_source_tick"]),)
    ):
        _append_reason(reasons, "selected_contact_tick_drift")
    if len(contact_rows) != 1 or contact_rows[0][2] != Fraction(1, 1):
        _append_reason(reasons, "selected_contact_peak_must_equal_one")

    on_table_rows = unit["on_table_peak"]
    if len(on_table_rows) != 1:
        _append_reason(reasons, "on_table_peak_must_have_one_row")
    else:
        if (
            on_table_rows[0][0] != phase["outcome_source_tick"]
            or on_table_rows[0][1] != phase["on_table_payment_tick"]
        ):
            _append_reason(reasons, "on_table_source_payment_tick_drift")
        if on_table_rows[0][2] != Fraction(1, 1):
            _append_reason(reasons, "on_table_peak_must_equal_one")

    placement_broad_rows = unit["placement_broad_reference"]
    placement_narrow_rows = unit["placement_narrow_reference"]
    placement_rows = unit["placement_peak"]
    for rows, label in (
        (placement_broad_rows, "placement_broad_reference"),
        (placement_narrow_rows, "placement_narrow_reference"),
        (placement_rows, "placement_peak"),
    ):
        if len(rows) != 1:
            _append_reason(reasons, f"{label}_must_have_one_row")
        elif (
            rows[0][0] != phase["outcome_source_tick"]
            or rows[0][1] != phase["placement_payment_tick"]
        ):
            _append_reason(reasons, f"{label}_source_payment_tick_drift")
    if (
        len(placement_broad_rows) == 1
        and len(placement_narrow_rows) == 1
        and len(placement_rows) == 1
    ):
        broad_raw = placement_broad_rows[0][2]
        narrow_raw = placement_narrow_rows[0][2]
        peak_raw = placement_rows[0][2]
        if not Fraction(0, 1) < broad_raw < narrow_raw < peak_raw:
            _append_reason(
                reasons, "placement_reference_raws_must_strictly_increase"
            )
        if peak_raw != Fraction(1, 1):
            _append_reason(reasons, "placement_peak_must_equal_one")

    recovery_ticks = tuple(
        deadline + age for age in range(RECOVERY_AGE_START, RECOVERY_AGE_END + 1)
    )
    for name, rows in unit["recovery"].items():
        if not _rows_have_ticks(rows, recovery_ticks):
            _append_reason(reasons, f"recovery_68_tick_rows_drift:{name}")
        if not _raws_in_positive_unit_interval(rows):
            _append_reason(reasons, f"recovery_raw_not_positive_unit:{name}")

    auxiliary_ticks = tuple(int(tick) for tick in phase["auxiliary_support_ticks"])
    for group_name in (
        "auxiliary_typical",
        "auxiliary_p95",
        "auxiliary_negative",
    ):
        for name, rows in unit[group_name].items():
            if not _rows_have_ticks(rows, auxiliary_ticks):
                _append_reason(
                    reasons, f"{group_name}_support_tick_drift:{name}"
                )
            if any(
                source < reveal or payment < source
                for source, payment, _raw in rows
            ):
                _append_reason(reasons, f"{group_name}_tick_order_drift:{name}")
    if any(
        payment < reveal
        for group in unit.values()
        for rows in (group.values() if isinstance(group, Mapping) else (group,))
        for _source, payment, _raw in rows
    ):
        _append_reason(reasons, "payment_before_reveal")

    if reasons:
        return {
            "shot_index": shot_index,
            "feasible": False,
            "elimination_reasons": reasons,
            "budgets": None,
            "hierarchy": None,
        }

    motion = _discounted_map(
        unit["motion"], candidate.motion_weights, reveal_tick=reveal
    )
    ten_reference = _discounted_map(
        unit["ten_reference"], candidate.ten_weights, reveal_tick=reveal
    )
    ten_peak = _discounted_map(
        unit["ten_peak"], candidate.ten_weights, reveal_tick=reveal
    )
    selected_contact = _discounted_rows(
        contact_rows,
        weight=candidate.selected_contact_weight,
        reveal_tick=reveal,
    )
    on_table = _discounted_rows(
        on_table_rows, weight=candidate.on_table_weight, reveal_tick=reveal
    )
    recovery = _discounted_map(
        unit["recovery"], candidate.recovery_weights, reveal_tick=reveal
    )
    auxiliary_typical = _discounted_map(
        unit["auxiliary_typical"],
        candidate.auxiliary_weights,
        reveal_tick=reveal,
    )
    auxiliary_p95 = _discounted_map(
        unit["auxiliary_p95"],
        candidate.auxiliary_weights,
        reveal_tick=reveal,
    )
    auxiliary_negative = _discounted_map(
        unit["auxiliary_negative"],
        candidate.auxiliary_weights,
        reveal_tick=reveal,
    )
    placement_broad = _discounted_rows(
        placement_broad_rows,
        weight=candidate.placement_weight,
        reveal_tick=reveal,
    )
    placement_narrow = _discounted_rows(
        placement_narrow_rows,
        weight=candidate.placement_weight,
        reveal_tick=reveal,
    )
    placement = _discounted_rows(
        placement_rows, weight=candidate.placement_weight, reveal_tick=reveal
    )
    motion_gap = ten_reference - motion
    table_gap = on_table - (ten_peak + selected_contact)
    auxiliary_guard = recovery + max(
        abs(auxiliary_typical),
        abs(auxiliary_p95),
        abs(auxiliary_negative),
    )
    placement_narrow_increment = placement_narrow - placement_broad
    placement_precision_increment = placement - placement_narrow
    minimum_main_gap = min(motion_gap, table_gap)
    if motion_gap <= 0:
        _append_reason(reasons, "motion_not_below_ten_reference")
    if table_gap <= 0:
        _append_reason(reasons, "ten_peak_plus_contact_not_below_on_table")
    if auxiliary_guard >= minimum_main_gap:
        _append_reason(reasons, "recovery_auxiliary_inverts_main_hierarchy")
    if placement_broad <= auxiliary_guard:
        _append_reason(reasons, "placement_broad_not_above_auxiliary_guard")
    if placement_narrow_increment <= auxiliary_guard:
        _append_reason(
            reasons, "placement_narrow_increment_not_above_auxiliary_guard"
        )
    if placement_precision_increment <= auxiliary_guard:
        _append_reason(
            reasons, "placement_precision_increment_not_above_auxiliary_guard"
        )
    if placement >= on_table:
        _append_reason(reasons, "placement_peak_not_below_on_table")

    budgets = {
        "motion": _fraction_payload(motion),
        "ten_reference": _fraction_payload(ten_reference),
        "ten_peak": _fraction_payload(ten_peak),
        "selected_contact_peak": _fraction_payload(selected_contact),
        "on_table_peak": _fraction_payload(on_table),
        "recovery": _fraction_payload(recovery),
        "auxiliary_typical": _fraction_payload(auxiliary_typical),
        "auxiliary_p95": _fraction_payload(auxiliary_p95),
        "auxiliary_negative": _fraction_payload(auxiliary_negative),
        "placement_broad_reference": _fraction_payload(placement_broad),
        "placement_narrow_reference": _fraction_payload(placement_narrow),
        "placement_peak_excluded_from_common_hierarchy": _fraction_payload(placement),
        "placement_narrow_increment": _fraction_payload(
            placement_narrow_increment
        ),
        "placement_precision_increment": _fraction_payload(
            placement_precision_increment
        ),
        "motion_to_ten_reference_gap": _fraction_payload(motion_gap),
        "ten_contact_to_table_gap": _fraction_payload(table_gap),
        "recovery_auxiliary_guard": _fraction_payload(auxiliary_guard),
    }
    hierarchy = {
        "motion_below_ten_reference": motion_gap > 0,
        "ten_peak_plus_contact_below_on_table": table_gap > 0,
        "recovery_auxiliary_below_both_main_gaps": (
            auxiliary_guard < minimum_main_gap
        ),
        "placement_may_satisfy_common_hierarchy": False,
        "placement_broad_above_auxiliary_guard": (
            placement_broad > auxiliary_guard
        ),
        "placement_narrow_increment_above_auxiliary_guard": (
            placement_narrow_increment > auxiliary_guard
        ),
        "placement_precision_increment_above_auxiliary_guard": (
            placement_precision_increment > auxiliary_guard
        ),
        "placement_peak_below_on_table": placement < on_table,
        "discount_origin_tick": reveal,
        "true_payment_ticks_used": True,
        "dt_applications_per_event": 1,
    }
    return {
        "shot_index": shot_index,
        "feasible": not reasons,
        "elimination_reasons": reasons,
        "budgets": budgets,
        "hierarchy": hierarchy,
    }


def _evaluate_candidate(
    candidate: _NumericCandidate,
    graph: Mapping[str, object],
    tape: Mapping[str, object],
    *,
    global_blockers: Sequence[str],
) -> dict[str, object]:
    shot_results = [
        _evaluate_candidate_shot(shot, candidate, graph)
        for shot in tape["shots"]
    ]
    reasons = list(global_blockers)
    for reason in candidate.admissibility_reasons:
        _append_reason(reasons, reason)
    for result in shot_results:
        for reason in result["elimination_reasons"]:
            _append_reason(
                reasons, f"shot_{result['shot_index']}:{reason}"
            )
    return {
        "candidate_id": candidate.candidate_id,
        "lexicographic_priority": list(candidate.priority),
        "feasible": not reasons,
        "elimination_reasons": reasons,
        "per_shot": shot_results,
    }


def materialize_numeric_authority(
    resolved_graph_receipt: object,
    four_shot_tape_receipt: object,
    candidate_set_receipt: object,
    *,
    trusted_input_roots: object,
) -> dict[str, object]:
    """Evaluate explicit numeric candidates without inventing any parameter.

    A structurally valid but scientifically failing evidence set returns a
    content-addressed BLOCKED materialization.  Malformed, unsealed, or
    cross-bound inputs raise :class:`FreshRewardBudgetError`.
    """

    graph = validate_numeric_resolved_graph_receipt(resolved_graph_receipt)
    tape = validate_numeric_four_shot_tape_receipt(four_shot_tape_receipt)
    candidate_receipt, candidates = _parse_numeric_candidates(
        candidate_set_receipt, graph=graph, tape=tape
    )
    trusted_roots = _validate_trusted_input_roots(
        trusted_input_roots,
        graph=graph,
        tape=tape,
        candidate_set=candidate_receipt,
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    expected_candidate_ids = frozenset(candidate_ids)
    for shot_index, shot in enumerate(tape["shots"]):
        actual_ids: list[str] = []
        for row in shot["candidate_unit_income"]:
            row_mapping = _exact_keys(
                row,
                _UNIT_INCOME_KEYS,
                label=f"shots[{shot_index}].candidate_unit_income[]",
            )
            actual_ids.append(
                _require_text(
                    row_mapping["candidate_id"],
                    label=f"shots[{shot_index}].candidate_unit_income.candidate_id",
                )
            )
        if (
            len(actual_ids) != len(set(actual_ids))
            or frozenset(actual_ids) != expected_candidate_ids
        ):
            raise FreshRewardBudgetError(
                f"shots[{shot_index}] candidate unit-income coverage differs"
            )

    global_blockers = list(_global_numeric_evidence_blockers(graph, tape))
    for shot_index, shot in enumerate(tape["shots"]):
        for reason in _cross_candidate_unit_income_blockers(shot, candidates):
            _append_reason(
                global_blockers, f"shot_{shot_index}_{reason}"
            )
    evaluations = [
        _evaluate_candidate(
            candidate,
            graph,
            tape,
            global_blockers=global_blockers,
        )
        for candidate in candidates
    ]
    feasible = [
        (candidate, evaluation)
        for candidate, evaluation in zip(candidates, evaluations)
        if evaluation["feasible"] is True
    ]
    selection: dict[str, object] | None = None
    selected_candidate: _NumericCandidate | None = None
    selected_evaluation: dict[str, object] | None = None
    blocked_reasons = list(global_blockers)
    if not feasible:
        _append_reason(blocked_reasons, "no_feasible_candidate")
    else:
        best_priority = min(candidate.priority for candidate, _evaluation in feasible)
        winners = [
            (candidate, evaluation)
            for candidate, evaluation in feasible
            if candidate.priority == best_priority
        ]
        if len(winners) != 1:
            _append_reason(blocked_reasons, "non_unique_lexicographic_winner")
        else:
            selected_candidate, selected_evaluation = winners[0]
            selection = {
                "semantics": LEXICOGRAPHIC_SELECTION_SEMANTICS,
                "candidate_id": selected_candidate.candidate_id,
                "lexicographic_priority": list(selected_candidate.priority),
                "feasible_candidate_count": len(feasible),
            }

    input_receipts = {
        "resolved_graph_receipt_sha256": graph["receipt_sha256"],
        "four_shot_tape_receipt_sha256": tape["receipt_sha256"],
        "candidate_set_receipt_sha256": candidate_receipt["receipt_sha256"],
        "source_closure_sha256": graph["source_closure_sha256"],
        "trusted_input_roots": trusted_roots,
    }
    authority_payload: dict[str, object] | None = None
    authority_sha256: str | None = None
    selected_parameters: Mapping[str, object] | None = None
    if selected_candidate is not None and selected_evaluation is not None:
        selected_parameters = selected_candidate.normalized_parameters
        cadence = (
            int(tape["shots"][1]["reveal_tick"])
            - int(tape["shots"][0]["reveal_tick"])
        )
        authority_payload = {
            "schema_version": NUMERIC_SCHEMA_VERSION,
            "kind": NUMERIC_AUTHORITY_KIND,
        "scope": NUMERIC_AUTHORITY_SCOPE,
        "diagnostic_unauthorized": NUMERIC_DIAGNOSTIC_UNAUTHORIZED,
        "runtime_integrated": NUMERIC_RUNTIME_INTEGRATED,
        "launch_authorized": NUMERIC_LAUNCH_AUTHORIZED,
        "input_receipts": input_receipts,
            "policy_clock": graph["policy_clock"],
            "discounted_budget_formula": (
                "sum_events((99/100)^(payment_tick-reveal_tick) * raw * "
                "manager_weight * (1/50))"
            ),
            "selected_candidate_id": selected_candidate.candidate_id,
            "lexicographic_priority": list(selected_candidate.priority),
            "selected_numeric_parameters": selected_parameters,
            "per_shot_budget_evidence": selected_evaluation["per_shot"],
            "capacity_evidence": {
                "cadence_ticks": cadence,
                "flight_horizon_ticks": tape["flight_horizon_ticks"],
                "flight_horizon_witness_sha256": tape[
                    "flight_horizon_witness_sha256"
                ],
                "required_worst_case_flight_capacity": (
                    int(tape["flight_horizon_ticks"]) // cadence + 1
                ),
                "flight_capacity": tape["flight_capacity"],
                "mailbox_horizon_ticks": tape["mailbox_horizon_ticks"],
                "mailbox_horizon_witness_sha256": tape[
                    "mailbox_horizon_witness_sha256"
                ],
                "required_worst_case_mailbox_capacity": (
                    int(tape["mailbox_horizon_ticks"]) // cadence + 1
                ),
                "mailbox_capacity": tape["mailbox_capacity"],
                "tail_closure_tick": tape["tail_closure_tick"],
            },
            "placement_may_satisfy_common_hierarchy": False,
        }
        authority_sha256 = canonical_sha256(authority_payload)

    ready = authority_sha256 is not None
    payload: dict[str, object] = {
        "schema_version": NUMERIC_SCHEMA_VERSION,
        "kind": NUMERIC_MATERIALIZATION_KIND,
        "status": READY_STATUS if ready else NUMERIC_BLOCKED_STATUS,
        "budget_authority_ready": ready,
        "launch_authorized": False,
        "authority_scope": NUMERIC_AUTHORITY_SCOPE,
        "authority_sha256": authority_sha256,
        "authority_payload": authority_payload,
        "input_receipts": input_receipts,
        "policy_clock": graph["policy_clock"],
        "global_evidence_blockers": list(global_blockers),
        "candidate_evaluations": evaluations,
        "selection": selection,
        "selected_numeric_parameters": selected_parameters,
    }
    if ready:
        payload["global_evidence_blockers"] = []
    else:
        payload["global_evidence_blockers"] = blocked_reasons
    result = dict(payload)
    result["materialization_sha256"] = canonical_sha256(payload)
    return result


def validate_numeric_materialization_receipt(
    value: object,
    *,
    resolved_graph_receipt: object,
    four_shot_tape_receipt: object,
    candidate_set_receipt: object,
    trusted_input_roots: object,
) -> dict[str, object]:
    """Recompute the output from externally supplied sealed evidence.

    A materialization is not self-authorizing: a caller must provide all three
    original inputs plus launcher-owned expected roots.  Merely editing and
    re-sealing a BLOCKED output therefore cannot manufacture READY authority.
    """

    _require_plain_json(value, label="numeric materialization")
    receipt = _exact_keys(
        value, _MATERIALIZATION_KEYS, label="numeric materialization"
    )
    if receipt["schema_version"] != NUMERIC_SCHEMA_VERSION or type(
        receipt["schema_version"]
    ) is not int:
        raise FreshRewardBudgetError(
            "numeric materialization schema_version must equal 2"
        )
    if receipt["kind"] != NUMERIC_MATERIALIZATION_KIND:
        raise FreshRewardBudgetError("numeric materialization kind differs")
    declared = _require_sha256(
        receipt["materialization_sha256"],
        label="numeric materialization.materialization_sha256",
    )
    payload = {
        key: item
        for key, item in receipt.items()
        if key != "materialization_sha256"
    }
    if canonical_sha256(payload) != declared:
        raise FreshRewardBudgetError("numeric materialization SHA mismatch")
    if receipt["launch_authorized"] is not False:
        raise FreshRewardBudgetError(
            "numeric budget authority cannot authorize a training launch"
        )
    if receipt["authority_scope"] != NUMERIC_AUTHORITY_SCOPE:
        raise FreshRewardBudgetError("numeric authority scope differs")

    ready = receipt["status"] == READY_STATUS
    if ready:
        if receipt["budget_authority_ready"] is not True:
            raise FreshRewardBudgetError("READY budget_authority_ready must be true")
        authority = receipt["authority_payload"]
        if type(authority) is not dict:
            raise FreshRewardBudgetError("READY authority payload must be present")
        authority_sha256 = _require_sha256(
            receipt["authority_sha256"], label="authority_sha256"
        )
        if canonical_sha256(authority) != authority_sha256:
            raise FreshRewardBudgetError("numeric authority SHA mismatch")
        if receipt["selection"] is None or receipt["selected_numeric_parameters"] is None:
            raise FreshRewardBudgetError("READY selection and numeric parameters required")
        if receipt["global_evidence_blockers"] != []:
            raise FreshRewardBudgetError("READY receipt cannot retain blockers")
    else:
        if receipt["status"] != NUMERIC_BLOCKED_STATUS:
            raise FreshRewardBudgetError("numeric materialization status differs")
        if receipt["budget_authority_ready"] is not False:
            raise FreshRewardBudgetError("BLOCKED budget_authority_ready must be false")
        for field in (
            "authority_sha256",
            "authority_payload",
            "selection",
            "selected_numeric_parameters",
        ):
            if receipt[field] is not None:
                raise FreshRewardBudgetError(f"BLOCKED {field} must be null")
        if type(receipt["global_evidence_blockers"]) is not list or not receipt[
            "global_evidence_blockers"
        ]:
            raise FreshRewardBudgetError("BLOCKED receipt must name blockers")
    expected = materialize_numeric_authority(
        resolved_graph_receipt,
        four_shot_tape_receipt,
        candidate_set_receipt,
        trusted_input_roots=trusted_input_roots,
    )
    if not _frozen_exact_equal(receipt, expected):
        raise FreshRewardBudgetError(
            "numeric materialization differs from external evidence recomputation"
        )
    return receipt


__all__ = [
    "BLOCKED_STATUS",
    "C04_SOURCE_SHA256",
    "CONTACT_TERM_COUNT",
    "DT_APPLICATION_SEMANTICS",
    "FreshRewardBudgetAudit",
    "FreshRewardBudgetError",
    "FreshRewardNumericProductionHold",
    "ACTION_TASK_RECEIPT_REF_FIELDS",
    "KIND",
    "LEGACY_LANDING_MANAGER_WEIGHT",
    "LEGACY_PROXIMITY_MANAGER_WEIGHT",
    "LEGACY_WINDOW_MANAGER_WEIGHTS",
    "LANDING_OUTCOME_SHOT_KEY_FIELDS",
    "LANDING_OUTCOME_SUCCESSOR_FIELDS",
    "CONSTRUCTED_EVIDENCE_SCOPE",
    "EXPLICIT_CANDIDATE_SCOPE",
    "LEXICOGRAPHIC_SELECTION_SEMANTICS",
    "MIN_DEADLINE_TO_NEXT_REVEAL_TICKS",
    "NUMERIC_AUTHORITY_KIND",
    "NUMERIC_AUTHORITY_SCOPE",
    "NUMERIC_DIAGNOSTIC_UNAUTHORIZED",
    "NUMERIC_LAUNCH_AUTHORIZED",
    "NUMERIC_PRODUCTION_HOLD_REASONS",
    "NUMERIC_RUNTIME_INTEGRATED",
    "NUMERIC_BLOCKED_STATUS",
    "NUMERIC_CANDIDATE_SET_KIND",
    "NUMERIC_GRAPH_KIND",
    "NUMERIC_MATERIALIZATION_KIND",
    "NUMERIC_SCHEMA_VERSION",
    "NUMERIC_STRIKE_KERNEL_KINDS",
    "NUMERIC_TAPE_KIND",
    "NumericRewardAuthorityOwner",
    "NumericRewardAuthorityReceipt",
    "ORDERED_CONTACT_TERMS",
    "ORDERED_CONTACT_TERM_NAMES",
    "POLICY_DT",
    "PPO_GAMMA",
    "PROHIBITED_FIXTURE_PLACEMENT_PROFILE",
    "READY_STATUS",
    "RECOVERY_AGE_COUNT",
    "RECOVERY_AGE_END",
    "RECOVERY_AGE_START",
    "SCHEMA_VERSION",
    "TRUSTED_INPUT_ROOT_KIND",
    "UNRESOLVED_REASON_CODES",
    "audit_receipt",
    "build_blocked_receipt",
    "canonical_sha256",
    "materialize_authority_sha256",
    "materialize_numeric_authority",
    "make_diagnostic_numeric_reward_authority_owner",
    "construct_production_numeric_reward_authority_owner",
    "validate_blocked_receipt",
    "validate_numeric_candidate_set_receipt",
    "validate_numeric_four_shot_tape_receipt",
    "validate_numeric_materialization_receipt",
    "validate_numeric_resolved_graph_receipt",
]
