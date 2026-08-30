"""Pure-device delayed ActionBall landing-outcome authority.

This module is the family-neutral R06 first cell.  It deliberately separates
the physical-flight grid ``[N, Kf]`` from the durable reward-mailbox grid
``[N, Km]``.  A settled physical slot may therefore be retired and reused
while its immutable settlement remains unpaid in the mailbox.

All scientific configuration and both capacities are explicit construction
inputs.  The fixed-cadence mutation methods perform no device-to-host
materialization.  A PPO boundary uses one packed transfer; the separate cold
checkpoint byte-root helper also uses one packed transfer and requires the
exact, unconsumed, mutation-current boundary receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import hmac
import inspect
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Mapping, Sequence
import weakref

import torch

import action_ball_ac_family_contract as _c10
import action_ball_continuous_runtime_transaction as _r05
import action_ball_continuous_runtime_transaction_device as _r05_device
import action_ball_fresh_reward_budget as _r03
import action_ball_full_mdp_diagnostic_capacity as _diagnostic_capacity
import action_ball_full_mdp_reveal_boundary as _reveal_boundary
import action_ball_full_mdp_row_identity as _row_identity
import action_ball_landing_outcome_mailbox as _c05
from action_ball_landing_placement import LandingPlacementProfile
from action_ball_landing_placement import LandingPlacementTaskIdentity
from action_ball_landing_placement import SELECTED_RUBBER_CONTACT_AUTHORITY
if __package__:
    from .action_ball_landing_placement_torch import (
        REASON_TO_CODE,
        score_landing_placement_torch,
    )
else:
    # A few dependency-focused diagnostics execute these exact bytes with a
    # file loader.  They may resolve only the canonical package module; a
    # top-level same-name alias is deliberately not a fallback.
    from whole_body_tracking.tasks.tracking.mdp.action_ball_landing_placement_torch import (
        REASON_TO_CODE,
        score_landing_placement_torch,
    )


SCHEMA_VERSION = 9
SCALE_TARGET_NUM_ENVS = 4096
TOKEN_BYTES = 32
R06_FULL_MDP_REWARD_TOP_BIND_API_FROZEN = True
# The current top/graph sources do not yet call R06's construction bind/open
# seam.  Keep this false until those two production callpoints are reviewed
# and focused together; a leaf-local success fixture is not runtime evidence.
R06_FULL_MDP_REWARD_RUNTIME_CONNECTED = False

# The disposable constructor below deliberately does not mint the formal R03
# capacity or C10 payment authorities.  Its historical ``N2`` names are
# temporary naming debt: runtime environment cardinality is derived from the
# exact live Env/Physical/ActionEpoch object join.  The two capacities below
# are K dimensions (physical flight slots and durable mailboxes), not N.
# These identifiers label a narrow same-process construction mode; they are
# not numeric evidence, launch permission, or portable receipt roots.
DIAGNOSTIC_N2_NO_SAVE_CONSTRUCTION_KIND = (
    "action_ball_r06_diagnostic_n2_no_save_construction_v1"
)
# Retained only as historical external ABI during the later neutral rename.
# R06 itself must not consume it for allocation, profile, or live validation.
DIAGNOSTIC_N2_NO_SAVE_NUM_ENVS = 2
DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY = 2
DIAGNOSTIC_N2_NO_SAVE_MAILBOX_CAPACITY = 2
DIAGNOSTIC_N2_NO_SAVE_RUN_IDS = {
    "A": "action-ball-full-mdp-a-n2-no-save",
    "C": "action-ball-full-mdp-c-n2-no-save",
}
DIAGNOSTIC_N2_NO_SAVE_CARRY_CHAIN_IDS = {
    "A": "action-ball-full-mdp-a-diagnostic-cold-chain-0",
    "C": "action-ball-full-mdp-c-diagnostic-cold-chain-0",
}
DIAGNOSTIC_N2_NO_SAVE = True
DIAGNOSTIC_UNAUTHORIZED = True
DIAGNOSTIC_CHECKPOINT_AUTHORIZED = False
DIAGNOSTIC_EXPORT_AUTHORIZED = False
DIAGNOSTIC_R10_AUTHORIZED = False
# A diagnostic row has no formal C10 projection.  Keep the ABI-width field
# explicitly empty instead of placing a diagnostic family/gain identity in a
# field whose formal name can be mistaken for C10 authority.
DIAGNOSTIC_N2_NO_FORMAL_C10_PROJECTION_SHA256 = "0" * 64

_HEX = frozenset("0123456789abcdef")
_INSTALL_AUTH_TOKEN = object()
_INSTALL_PACK_TOKEN = object()
_C10_PAYMENT_AUTH_TOKEN = object()
_CAPACITY_AUTH_TOKEN = object()
_PREPARED_INSTALL_AUTH_TOKEN = object()
_PHYSICAL_RETIRE_AUTH_TOKEN = object()
_PHYSICAL_PARK_AUTHORITY_MINT_TOKEN = object()
_PHYSICAL_RETIRE_CLEANUP_CAP_TOKEN = object()
_SELECTED_RESET_PHYSICAL_PARK_AUTHORITY_MINT_TOKEN = object()
_REVEAL_TERMINAL_AUTH_TOKEN = object()
_POSTPHYSICS_CONTACT_AUTH_TOKEN = object()
_ACTION_EPOCH_R06_CURRENT_SETTLEMENT_DELTA_TOKEN = object()
_R06_GLOBAL_DRAIN_OWNER_KIND = "r06_landing_outcome"
_R06_LEGACY_DRAIN_PROTOCOL = "legacy_r06_ppo_boundary_v1"
_R06_GLOBAL_DRAIN_PROTOCOL = "global_pre_optimizer_ppo_boundary_v1"
_R06_GLOBAL_DRAIN_SOURCE_SHA256 = (
    "674d4d1ab6c7f1ac7f8b6a0c32e25003d2c5ee784921dedb5e43cb29fc35122e"
)
_R06_GLOBAL_DRAIN_ACK_AUTHORITY_API_SHA256 = (
    "f759474e1576a151b37939d128b0ae2c58b02f4cf90007353b41fadad03d902d"
)
_R06_RETIRE_SUMMARY_MODULUS = 2147483647
_INSTALL_AUTH_KEY = hashlib.sha256(
    b"action_ball_landing_outcome_device_install_authority_v1\0"
    + Path(__file__).read_bytes()
).digest()
_C10_PAYMENT_AUTH_KEY = hashlib.sha256(
    b"action_ball_landing_outcome_device_c10_payment_authority_v1\0"
    + Path(__file__).read_bytes()
).digest()
_CAPACITY_AUTH_KEY = hashlib.sha256(
    b"action_ball_landing_outcome_capacity_authority_v1\0"
    + Path(__file__).read_bytes()
).digest()
_PREPARED_INSTALL_AUTH_KEY = hashlib.sha256(
    b"action_ball_landing_outcome_prepared_install_v1\0"
    + Path(__file__).read_bytes()
).digest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256 = hashlib.sha256(
    Path(_r05.__file__).read_bytes()
).hexdigest()
R05_RUNTIME_TRANSACTION_SOURCE_SHA256: str | None = (
    "82d71c987e51cf4b5940744b124b36f9055d7a655af5e2279e2a6841cb1077dc"
)
R05_FINAL_SOURCE_PIN_PENDING = False
if (
    R05_RUNTIME_TRANSACTION_SOURCE_SHA256 is not None
    and R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256
    != R05_RUNTIME_TRANSACTION_SOURCE_SHA256
):
    raise RuntimeError("R06 pinned R05 runtime-transaction source SHA drifted")
R05_RUNTIME_TRANSACTION_CONTRACT_MAPPING = {
    "schema_version": 2,
    "kind": "action_ball_continuous_runtime_transaction_r05_ingress_contract_v2",
    "r05_schema_version": _r05.SCHEMA_VERSION,
    "r05_final_source_pin_pending": R05_FINAL_SOURCE_PIN_PENDING,
    "integration_status": _r05.INTEGRATION_STATUS,
    "runtime_wiring_connected": _r05.RUNTIME_WIRING_CONNECTED,
    "committed_reveal_kind": _r05.CommittedReveal.KIND,
    "prepared_reveal_kind": _r05.PreparedReveal.KIND,
    "prepare_request_kind": _r05.ContinuousPrepareRequest.KIND,
    "prepared_ball_slot_reservation_kind": _r05.PreparedBallSlotReservation.KIND,
    "ball_slot_plan_kind": _r05.BallSlotPlan.KIND,
    "prepared_reveal_batch_kind": _r05.PreparedRevealBatch.KIND,
    "committed_reveal_batch_kind": _r05.CommittedRevealBatch.KIND,
    "reveal_final_install_row_kind": _r05.RevealFinalInstallRow.KIND,
    "reveal_final_preview_batch_kind": _r05.RevealFinalPreviewBatch.KIND,
    "terminal_boundary_authority_kind": (
        _r05.TERMINAL_BOUNDARY_AUTHORITY_KIND
    ),
    "terminal_boundary_participant_root_kind": (
        _r05.TerminalBoundaryParticipantRoot.KIND
    ),
    "terminal_boundary_censor_evidence_kind": (
        _r05.TerminalBoundaryCensorEvidence.KIND
    ),
    "terminal_boundary_projection_kind": _r05.TerminalBoundaryProjection.KIND,
    "terminal_boundary_decision_mapping_schema_version": (
        _r05.TERMINAL_BOUNDARY_DECISION_MAPPING_SCHEMA_VERSION
    ),
    "terminal_boundary_source_decision_pass": (
        _r05.TERMINAL_BOUNDARY_SOURCE_DECISION_PASS
    ),
    "infrastructure_censor_fact_kind": _r05.InfrastructureCensorFact.KIND,
    "prepared_terminal_content_pin_kind": (
        _r05.PreparedTerminalContentPin.KIND
    ),
    "prepared_reveal_terminal_claim_kind": (
        _r05.PREPARED_REVEAL_TERMINAL_CLAIM_KIND
    ),
    "terminal_decision_accept": _r05.TERMINAL_DECISION_ACCEPT,
    "terminal_decision_censor": _r05.TERMINAL_DECISION_CENSOR,
    "censored_reveal_batch_kind": _r05.CensoredRevealBatch.KIND,
    "censored_reveal_batch_schema_version": (
        _r05.CensoredRevealBatch.RECORD_SCHEMA_VERSION
    ),
}
R05_RUNTIME_TRANSACTION_CONTRACT_SHA256 = _canonical_sha256(
    R05_RUNTIME_TRANSACTION_CONTRACT_MAPPING
)
C05_LANDING_OUTCOME_SOURCE_SHA256 = (
    "cd2d32277c772c1cc1b2187d63034a27108a0d95a55af819da26d734b567b4ab"
)
if hashlib.sha256(Path(_c05.__file__).read_bytes()).hexdigest() != (
    C05_LANDING_OUTCOME_SOURCE_SHA256
):
    raise RuntimeError("R06 pinned C05 landing-outcome source SHA drifted")

FULL_MDP_REVEAL_BOUNDARY_OBSERVED_SOURCE_SHA256 = hashlib.sha256(
    Path(_reveal_boundary.__file__).read_bytes()
).hexdigest()
FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256: str | None = (
    "ec9db7ca2475bc8d4de474aeca9ce425feaeaef617fea7371f23d0ee5f8e25ab"
)
FULL_MDP_REVEAL_BOUNDARY_FINAL_SOURCE_PIN_PENDING = False
FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256 = (
    FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256
    or FULL_MDP_REVEAL_BOUNDARY_OBSERVED_SOURCE_SHA256
)
FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN = (
    "action_ball_full_mdp_reveal_boundary"
)
FULL_MDP_REVEAL_BOUNDARY_ROW_INTEGRITY_SHA256 = (
    "cfc212a4ef2fd2078df99114c28f55df93b0605e0a126049b24b07fc636b16aa"
)
FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256 = (
    "4e715720b741991905d7c6cf8aa5ddf6c5a1e617773b6132aa33368468736cdd"
)
if (
    FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256 is not None
    and FULL_MDP_REVEAL_BOUNDARY_OBSERVED_SOURCE_SHA256
    != FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256
) or (
    _reveal_boundary.PACKET_ROW_INTEGRITY_SCHEMA_SHA256
    != FULL_MDP_REVEAL_BOUNDARY_ROW_INTEGRITY_SHA256
) or (
    _reveal_boundary.RECEIPT_SCHEMA_SHA256
    != FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
):
    raise RuntimeError("R06 pinned all-owner reveal-boundary source SHA drifted")

TEXT_REGISTRY_SCHEMA_VERSION = 1
TEXT_REGISTRY_KIND = "action_ball_landing_outcome_text_registry_v1"
INSTALL_RECEIPT_KIND = "action_ball_landing_reveal_install_receipt_v2"
INSTALL_ROW_RECEIPT_KIND = "action_ball_landing_reveal_install_row_v2"
FULL_KEY_RECEIPT_KIND = "action_ball_landing_outcome_full_key_receipt_v2"
C10_PAYMENT_AUTHORITY_KIND = "action_ball_landing_outcome_c10_payment_authority_v1"
CAPACITY_AUTHORITY_KIND = "action_ball_landing_outcome_capacity_authority_v1"
CAPACITY_CLOCK_BINDING_KIND = (
    "action_ball_landing_outcome_capacity_control_step_clock_v1"
)
CAPACITY_INCLUSIVE_EVENT_ORDER_KIND = (
    "action_ball_landing_outcome_capacity_inclusive_event_order_v1"
)
PREPARED_INSTALL_RECEIPT_KIND = (
    "action_ball_landing_reveal_prepared_install_receipt_v1"
)
PREPARE_ATTEMPT_KIND = "action_ball_landing_reveal_prepare_attempt_v1"
PREPARED_COMMIT_RECEIPT_KIND = "action_ball_landing_reveal_commit_receipt_v1"
MAILBOX_ALLOCATION_POLICY = "r06_lowest_available_mailbox_slot_v1"
REVEAL_PREPARE_BOUNDARY_KIND = (
    "action_ball_landing_reveal_prepare_boundary_v1"
)
REVEAL_PREPARE_BOUNDARY_SCHEMA_VERSION = 1


def _sha256_hex(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise LandingOutcomeDeviceError(f"{label} must be lowercase SHA-256 hex")
    return value


def _exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise LandingOutcomeDeviceError(f"{label} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise LandingOutcomeDeviceError(f"{label} is outside its allowed range")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise LandingOutcomeDeviceError(f"{label} must be non-empty text")
    return value

RUNTIME_INTEGRATED = False
CUDA_PROFILED = False
FORMAL_EXACT_RESUME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
HOLD_REASONS = (
    "constructed runtime is not wired",
    "placement profile values are caller-owned and not adopted here",
    "flight and mailbox capacities are caller-owned and not adopted here",
    "manager weights and post-dt budget are caller-owned and not adopted here",
    "module-private tokens and source-derived HMAC keys are same-process consistency seals, not security capabilities",
    "the all-owner packet-v3 boundary exact source is pinned, but the constructed Motion/Racket/physical/R06/R05 five-phase orchestrator is not wired",
    "Racket, Motion, physical-ball, and R05 terminal authorities remain externally owned and may not be fabricated by R06",
    "the R05 owner-issued terminal failure-evidence ABI is pinned, but its real constructed coordinator consumer is not wired",
    "R06 hot reveal remains a production HOLD until Device-R05 publishes the complete full-key/task-SHA/action-UID identity and Physical publishes an independently validated late-launch authority",
    "R06 mints only its retained-token row and consumes only its owner-issued global receipt row",
    "R06 terminal publication blocks further mutation until the exact R05-last terminal receipt is acknowledged; the constructed coordinator remains external",
    "physical-first then R06-last retirement remains a constructed-runtime HOLD until the physical owner pins this final R06 source",
    "the physical scene owner has not yet consumed the typed lifecycle snapshot and post-physics settlement result in a real Isaac environment",
    "reveal-prepare N64 parity and N4096 profiler-off wall-time gates have not run",
    "R03 clock and inclusive event-order roots are bound and recomputed here; constructed-runtime witness adoption remains external",
    "CUDA-hidden structural tests do not authorize launch or CUDA scale",
)

# C05's canonical comparison tolerance.  This is a protocol constant, not a
# tunable scientific default.
C05_CROSSING_ABS_TOL_M = 1.0e-9

COMMON_ON_TABLE_CONSUMER = "common_on_table_outcome"
PLACEMENT_GUIDANCE_CONSUMER = "post_contact_placement_guidance"
CONSUMERS = (COMMON_ON_TABLE_CONSUMER, PLACEMENT_GUIDANCE_CONSUMER)
CONSUMER_COUNT = len(CONSUMERS)
FULL_CONSUMER_MASK = (1 << CONSUMER_COUNT) - 1

# Lean ActionEpoch R06 fact ABI.  These bits describe data, not authority: the
# only writer is cold-bound by ActionEpochOwner and the numeric payload comes
# from the already-retired live mailbox row.  Exact task/outcome/ball identity
# stays losslessly int64 in EpochIdentityPayload; it is never reconstructed
# from legacy shot_index, ball_generation, a receipt, or a digest.
R06_ACTION_EPOCH_PRESENT = 1 << 0
R06_ACTION_EPOCH_POLICY_ELIGIBLE = 1 << 1
R06_ACTION_EPOCH_SOURCE_VALID = 1 << 2
R06_ACTION_EPOCH_COMMON_ON_TABLE_F32 = 0
R06_ACTION_EPOCH_CANONICAL_TOTAL_F32 = 1
R06_ACTION_EPOCH_PLACEMENT_GAIN_F32 = 2
R06_ACTION_EPOCH_CROSSING_XY_F32 = slice(3, 5)
R06_ACTION_EPOCH_PLACEMENT_ERROR_F32 = 5
R06_ACTION_EPOCH_ON_TABLE_F32 = 6
R06_ACTION_EPOCH_CONTACT_VALID_F32 = 7
R06_ACTION_EPOCH_CROSSING_VALID_F32 = 8
R06_ACTION_EPOCH_NET_CROSSED_F32 = 9
R06_ACTION_EPOCH_NET_CLEAR_F32 = 10
R06_ACTION_EPOCH_FACT_F32_USED = 11
R06_ACTION_EPOCH_FACT_F32_WIDTH = 32

# These are causal inputs, not caller pins.  The Device-R05 child projection
# must carry the complete task identity that R06 will install, rather than a
# same-writer scalar digest or a value reconstructed by R06.  Keeping this
# public list exact makes the current production HOLD mechanically checkable
# while the upstream ABI is still incomplete.
DEVICE_R05_HOT_REVEAL_OWNER_KIND = "r06_flight"
DEVICE_R05_HOT_REVEAL_REQUIRED_PROJECTION_FIELDS = (
    "prepared_reveal",
    "owner_kind",
    "preview_identity",
    "selected_env_index",
    "selected_mask",
    "reset_generation",
    "swing_generation",
    "action_slot",
    "selected_candidate_identity",
    "full_key_sha256",
    "task_sha256",
    "action_uid",
)

C10_FAMILY_A = 0
C10_FAMILY_C = 1

PHASE_CONTACT = 0
PHASE_NET = 1
PHASE_LANDING = 2

FLIGHT_EMPTY = 0
FLIGHT_INBOUND = 1
FLIGHT_OPEN = 2
FLIGHT_SETTLED_RETAINED = 3

MAILBOX_EMPTY = 0
MAILBOX_SETTLED_UNPAID = 1
MAILBOX_PARTIALLY_PAID = 2
MAILBOX_PAID = 3

# Short aliases are kept for focused checkpoint/state tests.
EMPTY = FLIGHT_EMPTY
INBOUND = FLIGHT_INBOUND
OPEN = FLIGHT_OPEN
SETTLED_RETAINED = FLIGHT_SETTLED_RETAINED
SETTLED_UNPAID = MAILBOX_SETTLED_UNPAID
PARTIALLY_PAID = MAILBOX_PARTIALLY_PAID
PAID = MAILBOX_PAID

SETTLEMENT_CAUSE_NONE = 0
SETTLEMENT_CAUSE_FIRST_CROSSING = 1
SETTLEMENT_CAUSE_CONTACT_DEADLINE = 2
SETTLEMENT_CAUSE_CROSSING_HORIZON = 3
SETTLEMENT_CAUSE_NONFINITE = 4
SETTLEMENT_CAUSE_PRODUCER_CONTRACT_FAULT = 5
SETTLEMENT_CAUSE_ENGINE_OVERFLOW = 6
SETTLEMENT_CAUSE_PROTOCOL_FAULT = 7

_CONTROL_CROSSING_NONE = 0
_CONTROL_CROSSING_VALID = 1
_CONTROL_CROSSING_NONFINITE = 2

# Infrastructure settlement is intentionally outside C04's reason namespace.
CANONICAL_REASON_NOT_SCORED = -4

FAULT_INVALID_INSTALL = 1 << 0
FAULT_FLIGHT_COLLISION = 1 << 1
FAULT_MAILBOX_COLLISION = 1 << 2
FAULT_KEY_BINDING = 1 << 3
FAULT_GENERATION_BINDING = 1 << 4
FAULT_OBSERVATION_ORDINAL = 1 << 5
FAULT_INVALID_STAMP = 1 << 6
FAULT_STAMP_REGRESSION = 1 << 7
FAULT_CONTACT_ORDER = 1 << 8
FAULT_CROSSING_BEFORE_CONTACT = 1 << 9
FAULT_CROSSING_REPORT = 1 << 10
FAULT_NET_CONTRACT = 1 << 11
FAULT_FLIGHT_CONTINUITY = 1 << 12
FAULT_MAILBOX_COPY_COLLISION = 1 << 13
FAULT_DUPLICATE_VIEW = 1 << 14
FAULT_MISSED_PAYMENT = 1 << 15
FAULT_PAYMENT_BEFORE_VIEW = 1 << 16
FAULT_DUPLICATE_PAYMENT = 1 << 17
FAULT_INVALID_PAYMENT = 1 << 18
FAULT_PAYMENT_EPOCH = 1 << 19
FAULT_INVALID_RETIRE = 1 << 20
FAULT_INVALID_CLOSE = 1 << 21
FAULT_REPLAY = 1 << 22
FAULT_UNOBSERVED_LIVE_SLOT = 1 << 23
FAULT_INVALID_OBSERVATION = 1 << 24
FAULT_NONFINITE = 1 << 25
FAULT_PRODUCER_CONTRACT = 1 << 26
FAULT_ENGINE_OVERFLOW = 1 << 27
FAULT_TASK_DRAIN = 1 << 28
FAULT_BATCH_ABORT = 1 << 29
FAULT_SAFETY_CLEANUP = 1 << 30

FAULTS = (
    ("invalid_install", FAULT_INVALID_INSTALL),
    ("flight_collision", FAULT_FLIGHT_COLLISION),
    ("mailbox_collision", FAULT_MAILBOX_COLLISION),
    ("key_binding", FAULT_KEY_BINDING),
    ("generation_binding", FAULT_GENERATION_BINDING),
    ("observation_ordinal", FAULT_OBSERVATION_ORDINAL),
    ("invalid_stamp", FAULT_INVALID_STAMP),
    ("stamp_regression", FAULT_STAMP_REGRESSION),
    ("contact_order", FAULT_CONTACT_ORDER),
    ("crossing_before_contact", FAULT_CROSSING_BEFORE_CONTACT),
    ("crossing_report", FAULT_CROSSING_REPORT),
    ("net_contract", FAULT_NET_CONTRACT),
    ("flight_continuity", FAULT_FLIGHT_CONTINUITY),
    ("mailbox_copy_collision", FAULT_MAILBOX_COPY_COLLISION),
    ("duplicate_view", FAULT_DUPLICATE_VIEW),
    ("missed_payment", FAULT_MISSED_PAYMENT),
    ("payment_before_view", FAULT_PAYMENT_BEFORE_VIEW),
    ("duplicate_payment", FAULT_DUPLICATE_PAYMENT),
    ("invalid_payment", FAULT_INVALID_PAYMENT),
    ("payment_epoch", FAULT_PAYMENT_EPOCH),
    ("invalid_retire", FAULT_INVALID_RETIRE),
    ("invalid_close", FAULT_INVALID_CLOSE),
    ("replay", FAULT_REPLAY),
    ("unobserved_live_slot", FAULT_UNOBSERVED_LIVE_SLOT),
    ("invalid_observation", FAULT_INVALID_OBSERVATION),
    ("nonfinite", FAULT_NONFINITE),
    ("producer_contract", FAULT_PRODUCER_CONTRACT),
    ("engine_overflow", FAULT_ENGINE_OVERFLOW),
    ("task_drain", FAULT_TASK_DRAIN),
    ("batch_abort", FAULT_BATCH_ABORT),
    ("safety_cleanup", FAULT_SAFETY_CLEANUP),
)
# The lean ActionEpoch owner packs these R06-owned row causes into its existing
# optimizer-boundary transfer.  This is a distinct namespace from ``FAULTS``:
# the values are pinned to ActionEpoch's public row-fault ABI and are validated
# again when the exact owner is cold-bound below.
R06_EPOCH_ROW_FAULT_LAUNCH_SELECTION_CONTRACT = 1 << 11
R06_EPOCH_ROW_FAULT_LAUNCH_IDENTITY_CONTRACT = 1 << 12
R06_EPOCH_ROW_FAULT_OUTCOME_PROJECTION_DUPLICATE = 1 << 13
R06_EPOCH_ROW_FAULT_PAYMENT_PROJECTION_CONTRACT = 1 << 14
R06_EPOCH_ROW_FAULT_PAYMENT_MAILBOX_DUPLICATE = 1 << 15
R06_EPOCH_ROW_FAULT_PAYMENT_MISSING_OR_MISMATCHED = 1 << 16
R06_EPOCH_ROW_FAULT_PAYMENT_BEFORE_SETTLEMENT = 1 << 17
R06_EPOCH_ROW_FAULT_PAYMENT_HIGHWATER_REGRESSION = 1 << 18
R06_EPOCH_ROW_FAULT_PAYMENT_UNCONSUMED_DEBT_OVERWRITE = 1 << 19
R06_EPOCH_ROW_FAULT_CLOSED_PROJECTION_CONTRACT = 1 << 20
R06_EPOCH_ROW_FAULT_CLOSED_DEBT_MISMATCH = 1 << 21
R06_EPOCH_ROW_FAULT_CURRENT_FLIGHT_DUPLICATE = 1 << 22

R06_ACTION_EPOCH_ROW_FAULT_BINDINGS = (
    (
        "ROW_FAULT_R06_LAUNCH_SELECTION_CONTRACT",
        R06_EPOCH_ROW_FAULT_LAUNCH_SELECTION_CONTRACT,
        "r06_launch_selection_contract",
    ),
    (
        "ROW_FAULT_R06_LAUNCH_IDENTITY_CONTRACT",
        R06_EPOCH_ROW_FAULT_LAUNCH_IDENTITY_CONTRACT,
        "r06_launch_identity_contract",
    ),
    (
        "ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE",
        R06_EPOCH_ROW_FAULT_OUTCOME_PROJECTION_DUPLICATE,
        "r06_outcome_projection_duplicate",
    ),
    (
        "ROW_FAULT_R06_PAYMENT_PROJECTION_CONTRACT",
        R06_EPOCH_ROW_FAULT_PAYMENT_PROJECTION_CONTRACT,
        "r06_payment_projection_contract",
    ),
    (
        "ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE",
        R06_EPOCH_ROW_FAULT_PAYMENT_MAILBOX_DUPLICATE,
        "r06_payment_mailbox_duplicate",
    ),
    (
        "ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED",
        R06_EPOCH_ROW_FAULT_PAYMENT_MISSING_OR_MISMATCHED,
        "r06_payment_missing_or_mismatched",
    ),
    (
        "ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT",
        R06_EPOCH_ROW_FAULT_PAYMENT_BEFORE_SETTLEMENT,
        "r06_payment_before_settlement",
    ),
    (
        "ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION",
        R06_EPOCH_ROW_FAULT_PAYMENT_HIGHWATER_REGRESSION,
        "r06_payment_highwater_regression",
    ),
    (
        "ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE",
        R06_EPOCH_ROW_FAULT_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
        "r06_payment_unconsumed_debt_overwrite",
    ),
    (
        "ROW_FAULT_R06_CLOSED_PROJECTION_CONTRACT",
        R06_EPOCH_ROW_FAULT_CLOSED_PROJECTION_CONTRACT,
        "r06_closed_projection_contract",
    ),
    (
        "ROW_FAULT_R06_CLOSED_DEBT_MISMATCH",
        R06_EPOCH_ROW_FAULT_CLOSED_DEBT_MISMATCH,
        "r06_closed_debt_mismatch",
    ),
    (
        "ROW_FAULT_R06_CURRENT_FLIGHT_DUPLICATE",
        R06_EPOCH_ROW_FAULT_CURRENT_FLIGHT_DUPLICATE,
        "r06_current_flight_duplicate",
    ),
)
_R06_ACTION_EPOCH_ROW_FAULT_BITS = frozenset(
    bit for _constant, bit, _reason in R06_ACTION_EPOCH_ROW_FAULT_BINDINGS
)

# R06 owns this exact row schema.  The all-owner boundary merely packs the
# already-materialized verdict; it may neither reinterpret nor widen the
# child's fault domain.
R06_REVEAL_BOUNDARY_FAULT_SCHEMA = (
    _reveal_boundary.ActionBallFullMdpRevealBoundaryFaultSchema(
        schema_version=1,
        owner_kind="r06_flight",
        ordered_fault_bits=FAULTS,
        allowed_fault_mask=sum(bit for _name, bit in FAULTS),
        precedence=tuple(name for name, _bit in FAULTS),
    )
)
R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256 = (
    "338fe8c9d28465625bd1287624fb840e35565923d5ffab8a1fb56632fd533fbe"
)
if (
    R06_REVEAL_BOUNDARY_FAULT_SCHEMA.schema_sha256
    != R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256
):
    raise RuntimeError("R06 reveal-boundary fault-schema pin drifted")

INVARIANT_NAMES = (
    "flight_state",
    "mailbox_state",
    "flight_owner",
    "mailbox_owner",
    "reservation_mapping",
    "retained_mapping",
    "physical_retirement",
    "contact_state",
    "observation_sequence",
    "stamp_order",
    "consumer_state",
    "view_payment_epoch",
    "payment_value",
    "canonical_score",
    "replay_guard",
    "counters",
)

R06_GLOBAL_DRAIN_REQUIRED_FIELDS = (
    "mutation_version",
    "fault_count",
    "invariant_count",
    "terminal_resolution_total",
    "shared_normal_retire_total",
    "r06_only_orphan_retire_total",
    "shared_normal_retire_key_summary_0",
    "shared_normal_retire_key_summary_1",
)
R06_GLOBAL_DRAIN_FAULT_FIELDS = tuple(
    f"fault_{name}_count" for name, _bit in FAULTS
)
R06_GLOBAL_DRAIN_FLIGHT_STATE_FIELDS = tuple(
    f"flight_state_{name}_count"
    for name in ("empty", "inbound", "open", "settled_retained")
)
R06_GLOBAL_DRAIN_MAILBOX_STATE_FIELDS = tuple(
    f"mailbox_state_{name}_count"
    for name in ("empty", "settled_unpaid", "partially_paid", "paid")
)
R06_GLOBAL_DRAIN_INVARIANT_FIELDS = tuple(
    f"invariant_{name}_count" for name in INVARIANT_NAMES
)
R06_GLOBAL_DRAIN_PORTABLE_TOTAL_FIELDS = (
    "installed_total",
    "settled_total",
    "retired_total",
    "common_payment_total",
    "placement_payment_total",
    "closed_total",
)
R06_GLOBAL_DRAIN_FIELD_NAMES = (
    *R06_GLOBAL_DRAIN_REQUIRED_FIELDS,
    *R06_GLOBAL_DRAIN_FAULT_FIELDS,
    *R06_GLOBAL_DRAIN_FLIGHT_STATE_FIELDS,
    *R06_GLOBAL_DRAIN_MAILBOX_STATE_FIELDS,
    *R06_GLOBAL_DRAIN_INVARIANT_FIELDS,
    *R06_GLOBAL_DRAIN_PORTABLE_TOTAL_FIELDS,
)
R06_PPO_DRAIN_LEAF_SCHEMA = (
    _R06_GLOBAL_DRAIN_OWNER_KIND,
    tuple((name, "scalar", 0) for name in R06_GLOBAL_DRAIN_FIELD_NAMES),
)


def materialize_r06_ppo_drain_leaf_schema(
    *,
    leaf_schema_type: type,
    field_spec_type: type,
) -> object:
    """Build R06's complete dependency-neutral global-drain schema."""

    owner_kind, fields = R06_PPO_DRAIN_LEAF_SCHEMA
    return leaf_schema_type(
        owner_kind=owner_kind,
        fields=tuple(
            field_spec_type(
                name=name,
                cardinality=cardinality,
                minimum=minimum,
            )
            for name, cardinality, minimum in fields
        ),
    )

_INT_KEY_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "shot_index",
)
_DIGEST_KEY_FIELDS = (
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
    "run_id",
    "carry_chain_id",
    "source_sha256",
    "config_sha256",
    "receipt_content_sha256",
)
_KEY_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
    "run_id",
    "carry_chain_id",
    "shot_index",
    "source_sha256",
    "config_sha256",
    "receipt_content_sha256",
)
_MAX_ACTION_UID = (1 << 53) - 1


class LandingOutcomeDeviceError(RuntimeError):
    """Host ABI, checkpoint, or consumer-name contract failed."""


class LandingOutcomeDeviceR05HotRevealProductionHold(
    LandingOutcomeDeviceError
):
    """The causal Device-R05/Physical identity join is not complete yet."""


@dataclass(frozen=True)
class LandingOutcomeTextRegistry:
    """Canonical host registry that owns the exact run/carry text bytes."""

    run_ids: tuple[str, ...]
    carry_chain_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("run_ids", "carry_chain_ids"):
            raw = getattr(self, name)
            if type(raw) is not tuple:
                raise LandingOutcomeDeviceError(f"{name} must be an exact tuple")
            values = tuple(
                _nonempty_text(item, label=f"{name}[{index}]")
                for index, item in enumerate(raw)
            )
            if values != tuple(sorted(set(values))):
                raise LandingOutcomeDeviceError(
                    f"{name} must be unique and lexicographically ordered"
                )
            object.__setattr__(self, name, values)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_REGISTRY_SCHEMA_VERSION,
            "kind": TEXT_REGISTRY_KIND,
            "run_ids": list(self.run_ids),
            "carry_chain_ids": list(self.carry_chain_ids),
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        payload = self.payload()
        return {**payload, "canonical_sha256": _canonical_sha256(payload)}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        expected_registry_sha256: str,
    ) -> "LandingOutcomeTextRegistry":
        expected = {
            "schema_version",
            "kind",
            "run_ids",
            "carry_chain_ids",
            "canonical_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise LandingOutcomeDeviceError("text registry mapping fields differ")
        expected_root = _sha256_hex(
            expected_registry_sha256,
            label="expected_registry_sha256",
        )
        payload = {name: value[name] for name in expected if name != "canonical_sha256"}
        declared = _sha256_hex(
            value["canonical_sha256"],
            label="text_registry.canonical_sha256",
        )
        if (
            payload["schema_version"] != TEXT_REGISTRY_SCHEMA_VERSION
            or payload["kind"] != TEXT_REGISTRY_KIND
            or _canonical_sha256(payload) != declared
            or declared != expected_root
        ):
            raise LandingOutcomeDeviceError("text registry seal/root differs")
        run_ids = payload["run_ids"]
        carry_ids = payload["carry_chain_ids"]
        if not isinstance(run_ids, (tuple, list)) or not isinstance(
            carry_ids, (tuple, list)
        ):
            raise LandingOutcomeDeviceError("text registry rows must be sequences")
        result = cls(run_ids=tuple(run_ids), carry_chain_ids=tuple(carry_ids))
        if result.canonical_sha256 != declared:
            raise LandingOutcomeDeviceError("text registry normalization differs")
        return result

    def token_sha256(self, *, namespace: str, text: str) -> str:
        if namespace == "run_id":
            values = self.run_ids
        elif namespace == "carry_chain_id":
            values = self.carry_chain_ids
        else:
            raise LandingOutcomeDeviceError("text registry namespace differs")
        if text not in values:
            raise LandingOutcomeDeviceError(
                f"{namespace} is absent from the sealed text registry"
            )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DiagnosticN2NoSaveNamespaceProjection:
    """Opaque R06-owned A/C text namespace for the disposable diagnostic.

    A later R05 diagnostic constructor must present this exact object back to
    the same R06 owner.  Reading or independently reproducing the strings is
    not an ownership proof and this handle is intentionally non-portable.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> object:
        del cls
        raise TypeError("diagnostic R06 namespace projections are owner-issued only")

    def __reduce__(self) -> object:
        raise TypeError("diagnostic R06 namespace projections cannot be serialized")

    def __copy__(self) -> object:
        raise TypeError("diagnostic R06 namespace projections cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("diagnostic R06 namespace projections cannot be copied")


@dataclass(frozen=True)
class DiagnosticN2NoSaveNamespaceView:
    """Clone-only strings returned after the issuing R06 authenticates a handle."""

    family: str
    run_id: str
    carry_chain_id: str


@dataclass(frozen=True)
class _DiagnosticN2NoSaveNamespaceRecord:
    owner_ref: object
    family: str
    run_id: str
    carry_chain_id: str


_DIAGNOSTIC_N2_NAMESPACE_REGISTRY: weakref.WeakKeyDictionary[
    DiagnosticN2NoSaveNamespaceProjection,
    _DiagnosticN2NoSaveNamespaceRecord,
] = weakref.WeakKeyDictionary()
_DIAGNOSTIC_N2_REGISTRY_LOCK = threading.RLock()


class DiagnosticN2NoSaveConstructionBinding:
    """Module-issued, process-local authority to allocate one diagnostic R06."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> object:
        del cls
        raise TypeError("diagnostic R06 construction bindings are factory-issued only")

    def __reduce__(self) -> object:
        raise TypeError("diagnostic R06 construction bindings cannot be serialized")

    def __copy__(self) -> object:
        raise TypeError("diagnostic R06 construction bindings cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("diagnostic R06 construction bindings cannot be copied")


@dataclass(frozen=True)
class _DiagnosticN2NoSaveConstructionRecord:
    env: object
    env_cfg: object
    physical_owner: object
    scene_port: object
    scene_port_capability: object
    scene_spec: object
    diagnostic_capacity_binding: object
    num_envs: int
    family: str
    device: torch.device
    profile: LandingPlacementProfile
    text_registry: LandingOutcomeTextRegistry
    diagnostic_family_gain_identity_sha256: str


_DIAGNOSTIC_N2_CONSTRUCTION_REGISTRY: weakref.WeakKeyDictionary[
    DiagnosticN2NoSaveConstructionBinding,
    _DiagnosticN2NoSaveConstructionRecord,
] = weakref.WeakKeyDictionary()
_DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS: dict[
    int, DiagnosticN2NoSaveConstructionBinding
] = {}


def _issue_diagnostic_n2_no_save_construction_binding(
    record: _DiagnosticN2NoSaveConstructionRecord,
) -> DiagnosticN2NoSaveConstructionBinding:
    with _DIAGNOSTIC_N2_REGISTRY_LOCK:
        binding = object.__new__(DiagnosticN2NoSaveConstructionBinding)
        _DIAGNOSTIC_N2_CONSTRUCTION_REGISTRY[binding] = record
    return binding


def _owned_diagnostic_n2_no_save_construction_binding(
    value: object,
) -> _DiagnosticN2NoSaveConstructionRecord:
    if type(value) is not DiagnosticN2NoSaveConstructionBinding:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 construction requires its module-issued binding"
        )
    with _DIAGNOSTIC_N2_REGISTRY_LOCK:
        try:
            return _DIAGNOSTIC_N2_CONSTRUCTION_REGISTRY[value]
        except (KeyError, TypeError) as exc:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 construction binding is stale or foreign"
            ) from exc


@dataclass(frozen=True)
class LandingOutcomeCapacityAuthority:
    """Opaque R03-validated cadence/horizon/capacity authority."""

    _payload_json: bytes
    _auth_tag: bytes
    _token: object

    def _payload(self) -> dict[str, object]:
        return _owned_capacity_authority(self)

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._payload_json).hexdigest()

    def __getattr__(self, name: str) -> object:
        public = {
            "materialization_sha256",
            "numeric_authority_sha256",
            "resolved_graph_receipt_sha256",
            "four_shot_tape_receipt_sha256",
            "policy_clock",
            "control_step_clock_sha256",
            "inclusive_event_order_witness_sha256",
            "cadence_ticks",
            "flight_horizon_ticks",
            "flight_horizon_witness_sha256",
            "required_flight_slot_capacity",
            "flight_slot_capacity",
            "mailbox_horizon_ticks",
            "mailbox_horizon_witness_sha256",
            "required_mailbox_capacity",
            "mailbox_capacity",
            "tail_closure_tick",
        }
        if name not in public:
            raise AttributeError(name)
        return self._payload()[name]

    def to_mapping(self) -> dict[str, object]:
        payload = self._payload()
        return {**payload, "canonical_sha256": self.canonical_sha256}


def _owned_capacity_authority(value: object) -> dict[str, object]:
    if (
        type(value) is not LandingOutcomeCapacityAuthority
        or value._token is not _CAPACITY_AUTH_TOKEN
        or type(value._payload_json) is not bytes
        or type(value._auth_tag) is not bytes
    ):
        raise LandingOutcomeDeviceError(
            "capacity authority must come from build_landing_outcome_capacity_authority"
        )
    expected_tag = hmac.new(
        _CAPACITY_AUTH_KEY,
        value._payload_json,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(value._auth_tag, expected_tag):
        raise LandingOutcomeDeviceError("capacity authority authentication differs")
    try:
        payload = json.loads(value._payload_json.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LandingOutcomeDeviceError("capacity authority payload is invalid") from exc
    expected = {
        "schema_version",
        "kind",
        "materialization_sha256",
        "numeric_authority_sha256",
        "resolved_graph_receipt_sha256",
        "four_shot_tape_receipt_sha256",
        "policy_clock",
        "control_step_clock_sha256",
        "inclusive_event_order_witness_sha256",
        "cadence_ticks",
        "flight_horizon_ticks",
        "flight_horizon_witness_sha256",
        "required_flight_slot_capacity",
        "flight_slot_capacity",
        "mailbox_horizon_ticks",
        "mailbox_horizon_witness_sha256",
        "required_mailbox_capacity",
        "mailbox_capacity",
        "tail_closure_tick",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LandingOutcomeDeviceError("capacity authority fields differ")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["kind"] != CAPACITY_AUTHORITY_KIND
    ):
        raise LandingOutcomeDeviceError("capacity authority schema differs")
    for name in (
        "materialization_sha256",
        "numeric_authority_sha256",
        "resolved_graph_receipt_sha256",
        "four_shot_tape_receipt_sha256",
        "control_step_clock_sha256",
        "inclusive_event_order_witness_sha256",
        "flight_horizon_witness_sha256",
        "mailbox_horizon_witness_sha256",
    ):
        _sha256_hex(payload[name], label=f"capacity_authority.{name}")
    if not isinstance(payload["policy_clock"], dict):
        raise LandingOutcomeDeviceError("capacity authority policy clock differs")
    clock_binding_payload = {
        "kind": CAPACITY_CLOCK_BINDING_KIND,
        "resolved_graph_receipt_sha256": payload[
            "resolved_graph_receipt_sha256"
        ],
        "four_shot_tape_receipt_sha256": payload[
            "four_shot_tape_receipt_sha256"
        ],
        "policy_clock": payload["policy_clock"],
    }
    if _canonical_sha256(clock_binding_payload) != payload[
        "control_step_clock_sha256"
    ]:
        raise LandingOutcomeDeviceError("capacity control-step clock root differs")
    for name, minimum in (
        ("cadence_ticks", 1),
        ("flight_horizon_ticks", 0),
        ("required_flight_slot_capacity", 1),
        ("flight_slot_capacity", 1),
        ("mailbox_horizon_ticks", 0),
        ("required_mailbox_capacity", 1),
        ("mailbox_capacity", 1),
        ("tail_closure_tick", 0),
    ):
        _exact_int(payload[name], label=f"capacity_authority.{name}", minimum=minimum)
    cadence = payload["cadence_ticks"]
    required_flight = payload["flight_horizon_ticks"] // cadence + 1
    required_mailbox = payload["mailbox_horizon_ticks"] // cadence + 1
    if (
        payload["required_flight_slot_capacity"] != required_flight
        or payload["flight_slot_capacity"] != required_flight
        or payload["required_mailbox_capacity"] != required_mailbox
        or payload["mailbox_capacity"] < required_mailbox
    ):
        raise LandingOutcomeDeviceError("capacity authority H/C/K relation differs")
    inclusive_event_order_payload = {
        "kind": CAPACITY_INCLUSIVE_EVENT_ORDER_KIND,
        "control_step_clock_sha256": payload["control_step_clock_sha256"],
        "four_shot_tape_receipt_sha256": payload[
            "four_shot_tape_receipt_sha256"
        ],
        "interval_semantics": "closed_reveal_through_release_ticks",
        "same_tick_order": "new_reveal_admission_before_prior_owner_release",
        "capacity_formula": "floor(horizon_ticks/cadence_ticks)+1",
        "cadence_ticks": payload["cadence_ticks"],
        "flight_horizon_ticks": payload["flight_horizon_ticks"],
        "flight_horizon_witness_sha256": payload[
            "flight_horizon_witness_sha256"
        ],
        "mailbox_horizon_ticks": payload["mailbox_horizon_ticks"],
        "mailbox_horizon_witness_sha256": payload[
            "mailbox_horizon_witness_sha256"
        ],
        "tail_closure_tick": payload["tail_closure_tick"],
    }
    if _canonical_sha256(inclusive_event_order_payload) != payload[
        "inclusive_event_order_witness_sha256"
    ]:
        raise LandingOutcomeDeviceError("capacity inclusive event-order root differs")
    return payload


def build_landing_outcome_capacity_authority(
    numeric_materialization_receipt: object,
    *,
    expected_materialization_sha256: str,
    expected_numeric_authority_sha256: str,
    resolved_graph_receipt: object,
    four_shot_tape_receipt: object,
    candidate_set_receipt: object,
    trusted_input_roots: object,
) -> LandingOutcomeCapacityAuthority:
    """Recompute R03 READY authority from all external sealed evidence."""

    expected_materialization = _sha256_hex(
        expected_materialization_sha256,
        label="expected_materialization_sha256",
    )
    expected_numeric = _sha256_hex(
        expected_numeric_authority_sha256,
        label="expected_numeric_authority_sha256",
    )
    try:
        validated = _r03.validate_numeric_materialization_receipt(
            numeric_materialization_receipt,
            resolved_graph_receipt=resolved_graph_receipt,
            four_shot_tape_receipt=four_shot_tape_receipt,
            candidate_set_receipt=candidate_set_receipt,
            trusted_input_roots=trusted_input_roots,
        )
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "R03 numeric capacity authority verification failed"
        ) from exc
    if (
        validated.get("status") != _r03.READY_STATUS
        or validated.get("budget_authority_ready") is not True
        or validated.get("materialization_sha256") != expected_materialization
        or validated.get("authority_sha256") != expected_numeric
    ):
        raise LandingOutcomeDeviceError("R03 numeric capacity external roots differ")
    authority = validated.get("authority_payload")
    if not isinstance(authority, Mapping):
        raise LandingOutcomeDeviceError("R03 numeric authority payload differs")
    evidence = authority.get("capacity_evidence")
    if not isinstance(evidence, Mapping):
        raise LandingOutcomeDeviceError("R03 capacity evidence differs")
    expected_evidence = {
        "cadence_ticks",
        "flight_horizon_ticks",
        "flight_horizon_witness_sha256",
        "required_worst_case_flight_capacity",
        "flight_capacity",
        "mailbox_horizon_ticks",
        "mailbox_horizon_witness_sha256",
        "required_worst_case_mailbox_capacity",
        "mailbox_capacity",
        "tail_closure_tick",
    }
    if set(evidence) != expected_evidence:
        raise LandingOutcomeDeviceError("R03 capacity evidence fields differ")
    input_receipts = authority.get("input_receipts")
    policy_clock = authority.get("policy_clock")
    if not isinstance(input_receipts, Mapping) or not isinstance(
        policy_clock, Mapping
    ):
        raise LandingOutcomeDeviceError("R03 capacity clock/input binding differs")
    resolved_graph_sha = _sha256_hex(
        input_receipts.get("resolved_graph_receipt_sha256"),
        label="R03 resolved_graph_receipt_sha256",
    )
    four_shot_tape_sha = _sha256_hex(
        input_receipts.get("four_shot_tape_receipt_sha256"),
        label="R03 four_shot_tape_receipt_sha256",
    )
    clock_binding_payload = {
        "kind": CAPACITY_CLOCK_BINDING_KIND,
        "resolved_graph_receipt_sha256": resolved_graph_sha,
        "four_shot_tape_receipt_sha256": four_shot_tape_sha,
        "policy_clock": policy_clock,
        "policy_clock": policy_clock,
    }
    control_step_clock_sha = _canonical_sha256(clock_binding_payload)
    inclusive_event_order_payload = {
        "kind": CAPACITY_INCLUSIVE_EVENT_ORDER_KIND,
        "control_step_clock_sha256": control_step_clock_sha,
        "four_shot_tape_receipt_sha256": four_shot_tape_sha,
        "interval_semantics": "closed_reveal_through_release_ticks",
        "same_tick_order": "new_reveal_admission_before_prior_owner_release",
        "capacity_formula": "floor(horizon_ticks/cadence_ticks)+1",
        "cadence_ticks": evidence["cadence_ticks"],
        "flight_horizon_ticks": evidence["flight_horizon_ticks"],
        "flight_horizon_witness_sha256": evidence[
            "flight_horizon_witness_sha256"
        ],
        "mailbox_horizon_ticks": evidence["mailbox_horizon_ticks"],
        "mailbox_horizon_witness_sha256": evidence[
            "mailbox_horizon_witness_sha256"
        ],
        "tail_closure_tick": evidence["tail_closure_tick"],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CAPACITY_AUTHORITY_KIND,
        "materialization_sha256": expected_materialization,
        "numeric_authority_sha256": expected_numeric,
        "resolved_graph_receipt_sha256": resolved_graph_sha,
        "four_shot_tape_receipt_sha256": four_shot_tape_sha,
        "control_step_clock_sha256": control_step_clock_sha,
        "inclusive_event_order_witness_sha256": _canonical_sha256(
            inclusive_event_order_payload
        ),
        "cadence_ticks": evidence["cadence_ticks"],
        "flight_horizon_ticks": evidence["flight_horizon_ticks"],
        "flight_horizon_witness_sha256": evidence[
            "flight_horizon_witness_sha256"
        ],
        "required_flight_slot_capacity": evidence[
            "required_worst_case_flight_capacity"
        ],
        "flight_slot_capacity": evidence["flight_capacity"],
        "mailbox_horizon_ticks": evidence["mailbox_horizon_ticks"],
        "mailbox_horizon_witness_sha256": evidence[
            "mailbox_horizon_witness_sha256"
        ],
        "required_mailbox_capacity": evidence[
            "required_worst_case_mailbox_capacity"
        ],
        "mailbox_capacity": evidence["mailbox_capacity"],
        "tail_closure_tick": evidence["tail_closure_tick"],
    }
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    result = LandingOutcomeCapacityAuthority(
        _payload_json=payload_json,
        _auth_tag=hmac.new(
            _CAPACITY_AUTH_KEY,
            payload_json,
            hashlib.sha256,
        ).digest(),
        _token=_CAPACITY_AUTH_TOKEN,
    )
    _owned_capacity_authority(result)
    return result


@dataclass(frozen=True)
class LandingOutcomeC10FamilyPaymentAuthority:
    """Opaque C10-authenticated family identity and placement gain."""

    _payload_json: bytes
    _auth_tag: bytes
    _token: object

    def _payload(self) -> dict[str, object]:
        return _owned_c10_payment_authority(self)

    @property
    def family(self) -> str:
        return str(self._payload()["family"])

    @property
    def placement_treatment_gain(self) -> float:
        return float(self._payload()["placement_treatment_gain"])

    @property
    def projection_sha256(self) -> str:
        return str(self._payload()["projection_sha256"])

    @property
    def identity_sha256(self) -> str:
        return str(self._payload()["identity_sha256"])

    @property
    def contract_sha256(self) -> str:
        return str(self._payload()["contract_sha256"])

    @property
    def common_on_table_manager_weight(self) -> float:
        return float(self._payload()["common_on_table_manager_weight"])

    @property
    def placement_manager_shell_weight(self) -> float:
        return float(self._payload()["placement_manager_shell_weight"])

    @property
    def post_dt_budget_sha256(self) -> str:
        return str(self._payload()["post_dt_budget_sha256"])

    @property
    def landing_profile_sha256(self) -> str:
        return str(self._payload()["landing_profile_sha256"])

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._payload_json).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        payload = self._payload()
        return {**payload, "canonical_sha256": self.canonical_sha256}


def _owned_c10_payment_authority(
    value: object,
) -> dict[str, object]:
    if (
        type(value) is not LandingOutcomeC10FamilyPaymentAuthority
        or value._token is not _C10_PAYMENT_AUTH_TOKEN
        or type(value._payload_json) is not bytes
        or type(value._auth_tag) is not bytes
    ):
        raise LandingOutcomeDeviceError(
            "payment_authority must come from build_c10_family_payment_authority"
        )
    expected_tag = hmac.new(
        _C10_PAYMENT_AUTH_KEY,
        value._payload_json,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(value._auth_tag, expected_tag):
        raise LandingOutcomeDeviceError("C10 payment authority authentication differs")
    try:
        payload = json.loads(value._payload_json.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LandingOutcomeDeviceError("C10 payment authority payload is invalid") from exc
    expected = {
        "schema_version",
        "kind",
        "family",
        "placement_treatment_gain",
        "projection_sha256",
        "identity_sha256",
        "contract_sha256",
        "common_on_table_manager_weight",
        "placement_manager_shell_weight",
        "post_dt_budget_sha256",
        "landing_profile_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LandingOutcomeDeviceError("C10 payment authority fields differ")
    if payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != C10_PAYMENT_AUTHORITY_KIND:
        raise LandingOutcomeDeviceError("C10 payment authority schema differs")
    family = payload["family"]
    gain = payload["placement_treatment_gain"]
    if family not in ("A", "C") or gain != (1.0 if family == "A" else 0.0):
        raise LandingOutcomeDeviceError("C10 family/gain binding differs")
    for name in (
        "projection_sha256",
        "identity_sha256",
        "contract_sha256",
        "post_dt_budget_sha256",
        "landing_profile_sha256",
    ):
        _sha256_hex(payload[name], label=f"payment_authority.{name}")
    for name in (
        "common_on_table_manager_weight",
        "placement_manager_shell_weight",
    ):
        value_number = payload[name]
        if (
            isinstance(value_number, bool)
            or not isinstance(value_number, (int, float))
            or not math.isfinite(float(value_number))
            or float(value_number) <= 0.0
        ):
            raise LandingOutcomeDeviceError(f"payment_authority.{name} differs")
    return payload


def build_c10_family_payment_authority(
    projection: _c10.C10ResolvedProjection,
    *,
    expected_projection_sha256: str,
    expected_identity_sha256: str,
    expected_c10_contract_sha256: str,
) -> LandingOutcomeC10FamilyPaymentAuthority:
    """Authenticate one builder-owned C10 projection and freeze its family."""

    expected_projection = _sha256_hex(
        expected_projection_sha256,
        label="expected_projection_sha256",
    )
    expected_identity = _sha256_hex(
        expected_identity_sha256,
        label="expected_identity_sha256",
    )
    expected_contract = _sha256_hex(
        expected_c10_contract_sha256,
        label="expected_c10_contract_sha256",
    )
    try:
        artifact = _c10.c10_projection_artifact(projection)
    except Exception as exc:
        raise LandingOutcomeDeviceError("C10 projection authentication failed") from exc
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "canonical_sha256",
        "projection",
    }:
        raise LandingOutcomeDeviceError("C10 projection artifact schema differs")
    projection_payload = artifact["projection"]
    if not isinstance(projection_payload, Mapping):
        raise LandingOutcomeDeviceError("C10 projection payload differs")
    projection_sha = _sha256_hex(
        artifact["canonical_sha256"],
        label="C10 projection canonical_sha256",
    )
    if projection_sha != expected_projection:
        raise LandingOutcomeDeviceError("C10 projection external pin differs")
    identity = projection_payload.get("identity")
    if not isinstance(identity, Mapping):
        raise LandingOutcomeDeviceError("C10 non-policy identity lineage differs")
    identity_sha = _canonical_sha256(dict(identity))
    if identity_sha != expected_identity:
        raise LandingOutcomeDeviceError("C10 identity lineage external pin differs")
    if (
        expected_contract != _c10.C10_CONTRACT_AUTHORITY_SHA256
        or projection_payload.get("contract_authority_sha256") != expected_contract
    ):
        raise LandingOutcomeDeviceError("C10 contract authority pin differs")
    treatment = projection_payload.get("treatment")
    common_runtime = projection_payload.get("common_runtime")
    if not isinstance(treatment, Mapping) or not isinstance(common_runtime, Mapping):
        raise LandingOutcomeDeviceError("C10 treatment/common payload differs")
    common = common_runtime.get("contract")
    if not isinstance(common, Mapping):
        raise LandingOutcomeDeviceError("C10 common runtime contract differs")
    family = projection_payload.get("family")
    gain = treatment.get("post_contact_placement_treatment_gain")
    if family not in ("A", "C") or gain != (1.0 if family == "A" else 0.0):
        raise LandingOutcomeDeviceError("C10 family treatment reversal detected")
    common_weight = common.get("on_table_success_weight")
    placement_weight = common.get("post_contact_placement_manager_weight")
    post_dt_budget_sha = _sha256_hex(
        common.get("post_dt_budget_receipt_sha256"),
        label="C10 post_dt_budget_receipt_sha256",
    )
    placement_source = common.get("placement_source")
    if not isinstance(placement_source, Mapping):
        raise LandingOutcomeDeviceError("C10 placement source differs")
    profile_root = _sha256_hex(
        placement_source.get("canonical_profile_sha256"),
        label="C10 canonical_profile_sha256",
    )
    try:
        reparsed_profile = LandingPlacementProfile.from_mapping(
            placement_source.get("canonical_profile")
        )
    except Exception as exc:
        raise LandingOutcomeDeviceError("C10 canonical C04 profile seal differs") from exc
    if reparsed_profile.canonical_sha256 != profile_root:
        raise LandingOutcomeDeviceError("C10 canonical C04 profile root differs")
    for name, number in (
        ("on_table_success_weight", common_weight),
        ("post_contact_placement_manager_weight", placement_weight),
    ):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise LandingOutcomeDeviceError(f"C10 {name} differs")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": C10_PAYMENT_AUTHORITY_KIND,
        "family": family,
        "placement_treatment_gain": float(gain),
        "projection_sha256": projection_sha,
        "identity_sha256": identity_sha,
        "contract_sha256": expected_contract,
        "common_on_table_manager_weight": float(common_weight),
        "placement_manager_shell_weight": float(placement_weight),
        "post_dt_budget_sha256": post_dt_budget_sha,
        "landing_profile_sha256": profile_root,
    }
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return LandingOutcomeC10FamilyPaymentAuthority(
        _payload_json=payload_json,
        _auth_tag=hmac.new(
            _C10_PAYMENT_AUTH_KEY,
            payload_json,
            hashlib.sha256,
        ).digest(),
        _token=_C10_PAYMENT_AUTH_TOKEN,
    )


@dataclass(frozen=True)
class LandingOutcomeRuntimeBinding:
    """Explicit runtime authority; every field is mandatory."""

    common_on_table_manager_weight: float
    placement_manager_shell_weight: float
    post_dt_budget_sha256: str
    text_registry_sha256: str
    capacity_authority_sha256: str
    numeric_materialization_sha256: str
    numeric_authority_sha256: str
    resolved_graph_receipt_sha256: str
    four_shot_tape_receipt_sha256: str
    control_step_clock_sha256: str
    inclusive_event_order_witness_sha256: str
    flight_horizon_witness_sha256: str
    mailbox_horizon_witness_sha256: str
    cadence_ticks: int
    flight_horizon_ticks: int
    mailbox_horizon_ticks: int
    flight_slot_capacity: int
    mailbox_capacity: int
    tail_closure_tick: int
    r05_source_sha256: str
    r05_contract_sha256: str
    c05_source_sha256: str
    landing_profile_sha256: str
    c10_contract_sha256: str
    c10_projection_sha256: str
    c10_identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "common_on_table_manager_weight",
            "placement_manager_shell_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LandingOutcomeDeviceError(f"{name} must be finite and positive")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise LandingOutcomeDeviceError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "post_dt_budget_sha256",
            "text_registry_sha256",
            "capacity_authority_sha256",
            "numeric_materialization_sha256",
            "numeric_authority_sha256",
            "resolved_graph_receipt_sha256",
            "four_shot_tape_receipt_sha256",
            "control_step_clock_sha256",
            "inclusive_event_order_witness_sha256",
            "flight_horizon_witness_sha256",
            "mailbox_horizon_witness_sha256",
            "r05_source_sha256",
            "r05_contract_sha256",
            "c05_source_sha256",
            "landing_profile_sha256",
            "c10_contract_sha256",
            "c10_projection_sha256",
            "c10_identity_sha256",
        ):
            value = getattr(self, name)
            _sha256_hex(value, label=name)
        for name, minimum in (
            ("cadence_ticks", 1),
            ("flight_horizon_ticks", 0),
            ("mailbox_horizon_ticks", 0),
            ("flight_slot_capacity", 1),
            ("mailbox_capacity", 1),
            ("tail_closure_tick", 0),
        ):
            _exact_int(getattr(self, name), label=name, minimum=minimum)
        expected_r05_source = (
            R05_RUNTIME_TRANSACTION_SOURCE_SHA256
            or R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256
        )
        if self.r05_source_sha256 != expected_r05_source:
            raise LandingOutcomeDeviceError("runtime R05 source pin differs")
        if self.r05_contract_sha256 != R05_RUNTIME_TRANSACTION_CONTRACT_SHA256:
            raise LandingOutcomeDeviceError("runtime R05 contract pin differs")
        if self.c05_source_sha256 != C05_LANDING_OUTCOME_SOURCE_SHA256:
            raise LandingOutcomeDeviceError("runtime C05 source pin differs")
        if self.c10_contract_sha256 != _c10.C10_CONTRACT_AUTHORITY_SHA256:
            raise LandingOutcomeDeviceError("runtime C10 contract pin differs")


@dataclass(frozen=True)
class LandingRevealInstallReceipt:
    """Opaque host receipt sealing every R05-derived install row."""

    _payload_json: bytes
    _auth_tag: bytes
    _token: object

    def _payload(self) -> dict[str, object]:
        return _owned_install_receipt(self)

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        raw = self._payload()["rows"]
        return tuple(dict(row) for row in raw)

    @property
    def num_envs(self) -> int:
        return int(self._payload()["num_envs"])

    @property
    def r05_source_sha256(self) -> str:
        return str(self._payload()["r05_source_sha256"])

    @property
    def r05_contract_sha256(self) -> str:
        return str(self._payload()["r05_contract_sha256"])

    @property
    def r05_final_source_pin_pending(self) -> bool:
        return bool(self._payload()["r05_final_source_pin_pending"])

    @property
    def expected_reveal_final_preview_sha256(self) -> str:
        return str(self._payload()["expected_reveal_final_preview_sha256"])

    @property
    def reveal_final_preview_sha256(self) -> str:
        return str(self._payload()["reveal_final_preview_sha256"])

    @property
    def selected_env_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._payload()["selected_env_ids"])

    @property
    def c05_source_sha256(self) -> str:
        return str(self._payload()["c05_source_sha256"])

    @property
    def profile_sha256(self) -> str:
        return str(self._payload()["profile_sha256"])

    @property
    def text_registry_sha256(self) -> str:
        return str(self._payload()["text_registry_sha256"])

    @property
    def dtype(self) -> str:
        return str(self._payload()["dtype"])

    @property
    def device(self) -> str:
        return str(self._payload()["device"])

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._payload_json).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        payload = self._payload()
        return {**payload, "canonical_sha256": self.canonical_sha256}


def _owned_install_receipt(value: object) -> dict[str, object]:
    if (
        type(value) is not LandingRevealInstallReceipt
        or value._token is not _INSTALL_AUTH_TOKEN
        or type(value._payload_json) is not bytes
        or type(value._auth_tag) is not bytes
    ):
        raise LandingOutcomeDeviceError(
            "install receipt must come from build_landing_reveal_install[_batch]"
        )
    expected_tag = hmac.new(
        _INSTALL_AUTH_KEY,
        value._payload_json,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(value._auth_tag, expected_tag):
        raise LandingOutcomeDeviceError("install receipt authentication differs")
    try:
        payload = json.loads(value._payload_json.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LandingOutcomeDeviceError("install receipt payload is invalid") from exc
    expected = {
        "schema_version",
        "kind",
        "num_envs",
        "dtype",
        "device",
        "r05_source_sha256",
        "r05_contract_sha256",
        "r05_final_source_pin_pending",
        "c05_source_sha256",
        "profile_sha256",
        "text_registry_sha256",
        "expected_reveal_final_preview_sha256",
        "reveal_final_preview_sha256",
        "selected_env_ids",
        "rows",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LandingOutcomeDeviceError("install receipt fields differ")
    if payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != INSTALL_RECEIPT_KIND:
        raise LandingOutcomeDeviceError("install receipt schema differs")
    if payload["r05_final_source_pin_pending"] is not R05_FINAL_SOURCE_PIN_PENDING:
        raise LandingOutcomeDeviceError("install receipt R05 final-pin status differs")
    _exact_int(payload["num_envs"], label="install receipt num_envs", minimum=1)
    for name in (
        "r05_source_sha256",
        "r05_contract_sha256",
        "c05_source_sha256",
        "profile_sha256",
        "text_registry_sha256",
        "expected_reveal_final_preview_sha256",
        "reveal_final_preview_sha256",
    ):
        _sha256_hex(payload[name], label=f"install receipt {name}")
    if (
        payload["expected_reveal_final_preview_sha256"]
        != payload["reveal_final_preview_sha256"]
    ):
        raise LandingOutcomeDeviceError(
            "install receipt expected R05 preview pin differs"
        )
    if type(payload["dtype"]) is not str or type(payload["device"]) is not str:
        raise LandingOutcomeDeviceError("install receipt dtype/device differs")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise LandingOutcomeDeviceError("install receipt rows differ")
    selected_env_ids = payload["selected_env_ids"]
    if not isinstance(selected_env_ids, list):
        raise LandingOutcomeDeviceError("install receipt selected env ids differ")
    expected_row_keys = {
        "kind",
        "env_id",
        "expected_committed_reveal_sha256",
        "committed_reveal_sha256",
        "reveal_final_preview_sha256",
        "outcome_key",
        "run_id_token_sha256",
        "carry_chain_id_token_sha256",
        "full_key_sha256",
        "full_key_receipt_sha256",
        "task_identity_sha256",
        "target_semantic_sha256",
        "target_xy_m",
        "flight_slot",
        "mailbox_slot",
        "ball_generation",
        "reveal_control_step",
        "selected_contact_deadline_control_step",
        "first_crossing_horizon_control_step",
        "row_receipt_sha256",
    }
    prior_env = -1
    for row in raw_rows:
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise LandingOutcomeDeviceError("install row receipt fields differ")
        if row["kind"] != INSTALL_ROW_RECEIPT_KIND:
            raise LandingOutcomeDeviceError("install row receipt kind differs")
        row_payload = {name: row[name] for name in row if name != "row_receipt_sha256"}
        declared = _sha256_hex(
            row["row_receipt_sha256"],
            label="row_receipt_sha256",
        )
        if _canonical_sha256(row_payload) != declared:
            raise LandingOutcomeDeviceError("install row receipt SHA differs")
        env_id = _exact_int(row["env_id"], label="install row env_id", minimum=0)
        if env_id <= prior_env or env_id >= payload["num_envs"]:
            raise LandingOutcomeDeviceError("install receipt env order/range differs")
        prior_env = env_id
        for name in (
            "expected_committed_reveal_sha256",
            "committed_reveal_sha256",
            "reveal_final_preview_sha256",
            "run_id_token_sha256",
            "carry_chain_id_token_sha256",
            "full_key_sha256",
            "full_key_receipt_sha256",
            "task_identity_sha256",
            "target_semantic_sha256",
        ):
            _sha256_hex(row[name], label=f"install row {name}")
        if row["expected_committed_reveal_sha256"] != row["committed_reveal_sha256"]:
            raise LandingOutcomeDeviceError("install row expected committed pin differs")
        if (
            row["reveal_final_preview_sha256"]
            != payload["reveal_final_preview_sha256"]
        ):
            raise LandingOutcomeDeviceError("install row R05 preview pin differs")
        try:
            outcome_key = _c05.LandingOutcomeShotKey.from_mapping(row["outcome_key"])
        except Exception as exc:
            raise LandingOutcomeDeviceError("install row C05 key seal differs") from exc
        if outcome_key.env_id != env_id or outcome_key.canonical_sha256 != row["full_key_sha256"]:
            raise LandingOutcomeDeviceError("install row C05 key identity differs")
        if (
            hashlib.sha256(outcome_key.run_id.encode("utf-8")).hexdigest()
            != row["run_id_token_sha256"]
            or hashlib.sha256(outcome_key.carry_chain_id.encode("utf-8")).hexdigest()
            != row["carry_chain_id_token_sha256"]
        ):
            raise LandingOutcomeDeviceError("install row text token differs")
        target_xy = row["target_xy_m"]
        if (
            not isinstance(target_xy, list)
            or len(target_xy) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in target_xy
            )
        ):
            raise LandingOutcomeDeviceError("install row target XY differs")
        full_key_receipt_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": FULL_KEY_RECEIPT_KIND,
            "committed_reveal_sha256": row["committed_reveal_sha256"],
            "reveal_final_preview_sha256": row[
                "reveal_final_preview_sha256"
            ],
            "r05_source_sha256": payload["r05_source_sha256"],
            "r05_contract_sha256": payload["r05_contract_sha256"],
            "c05_source_sha256": payload["c05_source_sha256"],
            "text_registry_sha256": payload["text_registry_sha256"],
            "full_key_sha256": row["full_key_sha256"],
            "outcome_key": row["outcome_key"],
        }
        if _canonical_sha256(full_key_receipt_payload) != row["full_key_receipt_sha256"]:
            raise LandingOutcomeDeviceError("install row full-key receipt differs")
        for name in (
            "flight_slot",
            "mailbox_slot",
            "ball_generation",
            "reveal_control_step",
            "selected_contact_deadline_control_step",
            "first_crossing_horizon_control_step",
        ):
            _exact_int(row[name], label=f"install row {name}", minimum=0)
        if (
            row["selected_contact_deadline_control_step"]
            < row["reveal_control_step"]
            or row["first_crossing_horizon_control_step"]
            < row["selected_contact_deadline_control_step"]
        ):
            raise LandingOutcomeDeviceError("install row cadence order differs")
    if selected_env_ids != [row["env_id"] for row in raw_rows]:
        raise LandingOutcomeDeviceError("install receipt selected env/row order differs")
    return payload


@dataclass(frozen=True)
class DeviceLandingOutcomeKey:
    """C05's complete fourteen-field key in fixed-width device form.

    ``run_id`` and ``carry_chain_id`` are canonical UTF-8 SHA tokens.  Their
    exact text is owned by the immutable registry bound by
    :class:`LandingOutcomeRuntimeBinding`.
    """

    env_id: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    birth_sha256: torch.Tensor
    sample_sha256: torch.Tensor
    task_sha256: torch.Tensor
    run_id: torch.Tensor
    carry_chain_id: torch.Tensor
    shot_index: torch.Tensor
    source_sha256: torch.Tensor
    config_sha256: torch.Tensor
    receipt_content_sha256: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "DeviceLandingOutcomeKey":
        if not isinstance(value, Mapping):
            raise LandingOutcomeDeviceError("task_key must be a mapping")
        expected = set(_KEY_FIELDS)
        actual = set(value)
        if actual != expected:
            raise LandingOutcomeDeviceError(
                "task_key fields differ: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )
        return cls(**{name: value[name] for name in _KEY_FIELDS})

    def to_mapping(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in _KEY_FIELDS}


@dataclass(frozen=True)
class PhysicsStampBatch:
    control_step: torch.Tensor
    physics_substep: torch.Tensor
    event_phase: torch.Tensor


@dataclass(frozen=True)
class LandingRevealInstallDevicePack:
    """Read-only device tensors minted together with an install receipt."""

    _values: Mapping[str, object]
    _token: object

    def _tensor_clone(self, name: str) -> torch.Tensor:
        values = _owned_install_device_pack(self)
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise LandingOutcomeDeviceError(f"install device pack {name} differs")
        return value.detach().clone()

    @property
    def mask(self) -> torch.Tensor:
        return self._tensor_clone("mask")

    @property
    def flight_slot(self) -> torch.Tensor:
        return self._tensor_clone("flight_slot")

    @property
    def mailbox_slot(self) -> torch.Tensor:
        return self._tensor_clone("mailbox_slot")

    @property
    def task_key(self) -> DeviceLandingOutcomeKey:
        values = _owned_install_device_pack(self)
        key = values["task_key"]
        if not isinstance(key, DeviceLandingOutcomeKey):
            raise LandingOutcomeDeviceError("install device pack task_key differs")
        return DeviceLandingOutcomeKey(
            **{
                name: getattr(key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        )

    @property
    def full_key_sha256(self) -> torch.Tensor:
        return self._tensor_clone("full_key_sha256")

    @property
    def full_key_receipt_sha256(self) -> torch.Tensor:
        return self._tensor_clone("full_key_receipt_sha256")

    @property
    def committed_reveal_sha256(self) -> torch.Tensor:
        return self._tensor_clone("committed_reveal_sha256")

    @property
    def install_receipt_sha256(self) -> torch.Tensor:
        return self._tensor_clone("install_receipt_sha256")

    @property
    def ball_generation(self) -> torch.Tensor:
        return self._tensor_clone("ball_generation")

    @property
    def task_identity_token(self) -> torch.Tensor:
        return self._tensor_clone("task_identity_token")

    @property
    def target_xy_m(self) -> torch.Tensor:
        return self._tensor_clone("target_xy_m")

    @property
    def reveal_control_step(self) -> torch.Tensor:
        return self._tensor_clone("reveal_control_step")

    @property
    def selected_contact_deadline_control_step(self) -> torch.Tensor:
        return self._tensor_clone("selected_contact_deadline_control_step")

    @property
    def first_crossing_horizon_control_step(self) -> torch.Tensor:
        return self._tensor_clone("first_crossing_horizon_control_step")


_INSTALL_DEVICE_PACK_FIELDS = (
    "mask",
    "flight_slot",
    "mailbox_slot",
    "task_key",
    "full_key_sha256",
    "full_key_receipt_sha256",
    "committed_reveal_sha256",
    "install_receipt_sha256",
    "ball_generation",
    "task_identity_token",
    "target_xy_m",
    "reveal_control_step",
    "selected_contact_deadline_control_step",
    "first_crossing_horizon_control_step",
)


def _owned_install_device_pack(
    value: object,
) -> Mapping[str, object]:
    if (
        type(value) is not LandingRevealInstallDevicePack
        or value._token is not _INSTALL_PACK_TOKEN
        or not isinstance(value._values, Mapping)
        or set(value._values) != set(_INSTALL_DEVICE_PACK_FIELDS)
    ):
        raise LandingOutcomeDeviceError(
            "install device pack must come from build_landing_reveal_install[_batch]"
        )
    return value._values


@dataclass(frozen=True)
class LandingRevealInstall:
    """Factory-minted pair of sealed host receipt and private device pack."""

    _receipt: LandingRevealInstallReceipt
    _device_pack: LandingRevealInstallDevicePack
    _token: object

    @property
    def receipt(self) -> LandingRevealInstallReceipt:
        _owned_landing_reveal_install(self)
        return self._receipt

    @property
    def device_pack(self) -> LandingRevealInstallDevicePack:
        _owned_landing_reveal_install(self)
        return self._device_pack


def _owned_landing_reveal_install(
    value: object,
) -> tuple[dict[str, object], Mapping[str, object]]:
    if type(value) is not LandingRevealInstall or value._token is not _INSTALL_AUTH_TOKEN:
        raise LandingOutcomeDeviceError(
            "request must come from build_landing_reveal_install[_batch]"
        )
    receipt = _owned_install_receipt(value._receipt)
    pack = _owned_install_device_pack(value._device_pack)
    return receipt, pack


@dataclass(frozen=True)
class LandingRevealPreparedInstallReceipt:
    """Host-sealed preview/authority/after-image intent for one R06 child."""

    _payload_json: bytes
    _auth_tag: bytes
    _token: object

    def _payload(self) -> dict[str, object]:
        return _owned_prepared_install_receipt(self)

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._payload_json).hexdigest()

    @property
    def reveal_final_preview_sha256(self) -> str:
        return str(self._payload()["reveal_final_preview_sha256"])

    @property
    def selected_env_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._payload()["selected_env_ids"])

    @property
    def owner_checkpoint_before_sha256(self) -> str:
        return str(self._payload()["r05_owner_checkpoint_before_sha256"])

    @property
    def all_owner_install_root_sha256(self) -> str:
        return str(self._payload()["all_owner_install_root_sha256"])

    def to_mapping(self) -> dict[str, object]:
        return {**self._payload(), "canonical_sha256": self.canonical_sha256}


_PREPARED_INSTALL_RECEIPT_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "num_envs",
        "dtype",
        "device",
        "r05_source_sha256",
        "r05_contract_sha256",
        "r05_final_source_pin_pending",
        "c05_source_sha256",
        "profile_sha256",
        "text_registry_sha256",
        "capacity_authority",
        "capacity_authority_sha256",
        "payment_authority_sha256",
        "reveal_final_preview_sha256",
        "expected_reveal_final_preview_sha256",
        "r05_owner_checkpoint_before_sha256",
        "prepared_batch_sha256",
        "sampler_checkpoint_before_commit_sha256",
        "sampler_checkpoint_after_commit_sha256",
        "untouched_rows_before_sha256",
        "untouched_rows_after_sha256",
        "all_owner_install_root_sha256",
        "selected_env_ids",
        "mailbox_allocation_policy",
        "mailbox_allocation_sentinel",
        "first_crossing_horizon_control_steps",
        "preview_rows",
        "legacy_after_image_install_template_receipt_sha256",
        "after_image_authority_sha256",
    )
)


def _owned_prepared_install_receipt(value: object) -> dict[str, object]:
    if (
        type(value) is not LandingRevealPreparedInstallReceipt
        or value._token is not _PREPARED_INSTALL_AUTH_TOKEN
        or type(value._payload_json) is not bytes
        or type(value._auth_tag) is not bytes
    ):
        raise LandingOutcomeDeviceError(
            "prepared install receipt must come from preview preparation"
        )
    expected_tag = hmac.new(
        _PREPARED_INSTALL_AUTH_KEY,
        value._payload_json,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(value._auth_tag, expected_tag):
        raise LandingOutcomeDeviceError(
            "prepared install receipt consistency seal differs"
        )
    try:
        payload = json.loads(value._payload_json.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LandingOutcomeDeviceError(
            "prepared install receipt payload is invalid"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _PREPARED_INSTALL_RECEIPT_FIELDS:
        raise LandingOutcomeDeviceError("prepared install receipt fields differ")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["kind"] != PREPARED_INSTALL_RECEIPT_KIND
        or payload["r05_final_source_pin_pending"] is not R05_FINAL_SOURCE_PIN_PENDING
    ):
        raise LandingOutcomeDeviceError("prepared install receipt schema differs")
    for name in (
        "r05_source_sha256",
        "r05_contract_sha256",
        "c05_source_sha256",
        "profile_sha256",
        "text_registry_sha256",
        "capacity_authority_sha256",
        "payment_authority_sha256",
        "reveal_final_preview_sha256",
        "expected_reveal_final_preview_sha256",
        "r05_owner_checkpoint_before_sha256",
        "prepared_batch_sha256",
        "sampler_checkpoint_before_commit_sha256",
        "sampler_checkpoint_after_commit_sha256",
        "untouched_rows_before_sha256",
        "untouched_rows_after_sha256",
        "all_owner_install_root_sha256",
        "legacy_after_image_install_template_receipt_sha256",
        "after_image_authority_sha256",
    ):
        _sha256_hex(payload[name], label=f"prepared_install.{name}")
    selected = payload["selected_env_ids"]
    horizons = payload["first_crossing_horizon_control_steps"]
    rows = payload["preview_rows"]
    if (
        type(selected) is not list
        or selected != sorted(set(selected))
        or any(type(value) is not int or value < 0 for value in selected)
        or payload["mailbox_allocation_policy"] != MAILBOX_ALLOCATION_POLICY
        or type(payload["mailbox_allocation_sentinel"]) is not int
        or payload["mailbox_allocation_sentinel"] < 1
        or type(horizons) is not list
        or type(rows) is not list
        or not selected
        or not (len(selected) == len(horizons) == len(rows))
    ):
        raise LandingOutcomeDeviceError("prepared install selected rows differ")
    if payload["reveal_final_preview_sha256"] != payload[
        "expected_reveal_final_preview_sha256"
    ]:
        raise LandingOutcomeDeviceError("prepared install preview external root differs")
    return payload


@dataclass(frozen=True)
class LandingRevealPrepareAttempt:
    """Opaque boundary-free R06 after-image and retained-token authority."""

    _receipt: LandingRevealPreparedInstallReceipt
    _preview_token: object
    _legacy_install: LandingRevealInstall
    _allocated_mailbox_slot: torch.Tensor
    _tensor_swaps: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    _host_after_state: Mapping[str, object]
    _precomputed_result: DeviceMutationResult
    _device_owner_mutation_version: torch.Tensor
    _owner_identity: object
    _token: object

    @property
    def intent_receipt(self) -> LandingRevealPreparedInstallReceipt:
        _owned_prepare_attempt(self)
        return self._receipt

    @property
    def allocated_mailbox_slot(self) -> torch.Tensor:
        """Return a clone of the owner-selected device slot per environment."""

        return self._allocated_mailbox_slot.detach().clone()

    @property
    def canonical_sha256(self) -> str:
        return self.intent_receipt.canonical_sha256


def _owned_prepare_attempt(value: object) -> LandingRevealPrepareAttempt:
    if (
        type(value) is not LandingRevealPrepareAttempt
        or value._token is not _PREPARED_INSTALL_AUTH_TOKEN
        or type(value._preview_token) is not _r05.RevealFinalPreviewBatch
        or type(value._tensor_swaps) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or not all(isinstance(tensor, torch.Tensor) for tensor in pair)
            for pair in value._tensor_swaps
        )
        or not isinstance(value._host_after_state, Mapping)
        or not isinstance(value._precomputed_result, DeviceMutationResult)
        or not isinstance(value._allocated_mailbox_slot, torch.Tensor)
        or value._allocated_mailbox_slot.ndim != 1
        or value._allocated_mailbox_slot.dtype != torch.int64
        or not isinstance(value._device_owner_mutation_version, torch.Tensor)
        or tuple(value._device_owner_mutation_version.shape) not in ((), (1,))
        or value._device_owner_mutation_version.dtype != torch.int64
    ):
        raise LandingOutcomeDeviceError(
            "prepare attempt must come from prepare_from_reveal_final_preview"
        )
    _owned_prepared_install_receipt(value._receipt)
    _legacy_receipt, legacy_pack = _owned_landing_reveal_install(
        value._legacy_install
    )
    legacy_mask = legacy_pack.get("mask")
    if (
        not isinstance(legacy_mask, torch.Tensor)
        or tuple(value._allocated_mailbox_slot.shape)
        != tuple(legacy_mask.shape)
        or value._allocated_mailbox_slot.device != legacy_mask.device
    ):
        raise LandingOutcomeDeviceError(
            "prepare attempt mailbox allocation tensor differs"
        )
    return value


@dataclass(frozen=True)
class LandingRevealCommitReceipt:
    """Prevalidated ACCEPT receipt returned by the pure-copy publish."""

    schema_version: int
    kind: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_sha256: str
    r05_terminal_claim_sha256: str
    r05_terminal_boundary_authority_sha256: str
    r05_terminal_boundary_projection_sha256: str
    r05_terminal_content_pin_sha256: str
    r05_terminal_schema_version: int
    r05_terminal_content_bytes_base64: str
    r05_terminal_content_byte_length: int
    r05_terminal_content_bytes_sha256: str
    expected_r05_terminal_kind: str
    expected_r05_terminal_sha256: str
    r06_child_token_root_sha256: str
    reveal_final_preview_sha256: str
    selected_env_ids: tuple[int, ...]
    owner_mutation_version_before: int
    owner_mutation_version_after: int
    installed_count: int
    runtime_integrated: bool
    launch_authorized: bool

    @property
    def canonical_sha256(self) -> str:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = list(value) if isinstance(value, tuple) else value
        return _canonical_sha256(payload)


@dataclass(frozen=True)
class LandingRevealArmedInstall:
    """Opaque ACCEPT handle bound to one exact owner-issued global receipt."""

    _attempt: LandingRevealPrepareAttempt
    _r05_terminal_claim: object
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class LandingRevealCensoredInstall:
    """Opaque CENSOR handle whose chronology after-image is still private."""

    _attempt: LandingRevealPrepareAttempt
    _r05_terminal_claim: object
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class LandingRevealCensorReceipt:
    """Typed zero-install chronology receipt for one global CENSOR."""

    schema_version: int
    kind: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_sha256: str
    r05_terminal_claim_sha256: str
    r05_terminal_boundary_authority_sha256: str
    r05_terminal_boundary_projection_sha256: str
    r05_terminal_content_pin_sha256: str
    r05_terminal_schema_version: int
    r05_terminal_content_bytes_base64: str
    r05_terminal_content_byte_length: int
    r05_terminal_content_bytes_sha256: str
    expected_r05_terminal_kind: str
    expected_r05_terminal_sha256: str
    r06_child_token_root_sha256: str
    reveal_final_preview_sha256: str
    selected_env_ids: tuple[int, ...]
    owner_mutation_version_before: int
    owner_mutation_version_after: int
    installed_count: int
    censored_count: int
    policy_opportunity_created: bool
    runtime_integrated: bool
    launch_authorized: bool

    @property
    def canonical_sha256(self) -> str:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = list(value) if isinstance(value, tuple) else value
        return _canonical_sha256(payload)


@dataclass(frozen=True, eq=False)
class R06ChildTerminalToken:
    """Opaque proof that R06 copied one prevalidated child after-image."""

    _attempt: LandingRevealPrepareAttempt
    _child_receipt: LandingRevealCommitReceipt | LandingRevealCensorReceipt
    _r05_terminal_claim: object
    _decision: str
    _owner_identity: object
    _token: object


@dataclass
class _ActiveRevealPrepareLease:
    attempt: LandingRevealPrepareAttempt
    boundary_row: _reveal_boundary.ActionBallFullMdpRevealBoundaryDeviceRow
    censor_tensor_swaps: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    censor_host_after_state: Mapping[str, object]
    armed_install: LandingRevealArmedInstall | None = None
    censored_install: LandingRevealCensoredInstall | None = None
    commit_receipt: LandingRevealCommitReceipt | None = None
    censor_receipt: LandingRevealCensorReceipt | None = None
    r05_terminal_claim: object | None = None
    r05_terminal_claim_sha256: str | None = None
    r05_terminal_boundary_authority_sha256: str | None = None
    r05_terminal_boundary_projection_sha256: str | None = None
    r05_terminal_content_pin: object | None = None
    r05_terminal_content_pin_sha256: str | None = None
    r05_terminal_content_bytes_sha256: str | None = None
    r05_terminal_decision: str | None = None
    r05_terminal_kind: str | None = None
    r05_terminal_sha256: str | None = None
    global_boundary_receipt_sha256: str | None = None
    global_boundary_packet_sha256: str | None = None
    terminal_token: R06ChildTerminalToken | None = None
    published_terminal_token: R06ChildTerminalToken | None = None


def _r05_snapshot_sha256(slots: Sequence[_r05.BallSlotSnapshot]) -> str:
    return _r05.canonical_sha256([slot.to_mapping() for slot in slots])


def _validate_r05_slot_rows(
    slots: Sequence[_r05.BallSlotSnapshot],
    *,
    label: str,
) -> tuple[_r05.BallSlotSnapshot, ...]:
    rows = tuple(slots)
    if not rows or tuple(row.slot_index for row in rows) != tuple(range(len(rows))):
        raise LandingOutcomeDeviceError(f"{label} must be complete and index ordered")
    for name in ("owner_key_sha256", "inbound_ball_sha256"):
        identities = [
            getattr(row, name)
            for row in rows
            if getattr(row, name) is not None
        ]
        if len(identities) != len(set(identities)):
            raise LandingOutcomeDeviceError(f"{label} contains duplicate {name}")
    return rows


def _reparse_r05_committed_batch(
    value: object,
    *,
    expected_committed_reveal_batch_sha256: str,
) -> _r05.CommittedRevealBatch:
    expected = _sha256_hex(
        expected_committed_reveal_batch_sha256,
        label="expected_committed_reveal_batch_sha256",
    )
    if type(value) is _r05.CommittedRevealBatch:
        mapping: object = value.to_mapping()
    elif isinstance(value, Mapping):
        mapping = value
    else:
        raise LandingOutcomeDeviceError(
            "R05 authority must be one sealed CommittedRevealBatch or mapping"
        )
    try:
        batch = _r05.CommittedRevealBatch.from_mapping(mapping)
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "R05 committed reveal batch verification failed"
        ) from exc
    if batch.canonical_sha256 != expected:
        raise LandingOutcomeDeviceError("committed reveal batch external SHA pin differs")
    return batch


def _synthesize_r05_committed_rows_from_preview(
    preview: _r05.RevealFinalPreviewBatch,
) -> tuple[_r05.CommittedReveal, ...]:
    """Materialize deterministic row after-images without inventing a marker."""

    rows = tuple(
        _r05.CommittedReveal(
            integration_status=_r05.INTEGRATION_STATUS,
            phase=_r05.COMMITTED,
            runtime_wiring_connected=False,
            identity_committed=True,
            policy_opportunity_created=True,
            prepared_reveal=row.prepared_reveal,
            reveal_facts=row.reveal_facts,
            ball_slot_plan=row.ball_slot_plan,
            playback_release_requested=row.reveal_facts.ready_at_reveal,
        )
        for row in preview.reveal_final_rows
    )
    return tuple(
        _r05.CommittedReveal.from_mapping(row.to_mapping()) for row in rows
    )


def _validate_r05_install_row(
    committed_value: object,
    *,
    expected_committed_reveal_sha256: str,
    reveal_final_preview_sha256: str,
    profile: LandingPlacementProfile,
    registry: LandingOutcomeTextRegistry,
    mailbox_slot: int,
    first_crossing_horizon_control_step: int,
    num_envs: int,
    r05_source_sha256: str,
    r05_contract_sha256: str,
    c05_source_sha256: str,
) -> dict[str, object]:
    expected_committed = _sha256_hex(
        expected_committed_reveal_sha256,
        label="expected_committed_reveal_sha256",
    )
    try:
        if isinstance(committed_value, _r05.CommittedReveal):
            committed_mapping = committed_value.to_mapping()
        elif isinstance(committed_value, Mapping):
            committed_mapping = committed_value
        else:
            raise LandingOutcomeDeviceError(
                "committed_reveal must be CommittedReveal or sealed mapping"
            )
        # Even a typed object is serialized and reparsed so its nested values
        # cannot bypass R05/C05 canonical seals through mutable aliases.
        committed = _r05.CommittedReveal.from_mapping(committed_mapping)
    except LandingOutcomeDeviceError:
        raise
    except Exception as exc:
        raise LandingOutcomeDeviceError("R05 committed reveal verification failed") from exc
    if committed.canonical_sha256 != expected_committed:
        raise LandingOutcomeDeviceError("committed reveal external SHA pin differs")

    prepared = committed.prepared_reveal
    request = prepared.request
    facts = committed.reveal_facts
    selection = prepared.selection
    target_receipt = prepared.runtime_target_receipt
    ref = prepared.selected_task_ref

    if request.env_id >= num_envs:
        raise LandingOutcomeDeviceError("committed reveal env_id is out of range")
    selected_rows = tuple(
        row
        for row in prepared.candidates
        if row.cell_id == selection.cell_id and row.construction_feasible
    )
    if len(selected_rows) != 1:
        raise LandingOutcomeDeviceError(
            "R05 selected cell must name exactly one feasible candidate"
        )
    selected = selected_rows[0]
    if (
        selected.task_ref is None
        or selected.task_ref != ref
        or tuple(selection.target) != selected.target_xy_m
        or selection.semantic_sha256 != selected.target_semantic_sha256
    ):
        raise LandingOutcomeDeviceError("R05 selected candidate binding differs")
    candidate_cell_ids = tuple(row.cell_id for row in prepared.candidates)
    if len(candidate_cell_ids) != len(set(candidate_cell_ids)):
        raise LandingOutcomeDeviceError("R05 candidate cell identities are duplicated")

    expected_ref = {
        "env_id": request.env_id,
        "reset_generation": request.reset_generation,
        "swing_generation": request.runtime_swing_generation,
        "action_uid": request.action_uid,
        "action_slot": request.action_slot,
        "birth_sha256": request.birth_sha256,
    }
    if any(getattr(ref, name) != value for name, value in expected_ref.items()):
        raise LandingOutcomeDeviceError("R05 selected task/request binding differs")
    if (
        selection.env_id != request.env_id
        or selection.target_generation != request.sampler_generation
        or selection.frame_id != profile.frame_id
        or selection.frame_binding_sha256 != profile.frame_binding_sha256
        or len(selection.target) != 2
    ):
        raise LandingOutcomeDeviceError("R05 selection/profile/frame binding differs")
    runtime_target_xy = target_receipt.runtime_target_xy_m
    if (
        target_receipt.profile_sha256 != selection.profile_sha256
        or target_receipt.selection_authority_sha256
        != request.selection_authority_sha256
        or target_receipt.runtime_dtype != selection.runtime_dtype
        or target_receipt.target_generation != request.runtime_swing_generation
        or target_receipt.task_ref_sha256 != ref.canonical_sha256
        or (
            target_receipt.requested_target_x_m,
            target_receipt.requested_target_y_m,
        )
        != tuple(selection.target)
    ):
        raise LandingOutcomeDeviceError("R05 runtime target receipt binding differs")
    target_x, target_y = runtime_target_xy
    if not (
        profile.opponent_table_x_min_m
        <= target_x
        <= profile.opponent_table_x_max_m
        and profile.table_y_min_m <= target_y <= profile.table_y_max_m
    ):
        raise LandingOutcomeDeviceError("R05 target is outside the C04 profile")

    try:
        outcome_key = _c05.LandingOutcomeShotKey.from_mapping(
            prepared.outcome_key.to_mapping()
        )
    except Exception as exc:
        raise LandingOutcomeDeviceError("C05 outcome key verification failed") from exc
    expected_outcome = {
        **ref.runtime_dict(),
        "run_id": request.run_id,
        "carry_chain_id": request.carry_chain_id,
        "shot_index": request.outcome_shot_index,
        "source_sha256": request.source_sha256,
        "config_sha256": request.config_sha256,
        "receipt_content_sha256": selected.receipt_content_sha256,
    }
    if outcome_key.full_key_dict() != expected_outcome:
        raise LandingOutcomeDeviceError(
            "C05 outcome key front-eight/suffix binding differs"
        )

    expected_facts = {
        "env_id": request.env_id,
        "reset_generation": request.reset_generation,
        "scheduled_ordinal": request.scheduled_ordinal,
        "runtime_swing_generation": request.runtime_swing_generation,
        "sampler_generation": request.sampler_generation,
        "outcome_shot_index": request.outcome_shot_index,
        "schedule_sha256": request.schedule_sha256,
        "reveal_step": request.scheduled_reveal_step,
        "deadline_step": request.scheduled_deadline_step,
    }
    if any(getattr(facts, name) != value for name, value in expected_facts.items()):
        raise LandingOutcomeDeviceError("R05 reveal facts/request binding differs")

    prepared_slots = _validate_r05_slot_rows(
        request.ball_slots,
        label="R05 prepared ball slots",
    )
    reveal_slots = _validate_r05_slot_rows(
        facts.ball_slots,
        label="R05 reveal ball slots",
    )
    if len(prepared_slots) != len(reveal_slots):
        raise LandingOutcomeDeviceError("R05 ball-slot capacity changed at reveal")
    lifecycle_rank = {
        _r05.BALL_INBOUND: 0,
        _r05.BALL_OPEN: 1,
        _r05.BALL_SETTLED_UNPAID: 2,
        _r05.BALL_PAID: 3,
        _r05.BALL_CLOSED: 4,
    }
    for before, after in zip(prepared_slots, reveal_slots):
        if before.lifecycle_state == _r05.BALL_EMPTY:
            if after.lifecycle_state != _r05.BALL_EMPTY:
                raise LandingOutcomeDeviceError(
                    "R05 prepared empty slot acquired an owner before reveal"
                )
            continue
        if after.lifecycle_state == _r05.BALL_EMPTY:
            raise LandingOutcomeDeviceError("R05 prepared owner disappeared at reveal")
        if any(
            getattr(after, name) != getattr(before, name)
            for name in (
                "owner_key_sha256",
                "ball_generation",
                "inbound_ball_sha256",
            )
        ):
            raise LandingOutcomeDeviceError("R05 prior ball identity changed at reveal")
        if (
            lifecycle_rank[after.lifecycle_state]
            < lifecycle_rank[before.lifecycle_state]
            or (before.physical_retired and not after.physical_retired)
        ):
            raise LandingOutcomeDeviceError("R05 prior ball lifecycle regressed")

    if (
        selected.inbound_ball_generation != request.runtime_swing_generation
        or selected.inbound_ball_generation is None
        or selected.inbound_ball_sha256 is None
        or selected.installed_ball_dynamic_state_sha256 is None
    ):
        raise LandingOutcomeDeviceError("R05 selected inbound ball binding differs")
    reservation = prepared.prepared_ball_slot_reservation
    prepared_reusable = tuple(row.slot_index for row in prepared_slots if row.reusable)
    prepared_owners = tuple(
        row.owner_key_sha256
        for row in prepared_slots
        if row.owner_key_sha256 is not None
    )
    if (
        reservation.capacity != len(prepared_slots)
        or reservation.snapshot_sha256 != _r05_snapshot_sha256(prepared_slots)
        or reservation.previous_slot_index != request.previous_ball_slot_index
        or reservation.reusable_slot_indices != prepared_reusable
        or reservation.capacity_available_at_prepare != bool(prepared_reusable)
        or reservation.observed_prior_owner_key_sha256 != prepared_owners
        or reservation.new_ball_generation != selected.inbound_ball_generation
        or reservation.new_inbound_ball_sha256 != selected.inbound_ball_sha256
        or reservation.new_ball_dynamic_state_sha256
        != selected.installed_ball_dynamic_state_sha256
    ):
        raise LandingOutcomeDeviceError("R05 prepared ball reservation differs")

    plan = committed.ball_slot_plan
    previous_slot = request.previous_ball_slot_index
    if previous_slot is not None and reveal_slots[previous_slot].reusable:
        expected_selected_slot = previous_slot
    else:
        reveal_reusable = tuple(row.slot_index for row in reveal_slots if row.reusable)
        if not reveal_reusable:
            raise LandingOutcomeDeviceError("R05 reveal has no reusable flight slot")
        expected_selected_slot = min(reveal_reusable)
    selected_slot_row = reveal_slots[expected_selected_slot]
    preserved = tuple(
        row.owner_key_sha256
        for row in reveal_slots
        if row.slot_index != expected_selected_slot
        and row.lifecycle_state != _r05.BALL_EMPTY
        and row.owner_key_sha256 is not None
    )
    reused_owner = (
        None
        if selected_slot_row.lifecycle_state == _r05.BALL_EMPTY
        else selected_slot_row.owner_key_sha256
    )
    if (
        not selected_slot_row.reusable
        or plan.capacity != len(reveal_slots)
        or plan.snapshot_sha256 != _r05_snapshot_sha256(reveal_slots)
        or plan.selected_slot_index != expected_selected_slot
        or plan.previous_slot_index != previous_slot
        or plan.reused_previous_slot
        != (previous_slot is not None and expected_selected_slot == previous_slot)
        or plan.preserved_live_owner_key_sha256 != preserved
        or plan.new_ball_generation != selected.inbound_ball_generation
        or plan.new_inbound_ball_sha256 != selected.inbound_ball_sha256
        or plan.new_ball_dynamic_state_sha256
        != selected.installed_ball_dynamic_state_sha256
        or plan.physical_ball_install_payload_sha256
        != selected.physical_ball_install_payload_sha256
        or plan.reused_retired_owner_key_sha256 != reused_owner
    ):
        raise LandingOutcomeDeviceError("R05 final ball-slot plan differs")

    run_token_sha = registry.token_sha256(
        namespace="run_id",
        text=outcome_key.run_id,
    )
    carry_token_sha = registry.token_sha256(
        namespace="carry_chain_id",
        text=outcome_key.carry_chain_id,
    )
    task_identity = LandingPlacementTaskIdentity(
        frame_id=profile.frame_id,
        frame_binding_sha256=profile.frame_binding_sha256,
        profile_sha256=profile.canonical_sha256,
        task_receipt_sha256=ref.task_sha256,
        semantic_binding_sha256=selection.semantic_sha256,
        instance_binding_sha256=committed.canonical_sha256,
        target_x_m=target_x,
        target_y_m=target_y,
    )
    full_key_sha = outcome_key.canonical_sha256
    full_key_receipt_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": FULL_KEY_RECEIPT_KIND,
        "committed_reveal_sha256": committed.canonical_sha256,
        "reveal_final_preview_sha256": reveal_final_preview_sha256,
        "r05_source_sha256": r05_source_sha256,
        "r05_contract_sha256": r05_contract_sha256,
        "c05_source_sha256": c05_source_sha256,
        "text_registry_sha256": registry.canonical_sha256,
        "full_key_sha256": full_key_sha,
        "outcome_key": outcome_key.to_mapping(),
    }
    full_key_receipt_sha = _canonical_sha256(full_key_receipt_payload)
    return {
        "env_id": request.env_id,
        "expected_committed_reveal_sha256": expected_committed,
        "committed_reveal_sha256": committed.canonical_sha256,
        "reveal_final_preview_sha256": reveal_final_preview_sha256,
        "outcome_key": outcome_key,
        "run_token_sha256": run_token_sha,
        "carry_token_sha256": carry_token_sha,
        "full_key_sha256": full_key_sha,
        "full_key_receipt_sha256": full_key_receipt_sha,
        "task_identity_sha256": task_identity.canonical_sha256,
        "target_semantic_sha256": selection.semantic_sha256,
        "target_xy_m": (target_x, target_y),
        "flight_slot": plan.selected_slot_index,
        "mailbox_slot": mailbox_slot,
        "ball_generation": plan.new_ball_generation,
        "reveal_control_step": facts.reveal_step,
        "selected_contact_deadline_control_step": facts.deadline_step,
        "first_crossing_horizon_control_step": first_crossing_horizon_control_step,
    }


def _build_landing_reveal_install_batch_from_preview(
    reveal_final_preview: _r05.RevealFinalPreviewBatch,
    *,
    expected_reveal_final_preview_sha256: str,
    expected_committed_reveal_sha256: tuple[str, ...],
    expected_r05_source_sha256: str,
    expected_r05_contract_sha256: str,
    expected_c05_source_sha256: str,
    profile: LandingPlacementProfile,
    expected_profile_sha256: str,
    text_registry: LandingOutcomeTextRegistry | Mapping[str, object],
    expected_text_registry_sha256: str,
    mailbox_slots: tuple[int, ...],
    first_crossing_horizon_control_steps: tuple[int, ...],
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> LandingRevealInstall:
    """Build from an already owner-validated active R05 preview.

    This private helper has one production caller, which first asks the bound
    R05 owner to validate the exact active object identity.  Re-encoding and
    reparsing the same K rows here would only re-check the same writer's bytes;
    the global boundary still binds the retained canonical root.
    """

    preview = reveal_final_preview
    expected_preview = _sha256_hex(
        expected_reveal_final_preview_sha256,
        label="expected_reveal_final_preview_sha256",
    )
    if (
        type(preview) is not _r05.RevealFinalPreviewBatch
        or preview.canonical_sha256 != expected_preview
    ):
        raise LandingOutcomeDeviceError(
            "owner-validated R05 reveal-final preview root differs"
        )
    committed_reveals = _synthesize_r05_committed_rows_from_preview(preview)
    preview_sha = preview.canonical_sha256
    if not committed_reveals:
        raise LandingOutcomeDeviceError("committed reveal batch must be non-empty")
    for name, value in (
        ("expected_committed_reveal_sha256", expected_committed_reveal_sha256),
        ("mailbox_slots", mailbox_slots),
        (
            "first_crossing_horizon_control_steps",
            first_crossing_horizon_control_steps,
        ),
    ):
        if type(value) is not tuple or len(value) != len(committed_reveals):
            raise LandingOutcomeDeviceError(f"{name} must match committed_reveals")
    clean_num_envs = _exact_int(num_envs, label="num_envs", minimum=1)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise LandingOutcomeDeviceError("dtype must be an explicit floating torch dtype")
    clean_device = torch.device(device)
    if not isinstance(profile, LandingPlacementProfile):
        raise LandingOutcomeDeviceError("profile must be LandingPlacementProfile")
    expected_profile = _sha256_hex(
        expected_profile_sha256,
        label="expected_profile_sha256",
    )
    if profile.canonical_sha256 != expected_profile:
        raise LandingOutcomeDeviceError("C04 profile external pin differs")
    r05_source = _sha256_hex(
        expected_r05_source_sha256,
        label="expected_r05_source_sha256",
    )
    r05_contract = _sha256_hex(
        expected_r05_contract_sha256,
        label="expected_r05_contract_sha256",
    )
    c05_source = _sha256_hex(
        expected_c05_source_sha256,
        label="expected_c05_source_sha256",
    )
    expected_r05_source = (
        R05_RUNTIME_TRANSACTION_SOURCE_SHA256
        or R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256
    )
    if r05_source != expected_r05_source:
        raise LandingOutcomeDeviceError("R05 source external pin differs")
    if r05_contract != R05_RUNTIME_TRANSACTION_CONTRACT_SHA256:
        raise LandingOutcomeDeviceError("R05 contract external pin differs")
    if c05_source != C05_LANDING_OUTCOME_SOURCE_SHA256:
        raise LandingOutcomeDeviceError("C05 source external pin differs")
    expected_registry = _sha256_hex(
        expected_text_registry_sha256,
        label="expected_text_registry_sha256",
    )
    registry_mapping = (
        text_registry.to_mapping()
        if isinstance(text_registry, LandingOutcomeTextRegistry)
        else text_registry
    )
    registry = LandingOutcomeTextRegistry.from_mapping(
        registry_mapping,
        expected_registry_sha256=expected_registry,
    )

    # Every host row is validated before any device tensor is allocated.  A
    # single bad expected SHA or nested binding aborts the entire K-env pack.
    rows: list[dict[str, object]] = []
    for index, committed in enumerate(committed_reveals):
        mailbox_slot = _exact_int(
            mailbox_slots[index],
            label=f"mailbox_slots[{index}]",
            minimum=0,
        )
        horizon = _exact_int(
            first_crossing_horizon_control_steps[index],
            label=f"first_crossing_horizon_control_steps[{index}]",
            minimum=0,
        )
        row = _validate_r05_install_row(
            committed,
            expected_committed_reveal_sha256=expected_committed_reveal_sha256[index],
            reveal_final_preview_sha256=preview_sha,
            profile=profile,
            registry=registry,
            mailbox_slot=mailbox_slot,
            first_crossing_horizon_control_step=horizon,
            num_envs=clean_num_envs,
            r05_source_sha256=r05_source,
            r05_contract_sha256=r05_contract,
            c05_source_sha256=c05_source,
        )
        if horizon < row["selected_contact_deadline_control_step"]:
            raise LandingOutcomeDeviceError("crossing horizon precedes contact deadline")
        rows.append(row)
    env_ids = tuple(int(row["env_id"]) for row in rows)
    if env_ids != tuple(sorted(set(env_ids))):
        raise LandingOutcomeDeviceError(
            "committed reveal env_id values must be strictly increasing and unique"
        )
    if env_ids != preview.selected_env_ids:
        raise LandingOutcomeDeviceError("R05 preview selected env binding differs")

    receipt_rows: list[dict[str, object]] = []
    for row in rows:
        row_payload = {
            "kind": INSTALL_ROW_RECEIPT_KIND,
            "env_id": row["env_id"],
            "expected_committed_reveal_sha256": row[
                "expected_committed_reveal_sha256"
            ],
            "committed_reveal_sha256": row["committed_reveal_sha256"],
            "reveal_final_preview_sha256": row[
                "reveal_final_preview_sha256"
            ],
            "outcome_key": row["outcome_key"].to_mapping(),
            "run_id_token_sha256": row["run_token_sha256"],
            "carry_chain_id_token_sha256": row["carry_token_sha256"],
            "full_key_sha256": row["full_key_sha256"],
            "full_key_receipt_sha256": row["full_key_receipt_sha256"],
            "task_identity_sha256": row["task_identity_sha256"],
            "target_semantic_sha256": row["target_semantic_sha256"],
            "target_xy_m": list(row["target_xy_m"]),
            "flight_slot": row["flight_slot"],
            "mailbox_slot": row["mailbox_slot"],
            "ball_generation": row["ball_generation"],
            "reveal_control_step": row["reveal_control_step"],
            "selected_contact_deadline_control_step": row[
                "selected_contact_deadline_control_step"
            ],
            "first_crossing_horizon_control_step": row[
                "first_crossing_horizon_control_step"
            ],
        }
        receipt_rows.append(
            {**row_payload, "row_receipt_sha256": _canonical_sha256(row_payload)}
        )
    receipt_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": INSTALL_RECEIPT_KIND,
        "num_envs": clean_num_envs,
        "dtype": str(dtype),
        "device": str(clean_device),
        "r05_source_sha256": r05_source,
        "r05_contract_sha256": r05_contract,
        "r05_final_source_pin_pending": R05_FINAL_SOURCE_PIN_PENDING,
        "c05_source_sha256": c05_source,
        "profile_sha256": expected_profile,
        "text_registry_sha256": registry.canonical_sha256,
        "expected_reveal_final_preview_sha256": (
            expected_reveal_final_preview_sha256
        ),
        "reveal_final_preview_sha256": preview_sha,
        "selected_env_ids": list(preview.selected_env_ids),
        "rows": receipt_rows,
    }
    receipt_json = json.dumps(
        receipt_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    receipt = LandingRevealInstallReceipt(
        _payload_json=receipt_json,
        _auth_tag=hmac.new(
            _INSTALL_AUTH_KEY,
            receipt_json,
            hashlib.sha256,
        ).digest(),
        _token=_INSTALL_AUTH_TOKEN,
    )

    mask_values = [False] * clean_num_envs
    flight_values = [-1] * clean_num_envs
    mailbox_values = [-1] * clean_num_envs
    generation_values = [-1] * clean_num_envs
    reveal_values = [-1] * clean_num_envs
    deadline_values = [-1] * clean_num_envs
    horizon_values = [-1] * clean_num_envs
    target_values = [(0.0, 0.0)] * clean_num_envs
    int_key_values = {
        name: ([-1] * clean_num_envs if name == "env_id" else [0] * clean_num_envs)
        for name in _INT_KEY_FIELDS
    }
    digest_key_values = {
        name: [[0] * TOKEN_BYTES for _ in range(clean_num_envs)]
        for name in _DIGEST_KEY_FIELDS
    }
    full_key_values = [[0] * TOKEN_BYTES for _ in range(clean_num_envs)]
    full_key_receipt_values = [[0] * TOKEN_BYTES for _ in range(clean_num_envs)]
    committed_values = [[0] * TOKEN_BYTES for _ in range(clean_num_envs)]
    install_receipt_values = [[0] * TOKEN_BYTES for _ in range(clean_num_envs)]
    task_identity_values = [[0] * TOKEN_BYTES for _ in range(clean_num_envs)]
    for row, receipt_row in zip(rows, receipt_rows):
        env_id = int(row["env_id"])
        key = row["outcome_key"]
        if not isinstance(key, _c05.LandingOutcomeShotKey):
            raise LandingOutcomeDeviceError("validated C05 key type drifted")
        mask_values[env_id] = True
        flight_values[env_id] = int(row["flight_slot"])
        mailbox_values[env_id] = int(row["mailbox_slot"])
        generation_values[env_id] = int(row["ball_generation"])
        reveal_values[env_id] = int(row["reveal_control_step"])
        deadline_values[env_id] = int(row["selected_contact_deadline_control_step"])
        horizon_values[env_id] = int(row["first_crossing_horizon_control_step"])
        target_values[env_id] = tuple(row["target_xy_m"])
        for name in _INT_KEY_FIELDS:
            int_key_values[name][env_id] = int(getattr(key, name))
        for name in _DIGEST_KEY_FIELDS:
            digest = (
                row["run_token_sha256"]
                if name == "run_id"
                else row["carry_token_sha256"]
                if name == "carry_chain_id"
                else getattr(key, name)
            )
            digest_key_values[name][env_id] = list(bytes.fromhex(str(digest)))
        full_key_values[env_id] = list(bytes.fromhex(str(row["full_key_sha256"])))
        full_key_receipt_values[env_id] = list(
            bytes.fromhex(str(row["full_key_receipt_sha256"]))
        )
        committed_values[env_id] = list(
            bytes.fromhex(str(row["committed_reveal_sha256"]))
        )
        install_receipt_values[env_id] = list(
            bytes.fromhex(str(receipt_row["row_receipt_sha256"]))
        )
        task_identity_values[env_id] = list(
            bytes.fromhex(str(row["task_identity_sha256"]))
        )

    key = DeviceLandingOutcomeKey(
        **{
            **{
                name: torch.tensor(
                    int_key_values[name],
                    dtype=torch.int64,
                    device=clean_device,
                )
                for name in _INT_KEY_FIELDS
            },
            **{
                name: torch.tensor(
                    digest_key_values[name],
                    dtype=torch.uint8,
                    device=clean_device,
                )
                for name in _DIGEST_KEY_FIELDS
            },
        }
    )
    pack_values: dict[str, object] = {
        "mask": torch.tensor(mask_values, dtype=torch.bool, device=clean_device),
        "flight_slot": torch.tensor(
            flight_values, dtype=torch.int64, device=clean_device
        ),
        "mailbox_slot": torch.tensor(
            mailbox_values, dtype=torch.int64, device=clean_device
        ),
        "task_key": key,
        "full_key_sha256": torch.tensor(
            full_key_values, dtype=torch.uint8, device=clean_device
        ),
        "full_key_receipt_sha256": torch.tensor(
            full_key_receipt_values, dtype=torch.uint8, device=clean_device
        ),
        "committed_reveal_sha256": torch.tensor(
            committed_values, dtype=torch.uint8, device=clean_device
        ),
        "install_receipt_sha256": torch.tensor(
            install_receipt_values, dtype=torch.uint8, device=clean_device
        ),
        "ball_generation": torch.tensor(
            generation_values, dtype=torch.int64, device=clean_device
        ),
        "task_identity_token": torch.tensor(
            task_identity_values, dtype=torch.uint8, device=clean_device
        ),
        "target_xy_m": torch.tensor(
            target_values, dtype=dtype, device=clean_device
        ),
        "reveal_control_step": torch.tensor(
            reveal_values, dtype=torch.int64, device=clean_device
        ),
        "selected_contact_deadline_control_step": torch.tensor(
            deadline_values, dtype=torch.int64, device=clean_device
        ),
        "first_crossing_horizon_control_step": torch.tensor(
            horizon_values, dtype=torch.int64, device=clean_device
        ),
    }
    pack = LandingRevealInstallDevicePack(
        _values=pack_values,
        _token=_INSTALL_PACK_TOKEN,
    )
    return LandingRevealInstall(
        _receipt=receipt,
        _device_pack=pack,
        _token=_INSTALL_AUTH_TOKEN,
    )


def build_landing_reveal_install_batch(
    committed_reveal_batch: object,
    *,
    expected_committed_reveal_batch_sha256: str,
    expected_committed_reveal_sha256: tuple[str, ...],
    expected_r05_source_sha256: str,
    expected_r05_contract_sha256: str,
    expected_c05_source_sha256: str,
    profile: LandingPlacementProfile,
    expected_profile_sha256: str,
    text_registry: LandingOutcomeTextRegistry | Mapping[str, object],
    expected_text_registry_sha256: str,
    mailbox_slots: tuple[int, ...],
    first_crossing_horizon_control_steps: tuple[int, ...],
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> LandingRevealInstall:
    """Tombstone: committed R05 authority is too late for capacity admission."""

    raise LandingOutcomeDeviceError(
        "committed-batch landing ingress is tombstoned; use "
        "prepare_from_reveal_final_preview"
    )


def build_landing_reveal_install(
    committed_reveal_batch: object,
    *,
    expected_committed_reveal_batch_sha256: str,
    expected_committed_reveal_sha256: str,
    expected_r05_source_sha256: str,
    expected_r05_contract_sha256: str,
    expected_c05_source_sha256: str,
    profile: LandingPlacementProfile,
    expected_profile_sha256: str,
    text_registry: LandingOutcomeTextRegistry | Mapping[str, object],
    expected_text_registry_sha256: str,
    mailbox_slot: int,
    first_crossing_horizon_control_step: int,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> LandingRevealInstall:
    """Tombstone: even a singleton must prepare from the exact R05 preview."""

    raise LandingOutcomeDeviceError(
        "committed-batch landing ingress is tombstoned; use "
        "prepare_from_reveal_final_preview"
    )


def _device_values_from_install_receipt(
    receipt_payload: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Reconstruct every device-pack byte from the sealed host receipt."""

    num_envs = int(receipt_payload["num_envs"])
    rows = receipt_payload["rows"]
    if not isinstance(rows, list):
        raise LandingOutcomeDeviceError("install receipt rows differ")
    mask_values = [False] * num_envs
    flight_values = [-1] * num_envs
    mailbox_values = [-1] * num_envs
    generation_values = [-1] * num_envs
    reveal_values = [-1] * num_envs
    deadline_values = [-1] * num_envs
    horizon_values = [-1] * num_envs
    target_values = [(0.0, 0.0)] * num_envs
    int_key_values = {
        name: ([-1] * num_envs if name == "env_id" else [0] * num_envs)
        for name in _INT_KEY_FIELDS
    }
    digest_key_values = {
        name: [[0] * TOKEN_BYTES for _ in range(num_envs)]
        for name in _DIGEST_KEY_FIELDS
    }
    full_key_values = [[0] * TOKEN_BYTES for _ in range(num_envs)]
    full_key_receipt_values = [[0] * TOKEN_BYTES for _ in range(num_envs)]
    committed_values = [[0] * TOKEN_BYTES for _ in range(num_envs)]
    install_receipt_values = [[0] * TOKEN_BYTES for _ in range(num_envs)]
    task_identity_values = [[0] * TOKEN_BYTES for _ in range(num_envs)]
    for row in rows:
        if not isinstance(row, Mapping):
            raise LandingOutcomeDeviceError("install receipt row differs")
        env_id = int(row["env_id"])
        try:
            key = _c05.LandingOutcomeShotKey.from_mapping(row["outcome_key"])
        except Exception as exc:
            raise LandingOutcomeDeviceError("install receipt C05 key differs") from exc
        mask_values[env_id] = True
        flight_values[env_id] = int(row["flight_slot"])
        mailbox_values[env_id] = int(row["mailbox_slot"])
        generation_values[env_id] = int(row["ball_generation"])
        reveal_values[env_id] = int(row["reveal_control_step"])
        deadline_values[env_id] = int(row["selected_contact_deadline_control_step"])
        horizon_values[env_id] = int(row["first_crossing_horizon_control_step"])
        target_values[env_id] = tuple(float(item) for item in row["target_xy_m"])
        for name in _INT_KEY_FIELDS:
            int_key_values[name][env_id] = int(getattr(key, name))
        for name in _DIGEST_KEY_FIELDS:
            digest = (
                row["run_id_token_sha256"]
                if name == "run_id"
                else row["carry_chain_id_token_sha256"]
                if name == "carry_chain_id"
                else getattr(key, name)
            )
            digest_key_values[name][env_id] = list(bytes.fromhex(str(digest)))
        full_key_values[env_id] = list(bytes.fromhex(str(row["full_key_sha256"])))
        full_key_receipt_values[env_id] = list(
            bytes.fromhex(str(row["full_key_receipt_sha256"]))
        )
        committed_values[env_id] = list(
            bytes.fromhex(str(row["committed_reveal_sha256"]))
        )
        install_receipt_values[env_id] = list(
            bytes.fromhex(str(row["row_receipt_sha256"]))
        )
        task_identity_values[env_id] = list(
            bytes.fromhex(str(row["task_identity_sha256"]))
        )
    key = DeviceLandingOutcomeKey(
        **{
            **{
                name: torch.tensor(
                    int_key_values[name], dtype=torch.int64, device=device
                )
                for name in _INT_KEY_FIELDS
            },
            **{
                name: torch.tensor(
                    digest_key_values[name], dtype=torch.uint8, device=device
                )
                for name in _DIGEST_KEY_FIELDS
            },
        }
    )
    return {
        "mask": torch.tensor(mask_values, dtype=torch.bool, device=device),
        "flight_slot": torch.tensor(flight_values, dtype=torch.int64, device=device),
        "mailbox_slot": torch.tensor(mailbox_values, dtype=torch.int64, device=device),
        "task_key": key,
        "full_key_sha256": torch.tensor(
            full_key_values, dtype=torch.uint8, device=device
        ),
        "full_key_receipt_sha256": torch.tensor(
            full_key_receipt_values, dtype=torch.uint8, device=device
        ),
        "committed_reveal_sha256": torch.tensor(
            committed_values, dtype=torch.uint8, device=device
        ),
        "install_receipt_sha256": torch.tensor(
            install_receipt_values, dtype=torch.uint8, device=device
        ),
        "ball_generation": torch.tensor(
            generation_values, dtype=torch.int64, device=device
        ),
        "task_identity_token": torch.tensor(
            task_identity_values, dtype=torch.uint8, device=device
        ),
        "target_xy_m": torch.tensor(target_values, dtype=dtype, device=device),
        "reveal_control_step": torch.tensor(
            reveal_values, dtype=torch.int64, device=device
        ),
        "selected_contact_deadline_control_step": torch.tensor(
            deadline_values, dtype=torch.int64, device=device
        ),
        "first_crossing_horizon_control_step": torch.tensor(
            horizon_values, dtype=torch.int64, device=device
        ),
    }


@dataclass(frozen=True)
class PostPhysicsFlightBatch:
    observe_mask: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center_m: torch.Tensor
    current_ball_center_m: torch.Tensor
    observation_stamp: PhysicsStampBatch
    selected_contact_event: torch.Tensor
    selected_contact_ball_center_m: torch.Tensor
    selected_contact_outgoing_segment_anchor_m: torch.Tensor
    selected_contact_stamp: PhysicsStampBatch
    net_crossing_event: torch.Tensor
    net_clear_at_crossing: torch.Tensor
    net_crossing_stamp: PhysicsStampBatch
    crossing_report_delivered: torch.Tensor
    first_descending_crossing_event: torch.Tensor
    first_descending_crossing_xy_m: torch.Tensor
    first_descending_crossing_stamp: PhysicsStampBatch
    nonfinite_observation: torch.Tensor
    producer_contract_fault: torch.Tensor
    engine_overflow: torch.Tensor
    physical_publication_identity: object


@dataclass(frozen=True)
class DeviceMutationResult:
    accepted: torch.Tensor
    rejected: torch.Tensor
    fault_bits: torch.Tensor


@dataclass(frozen=True)
class FlightLifecycleSnapshotBatch:
    """Complete read-only device lifecycle root shared with physical ownership."""

    state: torch.Tensor
    task_key: DeviceLandingOutcomeKey
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    mailbox_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    physical_retired: torch.Tensor
    mailbox_state: torch.Tensor
    mailbox_task_key: DeviceLandingOutcomeKey
    mailbox_full_key_sha256: torch.Tensor
    mailbox_ball_generation: torch.Tensor
    mailbox_reserved_flight_slot: torch.Tensor
    mailbox_history_valid: torch.Tensor
    mailbox_physical_retired: torch.Tensor
    mutation_version: torch.Tensor


class ActionBallFullMdpObservationProjection:
    """Opaque owner-issued handle for the current clone-only R06 facts."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("R06 observation projections are owner-issued")

    def __copy__(self):
        raise TypeError("R06 observation projections cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("R06 observation projections cannot be copied")

    def __reduce__(self):
        raise TypeError("R06 observation projections cannot be serialized")

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError("R06 observation projections cannot be serialized")


@dataclass(frozen=True, eq=False, repr=False)
class ActionEpochR06CurrentFlightObservationView:
    """Clone-only semantic facts for the current live ActionEpoch flight.

    ``flight_slot`` is an engine-local locator for the paired Physical scene
    row, never part of shot identity or an observation feature.  ``-1`` means
    that the current ActionEpoch publication has no live R06 flight; all three
    latches are then false.
    """

    r06_owner: object
    publication_identity: object
    flight_slot: torch.Tensor
    contact_valid: torch.Tensor
    net_crossed: torch.Tensor
    net_clear: torch.Tensor


@dataclass(frozen=True)
class PreviousPaidActionEpochRows:
    """Clone-only latest paid-shot facts, one exact row per environment."""

    valid: torch.Tensor
    shot_key: _row_identity.ActionEpochShotKey
    publication_ordinal: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor


@dataclass(frozen=True)
class ActionEpochR06OutcomeRows:
    """Clone-only current settled mailbox rows for the bound Epoch owner."""

    valid: torch.Tensor
    shot_key: _row_identity.ActionEpochShotKey
    publication_ordinal: torch.Tensor
    settlement_step: torch.Tensor
    valid_bits: torch.Tensor
    fact_values: torch.Tensor
    outcome_code: torch.Tensor
    owner_fault_bits: torch.Tensor


@dataclass(frozen=True, eq=False, repr=False)
class ActionEpochR06CurrentSettlementDelta:
    """One-shot current-substep settlement facts for the bound Epoch owner."""

    rows: ActionEpochR06OutcomeRows
    sequence: int
    r06_owner: object
    epoch_owner: object
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class PostPhysicsMutationResult:
    """Post-physics verdict plus one exact owner-issued contact authority.

    ``new_valid_contact_mask`` is R06's causal, per-publication acceptance
    result.  It is neither the caller's raw event bit nor the cumulative
    ``flight_contact_valid`` state.  Consumers must validate and consume
    ``contact_authority`` through the issuing owner; the clone fields here are
    evidence, not forgeable authorization.
    """

    accepted: torch.Tensor
    rejected: torch.Tensor
    fault_bits: torch.Tensor
    settled_mask: torch.Tensor
    settlement_cause: torch.Tensor
    flight_slot: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    mutation_version: torch.Tensor
    physical_publication_identity: object
    new_valid_contact_mask: torch.Tensor
    selected_contact_stamp: PhysicsStampBatch
    contact_authority: "LandingOutcomePostPhysicsContactAuthority"


class LandingOutcomePostPhysicsContactAuthority:
    """Opaque one-shot identity for R06's causal contact publication."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("post-physics contact authorities are owner-issued")

    def __copy__(self):
        raise TypeError("post-physics contact authorities cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("post-physics contact authorities cannot be copied")

    def __reduce__(self):
        raise TypeError("post-physics contact authorities cannot be serialized")

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError("post-physics contact authorities cannot be serialized")

    @property
    def publication_identity(self) -> object:
        return _owned_post_physics_contact_row(self).publication_identity

    @property
    def new_valid_contact_mask(self) -> torch.Tensor:
        return (
            _owned_post_physics_contact_row(self)
            .new_valid_contact_mask.detach()
            .clone()
        )

    @property
    def task_key(self) -> DeviceLandingOutcomeKey:
        row = _owned_post_physics_contact_row(self)
        return DeviceLandingOutcomeKey(
            **{
                name: getattr(row.task_key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        )

    @property
    def full_key_sha256(self) -> torch.Tensor:
        return _owned_post_physics_contact_row(self).full_key_sha256.detach().clone()

    @property
    def ball_generation(self) -> torch.Tensor:
        return _owned_post_physics_contact_row(self).ball_generation.detach().clone()

    @property
    def flight_slot(self) -> torch.Tensor:
        return _owned_post_physics_contact_row(self).flight_slot.detach().clone()

    @property
    def observation_ordinal(self) -> torch.Tensor:
        return (
            _owned_post_physics_contact_row(self)
            .observation_ordinal.detach()
            .clone()
        )

    @property
    def selected_contact_stamp(self) -> PhysicsStampBatch:
        row = _owned_post_physics_contact_row(self)
        return PhysicsStampBatch(
            control_step=row.selected_contact_stamp.control_step.detach().clone(),
            physics_substep=(
                row.selected_contact_stamp.physics_substep.detach().clone()
            ),
            event_phase=row.selected_contact_stamp.event_phase.detach().clone(),
        )

    @property
    def mutation_version(self) -> torch.Tensor:
        return _owned_post_physics_contact_row(self).mutation_version.detach().clone()


@dataclass(frozen=True, eq=False)
class LandingOutcomePostPhysicsContactAuthorityView:
    """Clone-only causal facts released by consuming one exact authority."""

    publication_identity: object
    new_valid_contact_mask: torch.Tensor
    task_key: DeviceLandingOutcomeKey
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    flight_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    selected_contact_stamp: PhysicsStampBatch
    mutation_version: torch.Tensor


@dataclass(frozen=True, eq=False)
class PhysicalRetireCleanupMaskCapability:
    """Opaque exact R06 cleanup mask shared with the paired physical owner."""

    _device_mask: torch.Tensor
    _owner_identity: object
    _token: object

    @property
    def device_mask(self) -> torch.Tensor:
        return self._device_mask.detach().clone()


@dataclass(frozen=True, eq=False)
class LandingOutcomePhysicalParkPreparedTokenClaim:
    """Exact physical-owner claim over one retained R06 retire prepare."""

    r06_prepared_retire: object
    r06_cleanup_capability: PhysicalRetireCleanupMaskCapability
    physical_cleanup_capability: object
    _physical_prepared_token: object


class LandingOutcomePhysicalParkTokenAuthority:
    """Validate exact physical prepare/commit identity without caller pins."""

    __slots__ = (
        "_physical_owner",
        "_prepared_validator",
        "_committed_validator",
    )

    def __init__(
        self,
        *,
        physical_owner: object,
        prepared_validator: object,
        committed_validator: object,
        _token: object,
    ) -> None:
        if (
            _token is not _PHYSICAL_PARK_AUTHORITY_MINT_TOKEN
            or not callable(prepared_validator)
            or not callable(committed_validator)
            or getattr(prepared_validator, "__self__", None)
            is not physical_owner
            or getattr(committed_validator, "__self__", None)
            is not physical_owner
        ):
            raise LandingOutcomeDeviceError(
                "physical park authority must be factory-minted from one owner"
            )
        self._physical_owner = physical_owner
        self._prepared_validator = prepared_validator
        self._committed_validator = committed_validator

    @property
    def physical_owner(self) -> object:
        return self._physical_owner

    def require_owned_prepared_token(
        self,
        physical_prepared_token: object,
        *,
        expected_r06_prepared_retire: object,
    ) -> LandingOutcomePhysicalParkPreparedTokenClaim:
        claim = self._prepared_validator(physical_prepared_token)
        physical_capability = getattr(
            claim, "physical_cleanup_capability", None
        )
        physical_capability_type = type(physical_capability)
        owner_type = type(self._physical_owner)
        owner_module = sys.modules.get(owner_type.__module__)
        physical_mask = getattr(physical_capability, "_device_mask", None)
        if (
            type(claim) is not LandingOutcomePhysicalParkPreparedTokenClaim
            or claim._physical_prepared_token is not physical_prepared_token
            or claim.r06_prepared_retire is not expected_r06_prepared_retire
            or claim.r06_cleanup_capability
            is not expected_r06_prepared_retire._cleanup_capability
            or physical_capability_type.__name__
            != "PhysicalParkCleanupMaskCapability"
            or owner_module is None
            or physical_capability_type.__module__ != owner_type.__module__
            or getattr(
                owner_module,
                "PhysicalParkCleanupMaskCapability",
                None,
            )
            is not physical_capability_type
            or getattr(physical_capability, "_owner_identity", None)
            is not getattr(self._physical_owner, "_owner_identity", None)
            or getattr(physical_capability, "_prepared_token", None)
            is not physical_prepared_token
            or not isinstance(physical_mask, torch.Tensor)
            or tuple(physical_mask.shape)
            != tuple(expected_r06_prepared_retire._cleanup_mask.shape)
            or physical_mask.dtype != torch.bool
            or physical_mask.device
            != expected_r06_prepared_retire._cleanup_mask.device
        ):
            raise LandingOutcomeDeviceError(
                "physical park prepared token authority differs"
            )
        return claim

    def require_committed_prepared_token(
        self,
        physical_prepared_token: object,
        *,
        expected_r06_prepared_retire: object,
    ) -> LandingOutcomePhysicalParkPreparedTokenClaim:
        claim = self._committed_validator(physical_prepared_token)
        physical_capability = getattr(
            claim, "physical_cleanup_capability", None
        )
        physical_capability_type = type(physical_capability)
        owner_type = type(self._physical_owner)
        owner_module = sys.modules.get(owner_type.__module__)
        physical_mask = getattr(physical_capability, "_device_mask", None)
        if (
            type(claim) is not LandingOutcomePhysicalParkPreparedTokenClaim
            or claim._physical_prepared_token is not physical_prepared_token
            or claim.r06_prepared_retire is not expected_r06_prepared_retire
            or claim.r06_cleanup_capability
            is not expected_r06_prepared_retire._cleanup_capability
            or physical_capability_type.__name__
            != "PhysicalParkCleanupMaskCapability"
            or owner_module is None
            or physical_capability_type.__module__ != owner_type.__module__
            or getattr(
                owner_module,
                "PhysicalParkCleanupMaskCapability",
                None,
            )
            is not physical_capability_type
            or getattr(physical_capability, "_owner_identity", None)
            is not getattr(self._physical_owner, "_owner_identity", None)
            or getattr(physical_capability, "_prepared_token", None)
            is not physical_prepared_token
            or not isinstance(physical_mask, torch.Tensor)
            or tuple(physical_mask.shape)
            != tuple(expected_r06_prepared_retire._cleanup_mask.shape)
            or physical_mask.dtype != torch.bool
            or physical_mask.device
            != expected_r06_prepared_retire._cleanup_mask.device
        ):
            raise LandingOutcomeDeviceError(
                "physical park commit authority differs"
            )
        return claim


def mint_landing_outcome_physical_park_token_authority(
    physical_owner: object,
) -> LandingOutcomePhysicalParkTokenAuthority:
    """Mint one pair capability from the physical owner's exact bound methods."""

    owner_type = type(physical_owner)
    owner_module = sys.modules.get(owner_type.__module__)
    owner_source = None if owner_module is None else getattr(
        owner_module, "__file__", None
    )
    if (
        owner_type.__name__ != "ActionBallPhysicalFlightDeviceOwner"
        or owner_module is None
        or getattr(owner_module, owner_type.__name__, None) is not owner_type
        or owner_source is None
        or Path(owner_source).name
        != "action_ball_physical_flight_device.py"
    ):
        raise LandingOutcomeDeviceError(
            "physical park authority requires the exact physical owner class"
        )
    prepared_validator = getattr(
        physical_owner,
        "require_owned_r06_physical_park_prepared_token",
        None,
    )
    committed_validator = getattr(
        physical_owner,
        "require_committed_r06_physical_park_prepared_token",
        None,
    )
    return LandingOutcomePhysicalParkTokenAuthority(
        physical_owner=physical_owner,
        prepared_validator=prepared_validator,
        committed_validator=committed_validator,
        _token=_PHYSICAL_PARK_AUTHORITY_MINT_TOKEN,
    )


class _OpaqueLandingOutcomeSelectedResetCapability:
    """Non-copyable empty identity backed only by the owner registry."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("R06 selected-reset capabilities are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("R06 selected-reset capabilities are immutable")

    def __copy__(self):
        raise TypeError("R06 selected-reset capabilities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("R06 selected-reset capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("R06 selected-reset capabilities cannot be serialized")


class LandingOutcomeSelectedResetMaskCapability(
    _OpaqueLandingOutcomeSelectedResetCapability
):
    """Opaque identity for an owner-private selected-reset mask record."""

    __slots__ = ()


@dataclass(frozen=True)
class LandingOutcomeSelectedResetMaskView:
    """Fresh clone-only facts released by the exact mask capability."""

    prepared_reset: "PreparedLandingOutcomeSelectedReset"
    mask_capability: LandingOutcomeSelectedResetMaskCapability
    device_r05_owner: _r05_device.DeviceR05Owner
    device_r05_prepared_true_reset: _r05_device.DeviceR05PreparedTrueReset
    device_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor


class PreparedLandingOutcomeSelectedReset(
    _OpaqueLandingOutcomeSelectedResetCapability
):
    """Opaque identity for an owner-private R06 reset after-image."""

    __slots__ = ()


class ArmedLandingOutcomeSelectedReset(
    _OpaqueLandingOutcomeSelectedResetCapability
):
    """Opaque identity bound to one exact physical finalized park image."""

    __slots__ = ()


class LandingOutcomeSelectedResetCommitToken(
    _OpaqueLandingOutcomeSelectedResetCapability
):
    """Opaque exact proof that physical park preceded R06 retirement."""

    __slots__ = ()


class LandingOutcomeSelectedResetCompletionToken(
    _OpaqueLandingOutcomeSelectedResetCapability
):
    """Opaque leaf ACK; only the top reset owner may materialize audit bytes."""

    __slots__ = ()


@dataclass(frozen=True, eq=False)
class LandingOutcomeSelectedResetPhysicalParkPreparedTokenClaim:
    """Physical owner's exact claim over one R06 prepare and mask."""

    r06_prepared_reset: PreparedLandingOutcomeSelectedReset
    r06_mask_capability: LandingOutcomeSelectedResetMaskCapability
    physical_prepared_token: object


@dataclass(frozen=True, eq=False)
class LandingOutcomeSelectedResetPhysicalParkCommitTokenClaim:
    """Physical owner's exact claim that one selected park was committed."""

    r06_armed_reset: ArmedLandingOutcomeSelectedReset
    physical_prepared_token: object
    physical_commit_token: object


class LandingOutcomeSelectedResetPhysicalParkTokenAuthority:
    """Validate physical selected-reset identity without caller-owned pins."""

    __slots__ = (
        "_physical_owner",
        "_prepared_validator",
        "_committed_validator",
    )

    def __init__(
        self,
        *,
        physical_owner: object,
        prepared_validator: object,
        committed_validator: object,
        _token: object,
    ) -> None:
        if (
            _token is not _SELECTED_RESET_PHYSICAL_PARK_AUTHORITY_MINT_TOKEN
            or not callable(prepared_validator)
            or not callable(committed_validator)
            or getattr(prepared_validator, "__self__", None) is not physical_owner
            or getattr(committed_validator, "__self__", None) is not physical_owner
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset park authority must be factory-minted"
            )
        self._physical_owner = physical_owner
        self._prepared_validator = prepared_validator
        self._committed_validator = committed_validator

    @property
    def physical_owner(self) -> object:
        return self._physical_owner

    def require_owned_prepared_token(
        self,
        physical_prepared_token: object,
        *,
        expected_r06_prepared_reset: PreparedLandingOutcomeSelectedReset,
    ) -> LandingOutcomeSelectedResetPhysicalParkPreparedTokenClaim:
        claim = self._prepared_validator(
            physical_prepared_token,
            expected_r06_prepared_reset=expected_r06_prepared_reset,
        )
        if (
            type(claim)
            is not LandingOutcomeSelectedResetPhysicalParkPreparedTokenClaim
            or claim.r06_prepared_reset is not expected_r06_prepared_reset
            or type(claim.r06_mask_capability)
            is not LandingOutcomeSelectedResetMaskCapability
            or claim.physical_prepared_token is not physical_prepared_token
        ):
            raise LandingOutcomeDeviceError(
                "physical selected-reset prepared authority differs"
            )
        return claim

    def require_committed_park_token(
        self,
        physical_commit_token: object,
        *,
        expected_r06_armed_reset: ArmedLandingOutcomeSelectedReset,
    ) -> LandingOutcomeSelectedResetPhysicalParkCommitTokenClaim:
        claim = self._committed_validator(
            physical_commit_token,
            expected_r06_armed_reset=expected_r06_armed_reset,
        )
        if (
            type(claim)
            is not LandingOutcomeSelectedResetPhysicalParkCommitTokenClaim
            or claim.r06_armed_reset is not expected_r06_armed_reset
            or claim.physical_commit_token is not physical_commit_token
        ):
            raise LandingOutcomeDeviceError(
                "physical selected-reset commit authority differs"
            )
        return claim


def mint_landing_outcome_selected_reset_physical_park_token_authority(
    physical_owner: object,
) -> LandingOutcomeSelectedResetPhysicalParkTokenAuthority:
    """Mint the selected-reset pair capability from one exact physical owner."""

    owner_type = type(physical_owner)
    owner_module = sys.modules.get(owner_type.__module__)
    owner_source = None if owner_module is None else getattr(
        owner_module, "__file__", None
    )
    if (
        owner_type.__name__ != "ActionBallPhysicalFlightDeviceOwner"
        or owner_module is None
        or getattr(owner_module, owner_type.__name__, None) is not owner_type
        or owner_source is None
        or Path(owner_source).name != "action_ball_physical_flight_device.py"
    ):
        raise LandingOutcomeDeviceError(
            "selected-reset authority requires the exact physical owner class"
        )
    return LandingOutcomeSelectedResetPhysicalParkTokenAuthority(
        physical_owner=physical_owner,
        prepared_validator=getattr(
            physical_owner,
            "require_owned_selected_reset_prepared_token",
            None,
        ),
        committed_validator=getattr(
            physical_owner,
            "require_committed_selected_reset_park_token",
            None,
        ),
        _token=_SELECTED_RESET_PHYSICAL_PARK_AUTHORITY_MINT_TOKEN,
    )


@dataclass(frozen=True, eq=False)
class PreparedPhysicalRetire:
    """Opaque all-grid R06 retire after-image prepared without live writes."""

    _owner_identity: object
    _settlement_result: PostPhysicsMutationResult
    _accepted: torch.Tensor
    _rejected: torch.Tensor
    _fault_bits: torch.Tensor
    _normal_mask: torch.Tensor
    _cleanup_mask: torch.Tensor
    _cleanup_capability: PhysicalRetireCleanupMaskCapability
    _token: object

    @property
    def accepted(self) -> torch.Tensor:
        return self._accepted.detach().clone()

    @property
    def rejected(self) -> torch.Tensor:
        return self._rejected.detach().clone()

    @property
    def fault_bits(self) -> torch.Tensor:
        return self._fault_bits.detach().clone()

    @property
    def normal_mask(self) -> torch.Tensor:
        return self._normal_mask.detach().clone()

    @property
    def cleanup_mask(self) -> torch.Tensor:
        return self._cleanup_mask.detach().clone()


@dataclass(frozen=True, eq=False)
class ArmedPhysicalRetire:
    """Opaque R06 handle bound to one exact physical prepared token."""

    _prepared_retire: PreparedPhysicalRetire
    _physical_prepared_token: object
    _r06_cleanup_capability: PhysicalRetireCleanupMaskCapability
    _physical_cleanup_capability: object
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class PhysicalRetireMutationResult:
    """Exact all-grid retire verdict and complete final lifecycle root."""

    accepted: torch.Tensor
    rejected: torch.Tensor
    fault_bits: torch.Tensor
    normal_mask: torch.Tensor
    cleanup_mask: torch.Tensor
    portable_success_mask: torch.Tensor
    task_key: DeviceLandingOutcomeKey
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    mailbox_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    physical_retired: torch.Tensor
    mailbox_state: torch.Tensor
    mailbox_task_key: DeviceLandingOutcomeKey
    mailbox_full_key_sha256: torch.Tensor
    mailbox_ball_generation: torch.Tensor
    mailbox_reserved_flight_slot: torch.Tensor
    mailbox_history_valid: torch.Tensor
    mailbox_physical_retired: torch.Tensor
    mutation_version_before: torch.Tensor
    mutation_version_after: torch.Tensor
    initial_lifecycle_root: FlightLifecycleSnapshotBatch
    final_lifecycle_root: FlightLifecycleSnapshotBatch


@dataclass(frozen=True)
class _RetainedPostPhysicsSettlement:
    """Owner-private immutable facts behind one exact post-physics result."""

    result: PostPhysicsMutationResult
    settled_mask: torch.Tensor
    accepted: torch.Tensor
    rejected: torch.Tensor
    fault_bits: torch.Tensor
    settlement_cause: torch.Tensor
    flight_slot: torch.Tensor
    task_key: DeviceLandingOutcomeKey
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    mailbox_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    mutation_version: torch.Tensor


@dataclass(frozen=True, eq=False, repr=False)
class _ActionEpochOutcomeCandidateGrid:
    """One source-grid ABI shared by retained audit and current settlement."""

    candidate: torch.Tensor
    shot_key_values: torch.Tensor
    publication_ordinal: torch.Tensor
    settlement_step: torch.Tensor
    policy_eligible: torch.Tensor
    fact_values: torch.Tensor
    outcome_code: torch.Tensor
    owner_fault_bits: torch.Tensor


def _with_action_epoch_candidate(
    grid: _ActionEpochOutcomeCandidateGrid, candidate: torch.Tensor
) -> _ActionEpochOutcomeCandidateGrid:
    """Reuse one immutable fact grid while narrowing only its row mask."""

    return _ActionEpochOutcomeCandidateGrid(
        candidate=candidate,
        shot_key_values=grid.shot_key_values,
        publication_ordinal=grid.publication_ordinal,
        settlement_step=grid.settlement_step,
        policy_eligible=grid.policy_eligible,
        fact_values=grid.fact_values,
        outcome_code=grid.outcome_code,
        owner_fault_bits=grid.owner_fault_bits,
    )


@dataclass(frozen=True, eq=False)
class _RetainedPostPhysicsContactAuthority:
    """Owner-private registry row behind one opaque contact authority."""

    authority: LandingOutcomePostPhysicsContactAuthority
    publication_identity: object
    new_valid_contact_mask: torch.Tensor
    task_key: DeviceLandingOutcomeKey
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    flight_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    selected_contact_stamp: PhysicsStampBatch
    mutation_version: torch.Tensor
    consumed: bool


_POSTPHYSICS_CONTACT_REGISTRY: dict[
    int, _RetainedPostPhysicsContactAuthority
] = {}


def _owned_post_physics_contact_row(
    authority: LandingOutcomePostPhysicsContactAuthority,
) -> _RetainedPostPhysicsContactAuthority:
    if type(authority) is not LandingOutcomePostPhysicsContactAuthority:
        raise LandingOutcomeDeviceError(
            "post-physics contact authority type differs"
        )
    retained = _POSTPHYSICS_CONTACT_REGISTRY.get(id(authority))
    if retained is None or retained.authority is not authority:
        raise LandingOutcomeDeviceError(
            "post-physics contact authority is stale or foreign"
        )
    return retained


@dataclass
class _ActivePhysicalRetireLease:
    prepared_retire: PreparedPhysicalRetire
    retained_cleanup_mask: torch.Tensor
    tensor_swaps: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    host_after_state: Mapping[str, object]
    mutation_result: PhysicalRetireMutationResult
    armed_retire: ArmedPhysicalRetire | None = None


@dataclass
class _ActiveLandingOutcomeSelectedResetLease:
    prepared_reset: PreparedLandingOutcomeSelectedReset
    mask_capability: LandingOutcomeSelectedResetMaskCapability
    device_r05_owner: _r05_device.DeviceR05Owner
    device_r05_prepared_true_reset: _r05_device.DeviceR05PreparedTrueReset
    device_r05_prepared_projection: _r05_device.DeviceR05PreparedTrueResetProjection
    selected_env_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    tensor_swaps: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    host_after_state: Mapping[str, object]
    armed_reset: ArmedLandingOutcomeSelectedReset | None = None
    physical_prepared_token: object | None = None
    physical_commit_token: object | None = None
    commit_token: LandingOutcomeSelectedResetCommitToken | None = None
    completion_token: LandingOutcomeSelectedResetCompletionToken | None = None
    device_r05_true_reset_receipt: object | None = None


@dataclass
class _PreparedR06GlobalDrain:
    """One leaf-local lease over the sole global pre-optimizer pack."""

    pack: object
    authority: object
    update_index: int
    completed_environment_steps: int


def _clone_lifecycle_snapshot(
    value: FlightLifecycleSnapshotBatch,
) -> FlightLifecycleSnapshotBatch:
    return FlightLifecycleSnapshotBatch(
        state=value.state.detach().clone(),
        task_key=DeviceLandingOutcomeKey(
            **{
                name: getattr(value.task_key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        ),
        full_key_sha256=value.full_key_sha256.detach().clone(),
        ball_generation=value.ball_generation.detach().clone(),
        mailbox_slot=value.mailbox_slot.detach().clone(),
        observation_ordinal=value.observation_ordinal.detach().clone(),
        physical_retired=value.physical_retired.detach().clone(),
        mailbox_state=value.mailbox_state.detach().clone(),
        mailbox_task_key=DeviceLandingOutcomeKey(
            **{
                name: getattr(value.mailbox_task_key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        ),
        mailbox_full_key_sha256=(
            value.mailbox_full_key_sha256.detach().clone()
        ),
        mailbox_ball_generation=(
            value.mailbox_ball_generation.detach().clone()
        ),
        mailbox_reserved_flight_slot=(
            value.mailbox_reserved_flight_slot.detach().clone()
        ),
        mailbox_history_valid=value.mailbox_history_valid.detach().clone(),
        mailbox_physical_retired=(
            value.mailbox_physical_retired.detach().clone()
        ),
        mutation_version=value.mutation_version.detach().clone(),
    )


def _clone_physical_retire_result(
    value: PhysicalRetireMutationResult,
) -> PhysicalRetireMutationResult:
    initial_lifecycle = _clone_lifecycle_snapshot(
        value.initial_lifecycle_root
    )
    lifecycle = _clone_lifecycle_snapshot(value.final_lifecycle_root)
    return PhysicalRetireMutationResult(
        accepted=value.accepted.detach().clone(),
        rejected=value.rejected.detach().clone(),
        fault_bits=value.fault_bits.detach().clone(),
        normal_mask=value.normal_mask.detach().clone(),
        cleanup_mask=value.cleanup_mask.detach().clone(),
        portable_success_mask=value.portable_success_mask.detach().clone(),
        task_key=DeviceLandingOutcomeKey(
            **{
                name: getattr(value.task_key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        ),
        full_key_sha256=value.full_key_sha256.detach().clone(),
        ball_generation=value.ball_generation.detach().clone(),
        mailbox_slot=value.mailbox_slot.detach().clone(),
        observation_ordinal=value.observation_ordinal.detach().clone(),
        physical_retired=value.physical_retired.detach().clone(),
        mailbox_state=value.mailbox_state.detach().clone(),
        mailbox_task_key=DeviceLandingOutcomeKey(
            **{
                name: getattr(value.mailbox_task_key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        ),
        mailbox_full_key_sha256=(
            value.mailbox_full_key_sha256.detach().clone()
        ),
        mailbox_ball_generation=(
            value.mailbox_ball_generation.detach().clone()
        ),
        mailbox_reserved_flight_slot=(
            value.mailbox_reserved_flight_slot.detach().clone()
        ),
        mailbox_history_valid=value.mailbox_history_valid.detach().clone(),
        mailbox_physical_retired=(
            value.mailbox_physical_retired.detach().clone()
        ),
        mutation_version_before=value.mutation_version_before.detach().clone(),
        mutation_version_after=value.mutation_version_after.detach().clone(),
        initial_lifecycle_root=initial_lifecycle,
        final_lifecycle_root=lifecycle,
    )


@dataclass(frozen=True)
class SharedLandingOutcomeDeviceView:
    eligible: torch.Tensor
    policy_eligible: torch.Tensor
    mailbox_state: torch.Tensor
    physical_retired: torch.Tensor
    task_key: DeviceLandingOutcomeKey
    full_key_sha256: torch.Tensor
    full_key_receipt_sha256: torch.Tensor
    ball_generation: torch.Tensor
    reserved_flight_slot: torch.Tensor
    settlement_cause: torch.Tensor
    canonical_reason_code: torch.Tensor
    source_control_step: torch.Tensor
    source_physics_substep: torch.Tensor
    settlement_control_step: torch.Tensor
    settlement_physics_substep: torch.Tensor
    contact_valid: torch.Tensor
    selected_contact_ball_center_m: torch.Tensor
    first_plane_crossing_present: torch.Tensor
    first_plane_crossing_valid: torch.Tensor
    first_plane_crossing_nonfinite: torch.Tensor
    first_plane_crossing_xy_m: torch.Tensor
    first_plane_crossing_control_step: torch.Tensor
    first_plane_crossing_physics_substep: torch.Tensor
    ball_center_net_crossed: torch.Tensor
    ball_center_net_clear: torch.Tensor
    ball_center_net_crossing_control_step: torch.Tensor
    ball_center_net_crossing_physics_substep: torch.Tensor
    common_on_table_outcome: torch.Tensor
    on_opponent_table: torch.Tensor
    placement_error_m: torch.Tensor
    broad_kernel: torch.Tensor
    narrow_kernel: torch.Tensor
    blended_kernel: torch.Tensor
    table_gate: torch.Tensor
    canonical_total: torch.Tensor
    treatment_family_code: torch.Tensor
    c10_projection_sha256: torch.Tensor
    placement_treatment_gain: torch.Tensor
    consumer_view_epoch: torch.Tensor
    consumer_paid_mask: torch.Tensor
    payment_values: torch.Tensor
    fault_bits: torch.Tensor


@dataclass(frozen=True)
class _FullMdpRewardPaymentPayload:
    owner_identity: object
    cycle_identity: object
    consumer: str
    reward_epoch: torch.Tensor
    accepted: torch.Tensor
    rejected: torch.Tensor
    fault_bits: torch.Tensor


class LandingOutcomeFullMdpRewardPaymentVerdict:
    """Opaque R06 proof for one exact owner-accepted Reward payment."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> "LandingOutcomeFullMdpRewardPaymentVerdict":
        del cls
        raise TypeError("R06 full-MDP Reward verdicts are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("R06 full-MDP Reward verdicts are immutable")

    @property
    def accepted(self) -> torch.Tensor:
        return _owned_full_mdp_reward_payment(self).accepted.detach().clone()

    @property
    def rejected(self) -> torch.Tensor:
        return _owned_full_mdp_reward_payment(self).rejected.detach().clone()

    @property
    def fault_bits(self) -> torch.Tensor:
        return _owned_full_mdp_reward_payment(self).fault_bits.detach().clone()


@dataclass(frozen=True)
class _FullMdpRewardCyclePayload:
    owner_identity: object
    cycle_identity: object
    runtime_owner: object
    pre_reward_publication: object
    control_step: int
    reward_epoch: torch.Tensor


class LandingOutcomeFullMdpRewardCycleToken:
    """Opaque R06 capability for one exact top-opened Reward cycle."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> "LandingOutcomeFullMdpRewardCycleToken":
        del cls
        raise TypeError("R06 full-MDP Reward cycle tokens are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("R06 full-MDP Reward cycle tokens are immutable")

    def __copy__(self):
        raise TypeError("R06 full-MDP Reward cycle tokens cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("R06 full-MDP Reward cycle tokens cannot be copied")

    def __reduce__(self):
        raise TypeError("R06 full-MDP Reward cycle tokens cannot be serialized")


@dataclass(frozen=True)
class _FullMdpRewardClosePayload:
    owner_identity: object
    cycle_identity: object
    control_step: int
    pre_reward_publication: object
    runtime_owner: object
    ordered_payment_verdicts: tuple[
        LandingOutcomeFullMdpRewardPaymentVerdict, ...
    ]
    closed_rows: torch.Tensor


class LandingOutcomeFullMdpRewardCloseReceipt:
    """Opaque proof that R06 closed both payments and every closable row."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> "LandingOutcomeFullMdpRewardCloseReceipt":
        del cls
        raise TypeError("R06 full-MDP Reward close receipts are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("R06 full-MDP Reward close receipts are immutable")


@dataclass(frozen=True)
class _R06CheckpointLiveMutationPayload:
    owner_ref: weakref.ReferenceType[
        "ActionBallLandingOutcomeDeviceCoordinator"
    ]
    source_global_receipt_ref: weakref.ReferenceType[object]
    mutation_version: int
    update_index: int
    drain_sequence: int
    live_tensor_receipts: tuple[tuple[str, tuple[object, ...]], ...]


class R06CheckpointLiveMutationProjection:
    """Opaque one-process handle over R06's latest exact ACKed live epoch.

    This handle is not the public R10 value.  The R10 callback validates its
    owner-private registry row and current tensor epoch, then emits the common
    :class:`PpoDrainLeafLiveMutationProjection`.  Equality of caller-authored
    fields is therefore never authority.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> "R06CheckpointLiveMutationProjection":
        del cls
        raise TypeError("R06 checkpoint live projections are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("R06 checkpoint live projections are immutable")

    def __copy__(self):
        raise TypeError("R06 checkpoint live projections cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("R06 checkpoint live projections cannot be copied")

    def __reduce__(self):
        raise TypeError("R06 checkpoint live projections cannot be serialized")


_FULL_MDP_REWARD_PAYMENT_REGISTRY: weakref.WeakKeyDictionary[
    LandingOutcomeFullMdpRewardPaymentVerdict,
    _FullMdpRewardPaymentPayload,
] = weakref.WeakKeyDictionary()
_FULL_MDP_REWARD_CYCLE_REGISTRY: weakref.WeakKeyDictionary[
    LandingOutcomeFullMdpRewardCycleToken,
    _FullMdpRewardCyclePayload,
] = weakref.WeakKeyDictionary()
_FULL_MDP_REWARD_CLOSE_REGISTRY: weakref.WeakKeyDictionary[
    LandingOutcomeFullMdpRewardCloseReceipt,
    _FullMdpRewardClosePayload,
] = weakref.WeakKeyDictionary()
_R06_CHECKPOINT_LIVE_MUTATION_REGISTRY: weakref.WeakKeyDictionary[
    R06CheckpointLiveMutationProjection,
    _R06CheckpointLiveMutationPayload,
] = weakref.WeakKeyDictionary()


def _owned_full_mdp_reward_payment(
    value: object,
) -> _FullMdpRewardPaymentPayload:
    try:
        payload = _FULL_MDP_REWARD_PAYMENT_REGISTRY.get(value)
    except TypeError:
        payload = None
    if type(value) is not LandingOutcomeFullMdpRewardPaymentVerdict or payload is None:
        raise LandingOutcomeDeviceError(
            "R06 full-MDP Reward payment verdict is forged, stale, or foreign"
        )
    return payload


def _owned_full_mdp_reward_close(
    value: object,
) -> _FullMdpRewardClosePayload:
    try:
        payload = _FULL_MDP_REWARD_CLOSE_REGISTRY.get(value)
    except TypeError:
        payload = None
    if type(value) is not LandingOutcomeFullMdpRewardCloseReceipt or payload is None:
        raise LandingOutcomeDeviceError(
            "R06 full-MDP Reward close receipt is forged, stale, or foreign"
        )
    return payload


@dataclass(frozen=True)
class LandingOutcomeBoundaryReceipt:
    schema_version: int
    update_index: int
    drain_sequence: int
    mutation_version: int
    fault_counts: tuple[int, ...]
    flight_state_counts: tuple[int, ...]
    mailbox_state_counts: tuple[int, ...]
    invariant_counts: tuple[int, ...]
    installed_total: int
    settled_total: int
    retired_total: int
    common_payment_total: int
    placement_payment_total: int
    closed_total: int
    checkpoint_safe: bool
    device_to_host_transfers: int
    runtime_integrated: bool
    cuda_profiled: bool
    formal_exact_resume_integrated: bool
    launch_authorized: bool


@dataclass(frozen=True)
class _PolicyScoreGrid:
    """C04 outputs scattered only onto policy-valid settlement rows."""

    drain_fault: torch.Tensor
    reason_code: torch.Tensor
    on_opponent_table: torch.Tensor
    placement_error_m: torch.Tensor
    broad_kernel: torch.Tensor
    narrow_kernel: torch.Tensor
    blended_kernel: torch.Tensor
    table_gate: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class ActionEpochR06PostPhysicsResult:
    """Typed direct-lane telemetry; R06 privately owns the retire decision."""

    accepted: torch.Tensor
    rejected: torch.Tensor
    fault_bits: torch.Tensor
    settled_mask: torch.Tensor
    settlement_cause: torch.Tensor
    new_valid_contact_mask: torch.Tensor
    observation_ordinal: torch.Tensor
    mutation_version: torch.Tensor
    flight_slot: torch.Tensor


@dataclass(frozen=True)
class ActionEpochR06PostPhysicsSample:
    """Minimal per-substep stop/continuity verdict before control finalize."""

    accepted: torch.Tensor
    rejected: torch.Tensor
    settled_mask: torch.Tensor
    flight_slot: torch.Tensor


@dataclass(frozen=True)
class ActionEpochR06RetireResult:
    """Typed rows actually retired by R06's private direct-lane decision."""

    retired_mask: torch.Tensor
    mailbox_retired_mask: torch.Tensor
    flight_slot: torch.Tensor
    mutation_version: torch.Tensor


def _boundary_receipt_payload(
    receipt: LandingOutcomeBoundaryReceipt,
) -> dict[str, object]:
    if not isinstance(receipt, LandingOutcomeBoundaryReceipt):
        raise LandingOutcomeDeviceError("checkpoint boundary receipt type differs")
    result: dict[str, object] = {}
    for field in fields(receipt):
        value = getattr(receipt, field.name)
        result[field.name] = list(value) if isinstance(value, tuple) else value
    return result


def _tensor_manifest(tensors: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(tensors, Mapping):
        raise LandingOutcomeDeviceError("checkpoint tensors must be a mapping")
    result: dict[str, object] = {}
    for name in sorted(tensors):
        tensor = tensors[name]
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise LandingOutcomeDeviceError("checkpoint tensor manifest differs")
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return result


def _tensor_bytes_sha256(
    tensors: Mapping[str, object],
    *,
    expected_mutation_version: int | None = None,
) -> str:
    """Hash one sorted packed byte tensor with exactly one cold-path D2H."""

    if not isinstance(tensors, Mapping) or not tensors:
        raise LandingOutcomeDeviceError("checkpoint tensors must be non-empty")
    if expected_mutation_version is not None and (
        isinstance(expected_mutation_version, bool)
        or not isinstance(expected_mutation_version, int)
        or expected_mutation_version < 0
    ):
        raise LandingOutcomeDeviceError("expected mutation version differs")
    byte_rows: list[torch.Tensor] = []
    device: torch.device | None = None
    mutation_byte_range: tuple[int, int] | None = None
    byte_cursor = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise LandingOutcomeDeviceError("checkpoint tensor bytes differ")
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise LandingOutcomeDeviceError("checkpoint tensors span devices")
        byte_row = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        byte_rows.append(byte_row)
        next_cursor = byte_cursor + byte_row.numel()
        if name == "mutation_version":
            if tensor.shape != torch.Size([]) or tensor.dtype != torch.int64:
                raise LandingOutcomeDeviceError(
                    "checkpoint mutation-version tensor differs"
                )
            mutation_byte_range = (byte_cursor, next_cursor)
        byte_cursor = next_cursor
    packed = torch.cat(byte_rows, dim=0)
    host_bytes = bytes(packed.to(device="cpu").tolist())
    if expected_mutation_version is not None:
        if mutation_byte_range is None:
            raise LandingOutcomeDeviceError(
                "checkpoint mutation-version tensor is missing"
            )
        start, stop = mutation_byte_range
        actual_mutation_version = int.from_bytes(
            host_bytes[start:stop],
            byteorder=sys.byteorder,
            signed=True,
        )
        if actual_mutation_version != expected_mutation_version:
            raise LandingOutcomeDeviceError(
                "checkpoint receipt is stale after mutation"
            )
    return hashlib.sha256(host_bytes).hexdigest()


_CHECKPOINT_CONTENT_FIELDS = (
    "schema_version",
    "num_envs",
    "flight_slot_capacity",
    "mailbox_capacity",
    "dtype",
    "profile_payload",
    "runtime_binding",
    "payment_authority",
    "receipt",
    "mutation_version",
    "drain_sequence",
    "last_drained_update_index",
    "drain_protocol",
    "global_drain_adopted",
    "tensor_manifest",
    "tensor_bytes_sha256",
)


def _checkpoint_content_sha256(checkpoint: Mapping[str, object]) -> str:
    """Canonical root over metadata, boundary identity, manifest, and bytes root."""

    if not isinstance(checkpoint, Mapping):
        raise LandingOutcomeDeviceError("checkpoint must be a mapping")
    missing = set(_CHECKPOINT_CONTENT_FIELDS) - set(checkpoint)
    if missing:
        raise LandingOutcomeDeviceError(
            f"checkpoint content fields missing: {sorted(missing)!r}"
        )
    payload = {name: checkpoint[name] for name in _CHECKPOINT_CONTENT_FIELDS}
    payload["receipt"] = _boundary_receipt_payload(checkpoint["receipt"])
    try:
        return _canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise LandingOutcomeDeviceError("checkpoint content is not canonical JSON") from exc


def _expand(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    return mask.reshape(mask.shape + (1,) * (ndim - mask.ndim))


def _masked_copy_(destination: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> None:
    destination.copy_(torch.where(_expand(mask, destination.ndim), source, destination))


def _masked_copy_distinct_(
    destination: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Copy from distinct storage without materializing a ``where`` result.

    ``torch.where(..., out=destination)`` is safe when the selected source does
    not alias the destination.  Keep that ownership premise executable instead
    of silently applying the fast path to the generic helper: callers with a
    shared backing storage must continue to use :func:`_masked_copy_`.
    """

    if destination.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
        raise LandingOutcomeDeviceError(
            "distinct masked copy source must not alias destination storage"
        )
    torch.where(
        _expand(mask, destination.ndim),
        source,
        destination,
        out=destination,
    )


def _masked_fill_(destination: torch.Tensor, mask: torch.Tensor, value: int | float) -> None:
    destination.masked_fill_(_expand(mask, destination.ndim), value)


def _stamp_less_fields(
    left_control: torch.Tensor,
    left_substep: torch.Tensor,
    left_phase: torch.Tensor,
    right_control: torch.Tensor,
    right_substep: torch.Tensor,
    right_phase: torch.Tensor,
) -> torch.Tensor:
    return (left_control < right_control) | (
        (left_control == right_control)
        & (
            (left_substep < right_substep)
            | ((left_substep == right_substep) & (left_phase < right_phase))
        )
    )


def _stamp_less(left: PhysicsStampBatch, right: PhysicsStampBatch) -> torch.Tensor:
    return _stamp_less_fields(
        left.control_step,
        left.physics_substep,
        left.event_phase,
        right.control_step,
        right.physics_substep,
        right.event_phase,
    )


def _stamp_less_equal(left: PhysicsStampBatch, right: PhysicsStampBatch) -> torch.Tensor:
    equal = (
        (left.control_step == right.control_step)
        & (left.physics_substep == right.physics_substep)
        & (left.event_phase == right.event_phase)
    )
    return _stamp_less(left, right) | equal


def _diagnostic_n2_env_cfg_family(env_cfg: object) -> str:
    """Resolve A/C from the resident exact registered leaf type, never a field."""

    cfg_type = type(env_cfg)
    cfg_module = sys.modules.get(cfg_type.__module__)
    try:
        cfg_source = inspect.getsourcefile(cfg_type)
    except (OSError, TypeError):
        cfg_source = None
    roles = {
        "HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg": "A",
        "HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg": "C",
    }
    expected_module = (
        "whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg"
    )
    resolved = roles.get(cfg_type.__name__)
    family_resolver = None if cfg_module is None else getattr(
        cfg_module, "action_ball_full_mdp_family_role", None
    )
    if (
        resolved is None
        or cfg_type.__module__ != expected_module
        or cfg_module is None
        or getattr(cfg_module, cfg_type.__name__, None) is not cfg_type
        or cfg_source is None
        or Path(cfg_source).name != "hope_env_cfg.py"
        or not callable(family_resolver)
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 requires one exact registered full-MDP EnvCfg leaf"
        )
    try:
        owner_resolved = family_resolver(env_cfg)
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 EnvCfg family authority rejected"
        ) from exc
    if owner_resolved != resolved:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 EnvCfg family authority differs"
        )
    return resolved


def _diagnostic_n2_profile(scene_spec: object) -> LandingPlacementProfile:
    """Derive the disposable scoring profile from exact code-owned geometry."""

    try:
        import action_ball_full_mdp_canary_target_profile as canary
        from whole_body_tracking.tasks.table_tennis import geometry
        from whole_body_tracking.tasks.table_tennis import table_frame
        from whole_body_tracking.tasks.tracking.mdp.continuous_questions import (
            ContinuousQuestionCfg,
        )
        from whole_body_tracking.tasks.tracking.mdp.hope_commands import (
            RacketTargetCommandCfg,
        )
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 code-owned target/geometry sources are unavailable"
        ) from exc
    cq_cfg = ContinuousQuestionCfg()
    racket_cfg = RacketTargetCommandCfg()
    x_range = tuple(cq_cfg.aim_x_range)
    y_range = tuple(cq_cfg.aim_y_range)
    near_x = float(racket_cfg.vb_table_near_x)
    surface_z = float(racket_cfg.vb_table_surface_z)
    half_width = float(geometry.TABLE_WIDTH) / 2.0
    net_x = near_x + float(geometry.NET_X)
    far_x = near_x + float(geometry.TABLE_LENGTH)
    translation = table_frame.env_frame_offset(near_x, surface_z)
    expected_translation = (near_x, half_width, surface_z)
    radius = float(scene_spec.ball_radius_m)
    if (
        canary.DIAGNOSTIC_UNAUTHORIZED is not True
        or canary.CANARY_SAVE_CHECKPOINTS is not False
        or radius != float(geometry.BALL_RADIUS)
        or translation != expected_translation
        or not (
            near_x < net_x < float(x_range[0]) < float(x_range[1]) <= far_x
            and -half_width
            <= float(y_range[0])
            < float(y_range[1])
            <= half_width
        )
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 code-owned scene/profile binding differs"
        )
    broad_scale = math.hypot(
        float(x_range[1]) - float(x_range[0]),
        float(y_range[1]) - float(y_range[0]),
    ) / 2.0
    narrow_scale = 2.0 * radius
    frame_binding_sha256 = _canonical_sha256(
        {
            "kind": "action_ball_r06_diagnostic_n2_frame_binding_v1",
            "authorization": "diagnostic_code_provenance_not_physical_proof",
            "frame_id": canary.FRAME_ID,
            "hope_to_tracking_translation_m": translation,
            "table_length_m": float(geometry.TABLE_LENGTH),
            "table_width_m": float(geometry.TABLE_WIDTH),
            "table_surface_z_m": surface_z,
            "net_x_m": net_x,
            "net_height_m": float(geometry.NET_HEIGHT),
            "aim_x_range_m": x_range,
            "aim_y_range_m": y_range,
        }
    )
    return LandingPlacementProfile(
        frame_id=canary.FRAME_ID,
        frame_binding_sha256=frame_binding_sha256,
        contact_source_semantics=SELECTED_RUBBER_CONTACT_AUTHORITY,
        table_surface_z_m=surface_z,
        ball_radius_m=radius,
        ball_center_landing_plane_z_m=surface_z + radius,
        net_x_m=net_x,
        net_mesh_top_z_m=surface_z + float(geometry.NET_HEIGHT),
        ball_center_net_clear_z_m=(
            surface_z + float(geometry.NET_HEIGHT) + radius
        ),
        opponent_table_x_min_m=net_x,
        opponent_table_x_max_m=far_x,
        table_y_min_m=-half_width,
        table_y_max_m=half_width,
        alpha_broad=0.5,
        sigma_broad_m=broad_scale,
        sigma_narrow_m=narrow_scale,
        on_table_gate=1.0,
        off_table_gate=0.5,
    )


def _require_direct_row_tensor(
    owner: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    """Read a construction field directly and reject aliased row views."""

    try:
        value = vars(owner)[name]
    except (KeyError, TypeError) as exc:
        raise LandingOutcomeDeviceError(
            f"diagnostic R06 {label} direct storage is unavailable"
        ) from exc
    if (
        type(value) is not torch.Tensor
        or tuple(value.shape) != shape
        or value.dtype is not dtype
        or value.device != device
        or value.layout is not torch.strided
        or not value.is_contiguous()
        or (shape[0] > 1 and value.stride(0) <= 0)
    ):
        raise LandingOutcomeDeviceError(
            f"diagnostic R06 {label} direct storage rows differ or alias"
        )
    return value


def _diagnostic_live_cardinality_anchors(
    *,
    env: object,
    physical_owner: object,
    epoch_owner: object | None = None,
) -> tuple[int, object, object]:
    """Join exact owners and independently allocated scene rows while cold."""

    physical_type = type(physical_owner)
    physical_module = sys.modules.get(physical_type.__module__)
    try:
        physical_source = inspect.getsourcefile(physical_type)
    except (OSError, TypeError):
        physical_source = None
    if (
        physical_type.__name__ != "ActionBallPhysicalFlightDeviceOwner"
        or physical_module is None
        or getattr(physical_module, physical_type.__name__, None) is not physical_type
        or physical_source is None
        or Path(physical_source).name != "action_ball_physical_flight_device.py"
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 cardinality requires the exact Physical owner"
        )
    physical_fields = vars(physical_owner)
    scene_port = physical_fields.get("scene_port")
    scene_type = type(scene_port)
    scene_module = sys.modules.get(scene_type.__module__)
    try:
        scene_source = inspect.getsourcefile(scene_type)
    except (OSError, TypeError):
        scene_source = None
    if (
        scene_type.__name__ != "IsaacLabPhysicalFlightScenePort"
        or scene_module is None
        or getattr(scene_module, scene_type.__name__, None) is not scene_type
        or scene_source is None
        or Path(scene_source).name != "action_ball_full_mdp_ball_scene.py"
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 cardinality requires the exact Isaac scene port"
        )

    env_num_envs = getattr(env, "num_envs", None)
    physical_num_envs = physical_fields.get("num_envs")
    scene_fields = vars(scene_port)
    scene_num_envs = scene_fields.get("num_envs")
    for label, value in (
        ("env", env_num_envs),
        ("Physical owner", physical_num_envs),
        ("Physical scene", scene_num_envs),
    ):
        if type(value) is not int or value <= 0:
            raise LandingOutcomeDeviceError(
                f"diagnostic R06 {label} num_envs must be an exact positive integer"
            )
    if env_num_envs != physical_num_envs or env_num_envs != scene_num_envs:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 live Env/Physical/scene num_envs differ"
        )
    num_envs = env_num_envs
    device = physical_fields.get("device")
    try:
        env_device = torch.device(getattr(env, "device"))
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 Env device is unavailable"
        ) from exc
    if (
        type(device) is not torch.device
        or env_device != device
        or scene_fields.get("device") != device
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 Env/Physical/scene device differs"
        )
    flight_capacity = DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY
    state_width = getattr(physical_module, "STATE_WIDTH", None)
    if state_width != 13:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 Physical scene-state width differs"
        )

    env_scene = getattr(env, "scene", None)
    env_origins = _require_direct_row_tensor(
        scene_port,
        name="env_origins",
        shape=(num_envs, 3),
        dtype=torch.float32,
        device=device,
        label="scene env_origins",
    )
    all_env_ids = _require_direct_row_tensor(
        scene_port,
        name="_all_env_ids",
        shape=(num_envs,),
        dtype=torch.int64,
        device=device,
        label="scene _all_env_ids",
    )
    if getattr(env_scene, "env_origins", None) is not env_origins:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 Env and Physical scene origin objects differ"
        )
    assets = scene_fields.get("assets")
    if type(assets) is not tuple or len(assets) != flight_capacity:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 scene retained asset grid differs"
        )
    asset_root_storage: list[tuple[int, int, int, int]] = []
    for index, asset in enumerate(assets):
        data = getattr(asset, "data", None)
        root_state = getattr(data, "root_state_w", None)
        if (
            type(root_state) is not torch.Tensor
            or tuple(root_state.shape) != (num_envs, state_width)
            or root_state.dtype is not torch.float32
            or root_state.device != device
            or root_state.layout is not torch.strided
            or not root_state.is_contiguous()
            or (num_envs > 1 and root_state.stride(0) <= 0)
        ):
            raise LandingOutcomeDeviceError(
                f"diagnostic R06 scene asset {index} direct root rows differ or alias"
            )
        storage_pointer = root_state.untyped_storage().data_ptr()
        byte_start = root_state.storage_offset() * root_state.element_size()
        byte_end = byte_start + root_state.numel() * root_state.element_size()
        for prior_pointer, prior_start, prior_end, prior_index in asset_root_storage:
            if (
                storage_pointer == prior_pointer
                and byte_start < prior_end
                and prior_start < byte_end
            ):
                raise LandingOutcomeDeviceError(
                    "diagnostic R06 scene asset "
                    f"{index} root storage overlaps asset {prior_index}"
                )
        asset_root_storage.append((storage_pointer, byte_start, byte_end, index))

    capability = scene_fields.get("_scene_port_capability")
    if (
        capability is None
        or physical_fields.get("_scene_port_capability") is not capability
        or getattr(capability, "num_envs", None) != num_envs
        or getattr(capability, "flight_capacity", None) != flight_capacity
        or getattr(capability, "_port_identity", None)
        is not scene_fields.get("_identity")
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 Physical retained scene capability differs"
        )

    if epoch_owner is not None:
        epoch_type = type(epoch_owner)
        epoch_module = sys.modules.get(epoch_type.__module__)
        try:
            epoch_source = inspect.getsourcefile(epoch_type)
        except (OSError, TypeError):
            epoch_source = None
        if (
            epoch_type.__name__ != "ActionEpochOwner"
            or epoch_module is None
            or getattr(epoch_module, epoch_type.__name__, None) is not epoch_type
            or epoch_source is None
            or Path(epoch_source).name != "action_ball_full_mdp_epoch.py"
        ):
            raise LandingOutcomeDeviceError(
                "diagnostic R06 cardinality requires the exact ActionEpoch owner"
            )
        epoch_num_envs = getattr(epoch_owner, "num_envs", None)
        if type(epoch_num_envs) is not int or epoch_num_envs <= 0:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 ActionEpoch num_envs must be an exact positive integer"
            )
        if epoch_num_envs != num_envs:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 live Env/Physical/ActionEpoch num_envs differ"
            )
        if physical_fields.get("_action_epoch_owner") is not epoch_owner:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 requires Physical's exact bound ActionEpoch owner"
            )
        if getattr(epoch_owner, "device", None) != device:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 ActionEpoch device differs"
            )
    return num_envs, scene_port, capability


def construct_diagnostic_n2_no_save_r06(
    *,
    env: object,
    physical_owner: object,
    diagnostic_n2_capacity_binding: object,
) -> "ActionBallLandingOutcomeDeviceCoordinator":
    """Construct the real R06 type against the same diagnostic Physical scene.

    Shape, device, dtype, A/C treatment and text/profile facts are derived
    here.  No caller numeric parameter and no formal capacity/payment receipt
    is accepted.  The returned owner is diagnostic-unauthorized and cannot
    checkpoint, restore, export, or project R10 state.
    """

    env_cfg = getattr(env, "cfg", None)
    family = _diagnostic_n2_env_cfg_family(env_cfg)
    try:
        device = torch.device(getattr(env, "device"))
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 env device must be explicit"
        ) from exc
    scene_spec = getattr(env_cfg, "action_ball_full_mdp_ball_scene_spec", None)
    if (
        getattr(env_cfg, "action_ball_full_mdp_scene_capacity", None)
        != DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY
        or getattr(env_cfg, "action_ball_full_mdp_capacity_receipt_sha256", None)
        != ""
        or getattr(env_cfg, "checkpoint_path", None) is not None
        or getattr(env_cfg, "checkpoint_tolerant", False) is not False
        or getattr(env, "full_mdp_cold_restore_dormant", False) is not False
        or getattr(env, "_action_ball_r10_cold_restore_capsule", None) is not None
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 requires the exact no-save environment state"
        )
    try:
        _diagnostic_capacity.require_diagnostic_n2_capacity_binding(
            diagnostic_n2_capacity_binding,
            scene_spec=scene_spec,
        )
    except Exception as exc:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 capacity binding is stale or foreign"
        ) from exc
    num_envs = getattr(env, "num_envs", None)
    if type(num_envs) is not int or num_envs <= 0:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 env num_envs must be an exact positive integer"
        )
    live_cardinality = _diagnostic_live_cardinality_anchors(
        env=env,
        physical_owner=physical_owner,
    )
    if live_cardinality[0] != num_envs:
        raise LandingOutcomeDeviceError(
            "diagnostic R06 live cardinality changed during construction"
        )
    scene_port = live_cardinality[1]
    physical_fields = vars(physical_owner)
    if (
        physical_fields.get("_diagnostic_n2_no_save") is not True
        or hasattr(physical_owner, "capacity_receipt")
        or hasattr(physical_owner, "capacity_receipt_sha256")
        or physical_fields.get("flight_capacity")
        != DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY
        or getattr(scene_port, "flight_capacity", None)
        != DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY
        or getattr(scene_port, "spec", None) is not scene_spec
        or physical_fields.get("_r06_owner") is not None
    ):
        raise LandingOutcomeDeviceError(
            "diagnostic R06 requires the same unbound diagnostic Physical scene"
        )
    profile = _diagnostic_n2_profile(scene_spec)
    text_registry = LandingOutcomeTextRegistry(
        run_ids=(DIAGNOSTIC_N2_NO_SAVE_RUN_IDS[family],),
        carry_chain_ids=(DIAGNOSTIC_N2_NO_SAVE_CARRY_CHAIN_IDS[family],),
    )
    diagnostic_family_gain_identity_sha256 = _canonical_sha256(
        {
            "kind": "action_ball_r06_diagnostic_n2_payment_identity_v1",
            "authorization": "diagnostic_only_not_formal_c10_authority",
            "env_cfg_type": type(env_cfg).__name__,
            "family": family,
            "placement_treatment_gain": 1.0 if family == "A" else 0.0,
            "profile_sha256": profile.canonical_sha256,
        }
    )
    record = _DiagnosticN2NoSaveConstructionRecord(
        env=env,
        env_cfg=env_cfg,
        physical_owner=physical_owner,
        scene_port=live_cardinality[1],
        scene_port_capability=live_cardinality[2],
        scene_spec=scene_spec,
        diagnostic_capacity_binding=diagnostic_n2_capacity_binding,
        num_envs=num_envs,
        family=family,
        device=device,
        profile=profile,
        text_registry=text_registry,
        diagnostic_family_gain_identity_sha256=(
            diagnostic_family_gain_identity_sha256
        ),
    )
    pending_key = id(profile)
    with _DIAGNOSTIC_N2_REGISTRY_LOCK:
        if pending_key in _DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 construction profile identity collided"
            )
        _DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS[pending_key] = (
            _issue_diagnostic_n2_no_save_construction_binding(record)
        )
        try:
            owner = ActionBallLandingOutcomeDeviceCoordinator(
                num_envs=num_envs,
                flight_slot_capacity=DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY,
                mailbox_capacity=DIAGNOSTIC_N2_NO_SAVE_MAILBOX_CAPACITY,
                device=device,
                dtype=torch.float32,
                profile=profile,
                runtime_binding=None,
                payment_authority=None,
                capacity_authority=None,
                text_registry=text_registry,
            )
        finally:
            _DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS.pop(pending_key, None)
    return owner


class ActionBallLandingOutcomeDeviceCoordinator:
    """Two-grid, fixed-cadence, family-neutral landing authority."""

    def __init__(
        self,
        *,
        num_envs: int,
        flight_slot_capacity: int,
        mailbox_capacity: int,
        device: torch.device | str,
        dtype: torch.dtype,
        profile: LandingPlacementProfile,
        runtime_binding: LandingOutcomeRuntimeBinding,
        payment_authority: LandingOutcomeC10FamilyPaymentAuthority,
        capacity_authority: LandingOutcomeCapacityAuthority,
        text_registry: LandingOutcomeTextRegistry,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise LandingOutcomeDeviceError("num_envs must be a positive integer")
        for name, value in (
            ("flight_slot_capacity", flight_slot_capacity),
            ("mailbox_capacity", mailbox_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LandingOutcomeDeviceError(f"{name} must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise LandingOutcomeDeviceError("dtype must be an explicit floating torch dtype")
        if not isinstance(profile, LandingPlacementProfile):
            raise LandingOutcomeDeviceError("profile must be explicit LandingPlacementProfile")
        if type(text_registry) is not LandingOutcomeTextRegistry:
            raise LandingOutcomeDeviceError("text_registry must be the exact sealed type")
        diagnostic_record = None
        diagnostic_binding = None
        if (
            runtime_binding is None
            and payment_authority is None
            and capacity_authority is None
        ):
            with _DIAGNOSTIC_N2_REGISTRY_LOCK:
                diagnostic_binding = _DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS.pop(
                    id(profile), None
                )
        if diagnostic_binding is not None:
            diagnostic_record = _owned_diagnostic_n2_no_save_construction_binding(
                diagnostic_binding
            )
        if diagnostic_record is None:
            if not isinstance(runtime_binding, LandingOutcomeRuntimeBinding):
                raise LandingOutcomeDeviceError("runtime_binding must be explicit")
            payment_payload = _owned_c10_payment_authority(payment_authority)
            capacity_payload = _owned_capacity_authority(capacity_authority)
            if runtime_binding.landing_profile_sha256 != profile.canonical_sha256:
                raise LandingOutcomeDeviceError("runtime/C04 profile pin differs")
        else:
            if (
                runtime_binding is not None
                or payment_authority is not None
                or capacity_authority is not None
                or num_envs != diagnostic_record.num_envs
                or flight_slot_capacity != DIAGNOSTIC_N2_NO_SAVE_FLIGHT_CAPACITY
                or mailbox_capacity != DIAGNOSTIC_N2_NO_SAVE_MAILBOX_CAPACITY
                or torch.device(device) != diagnostic_record.device
                or dtype is not torch.float32
                or profile is not diagnostic_record.profile
                or text_registry is not diagnostic_record.text_registry
            ):
                raise LandingOutcomeDeviceError(
                    "diagnostic R06 construction facts differ from the sealed binding"
                )
            payment_payload = {
                "family": diagnostic_record.family,
                "placement_treatment_gain": (
                    1.0 if diagnostic_record.family == "A" else 0.0
                ),
                "projection_sha256": (
                    DIAGNOSTIC_N2_NO_FORMAL_C10_PROJECTION_SHA256
                ),
            }
            capacity_payload = None
        authority_pairs = () if diagnostic_record is not None else (
            (
                "common_on_table_manager_weight",
                payment_payload["common_on_table_manager_weight"],
                runtime_binding.common_on_table_manager_weight,
            ),
            (
                "placement_manager_shell_weight",
                payment_payload["placement_manager_shell_weight"],
                runtime_binding.placement_manager_shell_weight,
            ),
            (
                "post_dt_budget_sha256",
                payment_payload["post_dt_budget_sha256"],
                runtime_binding.post_dt_budget_sha256,
            ),
            (
                "landing_profile_sha256",
                payment_payload["landing_profile_sha256"],
                runtime_binding.landing_profile_sha256,
            ),
            (
                "c10_contract_sha256",
                payment_payload["contract_sha256"],
                runtime_binding.c10_contract_sha256,
            ),
            (
                "c10_projection_sha256",
                payment_payload["projection_sha256"],
                runtime_binding.c10_projection_sha256,
            ),
            (
                "c10_identity_sha256",
                payment_payload["identity_sha256"],
                runtime_binding.c10_identity_sha256,
            ),
        )
        if diagnostic_record is None:
            for name, authority_value, runtime_value in authority_pairs:
                if authority_value != runtime_value:
                    raise LandingOutcomeDeviceError(f"runtime/C10 {name} differs")
        capacity_pairs = () if diagnostic_record is not None else (
            (
                "capacity_authority_sha256",
                capacity_authority.canonical_sha256,
                runtime_binding.capacity_authority_sha256,
            ),
            (
                "numeric_materialization_sha256",
                capacity_payload["materialization_sha256"],
                runtime_binding.numeric_materialization_sha256,
            ),
            (
                "numeric_authority_sha256",
                capacity_payload["numeric_authority_sha256"],
                runtime_binding.numeric_authority_sha256,
            ),
            (
                "resolved_graph_receipt_sha256",
                capacity_payload["resolved_graph_receipt_sha256"],
                runtime_binding.resolved_graph_receipt_sha256,
            ),
            (
                "four_shot_tape_receipt_sha256",
                capacity_payload["four_shot_tape_receipt_sha256"],
                runtime_binding.four_shot_tape_receipt_sha256,
            ),
            (
                "control_step_clock_sha256",
                capacity_payload["control_step_clock_sha256"],
                runtime_binding.control_step_clock_sha256,
            ),
            (
                "inclusive_event_order_witness_sha256",
                capacity_payload["inclusive_event_order_witness_sha256"],
                runtime_binding.inclusive_event_order_witness_sha256,
            ),
            (
                "flight_horizon_witness_sha256",
                capacity_payload["flight_horizon_witness_sha256"],
                runtime_binding.flight_horizon_witness_sha256,
            ),
            (
                "mailbox_horizon_witness_sha256",
                capacity_payload["mailbox_horizon_witness_sha256"],
                runtime_binding.mailbox_horizon_witness_sha256,
            ),
            (
                "cadence_ticks",
                capacity_payload["cadence_ticks"],
                runtime_binding.cadence_ticks,
            ),
            (
                "flight_horizon_ticks",
                capacity_payload["flight_horizon_ticks"],
                runtime_binding.flight_horizon_ticks,
            ),
            (
                "mailbox_horizon_ticks",
                capacity_payload["mailbox_horizon_ticks"],
                runtime_binding.mailbox_horizon_ticks,
            ),
            (
                "tail_closure_tick",
                capacity_payload["tail_closure_tick"],
                runtime_binding.tail_closure_tick,
            ),
        )
        if diagnostic_record is None:
            for name, authority_value, runtime_value in capacity_pairs:
                if authority_value != runtime_value:
                    raise LandingOutcomeDeviceError(f"runtime/R03 {name} differs")
            if (
                flight_slot_capacity != capacity_payload["flight_slot_capacity"]
                or flight_slot_capacity
                != capacity_payload["required_flight_slot_capacity"]
                or mailbox_capacity != capacity_payload["mailbox_capacity"]
                or mailbox_capacity < capacity_payload["required_mailbox_capacity"]
                or runtime_binding.flight_slot_capacity != flight_slot_capacity
                or runtime_binding.mailbox_capacity != mailbox_capacity
            ):
                raise LandingOutcomeDeviceError("runtime/R03 H/C/K capacity differs")
            if text_registry.canonical_sha256 != runtime_binding.text_registry_sha256:
                raise LandingOutcomeDeviceError("runtime text registry root differs")

        self.num_envs = num_envs
        self.flight_slot_capacity = flight_slot_capacity
        self.mailbox_capacity = mailbox_capacity
        self.device = torch.device(device)
        self.dtype = dtype
        self.profile = profile
        self.text_registry = text_registry
        self._diagnostic_n2_no_save = diagnostic_record is not None
        self._diagnostic_unauthorized = diagnostic_record is not None
        self._diagnostic_n2_construction_record = diagnostic_record
        self._diagnostic_family_gain_identity_sha256 = (
            None
            if diagnostic_record is None
            else diagnostic_record.diagnostic_family_gain_identity_sha256
        )
        if diagnostic_record is None:
            self.runtime_binding = runtime_binding
            self.payment_authority = payment_authority
            self.capacity_authority = capacity_authority
        self._placement_treatment_gain = float(
            payment_payload["placement_treatment_gain"]
        )
        self._c10_family_code = (
            C10_FAMILY_A if payment_payload["family"] == "A" else C10_FAMILY_C
        )
        self.consumers = CONSUMERS
        self.full_mdp_reward_consumers = CONSUMERS
        self._consumer_index = {name: index for index, name in enumerate(CONSUMERS)}
        self._flight_shape = (num_envs, flight_slot_capacity)
        self._mailbox_shape = (num_envs, mailbox_capacity)
        self._flight_slot_ids = torch.arange(
            flight_slot_capacity, dtype=torch.int64, device=self.device
        ).unsqueeze(0)
        self._mailbox_slot_ids = torch.arange(
            mailbox_capacity, dtype=torch.int64, device=self.device
        ).unsqueeze(0)
        self._env_ids = torch.arange(
            num_envs, dtype=torch.int64, device=self.device
        )
        self._registry_root_token = torch.tensor(
            list(bytes.fromhex(text_registry.canonical_sha256)),
            dtype=torch.uint8,
            device=self.device,
        )
        self._c10_projection_token = torch.tensor(
            list(bytes.fromhex(str(payment_payload["projection_sha256"]))),
            dtype=torch.uint8,
            device=self.device,
        )

        # Physical flight grid.
        self._flight_state = torch.full(
            self._flight_shape, FLIGHT_EMPTY, dtype=torch.int8, device=self.device
        )
        self._flight_physical_retired = torch.ones(
            self._flight_shape, dtype=torch.bool, device=self.device
        )
        self._flight_fault_bits = torch.zeros(
            self._flight_shape, dtype=torch.int64, device=self.device
        )
        # Distinguish the receipt-free ActionEpoch lane from the frozen formal
        # checkpoint lane.  This is lifecycle provenance, not authorization.
        self._flight_action_epoch = torch.zeros(
            self._flight_shape, dtype=torch.bool, device=self.device
        )
        # Receipt-free ActionEpoch identity plane.  These are D05-owned scalar
        # identities retained through Physical's one-shot projection.  They do
        # not alias, derive, or summarize any legacy digest field.
        self._flight_action_uid = self._filled_int(self._flight_shape, -1)
        self._flight_action_slot = self._filled_int(self._flight_shape, -1)
        self._flight_reset_generation = self._filled_int(self._flight_shape, -1)
        self._flight_shot_index = self._filled_int(self._flight_shape, -1)
        self._flight_task_identity = self._filled_int(self._flight_shape, -1)
        self._flight_outcome_identity = self._filled_int(self._flight_shape, -1)
        self._flight_ball_identity = self._filled_int(self._flight_shape, -1)
        self._flight_publication_ordinal = self._filled_int(self._flight_shape, -1)
        self._flight_key_ints = self._new_key_int_buffers(self._flight_shape)
        self._flight_key_digests = self._new_key_digest_buffers(self._flight_shape)
        self._flight_full_key_sha256 = self._zeros_token(self._flight_shape)
        self._flight_full_key_receipt_sha256 = self._zeros_token(self._flight_shape)
        self._flight_committed_reveal_sha256 = self._zeros_token(self._flight_shape)
        self._flight_install_receipt_sha256 = self._zeros_token(self._flight_shape)
        self._flight_ball_generation = self._filled_int(self._flight_shape, -1)
        self._flight_mailbox_slot = self._filled_int(self._flight_shape, -1)
        self._flight_task_identity_token = self._zeros_token(self._flight_shape)
        self._flight_target_xy_m = self._zeros_float(self._flight_shape + (2,))
        self._flight_reveal_control_step = self._filled_int(self._flight_shape, -1)
        self._flight_contact_deadline_control_step = self._filled_int(
            self._flight_shape, -1
        )
        self._flight_crossing_horizon_control_step = self._filled_int(
            self._flight_shape, -1
        )
        self._flight_contact_valid = torch.zeros(
            self._flight_shape, dtype=torch.bool, device=self.device
        )
        self._flight_contact_ball_center_m = self._zeros_float(
            self._flight_shape + (3,)
        )
        self._flight_outgoing_anchor_m = self._zeros_float(
            self._flight_shape + (3,)
        )
        self._flight_contact_stamp_control = self._filled_int(self._flight_shape, -1)
        self._flight_contact_stamp_substep = torch.full(
            self._flight_shape, -1, dtype=torch.int32, device=self.device
        )
        self._flight_observation_ordinal = self._filled_int(self._flight_shape, -1)
        self._flight_last_ball_center_m = self._zeros_float(
            self._flight_shape + (3,)
        )
        self._flight_last_observation_control = self._filled_int(self._flight_shape, -1)
        self._flight_last_observation_substep = torch.full(
            self._flight_shape, -1, dtype=torch.int32, device=self.device
        )
        self._flight_net_crossed = torch.zeros(
            self._flight_shape, dtype=torch.bool, device=self.device
        )
        self._flight_net_clear = torch.zeros_like(self._flight_net_crossed)
        self._flight_net_stamp_control = self._filled_int(self._flight_shape, -1)
        self._flight_net_stamp_substep = torch.full(
            self._flight_shape, -1, dtype=torch.int32, device=self.device
        )

        # Durable mailbox grid and its independent reservation plane.
        self._mailbox_state = torch.full(
            self._mailbox_shape, MAILBOX_EMPTY, dtype=torch.int8, device=self.device
        )
        self._mailbox_reserved = torch.zeros(
            self._mailbox_shape, dtype=torch.bool, device=self.device
        )
        self._mailbox_history_valid = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_reservation_token = self._zeros_token(self._mailbox_shape)
        self._mailbox_reservation_generation = self._filled_int(
            self._mailbox_shape, -1
        )
        self._mailbox_reserved_flight_slot = self._filled_int(
            self._mailbox_shape, -1
        )
        self._mailbox_fault_bits = torch.zeros(
            self._mailbox_shape, dtype=torch.int64, device=self.device
        )
        self._mailbox_action_epoch = torch.zeros(
            self._mailbox_shape, dtype=torch.bool, device=self.device
        )
        self._mailbox_action_uid = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_action_slot = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_reset_generation = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_shot_index = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_task_identity = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_outcome_identity = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_ball_identity = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_publication_ordinal = self._filled_int(
            self._mailbox_shape, -1
        )
        self._mailbox_key_ints = self._new_key_int_buffers(self._mailbox_shape)
        self._mailbox_key_digests = self._new_key_digest_buffers(self._mailbox_shape)
        self._mailbox_full_key_sha256 = self._zeros_token(self._mailbox_shape)
        self._mailbox_full_key_receipt_sha256 = self._zeros_token(
            self._mailbox_shape
        )
        self._mailbox_committed_reveal_sha256 = self._zeros_token(
            self._mailbox_shape
        )
        self._mailbox_install_receipt_sha256 = self._zeros_token(
            self._mailbox_shape
        )
        self._mailbox_ball_generation = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_observation_ordinal = self._filled_int(
            self._mailbox_shape, -1
        )
        self._mailbox_task_identity_token = self._zeros_token(self._mailbox_shape)
        self._mailbox_target_xy_m = self._zeros_float(self._mailbox_shape + (2,))
        self._mailbox_physical_retired = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_settlement_cause = torch.full(
            self._mailbox_shape,
            SETTLEMENT_CAUSE_NONE,
            dtype=torch.int8,
            device=self.device,
        )
        self._mailbox_canonical_reason_code = self._filled_int(
            self._mailbox_shape, CANONICAL_REASON_NOT_SCORED
        )
        self._mailbox_policy_eligible = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_source_control_step = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_source_physics_substep = torch.full(
            self._mailbox_shape, -1, dtype=torch.int32, device=self.device
        )
        self._mailbox_settlement_control_step = self._filled_int(
            self._mailbox_shape, -1
        )
        self._mailbox_settlement_physics_substep = torch.full(
            self._mailbox_shape, -1, dtype=torch.int32, device=self.device
        )
        self._mailbox_contact_valid = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_contact_ball_center_m = self._zeros_float(
            self._mailbox_shape + (3,)
        )
        self._mailbox_crossing_present = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_crossing_valid = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_crossing_nonfinite = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_crossing_xy_m = self._zeros_float(
            self._mailbox_shape + (2,)
        )
        self._mailbox_crossing_stamp_control = self._filled_int(
            self._mailbox_shape, -1
        )
        self._mailbox_crossing_stamp_substep = torch.full(
            self._mailbox_shape, -1, dtype=torch.int32, device=self.device
        )
        self._mailbox_net_crossed = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_net_clear = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_net_stamp_control = self._filled_int(self._mailbox_shape, -1)
        self._mailbox_net_stamp_substep = torch.full(
            self._mailbox_shape, -1, dtype=torch.int32, device=self.device
        )
        self._mailbox_common_on_table = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_on_opponent_table = torch.zeros_like(self._mailbox_reserved)
        self._mailbox_placement_error_m = self._zeros_float(self._mailbox_shape)
        self._mailbox_broad_kernel = self._zeros_float(self._mailbox_shape)
        self._mailbox_narrow_kernel = self._zeros_float(self._mailbox_shape)
        self._mailbox_blended_kernel = self._zeros_float(self._mailbox_shape)
        self._mailbox_table_gate = self._zeros_float(self._mailbox_shape)
        self._mailbox_canonical_total = self._zeros_float(self._mailbox_shape)
        self._mailbox_view_epoch = torch.full(
            self._mailbox_shape + (CONSUMER_COUNT,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._mailbox_paid_mask = torch.zeros(
            self._mailbox_shape, dtype=torch.int64, device=self.device
        )
        self._mailbox_payment_values = self._zeros_float(
            self._mailbox_shape + (CONSUMER_COUNT,)
        )
        self._mailbox_payment_epoch = torch.full_like(self._mailbox_view_epoch, -1)
        self._mailbox_placement_treatment_gain = torch.full(
            self._mailbox_shape, -1.0, dtype=self.dtype, device=self.device
        )
        self._mailbox_c10_family_code = torch.full(
            self._mailbox_shape, -1, dtype=torch.int8, device=self.device
        )
        self._mailbox_c10_projection_sha256 = self._zeros_token(
            self._mailbox_shape
        )
        self._mailbox_consumer_blocked = torch.zeros(
            self._mailbox_shape + (CONSUMER_COUNT,),
            dtype=torch.bool,
            device=self.device,
        )

        # Rejected ingress/post/lifecycle traffic never mutates owner bytes.
        self._ingress_fault_bits = torch.zeros(
            (num_envs,), dtype=torch.int64, device=self.device
        )
        self._post_fault_bits = torch.zeros(
            self._flight_shape, dtype=torch.int64, device=self.device
        )
        self._lifecycle_fault_bits = torch.zeros(
            (num_envs,), dtype=torch.int64, device=self.device
        )
        self._device_sticky_poison = torch.zeros(
            (num_envs,), dtype=torch.bool, device=self.device
        )

        # Durable replay is a reset-generation/swing-generation high-water pair.
        self._replay_valid = torch.zeros(
            (num_envs,), dtype=torch.bool, device=self.device
        )
        self._replay_reset_generation = torch.full(
            (num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._replay_swing_generation = torch.full_like(
            self._replay_reset_generation, -1
        )
        self._replay_action_epoch = torch.zeros_like(self._replay_valid)
        self._replay_action_uid = self._filled_int((num_envs,), -1)
        self._replay_action_slot = self._filled_int((num_envs,), -1)
        self._replay_shot_index = self._filled_int((num_envs,), -1)
        self._replay_task_identity = self._filled_int((num_envs,), -1)
        self._replay_outcome_identity = self._filled_int((num_envs,), -1)
        self._replay_ball_identity = self._filled_int((num_envs,), -1)
        self._replay_publication_ordinal = self._filled_int((num_envs,), -1)
        self._replay_full_key_sha256 = self._zeros_token((num_envs,))

        # Previous-shot policy history is not mailbox history.  Mailbox slots
        # are reusable, so accepted close_paid atomically copies the raw facts
        # into this dedicated per-environment after-image before releasing the
        # slot.  It survives later slot reuse and is reset only for selected
        # true-reset rows through the existing shadow transaction.
        self._previous_paid_valid = torch.zeros(
            (num_envs,), dtype=torch.bool, device=self.device
        )
        self._previous_paid_action_epoch = torch.zeros_like(
            self._previous_paid_valid
        )
        self._previous_paid_action_uid = self._filled_int((num_envs,), -1)
        self._previous_paid_action_slot = self._filled_int((num_envs,), -1)
        self._previous_paid_reset_generation = self._filled_int((num_envs,), -1)
        self._previous_paid_shot_index = self._filled_int((num_envs,), -1)
        self._previous_paid_task_identity = self._filled_int((num_envs,), -1)
        self._previous_paid_outcome_identity = self._filled_int((num_envs,), -1)
        self._previous_paid_ball_identity = self._filled_int((num_envs,), -1)
        self._previous_paid_publication_ordinal = self._filled_int(
            (num_envs,), -1
        )
        self._previous_paid_key_ints = self._new_key_int_buffers((num_envs,))
        self._previous_paid_key_digests = self._new_key_digest_buffers(
            (num_envs,)
        )
        self._previous_paid_full_key_sha256 = self._zeros_token((num_envs,))
        self._previous_paid_ball_generation = self._filled_int((num_envs,), -1)
        self._previous_paid_observation_ordinal = self._filled_int((num_envs,), -1)
        self._previous_paid_settlement_control_step = self._filled_int(
            (num_envs,), -1
        )
        self._previous_paid_payment_step = self._filled_int((num_envs,), -1)
        self._previous_paid_payment_step_highwater = self._filled_int(
            (num_envs,), -1
        )
        self._previous_paid_selected_contact = torch.zeros_like(
            self._previous_paid_valid
        )
        self._previous_paid_first_crossing_valid = torch.zeros_like(
            self._previous_paid_valid
        )
        self._previous_paid_on_opponent_table = torch.zeros_like(
            self._previous_paid_valid
        )
        self._previous_paid_target_error_m = self._zeros_float((num_envs,))
        self._previous_paid_target_xy_m = self._zeros_float((num_envs, 2))

        # Durable per-env reset chronology is independent of replay ownership.
        # A selected true reset clears the replay token but advances this
        # generation high-water on device before R05 publishes its exact ACK.
        self._reset_generation_highwater = torch.zeros(
            (num_envs,), dtype=torch.int64, device=self.device
        )
        self._selected_reset_count = torch.zeros_like(
            self._reset_generation_highwater
        )

        self._installed_total = self._zero_counter()
        self._settled_total = self._zero_counter()
        self._retired_total = self._zero_counter()
        self._payment_totals = torch.zeros(
            (CONSUMER_COUNT,), dtype=torch.int64, device=self.device
        )
        self._closed_total = self._zero_counter()
        self._selected_reset_retired_flight_total = self._zero_counter()
        self._selected_reset_closed_mailbox_total = self._zero_counter()
        self._selected_reset_retired_payment_totals = torch.zeros(
            (CONSUMER_COUNT,), dtype=torch.int64, device=self.device
        )
        self._terminal_resolution_total = self._zero_counter()
        self._shared_normal_retire_total = self._zero_counter()
        self._r06_only_orphan_retire_total = self._zero_counter()
        self._shared_normal_retire_key_summaries = torch.zeros(
            (2,), dtype=torch.int64, device=self.device
        )
        self._fault_event_counts = torch.zeros(
            (len(FAULTS),), dtype=torch.int64, device=self.device
        )

        self._mutation_version = self._zero_counter()
        self._drain_sequence = 0
        self._last_drained_update_index = -1
        self._latest_receipt: LandingOutcomeBoundaryReceipt | None = None
        self._latest_global_drain_receipt: object | None = None
        self._latest_checkpoint_live_mutation_projection: (
            R06CheckpointLiveMutationProjection | None
        ) = None
        self._r06_checkpoint_consumed_global_receipt: object | None = None
        self._latest_receipt_consumed = False
        self._load_fresh = True
        self._active_r06_global_drain: _PreparedR06GlobalDrain | None = None
        self._r06_global_drain_adopted = False
        self._r06_global_drain_poisoned = False
        self._r06_global_drain_poison_reason: str | None = None
        self._active_reveal_prepare_lease: _ActiveRevealPrepareLease | None = None
        self._reveal_boundary_child_token_authority = (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority(
                owner_kind="r06_flight",
                validator=self._require_owned_reveal_prepared_token,
            )
        )
        self._reveal_boundary_owner: (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner | None
        ) = None
        self._reveal_boundary_lane: (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority | None
        ) = None
        self._reveal_boundary_source_sha256: str | None = None
        self._r05_terminal_owner: (
            _r05.ContinuousRuntimeTransactionOwner | None
        ) = None
        self._r05_terminal_boundary_authority: (
            _r05.TerminalBoundaryAuthority | None
        ) = None
        self._r05_terminal_source_sha256: str | None = None
        self._device_r05_reset_owner: _r05_device.DeviceR05Owner | None = None
        self._device_r05_genesis_projection: object | None = None
        self._device_r05_prepared_reset_validator: object | None = None
        self._device_r05_receipt_validator: object | None = None
        self._device_r05_hot_reveal_owner: _r05_device.DeviceR05Owner | None = None
        self._device_r05_hot_reveal_projection_validator: object | None = None
        self._physical_late_launch_owner: object | None = None
        self._physical_late_launch_publication_validator: object | None = None
        self._active_device_r05_hot_reveal: object | None = None
        self._physical_park_token_authority: (
            LandingOutcomePhysicalParkTokenAuthority | None
        ) = None
        self._selected_reset_physical_park_token_authority: (
            LandingOutcomeSelectedResetPhysicalParkTokenAuthority | None
        ) = None
        self._latest_post_physics_settlement: (
            _RetainedPostPhysicsSettlement | None
        ) = None
        self._action_epoch_post_physics_settled_mask: torch.Tensor | None = None
        self._action_epoch_post_physics_result: ActionEpochR06PostPhysicsResult | None = None
        self._prepared_action_epoch_current_settlement_delta: (
            _ActionEpochOutcomeCandidateGrid | None
        ) = None
        self._pending_action_epoch_current_settlement_delta: (
            ActionEpochR06CurrentSettlementDelta | None
        ) = None
        self._action_epoch_current_settlement_delta_sequence = 0
        # Fresh ActionEpoch direct post-physics is sampled at physics cadence,
        # while scoring/mailbox materialization is committed once at the final
        # substep of the control window.  These tensors are R06's sole private
        # terminal-candidate owner; Physical receives only a typed stop mask.
        self._action_epoch_control_pending_cause = torch.full(
            self._flight_shape,
            SETTLEMENT_CAUSE_NONE,
            dtype=torch.int8,
            device=self.device,
        )
        self._action_epoch_control_pending_crossing_kind = torch.full(
            self._flight_shape,
            _CONTROL_CROSSING_NONE,
            dtype=torch.int8,
            device=self.device,
        )
        self._action_epoch_control_pending_crossing_xy = torch.zeros(
            self._flight_shape + (2,), dtype=torch.float32, device=self.device
        )
        self._action_epoch_control_pending_crossing_control = torch.full(
            self._flight_shape, -1, dtype=torch.int64, device=self.device
        )
        self._action_epoch_control_pending_crossing_substep = torch.full(
            self._flight_shape, -1, dtype=torch.int32, device=self.device
        )
        self._action_epoch_control_substep_count: int | None = None
        self._action_epoch_control_replay: _ActionEpochOutcomeCandidateGrid | None = None
        self._action_epoch_control_replay_substep: torch.Tensor | None = None
        self._action_epoch_control_outcome_next_index = 0
        self._active_post_physics_contact_authority: (
            _RetainedPostPhysicsContactAuthority | None
        ) = None
        self._active_physical_retire_lease: (
            _ActivePhysicalRetireLease | None
        ) = None
        self._active_selected_reset_lease: (
            _ActiveLandingOutcomeSelectedResetLease | None
        ) = None
        self._latest_selected_reset_completion: (
            tuple[
                LandingOutcomeSelectedResetCommitToken,
                object,
                LandingOutcomeSelectedResetCompletionToken,
                _r05_device.DeviceR05PreparedTrueReset,
                _r05_device.DeviceR05TrueResetReceipt,
            ]
            | None
        ) = None
        self._full_mdp_reward_binding_window_open = True
        self._full_mdp_reward_graph_bound = False
        self._full_mdp_reward_runtime_owner: object | None = None
        self._full_mdp_reward_prepublication_validator: object | None = None
        self._active_full_mdp_reward_cycle_identity: object | None = None
        self._active_full_mdp_reward_cycle_token: (
            LandingOutcomeFullMdpRewardCycleToken | None
        ) = None
        self._active_full_mdp_reward_prepublication: object | None = None
        self._full_mdp_reward_poisoned = False
        self._full_mdp_reward_poison_reason: str | None = None
        self._full_mdp_reward_close_debt = False
        self._active_full_mdp_reward_epoch: torch.Tensor | None = None
        self._active_full_mdp_reward_verdicts: dict[
            str, LandingOutcomeFullMdpRewardPaymentVerdict
        ] = {}
        self._active_full_mdp_reward_views: set[str] = set()
        self._active_full_mdp_reward_runtime_owner: object | None = None
        self._latest_full_mdp_reward_close: (
            LandingOutcomeFullMdpRewardCloseReceipt | None
        ) = None
        # The lean epoch seam is a construction-bound object join.  It grants
        # no formal/checkpoint/export authority and carries no caller verdict,
        # receipt, or source-SHA success claim.
        self._action_ball_full_mdp_epoch_owner: object | None = None
        self._action_epoch_row_fault_latch: object | None = None
        self._full_mdp_observation_projection = object.__new__(
            ActionBallFullMdpObservationProjection
        )
        self._owner_identity = object()
        self._poisoned = False
        self._global_reveal_poison_reason: str | None = None
        self._reveal_prepare_boundary_sequence = 0
        self._reveal_prepare_boundary_transfer_count = 0
        self._reveal_prepare_boundary_bytes_total = 0
        self._reveal_prepare_boundary_elapsed_ns_total = 0
        self._diagnostic_n2_namespace_projection = None
        # A diagnostic selected reset is a live row mutation, not formal
        # checkpoint/export authority.  Retain only the cold semantic defaults
        # for row-shaped resettable tensors.  The mapping is never exposed and
        # is used only as a source for fresh masked copies; it is not accepted as
        # evidence that a later reset was correct.
        self._diagnostic_selected_reset_row_defaults: (
            dict[str, torch.Tensor] | None
        ) = None
        if diagnostic_record is not None:
            self._diagnostic_selected_reset_row_defaults = {
                name: tensor.detach().clone()
                for name, tensor in self._checkpoint_tensors().items()
                if self._selected_reset_row_tensor_name(name)
                and tensor.ndim >= 1
                and tensor.shape[0] == self.num_envs
            }
            projection = object.__new__(DiagnosticN2NoSaveNamespaceProjection)
            with _DIAGNOSTIC_N2_REGISTRY_LOCK:
                _DIAGNOSTIC_N2_NAMESPACE_REGISTRY[projection] = (
                    _DiagnosticN2NoSaveNamespaceRecord(
                        owner_ref=weakref.ref(self),
                        family=diagnostic_record.family,
                        run_id=DIAGNOSTIC_N2_NO_SAVE_RUN_IDS[
                            diagnostic_record.family
                        ],
                        carry_chain_id=DIAGNOSTIC_N2_NO_SAVE_CARRY_CHAIN_IDS[
                            diagnostic_record.family
                        ],
                    )
                )
            self._diagnostic_n2_namespace_projection = projection

    @property
    def flight_state(self) -> torch.Tensor:
        return self._flight_state.detach().clone()

    @property
    def mailbox_state(self) -> torch.Tensor:
        return self._mailbox_state.detach().clone()

    @property
    def diagnostic_n2_no_save(self) -> bool:
        return self._diagnostic_n2_no_save

    @property
    def diagnostic_unauthorized(self) -> bool:
        return self._diagnostic_unauthorized

    def diagnostic_n2_no_save_namespace_projection(
        self,
    ) -> DiagnosticN2NoSaveNamespaceProjection:
        """Issue the one stable namespace identity owned by this diagnostic R06."""

        projection = self._diagnostic_n2_namespace_projection
        if (
            not self._diagnostic_n2_no_save
            or type(projection) is not DiagnosticN2NoSaveNamespaceProjection
        ):
            raise LandingOutcomeDeviceError(
                "R06 owner is not the diagnostic N=2 namespace issuer"
            )
        self.require_owned_diagnostic_n2_no_save_namespace(projection)
        return projection

    def require_owned_diagnostic_n2_no_save_namespace(
        self,
        projection: DiagnosticN2NoSaveNamespaceProjection,
    ) -> DiagnosticN2NoSaveNamespaceView:
        """Authenticate owner identity before exposing clone-only R05 strings."""

        if type(projection) is not DiagnosticN2NoSaveNamespaceProjection:
            raise LandingOutcomeDeviceError(
                "diagnostic R06 namespace projection type differs"
            )
        with _DIAGNOSTIC_N2_REGISTRY_LOCK:
            try:
                record = _DIAGNOSTIC_N2_NAMESPACE_REGISTRY[projection]
            except (KeyError, TypeError) as exc:
                raise LandingOutcomeDeviceError(
                    "diagnostic R06 namespace projection is stale or foreign"
                ) from exc
        if (
            record.owner_ref() is not self
            or projection is not self._diagnostic_n2_namespace_projection
        ):
            raise LandingOutcomeDeviceError(
                "diagnostic R06 namespace projection is stale or foreign"
            )
        return DiagnosticN2NoSaveNamespaceView(
            family=record.family,
            run_id=record.run_id,
            carry_chain_id=record.carry_chain_id,
        )

    def _require_formal_only(self, operation: str) -> None:
        if self._diagnostic_n2_no_save:
            raise LandingOutcomeDeviceError(
                f"diagnostic N=2 no-save R06 cannot {operation}; formal authority is absent"
            )

    def __getstate__(self) -> dict[str, object]:
        if self._diagnostic_n2_no_save:
            raise TypeError("diagnostic N=2 no-save R06 cannot be exported")
        return dict(self.__dict__)

    def current_flight_lifecycle_snapshot(
        self,
    ) -> FlightLifecycleSnapshotBatch:
        """Expose the complete owner-rooted flight identity without D2H."""

        if not self._poisoned:
            self._require_operable(
                allow_pending_post_physics_settlement=True,
                allow_pending_post_physics_contact_authority=True,
            )
        return self._flight_lifecycle_snapshot()

    def action_ball_full_mdp_observation_projection(
        self,
    ) -> ActionBallFullMdpObservationProjection:
        """Return R06's stable opaque observation authority without mutation."""

        self._require_operable(
            allow_pending_post_physics_settlement=True,
            allow_pending_post_physics_contact_authority=True,
        )
        return self._full_mdp_observation_projection

    def bind_action_ball_full_mdp_epoch_owner(self, epoch_owner: object) -> None:
        """Cold-bind R06 as the sole writer of the epoch landing-fact slice.

        This is an object-identity construction join, not an authorization
        receipt.  It must happen before either owner starts its first live
        epoch; formal/save/export holds remain unchanged.
        """

        if __package__:
            from . import action_ball_full_mdp_epoch as epoch_v1
        else:
            import action_ball_full_mdp_epoch as epoch_v1

        shot_slot_capacity = getattr(epoch_owner, "shot_slot_capacity", None)
        if (
            type(shot_slot_capacity) is not int
            or shot_slot_capacity < 1
            or self._action_ball_full_mdp_epoch_owner is not None
            or not self._load_fresh
        ):
            raise LandingOutcomeDeviceError(
                "R06 epoch owner must be exact, shape/device-compatible, and cold"
            )
        diagnostic_record = self._diagnostic_n2_construction_record
        if diagnostic_record is None:
            if (
                type(epoch_owner) is not epoch_v1.ActionEpochOwner
                or epoch_owner.num_envs != self.num_envs
                or epoch_owner.device != self.device
            ):
                raise LandingOutcomeDeviceError(
                    "R06 epoch owner must be exact, shape/device-compatible, and cold"
                )
        else:
            live_cardinality = _diagnostic_live_cardinality_anchors(
                env=diagnostic_record.env,
                physical_owner=diagnostic_record.physical_owner,
                epoch_owner=epoch_owner,
            )
            if (
                live_cardinality[0] != self.num_envs
                or live_cardinality[1] is not diagnostic_record.scene_port
                or live_cardinality[2]
                is not diagnostic_record.scene_port_capability
                or getattr(diagnostic_record.physical_owner, "_r06_owner", None)
                is not self
            ):
                raise LandingOutcomeDeviceError(
                    "diagnostic R06 live Env/Physical/ActionEpoch owner join differs"
                )
        fault_latch = getattr(epoch_owner, "latch_runtime_row_fault", None)
        direct_fault_latch = getattr(
            epoch_v1.ActionEpochOwner, "latch_runtime_row_fault", None
        )
        try:
            epoch_fault_names = dict(
                getattr(epoch_v1, "ACTION_EPOCH_ROW_FAULT_NAMES", ())
            )
        except (TypeError, ValueError):
            epoch_fault_names = {}
        if (
            not callable(fault_latch)
            or not callable(direct_fault_latch)
            or getattr(fault_latch, "__self__", None) is not epoch_owner
            or getattr(fault_latch, "__func__", None) is not direct_fault_latch
            or any(
                getattr(epoch_v1, constant, None) != expected
                or epoch_fault_names.get(expected) != reason
                for constant, expected, reason
                in R06_ACTION_EPOCH_ROW_FAULT_BINDINGS
            )
        ):
            raise LandingOutcomeDeviceError(
                "R06 epoch owner named row-fault ABI differs"
            )
        epoch_owner.bind_fact_owner("r06_landing_outcome", self)
        epoch_owner.bind_async_owner("r06_landing_outcome", self)
        self._action_ball_full_mdp_epoch_owner = epoch_owner
        self._action_epoch_row_fault_latch = fault_latch
        if diagnostic_record is not None:
            # The exact Env/Physical/scene objects are a cold join, not runtime
            # state.  Once ActionEpoch accepts both R06 bindings, drop the
            # construction graph so this leaf cannot retain the Env or its
            # large device tensor graph for the owner's whole lifetime.
            self._diagnostic_n2_construction_record = None

    def _latch_action_epoch_row_fault(
        self,
        rows: torch.Tensor,
        *,
        epoch_reason_bit: int,
    ) -> torch.Tensor:
        """Latch one exact R06 cause and return rows still safe to use.

        This path performs no host read or synchronization.  The exact bound
        Epoch owner ORs the cause into its single existing packed
        optimizer-boundary word.  The legacy unbound seam masks only the
        current call for backward-compatible isolated tests; fresh V5 binds
        Epoch before any live row and therefore gets sticky row semantics.
        """

        if (
            type(rows) is not torch.Tensor
            or rows.dtype != torch.bool
            or rows.device != self.device
            or tuple(rows.shape) != (self.num_envs,)
            or not rows.is_contiguous()
            or type(epoch_reason_bit) is not int
            or epoch_reason_bit not in _R06_ACTION_EPOCH_ROW_FAULT_BITS
        ):
            raise LandingOutcomeDeviceError(
                "R06 named ActionEpoch row-fault ABI differs"
            )
        latch = self._action_epoch_row_fault_latch
        if latch is None:
            return ~rows
        epoch_safe = latch(
            "r06_landing_outcome", epoch_reason_bit, rows, owner=self
        )
        if (
            type(epoch_safe) is not torch.Tensor
            or epoch_safe.dtype != torch.bool
            or epoch_safe.device != self.device
            or tuple(epoch_safe.shape) != (self.num_envs,)
            or not epoch_safe.is_contiguous()
        ):
            raise LandingOutcomeDeviceError(
                "R06 named ActionEpoch row-fault safe mask differs"
            )
        return epoch_safe

    def install_action_ball_full_mdp_epoch_launch_from_physical(self) -> None:
        """Install exact launched rows from the permanently paired Physical owner.

        The caller supplies no tensor, portable identity, or verdict.  The
        sole Physical owner exposes one nonconstructible view only while its
        scene/slot launch transaction is active; R06 derives its mailbox slot
        locally and joins only Physical's owner-produced row key.
        An empty due mask is an ordinary device-only no-op.
        """

        epoch_owner = self._action_ball_full_mdp_epoch_owner
        park_authority = self._physical_park_token_authority
        if epoch_owner is None or park_authority is None:
            raise LandingOutcomeDeviceError(
                "R06 epoch launch requires bound ActionEpoch and Physical owners"
            )
        physical_owner = park_authority.physical_owner
        if __package__:
            from . import action_ball_physical_flight_device as physical_v1
        else:
            import action_ball_physical_flight_device as physical_v1

        projector = physical_owner.action_epoch_r06_launch_projection
        validator = physical_owner.require_owned_action_epoch_r06_launch_projection
        if (
            type(physical_owner) is not physical_v1.ActionBallPhysicalFlightDeviceOwner
            or projector.__func__
            is not physical_v1.ActionBallPhysicalFlightDeviceOwner.action_epoch_r06_launch_projection
            or validator.__func__
            is not physical_v1.ActionBallPhysicalFlightDeviceOwner.require_owned_action_epoch_r06_launch_projection
        ):
            raise LandingOutcomeDeviceError(
                "R06 epoch launch requires the exact bound Physical owner methods"
            )
        view = validator(projector())
        if (
            type(view) is not physical_v1.ActionEpochR06LaunchProjection
            or view.physical_owner is not physical_owner
            or view.epoch_owner is not epoch_owner
        ):
            raise LandingOutcomeDeviceError(
                "R06 epoch launch Physical projection is foreign"
            )
        ids = self._env_ids
        k = self.num_envs

        def exact(
            value: object,
            *,
            label: str,
            dtype: torch.dtype,
            width: int | None = None,
        ) -> torch.Tensor:
            shape = (k,) if width is None else (k, width)
            if (
                type(value) is not torch.Tensor
                or value.dtype != dtype
                or value.device != self.device
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                raise LandingOutcomeDeviceError(
                    "R06 epoch launch " + label + " ABI differs"
                )
            return value.detach()

        selected = exact(
            getattr(view, "selected_mask", None),
            label="selected_mask",
            dtype=torch.bool,
        )
        due = exact(view.due, label="due", dtype=torch.bool)
        flight_slot = exact(
            view.flight_slot, label="flight_slot", dtype=torch.int64
        )
        shot_key = _row_identity.require_action_epoch_shot_key(
            getattr(view, "shot_key", None),
            shape=(k,),
            device=self.device,
            label="R06 epoch launch shot_key",
        )
        publication_ordinal = exact(
            getattr(view, "publication_ordinal", None),
            label="publication_ordinal",
            dtype=torch.int64,
        )
        generation = shot_key.ball_generation
        action_uid = shot_key.action_uid
        action_slot = shot_key.action_slot
        reset_generation = shot_key.reset_generation
        shot_index = shot_key.shot_index
        task_identity = shot_key.task_identity
        outcome_identity = shot_key.outcome_identity
        ball_identity = shot_key.ball_identity
        target_xy = exact(
            view.target_xy_m,
            label="target_xy_m",
            dtype=torch.float32,
            width=2,
        )
        launch_step = exact(
            view.launch_control_step,
            label="launch_control_step",
            dtype=torch.int64,
        )
        contact_deadline = exact(
            view.contact_deadline_control_step,
            label="contact_deadline_control_step",
            dtype=torch.int64,
        )
        crossing_horizon = exact(
            view.crossing_horizon_control_step,
            label="crossing_horizon_control_step",
            dtype=torch.int64,
        )

        neutral_key = torch.ones_like(selected)
        for field in fields(_row_identity.ActionEpochShotKey):
            neutral_key &= getattr(shot_key, field.name).eq(-1)
        neutral = (
            ~due
            & flight_slot.eq(-1)
            & neutral_key
            & publication_ordinal.eq(-1)
            & target_xy.eq(0.0).all(dim=1)
            & launch_step.eq(-1)
            & contact_deadline.eq(-1)
            & crossing_horizon.eq(-1)
        )
        safe_rows = self._latch_action_epoch_row_fault(
            ((due & ~selected) | (~selected & ~neutral)).contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_LAUNCH_SELECTION_CONTRACT
            ),
        )
        due = due & safe_rows
        safe_ids = ids
        flight_in_range = flight_slot.ge(0) & flight_slot.lt(
            self.flight_slot_capacity
        )
        safe_flight = flight_slot.clamp(0, self.flight_slot_capacity - 1)
        selected_flight_state = self._flight_state[safe_ids, safe_flight]
        selected_flight_retired = self._flight_physical_retired[
            safe_ids, safe_flight
        ]

        mailbox_available = (
            self._mailbox_state.eq(MAILBOX_EMPTY)
            & ~self._mailbox_reserved
            & ~self._mailbox_history_valid
            & ~self._mailbox_physical_retired
        )
        mailbox_sentinel = self.mailbox_capacity
        mailbox_candidate = torch.where(
            mailbox_available[safe_ids],
            self._mailbox_slot_ids.expand(self._mailbox_shape)[safe_ids],
            torch.full(
                (k, self.mailbox_capacity),
                mailbox_sentinel,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        mailbox_slot = mailbox_candidate.amin(dim=1)
        mailbox_available_for_row = mailbox_slot.lt(self.mailbox_capacity)
        safe_mailbox = mailbox_slot.clamp(0, self.mailbox_capacity - 1)

        replay_ordered = (~self._replay_valid[safe_ids]) | (
            publication_ordinal.gt(self._replay_publication_ordinal[safe_ids])
            & (
                reset_generation.gt(self._replay_reset_generation[safe_ids])
                | (
                    reset_generation.eq(self._replay_reset_generation[safe_ids])
                    & generation.gt(self._replay_swing_generation[safe_ids])
                )
            )
        )
        expanded_shot_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(shot_key, field.name)
                .unsqueeze(1)
                .expand(k, self.flight_slot_capacity)
                .contiguous()
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        retained_flight_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(
                    self._flight_action_epoch_shot_key(), field.name
                )[safe_ids]
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        flight_identity_collision = (
            self._flight_action_epoch[safe_ids]
            & self._flight_state[safe_ids].ne(FLIGHT_EMPTY)
            & _row_identity.action_epoch_shot_key_equal(
                retained_flight_key, expanded_shot_key
            )
        ).any(dim=1)
        expanded_mailbox_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(shot_key, field.name)
                .unsqueeze(1)
                .expand(k, self.mailbox_capacity)
                .contiguous()
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        retained_mailbox_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(
                    self._mailbox_action_epoch_shot_key(), field.name
                )[safe_ids]
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        mailbox_identity_collision = (
            self._mailbox_action_epoch[safe_ids]
            & self._mailbox_history_valid[safe_ids]
            & _row_identity.action_epoch_shot_key_equal(
                retained_mailbox_key, expanded_mailbox_key
            )
        ).any(dim=1)
        identity_valid = _row_identity.action_epoch_shot_key_valid(shot_key)
        value_valid = (
            flight_in_range
            & selected_flight_state.eq(FLIGHT_EMPTY)
            & selected_flight_retired
            & mailbox_available_for_row
            & replay_ordered
            & ~flight_identity_collision
            & ~mailbox_identity_collision
            & action_uid.ge(1)
            & action_uid.le(_MAX_ACTION_UID)
            & action_slot.ge(0)
            & shot_index.ge(1)
            & generation.ge(0)
            & task_identity.gt(0)
            & outcome_identity.gt(0)
            & ball_identity.gt(0)
            & publication_ordinal.ge(0)
            & torch.isfinite(target_xy).all(dim=1)
            & target_xy[:, 0].ge(self.profile.opponent_table_x_min_m)
            & target_xy[:, 0].le(self.profile.opponent_table_x_max_m)
            & target_xy[:, 1].ge(self.profile.table_y_min_m)
            & target_xy[:, 1].le(self.profile.table_y_max_m)
            & launch_step.ge(0)
            & crossing_horizon.ge(contact_deadline)
        )
        safe_rows = self._latch_action_epoch_row_fault(
            (due & ~(identity_valid & value_valid)).contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_LAUNCH_IDENTITY_CONTRACT
            ),
        )
        due = due & safe_rows

        flight_target = torch.zeros(
            self._flight_shape, dtype=torch.bool, device=self.device
        )
        flight_target[safe_ids, safe_flight] = due
        mailbox_target = torch.zeros(
            self._mailbox_shape, dtype=torch.bool, device=self.device
        )
        mailbox_target[safe_ids, safe_mailbox] = due

        typed_rows = (
            (self._flight_action_uid, action_uid),
            (self._flight_action_slot, action_slot),
            (self._flight_reset_generation, reset_generation),
            (self._flight_shot_index, shot_index),
            (self._flight_task_identity, task_identity),
            (self._flight_outcome_identity, outcome_identity),
            (self._flight_ball_identity, ball_identity),
            (self._flight_publication_ordinal, publication_ordinal),
        )
        for destination, row in typed_rows:
            source = torch.full_like(destination, -1)
            source[safe_ids, safe_flight] = row
            _masked_copy_(destination, source, flight_target)

        def flight_rows(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(
                self._flight_shape + tuple(value.shape[1:]),
                dtype=value.dtype,
                device=self.device,
            )
            result[safe_ids, safe_flight] = value
            return result

        _masked_copy_(self._flight_ball_generation, flight_rows(generation), flight_target)
        _masked_copy_(self._flight_mailbox_slot, flight_rows(safe_mailbox), flight_target)
        _masked_copy_(self._flight_target_xy_m, flight_rows(target_xy), flight_target)
        _masked_copy_(self._flight_reveal_control_step, flight_rows(launch_step), flight_target)
        _masked_copy_(self._flight_contact_deadline_control_step, flight_rows(contact_deadline), flight_target)
        _masked_copy_(self._flight_crossing_horizon_control_step, flight_rows(crossing_horizon), flight_target)
        _masked_fill_(self._flight_state, flight_target, FLIGHT_INBOUND)
        _masked_fill_(self._flight_physical_retired, flight_target, 0)
        _masked_fill_(self._flight_fault_bits, flight_target, 0)
        _masked_fill_(self._flight_action_epoch, flight_target, 1)
        for destination in (
            self._flight_contact_valid,
            self._flight_net_crossed,
            self._flight_net_clear,
        ):
            _masked_fill_(destination, flight_target, 0)
        for destination in (
            self._flight_contact_stamp_control,
            self._flight_observation_ordinal,
            self._flight_last_observation_control,
            self._flight_net_stamp_control,
        ):
            _masked_fill_(destination, flight_target, -1)
        for destination in (
            self._flight_contact_stamp_substep,
            self._flight_last_observation_substep,
            self._flight_net_stamp_substep,
        ):
            _masked_fill_(destination, flight_target, -1)
        for destination in (
            self._flight_contact_ball_center_m,
            self._flight_outgoing_anchor_m,
            self._flight_last_ball_center_m,
        ):
            _masked_fill_(destination, flight_target, 0.0)

        _masked_fill_(self._mailbox_reserved, mailbox_target, 1)
        _masked_fill_(self._mailbox_action_epoch, mailbox_target, 1)
        for destination, row in (
            (self._mailbox_action_uid, action_uid),
            (self._mailbox_action_slot, action_slot),
            (self._mailbox_reset_generation, reset_generation),
            (self._mailbox_shot_index, shot_index),
            (self._mailbox_task_identity, task_identity),
            (self._mailbox_outcome_identity, outcome_identity),
            (self._mailbox_ball_identity, ball_identity),
            (self._mailbox_ball_generation, generation),
            (self._mailbox_publication_ordinal, publication_ordinal),
        ):
            source = torch.full_like(destination, -1)
            source[safe_ids, safe_mailbox] = row
            _masked_copy_(destination, source, mailbox_target)
        mailbox_generation = torch.full_like(
            self._mailbox_reservation_generation, -1
        )
        mailbox_generation[safe_ids, safe_mailbox] = generation
        _masked_copy_(
            self._mailbox_reservation_generation,
            mailbox_generation,
            mailbox_target,
        )
        mailbox_flight = torch.full_like(
            self._mailbox_reserved_flight_slot, -1
        )
        mailbox_flight[safe_ids, safe_mailbox] = safe_flight
        _masked_copy_(
            self._mailbox_reserved_flight_slot, mailbox_flight, mailbox_target
        )

        self._replay_valid[safe_ids] |= due
        self._replay_reset_generation[safe_ids] = torch.where(
            due, reset_generation, self._replay_reset_generation[safe_ids]
        )
        self._replay_swing_generation[safe_ids] = torch.where(
            due, generation, self._replay_swing_generation[safe_ids]
        )
        self._replay_action_epoch[safe_ids] |= due
        for destination, source in (
            (self._replay_action_uid, action_uid),
            (self._replay_action_slot, action_slot),
            (self._replay_shot_index, shot_index),
            (self._replay_task_identity, task_identity),
            (self._replay_outcome_identity, outcome_identity),
            (self._replay_ball_identity, ball_identity),
            (self._replay_publication_ordinal, publication_ordinal),
        ):
            destination[safe_ids] = torch.where(
                due, source, destination[safe_ids]
            )
        self._reset_generation_highwater[safe_ids] = torch.where(
            due,
            torch.maximum(
                self._reset_generation_highwater[safe_ids], reset_generation
            ),
            self._reset_generation_highwater[safe_ids],
        )
        self._installed_total.add_(due.to(torch.int64).sum())
        self._mutation_version.add_(due.any().to(torch.int64))

    @staticmethod
    def _action_epoch_selected(value: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        suffix = (1,) * (value.ndim - 2)
        index = slot.reshape(slot.shape[0], 1, *suffix).expand(
            slot.shape[0], 1, *value.shape[2:]
        )
        return torch.gather(value, 1, index).squeeze(1)

    def _flight_action_epoch_shot_key(self) -> _row_identity.ActionEpochShotKey:
        return _row_identity.ActionEpochShotKey(
            reset_generation=self._flight_reset_generation,
            ball_generation=self._flight_ball_generation,
            action_uid=self._flight_action_uid,
            action_slot=self._flight_action_slot,
            shot_index=self._flight_shot_index,
            task_identity=self._flight_task_identity,
            outcome_identity=self._flight_outcome_identity,
            ball_identity=self._flight_ball_identity,
        )

    def _mailbox_action_epoch_shot_key(self) -> _row_identity.ActionEpochShotKey:
        return _row_identity.ActionEpochShotKey(
            reset_generation=self._mailbox_reset_generation,
            ball_generation=self._mailbox_ball_generation,
            action_uid=self._mailbox_action_uid,
            action_slot=self._mailbox_action_slot,
            shot_index=self._mailbox_shot_index,
            task_identity=self._mailbox_task_identity,
            outcome_identity=self._mailbox_outcome_identity,
            ball_identity=self._mailbox_ball_identity,
        )

    def _gather_mailbox_action_epoch_shot_key(
        self, slot: torch.Tensor
    ) -> _row_identity.ActionEpochShotKey:
        return _row_identity.ActionEpochShotKey(
            **{
                field.name: torch.gather(
                    getattr(self._mailbox_action_epoch_shot_key(), field.name),
                    1,
                    slot,
                )
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )

    def _previous_paid_action_epoch_shot_key(
        self,
    ) -> _row_identity.ActionEpochShotKey:
        return _row_identity.ActionEpochShotKey(
            reset_generation=self._previous_paid_reset_generation,
            ball_generation=self._previous_paid_ball_generation,
            action_uid=self._previous_paid_action_uid,
            action_slot=self._previous_paid_action_slot,
            shot_index=self._previous_paid_shot_index,
            task_identity=self._previous_paid_task_identity,
            outcome_identity=self._previous_paid_outcome_identity,
            ball_identity=self._previous_paid_ball_identity,
        )

    def _pack_action_epoch_fact_grid(
        self,
        *,
        common_on_table: torch.Tensor,
        canonical_total: torch.Tensor,
        placement_gain: torch.Tensor,
        crossing_xy: torch.Tensor,
        placement_error: torch.Tensor,
        on_opponent_table: torch.Tensor,
        contact_valid: torch.Tensor,
        crossing_valid: torch.Tensor,
        net_crossed: torch.Tensor,
        net_clear: torch.Tensor,
    ) -> torch.Tensor:
        """Pack the one R06 fact ABI used by both retained and current grids."""

        used = torch.cat(
            (
                common_on_table.to(torch.float32).unsqueeze(-1),
                canonical_total.to(torch.float32).unsqueeze(-1),
                placement_gain.to(torch.float32).unsqueeze(-1),
                crossing_xy.to(torch.float32),
                placement_error.to(torch.float32).unsqueeze(-1),
                on_opponent_table.to(torch.float32).unsqueeze(-1),
                contact_valid.to(torch.float32).unsqueeze(-1),
                crossing_valid.to(torch.float32).unsqueeze(-1),
                net_crossed.to(torch.float32).unsqueeze(-1),
                net_clear.to(torch.float32).unsqueeze(-1),
            ),
            dim=-1,
        )
        return torch.cat(
            (
                used,
                torch.zeros(
                    canonical_total.shape
                    + (R06_ACTION_EPOCH_FACT_F32_WIDTH - R06_ACTION_EPOCH_FACT_F32_USED,),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
            dim=-1,
        )

    def _prepare_action_epoch_current_settlement_delta(
        self,
        *,
        settle: torch.Tensor,
        observation_stamp: PhysicsStampBatch,
        settlement_cause: torch.Tensor,
        policy_settlement: torch.Tensor,
        crossing_valid: torch.Tensor,
        crossing_xy: torch.Tensor,
        score: _PolicyScoreGrid,
    ) -> _ActionEpochOutcomeCandidateGrid:
        """Pack only this direct publication's source facts, never mailbox history."""

        policy = policy_settlement & settle
        score_total = score.total.reshape(self._flight_shape).to(torch.float32)
        score_error = score.placement_error_m.reshape(self._flight_shape).to(
            torch.float32
        )
        score_on_table = score.on_opponent_table.reshape(self._flight_shape)
        zeros = torch.zeros_like(score_total)
        canonical_total = torch.where(policy, score_total, zeros)
        placement_error = torch.where(policy, score_error, zeros)
        common_on_table = (
            policy
            & self._flight_contact_valid
            & crossing_valid
            & self._flight_net_crossed
            & self._flight_net_clear
            & score_on_table
        )
        fact_values = self._pack_action_epoch_fact_grid(
            common_on_table=common_on_table,
            canonical_total=canonical_total,
            placement_gain=torch.full_like(
                canonical_total, self._placement_treatment_gain
            ),
            crossing_xy=crossing_xy,
            placement_error=placement_error,
            on_opponent_table=torch.where(
                policy, score_on_table, torch.zeros_like(score_on_table)
            ),
            contact_valid=self._flight_contact_valid,
            crossing_valid=crossing_valid,
            net_crossed=self._flight_net_crossed,
            net_clear=self._flight_net_clear,
        )
        flight_key = self._flight_action_epoch_shot_key()
        return _ActionEpochOutcomeCandidateGrid(
            candidate=settle.detach().clone(),
            shot_key_values=torch.stack(
                tuple(getattr(flight_key, field.name) for field in fields(
                    _row_identity.ActionEpochShotKey
                )),
                dim=-1,
            ),
            publication_ordinal=self._flight_publication_ordinal,
            settlement_step=observation_stamp.control_step,
            policy_eligible=policy,
            fact_values=fact_values,
            outcome_code=settlement_cause.to(torch.int64),
            owner_fault_bits=self._flight_fault_bits,
        )

    def _project_action_epoch_outcome_candidates(
        self,
        grid: _ActionEpochOutcomeCandidateGrid,
    ) -> ActionEpochR06OutcomeRows:
        """Select and normalize one typed row from any owner-local candidate grid."""

        ordinal_grid = torch.where(
            grid.candidate,
            grid.publication_ordinal,
            torch.full_like(grid.publication_ordinal, -1),
        )
        newest = ordinal_grid.amax(dim=1)
        selected = grid.candidate & grid.publication_ordinal.eq(
            newest.unsqueeze(1)
        )
        safe_rows = self._latch_action_epoch_row_fault(
            selected.to(torch.int64).sum(dim=1).gt(1).contiguous(),
            epoch_reason_bit=R06_EPOCH_ROW_FAULT_OUTCOME_PROJECTION_DUPLICATE,
        )
        selected = selected & safe_rows.unsqueeze(1)
        valid = selected.any(dim=1)
        flight_slot = torch.argmax(selected.to(torch.int64), dim=1)

        def gather(value: torch.Tensor) -> torch.Tensor:
            suffix = (1,) * (value.ndim - 2)
            index = flight_slot.reshape(self.num_envs, 1, *suffix).expand(
                self.num_envs, 1, *value.shape[2:]
            )
            return torch.gather(value, 1, index).squeeze(1)

        key_values = gather(grid.shot_key_values)
        metadata = gather(
            torch.stack(
                (
                    grid.publication_ordinal,
                    grid.settlement_step,
                    grid.outcome_code.to(torch.int64),
                    grid.owner_fault_bits,
                    grid.policy_eligible.to(torch.int64),
                ),
                dim=-1,
            )
        )
        invalid_metadata = torch.tensor(
            (-1, -1, -1, 0, 0), dtype=torch.int64, device=self.device
        )
        metadata = torch.where(
            valid.unsqueeze(1), metadata, invalid_metadata.unsqueeze(0)
        )
        source_fault = metadata[:, 3]
        cause = metadata[:, 2]
        producer_cause = (
            cause.eq(SETTLEMENT_CAUSE_PRODUCER_CONTRACT_FAULT)
            | cause.eq(SETTLEMENT_CAUSE_ENGINE_OVERFLOW)
            | cause.eq(SETTLEMENT_CAUSE_PROTOCOL_FAULT)
        )
        source_fault = torch.where(
            valid & producer_cause & source_fault.eq(0),
            torch.full_like(source_fault, FAULT_PRODUCER_CONTRACT),
            source_fault,
        )
        source_fault = torch.where(
            valid & self._device_sticky_poison,
            torch.bitwise_or(
                source_fault, torch.full_like(source_fault, FAULT_SAFETY_CLEANUP)
            ),
            source_fault,
        )
        selected_facts = gather(grid.fact_values)
        numeric_finite = (
            torch.isfinite(
                selected_facts[:, R06_ACTION_EPOCH_CANONICAL_TOTAL_F32]
            )
            & torch.isfinite(
                selected_facts[:, R06_ACTION_EPOCH_PLACEMENT_GAIN_F32]
            )
            & torch.isfinite(
                selected_facts[:, R06_ACTION_EPOCH_CROSSING_XY_F32]
            ).all(dim=1)
            & torch.isfinite(
                selected_facts[:, R06_ACTION_EPOCH_PLACEMENT_ERROR_F32]
            )
        )
        source_fault = torch.where(
            valid & ~numeric_finite,
            torch.bitwise_or(
                source_fault, torch.full_like(source_fault, FAULT_NONFINITE)
            ),
            source_fault,
        )
        source_valid = valid & source_fault.eq(0)
        policy_eligible = source_valid & metadata[:, 4].to(torch.bool)
        valid_bits = (
            valid.to(torch.int64) * R06_ACTION_EPOCH_PRESENT
            + policy_eligible.to(torch.int64) * R06_ACTION_EPOCH_POLICY_ELIGIBLE
            + source_valid.to(torch.int64) * R06_ACTION_EPOCH_SOURCE_VALID
        )
        facts = torch.where(
            source_valid.unsqueeze(1), selected_facts, torch.zeros_like(selected_facts)
        )
        invalid_key = torch.full_like(key_values, -1)
        key_values = torch.where(valid.unsqueeze(1), key_values, invalid_key)
        key = _row_identity.ActionEpochShotKey(
            **{
                field.name: key_values[:, index].contiguous()
                for index, field in enumerate(fields(_row_identity.ActionEpochShotKey))
            }
        )
        return ActionEpochR06OutcomeRows(
            valid=valid.contiguous(),
            shot_key=key,
            publication_ordinal=metadata[:, 0].contiguous(),
            settlement_step=metadata[:, 1].contiguous(),
            valid_bits=valid_bits.contiguous(),
            fact_values=facts.contiguous(),
            outcome_code=cause.contiguous(),
            owner_fault_bits=source_fault.contiguous(),
        )

    def _mint_action_epoch_current_settlement_delta(
        self, retire_valid: torch.Tensor
    ) -> ActionEpochR06CurrentSettlementDelta:
        """Mint the one exact per-environment delta after physical retirement."""

        prepared = self._prepared_action_epoch_current_settlement_delta
        if (
            type(prepared) is not _ActionEpochOutcomeCandidateGrid
            or prepared.candidate is not self._action_epoch_post_physics_settled_mask
            or self._pending_action_epoch_current_settlement_delta is not None
        ):
            raise LandingOutcomeDeviceError(
                "R06 current-settlement delta lifetime differs"
            )
        projection = self._project_action_epoch_outcome_candidates(
            _with_action_epoch_candidate(prepared, retire_valid)
        )
        return self._mint_action_epoch_outcome_rows(projection)

    def _mint_action_epoch_outcome_rows(
        self, rows: ActionEpochR06OutcomeRows
    ) -> ActionEpochR06CurrentSettlementDelta:
        if (
            type(rows) is not ActionEpochR06OutcomeRows
            or self._pending_action_epoch_current_settlement_delta is not None
        ):
            raise LandingOutcomeDeviceError(
                "R06 current-settlement rows lifetime differs"
            )
        delta = ActionEpochR06CurrentSettlementDelta(
            rows=rows,
            sequence=self._action_epoch_current_settlement_delta_sequence,
            r06_owner=self,
            epoch_owner=self._action_ball_full_mdp_epoch_owner,
            _owner_identity=self,
            _token=_ACTION_EPOCH_R06_CURRENT_SETTLEMENT_DELTA_TOKEN,
        )
        self._action_epoch_current_settlement_delta_sequence += 1
        self._pending_action_epoch_current_settlement_delta = delta
        return delta

    def project_current_action_epoch_outcome_rows(
        self,
    ) -> ActionEpochR06OutcomeRows:
        """Project the newest retained settlement per environment by exact key."""

        self._require_operable()
        candidate = (
            self._mailbox_action_epoch
            & self._mailbox_history_valid
            & self._mailbox_physical_retired
            & self._mailbox_state.eq(MAILBOX_SETTLED_UNPAID)
        )
        values = self._pack_action_epoch_fact_grid(
            common_on_table=self._mailbox_common_on_table,
            canonical_total=self._mailbox_canonical_total,
            placement_gain=self._mailbox_placement_treatment_gain,
            crossing_xy=self._mailbox_crossing_xy_m,
            placement_error=self._mailbox_placement_error_m,
            on_opponent_table=self._mailbox_on_opponent_table,
            contact_valid=self._mailbox_contact_valid,
            crossing_valid=self._mailbox_crossing_valid,
            net_crossed=self._mailbox_net_crossed,
            net_clear=self._mailbox_net_clear,
        )
        mailbox_key = self._mailbox_action_epoch_shot_key()
        return self._project_action_epoch_outcome_candidates(
            _ActionEpochOutcomeCandidateGrid(
                candidate=candidate,
                shot_key_values=torch.stack(
                    tuple(
                        getattr(mailbox_key, field.name)
                        for field in fields(_row_identity.ActionEpochShotKey)
                    ),
                    dim=-1,
                ),
                publication_ordinal=self._mailbox_publication_ordinal,
                settlement_step=self._mailbox_settlement_control_step,
                policy_eligible=self._mailbox_policy_eligible,
                fact_values=values,
                outcome_code=self._mailbox_settlement_cause,
                owner_fault_bits=self._mailbox_fault_bits,
            )
        )

    def publish_action_ball_full_mdp_epoch_facts(self) -> None:
        """Push R06's exact one-shot current-settlement delta to Epoch."""

        owner = self._action_ball_full_mdp_epoch_owner
        delta = self._pending_action_epoch_current_settlement_delta
        if owner is None or type(delta) is not ActionEpochR06CurrentSettlementDelta:
            raise LandingOutcomeDeviceError("R06 ActionEpoch owner is not bound")
        if __package__:
            from . import action_ball_full_mdp_epoch as epoch_v1
        else:
            import action_ball_full_mdp_epoch as epoch_v1
        refresh = getattr(owner, "refresh_r06_outcome_rows", None)
        if (
            type(owner) is not epoch_v1.ActionEpochOwner
            or not callable(refresh)
            or getattr(refresh, "__self__", None) is not owner
            or getattr(refresh, "__func__", None)
            is not epoch_v1.ActionEpochOwner.refresh_r06_outcome_rows
        ):
            raise LandingOutcomeDeviceError(
                "R06 ActionEpoch current-settlement consumer differs"
            )
        refresh(delta)

    def publish_action_ball_full_mdp_epoch_control_substep_facts(
        self, *, substep_index: int
    ) -> None:
        """Replay one retained control-window outcome row in causal order."""

        replay = self._action_epoch_control_replay
        replay_substep = self._action_epoch_control_replay_substep
        count = self._action_epoch_control_substep_count
        owner = self._action_ball_full_mdp_epoch_owner
        if (
            type(replay) is not _ActionEpochOutcomeCandidateGrid
            or type(replay_substep) is not torch.Tensor
            or type(count) is not int
            or owner is None
            or type(substep_index) is not int
            or substep_index != self._action_epoch_control_outcome_next_index
            or substep_index < 0
            or substep_index >= count
            or self._pending_action_epoch_current_settlement_delta is not None
        ):
            raise LandingOutcomeDeviceError(
                "R06 ActionEpoch control outcome replay is stale or out of order"
            )
        rows = self._project_action_epoch_outcome_candidates(
            _with_action_epoch_candidate(
                replay, replay.candidate & replay_substep.eq(substep_index)
            )
        )
        self._mint_action_epoch_outcome_rows(rows)
        self.publish_action_ball_full_mdp_epoch_facts()
        if self._pending_action_epoch_current_settlement_delta is not None:
            raise LandingOutcomeDeviceError(
                "Epoch did not consume the control outcome delta"
            )
        self._action_epoch_control_outcome_next_index += 1
        if self._action_epoch_control_outcome_next_index == count:
            self._action_epoch_control_replay = None
            self._action_epoch_control_replay_substep = None
            self._action_epoch_control_outcome_next_index = 0
            self._action_epoch_control_substep_count = None

    def require_owned_action_epoch_current_settlement_delta(
        self,
        delta: ActionEpochR06CurrentSettlementDelta,
        *,
        expected_epoch_owner: object,
    ) -> ActionEpochR06CurrentSettlementDelta:
        """Consume the exact pending current-settlement delta once."""

        pending = self._pending_action_epoch_current_settlement_delta
        if (
            type(delta) is not ActionEpochR06CurrentSettlementDelta
            or pending is None
            or delta is not pending
            or delta.r06_owner is not self
            or delta.epoch_owner is not expected_epoch_owner
            or expected_epoch_owner is not self._action_ball_full_mdp_epoch_owner
            or delta._owner_identity is not self
            or delta._token is not _ACTION_EPOCH_R06_CURRENT_SETTLEMENT_DELTA_TOKEN
            or type(delta.sequence) is not int
            or delta.sequence != self._action_epoch_current_settlement_delta_sequence - 1
        ):
            raise LandingOutcomeDeviceError(
                "R06 current-settlement delta is stale, foreign, replayed, or owner-swapped"
            )
        self._pending_action_epoch_current_settlement_delta = None
        return delta

    def close_action_ball_full_mdp_epoch_reward_rows(self) -> None:
        """Fail-stop boundary for the owner-derived direct reward close."""

        owner = self._action_ball_full_mdp_epoch_owner
        if owner is None:
            raise LandingOutcomeDeviceError("R06 ActionEpoch owner is not bound")
        try:
            self._close_action_ball_full_mdp_epoch_reward_rows_impl()
        except BaseException:
            owner.poison_owner_write(
                "r06_landing_outcome", FAULT_SAFETY_CLEANUP, owner=self
            )
            raise

    def _close_action_ball_full_mdp_epoch_reward_rows_impl(self) -> None:
        """Close only a settled mailbox matching Epoch's exact paid-shot row."""

        owner = self._action_ball_full_mdp_epoch_owner
        if owner is None:
            raise LandingOutcomeDeviceError("R06 ActionEpoch owner is not bound")
        self._require_operable()
        if __package__:
            from . import action_ball_full_mdp_epoch as epoch_v1
        else:
            import action_ball_full_mdp_epoch as epoch_v1
        projector = getattr(owner, "project_current_reward_payment_rows", None)
        direct_projector = getattr(
            epoch_v1.ActionEpochOwner,
            "project_current_reward_payment_rows",
            None,
        )
        if (
            not callable(projector)
            or not callable(direct_projector)
            or getattr(projector, "__self__", None) is not owner
            or getattr(projector, "__func__", None) is not direct_projector
        ):
            raise LandingOutcomeDeviceError(
                "R06 payment close requires Epoch's exact no-arg projection"
            )
        payment = projector()
        if type(payment) is not epoch_v1.ActionEpochRewardPaymentRows:
            raise LandingOutcomeDeviceError(
                "R06 payment close projection type differs"
            )
        valid = self._tensor(
            payment.valid,
            name="reward_payment.valid",
            shape=(self.num_envs,),
            dtype=torch.bool,
        )
        payment_key = _row_identity.require_action_epoch_shot_key(
            payment.shot_key,
            shape=(self.num_envs,),
            device=self.device,
            label="R06 reward payment shot_key",
        )
        payment_step = self._tensor(
            payment.payment_step,
            name="reward_payment.payment_step",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        key_valid = _row_identity.action_epoch_shot_key_valid(payment_key)
        safe_rows = self._latch_action_epoch_row_fault(
            (valid & ~(key_valid & payment_step.ge(0))).contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_PAYMENT_PROJECTION_CONTRACT
            ),
        )
        valid = valid & safe_rows

        expanded_payment_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(payment_key, field.name)
                .unsqueeze(1)
                .expand(self._mailbox_shape)
                .contiguous()
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        mailbox_match = (
            valid.unsqueeze(1)
            & self._mailbox_action_epoch
            & self._mailbox_history_valid
            & self._mailbox_physical_retired
            & self._mailbox_state.eq(MAILBOX_SETTLED_UNPAID)
            & _row_identity.action_epoch_shot_key_equal(
                self._mailbox_action_epoch_shot_key(), expanded_payment_key
            )
        )
        safe_rows = self._latch_action_epoch_row_fault(
            mailbox_match.to(torch.int64).sum(dim=1).gt(1).contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_PAYMENT_MAILBOX_DUPLICATE
            ),
        )
        valid = valid & safe_rows
        mailbox_match = mailbox_match & safe_rows.unsqueeze(1)
        candidate = valid & mailbox_match.any(dim=1)
        previous_match = (
            self._previous_paid_valid
            & _row_identity.action_epoch_shot_key_equal(
                self._previous_paid_action_epoch_shot_key(), payment_key
            )
            & self._previous_paid_payment_step.eq(payment_step)
        )
        close_slot = torch.argmax(mailbox_match.to(torch.int64), dim=1)

        def gather(value: torch.Tensor) -> torch.Tensor:
            suffix = (1,) * (value.ndim - 2)
            index = close_slot.reshape(self.num_envs, 1, *suffix).expand(
                self.num_envs, 1, *value.shape[2:]
            )
            return torch.gather(value, 1, index).squeeze(1)

        retained_settlement_step = gather(
            self._mailbox_settlement_control_step
        )
        payment_before_settlement = (
            valid & mailbox_match.any(dim=1)
            & payment_step.lt(retained_settlement_step)
        )
        missing_or_mismatched = valid & ~candidate & ~previous_match
        highwater_regression = (
            valid
            & ~previous_match
            & candidate
            & payment_step.lt(self._previous_paid_payment_step_highwater)
        )
        unconsumed_debt_overwrite = (
            valid
            & ~previous_match
            & candidate
            & self._previous_paid_valid
        )
        self._latch_action_epoch_row_fault(
            missing_or_mismatched.contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_PAYMENT_MISSING_OR_MISMATCHED
            ),
        )
        self._latch_action_epoch_row_fault(
            payment_before_settlement.contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_PAYMENT_BEFORE_SETTLEMENT
            ),
        )
        self._latch_action_epoch_row_fault(
            highwater_regression.contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_PAYMENT_HIGHWATER_REGRESSION
            ),
        )
        safe_rows = self._latch_action_epoch_row_fault(
            unconsumed_debt_overwrite.contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_PAYMENT_UNCONSUMED_DEBT_OVERWRITE
            ),
        )
        candidate = candidate & safe_rows
        previous_match = previous_match & safe_rows
        close = mailbox_match & candidate.unsqueeze(1)

        for destination, source in (
            (self._previous_paid_action_uid, self._mailbox_action_uid),
            (self._previous_paid_action_slot, self._mailbox_action_slot),
            (self._previous_paid_reset_generation, self._mailbox_reset_generation),
            (self._previous_paid_shot_index, self._mailbox_shot_index),
            (self._previous_paid_task_identity, self._mailbox_task_identity),
            (self._previous_paid_outcome_identity, self._mailbox_outcome_identity),
            (self._previous_paid_ball_identity, self._mailbox_ball_identity),
            (
                self._previous_paid_publication_ordinal,
                self._mailbox_publication_ordinal,
            ),
            (self._previous_paid_ball_generation, self._mailbox_ball_generation),
            (
                self._previous_paid_observation_ordinal,
                self._mailbox_observation_ordinal,
            ),
            (
                self._previous_paid_settlement_control_step,
                self._mailbox_settlement_control_step,
            ),
            (
                self._previous_paid_selected_contact,
                self._mailbox_contact_valid,
            ),
            (
                self._previous_paid_first_crossing_valid,
                self._mailbox_crossing_valid,
            ),
            (
                self._previous_paid_on_opponent_table,
                self._mailbox_on_opponent_table,
            ),
            (
                self._previous_paid_target_error_m,
                self._mailbox_placement_error_m,
            ),
            (self._previous_paid_target_xy_m, self._mailbox_target_xy_m),
        ):
            _masked_copy_(destination, gather(source), candidate)
        _masked_copy_(self._previous_paid_payment_step, payment_step, candidate)
        _masked_copy_(
            self._previous_paid_payment_step_highwater,
            payment_step,
            candidate,
        )
        _masked_fill_(self._previous_paid_valid, candidate, 1)
        _masked_fill_(self._previous_paid_action_epoch, candidate, 1)

        _masked_fill_(self._mailbox_state, close, MAILBOX_EMPTY)
        _masked_fill_(self._mailbox_reserved, close, 0)
        _masked_fill_(self._mailbox_physical_retired, close, 0)
        _masked_fill_(self._mailbox_history_valid, close, 0)
        _masked_fill_(self._mailbox_action_epoch, close, 0)
        closed = close.to(torch.int64).sum()
        self._payment_totals.add_(closed)
        self._closed_total.add_(closed)
        self._mutation_version.add_(close.any().to(torch.int64))

    def project_previous_paid_action_epoch_rows(
        self,
    ) -> PreviousPaidActionEpochRows:
        """Return the durable paid-shot after-image without live-state aliases."""

        self._require_operable()
        return PreviousPaidActionEpochRows(
            valid=self._previous_paid_valid.detach().clone(),
            shot_key=self._previous_paid_action_epoch_shot_key().clone(),
            publication_ordinal=(
                self._previous_paid_publication_ordinal.detach().clone()
            ),
            settlement_step=(
                self._previous_paid_settlement_control_step.detach().clone()
            ),
            payment_step=self._previous_paid_payment_step.detach().clone(),
        )

    def consume_closed_action_epoch_rows(self) -> None:
        """Release only previous-paid debt closed by Epoch's exact row join."""

        owner = self._action_ball_full_mdp_epoch_owner
        if owner is None:
            raise LandingOutcomeDeviceError("R06 ActionEpoch owner is not bound")
        self._require_operable()
        if __package__:
            from . import action_ball_full_mdp_epoch as epoch_v1
        else:
            import action_ball_full_mdp_epoch as epoch_v1
        projector = getattr(owner, "project_current_closed_action_epoch_rows", None)
        direct_projector = getattr(
            epoch_v1.ActionEpochOwner,
            "project_current_closed_action_epoch_rows",
            None,
        )
        if (
            not callable(projector)
            or not callable(direct_projector)
            or getattr(projector, "__self__", None) is not owner
            or getattr(projector, "__func__", None) is not direct_projector
        ):
            raise LandingOutcomeDeviceError(
                "R06 debt release requires Epoch's exact no-arg row projection"
            )
        closed = projector(owner=self)
        if type(closed) is not epoch_v1.ActionEpochClosedRows:
            raise LandingOutcomeDeviceError("R06 closed-row projection type differs")
        valid = self._tensor(
            closed.valid,
            name="closed_action_epoch.valid",
            shape=(self.num_envs,),
            dtype=torch.bool,
        )
        closed_key = _row_identity.require_action_epoch_shot_key(
            closed.shot_key,
            shape=(self.num_envs,),
            device=self.device,
            label="R06 closed action epoch shot_key",
        )
        key_valid = _row_identity.action_epoch_shot_key_valid(closed_key)
        safe_rows = self._latch_action_epoch_row_fault(
            (valid & ~key_valid).contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_CLOSED_PROJECTION_CONTRACT
            ),
        )
        valid = valid & safe_rows
        release = (
            valid
            & self._previous_paid_valid
            & _row_identity.action_epoch_shot_key_equal(
                self._previous_paid_action_epoch_shot_key(), closed_key
            )
        )
        safe_rows = self._latch_action_epoch_row_fault(
            (valid & ~release).contiguous(),
            epoch_reason_bit=R06_EPOCH_ROW_FAULT_CLOSED_DEBT_MISMATCH,
        )
        release = release & safe_rows
        for destination in (
            self._previous_paid_action_uid,
            self._previous_paid_action_slot,
            self._previous_paid_reset_generation,
            self._previous_paid_shot_index,
            self._previous_paid_task_identity,
            self._previous_paid_outcome_identity,
            self._previous_paid_ball_identity,
            self._previous_paid_publication_ordinal,
            self._previous_paid_ball_generation,
            self._previous_paid_observation_ordinal,
            self._previous_paid_settlement_control_step,
            self._previous_paid_payment_step,
        ):
            _masked_fill_(destination, release, -1)
        for destination in (
            self._previous_paid_valid,
            self._previous_paid_action_epoch,
            self._previous_paid_selected_contact,
            self._previous_paid_first_crossing_valid,
            self._previous_paid_on_opponent_table,
        ):
            _masked_fill_(destination, release, 0)
        for destination in (
            self._previous_paid_target_error_m,
            self._previous_paid_target_xy_m,
            self._previous_paid_full_key_sha256,
        ):
            _masked_fill_(destination, release, 0)
        for name, destination in self._previous_paid_key_ints.items():
            _masked_fill_(destination, release, -1 if name == "env_id" else 0)
        for destination in self._previous_paid_key_digests.values():
            _masked_fill_(destination, release, 0)
        self._mutation_version.add_(release.any().to(torch.int64))

    def require_owned_action_epoch_current_flight_observation(
        self,
        projection: ActionBallFullMdpObservationProjection,
        *,
        current_shot_key: _row_identity.ActionEpochShotKey,
        current_publication_ordinal: torch.Tensor,
    ) -> ActionEpochR06CurrentFlightObservationView:
        """Select the current live typed flight at a closed owner boundary.

        The caller supplies only the current public Epoch identity and its
        non-identity publication chronology.  R06 performs the exact join
        against its private typed flight plane and returns no identity echo,
        mailbox, Reward, receipt, or raw storage view.
        """

        self._require_operable()
        if (
            type(projection) is not ActionBallFullMdpObservationProjection
            or projection is not self._full_mdp_observation_projection
        ):
            raise LandingOutcomeDeviceError(
                "R06 full-MDP observation projection is forged or foreign"
            )
        try:
            expected_key = _row_identity.require_action_epoch_shot_key(
                current_shot_key,
                shape=(self.num_envs,),
                device=self.device,
                label="R06 current observation shot_key",
            )
        except _row_identity.ActionEpochShotKeyError as exc:
            raise LandingOutcomeDeviceError(str(exc)) from exc
        if (
            type(current_publication_ordinal) is not torch.Tensor
            or tuple(current_publication_ordinal.shape) != (self.num_envs,)
            or current_publication_ordinal.dtype != torch.int64
            or current_publication_ordinal.device != self.device
            or current_publication_ordinal.layout != torch.strided
            or not current_publication_ordinal.is_contiguous()
        ):
            raise LandingOutcomeDeviceError(
                "R06 current observation publication_ordinal ABI differs"
            )
        expected_publication = current_publication_ordinal.detach()
        try:
            flight_key = _row_identity.require_action_epoch_shot_key(
                self._flight_action_epoch_shot_key(),
                shape=self._flight_shape,
                device=self.device,
                label="R06 private flight shot_key",
            )
        except _row_identity.ActionEpochShotKeyError as exc:
            raise LandingOutcomeDeviceError(str(exc)) from exc

        matches = (
            self._flight_action_epoch
            & self._flight_state.ne(FLIGHT_EMPTY)
            & _row_identity.action_epoch_shot_key_valid(flight_key)
            & _row_identity.action_epoch_shot_key_valid(expected_key)[:, None]
            & self._flight_publication_ordinal.ge(0)
            & expected_publication.ge(0)[:, None]
            & self._flight_publication_ordinal.eq(expected_publication[:, None])
        )
        for field in fields(_row_identity.ActionEpochShotKey):
            matches &= getattr(flight_key, field.name).eq(
                getattr(expected_key, field.name)[:, None]
            )
        safe_rows = self._latch_action_epoch_row_fault(
            matches.to(torch.int64).sum(dim=1).gt(1).contiguous(),
            epoch_reason_bit=(
                R06_EPOCH_ROW_FAULT_CURRENT_FLIGHT_DUPLICATE
            ),
        )
        matches = matches & safe_rows.unsqueeze(1)
        selected = torch.argmax(matches.to(torch.int64), dim=1)
        present = matches.any(dim=1)

        def gather(value: torch.Tensor) -> torch.Tensor:
            return torch.gather(value, 1, selected[:, None]).squeeze(1)

        selected_state = gather(self._flight_state)
        live = present & (
            selected_state.eq(FLIGHT_INBOUND) | selected_state.eq(FLIGHT_OPEN)
        )
        invalid_slot = torch.full_like(selected, -1)

        def selected_latch(value: torch.Tensor) -> torch.Tensor:
            gathered = gather(value)
            return torch.where(live, gathered, torch.zeros_like(gathered)).detach()

        return ActionEpochR06CurrentFlightObservationView(
            r06_owner=self,
            publication_identity=projection,
            flight_slot=torch.where(live, selected, invalid_slot).detach(),
            contact_valid=selected_latch(self._flight_contact_valid),
            net_crossed=selected_latch(self._flight_net_crossed),
            net_clear=selected_latch(self._flight_net_clear),
        )

    def _flight_lifecycle_snapshot(self) -> FlightLifecycleSnapshotBatch:
        key = self._key_storage("flight")
        mailbox_key = self._key_storage("mailbox")
        return FlightLifecycleSnapshotBatch(
            state=self._flight_state.detach().clone(),
            task_key=DeviceLandingOutcomeKey(
                **{
                    name: getattr(key, name).detach().clone()
                    for name in _KEY_FIELDS
                }
            ),
            full_key_sha256=self._flight_full_key_sha256.detach().clone(),
            ball_generation=self._flight_ball_generation.detach().clone(),
            mailbox_slot=self._flight_mailbox_slot.detach().clone(),
            observation_ordinal=(
                self._flight_observation_ordinal.detach().clone()
            ),
            physical_retired=self._flight_physical_retired.detach().clone(),
            mailbox_state=self._mailbox_state.detach().clone(),
            mailbox_task_key=DeviceLandingOutcomeKey(
                **{
                    name: getattr(mailbox_key, name).detach().clone()
                    for name in _KEY_FIELDS
                }
            ),
            mailbox_full_key_sha256=(
                self._mailbox_full_key_sha256.detach().clone()
            ),
            mailbox_ball_generation=(
                self._mailbox_ball_generation.detach().clone()
            ),
            mailbox_reserved_flight_slot=(
                self._mailbox_reserved_flight_slot.detach().clone()
            ),
            mailbox_history_valid=(
                self._mailbox_history_valid.detach().clone()
            ),
            mailbox_physical_retired=(
                self._mailbox_physical_retired.detach().clone()
            ),
            mutation_version=self._mutation_version.detach().clone(),
        )

    def reveal_boundary_child_token_authority(
        self,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority:
        """Return R06's one stable validator backed by its retained attempt."""

        authority = self._reveal_boundary_child_token_authority
        if (
            type(authority)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority
            or authority.owner_kind != "r06_flight"
        ):
            raise LandingOutcomeDeviceError(
                "R06 reveal-boundary child authority drifted"
            )
        return authority

    def reveal_boundary_fault_schema(
        self,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryFaultSchema:
        """Return R06's exact frozen lane schema for top-level construction."""

        return R06_REVEAL_BOUNDARY_FAULT_SCHEMA

    def bind_r05_terminal_owner(
        self,
        r05_owner: _r05.ContinuousRuntimeTransactionOwner,
    ) -> None:
        """Bind the sole R05 issuer of future terminal claims and receipts."""

        self._require_operable()
        self._require_formal_only("bind the formal R05 terminal owner")
        source_path = Path(_r05.__file__).resolve()
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        expected_r05_source = (
            R05_RUNTIME_TRANSACTION_SOURCE_SHA256
            or R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256
        )
        terminal_boundary_authority = getattr(
            r05_owner, "terminal_boundary_authority", None
        )
        owned_terminal_boundary_authority = None
        if (
            type(r05_owner) is _r05.ContinuousRuntimeTransactionOwner
            and type(terminal_boundary_authority)
            is _r05.TerminalBoundaryAuthority
        ):
            try:
                owned_terminal_boundary_authority = (
                    r05_owner.require_owned_terminal_boundary_authority(
                        terminal_boundary_authority,
                        expected_authority_sha256=(
                            terminal_boundary_authority.canonical_sha256
                        ),
                        expected_authority_domain=(
                            FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
                        ),
                        expected_authority_schema_sha256=(
                            FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
                        ),
                        expected_authority_source_sha256=(
                            FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256
                        ),
                    )
                )
            except Exception:
                owned_terminal_boundary_authority = None
        if (
            type(r05_owner) is not _r05.ContinuousRuntimeTransactionOwner
            or source_path.name
            != "action_ball_continuous_runtime_transaction.py"
            or source_sha256 != expected_r05_source
            or self.runtime_binding.r05_source_sha256 != source_sha256
            or type(terminal_boundary_authority)
            is not _r05.TerminalBoundaryAuthority
            or owned_terminal_boundary_authority
            is not terminal_boundary_authority
            or terminal_boundary_authority.status != "bound"
            or terminal_boundary_authority.authority_domain
            != FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
            or terminal_boundary_authority.authority_schema_sha256
            != FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
            or terminal_boundary_authority.authority_source_sha256
            != FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256
        ):
            raise LandingOutcomeDeviceError(
                "R06 terminal claims require the exact pinned R05 owner"
            )
        current = self._r05_terminal_owner
        if current is not None and current is not r05_owner:
            raise LandingOutcomeDeviceError(
                "R06 terminal-claim owner may not be rebound"
            )
        if current is r05_owner:
            return
        self._r05_terminal_owner = r05_owner
        self._r05_terminal_boundary_authority = terminal_boundary_authority
        self._r05_terminal_source_sha256 = source_sha256

    def _r05_terminal_binding(
        self,
    ) -> _r05.ContinuousRuntimeTransactionOwner:
        owner = self._r05_terminal_owner
        authority = self._r05_terminal_boundary_authority
        if (
            type(owner) is not _r05.ContinuousRuntimeTransactionOwner
            or type(authority) is not _r05.TerminalBoundaryAuthority
            or owner.terminal_boundary_authority is not authority
            or authority.status != "bound"
            or self._r05_terminal_source_sha256
            != self.runtime_binding.r05_source_sha256
        ):
            raise LandingOutcomeDeviceError(
                "R06 R05 terminal-claim owner is not bound"
            )
        return owner

    def bind_device_r05_reset_owner(
        self,
        r05_owner: _r05_device.DeviceR05Owner,
        *,
        prepared_reset_validator: object,
        r05_receipt_validator: object,
    ) -> None:
        """Bind Device-R05 through its exact construction genesis registry."""

        self._require_operable()
        self._require_formal_only("bind formal Device-R05 reset authority")
        self._bind_device_r05_reset_owner_exact(
            r05_owner,
            prepared_reset_validator=prepared_reset_validator,
            r05_receipt_validator=r05_receipt_validator,
            expected_world_reset_identity=None,
        )

    def bind_diagnostic_n2_device_r05_reset_owner(
        self,
        r05_owner: _r05_device.DeviceR05Owner,
        *,
        prepared_reset_validator: object,
        r05_receipt_validator: object,
    ) -> None:
        """Bind the exact diagnostic D05 child without granting formal authority."""

        self._require_operable()
        record = self._diagnostic_n2_construction_record
        physical_owner = (
            None if record is None else record.physical_owner
        )
        epoch_owner = getattr(physical_owner, "_action_epoch_owner", None)
        if (
            not self._diagnostic_n2_no_save
            or record is None
            or physical_owner is None
            or type(r05_owner) is not _r05_device.DeviceR05Owner
            or getattr(r05_owner, "_diagnostic_physical_owner", None)
            is not physical_owner
            or getattr(r05_owner, "_diagnostic_epoch_owner", None)
            is not epoch_owner
            or epoch_owner is None
        ):
            raise LandingOutcomeDeviceError(
                "diagnostic R06 requires the exact same diagnostic D05/Physical/epoch graph"
            )
        world_reset_identity = getattr(
            physical_owner, "_genesis_world_reset_identity", None
        )
        if world_reset_identity is None:
            raise LandingOutcomeDeviceError(
                "diagnostic R06/D05 world-reset identity is absent"
            )
        self._bind_device_r05_reset_owner_exact(
            r05_owner,
            prepared_reset_validator=prepared_reset_validator,
            r05_receipt_validator=r05_receipt_validator,
            expected_world_reset_identity=world_reset_identity,
        )

    def _bind_device_r05_reset_owner_exact(
        self,
        r05_owner: _r05_device.DeviceR05Owner,
        *,
        prepared_reset_validator: object,
        r05_receipt_validator: object,
        expected_world_reset_identity: object | None,
    ) -> None:
        """Consume one exact D05 genesis; this helper grants no run authority."""

        if not self._load_fresh:
            raise LandingOutcomeDeviceError(
                "R06 Device-R05 reset owner must bind before business mutation"
            )
        # Device-R05 state-view properties deliberately close its construction
        # window.  Bind through the exact opaque genesis registry first; its
        # clone-only [N] view below owns both shape and device compatibility.
        if type(r05_owner) is not _r05_device.DeviceR05Owner:
            raise LandingOutcomeDeviceError(
                "R06 selected reset requires the exact device R05 owner"
            )
        if self._device_r05_reset_owner is not None:
            raise LandingOutcomeDeviceError(
                "R06 device R05 reset owner may not be rebound"
            )
        genesis_projector = getattr(
            r05_owner, "project_owned_genesis_for_child", None
        )
        genesis_validator = getattr(
            r05_owner,
            "require_owned_genesis_projection",
            None,
        )
        if (
            not callable(genesis_projector)
            or not callable(genesis_validator)
            or not callable(prepared_reset_validator)
            or getattr(prepared_reset_validator, "__self__", None)
            is not r05_owner
            or getattr(prepared_reset_validator, "__func__", None)
            is not getattr(
                type(r05_owner),
                "require_owned_prepared_true_reset",
                None,
            )
            or not callable(r05_receipt_validator)
            or getattr(r05_receipt_validator, "__self__", None)
            is not r05_owner
            or getattr(r05_receipt_validator, "__func__", None)
            is not getattr(
                type(r05_owner),
                "require_owned_true_reset_receipt",
                None,
            )
        ):
            raise LandingOutcomeDeviceError(
                "R06 Device-R05 reset validators differ"
            )
        try:
            genesis_projection = genesis_projector(owner_kind="r06_flight")
            genesis = genesis_validator(
                genesis_projection,
                owner_kind="r06_flight",
            )
            repeated = genesis_validator(
                genesis_projection,
                owner_kind="r06_flight",
            )
        except Exception as exc:
            raise LandingOutcomeDeviceError(
                "R06 Device-R05 genesis authority differs"
            ) from exc
        generation = getattr(genesis, "reset_generation", None)
        if (
            type(genesis_projection)
            is not _r05_device.DeviceR05GenesisProjection
            or type(genesis) is not _r05_device.DeviceR05GenesisView
            or type(repeated) is not _r05_device.DeviceR05GenesisView
            or getattr(genesis, "device_r05_owner", None) is not r05_owner
            or getattr(genesis, "owner_kind", None) != "r06_flight"
            or getattr(genesis, "world_reset_identity", None) is None
            or (
                expected_world_reset_identity is not None
                and getattr(genesis, "world_reset_identity", None)
                is not expected_world_reset_identity
            )
            or repeated.device_r05_owner is not r05_owner
            or repeated.owner_kind != "r06_flight"
            or repeated.world_reset_identity
            is not genesis.world_reset_identity
            or not isinstance(generation, torch.Tensor)
            or tuple(generation.shape) != (self.num_envs,)
            or generation.dtype != torch.int64
            or generation.device != self.device
            or not isinstance(repeated.reset_generation, torch.Tensor)
            or repeated.reset_generation.device != self.device
            or repeated.reset_generation.dtype != torch.int64
            or tuple(repeated.reset_generation.shape) != (self.num_envs,)
            or not torch.equal(repeated.reset_generation, generation)
        ):
            raise LandingOutcomeDeviceError(
                "R06 Device-R05 genesis projection ABI differs"
            )
        self._device_r05_reset_owner = r05_owner
        self._device_r05_genesis_projection = genesis_projection
        self._device_r05_prepared_reset_validator = prepared_reset_validator
        self._device_r05_receipt_validator = r05_receipt_validator
        # Genesis is the sole construction authority for reset chronology.
        # Establish every row on device before any reveal/reset mutation; never
        # infer an initial generation from a later selected host tuple.
        self._reset_generation_highwater.copy_(generation)

    def bind_device_r05_hot_reveal(
        self,
        device_r05_owner: _r05_device.DeviceR05Owner,
        *,
        physical_late_launch_owner: object,
    ) -> None:
        """Construction-bind the future causal R06 hot-reveal ingress.

        The currently frozen Device-R05 child projection does not publish the
        complete R06 key: ``full_key_sha256``, the receipt ``task_sha256``, and
        ``action_uid`` are absent.  R06 must not derive those values from its
        own future destination rows, accept a portable R05 object graph, or
        reinterpret ``task_identity`` as any of the missing facts.  Therefore
        the exact production owner is authenticated first and this binder
        then fails with a typed HOLD before retaining either owner.

        Once the upstream ABI is extended, this same callpoint will also bind
        the Physical owner's exact late-launch publication validator.  Until
        then there is deliberately no fixture-success or caller-authored
        fallback path.
        """

        self._require_operable()
        self._require_formal_only("bind formal Device-R05 hot reveal authority")
        projection_validator = getattr(
            device_r05_owner,
            "require_owned_prepared_reveal_for_child",
            None,
        )
        if (
            type(device_r05_owner) is not _r05_device.DeviceR05Owner
            or not callable(projection_validator)
            or getattr(projection_validator, "__self__", None)
            is not device_r05_owner
            or getattr(projection_validator, "__func__", None)
            is not getattr(
                _r05_device.DeviceR05Owner,
                "require_owned_prepared_reveal_for_child",
                None,
            )
        ):
            raise LandingOutcomeDeviceError(
                "R06 hot reveal rejects portable, foreign, or partial R05 owners"
            )
        projection_fields = frozenset(
            getattr(
                _r05_device.DeviceR05PreparedRevealProjection,
                "__dataclass_fields__",
                {},
            )
        )
        missing = tuple(
            name
            for name in DEVICE_R05_HOT_REVEAL_REQUIRED_PROJECTION_FIELDS
            if name not in projection_fields
        )
        if missing:
            # Do not retain either owner on failure.  In particular, a later
            # factory retry after the causal ABI is upgraded must start from a
            # clean construction state rather than a half-bound authority.
            raise LandingOutcomeDeviceR05HotRevealProductionHold(
                "Device-R05 R06 child projection is missing causal fields: "
                + ", ".join(missing)
            )
        # The code below is intentionally unreachable for the current ABI.
        # It names the future independent Physical validation seam without
        # granting authority to a similarly named caller method.
        physical_validator = getattr(
            physical_late_launch_owner,
            "require_owned_late_launch_publication",
            None,
        )
        if not callable(physical_validator):
            raise LandingOutcomeDeviceR05HotRevealProductionHold(
                "Physical exact late-launch publication authority is unavailable"
            )
        raise LandingOutcomeDeviceR05HotRevealProductionHold(
            "R06 hot reveal retained after-image/arm/commit ABI is not frozen"
        )

    def _device_r05_reset_binding(self) -> _r05_device.DeviceR05Owner:
        owner = self._device_r05_reset_owner
        genesis_projection = self._device_r05_genesis_projection
        prepared_validator = self._device_r05_prepared_reset_validator
        receipt_validator = self._device_r05_receipt_validator
        if (
            type(owner) is not _r05_device.DeviceR05Owner
            or type(genesis_projection)
            is not _r05_device.DeviceR05GenesisProjection
            or not callable(prepared_validator)
            or getattr(prepared_validator, "__self__", None) is not owner
            or getattr(prepared_validator, "__func__", None)
            is not getattr(
                _r05_device.DeviceR05Owner,
                "require_owned_prepared_true_reset",
                None,
            )
            or not callable(receipt_validator)
            or getattr(receipt_validator, "__self__", None) is not owner
            or getattr(receipt_validator, "__func__", None)
            is not getattr(
                _r05_device.DeviceR05Owner,
                "require_owned_true_reset_receipt",
                None,
            )
        ):
            raise LandingOutcomeDeviceError(
                "R06 device R05 reset owner is not bound"
            )
        return owner

    def bind_reveal_boundary(
        self,
        boundary_owner: _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner,
    ) -> None:
        """Bind exactly R06's lane in the sole production packed owner."""

        self._require_operable()
        self._require_formal_only("bind the formal reveal boundary")
        if (
            type(boundary_owner)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner
        ):
            raise LandingOutcomeDeviceError(
                "R06 requires the exact all-owner reveal-boundary owner"
            )
        source_path = Path(_reveal_boundary.__file__).resolve()
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            source_path.name != "action_ball_full_mdp_reveal_boundary.py"
            or source_sha256
            != FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256
        ):
            raise LandingOutcomeDeviceError(
                "R06 reveal-boundary source pin differs"
            )
        authority = self.reveal_boundary_child_token_authority()
        lane = boundary_owner.lane_authority("r06_flight")
        schema = lane.fault_schema
        if (
            boundary_owner.child_token_authorities[3] is not authority
            or type(lane)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority
            or lane.owner_kind != "r06_flight"
            or lane.child_token_authority is not authority
            or boundary_owner.lane_authority("r06_flight") is not lane
            or boundary_owner.num_envs != self.num_envs
            or boundary_owner.device != self.device
            or type(schema)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryFaultSchema
            or schema.owner_kind != "r06_flight"
            or schema.schema_sha256
            != R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256
            or schema.to_mapping()
            != R06_REVEAL_BOUNDARY_FAULT_SCHEMA.to_mapping()
        ):
            raise LandingOutcomeDeviceError(
                "R06 reveal-boundary lane/schema differs"
            )
        current = self._reveal_boundary_owner
        if current is not None and current is not boundary_owner:
            raise LandingOutcomeDeviceError(
                "R06 reveal-boundary owner may not be rebound"
            )
        if current is boundary_owner:
            return
        self._reveal_boundary_owner = boundary_owner
        self._reveal_boundary_lane = lane
        self._reveal_boundary_source_sha256 = source_sha256

    def _reveal_boundary_binding(
        self,
    ) -> tuple[
        _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner,
        _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority,
    ]:
        owner = self._reveal_boundary_owner
        lane = self._reveal_boundary_lane
        authority = self.reveal_boundary_child_token_authority()
        if (
            type(owner)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner
            or type(lane)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority
            or owner.lane_authority("r06_flight") is not lane
            or lane.child_token_authority is not authority
            or lane.fault_schema.schema_sha256
            != R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256
            or self._reveal_boundary_source_sha256
            != FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256
        ):
            raise LandingOutcomeDeviceError(
                "R06 reveal-boundary owner/lane is not bound"
            )
        return owner, lane

    def _require_owned_reveal_prepared_token(
        self, prepared_token: object
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryPreparedTokenClaim:
        lease = self._active_reveal_prepare_lease
        if (
            self._poisoned
            or lease is None
            or prepared_token is not lease.attempt
            or type(prepared_token) is not LandingRevealPrepareAttempt
            or prepared_token._owner_identity is not self._owner_identity
            or prepared_token._token is not _PREPARED_INSTALL_AUTH_TOKEN
            or prepared_token._device_owner_mutation_version
            is not self._mutation_version
        ):
            raise LandingOutcomeDeviceError(
                "R06 reveal prepared token is forged, stale, or foreign"
            )
        return _reveal_boundary.ActionBallFullMdpRevealBoundaryPreparedTokenClaim(
            owner_kind="r06_flight",
            device_owner_mutation_version=(
                prepared_token._device_owner_mutation_version
            ),
            owner_token_root_sha256=prepared_token._receipt.canonical_sha256,
            reveal_final_preview_schema_version=(
                _r05.RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
            ),
            reveal_final_preview_sha256=(
                prepared_token._receipt.reveal_final_preview_sha256
            ),
            _prepared_token=prepared_token,
        )

    def bind_physical_park_token_authority(
        self,
        physical_owner: object,
        authority: LandingOutcomePhysicalParkTokenAuthority,
    ) -> None:
        """Bind the one physical owner allowed to arm/commit retirement."""

        self._require_operable()
        if type(authority) is not LandingOutcomePhysicalParkTokenAuthority:
            raise LandingOutcomeDeviceError(
                "physical park token authority type differs"
            )
        if authority.physical_owner is not physical_owner:
            raise LandingOutcomeDeviceError(
                "physical park owner/authority identity differs"
            )
        if getattr(physical_owner, "_r06_owner", None) is not self:
            raise LandingOutcomeDeviceError(
                "physical park owner did not register this exact R06 pair"
            )
        current = self._physical_park_token_authority
        if current is not None and current is not authority:
            raise LandingOutcomeDeviceError(
                "physical park token authority may not be rebound"
            )
        self._physical_park_token_authority = authority

    def bind_selected_reset_physical_park_token_authority(
        self,
        physical_owner: object,
        authority: LandingOutcomeSelectedResetPhysicalParkTokenAuthority,
    ) -> None:
        """Bind the only physical owner allowed to park selected rows."""

        self._require_operable()
        if (
            type(authority)
            is not LandingOutcomeSelectedResetPhysicalParkTokenAuthority
            or authority.physical_owner is not physical_owner
            or getattr(physical_owner, "_r06_owner", None) is not self
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset physical park owner/authority differs"
            )
        current = self._selected_reset_physical_park_token_authority
        if current is not None and current is not authority:
            raise LandingOutcomeDeviceError(
                "selected-reset physical park authority may not be rebound"
            )
        self._selected_reset_physical_park_token_authority = authority

    def _zero_counter(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.int64, device=self.device)

    def _filled_int(self, shape: tuple[int, ...], value: int) -> torch.Tensor:
        return torch.full(shape, value, dtype=torch.int64, device=self.device)

    def _zeros_float(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _zeros_token(self, prefix: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(prefix + (TOKEN_BYTES,), dtype=torch.uint8, device=self.device)

    def _new_key_int_buffers(self, prefix: tuple[int, ...]) -> dict[str, torch.Tensor]:
        result = {
            name: torch.zeros(prefix, dtype=torch.int64, device=self.device)
            for name in _INT_KEY_FIELDS
        }
        result["env_id"].fill_(-1)
        return result

    def _new_key_digest_buffers(
        self, prefix: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        return {name: self._zeros_token(prefix) for name in _DIGEST_KEY_FIELDS}

    def _tensor(
        self,
        value: object,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != self.device
        ):
            raise LandingOutcomeDeviceError(
                f"{name} must be device-local {dtype} shape {list(shape)}"
            )
        return value.detach()

    def _float_tensor(
        self, value: object, *, name: str, shape: tuple[int, ...]
    ) -> torch.Tensor:
        return self._tensor(value, name=name, shape=shape, dtype=self.dtype)

    def _key(
        self,
        value: DeviceLandingOutcomeKey | Mapping[str, torch.Tensor],
        *,
        prefix: tuple[int, ...],
    ) -> DeviceLandingOutcomeKey:
        key = value if isinstance(value, DeviceLandingOutcomeKey) else DeviceLandingOutcomeKey.from_mapping(value)
        for name in _INT_KEY_FIELDS:
            self._tensor(
                getattr(key, name), name=f"task_key.{name}", shape=prefix, dtype=torch.int64
            )
        for name in _DIGEST_KEY_FIELDS:
            self._tensor(
                getattr(key, name),
                name=f"task_key.{name}",
                shape=prefix + (TOKEN_BYTES,),
                dtype=torch.uint8,
            )
        return key

    def _stamp(
        self, value: PhysicsStampBatch, *, name: str, prefix: tuple[int, ...]
    ) -> PhysicsStampBatch:
        if not isinstance(value, PhysicsStampBatch):
            raise LandingOutcomeDeviceError(f"{name} must be PhysicsStampBatch")
        return PhysicsStampBatch(
            control_step=self._tensor(
                value.control_step,
                name=f"{name}.control_step",
                shape=prefix,
                dtype=torch.int64,
            ),
            physics_substep=self._tensor(
                value.physics_substep,
                name=f"{name}.physics_substep",
                shape=prefix,
                dtype=torch.int32,
            ),
            event_phase=self._tensor(
                value.event_phase,
                name=f"{name}.event_phase",
                shape=prefix,
                dtype=torch.int8,
            ),
        )

    def _key_storage(self, grid: str) -> DeviceLandingOutcomeKey:
        ints = self._flight_key_ints if grid == "flight" else self._mailbox_key_ints
        digests = (
            self._flight_key_digests if grid == "flight" else self._mailbox_key_digests
        )
        return DeviceLandingOutcomeKey(**{**ints, **digests})

    def _or_faults_(
        self, destination: torch.Tensor, mask: torch.Tensor, bits: torch.Tensor
    ) -> None:
        destination.copy_(torch.where(mask, torch.bitwise_or(destination, bits), destination))

    def _increment_mutation(self) -> None:
        self._mutation_version.add_(1)
        self._latest_receipt = None
        self._latest_global_drain_receipt = None
        self._latest_checkpoint_live_mutation_projection = None
        self._latest_receipt_consumed = False
        self._load_fresh = False

    def _record_fault_events(self, fault_bits: torch.Tensor) -> None:
        flattened = fault_bits.reshape(-1)
        for index, (_, bit) in enumerate(FAULTS):
            self._fault_event_counts[index].add_(
                (torch.bitwise_and(flattened, bit) != 0).to(torch.int64).sum()
            )

    def _accumulate_normal_retire_key_summaries_(
        self,
        destination: torch.Tensor,
        normal_mask: torch.Tensor,
        full_key_sha256: torch.Tensor,
    ) -> None:
        """Accumulate the physical/R06 shared key multiset on device."""

        key_bytes = full_key_sha256.to(torch.int64) + 1
        summaries = []
        for base in (257, 263):
            weights = torch.tensor(
                tuple(
                    pow(base, index, _R06_RETIRE_SUMMARY_MODULUS)
                    for index in range(TOKEN_BYTES)
                ),
                dtype=torch.int64,
                device=self.device,
            )
            row_hash = torch.remainder(
                (key_bytes * weights).sum(dim=-1),
                _R06_RETIRE_SUMMARY_MODULUS,
            )
            summaries.append(
                torch.remainder(
                    (row_hash * normal_mask.to(torch.int64)).sum(),
                    _R06_RETIRE_SUMMARY_MODULUS,
                )
            )
        batch = torch.stack(summaries)
        destination.copy_(
            torch.remainder(
                destination + batch,
                _R06_RETIRE_SUMMARY_MODULUS,
            )
        )

    @staticmethod
    def _mutation_result(
        accepted: torch.Tensor,
        rejected: torch.Tensor,
        fault_bits: torch.Tensor,
    ) -> DeviceMutationResult:
        return DeviceMutationResult(
            accepted=accepted.detach().clone(),
            rejected=rejected.detach().clone(),
            fault_bits=fault_bits.detach().clone(),
        )

    def _require_operable(
        self,
        *,
        allow_active_prepare: bool = False,
        allow_active_physical_retire: bool = False,
        allow_active_selected_reset: bool = False,
        allow_pending_post_physics_settlement: bool = False,
        allow_pending_post_physics_contact_authority: bool = False,
        allow_pending_action_epoch_current_settlement_delta: bool = False,
        allow_action_epoch_control_window: bool = False,
        allow_full_mdp_reward_cycle: bool = False,
    ) -> None:
        if self._poisoned:
            raise LandingOutcomeDeviceError(
                "landing outcome owner is poisoned and requires cold replacement"
            )
        if self._r06_global_drain_poisoned:
            raise LandingOutcomeDeviceError(
                "R06 global drain is poisoned and requires cold replacement"
            )
        if self._full_mdp_reward_poisoned and not allow_full_mdp_reward_cycle:
            raise LandingOutcomeDeviceError(
                "R06 full-MDP Reward is poisoned with retained close debt"
            )
        if self._active_r06_global_drain is not None:
            raise LandingOutcomeDeviceError(
                "global PPO drain lease blocks R06 mutation/drain/checkpoint"
            )
        if self._active_reveal_prepare_lease is not None and not allow_active_prepare:
            raise LandingOutcomeDeviceError(
                "exclusive reveal-prepare lease blocks owner mutation/drain/checkpoint"
            )
        if (
            self._active_physical_retire_lease is not None
            and not allow_active_physical_retire
        ):
            raise LandingOutcomeDeviceError(
                "exclusive physical-retire lease blocks owner mutation/drain/checkpoint"
            )
        if (
            self._active_selected_reset_lease is not None
            and not allow_active_selected_reset
        ):
            raise LandingOutcomeDeviceError(
                "exclusive selected-reset lease blocks owner mutation/drain/checkpoint"
            )
        if (
            self._latest_post_physics_settlement is not None
            and not allow_pending_post_physics_settlement
        ):
            raise LandingOutcomeDeviceError(
                "unclosed post-physics authority blocks owner mutation/drain/checkpoint"
            )
        if (
            self._active_post_physics_contact_authority is not None
            and not allow_pending_post_physics_contact_authority
        ):
            raise LandingOutcomeDeviceError(
                "unconsumed post-physics contact authority blocks owner mutation/drain/checkpoint"
            )
        if (
            self._pending_action_epoch_current_settlement_delta is not None
            and not allow_pending_action_epoch_current_settlement_delta
        ):
            raise LandingOutcomeDeviceError(
                "unconsumed ActionEpoch current-settlement delta blocks owner mutation/drain/checkpoint"
            )
        if (
            self._action_epoch_control_substep_count is not None
            and not allow_action_epoch_control_window
        ):
            raise LandingOutcomeDeviceError(
                "open ActionEpoch postphysics control window blocks unrelated mutation/drain/checkpoint"
            )
        if (
            (
                self._active_full_mdp_reward_cycle_identity is not None
                or self._full_mdp_reward_close_debt
            )
            and not allow_full_mdp_reward_cycle
        ):
            raise LandingOutcomeDeviceError(
                "open R06 full-MDP Reward cycle is stale debt and blocks reset/checkpoint/global drain"
            )

    def _new_prepare_shadow(self) -> "ActionBallLandingOutcomeDeviceCoordinator":
        self._require_formal_only("allocate a formal reveal-prepare shadow")
        shadow = ActionBallLandingOutcomeDeviceCoordinator(
            num_envs=self.num_envs,
            flight_slot_capacity=self.flight_slot_capacity,
            mailbox_capacity=self.mailbox_capacity,
            device=self.device,
            dtype=self.dtype,
            profile=self.profile,
            runtime_binding=self.runtime_binding,
            payment_authority=self.payment_authority,
            capacity_authority=self.capacity_authority,
            text_registry=self.text_registry,
        )
        live_tensors = self._checkpoint_tensors()
        shadow_tensors = shadow._checkpoint_tensors()
        if set(live_tensors) != set(shadow_tensors):
            raise LandingOutcomeDeviceError("prepare shadow tensor schema differs")
        for name in sorted(live_tensors):
            shadow_tensors[name].copy_(live_tensors[name])
        shadow._drain_sequence = self._drain_sequence
        shadow._last_drained_update_index = self._last_drained_update_index
        shadow._latest_receipt = self._latest_receipt
        shadow._latest_receipt_consumed = self._latest_receipt_consumed
        shadow._load_fresh = self._load_fresh
        return shadow

    def _fresh_reset_template(
        self,
    ) -> "ActionBallLandingOutcomeDeviceCoordinator":
        """Allocate owner defaults for a rare selected-reset after-image."""

        self._require_formal_only("allocate a formal selected-reset template")
        return ActionBallLandingOutcomeDeviceCoordinator(
            num_envs=self.num_envs,
            flight_slot_capacity=self.flight_slot_capacity,
            mailbox_capacity=self.mailbox_capacity,
            device=self.device,
            dtype=self.dtype,
            profile=self.profile,
            runtime_binding=self.runtime_binding,
            payment_authority=self.payment_authority,
            capacity_authority=self.capacity_authority,
            text_registry=self.text_registry,
        )

    @staticmethod
    def _selected_reset_row_tensor_name(name: str) -> bool:
        return (
            name.startswith("flight_")
            or name.startswith("mailbox_")
            or name.startswith("replay_")
            or name.startswith("previous_paid_")
            or name
            in (
                "ingress_fault_bits",
                "post_fault_bits",
                "lifecycle_fault_bits",
            )
        )

    def prepare_selected_reset(
        self,
        prepared_true_reset: _r05_device.DeviceR05PreparedTrueReset,
    ) -> PreparedLandingOutcomeSelectedReset:
        """Stage one selected ``[N,K]`` reset with no live write or D2H."""

        self._require_operable()
        if self._latest_selected_reset_completion is not None:
            raise LandingOutcomeDeviceError(
                "top owner has not consumed the previous R06 reset ACK"
            )
        if self._selected_reset_physical_park_token_authority is None:
            raise LandingOutcomeDeviceError(
                "selected reset requires a pre-bound physical park authority"
            )
        self._device_r05_reset_binding()
        validator = self._device_r05_prepared_reset_validator
        if not callable(validator):
            raise LandingOutcomeDeviceError(
                "device R05 prepared true-reset authority is unavailable"
            )
        try:
            claim = validator(prepared_true_reset, owner_kind="r06_flight")
        except Exception as exc:
            raise LandingOutcomeDeviceError(
                "device R05 prepared true-reset authority differs"
            ) from exc
        selected_mask = getattr(claim, "selected_mask", None)
        generation_before = getattr(claim, "generation_before", None)
        generation_after = getattr(claim, "generation_after", None)
        if (
            type(prepared_true_reset)
            is not _r05_device.DeviceR05PreparedTrueReset
            or getattr(claim, "prepared_true_reset", None)
            is not prepared_true_reset
            or getattr(claim, "owner_kind", None) != "r06_flight"
            or not isinstance(selected_mask, torch.Tensor)
            or tuple(selected_mask.shape) != (self.num_envs,)
            or selected_mask.dtype != torch.bool
            or selected_mask.device != self.device
            or not isinstance(generation_before, torch.Tensor)
            or tuple(generation_before.shape) != (self.num_envs,)
            or generation_before.dtype != torch.int64
            or generation_before.device != self.device
            or not isinstance(generation_after, torch.Tensor)
            or tuple(generation_after.shape) != (self.num_envs,)
            or generation_after.dtype != torch.int64
            or generation_after.device != self.device
        ):
            raise LandingOutcomeDeviceError(
                "device R05 selected-reset projection ABI differs"
            )
        selected_mask = selected_mask.detach()
        selected_flight = selected_mask.unsqueeze(1).expand(self._flight_shape)
        selected_mailbox = selected_mask.unsqueeze(1).expand(
            self._mailbox_shape
        )

        live_tensors = self._checkpoint_tensors()
        if self._diagnostic_n2_no_save:
            retained_defaults = self._diagnostic_selected_reset_row_defaults
            resettable_names = {
                name
                for name, tensor in live_tensors.items()
                if self._selected_reset_row_tensor_name(name)
                and tensor.ndim >= 1
                and tensor.shape[0] == self.num_envs
            }
            if (
                type(retained_defaults) is not dict
                or set(retained_defaults) != resettable_names
                or any(
                    not isinstance(retained_defaults[name], torch.Tensor)
                    or retained_defaults[name].shape != live_tensors[name].shape
                    or retained_defaults[name].dtype != live_tensors[name].dtype
                    or retained_defaults[name].device != live_tensors[name].device
                    for name in resettable_names
                )
            ):
                raise LandingOutcomeDeviceError(
                    "diagnostic selected-reset cold row defaults differ"
                )
            # Every after-image tensor owns independent storage.  No mutable
            # owner registry/list/object graph is shared with this transaction.
            shadow_tensors = {
                name: tensor.detach().clone()
                for name, tensor in live_tensors.items()
            }
            default_tensors = retained_defaults
        else:
            shadow = self._new_prepare_shadow()
            defaults = self._fresh_reset_template()
            shadow_tensors = shadow._checkpoint_tensors()
            default_tensors = defaults._checkpoint_tensors()
        for name in sorted(shadow_tensors):
            destination = shadow_tensors[name]
            if (
                self._selected_reset_row_tensor_name(name)
                and destination.ndim >= 1
                and destination.shape[0] == self.num_envs
            ):
                _masked_copy_(destination, default_tensors[name], selected_mask)

        # Selected-reset retirement is deliberately separate from ordinary
        # payment/close history.  These device counters preserve conservation
        # without pretending censored rows received a Reward payment.
        active_flight = selected_flight & (
            self._flight_state != FLIGHT_EMPTY
        )
        active_mailbox = selected_mailbox & (
            self._mailbox_state != MAILBOX_EMPTY
        )
        shadow_tensors["selected_reset_retired_flight_total"].add_(
            active_flight.to(torch.int64).sum()
        )
        shadow_tensors["selected_reset_closed_mailbox_total"].add_(
            active_mailbox.to(torch.int64).sum()
        )
        for index in range(CONSUMER_COUNT):
            paid = (
                torch.bitwise_and(self._mailbox_paid_mask, 1 << index) != 0
            )
            shadow_tensors["selected_reset_retired_payment_totals"][index].add_(
                (active_mailbox & paid).to(torch.int64).sum()
            )

        active_flight_generation = torch.where(
            self._flight_state != FLIGHT_EMPTY,
            self._flight_key_ints["reset_generation"],
            torch.zeros_like(self._flight_key_ints["reset_generation"]),
        ).amax(dim=1)
        mailbox_generation = torch.where(
            self._mailbox_history_valid,
            self._mailbox_key_ints["reset_generation"],
            torch.zeros_like(self._mailbox_key_ints["reset_generation"]),
        ).amax(dim=1)
        replay_generation = torch.where(
            self._replay_valid,
            self._replay_reset_generation,
            torch.zeros_like(self._replay_reset_generation),
        )
        observed_generation = torch.maximum(
            torch.maximum(active_flight_generation, mailbox_generation),
            replay_generation,
        )
        observed_highwater = torch.maximum(
            self._reset_generation_highwater,
            observed_generation,
        )
        generation_regression = selected_mask & (
            observed_highwater > generation_before
        )
        shadow_tensors["reset_generation_highwater"].copy_(
            torch.where(
                selected_mask,
                torch.maximum(observed_highwater, generation_after),
                self._reset_generation_highwater,
            )
        )
        shadow_tensors["lifecycle_fault_bits"].bitwise_or_(
            generation_regression.to(torch.int64)
            * FAULT_GENERATION_BINDING
        )
        shadow_tensors["device_sticky_poison"].bitwise_or_(
            generation_regression
        )
        regression_bits = (
            generation_regression.to(torch.int64)
            * FAULT_GENERATION_BINDING
        ).reshape(-1)
        shadow_fault_counts = shadow_tensors["fault_event_counts"]
        for index, (_, bit) in enumerate(FAULTS):
            shadow_fault_counts[index].add_(
                (torch.bitwise_and(regression_bits, bit) != 0)
                .to(torch.int64)
                .sum()
            )
        generation_step_fault = selected_mask & (
            (generation_before == torch.iinfo(torch.int64).max)
            | (
                generation_after
                != torch.where(
                    generation_before == torch.iinfo(torch.int64).max,
                    generation_before,
                    generation_before + 1,
                )
            )
        )
        shadow_tensors["lifecycle_fault_bits"].bitwise_or_(
            generation_step_fault.to(torch.int64)
            * FAULT_GENERATION_BINDING
        )
        shadow_tensors["device_sticky_poison"].bitwise_or_(
            generation_step_fault
        )
        step_bits = (
            generation_step_fault.to(torch.int64)
            * FAULT_GENERATION_BINDING
        ).reshape(-1)
        for index, (_, bit) in enumerate(FAULTS):
            shadow_fault_counts[index].add_(
                (torch.bitwise_and(step_bits, bit) != 0)
                .to(torch.int64)
                .sum()
            )
        shadow_tensors["selected_reset_count"].copy_(
            self._selected_reset_count + selected_mask.to(torch.int64)
        )
        shadow_tensors["mutation_version"].copy_(self._mutation_version + 1)

        tensor_swaps = tuple(
            (live_tensors[name], shadow_tensors[name])
            for name in sorted(live_tensors)
        )
        prepared = object.__new__(PreparedLandingOutcomeSelectedReset)
        capability = object.__new__(
            LandingOutcomeSelectedResetMaskCapability
        )
        self._active_selected_reset_lease = (
            _ActiveLandingOutcomeSelectedResetLease(
                prepared_reset=prepared,
                mask_capability=capability,
                device_r05_owner=self._device_r05_reset_binding(),
                device_r05_prepared_true_reset=prepared_true_reset,
                device_r05_prepared_projection=claim,
                selected_env_mask=selected_mask.detach().clone(),
                generation_before=generation_before.detach().clone(),
                generation_after=generation_after.detach().clone(),
                tensor_swaps=tensor_swaps,
                host_after_state={
                    "_latest_receipt": None,
                    "_latest_global_drain_receipt": None,
                    "_latest_receipt_consumed": False,
                    "_load_fresh": False,
                },
            )
        )
        return prepared

    def require_owned_selected_reset_prepare(
        self,
        prepared_reset: object,
        *,
        expected_device_r05_prepared: object | None = None,
        expected_device_r05_owner: object | None = None,
    ) -> PreparedLandingOutcomeSelectedReset:
        """Validate the exact currently leased selected-reset prepare."""

        self._require_operable(allow_active_selected_reset=True)
        bound_device_r05 = self._device_r05_reset_binding()
        lease = self._active_selected_reset_lease
        if (
            lease is None
            or type(prepared_reset) is not PreparedLandingOutcomeSelectedReset
            or prepared_reset is not lease.prepared_reset
            or lease.device_r05_owner is not bound_device_r05
            or type(lease.device_r05_prepared_true_reset)
            is not _r05_device.DeviceR05PreparedTrueReset
            or type(lease.device_r05_prepared_projection)
            is not _r05_device.DeviceR05PreparedTrueResetProjection
            or lease.device_r05_prepared_projection.prepared_true_reset
            is not lease.device_r05_prepared_true_reset
            or (
                expected_device_r05_prepared is not None
                and lease.device_r05_prepared_true_reset
                is not expected_device_r05_prepared
            )
            or (
                expected_device_r05_owner is not None
                and bound_device_r05 is not expected_device_r05_owner
            )
            or lease.selected_env_mask.device != self.device
            or lease.generation_before.device != self.device
            or lease.generation_after.device != self.device
            or type(lease.mask_capability)
            is not LandingOutcomeSelectedResetMaskCapability
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset prepare is stale, forged, or foreign"
            )
        return prepared_reset

    def selected_reset_mask_capability(
        self,
        prepared_reset: object,
    ) -> LandingOutcomeSelectedResetMaskCapability:
        """Return the sole retained selection authority for physical park."""

        self.require_owned_selected_reset_prepare(prepared_reset)
        lease = self._active_selected_reset_lease
        capability = None if lease is None else lease.mask_capability
        if (
            type(capability) is not LandingOutcomeSelectedResetMaskCapability
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset mask capability differs"
            )
        return capability

    def require_owned_selected_reset_mask_capability(
        self,
        capability: object,
        *,
        expected_prepared_reset: object,
    ) -> LandingOutcomeSelectedResetMaskView:
        """Validate one opaque mask identity and return only fresh clones."""

        prepared = self.require_owned_selected_reset_prepare(
            expected_prepared_reset
        )
        lease = self._active_selected_reset_lease
        if (
            lease is None
            or type(capability)
            is not LandingOutcomeSelectedResetMaskCapability
            or capability is not lease.mask_capability
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset mask capability is stale or foreign"
            )
        return LandingOutcomeSelectedResetMaskView(
            prepared_reset=prepared,
            mask_capability=capability,
            device_r05_owner=lease.device_r05_owner,
            device_r05_prepared_true_reset=(
                lease.device_r05_prepared_true_reset
            ),
            device_mask=lease.selected_env_mask.detach().clone(),
            generation_before=lease.generation_before.detach().clone(),
            generation_after=lease.generation_after.detach().clone(),
        )

    def arm_prevalidated_selected_reset(
        self,
        prepared_reset: PreparedLandingOutcomeSelectedReset,
        physical_prepared_token: object,
    ) -> ArmedLandingOutcomeSelectedReset:
        """Cross-bind one exact finalized physical after-image before commit."""

        prepared = self.require_owned_selected_reset_prepare(prepared_reset)
        lease = self._active_selected_reset_lease
        authority = self._selected_reset_physical_park_token_authority
        if lease is None or authority is None or lease.armed_reset is not None:
            raise LandingOutcomeDeviceError(
                "selected-reset prepare cannot be armed"
            )
        claim = authority.require_owned_prepared_token(
            physical_prepared_token,
            expected_r06_prepared_reset=prepared,
        )
        if claim.r06_mask_capability is not lease.mask_capability:
            raise LandingOutcomeDeviceError(
                "selected-reset physical mask authority changed"
            )
        armed = object.__new__(ArmedLandingOutcomeSelectedReset)
        lease.armed_reset = armed
        lease.physical_prepared_token = physical_prepared_token
        return armed

    def require_owned_selected_reset_arm(
        self,
        armed_reset: object,
        physical_prepared_token: object,
    ) -> ArmedLandingOutcomeSelectedReset:
        """Physical prearm callback over the exact retained R06 arm."""

        self._require_operable(allow_active_selected_reset=True)
        lease = self._active_selected_reset_lease
        if (
            lease is None
            or type(armed_reset) is not ArmedLandingOutcomeSelectedReset
            or armed_reset is not lease.armed_reset
            or physical_prepared_token is not lease.physical_prepared_token
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset arm is stale, forged, or foreign"
            )
        return armed_reset

    def commit_prevalidated_selected_reset(
        self,
        armed_reset: ArmedLandingOutcomeSelectedReset,
        physical_commit_token: object,
    ) -> LandingOutcomeSelectedResetCommitToken:
        """Retire R06 selected rows only after the exact physical park commit."""

        lease = self._active_selected_reset_lease
        authority = self._selected_reset_physical_park_token_authority
        if (
            self._poisoned
            or lease is None
            or type(armed_reset) is not ArmedLandingOutcomeSelectedReset
            or armed_reset is not lease.armed_reset
            or authority is None
            or lease.commit_token is not None
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset arm is not the active opaque handle"
            )
        try:
            claim = authority.require_committed_park_token(
                physical_commit_token,
                expected_r06_armed_reset=armed_reset,
            )
            if (
                claim.physical_prepared_token
                is not lease.physical_prepared_token
            ):
                raise LandingOutcomeDeviceError(
                    "selected-reset physical prepare identity changed"
                )
            for destination, after_image in lease.tensor_swaps:
                destination.copy_(after_image)
            for name, value in lease.host_after_state.items():
                self.__dict__[name] = value
            commit_token = object.__new__(
                LandingOutcomeSelectedResetCommitToken
            )
            lease.physical_commit_token = physical_commit_token
            lease.commit_token = commit_token
            return commit_token
        except Exception:
            # Physical park may already be visible.  Never roll either leaf
            # back or clear the debt after this ordering boundary.
            self._poisoned = True
            raise

    def require_owned_selected_reset_commit(
        self,
        commit_token: object,
        *,
        expected_prepared_true_reset: object,
    ) -> LandingOutcomeSelectedResetCommitToken:
        """Validate R06's exact commit against the top-retained R05 prepare."""

        lease = self._active_selected_reset_lease
        latest = self._latest_selected_reset_completion
        active_match = (
            lease is not None
            and commit_token is lease.commit_token
        )
        completed_match = (
            latest is not None
            and commit_token is latest[0]
        )
        retained_prepared = (
            lease.device_r05_prepared_true_reset
            if active_match and lease is not None
            else latest[3]
            if completed_match and latest is not None
            else None
        )
        if (
            self._poisoned
            or type(commit_token) is not LandingOutcomeSelectedResetCommitToken
            or not (active_match or completed_match)
            or retained_prepared is not expected_prepared_true_reset
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset commit is stale, forged, or foreign"
            )
        return commit_token

    def require_owned_selected_reset_physical_commit(
        self,
        commit_token: object,
        *,
        expected_prepared_true_reset: object,
        expected_device_r05_owner: object,
    ) -> LandingOutcomeSelectedResetCommitToken:
        """Cross-bind the physical ACK to R06's exact R05 owner and prepare."""

        owned = self.require_owned_selected_reset_commit(
            commit_token,
            expected_prepared_true_reset=expected_prepared_true_reset,
        )
        if self._device_r05_reset_binding() is not expected_device_r05_owner:
            raise LandingOutcomeDeviceError(
                "selected-reset physical commit R05 owner differs"
            )
        lease = self._active_selected_reset_lease
        latest = self._latest_selected_reset_completion
        retained_physical_commit = (
            lease.physical_commit_token
            if lease is not None and owned is lease.commit_token
            else latest[1]
            if latest is not None and owned is latest[0]
            else None
        )
        if (
            retained_physical_commit is None
            or (
                lease is not None
                and owned is lease.commit_token
                and lease.physical_prepared_token is None
            )
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset physical commit cross-binding differs"
            )
        return owned

    def complete_selected_reset_after_r05(
        self,
        commit_token: LandingOutcomeSelectedResetCommitToken,
        r05_true_reset_receipt: _r05_device.DeviceR05TrueResetReceipt,
    ) -> LandingOutcomeSelectedResetCompletionToken:
        """Clear the lease after exact device-R05 ACK; emit no audit bytes."""

        lease = self._active_selected_reset_lease
        if (
            lease is None
            or commit_token is not lease.commit_token
            or lease.physical_commit_token is None
        ):
            raise LandingOutcomeDeviceError(
                "selected-reset R05 completion lacks the exact R06 commit"
            )
        try:
            self.require_owned_selected_reset_commit(
                commit_token,
                expected_prepared_true_reset=(
                    lease.device_r05_prepared_true_reset
                ),
            )
            self._device_r05_reset_binding()
            validator = self._device_r05_receipt_validator
            if not callable(validator):
                raise LandingOutcomeDeviceError(
                    "R05 exact true-reset receipt authority is unavailable"
                )
            owned = validator(
                r05_true_reset_receipt,
                expected_prepared_true_reset=(
                    lease.device_r05_prepared_true_reset
                ),
            )
            if (
                owned is not r05_true_reset_receipt
                or type(owned) is not _r05_device.DeviceR05TrueResetReceipt
            ):
                raise LandingOutcomeDeviceError(
                    "device R05 true-reset exact acknowledgement differs"
                )
            completion = object.__new__(
                LandingOutcomeSelectedResetCompletionToken
            )
            lease.completion_token = completion
            lease.device_r05_true_reset_receipt = owned
            self._latest_selected_reset_completion = (
                commit_token,
                lease.physical_commit_token,
                completion,
                lease.device_r05_prepared_true_reset,
                owned,
            )
            self._active_selected_reset_lease = None
            return completion
        except Exception:
            # R06 bytes have already retired and R05 may already have reset.
            # Preserve the debt and fail-stop rather than fabricating rollback.
            self._poisoned = True
            raise

    def require_owned_selected_reset_completion(
        self,
        completion: object,
    ) -> LandingOutcomeSelectedResetCompletionToken:
        """Validate the latest exact opaque selected-reset ACK for the top."""

        latest = self._latest_selected_reset_completion
        if (
            latest is None
            or type(completion) is not LandingOutcomeSelectedResetCompletionToken
            or completion is not latest[2]
        ):
            raise LandingOutcomeDeviceError(
                "R06 selected-reset completion is stale or foreign"
            )
        return completion

    def consume_owned_selected_reset_completion(
        self,
        completion: object,
    ) -> LandingOutcomeSelectedResetCompletionToken:
        """Let the top owner consume the opaque R06 ACK exactly once."""

        owned = self.require_owned_selected_reset_completion(completion)
        self._latest_selected_reset_completion = None
        return owned

    def abort_selected_reset(
        self,
        prepared_reset: PreparedLandingOutcomeSelectedReset,
    ) -> None:
        """Discard only an unarmed private selected-reset after-image."""

        prepared = self.require_owned_selected_reset_prepare(prepared_reset)
        lease = self._active_selected_reset_lease
        if lease is None or lease.armed_reset is not None:
            raise LandingOutcomeDeviceError(
                "selected-reset abort is stale or crossed prearm"
            )
        if prepared is not lease.prepared_reset:
            raise LandingOutcomeDeviceError(
                "selected-reset abort token differs"
            )
        self._active_selected_reset_lease = None

    def poison_selected_reset(self, reason: str) -> None:
        """Idempotent top-level failure broadcast after selected-reset prearm."""

        if self._poisoned:
            return
        self._poisoned = True
        if type(reason) is str and reason.strip():
            self._global_reveal_poison_reason = reason

    def prepare_from_reveal_final_preview(
        self,
        reveal_final_preview: _r05.RevealFinalPreviewBatch,
        *,
        expected_reveal_final_preview_sha256: str,
    ) -> LandingRevealPrepareAttempt:
        """Stage a private R06 image with owner-selected mailbox slots and 0 D2H."""

        self._require_operable()
        self._require_formal_only(
            "prepare reveal rows without the missing formal H/C authority"
        )
        r05_owner = self._r05_terminal_binding()
        _boundary_owner, boundary_lane = self._reveal_boundary_binding()
        if type(reveal_final_preview) is not _r05.RevealFinalPreviewBatch:
            raise LandingOutcomeDeviceError(
                "production ingress requires the exact R05 "
                "RevealFinalPreviewBatch token"
            )
        expected_preview = _sha256_hex(
            expected_reveal_final_preview_sha256,
            label="expected_reveal_final_preview_sha256",
        )
        try:
            preview = r05_owner.require_owned_active_reveal_final_preview(
                reveal_final_preview,
                expected_reveal_final_preview_sha256=expected_preview,
            )
        except _r05.TransactionConflictError as exc:
            raise LandingOutcomeDeviceError(str(exc)) from exc
        except Exception as exc:
            raise LandingOutcomeDeviceError(
                "R06 requires the exact active unstaged R05 preview lease"
            ) from exc
        if (
            preview is reveal_final_preview
            or preview.canonical_sha256 != expected_preview
        ):
            raise LandingOutcomeDeviceError(
                "R06 retained R05 preview image differs"
            )
        selected_env_ids = tuple(preview.selected_env_ids)
        if (
            not selected_env_ids
            or selected_env_ids != tuple(sorted(set(selected_env_ids)))
            or selected_env_ids[-1] >= self.num_envs
        ):
            raise LandingOutcomeDeviceError(
                "R05 preview selected env ids are out of range/order"
            )
        for row in preview.reveal_final_rows:
            if (
                row.ball_slot_plan.capacity != self.flight_slot_capacity
                or len(row.pre_install_ball_slots) != self.flight_slot_capacity
                or len(row.post_install_ball_slots) != self.flight_slot_capacity
            ):
                raise LandingOutcomeDeviceError(
                    "R05 preview physical capacity differs from R03/R06 K"
                )
        horizons = tuple(
            row.reveal_facts.reveal_step
            + int(self.capacity_authority.flight_horizon_ticks)
            for row in preview.reveal_final_rows
        )
        committed_rows = _synthesize_r05_committed_rows_from_preview(preview)
        expected_committed_rows = tuple(
            row.canonical_sha256 for row in committed_rows
        )
        # The legacy private builder needs a host-shaped template, but the
        # sentinel is deliberately out of range and is never installed.  The
        # actual lowest available mailbox slot is selected below on device.
        mailbox_allocation_sentinel = self.mailbox_capacity
        mailbox_slot_template = tuple(
            mailbox_allocation_sentinel for _ in selected_env_ids
        )
        legacy_install = _build_landing_reveal_install_batch_from_preview(
            preview,
            expected_reveal_final_preview_sha256=preview.canonical_sha256,
            expected_committed_reveal_sha256=expected_committed_rows,
            expected_r05_source_sha256=self.runtime_binding.r05_source_sha256,
            expected_r05_contract_sha256=self.runtime_binding.r05_contract_sha256,
            expected_c05_source_sha256=self.runtime_binding.c05_source_sha256,
            profile=self.profile,
            expected_profile_sha256=self.runtime_binding.landing_profile_sha256,
            text_registry=self.text_registry,
            expected_text_registry_sha256=(
                self.runtime_binding.text_registry_sha256
            ),
            mailbox_slots=mailbox_slot_template,
            first_crossing_horizon_control_steps=horizons,
            num_envs=self.num_envs,
            device=self.device,
            dtype=self.dtype,
        )
        legacy_receipt_payload, legacy_pack = _owned_landing_reveal_install(
            legacy_install
        )
        selected_mask = legacy_pack["mask"]
        if not isinstance(selected_mask, torch.Tensor):
            raise LandingOutcomeDeviceError("R06 reveal selected mask differs")
        mailbox_available = (
            (self._mailbox_state == MAILBOX_EMPTY)
            & ~self._mailbox_reserved
            & ~self._mailbox_history_valid
            & ~self._mailbox_physical_retired
        )
        mailbox_candidates = torch.where(
            mailbox_available,
            self._mailbox_slot_ids.expand(self._mailbox_shape),
            torch.full(
                self._mailbox_shape,
                mailbox_allocation_sentinel,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        allocated_mailbox_slot = torch.where(
            selected_mask,
            mailbox_candidates.amin(dim=1),
            torch.full(
                (self.num_envs,),
                -1,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        legacy_rows = legacy_receipt_payload["rows"]
        preview_rows: list[dict[str, object]] = []
        for index, (preview_row, committed_row, legacy_row) in enumerate(
            zip(
                preview.reveal_final_rows,
                committed_rows,
                legacy_rows,
            )
        ):
            preview_rows.append(
                {
                    "env_id": preview_row.env_id,
                    "reveal_final_install_row_sha256": preview_row.canonical_sha256,
                    "expected_committed_reveal_sha256": (
                        committed_row.canonical_sha256
                    ),
                    "physical_ball_install_payload_sha256": (
                        preview_row.physical_ball_install_payload_sha256
                    ),
                    "pre_install_ball_slots_sha256": _r05_snapshot_sha256(
                        preview_row.pre_install_ball_slots
                    ),
                    "post_install_ball_slots_sha256": _r05_snapshot_sha256(
                        preview_row.post_install_ball_slots
                    ),
                    "outcome_key": (
                        preview_row.prepared_reveal.outcome_key.to_mapping()
                    ),
                    "full_key_sha256": (
                        preview_row.prepared_reveal.outcome_key.canonical_sha256
                    ),
                    "flight_slot": preview_row.ball_slot_plan.selected_slot_index,
                    "mailbox_allocation_policy": MAILBOX_ALLOCATION_POLICY,
                    "ball_generation": preview_row.ball_slot_plan.new_ball_generation,
                    "reveal_control_step": preview_row.reveal_facts.reveal_step,
                    "selected_contact_deadline_control_step": (
                        preview_row.reveal_facts.deadline_step
                    ),
                    "first_crossing_horizon_control_step": horizons[index],
                    "legacy_install_row_receipt_sha256": legacy_row[
                        "row_receipt_sha256"
                    ],
                }
            )
        after_image_authority_payload = {
            "kind": PREPARE_ATTEMPT_KIND,
            "reveal_final_preview_sha256": preview.canonical_sha256,
            "r05_owner_checkpoint_before_sha256": (
                preview.owner_checkpoint_before_sha256
            ),
            "legacy_install_receipt_sha256": legacy_install.receipt.canonical_sha256,
            "capacity_authority_sha256": self.capacity_authority.canonical_sha256,
            "payment_authority_sha256": self.payment_authority.canonical_sha256,
            "profile_sha256": self.profile.canonical_sha256,
            "text_registry_sha256": self.text_registry.canonical_sha256,
            "selected_env_ids": list(selected_env_ids),
            "mailbox_allocation_policy": MAILBOX_ALLOCATION_POLICY,
            "mailbox_allocation_sentinel": mailbox_allocation_sentinel,
            "first_crossing_horizon_control_steps": list(horizons),
            "preview_rows": preview_rows,
        }
        receipt_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": PREPARED_INSTALL_RECEIPT_KIND,
            "num_envs": self.num_envs,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "r05_source_sha256": self.runtime_binding.r05_source_sha256,
            "r05_contract_sha256": self.runtime_binding.r05_contract_sha256,
            "r05_final_source_pin_pending": R05_FINAL_SOURCE_PIN_PENDING,
            "c05_source_sha256": self.runtime_binding.c05_source_sha256,
            "profile_sha256": self.profile.canonical_sha256,
            "text_registry_sha256": self.text_registry.canonical_sha256,
            "capacity_authority": self.capacity_authority.to_mapping(),
            "capacity_authority_sha256": self.capacity_authority.canonical_sha256,
            "payment_authority_sha256": self.payment_authority.canonical_sha256,
            "reveal_final_preview_sha256": preview.canonical_sha256,
            "expected_reveal_final_preview_sha256": (
                expected_reveal_final_preview_sha256
            ),
            "r05_owner_checkpoint_before_sha256": (
                preview.owner_checkpoint_before_sha256
            ),
            "prepared_batch_sha256": preview.prepared_batch_sha256,
            "sampler_checkpoint_before_commit_sha256": (
                preview.sampler_checkpoint_before_commit_sha256
            ),
            "sampler_checkpoint_after_commit_sha256": (
                preview.sampler_checkpoint_after_commit_sha256
            ),
            "untouched_rows_before_sha256": preview.untouched_rows_before_sha256,
            "untouched_rows_after_sha256": preview.untouched_rows_after_sha256,
            "all_owner_install_root_sha256": (
                preview.all_owner_install_root_sha256
            ),
            "selected_env_ids": list(selected_env_ids),
            "mailbox_allocation_policy": MAILBOX_ALLOCATION_POLICY,
            "mailbox_allocation_sentinel": mailbox_allocation_sentinel,
            "first_crossing_horizon_control_steps": list(horizons),
            "preview_rows": preview_rows,
            "legacy_after_image_install_template_receipt_sha256": (
                legacy_install.receipt.canonical_sha256
            ),
            "after_image_authority_sha256": _canonical_sha256(
                after_image_authority_payload
            ),
        }
        receipt_json = json.dumps(
            receipt_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        receipt = LandingRevealPreparedInstallReceipt(
            _payload_json=receipt_json,
            _auth_tag=hmac.new(
                _PREPARED_INSTALL_AUTH_KEY,
                receipt_json,
                hashlib.sha256,
            ).digest(),
            _token=_PREPARED_INSTALL_AUTH_TOKEN,
        )
        _owned_prepared_install_receipt(receipt)

        shadow = self._new_prepare_shadow()
        precomputed_result = shadow._install_reveal_legacy_impl(
            legacy_install,
            allocated_mailbox_slot=allocated_mailbox_slot,
        )
        version_overflow = self._mutation_version.eq((1 << 63) - 1)
        version_fault = selected_mask & version_overflow
        boundary_fault_bits = torch.bitwise_or(
            precomputed_result.fault_bits,
            version_fault.to(torch.int64) * FAULT_BATCH_ABORT,
        )
        boundary_accepted = precomputed_result.accepted & ~version_fault
        boundary_rejected = selected_mask & ~boundary_accepted
        boundary_result = DeviceMutationResult(
            accepted=boundary_accepted.detach().clone(),
            rejected=boundary_rejected.detach().clone(),
            fault_bits=boundary_fault_bits.detach().clone(),
        )
        live_tensors = self._checkpoint_tensors()
        shadow_tensors = shadow._checkpoint_tensors()
        tensor_swaps = tuple(
            (live_tensors[name], shadow_tensors[name])
            for name in sorted(live_tensors)
        )
        attempt = LandingRevealPrepareAttempt(
            _receipt=receipt,
            _preview_token=preview,
            _legacy_install=legacy_install,
            _allocated_mailbox_slot=allocated_mailbox_slot.detach().clone(),
            _tensor_swaps=tensor_swaps,
            _host_after_state={
                "_latest_receipt": None,
                "_latest_global_drain_receipt": None,
                "_latest_receipt_consumed": False,
                "_load_fresh": False,
            },
            _precomputed_result=boundary_result,
            _device_owner_mutation_version=self._mutation_version,
            _owner_identity=self._owner_identity,
            _token=_PREPARED_INSTALL_AUTH_TOKEN,
        )
        _owned_prepare_attempt(attempt)
        censor_version_after = self._mutation_version.detach().clone()
        censor_version_after.add_(1)
        lease = _ActiveRevealPrepareLease(
            attempt=attempt,
            boundary_row=None,  # type: ignore[arg-type]
            censor_tensor_swaps=((self._mutation_version, censor_version_after),),
            censor_host_after_state={
                "_latest_receipt": None,
                "_latest_global_drain_receipt": None,
                "_latest_receipt_consumed": False,
                "_load_fresh": False,
            },
        )
        self._active_reveal_prepare_lease = lease
        try:
            boundary_row = boundary_lane.mint_device_row(
                prepared_token=attempt,
                selected_env_ids=selected_env_ids,
                pass_mask=boundary_result.accepted,
                fault_bits=boundary_result.fault_bits,
            )
        except Exception:
            self._active_reveal_prepare_lease = None
            raise
        lease.boundary_row = boundary_row
        return attempt

    def reveal_prepare_boundary_row(
        self,
        attempt: LandingRevealPrepareAttempt,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryDeviceRow:
        """Return the exact owner-retained row; never rebuild or transfer it."""

        self._require_operable(allow_active_prepare=True)
        _boundary_owner, boundary_lane = self._reveal_boundary_binding()
        lease = self._active_reveal_prepare_lease
        if (
            lease is None
            or attempt is not lease.attempt
            or type(attempt) is not LandingRevealPrepareAttempt
            or attempt._owner_identity is not self._owner_identity
            or lease.armed_install is not None
            or lease.censored_install is not None
        ):
            raise LandingOutcomeDeviceError(
                "R06 reveal prepare attempt is stale, foreign, or consumed"
            )
        row = lease.boundary_row
        if (
            type(row)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryDeviceRow
            or row.owner_kind != "r06_flight"
            or row.owner_token_root_sha256
            != attempt._receipt.canonical_sha256
            or row.reveal_final_preview_schema_version
            != _r05.RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
            or row.reveal_final_preview_sha256
            != attempt._receipt.reveal_final_preview_sha256
            or row.selected_env_ids != attempt._receipt.selected_env_ids
            or row.child_token_authority
            is not self._reveal_boundary_child_token_authority
        ):
            raise LandingOutcomeDeviceError(
                "R06 retained reveal-boundary row differs"
            )
        boundary_lane.require_owned_device_row(
            row, expected_prepared_token=attempt
        )
        return row

    def _poison_reveal_boundary_leaf(
        self,
        message: str,
        exc: BaseException | None = None,
    ) -> None:
        self._poisoned = True
        if self._global_reveal_poison_reason is None:
            self._global_reveal_poison_reason = message
        if exc is None:
            raise LandingOutcomeDeviceError(message)
        raise LandingOutcomeDeviceError(message) from exc

    def _require_global_reveal_decision(
        self,
        attempt: LandingRevealPrepareAttempt,
        global_boundary_receipt: object,
        *,
        expected_decision: str,
    ) -> tuple[
        _reveal_boundary.ActionBallFullMdpRevealBoundaryOwnerRow,
        str,
        str,
    ]:
        boundary_owner, boundary_lane = self._reveal_boundary_binding()
        lease = self._active_reveal_prepare_lease
        if (
            lease is None
            or type(attempt) is not LandingRevealPrepareAttempt
            or attempt is not lease.attempt
            or attempt._owner_identity is not self._owner_identity
            or lease.armed_install is not None
            or lease.censored_install is not None
            or type(global_boundary_receipt)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        ):
            self._poison_reveal_boundary_leaf(
                "R06 global reveal receipt is omitted, malformed, or duplicated"
            )
        assert lease is not None
        assert type(global_boundary_receipt) is (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        )
        if global_boundary_receipt.decision != expected_decision:
            raise LandingOutcomeDeviceError(
                "R06 global reveal receipt used the wrong typed arm"
            )
        try:
            owned_row = boundary_owner.require_owned_owner_row(
                global_boundary_receipt,
                owner_kind="r06_flight",
                expected_device_row=lease.boundary_row,
                expected_prepared_token=attempt,
                expected_fault_schema_sha256=(
                    R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256
                ),
                expected_reveal_final_preview_schema_version=(
                    _r05.RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
                ),
                expected_reveal_final_preview_sha256=(
                    attempt._receipt.reveal_final_preview_sha256
                ),
                expected_selected_env_ids=(
                    attempt._receipt.selected_env_ids
                ),
                expected_packet_sha256=(
                    global_boundary_receipt.packet_sha256
                ),
                expected_decision=expected_decision,
            )
        except Exception as exc:
            self._poison_reveal_boundary_leaf(
                "R06 global reveal owner row is malformed or stale", exc
            )
        assert isinstance(
            owned_row,
            _reveal_boundary.ActionBallFullMdpRevealBoundaryOwnerRow,
        )
        if (
            owned_row.owner_kind != "r06_flight"
            or owned_row.owner_token_root_sha256
            != attempt._receipt.canonical_sha256
            or owned_row.fault_schema_sha256
            != R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256
            or owned_row.allowed_fault_mask
            != R06_REVEAL_BOUNDARY_FAULT_SCHEMA.allowed_fault_mask
            or len(owned_row.selected_pass)
            != len(attempt._receipt.selected_env_ids)
            or len(owned_row.selected_fault_bits)
            != len(attempt._receipt.selected_env_ids)
            or owned_row.owner_mutation_version < 0
            or owned_row.owner_mutation_version >= (1 << 63) - 1
            or (
                expected_decision
                == _reveal_boundary.DECISION_ACCEPT
                and (
                    not all(owned_row.selected_pass)
                    or any(owned_row.selected_fault_bits)
                )
            )
        ):
            self._poison_reveal_boundary_leaf(
                "R06 decoded global reveal owner row differs"
            )
        return (
            owned_row,
            global_boundary_receipt.canonical_sha256,
            global_boundary_receipt.packet_sha256,
        )

    def _require_r05_terminal_claim(
        self,
        attempt: LandingRevealPrepareAttempt,
        global_boundary_receipt: object,
        prepared_r05_terminal_claim: object,
        *,
        expected_boundary_decision: str,
    ) -> tuple[
        object,
        str,
        str,
        str,
        str,
        str,
        str,
        object,
        str,
        str,
    ]:
        """Bind one exact R05-owned future terminal before child commit."""

        owner = self._r05_terminal_binding()
        terminal_boundary_authority = self._r05_terminal_boundary_authority
        if type(terminal_boundary_authority) is not _r05.TerminalBoundaryAuthority:
            self._poison_reveal_boundary_leaf(
                "R06 R05 terminal-boundary authority is unavailable"
            )
        if expected_boundary_decision == _reveal_boundary.DECISION_ACCEPT:
            expected_decision = _r05.TERMINAL_DECISION_ACCEPT
            expected_source_decision = (
                _r05.TERMINAL_BOUNDARY_SOURCE_DECISION_PASS
            )
            expected_terminal_type = _r05.CommittedRevealBatch
        elif expected_boundary_decision == _reveal_boundary.DECISION_CENSOR:
            expected_decision = _r05.TERMINAL_DECISION_CENSOR
            expected_source_decision = _r05.TERMINAL_DECISION_CENSOR
            expected_terminal_type = _r05.CensoredRevealBatch
        else:
            self._poison_reveal_boundary_leaf(
                "R06 R05 terminal decision differs"
            )
        expected_terminal_kind = expected_terminal_type.KIND
        if (
            type(prepared_r05_terminal_claim)
            is not _r05.PreparedRevealTerminalClaim
            or type(global_boundary_receipt)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        ):
            self._poison_reveal_boundary_leaf(
                "R06 requires an exact owner-issued R05 terminal claim"
            )
        assert type(global_boundary_receipt) is (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        )
        try:
            claim_sha256 = _sha256_hex(
                prepared_r05_terminal_claim.canonical_sha256,
                label="R05 prepared terminal claim root",
            )
            terminal_sha256 = _sha256_hex(
                prepared_r05_terminal_claim.terminal_sha256,
                label="R05 preclaimed terminal root",
            )
            receipt_sha256 = _sha256_hex(
                global_boundary_receipt.canonical_sha256,
                label="global reveal-boundary receipt root",
            )
            packet_sha256 = _sha256_hex(
                global_boundary_receipt.packet_sha256,
                label="global reveal-boundary packet root",
            )
            authority_sha256 = _sha256_hex(
                terminal_boundary_authority.canonical_sha256,
                label="R05 terminal-boundary authority root",
            )
            projection_sha256 = _sha256_hex(
                prepared_r05_terminal_claim.terminal_boundary_projection_sha256,
                label="R05 terminal-boundary projection root",
            )
            content_pin = prepared_r05_terminal_claim.terminal_content_pin
            content_pin_sha256 = _sha256_hex(
                content_pin.canonical_sha256,
                label="R05 prepared terminal content-pin root",
            )
            content_bytes_sha256 = _sha256_hex(
                content_pin.content_bytes_sha256,
                label="R05 prepared terminal content bytes root",
            )
            owned_claim = owner.require_owned_prepared_terminal_claim(
                prepared_r05_terminal_claim,
                expected_claim_sha256=claim_sha256,
                expected_decision=expected_decision,
                expected_reveal_final_preview_sha256=(
                    attempt._receipt.reveal_final_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=receipt_sha256,
                expected_global_boundary_packet_sha256=packet_sha256,
                expected_terminal_boundary_authority_sha256=(
                    authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    content_pin_sha256
                ),
                expected_terminal_kind=expected_terminal_kind,
                expected_terminal_sha256=terminal_sha256,
                expected_selected_env_ids=attempt._receipt.selected_env_ids,
            )
        except Exception as exc:
            self._poison_reveal_boundary_leaf(
                "R06 R05 terminal claim is stale or foreign", exc
            )
        projection = prepared_r05_terminal_claim.terminal_boundary_projection
        participant_roots = tuple(projection.ordered_participant_roots)
        expected_participant_roots = tuple(
            (row.owner_kind, row.owner_token_root_sha256)
            for row in global_boundary_receipt.ordered_owner_rows
        )
        if (
            owned_claim is not prepared_r05_terminal_claim
            or prepared_r05_terminal_claim.decision != expected_decision
            or prepared_r05_terminal_claim.selected_env_ids
            != attempt._receipt.selected_env_ids
            or prepared_r05_terminal_claim.reveal_final_preview_schema_version
            != _r05.RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
            or prepared_r05_terminal_claim.reveal_final_preview_sha256
            != attempt._receipt.reveal_final_preview_sha256
            or prepared_r05_terminal_claim.global_boundary_receipt_kind
            != _reveal_boundary.RECEIPT_KIND
            or prepared_r05_terminal_claim.global_boundary_receipt_sha256
            != receipt_sha256
            or prepared_r05_terminal_claim.global_boundary_packet_schema_version
            != _reveal_boundary.PACKET_SCHEMA_VERSION
            or prepared_r05_terminal_claim.global_boundary_packet_sha256
            != packet_sha256
            or prepared_r05_terminal_claim.terminal_kind
            != expected_terminal_kind
            or prepared_r05_terminal_claim.terminal_sha256
            != terminal_sha256
            or prepared_r05_terminal_claim.terminal_boundary_authority_sha256
            != authority_sha256
            or type(projection) is not _r05.TerminalBoundaryProjection
            or projection.canonical_sha256 != projection_sha256
            or projection.decision != expected_decision
            or projection.source_decision != expected_source_decision
            or projection.reveal_final_preview_schema_version
            != global_boundary_receipt.reveal_final_preview_schema_version
            or projection.reveal_final_preview_sha256
            != attempt._receipt.reveal_final_preview_sha256
            or projection.selected_env_ids
            != attempt._receipt.selected_env_ids
            or projection.boundary_receipt_kind
            != global_boundary_receipt.kind
            or projection.boundary_receipt_sha256 != receipt_sha256
            or projection.boundary_packet_schema_version
            != global_boundary_receipt.packet_schema_version
            or projection.boundary_packet_sha256 != packet_sha256
            or projection.authority_domain
            != FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
            or projection.authority_schema_sha256
            != FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
            or projection.authority_source_sha256
            != FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256
            or len(participant_roots) != len(expected_participant_roots)
            or any(
                type(row) is not _r05.TerminalBoundaryParticipantRoot
                or row.participant_domain != projection.authority_domain
                or (row.participant_kind, row.participant_root_sha256)
                != expected_participant_roots[index]
                for index, row in enumerate(participant_roots)
            )
            or type(content_pin) is not _r05.PreparedTerminalContentPin
            or content_pin.terminal_kind != expected_terminal_kind
            or content_pin.terminal_canonical_sha256 != terminal_sha256
            or content_pin.content_byte_length <= 0
        ):
            self._poison_reveal_boundary_leaf(
                "R06 R05 terminal claim facts differ"
            )
        return (
            prepared_r05_terminal_claim,
            claim_sha256,
            expected_decision,
            expected_terminal_kind,
            terminal_sha256,
            authority_sha256,
            projection_sha256,
            content_pin,
            content_pin_sha256,
            content_bytes_sha256,
        )

    def arm_global_reveal_boundary(
        self,
        attempt: LandingRevealPrepareAttempt,
        global_boundary_receipt: object,
        prepared_r05_terminal_claim: object,
    ) -> LandingRevealArmedInstall:
        """Validate only R06's exact row in one owner-issued ACCEPT receipt."""

        self._require_operable(allow_active_prepare=True)
        owned_row, receipt_sha256, packet_sha256 = (
            self._require_global_reveal_decision(
                attempt,
                global_boundary_receipt,
                expected_decision=_reveal_boundary.DECISION_ACCEPT,
            )
        )
        lease = self._active_reveal_prepare_lease
        assert lease is not None
        (
            terminal_claim,
            terminal_claim_sha256,
            terminal_decision,
            terminal_kind,
            terminal_sha256,
            terminal_boundary_authority_sha256,
            terminal_boundary_projection_sha256,
            terminal_content_pin,
            terminal_content_pin_sha256,
            terminal_content_bytes_sha256,
        ) = self._require_r05_terminal_claim(
            attempt,
            global_boundary_receipt,
            prepared_r05_terminal_claim,
            expected_boundary_decision=_reveal_boundary.DECISION_ACCEPT,
        )
        commit_receipt = LandingRevealCommitReceipt(
            schema_version=SCHEMA_VERSION,
            kind=PREPARED_COMMIT_RECEIPT_KIND,
            global_boundary_receipt_sha256=receipt_sha256,
            global_boundary_packet_sha256=packet_sha256,
            r05_terminal_claim_sha256=terminal_claim_sha256,
            r05_terminal_boundary_authority_sha256=(
                terminal_boundary_authority_sha256
            ),
            r05_terminal_boundary_projection_sha256=(
                terminal_boundary_projection_sha256
            ),
            r05_terminal_content_pin_sha256=terminal_content_pin_sha256,
            r05_terminal_schema_version=(
                terminal_content_pin.terminal_schema_version
            ),
            r05_terminal_content_bytes_base64=(
                terminal_content_pin.content_bytes_base64
            ),
            r05_terminal_content_byte_length=(
                terminal_content_pin.content_byte_length
            ),
            r05_terminal_content_bytes_sha256=(
                terminal_content_bytes_sha256
            ),
            expected_r05_terminal_kind=terminal_kind,
            expected_r05_terminal_sha256=terminal_sha256,
            r06_child_token_root_sha256=attempt._receipt.canonical_sha256,
            reveal_final_preview_sha256=(
                attempt._receipt.reveal_final_preview_sha256
            ),
            selected_env_ids=attempt._receipt.selected_env_ids,
            owner_mutation_version_before=owned_row.owner_mutation_version,
            owner_mutation_version_after=owned_row.owner_mutation_version + 1,
            installed_count=len(attempt._receipt.selected_env_ids),
            runtime_integrated=RUNTIME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )
        _sha256_hex(
            commit_receipt.canonical_sha256,
            label="R06 reveal commit receipt",
        )
        armed = LandingRevealArmedInstall(
            _attempt=attempt,
            _r05_terminal_claim=terminal_claim,
            _owner_identity=self._owner_identity,
            _token=_PREPARED_INSTALL_AUTH_TOKEN,
        )
        terminal_token = R06ChildTerminalToken(
            _attempt=attempt,
            _child_receipt=commit_receipt,
            _r05_terminal_claim=terminal_claim,
            _decision=_r05.TERMINAL_DECISION_ACCEPT,
            _owner_identity=self._owner_identity,
            _token=_REVEAL_TERMINAL_AUTH_TOKEN,
        )
        lease.commit_receipt = commit_receipt
        lease.armed_install = armed
        lease.r05_terminal_claim = terminal_claim
        lease.r05_terminal_claim_sha256 = terminal_claim_sha256
        lease.r05_terminal_boundary_authority_sha256 = (
            terminal_boundary_authority_sha256
        )
        lease.r05_terminal_boundary_projection_sha256 = (
            terminal_boundary_projection_sha256
        )
        lease.r05_terminal_content_pin = terminal_content_pin
        lease.r05_terminal_content_pin_sha256 = terminal_content_pin_sha256
        lease.r05_terminal_content_bytes_sha256 = (
            terminal_content_bytes_sha256
        )
        lease.r05_terminal_decision = terminal_decision
        lease.r05_terminal_kind = terminal_kind
        lease.r05_terminal_sha256 = terminal_sha256
        lease.global_boundary_receipt_sha256 = receipt_sha256
        lease.global_boundary_packet_sha256 = packet_sha256
        lease.terminal_token = terminal_token
        return armed

    def arm_censored_global_reveal_boundary(
        self,
        attempt: LandingRevealPrepareAttempt,
        global_boundary_receipt: object,
        prepared_r05_terminal_claim: object,
    ) -> LandingRevealCensoredInstall:
        """Validate one exact CENSOR while keeping chronology private."""

        self._require_operable(allow_active_prepare=True)
        owned_row, receipt_sha256, packet_sha256 = (
            self._require_global_reveal_decision(
                attempt,
                global_boundary_receipt,
                expected_decision=_reveal_boundary.DECISION_CENSOR,
            )
        )
        lease = self._active_reveal_prepare_lease
        assert lease is not None
        (
            terminal_claim,
            terminal_claim_sha256,
            terminal_decision,
            terminal_kind,
            terminal_sha256,
            terminal_boundary_authority_sha256,
            terminal_boundary_projection_sha256,
            terminal_content_pin,
            terminal_content_pin_sha256,
            terminal_content_bytes_sha256,
        ) = self._require_r05_terminal_claim(
            attempt,
            global_boundary_receipt,
            prepared_r05_terminal_claim,
            expected_boundary_decision=_reveal_boundary.DECISION_CENSOR,
        )
        censor_receipt = LandingRevealCensorReceipt(
            schema_version=SCHEMA_VERSION,
            kind="action_ball_landing_reveal_censor_receipt_v1",
            global_boundary_receipt_sha256=receipt_sha256,
            global_boundary_packet_sha256=packet_sha256,
            r05_terminal_claim_sha256=terminal_claim_sha256,
            r05_terminal_boundary_authority_sha256=(
                terminal_boundary_authority_sha256
            ),
            r05_terminal_boundary_projection_sha256=(
                terminal_boundary_projection_sha256
            ),
            r05_terminal_content_pin_sha256=terminal_content_pin_sha256,
            r05_terminal_schema_version=(
                terminal_content_pin.terminal_schema_version
            ),
            r05_terminal_content_bytes_base64=(
                terminal_content_pin.content_bytes_base64
            ),
            r05_terminal_content_byte_length=(
                terminal_content_pin.content_byte_length
            ),
            r05_terminal_content_bytes_sha256=(
                terminal_content_bytes_sha256
            ),
            expected_r05_terminal_kind=terminal_kind,
            expected_r05_terminal_sha256=terminal_sha256,
            r06_child_token_root_sha256=attempt._receipt.canonical_sha256,
            reveal_final_preview_sha256=(
                attempt._receipt.reveal_final_preview_sha256
            ),
            selected_env_ids=attempt._receipt.selected_env_ids,
            owner_mutation_version_before=owned_row.owner_mutation_version,
            owner_mutation_version_after=owned_row.owner_mutation_version + 1,
            installed_count=0,
            censored_count=len(attempt._receipt.selected_env_ids),
            policy_opportunity_created=False,
            runtime_integrated=RUNTIME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )
        _sha256_hex(
            censor_receipt.canonical_sha256,
            label="R06 reveal CENSOR receipt",
        )
        censored = LandingRevealCensoredInstall(
            _attempt=attempt,
            _r05_terminal_claim=terminal_claim,
            _owner_identity=self._owner_identity,
            _token=_PREPARED_INSTALL_AUTH_TOKEN,
        )
        terminal_token = R06ChildTerminalToken(
            _attempt=attempt,
            _child_receipt=censor_receipt,
            _r05_terminal_claim=terminal_claim,
            _decision=_r05.TERMINAL_DECISION_CENSOR,
            _owner_identity=self._owner_identity,
            _token=_REVEAL_TERMINAL_AUTH_TOKEN,
        )
        lease.censor_receipt = censor_receipt
        lease.censored_install = censored
        lease.r05_terminal_claim = terminal_claim
        lease.r05_terminal_claim_sha256 = terminal_claim_sha256
        lease.r05_terminal_boundary_authority_sha256 = (
            terminal_boundary_authority_sha256
        )
        lease.r05_terminal_boundary_projection_sha256 = (
            terminal_boundary_projection_sha256
        )
        lease.r05_terminal_content_pin = terminal_content_pin
        lease.r05_terminal_content_pin_sha256 = terminal_content_pin_sha256
        lease.r05_terminal_content_bytes_sha256 = (
            terminal_content_bytes_sha256
        )
        lease.r05_terminal_decision = terminal_decision
        lease.r05_terminal_kind = terminal_kind
        lease.r05_terminal_sha256 = terminal_sha256
        lease.global_boundary_receipt_sha256 = receipt_sha256
        lease.global_boundary_packet_sha256 = packet_sha256
        lease.terminal_token = terminal_token
        return censored

    def commit_prevalidated_reveal_install(
        self,
        armed_install: LandingRevealArmedInstall,
    ) -> R06ChildTerminalToken:
        """Publish the prebuilt ACCEPT image after one opaque identity gate."""

        lease = self._active_reveal_prepare_lease
        if (
            self._poisoned
            or lease is None
            or armed_install is not lease.armed_install
            or type(armed_install) is not LandingRevealArmedInstall
            or armed_install._owner_identity is not self._owner_identity
            or armed_install._token is not _PREPARED_INSTALL_AUTH_TOKEN
            or armed_install._r05_terminal_claim
            is not lease.r05_terminal_claim
            or lease.commit_receipt is None
            or lease.r05_terminal_claim is None
            or lease.r05_terminal_decision
            != _r05.TERMINAL_DECISION_ACCEPT
            or type(lease.terminal_token) is not R06ChildTerminalToken
            or lease.terminal_token._child_receipt is not lease.commit_receipt
            or lease.published_terminal_token is not None
        ):
            self._poisoned = True
            if self._global_reveal_poison_reason is None:
                self._global_reveal_poison_reason = (
                    "R06 ACCEPT child commit handle was stale or duplicated"
                )
            raise LandingOutcomeDeviceError(
                "R06 armed reveal handle is not active and prevalidated"
            )
        try:
            for destination, after_image in lease.attempt._tensor_swaps:
                destination.copy_(after_image)
            for name, value in lease.attempt._host_after_state.items():
                self.__dict__[name] = value
            lease.published_terminal_token = lease.terminal_token
            return lease.terminal_token
        except Exception:
            self.poison_global_reveal_epoch(
                "R06 ACCEPT child after-image publication failed"
            )
            raise

    def commit_censored_prevalidated(
        self,
        censored_install: LandingRevealCensoredInstall,
    ) -> R06ChildTerminalToken:
        """Publish only the prebuilt zero-business CENSOR chronology image."""

        lease = self._active_reveal_prepare_lease
        if (
            self._poisoned
            or lease is None
            or censored_install is not lease.censored_install
            or type(censored_install) is not LandingRevealCensoredInstall
            or censored_install._owner_identity is not self._owner_identity
            or censored_install._token is not _PREPARED_INSTALL_AUTH_TOKEN
            or censored_install._r05_terminal_claim
            is not lease.r05_terminal_claim
            or lease.censor_receipt is None
            or lease.r05_terminal_claim is None
            or lease.r05_terminal_decision
            != _r05.TERMINAL_DECISION_CENSOR
            or type(lease.terminal_token) is not R06ChildTerminalToken
            or lease.terminal_token._child_receipt is not lease.censor_receipt
            or lease.published_terminal_token is not None
        ):
            self._poisoned = True
            if self._global_reveal_poison_reason is None:
                self._global_reveal_poison_reason = (
                    "R06 CENSOR child commit handle was stale or duplicated"
                )
            raise LandingOutcomeDeviceError(
                "R06 censored reveal handle is not active and prevalidated"
            )
        try:
            for destination, after_image in lease.censor_tensor_swaps:
                destination.copy_(after_image)
            for name, value in lease.censor_host_after_state.items():
                self.__dict__[name] = value
            lease.published_terminal_token = lease.terminal_token
            return lease.terminal_token
        except Exception:
            self.poison_global_reveal_epoch(
                "R06 CENSOR child after-image publication failed"
            )
            raise

    def complete_global_reveal_epoch(
        self,
        child_terminal_token: R06ChildTerminalToken,
        r05_terminal_receipt: object,
    ) -> LandingRevealCommitReceipt | LandingRevealCensorReceipt:
        """Acknowledge exact R05-last publication and release this leaf epoch."""

        lease = self._active_reveal_prepare_lease
        owner = self._r05_terminal_owner
        if (
            self._poisoned
            or lease is None
            or type(child_terminal_token) is not R06ChildTerminalToken
            or child_terminal_token is not lease.published_terminal_token
            or child_terminal_token._token
            is not _REVEAL_TERMINAL_AUTH_TOKEN
            or child_terminal_token._owner_identity
            is not self._owner_identity
            or child_terminal_token._attempt is not lease.attempt
            or child_terminal_token._r05_terminal_claim
            is not lease.r05_terminal_claim
            or lease.r05_terminal_claim is None
            or lease.r05_terminal_claim_sha256 is None
            or lease.r05_terminal_boundary_authority_sha256 is None
            or lease.r05_terminal_boundary_projection_sha256 is None
            or type(lease.r05_terminal_content_pin)
            is not _r05.PreparedTerminalContentPin
            or lease.r05_terminal_content_pin_sha256 is None
            or lease.r05_terminal_content_bytes_sha256 is None
            or lease.r05_terminal_decision not in (
                _r05.TERMINAL_DECISION_ACCEPT,
                _r05.TERMINAL_DECISION_CENSOR,
            )
            or lease.r05_terminal_kind is None
            or lease.r05_terminal_sha256 is None
            or lease.global_boundary_receipt_sha256 is None
            or lease.global_boundary_packet_sha256 is None
            or type(owner) is not _r05.ContinuousRuntimeTransactionOwner
        ):
            self.poison_global_reveal_epoch(
                "R06 global reveal epoch acknowledgement is late, old, or foreign"
            )
            raise LandingOutcomeDeviceError(
                "R06 global reveal epoch acknowledgement is late, old, or foreign"
            )
        expected_child_type = (
            LandingRevealCommitReceipt
            if lease.r05_terminal_decision
            == _r05.TERMINAL_DECISION_ACCEPT
            else LandingRevealCensorReceipt
        )
        expected_terminal_type = (
            _r05.CommittedRevealBatch
            if lease.r05_terminal_decision
            == _r05.TERMINAL_DECISION_ACCEPT
            else _r05.CensoredRevealBatch
        )
        child_terminal_receipt = child_terminal_token._child_receipt
        if (
            type(child_terminal_receipt) is not expected_child_type
            or child_terminal_receipt.r05_terminal_claim_sha256
            != lease.r05_terminal_claim_sha256
            or child_terminal_receipt.r05_terminal_boundary_authority_sha256
            != lease.r05_terminal_boundary_authority_sha256
            or child_terminal_receipt.r05_terminal_boundary_projection_sha256
            != lease.r05_terminal_boundary_projection_sha256
            or child_terminal_receipt.r05_terminal_content_pin_sha256
            != lease.r05_terminal_content_pin_sha256
            or child_terminal_receipt.r05_terminal_schema_version
            != lease.r05_terminal_content_pin.terminal_schema_version
            or child_terminal_receipt.r05_terminal_content_bytes_base64
            != lease.r05_terminal_content_pin.content_bytes_base64
            or child_terminal_receipt.r05_terminal_content_byte_length
            != lease.r05_terminal_content_pin.content_byte_length
            or child_terminal_receipt.r05_terminal_content_bytes_sha256
            != lease.r05_terminal_content_bytes_sha256
            or child_terminal_receipt.expected_r05_terminal_kind
            != lease.r05_terminal_kind
            or child_terminal_receipt.expected_r05_terminal_sha256
            != lease.r05_terminal_sha256
            or child_terminal_receipt.reveal_final_preview_sha256
            != lease.attempt._receipt.reveal_final_preview_sha256
            or child_terminal_receipt.global_boundary_receipt_sha256
            != lease.global_boundary_receipt_sha256
            or child_terminal_receipt.global_boundary_packet_sha256
            != lease.global_boundary_packet_sha256
            or child_terminal_receipt.selected_env_ids
            != lease.attempt._receipt.selected_env_ids
        ):
            self.poison_global_reveal_epoch(
                "R06 pending R05 terminal facts differ"
            )
            raise LandingOutcomeDeviceError(
                "R06 pending R05 terminal facts differ"
            )
        try:
            actual = owner.require_owned_terminal_receipt(
                lease.r05_terminal_claim,
                r05_terminal_receipt,
                expected_claim_sha256=lease.r05_terminal_claim_sha256,
                expected_decision=lease.r05_terminal_decision,
                expected_reveal_final_preview_sha256=(
                    lease.attempt._receipt.reveal_final_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=(
                    lease.global_boundary_receipt_sha256
                ),
                expected_global_boundary_packet_sha256=(
                    lease.global_boundary_packet_sha256
                ),
                expected_terminal_boundary_authority_sha256=(
                    lease.r05_terminal_boundary_authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    lease.r05_terminal_boundary_projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    lease.r05_terminal_content_pin_sha256
                ),
                expected_terminal_kind=lease.r05_terminal_kind,
                expected_terminal_sha256=lease.r05_terminal_sha256,
                expected_selected_env_ids=(
                    lease.attempt._receipt.selected_env_ids
                ),
            )
        except Exception as exc:
            self.poison_global_reveal_epoch(
                "R06 R05 terminal receipt is stale or foreign"
            )
            raise LandingOutcomeDeviceError(
                "R06 R05 terminal receipt is stale or foreign"
            ) from exc
        if (
            actual is not r05_terminal_receipt
            or type(actual) is not expected_terminal_type
            or type(actual).KIND != lease.r05_terminal_kind
            or actual.canonical_sha256 != lease.r05_terminal_sha256
            or actual.selected_env_ids
            != lease.attempt._receipt.selected_env_ids
            or actual.reveal_final_preview.canonical_sha256
            != lease.attempt._receipt.reveal_final_preview_sha256
        ):
            self.poison_global_reveal_epoch(
                "R06 R05 terminal receipt facts differ"
            )
            raise LandingOutcomeDeviceError(
                "R06 R05 terminal receipt facts differ"
            )
        self._terminal_resolution_total.add_(
            len(lease.attempt._receipt.selected_env_ids)
        )
        self._active_reveal_prepare_lease = None
        return child_terminal_receipt

    def reveal_prepare_boundary(
        self, _attempt: LandingRevealPrepareAttempt
    ) -> None:
        """Tombstone the obsolete R06-only packed D2H boundary."""

        raise LandingOutcomeDeviceError(
            "R06 local reveal boundary is tombstoned; use the all-owner boundary"
        )

    def arm_all_owner_reveal_prepare_marker(self, *_args: object, **_kwargs: object) -> None:
        """Tombstone the obsolete caller-composed R05 marker path."""

        raise LandingOutcomeDeviceError(
            "R06 marker arm is tombstoned; consume the exact global receipt"
        )

    def censor_global_reveal_boundary(self, *_args: object, **_kwargs: object) -> None:
        """Tombstone the unsafe one-step CENSOR transition."""

        raise LandingOutcomeDeviceError(
            "use arm_censored_global_reveal_boundary then commit_censored_prevalidated"
        )

    def poison_global_reveal_epoch(self, reason: str) -> None:
        """Idempotently fail closed so a coordinator can broadcast poison."""

        if self._global_reveal_poison_reason is None:
            self._global_reveal_poison_reason = (
                reason
                if type(reason) is str and reason.strip() != ""
                else "global reveal coordinator poison"
            )
        self._poisoned = True

    def abort_prepared_reveal_install(
        self,
        attempt: LandingRevealPrepareAttempt,
        *,
        abort_capability: object | None = None,
    ) -> DeviceMutationResult:
        """Discard only one unarmed private attempt before global transfer."""

        self._require_operable(allow_active_prepare=True)
        lease = self._active_reveal_prepare_lease
        if (
            lease is None
            or type(attempt) is not LandingRevealPrepareAttempt
            or attempt is not lease.attempt
            or lease.armed_install is not None
            or lease.censored_install is not None
        ):
            raise LandingOutcomeDeviceError(
                "R06 abort requires the exact active unarmed attempt"
            )
        _boundary_owner, boundary_lane = self._reveal_boundary_binding()
        boundary_lane.require_abortable_device_row(
            lease.boundary_row,
            expected_prepared_token=attempt,
            abort_capability=abort_capability,
        )
        selected = lease.attempt._legacy_install._device_pack._values["mask"]
        result = DeviceMutationResult(
            accepted=torch.zeros_like(selected),
            rejected=selected.detach().clone(),
            fault_bits=torch.zeros(
                (self.num_envs,), dtype=torch.int64, device=self.device
            ),
        )
        self._active_reveal_prepare_lease = None
        return result


    def install_reveal(self, request: LandingRevealInstall) -> DeviceMutationResult:
        """Tombstone the post-commit ingress that could lose an opportunity."""

        raise LandingOutcomeDeviceError(
            "install_reveal is tombstoned; use preview prepare/boundary/arm/commit"
        )

    def _install_reveal_legacy_impl(
        self,
        request: LandingRevealInstall,
        *,
        allocated_mailbox_slot: torch.Tensor | None = None,
    ) -> DeviceMutationResult:
        """Apply a fully prevalidated private after-image on a shadow owner only."""

        receipt, pack = _owned_landing_reveal_install(request)
        if (
            receipt["num_envs"] != self.num_envs
            or receipt["dtype"] != str(self.dtype)
            or receipt["device"] != str(self.device)
            or receipt["r05_source_sha256"]
            != self.runtime_binding.r05_source_sha256
            or receipt["r05_contract_sha256"]
            != self.runtime_binding.r05_contract_sha256
            or receipt["c05_source_sha256"]
            != self.runtime_binding.c05_source_sha256
            or receipt["profile_sha256"] != self.profile.canonical_sha256
            or receipt["profile_sha256"]
            != self.runtime_binding.landing_profile_sha256
            or receipt["text_registry_sha256"]
            != self.runtime_binding.text_registry_sha256
        ):
            raise LandingOutcomeDeviceError("install receipt runtime roots differ")
        for row in receipt["rows"]:
            try:
                host_key = _c05.LandingOutcomeShotKey.from_mapping(row["outcome_key"])
                identity = LandingPlacementTaskIdentity(
                    frame_id=self.profile.frame_id,
                    frame_binding_sha256=self.profile.frame_binding_sha256,
                    profile_sha256=self.profile.canonical_sha256,
                    task_receipt_sha256=host_key.task_sha256,
                    semantic_binding_sha256=row["target_semantic_sha256"],
                    instance_binding_sha256=row["committed_reveal_sha256"],
                    target_x_m=row["target_xy_m"][0],
                    target_y_m=row["target_xy_m"][1],
                )
            except Exception as exc:
                raise LandingOutcomeDeviceError(
                    "install receipt C04 task identity cannot be reconstructed"
                ) from exc
            if identity.canonical_sha256 != row["task_identity_sha256"]:
                raise LandingOutcomeDeviceError("install receipt C04 task identity differs")
        expected_pack = _device_values_from_install_receipt(
            receipt,
            device=self.device,
            dtype=self.dtype,
        )
        n = (self.num_envs,)
        mask = self._tensor(pack["mask"], name="mask", shape=n, dtype=torch.bool)
        flight_slot = self._tensor(
            pack["flight_slot"], name="flight_slot", shape=n, dtype=torch.int64
        )
        template_mailbox_slot = self._tensor(
            pack["mailbox_slot"], name="mailbox_slot", shape=n, dtype=torch.int64
        )
        mailbox_slot = (
            template_mailbox_slot
            if allocated_mailbox_slot is None
            else self._tensor(
                allocated_mailbox_slot,
                name="allocated_mailbox_slot",
                shape=n,
                dtype=torch.int64,
            )
        )
        key = self._key(pack["task_key"], prefix=n)
        full_key = self._tensor(
            pack["full_key_sha256"],
            name="full_key_sha256",
            shape=n + (TOKEN_BYTES,),
            dtype=torch.uint8,
        )
        full_key_receipt = self._tensor(
            pack["full_key_receipt_sha256"],
            name="full_key_receipt_sha256",
            shape=n + (TOKEN_BYTES,),
            dtype=torch.uint8,
        )
        committed_reveal = self._tensor(
            pack["committed_reveal_sha256"],
            name="committed_reveal_sha256",
            shape=n + (TOKEN_BYTES,),
            dtype=torch.uint8,
        )
        install_receipt = self._tensor(
            pack["install_receipt_sha256"],
            name="install_receipt_sha256",
            shape=n + (TOKEN_BYTES,),
            dtype=torch.uint8,
        )
        generation = self._tensor(
            pack["ball_generation"],
            name="ball_generation",
            shape=n,
            dtype=torch.int64,
        )
        task_token = self._tensor(
            pack["task_identity_token"],
            name="task_identity_token",
            shape=n + (TOKEN_BYTES,),
            dtype=torch.uint8,
        )
        target = self._float_tensor(
            pack["target_xy_m"], name="target_xy_m", shape=n + (2,)
        )
        reveal = self._tensor(
            pack["reveal_control_step"],
            name="reveal_control_step",
            shape=n,
            dtype=torch.int64,
        )
        deadline = self._tensor(
            pack["selected_contact_deadline_control_step"],
            name="selected_contact_deadline_control_step",
            shape=n,
            dtype=torch.int64,
        )
        horizon = self._tensor(
            pack["first_crossing_horizon_control_step"],
            name="first_crossing_horizon_control_step",
            shape=n,
            dtype=torch.int64,
        )

        expected_key = expected_pack["task_key"]
        if not isinstance(expected_key, DeviceLandingOutcomeKey):
            raise LandingOutcomeDeviceError("expected install key differs")
        pack_row_match = mask == expected_pack["mask"]
        for actual, expected in (
            (flight_slot, expected_pack["flight_slot"]),
            (template_mailbox_slot, expected_pack["mailbox_slot"]),
            (generation, expected_pack["ball_generation"]),
            (reveal, expected_pack["reveal_control_step"]),
            (deadline, expected_pack["selected_contact_deadline_control_step"]),
            (horizon, expected_pack["first_crossing_horizon_control_step"]),
        ):
            pack_row_match = pack_row_match & torch.eq(actual, expected)
        for actual, expected in (
            (full_key, expected_pack["full_key_sha256"]),
            (full_key_receipt, expected_pack["full_key_receipt_sha256"]),
            (committed_reveal, expected_pack["committed_reveal_sha256"]),
            (install_receipt, expected_pack["install_receipt_sha256"]),
            (task_token, expected_pack["task_identity_token"]),
            (target, expected_pack["target_xy_m"]),
        ):
            pack_row_match = pack_row_match & torch.eq(actual, expected).all(dim=-1)
        for name in _INT_KEY_FIELDS:
            pack_row_match = pack_row_match & torch.eq(
                getattr(key, name), getattr(expected_key, name)
            )
        for name in _DIGEST_KEY_FIELDS:
            pack_row_match = pack_row_match & torch.eq(
                getattr(key, name), getattr(expected_key, name)
            ).all(dim=-1)
        batch_pack_match = pack_row_match.all()
        expected_mask = expected_pack["mask"]
        attempted = mask | expected_mask

        flight_in_range = (flight_slot >= 0) & (
            flight_slot < self.flight_slot_capacity
        )
        mailbox_in_range = (mailbox_slot >= 0) & (
            mailbox_slot < self.mailbox_capacity
        )
        safe_flight = flight_slot.clamp(0, self.flight_slot_capacity - 1)
        safe_mailbox = mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        selected_flight_state = torch.gather(
            self._flight_state.to(torch.int64), 1, safe_flight.unsqueeze(1)
        ).squeeze(1)
        selected_mailbox_reserved = torch.gather(
            self._mailbox_reserved, 1, safe_mailbox.unsqueeze(1)
        ).squeeze(1)
        selected_mailbox_state = torch.gather(
            self._mailbox_state.to(torch.int64), 1, safe_mailbox.unsqueeze(1)
        ).squeeze(1)

        key_values_valid = (
            (key.env_id == self._env_ids)
            & (key.reset_generation >= 1)
            & (key.swing_generation >= 0)
            & (key.action_uid >= 1)
            & (key.action_uid <= _MAX_ACTION_UID)
            & (key.action_slot >= 0)
            & (key.shot_index >= 1)
        )
        digest_values_valid = torch.ones_like(mask)
        for name in _DIGEST_KEY_FIELDS:
            digest_values_valid = digest_values_valid & getattr(key, name).ne(0).any(dim=-1)
        target_valid = (
            torch.isfinite(target).all(dim=-1)
            & (target[:, 0] >= self.profile.opponent_table_x_min_m)
            & (target[:, 0] <= self.profile.opponent_table_x_max_m)
            & (target[:, 1] >= self.profile.table_y_min_m)
            & (target[:, 1] <= self.profile.table_y_max_m)
        )
        scalar_values_valid = (
            (generation >= 0)
            & (reveal >= 0)
            & (deadline >= reveal)
            & (horizon >= deadline)
            & full_key.ne(0).any(dim=-1)
            & full_key_receipt.ne(0).any(dim=-1)
            & committed_reveal.ne(0).any(dim=-1)
            & install_receipt.ne(0).any(dim=-1)
            & task_token.ne(0).any(dim=-1)
        )
        valid_authority = batch_pack_match & pack_row_match & expected_mask
        invalid = attempted & ~(
            valid_authority
            & flight_in_range
            & mailbox_in_range
            & key_values_valid
            & digest_values_valid
            & target_valid
            & scalar_values_valid
        )
        poisoned_attempt = attempted & self._device_sticky_poison
        invalid = invalid | poisoned_attempt
        flight_collision = valid_authority & flight_in_range & (
            selected_flight_state != FLIGHT_EMPTY
        )
        mailbox_collision = valid_authority & mailbox_in_range & (
            selected_mailbox_reserved | (selected_mailbox_state != MAILBOX_EMPTY)
        )

        reset_generation = key.reset_generation
        swing_generation = key.swing_generation
        replay_ordered = (~self._replay_valid) | (
            (reset_generation > self._replay_reset_generation)
            | (
                (reset_generation == self._replay_reset_generation)
                & (swing_generation > self._replay_swing_generation)
            )
        )
        active_token_match = (
            torch.eq(
                self._flight_full_key_sha256,
                full_key.unsqueeze(1).expand(-1, self.flight_slot_capacity, -1),
            ).all(dim=-1)
            & (self._flight_state != FLIGHT_EMPTY)
        ).any(dim=1)
        mailbox_token_match = (
            torch.eq(
                self._mailbox_full_key_sha256,
                full_key.unsqueeze(1).expand(-1, self.mailbox_capacity, -1),
            ).all(dim=-1)
            & self._mailbox_history_valid
        ).any(dim=1)
        replay = valid_authority & (
            ~replay_ordered | active_token_match | mailbox_token_match
        )

        row_clear = valid_authority & ~(
            invalid | flight_collision | mailbox_collision | replay
        )
        batch_clear = ((~expected_mask) | row_clear).all()
        accepted = row_clear & batch_clear
        batch_abort = expected_mask & ~batch_clear
        rejected = attempted & ~accepted
        fault_bits = torch.zeros(n, dtype=torch.int64, device=self.device)
        fault_bits = torch.where(
            invalid, torch.full_like(fault_bits, FAULT_INVALID_INSTALL), fault_bits
        )
        fault_bits = torch.where(
            poisoned_attempt,
            torch.bitwise_or(
                fault_bits,
                torch.full_like(fault_bits, FAULT_SAFETY_CLEANUP),
            ),
            fault_bits,
        )
        fault_bits = torch.where(
            flight_collision,
            torch.bitwise_or(fault_bits, torch.full_like(fault_bits, FAULT_FLIGHT_COLLISION)),
            fault_bits,
        )
        fault_bits = torch.where(
            mailbox_collision,
            torch.bitwise_or(fault_bits, torch.full_like(fault_bits, FAULT_MAILBOX_COLLISION)),
            fault_bits,
        )
        fault_bits = torch.where(
            replay,
            torch.bitwise_or(fault_bits, torch.full_like(fault_bits, FAULT_REPLAY)),
            fault_bits,
        )
        fault_bits = torch.where(
            batch_abort,
            torch.bitwise_or(
                fault_bits,
                torch.full_like(fault_bits, FAULT_BATCH_ABORT),
            ),
            fault_bits,
        )
        self._ingress_fault_bits.bitwise_or_(fault_bits)

        flight_target = accepted.unsqueeze(1) & (
            self._flight_slot_ids == safe_flight.unsqueeze(1)
        )
        mailbox_target = accepted.unsqueeze(1) & (
            self._mailbox_slot_ids == safe_mailbox.unsqueeze(1)
        )
        for name in _INT_KEY_FIELDS:
            source = getattr(key, name).unsqueeze(1).expand(self._flight_shape)
            _masked_copy_(self._flight_key_ints[name], source, flight_target)
        for name in _DIGEST_KEY_FIELDS:
            source = getattr(key, name).unsqueeze(1).expand(
                self._flight_shape + (TOKEN_BYTES,)
            )
            _masked_copy_(self._flight_key_digests[name], source, flight_target)
        _masked_copy_(
            self._flight_full_key_sha256,
            full_key.unsqueeze(1).expand(self._flight_shape + (TOKEN_BYTES,)),
            flight_target,
        )
        _masked_copy_(
            self._flight_full_key_receipt_sha256,
            full_key_receipt.unsqueeze(1).expand(
                self._flight_shape + (TOKEN_BYTES,)
            ),
            flight_target,
        )
        _masked_copy_(
            self._flight_committed_reveal_sha256,
            committed_reveal.unsqueeze(1).expand(
                self._flight_shape + (TOKEN_BYTES,)
            ),
            flight_target,
        )
        _masked_copy_(
            self._flight_install_receipt_sha256,
            install_receipt.unsqueeze(1).expand(
                self._flight_shape + (TOKEN_BYTES,)
            ),
            flight_target,
        )
        _masked_fill_(self._flight_action_epoch, flight_target, 0)
        _masked_copy_(
            self._flight_ball_generation,
            generation.unsqueeze(1).expand(self._flight_shape),
            flight_target,
        )
        _masked_copy_(
            self._flight_mailbox_slot,
            safe_mailbox.unsqueeze(1).expand(self._flight_shape),
            flight_target,
        )
        _masked_copy_(
            self._flight_task_identity_token,
            task_token.unsqueeze(1).expand(self._flight_shape + (TOKEN_BYTES,)),
            flight_target,
        )
        _masked_copy_(
            self._flight_target_xy_m,
            target.unsqueeze(1).expand(self._flight_shape + (2,)),
            flight_target,
        )
        for destination, source in (
            (self._flight_reveal_control_step, reveal),
            (self._flight_contact_deadline_control_step, deadline),
            (self._flight_crossing_horizon_control_step, horizon),
        ):
            _masked_copy_(
                destination, source.unsqueeze(1).expand(self._flight_shape), flight_target
            )
        _masked_fill_(self._flight_state, flight_target, FLIGHT_INBOUND)
        _masked_fill_(self._flight_physical_retired, flight_target, 0)
        _masked_fill_(self._flight_fault_bits, flight_target, 0)
        for destination, value in (
            (self._flight_contact_valid, 0),
            (self._flight_net_crossed, 0),
            (self._flight_net_clear, 0),
        ):
            _masked_fill_(destination, flight_target, value)
        for destination in (
            self._flight_contact_stamp_control,
            self._flight_observation_ordinal,
            self._flight_last_observation_control,
            self._flight_net_stamp_control,
        ):
            _masked_fill_(destination, flight_target, -1)
        for destination in (
            self._flight_contact_stamp_substep,
            self._flight_last_observation_substep,
            self._flight_net_stamp_substep,
        ):
            _masked_fill_(destination, flight_target, -1)
        for destination in (
            self._flight_contact_ball_center_m,
            self._flight_outgoing_anchor_m,
            self._flight_last_ball_center_m,
        ):
            _masked_fill_(destination, flight_target, 0.0)

        _masked_fill_(self._mailbox_reserved, mailbox_target, 1)
        _masked_fill_(self._mailbox_action_epoch, mailbox_target, 0)
        _masked_copy_(
            self._mailbox_reservation_token,
            full_key.unsqueeze(1).expand(self._mailbox_shape + (TOKEN_BYTES,)),
            mailbox_target,
        )
        _masked_copy_(
            self._mailbox_reservation_generation,
            generation.unsqueeze(1).expand(self._mailbox_shape),
            mailbox_target,
        )
        _masked_copy_(
            self._mailbox_reserved_flight_slot,
            safe_flight.unsqueeze(1).expand(self._mailbox_shape),
            mailbox_target,
        )

        self._replay_valid.copy_(torch.where(accepted, torch.ones_like(mask), self._replay_valid))
        self._replay_reset_generation.copy_(
            torch.where(accepted, reset_generation, self._replay_reset_generation)
        )
        self._replay_swing_generation.copy_(
            torch.where(accepted, swing_generation, self._replay_swing_generation)
        )
        self._replay_action_epoch.copy_(
            torch.where(accepted, torch.zeros_like(mask), self._replay_action_epoch)
        )
        _masked_copy_(self._replay_full_key_sha256, full_key, accepted)
        self._reset_generation_highwater.copy_(
            torch.where(
                accepted,
                torch.maximum(
                    self._reset_generation_highwater,
                    reset_generation,
                ),
                self._reset_generation_highwater,
            )
        )
        self._installed_total.add_(accepted.to(torch.int64).sum())
        self._record_fault_events(fault_bits)
        self._increment_mutation()
        return self._mutation_result(accepted, rejected, fault_bits)

    def publish_post_physics(
        self, batch: PostPhysicsFlightBatch
    ) -> PostPhysicsMutationResult:
        """Consume one fixed-cadence observation for every physical slot."""

        result = self._publish_post_physics_impl(batch, action_epoch_direct=False)
        if type(result) is not PostPhysicsMutationResult:
            raise LandingOutcomeDeviceError("legacy postphysics result type differs")
        return result

    def publish_action_ball_full_mdp_epoch_post_physics(
        self,
    ) -> ActionEpochR06PostPhysicsResult:
        """Pull and settle Physical's one active typed packet without a caller payload."""

        return self._pull_action_ball_full_mdp_epoch_post_physics(
            defer_control_finalize=False
        )

    def sample_action_ball_full_mdp_epoch_post_physics(
        self,
    ) -> ActionEpochR06PostPhysicsSample:
        """Consume one causal substep and retain terminal work for control flush."""

        return self._pull_action_ball_full_mdp_epoch_post_physics(
            defer_control_finalize=True
        )

    def _pull_action_ball_full_mdp_epoch_post_physics(
        self, *, defer_control_finalize: bool
    ) -> ActionEpochR06PostPhysicsResult | ActionEpochR06PostPhysicsSample:
        if type(defer_control_finalize) is not bool:
            raise LandingOutcomeDeviceError(
                "R06 epoch postphysics finalize mode differs"
            )

        epoch_owner = self._action_ball_full_mdp_epoch_owner
        park_authority = self._physical_park_token_authority
        if epoch_owner is None or park_authority is None:
            raise LandingOutcomeDeviceError(
                "R06 epoch postphysics requires bound ActionEpoch and Physical owners"
            )
        if self._action_epoch_post_physics_result is not None:
            raise LandingOutcomeDeviceError(
                "R06 epoch postphysics requires retirement of the prior result"
            )
        physical_owner = park_authority.physical_owner
        if __package__:
            from . import action_ball_physical_flight_device as physical_v1
        else:
            import action_ball_physical_flight_device as physical_v1

        validator = physical_owner.require_owned_action_epoch_r06_postphysics_projection
        if (
            type(physical_owner) is not physical_v1.ActionBallPhysicalFlightDeviceOwner
            or getattr(validator, "__self__", None) is not physical_owner
            or getattr(validator, "__func__", None)
            is not physical_v1.ActionBallPhysicalFlightDeviceOwner.require_owned_action_epoch_r06_postphysics_projection
        ):
            raise LandingOutcomeDeviceError(
                "R06 epoch postphysics requires the exact bound Physical owner method"
            )
        view = validator()
        if (
            type(view) is not physical_v1.ActionEpochR06PostPhysicsProjection
            or view.physical_owner is not physical_owner
            or view.epoch_owner is not epoch_owner
        ):
            raise LandingOutcomeDeviceError(
                "R06 epoch postphysics Physical projection is foreign"
            )
        result = self._publish_post_physics_impl(
            view,
            action_epoch_direct=True,
            defer_action_epoch_control_finalize=defer_control_finalize,
        )
        expected = (
            ActionEpochR06PostPhysicsSample
            if defer_control_finalize
            else ActionEpochR06PostPhysicsResult
        )
        if type(result) is not expected:
            raise LandingOutcomeDeviceError("R06 epoch postphysics result type differs")
        return result

    def _publish_post_physics_impl(
        self,
        batch: object,
        *,
        action_epoch_direct: bool,
        defer_action_epoch_control_finalize: bool = False,
    ) -> (
        PostPhysicsMutationResult
        | ActionEpochR06PostPhysicsResult
        | ActionEpochR06PostPhysicsSample
    ):
        """Shared physics math after either legacy or typed-direct ingress."""

        self._require_operable(
            allow_action_epoch_control_window=(
                action_epoch_direct and defer_action_epoch_control_finalize
            )
        )
        if defer_action_epoch_control_finalize and not action_epoch_direct:
            raise LandingOutcomeDeviceError(
                "legacy postphysics cannot defer an ActionEpoch control finalize"
            )
        if not action_epoch_direct and not isinstance(batch, PostPhysicsFlightBatch):
            raise LandingOutcomeDeviceError("batch must be PostPhysicsFlightBatch")
        physical_publication_identity = (
            None if action_epoch_direct else batch.physical_publication_identity
        )
        if not action_epoch_direct and physical_publication_identity is None:
            raise LandingOutcomeDeviceError(
                "physical_publication_identity must be the exact physical publication"
            )
        p = self._flight_shape
        observe = self._tensor(
            batch.observe_mask, name="observe_mask", shape=p, dtype=torch.bool
        )
        full_key = None
        if not action_epoch_direct:
            full_key = self._tensor(
                batch.full_key_sha256,
                name="full_key_sha256",
                shape=p + (TOKEN_BYTES,),
                dtype=torch.uint8,
            )
        direct_shot_key = None
        direct_publication_ordinal = None
        if action_epoch_direct:
            direct_shot_key = _row_identity.require_action_epoch_shot_key(
                getattr(batch, "shot_key", None),
                shape=p,
                device=self.device,
                label="R06 epoch postphysics shot_key",
            )
            direct_publication_ordinal = self._tensor(
                getattr(batch, "publication_ordinal", None),
                name="publication_ordinal",
                shape=p,
                dtype=torch.int64,
            )
            generation = direct_shot_key.ball_generation
        else:
            generation = self._tensor(
                batch.ball_generation,
                name="ball_generation",
                shape=p,
                dtype=torch.int64,
            )
        ordinal = self._tensor(
            batch.observation_ordinal,
            name="observation_ordinal",
            shape=p,
            dtype=torch.int64,
        )
        previous = self._float_tensor(
            batch.previous_ball_center_m,
            name="previous_ball_center_m",
            shape=p + (3,),
        )
        current = self._float_tensor(
            batch.current_ball_center_m,
            name="current_ball_center_m",
            shape=p + (3,),
        )
        def stamp(value: object, *, name: str) -> PhysicsStampBatch:
            if not action_epoch_direct:
                return self._stamp(value, name=name, prefix=p)
            return PhysicsStampBatch(
                control_step=self._tensor(
                    getattr(value, "control_step", None),
                    name=name + ".control_step",
                    shape=p,
                    dtype=torch.int64,
                ),
                physics_substep=self._tensor(
                    getattr(value, "physics_substep", None),
                    name=name + ".physics_substep",
                    shape=p,
                    dtype=torch.int32,
                ),
                event_phase=self._tensor(
                    getattr(value, "event_phase", None),
                    name=name + ".event_phase",
                    shape=p,
                    dtype=torch.int8,
                ),
            )

        observation_stamp = stamp(
            batch.observation_stamp, name="observation_stamp"
        )
        contact_event = self._tensor(
            batch.selected_contact_event,
            name="selected_contact_event",
            shape=p,
            dtype=torch.bool,
        )
        contact_center = self._float_tensor(
            batch.selected_contact_ball_center_m,
            name="selected_contact_ball_center_m",
            shape=p + (3,),
        )
        outgoing_anchor = self._float_tensor(
            batch.selected_contact_outgoing_segment_anchor_m,
            name="selected_contact_outgoing_segment_anchor_m",
            shape=p + (3,),
        )
        contact_stamp = stamp(
            batch.selected_contact_stamp, name="selected_contact_stamp"
        )
        net_event = self._tensor(
            batch.net_crossing_event,
            name="net_crossing_event",
            shape=p,
            dtype=torch.bool,
        )
        net_clear = self._tensor(
            batch.net_clear_at_crossing,
            name="net_clear_at_crossing",
            shape=p,
            dtype=torch.bool,
        )
        net_stamp = stamp(batch.net_crossing_stamp, name="net_crossing_stamp")
        report_delivered = self._tensor(
            batch.crossing_report_delivered,
            name="crossing_report_delivered",
            shape=p,
            dtype=torch.bool,
        )
        crossing_event = self._tensor(
            batch.first_descending_crossing_event,
            name="first_descending_crossing_event",
            shape=p,
            dtype=torch.bool,
        )
        crossing_xy = self._float_tensor(
            batch.first_descending_crossing_xy_m,
            name="first_descending_crossing_xy_m",
            shape=p + (2,),
        )
        crossing_stamp = stamp(
            batch.first_descending_crossing_stamp,
            name="first_descending_crossing_stamp",
        )
        nonfinite_flag = self._tensor(
            batch.nonfinite_observation,
            name="nonfinite_observation",
            shape=p,
            dtype=torch.bool,
        )
        producer_fault = self._tensor(
            batch.producer_contract_fault,
            name="producer_contract_fault",
            shape=p,
            dtype=torch.bool,
        )
        engine_overflow = self._tensor(
            batch.engine_overflow,
            name="engine_overflow",
            shape=p,
            dtype=torch.bool,
        )

        direct_flight_slot = None
        if action_epoch_direct:
            direct_flight_slot = self._tensor(
                getattr(batch, "flight_slot", None),
                name="flight_slot",
                shape=p,
                dtype=torch.int64,
            )

        live = (
            (self._flight_state == FLIGHT_INBOUND)
            | (self._flight_state == FLIGHT_OPEN)
        )
        if action_epoch_direct:
            live = (
                live
                & self._flight_action_epoch
                & self._action_epoch_control_pending_cause.eq(
                    SETTLEMENT_CAUSE_NONE
                )
            )
        observed_nonlive = observe & ~live
        missing_live = live & ~observe
        if action_epoch_direct:
            assert direct_shot_key is not None
            assert direct_publication_ordinal is not None
            assert direct_flight_slot is not None
            primary_identity_match = (
                direct_flight_slot.eq(
                    self._flight_slot_ids.expand(self._flight_shape)
                )
                & _row_identity.action_epoch_shot_key_equal(
                    direct_shot_key, self._flight_action_epoch_shot_key()
                )
                & direct_publication_ordinal.eq(
                    self._flight_publication_ordinal
                )
            )
        else:
            assert full_key is not None
            primary_identity_match = torch.eq(
                full_key, self._flight_full_key_sha256
            ).all(dim=-1)
        generation_match = generation == self._flight_ball_generation
        identity_match = primary_identity_match & generation_match
        bound = observe & live & identity_match

        first_observation = self._flight_observation_ordinal < 0
        ordinal_ok = torch.where(
            first_observation,
            ordinal == 0,
            ordinal == (self._flight_observation_ordinal + 1),
        )
        # Formal receipt-backed rows retain the frozen one-control-step cadence.
        # ActionEpoch rows are sampled every physics substep, so their exact
        # chronology is ordinal plus the lexicographic stamp check below.
        cadence_ok = self._flight_action_epoch | (
            observation_stamp.control_step
            == (self._flight_reveal_control_step + ordinal)
        )
        action_epoch_first_stamp_ok = (
            ~self._flight_action_epoch
            | ~first_observation
            | observation_stamp.control_step.ge(
                self._flight_reveal_control_step
            )
        )
        continuity_ok = first_observation | torch.eq(
            previous, self._flight_last_ball_center_m
        ).all(dim=-1)
        observation_stamp_valid = (
            (observation_stamp.control_step >= 0)
            & (observation_stamp.physics_substep >= 0)
            & (observation_stamp.event_phase == PHASE_LANDING)
        )
        previous_observation = PhysicsStampBatch(
            control_step=self._flight_last_observation_control,
            physics_substep=self._flight_last_observation_substep,
            event_phase=torch.full(
                p, PHASE_LANDING, dtype=torch.int8, device=self.device
            ),
        )
        stamp_monotone = first_observation | _stamp_less(
            previous_observation, observation_stamp
        )

        contact_stamp_valid = (
            (contact_stamp.control_step >= self._flight_reveal_control_step)
            & (
                contact_stamp.control_step
                <= self._flight_contact_deadline_control_step
            )
            & (contact_stamp.physics_substep >= 0)
            & (contact_stamp.event_phase == PHASE_CONTACT)
            & _stamp_less_equal(contact_stamp, observation_stamp)
        )
        contact_geometry_finite = torch.isfinite(contact_center).all(dim=-1) & torch.isfinite(
            outgoing_anchor
        ).all(dim=-1)
        contact_anchor_match = torch.eq(contact_center, outgoing_anchor).all(dim=-1)
        valid_contact = (
            bound
            & contact_event
            & (self._flight_state == FLIGHT_INBOUND)
            & contact_stamp_valid
            & contact_geometry_finite
            & contact_anchor_match
        )
        effective_contact = self._flight_contact_valid | valid_contact
        effective_contact_control = torch.where(
            valid_contact,
            contact_stamp.control_step,
            self._flight_contact_stamp_control,
        )
        effective_contact_substep = torch.where(
            valid_contact,
            contact_stamp.physics_substep,
            self._flight_contact_stamp_substep,
        )

        segment_start = torch.where(
            valid_contact.unsqueeze(-1), outgoing_anchor, previous
        )
        finite_segment = torch.isfinite(segment_start).all(dim=-1) & torch.isfinite(
            current
        ).all(dim=-1)
        landing_plane = self.profile.ball_center_landing_plane_z_m
        start_z = segment_start[..., 2]
        current_z = current[..., 2]
        segment_crossing = (
            bound
            & effective_contact
            & finite_segment
            & (current_z < start_z)
            & (start_z >= landing_plane)
            & (current_z <= landing_plane)
        )
        denominator = current_z - start_z
        safe_denominator = torch.where(
            segment_crossing, denominator, torch.ones_like(denominator)
        )
        ratio = (landing_plane - start_z) / safe_denominator
        segment_xy = segment_start[..., :2] + ratio.unsqueeze(-1) * (
            current[..., :2] - segment_start[..., :2]
        )

        crossing_after_contact = _stamp_less_fields(
            effective_contact_control,
            effective_contact_substep,
            torch.full(p, PHASE_CONTACT, dtype=torch.int8, device=self.device),
            crossing_stamp.control_step,
            crossing_stamp.physics_substep,
            crossing_stamp.event_phase,
        )
        incoming_crossing_same_batch = (
            valid_contact & crossing_event & ~crossing_after_contact
        )
        effective_crossing_event = crossing_event & ~incoming_crossing_same_batch
        effective_report_delivered = report_delivered & ~incoming_crossing_same_batch
        report_missing = (
            effective_contact
            & ~effective_report_delivered
            & ~incoming_crossing_same_batch
        )
        both_crossings = (
            effective_report_delivered & effective_crossing_event & segment_crossing
        )
        event_only_crossing = (
            effective_report_delivered & effective_crossing_event & ~segment_crossing
        )
        segment_only_crossing = (
            effective_report_delivered & ~effective_crossing_event & segment_crossing
        )
        crossing_event_stamp_valid = (
            (crossing_stamp.control_step >= 0)
            & (crossing_stamp.physics_substep >= 0)
            & (crossing_stamp.event_phase == PHASE_LANDING)
            & _stamp_less_equal(crossing_stamp, observation_stamp)
            & (
                crossing_stamp.control_step
                <= self._flight_crossing_horizon_control_step
            )
            & _stamp_less_fields(
                effective_contact_control,
                effective_contact_substep,
                torch.full(p, PHASE_CONTACT, dtype=torch.int8, device=self.device),
                crossing_stamp.control_step,
                crossing_stamp.physics_substep,
                crossing_stamp.event_phase,
            )
        )
        both_xy_match = (
            torch.abs(crossing_xy - segment_xy) <= C05_CROSSING_ABS_TOL_M
        ).all(dim=-1)
        report_xy_finite = torch.isfinite(crossing_xy).all(dim=-1)

        net_stamp_valid = (
            (net_stamp.control_step >= 0)
            & (net_stamp.physics_substep >= 0)
            & (net_stamp.event_phase == PHASE_NET)
            & _stamp_less_equal(net_stamp, observation_stamp)
            & _stamp_less_fields(
                effective_contact_control,
                effective_contact_substep,
                torch.full(p, PHASE_CONTACT, dtype=torch.int8, device=self.device),
                net_stamp.control_step,
                net_stamp.physics_substep,
                net_stamp.event_phase,
            )
        )
        net_after_contact = _stamp_less_fields(
            effective_contact_control,
            effective_contact_substep,
            torch.full(p, PHASE_CONTACT, dtype=torch.int8, device=self.device),
            net_stamp.control_step,
            net_stamp.physics_substep,
            net_stamp.event_phase,
        )
        incoming_net_same_batch = valid_contact & net_event & ~net_after_contact
        effective_net_event = net_event & ~incoming_net_same_batch
        effective_net_clear_report = net_clear & ~incoming_net_same_batch
        duplicate_net = effective_net_event & self._flight_net_crossed
        clear_without_net_event = effective_net_clear_report & ~effective_net_event
        valid_new_net = (
            bound
            & effective_contact
            & effective_net_event
            & ~self._flight_net_crossed
            & net_stamp_valid
        )
        effective_net_crossed = self._flight_net_crossed | valid_new_net
        effective_net_clear = torch.where(
            valid_new_net, effective_net_clear_report, self._flight_net_clear
        )
        effective_net_control = torch.where(
            valid_new_net, net_stamp.control_step, self._flight_net_stamp_control
        )
        effective_net_substep = torch.where(
            valid_new_net, net_stamp.physics_substep, self._flight_net_stamp_substep
        )
        crossing_candidate = both_crossings | event_only_crossing | segment_only_crossing
        crossing_order_after_net = ~effective_net_crossed | _stamp_less_fields(
            effective_net_control,
            effective_net_substep,
            torch.full(p, PHASE_NET, dtype=torch.int8, device=self.device),
            torch.where(
                effective_report_delivered & effective_crossing_event,
                crossing_stamp.control_step,
                observation_stamp.control_step,
            ),
            torch.where(
                effective_report_delivered & effective_crossing_event,
                crossing_stamp.physics_substep,
                observation_stamp.physics_substep,
            ),
            torch.full(p, PHASE_LANDING, dtype=torch.int8, device=self.device),
        )

        ordinal_fault = bound & (
            ~ordinal_ok | ~cadence_ok | ~action_epoch_first_stamp_ok
        )
        continuity_fault = bound & ~continuity_ok
        observation_stamp_fault = bound & ~observation_stamp_valid
        stamp_regression_fault = bound & observation_stamp_valid & ~stamp_monotone
        contact_order_fault = bound & (
            (contact_event & (self._flight_state != FLIGHT_INBOUND))
            | (contact_event & ~contact_stamp_valid)
        )
        # Incoming/pre-contact net and landing evidence belongs to the incoming
        # ball and is deliberately ignored.  The selected outgoing anchor is
        # the first segment eligible for this shot's net/crossing authority.
        crossing_before_contact = torch.zeros_like(bound)
        crossing_report_fault = bound & (
            report_missing
            | (
                effective_contact
                & effective_crossing_event
                & ~crossing_event_stamp_valid
            )
            | (both_crossings & report_xy_finite & ~both_xy_match)
        )
        net_fault = bound & effective_contact & ~incoming_net_same_batch & (
            duplicate_net
            | clear_without_net_event
            | (effective_net_event & ~net_stamp_valid)
            | (crossing_candidate & ~crossing_order_after_net)
        )
        late_deadline_fault = (
            bound
            & (self._flight_state == FLIGHT_INBOUND)
            & (
                observation_stamp.control_step
                > self._flight_contact_deadline_control_step
            )
        )
        late_horizon_fault = (
            bound
            & effective_contact
            & (
                observation_stamp.control_step
                > self._flight_crossing_horizon_control_step
            )
        )
        actual_nonfinite = bound & (
            nonfinite_flag
            | ~torch.isfinite(previous).all(dim=-1)
            | ~torch.isfinite(current).all(dim=-1)
            | (contact_event & ~contact_geometry_finite)
            | (
                effective_contact
                & effective_report_delivered
                & effective_crossing_event
                & ~report_xy_finite
            )
        )

        protocol_fault = (
            ordinal_fault
            | continuity_fault
            | observation_stamp_fault
            | stamp_regression_fault
            | contact_order_fault
            | crossing_before_contact
            | net_fault
            | late_deadline_fault
            | late_horizon_fault
        )
        contact_anchor_fault = (
            bound
            & contact_event
            & (self._flight_state == FLIGHT_INBOUND)
            & contact_stamp_valid
            & contact_geometry_finite
            & ~contact_anchor_match
        )
        effective_producer_fault = (
            producer_fault | crossing_report_fault | contact_anchor_fault
        )
        policy_nonfinite_crossing = (
            bound
            & effective_contact
            & effective_report_delivered
            & effective_crossing_event
            & crossing_event_stamp_valid
            & ~report_xy_finite
            & ~protocol_fault
            & ~effective_producer_fault
            & ~engine_overflow
        )
        infra_nonfinite = actual_nonfinite & ~policy_nonfinite_crossing
        safe_crossing = (
            bound
            & effective_contact
            & crossing_candidate
            & ~protocol_fault
            & ~actual_nonfinite
            & ~effective_producer_fault
            & ~engine_overflow
        )
        chosen_crossing_xy = torch.where(
            (effective_report_delivered & effective_crossing_event).unsqueeze(-1),
            crossing_xy,
            segment_xy,
        )
        effective_crossing_stamp = PhysicsStampBatch(
            control_step=torch.where(
                effective_report_delivered & effective_crossing_event,
                crossing_stamp.control_step,
                observation_stamp.control_step,
            ),
            physics_substep=torch.where(
                effective_report_delivered & effective_crossing_event,
                crossing_stamp.physics_substep,
                observation_stamp.physics_substep,
            ),
            event_phase=torch.full(
                p, PHASE_LANDING, dtype=torch.int8, device=self.device
            ),
        )

        no_contact_deadline = (
            bound
            & ~effective_contact
            & ~protocol_fault
            & ~infra_nonfinite
            & ~policy_nonfinite_crossing
            & ~effective_producer_fault
            & ~engine_overflow
            & (
                observation_stamp.control_step
                == self._flight_contact_deadline_control_step
            )
        )
        crossing_horizon = (
            bound
            & effective_contact
            & ~safe_crossing
            & ~protocol_fault
            & ~infra_nonfinite
            & ~policy_nonfinite_crossing
            & ~effective_producer_fault
            & ~engine_overflow
            & (
                observation_stamp.control_step
                == self._flight_crossing_horizon_control_step
            )
        )

        settlement_cause = torch.full(
            p, SETTLEMENT_CAUSE_NONE, dtype=torch.int8, device=self.device
        )
        settlement_cause = torch.where(
            safe_crossing,
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_FIRST_CROSSING),
            settlement_cause,
        )
        settlement_cause = torch.where(
            no_contact_deadline,
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_CONTACT_DEADLINE),
            settlement_cause,
        )
        settlement_cause = torch.where(
            crossing_horizon,
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_CROSSING_HORIZON),
            settlement_cause,
        )
        settlement_cause = torch.where(
            policy_nonfinite_crossing | (bound & infra_nonfinite),
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_NONFINITE),
            settlement_cause,
        )
        settlement_cause = torch.where(
            bound & effective_producer_fault,
            torch.full_like(
                settlement_cause, SETTLEMENT_CAUSE_PRODUCER_CONTRACT_FAULT
            ),
            settlement_cause,
        )
        settlement_cause = torch.where(
            bound & engine_overflow,
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_ENGINE_OVERFLOW),
            settlement_cause,
        )
        settlement_cause = torch.where(
            bound & protocol_fault,
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_PROTOCOL_FAULT),
            settlement_cause,
        )
        settle = settlement_cause != SETTLEMENT_CAUSE_NONE

        fault_bits = torch.zeros(p, dtype=torch.int64, device=self.device)
        fault_bits = self._with_fault(fault_bits, observed_nonlive, FAULT_INVALID_OBSERVATION)
        fault_bits = self._with_fault(fault_bits, missing_live, FAULT_UNOBSERVED_LIVE_SLOT)
        fault_bits = self._with_fault(
            fault_bits, observe & live & ~primary_identity_match, FAULT_KEY_BINDING
        )
        fault_bits = self._with_fault(
            fault_bits,
            observe & live & primary_identity_match & ~generation_match,
            FAULT_GENERATION_BINDING,
        )
        fault_bits = self._with_fault(fault_bits, ordinal_fault, FAULT_OBSERVATION_ORDINAL)
        fault_bits = self._with_fault(fault_bits, observation_stamp_fault, FAULT_INVALID_STAMP)
        fault_bits = self._with_fault(fault_bits, stamp_regression_fault, FAULT_STAMP_REGRESSION)
        fault_bits = self._with_fault(fault_bits, contact_order_fault, FAULT_CONTACT_ORDER)
        fault_bits = self._with_fault(
            fault_bits, crossing_before_contact, FAULT_CROSSING_BEFORE_CONTACT
        )
        fault_bits = self._with_fault(fault_bits, crossing_report_fault, FAULT_CROSSING_REPORT)
        fault_bits = self._with_fault(fault_bits, net_fault, FAULT_NET_CONTRACT)
        fault_bits = self._with_fault(fault_bits, continuity_fault, FAULT_FLIGHT_CONTINUITY)
        fault_bits = self._with_fault(fault_bits, infra_nonfinite, FAULT_NONFINITE)
        fault_bits = self._with_fault(
            fault_bits, bound & effective_producer_fault, FAULT_PRODUCER_CONTRACT
        )
        fault_bits = self._with_fault(fault_bits, bound & engine_overflow, FAULT_ENGINE_OVERFLOW)

        self._post_fault_bits.bitwise_or_(fault_bits)
        owner_fault_mask = bound & (fault_bits != 0)
        self._flight_fault_bits.copy_(
            torch.where(
                owner_fault_mask,
                torch.bitwise_or(self._flight_fault_bits, fault_bits),
                self._flight_fault_bits,
            )
        )

        # Apply valid contact/net facts before scoring/copying settlement.
        _masked_fill_(self._flight_contact_valid, valid_contact, 1)
        _masked_copy_(self._flight_contact_ball_center_m, contact_center, valid_contact)
        _masked_copy_(self._flight_outgoing_anchor_m, outgoing_anchor, valid_contact)
        _masked_copy_(
            self._flight_contact_stamp_control,
            contact_stamp.control_step,
            valid_contact,
        )
        _masked_copy_(
            self._flight_contact_stamp_substep,
            contact_stamp.physics_substep,
            valid_contact,
        )
        _masked_fill_(self._flight_state, valid_contact, FLIGHT_OPEN)
        _masked_fill_(self._flight_net_crossed, valid_new_net, 1)
        _masked_copy_(
            self._flight_net_clear, effective_net_clear_report, valid_new_net
        )
        _masked_copy_(
            self._flight_net_stamp_control, net_stamp.control_step, valid_new_net
        )
        _masked_copy_(
            self._flight_net_stamp_substep, net_stamp.physics_substep, valid_new_net
        )

        if action_epoch_direct and defer_action_epoch_control_finalize:
            assert direct_flight_slot is not None
            return self._sample_action_epoch_control_substep(
                bound=bound,
                missing_live=missing_live,
                observe=observe,
                fault_bits=fault_bits,
                settlement_cause=settlement_cause,
                policy_nonfinite_crossing=policy_nonfinite_crossing,
                chosen_crossing_xy=chosen_crossing_xy,
                effective_crossing_stamp=effective_crossing_stamp,
                observation_stamp=observation_stamp,
                observation_ordinal=ordinal,
                current_ball_center_m=current,
                direct_flight_slot=direct_flight_slot,
            )

        policy_settlement = (
            (settlement_cause == SETTLEMENT_CAUSE_FIRST_CROSSING)
            | (settlement_cause == SETTLEMENT_CAUSE_CONTACT_DEADLINE)
            | (settlement_cause == SETTLEMENT_CAUSE_CROSSING_HORIZON)
            | policy_nonfinite_crossing
        )
        crossing_present = (
            (settlement_cause == SETTLEMENT_CAUSE_FIRST_CROSSING)
            | policy_nonfinite_crossing
        )
        crossing_valid = settlement_cause == SETTLEMENT_CAUSE_FIRST_CROSSING
        crossing_nonfinite = policy_nonfinite_crossing
        safe_xy = torch.where(
            crossing_valid.unsqueeze(-1),
            chosen_crossing_xy,
            torch.zeros_like(chosen_crossing_xy),
        )
        direct_score = self._score_action_epoch_policy_subset(
            policy_mask=policy_settlement & self._flight_action_epoch,
            target_xy_m=self._flight_target_xy_m,
            contact_valid=self._flight_contact_valid,
            crossing_present=crossing_present,
            crossing_valid=crossing_valid,
            crossing_nonfinite=crossing_nonfinite,
            crossing_xy_m=safe_xy,
            net_crossed=self._flight_net_crossed,
            net_clear=self._flight_net_clear,
        )
        if action_epoch_direct:
            score = direct_score
        else:
            legacy_score = self._score_policy_subset(
                prefix=self._flight_shape,
                policy_mask=policy_settlement & ~self._flight_action_epoch,
                task_identity_token=self._flight_task_identity_token,
                target_xy_m=self._flight_target_xy_m,
                contact_valid=self._flight_contact_valid,
                crossing_present=crossing_present,
                crossing_valid=crossing_valid,
                crossing_nonfinite=crossing_nonfinite,
                crossing_xy_m=safe_xy,
                net_crossed=self._flight_net_crossed,
                net_clear=self._flight_net_clear,
            )
            choose_direct = self._flight_action_epoch

            def merge_score(name: str) -> torch.Tensor:
                return torch.where(
                    choose_direct,
                    getattr(direct_score, name),
                    getattr(legacy_score, name),
                )

            score = _PolicyScoreGrid(
                drain_fault=merge_score("drain_fault"),
                reason_code=merge_score("reason_code"),
                on_opponent_table=merge_score("on_opponent_table"),
                placement_error_m=merge_score("placement_error_m"),
                broad_kernel=merge_score("broad_kernel"),
                narrow_kernel=merge_score("narrow_kernel"),
                blended_kernel=merge_score("blended_kernel"),
                table_gate=merge_score("table_gate"),
                total=merge_score("total"),
            )
        task_drain_fault = policy_settlement & score.drain_fault
        settlement_cause = torch.where(
            task_drain_fault,
            torch.full_like(settlement_cause, SETTLEMENT_CAUSE_PROTOCOL_FAULT),
            settlement_cause,
        )
        policy_settlement = policy_settlement & ~task_drain_fault
        settle = settlement_cause != SETTLEMENT_CAUSE_NONE
        task_drain_bits = torch.where(
            task_drain_fault,
            torch.full_like(fault_bits, FAULT_TASK_DRAIN),
            torch.zeros_like(fault_bits),
        )
        fault_bits = torch.bitwise_or(fault_bits, task_drain_bits)
        self._post_fault_bits.bitwise_or_(task_drain_bits)
        self._flight_fault_bits.copy_(
            torch.where(
                task_drain_fault,
                torch.bitwise_or(self._flight_fault_bits, task_drain_bits),
                self._flight_fault_bits,
            )
        )

        reservation_ok = (
            self._action_epoch_flight_reservation_ok()
            if action_epoch_direct
            else self._flight_reservation_ok()
        )
        copy_collision = settle & ~reservation_ok
        if_fault = torch.where(
            copy_collision,
            torch.full_like(fault_bits, FAULT_MAILBOX_COPY_COLLISION),
            torch.zeros_like(fault_bits),
        )
        fault_bits = torch.bitwise_or(fault_bits, if_fault)
        self._post_fault_bits.bitwise_or_(if_fault)
        self._flight_fault_bits.copy_(
            torch.where(
                copy_collision,
                torch.bitwise_or(self._flight_fault_bits, if_fault),
                self._flight_fault_bits,
            )
        )
        settle = settle & reservation_ok

        self._copy_settlements(
            settle=settle,
            observation_ordinal=ordinal,
            settlement_cause=settlement_cause,
            policy_settlement=policy_settlement & settle,
            crossing_present=crossing_present & settle,
            crossing_valid=crossing_valid & settle,
            crossing_nonfinite=crossing_nonfinite & settle,
            crossing_xy=safe_xy,
            observation_stamp=observation_stamp,
            crossing_stamp=effective_crossing_stamp,
            score=score,
            action_epoch_direct=action_epoch_direct,
        )

        # The terminal observation is part of the retained flight identity;
        # retirement must not borrow an ordinal from the physical caller.
        update_observation = bound
        _masked_copy_(
            self._flight_observation_ordinal, ordinal, update_observation
        )
        _masked_copy_(
            self._flight_last_ball_center_m, current, update_observation
        )
        _masked_copy_(
            self._flight_last_observation_control,
            observation_stamp.control_step,
            update_observation,
        )
        _masked_copy_(
            self._flight_last_observation_substep,
            observation_stamp.physics_substep,
            update_observation,
        )

        accepted = bound & (fault_bits == 0)
        rejected = missing_live | (observe & ~accepted)
        self._record_fault_events(fault_bits)
        self._increment_mutation()
        if action_epoch_direct:
            if self._prepared_action_epoch_current_settlement_delta is not None:
                raise LandingOutcomeDeviceError(
                    "R06 epoch postphysics current-settlement source was not retired"
                )
            prepared_delta = self._prepare_action_epoch_current_settlement_delta(
                settle=settle,
                observation_stamp=observation_stamp,
                settlement_cause=settlement_cause,
                policy_settlement=policy_settlement,
                crossing_valid=crossing_valid,
                crossing_xy=safe_xy,
                score=score,
            )
            result = ActionEpochR06PostPhysicsResult(
                accepted=accepted.detach().clone(),
                rejected=rejected.detach().clone(),
                fault_bits=fault_bits.detach().clone(),
                settled_mask=settle.detach().clone(),
                settlement_cause=torch.where(
                    settle,
                    settlement_cause,
                    torch.full_like(settlement_cause, SETTLEMENT_CAUSE_NONE),
                ).detach().clone(),
                new_valid_contact_mask=valid_contact.detach().clone(),
                observation_ordinal=ordinal.detach().clone(),
                mutation_version=self._mutation_version.detach().clone(),
                flight_slot=direct_flight_slot.detach().clone(),
            )
            self._action_epoch_post_physics_result = result
            self._prepared_action_epoch_current_settlement_delta = prepared_delta
            self._action_epoch_post_physics_settled_mask = prepared_delta.candidate
            return result
        contact_authority = object.__new__(
            LandingOutcomePostPhysicsContactAuthority
        )
        result = PostPhysicsMutationResult(
            accepted=accepted.detach().clone(),
            rejected=rejected.detach().clone(),
            fault_bits=fault_bits.detach().clone(),
            settled_mask=settle.detach().clone(),
            settlement_cause=torch.where(
                settle,
                settlement_cause,
                torch.full_like(settlement_cause, SETTLEMENT_CAUSE_NONE),
            )
            .detach()
            .clone(),
            flight_slot=self._flight_slot_ids.expand(self._flight_shape)
            .detach()
            .clone(),
            full_key_sha256=self._flight_full_key_sha256.detach().clone(),
            ball_generation=self._flight_ball_generation.detach().clone(),
            mutation_version=self._mutation_version.detach().clone(),
            physical_publication_identity=physical_publication_identity,
            new_valid_contact_mask=valid_contact.detach().clone(),
            selected_contact_stamp=PhysicsStampBatch(
                control_step=contact_stamp.control_step.detach().clone(),
                physics_substep=contact_stamp.physics_substep.detach().clone(),
                event_phase=contact_stamp.event_phase.detach().clone(),
            ),
            contact_authority=contact_authority,
        )
        key = self._key_storage("flight")
        self._latest_post_physics_settlement = _RetainedPostPhysicsSettlement(
            result=result,
            settled_mask=settle.detach().clone(),
            accepted=accepted.detach().clone(),
            rejected=rejected.detach().clone(),
            fault_bits=fault_bits.detach().clone(),
            settlement_cause=result.settlement_cause.detach().clone(),
            flight_slot=result.flight_slot.detach().clone(),
            task_key=DeviceLandingOutcomeKey(
                **{
                    name: getattr(key, name).detach().clone()
                    for name in _KEY_FIELDS
                }
            ),
            full_key_sha256=self._flight_full_key_sha256.detach().clone(),
            ball_generation=self._flight_ball_generation.detach().clone(),
            mailbox_slot=self._flight_mailbox_slot.detach().clone(),
            observation_ordinal=(
                self._flight_observation_ordinal.detach().clone()
            ),
            mutation_version=self._mutation_version.detach().clone(),
        )
        retained_contact_authority = _RetainedPostPhysicsContactAuthority(
                authority=contact_authority,
                publication_identity=physical_publication_identity,
                new_valid_contact_mask=valid_contact.detach().clone(),
                task_key=DeviceLandingOutcomeKey(
                    **{
                        name: getattr(key, name).detach().clone()
                        for name in _KEY_FIELDS
                    }
                ),
                full_key_sha256=(
                    self._flight_full_key_sha256.detach().clone()
                ),
                ball_generation=(
                    self._flight_ball_generation.detach().clone()
                ),
                flight_slot=result.flight_slot.detach().clone(),
                observation_ordinal=(
                    self._flight_observation_ordinal.detach().clone()
                ),
                selected_contact_stamp=PhysicsStampBatch(
                    control_step=contact_stamp.control_step.detach().clone(),
                    physics_substep=(
                        contact_stamp.physics_substep.detach().clone()
                    ),
                    event_phase=contact_stamp.event_phase.detach().clone(),
                ),
                mutation_version=self._mutation_version.detach().clone(),
                consumed=False,
        )
        self._active_post_physics_contact_authority = retained_contact_authority
        _POSTPHYSICS_CONTACT_REGISTRY[id(contact_authority)] = (
            retained_contact_authority
        )
        return result

    def _sample_action_epoch_control_substep(
        self,
        *,
        bound: torch.Tensor,
        missing_live: torch.Tensor,
        observe: torch.Tensor,
        fault_bits: torch.Tensor,
        settlement_cause: torch.Tensor,
        policy_nonfinite_crossing: torch.Tensor,
        chosen_crossing_xy: torch.Tensor,
        effective_crossing_stamp: PhysicsStampBatch,
        observation_stamp: PhysicsStampBatch,
        observation_ordinal: torch.Tensor,
        current_ball_center_m: torch.Tensor,
        direct_flight_slot: torch.Tensor,
    ) -> ActionEpochR06PostPhysicsSample:
        """Advance the causal R06 FSA without scoring or mailbox materialization."""

        if self._action_epoch_control_replay is not None:
            raise LandingOutcomeDeviceError(
                "R06 ActionEpoch postphysics sample crossed unconsumed replay"
            )
        if self._action_epoch_control_substep_count is None:
            self._action_epoch_control_substep_count = 0
        self._action_epoch_control_substep_count += 1

        crossing_valid = settlement_cause.eq(SETTLEMENT_CAUSE_FIRST_CROSSING)
        safe_xy = torch.where(
            crossing_valid.unsqueeze(-1),
            chosen_crossing_xy,
            torch.zeros_like(chosen_crossing_xy),
        )
        settle = settlement_cause.ne(SETTLEMENT_CAUSE_NONE)
        reservation_ok = self._action_epoch_flight_reservation_ok()
        copy_collision = settle & ~reservation_ok
        collision_bits = torch.where(
            copy_collision,
            torch.full_like(fault_bits, FAULT_MAILBOX_COPY_COLLISION),
            torch.zeros_like(fault_bits),
        )
        fault_bits = torch.bitwise_or(fault_bits, collision_bits)
        self._post_fault_bits.bitwise_or_(collision_bits)
        self._flight_fault_bits.copy_(
            torch.where(
                copy_collision,
                torch.bitwise_or(self._flight_fault_bits, collision_bits),
                self._flight_fault_bits,
            )
        )
        settle &= reservation_ok
        new_pending = settle & self._action_epoch_control_pending_cause.eq(
            SETTLEMENT_CAUSE_NONE
        )
        _masked_copy_(
            self._action_epoch_control_pending_cause,
            settlement_cause,
            new_pending,
        )
        _masked_copy_(
            self._action_epoch_control_pending_crossing_kind,
            torch.where(
                crossing_valid,
                torch.full_like(
                    settlement_cause, _CONTROL_CROSSING_VALID
                ),
                torch.where(
                    policy_nonfinite_crossing,
                    torch.full_like(
                        settlement_cause, _CONTROL_CROSSING_NONFINITE
                    ),
                    torch.full_like(
                        settlement_cause, _CONTROL_CROSSING_NONE
                    ),
                ),
            ),
            new_pending,
        )
        _masked_copy_(
            self._action_epoch_control_pending_crossing_xy,
            safe_xy,
            new_pending,
        )
        _masked_copy_(
            self._action_epoch_control_pending_crossing_control,
            effective_crossing_stamp.control_step,
            new_pending,
        )
        _masked_copy_(
            self._action_epoch_control_pending_crossing_substep,
            effective_crossing_stamp.physics_substep,
            new_pending,
        )
        _masked_copy_(
            self._flight_observation_ordinal, observation_ordinal, bound
        )
        _masked_copy_(
            self._flight_last_ball_center_m, current_ball_center_m, bound
        )
        _masked_copy_(
            self._flight_last_observation_control,
            observation_stamp.control_step,
            bound,
        )
        _masked_copy_(
            self._flight_last_observation_substep,
            observation_stamp.physics_substep,
            bound,
        )

        accepted = bound & fault_bits.eq(0)
        rejected = missing_live | (observe & ~accepted)
        self._record_fault_events(fault_bits)
        self._increment_mutation()
        return ActionEpochR06PostPhysicsSample(
            accepted=accepted.detach().clone(),
            rejected=rejected.detach().clone(),
            settled_mask=settle.detach().clone(),
            flight_slot=direct_flight_slot.detach().clone(),
        )

    def finalize_action_ball_full_mdp_epoch_post_physics_control(
        self, *, physics_substeps_per_control: int
    ) -> ActionEpochR06PostPhysicsResult:
        """Score and materialize one sampled ActionEpoch control window."""

        sampled_count = self._action_epoch_control_substep_count
        if (
            type(physics_substeps_per_control) is not int
            or physics_substeps_per_control < 1
            or sampled_count != physics_substeps_per_control
            or self._action_epoch_post_physics_result is not None
            or self._prepared_action_epoch_current_settlement_delta is not None
            or self._action_epoch_control_replay is not None
        ):
            raise LandingOutcomeDeviceError(
                "R06 ActionEpoch control finalize lifetime differs"
            )
        cause = self._action_epoch_control_pending_cause
        pending = cause.ne(SETTLEMENT_CAUSE_NONE)
        crossing_kind = self._action_epoch_control_pending_crossing_kind
        crossing_present = crossing_kind.ne(_CONTROL_CROSSING_NONE)
        crossing_valid = crossing_kind.eq(_CONTROL_CROSSING_VALID)
        crossing_nonfinite = crossing_kind.eq(_CONTROL_CROSSING_NONFINITE)
        policy = pending & (
            cause.eq(SETTLEMENT_CAUSE_FIRST_CROSSING)
            | cause.eq(SETTLEMENT_CAUSE_CONTACT_DEADLINE)
            | cause.eq(SETTLEMENT_CAUSE_CROSSING_HORIZON)
            | crossing_nonfinite
        )
        score = self._score_action_epoch_policy_subset(
            policy_mask=policy,
            target_xy_m=self._flight_target_xy_m,
            contact_valid=self._flight_contact_valid,
            crossing_present=crossing_present,
            crossing_valid=crossing_valid,
            crossing_nonfinite=crossing_nonfinite,
            crossing_xy_m=self._action_epoch_control_pending_crossing_xy,
            net_crossed=self._flight_net_crossed,
            net_clear=self._flight_net_clear,
        )
        # The typed ActionEpoch scorer has no task-drain branch.  Keep that
        # first-principles contract explicit: a later scorer change may not
        # silently move physical-stop authority from the substep sampler.
        if score.drain_fault.shape != self._flight_shape:
            raise LandingOutcomeDeviceError(
                "R06 ActionEpoch control score drain ABI differs"
            )
        observation_stamp = PhysicsStampBatch(
            control_step=self._flight_last_observation_control,
            physics_substep=self._flight_last_observation_substep,
            event_phase=torch.full(
                self._flight_shape,
                PHASE_LANDING,
                dtype=torch.int8,
                device=self.device,
            ),
        )
        crossing_stamp = PhysicsStampBatch(
            control_step=self._action_epoch_control_pending_crossing_control,
            physics_substep=self._action_epoch_control_pending_crossing_substep,
            event_phase=torch.full(
                self._flight_shape,
                PHASE_LANDING,
                dtype=torch.int8,
                device=self.device,
            ),
        )
        self._copy_settlements(
            settle=pending,
            observation_ordinal=self._flight_observation_ordinal,
            settlement_cause=cause,
            policy_settlement=policy,
            crossing_present=crossing_present,
            crossing_valid=crossing_valid,
            crossing_nonfinite=crossing_nonfinite,
            crossing_xy=self._action_epoch_control_pending_crossing_xy,
            observation_stamp=observation_stamp,
            crossing_stamp=crossing_stamp,
            score=score,
            action_epoch_direct=True,
        )
        prepared = self._prepare_action_epoch_current_settlement_delta(
            settle=pending,
            observation_stamp=observation_stamp,
            settlement_cause=cause,
            policy_settlement=policy,
            crossing_valid=crossing_valid,
            crossing_xy=self._action_epoch_control_pending_crossing_xy,
            score=score,
        )
        fault_bits = self._flight_fault_bits
        result = ActionEpochR06PostPhysicsResult(
            accepted=(pending & fault_bits.eq(0)).detach().clone(),
            rejected=(pending & fault_bits.ne(0)).detach().clone(),
            fault_bits=fault_bits.detach().clone(),
            settled_mask=pending.detach().clone(),
            settlement_cause=torch.where(
                pending,
                cause,
                torch.full_like(
                    cause,
                    SETTLEMENT_CAUSE_NONE,
                ),
            ).detach().clone(),
            new_valid_contact_mask=torch.zeros_like(pending),
            observation_ordinal=self._flight_observation_ordinal.detach().clone(),
            mutation_version=self._mutation_version.detach().clone(),
            flight_slot=self._flight_slot_ids.expand(self._flight_shape)
            .detach().clone(),
        )
        self._action_epoch_post_physics_result = result
        self._prepared_action_epoch_current_settlement_delta = prepared
        self._action_epoch_post_physics_settled_mask = prepared.candidate
        return result

    def retire_action_ball_full_mdp_epoch_post_physics(
        self,
    ) -> ActionEpochR06RetireResult:
        """Retire exactly the private typed settlement from the preceding pull."""

        result = self._action_epoch_post_physics_result
        settled = self._action_epoch_post_physics_settled_mask
        if type(result) is not ActionEpochR06PostPhysicsResult or settled is None:
            raise LandingOutcomeDeviceError(
                "R06 epoch retire requires the exact preceding direct publication"
            )
        if settled is not self._action_epoch_post_physics_settled_mask:
            raise LandingOutcomeDeviceError("R06 epoch retire state is inconsistent")

        safe_mailbox = self._flight_mailbox_slot.clamp(
            0, self.mailbox_capacity - 1
        )
        mailbox_target = (
            settled.unsqueeze(-1)
            & (
                safe_mailbox.unsqueeze(-1)
                == self._mailbox_slot_ids.unsqueeze(1)
            )
        ).any(dim=1)
        retire_valid = (
            settled
            & self._flight_action_epoch
            & self._flight_state.eq(FLIGHT_SETTLED_RETAINED)
            & ~self._flight_physical_retired
            & self._action_epoch_settled_mailbox_ok()
        )
        # Physical is the cross-owner consumer of both this settlement and the
        # actual retire result.  It masks any mismatch from scene writes and
        # reports one named control-boundary fault.  An asynchronous assertion
        # here used to continue mutating R06 after poisoning CUDA, so the next
        # unrelated PhysX call carried an untraceable device-side assert.
        mailbox_retired = mailbox_target & torch.gather(
            retire_valid, 1, self._mailbox_reserved_flight_slot.clamp(
                0, self.flight_slot_capacity - 1
            )
        )

        _masked_fill_(self._flight_state, retire_valid, FLIGHT_EMPTY)
        _masked_fill_(self._flight_physical_retired, retire_valid, 1)
        _masked_fill_(self._flight_action_epoch, retire_valid, 0)
        _masked_fill_(self._mailbox_physical_retired, mailbox_retired, 1)
        self._retired_total.add_(retire_valid.to(torch.int64).sum())
        self._shared_normal_retire_total.add_(retire_valid.to(torch.int64).sum())
        self._mutation_version.add_(retire_valid.any().to(torch.int64))

        retire_result = ActionEpochR06RetireResult(
            retired_mask=retire_valid.detach().clone(),
            mailbox_retired_mask=mailbox_retired.detach().clone(),
            flight_slot=result.flight_slot.detach().clone(),
            mutation_version=self._mutation_version.detach().clone(),
        )
        if self._action_epoch_control_substep_count is not None:
            prepared = self._prepared_action_epoch_current_settlement_delta
            if type(prepared) is not _ActionEpochOutcomeCandidateGrid:
                raise LandingOutcomeDeviceError(
                    "R06 ActionEpoch control outcome source differs"
                )
            self._action_epoch_control_replay = (
                _with_action_epoch_candidate(prepared, retire_valid)
            )
            self._action_epoch_control_replay_substep = (
                self._flight_last_observation_substep.detach().clone()
            )
            self._action_epoch_control_outcome_next_index = 0
            self._action_epoch_control_pending_cause.fill_(
                SETTLEMENT_CAUSE_NONE
            )
            self._action_epoch_control_pending_crossing_kind.fill_(
                _CONTROL_CROSSING_NONE
            )
            self._action_epoch_control_pending_crossing_xy.zero_()
            self._action_epoch_control_pending_crossing_control.fill_(-1)
            self._action_epoch_control_pending_crossing_substep.fill_(-1)
        else:
            self._mint_action_epoch_current_settlement_delta(retire_valid)
        self._action_epoch_post_physics_result = None
        self._action_epoch_post_physics_settled_mask = None
        self._prepared_action_epoch_current_settlement_delta = None
        return retire_result

    def consume_owned_post_physics_contact_authority(
        self,
        authority: LandingOutcomePostPhysicsContactAuthority,
        *,
        expected_publication_identity: object,
    ) -> LandingOutcomePostPhysicsContactAuthorityView:
        """Consume one exact causal contact publication without host sync."""

        self._require_operable(
            allow_pending_post_physics_settlement=True,
            allow_pending_post_physics_contact_authority=True,
        )
        retained = self._active_post_physics_contact_authority
        if (
            type(authority) is not LandingOutcomePostPhysicsContactAuthority
            or retained is None
            or authority is not retained.authority
            or retained.consumed
            or expected_publication_identity is not retained.publication_identity
        ):
            raise LandingOutcomeDeviceError(
                "post-physics contact authority is stale, foreign, replayed, or publication-swapped"
            )
        view = LandingOutcomePostPhysicsContactAuthorityView(
            publication_identity=retained.publication_identity,
            new_valid_contact_mask=(
                retained.new_valid_contact_mask.detach().clone()
            ),
            task_key=DeviceLandingOutcomeKey(
                **{
                    name: getattr(retained.task_key, name).detach().clone()
                    for name in _KEY_FIELDS
                }
            ),
            full_key_sha256=retained.full_key_sha256.detach().clone(),
            ball_generation=retained.ball_generation.detach().clone(),
            flight_slot=retained.flight_slot.detach().clone(),
            observation_ordinal=(
                retained.observation_ordinal.detach().clone()
            ),
            selected_contact_stamp=PhysicsStampBatch(
                control_step=(
                    retained.selected_contact_stamp.control_step.detach().clone()
                ),
                physics_substep=(
                    retained.selected_contact_stamp.physics_substep.detach().clone()
                ),
                event_phase=(
                    retained.selected_contact_stamp.event_phase.detach().clone()
                ),
            ),
            mutation_version=retained.mutation_version.detach().clone(),
        )
        _POSTPHYSICS_CONTACT_REGISTRY.pop(id(authority), None)
        self._active_post_physics_contact_authority = None
        return view

    def _with_fault(
        self, bits: torch.Tensor, mask: torch.Tensor, value: int
    ) -> torch.Tensor:
        return torch.where(
            mask,
            torch.bitwise_or(bits, torch.full_like(bits, value)),
            bits,
        )

    def _score_policy_subset(
        self,
        *,
        prefix: tuple[int, int],
        policy_mask: torch.Tensor,
        task_identity_token: torch.Tensor,
        target_xy_m: torch.Tensor,
        contact_valid: torch.Tensor,
        crossing_present: torch.Tensor,
        crossing_valid: torch.Tensor,
        crossing_nonfinite: torch.Tensor,
        crossing_xy_m: torch.Tensor,
        net_crossed: torch.Tensor,
        net_clear: torch.Tensor,
    ) -> _PolicyScoreGrid:
        """Call C04 on policy-valid rows only, then scatter without host sync."""

        total = prefix[0] * prefix[1]
        selected = policy_mask.reshape(total)
        tokens = task_identity_token.reshape(total, TOKEN_BYTES)[selected]
        targets = target_xy_m.reshape(total, 2)[selected]
        score = score_landing_placement_torch(
            self.profile,
            frame_id=self.profile.frame_id,
            profile_sha256=self.profile.canonical_sha256,
            expected_task_identity_token=tokens,
            facts_task_identity_token=tokens,
            expected_target_xy_m=targets,
            facts_target_xy_m=targets,
            contact_valid=contact_valid.reshape(total)[selected],
            first_plane_crossing_present=crossing_present.reshape(total)[selected],
            first_plane_crossing_valid=crossing_valid.reshape(total)[selected],
            first_plane_crossing_nonfinite=crossing_nonfinite.reshape(total)[selected],
            first_plane_crossing_xy_m=crossing_xy_m.reshape(total, 2)[selected],
            ball_center_net_crossed=net_crossed.reshape(total)[selected],
            ball_center_net_clear=net_clear.reshape(total)[selected],
        )

        def scatter(value: torch.Tensor, fill: int | float | bool) -> torch.Tensor:
            destination = torch.full(
                (total,), fill, dtype=value.dtype, device=self.device
            )
            destination.masked_scatter_(selected, value)
            return destination.reshape(prefix)

        return _PolicyScoreGrid(
            drain_fault=scatter(score.drain_fault, False),
            reason_code=scatter(score.reason_code, CANONICAL_REASON_NOT_SCORED),
            on_opponent_table=scatter(score.on_opponent_table, False),
            placement_error_m=scatter(score.placement_error_m, 0.0),
            broad_kernel=scatter(score.broad_kernel, 0.0),
            narrow_kernel=scatter(score.narrow_kernel, 0.0),
            blended_kernel=scatter(score.blended_kernel, 0.0),
            table_gate=scatter(score.table_gate, 0.0),
            total=scatter(score.total, 0.0),
        )

    def _score_action_epoch_policy_subset(
        self,
        *,
        policy_mask: torch.Tensor,
        target_xy_m: torch.Tensor,
        contact_valid: torch.Tensor,
        crossing_present: torch.Tensor,
        crossing_valid: torch.Tensor,
        crossing_nonfinite: torch.Tensor,
        crossing_xy_m: torch.Tensor,
        net_crossed: torch.Tensor,
        net_clear: torch.Tensor,
    ) -> _PolicyScoreGrid:
        """Score trusted typed-identity rows without fabricating a SHA token."""

        total = policy_mask.numel()
        prefix = policy_mask.shape
        selected = policy_mask.reshape(total)
        targets = target_xy_m.reshape(total, 2)[selected]
        crossings = crossing_xy_m.reshape(total, 2)[selected]
        finite = torch.isfinite(crossings).all(dim=-1)
        nonfinite = crossing_nonfinite.reshape(total)[selected] | ~finite
        present = crossing_present.reshape(total)[selected]
        first = crossing_valid.reshape(total)[selected] & present & ~nonfinite
        contact = contact_valid.reshape(total)[selected]
        crossed = net_crossed.reshape(total)[selected]
        clear = net_clear.reshape(total)[selected]
        safe_xy = torch.where(finite.unsqueeze(1), crossings, targets)
        x = safe_xy[:, 0]
        y = safe_xy[:, 1]
        opponent = first & x.gt(self.profile.net_x_m)
        on_table = (
            opponent
            & x.ge(self.profile.opponent_table_x_min_m)
            & x.le(self.profile.opponent_table_x_max_m)
            & y.ge(self.profile.table_y_min_m)
            & y.le(self.profile.table_y_max_m)
        )
        error = torch.linalg.vector_norm(safe_xy - targets, dim=-1)
        broad = 1.0 / (1.0 + torch.square(error / self.profile.sigma_broad_m))
        narrow = torch.exp(-torch.square(error / self.profile.sigma_narrow_m))
        blended = self.profile.alpha_broad * broad + (
            1.0 - self.profile.alpha_broad
        ) * narrow
        kernel_valid = first
        zero = torch.zeros_like(error)
        error = torch.where(kernel_valid, error, zero)
        broad = torch.where(kernel_valid, broad, zero)
        narrow = torch.where(kernel_valid, narrow, zero)
        blended = torch.where(kernel_valid, blended, zero)
        eligible = contact & first & crossed & clear & opponent
        gate = torch.where(
            eligible,
            torch.where(
                on_table,
                torch.full_like(error, self.profile.on_table_gate),
                torch.full_like(error, self.profile.off_table_gate),
            ),
            zero,
        )
        score = gate * blended
        reason = torch.full(
            error.shape,
            REASON_TO_CODE["scored_off_table"],
            dtype=torch.int64,
            device=self.device,
        )
        reason = torch.where(
            on_table,
            torch.full_like(reason, REASON_TO_CODE["scored_on_table"]),
            reason,
        )
        reason = torch.where(
            ~opponent,
            torch.full_like(reason, REASON_TO_CODE["not_opponent_bound"]),
            reason,
        )
        reason = torch.where(
            ~clear,
            torch.full_like(reason, REASON_TO_CODE["net_not_clear"]),
            reason,
        )
        reason = torch.where(
            ~crossed,
            torch.full_like(reason, REASON_TO_CODE["net_not_crossed"]),
            reason,
        )
        reason = torch.where(
            ~first,
            torch.full_like(reason, REASON_TO_CODE["no_crossing"]),
            reason,
        )
        reason = torch.where(
            nonfinite,
            torch.full_like(reason, REASON_TO_CODE["nonfinite"]),
            reason,
        )
        reason = torch.where(
            ~contact,
            torch.full_like(reason, REASON_TO_CODE["no_contact"]),
            reason,
        )

        def scatter(value: torch.Tensor, fill: int | float | bool) -> torch.Tensor:
            destination = torch.full(
                (total,), fill, dtype=value.dtype, device=self.device
            )
            destination.masked_scatter_(selected, value)
            return destination.reshape(prefix)

        return _PolicyScoreGrid(
            drain_fault=torch.zeros(
                prefix, dtype=torch.bool, device=self.device
            ),
            reason_code=scatter(reason, CANONICAL_REASON_NOT_SCORED),
            on_opponent_table=scatter(on_table, False),
            placement_error_m=scatter(error, 0.0),
            broad_kernel=scatter(broad, 0.0),
            narrow_kernel=scatter(narrow, 0.0),
            blended_kernel=scatter(blended, 0.0),
            table_gate=scatter(gate, 0.0),
            total=scatter(score, 0.0),
        )

    def _flight_reservation_ok(self) -> torch.Tensor:
        safe = self._flight_mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        reserved = torch.gather(self._mailbox_reserved, 1, safe)
        state = torch.gather(self._mailbox_state.to(torch.int64), 1, safe)
        generation = torch.gather(self._mailbox_reservation_generation, 1, safe)
        flight_slot = torch.gather(self._mailbox_reserved_flight_slot, 1, safe)
        token = torch.gather(
            self._mailbox_reservation_token,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        legacy_identity = (
            ~self._flight_action_epoch
            & torch.eq(token, self._flight_full_key_sha256).all(dim=-1)
        )
        typed_key_match = _row_identity.action_epoch_shot_key_equal(
            self._gather_mailbox_action_epoch_shot_key(safe),
            self._flight_action_epoch_shot_key(),
        )
        direct_identity = (
            self._flight_action_epoch
            & torch.gather(self._mailbox_action_epoch, 1, safe)
            & torch.gather(self._mailbox_publication_ordinal, 1, safe).eq(
                self._flight_publication_ordinal
            )
            & typed_key_match
        )
        return (
            (self._flight_mailbox_slot >= 0)
            & (self._flight_mailbox_slot < self.mailbox_capacity)
            & reserved
            & (state == MAILBOX_EMPTY)
            & (generation == self._flight_ball_generation)
            & (flight_slot == self._flight_slot_ids.expand(self._flight_shape))
            & (legacy_identity | direct_identity)
        )

    def _action_epoch_flight_reservation_ok(self) -> torch.Tensor:
        """Join direct flight/mailbox reservations using typed identities only."""

        safe = self._flight_mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        typed_key_match = _row_identity.action_epoch_shot_key_equal(
            self._gather_mailbox_action_epoch_shot_key(safe),
            self._flight_action_epoch_shot_key(),
        )
        return (
            self._flight_action_epoch
            & self._flight_mailbox_slot.ge(0)
            & self._flight_mailbox_slot.lt(self.mailbox_capacity)
            & torch.gather(self._mailbox_reserved, 1, safe)
            & torch.gather(self._mailbox_action_epoch, 1, safe)
            & torch.gather(self._mailbox_state.to(torch.int64), 1, safe).eq(
                MAILBOX_EMPTY
            )
            & torch.gather(
                self._mailbox_reservation_generation, 1, safe
            ).eq(self._flight_ball_generation)
            & torch.gather(self._mailbox_reserved_flight_slot, 1, safe).eq(
                self._flight_slot_ids.expand(self._flight_shape)
            )
            & torch.gather(self._mailbox_publication_ordinal, 1, safe).eq(
                self._flight_publication_ordinal
            )
            & typed_key_match
        )

    def _action_epoch_settled_mailbox_ok(self) -> torch.Tensor:
        """Join a settled direct flight to its typed mailbox after-image."""

        safe = self._flight_mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        typed_key_match = _row_identity.action_epoch_shot_key_equal(
            self._gather_mailbox_action_epoch_shot_key(safe),
            self._flight_action_epoch_shot_key(),
        )
        return (
            self._flight_action_epoch
            & self._flight_mailbox_slot.ge(0)
            & self._flight_mailbox_slot.lt(self.mailbox_capacity)
            & torch.gather(self._mailbox_reserved, 1, safe)
            & torch.gather(self._mailbox_action_epoch, 1, safe)
            & torch.gather(self._mailbox_state.to(torch.int64), 1, safe).ne(
                MAILBOX_EMPTY
            )
            & torch.gather(self._mailbox_ball_generation, 1, safe).eq(
                self._flight_ball_generation
            )
            & torch.gather(self._mailbox_reserved_flight_slot, 1, safe).eq(
                self._flight_slot_ids.expand(self._flight_shape)
            )
            & torch.gather(self._mailbox_publication_ordinal, 1, safe).eq(
                self._flight_publication_ordinal
            )
            & typed_key_match
        )

    def _copy_settlements(
        self,
        *,
        settle: torch.Tensor,
        observation_ordinal: torch.Tensor,
        settlement_cause: torch.Tensor,
        policy_settlement: torch.Tensor,
        crossing_present: torch.Tensor,
        crossing_valid: torch.Tensor,
        crossing_nonfinite: torch.Tensor,
        crossing_xy: torch.Tensor,
        observation_stamp: PhysicsStampBatch,
        crossing_stamp: PhysicsStampBatch,
        score: object,
        action_epoch_direct: bool = False,
    ) -> None:
        # Every source below is owned by a flight/score/constant plane while
        # every destination is a mailbox plane.  The distinct-storage helper
        # makes that cross-owner premise executable and fuses where+copy into
        # one device operation without a per-substep host verdict.
        masked_copy = _masked_copy_distinct_
        score_fields = {
            "canonical_reason_code": score.reason_code.reshape(self._flight_shape),
            "on_opponent_table": score.on_opponent_table.reshape(self._flight_shape),
            "placement_error_m": score.placement_error_m.reshape(self._flight_shape),
            "broad_kernel": score.broad_kernel.reshape(self._flight_shape),
            "narrow_kernel": score.narrow_kernel.reshape(self._flight_shape),
            "blended_kernel": score.blended_kernel.reshape(self._flight_shape),
            "table_gate": score.table_gate.reshape(self._flight_shape),
            "canonical_total": score.total.reshape(self._flight_shape),
        }
        for flight_index in range(self.flight_slot_capacity):
            selected = settle[:, flight_index]
            mailbox_slot = self._flight_mailbox_slot[:, flight_index].clamp(
                0, self.mailbox_capacity - 1
            )
            target = selected.unsqueeze(1) & (
                self._mailbox_slot_ids == mailbox_slot.unsqueeze(1)
            )
            if not action_epoch_direct:
                for name in _INT_KEY_FIELDS:
                    source = self._flight_key_ints[name][
                        :, flight_index
                    ].unsqueeze(1).expand(self._mailbox_shape)
                    masked_copy(self._mailbox_key_ints[name], source, target)
                for name in _DIGEST_KEY_FIELDS:
                    source = self._flight_key_digests[name][
                        :, flight_index
                    ].unsqueeze(1).expand(
                        self._mailbox_shape + (TOKEN_BYTES,)
                    )
                    masked_copy(self._mailbox_key_digests[name], source, target)
            for destination, source_grid in (
                (self._mailbox_action_uid, self._flight_action_uid),
                (self._mailbox_action_slot, self._flight_action_slot),
                (self._mailbox_reset_generation, self._flight_reset_generation),
                (self._mailbox_shot_index, self._flight_shot_index),
                (self._mailbox_task_identity, self._flight_task_identity),
                (self._mailbox_outcome_identity, self._flight_outcome_identity),
                (self._mailbox_ball_identity, self._flight_ball_identity),
                (
                    self._mailbox_publication_ordinal,
                    self._flight_publication_ordinal,
                ),
            ):
                source = source_grid[:, flight_index].unsqueeze(1).expand(
                    self._mailbox_shape
                )
                masked_copy(destination, source, target)
            if not action_epoch_direct:
                for destination, source in (
                    (
                        self._mailbox_full_key_sha256,
                        self._flight_full_key_sha256[:, flight_index],
                    ),
                    (
                        self._mailbox_full_key_receipt_sha256,
                        self._flight_full_key_receipt_sha256[:, flight_index],
                    ),
                    (
                        self._mailbox_committed_reveal_sha256,
                        self._flight_committed_reveal_sha256[:, flight_index],
                    ),
                    (
                        self._mailbox_install_receipt_sha256,
                        self._flight_install_receipt_sha256[:, flight_index],
                    ),
                    (
                        self._mailbox_task_identity_token,
                        self._flight_task_identity_token[:, flight_index],
                    ),
                ):
                    masked_copy(
                        destination,
                        source.unsqueeze(1).expand(
                            self._mailbox_shape + (TOKEN_BYTES,)
                        ),
                        target,
                    )
            masked_copy(
                self._mailbox_target_xy_m,
                self._flight_target_xy_m[:, flight_index].unsqueeze(1).expand(
                    self._mailbox_shape + (2,)
                ),
                target,
            )
            scalar_copies = (
                (self._mailbox_ball_generation, self._flight_ball_generation[:, flight_index]),
                (
                    self._mailbox_observation_ordinal,
                    observation_ordinal[:, flight_index],
                ),
                (self._mailbox_reserved_flight_slot, torch.full(
                    (self.num_envs,), flight_index, dtype=torch.int64, device=self.device
                )),
                (self._mailbox_settlement_cause, settlement_cause[:, flight_index]),
                (self._mailbox_contact_valid, self._flight_contact_valid[:, flight_index]),
                (self._mailbox_crossing_present, crossing_present[:, flight_index]),
                (self._mailbox_crossing_valid, crossing_valid[:, flight_index]),
                (
                    self._mailbox_crossing_nonfinite,
                    crossing_nonfinite[:, flight_index],
                ),
                (self._mailbox_net_crossed, self._flight_net_crossed[:, flight_index]),
                (self._mailbox_net_clear, self._flight_net_clear[:, flight_index]),
                (self._mailbox_net_stamp_control, self._flight_net_stamp_control[:, flight_index]),
                (self._mailbox_net_stamp_substep, self._flight_net_stamp_substep[:, flight_index]),
                (self._mailbox_source_control_step, self._flight_contact_stamp_control[:, flight_index]),
                (self._mailbox_source_physics_substep, self._flight_contact_stamp_substep[:, flight_index]),
            )
            for destination, source in scalar_copies:
                masked_copy(
                    destination, source.unsqueeze(1).expand(self._mailbox_shape), target
                )
            masked_copy(
                self._mailbox_contact_ball_center_m,
                self._flight_contact_ball_center_m[:, flight_index].unsqueeze(1).expand(
                    self._mailbox_shape + (3,)
                ),
                target,
            )
            masked_copy(
                self._mailbox_crossing_xy_m,
                crossing_xy[:, flight_index].unsqueeze(1).expand(
                    self._mailbox_shape + (2,)
                ),
                target,
            )
            crossing_control = torch.where(
                crossing_present[:, flight_index],
                crossing_stamp.control_step[:, flight_index],
                torch.full(
                    (self.num_envs,), -1, dtype=torch.int64, device=self.device
                ),
            )
            crossing_substep = torch.where(
                crossing_present[:, flight_index],
                crossing_stamp.physics_substep[:, flight_index],
                torch.full(
                    (self.num_envs,), -1, dtype=torch.int32, device=self.device
                ),
            )
            masked_copy(
                self._mailbox_crossing_stamp_control,
                crossing_control.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            masked_copy(
                self._mailbox_crossing_stamp_substep,
                crossing_substep.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            policy = policy_settlement[:, flight_index]
            policy_target = target & policy.unsqueeze(1)
            masked_copy(
                self._mailbox_policy_eligible,
                policy.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            canonical_reason = torch.where(
                policy,
                score_fields["canonical_reason_code"][:, flight_index],
                torch.full(
                    (self.num_envs,),
                    CANONICAL_REASON_NOT_SCORED,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
            masked_copy(
                self._mailbox_canonical_reason_code,
                canonical_reason.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            for name, destination in (
                ("on_opponent_table", self._mailbox_on_opponent_table),
                ("placement_error_m", self._mailbox_placement_error_m),
                ("broad_kernel", self._mailbox_broad_kernel),
                ("narrow_kernel", self._mailbox_narrow_kernel),
                ("blended_kernel", self._mailbox_blended_kernel),
                ("table_gate", self._mailbox_table_gate),
                ("canonical_total", self._mailbox_canonical_total),
            ):
                values = score_fields[name][:, flight_index]
                values = torch.where(policy, values, torch.zeros_like(values))
                masked_copy(
                    destination,
                    values.unsqueeze(1).expand(self._mailbox_shape),
                    target,
                )
            common = (
                policy
                & self._flight_contact_valid[:, flight_index]
                & crossing_valid[:, flight_index]
                & self._flight_net_crossed[:, flight_index]
                & self._flight_net_clear[:, flight_index]
                & score_fields["on_opponent_table"][:, flight_index]
            )
            masked_copy(
                self._mailbox_common_on_table,
                common.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            settlement_control = observation_stamp.control_step[:, flight_index]
            settlement_substep = observation_stamp.physics_substep[:, flight_index]
            masked_copy(
                self._mailbox_settlement_control_step,
                settlement_control.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            masked_copy(
                self._mailbox_settlement_physics_substep,
                settlement_substep.unsqueeze(1).expand(self._mailbox_shape),
                target,
            )
            masked_copy(
                self._mailbox_fault_bits,
                self._flight_fault_bits[:, flight_index].unsqueeze(1).expand(
                    self._mailbox_shape
                ),
                target,
            )
            for destination, value in (
                (self._mailbox_physical_retired, 0),
                (self._mailbox_paid_mask, 0),
            ):
                _masked_fill_(destination, target, value)
            _masked_fill_(self._mailbox_view_epoch, target, -1)
            _masked_fill_(self._mailbox_payment_epoch, target, -1)
            _masked_fill_(self._mailbox_payment_values, target, 0.0)
            _masked_fill_(
                self._mailbox_placement_treatment_gain,
                target,
                self._placement_treatment_gain,
            )
            _masked_fill_(
                self._mailbox_c10_family_code,
                target,
                self._c10_family_code,
            )
            if not action_epoch_direct:
                masked_copy(
                    self._mailbox_c10_projection_sha256,
                    self._c10_projection_token.reshape(1, 1, TOKEN_BYTES).expand(
                        self._mailbox_shape + (TOKEN_BYTES,)
                    ),
                    target,
                )
            _masked_fill_(self._mailbox_consumer_blocked, target, 0)
            masked_copy(
                self._mailbox_action_epoch,
                self._flight_action_epoch[:, flight_index]
                .unsqueeze(1)
                .expand(self._mailbox_shape),
                target,
            )
            _masked_fill_(self._mailbox_history_valid, target, 1)
            _masked_fill_(self._mailbox_state, target, MAILBOX_SETTLED_UNPAID)
            _masked_fill_(
                self._flight_state[:, flight_index], selected, FLIGHT_SETTLED_RETAINED
            )
        self._settled_total.add_(settle.to(torch.int64).sum())

    def prepare_physical_retire(
        self,
        settlement_result: PostPhysicsMutationResult,
        retire_mask: torch.Tensor,
    ) -> PreparedPhysicalRetire:
        """Prebuild one atomic ``[N,K]`` ledger after-image with zero D2H.

        The post-physics result is an exact owner-retained capability.  The
        caller supplies no slot, key, generation, settlement root or version.
        A subset/superset mask rejects the entire requested/omitted union.
        """

        self._require_operable(
            allow_pending_post_physics_settlement=True,
        )
        if self._physical_park_token_authority is None:
            raise LandingOutcomeDeviceError(
                "physical park token authority is not bound"
            )
        retained = self._latest_post_physics_settlement
        if (
            type(settlement_result) is not PostPhysicsMutationResult
            or retained is None
            or settlement_result is not retained.result
            or retire_mask is not settlement_result.settled_mask
        ):
            raise LandingOutcomeDeviceError(
                "physical retire requires the exact latest result/mask capability"
            )
        requested = self._tensor(
            retire_mask,
            name="retire_mask",
            shape=self._flight_shape,
            dtype=torch.bool,
        )
        result_accepted = self._tensor(
            settlement_result.accepted,
            name="settlement_result.accepted",
            shape=self._flight_shape,
            dtype=torch.bool,
        )
        result_rejected = self._tensor(
            settlement_result.rejected,
            name="settlement_result.rejected",
            shape=self._flight_shape,
            dtype=torch.bool,
        )
        result_fault_bits = self._tensor(
            settlement_result.fault_bits,
            name="settlement_result.fault_bits",
            shape=self._flight_shape,
            dtype=torch.int64,
        )
        result_settlement_cause = self._tensor(
            settlement_result.settlement_cause,
            name="settlement_result.settlement_cause",
            shape=self._flight_shape,
            dtype=torch.int8,
        )
        result_flight_slot = self._tensor(
            settlement_result.flight_slot,
            name="settlement_result.flight_slot",
            shape=self._flight_shape,
            dtype=torch.int64,
        )
        result_full_key = self._tensor(
            settlement_result.full_key_sha256,
            name="settlement_result.full_key_sha256",
            shape=self._flight_shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
        )
        result_generation = self._tensor(
            settlement_result.ball_generation,
            name="settlement_result.ball_generation",
            shape=self._flight_shape,
            dtype=torch.int64,
        )
        result_mutation_version = self._tensor(
            settlement_result.mutation_version,
            name="settlement_result.mutation_version",
            shape=(),
            dtype=torch.int64,
        )
        expected = retained.settled_mask
        result_content_match = (
            torch.eq(requested, expected).all()
            & torch.eq(result_accepted, retained.accepted).all()
            & torch.eq(result_rejected, retained.rejected).all()
            & torch.eq(result_fault_bits, retained.fault_bits).all()
            & torch.eq(
                result_settlement_cause,
                retained.settlement_cause,
            ).all()
            & torch.eq(result_flight_slot, retained.flight_slot).all()
            & torch.eq(result_full_key, retained.full_key_sha256).all()
            & torch.eq(result_generation, retained.ball_generation).all()
            & torch.eq(
                result_mutation_version,
                retained.mutation_version,
            ).all()
        )
        attempted = requested | expected | ~result_content_match
        exact_mask = result_content_match

        mailbox_slot = self._flight_mailbox_slot
        mailbox_in_range = (mailbox_slot >= 0) & (
            mailbox_slot < self.mailbox_capacity
        )
        safe_mailbox = mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        mailbox_state = torch.gather(
            self._mailbox_state.to(torch.int64), 1, safe_mailbox
        )
        mailbox_generation = torch.gather(
            self._mailbox_ball_generation, 1, safe_mailbox
        )
        mailbox_retired = torch.gather(
            self._mailbox_physical_retired, 1, safe_mailbox
        )
        mailbox_full_key = torch.gather(
            self._mailbox_full_key_sha256,
            1,
            safe_mailbox.unsqueeze(-1).expand(
                self._flight_shape + (TOKEN_BYTES,)
            ),
        )
        current_key = self._key_storage("flight")
        retained_key = retained.task_key
        identity = torch.eq(
            self._flight_full_key_sha256,
            retained.full_key_sha256,
        ).all(dim=-1)
        identity = identity & (
            self._flight_ball_generation == retained.ball_generation
        )
        identity = identity & (mailbox_slot == retained.mailbox_slot)
        identity = identity & (
            self._flight_observation_ordinal == retained.observation_ordinal
        )
        for name in _INT_KEY_FIELDS:
            identity = identity & torch.eq(
                getattr(current_key, name), getattr(retained_key, name)
            )
        for name in _DIGEST_KEY_FIELDS:
            identity = identity & torch.eq(
                getattr(current_key, name), getattr(retained_key, name)
            ).all(dim=-1)
        mailbox_binding_valid = (
            mailbox_in_range
            & (mailbox_state != MAILBOX_EMPTY)
            & (mailbox_generation == self._flight_ball_generation)
            & torch.eq(
                mailbox_full_key, self._flight_full_key_sha256
            ).all(dim=-1)
        )
        identity_row_valid = (
            (self._flight_state == FLIGHT_SETTLED_RETAINED)
            & mailbox_binding_valid
            & identity
        )
        normal_candidate = (
            expected
            & identity_row_valid
            & retained.accepted
            & ~retained.rejected
            & (retained.fault_bits == 0)
            & ~self._flight_physical_retired
            & ~mailbox_retired
        )
        accepted = expected & exact_mask
        rejected = attempted & ~exact_mask
        faults = torch.where(
            rejected,
            torch.full(
                self._flight_shape,
                FAULT_INVALID_RETIRE,
                dtype=torch.int64,
                device=self.device,
            ),
            torch.zeros(
                self._flight_shape,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        normal = accepted & normal_candidate
        cleanup = accepted & ~normal_candidate
        faults = torch.bitwise_or(
            faults,
            cleanup.to(torch.int64) * FAULT_SAFETY_CLEANUP,
        )

        initial_root = self._flight_lifecycle_snapshot()
        shadow = self._new_prepare_shadow()
        _masked_fill_(shadow._flight_state, accepted, FLIGHT_EMPTY)
        _masked_fill_(shadow._flight_physical_retired, accepted, 1)
        _masked_fill_(shadow._flight_action_epoch, accepted, 0)
        mailbox_target = (
            (accepted & mailbox_binding_valid).unsqueeze(-1)
            & (
                safe_mailbox.unsqueeze(-1)
                == shadow._mailbox_slot_ids.unsqueeze(1)
            )
        ).any(dim=1)
        _masked_fill_(shadow._mailbox_physical_retired, mailbox_target, 1)
        env_faults = (
            (torch.bitwise_and(faults, FAULT_INVALID_RETIRE) != 0)
            .any(dim=1)
            .to(torch.int64)
            * FAULT_INVALID_RETIRE
        )
        env_faults = torch.bitwise_or(
            env_faults,
            (
                (torch.bitwise_and(faults, FAULT_SAFETY_CLEANUP) != 0)
                .any(dim=1)
                .to(torch.int64)
                * FAULT_SAFETY_CLEANUP
            ),
        )
        shadow._lifecycle_fault_bits.bitwise_or_(env_faults)
        shadow._device_sticky_poison.bitwise_or_(cleanup.any(dim=1))
        shadow._retired_total.add_(accepted.to(torch.int64).sum())
        shadow._shared_normal_retire_total.add_(normal.to(torch.int64).sum())
        shadow._r06_only_orphan_retire_total.add_(
            cleanup.to(torch.int64).sum()
        )
        shadow._accumulate_normal_retire_key_summaries_(
            shadow._shared_normal_retire_key_summaries,
            normal,
            self._flight_full_key_sha256,
        )
        shadow._record_fault_events(faults)
        changed = (accepted | rejected).any().to(torch.int64)
        shadow._mutation_version.add_(changed)
        shadow._latest_receipt = None
        shadow._latest_global_drain_receipt = None
        shadow._latest_receipt_consumed = False
        shadow._load_fresh = False

        retained_cleanup_mask = cleanup.detach().clone()
        cleanup_capability = PhysicalRetireCleanupMaskCapability(
            _device_mask=cleanup.detach().clone(),
            _owner_identity=self._owner_identity,
            _token=_PHYSICAL_RETIRE_CLEANUP_CAP_TOKEN,
        )
        prepared = PreparedPhysicalRetire(
            _owner_identity=self._owner_identity,
            _settlement_result=settlement_result,
            _accepted=accepted.detach().clone(),
            _rejected=rejected.detach().clone(),
            _fault_bits=faults.detach().clone(),
            _normal_mask=normal.detach().clone(),
            _cleanup_mask=retained_cleanup_mask,
            _cleanup_capability=cleanup_capability,
            _token=_PHYSICAL_RETIRE_AUTH_TOKEN,
        )
        final_root = shadow._flight_lifecycle_snapshot()
        result = PhysicalRetireMutationResult(
            accepted=accepted.detach().clone(),
            rejected=rejected.detach().clone(),
            fault_bits=faults.detach().clone(),
            normal_mask=normal.detach().clone(),
            cleanup_mask=cleanup.detach().clone(),
            portable_success_mask=(
                normal & ~cleanup.any()
            ).detach().clone(),
            task_key=final_root.task_key,
            full_key_sha256=final_root.full_key_sha256.detach().clone(),
            ball_generation=final_root.ball_generation.detach().clone(),
            mailbox_slot=final_root.mailbox_slot.detach().clone(),
            observation_ordinal=(
                final_root.observation_ordinal.detach().clone()
            ),
            physical_retired=final_root.physical_retired.detach().clone(),
            mailbox_state=final_root.mailbox_state.detach().clone(),
            mailbox_task_key=DeviceLandingOutcomeKey(
                **{
                    name: getattr(
                        final_root.mailbox_task_key,
                        name,
                    ).detach().clone()
                    for name in _KEY_FIELDS
                }
            ),
            mailbox_full_key_sha256=(
                final_root.mailbox_full_key_sha256.detach().clone()
            ),
            mailbox_ball_generation=(
                final_root.mailbox_ball_generation.detach().clone()
            ),
            mailbox_reserved_flight_slot=(
                final_root.mailbox_reserved_flight_slot.detach().clone()
            ),
            mailbox_history_valid=(
                final_root.mailbox_history_valid.detach().clone()
            ),
            mailbox_physical_retired=(
                final_root.mailbox_physical_retired.detach().clone()
            ),
            mutation_version_before=self._mutation_version.detach().clone(),
            mutation_version_after=final_root.mutation_version.detach().clone(),
            initial_lifecycle_root=initial_root,
            final_lifecycle_root=final_root,
        )
        live_tensors = self._checkpoint_tensors()
        shadow_tensors = shadow._checkpoint_tensors()
        # The final shared-normal/orphan partition is fixed only after the
        # physical cleanup capability joins during prearm.
        shadow._shared_normal_retire_total.copy_(
            self._shared_normal_retire_total
        )
        shadow._r06_only_orphan_retire_total.copy_(
            self._r06_only_orphan_retire_total
        )
        shadow._shared_normal_retire_key_summaries.copy_(
            self._shared_normal_retire_key_summaries
        )
        tensor_swaps = tuple(
            (live_tensors[name], shadow_tensors[name])
            for name in sorted(live_tensors)
        )
        self._active_physical_retire_lease = _ActivePhysicalRetireLease(
            prepared_retire=prepared,
            retained_cleanup_mask=cleanup.detach().clone(),
            tensor_swaps=tensor_swaps,
            host_after_state={
                "_latest_receipt": None,
                "_latest_global_drain_receipt": None,
                "_latest_receipt_consumed": False,
                "_load_fresh": False,
                "_latest_post_physics_settlement": None,
            },
            mutation_result=result,
        )
        return prepared

    def prepared_physical_retire_result(
        self,
        prepared_retire: PreparedPhysicalRetire,
    ) -> PhysicalRetireMutationResult:
        """Return a clone of the immutable predicted result for physical prearm."""

        self._require_operable(
            allow_active_physical_retire=True,
            allow_pending_post_physics_settlement=True,
        )
        lease = self._active_physical_retire_lease
        if (
            lease is None
            or type(prepared_retire) is not PreparedPhysicalRetire
            or prepared_retire is not lease.prepared_retire
            or prepared_retire._owner_identity is not self._owner_identity
            or prepared_retire._token is not _PHYSICAL_RETIRE_AUTH_TOKEN
            or lease.armed_retire is not None
        ):
            raise LandingOutcomeDeviceError(
                "physical retire result requires the exact active prepare"
            )
        return _clone_physical_retire_result(lease.mutation_result)

    def physical_retire_cleanup_capability(
        self,
        prepared_retire: PreparedPhysicalRetire,
    ) -> PhysicalRetireCleanupMaskCapability:
        """Return the exact device cleanup-mask capability for physical pairing."""

        self._require_operable(
            allow_active_physical_retire=True,
            allow_pending_post_physics_settlement=True,
        )
        lease = self._active_physical_retire_lease
        capability = prepared_retire._cleanup_capability
        if (
            lease is None
            or type(prepared_retire) is not PreparedPhysicalRetire
            or prepared_retire is not lease.prepared_retire
            or type(capability) is not PhysicalRetireCleanupMaskCapability
            or capability._owner_identity is not self._owner_identity
            or capability._token is not _PHYSICAL_RETIRE_CLEANUP_CAP_TOKEN
            or capability._device_mask is prepared_retire._cleanup_mask
            or tuple(capability._device_mask.shape) != self._flight_shape
            or capability._device_mask.dtype != torch.bool
            or capability._device_mask.device != self.device
        ):
            raise LandingOutcomeDeviceError(
                "physical retire cleanup capability is stale or foreign"
            )
        return capability

    def _prearm_physical_cleanup_union(
        self,
        lease: _ActivePhysicalRetireLease,
        claim: LandingOutcomePhysicalParkPreparedTokenClaim,
    ) -> None:
        """Fold both owners' device cleanup masks into private after-images."""

        prepared = lease.prepared_retire
        previous = lease.mutation_result
        retained_cleanup = lease.retained_cleanup_mask
        physical_capability = claim.physical_cleanup_capability
        physical_mask = getattr(physical_capability, "_device_mask", None)
        r06_capability_mask = claim.r06_cleanup_capability._device_mask
        if (
            claim.r06_cleanup_capability
            is not prepared._cleanup_capability
            or not isinstance(retained_cleanup, torch.Tensor)
            or tuple(retained_cleanup.shape) != self._flight_shape
            or retained_cleanup.dtype != torch.bool
            or retained_cleanup.device != self.device
            or not isinstance(prepared._cleanup_mask, torch.Tensor)
            or tuple(prepared._cleanup_mask.shape) != self._flight_shape
            or prepared._cleanup_mask.dtype != torch.bool
            or prepared._cleanup_mask.device != self.device
            or not isinstance(r06_capability_mask, torch.Tensor)
            or tuple(r06_capability_mask.shape) != self._flight_shape
            or r06_capability_mask.dtype != torch.bool
            or r06_capability_mask.device != self.device
            or not isinstance(physical_mask, torch.Tensor)
            or tuple(physical_mask.shape) != self._flight_shape
            or physical_mask.dtype != torch.bool
            or physical_mask.device != self.device
        ):
            raise LandingOutcomeDeviceError(
                "physical park cleanup-mask capability differs"
            )
        r06_capability_match = torch.eq(
            r06_capability_mask,
            retained_cleanup,
        ).all() & torch.eq(
            prepared._cleanup_mask,
            retained_cleanup,
        ).all()
        combined_cleanup = torch.bitwise_or(
            retained_cleanup,
            physical_mask,
        )
        physical_only_cleanup = physical_mask & ~retained_cleanup
        cleanup_env = combined_cleanup.any(dim=1) | ~r06_capability_match
        changed = (
            previous.accepted
            | previous.rejected
            | physical_mask
        ).any() | ~r06_capability_match
        changed = changed.to(torch.int64)
        rowwise_normal = previous.normal_mask & ~combined_cleanup
        version_after: torch.Tensor | None = None
        for destination, after_image in lease.tensor_swaps:
            if destination is self._device_sticky_poison:
                after_image.bitwise_or_(cleanup_env)
            elif destination is self._lifecycle_fault_bits:
                after_image.bitwise_or_(
                    cleanup_env.to(torch.int64) * FAULT_SAFETY_CLEANUP
                )
            elif destination is self._fault_event_counts:
                safety_index = len(FAULTS) - 1
                after_image[safety_index].add_(
                    physical_only_cleanup.to(torch.int64).sum()
                    + (~r06_capability_match).to(torch.int64)
                    * self.num_envs
                    * self.flight_slot_capacity
                )
            elif destination is self._mutation_version:
                after_image.copy_(
                    lease.mutation_result.mutation_version_before + changed
                )
                version_after = after_image
            elif destination is self._shared_normal_retire_total:
                after_image.copy_(
                    self._shared_normal_retire_total
                    + rowwise_normal.to(torch.int64).sum()
                )
            elif destination is self._r06_only_orphan_retire_total:
                after_image.copy_(
                    self._r06_only_orphan_retire_total
                    + retained_cleanup.to(torch.int64).sum()
                )
            elif destination is self._shared_normal_retire_key_summaries:
                after_image.copy_(self._shared_normal_retire_key_summaries)
                self._accumulate_normal_retire_key_summaries_(
                    after_image,
                    rowwise_normal,
                    self._flight_full_key_sha256,
                )
        if version_after is None:
            raise LandingOutcomeDeviceError(
                "physical retire mutation-version after-image is missing"
            )
        combined_fault_bits = torch.bitwise_or(
            previous.fault_bits,
            physical_mask.to(torch.int64) * FAULT_SAFETY_CLEANUP,
        )
        combined_fault_bits = torch.bitwise_or(
            combined_fault_bits,
            (~r06_capability_match).to(torch.int64)
            * torch.full_like(
                combined_fault_bits,
                FAULT_SAFETY_CLEANUP,
            ),
        )
        portable_success = (
            rowwise_normal
            & ~combined_cleanup.any()
            & r06_capability_match
        )
        combined_accepted = previous.accepted | physical_mask
        combined_rejected = previous.rejected & ~combined_accepted
        previous_root = previous.final_lifecycle_root
        final_root = FlightLifecycleSnapshotBatch(
            state=previous_root.state,
            task_key=previous_root.task_key,
            full_key_sha256=previous_root.full_key_sha256,
            ball_generation=previous_root.ball_generation,
            mailbox_slot=previous_root.mailbox_slot,
            observation_ordinal=previous_root.observation_ordinal,
            physical_retired=previous_root.physical_retired,
            mailbox_state=previous_root.mailbox_state,
            mailbox_task_key=previous_root.mailbox_task_key,
            mailbox_full_key_sha256=(
                previous_root.mailbox_full_key_sha256
            ),
            mailbox_ball_generation=(
                previous_root.mailbox_ball_generation
            ),
            mailbox_reserved_flight_slot=(
                previous_root.mailbox_reserved_flight_slot
            ),
            mailbox_history_valid=previous_root.mailbox_history_valid,
            mailbox_physical_retired=(
                previous_root.mailbox_physical_retired
            ),
            mutation_version=version_after.detach().clone(),
        )
        lease.mutation_result = PhysicalRetireMutationResult(
            accepted=combined_accepted,
            rejected=combined_rejected,
            fault_bits=combined_fault_bits,
            normal_mask=rowwise_normal,
            cleanup_mask=combined_cleanup,
            portable_success_mask=portable_success,
            task_key=previous.task_key,
            full_key_sha256=previous.full_key_sha256,
            ball_generation=previous.ball_generation,
            mailbox_slot=previous.mailbox_slot,
            observation_ordinal=previous.observation_ordinal,
            physical_retired=previous.physical_retired,
            mailbox_state=previous.mailbox_state,
            mailbox_task_key=previous.mailbox_task_key,
            mailbox_full_key_sha256=previous.mailbox_full_key_sha256,
            mailbox_ball_generation=previous.mailbox_ball_generation,
            mailbox_reserved_flight_slot=(
                previous.mailbox_reserved_flight_slot
            ),
            mailbox_history_valid=previous.mailbox_history_valid,
            mailbox_physical_retired=previous.mailbox_physical_retired,
            mutation_version_before=previous.mutation_version_before,
            mutation_version_after=version_after.detach().clone(),
            initial_lifecycle_root=previous.initial_lifecycle_root,
            final_lifecycle_root=final_root,
        )

    def arm_physical_retire(
        self,
        prepared_retire: PreparedPhysicalRetire,
        physical_prepared_token: object,
    ) -> ArmedPhysicalRetire:
        """Bind one exact physical prepared token; perform no device work."""

        self._require_operable(
            allow_active_physical_retire=True,
            allow_pending_post_physics_settlement=True,
        )
        lease = self._active_physical_retire_lease
        authority = self._physical_park_token_authority
        if (
            lease is None
            or prepared_retire is not lease.prepared_retire
            or type(prepared_retire) is not PreparedPhysicalRetire
            or prepared_retire._owner_identity is not self._owner_identity
            or prepared_retire._token is not _PHYSICAL_RETIRE_AUTH_TOKEN
            or lease.armed_retire is not None
            or authority is None
        ):
            raise LandingOutcomeDeviceError(
                "physical retire prepare is stale, foreign, or already armed"
            )
        claim = authority.require_owned_prepared_token(
            physical_prepared_token,
            expected_r06_prepared_retire=prepared_retire,
        )
        self._prearm_physical_cleanup_union(lease, claim)
        armed = ArmedPhysicalRetire(
            _prepared_retire=prepared_retire,
            _physical_prepared_token=physical_prepared_token,
            _r06_cleanup_capability=claim.r06_cleanup_capability,
            _physical_cleanup_capability=(
                claim.physical_cleanup_capability
            ),
            _owner_identity=self._owner_identity,
            _token=_PHYSICAL_RETIRE_AUTH_TOKEN,
        )
        lease.armed_retire = armed
        return armed

    def armed_physical_retire_result(
        self,
        armed_retire: ArmedPhysicalRetire,
    ) -> PhysicalRetireMutationResult:
        """Expose the final cross-bound result for physical arm validation."""

        self._require_operable(
            allow_active_physical_retire=True,
            allow_pending_post_physics_settlement=True,
        )
        lease = self._active_physical_retire_lease
        if (
            lease is None
            or type(armed_retire) is not ArmedPhysicalRetire
            or armed_retire is not lease.armed_retire
            or armed_retire._owner_identity is not self._owner_identity
            or armed_retire._token is not _PHYSICAL_RETIRE_AUTH_TOKEN
        ):
            raise LandingOutcomeDeviceError(
                "physical retire result requires the exact armed handle"
            )
        return _clone_physical_retire_result(lease.mutation_result)

    def commit_prevalidated_physical_retire(
        self,
        armed_retire: ArmedPhysicalRetire,
    ) -> PhysicalRetireMutationResult:
        """Publish only after the exact physical prepared token is committed."""

        lease = self._active_physical_retire_lease
        authority = self._physical_park_token_authority
        if (
            self._poisoned
            or lease is None
            or armed_retire is not lease.armed_retire
            or type(armed_retire) is not ArmedPhysicalRetire
            or armed_retire._owner_identity is not self._owner_identity
            or armed_retire._token is not _PHYSICAL_RETIRE_AUTH_TOKEN
            or authority is None
        ):
            raise LandingOutcomeDeviceError(
                "armed physical retire is not the active opaque handle"
            )
        try:
            committed_claim = authority.require_committed_prepared_token(
                armed_retire._physical_prepared_token,
                expected_r06_prepared_retire=lease.prepared_retire,
            )
            if (
                committed_claim.r06_cleanup_capability
                is not armed_retire._r06_cleanup_capability
                or committed_claim.physical_cleanup_capability
                is not armed_retire._physical_cleanup_capability
            ):
                raise LandingOutcomeDeviceError(
                    "physical park cleanup capability changed after arm"
                )
            for destination, after_image in lease.tensor_swaps:
                destination.copy_(after_image)
            for name, value in lease.host_after_state.items():
                self.__dict__[name] = value
            result = lease.mutation_result
            self._active_physical_retire_lease = None
            return result
        except Exception:
            # The physical scene may already be parked.  Keep R06 capacity
            # occupied, poison, and never fabricate a rollback.
            self._poisoned = True
            self._active_physical_retire_lease = None
            raise

    def abort_physical_retire(
        self,
        prepared_retire: PreparedPhysicalRetire,
    ) -> None:
        """Discard an unarmed private retire after-image without live writes."""

        self._require_operable(
            allow_active_physical_retire=True,
            allow_pending_post_physics_settlement=True,
        )
        lease = self._active_physical_retire_lease
        if (
            lease is None
            or prepared_retire is not lease.prepared_retire
            or prepared_retire._owner_identity is not self._owner_identity
            or prepared_retire._token is not _PHYSICAL_RETIRE_AUTH_TOKEN
            or lease.armed_retire is not None
        ):
            raise LandingOutcomeDeviceError(
                "physical retire abort token is stale, foreign, or armed"
            )
        self._active_physical_retire_lease = None

    def retire_physical(self, **_caller_pins: object) -> DeviceMutationResult:
        """Tombstone the caller-selected slot/key/generation retire ABI."""

        raise LandingOutcomeDeviceError(
            "retire_physical is tombstoned; use all-grid prepare/arm/commit"
        )

    def _combined_fault_counts(self) -> list[torch.Tensor]:
        return [self._fault_event_counts[index] for index in range(len(FAULTS))]

    def _invariant_counts(self) -> list[torch.Tensor]:
        if self._action_ball_full_mdp_epoch_owner is not None:
            return self._action_epoch_invariant_counts()
        flight_active = self._flight_state != FLIGHT_EMPTY
        mailbox_active = self._mailbox_state != MAILBOX_EMPTY
        flight_state_bad = ~(
            (self._flight_state == FLIGHT_EMPTY)
            | (self._flight_state == FLIGHT_INBOUND)
            | (self._flight_state == FLIGHT_OPEN)
            | (self._flight_state == FLIGHT_SETTLED_RETAINED)
        )
        mailbox_state_bad = ~(
            (self._mailbox_state == MAILBOX_EMPTY)
            | (self._mailbox_state == MAILBOX_SETTLED_UNPAID)
            | (self._mailbox_state == MAILBOX_PARTIALLY_PAID)
            | (self._mailbox_state == MAILBOX_PAID)
        )
        flight_digest_bad = torch.zeros_like(flight_active)
        mailbox_digest_bad = torch.zeros_like(mailbox_active)
        for name in _DIGEST_KEY_FIELDS:
            required_flight = ~self._flight_action_epoch
            required_mailbox = ~self._mailbox_action_epoch
            flight_digest_bad |= required_flight & ~self._flight_key_digests[
                name
            ].ne(0).any(dim=-1)
            mailbox_digest_bad |= required_mailbox & ~self._mailbox_key_digests[
                name
            ].ne(0).any(dim=-1)
        flight_key_values_bad = (
            (self._flight_key_ints["reset_generation"] < 1)
            | (self._flight_key_ints["swing_generation"] < 0)
            | (self._flight_key_ints["action_uid"] < 1)
            | (self._flight_key_ints["action_uid"] > _MAX_ACTION_UID)
            | (self._flight_key_ints["action_slot"] < 0)
            | (self._flight_key_ints["shot_index"] < 1)
        )
        mailbox_key_values_bad = (
            (self._mailbox_key_ints["reset_generation"] < 1)
            | (self._mailbox_key_ints["swing_generation"] < 0)
            | (self._mailbox_key_ints["action_uid"] < 1)
            | (self._mailbox_key_ints["action_uid"] > _MAX_ACTION_UID)
            | (self._mailbox_key_ints["action_slot"] < 0)
            | (self._mailbox_key_ints["shot_index"] < 1)
        )
        flight_target_bad = (
            ~torch.isfinite(self._flight_target_xy_m).all(dim=-1)
            | (self._flight_target_xy_m[..., 0] < self.profile.opponent_table_x_min_m)
            | (self._flight_target_xy_m[..., 0] > self.profile.opponent_table_x_max_m)
            | (self._flight_target_xy_m[..., 1] < self.profile.table_y_min_m)
            | (self._flight_target_xy_m[..., 1] > self.profile.table_y_max_m)
        )
        mailbox_target_bad = (
            ~torch.isfinite(self._mailbox_target_xy_m).all(dim=-1)
            | (self._mailbox_target_xy_m[..., 0] < self.profile.opponent_table_x_min_m)
            | (self._mailbox_target_xy_m[..., 0] > self.profile.opponent_table_x_max_m)
            | (self._mailbox_target_xy_m[..., 1] < self.profile.table_y_min_m)
            | (self._mailbox_target_xy_m[..., 1] > self.profile.table_y_max_m)
        )
        flight_typed_bad = self._flight_action_epoch & (
            (self._flight_action_uid < 1)
            | (self._flight_action_uid > _MAX_ACTION_UID)
            | (self._flight_action_slot < 0)
            | (self._flight_reset_generation < 1)
            | (self._flight_shot_index < 1)
            | (self._flight_task_identity < 1)
            | (self._flight_outcome_identity < 1)
            | (self._flight_ball_identity < 1)
            | (self._flight_publication_ordinal < 0)
        )
        flight_legacy_bad = ~self._flight_action_epoch & (
            ~self._flight_full_key_sha256.ne(0).any(dim=-1)
            | ~self._flight_task_identity_token.ne(0).any(dim=-1)
            | (self._flight_key_ints["env_id"] != self._env_ids.unsqueeze(1))
            | flight_digest_bad
            | flight_key_values_bad
        )
        flight_owner_bad = flight_active & (
            (self._flight_ball_generation < 0)
            | flight_typed_bad
            | flight_legacy_bad
            | (
                ~self._flight_action_epoch
                & ~self._flight_full_key_receipt_sha256.ne(0).any(dim=-1)
            )
            | (
                ~self._flight_action_epoch
                & ~self._flight_committed_reveal_sha256.ne(0).any(dim=-1)
            )
            | (
                ~self._flight_action_epoch
                & ~self._flight_install_receipt_sha256.ne(0).any(dim=-1)
            )
            | flight_target_bad
            | (self._flight_reveal_control_step < 0)
            | (
                self._flight_contact_deadline_control_step
                < self._flight_reveal_control_step
            )
            | (
                self._flight_crossing_horizon_control_step
                < self._flight_contact_deadline_control_step
            )
        )
        mailbox_record = self._mailbox_history_valid
        mailbox_typed_bad = self._mailbox_action_epoch & (
            (self._mailbox_action_uid < 1)
            | (self._mailbox_action_uid > _MAX_ACTION_UID)
            | (self._mailbox_action_slot < 0)
            | (self._mailbox_reset_generation < 1)
            | (self._mailbox_shot_index < 1)
            | (self._mailbox_task_identity < 1)
            | (self._mailbox_outcome_identity < 1)
            | (self._mailbox_ball_identity < 1)
            | (self._mailbox_publication_ordinal < 0)
        )
        mailbox_legacy_bad = ~self._mailbox_action_epoch & (
            ~self._mailbox_full_key_sha256.ne(0).any(dim=-1)
            | ~self._mailbox_task_identity_token.ne(0).any(dim=-1)
            | (self._mailbox_key_ints["env_id"] != self._env_ids.unsqueeze(1))
            | mailbox_digest_bad
            | mailbox_key_values_bad
        )
        mailbox_record_bad = mailbox_record & (
            (self._mailbox_ball_generation < 0)
            | mailbox_typed_bad
            | mailbox_legacy_bad
            | (
                ~self._mailbox_action_epoch
                & ~self._mailbox_full_key_receipt_sha256.ne(0).any(dim=-1)
            )
            | (
                ~self._mailbox_action_epoch
                & ~self._mailbox_committed_reveal_sha256.ne(0).any(dim=-1)
            )
            | (
                ~self._mailbox_action_epoch
                & ~self._mailbox_install_receipt_sha256.ne(0).any(dim=-1)
            )
            | mailbox_target_bad
            | (self._mailbox_observation_ordinal < 0)
            | (self._mailbox_settlement_cause < SETTLEMENT_CAUSE_FIRST_CROSSING)
            | (self._mailbox_settlement_cause > SETTLEMENT_CAUSE_PROTOCOL_FAULT)
            | (self._mailbox_c10_family_code != self._c10_family_code)
            | (
                self._mailbox_placement_treatment_gain
                != self._placement_treatment_gain
            )
            | ~torch.eq(
                self._mailbox_c10_projection_sha256,
                self._c10_projection_token.reshape(1, 1, TOKEN_BYTES).expand(
                    self._mailbox_shape + (TOKEN_BYTES,)
                ),
            ).all(dim=-1)
        )
        mailbox_owner_bad = (
            (mailbox_active & ~mailbox_record)
            | mailbox_record_bad
            | (
                mailbox_active
                & (
                    ~self._mailbox_reserved
                    | (
                        self._mailbox_reservation_generation
                        != self._mailbox_ball_generation
                    )
                    | (self._mailbox_reserved_flight_slot < 0)
                    | (
                        self._mailbox_reserved_flight_slot
                        >= self.flight_slot_capacity
                    )
                    | (
                        ~self._mailbox_action_epoch
                        & ~torch.eq(
                            self._mailbox_reservation_token,
                            self._mailbox_full_key_sha256,
                        ).all(dim=-1)
                    )
                )
            )
        )
        association = self._flight_mailbox_association()
        flight_association_count = association.to(torch.int64).sum(dim=1)
        mailbox_association_count = association.to(torch.int64).sum(dim=2)
        preflight = (
            (self._flight_state == FLIGHT_INBOUND)
            | (self._flight_state == FLIGHT_OPEN)
        )
        retained = self._flight_state == FLIGHT_SETTLED_RETAINED
        mailbox_preflight_count = (
            association & preflight.unsqueeze(1)
        ).to(torch.int64).sum(dim=2)
        mailbox_retained_count = (
            association & retained.unsqueeze(1)
        ).to(torch.int64).sum(dim=2)
        reserved_empty = self._mailbox_reserved & ~mailbox_active
        reservation_bad = (
            (flight_active & (flight_association_count != 1)).to(torch.int64).sum()
            + (
                reserved_empty
                & (
                    (mailbox_association_count != 1)
                    | (mailbox_preflight_count != 1)
                    | (mailbox_retained_count != 0)
                )
            ).to(torch.int64).sum()
        ).reshape(1)
        active_unretired = mailbox_active & ~self._mailbox_physical_retired
        active_retired = mailbox_active & self._mailbox_physical_retired
        retained_bad = (
            (
                retained
                & (
                    (flight_association_count != 1)
                    | ~self._flight_retained_mailbox_active()
                )
            ).to(torch.int64).sum()
            + (
                active_unretired
                & (
                    (mailbox_association_count != 1)
                    | (mailbox_retained_count != 1)
                    | (mailbox_preflight_count != 0)
                )
            ).to(torch.int64).sum()
            + (
                active_retired
                & (
                    (mailbox_association_count != 0)
                    | (mailbox_retained_count != 0)
                    | (mailbox_preflight_count != 0)
                )
            ).to(torch.int64).sum()
        ).reshape(1)
        retirement_bad = (
            (self._mailbox_physical_retired & ~mailbox_active)
            .to(torch.int64)
            .sum()
            + (flight_active & self._flight_physical_retired)
            .to(torch.int64)
            .sum()
            + ((~flight_active) & ~self._flight_physical_retired)
            .to(torch.int64)
            .sum()
        ).reshape(1)
        contact_bad = (
            ((self._flight_state == FLIGHT_OPEN) & ~self._flight_contact_valid)
            | ((self._flight_state == FLIGHT_INBOUND) & self._flight_contact_valid)
        )
        observation_bad = flight_active & (
            (self._flight_observation_ordinal < -1)
            | (
                (self._flight_observation_ordinal >= 0)
                & (self._flight_last_observation_control < 0)
            )
            | (
                (self._flight_observation_ordinal >= 0)
                & (
                    self._flight_last_observation_control
                    != (
                        self._flight_reveal_control_step
                        + self._flight_observation_ordinal
                    )
                )
            )
        )
        flight_contact_before_net = _stamp_less_fields(
            self._flight_contact_stamp_control,
            self._flight_contact_stamp_substep,
            torch.full(
                self._flight_shape,
                PHASE_CONTACT,
                dtype=torch.int8,
                device=self.device,
            ),
            self._flight_net_stamp_control,
            self._flight_net_stamp_substep,
            torch.full(
                self._flight_shape,
                PHASE_NET,
                dtype=torch.int8,
                device=self.device,
            ),
        )
        flight_stamp_bad = flight_active & (
            (self._flight_contact_valid & (self._flight_contact_stamp_control < 0))
            | (
                self._flight_net_crossed
                & (
                    (self._flight_net_stamp_control < 0)
                    | ~self._flight_contact_valid
                    | ~flight_contact_before_net
                )
            )
        )
        mailbox_contact_before_net = _stamp_less_fields(
            self._mailbox_source_control_step,
            self._mailbox_source_physics_substep,
            torch.full(
                self._mailbox_shape,
                PHASE_CONTACT,
                dtype=torch.int8,
                device=self.device,
            ),
            self._mailbox_net_stamp_control,
            self._mailbox_net_stamp_substep,
            torch.full(
                self._mailbox_shape,
                PHASE_NET,
                dtype=torch.int8,
                device=self.device,
            ),
        )
        mailbox_contact_before_crossing = _stamp_less_fields(
            self._mailbox_source_control_step,
            self._mailbox_source_physics_substep,
            torch.full(
                self._mailbox_shape,
                PHASE_CONTACT,
                dtype=torch.int8,
                device=self.device,
            ),
            self._mailbox_crossing_stamp_control,
            self._mailbox_crossing_stamp_substep,
            torch.full(
                self._mailbox_shape,
                PHASE_LANDING,
                dtype=torch.int8,
                device=self.device,
            ),
        )
        mailbox_crossing_before_settlement = _stamp_less_fields(
            self._mailbox_crossing_stamp_control,
            self._mailbox_crossing_stamp_substep,
            torch.full(
                self._mailbox_shape,
                PHASE_LANDING,
                dtype=torch.int8,
                device=self.device,
            ),
            self._mailbox_settlement_control_step,
            self._mailbox_settlement_physics_substep,
            torch.full(
                self._mailbox_shape,
                PHASE_LANDING,
                dtype=torch.int8,
                device=self.device,
            ),
        ) | (
            (self._mailbox_crossing_stamp_control == self._mailbox_settlement_control_step)
            & (
                self._mailbox_crossing_stamp_substep
                == self._mailbox_settlement_physics_substep
            )
        )
        mailbox_stamp_bad = mailbox_record & (
            (self._mailbox_settlement_control_step < 0)
            | (self._mailbox_settlement_physics_substep < 0)
            | (self._mailbox_contact_valid & (self._mailbox_source_control_step < 0))
            | (
                self._mailbox_net_crossed
                & (
                    (self._mailbox_net_stamp_control < 0)
                    | ~self._mailbox_contact_valid
                    | ~mailbox_contact_before_net
                )
            )
            | (
                self._mailbox_crossing_present
                & (
                    (self._mailbox_crossing_stamp_control < 0)
                    | ~self._mailbox_contact_valid
                    | ~mailbox_contact_before_crossing
                    | ~mailbox_crossing_before_settlement
                    | (
                        self._mailbox_net_crossed
                        & ~_stamp_less_fields(
                            self._mailbox_net_stamp_control,
                            self._mailbox_net_stamp_substep,
                            torch.full(
                                self._mailbox_shape,
                                PHASE_NET,
                                dtype=torch.int8,
                                device=self.device,
                            ),
                            self._mailbox_crossing_stamp_control,
                            self._mailbox_crossing_stamp_substep,
                            torch.full(
                                self._mailbox_shape,
                                PHASE_LANDING,
                                dtype=torch.int8,
                                device=self.device,
                            ),
                        )
                    )
                )
            )
        )
        stamp_bad = (
            flight_stamp_bad.to(torch.int64).sum()
            + mailbox_stamp_bad.to(torch.int64).sum()
        ).reshape(1)
        paid_count = (
            torch.bitwise_and(self._mailbox_paid_mask, 1) != 0
        ).to(torch.int64) + (
            torch.bitwise_and(self._mailbox_paid_mask, 2) != 0
        ).to(torch.int64)
        consumer_bad = (
            mailbox_active
            & (
                ((self._mailbox_state == MAILBOX_SETTLED_UNPAID) & (paid_count != 0))
                | ((self._mailbox_state == MAILBOX_PARTIALLY_PAID) & (paid_count != 1))
                | ((self._mailbox_state == MAILBOX_PAID) & (paid_count != 2))
            )
        ) | (
            mailbox_record & ~mailbox_active & (paid_count != CONSUMER_COUNT)
        )
        view_payment_bad = torch.zeros_like(mailbox_active)
        for index in range(CONSUMER_COUNT):
            paid = torch.bitwise_and(self._mailbox_paid_mask, 1 << index) != 0
            viewed = self._mailbox_view_epoch[..., index] >= 0
            view_payment_bad = view_payment_bad | (
                mailbox_record
                & (
                    (
                        viewed
                        & (
                            self._mailbox_view_epoch[..., index]
                            != self._mailbox_settlement_control_step
                        )
                    )
                    | (
                        paid
                        & (
                            ~viewed
                            | (
                                self._mailbox_payment_epoch[..., index]
                                != self._mailbox_view_epoch[..., index]
                            )
                        )
                    )
                )
            )
        expected_common = self._mailbox_common_on_table.to(self.dtype)
        common_paid = torch.bitwise_and(self._mailbox_paid_mask, 1) != 0
        placement_paid = torch.bitwise_and(self._mailbox_paid_mask, 2) != 0
        expected_placement = (
            self._mailbox_canonical_total * self._mailbox_placement_treatment_gain
        )
        payment_bad = mailbox_record & (
            (
                common_paid
                & ~torch.eq(self._mailbox_payment_values[..., 0], expected_common)
            )
            | (
                placement_paid
                & (
                    ~(
                        (self._mailbox_placement_treatment_gain == 0.0)
                        | (self._mailbox_placement_treatment_gain == 1.0)
                    )
                    | ~torch.eq(
                        self._mailbox_payment_values[..., 1], expected_placement
                    )
                )
            )
        )
        canonical_bad = self._canonical_invariant_bad(mailbox_record)
        previous_digest_bad = torch.zeros_like(self._previous_paid_valid)
        for name in _DIGEST_KEY_FIELDS:
            required_previous = (
                torch.ones_like(self._previous_paid_valid)
                if name == "task_sha256"
                else ~self._previous_paid_action_epoch
            )
            previous_digest_bad |= required_previous & ~self._previous_paid_key_digests[
                name
            ].ne(0).any(dim=-1)
        previous_paid_bad = self._previous_paid_valid & (
            (self._previous_paid_key_ints["env_id"] != self._env_ids)
            | (self._previous_paid_key_ints["reset_generation"] < 1)
            | (self._previous_paid_key_ints["swing_generation"] < 0)
            | (self._previous_paid_key_ints["action_uid"] < 1)
            | (self._previous_paid_key_ints["action_uid"] > _MAX_ACTION_UID)
            | (self._previous_paid_key_ints["action_slot"] < 0)
            | (self._previous_paid_key_ints["shot_index"] < 1)
            | previous_digest_bad
            | ~self._previous_paid_full_key_sha256.ne(0).any(dim=-1)
            | (self._previous_paid_ball_generation < 0)
            | (self._previous_paid_observation_ordinal < 0)
            | (self._previous_paid_settlement_control_step < 0)
            | ~torch.isfinite(self._previous_paid_target_error_m)
            | (self._previous_paid_target_error_m < 0.0)
            | ~torch.isfinite(self._previous_paid_target_xy_m).all(dim=-1)
            | (
                self._previous_paid_target_xy_m[:, 0]
                < self.profile.opponent_table_x_min_m
            )
            | (
                self._previous_paid_target_xy_m[:, 0]
                > self.profile.opponent_table_x_max_m
            )
            | (
                self._previous_paid_target_xy_m[:, 1]
                < self.profile.table_y_min_m
            )
            | (
                self._previous_paid_target_xy_m[:, 1]
                > self.profile.table_y_max_m
            )
        )
        previous_after_replay = (
            (
                self._previous_paid_key_ints["reset_generation"]
                > self._replay_reset_generation
            )
            | (
                (
                    self._previous_paid_key_ints["reset_generation"]
                    == self._replay_reset_generation
                )
                & (
                    self._previous_paid_key_ints["swing_generation"]
                    > self._replay_swing_generation
                )
            )
        )
        previous_at_replay = (
            (
                self._previous_paid_key_ints["reset_generation"]
                == self._replay_reset_generation
            )
            & (
                self._previous_paid_key_ints["swing_generation"]
                == self._replay_swing_generation
            )
        )
        previous_paid_bad = previous_paid_bad | (
            self._previous_paid_valid
            & (
                ~self._replay_valid
                | previous_after_replay
                | (
                    self._previous_paid_key_ints["reset_generation"]
                    > self._reset_generation_highwater
                )
                | (
                    previous_at_replay
                    & ~torch.eq(
                        self._previous_paid_full_key_sha256,
                        self._replay_full_key_sha256,
                    ).all(dim=-1)
                )
            )
        )
        replay_bad = self._replay_valid & (
            (self._replay_reset_generation < 1)
            | (self._replay_swing_generation < 0)
            | ~self._replay_full_key_sha256.ne(0).any(dim=-1)
            | (
                self._replay_reset_generation
                > self._reset_generation_highwater
            )
        )
        generation_highwater_bad = (
            flight_active
            & (
                self._flight_key_ints["reset_generation"]
                > self._reset_generation_highwater.unsqueeze(1)
            )
        ).any(dim=1) | (
            mailbox_record
            & (
                self._mailbox_key_ints["reset_generation"]
                > self._reset_generation_highwater.unsqueeze(1)
            )
        ).any(dim=1)
        flight_after_replay = (
            (
                self._flight_key_ints["reset_generation"]
                > self._replay_reset_generation.unsqueeze(1)
            )
            | (
                (
                    self._flight_key_ints["reset_generation"]
                    == self._replay_reset_generation.unsqueeze(1)
                )
                & (
                    self._flight_key_ints["swing_generation"]
                    > self._replay_swing_generation.unsqueeze(1)
                )
            )
        )
        mailbox_after_replay = (
            (
                self._mailbox_key_ints["reset_generation"]
                > self._replay_reset_generation.unsqueeze(1)
            )
            | (
                (
                    self._mailbox_key_ints["reset_generation"]
                    == self._replay_reset_generation.unsqueeze(1)
                )
                & (
                    self._mailbox_key_ints["swing_generation"]
                    > self._replay_swing_generation.unsqueeze(1)
                )
            )
        )
        flight_at_replay = (
            (
                self._flight_key_ints["reset_generation"]
                == self._replay_reset_generation.unsqueeze(1)
            )
            & (
                self._flight_key_ints["swing_generation"]
                == self._replay_swing_generation.unsqueeze(1)
            )
        )
        mailbox_at_replay = (
            (
                self._mailbox_key_ints["reset_generation"]
                == self._replay_reset_generation.unsqueeze(1)
            )
            & (
                self._mailbox_key_ints["swing_generation"]
                == self._replay_swing_generation.unsqueeze(1)
            )
        )
        flight_replay_token_match = torch.eq(
            self._flight_full_key_sha256,
            self._replay_full_key_sha256.unsqueeze(1),
        ).all(dim=-1)
        mailbox_replay_token_match = torch.eq(
            self._mailbox_full_key_sha256,
            self._replay_full_key_sha256.unsqueeze(1),
        ).all(dim=-1)
        replay_owner_present = (
            flight_active & flight_at_replay & flight_replay_token_match
        ).any(dim=1) | (
            mailbox_record & mailbox_at_replay & mailbox_replay_token_match
        ).any(dim=1) | (
            self._previous_paid_valid
            & previous_at_replay
            & torch.eq(
                self._previous_paid_full_key_sha256,
                self._replay_full_key_sha256,
            ).all(dim=-1)
        )
        replay_bad = replay_bad | generation_highwater_bad | (
            flight_active
            & (
                flight_after_replay
                | (flight_at_replay & ~flight_replay_token_match)
            )
        ).any(dim=1) | (
            mailbox_record
            & (
                mailbox_after_replay
                | (mailbox_at_replay & ~mailbox_replay_token_match)
            )
        ).any(dim=1) | (self._replay_valid & ~replay_owner_present)
        nonempty_flights = flight_active.to(torch.int64).sum()
        nonempty_mailboxes = mailbox_active.to(torch.int64).sum()
        payment_count_0 = (mailbox_active & common_paid).to(torch.int64).sum()
        payment_count_1 = (mailbox_active & placement_paid).to(torch.int64).sum()
        counter_bad = (
            (self._mutation_version < 0)
            | (self._fault_event_counts < 0).any()
            | (self._reset_generation_highwater < 0).any()
            | (self._selected_reset_count < 0).any()
            | (self._installed_total < 0)
            | (self._settled_total < 0)
            | (self._retired_total < 0)
            | (self._closed_total < 0)
            | (self._selected_reset_retired_flight_total < 0)
            | (self._selected_reset_closed_mailbox_total < 0)
            | (self._selected_reset_retired_payment_totals < 0).any()
            | (self._terminal_resolution_total < 0)
            | (self._shared_normal_retire_total < 0)
            | (self._r06_only_orphan_retire_total < 0)
            | (self._shared_normal_retire_key_summaries < 0).any()
            | (
                self._shared_normal_retire_key_summaries
                >= _R06_RETIRE_SUMMARY_MODULUS
            ).any()
            | (self._payment_totals[0] < 0)
            | (self._payment_totals[1] < 0)
            | (self._settled_total > self._installed_total)
            | (self._retired_total > self._settled_total)
            | (self._closed_total > self._retired_total)
            | (
                self._retired_total
                + self._selected_reset_retired_flight_total
                > self._installed_total
            )
            | (
                self._shared_normal_retire_total
                + self._r06_only_orphan_retire_total
                > self._retired_total
            )
            | (
                self._closed_total
                + self._selected_reset_closed_mailbox_total
                > self._settled_total
            )
            | (
                self._selected_reset_retired_payment_totals
                > self._selected_reset_closed_mailbox_total
            ).any()
            | (self._payment_totals[0] < self._closed_total)
            | (self._payment_totals[1] < self._closed_total)
            | (self._payment_totals[0] > self._settled_total)
            | (self._payment_totals[1] > self._settled_total)
            | (
                self._installed_total
                - self._retired_total
                - self._selected_reset_retired_flight_total
                != nonempty_flights
            )
            | (
                self._settled_total
                - self._closed_total
                - self._selected_reset_closed_mailbox_total
                != nonempty_mailboxes
            )
            | (
                self._payment_totals[0]
                != payment_count_0
                + self._closed_total
                + self._selected_reset_retired_payment_totals[0]
            )
            | (
                self._payment_totals[1]
                != payment_count_1
                + self._closed_total
                + self._selected_reset_retired_payment_totals[1]
            )
        ).reshape(1)
        # Previous-paid is folded into mailbox ownership: it is a durable
        # projection of one closed mailbox, not a new lifecycle/counter lane.
        mailbox_owner_bad = mailbox_owner_bad | (
            previous_paid_bad.unsqueeze(1)
            & (self._mailbox_slot_ids == 0)
        )
        masks = (
            flight_state_bad,
            mailbox_state_bad,
            flight_owner_bad,
            mailbox_owner_bad,
            reservation_bad,
            retained_bad,
            retirement_bad,
            contact_bad,
            observation_bad,
            stamp_bad,
            consumer_bad,
            view_payment_bad,
            payment_bad,
            canonical_bad,
            replay_bad,
            counter_bad,
        )
        return [mask.to(torch.int64).sum() for mask in masks]

    def _action_epoch_invariant_counts(self) -> list[torch.Tensor]:
        """Direct-lane invariants over typed identity and numeric facts only."""

        flight_active = self._flight_state.ne(FLIGHT_EMPTY)
        mailbox_active = self._mailbox_state.ne(MAILBOX_EMPTY)
        mailbox_record = self._mailbox_history_valid
        flight_state_bad = ~(
            self._flight_state.eq(FLIGHT_EMPTY)
            | self._flight_state.eq(FLIGHT_INBOUND)
            | self._flight_state.eq(FLIGHT_OPEN)
            | self._flight_state.eq(FLIGHT_SETTLED_RETAINED)
        )
        mailbox_state_bad = ~(
            self._mailbox_state.eq(MAILBOX_EMPTY)
            | self._mailbox_state.eq(MAILBOX_SETTLED_UNPAID)
            | self._mailbox_state.eq(MAILBOX_PARTIALLY_PAID)
            | self._mailbox_state.eq(MAILBOX_PAID)
        )
        flight_typed_bad = flight_active & (
            ~self._flight_action_epoch
            | self._flight_action_uid.lt(1)
            | self._flight_action_uid.gt(_MAX_ACTION_UID)
            | self._flight_action_slot.lt(0)
            | self._flight_reset_generation.lt(1)
            | self._flight_shot_index.lt(1)
            | self._flight_task_identity.lt(1)
            | self._flight_outcome_identity.lt(1)
            | self._flight_ball_identity.lt(1)
            | self._flight_publication_ordinal.lt(0)
            | self._flight_ball_generation.lt(0)
            | ~torch.isfinite(self._flight_target_xy_m).all(dim=-1)
            | self._flight_target_xy_m[..., 0].lt(
                self.profile.opponent_table_x_min_m
            )
            | self._flight_target_xy_m[..., 0].gt(
                self.profile.opponent_table_x_max_m
            )
            | self._flight_target_xy_m[..., 1].lt(self.profile.table_y_min_m)
            | self._flight_target_xy_m[..., 1].gt(self.profile.table_y_max_m)
            | self._flight_reveal_control_step.lt(0)
            | self._flight_contact_deadline_control_step.lt(
                self._flight_reveal_control_step
            )
            | self._flight_crossing_horizon_control_step.lt(
                self._flight_contact_deadline_control_step
            )
        )
        mailbox_typed_bad = mailbox_record & (
            ~self._mailbox_action_epoch
            | self._mailbox_action_uid.lt(1)
            | self._mailbox_action_uid.gt(_MAX_ACTION_UID)
            | self._mailbox_action_slot.lt(0)
            | self._mailbox_reset_generation.lt(1)
            | self._mailbox_shot_index.lt(1)
            | self._mailbox_task_identity.lt(1)
            | self._mailbox_outcome_identity.lt(1)
            | self._mailbox_ball_identity.lt(1)
            | self._mailbox_publication_ordinal.lt(0)
            | self._mailbox_ball_generation.lt(0)
            | self._mailbox_observation_ordinal.lt(0)
            | self._mailbox_settlement_control_step.lt(0)
            | self._mailbox_settlement_physics_substep.lt(0)
            | ~torch.isfinite(self._mailbox_target_xy_m).all(dim=-1)
            | ~torch.isfinite(self._mailbox_placement_error_m)
            | self._mailbox_placement_error_m.lt(0.0)
            | self._mailbox_c10_family_code.ne(self._c10_family_code)
            | self._mailbox_placement_treatment_gain.ne(
                self._placement_treatment_gain
            )
        )
        mailbox_owner_bad = (
            (mailbox_active & ~mailbox_record)
            | mailbox_typed_bad
            | (
                mailbox_active
                & (
                    ~self._mailbox_reserved
                    | self._mailbox_reservation_generation.ne(
                        self._mailbox_ball_generation
                    )
                    | self._mailbox_reserved_flight_slot.lt(0)
                    | self._mailbox_reserved_flight_slot.ge(
                        self.flight_slot_capacity
                    )
                )
            )
        )
        association = self._action_epoch_flight_mailbox_association()
        flight_count = association.to(torch.int64).sum(dim=1)
        mailbox_count = association.to(torch.int64).sum(dim=2)
        preflight = self._flight_state.eq(FLIGHT_INBOUND) | self._flight_state.eq(
            FLIGHT_OPEN
        )
        retained = self._flight_state.eq(FLIGHT_SETTLED_RETAINED)
        mailbox_preflight = (association & preflight.unsqueeze(1)).to(
            torch.int64
        ).sum(dim=2)
        mailbox_retained = (association & retained.unsqueeze(1)).to(
            torch.int64
        ).sum(dim=2)
        reserved_empty = self._mailbox_reserved & ~mailbox_active
        reservation_bad = (
            (flight_active & flight_count.ne(1)).to(torch.int64).sum()
            + (
                reserved_empty
                & (
                    mailbox_count.ne(1)
                    | mailbox_preflight.ne(1)
                    | mailbox_retained.ne(0)
                )
            ).to(torch.int64).sum()
        ).reshape(1)
        active_unretired = mailbox_active & ~self._mailbox_physical_retired
        active_retired = mailbox_active & self._mailbox_physical_retired
        retained_bad = (
            (
                retained
                & (
                    flight_count.ne(1)
                    | ~self._action_epoch_flight_retained_mailbox_active()
                )
            ).to(torch.int64).sum()
            + (
                active_unretired
                & (
                    mailbox_count.ne(1)
                    | mailbox_retained.ne(1)
                    | mailbox_preflight.ne(0)
                )
            ).to(torch.int64).sum()
            + (
                active_retired
                & (
                    mailbox_count.ne(0)
                    | mailbox_retained.ne(0)
                    | mailbox_preflight.ne(0)
                )
            ).to(torch.int64).sum()
        ).reshape(1)
        retirement_bad = (
            (self._mailbox_physical_retired & ~mailbox_active).to(torch.int64).sum()
            + (flight_active & self._flight_physical_retired).to(torch.int64).sum()
            + ((~flight_active) & ~self._flight_physical_retired).to(torch.int64).sum()
        ).reshape(1)
        contact_bad = (
            (self._flight_state.eq(FLIGHT_OPEN) & ~self._flight_contact_valid)
            | (self._flight_state.eq(FLIGHT_INBOUND) & self._flight_contact_valid)
        )
        observation_bad = flight_active & (
            self._flight_observation_ordinal.lt(-1)
            | (
                self._flight_observation_ordinal.ge(0)
                & self._flight_last_observation_control.lt(0)
            )
        )
        flight_stamp_bad = flight_active & (
            self._flight_contact_valid
            & self._flight_contact_stamp_control.lt(0)
        )
        mailbox_stamp_bad = mailbox_record & (
            self._mailbox_contact_valid
            & self._mailbox_source_control_step.lt(0)
        )
        stamp_bad = (
            flight_stamp_bad.to(torch.int64).sum()
            + mailbox_stamp_bad.to(torch.int64).sum()
        ).reshape(1)
        consumer_bad = mailbox_record & (
            self._mailbox_paid_mask.ne(0)
            | self._mailbox_state.ne(MAILBOX_SETTLED_UNPAID)
        )
        view_payment_bad = mailbox_record & (
            self._mailbox_view_epoch.ne(-1).any(dim=-1)
            | self._mailbox_payment_epoch.ne(-1).any(dim=-1)
        )
        payment_bad = mailbox_record & self._mailbox_payment_values.ne(0.0).any(
            dim=-1
        )
        canonical_bad = self._action_epoch_canonical_invariant_bad(mailbox_record)
        previous_typed_bad = self._previous_paid_valid & (
            ~self._previous_paid_action_epoch
            | self._previous_paid_action_uid.lt(1)
            | self._previous_paid_action_uid.gt(_MAX_ACTION_UID)
            | self._previous_paid_action_slot.lt(0)
            | self._previous_paid_reset_generation.lt(1)
            | self._previous_paid_shot_index.lt(1)
            | self._previous_paid_task_identity.lt(1)
            | self._previous_paid_outcome_identity.lt(1)
            | self._previous_paid_ball_identity.lt(1)
            | self._previous_paid_publication_ordinal.lt(0)
            | self._previous_paid_ball_generation.lt(0)
            | self._previous_paid_observation_ordinal.lt(0)
            | self._previous_paid_settlement_control_step.lt(0)
            | self._previous_paid_payment_step.lt(
                self._previous_paid_settlement_control_step
            )
            | self._previous_paid_payment_step.ne(
                self._previous_paid_payment_step_highwater
            )
            | ~torch.isfinite(self._previous_paid_target_error_m)
            | self._previous_paid_target_error_m.lt(0.0)
            | ~torch.isfinite(self._previous_paid_target_xy_m).all(dim=-1)
        )
        replay_typed_bad = self._replay_valid & (
            ~self._replay_action_epoch
            | self._replay_reset_generation.lt(1)
            | self._replay_swing_generation.lt(0)
            | self._replay_action_uid.lt(1)
            | self._replay_action_uid.gt(_MAX_ACTION_UID)
            | self._replay_action_slot.lt(0)
            | self._replay_shot_index.lt(1)
            | self._replay_task_identity.lt(1)
            | self._replay_outcome_identity.lt(1)
            | self._replay_ball_identity.lt(1)
            | self._replay_publication_ordinal.lt(0)
            | self._replay_reset_generation.gt(self._reset_generation_highwater)
        )
        replay_owner = (
            flight_active
            & self._flight_action_epoch
            & self._flight_reset_generation.eq(
                self._replay_reset_generation.unsqueeze(1)
            )
            & self._flight_ball_generation.eq(
                self._replay_swing_generation.unsqueeze(1)
            )
            & self._flight_action_uid.eq(self._replay_action_uid.unsqueeze(1))
            & self._flight_action_slot.eq(self._replay_action_slot.unsqueeze(1))
            & self._flight_task_identity.eq(self._replay_task_identity.unsqueeze(1))
            & self._flight_outcome_identity.eq(
                self._replay_outcome_identity.unsqueeze(1)
            )
            & self._flight_ball_identity.eq(self._replay_ball_identity.unsqueeze(1))
            & self._flight_publication_ordinal.eq(
                self._replay_publication_ordinal.unsqueeze(1)
            )
        ).any(dim=1) | (
            mailbox_record
            & self._mailbox_action_epoch
            & self._mailbox_reset_generation.eq(
                self._replay_reset_generation.unsqueeze(1)
            )
            & self._mailbox_ball_generation.eq(
                self._replay_swing_generation.unsqueeze(1)
            )
            & self._mailbox_action_uid.eq(self._replay_action_uid.unsqueeze(1))
            & self._mailbox_action_slot.eq(self._replay_action_slot.unsqueeze(1))
            & self._mailbox_task_identity.eq(
                self._replay_task_identity.unsqueeze(1)
            )
            & self._mailbox_outcome_identity.eq(
                self._replay_outcome_identity.unsqueeze(1)
            )
            & self._mailbox_ball_identity.eq(
                self._replay_ball_identity.unsqueeze(1)
            )
            & self._mailbox_publication_ordinal.eq(
                self._replay_publication_ordinal.unsqueeze(1)
            )
        ).any(dim=1) | (
            self._previous_paid_valid
            & self._previous_paid_action_epoch
            & self._previous_paid_reset_generation.eq(self._replay_reset_generation)
            & self._previous_paid_ball_generation.eq(self._replay_swing_generation)
            & self._previous_paid_action_uid.eq(self._replay_action_uid)
            & self._previous_paid_action_slot.eq(self._replay_action_slot)
            & self._previous_paid_task_identity.eq(self._replay_task_identity)
            & self._previous_paid_outcome_identity.eq(self._replay_outcome_identity)
            & self._previous_paid_ball_identity.eq(self._replay_ball_identity)
            & self._previous_paid_publication_ordinal.eq(
                self._replay_publication_ordinal
            )
        )
        replay_bad = replay_typed_bad | (self._replay_valid & ~replay_owner)
        nonempty_flights = flight_active.to(torch.int64).sum()
        nonempty_mailboxes = mailbox_active.to(torch.int64).sum()
        counter_bad = (
            self._mutation_version.lt(0)
            | self._previous_paid_payment_step_highwater.lt(-1).any()
            | self._fault_event_counts.lt(0).any()
            | self._installed_total.lt(0)
            | self._settled_total.lt(0)
            | self._retired_total.lt(0)
            | self._closed_total.lt(0)
            | self._settled_total.gt(self._installed_total)
            | self._retired_total.gt(self._settled_total)
            | self._closed_total.gt(self._retired_total)
            | (
                self._installed_total
                - self._retired_total
                - self._selected_reset_retired_flight_total
            ).ne(nonempty_flights)
            | (
                self._settled_total
                - self._closed_total
                - self._selected_reset_closed_mailbox_total
            ).ne(nonempty_mailboxes)
            | self._payment_totals[0].ne(
                self._closed_total + self._selected_reset_retired_payment_totals[0]
            )
            | self._payment_totals[1].ne(
                self._closed_total + self._selected_reset_retired_payment_totals[1]
            )
        ).reshape(1)
        mailbox_owner_bad |= previous_typed_bad.unsqueeze(1) & self._mailbox_slot_ids.eq(0)
        masks = (
            flight_state_bad,
            mailbox_state_bad,
            flight_typed_bad,
            mailbox_owner_bad,
            reservation_bad,
            retained_bad,
            retirement_bad,
            contact_bad,
            observation_bad,
            stamp_bad,
            consumer_bad,
            view_payment_bad,
            payment_bad,
            canonical_bad,
            replay_bad,
            counter_bad,
        )
        return [mask.to(torch.int64).sum() for mask in masks]

    def _flight_mailbox_association(self) -> torch.Tensor:
        """Exact reservation edges as ``[N, Km, Kf]`` without reduction."""

        mailbox_slot = self._mailbox_slot_ids.unsqueeze(2)
        flight_slot = self._flight_slot_ids.unsqueeze(1)
        base = (
            self._mailbox_reserved.unsqueeze(2)
            & (self._flight_state != FLIGHT_EMPTY).unsqueeze(1)
            & (
                self._flight_mailbox_slot.unsqueeze(1)
                == mailbox_slot
            )
            & (
                self._mailbox_reserved_flight_slot.unsqueeze(2)
                == flight_slot
            )
            & (
                self._mailbox_reservation_generation.unsqueeze(2)
                == self._flight_ball_generation.unsqueeze(1)
            )
        )
        legacy = (
            ~self._mailbox_action_epoch.unsqueeze(2)
            & ~self._flight_action_epoch.unsqueeze(1)
            & torch.eq(
                self._mailbox_reservation_token.unsqueeze(2),
                self._flight_full_key_sha256.unsqueeze(1),
            ).all(dim=-1)
        )
        mailbox_key = self._mailbox_action_epoch_shot_key()
        flight_key = self._flight_action_epoch_shot_key()
        typed_key_match = _row_identity.action_epoch_shot_key_equal(
            _row_identity.ActionEpochShotKey(
                **{
                    field.name: getattr(mailbox_key, field.name)
                    .unsqueeze(2)
                    .expand(
                        self.num_envs,
                        self.mailbox_capacity,
                        self.flight_slot_capacity,
                    )
                    .contiguous()
                    for field in fields(_row_identity.ActionEpochShotKey)
                }
            ),
            _row_identity.ActionEpochShotKey(
                **{
                    field.name: getattr(flight_key, field.name)
                    .unsqueeze(1)
                    .expand(
                        self.num_envs,
                        self.mailbox_capacity,
                        self.flight_slot_capacity,
                    )
                    .contiguous()
                    for field in fields(_row_identity.ActionEpochShotKey)
                }
            ),
        )
        direct = (
            self._mailbox_action_epoch.unsqueeze(2)
            & self._flight_action_epoch.unsqueeze(1)
            & self._mailbox_publication_ordinal.unsqueeze(2).eq(
                self._flight_publication_ordinal.unsqueeze(1)
            )
            & typed_key_match
        )
        return base & (legacy | direct)

    def _action_epoch_flight_mailbox_association(self) -> torch.Tensor:
        """Direct reservation edges using the complete typed identity."""

        mailbox_slot = self._mailbox_slot_ids.unsqueeze(2)
        flight_slot = self._flight_slot_ids.unsqueeze(1)
        mailbox_key = self._mailbox_action_epoch_shot_key()
        flight_key = self._flight_action_epoch_shot_key()
        typed_key_match = _row_identity.action_epoch_shot_key_equal(
            _row_identity.ActionEpochShotKey(
                **{
                    field.name: getattr(mailbox_key, field.name)
                    .unsqueeze(2)
                    .expand(
                        self.num_envs,
                        self.mailbox_capacity,
                        self.flight_slot_capacity,
                    )
                    .contiguous()
                    for field in fields(_row_identity.ActionEpochShotKey)
                }
            ),
            _row_identity.ActionEpochShotKey(
                **{
                    field.name: getattr(flight_key, field.name)
                    .unsqueeze(1)
                    .expand(
                        self.num_envs,
                        self.mailbox_capacity,
                        self.flight_slot_capacity,
                    )
                    .contiguous()
                    for field in fields(_row_identity.ActionEpochShotKey)
                }
            ),
        )
        return (
            self._mailbox_reserved.unsqueeze(2)
            & self._mailbox_action_epoch.unsqueeze(2)
            & self._flight_action_epoch.unsqueeze(1)
            & self._flight_state.ne(FLIGHT_EMPTY).unsqueeze(1)
            & self._flight_mailbox_slot.unsqueeze(1).eq(mailbox_slot)
            & self._mailbox_reserved_flight_slot.unsqueeze(2).eq(flight_slot)
            & self._mailbox_reservation_generation.unsqueeze(2).eq(
                self._flight_ball_generation.unsqueeze(1)
            )
            & self._mailbox_publication_ordinal.unsqueeze(2).eq(
                self._flight_publication_ordinal.unsqueeze(1)
            )
            & typed_key_match
        )

    def _action_epoch_flight_retained_mailbox_active(self) -> torch.Tensor:
        """Direct retained-flight join without consulting legacy key planes."""

        safe = self._flight_mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        return (
            self._action_epoch_settled_mailbox_ok()
            & ~torch.gather(self._mailbox_physical_retired, 1, safe)
            & torch.gather(self._mailbox_reservation_generation, 1, safe).eq(
                self._flight_ball_generation
            )
            & torch.gather(self._mailbox_target_xy_m, 1, safe.unsqueeze(-1).expand(
                self._flight_shape + (2,)
            )).eq(self._flight_target_xy_m).all(dim=-1)
        )

    def _flight_retained_mailbox_active(self) -> torch.Tensor:
        safe = self._flight_mailbox_slot.clamp(0, self.mailbox_capacity - 1)
        state = torch.gather(self._mailbox_state.to(torch.int64), 1, safe)
        reserved = torch.gather(self._mailbox_reserved, 1, safe)
        physical_retired = torch.gather(
            self._mailbox_physical_retired, 1, safe
        )
        generation = torch.gather(self._mailbox_ball_generation, 1, safe)
        reservation_generation = torch.gather(
            self._mailbox_reservation_generation, 1, safe
        )
        reserved_flight_slot = torch.gather(
            self._mailbox_reserved_flight_slot, 1, safe
        )
        token = torch.gather(
            self._mailbox_full_key_sha256,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        receipt = torch.gather(
            self._mailbox_full_key_receipt_sha256,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        committed = torch.gather(
            self._mailbox_committed_reveal_sha256,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        install_receipt = torch.gather(
            self._mailbox_install_receipt_sha256,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        reservation_token = torch.gather(
            self._mailbox_reservation_token,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        task_identity = torch.gather(
            self._mailbox_task_identity_token,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
        )
        target_xy = torch.gather(
            self._mailbox_target_xy_m,
            1,
            safe.unsqueeze(-1).expand(self._flight_shape + (2,)),
        )
        matches = (
            (self._flight_mailbox_slot >= 0)
            & (self._flight_mailbox_slot < self.mailbox_capacity)
            & (state != MAILBOX_EMPTY)
            & reserved
            & ~physical_retired
            & (generation == self._flight_ball_generation)
            & (reservation_generation == self._flight_ball_generation)
            & (
                reserved_flight_slot
                == self._flight_slot_ids.expand(self._flight_shape)
            )
            & torch.eq(token, self._flight_full_key_sha256).all(dim=-1)
            & torch.eq(
                reservation_token, self._flight_full_key_sha256
            ).all(dim=-1)
            & torch.eq(receipt, self._flight_full_key_receipt_sha256).all(dim=-1)
            & torch.eq(committed, self._flight_committed_reveal_sha256).all(dim=-1)
            & torch.eq(
                install_receipt, self._flight_install_receipt_sha256
            ).all(dim=-1)
            & torch.eq(
                task_identity, self._flight_task_identity_token
            ).all(dim=-1)
            & torch.eq(target_xy, self._flight_target_xy_m).all(dim=-1)
        )
        for name in _INT_KEY_FIELDS:
            matches = matches & (
                torch.gather(self._mailbox_key_ints[name], 1, safe)
                == self._flight_key_ints[name]
            )
        for name in _DIGEST_KEY_FIELDS:
            mailbox_value = torch.gather(
                self._mailbox_key_digests[name],
                1,
                safe.unsqueeze(-1).expand(self._flight_shape + (TOKEN_BYTES,)),
            )
            matches = matches & torch.eq(
                mailbox_value, self._flight_key_digests[name]
            ).all(dim=-1)
        return matches

    def _canonical_invariant_bad(self, mailbox_active: torch.Tensor) -> torch.Tensor:
        policy = mailbox_active & self._mailbox_policy_eligible
        infra = mailbox_active & ~self._mailbox_policy_eligible
        score = self._score_policy_subset(
            prefix=self._mailbox_shape,
            policy_mask=policy,
            task_identity_token=self._mailbox_task_identity_token,
            target_xy_m=self._mailbox_target_xy_m,
            contact_valid=self._mailbox_contact_valid,
            crossing_present=self._mailbox_crossing_present,
            crossing_valid=self._mailbox_crossing_valid,
            crossing_nonfinite=self._mailbox_crossing_nonfinite,
            crossing_xy_m=self._mailbox_crossing_xy_m,
            net_crossed=self._mailbox_net_crossed,
            net_clear=self._mailbox_net_clear,
        )
        reason_bad = policy & (
            self._mailbox_canonical_reason_code != score.reason_code
        )
        score_bad = policy & (
            ~torch.eq(self._mailbox_canonical_total, score.total)
            | ~torch.eq(
                self._mailbox_on_opponent_table,
                score.on_opponent_table,
            )
            | ~torch.eq(
                self._mailbox_placement_error_m,
                score.placement_error_m,
            )
            | ~torch.eq(
                self._mailbox_broad_kernel,
                score.broad_kernel,
            )
            | ~torch.eq(
                self._mailbox_narrow_kernel,
                score.narrow_kernel,
            )
            | ~torch.eq(
                self._mailbox_blended_kernel,
                score.blended_kernel,
            )
            | ~torch.eq(
                self._mailbox_table_gate,
                score.table_gate,
            )
        )
        infra_bad = infra & (
            (self._mailbox_canonical_reason_code != CANONICAL_REASON_NOT_SCORED)
            | (self._mailbox_canonical_total != 0.0)
            | self._mailbox_common_on_table
        )
        common_expected = (
            policy
            & self._mailbox_contact_valid
            & self._mailbox_crossing_valid
            & self._mailbox_net_crossed
            & self._mailbox_net_clear
            & self._mailbox_on_opponent_table
        )
        common_bad = mailbox_active & (
            self._mailbox_common_on_table != common_expected
        )
        return reason_bad | score_bad | infra_bad | common_bad

    def _action_epoch_canonical_invariant_bad(
        self, mailbox_active: torch.Tensor
    ) -> torch.Tensor:
        """Recompute direct scores without legacy task-token authority."""

        policy = mailbox_active & self._mailbox_policy_eligible
        infra = mailbox_active & ~self._mailbox_policy_eligible
        score = self._score_action_epoch_policy_subset(
            policy_mask=policy,
            target_xy_m=self._mailbox_target_xy_m,
            contact_valid=self._mailbox_contact_valid,
            crossing_present=self._mailbox_crossing_present,
            crossing_valid=self._mailbox_crossing_valid,
            crossing_nonfinite=self._mailbox_crossing_nonfinite,
            crossing_xy_m=self._mailbox_crossing_xy_m,
            net_crossed=self._mailbox_net_crossed,
            net_clear=self._mailbox_net_clear,
        )
        reason_bad = policy & self._mailbox_canonical_reason_code.ne(
            score.reason_code
        )
        score_bad = policy & (
            self._mailbox_canonical_total.ne(score.total)
            | self._mailbox_on_opponent_table.ne(score.on_opponent_table)
            | self._mailbox_placement_error_m.ne(score.placement_error_m)
            | self._mailbox_broad_kernel.ne(score.broad_kernel)
            | self._mailbox_narrow_kernel.ne(score.narrow_kernel)
            | self._mailbox_blended_kernel.ne(score.blended_kernel)
            | self._mailbox_table_gate.ne(score.table_gate)
        )
        infra_bad = infra & (
            self._mailbox_canonical_reason_code.ne(CANONICAL_REASON_NOT_SCORED)
            | self._mailbox_canonical_total.ne(0.0)
            | self._mailbox_common_on_table
        )
        common_expected = (
            policy
            & self._mailbox_contact_valid
            & self._mailbox_crossing_valid
            & self._mailbox_net_crossed
            & self._mailbox_net_clear
            & self._mailbox_on_opponent_table
        )
        return (
            reason_bad
            | score_bad
            | infra_bad
            | (mailbox_active & self._mailbox_common_on_table.ne(common_expected))
        )

    @staticmethod
    def _r06_global_owner_row_values(owner_row: object) -> dict[str, int]:
        if getattr(owner_row, "owner_kind", None) != _R06_GLOBAL_DRAIN_OWNER_KIND:
            raise LandingOutcomeDeviceError(
                "global drain owner row is not the R06 row"
            )
        raw = getattr(owner_row, "values", None)
        if not isinstance(raw, tuple):
            raise LandingOutcomeDeviceError(
                "global drain R06 row values must be a tuple"
            )
        names = []
        result: dict[str, int] = {}
        for item in raw:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
            ):
                raise LandingOutcomeDeviceError(
                    "global drain R06 row contains a non-scalar field"
                )
            name, value = item
            names.append(name)
            result[name] = value
        if tuple(names) != R06_GLOBAL_DRAIN_FIELD_NAMES:
            raise LandingOutcomeDeviceError(
                "global drain R06 row schema differs"
            )
        return result

    def _r06_global_drain_values(self) -> torch.Tensor:
        fault_counts = torch.stack(self._combined_fault_counts())
        invariant_counts = torch.stack(self._invariant_counts())
        flight_state_counts = torch.stack(
            tuple(
                (self._flight_state == state).to(torch.int64).sum()
                for state in (
                    FLIGHT_EMPTY,
                    FLIGHT_INBOUND,
                    FLIGHT_OPEN,
                    FLIGHT_SETTLED_RETAINED,
                )
            )
        )
        mailbox_state_counts = torch.stack(
            tuple(
                (self._mailbox_state == state).to(torch.int64).sum()
                for state in (
                    MAILBOX_EMPTY,
                    MAILBOX_SETTLED_UNPAID,
                    MAILBOX_PARTIALLY_PAID,
                    MAILBOX_PAID,
                )
            )
        )
        values = torch.cat(
            (
                self._mutation_version.reshape(1),
                fault_counts.sum().reshape(1),
                invariant_counts.sum().reshape(1),
                self._terminal_resolution_total.reshape(1),
                self._shared_normal_retire_total.reshape(1),
                self._r06_only_orphan_retire_total.reshape(1),
                self._shared_normal_retire_key_summaries,
                fault_counts,
                flight_state_counts,
                mailbox_state_counts,
                invariant_counts,
                self._installed_total.reshape(1),
                self._settled_total.reshape(1),
                self._retired_total.reshape(1),
                self._payment_totals,
                self._closed_total.reshape(1),
            )
        ).contiguous()
        if values.shape != (len(R06_GLOBAL_DRAIN_FIELD_NAMES),):
            raise LandingOutcomeDeviceError(
                "R06 global drain row width differs"
            )
        return values

    @staticmethod
    def _r06_checkpoint_tensor_receipt(
        tensor: torch.Tensor,
    ) -> tuple[object, ...]:
        """Seal exact live tensor identity/layout/version without a D2H read."""

        if not torch.is_tensor(tensor):
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live epoch contains a non-tensor"
            )
        try:
            version = tensor._version
        except RuntimeError as exc:
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live tensor has no observable mutation epoch"
            ) from exc
        if type(version) is not int:
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live tensor mutation epoch differs"
            )
        return (
            tensor,
            id(tensor),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.data_ptr()),
            version,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            tensor.dtype,
            tensor.device,
        )

    @staticmethod
    def _r06_checkpoint_tensor_matches_receipt(
        tensor: torch.Tensor,
        receipt: tuple[object, ...],
    ) -> bool:
        """Check one retained live tensor epoch without reading its value."""

        return (
            type(receipt) is tuple
            and len(receipt) == 10
            and receipt[0] is tensor
            and id(tensor) == receipt[1]
            and int(tensor.untyped_storage().data_ptr()) == receipt[2]
            and int(tensor.data_ptr()) == receipt[3]
            and type(tensor._version) is int
            and tensor._version == receipt[4]
            and tuple(tensor.shape) == receipt[5]
            and tuple(tensor.stride()) == receipt[6]
            and int(tensor.storage_offset()) == receipt[7]
            and tensor.dtype == receipt[8]
            and tensor.device == receipt[9]
        )

    def _r06_checkpoint_live_tensor_receipts(
        self,
    ) -> tuple[tuple[str, tuple[object, ...]], ...]:
        """Seal every persistent tensor that can affect future R06 behavior."""

        tensors = self._checkpoint_tensors()
        names = tuple(sorted(tensors))
        if not names or len(names) != len(set(names)):
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live tensor inventory differs"
            )
        return tuple(
            (name, self._r06_checkpoint_tensor_receipt(tensors[name]))
            for name in names
        )

    def _r06_checkpoint_leaf_idle(self) -> bool:
        """Return the complete local idle/debt fact without peer inspection."""

        return not any(
            value is not None
            for value in (
                self._active_r06_global_drain,
                self._active_reveal_prepare_lease,
                self._active_physical_retire_lease,
                self._active_selected_reset_lease,
                self._latest_post_physics_settlement,
                self._active_post_physics_contact_authority,
                self._latest_selected_reset_completion,
                self._active_full_mdp_reward_cycle_identity,
                True if self._full_mdp_reward_close_debt else None,
                True if self._full_mdp_reward_poisoned else None,
            )
        )

    def _mint_r06_checkpoint_live_mutation_projection(
        self,
        *,
        global_receipt: object,
        portable: LandingOutcomeBoundaryReceipt,
    ) -> R06CheckpointLiveMutationProjection:
        """Mint the process-local R10 handle only after the exact leaf ACK."""

        if (
            not self._r06_checkpoint_leaf_idle()
            or self._poisoned
            or self._r06_global_drain_poisoned
            or portable.mutation_version < 0
        ):
            raise LandingOutcomeDeviceError(
                "R06 exact global ACK did not end on an idle healthy leaf"
            )
        projection = object.__new__(R06CheckpointLiveMutationProjection)
        payload = _R06CheckpointLiveMutationPayload(
            owner_ref=weakref.ref(self),
            source_global_receipt_ref=weakref.ref(global_receipt),
            mutation_version=portable.mutation_version,
            update_index=portable.update_index,
            drain_sequence=portable.drain_sequence,
            live_tensor_receipts=self._r06_checkpoint_live_tensor_receipts(),
        )
        _R06_CHECKPOINT_LIVE_MUTATION_REGISTRY[projection] = payload
        self._latest_checkpoint_live_mutation_projection = projection
        return projection

    def project_checkpoint_live_mutation(
        self,
    ) -> R06CheckpointLiveMutationProjection:
        """Issue the one opaque R10 handle for the latest exact global ACK.

        The global receipt is retained only because this leaf received it from
        the construction-bound drain owner and validated its exact ACK
        authority.  This method never asks the global owner for state and it
        refuses to mint twice from one ACK.
        """

        self._require_formal_only("project an R10 checkpoint mutation")
        try:
            if __package__:
                from . import action_ball_full_mdp_ppo_drain as drain
            else:
                from whole_body_tracking.tasks.tracking.mdp import (
                    action_ball_full_mdp_ppo_drain as drain,
                )
        except (ImportError, ModuleNotFoundError):
            import action_ball_full_mdp_ppo_drain as drain

        current = self._latest_checkpoint_live_mutation_projection
        if current is not None:
            return self.require_owned_checkpoint_live_mutation_projection(current)
        global_receipt = self._latest_global_drain_receipt
        portable = self._latest_receipt
        if (
            global_receipt is None
            or type(global_receipt) is not drain.PreOptimizerPpoBoundaryReceipt
            or global_receipt.acknowledged is not True
            or global_receipt is self._r06_checkpoint_consumed_global_receipt
            or portable is None
            or portable.update_index != global_receipt.update_index
            or portable.drain_sequence != self._drain_sequence
            or portable.update_index != self._last_drained_update_index
            or not self._r06_global_drain_adopted
            or self._poisoned
            or self._r06_global_drain_poisoned
            or not self._r06_checkpoint_leaf_idle()
        ):
            raise LandingOutcomeDeviceError(
                "R06 R10 projection requires one fresh exact global ACK and an idle healthy leaf"
            )
        return self._mint_r06_checkpoint_live_mutation_projection(
            global_receipt=global_receipt,
            portable=portable,
        )

    def current_checkpoint_mutation_projection(
        self,
        boundary: object,
        owner_kind: str,
    ) -> object:
        """Project R06's current live version from its latest exact global ACK.

        This callback does not receive a receipt, expected mutation number, or
        global-drain owner.  It consults only the leaf's retained owner-issued
        handle, verifies the current local tensor epoch and debt/idle state,
        and emits R10's common primitive projection.  A cold restore clears
        the process-local handle, so restored bytes require a new global ACK.
        """

        self._require_formal_only("participate in R10 checkpoint projection")
        try:
            import action_ball_full_mdp_checkpoint as checkpoint
            if __package__:
                from . import action_ball_full_mdp_ppo_drain as drain
            else:
                from whole_body_tracking.tasks.tracking.mdp import (
                    action_ball_full_mdp_ppo_drain as drain,
                )
        except (ImportError, ModuleNotFoundError):
            import action_ball_full_mdp_checkpoint as checkpoint
            import action_ball_full_mdp_ppo_drain as drain

        if owner_kind != _R06_GLOBAL_DRAIN_OWNER_KIND:
            raise LandingOutcomeDeviceError("R06 R10 leaf role differs")
        if type(boundary) is not checkpoint.CheckpointBoundary:
            raise LandingOutcomeDeviceError(
                "R06 R10 projection requires the exact checkpoint boundary"
            )
        checkpoint.validate_checkpoint_boundary(boundary)
        projection = self.project_checkpoint_live_mutation()
        self.require_owned_checkpoint_live_mutation_projection(projection)
        try:
            payload = _R06_CHECKPOINT_LIVE_MUTATION_REGISTRY.get(projection)
        except TypeError:
            payload = None
        global_receipt = (
            None if payload is None else payload.source_global_receipt_ref()
        )
        portable = self._latest_receipt
        tensors = self._checkpoint_tensors()
        if (
            type(projection) is not R06CheckpointLiveMutationProjection
            or payload is None
            or payload.owner_ref() is not self
            or global_receipt is None
            or global_receipt is not self._latest_global_drain_receipt
            or type(global_receipt) is not drain.PreOptimizerPpoBoundaryReceipt
            or global_receipt.acknowledged is not True
            or portable is None
            or portable.mutation_version != payload.mutation_version
            or portable.update_index != payload.update_index
            or portable.drain_sequence != payload.drain_sequence
            or boundary.update_index != payload.update_index
            or self._last_drained_update_index != payload.update_index
            or self._drain_sequence != payload.drain_sequence
            or not self._r06_global_drain_adopted
            or self._poisoned
            or self._r06_global_drain_poisoned
            or not self._r06_checkpoint_leaf_idle()
            or tuple(name for name, _receipt in payload.live_tensor_receipts)
            != tuple(sorted(tensors))
            or any(
                not self._r06_checkpoint_tensor_matches_receipt(
                    tensors[name], tensor_receipt
                )
                for name, tensor_receipt in payload.live_tensor_receipts
            )
        ):
            raise LandingOutcomeDeviceError(
                "R06 R10 projection lacks its exact latest global ACK, live epoch, or idle debt closure"
            )
        result = checkpoint.PpoDrainLeafLiveMutationProjection(
            schema_version=checkpoint.SCHEMA_VERSION,
            kind="action_ball_r10_leaf_live_mutation_projection_v1",
            owner_kind=_R06_GLOBAL_DRAIN_OWNER_KIND,
            mutation_version=payload.mutation_version,
        )
        self.consume_owned_checkpoint_live_mutation_projection(projection)
        return result

    def require_owned_checkpoint_live_mutation_projection(
        self,
        projection: object,
    ) -> R06CheckpointLiveMutationProjection:
        """Validate the exact latest owner-issued handle without D2H.

        This method returns only the same opaque handle.  It does not expose a
        mutation value, receipt, digest, or tensor alias and therefore cannot
        substitute for the independent runner highwater joined by R10.
        """

        try:
            payload = _R06_CHECKPOINT_LIVE_MUTATION_REGISTRY.get(projection)
        except TypeError:
            payload = None
        global_receipt = (
            None if payload is None else payload.source_global_receipt_ref()
        )
        tensors = self._checkpoint_tensors()
        if (
            type(projection) is not R06CheckpointLiveMutationProjection
            or projection is not self._latest_checkpoint_live_mutation_projection
            or payload is None
            or payload.owner_ref() is not self
            or global_receipt is None
            or global_receipt is not self._latest_global_drain_receipt
            or getattr(global_receipt, "acknowledged", None) is not True
            or self._latest_receipt is None
            or self._latest_receipt.mutation_version
            != payload.mutation_version
            or self._latest_receipt.update_index != payload.update_index
            or self._latest_receipt.drain_sequence != payload.drain_sequence
            or self._last_drained_update_index != payload.update_index
            or self._drain_sequence != payload.drain_sequence
            or not self._r06_global_drain_adopted
            or self._poisoned
            or self._r06_global_drain_poisoned
            or not self._r06_checkpoint_leaf_idle()
            or tuple(name for name, _receipt in payload.live_tensor_receipts)
            != tuple(sorted(tensors))
            or any(
                not self._r06_checkpoint_tensor_matches_receipt(
                    tensors[name], tensor_receipt
                )
                for name, tensor_receipt in payload.live_tensor_receipts
            )
        ):
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live projection is foreign, stale, replayed, mutated, or debt-bearing"
            )
        return projection

    def consume_owned_checkpoint_live_mutation_projection(
        self,
        projection: object,
    ) -> R06CheckpointLiveMutationProjection:
        """Consume exactly one owner-issued projection handle."""

        owned = self.require_owned_checkpoint_live_mutation_projection(projection)
        payload = _R06_CHECKPOINT_LIVE_MUTATION_REGISTRY.get(owned)
        if payload is None:
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live projection was already consumed"
            )
        global_receipt = payload.source_global_receipt_ref()
        if global_receipt is None:
            raise LandingOutcomeDeviceError(
                "R06 checkpoint live projection lost its exact global ACK"
            )
        del _R06_CHECKPOINT_LIVE_MUTATION_REGISTRY[owned]
        self._latest_checkpoint_live_mutation_projection = None
        self._r06_checkpoint_consumed_global_receipt = global_receipt
        return owned

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Freeze the complete R06 portable row for the sole global D2H."""

        # A dedicated Reward protocol poison must reach the sole global
        # device row so the same failing batch is observable.  It still cannot
        # ACK because the row's fault_count is nonzero.
        self._require_operable(
            allow_full_mdp_reward_cycle=self._full_mdp_reward_poisoned
        )
        if type(update_index) is not int or update_index < 0:
            raise LandingOutcomeDeviceError(
                "global R06 update_index must be a non-negative exact int"
            )
        if (
            type(completed_environment_steps) is not int
            or completed_environment_steps < 0
        ):
            raise LandingOutcomeDeviceError(
                "completed_environment_steps must be a non-negative exact int"
            )
        if update_index <= self._last_drained_update_index:
            raise LandingOutcomeDeviceError(
                "global R06 update_index must strictly advance"
            )
        if self._drain_sequence != 0 and not self._r06_global_drain_adopted:
            raise LandingOutcomeDeviceError(
                "a legacy-drained R06 owner cannot enter the global drain"
            )
        if (
            getattr(authority, "owner_kind", None)
            != _R06_GLOBAL_DRAIN_OWNER_KIND
            or tuple(getattr(authority, "field_names", ()))
            != R06_GLOBAL_DRAIN_FIELD_NAMES
            or getattr(authority, "expected_width", None)
            != len(R06_GLOBAL_DRAIN_FIELD_NAMES)
        ):
            raise LandingOutcomeDeviceError(
                "global drain R06 authority schema differs"
            )
        mint = getattr(authority, "mint_device_pack", None)
        require_owned_ack = getattr(authority, "require_owned_ack", None)
        if __package__:
            from . import action_ball_full_mdp_ppo_drain as drain
        else:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_ppo_drain as drain,
            )
        if (
            type(authority) is not drain.LeafDevicePackAuthority
            or not callable(mint)
            or getattr(mint, "__self__", None) is not authority
            or getattr(mint, "__func__", None)
            is not drain.LeafDevicePackAuthority.mint_device_pack
            or not callable(require_owned_ack)
            or getattr(require_owned_ack, "__self__", None) is not authority
            or getattr(require_owned_ack, "__func__", None)
            is not drain.LeafDevicePackAuthority.require_owned_ack
        ):
            raise LandingOutcomeDeviceError(
                "global drain R06 exact authority API differs"
            )
        values = self._r06_global_drain_values()
        pack = mint(leaf=self, values=values)
        # Adoption is monotonic even if a clean pre-transfer boundary aborts.
        # One leaf must never alternate independent and global D2H protocols.
        self._r06_global_drain_adopted = True
        self._active_r06_global_drain = _PreparedR06GlobalDrain(
            pack=pack,
            authority=authority,
            update_index=update_index,
            completed_environment_steps=completed_environment_steps,
        )
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        pack: object,
    ) -> None:
        """Release an exact pre-transfer pack without changing R06 facts."""

        active = self._active_r06_global_drain
        if active is None or pack is not active.pack:
            raise LandingOutcomeDeviceError(
                "global drain R06 abort pack is stale or foreign"
            )
        self._active_r06_global_drain = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Materialize the legacy R06 receipt from the one global D2H row."""

        active = self._active_r06_global_drain
        try:
            if self._r06_global_drain_poisoned or self._poisoned:
                raise LandingOutcomeDeviceError(
                    "global drain R06 acknowledgement owner is poisoned"
                )
            if active is None or pack is not active.pack:
                raise LandingOutcomeDeviceError(
                    "global drain R06 ACK pack is stale or foreign"
                )
            active.authority.require_owned_ack(
                leaf=self,
                pack=pack,
                receipt=receipt,
                owner_row=owner_row,
            )
            row = self._r06_global_owner_row_values(owner_row)
            owner_rows = getattr(receipt, "owner_rows", None)
            if (
                not isinstance(owner_rows, tuple)
                or sum(value is owner_row for value in owner_rows) != 1
                or getattr(receipt, "num_envs", None) != self.num_envs
                or getattr(receipt, "acknowledged", None) is not False
                or getattr(receipt, "update_index", None)
                != active.update_index
                or getattr(receipt, "completed_environment_steps", None)
                != active.completed_environment_steps
                or getattr(receipt, "drain_sequence", None)
                != self._drain_sequence + 1
                or getattr(receipt, "device_to_host_transfers", None) != 1
            ):
                raise LandingOutcomeDeviceError(
                    "global drain R06 acknowledgement boundary differs"
                )
            fault_counts = tuple(
                row[name] for name in R06_GLOBAL_DRAIN_FAULT_FIELDS
            )
            invariant_counts = tuple(
                row[name] for name in R06_GLOBAL_DRAIN_INVARIANT_FIELDS
            )
            if row["fault_count"] != sum(fault_counts):
                raise LandingOutcomeDeviceError(
                    "global R06 aggregate fault count differs"
                )
            if row["invariant_count"] != sum(invariant_counts):
                raise LandingOutcomeDeviceError(
                    "global R06 aggregate invariant count differs"
                )
            if row["fault_count"] != 0 or row["invariant_count"] != 0:
                raise LandingOutcomeDeviceError(
                    "faulted R06 row cannot acknowledge an optimizer update"
                )
            portable = LandingOutcomeBoundaryReceipt(
                schema_version=SCHEMA_VERSION,
                update_index=active.update_index,
                drain_sequence=getattr(receipt, "drain_sequence"),
                mutation_version=row["mutation_version"],
                fault_counts=fault_counts,
                flight_state_counts=tuple(
                    row[name]
                    for name in R06_GLOBAL_DRAIN_FLIGHT_STATE_FIELDS
                ),
                mailbox_state_counts=tuple(
                    row[name]
                    for name in R06_GLOBAL_DRAIN_MAILBOX_STATE_FIELDS
                ),
                invariant_counts=invariant_counts,
                installed_total=row["installed_total"],
                settled_total=row["settled_total"],
                retired_total=row["retired_total"],
                common_payment_total=row["common_payment_total"],
                placement_payment_total=row["placement_payment_total"],
                closed_total=row["closed_total"],
                checkpoint_safe=True,
                device_to_host_transfers=1,
                runtime_integrated=RUNTIME_INTEGRATED,
                cuda_profiled=CUDA_PROFILED,
                formal_exact_resume_integrated=(
                    FORMAL_EXACT_RESUME_INTEGRATED
                ),
                launch_authorized=LAUNCH_AUTHORIZED,
            )
        except BaseException as exc:
            self.poison_pre_optimizer_ppo_boundary(
                reason=(
                    "R06 global drain acknowledgement failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            raise
        self._drain_sequence = portable.drain_sequence
        self._last_drained_update_index = portable.update_index
        self._latest_receipt = portable
        self._latest_global_drain_receipt = receipt
        self._latest_receipt_consumed = False
        self._active_r06_global_drain = None
        try:
            # Seal the leaf's live epoch at its exact causal ACK.  The global
            # receipt becomes usable only after the construction-bound drain
            # owner has ACKed every leaf; project_checkpoint_live_mutation()
            # checks that later transition without rereading the owner.
            self._mint_r06_checkpoint_live_mutation_projection(
                global_receipt=receipt,
                portable=portable,
            )
        except BaseException as exc:
            self.poison_pre_optimizer_ppo_boundary(
                reason=(
                    "R06 live checkpoint epoch seal failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            raise

    def poison_pre_optimizer_ppo_boundary(self, *, reason: object) -> None:
        """Sticky fail-stop after global transfer/optimizer/partial ACK."""

        if self._r06_global_drain_poison_reason is None:
            self._r06_global_drain_poison_reason = (
                reason
                if type(reason) is str and bool(reason) and reason.isascii()
                else "unspecified R06 global PPO drain failure"
            )
        self._r06_global_drain_poisoned = True
        self._poisoned = True

    @property
    def latest_pre_optimizer_ppo_boundary_receipt(
        self,
    ) -> LandingOutcomeBoundaryReceipt:
        if (
            not self._r06_global_drain_adopted
            or self._latest_receipt is None
        ):
            raise LandingOutcomeDeviceError(
                "no acknowledged global R06 boundary receipt is available"
            )
        return self._latest_receipt

    def require_owned_pre_optimizer_ppo_boundary_receipt(
        self,
        global_receipt: object,
    ) -> LandingOutcomeBoundaryReceipt:
        """Project this R06 audit only from its exact acknowledged global ACK."""

        self._require_operable()
        portable = self._latest_receipt
        if (
            not self._r06_global_drain_adopted
            or portable is None
            or global_receipt is not self._latest_global_drain_receipt
            or getattr(global_receipt, "acknowledged", None) is not True
            or getattr(global_receipt, "update_index", None)
            != portable.update_index
            or getattr(global_receipt, "drain_sequence", None)
            != portable.drain_sequence
        ):
            raise LandingOutcomeDeviceError(
                "global receipt does not own the latest portable R06 audit"
            )
        return portable

    def drain_ppo_boundary(self, *, update_index: int) -> LandingOutcomeBoundaryReceipt:
        """Produce the sole packed D2H boundary receipt for this mutation state."""

        self._require_operable()
        if self._r06_global_drain_adopted:
            raise LandingOutcomeDeviceError(
                "legacy R06 drain is disabled after global PPO drain adoption"
            )
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise LandingOutcomeDeviceError("update_index must be an integer")
        if update_index <= self._last_drained_update_index:
            raise LandingOutcomeDeviceError("update_index must be strictly newer")
        fault_counts = self._combined_fault_counts()
        flight_state_counts = [
            (self._flight_state == state).to(torch.int64).sum()
            for state in (
                FLIGHT_EMPTY,
                FLIGHT_INBOUND,
                FLIGHT_OPEN,
                FLIGHT_SETTLED_RETAINED,
            )
        ]
        mailbox_state_counts = [
            (self._mailbox_state == state).to(torch.int64).sum()
            for state in (
                MAILBOX_EMPTY,
                MAILBOX_SETTLED_UNPAID,
                MAILBOX_PARTIALLY_PAID,
                MAILBOX_PAID,
            )
        ]
        invariant_counts = self._invariant_counts()
        packed = torch.stack(
            fault_counts
            + flight_state_counts
            + mailbox_state_counts
            + invariant_counts
            + [
                self._mutation_version,
                self._installed_total,
                self._settled_total,
                self._retired_total,
                self._payment_totals[0],
                self._payment_totals[1],
                self._closed_total,
            ]
        )
        host_values = packed.to(device="cpu").tolist()
        cursor = 0
        host_faults = tuple(host_values[cursor : cursor + len(FAULTS)])
        cursor += len(FAULTS)
        host_flight_states = tuple(host_values[cursor : cursor + 4])
        cursor += 4
        host_mailbox_states = tuple(host_values[cursor : cursor + 4])
        cursor += 4
        host_invariants = tuple(host_values[cursor : cursor + len(INVARIANT_NAMES)])
        cursor += len(INVARIANT_NAMES)
        host_mutation_version = host_values[cursor]
        cursor += 1
        totals = host_values[cursor : cursor + 6]
        self._drain_sequence += 1
        self._last_drained_update_index = update_index
        receipt = LandingOutcomeBoundaryReceipt(
            schema_version=SCHEMA_VERSION,
            update_index=update_index,
            drain_sequence=self._drain_sequence,
            mutation_version=host_mutation_version,
            fault_counts=host_faults,
            flight_state_counts=host_flight_states,
            mailbox_state_counts=host_mailbox_states,
            invariant_counts=host_invariants,
            installed_total=totals[0],
            settled_total=totals[1],
            retired_total=totals[2],
            common_payment_total=totals[3],
            placement_payment_total=totals[4],
            closed_total=totals[5],
            checkpoint_safe=(not any(host_invariants)) and (not any(host_faults)),
            device_to_host_transfers=1,
            runtime_integrated=RUNTIME_INTEGRATED,
            cuda_profiled=CUDA_PROFILED,
            formal_exact_resume_integrated=FORMAL_EXACT_RESUME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )
        self._latest_receipt = receipt
        self._latest_receipt_consumed = False
        return receipt

    def _checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                if name in (
                    "_env_ids",
                    "_flight_slot_ids",
                    "_mailbox_slot_ids",
                    "_registry_root_token",
                    "_c10_projection_token",
                ):
                    continue
                # Python 3.8 exact-Pod compatibility; all owned tensor names
                # are private by construction.
                result[name[1:] if name.startswith("_") else name] = value
            elif name in (
                "_flight_key_ints",
                "_flight_key_digests",
                "_mailbox_key_ints",
                "_mailbox_key_digests",
                "_previous_paid_key_ints",
                "_previous_paid_key_digests",
            ):
                prefix = name[1:] if name.startswith("_") else name
                for field_name, tensor in value.items():
                    result[f"{prefix}.{field_name}"] = tensor
        return result

    def state_dict(
        self, receipt: LandingOutcomeBoundaryReceipt
    ) -> dict[str, object]:
        """Create one checkpoint from the exact latest safe boundary receipt."""

        self._require_operable()
        self._require_formal_only("export a checkpoint")
        if receipt is not self._latest_receipt:
            raise LandingOutcomeDeviceError("checkpoint receipt is not the latest identity")
        if self._latest_receipt_consumed:
            raise LandingOutcomeDeviceError("checkpoint receipt was already consumed")
        if not receipt.checkpoint_safe:
            raise LandingOutcomeDeviceError("checkpoint receipt reports invariant failures")
        tensors = {
            name: tensor.detach().clone()
            for name, tensor in self._checkpoint_tensors().items()
        }
        tensor_bytes_sha256 = _tensor_bytes_sha256(
            tensors,
            expected_mutation_version=receipt.mutation_version,
        )
        checkpoint: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "num_envs": self.num_envs,
            "flight_slot_capacity": self.flight_slot_capacity,
            "mailbox_capacity": self.mailbox_capacity,
            "dtype": str(self.dtype),
            "profile_payload": self.profile.payload(),
            "runtime_binding": {
                field.name: getattr(self.runtime_binding, field.name)
                for field in fields(self.runtime_binding)
            },
            "payment_authority": self.payment_authority.to_mapping(),
            "receipt": receipt,
            "mutation_version": receipt.mutation_version,
            "drain_sequence": self._drain_sequence,
            "last_drained_update_index": self._last_drained_update_index,
            "drain_protocol": (
                _R06_GLOBAL_DRAIN_PROTOCOL
                if self._r06_global_drain_adopted
                else _R06_LEGACY_DRAIN_PROTOCOL
            ),
            "global_drain_adopted": self._r06_global_drain_adopted,
            "tensor_manifest": _tensor_manifest(tensors),
            "tensor_bytes_sha256": tensor_bytes_sha256,
            "tensors": tensors,
        }
        checkpoint["checkpoint_content_sha256"] = _checkpoint_content_sha256(
            checkpoint
        )
        self._latest_receipt_consumed = True
        return checkpoint

    def load_state_dict(
        self,
        checkpoint: Mapping[str, object],
        *,
        expected_checkpoint_content_sha256: str,
    ) -> None:
        """Validate fully in a fresh shadow owner before committing restore."""

        self._require_operable()
        self._require_formal_only("restore a checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise LandingOutcomeDeviceError("checkpoint must be a mapping")
        if not self._load_fresh:
            raise LandingOutcomeDeviceError("load_state_dict requires a fresh owner")
        expected_content_root = _sha256_hex(
            expected_checkpoint_content_sha256,
            label="expected_checkpoint_content_sha256",
        )
        expected_metadata = {
            "schema_version": SCHEMA_VERSION,
            "num_envs": self.num_envs,
            "flight_slot_capacity": self.flight_slot_capacity,
            "mailbox_capacity": self.mailbox_capacity,
            "dtype": str(self.dtype),
            "profile_payload": self.profile.payload(),
            "runtime_binding": {
                field.name: getattr(self.runtime_binding, field.name)
                for field in fields(self.runtime_binding)
            },
            "payment_authority": self.payment_authority.to_mapping(),
        }
        expected_top_level = set(expected_metadata) | {
            "receipt",
            "mutation_version",
            "drain_sequence",
            "last_drained_update_index",
            "drain_protocol",
            "global_drain_adopted",
            "tensor_manifest",
            "tensor_bytes_sha256",
            "checkpoint_content_sha256",
            "tensors",
        }
        if set(checkpoint) != expected_top_level:
            raise LandingOutcomeDeviceError("checkpoint top-level names differ")
        for name, expected in expected_metadata.items():
            if checkpoint.get(name) != expected:
                raise LandingOutcomeDeviceError(f"checkpoint {name} differs")
        receipt = checkpoint.get("receipt")
        if not isinstance(receipt, LandingOutcomeBoundaryReceipt):
            raise LandingOutcomeDeviceError("checkpoint receipt type differs")
        if (
            receipt.schema_version != SCHEMA_VERSION
            or not receipt.checkpoint_safe
            or receipt.device_to_host_transfers != 1
            or any(receipt.fault_counts)
            or any(receipt.invariant_counts)
            or receipt.runtime_integrated is not RUNTIME_INTEGRATED
            or receipt.cuda_profiled is not CUDA_PROFILED
            or receipt.formal_exact_resume_integrated
            is not FORMAL_EXACT_RESUME_INTEGRATED
            or receipt.launch_authorized is not LAUNCH_AUTHORIZED
        ):
            raise LandingOutcomeDeviceError("checkpoint receipt is not safe")
        declared_content_root = _sha256_hex(
            checkpoint.get("checkpoint_content_sha256"),
            label="checkpoint_content_sha256",
        )
        if declared_content_root != expected_content_root:
            raise LandingOutcomeDeviceError("checkpoint external content root differs")
        tensors = checkpoint.get("tensors")
        if not isinstance(tensors, Mapping):
            raise LandingOutcomeDeviceError("checkpoint tensors must be a mapping")
        expected_tensors = self._checkpoint_tensors()
        if set(tensors) != set(expected_tensors):
            raise LandingOutcomeDeviceError("checkpoint tensor names differ")
        manifest = _tensor_manifest(tensors)
        if checkpoint.get("tensor_manifest") != manifest:
            raise LandingOutcomeDeviceError("checkpoint tensor manifest differs")
        declared_tensor_root = _sha256_hex(
            checkpoint.get("tensor_bytes_sha256"),
            label="tensor_bytes_sha256",
        )
        if _tensor_bytes_sha256(tensors) != declared_tensor_root:
            raise LandingOutcomeDeviceError("checkpoint tensor bytes root differs")
        if _checkpoint_content_sha256(checkpoint) != declared_content_root:
            raise LandingOutcomeDeviceError("checkpoint content root differs")

        shadow = ActionBallLandingOutcomeDeviceCoordinator(
            num_envs=self.num_envs,
            flight_slot_capacity=self.flight_slot_capacity,
            mailbox_capacity=self.mailbox_capacity,
            device=self.device,
            dtype=self.dtype,
            profile=self.profile,
            runtime_binding=self.runtime_binding,
            payment_authority=self.payment_authority,
            capacity_authority=self.capacity_authority,
            text_registry=self.text_registry,
        )
        shadow_tensors = shadow._checkpoint_tensors()
        for name, destination in shadow_tensors.items():
            source = tensors[name]
            if (
                not isinstance(source, torch.Tensor)
                or tuple(source.shape) != tuple(destination.shape)
                or source.dtype != destination.dtype
            ):
                raise LandingOutcomeDeviceError(f"checkpoint tensor {name} differs")
            destination.copy_(source.to(device=self.device))
        mutation_version = checkpoint.get("mutation_version")
        drain_sequence = checkpoint.get("drain_sequence")
        last_update = checkpoint.get("last_drained_update_index")
        drain_protocol = checkpoint.get("drain_protocol")
        global_drain_adopted = checkpoint.get("global_drain_adopted")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (mutation_version, drain_sequence, last_update)
        ):
            raise LandingOutcomeDeviceError("checkpoint host counters differ")
        if mutation_version < 0 or drain_sequence < 1 or last_update < 0:
            raise LandingOutcomeDeviceError("checkpoint host counters are out of range")
        if (
            type(global_drain_adopted) is not bool
            or drain_protocol
            != (
                _R06_GLOBAL_DRAIN_PROTOCOL
                if global_drain_adopted
                else _R06_LEGACY_DRAIN_PROTOCOL
            )
        ):
            raise LandingOutcomeDeviceError(
                "checkpoint drain protocol/adoption binding differs"
            )
        if (
            receipt.mutation_version != mutation_version
            or receipt.drain_sequence != drain_sequence
            or receipt.update_index != last_update
        ):
            raise LandingOutcomeDeviceError("checkpoint receipt/counter binding differs")
        shadow._drain_sequence = drain_sequence
        shadow._last_drained_update_index = last_update
        validation = shadow.drain_ppo_boundary(update_index=last_update + 1)
        if not validation.checkpoint_safe:
            raise LandingOutcomeDeviceError("checkpoint tensor invariants failed")
        if (
            validation.mutation_version != mutation_version
            or validation.fault_counts != receipt.fault_counts
            or validation.flight_state_counts != receipt.flight_state_counts
            or validation.mailbox_state_counts != receipt.mailbox_state_counts
            or validation.invariant_counts != receipt.invariant_counts
            or validation.installed_total != receipt.installed_total
            or validation.settled_total != receipt.settled_total
            or validation.retired_total != receipt.retired_total
            or validation.common_payment_total != receipt.common_payment_total
            or validation.placement_payment_total != receipt.placement_payment_total
            or validation.closed_total != receipt.closed_total
        ):
            raise LandingOutcomeDeviceError("checkpoint receipt does not match tensors")
        for name, destination in expected_tensors.items():
            destination.copy_(shadow_tensors[name])
        self._drain_sequence = drain_sequence
        self._last_drained_update_index = last_update
        self._latest_receipt = None
        self._latest_global_drain_receipt = None
        self._latest_receipt_consumed = False
        self._r06_global_drain_adopted = global_drain_adopted
        self._load_fresh = False


__all__ = [
    "ActionBallFullMdpObservationProjection",
    "ActionEpochR06CurrentFlightObservationView",
    "ActionEpochR06OutcomeRows",
    "ActionEpochR06PostPhysicsResult",
    "ActionEpochR06PostPhysicsSample",
    "ActionEpochR06RetireResult",
    "ActionBallLandingOutcomeDeviceCoordinator",
    "ArmedPhysicalRetire",
    "CANONICAL_REASON_NOT_SCORED",
    "C05_LANDING_OUTCOME_SOURCE_SHA256",
    "C05_CROSSING_ABS_TOL_M",
    "C10_FAMILY_A",
    "C10_FAMILY_C",
    "COMMON_ON_TABLE_CONSUMER",
    "CONSUMERS",
    "CUDA_PROFILED",
    "DEVICE_R05_HOT_REVEAL_OWNER_KIND",
    "DEVICE_R05_HOT_REVEAL_REQUIRED_PROJECTION_FIELDS",
    "DeviceLandingOutcomeKey",
    "DeviceMutationResult",
    "EMPTY",
    "FAULTS",
    "FAULT_BATCH_ABORT",
    "FAULT_CONTACT_ORDER",
    "FAULT_CROSSING_BEFORE_CONTACT",
    "FAULT_CROSSING_REPORT",
    "FAULT_DUPLICATE_PAYMENT",
    "FAULT_DUPLICATE_VIEW",
    "FAULT_ENGINE_OVERFLOW",
    "FAULT_FLIGHT_COLLISION",
    "FAULT_FLIGHT_CONTINUITY",
    "FAULT_GENERATION_BINDING",
    "FAULT_INVALID_CLOSE",
    "FAULT_INVALID_INSTALL",
    "FAULT_INVALID_OBSERVATION",
    "FAULT_INVALID_PAYMENT",
    "FAULT_INVALID_RETIRE",
    "FAULT_INVALID_STAMP",
    "FAULT_KEY_BINDING",
    "FAULT_MAILBOX_COLLISION",
    "FAULT_MAILBOX_COPY_COLLISION",
    "FAULT_MISSED_PAYMENT",
    "FAULT_NET_CONTRACT",
    "FAULT_NONFINITE",
    "FAULT_OBSERVATION_ORDINAL",
    "FAULT_PAYMENT_BEFORE_VIEW",
    "FAULT_PAYMENT_EPOCH",
    "FAULT_PRODUCER_CONTRACT",
    "FAULT_REPLAY",
    "FAULT_SAFETY_CLEANUP",
    "FAULT_STAMP_REGRESSION",
    "FAULT_TASK_DRAIN",
    "FAULT_UNOBSERVED_LIVE_SLOT",
    "FLIGHT_EMPTY",
    "FLIGHT_INBOUND",
    "FlightLifecycleSnapshotBatch",
    "FLIGHT_OPEN",
    "FLIGHT_SETTLED_RETAINED",
    "FORMAL_EXACT_RESUME_INTEGRATED",
    "FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN",
    "FULL_MDP_REVEAL_BOUNDARY_EFFECTIVE_SOURCE_SHA256",
    "FULL_MDP_REVEAL_BOUNDARY_FINAL_SOURCE_PIN_PENDING",
    "FULL_MDP_REVEAL_BOUNDARY_OBSERVED_SOURCE_SHA256",
    "FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256",
    "FULL_MDP_REVEAL_BOUNDARY_ROW_INTEGRITY_SHA256",
    "FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256",
    "HOLD_REASONS",
    "INBOUND",
    "LAUNCH_AUTHORIZED",
    "ArmedLandingOutcomeSelectedReset",
    "LandingOutcomeBoundaryReceipt",
    "LandingOutcomeCapacityAuthority",
    "LandingOutcomeC10FamilyPaymentAuthority",
    "LandingOutcomeDeviceError",
    "LandingOutcomeDeviceR05HotRevealProductionHold",
    "DiagnosticN2NoSaveConstructionBinding",
    "DiagnosticN2NoSaveNamespaceProjection",
    "DiagnosticN2NoSaveNamespaceView",
    "LandingOutcomeFullMdpRewardCloseReceipt",
    "LandingOutcomeFullMdpRewardCycleToken",
    "LandingOutcomeFullMdpRewardPaymentVerdict",
    "LandingOutcomePhysicalParkPreparedTokenClaim",
    "LandingOutcomePhysicalParkTokenAuthority",
    "LandingOutcomePostPhysicsContactAuthority",
    "LandingOutcomePostPhysicsContactAuthorityView",
    "LandingOutcomeSelectedResetCommitToken",
    "LandingOutcomeSelectedResetCompletionToken",
    "LandingOutcomeSelectedResetMaskCapability",
    "LandingOutcomeSelectedResetMaskView",
    "LandingOutcomeSelectedResetPhysicalParkCommitTokenClaim",
    "LandingOutcomeSelectedResetPhysicalParkPreparedTokenClaim",
    "LandingOutcomeSelectedResetPhysicalParkTokenAuthority",
    "LandingOutcomeRuntimeBinding",
    "LandingOutcomeTextRegistry",
    "LandingRevealInstall",
    "LandingRevealInstallDevicePack",
    "LandingRevealInstallReceipt",
    "LandingRevealArmedInstall",
    "LandingRevealCensoredInstall",
    "LandingRevealCensorReceipt",
    "LandingRevealCommitReceipt",
    "LandingRevealPrepareAttempt",
    "LandingRevealPreparedInstallReceipt",
    "MAILBOX_ALLOCATION_POLICY",
    "MAILBOX_EMPTY",
    "MAILBOX_PAID",
    "MAILBOX_PARTIALLY_PAID",
    "MAILBOX_SETTLED_UNPAID",
    "OPEN",
    "PAID",
    "PARTIALLY_PAID",
    "PhysicalRetireCleanupMaskCapability",
    "PhysicsStampBatch",
    "PHASE_CONTACT",
    "PHASE_LANDING",
    "PHASE_NET",
    "PLACEMENT_GUIDANCE_CONSUMER",
    "PostPhysicsFlightBatch",
    "PostPhysicsMutationResult",
    "PreviousPaidActionEpochRows",
    "PreparedPhysicalRetire",
    "PreparedLandingOutcomeSelectedReset",
    "PhysicalRetireMutationResult",
    "RUNTIME_INTEGRATED",
    "R06ChildTerminalToken",
    "R06_ACTION_EPOCH_CANONICAL_TOTAL_F32",
    "R06_ACTION_EPOCH_COMMON_ON_TABLE_F32",
    "R06_ACTION_EPOCH_CONTACT_VALID_F32",
    "R06_ACTION_EPOCH_CROSSING_VALID_F32",
    "R06_ACTION_EPOCH_CROSSING_XY_F32",
    "R06_ACTION_EPOCH_FACT_F32_USED",
    "R06_ACTION_EPOCH_FACT_F32_WIDTH",
    "R06_ACTION_EPOCH_NET_CLEAR_F32",
    "R06_ACTION_EPOCH_NET_CROSSED_F32",
    "R06_ACTION_EPOCH_ON_TABLE_F32",
    "R06_ACTION_EPOCH_PLACEMENT_ERROR_F32",
    "R06_ACTION_EPOCH_PLACEMENT_GAIN_F32",
    "R06_ACTION_EPOCH_POLICY_ELIGIBLE",
    "R06_ACTION_EPOCH_PRESENT",
    "R06_ACTION_EPOCH_SOURCE_VALID",
    "R06_REVEAL_BOUNDARY_FAULT_SCHEMA",
    "R06_REVEAL_BOUNDARY_FAULT_SCHEMA_SHA256",
    "R06_GLOBAL_DRAIN_FIELD_NAMES",
    "R06_GLOBAL_DRAIN_REQUIRED_FIELDS",
    "R06_PPO_DRAIN_LEAF_SCHEMA",
    "R06_FULL_MDP_REWARD_RUNTIME_CONNECTED",
    "R06_FULL_MDP_REWARD_TOP_BIND_API_FROZEN",
    "R06CheckpointLiveMutationProjection",
    "R05_RUNTIME_TRANSACTION_CONTRACT_MAPPING",
    "R05_RUNTIME_TRANSACTION_CONTRACT_SHA256",
    "R05_FINAL_SOURCE_PIN_PENDING",
    "R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256",
    "R05_RUNTIME_TRANSACTION_SOURCE_SHA256",
    "SCHEMA_VERSION",
    "SETTLED_RETAINED",
    "SETTLED_UNPAID",
    "SETTLEMENT_CAUSE_CONTACT_DEADLINE",
    "SETTLEMENT_CAUSE_CROSSING_HORIZON",
    "SETTLEMENT_CAUSE_ENGINE_OVERFLOW",
    "SETTLEMENT_CAUSE_FIRST_CROSSING",
    "SETTLEMENT_CAUSE_NONE",
    "SETTLEMENT_CAUSE_NONFINITE",
    "SETTLEMENT_CAUSE_PRODUCER_CONTRACT_FAULT",
    "SETTLEMENT_CAUSE_PROTOCOL_FAULT",
    "SharedLandingOutcomeDeviceView",
    "build_c10_family_payment_authority",
    "build_landing_outcome_capacity_authority",
    "build_landing_reveal_install",
    "build_landing_reveal_install_batch",
    "construct_diagnostic_n2_no_save_r06",
    "mint_landing_outcome_physical_park_token_authority",
    "mint_landing_outcome_selected_reset_physical_park_token_authority",
    "materialize_r06_ppo_drain_leaf_schema",
]
