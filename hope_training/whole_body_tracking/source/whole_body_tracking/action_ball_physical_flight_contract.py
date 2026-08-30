"""Portable, fail-closed contract for the constructed physical-ball owner.

This module deliberately owns no Isaac object and imports no project module.  It
freezes the bytes exchanged at the R05/physical-owner boundary so a later scene
adapter cannot replace full ball state with a digest, silently choose one body,
or make a prepared reveal public before every owner has passed preflight.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields
import hashlib
import json
import math
from numbers import Real
import struct
from typing import ClassVar, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 2
INTEGRATION_STATUS = "PRE_INTEGRATION_HOLD"
RUNTIME_INTEGRATED = False
POD_VALIDATED = False
LAUNCH_AUTHORIZED = False
RUNTIME_DTYPE = "float32"
QUANTIZATION_CONTRACT = (
    "ieee754_binary32_round_ties_to_even_big_endian_bytes_v1"
)
ZERO_CANONICALIZATION = "all_binary32_zero_is_positive_zero_v1"
CAPACITY_FORMULA = (
    "max_flight_horizon_control_steps//cadence_control_steps+1"
)
INCLUSIVE_INTERVAL_SEMANTICS = (
    "reveal_through_flight_horizon_both_endpoints_inclusive_v1"
)
SAME_TICK_ORDERING = (
    "new_reveal_admission_precedes_equal_horizon_prior_flight_release_v1"
)

POSITION_FRAME = "environment_local_world_aligned_m_v1"
QUATERNION_ORDER = "wxyz"
LINEAR_VELOCITY_FRAME = "world_m_per_s_v1"
ANGULAR_VELOCITY_FRAME = "world_rad_per_s_v1"
INSTALL_STATE_EPOCH = "after_reveal_before_first_physics_substep_v1"

TASK_REF_KIND = "action_ball_continuous_task_receipt_ref_v2"
OUTCOME_KEY_KIND = "action_ball_landing_outcome_shot_key_v1"
REVEAL_FINAL_PREVIEW_KIND = (
    "action_ball_continuous_reveal_final_preview_batch_v2"
)
R05_REVEAL_FINAL_PREVIEW_SCHEMA_SHA256 = hashlib.sha256(
    b"action_ball_continuous_reveal_final_preview_batch_v2:"
    b"integration_status,phase,public_visible,policy_opportunity_created,"
    b"owner_checkpoint_before_sha256,prepared_batch,"
    b"sampler_checkpoint_before_commit_sha256,"
    b"sampler_checkpoint_after_commit_sha256,untouched_rows_before_sha256,"
    b"untouched_rows_after_sha256,sampler_checkpoint_before_commit,"
    b"sampler_checkpoint_after_commit,reveal_final_rows,"
    b"all_owner_install_root_sha256"
).hexdigest()
COMMITTED_REVEAL_BATCH_KIND = (
    "action_ball_continuous_committed_reveal_batch_v2"
)
CENSORED_REVEAL_BATCH_KIND = (
    "action_ball_continuous_censored_reveal_batch_v2"
)
PREPARED_REVEAL_TERMINAL_CLAIM_KIND = (
    "action_ball_continuous_prepared_reveal_terminal_claim_v1"
)
R05_TERMINAL_BOUNDARY_AUTHORITY_KIND = (
    "action_ball_continuous_terminal_boundary_authority_v1"
)
R05_TERMINAL_BOUNDARY_PROJECTION_KIND = (
    "action_ball_continuous_terminal_boundary_projection_v1"
)
R05_TERMINAL_BOUNDARY_PARTICIPANT_ROOT_KIND = (
    "action_ball_continuous_terminal_boundary_participant_root_v1"
)
R05_TERMINAL_BOUNDARY_CENSOR_EVIDENCE_KIND = (
    "action_ball_continuous_terminal_boundary_censor_evidence_v1"
)
R05_PREPARED_TERMINAL_CONTENT_PIN_KIND = (
    "action_ball_continuous_prepared_terminal_content_pin_v1"
)
R05_TERMINAL_BOUNDARY_PROJECTION_SCHEMA_SHA256 = hashlib.sha256(
    b"action_ball_continuous_terminal_boundary_projection_v1:"
    b"authority_domain,authority_schema_sha256,authority_source_sha256,"
    b"decision_mapping_schema_version,source_decision,decision,"
    b"reveal_final_preview_schema_version,reveal_final_preview_sha256,"
    b"selected_env_ids,boundary_receipt_kind,boundary_receipt_sha256,"
    b"boundary_packet_schema_version,boundary_packet_sha256,"
    b"ordered_participant_roots,ordered_censor_evidence"
).hexdigest()
R05_PREPARED_TERMINAL_CONTENT_PIN_SCHEMA_SHA256 = hashlib.sha256(
    b"action_ball_continuous_prepared_terminal_content_pin_v1:"
    b"terminal_schema_version,terminal_kind,terminal_canonical_sha256,"
    b"content_bytes_base64,content_byte_length,content_bytes_sha256"
).hexdigest()
REVEAL_PREPARE_BOUNDARY_MARKER_KIND = (
    "action_ball_continuous_reveal_prepare_boundary_marker_v1"
)
REVEAL_PREPARE_BOUNDARY_MARKER_SCHEMA_SHA256 = hashlib.sha256(
    b"action_ball_continuous_reveal_prepare_boundary_marker_v1:"
    b"selected_env_ids,reveal_final_preview_sha256,boundary_packet_version,"
    b"boundary_packet_root_sha256,boundary_transfer_count,selected_pass_count,"
    b"selected_fault_count,ordered_child_token_roots"
).hexdigest()
R05_PREARM_CHILD_OWNER_KINDS = (
    "motion",
    "racket",
    "physical_ball",
    "r06_flight",
)
FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND = (
    "action_ball_full_mdp_reveal_boundary_receipt_v1"
)
FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256 = (
    "ec9db7ca2475bc8d4de474aeca9ce425feaeaef617fea7371f23d0ee5f8e25ab"
)
FULL_MDP_REVEAL_BOUNDARY_PACKET_SCHEMA_VERSION = 4
FULL_MDP_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256 = (
    "cfc212a4ef2fd2078df99114c28f55df93b0605e0a126049b24b07fc636b16aa"
)
FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256 = (
    "4e715720b741991905d7c6cf8aa5ddf6c5a1e617773b6132aa33368468736cdd"
)
FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER = R05_PREARM_CHILD_OWNER_KINDS
FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN = (
    "action_ball_full_mdp_reveal_boundary"
)
FULL_MDP_REVEAL_DECISION_ACCEPT = "ACCEPT"
FULL_MDP_REVEAL_DECISION_CENSOR = "CENSOR"
LEGACY_DIGEST_ONLY_REVEAL_FINAL_PREVIEW_KIND = (
    "action_ball_continuous_reveal_final_preview_batch_v1"
)
PHYSICAL_SETTLEMENT_AUTHORITY_KIND = (
    "action_ball_physical_flight_settlement_authority_v2"
)
PHYSICAL_SETTLEMENT_AUTHORITY_SCHEMA_SHA256 = hashlib.sha256(
    b"action_ball_physical_flight_settlement_authority_v2:"
    b"mailbox_lifecycle,r06_owner_mutation_version,r06_after_root_sha256,"
    b"physical_retire_rows"
).hexdigest()
PHYSICAL_ZERO_OPEN_RESET_CLOSURE_KIND = (
    "action_ball_physical_zero_open_reset_closure_v2"
)
PHYSICAL_OWNER_STATE_KIND = "action_ball_physical_flight_owner_state_v4"
PHYSICAL_OWNER_STATE_SCHEMA = {
    "schema_version": 4,
    "kind": PHYSICAL_OWNER_STATE_KIND,
    "slot_mutation_semantics": "per_slot_monotone_v1",
    "owner_mutation_semantics": "owner_operation_highwater_v1",
    "scene_state_encoding": "contiguous_float32_little_endian_bytes_v1",
    "flight_lifecycle_encoding": "r06_empty_inbound_open_settled_i8_v1",
    "complete_slot_grid": True,
}
PHYSICAL_OWNER_STATE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        PHYSICAL_OWNER_STATE_SCHEMA,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
PHYSICAL_OWNER_STATE_REQUIRED_FIELDS = frozenset(
    PHYSICAL_OWNER_STATE_SCHEMA
) | frozenset(
    (
        "capacity_receipt_sha256",
        "num_envs",
        "flight_capacity",
        "owner_mutation_version",
        "next_prepare_nonce",
        "reset_generation",
        "slot_snapshots",
        "scene_state_shape",
        "scene_state_f32_base64",
        "scene_state_byte_length",
        "scene_state_bytes_sha256",
        "flight_lifecycle_code",
        "observation_ordinal",
        "previous_ball_center_m",
        "device_fault",
        "pending_r06_settlement_ack",
        "poisoned",
    )
)
R05_BALL_SLOT_SNAPSHOT_KIND = "action_ball_continuous_ball_slot_snapshot_v1"
R05_BALL_SLOT_PLAN_KIND = "action_ball_continuous_ball_slot_plan_v2"
R05_REVEAL_FINAL_INSTALL_ROW_KIND = (
    "action_ball_continuous_reveal_final_install_row_v2"
)
R05_COMMITTED_REVEAL_KIND = "action_ball_continuous_committed_reveal_v2"

STATE_COMPONENTS = (
    "position_env_x_m",
    "position_env_y_m",
    "position_env_z_m",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "linear_velocity_world_x_mps",
    "linear_velocity_world_y_mps",
    "linear_velocity_world_z_mps",
    "angular_velocity_world_x_radps",
    "angular_velocity_world_y_radps",
    "angular_velocity_world_z_radps",
)

SLOT_PARKED = "PARKED"
SLOT_IN_FLIGHT = "IN_FLIGHT"
SLOT_SETTLED_RETAINED = "SETTLED_RETAINED"
SLOT_RETIRED = "RETIRED"
SLOT_LIFECYCLES = frozenset(
    (SLOT_PARKED, SLOT_IN_FLIGHT, SLOT_SETTLED_RETAINED, SLOT_RETIRED)
)

MAX_ACTION_UID = (1 << 53) - 1
_HEX = frozenset("0123456789abcdef")
_LEGACY_DIGEST_ONLY_INSTALL_KINDS = frozenset(
    (
        "action_ball_physical_ball_install_payload_v0",
        "action_ball_physical_ball_install_digest_v1",
    )
)
_LEGACY_DIGEST_ONLY_CHECKPOINT_KINDS = frozenset(
    (
        "action_ball_physical_flight_checkpoint_digest_v1",
        "action_ball_physical_ball_checkpoint_v0",
    )
)


class PhysicalFlightContractError(ValueError):
    """The portable physical-flight contract was violated."""


class ExternalContentPinError(PhysicalFlightContractError):
    """Externally supplied bytes do not match the caller's content pin."""


class DigestOnlyPayloadTombstonedError(PhysicalFlightContractError):
    """A superseded digest-only physical payload was presented."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite canonical JSON as ASCII bytes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PhysicalFlightContractError(
            "value is not finite canonical JSON"
        ) from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_sha256(payload)
    return result


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise PhysicalFlightContractError(f"{label} must be an exact int")
    if value < minimum or (maximum is not None and value > maximum):
        raise PhysicalFlightContractError(f"{label} is outside its allowed range")
    return value


def _optional_plain_int(
    value: object, *, label: str, minimum: int = 0
) -> Optional[int]:
    if value is None:
        return None
    return _plain_int(value, label=label, minimum=minimum)


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise PhysicalFlightContractError(f"{label} must be an exact bool")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise PhysicalFlightContractError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PhysicalFlightContractError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _optional_sha256(value: object, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _sha256(value, label=label)


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PhysicalFlightContractError(f"{label} must be an ordered sequence")
    return tuple(value)


def _ordered_unique_env_ids(value: object, *, label: str) -> tuple[int, ...]:
    rows = tuple(
        _plain_int(item, label=f"{label}[]")
        for item in _sequence(value, label=label)
    )
    if not rows or rows != tuple(sorted(set(rows))):
        raise PhysicalFlightContractError(
            f"{label} must be non-empty, sorted, and unique"
        )
    return rows


def _encode(value: object) -> object:
    if isinstance(value, _SealedRecord):
        return value.to_mapping()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _verified_values(
    value: object,
    *,
    cls: type,
    kind: str,
    schema_version: int,
) -> dict[str, object]:
    label = cls.__name__
    if not isinstance(value, Mapping):
        raise PhysicalFlightContractError(f"{label} must be a mapping")
    names = tuple(field.name for field in fields(cls))
    expected_payload = frozenset(("schema_version", "kind", *names))
    expected = expected_payload | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise PhysicalFlightContractError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    payload = {key: value[key] for key in expected_payload}
    if type(payload["schema_version"]) is not int or (
        payload["schema_version"] != schema_version
    ):
        raise PhysicalFlightContractError(f"{label} schema_version differs")
    if payload["kind"] != kind:
        raise PhysicalFlightContractError(f"{label} kind differs")
    declared = _sha256(
        value["canonical_sha256"], label=f"{label}.canonical_sha256"
    )
    if canonical_sha256(payload) != declared:
        raise PhysicalFlightContractError(f"{label} canonical SHA differs")
    return {name: payload[name] for name in names}


def _verified_external_sealed_mapping(
    value: object,
    *,
    label: str,
    expected_kind: str,
    expected_schema_version: int,
    expected_payload_fields: frozenset[str],
) -> Mapping[str, object]:
    """Verify one nested external record instead of trusting the outer seal."""

    if not isinstance(value, Mapping):
        raise PhysicalFlightContractError(f"{label} must be a sealed mapping")
    expected = expected_payload_fields | {
        "schema_version",
        "kind",
        "canonical_sha256",
    }
    if frozenset(value) != expected:
        raise PhysicalFlightContractError(f"{label} keys differ")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != expected_schema_version
        or value["kind"] != expected_kind
    ):
        raise PhysicalFlightContractError(f"{label} schema/kind differs")
    declared = _sha256(
        value["canonical_sha256"], label=f"{label}.canonical_sha256"
    )
    payload = {
        key: value[key]
        for key in value
        if key != "canonical_sha256"
    }
    if canonical_sha256(payload) != declared:
        raise PhysicalFlightContractError(f"{label} canonical SHA differs")
    return value


_R05_SLOT_FIELDS = frozenset(
    (
        "slot_index",
        "lifecycle_state",
        "physical_retired",
        "owner_key_sha256",
        "ball_generation",
        "inbound_ball_sha256",
        "dynamic_state_sha256",
    )
)


def _verified_r05_ball_slots(
    value: object, *, label: str, capacity: int
) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(value, label=label)
    if len(rows) != capacity:
        raise PhysicalFlightContractError(f"{label} width differs from capacity")
    result = tuple(
        _verified_external_sealed_mapping(
            row,
            label=f"{label}[{index}]",
            expected_kind=R05_BALL_SLOT_SNAPSHOT_KIND,
            expected_schema_version=1,
            expected_payload_fields=_R05_SLOT_FIELDS,
        )
        for index, row in enumerate(rows)
    )
    if tuple(row["slot_index"] for row in result) != tuple(range(capacity)):
        raise PhysicalFlightContractError(f"{label} slot order differs")
    return result


class _SealedRecord:
    KIND: ClassVar[str]
    RECORD_SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.RECORD_SCHEMA_VERSION,
            "kind": self.KIND,
            **{
                field.name: _encode(getattr(self, field.name))
                for field in fields(self)
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def _mapping_values(cls, value: object) -> dict[str, object]:
        return _verified_values(
            value,
            cls=cls,
            kind=cls.KIND,
            schema_version=cls.RECORD_SCHEMA_VERSION,
        )

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        return values

    @classmethod
    def _from_mapping_unpinned(cls, value: object):
        return cls(**cls._decode_values(cls._mapping_values(value)))

    @classmethod
    def _reject_legacy(cls, value: object) -> None:
        del value

    @classmethod
    def from_mapping(
        cls, value: object, *, expected_canonical_sha256: str
    ):
        cls._reject_legacy(value)
        expected = _sha256(
            expected_canonical_sha256,
            label=f"{cls.__name__}.expected_canonical_sha256",
        )
        if not isinstance(value, Mapping):
            raise PhysicalFlightContractError(
                f"{cls.__name__} must be a mapping"
            )
        declared = value.get("canonical_sha256")
        if declared != expected:
            raise ExternalContentPinError(
                f"{cls.__name__} differs from its external content pin"
            )
        result = cls._from_mapping_unpinned(value)
        if result.canonical_sha256 != expected:
            raise ExternalContentPinError(
                f"{cls.__name__} reconstructed content differs from its pin"
            )
        return result


def _canonical_mapping_from_base64(
    value: object,
    *,
    label: str,
    byte_length: object,
    bytes_sha256: object,
) -> tuple[bytes, Mapping[str, object]]:
    encoded = _text(value, label=label)
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise PhysicalFlightContractError(f"{label} is not strict base64") from exc
    expected_length = _plain_int(byte_length, label=f"{label}.byte_length")
    expected_sha = _sha256(bytes_sha256, label=f"{label}.bytes_sha256")
    if len(raw) != expected_length or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise PhysicalFlightContractError(f"{label} byte pin differs")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalFlightContractError(
            f"{label} is not canonical JSON ASCII"
        ) from exc
    if not isinstance(decoded, Mapping) or canonical_json_bytes(decoded) != raw:
        raise PhysicalFlightContractError(
            f"{label} bytes are not one canonical JSON mapping"
        )
    return raw, decoded


@dataclass(frozen=True)
class CanonicalJsonContentPin(_SealedRecord):
    """Full canonical bytes plus both source and byte-level digests."""

    KIND: ClassVar[str] = "action_ball_external_canonical_json_content_pin_v2"

    source_schema_version: int
    source_kind: str
    source_schema_sha256: str
    source_canonical_sha256: str
    content_bytes_base64: str
    content_byte_length: int
    content_bytes_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_schema_version",
            _plain_int(
                self.source_schema_version,
                label="source_schema_version",
                minimum=1,
            ),
        )
        object.__setattr__(self, "source_kind", _text(self.source_kind, label="source_kind"))
        for name in (
            "source_schema_sha256",
            "source_canonical_sha256",
            "content_bytes_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        raw, decoded = _canonical_mapping_from_base64(
            self.content_bytes_base64,
            label="content_bytes_base64",
            byte_length=self.content_byte_length,
            bytes_sha256=self.content_bytes_sha256,
        )
        object.__setattr__(self, "content_byte_length", len(raw))
        if (
            type(decoded.get("schema_version")) is not int
            or decoded.get("schema_version") != self.source_schema_version
            or decoded.get("kind") != self.source_kind
            or decoded.get("canonical_sha256") != self.source_canonical_sha256
        ):
            raise PhysicalFlightContractError(
                "external source schema/kind/canonical digest differs"
            )
        expected_keys = frozenset(decoded) - {"canonical_sha256"}
        if "canonical_sha256" not in decoded or canonical_sha256(
            {key: decoded[key] for key in expected_keys}
        ) != self.source_canonical_sha256:
            raise PhysicalFlightContractError(
                "external source declared canonical SHA is invalid"
            )

    @property
    def decoded_mapping(self) -> Mapping[str, object]:
        _, decoded = _canonical_mapping_from_base64(
            self.content_bytes_base64,
            label="content_bytes_base64",
            byte_length=self.content_byte_length,
            bytes_sha256=self.content_bytes_sha256,
        )
        return decoded

    @classmethod
    def from_sealed_mapping(
        cls,
        value: object,
        *,
        expected_source_kind: str,
        source_schema_sha256: str,
    ) -> "CanonicalJsonContentPin":
        if not isinstance(value, Mapping):
            raise PhysicalFlightContractError("external source must be a mapping")
        source_kind = _text(expected_source_kind, label="expected_source_kind")
        if value.get("kind") != source_kind:
            raise PhysicalFlightContractError("external source kind differs")
        schema_version = _plain_int(
            value.get("schema_version"),
            label="external_source.schema_version",
            minimum=1,
        )
        source_sha = _sha256(
            value.get("canonical_sha256"),
            label="external_source.canonical_sha256",
        )
        raw = canonical_json_bytes(dict(value))
        return cls(
            source_schema_version=schema_version,
            source_kind=source_kind,
            source_schema_sha256=source_schema_sha256,
            source_canonical_sha256=source_sha,
            content_bytes_base64=base64.b64encode(raw).decode("ascii"),
            content_byte_length=len(raw),
            content_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class PhysicalFlightTaskRef(_SealedRecord):
    """Exact field-for-field R05 task identity."""

    KIND: ClassVar[str] = TASK_REF_KIND

    env_id: int
    reset_generation: int
    swing_generation: int
    action_uid: int
    action_slot: int
    birth_sha256: str
    sample_sha256: str
    task_sha256: str

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("env_id", 0, None),
            ("reset_generation", 1, None),
            ("swing_generation", 0, None),
            ("action_uid", 1, MAX_ACTION_UID),
            ("action_slot", 0, None),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    label=f"task_ref.{name}",
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        for name in ("birth_sha256", "sample_sha256", "task_sha256"):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), label=f"task_ref.{name}"),
            )


@dataclass(frozen=True)
class PhysicalFlightOutcomeKey(_SealedRecord):
    """Exact full 14-field C05 mailbox identity."""

    KIND: ClassVar[str] = OUTCOME_KEY_KIND
    RECORD_SCHEMA_VERSION: ClassVar[int] = 1

    env_id: int
    reset_generation: int
    swing_generation: int
    action_uid: int
    action_slot: int
    birth_sha256: str
    sample_sha256: str
    task_sha256: str
    run_id: str
    carry_chain_id: str
    shot_index: int
    source_sha256: str
    config_sha256: str
    receipt_content_sha256: str

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("env_id", 0, None),
            ("reset_generation", 1, None),
            ("swing_generation", 0, None),
            ("action_uid", 1, MAX_ACTION_UID),
            ("action_slot", 0, None),
            ("shot_index", 1, None),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    label=f"outcome_key.{name}",
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        for name in (
            "birth_sha256",
            "sample_sha256",
            "task_sha256",
            "source_sha256",
            "config_sha256",
            "receipt_content_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), label=f"outcome_key.{name}"),
            )
        for name in ("run_id", "carry_chain_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), label=f"outcome_key.{name}")
            )

    @property
    def task_ref(self) -> PhysicalFlightTaskRef:
        return PhysicalFlightTaskRef(
            env_id=self.env_id,
            reset_generation=self.reset_generation,
            swing_generation=self.swing_generation,
            action_uid=self.action_uid,
            action_slot=self.action_slot,
            birth_sha256=self.birth_sha256,
            sample_sha256=self.sample_sha256,
            task_sha256=self.task_sha256,
        )


def _f32(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhysicalFlightContractError(f"{label} must be a finite number")
    source = float(value)
    if not math.isfinite(source):
        raise PhysicalFlightContractError(f"{label} must be finite")
    try:
        result = struct.unpack(">f", struct.pack(">f", source))[0]
    except (OverflowError, struct.error) as exc:
        raise PhysicalFlightContractError(
            f"{label} is outside finite binary32"
        ) from exc
    if not math.isfinite(result):
        raise PhysicalFlightContractError(f"{label} is outside finite binary32")
    return 0.0 if result == 0.0 else result


def _f32_vector(value: object, *, width: int, label: str) -> tuple[float, ...]:
    rows = _sequence(value, label=label)
    if len(rows) != width:
        raise PhysicalFlightContractError(
            f"{label} must contain exactly {width} values"
        )
    return tuple(_f32(item, label=f"{label}[{index}]") for index, item in enumerate(rows))


def _f32_be_hex(value: float) -> str:
    clean = _f32(value, label="binary32 value")
    return struct.pack(">f", clean).hex()


def _f32_from_be_hex(value: object, *, label: str) -> float:
    if (
        type(value) is not str
        or len(value) != 8
        or any(character not in _HEX for character in value)
    ):
        raise PhysicalFlightContractError(
            f"{label} must be eight lowercase big-endian binary32 hex digits"
        )
    if value == "80000000":
        raise PhysicalFlightContractError(
            f"{label} encodes tombstoned negative zero"
        )
    result = struct.unpack(">f", bytes.fromhex(value))[0]
    if not math.isfinite(result):
        raise PhysicalFlightContractError(f"{label} must encode finite binary32")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class CanonicalPhysicalBallStateF32(_SealedRecord):
    """The complete 13-component install state, encoded only as f32 bytes."""

    KIND: ClassVar[str] = "action_ball_canonical_physical_ball_state_f32_v2"

    position_env_m: Tuple[float, float, float]
    quaternion_wxyz: Tuple[float, float, float, float]
    linear_velocity_world_mps: Tuple[float, float, float]
    angular_velocity_world_radps: Tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_env_m",
            _f32_vector(self.position_env_m, width=3, label="position_env_m"),
        )
        quaternion = _f32_vector(
            self.quaternion_wxyz, width=4, label="quaternion_wxyz"
        )
        if all(value == 0.0 for value in quaternion):
            raise PhysicalFlightContractError("quaternion_wxyz cannot be all zero")
        object.__setattr__(self, "quaternion_wxyz", quaternion)
        object.__setattr__(
            self,
            "linear_velocity_world_mps",
            _f32_vector(
                self.linear_velocity_world_mps,
                width=3,
                label="linear_velocity_world_mps",
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity_world_radps",
            _f32_vector(
                self.angular_velocity_world_radps,
                width=3,
                label="angular_velocity_world_radps",
            ),
        )

    @property
    def ordered_values(self) -> tuple[float, ...]:
        return (
            *self.position_env_m,
            *self.quaternion_wxyz,
            *self.linear_velocity_world_mps,
            *self.angular_velocity_world_radps,
        )

    @property
    def state_bytes(self) -> bytes:
        """Return the authoritative contiguous 52-byte dynamic state."""

        return b"".join(struct.pack(">f", value) for value in self.ordered_values)

    @property
    def state_bytes_sha256(self) -> str:
        return hashlib.sha256(self.state_bytes).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "runtime_dtype": RUNTIME_DTYPE,
            "quantization_contract": QUANTIZATION_CONTRACT,
            "zero_canonicalization": ZERO_CANONICALIZATION,
            "components": list(STATE_COMPONENTS),
            "state_f32_be_hex": self.state_bytes.hex(),
            "state_bytes_sha256": self.state_bytes_sha256,
        }

    @classmethod
    def _from_mapping_unpinned(cls, value: object):
        if not isinstance(value, Mapping):
            raise PhysicalFlightContractError(
                "CanonicalPhysicalBallStateF32 must be a mapping"
            )
        expected_payload = frozenset(
            (
                "schema_version",
                "kind",
                "runtime_dtype",
                "quantization_contract",
                "zero_canonicalization",
                "components",
                "state_f32_be_hex",
                "state_bytes_sha256",
            )
        )
        expected = expected_payload | {"canonical_sha256"}
        if frozenset(value) != expected:
            raise PhysicalFlightContractError(
                "CanonicalPhysicalBallStateF32 keys differ"
            )
        payload = {key: value[key] for key in expected_payload}
        if (
            payload["schema_version"] != SCHEMA_VERSION
            or payload["kind"] != cls.KIND
            or payload["runtime_dtype"] != RUNTIME_DTYPE
            or payload["quantization_contract"] != QUANTIZATION_CONTRACT
            or payload["zero_canonicalization"] != ZERO_CANONICALIZATION
            or tuple(payload["components"]) != STATE_COMPONENTS
        ):
            raise PhysicalFlightContractError(
                "CanonicalPhysicalBallStateF32 metadata differs"
            )
        declared = _sha256(
            value["canonical_sha256"], label="state.canonical_sha256"
        )
        if canonical_sha256(payload) != declared:
            raise PhysicalFlightContractError("state canonical SHA differs")
        encoded_blob = payload["state_f32_be_hex"]
        if (
            type(encoded_blob) is not str
            or len(encoded_blob) != 8 * len(STATE_COMPONENTS)
            or any(character not in _HEX for character in encoded_blob)
        ):
            raise PhysicalFlightContractError(
                "state_f32_be_hex must be one 104-character lowercase blob"
            )
        encoded = tuple(
            encoded_blob[index : index + 8]
            for index in range(0, len(encoded_blob), 8)
        )
        decoded = tuple(
            _f32_from_be_hex(item, label=f"state_f32_be_hex[{index}]")
            for index, item in enumerate(encoded)
        )
        result = cls(
            position_env_m=decoded[0:3],
            quaternion_wxyz=decoded[3:7],
            linear_velocity_world_mps=decoded[7:10],
            angular_velocity_world_radps=decoded[10:13],
        )
        declared_bytes_sha = _sha256(
            payload["state_bytes_sha256"], label="state_bytes_sha256"
        )
        if declared_bytes_sha != result.state_bytes_sha256:
            raise PhysicalFlightContractError("state byte SHA differs")
        return result


@dataclass(frozen=True)
class FrozenFlightCapacityReceipt(_SealedRecord):
    """Fresh-only, externally pinned C/H/K authority with no defaults."""

    KIND: ClassVar[str] = "action_ball_frozen_flight_capacity_receipt_v2"

    integration_status: str
    numeric_authority: CanonicalJsonContentPin
    fixed_tape_sha256: str
    clock_kind: str
    control_step_clock_root_sha256: str
    cadence_control_steps: int
    max_flight_horizon_control_steps: int
    inclusive_interval_semantics: str
    same_tick_ordering: str
    capacity_formula: str
    required_inclusive_flight_capacity: int
    configured_flight_capacity: int

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError(
                "capacity receipt lost PRE_INTEGRATION_HOLD status"
            )
        if not isinstance(self.numeric_authority, CanonicalJsonContentPin):
            raise PhysicalFlightContractError(
                "numeric_authority must contain complete pinned source bytes"
            )
        object.__setattr__(
            self,
            "fixed_tape_sha256",
            _sha256(self.fixed_tape_sha256, label="fixed_tape_sha256"),
        )
        object.__setattr__(self, "clock_kind", _text(self.clock_kind, label="clock_kind"))
        object.__setattr__(
            self,
            "control_step_clock_root_sha256",
            _sha256(
                self.control_step_clock_root_sha256,
                label="control_step_clock_root_sha256",
            ),
        )
        cadence = _plain_int(
            self.cadence_control_steps,
            label="cadence_control_steps",
            minimum=1,
        )
        horizon = _plain_int(
            self.max_flight_horizon_control_steps,
            label="max_flight_horizon_control_steps",
            minimum=0,
        )
        required = _plain_int(
            self.required_inclusive_flight_capacity,
            label="required_inclusive_flight_capacity",
            minimum=1,
        )
        configured = _plain_int(
            self.configured_flight_capacity,
            label="configured_flight_capacity",
            minimum=1,
        )
        if (
            self.inclusive_interval_semantics != INCLUSIVE_INTERVAL_SEMANTICS
            or self.same_tick_ordering != SAME_TICK_ORDERING
            or self.capacity_formula != CAPACITY_FORMULA
        ):
            raise PhysicalFlightContractError(
                "capacity inclusive semantics/formula differs"
            )
        derived = horizon // cadence + 1
        if required != derived or configured != required:
            raise PhysicalFlightContractError(
                "capacity must equal the frozen inclusive C/H derivation"
            )
        authority = self.numeric_authority.decoded_mapping
        required_authority_fields = (
            "clock_kind",
            "control_step_clock_root_sha256",
            "cadence_control_steps",
            "max_flight_horizon_control_steps",
            "flight_capacity",
            "inclusive_interval_semantics",
            "same_tick_ordering",
            "fixed_tape_sha256",
            "source_sha256",
            "config_sha256",
            "contract_sha256",
        )
        if any(name not in authority for name in required_authority_fields):
            raise PhysicalFlightContractError(
                "numeric authority lacks clock/C/H/K/source/config/contract pins"
            )
        for name in ("source_sha256", "config_sha256", "contract_sha256"):
            _sha256(authority[name], label=f"numeric_authority.{name}")
        if (
            authority["clock_kind"] != self.clock_kind
            or authority["control_step_clock_root_sha256"]
            != self.control_step_clock_root_sha256
            or type(authority["cadence_control_steps"]) is not int
            or authority["cadence_control_steps"] != cadence
            or type(authority["max_flight_horizon_control_steps"]) is not int
            or authority["max_flight_horizon_control_steps"] != horizon
            or type(authority["flight_capacity"]) is not int
            or authority["flight_capacity"] != configured
            or authority["inclusive_interval_semantics"]
            != INCLUSIVE_INTERVAL_SEMANTICS
            or authority["same_tick_ordering"] != SAME_TICK_ORDERING
            or authority["fixed_tape_sha256"] != self.fixed_tape_sha256
        ):
            raise PhysicalFlightContractError(
                "numeric authority content differs from frozen C/H/K receipt"
            )
        object.__setattr__(self, "cadence_control_steps", cadence)
        object.__setattr__(self, "max_flight_horizon_control_steps", horizon)
        object.__setattr__(self, "required_inclusive_flight_capacity", required)
        object.__setattr__(self, "configured_flight_capacity", configured)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["numeric_authority"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["numeric_authority"]
        )
        return values


def installed_ball_state_binding_sha256(
    *,
    state_f32_sha256: str,
    frame_id: str,
    frame_binding_sha256: str,
    env_id: int,
    reveal_control_step: int,
    selected_contact_deadline_control_step: int,
    first_crossing_horizon_control_step: int,
    task_ref_sha256: str,
    outcome_key_sha256: str,
) -> str:
    """Hash the f32 state together with its frame, epoch, timing, and owner."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_ball_installed_state_binding_v2",
        "runtime_dtype": RUNTIME_DTYPE,
        "quantization_contract": QUANTIZATION_CONTRACT,
        "zero_canonicalization": ZERO_CANONICALIZATION,
        "position_frame": POSITION_FRAME,
        "quaternion_order": QUATERNION_ORDER,
        "linear_velocity_frame": LINEAR_VELOCITY_FRAME,
        "angular_velocity_frame": ANGULAR_VELOCITY_FRAME,
        "state_epoch": INSTALL_STATE_EPOCH,
        "state_f32_sha256": _sha256(
            state_f32_sha256, label="state_f32_sha256"
        ),
        "frame_id": _text(frame_id, label="frame_id"),
        "frame_binding_sha256": _sha256(
            frame_binding_sha256, label="frame_binding_sha256"
        ),
        "env_id": _plain_int(env_id, label="env_id"),
        "reveal_control_step": _plain_int(
            reveal_control_step, label="reveal_control_step"
        ),
        "selected_contact_deadline_control_step": _plain_int(
            selected_contact_deadline_control_step,
            label="selected_contact_deadline_control_step",
        ),
        "first_crossing_horizon_control_step": _plain_int(
            first_crossing_horizon_control_step,
            label="first_crossing_horizon_control_step",
        ),
        "task_ref_sha256": _sha256(
            task_ref_sha256, label="task_ref_sha256"
        ),
        "outcome_key_sha256": _sha256(
            outcome_key_sha256, label="outcome_key_sha256"
        ),
    }
    if payload["selected_contact_deadline_control_step"] <= payload[
        "reveal_control_step"
    ] or payload["first_crossing_horizon_control_step"] < payload[
        "selected_contact_deadline_control_step"
    ]:
        raise PhysicalFlightContractError("installed state timing order differs")
    return canonical_sha256(payload)


@dataclass(frozen=True)
class PhysicalBallInstallPayload(_SealedRecord):
    """Complete fresh physical-ball install payload; digest-only is forbidden."""

    KIND: ClassVar[str] = "action_ball_physical_ball_install_payload_v2"

    integration_status: str
    capacity_receipt: FrozenFlightCapacityReceipt
    capacity_receipt_sha256: str
    env_id: int
    flight_slot: int
    ball_generation: int
    task_ref: PhysicalFlightTaskRef
    task_ref_sha256: str
    outcome_key: PhysicalFlightOutcomeKey
    outcome_key_sha256: str
    ball_construction_receipt_sha256: str
    inbound_ball_sha256: str
    frame_id: str
    frame_binding_authority: CanonicalJsonContentPin
    frame_binding_sha256: str
    position_frame: str
    quaternion_order: str
    linear_velocity_frame: str
    angular_velocity_frame: str
    state_epoch: str
    reveal_control_step: int
    selected_contact_deadline_control_step: int
    first_crossing_horizon_control_step: int
    state_f32: CanonicalPhysicalBallStateF32
    state_f32_sha256: str
    installed_ball_state_sha256: str

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError(
                "install payload lost PRE_INTEGRATION_HOLD status"
            )
        if not isinstance(self.capacity_receipt, FrozenFlightCapacityReceipt):
            raise PhysicalFlightContractError("capacity_receipt type differs")
        if not isinstance(self.task_ref, PhysicalFlightTaskRef):
            raise PhysicalFlightContractError("task_ref type differs")
        if not isinstance(self.outcome_key, PhysicalFlightOutcomeKey):
            raise PhysicalFlightContractError("outcome_key type differs")
        if not isinstance(self.state_f32, CanonicalPhysicalBallStateF32):
            raise PhysicalFlightContractError("state_f32 type differs")
        if not isinstance(self.frame_binding_authority, CanonicalJsonContentPin):
            raise PhysicalFlightContractError(
                "frame binding must contain complete pinned transform bytes"
            )
        for name in (
            "capacity_receipt_sha256",
            "task_ref_sha256",
            "outcome_key_sha256",
            "ball_construction_receipt_sha256",
            "inbound_ball_sha256",
            "frame_binding_sha256",
            "state_f32_sha256",
            "installed_ball_state_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        env_id = _plain_int(self.env_id, label="env_id")
        slot = _plain_int(self.flight_slot, label="flight_slot")
        generation = _plain_int(self.ball_generation, label="ball_generation")
        if slot >= self.capacity_receipt.configured_flight_capacity:
            raise PhysicalFlightContractError("flight_slot is outside frozen capacity")
        if (
            self.capacity_receipt_sha256
            != self.capacity_receipt.canonical_sha256
            or self.task_ref_sha256 != self.task_ref.canonical_sha256
            or self.outcome_key_sha256 != self.outcome_key.canonical_sha256
            or self.state_f32_sha256 != self.state_f32.state_bytes_sha256
            or self.frame_binding_sha256
            != self.frame_binding_authority.source_canonical_sha256
        ):
            raise PhysicalFlightContractError("install nested content digest differs")
        if self.task_ref != self.outcome_key.task_ref:
            raise PhysicalFlightContractError("task/outcome full identity differs")
        if (
            env_id != self.task_ref.env_id
            or generation != self.task_ref.swing_generation
        ):
            raise PhysicalFlightContractError("install env/generation identity differs")
        object.__setattr__(self, "frame_id", _text(self.frame_id, label="frame_id"))
        frame_authority = self.frame_binding_authority.decoded_mapping
        required_frame_fields = (
            "frame_id",
            "env_id",
            "transform_semantics",
            "env_origin_world_f32_be_hex",
        )
        if any(name not in frame_authority for name in required_frame_fields):
            raise PhysicalFlightContractError(
                "frame binding authority lacks frame/env/transform content"
            )
        origin_hex = frame_authority["env_origin_world_f32_be_hex"]
        if (
            frame_authority["frame_id"] != self.frame_id
            or type(frame_authority["env_id"]) is not int
            or frame_authority["env_id"] != env_id
            or frame_authority["transform_semantics"] != POSITION_FRAME
            or not isinstance(origin_hex, Sequence)
            or isinstance(origin_hex, (str, bytes))
            or len(origin_hex) != 3
        ):
            raise PhysicalFlightContractError(
                "frame binding authority differs from install frame/env"
            )
        for index, component in enumerate(origin_hex):
            _f32_from_be_hex(
                component,
                label=f"frame_binding.env_origin_world_f32_be_hex[{index}]",
            )
        if (
            self.position_frame != POSITION_FRAME
            or self.quaternion_order != QUATERNION_ORDER
            or self.linear_velocity_frame != LINEAR_VELOCITY_FRAME
            or self.angular_velocity_frame != ANGULAR_VELOCITY_FRAME
            or self.state_epoch != INSTALL_STATE_EPOCH
        ):
            raise PhysicalFlightContractError("install frame/state epoch differs")
        reveal = _plain_int(
            self.reveal_control_step, label="reveal_control_step"
        )
        deadline = _plain_int(
            self.selected_contact_deadline_control_step,
            label="selected_contact_deadline_control_step",
        )
        horizon = _plain_int(
            self.first_crossing_horizon_control_step,
            label="first_crossing_horizon_control_step",
        )
        if horizon != reveal + self.capacity_receipt.max_flight_horizon_control_steps:
            raise PhysicalFlightContractError(
                "first-crossing horizon must equal reveal plus frozen H"
            )
        expected_installed_sha = installed_ball_state_binding_sha256(
            state_f32_sha256=self.state_f32_sha256,
            frame_id=self.frame_id,
            frame_binding_sha256=self.frame_binding_sha256,
            env_id=env_id,
            reveal_control_step=reveal,
            selected_contact_deadline_control_step=deadline,
            first_crossing_horizon_control_step=horizon,
            task_ref_sha256=self.task_ref_sha256,
            outcome_key_sha256=self.outcome_key_sha256,
        )
        if self.installed_ball_state_sha256 != expected_installed_sha:
            raise PhysicalFlightContractError(
                "installed_ball_state_sha256 does not bind complete state semantics"
            )
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "flight_slot", slot)
        object.__setattr__(self, "ball_generation", generation)
        object.__setattr__(self, "reveal_control_step", reveal)
        object.__setattr__(
            self, "selected_contact_deadline_control_step", deadline
        )
        object.__setattr__(self, "first_crossing_horizon_control_step", horizon)

    @classmethod
    def _reject_legacy(cls, value: object) -> None:
        if not isinstance(value, Mapping):
            return
        if value.get("kind") in _LEGACY_DIGEST_ONLY_INSTALL_KINDS or (
            "installed_ball_state_sha256" in value and "state_f32" not in value
        ):
            raise DigestOnlyPayloadTombstonedError(
                "digest-only physical-ball install payload is tombstoned"
            )

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["capacity_receipt"] = FrozenFlightCapacityReceipt._from_mapping_unpinned(
            values["capacity_receipt"]
        )
        values["task_ref"] = PhysicalFlightTaskRef._from_mapping_unpinned(
            values["task_ref"]
        )
        values["outcome_key"] = PhysicalFlightOutcomeKey._from_mapping_unpinned(
            values["outcome_key"]
        )
        values["frame_binding_authority"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["frame_binding_authority"]
        )
        values["state_f32"] = CanonicalPhysicalBallStateF32._from_mapping_unpinned(
            values["state_f32"]
        )
        return values


def _outcome_digest(value: Optional[PhysicalFlightOutcomeKey]) -> Optional[str]:
    return None if value is None else value.canonical_sha256


@dataclass(frozen=True)
class PhysicalFlightSlotSnapshot(_SealedRecord):
    """One read-only scene slot joined to its complete owner key.

    ``mutation_version`` is a per-slot counter.  It advances only when this
    slot changes, so a masked true reset can prove that every unselected slot
    remained byte-exact.  Owner-wide operation ordering is carried separately
    by receipt/checkpoint ``mutation_version`` fields.
    """

    KIND: ClassVar[str] = "action_ball_physical_flight_slot_snapshot_v2"

    capacity_receipt_sha256: str
    capacity_value: int
    env_id: int
    slot_index: int
    scene_body_name: str
    lifecycle: str
    ball_generation: Optional[int]
    inbound_ball_sha256: Optional[str]
    outcome_key: Optional[PhysicalFlightOutcomeKey]
    outcome_key_sha256: Optional[str]
    install_payload_sha256: Optional[str]
    installed_ball_state_sha256: Optional[str]
    current_state_f32: Optional[CanonicalPhysicalBallStateF32]
    current_state_f32_sha256: Optional[str]
    reveal_control_step: Optional[int]
    last_control_step: int
    last_physics_substep: int
    last_sim_step: int
    mutation_version: int
    physically_parked: bool
    published_to_runtime: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capacity_receipt_sha256",
            _sha256(self.capacity_receipt_sha256, label="capacity_receipt_sha256"),
        )
        capacity = _plain_int(self.capacity_value, label="capacity_value", minimum=1)
        env_id = _plain_int(self.env_id, label="env_id")
        slot = _plain_int(self.slot_index, label="slot_index")
        if slot >= capacity:
            raise PhysicalFlightContractError("slot_index is outside capacity")
        object.__setattr__(
            self, "scene_body_name", _text(self.scene_body_name, label="scene_body_name")
        )
        lifecycle = _text(self.lifecycle, label="lifecycle")
        if lifecycle not in SLOT_LIFECYCLES:
            raise PhysicalFlightContractError("slot lifecycle is unknown")
        parked = _exact_bool(self.physically_parked, label="physically_parked")
        published = _exact_bool(
            self.published_to_runtime, label="published_to_runtime"
        )
        generation = _optional_plain_int(
            self.ball_generation, label="ball_generation"
        )
        inbound_sha = _optional_sha256(
            self.inbound_ball_sha256, label="inbound_ball_sha256"
        )
        outcome_sha = _optional_sha256(
            self.outcome_key_sha256, label="outcome_key_sha256"
        )
        install_sha = _optional_sha256(
            self.install_payload_sha256, label="install_payload_sha256"
        )
        installed_state_sha = _optional_sha256(
            self.installed_ball_state_sha256,
            label="installed_ball_state_sha256",
        )
        current_sha = _optional_sha256(
            self.current_state_f32_sha256,
            label="current_state_f32_sha256",
        )
        reveal = _optional_plain_int(
            self.reveal_control_step, label="reveal_control_step"
        )
        for name in (
            "last_control_step",
            "last_physics_substep",
            "last_sim_step",
            "mutation_version",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        if lifecycle == SLOT_PARKED:
            if any(
                value is not None
                for value in (
                    generation,
                    inbound_sha,
                    self.outcome_key,
                    outcome_sha,
                    install_sha,
                    installed_state_sha,
                    self.current_state_f32,
                    current_sha,
                    reveal,
                )
            ) or not parked or published:
                raise PhysicalFlightContractError(
                    "parked slot carries a live owner/state/publication"
                )
        else:
            if not isinstance(self.outcome_key, PhysicalFlightOutcomeKey) or not isinstance(
                self.current_state_f32, CanonicalPhysicalBallStateF32
            ):
                raise PhysicalFlightContractError(
                    "owned slot lacks complete outcome key or current state"
                )
            if any(
                value is None
                for value in (
                    generation,
                    inbound_sha,
                    outcome_sha,
                    install_sha,
                    installed_state_sha,
                    current_sha,
                    reveal,
                )
            ):
                raise PhysicalFlightContractError("owned slot identity is incomplete")
            if (
                outcome_sha != self.outcome_key.canonical_sha256
                or current_sha != self.current_state_f32.state_bytes_sha256
                or env_id != self.outcome_key.env_id
                or generation != self.outcome_key.swing_generation
            ):
                raise PhysicalFlightContractError("slot owner/state join differs")
            should_be_live = lifecycle in (SLOT_IN_FLIGHT, SLOT_SETTLED_RETAINED)
            if parked == should_be_live or published != should_be_live:
                raise PhysicalFlightContractError(
                    "slot physical/public lifecycle flags differ"
                )
        object.__setattr__(self, "capacity_value", capacity)
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "slot_index", slot)
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(self, "ball_generation", generation)
        object.__setattr__(self, "inbound_ball_sha256", inbound_sha)
        object.__setattr__(self, "outcome_key_sha256", outcome_sha)
        object.__setattr__(self, "install_payload_sha256", install_sha)
        object.__setattr__(self, "installed_ball_state_sha256", installed_state_sha)
        object.__setattr__(self, "current_state_f32_sha256", current_sha)
        object.__setattr__(self, "reveal_control_step", reveal)
        object.__setattr__(self, "physically_parked", parked)
        object.__setattr__(self, "published_to_runtime", published)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        if values["outcome_key"] is not None:
            values["outcome_key"] = PhysicalFlightOutcomeKey._from_mapping_unpinned(
                values["outcome_key"]
            )
        if values["current_state_f32"] is not None:
            values["current_state_f32"] = CanonicalPhysicalBallStateF32._from_mapping_unpinned(
                values["current_state_f32"]
            )
        return values


def physical_slot_root(
    slots: Sequence[PhysicalFlightSlotSnapshot],
) -> str:
    rows = tuple(slots)
    if any(not isinstance(row, PhysicalFlightSlotSnapshot) for row in rows):
        raise PhysicalFlightContractError("slot root requires typed snapshots")
    keys = tuple((row.env_id, row.slot_index) for row in rows)
    if keys != tuple(sorted(set(keys))):
        raise PhysicalFlightContractError(
            "slot root rows must be sorted and unique by env/slot"
        )
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "action_ball_physical_slot_root_v2",
            "rows": [
                {
                    "env_id": row.env_id,
                    "slot_index": row.slot_index,
                    "slot_snapshot_sha256": row.canonical_sha256,
                }
                for row in rows
            ],
        }
    )


def physical_owner_checkpoint_root(
    *,
    capacity_receipt_sha256: str,
    num_envs: int,
    flight_capacity: int,
    mutation_version: int,
    next_prepare_nonce: int,
    reset_generations: Sequence[int],
    slots: Sequence[PhysicalFlightSlotSnapshot],
    poisoned: bool,
) -> str:
    """Canonical complete portable-owner root used by mutation receipts."""

    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "action_ball_physical_owner_checkpoint_root_v1",
            "capacity_receipt_sha256": _sha256(
                capacity_receipt_sha256,
                label="capacity_receipt_sha256",
            ),
            "num_envs": _plain_int(num_envs, label="num_envs", minimum=1),
            "flight_capacity": _plain_int(
                flight_capacity,
                label="flight_capacity",
                minimum=1,
            ),
            "mutation_version": _plain_int(
                mutation_version,
                label="mutation_version",
            ),
            "next_prepare_nonce": _plain_int(
                next_prepare_nonce,
                label="next_prepare_nonce",
                minimum=1,
            ),
            "reset_generations": [
                _plain_int(item, label="reset_generations[]", minimum=1)
                for item in _sequence(
                    reset_generations,
                    label="reset_generations",
                )
            ],
            "slot_root_sha256": physical_slot_root(tuple(slots)),
            "poisoned": _exact_bool(poisoned, label="poisoned"),
        }
    )


@dataclass(frozen=True)
class PreparedPhysicalInstallRow(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_prepared_physical_install_row_v2"

    env_id: int
    slot_index: int
    pre_slot_snapshot: PhysicalFlightSlotSnapshot
    pre_slot_snapshot_sha256: str
    install_payload: PhysicalBallInstallPayload
    install_payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.pre_slot_snapshot, PhysicalFlightSlotSnapshot):
            raise PhysicalFlightContractError("pre_slot_snapshot type differs")
        if not isinstance(self.install_payload, PhysicalBallInstallPayload):
            raise PhysicalFlightContractError("install_payload type differs")
        env_id = _plain_int(self.env_id, label="env_id")
        slot = _plain_int(self.slot_index, label="slot_index")
        for name in ("pre_slot_snapshot_sha256", "install_payload_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        if (
            self.pre_slot_snapshot_sha256
            != self.pre_slot_snapshot.canonical_sha256
            or self.install_payload_sha256 != self.install_payload.canonical_sha256
            or (env_id, slot)
            != (self.pre_slot_snapshot.env_id, self.pre_slot_snapshot.slot_index)
            or (env_id, slot)
            != (self.install_payload.env_id, self.install_payload.flight_slot)
            or self.pre_slot_snapshot.capacity_receipt_sha256
            != self.install_payload.capacity_receipt_sha256
            or self.pre_slot_snapshot.lifecycle
            not in (SLOT_PARKED, SLOT_RETIRED)
            or not self.pre_slot_snapshot.physically_parked
            or self.pre_slot_snapshot.published_to_runtime
        ):
            raise PhysicalFlightContractError("prepared physical row binding differs")
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "slot_index", slot)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["pre_slot_snapshot"] = PhysicalFlightSlotSnapshot._from_mapping_unpinned(
            values["pre_slot_snapshot"]
        )
        values["install_payload"] = PhysicalBallInstallPayload._from_mapping_unpinned(
            values["install_payload"]
        )
        return values


_R05_PREVIEW_FIELDS = frozenset(
    (
        "integration_status",
        "phase",
        "public_visible",
        "policy_opportunity_created",
        "owner_checkpoint_before_sha256",
        "prepared_batch",
        "sampler_checkpoint_before_commit_sha256",
        "sampler_checkpoint_after_commit_sha256",
        "untouched_rows_before_sha256",
        "untouched_rows_after_sha256",
        "sampler_checkpoint_before_commit",
        "sampler_checkpoint_after_commit",
        "reveal_final_rows",
        "all_owner_install_root_sha256",
    )
)
_R05_REVEAL_FINAL_ROW_FIELDS = frozenset(
    (
        "integration_status",
        "phase",
        "public_visible",
        "policy_opportunity_created",
        "prepared_reveal",
        "reveal_facts",
        "ball_slot_plan",
        "selected_task_ref_sha256",
        "outcome_key_sha256",
        "physical_ball_install_payload_sha256",
        "pre_install_ball_slots",
        "post_install_ball_slots",
    )
)
_R05_BALL_SLOT_PLAN_FIELDS = frozenset(
    (
        "capacity",
        "snapshot_sha256",
        "selected_slot_index",
        "previous_slot_index",
        "reused_previous_slot",
        "preserved_live_owner_key_sha256",
        "new_ball_generation",
        "new_inbound_ball_sha256",
        "new_ball_dynamic_state_sha256",
        "physical_ball_install_payload_sha256",
        "reused_retired_owner_key_sha256",
    )
)
_R05_COMMITTED_BATCH_FIELDS = frozenset(
    (
        "integration_status",
        "phase",
        "runtime_wiring_connected",
        "identity_committed",
        "policy_opportunity_created",
        "reveal_final_preview",
        "prepared_batch",
        "sampler_checkpoint_before_commit_sha256",
        "sampler_checkpoint_after_commit_sha256",
        "untouched_rows_before_sha256",
        "untouched_rows_after_sha256",
        "sampler_checkpoint_before_commit",
        "sampler_checkpoint_after_commit",
        "committed_reveals",
    )
)
_R05_COMMITTED_REVEAL_FIELDS = frozenset(
    (
        "integration_status",
        "phase",
        "runtime_wiring_connected",
        "identity_committed",
        "policy_opportunity_created",
        "prepared_reveal",
        "reveal_facts",
        "ball_slot_plan",
        "playback_release_requested",
    )
)
_R05_REVEAL_PREPARE_BOUNDARY_MARKER_FIELDS = frozenset(
    (
        "selected_env_ids",
        "reveal_final_preview_sha256",
        "boundary_packet_version",
        "boundary_packet_root_sha256",
        "boundary_transfer_count",
        "selected_pass_count",
        "selected_fault_count",
        "ordered_child_token_roots",
    )
)
_FULL_MDP_REVEAL_BOUNDARY_RECEIPT_FIELDS = frozenset(
    (
        "packet_schema_version",
        "boundary_sequence",
        "reveal_final_preview_schema_version",
        "reveal_final_preview_sha256",
        "num_envs",
        "selected_env_ids",
        "ordered_owner_kinds",
        "ordered_owner_rows",
        "packet_nbytes",
        "packet_sha256",
        "device_type",
        "device_index",
        "boundary_transfer_count",
        "transfer_attempt_count_total",
        "transfer_success_count_total",
        "transfer_bytes_total",
        "transfer_elapsed_ns_total",
        "selected_pass_count",
        "selected_fault_count",
        "decision",
        "d05_construction_admissible",
        "d05_owner_fault_present",
        "d05_selected_primary_fault",
    )
)
_FULL_MDP_REVEAL_BOUNDARY_OWNER_ROW_FIELDS = frozenset(
    (
        "kind",
        "owner_kind",
        "owner_mutation_version",
        "owner_token_root_sha256",
        "fault_schema_sha256",
        "allowed_fault_mask",
        "selected_pass",
        "selected_fault_bits",
    )
)
_PHYSICAL_SETTLEMENT_AUTHORITY_FIELDS = frozenset(
    (
        "mailbox_lifecycle",
        "r06_owner_mutation_version",
        "r06_after_root_sha256",
        "physical_retire_rows",
    )
)
_PHYSICAL_ZERO_OPEN_RESET_CLOSURE_FIELDS = frozenset(
    (
        "selected_env_ids",
        "open_flight_count",
        "open_mailbox_count",
    )
)
_R05_TERMINAL_BOUNDARY_PROJECTION_FIELDS = frozenset(
    (
        "authority_domain",
        "authority_schema_sha256",
        "authority_source_sha256",
        "decision_mapping_schema_version",
        "source_decision",
        "decision",
        "reveal_final_preview_schema_version",
        "reveal_final_preview_sha256",
        "selected_env_ids",
        "boundary_receipt_kind",
        "boundary_receipt_sha256",
        "boundary_packet_schema_version",
        "boundary_packet_sha256",
        "ordered_participant_roots",
        "ordered_censor_evidence",
    )
)
_R05_TERMINAL_BOUNDARY_PARTICIPANT_ROOT_FIELDS = frozenset(
    ("participant_domain", "participant_kind", "participant_root_sha256")
)
_R05_TERMINAL_BOUNDARY_CENSOR_EVIDENCE_FIELDS = frozenset(
    (
        "env_id",
        "primary_failure_env_id",
        "participant_domain",
        "participant_kind",
        "participant_root_sha256",
        "failure_receipt_sha256",
        "reason",
        "censor_fact_sha256",
        "producer_schema_sha256",
        "producer_source_sha256",
    )
)
_R05_PREPARED_TERMINAL_CONTENT_PIN_FIELDS = frozenset(
    (
        "terminal_schema_version",
        "terminal_kind",
        "terminal_canonical_sha256",
        "content_bytes_base64",
        "content_byte_length",
        "content_bytes_sha256",
    )
)


def _verified_reveal_prepare_boundary_marker(
    pin: CanonicalJsonContentPin,
) -> tuple[Mapping[str, object], tuple[tuple[str, str], ...]]:
    """Validate the complete sealed R05 global prearm marker bytes."""

    if not isinstance(pin, CanonicalJsonContentPin):
        raise PhysicalFlightContractError(
            "commit lacks the full R05 reveal-prepare boundary marker"
        )
    if (
        pin.source_kind != REVEAL_PREPARE_BOUNDARY_MARKER_KIND
        or pin.source_schema_version != 1
        or pin.source_schema_sha256
        != REVEAL_PREPARE_BOUNDARY_MARKER_SCHEMA_SHA256
    ):
        raise PhysicalFlightContractError(
            "reveal-prepare boundary marker schema pin differs"
        )
    marker = _verified_external_sealed_mapping(
        pin.decoded_mapping,
        label="global_prearm_marker",
        expected_kind=REVEAL_PREPARE_BOUNDARY_MARKER_KIND,
        expected_schema_version=1,
        expected_payload_fields=_R05_REVEAL_PREPARE_BOUNDARY_MARKER_FIELDS,
    )
    selected = _ordered_unique_env_ids(
        marker["selected_env_ids"],
        label="global_prearm_marker.selected_env_ids",
    )
    preview_sha = _sha256(
        marker["reveal_final_preview_sha256"],
        label="global_prearm_marker.reveal_final_preview_sha256",
    )
    packet_version = _plain_int(
        marker["boundary_packet_version"],
        label="global_prearm_marker.boundary_packet_version",
        minimum=1,
    )
    packet_root = _sha256(
        marker["boundary_packet_root_sha256"],
        label="global_prearm_marker.boundary_packet_root_sha256",
    )
    transfer_count = _plain_int(
        marker["boundary_transfer_count"],
        label="global_prearm_marker.boundary_transfer_count",
    )
    pass_count = _plain_int(
        marker["selected_pass_count"],
        label="global_prearm_marker.selected_pass_count",
    )
    fault_count = _plain_int(
        marker["selected_fault_count"],
        label="global_prearm_marker.selected_fault_count",
    )
    raw_roots = _sequence(
        marker["ordered_child_token_roots"],
        label="global_prearm_marker.ordered_child_token_roots",
    )
    if len(raw_roots) != len(R05_PREARM_CHILD_OWNER_KINDS):
        raise PhysicalFlightContractError(
            "reveal-prepare boundary child-token width differs"
        )
    roots: list[tuple[str, str]] = []
    for index, (expected_kind, raw) in enumerate(
        zip(R05_PREARM_CHILD_OWNER_KINDS, raw_roots)
    ):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
            or type(raw[0]) is not str
            or raw[0] != expected_kind
        ):
            raise PhysicalFlightContractError(
                "reveal-prepare boundary child-token order/kind differs"
            )
        roots.append(
            (
                expected_kind,
                _sha256(
                    raw[1],
                    label=(
                        "global_prearm_marker.ordered_child_token_roots"
                        f"[{index}][1]"
                    ),
                ),
            )
        )
    if transfer_count != 1 or pass_count != len(selected) or fault_count != 0:
        raise PhysicalFlightContractError(
            "reveal-prepare boundary is not one all-selected pass packet"
        )
    # Make every parsed primitive part of the validation, even though the
    # canonical nested mapping itself is returned unchanged.
    if not preview_sha or packet_version < 1 or not packet_root:
        raise PhysicalFlightContractError("reveal-prepare boundary marker differs")
    return marker, tuple(roots)


def _verified_full_mdp_reveal_boundary_receipt(
    pin: CanonicalJsonContentPin,
    *,
    expected_decision: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Validate the complete portable receipt bytes without granting authority."""

    if type(pin) is not CanonicalJsonContentPin:
        raise PhysicalFlightContractError(
            "physical terminal lacks the full all-owner boundary receipt"
        )
    if (
        pin.source_kind != FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND
        or pin.source_schema_version != 1
        or pin.source_schema_sha256
        != FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
    ):
        raise PhysicalFlightContractError(
            "all-owner reveal boundary receipt schema pin differs"
        )
    receipt = _verified_external_sealed_mapping(
        pin.decoded_mapping,
        label="global_reveal_boundary_receipt",
        expected_kind=FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
        expected_schema_version=1,
        expected_payload_fields=_FULL_MDP_REVEAL_BOUNDARY_RECEIPT_FIELDS,
    )
    packet_schema = _plain_int(
        receipt["packet_schema_version"],
        label="global_reveal_boundary_receipt.packet_schema_version",
        minimum=1,
    )
    sequence = _plain_int(
        receipt["boundary_sequence"],
        label="global_reveal_boundary_receipt.boundary_sequence",
        minimum=1,
    )
    preview_schema = _plain_int(
        receipt["reveal_final_preview_schema_version"],
        label=(
            "global_reveal_boundary_receipt."
            "reveal_final_preview_schema_version"
        ),
        minimum=1,
    )
    preview_root = _sha256(
        receipt["reveal_final_preview_sha256"],
        label="global_reveal_boundary_receipt.reveal_final_preview_sha256",
    )
    num_envs = _plain_int(
        receipt["num_envs"],
        label="global_reveal_boundary_receipt.num_envs",
        minimum=1,
    )
    selected = _ordered_unique_env_ids(
        receipt["selected_env_ids"],
        label="global_reveal_boundary_receipt.selected_env_ids",
    )
    if selected[-1] >= num_envs:
        raise PhysicalFlightContractError(
            "all-owner reveal boundary selected environment is out of range"
        )
    kinds = tuple(
        _text(value, label="global_reveal_boundary_receipt.owner_kind")
        for value in _sequence(
            receipt["ordered_owner_kinds"],
            label="global_reveal_boundary_receipt.ordered_owner_kinds",
        )
    )
    raw_rows = _sequence(
        receipt["ordered_owner_rows"],
        label="global_reveal_boundary_receipt.ordered_owner_rows",
    )
    if (
        packet_schema != FULL_MDP_REVEAL_BOUNDARY_PACKET_SCHEMA_VERSION
        or kinds != FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER
        or len(raw_rows) != len(kinds)
    ):
        raise PhysicalFlightContractError(
            "all-owner reveal boundary packet/owner order differs"
        )
    rows: list[Mapping[str, object]] = []
    for index, (kind, raw) in enumerate(zip(kinds, raw_rows)):
        if not isinstance(raw, Mapping) or frozenset(raw) != (
            _FULL_MDP_REVEAL_BOUNDARY_OWNER_ROW_FIELDS
        ):
            raise PhysicalFlightContractError(
                "all-owner reveal boundary owner-row fields differ"
            )
        if (
            raw.get("kind")
            != "action_ball_full_mdp_reveal_boundary_owner_row_v1"
            or raw.get("owner_kind") != kind
        ):
            raise PhysicalFlightContractError(
                "all-owner reveal boundary owner-row kind/order differs"
            )
        _plain_int(
            raw["owner_mutation_version"],
            label=f"global_reveal_boundary_receipt.owner_rows[{index}].version",
        )
        _sha256(
            raw["owner_token_root_sha256"],
            label=f"global_reveal_boundary_receipt.owner_rows[{index}].token",
        )
        _sha256(
            raw["fault_schema_sha256"],
            label=f"global_reveal_boundary_receipt.owner_rows[{index}].schema",
        )
        allowed = _plain_int(
            raw["allowed_fault_mask"],
            label=f"global_reveal_boundary_receipt.owner_rows[{index}].mask",
            minimum=1,
            maximum=(1 << 63) - 1,
        )
        passes = tuple(
            _exact_bool(
                value,
                label=(
                    f"global_reveal_boundary_receipt.owner_rows[{index}]."
                    "selected_pass[]"
                ),
            )
            for value in _sequence(
                raw["selected_pass"],
                label=(
                    f"global_reveal_boundary_receipt.owner_rows[{index}]."
                    "selected_pass"
                ),
            )
        )
        faults = tuple(
            _plain_int(
                value,
                label=(
                    f"global_reveal_boundary_receipt.owner_rows[{index}]."
                    "selected_fault_bits[]"
                ),
            )
            for value in _sequence(
                raw["selected_fault_bits"],
                label=(
                    f"global_reveal_boundary_receipt.owner_rows[{index}]."
                    "selected_fault_bits"
                ),
            )
        )
        if (
            len(passes) != len(selected)
            or len(faults) != len(selected)
            or any(fault & ~allowed for fault in faults)
            or any(
                (passed and fault != 0)
                or ((not passed) and fault == 0)
                for passed, fault in zip(passes, faults)
            )
        ):
            raise PhysicalFlightContractError(
                "all-owner reveal boundary owner-row verdict differs"
            )
        rows.append(raw)
    packet_nbytes = _plain_int(
        receipt["packet_nbytes"],
        label="global_reveal_boundary_receipt.packet_nbytes",
        minimum=1,
    )
    _sha256(
        receipt["packet_sha256"],
        label="global_reveal_boundary_receipt.packet_sha256",
    )
    device_type = _text(
        receipt["device_type"],
        label="global_reveal_boundary_receipt.device_type",
    )
    device_index = receipt["device_index"]
    if device_index is not None:
        _plain_int(
            device_index,
            label="global_reveal_boundary_receipt.device_index",
        )
    attempt_total = _plain_int(
        receipt["transfer_attempt_count_total"],
        label="global_reveal_boundary_receipt.transfer_attempt_count_total",
        minimum=1,
    )
    success_total = _plain_int(
        receipt["transfer_success_count_total"],
        label="global_reveal_boundary_receipt.transfer_success_count_total",
        minimum=1,
    )
    bytes_total = _plain_int(
        receipt["transfer_bytes_total"],
        label="global_reveal_boundary_receipt.transfer_bytes_total",
        minimum=1,
    )
    _plain_int(
        receipt["transfer_elapsed_ns_total"],
        label="global_reveal_boundary_receipt.transfer_elapsed_ns_total",
    )
    pass_count = _plain_int(
        receipt["selected_pass_count"],
        label="global_reveal_boundary_receipt.selected_pass_count",
    )
    fault_count = _plain_int(
        receipt["selected_fault_count"],
        label="global_reveal_boundary_receipt.selected_fault_count",
    )
    computed_pass = sum(
        all(bool(row["selected_pass"][offset]) for row in rows)
        for offset in range(len(selected))
    )
    decision = _text(
        receipt["decision"],
        label="global_reveal_boundary_receipt.decision",
    )
    d05_admissible = _exact_bool(
        receipt["d05_construction_admissible"],
        label=(
            "global_reveal_boundary_receipt."
            "d05_construction_admissible"
        ),
    )
    d05_owner_fault = _exact_bool(
        receipt["d05_owner_fault_present"],
        label="global_reveal_boundary_receipt.d05_owner_fault_present",
    )
    d05_faults = tuple(
        _plain_int(
            value,
            label=(
                "global_reveal_boundary_receipt."
                "d05_selected_primary_fault[]"
            ),
            maximum=(1 << 63) - 1,
        )
        for value in _sequence(
            receipt["d05_selected_primary_fault"],
            label=(
                "global_reveal_boundary_receipt."
                "d05_selected_primary_fault"
            ),
        )
    )
    if (
        not sequence
        or not preview_schema
        or not preview_root
        or packet_nbytes != 256 + 55 * num_envs
        or device_type not in ("cpu", "cuda")
        or (device_type == "cpu" and device_index is not None)
        or _plain_int(
            receipt["boundary_transfer_count"],
            label="global_reveal_boundary_receipt.boundary_transfer_count",
        )
        != 1
        or success_total > attempt_total
        or bytes_total < packet_nbytes
        or pass_count != computed_pass
        or fault_count != len(selected) - computed_pass
        or pass_count + fault_count != len(selected)
        or len(d05_faults) != len(selected)
        or d05_owner_fault != any(d05_faults)
        or decision != expected_decision
        or decision not in (
            FULL_MDP_REVEAL_DECISION_ACCEPT,
            FULL_MDP_REVEAL_DECISION_CENSOR,
        )
        or (
            decision == FULL_MDP_REVEAL_DECISION_ACCEPT
            and (
                fault_count != 0
                or d05_owner_fault
                or not d05_admissible
            )
        )
        or (
            decision == FULL_MDP_REVEAL_DECISION_CENSOR
            and fault_count == 0
            and not d05_owner_fault
        )
    ):
        raise PhysicalFlightContractError(
            "all-owner reveal boundary receipt conservation/decision differs"
        )
    return receipt, rows[
        FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER.index("physical_ball")
    ]


def _verified_r05_terminal_evidence(
    *,
    boundary_receipt: Mapping[str, object],
    terminal_boundary_projection: CanonicalJsonContentPin,
    terminal_content_pin: CanonicalJsonContentPin,
    expected_decision: str,
    expected_claim: "R05TerminalClaimProjection",
) -> None:
    """Verify the full portable projection/content behind an opaque R05 claim.

    These pins remain integrity evidence, not owner authority.  Production
    admission additionally requires the exact R05 owner-registry identity.
    """

    if type(expected_claim) is not R05TerminalClaimProjection:
        raise PhysicalFlightContractError(
            "R05 terminal claim projection type differs"
        )

    if (
        type(terminal_boundary_projection) is not CanonicalJsonContentPin
        or terminal_boundary_projection.source_kind
        != R05_TERMINAL_BOUNDARY_PROJECTION_KIND
        or terminal_boundary_projection.source_schema_version != 1
        or terminal_boundary_projection.source_schema_sha256
        != R05_TERMINAL_BOUNDARY_PROJECTION_SCHEMA_SHA256
    ):
        raise PhysicalFlightContractError(
            "R05 terminal boundary projection schema pin differs"
        )
    projection = _verified_external_sealed_mapping(
        terminal_boundary_projection.decoded_mapping,
        label="r05_terminal_boundary_projection",
        expected_kind=R05_TERMINAL_BOUNDARY_PROJECTION_KIND,
        expected_schema_version=1,
        expected_payload_fields=_R05_TERMINAL_BOUNDARY_PROJECTION_FIELDS,
    )
    if (
        projection["authority_domain"]
        != FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
        or _plain_int(
            projection["decision_mapping_schema_version"],
            label="r05_terminal_boundary_projection.decision_mapping_schema_version",
            minimum=1,
        )
        != 1
        or projection["decision"] != expected_decision
        or projection["source_decision"]
        not in (
            expected_decision,
            "PASS" if expected_decision == FULL_MDP_REVEAL_DECISION_ACCEPT else "CENSOR",
        )
        or _plain_int(
            projection["reveal_final_preview_schema_version"],
            label="r05_terminal_boundary_projection.preview_schema",
            minimum=1,
        )
        != expected_claim.reveal_final_preview_schema_version
        or _sha256(
            projection["reveal_final_preview_sha256"],
            label="r05_terminal_boundary_projection.preview_sha256",
        )
        != expected_claim.reveal_final_preview_sha256
        or _ordered_unique_env_ids(
            projection["selected_env_ids"],
            label="r05_terminal_boundary_projection.selected_env_ids",
        )
        != expected_claim.selected_env_ids
        or projection["boundary_receipt_kind"]
        != expected_claim.global_boundary_receipt_kind
        or _sha256(
            projection["boundary_receipt_sha256"],
            label="r05_terminal_boundary_projection.boundary_receipt_sha256",
        )
        != expected_claim.global_boundary_receipt_sha256
        or _plain_int(
            projection["boundary_packet_schema_version"],
            label="r05_terminal_boundary_projection.packet_schema",
            minimum=1,
        )
        != expected_claim.global_boundary_packet_schema_version
        or _sha256(
            projection["boundary_packet_sha256"],
            label="r05_terminal_boundary_projection.packet_sha256",
        )
        != expected_claim.global_boundary_packet_sha256
        or terminal_boundary_projection.source_canonical_sha256
        != expected_claim.terminal_boundary_projection_sha256
    ):
        raise PhysicalFlightContractError(
            "R05 terminal boundary projection/claim differs"
        )

    authority_schema = _sha256(
        projection["authority_schema_sha256"],
        label="r05_terminal_boundary_projection.authority_schema_sha256",
    )
    authority_source = _sha256(
        projection["authority_source_sha256"],
        label="r05_terminal_boundary_projection.authority_source_sha256",
    )
    authority_sha = canonical_sha256(
        {
            "schema_version": 1,
            "kind": R05_TERMINAL_BOUNDARY_AUTHORITY_KIND,
            "authority_domain": projection["authority_domain"],
            "authority_schema_sha256": authority_schema,
            "authority_source_sha256": authority_source,
        }
    )
    if authority_sha != expected_claim.terminal_boundary_authority_sha256:
        raise PhysicalFlightContractError(
            "R05 terminal boundary authority projection differs"
        )

    raw_participants = _sequence(
        projection["ordered_participant_roots"],
        label="r05_terminal_boundary_projection.ordered_participant_roots",
    )
    raw_boundary_rows = _sequence(
        boundary_receipt["ordered_owner_rows"],
        label="global_reveal_boundary_receipt.ordered_owner_rows",
    )
    if len(raw_participants) != len(FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER):
        raise PhysicalFlightContractError(
            "R05 terminal boundary participant width differs"
        )
    participant_rows: list[Mapping[str, object]] = []
    for index, (owner_kind, raw, boundary_row) in enumerate(
        zip(
            FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER,
            raw_participants,
            raw_boundary_rows,
        )
    ):
        participant = _verified_external_sealed_mapping(
            raw,
            label=f"r05_terminal_boundary_projection.participant[{index}]",
            expected_kind=R05_TERMINAL_BOUNDARY_PARTICIPANT_ROOT_KIND,
            expected_schema_version=1,
            expected_payload_fields=(
                _R05_TERMINAL_BOUNDARY_PARTICIPANT_ROOT_FIELDS
            ),
        )
        if (
            participant["participant_domain"]
            != FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
            or participant["participant_kind"] != owner_kind
            or _sha256(
                participant["participant_root_sha256"],
                label=f"r05 terminal participant[{index}] root",
            )
            != boundary_row["owner_token_root_sha256"]
        ):
            raise PhysicalFlightContractError(
                "R05 terminal participant root/order differs"
            )
        participant_rows.append(participant)

    raw_evidence = _sequence(
        projection["ordered_censor_evidence"],
        label="r05_terminal_boundary_projection.ordered_censor_evidence",
    )
    if expected_decision == FULL_MDP_REVEAL_DECISION_ACCEPT:
        if raw_evidence:
            raise PhysicalFlightContractError(
                "R05 ACCEPT projection carries CENSOR evidence"
            )
    elif len(raw_evidence) != len(expected_claim.selected_env_ids):
        raise PhysicalFlightContractError(
            "R05 CENSOR projection evidence width differs"
        )
    else:
        evidence_rows: list[Mapping[str, object]] = []
        for index, raw in enumerate(raw_evidence):
            evidence = _verified_external_sealed_mapping(
                raw,
                label=f"r05_terminal_boundary_projection.censor[{index}]",
                expected_kind=R05_TERMINAL_BOUNDARY_CENSOR_EVIDENCE_KIND,
                expected_schema_version=1,
                expected_payload_fields=(
                    _R05_TERMINAL_BOUNDARY_CENSOR_EVIDENCE_FIELDS
                ),
            )
            if (
                _plain_int(evidence["env_id"], label="censor evidence env")
                != expected_claim.selected_env_ids[index]
                or evidence["participant_domain"]
                != FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
                or evidence["participant_kind"]
                not in FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER
            ):
                raise PhysicalFlightContractError(
                    "R05 CENSOR evidence environment/participant differs"
                )
            participant_index = FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER.index(
                evidence["participant_kind"]
            )
            boundary_row = raw_boundary_rows[participant_index]
            for name in (
                "participant_root_sha256",
                "failure_receipt_sha256",
                "censor_fact_sha256",
                "producer_schema_sha256",
                "producer_source_sha256",
            ):
                _sha256(evidence[name], label=f"censor evidence {name}")
            if (
                evidence["participant_root_sha256"]
                != boundary_row["owner_token_root_sha256"]
                or evidence["failure_receipt_sha256"]
                != canonical_sha256(boundary_row)
            ):
                raise PhysicalFlightContractError(
                    "R05 CENSOR evidence boundary row differs"
                )
            evidence_rows.append(evidence)
        first = evidence_rows[0]
        if (
            first["primary_failure_env_id"]
            not in expected_claim.selected_env_ids
            or any(
                tuple(
                    row[name]
                    for name in (
                        "primary_failure_env_id",
                        "participant_domain",
                        "participant_kind",
                        "participant_root_sha256",
                        "failure_receipt_sha256",
                        "reason",
                        "producer_schema_sha256",
                        "producer_source_sha256",
                    )
                )
                != tuple(
                    first[name]
                    for name in (
                        "primary_failure_env_id",
                        "participant_domain",
                        "participant_kind",
                        "participant_root_sha256",
                        "failure_receipt_sha256",
                        "reason",
                        "producer_schema_sha256",
                        "producer_source_sha256",
                    )
                )
                for row in evidence_rows[1:]
            )
        ):
            raise PhysicalFlightContractError(
                "R05 CENSOR batch-primary evidence differs"
            )

    if (
        type(terminal_content_pin) is not CanonicalJsonContentPin
        or terminal_content_pin.source_kind
        != R05_PREPARED_TERMINAL_CONTENT_PIN_KIND
        or terminal_content_pin.source_schema_version != 1
        or terminal_content_pin.source_schema_sha256
        != R05_PREPARED_TERMINAL_CONTENT_PIN_SCHEMA_SHA256
    ):
        raise PhysicalFlightContractError(
            "R05 prepared terminal content schema pin differs"
        )
    content = _verified_external_sealed_mapping(
        terminal_content_pin.decoded_mapping,
        label="r05_terminal_content_pin",
        expected_kind=R05_PREPARED_TERMINAL_CONTENT_PIN_KIND,
        expected_schema_version=1,
        expected_payload_fields=_R05_PREPARED_TERMINAL_CONTENT_PIN_FIELDS,
    )
    try:
        raw_terminal = base64.b64decode(
            _text(
                content["content_bytes_base64"],
                label="r05_terminal_content_pin.content_bytes_base64",
            ).encode("ascii"),
            validate=True,
        )
        terminal_mapping = json.loads(raw_terminal.decode("ascii"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhysicalFlightContractError(
            "R05 prepared terminal content bytes are invalid"
        ) from exc
    expected_terminal_kind = (
        COMMITTED_REVEAL_BATCH_KIND
        if expected_decision == FULL_MDP_REVEAL_DECISION_ACCEPT
        else CENSORED_REVEAL_BATCH_KIND
    )
    marker_name = (
        "global_prearm_marker"
        if expected_decision == FULL_MDP_REVEAL_DECISION_ACCEPT
        else "terminal_boundary_marker"
    )
    marker = terminal_mapping.get(marker_name)
    if (
        _plain_int(
            content["terminal_schema_version"],
            label="r05_terminal_content_pin.terminal_schema_version",
            minimum=1,
        )
        != 2
        or content["terminal_kind"] != expected_terminal_kind
        or _sha256(
            content["terminal_canonical_sha256"],
            label="r05_terminal_content_pin.terminal_canonical_sha256",
        )
        != expected_claim.terminal_sha256
        or _plain_int(
            content["content_byte_length"],
            label="r05_terminal_content_pin.content_byte_length",
            minimum=1,
        )
        != len(raw_terminal)
        or _sha256(
            content["content_bytes_sha256"],
            label="r05_terminal_content_pin.content_bytes_sha256",
        )
        != hashlib.sha256(raw_terminal).hexdigest()
        or canonical_json_bytes(terminal_mapping) != raw_terminal
        or terminal_mapping.get("schema_version") != 2
        or terminal_mapping.get("kind") != expected_terminal_kind
        or terminal_mapping.get("canonical_sha256")
        != expected_claim.terminal_sha256
        or not isinstance(marker, Mapping)
        or marker.get("terminal_boundary_authority_sha256")
        != expected_claim.terminal_boundary_authority_sha256
        or marker.get("terminal_boundary_projection")
        != terminal_boundary_projection.decoded_mapping
        or terminal_content_pin.source_canonical_sha256
        != expected_claim.terminal_content_pin_sha256
    ):
        raise PhysicalFlightContractError(
            "R05 prepared terminal content/claim differs"
        )


def _preview_row_bindings(pin: CanonicalJsonContentPin) -> tuple[Mapping[str, object], ...]:
    if pin.source_kind == LEGACY_DIGEST_ONLY_REVEAL_FINAL_PREVIEW_KIND:
        raise DigestOnlyPayloadTombstonedError(
            "R05-v1 aliases full physical payload SHA to dynamic-state SHA"
        )
    if pin.source_kind != REVEAL_FINAL_PREVIEW_KIND:
        raise PhysicalFlightContractError("prepare source is not RevealFinalPreviewBatch")
    preview = pin.decoded_mapping
    if (
        pin.source_schema_version != 2
        or frozenset(preview)
        != _R05_PREVIEW_FIELDS
        | {"schema_version", "kind", "canonical_sha256"}
        or preview["integration_status"] != INTEGRATION_STATUS
        or preview["phase"] != "REVEAL_FINAL_PREVIEWED"
        or type(preview["public_visible"]) is not bool
        or preview["public_visible"]
        or type(preview["policy_opportunity_created"]) is not bool
        or preview["policy_opportunity_created"]
    ):
        raise PhysicalFlightContractError(
            "RevealFinalPreviewBatch-v2 strict schema/privacy differs"
        )
    rows = _sequence(preview.get("reveal_final_rows"), label="reveal_final_rows")
    if not rows:
        raise PhysicalFlightContractError("RevealFinalPreviewBatch rows differ")
    verified = tuple(
        _verified_external_sealed_mapping(
            row,
            label=f"reveal_final_rows[{index}]",
            expected_kind=R05_REVEAL_FINAL_INSTALL_ROW_KIND,
            expected_schema_version=2,
            expected_payload_fields=_R05_REVEAL_FINAL_ROW_FIELDS,
        )
        for index, row in enumerate(rows)
    )
    return verified


def _r05_slot_matches_physical_preimage(
    r05_slot: Mapping[str, object],
    physical: PhysicalFlightSlotSnapshot,
) -> bool:
    try:
        retired = _exact_bool(
            r05_slot["physical_retired"], label="r05_slot.physical_retired"
        )
        owner_sha = _optional_sha256(
            r05_slot["owner_key_sha256"], label="r05_slot.owner_key_sha256"
        )
        generation = _optional_plain_int(
            r05_slot["ball_generation"], label="r05_slot.ball_generation"
        )
        inbound_sha = _optional_sha256(
            r05_slot["inbound_ball_sha256"],
            label="r05_slot.inbound_ball_sha256",
        )
        dynamic_sha = _optional_sha256(
            r05_slot["dynamic_state_sha256"],
            label="r05_slot.dynamic_state_sha256",
        )
        lifecycle = _text(
            r05_slot["lifecycle_state"], label="r05_slot.lifecycle_state"
        )
    except (KeyError, PhysicalFlightContractError):
        return False
    if owner_sha is None:
        return (
            retired
            and lifecycle == "empty"
            and physical.lifecycle == SLOT_PARKED
            and physical.outcome_key_sha256 is None
            and generation is None
            and inbound_sha is None
            and dynamic_sha is None
        )
    allowed_r05_lifecycles = {
        SLOT_IN_FLIGHT: frozenset(
            ("inbound", "open", "settled_unpaid", "paid")
        ),
        SLOT_SETTLED_RETAINED: frozenset(("settled_unpaid", "paid")),
        SLOT_RETIRED: frozenset(("settled_unpaid", "paid", "closed")),
    }.get(physical.lifecycle, frozenset())
    expected_retired = physical.lifecycle == SLOT_RETIRED
    return (
        physical.outcome_key_sha256 == owner_sha
        and physical.ball_generation == generation
        and physical.inbound_ball_sha256 == inbound_sha
        and physical.current_state_f32_sha256 == dynamic_sha
        and lifecycle in allowed_r05_lifecycles
        and retired == expected_retired
        and physical.physically_parked == expected_retired
        and physical.published_to_runtime != expected_retired
        and inbound_sha is not None
    )


@dataclass(frozen=True)
class PhysicalInstallPrepareReceipt(_SealedRecord):
    """Private, nonmutating physical-owner preflight token."""

    KIND: ClassVar[str] = "action_ball_physical_install_prepare_receipt_v4"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 4

    integration_status: str
    capacity_receipt_sha256: str
    reveal_final_preview: CanonicalJsonContentPin
    num_envs: int
    reset_generations: Tuple[int, ...]
    physical_owner_checkpoint_before_sha256: str
    mutation_version_before: int
    prepare_nonce: int
    selected_env_ids: Tuple[int, ...]
    pre_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    rows: Tuple[PreparedPhysicalInstallRow, ...]
    pre_slots_root_sha256: str
    live_state_mutated: bool
    runtime_publication_created: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError("prepare integration status differs")
        if not isinstance(self.reveal_final_preview, CanonicalJsonContentPin):
            raise PhysicalFlightContractError("prepare lacks full preview content pin")
        for name in (
            "capacity_receipt_sha256",
            "physical_owner_checkpoint_before_sha256",
            "pre_slots_root_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        object.__setattr__(
            self,
            "mutation_version_before",
            _plain_int(self.mutation_version_before, label="mutation_version_before"),
        )
        object.__setattr__(
            self,
            "prepare_nonce",
            _plain_int(self.prepare_nonce, label="prepare_nonce", minimum=1),
        )
        selected = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        num_envs = _plain_int(self.num_envs, label="num_envs", minimum=1)
        reset_generations = tuple(
            _plain_int(item, label="reset_generations[]", minimum=1)
            for item in _sequence(
                self.reset_generations,
                label="reset_generations",
            )
        )
        if (
            len(reset_generations) != num_envs
            or any(env_id >= num_envs for env_id in selected)
        ):
            raise PhysicalFlightContractError(
                "prepare owner/reset-generation width differs"
            )
        rows = tuple(self.rows)
        if any(not isinstance(row, PreparedPhysicalInstallRow) for row in rows):
            raise PhysicalFlightContractError("prepare rows type differs")
        if tuple(row.env_id for row in rows) != selected:
            raise PhysicalFlightContractError("prepare selected row order differs")
        if len({(row.env_id, row.slot_index) for row in rows}) != len(rows):
            raise PhysicalFlightContractError("prepare rows duplicate env/slot")
        pre_slots = tuple(self.pre_slot_snapshots)
        if any(not isinstance(row, PhysicalFlightSlotSnapshot) for row in pre_slots):
            raise PhysicalFlightContractError("prepare pre-slot projection type differs")
        capacity = rows[0].install_payload.capacity_receipt.configured_flight_capacity
        expected_keys = tuple(
            (env_id, slot_index)
            for env_id in range(num_envs)
            for slot_index in range(capacity)
        )
        if tuple((row.env_id, row.slot_index) for row in pre_slots) != expected_keys:
            raise PhysicalFlightContractError(
                "prepare must bind the complete selected-env physical slot grid"
            )
        by_key = {(row.env_id, row.slot_index): row for row in pre_slots}
        if any(
            by_key[(row.env_id, row.slot_index)] != row.pre_slot_snapshot
            for row in rows
        ):
            raise PhysicalFlightContractError(
                "prepare selected rows differ from full pre-slot projection"
            )
        if (
            any(
                slot.capacity_receipt_sha256 != self.capacity_receipt_sha256
                or slot.capacity_value != capacity
                or slot.mutation_version > self.mutation_version_before
                for slot in pre_slots
            )
            or any(
                row.install_payload.capacity_receipt_sha256
                != self.capacity_receipt_sha256
                for row in rows
            )
            or len({slot.scene_body_name for slot in pre_slots}) != len(pre_slots)
            or any(
                slot.outcome_key is not None
                and slot.outcome_key.reset_generation
                != reset_generations[slot.env_id]
                for slot in pre_slots
            )
        ):
            raise PhysicalFlightContractError("prepare capacity/version differs")
        if self.pre_slots_root_sha256 != physical_slot_root(pre_slots):
            raise PhysicalFlightContractError("prepare pre-slot root differs")
        if self.physical_owner_checkpoint_before_sha256 != physical_owner_checkpoint_root(
            capacity_receipt_sha256=self.capacity_receipt_sha256,
            num_envs=num_envs,
            flight_capacity=capacity,
            mutation_version=self.mutation_version_before,
            next_prepare_nonce=self.prepare_nonce,
            reset_generations=reset_generations,
            slots=pre_slots,
            poisoned=False,
        ):
            raise PhysicalFlightContractError(
                "prepare complete owner checkpoint root differs"
            )
        preview_rows = _preview_row_bindings(self.reveal_final_preview)
        if len(preview_rows) != len(rows):
            raise PhysicalFlightContractError("preview/physical row width differs")
        for external, physical in zip(preview_rows, rows):
            try:
                facts = external["reveal_facts"]
                prepared = external["prepared_reveal"]
                preview_env = facts["env_id"]
                task_mapping = prepared["selected_task_ref"]
                outcome_mapping = prepared["outcome_key"]
            except (KeyError, TypeError) as exc:
                raise PhysicalFlightContractError(
                    "RevealFinalPreviewBatch row schema differs"
                ) from exc
            payload = physical.install_payload
            plan = _verified_external_sealed_mapping(
                external["ball_slot_plan"],
                label=f"env{physical.env_id}.ball_slot_plan",
                expected_kind=R05_BALL_SLOT_PLAN_KIND,
                expected_schema_version=2,
                expected_payload_fields=_R05_BALL_SLOT_PLAN_FIELDS,
            )
            preview_task = PhysicalFlightTaskRef._from_mapping_unpinned(
                task_mapping
            )
            preview_outcome = PhysicalFlightOutcomeKey._from_mapping_unpinned(
                outcome_mapping
            )
            r05_before = _verified_r05_ball_slots(
                external["pre_install_ball_slots"],
                label=f"env{physical.env_id}.pre_install_ball_slots",
                capacity=capacity,
            )
            r05_after = _verified_r05_ball_slots(
                external["post_install_ball_slots"],
                label=f"env{physical.env_id}.post_install_ball_slots",
                capacity=capacity,
            )
            env_physical_slots = tuple(
                by_key[(physical.env_id, slot_index)]
                for slot_index in range(capacity)
            )
            selected_slot = physical.slot_index
            installed_r05 = r05_after[selected_slot]
            preserved_live = tuple(
                row["owner_key_sha256"]
                for index, row in enumerate(r05_before)
                if index != selected_slot
                and row["owner_key_sha256"] is not None
                and row["physical_retired"] is False
            )
            if (
                preview_env != physical.env_id
                or plan["capacity"]
                != payload.capacity_receipt.configured_flight_capacity
                or plan["selected_slot_index"] != physical.slot_index
                or plan["new_ball_generation"] != payload.ball_generation
                or plan["new_inbound_ball_sha256"] != payload.inbound_ball_sha256
                or plan["new_ball_dynamic_state_sha256"]
                != payload.state_f32_sha256
                or plan["physical_ball_install_payload_sha256"]
                != payload.canonical_sha256
                or plan["snapshot_sha256"]
                != canonical_sha256(list(r05_before))
                or tuple(plan["preserved_live_owner_key_sha256"])
                != preserved_live
                or any(
                    after != before
                    for index, (before, after) in enumerate(
                        zip(r05_before, r05_after)
                    )
                    if index != selected_slot
                )
                or installed_r05["lifecycle_state"] != "inbound"
                or installed_r05["physical_retired"] is not False
                or installed_r05["owner_key_sha256"]
                != payload.outcome_key_sha256
                or installed_r05["ball_generation"] != payload.ball_generation
                or installed_r05["inbound_ball_sha256"]
                != payload.inbound_ball_sha256
                or installed_r05["dynamic_state_sha256"]
                != payload.state_f32_sha256
                or any(
                    not _r05_slot_matches_physical_preimage(r05_slot, slot)
                    for r05_slot, slot in zip(r05_before, env_physical_slots)
                )
                or external.get("physical_ball_install_payload_sha256")
                != payload.canonical_sha256
                or external.get("selected_task_ref_sha256") != payload.task_ref_sha256
                or external.get("outcome_key_sha256") != payload.outcome_key_sha256
                or preview_task != payload.task_ref
                or preview_outcome != payload.outcome_key
                or facts.get("reveal_step") != payload.reveal_control_step
                or facts.get("deadline_step")
                != payload.selected_contact_deadline_control_step
            ):
                raise PhysicalFlightContractError(
                    "preview and complete physical install payload differ"
                )
        if _exact_bool(self.live_state_mutated, label="live_state_mutated") or _exact_bool(
            self.runtime_publication_created,
            label="runtime_publication_created",
        ):
            raise PhysicalFlightContractError(
                "prepare cannot mutate or publish live state"
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "num_envs", num_envs)
        object.__setattr__(self, "reset_generations", reset_generations)
        object.__setattr__(self, "pre_slot_snapshots", pre_slots)
        object.__setattr__(self, "rows", rows)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["reveal_final_preview"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["reveal_final_preview"]
        )
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["reset_generations"] = tuple(values["reset_generations"])
        values["pre_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["pre_slot_snapshots"]
        )
        values["rows"] = tuple(
            PreparedPhysicalInstallRow._from_mapping_unpinned(row)
            for row in values["rows"]
        )
        return values


@dataclass(frozen=True)
class CommittedPhysicalInstallRow(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_committed_physical_install_row_v2"

    env_id: int
    slot_index: int
    install_payload_sha256: str
    committed_slot_snapshot: PhysicalFlightSlotSnapshot
    committed_slot_snapshot_sha256: str

    def __post_init__(self) -> None:
        env_id = _plain_int(self.env_id, label="env_id")
        slot = _plain_int(self.slot_index, label="slot_index")
        for name in ("install_payload_sha256", "committed_slot_snapshot_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        snapshot = self.committed_slot_snapshot
        if not isinstance(snapshot, PhysicalFlightSlotSnapshot):
            raise PhysicalFlightContractError("committed snapshot type differs")
        if (
            snapshot.canonical_sha256 != self.committed_slot_snapshot_sha256
            or (env_id, slot) != (snapshot.env_id, snapshot.slot_index)
            or snapshot.install_payload_sha256 != self.install_payload_sha256
            or snapshot.lifecycle != SLOT_IN_FLIGHT
        ):
            raise PhysicalFlightContractError("committed row binding differs")
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "slot_index", slot)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["committed_slot_snapshot"] = PhysicalFlightSlotSnapshot._from_mapping_unpinned(
            values["committed_slot_snapshot"]
        )
        return values


@dataclass(frozen=True)
class R05TerminalClaimProjection(_SealedRecord):
    """Portable canonical projection of one opaque owner-issued R05 claim.

    The projection is evidence, never authority.  Runtime admission must still
    use R05's retained claim object and owner registry; this record only makes
    every public claim scalar independently hash-checkable by downstream
    receipt readers.
    """

    KIND: ClassVar[str] = PREPARED_REVEAL_TERMINAL_CLAIM_KIND
    RECORD_SCHEMA_VERSION: ClassVar[int] = 1

    decision: str
    selected_env_ids: Tuple[int, ...]
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    global_boundary_receipt_kind: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_schema_version: int
    global_boundary_packet_sha256: str
    terminal_boundary_authority_sha256: str
    terminal_boundary_projection_sha256: str
    terminal_content_pin_sha256: str
    terminal_kind: str
    terminal_sha256: str

    def __post_init__(self) -> None:
        if self.decision not in (
            FULL_MDP_REVEAL_DECISION_ACCEPT,
            FULL_MDP_REVEAL_DECISION_CENSOR,
        ):
            raise PhysicalFlightContractError(
                "R05 terminal claim decision differs"
            )
        selected = _ordered_unique_env_ids(
            self.selected_env_ids,
            label="r05_terminal_claim.selected_env_ids",
        )
        preview_schema = _plain_int(
            self.reveal_final_preview_schema_version,
            label="r05_terminal_claim.reveal_final_preview_schema_version",
            minimum=1,
        )
        packet_schema = _plain_int(
            self.global_boundary_packet_schema_version,
            label="r05_terminal_claim.global_boundary_packet_schema_version",
            minimum=1,
        )
        for name in (
            "reveal_final_preview_sha256",
            "global_boundary_receipt_sha256",
            "global_boundary_packet_sha256",
            "terminal_boundary_authority_sha256",
            "terminal_boundary_projection_sha256",
            "terminal_content_pin_sha256",
            "terminal_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), label=f"r05_terminal_claim.{name}"),
            )
        expected_terminal_kind = (
            COMMITTED_REVEAL_BATCH_KIND
            if self.decision == FULL_MDP_REVEAL_DECISION_ACCEPT
            else CENSORED_REVEAL_BATCH_KIND
        )
        if (
            self.global_boundary_receipt_kind
            != FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND
            or self.terminal_kind != expected_terminal_kind
        ):
            raise PhysicalFlightContractError(
                "R05 terminal claim kind/decision differs"
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(
            self, "reveal_final_preview_schema_version", preview_schema
        )
        object.__setattr__(
            self, "global_boundary_packet_schema_version", packet_schema
        )

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        return values


@dataclass(frozen=True)
class PhysicalInstallCommitReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_physical_install_child_terminal_receipt_v5"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 5

    integration_status: str
    prepare_receipt: PhysicalInstallPrepareReceipt
    prepare_receipt_sha256: str
    global_reveal_boundary_receipt: CanonicalJsonContentPin
    global_reveal_boundary_receipt_sha256: str
    physical_boundary_fault_schema_sha256: str
    r05_terminal_claim: R05TerminalClaimProjection
    r05_terminal_claim_sha256: str
    r05_terminal_boundary_projection: CanonicalJsonContentPin
    r05_terminal_content_pin: CanonicalJsonContentPin
    r05_terminal_kind: str
    r05_terminal_sha256: str
    physical_owner_checkpoint_before_sha256: str
    physical_owner_checkpoint_after_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    rows: Tuple[CommittedPhysicalInstallRow, ...]
    committed_slots_root_sha256: str
    live_state_mutated: bool
    runtime_publication_created: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError("commit integration status differs")
        if not isinstance(self.prepare_receipt, PhysicalInstallPrepareReceipt):
            raise PhysicalFlightContractError("commit prepare receipt type differs")
        for name in (
            "prepare_receipt_sha256",
            "global_reveal_boundary_receipt_sha256",
            "physical_boundary_fault_schema_sha256",
            "r05_terminal_claim_sha256",
            "r05_terminal_sha256",
            "physical_owner_checkpoint_before_sha256",
            "physical_owner_checkpoint_after_sha256",
            "committed_slots_root_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        before = _plain_int(self.mutation_version_before, label="mutation_version_before")
        after = _plain_int(self.mutation_version_after, label="mutation_version_after")
        if (
            self.prepare_receipt_sha256 != self.prepare_receipt.canonical_sha256
            or self.physical_owner_checkpoint_before_sha256
            != self.prepare_receipt.physical_owner_checkpoint_before_sha256
            or before != self.prepare_receipt.mutation_version_before
            or after != before + 1
        ):
            raise PhysicalFlightContractError("commit prepare/version binding differs")
        boundary, physical_row = _verified_full_mdp_reveal_boundary_receipt(
            self.global_reveal_boundary_receipt,
            expected_decision=FULL_MDP_REVEAL_DECISION_ACCEPT,
        )
        _verified_r05_terminal_evidence(
            boundary_receipt=boundary,
            terminal_boundary_projection=(
                self.r05_terminal_boundary_projection
            ),
            terminal_content_pin=self.r05_terminal_content_pin,
            expected_decision=FULL_MDP_REVEAL_DECISION_ACCEPT,
            expected_claim=self.r05_terminal_claim,
        )
        if (
            self.global_reveal_boundary_receipt_sha256
            != self.global_reveal_boundary_receipt.source_canonical_sha256
            or tuple(boundary["selected_env_ids"])
            != self.prepare_receipt.selected_env_ids
            or boundary["reveal_final_preview_sha256"]
            != self.prepare_receipt.reveal_final_preview.source_canonical_sha256
            or boundary["reveal_final_preview_schema_version"]
            != self.prepare_receipt.reveal_final_preview.source_schema_version
            or physical_row["owner_token_root_sha256"]
            != self.prepare_receipt.canonical_sha256
            or physical_row["owner_mutation_version"] != before
            or physical_row["fault_schema_sha256"]
            != self.physical_boundary_fault_schema_sha256
            or tuple(physical_row["selected_pass"])
            != (True,) * len(self.prepare_receipt.selected_env_ids)
            or any(physical_row["selected_fault_bits"])
            or type(self.r05_terminal_claim) is not R05TerminalClaimProjection
            or self.r05_terminal_claim.canonical_sha256
            != self.r05_terminal_claim_sha256
            or self.r05_terminal_claim.decision
            != FULL_MDP_REVEAL_DECISION_ACCEPT
            or self.r05_terminal_claim.selected_env_ids
            != self.prepare_receipt.selected_env_ids
            or self.r05_terminal_claim.reveal_final_preview_schema_version
            != self.prepare_receipt.reveal_final_preview.source_schema_version
            or self.r05_terminal_claim.reveal_final_preview_sha256
            != self.prepare_receipt.reveal_final_preview.source_canonical_sha256
            or self.r05_terminal_claim.global_boundary_receipt_kind
            != self.global_reveal_boundary_receipt.source_kind
            or self.r05_terminal_claim.global_boundary_receipt_sha256
            != self.global_reveal_boundary_receipt.source_canonical_sha256
            or self.r05_terminal_claim.global_boundary_packet_schema_version
            != boundary["packet_schema_version"]
            or self.r05_terminal_claim.global_boundary_packet_sha256
            != boundary["packet_sha256"]
            or self.r05_terminal_claim.terminal_kind
            != self.r05_terminal_kind
            or self.r05_terminal_claim.terminal_sha256
            != self.r05_terminal_sha256
            or self.r05_terminal_kind != COMMITTED_REVEAL_BATCH_KIND
        ):
            raise PhysicalFlightContractError(
                "commit global boundary/terminal nested binding differs"
            )
        rows = tuple(self.rows)
        if any(not isinstance(row, CommittedPhysicalInstallRow) for row in rows):
            raise PhysicalFlightContractError("commit rows type differs")
        prepared = self.prepare_receipt.rows
        if len(rows) != len(prepared):
            raise PhysicalFlightContractError("commit row width differs")
        after_slots = list(self.prepare_receipt.pre_slot_snapshots)
        for row, prior in zip(rows, prepared):
            snapshot = row.committed_slot_snapshot
            if (
                (row.env_id, row.slot_index, row.install_payload_sha256)
                != (prior.env_id, prior.slot_index, prior.install_payload_sha256)
                or snapshot.outcome_key != prior.install_payload.outcome_key
                or snapshot.capacity_receipt_sha256
                != prior.install_payload.capacity_receipt_sha256
                or snapshot.capacity_value
                != prior.install_payload.capacity_receipt.configured_flight_capacity
                or snapshot.scene_body_name
                != prior.pre_slot_snapshot.scene_body_name
                or snapshot.ball_generation
                != prior.install_payload.ball_generation
                or snapshot.inbound_ball_sha256
                != prior.install_payload.inbound_ball_sha256
                or snapshot.reveal_control_step
                != prior.install_payload.reveal_control_step
                or snapshot.last_control_step
                != prior.install_payload.reveal_control_step
                or snapshot.last_physics_substep != 0
                or snapshot.last_sim_step != 0
                or snapshot.installed_ball_state_sha256
                != prior.install_payload.installed_ball_state_sha256
                or snapshot.current_state_f32 != prior.install_payload.state_f32
                or snapshot.mutation_version
                != prior.pre_slot_snapshot.mutation_version + 1
            ):
                raise PhysicalFlightContractError("commit after-image differs")
            after_slots[
                row.env_id * prior.install_payload.capacity_receipt.configured_flight_capacity
                + row.slot_index
            ] = snapshot
        if self.committed_slots_root_sha256 != physical_slot_root(
            tuple(row.committed_slot_snapshot for row in rows)
        ):
            raise PhysicalFlightContractError("commit slot root differs")
        if self.physical_owner_checkpoint_after_sha256 != physical_owner_checkpoint_root(
            capacity_receipt_sha256=self.prepare_receipt.capacity_receipt_sha256,
            num_envs=self.prepare_receipt.num_envs,
            flight_capacity=(
                prepared[0].install_payload.capacity_receipt.configured_flight_capacity
            ),
            mutation_version=after,
            next_prepare_nonce=self.prepare_receipt.prepare_nonce + 1,
            reset_generations=self.prepare_receipt.reset_generations,
            slots=tuple(after_slots),
            poisoned=False,
        ):
            raise PhysicalFlightContractError(
                "commit complete owner checkpoint root differs"
            )
        if not _exact_bool(self.live_state_mutated, label="live_state_mutated") or not _exact_bool(
            self.runtime_publication_created,
            label="runtime_publication_created",
        ):
            raise PhysicalFlightContractError("commit must mutate and publish atomically")
        object.__setattr__(self, "mutation_version_before", before)
        object.__setattr__(self, "mutation_version_after", after)
        object.__setattr__(self, "rows", rows)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["prepare_receipt"] = PhysicalInstallPrepareReceipt._from_mapping_unpinned(
            values["prepare_receipt"]
        )
        values["global_reveal_boundary_receipt"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["global_reveal_boundary_receipt"]
        )
        values["r05_terminal_claim"] = R05TerminalClaimProjection._from_mapping_unpinned(
            values["r05_terminal_claim"]
        )
        values["r05_terminal_boundary_projection"] = (
            CanonicalJsonContentPin._from_mapping_unpinned(
                values["r05_terminal_boundary_projection"]
            )
        )
        values["r05_terminal_content_pin"] = (
            CanonicalJsonContentPin._from_mapping_unpinned(
                values["r05_terminal_content_pin"]
            )
        )
        values["rows"] = tuple(
            CommittedPhysicalInstallRow._from_mapping_unpinned(row)
            for row in values["rows"]
        )
        return values


@dataclass(frozen=True)
class PhysicalInstallCensorReceipt(_SealedRecord):
    """Typed zero-install chronology publication for a global CENSOR receipt."""

    KIND: ClassVar[str] = "action_ball_physical_install_censor_child_terminal_receipt_v1"

    integration_status: str
    prepare_receipt: PhysicalInstallPrepareReceipt
    prepare_receipt_sha256: str
    global_reveal_boundary_receipt: CanonicalJsonContentPin
    global_reveal_boundary_receipt_sha256: str
    physical_boundary_fault_schema_sha256: str
    r05_terminal_claim: R05TerminalClaimProjection
    r05_terminal_claim_sha256: str
    r05_terminal_boundary_projection: CanonicalJsonContentPin
    r05_terminal_content_pin: CanonicalJsonContentPin
    r05_terminal_kind: str
    r05_terminal_sha256: str
    physical_owner_checkpoint_before_sha256: str
    physical_owner_checkpoint_after_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    slots_root_before_sha256: str
    slots_root_after_sha256: str
    scene_state_mutated: bool
    slot_state_mutated: bool
    owner_chronology_mutated: bool
    runtime_publication_created: bool
    policy_opportunity_created: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError("censor integration status differs")
        prepare = self.prepare_receipt
        if type(prepare) is not PhysicalInstallPrepareReceipt:
            raise PhysicalFlightContractError("censor prepare receipt type differs")
        for name in (
            "prepare_receipt_sha256",
            "global_reveal_boundary_receipt_sha256",
            "physical_boundary_fault_schema_sha256",
            "r05_terminal_claim_sha256",
            "r05_terminal_sha256",
            "physical_owner_checkpoint_before_sha256",
            "physical_owner_checkpoint_after_sha256",
            "slots_root_before_sha256",
            "slots_root_after_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        before = _plain_int(self.mutation_version_before, label="mutation_version_before")
        after = _plain_int(self.mutation_version_after, label="mutation_version_after")
        boundary, physical_row = _verified_full_mdp_reveal_boundary_receipt(
            self.global_reveal_boundary_receipt,
            expected_decision=FULL_MDP_REVEAL_DECISION_CENSOR,
        )
        _verified_r05_terminal_evidence(
            boundary_receipt=boundary,
            terminal_boundary_projection=(
                self.r05_terminal_boundary_projection
            ),
            terminal_content_pin=self.r05_terminal_content_pin,
            expected_decision=FULL_MDP_REVEAL_DECISION_CENSOR,
            expected_claim=self.r05_terminal_claim,
        )
        if (
            self.prepare_receipt_sha256 != prepare.canonical_sha256
            or self.global_reveal_boundary_receipt_sha256
            != self.global_reveal_boundary_receipt.source_canonical_sha256
            or tuple(boundary["selected_env_ids"]) != prepare.selected_env_ids
            or boundary["reveal_final_preview_schema_version"]
            != prepare.reveal_final_preview.source_schema_version
            or boundary["reveal_final_preview_sha256"]
            != prepare.reveal_final_preview.source_canonical_sha256
            or physical_row["owner_token_root_sha256"]
            != prepare.canonical_sha256
            or physical_row["owner_mutation_version"] != before
            or physical_row["fault_schema_sha256"]
            != self.physical_boundary_fault_schema_sha256
            or tuple(physical_row["selected_pass"])
            != (True,) * len(prepare.selected_env_ids)
            or any(physical_row["selected_fault_bits"])
            or type(self.r05_terminal_claim) is not R05TerminalClaimProjection
            or self.r05_terminal_claim.canonical_sha256
            != self.r05_terminal_claim_sha256
            or self.r05_terminal_claim.decision
            != FULL_MDP_REVEAL_DECISION_CENSOR
            or self.r05_terminal_claim.selected_env_ids
            != prepare.selected_env_ids
            or self.r05_terminal_claim.reveal_final_preview_schema_version
            != prepare.reveal_final_preview.source_schema_version
            or self.r05_terminal_claim.reveal_final_preview_sha256
            != prepare.reveal_final_preview.source_canonical_sha256
            or self.r05_terminal_claim.global_boundary_receipt_kind
            != self.global_reveal_boundary_receipt.source_kind
            or self.r05_terminal_claim.global_boundary_receipt_sha256
            != self.global_reveal_boundary_receipt.source_canonical_sha256
            or self.r05_terminal_claim.global_boundary_packet_schema_version
            != boundary["packet_schema_version"]
            or self.r05_terminal_claim.global_boundary_packet_sha256
            != boundary["packet_sha256"]
            or self.r05_terminal_claim.terminal_kind
            != self.r05_terminal_kind
            or self.r05_terminal_claim.terminal_sha256
            != self.r05_terminal_sha256
            or self.r05_terminal_kind
            != CENSORED_REVEAL_BATCH_KIND
            or before != prepare.mutation_version_before
            or after != before + 1
            or self.physical_owner_checkpoint_before_sha256
            != prepare.physical_owner_checkpoint_before_sha256
            or self.slots_root_before_sha256 != prepare.pre_slots_root_sha256
            or self.slots_root_after_sha256 != self.slots_root_before_sha256
            or self.physical_owner_checkpoint_after_sha256
            != physical_owner_checkpoint_root(
                capacity_receipt_sha256=prepare.capacity_receipt_sha256,
                num_envs=prepare.num_envs,
                flight_capacity=(
                    prepare.rows[0]
                    .install_payload.capacity_receipt.configured_flight_capacity
                ),
                mutation_version=after,
                next_prepare_nonce=prepare.prepare_nonce + 1,
                reset_generations=prepare.reset_generations,
                slots=prepare.pre_slot_snapshots,
                poisoned=False,
            )
        ):
            raise PhysicalFlightContractError(
                "censor global boundary/chronology binding differs"
            )
        if (
            _exact_bool(self.scene_state_mutated, label="scene_state_mutated")
            or _exact_bool(self.slot_state_mutated, label="slot_state_mutated")
            or not _exact_bool(
                self.owner_chronology_mutated,
                label="owner_chronology_mutated",
            )
            or _exact_bool(
                self.runtime_publication_created,
                label="runtime_publication_created",
            )
            or _exact_bool(
                self.policy_opportunity_created,
                label="policy_opportunity_created",
            )
        ):
            raise PhysicalFlightContractError(
                "censor must publish chronology with zero install/opportunity"
            )
        object.__setattr__(self, "mutation_version_before", before)
        object.__setattr__(self, "mutation_version_after", after)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["prepare_receipt"] = PhysicalInstallPrepareReceipt._from_mapping_unpinned(
            values["prepare_receipt"]
        )
        values["global_reveal_boundary_receipt"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["global_reveal_boundary_receipt"]
        )
        values["r05_terminal_claim"] = R05TerminalClaimProjection._from_mapping_unpinned(
            values["r05_terminal_claim"]
        )
        values["r05_terminal_boundary_projection"] = (
            CanonicalJsonContentPin._from_mapping_unpinned(
                values["r05_terminal_boundary_projection"]
            )
        )
        values["r05_terminal_content_pin"] = (
            CanonicalJsonContentPin._from_mapping_unpinned(
                values["r05_terminal_content_pin"]
            )
        )
        return values


@dataclass(frozen=True)
class PhysicalInstallAbortReceipt(_SealedRecord):
    """Abort consumes a private prepare and proves zero live mutation."""

    KIND: ClassVar[str] = "action_ball_physical_install_abort_receipt_v4"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 4

    integration_status: str
    prepare_receipt: PhysicalInstallPrepareReceipt
    prepare_receipt_sha256: str
    physical_owner_checkpoint_before_sha256: str
    physical_owner_checkpoint_after_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    live_state_mutated: bool
    runtime_publication_created: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS or not isinstance(
            self.prepare_receipt, PhysicalInstallPrepareReceipt
        ):
            raise PhysicalFlightContractError("abort authority/status differs")
        for name in (
            "prepare_receipt_sha256",
            "physical_owner_checkpoint_before_sha256",
            "physical_owner_checkpoint_after_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        before = _plain_int(self.mutation_version_before, label="mutation_version_before")
        after = _plain_int(self.mutation_version_after, label="mutation_version_after")
        if (
            self.prepare_receipt_sha256 != self.prepare_receipt.canonical_sha256
            or self.physical_owner_checkpoint_before_sha256
            != self.prepare_receipt.physical_owner_checkpoint_before_sha256
            or self.physical_owner_checkpoint_after_sha256
            != self.physical_owner_checkpoint_before_sha256
            or before != self.prepare_receipt.mutation_version_before
            or after != before
            or _exact_bool(self.live_state_mutated, label="live_state_mutated")
            or _exact_bool(
                self.runtime_publication_created,
                label="runtime_publication_created",
            )
        ):
            raise PhysicalFlightContractError("abort did not preserve private live state")
        object.__setattr__(self, "mutation_version_before", before)
        object.__setattr__(self, "mutation_version_after", after)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["prepare_receipt"] = PhysicalInstallPrepareReceipt._from_mapping_unpinned(
            values["prepare_receipt"]
        )
        return values


@dataclass(frozen=True)
class PhysicalRetireRow(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_physical_retire_row_v2"

    env_id: int
    slot_index: int
    outcome_key: PhysicalFlightOutcomeKey
    outcome_key_sha256: str
    settlement_authority: CanonicalJsonContentPin
    pre_slot_snapshot: PhysicalFlightSlotSnapshot
    post_slot_snapshot: PhysicalFlightSlotSnapshot

    def __post_init__(self) -> None:
        env_id = _plain_int(self.env_id, label="env_id")
        slot = _plain_int(self.slot_index, label="slot_index")
        if not isinstance(self.outcome_key, PhysicalFlightOutcomeKey) or not isinstance(
            self.settlement_authority, CanonicalJsonContentPin
        ):
            raise PhysicalFlightContractError("retire key/settlement authority differs")
        authority = _verified_external_sealed_mapping(
            self.settlement_authority.decoded_mapping,
            label="settlement_authority",
            expected_kind=PHYSICAL_SETTLEMENT_AUTHORITY_KIND,
            expected_schema_version=2,
            expected_payload_fields=_PHYSICAL_SETTLEMENT_AUTHORITY_FIELDS,
        )
        if (
            self.settlement_authority.source_kind
            != PHYSICAL_SETTLEMENT_AUTHORITY_KIND
            or self.settlement_authority.source_schema_version != 2
            or self.settlement_authority.source_schema_sha256
            != PHYSICAL_SETTLEMENT_AUTHORITY_SCHEMA_SHA256
            or authority["mailbox_lifecycle"] != "SETTLED_UNPAID"
            or type(authority["r06_owner_mutation_version"]) is not int
            or authority["r06_owner_mutation_version"] < 1
        ):
            raise PhysicalFlightContractError(
                "retire settlement authority lifecycle/version differs"
            )
        _sha256(
            authority["r06_after_root_sha256"],
            label="settlement_authority.r06_after_root_sha256",
        )
        authority_rows = _sequence(
            authority["physical_retire_rows"],
            label="settlement_authority.physical_retire_rows",
        )
        normalized_authority_rows = []
        for index, authority_row in enumerate(authority_rows):
            if (
                not isinstance(authority_row, Mapping)
                or frozenset(authority_row)
                != {
                    "env_id",
                    "slot_index",
                    "outcome_key_sha256",
                    "ball_generation",
                }
            ):
                raise PhysicalFlightContractError(
                    "retire settlement authority row schema differs"
                )
            normalized_authority_rows.append(
                (
                    _plain_int(
                        authority_row["env_id"],
                        label=f"settlement_authority.rows[{index}].env_id",
                    ),
                    _plain_int(
                        authority_row["slot_index"],
                        label=f"settlement_authority.rows[{index}].slot_index",
                    ),
                    _sha256(
                        authority_row["outcome_key_sha256"],
                        label=(
                            f"settlement_authority.rows[{index}]"
                            ".outcome_key_sha256"
                        ),
                    ),
                    _plain_int(
                        authority_row["ball_generation"],
                        label=(
                            f"settlement_authority.rows[{index}]"
                            ".ball_generation"
                        ),
                    ),
                )
            )
        if tuple(normalized_authority_rows) != tuple(
            sorted(set(normalized_authority_rows))
        ):
            raise PhysicalFlightContractError(
                "retire settlement authority rows are not sorted/unique"
            )
        if not isinstance(self.pre_slot_snapshot, PhysicalFlightSlotSnapshot) or not isinstance(
            self.post_slot_snapshot, PhysicalFlightSlotSnapshot
        ):
            raise PhysicalFlightContractError("retire slot snapshot type differs")
        object.__setattr__(
            self,
            "outcome_key_sha256",
            _sha256(self.outcome_key_sha256, label="outcome_key_sha256"),
        )
        pre = self.pre_slot_snapshot
        post = self.post_slot_snapshot
        if (
            self.outcome_key_sha256 != self.outcome_key.canonical_sha256
            or (
                env_id,
                slot,
                self.outcome_key_sha256,
                self.outcome_key.swing_generation,
            )
            not in normalized_authority_rows
            or (env_id, slot) != (pre.env_id, pre.slot_index)
            or (env_id, slot) != (post.env_id, post.slot_index)
            or pre.outcome_key != self.outcome_key
            or post.outcome_key != self.outcome_key
            or pre.lifecycle not in (SLOT_IN_FLIGHT, SLOT_SETTLED_RETAINED)
            or post.lifecycle != SLOT_RETIRED
            or pre.capacity_receipt_sha256 != post.capacity_receipt_sha256
            or pre.capacity_value != post.capacity_value
            or pre.scene_body_name != post.scene_body_name
            or pre.ball_generation != post.ball_generation
            or pre.inbound_ball_sha256 != post.inbound_ball_sha256
            or pre.outcome_key_sha256 != post.outcome_key_sha256
            or pre.install_payload_sha256 != post.install_payload_sha256
            or pre.installed_ball_state_sha256
            != post.installed_ball_state_sha256
            or pre.current_state_f32 != post.current_state_f32
            or pre.current_state_f32_sha256
            != post.current_state_f32_sha256
            or pre.reveal_control_step != post.reveal_control_step
            or pre.last_control_step != post.last_control_step
            or pre.last_physics_substep != post.last_physics_substep
            or pre.last_sim_step != post.last_sim_step
            or post.mutation_version != pre.mutation_version + 1
        ):
            raise PhysicalFlightContractError("retire row identity/after-image differs")
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "slot_index", slot)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["outcome_key"] = PhysicalFlightOutcomeKey._from_mapping_unpinned(
            values["outcome_key"]
        )
        values["settlement_authority"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["settlement_authority"]
        )
        values["pre_slot_snapshot"] = PhysicalFlightSlotSnapshot._from_mapping_unpinned(
            values["pre_slot_snapshot"]
        )
        values["post_slot_snapshot"] = PhysicalFlightSlotSnapshot._from_mapping_unpinned(
            values["post_slot_snapshot"]
        )
        return values


@dataclass(frozen=True)
class PhysicalRetireReceipt(_SealedRecord):
    """Release only physical flight; mailbox/outcome ownership remains R06."""

    KIND: ClassVar[str] = "action_ball_physical_retire_receipt_v3"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 3

    integration_status: str
    physical_owner_checkpoint_before_sha256: str
    physical_owner_checkpoint_after_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    num_envs: int
    flight_capacity: int
    reset_generations: Tuple[int, ...]
    next_prepare_nonce: int
    pre_owner_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    post_owner_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    rows: Tuple[PhysicalRetireRow, ...]
    pre_slots_root_sha256: str
    post_slots_root_sha256: str
    physical_flight_released: bool
    mailbox_lifecycle_mutated: bool
    scene_bodies_parked: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError("retire integration status differs")
        for name in (
            "physical_owner_checkpoint_before_sha256",
            "physical_owner_checkpoint_after_sha256",
            "pre_slots_root_sha256",
            "post_slots_root_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        before = _plain_int(self.mutation_version_before, label="mutation_version_before")
        after = _plain_int(self.mutation_version_after, label="mutation_version_after")
        num_envs = _plain_int(self.num_envs, label="num_envs", minimum=1)
        capacity = _plain_int(
            self.flight_capacity,
            label="flight_capacity",
            minimum=1,
        )
        reset_generations = tuple(
            _plain_int(item, label="reset_generations[]", minimum=1)
            for item in _sequence(
                self.reset_generations,
                label="reset_generations",
            )
        )
        nonce = _plain_int(
            self.next_prepare_nonce,
            label="next_prepare_nonce",
            minimum=1,
        )
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, PhysicalRetireRow) for row in rows):
            raise PhysicalFlightContractError("retire rows type differs")
        if tuple((row.env_id, row.slot_index) for row in rows) != tuple(
            sorted(set((row.env_id, row.slot_index) for row in rows))
        ):
            raise PhysicalFlightContractError("retire rows must be sorted/unique")
        authority_roots = {
            row.settlement_authority.canonical_sha256 for row in rows
        }
        if len(authority_roots) != 1:
            raise PhysicalFlightContractError(
                "retire rows must consume one exact settlement authority"
            )
        authority_rows = rows[0].settlement_authority.decoded_mapping.get(
            "physical_retire_rows"
        )
        expected_authority_rows = [
            {
                "env_id": row.env_id,
                "slot_index": row.slot_index,
                "outcome_key_sha256": row.outcome_key_sha256,
                "ball_generation": row.outcome_key.swing_generation,
            }
            for row in rows
        ]
        if authority_rows != expected_authority_rows:
            raise PhysicalFlightContractError(
                "retire receipt rows differ from exact settlement authority rows"
            )
        if after != before + 1:
            raise PhysicalFlightContractError("retire mutation version differs")
        pre_owner = tuple(self.pre_owner_slot_snapshots)
        post_owner = tuple(self.post_owner_slot_snapshots)
        expected_keys = tuple(
            (env_id, slot_index)
            for env_id in range(num_envs)
            for slot_index in range(capacity)
        )
        if (
            len(reset_generations) != num_envs
            or tuple((slot.env_id, slot.slot_index) for slot in pre_owner)
            != expected_keys
            or tuple((slot.env_id, slot.slot_index) for slot in post_owner)
            != expected_keys
            or any(slot.mutation_version > before for slot in pre_owner)
            or any(slot.mutation_version > after for slot in post_owner)
        ):
            raise PhysicalFlightContractError(
                "retire complete owner grid/version differs"
            )
        row_keys = {(row.env_id, row.slot_index): row for row in rows}
        for pre_slot, post_slot in zip(pre_owner, post_owner):
            row = row_keys.get((pre_slot.env_id, pre_slot.slot_index))
            if row is None:
                if post_slot != pre_slot:
                    raise PhysicalFlightContractError(
                        "retire mutated a slot outside its exact rows"
                    )
            elif (
                row.pre_slot_snapshot != pre_slot
                or row.post_slot_snapshot != post_slot
            ):
                raise PhysicalFlightContractError(
                    "retire rows differ from complete owner grid"
                )
        capacity_sha = pre_owner[0].capacity_receipt_sha256
        if (
            any(
                slot.capacity_receipt_sha256 != capacity_sha
                or slot.capacity_value != capacity
                or (
                    slot.outcome_key is not None
                    and slot.outcome_key.reset_generation
                    != reset_generations[slot.env_id]
                )
                for slot in (*pre_owner, *post_owner)
            )
            or self.physical_owner_checkpoint_before_sha256
            != physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity_sha,
                num_envs=num_envs,
                flight_capacity=capacity,
                mutation_version=before,
                next_prepare_nonce=nonce,
                reset_generations=reset_generations,
                slots=pre_owner,
                poisoned=False,
            )
            or self.physical_owner_checkpoint_after_sha256
            != physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity_sha,
                num_envs=num_envs,
                flight_capacity=capacity,
                mutation_version=after,
                next_prepare_nonce=nonce,
                reset_generations=reset_generations,
                slots=post_owner,
                poisoned=False,
            )
        ):
            raise PhysicalFlightContractError(
                "retire complete owner checkpoint root differs"
            )
        if self.pre_slots_root_sha256 != physical_slot_root(
            tuple(row.pre_slot_snapshot for row in rows)
        ) or self.post_slots_root_sha256 != physical_slot_root(
            tuple(row.post_slot_snapshot for row in rows)
        ):
            raise PhysicalFlightContractError("retire slot roots differ")
        if not _exact_bool(
            self.physical_flight_released, label="physical_flight_released"
        ) or _exact_bool(
            self.mailbox_lifecycle_mutated, label="mailbox_lifecycle_mutated"
        ) or not _exact_bool(self.scene_bodies_parked, label="scene_bodies_parked"):
            raise PhysicalFlightContractError(
                "retire crossed the physical/mailbox ownership boundary"
            )
        object.__setattr__(self, "mutation_version_before", before)
        object.__setattr__(self, "mutation_version_after", after)
        object.__setattr__(self, "num_envs", num_envs)
        object.__setattr__(self, "flight_capacity", capacity)
        object.__setattr__(self, "reset_generations", reset_generations)
        object.__setattr__(self, "next_prepare_nonce", nonce)
        object.__setattr__(self, "pre_owner_slot_snapshots", pre_owner)
        object.__setattr__(self, "post_owner_slot_snapshots", post_owner)
        object.__setattr__(self, "rows", rows)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["reset_generations"] = tuple(values["reset_generations"])
        values["pre_owner_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["pre_owner_slot_snapshots"]
        )
        values["post_owner_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["post_owner_slot_snapshots"]
        )
        values["rows"] = tuple(
            PhysicalRetireRow._from_mapping_unpinned(row)
            for row in values["rows"]
        )
        return values


@dataclass(frozen=True)
class PhysicalTrueResetRow(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_physical_true_reset_row_v2"

    env_id: int
    prior_reset_generation: int
    next_reset_generation: int
    pre_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    post_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]

    def __post_init__(self) -> None:
        env_id = _plain_int(self.env_id, label="env_id")
        prior = _plain_int(
            self.prior_reset_generation,
            label="prior_reset_generation",
            minimum=1,
        )
        next_generation = _plain_int(
            self.next_reset_generation,
            label="next_reset_generation",
            minimum=2,
        )
        before = tuple(self.pre_slot_snapshots)
        after = tuple(self.post_slot_snapshots)
        if next_generation != prior + 1 or not before or len(before) != len(after):
            raise PhysicalFlightContractError("true-reset generation/slot width differs")
        expected_slots = tuple(range(len(before)))
        if (
            tuple(row.slot_index for row in before) != expected_slots
            or tuple(row.slot_index for row in after) != expected_slots
            or any(row.env_id != env_id for row in (*before, *after))
            or any(
                pre.capacity_receipt_sha256 != post.capacity_receipt_sha256
                or pre.capacity_value != post.capacity_value
                or pre.scene_body_name != post.scene_body_name
                or post.last_control_step != 0
                or post.last_physics_substep != 0
                or post.last_sim_step != 0
                for pre, post in zip(before, after)
            )
            or any(
                row.outcome_key is not None
                and row.outcome_key.reset_generation != prior
                for row in before
            )
            or any(row.lifecycle != SLOT_PARKED for row in after)
            or any(
                post.mutation_version != pre.mutation_version + 1
                for pre, post in zip(before, after)
            )
        ):
            raise PhysicalFlightContractError("true-reset slot projection differs")
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "prior_reset_generation", prior)
        object.__setattr__(self, "next_reset_generation", next_generation)
        object.__setattr__(self, "pre_slot_snapshots", before)
        object.__setattr__(self, "post_slot_snapshots", after)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["pre_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["pre_slot_snapshots"]
        )
        values["post_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["post_slot_snapshots"]
        )
        return values


@dataclass(frozen=True)
class PhysicalTrueResetReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_physical_true_reset_receipt_v3"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 3

    integration_status: str
    zero_open_all_owner_closure: CanonicalJsonContentPin
    selected_env_ids: Tuple[int, ...]
    rows: Tuple[PhysicalTrueResetRow, ...]
    physical_owner_checkpoint_before_sha256: str
    physical_owner_checkpoint_after_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    num_envs: int
    flight_capacity: int
    reset_generations_before: Tuple[int, ...]
    reset_generations_after: Tuple[int, ...]
    next_prepare_nonce: int
    pre_owner_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    post_owner_slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    selected_slots_root_before_sha256: str
    selected_slots_root_after_sha256: str
    unselected_slots_root_before_sha256: str
    unselected_slots_root_after_sha256: str
    env_reset_invoked: bool
    mailbox_lifecycle_mutated: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS or not isinstance(
            self.zero_open_all_owner_closure, CanonicalJsonContentPin
        ):
            raise PhysicalFlightContractError("true-reset closure/status differs")
        closure = _verified_external_sealed_mapping(
            self.zero_open_all_owner_closure.decoded_mapping,
            label="zero_open_all_owner_closure",
            expected_kind=PHYSICAL_ZERO_OPEN_RESET_CLOSURE_KIND,
            expected_schema_version=2,
            expected_payload_fields=_PHYSICAL_ZERO_OPEN_RESET_CLOSURE_FIELDS,
        )
        for name in (
            "physical_owner_checkpoint_before_sha256",
            "physical_owner_checkpoint_after_sha256",
            "selected_slots_root_before_sha256",
            "selected_slots_root_after_sha256",
            "unselected_slots_root_before_sha256",
            "unselected_slots_root_after_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        selected = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        rows = tuple(self.rows)
        if any(not isinstance(row, PhysicalTrueResetRow) for row in rows) or tuple(
            row.env_id for row in rows
        ) != selected:
            raise PhysicalFlightContractError("true-reset selected rows differ")
        before = _plain_int(self.mutation_version_before, label="mutation_version_before")
        after = _plain_int(self.mutation_version_after, label="mutation_version_after")
        num_envs = _plain_int(self.num_envs, label="num_envs", minimum=1)
        capacity = _plain_int(
            self.flight_capacity,
            label="flight_capacity",
            minimum=1,
        )
        reset_before = tuple(
            _plain_int(item, label="reset_generations_before[]", minimum=1)
            for item in _sequence(
                self.reset_generations_before,
                label="reset_generations_before",
            )
        )
        reset_after = tuple(
            _plain_int(item, label="reset_generations_after[]", minimum=1)
            for item in _sequence(
                self.reset_generations_after,
                label="reset_generations_after",
            )
        )
        nonce = _plain_int(
            self.next_prepare_nonce,
            label="next_prepare_nonce",
            minimum=1,
        )
        before_slots = tuple(slot for row in rows for slot in row.pre_slot_snapshots)
        after_slots = tuple(slot for row in rows for slot in row.post_slot_snapshots)
        pre_owner = tuple(self.pre_owner_slot_snapshots)
        post_owner = tuple(self.post_owner_slot_snapshots)
        expected_keys = tuple(
            (env_id, slot_index)
            for env_id in range(num_envs)
            for slot_index in range(capacity)
        )
        selected_set = set(selected)
        pre_selected = tuple(
            slot for slot in pre_owner if slot.env_id in selected_set
        )
        post_selected = tuple(
            slot for slot in post_owner if slot.env_id in selected_set
        )
        pre_unselected = tuple(
            slot for slot in pre_owner if slot.env_id not in selected_set
        )
        post_unselected = tuple(
            slot for slot in post_owner if slot.env_id not in selected_set
        )
        capacity_sha = pre_owner[0].capacity_receipt_sha256
        if (
            self.zero_open_all_owner_closure.source_kind
            != PHYSICAL_ZERO_OPEN_RESET_CLOSURE_KIND
            or self.zero_open_all_owner_closure.source_schema_version != 2
            or closure["selected_env_ids"] != list(selected)
            or closure["open_flight_count"] != 0
            or closure["open_mailbox_count"] != 0
            or after != before + 1
            or len(reset_before) != num_envs
            or len(reset_after) != num_envs
            or any(
                reset_after[env_id]
                != reset_before[env_id] + (1 if env_id in selected_set else 0)
                for env_id in range(num_envs)
            )
            or any(
                reset_before[row.env_id] != row.prior_reset_generation
                or reset_after[row.env_id] != row.next_reset_generation
                for row in rows
            )
            or tuple((slot.env_id, slot.slot_index) for slot in pre_owner)
            != expected_keys
            or tuple((slot.env_id, slot.slot_index) for slot in post_owner)
            != expected_keys
            or pre_selected != before_slots
            or post_selected != after_slots
            or pre_unselected != post_unselected
            or any(
                slot.capacity_receipt_sha256 != capacity_sha
                or slot.capacity_value != capacity
                for slot in (*pre_owner, *post_owner)
            )
            or any(
                slot.lifecycle not in (SLOT_PARKED, SLOT_RETIRED)
                or slot.published_to_runtime
                or not slot.physically_parked
                for slot in pre_selected
            )
            or any(slot.mutation_version > before for slot in pre_owner)
            or any(slot.mutation_version > after for slot in post_owner)
            or any(
                slot.outcome_key is not None
                and slot.outcome_key.reset_generation != reset_before[slot.env_id]
                for slot in pre_owner
            )
            or any(
                slot.outcome_key is not None
                and slot.outcome_key.reset_generation != reset_after[slot.env_id]
                for slot in post_owner
            )
            or self.selected_slots_root_before_sha256
            != physical_slot_root(before_slots)
            or self.selected_slots_root_after_sha256 != physical_slot_root(after_slots)
            or self.unselected_slots_root_before_sha256
            != self.unselected_slots_root_after_sha256
            or self.unselected_slots_root_before_sha256
            != physical_slot_root(pre_unselected)
            or self.selected_slots_root_before_sha256
            != physical_slot_root(pre_selected)
            or self.selected_slots_root_after_sha256
            != physical_slot_root(post_selected)
            or self.physical_owner_checkpoint_before_sha256
            != physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity_sha,
                num_envs=num_envs,
                flight_capacity=capacity,
                mutation_version=before,
                next_prepare_nonce=nonce,
                reset_generations=reset_before,
                slots=pre_owner,
                poisoned=False,
            )
            or self.physical_owner_checkpoint_after_sha256
            != physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity_sha,
                num_envs=num_envs,
                flight_capacity=capacity,
                mutation_version=after,
                next_prepare_nonce=nonce,
                reset_generations=reset_after,
                slots=post_owner,
                poisoned=False,
            )
            or _exact_bool(self.env_reset_invoked, label="env_reset_invoked")
            or _exact_bool(
                self.mailbox_lifecycle_mutated, label="mailbox_lifecycle_mutated"
            )
        ):
            raise PhysicalFlightContractError("true-reset ownership/parity differs")
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "mutation_version_before", before)
        object.__setattr__(self, "mutation_version_after", after)
        object.__setattr__(self, "num_envs", num_envs)
        object.__setattr__(self, "flight_capacity", capacity)
        object.__setattr__(self, "reset_generations_before", reset_before)
        object.__setattr__(self, "reset_generations_after", reset_after)
        object.__setattr__(self, "next_prepare_nonce", nonce)
        object.__setattr__(self, "pre_owner_slot_snapshots", pre_owner)
        object.__setattr__(self, "post_owner_slot_snapshots", post_owner)

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["zero_open_all_owner_closure"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["zero_open_all_owner_closure"]
        )
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["reset_generations_before"] = tuple(
            values["reset_generations_before"]
        )
        values["reset_generations_after"] = tuple(
            values["reset_generations_after"]
        )
        values["pre_owner_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["pre_owner_slot_snapshots"]
        )
        values["post_owner_slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["post_owner_slot_snapshots"]
        )
        values["rows"] = tuple(
            PhysicalTrueResetRow._from_mapping_unpinned(row)
            for row in values["rows"]
        )
        return values


@dataclass(frozen=True)
class PhysicalFlightCheckpointReceipt(_SealedRecord):
    """Complete checkpoint: typed slots plus full canonical owner-state bytes.

    The checkpoint mutation version is the owner-wide high-water mark.  Slot
    versions are per-slot and may be lower after masked operations, but may
    never be ahead of the owner high-water mark.
    """

    KIND: ClassVar[str] = "action_ball_physical_flight_checkpoint_receipt_v2"

    integration_status: str
    capacity_receipt: FrozenFlightCapacityReceipt
    capacity_receipt_sha256: str
    checkpoint_boundary_authority: CanonicalJsonContentPin
    num_envs: int
    flight_capacity: int
    mutation_version: int
    next_prepare_nonce: int
    pending_prepare_receipt_sha256: Optional[str]
    slot_snapshots: Tuple[PhysicalFlightSlotSnapshot, ...]
    slot_root_sha256: str
    owner_state_schema_sha256: str
    owner_state_bytes_base64: str
    owner_state_byte_length: int
    owner_state_bytes_sha256: str
    complete_env_step: bool
    env_reset_invoked: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise PhysicalFlightContractError("checkpoint integration status differs")
        if not isinstance(self.capacity_receipt, FrozenFlightCapacityReceipt) or not isinstance(
            self.checkpoint_boundary_authority, CanonicalJsonContentPin
        ):
            raise PhysicalFlightContractError("checkpoint authority type differs")
        for name in (
            "capacity_receipt_sha256",
            "slot_root_sha256",
            "owner_state_schema_sha256",
            "owner_state_bytes_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        if self.capacity_receipt_sha256 != self.capacity_receipt.canonical_sha256:
            raise PhysicalFlightContractError("checkpoint capacity digest differs")
        num_envs = _plain_int(self.num_envs, label="num_envs", minimum=1)
        capacity = _plain_int(self.flight_capacity, label="flight_capacity", minimum=1)
        version = _plain_int(self.mutation_version, label="mutation_version")
        nonce = _plain_int(self.next_prepare_nonce, label="next_prepare_nonce", minimum=1)
        pending = _optional_sha256(
            self.pending_prepare_receipt_sha256,
            label="pending_prepare_receipt_sha256",
        )
        if capacity != self.capacity_receipt.configured_flight_capacity:
            raise PhysicalFlightContractError("checkpoint capacity value differs")
        slots = tuple(self.slot_snapshots)
        if any(not isinstance(row, PhysicalFlightSlotSnapshot) for row in slots):
            raise PhysicalFlightContractError("checkpoint slot type differs")
        expected_keys = tuple(
            (env_id, slot_index)
            for env_id in range(num_envs)
            for slot_index in range(capacity)
        )
        if tuple((row.env_id, row.slot_index) for row in slots) != expected_keys or any(
            row.capacity_receipt_sha256 != self.capacity_receipt_sha256
            or row.capacity_value != capacity
            or row.mutation_version > version
            for row in slots
        ):
            raise PhysicalFlightContractError("checkpoint full slot grid differs")
        if self.slot_root_sha256 != physical_slot_root(slots):
            raise PhysicalFlightContractError("checkpoint slot root differs")
        owner_raw, owner = _canonical_mapping_from_base64(
            self.owner_state_bytes_base64,
            label="owner_state_bytes_base64",
            byte_length=self.owner_state_byte_length,
            bytes_sha256=self.owner_state_bytes_sha256,
        )
        if (
            self.owner_state_schema_sha256
            != PHYSICAL_OWNER_STATE_SCHEMA_SHA256
            or frozenset(owner) != PHYSICAL_OWNER_STATE_REQUIRED_FIELDS
            or any(
                owner.get(key) != value
                for key, value in PHYSICAL_OWNER_STATE_SCHEMA.items()
            )
            or owner.get("capacity_receipt_sha256")
            != self.capacity_receipt_sha256
            or owner.get("num_envs") != num_envs
            or owner.get("flight_capacity") != capacity
            or owner.get("owner_mutation_version") != version
            or owner.get("next_prepare_nonce") != nonce
            or owner.get("poisoned") is not False
        ):
            raise PhysicalFlightContractError(
                "checkpoint owner-state schema/header differs"
            )
        owner_slots_raw = _sequence(
            owner["slot_snapshots"], label="owner_state.slot_snapshots"
        )
        try:
            owner_slots = tuple(
                PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
                for row in owner_slots_raw
            )
        except Exception as exc:
            raise PhysicalFlightContractError(
                "checkpoint owner-state slot bytes differ"
            ) from exc
        if owner_slots != slots:
            raise PhysicalFlightContractError(
                "checkpoint owner-state slots differ from receipt"
            )
        reset_generation = _sequence(
            owner["reset_generation"], label="owner_state.reset_generation"
        )
        if len(reset_generation) != num_envs or any(
            type(item) is not int or item < 1 for item in reset_generation
        ):
            raise PhysicalFlightContractError(
                "checkpoint owner-state reset generations differ"
            )
        if any(
            slot.outcome_key is not None
            and slot.outcome_key.reset_generation
            != reset_generation[slot.env_id]
            for slot in slots
        ):
            raise PhysicalFlightContractError(
                "checkpoint slot/reset-generation join differs"
            )

        scene_b64 = owner["scene_state_f32_base64"]
        scene_length = num_envs * capacity * len(STATE_COMPONENTS) * 4
        if type(scene_b64) is not str:
            raise PhysicalFlightContractError(
                "checkpoint owner-state scene bytes are missing"
            )
        try:
            scene_raw = base64.b64decode(scene_b64.encode("ascii"), validate=True)
        except (ValueError, UnicodeError) as exc:
            raise PhysicalFlightContractError(
                "checkpoint owner-state scene base64 differs"
            ) from exc
        if (
            owner["scene_state_shape"]
            != [num_envs, capacity, len(STATE_COMPONENTS)]
            or owner["scene_state_byte_length"] != scene_length
            or len(scene_raw) != scene_length
            or owner["scene_state_bytes_sha256"]
            != hashlib.sha256(scene_raw).hexdigest()
            or any(
                not math.isfinite(value[0])
                for value in struct.iter_unpack("<f", scene_raw)
            )
        ):
            raise PhysicalFlightContractError(
                "checkpoint owner-state scene byte pin differs"
            )
        park_values = (
            0.0,
            0.0,
            -20.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        row_bytes = len(STATE_COMPONENTS) * 4
        for index, slot in enumerate(slots):
            expected_values = (
                slot.current_state_f32.ordered_values
                if slot.lifecycle in (SLOT_IN_FLIGHT, SLOT_SETTLED_RETAINED)
                else park_values
            )
            if scene_raw[index * row_bytes : (index + 1) * row_bytes] != struct.pack(
                "<13f", *expected_values
            ):
                raise PhysicalFlightContractError(
                    "checkpoint scene bytes/portable slot state join differs"
                )

        def matrix(value: object, *, label: str) -> tuple[tuple[object, ...], ...]:
            outer = _sequence(value, label=label)
            if len(outer) != num_envs:
                raise PhysicalFlightContractError(f"{label} env width differs")
            result = tuple(
                _sequence(row, label=f"{label}[{env_id}]")
                for env_id, row in enumerate(outer)
            )
            if any(len(row) != capacity for row in result):
                raise PhysicalFlightContractError(f"{label} slot width differs")
            return result

        lifecycle = matrix(
            owner["flight_lifecycle_code"],
            label="owner_state.flight_lifecycle_code",
        )
        ordinal = matrix(
            owner["observation_ordinal"],
            label="owner_state.observation_ordinal",
        )
        faults = matrix(owner["device_fault"], label="owner_state.device_fault")
        previous_rows = matrix(
            owner["previous_ball_center_m"],
            label="owner_state.previous_ball_center_m",
        )
        if any(
            type(value) is not int or value not in (0, 1, 2, 3)
            for row in lifecycle
            for value in row
        ) or any(
            type(value) is not int or value < -1
            for row in ordinal
            for value in row
        ) or any(
            type(value) is not bool or value
            for row in faults
            for value in row
        ):
            raise PhysicalFlightContractError(
                "checkpoint owner-state lifecycle/ordinal/fault differs"
            )
        for env_id, row in enumerate(previous_rows):
            for slot_index, vector in enumerate(row):
                _f32_vector(
                    vector,
                    width=3,
                    label=(
                        "owner_state.previous_ball_center_m"
                        f"[{env_id}][{slot_index}]"
                    ),
                )
        for slot in slots:
            code = lifecycle[slot.env_id][slot.slot_index]
            if (
                (slot.lifecycle == SLOT_PARKED and code != 0)
                or (slot.lifecycle == SLOT_RETIRED and code != 0)
                or (
                    slot.lifecycle == SLOT_SETTLED_RETAINED
                    and code != 3
                )
                or (
                    slot.lifecycle == SLOT_IN_FLIGHT
                    and code not in (1, 2, 3)
                )
            ):
                raise PhysicalFlightContractError(
                    "checkpoint portable/device lifecycle join differs"
                )
        pending_r06 = owner["pending_r06_settlement_ack"]
        if pending_r06 is not None:
            raise PhysicalFlightContractError(
                "complete env-step checkpoint cannot retain transient R06 settlement"
            )
        if any(
            lifecycle[slot.env_id][slot.slot_index] == 3
            and slot.published_to_runtime
            for slot in slots
        ):
            raise PhysicalFlightContractError(
                "complete env-step checkpoint cannot retain an unretired settled flight"
            )
        if owner_raw != canonical_json_bytes(owner):
            raise PhysicalFlightContractError(
                "checkpoint owner-state bytes are not canonical"
            )
        if not _exact_bool(self.complete_env_step, label="complete_env_step") or _exact_bool(
            self.env_reset_invoked, label="env_reset_invoked"
        ):
            raise PhysicalFlightContractError(
                "checkpoint must be a complete non-reset env-step boundary"
            )
        if pending is not None:
            raise PhysicalFlightContractError(
                "complete env-step checkpoint cannot retain a prepared install"
            )
        object.__setattr__(self, "num_envs", num_envs)
        object.__setattr__(self, "flight_capacity", capacity)
        object.__setattr__(self, "mutation_version", version)
        object.__setattr__(self, "next_prepare_nonce", nonce)
        object.__setattr__(self, "pending_prepare_receipt_sha256", pending)
        object.__setattr__(self, "slot_snapshots", slots)

    @classmethod
    def _reject_legacy(cls, value: object) -> None:
        if not isinstance(value, Mapping):
            return
        if value.get("kind") in _LEGACY_DIGEST_ONLY_CHECKPOINT_KINDS or (
            "owner_state_bytes_sha256" in value
            and "owner_state_bytes_base64" not in value
        ):
            raise DigestOnlyPayloadTombstonedError(
                "digest-only physical checkpoint is tombstoned"
            )

    @classmethod
    def _decode_values(cls, values: dict[str, object]) -> dict[str, object]:
        values["capacity_receipt"] = FrozenFlightCapacityReceipt._from_mapping_unpinned(
            values["capacity_receipt"]
        )
        values["checkpoint_boundary_authority"] = CanonicalJsonContentPin._from_mapping_unpinned(
            values["checkpoint_boundary_authority"]
        )
        values["slot_snapshots"] = tuple(
            PhysicalFlightSlotSnapshot._from_mapping_unpinned(row)
            for row in values["slot_snapshots"]
        )
        return values


__all__ = [
    "ANGULAR_VELOCITY_FRAME",
    "CAPACITY_FORMULA",
    "CENSORED_REVEAL_BATCH_KIND",
    "COMMITTED_REVEAL_BATCH_KIND",
    "CanonicalJsonContentPin",
    "CanonicalPhysicalBallStateF32",
    "DigestOnlyPayloadTombstonedError",
    "ExternalContentPinError",
    "FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN",
    "FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER",
    "FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND",
    "FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256",
    "FULL_MDP_REVEAL_BOUNDARY_PACKET_SCHEMA_VERSION",
    "FULL_MDP_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256",
    "FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256",
    "FULL_MDP_REVEAL_DECISION_ACCEPT",
    "FULL_MDP_REVEAL_DECISION_CENSOR",
    "FrozenFlightCapacityReceipt",
    "INCLUSIVE_INTERVAL_SEMANTICS",
    "INSTALL_STATE_EPOCH",
    "INTEGRATION_STATUS",
    "LAUNCH_AUTHORIZED",
    "LINEAR_VELOCITY_FRAME",
    "POSITION_FRAME",
    "PhysicalBallInstallPayload",
    "PhysicalFlightCheckpointReceipt",
    "PhysicalFlightContractError",
    "PhysicalFlightOutcomeKey",
    "PhysicalFlightSlotSnapshot",
    "PhysicalFlightTaskRef",
    "PhysicalInstallAbortReceipt",
    "PhysicalInstallCensorReceipt",
    "PhysicalInstallCommitReceipt",
    "PhysicalInstallPrepareReceipt",
    "PHYSICAL_OWNER_STATE_KIND",
    "PHYSICAL_OWNER_STATE_REQUIRED_FIELDS",
    "PHYSICAL_OWNER_STATE_SCHEMA",
    "PHYSICAL_OWNER_STATE_SCHEMA_SHA256",
    "PhysicalRetireReceipt",
    "PhysicalRetireRow",
    "PhysicalTrueResetReceipt",
    "PhysicalTrueResetRow",
    "PreparedPhysicalInstallRow",
    "QUANTIZATION_CONTRACT",
    "QUATERNION_ORDER",
    "R05TerminalClaimProjection",
    "R05_PREPARED_TERMINAL_CONTENT_PIN_KIND",
    "R05_PREPARED_TERMINAL_CONTENT_PIN_SCHEMA_SHA256",
    "R05_TERMINAL_BOUNDARY_AUTHORITY_KIND",
    "R05_TERMINAL_BOUNDARY_CENSOR_EVIDENCE_KIND",
    "R05_TERMINAL_BOUNDARY_PARTICIPANT_ROOT_KIND",
    "R05_TERMINAL_BOUNDARY_PROJECTION_KIND",
    "R05_TERMINAL_BOUNDARY_PROJECTION_SCHEMA_SHA256",
    "PREPARED_REVEAL_TERMINAL_CLAIM_KIND",
    "REVEAL_FINAL_PREVIEW_KIND",
    "R05_REVEAL_FINAL_PREVIEW_SCHEMA_SHA256",
    "REVEAL_PREPARE_BOUNDARY_MARKER_KIND",
    "REVEAL_PREPARE_BOUNDARY_MARKER_SCHEMA_SHA256",
    "R05_PREARM_CHILD_OWNER_KINDS",
    "RUNTIME_INTEGRATED",
    "RUNTIME_DTYPE",
    "SAME_TICK_ORDERING",
    "SCHEMA_VERSION",
    "POD_VALIDATED",
    "SLOT_IN_FLIGHT",
    "SLOT_LIFECYCLES",
    "SLOT_PARKED",
    "SLOT_RETIRED",
    "SLOT_SETTLED_RETAINED",
    "STATE_COMPONENTS",
    "ZERO_CANONICALIZATION",
    "canonical_json_bytes",
    "canonical_sha256",
    "installed_ball_state_binding_sha256",
    "physical_slot_root",
    "physical_owner_checkpoint_root",
]
