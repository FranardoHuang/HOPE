"""Leaf protocol for the one all-owner ActionBall reveal boundary.

This module owns no R05, Motion, Racket, physical-scene, R06, Manager, or
environment state.  Its only live operation is one fixed-layout packed
device-to-host boundary over four already-prepared owner rows.  The rows are
deliberately boundary-free: each binds only its own R05 preview, selected
environment identities, mutation version, canonical token root, pass mask,
and fault bits.  No row may depend on the later packet, marker, receipt, or
another owner.

Packet v4 is one fixed ``257 + 55*N`` device row.  It retains the four child
rows from v3 and appends one independent Device-R05 pre-transfer section to
the same tensor and therefore the same sole D2H.  The D05 section is not a
fifth child: it carries selection rank, construction admissibility, producer
fault, and counter-overflow facts already retained by the D05 owner before
the transfer.  Each positional 32-byte child token
root carrier is XOR-bound to the owner's fault-schema root and a deterministic
four-word checksum over the exact live device version plus the full pass/fault
row.  This detects ordinary single or coupled packet/row drift without another
host transfer.  It is deliberately not a secret, MAC, or hostile-process
security boundary.

A well-formed owner fault produces a typed ``CENSOR`` receipt.  Packet shape,
ordering, root, version, selection, or replay mismatches instead poison this
boundary owner because they make the evidence unattributable.  Object identity
and the private receipt registry prevent accidental copies from being treated
as owner-issued receipts; they are transaction provenance, not a secret, MAC,
or hostile-process security boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import threading
import time
from typing import Callable, ClassVar, Mapping, Optional, Protocol, Sequence, Tuple
import weakref

import torch


SCHEMA_VERSION = 1
PACKET_SCHEMA_VERSION = 4
PACKET_KIND = "action_ball_full_mdp_reveal_boundary_packet_v4"
RECEIPT_KIND = "action_ball_full_mdp_reveal_boundary_receipt_v1"
OWNER_ROW_KIND = "action_ball_full_mdp_reveal_boundary_owner_row_v1"
FAULT_SCHEMA_KIND = "action_ball_full_mdp_reveal_boundary_fault_schema_v1"

OWNER_ORDER = (
    "motion",
    "racket",
    "physical_ball",
    "r06_flight",
)
OWNER_COUNT = 4
TOKEN_NBYTES = 32
INT64_NBYTES = 8
HEADER_INT64_COUNT = 8
HEADER_NBYTES = HEADER_INT64_COUNT * INT64_NBYTES
PACKET_FIXED_NBYTES = 256
PACKET_PER_ENV_NBYTES = 55
FAULT_WORD_NBYTES = INT64_NBYTES
D05_COUNTER_OVERFLOW_FAULT_BIT = 1 << 62
D05_SELECTION_MISMATCH_FAULT_BIT = 1 << 61

DECISION_ACCEPT = "ACCEPT"
DECISION_CENSOR = "CENSOR"
DECISIONS = (DECISION_ACCEPT, DECISION_CENSOR)

# The boundary seam below is implemented, but the complete production graph is
# still held because current task/cadence/question authorities do not yet issue
# every identity required by the four hot leaves.  This flag therefore stays
# false: one working causal seam must not be reported as whole-graph closure.
D05_PRETRANSFER_ADAPTER_READY = False
D05_DECISION_CONSTRUCTION_REJECT = "CONSTRUCTION_REJECT"
D05_PRETRANSFER_ADAPTER_CONTRACT = (
    (
        "schema_kind",
        "action_ball_full_mdp_d05_pretransfer_adapter_contract_v1",
    ),
    (
        "authority",
        "device_r05_owner_issued_opaque_pretransfer_token",
    ),
    (
        "same_transfer_facts",
        (
            "selected_env_index",
            "selected_mask",
            "construction_admissible",
            "producer_fault",
            "counter_overflow_fault",
        ),
    ),
    (
        "decision_precedence",
        (
            "any_d05_or_child_owner_fault_then_CENSOR",
            "else_any_construction_inadmissible_then_CONSTRUCTION_REJECT",
            "else_ACCEPT",
        ),
    ),
    (
        "construction_reject_semantics",
        "ordinary_no_eligible_without_fault_is_non_poisoning",
    ),
    (
        "transfer_semantics",
        "all_facts_in_the_existing_single_packed_device_to_host_transfer",
    ),
    (
        "forbidden_authority",
        (
            "caller_boolean_override",
            "post_transfer_fact_injection",
            "second_device_to_host_transfer",
            "assert_async_same_batch_authorization",
        ),
    ),
)
D05_PRETRANSFER_ADAPTER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        D05_PRETRANSFER_ADAPTER_CONTRACT,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()

PACKET_ROW_INTEGRITY_SCHEMA = (
    ("schema_kind", "action_ball_full_mdp_packet_row_integrity_v1"),
    ("pass_marker_bit", 63),
    ("verdict_word", "fault_bits xor (int(pass_mask) left_shift 63)"),
    ("environment_index", "i=env_id+1 over every env_id in full_N_row"),
    ("owner_index", "owner=position_in_fixed_OWNER_ORDER+1"),
    ("owner_domain_terms", (40503, 34283, 49843, 10196)),
    ("version_coefficients", (1, 5, 257, 65537)),
    ("lane_0", "v+owner*40503+sum(verdict+i*3)"),
    ("lane_1", "v*5+owner*34283+sum(verdict*(i*5+1)+i*17)"),
    (
        "lane_2",
        "v*257+owner*49843+sum(verdict*(i*i+3)+i*65537)",
    ),
    (
        "lane_3",
        "v*65537+owner*10196+sum((verdict xor (i*257))*(i+17))",
    ),
    ("arithmetic", "twos_complement_uint64_modulo"),
    (
        "root_carrier",
        "actual_root_xor_fault_schema_root_xor_four_little_endian_integrity_words",
    ),
    ("security_semantics", "deterministic_corruption_check_not_mac_or_secret"),
)
PACKET_ROW_INTEGRITY_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        PACKET_ROW_INTEGRITY_SCHEMA,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()

RECEIPT_SCHEMA = (
    (
        "schema_kind",
        "action_ball_full_mdp_reveal_boundary_receipt_schema_v1",
    ),
    ("schema_version", SCHEMA_VERSION),
    ("receipt_kind", RECEIPT_KIND),
    ("packet_schema_version", PACKET_SCHEMA_VERSION),
    ("owner_order", OWNER_ORDER),
    (
        "decisions",
        (*DECISIONS, D05_DECISION_CONSTRUCTION_REJECT),
    ),
    (
        "public_to_mapping_fields",
        (
            "schema_version",
            "kind",
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
        ),
    ),
    (
        "owner_row_to_mapping_fields",
        (
            "kind",
            "owner_kind",
            "owner_mutation_version",
            "owner_token_root_sha256",
            "fault_schema_sha256",
            "allowed_fault_mask",
            "selected_pass",
            "selected_fault_bits",
        ),
    ),
    ("boundary_transfer_count_semantics", "operation_local_exactly_one"),
    ("transfer_totals_semantics", "cumulative_after_this_transfer"),
    (
        "owner_row_semantics",
        "host_version_and_selected_verdicts_decode_only_from_sole_packet",
    ),
    (
        "packet_owner_root_semantics",
        PACKET_ROW_INTEGRITY_SCHEMA,
    ),
    (
        "authority_semantics",
        "portable_public_mapping_and_sha256_are_evidence_not_capability",
    ),
)
RECEIPT_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        RECEIPT_SCHEMA,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()

_HEX = frozenset("0123456789abcdef")
_LANE_AUTH_TOKEN = object()
_RECEIPT_AUTH_TOKEN = object()
_ABORT_AUTH_TOKEN = object()
_PacketMutator = Optional[Callable[[torch.Tensor], torch.Tensor]]


class ActionBallFullMdpRevealBoundaryError(RuntimeError):
    """The all-owner reveal boundary contract was not satisfied."""


class ActionBallFullMdpRevealBoundaryPoisonedError(
    ActionBallFullMdpRevealBoundaryError
):
    """The boundary owner observed evidence that cannot be attributed."""


class ActionBallFullMdpRevealBoundaryD05AdapterHold(
    ActionBallFullMdpRevealBoundaryError
):
    """Device-R05 facts were not supplied before the sole packed transfer."""


class ActionBallFullMdpRevealBoundaryD05PreTransferToken:
    """Future opaque Device-R05 capability; ordinary construction is invalid."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("Device-R05 pre-transfer tokens are owner-issued")

    def __copy__(self):
        raise TypeError("Device-R05 pre-transfer tokens cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Device-R05 pre-transfer tokens cannot be copied")

    def __reduce__(self):
        raise TypeError("Device-R05 pre-transfer tokens cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Device-R05 pre-transfer tokens cannot be serialized")


@dataclass(frozen=True)
class ActionBallFullMdpRevealBoundaryD05PreTransferView:
    """Clone-only facts the Device-R05 authority issues pre-transfer.

    The boundary must pack these exact device tensors into its existing sole
    transfer.  ``owner_fault_present`` is deliberately absent: it is derived
    inside that packet from ``producer_fault`` and ``counter_overflow_fault``;
    callers cannot override it with a bool.
    """

    preview_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor
    construction_admissible: torch.Tensor
    producer_fault: torch.Tensor
    counter_overflow_fault: torch.Tensor


class ActionBallFullMdpRevealBoundaryD05PreTransferAuthority(Protocol):
    """Exact validator for one Device-R05 owner-issued capability."""

    def require_owned_r05_pretransfer_boundary_input(
        self,
        token: ActionBallFullMdpRevealBoundaryD05PreTransferToken,
        *,
        device: torch.device,
        num_envs: int,
    ) -> ActionBallFullMdpRevealBoundaryD05PreTransferView: ...


def packet_nbytes(num_envs: int) -> int:
    """Return the exact v4 packet width ``257 + 55 * num_envs``."""

    count = _plain_int(num_envs, label="num_envs", minimum=1)
    return PACKET_FIXED_NBYTES + PACKET_PER_ENV_NBYTES * count


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise ActionBallFullMdpRevealBoundaryError(
            f"{label} must be an exact int"
        )
    if value < minimum or (maximum is not None and value > maximum):
        raise ActionBallFullMdpRevealBoundaryError(
            f"{label} is outside its range"
        )
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ActionBallFullMdpRevealBoundaryError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _ordered_env_ids(
    values: object,
    *,
    label: str,
    num_envs: int,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ActionBallFullMdpRevealBoundaryError(
            f"{label} must be an ordered environment sequence"
        )
    rows = tuple(values)
    if (
        not rows
        or any(type(value) is not int for value in rows)
        or rows != tuple(sorted(set(rows)))
        or rows[0] < 0
        or rows[-1] >= num_envs
    ):
        raise ActionBallFullMdpRevealBoundaryError(
            f"{label} must be sorted, unique, non-empty, and in range"
        )
    return rows


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ActionBallFullMdpRevealBoundaryError(
            "boundary receipt is not canonical ASCII JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_tensor(
    value: object,
    *,
    label: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
    ):
        raise ActionBallFullMdpRevealBoundaryError(
            f"{label} tensor shape/dtype/device differs"
        )
    return value


def _encode_int64_little_endian(values: torch.Tensor) -> torch.Tensor:
    """Encode contiguous int64 values as explicit little-endian bytes."""

    if not isinstance(values, torch.Tensor) or values.dtype != torch.int64:
        raise ActionBallFullMdpRevealBoundaryError(
            "little-endian encoder requires an int64 tensor"
        )
    shifts = torch.arange(
        0, 64, 8, dtype=torch.int64, device=values.device
    )
    return torch.bitwise_and(
        torch.bitwise_right_shift(values.reshape(-1, 1), shifts), 0xFF
    ).to(dtype=torch.uint8).reshape(-1)


def _device_row_integrity_words(
    *,
    owner_index: int,
    device_owner_mutation_version: torch.Tensor,
    pass_mask: torch.Tensor,
    fault_bits: torch.Tensor,
) -> torch.Tensor:
    """Return four non-secret device checksums without synchronizing host."""

    indices = torch.arange(
        1,
        pass_mask.shape[0] + 1,
        dtype=torch.int64,
        device=pass_mask.device,
    )
    pass_marker = torch.bitwise_left_shift(
        pass_mask.to(dtype=torch.int64), 63
    )
    verdicts = torch.bitwise_xor(fault_bits, pass_marker)
    lanes = torch.stack(
        (
            verdicts + indices * 3,
            verdicts * (indices * 5 + 1) + indices * 17,
            verdicts * (indices * indices + 3) + indices * 65537,
            torch.bitwise_xor(verdicts, indices * 257) * (indices + 17),
        ),
        dim=0,
    ).sum(dim=1)
    version = device_owner_mutation_version.reshape(())
    version_terms = torch.stack(
        (version, version * 5, version * 257, version * 65537),
        dim=0,
    )
    owner = owner_index + 1
    owner_terms = torch.tensor(
        (owner * 40503, owner * 34283, owner * 49843, owner * 10196),
        dtype=torch.int64,
        device=pass_mask.device,
    )
    return version_terms + owner_terms + lanes


def _host_row_integrity_mask(
    *,
    owner_index: int,
    owner_mutation_version: int,
    pass_mask: Sequence[bool],
    fault_bits: Sequence[int],
) -> bytes:
    """Mirror packet-v3 checksum arithmetic over decoded host facts."""

    modulo_mask = (1 << 64) - 1
    owner = owner_index + 1
    accumulators = [
        owner_mutation_version + owner * 40503,
        owner_mutation_version * 5 + owner * 34283,
        owner_mutation_version * 257 + owner * 49843,
        owner_mutation_version * 65537 + owner * 10196,
    ]
    for env_index, (passed, fault) in enumerate(
        zip(pass_mask, fault_bits), start=1
    ):
        verdict = (fault ^ ((1 if passed else 0) << 63)) & modulo_mask
        accumulators[0] = (
            accumulators[0] + verdict + env_index * 3
        ) & modulo_mask
        accumulators[1] = (
            accumulators[1]
            + verdict * (env_index * 5 + 1)
            + env_index * 17
        ) & modulo_mask
        accumulators[2] = (
            accumulators[2]
            + verdict * (env_index * env_index + 3)
            + env_index * 65537
        ) & modulo_mask
        accumulators[3] = (
            accumulators[3]
            + (verdict ^ (env_index * 257)) * (env_index + 17)
        ) & modulo_mask
    return b"".join(
        value.to_bytes(INT64_NBYTES, byteorder="little", signed=False)
        for value in accumulators
    )


class _Identity:
    """Weak-referenceable single-use identity; it carries no secret."""


@dataclass(frozen=True)
class ActionBallFullMdpRevealBoundaryFaultSchema:
    """Externally pinned known-bit domain for one owner's censor facts."""

    KIND: ClassVar[str] = FAULT_SCHEMA_KIND

    schema_version: int
    owner_kind: str
    ordered_fault_bits: Tuple[Tuple[str, int], ...]
    allowed_fault_mask: int
    precedence: Tuple[str, ...]

    def __post_init__(self) -> None:
        version = _plain_int(
            self.schema_version,
            label="fault_schema_version",
            minimum=1,
        )
        if self.owner_kind not in OWNER_ORDER:
            raise ActionBallFullMdpRevealBoundaryError(
                "fault schema owner kind differs"
            )
        raw_bits = tuple(self.ordered_fault_bits)
        bits = []
        for index, raw in enumerate(raw_bits):
            if (
                not isinstance(raw, Sequence)
                or isinstance(raw, (str, bytes))
                or len(raw) != 2
                or type(raw[0]) is not str
                or not raw[0]
                or type(raw[1]) is not int
                or raw[1] <= 0
                or raw[1] > (1 << 62)
                or raw[1] & (raw[1] - 1)
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    f"ordered_fault_bits[{index}] differs"
                )
            bits.append((raw[0], raw[1]))
        if (
            not bits
            or len({name for name, _ in bits}) != len(bits)
            or len({bit for _, bit in bits}) != len(bits)
            or tuple(bit for _, bit in bits)
            != tuple(sorted(bit for _, bit in bits))
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "ordered fault bit names/order differ"
            )
        expected_mask = 0
        for _, bit in bits:
            expected_mask |= bit
        mask = _plain_int(
            self.allowed_fault_mask,
            label="allowed_fault_mask",
            minimum=1,
            maximum=(1 << 63) - 1,
        )
        precedence = tuple(self.precedence)
        names = tuple(name for name, _ in bits)
        if mask != expected_mask or (
            len(precedence) != len(names)
            or set(precedence) != set(names)
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "fault schema mask/precedence differs"
            )
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "ordered_fault_bits", tuple(bits))
        object.__setattr__(self, "allowed_fault_mask", mask)
        object.__setattr__(self, "precedence", precedence)

    @property
    def schema_sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": self.KIND,
            "schema_version": self.schema_version,
            "owner_kind": self.owner_kind,
            "ordered_fault_bits": [
                [name, bit] for name, bit in self.ordered_fault_bits
            ],
            "allowed_fault_mask": self.allowed_fault_mask,
            "precedence": list(self.precedence),
        }


@dataclass(frozen=True, eq=False)
class ActionBallFullMdpRevealBoundaryPreparedTokenClaim:
    """Typed facts returned by one child owner's retained-token validator."""

    owner_kind: str
    device_owner_mutation_version: torch.Tensor = field(repr=False)
    owner_token_root_sha256: str
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    _prepared_token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self.owner_kind not in OWNER_ORDER:
            raise ActionBallFullMdpRevealBoundaryError(
                "prepared token owner kind differs"
            )
        device_version = self.device_owner_mutation_version
        if (
            not isinstance(device_version, torch.Tensor)
            or tuple(device_version.shape) not in ((), (1,))
            or device_version.dtype != torch.int64
            or device_version.device.type not in ("cpu", "cuda")
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "prepared token device mutation version differs"
            )
        object.__setattr__(
            self,
            "owner_token_root_sha256",
            _sha256(
                self.owner_token_root_sha256,
                label="prepared_token.owner_token_root_sha256",
            ),
        )
        object.__setattr__(
            self,
            "reveal_final_preview_schema_version",
            _plain_int(
                self.reveal_final_preview_schema_version,
                label="prepared_token.reveal_final_preview_schema_version",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "reveal_final_preview_sha256",
            _sha256(
                self.reveal_final_preview_sha256,
                label="prepared_token.reveal_final_preview_sha256",
            ),
        )


class ActionBallFullMdpRevealBoundaryChildTokenAuthority:
    """Bind one lane to an actual child owner's exact-token registry."""

    __slots__ = ("_owner_kind", "_validator")

    def __init__(
        self,
        *,
        owner_kind: str,
        validator: Callable[
            [object], ActionBallFullMdpRevealBoundaryPreparedTokenClaim
        ],
    ) -> None:
        if owner_kind not in OWNER_ORDER or not callable(validator):
            raise ActionBallFullMdpRevealBoundaryError(
                "child token authority kind/validator differs"
            )
        self._owner_kind = owner_kind
        self._validator = validator

    @property
    def owner_kind(self) -> str:
        return self._owner_kind

    def require_owned_prepared_token(
        self, prepared_token: object
    ) -> ActionBallFullMdpRevealBoundaryPreparedTokenClaim:
        """Revalidate one exact token against the child-retained registry."""

        claim = self._validator(prepared_token)
        if (
            type(claim)
            is not ActionBallFullMdpRevealBoundaryPreparedTokenClaim
            or claim.owner_kind != self.owner_kind
            or claim._prepared_token is not prepared_token
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                f"{self.owner_kind} child token authority differs"
            )
        return claim


@dataclass(frozen=True, eq=False)
class ActionBallFullMdpRevealBoundaryDeviceRow:
    """Opaque boundary-free device row minted through one owner lane."""

    _owner_kind: str
    _owner_token_root_sha256: str
    _reveal_final_preview_schema_version: int
    _reveal_final_preview_sha256: str
    _selected_env_ids: Tuple[int, ...]
    _pass_mask: torch.Tensor = field(repr=False)
    _fault_bits: torch.Tensor = field(repr=False)
    _device_owner_mutation_version: torch.Tensor = field(repr=False)
    _device_owner_token_root: torch.Tensor = field(repr=False)
    _device_fault_schema_root: torch.Tensor = field(repr=False)
    _fault_schema: ActionBallFullMdpRevealBoundaryFaultSchema = field(
        repr=False
    )
    _prepared_token: object = field(repr=False)
    _prepared_token_claim: ActionBallFullMdpRevealBoundaryPreparedTokenClaim = (
        field(repr=False)
    )
    _child_token_authority: ActionBallFullMdpRevealBoundaryChildTokenAuthority = (
        field(repr=False)
    )
    _issuer: object = field(repr=False)
    _identity: _Identity = field(repr=False)

    @property
    def owner_kind(self) -> str:
        return self._owner_kind

    @property
    def owner_token_root_sha256(self) -> str:
        return self._owner_token_root_sha256

    @property
    def reveal_final_preview_schema_version(self) -> int:
        return self._reveal_final_preview_schema_version

    @property
    def reveal_final_preview_sha256(self) -> str:
        return self._reveal_final_preview_sha256

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self._selected_env_ids

    @property
    def fault_schema(self) -> ActionBallFullMdpRevealBoundaryFaultSchema:
        return self._fault_schema

    @property
    def child_token_authority(
        self,
    ) -> ActionBallFullMdpRevealBoundaryChildTokenAuthority:
        return self._child_token_authority


class ActionBallFullMdpRevealBoundaryLaneAuthority:
    """Fixed-kind capability used by one child owner to mint device rows."""

    __slots__ = (
        "_boundary_owner",
        "_owner_kind",
        "_fault_schema",
        "_child_token_authority",
        "_minted_rows",
        "_minted_tokens_by_root",
        "_last_nonweak_token_root",
        "_token",
    )

    def __init__(
        self,
        boundary_owner: "ActionBallFullMdpRevealBoundaryOwner",
        owner_kind: str,
        fault_schema: ActionBallFullMdpRevealBoundaryFaultSchema,
        child_token_authority: ActionBallFullMdpRevealBoundaryChildTokenAuthority,
        *,
        _token: object,
    ) -> None:
        if (
            _token is not _LANE_AUTH_TOKEN
            or owner_kind not in OWNER_ORDER
            or type(child_token_authority)
            is not ActionBallFullMdpRevealBoundaryChildTokenAuthority
            or child_token_authority.owner_kind != owner_kind
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "lane authority must come from the boundary owner"
            )
        self._boundary_owner = boundary_owner
        self._owner_kind = owner_kind
        self._fault_schema = fault_schema
        self._child_token_authority = child_token_authority
        self._minted_rows: weakref.WeakSet[
            ActionBallFullMdpRevealBoundaryDeviceRow
        ] = weakref.WeakSet()
        # A root only needs replay protection while its exact child-owned
        # token can still be presented.  A permanent set turns one root per
        # reveal into an unbounded training-lifetime ledger.  Weak values keep
        # arbitrary-old live-token replay protection without retaining
        # already-retired epochs themselves.
        self._minted_tokens_by_root: weakref.WeakValueDictionary[
            str, object
        ] = weakref.WeakValueDictionary()
        self._last_nonweak_token_root: Optional[str] = None
        self._token = _token

    @property
    def owner_kind(self) -> str:
        return self._owner_kind

    @property
    def fault_schema(self) -> ActionBallFullMdpRevealBoundaryFaultSchema:
        return self._fault_schema

    @property
    def child_token_authority(
        self,
    ) -> ActionBallFullMdpRevealBoundaryChildTokenAuthority:
        return self._child_token_authority

    def mint_device_row(
        self,
        *,
        prepared_token: object,
        selected_env_ids: Sequence[int],
        pass_mask: torch.Tensor,
        fault_bits: torch.Tensor,
    ) -> ActionBallFullMdpRevealBoundaryDeviceRow:
        """Clone one child preflight row without any host synchronization."""

        owner = self._boundary_owner
        owner._require_operable()
        claim = self._child_token_authority.require_owned_prepared_token(
            prepared_token
        )
        token_root = claim.owner_token_root_sha256
        preview_schema = claim.reveal_final_preview_schema_version
        preview_root = claim.reveal_final_preview_sha256
        selected = _ordered_env_ids(
            selected_env_ids,
            label="selected_env_ids",
            num_envs=owner.num_envs,
        )
        pass_tensor = _require_tensor(
            pass_mask,
            label=f"{self.owner_kind}.pass_mask",
            shape=(owner.num_envs,),
            dtype=torch.bool,
            device=owner.device,
        ).detach().clone()
        fault_tensor = _require_tensor(
            fault_bits,
            label=f"{self.owner_kind}.fault_bits",
            shape=(owner.num_envs,),
            dtype=torch.int64,
            device=owner.device,
        ).detach().clone()
        source_device_version = claim.device_owner_mutation_version
        if source_device_version.device != owner.device:
            raise ActionBallFullMdpRevealBoundaryError(
                f"{self.owner_kind}.device_owner_mutation_version device differs"
            )
        device_version = source_device_version.detach().clone().reshape(1)
        device_root = torch.tensor(
            tuple(bytes.fromhex(token_root)),
            dtype=torch.uint8,
            device=owner.device,
        )
        device_fault_schema_root = torch.tensor(
            tuple(bytes.fromhex(self.fault_schema.schema_sha256)),
            dtype=torch.uint8,
            device=owner.device,
        )
        row = ActionBallFullMdpRevealBoundaryDeviceRow(
            _owner_kind=self.owner_kind,
            _owner_token_root_sha256=token_root,
            _reveal_final_preview_schema_version=preview_schema,
            _reveal_final_preview_sha256=preview_root,
            _selected_env_ids=selected,
            _pass_mask=pass_tensor,
            _fault_bits=fault_tensor,
            _device_owner_mutation_version=device_version,
            _device_owner_token_root=device_root,
            _device_fault_schema_root=device_fault_schema_root,
            _fault_schema=self.fault_schema,
            _prepared_token=prepared_token,
            _prepared_token_claim=claim,
            _child_token_authority=self._child_token_authority,
            _issuer=self,
            _identity=_Identity(),
        )
        with owner._lock:
            owner._require_operable()
            if (
                token_root in self._minted_tokens_by_root
                or token_root == self._last_nonweak_token_root
            ):
                owner._poison(
                    f"{self.owner_kind} boundary token root was already minted"
                )
            try:
                self._minted_tokens_by_root[token_root] = prepared_token
            except TypeError:
                # Some legitimate opaque owner tokens use ``object()`` and
                # cannot be weak-referenced.  Keep only the most recent such
                # root; the child authority remains responsible for accepting
                # only its exact currently retained token.
                self._last_nonweak_token_root = token_root
            self._minted_rows.add(row)
        return row

    def require_owned_device_row(
        self,
        row: ActionBallFullMdpRevealBoundaryDeviceRow,
        *,
        expected_prepared_token: object,
    ) -> ActionBallFullMdpRevealBoundaryDeviceRow:
        """Require the exact row and exact child-retained prepared token."""

        if (
            type(row) is not ActionBallFullMdpRevealBoundaryDeviceRow
            or row._issuer is not self
            or row not in self._minted_rows
            or row._child_token_authority is not self._child_token_authority
            or row._prepared_token is not expected_prepared_token
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                f"{self.owner_kind} boundary row was not minted by this lane"
            )
        claim = self._child_token_authority.require_owned_prepared_token(
            expected_prepared_token
        )
        if (
            row._prepared_token_claim.device_owner_mutation_version
            is not claim.device_owner_mutation_version
            or row._prepared_token_claim.owner_kind != claim.owner_kind
            or row.owner_token_root_sha256 != claim.owner_token_root_sha256
            or row.reveal_final_preview_schema_version
            != claim.reveal_final_preview_schema_version
            or row.reveal_final_preview_sha256
            != claim.reveal_final_preview_sha256
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                f"{self.owner_kind} boundary row/token facts differ"
            )
        return row

    def require_abortable_device_row(
        self,
        row: ActionBallFullMdpRevealBoundaryDeviceRow,
        *,
        expected_prepared_token: object,
        abort_capability: Optional[
            ActionBallFullMdpRevealBoundaryAbortCapability
        ],
    ) -> ActionBallFullMdpRevealBoundaryDeviceRow:
        """Consume one exact pre-transfer permission for child-local abort."""

        owner = self._boundary_owner
        with owner._lock:
            owner._require_operable()
            owned = self.require_owned_device_row(
                row,
                expected_prepared_token=expected_prepared_token,
            )
            active = owner._active_attempt
            if (
                row._identity in owner._consumed_row_identities
                or row._identity in owner._abort_claimed_row_identities
                or (
                    active is not None
                    and any(candidate is row for candidate in active._ordered_rows)
                )
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    f"{self.owner_kind} boundary row is not abortable"
                )
            was_global_aborted = (
                row._identity in owner._aborted_row_identities
            )
            if was_global_aborted:
                owner._require_owned_abort_capability(
                    abort_capability,
                    expected_row=row,
                )
            elif abort_capability is not None:
                raise ActionBallFullMdpRevealBoundaryError(
                    f"{self.owner_kind} local row has no global abort capability"
                )
            owner._abort_claimed_row_identities.add(row._identity)
            return owned


@dataclass(frozen=True, eq=False)
class ActionBallFullMdpRevealBoundaryAttempt:
    """Opaque exclusive lease over four exact boundary-free rows."""

    _reveal_final_preview_schema_version: int
    _reveal_final_preview_sha256: str
    _selected_env_ids: Tuple[int, ...]
    _selected_mask: torch.Tensor = field(repr=False)
    _ordered_rows: Tuple[ActionBallFullMdpRevealBoundaryDeviceRow, ...] = field(
        repr=False
    )
    _d05_pretransfer_authority: ActionBallFullMdpRevealBoundaryD05PreTransferAuthority = field(
        repr=False
    )
    _d05_pretransfer_token: ActionBallFullMdpRevealBoundaryD05PreTransferToken = field(
        repr=False
    )
    _d05_pretransfer_view: ActionBallFullMdpRevealBoundaryD05PreTransferView = field(
        repr=False
    )
    _owner_identity: object = field(repr=False)
    _identity: _Identity = field(repr=False)

    @property
    def reveal_final_preview_schema_version(self) -> int:
        return self._reveal_final_preview_schema_version

    @property
    def reveal_final_preview_sha256(self) -> str:
        return self._reveal_final_preview_sha256

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self._selected_env_ids

    @property
    def ordered_owner_token_roots(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (row.owner_kind, row.owner_token_root_sha256)
            for row in self._ordered_rows
        )


@dataclass(frozen=True, eq=False)
class ActionBallFullMdpRevealBoundaryAbortCapability:
    """Opaque owner-issued proof that one global attempt aborted pre-transfer."""

    _attempt_identity: _Identity = field(repr=False)
    _ordered_rows: Tuple[ActionBallFullMdpRevealBoundaryDeviceRow, ...] = field(
        repr=False
    )
    _owner_identity: object = field(repr=False)
    _auth_token: object = field(repr=False)

    @property
    def ordered_owner_kinds(self) -> Tuple[str, ...]:
        return tuple(row.owner_kind for row in self._ordered_rows)


@dataclass(frozen=True)
class ActionBallFullMdpRevealBoundaryOwnerRow:
    """Exact host row decoded from the one packed transfer."""

    KIND: ClassVar[str] = OWNER_ROW_KIND

    owner_kind: str
    owner_mutation_version: int
    owner_token_root_sha256: str
    fault_schema_sha256: str
    allowed_fault_mask: int
    selected_pass: Tuple[bool, ...]
    selected_fault_bits: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.owner_kind not in OWNER_ORDER:
            raise ActionBallFullMdpRevealBoundaryError(
                "owner row kind differs"
            )
        object.__setattr__(
            self,
            "owner_mutation_version",
            _plain_int(
                self.owner_mutation_version,
                label="owner_mutation_version",
            ),
        )
        object.__setattr__(
            self,
            "owner_token_root_sha256",
            _sha256(
                self.owner_token_root_sha256,
                label="owner_token_root_sha256",
            ),
        )
        object.__setattr__(
            self,
            "fault_schema_sha256",
            _sha256(
                self.fault_schema_sha256,
                label="fault_schema_sha256",
            ),
        )
        allowed_fault_mask = _plain_int(
            self.allowed_fault_mask,
            label="allowed_fault_mask",
            minimum=1,
            maximum=(1 << 63) - 1,
        )
        object.__setattr__(
            self, "allowed_fault_mask", allowed_fault_mask
        )
        passes = tuple(self.selected_pass)
        faults = tuple(self.selected_fault_bits)
        if (
            not passes
            or len(passes) != len(faults)
            or any(type(value) is not bool for value in passes)
            or any(
                type(value) is not int
                or value < 0
                or value & ~allowed_fault_mask
                for value in faults
            )
            or any(
                (passed and fault != 0)
                or ((not passed) and fault == 0)
                for passed, fault in zip(passes, faults)
            )
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "owner row pass/fault values differ"
            )
        object.__setattr__(self, "selected_pass", passes)
        object.__setattr__(self, "selected_fault_bits", faults)

    @property
    def selected_pass_count(self) -> int:
        return sum(self.selected_pass)

    @property
    def selected_fault_count(self) -> int:
        return len(self.selected_pass) - self.selected_pass_count

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": self.KIND,
            "owner_kind": self.owner_kind,
            "owner_mutation_version": self.owner_mutation_version,
            "owner_token_root_sha256": self.owner_token_root_sha256,
            "fault_schema_sha256": self.fault_schema_sha256,
            "allowed_fault_mask": self.allowed_fault_mask,
            "selected_pass": list(self.selected_pass),
            "selected_fault_bits": list(self.selected_fault_bits),
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())


@dataclass(frozen=True)
class ActionBallFullMdpRevealBoundaryReceipt:
    """Owner-issued PASS/CENSOR receipt for one exact packed boundary."""

    schema_version: int
    kind: str
    packet_schema_version: int
    boundary_sequence: int
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    num_envs: int
    selected_env_ids: Tuple[int, ...]
    ordered_owner_kinds: Tuple[str, ...]
    ordered_owner_rows: Tuple[ActionBallFullMdpRevealBoundaryOwnerRow, ...]
    packet_nbytes: int
    packet_sha256: str
    device_type: str
    device_index: Optional[int]
    boundary_transfer_count: int
    transfer_attempt_count_total: int
    transfer_success_count_total: int
    transfer_bytes_total: int
    transfer_elapsed_ns_total: int
    selected_pass_count: int
    selected_fault_count: int
    decision: str
    d05_construction_admissible: bool
    d05_owner_fault_present: bool
    d05_selected_primary_fault: Tuple[int, ...]
    _d05_preview_identity: object = field(repr=False, compare=False)
    _d05_pretransfer_token: Optional[
        ActionBallFullMdpRevealBoundaryD05PreTransferToken
    ] = field(repr=False, compare=False)
    _d05_pretransfer_view: ActionBallFullMdpRevealBoundaryD05PreTransferView = field(
        repr=False, compare=False
    )
    _device_rows: Tuple[ActionBallFullMdpRevealBoundaryDeviceRow, ...] = field(
        repr=False, compare=False
    )
    _owner_identity: object = field(repr=False, compare=False)
    _auth_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.kind != RECEIPT_KIND
            or self.packet_schema_version != PACKET_SCHEMA_VERSION
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt schema differs"
            )
        _plain_int(self.boundary_sequence, label="boundary_sequence", minimum=1)
        _plain_int(
            self.reveal_final_preview_schema_version,
            label="reveal_final_preview_schema_version",
            minimum=1,
        )
        _sha256(
            self.reveal_final_preview_sha256,
            label="reveal_final_preview_sha256",
        )
        count = _plain_int(self.num_envs, label="num_envs", minimum=1)
        selected = _ordered_env_ids(
            self.selected_env_ids,
            label="selected_env_ids",
            num_envs=count,
        )
        kinds = tuple(self.ordered_owner_kinds)
        rows = tuple(self.ordered_owner_rows)
        if (
            kinds != OWNER_ORDER
            or len(rows) != OWNER_COUNT
            or tuple(row.owner_kind for row in rows) != OWNER_ORDER
            or any(len(row.selected_pass) != len(selected) for row in rows)
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt owner order/width differs"
            )
        if self.packet_nbytes != packet_nbytes(count):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt packet width differs"
            )
        _sha256(self.packet_sha256, label="packet_sha256")
        if self.device_type not in ("cpu", "cuda"):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt device type differs"
            )
        if self.device_index is not None:
            _plain_int(self.device_index, label="device_index")
        if self.device_type == "cpu" and self.device_index is not None:
            raise ActionBallFullMdpRevealBoundaryError(
                "CPU boundary receipt must not name a device index"
            )
        if self.boundary_transfer_count != 1:
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt did not use exactly one packed transfer"
            )
        attempt_total = _plain_int(
            self.transfer_attempt_count_total,
            label="transfer_attempt_count_total",
            minimum=1,
        )
        success_total = _plain_int(
            self.transfer_success_count_total,
            label="transfer_success_count_total",
            minimum=1,
        )
        if success_total > attempt_total:
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt transfer totals differ"
            )
        _plain_int(self.transfer_bytes_total, label="transfer_bytes_total", minimum=1)
        _plain_int(
            self.transfer_elapsed_ns_total,
            label="transfer_elapsed_ns_total",
        )
        pass_count = _plain_int(
            self.selected_pass_count,
            label="selected_pass_count",
        )
        fault_count = _plain_int(
            self.selected_fault_count,
            label="selected_fault_count",
        )
        computed_pass_count = sum(
            all(row.selected_pass[index] for row in rows)
            for index in range(len(selected))
        )
        if (
            pass_count + fault_count != len(selected)
            or pass_count != computed_pass_count
            or fault_count != len(selected) - computed_pass_count
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt selected conservation differs"
            )
        d05_faults = tuple(self.d05_selected_primary_fault)
        if (
            type(self.d05_construction_admissible) is not bool
            or type(self.d05_owner_fault_present) is not bool
            or len(d05_faults) != len(selected)
            or any(type(value) is not int or value < 0 for value in d05_faults)
            or self.d05_owner_fault_present != any(d05_faults)
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt D05 typed facts differ"
            )
        if self.decision not in (*DECISIONS, D05_DECISION_CONSTRUCTION_REJECT) or (
            self.decision == DECISION_ACCEPT
            and (
                pass_count != len(selected)
                or fault_count != 0
                or self.d05_owner_fault_present
                or not self.d05_construction_admissible
            )
        ) or (
            self.decision == DECISION_CENSOR
            and fault_count == 0
            and not self.d05_owner_fault_present
        ) or (
            self.decision == D05_DECISION_CONSTRUCTION_REJECT
            and (
                self.d05_owner_fault_present
                or self.d05_construction_admissible
                or fault_count != 0
            )
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt decision differs"
            )
        if self._auth_token is not _RECEIPT_AUTH_TOKEN:
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt was not owner-issued"
            )
        device_rows = tuple(self._device_rows)
        if (
            len(device_rows) != OWNER_COUNT
            or any(
                type(row) is not ActionBallFullMdpRevealBoundaryDeviceRow
                for row in device_rows
            )
            or tuple(row.owner_kind for row in device_rows) != OWNER_ORDER
            or any(
                device_row.owner_token_root_sha256
                != rows[index].owner_token_root_sha256
                or device_row.fault_schema.schema_sha256
                != rows[index].fault_schema_sha256
                or device_row.reveal_final_preview_schema_version
                != self.reveal_final_preview_schema_version
                or device_row.reveal_final_preview_sha256
                != self.reveal_final_preview_sha256
                or device_row.selected_env_ids != selected
                for index, device_row in enumerate(device_rows)
            )
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary receipt private device rows differ"
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "ordered_owner_kinds", kinds)
        object.__setattr__(self, "ordered_owner_rows", rows)
        object.__setattr__(self, "_device_rows", device_rows)

    @property
    def success(self) -> bool:
        return self.decision == DECISION_ACCEPT

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "packet_schema_version": self.packet_schema_version,
            "boundary_sequence": self.boundary_sequence,
            "reveal_final_preview_schema_version": (
                self.reveal_final_preview_schema_version
            ),
            "reveal_final_preview_sha256": self.reveal_final_preview_sha256,
            "num_envs": self.num_envs,
            "selected_env_ids": list(self.selected_env_ids),
            "ordered_owner_kinds": list(self.ordered_owner_kinds),
            "ordered_owner_rows": [
                row.to_mapping() for row in self.ordered_owner_rows
            ],
            "packet_nbytes": self.packet_nbytes,
            "packet_sha256": self.packet_sha256,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "boundary_transfer_count": self.boundary_transfer_count,
            "transfer_attempt_count_total": self.transfer_attempt_count_total,
            "transfer_success_count_total": self.transfer_success_count_total,
            "transfer_bytes_total": self.transfer_bytes_total,
            "transfer_elapsed_ns_total": self.transfer_elapsed_ns_total,
            "selected_pass_count": self.selected_pass_count,
            "selected_fault_count": self.selected_fault_count,
            "decision": self.decision,
            "d05_construction_admissible": self.d05_construction_admissible,
            "d05_owner_fault_present": self.d05_owner_fault_present,
            "d05_selected_primary_fault": list(
                self.d05_selected_primary_fault
            ),
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())


class ActionBallFullMdpRevealBoundaryOwner:
    """Own the sole packed transfer and its exact receipt registry."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        owner_fault_schemas: Sequence[
            ActionBallFullMdpRevealBoundaryFaultSchema
        ],
        child_token_authorities: Sequence[
            ActionBallFullMdpRevealBoundaryChildTokenAuthority
        ],
    ) -> None:
        self._num_envs = _plain_int(num_envs, label="num_envs", minimum=1)
        self._device = torch.device(device)
        if self._device.type not in ("cpu", "cuda") or (
            self._device.type == "cuda" and self._device.index is None
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary device must be CPU or explicitly indexed CUDA"
            )
        if isinstance(owner_fault_schemas, (str, bytes)) or not isinstance(
            owner_fault_schemas, Sequence
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "owner fault schemas must be one ordered sequence"
            )
        fault_schemas = tuple(owner_fault_schemas)
        if (
            len(fault_schemas) != OWNER_COUNT
            or any(
                type(row) is not ActionBallFullMdpRevealBoundaryFaultSchema
                for row in fault_schemas
            )
            or tuple(row.owner_kind for row in fault_schemas) != OWNER_ORDER
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "owner fault schema order differs"
            )
        if isinstance(child_token_authorities, (str, bytes)) or not isinstance(
            child_token_authorities, Sequence
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "child token authorities must be one ordered sequence"
            )
        token_authorities = tuple(child_token_authorities)
        if (
            len(token_authorities) != OWNER_COUNT
            or any(
                type(authority)
                is not ActionBallFullMdpRevealBoundaryChildTokenAuthority
                for authority in token_authorities
            )
            or tuple(authority.owner_kind for authority in token_authorities)
            != OWNER_ORDER
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "child token authority order differs"
            )
        self._owner_fault_schemas = fault_schemas
        self._child_token_authorities = token_authorities
        self._identity = object()
        self._lock = threading.RLock()
        self._poisoned = False
        self._active_attempt: Optional[
            ActionBallFullMdpRevealBoundaryAttempt
        ] = None
        self._consumed_attempt_identities: weakref.WeakSet[_Identity] = (
            weakref.WeakSet()
        )
        self._consumed_row_identities: weakref.WeakSet[_Identity] = (
            weakref.WeakSet()
        )
        self._aborted_row_identities: weakref.WeakSet[_Identity] = (
            weakref.WeakSet()
        )
        self._abort_claimed_row_identities: weakref.WeakSet[_Identity] = (
            weakref.WeakSet()
        )
        self._abort_capability_registry: weakref.WeakSet[
            ActionBallFullMdpRevealBoundaryAbortCapability
        ] = weakref.WeakSet()
        self._receipt_registry: weakref.WeakValueDictionary[
            str, ActionBallFullMdpRevealBoundaryReceipt
        ] = weakref.WeakValueDictionary()
        self._boundary_sequence = 0
        self._transfer_attempt_count = 0
        self._transfer_success_count = 0
        self._transfer_bytes_total = 0
        self._transfer_elapsed_ns_total = 0
        self._minted_receipt_count = 0
        self._lane_authorities = tuple(
            ActionBallFullMdpRevealBoundaryLaneAuthority(
                self,
                owner_kind,
                fault_schemas[index],
                token_authorities[index],
                _token=_LANE_AUTH_TOKEN,
            )
            for index, owner_kind in enumerate(OWNER_ORDER)
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def poisoned(self) -> bool:
        with self._lock:
            return self._poisoned

    @property
    def boundary_sequence(self) -> int:
        with self._lock:
            return self._boundary_sequence

    @property
    def transfer_count(self) -> int:
        with self._lock:
            return self._transfer_success_count

    @property
    def transfer_attempt_count(self) -> int:
        with self._lock:
            return self._transfer_attempt_count

    @property
    def transfer_success_count(self) -> int:
        with self._lock:
            return self._transfer_success_count

    @property
    def transfer_bytes_total(self) -> int:
        with self._lock:
            return self._transfer_bytes_total

    @property
    def transfer_elapsed_ns_total(self) -> int:
        with self._lock:
            return self._transfer_elapsed_ns_total

    @property
    def minted_receipt_count(self) -> int:
        with self._lock:
            return self._minted_receipt_count

    @property
    def lane_authorities(
        self,
    ) -> Tuple[ActionBallFullMdpRevealBoundaryLaneAuthority, ...]:
        return self._lane_authorities

    @property
    def owner_fault_schemas(
        self,
    ) -> Tuple[ActionBallFullMdpRevealBoundaryFaultSchema, ...]:
        return self._owner_fault_schemas

    @property
    def child_token_authorities(
        self,
    ) -> Tuple[ActionBallFullMdpRevealBoundaryChildTokenAuthority, ...]:
        return self._child_token_authorities

    def lane_authority(
        self, owner_kind: str
    ) -> ActionBallFullMdpRevealBoundaryLaneAuthority:
        if owner_kind not in OWNER_ORDER:
            raise ActionBallFullMdpRevealBoundaryError(
                "unknown all-owner reveal lane kind"
            )
        return self._lane_authorities[OWNER_ORDER.index(owner_kind)]

    def require_owned_r05_reveal_boundary(
        self,
        preview: object,
        receipt: object,
        *,
        owner_view: object,
    ) -> object:
        """Tombstone the retired post-transfer injection ABI forever."""

        del preview, receipt, owner_view
        raise ActionBallFullMdpRevealBoundaryD05AdapterHold(
            "post-transfer Device-R05 fact injection is retired; use the "
            "owner-issued pre-transfer token in prepare_boundary_attempt"
        )

    def project_owned_r05_reveal_boundary(
        self,
        preview: object,
        receipt: object,
    ) -> object:
        """Project D05 facts already decoded by the sole packed transfer."""

        with self._lock:
            self._require_operable()
            if type(receipt) is not ActionBallFullMdpRevealBoundaryReceipt:
                raise ActionBallFullMdpRevealBoundaryError(
                    "Device-R05 boundary receipt type differs"
                )
            registered = self._receipt_registry.get(receipt.canonical_sha256)
            if (
                registered is not receipt
                or receipt._owner_identity is not self._identity
                or receipt._d05_preview_identity is not preview
                or receipt._d05_pretransfer_token is None
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "Device-R05 boundary preview/token/receipt differs"
                )
            view = receipt._d05_pretransfer_view
            selected_fault = torch.bitwise_or(
                view.producer_fault,
                view.counter_overflow_fault.to(torch.int64)
                * D05_COUNTER_OVERFLOW_FAULT_BIT,
            )
            try:
                import importlib

                module = importlib.import_module(
                    "action_ball_continuous_runtime_transaction_device"
                )
            except ModuleNotFoundError:
                module = importlib.import_module(
                    "whole_body_tracking.action_ball_continuous_runtime_transaction_device"
                )
            decision = {
                DECISION_ACCEPT: module.DECISION_ACCEPT,
                DECISION_CENSOR: module.DECISION_CENSOR,
                D05_DECISION_CONSTRUCTION_REJECT: (
                    module.DECISION_CONSTRUCTION_REJECT
                ),
            }[receipt.decision]
            return module.DeviceRevealBoundaryProjection(
                preview_identity=preview,
                construction_admissible=(
                    receipt.d05_construction_admissible
                ),
                owner_fault_present=(
                    receipt.d05_owner_fault_present
                    or receipt.selected_fault_count > 0
                ),
                decision=decision,
                primary_fault=selected_fault.clone(),
                transfer_sequence=receipt.boundary_sequence,
            )

    def _require_operable(self) -> None:
        with self._lock:
            if self._poisoned:
                raise ActionBallFullMdpRevealBoundaryPoisonedError(
                    "all-owner reveal boundary is poisoned"
                )

    def _poison(self, message: str) -> None:
        self._poisoned = True
        self._active_attempt = None
        raise ActionBallFullMdpRevealBoundaryPoisonedError(message)

    def _poison_from_exception(
        self, message: str, exc: BaseException
    ) -> None:
        self._poisoned = True
        self._active_attempt = None
        raise ActionBallFullMdpRevealBoundaryPoisonedError(message) from exc

    def prepare_boundary_attempt(
        self,
        *,
        reveal_final_preview_schema_version: int,
        reveal_final_preview_sha256: str,
        selected_env_ids: Sequence[int],
        ordered_owner_rows: Sequence[
            ActionBallFullMdpRevealBoundaryDeviceRow
        ],
        d05_pretransfer_authority: Optional[
            ActionBallFullMdpRevealBoundaryD05PreTransferAuthority
        ] = None,
        d05_pretransfer_token: Optional[
            ActionBallFullMdpRevealBoundaryD05PreTransferToken
        ] = None,
    ) -> ActionBallFullMdpRevealBoundaryAttempt:
        """Install one exclusive private lease without crossing to host."""

        with self._lock:
            self._require_operable()
            if self._active_attempt is not None:
                self._poison("another all-owner boundary attempt is active")
            # A public production attempt always requires the exact D05 owner
            # capability.  Missing authority is not a diagnostic row and may
            # not create an active lease that could later mint ACCEPT.
            validator = getattr(
                d05_pretransfer_authority,
                "require_owned_r05_pretransfer_boundary_input",
                None,
            )
            if (
                not callable(validator)
                or type(d05_pretransfer_token)
                is not ActionBallFullMdpRevealBoundaryD05PreTransferToken
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "Device-R05 pre-transfer production authority/token is required"
                )
            preview_schema = _plain_int(
                reveal_final_preview_schema_version,
                label="reveal_final_preview_schema_version",
                minimum=1,
            )
            preview_root = _sha256(
                reveal_final_preview_sha256,
                label="reveal_final_preview_sha256",
            )
            selected = _ordered_env_ids(
                selected_env_ids,
                label="selected_env_ids",
                num_envs=self.num_envs,
            )
            if isinstance(ordered_owner_rows, (str, bytes)) or not isinstance(
                ordered_owner_rows, Sequence
            ):
                self._poison("all-owner boundary rows are not a sequence")
            rows = tuple(ordered_owner_rows)
            if (
                len(rows) != OWNER_COUNT
                or tuple(
                    row.owner_kind
                    if type(row) is ActionBallFullMdpRevealBoundaryDeviceRow
                    else None
                    for row in rows
                )
                != OWNER_ORDER
                or len({id(row) for row in rows}) != OWNER_COUNT
            ):
                self._poison(
                    "all-owner boundary rows are omitted, duplicated, or reordered"
                )
            for index, row in enumerate(rows):
                authority = self._lane_authorities[index]
                try:
                    authority.require_owned_device_row(
                        row,
                        expected_prepared_token=row._prepared_token,
                    )
                except BaseException as exc:
                    self._poison_from_exception(
                        "all-owner boundary row authority/token differs", exc
                    )
                if (
                    row._identity in self._consumed_row_identities
                    or row._identity in self._aborted_row_identities
                    or row._identity in self._abort_claimed_row_identities
                    or row.reveal_final_preview_schema_version != preview_schema
                    or row.reveal_final_preview_sha256 != preview_root
                    or row.selected_env_ids != selected
                    or row._pass_mask.device != self.device
                    or row._fault_bits.device != self.device
                ):
                    self._poison(
                        "all-owner boundary row authority/preview/selection differs"
                    )
            selected_mask = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            selected_index = torch.tensor(
                selected, dtype=torch.int64, device=self.device
            )
            selected_mask.index_fill_(0, selected_index, True)
            try:
                source_view = validator(
                    d05_pretransfer_token,
                    device=self.device,
                    num_envs=self.num_envs,
                )
                if type(source_view) is not ActionBallFullMdpRevealBoundaryD05PreTransferView:
                    raise ActionBallFullMdpRevealBoundaryError(
                        "Device-R05 pre-transfer view type differs"
                    )
                k = len(selected)
                raw_selected_index = _require_tensor(
                    source_view.selected_env_index,
                    label="d05.selected_env_index",
                    shape=(k,), dtype=torch.int64, device=self.device,
                )
                raw_selected_mask = _require_tensor(
                    source_view.selected_mask,
                    label="d05.selected_mask",
                    shape=(self.num_envs,), dtype=torch.bool, device=self.device,
                )
                d05_view = ActionBallFullMdpRevealBoundaryD05PreTransferView(
                    preview_identity=source_view.preview_identity,
                    # The packet layout uses the boundary-owned canonical
                    # selection.  Exact D05-vs-boundary drift is carried as a
                    # typed producer fault below; indexing the raw permutation
                    # here would make its own fault attribution ambiguous.
                    selected_env_index=selected_index.clone(),
                    selected_mask=selected_mask.clone(),
                    construction_admissible=_require_tensor(
                        source_view.construction_admissible,
                        label="d05.construction_admissible",
                        shape=(k,), dtype=torch.bool, device=self.device,
                    ).clone(),
                    producer_fault=_require_tensor(
                        source_view.producer_fault,
                        label="d05.producer_fault",
                        shape=(k,), dtype=torch.int64, device=self.device,
                    ).clone().bitwise_or(
                        torch.logical_or(
                            raw_selected_index.ne(selected_index),
                            torch.any(
                                raw_selected_mask.ne(selected_mask)
                            ).expand(k),
                        ).to(torch.int64)
                        * D05_SELECTION_MISMATCH_FAULT_BIT
                    ),
                    counter_overflow_fault=_require_tensor(
                        source_view.counter_overflow_fault,
                        label="d05.counter_overflow_fault",
                        shape=(k,), dtype=torch.bool, device=self.device,
                    ).clone(),
                )
                d05_token = d05_pretransfer_token
            except BaseException as exc:
                self._poison_from_exception(
                    "Device-R05 pre-transfer validation failed", exc
                )

            # Selection belongs to two independent writers.  The exact
            # comparison above stays device-side and its typed fault enters the
            # same packed transfer; it is never an ordinary construction miss.
            attempt = ActionBallFullMdpRevealBoundaryAttempt(
                _reveal_final_preview_schema_version=preview_schema,
                _reveal_final_preview_sha256=preview_root,
                _selected_env_ids=selected,
                _selected_mask=selected_mask,
                _ordered_rows=rows,
                _d05_pretransfer_authority=d05_pretransfer_authority,
                _d05_pretransfer_token=d05_token,
                _d05_pretransfer_view=d05_view,
                _owner_identity=self._identity,
                _identity=_Identity(),
            )
            self._active_attempt = attempt
            return attempt

    def abort_boundary_attempt(
        self,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
    ) -> ActionBallFullMdpRevealBoundaryAbortCapability:
        """Retire one exact active attempt before any packed transfer."""

        with self._lock:
            self._require_operable()
            if (
                type(attempt) is not ActionBallFullMdpRevealBoundaryAttempt
                or attempt._owner_identity is not self._identity
                or attempt._identity in self._consumed_attempt_identities
                or attempt is not self._active_attempt
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "all-owner boundary attempt is not abortable"
                )
            capability = ActionBallFullMdpRevealBoundaryAbortCapability(
                _attempt_identity=attempt._identity,
                _ordered_rows=attempt._ordered_rows,
                _owner_identity=self._identity,
                _auth_token=_ABORT_AUTH_TOKEN,
            )
            self._abort_capability_registry.add(capability)
            self._consumed_attempt_identities.add(attempt._identity)
            for row in attempt._ordered_rows:
                self._aborted_row_identities.add(row._identity)
            self._active_attempt = None
            return capability

    def _require_owned_abort_capability(
        self,
        capability: Optional[
            ActionBallFullMdpRevealBoundaryAbortCapability
        ],
        *,
        expected_row: ActionBallFullMdpRevealBoundaryDeviceRow,
    ) -> ActionBallFullMdpRevealBoundaryAbortCapability:
        if (
            type(capability)
            is not ActionBallFullMdpRevealBoundaryAbortCapability
            or capability not in self._abort_capability_registry
            or capability._owner_identity is not self._identity
            or capability._auth_token is not _ABORT_AUTH_TOKEN
            or not any(
                candidate is expected_row
                for candidate in capability._ordered_rows
            )
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "global boundary abort capability differs"
            )
        return capability

    def transfer_once(
        self,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
    ) -> ActionBallFullMdpRevealBoundaryReceipt:
        """Perform the sole packed host transfer for one active attempt."""

        return self._transfer_once(attempt, packet_mutator=None)

    def _transfer_once_with_packet_mutator_for_test(
        self,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
        packet_mutator: Callable[[torch.Tensor], torch.Tensor],
    ) -> ActionBallFullMdpRevealBoundaryReceipt:
        """Test-only byte mutation seam; production callers use ``transfer_once``."""

        if not callable(packet_mutator):
            raise ActionBallFullMdpRevealBoundaryError(
                "test packet mutator must be callable"
            )
        return self._transfer_once(attempt, packet_mutator=packet_mutator)

    def _build_packet(
        self,
        *,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
        boundary_sequence: int,
        expected_nbytes: int,
    ) -> torch.Tensor:
        """Build the sole packet on-device without any host read."""

        header = torch.tensor(
            (
                PACKET_SCHEMA_VERSION,
                boundary_sequence,
                self.num_envs,
                OWNER_COUNT,
                len(attempt.selected_env_ids),
                attempt.reveal_final_preview_schema_version,
                FAULT_WORD_NBYTES,
                expected_nbytes,
            ),
            dtype=torch.int64,
            device=self.device,
        )
        preview_root = torch.tensor(
            tuple(bytes.fromhex(attempt.reveal_final_preview_sha256)),
            dtype=torch.uint8,
            device=self.device,
        )
        owner_versions = torch.stack(
            tuple(
                row._device_owner_mutation_version
                for row in attempt._ordered_rows
            ),
            dim=0,
        ).reshape(-1)
        owner_version_bytes = _encode_int64_little_endian(owner_versions)
        row_integrity_mask = _encode_int64_little_endian(
            torch.stack(
                tuple(
                    _device_row_integrity_words(
                        owner_index=index,
                        device_owner_mutation_version=(
                            row._prepared_token_claim.device_owner_mutation_version
                        ),
                        pass_mask=row._pass_mask,
                        fault_bits=row._fault_bits,
                    )
                    for index, row in enumerate(attempt._ordered_rows)
                ),
                dim=0,
            )
        )
        owner_roots = torch.stack(
            tuple(
                row._device_owner_token_root for row in attempt._ordered_rows
            ),
            dim=0,
        ).reshape(-1).bitwise_xor(
            torch.stack(
                tuple(
                    row._device_fault_schema_root
                    for row in attempt._ordered_rows
                ),
                dim=0,
            ).reshape(-1)
        ).bitwise_xor(row_integrity_mask)
        pass_masks = torch.stack(
            tuple(row._pass_mask for row in attempt._ordered_rows), dim=0
        ).to(dtype=torch.uint8).reshape(-1)
        fault_bits = torch.stack(
            tuple(row._fault_bits for row in attempt._ordered_rows), dim=0
        ).contiguous()
        d05 = attempt._d05_pretransfer_view
        selected_rank = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        selected_rank.index_copy_(
            0,
            d05.selected_env_index,
            torch.arange(
                d05.selected_env_index.shape[0],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        d05_status = d05.selected_mask.to(torch.uint8).clone()
        d05_status.index_add_(
            0,
            d05.selected_env_index,
            d05.construction_admissible.to(torch.uint8) * 2,
        )
        d05_producer_fault = torch.zeros(
            (self.num_envs,), dtype=torch.int64, device=self.device
        )
        d05_producer_fault.index_copy_(
            0, d05.selected_env_index, d05.producer_fault
        )
        d05_counter_overflow = torch.zeros(
            (self.num_envs,), dtype=torch.uint8, device=self.device
        )
        d05_counter_overflow.index_copy_(
            0,
            d05.selected_env_index,
            d05.counter_overflow_fault.to(torch.uint8),
        )
        return torch.cat(
            (
                _encode_int64_little_endian(header),
                preview_root,
                attempt._selected_mask.to(dtype=torch.uint8),
                owner_version_bytes,
                owner_roots,
                pass_masks,
                _encode_int64_little_endian(fault_bits),
                _encode_int64_little_endian(selected_rank),
                d05_status,
                _encode_int64_little_endian(d05_producer_fault),
                d05_counter_overflow,
            ),
            dim=0,
        )

    def _transfer_once(
        self,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
        *,
        packet_mutator: _PacketMutator,
    ) -> ActionBallFullMdpRevealBoundaryReceipt:
        with self._lock:
            self._require_operable()
            if (
                type(attempt) is not ActionBallFullMdpRevealBoundaryAttempt
                or attempt._owner_identity is not self._identity
                or attempt._identity in self._consumed_attempt_identities
                or attempt is not self._active_attempt
            ):
                self._poison(
                    "all-owner reveal boundary attempt is stale, foreign, or consumed"
                )
            # Consume before any packet construction or synchronization.  A
            # failed transfer is never retryable.
            self._consumed_attempt_identities.add(attempt._identity)
            for row in attempt._ordered_rows:
                self._consumed_row_identities.add(row._identity)

            next_sequence = self._boundary_sequence + 1
            expected_nbytes = packet_nbytes(self.num_envs)
            try:
                packet = self._build_packet(
                    attempt=attempt,
                    boundary_sequence=next_sequence,
                    expected_nbytes=expected_nbytes,
                )
            except BaseException as exc:
                self._poison_from_exception(
                    "packed all-owner boundary construction failed", exc
                )
            if packet_mutator is not None:
                try:
                    packet = packet_mutator(packet)
                except BaseException as exc:
                    self._poison_from_exception(
                        "test packet mutation failed before the packed transfer",
                        exc,
                    )
            if not isinstance(packet, torch.Tensor):
                self._poison("packed boundary mutation did not return a tensor")

            transfer_started_ns = time.perf_counter_ns()
            self._transfer_attempt_count += 1
            try:
                host_packet = packet.to(device="cpu", non_blocking=False)
                host_bytes = host_packet.contiguous().numpy().tobytes()
            except BaseException as exc:
                self._poison_from_exception(
                    "packed all-owner boundary transfer failed", exc
                )
            transfer_elapsed_ns = time.perf_counter_ns() - transfer_started_ns
            self._transfer_success_count += 1
            self._transfer_bytes_total += len(host_bytes)
            self._transfer_elapsed_ns_total += transfer_elapsed_ns
            self._boundary_sequence = next_sequence

            try:
                decoded = self._decode_packet(
                    host_bytes=host_bytes,
                    attempt=attempt,
                    expected_sequence=next_sequence,
                    expected_nbytes=expected_nbytes,
                )
            except BaseException as exc:
                self._poison_from_exception(str(exc), exc)

            try:
                return self._mint_receipt(
                    attempt=attempt,
                    boundary_sequence=next_sequence,
                    host_bytes=host_bytes,
                    decoded=decoded,
                )
            except BaseException as exc:
                self._poison_from_exception(
                    "packed all-owner boundary receipt mint failed", exc
                )

    def _mint_receipt(
        self,
        *,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
        boundary_sequence: int,
        host_bytes: bytes,
        decoded: Mapping[str, object],
    ) -> ActionBallFullMdpRevealBoundaryReceipt:
        receipt = ActionBallFullMdpRevealBoundaryReceipt(
            schema_version=SCHEMA_VERSION,
            kind=RECEIPT_KIND,
            packet_schema_version=PACKET_SCHEMA_VERSION,
            boundary_sequence=boundary_sequence,
            reveal_final_preview_schema_version=(
                attempt.reveal_final_preview_schema_version
            ),
            reveal_final_preview_sha256=attempt.reveal_final_preview_sha256,
            num_envs=self.num_envs,
            selected_env_ids=attempt.selected_env_ids,
            ordered_owner_kinds=OWNER_ORDER,
            ordered_owner_rows=decoded["owner_rows"],
            packet_nbytes=len(host_bytes),
            packet_sha256=hashlib.sha256(host_bytes).hexdigest(),
            device_type=self.device.type,
            device_index=self.device.index,
            boundary_transfer_count=1,
            transfer_attempt_count_total=self._transfer_attempt_count,
            transfer_success_count_total=self._transfer_success_count,
            transfer_bytes_total=self._transfer_bytes_total,
            transfer_elapsed_ns_total=self._transfer_elapsed_ns_total,
            selected_pass_count=decoded["selected_pass_count"],
            selected_fault_count=decoded["selected_fault_count"],
            decision=decoded["decision"],
            d05_construction_admissible=(
                decoded["d05_construction_admissible"]
            ),
            d05_owner_fault_present=decoded["d05_owner_fault_present"],
            d05_selected_primary_fault=decoded[
                "d05_selected_primary_fault"
            ],
            _d05_preview_identity=(
                attempt._d05_pretransfer_view.preview_identity
            ),
            _d05_pretransfer_token=attempt._d05_pretransfer_token,
            _d05_pretransfer_view=attempt._d05_pretransfer_view,
            _device_rows=attempt._ordered_rows,
            _owner_identity=self._identity,
            _auth_token=_RECEIPT_AUTH_TOKEN,
        )
        self._minted_receipt_count += 1
        self._receipt_registry[receipt.canonical_sha256] = receipt
        self._active_attempt = None
        return receipt

    def _decode_packet(
        self,
        *,
        host_bytes: bytes,
        attempt: ActionBallFullMdpRevealBoundaryAttempt,
        expected_sequence: int,
        expected_nbytes: int,
    ) -> dict[str, object]:
        if len(host_bytes) != expected_nbytes:
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary byte width differs"
            )
        cursor = 0

        def read_int64() -> int:
            nonlocal cursor
            stop = cursor + INT64_NBYTES
            if stop > len(host_bytes):
                raise ActionBallFullMdpRevealBoundaryError(
                    "packed all-owner boundary is truncated"
                )
            value = int.from_bytes(
                host_bytes[cursor:stop], byteorder="little", signed=True
            )
            cursor = stop
            return value

        header = tuple(read_int64() for _ in range(HEADER_INT64_COUNT))
        if header != (
            PACKET_SCHEMA_VERSION,
            expected_sequence,
            self.num_envs,
            OWNER_COUNT,
            len(attempt.selected_env_ids),
            attempt.reveal_final_preview_schema_version,
            FAULT_WORD_NBYTES,
            expected_nbytes,
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary header/version differs"
            )
        preview_stop = cursor + TOKEN_NBYTES
        preview_bytes = host_bytes[cursor:preview_stop]
        cursor = preview_stop
        if preview_bytes.hex() != attempt.reveal_final_preview_sha256:
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary preview root differs"
            )
        selected_stop = cursor + self.num_envs
        selected_bytes = host_bytes[cursor:selected_stop]
        cursor = selected_stop
        if any(value not in (0, 1) for value in selected_bytes):
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary selected mask is malformed"
            )
        selected_env_ids = tuple(
            index for index, value in enumerate(selected_bytes) if value == 1
        )
        if selected_env_ids != attempt.selected_env_ids:
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary selected mask differs"
            )

        versions = tuple(read_int64() for _ in range(OWNER_COUNT))
        if any(value < 0 for value in versions):
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary owner version is malformed"
            )
        masked_roots = []
        for _ in range(OWNER_COUNT):
            stop = cursor + TOKEN_NBYTES
            masked_roots.append(host_bytes[cursor:stop])
            cursor = stop

        pass_rows = []
        for _ in range(OWNER_COUNT):
            stop = cursor + self.num_envs
            values = host_bytes[cursor:stop]
            cursor = stop
            if any(value not in (0, 1) for value in values):
                raise ActionBallFullMdpRevealBoundaryError(
                    "packed all-owner boundary pass mask is malformed"
                )
            pass_rows.append(tuple(value == 1 for value in values))
        fault_rows = tuple(
            tuple(read_int64() for _ in range(self.num_envs))
            for _ in range(OWNER_COUNT)
        )
        d05_selected_rank = tuple(read_int64() for _ in range(self.num_envs))
        stop = cursor + self.num_envs
        d05_status_bytes = host_bytes[cursor:stop]
        cursor = stop
        d05_producer_fault = tuple(
            read_int64() for _ in range(self.num_envs)
        )
        stop = cursor + self.num_envs
        d05_overflow_bytes = host_bytes[cursor:stop]
        cursor = stop
        if cursor != len(host_bytes):
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary trailing bytes differ"
            )

        selected_set = set(attempt.selected_env_ids)
        expected_rank = [-1] * self.num_envs
        for rank, env_id in enumerate(attempt.selected_env_ids):
            expected_rank[env_id] = rank
        if (
            d05_selected_rank != tuple(expected_rank)
            or any(value not in (0, 1, 3) for value in d05_status_bytes)
            or any(value not in (0, 1) for value in d05_overflow_bytes)
            or any(
                value < 0 or value >= D05_COUNTER_OVERFLOW_FAULT_BIT
                for value in d05_producer_fault
            )
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "packed Device-R05 rank/admissibility/fault row is malformed"
            )
        for env_id in range(self.num_envs):
            if env_id not in selected_set and (
                d05_status_bytes[env_id]
                or d05_producer_fault[env_id]
                or d05_overflow_bytes[env_id]
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "packed Device-R05 selected-outside row differs"
                )
        for owner_index in range(OWNER_COUNT):
            fault_schema = attempt._ordered_rows[owner_index].fault_schema
            for env_id in range(self.num_envs):
                passed = pass_rows[owner_index][env_id]
                fault = fault_rows[owner_index][env_id]
                if fault < 0 or fault & ~fault_schema.allowed_fault_mask:
                    raise ActionBallFullMdpRevealBoundaryError(
                        "packed all-owner boundary fault bits are malformed or unknown"
                    )
                if env_id not in selected_set:
                    if passed or fault != 0:
                        raise ActionBallFullMdpRevealBoundaryError(
                            "packed all-owner boundary selected-outside row differs"
                        )
                elif not (
                    (passed and fault == 0) or ((not passed) and fault != 0)
                ):
                    raise ActionBallFullMdpRevealBoundaryError(
                        "packed all-owner boundary pass/fault relation differs"
                    )

        roots = []
        for owner_index in range(OWNER_COUNT):
            integrity_mask = _host_row_integrity_mask(
                owner_index=owner_index,
                owner_mutation_version=versions[owner_index],
                pass_mask=pass_rows[owner_index],
                fault_bits=fault_rows[owner_index],
            )
            fault_schema_root = bytes.fromhex(
                attempt._ordered_rows[
                    owner_index
                ].fault_schema.schema_sha256
            )
            roots.append(
                bytes(
                    value ^ integrity ^ schema
                    for value, integrity, schema in zip(
                        masked_roots[owner_index],
                        integrity_mask,
                        fault_schema_root,
                    )
                ).hex()
            )
        expected_roots = tuple(
            row.owner_token_root_sha256 for row in attempt._ordered_rows
        )
        if tuple(roots) != expected_roots:
            raise ActionBallFullMdpRevealBoundaryError(
                "packed all-owner boundary owner root differs from row integrity"
            )

        owner_rows = tuple(
            ActionBallFullMdpRevealBoundaryOwnerRow(
                owner_kind=OWNER_ORDER[index],
                owner_mutation_version=versions[index],
                owner_token_root_sha256=roots[index],
                fault_schema_sha256=(
                    attempt._ordered_rows[index].fault_schema.schema_sha256
                ),
                allowed_fault_mask=(
                    attempt._ordered_rows[index].fault_schema.allowed_fault_mask
                ),
                selected_pass=tuple(
                    pass_rows[index][env_id]
                    for env_id in attempt.selected_env_ids
                ),
                selected_fault_bits=tuple(
                    fault_rows[index][env_id]
                    for env_id in attempt.selected_env_ids
                ),
            )
            for index in range(OWNER_COUNT)
        )
        selected_pass_count = sum(
            all(pass_rows[owner][env_id] for owner in range(OWNER_COUNT))
            for env_id in attempt.selected_env_ids
        )
        selected_fault_count = (
            len(attempt.selected_env_ids) - selected_pass_count
        )
        d05_selected_primary_fault = tuple(
            d05_producer_fault[env_id]
            | (
                D05_COUNTER_OVERFLOW_FAULT_BIT
                if d05_overflow_bytes[env_id]
                else 0
            )
            for env_id in attempt.selected_env_ids
        )
        d05_owner_fault_present = any(d05_selected_primary_fault)
        d05_construction_admissible = all(
            d05_status_bytes[env_id] == 3
            for env_id in attempt.selected_env_ids
        )
        if selected_fault_count or d05_owner_fault_present:
            decision = DECISION_CENSOR
        elif not d05_construction_admissible:
            decision = D05_DECISION_CONSTRUCTION_REJECT
        else:
            decision = DECISION_ACCEPT
        return {
            "owner_rows": owner_rows,
            "selected_pass_count": selected_pass_count,
            "selected_fault_count": selected_fault_count,
            "decision": decision,
            "d05_construction_admissible": d05_construction_admissible,
            "d05_owner_fault_present": d05_owner_fault_present,
            "d05_selected_primary_fault": d05_selected_primary_fault,
        }

    def require_owned_receipt(
        self,
        receipt: ActionBallFullMdpRevealBoundaryReceipt,
        *,
        expected_reveal_final_preview_schema_version: int,
        expected_reveal_final_preview_sha256: str,
        expected_selected_env_ids: Sequence[int],
        expected_packet_sha256: Optional[str] = None,
        expected_decision: Optional[str] = None,
    ) -> ActionBallFullMdpRevealBoundaryReceipt:
        """Require the exact registered receipt and its preview/env binding."""

        with self._lock:
            self._require_operable()
            if type(receipt) is not ActionBallFullMdpRevealBoundaryReceipt:
                raise ActionBallFullMdpRevealBoundaryError(
                    "boundary receipt type differs"
                )
            preview_schema = _plain_int(
                expected_reveal_final_preview_schema_version,
                label="expected_reveal_final_preview_schema_version",
                minimum=1,
            )
            preview_root = _sha256(
                expected_reveal_final_preview_sha256,
                label="expected_reveal_final_preview_sha256",
            )
            selected = _ordered_env_ids(
                expected_selected_env_ids,
                label="expected_selected_env_ids",
                num_envs=self.num_envs,
            )
            registered = self._receipt_registry.get(receipt.canonical_sha256)
            if (
                registered is not receipt
                or receipt._owner_identity is not self._identity
                or receipt._auth_token is not _RECEIPT_AUTH_TOKEN
                or receipt.reveal_final_preview_schema_version
                != preview_schema
                or receipt.reveal_final_preview_sha256 != preview_root
                or receipt.selected_env_ids != selected
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "boundary receipt owner/preview/selection differs"
                )
            if expected_packet_sha256 is not None and receipt.packet_sha256 != _sha256(
                expected_packet_sha256, label="expected_packet_sha256"
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "boundary receipt packet root differs"
                )
            if expected_decision is not None and (
                expected_decision
                not in (*DECISIONS, D05_DECISION_CONSTRUCTION_REJECT)
                or receipt.decision != expected_decision
            ):
                raise ActionBallFullMdpRevealBoundaryError(
                    "boundary receipt decision differs"
                )
            return receipt

    def require_owned_owner_row(
        self,
        receipt: ActionBallFullMdpRevealBoundaryReceipt,
        *,
        owner_kind: str,
        expected_device_row: ActionBallFullMdpRevealBoundaryDeviceRow,
        expected_prepared_token: object,
        expected_fault_schema_sha256: str,
        expected_reveal_final_preview_schema_version: int,
        expected_reveal_final_preview_sha256: str,
        expected_selected_env_ids: Sequence[int],
        expected_packet_sha256: str,
        expected_decision: str,
    ) -> ActionBallFullMdpRevealBoundaryOwnerRow:
        """Return one exact child row only from this owner's real receipt."""

        if owner_kind not in OWNER_ORDER:
            raise ActionBallFullMdpRevealBoundaryError(
                "unknown all-owner reveal row kind"
            )
        expected_fault_schema = _sha256(
            expected_fault_schema_sha256,
            label="expected_fault_schema_sha256",
        )
        if expected_decision not in (*DECISIONS, D05_DECISION_CONSTRUCTION_REJECT):
            raise ActionBallFullMdpRevealBoundaryError(
                "expected boundary decision differs"
            )
        owned = self.require_owned_receipt(
            receipt,
            expected_reveal_final_preview_schema_version=(
                expected_reveal_final_preview_schema_version
            ),
            expected_reveal_final_preview_sha256=(
                expected_reveal_final_preview_sha256
            ),
            expected_selected_env_ids=expected_selected_env_ids,
            expected_packet_sha256=expected_packet_sha256,
            expected_decision=expected_decision,
        )
        owner_index = OWNER_ORDER.index(owner_kind)
        device_row = self._lane_authorities[
            owner_index
        ].require_owned_device_row(
            expected_device_row,
            expected_prepared_token=expected_prepared_token,
        )
        token_claim = self._child_token_authorities[
            owner_index
        ].require_owned_prepared_token(expected_prepared_token)
        row = owned.ordered_owner_rows[owner_index]
        if (
            owned._device_rows[owner_index] is not device_row
            or row.owner_token_root_sha256
            != token_claim.owner_token_root_sha256
            or row.fault_schema_sha256 != expected_fault_schema
            or expected_fault_schema
            != self._owner_fault_schemas[owner_index].schema_sha256
        ):
            raise ActionBallFullMdpRevealBoundaryError(
                "boundary owner row identity/version/root/schema differs"
            )
        return row


__all__ = [
    "SCHEMA_VERSION",
    "PACKET_SCHEMA_VERSION",
    "PACKET_KIND",
    "RECEIPT_KIND",
    "OWNER_ROW_KIND",
    "FAULT_SCHEMA_KIND",
    "OWNER_ORDER",
    "OWNER_COUNT",
    "TOKEN_NBYTES",
    "INT64_NBYTES",
    "HEADER_INT64_COUNT",
    "HEADER_NBYTES",
    "PACKET_FIXED_NBYTES",
    "PACKET_PER_ENV_NBYTES",
    "FAULT_WORD_NBYTES",
    "D05_SELECTION_MISMATCH_FAULT_BIT",
    "DECISION_ACCEPT",
    "DECISION_CENSOR",
    "DECISIONS",
    "D05_PRETRANSFER_ADAPTER_READY",
    "D05_DECISION_CONSTRUCTION_REJECT",
    "D05_PRETRANSFER_ADAPTER_CONTRACT",
    "D05_PRETRANSFER_ADAPTER_CONTRACT_SHA256",
    "PACKET_ROW_INTEGRITY_SCHEMA",
    "PACKET_ROW_INTEGRITY_SCHEMA_SHA256",
    "RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA_SHA256",
    "packet_nbytes",
    "ActionBallFullMdpRevealBoundaryError",
    "ActionBallFullMdpRevealBoundaryPoisonedError",
    "ActionBallFullMdpRevealBoundaryD05AdapterHold",
    "ActionBallFullMdpRevealBoundaryD05PreTransferToken",
    "ActionBallFullMdpRevealBoundaryD05PreTransferView",
    "ActionBallFullMdpRevealBoundaryD05PreTransferAuthority",
    "ActionBallFullMdpRevealBoundaryFaultSchema",
    "ActionBallFullMdpRevealBoundaryPreparedTokenClaim",
    "ActionBallFullMdpRevealBoundaryChildTokenAuthority",
    "ActionBallFullMdpRevealBoundaryDeviceRow",
    "ActionBallFullMdpRevealBoundaryLaneAuthority",
    "ActionBallFullMdpRevealBoundaryAttempt",
    "ActionBallFullMdpRevealBoundaryAbortCapability",
    "ActionBallFullMdpRevealBoundaryOwnerRow",
    "ActionBallFullMdpRevealBoundaryReceipt",
    "ActionBallFullMdpRevealBoundaryOwner",
]
