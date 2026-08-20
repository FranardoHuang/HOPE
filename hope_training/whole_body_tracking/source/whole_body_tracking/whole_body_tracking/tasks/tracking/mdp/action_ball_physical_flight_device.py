"""Fresh multi-slot physical-flight owner for the ActionBall full MDP.

The owner consumes the portable ``action_ball_physical_flight_contract`` and
never imports or delegates to the legacy ``PhysicalBallManager``/``pb_ball``
path.  It implements the transaction semantics that can be closed without an
Isaac process:

* explicit receipt-owned ``K`` with an ``[N, K]`` scene projection;
* private prepare token, zero-mutation abort, and prevalidated child commit;
* distinct raw 52-byte state and full install-envelope roots;
* device-shaped post-physics publication joined to a typed R06 flight view;
* R06-authorized settle/retire input, masked true reset, and full externally
  pinned checkpoint bytes.

The real Isaac writer can still raise asynchronously.  Such a failure poisons
the owner and is never represented as rollback.  CUDA transaction/checkpoint
integration remains fail-closed until the single packed reveal boundary and
real Isaac continuation tests are wired; pure CPU tests do not authorize a
launch.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import torch

import action_ball_continuous_runtime_transaction as _r05
import action_ball_continuous_runtime_transaction_device as _r05_device
import action_ball_full_mdp_diagnostic_capacity as _diagnostic_capacity
import action_ball_full_mdp_row_identity as _row_identity
import action_ball_full_mdp_reset_genesis as _reset_genesis
import action_ball_full_mdp_reveal_boundary as _reveal_boundary
import action_ball_physical_flight_contract as _flight


SCHEMA_VERSION = 1
TOKEN_BYTES = 32
STATE_WIDTH = 13
CONTRACT_SOURCE_SHA256 = (
    "beff093949dda4fb8aab963217cd31d37c1d59875440a7043526864d224f2675"
)
# The terminal owner derives CENSOR facts from owner-issued evidence and closes
# its authority bind window before the first business operation or state read.
R05_SOURCE_SHA256: str | None = (
    "82d71c987e51cf4b5940744b124b36f9055d7a655af5e2279e2a6841cb1077dc"
)
# R06 is being changed in the same integration wave.  Keep the whole-file pin
# closed until those sibling bytes freeze; exact runtime types and retained
# direct methods still fail closed at the owner boundary below.
R06_SOURCE_SHA256: str | None = None
REVEAL_BOUNDARY_SOURCE_SHA256 = (
    "a5762b2e4838a3bdc58c2a30822467d27e4fb1006a37fcc3faf3948f7c2c24fe"
)
OWNER_STATE_SCHEMA = dict(_flight.PHYSICAL_OWNER_STATE_SCHEMA)
OWNER_STATE_SCHEMA_SHA256 = _flight.PHYSICAL_OWNER_STATE_SCHEMA_SHA256

R06_FLIGHT_EMPTY = 0
R06_FLIGHT_INBOUND = 1
R06_FLIGHT_OPEN = 2
R06_FLIGHT_SETTLED_RETAINED = 3

_ACTION_EPOCH_SUBSTEP_DENSE = "dense"
_ACTION_EPOCH_SUBSTEP_IDLE = "idle"

RUNTIME_INTEGRATED = False
CUDA_REVEAL_BOUNDARY_INTEGRATED = False
ISAAC_POSTPHYSICS_VALIDATED = False
FORMAL_EXACT_RESUME_INTEGRATED = False
R10_CHECKPOINT_ADAPTER_INTEGRATED = False
R05_REVEAL_SOURCE_PIN_PENDING = R05_SOURCE_SHA256 is None
PHYSICAL_SELECTED_RESET_DEVICE_PARK_INTEGRATED = False
PHYSICAL_SELECTED_RESET_R05_ACK_INTEGRATED = False
LAUNCH_AUTHORIZED = False
INTEGRATION_RESIDUALS = (
    "replace the CPU reference transaction with the one packed CUDA reveal boundary",
    "bind the real Isaac scene writer and prove post-physics ordering against R06",
    "bind R06 device retirement without a post-physics device-to-host transfer",
    "register the scene and owner bytes in the R10 whole-checkpoint transaction",
    "bind and test the exact device-R05 true-reset receipt authority before selected-reset child ACK",
    "prove physical selected-reset generation continuity and unselected byte parity on CUDA",
    "freeze and source-pin the runtime-owned immutable shared checkpoint join provider",
    "run N=64 semantics and N=4096 profiler-off Pod1 gates",
)
CHECKPOINT_JOIN_PROVIDER_API_SCHEMA_SHA256 = (
    "dc1cc5540bd73612e9677930bba14d83a809dd43c7c68fd48795642b40aa92d2"
)
PHYSICAL_GLOBAL_DRAIN_SOURCE_SHA256 = (
    "674d4d1ab6c7f1ac7f8b6a0c32e25003d2c5ee784921dedb5e43cb29fc35122e"
)
PHYSICAL_GLOBAL_DRAIN_ACK_AUTHORITY_API_SHA256 = (
    "f759474e1576a151b37939d128b0ae2c58b02f4cf90007353b41fadad03d902d"
)
PHYSICAL_GLOBAL_DRAIN_OWNER_KIND = "physical_ball"
PHYSICAL_HOT_FAULT_D05_PROJECTION = 1 << 0
PHYSICAL_HOT_FAULT_QUESTION_BINDING = 1 << 1
PHYSICAL_HOT_FAULT_IDENTITY = 1 << 2
PHYSICAL_HOT_FAULT_SCENE_PRECONDITION = 1 << 3
PHYSICAL_HOT_FAULT_LAUNCH_TICK = 1 << 4
PHYSICAL_EPOCH_FAULT_LAUNCH_SOURCE = 1 << 40
PHYSICAL_EPOCH_FAULT_POSTPHYSICS_PRODUCER = 1 << 41
PHYSICAL_EPOCH_FAULT_POSTPHYSICS_NONFINITE = 1 << 42
PHYSICAL_EPOCH_FACT_PRESENT = 1 << 0
PHYSICAL_EPOCH_FACT_SELECTED_CONTACT = 1 << 1
# Physical's fixed epoch fact slice is intentionally small.  Reward consumes
# only the two validity bits below; the remaining values retain causal scene
# data for diagnostics without turning the old payment ledger into authority.
PHYSICAL_EPOCH_FACT_CURRENT_CENTER = slice(0, 3)
PHYSICAL_EPOCH_FACT_CONTACT_CENTER = slice(3, 6)
PHYSICAL_EPOCH_FACT_OUTGOING_ANCHOR = slice(6, 9)
PHYSICAL_EPOCH_FACT_OBSERVATION_ORDINAL = 9
PHYSICAL_EPOCH_FACT_F32_WIDTH = 32
PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES = (
    "mutation_version",
    "fault_count",
    "invariant_count",
    "terminal_resolution_total",
    "shared_normal_retire_total",
    "physical_only_orphan_park_total",
    "shared_normal_retire_key_summary_0",
    "shared_normal_retire_key_summary_1",
    "selected_contact_pending_count",
    "selected_contact_view_total",
    "selected_contact_payment_total",
    "selected_contact_ledger_fault_count",
)
PHYSICAL_PPO_DRAIN_LEAF_SCHEMA = (
    PHYSICAL_GLOBAL_DRAIN_OWNER_KIND,
    tuple((name, "scalar", 0) for name in PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES),
)
_PHYSICAL_RETIRE_SUMMARY_MODULUS = 2147483647


def materialize_physical_ppo_drain_leaf_schema(
    *,
    leaf_schema_type: type,
    field_spec_type: type,
) -> object:
    """Build the dependency-neutral physical leaf schema for the top owner."""

    owner_kind, fields = PHYSICAL_PPO_DRAIN_LEAF_SCHEMA
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
_PREPARED_TOKEN = object()
_ARMED_TOKEN = object()
_CHILD_TERMINAL_TOKEN = object()
_POSTPHYSICS_TOKEN = object()
_POSTPHYSICS_CAPTURE_REQUEST_TOKEN = object()
_POSTPHYSICS_CAPTURE_FACTS_TOKEN = object()
_CONTACT_REWARD_VIEW_TOKEN = object()
_CONTACT_REWARD_PAYMENT_TOKEN = object()
_R06_ACK_TOKEN = object()
_TENSOR_SCENE_PORT_TOKEN = object()
_TENSOR_SCENE_WRITE_TOKEN = object()
_PHYSICAL_PARK_CLEANUP_TOKEN = object()
_PHYSICAL_RETIRE_PREPARE_TOKEN = object()
_PHYSICAL_RETIRE_ARM_TOKEN = object()
_PHYSICAL_RETIRE_COMMIT_TOKEN = object()
_PHYSICAL_R10_LIVE_ACK_TOKEN = object()
_PHYSICAL_HOT_PREPARE_TOKEN = object()
_PHYSICAL_HOT_COMMIT_TOKEN = object()
_PHYSICAL_LATE_LAUNCH_PUBLICATION_TOKEN = object()
_ACTION_EPOCH_SCENE_WRITE_TOKEN = object()
_ACTION_EPOCH_R06_LAUNCH_TOKEN = object()
_ACTION_EPOCH_R06_POSTPHYSICS_TOKEN = object()
_ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN = object()

_PHYSICAL_REVEAL_FAULT_SCHEMA = (
    _reveal_boundary.ActionBallFullMdpRevealBoundaryFaultSchema(
        schema_version=1,
        owner_kind="physical_ball",
        ordered_fault_bits=(("physical_prearm_contract_fault", 1),),
        allowed_fault_mask=1,
        precedence=("physical_prearm_contract_fault",),
    )
)


class PhysicalFlightDeviceError(RuntimeError):
    """Fresh physical owner schema, transaction, or runtime error."""


class PhysicalFlightOwnerPoisonedError(PhysicalFlightDeviceError):
    """A prevalidated child publication failed and rollback is untrustworthy."""


class PhysicalLateLaunchProductionHold(PhysicalFlightDeviceError):
    """The exact cross-owner late-launch receipt graph is not frozen yet."""


class PhysicalEpochIntegrationHold(PhysicalFlightDeviceError):
    """The lean epoch is missing one causal producer or active transition."""


def _verify_contract_source() -> None:
    actual = hashlib.sha256(Path(_flight.__file__).read_bytes()).hexdigest()
    if actual != CONTRACT_SOURCE_SHA256:
        raise PhysicalFlightDeviceError(
            "physical-flight portable contract differs from the reviewed source pin"
        )
    if R05_SOURCE_SHA256 is not None:
        r05_actual = hashlib.sha256(Path(_r05.__file__).read_bytes()).hexdigest()
        if r05_actual != R05_SOURCE_SHA256:
            raise PhysicalFlightDeviceError(
                "R05 reveal-prepare marker owner differs from the reviewed source pin"
            )
    boundary_actual = hashlib.sha256(
        Path(_reveal_boundary.__file__).read_bytes()
    ).hexdigest()
    if boundary_actual != REVEAL_BOUNDARY_SOURCE_SHA256:
        raise PhysicalFlightDeviceError(
            "all-owner reveal boundary differs from the reviewed source pin"
        )


def _global_drain_ack_authority_api_sha256(drain: object) -> str:
    """Hash the one construction-bound ACK method with the top ABI recipe."""

    authority_type = getattr(drain, "LeafDevicePackAuthority", None)
    source_path = inspect.getsourcefile(authority_type)
    if authority_type is None or source_path is None:
        raise PhysicalFlightDeviceError(
            "global drain ACK authority source is unavailable"
        )
    try:
        source_text = Path(source_path).read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        authority_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LeafDevicePackAuthority"
        )
        method_node = next(
            node
            for node in authority_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "require_owned_ack"
        )
        method_source = ast.get_source_segment(
            source_text,
            method_node,
            padded=False,
        )
    except (OSError, UnicodeError, SyntaxError, StopIteration) as exc:
        raise PhysicalFlightDeviceError(
            "global drain ACK authority API cannot be pinned"
        ) from exc
    if type(method_source) is not str or not method_source:
        raise PhysicalFlightDeviceError(
            "global drain ACK authority method source is unavailable"
        )
    payload = {
        "fields": (),
        "methods": (("require_owned_ack", method_source),),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhysicalFlightDeviceError(f"{label} must be an exact int >= {minimum}")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PhysicalFlightDeviceError(f"{label} must be one lowercase SHA-256")
    return value


def _digest_bytes(value: str, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8, device=device)


def _require_final_r06_type(value: object, *, expected_name: str) -> object:
    value_type = type(value)
    try:
        source = inspect.getsourcefile(value_type)
    except (OSError, TypeError):
        source = None
    module = sys.modules.get(value_type.__module__)
    expected_source = Path(__file__).with_name(
        "action_ball_landing_outcome_device.py"
    ).resolve()
    if (
        value_type.__name__ != expected_name
        or source is None
        or module is None
        or Path(source).resolve() != expected_source
        or getattr(module, expected_name, None) is not value_type
    ):
        raise PhysicalFlightDeviceError(f"R06 {expected_name} type differs")
    if R06_SOURCE_SHA256 is not None:
        try:
            actual = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except OSError as exc:
            raise PhysicalFlightDeviceError(
                f"R06 {expected_name} source cannot be pinned"
            ) from exc
        if actual != R06_SOURCE_SHA256:
            raise PhysicalFlightDeviceError(
                f"R06 {expected_name} source differs from the reviewed final pin"
            )
    return value


def _r06_mutation_version(value: object, *, label: str, device: torch.device) -> int:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != torch.Size([])
        or value.dtype != torch.int64
        or value.device != device
    ):
        raise PhysicalFlightDeviceError(f"{label} scalar ABI differs")
    return _exact_int(int(value.item()), label=label)


def _r06_device_mutation_version(
    value: object,
    *,
    label: str,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != torch.Size([])
        or value.dtype != torch.int64
        or value.device != device
    ):
        raise PhysicalFlightDeviceError(f"{label} scalar ABI differs")
    return value


def _digest_hex(value: torch.Tensor) -> str:
    if value.device.type != "cpu":
        raise PhysicalFlightDeviceError(
            "CUDA digest materialization is forbidden outside the packed boundary"
        )
    if value.shape != (TOKEN_BYTES,) or value.dtype != torch.uint8:
        raise PhysicalFlightDeviceError("digest tensor ABI differs")
    return bytes(value.tolist()).hex()


def _canonical_state_tensor(
    value: _flight.CanonicalPhysicalBallStateF32,
    *,
    device: torch.device,
) -> torch.Tensor:
    if type(value) is not _flight.CanonicalPhysicalBallStateF32:
        raise PhysicalFlightDeviceError("physical state must be canonical float32")
    return torch.tensor(value.ordered_values, dtype=torch.float32, device=device)


def _canonical_state_from_cpu_tensor(
    value: torch.Tensor,
) -> _flight.CanonicalPhysicalBallStateF32:
    if value.device.type != "cpu" or value.shape != (STATE_WIDTH,) or value.dtype != torch.float32:
        raise PhysicalFlightDeviceError("CPU canonical state tensor ABI differs")
    rows = tuple(float(item) for item in value.tolist())
    return _flight.CanonicalPhysicalBallStateF32(
        position_env_m=rows[0:3],
        quaternion_wxyz=rows[3:7],
        linear_velocity_world_mps=rows[7:10],
        angular_velocity_world_radps=rows[10:13],
    )


def _tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
    ):
        raise PhysicalFlightDeviceError(
            f"{label} must have shape={shape}, dtype={dtype}, device={device}"
        )
    return value


def _device_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return one device bool over exact tensor bytes, including NaN/-0 bits."""

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        raise PhysicalFlightDeviceError("device bitwise tensor ABI differs")
    return torch.eq(
        left.contiguous().view(torch.uint8),
        right.contiguous().view(torch.uint8),
    ).all()


@dataclass(frozen=True)
class TensorSceneWriteHandle:
    state_after: torch.Tensor
    selected_mask: torch.Tensor
    _port_identity: object
    _write_nonce: int
    _token: object


@dataclass(frozen=True, eq=False)
class TensorScenePortCapability:
    num_envs: int
    flight_capacity: int
    device: torch.device
    _port_identity: object
    _token: object


@dataclass(frozen=True, eq=False)
class TensorSceneApplyReceipt:
    write_nonce: int
    full_grid_write: bool
    _port_identity: object
    _handle_identity: object
    _token: object


@dataclass(frozen=True, eq=False)
class TensorSceneAbortReceipt:
    write_nonce: int
    _port_identity: object
    _handle_identity: object
    _token: object


class TensorPhysicalFlightScenePort:
    """Deterministic tensor scene used by focused tests and pure composition.

    This is not an Isaac runtime substitute.  It exists so transaction and
    checkpoint semantics can be tested without fabricating an Isaac GO.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        flight_capacity: int,
        device: torch.device | str = "cpu",
        park_position_env_m: tuple[float, float, float] = (0.0, 0.0, -20.0),
    ) -> None:
        self.num_envs = _exact_int(num_envs, label="num_envs", minimum=1)
        self.flight_capacity = _exact_int(
            flight_capacity, label="flight_capacity", minimum=1
        )
        requested_device = torch.device(device)
        self._state = torch.zeros(
            (self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=requested_device,
        )
        self.device = self._state.device
        self._state[..., :3] = torch.tensor(
            park_position_env_m, dtype=torch.float32, device=self.device
        )
        self._state[..., 3] = 1.0
        self.apply_count = 0
        self.fail_next_apply = False
        self._identity = object()
        self._next_write_nonce = 1
        self._active_write_handles: dict[int, TensorSceneWriteHandle] = {}
        self._scene_port_capability = TensorScenePortCapability(
            num_envs=self.num_envs,
            flight_capacity=self.flight_capacity,
            device=self.device,
            _port_identity=self._identity,
            _token=_TENSOR_SCENE_PORT_TOKEN,
        )

    @property
    def scene_port_capability(self) -> TensorScenePortCapability:
        return self._scene_port_capability

    def require_owned_scene_port_capability(
        self, value: object
    ) -> TensorScenePortCapability:
        if (
            type(value) is not TensorScenePortCapability
            or value is not self._scene_port_capability
            or value._port_identity is not self._identity
            or value._token is not _TENSOR_SCENE_PORT_TOKEN
        ):
            raise PhysicalFlightDeviceError(
                "tensor scene-port capability is stale or foreign"
            )
        return value

    def read_state_env(self) -> torch.Tensor:
        return self._state.clone()

    def preflight_write(
        self,
        state_env: object,
        selected_mask: object,
        *,
        device_faults_bound_in_reveal_row: bool,
    ) -> TensorSceneWriteHandle:
        state = _tensor(
            state_env,
            label="scene state",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        mask = _tensor(
            selected_mask,
            label="scene selected mask",
            shape=(self.num_envs, self.flight_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        if (
            type(device_faults_bound_in_reveal_row) is not bool
            or not device_faults_bound_in_reveal_row
        ):
            raise PhysicalFlightDeviceError(
                "scene state lacks packed-boundary device-fault binding"
            )
        nonce = self._next_write_nonce
        self._next_write_nonce += 1
        handle = TensorSceneWriteHandle(
            state_after=state.detach().clone(),
            selected_mask=mask.detach().clone(),
            _port_identity=self._identity,
            _write_nonce=nonce,
            _token=_TENSOR_SCENE_WRITE_TOKEN,
        )
        self._active_write_handles[nonce] = handle
        return handle

    def apply_prevalidated_write(self, handle: object) -> TensorSceneApplyReceipt:
        if (
            type(handle) is not TensorSceneWriteHandle
            or handle._port_identity is not self._identity
            or handle._token is not _TENSOR_SCENE_WRITE_TOKEN
            or self._active_write_handles.get(handle._write_nonce) is not handle
        ):
            raise PhysicalFlightDeviceError("tensor scene handle was not prevalidated")
        if self.fail_next_apply:
            self.fail_next_apply = False
            raise RuntimeError("injected scene publication failure")
        self._state.copy_(
            torch.where(
                handle.selected_mask.unsqueeze(-1),
                handle.state_after,
                self._state,
            )
        )
        self._active_write_handles.pop(handle._write_nonce, None)
        self.apply_count += 1
        return TensorSceneApplyReceipt(
            write_nonce=handle._write_nonce,
            full_grid_write=True,
            _port_identity=self._identity,
            _handle_identity=handle,
            _token=_TENSOR_SCENE_WRITE_TOKEN,
        )

    def require_owned_apply_receipt(
        self,
        handle: object,
        receipt: object,
    ) -> TensorSceneApplyReceipt:
        if (
            type(handle) is not TensorSceneWriteHandle
            or type(receipt) is not TensorSceneApplyReceipt
            or receipt._port_identity is not self._identity
            or receipt._handle_identity is not handle
            or receipt._token is not _TENSOR_SCENE_WRITE_TOKEN
            or receipt.write_nonce != handle._write_nonce
            or receipt.full_grid_write is not True
            or self._active_write_handles.get(handle._write_nonce) is not None
        ):
            raise PhysicalFlightDeviceError(
                "tensor scene apply receipt is stale or foreign"
            )
        return receipt

    def abort_prevalidated_write(
        self, handle: object
    ) -> TensorSceneAbortReceipt:
        if (
            type(handle) is not TensorSceneWriteHandle
            or handle._port_identity is not self._identity
            or handle._token is not _TENSOR_SCENE_WRITE_TOKEN
            or self._active_write_handles.get(handle._write_nonce) is not handle
        ):
            raise PhysicalFlightDeviceError(
                "tensor scene abort handle is stale or foreign"
            )
        self._active_write_handles.pop(handle._write_nonce, None)
        return TensorSceneAbortReceipt(
            write_nonce=handle._write_nonce,
            _port_identity=self._identity,
            _handle_identity=handle,
            _token=_TENSOR_SCENE_WRITE_TOKEN,
        )

    def require_owned_abort_receipt(
        self,
        handle: object,
        receipt: object,
    ) -> TensorSceneAbortReceipt:
        if (
            type(handle) is not TensorSceneWriteHandle
            or type(receipt) is not TensorSceneAbortReceipt
            or receipt._port_identity is not self._identity
            or receipt._handle_identity is not handle
            or receipt._token is not _TENSOR_SCENE_WRITE_TOKEN
            or receipt.write_nonce != handle._write_nonce
            or self._active_write_handles.get(handle._write_nonce) is not None
        ):
            raise PhysicalFlightDeviceError(
                "tensor scene abort receipt is stale or foreign"
            )
        return receipt


@dataclass(frozen=True)
class PhysicalFlightSceneSnapshotNK:
    """Read-only-by-contract ``[N,K]`` physical scene projection."""

    state_env_f32: torch.Tensor
    lifecycle_code: torch.Tensor
    ball_generation: torch.Tensor
    outcome_key_sha256: torch.Tensor
    install_payload_sha256: torch.Tensor
    installed_ball_state_sha256: torch.Tensor
    reveal_control_step: torch.Tensor
    observation_ordinal: torch.Tensor
    physically_parked: torch.Tensor
    published_to_runtime: torch.Tensor
    owner_fault: torch.Tensor
    slot_mutation_version: torch.Tensor
    owner_mutation_version: int


@dataclass(frozen=True)
class R06PhysicalFlightReadOnlySnapshot:
    """Typed device projection that R06 must mint after its own mutation."""

    flight_state: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    observation_ordinal: torch.Tensor
    owner_mutation_version: int


@dataclass(frozen=True)
class AcknowledgedR06PhysicalSnapshot:
    """Single-use authority minted only after the physical/R06 ack join."""

    _snapshot_root_sha256: str
    _owner_mutation_version: int
    _owner_identity: object
    _token: object

    @property
    def snapshot_root_sha256(self) -> str:
        return self._snapshot_root_sha256

    @property
    def owner_mutation_version(self) -> int:
        return self._owner_mutation_version


def r06_physical_snapshot_root(
    value: R06PhysicalFlightReadOnlySnapshot,
) -> str:
    """Canonical root of the complete CPU R06 flight after-image."""

    if type(value) is not R06PhysicalFlightReadOnlySnapshot:
        raise PhysicalFlightDeviceError("R06 physical snapshot type differs")
    state = value.flight_state
    key = value.full_key_sha256
    generation = value.ball_generation
    ordinal = value.observation_ordinal
    if (
        not isinstance(state, torch.Tensor)
        or state.device.type != "cpu"
        or state.dtype != torch.int8
        or state.ndim != 2
        or not isinstance(key, torch.Tensor)
        or key.device.type != "cpu"
        or key.dtype != torch.uint8
        or tuple(key.shape) != tuple(state.shape) + (TOKEN_BYTES,)
        or not isinstance(generation, torch.Tensor)
        or generation.device.type != "cpu"
        or generation.dtype != torch.int64
        or tuple(generation.shape) != tuple(state.shape)
        or not isinstance(ordinal, torch.Tensor)
        or ordinal.device.type != "cpu"
        or ordinal.dtype != torch.int64
        or tuple(ordinal.shape) != tuple(state.shape)
    ):
        raise PhysicalFlightDeviceError("R06 physical snapshot tensor ABI differs")
    version = _exact_int(
        value.owner_mutation_version,
        label="R06 owner_mutation_version",
    )
    key_hex = [
        [bytes(key[env_id, slot].tolist()).hex() for slot in range(state.shape[1])]
        for env_id in range(state.shape[0])
    ]
    return _flight.canonical_sha256(
        {
            "schema_version": 1,
            "kind": "action_ball_r06_physical_read_only_snapshot_root_v1",
            "flight_state": state.tolist(),
            "full_key_sha256": key_hex,
            "ball_generation": generation.tolist(),
            "observation_ordinal": ordinal.tolist(),
            "owner_mutation_version": version,
        }
    )


def _r06_device_key_equal(left: object, right: object) -> bool:
    if type(left) is not type(right) or type(left).__name__ != "DeviceLandingOutcomeKey":
        return False
    try:
        key_fields = fields(type(left))
    except TypeError:
        return False
    return all(
        isinstance(getattr(left, field.name, None), torch.Tensor)
        and isinstance(getattr(right, field.name, None), torch.Tensor)
        and torch.equal(
            getattr(left, field.name),
            getattr(right, field.name),
        )
        for field in key_fields
    )


def _r06_lifecycle_snapshot_equal(left: object, right: object) -> bool:
    if (
        type(left) is not type(right)
        or type(left).__name__ != "FlightLifecycleSnapshotBatch"
        or not _r06_device_key_equal(left.task_key, right.task_key)
    ):
        return False
    return all(
        isinstance(getattr(left, name, None), torch.Tensor)
        and isinstance(getattr(right, name, None), torch.Tensor)
        and torch.equal(getattr(left, name), getattr(right, name))
        for name in (
            "state",
            "full_key_sha256",
            "ball_generation",
            "mailbox_slot",
            "observation_ordinal",
            "physical_retired",
            "mailbox_physical_retired",
            "mutation_version",
        )
    )


def _r06_retire_result_equal(left: object, right: object) -> bool:
    if type(left) is not type(right) or type(left).__name__ != "PhysicalRetireMutationResult":
        return False
    tensor_fields = (
        "accepted",
        "rejected",
        "fault_bits",
        "normal_mask",
        "cleanup_mask",
        "portable_success_mask",
        "full_key_sha256",
        "ball_generation",
        "mailbox_slot",
        "observation_ordinal",
        "physical_retired",
        "mailbox_physical_retired",
        "mutation_version_before",
        "mutation_version_after",
    )
    if not all(
        isinstance(getattr(left, name, None), torch.Tensor)
        and isinstance(getattr(right, name, None), torch.Tensor)
        and torch.equal(getattr(left, name), getattr(right, name))
        for name in tensor_fields
    ):
        return False
    if not _r06_device_key_equal(left.task_key, right.task_key):
        return False
    return _r06_lifecycle_snapshot_equal(
        left.final_lifecycle_root,
        right.final_lifecycle_root,
    )


def _device_tensor_mismatch(
    left: object,
    right: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return one resident scalar mismatch without synchronizing the device."""

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.device != device
        or right.device != device
        or left.dtype != right.dtype
        or tuple(left.shape) != tuple(right.shape)
    ):
        return torch.ones((), dtype=torch.bool, device=device)
    return ~torch.eq(left, right).all()


def _r06_device_key_mismatch(
    left: object,
    right: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compare an exact R06 key entirely on its resident device."""

    mismatch = torch.zeros((), dtype=torch.bool, device=device)
    if (
        type(left) is not type(right)
        or type(left).__name__ != "DeviceLandingOutcomeKey"
    ):
        return torch.ones_like(mismatch)
    try:
        key_fields = fields(type(left))
    except TypeError:
        return torch.ones_like(mismatch)
    for field in key_fields:
        mismatch.logical_or_(
            _device_tensor_mismatch(
                getattr(left, field.name, None),
                getattr(right, field.name, None),
                device=device,
            )
        )
    return mismatch


def _r06_lifecycle_snapshot_device_mismatch(
    left: object,
    right: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compare the complete R06 lifecycle root with no host observation."""

    mismatch = torch.zeros((), dtype=torch.bool, device=device)
    if (
        type(left) is not type(right)
        or type(left).__name__ != "FlightLifecycleSnapshotBatch"
    ):
        return torch.ones_like(mismatch)
    mismatch.logical_or_(
        _r06_device_key_mismatch(left.task_key, right.task_key, device=device)
    )
    mismatch.logical_or_(
        _r06_device_key_mismatch(
            left.mailbox_task_key,
            right.mailbox_task_key,
            device=device,
        )
    )
    for name in (
        "state",
        "full_key_sha256",
        "ball_generation",
        "mailbox_slot",
        "observation_ordinal",
        "physical_retired",
        "mailbox_state",
        "mailbox_full_key_sha256",
        "mailbox_ball_generation",
        "mailbox_reserved_flight_slot",
        "mailbox_history_valid",
        "mailbox_physical_retired",
        "mutation_version",
    ):
        mismatch.logical_or_(
            _device_tensor_mismatch(
                getattr(left, name, None),
                getattr(right, name, None),
                device=device,
            )
        )
    return mismatch


def _r06_retire_result_device_mismatch(
    left: object,
    right: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compare an R06 retire result while keeping the verdict on device."""

    mismatch = torch.zeros((), dtype=torch.bool, device=device)
    if (
        type(left) is not type(right)
        or type(left).__name__ != "PhysicalRetireMutationResult"
    ):
        return torch.ones_like(mismatch)
    for name in (
        "accepted",
        "rejected",
        "fault_bits",
        "normal_mask",
        "cleanup_mask",
        "portable_success_mask",
        "full_key_sha256",
        "ball_generation",
        "mailbox_slot",
        "observation_ordinal",
        "physical_retired",
        "mailbox_state",
        "mailbox_full_key_sha256",
        "mailbox_ball_generation",
        "mailbox_reserved_flight_slot",
        "mailbox_history_valid",
        "mailbox_physical_retired",
        "mutation_version_before",
        "mutation_version_after",
    ):
        mismatch.logical_or_(
            _device_tensor_mismatch(
                getattr(left, name, None),
                getattr(right, name, None),
                device=device,
            )
        )
    mismatch.logical_or_(
        _r06_device_key_mismatch(left.task_key, right.task_key, device=device)
    )
    mismatch.logical_or_(
        _r06_device_key_mismatch(
            left.mailbox_task_key,
            right.mailbox_task_key,
            device=device,
        )
    )
    mismatch.logical_or_(
        _r06_lifecycle_snapshot_device_mismatch(
            left.initial_lifecycle_root,
            right.initial_lifecycle_root,
            device=device,
        )
    )
    mismatch.logical_or_(
        _r06_lifecycle_snapshot_device_mismatch(
            left.final_lifecycle_root,
            right.final_lifecycle_root,
            device=device,
        )
    )
    return mismatch


@dataclass(frozen=True)
class PhysicsStampGrid:
    control_step: torch.Tensor
    physics_substep: torch.Tensor
    event_phase: torch.Tensor


@dataclass(frozen=True)
class IsaacPostPhysicsFacts:
    """Engine facts sampled after physics for the complete ``[N,K]`` grid."""

    observation_stamp: PhysicsStampGrid
    current_state_env_f32: torch.Tensor
    selected_contact_event: torch.Tensor
    selected_contact_ball_center_m: torch.Tensor
    selected_contact_outgoing_segment_anchor_m: torch.Tensor
    selected_contact_stamp: PhysicsStampGrid
    net_crossing_event: torch.Tensor
    net_clear_at_crossing: torch.Tensor
    net_crossing_stamp: PhysicsStampGrid
    crossing_report_delivered: torch.Tensor
    first_descending_crossing_event: torch.Tensor
    first_descending_crossing_xy_m: torch.Tensor
    first_descending_crossing_stamp: PhysicsStampGrid
    nonfinite_observation: torch.Tensor
    producer_contract_fault: torch.Tensor
    engine_overflow: torch.Tensor
    # These two fields are deliberately optional for the pure tensor diagnostic
    # surface.  The real post-scene-update callpoint accepts only the one object
    # sealed by ``capture_post_physics_facts`` on this owner.
    _owner_identity: object | None = None
    _capture_token: object | None = None


@dataclass(frozen=True, eq=False)
class PhysicalPostPhysicsCaptureRequest:
    """Owner-issued, one-substep input to the exact concrete scene producer."""

    exact_stamp: tuple[int, int, int, int, int]
    observe_mask: torch.Tensor
    flight_slot: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center_m: torch.Tensor
    current_state_env_f32: torch.Tensor | None
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class PhysicalEpochSelectedContactFacts:
    """Clone-only direct Reward facts from the latest real engine capture.

    ``eligible`` means that the selected launched ball had a real selected
    rubber contact on this post-physics source step.  An ordinary live miss is
    ``present=True, eligible=False, producer_fault_bits=0``.  Missing or stale
    producers are never converted to a miss.
    """

    present: torch.Tensor
    eligible: torch.Tensor
    source_step: torch.Tensor
    producer_fault_bits: torch.Tensor
    current_ball_center_m: torch.Tensor
    selected_contact_ball_center_m: torch.Tensor
    selected_contact_outgoing_segment_anchor_m: torch.Tensor


@dataclass(frozen=True)
class _ActionEpochPendingLaunchState:
    """Full-N Physical-private rows waiting for their Motion launch tick."""

    pending: torch.Tensor
    flight_slot: torch.Tensor
    shot_key: _row_identity.ActionEpochShotKey
    publication_ordinal: torch.Tensor
    physical_state_f32: torch.Tensor
    target_xy_m: torch.Tensor
    launch_control_step: torch.Tensor
    contact_deadline_control_step: torch.Tensor
    crossing_horizon_control_step: torch.Tensor


@dataclass(frozen=True)
class _ActionEpochDirectState:
    """Clone-only pending/live Physical identity state used by selected reset."""

    pending_launch: _ActionEpochPendingLaunchState | None
    active_flight_slot: torch.Tensor
    flight_shot_key: _row_identity.ActionEpochShotKey
    flight_publication_ordinal: torch.Tensor


class ActionEpochSceneWriteProjection:
    """One-shot Physical-owned scene after-image for launch or retirement.

    Construction is deliberately private.  The exact cold-bound scene port
    consumes this object through Physical's validator; callers never supply a
    state tensor, slot tensor, mask, tick, or verdict to the scene writer.
    """

    __slots__ = (
        "kind",
        "state_env_f32",
        "selected_mask",
        "physical_owner",
        "epoch_owner",
        "_owner_identity",
        "_token",
    )

    def __init__(
        self,
        *,
        kind: str,
        state_env_f32: torch.Tensor,
        selected_mask: torch.Tensor,
        physical_owner: object,
        epoch_owner: object,
        owner_identity: object,
        _token: object,
    ) -> None:
        if _token is not _ACTION_EPOCH_SCENE_WRITE_TOKEN:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch scene write projection is Physical-owned"
            )
        self.kind = kind
        self.state_env_f32 = state_env_f32
        self.selected_mask = selected_mask
        self.physical_owner = physical_owner
        self.epoch_owner = epoch_owner
        self._owner_identity = owner_identity
        self._token = _token


class ActionEpochR06LaunchProjection:
    """One-shot Physical-owned rows needed to establish R06 live flights.

    R06 may read this view only while the exact Physical launch transaction is
    active.  It contains D05-owned accepted identities and Physical's private
    K-slot allocation; no caller supplies launch tensors or an outcome verdict.
    """

    __slots__ = (
        "selected_mask",
        "due",
        "late_launch",
        "flight_slot",
        "shot_key",
        "publication_ordinal",
        "target_xy_m",
        "launch_control_step",
        "contact_deadline_control_step",
        "crossing_horizon_control_step",
        "physical_owner",
        "epoch_owner",
        "_owner_identity",
        "_token",
    )

    def __init__(
        self,
        *,
        selected_mask: torch.Tensor,
        due: torch.Tensor,
        late_launch: torch.Tensor,
        flight_slot: torch.Tensor,
        shot_key: _row_identity.ActionEpochShotKey,
        publication_ordinal: torch.Tensor,
        target_xy_m: torch.Tensor,
        launch_control_step: torch.Tensor,
        contact_deadline_control_step: torch.Tensor,
        crossing_horizon_control_step: torch.Tensor,
        physical_owner: object,
        epoch_owner: object,
        owner_identity: object,
        _token: object,
    ) -> None:
        if _token is not _ACTION_EPOCH_R06_LAUNCH_TOKEN:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch R06 launch projection is Physical-owned"
            )
        self.selected_mask = selected_mask
        self.due = due
        self.late_launch = late_launch
        self.flight_slot = flight_slot
        self.shot_key = shot_key
        self.publication_ordinal = publication_ordinal
        self.target_xy_m = target_xy_m
        self.launch_control_step = launch_control_step
        self.contact_deadline_control_step = contact_deadline_control_step
        self.crossing_horizon_control_step = crossing_horizon_control_step
        self.physical_owner = physical_owner
        self.epoch_owner = epoch_owner
        self._owner_identity = owner_identity
        self._token = _token


class ActionEpochR06PostPhysicsProjection:
    """One active Physical-owned typed postphysics packet for direct R06.

    The exact bound R06 owner may pull this object only while Physical's
    post-scene-update transaction is active.  Typed D05/epoch identities, not
    a digest or caller batch, join each live ``[N,K]`` slot to engine facts.
    """

    __slots__ = (
        "observe_mask",
        "flight_slot",
        "shot_key",
        "publication_ordinal",
        "observation_ordinal",
        "previous_ball_center_m",
        "current_ball_center_m",
        "observation_stamp",
        "selected_contact_event",
        "selected_contact_ball_center_m",
        "selected_contact_outgoing_segment_anchor_m",
        "selected_contact_stamp",
        "net_crossing_event",
        "net_clear_at_crossing",
        "net_crossing_stamp",
        "crossing_report_delivered",
        "first_descending_crossing_event",
        "first_descending_crossing_xy_m",
        "first_descending_crossing_stamp",
        "nonfinite_observation",
        "producer_contract_fault",
        "engine_overflow",
        "owner_fault_bits",
        "fact_valid_bits",
        "fact_source_step",
        "fact_f32",
        "physical_owner",
        "epoch_owner",
        "_owner_identity",
        "_token",
    )

    def __init__(self, *, _token: object, **values: object) -> None:
        if _token is not _ACTION_EPOCH_R06_POSTPHYSICS_TOKEN:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch R06 postphysics projection is Physical-owned"
            )
        for name in self.__slots__:
            if name == "_token":
                continue
            setattr(self, name, values[name])
        self._token = _token


class ActionEpochPhysicsFactAllocationProjection:
    """One-prephysics-call Physical-owned live K-grid for face attribution.

    The projection joins already-live Physical slots with rows due in the
    current Motion tick.  Racket and the scene fact owner may consume it only
    while ``launch_action_epoch`` is active; no caller chooses an environment,
    flight slot, face, or verdict.
    """

    __slots__ = (
        "active_mask",
        "launch_due_mask",
        "flight_slot",
        "shot_key",
        "publication_ordinal",
        "full_key_sha256",
        "physical_owner",
        "epoch_owner",
        "_owner_identity",
        "_token",
    )

    def __init__(
        self,
        *,
        active_mask: torch.Tensor,
        launch_due_mask: torch.Tensor,
        flight_slot: torch.Tensor,
        shot_key: _row_identity.ActionEpochShotKey,
        publication_ordinal: torch.Tensor,
        full_key_sha256: torch.Tensor,
        physical_owner: object,
        epoch_owner: object,
        owner_identity: object,
        _token: object,
    ) -> None:
        if _token is not _ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch physics-fact allocation is Physical-owned"
            )
        self.active_mask = active_mask
        self.launch_due_mask = launch_due_mask
        self.flight_slot = flight_slot
        self.shot_key = shot_key
        self.publication_ordinal = publication_ordinal
        self.full_key_sha256 = full_key_sha256
        self.physical_owner = physical_owner
        self.epoch_owner = epoch_owner
        self._owner_identity = owner_identity
        self._token = _token

    @property
    def ball_generation(self) -> torch.Tensor:
        """Compatibility view for the still-separate Racket/scene migration."""

        return self.shot_key.ball_generation

    @property
    def action_uid(self) -> torch.Tensor:
        return self.shot_key.action_uid

    @property
    def reset_generation(self) -> torch.Tensor:
        return self.shot_key.reset_generation

    @property
    def action_slot(self) -> torch.Tensor:
        return self.shot_key.action_slot

    @property
    def task_identity(self) -> torch.Tensor:
        return self.shot_key.task_identity


@dataclass(frozen=True)
class _PostPhysicsCaptureImage:
    exact_stamp: tuple[int, int, int, int, int]
    observe_mask: torch.Tensor
    flight_slot: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    observation_ordinal: torch.Tensor
    current_state_env_f32: torch.Tensor
    slot_version: torch.Tensor
    owner_mutation_version: int


@dataclass(frozen=True)
class PhysicalPostPhysicsPublication:
    """R06-compatible post-physics packet plus owner join faults."""

    observe_mask: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center_m: torch.Tensor
    current_ball_center_m: torch.Tensor
    observation_stamp: PhysicsStampGrid
    selected_contact_event: torch.Tensor
    selected_contact_ball_center_m: torch.Tensor
    selected_contact_outgoing_segment_anchor_m: torch.Tensor
    selected_contact_stamp: PhysicsStampGrid
    net_crossing_event: torch.Tensor
    net_clear_at_crossing: torch.Tensor
    net_crossing_stamp: PhysicsStampGrid
    crossing_report_delivered: torch.Tensor
    first_descending_crossing_event: torch.Tensor
    first_descending_crossing_xy_m: torch.Tensor
    first_descending_crossing_stamp: PhysicsStampGrid
    nonfinite_observation: torch.Tensor
    producer_contract_fault: torch.Tensor
    engine_overflow: torch.Tensor
    owner_join_fault: torch.Tensor
    _owner_identity: object
    _token: object


@dataclass(frozen=True, eq=False, repr=False)
class PhysicalSelectedContactRewardView:
    """Owner-issued, exact one-publication Reward view.

    Construction is not authority: payment additionally requires exact object
    identity in the Physical owner's active registry.
    """

    eligible: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    flight_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    selected_contact_control_step: torch.Tensor
    selected_contact_physics_substep: torch.Tensor
    selected_contact_event_phase: torch.Tensor
    _owner_identity: object
    _token: object


@dataclass(frozen=True, eq=False, repr=False)
class PhysicalSelectedContactRewardPaymentResult:
    """Owner-issued device verdict for one exact selected-contact payment.

    Dataclass construction is not authority.  The active Physical owner keeps
    the exact verdict object and its private after-image until the top Reward
    coordinator validates and closes the cycle.
    """

    accepted: torch.Tensor
    rejected: torch.Tensor
    paid_raw_reward: torch.Tensor
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class PhysicalSettleRetireInput:
    """R06 after-image authorizing physical-only retirement."""

    retire_mask: torch.Tensor
    r06_ack: AcknowledgedR06PhysicalSnapshot


@dataclass(frozen=True)
class PhysicalRetireDeviceResult:
    accepted: torch.Tensor
    rejected: torch.Tensor
    owner_join_fault: torch.Tensor
    portable_receipt: Optional[_flight.PhysicalRetireReceipt]


@dataclass(frozen=True, eq=False)
class PhysicalParkCleanupMaskCapability:
    """Owner-retained physical orphan/join cleanup mask shared with paired R06."""

    _device_mask: torch.Tensor
    _owner_identity: object
    _prepared_token: object
    _token: object

    @property
    def device_mask(self) -> torch.Tensor:
        return self._device_mask.detach().clone()


class PreparedPhysicalSettleRetire:
    """Opaque physical park after-image bound to one exact R06 prepare."""

    __slots__ = (
        "_r06_prepared_retire",
        "_r06_cleanup_capability",
        "_physical_cleanup_capability",
        "_owner_identity",
        "_token",
    )

    def __init__(
        self,
        *,
        r06_prepared_retire: object,
        r06_cleanup_capability: object,
        physical_cleanup_mask: torch.Tensor,
        owner_identity: object,
        _token: object,
    ) -> None:
        object.__setattr__(self, "_r06_prepared_retire", r06_prepared_retire)
        object.__setattr__(self, "_r06_cleanup_capability", r06_cleanup_capability)
        object.__setattr__(self, "_owner_identity", owner_identity)
        object.__setattr__(self, "_token", _token)
        object.__setattr__(
            self,
            "_physical_cleanup_capability",
            PhysicalParkCleanupMaskCapability(
                _device_mask=physical_cleanup_mask,
                _owner_identity=owner_identity,
                _prepared_token=self,
                _token=_PHYSICAL_PARK_CLEANUP_TOKEN,
            ),
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("physical settle-retire token is immutable")


@dataclass(frozen=True, eq=False)
class ArmedPhysicalSettleRetire:
    """No-fail physical park handle after both owners cross-bind after-images."""

    _prepared_retire: PreparedPhysicalSettleRetire
    _r06_armed_retire: object
    _owner_identity: object
    _token: object


@dataclass(frozen=True, eq=False)
class PhysicalSettleRetireCommitToken:
    """Opaque proof that the physical scene was parked before R06 commit."""

    _armed_retire: ArmedPhysicalSettleRetire
    _owner_identity: object
    _token: object


class _OpaquePhysicalSelectedResetCapability:
    """Non-copyable base for owner-registry selected-reset identities."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("physical selected-reset capabilities are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("physical selected-reset capabilities are immutable")

    def __copy__(self):
        raise TypeError("physical selected-reset capabilities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("physical selected-reset capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("physical selected-reset capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("physical selected-reset capabilities cannot be serialized")


class StagedPhysicalSelectedTrueReset(_OpaquePhysicalSelectedResetCapability):
    """Empty capability; all selected-reset facts remain owner-private."""

    __slots__ = ()


class FinalizedPhysicalSelectedTrueReset(_OpaquePhysicalSelectedResetCapability):
    """Empty capability for an allocation-complete physical after-image."""

    __slots__ = ()


class ArmedPhysicalSelectedTrueReset(_OpaquePhysicalSelectedResetCapability):
    """Empty capability after the physical/R06 no-fail prearm boundary."""

    __slots__ = ()


class PhysicalSelectedTrueResetParkCommitToken(
    _OpaquePhysicalSelectedResetCapability
):
    """Empty capability proving physical park preceded R06 retirement."""

    __slots__ = ()


class PhysicalSelectedTrueResetCompletionToken(
    _OpaquePhysicalSelectedResetCapability
):
    """Empty child settlement capability; it makes no health claim."""

    __slots__ = ()


@dataclass(frozen=True)
class _PhysicalSelectedResetStageRecord:
    """Owner-private data behind one empty staged capability."""

    capability: StagedPhysicalSelectedTrueReset
    r06_prepared_reset: object
    r06_mask_capability: object
    device_r05_prepared_true_reset: object
    device_r05_prepared_projection: object
    selected_env_mask: torch.Tensor
    reset_generation_before: torch.Tensor
    reset_generation_after: torch.Tensor
    generation_contract_fault: torch.Tensor


@dataclass(frozen=True)
class _PhysicalSelectedResetCompletionRecord:
    """Owner-private causal identities behind one empty child settlement."""

    capability: PhysicalSelectedTrueResetCompletionToken
    park_commit_token: PhysicalSelectedTrueResetParkCommitToken
    r06_commit_token: object
    device_r05_true_reset_receipt: _r05_device.DeviceR05TrueResetReceipt


@dataclass(frozen=True)
class _PhysicalSelectedResetStaleWitness:
    scene_state: torch.Tensor
    lifecycle: torch.Tensor
    generation: torch.Tensor
    outcome_sha: torch.Tensor
    install_sha: torch.Tensor
    installed_state_sha: torch.Tensor
    reveal_step: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center: torch.Tensor
    parked: torch.Tensor
    published: torch.Tensor
    slot_version: torch.Tensor
    device_fault: torch.Tensor
    reset_generation: torch.Tensor
    action_epoch_direct_state: _ActionEpochDirectState
    owner_mutation_version: int
    next_prepare_nonce: int


@dataclass(frozen=True)
class _PhysicalSelectedResetImage:
    selected_env_mask: torch.Tensor
    selected_slot_mask: torch.Tensor
    scene_handle: object
    scene_after: torch.Tensor
    lifecycle_after: torch.Tensor
    generation_after: torch.Tensor
    outcome_sha_after: torch.Tensor
    install_sha_after: torch.Tensor
    installed_state_sha_after: torch.Tensor
    reveal_step_after: torch.Tensor
    observation_ordinal_after: torch.Tensor
    previous_ball_center_after: torch.Tensor
    parked_after: torch.Tensor
    published_after: torch.Tensor
    slot_version_after: torch.Tensor
    device_fault_after: torch.Tensor
    reset_generation_after: torch.Tensor
    action_epoch_direct_state_after: _ActionEpochDirectState
    stale_witness: _PhysicalSelectedResetStaleWitness
    r06_armed_reset: object | None = None
    scene_apply_receipt: object | None = None
    r06_commit_token: object | None = None


@dataclass(frozen=True)
class _PreparedPhysicalGlobalDrain:
    """One exact physical row retained until abort or global ACK."""

    pack: object
    authority: object
    update_index: int
    completed_environment_steps: int
    mutation_version: int


@dataclass(frozen=True, eq=False)
class _PhysicalR10LiveMutationAck:
    """Owner-issued process-local proof of one exact global leaf ACK."""

    owner_identity: object
    authority: object
    receipt: object
    update_index: int
    completed_environment_steps: int
    drain_sequence: int
    mutation_version: int
    token: object


@dataclass(frozen=True)
class PhysicalDeviceInstallPrepareReceipt:
    """Host-portable identity for a CUDA-resident physical after-image.

    The record deliberately does not claim a portable snapshot of every live
    ball.  Exact live-scene/lifecycle preconditions remain device-resident and
    are carried by the physical boundary fault row through the sole packed
    reveal transfer.
    """

    schema_version: int
    kind: str
    integration_status: str
    capacity_receipt_sha256: str
    reveal_final_preview: _flight.CanonicalJsonContentPin
    num_envs: int
    reset_generations: tuple[int, ...]
    mutation_version_before: int
    prepare_nonce: int
    selected_env_ids: tuple[int, ...]
    selected_slot_indices: tuple[int, ...]
    install_payload_sha256s: tuple[str, ...]
    device_preconditions_bound_in_boundary_row: bool
    live_state_mutated: bool
    runtime_publication_created: bool

    KIND = "action_ball_physical_device_install_prepare_receipt_v1"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != self.KIND:
            raise PhysicalFlightDeviceError("device prepare receipt kind differs")
        if self.integration_status != _flight.INTEGRATION_STATUS:
            raise PhysicalFlightDeviceError(
                "device prepare integration status differs"
            )
        _sha256(self.capacity_receipt_sha256, label="capacity_receipt_sha256")
        if type(self.reveal_final_preview) is not _flight.CanonicalJsonContentPin:
            raise PhysicalFlightDeviceError("device prepare preview pin differs")
        _exact_int(self.num_envs, label="num_envs", minimum=1)
        _exact_int(
            self.mutation_version_before,
            label="mutation_version_before",
        )
        _exact_int(self.prepare_nonce, label="prepare_nonce", minimum=1)
        selected = tuple(self.selected_env_ids)
        slots = tuple(self.selected_slot_indices)
        payloads = tuple(self.install_payload_sha256s)
        resets = tuple(self.reset_generations)
        if (
            not selected
            or selected != tuple(sorted(set(selected)))
            or len(slots) != len(selected)
            or len(payloads) != len(selected)
            or len(resets) != self.num_envs
            or any(type(value) is not int or value < 0 for value in selected)
            or any(type(value) is not int or value < 0 for value in slots)
            or any(type(value) is not int or value < 1 for value in resets)
        ):
            raise PhysicalFlightDeviceError("device prepare row binding differs")
        for value in payloads:
            _sha256(value, label="install_payload_sha256")
        if (
            self.device_preconditions_bound_in_boundary_row is not True
            or self.live_state_mutated is not False
            or self.runtime_publication_created is not False
        ):
            raise PhysicalFlightDeviceError(
                "device prepare mutation/precondition claim differs"
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "selected_slot_indices", slots)
        object.__setattr__(self, "install_payload_sha256s", payloads)
        object.__setattr__(self, "reset_generations", resets)

    @property
    def canonical_sha256(self) -> str:
        return _flight.canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "integration_status": self.integration_status,
            "capacity_receipt_sha256": self.capacity_receipt_sha256,
            "reveal_final_preview": self.reveal_final_preview.to_mapping(),
            "num_envs": self.num_envs,
            "reset_generations": list(self.reset_generations),
            "mutation_version_before": self.mutation_version_before,
            "prepare_nonce": self.prepare_nonce,
            "selected_env_ids": list(self.selected_env_ids),
            "selected_slot_indices": list(self.selected_slot_indices),
            "install_payload_sha256s": list(self.install_payload_sha256s),
            "device_preconditions_bound_in_boundary_row": (
                self.device_preconditions_bound_in_boundary_row
            ),
            "live_state_mutated": self.live_state_mutated,
            "runtime_publication_created": self.runtime_publication_created,
        }


@dataclass(frozen=True)
class PhysicalDeviceInstallAbortReceipt:
    schema_version: int
    kind: str
    prepare_receipt_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    live_state_mutated: bool
    runtime_publication_created: bool

    KIND = "action_ball_physical_device_install_abort_receipt_v1"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != self.KIND:
            raise PhysicalFlightDeviceError("device abort receipt kind differs")
        _sha256(self.prepare_receipt_sha256, label="prepare_receipt_sha256")
        if (
            self.mutation_version_before != self.mutation_version_after
            or self.live_state_mutated is not False
            or self.runtime_publication_created is not False
        ):
            raise PhysicalFlightDeviceError("device abort mutation claim differs")

    @property
    def canonical_sha256(self) -> str:
        return _flight.canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "prepare_receipt_sha256": self.prepare_receipt_sha256,
            "mutation_version_before": self.mutation_version_before,
            "mutation_version_after": self.mutation_version_after,
            "live_state_mutated": self.live_state_mutated,
            "runtime_publication_created": self.runtime_publication_created,
        }


@dataclass(frozen=True)
class PhysicalDeviceInstallTerminalReceipt:
    """Completion evidence for a resident CUDA state, not a numeric snapshot."""

    schema_version: int
    kind: str
    prepare_receipt: PhysicalDeviceInstallPrepareReceipt
    prepare_receipt_sha256: str
    decision: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_sha256: str
    physical_boundary_fault_schema_sha256: str
    r05_terminal_claim_sha256: str
    r05_terminal_kind: str
    r05_terminal_sha256: str
    mutation_version_before: int
    mutation_version_after: int
    selected_env_ids: tuple[int, ...]
    selected_slot_indices: tuple[int, ...]
    install_payload_sha256s: tuple[str, ...]
    device_state_resident: bool
    scene_state_mutated: bool
    runtime_publication_created: bool
    policy_opportunity_created: bool

    KIND = "action_ball_physical_device_install_terminal_receipt_v1"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != self.KIND:
            raise PhysicalFlightDeviceError("device terminal receipt kind differs")
        if type(self.prepare_receipt) is not PhysicalDeviceInstallPrepareReceipt:
            raise PhysicalFlightDeviceError("device terminal prepare receipt differs")
        for name in (
            "prepare_receipt_sha256",
            "global_boundary_receipt_sha256",
            "global_boundary_packet_sha256",
            "physical_boundary_fault_schema_sha256",
            "r05_terminal_claim_sha256",
            "r05_terminal_sha256",
        ):
            _sha256(getattr(self, name), label=name)
        if (
            self.prepare_receipt_sha256
            != self.prepare_receipt.canonical_sha256
            or self.mutation_version_before
            != self.prepare_receipt.mutation_version_before
            or self.mutation_version_after != self.mutation_version_before + 1
            or self.selected_env_ids
            != self.prepare_receipt.selected_env_ids
            or self.selected_slot_indices
            != self.prepare_receipt.selected_slot_indices
            or self.install_payload_sha256s
            != self.prepare_receipt.install_payload_sha256s
            or self.device_state_resident is not True
        ):
            raise PhysicalFlightDeviceError("device terminal receipt binding differs")
        accept = self.decision == _reveal_boundary.DECISION_ACCEPT
        censor = self.decision == _reveal_boundary.DECISION_CENSOR
        if not (accept or censor):
            raise PhysicalFlightDeviceError("device terminal decision differs")
        expected_terminal_kind = (
            _r05.CommittedRevealBatch.KIND
            if accept
            else _r05.CensoredRevealBatch.KIND
        )
        if (
            self.r05_terminal_kind != expected_terminal_kind
            or
            self.scene_state_mutated is not accept
            or self.runtime_publication_created is not accept
            or self.policy_opportunity_created is not accept
        ):
            raise PhysicalFlightDeviceError(
                "device terminal decision/publication claim differs"
            )

    @property
    def canonical_sha256(self) -> str:
        return _flight.canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "prepare_receipt_sha256": self.prepare_receipt_sha256,
            "decision": self.decision,
            "global_boundary_receipt_sha256": (
                self.global_boundary_receipt_sha256
            ),
            "global_boundary_packet_sha256": self.global_boundary_packet_sha256,
            "physical_boundary_fault_schema_sha256": (
                self.physical_boundary_fault_schema_sha256
            ),
            "r05_terminal_claim_sha256": self.r05_terminal_claim_sha256,
            "r05_terminal_kind": self.r05_terminal_kind,
            "r05_terminal_sha256": self.r05_terminal_sha256,
            "mutation_version_before": self.mutation_version_before,
            "mutation_version_after": self.mutation_version_after,
            "selected_env_ids": list(self.selected_env_ids),
            "selected_slot_indices": list(self.selected_slot_indices),
            "install_payload_sha256s": list(self.install_payload_sha256s),
            "device_state_resident": self.device_state_resident,
            "scene_state_mutated": self.scene_state_mutated,
            "runtime_publication_created": self.runtime_publication_created,
            "policy_opportunity_created": self.policy_opportunity_created,
        }


@dataclass(frozen=True)
class PreparedPhysicalFlightInstall:
    """Opaque single-use child after-image produced by prepare."""

    _prepare_receipt: (
        _flight.PhysicalInstallPrepareReceipt
        | PhysicalDeviceInstallPrepareReceipt
    )
    _owner_identity: object
    _token: object

    @property
    def prepare_receipt(
        self,
    ) -> _flight.PhysicalInstallPrepareReceipt | PhysicalDeviceInstallPrepareReceipt:
        return self._prepare_receipt

    @property
    def canonical_sha256(self) -> str:
        return self._prepare_receipt.canonical_sha256


@dataclass(frozen=True, eq=False)
class ArmedPhysicalFlightInstall:
    """Opaque single-use handle for one fully prevalidated child terminal."""

    _prepared_install: PreparedPhysicalFlightInstall
    _decision: str
    _owner_identity: object
    _token: object


@dataclass(frozen=True, eq=False)
class PhysicalChildTerminalToken:
    """Opaque proof that this leaf executed its prevalidated terminal swap.

    It intentionally carries no portable success semantics.  The portable
    ACCEPT/CENSOR receipt remains private until the exact R05 terminal receipt
    is acknowledged by :meth:`complete_global_reveal_epoch`.
    """

    _armed_install: ArmedPhysicalFlightInstall
    _decision: str
    _owner_identity: object
    _token: object


class PhysicalHotPreparedInstall:
    """Empty identity for a fully prevalidated Physical hot after-image.

    The object carries no numeric fields.  Device-R05, the Physical question
    owner, and Motion chronology remain in the Physical owner's private
    registry, so replacing ``t_effective`` or a candidate row cannot be hidden
    in a caller-authored dataclass.
    """

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("Physical hot prepares are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Physical hot prepares are immutable")

    def __copy__(self):
        raise TypeError("Physical hot prepares cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Physical hot prepares cannot be copied")

    def __reduce__(self):
        raise TypeError("Physical hot prepares cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Physical hot prepares cannot be serialized")


class PhysicalHotChildCommitToken:
    """Empty proof that reveal retained one after-image without a scene write."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("Physical hot commit tokens are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Physical hot commit tokens are immutable")


class PhysicalLateLaunchPublication:
    """Empty, one-shot identity for an exact launch-tick scene publication."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("Physical late-launch publications are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Physical late-launch publications are immutable")

    def __copy__(self):
        raise TypeError("Physical late-launch publications cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Physical late-launch publications cannot be copied")

    def __reduce__(self):
        raise TypeError("Physical late-launch publications cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Physical late-launch publications cannot be serialized")


@dataclass(frozen=True)
class PhysicalLateLaunchPublicationView:
    """Clone-only R06 ingress after the Physical scene owner wrote at launch.

    ``contact_deadline_control_step`` is the question-owned complete Motion
    task close, never C01's short policy-opportunity close.  ``selected_contact``
    is absent because a normal
    no-contact launch is telemetry, never an infrastructure fault.
    """

    device_r05_projection_identity: object
    physical_publication_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor
    flight_slot: torch.Tensor
    full_key_sha256: torch.Tensor
    task_sha256: torch.Tensor
    action_uid: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_slot: torch.Tensor
    candidate_identity: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_ticks: torch.Tensor
    effective_contact_horizon_s: torch.Tensor
    physical_state_f32: torch.Tensor
    producer_fault: torch.Tensor


@dataclass(frozen=True)
class _PhysicalHotPreparedRecord:
    capability: PhysicalHotPreparedInstall
    device_r05_projection: object
    device_r05_projection_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor
    flight_slot: torch.Tensor
    full_key_sha256: torch.Tensor
    task_sha256: torch.Tensor
    action_uid: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_slot: torch.Tensor
    candidate_identity: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_ticks: torch.Tensor
    effective_contact_horizon_s: torch.Tensor
    physical_state_f32: torch.Tensor
    producer_fault: torch.Tensor
    prearm_fault: torch.Tensor
    scene_state_before: torch.Tensor
    owner_mutation_version: int
    stage: str


@dataclass(frozen=True)
class _PhysicalHotCommitRecord:
    capability: PhysicalHotChildCommitToken
    prepared: PhysicalHotPreparedInstall
    device_r05_projection: object
    stage: str


@dataclass(frozen=True)
class _PhysicalLateLaunchRecord:
    publication: PhysicalLateLaunchPublication
    prepared: PhysicalHotPreparedInstall
    device_r05_projection: object
    physical_publication_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor
    flight_slot: torch.Tensor
    full_key_sha256: torch.Tensor
    task_sha256: torch.Tensor
    action_uid: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_slot: torch.Tensor
    candidate_identity: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_ticks: torch.Tensor
    effective_contact_horizon_s: torch.Tensor
    physical_state_f32: torch.Tensor
    producer_fault: torch.Tensor
    stage: str


@dataclass(frozen=True)
class _PreparedPhysicalInstallImage:
    """Owner-private after-image; never returned with the public token."""

    after_slots: tuple[_flight.PhysicalFlightSlotSnapshot, ...]
    after_scene_state: torch.Tensor
    selected_mask: torch.Tensor
    prepare_fault_by_env: torch.Tensor
    install_payloads: tuple[_flight.PhysicalBallInstallPayload, ...]
    scene_handle: object
    stale_witness: "_PhysicalPrearmStaleWitness"


@dataclass(frozen=True)
class _PhysicalPrearmStaleWitness:
    scene_state: torch.Tensor
    lifecycle: torch.Tensor
    generation: torch.Tensor
    outcome_sha: torch.Tensor
    install_sha: torch.Tensor
    installed_state_sha: torch.Tensor
    reveal_step: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center: torch.Tensor
    parked: torch.Tensor
    published: torch.Tensor
    slot_version: torch.Tensor
    device_fault: torch.Tensor
    owner_mutation_version: int
    next_prepare_nonce: int


@dataclass(frozen=True)
class _ArmedPhysicalInstallImage:
    """Owner-private, allocation-complete physical child after-image."""

    after_slots: tuple[_flight.PhysicalFlightSlotSnapshot, ...]
    scene_handle: object | None
    decision: str
    global_boundary_receipt: object
    r05_terminal_claim: object
    r05_terminal_owner: object
    portable_receipt: (
        _flight.PhysicalInstallCommitReceipt
        | _flight.PhysicalInstallCensorReceipt
        | PhysicalDeviceInstallTerminalReceipt
    )
    lifecycle: torch.Tensor
    generation: torch.Tensor
    outcome_sha: torch.Tensor
    install_sha: torch.Tensor
    installed_state_sha: torch.Tensor
    reveal_step: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center: torch.Tensor
    parked: torch.Tensor
    published: torch.Tensor
    slot_version: torch.Tensor
    mutation_version: int
    next_prepare_nonce: int
    child_terminal_token: PhysicalChildTerminalToken
    global_epoch_image: "_GlobalRevealEpochImage"


@dataclass(frozen=True)
class _GlobalRevealEpochImage:
    """Owner-private completion witness retained after child publication."""

    decision: str
    r05_terminal_claim: object
    r05_terminal_owner: object
    expected_terminal_kind: str
    expected_terminal_sha256: str
    expected_claim_sha256: str
    expected_preview_sha256: str
    expected_boundary_receipt_sha256: str
    expected_packet_sha256: str
    expected_terminal_boundary_authority_sha256: str
    expected_terminal_boundary_projection_sha256: str
    expected_terminal_content_pin_sha256: str
    expected_selected_env_ids: tuple[int, ...]
    portable_receipt: (
        _flight.PhysicalInstallCommitReceipt
        | _flight.PhysicalInstallCensorReceipt
        | PhysicalDeviceInstallTerminalReceipt
    )


@dataclass(frozen=True)
class _PostPhysicsImage:
    """Owner-private binding immune to mutation of the R06-facing packet."""

    observe_mask: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    observation_ordinal: torch.Tensor
    current_state_f32: torch.Tensor
    current_ball_center_m: torch.Tensor
    observation_control_step: torch.Tensor
    observation_physics_substep: torch.Tensor
    owner_join_fault: torch.Tensor
    physical_orphan_mask: torch.Tensor
    r06_orphan_mask: torch.Tensor
    r06_owner_mutation_version_before: int | torch.Tensor
    r06_owner: object
    r06_batch: object
    r06_flight_state_before: torch.Tensor
    r06_mailbox_slot_before: torch.Tensor


@dataclass(frozen=True)
class _PhysicalContactRewardViewRecord:
    """Owner-private identity behind one issued Reward view."""

    view: PhysicalSelectedContactRewardView
    view_issued: bool


@dataclass(frozen=True)
class _AcknowledgedR06Image:
    """Owner-private R06 after-image consumed only by settle/retire."""

    flight_state: torch.Tensor
    full_key_sha256: torch.Tensor
    ball_generation: torch.Tensor
    mailbox_slot: torch.Tensor
    observation_ordinal: torch.Tensor
    current_state_f32: torch.Tensor
    owner_mutation_version: int | torch.Tensor
    snapshot_root_sha256: str | None
    settlement_authority: Optional[_flight.CanonicalJsonContentPin]
    r06_owner: object | None
    terminal_fault_mask: torch.Tensor
    cleanup_only_mask: torch.Tensor
    physical_orphan_mask: torch.Tensor
    r06_orphan_mask: torch.Tensor
    r06_settled_mask: torch.Tensor
    settled_mask: torch.Tensor


@dataclass(frozen=True)
class _PhysicalRetireStaleWitness:
    """Complete physical prearm image checked before R06 becomes armed."""

    scene_state: torch.Tensor
    lifecycle: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center: torch.Tensor
    parked: torch.Tensor
    published: torch.Tensor
    slot_version: torch.Tensor
    device_fault: torch.Tensor
    owner_mutation_version: int
    postphysics_result: object


@dataclass(frozen=True)
class _PreparedPhysicalSettleRetireImage:
    """Private scene/metadata after-image retained across R06 arm and commit."""

    postphysics_result: object
    r06_predicted_result: object
    r06_armed_result: object | None
    claim: object
    stale_witness: _PhysicalRetireStaleWitness
    park_mask: torch.Tensor
    physical_cleanup_mask: torch.Tensor
    scene_handle: object | None
    after_slots: tuple[_flight.PhysicalFlightSlotSnapshot, ...]
    lifecycle: torch.Tensor
    observation_ordinal: torch.Tensor
    previous_ball_center: torch.Tensor
    parked: torch.Tensor
    published: torch.Tensor
    slot_version: torch.Tensor
    device_fault: torch.Tensor
    portable_receipt: Optional[_flight.PhysicalRetireReceipt]


def _stamp_grid(
    value: object,
    *,
    label: str,
    shape: tuple[int, int],
    device: torch.device,
) -> PhysicsStampGrid:
    if type(value) is not PhysicsStampGrid:
        raise PhysicalFlightDeviceError(f"{label} must be PhysicsStampGrid")
    return PhysicsStampGrid(
        control_step=_tensor(
            value.control_step,
            label=f"{label}.control_step",
            shape=shape,
            dtype=torch.int64,
            device=device,
        ),
        physics_substep=_tensor(
            value.physics_substep,
            label=f"{label}.physics_substep",
            shape=shape,
            dtype=torch.int32,
            device=device,
        ),
        event_phase=_tensor(
            value.event_phase,
            label=f"{label}.event_phase",
            shape=shape,
            dtype=torch.int8,
            device=device,
        ),
    )


class ActionBallPhysicalFlightDeviceOwner:
    """Explicit-capacity physical owner with no legacy scene fallback."""


    def __init__(
        self,
        *,
        num_envs: int,
        scene_body_names: Sequence[str],
        scene_port: object,
        capacity_receipt: _flight.FrozenFlightCapacityReceipt | None = None,
        expected_capacity_receipt_sha256: str | None = None,
        diagnostic_n2_capacity_binding: object | None = None,
        reset_genesis_authority: object | None = None,
        reset_genesis_receipt: object | None = None,
    ) -> None:
        _verify_contract_source()
        self._owner_identity = object()
        self.num_envs = _exact_int(num_envs, label="num_envs", minimum=1)
        formal_capacity = type(capacity_receipt) is _flight.FrozenFlightCapacityReceipt
        diagnostic_capacity = capacity_receipt is None and expected_capacity_receipt_sha256 is None
        if formal_capacity:
            if diagnostic_n2_capacity_binding is not None:
                raise PhysicalFlightDeviceError(
                    "formal and diagnostic capacity inputs are mutually exclusive"
                )
            expected = _sha256(
                expected_capacity_receipt_sha256,
                label="expected_capacity_receipt_sha256",
            )
            if capacity_receipt.canonical_sha256 != expected:
                raise PhysicalFlightDeviceError("capacity receipt external pin differs")
            self.capacity_receipt = capacity_receipt
            self.capacity_receipt_sha256 = expected
            self.flight_capacity = capacity_receipt.configured_flight_capacity
            self._diagnostic_n2_no_save = False
        elif diagnostic_capacity:
            try:
                scene_spec = getattr(scene_port, "spec")
                binding = _diagnostic_capacity.require_diagnostic_n2_capacity_binding(
                    diagnostic_n2_capacity_binding,
                    scene_spec=scene_spec,
                )
            except BaseException as exc:
                raise PhysicalFlightDeviceError(
                    "diagnostic N=2 capacity binding differs"
                ) from exc
            self.flight_capacity = binding.flight_capacity
            self._diagnostic_n2_no_save = True
        else:
            raise PhysicalFlightDeviceError(
                "physical owner requires exactly one formal or diagnostic capacity authority"
            )
        names = tuple(scene_body_names)
        if (
            len(names) != self.flight_capacity
            or len(set(names)) != len(names)
            or any(type(name) is not str or not name.strip() for name in names)
            or any(name == "pb_ball" or "PhysicalBallManager" in name for name in names)
        ):
            raise PhysicalFlightDeviceError("fresh scene body names differ")
        self.scene_body_names = names
        scene_type = type(scene_port)
        if scene_type is TensorPhysicalFlightScenePort:
            capability = scene_port.require_owned_scene_port_capability(
                scene_port.scene_port_capability
            )
        else:
            expected_module = (
                "whole_body_tracking.tasks.tracking.config.agibot_a3."
                "action_ball_full_mdp_ball_scene"
            )
            source = inspect.getsourcefile(scene_type)
            module = sys.modules.get(scene_type.__module__)
            if (
                scene_type.__name__ != "IsaacLabPhysicalFlightScenePort"
                or scene_type.__module__ != expected_module
                or source is None
                or module is None
                or getattr(module, "IsaacLabPhysicalFlightScenePort", None)
                is not scene_type
            ):
                raise PhysicalFlightDeviceError(
                    "scene port must be an exact reviewed concrete implementation"
                )
            capability = scene_port.require_owned_scene_port_capability(
                scene_port.scene_port_capability
            )
        for name, expected_value in (
            ("num_envs", self.num_envs),
            ("flight_capacity", self.flight_capacity),
        ):
            if getattr(scene_port, name, None) != expected_value:
                raise PhysicalFlightDeviceError(f"scene port {name} differs")
        if (
            getattr(capability, "num_envs", None) != self.num_envs
            or getattr(capability, "flight_capacity", None)
            != self.flight_capacity
            or not callable(getattr(scene_port, "read_state_env", None))
            or not callable(getattr(scene_port, "preflight_write", None))
            or not callable(getattr(scene_port, "apply_prevalidated_write", None))
            or not callable(getattr(scene_port, "require_owned_apply_receipt", None))
            or not callable(getattr(scene_port, "abort_prevalidated_write", None))
            or not callable(getattr(scene_port, "require_owned_abort_receipt", None))
        ):
            raise PhysicalFlightDeviceError("scene port capability/API differs")
        self.scene_port = scene_port
        self._scene_port_capability = capability
        capture_function = inspect.getattr_static(
            scene_type, "capture_post_physics_facts", None
        )
        if capture_function is not None and not inspect.isfunction(capture_function):
            raise PhysicalFlightDeviceError(
                "scene postphysics producer must be one direct concrete method"
            )
        # Retain the class dictionary identity.  A later instance/class patch,
        # proxy, or caller-supplied callable is not fact authority.
        self._postphysics_scene_capture_function = capture_function
        self.device = torch.device(scene_port.device)
        initial = _tensor(
            scene_port.read_state_env(),
            label="initial scene state",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        if self.device.type == "cpu" and not bool(torch.isfinite(initial).all()):
            raise PhysicalFlightDeviceError("initial scene state is nonfinite")
        self._park_state_template = initial.detach().clone()

        shape = (self.num_envs, self.flight_capacity)
        self._fixed_flight_slot_grid = torch.arange(
            self.flight_capacity, dtype=torch.int64, device=self.device
        ).unsqueeze(0).expand(shape)
        self._lifecycle = torch.zeros(shape, dtype=torch.int8, device=self.device)
        self._generation = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        self._outcome_sha = torch.zeros(shape + (TOKEN_BYTES,), dtype=torch.uint8, device=self.device)
        self._install_sha = torch.zeros_like(self._outcome_sha)
        self._installed_state_sha = torch.zeros_like(self._outcome_sha)
        self._reveal_step = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        self._observation_ordinal = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        self._previous_ball_center = initial[..., :3].clone()
        self._parked = torch.ones(shape, dtype=torch.bool, device=self.device)
        self._published = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._slot_version = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self._device_fault = torch.zeros(shape, dtype=torch.bool, device=self.device)

        if self.device.type != "cpu" or self._diagnostic_n2_no_save:
            # The CUDA hot path cannot materialize portable dataclasses without
            # the not-yet-wired packed boundary.  Keep an empty host projection
            # and fail before transaction/checkpoint methods use it.
            self._host_slots: tuple[_flight.PhysicalFlightSlotSnapshot, ...] = ()
        else:
            self._host_slots = tuple(
                self._parked_snapshot(
                    env_id=env_id,
                    slot_index=slot_index,
                    state=_canonical_state_from_cpu_tensor(initial[env_id, slot_index]),
                    version=0,
                )
                for env_id in range(self.num_envs)
                for slot_index in range(self.flight_capacity)
            )
        self._mutation_version = 0
        self._device_mutation_version = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )
        self._terminal_resolution_total = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )
        self._shared_normal_retire_total = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )
        self._physical_only_orphan_park_total = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )
        self._shared_normal_retire_key_summaries = torch.zeros(
            (2,), dtype=torch.int64, device=self.device
        )
        self._selected_contact_pending = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._selected_contact_pending_flight_slot = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._selected_contact_pending_full_key_sha256 = torch.zeros(
            (self.num_envs, TOKEN_BYTES), dtype=torch.uint8, device=self.device
        )
        self._selected_contact_pending_ball_generation = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._selected_contact_pending_observation_ordinal = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._selected_contact_pending_control_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._selected_contact_pending_physics_substep = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=self.device
        )
        self._selected_contact_pending_event_phase = torch.full(
            (self.num_envs,), -1, dtype=torch.int8, device=self.device
        )
        self._selected_contact_viewed = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._selected_contact_view_total = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )
        self._selected_contact_payment_total = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )
        self._selected_contact_ledger_fault = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._active_selected_contact_reward_view: (
            _PhysicalContactRewardViewRecord | None
        ) = None
        self._selected_contact_reward_cycle_open = False
        self._last_paid_selected_contact_reward_view: (
            PhysicalSelectedContactRewardView | None
        ) = None
        self._physical_reward_poisoned = False
        self._physical_reward_poison_reason: str | None = None
        self._diagnostic_paid_selected_contact_reward_verdicts: set[int] = set()
        self._active_physical_global_drain: (
            _PreparedPhysicalGlobalDrain | None
        ) = None
        self._physical_global_drain_sequence = 0
        self._physical_global_drain_last_update_index = -1
        self._physical_global_drain_last_completed_environment_steps = -1
        self._physical_global_drain_last_acknowledged_mutation_version = -1
        self._physical_checkpoint_requires_global_drain_ack = False
        self._physical_global_drain_authority_type: type[object] | None = None
        self._physical_global_drain_ack_method: object | None = None
        self._physical_global_drain_authority: object | None = None
        self._physical_global_drain_last_acknowledged_receipt: object | None = None
        self._physical_checkpoint_live_join_required = True
        self._physical_checkpoint_live_ack: _PhysicalR10LiveMutationAck | None = None
        self._physical_checkpoint_last_live_projection: object | None = None
        self._physical_checkpoint_last_live_boundary: object | None = None
        self._physical_checkpoint_last_live_receipt: object | None = None
        self._physical_global_drain_poisoned = False
        self._physical_global_drain_poison_reason: str | None = None
        self._next_prepare_nonce = 1
        self._active_prepare: PreparedPhysicalFlightInstall | None = None
        self._active_prepare_image: _PreparedPhysicalInstallImage | None = None
        self._active_armed: ArmedPhysicalFlightInstall | None = None
        self._active_armed_image: _ArmedPhysicalInstallImage | None = None
        self._active_child_terminal: PhysicalChildTerminalToken | None = None
        self._physical_hot_prepared_records: dict[
            PhysicalHotPreparedInstall, _PhysicalHotPreparedRecord
        ] = {}
        self._physical_hot_commit_records: dict[
            PhysicalHotChildCommitToken, _PhysicalHotCommitRecord
        ] = {}
        self._physical_late_launch_records: dict[
            PhysicalLateLaunchPublication, _PhysicalLateLaunchRecord
        ] = {}
        self._active_physical_hot_prepare: PhysicalHotPreparedInstall | None = None
        self._active_physical_hot_commit: PhysicalHotChildCommitToken | None = None
        self._active_physical_late_launch: PhysicalLateLaunchPublication | None = None
        # Lean ActionEpoch lane.  These are exact construction identities and
        # owner-private retained task bytes, not readiness receipts.  The old
        # portable/R10 transaction remains untouched and HOLD in diagnostic
        # N=2 mode.
        self._action_epoch_owner: object | None = None
        self._action_epoch_device_r05_owner: object | None = None
        self._action_epoch_motion_owner: object | None = None
        self._action_epoch_racket_owner: object | None = None
        # ActionEpoch shot slots and Physical scene flight slots are distinct
        # dimensions (today S=1 and K=2).  Pending candidate rows are retained
        # as one full-N masked image; a peer publication never invalidates an
        # older row merely because the journal ordinal advanced.
        self._action_epoch_pending_launch: _ActionEpochPendingLaunchState | None = None
        self._action_epoch_active_flight_slot = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        empty_flight_key = _row_identity.empty_action_epoch_shot_key(
            shape, device=self.device
        )
        # ``_generation`` is Physical's one real swing/ball generation plane;
        # the typed key references it instead of retaining a duplicate writer.
        self._action_epoch_flight_shot_key = _row_identity.ActionEpochShotKey(
            reset_generation=empty_flight_key.reset_generation,
            ball_generation=self._generation,
            action_uid=empty_flight_key.action_uid,
            action_slot=empty_flight_key.action_slot,
            shot_index=empty_flight_key.shot_index,
            task_identity=empty_flight_key.task_identity,
            outcome_identity=empty_flight_key.outcome_identity,
            ball_identity=empty_flight_key.ball_identity,
        )
        self._action_epoch_flight_publication_ordinal = torch.full(
            shape, -1, dtype=torch.int64, device=self.device
        )
        self._action_epoch_active_scene_write: (
            ActionEpochSceneWriteProjection | None
        ) = None
        self._action_epoch_active_r06_launch: (
            ActionEpochR06LaunchProjection | None
        ) = None
        self._action_epoch_active_r06_postphysics: (
            ActionEpochR06PostPhysicsProjection | None
        ) = None
        self._action_epoch_active_physics_fact_allocation: (
            ActionEpochPhysicsFactAllocationProjection | None
        ) = None
        # One D2H reduction at the post-command control boundary decides
        # whether the *next* control step has any Physical/R06/scene work.
        # The per-substep pair kind remains host-only and may never cross a
        # pre/post boundary.
        self._action_epoch_host_activity_control_step: int | None = None
        self._action_epoch_host_activity_has_work = True
        self._action_epoch_substep_pair: str | None = None
        self._action_epoch_empty_env_mask = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._action_epoch_runtime_call_active = False
        self._active_global_reveal_epoch_image: (
            _GlobalRevealEpochImage | None
        ) = None
        self._active_postphysics: PhysicalPostPhysicsPublication | None = None
        self._active_postphysics_image: _PostPhysicsImage | None = None
        self._active_postphysics_capture: IsaacPostPhysicsFacts | None = None
        self._active_postphysics_capture_image: _PostPhysicsCaptureImage | None = None
        self._last_postphysics_exact_stamp: tuple[int, int, int, int, int] | None = None
        self._active_r06_ack: object | None = None
        self._active_r06_ack_image: _AcknowledgedR06Image | None = None
        self._r06_owner: object | None = None
        self._r06_contact_authority_consumer: object | None = None
        self._r06_park_token_authority: object | None = None
        self._active_physical_retire_prepare: (
            PreparedPhysicalSettleRetire | None
        ) = None
        self._active_physical_retire_image: (
            _PreparedPhysicalSettleRetireImage | None
        ) = None
        self._active_physical_retire_arm: ArmedPhysicalSettleRetire | None = None
        self._active_physical_retire_commit: (
            PhysicalSettleRetireCommitToken | None
        ) = None
        self._active_selected_reset_stage: (
            StagedPhysicalSelectedTrueReset | None
        ) = None
        self._active_selected_reset_stage_record: (
            _PhysicalSelectedResetStageRecord | None
        ) = None
        self._active_selected_reset_finalize: (
            FinalizedPhysicalSelectedTrueReset | None
        ) = None
        self._active_selected_reset_arm: (
            ArmedPhysicalSelectedTrueReset | None
        ) = None
        self._active_selected_reset_commit: (
            PhysicalSelectedTrueResetParkCommitToken | None
        ) = None
        self._active_selected_reset_image: (
            _PhysicalSelectedResetImage | None
        ) = None
        self._selected_reset_completion_token: (
            PhysicalSelectedTrueResetCompletionToken | None
        ) = None
        self._selected_reset_completion_record: (
            _PhysicalSelectedResetCompletionRecord | None
        ) = None
        self._reveal_boundary_child_authority = (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority(
                owner_kind="physical_ball",
                validator=self._require_owned_reveal_boundary_prepared_token,
            )
        )
        self._reveal_boundary_owner: Optional[
            _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner
        ] = None
        self._reveal_boundary_lane_authority: Optional[
            _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority
        ] = None
        self._r05_terminal_owner: Optional[
            _r05.ContinuousRuntimeTransactionOwner
        ] = None
        self._device_r05_reset_owner: object | None = None
        self._device_r05_prepared_reset_validator: object | None = None
        self._device_r05_receipt_validator: object | None = None
        self._r06_selected_reset_park_token_authority: object | None = None
        self._active_reveal_boundary_row: Optional[
            _reveal_boundary.ActionBallFullMdpRevealBoundaryDeviceRow
        ] = None
        self._poisoned = False
        self._poison_reason: str | None = None
        if (
            type(reset_genesis_authority)
            is not _reset_genesis.ActionBallFullMdpResetGenesisAuthority
            or type(reset_genesis_receipt)
            is not _reset_genesis.ActionBallFullMdpResetGenesisReceipt
        ):
            raise PhysicalFlightDeviceError(
                "Physical reset genesis authority/receipt differs"
            )
        try:
            genesis_projection = (
                reset_genesis_authority.require_owned_physical_genesis(
                    reset_genesis_receipt,
                    device=self.device,
                    num_envs=self.num_envs,
                )
            )
        except BaseException as exc:
            raise PhysicalFlightDeviceError(
                "Physical reset genesis projection failed"
            ) from exc
        if (
            type(genesis_projection)
            is not _reset_genesis.PhysicalResetGenesisProjection
            or genesis_projection.world_reset_identity is None
        ):
            raise PhysicalFlightDeviceError(
                "Physical reset genesis projection type differs"
            )
        generation = genesis_projection.reset_generations
        if (
            type(generation) is not torch.Tensor
            or generation.shape != (self.num_envs,)
            or generation.dtype != torch.int64
            or generation.device != self.device
        ):
            raise PhysicalFlightDeviceError(
                "Physical reset genesis must be device int64[N]"
            )
        valid_generation = torch.logical_and(
            generation >= 1,
            generation < torch.iinfo(torch.int64).max,
        ).all()
        if self.device.type == "cpu":
            if not bool(valid_generation):
                raise PhysicalFlightDeviceError(
                    "Physical reset genesis has no positive int64 continuation"
                )
        else:
            # `_assert_async` alone is not authorization.  This cold
            # constructor synchronizes the exact device so an invalid
            # independent projection blocks before the owner escapes,
            # without copying the tensor to host memory.
            torch._assert_async(valid_generation)
            torch.cuda.synchronize(self.device)
        self._device_reset_generation = generation.detach().clone()
        self._reset_generation = []
        self._host_reset_generation_projection_current = False
        self._genesis_world_reset_identity = (
            genesis_projection.world_reset_identity
        )

    def boundary_child_token_authority(
        self,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority:
        """Return the stable validator capability registered before boundary construction."""

        return self._reveal_boundary_child_authority

    def reveal_boundary_fault_schema(
        self,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryFaultSchema:
        """Return the exact physical lane fault-bit schema."""

        return _PHYSICAL_REVEAL_FAULT_SCHEMA

    def bind_reveal_boundary_owner(
        self,
        boundary_owner: _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner,
    ) -> None:
        """One-time bind to the exact owner/lane built with this child authority."""

        self._require_operable()
        if self._reveal_boundary_owner is not None:
            self._poisoned = True
            self._poison_reason = "physical reveal boundary owner cannot be rebound"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        if type(boundary_owner) is not _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner:
            raise PhysicalFlightDeviceError("physical reveal boundary owner type differs")
        physical_index = _reveal_boundary.OWNER_ORDER.index("physical_ball")
        lane = boundary_owner.lane_authority("physical_ball")
        if (
            boundary_owner.num_envs != self.num_envs
            or boundary_owner.device != self.device
            or boundary_owner.child_token_authorities[physical_index]
            is not self._reveal_boundary_child_authority
            or boundary_owner.owner_fault_schemas[physical_index]
            is not _PHYSICAL_REVEAL_FAULT_SCHEMA
            or type(lane)
            is not _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority
            or lane.child_token_authority
            is not self._reveal_boundary_child_authority
            or lane.fault_schema is not _PHYSICAL_REVEAL_FAULT_SCHEMA
        ):
            raise PhysicalFlightDeviceError(
                "physical reveal boundary owner/lane registration differs"
            )
        self._reveal_boundary_owner = boundary_owner
        self._reveal_boundary_lane_authority = lane

    def bind_r05_terminal_owner(
        self,
        r05_owner: _r05.ContinuousRuntimeTransactionOwner,
    ) -> None:
        """One-time bind to the R05 owner that retains terminal claim identity."""

        self._require_operable()
        if self._r05_terminal_owner is not None:
            self._poisoned = True
            self._poison_reason = "physical R05 terminal owner cannot be rebound"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        if type(r05_owner) is not _r05.ContinuousRuntimeTransactionOwner:
            raise PhysicalFlightDeviceError("physical R05 terminal owner type differs")
        self._r05_terminal_owner = r05_owner

    def bind_device_r05_reset_owner(
        self,
        device_r05_owner: _r05_device.DeviceR05Owner,
        *,
        prepared_reset_validator: object,
        r05_receipt_validator: object,
    ) -> None:
        """Bind one Device-R05 owner through its independent genesis fact."""

        self._require_operable(allow_diagnostic_scene_observation=True)
        if self._device_r05_reset_owner is not None:
            raise PhysicalFlightDeviceError(
                "physical device-R05 reset owner cannot be rebound"
            )
        if self._mutation_version != 0 or self._next_prepare_nonce != 1:
            raise PhysicalFlightDeviceError(
                "physical Device-R05 reset owner must bind before business mutation"
            )
        self._require_checkpoint_idle()
        # Device-R05 state-view properties deliberately close its
        # construction window.  The exact opaque genesis view below is the
        # sole construction-safe source of the [N] shape and device facts.
        if type(device_r05_owner) is not _r05_device.DeviceR05Owner:
            raise PhysicalFlightDeviceError(
                "physical selected reset requires the exact device-R05 owner"
            )
        try:
            genesis_projection = (
                device_r05_owner.project_owned_genesis_for_child(
                    owner_kind="physical_ball"
                )
            )
            genesis_view = device_r05_owner.require_owned_genesis_projection(
                genesis_projection,
                owner_kind="physical_ball",
            )
            repeated_genesis_view = (
                device_r05_owner.require_owned_genesis_projection(
                    genesis_projection,
                    owner_kind="physical_ball",
                )
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "physical/device-R05 genesis authority differs"
            ) from exc
        projection_type = getattr(_r05_device, "DeviceR05GenesisProjection", None)
        view_type = getattr(_r05_device, "DeviceR05GenesisView", None)
        generation = getattr(genesis_view, "reset_generation", None)
        if (
            projection_type is None
            or view_type is None
            or type(genesis_projection) is not projection_type
            or type(genesis_view) is not view_type
            or type(repeated_genesis_view) is not view_type
            or getattr(genesis_view, "device_r05_owner", None)
            is not device_r05_owner
            or getattr(genesis_view, "owner_kind", None)
            != "physical_ball"
            or getattr(genesis_view, "world_reset_identity", None)
            is None
            or genesis_view.world_reset_identity
            is not self._genesis_world_reset_identity
            or repeated_genesis_view.device_r05_owner
            is not device_r05_owner
            or repeated_genesis_view.owner_kind != "physical_ball"
            or repeated_genesis_view.world_reset_identity
            is not genesis_view.world_reset_identity
            or not torch.equal(
                repeated_genesis_view.reset_generation,
                generation,
            )
            or not isinstance(generation, torch.Tensor)
            or tuple(generation.shape) != (self.num_envs,)
            or generation.dtype != torch.int64
            or generation.device != self.device
            or not callable(prepared_reset_validator)
            or getattr(prepared_reset_validator, "__self__", None)
            is not device_r05_owner
            or getattr(prepared_reset_validator, "__func__", None)
            is not getattr(
                type(device_r05_owner),
                "require_owned_prepared_true_reset",
                None,
            )
            or not callable(r05_receipt_validator)
            or getattr(r05_receipt_validator, "__self__", None)
            is not device_r05_owner
            or getattr(r05_receipt_validator, "__func__", None)
            is not getattr(
                type(device_r05_owner),
                "require_owned_true_reset_receipt",
                None,
            )
        ):
            raise PhysicalFlightDeviceError(
                "physical/device-R05 construction authority differs"
            )
        self._device_reset_generation = generation.detach().clone()
        # Only the global drain may ever materialize a future host view.
        self._host_reset_generation_projection_current = False
        self._device_r05_reset_owner = device_r05_owner
        self._device_r05_prepared_reset_validator = prepared_reset_validator
        self._device_r05_receipt_validator = r05_receipt_validator

    def bind_device_r05_hot_reveal_owner(
        self,
        device_r05_owner: object,
        *,
        physical_question_owner: object,
        motion_contact_launch_authority: object,
        full_key_authority: object,
    ) -> None:
        """Fail closed until the exact late-launch receipt graph is complete.

        Device-R05 currently exposes the exact ``physical_ball`` child row,
        but not the ``full_key_sha256``/task digest/action UID/flight slot
        needed by R06.  The Physical question producer exposes a numerical
        final batch but no owner-issued final receipt validator, and Motion has
        not frozen the exact contact/launch chronology receipt.  Accepting
        those values as caller tensors here would be a decorative gate, not a
        causal authority join.
        """

        del (
            device_r05_owner,
            physical_question_owner,
            motion_contact_launch_authority,
            full_key_authority,
        )
        raise PhysicalLateLaunchProductionHold(
            "production Physical hot/late-launch remains HOLD: Device-R05 "
            "full-key/task/action/slot identity, an owner-issued Physical "
            "question final receipt, and Motion exact contact/launch receipt "
            "are not frozen"
        )

    def bind_action_epoch_owner(
        self,
        epoch_owner: object,
        *,
        device_r05_owner: object,
        motion_owner: object,
        racket_owner: object,
    ) -> None:
        """Cold-bind the exact lean epoch, D05 source, and Motion clock.

        Object identity is used only to retain the sole producers.  Every hot
        value is re-read through D05's exact full-N ACCEPT method or Motion's
        exact current observation method.  No receipt, source SHA, caller
        tensor, or caller verdict authorizes the scene write.
        """

        if (
            self._action_epoch_owner is epoch_owner
            and self._action_epoch_device_r05_owner is device_r05_owner
            and self._action_epoch_motion_owner is motion_owner
            and self._action_epoch_racket_owner is racket_owner
        ):
            return
        if (
            self._action_epoch_owner is not None
            or self._action_epoch_device_r05_owner is not None
            or self._action_epoch_motion_owner is not None
            or self._action_epoch_racket_owner is not None
            or self._mutation_version != 0
            or self._action_epoch_pending_launch is not None
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch sources may not be rebound or bound after mutation"
            )
        try:
            import action_ball_continuous_runtime_transaction_device as device_r05
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_continuous_runtime_transaction_device as device_r05
            import action_ball_full_mdp_epoch as epoch

        d05_projector = getattr(
            device_r05_owner, "require_owned_action_epoch_accepted", None
        )
        motion_projector = getattr(
            motion_owner, "action_ball_continuous_motion_observation_projection", None
        )
        motion_validator = getattr(
            motion_owner, "require_owned_action_ball_continuous_motion_observation", None
        )
        racket_physical_binder = getattr(
            racket_owner,
            "bind_action_ball_full_mdp_racket_selected_rubber_physical_owner",
            None,
        )
        direct_racket_physical_binder = getattr(
            type(racket_owner),
            "bind_action_ball_full_mdp_racket_selected_rubber_physical_owner",
            None,
        )
        racket_face_view = getattr(
            racket_owner, "action_ball_full_mdp_action_epoch_selected_rubber_view", None
        )
        direct_racket_face_view = getattr(
            type(racket_owner),
            "action_ball_full_mdp_action_epoch_selected_rubber_view",
            None,
        )
        required_epoch_methods = (
            "current",
            "poison_owner_write",
            "bind_fact_owner",
            "publish_owner_facts",
            "merge_runtime_owner_fault",
            "bind_async_owner",
            "refresh_physical_launch_rows",
            "refresh_physical_postphysics_rows",
        )
        if (
            type(epoch_owner) is not epoch.ActionEpochOwner
            or type(device_r05_owner) is not device_r05.DeviceR05Owner
            or epoch_owner.num_envs != self.num_envs
            or epoch_owner.shot_slot_capacity < 1
            or epoch_owner.device != self.device
            or getattr(device_r05_owner, "_num_envs", None) != self.num_envs
            or torch.device(getattr(device_r05_owner, "_device", "cpu"))
            != self.device
            or getattr(motion_owner, "num_envs", None) != self.num_envs
            or torch.device(getattr(motion_owner, "device", "cpu"))
            != self.device
            or any(
                not callable(getattr(epoch_owner, name, None))
                or getattr(getattr(epoch_owner, name), "__self__", None)
                is not epoch_owner
                or getattr(getattr(epoch_owner, name), "__func__", None)
                is not getattr(epoch.ActionEpochOwner, name, None)
                for name in required_epoch_methods
            )
            or not callable(d05_projector)
            or getattr(d05_projector, "__self__", None) is not device_r05_owner
            or getattr(d05_projector, "__func__", None)
            is not device_r05.DeviceR05Owner.require_owned_action_epoch_accepted
            or not callable(motion_projector)
            or getattr(motion_projector, "__self__", None) is not motion_owner
            or getattr(motion_projector, "__func__", None)
            is not getattr(type(motion_owner), "action_ball_continuous_motion_observation_projection", None)
            or not callable(motion_validator)
            or getattr(motion_validator, "__self__", None) is not motion_owner
            or getattr(motion_validator, "__func__", None)
            is not getattr(type(motion_owner), "require_owned_action_ball_continuous_motion_observation", None)
            or not callable(racket_physical_binder)
            or not callable(direct_racket_physical_binder)
            or getattr(racket_physical_binder, "__self__", None) is not racket_owner
            or getattr(racket_physical_binder, "__func__", None)
            is not direct_racket_physical_binder
            or not callable(racket_face_view)
            or not callable(direct_racket_face_view)
            or getattr(racket_face_view, "__self__", None) is not racket_owner
            or getattr(racket_face_view, "__func__", None) is not direct_racket_face_view
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch/D05/Motion construction sources differ"
            )
        required_scene_methods = (
            "bind_action_epoch_scene_writer",
            "preflight_action_epoch_write",
            "arm_action_epoch_physics_fact_source",
            "action_epoch_physics_fact_activity_mask",
            "begin_action_epoch_idle_physics_fact_source",
            "complete_action_epoch_idle_physics_fact_source",
        )
        scene_type = type(self.scene_port)
        if scene_type is not TensorPhysicalFlightScenePort and any(
            not callable(getattr(self.scene_port, name, None))
            or getattr(getattr(self.scene_port, name), "__self__", None)
            is not self.scene_port
            or getattr(getattr(self.scene_port, name), "__func__", None)
            is not getattr(scene_type, name, None)
            for name in required_scene_methods
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch scene writer/fact-source API is absent or patched"
            )
        scene_binder = getattr(
            self.scene_port, "bind_action_epoch_scene_writer", None
        )
        self._action_epoch_owner = epoch_owner
        self._action_epoch_device_r05_owner = device_r05_owner
        self._action_epoch_motion_owner = motion_owner
        self._action_epoch_racket_owner = racket_owner
        if type(self.scene_port) is not TensorPhysicalFlightScenePort:
            try:
                scene_binder(self, epoch_owner)
            except BaseException as exc:
                self._action_epoch_owner = None
                self._action_epoch_device_r05_owner = None
                self._action_epoch_motion_owner = None
                self._action_epoch_racket_owner = None
                raise PhysicalEpochIntegrationHold(
                    "Physical/scene ActionEpoch bind failed"
                ) from exc
        try:
            racket_physical_binder(self)
            if type(self.scene_port) is not TensorPhysicalFlightScenePort:
                scene_fact_binder = self.scene_port.bind_action_epoch_physics_fact_source
                scene_fact_binder(
                    physical_owner=self,
                    epoch_owner=epoch_owner,
                    racket_owner=racket_owner,
                )
            epoch_owner.bind_fact_owner("physical_ball", self)
            epoch_owner.bind_async_owner("physical_ball", self)
        except BaseException as exc:
            self._poisoned = True
            self._poison_reason = (
                "Physical ActionEpoch fact bind failed after scene bind"
            )
            raise PhysicalEpochIntegrationHold(self._poison_reason) from exc

    def action_epoch_scene_write_projection(
        self,
    ) -> ActionEpochSceneWriteProjection:
        """Return the sole active one-shot scene after-image without cloning."""

        view = self._action_epoch_active_scene_write
        if (
            type(view) is not ActionEpochSceneWriteProjection
            or view.physical_owner is not self
            or view.epoch_owner is not self._action_epoch_owner
            or view._owner_identity is not self._owner_identity
            or view._token is not _ACTION_EPOCH_SCENE_WRITE_TOKEN
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical has no active ActionEpoch scene write"
            )
        return view

    def require_owned_action_epoch_scene_write_projection(
        self, view: object
    ) -> ActionEpochSceneWriteProjection:
        """Validate the exact active Physical-owned scene after-image."""

        if view is not self._action_epoch_active_scene_write:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch scene write projection is stale or foreign"
            )
        return self.action_epoch_scene_write_projection()

    def action_epoch_r06_launch_projection(
        self,
    ) -> ActionEpochR06LaunchProjection:
        """Return the exact one-shot launch rows while R06 installation is active."""

        view = self._action_epoch_active_r06_launch
        if (
            type(view) is not ActionEpochR06LaunchProjection
            or view.physical_owner is not self
            or view.epoch_owner is not self._action_epoch_owner
            or view._owner_identity is not self._owner_identity
            or view._token is not _ACTION_EPOCH_R06_LAUNCH_TOKEN
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical has no active ActionEpoch R06 launch"
            )
        return view

    def require_owned_action_epoch_r06_launch_projection(
        self, view: object
    ) -> ActionEpochR06LaunchProjection:
        """Validate R06's exact current Physical-owned launch view."""

        if view is not self._action_epoch_active_r06_launch:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch R06 launch projection is stale or foreign"
            )
        return self.action_epoch_r06_launch_projection()

    def require_owned_action_epoch_r06_postphysics_projection(
        self,
    ) -> ActionEpochR06PostPhysicsProjection:
        """Return the sole active direct-R06 packet without caller payloads."""

        view = self._action_epoch_active_r06_postphysics
        if (
            type(view) is not ActionEpochR06PostPhysicsProjection
            or view.physical_owner is not self
            or view.epoch_owner is not self._action_epoch_owner
            or view._owner_identity is not self._owner_identity
            or view._token is not _ACTION_EPOCH_R06_POSTPHYSICS_TOKEN
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical has no active ActionEpoch R06 postphysics projection"
            )
        return view

    def action_epoch_physics_fact_allocation(
        self,
    ) -> ActionEpochPhysicsFactAllocationProjection:
        """Return the exact active K-grid for Racket/scene face attribution."""

        view = self._action_epoch_active_physics_fact_allocation
        if (
            type(view) is not ActionEpochPhysicsFactAllocationProjection
            or view.physical_owner is not self
            or view.epoch_owner is not self._action_epoch_owner
            or view._owner_identity is not self._owner_identity
            or view._token is not _ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical has no active ActionEpoch physics-fact allocation"
            )
        return view

    def require_owned_action_epoch_physics_fact_allocation(
        self, view: object
    ) -> ActionEpochPhysicsFactAllocationProjection:
        """Validate the one current Physical-owned live K-grid."""

        if view is not self._action_epoch_active_physics_fact_allocation:
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch physics-fact allocation is stale or foreign"
            )
        return self.action_epoch_physics_fact_allocation()

    def _arm_current_action_epoch_physics_fact_source(
        self,
        *,
        launch_due_mask: torch.Tensor,
        selected_env_index: torch.Tensor | None = None,
        flight_slot: torch.Tensor | None = None,
        shot_key: _row_identity.ActionEpochShotKey | None = None,
        publication_ordinal: torch.Tensor | None = None,
    ) -> None:
        """Arm one substep from Physical's persistent K-grid.

        ``launch_due_mask`` is the owner-derived full ``[N,K]`` write mask.
        Optional selected rows are only the still-private D05 projection being
        installed by this same method caller.  Existing live rows always come
        from Physical's retained slot identity grids, so postphysics can
        consume a fresh scene authority on every decimation substep.
        """

        shape = (self.num_envs, self.flight_capacity)
        if (
            type(launch_due_mask) is not torch.Tensor
            or launch_due_mask.dtype != torch.bool
            or launch_due_mask.device != self.device
            or tuple(launch_due_mask.shape) != shape
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch launch-due K-grid ABI differs"
            )
        live_or_due = (self._published & ~self._parked) | launch_due_mask
        shot_key_grid = self._action_epoch_flight_shot_key.clone()
        ordinal_grid = self._action_epoch_flight_publication_ordinal.detach().clone()
        hash_grid = self._outcome_sha.detach().clone()
        if shot_key is not None:
            if (
                selected_env_index is None
                or flight_slot is None
                or publication_ordinal is None
            ):
                raise PhysicalEpochIntegrationHold(
                    "Physical ActionEpoch due rows lack a private K mapping"
                )
            row_count = selected_env_index.shape[0]
            _row_identity.require_action_epoch_shot_key(
                shot_key,
                shape=(row_count,),
                device=self.device,
                label="Physical ActionEpoch due shot_key",
            )
            if (
                type(publication_ordinal) is not torch.Tensor
                or publication_ordinal.dtype != torch.int64
                or publication_ordinal.device != self.device
                or tuple(publication_ordinal.shape) != (row_count,)
            ):
                raise PhysicalEpochIntegrationHold(
                    "Physical ActionEpoch due publication ordinal ABI differs"
                )
            for field in fields(_row_identity.ActionEpochShotKey):
                getattr(shot_key_grid, field.name)[selected_env_index, flight_slot] = (
                    getattr(shot_key, field.name)
                )
            ordinal_grid[selected_env_index, flight_slot] = publication_ordinal

        def masked_identity(value: torch.Tensor) -> torch.Tensor:
            return torch.where(live_or_due, value, torch.full_like(value, -1))

        masked_shot_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: masked_identity(getattr(shot_key_grid, field.name))
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )

        fixed_slot_grid = torch.arange(
            self.flight_capacity, dtype=torch.int64, device=self.device
        ).unsqueeze(0).expand(shape)
        self._action_epoch_active_physics_fact_allocation = (
            ActionEpochPhysicsFactAllocationProjection(
                active_mask=live_or_due.detach().clone(),
                launch_due_mask=launch_due_mask.detach().clone(),
                flight_slot=fixed_slot_grid.detach().clone(),
                shot_key=masked_shot_key,
                publication_ordinal=masked_identity(ordinal_grid),
                full_key_sha256=torch.where(
                    live_or_due.unsqueeze(-1), hash_grid, torch.zeros_like(hash_grid)
                ),
                physical_owner=self,
                epoch_owner=self._action_epoch_owner,
                owner_identity=self._owner_identity,
                _token=_ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN,
            )
        )
        try:
            if type(self.scene_port) is not TensorPhysicalFlightScenePort:
                self.scene_port.arm_action_epoch_physics_fact_source()
        finally:
            self._action_epoch_active_physics_fact_allocation = None

    def _prepare_action_epoch_scene_write(
        self,
        *,
        kind: str,
        state_env_f32: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> object:
        if kind not in ("launch", "retire"):
            raise PhysicalEpochIntegrationHold(
                "ActionEpoch scene write kind differs"
            )
        if self._action_epoch_active_scene_write is not None:
            raise PhysicalEpochIntegrationHold(
                "another ActionEpoch scene write is already active"
            )
        view = ActionEpochSceneWriteProjection(
            kind=kind,
            state_env_f32=state_env_f32.detach().clone(),
            selected_mask=selected_mask.detach().clone(),
            physical_owner=self,
            epoch_owner=self._action_epoch_owner,
            owner_identity=self._owner_identity,
            _token=_ACTION_EPOCH_SCENE_WRITE_TOKEN,
        )
        self._action_epoch_active_scene_write = view
        try:
            if type(self.scene_port) is TensorPhysicalFlightScenePort:
                return self.scene_port.preflight_write(
                    view.state_env_f32,
                    view.selected_mask,
                    device_faults_bound_in_reveal_row=True,
                )
            return self.scene_port.preflight_action_epoch_write()
        except BaseException:
            self._action_epoch_active_scene_write = None
            raise

    @staticmethod
    def _gather_action_epoch_shot_key(
        value: _row_identity.ActionEpochShotKey,
        row: torch.Tensor,
        slot: torch.Tensor,
    ) -> _row_identity.ActionEpochShotKey:
        return _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(value, field.name)[row, slot].detach().clone()
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )

    @staticmethod
    def _clone_action_epoch_pending(
        value: _ActionEpochPendingLaunchState | None,
    ) -> _ActionEpochPendingLaunchState | None:
        if value is None:
            return None
        return replace(
            value,
            shot_key=value.shot_key.clone(),
            **{
                field.name: getattr(value, field.name).detach().clone()
                for field in fields(_ActionEpochPendingLaunchState)
                if field.name != "shot_key"
            },
        )

    def _clone_action_epoch_direct_state(self) -> _ActionEpochDirectState:
        return _ActionEpochDirectState(
            pending_launch=self._clone_action_epoch_pending(
                self._action_epoch_pending_launch
            ),
            active_flight_slot=self._action_epoch_active_flight_slot.detach().clone(),
            flight_shot_key=self._action_epoch_flight_shot_key.clone(),
            flight_publication_ordinal=(
                self._action_epoch_flight_publication_ordinal.detach().clone()
            ),
        )

    def _selected_reset_action_epoch_direct_state(
        self, selected_env_mask: torch.Tensor
    ) -> _ActionEpochDirectState:
        before = self._clone_action_epoch_direct_state()

        def clear_rows(value: torch.Tensor) -> torch.Tensor:
            mask = selected_env_mask.reshape(
                self.num_envs, *((1,) * (value.ndim - 1))
            )
            fill = False if value.dtype == torch.bool else 0 if value.is_floating_point() else -1
            return torch.where(mask, torch.full_like(value, fill), value)

        pending = before.pending_launch
        if pending is not None:
            pending = replace(
                pending,
                shot_key=_row_identity.ActionEpochShotKey(
                    **{
                        field.name: clear_rows(getattr(pending.shot_key, field.name))
                        for field in fields(_row_identity.ActionEpochShotKey)
                    }
                ),
                **{
                    field.name: clear_rows(getattr(pending, field.name))
                    for field in fields(_ActionEpochPendingLaunchState)
                    if field.name != "shot_key"
                },
            )
        return _ActionEpochDirectState(
            pending_launch=pending,
            active_flight_slot=clear_rows(before.active_flight_slot),
            flight_shot_key=_row_identity.ActionEpochShotKey(
                **{
                    field.name: clear_rows(getattr(before.flight_shot_key, field.name))
                    for field in fields(_row_identity.ActionEpochShotKey)
                }
            ),
            flight_publication_ordinal=clear_rows(before.flight_publication_ordinal),
        )

    def _action_epoch_direct_state_mismatch(
        self, expected: _ActionEpochDirectState
    ) -> torch.Tensor:
        actual = self._clone_action_epoch_direct_state()
        if (actual.pending_launch is None) != (expected.pending_launch is None):
            return torch.ones((), dtype=torch.bool, device=self.device)
        pairs = [
            (actual.active_flight_slot, expected.active_flight_slot),
            (actual.flight_publication_ordinal, expected.flight_publication_ordinal),
            *((getattr(actual.flight_shot_key, f.name), getattr(expected.flight_shot_key, f.name))
              for f in fields(_row_identity.ActionEpochShotKey)),
        ]
        if actual.pending_launch is not None:
            pairs.extend(
                (getattr(actual.pending_launch, f.name), getattr(expected.pending_launch, f.name))
                for f in fields(_ActionEpochPendingLaunchState) if f.name != "shot_key"
            )
            pairs.extend(
                (getattr(actual.pending_launch.shot_key, f.name), getattr(expected.pending_launch.shot_key, f.name))
                for f in fields(_row_identity.ActionEpochShotKey)
            )
        mismatch = torch.zeros((), dtype=torch.bool, device=self.device)
        for left, right in pairs:
            mismatch |= ~_device_bitwise_equal(left, right)
        return mismatch

    def retain_action_epoch_launch(self, token: object) -> None:
        """Stage only D05's full-N ACCEPT view during its r05 writer."""

        d05_owner = self._action_epoch_device_r05_owner
        if d05_owner is None:
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch D05 owner is not construction-bound"
            )
        try:
            import action_ball_continuous_runtime_transaction_device as device_r05
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_continuous_runtime_transaction_device as device_r05
            import action_ball_full_mdp_epoch as epoch
        view = d05_owner.require_owned_action_epoch_accepted(
            token, owner_kind="physical_ball"
        )
        if (
            type(view) is not device_r05.DeviceR05AcceptedRowsView
            or view.transaction is not token
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch accepted-row authority differs"
            )
        shape = (self.num_envs, 1)

        def exact(value: object, *, label: str, dtype: torch.dtype,
                  width: int | None = None) -> torch.Tensor:
            expected = shape if width is None else shape + (width,)
            if (type(value) is not torch.Tensor or value.dtype != dtype
                    or value.device != self.device or tuple(value.shape) != expected
                    or not value.is_contiguous()):
                raise PhysicalEpochIntegrationHold(
                    "Physical accepted " + label + " ABI differs"
                )
            return value.detach().clone()

        key_grid = _row_identity.require_action_epoch_shot_key(
            view.identity.shot_key, shape=shape, device=self.device,
            label="Physical accepted shot_key",
        )
        key = _row_identity.ActionEpochShotKey(
            **{f.name: getattr(key_grid, f.name)[:, 0].clone()
               for f in fields(_row_identity.ActionEpochShotKey)}
        )
        task_valid = exact(
            view.task.task_valid, label="task_valid", dtype=torch.bool
        )[:, 0]
        task = exact(
            view.task.task_f32, label="task_f32", dtype=torch.float32,
            width=epoch.TASK_F32_WIDTH,
        )[:, 0]
        publication = exact(
            view.publication_ordinal, label="publication_ordinal",
            dtype=torch.int64,
        )[:, 0]
        target_xy = exact(
            view.target_xy_m, label="target_xy_m", dtype=torch.float32, width=2
        )[:, 0]
        launch_tick = exact(
            view.clocks.launch_tick, label="launch_tick", dtype=torch.int64
        )[:, 0]
        deadline_tick = exact(
            view.clocks.deadline_tick, label="deadline_tick", dtype=torch.int64
        )[:, 0]
        horizon_tick = exact(
            view.clocks.next_reveal_tick, label="next_reveal_tick", dtype=torch.int64
        )[:, 0]
        physical_start = epoch.MOTION_TASK_F32_WIDTH + epoch.RACKET_TASK_F32_WIDTH
        physical_state = task[:, physical_start:].contiguous()
        available = (
            self._parked & ~self._published
            & self._lifecycle.eq(R06_FLIGHT_EMPTY) & ~self._device_fault
        )
        has_available = available.any(dim=1)
        flight_slot = available.to(torch.int64).argmax(dim=1)
        before = self._action_epoch_pending_launch
        if before is None:
            before = _ActionEpochPendingLaunchState(
                pending=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                flight_slot=torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device),
                shot_key=_row_identity.empty_action_epoch_shot_key((self.num_envs,), device=self.device),
                publication_ordinal=torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device),
                physical_state_f32=torch.zeros((self.num_envs, STATE_WIDTH), dtype=torch.float32, device=self.device),
                target_xy_m=torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device),
                launch_control_step=torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device),
                contact_deadline_control_step=torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device),
                crossing_horizon_control_step=torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device),
            )
        valid = ~task_valid | (
            _row_identity.action_epoch_shot_key_valid(key)
            & publication.ge(0) & key.action_slot.eq(0)
            & has_available & ~before.pending
            & self._action_epoch_active_flight_slot.lt(0)
            & torch.isfinite(physical_state).all(dim=1)
            & torch.isfinite(target_xy).all(dim=1)
            & deadline_tick.ge(launch_tick) & horizon_tick.gt(deadline_tick)
        )
        torch._assert_async(
            torch.all(valid), "Physical D05 ACCEPT rows are not launchable"
        )

        def merge(value: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
            mask = task_valid.reshape(self.num_envs, *((1,) * (prior.ndim - 1)))
            return torch.where(mask, value, prior)

        self._action_epoch_pending_launch = _ActionEpochPendingLaunchState(
            pending=before.pending | task_valid,
            flight_slot=merge(flight_slot, before.flight_slot),
            shot_key=_row_identity.ActionEpochShotKey(
                **{f.name: merge(getattr(key, f.name), getattr(before.shot_key, f.name))
                   for f in fields(_row_identity.ActionEpochShotKey)}
            ),
            publication_ordinal=merge(publication, before.publication_ordinal),
            physical_state_f32=merge(physical_state, before.physical_state_f32),
            target_xy_m=merge(target_xy, before.target_xy_m),
            launch_control_step=merge(launch_tick, before.launch_control_step),
            contact_deadline_control_step=merge(
                deadline_tick, before.contact_deadline_control_step
            ),
            crossing_horizon_control_step=merge(
                horizon_tick, before.crossing_horizon_control_step
            ),
        )

    def refresh_action_epoch_host_activity(
        self, *, next_control_step: int
    ) -> None:
        """Cache one multi-writer verdict after D05; synchronize exactly once."""

        self._require_operable(allow_diagnostic_scene_observation=True)
        if type(next_control_step) is not int or next_control_step < 1:
            raise PhysicalEpochIntegrationHold(
                "Physical activity refresh control step differs"
            )
        active_transactions = (
            self._action_epoch_substep_pair,
            self._action_epoch_active_scene_write,
            self._action_epoch_active_r06_launch,
            self._action_epoch_active_r06_postphysics,
            self._action_epoch_active_physics_fact_allocation,
            self._active_postphysics_capture,
            self._active_postphysics_capture_image,
        )
        if any(value is not None for value in active_transactions):
            raise PhysicalEpochIntegrationHold(
                "Physical activity refresh crossed an active substep pair"
            )
        if next_control_step != self._action_epoch_expected_control(
            require_control_boundary=True
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical activity refresh is stale, skipped, or replayed"
            )
        if self._action_epoch_host_activity_control_step == next_control_step:
            raise PhysicalEpochIntegrationHold(
                "Physical activity refresh is stale, skipped, or replayed"
            )

        # Fail dense if any part of the reduction raises.  A prior cached idle
        # verdict can therefore never leak into a later control step.
        self._action_epoch_host_activity_control_step = None
        self._action_epoch_host_activity_has_work = True
        pending = self._action_epoch_pending_launch
        pending_rows = (
            self._action_epoch_empty_env_mask
            if pending is None
            else pending.pending
        )
        physical_work = (
            pending_rows
            | (
                (self._published & ~self._parked)
                | self._lifecycle.ne(R06_FLIGHT_EMPTY)
                | self._action_epoch_flight_publication_ordinal.ge(0)
            ).any(dim=1)
            | self._action_epoch_active_flight_slot.ge(0)
            | self._selected_contact_pending
        )
        shape = (self.num_envs, self.flight_capacity)
        r06_state = _tensor(
            getattr(self._r06_owner, "flight_state", None),
            label="R06 activity census",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        r06_work = r06_state.ne(R06_FLIGHT_EMPTY).any(dim=1)
        if type(self.scene_port) is TensorPhysicalFlightScenePort:
            scene_work = self._action_epoch_empty_env_mask
        else:
            scene_activity = _tensor(
                self.scene_port.action_epoch_physics_fact_activity_mask(),
                label="scene activity census",
                shape=shape,
                dtype=torch.bool,
                device=self.device,
            )
            scene_work = scene_activity.any(dim=1)
        self._action_epoch_host_activity_has_work = bool(
            torch.any(physical_work | r06_work | scene_work).item()
        )
        self._action_epoch_host_activity_control_step = next_control_step

    def _action_epoch_expected_control(
        self, *, require_control_boundary: bool
    ) -> int:
        previous = self._last_postphysics_exact_stamp
        if previous is None:
            return 1
        control, substep, decimation, _sim_step, _phase = previous
        at_boundary = substep == decimation - 1
        if require_control_boundary and not at_boundary:
            raise PhysicalEpochIntegrationHold(
                "Physical activity refresh preceded the final physics substep"
            )
        return control + int(at_boundary)

    def _action_epoch_cached_idle_for_next_substep(self) -> bool:
        if self._action_epoch_substep_pair is not None:
            raise PhysicalEpochIntegrationHold(
                "Physical pre-physics call crossed an unconsumed post pair"
            )
        cached_control = self._action_epoch_host_activity_control_step
        if cached_control is None:
            return False
        if cached_control != self._action_epoch_expected_control(
            require_control_boundary=False
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical cached activity control step is stale"
            )
        return not self._action_epoch_host_activity_has_work

    def launch_action_epoch(self) -> None:
        """Launch full-N pending rows whose exact Motion-owned tick is due."""

        if self._action_epoch_cached_idle_for_next_substep():
            if type(self.scene_port) is not TensorPhysicalFlightScenePort:
                self.scene_port.begin_action_epoch_idle_physics_fact_source()
            self._action_epoch_substep_pair = _ACTION_EPOCH_SUBSTEP_IDLE
            return

        epoch_owner = self._action_epoch_owner
        motion_owner = self._action_epoch_motion_owner
        pending = self._action_epoch_pending_launch
        if (
            epoch_owner is None
            or motion_owner is None
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch launch sources are not construction-bound"
            )
        empty_due = torch.zeros(
            (self.num_envs, self.flight_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        # Before the first reveal, or after reset has removed the retained
        # launch, still arm an exact empty/current K-grid.  This is the
        # branchless per-substep scene authority transaction; an empty arm is
        # not evidence that physics ran and creates no fact by itself.
        if (
            pending is None
        ):
            self._arm_current_action_epoch_physics_fact_source(
                launch_due_mask=empty_due
            )
            self._action_epoch_substep_pair = _ACTION_EPOCH_SUBSTEP_DENSE
            return
        try:
            token = motion_owner.action_ball_continuous_motion_observation_projection()
            motion = motion_owner.require_owned_action_ball_continuous_motion_observation(
                token
            )
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_full_mdp_epoch as epoch
        except BaseException as exc:
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch current Motion producer is absent or stale"
            ) from exc
        ids = torch.arange(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        current_tick = getattr(motion, "control_tick", None)
        if (
            getattr(motion, "motion_owner", None) is not motion_owner
            or type(current_tick) is not torch.Tensor
            or current_tick.dtype != torch.int64
            or current_tick.device != self.device
            or tuple(current_tick.shape) != (self.num_envs,)
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical ActionEpoch Motion clock ABI differs"
            )
        record = epoch_owner.current()
        shot_slots = record.current_task_slot
        safe_shot_slots = shot_slots.clamp(0, epoch_owner.shot_slot_capacity - 1)
        record_key_grid = _row_identity.require_action_epoch_shot_key(
            record.identity.shot_key,
            shape=(self.num_envs, epoch_owner.shot_slot_capacity),
            device=self.device,
            label="Physical launch current shot_key",
        )
        current_key = self._gather_action_epoch_shot_key(
            record_key_grid, ids, safe_shot_slots
        )
        key_current = _row_identity.action_epoch_shot_key_equal(
            current_key, pending.shot_key
        )
        identity_current = (
            getattr(motion, "action_uid").eq(pending.shot_key.action_uid)
            & getattr(motion, "task_identity").eq(pending.shot_key.task_identity)
            & getattr(motion, "reset_generation").eq(
                pending.shot_key.reset_generation
            )
            & getattr(motion, "swing_generation").eq(
                pending.shot_key.ball_generation
            )
        )
        phase_current = record.phase[ids, safe_shot_slots].eq(
            epoch.PHASE_REVEAL_COMMITTED
        )
        reached = current_tick.ge(pending.launch_control_step)
        launch_due = (
            pending.pending
            & phase_current
            & identity_current
            & key_current
            & reached
        )
        invalid_due = pending.pending & reached & (
            ~phase_current | ~identity_current | ~key_current
        )
        torch._assert_async(
            torch.all(~invalid_due),
            "Physical ActionEpoch due row lost its exact Motion/epoch identity",
        )

        state_before = self.scene_port.read_state_env()
        state_after = state_before.clone()
        safe_flight_slots = pending.flight_slot.clamp(0, self.flight_capacity - 1)
        selected_before = state_before[ids, safe_flight_slots]
        state_after[ids, safe_flight_slots] = torch.where(
            launch_due[:, None], pending.physical_state_f32, selected_before
        )
        slot_mask = torch.zeros(
            (self.num_envs, self.flight_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        slot_mask[ids, safe_flight_slots] = launch_due
        writes_started = False
        try:
            self._arm_current_action_epoch_physics_fact_source(
                launch_due_mask=slot_mask,
                selected_env_index=ids,
                flight_slot=safe_flight_slots,
                shot_key=pending.shot_key,
                publication_ordinal=pending.publication_ordinal,
            )
            handle = self._prepare_action_epoch_scene_write(
                kind="launch",
                state_env_f32=state_after,
                selected_mask=slot_mask,
            )
            writes_started = True
            receipt = self.scene_port.apply_prevalidated_write(handle)
            self.scene_port.require_owned_apply_receipt(handle, receipt)
            self._action_epoch_active_scene_write = None
            selected_lifecycle = self._lifecycle[ids, safe_flight_slots]
            self._lifecycle[ids, safe_flight_slots] = torch.where(
                launch_due,
                torch.full_like(selected_lifecycle, R06_FLIGHT_INBOUND),
                selected_lifecycle,
            )
            selected_reveal = self._reveal_step[ids, safe_flight_slots]
            self._reveal_step[ids, safe_flight_slots] = torch.where(
                launch_due, pending.launch_control_step, selected_reveal
            )
            previous_center = self._previous_ball_center[ids, safe_flight_slots]
            self._previous_ball_center[ids, safe_flight_slots] = torch.where(
                launch_due[:, None],
                pending.physical_state_f32[:, :3],
                previous_center,
            )
            self._parked[ids, safe_flight_slots] &= ~launch_due
            self._published[ids, safe_flight_slots] |= launch_due
            self._slot_version[ids, safe_flight_slots] += launch_due.to(torch.int64)
            self._action_epoch_active_flight_slot[ids] = torch.where(
                launch_due,
                safe_flight_slots,
                self._action_epoch_active_flight_slot[ids],
            )
            for field in fields(_row_identity.ActionEpochShotKey):
                destination = getattr(self._action_epoch_flight_shot_key, field.name)
                selected = getattr(pending.shot_key, field.name)
                current_identity = destination[ids, safe_flight_slots]
                destination[ids, safe_flight_slots] = torch.where(
                    launch_due, selected, current_identity
                )
            selected_ordinal = self._action_epoch_flight_publication_ordinal[
                ids, safe_flight_slots
            ]
            self._action_epoch_flight_publication_ordinal[
                ids, safe_flight_slots
            ] = torch.where(
                launch_due, pending.publication_ordinal, selected_ordinal
            )
            # A planned launch after reveal is not late.  A due row observed
            # after its exact Motion-owned launch tick is late.
            late_launch = launch_due & current_tick.gt(
                pending.launch_control_step
            )
            if self._action_epoch_active_r06_launch is not None:
                raise PhysicalEpochIntegrationHold(
                    "another Physical ActionEpoch R06 launch is active"
                )
            selected = pending.pending

            def selected_or(value: torch.Tensor, fill: int | float) -> torch.Tensor:
                mask = selected.reshape(
                    self.num_envs, *((1,) * (value.ndim - 1))
                )
                return torch.where(mask, value, torch.full_like(value, fill))

            self._action_epoch_active_r06_launch = ActionEpochR06LaunchProjection(
                selected_mask=selected.detach().clone(),
                due=launch_due.detach().clone(),
                late_launch=selected_or(late_launch, False),
                flight_slot=selected_or(safe_flight_slots, -1),
                shot_key=_row_identity.ActionEpochShotKey(
                    **{
                        field.name: selected_or(
                            getattr(pending.shot_key, field.name), -1
                        )
                        for field in fields(_row_identity.ActionEpochShotKey)
                    }
                ),
                publication_ordinal=selected_or(
                    pending.publication_ordinal, -1
                ),
                target_xy_m=selected_or(pending.target_xy_m, 0.0),
                launch_control_step=selected_or(
                    pending.launch_control_step, -1
                ),
                contact_deadline_control_step=(
                    selected_or(pending.contact_deadline_control_step, -1)
                ),
                crossing_horizon_control_step=(
                    selected_or(pending.crossing_horizon_control_step, -1)
                ),
                physical_owner=self,
                epoch_owner=epoch_owner,
                owner_identity=self._owner_identity,
                _token=_ACTION_EPOCH_R06_LAUNCH_TOKEN,
            )
            r06_owner = self._r06_owner
            r06_install = getattr(
                r06_owner,
                "install_action_ball_full_mdp_epoch_launch_from_physical",
                None,
            )
            direct_r06_install = getattr(
                type(r06_owner),
                "install_action_ball_full_mdp_epoch_launch_from_physical",
                None,
            )
            if (
                r06_owner is None
                or not callable(r06_install)
                or not callable(direct_r06_install)
                or getattr(r06_install, "__self__", None) is not r06_owner
                or getattr(r06_install, "__func__", None) is not direct_r06_install
            ):
                raise PhysicalEpochIntegrationHold(
                    "R06 exact ActionEpoch launch installer is absent or patched"
                )
            r06_install()
            epoch_owner.refresh_physical_launch_rows()
            self._action_epoch_active_r06_launch = None
            self._action_epoch_pending_launch = replace(
                pending, pending=pending.pending & ~launch_due
            )
            self._action_epoch_substep_pair = _ACTION_EPOCH_SUBSTEP_DENSE
        except BaseException:
            self._action_epoch_active_scene_write = None
            self._action_epoch_active_r06_launch = None
            self._action_epoch_active_physics_fact_allocation = None
            if writes_started:
                self._poisoned = True
                self._poison_reason = "Physical ActionEpoch scene launch failed"
                epoch_owner.poison_owner_write(
                    "physical_ball",
                    PHYSICAL_EPOCH_FAULT_LAUNCH_SOURCE,
                    owner=self,
                )
            raise

    def _prepare_physical_hot_late_launch_for_test(
        self,
        *,
        device_r05_projection: object,
        physical_question_final: object,
        contact_tick: torch.Tensor,
        launch_tick: torch.Tensor,
        motion_tick_s: float,
        flight_slot: torch.Tensor,
        full_key_sha256: torch.Tensor,
        task_sha256: torch.Tensor,
        action_uid: torch.Tensor,
    ) -> PhysicalHotPreparedInstall:
        """Focused numerical seam; it never authorizes production binding.

        All validation and tensor copies happen before an irreversible child
        boundary.  This seam deliberately requires the owner-issued D05 child
        projection object and a Physical-question final batch.  It is private
        because those upstream modules have not yet frozen final authority
        validators; callers cannot turn this test seam into a runtime GO.
        """

        self._require_operable()
        if (
            self._active_physical_hot_prepare is not None
            or self._active_physical_hot_commit is not None
            or self._active_physical_late_launch is not None
        ):
            raise PhysicalFlightDeviceError(
                "one Physical hot/late-launch transaction is already active"
            )
        if type(device_r05_projection) is not _r05_device.DeviceR05PreparedRevealProjection:
            raise PhysicalFlightDeviceError(
                "Physical hot prepare requires the exact Device-R05 projection type"
            )
        if getattr(device_r05_projection, "owner_kind", None) != "physical_ball":
            raise PhysicalFlightDeviceError(
                "Physical hot Device-R05 owner kind must be physical_ball"
            )
        selected_env_index = device_r05_projection.selected_env_index
        if (
            not isinstance(selected_env_index, torch.Tensor)
            or selected_env_index.dtype != torch.int64
            or selected_env_index.device != self.device
            or selected_env_index.ndim != 1
            or selected_env_index.numel() < 1
            or not selected_env_index.is_contiguous()
        ):
            raise PhysicalFlightDeviceError(
                "Physical hot selected_env_index ABI differs"
            )
        k = selected_env_index.shape[0]
        selected_env_index_in_range = (
            (selected_env_index >= 0) & (selected_env_index < self.num_envs)
        )
        safe_selected_env_index = selected_env_index.clamp(
            min=0, max=self.num_envs - 1
        )
        sorted_env_index, _ = torch.sort(selected_env_index)
        duplicate_env_index = torch.zeros_like(
            selected_env_index, dtype=torch.bool
        )
        if k > 1:
            duplicate_sorted = sorted_env_index[1:].eq(
                sorted_env_index[:-1]
            )
            duplicate_env_index[1:] |= duplicate_sorted
            duplicate_env_index[:-1] |= duplicate_sorted
        selected_mask = _tensor(
            device_r05_projection.selected_mask,
            label="Physical hot selected_mask",
            shape=(self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )
        expected_mask = torch.zeros_like(selected_mask)
        expected_mask[safe_selected_env_index] = selected_env_index_in_range
        mask_index_valid = torch.eq(selected_mask, expected_mask).all()

        try:
            import action_ball_physical_question_device as question
        except ImportError as exc:
            raise PhysicalLateLaunchProductionHold(
                "Physical question module is unavailable"
            ) from exc
        if type(physical_question_final) is not question.PhysicalQuestionFinalBatch:
            raise PhysicalFlightDeviceError(
                "Physical hot question final batch type differs"
            )

        candidate_identity = _tensor(
            device_r05_projection.selected_candidate_identity,
            label="Physical hot selected candidate identity",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        question_candidate = _tensor(
            physical_question_final.candidate_identity,
            label="Physical question candidate identity",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        contact = _tensor(
            contact_tick,
            label="Physical exact contact tick",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        launch = _tensor(
            launch_tick,
            label="Physical exact launch tick",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        question_contact = _tensor(
            physical_question_final.contact_tick,
            label="Physical question contact tick",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        question_launch = _tensor(
            physical_question_final.launch_tick,
            label="Physical question launch tick",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        physical_state = _tensor(
            physical_question_final.physical_state_f32,
            label="Physical exact launch state",
            shape=(k, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        d05_state = _tensor(
            device_r05_projection.numeric_f32,
            label="Physical Device-R05 selected state",
            shape=(k, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        horizon_s = _tensor(
            physical_question_final.effective_contact_horizon_s,
            label="Physical effective contact horizon",
            shape=(k,),
            dtype=torch.float32,
            device=self.device,
        )
        producer_fault = _tensor(
            physical_question_final.producer_fault,
            label="Physical question producer fault",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        construction_reason = _tensor(
            physical_question_final.construction_reason,
            label="Physical question construction reason",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        selected_reason = _tensor(
            device_r05_projection.selected_construction_reason,
            label="Physical Device-R05 selected reason",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        slot = _tensor(
            flight_slot,
            label="Physical flight slot",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        full_key = _tensor(
            full_key_sha256,
            label="Physical full key SHA-256",
            shape=(k, TOKEN_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        task = _tensor(
            task_sha256,
            label="Physical task SHA-256",
            shape=(k, TOKEN_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        uid = _tensor(
            action_uid,
            label="Physical action UID",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        reset_generation = _tensor(
            device_r05_projection.reset_generation,
            label="Physical reset generation",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        swing_generation = _tensor(
            device_r05_projection.swing_generation,
            label="Physical swing generation",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        action_slot = _tensor(
            device_r05_projection.action_slot,
            label="Physical action slot",
            shape=(k,),
            dtype=torch.int64,
            device=self.device,
        )
        chosen_horizon = contact - launch
        if (
            type(motion_tick_s) is not float
            or not math.isfinite(motion_tick_s)
            or motion_tick_s <= 0.0
        ):
            raise PhysicalFlightDeviceError(
                "Physical Motion tick duration must be a positive finite float"
            )
        expected_horizon_s = chosen_horizon.to(torch.float32) * motion_tick_s
        safe_slot = slot.clamp(min=0, max=self.flight_capacity - 1)
        selected_scene_state = self.scene_port.read_state_env()[
            safe_selected_env_index, safe_slot
        ]
        d05_binding_valid = (
            selected_env_index_in_range
            & ~duplicate_env_index
            & mask_index_valid
            & selected_reason.eq(_r05_device.QUESTION_CONSTRUCTION_REASON_ADMITTED)
            & (slot >= 0)
            & (slot < self.flight_capacity)
            & (reset_generation >= 1)
            & (swing_generation >= 0)
            & (action_slot >= 0)
        )
        question_binding_valid = (
            torch.eq(candidate_identity, question_candidate)
            & torch.eq(contact, question_contact)
            & torch.eq(launch, question_launch)
            & (contact > launch)
            & (launch >= 0)
            & torch.isfinite(horizon_s)
            & (horizon_s > 0.0)
            & torch.eq(horizon_s, expected_horizon_s)
            & torch.all(torch.isfinite(physical_state), dim=1)
            & torch.all(torch.eq(physical_state, d05_state), dim=1)
            & construction_reason.eq(question.CONSTRUCTION_REASON_ADMITTED)
            & producer_fault.eq(0)
        )
        identity_valid = (
            (uid >= 0)
            & torch.any(full_key.ne(0), dim=1)
            & torch.any(task.ne(0), dim=1)
        )
        scene_valid = (
            self._device_reset_generation[safe_selected_env_index].eq(
                reset_generation
            )
            & self._lifecycle[
                safe_selected_env_index, safe_slot
            ].eq(R06_FLIGHT_EMPTY)
            & self._parked[safe_selected_env_index, safe_slot]
            & ~self._published[safe_selected_env_index, safe_slot]
            & ~self._device_fault[safe_selected_env_index, safe_slot]
        )
        prearm_fault = torch.zeros(
            (k,), dtype=torch.int64, device=self.device
        )
        prearm_fault = torch.where(
            ~d05_binding_valid,
            torch.bitwise_or(
                prearm_fault,
                torch.full_like(
                    prearm_fault, PHYSICAL_HOT_FAULT_D05_PROJECTION
                ),
            ),
            prearm_fault,
        )
        prearm_fault = torch.where(
            ~question_binding_valid,
            torch.bitwise_or(
                prearm_fault,
                torch.full_like(
                    prearm_fault, PHYSICAL_HOT_FAULT_QUESTION_BINDING
                ),
            ),
            prearm_fault,
        )
        prearm_fault = torch.where(
            ~identity_valid,
            torch.bitwise_or(
                prearm_fault,
                torch.full_like(prearm_fault, PHYSICAL_HOT_FAULT_IDENTITY),
            ),
            prearm_fault,
        )
        prearm_fault = torch.where(
            ~scene_valid,
            torch.bitwise_or(
                prearm_fault,
                torch.full_like(
                    prearm_fault, PHYSICAL_HOT_FAULT_SCENE_PRECONDITION
                ),
            ),
            prearm_fault,
        ).contiguous()

        capability = object.__new__(PhysicalHotPreparedInstall)
        record = _PhysicalHotPreparedRecord(
            capability=capability,
            device_r05_projection=device_r05_projection,
            device_r05_projection_identity=device_r05_projection.preview_identity,
            selected_env_index=safe_selected_env_index.clone(),
            selected_mask=selected_mask.clone(),
            flight_slot=safe_slot.clone(),
            full_key_sha256=full_key.clone(),
            task_sha256=task.clone(),
            action_uid=uid.clone(),
            reset_generation=reset_generation.clone(),
            swing_generation=swing_generation.clone(),
            action_slot=action_slot.clone(),
            candidate_identity=candidate_identity.clone(),
            contact_tick=contact.clone(),
            launch_tick=launch.clone(),
            chosen_horizon_ticks=chosen_horizon.clone(),
            effective_contact_horizon_s=horizon_s.clone(),
            physical_state_f32=physical_state.clone(),
            producer_fault=producer_fault.clone(),
            prearm_fault=prearm_fault,
            scene_state_before=selected_scene_state.clone(),
            owner_mutation_version=self._mutation_version,
            stage="prepared",
        )
        self._physical_hot_prepared_records[capability] = record
        self._active_physical_hot_prepare = capability
        return capability

    def _physical_hot_prearm_fault_for_test(
        self, prepared: PhysicalHotPreparedInstall
    ) -> torch.Tensor:
        """Return the device row that the future global boundary must consume."""

        record = self._physical_hot_prepared_records.get(prepared)
        if (
            type(prepared) is not PhysicalHotPreparedInstall
            or prepared is not self._active_physical_hot_prepare
            or record is None
            or record.capability is not prepared
            or record.stage != "prepared"
        ):
            raise PhysicalFlightDeviceError(
                "Physical hot prepared capability is stale or foreign"
            )
        return record.prearm_fault.clone()

    def _commit_physical_hot_reveal_for_test(
        self, prepared: PhysicalHotPreparedInstall
    ) -> PhysicalHotChildCommitToken:
        """Cross the reveal boundary by retaining bytes only; never write scene."""

        record = self._physical_hot_prepared_records.get(prepared)
        if (
            type(prepared) is not PhysicalHotPreparedInstall
            or prepared is not self._active_physical_hot_prepare
            or record is None
            or record.capability is not prepared
            or record.stage != "prepared"
        ):
            raise PhysicalFlightDeviceError(
                "Physical hot prepared capability is stale or foreign"
            )
        if self.device.type == "cpu" and bool(torch.any(record.prearm_fault)):
            raise PhysicalFlightDeviceError(
                "Physical hot boundary rejected the prearm fault row"
            )
        token = object.__new__(PhysicalHotChildCommitToken)
        self._physical_hot_prepared_records[prepared] = replace(
            record, stage="retained"
        )
        self._physical_hot_commit_records[token] = _PhysicalHotCommitRecord(
            capability=token,
            prepared=prepared,
            device_r05_projection=record.device_r05_projection,
            stage="retained",
        )
        self._active_physical_hot_prepare = None
        self._active_physical_hot_commit = token
        return token

    def _publish_physical_late_launch_for_test(
        self,
        child_commit: PhysicalHotChildCommitToken,
        *,
        current_motion_tick: torch.Tensor,
    ) -> PhysicalLateLaunchPublication:
        """Write the retained exact 13D state only at the exact launch tick."""

        commit = self._physical_hot_commit_records.get(child_commit)
        record = None if commit is None else self._physical_hot_prepared_records.get(
            commit.prepared
        )
        if (
            type(child_commit) is not PhysicalHotChildCommitToken
            or child_commit is not self._active_physical_hot_commit
            or commit is None
            or commit.capability is not child_commit
            or commit.stage != "retained"
            or record is None
            or record.stage != "retained"
        ):
            raise PhysicalFlightDeviceError(
                "Physical hot child commit is stale or foreign"
            )
        tick = _tensor(
            current_motion_tick,
            label="Physical current Motion tick",
            shape=tuple(record.launch_tick.shape),
            dtype=torch.int64,
            device=self.device,
        )
        if not torch.equal(tick, record.launch_tick):
            raise PhysicalFlightDeviceError(
                "Physical late launch is not the exact Motion launch tick"
            )
        scene_before = self.scene_port.read_state_env()
        selected_before = scene_before[
            record.selected_env_index, record.flight_slot
        ]
        if (
            self._mutation_version != record.owner_mutation_version
            or not torch.equal(selected_before, record.scene_state_before)
        ):
            raise PhysicalFlightDeviceError(
                "Physical scene changed between reveal and exact launch"
            )
        scene_after = scene_before.clone()
        scene_after[
            record.selected_env_index, record.flight_slot
        ] = record.physical_state_f32
        slot_mask = torch.zeros(
            (self.num_envs, self.flight_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        slot_mask[record.selected_env_index, record.flight_slot] = True
        handle = self.scene_port.preflight_write(
            scene_after,
            slot_mask,
            device_faults_bound_in_reveal_row=True,
        )
        try:
            receipt = self.scene_port.apply_prevalidated_write(handle)
            self.scene_port.require_owned_apply_receipt(handle, receipt)
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "Physical exact late-launch scene write failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc

        self._lifecycle[
            record.selected_env_index, record.flight_slot
        ] = R06_FLIGHT_INBOUND
        self._generation[
            record.selected_env_index, record.flight_slot
        ] = record.swing_generation
        self._outcome_sha[
            record.selected_env_index, record.flight_slot
        ].copy_(record.full_key_sha256)
        self._reveal_step[
            record.selected_env_index, record.flight_slot
        ] = record.launch_tick
        self._previous_ball_center[
            record.selected_env_index, record.flight_slot
        ].copy_(record.physical_state_f32[:, :3])
        self._parked[record.selected_env_index, record.flight_slot] = False
        self._published[record.selected_env_index, record.flight_slot] = True
        self._slot_version[record.selected_env_index, record.flight_slot] += 1
        self._set_owner_mutation_version(self._mutation_version + 1)

        publication = object.__new__(PhysicalLateLaunchPublication)
        publication_identity = object()
        self._physical_late_launch_records[publication] = _PhysicalLateLaunchRecord(
            publication=publication,
            prepared=commit.prepared,
            device_r05_projection=record.device_r05_projection,
            physical_publication_identity=publication_identity,
            selected_env_index=record.selected_env_index.clone(),
            selected_mask=record.selected_mask.clone(),
            flight_slot=record.flight_slot.clone(),
            full_key_sha256=record.full_key_sha256.clone(),
            task_sha256=record.task_sha256.clone(),
            action_uid=record.action_uid.clone(),
            reset_generation=record.reset_generation.clone(),
            swing_generation=record.swing_generation.clone(),
            action_slot=record.action_slot.clone(),
            candidate_identity=record.candidate_identity.clone(),
            contact_tick=record.contact_tick.clone(),
            launch_tick=record.launch_tick.clone(),
            chosen_horizon_ticks=record.chosen_horizon_ticks.clone(),
            effective_contact_horizon_s=record.effective_contact_horizon_s.clone(),
            physical_state_f32=record.physical_state_f32.clone(),
            producer_fault=record.producer_fault.clone(),
            stage="published",
        )
        self._physical_hot_commit_records[child_commit] = replace(
            commit, stage="published"
        )
        self._active_physical_hot_commit = None
        self._active_physical_late_launch = publication
        return publication

    def require_owned_late_launch_publication(
        self,
        publication: object,
        *,
        expected_device_r05_projection: object,
    ) -> PhysicalLateLaunchPublicationView:
        """Validate exact publication identity and return clone-only R06 facts."""

        record = self._physical_late_launch_records.get(publication)
        if (
            type(publication) is not PhysicalLateLaunchPublication
            or publication is not self._active_physical_late_launch
            or record is None
            or record.publication is not publication
            or record.stage != "published"
            or expected_device_r05_projection is not record.device_r05_projection
        ):
            raise PhysicalFlightDeviceError(
                "Physical late-launch publication is stale or foreign"
            )
        return PhysicalLateLaunchPublicationView(
            device_r05_projection_identity=(
                record.device_r05_projection.preview_identity
            ),
            physical_publication_identity=record.physical_publication_identity,
            selected_env_index=record.selected_env_index.clone(),
            selected_mask=record.selected_mask.clone(),
            flight_slot=record.flight_slot.clone(),
            full_key_sha256=record.full_key_sha256.clone(),
            task_sha256=record.task_sha256.clone(),
            action_uid=record.action_uid.clone(),
            reset_generation=record.reset_generation.clone(),
            swing_generation=record.swing_generation.clone(),
            action_slot=record.action_slot.clone(),
            candidate_identity=record.candidate_identity.clone(),
            contact_tick=record.contact_tick.clone(),
            launch_tick=record.launch_tick.clone(),
            chosen_horizon_ticks=record.chosen_horizon_ticks.clone(),
            effective_contact_horizon_s=(
                record.effective_contact_horizon_s.clone()
            ),
            physical_state_f32=record.physical_state_f32.clone(),
            producer_fault=record.producer_fault.clone(),
        )

    def bind_r06_owner(self, r06_owner: object) -> None:
        """Permanently pair this physical owner with one exact R06 owner."""

        self._require_operable(allow_action_epoch_construction=True)
        if self._diagnostic_n2_no_save and self._action_epoch_owner is None:
            raise PhysicalEpochIntegrationHold(
                "diagnostic Physical/R06 binding requires the exact ActionEpoch first"
            )
        if self._r06_owner is r06_owner:
            return
        if self._r06_owner is not None:
            self._poisoned = True
            self._poison_reason = "physical R06 owner cannot be rebound"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        owner = _require_final_r06_type(
            r06_owner,
            expected_name="ActionBallLandingOutcomeDeviceCoordinator",
        )
        if (
            getattr(owner, "num_envs", None) != self.num_envs
            or getattr(owner, "flight_slot_capacity", None)
            != self.flight_capacity
            or getattr(owner, "device", None) != self.device
            or getattr(owner, "dtype", None) != torch.float32
        ):
            raise PhysicalFlightDeviceError(
                "R06 coordinator physical grid/device ABI differs"
            )
        contact_consumer = getattr(
            owner,
            "consume_owned_post_physics_contact_authority",
            None,
        )
        direct_contact_consumer = getattr(
            type(owner),
            "consume_owned_post_physics_contact_authority",
            None,
        )
        if (
            not callable(contact_consumer)
            or not callable(direct_contact_consumer)
            or getattr(contact_consumer, "__self__", None) is not owner
            or getattr(contact_consumer, "__func__", None)
            is not direct_contact_consumer
        ):
            raise PhysicalFlightDeviceError(
                "R06 contact-authority consumer must be one direct owner method"
            )
        module = sys.modules.get(type(owner).__module__)
        factory = None if module is None else getattr(
            module,
            "mint_landing_outcome_physical_park_token_authority",
            None,
        )
        if not callable(factory):
            raise PhysicalFlightDeviceError(
                "R06 physical park authority factory is unavailable"
            )
        self._r06_owner = owner
        self._r06_contact_authority_consumer = contact_consumer
        try:
            authority = factory(self)
            owner.bind_physical_park_token_authority(self, authority)
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "physical/R06 pair binding failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        self._r06_park_token_authority = authority

        selected_factory = None if module is None else getattr(
            module,
            "mint_landing_outcome_selected_reset_physical_park_token_authority",
            None,
        )
        selected_bind = getattr(
            owner,
            "bind_selected_reset_physical_park_token_authority",
            None,
        )
        if not callable(selected_factory) or not callable(selected_bind):
            self._poisoned = True
            self._poison_reason = (
                "R06 selected-reset physical park authority is unavailable"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        try:
            selected_authority = selected_factory(self)
            selected_bind(self, selected_authority)
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "physical/R06 selected-reset pair binding failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        self._r06_selected_reset_park_token_authority = selected_authority

    @staticmethod
    def _r05_terminal_claim_projection(
        claim: _r05.PreparedRevealTerminalClaim,
    ) -> _flight.R05TerminalClaimProjection:
        if type(claim) is not _r05.PreparedRevealTerminalClaim:
            raise PhysicalFlightDeviceError("R05 terminal claim type differs")
        projection = _flight.R05TerminalClaimProjection(
            decision=claim.decision,
            selected_env_ids=claim.selected_env_ids,
            reveal_final_preview_schema_version=(
                claim.reveal_final_preview_schema_version
            ),
            reveal_final_preview_sha256=claim.reveal_final_preview_sha256,
            global_boundary_receipt_kind=claim.global_boundary_receipt_kind,
            global_boundary_receipt_sha256=(
                claim.global_boundary_receipt_sha256
            ),
            global_boundary_packet_schema_version=(
                claim.global_boundary_packet_schema_version
            ),
            global_boundary_packet_sha256=(
                claim.global_boundary_packet_sha256
            ),
            terminal_boundary_authority_sha256=(
                claim.terminal_boundary_authority_sha256
            ),
            terminal_boundary_projection_sha256=(
                claim.terminal_boundary_projection_sha256
            ),
            terminal_content_pin_sha256=(
                claim.terminal_content_pin_sha256
            ),
            terminal_kind=claim.terminal_kind,
            terminal_sha256=claim.terminal_sha256,
        )
        if projection.canonical_sha256 != claim.canonical_sha256:
            raise PhysicalFlightDeviceError(
                "R05 terminal claim canonical projection differs"
            )
        return projection

    @staticmethod
    def _r05_terminal_evidence_pins(
        claim: _r05.PreparedRevealTerminalClaim,
    ) -> tuple[
        _flight.CanonicalJsonContentPin,
        _flight.CanonicalJsonContentPin,
    ]:
        """Seal detached full claim evidence without treating it as authority."""

        if type(claim) is not _r05.PreparedRevealTerminalClaim:
            raise PhysicalFlightDeviceError("R05 terminal claim type differs")
        try:
            projection_mapping = claim.terminal_boundary_projection.to_mapping()
            content_mapping = claim.terminal_content_pin.to_mapping()
            projection_pin = _flight.CanonicalJsonContentPin.from_sealed_mapping(
                projection_mapping,
                expected_source_kind=(
                    _flight.R05_TERMINAL_BOUNDARY_PROJECTION_KIND
                ),
                source_schema_sha256=(
                    _flight.R05_TERMINAL_BOUNDARY_PROJECTION_SCHEMA_SHA256
                ),
            )
            content_pin = _flight.CanonicalJsonContentPin.from_sealed_mapping(
                content_mapping,
                expected_source_kind=(
                    _flight.R05_PREPARED_TERMINAL_CONTENT_PIN_KIND
                ),
                source_schema_sha256=(
                    _flight.R05_PREPARED_TERMINAL_CONTENT_PIN_SCHEMA_SHA256
                ),
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "R05 terminal claim full evidence cannot be sealed"
            ) from exc
        if (
            projection_pin.source_canonical_sha256
            != claim.terminal_boundary_projection_sha256
            or content_pin.source_canonical_sha256
            != claim.terminal_content_pin_sha256
        ):
            raise PhysicalFlightDeviceError(
                "R05 terminal claim full evidence root differs"
            )
        return projection_pin, content_pin

    def _require_owned_reveal_boundary_prepared_token(
        self,
        value: object,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryPreparedTokenClaim:
        handle = self._owned_prepare(value)
        receipt = handle.prepare_receipt
        return _reveal_boundary.ActionBallFullMdpRevealBoundaryPreparedTokenClaim(
            owner_kind="physical_ball",
            device_owner_mutation_version=self._device_mutation_version,
            owner_token_root_sha256=handle.canonical_sha256,
            reveal_final_preview_schema_version=(
                receipt.reveal_final_preview.source_schema_version
            ),
            reveal_final_preview_sha256=(
                receipt.reveal_final_preview.source_canonical_sha256
            ),
            _prepared_token=handle,
        )

    def _require_operable(
        self,
        *,
        allow_global_reveal_epoch: bool = False,
        allow_selected_reset: bool = False,
        allow_diagnostic_scene_observation: bool = False,
        allow_action_epoch_construction: bool = False,
        diagnostic_selected_reset_capability: object | None = None,
    ) -> None:
        stage_record = self._active_selected_reset_stage_record
        diagnostic_selected_reset_active = (
            allow_selected_reset
            and type(diagnostic_selected_reset_capability)
            is StagedPhysicalSelectedTrueReset
            and diagnostic_selected_reset_capability
            is self._active_selected_reset_stage
            and stage_record is not None
            and stage_record.capability
            is diagnostic_selected_reset_capability
        )
        if (
            self._diagnostic_n2_no_save
            and not allow_diagnostic_scene_observation
            and not allow_action_epoch_construction
            and not diagnostic_selected_reset_active
            and not self._action_epoch_runtime_call_active
        ):
            raise PhysicalFlightDeviceError(
                "diagnostic N=2 Physical owner rejects this lane; portable/legacy "
                "mutation, checkpoint, R10, and late launch remain HOLD"
            )
        self._require_operable_invariants(
            allow_global_reveal_epoch=allow_global_reveal_epoch,
            allow_selected_reset=allow_selected_reset,
        )

    def _require_operable_invariants(
        self,
        *,
        allow_global_reveal_epoch: bool = False,
        allow_selected_reset: bool = False,
    ) -> None:
        """Check fail-stop and lease state without authorizing a run mode."""

        if self._poisoned:
            raise PhysicalFlightOwnerPoisonedError(
                self._poison_reason or "physical owner is poisoned"
            )
        if self._physical_reward_poisoned:
            raise PhysicalFlightOwnerPoisonedError(
                self._physical_reward_poison_reason
                or "Physical Reward owner is poisoned"
            )
        if self._active_physical_global_drain is not None:
            raise PhysicalFlightDeviceError(
                "physical mutation cannot cross an active global PPO drain"
            )
        if (
            not allow_global_reveal_epoch
            and (
                self._active_child_terminal is not None
                or self._active_global_reveal_epoch_image is not None
            )
        ):
            raise PhysicalFlightDeviceError(
                "physical mutation cannot cross an unacknowledged global reveal epoch"
            )
        if (
            not allow_selected_reset
            and self._active_selected_reset_stage is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical mutation cannot cross an active selected-reset lease"
            )

    def _require_selected_contact_reward_cycle_closed(
        self, *, label: str
    ) -> None:
        if self._action_epoch_runtime_call_active:
            return
        if (
            self._selected_contact_reward_cycle_open
            or self._active_selected_contact_reward_view is not None
        ):
            self._selected_contact_ledger_fault.logical_or_(
                self._selected_contact_pending
            )
            raise PhysicalFlightDeviceError(
                f"{label} cannot cross an unpaid or unclosed selected-contact Reward cycle"
            )

    def _poison_physical_reward(self, reason: object) -> None:
        """Dedicated fail-stop for an irreversible Physical Reward failure."""

        if self._physical_reward_poisoned:
            return
        self._physical_reward_poison_reason = (
            reason
            if type(reason) is str and bool(reason.strip())
            else "unspecified Physical Reward failure"
        )
        self._physical_reward_poisoned = True

    def poison_global_reveal_epoch(self, reason: str) -> None:
        """Fail-stop this leaf without depending on a surviving epoch handle.

        This is the coordinator's post-transfer exception-broadcast seam.  It
        deliberately performs no scene write, chronology advance, lease
        cleanup, or rollback.  Repeated broadcasts are harmless and retain the
        first causal reason so one failing child cannot prevent the remaining
        owners from being poisoned.
        """

        if self._poisoned:
            return
        if type(reason) is not str or not reason.strip():
            reason = "unspecified global reveal epoch failure"
        self._poisoned = True
        self._poison_reason = reason

    def _require_globally_acknowledged_checkpoint_frontier(self) -> None:
        """Require current physical bytes to equal one exact healthy drain ACK."""

        if (
            self._physical_checkpoint_requires_global_drain_ack
            or self._physical_global_drain_last_acknowledged_mutation_version
            != self._mutation_version
            or self._physical_global_drain_last_update_index < 0
            or self._physical_global_drain_last_completed_environment_steps < 0
            or self._physical_global_drain_sequence < 1
            or self._physical_global_drain_poisoned
            or self._physical_global_drain_authority_type is None
            or self._physical_global_drain_ack_method is None
        ):
            raise PhysicalFlightDeviceError(
                "physical checkpoint lacks the exact globally ACKed mutation frontier"
            )

    def current_checkpoint_mutation_projection(
        self,
        boundary: object,
        owner_kind: str,
    ) -> object:
        """Project the live Physical version from this leaf's latest ACK.

        The callback is construction-bound by R10.  It accepts no caller
        version or drain receipt, never queries the global drain owner, and
        performs no device-to-host transfer.  Process-local authority/receipt
        identities are deliberately absent after cold restore, so restored
        chronology must complete a fresh global ACK before R10 can read it.
        """

        try:
            import action_ball_full_mdp_checkpoint as checkpoint
            canonical_drain = sys.modules.get(
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_full_mdp_ppo_drain"
            )
            focused_drain = sys.modules.get(
                "action_ball_full_mdp_ppo_drain"
            )
            if canonical_drain is not None:
                drain = canonical_drain
            elif focused_drain is not None:
                drain = focused_drain
            elif __package__:
                from . import action_ball_full_mdp_ppo_drain as drain
            else:
                from whole_body_tracking.tasks.tracking.mdp import (
                    action_ball_full_mdp_ppo_drain as drain,
                )
        except (ImportError, ModuleNotFoundError):
            import action_ball_full_mdp_checkpoint as checkpoint
            import action_ball_full_mdp_ppo_drain as drain

        if owner_kind != PHYSICAL_GLOBAL_DRAIN_OWNER_KIND:
            raise PhysicalFlightDeviceError(
                "physical R10 live-mutation projection role differs"
            )
        if type(boundary) is not checkpoint.CheckpointBoundary:
            raise PhysicalFlightDeviceError(
                "physical R10 live-mutation projection requires the exact checkpoint boundary"
            )
        checkpoint.validate_checkpoint_boundary(boundary)

        authority = self._physical_global_drain_authority
        receipt = self._physical_global_drain_last_acknowledged_receipt
        live_ack = self._physical_checkpoint_live_ack
        authority_type = self._physical_global_drain_authority_type
        authority_ack_method = self._physical_global_drain_ack_method
        if (
            self._poisoned
            or self._physical_global_drain_poisoned
            or self._active_physical_global_drain is not None
            or self._physical_checkpoint_requires_global_drain_ack
            or self._physical_checkpoint_live_join_required
            or type(live_ack) is not _PhysicalR10LiveMutationAck
            or live_ack.owner_identity is not self._owner_identity
            or live_ack.token is not _PHYSICAL_R10_LIVE_ACK_TOKEN
            or live_ack.authority is not authority
            or live_ack.receipt is not receipt
            or live_ack.update_index
            != self._physical_global_drain_last_update_index
            or live_ack.completed_environment_steps
            != self._physical_global_drain_last_completed_environment_steps
            or live_ack.drain_sequence != self._physical_global_drain_sequence
            or live_ack.mutation_version
            != self._physical_global_drain_last_acknowledged_mutation_version
            or type(authority) is not drain.LeafDevicePackAuthority
            or type(authority) is not authority_type
            or type(authority).require_owned_ack is not authority_ack_method
            or type(receipt) is not drain.PreOptimizerPpoBoundaryReceipt
            or receipt.acknowledged is not True
            or receipt.update_index
            != self._physical_global_drain_last_update_index
            or receipt.completed_environment_steps
            != self._physical_global_drain_last_completed_environment_steps
            or receipt.drain_sequence != self._physical_global_drain_sequence
            or receipt.num_envs != self.num_envs
            or boundary.update_index != receipt.update_index
            or len(boundary.worlds) != self.num_envs
            or tuple(world.world_id for world in boundary.worlds)
            != tuple(range(self.num_envs))
            or tuple(world.reset_generation for world in boundary.worlds)
            != tuple(self._reset_generation)
            or type(self._mutation_version) is not int
            or self._mutation_version
            != self._physical_global_drain_last_acknowledged_mutation_version
        ):
            raise PhysicalFlightDeviceError(
                "physical R10 live-mutation projection lacks its exact latest global ACK or live join"
            )
        self._require_checkpoint_idle()
        physical_rows = tuple(
            row
            for row in receipt.owner_rows
            if row.owner_kind == PHYSICAL_GLOBAL_DRAIN_OWNER_KIND
        )
        if (
            len(physical_rows) != 1
            or self._physical_global_owner_row_values(physical_rows[0]).get(
                "mutation_version"
            )
            != self._mutation_version
        ):
            raise PhysicalFlightDeviceError(
                "physical R10 live mutation differs from its ACKed owner row"
            )
        retained = self._physical_checkpoint_last_live_projection
        if retained is not None:
            if (
                self._physical_checkpoint_last_live_boundary is boundary
                and self._physical_checkpoint_last_live_receipt is receipt
            ):
                return retained
            raise PhysicalFlightDeviceError(
                "physical R10 live-mutation boundary is foreign or replayed"
            )
        projection = checkpoint.PpoDrainLeafLiveMutationProjection(
            schema_version=1,
            kind="action_ball_r10_leaf_live_mutation_projection_v1",
            owner_kind=PHYSICAL_GLOBAL_DRAIN_OWNER_KIND,
            mutation_version=self._mutation_version,
        )
        self._physical_checkpoint_last_live_projection = projection
        self._physical_checkpoint_last_live_boundary = boundary
        self._physical_checkpoint_last_live_receipt = receipt
        return projection

    def _require_checkpoint_idle(
        self,
    ) -> None:
        if (
            self._active_prepare is not None
            or self._active_armed is not None
            or self._active_child_terminal is not None
            or self._active_global_reveal_epoch_image is not None
            or self._active_reveal_boundary_row is not None
            or self._active_postphysics is not None
            or self._active_postphysics_capture is not None
            or self._active_postphysics_capture_image is not None
            or self._action_epoch_substep_pair is not None
            or self._active_r06_ack is not None
            or self._active_physical_hot_prepare is not None
            or self._active_physical_hot_commit is not None
            or self._active_physical_late_launch is not None
            or self._active_physical_retire_prepare is not None
            or self._active_physical_retire_arm is not None
            or self._active_physical_retire_commit is not None
            or self._active_selected_reset_stage is not None
            or self._active_selected_reset_finalize is not None
            or self._active_selected_reset_arm is not None
            or self._active_selected_reset_commit is not None
            or self._selected_reset_completion_token is not None
            or self._active_selected_contact_reward_view is not None
            or self._selected_contact_reward_cycle_open
            or self._active_physical_global_drain is not None
        ):
            raise PhysicalFlightDeviceError(
                "checkpoint cannot cross a physical reveal/postphysics/retire/reset/contact-payment lease"
            )

    def _require_cpu_reference(self) -> None:
        if self.device.type != "cpu":
            raise PhysicalFlightDeviceError(
                "CUDA physical transaction is fail-closed until the packed reveal boundary is integrated"
            )

    def _set_owner_mutation_version(self, value: int) -> None:
        version = _exact_int(value, label="owner mutation version")
        self._mutation_version = version
        self._device_mutation_version.fill_(version)
        self._physical_checkpoint_requires_global_drain_ack = True
        self._physical_checkpoint_live_ack = None
        self._physical_checkpoint_last_live_projection = None
        self._physical_checkpoint_last_live_boundary = None
        self._physical_checkpoint_last_live_receipt = None

    def _advance_owner_mutation_version(self) -> None:
        self._set_owner_mutation_version(self._mutation_version + 1)

    def _accumulate_normal_retire_key_summaries_(
        self,
        normal_mask: torch.Tensor,
        full_key_sha256: torch.Tensor,
    ) -> None:
        """Accumulate the shared physical/R06 key multiset on device."""

        key_bytes = full_key_sha256.to(torch.int64) + 1
        summaries = []
        for base in (257, 263):
            weights = torch.tensor(
                tuple(
                    pow(base, index, _PHYSICAL_RETIRE_SUMMARY_MODULUS)
                    for index in range(TOKEN_BYTES)
                ),
                dtype=torch.int64,
                device=self.device,
            )
            row_hash = torch.remainder(
                (key_bytes * weights).sum(dim=-1),
                _PHYSICAL_RETIRE_SUMMARY_MODULUS,
            )
            summaries.append(
                torch.remainder(
                    (row_hash * normal_mask.to(torch.int64)).sum(),
                    _PHYSICAL_RETIRE_SUMMARY_MODULUS,
                )
            )
        self._shared_normal_retire_key_summaries.copy_(
            torch.remainder(
                self._shared_normal_retire_key_summaries
                + torch.stack(summaries),
                _PHYSICAL_RETIRE_SUMMARY_MODULUS,
            )
        )

    def _apply_scene_write(self, handle: object) -> object:
        receipt = self.scene_port.apply_prevalidated_write(handle)
        owned = self.scene_port.require_owned_apply_receipt(handle, receipt)
        if owned is not receipt:
            raise PhysicalFlightDeviceError(
                "scene apply receipt identity differs"
            )
        if self._action_epoch_active_scene_write is not None:
            self._action_epoch_active_scene_write = None
        return receipt

    def _abort_scene_write(self, handle: object) -> object:
        receipt = self.scene_port.abort_prevalidated_write(handle)
        owned = self.scene_port.require_owned_abort_receipt(handle, receipt)
        if owned is not receipt:
            raise PhysicalFlightDeviceError(
                "scene abort receipt identity differs"
            )
        if self._action_epoch_active_scene_write is not None:
            self._action_epoch_active_scene_write = None
        return receipt

    def _slot_offset(self, env_id: int, slot_index: int) -> int:
        return env_id * self.flight_capacity + slot_index

    def _parked_snapshot(
        self,
        *,
        env_id: int,
        slot_index: int,
        state: _flight.CanonicalPhysicalBallStateF32,
        version: int,
    ) -> _flight.PhysicalFlightSlotSnapshot:
        del state  # Parked portable snapshots deliberately carry no live state.
        return _flight.PhysicalFlightSlotSnapshot(
            capacity_receipt_sha256=self.capacity_receipt_sha256,
            capacity_value=self.flight_capacity,
            env_id=env_id,
            slot_index=slot_index,
            scene_body_name=f"env{env_id:06d}/{self.scene_body_names[slot_index]}",
            lifecycle=_flight.SLOT_PARKED,
            ball_generation=None,
            inbound_ball_sha256=None,
            outcome_key=None,
            outcome_key_sha256=None,
            install_payload_sha256=None,
            installed_ball_state_sha256=None,
            current_state_f32=None,
            current_state_f32_sha256=None,
            reveal_control_step=None,
            last_control_step=0,
            last_physics_substep=0,
            last_sim_step=0,
            mutation_version=version,
            physically_parked=True,
            published_to_runtime=False,
        )

    def _owner_checkpoint_sha_for(
        self,
        *,
        slots: tuple[_flight.PhysicalFlightSlotSnapshot, ...],
        mutation_version: int,
        next_prepare_nonce: int,
        poisoned: bool,
        reset_generations: Optional[Sequence[int]] = None,
    ) -> str:
        reset_values = tuple(
            self._reset_generation if reset_generations is None else reset_generations
        )
        return _flight.physical_owner_checkpoint_root(
            capacity_receipt_sha256=self.capacity_receipt_sha256,
            num_envs=self.num_envs,
            flight_capacity=self.flight_capacity,
            mutation_version=mutation_version,
            next_prepare_nonce=next_prepare_nonce,
            reset_generations=reset_values,
            slots=slots,
            poisoned=poisoned,
        )

    def _owner_checkpoint_sha(self) -> str:
        return self._owner_checkpoint_sha_for(
            slots=self._host_slots,
            mutation_version=self._mutation_version,
            next_prepare_nonce=self._next_prepare_nonce,
            poisoned=self._poisoned,
        )

    def scene_snapshot(self) -> PhysicalFlightSceneSnapshotNK:
        self._require_operable(allow_diagnostic_scene_observation=True)
        return PhysicalFlightSceneSnapshotNK(
            state_env_f32=_tensor(
                self.scene_port.read_state_env(),
                label="scene state",
                shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
                dtype=torch.float32,
                device=self.device,
            ).detach().clone(),
            lifecycle_code=self._lifecycle.clone(),
            ball_generation=self._generation.clone(),
            outcome_key_sha256=self._outcome_sha.clone(),
            install_payload_sha256=self._install_sha.clone(),
            installed_ball_state_sha256=self._installed_state_sha.clone(),
            reveal_control_step=self._reveal_step.clone(),
            observation_ordinal=self._observation_ordinal.clone(),
            physically_parked=self._parked.clone(),
            published_to_runtime=self._published.clone(),
            owner_fault=self._device_fault.clone(),
            slot_mutation_version=self._slot_version.clone(),
            owner_mutation_version=self._mutation_version,
        )

    def _pending_r06_settlement_ack_for_legacy_checkpoint_rejection(
        self,
    ) -> Optional[AcknowledgedR06PhysicalSnapshot]:
        """Return the owner-minted opaque authority restored at a checkpoint."""

        self._require_operable()
        return self._active_r06_ack

    def _rebind_restored_r06_settlement_owner_for_legacy_rejection(
        self,
        ack: AcknowledgedR06PhysicalSnapshot,
        *,
        r06_owner: object,
    ) -> None:
        """Cross-bind a restored pending ack to the separately restored R06 owner."""

        self._require_operable()
        self._require_cpu_reference()
        if (
            type(ack) is not AcknowledgedR06PhysicalSnapshot
            or ack._token is not _R06_ACK_TOKEN
            or ack._owner_identity is not self._owner_identity
            or self._active_r06_ack is not ack
            or self._active_r06_ack_image is None
            or self._active_r06_ack_image.r06_owner is not None
        ):
            raise PhysicalFlightDeviceError(
                "restored R06 settlement acknowledgement is stale or already bound"
            )
        owner = _require_final_r06_type(
            r06_owner,
            expected_name="ActionBallLandingOutcomeDeviceCoordinator",
        )
        image = self._active_r06_ack_image
        snapshot = _require_final_r06_type(
            owner.current_flight_lifecycle_snapshot(),
            expected_name="FlightLifecycleSnapshotBatch",
        )
        shape = (self.num_envs, self.flight_capacity)
        state = _tensor(
            snapshot.state,
            label="restored R06 state",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        key = _tensor(
            snapshot.full_key_sha256,
            label="restored R06 key",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        generation = _tensor(
            snapshot.ball_generation,
            label="restored R06 generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        mailbox_slot = _tensor(
            snapshot.mailbox_slot,
            label="restored R06 mailbox slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        version = _r06_mutation_version(
            snapshot.mutation_version,
            label="restored R06 mutation_version",
            device=self.device,
        )
        root = r06_physical_snapshot_root(
            R06PhysicalFlightReadOnlySnapshot(
                flight_state=state,
                full_key_sha256=key,
                ball_generation=generation,
                observation_ordinal=image.observation_ordinal,
                owner_mutation_version=version,
            )
        )
        mailbox_state = owner.mailbox_state
        if (
            root != image.snapshot_root_sha256
            or version != image.owner_mutation_version
            or not torch.equal(mailbox_slot, image.mailbox_slot)
            or not isinstance(mailbox_state, torch.Tensor)
            or mailbox_state.device != self.device
            or mailbox_state.dtype != torch.int8
            or mailbox_state.ndim != 2
            or mailbox_state.shape[0] != self.num_envs
            or mailbox_state.shape[1] < 1
        ):
            raise PhysicalFlightDeviceError(
                "restored physical/R06 checkpoint join differs"
            )
        settled = state == R06_FLIGHT_SETTLED_RETAINED
        safe_mailbox = mailbox_slot.clamp(0, mailbox_state.shape[1] - 1)
        gathered = torch.gather(
            mailbox_state.to(torch.int64), 1, safe_mailbox
        )
        if bool(
            (
                settled
                & (
                    (mailbox_slot < 0)
                    | (mailbox_slot >= mailbox_state.shape[1])
                    | (gathered != 1)
                )
            ).any()
        ):
            raise PhysicalFlightDeviceError(
                "restored R06 settlement mailbox join differs"
            )
        self._active_r06_ack_image = replace(image, r06_owner=owner)

    def prepare_install(
        self,
        *,
        reveal_final_preview: _flight.CanonicalJsonContentPin,
        expected_reveal_final_preview_sha256: str,
        install_payloads: Sequence[_flight.PhysicalBallInstallPayload],
        expected_install_payload_sha256s: Sequence[str],
    ) -> PreparedPhysicalFlightInstall:
        """Production tombstone for the former caller-authored preview pin."""

        del (
            reveal_final_preview,
            expected_reveal_final_preview_sha256,
            install_payloads,
            expected_install_payload_sha256s,
        )
        raise PhysicalFlightDeviceError(
            "prepare_install is a production tombstone; use "
            "prepare_from_reveal_final_preview"
        )

    def prepare_from_reveal_final_preview(
        self,
        reveal_final_preview: _r05.RevealFinalPreviewBatch,
        *,
        install_payloads: Sequence[_flight.PhysicalBallInstallPayload],
        expected_install_payload_sha256s: Sequence[str],
    ) -> PreparedPhysicalFlightInstall:
        """Prepare from the exact active R05 object; terminal claim stays mandatory."""

        self._require_operable()
        if not self._host_reset_generation_projection_current:
            raise PhysicalFlightDeviceError(
                "portable R05 physical ingress is diagnostic-only after a device selected reset"
            )
        if type(reveal_final_preview) is not _r05.RevealFinalPreviewBatch:
            raise PhysicalFlightDeviceError(
                "reveal-final preview must be the exact R05 type"
            )
        r05_owner = self._r05_terminal_owner
        if type(r05_owner) is not _r05.ContinuousRuntimeTransactionOwner:
            raise PhysicalFlightDeviceError(
                "physical prepare has no exact bound R05 owner"
            )
        preview_root = reveal_final_preview.canonical_sha256
        try:
            owned_preview = r05_owner.require_owned_active_reveal_final_preview(
                reveal_final_preview,
                expected_reveal_final_preview_sha256=preview_root,
            )
            if (
                owned_preview is reveal_final_preview
                or owned_preview.canonical_sha256 != preview_root
            ):
                raise PhysicalFlightDeviceError(
                    "physical retained R05 preview image differs"
                )
            mapping = owned_preview.to_mapping()
            preview_pin = _flight.CanonicalJsonContentPin.from_sealed_mapping(
                mapping,
                expected_source_kind=_flight.REVEAL_FINAL_PREVIEW_KIND,
                source_schema_sha256=(
                    _flight.R05_REVEAL_FINAL_PREVIEW_SCHEMA_SHA256
                ),
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "R05 reveal-final preview canonical bytes differ"
            ) from exc
        if (
            preview_pin.source_canonical_sha256
            != preview_root
        ):
            raise PhysicalFlightDeviceError(
                "R05 reveal-final preview root differs"
            )
        if self.device.type == "cpu":
            return self._prepare_install_from_content_pin_cpu_reference(
                reveal_final_preview=preview_pin,
                expected_reveal_final_preview_sha256=(
                    preview_pin.source_canonical_sha256
                ),
                install_payloads=install_payloads,
                expected_install_payload_sha256s=(
                    expected_install_payload_sha256s
                ),
            )
        return self._prepare_install_from_content_pin_device(
            reveal_final_preview=preview_pin,
            expected_reveal_final_preview_sha256=(
                preview_pin.source_canonical_sha256
            ),
            install_payloads=install_payloads,
            expected_install_payload_sha256s=expected_install_payload_sha256s,
        )

    def _prepare_install_from_content_pin_device(
        self,
        *,
        reveal_final_preview: _flight.CanonicalJsonContentPin,
        expected_reveal_final_preview_sha256: str,
        install_payloads: Sequence[_flight.PhysicalBallInstallPayload],
        expected_install_payload_sha256s: Sequence[str],
    ) -> PreparedPhysicalFlightInstall:
        """Prepare one CUDA after-image without materializing live state on host."""

        self._require_operable()
        self._require_selected_contact_reward_cycle_closed(label="reveal prepare")
        if self.device.type == "cpu":
            raise PhysicalFlightDeviceError(
                "device install helper requires a non-CPU owner"
            )
        if self._active_prepare is not None:
            raise PhysicalFlightDeviceError("one physical prepare is already active")
        if self._active_postphysics is not None:
            raise PhysicalFlightDeviceError(
                "prepare cannot cross postphysics acknowledgement"
            )
        if self._active_r06_ack is not None:
            raise PhysicalFlightDeviceError(
                "prepare cannot cross an unconsumed R06 settlement authority"
            )
        if type(reveal_final_preview) is not _flight.CanonicalJsonContentPin:
            raise PhysicalFlightDeviceError(
                "reveal preview must contain full canonical bytes"
            )
        expected_preview = _sha256(
            expected_reveal_final_preview_sha256,
            label="expected_reveal_final_preview_sha256",
        )
        if reveal_final_preview.source_canonical_sha256 != expected_preview:
            raise PhysicalFlightDeviceError("reveal preview external pin differs")
        payloads = tuple(install_payloads)
        expected_payloads = tuple(expected_install_payload_sha256s)
        if not payloads or len(payloads) != len(expected_payloads):
            raise PhysicalFlightDeviceError("install payload batch width differs")
        if any(type(row) is not _flight.PhysicalBallInstallPayload for row in payloads):
            raise PhysicalFlightDeviceError("install payload type differs")
        env_ids = tuple(row.env_id for row in payloads)
        if env_ids != tuple(sorted(set(env_ids))) or any(
            env_id >= self.num_envs for env_id in env_ids
        ):
            raise PhysicalFlightDeviceError(
                "install env ids must be sorted/unique/in-range"
            )
        if tuple(row.canonical_sha256 for row in payloads) != expected_payloads:
            raise PhysicalFlightDeviceError("install payload external pin differs")
        if any(
            row.capacity_receipt != self.capacity_receipt
            or row.capacity_receipt_sha256 != self.capacity_receipt_sha256
            or row.flight_slot >= self.flight_capacity
            for row in payloads
        ):
            raise PhysicalFlightDeviceError("install payload capacity differs")
        if any(
            row.task_ref.reset_generation != self._reset_generation[row.env_id]
            for row in payloads
        ):
            raise PhysicalFlightDeviceError(
                "install reset-generation binding differs"
            )

        scene_before = _tensor(
            self.scene_port.read_state_env(),
            label="device install scene state",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        ).detach().clone()
        after_state = scene_before.clone()
        selected_mask = torch.zeros(
            (self.num_envs, self.flight_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        for payload in payloads:
            after_state[payload.env_id, payload.flight_slot].copy_(
                _canonical_state_tensor(payload.state_f32, device=self.device)
            )
            selected_mask[payload.env_id, payload.flight_slot] = True

        finite_after = torch.isfinite(after_state).all(dim=-1)
        parked_scene = torch.eq(
            scene_before,
            self._park_state_template,
        ).all(dim=-1)
        valid_precondition = (
            finite_after
            & parked_scene
            & (self._lifecycle == R06_FLIGHT_EMPTY)
            & self._parked
            & ~self._published
            & ~self._device_fault
        )
        prepare_fault_by_env = (
            selected_mask & ~valid_precondition
        ).any(dim=1)
        receipt = PhysicalDeviceInstallPrepareReceipt(
            schema_version=1,
            kind=PhysicalDeviceInstallPrepareReceipt.KIND,
            integration_status=_flight.INTEGRATION_STATUS,
            capacity_receipt_sha256=self.capacity_receipt_sha256,
            reveal_final_preview=reveal_final_preview,
            num_envs=self.num_envs,
            reset_generations=tuple(self._reset_generation),
            mutation_version_before=self._mutation_version,
            prepare_nonce=self._next_prepare_nonce,
            selected_env_ids=env_ids,
            selected_slot_indices=tuple(row.flight_slot for row in payloads),
            install_payload_sha256s=expected_payloads,
            device_preconditions_bound_in_boundary_row=True,
            live_state_mutated=False,
            runtime_publication_created=False,
        )
        scene_handle = self.scene_port.preflight_write(
            after_state,
            selected_mask,
            device_faults_bound_in_reveal_row=True,
        )
        handle = PreparedPhysicalFlightInstall(
            _prepare_receipt=receipt,
            _owner_identity=self._owner_identity,
            _token=_PREPARED_TOKEN,
        )
        self._active_prepare = handle
        self._active_prepare_image = _PreparedPhysicalInstallImage(
            after_slots=(),
            after_scene_state=after_state,
            selected_mask=selected_mask,
            prepare_fault_by_env=prepare_fault_by_env,
            install_payloads=payloads,
            scene_handle=scene_handle,
            stale_witness=_PhysicalPrearmStaleWitness(
                scene_state=scene_before,
                lifecycle=self._lifecycle.clone(),
                generation=self._generation.clone(),
                outcome_sha=self._outcome_sha.clone(),
                install_sha=self._install_sha.clone(),
                installed_state_sha=self._installed_state_sha.clone(),
                reveal_step=self._reveal_step.clone(),
                observation_ordinal=self._observation_ordinal.clone(),
                previous_ball_center=self._previous_ball_center.clone(),
                parked=self._parked.clone(),
                published=self._published.clone(),
                slot_version=self._slot_version.clone(),
                device_fault=self._device_fault.clone(),
                owner_mutation_version=self._mutation_version,
                next_prepare_nonce=self._next_prepare_nonce,
            ),
        )
        return handle

    def _prepare_install_from_content_pin_cpu_reference(
        self,
        *,
        reveal_final_preview: _flight.CanonicalJsonContentPin,
        expected_reveal_final_preview_sha256: str,
        install_payloads: Sequence[_flight.PhysicalBallInstallPayload],
        expected_install_payload_sha256s: Sequence[str],
    ) -> PreparedPhysicalFlightInstall:
        """Unit-only content-pin helper; production callers pass exact R05."""

        self._require_operable()
        self._require_selected_contact_reward_cycle_closed(label="reveal prepare")
        self._require_cpu_reference()
        if self._active_prepare is not None:
            raise PhysicalFlightDeviceError("one physical prepare is already active")
        if self._active_postphysics is not None:
            raise PhysicalFlightDeviceError("prepare cannot cross postphysics acknowledgement")
        if self._active_r06_ack is not None:
            raise PhysicalFlightDeviceError(
                "prepare cannot cross an unconsumed R06 settlement authority"
            )
        if type(reveal_final_preview) is not _flight.CanonicalJsonContentPin:
            raise PhysicalFlightDeviceError("reveal preview must contain full canonical bytes")
        expected_preview = _sha256(
            expected_reveal_final_preview_sha256,
            label="expected_reveal_final_preview_sha256",
        )
        if reveal_final_preview.source_canonical_sha256 != expected_preview:
            raise PhysicalFlightDeviceError("reveal preview external pin differs")
        payloads = tuple(install_payloads)
        expected_payloads = tuple(expected_install_payload_sha256s)
        if not payloads or len(payloads) != len(expected_payloads):
            raise PhysicalFlightDeviceError("install payload batch width differs")
        if any(type(row) is not _flight.PhysicalBallInstallPayload for row in payloads):
            raise PhysicalFlightDeviceError("install payload type differs")
        env_ids = tuple(row.env_id for row in payloads)
        if env_ids != tuple(sorted(set(env_ids))) or any(
            env_id >= self.num_envs for env_id in env_ids
        ):
            raise PhysicalFlightDeviceError("install env ids must be sorted/unique/in-range")
        if tuple(row.canonical_sha256 for row in payloads) != expected_payloads:
            raise PhysicalFlightDeviceError("install payload external pin differs")
        if any(
            row.capacity_receipt != self.capacity_receipt
            or row.capacity_receipt_sha256 != self.capacity_receipt_sha256
            for row in payloads
        ):
            raise PhysicalFlightDeviceError("install payload capacity differs")
        if any(
            row.task_ref.reset_generation != self._reset_generation[row.env_id]
            for row in payloads
        ):
            raise PhysicalFlightDeviceError("install reset-generation binding differs")
        before_checkpoint = self._owner_checkpoint_sha()
        complete_owner_grid = self._host_slots
        prepared_rows: list[_flight.PreparedPhysicalInstallRow] = []
        after_slots = list(self._host_slots)
        scene_before = self.scene_port.read_state_env().clone()
        after_state = scene_before.clone()
        selected_mask = torch.zeros(
            (self.num_envs, self.flight_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        for payload in payloads:
            offset = self._slot_offset(payload.env_id, payload.flight_slot)
            pre = self._host_slots[offset]
            prepared_rows.append(
                _flight.PreparedPhysicalInstallRow(
                    env_id=payload.env_id,
                    slot_index=payload.flight_slot,
                    pre_slot_snapshot=pre,
                    pre_slot_snapshot_sha256=pre.canonical_sha256,
                    install_payload=payload,
                    install_payload_sha256=payload.canonical_sha256,
                )
            )
            live = _flight.PhysicalFlightSlotSnapshot(
                capacity_receipt_sha256=self.capacity_receipt_sha256,
                capacity_value=self.flight_capacity,
                env_id=payload.env_id,
                slot_index=payload.flight_slot,
                scene_body_name=pre.scene_body_name,
                lifecycle=_flight.SLOT_IN_FLIGHT,
                ball_generation=payload.ball_generation,
                inbound_ball_sha256=payload.inbound_ball_sha256,
                outcome_key=payload.outcome_key,
                outcome_key_sha256=payload.outcome_key_sha256,
                install_payload_sha256=payload.canonical_sha256,
                installed_ball_state_sha256=payload.installed_ball_state_sha256,
                current_state_f32=payload.state_f32,
                current_state_f32_sha256=payload.state_f32_sha256,
                reveal_control_step=payload.reveal_control_step,
                last_control_step=payload.reveal_control_step,
                last_physics_substep=0,
                last_sim_step=0,
                mutation_version=pre.mutation_version + 1,
                physically_parked=False,
                published_to_runtime=True,
            )
            after_slots[offset] = live
            after_state[payload.env_id, payload.flight_slot].copy_(
                _canonical_state_tensor(payload.state_f32, device=self.device)
            )
            selected_mask[payload.env_id, payload.flight_slot] = True

        if not bool(torch.isfinite(after_state[selected_mask]).all()):
            raise PhysicalFlightDeviceError("selected physical install state is nonfinite")

        receipt = _flight.PhysicalInstallPrepareReceipt(
            integration_status=_flight.INTEGRATION_STATUS,
            capacity_receipt_sha256=self.capacity_receipt_sha256,
            reveal_final_preview=reveal_final_preview,
            num_envs=self.num_envs,
            reset_generations=tuple(self._reset_generation),
            physical_owner_checkpoint_before_sha256=before_checkpoint,
            mutation_version_before=self._mutation_version,
            prepare_nonce=self._next_prepare_nonce,
            selected_env_ids=env_ids,
            pre_slot_snapshots=complete_owner_grid,
            rows=tuple(prepared_rows),
            pre_slots_root_sha256=_flight.physical_slot_root(complete_owner_grid),
            live_state_mutated=False,
            runtime_publication_created=False,
        )
        scene_handle = self.scene_port.preflight_write(
            after_state,
            selected_mask,
            device_faults_bound_in_reveal_row=True,
        )
        handle = PreparedPhysicalFlightInstall(
            _prepare_receipt=receipt,
            _owner_identity=self._owner_identity,
            _token=_PREPARED_TOKEN,
        )
        self._active_prepare = handle
        self._active_prepare_image = _PreparedPhysicalInstallImage(
            after_slots=tuple(after_slots),
            after_scene_state=after_state,
            selected_mask=selected_mask,
            prepare_fault_by_env=torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            ),
            install_payloads=payloads,
            scene_handle=scene_handle,
            stale_witness=_PhysicalPrearmStaleWitness(
                scene_state=scene_before,
                lifecycle=self._lifecycle.clone(),
                generation=self._generation.clone(),
                outcome_sha=self._outcome_sha.clone(),
                install_sha=self._install_sha.clone(),
                installed_state_sha=self._installed_state_sha.clone(),
                reveal_step=self._reveal_step.clone(),
                observation_ordinal=self._observation_ordinal.clone(),
                previous_ball_center=self._previous_ball_center.clone(),
                parked=self._parked.clone(),
                published=self._published.clone(),
                slot_version=self._slot_version.clone(),
                device_fault=self._device_fault.clone(),
                owner_mutation_version=self._mutation_version,
                next_prepare_nonce=self._next_prepare_nonce,
            ),
        )
        return handle

    def _owned_prepare(self, value: object) -> PreparedPhysicalFlightInstall:
        if (
            type(value) is not PreparedPhysicalFlightInstall
            or value._token is not _PREPARED_TOKEN
            or value._owner_identity is not self._owner_identity
            or self._active_prepare is not value
            or self._active_prepare_image is None
            or self._active_armed is not None
        ):
            raise PhysicalFlightDeviceError("physical prepare token is stale or foreign")
        return value

    def _owned_armed(self, value: object) -> ArmedPhysicalFlightInstall:
        if (
            type(value) is not ArmedPhysicalFlightInstall
            or value._token is not _ARMED_TOKEN
            or value._owner_identity is not self._owner_identity
            or self._active_armed is not value
            or self._active_armed_image is None
        ):
            raise PhysicalFlightDeviceError(
                "physical armed token is stale or foreign"
            )
        return value

    def _require_prearm_stale_witness(
        self, image: _PreparedPhysicalInstallImage
    ) -> None:
        witness = image.stale_witness
        try:
            current_scene = _tensor(
                self.scene_port.read_state_env(),
                label="prearm current scene state",
                shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
                dtype=torch.float32,
                device=self.device,
            )
            matches = (
                torch.equal(current_scene, witness.scene_state)
                and torch.equal(self._lifecycle, witness.lifecycle)
                and torch.equal(self._generation, witness.generation)
                and torch.equal(self._outcome_sha, witness.outcome_sha)
                and torch.equal(self._install_sha, witness.install_sha)
                and torch.equal(
                    self._installed_state_sha,
                    witness.installed_state_sha,
                )
                and torch.equal(self._reveal_step, witness.reveal_step)
                and torch.equal(
                    self._observation_ordinal,
                    witness.observation_ordinal,
                )
                and torch.equal(
                    self._previous_ball_center,
                    witness.previous_ball_center,
                )
                and torch.equal(self._parked, witness.parked)
                and torch.equal(self._published, witness.published)
                and torch.equal(self._slot_version, witness.slot_version)
                and torch.equal(self._device_fault, witness.device_fault)
                and self._mutation_version == witness.owner_mutation_version
                and self._next_prepare_nonce == witness.next_prepare_nonce
                and self._active_postphysics is None
                and self._active_r06_ack is None
            )
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "physical prearm stale witness could not be read"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        if not matches:
            self._poisoned = True
            self._poison_reason = (
                "physical scene/device state drifted after boundary-free prepare"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)

    def _device_prearm_fault_by_env(
        self,
        image: _PreparedPhysicalInstallImage,
    ) -> torch.Tensor:
        """Return CUDA-resident prearm faults without a host truth conversion."""

        witness = image.stale_witness
        if (
            self._mutation_version != witness.owner_mutation_version
            or self._next_prepare_nonce != witness.next_prepare_nonce
            or self._active_postphysics is not None
            or self._active_r06_ack is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical owner chronology changed after device prepare"
            )
        current_scene = _tensor(
            self.scene_port.read_state_env(),
            label="device prearm current scene state",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        slot_fault = torch.ne(current_scene, witness.scene_state).any(dim=-1)
        slot_fault |= torch.ne(self._lifecycle, witness.lifecycle)
        slot_fault |= torch.ne(self._generation, witness.generation)
        slot_fault |= torch.ne(self._outcome_sha, witness.outcome_sha).any(dim=-1)
        slot_fault |= torch.ne(self._install_sha, witness.install_sha).any(dim=-1)
        slot_fault |= torch.ne(
            self._installed_state_sha,
            witness.installed_state_sha,
        ).any(dim=-1)
        slot_fault |= torch.ne(self._reveal_step, witness.reveal_step)
        slot_fault |= torch.ne(
            self._observation_ordinal,
            witness.observation_ordinal,
        )
        slot_fault |= torch.ne(
            self._previous_ball_center,
            witness.previous_ball_center,
        ).any(dim=-1)
        slot_fault |= torch.ne(self._parked, witness.parked)
        slot_fault |= torch.ne(self._published, witness.published)
        slot_fault |= torch.ne(self._slot_version, witness.slot_version)
        slot_fault |= torch.ne(self._device_fault, witness.device_fault)
        return image.prepare_fault_by_env | slot_fault.any(dim=1)

    def prearm_reveal_boundary(
        self,
        prepared: PreparedPhysicalFlightInstall,
        *,
        boundary_owner: _reveal_boundary.ActionBallFullMdpRevealBoundaryOwner,
        lane_authority: _reveal_boundary.ActionBallFullMdpRevealBoundaryLaneAuthority,
    ) -> _reveal_boundary.ActionBallFullMdpRevealBoundaryDeviceRow:
        """Mint the boundary-free physical row through the pre-bound exact lane."""

        self._require_operable()
        handle = self._owned_prepare(prepared)
        image = self._active_prepare_image
        if image is None:
            raise PhysicalFlightDeviceError("physical private after-image is missing")
        if self.device.type == "cpu":
            self._require_prearm_stale_witness(image)
            fault_by_env = image.prepare_fault_by_env
        else:
            fault_by_env = self._device_prearm_fault_by_env(image)
        if (
            boundary_owner is not self._reveal_boundary_owner
            or lane_authority is not self._reveal_boundary_lane_authority
            or self._reveal_boundary_owner is None
            or self._reveal_boundary_lane_authority is None
        ):
            raise PhysicalFlightDeviceError(
                "physical reveal boundary owner/lane is not the exact pre-bound pair"
            )
        if self._active_reveal_boundary_row is not None:
            self._poisoned = True
            self._poison_reason = "physical reveal boundary row was already minted"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        try:
            selected_mask = image.selected_mask.any(dim=1)
            pass_mask = selected_mask & ~fault_by_env
            fault_bits = (
                selected_mask & fault_by_env
            ).to(dtype=torch.int64)
            row = lane_authority.mint_device_row(
                prepared_token=handle,
                selected_env_ids=handle.prepare_receipt.selected_env_ids,
                pass_mask=pass_mask,
                fault_bits=fault_bits,
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "physical reveal boundary row prearm failed"
            ) from exc
        self._active_reveal_boundary_row = row
        return row

    def abort_install(
        self,
        value: PreparedPhysicalFlightInstall | ArmedPhysicalFlightInstall,
        *,
        boundary_abort_capability: Optional[
            _reveal_boundary.ActionBallFullMdpRevealBoundaryAbortCapability
        ] = None,
    ) -> _flight.PhysicalInstallAbortReceipt | PhysicalDeviceInstallAbortReceipt:
        self._require_operable()
        if type(value) is PreparedPhysicalFlightInstall:
            handle = self._owned_prepare(value)
        elif type(value) is ArmedPhysicalFlightInstall:
            armed = self._owned_armed(value)
            handle = armed._prepared_install
        else:
            raise PhysicalFlightDeviceError(
                "physical abort token is stale or foreign"
            )
        image = self._active_prepare_image
        if image is None:
            raise PhysicalFlightDeviceError("physical private after-image is missing")
        checkpoint: str | None = None
        if self.device.type == "cpu":
            checkpoint = self._owner_checkpoint_sha()
            if (
                checkpoint
                != handle.prepare_receipt.physical_owner_checkpoint_before_sha256
            ):
                raise PhysicalFlightDeviceError("physical owner changed after prepare")
            self._require_prearm_stale_witness(image)
        elif (
            self._mutation_version
            != handle.prepare_receipt.mutation_version_before
        ):
            raise PhysicalFlightDeviceError("physical owner changed after prepare")
        row = self._active_reveal_boundary_row
        lane = self._reveal_boundary_lane_authority
        if row is not None:
            if lane is None:
                raise PhysicalFlightDeviceError(
                    "physical reveal boundary lane disappeared"
                )
            try:
                lane.require_abortable_device_row(
                    row,
                    expected_prepared_token=handle,
                    abort_capability=boundary_abort_capability,
                )
            except Exception as exc:
                raise PhysicalFlightDeviceError(
                    "physical reveal boundary row is not abortable"
                ) from exc
        elif boundary_abort_capability is not None:
            raise PhysicalFlightDeviceError(
                "physical prepare has no boundary row for this abort capability"
            )
        self._abort_scene_write(image.scene_handle)
        self._active_prepare = None
        self._active_prepare_image = None
        self._active_armed = None
        self._active_armed_image = None
        self._active_reveal_boundary_row = None
        if self.device.type != "cpu":
            return PhysicalDeviceInstallAbortReceipt(
                schema_version=1,
                kind=PhysicalDeviceInstallAbortReceipt.KIND,
                prepare_receipt_sha256=handle.prepare_receipt.canonical_sha256,
                mutation_version_before=self._mutation_version,
                mutation_version_after=self._mutation_version,
                live_state_mutated=False,
                runtime_publication_created=False,
            )
        assert checkpoint is not None
        return _flight.PhysicalInstallAbortReceipt(
            integration_status=_flight.INTEGRATION_STATUS,
            prepare_receipt=handle.prepare_receipt,
            prepare_receipt_sha256=handle.prepare_receipt.canonical_sha256,
            physical_owner_checkpoint_before_sha256=checkpoint,
            physical_owner_checkpoint_after_sha256=checkpoint,
            mutation_version_before=self._mutation_version,
            mutation_version_after=self._mutation_version,
            live_state_mutated=False,
            runtime_publication_created=False,
        )

    @staticmethod
    def _r05_terminal_expectations(
        claim: _r05.PreparedRevealTerminalClaim,
    ) -> dict[str, object]:
        return {
            "expected_claim_sha256": claim.canonical_sha256,
            "expected_decision": claim.decision,
            "expected_reveal_final_preview_sha256": (
                claim.reveal_final_preview_sha256
            ),
            "expected_global_boundary_receipt_sha256": (
                claim.global_boundary_receipt_sha256
            ),
            "expected_global_boundary_packet_sha256": (
                claim.global_boundary_packet_sha256
            ),
            "expected_terminal_boundary_authority_sha256": (
                claim.terminal_boundary_authority_sha256
            ),
            "expected_terminal_boundary_projection_sha256": (
                claim.terminal_boundary_projection_sha256
            ),
            "expected_terminal_content_pin_sha256": (
                claim.terminal_content_pin_sha256
            ),
            "expected_terminal_kind": claim.terminal_kind,
            "expected_terminal_sha256": claim.terminal_sha256,
            "expected_selected_env_ids": claim.selected_env_ids,
        }

    def _arm_global_reveal_terminal(
        self,
        prepared: PreparedPhysicalFlightInstall,
        global_boundary_receipt: (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        ),
        r05_terminal_claim: _r05.PreparedRevealTerminalClaim,
        *,
        expected_decision: str,
    ) -> ArmedPhysicalFlightInstall:
        """Perform every fallible child validation before R05's final arm."""

        self._require_operable()
        handle = self._owned_prepare(prepared)
        image = self._active_prepare_image
        boundary_owner = self._reveal_boundary_owner
        row = self._active_reveal_boundary_row
        r05_owner = self._r05_terminal_owner
        if (
            image is None
            or boundary_owner is None
            or row is None
            or r05_owner is None
        ):
            raise PhysicalFlightDeviceError(
                "physical global reveal arm lacks its exact prebound owners/row"
            )
        if self.device.type == "cpu":
            self._require_prearm_stale_witness(image)
        receipt = handle.prepare_receipt
        if expected_decision not in (
            _reveal_boundary.DECISION_ACCEPT,
            _reveal_boundary.DECISION_CENSOR,
        ):
            raise PhysicalFlightDeviceError("physical terminal decision differs")
        try:
            physical_owner_row = boundary_owner.require_owned_owner_row(
                global_boundary_receipt,
                owner_kind="physical_ball",
                expected_device_row=row,
                expected_prepared_token=handle,
                expected_fault_schema_sha256=(
                    _PHYSICAL_REVEAL_FAULT_SCHEMA.schema_sha256
                ),
                expected_reveal_final_preview_schema_version=(
                    receipt.reveal_final_preview.source_schema_version
                ),
                expected_reveal_final_preview_sha256=(
                    receipt.reveal_final_preview.source_canonical_sha256
                ),
                expected_selected_env_ids=receipt.selected_env_ids,
                expected_packet_sha256=global_boundary_receipt.packet_sha256,
                expected_decision=expected_decision,
            )
            physical_row_is_legal = (
                physical_owner_row.selected_pass
                == (True,) * len(receipt.selected_env_ids)
                and not any(physical_owner_row.selected_fault_bits)
            )
            if expected_decision == _reveal_boundary.DECISION_CENSOR:
                physical_row_is_legal = all(
                    passed ^ (fault_bits != 0)
                    for passed, fault_bits in zip(
                        physical_owner_row.selected_pass,
                        physical_owner_row.selected_fault_bits,
                    )
                )
            if (
                physical_owner_row.owner_mutation_version
                != self._mutation_version
                or physical_owner_row.owner_token_root_sha256
                != handle.canonical_sha256
                or not physical_row_is_legal
            ):
                raise PhysicalFlightDeviceError(
                    "physical boundary row verdict/version/root differs"
                )
            expectations = self._r05_terminal_expectations(
                r05_terminal_claim
            )
            if (
                r05_terminal_claim.decision != expected_decision
                or r05_terminal_claim.selected_env_ids
                != receipt.selected_env_ids
                or r05_terminal_claim.reveal_final_preview_schema_version
                != receipt.reveal_final_preview.source_schema_version
                or r05_terminal_claim.reveal_final_preview_sha256
                != receipt.reveal_final_preview.source_canonical_sha256
                or r05_terminal_claim.global_boundary_receipt_sha256
                != global_boundary_receipt.canonical_sha256
                or r05_terminal_claim.global_boundary_packet_sha256
                != global_boundary_receipt.packet_sha256
            ):
                raise PhysicalFlightDeviceError(
                    "R05 terminal claim physical preview/boundary differs"
                )
            owned_claim = r05_owner.require_owned_prepared_terminal_claim(
                r05_terminal_claim,
                **expectations,
            )
            if owned_claim is not r05_terminal_claim:
                raise PhysicalFlightDeviceError(
                    "R05 terminal claim identity differs"
                )
            boundary_mapping = global_boundary_receipt.to_mapping()
            boundary_mapping["canonical_sha256"] = (
                global_boundary_receipt.canonical_sha256
            )
            boundary_pin = _flight.CanonicalJsonContentPin.from_sealed_mapping(
                boundary_mapping,
                expected_source_kind=(
                    _flight.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND
                ),
                source_schema_sha256=(
                    _flight.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
                ),
            )
            claim_projection = self._r05_terminal_claim_projection(
                r05_terminal_claim
            )
            projection_pin, content_pin = self._r05_terminal_evidence_pins(
                r05_terminal_claim
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "physical global reveal terminal arm validation failed"
            ) from exc

        if self.device.type == "cpu":
            if (
                self._owner_checkpoint_sha()
                != receipt.physical_owner_checkpoint_before_sha256
            ):
                raise PhysicalFlightDeviceError("physical owner changed after prepare")
        elif self._mutation_version != receipt.mutation_version_before:
            raise PhysicalFlightDeviceError("physical owner changed after prepare")
        before_version = self._mutation_version
        next_version = before_version + 1
        next_nonce = self._next_prepare_nonce + 1
        lifecycle = self._lifecycle.clone()
        generation = self._generation.clone()
        outcome_sha = self._outcome_sha.clone()
        install_sha = self._install_sha.clone()
        installed_state_sha = self._installed_state_sha.clone()
        reveal_step = self._reveal_step.clone()
        observation_ordinal = self._observation_ordinal.clone()
        previous_ball_center = self._previous_ball_center.clone()
        parked = self._parked.clone()
        published = self._published.clone()
        slot_version = self._slot_version.clone()

        if expected_decision == _reveal_boundary.DECISION_ACCEPT:
            after_slots = image.after_slots
            scene_handle: object | None = image.scene_handle
            committed_rows = ()
            if self.device.type == "cpu":
                committed_rows = tuple(
                    _flight.CommittedPhysicalInstallRow(
                        env_id=prepared_row.env_id,
                        slot_index=prepared_row.slot_index,
                        install_payload_sha256=(
                            prepared_row.install_payload_sha256
                        ),
                        committed_slot_snapshot=after_slots[
                            self._slot_offset(
                                prepared_row.env_id,
                                prepared_row.slot_index,
                            )
                        ],
                        committed_slot_snapshot_sha256=after_slots[
                            self._slot_offset(
                                prepared_row.env_id,
                                prepared_row.slot_index,
                            )
                        ].canonical_sha256,
                    )
                    for prepared_row in receipt.rows
                )
            for payload in image.install_payloads:
                env_id = payload.env_id
                slot = payload.flight_slot
                lifecycle[env_id, slot] = R06_FLIGHT_INBOUND
                generation[env_id, slot] = payload.ball_generation
                outcome_sha[env_id, slot].copy_(
                    _digest_bytes(
                        payload.outcome_key_sha256,
                        device=self.device,
                    )
                )
                install_sha[env_id, slot].copy_(
                    _digest_bytes(payload.canonical_sha256, device=self.device)
                )
                installed_state_sha[env_id, slot].copy_(
                    _digest_bytes(
                        payload.installed_ball_state_sha256,
                        device=self.device,
                    )
                )
                reveal_step[env_id, slot] = payload.reveal_control_step
                observation_ordinal[env_id, slot] = -1
                previous_ball_center[env_id, slot].copy_(
                    image.after_scene_state[env_id, slot, :3]
                )
                parked[env_id, slot] = False
                published[env_id, slot] = True
                slot_version[env_id, slot] += 1
            portable_receipt: (
                _flight.PhysicalInstallCommitReceipt
                | _flight.PhysicalInstallCensorReceipt
                | PhysicalDeviceInstallTerminalReceipt
            )
            if self.device.type == "cpu":
                after_checkpoint = self._owner_checkpoint_sha_for(
                    slots=after_slots,
                    mutation_version=next_version,
                    next_prepare_nonce=next_nonce,
                    poisoned=False,
                )
                portable_receipt = _flight.PhysicalInstallCommitReceipt(
                    integration_status=_flight.INTEGRATION_STATUS,
                    prepare_receipt=receipt,
                    prepare_receipt_sha256=receipt.canonical_sha256,
                    global_reveal_boundary_receipt=boundary_pin,
                    global_reveal_boundary_receipt_sha256=(
                        global_boundary_receipt.canonical_sha256
                    ),
                    physical_boundary_fault_schema_sha256=(
                        _PHYSICAL_REVEAL_FAULT_SCHEMA.schema_sha256
                    ),
                    r05_terminal_claim=claim_projection,
                    r05_terminal_claim_sha256=(
                        r05_terminal_claim.canonical_sha256
                    ),
                    r05_terminal_boundary_projection=projection_pin,
                    r05_terminal_content_pin=content_pin,
                    r05_terminal_kind=r05_terminal_claim.terminal_kind,
                    r05_terminal_sha256=r05_terminal_claim.terminal_sha256,
                    physical_owner_checkpoint_before_sha256=(
                        receipt.physical_owner_checkpoint_before_sha256
                    ),
                    physical_owner_checkpoint_after_sha256=after_checkpoint,
                    mutation_version_before=before_version,
                    mutation_version_after=next_version,
                    rows=committed_rows,
                    committed_slots_root_sha256=_flight.physical_slot_root(
                        tuple(
                            committed.committed_slot_snapshot
                            for committed in committed_rows
                        )
                    ),
                    live_state_mutated=True,
                    runtime_publication_created=True,
                )
            else:
                portable_receipt = PhysicalDeviceInstallTerminalReceipt(
                    schema_version=1,
                    kind=PhysicalDeviceInstallTerminalReceipt.KIND,
                    prepare_receipt=receipt,
                    prepare_receipt_sha256=receipt.canonical_sha256,
                    decision=expected_decision,
                    global_boundary_receipt_sha256=(
                        global_boundary_receipt.canonical_sha256
                    ),
                    global_boundary_packet_sha256=(
                        global_boundary_receipt.packet_sha256
                    ),
                    physical_boundary_fault_schema_sha256=(
                        _PHYSICAL_REVEAL_FAULT_SCHEMA.schema_sha256
                    ),
                    r05_terminal_claim_sha256=(
                        r05_terminal_claim.canonical_sha256
                    ),
                    r05_terminal_kind=r05_terminal_claim.terminal_kind,
                    r05_terminal_sha256=r05_terminal_claim.terminal_sha256,
                    mutation_version_before=before_version,
                    mutation_version_after=next_version,
                    selected_env_ids=receipt.selected_env_ids,
                    selected_slot_indices=receipt.selected_slot_indices,
                    install_payload_sha256s=receipt.install_payload_sha256s,
                    device_state_resident=True,
                    scene_state_mutated=True,
                    runtime_publication_created=True,
                    policy_opportunity_created=True,
                )
        else:
            # The prepared scene write is disposed while failure is still
            # allowed.  The terminal commit itself is a zero-scene chronology
            # publication and cannot allocate or abort.
            self._abort_scene_write(image.scene_handle)
            after_slots = self._host_slots
            scene_handle = None
            if self.device.type == "cpu":
                before_slots_root = _flight.physical_slot_root(after_slots)
                after_checkpoint = self._owner_checkpoint_sha_for(
                    slots=after_slots,
                    mutation_version=next_version,
                    next_prepare_nonce=next_nonce,
                    poisoned=False,
                )
                portable_receipt = _flight.PhysicalInstallCensorReceipt(
                    integration_status=_flight.INTEGRATION_STATUS,
                    prepare_receipt=receipt,
                    prepare_receipt_sha256=receipt.canonical_sha256,
                    global_reveal_boundary_receipt=boundary_pin,
                    global_reveal_boundary_receipt_sha256=(
                        global_boundary_receipt.canonical_sha256
                    ),
                    physical_boundary_fault_schema_sha256=(
                        _PHYSICAL_REVEAL_FAULT_SCHEMA.schema_sha256
                    ),
                    r05_terminal_claim=claim_projection,
                    r05_terminal_claim_sha256=(
                        r05_terminal_claim.canonical_sha256
                    ),
                    r05_terminal_boundary_projection=projection_pin,
                    r05_terminal_content_pin=content_pin,
                    r05_terminal_kind=r05_terminal_claim.terminal_kind,
                    r05_terminal_sha256=r05_terminal_claim.terminal_sha256,
                    physical_owner_checkpoint_before_sha256=(
                        receipt.physical_owner_checkpoint_before_sha256
                    ),
                    physical_owner_checkpoint_after_sha256=after_checkpoint,
                    mutation_version_before=before_version,
                    mutation_version_after=next_version,
                    slots_root_before_sha256=before_slots_root,
                    slots_root_after_sha256=before_slots_root,
                    scene_state_mutated=False,
                    slot_state_mutated=False,
                    owner_chronology_mutated=True,
                    runtime_publication_created=False,
                    policy_opportunity_created=False,
                )
            else:
                portable_receipt = PhysicalDeviceInstallTerminalReceipt(
                    schema_version=1,
                    kind=PhysicalDeviceInstallTerminalReceipt.KIND,
                    prepare_receipt=receipt,
                    prepare_receipt_sha256=receipt.canonical_sha256,
                    decision=expected_decision,
                    global_boundary_receipt_sha256=(
                        global_boundary_receipt.canonical_sha256
                    ),
                    global_boundary_packet_sha256=(
                        global_boundary_receipt.packet_sha256
                    ),
                    physical_boundary_fault_schema_sha256=(
                        _PHYSICAL_REVEAL_FAULT_SCHEMA.schema_sha256
                    ),
                    r05_terminal_claim_sha256=(
                        r05_terminal_claim.canonical_sha256
                    ),
                    r05_terminal_kind=r05_terminal_claim.terminal_kind,
                    r05_terminal_sha256=r05_terminal_claim.terminal_sha256,
                    mutation_version_before=before_version,
                    mutation_version_after=next_version,
                    selected_env_ids=receipt.selected_env_ids,
                    selected_slot_indices=receipt.selected_slot_indices,
                    install_payload_sha256s=receipt.install_payload_sha256s,
                    device_state_resident=True,
                    scene_state_mutated=False,
                    runtime_publication_created=False,
                    policy_opportunity_created=False,
                )

        armed = ArmedPhysicalFlightInstall(
            _prepared_install=handle,
            _decision=expected_decision,
            _owner_identity=self._owner_identity,
            _token=_ARMED_TOKEN,
        )
        child_terminal = PhysicalChildTerminalToken(
            _armed_install=armed,
            _decision=expected_decision,
            _owner_identity=self._owner_identity,
            _token=_CHILD_TERMINAL_TOKEN,
        )
        epoch_image = _GlobalRevealEpochImage(
            decision=expected_decision,
            r05_terminal_claim=r05_terminal_claim,
            r05_terminal_owner=r05_owner,
            expected_terminal_kind=r05_terminal_claim.terminal_kind,
            expected_terminal_sha256=r05_terminal_claim.terminal_sha256,
            expected_claim_sha256=r05_terminal_claim.canonical_sha256,
            expected_preview_sha256=(
                r05_terminal_claim.reveal_final_preview_sha256
            ),
            expected_boundary_receipt_sha256=(
                r05_terminal_claim.global_boundary_receipt_sha256
            ),
            expected_packet_sha256=(
                r05_terminal_claim.global_boundary_packet_sha256
            ),
            expected_terminal_boundary_authority_sha256=(
                r05_terminal_claim.terminal_boundary_authority_sha256
            ),
            expected_terminal_boundary_projection_sha256=(
                r05_terminal_claim.terminal_boundary_projection_sha256
            ),
            expected_terminal_content_pin_sha256=(
                r05_terminal_claim.terminal_content_pin_sha256
            ),
            expected_selected_env_ids=r05_terminal_claim.selected_env_ids,
            portable_receipt=portable_receipt,
        )
        self._active_armed = armed
        self._active_armed_image = _ArmedPhysicalInstallImage(
            after_slots=after_slots,
            scene_handle=scene_handle,
            decision=expected_decision,
            global_boundary_receipt=global_boundary_receipt,
            r05_terminal_claim=r05_terminal_claim,
            r05_terminal_owner=r05_owner,
            portable_receipt=portable_receipt,
            lifecycle=lifecycle,
            generation=generation,
            outcome_sha=outcome_sha,
            install_sha=install_sha,
            installed_state_sha=installed_state_sha,
            reveal_step=reveal_step,
            observation_ordinal=observation_ordinal,
            previous_ball_center=previous_ball_center,
            parked=parked,
            published=published,
            slot_version=slot_version,
            mutation_version=next_version,
            next_prepare_nonce=next_nonce,
            child_terminal_token=child_terminal,
            global_epoch_image=epoch_image,
        )
        return armed

    def arm_global_reveal_boundary(
        self,
        prepared: PreparedPhysicalFlightInstall,
        global_boundary_receipt: (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        ),
        r05_terminal_claim: _r05.PreparedRevealTerminalClaim,
    ) -> ArmedPhysicalFlightInstall:
        return self._arm_global_reveal_terminal(
            prepared,
            global_boundary_receipt,
            r05_terminal_claim,
            expected_decision=_reveal_boundary.DECISION_ACCEPT,
        )

    def arm_censored_global_reveal_boundary(
        self,
        prepared: PreparedPhysicalFlightInstall,
        global_boundary_receipt: (
            _reveal_boundary.ActionBallFullMdpRevealBoundaryReceipt
        ),
        r05_terminal_claim: _r05.PreparedRevealTerminalClaim,
    ) -> ArmedPhysicalFlightInstall:
        return self._arm_global_reveal_terminal(
            prepared,
            global_boundary_receipt,
            r05_terminal_claim,
            expected_decision=_reveal_boundary.DECISION_CENSOR,
        )

    def arm_all_owner_reveal_prepare_marker(self, *args, **kwargs):
        del args, kwargs
        raise PhysicalFlightDeviceError(
            "legacy R05 marker arm is a production tombstone; consume the exact "
            "all-owner receipt and owner-issued terminal claim"
        )

    def _commit_prevalidated_child_terminal(
        self,
        armed_install: ArmedPhysicalFlightInstall,
        *,
        expected_decision: str,
    ) -> PhysicalChildTerminalToken:
        if self._poisoned:
            raise PhysicalFlightOwnerPoisonedError(
                self._poison_reason or "physical owner is poisoned"
            )
        armed = self._owned_armed(armed_install)
        image = self._active_armed_image
        if image is None or image.decision != expected_decision:
            raise PhysicalFlightDeviceError(
                "physical armed terminal decision/image differs"
            )
        try:
            if expected_decision == _reveal_boundary.DECISION_ACCEPT:
                self._apply_scene_write(image.scene_handle)
            elif image.scene_handle is not None:
                raise PhysicalFlightDeviceError(
                    "physical CENSOR terminal retained a scene write"
                )
            self._host_slots = image.after_slots
            self._lifecycle.copy_(image.lifecycle)
            self._generation.copy_(image.generation)
            self._outcome_sha.copy_(image.outcome_sha)
            self._install_sha.copy_(image.install_sha)
            self._installed_state_sha.copy_(image.installed_state_sha)
            self._reveal_step.copy_(image.reveal_step)
            self._observation_ordinal.copy_(image.observation_ordinal)
            self._previous_ball_center.copy_(image.previous_ball_center)
            self._parked.copy_(image.parked)
            self._published.copy_(image.published)
            self._slot_version.copy_(image.slot_version)
            self._set_owner_mutation_version(image.mutation_version)
            self._next_prepare_nonce = image.next_prepare_nonce
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = (
                "prevalidated physical child terminal publication failed; "
                "rollback is untrusted"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        child_terminal = image.child_terminal_token
        self._active_child_terminal = child_terminal
        self._active_global_reveal_epoch_image = image.global_epoch_image
        self._active_prepare = None
        self._active_prepare_image = None
        self._active_armed = None
        self._active_armed_image = None
        self._active_reveal_boundary_row = None
        return child_terminal

    def commit_prevalidated_reveal_install(
        self,
        armed_install: ArmedPhysicalFlightInstall,
    ) -> PhysicalChildTerminalToken:
        return self._commit_prevalidated_child_terminal(
            armed_install,
            expected_decision=_reveal_boundary.DECISION_ACCEPT,
        )

    def commit_censored_prevalidated(
        self,
        armed_install: ArmedPhysicalFlightInstall,
    ) -> PhysicalChildTerminalToken:
        return self._commit_prevalidated_child_terminal(
            armed_install,
            expected_decision=_reveal_boundary.DECISION_CENSOR,
        )

    def commit_prevalidated_install(self, *args, **kwargs):
        del args, kwargs
        raise PhysicalFlightDeviceError(
            "direct portable physical commit is a production tombstone; use the "
            "opaque child terminal then acknowledge R05-last completion"
        )

    def complete_global_reveal_epoch(
        self,
        child_terminal: PhysicalChildTerminalToken,
        r05_terminal_receipt: _r05.CommittedRevealBatch | _r05.CensoredRevealBatch,
    ) -> (
        _flight.PhysicalInstallCommitReceipt
        | _flight.PhysicalInstallCensorReceipt
        | PhysicalDeviceInstallTerminalReceipt
    ):
        """Expose portable evidence only after exact R05-last publication."""

        self._require_operable(allow_global_reveal_epoch=True)
        image = self._active_global_reveal_epoch_image
        if (
            image is None
            or type(child_terminal) is not PhysicalChildTerminalToken
            or child_terminal is not self._active_child_terminal
            or child_terminal._token is not _CHILD_TERMINAL_TOKEN
            or child_terminal._owner_identity is not self._owner_identity
            or child_terminal._decision != image.decision
        ):
            self.poison_global_reveal_epoch(
                "physical global reveal completion token is stale or foreign"
            )
            raise PhysicalFlightOwnerPoisonedError(
                self._poison_reason or "physical global reveal completion failed"
            )
        try:
            actual = image.r05_terminal_owner.require_owned_terminal_receipt(
                image.r05_terminal_claim,
                r05_terminal_receipt,
                expected_claim_sha256=image.expected_claim_sha256,
                expected_decision=image.decision,
                expected_reveal_final_preview_sha256=(
                    image.expected_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=(
                    image.expected_boundary_receipt_sha256
                ),
                expected_global_boundary_packet_sha256=(
                    image.expected_packet_sha256
                ),
                expected_terminal_boundary_authority_sha256=(
                    image.expected_terminal_boundary_authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    image.expected_terminal_boundary_projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    image.expected_terminal_content_pin_sha256
                ),
                expected_terminal_kind=image.expected_terminal_kind,
                expected_terminal_sha256=image.expected_terminal_sha256,
                expected_selected_env_ids=image.expected_selected_env_ids,
            )
            if actual is not r05_terminal_receipt:
                raise PhysicalFlightDeviceError(
                    "R05 terminal receipt identity differs"
                )
        except Exception as exc:
            self.poison_global_reveal_epoch(
                "physical global reveal completion R05 acknowledgement failed"
            )
            raise PhysicalFlightOwnerPoisonedError(
                self._poison_reason or "physical global reveal completion failed"
            ) from exc
        portable = image.portable_receipt
        self._terminal_resolution_total.add_(
            len(image.expected_selected_env_ids)
        )
        self._active_child_terminal = None
        self._active_global_reveal_epoch_image = None
        return portable

    @staticmethod
    def _exact_postphysics_stamp(value: object) -> tuple[int, int, int, int, int]:
        """Authenticate the exact env-owned post-scene-update stamp type."""

        expected_module_name = "whole_body_tracking.tasks.tracking.full_mdp_env"
        module = sys.modules.get(expected_module_name)
        expected_type = getattr(module, "FullMdpPhysicsSubstepStamp", None)
        event_phase_type = getattr(module, "FullMdpPhysicsEventPhase", None)
        if (
            expected_type is None
            or event_phase_type is None
            or getattr(expected_type, "__module__", None) != expected_module_name
            or getattr(expected_type, "__qualname__", None)
            != "FullMdpPhysicsSubstepStamp"
            or getattr(module, "FullMdpPhysicsSubstepStamp", None)
            is not expected_type
            or type(value) is not expected_type
            or type(value.event_phase) is not event_phase_type
            or value.event_phase is not event_phase_type.POST_SCENE_UPDATE
            or tuple(getattr(expected_type, "_fields", ()))
            != (
                "control_step",
                "physics_substep",
                "physics_substeps_per_control",
                "sim_step",
                "event_phase",
            )
        ):
            raise PhysicalFlightDeviceError(
                "postphysics stamp is not the exact env-owned post-scene-update ABI"
            )
        exact = value.exact_tuple()
        if (
            type(exact) is not tuple
            or len(exact) != 5
            or any(type(item) is not int for item in exact)
        ):
            raise PhysicalFlightDeviceError("postphysics exact stamp payload differs")
        control, substep, decimation, sim_step, phase = exact
        if (
            control < 1
            or decimation < 1
            or substep < 0
            or substep >= decimation
            or sim_step < 1
            or phase != int(event_phase_type.POST_SCENE_UPDATE)
        ):
            raise PhysicalFlightDeviceError("postphysics exact stamp range differs")
        return exact

    def _poison_postphysics_capture(
        self, reason: str, *, cause: BaseException | None = None
    ) -> None:
        self._poisoned = True
        self._poison_reason = reason
        self._active_postphysics_capture = None
        self._active_postphysics_capture_image = None
        self._active_postphysics = None
        self._active_postphysics_image = None
        error = PhysicalFlightOwnerPoisonedError(reason)
        if cause is None:
            raise error
        raise error from cause

    def _validate_postphysics_capture_boundary(
        self, stamp: object
    ) -> tuple[int, int, int, int, int]:
        """Validate one exact stamp before either empty or dense publication."""

        self._require_operable(allow_diagnostic_scene_observation=True)
        self._require_selected_contact_reward_cycle_closed(label="postphysics")
        if (
            self._active_postphysics_capture is not None
            or self._active_postphysics_capture_image is not None
            or self._active_postphysics is not None
            or self._active_r06_ack is not None
        ):
            self._poison_postphysics_capture(
                "postphysics capture was duplicated or crossed an unretired slot"
            )
        try:
            exact = self._exact_postphysics_stamp(stamp)
        except BaseException as exc:
            self._poison_postphysics_capture(
                "postphysics capture received a wrong stamp", cause=exc
            )
        previous = self._last_postphysics_exact_stamp
        if previous is None:
            if exact[1] != 0:
                self._poison_postphysics_capture(
                    "first postphysics capture did not start at substep zero"
                )
        else:
            p_control, p_substep, p_decimation, p_sim_step, _ = previous
            expected_control = p_control
            expected_substep = p_substep + 1
            if expected_substep == p_decimation:
                expected_control += 1
                expected_substep = 0
            if (
                exact[0] != expected_control
                or exact[1] != expected_substep
                or exact[2] != p_decimation
                or exact[3] != p_sim_step + 1
            ):
                self._poison_postphysics_capture(
                    "postphysics stamp was stale, skipped, duplicated or reordered"
                )
        return exact

    def capture_post_physics_facts(self, stamp: object) -> IsaacPostPhysicsFacts:
        """Capture once from the construction-owned producer after scene update.

        This is the only production entry.  It never manufactures an all-miss
        packet when the Isaac producer is absent: missing exact input is HOLD.
        The retained request binds every live slot to key, generation and the
        next observation ordinal before R06 sees any bytes.
        """

        exact = self._validate_postphysics_capture_boundary(stamp)

        scene_type = type(self.scene_port)
        retained_function = self._postphysics_scene_capture_function
        bound_capture = getattr(self.scene_port, "capture_post_physics_facts", None)
        if (
            scene_type is TensorPhysicalFlightScenePort
            or retained_function is None
            or not inspect.isfunction(retained_function)
            or inspect.getattr_static(
                scene_type, "capture_post_physics_facts", None
            )
            is not retained_function
            or not inspect.ismethod(bound_capture)
            or bound_capture.__self__ is not self.scene_port
            or bound_capture.__func__ is not retained_function
        ):
            self._poison_postphysics_capture(
                "exact concrete Isaac postphysics producer is absent; runtime HOLD"
            )

        shape = (self.num_envs, self.flight_capacity)
        observe = self._published & ~self._parked
        flight_slot = self._fixed_flight_slot_grid
        next_ordinal = torch.where(
            observe, self._observation_ordinal + 1, self._observation_ordinal
        )
        request = PhysicalPostPhysicsCaptureRequest(
            exact_stamp=exact,
            observe_mask=observe,
            flight_slot=flight_slot,
            full_key_sha256=self._outcome_sha,
            ball_generation=self._generation,
            observation_ordinal=next_ordinal,
            previous_ball_center_m=self._previous_ball_center,
            # The exact bound scene producer owns the single post-scene read
            # and returns it in ``IsaacPostPhysicsFacts``.  ``None`` is legal
            # only with this owner-issued identity/token pair; direct test or
            # foreign requests must still supply and match an explicit state.
            current_state_env_f32=None,
            _owner_identity=self._owner_identity,
            _token=_POSTPHYSICS_CAPTURE_REQUEST_TOKEN,
        )
        try:
            raw = bound_capture(request)
        except BaseException as exc:
            self._poison_postphysics_capture(
                "exact concrete Isaac postphysics producer failed", cause=exc
            )
        if (
            type(raw) is not IsaacPostPhysicsFacts
            or raw._owner_identity is not request
            or raw._capture_token is not request._token
        ):
            self._poison_postphysics_capture(
                "exact concrete Isaac postphysics producer returned an unjoined or partial ABI"
            )
        state = _tensor(
            raw.current_state_env_f32,
            label="captured postphysics scene state",
            shape=shape + (STATE_WIDTH,),
            dtype=torch.float32,
            device=self.device,
        )
        facts = replace(
            raw,
            _owner_identity=self._owner_identity,
            _capture_token=_POSTPHYSICS_CAPTURE_FACTS_TOKEN,
        )
        self._active_postphysics_capture = facts
        self._active_postphysics_capture_image = _PostPhysicsCaptureImage(
            exact_stamp=exact,
            observe_mask=observe.detach().clone(),
            flight_slot=flight_slot.detach().clone(),
            full_key_sha256=self._outcome_sha.detach().clone(),
            ball_generation=self._generation.detach().clone(),
            observation_ordinal=next_ordinal.detach().clone(),
            current_state_env_f32=state.detach().clone(),
            slot_version=self._slot_version.detach().clone(),
            owner_mutation_version=self._mutation_version,
        )
        self._last_postphysics_exact_stamp = exact
        return facts

    def publish_action_epoch_post_physics(
        self, stamp: object
    ) -> None:
        """Capture the real engine producer and publish Physical epoch facts.

        This diagnostic lane bypasses the old Physical reward-payment graph,
        but deliberately reuses the exact owned R06 packet and retire core so
        the same capture settles both owners before R06 publishes its epoch
        facts.  The real scene producer is still mandatory.  A real
        observation with no selected contact publishes an ordinary zero-reward
        fact; source absence, malformed/non-finite data, and engine overflow
        publish typed fault bits and zeroed payload.
        """

        if not self._action_epoch_runtime_call_active:
            self._action_epoch_runtime_call_active = True
            try:
                return self.publish_action_epoch_post_physics(stamp)
            except BaseException:
                epoch_owner = self._action_epoch_owner
                self._poisoned = True
                self._poison_reason = (
                    "Physical ActionEpoch postphysics transaction failed"
                )
                self._active_postphysics_capture = None
                self._active_postphysics_capture_image = None
                self._active_postphysics = None
                self._active_postphysics_image = None
                self._active_r06_ack = None
                self._active_r06_ack_image = None
                self._action_epoch_active_r06_postphysics = None
                r06_owner = self._r06_owner
                poison_r06 = getattr(r06_owner, "poison_global_reveal_epoch", None)
                if callable(poison_r06):
                    try:
                        poison_r06(
                            "Physical ActionEpoch postphysics transaction failed"
                        )
                    except BaseException:
                        pass
                if epoch_owner is not None:
                    epoch_owner.poison_owner_write(
                        "physical_ball",
                        PHYSICAL_EPOCH_FAULT_POSTPHYSICS_PRODUCER,
                        owner=self,
                    )
                raise
            finally:
                self._action_epoch_runtime_call_active = False
        pair = self._action_epoch_substep_pair
        if pair is None:
            raise PhysicalEpochIntegrationHold(
                "Physical postphysics has no exact pre-physics pair"
            )
        if pair == _ACTION_EPOCH_SUBSTEP_IDLE:
            exact = self._validate_postphysics_capture_boundary(stamp)
            if type(self.scene_port) is not TensorPhysicalFlightScenePort:
                self.scene_port.complete_action_epoch_idle_physics_fact_source(
                    exact
                )
            self._last_postphysics_exact_stamp = exact
            self._action_epoch_substep_pair = None
            return None
        if pair != _ACTION_EPOCH_SUBSTEP_DENSE:
            raise PhysicalEpochIntegrationHold(
                "Physical postphysics pre-pair kind differs"
            )
        epoch_owner = self._action_epoch_owner
        if epoch_owner is None:
            raise PhysicalEpochIntegrationHold(
                "Physical postphysics has no cold-bound ActionEpoch owner"
            )
        try:
            facts = self.capture_post_physics_facts(stamp)
        except BaseException:
            # ``capture_post_physics_facts`` itself sticky-poisons Physical for
            # absent/wrong producers.  Do not invent an all-miss epoch packet.
            raise
        image = self._active_postphysics_capture_image
        if (
            facts is not self._active_postphysics_capture
            or image is None
            or facts._owner_identity is not self._owner_identity
            or facts._capture_token is not _POSTPHYSICS_CAPTURE_FACTS_TOKEN
        ):
            self._poison_postphysics_capture(
                "Physical epoch did not consume the exact one-shot engine capture"
            )

        shape = (self.num_envs, self.flight_capacity)

        def exact(
            value: object,
            *,
            label: str,
            dtype: torch.dtype,
            width: int | None = None,
        ) -> torch.Tensor:
            expected = shape if width is None else shape + (width,)
            if (
                type(value) is not torch.Tensor
                or value.dtype != dtype
                or value.device != self.device
                or tuple(value.shape) != expected
            ):
                self._poison_postphysics_capture(
                    "Physical epoch engine fact ABI differs: " + label
                )
            return value

        observe = image.observe_mask
        contact = exact(
            facts.selected_contact_event,
            label="selected_contact_event",
            dtype=torch.bool,
        )
        current_center = image.current_state_env_f32[..., :3]
        contact_center = exact(
            facts.selected_contact_ball_center_m,
            label="selected_contact_ball_center_m",
            dtype=torch.float32,
            width=3,
        )
        outgoing_anchor = exact(
            facts.selected_contact_outgoing_segment_anchor_m,
            label="selected_contact_outgoing_segment_anchor_m",
            dtype=torch.float32,
            width=3,
        )
        producer_fault = exact(
            facts.producer_contract_fault,
            label="producer_contract_fault",
            dtype=torch.bool,
        )
        nonfinite = exact(
            facts.nonfinite_observation,
            label="nonfinite_observation",
            dtype=torch.bool,
        )
        overflow = exact(
            facts.engine_overflow,
            label="engine_overflow",
            dtype=torch.bool,
        )
        observation_stamp = _stamp_grid(
            facts.observation_stamp,
            label="Physical epoch observation_stamp",
            shape=shape,
            device=self.device,
        )
        control, substep, _, _, _ = image.exact_stamp
        stamp_fault = observe & (
            observation_stamp.control_step.ne(control)
            | observation_stamp.physics_substep.ne(substep)
            | observation_stamp.event_phase.ne(2)
        )
        data_nonfinite = observe & (
            ~torch.isfinite(current_center).all(dim=-1)
            | (contact & ~torch.isfinite(contact_center).all(dim=-1))
            | (contact & ~torch.isfinite(outgoing_anchor).all(dim=-1))
        )
        producer_bad = observe & (producer_fault | overflow | stamp_fault)
        nonfinite_bad = observe & (nonfinite | data_nonfinite)
        row_fault = (
            producer_bad.to(torch.int64) * PHYSICAL_EPOCH_FAULT_POSTPHYSICS_PRODUCER
            | nonfinite_bad.to(torch.int64) * PHYSICAL_EPOCH_FAULT_POSTPHYSICS_NONFINITE
        )

        present = observe & row_fault.eq(0)
        eligible = present & contact
        valid_bits = (
            present.to(torch.int64) * PHYSICAL_EPOCH_FACT_PRESENT
            | eligible.to(torch.int64) * PHYSICAL_EPOCH_FACT_SELECTED_CONTACT
        )
        source_step = torch.where(
            present,
            observation_stamp.control_step,
            torch.full_like(observation_stamp.control_step, -1),
        )
        values = torch.zeros(
            shape + (PHYSICAL_EPOCH_FACT_F32_WIDTH,),
            dtype=torch.float32,
            device=self.device,
        )
        values[..., PHYSICAL_EPOCH_FACT_CURRENT_CENTER] = torch.where(
            present[..., None], current_center, torch.zeros_like(current_center)
        )
        values[..., PHYSICAL_EPOCH_FACT_CONTACT_CENTER] = torch.where(
            eligible[..., None], contact_center, torch.zeros_like(contact_center)
        )
        values[..., PHYSICAL_EPOCH_FACT_OUTGOING_ANCHOR] = torch.where(
            eligible[..., None], outgoing_anchor, torch.zeros_like(outgoing_anchor)
        )
        values[..., PHYSICAL_EPOCH_FACT_OBSERVATION_ORDINAL] = torch.where(
            present,
            image.observation_ordinal.to(torch.float32),
            torch.zeros_like(image.observation_ordinal, dtype=torch.float32),
        )
        owner = self._r06_owner
        publish_direct = getattr(
            owner, "publish_action_ball_full_mdp_epoch_post_physics", None
        )
        retire_direct = getattr(
            owner, "retire_action_ball_full_mdp_epoch_post_physics", None
        )
        owner_type = type(owner)
        if (
            owner is None
            or not callable(publish_direct)
            or not callable(retire_direct)
            or getattr(publish_direct, "__self__", None) is not owner
            or getattr(retire_direct, "__self__", None) is not owner
            or getattr(publish_direct, "__func__", None)
            is not getattr(owner_type, "publish_action_ball_full_mdp_epoch_post_physics", None)
            or getattr(retire_direct, "__func__", None)
            is not getattr(owner_type, "retire_action_ball_full_mdp_epoch_post_physics", None)
        ):
            raise PhysicalEpochIntegrationHold(
                "R06 direct ActionEpoch postphysics/retire owner API is absent or patched"
            )
        fixed_flight_slot = self._fixed_flight_slot_grid
        self._action_epoch_active_r06_postphysics = (
            ActionEpochR06PostPhysicsProjection(
                observe_mask=observe.detach().clone(),
                flight_slot=fixed_flight_slot.detach().clone(),
                shot_key=self._action_epoch_flight_shot_key.clone(),
                publication_ordinal=(
                    self._action_epoch_flight_publication_ordinal.detach().clone()
                ),
                observation_ordinal=image.observation_ordinal.detach().clone(),
                previous_ball_center_m=(
                    self._previous_ball_center.detach().clone()
                ),
                current_ball_center_m=current_center.detach().clone(),
                observation_stamp=observation_stamp,
                selected_contact_event=contact.detach().clone(),
                selected_contact_ball_center_m=contact_center.detach().clone(),
                selected_contact_outgoing_segment_anchor_m=(
                    outgoing_anchor.detach().clone()
                ),
                selected_contact_stamp=_stamp_grid(
                    facts.selected_contact_stamp,
                    label="Physical epoch selected_contact_stamp",
                    shape=shape,
                    device=self.device,
                ),
                net_crossing_event=exact(
                    facts.net_crossing_event,
                    label="net_crossing_event",
                    dtype=torch.bool,
                ),
                net_clear_at_crossing=exact(
                    facts.net_clear_at_crossing,
                    label="net_clear_at_crossing",
                    dtype=torch.bool,
                ),
                net_crossing_stamp=_stamp_grid(
                    facts.net_crossing_stamp,
                    label="Physical epoch net_crossing_stamp",
                    shape=shape,
                    device=self.device,
                ),
                crossing_report_delivered=exact(
                    facts.crossing_report_delivered,
                    label="crossing_report_delivered",
                    dtype=torch.bool,
                ),
                first_descending_crossing_event=exact(
                    facts.first_descending_crossing_event,
                    label="first_descending_crossing_event",
                    dtype=torch.bool,
                ),
                first_descending_crossing_xy_m=exact(
                    facts.first_descending_crossing_xy_m,
                    label="first_descending_crossing_xy_m",
                    dtype=torch.float32,
                    width=2,
                ),
                first_descending_crossing_stamp=_stamp_grid(
                    facts.first_descending_crossing_stamp,
                    label="Physical epoch first_descending_crossing_stamp",
                    shape=shape,
                    device=self.device,
                ),
                nonfinite_observation=nonfinite.detach().clone(),
                producer_contract_fault=producer_fault.detach().clone(),
                engine_overflow=overflow.detach().clone(),
                owner_fault_bits=row_fault.detach().clone(),
                fact_valid_bits=valid_bits.detach().clone(),
                fact_source_step=source_step.detach().clone(),
                fact_f32=values.detach().clone(),
                physical_owner=self,
                epoch_owner=epoch_owner,
                _owner_identity=self._owner_identity,
                _token=_ACTION_EPOCH_R06_POSTPHYSICS_TOKEN,
            )
        )
        refresh_epoch = getattr(
            epoch_owner, "refresh_physical_postphysics_rows", None
        )
        direct_refresh_epoch = getattr(
            type(epoch_owner), "refresh_physical_postphysics_rows", None
        )
        if (
            not callable(refresh_epoch)
            or not callable(direct_refresh_epoch)
            or getattr(refresh_epoch, "__self__", None) is not epoch_owner
            or getattr(refresh_epoch, "__func__", None) is not direct_refresh_epoch
        ):
            raise PhysicalEpochIntegrationHold(
                "Epoch exact Physical postphysics row consumer is absent or patched"
            )
        refresh_epoch()
        post_result = _require_final_r06_type(
            publish_direct(), expected_name="ActionEpochR06PostPhysicsResult"
        )
        retire_result = _require_final_r06_type(
            retire_direct(), expected_name="ActionEpochR06RetireResult"
        )
        self.require_owned_action_epoch_r06_postphysics_projection()
        retired = _tensor(
            getattr(retire_result, "retired_mask", None),
            label="R06 direct retired_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        settled = _tensor(
            getattr(post_result, "settled_mask", None),
            label="R06 direct settled_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        accepted = _tensor(
            getattr(post_result, "accepted", None),
            label="R06 direct accepted",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        rejected = _tensor(
            getattr(post_result, "rejected", None),
            label="R06 direct rejected",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        slot_match = torch.ones(shape, dtype=torch.bool, device=self.device)
        for result in (post_result, retire_result):
            slot_match &= _tensor(
                getattr(result, "flight_slot", None),
                label="R06 direct flight_slot",
                shape=shape,
                dtype=torch.int64,
                device=self.device,
            ).eq(fixed_flight_slot)
        torch._assert_async(
            torch.all(
                slot_match & ~(accepted & rejected)
                & ~(retired & ~settled)
            ),
            "R06 direct postphysics/retire result is internally inconsistent",
        )
        scene_handle = self._prepare_action_epoch_scene_write(
            kind="retire",
            state_env_f32=self._park_state_template,
            selected_mask=retired,
        )
        self._apply_scene_write(scene_handle)
        self._lifecycle = torch.where(
            retired, torch.full_like(self._lifecycle, R06_FLIGHT_EMPTY), self._lifecycle
        )
        self._parked |= retired
        self._published &= ~retired
        self._slot_version += retired.to(torch.int64)
        self._action_epoch_active_r06_postphysics = None
        r06_publish = getattr(
            self._r06_owner, "publish_action_ball_full_mdp_epoch_facts", None
        )
        direct_r06_publish = getattr(
            type(self._r06_owner),
            "publish_action_ball_full_mdp_epoch_facts",
            None,
        )
        if (
            not callable(r06_publish)
            or not callable(direct_r06_publish)
            or getattr(r06_publish, "__self__", None) is not self._r06_owner
            or getattr(r06_publish, "__func__", None) is not direct_r06_publish
        ):
            raise PhysicalEpochIntegrationHold(
                "R06 ActionEpoch fact publisher is absent or patched"
            )
        r06_publish()
        retired = ~self._published & self._parked
        active_slots = self._action_epoch_active_flight_slot
        safe_active_slots = active_slots.clamp(0, self.flight_capacity - 1)
        active_slot_valid = active_slots.ge(0) & active_slots.lt(
            self.flight_capacity
        )
        active_slot_mask = active_slot_valid[:, None] & fixed_flight_slot.eq(
            safe_active_slots[:, None]
        )
        clear_slot_mask = retired & active_slot_mask
        env_retired = clear_slot_mask.any(dim=1)
        for field in fields(_row_identity.ActionEpochShotKey):
            identity_grid = getattr(self._action_epoch_flight_shot_key, field.name)
            identity_grid.copy_(
                torch.where(
                    clear_slot_mask,
                    torch.full_like(identity_grid, -1),
                    identity_grid,
                )
            )
        self._action_epoch_flight_publication_ordinal.copy_(
            torch.where(
                clear_slot_mask,
                torch.full_like(
                    self._action_epoch_flight_publication_ordinal, -1
                ),
                self._action_epoch_flight_publication_ordinal,
            )
        )
        self._action_epoch_active_flight_slot = torch.where(
            env_retired,
            torch.full_like(active_slots, -1),
            active_slots,
        )
        self._active_postphysics_capture = None
        self._active_postphysics_capture_image = None
        self._action_epoch_substep_pair = None
        return None

    def action_epoch_reward_facts_v1(
        self, epoch_record: object
    ) -> PhysicalEpochSelectedContactFacts:
        """Decode only this Physical owner's immutable current epoch slice.

        The record version is part of the read boundary; scalar journal epoch
        is never used as shot identity.  Validity bits, not numeric payloads,
        decide whether a real observation and selected contact exist.
        """

        epoch_owner = self._action_epoch_owner
        if epoch_owner is None:
            raise PhysicalEpochIntegrationHold(
                "Physical reward facts have no cold-bound ActionEpoch owner"
            )
        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_full_mdp_epoch as epoch
        current = epoch_owner.current()
        if (
            type(epoch_record) is not epoch.ActionEpochRecord
            or type(current) is not epoch.ActionEpochRecord
            or epoch_record.version != current.version
        ):
            raise PhysicalEpochIntegrationHold(
                "Physical reward facts require the exact current epoch version"
            )
        owner_slot = epoch.OWNER_ORDER.index("physical_ball")
        env_ids = torch.arange(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        shot_slots = current.current_task_slot
        shot_slot_valid = shot_slots.ge(0) & shot_slots.lt(
            epoch_owner.shot_slot_capacity
        )
        safe_shot_slots = shot_slots.clamp(
            min=0, max=epoch_owner.shot_slot_capacity - 1
        )
        bits = current.fact_valid_bits[
            env_ids, safe_shot_slots, owner_slot
        ]
        faults = current.owner_fault_bits[
            env_ids, safe_shot_slots, owner_slot
        ]
        present = (
            shot_slot_valid
            & faults.eq(0)
            & torch.bitwise_and(bits, PHYSICAL_EPOCH_FACT_PRESENT).ne(0)
        )
        eligible = (
            present
            & torch.bitwise_and(bits, PHYSICAL_EPOCH_FACT_SELECTED_CONTACT).ne(0)
        )
        source = current.fact_source_step[
            env_ids, safe_shot_slots, owner_slot
        ]
        values = current.fact_f32[
            env_ids, safe_shot_slots, owner_slot
        ]
        return PhysicalEpochSelectedContactFacts(
            present=present.detach().clone(),
            eligible=eligible.detach().clone(),
            source_step=torch.where(
                present, source, torch.full_like(source, -1)
            ).detach().clone(),
            producer_fault_bits=faults.detach().clone(),
            current_ball_center_m=torch.where(
                present[:, None],
                values[:, PHYSICAL_EPOCH_FACT_CURRENT_CENTER],
                torch.zeros_like(values[:, PHYSICAL_EPOCH_FACT_CURRENT_CENTER]),
            ).detach().clone(),
            selected_contact_ball_center_m=torch.where(
                eligible[:, None],
                values[:, PHYSICAL_EPOCH_FACT_CONTACT_CENTER],
                torch.zeros_like(values[:, PHYSICAL_EPOCH_FACT_CONTACT_CENTER]),
            ).detach().clone(),
            selected_contact_outgoing_segment_anchor_m=torch.where(
                eligible[:, None],
                values[:, PHYSICAL_EPOCH_FACT_OUTGOING_ANCHOR],
                torch.zeros_like(values[:, PHYSICAL_EPOCH_FACT_OUTGOING_ANCHOR]),
            ).detach().clone(),
        )

    def action_epoch_selected_contact_reward_facts(
        self,
    ) -> PhysicalEpochSelectedContactFacts:
        """Compatibility spelling for the immutable current epoch decoder."""

        if self._action_epoch_owner is None:
            raise PhysicalEpochIntegrationHold(
                "Physical selected-contact reward has no ActionEpoch owner"
            )
        return self.action_epoch_reward_facts_v1(
            self._action_epoch_owner.current()
        )

    def build_post_physics_publication(
        self,
        *,
        facts: IsaacPostPhysicsFacts,
    ) -> PhysicalPostPhysicsPublication:
        """Consume a sealed production capture or a pure Tensor diagnostic."""

        if type(self.scene_port) is TensorPhysicalFlightScenePort:
            return self._build_post_physics_publication_diagnostic(facts=facts)
        try:
            return self._consume_post_physics_capture(facts=facts)
        except PhysicalFlightOwnerPoisonedError:
            raise
        except BaseException as exc:
            self._poison_postphysics_capture(
                "postphysics captured packet was partial or malformed", cause=exc
            )

    def _consume_post_physics_capture(
        self,
        *,
        facts: IsaacPostPhysicsFacts,
    ) -> PhysicalPostPhysicsPublication:
        """Consume one sealed production capture; retain direct tensor diagnostics."""

        image = self._active_postphysics_capture_image
        if (
            type(facts) is not IsaacPostPhysicsFacts
            or facts is not self._active_postphysics_capture
            or facts._owner_identity is not self._owner_identity
            or facts._capture_token is not _POSTPHYSICS_CAPTURE_FACTS_TOKEN
            or image is None
        ):
            self._poison_postphysics_capture(
                "postphysics build did not consume the exact one-shot capture"
            )
        shape = (self.num_envs, self.flight_capacity)
        current_state = _tensor(
            self.scene_port.read_state_env(),
            label="postphysics captured-state join",
            shape=shape + (STATE_WIDTH,),
            dtype=torch.float32,
            device=self.device,
        )
        stale = (
            (self._slot_version != image.slot_version)
            | (self._published & ~self._parked) != image.observe_mask
            | ~torch.eq(self._outcome_sha, image.full_key_sha256).all(dim=-1)
            | (self._generation != image.ball_generation)
            | (
                torch.where(
                    image.observe_mask,
                    self._observation_ordinal + 1,
                    self._observation_ordinal,
                )
                != image.observation_ordinal
            )
            | ~torch.eq(current_state, image.current_state_env_f32).all(dim=-1)
        )
        if self._mutation_version != image.owner_mutation_version:
            stale = torch.ones_like(stale)
        control, substep, _, _, _ = image.exact_stamp

        def stamp_mismatch(value: object, active: torch.Tensor) -> torch.Tensor:
            grid = _stamp_grid(
                value,
                label="captured event stamp",
                shape=shape,
                device=self.device,
            )
            return active & (
                (grid.control_step != control)
                | (grid.physics_substep != substep)
            )

        observe = image.observe_mask
        contact = _tensor(
            facts.selected_contact_event,
            label="captured selected_contact_event",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        net = _tensor(
            facts.net_crossing_event,
            label="captured net_crossing_event",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        crossing = _tensor(
            facts.first_descending_crossing_event,
            label="captured first_descending_crossing_event",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        stale = (
            stale
            | stamp_mismatch(facts.observation_stamp, observe)
            | stamp_mismatch(facts.selected_contact_stamp, contact)
            | stamp_mismatch(facts.net_crossing_stamp, net)
            | stamp_mismatch(facts.first_descending_crossing_stamp, crossing)
        )
        if self.device.type == "cpu" and bool(stale.any()):
            self._poison_postphysics_capture(
                "postphysics capture became stale or joined the wrong slot"
            )
        if self.device.type != "cpu":
            # Sticky integrity evidence remains device-only until the one PPO
            # drain; there is no hot-path D2H synchronization before R06.
            self._device_fault.logical_or_(stale)
        producer_fault = _tensor(
            facts.producer_contract_fault,
            label="captured producer_contract_fault",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        ) | stale
        sealed = replace(facts, producer_contract_fault=producer_fault)
        try:
            publication = self._build_post_physics_publication_diagnostic(
                facts=sealed,
                _capture_authority=_POSTPHYSICS_CAPTURE_FACTS_TOKEN,
            )
        except BaseException as exc:
            self._poison_postphysics_capture(
                "postphysics captured packet was only partially built", cause=exc
            )
        if self.device.type == "cpu" and bool(publication.owner_join_fault.any()):
            self._poison_postphysics_capture(
                "postphysics capture joined an orphan physical or R06 slot"
            )
        if self.device.type != "cpu":
            self._device_fault.logical_or_(publication.owner_join_fault)
        return publication

    def _build_post_physics_publication_diagnostic(
        self,
        *,
        facts: IsaacPostPhysicsFacts,
        _capture_authority: object | None = None,
    ) -> PhysicalPostPhysicsPublication:
        """Freeze one actual-R06 packet against one exact coordinator object."""

        if (
            type(self.scene_port) is not TensorPhysicalFlightScenePort
            and _capture_authority is not _POSTPHYSICS_CAPTURE_FACTS_TOKEN
        ):
            self._poison_postphysics_capture(
                "production postphysics build bypassed owner capture authority"
            )
        self._require_operable()
        self._require_selected_contact_reward_cycle_closed(label="postphysics")
        if self._active_postphysics is not None:
            raise PhysicalFlightDeviceError(
                "one postphysics publication is already awaiting R06 acknowledgement"
            )
        if self._active_r06_ack is not None:
            raise PhysicalFlightDeviceError(
                "settled R06 acknowledgement must be consumed before another observation"
            )
        if self._active_prepare is not None:
            raise PhysicalFlightDeviceError(
                "postphysics cannot cross an active physical prepare lease"
            )
        shape = (self.num_envs, self.flight_capacity)
        owner = self._r06_owner
        if owner is None or self._r06_park_token_authority is None:
            raise PhysicalFlightDeviceError(
                "postphysics requires the permanently paired R06 owner"
            )
        owner = _require_final_r06_type(
            owner, expected_name="ActionBallLandingOutcomeDeviceCoordinator"
        )
        if (
            getattr(owner, "num_envs", None) != self.num_envs
            or getattr(owner, "flight_slot_capacity", None)
            != self.flight_capacity
            or getattr(owner, "device", None) != self.device
            or getattr(owner, "dtype", None) != torch.float32
        ):
            raise PhysicalFlightDeviceError(
                "R06 coordinator physical grid/device ABI differs"
            )
        try:
            r06_before = owner.current_flight_lifecycle_snapshot()
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "R06 before lifecycle snapshot could not be acquired"
            ) from exc
        snapshot = _require_final_r06_type(
            r06_before,
            expected_name="FlightLifecycleSnapshotBatch",
        )
        r06_version_tensor = _r06_device_mutation_version(
            snapshot.mutation_version,
            label="R06 owner_mutation_version before",
            device=self.device,
        )
        r06_version_before: int | torch.Tensor = (
            _r06_mutation_version(
                r06_version_tensor,
                label="R06 owner_mutation_version before",
                device=self.device,
            )
            if self.device.type == "cpu"
            else r06_version_tensor.detach().clone()
        )
        state = _tensor(
            self.scene_port.read_state_env(),
            label="postphysics scene state",
            shape=shape + (STATE_WIDTH,),
            dtype=torch.float32,
            device=self.device,
        )
        scene_nonfinite = ~torch.isfinite(state).all(dim=-1)
        if self.device.type == "cpu" and bool(scene_nonfinite.any()):
            raise PhysicalFlightDeviceError(
                "postphysics scene state is nonfinite before R06 publication"
            )
        r06_state = _tensor(
            snapshot.state,
            label="R06 flight_state",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        r06_key = _tensor(
            snapshot.full_key_sha256,
            label="R06 full_key_sha256",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        r06_generation = _tensor(
            snapshot.ball_generation,
            label="R06 ball_generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        r06_mailbox_slot = _tensor(
            snapshot.mailbox_slot,
            label="R06 mailbox_slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        r06_ordinal = self._observation_ordinal.clone()
        live = (r06_state == R06_FLIGHT_INBOUND) | (r06_state == R06_FLIGHT_OPEN)
        owner_live = self._published & ~self._parked
        key_match = torch.eq(r06_key, self._outcome_sha).all(dim=-1)
        generation_match = r06_generation == self._generation
        physical_orphan = owner_live & ~live
        r06_orphan = live & ~owner_live
        join_fault = physical_orphan | r06_orphan
        join_fault = join_fault | (live & ~(key_match & generation_match))

        def bool_grid(name: str) -> torch.Tensor:
            return _tensor(
                getattr(facts, name),
                label=name,
                shape=shape,
                dtype=torch.bool,
                device=self.device,
            )

        def float_grid(name: str, width: int) -> torch.Tensor:
            return _tensor(
                getattr(facts, name),
                label=name,
                shape=shape + (width,),
                dtype=torch.float32,
                device=self.device,
            )

        contact_event = bool_grid("selected_contact_event")
        net_event = bool_grid("net_crossing_event")
        crossing_event = bool_grid("first_descending_crossing_event")

        def checked_stamp(
            value: object,
            *,
            label: str,
            active: torch.Tensor,
            expected_phase: int,
        ) -> tuple[PhysicsStampGrid, torch.Tensor]:
            stamp = _stamp_grid(
                value,
                label=label,
                shape=shape,
                device=self.device,
            )
            expected_phase_grid = torch.where(
                active,
                torch.full(
                    shape,
                    expected_phase,
                    dtype=torch.int8,
                    device=self.device,
                ),
                torch.full(
                    shape,
                    -1,
                    dtype=torch.int8,
                    device=self.device,
                ),
            )
            stamp_fault = (
                (stamp.event_phase != expected_phase_grid)
                | ((stamp.control_step < 0) != ~active)
                | ((stamp.physics_substep < 0) != ~active)
            )
            if self.device.type == "cpu" and bool(stamp_fault.any()):
                raise PhysicalFlightDeviceError(
                    f"{label} active mask/phase encoding differs from final R06"
                )
            return stamp, stamp_fault

        observation_stamp, observation_stamp_fault = checked_stamp(
            facts.observation_stamp,
            label="observation_stamp",
            active=live,
            expected_phase=2,
        )
        contact_stamp, contact_stamp_fault = checked_stamp(
            facts.selected_contact_stamp,
            label="selected_contact_stamp",
            active=contact_event,
            expected_phase=0,
        )
        net_stamp, net_stamp_fault = checked_stamp(
            facts.net_crossing_stamp,
            label="net_crossing_stamp",
            active=net_event,
            expected_phase=1,
        )
        crossing_stamp, crossing_stamp_fault = checked_stamp(
            facts.first_descending_crossing_stamp,
            label="first_descending_crossing_stamp",
            active=crossing_event,
            expected_phase=2,
        )
        producer_fault = (
            bool_grid("producer_contract_fault")
            | join_fault
            | observation_stamp_fault
            | contact_stamp_fault
            | net_stamp_fault
            | crossing_stamp_fault
        )
        next_ordinal = torch.where(live, r06_ordinal + 1, r06_ordinal)
        publication = PhysicalPostPhysicsPublication(
            observe_mask=live.clone(),
            full_key_sha256=r06_key.clone(),
            ball_generation=r06_generation.clone(),
            observation_ordinal=next_ordinal.clone(),
            previous_ball_center_m=self._previous_ball_center.clone(),
            current_ball_center_m=state[..., :3].clone(),
            observation_stamp=observation_stamp,
            selected_contact_event=contact_event.clone(),
            selected_contact_ball_center_m=float_grid(
                "selected_contact_ball_center_m", 3
            ).clone(),
            selected_contact_outgoing_segment_anchor_m=float_grid(
                "selected_contact_outgoing_segment_anchor_m", 3
            ).clone(),
            selected_contact_stamp=contact_stamp,
            net_crossing_event=net_event.clone(),
            net_clear_at_crossing=bool_grid("net_clear_at_crossing").clone(),
            net_crossing_stamp=net_stamp,
            crossing_report_delivered=bool_grid(
                "crossing_report_delivered"
            ).clone(),
            first_descending_crossing_event=crossing_event.clone(),
            first_descending_crossing_xy_m=float_grid(
                "first_descending_crossing_xy_m", 2
            ).clone(),
            first_descending_crossing_stamp=crossing_stamp,
            nonfinite_observation=(
                bool_grid("nonfinite_observation") | scene_nonfinite
            ).clone(),
            producer_contract_fault=producer_fault.clone(),
            engine_overflow=bool_grid("engine_overflow").clone(),
            owner_join_fault=join_fault.clone(),
            _owner_identity=self._owner_identity,
            _token=_POSTPHYSICS_TOKEN,
        )
        module = sys.modules.get(type(owner).__module__)
        stamp_type = getattr(module, "PhysicsStampBatch", None)
        batch_type = getattr(module, "PostPhysicsFlightBatch", None)
        if stamp_type is None or batch_type is None:
            raise PhysicalFlightDeviceError("final R06 batch constructors are unavailable")

        def actual_stamp(value: PhysicsStampGrid) -> object:
            return stamp_type(
                control_step=value.control_step.clone(),
                physics_substep=value.physics_substep.clone(),
                event_phase=value.event_phase.clone(),
            )

        r06_batch = batch_type(
            observe_mask=publication.observe_mask.clone(),
            full_key_sha256=publication.full_key_sha256.clone(),
            ball_generation=publication.ball_generation.clone(),
            observation_ordinal=publication.observation_ordinal.clone(),
            previous_ball_center_m=publication.previous_ball_center_m.clone(),
            current_ball_center_m=publication.current_ball_center_m.clone(),
            observation_stamp=actual_stamp(publication.observation_stamp),
            selected_contact_event=publication.selected_contact_event.clone(),
            selected_contact_ball_center_m=(
                publication.selected_contact_ball_center_m.clone()
            ),
            selected_contact_outgoing_segment_anchor_m=(
                publication.selected_contact_outgoing_segment_anchor_m.clone()
            ),
            selected_contact_stamp=actual_stamp(publication.selected_contact_stamp),
            net_crossing_event=publication.net_crossing_event.clone(),
            net_clear_at_crossing=publication.net_clear_at_crossing.clone(),
            net_crossing_stamp=actual_stamp(publication.net_crossing_stamp),
            crossing_report_delivered=(
                publication.crossing_report_delivered.clone()
            ),
            first_descending_crossing_event=(
                publication.first_descending_crossing_event.clone()
            ),
            first_descending_crossing_xy_m=(
                publication.first_descending_crossing_xy_m.clone()
            ),
            first_descending_crossing_stamp=actual_stamp(
                publication.first_descending_crossing_stamp
            ),
            nonfinite_observation=publication.nonfinite_observation.clone(),
            producer_contract_fault=publication.producer_contract_fault.clone(),
            engine_overflow=publication.engine_overflow.clone(),
            physical_publication_identity=publication,
        )
        self._active_postphysics = publication
        self._active_postphysics_image = _PostPhysicsImage(
            observe_mask=live.clone(),
            full_key_sha256=r06_key.clone(),
            ball_generation=r06_generation.clone(),
            observation_ordinal=next_ordinal.clone(),
            current_state_f32=state.clone(),
            current_ball_center_m=state[..., :3].clone(),
            observation_control_step=observation_stamp.control_step.clone(),
            observation_physics_substep=(
                observation_stamp.physics_substep.clone()
            ),
            owner_join_fault=join_fault.clone(),
            physical_orphan_mask=physical_orphan.clone(),
            r06_orphan_mask=r06_orphan.clone(),
            r06_owner_mutation_version_before=r06_version_before,
            r06_owner=owner,
            r06_batch=r06_batch,
            r06_flight_state_before=r06_state.clone(),
            r06_mailbox_slot_before=r06_mailbox_slot.clone(),
        )
        return publication

    def publish_post_physics_to_r06(
        self,
        publication: PhysicalPostPhysicsPublication,
    ) -> object:
        """Publish once; any partial/mismatched R06 mutation is fail-stop."""

        try:
            return self._publish_post_physics_to_r06_owned(publication)
        except PhysicalFlightOwnerPoisonedError:
            raise
        except BaseException as exc:
            self._active_postphysics = None
            self._active_postphysics_image = None
            self._poisoned = True
            self._poison_reason = "actual R06 postphysics publication was partial"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc

    def _publish_post_physics_to_r06_owned(
        self,
        publication: PhysicalPostPhysicsPublication,
    ) -> object:
        """Publish one exact R06 mutation and retain its opaque result lease."""

        self._require_operable()
        self._require_selected_contact_reward_cycle_closed(
            label="postphysics publication"
        )
        shape = (self.num_envs, self.flight_capacity)
        if (
            type(publication) is not PhysicalPostPhysicsPublication
            or publication._token is not _POSTPHYSICS_TOKEN
            or publication._owner_identity is not self._owner_identity
            or self._active_postphysics is not publication
            or self._active_postphysics_image is None
        ):
            raise PhysicalFlightDeviceError("postphysics publication is stale or foreign")
        image = self._active_postphysics_image
        owner = image.r06_owner
        try:
            result = owner.publish_post_physics(image.r06_batch)
            result = _require_final_r06_type(
                result,
                expected_name="PostPhysicsMutationResult",
            )
            snapshot = owner.current_flight_lifecycle_snapshot()
            mailbox_state = owner.mailbox_state
        except Exception as exc:
            self._active_postphysics = None
            self._active_postphysics_image = None
            self._poisoned = True
            self._poison_reason = "actual R06 postphysics publication failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        snapshot = _require_final_r06_type(
            snapshot,
            expected_name="FlightLifecycleSnapshotBatch",
        )
        accepted = _tensor(
            result.accepted,
            label="R06 postphysics accepted",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        rejected = _tensor(
            result.rejected,
            label="R06 postphysics rejected",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        fault_bits = _tensor(
            result.fault_bits,
            label="R06 postphysics fault_bits",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        settled_result = _tensor(
            result.settled_mask,
            label="R06 postphysics settled_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        settlement_cause = _tensor(
            result.settlement_cause,
            label="R06 postphysics settlement_cause",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        after_state = _tensor(
            snapshot.state,
            label="R06 after flight_state",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        after_key = _tensor(
            snapshot.full_key_sha256,
            label="R06 after full_key_sha256",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        after_generation = _tensor(
            snapshot.ball_generation,
            label="R06 after ball_generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        mailbox_slot = _tensor(
            snapshot.mailbox_slot,
            label="R06 after mailbox_slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        after_ordinal = image.observation_ordinal
        after_version_tensor = _r06_device_mutation_version(
            snapshot.mutation_version,
            label="R06 owner_mutation_version after",
            device=self.device,
        )
        result_version_tensor = _r06_device_mutation_version(
            result.mutation_version,
            label="R06 postphysics result mutation_version",
            device=self.device,
        )
        if self.device.type == "cpu":
            after_version: int | torch.Tensor = _r06_mutation_version(
                after_version_tensor,
                label="R06 owner_mutation_version after",
                device=self.device,
            )
            result_version: int | torch.Tensor = _r06_mutation_version(
                result_version_tensor,
                label="R06 postphysics result mutation_version",
                device=self.device,
            )
            version_mismatch: bool | torch.Tensor = (
                after_version
                != image.r06_owner_mutation_version_before + 1
                or result_version != after_version
            )
        else:
            before_version_tensor = image.r06_owner_mutation_version_before
            if not isinstance(before_version_tensor, torch.Tensor):
                raise PhysicalFlightDeviceError(
                    "R06 device pre-publication version identity differs"
                )
            after_version = after_version_tensor.detach().clone()
            result_version = result_version_tensor.detach().clone()
            version_mismatch = (
                (after_version_tensor != before_version_tensor + 1)
                | (result_version_tensor != after_version_tensor)
            )
        result_key = _tensor(
            result.full_key_sha256,
            label="R06 postphysics result full_key_sha256",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        result_generation = _tensor(
            result.ball_generation,
            label="R06 postphysics result ball_generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        result_slot = _tensor(
            result.flight_slot,
            label="R06 postphysics result flight_slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        expected_slot = torch.arange(
            self.flight_capacity,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(0).expand(shape)
        before_state = image.r06_flight_state_before
        legal_transition = (
            ((before_state == R06_FLIGHT_INBOUND) & (
                (after_state == R06_FLIGHT_INBOUND)
                | (after_state == R06_FLIGHT_OPEN)
                | (after_state == R06_FLIGHT_SETTLED_RETAINED)
            ))
            | ((before_state == R06_FLIGHT_OPEN) & (
                (after_state == R06_FLIGHT_OPEN)
                | (after_state == R06_FLIGHT_SETTLED_RETAINED)
            ))
        )
        after_join_fault = image.observe_mask & (
            ~torch.eq(after_key, image.full_key_sha256).all(dim=-1)
            | (after_generation != image.ball_generation)
            | ~legal_transition
        )
        after_join_fault = after_join_fault | (
            ~torch.eq(result_key, after_key).all(dim=-1)
            | (result_generation != after_generation)
            | (result_slot != expected_slot)
        )
        unchanged = ~image.observe_mask
        after_join_fault = after_join_fault | (
            unchanged
            & (
                (after_state != before_state)
                | ~torch.eq(after_key, image.full_key_sha256).all(dim=-1)
                | (after_generation != image.ball_generation)
                | (mailbox_slot != image.r06_mailbox_slot_before)
            )
        )
        after_join_fault = after_join_fault | (
            image.observe_mask
            & (mailbox_slot != image.r06_mailbox_slot_before)
        )
        if self.device.type == "cpu":
            if version_mismatch:
                after_join_fault = torch.ones_like(after_join_fault)
        else:
            after_join_fault = after_join_fault | version_mismatch
        terminal_fault = (
            image.observe_mask
            & settled_result
            & ~accepted
            & rejected
            & (fault_bits != 0)
        )
        normal_accepted = (
            image.observe_mask
            & accepted
            & ~rejected
            & (fault_bits == 0)
        )
        processed = normal_accepted | terminal_fault
        poisoned = (
            (image.observe_mask & ~processed)
            | (accepted & ~image.observe_mask)
            | (rejected & ~image.observe_mask)
            | ((fault_bits != 0) & ~image.observe_mask)
            | (settled_result & ~image.observe_mask)
            | (
                image.owner_join_fault
                & ~(terminal_fault | image.physical_orphan_mask)
            )
            | after_join_fault
        )
        safe_accepted = processed & ~poisoned
        settled = settled_result & safe_accepted
        poisoned = poisoned | (
            settled
            & (
                (after_state != R06_FLIGHT_SETTLED_RETAINED)
                | (settlement_cause <= 0)
            )
        )
        poisoned = poisoned | (
            ~settled_result & (settlement_cause != 0)
        )
        poisoned = poisoned | (
            image.observe_mask
            & ((after_state == R06_FLIGHT_SETTLED_RETAINED) != settled_result)
        )
        mailbox_abi_valid = (
            isinstance(mailbox_state, torch.Tensor)
            and mailbox_state.device == self.device
            and mailbox_state.dtype == torch.int8
            and mailbox_state.ndim == 2
            and mailbox_state.shape[0] == self.num_envs
            and mailbox_state.shape[1] > 0
        )
        if not mailbox_abi_valid:
            poisoned = torch.ones_like(poisoned)
        else:
            mailbox_in_range = (
                (mailbox_slot >= 0)
                & (mailbox_slot < mailbox_state.shape[1])
            )
            safe_mailbox_slot = mailbox_slot.clamp(
                0, mailbox_state.shape[1] - 1
            )
            gathered_mailbox_state = torch.gather(
                mailbox_state.to(torch.int64),
                1,
                safe_mailbox_slot,
            )
            poisoned = poisoned | (
                processed
                & (
                    ~mailbox_in_range
                    | (
                        settled_result
                        & (gathered_mailbox_state != 1)
                    )
                    | (
                        ~settled_result
                        & (gathered_mailbox_state != 0)
                    )
                )
            )
        safe_accepted = processed & ~poisoned
        settled = settled_result & safe_accepted
        self._active_postphysics = None
        self._active_postphysics_image = None
        if self.device.type == "cpu" and bool(poisoned.any()):
            self._device_fault.logical_or_(poisoned)
            self._poisoned = True
            self._poison_reason = (
                "R06 rejected or mismatched an observed physical slot"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        if self.device.type != "cpu":
            # Device-detected integrity faults remain sticky and policy-hidden.
            # Their single host observation is the registered PPO-boundary
            # drain; this hot path never synchronizes to raise immediately.
            self._device_fault.logical_or_(poisoned)
        terminal_fault = terminal_fault & safe_accepted

        try:
            contact_consumer = self._r06_contact_authority_consumer
            if not callable(contact_consumer):
                raise PhysicalFlightDeviceError(
                    "R06 causal selected-contact consumer was not retained"
                )
            contact_view = contact_consumer(
                result.contact_authority,
                expected_publication_identity=publication,
            )
            contact_view = _require_final_r06_type(
                contact_view,
                expected_name="LandingOutcomePostPhysicsContactAuthorityView",
            )
        except Exception as exc:
            self._device_fault.logical_or_(image.observe_mask)
            self._poisoned = True
            self._poison_reason = (
                "R06 causal selected-contact authority was unavailable"
            )
            raise PhysicalFlightOwnerPoisonedError(
                self._poison_reason
            ) from exc

        contact_mask = _tensor(
            contact_view.new_valid_contact_mask,
            label="R06 new valid selected-contact mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        contact_slot = _tensor(
            contact_view.flight_slot,
            label="R06 selected-contact flight slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        contact_key = _tensor(
            contact_view.full_key_sha256,
            label="R06 selected-contact full key",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        contact_generation = _tensor(
            contact_view.ball_generation,
            label="R06 selected-contact generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        contact_ordinal = _tensor(
            contact_view.observation_ordinal,
            label="R06 selected-contact observation ordinal",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        contact_stamp = getattr(contact_view, "selected_contact_stamp", None)
        contact_control = _tensor(
            getattr(contact_stamp, "control_step", None),
            label="R06 selected-contact control step",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        contact_substep = _tensor(
            getattr(contact_stamp, "physics_substep", None),
            label="R06 selected-contact physics substep",
            shape=shape,
            dtype=torch.int32,
            device=self.device,
        )
        contact_phase = _tensor(
            getattr(contact_stamp, "event_phase", None),
            label="R06 selected-contact event phase",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        contact_version = _r06_device_mutation_version(
            contact_view.mutation_version,
            label="R06 selected-contact mutation_version",
            device=self.device,
        )
        contact_rows_per_env = contact_mask.to(torch.int64).sum(dim=1)
        contact_collision = contact_rows_per_env > 1
        safe_slot = torch.argmax(contact_mask.to(torch.int64), dim=1)
        gather_digest = safe_slot.reshape(self.num_envs, 1, 1).expand(
            -1, 1, TOKEN_BYTES
        )
        gathered_key = torch.gather(contact_key, 1, gather_digest).squeeze(1)
        gathered_generation = torch.gather(
            contact_generation, 1, safe_slot.unsqueeze(1)
        ).squeeze(1)
        gathered_ordinal = torch.gather(
            contact_ordinal, 1, safe_slot.unsqueeze(1)
        ).squeeze(1)
        gathered_control = torch.gather(
            contact_control, 1, safe_slot.unsqueeze(1)
        ).squeeze(1)
        gathered_substep = torch.gather(
            contact_substep, 1, safe_slot.unsqueeze(1)
        ).squeeze(1)
        gathered_phase = torch.gather(
            contact_phase, 1, safe_slot.unsqueeze(1)
        ).squeeze(1)
        new_contact = contact_rows_per_env == 1
        contact_join_fault = (
            (contact_view.publication_identity is not publication)
            | (
                contact_mask
                & (
                    ~safe_accepted
                    | ~torch.eq(contact_key, after_key).all(dim=-1)
                    | (contact_generation != after_generation)
                    | (contact_slot != expected_slot)
                    | (contact_ordinal != after_ordinal)
                    | (contact_control < 0)
                    | (contact_substep < 0)
                    | (contact_phase != 0)
                )
            ).any(dim=1)
            | _r06_device_key_mismatch(
                contact_view.task_key,
                snapshot.task_key,
                device=self.device,
            )
        )
        if self.device.type == "cpu":
            if int(contact_version.item()) != int(after_version):
                contact_join_fault = torch.ones_like(contact_join_fault)
        else:
            contact_join_fault = contact_join_fault | (
                contact_version != after_version_tensor
            )
        ledger_fault = (
            contact_collision
            | contact_join_fault
            | (new_contact & self._selected_contact_pending)
        )
        accepted_contact = new_contact & ~ledger_fault
        self._selected_contact_ledger_fault.logical_or_(ledger_fault)
        self._device_fault.logical_or_(
            ledger_fault.unsqueeze(1).expand(shape)
        )
        if not self._action_epoch_runtime_call_active:
            self._selected_contact_pending.logical_or_(accepted_contact)
            self._selected_contact_reward_cycle_open = True
            self._selected_contact_pending_flight_slot.copy_(
                torch.where(
                    accepted_contact,
                    safe_slot,
                    self._selected_contact_pending_flight_slot,
                )
            )
            self._selected_contact_pending_full_key_sha256.copy_(
                torch.where(
                    accepted_contact.unsqueeze(-1),
                    gathered_key,
                    self._selected_contact_pending_full_key_sha256,
                )
            )
            self._selected_contact_pending_ball_generation.copy_(
                torch.where(
                    accepted_contact,
                    gathered_generation,
                    self._selected_contact_pending_ball_generation,
                )
            )
            self._selected_contact_pending_observation_ordinal.copy_(
                torch.where(
                    accepted_contact,
                    gathered_ordinal,
                    self._selected_contact_pending_observation_ordinal,
                )
            )
            self._selected_contact_pending_control_step.copy_(
                torch.where(
                    accepted_contact,
                    gathered_control,
                    self._selected_contact_pending_control_step,
                )
            )
            self._selected_contact_pending_physics_substep.copy_(
                torch.where(
                    accepted_contact,
                    gathered_substep,
                    self._selected_contact_pending_physics_substep,
                )
            )
            self._selected_contact_pending_event_phase.copy_(
                torch.where(
                    accepted_contact,
                    gathered_phase,
                    self._selected_contact_pending_event_phase,
                )
            )
            self._selected_contact_viewed.copy_(
                torch.where(
                    accepted_contact,
                    torch.zeros_like(self._selected_contact_viewed),
                    self._selected_contact_viewed,
                )
            )
        if self.device.type == "cpu" and bool(ledger_fault.any()):
            self._poisoned = True
            self._poison_reason = (
                "R06 causal selected-contact authority collided or mismatched"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)

        physical_orphan = image.physical_orphan_mask.clone()
        r06_orphan = image.r06_orphan_mask & terminal_fault
        retirement_mask = settled | physical_orphan
        cleanup_only = (
            (terminal_fault & image.owner_join_fault) | physical_orphan
        )
        authority_rows: list[dict[str, object]] = []
        if self.device.type == "cpu":
            for env_id, slot in [
                (int(index[0]), int(index[1]))
                for index in settled.nonzero(as_tuple=False).tolist()
            ]:
                if not bool(cleanup_only[env_id, slot]):
                    authority_rows.append(
                        {
                            "env_id": env_id,
                            "slot_index": slot,
                            "outcome_key_sha256": _digest_hex(
                                after_key[env_id, slot]
                            ),
                            "ball_generation": int(
                                after_generation[env_id, slot]
                            ),
                        }
                    )

            after_snapshot = R06PhysicalFlightReadOnlySnapshot(
                flight_state=after_state,
                full_key_sha256=after_key,
                ball_generation=after_generation,
                observation_ordinal=after_ordinal,
                owner_mutation_version=after_version,
            )
            after_root: str | None = r06_physical_snapshot_root(after_snapshot)
        else:
            after_root = None
        settlement_authority: Optional[_flight.CanonicalJsonContentPin] = None
        if authority_rows and after_root is not None:
            authority_payload = {
                "schema_version": 2,
                "kind": _flight.PHYSICAL_SETTLEMENT_AUTHORITY_KIND,
                "mailbox_lifecycle": "SETTLED_UNPAID",
                "r06_owner_mutation_version": after_version,
                "r06_after_root_sha256": after_root,
                "physical_retire_rows": authority_rows,
            }
            authority_mapping = {
                **authority_payload,
                "canonical_sha256": _flight.canonical_sha256(
                    authority_payload
                ),
            }
            settlement_authority = (
                _flight.CanonicalJsonContentPin.from_sealed_mapping(
                    authority_mapping,
                    expected_source_kind=(
                        _flight.PHYSICAL_SETTLEMENT_AUTHORITY_KIND
                    ),
                    source_schema_sha256=(
                        _flight.PHYSICAL_SETTLEMENT_AUTHORITY_SCHEMA_SHA256
                    ),
                )
            )

        after_slots = list(self._host_slots)
        physical_metadata_update = safe_accepted & ~r06_orphan
        if self.device.type == "cpu":
            for env_id, slot in [
                (int(index[0]), int(index[1]))
                for index in physical_metadata_update.nonzero(
                    as_tuple=False
                ).tolist()
            ]:
                pre = self._host_slots[self._slot_offset(env_id, slot)]
                current_state = _canonical_state_from_cpu_tensor(
                    image.current_state_f32[env_id, slot]
                )
                after_slots[self._slot_offset(env_id, slot)] = replace(
                    pre,
                    lifecycle=(
                        _flight.SLOT_SETTLED_RETAINED
                        if int(after_state[env_id, slot])
                        == R06_FLIGHT_SETTLED_RETAINED
                        else _flight.SLOT_IN_FLIGHT
                    ),
                    current_state_f32=current_state,
                    current_state_f32_sha256=current_state.state_bytes_sha256,
                    last_control_step=int(
                        image.observation_control_step[env_id, slot]
                    ),
                    last_physics_substep=int(
                        image.observation_physics_substep[env_id, slot]
                    ),
                    mutation_version=pre.mutation_version + 1,
                )

        self._host_slots = tuple(after_slots)
        self._lifecycle.copy_(
            torch.where(physical_metadata_update, after_state, self._lifecycle)
        )
        self._observation_ordinal.copy_(
            torch.where(
                physical_metadata_update,
                image.observation_ordinal,
                self._observation_ordinal,
            )
        )
        self._previous_ball_center.copy_(
            torch.where(
                physical_metadata_update.unsqueeze(-1),
                image.current_ball_center_m,
                self._previous_ball_center,
            )
        )
        self._slot_version.add_(physical_metadata_update.to(torch.int64))
        self._device_fault.logical_or_(terminal_fault | physical_orphan)
        self._advance_owner_mutation_version()
        self._active_r06_ack = result
        self._active_r06_ack_image = _AcknowledgedR06Image(
            flight_state=after_state.clone(),
            full_key_sha256=after_key.clone(),
            ball_generation=after_generation.clone(),
            mailbox_slot=mailbox_slot.clone(),
            observation_ordinal=after_ordinal.clone(),
            current_state_f32=image.current_state_f32.clone(),
            owner_mutation_version=after_version,
            snapshot_root_sha256=after_root,
            settlement_authority=settlement_authority,
            r06_owner=owner,
            terminal_fault_mask=terminal_fault.clone(),
            cleanup_only_mask=cleanup_only.clone(),
            physical_orphan_mask=physical_orphan.clone(),
            r06_orphan_mask=r06_orphan.clone(),
            r06_settled_mask=settled.clone(),
            settled_mask=retirement_mask.clone(),
        )
        return result

    def selected_contact_reward_view(self) -> PhysicalSelectedContactRewardView:
        """Issue the exact pending diagnostic contact view once."""

        self._require_operable()
        if self._active_selected_contact_reward_view is not None:
            self._selected_contact_ledger_fault.logical_or_(
                torch.ones_like(self._selected_contact_pending)
            )
            raise PhysicalFlightDeviceError(
                "selected-contact Reward view was requested twice"
            )
        eligible = self._selected_contact_pending & ~self._selected_contact_viewed
        view = PhysicalSelectedContactRewardView(
            eligible=eligible.detach().clone(),
            full_key_sha256=torch.where(
                eligible.unsqueeze(-1),
                self._selected_contact_pending_full_key_sha256,
                torch.zeros_like(self._selected_contact_pending_full_key_sha256),
            ).detach().clone(),
            ball_generation=torch.where(
                eligible,
                self._selected_contact_pending_ball_generation,
                torch.full_like(self._selected_contact_pending_ball_generation, -1),
            ).detach().clone(),
            flight_slot=torch.where(
                eligible,
                self._selected_contact_pending_flight_slot,
                torch.full_like(self._selected_contact_pending_flight_slot, -1),
            ).detach().clone(),
            observation_ordinal=torch.where(
                eligible,
                self._selected_contact_pending_observation_ordinal,
                torch.full_like(
                    self._selected_contact_pending_observation_ordinal, -1
                ),
            ).detach().clone(),
            selected_contact_control_step=torch.where(
                eligible,
                self._selected_contact_pending_control_step,
                torch.full_like(self._selected_contact_pending_control_step, -1),
            ).detach().clone(),
            selected_contact_physics_substep=torch.where(
                eligible,
                self._selected_contact_pending_physics_substep,
                torch.full_like(
                    self._selected_contact_pending_physics_substep, -1
                ),
            ).detach().clone(),
            selected_contact_event_phase=torch.where(
                eligible,
                self._selected_contact_pending_event_phase,
                torch.full_like(self._selected_contact_pending_event_phase, -1),
            ).detach().clone(),
            _owner_identity=self._owner_identity,
            _token=_CONTACT_REWARD_VIEW_TOKEN,
        )
        self._selected_contact_viewed.logical_or_(eligible)
        self._selected_contact_view_total.add_(self.num_envs)
        self._active_selected_contact_reward_view = _PhysicalContactRewardViewRecord(
            view=view,
            view_issued=True,
        )
        self._advance_owner_mutation_version()
        return view

    def record_selected_contact_reward_payment(
        self,
        view: PhysicalSelectedContactRewardView,
        *,
        raw_reward: torch.Tensor,
    ) -> PhysicalSelectedContactRewardPaymentResult:
        """Finish the local diagnostic contact ledger; zero still pays."""

        self._require_operable()
        record = self._active_selected_contact_reward_view
        if id(view) in self._diagnostic_paid_selected_contact_reward_verdicts:
            self._selected_contact_ledger_fault.logical_or_(
                torch.ones_like(self._selected_contact_pending)
            )
            raise PhysicalFlightDeviceError(
                "selected-contact Reward payment view is stale, foreign, or replayed"
            )
        value = _tensor(
            raw_reward,
            label="selected-contact raw reward",
            shape=(self.num_envs,),
            dtype=torch.float32,
            device=self.device,
        )
        if (
            type(view) is not PhysicalSelectedContactRewardView
            or view._owner_identity is not self._owner_identity
            or view._token is not _CONTACT_REWARD_VIEW_TOKEN
            or record is None
            or record.view is not view
            or not record.view_issued
        ):
            self._selected_contact_ledger_fault.logical_or_(
                torch.ones_like(self._selected_contact_pending)
            )
            raise PhysicalFlightDeviceError(
                "selected-contact Reward payment view is stale, foreign, or replayed"
            )
        eligible = _tensor(
            view.eligible,
            label="selected-contact Reward eligible",
            shape=(self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )
        accepted = (
            eligible
            & self._selected_contact_pending
            & self._selected_contact_viewed
            & torch.isfinite(value)
        )
        rejected = eligible & ~accepted
        self._selected_contact_ledger_fault.logical_or_(rejected)
        self._device_fault.logical_or_(
            rejected.unsqueeze(1).expand(self.num_envs, self.flight_capacity)
        )
        self._selected_contact_pending.logical_and_(~accepted)
        self._selected_contact_viewed.logical_and_(~accepted)
        self._selected_contact_payment_total.add_(self.num_envs)
        paid = torch.where(accepted, value, torch.zeros_like(value))
        self._active_selected_contact_reward_view = None
        self._last_paid_selected_contact_reward_view = view
        verdict = PhysicalSelectedContactRewardPaymentResult(
            accepted=accepted.detach().clone(),
            rejected=rejected.detach().clone(),
            paid_raw_reward=paid.detach().clone(),
            _owner_identity=self._owner_identity,
            _token=_CONTACT_REWARD_PAYMENT_TOKEN,
        )
        self._diagnostic_paid_selected_contact_reward_verdicts.add(id(view))
        self._selected_contact_reward_cycle_open = False
        self._advance_owner_mutation_version()
        if self.device.type == "cpu" and bool(rejected.any()):
            self._poison_physical_reward(
                "selected-contact Reward payment was invalid"
            )
            raise PhysicalFlightOwnerPoisonedError(
                self._physical_reward_poison_reason
                or "selected-contact Reward payment was invalid"
            )
        return verdict

    def _validate_r06_retire_result(
        self,
        value: object,
        *,
        label: str,
    ) -> object:
        result = _require_final_r06_type(
            value,
            expected_name="PhysicalRetireMutationResult",
        )
        shape = (self.num_envs, self.flight_capacity)
        accepted = _tensor(
            result.accepted,
            label=f"{label}.accepted",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        rejected = _tensor(
            result.rejected,
            label=f"{label}.rejected",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        fault_bits = _tensor(
            result.fault_bits,
            label=f"{label}.fault_bits",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        normal = _tensor(
            result.normal_mask,
            label=f"{label}.normal_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        cleanup = _tensor(
            result.cleanup_mask,
            label=f"{label}.cleanup_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        portable = _tensor(
            result.portable_success_mask,
            label=f"{label}.portable_success_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        root = _require_final_r06_type(
            result.final_lifecycle_root,
            expected_name="FlightLifecycleSnapshotBatch",
        )
        _tensor(
            root.state,
            label=f"{label}.final.state",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        _tensor(
            root.full_key_sha256,
            label=f"{label}.final.full_key_sha256",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        _tensor(
            root.ball_generation,
            label=f"{label}.final.ball_generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        _tensor(
            root.mailbox_slot,
            label=f"{label}.final.mailbox_slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        _tensor(
            root.observation_ordinal,
            label=f"{label}.final.observation_ordinal",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        _tensor(
            root.physical_retired,
            label=f"{label}.final.physical_retired",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        mailbox_retired = root.mailbox_physical_retired
        if (
            not isinstance(mailbox_retired, torch.Tensor)
            or mailbox_retired.device != self.device
            or mailbox_retired.dtype != torch.bool
            or mailbox_retired.ndim != 2
            or mailbox_retired.shape[0] != self.num_envs
        ):
            raise PhysicalFlightDeviceError(
                f"{label}.final.mailbox_physical_retired ABI differs"
            )
        _r06_device_mutation_version(
            result.mutation_version_before,
            label=f"{label}.mutation_version_before",
            device=self.device,
        )
        _r06_device_mutation_version(
            result.mutation_version_after,
            label=f"{label}.mutation_version_after",
            device=self.device,
        )
        if self.device.type != "cpu":
            # The exact R06 owner has already constructed this typed result.
            # Cross-owner numeric equality remains a device verdict and is
            # folded into the physical cleanup/fault mask by the caller.
            return result
        if (
            not torch.equal(accepted, normal | cleanup)
            or bool((normal & cleanup).any())
            or bool((accepted & rejected).any())
            or bool((portable & ~normal).any())
            or (bool(cleanup.any()) and bool(portable.any()))
            or not torch.equal(result.full_key_sha256, root.full_key_sha256)
            or not torch.equal(result.ball_generation, root.ball_generation)
            or not torch.equal(result.mailbox_slot, root.mailbox_slot)
            or not torch.equal(
                result.observation_ordinal,
                root.observation_ordinal,
            )
            or not torch.equal(result.physical_retired, root.physical_retired)
            or not torch.equal(
                result.mailbox_physical_retired,
                root.mailbox_physical_retired,
            )
            or not torch.equal(
                result.mutation_version_after,
                root.mutation_version,
            )
            or not _r06_device_key_equal(result.task_key, root.task_key)
            or bool((cleanup & (fault_bits == 0)).any())
        ):
            raise PhysicalFlightDeviceError(
                f"{label} partition/final lifecycle root differs"
            )
        return result

    def prepare_settle_retire(
        self,
        r06_prepared_retire: object,
    ) -> PreparedPhysicalSettleRetire:
        """Prebuild the physical park after-image for one exact R06 prepare."""

        self._require_operable()
        if (
            self._active_prepare is not None
            or self._active_postphysics is not None
            or self._active_physical_retire_prepare is not None
            or self._active_physical_retire_arm is not None
            or self._active_physical_retire_commit is not None
            or self._active_physical_global_drain is not None
            or self._active_r06_ack is None
            or self._active_r06_ack_image is None
        ):
            raise PhysicalFlightDeviceError(
                "physical settle-retire prepare crossed another active lease"
            )
        owner = self._r06_owner
        if owner is None:
            raise PhysicalFlightDeviceError("physical R06 owner is not paired")
        prepared = _require_final_r06_type(
            r06_prepared_retire,
            expected_name="PreparedPhysicalRetire",
        )
        post_result = self._active_r06_ack
        if getattr(prepared, "_settlement_result", None) is not post_result:
            raise PhysicalFlightDeviceError(
                "R06 retire prepare does not retain the exact postphysics result"
            )
        try:
            predicted = self._validate_r06_retire_result(
                owner.prepared_physical_retire_result(prepared),
                label="R06 predicted retire",
            )
            r06_cleanup_capability = _require_final_r06_type(
                owner.physical_retire_cleanup_capability(prepared),
                expected_name="PhysicalRetireCleanupMaskCapability",
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "R06 retire prepared evidence could not be acquired"
            ) from exc
        ack_image = self._active_r06_ack_image
        if self.device.type == "cpu" and (
            bool(predicted.rejected.any())
            or not torch.equal(predicted.accepted, ack_image.r06_settled_mask)
            or not torch.equal(
                getattr(r06_cleanup_capability, "_device_mask", None),
                predicted.cleanup_mask,
            )
        ):
            try:
                owner.abort_physical_retire(prepared)
            finally:
                self._poisoned = True
                self._poison_reason = (
                    "R06 retire prepare rejected or changed the retained settlement"
                )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)

        scene_before = _tensor(
            self.scene_port.read_state_env(),
            label="physical retire scene before",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        ).clone()
        if self.device.type == "cpu" and not torch.equal(
            scene_before,
            ack_image.current_state_f32,
        ):
            try:
                owner.abort_physical_retire(prepared)
            finally:
                self._poisoned = True
                self._poison_reason = (
                    "physical scene advanced between postphysics settlement and park prepare"
                )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)

        physical_live = self._published & ~self._parked
        physical_cleanup = ack_image.physical_orphan_mask.clone()
        if self.device.type != "cpu":
            capability_mask = getattr(
                r06_cleanup_capability,
                "_device_mask",
                None,
            )
            capability_mismatch = _device_tensor_mismatch(
                capability_mask,
                predicted.cleanup_mask,
                device=self.device,
            )
            scene_mismatch = ~torch.eq(
                scene_before,
                ack_image.current_state_f32,
            ).all(dim=-1)
            predicted_mismatch = (
                predicted.rejected
                | (predicted.accepted != ack_image.r06_settled_mask)
                | capability_mismatch
            )
            physical_cleanup.logical_or_(scene_mismatch | predicted_mismatch)
        combined_cleanup = predicted.cleanup_mask | physical_cleanup
        park_mask = (predicted.accepted & physical_live) | physical_cleanup
        metadata_mask = park_mask | combined_cleanup
        handle = PreparedPhysicalSettleRetire(
            r06_prepared_retire=prepared,
            r06_cleanup_capability=r06_cleanup_capability,
            physical_cleanup_mask=physical_cleanup.detach().clone(),
            owner_identity=self._owner_identity,
            _token=_PHYSICAL_RETIRE_PREPARE_TOKEN,
        )
        module = sys.modules.get(type(owner).__module__)
        claim_type = None if module is None else getattr(
            module,
            "LandingOutcomePhysicalParkPreparedTokenClaim",
            None,
        )
        if claim_type is None:
            raise PhysicalFlightDeviceError(
                "R06 physical park prepared-token claim type is unavailable"
            )
        claim = claim_type(
            r06_prepared_retire=prepared,
            r06_cleanup_capability=r06_cleanup_capability,
            physical_cleanup_capability=(
                handle._physical_cleanup_capability
            ),
            _physical_prepared_token=handle,
        )

        park_state = scene_before.clone()
        park_state[..., :3] = torch.tensor(
            (0.0, 0.0, -20.0),
            dtype=torch.float32,
            device=self.device,
        )
        park_state[..., 3:7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float32,
            device=self.device,
        )
        park_state[..., 7:] = 0.0
        scene_handle = None
        if self.device.type != "cpu" or bool(park_mask.any()):
            try:
                scene_handle = self._prepare_action_epoch_scene_write(
                    kind="retire",
                    state_env_f32=park_state,
                    selected_mask=park_mask,
                )
            except Exception:
                owner.abort_physical_retire(prepared)
                raise

        after_slots = list(self._host_slots)
        if self.device.type == "cpu":
            for env_id, slot in (
                (int(index[0]), int(index[1]))
                for index in metadata_mask.nonzero(as_tuple=False).tolist()
            ):
                pre = after_slots[self._slot_offset(env_id, slot)]
                if bool(park_mask[env_id, slot]):
                    after_slots[self._slot_offset(env_id, slot)] = replace(
                        pre,
                        lifecycle=_flight.SLOT_RETIRED,
                        mutation_version=pre.mutation_version + 1,
                        physically_parked=True,
                        published_to_runtime=False,
                    )
                else:
                    after_slots[self._slot_offset(env_id, slot)] = replace(
                        pre,
                        mutation_version=pre.mutation_version + 1,
                    )
        lifecycle = torch.where(
            park_mask,
            torch.full_like(self._lifecycle, R06_FLIGHT_EMPTY),
            self._lifecycle,
        )
        # Portable retired rows retain the last observed physical state.  Only
        # the scene tensor moves to the park pose; otherwise a retire receipt
        # would erase the evidence it is meant to preserve.
        previous_center = self._previous_ball_center.clone()
        parked = self._parked | park_mask
        published = self._published & ~park_mask
        slot_version = self._slot_version + metadata_mask.to(torch.int64)
        device_fault = self._device_fault | combined_cleanup
        self._active_physical_retire_prepare = handle
        self._active_physical_retire_image = _PreparedPhysicalSettleRetireImage(
            postphysics_result=post_result,
            r06_predicted_result=predicted,
            r06_armed_result=None,
            claim=claim,
            stale_witness=_PhysicalRetireStaleWitness(
                scene_state=scene_before.detach().clone(),
                lifecycle=self._lifecycle.detach().clone(),
                observation_ordinal=self._observation_ordinal.detach().clone(),
                previous_ball_center=self._previous_ball_center.detach().clone(),
                parked=self._parked.detach().clone(),
                published=self._published.detach().clone(),
                slot_version=self._slot_version.detach().clone(),
                device_fault=self._device_fault.detach().clone(),
                owner_mutation_version=self._mutation_version,
                postphysics_result=post_result,
            ),
            park_mask=park_mask.detach().clone(),
            physical_cleanup_mask=physical_cleanup.detach().clone(),
            scene_handle=scene_handle,
            after_slots=tuple(after_slots),
            lifecycle=lifecycle,
            observation_ordinal=self._observation_ordinal.clone(),
            previous_ball_center=previous_center,
            parked=parked,
            published=published,
            slot_version=slot_version,
            device_fault=device_fault,
            portable_receipt=None,
        )
        return handle

    def _fail_unarmed_physical_retire(self, reason: str) -> None:
        """Abort both unarmed after-images, then enter sticky HOLD."""

        image = self._active_physical_retire_image
        prepared = self._active_physical_retire_prepare
        failures: list[BaseException] = []
        if image is not None and image.scene_handle is not None:
            try:
                self._abort_scene_write(image.scene_handle)
            except BaseException as exc:
                failures.append(exc)
        if prepared is not None and self._r06_owner is not None:
            try:
                self._r06_owner.abort_physical_retire(
                    prepared._r06_prepared_retire
                )
            except BaseException as exc:
                failures.append(exc)
        if self._r06_owner is not None:
            try:
                self._r06_owner.poison_global_reveal_epoch(reason)
            except BaseException as exc:
                failures.append(exc)
        self._active_physical_retire_prepare = None
        self._active_physical_retire_image = None
        self._poisoned = True
        self._poison_reason = (
            reason if not failures else f"{reason}; paired abort also failed"
        )
        raise PhysicalFlightOwnerPoisonedError(self._poison_reason)

    def _require_physical_retire_stale_witness(
        self,
        image: _PreparedPhysicalSettleRetireImage,
    ) -> None:
        """Reject scene/device drift while both owners are still abortable."""

        witness = image.stale_witness
        if self.device.type != "cpu":
            # The production coordinator calls prepare -> R06 arm without
            # yielding ownership of either leaf.  Re-reading CUDA values here
            # would be a second, synchronizing gate over a state that cannot
            # legitimately advance in that interval.  Exact active token and
            # owner identities below remain the authority for this handoff.
            if (
                witness.postphysics_result is not self._active_r06_ack
                or witness.owner_mutation_version != self._mutation_version
            ):
                self._fail_unarmed_physical_retire(
                    "physical owner chronology changed after retire prepare"
                )
            return
        try:
            scene_now = _tensor(
                self.scene_port.read_state_env(),
                label="physical retire prearm scene witness",
                shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
                dtype=torch.float32,
                device=self.device,
            )
            stable = (
                witness.postphysics_result is self._active_r06_ack
                and witness.owner_mutation_version == self._mutation_version
                and torch.equal(scene_now, witness.scene_state)
                and torch.equal(self._lifecycle, witness.lifecycle)
                and torch.equal(
                    self._observation_ordinal,
                    witness.observation_ordinal,
                )
                and torch.equal(
                    self._previous_ball_center,
                    witness.previous_ball_center,
                )
                and torch.equal(self._parked, witness.parked)
                and torch.equal(self._published, witness.published)
                and torch.equal(self._slot_version, witness.slot_version)
                and torch.equal(self._device_fault, witness.device_fault)
            )
        except BaseException:
            stable = False
        if not stable:
            self._fail_unarmed_physical_retire(
                "physical scene/device changed after retire prepare"
            )

    def require_owned_r06_physical_park_prepared_token(
        self,
        physical_prepared_token: object,
    ) -> object:
        """R06 callback: validate the exact retained physical park prepare."""

        image = self._active_physical_retire_image
        if (
            self._poisoned
            or type(physical_prepared_token)
            is not PreparedPhysicalSettleRetire
            or physical_prepared_token
            is not self._active_physical_retire_prepare
            or physical_prepared_token._owner_identity
            is not self._owner_identity
            or physical_prepared_token._token
            is not _PHYSICAL_RETIRE_PREPARE_TOKEN
            or image is None
            or image.claim._physical_prepared_token
            is not physical_prepared_token
        ):
            raise PhysicalFlightDeviceError(
                "physical park prepared token is stale or foreign"
            )
        self._require_physical_retire_stale_witness(image)
        self._require_physical_retire_capability_integrity(
            physical_prepared_token,
            image,
            abortable=True,
        )
        return image.claim

    def _require_physical_retire_capability_integrity(
        self,
        physical_prepared_token: PreparedPhysicalSettleRetire,
        image: _PreparedPhysicalSettleRetireImage,
        *,
        abortable: bool,
    ) -> None:
        """Reject in-place mutation of either exported cleanup capability.

        Both R06 and physical retain private third copies of their device masks.
        The public opaque objects necessarily expose a tensor to the paired
        owner, so object identity alone is not evidence that the tensor contents
        still match the prepared after-image.  The current retirement path is a
        CPU reference and may synchronously reject here.  CUDA remains
        fail-closed until this equality is folded into the sole packed device
        verdict without a host branch.
        """

        claim = image.claim
        r06_capability = physical_prepared_token._r06_cleanup_capability
        physical_capability = physical_prepared_token._physical_cleanup_capability
        prepared_r06 = physical_prepared_token._r06_prepared_retire
        predicted_cleanup = image.r06_predicted_result.cleanup_mask
        r06_mask = getattr(r06_capability, "_device_mask", None)
        prepared_r06_mask = getattr(prepared_r06, "_cleanup_mask", None)
        physical_mask = getattr(physical_capability, "_device_mask", None)
        shape = (self.num_envs, self.flight_capacity)
        def fail(reason: str) -> None:
            if abortable:
                self._fail_unarmed_physical_retire(reason)
            self._poisoned = True
            self._poison_reason = reason
            raise PhysicalFlightOwnerPoisonedError(reason)

        if (
            claim.r06_prepared_retire is not prepared_r06
            or claim.r06_cleanup_capability is not r06_capability
            or claim.physical_cleanup_capability is not physical_capability
            or claim._physical_prepared_token is not physical_prepared_token
            or type(physical_capability) is not PhysicalParkCleanupMaskCapability
            or physical_capability._owner_identity is not self._owner_identity
            or physical_capability._prepared_token is not physical_prepared_token
            or physical_capability._token is not _PHYSICAL_PARK_CLEANUP_TOKEN
            or not isinstance(r06_mask, torch.Tensor)
            or not isinstance(prepared_r06_mask, torch.Tensor)
            or not isinstance(physical_mask, torch.Tensor)
            or tuple(r06_mask.shape) != shape
            or tuple(prepared_r06_mask.shape) != shape
            or tuple(physical_mask.shape) != shape
            or r06_mask.dtype != torch.bool
            or prepared_r06_mask.dtype != torch.bool
            or physical_mask.dtype != torch.bool
            or r06_mask.device != self.device
            or prepared_r06_mask.device != self.device
            or physical_mask.device != self.device
        ):
            fail("physical/R06 cleanup capability structure changed")
        if self.device.type != "cpu":
            # R06 retains its own third copy and folds any R06 capability drift
            # into its device cleanup verdict.  The physical capability is
            # owner-private and the fresh top-level call graph does not expose
            # a mutation point between prepare and arm.
            return
        if (
            not torch.equal(r06_mask, predicted_cleanup)
            or not torch.equal(prepared_r06_mask, predicted_cleanup)
            or not torch.equal(physical_mask, image.physical_cleanup_mask)
        ):
            fail("physical/R06 cleanup capability was mutated in place")

    def require_committed_r06_physical_park_prepared_token(
        self,
        physical_prepared_token: object,
    ) -> object:
        """R06 callback: prove that the exact physical park already committed."""

        image = self._active_physical_retire_image
        committed = self._active_physical_retire_commit
        if (
            type(physical_prepared_token)
            is not PreparedPhysicalSettleRetire
            or physical_prepared_token
            is not self._active_physical_retire_prepare
            or physical_prepared_token._owner_identity
            is not self._owner_identity
            or physical_prepared_token._token
            is not _PHYSICAL_RETIRE_PREPARE_TOKEN
            or image is None
            or type(committed) is not PhysicalSettleRetireCommitToken
            or committed._owner_identity is not self._owner_identity
            or committed._token is not _PHYSICAL_RETIRE_COMMIT_TOKEN
            or committed._armed_retire._prepared_retire
            is not physical_prepared_token
        ):
            raise PhysicalFlightDeviceError(
                "physical park commit token is stale or foreign"
            )
        self._require_physical_retire_capability_integrity(
            physical_prepared_token,
            image,
            abortable=False,
        )
        return image.claim

    def arm_prevalidated_settle_retire(
        self,
        prepared_retire: PreparedPhysicalSettleRetire,
        r06_armed_retire: object,
    ) -> ArmedPhysicalSettleRetire:
        """Cross-check R06's final union and retain one no-fail park handle."""

        self._require_operable()
        image = self._active_physical_retire_image
        if (
            type(prepared_retire) is not PreparedPhysicalSettleRetire
            or prepared_retire is not self._active_physical_retire_prepare
            or prepared_retire._owner_identity is not self._owner_identity
            or prepared_retire._token is not _PHYSICAL_RETIRE_PREPARE_TOKEN
            or image is None
            or self._active_physical_retire_arm is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical settle-retire prepare is stale or foreign"
            )
        armed_r06 = _require_final_r06_type(
            r06_armed_retire,
            expected_name="ArmedPhysicalRetire",
        )
        if (
            getattr(armed_r06, "_prepared_retire", None)
            is not prepared_retire._r06_prepared_retire
            or getattr(armed_r06, "_physical_prepared_token", None)
            is not prepared_retire
        ):
            raise PhysicalFlightDeviceError(
                "R06 armed retire does not bind the physical prepare"
            )
        cross = self._validate_r06_retire_result(
            self._r06_owner.armed_physical_retire_result(armed_r06),
            label="R06 armed retire",
        )
        predicted = image.r06_predicted_result
        physical_cleanup = image.physical_cleanup_mask
        expected_cleanup = predicted.cleanup_mask | physical_cleanup
        expected_accepted = predicted.accepted | physical_cleanup
        expected_normal = predicted.normal_mask & ~expected_cleanup
        expected_rejected = predicted.rejected & ~expected_accepted
        expected_portable = expected_normal & ~expected_cleanup.any()
        physical_live = self._published & ~self._parked
        expected_park = (cross.accepted & physical_live) | physical_cleanup
        if self.device.type == "cpu" and (
            not torch.equal(cross.cleanup_mask, expected_cleanup)
            or not torch.equal(cross.accepted, expected_accepted)
            or not torch.equal(cross.normal_mask, expected_normal)
            or not torch.equal(cross.rejected, expected_rejected)
            or not torch.equal(cross.portable_success_mask, expected_portable)
            or not torch.equal(image.park_mask, expected_park)
        ):
            self._poisoned = True
            self._poison_reason = "R06 armed cleanup union differs"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        armed = ArmedPhysicalSettleRetire(
            _prepared_retire=prepared_retire,
            _r06_armed_retire=armed_r06,
            _owner_identity=self._owner_identity,
            _token=_PHYSICAL_RETIRE_ARM_TOKEN,
        )
        self._active_physical_retire_arm = armed
        cross_mismatch = torch.zeros(
            (), dtype=torch.bool, device=self.device
        )
        if self.device.type != "cpu":
            cross_mismatch = (
                ~torch.eq(cross.cleanup_mask, expected_cleanup).all()
                | ~torch.eq(cross.accepted, expected_accepted).all()
                | ~torch.eq(cross.normal_mask, expected_normal).all()
                | ~torch.eq(cross.rejected, expected_rejected).all()
                | ~torch.eq(
                    cross.portable_success_mask,
                    expected_portable,
                ).all()
                | ~torch.eq(image.park_mask, expected_park).all()
            )
        self._active_physical_retire_image = replace(
            image,
            r06_armed_result=cross,
            device_fault=(
                self._device_fault
                | cross.cleanup_mask
                | cross_mismatch
            ),
        )
        return armed

    def commit_prevalidated_settle_retire(
        self,
        armed_retire: ArmedPhysicalSettleRetire,
    ) -> PhysicalSettleRetireCommitToken:
        """Publish physical park/metadata first; expose only an opaque token."""

        self._require_operable()
        image = self._active_physical_retire_image
        if (
            type(armed_retire) is not ArmedPhysicalSettleRetire
            or armed_retire is not self._active_physical_retire_arm
            or armed_retire._owner_identity is not self._owner_identity
            or armed_retire._token is not _PHYSICAL_RETIRE_ARM_TOKEN
            or image is None
            or image.r06_armed_result is None
            or self._active_physical_retire_commit is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical settle-retire arm is stale or foreign"
            )
        try:
            if image.scene_handle is not None:
                self._apply_scene_write(image.scene_handle)
            if self.device.type == "cpu":
                self._host_slots = image.after_slots
            self._lifecycle.copy_(image.lifecycle)
            self._observation_ordinal.copy_(image.observation_ordinal)
            self._previous_ball_center.copy_(image.previous_ball_center)
            self._parked.copy_(image.parked)
            self._published.copy_(image.published)
            self._slot_version.copy_(image.slot_version)
            self._device_fault.copy_(image.device_fault)
            if self.device.type != "cpu":
                # A closed all-grid retire transaction is itself physical-owner
                # chronology, including the exact empty transaction.
                self._advance_owner_mutation_version()
            elif bool(
                (image.park_mask | image.r06_armed_result.cleanup_mask).any()
            ):
                self._advance_owner_mutation_version()
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "physical retirement scene publication failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        committed = PhysicalSettleRetireCommitToken(
            _armed_retire=armed_retire,
            _owner_identity=self._owner_identity,
            _token=_PHYSICAL_RETIRE_COMMIT_TOKEN,
        )
        self._active_physical_retire_commit = committed
        return committed

    def complete_prevalidated_settle_retire(
        self,
        physical_commit_token: PhysicalSettleRetireCommitToken,
        actual_r06_result: object,
    ) -> PhysicalRetireDeviceResult:
        """Validate actual R06-last publication, then expose final typed result."""

        image = self._active_physical_retire_image
        if (
            type(physical_commit_token) is not PhysicalSettleRetireCommitToken
            or physical_commit_token
            is not self._active_physical_retire_commit
            or physical_commit_token._owner_identity is not self._owner_identity
            or physical_commit_token._token
            is not _PHYSICAL_RETIRE_COMMIT_TOKEN
            or image is None
            or image.r06_armed_result is None
        ):
            raise PhysicalFlightDeviceError(
                "physical settle-retire commit token is stale or foreign"
            )
        try:
            actual = self._validate_r06_retire_result(
                actual_r06_result,
                label="R06 committed retire",
            )
            current = _require_final_r06_type(
                self._r06_owner.current_flight_lifecycle_snapshot(),
                expected_name="FlightLifecycleSnapshotBatch",
            )
            if self.device.type == "cpu" and (
                not _r06_retire_result_equal(actual, image.r06_armed_result)
                or not _r06_lifecycle_snapshot_equal(
                    current,
                    actual.final_lifecycle_root,
                )
            ):
                raise PhysicalFlightDeviceError(
                    "actual R06 retire differs from the prevalidated final root"
                )
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "actual R06 physical retirement failed after park"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc

        accepted = actual.accepted.detach().clone()
        rejected = actual.rejected.detach().clone()
        cleanup = actual.cleanup_mask.detach().clone()
        completion_mismatch = torch.zeros_like(cleanup)
        if self.device.type != "cpu":
            completion_mismatch = (
                _r06_retire_result_device_mismatch(
                    actual,
                    image.r06_armed_result,
                    device=self.device,
                )
                | _r06_lifecycle_snapshot_device_mismatch(
                    current,
                    actual.final_lifecycle_root,
                    device=self.device,
                )
            ).expand_as(cleanup)
            self._device_fault.logical_or_(completion_mismatch)
        result = PhysicalRetireDeviceResult(
            accepted=accepted,
            rejected=rejected,
            owner_join_fault=(
                cleanup | rejected | completion_mismatch
            ).detach().clone(),
            portable_receipt=image.portable_receipt,
        )
        self._shared_normal_retire_total.add_(
            actual.normal_mask.to(torch.int64).sum()
        )
        self._physical_only_orphan_park_total.add_(
            image.physical_cleanup_mask.to(torch.int64).sum()
        )
        self._accumulate_normal_retire_key_summaries_(
            actual.normal_mask,
            actual.full_key_sha256,
        )
        self._active_r06_ack = None
        self._active_r06_ack_image = None
        self._active_physical_retire_prepare = None
        self._active_physical_retire_image = None
        self._active_physical_retire_arm = None
        self._active_physical_retire_commit = None
        if self.device.type == "cpu" and bool(cleanup.any()):
            self._poisoned = True
            self._poison_reason = (
                "fault/orphan settlement retired; launch remains HOLD"
            )
        return result

    def abort_prepared_settle_retire(
        self,
        prepared_retire: PreparedPhysicalSettleRetire,
    ) -> None:
        """Abort both unarmed after-images; retain the postphysics authority."""

        self._require_operable()
        image = self._active_physical_retire_image
        if (
            type(prepared_retire) is not PreparedPhysicalSettleRetire
            or prepared_retire is not self._active_physical_retire_prepare
            or prepared_retire._owner_identity is not self._owner_identity
            or prepared_retire._token is not _PHYSICAL_RETIRE_PREPARE_TOKEN
            or image is None
            or self._active_physical_retire_arm is not None
            or self._active_physical_retire_commit is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical settle-retire abort token is stale, foreign, or armed"
            )
        try:
            if image.scene_handle is not None:
                self._abort_scene_write(image.scene_handle)
            self._r06_owner.abort_physical_retire(
                prepared_retire._r06_prepared_retire
            )
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "physical/R06 retirement abort failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        self._active_physical_retire_prepare = None
        self._active_physical_retire_image = None

    def retire_post_physics_to_r06(
        self,
        postphysics_result: object,
    ) -> PhysicalRetireDeviceResult:
        """Close one postphysics lease, including the exact empty transaction."""

        self._require_operable()
        if postphysics_result is not self._active_r06_ack:
            raise PhysicalFlightDeviceError(
                "retirement requires the exact retained postphysics result"
            )
        owner = self._r06_owner
        if owner is None:
            raise PhysicalFlightDeviceError("physical R06 owner is not paired")
        r06_prepared: object | None = None
        physical_prepared: PreparedPhysicalSettleRetire | None = None
        try:
            r06_prepared = owner.prepare_physical_retire(
                postphysics_result,
                postphysics_result.settled_mask,
            )
            physical_prepared = self.prepare_settle_retire(r06_prepared)
            r06_armed = owner.arm_physical_retire(
                r06_prepared,
                physical_prepared,
            )
            physical_armed = self.arm_prevalidated_settle_retire(
                physical_prepared,
                r06_armed,
            )
            physical_committed = self.commit_prevalidated_settle_retire(
                physical_armed
            )
            actual = owner.commit_prevalidated_physical_retire(r06_armed)
            return self.complete_prevalidated_settle_retire(
                physical_committed,
                actual,
            )
        except BaseException:
            if (
                physical_prepared is not None
                and self._active_physical_retire_arm is None
            ):
                try:
                    self.abort_prepared_settle_retire(physical_prepared)
                except BaseException:
                    pass
            elif r06_prepared is not None and physical_prepared is None:
                try:
                    owner.abort_physical_retire(r06_prepared)
                except BaseException:
                    pass
            try:
                owner.poison_global_reveal_epoch(
                    "postphysics physical/R06 retirement transaction failed"
                )
            except BaseException:
                pass
            self.poison_global_reveal_epoch(
                "postphysics physical/R06 retirement transaction failed"
            )
            raise

    def settle_retire(self, *_args: object, **_kwargs: object) -> None:
        """Tombstone the caller-selected legacy retirement ABI."""

        raise PhysicalFlightDeviceError(
            "settle_retire is tombstoned; use exact postphysics two-phase retirement"
        )

    def _selected_reset_r06(self) -> object:
        owner = self._r06_owner
        if owner is None or self._r06_selected_reset_park_token_authority is None:
            raise PhysicalFlightDeviceError(
                "physical selected reset requires the exact bound R06 authority"
            )
        return owner

    def _require_selected_reset_idle(self) -> None:
        if (
            self._active_prepare is not None
            or self._active_armed is not None
            or self._active_child_terminal is not None
            or self._active_global_reveal_epoch_image is not None
            or self._active_reveal_boundary_row is not None
            or self._active_postphysics is not None
            or self._action_epoch_substep_pair is not None
            or self._active_r06_ack is not None
            or self._active_physical_retire_prepare is not None
            or self._active_physical_retire_arm is not None
            or self._active_physical_retire_commit is not None
            or self._active_physical_global_drain is not None
        ):
            raise PhysicalFlightDeviceError(
                "selected reset cannot cross a reveal/postphysics/retire/checkpoint lease"
            )
        self._require_selected_contact_reward_cycle_closed(
            label="selected reset"
        )

    def stage_selected_true_reset(
        self,
        r06_prepared_reset: object,
    ) -> StagedPhysicalSelectedTrueReset:
        """Retain sole R06 selection and its exact device-R05 prepare."""

        self._require_operable_invariants()
        self._require_selected_reset_idle()
        if self._active_selected_reset_stage is not None:
            raise PhysicalFlightDeviceError(
                "one physical selected reset is already active"
            )
        if self._selected_reset_completion_token is not None:
            raise PhysicalFlightDeviceError(
                "top owner has not consumed the previous physical reset ACK"
            )
        r06 = self._selected_reset_r06()
        device_r05_owner = self._device_r05_reset_owner
        if (
            type(device_r05_owner) is not _r05_device.DeviceR05Owner
            or not callable(self._device_r05_prepared_reset_validator)
            or not callable(self._device_r05_receipt_validator)
        ):
            raise PhysicalFlightDeviceError(
                "physical selected reset requires the exact bound device-R05 owner"
            )
        try:
            prepared = r06.require_owned_selected_reset_prepare(
                r06_prepared_reset,
                expected_device_r05_owner=device_r05_owner,
            )
            capability = r06.selected_reset_mask_capability(prepared)
            view = r06.require_owned_selected_reset_mask_capability(
                capability,
                expected_prepared_reset=prepared,
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "R06 selected-reset prepare authority differs"
            ) from exc
        view_type = getattr(
            sys.modules.get(type(r06).__module__),
            "LandingOutcomeSelectedResetMaskView",
            None,
        )
        device_r05_prepared = getattr(
            view, "device_r05_prepared_true_reset", None
        )
        device_mask = getattr(view, "device_mask", None)
        reset_generation_before = getattr(view, "generation_before", None)
        reset_generation_after = getattr(view, "generation_after", None)
        if (
            prepared is not r06_prepared_reset
            or view_type is None
            or type(view) is not view_type
            or getattr(view, "prepared_reset", None) is not prepared
            or getattr(view, "mask_capability", None) is not capability
            or getattr(view, "device_r05_owner", None)
            is not device_r05_owner
            or type(device_r05_prepared)
            is not _r05_device.DeviceR05PreparedTrueReset
            or not isinstance(device_mask, torch.Tensor)
            or tuple(device_mask.shape) != (self.num_envs,)
            or device_mask.dtype != torch.bool
            or device_mask.device != self.device
            or not isinstance(reset_generation_before, torch.Tensor)
            or tuple(reset_generation_before.shape) != (self.num_envs,)
            or reset_generation_before.dtype != torch.int64
            or reset_generation_before.device != self.device
            or not isinstance(reset_generation_after, torch.Tensor)
            or tuple(reset_generation_after.shape) != (self.num_envs,)
            or reset_generation_after.dtype != torch.int64
            or reset_generation_after.device != self.device
        ):
            raise PhysicalFlightDeviceError(
                "R06 selected-reset mask capability ABI differs"
            )
        if self._diagnostic_n2_no_save:
            r06_module = sys.modules.get(type(r06).__module__)
            r06_selected_park_authority_type = (
                None
                if r06_module is None
                else getattr(
                    r06_module,
                    "LandingOutcomeSelectedResetPhysicalParkTokenAuthority",
                    None,
                )
            )
            r06_selected_park_authority = getattr(
                r06,
                "_selected_reset_physical_park_token_authority",
                None,
            )
            if (
                getattr(r06, "_diagnostic_n2_no_save", None) is not True
                or getattr(r06, "_diagnostic_n2_construction_record", None)
                is not None
                or self._r06_owner is not r06
                or getattr(r06, "_device_r05_reset_owner", None)
                is not device_r05_owner
                or self._action_epoch_device_r05_owner
                is not device_r05_owner
                or getattr(r06, "_action_ball_full_mdp_epoch_owner", None)
                is not self._action_epoch_owner
                or getattr(device_r05_owner, "_diagnostic_physical_owner", None)
                is not self
                or getattr(device_r05_owner, "_diagnostic_epoch_owner", None)
                is not self._action_epoch_owner
                or self._action_epoch_owner is None
                or getattr(
                    self.scene_port, "_action_epoch_physical_owner", None
                )
                is not self
                or getattr(self.scene_port, "_action_epoch_owner", None)
                is not self._action_epoch_owner
                or r06_selected_park_authority_type is None
                or type(r06_selected_park_authority)
                is not r06_selected_park_authority_type
                or r06_selected_park_authority
                is not self._r06_selected_reset_park_token_authority
                or getattr(r06_selected_park_authority, "physical_owner", None)
                is not self
            ):
                raise PhysicalFlightDeviceError(
                    "diagnostic selected reset requires the exact same "
                    "R06/D05/Physical/ActionEpoch scene graph"
                )
        selected_env_mask = device_mask.detach().clone()
        generation_contract_fault = (
            selected_env_mask
            & (
                (reset_generation_before != self._device_reset_generation)
                | (
                    reset_generation_before
                    == torch.iinfo(torch.int64).max
                )
                | (
                    reset_generation_after
                    != torch.where(
                        reset_generation_before
                        == torch.iinfo(torch.int64).max,
                        reset_generation_before,
                        reset_generation_before + 1,
                    )
                )
            )
        )
        staged = object.__new__(StagedPhysicalSelectedTrueReset)
        record = _PhysicalSelectedResetStageRecord(
            capability=staged,
            r06_prepared_reset=prepared,
            r06_mask_capability=capability,
            device_r05_prepared_true_reset=device_r05_prepared,
            device_r05_prepared_projection=view,
            selected_env_mask=selected_env_mask,
            reset_generation_before=reset_generation_before.detach().clone(),
            reset_generation_after=reset_generation_after.detach().clone(),
            generation_contract_fault=(
                generation_contract_fault.detach().clone()
            ),
        )
        self._active_selected_reset_stage = staged
        self._active_selected_reset_stage_record = record
        return staged

    def _owned_selected_reset_stage(
        self, value: object
    ) -> StagedPhysicalSelectedTrueReset:
        record = self._active_selected_reset_stage_record
        if (
            type(value) is not StagedPhysicalSelectedTrueReset
            or value is not self._active_selected_reset_stage
            or record is None
            or record.capability is not value
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset stage token is stale or foreign"
            )
        return value

    def finalize_selected_true_reset(
        self,
        staged_reset: StagedPhysicalSelectedTrueReset,
    ) -> FinalizedPhysicalSelectedTrueReset:
        """Preflight one full-grid masked park with no per-K Python mutation."""

        self._require_operable(
            allow_selected_reset=True,
            diagnostic_selected_reset_capability=(
                self._active_selected_reset_stage
            ),
        )
        self._require_selected_reset_idle()
        staged = self._owned_selected_reset_stage(staged_reset)
        stage_record = self._active_selected_reset_stage_record
        if stage_record is None or stage_record.capability is not staged:
            raise PhysicalFlightDeviceError(
                "physical selected-reset private stage record was lost"
            )
        if self._active_selected_reset_finalize is not None:
            raise PhysicalFlightDeviceError(
                "physical selected reset was already finalized"
            )
        selected_env_mask = stage_record.selected_env_mask
        selected_slot_mask = selected_env_mask[:, None].expand(
            self.num_envs, self.flight_capacity
        )
        action_epoch_direct_before = self._clone_action_epoch_direct_state()
        action_epoch_direct_after = self._selected_reset_action_epoch_direct_state(
            selected_env_mask
        )
        scene_before = _tensor(
            self.scene_port.read_state_env(),
            label="selected-reset scene state",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        park = self._park_state_template
        scene_after = torch.where(
            selected_slot_mask.unsqueeze(-1), park, scene_before
        )
        lifecycle_after = torch.where(
            selected_slot_mask,
            torch.zeros_like(self._lifecycle),
            self._lifecycle,
        )
        generation_after = torch.where(
            selected_slot_mask,
            torch.full_like(self._generation, -1),
            self._generation,
        )
        zero_digest = torch.zeros_like(self._outcome_sha)
        outcome_after = torch.where(
            selected_slot_mask.unsqueeze(-1), zero_digest, self._outcome_sha
        )
        install_after = torch.where(
            selected_slot_mask.unsqueeze(-1), zero_digest, self._install_sha
        )
        installed_state_after = torch.where(
            selected_slot_mask.unsqueeze(-1),
            zero_digest,
            self._installed_state_sha,
        )
        reveal_after = torch.where(
            selected_slot_mask,
            torch.full_like(self._reveal_step, -1),
            self._reveal_step,
        )
        ordinal_after = torch.where(
            selected_slot_mask,
            torch.full_like(self._observation_ordinal, -1),
            self._observation_ordinal,
        )
        previous_after = torch.where(
            selected_slot_mask.unsqueeze(-1),
            park[..., :3],
            self._previous_ball_center,
        )
        parked_after = self._parked | selected_slot_mask
        published_after = self._published & ~selected_slot_mask
        slot_version_after = self._slot_version + selected_slot_mask.to(
            torch.int64
        )
        # A true reset may safely park/clear live task state, but it is not an
        # audit authority and must never erase an unreported device fault.
        # Newly detected generation faults therefore join monotonically with
        # the retained fault tensor and remain visible to the sole PPO drain.
        device_fault_after = self._device_fault | (
            selected_slot_mask
            & stage_record.generation_contract_fault[:, None]
        )
        reset_generation_after = torch.where(
            selected_env_mask,
            torch.where(
                stage_record.reset_generation_before
                == torch.iinfo(torch.int64).max,
                stage_record.reset_generation_before,
                stage_record.reset_generation_after,
            ),
            self._device_reset_generation,
        )
        scene_handle = self._prepare_action_epoch_scene_write(
            kind="retire",
            state_env_f32=scene_after,
            selected_mask=selected_slot_mask,
        )
        finalized = object.__new__(FinalizedPhysicalSelectedTrueReset)
        self._active_selected_reset_finalize = finalized
        self._active_selected_reset_image = _PhysicalSelectedResetImage(
            selected_env_mask=selected_env_mask,
            selected_slot_mask=selected_slot_mask,
            scene_handle=scene_handle,
            scene_after=scene_after,
            lifecycle_after=lifecycle_after,
            generation_after=generation_after,
            outcome_sha_after=outcome_after,
            install_sha_after=install_after,
            installed_state_sha_after=installed_state_after,
            reveal_step_after=reveal_after,
            observation_ordinal_after=ordinal_after,
            previous_ball_center_after=previous_after,
            parked_after=parked_after,
            published_after=published_after,
            slot_version_after=slot_version_after,
            device_fault_after=device_fault_after,
            reset_generation_after=reset_generation_after,
            action_epoch_direct_state_after=action_epoch_direct_after,
            stale_witness=_PhysicalSelectedResetStaleWitness(
                scene_state=scene_before.detach().clone(),
                lifecycle=self._lifecycle.detach().clone(),
                generation=self._generation.detach().clone(),
                outcome_sha=self._outcome_sha.detach().clone(),
                install_sha=self._install_sha.detach().clone(),
                installed_state_sha=self._installed_state_sha.detach().clone(),
                reveal_step=self._reveal_step.detach().clone(),
                observation_ordinal=self._observation_ordinal.detach().clone(),
                previous_ball_center=self._previous_ball_center.detach().clone(),
                parked=self._parked.detach().clone(),
                published=self._published.detach().clone(),
                slot_version=self._slot_version.detach().clone(),
                device_fault=self._device_fault.detach().clone(),
                reset_generation=(
                    self._device_reset_generation.detach().clone()
                ),
                action_epoch_direct_state=action_epoch_direct_before,
                owner_mutation_version=self._mutation_version,
                next_prepare_nonce=self._next_prepare_nonce,
            ),
        )
        return finalized

    def abort_selected_true_reset(
        self,
        selected_reset: StagedPhysicalSelectedTrueReset
        | FinalizedPhysicalSelectedTrueReset,
    ) -> None:
        """Release only a prearm reset whose scene writer never ran."""

        self._require_operable(
            allow_selected_reset=True,
            diagnostic_selected_reset_capability=(
                self._active_selected_reset_stage
            ),
        )
        staged = self._active_selected_reset_stage
        finalized = self._active_selected_reset_finalize
        image = self._active_selected_reset_image
        if (
            staged is None
            or self._active_selected_reset_arm is not None
            or self._active_selected_reset_commit is not None
            or (
                selected_reset is not staged
                and selected_reset is not finalized
            )
        ):
            raise PhysicalFlightDeviceError(
                "selected-reset abort is stale, foreign, or crossed prearm"
            )
        try:
            if finalized is not None:
                if image is None:
                    raise PhysicalFlightDeviceError(
                        "finalized selected reset lacks its scene preflight"
                    )
                self._abort_scene_write(image.scene_handle)
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = (
                "physical selected-reset prearm abort failed"
            )
            raise PhysicalFlightOwnerPoisonedError(
                self._poison_reason
            ) from exc
        self._active_selected_reset_stage = None
        self._active_selected_reset_stage_record = None
        self._active_selected_reset_finalize = None
        self._active_selected_reset_image = None

    def poison_selected_reset(self, reason: str) -> None:
        """Idempotent coordinator broadcast after selected-reset prearm."""

        if self._poisoned:
            return
        self._poisoned = True
        self._poison_reason = (
            reason.strip()
            if type(reason) is str and reason.strip()
            else "selected-reset coordinator poisoned the physical leaf"
        )

    def require_owned_selected_reset_prepared_token(
        self,
        physical_prepared_token: object,
        *,
        expected_r06_prepared_reset: object,
    ) -> object:
        """R06 callback over the exact finalized physical reset token."""

        finalized = self._active_selected_reset_finalize
        staged = self._active_selected_reset_stage
        stage_record = self._active_selected_reset_stage_record
        if (
            type(physical_prepared_token)
            is not FinalizedPhysicalSelectedTrueReset
            or physical_prepared_token is not finalized
            or staged is None
            or stage_record is None
            or stage_record.capability is not staged
            or stage_record.r06_prepared_reset
            is not expected_r06_prepared_reset
            or self._active_selected_reset_image is None
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset prepared token is stale or foreign"
            )
        module = sys.modules.get(type(expected_r06_prepared_reset).__module__)
        claim_type = None if module is None else getattr(
            module,
            "LandingOutcomeSelectedResetPhysicalParkPreparedTokenClaim",
            None,
        )
        if claim_type is None:
            raise PhysicalFlightDeviceError(
                "R06 selected-reset prepared claim type is unavailable"
            )
        return claim_type(
            r06_prepared_reset=expected_r06_prepared_reset,
            r06_mask_capability=stage_record.r06_mask_capability,
            physical_prepared_token=physical_prepared_token,
        )

    def _selected_reset_stale_mismatch(self) -> torch.Tensor:
        image = self._active_selected_reset_image
        if image is None:
            raise PhysicalFlightDeviceError(
                "physical selected-reset image is unavailable"
            )
        witness = image.stale_witness
        current_scene = _tensor(
            self.scene_port.read_state_env(),
            label="selected-reset current scene",
            shape=(self.num_envs, self.flight_capacity, STATE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        mismatch = (
            ~_device_bitwise_equal(current_scene, witness.scene_state)
            | ~torch.eq(self._lifecycle, witness.lifecycle).all()
            | ~torch.eq(self._generation, witness.generation).all()
            | ~torch.eq(self._outcome_sha, witness.outcome_sha).all()
            | ~torch.eq(self._install_sha, witness.install_sha).all()
            | ~torch.eq(
                self._installed_state_sha, witness.installed_state_sha
            ).all()
            | ~torch.eq(self._reveal_step, witness.reveal_step).all()
            | ~torch.eq(
                self._observation_ordinal, witness.observation_ordinal
            ).all()
            | ~_device_bitwise_equal(
                self._previous_ball_center, witness.previous_ball_center
            )
            | ~torch.eq(self._parked, witness.parked).all()
            | ~torch.eq(self._published, witness.published).all()
            | ~torch.eq(self._slot_version, witness.slot_version).all()
            | ~torch.eq(self._device_fault, witness.device_fault).all()
            | ~torch.eq(
                self._device_reset_generation,
                witness.reset_generation,
            ).all()
            | self._action_epoch_direct_state_mismatch(
                witness.action_epoch_direct_state
            )
        )
        if (
            self._mutation_version != witness.owner_mutation_version
            or self._next_prepare_nonce != witness.next_prepare_nonce
        ):
            mismatch = torch.ones((), dtype=torch.bool, device=self.device)
        return mismatch

    def prearm_selected_true_reset(
        self,
        finalized_reset: FinalizedPhysicalSelectedTrueReset,
        r06_armed_reset: object,
    ) -> ArmedPhysicalSelectedTrueReset:
        """Cross-bind R06's armed after-image; no scene or owner mutation."""

        self._require_operable(
            allow_selected_reset=True,
            diagnostic_selected_reset_capability=(
                self._active_selected_reset_stage
            ),
        )
        image = self._active_selected_reset_image
        if (
            type(finalized_reset) is not FinalizedPhysicalSelectedTrueReset
            or finalized_reset is not self._active_selected_reset_finalize
            or image is None
            or self._active_selected_reset_arm is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset finalized token is stale or foreign"
            )
        r06 = self._selected_reset_r06()
        try:
            owned_arm = r06.require_owned_selected_reset_arm(
                r06_armed_reset, finalized_reset
            )
        except Exception as exc:
            raise PhysicalFlightDeviceError(
                "R06 selected-reset arm authority differs"
            ) from exc
        if owned_arm is not r06_armed_reset:
            raise PhysicalFlightDeviceError(
                "R06 selected-reset arm identity differs"
            )
        mismatch = self._selected_reset_stale_mismatch()
        self._active_selected_reset_image = replace(
            image,
            device_fault_after=(
                image.device_fault_after
                | self._device_fault
                | (image.selected_slot_mask & mismatch)
            ),
            r06_armed_reset=r06_armed_reset,
        )
        armed = object.__new__(ArmedPhysicalSelectedTrueReset)
        self._active_selected_reset_arm = armed
        return armed

    def commit_prevalidated_selected_true_reset(
        self,
        armed_reset: ArmedPhysicalSelectedTrueReset,
    ) -> PhysicalSelectedTrueResetParkCommitToken:
        """Park physical rows first; any writer failure poisons without rollback."""

        self._require_operable(
            allow_selected_reset=True,
            diagnostic_selected_reset_capability=(
                self._active_selected_reset_stage
            ),
        )
        image = self._active_selected_reset_image
        if (
            type(armed_reset) is not ArmedPhysicalSelectedTrueReset
            or armed_reset is not self._active_selected_reset_arm
            or image is None
            or image.r06_armed_reset is None
            or self._active_selected_reset_commit is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset arm is stale or foreign"
            )
        try:
            scene_receipt = self._apply_scene_write(image.scene_handle)
            self._lifecycle.copy_(image.lifecycle_after)
            self._generation.copy_(image.generation_after)
            self._outcome_sha.copy_(image.outcome_sha_after)
            self._install_sha.copy_(image.install_sha_after)
            self._installed_state_sha.copy_(image.installed_state_sha_after)
            self._reveal_step.copy_(image.reveal_step_after)
            self._observation_ordinal.copy_(image.observation_ordinal_after)
            self._previous_ball_center.copy_(
                image.previous_ball_center_after
            )
            self._parked.copy_(image.parked_after)
            self._published.copy_(image.published_after)
            self._slot_version.copy_(image.slot_version_after)
            self._device_fault.copy_(image.device_fault_after)
            self._device_reset_generation.copy_(
                image.reset_generation_after
            )
            direct_after = image.action_epoch_direct_state_after
            self._action_epoch_pending_launch = self._clone_action_epoch_pending(
                direct_after.pending_launch
            )
            self._action_epoch_active_flight_slot.copy_(
                direct_after.active_flight_slot
            )
            for field in fields(_row_identity.ActionEpochShotKey):
                getattr(self._action_epoch_flight_shot_key, field.name).copy_(
                    getattr(direct_after.flight_shot_key, field.name)
                )
            self._action_epoch_flight_publication_ordinal.copy_(
                direct_after.flight_publication_ordinal
            )
            self._action_epoch_host_activity_control_step = None
            self._action_epoch_host_activity_has_work = True
            self._host_reset_generation_projection_current = False
            self._advance_owner_mutation_version()
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = (
                "physical selected-reset park publication failed; rollback is untrusted"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        token = object.__new__(PhysicalSelectedTrueResetParkCommitToken)
        self._active_selected_reset_image = replace(
            image,
            scene_apply_receipt=scene_receipt,
        )
        self._active_selected_reset_commit = token
        return token

    def require_owned_selected_reset_commit(
        self,
        physical_commit_token: object,
    ) -> PhysicalSelectedTrueResetParkCommitToken:
        """Repeatably validate the active physical-first park for the top owner."""

        token = self._active_selected_reset_commit
        if (
            type(physical_commit_token)
            is not PhysicalSelectedTrueResetParkCommitToken
            or physical_commit_token is not token
            or self._active_selected_reset_arm is None
            or self._active_selected_reset_finalize is None
            or self._active_selected_reset_image is None
            or self._active_selected_reset_image.scene_apply_receipt is None
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset park commit is stale or foreign"
            )
        return token

    def require_committed_selected_reset_park_token(
        self,
        physical_commit_token: object,
        *,
        expected_r06_armed_reset: object,
    ) -> object:
        """R06 callback proving the exact physical-first park."""

        token = self._active_selected_reset_commit
        finalized = self._active_selected_reset_finalize
        if (
            type(physical_commit_token)
            is not PhysicalSelectedTrueResetParkCommitToken
            or physical_commit_token is not token
            or finalized is None
            or self._active_selected_reset_arm is None
            or self._active_selected_reset_image is None
            or self._active_selected_reset_image.r06_armed_reset
            is not expected_r06_armed_reset
            or self._active_selected_reset_image.scene_apply_receipt is None
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset park commit is stale or foreign"
            )
        module = sys.modules.get(type(expected_r06_armed_reset).__module__)
        claim_type = None if module is None else getattr(
            module,
            "LandingOutcomeSelectedResetPhysicalParkCommitTokenClaim",
            None,
        )
        if claim_type is None:
            raise PhysicalFlightDeviceError(
                "R06 selected-reset park commit claim type is unavailable"
            )
        return claim_type(
            r06_armed_reset=expected_r06_armed_reset,
            physical_prepared_token=finalized,
            physical_commit_token=physical_commit_token,
        )

    def acknowledge_r06_selected_reset_commit(
        self,
        physical_commit_token: PhysicalSelectedTrueResetParkCommitToken,
        r06_commit_token: object,
    ) -> None:
        """Retain R06-last identity; lease stays closed pending R05-last."""

        self._require_operable(
            allow_selected_reset=True,
            diagnostic_selected_reset_capability=(
                self._active_selected_reset_stage
            ),
        )
        image = self._active_selected_reset_image
        if (
            physical_commit_token is not self._active_selected_reset_commit
            or image is None
            or image.r06_commit_token is not None
        ):
            self._poisoned = True
            self._poison_reason = (
                "R06 selected-reset acknowledgement differs after physical park"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        r06 = self._selected_reset_r06()
        try:
            stage_record = self._active_selected_reset_stage_record
            if (
                stage_record is None
                or stage_record.capability
                is not self._active_selected_reset_stage
            ):
                raise PhysicalFlightDeviceError(
                    "physical selected-reset stage was lost after park"
                )
            owned = r06.require_owned_selected_reset_physical_commit(
                r06_commit_token,
                expected_prepared_true_reset=(
                    stage_record.device_r05_prepared_true_reset
                ),
                expected_device_r05_owner=self._device_r05_reset_owner,
            )
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = (
                "R06 selected-reset commit failed after physical park"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        if owned is not r06_commit_token:
            self._poisoned = True
            self._poison_reason = (
                "R06 selected-reset commit identity differs after physical park"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        self._active_selected_reset_image = replace(
            image, r06_commit_token=r06_commit_token
        )

    def complete_selected_true_reset_after_r05(
        self,
        physical_commit_token: PhysicalSelectedTrueResetParkCommitToken,
        r06_commit_token: object,
        device_r05_true_reset_receipt: _r05_device.DeviceR05TrueResetReceipt,
    ) -> PhysicalSelectedTrueResetCompletionToken:
        """ACK device-R05-last and return a single-use opaque child token."""

        self._require_operable(
            allow_selected_reset=True,
            diagnostic_selected_reset_capability=(
                self._active_selected_reset_stage
            ),
        )
        image = self._active_selected_reset_image
        stage_record = self._active_selected_reset_stage_record
        if (
            physical_commit_token is not self._active_selected_reset_commit
            or image is None
            or image.r06_commit_token is not r06_commit_token
            or self._active_selected_reset_stage is None
            or stage_record is None
            or stage_record.capability is not self._active_selected_reset_stage
        ):
            self._poisoned = True
            self._poison_reason = (
                "physical selected-reset R05 completion differs after park"
            )
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        r05_owner = self._device_r05_reset_owner
        validator = self._device_r05_receipt_validator
        if not callable(validator):
            raise PhysicalFlightDeviceError(
                "device-R05 exact true-reset receipt authority is PIN_PENDING/HOLD"
            )
        try:
            owned = validator(
                device_r05_true_reset_receipt,
                expected_prepared_true_reset=(
                    stage_record.device_r05_prepared_true_reset
                ),
            )
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "R05 true-reset exact acknowledgement failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        if (
            owned is not device_r05_true_reset_receipt
            or type(owned) is not _r05_device.DeviceR05TrueResetReceipt
        ):
            self._poisoned = True
            self._poison_reason = "R05 true-reset receipt identity differs"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason)
        token = object.__new__(PhysicalSelectedTrueResetCompletionToken)
        completion_record = _PhysicalSelectedResetCompletionRecord(
            capability=token,
            park_commit_token=physical_commit_token,
            r06_commit_token=r06_commit_token,
            device_r05_true_reset_receipt=owned,
        )
        self._active_selected_reset_stage = None
        self._active_selected_reset_stage_record = None
        self._active_selected_reset_finalize = None
        self._active_selected_reset_arm = None
        self._active_selected_reset_commit = None
        self._active_selected_reset_image = None
        self._selected_reset_completion_token = token
        self._selected_reset_completion_record = completion_record
        return token

    def require_owned_selected_reset_completion(
        self,
        completion: object,
    ) -> PhysicalSelectedTrueResetCompletionToken:
        """Validate the latest exact opaque child ACK for the top owner."""

        token = self._selected_reset_completion_token
        record = self._selected_reset_completion_record
        if (
            type(completion) is not PhysicalSelectedTrueResetCompletionToken
            or completion is not token
            or record is None
            or record.capability is not completion
        ):
            raise PhysicalFlightDeviceError(
                "physical selected-reset completion is stale or foreign"
            )
        return completion

    def consume_owned_selected_reset_completion(
        self,
        completion: object,
    ) -> PhysicalSelectedTrueResetCompletionToken:
        """Let the top owner consume the opaque physical ACK exactly once."""

        owned = self.require_owned_selected_reset_completion(completion)
        self._selected_reset_completion_token = None
        self._selected_reset_completion_record = None
        return owned

    @staticmethod
    def _physical_global_owner_row_values(
        owner_row: object,
    ) -> dict[str, int]:
        if (
            getattr(owner_row, "owner_kind", None)
            != PHYSICAL_GLOBAL_DRAIN_OWNER_KIND
        ):
            raise PhysicalFlightDeviceError(
                "physical global owner row kind differs"
            )
        values = getattr(owner_row, "values", None)
        if type(values) is not tuple:
            raise PhysicalFlightDeviceError(
                "physical global owner row values differ"
            )
        names: list[str] = []
        result: dict[str, int] = {}
        for item in values:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
            ):
                raise PhysicalFlightDeviceError(
                    "physical global owner row contains non-scalars"
                )
            names.append(item[0])
            result[item[0]] = item[1]
        if tuple(names) != PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES:
            raise PhysicalFlightDeviceError(
                "physical global owner row schema differs"
            )
        return result

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Freeze physical counters for the sole global device-to-host row."""

        self._require_operable()
        self._require_checkpoint_idle()
        if self._physical_global_drain_poisoned:
            raise PhysicalFlightOwnerPoisonedError(
                self._physical_global_drain_poison_reason
                or "physical global PPO drain is poisoned"
            )
        if type(update_index) is not int or update_index < 0:
            raise PhysicalFlightDeviceError(
                "physical global drain update_index differs"
            )
        if (
            type(completed_environment_steps) is not int
            or completed_environment_steps < 0
        ):
            raise PhysicalFlightDeviceError(
                "physical completed_environment_steps differs"
            )
        if (
            update_index <= self._physical_global_drain_last_update_index
            or self._active_physical_global_drain is not None
        ):
            raise PhysicalFlightDeviceError(
                "physical global drain chronology differs"
            )
        if (
            getattr(authority, "owner_kind", None)
            != PHYSICAL_GLOBAL_DRAIN_OWNER_KIND
            or tuple(getattr(authority, "field_names", ()))
            != PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES
            or getattr(authority, "expected_width", None)
            != len(PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES)
        ):
            raise PhysicalFlightDeviceError(
                "physical global drain authority schema differs"
            )
        mint = getattr(authority, "mint_device_pack", None)
        require_owned_ack = getattr(authority, "require_owned_ack", None)
        canonical_drain = sys.modules.get(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        )
        focused_drain = sys.modules.get("action_ball_full_mdp_ppo_drain")
        if canonical_drain is not None:
            drain = canonical_drain
        elif focused_drain is not None:
            drain = focused_drain
        elif __package__:
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
            raise PhysicalFlightDeviceError(
                "physical global drain authority API differs"
            )
        authority_type = self._physical_global_drain_authority_type
        authority_ack_method = self._physical_global_drain_ack_method
        if authority_type is None:
            try:
                drain_source_sha256 = hashlib.sha256(
                    Path(drain.__file__).read_bytes()
                ).hexdigest()
                drain_ack_api_sha256 = (
                    _global_drain_ack_authority_api_sha256(drain)
                )
            except (OSError, TypeError) as exc:
                raise PhysicalFlightDeviceError(
                    "physical global drain authority pins are unavailable"
                ) from exc
            if (
                drain_source_sha256 != PHYSICAL_GLOBAL_DRAIN_SOURCE_SHA256
                or drain_ack_api_sha256
                != PHYSICAL_GLOBAL_DRAIN_ACK_AUTHORITY_API_SHA256
            ):
                raise PhysicalFlightDeviceError(
                    "physical global drain authority source/API differs"
                )
            self._physical_global_drain_authority_type = type(authority)
            self._physical_global_drain_ack_method = type(
                authority
            ).require_owned_ack
        elif (
            type(authority) is not authority_type
            or type(authority).require_owned_ack is not authority_ack_method
        ):
            raise PhysicalFlightDeviceError(
                "physical global drain authority identity changed"
            )
        fault_count = self._device_fault.to(torch.int64).sum().reshape(1)
        invariant_count = (
            (self._slot_version < 0).to(torch.int64).sum()
            + (self._device_reset_generation < 1).to(torch.int64).sum()
            + (self._terminal_resolution_total < 0).to(torch.int64).sum()
            + (self._shared_normal_retire_total < 0).to(torch.int64).sum()
            + (self._physical_only_orphan_park_total < 0)
            .to(torch.int64)
            .sum()
            + self._selected_contact_pending.to(torch.int64).sum()
            + self._selected_contact_ledger_fault.to(torch.int64).sum()
            + torch.full(
                (1,),
                int(self._selected_contact_reward_cycle_open),
                dtype=torch.int64,
                device=self.device,
            ).sum()
            + (self._selected_contact_view_total < 0).to(torch.int64).sum()
            + (self._selected_contact_payment_total < 0).to(torch.int64).sum()
            + (
                self._selected_contact_payment_total
                > self._selected_contact_view_total
            ).to(torch.int64).sum()
        ).reshape(1)
        values = torch.cat(
            (
                self._device_mutation_version,
                fault_count,
                invariant_count,
                self._terminal_resolution_total,
                self._shared_normal_retire_total,
                self._physical_only_orphan_park_total,
                self._shared_normal_retire_key_summaries,
                self._selected_contact_pending.to(torch.int64).sum().reshape(1),
                self._selected_contact_view_total,
                self._selected_contact_payment_total,
                self._selected_contact_ledger_fault.to(torch.int64).sum().reshape(1),
            )
        ).contiguous()
        pack = mint(leaf=self, values=values)
        self._active_physical_global_drain = _PreparedPhysicalGlobalDrain(
            pack=pack,
            authority=authority,
            update_index=update_index,
            completed_environment_steps=completed_environment_steps,
            mutation_version=self._mutation_version,
        )
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        pack: object,
    ) -> None:
        """Release one exact pre-transfer pack without changing facts."""

        active = self._active_physical_global_drain
        if (
            self._physical_global_drain_poisoned
            or active is None
            or pack is not active.pack
        ):
            raise PhysicalFlightDeviceError(
                "physical global drain abort pack is stale or foreign"
            )
        if self._mutation_version != active.mutation_version:
            self.poison_pre_optimizer_ppo_boundary(
                reason="physical global drain changed before abort"
            )
            raise PhysicalFlightOwnerPoisonedError(
                self._physical_global_drain_poison_reason
                or "physical global drain changed before abort"
            )
        self._active_physical_global_drain = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Acknowledge the exact physical row after the sole global transfer."""

        active = self._active_physical_global_drain
        try:
            if (
                self._physical_global_drain_poisoned
                or active is None
                or pack is not active.pack
            ):
                raise PhysicalFlightDeviceError(
                    "physical global drain ACK pack is stale or foreign"
                )
            active.authority.require_owned_ack(
                leaf=self,
                pack=pack,
                receipt=receipt,
                owner_row=owner_row,
            )
            canonical_drain = sys.modules.get(
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_full_mdp_ppo_drain"
            )
            focused_drain = sys.modules.get(
                "action_ball_full_mdp_ppo_drain"
            )
            if canonical_drain is not None:
                drain = canonical_drain
            elif focused_drain is not None:
                drain = focused_drain
            elif __package__:
                from . import action_ball_full_mdp_ppo_drain as drain
            else:
                from whole_body_tracking.tasks.tracking.mdp import (
                    action_ball_full_mdp_ppo_drain as drain,
                )
            owner_rows = getattr(receipt, "owner_rows", None)
            if (
                type(receipt) is not drain.PreOptimizerPpoBoundaryReceipt
                or type(owner_row) is not drain.OwnerDrainRow
                or type(owner_rows) is not tuple
                or sum(row is owner_row for row in owner_rows) != 1
                or getattr(receipt, "update_index", None)
                != active.update_index
                or getattr(receipt, "completed_environment_steps", None)
                != active.completed_environment_steps
                or getattr(receipt, "drain_sequence", None)
                != self._physical_global_drain_sequence + 1
                or getattr(receipt, "device_to_host_transfers", None) != 1
                or getattr(receipt, "acknowledged", None) is not False
            ):
                raise PhysicalFlightDeviceError(
                    "physical global drain receipt boundary differs"
                )
            row = self._physical_global_owner_row_values(owner_row)
            if (
                row["mutation_version"] != active.mutation_version
                or self._mutation_version != active.mutation_version
                or row["fault_count"] != 0
                or row["invariant_count"] != 0
            ):
                raise PhysicalFlightDeviceError(
                    "physical global drain row is faulted or stale"
                )
        except BaseException as exc:
            self.poison_pre_optimizer_ppo_boundary(
                reason=(
                    "physical global drain acknowledgement failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            raise
        self._physical_global_drain_sequence = receipt.drain_sequence
        self._physical_global_drain_last_update_index = active.update_index
        self._physical_global_drain_last_completed_environment_steps = (
            active.completed_environment_steps
        )
        self._physical_global_drain_last_acknowledged_mutation_version = (
            active.mutation_version
        )
        # Retain only the causal writer and the exact receipt object.  The
        # receipt is marked ACKed by the global owner after every leaf callback
        # returns, so R10 cannot consume this join during a partial ACK.
        self._physical_global_drain_authority = active.authority
        self._physical_global_drain_last_acknowledged_receipt = receipt
        self._physical_checkpoint_live_ack = _PhysicalR10LiveMutationAck(
            owner_identity=self._owner_identity,
            authority=active.authority,
            receipt=receipt,
            update_index=active.update_index,
            completed_environment_steps=active.completed_environment_steps,
            drain_sequence=receipt.drain_sequence,
            mutation_version=active.mutation_version,
            token=_PHYSICAL_R10_LIVE_ACK_TOKEN,
        )
        self._physical_checkpoint_live_join_required = False
        self._physical_checkpoint_last_live_projection = None
        self._physical_checkpoint_last_live_boundary = None
        self._physical_checkpoint_last_live_receipt = None
        self._physical_checkpoint_requires_global_drain_ack = False
        self._active_physical_global_drain = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason: object) -> None:
        """Idempotently fail-stop physical after an irreversible drain failure."""

        if self._physical_global_drain_poisoned:
            return
        self._physical_global_drain_poisoned = True
        self._physical_global_drain_poison_reason = (
            reason
            if type(reason) is str and bool(reason.strip())
            else "unspecified physical global PPO drain failure"
        )
        self.poison_global_reveal_epoch(
            self._physical_global_drain_poison_reason
        )

    def _legacy_settle_retire_tombstone(
        self, value: PhysicalSettleRetireInput
    ) -> PhysicalRetireDeviceResult:
        """Unreachable retained text pending final source compaction."""

        raise PhysicalFlightDeviceError("legacy settle-retire path is disabled")

        self._require_operable()
        self._require_cpu_reference()
        if self._active_postphysics is not None:
            raise PhysicalFlightDeviceError(
                "retirement cannot cross postphysics acknowledgement"
            )
        if self._active_prepare is not None:
            raise PhysicalFlightDeviceError(
                "retirement cannot cross an active physical prepare lease"
            )
        if type(value) is not PhysicalSettleRetireInput:
            raise PhysicalFlightDeviceError("settle-retire input type differs")
        ack = value.r06_ack
        if (
            type(ack) is not AcknowledgedR06PhysicalSnapshot
            or ack._token is not _R06_ACK_TOKEN
            or ack._owner_identity is not self._owner_identity
            or self._active_r06_ack is not ack
            or self._active_r06_ack_image is None
        ):
            raise PhysicalFlightDeviceError(
                "R06 settlement acknowledgement is stale or foreign"
            )
        ack_image = self._active_r06_ack_image
        settlement_authority = ack_image.settlement_authority
        terminal_fault = ack_image.terminal_fault_mask.clone()
        cleanup_only = ack_image.cleanup_only_mask.clone()
        physical_orphan = ack_image.physical_orphan_mask.clone()
        r06_orphan = ack_image.r06_orphan_mask.clone()
        r06_settled = ack_image.r06_settled_mask.clone()
        r06_owner = ack_image.r06_owner
        if r06_owner is None:
            raise PhysicalFlightDeviceError(
                "restored R06 settlement authority must be rebound before retire"
            )
        _require_final_r06_type(
            r06_owner,
            expected_name="ActionBallLandingOutcomeDeviceCoordinator",
        )
        shape = (self.num_envs, self.flight_capacity)
        mask = _tensor(
            value.retire_mask,
            label="retire_mask",
            shape=shape,
            dtype=torch.bool,
            device=self.device,
        )
        if not torch.equal(mask, ack_image.settled_mask):
            raise PhysicalFlightDeviceError(
                "retire mask must equal the owner-minted settlement set"
            )
        r06_state = _tensor(
            ack_image.flight_state,
            label="R06 retire state",
            shape=shape,
            dtype=torch.int8,
            device=self.device,
        )
        r06_key = _tensor(
            ack_image.full_key_sha256,
            label="R06 retire key",
            shape=shape + (TOKEN_BYTES,),
            dtype=torch.uint8,
            device=self.device,
        )
        r06_generation = _tensor(
            ack_image.ball_generation,
            label="R06 retire generation",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        r06_mailbox_slot = _tensor(
            ack_image.mailbox_slot,
            label="R06 retire mailbox slot",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        _tensor(
            ack_image.observation_ordinal,
            label="R06 retire observation ordinal",
            shape=shape,
            dtype=torch.int64,
            device=self.device,
        )
        after_root = ack_image.snapshot_root_sha256
        physical_live = self._published & ~self._parked
        physical_already_parked = ~self._published & self._parked
        r06_retained = r06_state == R06_FLIGHT_SETTLED_RETAINED
        joined_or_cleanup_live = (
            r06_retained
            & physical_live
            & (
                (
                    torch.eq(r06_key, self._outcome_sha).all(dim=-1)
                    & (r06_generation == self._generation)
                )
                | (cleanup_only & ~physical_orphan)
            )
        )
        physical_only_cleanup = (
            physical_orphan
            & (r06_state == R06_FLIGHT_EMPTY)
            & physical_live
        )
        r06_only_cleanup = (
            r06_orphan
            & r06_retained
            & physical_already_parked
        )
        valid = (
            joined_or_cleanup_live
            | physical_only_cleanup
            | r06_only_cleanup
        )
        if (
            bool((physical_orphan & r06_settled).any())
            or bool((r06_orphan & ~r06_settled).any())
            or bool((physical_orphan & r06_orphan).any())
        ):
            raise PhysicalFlightDeviceError(
                "owner-private orphan retirement masks are inconsistent"
            )
        accepted = mask & valid
        rejected = mask & ~valid
        if bool(rejected.any()):
            raise PhysicalFlightDeviceError(
                "retire mask contains a row outside the acknowledged settlement"
            )
        portable_receipt: Optional[_flight.PhysicalRetireReceipt] = None
        before_version = self._mutation_version
        host_rows: list[_flight.PhysicalRetireRow] = []
        after_slots = list(self._host_slots)
        accepted_indices = [
            (int(index[0]), int(index[1]))
            for index in accepted.nonzero(as_tuple=False).tolist()
        ]
        physical_park_mask = accepted & ~r06_orphan
        r06_retire_mask = accepted & ~physical_orphan
        expected_authority_rows: list[dict[str, object]] = []
        for env_id, slot in accepted_indices:
            pre = self._host_slots[self._slot_offset(env_id, slot)]
            if not bool(cleanup_only[env_id, slot]):
                expected_authority_rows.append(
                    {
                        "env_id": env_id,
                        "slot_index": slot,
                        "outcome_key_sha256": pre.outcome_key_sha256,
                        "ball_generation": pre.ball_generation,
                    }
                )
            if bool(r06_orphan[env_id, slot]):
                continue
            post = replace(
                pre,
                lifecycle=_flight.SLOT_RETIRED,
                mutation_version=pre.mutation_version + 1,
                physically_parked=True,
                published_to_runtime=False,
            )
            after_slots[self._slot_offset(env_id, slot)] = post
            if (
                settlement_authority is not None
                and not bool(cleanup_only[env_id, slot])
            ):
                host_rows.append(
                    _flight.PhysicalRetireRow(
                        env_id=env_id,
                        slot_index=slot,
                        outcome_key=pre.outcome_key,
                        outcome_key_sha256=pre.outcome_key_sha256,
                        settlement_authority=settlement_authority,
                        pre_slot_snapshot=pre,
                        post_slot_snapshot=post,
                    )
                )
        if settlement_authority is not None:
            authority_mapping = settlement_authority.decoded_mapping
            if (
                frozenset(authority_mapping)
                != {
                    "schema_version",
                    "kind",
                    "mailbox_lifecycle",
                    "r06_owner_mutation_version",
                    "r06_after_root_sha256",
                    "physical_retire_rows",
                    "canonical_sha256",
                }
                or authority_mapping.get("mailbox_lifecycle")
                != "SETTLED_UNPAID"
                or authority_mapping.get("r06_owner_mutation_version")
                != ack_image.owner_mutation_version
                or authority_mapping.get("r06_after_root_sha256") != after_root
                or authority_mapping.get("physical_retire_rows")
                != expected_authority_rows
            ):
                raise PhysicalFlightDeviceError(
                    "settlement authority differs from the exact R06 retire join"
                )
        elif bool((accepted & ~cleanup_only).any()):
            raise PhysicalFlightDeviceError(
                "normal settlement lacks an exact R06 authority"
            )
        if not accepted_indices:
            raise PhysicalFlightDeviceError(
                "settlement acknowledgement requires at least one exact retire row"
            )
        before_checkpoint = self._owner_checkpoint_sha()
        predicted_after_checkpoint = self._owner_checkpoint_sha_for(
            slots=tuple(after_slots),
            mutation_version=before_version + 1,
            next_prepare_nonce=self._next_prepare_nonce,
            poisoned=bool(((terminal_fault | cleanup_only) & accepted).any()),
        )
        if (
            settlement_authority is not None
            and not bool(cleanup_only.any())
            and not bool(terminal_fault.any())
        ):
            portable_receipt = _flight.PhysicalRetireReceipt(
                integration_status=_flight.INTEGRATION_STATUS,
                physical_owner_checkpoint_before_sha256=before_checkpoint,
                physical_owner_checkpoint_after_sha256=predicted_after_checkpoint,
                mutation_version_before=before_version,
                mutation_version_after=before_version + 1,
                num_envs=self.num_envs,
                flight_capacity=self.flight_capacity,
                reset_generations=tuple(self._reset_generation),
                next_prepare_nonce=self._next_prepare_nonce,
                pre_owner_slot_snapshots=self._host_slots,
                post_owner_slot_snapshots=tuple(after_slots),
                rows=tuple(host_rows),
                pre_slots_root_sha256=_flight.physical_slot_root(
                    tuple(row.pre_slot_snapshot for row in host_rows)
                ),
                post_slots_root_sha256=_flight.physical_slot_root(
                    tuple(row.post_slot_snapshot for row in host_rows)
                ),
                physical_flight_released=True,
                mailbox_lifecycle_mutated=False,
                scene_bodies_parked=True,
            )
        if bool(physical_park_mask.any()):
            park_state = self.scene_port.read_state_env().clone()
            park_state[..., :3] = torch.tensor(
                (0.0, 0.0, -20.0), dtype=torch.float32, device=self.device
            )
            park_state[..., 3:7] = torch.tensor(
                (1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device
            )
            park_state[..., 7:] = 0.0
            scene_handle = self.scene_port.preflight_write(
                park_state,
                physical_park_mask,
                device_faults_bound_in_reveal_row=True,
            )
            try:
                # Safety order is physical-first: an R06 failure may leak capacity,
                # but can never create an EMPTY owner row for a still-live ball.
                self._apply_scene_write(scene_handle)
            except Exception as exc:
                self._active_r06_ack = None
                self._active_r06_ack_image = None
                self._poisoned = True
                self._poison_reason = "physical retirement scene publication failed"
                raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        try:
            before_r06 = _require_final_r06_type(
                r06_owner.current_flight_lifecycle_snapshot(),
                expected_name="FlightLifecycleSnapshotBatch",
            )
            r06_version_before_retire = _r06_mutation_version(
                before_r06.mutation_version,
                label="R06 pre-retire mutation_version",
                device=self.device,
            )
            before_snapshot = R06PhysicalFlightReadOnlySnapshot(
                flight_state=_tensor(
                    before_r06.state,
                    label="R06 pre-retire state",
                    shape=shape,
                    dtype=torch.int8,
                    device=self.device,
                ),
                full_key_sha256=_tensor(
                    before_r06.full_key_sha256,
                    label="R06 pre-retire key",
                    shape=shape + (TOKEN_BYTES,),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                ball_generation=_tensor(
                    before_r06.ball_generation,
                    label="R06 pre-retire generation",
                    shape=shape,
                    dtype=torch.int64,
                    device=self.device,
                ),
                observation_ordinal=ack_image.observation_ordinal,
                owner_mutation_version=r06_version_before_retire,
            )
            before_mailbox_slot = _tensor(
                before_r06.mailbox_slot,
                label="R06 pre-retire mailbox slot",
                shape=shape,
                dtype=torch.int64,
                device=self.device,
            )
            if (
                r06_version_before_retire != ack_image.owner_mutation_version
                or r06_physical_snapshot_root(before_snapshot) != after_root
                or not torch.equal(before_mailbox_slot, r06_mailbox_slot)
            ):
                raise PhysicalFlightDeviceError(
                    "R06 owner advanced after settlement acknowledgement"
                )

            retire_call_count = 0
            for slot in range(self.flight_capacity):
                column_mask = r06_retire_mask[:, slot]
                if not bool(column_mask.any()):
                    continue
                actual_result = _require_final_r06_type(
                    r06_owner.retire_physical(
                        mask=column_mask.clone(),
                        flight_slot=torch.full(
                            (self.num_envs,),
                            slot,
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        full_key_sha256=r06_key[:, slot].clone(),
                        ball_generation=r06_generation[:, slot].clone(),
                    ),
                    expected_name="DeviceMutationResult",
                )
                actual_accepted = _tensor(
                    actual_result.accepted,
                    label="R06 retire accepted",
                    shape=(self.num_envs,),
                    dtype=torch.bool,
                    device=self.device,
                )
                actual_rejected = _tensor(
                    actual_result.rejected,
                    label="R06 retire rejected",
                    shape=(self.num_envs,),
                    dtype=torch.bool,
                    device=self.device,
                )
                actual_fault = _tensor(
                    actual_result.fault_bits,
                    label="R06 retire fault_bits",
                    shape=(self.num_envs,),
                    dtype=torch.int64,
                    device=self.device,
                )
                if (
                    not torch.equal(actual_accepted, column_mask)
                    or bool(actual_rejected.any())
                    or bool((actual_fault != 0).any())
                ):
                    raise PhysicalFlightDeviceError(
                        "actual R06 physical retirement was rejected"
                    )
                retire_call_count += 1

            after_r06 = _require_final_r06_type(
                r06_owner.current_flight_lifecycle_snapshot(),
                expected_name="FlightLifecycleSnapshotBatch",
            )
            after_r06_state = _tensor(
                after_r06.state,
                label="R06 post-retire state",
                shape=shape,
                dtype=torch.int8,
                device=self.device,
            )
            after_r06_key = _tensor(
                after_r06.full_key_sha256,
                label="R06 post-retire key",
                shape=shape + (TOKEN_BYTES,),
                dtype=torch.uint8,
                device=self.device,
            )
            after_r06_generation = _tensor(
                after_r06.ball_generation,
                label="R06 post-retire generation",
                shape=shape,
                dtype=torch.int64,
                device=self.device,
            )
            after_r06_mailbox_slot = _tensor(
                after_r06.mailbox_slot,
                label="R06 post-retire mailbox slot",
                shape=shape,
                dtype=torch.int64,
                device=self.device,
            )
            after_r06_version = _r06_mutation_version(
                after_r06.mutation_version,
                label="R06 post-retire mutation_version",
                device=self.device,
            )
            expected_after_state = torch.where(
                r06_retire_mask,
                torch.zeros_like(r06_state),
                r06_state,
            )
            mailbox_state = r06_owner.mailbox_state
            if (
                after_r06_version
                != r06_version_before_retire + retire_call_count
                or not torch.equal(after_r06_state, expected_after_state)
                or not torch.equal(after_r06_key, r06_key)
                or not torch.equal(after_r06_generation, r06_generation)
                or not torch.equal(after_r06_mailbox_slot, r06_mailbox_slot)
                or not isinstance(mailbox_state, torch.Tensor)
                or mailbox_state.dtype != torch.int8
                or mailbox_state.device != self.device
                or mailbox_state.ndim != 2
                or mailbox_state.shape[0] != self.num_envs
            ):
                raise PhysicalFlightDeviceError(
                    "actual R06 post-retire snapshot differs"
                )
            for env_id, slot in [
                (int(index[0]), int(index[1]))
                for index in r06_retire_mask.nonzero(as_tuple=False).tolist()
            ]:
                mailbox = int(r06_mailbox_slot[env_id, slot])
                if (
                    mailbox < 0
                    or mailbox >= mailbox_state.shape[1]
                    or int(mailbox_state[env_id, mailbox]) != 1
                ):
                    raise PhysicalFlightDeviceError(
                        "R06 retirement mutated the settlement mailbox lifecycle"
                    )
        except Exception as exc:
            self._active_r06_ack = None
            self._active_r06_ack_image = None
            self._poisoned = True
            self._poison_reason = "actual R06 physical retirement failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        try:
            self._parked.copy_(
                torch.where(
                    physical_park_mask,
                    torch.ones_like(self._parked),
                    self._parked,
                )
            )
            self._published.copy_(
                torch.where(
                    physical_park_mask,
                    torch.zeros_like(self._published),
                    self._published,
                )
            )
            self._lifecycle.copy_(
                torch.where(
                    physical_park_mask,
                    torch.zeros_like(self._lifecycle),
                    self._lifecycle,
                )
            )
            self._slot_version.add_(physical_park_mask.to(torch.int64))
            self._host_slots = tuple(after_slots)
            self._advance_owner_mutation_version()
            self._active_r06_ack = None
            self._active_r06_ack_image = None
            if bool(((terminal_fault | cleanup_only) & accepted).any()):
                self._poisoned = True
                self._poison_reason = (
                    "fault/orphan settlement retired; launch remains HOLD"
                )
        except Exception as exc:
            self._active_r06_ack = None
            self._active_r06_ack_image = None
            self._poisoned = True
            self._poison_reason = "physical retirement metadata publication failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        # Deliberately do not clear key/generation/install fields.  A retired
        # physical slot retains its full identity until reuse/true reset.
        return PhysicalRetireDeviceResult(
            accepted=accepted,
            rejected=rejected,
            owner_join_fault=rejected | terminal_fault | cleanup_only,
            portable_receipt=portable_receipt,
        )

    def true_reset_many(
        self,
        *,
        selected_env_ids: Sequence[int],
        prior_reset_generations: Sequence[int],
        zero_open_all_owner_closure: _flight.CanonicalJsonContentPin,
        expected_zero_open_all_owner_closure_sha256: str,
    ) -> _flight.PhysicalTrueResetReceipt:
        """Park all K selected slots while preserving unselected bytes exactly."""

        self._require_operable()
        self._require_selected_contact_reward_cycle_closed(
            label="legacy true reset"
        )
        if self._device_r05_reset_owner is not None:
            raise PhysicalFlightDeviceError(
                "legacy host true_reset_many is diagnostic-only after "
                "Device-R05 binding; use selected device reset"
            )
        self._require_cpu_reference()
        if self._active_prepare is not None:
            raise PhysicalFlightDeviceError("true reset cannot cross an active prepare")
        if self._active_postphysics is not None:
            raise PhysicalFlightDeviceError(
                "true reset cannot cross postphysics acknowledgement"
            )
        if self._action_epoch_substep_pair is not None:
            raise PhysicalFlightDeviceError(
                "true reset cannot cross an ActionEpoch physics pair"
            )
        if self._active_r06_ack is not None:
            raise PhysicalFlightDeviceError(
                "true reset cannot cross an unconsumed R06 settlement authority"
            )
        selected = tuple(selected_env_ids)
        if (
            not selected
            or selected != tuple(sorted(set(selected)))
            or any(type(env_id) is not int or env_id < 0 or env_id >= self.num_envs for env_id in selected)
        ):
            raise PhysicalFlightDeviceError("selected_env_ids must be sorted/unique/in-range")
        prior = tuple(prior_reset_generations)
        if len(prior) != len(selected) or any(type(item) is not int or item < 1 for item in prior):
            raise PhysicalFlightDeviceError("prior reset generations differ")
        expected_closure = _sha256(
            expected_zero_open_all_owner_closure_sha256,
            label="expected_zero_open_all_owner_closure_sha256",
        )
        if (
            type(zero_open_all_owner_closure) is not _flight.CanonicalJsonContentPin
            or zero_open_all_owner_closure.canonical_sha256 != expected_closure
            or zero_open_all_owner_closure.source_kind
            != _flight.PHYSICAL_ZERO_OPEN_RESET_CLOSURE_KIND
        ):
            raise PhysicalFlightDeviceError("zero-open closure external pin differs")
        closure_mapping = zero_open_all_owner_closure.decoded_mapping
        if (
            closure_mapping.get("selected_env_ids") != list(selected)
            or closure_mapping.get("open_flight_count") != 0
            or closure_mapping.get("open_mailbox_count") != 0
        ):
            raise PhysicalFlightDeviceError(
                "zero-open closure does not bind the selected envs and zero owner counts"
            )
        if any(self._reset_generation[env_id] != generation for env_id, generation in zip(selected, prior)):
            raise PhysicalFlightDeviceError("true-reset generation binding differs")
        selected_index = list(selected)
        if bool(self._published[selected_index, :].any()) or bool(
            self._device_fault[selected_index, :].any()
        ):
            raise PhysicalFlightDeviceError(
                "true reset requires zero open physical flights and zero owner faults"
            )

        before_checkpoint = self._owner_checkpoint_sha()
        before_version = self._mutation_version
        selected_set = set(selected)
        before_selected = tuple(
            slot for slot in self._host_slots if slot.env_id in selected_set
        )
        before_unselected = tuple(
            slot for slot in self._host_slots if slot.env_id not in selected_set
        )
        after_slots = list(self._host_slots)
        rows: list[_flight.PhysicalTrueResetRow] = []
        for env_id, generation in zip(selected, prior):
            env_before = tuple(
                self._host_slots[self._slot_offset(env_id, slot)]
                for slot in range(self.flight_capacity)
            )
            env_after = tuple(
                self._parked_snapshot(
                    env_id=env_id,
                    slot_index=slot,
                    state=_flight.CanonicalPhysicalBallStateF32(
                        position_env_m=(0.0, 0.0, -20.0),
                        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                        linear_velocity_world_mps=(0.0, 0.0, 0.0),
                        angular_velocity_world_radps=(0.0, 0.0, 0.0),
                    ),
                    version=env_before[slot].mutation_version + 1,
                )
                for slot in range(self.flight_capacity)
            )
            for slot, snapshot in enumerate(env_after):
                after_slots[self._slot_offset(env_id, slot)] = snapshot
            rows.append(
                _flight.PhysicalTrueResetRow(
                    env_id=env_id,
                    prior_reset_generation=generation,
                    next_reset_generation=generation + 1,
                    pre_slot_snapshots=env_before,
                    post_slot_snapshots=env_after,
                )
            )
        selected_mask = torch.zeros(
            (self.num_envs, self.flight_capacity), dtype=torch.bool, device=self.device
        )
        selected_mask[list(selected), :] = True
        park_state = self.scene_port.read_state_env().clone()
        park_state[list(selected), :, :] = 0.0
        park_state[list(selected), :, 2] = -20.0
        park_state[list(selected), :, 3] = 1.0
        scene_handle = self.scene_port.preflight_write(
            park_state,
            selected_mask,
            device_faults_bound_in_reveal_row=True,
        )
        predicted_reset_generations = list(self._reset_generation)
        for env_id in selected:
            predicted_reset_generations[env_id] += 1
        after_slots_tuple = tuple(after_slots)
        after_selected = tuple(
            slot for slot in after_slots_tuple if slot.env_id in selected_set
        )
        after_unselected = tuple(
            slot for slot in after_slots_tuple if slot.env_id not in selected_set
        )
        predicted_after_checkpoint = self._owner_checkpoint_sha_for(
            slots=after_slots_tuple,
            mutation_version=before_version + 1,
            next_prepare_nonce=self._next_prepare_nonce,
            poisoned=False,
            reset_generations=predicted_reset_generations,
        )
        prevalidated_receipt = _flight.PhysicalTrueResetReceipt(
            integration_status=_flight.INTEGRATION_STATUS,
            zero_open_all_owner_closure=zero_open_all_owner_closure,
            selected_env_ids=selected,
            rows=tuple(rows),
            physical_owner_checkpoint_before_sha256=before_checkpoint,
            physical_owner_checkpoint_after_sha256=predicted_after_checkpoint,
            mutation_version_before=before_version,
            mutation_version_after=before_version + 1,
            num_envs=self.num_envs,
            flight_capacity=self.flight_capacity,
            reset_generations_before=tuple(self._reset_generation),
            reset_generations_after=tuple(predicted_reset_generations),
            next_prepare_nonce=self._next_prepare_nonce,
            pre_owner_slot_snapshots=self._host_slots,
            post_owner_slot_snapshots=after_slots_tuple,
            selected_slots_root_before_sha256=_flight.physical_slot_root(before_selected),
            selected_slots_root_after_sha256=_flight.physical_slot_root(after_selected),
            unselected_slots_root_before_sha256=_flight.physical_slot_root(before_unselected),
            unselected_slots_root_after_sha256=_flight.physical_slot_root(after_unselected),
            env_reset_invoked=False,
            mailbox_lifecycle_mutated=False,
        )
        try:
            self._apply_scene_write(scene_handle)
            self._host_slots = after_slots_tuple
            self._lifecycle[list(selected), :] = R06_FLIGHT_EMPTY
            self._generation[list(selected), :] = -1
            self._outcome_sha[list(selected), :, :] = 0
            self._install_sha[list(selected), :, :] = 0
            self._installed_state_sha[list(selected), :, :] = 0
            self._reveal_step[list(selected), :] = -1
            self._observation_ordinal[list(selected), :] = -1
            self._previous_ball_center[list(selected), :, :] = park_state[
                list(selected), :, :3
            ]
            self._parked[list(selected), :] = True
            self._published[list(selected), :] = False
            self._device_fault[list(selected), :] = False
            self._slot_version[list(selected), :] += 1
            self._reset_generation = predicted_reset_generations
            self._action_epoch_host_activity_control_step = None
            self._action_epoch_host_activity_has_work = True
            self._advance_owner_mutation_version()
        except Exception as exc:
            self._poisoned = True
            self._poison_reason = "true-reset scene publication failed"
            raise PhysicalFlightOwnerPoisonedError(self._poison_reason) from exc
        return prevalidated_receipt

    def _require_action_epoch_checkpoint_clear(self) -> None:
        """HOLD until lean pending/live keys have a portable schema."""

        pending = self._action_epoch_pending_launch
        if (
            (pending is not None and bool(pending.pending.any()))
            or bool(self._action_epoch_active_flight_slot.ge(0).any())
            or bool(self._action_epoch_flight_publication_ordinal.ge(0).any())
        ):
            raise PhysicalEpochIntegrationHold(
                "checkpoint HOLD: lean ActionEpoch pending/live Physical state "
                "has no portable checkpoint schema"
            )

__all__ = [
    "AcknowledgedR06PhysicalSnapshot",
    "ActionEpochPhysicsFactAllocationProjection",
    "ActionEpochR06LaunchProjection",
    "ActionEpochR06PostPhysicsProjection",
    "ActionEpochSceneWriteProjection",
    "ActionBallPhysicalFlightDeviceOwner",
    "ArmedPhysicalSelectedTrueReset",
    "ArmedPhysicalFlightInstall",
    "CONTRACT_SOURCE_SHA256",
    "CUDA_REVEAL_BOUNDARY_INTEGRATED",
    "FORMAL_EXACT_RESUME_INTEGRATED",
    "INTEGRATION_RESIDUALS",
    "ISAAC_POSTPHYSICS_VALIDATED",
    "IsaacPostPhysicsFacts",
    "PhysicalPostPhysicsCaptureRequest",
    "LAUNCH_AUTHORIZED",
    "OWNER_STATE_SCHEMA_SHA256",
    "PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES",
    "PHYSICAL_GLOBAL_DRAIN_OWNER_KIND",
    "PHYSICAL_HOT_FAULT_D05_PROJECTION",
    "PHYSICAL_HOT_FAULT_IDENTITY",
    "PHYSICAL_HOT_FAULT_LAUNCH_TICK",
    "PHYSICAL_HOT_FAULT_QUESTION_BINDING",
    "PHYSICAL_HOT_FAULT_SCENE_PRECONDITION",
    "PHYSICAL_EPOCH_FACT_PRESENT",
    "PHYSICAL_EPOCH_FACT_SELECTED_CONTACT",
    "PHYSICAL_EPOCH_FAULT_LAUNCH_SOURCE",
    "PHYSICAL_EPOCH_FAULT_POSTPHYSICS_NONFINITE",
    "PHYSICAL_EPOCH_FAULT_POSTPHYSICS_PRODUCER",
    "PHYSICAL_PPO_DRAIN_LEAF_SCHEMA",
    "PhysicalFlightDeviceError",
    "PhysicalFlightOwnerPoisonedError",
    "PhysicalEpochIntegrationHold",
    "PhysicalEpochSelectedContactFacts",
    "PhysicalHotChildCommitToken",
    "PhysicalHotPreparedInstall",
    "PhysicalLateLaunchProductionHold",
    "PhysicalLateLaunchPublication",
    "PhysicalLateLaunchPublicationView",
    "PHYSICAL_SELECTED_RESET_DEVICE_PARK_INTEGRATED",
    "PHYSICAL_SELECTED_RESET_R05_ACK_INTEGRATED",
    "PhysicalFlightSceneSnapshotNK",
    "PhysicalDeviceInstallAbortReceipt",
    "PhysicalDeviceInstallPrepareReceipt",
    "PhysicalDeviceInstallTerminalReceipt",
    "PhysicalPostPhysicsPublication",
    "PhysicalRetireDeviceResult",
    "PhysicalSelectedContactRewardPaymentResult",
    "PhysicalSelectedContactRewardView",
    "PhysicalSettleRetireInput",
    "PhysicalSelectedTrueResetCompletionToken",
    "PhysicalSelectedTrueResetParkCommitToken",
    "PhysicsStampGrid",
    "PreparedPhysicalFlightInstall",
    "FinalizedPhysicalSelectedTrueReset",
    "R06PhysicalFlightReadOnlySnapshot",
    "R10_CHECKPOINT_ADAPTER_INTEGRATED",
    "RUNTIME_INTEGRATED",
    "R05_SOURCE_SHA256",
    "R05_REVEAL_SOURCE_PIN_PENDING",
    "R06_SOURCE_SHA256",
    "TensorPhysicalFlightScenePort",
    "StagedPhysicalSelectedTrueReset",
    "materialize_physical_ppo_drain_leaf_schema",
    "r06_physical_snapshot_root",
]
