"""Private Lean-root carry transaction groundwork.

This module is deliberately not a checkpoint, persistence, or load API.  It
only freezes an in-memory owner graph, performs one composite device-to-host
copy, and can install that private carry into a separately constructed graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

import torch

class _LeanCarryError(RuntimeError):
    pass

@dataclass(frozen=True)
class _LeanCarryTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    disposition: str = "copy"
    placement: str = "device"

@dataclass(frozen=True)
class _LeanCarrySchema:
    role: str
    scalar_fields: tuple[tuple[str, type], ...]
    tensor_fields: tuple[_LeanCarryTensorSpec, ...]

@dataclass(frozen=True)
class _LeanCarryCapture:
    scalars: tuple[object, ...]
    tensors: tuple[torch.Tensor, ...]

@dataclass(frozen=True)
class _LeanCarryStage:
    scalars: tuple[object, ...]
    staging: tuple[torch.Tensor, ...]
    targets: tuple[torch.Tensor, ...]
    commit_started: bool = False

class _LeanCarryLease:
    __slots__ = ("coordinator", "generation", "kind")

    def __init__(self, coordinator: "_LeanCarryCoordinator", generation: int, kind: str):
        self.coordinator = coordinator
        self.generation = generation
        self.kind = kind

@dataclass(frozen=True)
class _ImageState:
    schemas: tuple[_LeanCarrySchema, ...]
    scalars: tuple[tuple[object, ...], ...]
    host_tensors: tuple[tuple[torch.Tensor, ...], ...]
    source_ids: tuple[int, ...]
    source_device: torch.device
    source_tensors: tuple[tuple[torch.Tensor, ...], ...]

class _LeanCarryImage:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("carry image is owner-minted")

class _LeanCarryPrepared:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("carry prepare authority is owner-minted")

@dataclass
class _PreparedState:
    image: _LeanCarryImage
    image_state: _ImageState
    lease: _LeanCarryLease
    stages: tuple[_LeanCarryStage, ...]
    commit_started: bool = False

_PROCESS_LOCK = threading.RLock()
_PROCESS_POISON_REASON: Optional[str] = None
_IMAGE_STATES: dict[_LeanCarryImage, _ImageState] = {}
_IMAGE_BUSY: set[_LeanCarryImage] = set()

def _poison_process(reason: str) -> None:
    global _PROCESS_POISON_REASON
    with _PROCESS_LOCK:
        if _PROCESS_POISON_REASON is None:
            _PROCESS_POISON_REASON = reason

def _require_process_healthy() -> None:
    with _PROCESS_LOCK:
        if _PROCESS_POISON_REASON is not None:
            raise _LeanCarryError(
                "Lean carry process is poisoned; retry is forbidden: "
                + _PROCESS_POISON_REASON
            )


def _single_composite_d2h(packed: torch.Tensor) -> torch.Tensor:
    """The one business device-to-host transfer for the whole owner graph."""

    return packed.to(device="cpu", copy=True).contiguous()


def _storage_interval(value: torch.Tensor) -> tuple[torch.device, int, int, int]:
    pointer = value.untyped_storage().data_ptr()
    start = value.storage_offset() * value.element_size()
    return value.device, pointer, start, start + value.numel() * value.element_size()


def _require_disjoint(values, *, label: str) -> None:
    occupied: list[tuple[torch.device, int, int, int, str]] = []
    for name, value in values:
        device, pointer, start, end = _storage_interval(value)
        for prior_device, prior_pointer, prior_start, prior_end, prior_name in occupied:
            if (
                device == prior_device and pointer == prior_pointer
                and start < prior_end and prior_start < end
            ):
                raise _LeanCarryError(f"{label}.{name} aliases {label}.{prior_name}")
        occupied.append((device, pointer, start, end, name))


def _require_schema(value: object, *, role: str) -> _LeanCarrySchema:
    if (
        type(value) is not _LeanCarrySchema
        or value.role != role
        or not role
        or type(value.scalar_fields) is not tuple
        or type(value.tensor_fields) is not tuple
        or len({name for name, _ in value.scalar_fields}) != len(value.scalar_fields)
        or len({field.name for field in value.tensor_fields}) != len(value.tensor_fields)
    ):
        raise _LeanCarryError(role + " carry schema differs")
    for name, exact_type in value.scalar_fields:
        if type(name) is not str or not name or type(exact_type) is not type:
            raise _LeanCarryError(role + " scalar schema differs")
    for field in value.tensor_fields:
        if (
            type(field) is not _LeanCarryTensorSpec
            or type(field.name) is not str
            or not field.name
            or type(field.shape) is not tuple
            or any(type(size) is not int or size < 0 for size in field.shape)
            or not isinstance(field.dtype, torch.dtype)
            or field.disposition not in ("copy", "attest")
            or field.placement not in ("device", "host")
        ):
            raise _LeanCarryError(role + " tensor schema differs")
    return value


def _require_scalars(schema: _LeanCarrySchema, values: object, *, label: str) -> tuple:
    if type(values) is not tuple or len(values) != len(schema.scalar_fields):
        raise _LeanCarryError(label + " scalar arity differs")
    for value, (name, exact_type) in zip(values, schema.scalar_fields):
        if type(value) is not exact_type:
            raise _LeanCarryError(label + "." + name + " scalar type differs")
    return values


def _require_tensors(schema: _LeanCarrySchema, values: object, *, label: str):
    if type(values) is not tuple or len(values) != len(schema.tensor_fields):
        raise _LeanCarryError(label + " tensor arity differs")
    named = []
    for value, field in zip(values, schema.tensor_fields):
        if (
            type(value) is not torch.Tensor
            or value.numel() == 0
            or value.dtype is not field.dtype
            or tuple(value.shape) != field.shape
            or not value.is_contiguous()
        ):
            raise _LeanCarryError(label + "." + field.name + " tensor ABI differs")
        if field.placement == "host" and value.device.type != "cpu":
            raise _LeanCarryError(label + "." + field.name + " must be host-resident")
        named.append((field.name, value))
    return tuple(named)


def _exact_method(owner: object, name: str):
    function = vars(type(owner)).get(name)
    bound = getattr(owner, name, None)
    if (
        not callable(function)
        or not callable(bound)
        or getattr(bound, "__self__", None) is not owner
        or getattr(bound, "__func__", None) is not function
    ):
        raise _LeanCarryError(type(owner).__name__ + " lacks exact " + name)
    return bound


def _require_leaf_mutable(owner: object) -> None:
    coordinator = getattr(owner, "_lean_carry_coordinator", None)
    if type(coordinator) is _LeanCarryCoordinator and coordinator._active_lease is not None:
        raise _LeanCarryError("owner mutation overlaps a graph-wide carry lease")
    _require_process_healthy()


class _LeanCarryCoordinator:
    """One root-private registry, capture lease, and restore transaction."""

    def __init__(self, *, root: object, mandatory_roles: tuple[str, ...]) -> None:
        if root is None or type(mandatory_roles) is not tuple or not mandatory_roles:
            raise _LeanCarryError("carry root/mandatory role ABI differs")
        if mandatory_roles[0] != "root" or len(set(mandatory_roles)) != len(mandatory_roles):
            raise _LeanCarryError("carry roles must be unique and root-first")
        self._root = root
        self._mandatory_roles = mandatory_roles
        self._owners: dict[str, object] = {}
        self._schemas: dict[str, _LeanCarrySchema] = {}
        self._callbacks: dict[str, tuple[object, ...]] = {}
        self._construction_views: dict[str, tuple[torch.Tensor, ...]] = {}
        self._active_lease: Optional[_LeanCarryLease] = None
        self._prepared: Optional[_LeanCarryPrepared] = None
        self._prepared_state: Optional[_PreparedState] = None
        self._generation = 0
        self._lock = threading.RLock()

    def _register(self, role: str, owner: object) -> None:
        with self._lock:
            _require_process_healthy()
            if (
                role not in self._mandatory_roles or role in self._owners or owner is None
                or any(owner is installed for installed in self._owners.values())
            ):
                raise _LeanCarryError("carry role is foreign, duplicate, or null: " + str(role))
            names = (
                "_lean_carry_schema",
                "_lean_carry_construction_views",
                "_lean_carry_capture", "_lean_carry_stage",
                "_lean_carry_target_views", "_lean_carry_apply_scalars",
            )
            callbacks = tuple(_exact_method(owner, name) for name in names)
            schema = _require_schema(callbacks[0](), role=role)
            construction = callbacks[1]()
            _require_tensors(schema, construction, label=role + ".construction")
            installed = getattr(owner, "_lean_carry_coordinator", None)
            if installed is not None:
                raise _LeanCarryError(role + " already belongs to a carry root")
            setattr(owner, "_lean_carry_coordinator", self)
            self._owners[role] = owner
            self._schemas[role] = schema
            self._callbacks[role] = callbacks
            self._construction_views[role] = construction

    def _require_complete(self) -> None:
        missing = tuple(role for role in self._mandatory_roles if role not in self._owners)
        if missing:
            raise _LeanCarryError(
                "Lean carry preflight lacks mandatory roles: " + ",".join(missing)
            )

    def _capture(self) -> _LeanCarryImage:
        with self._lock:
            _require_process_healthy()
            self._require_complete()
            if self._active_lease is not None or self._prepared is not None:
                raise _LeanCarryError("carry root already has an active lease")
            self._generation += 1
            lease = _LeanCarryLease(self, self._generation, "capture")
            self._active_lease = lease
            try:
                schemas = tuple(self._schemas[role] for role in self._mandatory_roles)
                captures = []
                flat = []
                device = None
                for role, schema in zip(self._mandatory_roles, schemas):
                    callbacks = self._callbacks[role]
                    if _require_schema(callbacks[0](), role=role) != schema:
                        raise _LeanCarryError(role + " construction schema changed")
                    capture = callbacks[2](lease)
                    if type(capture) is not _LeanCarryCapture:
                        raise _LeanCarryError(role + " capture type differs")
                    _require_scalars(schema, capture.scalars, label=role + ".source")
                    named = _require_tensors(schema, capture.tensors, label=role + ".source")
                    for field, (name, value) in zip(schema.tensor_fields, named):
                        if field.placement == "device":
                            if device is None:
                                device = value.device
                            elif value.device != device:
                                raise _LeanCarryError("owner graph spans devices")
                        flat.append((role + "." + name, field, value))
                    captures.append(capture)
                _require_disjoint(
                    tuple((name, value) for name, _field, value in flat),
                    label="source",
                )
                device_flat = tuple(
                    (index, name, value)
                    for index, (name, field, value) in enumerate(flat)
                    if field.placement == "device"
                )
                raw = tuple(
                    value.detach().reshape(-1).view(torch.uint8)
                    for _index, _name, value in device_flat
                )
                if not raw:
                    raise _LeanCarryError("owner graph has no tensor carry")
                packed = torch.cat(raw, dim=0).contiguous()
                host = _single_composite_d2h(packed)
                host_tensors = [
                    value.detach().clone().contiguous()
                    if field.placement == "host" else None
                    for _name, field, value in flat
                ]
                offset = 0
                for index, _name, value in device_flat:
                    nbytes = value.numel() * value.element_size()
                    # The packed byte offset need not satisfy the next dtype's
                    # alignment.  Clone each logical slice to offset zero while
                    # retaining the one physical D2H above.
                    logical = host.narrow(0, offset, nbytes).clone()
                    host_tensors[index] = logical.view(value.dtype).reshape(value.shape)
                    offset += nbytes
                grouped = []
                cursor = 0
                for schema in schemas:
                    width = len(schema.tensor_fields)
                    grouped.append(tuple(host_tensors[cursor:cursor + width]))
                    cursor += width
                state = _ImageState(
                    schemas,
                    tuple(capture.scalars for capture in captures),
                    tuple(grouped),
                    tuple(id(self._owners[role]) for role in self._mandatory_roles),
                    device,
                    tuple(capture.tensors for capture in captures),
                )
                image = object.__new__(_LeanCarryImage)
                with _PROCESS_LOCK:
                    _IMAGE_STATES[image] = state
                return image
            finally:
                self._active_lease = None

    def _prepare(self, image: object) -> _LeanCarryPrepared:
        with self._lock:
            _require_process_healthy()
            self._require_complete()
            if (
                type(image) is not _LeanCarryImage
                or self._active_lease is not None or self._prepared is not None
            ):
                raise _LeanCarryError("carry image is foreign, stale, or replayed")
            with _PROCESS_LOCK:
                state = _IMAGE_STATES.get(image)
                if state is None or image in _IMAGE_BUSY:
                    raise _LeanCarryError("carry image is foreign, stale, or replayed")
                _IMAGE_BUSY.add(image)
            schemas = tuple(self._schemas[role] for role in self._mandatory_roles)
            if state.schemas != schemas:
                with _PROCESS_LOCK:
                    _IMAGE_BUSY.discard(image)
                raise _LeanCarryError("source and target graph ABI differs")
            target_ids = tuple(
                id(self._owners[role]) for role in self._mandatory_roles
            )
            if set(target_ids).intersection(state.source_ids):
                with _PROCESS_LOCK:
                    _IMAGE_BUSY.discard(image)
                raise _LeanCarryError(
                    "source and target carry owner identities overlap"
                )
            self._generation += 1
            lease = _LeanCarryLease(self, self._generation, "prepare")
            self._active_lease = lease
            try:
                stages = []
                staging_named = []
                target_named = []
                for role, schema, scalars, host in zip(
                    self._mandatory_roles, schemas, state.scalars, state.host_tensors
                ):
                    stage = self._callbacks[role][3](lease, scalars, host)
                    if type(stage) is not _LeanCarryStage or stage.commit_started:
                        raise _LeanCarryError(role + " stage type differs")
                    _require_scalars(schema, stage.scalars, label=role + ".stage")
                    if stage.scalars != scalars:
                        raise _LeanCarryError(role + " staged scalars changed")
                    staged = _require_tensors(schema, stage.staging, label=role + ".stage")
                    targets = _require_tensors(schema, stage.targets, label=role + ".target")
                    staging_named.extend((role + "." + name, value) for name, value in staged)
                    target_named.extend((role + "." + name, value) for name, value in targets)
                    for field, source, target in zip(schema.tensor_fields, stage.staging, stage.targets):
                        if field.placement == "device" and (
                            source.device != state.source_device
                            or target.device != state.source_device
                        ):
                            raise _LeanCarryError(role + " source/target device differs")
                        if field.disposition == "attest" and not torch.equal(source, target):
                            raise _LeanCarryError(role + "." + field.name + " attestation differs")
                    if any(
                        target is not frozen
                        for target, frozen in zip(
                            stage.targets, self._construction_views[role]
                        )
                    ):
                        raise _LeanCarryError(role + " construction target identity changed")
                    stages.append(_LeanCarryStage(stage.scalars, stage.staging, stage.targets))
                host_named = []
                for role, schema, values in zip(
                    self._mandatory_roles, schemas, state.host_tensors
                ):
                    host_named.extend(
                        (role + "." + field.name, value)
                        for field, value in zip(schema.tensor_fields, values)
                    )
                _require_disjoint(
                    [
                        *(
                            ("source." + role + "." + field.name, value)
                            for role, schema, values in zip(
                                self._mandatory_roles, schemas, state.source_tensors
                            )
                            for field, value in zip(schema.tensor_fields, values)
                        ),
                        *(("host." + name, value) for name, value in host_named),
                        *(("staging." + name, value) for name, value in staging_named),
                        *(("target." + name, value) for name, value in target_named),
                    ],
                    label="graph",
                )
                cross = _exact_method(self._root, "_lean_carry_cross_validate")
                cross(
                    lease,
                    tuple(state.scalars),
                    tuple(state.host_tensors),
                    tuple(stage.scalars for stage in stages),
                )
                prepared = object.__new__(_LeanCarryPrepared)
                self._prepared = prepared
                self._prepared_state = _PreparedState(
                    image, state, lease, tuple(stages)
                )
                return prepared
            except BaseException:
                self._active_lease = None
                with _PROCESS_LOCK:
                    _IMAGE_BUSY.discard(image)
                raise

    def _discard(self, image: object) -> None:
        """Tombstone one unused owner-minted image without exposing its state."""

        with self._lock, _PROCESS_LOCK:
            if (
                type(image) is not _LeanCarryImage
                or image not in _IMAGE_STATES
                or image in _IMAGE_BUSY
            ):
                raise _LeanCarryError("carry discard authority differs")
            _IMAGE_STATES.pop(image)

    def _abort(self, prepared: object) -> None:
        with self._lock:
            state = self._prepared_state
            if (
                type(prepared) is not _LeanCarryPrepared
                or prepared is not self._prepared or state is None
                or state.commit_started or state.lease is not self._active_lease
            ):
                raise _LeanCarryError("carry abort authority differs")
            with _PROCESS_LOCK:
                _IMAGE_BUSY.discard(state.image)
            self._prepared = None
            self._prepared_state = None
            self._active_lease = None

    def _commit(self, prepared: object) -> None:
        with self._lock:
            _require_process_healthy()
            state = self._prepared_state
            if (
                type(prepared) is not _LeanCarryPrepared
                or prepared is not self._prepared or state is None
                or state.commit_started or state.lease is not self._active_lease
            ):
                raise _LeanCarryError("carry commit authority differs")
            try:
                recheck_named = []
                for role, schema, stage in zip(
                    self._mandatory_roles,
                    state.image_state.schemas,
                    state.stages,
                ):
                    staged = _require_tensors(
                        schema, stage.staging, label=role + ".commit-stage"
                    )
                    current = self._callbacks[role][4](state.lease, stage)
                    current_named = _require_tensors(
                        schema, current, label=role + ".commit-target"
                    )
                    if any(now is not frozen for now, frozen in zip(current, stage.targets)):
                        raise _LeanCarryError(role + " target identity changed")
                    if any(
                        field.disposition == "attest" and not torch.equal(source, target)
                        for field, source, target in zip(
                            schema.tensor_fields, stage.staging, current
                        )
                    ):
                        raise _LeanCarryError(role + " attestation changed")
                    recheck_named.extend(
                        (prefix + role + "." + name, value)
                        for prefix, named in (
                            ("staging.", staged), ("target.", current_named)
                        )
                        for name, value in named
                    )
                _require_disjoint(
                    [
                        *(
                            ("source." + role + "." + field.name, value)
                            for role, schema, values in zip(
                                self._mandatory_roles,
                                state.image_state.schemas,
                                state.image_state.source_tensors,
                            )
                            for field, value in zip(schema.tensor_fields, values)
                        ),
                        *recheck_named,
                    ],
                    label="commit-graph",
                )
                state.commit_started = True
                armed = tuple(
                    _LeanCarryStage(stage.scalars, stage.staging, stage.targets, True)
                    for stage in state.stages
                )
                state.stages = armed
                for schema, stage in zip(state.image_state.schemas, armed):
                    for field, source, target in zip(
                        schema.tensor_fields, stage.staging, stage.targets
                    ):
                        if field.disposition == "copy":
                            target.copy_(source)
                for role, stage in zip(self._mandatory_roles, armed):
                    self._callbacks[role][5](state.lease, stage)
            except BaseException as exc:
                if state.commit_started:
                    _poison_process("partial carry commit: " + type(exc).__name__)
                raise
            else:
                with _PROCESS_LOCK:
                    _IMAGE_BUSY.discard(state.image)
                    _IMAGE_STATES.pop(state.image, None)
                self._prepared = None
                self._prepared_state = None
                self._active_lease = None


__all__ = []
