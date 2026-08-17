"""Independent reset-genesis owner for the fresh full-MDP runtime.

The initial world-reset generation is a protocol constant: every live world
starts at generation one.  Callers therefore provide only the exact device and
world count; they cannot supply, sign, or mutate genesis bytes.  One opaque
receipt may be projected once to each of the four named consumers.  All
projections share one owner-issued world identity and receive separate clones
of the same retained ``int64[N]`` tensor.

This cold construction owner deliberately has no hashes, caller-authored
tuples, launch flags, or runtime-manager dependencies.  The receiving modules
remain responsible for their own exact projection-type checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import threading
from typing import Type, cast

import torch


_ENV_MODULE = "whole_body_tracking.tasks.tracking.full_mdp_env"
_ENV_PROJECTION = "FullMdpResetGenesisProjection"
_R05_MODULE = "action_ball_continuous_runtime_transaction_device"
_R05_PROJECTION = "DeviceGenesisProjection"


class ActionBallFullMdpResetGenesisError(RuntimeError):
    """The independent genesis capability or requested projection differs."""


class _OpaqueCapability:
    """Identity-only capability minted through ``object.__new__`` internally."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("reset-genesis capabilities are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("reset-genesis capabilities are immutable")

    def __copy__(self):
        raise TypeError("reset-genesis capabilities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("reset-genesis capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("reset-genesis capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("reset-genesis capabilities cannot be serialized")


class ActionBallFullMdpResetGenesisReceipt(_OpaqueCapability):
    """Opaque registry key authorizing the four exact genesis projections."""


class _WorldResetIdentity(_OpaqueCapability):
    """Shared identity joining env, Device-R05, Physical, and ActionEpoch."""


class ActionBallFullMdpResetGenesisAuthority(_OpaqueCapability):
    """Owner exposing exactly one projection to each named consumer."""

    def require_owned_full_mdp_reset_genesis(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
    ) -> object:
        """Project one clone to the exact fresh full-MDP environment type."""

        return _project(
            self,
            receipt,
            consumer="full_mdp_env",
            module_name=_ENV_MODULE,
            projection_name=_ENV_PROJECTION,
            device=device,
            num_envs=num_envs,
        )

    def require_owned_r05_genesis(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
    ) -> object:
        """Project one clone to the exact Device-R05 genesis type."""

        return _project(
            self,
            receipt,
            consumer="device_r05",
            module_name=_R05_MODULE,
            projection_name=_R05_PROJECTION,
            device=device,
            num_envs=num_envs,
        )

    def require_owned_physical_genesis(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
    ) -> "PhysicalResetGenesisProjection":
        """Project one clone to the fresh Physical construction boundary."""

        projection = _project(
            self,
            receipt,
            consumer="physical",
            module_name=None,
            projection_name=None,
            device=device,
            num_envs=num_envs,
        )
        if type(projection) is not PhysicalResetGenesisProjection:
            raise AssertionError("internal Physical genesis projection differs")
        return projection

    def require_owned_action_epoch_genesis(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
    ) -> "ActionBallFullMdpActionEpochGenesisProjection":
        """Project one clone to the fresh ActionEpoch construction boundary."""

        projection = _project(
            self,
            receipt,
            consumer="action_epoch",
            module_name=None,
            projection_name=None,
            device=device,
            num_envs=num_envs,
        )
        if type(projection) is not ActionBallFullMdpActionEpochGenesisProjection:
            raise AssertionError("internal ActionEpoch genesis projection differs")
        return projection


@dataclass(frozen=True, slots=True)
class ActionBallFullMdpResetGenesisIssue:
    """Construction result containing only owner-issued capabilities."""

    authority: ActionBallFullMdpResetGenesisAuthority
    receipt: ActionBallFullMdpResetGenesisReceipt


@dataclass(frozen=True, slots=True)
class PhysicalResetGenesisProjection:
    """Clone-only code-owned genesis input for the Physical device owner."""

    world_reset_identity: object
    reset_generations: torch.Tensor


@dataclass(frozen=True, slots=True)
class ActionBallFullMdpActionEpochGenesisProjection:
    """Clone-only genesis input for the single ActionEpoch device owner."""

    world_reset_identity: object
    reset_generations: torch.Tensor


@dataclass(slots=True)
class _GenesisRecord:
    authority: ActionBallFullMdpResetGenesisAuthority
    world_reset_identity: _WorldResetIdentity
    device: torch.device
    num_envs: int
    reset_generations: torch.Tensor
    full_mdp_env_projected: bool = False
    device_r05_projected: bool = False
    physical_projected: bool = False
    action_epoch_projected: bool = False


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[ActionBallFullMdpResetGenesisReceipt, _GenesisRecord] = {}


def _mint(capability_type: Type[_OpaqueCapability]) -> _OpaqueCapability:
    return object.__new__(capability_type)


def _require_request(
    *, device: object, num_envs: object
) -> tuple[torch.device, int]:
    if type(device) is not torch.device:
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis device must be an exact torch.device"
        )
    if (
        device.type not in ("cpu", "cuda")
        or (device.type == "cpu" and device.index is not None)
        or (device.type == "cuda" and device.index is None)
    ):
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis device must be unindexed CPU or indexed CUDA"
        )
    if type(num_envs) is not int or num_envs < 1:
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis num_envs must be a positive exact int"
        )
    return device, num_envs


def _projection_type(module_name: str, projection_name: str) -> type:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ActionBallFullMdpResetGenesisError(
            f"exact genesis consumer module {module_name!r} is unavailable"
        ) from exc
    projection_type = vars(module).get(projection_name)
    if type(projection_type) is not type:
        raise ActionBallFullMdpResetGenesisError(
            f"exact genesis projection {projection_name!r} is unavailable"
        )
    return projection_type


def _project(
    authority: object,
    receipt: object,
    *,
    consumer: str,
    module_name: str | None,
    projection_name: str | None,
    device: object,
    num_envs: object,
) -> object:
    requested_device, requested_num_envs = _require_request(
        device=device, num_envs=num_envs
    )
    if type(authority) is not ActionBallFullMdpResetGenesisAuthority:
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis authority is foreign"
        )
    if type(receipt) is not ActionBallFullMdpResetGenesisReceipt:
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis receipt type differs"
        )
    if consumer in ("physical", "action_epoch"):
        if module_name is not None or projection_name is not None:
            raise ActionBallFullMdpResetGenesisError(
                f"{consumer} genesis projection routing differs"
            )
        projection_type = (
            PhysicalResetGenesisProjection
            if consumer == "physical"
            else ActionBallFullMdpActionEpochGenesisProjection
        )
    else:
        if type(module_name) is not str or type(projection_name) is not str:
            raise ActionBallFullMdpResetGenesisError(
                "exact genesis projection route is absent"
            )
        projection_type = _projection_type(module_name, projection_name)
    with _REGISTRY_LOCK:
        record = _REGISTRY.get(receipt)
        if record is None or record.authority is not authority:
            raise ActionBallFullMdpResetGenesisError(
                "reset-genesis receipt is foreign or unregistered"
            )
        if (
            requested_device != record.device
            or requested_num_envs != record.num_envs
        ):
            raise ActionBallFullMdpResetGenesisError(
                "reset-genesis projection device or num_envs differs"
            )
        projected_attr = {
            "full_mdp_env": "full_mdp_env_projected",
            "device_r05": "device_r05_projected",
            "physical": "physical_projected",
            "action_epoch": "action_epoch_projected",
        }.get(consumer)
        if projected_attr is None:
            raise ActionBallFullMdpResetGenesisError(
                "reset-genesis consumer is not registered"
            )
        if getattr(record, projected_attr):
            raise ActionBallFullMdpResetGenesisError(
                f"reset genesis was already projected to {consumer}"
            )
        projection = projection_type(
            world_reset_identity=record.world_reset_identity,
            reset_generations=record.reset_generations.clone(),
        )
        setattr(record, projected_attr, True)
        if (
            record.full_mdp_env_projected
            and record.device_r05_projected
            and record.physical_projected
            and record.action_epoch_projected
        ):
            _REGISTRY.pop(receipt)
        return projection


def issue_action_ball_full_mdp_reset_genesis(
    *, num_envs: int, device: torch.device
) -> ActionBallFullMdpResetGenesisIssue:
    """Mint one production genesis fixed to generation one on ``device``.

    Generation values are intentionally not an argument.  This prevents a
    caller-authored tensor from masquerading as independent reset chronology.
    """

    exact_device, exact_num_envs = _require_request(
        device=device, num_envs=num_envs
    )
    authority = cast(
        ActionBallFullMdpResetGenesisAuthority,
        _mint(ActionBallFullMdpResetGenesisAuthority),
    )
    receipt = cast(
        ActionBallFullMdpResetGenesisReceipt,
        _mint(ActionBallFullMdpResetGenesisReceipt),
    )
    world_reset_identity = cast(
        _WorldResetIdentity, _mint(_WorldResetIdentity)
    )
    record = _GenesisRecord(
        authority=authority,
        world_reset_identity=world_reset_identity,
        device=exact_device,
        num_envs=exact_num_envs,
        reset_generations=torch.ones(
            (exact_num_envs,), dtype=torch.int64, device=exact_device
        ),
    )
    with _REGISTRY_LOCK:
        _REGISTRY[receipt] = record
    return ActionBallFullMdpResetGenesisIssue(
        authority=authority, receipt=receipt
    )


def discard_unpublished_action_ball_full_mdp_reset_genesis(
    *,
    authority: ActionBallFullMdpResetGenesisAuthority,
    receipt: ActionBallFullMdpResetGenesisReceipt,
) -> None:
    """Release a cold issue only before any consumer observed it.

    Once any exact projection has escaped, failure is necessarily partial
    and cannot honestly be described as rollback.  Successful four-consumer
    projection removes the registry record automatically.
    """

    if type(authority) is not ActionBallFullMdpResetGenesisAuthority:
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis discard authority is foreign"
        )
    if type(receipt) is not ActionBallFullMdpResetGenesisReceipt:
        raise ActionBallFullMdpResetGenesisError(
            "reset-genesis discard receipt type differs"
        )
    with _REGISTRY_LOCK:
        record = _REGISTRY.get(receipt)
        if record is None or record.authority is not authority:
            raise ActionBallFullMdpResetGenesisError(
                "reset-genesis discard receipt is foreign or unregistered"
            )
        if (
            record.full_mdp_env_projected
            or record.device_r05_projected
            or record.physical_projected
            or record.action_epoch_projected
        ):
            raise ActionBallFullMdpResetGenesisError(
                "projected reset genesis cannot be rolled back"
            )
        _REGISTRY.pop(receipt)


def retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
    *,
    authority: ActionBallFullMdpResetGenesisAuthority,
    receipt: ActionBallFullMdpResetGenesisReceipt,
) -> None:
    """Retire a failed cold bundle without pretending to roll it back.

    Physical and ActionEpoch may already have consumed their dedicated clones
    when a later cold constructor fails.  Those observations cannot be undone.
    This operation releases the retained record and permanently revokes the
    still-unissued projections.  It refuses any record already observed by env
    or Device-R05 because those are continuing consumers outside the failed
    cold bundle.
    """

    if type(authority) is not ActionBallFullMdpResetGenesisAuthority:
        raise ActionBallFullMdpResetGenesisError(
            "failed reset-genesis retirement authority is foreign"
        )
    if type(receipt) is not ActionBallFullMdpResetGenesisReceipt:
        raise ActionBallFullMdpResetGenesisError(
            "failed reset-genesis retirement receipt type differs"
        )
    with _REGISTRY_LOCK:
        record = _REGISTRY.get(receipt)
        if record is None or record.authority is not authority:
            raise ActionBallFullMdpResetGenesisError(
                "failed reset-genesis retirement receipt is foreign or unregistered"
            )
        if record.full_mdp_env_projected or record.device_r05_projected:
            raise ActionBallFullMdpResetGenesisError(
                "published env or Device-R05 genesis cannot be retired as a failed cold bundle"
            )
        _REGISTRY.pop(receipt)


__all__ = [
    "ActionBallFullMdpResetGenesisAuthority",
    "ActionBallFullMdpActionEpochGenesisProjection",
    "ActionBallFullMdpResetGenesisError",
    "ActionBallFullMdpResetGenesisIssue",
    "ActionBallFullMdpResetGenesisReceipt",
    "PhysicalResetGenesisProjection",
    "discard_unpublished_action_ball_full_mdp_reset_genesis",
    "issue_action_ball_full_mdp_reset_genesis",
    "retire_failed_unpublished_action_ball_full_mdp_reset_genesis",
]
