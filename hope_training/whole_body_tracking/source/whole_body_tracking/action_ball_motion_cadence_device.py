#!/usr/bin/env python3
"""Construction-bound Motion cadence authority for Device-R05.

The authority consumes Motion's already-published opaque observation token;
callers cannot provide task, cadence, action, or tick tensors.  Device-R05
supplies the live state it owns.  Cross-owner value mismatches become one
device-resident producer-fault lane and therefore enter the same reveal-batch
CENSOR boundary; this module never synchronizes tensor values to authorize a
write.

Outcome and ball handles are deliberately absent: Device-R05 reserves them
only after exact question selection and publishes them only in its terminal
ACCEPT after-image.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import hashlib
import json
from pathlib import Path
from types import FunctionType, MappingProxyType, MethodType
from typing import NoReturn

import torch

try:
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_full_mdp_action_strata as _action_strata,
    )
except ImportError:
    import action_ball_full_mdp_action_strata as _action_strata

try:
    import action_ball_continuous_runtime_transaction_device as _r05
except ImportError:  # Installed package import.
    from . import action_ball_continuous_runtime_transaction_device as _r05
try:
    import action_ball_continuous_successor as _c01
    import action_ball_recovery_sequence as _c02
except ImportError:  # Installed package import.
    from . import action_ball_continuous_successor as _c01
    from . import action_ball_recovery_sequence as _c02


RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

CADENCE_FAULT_RESET_GENERATION = 1 << 51
CADENCE_FAULT_SCHEDULED_ORDINAL = 1 << 52
CADENCE_FAULT_MOTION_TASK = 1 << 53
CADENCE_FAULT_COUNTER_EXHAUSTED = 1 << 55
CADENCE_FAULT_NOT_SYNCHRONIZED_REVEAL = 1 << 56

_I64_MAX = torch.iinfo(torch.int64).max
_CONSTRUCTION_KEY = object()
_DIAGNOSTIC_PARENT_KEY = object()
_MOTION_PROFILE_KIND = (
    "whole_body_tracking.action_ball_continuous_motion_projection_v1"
)
_READY_REFERENCE_KIND = "completed_action_frame0_zero_velocity_v1"
_DIAGNOSTIC_FROZEN_AT_STEP = 0
_DIAGNOSTIC_SEQUENCE_ORIGIN_STEP = 0
_DIAGNOSTIC_FIRST_REVEAL_STEP = 2
_DIAGNOSTIC_UPCOMING_ACTION_SLOT = 0
_DIAGNOSTIC_DEADLINE_OFFSET_STEPS = 2
# The slowest code-pinned action must finish its complete question-owned
# Motion suffix before C02's final recovery-eligible tick, followed by one
# complete hidden tick.  The timing helper re-derives the catalog maximum from
# the pinned manifest and fixed policy step; no caller numeric or copied magic
# value can shorten the schedule.


def _diagnostic_cadence_steps() -> int:
    """Derive the recurring cadence from the pinned timing owner once."""

    try:
        import action_ball_full_mdp_diagnostic_action_timing as timing
    except ImportError:  # Installed package import.
        from . import action_ball_full_mdp_diagnostic_action_timing as timing
    return (
        timing.diagnostic_catalog_max_task_close_ticks()
        + _c02.RECOVERY_END_OFFSET_TICKS
        + 2
    )
_DIAGNOSTIC_SCHEDULED_SHOT_COUNT = 4
_COMMANDS_SOURCE = (
    Path(__file__).resolve().parent
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "commands.py"
).resolve()


class MotionCadenceAuthorityError(RuntimeError):
    """The exact Motion publication cannot form a trustworthy D05 row."""


class MotionCadenceAuthorityConflictError(MotionCadenceAuthorityError):
    """A foreign owner/token or incompatible structural binding was used."""


class MotionCadenceProductionSourceHold(MotionCadenceAuthorityError):
    """No production C01/C02 authority instance exists for Motion yet."""


class DiagnosticMotionProfileReceipt:
    """Opaque receipt for one source-bound, no-save lean profile."""

    __slots__ = ()

    def __new__(cls) -> NoReturn:
        raise TypeError("diagnostic Motion profile receipts are owner-issued")


@dataclass(frozen=True)
class DiagnosticMotionParentActionIdentity:
    """Parent-retained code-owned action identity after exact Motion bind."""

    authority: "DiagnosticMotionParentScheduleAuthority"
    motion_owner: object
    action_slot: int
    action_uid: int
    action_uids: tuple[int, ...]


def _fail(message: str) -> NoReturn:
    raise MotionCadenceAuthorityConflictError(message)


def _require_action_uid_table(
    value: object,
    *,
    expected_length: int,
) -> tuple[int, ...]:
    if (
        type(expected_length) is not int
        or expected_length <= 0
        or type(value) is not tuple
        or len(value) != expected_length
    ):
        _fail("diagnostic Motion action identity table differs")
    if (
        any(type(uid) is not int or uid <= 0 for uid in value)
        or len(set(value)) != expected_length
    ):
        _fail("diagnostic Motion action identity table differs")
    return value


def _require_exact_motion_owner(value: object) -> object:
    value_type = type(value)
    try:
        source = inspect.getsourcefile(value_type)
    except (TypeError, OSError):
        source = None
    if (
        value_type.__name__ != "MotionCommand"
        or value_type.__qualname__ != "MotionCommand"
        or source is None
        or Path(source).resolve() != _COMMANDS_SOURCE
    ):
        _fail("Motion cadence owner exact class/source identity differs")
    expected = (
        "action_ball_continuous_current_projection",
        "action_ball_continuous_motion_observation_projection",
        "require_owned_action_ball_continuous_motion_observation",
        "_action_ball_continuous_code_owned_action_uids",
        "bind_action_ball_continuous_parent_authorities",
    )
    for name in expected:
        bound = getattr(value, name, None)
        declared = getattr(value_type, name, None)
        try:
            declared_source = inspect.getsourcefile(declared)
        except (TypeError, OSError):
            declared_source = None
        if (
            type(bound) is not MethodType
            or bound.__self__ is not value
            or bound.__func__ is not declared
            or type(declared) is not FunctionType
            or declared.__name__ != name
            or declared.__qualname__ != f"MotionCommand.{name}"
            or declared_source is None
            or Path(declared_source).resolve() != _COMMANDS_SOURCE
        ):
            _fail(f"Motion cadence owner method differs: {name}")
    return value


def _require_tensor(
    value: object,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device != device
        or value.dtype is not dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
    ):
        _fail(
            f"{name} must be contiguous {dtype} on {device} with shape {shape}"
        )
    return value


def _canonical_json_bytes(value: object) -> bytes:
    # Match MotionCommand's config-integrity encoding exactly.  This remains
    # an integrity checksum; parent authority comes from the opaque owner.
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _source_sha256(module: object, *, label: str) -> str:
    source_name = getattr(module, "__file__", None)
    if type(source_name) is not str:
        _fail(f"{label} contract source is absent")
    source = Path(source_name).resolve()
    if not source.is_file():
        _fail(f"{label} contract source is absent")
    return hashlib.sha256(source.read_bytes()).hexdigest()


class DiagnosticMotionParentScheduleAuthority:
    """Code-owned cardinality-generic canary over C01/C02 typed cadence.

    This is deliberately not a production C01 or C02 authority: no complete
    question/target/ball tape or recovery-row owner exists in the production
    graph.  The two hashes are diagnostic source bindings, not safety or
    launch facts.  The profile self-hash remains only a config-integrity check.
    """

    __slots__ = (
        "_receipt",
        "_schedule",
        "_profile",
        "_bound_motion",
        "_bound_action_slot",
        "_bound_action_uid",
        "_bound_action_uids",
    )

    def __init__(self, key: object) -> None:
        if key is not _DIAGNOSTIC_PARENT_KEY:
            raise TypeError(
                "diagnostic Motion schedule authorities are factory-constructed"
            )
        c01_source_sha256 = _source_sha256(_c01, label="C01")
        c02_source_sha256 = _source_sha256(_c02, label="C02")
        cadence_steps = _diagnostic_cadence_steps()
        schedule_authority_sha256 = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "kind": "diagnostic_c01_typed_schedule_source_v1",
                    "diagnostic_unauthorized": True,
                    "c01_source_sha256": c01_source_sha256,
                    "clock_kind": _c01.CLOCK_KIND,
                    "frozen_at_step": _DIAGNOSTIC_FROZEN_AT_STEP,
                    "sequence_origin_step": _DIAGNOSTIC_SEQUENCE_ORIGIN_STEP,
                    "first_reveal_step": _DIAGNOSTIC_FIRST_REVEAL_STEP,
                    "cadence_steps": cadence_steps,
                    "deadline_offset_steps": (
                        _DIAGNOSTIC_DEADLINE_OFFSET_STEPS
                    ),
                    "scheduled_shot_count": (
                        _DIAGNOSTIC_SCHEDULED_SHOT_COUNT
                    ),
                    "upcoming_action_slot": (
                        _DIAGNOSTIC_UPCOMING_ACTION_SLOT
                    ),
                }
            )
        ).hexdigest()
        clock_epoch_sha256 = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "kind": "diagnostic_episode_clock_source_v1",
                    "diagnostic_unauthorized": True,
                    "c01_source_sha256": c01_source_sha256,
                    "clock_kind": _c01.CLOCK_KIND,
                }
            )
        ).hexdigest()
        schedule = _c01.FrozenCadenceReceipt(
            clock_kind=_c01.CLOCK_KIND,
            clock_epoch_sha256=clock_epoch_sha256,
            schedule_authority_sha256=schedule_authority_sha256,
            frozen_at_step=_DIAGNOSTIC_FROZEN_AT_STEP,
            sequence_origin_step=_DIAGNOSTIC_SEQUENCE_ORIGIN_STEP,
            first_reveal_step=_DIAGNOSTIC_FIRST_REVEAL_STEP,
            cadence_steps=cadence_steps,
            deadline_offset_steps=_DIAGNOSTIC_DEADLINE_OFFSET_STEPS,
            scheduled_shot_count=_DIAGNOSTIC_SCHEDULED_SHOT_COUNT,
        )
        continuous_source_binding = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "kind": "diagnostic_c01_source_binding_v1",
                    "diagnostic_unauthorized": True,
                    "c01_source_sha256": c01_source_sha256,
                    "frozen_cadence_receipt_sha256": (
                        schedule.canonical_sha256
                    ),
                }
            )
        ).hexdigest()
        recovery_source_binding = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "kind": "diagnostic_c02_source_binding_v1",
                    "diagnostic_unauthorized": True,
                    "c02_source_sha256": c02_source_sha256,
                    "policy_rate_hz": _c02.POLICY_RATE_HZ,
                    "recovery_start_offset_ticks": (
                        _c02.RECOVERY_START_OFFSET_TICKS
                    ),
                    "recovery_end_offset_ticks": (
                        _c02.RECOVERY_END_OFFSET_TICKS
                    ),
                    "frozen_cadence_receipt_sha256": (
                        schedule.canonical_sha256
                    ),
                }
            )
        ).hexdigest()
        payload = {
            "schema_version": 1,
            "kind": _MOTION_PROFILE_KIND,
            "clock_kind": _c01.CLOCK_KIND,
            "continuous_contract_authority_sha256": (
                continuous_source_binding
            ),
            "recovery_contract_authority_sha256": recovery_source_binding,
            "ready_reference_kind": _READY_REFERENCE_KIND,
        }
        self._receipt = object.__new__(DiagnosticMotionProfileReceipt)
        self._schedule = schedule
        self._profile = {
            **payload,
            "canonical_sha256": hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest(),
        }
        self._bound_motion = None
        self._bound_action_slot = None
        self._bound_action_uid = None
        self._bound_action_uids = None

    def __copy__(self):
        raise TypeError("diagnostic Motion schedule authorities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("diagnostic Motion schedule authorities cannot be copied")

    def __reduce__(self):
        raise TypeError("diagnostic Motion schedule authorities cannot be saved")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("diagnostic Motion schedule authorities cannot be saved")

    def issue(self) -> DiagnosticMotionProfileReceipt:
        return self._receipt

    def require_owned_motion_profile(
        self, receipt: object
    ) -> dict[str, object]:
        if receipt is not self._receipt:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion profile receipt is foreign"
            )
        return dict(self._profile)

    def bind_exact_parent_schedule(
        self,
        motion_owner: object,
        receipt: object,
    ) -> None:
        if receipt is not self._receipt:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion profile receipt is foreign"
            )
        motion = _require_exact_motion_owner(motion_owner)
        num_envs = getattr(motion, "num_envs", None)
        if type(num_envs) is not int or num_envs <= 0:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion num_envs must be a positive exact int"
            )
        clip_id = getattr(motion, "clip_id", None)
        # The exact Motion method resolves the pinned catalog before broker
        # bind, and later requires the broker to prove the same complete tuple.
        try:
            frozen_action_uids = (
                motion._action_ball_continuous_code_owned_action_uids()
            )
        except Exception as exc:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic upcoming Motion action identity differs"
            ) from exc
        if (
            type(clip_id) is not torch.Tensor
            or clip_id.dtype != torch.int64
            or tuple(clip_id.shape) != (num_envs,)
            or clip_id.device != torch.device(getattr(motion, "device", "cpu"))
        ):
            raise MotionCadenceAuthorityConflictError(
                "diagnostic upcoming Motion action identity differs"
            )
        frozen_action_uids = _require_action_uid_table(
            frozen_action_uids,
            expected_length=getattr(motion.motion, "num_segments", -1),
        )
        if len(frozen_action_uids) <= _DIAGNOSTIC_UPCOMING_ACTION_SLOT:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic upcoming Motion action identity differs"
            )
        upcoming_action_uid = frozen_action_uids[
            _DIAGNOSTIC_UPCOMING_ACTION_SLOT
        ]
        retained = getattr(
            motion, "_action_ball_continuous_motion_profile", None
        )
        if retained is None or dict(retained) != self._profile:
            raise MotionCadenceAuthorityConflictError(
                "Motion was not constructed from the retained canary profile"
            )
        if self._bound_motion is not None and self._bound_motion is not motion:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion parent schedule cannot be rebound"
            )
        if self._bound_motion is motion:
            if (
                self._bound_action_uids != frozen_action_uids
                or self._bound_action_slot
                != _DIAGNOSTIC_UPCOMING_ACTION_SLOT
                or self._bound_action_uid != upcoming_action_uid
            ):
                raise MotionCadenceAuthorityConflictError(
                    "diagnostic Motion parent action identity drifted"
                )
            return
        schedule = self._schedule
        continuous_sha256 = self._profile[
            "continuous_contract_authority_sha256"
        ]
        recovery_sha256 = self._profile[
            "recovery_contract_authority_sha256"
        ]
        expected_projection = {
            "frozen_at_step": schedule.frozen_at_step,
            "sequence_origin_step": schedule.sequence_origin_step,
            "first_reveal_step": schedule.first_reveal_step,
            "cadence_steps": schedule.cadence_steps,
            "deadline_offset_steps": schedule.deadline_offset_steps,
            "upcoming_action_slot": _DIAGNOSTIC_UPCOMING_ACTION_SLOT,
            "upcoming_action_uid": upcoming_action_uid,
        }
        try:
            motion.bind_action_ball_continuous_parent_authorities(
                continuous_contract_authority_sha256=continuous_sha256,
                recovery_contract_authority_sha256=recovery_sha256,
                **expected_projection,
            )
        except Exception as exc:
            raise MotionCadenceAuthorityConflictError(
                "exact Motion parent schedule bind failed"
            ) from exc

        # Retain parent truth only after the exact Motion binder has published
        # its complete after-image.  This is the one construction boundary at
        # which reading Motion's private schedule/binding is necessary; later
        # identity projections use only the parent's frozen tuple.
        motion_projection = getattr(
            motion, "_action_ball_continuous_schedule_projection", None
        )
        motion_binding = getattr(
            motion, "_action_ball_continuous_parent_authority_binding", None
        )
        if (
            type(motion_projection) is not MappingProxyType
            or dict(motion_projection) != expected_projection
            or any(
                type(value) is not int
                for value in motion_projection.values()
            )
            or type(motion_binding) is not tuple
            or len(motion_binding) != 4
            or motion_binding[0] is not retained
            or motion_binding[1] is not motion_projection
            or motion_binding[2] != continuous_sha256
            or motion_binding[3] != recovery_sha256
        ):
            raise MotionCadenceAuthorityConflictError(
                "exact Motion parent schedule bind after-image differs"
            )
        self._bound_motion = motion
        self._bound_action_slot = _DIAGNOSTIC_UPCOMING_ACTION_SLOT
        self._bound_action_uid = upcoming_action_uid
        self._bound_action_uids = frozen_action_uids

    def project_bound_action_identity(
        self,
        receipt: object,
        *,
        motion_owner: object,
    ) -> DiagnosticMotionParentActionIdentity:
        """Project the exact parent-retained action table after cold bind."""

        if receipt is not self._receipt:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion profile receipt is foreign"
            )
        if self._bound_motion is None:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion parent schedule is not bound"
            )
        motion = _require_exact_motion_owner(motion_owner)
        if self._bound_motion is not motion:
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion parent action identity owner is foreign"
            )
        action_uids = _require_action_uid_table(
            self._bound_action_uids,
            expected_length=getattr(motion.motion, "num_segments", -1),
        )
        action_slot = self._bound_action_slot
        action_uid = self._bound_action_uid
        if (
            type(action_slot) is not int
            or action_slot < 0
            or action_slot >= len(action_uids)
            or type(action_uid) is not int
            or action_uid <= 0
            or action_uids[action_slot] != action_uid
        ):
            raise MotionCadenceAuthorityConflictError(
                "diagnostic Motion parent action slot/UID identity drifted"
            )
        return DiagnosticMotionParentActionIdentity(
            authority=self,
            motion_owner=motion,
            action_slot=action_slot,
            action_uid=action_uid,
            action_uids=action_uids,
        )


class ActionBallMotionCadenceAuthority:
    """Exact adapter over one construction-bound MotionCommand owner."""

    __slots__ = ("_motion", "_action_family_catalog")

    def __init__(self, key: object, *, motion_owner: object) -> None:
        if key is not _CONSTRUCTION_KEY:
            raise TypeError("Motion cadence authorities are factory-constructed")
        self._motion = _require_exact_motion_owner(motion_owner)
        try:
            uids = self._motion._action_ball_continuous_code_owned_action_uids()
            families = tuple(self._motion.cfg.clip_family_per_clip)
            codes = tuple(
                _action_strata.STROKE_FAMILY_NAMES.index(family)
                for family in families
            )
            self._action_family_catalog = (
                _action_strata.ActionStrokeFamilyCatalog(uids, codes)
            )
        except Exception as exc:
            raise MotionCadenceAuthorityConflictError(
                "Motion construction UID/family catalog differs"
            ) from exc

    def __copy__(self):
        raise TypeError("Motion cadence authorities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Motion cadence authorities cannot be copied")

    def __reduce__(self):
        raise TypeError("Motion cadence authorities cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Motion cadence authorities cannot be serialized")

    def project_action_stroke_family_catalog(
        self,
    ) -> _action_strata.ActionStrokeFamilyCatalog:
        return self._action_family_catalog.clone()

    def project_current_action_epoch_rows(self) -> object:
        """Return Motion's exact current full-N cadence/close projection.

        The construction-bound Motion owner is the only input.  In particular,
        this call accepts no selected indices, mask, verdict, task identity, or
        receipt.  ``reveal_due`` is the sole due mask; ``closed_mask`` and
        ``close_reason`` are Motion mechanics for the current tick, not a
        second ActionEpoch lifecycle or full-shot-key writer.
        """

        method = getattr(
            self._motion, "action_ball_continuous_current_projection", None
        )
        if (
            not callable(method)
            or getattr(method, "__self__", None) is not self._motion
            or getattr(method, "__func__", None)
            is not getattr(
                type(self._motion),
                "action_ball_continuous_current_projection",
                None,
            )
        ):
            raise MotionCadenceAuthorityConflictError(
                "Motion current row projection method differs"
            )
        return method()

def construct_production_motion_cadence_authority(
    *, motion_owner: object
) -> ActionBallMotionCadenceAuthority:
    """Bind exactly one real MotionCommand; no Protocol-shaped substitute."""

    return ActionBallMotionCadenceAuthority(
        _CONSTRUCTION_KEY, motion_owner=motion_owner
    )


def build_action_ball_full_mdp_diagnostic_motion_profile() -> tuple[
    DiagnosticMotionParentScheduleAuthority,
    DiagnosticMotionProfileReceipt,
    dict[str, object],
]:
    """Build the source-bound no-save lean profile before CommandManager.

    The returned mapping may populate ``MotionCommandCfg`` before the command
    owner exists.  The authority and opaque receipt must then survive until
    ``bind_exact_parent_schedule`` runs on that exact constructed Motion owner.
    This function is diagnostic evidence only and never authorizes launch.
    """

    owner = DiagnosticMotionParentScheduleAuthority(_DIAGNOSTIC_PARENT_KEY)
    receipt = owner.issue()
    return owner, receipt, owner.require_owned_motion_profile(receipt)


def construct_production_motion_parent_schedule_authority() -> NoReturn:
    """Keep production blocked until real C01 and C02 instances exist."""

    raise MotionCadenceProductionSourceHold(
        "production Motion cadence lacks owner-issued C01 four-shot and C02 "
        "recovery-sequence authority instances; source pins and a profile "
        "self-hash are diagnostic integrity only"
    )


__all__ = (
    "ActionBallMotionCadenceAuthority",
    "CADENCE_FAULT_COUNTER_EXHAUSTED",
    "CADENCE_FAULT_MOTION_TASK",
    "CADENCE_FAULT_NOT_SYNCHRONIZED_REVEAL",
    "CADENCE_FAULT_RESET_GENERATION",
    "CADENCE_FAULT_SCHEDULED_ORDINAL",
    "DIAGNOSTIC_UNAUTHORIZED",
    "DiagnosticMotionParentActionIdentity",
    "DiagnosticMotionParentScheduleAuthority",
    "DiagnosticMotionProfileReceipt",
    "LAUNCH_AUTHORIZED",
    "MotionCadenceAuthorityConflictError",
    "MotionCadenceAuthorityError",
    "MotionCadenceProductionSourceHold",
    "RUNTIME_INTEGRATED",
    "build_action_ball_full_mdp_diagnostic_motion_profile",
    "construct_production_motion_parent_schedule_authority",
    "construct_production_motion_cadence_authority",
)
