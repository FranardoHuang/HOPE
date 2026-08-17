"""N=2 construction binding for the disposable ActionBall diagnostic.

This is deliberately *not* a capacity receipt.  Formal physical-flight
transactions require ``FrozenFlightCapacityReceipt``; this opaque object merely
lets the same scene port and Physical owner allocate the code-owned two-body
plant before a formal C/H/K authority exists.  It cannot be caller-created,
serialized, used as a digest, or promoted into a portable transaction.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import NoReturn


DIAGNOSTIC_CAPACITY_KIND = "action_ball_full_mdp_code_owned_diagnostic_n2_capacity_v1"
DIAGNOSTIC_FLIGHT_CAPACITY = 2
DIAGNOSTIC_UNAUTHORIZED = True
NO_CHECKPOINT = True
NO_LATE_LAUNCH = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

_TOKEN = object()


class ActionBallFullMdpDiagnosticCapacityError(RuntimeError):
    """The narrow N=2 construction-only capacity binding was misused."""


class DiagnosticN2CapacityBinding:
    """Factory-issued identity binding to one exact diagnostic scene spec."""

    __slots__ = ("_scene_spec", "_token")

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("diagnostic N=2 capacity bindings are factory-issued only")

    @property
    def flight_capacity(self) -> int:
        return DIAGNOSTIC_FLIGHT_CAPACITY

    @property
    def diagnostic_unauthorized(self) -> bool:
        return DIAGNOSTIC_UNAUTHORIZED

    @property
    def no_checkpoint(self) -> bool:
        return NO_CHECKPOINT

    @property
    def no_late_launch(self) -> bool:
        return NO_LATE_LAUNCH

    def __reduce__(self) -> NoReturn:
        raise TypeError("diagnostic N=2 capacity bindings cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("diagnostic N=2 capacity bindings cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("diagnostic N=2 capacity bindings cannot be copied")


def _require_exact_diagnostic_scene_spec(scene_spec: object) -> None:
    """Check the actual class definition, rather than a caller flag or digest."""

    scene_type = type(scene_spec)
    module = sys.modules.get(scene_type.__module__)
    source = inspect.getsourcefile(scene_type)
    if (
        scene_type.__name__ != "ActionBallFullMdpDiagnosticBallSceneSpec"
        or module is None
        or getattr(module, scene_type.__name__, None) is not scene_type
        or source is None
        or Path(source).name != "action_ball_full_mdp_ball_scene.py"
    ):
        raise ActionBallFullMdpDiagnosticCapacityError(
            "diagnostic capacity requires the exact diagnostic N=2 scene spec"
        )


def construct_diagnostic_n2_capacity_binding(
    scene_spec: object,
) -> DiagnosticN2CapacityBinding:
    """Bind the fixed diagnostic capacity to the exact pre-super scene object."""

    _require_exact_diagnostic_scene_spec(scene_spec)
    if (
        scene_spec.kind
        != "action_ball_full_mdp_code_owned_diagnostic_ball_scene_spec_v1"
        or scene_spec.capacity_authority_kind != DIAGNOSTIC_CAPACITY_KIND
        or scene_spec.flight_capacity != DIAGNOSTIC_FLIGHT_CAPACITY
        or scene_spec.formal_capacity_receipt_sha256 is not None
        or hasattr(scene_spec, "capacity_receipt_sha256")
    ):
        raise ActionBallFullMdpDiagnosticCapacityError(
            "diagnostic scene cannot become a formal capacity authority"
        )
    result = object.__new__(DiagnosticN2CapacityBinding)
    object.__setattr__(result, "_scene_spec", scene_spec)
    object.__setattr__(result, "_token", _TOKEN)
    return result


def require_diagnostic_n2_capacity_binding(
    value: object,
    *,
    scene_spec: object,
) -> DiagnosticN2CapacityBinding:
    """Require this exact binding and its exact pre-super scene instance."""

    if (
        type(value) is not DiagnosticN2CapacityBinding
        or value._token is not _TOKEN
        or value._scene_spec is not scene_spec
    ):
        raise ActionBallFullMdpDiagnosticCapacityError(
            "diagnostic N=2 capacity binding is stale or foreign"
        )
    # Re-check the exact class and immutable scene constraints at consumption.
    _require_exact_diagnostic_scene_spec(scene_spec)
    if (
        scene_spec.kind
        != "action_ball_full_mdp_code_owned_diagnostic_ball_scene_spec_v1"
        or scene_spec.capacity_authority_kind != DIAGNOSTIC_CAPACITY_KIND
        or scene_spec.flight_capacity != DIAGNOSTIC_FLIGHT_CAPACITY
        or scene_spec.formal_capacity_receipt_sha256 is not None
        or hasattr(scene_spec, "capacity_receipt_sha256")
    ):
        raise ActionBallFullMdpDiagnosticCapacityError(
            "diagnostic scene cannot become a formal capacity authority"
        )
    return value


__all__ = [
    "ActionBallFullMdpDiagnosticCapacityError",
    "DIAGNOSTIC_CAPACITY_KIND",
    "DIAGNOSTIC_FLIGHT_CAPACITY",
    "DIAGNOSTIC_UNAUTHORIZED",
    "DiagnosticN2CapacityBinding",
    "LAUNCH_AUTHORIZED",
    "NO_CHECKPOINT",
    "NO_LATE_LAUNCH",
    "RUNTIME_INTEGRATED",
    "construct_diagnostic_n2_capacity_binding",
    "require_diagnostic_n2_capacity_binding",
]
