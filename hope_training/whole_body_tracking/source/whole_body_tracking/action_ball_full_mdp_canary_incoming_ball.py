#!/usr/bin/env python3
"""Code-owned initial incoming ball for the no-save lean diagnostic.

This owner closes one narrow input gap: it materializes the first N incoming
ball rows without accepting caller tensors, ranges, digests, or identities.
It does not sample a continuing question stream, write the Isaac scene, bind
R06, authorize the runtime factory, or turn an ordinary later miss into a
producer fault.

The values are the component-wise centres of the production
``ContinuousQuestionCfg`` default contact-position and incoming-velocity
boxes.  Angular velocity is deliberately zero for this first deterministic
canary.  Config provenance describes that construction; it is not independent
physics evidence or a safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NoReturn

import torch


CANARY_SAVE_CHECKPOINTS = False
DIAGNOSTIC_UNAUTHORIZED = True
FORMAL_PROFILE = False
FORMAL_RUNTIME_AUTHORIZED = False
PRODUCTION_INTEGRATED = False
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

# A deterministic diagnostic lookback, not a sampled policy input.  It is used
# only to reject a code-default tuple that could not even have started beyond
# the net under the optimistic linear lower bound already used by the
# production question module.
CANARY_BIRTH_LOOKBACK_S = 0.5
BALL_BIRTH_NET_MARGIN_M = 0.05

_EXPECTED_POSITION_DEFAULT = (
    (0.50, 0.62),
    (-0.45, 0.45),
    (0.80, 1.20),
)
_EXPECTED_VELOCITY_DEFAULT = (
    (-4.5, -2.0),
    (-0.6, 0.6),
    (-1.0, 0.5),
)
_PROJECTION_TOKEN = object()


class ActionBallFullMdpCanaryIncomingBallError(RuntimeError):
    """The code-owned incoming-ball projection cannot be trusted."""


@dataclass(frozen=True)
class _IncomingBallRecord:
    selected_env_index: torch.Tensor
    ball_source_identity: torch.Tensor
    contact_position_env_m: torch.Tensor
    incoming_linear_velocity_world_mps: torch.Tensor
    incoming_angular_velocity_world_radps: torch.Tensor
    producer_fault: torch.Tensor


@dataclass(frozen=True)
class ActionBallFullMdpCanaryIncomingBallSnapshot:
    """Clone-only data returned by cold construction after token consumption."""

    selected_env_index: torch.Tensor
    ball_source_identity: torch.Tensor
    contact_position_env_m: torch.Tensor
    incoming_linear_velocity_world_mps: torch.Tensor
    incoming_angular_velocity_world_radps: torch.Tensor
    producer_fault: torch.Tensor
    diagnostic_unauthorized: bool = True
    formal_runtime_authorized: bool = False
    runtime_integrated: bool = False
    launch_authorized: bool = False


class ActionBallFullMdpCanaryIncomingBallProjection:
    """One-shot opaque identity carrying clone-only diagnostic tensors."""

    __slots__ = (
        "_selected_env_index",
        "_ball_source_identity",
        "_contact_position_env_m",
        "_incoming_linear_velocity_world_mps",
        "_incoming_angular_velocity_world_radps",
        "_producer_fault",
        "_owner_identity",
        "_token",
    )

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("canary incoming-ball projections are owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("canary incoming-ball projections are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("canary incoming-ball projections cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("canary incoming-ball projections cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("canary incoming-ball projections cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("canary incoming-ball projections cannot be serialized")

    @property
    def selected_env_index(self) -> torch.Tensor:
        return self._selected_env_index.clone()

    @property
    def ball_source_identity(self) -> torch.Tensor:
        return self._ball_source_identity.clone()

    @property
    def contact_position_env_m(self) -> torch.Tensor:
        return self._contact_position_env_m.clone()

    @property
    def incoming_linear_velocity_world_mps(self) -> torch.Tensor:
        return self._incoming_linear_velocity_world_mps.clone()

    @property
    def incoming_angular_velocity_world_radps(self) -> torch.Tensor:
        return self._incoming_angular_velocity_world_radps.clone()

    @property
    def producer_fault(self) -> torch.Tensor:
        return self._producer_fault.clone()

    @property
    def diagnostic_unauthorized(self) -> bool:
        return True

    @property
    def formal_runtime_authorized(self) -> bool:
        return False

    @property
    def runtime_integrated(self) -> bool:
        return False

    @property
    def launch_authorized(self) -> bool:
        return False


def _exact_device(value: object) -> torch.device:
    if type(value) is not torch.device:
        raise ActionBallFullMdpCanaryIncomingBallError(
            "device must be an exact torch.device, never a string or inferred device"
        )
    if value.type == "cpu":
        if value.index is not None:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "CPU canary device must not carry an index"
            )
    elif value.type == "cuda":
        if type(value.index) is not int or value.index < 0:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "CUDA canary device must carry one exact nonnegative index"
            )
        if not torch.cuda.is_available() or value.index >= torch.cuda.device_count():
            raise ActionBallFullMdpCanaryIncomingBallError(
                "indexed CUDA canary device is unavailable"
            )
    else:
        raise ActionBallFullMdpCanaryIncomingBallError(
            "lean canary supports only exact CPU or indexed CUDA devices"
        )
    return value


def _box3(value: object, *, label: str) -> tuple[tuple[float, float], ...]:
    if type(value) is not tuple or len(value) != 3:
        raise ActionBallFullMdpCanaryIncomingBallError(
            f"{label} must be an exact three-axis tuple"
        )
    rows: list[tuple[float, float]] = []
    for axis, pair in enumerate(value):
        if type(pair) is not tuple or len(pair) != 2:
            raise ActionBallFullMdpCanaryIncomingBallError(
                f"{label}[{axis}] must be an exact two-value tuple"
            )
        lower, upper = pair
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or not float(lower) < float(upper)
        ):
            raise ActionBallFullMdpCanaryIncomingBallError(
                f"{label}[{axis}] must be finite, ordered, and nondegenerate"
            )
        rows.append((float(lower), float(upper)))
    return tuple(rows)


def _production_default_centres(
    *, racket_cfg: object
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    from whole_body_tracking.tasks.table_tennis import geometry
    from whole_body_tracking.tasks.tracking.mdp.continuous_questions import (
        ContinuousQuestionCfg,
        ball_birth_x_lower_bound_m,
    )
    from whole_body_tracking.tasks.tracking.mdp.hope_commands import (
        RacketTargetCommandCfg,
    )

    if type(racket_cfg) is not RacketTargetCommandCfg:
        raise ActionBallFullMdpCanaryIncomingBallError(
            "racket_cfg must be the exact constructed RacketTargetCommandCfg"
        )

    config = ContinuousQuestionCfg()
    position_box = _box3(config.pos_range, label="ContinuousQuestionCfg.pos_range")
    velocity_box = _box3(config.vel_range, label="ContinuousQuestionCfg.vel_range")
    if (
        position_box != _EXPECTED_POSITION_DEFAULT
        or velocity_box != _EXPECTED_VELOCITY_DEFAULT
        or config.pos_range_per_clip is not None
        or config.vel_range_per_clip is not None
    ):
        raise ActionBallFullMdpCanaryIncomingBallError(
            "consumed ContinuousQuestionCfg incoming-ball defaults drifted"
        )

    contact = tuple((lower + upper) * 0.5 for lower, upper in position_box)
    velocity = tuple((lower + upper) * 0.5 for lower, upper in velocity_box)
    if velocity[0] >= 0.0:
        raise ActionBallFullMdpCanaryIncomingBallError(
            "code-owned incoming ball does not approach the player along -x"
        )

    near_x = float(racket_cfg.vb_table_near_x)
    net_x = near_x + float(geometry.NET_X)
    birth_x = ball_birth_x_lower_bound_m(
        contact[0], velocity[0], CANARY_BIRTH_LOOKBACK_S
    )
    if not all(math.isfinite(value) for value in (*contact, *velocity, net_x, birth_x)):
        raise ActionBallFullMdpCanaryIncomingBallError(
            "code-owned incoming-ball formula produced a nonfinite value"
        )
    if birth_x < net_x + BALL_BIRTH_NET_MARGIN_M:
        raise ActionBallFullMdpCanaryIncomingBallError(
            "code-owned incoming ball cannot start beyond the net at canary lookback"
        )
    return contact, velocity, birth_x


class ActionBallFullMdpCanaryIncomingBallOwner:
    """Single-use owner for deterministic first incoming-ball rows."""

    __slots__ = (
        "num_envs",
        "device",
        "_identity",
        "_record",
        "_issued_projection",
        "_consumed",
        "_poison_reason",
        "_birth_x_lower_bound_m",
    )

    def __init__(
        self, *, num_envs: int, device: torch.device, racket_cfg: object
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "initial incoming-ball num_envs must be a positive exact int"
            )
        exact_device = _exact_device(device)
        contact, velocity, birth_x = _production_default_centres(
            racket_cfg=racket_cfg
        )
        selected = torch.arange(num_envs, dtype=torch.int64, device=exact_device)
        source = torch.arange(1, num_envs + 1, dtype=torch.int64, device=exact_device)
        contact_tensor = torch.tensor(
            contact, dtype=torch.float32, device=exact_device
        ).repeat(num_envs, 1).contiguous()
        velocity_tensor = torch.tensor(
            velocity, dtype=torch.float32, device=exact_device
        ).repeat(num_envs, 1).contiguous()
        omega = torch.zeros(
            (num_envs, 3), dtype=torch.float32, device=exact_device
        )
        producer_fault = torch.zeros(
            (num_envs,), dtype=torch.bool, device=exact_device
        )
        self.num_envs = num_envs
        self.device = exact_device
        self._identity = object()
        self._record = _IncomingBallRecord(
            selected_env_index=selected,
            ball_source_identity=source,
            contact_position_env_m=contact_tensor,
            incoming_linear_velocity_world_mps=velocity_tensor,
            incoming_angular_velocity_world_radps=omega,
            producer_fault=producer_fault,
        )
        self._issued_projection: (
            ActionBallFullMdpCanaryIncomingBallProjection | None
        ) = None
        self._consumed = False
        self._poison_reason: str | None = None
        self._birth_x_lower_bound_m = birth_x

    def __copy__(self) -> NoReturn:
        raise TypeError("canary incoming-ball owners cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("canary incoming-ball owners cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("canary incoming-ball owners cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("canary incoming-ball owners cannot be serialized")

    @property
    def birth_x_lower_bound_m(self) -> float:
        return self._birth_x_lower_bound_m

    def provenance(self) -> dict[str, object]:
        """Describe the formula source without claiming independent safety proof."""

        return {
            "semantics": "diagnostic_config_provenance_not_safety_evidence",
            "input_source": (
                "production ContinuousQuestionCfg pos_range/vel_range defaults"
            ),
            "contact_formula": "componentwise midpoint of pos_range",
            "linear_velocity_formula": "componentwise midpoint of vel_range",
            "angular_velocity_formula": "zero vector for deterministic first canary",
            "birth_sanity": (
                "contact_x + abs(incoming_vx) * 0.5 >= canonical_net_x + 0.05"
            ),
            "diagnostic_unauthorized": True,
            "formal_profile": False,
            "formal_runtime_authorized": False,
            "production_integrated": False,
            "runtime_integrated": False,
            "launch_authorized": False,
        }

    def mint_projection(self) -> ActionBallFullMdpCanaryIncomingBallProjection:
        if self._poison_reason is not None:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "incoming-ball owner is poisoned: " + self._poison_reason
            )
        if self._issued_projection is not None:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "incoming-ball projection was already minted"
            )
        record = self._record
        projection = object.__new__(ActionBallFullMdpCanaryIncomingBallProjection)
        for name in (
            "selected_env_index",
            "ball_source_identity",
            "contact_position_env_m",
            "incoming_linear_velocity_world_mps",
            "incoming_angular_velocity_world_radps",
            "producer_fault",
        ):
            object.__setattr__(projection, "_" + name, getattr(record, name).clone())
        object.__setattr__(projection, "_owner_identity", self._identity)
        object.__setattr__(projection, "_token", _PROJECTION_TOKEN)
        self._issued_projection = projection
        return projection

    def _reject(self, reason: str) -> NoReturn:
        self._poison_reason = reason
        raise ActionBallFullMdpCanaryIncomingBallError(reason)

    def require_owned_projection(
        self, value: object
    ) -> ActionBallFullMdpCanaryIncomingBallSnapshot:
        """Consume one token and return owner-private clones.

        This is a cold lean construction boundary.  The equality checks may
        synchronize an indexed CUDA device and therefore this method must not
        be called from a physics, reward, observation, or optimizer hot path.
        The returned snapshot is data, not another authority token.
        """

        if self._poison_reason is not None:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "incoming-ball owner is poisoned: " + self._poison_reason
            )
        if self._consumed:
            raise ActionBallFullMdpCanaryIncomingBallError(
                "incoming-ball projection was already consumed"
            )
        if (
            type(value) is not ActionBallFullMdpCanaryIncomingBallProjection
            or value is not self._issued_projection
            or value._owner_identity is not self._identity
            or value._token is not _PROJECTION_TOKEN
        ):
            self._reject("incoming-ball projection is foreign or not owner-issued")

        record = self._record
        n = self.num_envs
        specifications = (
            ("selected_env_index", (n,), torch.int64),
            ("ball_source_identity", (n,), torch.int64),
            ("contact_position_env_m", (n, 3), torch.float32),
            ("incoming_linear_velocity_world_mps", (n, 3), torch.float32),
            ("incoming_angular_velocity_world_radps", (n, 3), torch.float32),
            ("producer_fault", (n,), torch.bool),
        )
        for name, shape, dtype in specifications:
            actual = getattr(value, "_" + name, None)
            expected = getattr(record, name)
            if (
                type(actual) is not torch.Tensor
                or tuple(actual.shape) != shape
                or actual.dtype is not dtype
                or actual.device != self.device
                or not actual.is_contiguous()
                or not torch.equal(actual, expected)
            ):
                self._reject(f"incoming-ball projection {name} was mutated or moved")
        if not bool(torch.all(value._ball_source_identity > 0)):
            self._reject("incoming-ball source identities are not positive")
        if bool(torch.any(value._producer_fault)):
            self._reject("code-owned initial incoming-ball producer faulted")
        self._consumed = True
        return ActionBallFullMdpCanaryIncomingBallSnapshot(
            selected_env_index=record.selected_env_index.clone(),
            ball_source_identity=record.ball_source_identity.clone(),
            contact_position_env_m=record.contact_position_env_m.clone(),
            incoming_linear_velocity_world_mps=(
                record.incoming_linear_velocity_world_mps.clone()
            ),
            incoming_angular_velocity_world_radps=(
                record.incoming_angular_velocity_world_radps.clone()
            ),
            producer_fault=record.producer_fault.clone(),
        )


def construct_action_ball_full_mdp_canary_incoming_ball_owner(
    *, num_envs: int, device: torch.device, racket_cfg: object
) -> ActionBallFullMdpCanaryIncomingBallOwner:
    """Construct the generic-N owner without accepting numerical payload."""

    return ActionBallFullMdpCanaryIncomingBallOwner(
        num_envs=num_envs,
        device=device,
        racket_cfg=racket_cfg,
    )


__all__ = [
    "ActionBallFullMdpCanaryIncomingBallError",
    "ActionBallFullMdpCanaryIncomingBallOwner",
    "ActionBallFullMdpCanaryIncomingBallProjection",
    "ActionBallFullMdpCanaryIncomingBallSnapshot",
    "BALL_BIRTH_NET_MARGIN_M",
    "CANARY_BIRTH_LOOKBACK_S",
    "CANARY_SAVE_CHECKPOINTS",
    "DIAGNOSTIC_UNAUTHORIZED",
    "FORMAL_PROFILE",
    "FORMAL_RUNTIME_AUTHORIZED",
    "LAUNCH_AUTHORIZED",
    "PRODUCTION_INTEGRATED",
    "RUNTIME_INTEGRATED",
    "construct_action_ball_full_mdp_canary_incoming_ball_owner",
]
