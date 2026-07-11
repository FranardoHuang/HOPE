"""Versioned flat racket-command wire shared with the native A3 runner.

Schema 1 is the deployed legacy position/velocity command. Schema 2 appends the
world-frame physical striking-face-B normal (opponent-facing +X) and reserved
spin-rho scalar required by the 179-D ``deploy_parity_face179`` runner. After
the runner selects the clip it multiplies only this normal by the frozen
``mount_normal_sign_per_clip=[+1,-1]`` to recover the raw mount +Y/A command
used by the train bank and actor. Position/velocity are never sign-flipped. The
current Phase-1 contract freezes rho to exactly zero; a future spin curriculum
needs a new reviewed contract instead of silently assigning meaning to it.
"""

from __future__ import annotations

from collections.abc import Sequence
import math


RACKET_FLAT_SCHEMA_V1 = 1
RACKET_FLAT_SCHEMA_V2_FACE179 = 2
RACKET_FLAT_V1_SIZE = 12
RACKET_FLAT_V2_SIZE = 16
UNIT_NORMAL_TOL = 1.0e-6
FACE_NORMAL_MIN_X = 1.0e-6


def _finite_vec3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out  # type: ignore[return-value]


def pack_racket_command_flat(
    *,
    schema: int,
    valid: bool,
    swing_sign: float,
    position_w: Sequence[float],
    velocity_w: Sequence[float],
    time_to_strike: float,
    strike_time: float,
    frame_code: int = 0,
    normal_cmd_w: Sequence[float] | None = None,
    rho: float = 0.0,
) -> list[float]:
    """Build one canonical row; ``normal_cmd_w`` is physical face B, not raw mount A."""

    if schema not in (RACKET_FLAT_SCHEMA_V1, RACKET_FLAT_SCHEMA_V2_FACE179):
        raise ValueError(f"unsupported racket flat schema {schema}")
    if frame_code not in (0, 1):
        raise ValueError("frame_code must be 0 (world/table) or 1 (base_link)")
    if schema == RACKET_FLAT_SCHEMA_V2_FACE179 and frame_code != 0:
        raise ValueError(
            "Phase-1 schema 2 requires world/table frame_code=0; 3D base-link normal semantics "
            "are not frozen"
        )
    p = _finite_vec3(position_w, name="position_w")
    v = _finite_vec3(velocity_w, name="velocity_w")
    scalars = (float(swing_sign), float(time_to_strike), float(strike_time))
    if not all(math.isfinite(x) for x in scalars):
        raise ValueError("swing_sign/time_to_strike/strike_time must be finite")
    row = [
        float(schema),
        1.0 if valid else 0.0,
        scalars[0],
        *p,
        *v,
        scalars[1],
        scalars[2],
        float(frame_code),
    ]
    if schema == RACKET_FLAT_SCHEMA_V1:
        return row

    if normal_cmd_w is None:
        raise ValueError("schema 2 requires normal_cmd_w")
    n = _finite_vec3(normal_cmd_w, name="normal_cmd_w")
    norm = math.sqrt(sum(x * x for x in n))
    if abs(norm - 1.0) > UNIT_NORMAL_TOL:
        raise ValueError(
            f"schema 2 normal_cmd_w must already be unit length within {UNIT_NORMAL_TOL:g}"
        )
    if n[0] <= FACE_NORMAL_MIN_X:
        raise ValueError(
            "schema 2 normal_cmd_w must be opponent-facing in the HOPE world/table frame "
            f"(x > {FACE_NORMAL_MIN_X:g})"
        )
    rho_f = float(rho)
    if not math.isfinite(rho_f) or rho_f != 0.0:
        raise ValueError("Phase-1 schema 2 rho placeholder must be finite and exactly zero")
    return [*row, *n, rho_f]


def pack_invalid_racket_command_flat(*, schema: int) -> list[float]:
    """Return an exact finite invalid row that safely revokes a live command."""

    if schema == RACKET_FLAT_SCHEMA_V1:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if schema == RACKET_FLAT_SCHEMA_V2_FACE179:
        # Tail remains structurally valid even though valid=0; +X is the
        # canonical opponent-facing placeholder and rho stays reserved-zero.
        return [
            2.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ]
    raise ValueError(f"unsupported racket flat schema {schema}")


def pack_racket_command_flat_fail_closed(**kwargs) -> tuple[list[float], str | None]:
    """Pack a command; formal schema-2 contract errors become an invalid row.

    A bad planner payload must be visible to the subscriber as a revocation,
    not turn into a missing publication that leaves the previous face tuple
    eligible until its longer freshness timeout.
    """

    try:
        return pack_racket_command_flat(**kwargs), None
    except (TypeError, ValueError, OverflowError) as exc:
        if kwargs.get("schema") != RACKET_FLAT_SCHEMA_V2_FACE179:
            raise
        return pack_invalid_racket_command_flat(schema=RACKET_FLAT_SCHEMA_V2_FACE179), str(exc)
