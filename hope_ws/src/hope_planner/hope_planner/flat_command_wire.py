"""Versioned flat racket-command wire shared with the native A3 runner.

Schema 1 is the deployed legacy position/velocity command. Schema 2 appends the
world-frame physical striking-face-B normal (opponent-facing +X) and reserved
spin-rho scalar required by the 179-D ``deploy_parity_face179`` runner. After
the runner selects the clip it multiplies only this normal by the frozen
``mount_normal_sign_per_clip=[+1,-1]`` to recover the raw mount +Y/A command
used by the train bank and actor. Position/velocity are never sign-flipped. The
current Phase-1 contract freezes rho to exactly zero; a future spin curriculum
needs a new reviewed contract instead of silently assigning meaning to it.

Formal Gate3 uses schema 3, which extends that face-B row with a shared
base-control epoch, a strictly increasing command sequence, an exact
base-sequence reference and a same-host monotonic source stamp. Those payload
fields, rather than cross-topic receive order, define causality.

Schema 4 appends exact ``task_id`` and ``task_revision`` counters.  A positive
pair identifies one observed ball/task and orders every valid or invalid
refinement of it.  The zero/zero pair is reserved for an invalid global revoke
when no task identity can safely be asserted; it is never a valid command.
``command_sequence`` remains transport ordering and must not be reused as a
ball identity. Schema 3 is intentionally left byte-for-byte unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
import math


RACKET_FLAT_SCHEMA_V1 = 1
RACKET_FLAT_SCHEMA_V2_FACE179 = 2
RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH = 3
RACKET_FLAT_SCHEMA_V4_FACE179_TASK = 4
RACKET_FLAT_V1_SIZE = 12
RACKET_FLAT_V2_SIZE = 16
RACKET_FLAT_V3_SIZE = 20
RACKET_FLAT_V4_SIZE = 22
BASE_FLAT_SCHEMA_V1 = 1
BASE_FLAT_SCHEMA_V2_EPOCH = 2
BASE_FLAT_V1_SIZE = 9
BASE_FLAT_V2_SIZE = 12
MAX_EXACT_FLOAT64_INTEGER = (1 << 53) - 1
UNIT_NORMAL_TOL = 1.0e-6
FACE_NORMAL_MIN_X = 1.0e-6


def _finite_vec3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out  # type: ignore[return-value]


def _exact_counter(value: int | float | None, *, name: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} is required and must be an integer")
    number = float(value)
    if (not math.isfinite(number) or not number.is_integer()
            or number < 0.0 or number > MAX_EXACT_FLOAT64_INTEGER):
        raise ValueError(
            f"{name} must be an exact integer in [0,{MAX_EXACT_FLOAT64_INTEGER}]"
        )
    return int(number)


def _schema4_task_identity(
    task_id: int | float | None,
    task_revision: int | float | None,
    *,
    valid: bool,
) -> tuple[int, int]:
    """Validate an active positive pair or an invalid zero/zero revoke."""

    if not valid and task_id is None and task_revision is None:
        task_id = task_revision = 0
    task = _exact_counter(task_id, name="task_id")
    revision = _exact_counter(task_revision, name="task_revision")
    if (task == 0) != (revision == 0):
        raise ValueError("task_id and task_revision must both be zero or both be positive")
    if valid and task == 0:
        raise ValueError("valid schema-4 command requires positive task_id/task_revision")
    return task, revision


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
    control_epoch: int | float | None = None,
    command_sequence: int | float | None = None,
    base_sequence_ref: int | float | None = None,
    source_monotonic_s: float | None = None,
    task_id: int | float | None = None,
    task_revision: int | float | None = None,
) -> list[float]:
    """Build one canonical row; ``normal_cmd_w`` is physical face B, not raw mount A."""

    face_schema = schema in (
        RACKET_FLAT_SCHEMA_V2_FACE179,
        RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
    )
    if schema not in (
        RACKET_FLAT_SCHEMA_V1,
        RACKET_FLAT_SCHEMA_V2_FACE179,
        RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
    ):
        raise ValueError(f"unsupported racket flat schema {schema}")
    if frame_code not in (0, 1):
        raise ValueError("frame_code must be 0 (world/table) or 1 (base_link)")
    if face_schema and frame_code != 0:
        raise ValueError(
            "formal face schemas require world/table frame_code=0; 3D base-link normal semantics "
            "are not frozen"
        )
    p = _finite_vec3(position_w, name="position_w")
    v = _finite_vec3(velocity_w, name="velocity_w")
    scalars = (float(swing_sign), float(time_to_strike), float(strike_time))
    if not all(math.isfinite(x) for x in scalars):
        raise ValueError("swing_sign/time_to_strike/strike_time must be finite")
    if face_schema and valid and scalars[0] not in (-1.0, 1.0):
        raise ValueError(
            "formal face schemas require an explicit swing_sign of +1 forehand or -1 backhand"
        )
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
        raise ValueError("face schemas require normal_cmd_w")
    n = _finite_vec3(normal_cmd_w, name="normal_cmd_w")
    norm = math.sqrt(sum(x * x for x in n))
    if abs(norm - 1.0) > UNIT_NORMAL_TOL:
        raise ValueError(
            f"face-schema normal_cmd_w must already be unit length within {UNIT_NORMAL_TOL:g}"
        )
    if n[0] <= FACE_NORMAL_MIN_X:
        raise ValueError(
            "face-schema normal_cmd_w must be opponent-facing in the HOPE world/table frame "
            f"(x > {FACE_NORMAL_MIN_X:g})"
        )
    rho_f = float(rho)
    if not math.isfinite(rho_f) or rho_f != 0.0:
        raise ValueError("Phase-1 schema 2 rho placeholder must be finite and exactly zero")
    face_row = [*row, *n, rho_f]
    if schema == RACKET_FLAT_SCHEMA_V2_FACE179:
        return face_row
    epoch = _exact_counter(control_epoch, name="control_epoch")
    sequence = _exact_counter(command_sequence, name="command_sequence")
    base_ref = _exact_counter(base_sequence_ref, name="base_sequence_ref")
    stamp = float(source_monotonic_s) if source_monotonic_s is not None else math.nan
    if not math.isfinite(stamp) or stamp < 0.0:
        raise ValueError("source_monotonic_s must be finite and non-negative")
    formal_row = [*face_row, float(epoch), float(sequence), float(base_ref), stamp]
    if schema == RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH:
        return formal_row
    task, revision = _schema4_task_identity(
        task_id, task_revision, valid=bool(valid)
    )
    return [*formal_row, float(task), float(revision)]


def pack_base_pose_flat(
    *,
    schema: int,
    valid: bool,
    position_w: Sequence[float],
    quaternion_wxyz: Sequence[float],
    control_epoch: int | float | None = None,
    base_sequence: int | float | None = None,
    source_monotonic_s: float | None = None,
) -> list[float]:
    """Build one base-pose row; schema 2 carries the formal control epoch."""

    if schema not in (BASE_FLAT_SCHEMA_V1, BASE_FLAT_SCHEMA_V2_EPOCH):
        raise ValueError(f"unsupported base flat schema {schema}")
    p = _finite_vec3(position_w, name="position_w")
    if len(quaternion_wxyz) != 4:
        raise ValueError("quaternion_wxyz must contain exactly four values")
    q = tuple(float(v) for v in quaternion_wxyz)
    if not all(math.isfinite(v) for v in q):
        raise ValueError("quaternion_wxyz must be finite")
    q_norm = math.sqrt(sum(v * v for v in q))
    if q_norm < 1.0e-6:
        raise ValueError("quaternion_wxyz norm must be at least 1e-6")
    q = tuple(v / q_norm for v in q)
    row = [float(schema), 1.0 if valid else 0.0, *p, *q]
    if schema == BASE_FLAT_SCHEMA_V1:
        return row
    epoch = _exact_counter(control_epoch, name="control_epoch")
    sequence = _exact_counter(base_sequence, name="base_sequence")
    stamp = float(source_monotonic_s) if source_monotonic_s is not None else math.nan
    if not math.isfinite(stamp) or stamp < 0.0:
        raise ValueError("source_monotonic_s must be finite and non-negative")
    return [*row, float(epoch), float(sequence), stamp]


def pack_invalid_base_pose_flat(
    *,
    schema: int,
    control_epoch: int | float | None = None,
    base_sequence: int | float | None = None,
    source_monotonic_s: float | None = None,
) -> list[float]:
    """Return a canonical finite base revocation for the selected schema."""

    return pack_base_pose_flat(
        schema=schema,
        valid=False,
        position_w=(0.0, 0.0, 0.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        control_epoch=control_epoch,
        base_sequence=base_sequence,
        source_monotonic_s=source_monotonic_s,
    )


def pack_invalid_racket_command_flat(
    *,
    schema: int,
    control_epoch: int | float | None = None,
    command_sequence: int | float | None = None,
    base_sequence_ref: int | float | None = None,
    source_monotonic_s: float | None = None,
    task_id: int | float | None = None,
    task_revision: int | float | None = None,
) -> list[float]:
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
    if schema in (
        RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
    ):
        epoch = _exact_counter(control_epoch, name="control_epoch")
        sequence = _exact_counter(command_sequence, name="command_sequence")
        base_ref = _exact_counter(base_sequence_ref, name="base_sequence_ref")
        stamp = float(source_monotonic_s) if source_monotonic_s is not None else math.nan
        if not math.isfinite(stamp) or stamp < 0.0:
            raise ValueError("source_monotonic_s must be finite and non-negative")
        formal_row = [
            float(schema), 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            float(epoch), float(sequence), float(base_ref), stamp,
        ]
        if schema == RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH:
            return formal_row
        task, revision = _schema4_task_identity(
            task_id, task_revision, valid=False
        )
        return [*formal_row, float(task), float(revision)]
    raise ValueError(f"unsupported racket flat schema {schema}")


def pack_racket_command_flat_fail_closed(**kwargs) -> tuple[list[float], str | None]:
    """Pack a command; formal face-contract errors become an invalid row.

    A bad planner payload must be visible to the subscriber as a revocation,
    not turn into a missing publication that leaves the previous face tuple
    eligible until its longer freshness timeout.
    """

    try:
        return pack_racket_command_flat(**kwargs), None
    except (TypeError, ValueError, OverflowError) as exc:
        schema = kwargs.get("schema")
        if schema not in (
            RACKET_FLAT_SCHEMA_V2_FACE179,
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        ):
            raise
        invalid_kwargs = dict(
            schema=schema,
            control_epoch=kwargs.get("control_epoch"),
            command_sequence=kwargs.get("command_sequence"),
            base_sequence_ref=kwargs.get("base_sequence_ref"),
            source_monotonic_s=kwargs.get("source_monotonic_s"),
            task_id=kwargs.get("task_id"),
            task_revision=kwargs.get("task_revision"),
        )
        try:
            invalid = pack_invalid_racket_command_flat(**invalid_kwargs)
        except ValueError as identity_exc:
            if schema != RACKET_FLAT_SCHEMA_V4_FACE179_TASK:
                raise
            # A malformed task identity must not leave the prior command
            # eligible.  Publish an anonymous global revoke instead of
            # inventing an active task.  Causality metadata is still required;
            # if it is malformed this second pack raises.
            invalid_kwargs["task_id"] = 0
            invalid_kwargs["task_revision"] = 0
            invalid = pack_invalid_racket_command_flat(**invalid_kwargs)
            return invalid, f"{exc}; task identity revoked globally: {identity_exc}"
        return invalid, str(exc)
