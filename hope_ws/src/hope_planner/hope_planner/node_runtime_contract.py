"""Pure runtime contracts used by the ROS planner node.

This module deliberately has no ROS imports.  The geometry, cadence, and side
selection rules can therefore be unit-tested on a development machine before
they are wired into the vendor MuJoCo or robot launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


_QUATERNION_EPS = 1.0e-9
MAX_EXACT_FLOAT64_INTEGER = (1 << 53) - 1


class FormalWireExhaustion(RuntimeError):
    """The reserved terminal-invalid counter must be spent before shutdown."""


class FormalBaseBarrierRejection(ValueError):
    """A candidate is old relative to an existing revoke barrier.

    This is not new bad-sample evidence. Callers must keep the original barrier
    fixed so a constant-latency source can eventually advance beyond it.
    """


@dataclass
class FormalWireCounters:
    """Non-wrapping formal epoch/sequences with MAX reserved for one revoke."""

    control_epoch: int = 1
    racket_sequence: int = 0
    base_sequence: int = 0
    exhausted: bool = False

    def _next(self, field: str) -> int:
        if self.exhausted:
            raise RuntimeError("formal flat wire is permanently exhausted")
        value = int(getattr(self, field))
        if value >= MAX_EXACT_FLOAT64_INTEGER - 1:
            raise FormalWireExhaustion(f"{field} exhausted")
        value += 1
        setattr(self, field, value)
        return value

    def next_racket(self) -> int:
        return self._next("racket_sequence")

    def next_base(self) -> int:
        return self._next("base_sequence")

    def advance_epoch(self) -> int:
        return self._next("control_epoch")

    def reserve_terminal_invalid(self) -> tuple[int, int, int]:
        if self.exhausted:
            raise RuntimeError("formal flat wire is permanently exhausted")
        self.control_epoch = MAX_EXACT_FLOAT64_INTEGER
        self.racket_sequence = MAX_EXACT_FLOAT64_INTEGER
        self.base_sequence = MAX_EXACT_FLOAT64_INTEGER
        self.exhausted = True
        return self.control_epoch, self.racket_sequence, self.base_sequence


def base_pose_is_fresh(
    source_monotonic_s: float | None,
    now_monotonic_s: float,
    max_age_s: float,
) -> bool:
    """Return whether a mapped base source sample is usable for formal side.

    Formal schema 3 maps the ROS header source age into process-monotonic time;
    publication/receive time must never grant the source a new lease. Missing,
    non-finite, future-dated, or over-age samples fail closed. An invalid
    configured age is a startup contract error rather than a silent
    always-stale planner.
    """

    max_age_s = float(max_age_s)
    if not math.isfinite(max_age_s) or max_age_s < 0.0:
        raise ValueError("base_pose_max_age_s must be finite and non-negative")
    now_monotonic_s = float(now_monotonic_s)
    if source_monotonic_s is None:
        return False
    source_monotonic_s = float(source_monotonic_s)
    if not (math.isfinite(source_monotonic_s) and math.isfinite(now_monotonic_s)):
        return False
    age_s = now_monotonic_s - source_monotonic_s
    return 0.0 <= age_s <= max_age_s


@dataclass
class FormalBaseLease:
    """One-shot source-age lease for the formal schema-3 base.

    ``expire`` and ``invalidate`` return ``True`` exactly once per accepted
    lease.  The ROS wrapper uses that transition to publish both canonical
    base and racket revocations; merely clearing Python-side geometry is not a
    control-plane revoke.
    """

    base_source_monotonic_s: float | None = None
    active: bool = False

    def accept(
        self,
        base_source_monotonic_s: float,
        *,
        now_monotonic_s: float,
        max_age_s: float,
    ) -> None:
        base_source_monotonic_s = float(base_source_monotonic_s)
        if (not math.isfinite(base_source_monotonic_s)
                or base_source_monotonic_s < 0.0):
            raise ValueError("mapped base source monotonic time must be finite and non-negative")
        if not base_pose_is_fresh(
            base_source_monotonic_s, now_monotonic_s, max_age_s
        ):
            raise ValueError("mapped base source lease is stale")
        self.base_source_monotonic_s = base_source_monotonic_s
        self.active = True

    def fresh(self, now_monotonic_s: float, max_age_s: float) -> bool:
        return self.active and base_pose_is_fresh(
            self.base_source_monotonic_s, now_monotonic_s, max_age_s
        )

    def invalidate(self) -> bool:
        transitioned = self.active
        self.active = False
        self.base_source_monotonic_s = None
        return transitioned

    def expire(self, now_monotonic_s: float, max_age_s: float) -> bool:
        if not self.active or self.fresh(now_monotonic_s, max_age_s):
            return False
        return self.invalidate()


@dataclass
class FormalBaseSourceState:
    """Pure admission state for the schema-3 base source lease.

    ``expire_before_admission`` is deliberately a separate first phase.  When
    it transitions, the caller publishes the dual-topic revoke and advances
    the wire epoch before attempting to install the callback's candidate.  A
    delayed candidate whose mapped source is at/before that barrier cannot
    recover the new epoch merely because it arrived after the revoke.
    """

    lease: FormalBaseLease = field(default_factory=FormalBaseLease)
    revoke_barrier_monotonic_s: float = -1.0

    def expire_before_admission(
        self, now_monotonic_s: float, max_age_s: float
    ) -> bool:
        now_monotonic_s = float(now_monotonic_s)
        if not math.isfinite(now_monotonic_s) or now_monotonic_s < 0.0:
            raise ValueError("base admission monotonic time must be finite and non-negative")
        if not self.lease.expire(now_monotonic_s, max_age_s):
            return False
        self.revoke_barrier_monotonic_s = max(
            self.revoke_barrier_monotonic_s, now_monotonic_s
        )
        return True

    def revoke(self, now_monotonic_s: float, *, force: bool = False) -> bool:
        now_monotonic_s = float(now_monotonic_s)
        if not math.isfinite(now_monotonic_s) or now_monotonic_s < 0.0:
            raise ValueError("base revoke monotonic time must be finite and non-negative")
        transitioned = self.lease.invalidate()
        if not (transitioned or force):
            return False
        self.revoke_barrier_monotonic_s = max(
            self.revoke_barrier_monotonic_s, now_monotonic_s
        )
        return True

    def validate_candidate(
        self,
        base_source_monotonic_s: float,
        *,
        now_monotonic_s: float,
        max_age_s: float,
    ) -> None:
        base_source_monotonic_s = float(base_source_monotonic_s)
        if (not math.isfinite(base_source_monotonic_s)
                or base_source_monotonic_s < 0.0):
            raise ValueError("mapped base source monotonic time must be finite and non-negative")
        if base_source_monotonic_s <= self.revoke_barrier_monotonic_s:
            raise FormalBaseBarrierRejection(
                "mapped base source predates the latest revoke barrier"
            )
        if not base_pose_is_fresh(
            base_source_monotonic_s, now_monotonic_s, max_age_s
        ):
            raise ValueError("mapped base source lease is stale")

    def accept_source(
        self,
        base_source_monotonic_s: float,
        *,
        now_monotonic_s: float,
        max_age_s: float,
    ) -> None:
        self.validate_candidate(
            base_source_monotonic_s,
            now_monotonic_s=now_monotonic_s,
            max_age_s=max_age_s,
        )
        self.lease.accept(
            base_source_monotonic_s,
            now_monotonic_s=now_monotonic_s,
            max_age_s=max_age_s,
        )

    def ready_for_source(
        self,
        base_source_monotonic_s: float,
        now_monotonic_s: float,
        max_age_s: float,
    ) -> bool:
        try:
            source = float(base_source_monotonic_s)
        except (TypeError, ValueError):
            return False
        return (
            self.lease.active
            and math.isfinite(source)
            and self.lease.base_source_monotonic_s == source
            and self.lease.fresh(now_monotonic_s, max_age_s)
        )


@dataclass
class FormalBasePosePlausibilityGuard:
    """Immutable formal-base source-frame and source-time continuity contract."""

    min_source: tuple[float, float, float] = (-3.0, -3.0, 0.4)
    max_source: tuple[float, float, float] = (3.0, 3.0, 1.5)
    linear_slack_m: float = 0.05
    max_linear_speed_mps: float = 8.0
    angular_slack_rad: float = 0.15
    max_angular_speed_radps: float = 12.0
    last_position_source: np.ndarray | None = None
    last_quaternion_wxyz: np.ndarray | None = None
    last_source_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        lo = np.asarray(self.min_source, dtype=float)
        hi = np.asarray(self.max_source, dtype=float)
        scalars = np.asarray(
            [
                self.linear_slack_m,
                self.max_linear_speed_mps,
                self.angular_slack_rad,
                self.max_angular_speed_radps,
            ],
            dtype=float,
        )
        if (
            lo.shape != (3,)
            or hi.shape != (3,)
            or not np.isfinite(lo).all()
            or not np.isfinite(hi).all()
            or not np.all(lo < hi)
            or not np.isfinite(scalars).all()
            or np.any(scalars < 0.0)
        ):
            raise ValueError("formal base plausibility contract is invalid")

    def validate(
        self,
        position_source: Sequence[float],
        quaternion_wxyz: Sequence[float],
        source_monotonic_s: float,
    ) -> None:
        p = np.asarray(position_source, dtype=float)
        q = np.asarray(quaternion_wxyz, dtype=float)
        source = float(source_monotonic_s)
        if (
            p.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
            or not math.isfinite(source)
            or source < 0.0
        ):
            raise ValueError("formal base pose/source must be finite and shaped exactly")
        lo = np.asarray(self.min_source, dtype=float)
        hi = np.asarray(self.max_source, dtype=float)
        if np.any(p < lo) or np.any(p > hi):
            raise ValueError("formal base pose lies outside the frozen source-frame workspace")
        q_norm = float(np.linalg.norm(q))
        if q_norm < _QUATERNION_EPS:
            raise ValueError("formal base quaternion norm is too small")
        q = q / q_norm
        if self.last_source_monotonic_s is None:
            return
        dt = source - self.last_source_monotonic_s
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("formal base pose source time did not advance")
        assert self.last_position_source is not None
        assert self.last_quaternion_wxyz is not None
        linear_jump = float(np.linalg.norm(p - self.last_position_source))
        if linear_jump > self.linear_slack_m + self.max_linear_speed_mps * dt:
            raise ValueError("formal base displacement exceeds source-time bound")
        dot = float(abs(np.dot(q, self.last_quaternion_wxyz)))
        angular_jump = 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
        if angular_jump > self.angular_slack_rad + self.max_angular_speed_radps * dt:
            raise ValueError("formal base angular jump exceeds source-time bound")

    def commit(
        self,
        position_source: Sequence[float],
        quaternion_wxyz: Sequence[float],
        source_monotonic_s: float,
    ) -> None:
        self.validate(position_source, quaternion_wxyz, source_monotonic_s)
        q = np.asarray(quaternion_wxyz, dtype=float)
        self.last_position_source = np.asarray(position_source, dtype=float).copy()
        self.last_quaternion_wxyz = q / np.linalg.norm(q)
        self.last_source_monotonic_s = float(source_monotonic_s)


@dataclass(frozen=True)
class FormalSourceFrameContract:
    """Exact ROS header frames required by formal schema-3 source topics."""

    ball_frame_id: str
    base_frame_id: str
    common_frame_required: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("ball", self.ball_frame_id),
            ("base", self.base_frame_id),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(
                    f"formal {label} source frame_id must be a non-empty exact string"
                )
        if type(self.common_frame_required) is not bool:
            raise ValueError("formal_common_frame_required must be an exact boolean")
        if not self.common_frame_required:
            raise ValueError(
                "formal common-frame checking cannot be disabled without a "
                "content-bound explicit rigid transform"
            )
        if self.ball_frame_id != self.base_frame_id:
            raise ValueError(
                "formal ball/base sources must share one common frame; "
                f"got ball={self.ball_frame_id!r}, base={self.base_frame_id!r}"
            )

    @staticmethod
    def _validate(actual: object, expected: str, label: str) -> None:
        if not isinstance(actual, str) or actual != expected:
            raise ValueError(
                f"{label} source frame_id mismatch: expected {expected!r}, got {actual!r}"
            )

    def validate_ball(self, actual: object) -> None:
        self._validate(actual, self.ball_frame_id, "ball")

    def validate_base(self, actual: object) -> None:
        self._validate(actual, self.base_frame_id, "base")


def validate_formal_source_clock_mode(
    use_sim_time: object, configured_clock_mode: object
) -> str:
    """Fail closed unless ROS time authority and formal mapping mode agree."""

    if type(use_sim_time) is not bool:
        raise ValueError("formal schema 3 requires use_sim_time to be an exact boolean")
    expected = "sim" if use_sim_time else "system"
    if not isinstance(configured_clock_mode, str) or configured_clock_mode != expected:
        raise ValueError(
            "formal_source_clock_mode must exactly match use_sim_time "
            f"({expected!r})"
        )
    return expected


@dataclass
class SourceStampGuard:
    """Reject stale, future, duplicate, or regressing ROS source samples.

    The flat-wire monotonic stamp proves when the planner published a row.  It
    cannot by itself prove that the PoseStamped/PoseArray used to make that row
    was current.  Formal schema 3 therefore admits only strictly newer source
    header stamps whose age is bounded in the node's active ROS clock domain.
    A rejected source never advances the guard, so replay cannot refresh age.
    """

    max_age_s: float
    future_tolerance_s: float = 0.02
    last_accepted_stamp_s: float | None = None

    def __post_init__(self) -> None:
        self.max_age_s = float(self.max_age_s)
        self.future_tolerance_s = float(self.future_tolerance_s)
        if not math.isfinite(self.max_age_s) or self.max_age_s < 0.0:
            raise ValueError("source stamp max age must be finite and non-negative")
        if (not math.isfinite(self.future_tolerance_s)
                or self.future_tolerance_s < 0.0):
            raise ValueError("source stamp future tolerance must be finite and non-negative")

    def validate(self, source_stamp_s: float, now_s: float) -> None:
        source_stamp_s = float(source_stamp_s)
        now_s = float(now_s)
        if not (math.isfinite(source_stamp_s) and math.isfinite(now_s)):
            raise ValueError("source and current ROS stamps must be finite")
        if source_stamp_s < 0.0:
            raise ValueError("source stamp must be non-negative")
        age_s = now_s - source_stamp_s
        if age_s < -self.future_tolerance_s:
            raise ValueError("source stamp is too far in the future")
        if age_s > self.max_age_s:
            raise ValueError("source stamp is stale")
        if (self.last_accepted_stamp_s is not None
                and source_stamp_s <= self.last_accepted_stamp_s):
            raise ValueError("source stamp duplicated or regressed")

    def commit(self, source_stamp_s: float) -> None:
        source_stamp_s = float(source_stamp_s)
        if (not math.isfinite(source_stamp_s) or source_stamp_s < 0.0
                or (self.last_accepted_stamp_s is not None
                    and source_stamp_s <= self.last_accepted_stamp_s)):
            raise ValueError("source stamp cannot be committed")
        self.last_accepted_stamp_s = source_stamp_s

    def accept(self, source_stamp_s: float, now_s: float) -> None:
        self.validate(source_stamp_s, now_s)
        self.commit(source_stamp_s)


def ros_source_to_monotonic(
    source_stamp_s: float, now_ros_s: float, now_monotonic_s: float
) -> float:
    """Map same-host ROS source age onto CLOCK_MONOTONIC without refreshing it."""

    source_stamp_s = float(source_stamp_s)
    now_ros_s = float(now_ros_s)
    now_monotonic_s = float(now_monotonic_s)
    if not all(math.isfinite(v) for v in (source_stamp_s, now_ros_s, now_monotonic_s)):
        raise ValueError("source/ROS/monotonic clocks must be finite")
    if source_stamp_s < 0.0:
        raise ValueError("source stamp must be non-negative")
    return now_monotonic_s - max(0.0, now_ros_s - source_stamp_s)


def ros_stamp_fields_to_seconds(sec: int, nanosec: int) -> float:
    """Validate builtin_interfaces/Time fields before float conversion."""

    if isinstance(sec, bool) or isinstance(nanosec, bool):
        raise ValueError("ROS source stamp fields must be integers")
    sec = int(sec)
    nanosec = int(nanosec)
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError("ROS source stamp fields are out of range")
    return float(sec) + float(nanosec) * 1.0e-9


def corrected_base_pose(
    marker_position_w: Sequence[float],
    marker_quaternion_wxyz: Sequence[float],
    marker_to_base_xyz: Sequence[float],
    *,
    policy_z_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a marker-cluster pose into the policy base pose.

    ``marker_to_base_xyz`` is expressed in marker-local axes.  It must be
    rotated by the marker orientation before it is added to the world marker
    position.  A missing/all-zero orientation cannot define either that offset
    rotation or the policy's yaw frame, so it fails closed together with every
    malformed/non-finite input.

    Returns ``(base_position_w, normalized_quaternion_wxyz)``.  The configured
    policy-frame Z offset is applied only to the returned base position.
    """

    marker = np.asarray(marker_position_w, dtype=float)
    quat = np.asarray(marker_quaternion_wxyz, dtype=float)
    offset = np.asarray(marker_to_base_xyz, dtype=float)
    if marker.shape != (3,) or offset.shape != (3,) or quat.shape != (4,):
        raise ValueError("marker/base vectors must be xyz and quaternion must be wxyz")
    z_offset = float(policy_z_offset)
    if not (np.isfinite(marker).all() and np.isfinite(offset).all()
            and np.isfinite(quat).all() and math.isfinite(z_offset)):
        raise ValueError("marker pose, marker-to-base offset, and policy Z offset must be finite")

    quat_norm = float(np.linalg.norm(quat))
    if quat_norm <= _QUATERNION_EPS:
        raise ValueError("marker/base orientation quaternion is missing")
    quat_unit = quat / quat_norm

    qw, qx, qy, qz = quat_unit
    rotation_w_marker = np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )
    base = marker + rotation_w_marker @ offset
    base[2] += z_offset
    return base, quat_unit


def base_yaw_relative_y(
    intercept_position_w: Sequence[float],
    corrected_base_position_w: Sequence[float],
    corrected_base_quaternion_wxyz: Sequence[float],
) -> float:
    """Return ``(yaw(base)^-1 * (intercept_w - base_w)).y``.

    This is deliberately the same yaw-only frame used by the C++ runner for
    ``tgt_b``.  Subtracting world Y alone is not equivalent once the base has a
    non-zero heading because world-X reach then contributes to base-frame Y.
    """

    intercept = np.asarray(intercept_position_w, dtype=float)
    base = np.asarray(corrected_base_position_w, dtype=float)
    quat = np.asarray(corrected_base_quaternion_wxyz, dtype=float)
    if intercept.shape != (3,) or base.shape != (3,) or quat.shape != (4,):
        raise ValueError("intercept/base vectors must be xyz and quaternion must be wxyz")
    if not (np.isfinite(intercept).all() and np.isfinite(base).all()
            and np.isfinite(quat).all()):
        raise ValueError("intercept, corrected base, and base orientation must be finite")
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm <= _QUATERNION_EPS:
        raise ValueError("corrected base orientation quaternion is missing")
    qw, qx, qy, qz = quat / quat_norm
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    delta = intercept - base
    return float(-math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1])


@dataclass
class SolveCadence:
    """Deterministic solve admission for a high-rate measurement callback.

    A rejected sample is still ingested by the estimator, but the expensive
    Stage-2/3 solve and all command publications are skipped.  Consequently the
    planner/ROS latest-value cache remains byte-for-byte unchanged; a skipped
    sample never refreshes a stale command's subscriber-side age.
    """

    period_s: float = 0.0
    last_solve_t: float | None = None

    def __post_init__(self) -> None:
        self.period_s = float(self.period_s)
        if not math.isfinite(self.period_s) or self.period_s < 0.0:
            raise ValueError("solve_period_s must be finite and non-negative")

    def admit(self, timestamp_s: float) -> bool:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("planner timestamp must be finite")
        if self.period_s == 0.0 or self.last_solve_t is None:
            self.last_solve_t = timestamp_s
            return True
        elapsed = timestamp_s - self.last_solve_t
        boundary_tolerance = max(1.0e-12, self.period_s * 1.0e-12)
        if elapsed < 0.0 or elapsed >= self.period_s - boundary_tolerance:
            # A clock regression is not silently throttled against a future
            # timestamp.  The planner's existing exception guard remains the
            # authority for any bad trajectory fit that follows.
            self.last_solve_t = timestamp_s
            return True
        return False


@dataclass
class SwingSideSelector:
    """Stateful FH/BH selection in the current base-yaw frame.

    Sign convention is the deploy contract: ``+1`` forehand, ``-1``
    backhand.  The Schmitt band prevents centimetre-scale prediction noise
    from changing clips near the split.
    """

    split_y: float = 0.0
    hysteresis_y: float = 0.04
    sign: float = 0.0

    def __post_init__(self) -> None:
        self.split_y = float(self.split_y)
        self.hysteresis_y = float(self.hysteresis_y)
        if not math.isfinite(self.split_y):
            raise ValueError("swing_side_split_y must be finite")
        if not math.isfinite(self.hysteresis_y) or self.hysteresis_y < 0.0:
            raise ValueError("swing_side_hysteresis_y must be finite and non-negative")
        if self.sign not in (-1.0, 0.0, 1.0):
            raise ValueError("initial swing sign must be -1, 0, or +1")

    def select(
        self,
        intercept_position_w: Sequence[float],
        corrected_base_position_w: Sequence[float],
        corrected_base_quaternion_wxyz: Sequence[float],
    ) -> float:
        relative_y = base_yaw_relative_y(
            intercept_position_w,
            corrected_base_position_w,
            corrected_base_quaternion_wxyz,
        )
        low = self.split_y - self.hysteresis_y
        high = self.split_y + self.hysteresis_y
        if self.sign > 0.5:
            if relative_y > high:
                self.sign = -1.0
        elif self.sign < -0.5:
            if relative_y < low:
                self.sign = 1.0
        else:
            self.sign = 1.0 if relative_y < self.split_y else -1.0
        return self.sign
