"""Pure exactly-once ball/task lifecycle for formal planner schema 4.

The state machine is deliberately independent of ROS and of the trajectory
solver.  A solver changing between valid and invalid is evidence about one
already-observed ball, not permission to allocate another task.  Likewise,
an inbound/plane predicate may jitter after a task starts without changing its
identity.  Only an explicit close followed by an explicit safe rearm can
allocate the next task in the same control epoch.

Process restart and control-epoch change both begin disarmed.  The caller must
prove its external no-ball/new-serve rearm barrier before calling
``explicit_rearm``; this module never infers that safety fact from a newly
valid solver result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .flat_command_wire import MAX_EXACT_FLOAT64_INTEGER


class FormalTaskState(str, Enum):
    """Control-visible states of one planner producer session."""

    DISARMED = "disarmed"
    ARMED = "armed"
    ACTIVE = "active"
    CLOSED_WAIT_REARM = "closed_wait_rearm"


class FormalTaskEpochError(ValueError):
    """A control epoch or task counter is not an exact monotonic integer."""


class FormalTaskTransitionError(RuntimeError):
    """A lifecycle transition would weaken the exactly-once contract."""


class FormalTaskCounterExhaustion(RuntimeError):
    """The current epoch cannot encode another exact task or revision."""


class FormalBallTrackDecision(str, Enum):
    """Boundary evidence emitted by :class:`FormalBallTrackBoundary`."""

    NONE = "none"
    SAFE_REARM = "safe_rearm"
    CLOSE_ACTIVE = "close_active"
    CLOSE_AND_REARM = "close_and_rearm"


@dataclass
class FormalBallTrackBoundary:
    """Conservative physical-ball boundary detector for the ROS producer.

    Transport sequence numbers cannot identify a ball.  This helper accepts
    only two positive new-ball proofs: a sustained no-ball gap followed by a
    clearly inbound track, or a contact-sized velocity discontinuity whose
    post-contact velocity is clearly inbound.  A live task closes after the
    measured ball passes the strike plane, becomes clearly outbound at a
    discontinuity, or exceeds the latest predicted strike deadline plus a
    bounded grace period.

    The helper never allocates ids and never infers solver validity.  It only
    tells ``FormalTaskLifecycle`` when the external physical boundary is safe
    enough to call ``explicit_rearm`` or ``close``.
    """

    no_ball_rearm_s: float = 0.10
    plane_close_margin_m: float = 0.02
    deadline_close_grace_s: float = 0.08
    inbound_vx_threshold_mps: float = -0.30
    outbound_vx_threshold_mps: float = 0.30
    absent_since_s: float | None = None
    gap_rearm_pending: bool = False
    latest_strike_deadline_s: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "no_ball_rearm_s",
            "plane_close_margin_m",
            "deadline_close_grace_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)
        for name in ("inbound_vx_threshold_mps", "outbound_vx_threshold_mps"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.inbound_vx_threshold_mps >= 0.0:
            raise ValueError("inbound_vx_threshold_mps must be negative")
        if self.outbound_vx_threshold_mps <= 0.0:
            raise ValueError("outbound_vx_threshold_mps must be positive")

    @staticmethod
    def _time(value: float, *, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def reset_epoch(self) -> None:
        """Discard every old-ball proof at a producer/control restart."""

        self.absent_since_s = None
        self.gap_rearm_pending = False
        self.latest_strike_deadline_s = None

    def observe_absent(
        self, source_time_s: float, *, task_active: bool
    ) -> FormalBallTrackDecision:
        """Record one trustworthy source tick with no ball pose."""

        now = self._time(source_time_s, name="source_time_s")
        if self.absent_since_s is None:
            self.absent_since_s = now
        elif now < self.absent_since_s:
            raise ValueError("no-ball source time regressed")
        if now - self.absent_since_s >= self.no_ball_rearm_s:
            self.gap_rearm_pending = True
        if (
            task_active
            and self.latest_strike_deadline_s is not None
            and now
            > self.latest_strike_deadline_s + self.deadline_close_grace_s
        ):
            self.latest_strike_deadline_s = None
            return FormalBallTrackDecision.CLOSE_ACTIVE
        return FormalBallTrackDecision.NONE

    def observe_present(
        self,
        source_time_s: float,
        *,
        ball_x_m: float,
        ball_vx_mps: float | None,
        discontinuity_detected: bool,
        task_active: bool,
        strike_plane_x_m: float,
        predicted_strike_time_s: float | None,
    ) -> FormalBallTrackDecision:
        """Classify one ball-track sample without changing task identity."""

        now = self._time(source_time_s, name="source_time_s")
        x = float(ball_x_m)
        plane = float(strike_plane_x_m)
        if not (math.isfinite(x) and math.isfinite(plane)):
            raise ValueError("ball/strike-plane positions must be finite")
        if type(discontinuity_detected) is not bool:
            raise TypeError("discontinuity_detected must be an exact boolean")
        if type(task_active) is not bool:
            raise TypeError("task_active must be an exact boolean")
        vx = None if ball_vx_mps is None else float(ball_vx_mps)
        if vx is not None and not math.isfinite(vx):
            raise ValueError("ball_vx_mps must be finite when present")
        if predicted_strike_time_s is not None:
            deadline = self._time(
                predicted_strike_time_s, name="predicted_strike_time_s"
            )
            if task_active:
                self.latest_strike_deadline_s = deadline

        # A present sample ends the raw gap, but a completed gap proof stays
        # pending until a direction estimate proves that this is inbound.
        self.absent_since_s = None
        inbound = vx is not None and vx <= self.inbound_vx_threshold_mps
        outbound = vx is not None and vx >= self.outbound_vx_threshold_mps
        new_ball = inbound and (
            self.gap_rearm_pending or discontinuity_detected
        )
        passed_plane = x <= plane - self.plane_close_margin_m
        deadline_passed = (
            self.latest_strike_deadline_s is not None
            and now
            > self.latest_strike_deadline_s + self.deadline_close_grace_s
        )
        outbound_contact = discontinuity_detected and outbound
        # A sustained trustworthy no-ball gap followed by a clearly inbound
        # track is itself the strongest available physical new-ball proof.  If
        # the old task is still ACTIVE, close and rearm atomically; otherwise a
        # fast next serve arriving before the old predicted deadline would be
        # consumed as another revision of the previous physical ball.
        close = task_active and (
            new_ball or passed_plane or deadline_passed or outbound_contact
        )

        if new_ball:
            self.gap_rearm_pending = False
        if close:
            self.latest_strike_deadline_s = None
            if new_ball:
                return FormalBallTrackDecision.CLOSE_AND_REARM
            return FormalBallTrackDecision.CLOSE_ACTIVE
        if not task_active and new_ball:
            return FormalBallTrackDecision.SAFE_REARM
        return FormalBallTrackDecision.NONE


@dataclass(frozen=True)
class FormalTaskRevision:
    """Identity to append to one valid or invalid schema-4 publication."""

    control_epoch: int
    task_id: int
    task_revision: int
    valid: bool


def _exact_epoch(value: int | float, *, name: str = "control_epoch") -> int:
    if isinstance(value, bool):
        raise FormalTaskEpochError(f"{name} must be an exact integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalTaskEpochError(f"{name} must be an exact integer") from exc
    if (
        not number.is_integer()
        or number < 0.0
        or number > MAX_EXACT_FLOAT64_INTEGER
    ):
        raise FormalTaskEpochError(
            f"{name} must be an exact integer in "
            f"[0,{MAX_EXACT_FLOAT64_INTEGER}]"
        )
    return int(number)


def _exact_bool(value: bool, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


@dataclass
class FormalTaskLifecycle:
    """Allocate one task id per explicitly armed inbound ball.

    ``publish`` owns task revisions for both valid and invalid planner rows.
    Once ACTIVE, ``inbound_track_ready`` is intentionally ignored: a noisy
    plane/velocity predicate cannot split one physical ball into two tasks.
    ``solver_valid`` only controls the row's valid bit.
    """

    control_epoch: int | None = None
    state: FormalTaskState = FormalTaskState.DISARMED
    last_task_id: int = 0
    active_task_id: int | None = None
    active_revision: int = 0
    exhausted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, FormalTaskState):
            try:
                self.state = FormalTaskState(self.state)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid formal task lifecycle state") from exc
        if self.control_epoch is not None:
            self.control_epoch = _exact_epoch(self.control_epoch)
        for field_name in ("last_task_id", "active_revision"):
            value = _exact_epoch(getattr(self, field_name), name=field_name)
            setattr(self, field_name, value)
        if self.active_task_id is not None:
            self.active_task_id = _exact_epoch(
                self.active_task_id, name="active_task_id"
            )
            if self.active_task_id == 0:
                raise ValueError("active_task_id must be positive")
        if type(self.exhausted) is not bool:
            raise TypeError("exhausted must be an exact boolean")
        self._validate_snapshot()

    def _validate_snapshot(self) -> None:
        if self.state is FormalTaskState.ACTIVE:
            if self.active_task_id is None or self.active_revision <= 0:
                raise ValueError("ACTIVE task requires positive identity and revision")
            if self.active_task_id != self.last_task_id:
                raise ValueError("ACTIVE task must be the latest allocated task")
        elif self.active_task_id is not None or self.active_revision != 0:
            raise ValueError("non-ACTIVE state cannot retain an active task")
        if self.exhausted and self.state is not FormalTaskState.DISARMED:
            raise ValueError("an exhausted lifecycle must be DISARMED")

    def observe_epoch(self, control_epoch: int | float) -> bool:
        """Observe an epoch, disarming on first sight or strict advancement.

        Returns ``True`` when the caller crossed a session boundary.  A
        regressed epoch is malformed/replayed input and raises rather than
        silently resurrecting an earlier task namespace.
        """

        epoch = _exact_epoch(control_epoch)
        if self.control_epoch is None:
            self.control_epoch = epoch
            self._reset_epoch_state()
            return True
        if epoch < self.control_epoch:
            raise FormalTaskEpochError(
                f"control_epoch regressed from {self.control_epoch} to {epoch}"
            )
        if epoch == self.control_epoch:
            return False
        self.control_epoch = epoch
        self._reset_epoch_state()
        return True

    def _reset_epoch_state(self) -> None:
        self.state = FormalTaskState.DISARMED
        self.last_task_id = 0
        self.active_task_id = None
        self.active_revision = 0
        self.exhausted = False

    def disarm(self, control_epoch: int | float) -> None:
        """Fail closed in the current or a newer epoch without auto-rearm."""

        self.observe_epoch(control_epoch)
        self.state = FormalTaskState.DISARMED
        self.active_task_id = None
        self.active_revision = 0

    def explicit_rearm(
        self,
        control_epoch: int | float,
        *,
        no_ball_or_new_serve_confirmed: bool,
    ) -> None:
        """Arm only after the caller proves the external rearm barrier."""

        # Session change is safety-significant even if the rearm payload is
        # malformed: disarm the old task namespace before validating the new
        # event's auxiliary fields.
        self.observe_epoch(control_epoch)
        safe = _exact_bool(
            no_ball_or_new_serve_confirmed,
            name="no_ball_or_new_serve_confirmed",
        )
        if not safe:
            raise FormalTaskTransitionError(
                "explicit rearm requires a confirmed no-ball/new-serve barrier"
            )
        if self.exhausted:
            raise FormalTaskCounterExhaustion(
                "task identity space is exhausted for this control epoch"
            )
        if self.state is FormalTaskState.ACTIVE:
            raise FormalTaskTransitionError("cannot rearm while a task is active")
        # Repeating the same explicit safe proof while already armed is
        # idempotent and cannot allocate or consume a task.
        self.state = FormalTaskState.ARMED

    def publish(
        self,
        control_epoch: int | float,
        *,
        inbound_track_ready: bool,
        solver_valid: bool,
    ) -> FormalTaskRevision | None:
        """Return identity for one publication, or ``None`` while disarmed.

        In ARMED state, the first stable inbound observation allocates a task
        even when the solver is invalid.  That makes invalid->valid recovery a
        revision of the same ball.  In ACTIVE state, inbound predicate jitter
        is ignored and every publication advances exactly one revision.
        """

        crossed_epoch = self.observe_epoch(control_epoch)
        inbound = _exact_bool(inbound_track_ready, name="inbound_track_ready")
        valid = _exact_bool(solver_valid, name="solver_valid")
        if crossed_epoch:
            return None  # a new/restarted producer is always disarmed first
        if self.state in (
            FormalTaskState.DISARMED,
            FormalTaskState.CLOSED_WAIT_REARM,
        ):
            return None
        if self.state is FormalTaskState.ARMED:
            if not inbound:
                return None
            self._start_task()
        return self._advance_revision(valid=valid)

    def _start_task(self) -> None:
        if self.exhausted or self.last_task_id >= MAX_EXACT_FLOAT64_INTEGER:
            self._exhaust("task_id exhausted")
        self.last_task_id += 1
        self.active_task_id = self.last_task_id
        self.active_revision = 0
        self.state = FormalTaskState.ACTIVE

    def _advance_revision(self, *, valid: bool) -> FormalTaskRevision:
        if self.state is not FormalTaskState.ACTIVE or self.active_task_id is None:
            raise FormalTaskTransitionError("cannot revise without an active task")
        if self.active_revision >= MAX_EXACT_FLOAT64_INTEGER:
            self._exhaust("task_revision exhausted")
        self.active_revision += 1
        assert self.control_epoch is not None
        return FormalTaskRevision(
            control_epoch=self.control_epoch,
            task_id=self.active_task_id,
            task_revision=self.active_revision,
            valid=valid,
        )

    def close(self, control_epoch: int | float) -> FormalTaskRevision | None:
        """Close ACTIVE once and return its final invalid publication identity.

        If an epoch boundary is observed, no old-epoch terminal row is
        fabricated: the lifecycle is disarmed and returns ``None``.  Within
        one epoch, duplicate close is a loud integration error.
        """

        if self.observe_epoch(control_epoch):
            return None
        if self.state is not FormalTaskState.ACTIVE:
            raise FormalTaskTransitionError("close requires an active task")
        terminal = self._advance_revision(valid=False)
        self.state = FormalTaskState.CLOSED_WAIT_REARM
        self.active_task_id = None
        self.active_revision = 0
        return terminal

    def _exhaust(self, message: str) -> None:
        self.state = FormalTaskState.DISARMED
        self.active_task_id = None
        self.active_revision = 0
        self.exhausted = True
        raise FormalTaskCounterExhaustion(message)
