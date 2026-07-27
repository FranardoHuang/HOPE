"""Pure task-first curriculum primitives for an arbitrary action library.

The runtime integration intentionally lives elsewhere.  This module owns only the
deterministic state machine needed to answer three questions:

* which action should receive the next training sample;
* how wide each task axis may currently range for that action; and
* whether evidence is strong enough to widen or bad enough to roll back.

Every action advances independently through ``position -> speed -> face -> base``.
An axis widens through the fixed levels ``0, .25, .5, .75, 1`` and the next axis
does not start until the previous one reaches its full range.  Promotion and
rollback use Wilson confidence bounds instead of raw empirical rates.

The checkpoint contract is deliberately fail-closed.  A saved state is bound to
the exact action manifest SHA-256, action order, gate configuration, schema, and
sampler cursor.  ``load_state_dict`` has no lenient mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
SAMPLER_SCHEMA_VERSION = 1
AXES: Tuple[str, ...] = ("position", "speed", "face", "base")
LEVELS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
STALL_POLICIES: Tuple[str, ...] = ("fail", "freeze")
_MAX_LEVEL_INDEX = len(LEVELS) - 1
_WILSON_Z_95 = 1.959963984540054


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _is_plain_number(value: object) -> bool:
    return type(value) in (int, float)


def _finite_float(value: object, *, name: str) -> float:
    if not _is_plain_number(value):
        raise TypeError(f"{name} must be a plain int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_exact_keys(
    value: Mapping[str, object],
    expected: Sequence[str],
    *,
    name: str,
    ordered: bool = False,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = tuple(value.keys())
    expected_tuple = tuple(expected)
    if ordered:
        if actual != expected_tuple:
            raise ValueError(
                f"{name} keys/order must be exactly {expected_tuple!r}; got {actual!r}"
            )
    elif set(actual) != set(expected_tuple) or len(actual) != len(expected_tuple):
        raise ValueError(
            f"{name} keys must be exactly {expected_tuple!r}; got {actual!r}"
        )


def _validate_manifest_sha256(value: object) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError("manifest_sha256 must be exactly 64 lowercase hexadecimal characters")
    return value


def _validate_action_order(action_order: object) -> Tuple[str, ...]:
    if isinstance(action_order, (str, bytes)) or not isinstance(action_order, Sequence):
        raise TypeError("action_order must be a sequence of action names")
    result = tuple(action_order)
    if not result:
        raise ValueError("action_order must contain at least one action")
    if any(not isinstance(action, str) or not action for action in result):
        raise ValueError("every action name must be a non-empty string")
    if len(set(result)) != len(result):
        raise ValueError("action_order must not contain duplicate action names")
    return result


def wilson_interval(
    successes: int,
    attempts: int,
    *,
    z: float = _WILSON_Z_95,
) -> Tuple[float, float]:
    """Return the two-sided Wilson score interval for a Bernoulli rate.

    Zero observations deliberately return the maximally uninformative interval
    ``(0, 1)``.  Callers must still enforce ``min_attempts`` before treating either
    bound as evidence.
    """

    if not _is_plain_int(successes) or not _is_plain_int(attempts):
        raise TypeError("successes and attempts must be plain integers")
    if attempts < 0:
        raise ValueError("attempts must be non-negative")
    if successes < 0 or successes > attempts:
        raise ValueError("successes must be between zero and attempts")
    z_value = _finite_float(z, name="z")
    if z_value <= 0.0:
        raise ValueError("z must be positive")
    if attempts == 0:
        return 0.0, 1.0

    n = float(attempts)
    p_hat = float(successes) / n
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / n
    center = (p_hat + z_squared / (2.0 * n)) / denominator
    margin = (
        z_value
        * math.sqrt((p_hat * (1.0 - p_hat) + z_squared / (4.0 * n)) / n)
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == attempts else min(1.0, center + margin)
    return lower, upper


def scaled_interval(
    center: float,
    full_interval: Sequence[float],
    level: float,
) -> Tuple[float, float]:
    """Scale an interval from a center point toward its full asymmetric bounds."""

    center_value = _finite_float(center, name="center")
    if (
        isinstance(full_interval, (str, bytes))
        or not isinstance(full_interval, Sequence)
        or len(full_interval) != 2
    ):
        raise TypeError("full_interval must be a two-element sequence")
    lower = _finite_float(full_interval[0], name="full_interval lower bound")
    upper = _finite_float(full_interval[1], name="full_interval upper bound")
    level_value = _finite_float(level, name="level")
    if lower > upper:
        raise ValueError("full_interval lower bound must not exceed upper bound")
    if center_value < lower or center_value > upper:
        raise ValueError("center must lie inside full_interval")
    if level_value < 0.0 or level_value > 1.0:
        raise ValueError("level must be in [0, 1]")
    return (
        center_value + level_value * (lower - center_value),
        center_value + level_value * (upper - center_value),
    )


def scaled_axis_ranges(
    centers: Sequence[float],
    full_ranges: Sequence[Sequence[float]],
    level: float,
) -> Tuple[Tuple[float, float], ...]:
    """Scale every dimension of one task axis by the same level."""

    if isinstance(centers, (str, bytes)) or not isinstance(centers, Sequence):
        raise TypeError("centers must be a sequence")
    if isinstance(full_ranges, (str, bytes)) or not isinstance(full_ranges, Sequence):
        raise TypeError("full_ranges must be a sequence")
    if len(centers) != len(full_ranges):
        raise ValueError("centers and full_ranges must have the same length")
    if not centers:
        raise ValueError("an axis must contain at least one dimension")
    return tuple(
        scaled_interval(center, full_interval, level)
        for center, full_interval in zip(centers, full_ranges)
    )


def compute_axis_ranges(
    centers_by_axis: Mapping[str, Sequence[float]],
    full_ranges_by_axis: Mapping[str, Sequence[Sequence[float]]],
    levels_by_axis: Mapping[str, float],
) -> Dict[str, Tuple[Tuple[float, float], ...]]:
    """Compute all four task-axis ranges using only the fixed curriculum levels."""

    _require_exact_keys(centers_by_axis, AXES, name="centers_by_axis")
    _require_exact_keys(full_ranges_by_axis, AXES, name="full_ranges_by_axis")
    _require_exact_keys(levels_by_axis, AXES, name="levels_by_axis")
    result: Dict[str, Tuple[Tuple[float, float], ...]] = {}
    for axis in AXES:
        level = _finite_float(levels_by_axis[axis], name=f"{axis} level")
        if level not in LEVELS:
            raise ValueError(f"{axis} level must be one of {LEVELS!r}")
        result[axis] = scaled_axis_ranges(
            centers_by_axis[axis],
            full_ranges_by_axis[axis],
            level,
        )
    return result


@dataclass(frozen=True)
class GateConfig:
    """Evidence, hysteresis, dwell, and stall policy for every frontier."""

    min_attempts: int
    enter_success_lower_bound: float
    exit_success_lower_bound: float
    enter_unsafe_upper_bound: float
    exit_unsafe_upper_bound: float
    enter_dwell_updates: int
    exit_dwell_updates: int
    max_stall_updates: int
    stall_policy: str = "fail"
    confidence_z: float = _WILSON_Z_95

    def __post_init__(self) -> None:
        for name in (
            "min_attempts",
            "enter_dwell_updates",
            "exit_dwell_updates",
            "max_stall_updates",
        ):
            value = getattr(self, name)
            if not _is_plain_int(value):
                raise TypeError(f"{name} must be a plain integer")
            if value < 1:
                raise ValueError(f"{name} must be at least one")

        for name in (
            "enter_success_lower_bound",
            "exit_success_lower_bound",
            "enter_unsafe_upper_bound",
            "exit_unsafe_upper_bound",
        ):
            value = _finite_float(getattr(self, name), name=name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)

        confidence_z = _finite_float(self.confidence_z, name="confidence_z")
        if confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")
        object.__setattr__(self, "confidence_z", confidence_z)

        if self.enter_success_lower_bound < self.exit_success_lower_bound:
            raise ValueError(
                "enter_success_lower_bound must be at least exit_success_lower_bound"
            )
        if self.enter_unsafe_upper_bound > self.exit_unsafe_upper_bound:
            raise ValueError(
                "enter_unsafe_upper_bound must not exceed exit_unsafe_upper_bound"
            )
        if self.max_stall_updates < max(
            self.enter_dwell_updates, self.exit_dwell_updates
        ):
            raise ValueError(
                "max_stall_updates must allow the longer dwell interval to complete"
            )
        if self.stall_policy not in STALL_POLICIES:
            raise ValueError(f"stall_policy must be one of {STALL_POLICIES!r}")

    def as_dict(self) -> Dict[str, object]:
        return {
            "min_attempts": self.min_attempts,
            "enter_success_lower_bound": self.enter_success_lower_bound,
            "exit_success_lower_bound": self.exit_success_lower_bound,
            "enter_unsafe_upper_bound": self.enter_unsafe_upper_bound,
            "exit_unsafe_upper_bound": self.exit_unsafe_upper_bound,
            "enter_dwell_updates": self.enter_dwell_updates,
            "exit_dwell_updates": self.exit_dwell_updates,
            "max_stall_updates": self.max_stall_updates,
            "stall_policy": self.stall_policy,
            "confidence_z": self.confidence_z,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "GateConfig":
        expected = (
            "min_attempts",
            "enter_success_lower_bound",
            "exit_success_lower_bound",
            "enter_unsafe_upper_bound",
            "exit_unsafe_upper_bound",
            "enter_dwell_updates",
            "exit_dwell_updates",
            "max_stall_updates",
            "stall_policy",
            "confidence_z",
        )
        _require_exact_keys(state, expected, name="gate_config")
        return cls(**{name: state[name] for name in expected})


@dataclass(frozen=True)
class OutcomeCounts:
    """Disjoint aggregate outcomes for one action's current frontier."""

    attempts: int
    successes: int
    unsafe_failures: int = 0

    def __post_init__(self) -> None:
        for name in ("attempts", "successes", "unsafe_failures"):
            value = getattr(self, name)
            if not _is_plain_int(value):
                raise TypeError(f"{name} must be a plain integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.successes > self.attempts:
            raise ValueError("successes must not exceed attempts")
        if self.unsafe_failures > self.attempts:
            raise ValueError("unsafe_failures must not exceed attempts")
        if self.successes + self.unsafe_failures > self.attempts:
            raise ValueError("successes and unsafe_failures must be disjoint outcomes")


@dataclass(frozen=True)
class GateDecision:
    enough_attempts: bool
    success_lower: float
    success_upper: float
    unsafe_lower: float
    unsafe_upper: float
    enter_ok: bool
    exit_bad: bool
    enter_blockers: Tuple[str, ...]
    exit_reasons: Tuple[str, ...]


def evaluate_gate(counts: OutcomeCounts, config: GateConfig) -> GateDecision:
    """Evaluate one evidence window against confidence-bound hysteresis."""

    if not isinstance(counts, OutcomeCounts):
        raise TypeError("counts must be an OutcomeCounts instance")
    if not isinstance(config, GateConfig):
        raise TypeError("config must be a GateConfig instance")
    success_lower, success_upper = wilson_interval(
        counts.successes,
        counts.attempts,
        z=config.confidence_z,
    )
    unsafe_lower, unsafe_upper = wilson_interval(
        counts.unsafe_failures,
        counts.attempts,
        z=config.confidence_z,
    )
    enough_attempts = counts.attempts >= config.min_attempts

    blockers = []
    if not enough_attempts:
        blockers.append("minimum_attempts")
    if success_lower < config.enter_success_lower_bound:
        blockers.append("success_lower_bound")
    if unsafe_upper > config.enter_unsafe_upper_bound:
        blockers.append("unsafe_upper_bound")
    enter_ok = not blockers

    exit_reasons = []
    if enough_attempts:
        if success_lower < config.exit_success_lower_bound:
            exit_reasons.append("success_lower_bound")
        if unsafe_upper > config.exit_unsafe_upper_bound:
            exit_reasons.append("unsafe_upper_bound")
    return GateDecision(
        enough_attempts=enough_attempts,
        success_lower=success_lower,
        success_upper=success_upper,
        unsafe_lower=unsafe_lower,
        unsafe_upper=unsafe_upper,
        enter_ok=enter_ok,
        exit_bad=bool(exit_reasons),
        enter_blockers=tuple(blockers),
        exit_reasons=tuple(exit_reasons),
    )


class BalancedActionSampler:
    """Deterministic round-robin sampler with an exact-resume cursor."""

    def __init__(self, action_order: Sequence[str]) -> None:
        self._action_order = _validate_action_order(action_order)
        self._cursor = 0
        self._draw_count = 0

    @property
    def action_order(self) -> Tuple[str, ...]:
        return self._action_order

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def draw_count(self) -> int:
        return self._draw_count

    def sample_indices(self, count: int) -> Tuple[int, ...]:
        if not _is_plain_int(count):
            raise TypeError("count must be a plain integer")
        if count < 0:
            raise ValueError("count must be non-negative")
        size = len(self._action_order)
        result = tuple((self._cursor + offset) % size for offset in range(count))
        self._draw_count += count
        self._cursor = self._draw_count % size
        return result

    def sample(self, count: int) -> Tuple[str, ...]:
        return tuple(self._action_order[index] for index in self.sample_indices(count))

    def state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "action_order": list(self._action_order),
            "cursor": self._cursor,
            "draw_count": self._draw_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = ("schema_version", "action_order", "cursor", "draw_count")
        _require_exact_keys(state, expected, name="sampler state")
        schema_version = state["schema_version"]
        if not _is_plain_int(schema_version) or schema_version != SAMPLER_SCHEMA_VERSION:
            raise ValueError(
                f"sampler schema_version must be {SAMPLER_SCHEMA_VERSION}"
            )
        if not isinstance(state["action_order"], list):
            raise TypeError("sampler action_order must be a list")
        action_order = _validate_action_order(state["action_order"])
        if action_order != self._action_order:
            raise ValueError("sampler action_order does not match this sampler")
        cursor = state["cursor"]
        draw_count = state["draw_count"]
        if not _is_plain_int(cursor) or not _is_plain_int(draw_count):
            raise TypeError("sampler cursor and draw_count must be plain integers")
        if draw_count < 0:
            raise ValueError("sampler draw_count must be non-negative")
        if cursor < 0 or cursor >= len(self._action_order):
            raise ValueError("sampler cursor is outside the action order")
        if cursor != draw_count % len(self._action_order):
            raise ValueError("sampler cursor is inconsistent with draw_count")
        self._cursor = cursor
        self._draw_count = draw_count


@dataclass(frozen=True)
class UpdateResult:
    action: str
    axis: str
    kind: str
    from_level: float
    to_level: float
    decision: Optional[GateDecision]


class CurriculumStalledError(RuntimeError):
    """Raised by the fail policy before any update in the batch is committed."""

    def __init__(self, *, action: str, axis: str, stall_updates: int) -> None:
        self.action = action
        self.axis = axis
        self.stall_updates = stall_updates
        super().__init__(
            f"task-first curriculum stalled for action {action!r}, axis {axis!r} "
            f"after {stall_updates} updates"
        )


class TaskFirstCurriculum:
    """Per-action sequential task-range curriculum with exact checkpoint identity."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        action_order: Sequence[str],
        gate_config: GateConfig,
    ) -> None:
        self._manifest_sha256 = _validate_manifest_sha256(manifest_sha256)
        self._action_order = _validate_action_order(action_order)
        if not isinstance(gate_config, GateConfig):
            raise TypeError("gate_config must be a GateConfig instance")
        self._gate_config = gate_config
        self._level_indices = {
            action: [0 for _ in AXES] for action in self._action_order
        }
        self._enter_dwell = {
            action: [0 for _ in AXES] for action in self._action_order
        }
        self._exit_dwell = {
            action: [0 for _ in AXES] for action in self._action_order
        }
        self._stall_updates = {
            action: [0 for _ in AXES] for action in self._action_order
        }
        self._frozen = {action: False for action in self._action_order}
        self._sampler = BalancedActionSampler(self._action_order)

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def action_order(self) -> Tuple[str, ...]:
        return self._action_order

    @property
    def gate_config(self) -> GateConfig:
        return self._gate_config

    @property
    def sampler(self) -> BalancedActionSampler:
        return self._sampler

    def _require_action(self, action: str) -> None:
        if action not in self._level_indices:
            raise KeyError(f"unknown action {action!r}")

    def level_indices(self, action: str) -> Tuple[int, ...]:
        self._require_action(action)
        return tuple(self._level_indices[action])

    def levels(self, action: str) -> Dict[str, float]:
        self._require_action(action)
        return {
            axis: LEVELS[index]
            for axis, index in zip(AXES, self._level_indices[action])
        }

    def axis_level(self, action: str, axis: str) -> float:
        self._require_action(action)
        if axis not in AXES:
            raise KeyError(f"unknown axis {axis!r}")
        return LEVELS[self._level_indices[action][AXES.index(axis)]]

    def active_axis(self, action: str) -> str:
        self._require_action(action)
        return AXES[self._active_axis_index(self._level_indices[action])]

    def is_complete(self, action: str) -> bool:
        self._require_action(action)
        return self._is_complete_indices(self._level_indices[action])

    def is_frozen(self, action: str) -> bool:
        self._require_action(action)
        return self._frozen[action]

    def sample_actions(self, count: int) -> Tuple[str, ...]:
        return self._sampler.sample(count)

    @staticmethod
    def _is_complete_indices(indices: Sequence[int]) -> bool:
        return all(index == _MAX_LEVEL_INDEX for index in indices)

    @staticmethod
    def _active_axis_index(indices: Sequence[int]) -> int:
        for index, level_index in enumerate(indices):
            if level_index < _MAX_LEVEL_INDEX:
                return index
        return len(AXES) - 1

    @staticmethod
    def _rollback_axis_index(indices: Sequence[int], active_index: int) -> Optional[int]:
        if indices[active_index] > 0:
            return active_index
        for index in range(active_index - 1, -1, -1):
            if indices[index] > 0:
                return index
        return None

    @staticmethod
    def _reset_action_counters(
        action: str,
        enter_dwell: Dict[str, list],
        exit_dwell: Dict[str, list],
        stall_updates: Dict[str, list],
    ) -> None:
        enter_dwell[action] = [0 for _ in AXES]
        exit_dwell[action] = [0 for _ in AXES]
        stall_updates[action] = [0 for _ in AXES]

    def advance(
        self,
        evidence_by_action: Mapping[str, OutcomeCounts],
    ) -> Tuple[UpdateResult, ...]:
        """Advance one evidence update atomically across all actions.

        The evidence mapping must contain each bound action exactly once.  If the
        ``fail`` stall policy fires for any action, none of the actions' progress is
        committed.
        """

        _require_exact_keys(
            evidence_by_action,
            self._action_order,
            name="evidence_by_action",
        )
        for action in self._action_order:
            if not isinstance(evidence_by_action[action], OutcomeCounts):
                raise TypeError(
                    f"evidence for action {action!r} must be an OutcomeCounts instance"
                )

        level_indices = {
            action: list(values) for action, values in self._level_indices.items()
        }
        enter_dwell = {
            action: list(values) for action, values in self._enter_dwell.items()
        }
        exit_dwell = {
            action: list(values) for action, values in self._exit_dwell.items()
        }
        stall_updates = {
            action: list(values) for action, values in self._stall_updates.items()
        }
        frozen = dict(self._frozen)
        results = []

        for action in self._action_order:
            indices = level_indices[action]
            complete = self._is_complete_indices(indices)
            active_index = self._active_axis_index(indices)
            active_axis = AXES[active_index]
            active_level = LEVELS[indices[active_index]]

            if frozen[action]:
                results.append(
                    UpdateResult(
                        action=action,
                        axis=active_axis,
                        kind="frozen_hold",
                        from_level=active_level,
                        to_level=active_level,
                        decision=None,
                    )
                )
                continue

            decision = evaluate_gate(evidence_by_action[action], self._gate_config)
            if decision.exit_bad:
                exit_dwell[action][active_index] += 1
            else:
                exit_dwell[action][active_index] = 0

            if (
                decision.exit_bad
                and exit_dwell[action][active_index]
                >= self._gate_config.exit_dwell_updates
            ):
                rollback_index = self._rollback_axis_index(indices, active_index)
                if rollback_index is not None:
                    from_level = LEVELS[indices[rollback_index]]
                    indices[rollback_index] -= 1
                    to_level = LEVELS[indices[rollback_index]]
                    self._reset_action_counters(
                        action,
                        enter_dwell,
                        exit_dwell,
                        stall_updates,
                    )
                    results.append(
                        UpdateResult(
                            action=action,
                            axis=AXES[rollback_index],
                            kind="retreat",
                            from_level=from_level,
                            to_level=to_level,
                            decision=decision,
                        )
                    )
                    continue

            if complete:
                results.append(
                    UpdateResult(
                        action=action,
                        axis=active_axis,
                        kind="complete_hold",
                        from_level=active_level,
                        to_level=active_level,
                        decision=decision,
                    )
                )
                continue

            if decision.enter_ok:
                enter_dwell[action][active_index] += 1
            else:
                # Promotion is a stability claim, so its dwell must be
                # consecutive.  Carrying an old good window through neutral or
                # unsafe evidence would let sparse successes eventually widen
                # the task domain even though competence never stayed above
                # the entry gate.
                enter_dwell[action][active_index] = 0
            if (
                decision.enter_ok
                and enter_dwell[action][active_index]
                >= self._gate_config.enter_dwell_updates
            ):
                from_level = LEVELS[indices[active_index]]
                indices[active_index] += 1
                to_level = LEVELS[indices[active_index]]
                self._reset_action_counters(
                    action,
                    enter_dwell,
                    exit_dwell,
                    stall_updates,
                )
                results.append(
                    UpdateResult(
                        action=action,
                        axis=active_axis,
                        kind="promote",
                        from_level=from_level,
                        to_level=to_level,
                        decision=decision,
                    )
                )
                continue

            stall_updates[action][active_index] += 1
            if (
                stall_updates[action][active_index]
                >= self._gate_config.max_stall_updates
            ):
                if self._gate_config.stall_policy == "fail":
                    raise CurriculumStalledError(
                        action=action,
                        axis=active_axis,
                        stall_updates=stall_updates[action][active_index],
                    )
                frozen[action] = True
                results.append(
                    UpdateResult(
                        action=action,
                        axis=active_axis,
                        kind="freeze",
                        from_level=active_level,
                        to_level=active_level,
                        decision=decision,
                    )
                )
                continue

            results.append(
                UpdateResult(
                    action=action,
                    axis=active_axis,
                    kind="hold",
                    from_level=active_level,
                    to_level=active_level,
                    decision=decision,
                )
            )

        self._level_indices = level_indices
        self._enter_dwell = enter_dwell
        self._exit_dwell = exit_dwell
        self._stall_updates = stall_updates
        self._frozen = frozen
        return tuple(results)

    def state_dict(self) -> Dict[str, object]:
        progress = {}
        for action in self._action_order:
            progress[action] = {
                "level_indices": list(self._level_indices[action]),
                "enter_dwell": list(self._enter_dwell[action]),
                "exit_dwell": list(self._exit_dwell[action]),
                "stall_updates": list(self._stall_updates[action]),
                "frozen": self._frozen[action],
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": self._manifest_sha256,
            "action_order": list(self._action_order),
            "axes": list(AXES),
            "levels": list(LEVELS),
            "gate_config": self._gate_config.as_dict(),
            "progress": progress,
            "sampler": self._sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Strictly validate a complete state before mutating this curriculum."""

        expected = (
            "schema_version",
            "manifest_sha256",
            "action_order",
            "axes",
            "levels",
            "gate_config",
            "progress",
            "sampler",
        )
        _require_exact_keys(state, expected, name="curriculum state")
        schema_version = state["schema_version"]
        if not _is_plain_int(schema_version) or schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if state["manifest_sha256"] != self._manifest_sha256:
            raise ValueError("manifest_sha256 does not match this curriculum")
        if not isinstance(state["action_order"], list):
            raise TypeError("action_order state must be a list")
        action_order = _validate_action_order(state["action_order"])
        if action_order != self._action_order:
            raise ValueError("action_order does not match this curriculum")
        if state["axes"] != list(AXES):
            raise ValueError(f"axes must be exactly {list(AXES)!r}")
        if state["levels"] != list(LEVELS):
            raise ValueError(f"levels must be exactly {list(LEVELS)!r}")

        restored_config = GateConfig.from_dict(state["gate_config"])
        if restored_config != self._gate_config:
            raise ValueError("gate_config does not match this curriculum")

        progress = state["progress"]
        _require_exact_keys(
            progress,
            self._action_order,
            name="progress",
            ordered=True,
        )
        restored_levels = {}
        restored_enter = {}
        restored_exit = {}
        restored_stall = {}
        restored_frozen = {}
        progress_keys = (
            "level_indices",
            "enter_dwell",
            "exit_dwell",
            "stall_updates",
            "frozen",
        )
        for action in self._action_order:
            action_state = progress[action]
            _require_exact_keys(
                action_state,
                progress_keys,
                name=f"progress[{action!r}]",
            )
            lists = {}
            for name in (
                "level_indices",
                "enter_dwell",
                "exit_dwell",
                "stall_updates",
            ):
                values = action_state[name]
                if not isinstance(values, list) or len(values) != len(AXES):
                    raise ValueError(
                        f"progress[{action!r}][{name!r}] must be a four-element list"
                    )
                if any(not _is_plain_int(value) for value in values):
                    raise TypeError(
                        f"progress[{action!r}][{name!r}] must contain plain integers"
                    )
                lists[name] = list(values)

            levels = lists["level_indices"]
            if any(index < 0 or index > _MAX_LEVEL_INDEX for index in levels):
                raise ValueError(
                    f"progress[{action!r}]['level_indices'] contains an invalid level"
                )
            first_incomplete = next(
                (index for index, value in enumerate(levels) if value < _MAX_LEVEL_INDEX),
                len(AXES),
            )
            if any(value != 0 for value in levels[first_incomplete + 1 :]):
                raise ValueError(
                    f"progress[{action!r}] violates sequential axis promotion"
                )
            for name in ("enter_dwell", "exit_dwell", "stall_updates"):
                if any(value < 0 for value in lists[name]):
                    raise ValueError(
                        f"progress[{action!r}][{name!r}] must be non-negative"
                    )
            is_frozen = action_state["frozen"]
            if type(is_frozen) is not bool:
                raise TypeError(f"progress[{action!r}]['frozen'] must be a bool")

            complete = self._is_complete_indices(levels)
            active_index = self._active_axis_index(levels)
            inactive_indices = tuple(
                index for index in range(len(AXES)) if index != active_index
            )
            for name in ("enter_dwell", "exit_dwell", "stall_updates"):
                if any(lists[name][index] != 0 for index in inactive_indices):
                    raise ValueError(
                        f"progress[{action!r}][{name!r}] has unreachable "
                        "non-active-axis counters"
                    )
            active_enter = lists["enter_dwell"][active_index]
            active_exit = lists["exit_dwell"][active_index]
            active_stall = lists["stall_updates"][active_index]
            if active_enter >= self._gate_config.enter_dwell_updates:
                raise ValueError(
                    f"progress[{action!r}] has an unreachable enter dwell counter"
                )
            if complete:
                if is_frozen or active_enter != 0 or active_stall != 0:
                    raise ValueError(
                        f"progress[{action!r}] has unreachable completed counters"
                    )
                if active_exit >= self._gate_config.exit_dwell_updates:
                    raise ValueError(
                        f"progress[{action!r}] has an unreachable exit dwell counter"
                    )
            else:
                if active_enter + active_exit > active_stall:
                    raise ValueError(
                        f"progress[{action!r}] has counters ahead of its stall clock"
                    )
                rollback_index = self._rollback_axis_index(levels, active_index)
                if (
                    rollback_index is not None
                    and active_exit >= self._gate_config.exit_dwell_updates
                ):
                    raise ValueError(
                        f"progress[{action!r}] has an unreachable exit dwell counter"
                    )
                if is_frozen:
                    if (
                        self._gate_config.stall_policy != "freeze"
                        or active_stall != self._gate_config.max_stall_updates
                    ):
                        raise ValueError(
                            f"progress[{action!r}] has an unreachable frozen state"
                        )
                elif active_stall >= self._gate_config.max_stall_updates:
                    raise ValueError(
                        f"progress[{action!r}] has an unreachable stall counter"
                    )

            restored_levels[action] = levels
            restored_enter[action] = lists["enter_dwell"]
            restored_exit[action] = lists["exit_dwell"]
            restored_stall[action] = lists["stall_updates"]
            restored_frozen[action] = is_frozen

        restored_sampler = BalancedActionSampler(self._action_order)
        restored_sampler.load_state_dict(state["sampler"])

        self._level_indices = restored_levels
        self._enter_dwell = restored_enter
        self._exit_dwell = restored_exit
        self._stall_updates = restored_stall
        self._frozen = restored_frozen
        self._sampler = restored_sampler
