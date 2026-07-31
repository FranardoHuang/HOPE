"""Deploy-faithful action terms.

ClampedJointPositionAction (2026-07-05): the C++ deploy runner clamps q_des to the
A3 joint limits before publishing (pp_joint_limits.hpp) — a SAFETY feature — but
training ran with no clamp (clip_actions=null, and PhysX implicit drives accept
out-of-range targets as "saturated torque, please"). The policy legitimately
learned to command PAST the ankle limit to buy kp-saturated torque when arresting
a forward tip (118 Nm requested -> clamp cut it to ~41 Nm on 34% of bare-hold
ticks in the Gate 2.5 P2 log = ~65% of the tipping-arrest torque silently
removed at deploy). Clamping the PROCESSED action (the joint-position target) in
training makes train == deploy so the policy learns torque strategies that
survive the runner's clamp.

DEFAULT ON since 2026-07-06 (franco ruling): jiayi found the unclamped P2 product
line CANNOT EVEN STAND in the MuJoCo gate — the policy's balance strategy leans
on out-of-range q_des torque the deploy runner will never grant. This is a
train==deploy correctness alignment, not a tunable: every future run trains
clamped. `clamp=False` remains available ONLY for explicit legacy-reproduction /
control arms (`actions: qdes_clamp: false` in the task YAML), and batch
comparisons must keep clamp state uniform within the batch.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class _PhysicsSubstepJointSafetyLedger:
    """One-policy-step, read-only-exportable hard-limit readback transcript.

    The ledger records exactly ``decimation`` calls made through ``apply_actions`` followed by one
    post-step readback requested by the joint DoneTerms.  It proves that the guard observed a
    fresh articulation timestamp and what q/qdot/gaps it saw; it does *not* prove a stopping
    distance because no acceleration/deceleration bound is assumed.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        num_envs: int,
        joint_count: int,
        expected_apply_calls: int,
        physics_dt_s: float,
        device: torch.device | str,
        dtype: torch.dtype,
        retain_dense_records: bool = True,
    ) -> None:
        self._num_envs = int(num_envs)
        self._joint_count = int(joint_count)
        self._expected_apply_calls = int(expected_apply_calls)
        self._physics_dt_s = float(physics_dt_s)
        self._device = torch.device(device)
        self._dtype = dtype
        self._retain_dense_records = bool(retain_dense_records)
        self._policy_step_sequence = -1
        self._policy_start_timestamp_s: float | None = None
        self._last_record_timestamp_s: float | None = None
        self._apply_call_count = 0
        self._post_readback_count = 0
        self._records: list[dict[str, Any]] = []
        self._has_started = False
        self._invalid_envs = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._aggregate_apply_count = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self._aggregate_post_count = torch.zeros_like(
            self._aggregate_apply_count
        )
        self._aggregate_valid_record_count = torch.zeros_like(
            self._aggregate_apply_count
        )
        aggregate_shape = (self._num_envs, self._joint_count)
        self._aggregate_hard_crossing_count = torch.zeros(
            aggregate_shape, dtype=torch.long, device=self._device
        )
        self._aggregate_actual_hard_edge_count = torch.zeros_like(
            self._aggregate_hard_crossing_count
        )
        self._aggregate_min_lower_gap = torch.full(
            aggregate_shape,
            float("inf"),
            dtype=self._dtype,
            device=self._device,
        )
        self._aggregate_min_upper_gap = torch.full_like(
            self._aggregate_min_lower_gap, float("inf")
        )

    @staticmethod
    def _finite_timestamp(value: object, *, context: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(f"{context} requires a finite articulation sim timestamp")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise RuntimeError(f"{context} requires a finite articulation sim timestamp")
        return parsed

    def begin_policy_step(self, timestamp_s: object) -> None:
        """Start a policy transcript, refusing to overwrite an incomplete previous step."""

        if self._has_started and not self.is_complete:
            raise RuntimeError(
                "joint-safety ledger cannot begin a new policy step before exactly "
                f"{self._expected_apply_calls} apply_actions calls and one post-step readback; "
                f"got apply={self._apply_call_count} post={self._post_readback_count}"
            )
        self._policy_step_sequence += 1
        self._policy_start_timestamp_s = self._finite_timestamp(
            timestamp_s, context="joint-safety ledger policy start"
        )
        self._last_record_timestamp_s = None
        self._apply_call_count = 0
        self._post_readback_count = 0
        self._records = []
        self._invalid_envs.zero_()
        self._aggregate_apply_count.zero_()
        self._aggregate_post_count.zero_()
        self._aggregate_valid_record_count.zero_()
        self._aggregate_hard_crossing_count.zero_()
        self._aggregate_actual_hard_edge_count.zero_()
        self._aggregate_min_lower_gap.fill_(float("inf"))
        self._aggregate_min_upper_gap.fill_(float("inf"))
        self._has_started = True

    @property
    def has_started(self) -> bool:
        return self._has_started

    @property
    def is_complete(self) -> bool:
        return (
            self._has_started
            and self._apply_call_count == self._expected_apply_calls
            and self._post_readback_count == 1
        )

    @property
    def post_readback_recorded(self) -> bool:
        return self._post_readback_count == 1

    def _validate_record_tensors(
        self,
        *,
        q: torch.Tensor,
        qdot: torch.Tensor,
        lower_gap: torch.Tensor,
        upper_gap: torch.Tensor,
        hard_crossing: torch.Tensor,
        actual_hard_edge: torch.Tensor,
    ) -> None:
        expected = (self._num_envs, self._joint_count)
        for name, value in (
            ("q", q),
            ("qdot", qdot),
            ("lower_gap", lower_gap),
            ("upper_gap", upper_gap),
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != expected
                or value.device != self._device
                or value.dtype != self._dtype
            ):
                raise RuntimeError(
                    f"joint-safety ledger {name} must match {expected}, "
                    f"device={self._device}, dtype={self._dtype}"
                )
        for name, value in (
            ("hard_crossing", hard_crossing),
            ("actual_hard_edge", actual_hard_edge),
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != expected
                or value.device != self._device
                or value.dtype != torch.bool
            ):
                raise RuntimeError(
                    f"joint-safety ledger {name} must be a bool tensor shaped {expected}"
                )

    def record(
        self,
        *,
        kind: str,
        timestamp_s: object,
        joint_pos_timestamp_s: object,
        joint_vel_timestamp_s: object,
        q: torch.Tensor,
        qdot: torch.Tensor,
        lower_gap: torch.Tensor,
        upper_gap: torch.Tensor,
        hard_crossing: torch.Tensor,
        actual_hard_edge: torch.Tensor,
    ) -> None:
        """Append one apply or post readback with a strict timestamp/call-order invariant."""

        if not self._has_started or self._policy_start_timestamp_s is None:
            raise RuntimeError("joint-safety ledger record arrived before policy-step begin")
        if kind == "apply":
            if self._post_readback_count:
                raise RuntimeError("joint-safety apply_actions call arrived after post-step readback")
            call_index = self._apply_call_count
            if call_index >= self._expected_apply_calls:
                raise RuntimeError(
                    "joint-safety apply_actions call count exceeded configured decimation"
                )
        elif kind == "post":
            if self._apply_call_count != self._expected_apply_calls:
                raise RuntimeError(
                    "joint-safety post-step readback requires exactly "
                    f"{self._expected_apply_calls} prior apply_actions calls; "
                    f"got {self._apply_call_count}"
                )
            if self._post_readback_count:
                raise RuntimeError("joint-safety post-step readback was recorded twice")
            call_index = self._expected_apply_calls
        else:
            raise ValueError("joint-safety ledger kind must be 'apply' or 'post'")

        timestamp = self._finite_timestamp(
            timestamp_s, context=f"joint-safety {kind} readback"
        )
        joint_pos_timestamp = self._finite_timestamp(
            joint_pos_timestamp_s,
            context=f"joint-safety {kind} joint_pos buffer",
        )
        joint_vel_timestamp = self._finite_timestamp(
            joint_vel_timestamp_s,
            context=f"joint-safety {kind} joint_vel buffer",
        )
        if (
            joint_pos_timestamp != timestamp
            or joint_vel_timestamp != timestamp
        ):
            raise RuntimeError(
                "joint-safety q/qdot lazy-buffer timestamps are not fresh"
            )
        expected_timestamp = (
            self._policy_start_timestamp_s + call_index * self._physics_dt_s
        )
        if not math.isclose(
            timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise RuntimeError(
                "joint-safety readback timestamp is stale or skipped: "
                f"kind={kind} index={call_index} actual={timestamp} "
                f"expected={expected_timestamp}"
            )
        if (
            self._last_record_timestamp_s is not None
            and not timestamp > self._last_record_timestamp_s
        ):
            raise RuntimeError(
                "joint-safety articulation timestamp did not advance between substeps"
            )
        self._validate_record_tensors(
            q=q,
            qdot=qdot,
            lower_gap=lower_gap,
            upper_gap=upper_gap,
            hard_crossing=hard_crossing,
            actual_hard_edge=actual_hard_edge,
        )
        # Validation above is side-effect free.  Commit call counters and the record together so a
        # caught stale/tensor error can never manufacture a "complete" transcript with a hole.
        if kind == "apply":
            self._apply_call_count += 1
        else:
            self._post_readback_count = 1
        self._last_record_timestamp_s = timestamp
        valid = ~self._invalid_envs
        if self._retain_dense_records:
            self._records.append(
                {
                    "kind": kind,
                    "call_index": call_index,
                    "timestamp_s": timestamp,
                    "joint_pos_timestamp_s": joint_pos_timestamp,
                    "joint_vel_timestamp_s": joint_vel_timestamp,
                    "env_valid": valid.detach().clone(),
                    "q": q.detach().clone(),
                    "qdot": qdot.detach().clone(),
                    "lower_gap": lower_gap.detach().clone(),
                    "upper_gap": upper_gap.detach().clone(),
                    "hard_crossing": hard_crossing.detach().clone(),
                    "actual_hard_edge": actual_hard_edge.detach().clone(),
                }
            )
        valid_long = valid.to(dtype=torch.long)
        valid_joint = valid[:, None]
        self._aggregate_valid_record_count += valid_long
        if kind == "apply":
            self._aggregate_apply_count += valid_long
        else:
            self._aggregate_post_count += valid_long
        self._aggregate_hard_crossing_count += (
            hard_crossing & valid_joint
        ).to(dtype=torch.long)
        self._aggregate_actual_hard_edge_count += (
            actual_hard_edge & valid_joint
        ).to(dtype=torch.long)
        self._aggregate_min_lower_gap = torch.minimum(
            self._aggregate_min_lower_gap,
            torch.where(
                valid_joint,
                torch.where(
                    torch.isfinite(lower_gap),
                    lower_gap,
                    torch.full_like(lower_gap, float("-inf")),
                ),
                torch.full_like(lower_gap, float("inf")),
            ),
        )
        self._aggregate_min_upper_gap = torch.minimum(
            self._aggregate_min_upper_gap,
            torch.where(
                valid_joint,
                torch.where(
                    torch.isfinite(upper_gap),
                    upper_gap,
                    torch.full_like(upper_gap, float("-inf")),
                ),
                torch.full_like(upper_gap, float("inf")),
            ),
        )

    def reset_envs(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Atomically invalidate and clear every retained readback row for reset environments."""

        self._invalid_envs[env_ids] = True
        self._aggregate_apply_count[env_ids] = 0
        self._aggregate_post_count[env_ids] = 0
        self._aggregate_valid_record_count[env_ids] = 0
        self._aggregate_hard_crossing_count[env_ids] = 0
        self._aggregate_actual_hard_edge_count[env_ids] = 0
        self._aggregate_min_lower_gap[env_ids] = float("inf")
        self._aggregate_min_upper_gap[env_ids] = float("inf")
        for record in self._records:
            record["env_valid"][env_ids] = False
            for name in ("q", "qdot", "lower_gap", "upper_gap"):
                record[name][env_ids] = float("nan")
            record["hard_crossing"][env_ids] = False
            record["actual_hard_edge"][env_ids] = False

    def snapshot(
        self,
        *,
        qdes_env_latch: torch.Tensor,
        crossing_env_latch: torch.Tensor,
        qdes_joint_latch: torch.Tensor,
        crossing_joint_latch: torch.Tensor,
        qdes_joint_count: torch.Tensor,
        crossing_joint_count: torch.Tensor,
        substep_crossing_joint_latch: torch.Tensor,
        substep_actual_joint_latch: torch.Tensor,
        substep_crossing_joint_count: torch.Tensor,
        substep_actual_joint_count: torch.Tensor,
    ) -> dict[str, Any]:
        """Return detached clones; callers cannot mutate live safety state."""

        record_count = len(self._records)
        shape = (0, self._num_envs, self._joint_count)
        env_shape = (0, self._num_envs)

        def stack(name: str, *, dtype: torch.dtype, tensor_shape: tuple[int, ...]):
            if not self._records:
                return torch.empty(
                    tensor_shape, dtype=dtype, device=self._device
                )
            return torch.stack([record[name] for record in self._records], dim=0)

        return {
            "schema_version": self._SCHEMA_VERSION,
            "policy_step_sequence": self._policy_step_sequence,
            "policy_start_timestamp_s": self._policy_start_timestamp_s,
            "expected_apply_calls": self._expected_apply_calls,
            "physics_dt_s": self._physics_dt_s,
            "apply_call_count": self._apply_call_count,
            "post_readback_count": self._post_readback_count,
            "complete": self.is_complete,
            "record_count": record_count,
            "record_kind": tuple(record["kind"] for record in self._records),
            "call_index": tuple(record["call_index"] for record in self._records),
            "timestamp_s": tuple(record["timestamp_s"] for record in self._records),
            "joint_pos_timestamp_s": tuple(
                record["joint_pos_timestamp_s"] for record in self._records
            ),
            "joint_vel_timestamp_s": tuple(
                record["joint_vel_timestamp_s"] for record in self._records
            ),
            "env_valid": stack(
                "env_valid", dtype=torch.bool, tensor_shape=env_shape
            ),
            "q": stack("q", dtype=self._dtype, tensor_shape=shape),
            "qdot": stack("qdot", dtype=self._dtype, tensor_shape=shape),
            "hard_lower_gap": stack(
                "lower_gap", dtype=self._dtype, tensor_shape=shape
            ),
            "hard_upper_gap": stack(
                "upper_gap", dtype=self._dtype, tensor_shape=shape
            ),
            "hard_crossing": stack(
                "hard_crossing", dtype=torch.bool, tensor_shape=shape
            ),
            "actual_hard_edge": stack(
                "actual_hard_edge", dtype=torch.bool, tensor_shape=shape
            ),
            "qdes_env_latch": qdes_env_latch.detach().clone(),
            "crossing_env_latch": crossing_env_latch.detach().clone(),
            "qdes_joint_latch": qdes_joint_latch.detach().clone(),
            "crossing_joint_latch": crossing_joint_latch.detach().clone(),
            "qdes_joint_count": qdes_joint_count.detach().clone(),
            "crossing_joint_count": crossing_joint_count.detach().clone(),
            "substep_crossing_joint_latch": (
                substep_crossing_joint_latch.detach().clone()
            ),
            "substep_actual_joint_latch": (
                substep_actual_joint_latch.detach().clone()
            ),
            "substep_crossing_joint_count": (
                substep_crossing_joint_count.detach().clone()
            ),
            "substep_actual_joint_count": (
                substep_actual_joint_count.detach().clone()
            ),
        }

    def aggregate_rows(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reduce the live transcript on-device without materializing an export snapshot."""

        return {
            "apply_count": self._aggregate_apply_count[env_ids],
            "post_count": self._aggregate_post_count[env_ids],
            "valid_record_count": self._aggregate_valid_record_count[env_ids],
            "hard_crossing_count": (
                self._aggregate_hard_crossing_count[env_ids]
            ),
            "actual_hard_edge_count": (
                self._aggregate_actual_hard_edge_count[env_ids]
            ),
            "min_lower_gap": self._aggregate_min_lower_gap[env_ids],
            "min_upper_gap": self._aggregate_min_upper_gap[env_ids],
        }

    def unsafe_env_mask(self) -> torch.Tensor:
        """Return current-step physical crossing/edge evidence without export clones."""

        return torch.any(
            self._aggregate_hard_crossing_count.gt(0)
            | self._aggregate_actual_hard_edge_count.gt(0),
            dim=1,
        )


class _PhysicsSubstepTableContactLatch:
    """One episode-sticky table-contact bit sampled across one policy step.

    ``apply_actions`` runs before each physics substep.  Its first call therefore has no current
    policy-step result to observe and must be skipped.  Calls 2..N observe substeps 1..N-1; the
    table DoneTerm observes substep N after the loop.  A hit remains sticky until that environment
    is reset.  Merely starting the next policy step never clears safety evidence.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        expected_apply_calls: int,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0 or expected_apply_calls < 2:
            raise ValueError(
                "table-contact latch needs positive environments and at least "
                "two physics substeps so a stale post-table-reset report can "
                "be quarantined without hiding persistent contact"
            )
        self._num_envs = int(num_envs)
        self._expected_apply_calls = int(expected_apply_calls)
        self._hit = torch.zeros(
            self._num_envs, dtype=torch.bool, device=device
        )
        self._final_substep_hit = torch.zeros_like(self._hit)
        # PhysX contact-force buffers are not advanced by an articulation
        # reset.  The first force read after a reset can therefore still
        # describe the terminal pose from the preceding episode even though
        # the articulation has already been restored.  Suppress exactly that
        # first post-reset substep per environment.  A contact that persists
        # into substep two remains observable; the one-substep quarantine is
        # valid only with the separately certified table-clear reset pose.
        self._quarantine_first_sample_after_table_reset = torch.zeros(
            self._num_envs, dtype=torch.bool, device=device
        )
        self._active = False
        self._finalized = False
        self._apply_count = 0
        self._sample_count = 0

    @property
    def hit(self) -> torch.Tensor:
        return self._hit

    @property
    def apply_count(self) -> int:
        return self._apply_count

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def active(self) -> bool:
        return self._active

    def _validate_mask(self, value: torch.Tensor, *, context: str) -> None:
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.bool
            or tuple(value.shape) != (self._num_envs,)
            or value.device != self._hit.device
        ):
            raise RuntimeError(
                f"{context} must be a same-device bool [num_envs] mask"
            )

    def _record_current_hit(self, current_hit: torch.Tensor) -> None:
        """Consume one fresh physics sample with a one-substep reset quarantine."""

        self._validate_mask(current_hit, context="table-contact physics readback")
        self._hit.logical_or_(
            current_hit
            & ~self._quarantine_first_sample_after_table_reset
        )
        self._quarantine_first_sample_after_table_reset.zero_()
        self._sample_count += 1

    def begin_policy_step(self) -> None:
        if self._active and not self._finalized:
            raise RuntimeError(
                "previous table-contact policy step was not finalized"
            )
        self._active = True
        self._finalized = False
        self._apply_count = 0
        self._sample_count = 0

    def record_apply(self, current_hit: torch.Tensor | None) -> None:
        if not self._active or self._finalized:
            raise RuntimeError(
                "table-contact apply readback arrived outside an active policy step"
            )
        if self._apply_count >= self._expected_apply_calls:
            raise RuntimeError(
                "table-contact apply readback count exceeded configured decimation"
            )
        # Apply call zero precedes physics substep one.  Sampling here would import the previous
        # control step's final sensor buffers into the new step.
        if self._apply_count == 0:
            if current_hit is not None:
                raise RuntimeError(
                    "first table-contact apply call must skip stale sensor data"
                )
        else:
            if current_hit is None:
                raise RuntimeError(
                    "table-contact apply readback is missing a current hit mask"
                )
            self._record_current_hit(current_hit)
        self._apply_count += 1

    def finalize(self, current_hit: torch.Tensor) -> torch.Tensor:
        if not self._active:
            raise RuntimeError(
                "table-contact post-step readback has no active policy step"
            )
        if self._finalized:
            return self._hit
        if self._apply_count != self._expected_apply_calls:
            raise RuntimeError(
                "table-contact post-step readback requires exactly "
                f"{self._expected_apply_calls} apply calls"
            )
        self._validate_mask(
            current_hit, context="table-contact post-step readback"
        )
        self._final_substep_hit.copy_(current_hit)
        self._record_current_hit(current_hit)
        if self._sample_count != self._expected_apply_calls:
            raise RuntimeError(
                "table-contact policy step did not sample every physics substep"
            )
        self._finalized = True
        return self._hit

    def reset_envs(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> None:
        if self._active and not self._finalized:
            raise RuntimeError(
                "table-contact reset cannot discard an unfinalized policy step"
            )
        ids = slice(None) if env_ids is None else env_ids
        # Only a final-substep table report can remain as PhysX's next readable
        # report after reset.  An earlier transient table hit already ended
        # with a clean report, while fall/timeout/joint-safety resets have no
        # table report to quarantine.  Preserve an unconsumed quarantine
        # across repeated resets by taking the union before clearing evidence.
        self._quarantine_first_sample_after_table_reset[ids] = (
            self._quarantine_first_sample_after_table_reset[ids]
            | self._final_substep_hit[ids]
        )
        self._hit[ids] = False
        self._final_substep_hit[ids] = False


def _consecutive_physics_timestamp_mask(
    current_timestamp: torch.Tensor,
    previous_timestamp: torch.Tensor,
    physics_dt_s: float,
) -> torch.Tensor:
    """Return rows whose float timestamp advanced by exactly one physics step.

    Isaac sensor timestamps are commonly float32 accumulators.  A fixed nanosecond tolerance
    rejects valid ``0.005`` s increments after only a handful of additions, while a tolerance
    proportional only to the step can hide skipped frames.  Bound normal floating-point
    accumulation error by timestamp magnitude, but cap it at one quarter of a physics step so a
    repeated or skipped substep still fails closed.
    """

    if (
        not torch.is_tensor(current_timestamp)
        or not torch.is_tensor(previous_timestamp)
        or current_timestamp.shape != previous_timestamp.shape
        or current_timestamp.dtype != previous_timestamp.dtype
        or not current_timestamp.dtype.is_floating_point
    ):
        raise RuntimeError(
            "physics timestamps must be same-shape floating-point tensors"
        )
    expected = torch.full_like(current_timestamp, float(physics_dt_s))
    magnitude = torch.maximum(
        torch.maximum(current_timestamp.abs(), previous_timestamp.abs()),
        torch.ones_like(current_timestamp),
    )
    roundoff = magnitude * (8.0 * torch.finfo(current_timestamp.dtype).eps)
    tolerance = torch.clamp(
        roundoff,
        min=8.0 * torch.finfo(current_timestamp.dtype).eps,
        max=float(physics_dt_s) * 0.25,
    )
    delta = current_timestamp - previous_timestamp
    return (
        torch.isfinite(current_timestamp)
        & torch.isfinite(previous_timestamp)
        & ((delta - expected).abs() <= tolerance)
    )


class ClampedJointPositionAction(JointPositionAction):
    """JointPositionAction with an OPTIONAL q_des clamp to the articulation's (soft)
    joint position limits — mirrors the deploy runner's clamp when cfg.clamp=True;
    behaviorally identical to the stock action when cfg.clamp=False (default)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._safety_env = env
        self._clamp_enabled = bool(getattr(cfg, "clamp", False))
        # Diagnostic ActionBall runs are explicitly non-promotable, but they still keep the
        # physical q_des clamp, hard-limit termination and all per-environment safety counters.
        # The command flag is launch-owned and already present on the resolved environment config
        # before ActionManager constructs this term, so read that exact source instead of relying
        # on CommandManager construction order.
        commands_cfg = getattr(getattr(env, "cfg", None), "commands", None)
        racket_cfg = getattr(commands_cfg, "racket_target", None)
        diagnostic_unauthorized = getattr(
            racket_cfg, "action_ball_diagnostic_unauthorized", False
        )
        if type(diagnostic_unauthorized) is not bool:
            raise ValueError(
                "racket_target.action_ball_diagnostic_unauthorized must be an exact boolean"
            )
        self._joint_safety_diagnostic_compact_evidence = bool(
            getattr(racket_cfg, "target_mode", None) == "action_ball"
            and diagnostic_unauthorized
        )
        # The fixed-domain ActionBall screen intentionally bypasses the heavyweight formal
        # joint-safety receipt transaction.  Keep one small device-side diagnostic instead so a
        # reset storm can still be attributed to an exact joint, side, and episode-age bucket.
        # This state is never cleared by per-environment reset and is copied to the host only once
        # at the PPO update boundary by ``consume_actual_joint_forbidden_diagnostic``.
        self._actual_joint_forbidden_diagnostic_enabled = (
            self._joint_safety_diagnostic_compact_evidence
        )
        diagnostic_all_joint_names = tuple(
            str(name)
            for name in getattr(
                self._asset,
                "joint_names",
                getattr(self._asset.data, "joint_names", ()),
            )
        )
        if isinstance(self._joint_ids, slice):
            diagnostic_joint_ids = tuple(
                range(len(diagnostic_all_joint_names))
            )[self._joint_ids]
        else:
            diagnostic_joint_ids = self._joint_ids
            if torch.is_tensor(diagnostic_joint_ids):
                diagnostic_joint_ids = diagnostic_joint_ids.detach().to(
                    device="cpu"
                ).tolist()
            diagnostic_joint_ids = tuple(
                int(joint_id) for joint_id in diagnostic_joint_ids
            )
        if (
            self._actual_joint_forbidden_diagnostic_enabled
            and tuple(diagnostic_joint_ids)
            != tuple(range(len(diagnostic_all_joint_names)))
        ):
            raise ValueError(
                "actual-joint diagnostic requires full articulation identity joint order"
            )
        diagnostic_joint_names = tuple(
            diagnostic_all_joint_names[joint_id]
            for joint_id in diagnostic_joint_ids
        )
        if len(diagnostic_joint_names) != int(self._processed_actions.shape[1]):
            raise ValueError(
                "actual-joint diagnostic joint names must match the protected action order"
            )
        self._actual_joint_forbidden_diagnostic_joint_names = (
            diagnostic_joint_names
        )
        self._actual_joint_forbidden_diagnostic_categories = (
            "current_lower",
            "current_upper",
            "current_nonfinite_or_invalid",
            "substep_actual_hard_edge",
            "pre_apply_nonfinite_qdes",
            "pre_apply_predicted_crossing",
        )
        self._actual_joint_forbidden_diagnostic_counts = torch.zeros(
            (
                2,
                len(self._actual_joint_forbidden_diagnostic_categories),
                self._processed_actions.shape[1],
            ),
            dtype=torch.long,
            device=self._processed_actions.device,
        )
        self._actual_joint_forbidden_diagnostic_event_count = torch.zeros(
            2, dtype=torch.long, device=self._processed_actions.device
        )
        self._actual_joint_forbidden_diagnostic_hard_terminal_count = (
            torch.zeros_like(
                self._actual_joint_forbidden_diagnostic_event_count
            )
        )
        self._actual_joint_forbidden_diagnostic_age_sum = torch.zeros_like(
            self._actual_joint_forbidden_diagnostic_event_count
        )
        self._actual_joint_forbidden_diagnostic_age_max = torch.full_like(
            self._actual_joint_forbidden_diagnostic_event_count, -1
        )
        # Keep deploy-space q_des history next to the action term that owns the affine transform
        # and safety clamp.  ActionManager.prev_action is the previous *normalized actor output*;
        # it cannot attest the target the PD controller actually received after offset/scale/clamp.
        # A separate validity bit is essential because JointAction.reset() clears only raw_actions
        # and intentionally leaves processed_actions untouched in Isaac Lab 2.1.
        self._previous_processed_qdes = torch.zeros_like(self._processed_actions)
        self._previous_processed_qdes_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._processed_qdes_valid = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        # Current affine-transformed target BEFORE the deploy-parity clamp.  This is deliberately
        # not ``raw_actions``: the actor emits normalized actions, while the safety question is
        # whether ``offset + scale * action`` asked the PD controller to drive toward a forbidden
        # joint position.  Keeping this pre-clamp value lets a termination see through the clamp;
        # otherwise a wildly out-of-range request becomes indistinguishable from an exact boundary
        # request after ``torch.clamp``.
        self._pre_clamp_qdes = torch.zeros_like(self._processed_actions)
        self._pre_clamp_qdes_valid = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        # Nearest-point projection of the current affine request into the exact target envelope
        # that may be sent to the drive.  This remains separate from ``processed_actions`` because
        # the pre-apply guard may replace the nominal projection with a state-derived brake target.
        # A dense projection reward therefore needs both tensors to distinguish "the actor asked
        # outside the envelope" from "the guard braked an already-dangerous plant state".
        self._nominal_projected_qdes = torch.zeros_like(self._processed_actions)
        self._nominal_projection_span = torch.zeros_like(self._processed_actions)
        self._nominal_projected_qdes_valid = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        # Optional pre-physics, one-policy-step guard.  It is opt-in so every legacy finite-action
        # path remains numerically unchanged; safety task leaves must bind all parameters
        # explicitly and pin the declared policy horizon to env.step_dt.
        self._pre_apply_limit_guard_enabled = bool(
            getattr(cfg, "pre_apply_limit_guard", False)
        )
        finite_projection = getattr(
            cfg, "project_finite_preclamp_qdes_without_termination", False
        )
        if type(finite_projection) is not bool:
            raise ValueError(
                "project_finite_preclamp_qdes_without_termination must be an exact boolean"
            )
        self._project_finite_preclamp_qdes_without_termination = finite_projection
        projection_soft_inset_fraction = getattr(
            cfg, "finite_projection_soft_envelope_inset_fraction", 0.05
        )
        if finite_projection:
            if (
                isinstance(projection_soft_inset_fraction, bool)
                or not isinstance(projection_soft_inset_fraction, (int, float))
                or not math.isfinite(float(projection_soft_inset_fraction))
                or float(projection_soft_inset_fraction) < 0.0
                or float(projection_soft_inset_fraction) >= 0.5
            ):
                raise ValueError(
                    "finite_projection_soft_envelope_inset_fraction must be finite "
                    "and lie in [0, 0.5)"
                )
            self._finite_projection_soft_envelope_inset_fraction = float(
                projection_soft_inset_fraction
            )
        else:
            # Keep every non-ActionBall clamp target byte-identical.  The candidate inset is owned
            # exclusively by the explicit finite-projection mode.
            self._finite_projection_soft_envelope_inset_fraction = 0.0
        self._pre_apply_guard_policy_dt_s = None
        self._pre_apply_guard_margin_rad = None
        self._pre_apply_guard_margin_fraction = None
        self._pre_apply_guard_decimation = None
        self._pre_apply_guard_physics_dt_s = None
        self._current_substep_guard_envelopes: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None
        self._joint_safety_terminal_archive_capacity = None
        self._table_contact_substep_guard_enabled = bool(
            getattr(cfg, "table_contact_substep_guard", False)
        )
        self._table_contact_guard_termination_term: str | None = None
        self._table_contact_guard_decimation: int | None = None
        self._table_contact_guard_physics_dt_s: float | None = None
        self._table_contact_resolved_params_cache: dict[str, Any] | None = None
        self._table_contact_timestamp_sensors: tuple[Any, ...] | None = None
        self._table_contact_timestamp_data_contract_validated = False
        self._table_contact_last_sensor_timestamp: torch.Tensor | None = None
        self._table_contact_latch: _PhysicsSubstepTableContactLatch | None = None
        if self._table_contact_substep_guard_enabled:
            term_name = getattr(
                cfg, "table_contact_guard_termination_term", None
            )
            expected_table_decimation = getattr(
                cfg, "table_contact_guard_expected_decimation", None
            )
            runtime_table_decimation = getattr(
                getattr(env, "cfg", None), "decimation", None
            )
            runtime_table_physics_dt = getattr(env, "physics_dt", None)
            if not isinstance(term_name, str) or not term_name:
                raise ValueError(
                    "table_contact_guard_termination_term must be a non-empty name"
                )
            if (
                isinstance(expected_table_decimation, bool)
                or not isinstance(expected_table_decimation, int)
                or expected_table_decimation <= 0
                or runtime_table_decimation != expected_table_decimation
            ):
                raise ValueError(
                    "table-contact guard expected decimation must exactly match "
                    "runtime env.cfg.decimation"
                )
            if (
                isinstance(runtime_table_physics_dt, bool)
                or not isinstance(runtime_table_physics_dt, Real)
                or not math.isfinite(float(runtime_table_physics_dt))
                or float(runtime_table_physics_dt) <= 0.0
            ):
                raise ValueError(
                    "table-contact guard requires a finite positive physics_dt"
                )
            self._table_contact_guard_termination_term = term_name
            self._table_contact_guard_decimation = (
                expected_table_decimation
            )
            self._table_contact_guard_physics_dt_s = float(
                runtime_table_physics_dt
            )
            self._table_contact_latch = _PhysicsSubstepTableContactLatch(
                num_envs=self.num_envs,
                expected_apply_calls=expected_table_decimation,
                device=self.device,
            )
        if self._pre_apply_limit_guard_enabled:
            if not self._clamp_enabled:
                raise ValueError(
                    "pre_apply_limit_guard requires the deploy-parity q_des clamp"
                )
            policy_dt = getattr(cfg, "pre_apply_guard_policy_dt_s", None)
            margin_rad = getattr(cfg, "pre_apply_guard_margin_rad", None)
            margin_fraction = getattr(
                cfg, "pre_apply_guard_margin_fraction", None
            )
            if (
                isinstance(policy_dt, bool)
                or not isinstance(policy_dt, Real)
                or not math.isfinite(float(policy_dt))
                or float(policy_dt) <= 0.0
            ):
                raise ValueError(
                    "pre_apply_guard_policy_dt_s must be explicitly finite and > 0 seconds"
                )
            if (
                isinstance(margin_rad, bool)
                or not isinstance(margin_rad, Real)
                or not math.isfinite(float(margin_rad))
                or float(margin_rad) < 0.0
            ):
                raise ValueError(
                    "pre_apply_guard_margin_rad must be explicitly finite and >= 0 radians"
                )
            if (
                isinstance(margin_fraction, bool)
                or not isinstance(margin_fraction, Real)
                or not math.isfinite(float(margin_fraction))
                or not 0.0 <= float(margin_fraction) < 0.5
            ):
                raise ValueError(
                    "pre_apply_guard_margin_fraction must be explicitly finite and in [0, 0.5)"
                )
            runtime_policy_dt = getattr(env, "step_dt", None)
            if (
                isinstance(runtime_policy_dt, bool)
                or not isinstance(runtime_policy_dt, Real)
                or not math.isfinite(float(runtime_policy_dt))
                or float(policy_dt) != float(runtime_policy_dt)
            ):
                raise ValueError(
                    "pre_apply_guard_policy_dt_s must exactly match runtime env.step_dt"
                )
            self._pre_apply_guard_policy_dt_s = float(policy_dt)
            self._pre_apply_guard_margin_rad = float(margin_rad)
            self._pre_apply_guard_margin_fraction = float(margin_fraction)
            runtime_decimation = getattr(getattr(env, "cfg", None), "decimation", None)
            runtime_physics_dt = getattr(env, "physics_dt", None)
            expected_decimation = getattr(
                cfg, "pre_apply_guard_expected_decimation", None
            )
            if (
                isinstance(expected_decimation, bool)
                or not isinstance(expected_decimation, int)
                or expected_decimation <= 0
            ):
                raise ValueError(
                    "pre_apply_guard_expected_decimation must be an explicit "
                    "positive integer"
                )
            if (
                isinstance(runtime_decimation, bool)
                or not isinstance(runtime_decimation, int)
                or runtime_decimation <= 0
                or runtime_decimation != expected_decimation
            ):
                raise ValueError(
                    "pre_apply_limit_guard requires env.cfg.decimation to exactly "
                    "match pre_apply_guard_expected_decimation"
                )
            if (
                isinstance(runtime_physics_dt, bool)
                or not isinstance(runtime_physics_dt, Real)
                or not math.isfinite(float(runtime_physics_dt))
                or float(runtime_physics_dt) <= 0.0
                or not math.isclose(
                    float(runtime_physics_dt) * runtime_decimation,
                    float(policy_dt),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    "pre_apply_limit_guard requires physics_dt * decimation == "
                    "pre_apply_guard_policy_dt_s"
                )
            self._pre_apply_guard_decimation = runtime_decimation
            self._pre_apply_guard_physics_dt_s = float(runtime_physics_dt)
            archive_capacity = getattr(
                cfg, "pre_apply_guard_terminal_archive_capacity", None
            )
            if (
                isinstance(archive_capacity, bool)
                or not isinstance(archive_capacity, int)
                or archive_capacity <= 0
            ):
                raise ValueError(
                    "pre_apply_guard_terminal_archive_capacity must be an explicit "
                    "positive integer"
                )
            self._joint_safety_terminal_archive_capacity = archive_capacity
        if (
            self._project_finite_preclamp_qdes_without_termination
            and not self._pre_apply_limit_guard_enabled
        ):
            raise ValueError(
                "project_finite_preclamp_qdes_without_termination requires the "
                "pre-apply limit guard and deploy-parity q_des clamp"
            )
        self._pre_apply_qdes_violation_latch = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        self._pre_apply_crossing_violation_latch = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        self._pre_apply_qdes_violation_joint_latch = torch.zeros_like(
            self._processed_actions, dtype=torch.bool
        )
        self._pre_apply_crossing_violation_joint_latch = torch.zeros_like(
            self._processed_actions, dtype=torch.bool
        )
        self._pre_apply_qdes_violation_joint_count = torch.zeros_like(
            self._processed_actions, dtype=torch.long
        )
        self._pre_apply_crossing_violation_joint_count = torch.zeros_like(
            self._processed_actions, dtype=torch.long
        )
        self._substep_hard_crossing_latch = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        self._substep_actual_hard_edge_latch = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        self._substep_hard_crossing_joint_latch = torch.zeros_like(
            self._processed_actions, dtype=torch.bool
        )
        self._substep_actual_hard_edge_joint_latch = torch.zeros_like(
            self._processed_actions, dtype=torch.bool
        )
        self._substep_hard_crossing_joint_count = torch.zeros_like(
            self._processed_actions, dtype=torch.long
        )
        self._substep_actual_hard_edge_joint_count = torch.zeros_like(
            self._processed_actions, dtype=torch.long
        )
        self._joint_safety_ledger = None
        if self._pre_apply_limit_guard_enabled:
            assert self._pre_apply_guard_decimation is not None
            assert self._pre_apply_guard_physics_dt_s is not None
            self._joint_safety_ledger = _PhysicsSubstepJointSafetyLedger(
                num_envs=self.num_envs,
                joint_count=self._processed_actions.shape[1],
                expected_apply_calls=self._pre_apply_guard_decimation,
                physics_dt_s=self._pre_apply_guard_physics_dt_s,
                device=self._processed_actions.device,
                dtype=self._processed_actions.dtype,
                retain_dense_records=(
                    not self._joint_safety_diagnostic_compact_evidence
                ),
            )
        self._joint_safety_episode_sequence = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._joint_safety_all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Reuse one device-side duplicate detector for reset batches.  The prior
        # ``torch.unique(...).numel()``/``.item()`` validation synchronized CUDA
        # to the host on every reset, which is especially expensive for the
        # short ActionBall episodes.  CPU callers keep the simple synchronous
        # checks; CUDA uses this fixed-size scratch plus ``_assert_async``.
        self._joint_safety_env_id_validation_counts = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._joint_safety_current_identity: dict[str, Any] | None = None
        self._joint_safety_diagnostic_first_policy_step_sequence: int | None = (
            None
        )
        self._joint_safety_diagnostic_last_policy_step_sequence: int | None = (
            None
        )
        self._joint_safety_cached_action_uid = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._joint_safety_cached_birth_generation = torch.full_like(
            self._joint_safety_cached_action_uid, -1
        )
        self._joint_safety_cached_birth_receipt: list[str | None] = [
            None
        ] * self.num_envs
        self._joint_safety_pending_birth_receipt_env_ids: set[int] = (
            set()
            if self._joint_safety_diagnostic_compact_evidence
            else set(range(self.num_envs))
        )
        self._joint_safety_step_qdes_count_start = torch.zeros_like(
            self._pre_apply_qdes_violation_joint_count
        )
        self._joint_safety_step_crossing_count_start = torch.zeros_like(
            self._pre_apply_crossing_violation_joint_count
        )
        self._joint_safety_current_accumulated_envs = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._joint_safety_current_step_summary_filled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._joint_safety_current_step_complete = torch.zeros_like(
            self._joint_safety_current_step_summary_filled
        )
        self._joint_safety_current_step_apply_count = torch.zeros(
            self.num_envs, dtype=torch.uint8, device=self.device
        )
        self._joint_safety_current_step_post_count = torch.zeros_like(
            self._joint_safety_current_step_apply_count
        )
        self._joint_safety_current_step_timestamp_pass = torch.zeros_like(
            self._joint_safety_current_step_summary_filled
        )
        self._joint_safety_current_step_qdes_joint_count = torch.zeros_like(
            self._processed_actions, dtype=torch.uint8
        )
        self._joint_safety_current_step_policy_crossing_joint_count = (
            torch.zeros_like(self._processed_actions, dtype=torch.uint8)
        )
        self._joint_safety_current_step_substep_crossing_joint_count = (
            torch.zeros_like(self._processed_actions, dtype=torch.uint8)
        )
        self._joint_safety_current_step_actual_hard_edge_joint_count = (
            torch.zeros_like(self._processed_actions, dtype=torch.uint8)
        )
        self._joint_safety_current_step_minimum_hard_gap = torch.full_like(
            self._processed_actions, float("inf")
        )
        self._joint_safety_current_step_summary_published = True
        self._joint_safety_policy_step_summaries: list[dict[str, Any]] = []
        self._joint_safety_policy_step_summary_bytes = 0
        self._joint_safety_policy_step_summary_overflow_latch = False
        self._joint_safety_policy_step_summary_overflow_count = 0
        self._joint_safety_accumulator_consume_sequence = 0
        self._joint_safety_current_accumulator_consume_sequence = 0
        self._joint_safety_accumulator_policy_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._joint_safety_accumulator_complete_steps = torch.zeros_like(
            self._joint_safety_accumulator_policy_steps
        )
        self._joint_safety_accumulator_incomplete_steps = torch.zeros_like(
            self._joint_safety_accumulator_policy_steps
        )
        self._joint_safety_accumulator_apply_readbacks = torch.zeros_like(
            self._joint_safety_accumulator_policy_steps
        )
        self._joint_safety_accumulator_post_readbacks = torch.zeros_like(
            self._joint_safety_accumulator_policy_steps
        )
        self._joint_safety_accumulator_timestamp_passes = torch.zeros_like(
            self._joint_safety_accumulator_policy_steps
        )
        self._joint_safety_accumulator_hard_crossing_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._joint_safety_accumulator_actual_hard_edge_latch = torch.zeros_like(
            self._joint_safety_accumulator_hard_crossing_latch
        )
        self._joint_safety_accumulator_qdes_joint_count = torch.zeros_like(
            self._pre_apply_qdes_violation_joint_count
        )
        self._joint_safety_accumulator_policy_crossing_joint_count = (
            torch.zeros_like(self._pre_apply_crossing_violation_joint_count)
        )
        self._joint_safety_accumulator_substep_crossing_joint_count = (
            torch.zeros_like(self._substep_hard_crossing_joint_count)
        )
        self._joint_safety_accumulator_actual_hard_edge_joint_count = (
            torch.zeros_like(self._substep_actual_hard_edge_joint_count)
        )
        self._joint_safety_accumulator_min_hard_lower_gap = torch.full_like(
            self._processed_actions, float("inf")
        )
        self._joint_safety_accumulator_min_hard_upper_gap = torch.full_like(
            self._processed_actions, float("inf")
        )
        self._joint_safety_terminal_archives: list[dict[str, Any]] = []
        self._joint_safety_terminal_archive_index: dict[
            tuple[int, int], int
        ] = {}
        self._joint_safety_next_archive_sequence = 0
        self._joint_safety_terminal_archive_bytes = 0
        self._joint_safety_archive_overflow_latch = False
        self._joint_safety_archive_overflow_count = 0
        # Runner-side validation/persistence is intentionally two-phase.  A prepared consume
        # freezes the action term until the runner has durably written its evidence and completed
        # the optimizer update.  Only the exact returned token may acknowledge and clear it.
        #
        # The revision plus the structural fingerprint below are deliberately independent from
        # ``consume_sequence``: the latter advances only after acknowledgement, whereas every
        # public physics/reset mutation advances the revision.  This makes a stale acknowledgement
        # fail closed without copying the large ledger a second time.
        self._joint_safety_evidence_revision = 0
        self._joint_safety_next_prepare_sequence = 0
        self._joint_safety_pending_consume_token: tuple[Any, ...] | None = None
        # action_acc(mjlab 档①第三项)原料:raw 动作(actor 归一化输出)的两步历史。
        # isaaclab ActionManager 只存 prev_action(a_{t-1});二阶差分
        # ||a_t - 2a_{t-1} + a_{t-2}||² 还要 a_{t-2},只能在动作项里自存。有效位与上面
        # previous_processed_qdes 同一套路:reset 后前两步没有真实历史,有效位=False ->
        # action_acc_l2 那两步不计费,episode 边界永远造不出虚构的"掉头"罚。
        self._prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._prev_prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._raw_actions_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_raw_actions_valid = torch.zeros_like(self._raw_actions_valid)
        self._prev_prev_raw_actions_valid = torch.zeros_like(self._raw_actions_valid)
        if self._clamp_enabled:
            print("[hope_actions] q_des CLAMP ACTIVE: processed joint targets clamped to "
                  "joint limits (train==deploy, pp_joint_limits parity)", flush=True)

    def _articulation_sim_timestamp(self) -> float:
        timestamp = getattr(self._asset.data, "_sim_timestamp", None)
        return _PhysicsSubstepJointSafetyLedger._finite_timestamp(
            timestamp, context="ClampedJointPositionAction"
        )

    @staticmethod
    def _joint_safety_export_clone(value: Any) -> Any:
        """Recursively clone an export so monitoring code has no mutation channel."""

        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: ClampedJointPositionAction._joint_safety_export_clone(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return tuple(
                ClampedJointPositionAction._joint_safety_export_clone(item)
                for item in value
            )
        return value

    @staticmethod
    def _joint_safety_payload_bytes(value: Any) -> int:
        """Count retained tensor/string payload bytes (excluding Python allocator overhead)."""

        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, dict):
            return sum(
                ClampedJointPositionAction._joint_safety_payload_bytes(item)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return sum(
                ClampedJointPositionAction._joint_safety_payload_bytes(item)
                for item in value
            )
        return 0

    def _joint_safety_refuse_pending_mutation(self, operation: str) -> None:
        """Stop physics/reset mutation while a runner-owned evidence snapshot is pending."""

        if self._joint_safety_pending_consume_token is not None:
            raise RuntimeError(
                "joint-safety ledger consume is prepared but not acknowledged; "
                f"refusing {operation} so the frozen evidence cannot change"
            )

    def _joint_safety_mark_evidence_mutation(self) -> None:
        """Advance the monotonic mutation generation used by consume acknowledgements."""

        self._joint_safety_evidence_revision += 1

    @staticmethod
    def _joint_safety_fingerprint_value(hasher: Any, value: Any) -> None:
        """Fingerprint object identity plus tensor mutation versions without copying payloads.

        Prepared evidence is frozen by the public mutation guards.  The structural fingerprint is
        a second line of defence: tensor ``_version`` detects any in-place private mutation, while
        container/object identity detects replacement or reordering.  Large immutable primitive
        tuples (notably per-environment SHA receipts) need not be re-hashed element by element.
        """

        if torch.is_tensor(value):
            is_inference = bool(torch.is_inference(value))
            if is_inference:
                # Tensors created under ``torch.inference_mode`` deliberately have no version
                # counter.  They are common in rollout-side policy-step summaries, so absence of
                # ``_version`` is a tensor property, not evidence corruption.  Public mutations
                # remain covered by ``_joint_safety_evidence_revision``; identity, storage and
                # metadata below still make replacement/resize/storage drift fail closed.
                tensor_version = None
            else:
                tensor_version = int(value._version)
            descriptor = (
                "tensor",
                id(value),
                is_inference,
                tensor_version,
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                str(value.layout),
                tuple(value.stride()),
                int(value.storage_offset()),
                int(value.data_ptr()),
            )
            hasher.update(repr(descriptor).encode("utf-8"))
            return
        if isinstance(value, dict):
            hasher.update(
                repr(("dict", id(value), len(value))).encode("utf-8")
            )
            for key in sorted(
                value,
                key=lambda item: (type(item).__name__, repr(item)),
            ):
                ClampedJointPositionAction._joint_safety_fingerprint_value(
                    hasher, key
                )
                ClampedJointPositionAction._joint_safety_fingerprint_value(
                    hasher, value[key]
                )
            return
        if isinstance(value, list):
            hasher.update(
                repr(("list", id(value), len(value))).encode("utf-8")
            )
            for item in value:
                ClampedJointPositionAction._joint_safety_fingerprint_value(
                    hasher, item
                )
            return
        if isinstance(value, tuple):
            hasher.update(
                repr(("tuple", id(value), len(value))).encode("utf-8")
            )
            if len(value) > 64 and all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in value
            ):
                # Tuples are immutable; replacing this receipt vector changes its object id.
                return
            for item in value:
                ClampedJointPositionAction._joint_safety_fingerprint_value(
                    hasher, item
                )
            return
        if isinstance(value, set):
            hasher.update(
                repr(
                    (
                        "set",
                        id(value),
                        len(value),
                        tuple(sorted(repr(item) for item in value)),
                    )
                ).encode("utf-8")
            )
            return
        if isinstance(value, _PhysicsSubstepJointSafetyLedger):
            hasher.update(
                repr(("physics_ledger", id(value))).encode("utf-8")
            )
            for name in sorted(value.__dict__):
                hasher.update(name.encode("utf-8"))
                ClampedJointPositionAction._joint_safety_fingerprint_value(
                    hasher, value.__dict__[name]
                )
            return
        hasher.update(
            repr((type(value).__name__, value)).encode("utf-8")
        )

    def _joint_safety_evidence_fingerprint(self) -> str:
        """Return a cheap, mutation-sensitive digest of every retained safety evidence field."""

        hasher = hashlib.sha256()
        excluded = {
            "_joint_safety_pending_consume_token",
            "_joint_safety_next_prepare_sequence",
        }
        included_extra = {
            "_current_substep_guard_envelopes",
            "_pre_clamp_qdes",
            "_pre_clamp_qdes_valid",
            "_processed_actions",
            "_processed_qdes_valid",
            "_previous_processed_qdes",
            "_previous_processed_qdes_valid",
        }
        field_names = sorted(
            name
            for name in self.__dict__
            if name not in excluded
            and (
                name.startswith("_joint_safety_")
                or name.startswith("_pre_apply_")
                or name.startswith("_substep_")
                or name in included_extra
            )
        )
        for name in field_names:
            hasher.update(name.encode("utf-8"))
            self._joint_safety_fingerprint_value(
                hasher, self.__dict__[name]
            )
        return hasher.hexdigest()

    def _joint_safety_env_id_tensor(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> torch.Tensor:
        """Normalize reset/archive ids without accepting duplicates or out-of-range rows."""

        if env_ids is None:
            return self._joint_safety_all_env_ids
        elif isinstance(env_ids, slice):
            return self._joint_safety_all_env_ids[env_ids]
        elif torch.is_tensor(env_ids):
            if (
                env_ids.dtype == torch.bool
                or torch.is_floating_point(env_ids)
                or env_ids.is_complex()
            ):
                raise TypeError(
                    "joint-safety environment ids must be integer values"
                )
            ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            try:
                raw_ids = list(env_ids)
            except TypeError as exc:
                raise TypeError(
                    "joint-safety environment ids must be an integer sequence"
                ) from exc
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_ids
            ):
                raise TypeError(
                    "joint-safety environment ids must be integer values"
                )
            ids = torch.as_tensor(
                raw_ids, dtype=torch.long, device=self.device
            ).reshape(-1)
        if ids.numel() == 0:
            return ids
        in_range = torch.all(ids.ge(0) & ids.lt(self.num_envs))
        if ids.device.type == "cpu":
            if not bool(in_range):
                raise IndexError(
                    "joint-safety environment id is outside the batch"
                )
            if torch.unique(ids).numel() != ids.numel():
                raise ValueError(
                    "joint-safety environment ids must be unique"
                )
        else:
            # Clamp only the validation indices so an out-of-range caller
            # cannot address outside the fixed scratch while the asynchronous
            # fail-closed assertion is pending.  The returned ids remain
            # untouched and therefore can never be silently accepted.
            torch._assert_async(in_range)
            counts = self._joint_safety_env_id_validation_counts
            counts.zero_()
            counts.scatter_add_(
                0,
                torch.clamp(ids, min=0, max=self.num_envs - 1),
                torch.ones_like(ids, dtype=counts.dtype),
            )
            torch._assert_async(torch.all(counts.le(1)))
        return ids

    def _joint_safety_episode_lengths(self) -> torch.Tensor:
        """Return current episode lengths, or an explicit unavailable sentinel."""

        lengths = getattr(self._safety_env, "episode_length_buf", None)
        if lengths is None:
            return torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
        if (
            not torch.is_tensor(lengths)
            or tuple(lengths.shape) != (self.num_envs,)
            or lengths.device != self._processed_actions.device
            or lengths.dtype == torch.bool
            or lengths.dtype.is_floating_point
        ):
            raise RuntimeError(
                "joint-safety identity requires integer episode_length_buf shaped (num_envs,)"
            )
        return lengths.to(dtype=torch.long).detach().clone()

    def _capture_joint_safety_identity(self) -> dict[str, Any]:
        """Capture episode plus immutable action-ball birth identity for this policy step."""

        action_uid = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        birth_generation = torch.full_like(action_uid, -1)
        swing_generation = torch.full_like(action_uid, -1)
        receipts: tuple[str | None, ...] = tuple([None] * self.num_envs)
        action_ball_enabled = False
        manager = getattr(self._safety_env, "command_manager", None)
        command = None
        if manager is not None:
            getter = getattr(manager, "get_term", None)
            if callable(getter):
                try:
                    command = getter("racket_target")
                except (KeyError, ValueError):
                    command = None
        if command is not None:
            enabled = getattr(command, "action_ball_enabled", False)
            if not isinstance(enabled, bool):
                raise RuntimeError("action_ball_enabled identity flag must be bool")
            action_ball_enabled = enabled
        if action_ball_enabled:
            env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            action_uid = command.action_ball_action_uid_for_envs(env_ids)
            birth_generation = command.action_ball_episode_generation
            swing_generation = command.action_ball_swing_generation
            for name, value in (
                ("action_uid", action_uid),
                ("episode_generation", birth_generation),
                ("swing_generation", swing_generation),
            ):
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != (self.num_envs,)
                    or value.device != self._processed_actions.device
                    or value.dtype == torch.bool
                    or value.dtype.is_floating_point
                ):
                    raise RuntimeError(
                        f"joint-safety action-ball {name} must be an integer "
                        "tensor shaped (num_envs,) on the action device"
                    )
            action_uid = action_uid.to(dtype=torch.long).detach().clone()
            birth_generation = (
                birth_generation.to(dtype=torch.long).detach().clone()
            )
            swing_generation = (
                swing_generation.to(dtype=torch.long).detach().clone()
            )
            pending_ids = sorted(
                self._joint_safety_pending_birth_receipt_env_ids
            )
            pending_tensor = torch.as_tensor(
                pending_ids, dtype=torch.long, device=self.device
            )
            identity_matches = (
                action_uid.eq(self._joint_safety_cached_action_uid)
                & birth_generation.eq(
                    self._joint_safety_cached_birth_generation
                )
            )
            if pending_tensor.numel():
                identity_matches[pending_tensor] = True
            identity_stable = torch.all(identity_matches)
            if identity_stable.device.type == "cpu":
                if not bool(identity_stable):
                    raise RuntimeError(
                        "joint-safety action-ball birth identity changed without "
                        "an action reset"
                    )
            else:
                torch._assert_async(identity_stable)
            for env_id in pending_ids:
                receipt = command.action_ball_birth_receipt_sha256(env_id)
                if not isinstance(receipt, str) or len(receipt) != 64:
                    raise RuntimeError(
                        "joint-safety action-ball birth receipt must be a SHA-256 string"
                    )
                try:
                    int(receipt, 16)
                except ValueError as exc:
                    raise RuntimeError(
                        "joint-safety action-ball birth receipt must be hexadecimal"
                    ) from exc
                self._joint_safety_cached_birth_receipt[env_id] = receipt
            if pending_tensor.numel():
                self._joint_safety_cached_action_uid[pending_tensor] = (
                    action_uid[pending_tensor]
                )
                self._joint_safety_cached_birth_generation[pending_tensor] = (
                    birth_generation[pending_tensor]
                )
                self._joint_safety_pending_birth_receipt_env_ids.clear()
            receipts = tuple(self._joint_safety_cached_birth_receipt)
            if pending_ids and any(receipt is None for receipt in receipts):
                raise RuntimeError(
                    "joint-safety action-ball identity cache has a missing birth receipt"
                )
        else:
            self._joint_safety_pending_birth_receipt_env_ids.clear()
        return {
            "action_episode_sequence": (
                self._joint_safety_episode_sequence.detach().clone()
            ),
            "episode_length": self._joint_safety_episode_lengths(),
            "action_ball_enabled": action_ball_enabled,
            "action_uid": action_uid,
            "birth_generation": birth_generation,
            "swing_generation": swing_generation,
            "birth_receipt_sha256": receipts,
        }

    def _joint_safety_step_count_deltas(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qdes_delta = (
            self._pre_apply_qdes_violation_joint_count
            - self._joint_safety_step_qdes_count_start
        )
        crossing_delta = (
            self._pre_apply_crossing_violation_joint_count
            - self._joint_safety_step_crossing_count_start
        )
        valid = torch.all(qdes_delta.ge(0) & crossing_delta.ge(0))
        if valid.device.type == "cpu":
            if not bool(valid):
                raise RuntimeError(
                    "joint-safety policy-step counters moved backwards"
                )
        else:
            torch._assert_async(valid)
        return qdes_delta, crossing_delta

    def _joint_safety_live_snapshot(self) -> dict[str, Any]:
        """Build the live last-policy-step transcript without accumulator recursion."""

        ledger = self._joint_safety_ledger
        if ledger is None:
            return {
                "schema_version": _PhysicsSubstepJointSafetyLedger._SCHEMA_VERSION,
                "enabled": False,
            }
        qdes_delta, policy_crossing_delta = (
            self._joint_safety_step_count_deltas()
        )
        snapshot = ledger.snapshot(
            qdes_env_latch=self._pre_apply_qdes_violation_latch,
            crossing_env_latch=(
                self._pre_apply_crossing_violation_latch
                | self._substep_hard_crossing_latch
                | self._substep_actual_hard_edge_latch
            ),
            qdes_joint_latch=self._pre_apply_qdes_violation_joint_latch,
            crossing_joint_latch=(
                self._pre_apply_crossing_violation_joint_latch
                | self._substep_hard_crossing_joint_latch
                | self._substep_actual_hard_edge_joint_latch
            ),
            qdes_joint_count=self._pre_apply_qdes_violation_joint_count,
            crossing_joint_count=self._pre_apply_crossing_violation_joint_count,
            substep_crossing_joint_latch=(
                self._substep_hard_crossing_joint_latch
            ),
            substep_actual_joint_latch=(
                self._substep_actual_hard_edge_joint_latch
            ),
            substep_crossing_joint_count=(
                self._substep_hard_crossing_joint_count
            ),
            substep_actual_joint_count=(
                self._substep_actual_hard_edge_joint_count
            ),
        )
        snapshot["enabled"] = True
        snapshot["step_qdes_joint_count"] = qdes_delta.detach().clone()
        snapshot["step_policy_crossing_joint_count"] = (
            policy_crossing_delta.detach().clone()
        )
        snapshot["current_action_episode_identity"] = (
            None
            if self._joint_safety_current_identity is None
            else self._joint_safety_export_clone(
                self._joint_safety_current_identity
            )
        )
        return snapshot

    def _accumulate_joint_safety_live(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> None:
        """Fold unconsumed live rows into update-scale statistics exactly once."""

        ledger = self._joint_safety_ledger
        if ledger is None or not ledger.has_started:
            return
        if self._joint_safety_diagnostic_compact_evidence and (
            env_ids is None
            or (
                isinstance(env_ids, slice)
                and env_ids == slice(None)
            )
        ):
            self._accumulate_joint_safety_diagnostic_full_batch(ledger)
            return
        ids = self._joint_safety_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        newly_accumulated = ~self._joint_safety_current_accumulated_envs[ids]
        newly_accumulated_long = newly_accumulated.to(dtype=torch.long)
        newly_accumulated_joint = newly_accumulated[:, None]
        reduced = ledger.aggregate_rows(ids)
        apply_counts = reduced["apply_count"] * newly_accumulated_long
        post_counts = reduced["post_count"] * newly_accumulated_long
        crossing_counts = (
            reduced["hard_crossing_count"]
            * newly_accumulated_joint.to(dtype=torch.long)
        )
        actual_counts = (
            reduced["actual_hard_edge_count"]
            * newly_accumulated_joint.to(dtype=torch.long)
        )
        min_lower = torch.where(
            newly_accumulated_joint,
            reduced["min_lower_gap"],
            torch.full_like(reduced["min_lower_gap"], float("inf")),
        )
        min_upper = torch.where(
            newly_accumulated_joint,
            reduced["min_upper_gap"],
            torch.full_like(reduced["min_upper_gap"], float("inf")),
        )
        expected_records = ledger._expected_apply_calls + 1
        env_complete = (
            ledger.is_complete
            & apply_counts.eq(ledger._expected_apply_calls)
            & post_counts.eq(1)
            & reduced["valid_record_count"].eq(expected_records)
        )
        qdes_delta, policy_crossing_delta = (
            self._joint_safety_step_count_deltas()
        )
        compact_counts_valid = torch.all(
            qdes_delta[ids].le(255)
            & policy_crossing_delta[ids].le(255)
            & crossing_counts.le(255)
            & actual_counts.le(255)
        )
        if compact_counts_valid.device.type == "cpu":
            if not bool(compact_counts_valid):
                raise RuntimeError(
                    "joint-safety per-step compact count exceeded uint8"
                )
        else:
            torch._assert_async(compact_counts_valid)
        self._joint_safety_current_step_summary_filled[ids] |= (
            newly_accumulated
        )
        self._joint_safety_current_step_complete[ids] = torch.where(
            newly_accumulated,
            env_complete,
            self._joint_safety_current_step_complete[ids],
        )
        self._joint_safety_current_step_apply_count[ids] = torch.where(
            newly_accumulated,
            apply_counts.to(dtype=torch.uint8),
            self._joint_safety_current_step_apply_count[ids],
        )
        self._joint_safety_current_step_post_count[ids] = torch.where(
            newly_accumulated,
            post_counts.to(dtype=torch.uint8),
            self._joint_safety_current_step_post_count[ids],
        )
        self._joint_safety_current_step_timestamp_pass[ids] = torch.where(
            newly_accumulated,
            env_complete,
            self._joint_safety_current_step_timestamp_pass[ids],
        )
        compact_mask = newly_accumulated_joint
        for destination, source in (
            (
                self._joint_safety_current_step_qdes_joint_count,
                qdes_delta[ids],
            ),
            (
                self._joint_safety_current_step_policy_crossing_joint_count,
                policy_crossing_delta[ids],
            ),
            (
                self._joint_safety_current_step_substep_crossing_joint_count,
                crossing_counts,
            ),
            (
                self._joint_safety_current_step_actual_hard_edge_joint_count,
                actual_counts,
            ),
        ):
            destination[ids] = torch.where(
                compact_mask,
                source.to(dtype=torch.uint8),
                destination[ids],
            )
        self._joint_safety_current_step_minimum_hard_gap[ids] = torch.where(
            compact_mask,
            torch.minimum(min_lower, min_upper),
            self._joint_safety_current_step_minimum_hard_gap[ids],
        )
        self._joint_safety_accumulator_policy_steps[ids] += (
            newly_accumulated_long
        )
        self._joint_safety_accumulator_complete_steps[ids] += (
            env_complete.to(dtype=torch.long) * newly_accumulated_long
        )
        self._joint_safety_accumulator_incomplete_steps[ids] += (
            (~env_complete).to(dtype=torch.long) * newly_accumulated_long
        )
        self._joint_safety_accumulator_apply_readbacks[ids] += apply_counts
        self._joint_safety_accumulator_post_readbacks[ids] += post_counts
        self._joint_safety_accumulator_timestamp_passes[ids] += (
            env_complete.to(dtype=torch.long) * newly_accumulated_long
        )
        self._joint_safety_accumulator_hard_crossing_latch[ids] |= torch.any(
            crossing_counts.gt(0), dim=1
        )
        self._joint_safety_accumulator_actual_hard_edge_latch[ids] |= torch.any(
            actual_counts.gt(0), dim=1
        )
        self._joint_safety_accumulator_substep_crossing_joint_count[ids] += (
            crossing_counts
        )
        self._joint_safety_accumulator_actual_hard_edge_joint_count[ids] += (
            actual_counts
        )
        self._joint_safety_accumulator_qdes_joint_count[ids] += (
            qdes_delta[ids] * newly_accumulated_joint.to(dtype=torch.long)
        )
        self._joint_safety_accumulator_policy_crossing_joint_count[ids] += (
            policy_crossing_delta[ids]
            * newly_accumulated_joint.to(dtype=torch.long)
        )
        self._joint_safety_accumulator_min_hard_lower_gap[ids] = torch.minimum(
            self._joint_safety_accumulator_min_hard_lower_gap[ids], min_lower
        )
        self._joint_safety_accumulator_min_hard_upper_gap[ids] = torch.minimum(
            self._joint_safety_accumulator_min_hard_upper_gap[ids], min_upper
        )
        self._joint_safety_current_accumulated_envs[ids] |= newly_accumulated

    def _accumulate_joint_safety_diagnostic_full_batch(
        self, ledger: _PhysicsSubstepJointSafetyLedger
    ) -> None:
        """Fold one diagnostic policy step without gather copies or dead summaries.

        Unauthorized ActionBall diagnostics never publish per-policy-step
        identities or dense readback transcripts.  Their formal evidence is the
        update-scale device accumulator consumed by the runner.  The generic
        path above nevertheless gathered every aggregate through a 4096-row
        index tensor and populated five dense summary buffers that the
        diagnostic runner requires to stay empty.  This full-batch path writes
        the exact same counters/minima directly into the accumulator.

        ``newly_accumulated`` is retained rather than assumed all-true so direct
        partial-reset tests and any future manager ordering remain exactly-once.
        """

        newly_accumulated = ~self._joint_safety_current_accumulated_envs
        newly_long = newly_accumulated.to(dtype=torch.long)
        newly_joint = newly_accumulated[:, None]
        apply_counts = ledger._aggregate_apply_count * newly_long
        post_counts = ledger._aggregate_post_count * newly_long
        crossing_counts = (
            ledger._aggregate_hard_crossing_count
            * newly_joint.to(dtype=torch.long)
        )
        actual_counts = (
            ledger._aggregate_actual_hard_edge_count
            * newly_joint.to(dtype=torch.long)
        )
        min_lower = torch.where(
            newly_joint,
            ledger._aggregate_min_lower_gap,
            torch.full_like(
                ledger._aggregate_min_lower_gap, float("inf")
            ),
        )
        min_upper = torch.where(
            newly_joint,
            ledger._aggregate_min_upper_gap,
            torch.full_like(
                ledger._aggregate_min_upper_gap, float("inf")
            ),
        )
        expected_records = ledger._expected_apply_calls + 1
        env_complete = (
            ledger.is_complete
            & apply_counts.eq(ledger._expected_apply_calls)
            & post_counts.eq(1)
            & ledger._aggregate_valid_record_count.eq(expected_records)
        )
        qdes_delta, policy_crossing_delta = (
            self._joint_safety_step_count_deltas()
        )
        qdes_counts = qdes_delta * newly_joint.to(dtype=torch.long)
        policy_crossing_counts = (
            policy_crossing_delta * newly_joint.to(dtype=torch.long)
        )
        compact_counts_valid = torch.all(
            qdes_counts.le(255)
            & policy_crossing_counts.le(255)
            & crossing_counts.le(255)
            & actual_counts.le(255)
        )
        if compact_counts_valid.device.type == "cpu":
            if not bool(compact_counts_valid):
                raise RuntimeError(
                    "joint-safety per-step compact count exceeded uint8"
                )
        else:
            torch._assert_async(compact_counts_valid)

        self._joint_safety_accumulator_policy_steps.add_(newly_long)
        self._joint_safety_accumulator_complete_steps.add_(
            env_complete.to(dtype=torch.long) * newly_long
        )
        self._joint_safety_accumulator_incomplete_steps.add_(
            (~env_complete).to(dtype=torch.long) * newly_long
        )
        self._joint_safety_accumulator_apply_readbacks.add_(apply_counts)
        self._joint_safety_accumulator_post_readbacks.add_(post_counts)
        self._joint_safety_accumulator_timestamp_passes.add_(
            env_complete.to(dtype=torch.long) * newly_long
        )
        self._joint_safety_accumulator_hard_crossing_latch.logical_or_(
            torch.any(crossing_counts.gt(0), dim=1)
        )
        self._joint_safety_accumulator_actual_hard_edge_latch.logical_or_(
            torch.any(actual_counts.gt(0), dim=1)
        )
        self._joint_safety_accumulator_substep_crossing_joint_count.add_(
            crossing_counts
        )
        self._joint_safety_accumulator_actual_hard_edge_joint_count.add_(
            actual_counts
        )
        self._joint_safety_accumulator_qdes_joint_count.add_(qdes_counts)
        self._joint_safety_accumulator_policy_crossing_joint_count.add_(
            policy_crossing_counts
        )
        torch.minimum(
            self._joint_safety_accumulator_min_hard_lower_gap,
            min_lower,
            out=self._joint_safety_accumulator_min_hard_lower_gap,
        )
        torch.minimum(
            self._joint_safety_accumulator_min_hard_upper_gap,
            min_upper,
            out=self._joint_safety_accumulator_min_hard_upper_gap,
        )
        self._joint_safety_current_accumulated_envs.logical_or_(
            newly_accumulated
        )

    def _reset_joint_safety_current_step_summary(self) -> None:
        if self._joint_safety_diagnostic_compact_evidence:
            # Diagnostic PPO updates consume only the update-scale device
            # accumulator.  Clearing and then refilling the dense per-step
            # summary tensors was pure bandwidth: the runner explicitly
            # requires ``identity_bound_policy_steps == ()``.
            self._joint_safety_current_step_summary_published = False
            return
        self._joint_safety_current_step_summary_filled.zero_()
        self._joint_safety_current_step_complete.zero_()
        self._joint_safety_current_step_apply_count.zero_()
        self._joint_safety_current_step_post_count.zero_()
        self._joint_safety_current_step_timestamp_pass.zero_()
        self._joint_safety_current_step_qdes_joint_count.zero_()
        self._joint_safety_current_step_policy_crossing_joint_count.zero_()
        self._joint_safety_current_step_substep_crossing_joint_count.zero_()
        self._joint_safety_current_step_actual_hard_edge_joint_count.zero_()
        self._joint_safety_current_step_minimum_hard_gap.fill_(float("inf"))
        self._joint_safety_current_step_summary_published = False

    def _publish_joint_safety_policy_step_summary(self) -> None:
        """Retain one compact identity-bound batch summary until PPO-update consume."""

        if self._joint_safety_current_step_summary_published:
            return
        ledger = self._joint_safety_ledger
        identity = self._joint_safety_current_identity
        if ledger is None or identity is None:
            raise RuntimeError(
                "joint-safety policy-step summary has no live ledger/identity"
            )
        all_filled = torch.all(
            self._joint_safety_current_step_summary_filled
        )
        if all_filled.device.type == "cpu":
            if not bool(all_filled):
                raise RuntimeError(
                    "joint-safety policy-step summary is missing environment rows"
                )
        else:
            torch._assert_async(all_filled)
        assert self._joint_safety_terminal_archive_capacity is not None
        if (
            len(self._joint_safety_policy_step_summaries)
            >= self._joint_safety_terminal_archive_capacity
        ):
            self._joint_safety_policy_step_summary_overflow_latch = True
            self._joint_safety_policy_step_summary_overflow_count += 1
            raise RuntimeError(
                "joint-safety policy-step summary overflow; runner failed to "
                "consume within the explicit capacity"
            )
        summary = {
            "schema_version": 1,
            "policy_step_sequence": ledger._policy_step_sequence,
            "policy_start_timestamp_s": ledger._policy_start_timestamp_s,
            "expected_apply_calls": ledger._expected_apply_calls,
            "physics_dt_s": ledger._physics_dt_s,
            "included_in_accumulator": True,
            "accumulator_consume_sequence": (
                self._joint_safety_current_accumulator_consume_sequence
            ),
            "full_joint_identity_order": True,
            "count_dtype": "uint8",
            "action_identity": self._joint_safety_export_clone(identity),
            "row_filled": (
                self._joint_safety_current_step_summary_filled.detach().clone()
            ),
            "complete": (
                self._joint_safety_current_step_complete.detach().clone()
            ),
            "apply_readback_count": (
                self._joint_safety_current_step_apply_count.detach().clone()
            ),
            "post_readback_count": (
                self._joint_safety_current_step_post_count.detach().clone()
            ),
            "timestamp_invariant_pass": (
                self._joint_safety_current_step_timestamp_pass.detach().clone()
            ),
            "qdes_joint_count": (
                self._joint_safety_current_step_qdes_joint_count.detach().clone()
            ),
            "policy_crossing_joint_count": (
                self._joint_safety_current_step_policy_crossing_joint_count.detach().clone()
            ),
            "substep_hard_crossing_joint_count": (
                self._joint_safety_current_step_substep_crossing_joint_count.detach().clone()
            ),
            "actual_hard_edge_joint_count": (
                self._joint_safety_current_step_actual_hard_edge_joint_count.detach().clone()
            ),
            "minimum_hard_gap": (
                self._joint_safety_current_step_minimum_hard_gap.detach().clone()
            ),
        }
        payload_bytes = self._joint_safety_payload_bytes(summary)
        summary["payload_bytes"] = payload_bytes
        self._joint_safety_policy_step_summaries.append(summary)
        self._joint_safety_policy_step_summary_bytes += payload_bytes
        self._joint_safety_current_step_summary_published = True

    def _joint_safety_reset_flags(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        terminated = torch.zeros(
            ids.numel(), dtype=torch.bool, device=self.device
        )
        timed_out = torch.zeros_like(terminated)
        manager = getattr(self._safety_env, "termination_manager", None)
        # ManagerBasedRLEnv copies these two tensors before entering _reset_idx.  Prefer those
        # episode-ending snapshots because an individual manager may already have cleared its
        # internal buffers by the time ActionManager.reset reaches this term.
        raw_terminated = getattr(
            self._safety_env, "reset_terminated", None
        )
        raw_time_outs = getattr(
            self._safety_env, "reset_time_outs", None
        )
        if (raw_terminated is None) != (raw_time_outs is None):
            raise RuntimeError(
                "joint-safety reset flags require paired env.reset_terminated/"
                "reset_time_outs from one epoch"
            )
        if raw_terminated is None and manager is not None:
            raw_terminated = getattr(manager, "terminated", None)
            raw_time_outs = getattr(manager, "time_outs", None)
            if (raw_terminated is None) != (raw_time_outs is None):
                raise RuntimeError(
                    "joint-safety reset flags require paired termination-manager "
                    "terminated/time_outs from one epoch"
                )
        if raw_terminated is None or raw_time_outs is None:
            return terminated, timed_out, False
        for name, value in (
            ("terminated", raw_terminated),
            ("time_outs", raw_time_outs),
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != (self.num_envs,)
                or value.device != self._processed_actions.device
                or value.dtype != torch.bool
            ):
                raise RuntimeError(
                    f"joint-safety reset identity requires bool {name} shaped (num_envs,)"
                )
        return raw_terminated[ids].clone(), raw_time_outs[ids].clone(), True

    def _joint_safety_archive_live(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        *,
        reason: str,
        reset_observed: bool,
    ) -> None:
        """Batch-copy terminal rows to CPU before live reset state is cleared."""

        ledger = self._joint_safety_ledger
        if ledger is None or not ledger.has_started:
            return
        if self._joint_safety_diagnostic_compact_evidence:
            # The fixed-domain diagnostic already retains immutable per-policy-
            # step hard-edge counters/minimum gaps plus the device-side
            # joint/side/episode-age attribution ledger.  Materializing one
            # full q/qdot transcript per reset changes no policy input, reward,
            # termination, or optimizer sample, but makes short-episode
            # training O(number of resets) in Python and CPU transfers.  Formal
            # ActionBall keeps the complete reset transcript path below.
            return
        ids = self._joint_safety_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        env_id_list = ids.detach().to(device="cpu").tolist()
        keys = [
            (ledger._policy_step_sequence, env_id)
            for env_id in env_id_list
        ]
        new_count = sum(
            key not in self._joint_safety_terminal_archive_index
            for key in keys
        )
        assert self._joint_safety_terminal_archive_capacity is not None
        if (
            len(self._joint_safety_terminal_archives) + new_count
            > self._joint_safety_terminal_archive_capacity
        ):
            self._joint_safety_archive_overflow_latch = True
            self._joint_safety_archive_overflow_count += 1
            raise RuntimeError(
                "joint-safety terminal archive overflow; refusing the entire "
                "reset batch before overwriting or partially archiving evidence"
            )

        identity = self._joint_safety_current_identity
        if identity is None:
            identity = self._capture_joint_safety_identity()
        qdes_delta, policy_crossing_delta = (
            self._joint_safety_step_count_deltas()
        )
        unsafe_flags = (
            torch.any(qdes_delta[ids].gt(0), dim=1)
            | torch.any(policy_crossing_delta[ids].gt(0), dim=1)
            | ledger.unsafe_env_mask()[ids]
        ).detach().to(device="cpu").tolist()
        terminated, timed_out, termination_available = (
            self._joint_safety_reset_flags(ids)
            if reset_observed
            else (
                torch.zeros(ids.numel(), dtype=torch.bool, device=self.device),
                torch.zeros(ids.numel(), dtype=torch.bool, device=self.device),
                False,
            )
        )
        terminated_cpu = terminated.detach().to(device="cpu")
        timed_out_cpu = timed_out.detach().to(device="cpu")
        final_episode_length_cpu = self._joint_safety_episode_lengths()[
            ids
        ].detach().to(device="cpu")

        def cpu_batch(value: torch.Tensor) -> torch.Tensor:
            return value[ids].detach().to(device="cpu")

        identity_cpu = {
            name: cpu_batch(identity[name])
            for name in (
                "action_episode_sequence",
                "episode_length",
                "action_uid",
                "birth_generation",
                "swing_generation",
            )
        }
        records = ledger._records
        if records:
            env_valid_cpu = torch.stack(
                [record["env_valid"][ids] for record in records], dim=0
            ).detach().to(device="cpu")

            def record_batch(name: str) -> torch.Tensor:
                return torch.stack(
                    [record[name][ids] for record in records], dim=0
                ).detach().to(device="cpu")

            q_cpu = record_batch("q")
            qdot_cpu = record_batch("qdot")
            lower_gap_cpu = record_batch("lower_gap")
            upper_gap_cpu = record_batch("upper_gap")
            hard_crossing_cpu = record_batch("hard_crossing")
            actual_hard_edge_cpu = record_batch("actual_hard_edge")
        else:
            env_valid_cpu = torch.empty(
                (0, ids.numel()), dtype=torch.bool
            )
            empty_float = torch.empty(
                (0, ids.numel(), self._processed_actions.shape[1]),
                dtype=self._processed_actions.dtype,
            )
            empty_bool = torch.empty(
                empty_float.shape, dtype=torch.bool
            )
            q_cpu = empty_float
            qdot_cpu = empty_float.clone()
            lower_gap_cpu = empty_float.clone()
            upper_gap_cpu = empty_float.clone()
            hard_crossing_cpu = empty_bool
            actual_hard_edge_cpu = empty_bool.clone()

        per_env_cpu = {
            "qdes_env_latch": cpu_batch(
                self._pre_apply_qdes_violation_latch
            ),
            "crossing_env_latch": cpu_batch(
                self._pre_apply_crossing_violation_latch
                | self._substep_hard_crossing_latch
                | self._substep_actual_hard_edge_latch
            ),
            "qdes_joint_latch": cpu_batch(
                self._pre_apply_qdes_violation_joint_latch
            ),
            "crossing_joint_latch": cpu_batch(
                self._pre_apply_crossing_violation_joint_latch
                | self._substep_hard_crossing_joint_latch
                | self._substep_actual_hard_edge_joint_latch
            ),
            "qdes_joint_count": cpu_batch(
                self._pre_apply_qdes_violation_joint_count
            ),
            "crossing_joint_count": cpu_batch(
                self._pre_apply_crossing_violation_joint_count
            ),
            "substep_crossing_joint_latch": cpu_batch(
                self._substep_hard_crossing_joint_latch
            ),
            "substep_actual_joint_latch": cpu_batch(
                self._substep_actual_hard_edge_joint_latch
            ),
            "substep_crossing_joint_count": cpu_batch(
                self._substep_hard_crossing_joint_count
            ),
            "substep_actual_joint_count": cpu_batch(
                self._substep_actual_hard_edge_joint_count
            ),
            "step_qdes_joint_count": cpu_batch(qdes_delta),
            "step_policy_crossing_joint_count": cpu_batch(
                policy_crossing_delta
            ),
        }

        if len(env_id_list) != len(keys):
            raise RuntimeError(
                "joint-safety archive identity count does not match selected env count"
            )
        for local_index, (env_id, key) in enumerate(zip(env_id_list, keys)):
            existing_index = self._joint_safety_terminal_archive_index.get(key)
            reasons = (
                ("unsafe", reason)
                if unsafe_flags[local_index]
                else (reason,)
            )
            if existing_index is not None:
                existing = self._joint_safety_terminal_archives[existing_index]
                existing["reasons"] = tuple(
                    dict.fromkeys((*existing["reasons"], *reasons))
                )
                existing["reset_hook_observed"] = (
                    bool(existing["reset_hook_observed"])
                    or reset_observed
                )
                existing["termination_status_available"] = (
                    bool(existing["termination_status_available"])
                    or termination_available
                )
                existing["terminated"] = bool(existing["terminated"]) or bool(
                    terminated_cpu[local_index].item()
                )
                existing["timed_out"] = bool(existing["timed_out"]) or bool(
                    timed_out_cpu[local_index].item()
                )
                old_payload_bytes = int(existing["payload_bytes"])
                new_payload_bytes = self._joint_safety_payload_bytes(existing)
                existing["payload_bytes"] = new_payload_bytes
                self._joint_safety_terminal_archive_bytes += (
                    new_payload_bytes - old_payload_bytes
                )
                continue

            transcript = {
                "schema_version": ledger._SCHEMA_VERSION,
                "policy_step_sequence": ledger._policy_step_sequence,
                "policy_start_timestamp_s": ledger._policy_start_timestamp_s,
                "expected_apply_calls": ledger._expected_apply_calls,
                "physics_dt_s": ledger._physics_dt_s,
                "apply_call_count": ledger._apply_call_count,
                "post_readback_count": ledger._post_readback_count,
                "complete": (
                    ledger.is_complete
                    and bool(env_valid_cpu[:, local_index].all().item())
                ),
                "record_count": len(records),
                "record_kind": tuple(record["kind"] for record in records),
                "call_index": tuple(record["call_index"] for record in records),
                "timestamp_s": tuple(
                    record["timestamp_s"] for record in records
                ),
                "joint_pos_timestamp_s": tuple(
                    record["joint_pos_timestamp_s"] for record in records
                ),
                "joint_vel_timestamp_s": tuple(
                    record["joint_vel_timestamp_s"] for record in records
                ),
                "env_valid": env_valid_cpu[:, local_index].clone(),
                "q": q_cpu[:, local_index].clone(),
                "qdot": qdot_cpu[:, local_index].clone(),
                "hard_lower_gap": lower_gap_cpu[:, local_index].clone(),
                "hard_upper_gap": upper_gap_cpu[:, local_index].clone(),
                "hard_crossing": hard_crossing_cpu[
                    :, local_index
                ].clone(),
                "actual_hard_edge": actual_hard_edge_cpu[
                    :, local_index
                ].clone(),
            }
            for name, batch in per_env_cpu.items():
                transcript[name] = batch[local_index].clone()
            receipt = identity["birth_receipt_sha256"][env_id]
            entry = {
                "archive_sequence": self._joint_safety_next_archive_sequence,
                "env_id": env_id,
                "policy_step_sequence": ledger._policy_step_sequence,
                "action_episode_sequence": int(
                    identity_cpu["action_episode_sequence"][
                        local_index
                    ].item()
                ),
                "episode_length": int(
                    final_episode_length_cpu[local_index].item()
                ),
                "episode_length_at_policy_start": int(
                    identity_cpu["episode_length"][local_index].item()
                ),
                "episode_length_at_reset_hook": int(
                    final_episode_length_cpu[local_index].item()
                ),
                "action_ball_enabled": bool(identity["action_ball_enabled"]),
                "action_uid": int(
                    identity_cpu["action_uid"][local_index].item()
                ),
                "birth_generation": int(
                    identity_cpu["birth_generation"][local_index].item()
                ),
                "swing_generation": int(
                    identity_cpu["swing_generation"][local_index].item()
                ),
                "birth_receipt_sha256": receipt,
                "reasons": tuple(dict.fromkeys(reasons)),
                "reset_hook_observed": reset_observed,
                "termination_status_available": termination_available,
                "terminated": bool(terminated_cpu[local_index].item()),
                "timed_out": bool(timed_out_cpu[local_index].item()),
                "included_in_accumulator": True,
                "accumulator_consume_sequence": (
                    self._joint_safety_current_accumulator_consume_sequence
                ),
                "transcript": transcript,
            }
            payload_bytes = self._joint_safety_payload_bytes(entry)
            entry["payload_bytes"] = payload_bytes
            archive_slot = len(self._joint_safety_terminal_archives)
            self._joint_safety_terminal_archives.append(entry)
            self._joint_safety_terminal_archive_bytes += payload_bytes
            self._joint_safety_terminal_archive_index[key] = archive_slot
            self._joint_safety_next_archive_sequence += 1

    def _joint_safety_accumulator_snapshot(self) -> dict[str, Any]:
        policy_steps = self._joint_safety_accumulator_policy_steps
        return {
            "consume_sequence": self._joint_safety_accumulator_consume_sequence,
            "has_data": bool(
                torch.any(policy_steps.gt(0)).item()
                or self._joint_safety_terminal_archives
                or self._joint_safety_policy_step_summaries
            ),
            "identity_bound_policy_step_count": len(
                self._joint_safety_policy_step_summaries
            ),
            "policy_step_count": policy_steps.detach().clone(),
            "complete_policy_step_count": (
                self._joint_safety_accumulator_complete_steps.detach().clone()
            ),
            "incomplete_policy_step_count": (
                self._joint_safety_accumulator_incomplete_steps.detach().clone()
            ),
            "apply_readback_count": (
                self._joint_safety_accumulator_apply_readbacks.detach().clone()
            ),
            "post_readback_count": (
                self._joint_safety_accumulator_post_readbacks.detach().clone()
            ),
            "timestamp_invariant_pass_count": (
                self._joint_safety_accumulator_timestamp_passes.detach().clone()
            ),
            "hard_crossing_latch": (
                self._joint_safety_accumulator_hard_crossing_latch.detach().clone()
            ),
            "actual_hard_edge_latch": (
                self._joint_safety_accumulator_actual_hard_edge_latch.detach().clone()
            ),
            "qdes_joint_count": (
                self._joint_safety_accumulator_qdes_joint_count.detach().clone()
            ),
            "policy_crossing_joint_count": (
                self._joint_safety_accumulator_policy_crossing_joint_count.detach().clone()
            ),
            "substep_hard_crossing_joint_count": (
                self._joint_safety_accumulator_substep_crossing_joint_count.detach().clone()
            ),
            "actual_hard_edge_joint_count": (
                self._joint_safety_accumulator_actual_hard_edge_joint_count.detach().clone()
            ),
            "minimum_hard_lower_gap": (
                self._joint_safety_accumulator_min_hard_lower_gap.detach().clone()
            ),
            "minimum_hard_upper_gap": (
                self._joint_safety_accumulator_min_hard_upper_gap.detach().clone()
            ),
        }

    def _joint_safety_prepare_view(self) -> dict[str, Any]:
        """Return the frozen, borrowed evidence view used only by the runner.

        Unlike :meth:`joint_safety_ledger_snapshot`, this private consume boundary neither exports
        the duplicate live physics batch nor clones dense tensors.  The pending-consume mutation
        guard plus the acknowledgement fingerprint freeze every referenced object while the
        runner validates and sparsifies on the simulator device.  The view is invalid after
        acknowledgement and must never be exposed as a general monitoring API.
        """

        policy_steps = self._joint_safety_accumulator_policy_steps
        since_last_consume = {
            "consume_sequence": self._joint_safety_accumulator_consume_sequence,
            "has_data": bool(
                torch.any(policy_steps.gt(0)).item()
                or self._joint_safety_terminal_archives
                or self._joint_safety_policy_step_summaries
            ),
            "identity_bound_policy_step_count": len(
                self._joint_safety_policy_step_summaries
            ),
            "policy_step_count": policy_steps.detach(),
            "complete_policy_step_count": (
                self._joint_safety_accumulator_complete_steps.detach()
            ),
            "incomplete_policy_step_count": (
                self._joint_safety_accumulator_incomplete_steps.detach()
            ),
            "apply_readback_count": (
                self._joint_safety_accumulator_apply_readbacks.detach()
            ),
            "post_readback_count": (
                self._joint_safety_accumulator_post_readbacks.detach()
            ),
            "timestamp_invariant_pass_count": (
                self._joint_safety_accumulator_timestamp_passes.detach()
            ),
            "hard_crossing_latch": (
                self._joint_safety_accumulator_hard_crossing_latch.detach()
            ),
            "actual_hard_edge_latch": (
                self._joint_safety_accumulator_actual_hard_edge_latch.detach()
            ),
            "qdes_joint_count": (
                self._joint_safety_accumulator_qdes_joint_count.detach()
            ),
            "policy_crossing_joint_count": (
                self._joint_safety_accumulator_policy_crossing_joint_count.detach()
            ),
            "substep_hard_crossing_joint_count": (
                self._joint_safety_accumulator_substep_crossing_joint_count.detach()
            ),
            "actual_hard_edge_joint_count": (
                self._joint_safety_accumulator_actual_hard_edge_joint_count.detach()
            ),
            "minimum_hard_lower_gap": (
                self._joint_safety_accumulator_min_hard_lower_gap.detach()
            ),
            "minimum_hard_upper_gap": (
                self._joint_safety_accumulator_min_hard_upper_gap.detach()
            ),
        }
        return {
            "schema_version": _PhysicsSubstepJointSafetyLedger._SCHEMA_VERSION,
            "enabled": True,
            "diagnostic_compact_evidence": (
                self._joint_safety_diagnostic_compact_evidence
            ),
            "diagnostic_first_policy_step_sequence": (
                self._joint_safety_diagnostic_first_policy_step_sequence
            ),
            "diagnostic_last_policy_step_sequence": (
                self._joint_safety_diagnostic_last_policy_step_sequence
            ),
            "since_last_consume": since_last_consume,
            # Shallow immutable containers bind the retained entries without copying their dense
            # tensors.  Terminal archives are already retained on CPU.
            "terminal_archives": tuple(self._joint_safety_terminal_archives),
            "identity_bound_policy_steps": tuple(
                self._joint_safety_policy_step_summaries
            ),
            "policy_step_summary_capacity": (
                self._joint_safety_terminal_archive_capacity
            ),
            "policy_step_summary_used": len(
                self._joint_safety_policy_step_summaries
            ),
            "policy_step_summary_payload_bytes": (
                self._joint_safety_policy_step_summary_bytes
            ),
            "policy_step_summary_overflow_latch": (
                self._joint_safety_policy_step_summary_overflow_latch
            ),
            "policy_step_summary_overflow_count": (
                self._joint_safety_policy_step_summary_overflow_count
            ),
            "terminal_archive_capacity": (
                self._joint_safety_terminal_archive_capacity
            ),
            "terminal_archive_used": len(
                self._joint_safety_terminal_archives
            ),
            "terminal_archive_payload_bytes": (
                self._joint_safety_terminal_archive_bytes
            ),
            "terminal_archive_overflow_latch": (
                self._joint_safety_archive_overflow_latch
            ),
            "terminal_archive_overflow_count": (
                self._joint_safety_archive_overflow_count
            ),
        }

    def _clear_joint_safety_consumed_state(self) -> None:
        self._joint_safety_accumulator_policy_steps.zero_()
        self._joint_safety_accumulator_complete_steps.zero_()
        self._joint_safety_accumulator_incomplete_steps.zero_()
        self._joint_safety_accumulator_apply_readbacks.zero_()
        self._joint_safety_accumulator_post_readbacks.zero_()
        self._joint_safety_accumulator_timestamp_passes.zero_()
        self._joint_safety_accumulator_hard_crossing_latch.zero_()
        self._joint_safety_accumulator_actual_hard_edge_latch.zero_()
        self._joint_safety_accumulator_qdes_joint_count.zero_()
        self._joint_safety_accumulator_policy_crossing_joint_count.zero_()
        self._joint_safety_accumulator_substep_crossing_joint_count.zero_()
        self._joint_safety_accumulator_actual_hard_edge_joint_count.zero_()
        self._joint_safety_accumulator_min_hard_lower_gap.fill_(float("inf"))
        self._joint_safety_accumulator_min_hard_upper_gap.fill_(float("inf"))
        self._joint_safety_terminal_archives.clear()
        self._joint_safety_terminal_archive_index.clear()
        self._joint_safety_terminal_archive_bytes = 0
        self._joint_safety_policy_step_summaries.clear()
        self._joint_safety_policy_step_summary_bytes = 0
        self._joint_safety_diagnostic_first_policy_step_sequence = None
        self._joint_safety_diagnostic_last_policy_step_sequence = None
        self._joint_safety_accumulator_consume_sequence += 1

    def _record_physics_joint_safety_readback(
        self,
        *,
        kind: str,
        adjust_target: bool,
    ) -> None:
        """Read fresh q/qdot, update hard-limit ledgers, and optionally brake before a write."""

        if not self._pre_apply_limit_guard_enabled:
            return
        ledger = self._joint_safety_ledger
        if (
            ledger is None
            or self._pre_apply_guard_physics_dt_s is None
            or self._pre_apply_guard_policy_dt_s is None
        ):
            raise RuntimeError("physics-substep joint guard is enabled without a ledger")
        envelopes = self._current_substep_guard_envelopes
        if envelopes is None:
            raise RuntimeError(
                "physics-substep joint guard has no policy-step envelope receipt"
            )
        (
            soft_lower,
            soft_upper,
            hard_lower,
            hard_upper,
            target_lower,
            target_upper,
        ) = envelopes
        data = self._asset.data
        timestamp_before = self._articulation_sim_timestamp()
        # Access the lazy articulation properties before reading _sim_timestamp.  The timestamp
        # transcript then attests which simulator state supplied these tensors.
        joint_pos = data.joint_pos[:, self._joint_ids]
        joint_vel = data.joint_vel[:, self._joint_ids]
        expected = tuple(self._processed_actions.shape)
        for name, value in (("joint_pos", joint_pos), ("joint_vel", joint_vel)):
            if (
                tuple(value.shape) != expected
                or value.device != self._processed_actions.device
                or value.dtype != self._processed_actions.dtype
            ):
                raise RuntimeError(
                    f"physics-substep joint guard requires {name} to match q_des"
                )
        timestamp = self._articulation_sim_timestamp()
        if timestamp != timestamp_before:
            raise RuntimeError(
                "joint-safety articulation timestamp changed while reading q/qdot"
            )
        joint_pos_buffer = getattr(data, "_joint_pos", None)
        joint_vel_buffer = getattr(data, "_joint_vel", None)
        if joint_pos_buffer is None or joint_vel_buffer is None:
            raise RuntimeError(
                "joint-safety requires Isaac ArticulationData q/qdot timestamp buffers"
            )
        joint_pos_timestamp = getattr(joint_pos_buffer, "timestamp", None)
        joint_vel_timestamp = getattr(joint_vel_buffer, "timestamp", None)
        default_qdes = data.default_joint_pos[:, self._joint_ids]
        if (
            tuple(default_qdes.shape) != expected
            or default_qdes.device != self._processed_actions.device
            or default_qdes.dtype != self._processed_actions.dtype
        ):
            raise RuntimeError(
                "physics-substep joint guard requires default_joint_pos to match q_des"
            )
        state_finite = torch.isfinite(joint_pos) & torch.isfinite(joint_vel)
        safe_pos = torch.where(torch.isfinite(joint_pos), joint_pos, default_qdes)
        safe_vel = torch.where(
            torch.isfinite(joint_vel), joint_vel, torch.zeros_like(joint_vel)
        )
        fallback_valid = torch.all(torch.isfinite(safe_pos) & torch.isfinite(safe_vel))
        if fallback_valid.device.type == "cpu":
            if not bool(fallback_valid):
                raise RuntimeError(
                    "physics-substep joint guard has no finite q/qdot fallback"
                )
        else:
            torch._assert_async(fallback_valid)

        lower_gap = joint_pos - hard_lower
        upper_gap = hard_upper - joint_pos
        actual_hard_edge = (
            ~torch.isfinite(joint_pos)
            | lower_gap.le(0.0)
            | upper_gap.le(0.0)
        )
        assert self._pre_apply_guard_margin_rad is not None
        assert self._pre_apply_guard_margin_fraction is not None
        hard_travel = hard_upper - hard_lower
        inset = (
            self._pre_apply_guard_margin_rad
            + self._pre_apply_guard_margin_fraction * hard_travel
        )
        hard_inner_lower = hard_lower + inset
        hard_inner_upper = hard_upper - inset
        # Keep the validated control/reaction horizon at every fresh substep readback.  Shrinking
        # the prediction to one physics tick after the policy-step check lets an implicit drive
        # accelerate outward during the first substep, then notices the crossing only when there is
        # no longer enough room to brake before the terminal inset.  Re-evaluating the same policy
        # horizon from fresh q/qdot is a receding safety guard; it does not alter nominal targets.
        guard_horizon_s = self._pre_apply_guard_policy_dt_s
        ballistic_next = safe_pos + safe_vel * guard_horizon_s
        hard_crossing = (
            ~state_finite
            | safe_pos.le(hard_inner_lower)
            | safe_pos.ge(hard_inner_upper)
            | ballistic_next.le(hard_inner_lower)
            | ballistic_next.ge(hard_inner_upper)
        )

        ledger.record(
            kind=kind,
            timestamp_s=timestamp,
            joint_pos_timestamp_s=joint_pos_timestamp,
            joint_vel_timestamp_s=joint_vel_timestamp,
            q=joint_pos,
            qdot=joint_vel,
            lower_gap=lower_gap,
            upper_gap=upper_gap,
            hard_crossing=hard_crossing,
            actual_hard_edge=actual_hard_edge,
        )
        # Only mutate sticky episode state after the transcript accepted this readback.  A stale
        # timestamp or malformed tensor therefore leaves both ledger and counters unchanged.
        self._substep_hard_crossing_joint_latch.logical_or_(hard_crossing)
        self._substep_actual_hard_edge_joint_latch.logical_or_(actual_hard_edge)
        self._substep_hard_crossing_joint_count.add_(
            hard_crossing.to(dtype=torch.long)
        )
        self._substep_actual_hard_edge_joint_count.add_(
            actual_hard_edge.to(dtype=torch.long)
        )
        self._substep_hard_crossing_latch.logical_or_(
            torch.any(hard_crossing, dim=1)
        )
        self._substep_actual_hard_edge_latch.logical_or_(
            torch.any(actual_hard_edge, dim=1)
        )

        if adjust_target:
            guard = hard_crossing | actual_hard_edge
            brake_target = torch.clamp(
                safe_pos - safe_vel * guard_horizon_s,
                min=target_lower,
                max=target_upper,
            )
            nominal_target = torch.clamp(
                self._processed_actions, min=target_lower, max=target_upper
            )
            self._processed_actions = torch.where(
                guard, brake_target, nominal_target
            )
            safe_target = torch.all(
                torch.isfinite(self._processed_actions)
                & self._processed_actions.ge(soft_lower)
                & self._processed_actions.le(soft_upper)
            )
            if safe_target.device.type == "cpu":
                if not bool(safe_target):
                    raise RuntimeError(
                        "physics-substep joint guard produced an unsafe q_des target"
                    )
            else:
                torch._assert_async(safe_target)

    @staticmethod
    def _assert_table_contact_device(
        condition: torch.Tensor, message: str
    ) -> None:
        """Fail synchronously on CPU and without a device-to-host sync on CUDA."""

        if condition.device.type == "cpu":
            if not bool(condition):
                raise RuntimeError(message)
        else:
            torch._assert_async(condition)

    def _resolved_table_contact_params(self) -> dict[str, Any]:
        """Resolve and cross-bind the table DoneTerm once, outside the physics hot path."""

        if not self._table_contact_substep_guard_enabled:
            raise RuntimeError("table-contact substep guard is not enabled")
        cached = self._table_contact_resolved_params_cache
        if cached is not None:
            return cached
        term_name = self._table_contact_guard_termination_term
        manager = getattr(self._safety_env, "termination_manager", None)
        get_term_cfg = getattr(manager, "get_term_cfg", None)
        if term_name is None or not callable(get_term_cfg):
            raise RuntimeError(
                "table-contact guard cannot resolve its termination term"
            )
        cfg = get_term_cfg(term_name)
        params = getattr(cfg, "params", None)
        if not isinstance(params, dict):
            raise RuntimeError(
                "table-contact guard termination parameters are malformed"
            )
        if params.get("require_substep_latch") is not True:
            raise RuntimeError(
                "table-contact action guard requires the DoneTerm to consume its latch"
            )
        from .terminations import robot_hit_table

        if getattr(cfg, "func", None) is not robot_hit_table or bool(
            getattr(cfg, "time_out", False)
        ):
            raise RuntimeError(
                "table-contact guard must bind the non-timeout robot_hit_table DoneTerm"
            )
        action_name = params.get("action_name")
        action_manager = getattr(self._safety_env, "action_manager", None)
        get_action_term = getattr(action_manager, "get_term", None)
        if (
            not isinstance(action_name, str)
            or not callable(get_action_term)
            or get_action_term(action_name) is not self
        ):
            raise RuntimeError(
                "table-contact DoneTerm action_name does not point back to this action term"
            )

        required = (
            "sensor_cfg",
            "filtered_sensor_cfg",
            "asset_cfg",
            "near_x",
            "surface_z",
        )
        missing = [name for name in required if name not in params]
        if missing:
            raise RuntimeError(
                "table-contact guard DoneTerm is missing resolved parameters: "
                f"{missing}"
            )
        if params.get("full_table_assembly") is True:
            exact_cfgs = params.get("full_table_filtered_sensor_cfgs")
            expected_source_paths = params.get(
                "expected_full_table_source_prim_paths"
            )
            expected_robot_body_names = params.get(
                "expected_full_robot_body_names"
            )
            if (
                not isinstance(exact_cfgs, (tuple, list))
                or len(exact_cfgs) != 0
            ):
                raise RuntimeError(
                    "full table-contact assembly must not install pair-filtered sensors"
                )
            if (
                not isinstance(expected_source_paths, (tuple, list))
                or len(expected_source_paths) != 5
            ):
                raise RuntimeError(
                    "full table-contact assembly requires exact five source prim paths"
                )
            if (
                not isinstance(expected_robot_body_names, (tuple, list))
                or len(expected_robot_body_names) != 32
                or any(
                    not isinstance(name, str) or not name
                    for name in expected_robot_body_names
                )
                or len(set(expected_robot_body_names)) != 32
            ):
                raise RuntimeError(
                    "full table-contact assembly requires the exact ordered "
                    "32-body A3 unfiltered-force contract"
                )
            foot_names = params.get("foot_body_names")
            racket_body_name = params.get("racket_body_name")
            blade_center = params.get(
                "racket_blade_center_offset_wrist_m"
            )
            blade_half = params.get("racket_blade_half_extents_m")
            if (
                not isinstance(foot_names, (tuple, list))
                or len(foot_names) != 2
                or len(set(foot_names)) != 2
                or any(name not in expected_robot_body_names for name in foot_names)
                or not isinstance(racket_body_name, str)
                or racket_body_name not in expected_robot_body_names
                or not isinstance(blade_center, (tuple, list))
                or len(blade_center) != 3
                or not isinstance(blade_half, (tuple, list))
                or len(blade_half) != 3
            ):
                raise RuntimeError(
                    "full table-contact assembly has malformed A3 geometric proxy metadata"
                )
        resolved = dict(params)
        if params.get("full_table_assembly") is True:
            resolved["full_table_filtered_sensor_cfgs"] = ()
            resolved["expected_full_table_source_prim_paths"] = tuple(
                expected_source_paths
            )
            resolved["expected_full_robot_body_names"] = tuple(
                expected_robot_body_names
            )
        self._table_contact_resolved_params_cache = resolved
        return resolved

    def _table_contact_sensor_timestamps(
        self, params: dict[str, Any], *, require_data_fresh: bool
    ) -> torch.Tensor:
        """Read both sensor clocks without synchronizing CUDA to the host.

        At policy start only the raw clock is snapshotted; no force-buffer access is triggered.
        A real substep sample additionally requires ``_timestamp_last_update`` equality after the
        force buffers were read, proving the current physics frame—not a lazy stale frame—was
        consumed.
        """

        sensors = getattr(self, "_table_contact_timestamp_sensors", None)
        if sensors is None:
            if params.get("full_table_assembly") is True:
                # The existing whole-body unfiltered sensor is the only physics reporter.  The
                # five table colliders are static geometry and own no ContactSensor/GPU view.
                sensor_cfgs = (params["sensor_cfg"],)
            else:
                sensor_cfgs = (
                    params["sensor_cfg"],
                    params["filtered_sensor_cfg"],
                )
            sensor_names = tuple(
                getattr(cfg, "name", None) for cfg in sensor_cfgs
            )
            if (
                not sensor_names
                or any(
                    not isinstance(name, str) or not name
                    for name in sensor_names
                )
                or len(set(sensor_names)) != len(sensor_names)
            ):
                raise RuntimeError(
                    "table-contact sensor clocks require non-empty unique sensor names"
                )
            sensors = tuple(
                self._safety_env.scene.sensors[name]
                for name in sensor_names
            )
            for sensor_name, sensor in zip(sensor_names, sensors):
                update_period = getattr(
                    getattr(sensor, "cfg", None), "update_period", None
                )
                timestamp = getattr(sensor, "_timestamp", None)
                if (
                    isinstance(update_period, bool)
                    or not isinstance(update_period, Real)
                    or float(update_period) != 0.0
                ):
                    raise RuntimeError(
                        f"table-contact sensor {sensor_name!r} must update every physics step"
                    )
                if (
                    not torch.is_tensor(timestamp)
                    or tuple(timestamp.shape) != (self.num_envs,)
                    or not timestamp.dtype.is_floating_point
                    or timestamp.device != self._processed_actions.device
                ):
                    raise RuntimeError(
                        f"table-contact sensor {sensor_name!r} did not provide a "
                        "per-environment floating-point physics timestamp"
                    )
            self._table_contact_timestamp_sensors = sensors

        timestamp_stack = torch.stack(
            tuple(sensor._timestamp for sensor in sensors), dim=0
        )
        valid = torch.isfinite(timestamp_stack)
        valid &= timestamp_stack.eq(timestamp_stack[0:1])
        if require_data_fresh:
            if not getattr(
                self,
                "_table_contact_timestamp_data_contract_validated",
                False,
            ):
                for sensor in sensors:
                    timestamp = sensor._timestamp
                    last_update = getattr(
                        sensor, "_timestamp_last_update", None
                    )
                    if (
                        not torch.is_tensor(last_update)
                        or tuple(last_update.shape) != (self.num_envs,)
                        or last_update.dtype != timestamp.dtype
                        or last_update.device != timestamp.device
                    ):
                        raise RuntimeError(
                            "a table-contact sensor did not provide a fresh "
                            "per-environment physics timestamp"
                        )
                self._table_contact_timestamp_data_contract_validated = True
            last_update_stack = torch.stack(
                tuple(
                    sensor._timestamp_last_update for sensor in sensors
                ),
                dim=0,
            )
            valid &= timestamp_stack.eq(last_update_stack)
        self._assert_table_contact_device(
            torch.all(valid),
            "one or more table-contact sensors were stale, non-finite, "
            "or sampled different physics frames",
        )
        # ``stack`` owns fresh storage, so this row remains a valid baseline if
        # Isaac updates the sensor tensors in place before the next substep.
        return timestamp_stack[0]

    def _sample_table_contact_current(self) -> torch.Tensor:
        """Read the exact resolved ``robot_hit_table`` current-sensor kernel."""

        params = self._resolved_table_contact_params()
        from .terminations import sample_robot_table_contact_current

        result = sample_robot_table_contact_current(
            self._safety_env,
            sensor_cfg=params["sensor_cfg"],
            filtered_sensor_cfg=params["filtered_sensor_cfg"],
            full_table_filtered_sensor_cfgs=params.get(
                "full_table_filtered_sensor_cfgs", ()
            ),
            expected_full_table_source_prim_paths=params.get(
                "expected_full_table_source_prim_paths", ()
            ),
            expected_full_robot_body_names=params.get(
                "expected_full_robot_body_names", ()
            ),
            asset_cfg=params["asset_cfg"],
            near_x=params["near_x"],
            surface_z=params["surface_z"],
            force_threshold=params.get("force_threshold", 1.0),
            margin=params.get("margin", 0.02),
            full_table_assembly=params.get("full_table_assembly", False),
            keepout_floor_z=params.get("keepout_floor_z", 0.0),
            body_proxy_radius_m=params.get("body_proxy_radius_m", 0.18),
            foot_proxy_radius_m=params.get("foot_proxy_radius_m", 0.10),
            wrist_proxy_radius_m=params.get("wrist_proxy_radius_m", 0.08),
            foot_body_names=params.get("foot_body_names", ()),
            racket_body_name=params.get(
                "racket_body_name", "right_wrist_yaw_Link"
            ),
            racket_blade_center_offset_wrist_m=params.get(
                "racket_blade_center_offset_wrist_m",
                (0.206194, 0.025474, 0.028020),
            ),
            racket_blade_half_extents_m=params.get(
                "racket_blade_half_extents_m",
                (0.082, 0.008, 0.082),
            ),
        )
        current_timestamp = self._table_contact_sensor_timestamps(
            params, require_data_fresh=True
        )
        previous_timestamp = self._table_contact_last_sensor_timestamp
        physics_dt = self._table_contact_guard_physics_dt_s
        if previous_timestamp is None or physics_dt is None:
            raise RuntimeError(
                "table-contact sensor sampling is missing its policy-step baseline"
            )
        self._assert_table_contact_device(
            torch.all(
                _consecutive_physics_timestamp_mask(
                    current_timestamp, previous_timestamp, physics_dt
                )
            ),
            "table-contact sensor samples are not consecutive physics substeps",
        )
        self._table_contact_last_sensor_timestamp = current_timestamp
        return result

    def apply_actions(self) -> None:
        """Record/guard every physics-substep action write, then dispatch the safe target."""

        if self._table_contact_latch is not None:
            if self._table_contact_latch.apply_count == 0:
                current_table_hit = None
            else:
                current_table_hit = self._sample_table_contact_current()
            self._table_contact_latch.record_apply(current_table_hit)
        if self._pre_apply_limit_guard_enabled:
            self._joint_safety_refuse_pending_mutation("apply_actions")
            self._joint_safety_mark_evidence_mutation()
            self._record_physics_joint_safety_readback(
                kind="apply", adjust_target=True
            )
        super().apply_actions()

    def finalize_table_contact_substep_readback(self) -> torch.Tensor:
        """Idempotently sample the last substep and return the episode-sticky hit mask."""

        latch = self._table_contact_latch
        if latch is None:
            raise RuntimeError("table-contact substep guard is not enabled")
        if latch.finalized:
            return latch.hit
        return latch.finalize(self._sample_table_contact_current())

    def finalize_joint_safety_post_step_readback(self) -> None:
        """Idempotently capture the state after the last physics substep for DoneTerms."""

        if not self._pre_apply_limit_guard_enabled:
            return
        self._joint_safety_refuse_pending_mutation(
            "finalize_joint_safety_post_step_readback"
        )
        ledger = self._joint_safety_ledger
        if ledger is None:
            raise RuntimeError("joint-safety post-step readback has no ledger")
        if ledger.post_readback_recorded:
            return
        self._joint_safety_mark_evidence_mutation()
        self._record_physics_joint_safety_readback(
            kind="post", adjust_target=False
        )
        self._accumulate_joint_safety_live(slice(None))
        if self._joint_safety_diagnostic_compact_evidence:
            sequence = ledger._policy_step_sequence
            previous = (
                self._joint_safety_diagnostic_last_policy_step_sequence
            )
            if previous is not None and sequence != previous + 1:
                raise RuntimeError(
                    "diagnostic joint-safety policy-step sequence is not contiguous"
                )
            if (
                self._joint_safety_diagnostic_first_policy_step_sequence
                is None
            ):
                self._joint_safety_diagnostic_first_policy_step_sequence = (
                    sequence
                )
            self._joint_safety_diagnostic_last_policy_step_sequence = sequence
            # The update-scale device accumulator is the diagnostic evidence.
            # Mark the live step published so the next policy step may begin;
            # formal runs still retain the identity-bound dense summary below.
            self._joint_safety_current_step_summary_published = True
        else:
            self._publish_joint_safety_policy_step_summary()

    def record_actual_joint_forbidden_diagnostic(
        self,
        *,
        current_lower: torch.Tensor,
        current_upper: torch.Tensor,
        current_nonfinite_or_invalid: torch.Tensor,
        observed_event: torch.Tensor,
        hard_terminal: torch.Tensor,
        episode_age: torch.Tensor,
    ) -> None:
        """Accumulate soft-band events and hard terminals without a rollout host sync."""

        if not self._actual_joint_forbidden_diagnostic_enabled:
            return
        expected_joint_shape = tuple(self._processed_actions.shape)
        for name, value in (
            ("current_lower", current_lower),
            ("current_upper", current_upper),
            ("current_nonfinite_or_invalid", current_nonfinite_or_invalid),
        ):
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.bool
                or tuple(value.shape) != expected_joint_shape
                or value.device != self._processed_actions.device
            ):
                raise RuntimeError(
                    f"actual-joint diagnostic {name} must be a same-device bool tensor "
                    f"shaped {expected_joint_shape}"
                )
        for name, value in (
            ("observed_event", observed_event),
            ("hard_terminal", hard_terminal),
        ):
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.bool
                or tuple(value.shape) != (self.num_envs,)
                or value.device != self._processed_actions.device
            ):
                raise RuntimeError(
                    f"actual-joint diagnostic {name} must be a same-device bool tensor "
                    f"shaped [{self.num_envs}]"
                )
        hard_subset = torch.all(~hard_terminal | observed_event)
        if hard_subset.device.type == "cpu":
            if not bool(hard_subset):
                raise RuntimeError(
                    "actual-joint diagnostic hard_terminal must be a subset of observed_event"
                )
        else:
            torch._assert_async(hard_subset)
        if (
            not torch.is_tensor(episode_age)
            or episode_age.dtype == torch.bool
            or torch.is_floating_point(episode_age)
            or tuple(episode_age.shape) != (self.num_envs,)
            or episode_age.device != self._processed_actions.device
        ):
            raise RuntimeError(
                "actual-joint diagnostic episode_age must be a same-device integer tensor "
                f"shaped [{self.num_envs}]"
            )
        substep_actual = self._substep_actual_hard_edge_joint_latch
        qdes_delta, crossing_delta = self._joint_safety_step_count_deltas()
        # Use this policy step's deltas, not the episode-sticky latches.  Predicted crossing is a
        # recoverable brake event and no longer terminates, so a sticky bit would otherwise be
        # charged again on every later inner-band observation in the same episode.
        qdes_request = qdes_delta.gt(0)
        predicted_crossing = crossing_delta.gt(0)
        categories = torch.stack(
            (
                current_lower,
                current_upper,
                current_nonfinite_or_invalid,
                substep_actual,
                qdes_request,
                predicted_crossing,
            ),
            dim=0,
        )
        categories = categories & observed_event.view(1, self.num_envs, 1)
        initial = episode_age <= 1
        age_masks = (
            torch.stack((initial, ~initial), dim=0)
            & observed_event.unsqueeze(0)
        )
        hard_terminal_age_masks = (
            torch.stack((initial, ~initial), dim=0)
            & hard_terminal.unsqueeze(0)
        )
        self._actual_joint_forbidden_diagnostic_counts.add_(
            (
                categories.unsqueeze(0)
                & age_masks.view(2, 1, self.num_envs, 1)
            ).sum(dim=2)
        )
        self._actual_joint_forbidden_diagnostic_event_count.add_(
            age_masks.sum(dim=1)
        )
        self._actual_joint_forbidden_diagnostic_hard_terminal_count.add_(
            hard_terminal_age_masks.sum(dim=1)
        )
        age_long = episode_age.to(dtype=torch.long)
        self._actual_joint_forbidden_diagnostic_age_sum.add_(
            (age_masks * age_long.unsqueeze(0)).sum(dim=1)
        )
        self._actual_joint_forbidden_diagnostic_age_max.copy_(
            torch.maximum(
                self._actual_joint_forbidden_diagnostic_age_max,
                torch.where(
                    age_masks,
                    age_long.unsqueeze(0),
                    torch.full_like(age_masks, -1, dtype=torch.long),
                ).amax(dim=1),
            )
        )

    @property
    def actual_joint_forbidden_diagnostic_enabled(self) -> bool:
        """Whether the non-promotable update-boundary attribution is active."""

        return self._actual_joint_forbidden_diagnostic_enabled

    def consume_actual_joint_forbidden_diagnostic(self) -> dict[str, Any]:
        """Copy and clear the small diagnostic aggregate at one PPO update boundary."""

        if not self._actual_joint_forbidden_diagnostic_enabled:
            return {"enabled": False}
        packed = torch.cat(
            (
                self._actual_joint_forbidden_diagnostic_counts.reshape(-1),
                self._actual_joint_forbidden_diagnostic_event_count,
                self._actual_joint_forbidden_diagnostic_hard_terminal_count,
                self._actual_joint_forbidden_diagnostic_age_sum,
                self._actual_joint_forbidden_diagnostic_age_max,
            )
        ).detach().to(device="cpu")
        packed_values = [int(value) for value in packed.tolist()]
        count_size = self._actual_joint_forbidden_diagnostic_counts.numel()
        counts = torch.tensor(
            packed_values[:count_size], dtype=torch.long
        ).reshape(self._actual_joint_forbidden_diagnostic_counts.shape)
        offset = count_size
        event_count = packed_values[offset : offset + 2]
        hard_terminal_count = packed_values[offset + 2 : offset + 4]
        age_sum = packed_values[offset + 4 : offset + 6]
        age_max = packed_values[offset + 6 : offset + 8]
        age_bucket_names = ("episode_age_le_1", "episode_age_gt_1")
        buckets: dict[str, Any] = {}
        for bucket_index, bucket_name in enumerate(age_bucket_names):
            by_joint = []
            for joint_index, joint_name in enumerate(
                self._actual_joint_forbidden_diagnostic_joint_names
            ):
                category_counts = {
                    category: int(counts[bucket_index, category_index, joint_index])
                    for category_index, category in enumerate(
                        self._actual_joint_forbidden_diagnostic_categories
                    )
                }
                if any(category_counts.values()):
                    by_joint.append(
                        {"joint": joint_name, "counts": category_counts}
                    )
            count = event_count[bucket_index]
            buckets[bucket_name] = {
                "safety_event_count": count,
                "hard_terminal_count": hard_terminal_count[bucket_index],
                "mean_safety_event_episode_age": (
                    float(age_sum[bucket_index]) / float(count)
                    if count > 0
                    else None
                ),
                "max_safety_event_episode_age": (
                    age_max[bucket_index] if age_max[bucket_index] >= 0 else None
                ),
                "by_joint": by_joint,
            }
        self._actual_joint_forbidden_diagnostic_counts.zero_()
        self._actual_joint_forbidden_diagnostic_event_count.zero_()
        self._actual_joint_forbidden_diagnostic_hard_terminal_count.zero_()
        self._actual_joint_forbidden_diagnostic_age_sum.zero_()
        self._actual_joint_forbidden_diagnostic_age_max.fill_(-1)
        return {
            "enabled": True,
            "joint_order": list(
                self._actual_joint_forbidden_diagnostic_joint_names
            ),
            "categories": list(
                self._actual_joint_forbidden_diagnostic_categories
            ),
            "age_buckets": buckets,
            "total_safety_event_count": int(sum(event_count)),
            "total_hard_terminal_count": int(sum(hard_terminal_count)),
        }

    def joint_safety_ledger_snapshot(self) -> dict[str, Any]:
        """Read-only live + since-consume + terminal archive export for monitoring."""

        snapshot = self._joint_safety_live_snapshot()
        if not snapshot["enabled"]:
            return snapshot
        snapshot["since_last_consume"] = (
            self._joint_safety_accumulator_snapshot()
        )
        snapshot["terminal_archives"] = tuple(
            self._joint_safety_export_clone(entry)
            for entry in self._joint_safety_terminal_archives
        )
        snapshot["identity_bound_policy_steps"] = tuple(
            self._joint_safety_export_clone(summary)
            for summary in self._joint_safety_policy_step_summaries
        )
        snapshot["policy_step_summary_capacity"] = (
            self._joint_safety_terminal_archive_capacity
        )
        snapshot["policy_step_summary_used"] = len(
            self._joint_safety_policy_step_summaries
        )
        snapshot["policy_step_summary_payload_bytes"] = (
            self._joint_safety_policy_step_summary_bytes
        )
        snapshot["policy_step_summary_overflow_latch"] = (
            self._joint_safety_policy_step_summary_overflow_latch
        )
        snapshot["policy_step_summary_overflow_count"] = (
            self._joint_safety_policy_step_summary_overflow_count
        )
        snapshot["terminal_archive_capacity"] = (
            self._joint_safety_terminal_archive_capacity
        )
        snapshot["terminal_archive_used"] = len(
            self._joint_safety_terminal_archives
        )
        snapshot["terminal_archive_payload_bytes"] = (
            self._joint_safety_terminal_archive_bytes
        )
        snapshot["terminal_archive_overflow_latch"] = (
            self._joint_safety_archive_overflow_latch
        )
        snapshot["terminal_archive_overflow_count"] = (
            self._joint_safety_archive_overflow_count
        )
        return snapshot

    def prepare_joint_safety_ledger_consume(
        self,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Freeze and export evidence without clearing it.

        The runner must first validate and durably persist the returned snapshot.  It may then run
        the optimizer and call :meth:`acknowledge_joint_safety_ledger` with the opaque token.
        Until acknowledgement, every physics/reset mutation and a second prepare fail closed.
        """

        if self._joint_safety_ledger is None:
            raise RuntimeError(
                "cannot prepare a joint-safety consume when the ledger is disabled"
            )
        self._joint_safety_refuse_pending_mutation("a second consume prepare")
        # Do not call ``joint_safety_ledger_snapshot()`` here.  Its public monitoring contract
        # intentionally includes the full live q/qdot/gap/mask transcript.  That transcript is
        # already represented by the retained identity-bound summaries and terminal archives at a
        # PPO boundary, so exporting it again would duplicate a large device batch without adding
        # update evidence.
        snapshot = self._joint_safety_prepare_view()
        fingerprint = self._joint_safety_evidence_fingerprint()
        token: tuple[Any, ...] = (
            "hope_joint_safety_consume",
            1,
            self._joint_safety_next_prepare_sequence,
            self._joint_safety_accumulator_consume_sequence,
            self._joint_safety_evidence_revision,
            fingerprint,
        )
        self._joint_safety_pending_consume_token = token
        return token, snapshot

    def acknowledge_joint_safety_ledger(
        self, token: tuple[Any, ...]
    ) -> None:
        """Clear exactly the prepared generation after persistence and optimizer success."""

        pending = self._joint_safety_pending_consume_token
        if pending is None:
            raise RuntimeError(
                "joint-safety ledger has no prepared consume to acknowledge"
            )
        if not isinstance(token, tuple) or token != pending:
            raise RuntimeError(
                "joint-safety consume acknowledgement token does not match the "
                "prepared evidence generation"
            )
        if (
            self._joint_safety_next_prepare_sequence != pending[2]
            or self._joint_safety_accumulator_consume_sequence != pending[3]
            or self._joint_safety_evidence_revision != pending[4]
            or self._joint_safety_evidence_fingerprint() != pending[5]
        ):
            raise RuntimeError(
                "joint-safety evidence changed after consume prepare; refusing "
                "destructive acknowledgement"
            )
        self._clear_joint_safety_consumed_state()
        self._joint_safety_pending_consume_token = None
        self._joint_safety_next_prepare_sequence += 1
        self._joint_safety_mark_evidence_mutation()

    def consume_joint_safety_ledger(self) -> dict[str, Any]:
        """Unconditionally reject the obsolete destructive one-shot API.

        This name remains only so stale monitors/runners fail with an actionable error instead of
        silently deleting evidence before validation, durable persistence, and optimizer success.
        Disabled ledgers are not an exception: callers must use the read-only snapshot API for
        monitoring and the explicit prepare/ack transaction for consumption.
        """

        raise RuntimeError(
            "destructive joint-safety one-shot consume is disabled; use "
            "prepare_joint_safety_ledger_consume() and acknowledge only after "
            "validation, durable persistence, and optimizer success"
        )

    def _action_ball_dynamic_ready_env_ids(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Normalize the manager-owned reset subset without a host round-trip."""

        if torch.is_tensor(env_ids):
            if (
                env_ids.dtype == torch.bool
                or torch.is_floating_point(env_ids)
                or env_ids.is_complex()
            ):
                raise TypeError(
                    "action-ball dynamic-ready env ids must be integers"
                )
            ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            raw_ids = list(env_ids)
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_ids
            ):
                raise TypeError(
                    "action-ball dynamic-ready env ids must be integers"
                )
            ids = torch.as_tensor(
                raw_ids, dtype=torch.long, device=self.device
            ).reshape(-1)
        if ids.numel() > 0:
            in_range = torch.all(ids.ge(0) & ids.lt(self.num_envs))
            if in_range.device.type == "cpu":
                if not bool(in_range):
                    raise IndexError(
                        "action-ball dynamic-ready env id is outside the batch"
                    )
            else:
                torch._assert_async(in_range)
        return ids

    def _action_ball_dynamic_ready_manager(self):
        manager = getattr(self._safety_env, "action_manager", None)
        get_term = getattr(manager, "get_term", None)
        if (
            manager is None
            or not callable(get_term)
            or get_term("joint_pos") is not self
        ):
            raise RuntimeError(
                "action-ball dynamic-ready state requires this exact "
                "joint_pos term to be owned by ActionManager"
            )
        manager_action = getattr(manager, "_action", None)
        manager_previous = getattr(manager, "_prev_action", None)
        expected = tuple(self._raw_actions.shape)
        for name, value in (
            ("ActionManager._action", manager_action),
            ("ActionManager._prev_action", manager_previous),
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != expected
                or value.device != self._raw_actions.device
                or value.dtype != self._raw_actions.dtype
            ):
                raise RuntimeError(
                    "action-ball dynamic-ready requires one identity-ordered "
                    f"31-D action term; {name} differs from the term buffer"
                )
        return manager

    def _action_ball_dynamic_ready_target_envelope(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact executable q_des envelope for reset rows."""

        if not self._clamp_enabled:
            raise RuntimeError(
                "action-ball dynamic-ready requires the deploy-parity q_des clamp"
            )
        limits = self._asset.data.soft_joint_pos_limits[
            :, self._joint_ids, :
        ]
        expected = tuple(self._processed_actions.shape) + (2,)
        if (
            tuple(limits.shape) != expected
            or limits.device != self._processed_actions.device
            or limits.dtype != self._processed_actions.dtype
        ):
            raise RuntimeError(
                "action-ball dynamic-ready soft limits differ from q_des"
            )
        lower = limits[..., 0]
        upper = limits[..., 1]
        if self._pre_apply_limit_guard_enabled:
            if (
                self._pre_apply_guard_margin_rad is None
                or self._pre_apply_guard_margin_fraction is None
            ):
                raise RuntimeError(
                    "action-ball dynamic-ready found an incomplete pre-apply guard"
                )
            hard_limits = self._asset.data.joint_pos_limits[
                :, self._joint_ids, :
            ]
            if (
                tuple(hard_limits.shape) != expected
                or hard_limits.device != self._processed_actions.device
                or hard_limits.dtype != self._processed_actions.dtype
            ):
                raise RuntimeError(
                    "action-ball dynamic-ready hard limits differ from q_des"
                )
            hard_lower = hard_limits[..., 0]
            hard_upper = hard_limits[..., 1]
            hard_travel = hard_upper - hard_lower
            inset = (
                self._pre_apply_guard_margin_rad
                + self._pre_apply_guard_margin_fraction * hard_travel
            )
            lower = torch.maximum(lower, hard_lower + inset)
            upper = torch.minimum(upper, hard_upper - inset)
        if self._project_finite_preclamp_qdes_without_termination:
            soft_limits = self._asset.data.soft_joint_pos_limits[
                :, self._joint_ids, :
            ]
            soft_lower = soft_limits[..., 0]
            soft_upper = soft_limits[..., 1]
            projection_inset = (
                self._finite_projection_soft_envelope_inset_fraction
                * (soft_upper - soft_lower)
            )
            lower = torch.maximum(lower, soft_lower + projection_inset)
            upper = torch.minimum(upper, soft_upper - projection_inset)
        return lower, upper

    def snapshot_action_ball_dynamic_ready_state(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> dict[str, Any]:
        """Snapshot exactly the reset-coupled manager and action-term rows."""

        ids = self._action_ball_dynamic_ready_env_ids(env_ids)
        manager = self._action_ball_dynamic_ready_manager()
        return {
            "manager_action": manager._action[ids].clone(),
            "manager_prev_action": manager._prev_action[ids].clone(),
            "raw_actions": self._raw_actions[ids].clone(),
            "processed_actions": self._processed_actions[ids].clone(),
            "previous_processed_qdes": self._previous_processed_qdes[
                ids
            ].clone(),
            "pre_clamp_qdes": self._pre_clamp_qdes[ids].clone(),
            "nominal_projected_qdes": self._nominal_projected_qdes[
                ids
            ].clone(),
            "nominal_projection_span": self._nominal_projection_span[
                ids
            ].clone(),
            "prev_raw_actions": self._prev_raw_actions[ids].clone(),
            "prev_prev_raw_actions": self._prev_prev_raw_actions[
                ids
            ].clone(),
            "processed_qdes_valid": self._processed_qdes_valid[ids].clone(),
            "previous_processed_qdes_valid": (
                self._previous_processed_qdes_valid[ids].clone()
            ),
            "pre_clamp_qdes_valid": self._pre_clamp_qdes_valid[ids].clone(),
            "nominal_projected_qdes_valid": (
                self._nominal_projected_qdes_valid[ids].clone()
            ),
            "raw_actions_valid": self._raw_actions_valid[ids].clone(),
            "prev_raw_actions_valid": self._prev_raw_actions_valid[
                ids
            ].clone(),
            "prev_prev_raw_actions_valid": (
                self._prev_prev_raw_actions_valid[ids].clone()
            ),
        }

    def restore_action_ball_dynamic_ready_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        state: dict[str, Any],
    ) -> None:
        """Restore a prior dynamic-ready transaction snapshot exactly."""

        ids = self._action_ball_dynamic_ready_env_ids(env_ids)
        manager = self._action_ball_dynamic_ready_manager()
        expected_keys = {
            "manager_action",
            "manager_prev_action",
            "raw_actions",
            "processed_actions",
            "previous_processed_qdes",
            "pre_clamp_qdes",
            "nominal_projected_qdes",
            "nominal_projection_span",
            "prev_raw_actions",
            "prev_prev_raw_actions",
            "processed_qdes_valid",
            "previous_processed_qdes_valid",
            "pre_clamp_qdes_valid",
            "nominal_projected_qdes_valid",
            "raw_actions_valid",
            "prev_raw_actions_valid",
            "prev_prev_raw_actions_valid",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise RuntimeError(
                "action-ball dynamic-ready rollback state is malformed"
            )
        float_rows = {
            "manager_action": manager._action,
            "manager_prev_action": manager._prev_action,
            "raw_actions": self._raw_actions,
            "processed_actions": self._processed_actions,
            "previous_processed_qdes": self._previous_processed_qdes,
            "pre_clamp_qdes": self._pre_clamp_qdes,
            "nominal_projected_qdes": self._nominal_projected_qdes,
            "nominal_projection_span": self._nominal_projection_span,
            "prev_raw_actions": self._prev_raw_actions,
            "prev_prev_raw_actions": self._prev_prev_raw_actions,
        }
        bool_rows = {
            "processed_qdes_valid": self._processed_qdes_valid,
            "previous_processed_qdes_valid": (
                self._previous_processed_qdes_valid
            ),
            "pre_clamp_qdes_valid": self._pre_clamp_qdes_valid,
            "nominal_projected_qdes_valid": (
                self._nominal_projected_qdes_valid
            ),
            "raw_actions_valid": self._raw_actions_valid,
            "prev_raw_actions_valid": self._prev_raw_actions_valid,
            "prev_prev_raw_actions_valid": (
                self._prev_prev_raw_actions_valid
            ),
        }
        for name, target in (*float_rows.items(), *bool_rows.items()):
            value = state[name]
            expected_shape = (ids.numel(), *target.shape[1:])
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != expected_shape
                or value.device != target.device
                or value.dtype != target.dtype
            ):
                raise RuntimeError(
                    "action-ball dynamic-ready rollback field differs from "
                    f"its live buffer: {name}"
                )
        for name, target in float_rows.items():
            target[ids] = state[name]
        for name, target in bool_rows.items():
            target[ids] = state[name]

    def install_action_ball_dynamic_ready_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        normalized_action: torch.Tensor,
        hold_qdes: torch.Tensor,
        *,
        capture_rollback: bool = True,
    ) -> dict[str, Any] | None:
        """Atomically install one action-specific actor/q_des reset state.

        Raw-history validity deliberately remains false: the ready value is an
        initialization condition, not a sampled policy transition.  Processed
        history is valid so the first actor step is compared against the actual
        controller target rather than a stale target from the retired episode.

        ``capture_rollback=False`` is reserved for fail-stop diagnostic runs:
        it skips the per-environment rollback clones, and any later exception
        must terminate that run rather than attempt recovery or retry.
        """

        if type(capture_rollback) is not bool:
            raise TypeError("capture_rollback must be one exact bool")
        ids = self._action_ball_dynamic_ready_env_ids(env_ids)
        manager = self._action_ball_dynamic_ready_manager()
        expected_shape = (ids.numel(), self._processed_actions.shape[1])
        for name, value in (
            ("normalized_action", normalized_action),
            ("hold_qdes", hold_qdes),
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != expected_shape
                or value.device != self._processed_actions.device
                or value.dtype != self._processed_actions.dtype
            ):
                raise RuntimeError(
                    "action-ball dynamic-ready "
                    f"{name} must match the selected action rows"
                )
        finite = torch.all(
            torch.isfinite(normalized_action) & torch.isfinite(hold_qdes)
        )
        target_lower, target_upper = (
            self._action_ball_dynamic_ready_target_envelope()
        )
        selected_lower = target_lower[ids]
        selected_upper = target_upper[ids]
        target_span = target_upper - target_lower
        selected_span = target_span[ids]
        structurally_valid = finite & torch.all(
            torch.isfinite(selected_lower)
            & torch.isfinite(selected_upper)
            & selected_lower.lt(selected_upper)
            & hold_qdes.ge(selected_lower)
            & hold_qdes.le(selected_upper)
        )
        if structurally_valid.device.type == "cpu":
            if not bool(structurally_valid):
                raise RuntimeError(
                    "action-ball dynamic-ready contains non-finite state or "
                    "an invalid executable q_des envelope"
                )
        else:
            torch._assert_async(structurally_valid)

        state = (
            self.snapshot_action_ball_dynamic_ready_state(ids)
            if capture_rollback
            else None
        )
        try:
            manager._action[ids] = normalized_action
            manager._prev_action[ids] = normalized_action
            self._raw_actions[ids] = normalized_action
            self._prev_raw_actions[ids] = normalized_action
            self._prev_prev_raw_actions[ids] = normalized_action
            self._processed_actions[ids] = hold_qdes
            self._previous_processed_qdes[ids] = hold_qdes
            self._pre_clamp_qdes[ids] = hold_qdes
            self._nominal_projected_qdes[ids] = hold_qdes
            self._nominal_projection_span[ids] = selected_span
            self._processed_qdes_valid[ids] = True
            self._previous_processed_qdes_valid[ids] = True
            self._pre_clamp_qdes_valid[ids] = True
            self._nominal_projected_qdes_valid[ids] = True
            self._raw_actions_valid[ids] = False
            self._prev_raw_actions_valid[ids] = False
            self._prev_prev_raw_actions_valid[ids] = False
        except Exception:
            if state is not None:
                self.restore_action_ball_dynamic_ready_state(ids, state)
            raise
        return state

    def process_actions(self, actions: torch.Tensor):
        if self._table_contact_latch is not None:
            params = self._resolved_table_contact_params()
            baseline_timestamp = self._table_contact_sensor_timestamps(
                params, require_data_fresh=False
            )
            self._table_contact_latch.begin_policy_step()
            self._table_contact_last_sensor_timestamp = baseline_timestamp
        if self._pre_apply_limit_guard_enabled:
            self._joint_safety_refuse_pending_mutation("process_actions")
            self._joint_safety_mark_evidence_mutation()
            if (
                self._joint_safety_archive_overflow_latch
                or self._joint_safety_policy_step_summary_overflow_latch
            ):
                raise RuntimeError(
                    "joint-safety evidence capacity overflow is sticky; "
                    "the run must stop"
                )
            ledger = self._joint_safety_ledger
            if ledger is None:
                raise RuntimeError("pre-apply joint guard is missing its substep ledger")
            if (
                ledger.has_started
                and ledger.is_complete
                and not self._joint_safety_current_step_summary_published
            ):
                raise RuntimeError(
                    "joint-safety complete policy step was not published before overwrite"
                )
            ledger.begin_policy_step(self._articulation_sim_timestamp())
            if self._joint_safety_diagnostic_compact_evidence:
                # Diagnostic screens are explicitly non-promotable and already
                # bind one frozen action/ball/task identity at the command
                # layer.  Re-cloning every environment's action UID,
                # generation and receipt on every policy step changes no
                # safety decision, reward or optimizer sample.
                self._joint_safety_current_identity = None
            else:
                self._joint_safety_current_identity = (
                    self._capture_joint_safety_identity()
                )
            self._joint_safety_current_accumulator_consume_sequence = (
                self._joint_safety_accumulator_consume_sequence
            )
            self._joint_safety_step_qdes_count_start.copy_(
                self._pre_apply_qdes_violation_joint_count
            )
            self._joint_safety_step_crossing_count_start.copy_(
                self._pre_apply_crossing_violation_joint_count
            )
            self._joint_safety_current_accumulated_envs.zero_()
            self._reset_joint_safety_current_step_summary()
        # raw 动作历史左移一格(a_{t-2} <- a_{t-1} <- 当前 raw):super() 马上会把 raw
        # 覆盖成新动作 a_t,所以搬运必须在 super() 之前。有效位跟着一起移——reset 后要
        # 连续吃到两次真动作,历史才算齐(见 raw_action_history_valid)。
        self._prev_prev_raw_actions.copy_(self._prev_raw_actions)
        self._prev_prev_raw_actions_valid.copy_(self._prev_raw_actions_valid)
        self._prev_raw_actions.copy_(self._raw_actions)
        self._prev_raw_actions_valid.copy_(self._raw_actions_valid)
        # Snapshot the preceding deploy-space target before JointPositionAction overwrites it.
        # On the first action after reset the numeric snapshot is deliberately ignored by the
        # copied validity bit, so an episode boundary can never create a fictitious slew charge.
        self._previous_processed_qdes.copy_(self._processed_actions)
        self._previous_processed_qdes_valid.copy_(self._processed_qdes_valid)
        super().process_actions(actions)
        # ``JointPositionAction.process_actions`` has now applied the configured scale and offset,
        # but the HOPE clamp has not run yet.  Snapshot exactly that deploy-space request.
        self._pre_clamp_qdes.copy_(self._processed_actions)
        self._pre_clamp_qdes_valid.fill_(True)
        # DoneTerms run after physics, so merely remembering NaN/Inf is not enough: torch.clamp
        # preserves NaN and PhysX would consume it before the termination could fire.  Preserve the
        # original non-finite evidence above, but substitute the last valid, already-clamped target
        # for the value actually sent to the drive.  On the first step after reset there is no
        # history, so use that environment's current configured default joint position.  A broken
        # fallback fails before apply_actions rather than laundering a NaN into the simulator.
        default_qdes = self._asset.data.default_joint_pos[:, self._joint_ids]
        if (
            tuple(default_qdes.shape) != tuple(self._processed_actions.shape)
            or default_qdes.device != self._processed_actions.device
            or default_qdes.dtype != self._processed_actions.dtype
        ):
            raise RuntimeError(
                "ClampedJointPositionAction requires default_joint_pos to match the "
                "affine q_des shape/device/dtype"
            )
        previous_is_safe = (
            self._previous_processed_qdes_valid[:, None]
            & torch.isfinite(self._previous_processed_qdes)
        )
        finite_fallback = torch.where(
            previous_is_safe,
            self._previous_processed_qdes,
            default_qdes,
        )
        fallback_valid = torch.all(torch.isfinite(finite_fallback))
        if fallback_valid.device.type == "cpu":
            if not bool(fallback_valid):
                raise RuntimeError(
                    "ClampedJointPositionAction has no finite q_des fallback "
                    "(previous target/default joint position are non-finite)"
                )
        else:
            torch._assert_async(fallback_valid)
        self._processed_actions = torch.where(
            torch.isfinite(self._processed_actions),
            self._processed_actions,
            finite_fallback,
        )
        if self._clamp_enabled:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
            expected_limit_shape = tuple(self._processed_actions.shape) + (2,)
            if (
                tuple(limits.shape) != expected_limit_shape
                or limits.device != self._processed_actions.device
                or limits.dtype != self._processed_actions.dtype
            ):
                raise RuntimeError(
                    "ClampedJointPositionAction requires soft_joint_pos_limits to match "
                    "q_des shape/device/dtype"
                )
            lower = limits[..., 0]
            upper = limits[..., 1]
            travel = upper - lower
            limits_valid = torch.all(
                torch.isfinite(lower)
                & torch.isfinite(upper)
                & travel.gt(0.0)
            )
            if limits_valid.device.type == "cpu":
                if not bool(limits_valid):
                    raise RuntimeError(
                        "ClampedJointPositionAction requires finite soft joint limits "
                        "with lower < upper"
                    )
            else:
                torch._assert_async(limits_valid)

            if self._pre_apply_limit_guard_enabled:
                assert self._pre_apply_guard_policy_dt_s is not None
                assert self._pre_apply_guard_margin_rad is not None
                assert self._pre_apply_guard_margin_fraction is not None
                hard_limits = self._asset.data.joint_pos_limits[
                    :, self._joint_ids, :
                ]
                if (
                    tuple(hard_limits.shape) != expected_limit_shape
                    or hard_limits.device != self._processed_actions.device
                    or hard_limits.dtype != self._processed_actions.dtype
                ):
                    raise RuntimeError(
                        "pre_apply_limit_guard requires joint_pos_limits to match "
                        "q_des shape/device/dtype"
                    )
                hard_lower = hard_limits[..., 0]
                hard_upper = hard_limits[..., 1]
                hard_travel = hard_upper - hard_lower
                inset = (
                    self._pre_apply_guard_margin_rad
                    + self._pre_apply_guard_margin_fraction * hard_travel
                )
                hard_inner_lower = hard_lower + inset
                hard_inner_upper = hard_upper - inset
                envelope_valid = torch.all(
                    torch.isfinite(hard_lower)
                    & torch.isfinite(hard_upper)
                    & hard_travel.gt(0.0)
                    & torch.isfinite(hard_inner_lower)
                    & torch.isfinite(hard_inner_upper)
                    & hard_inner_lower.lt(hard_inner_upper)
                    & lower.ge(hard_lower)
                    & upper.le(hard_upper)
                )
                if envelope_valid.device.type == "cpu":
                    if not bool(envelope_valid):
                        raise RuntimeError(
                            "pre_apply_limit_guard margins consume or invalidate the "
                            "hard joint envelope, or the soft deploy envelope is not "
                            "contained by the hard envelope"
                        )
                else:
                    torch._assert_async(envelope_valid)
                # Commands always remain inside the deploy soft envelope.  If an explicitly
                # configured hard guard inset is narrower still, use the intersection.
                target_lower = torch.maximum(lower, hard_inner_lower)
                target_upper = torch.minimum(upper, hard_inner_upper)
                if self._project_finite_preclamp_qdes_without_termination:
                    # ActionBall keeps the raw Gaussian proposal and PPO log-probability untouched,
                    # but executes a nearest-point projection with five percent of the existing
                    # soft span reserved on each side.  Intersect with the physical guard envelope
                    # rather than replacing it, so this can only increase safety.
                    projection_inset = (
                        self._finite_projection_soft_envelope_inset_fraction * travel
                    )
                    target_lower = torch.maximum(
                        target_lower, lower + projection_inset
                    )
                    target_upper = torch.minimum(
                        target_upper, upper - projection_inset
                    )
                target_envelope_valid = torch.all(target_lower.lt(target_upper))
                if target_envelope_valid.device.type == "cpu":
                    if not bool(target_envelope_valid):
                        raise RuntimeError(
                            "pre_apply_limit_guard hard inset and soft deploy envelope "
                            "have no interior intersection"
                        )
                else:
                    torch._assert_async(target_envelope_valid)
                # Freeze one validated receipt for all four physics writes plus the final
                # readback.  Static URDF/config limits are revalidated once per policy step,
                # rather than launching the same full-batch checks at every substep.
                self._current_substep_guard_envelopes = (
                    lower,
                    upper,
                    hard_lower,
                    hard_upper,
                    target_lower,
                    target_upper,
                )

                pre_qdes = self._pre_clamp_qdes
                qdes_nonfinite = ~torch.isfinite(pre_qdes)
                qdes_forbidden_request = (
                    qdes_nonfinite
                    | pre_qdes.le(hard_inner_lower)
                    | pre_qdes.ge(hard_inner_upper)
                )

                joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
                joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
                for name, value in (
                    ("joint_pos", joint_pos),
                    ("joint_vel", joint_vel),
                ):
                    if (
                        tuple(value.shape) != tuple(self._processed_actions.shape)
                        or value.device != self._processed_actions.device
                        or value.dtype != self._processed_actions.dtype
                    ):
                        raise RuntimeError(
                            "pre_apply_limit_guard requires runtime "
                            f"{name} to match q_des shape/device/dtype"
                        )
                state_finite = torch.isfinite(joint_pos) & torch.isfinite(joint_vel)
                safe_joint_pos = torch.where(
                    torch.isfinite(joint_pos), joint_pos, default_qdes
                )
                safe_joint_vel = torch.where(
                    torch.isfinite(joint_vel),
                    joint_vel,
                    torch.zeros_like(joint_vel),
                )
                ballistic_next = (
                    safe_joint_pos
                    + safe_joint_vel * self._pre_apply_guard_policy_dt_s
                )
                crossing_violation = (
                    ~state_finite
                    | safe_joint_pos.le(hard_inner_lower)
                    | safe_joint_pos.ge(hard_inner_upper)
                    | ballistic_next.le(hard_inner_lower)
                    | ballistic_next.ge(hard_inner_upper)
                )
                # Legacy tasks preserve the historical behavior: any affine request outside the
                # hard-inner envelope is replaced by a brake target and terminates later.  The
                # ActionBall projection mode treats a *finite* request as an ordinary constrained
                # action instead: execute its nearest safe projection and teach the policy through
                # the bounded projection penalty.  Non-finite requests and dangerous plant
                # state/crossing predictions remain brake-and-terminate events.
                qdes_safety_violation = (
                    qdes_nonfinite
                    if self._project_finite_preclamp_qdes_without_termination
                    else qdes_forbidden_request
                )
                per_joint_guard = qdes_safety_violation | crossing_violation
                # These latches feed the formal hard-safety transcript, whose nonzero rows must
                # have a terminal archive before PPO.  A finite constrained-action saturation is
                # not a hard event in projection mode; its separate Reward ledger below preserves
                # side/count/distance evidence without falsely fencing the optimizer.
                self._pre_apply_qdes_violation_joint_latch.logical_or_(
                    qdes_safety_violation
                )
                self._pre_apply_crossing_violation_joint_latch.logical_or_(
                    crossing_violation
                )
                self._pre_apply_qdes_violation_joint_count.add_(
                    qdes_safety_violation.to(dtype=torch.long)
                )
                self._pre_apply_crossing_violation_joint_count.add_(
                    crossing_violation.to(dtype=torch.long)
                )
                self._pre_apply_qdes_violation_latch.logical_or_(
                    torch.any(qdes_safety_violation, dim=1)
                )
                self._pre_apply_crossing_violation_latch.logical_or_(
                    torch.any(crossing_violation, dim=1)
                )

                # No guessed acceleration/deceleration constant: mirror one current-velocity
                # horizon inward, then project to the explicitly inset soft envelope.  This is a
                # conservative derived brake request, not a proof of substep stopping distance.
                brake_target = torch.clamp(
                    safe_joint_pos
                    - safe_joint_vel * self._pre_apply_guard_policy_dt_s,
                    min=target_lower,
                    max=target_upper,
                )
                # Keep the nominal projection finite even when the actor emitted NaN/Inf.
                # The request remains terminal, while RewardManager may still evaluate the
                # projection-distance term before the reset is applied.  Reuse the already
                # validated finite brake target only as the projection anchor; the reward maps
                # the non-finite raw request to one full envelope span independently.
                nominal_source = torch.where(
                    qdes_nonfinite,
                    brake_target,
                    self._processed_actions,
                )
                nominal_target = torch.clamp(
                    nominal_source,
                    min=target_lower,
                    max=target_upper,
                )
                self._nominal_projected_qdes.copy_(nominal_target)
                self._nominal_projection_span.copy_(
                    target_upper - target_lower
                )
                self._processed_actions = torch.where(
                    per_joint_guard, brake_target, nominal_target
                )
            else:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=lower, max=upper
                )
                self._nominal_projected_qdes.copy_(self._processed_actions)
                self._nominal_projection_span.copy_(upper - lower)

            processed_safe = torch.all(
                torch.isfinite(self._processed_actions)
                & self._processed_actions.ge(lower)
                & self._processed_actions.le(upper)
            )
            if processed_safe.device.type == "cpu":
                if not bool(processed_safe):
                    raise RuntimeError(
                        "ClampedJointPositionAction produced a non-finite or "
                        "out-of-soft-envelope q_des"
                    )
            else:
                torch._assert_async(processed_safe)
        self._processed_qdes_valid.fill_(True)
        self._nominal_projected_qdes_valid.fill_(bool(self._clamp_enabled))
        self._raw_actions_valid.fill_(True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate deploy-space history for reset environments.

        Isaac's ActionManager resolves ``None`` to ``slice(None)`` before calling terms, but
        accepting ``None`` here as well keeps direct callers safe and matches the base API.
        """

        if self._table_contact_latch is not None:
            self._table_contact_latch.reset_envs(env_ids)
        if self._joint_safety_ledger is not None:
            self._joint_safety_refuse_pending_mutation("reset")
            self._joint_safety_mark_evidence_mutation()
            ids_tensor = self._joint_safety_env_id_tensor(env_ids)
            # Isaac resets terminal rows inside env.step, before the runner can inspect them.
            # Fold and archive first; only then may live state be invalidated and cleared.
            if (
                self._joint_safety_diagnostic_compact_evidence
                and self._joint_safety_ledger.post_readback_recorded
            ):
                # The post-step DoneTerm already folded the entire batch through
                # the compact full-batch path.  Re-entering the generic indexed
                # reducer here used to gather every per-joint aggregate again
                # for rows whose exactly-once bit was already set.
                accumulated = torch.all(
                    self._joint_safety_current_accumulated_envs[ids_tensor]
                )
                if accumulated.device.type == "cpu":
                    if not bool(accumulated):
                        raise RuntimeError(
                            "diagnostic reset observed rows not accumulated by "
                            "the completed post-step readback"
                        )
                else:
                    torch._assert_async(accumulated)
            else:
                self._accumulate_joint_safety_live(ids_tensor)
            self._joint_safety_archive_live(
                ids_tensor, reason="reset", reset_observed=True
            )
            ids: Sequence[int] | torch.Tensor | slice = ids_tensor
        else:
            # Preserve the stock reset call/index path when the opt-in guard is disabled.
            ids = slice(None) if env_ids is None else env_ids
        super().reset(env_ids=env_ids)
        self._processed_qdes_valid[ids] = False
        self._previous_processed_qdes_valid[ids] = False
        self._pre_clamp_qdes[ids] = 0.0
        self._pre_clamp_qdes_valid[ids] = False
        self._nominal_projected_qdes[ids] = 0.0
        self._nominal_projection_span[ids] = 0.0
        self._nominal_projected_qdes_valid[ids] = False
        self._pre_apply_qdes_violation_latch[ids] = False
        self._pre_apply_crossing_violation_latch[ids] = False
        self._pre_apply_qdes_violation_joint_latch[ids] = False
        self._pre_apply_crossing_violation_joint_latch[ids] = False
        self._pre_apply_qdes_violation_joint_count[ids] = 0
        self._pre_apply_crossing_violation_joint_count[ids] = 0
        self._substep_hard_crossing_latch[ids] = False
        self._substep_actual_hard_edge_latch[ids] = False
        self._substep_hard_crossing_joint_latch[ids] = False
        self._substep_actual_hard_edge_joint_latch[ids] = False
        self._substep_hard_crossing_joint_count[ids] = 0
        self._substep_actual_hard_edge_joint_count[ids] = 0
        self._joint_safety_step_qdes_count_start[ids] = 0
        self._joint_safety_step_crossing_count_start[ids] = 0
        if self._joint_safety_ledger is not None:
            self._joint_safety_ledger.reset_envs(ids)
            self._joint_safety_episode_sequence[ids] += 1
            if not self._joint_safety_diagnostic_compact_evidence:
                self._joint_safety_pending_birth_receipt_env_ids.update(
                    ids.detach().to(device="cpu").tolist()
                )
        # raw 动作历史清零 + 有效位清 False(清零对齐 ActionManager 对 action/prev_action
        # 的 reset 语义;有效位保证清零值不会被 action_acc_l2 当成真历史计费)。
        self._prev_raw_actions[ids] = 0.0
        self._prev_prev_raw_actions[ids] = 0.0
        self._raw_actions_valid[ids] = False
        self._prev_raw_actions_valid[ids] = False
        self._prev_prev_raw_actions_valid[ids] = False

    @property
    def previous_processed_qdes(self) -> torch.Tensor:
        """Previous affine-transformed and clamp-applied joint-position target."""

        return self._previous_processed_qdes

    @property
    def previous_processed_qdes_valid(self) -> torch.Tensor:
        """Per-environment validity of :attr:`previous_processed_qdes`."""

        return self._previous_processed_qdes_valid

    @property
    def pre_clamp_qdes(self) -> torch.Tensor:
        """Current affine-transformed joint target before the deploy-parity clamp.

        This tensor is in articulation joint-position units (radians for A3 revolute joints), not
        normalized policy-action units.  Consult :attr:`pre_clamp_qdes_valid` before using a row.
        """

        return self._pre_clamp_qdes

    @property
    def pre_clamp_qdes_valid(self) -> torch.Tensor:
        """Per-environment validity of :attr:`pre_clamp_qdes`.

        Reset environments are invalid until their first subsequent call to
        :meth:`process_actions`, so stale targets can never terminate a newly reset episode.
        """

        return self._pre_clamp_qdes_valid

    @property
    def nominal_projected_qdes(self) -> torch.Tensor:
        """Nearest safe target-envelope projection before any plant-state brake override."""

        return self._nominal_projected_qdes

    @property
    def nominal_projection_span(self) -> torch.Tensor:
        """Per-environment target-envelope width used to normalize projection distance."""

        return self._nominal_projection_span

    @property
    def nominal_projected_qdes_valid(self) -> torch.Tensor:
        """Per-environment validity of the current nominal projection and span."""

        return self._nominal_projected_qdes_valid

    @property
    def finite_preclamp_qdes_projection_enabled(self) -> bool:
        """Whether finite raw affine requests are projected and shaped instead of terminal."""

        return self._project_finite_preclamp_qdes_without_termination

    @property
    def finite_projection_soft_envelope_inset_fraction(self) -> float:
        """Per-side soft-span reserve used by the finite ActionBall projection."""

        return self._finite_projection_soft_envelope_inset_fraction

    @property
    def pre_apply_joint_safety_latch(self) -> torch.Tensor:
        """Sticky per-env pre-physics q_des/state-crossing violation; reset clears it."""

        return (
            self._pre_apply_qdes_violation_latch
            | self._pre_apply_crossing_violation_latch
            | self._substep_hard_crossing_latch
            | self._substep_actual_hard_edge_latch
        )

    @property
    def physical_hard_safety_latch(self) -> torch.Tensor:
        """Plant-state/crossing hard-safety union, excluding a finite projected request."""

        return (
            self._pre_apply_crossing_violation_latch
            | self._substep_hard_crossing_latch
            | self._substep_actual_hard_edge_latch
        )

    @property
    def pre_apply_qdes_violation_latch(self) -> torch.Tensor:
        """Sticky terminal q_des latch.

        Legacy mode includes finite hard-inner requests.  ActionBall projection mode records only
        non-finite requests here; finite saturation has its own non-terminal Reward ledger.
        """

        return self._pre_apply_qdes_violation_latch

    @property
    def pre_apply_crossing_violation_latch(self) -> torch.Tensor:
        """Sticky per-env current/ballistic joint-state envelope-crossing violation."""

        return self._pre_apply_crossing_violation_latch

    @property
    def pre_apply_qdes_violation_joint_latch(self) -> torch.Tensor:
        """Sticky q_des-envelope violation bits in exact action/articulation joint order."""

        return self._pre_apply_qdes_violation_joint_latch

    @property
    def pre_apply_crossing_violation_joint_latch(self) -> torch.Tensor:
        """Sticky current/ballistic crossing bits in exact articulation joint order."""

        return self._pre_apply_crossing_violation_joint_latch

    @property
    def pre_apply_qdes_violation_joint_count(self) -> torch.Tensor:
        """Per-episode q_des guard activation count for every env × joint."""

        return self._pre_apply_qdes_violation_joint_count

    @property
    def pre_apply_crossing_violation_joint_count(self) -> torch.Tensor:
        """Per-episode state-crossing guard activation count for every env × joint."""

        return self._pre_apply_crossing_violation_joint_count

    @property
    def physics_substep_hard_crossing_latch(self) -> torch.Tensor:
        """Sticky per-env hard-crossing prediction observed by apply/post readbacks."""

        return self._substep_hard_crossing_latch

    @property
    def physics_substep_actual_hard_edge_latch(self) -> torch.Tensor:
        """Sticky per-env actual hard-edge/non-finite q readback."""

        return self._substep_actual_hard_edge_latch

    @property
    def physics_substep_hard_crossing_joint_latch(self) -> torch.Tensor:
        return self._substep_hard_crossing_joint_latch

    @property
    def physics_substep_actual_hard_edge_joint_latch(self) -> torch.Tensor:
        return self._substep_actual_hard_edge_joint_latch

    @property
    def physics_substep_hard_crossing_joint_count(self) -> torch.Tensor:
        return self._substep_hard_crossing_joint_count

    @property
    def physics_substep_actual_hard_edge_joint_count(self) -> torch.Tensor:
        return self._substep_actual_hard_edge_joint_count

    @property
    def prev_raw_actions(self) -> torch.Tensor:
        """上一步 raw 动作 a_{t-1}(actor 归一化输出;与 ActionManager.prev_action 同值,
        自存一份让 action_acc_l2 的三份原料同源、同一套 reset 语义)。"""

        return self._prev_raw_actions

    @property
    def prev_prev_raw_actions(self) -> torch.Tensor:
        """上上步 raw 动作 a_{t-2}(isaaclab 不存这一步,action_acc_l2 的自存缓冲)。"""

        return self._prev_prev_raw_actions

    @property
    def raw_action_history_valid(self) -> torch.Tensor:
        """Per-env bool:a_{t-1} 与 a_{t-2} 都是真历史才 True(reset 后前两步 False)。"""

        return self._prev_raw_actions_valid & self._prev_prev_raw_actions_valid


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction
    # ON by default (franco 2026-07-06, after jiayi's P2-cannot-stand-in-MuJoCo finding).
    # Set `actions: qdes_clamp: false` in a task YAML ONLY for legacy-reproduction arms.
    clamp: bool = True
    # Opt-in pre-physics limit-crossing guard.  OFF keeps every legacy finite-action trajectory
    # byte-identical.  Safety task leaves must explicitly bind all three values below; ``None`` is
    # intentional so enabling the guard can never inherit a guessed time horizon or safety inset.
    pre_apply_limit_guard: bool = False
    pre_apply_guard_policy_dt_s: float | None = None
    pre_apply_guard_margin_rad: float | None = None
    pre_apply_guard_margin_fraction: float | None = None
    pre_apply_guard_expected_decimation: int | None = None
    # ActionBall-only constrained-action mode.  The raw Gaussian action and PPO log-probability are
    # untouched; finite affine q_des requests outside the hard-inner envelope are projected into the
    # existing safe target envelope and receive a dense projection penalty instead of terminating.
    # Legacy/default tasks retain the historical terminal behavior.
    project_finite_preclamp_qdes_without_termination: bool = False
    # Per-side reserve inside the existing soft q_des envelope for the ActionBall finite-projection
    # mode.  Five percent of the soft span leaves the current four N1 teacher trajectories inside
    # the executable envelope while adding plant-state overshoot room.  Ignored when projection mode
    # is off, preserving every non-ActionBall target byte-for-byte.
    finite_projection_soft_envelope_inset_fraction: float = 0.05
    # Explicit finite queue bound for terminal/unsafe full transcripts.  ``None`` is deliberately
    # invalid when the guard is enabled; overflow is sticky and raises before evidence is replaced.
    pre_apply_guard_terminal_archive_capacity: int | None = None
    # ActionBall-only table-assembly sensor latch.  ``apply_actions`` samples substeps 1..3 on
    # calls 2..4 and ``robot_hit_table`` samples substep 4 after the loop.  The term name binds the
    # sampler to the already-resolved SceneEntityCfg/body IDs and geometry parameters.
    table_contact_substep_guard: bool = False
    table_contact_guard_termination_term: str | None = None
    table_contact_guard_expected_decimation: int | None = None
