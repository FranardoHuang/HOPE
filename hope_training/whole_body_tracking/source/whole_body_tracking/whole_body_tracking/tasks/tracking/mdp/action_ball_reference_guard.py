"""Exact per-control-step accounting for ActionBall reference envelopes.

The three reference-relative predicates are useful diagnostics even when they
must not reset an ActionBall episode.  This module deliberately owns only
instrumentation state:

* it never changes a termination verdict, reward, observation, task, or RNG;
* it books one immutable raw snapshot after all three predicates have run;
* it initially classifies every raw union sample as ``reference_only``;
* the normal true-reset path later reclassifies the exact subset that also hit
  a physical/table/joint hard guard in the same control step.

Keeping the provisional classification and reset-time correction in one class
is what makes partial vector resets exact: an environment is corrected at most
once, while non-resetting rows remain valid ``reference_only`` observations.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch


REFERENCE_GUARD_PHASE_GATED = "phase_gated"
REFERENCE_GUARD_METRICS_ONLY = "metrics_only"
REFERENCE_GUARD_MODES = (
    REFERENCE_GUARD_PHASE_GATED,
    REFERENCE_GUARD_METRICS_ONLY,
)
REFERENCE_GUARD_REASONS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)
REFERENCE_GUARD_HARD_REASONS = (
    "base_fell_tilt",
    "base_too_low",
    "robot_hit_table",
    "joint_qdes_forbidden",
    "joint_actual_forbidden",
)
REFERENCE_GUARD_TIMING_PHASES = ("pre", "strike", "post")
REFERENCE_GUARD_CURRICULUM_PHASES = ("center", "noncenter")


def validate_reference_guard_mode(value: object) -> str:
    """Return one exact mode string; aliases would make receipts ambiguous."""

    if type(value) is not str or value not in REFERENCE_GUARD_MODES:
        raise ValueError(
            "reference_guard_mode must be exactly one of "
            f"{REFERENCE_GUARD_MODES}; got {value!r}"
        )
    return value


def reference_guard_counter_names() -> tuple[str, ...]:
    """Stable scalar-counter schema consumed with the exact behavior ledger."""

    names = [
        "reference_guard_sample_count",
        *(
            f"reference_guard_{reason}_count"
            for reason in REFERENCE_GUARD_REASONS
        ),
        "reference_guard_union_count",
        "reference_guard_reference_only_count",
        "reference_guard_reference_and_hard_count",
        "reference_guard_hard_without_snapshot_count",
    ]
    for phase in REFERENCE_GUARD_TIMING_PHASES:
        names.extend(
            (
                f"reference_guard_{phase}_sample_count",
                f"reference_guard_{phase}_union_count",
                f"reference_guard_{phase}_reference_only_count",
                f"reference_guard_{phase}_reference_and_hard_count",
                *(
                    f"reference_guard_{phase}_{reason}_count"
                    for reason in REFERENCE_GUARD_REASONS
                ),
            )
        )
    for phase in REFERENCE_GUARD_CURRICULUM_PHASES:
        names.extend(
            (
                f"reference_guard_{phase}_sample_count",
                f"reference_guard_{phase}_union_count",
                f"reference_guard_{phase}_reference_only_count",
                f"reference_guard_{phase}_reference_and_hard_count",
            )
        )
    if len(names) != len(set(names)):
        raise RuntimeError("reference-guard counter schema contains duplicate names")
    return tuple(names)


REFERENCE_GUARD_COUNTER_NAMES = reference_guard_counter_names()
REFERENCE_GUARD_COUNTER_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        list(REFERENCE_GUARD_COUNTER_NAMES),
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
REFERENCE_GUARD_CONTRACT_PAYLOAD = {
    "schema_version": 1,
    "kind": "whole_body_tracking.action_ball.reference_guard",
    "modes": list(REFERENCE_GUARD_MODES),
    "reference_reasons": list(REFERENCE_GUARD_REASONS),
    "hard_reasons": list(REFERENCE_GUARD_HARD_REASONS),
    "timing_phases": list(REFERENCE_GUARD_TIMING_PHASES),
    "curriculum_phases": list(REFERENCE_GUARD_CURRICULUM_PHASES),
    "counter_names": list(REFERENCE_GUARD_COUNTER_NAMES),
    "counter_schema_sha256": REFERENCE_GUARD_COUNTER_SCHEMA_SHA256,
}
REFERENCE_GUARD_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        REFERENCE_GUARD_CONTRACT_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class ActionBallReferenceGuardMetrics:
    """Stage and book one raw three-predicate snapshot per simulator step."""

    def __init__(self, *, num_envs: int, device: torch.device | str):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("reference-guard num_envs must be a positive integer")
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._reason_masks = {
            reason: torch.zeros(
                num_envs, dtype=torch.bool, device=self._device
            )
            for reason in REFERENCE_GUARD_REASONS
        }
        self._seen: set[str] = set()
        self._step_token: int | None = None
        self._booked = False
        self._union = torch.zeros(
            num_envs, dtype=torch.bool, device=self._device
        )
        self._timing_phase = torch.full(
            (num_envs,), -1, dtype=torch.int8, device=self._device
        )
        self._center = torch.zeros(
            num_envs, dtype=torch.bool, device=self._device
        )
        self._hard_adjusted = torch.zeros(
            num_envs, dtype=torch.bool, device=self._device
        )
        self._validated_ledger_id: int | None = None

    def has_complete_step(self, step_token: object) -> bool:
        """Return host metadata only; never synchronize a device tensor."""

        return (
            type(step_token) is int
            and self._step_token == step_token
            and self._booked
        )

    def _require_ledger(
        self, ledger: Mapping[str, torch.Tensor]
    ) -> None:
        if id(ledger) == self._validated_ledger_id:
            return
        missing = [
            name for name in REFERENCE_GUARD_COUNTER_NAMES if name not in ledger
        ]
        if missing:
            raise RuntimeError(
                "reference-guard exact ledger is missing counters "
                f"{missing}"
            )
        for name in REFERENCE_GUARD_COUNTER_NAMES:
            value = ledger[name]
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.long
                or value.ndim != 0
                or value.device != self._device
            ):
                raise RuntimeError(
                    f"reference-guard counter {name!r} must be a same-device "
                    "scalar int64 tensor"
                )
        self._validated_ledger_id = id(ledger)

    def _require_mask(self, value: object, *, name: str) -> torch.Tensor:
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.bool
            or tuple(value.shape) != (self._num_envs,)
            or value.device != self._device
        ):
            raise RuntimeError(
                f"reference-guard {name} must be a same-device bool tensor "
                f"shaped [{self._num_envs}]"
            )
        return value

    def _begin_step(self, step_token: int) -> None:
        if type(step_token) is not int or step_token < 0:
            raise RuntimeError(
                "reference-guard metrics require a non-negative integer "
                "env.common_step_counter"
            )
        if self._step_token == step_token:
            return
        if self._seen and not self._booked:
            raise RuntimeError(
                "reference-guard control step ended before all three raw "
                f"predicates ran: token={self._step_token}, seen={sorted(self._seen)}"
            )
        if self._step_token is not None and step_token <= self._step_token:
            raise RuntimeError(
                "reference-guard simulator-step token must increase strictly"
            )
        self._step_token = step_token
        self._seen.clear()
        self._booked = False
        self._union.zero_()
        self._timing_phase.fill_(-1)
        self._center.zero_()
        self._hard_adjusted.zero_()

    @staticmethod
    def _add_many(
        ledger: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> None:
        if set(masks) != set(REFERENCE_GUARD_COUNTER_NAMES):
            raise RuntimeError(
                "reference-guard step mask schema differs from counter schema"
            )
        ordered_masks = [
            masks[name].detach() for name in REFERENCE_GUARD_COUNTER_NAMES
        ]
        counts = torch.stack(ordered_masks, dim=0).sum(
            dim=1, dtype=torch.long
        )
        torch._foreach_add_(
            [ledger[name] for name in REFERENCE_GUARD_COUNTER_NAMES],
            list(counts.unbind()),
        )

    def record(
        self,
        *,
        reason: str,
        raw_mask: torch.Tensor,
        step_token: int,
        pre_strike: torch.Tensor,
        strike_window: torch.Tensor,
        center_phase: torch.Tensor,
        ledger: Mapping[str, torch.Tensor],
    ) -> None:
        """Record one reason and atomically book the step after reason three."""

        if reason not in REFERENCE_GUARD_REASONS:
            raise ValueError(f"unknown reference-guard reason {reason!r}")
        raw = self._require_mask(raw_mask, name=f"{reason} raw mask")
        pre = self._require_mask(pre_strike, name="pre_strike mask")
        strike = self._require_mask(
            strike_window, name="strike_window mask"
        )
        center = self._require_mask(
            center_phase, name="curriculum center mask"
        )
        self._require_ledger(ledger)
        self._begin_step(step_token)
        if reason in self._seen:
            raise RuntimeError(
                f"reference-guard reason {reason!r} ran twice in step {step_token}"
            )
        self._reason_masks[reason].copy_(raw.detach())
        self._seen.add(reason)
        if len(self._seen) != len(REFERENCE_GUARD_REASONS):
            return

        union = self._reason_masks[REFERENCE_GUARD_REASONS[0]].clone()
        for other in REFERENCE_GUARD_REASONS[1:]:
            union |= self._reason_masks[other]
        self._union.copy_(union)
        # Strike has precedence over pre/post because its symmetric window
        # intentionally spans both sides of t_hit.
        self._timing_phase.copy_(
            torch.where(
                strike,
                torch.ones_like(self._timing_phase),
                torch.where(
                    pre,
                    torch.zeros_like(self._timing_phase),
                    torch.full_like(self._timing_phase, 2),
                ),
            )
        )
        self._center.copy_(center)

        all_rows = torch.ones_like(union)
        no_rows = torch.zeros_like(union)
        counter_masks = {
            "reference_guard_sample_count": all_rows,
            "reference_guard_anchor_pos_count": self._reason_masks[
                "anchor_pos"
            ],
            "reference_guard_anchor_ori_count": self._reason_masks[
                "anchor_ori"
            ],
            "reference_guard_ee_body_pos_count": self._reason_masks[
                "ee_body_pos"
            ],
            "reference_guard_union_count": union,
            # A hard guard can only be known after the TerminationManager has
            # evaluated all terms.  Book provisionally, then correct the exact
            # terminal subset in ``adjust_hard_overlap``.
            "reference_guard_reference_only_count": union,
            "reference_guard_reference_and_hard_count": no_rows,
            "reference_guard_hard_without_snapshot_count": no_rows,
        }
        for index, phase in enumerate(REFERENCE_GUARD_TIMING_PHASES):
            phase_mask = self._timing_phase == index
            counter_masks[
                f"reference_guard_{phase}_sample_count"
            ] = phase_mask
            counter_masks[
                f"reference_guard_{phase}_union_count"
            ] = union & phase_mask
            counter_masks[
                f"reference_guard_{phase}_reference_only_count"
            ] = union & phase_mask
            counter_masks[
                f"reference_guard_{phase}_reference_and_hard_count"
            ] = no_rows
            for reason_name, reason_mask in self._reason_masks.items():
                counter_masks[
                    f"reference_guard_{phase}_{reason_name}_count"
                ] = reason_mask & phase_mask
        for is_center, phase in (
            (True, "center"),
            (False, "noncenter"),
        ):
            phase_mask = self._center if is_center else ~self._center
            counter_masks[
                f"reference_guard_{phase}_sample_count"
            ] = phase_mask
            counter_masks[
                f"reference_guard_{phase}_union_count"
            ] = union & phase_mask
            counter_masks[
                f"reference_guard_{phase}_reference_only_count"
            ] = union & phase_mask
            counter_masks[
                f"reference_guard_{phase}_reference_and_hard_count"
            ] = no_rows
        self._add_many(ledger, counter_masks)
        self._booked = True

    def adjust_hard_overlap(
        self,
        *,
        env_ids: torch.Tensor,
        hard_mask: torch.Tensor,
        step_token: int,
        ledger: Mapping[str, torch.Tensor],
    ) -> None:
        """Move same-step raw-reference+hard rows out of ``reference_only``."""

        self._require_ledger(ledger)
        if (
            not torch.is_tensor(env_ids)
            or env_ids.dtype != torch.long
            or env_ids.ndim != 1
            or env_ids.device != self._device
        ):
            raise RuntimeError(
                "reference-guard reset env_ids must be a same-device int64 vector"
            )
        if (
            not torch.is_tensor(hard_mask)
            or hard_mask.dtype != torch.bool
            or tuple(hard_mask.shape) != tuple(env_ids.shape)
            or hard_mask.device != self._device
        ):
            raise RuntimeError(
                "reference-guard hard mask must align exactly with reset env_ids"
            )
        if not self.has_complete_step(step_token):
            # Construction/global/manual resets can legally run without a
            # TerminationManager predicate pass.  Their all-false hard mask is
            # a no-op.  A real hard reset without its raw reference snapshot
            # is accumulated asynchronously and rejected once per PPO update
            # by ``validate_conservation``; never synchronize CUDA here.
            ledger["reference_guard_hard_without_snapshot_count"].add_(
                hard_mask.detach().sum(dtype=torch.long)
            )
            return
        selected_union = self._union[env_ids]
        overlap = selected_union & hard_mask
        # Reset plumbing can revisit a selected row in the same simulator
        # step.  Keep the correction idempotent on-device: a host-side
        # ``bool(tensor.any())`` here would serialize every terminal step with
        # CUDA and distort the very rollout timing this treatment measures.
        fresh = overlap & ~self._hard_adjusted[env_ids]
        amount = fresh.detach().sum(dtype=torch.long)
        ledger["reference_guard_reference_only_count"].sub_(amount)
        ledger["reference_guard_reference_and_hard_count"].add_(amount)
        selected_timing = self._timing_phase[env_ids]
        for index, phase in enumerate(REFERENCE_GUARD_TIMING_PHASES):
            phase_overlap = fresh & (selected_timing == index)
            phase_amount = phase_overlap.detach().sum(dtype=torch.long)
            ledger[f"reference_guard_{phase}_reference_only_count"].sub_(
                phase_amount
            )
            ledger[
                f"reference_guard_{phase}_reference_and_hard_count"
            ].add_(phase_amount)
        selected_center = self._center[env_ids]
        for is_center, phase in (
            (True, "center"),
            (False, "noncenter"),
        ):
            phase_overlap = fresh & (selected_center == is_center)
            phase_amount = phase_overlap.detach().sum(dtype=torch.long)
            ledger[f"reference_guard_{phase}_reference_only_count"].sub_(
                phase_amount
            )
            ledger[
                f"reference_guard_{phase}_reference_and_hard_count"
            ].add_(phase_amount)
        self._hard_adjusted[env_ids] |= overlap

    def validate_conservation(
        self, ledger: Mapping[str, torch.Tensor]
    ) -> None:
        """Fail before publication if any additive partition drifted."""

        self._require_ledger(ledger)
        terminal_reset = ledger.get("terminal_reset_count")
        if terminal_reset is not None and (
            not torch.is_tensor(terminal_reset)
            or terminal_reset.dtype != torch.long
            or terminal_reset.ndim != 0
            or terminal_reset.device != self._device
        ):
            raise RuntimeError(
                "reference-guard terminal_reset_count must be a same-device "
                "scalar int64 tensor"
            )
        snapshot_names = list(REFERENCE_GUARD_COUNTER_NAMES)
        if terminal_reset is not None:
            snapshot_names.append("terminal_reset_count")
        rows = torch.stack(
            [ledger[name] for name in snapshot_names]
        ).detach().cpu().tolist()
        values = dict(zip(snapshot_names, rows))
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise RuntimeError(
                "reference-guard counters must remain non-negative integers"
            )

        union = values["reference_guard_union_count"]
        reference_only = values["reference_guard_reference_only_count"]
        reference_and_hard = values[
            "reference_guard_reference_and_hard_count"
        ]
        if union != reference_only + reference_and_hard:
            raise RuntimeError(
                "reference-guard union does not partition into "
                "reference_only + reference_and_hard"
            )
        if values["reference_guard_hard_without_snapshot_count"] != 0:
            raise RuntimeError(
                "reference-guard observed a hard reset without a complete "
                "same-step raw snapshot"
            )
        if any(
            values[f"reference_guard_{reason}_count"] > union
            for reason in REFERENCE_GUARD_REASONS
        ):
            raise RuntimeError(
                "reference-guard reason count exceeds its raw union"
            )
        if terminal_reset is not None:
            if reference_and_hard > values["terminal_reset_count"]:
                raise RuntimeError(
                    "reference-and-hard count exceeds terminal resets"
                )

        for phases in (
            REFERENCE_GUARD_TIMING_PHASES,
            REFERENCE_GUARD_CURRICULUM_PHASES,
        ):
            for suffix, expected in (
                ("sample_count", values["reference_guard_sample_count"]),
                ("union_count", union),
                ("reference_only_count", reference_only),
                ("reference_and_hard_count", reference_and_hard),
            ):
                actual = sum(
                    values[f"reference_guard_{phase}_{suffix}"]
                    for phase in phases
                )
                if actual != expected:
                    raise RuntimeError(
                        "reference-guard phase partition drifted: "
                        f"phases={phases}, suffix={suffix}, "
                        f"actual={actual}, expected={expected}"
                    )
        for reason in REFERENCE_GUARD_REASONS:
            for phase in REFERENCE_GUARD_TIMING_PHASES:
                if (
                    values[f"reference_guard_{phase}_{reason}_count"]
                    > values[f"reference_guard_{phase}_union_count"]
                ):
                    raise RuntimeError(
                        "reference-guard phase reason count exceeds its "
                        f"phase union: phase={phase!r}, reason={reason!r}"
                    )
            timing_total = sum(
                values[f"reference_guard_{phase}_{reason}_count"]
                for phase in REFERENCE_GUARD_TIMING_PHASES
            )
            if timing_total != values[f"reference_guard_{reason}_count"]:
                raise RuntimeError(
                    "reference-guard timing/reason partition drifted: "
                    f"reason={reason!r}"
                )
