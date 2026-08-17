"""Lean diagnostic ActionBall runtime owner and PPO epoch drain boundary.

This module deliberately owns one external boundary only: the handoff from the
device-resident :class:`ActionEpochOwner` log to one PPO optimizer update.  It
does not recreate the former per-leaf receipt graph, and it cannot authorize a
formal run, checkpoint, export, deployment, or robot command.

The opaque boundary below is chronology, not evidence.  Safety and learning
facts remain in the epoch record and its packed deltas; an object identity or a
source digest is never accepted as proof that those facts are true.
"""

from __future__ import annotations

from dataclasses import fields
import importlib
import struct
import sys
import threading
from typing import Optional

import torch

try:
    from . import action_ball_full_mdp_epoch as epoch_v1
except ImportError:  # Focused source-file tests avoid importing Isaac Lab.
    import action_ball_full_mdp_epoch as epoch_v1

try:
    from . import action_ball_full_mdp_lean_checkpoint_txn as carry_txn
except ImportError:  # Focused source-file tests avoid importing Isaac Lab.
    import action_ball_full_mdp_lean_checkpoint_txn as carry_txn

try:
    import action_ball_continuous_runtime_transaction_device as device_r05
except ImportError:  # Package-mode tests may expose the sibling through mdp.
    from . import action_ball_continuous_runtime_transaction_device as device_r05


try:
    from . import action_ball_full_mdp_drain_summary as drain_v2
except ImportError:  # Focused source-file tests avoid importing Isaac Lab.
    import action_ball_full_mdp_drain_summary as drain_v2


DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_DEPENDENCY_KIND = "action_ball_epoch_runtime_dependencies_v1"

DRAIN_SCHEMA_VERSION = drain_v2.DRAIN_SCHEMA_VERSION
DRAIN_SUMMARY_KIND = drain_v2.DRAIN_SUMMARY_KIND
ActionEpochPpoBoundarySummary = drain_v2.ActionEpochPpoBoundarySummary
EpochDrainFrontier = drain_v2.EpochDrainFrontier


class ActionBallFullMdpLeanRuntimeError(RuntimeError):
    """The diagnostic owner boundary is stale, foreign, or poisoned."""


class _OptimizerBoundary:
    """Owner-issued identity with no caller-readable authority fields."""

    __slots__ = ()


class _SelectedResetPackedPreflight:
    """Opaque identity minted only after the sole packed D2H is clean."""

    __slots__ = ()

    def __new__(cls):
        raise TypeError("selected-reset preflight is owner-minted")


def _plain_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ActionBallFullMdpLeanRuntimeError(
            f"{label} must be a non-negative exact int"
        )
    return value


def _continuation_tensor_items(value: drain_v2.ActionEpochDrainContinuation):
    items = [("continuation.occupied", value.occupied)]
    items.extend(
        ("continuation.key." + field.name, getattr(value.key, field.name))
        for field in fields(epoch_v1.ActionEpochShotKey)
    )
    items.extend(
        ("continuation." + field.name, getattr(value, field.name))
        for field in fields(drain_v2.ActionEpochDrainContinuation)
        if field.name not in ("occupied", "key")
    )
    return tuple(items)


class ActionBallFullMdpLeanRuntimeOwner:
    """Single owner for an epoch journal and its irreversible optimizer edge."""

    def __init__(
        self,
        *,
        env: object,
        runtime_lease: object,
        epoch_owner: epoch_v1.ActionEpochOwner,
        reward_graph: object,
        r05_runtime: object,
        motion: object,
        racket: object,
        physical_ball: object,
        r06_landing_outcome: object,
        r03_strike_fact: object,
        r07_recovery: object,
    ) -> None:
        if env is None or runtime_lease is None:
            raise ActionBallFullMdpLeanRuntimeError(
                "env and runtime_lease must be exact non-null identities"
            )
        if type(epoch_owner) is not epoch_v1.ActionEpochOwner:
            raise ActionBallFullMdpLeanRuntimeError(
                "epoch_owner must be the exact ActionEpochOwner"
            )
        self._env = env
        self._runtime_lease = runtime_lease
        self._epoch = epoch_owner
        # These are construction joins only.  A later coordinator may consume
        # the identities directly; this owner never turns their mere presence
        # into a readiness or safety claim.
        self._r05_runtime = r05_runtime
        self._motion = motion
        self._racket = racket
        self._physical_ball = physical_ball
        self._r06_landing_outcome = r06_landing_outcome
        self._r03_strike_fact = r03_strike_fact
        self._r07_recovery = r07_recovery
        package = __package__
        name = (
            package + ".action_ball_full_mdp_lean_rewards"
            if package
            else "action_ball_full_mdp_lean_rewards"
        )
        rewards = importlib.import_module(name)
        if (
            type(reward_graph) is not rewards.LeanActionEpochRewardGraph
            or reward_graph.epoch_owner is not epoch_owner
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "reward_graph must be the exact graph for this ActionEpoch"
            )
        self._reward_graph_identity = reward_graph

        self._ppo_drain = self
        self._next_update_index = 0
        self._operation_sequence = 0
        self._drain_sequence = 0
        self._last_completed_environment_steps = 0
        self._acked_commit_end = 0

        self._active_boundary: Optional[_OptimizerBoundary] = None
        self._active_update_index: Optional[int] = None
        self._active_completed_environment_steps: Optional[int] = None
        self._active_start: Optional[int] = None
        self._active_end: Optional[int] = None
        self._active_decoded_drain: Optional[drain_v2.DecodedEpochDrain] = None
        self._active_post_update_summary: Optional[
            ActionEpochPpoBoundarySummary
        ] = None
        self._materialize_started = False
        self._drain_materialized = False
        self._optimizer_returned = False

        self._acked_continuation = drain_v2.ActionEpochDrainContinuation.empty(
            num_envs=self._epoch.num_envs,
            shot_slot_capacity=self._epoch.shot_slot_capacity,
        )
        self._poisoned = False
        self._poison_reason: Optional[str] = None
        self._operation_active = False
        self._last_before_policy_control_step = 0
        self._pending_after_command_control_step: Optional[int] = None
        self._genesis_after_command_completed = False
        self._last_prephysics_position: Optional[tuple[int, int, int, int]] = None
        self._last_postphysics_position: Optional[tuple[int, int, int, int]] = None
        self._last_reward_cycle_count = 0
        self._last_reward_control_step = 0
        self._durable_ack_update_index = -1
        self._durable_ack_segment_id = None
        self._durable_ack_rank = None
        self._durable_pending_byte_start = 0
        self._durable_pending_byte_end = 0
        self._durable_ack_byte_start = 0
        self._durable_ack_byte_end = 0
        self._business_generation = 0
        self._durable_ack_business_generation = -1
        self._pending_durable_ack_summary = None
        self._selected_reset_event: Optional[object] = None
        self._selected_reset_projection: Optional[object] = None
        self._selected_reset_prepared: Optional[object] = None
        self._selected_reset_child_commits: Optional[tuple[object, ...]] = None
        self._selected_reset_child_commits_started = False
        self._selected_reset_r05_receipt: Optional[object] = None
        self._selected_reset_completions: Optional[dict[str, object]] = None
        self._selected_reset_env_record: Optional[object] = None
        self._selected_reset_live_ledger_identity: Optional[object] = None
        self._selected_reset_epoch_prepared: Optional[object] = None
        self._selected_reset_packed_preflight: Optional[object] = None
        self._selected_reset_leaf_completions_consumed = False
        self._lock = threading.RLock()
        self._lean_carry_coordinator = None
        coordinator = carry_txn._LeanCarryCoordinator(
            root=self,
            mandatory_roles=("root", "epoch", "milestone", "reward", "d05"),
        )
        coordinator._register("root", self)
        if self._epoch._action_uids_by_slot.numel() > 0:
            coordinator._register("epoch", self._epoch)
            coordinator._register("milestone", self._epoch.milestone)
            coordinator._register("reward", self._reward_graph_identity)
        d05_schema = getattr(type(self._r05_runtime), "_lean_carry_schema", None)
        if callable(d05_schema):
            coordinator._register("d05", self._r05_runtime)

    @classmethod
    def create_from_env(
        cls,
        env: object,
        lease: object,
    ) -> "ActionBallFullMdpLeanRuntimeOwner":
        """Return the factory-installed identity; never construct a duplicate."""

        if env is None or lease is None:
            raise ActionBallFullMdpLeanRuntimeError(
                "runtime lookup requires an exact env and non-null lease"
            )
        if getattr(env, "action_ball_full_mdp_runtime_lease", None) is not lease:
            raise ActionBallFullMdpLeanRuntimeError(
                "runtime lookup lease does not belong to this environment"
            )
        getter = getattr(env, "action_ball_full_mdp_lean_runtime_owner", None)
        expected_getter = getattr(type(env), "action_ball_full_mdp_lean_runtime_owner", None)
        if (
            not callable(getter)
            or getattr(getter, "__self__", None) is not env
            or expected_getter is None
            or getattr(getter, "__func__", None) is not expected_getter
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "environment omitted its exact bound lean-owner getter"
            )
        owner = getter(lease)
        if (
            type(owner) is not cls
            or owner.full_mdp_runtime_env is not env
            or owner.full_mdp_runtime_lease is not lease
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "environment returned a foreign or duplicate lean owner"
            )
        return owner

    @property
    def full_mdp_runtime_env(self) -> object:
        return self._env

    @property
    def full_mdp_runtime_lease(self) -> object:
        return self._runtime_lease

    @property
    def epoch_owner(self) -> epoch_v1.ActionEpochOwner:
        return self._epoch

    @property
    def component_identities(self) -> tuple[tuple[str, object], ...]:
        """Construction identities only; never a readiness verdict."""

        return (
            ("r05_runtime", self._r05_runtime),
            ("motion", self._motion),
            ("racket", self._racket),
            ("physical_ball", self._physical_ball),
            ("r06_landing_outcome", self._r06_landing_outcome),
            ("r03_strike_fact", self._r03_strike_fact),
            ("r07_recovery", self._r07_recovery),
        )

    @property
    def diagnostic_dependency_kind(self) -> str:
        return DIAGNOSTIC_DEPENDENCY_KIND

    @property
    def diagnostic_unauthorized(self) -> bool:
        return True

    @property
    def launch_authorized(self) -> bool:
        return False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def poison_reason(self) -> Optional[str]:
        return self._poison_reason

    def _lean_carry_schema(self) -> carry_txn._LeanCarrySchema:
        continuation = _continuation_tensor_items(self._acked_continuation)
        return carry_txn._LeanCarrySchema(
            "root",
            (
                ("next_update_index", int),
                ("operation_sequence", int),
                ("drain_sequence", int),
                ("last_completed_environment_steps", int),
                ("acked_commit_end", int),
                ("last_reward_cycle_count", int),
                ("last_reward_control_step", int),
                ("last_before_policy_control_step", int),
                ("genesis_after_command_completed", bool),
                ("last_prephysics_position", tuple),
                ("last_postphysics_position", tuple),
                ("durable_ack_update_index", int),
                ("durable_ack_segment_id", str),
                ("durable_ack_rank", int),
                ("durable_pending_byte_start", int),
                ("durable_pending_byte_end", int),
                ("durable_ack_byte_start", int),
                ("durable_ack_byte_end", int),
                ("business_generation", int),
                ("durable_ack_business_generation", int),
            ),
            tuple(
                carry_txn._LeanCarryTensorSpec(
                    name, tuple(value.shape), value.dtype, "copy", "host"
                )
                for name, value in continuation
            ),
        )

    def _lean_carry_construction_views(self):
        return tuple(value for _name, value in _continuation_tensor_items(
            self._acked_continuation
        ))

    def _lean_carry_capture(self, lease: object) -> carry_txn._LeanCarryCapture:
        if (
            getattr(lease, "coordinator", None) is not self._lean_carry_coordinator
            or getattr(lease, "kind", None) != "capture"
        ):
            raise ActionBallFullMdpLeanRuntimeError("root carry lease differs")
        self.require_healthy()
        if (
            self._operation_active or self._active_boundary is not None
            or self._durable_ack_update_index != self._next_update_index - 1
            or self._durable_ack_update_index < 0
            or self._business_generation != self._durable_ack_business_generation
            or self._pending_durable_ack_summary is not None
            or self._pending_after_command_control_step is not None
            or self._selected_reset_event is not None
            or self._selected_reset_projection is not None
            or self._selected_reset_prepared is not None
            or self._selected_reset_child_commits_started
            or self._selected_reset_r05_receipt is not None
            or self._selected_reset_env_record is not None
            or self._selected_reset_epoch_prepared is not None
            or self._selected_reset_packed_preflight is not None
            or not self._genesis_after_command_completed
            or type(self._last_prephysics_position) is not tuple
            or type(self._last_postphysics_position) is not tuple
            or self._last_prephysics_position[:3]
            != self._last_postphysics_position[:3]
            or self._last_postphysics_position[3]
            != self._last_prephysics_position[3] + 1
            or self._last_prephysics_position[0] != self._last_reward_control_step
            or self._last_prephysics_position[1]
            != self._last_prephysics_position[2] - 1
            or type(self._durable_ack_segment_id) is not str
            or type(self._durable_ack_rank) is not int
            or self._durable_pending_byte_start < 0
            or self._durable_pending_byte_end <= self._durable_pending_byte_start
            or self._durable_ack_byte_start != self._durable_pending_byte_end
            or self._durable_ack_byte_end <= self._durable_ack_byte_start
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "root carry source is not one durable ACK boundary"
            )
        return carry_txn._LeanCarryCapture((
            self._next_update_index,
            self._operation_sequence,
            self._drain_sequence,
            self._last_completed_environment_steps,
            self._acked_commit_end,
            self._last_reward_cycle_count,
            self._last_reward_control_step,
            self._last_before_policy_control_step,
            self._genesis_after_command_completed,
            self._last_prephysics_position,
            self._last_postphysics_position,
            self._durable_ack_update_index,
            self._durable_ack_segment_id,
            self._durable_ack_rank,
            self._durable_pending_byte_start,
            self._durable_pending_byte_end,
            self._durable_ack_byte_start,
            self._durable_ack_byte_end,
            self._business_generation,
            self._durable_ack_business_generation,
        ), self._lean_carry_construction_views())

    def _lean_carry_stage(self, lease, scalars, host_tensors):
        if (
            getattr(lease, "coordinator", None) is not self._lean_carry_coordinator
            or getattr(lease, "kind", None) != "prepare"
            or self._next_update_index != 0
            or self._operation_sequence != 0
            or self._drain_sequence != 0
            or self._last_completed_environment_steps != 0
            or self._acked_commit_end != 0
            or self._last_reward_cycle_count != 0
            or self._last_reward_control_step != 0
            or self._last_before_policy_control_step != 0
            or self._pending_after_command_control_step is not None
            or self._durable_ack_update_index != -1
            or self._durable_ack_segment_id is not None
            or self._durable_ack_rank is not None
            or self._durable_pending_byte_start != 0
            or self._durable_pending_byte_end != 0
            or self._durable_ack_byte_start != 0
            or self._durable_ack_byte_end != 0
            or self._durable_ack_business_generation != -1
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "root carry target is not dormant genesis"
            )
        targets = self._lean_carry_construction_views()
        empty = drain_v2.ActionEpochDrainContinuation.empty(
            num_envs=self._epoch.num_envs,
            shot_slot_capacity=self._epoch.shot_slot_capacity,
        )
        expected = tuple(value for _name, value in _continuation_tensor_items(empty))
        if any(not torch.equal(value, baseline) for value, baseline in zip(targets, expected)):
            raise ActionBallFullMdpLeanRuntimeError(
                "root carry continuation target is not dormant"
            )
        return carry_txn._LeanCarryStage(
            scalars,
            tuple(value.detach().clone().contiguous() for value in host_tensors),
            targets,
        )

    def _lean_carry_target_views(self, lease, stage):
        if lease is not self._lean_carry_coordinator._active_lease:
            raise ActionBallFullMdpLeanRuntimeError("root carry target lease differs")
        return self._lean_carry_construction_views()

    def _lean_carry_apply_scalars(self, lease, stage) -> None:
        if not stage.commit_started or lease is not self._lean_carry_coordinator._active_lease:
            raise ActionBallFullMdpLeanRuntimeError("root carry commit was not armed")
        (
            self._next_update_index,
            self._operation_sequence,
            self._drain_sequence,
            self._last_completed_environment_steps,
            self._acked_commit_end,
            self._last_reward_cycle_count,
            self._last_reward_control_step,
            self._last_before_policy_control_step,
            self._genesis_after_command_completed,
            self._last_prephysics_position,
            self._last_postphysics_position,
            self._durable_ack_update_index,
            self._durable_ack_segment_id,
            self._durable_ack_rank,
            self._durable_pending_byte_start,
            self._durable_pending_byte_end,
            self._durable_ack_byte_start,
            self._durable_ack_byte_end,
            self._business_generation,
            self._durable_ack_business_generation,
        ) = stage.scalars

    def _lean_carry_cross_validate(
        self, lease, source_scalars, host_tensors, staged_scalars
    ) -> None:
        if (
            lease is not self._lean_carry_coordinator._active_lease
            or source_scalars != staged_scalars
            or len(source_scalars) != 5
            or len(host_tensors) != 5
        ):
            raise ActionBallFullMdpLeanRuntimeError("cross-owner carry phase differs")
        root = source_scalars[0]
        epoch = source_scalars[1]
        reward = source_scalars[3]
        schemas = self._lean_carry_coordinator._schemas
        try:
            epoch_reset, d05_reset = tuple(
                host_tensors[index][tuple(
                    field.name for field in schemas[role].tensor_fields
                ).index("reset_generation")]
                for index, role in ((1, "epoch"), (4, "d05"))
            )
        except (KeyError, ValueError, IndexError) as exc:
            raise ActionBallFullMdpLeanRuntimeError(
                "cross-owner reset-generation ABI differs"
            ) from exc
        if (
            root[4] != epoch[2]
            or epoch[1] != epoch[2]
            or root[5] != reward[0]
            or reward[0] != reward[1]
            or tuple(float(value) for value in host_tensors[3][0].tolist())
            != reward[3]
            or not torch.equal(epoch_reset, d05_reset)
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "cross-owner ACK/reward chronology differs"
            )

    def _capture_private_carry_for_save(self) -> object:
        """Exercise the real private preflight; never authorize persistence."""

        try:
            image = self._lean_carry_coordinator._capture()
        except carry_txn._LeanCarryError as exc:
            raise ActionBallFullMdpLeanRuntimeError(
                "single_action_lean forbids checkpoint/save before write: "
                + str(exc)
            ) from exc
        self._lean_carry_coordinator._discard(image)
        return None

    def _record_durable_epoch_ack_span(
        self, summary: object, *, update_index: int, segment_id: str, rank: int,
        pending_byte_start: int, pending_byte_end: int,
        ack_byte_start: int, ack_byte_end: int,
    ) -> None:
        """Latch one already-fsynced EPOCH_ACK span after destructive ACK."""

        with self._lock:
            carry_txn._require_leaf_mutable(self)
            self.require_healthy()
            if (
                summary is not self._pending_durable_ack_summary
                or type(summary) is not ActionEpochPpoBoundarySummary
                or type(update_index) is not int
                or update_index != self._next_update_index - 1
                or update_index != self._durable_ack_update_index + 1
                or summary.frontier.update_index != update_index
                or summary.frontier.completed_environment_steps
                != self._last_completed_environment_steps
                or summary.frontier.operation_sequence != self._operation_sequence
                or summary.frontier.drain_sequence != self._drain_sequence
                or summary.frontier.end_commit != self._acked_commit_end
                or type(segment_id) is not str or not segment_id
                or (self._durable_ack_segment_id is not None
                    and segment_id != self._durable_ack_segment_id)
                or type(rank) is not int or rank < 0
                or (self._durable_ack_rank is not None and rank != self._durable_ack_rank)
                or type(pending_byte_start) is not int
                or type(pending_byte_end) is not int
                or type(ack_byte_start) is not int
                or type(ack_byte_end) is not int
                or pending_byte_start != self._durable_ack_byte_end
                or pending_byte_end <= pending_byte_start
                or ack_byte_start != pending_byte_end
                or ack_byte_end <= ack_byte_start
            ):
                self._poison_locked("durable EPOCH_ACK span chronology differs")
                raise ActionBallFullMdpLeanRuntimeError(
                    "durable EPOCH_ACK span chronology differs"
                )
            self._durable_ack_update_index = update_index
            self._durable_ack_segment_id = segment_id
            self._durable_ack_rank = rank
            self._durable_pending_byte_start = pending_byte_start
            self._durable_pending_byte_end = pending_byte_end
            self._durable_ack_byte_start = ack_byte_start
            self._durable_ack_byte_end = ack_byte_end
            self._durable_ack_business_generation = self._business_generation
            self._pending_durable_ack_summary = None

    def _enter(self, operation: str) -> None:
        carry_txn._require_leaf_mutable(self)
        if self._operation_active:
            self._poisoned = True
            if self._poison_reason is None:
                self._poison_reason = "lean runtime boundary re-entered: " + operation
            raise ActionBallFullMdpLeanRuntimeError(
                "lean runtime boundary re-entry poisoned the owner"
            )
        self._operation_active = True
        self._business_generation += 1

    def _leave(self) -> None:
        self._operation_active = False

    def _poison_locked(self, reason: object) -> None:
        clean = reason.strip() if type(reason) is str else ""
        if not clean:
            clean = "lean runtime optimizer boundary failed"
        if not self._poisoned:
            self._poison_reason = clean
        self._poisoned = True

    def require_healthy(self) -> None:
        with self._lock:
            carry_txn._require_process_healthy()
            if self._poisoned:
                raise ActionBallFullMdpLeanRuntimeError(
                    "lean runtime owner is poisoned; retry is forbidden: "
                    + str(self._poison_reason)
                )
            if self._env is None or self._runtime_lease is None:
                self._poison_locked("runtime env/lease identity was lost")
                raise ActionBallFullMdpLeanRuntimeError(
                    "runtime env/lease identity was lost"
                )
            if type(self._epoch) is not epoch_v1.ActionEpochOwner:
                self._poison_locked("epoch owner identity/type changed")
                raise ActionBallFullMdpLeanRuntimeError(
                    "epoch owner identity/type changed"
                )
            if self._epoch.poisoned:
                self._poison_locked("device epoch owner decoded a terminal fault")
                raise ActionBallFullMdpLeanRuntimeError(
                    "device epoch owner decoded a terminal fault"
                )

    @staticmethod
    def _bound_plain_method(owner: object, name: str):
        function = vars(type(owner)).get(name)
        bound = getattr(owner, name, None)
        if (
            not callable(function)
            or not callable(bound)
            or getattr(bound, "__self__", None) is not owner
            or getattr(bound, "__func__", None) is not function
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "construction-bound component lacks exact " + name
            )
        return bound

    def _reward_graph(self) -> object:
        # Construction already joins the exact graph type and Epoch identity.
        # Runtime accounting reads that retained identity directly; it does not
        # mint a second environment authority at every control/update boundary.
        return self._reward_graph_identity

    def prepare_pre_optimizer_ppo_boundary(
        self,
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Freeze, materialize, and decode the packed suffix exactly once."""

        with self._lock:
            self._enter("prepare")
            try:
                self.require_healthy()
                update = _plain_nonnegative_int(update_index, label="update_index")
                completed = _plain_nonnegative_int(
                    completed_environment_steps,
                    label="completed_environment_steps",
                )
                if update != self._next_update_index:
                    self._poison_locked("PPO update index is stale or skipped")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "PPO update index is stale or skipped"
                    )
                if completed <= self._last_completed_environment_steps:
                    self._poison_locked(
                        "completed environment steps did not advance"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "completed environment steps did not advance"
                    )
                if self._active_boundary is not None:
                    self._poison_locked("one optimizer boundary is already active")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "one optimizer boundary is already active"
                    )

                graph = self._reward_graph()
                if (
                    graph.poisoned
                    or graph.cycle_open
                    or graph.actual_closed_cycle_count
                    != graph.completed_cycle_count
                ):
                    self._poison_locked(
                        "Reward graph has unfinished actual-buffer accounting"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "Reward graph has unfinished actual-buffer accounting"
                    )

                start, end = self._epoch.prepare_drain()
                if start != self._acked_commit_end:
                    self._poison_locked(
                        "epoch drain start differs from the ACKed frontier"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "epoch drain start differs from the ACKed frontier"
                    )
                boundary = _OptimizerBoundary()
                self._active_boundary = boundary
                self._active_update_index = update
                self._active_completed_environment_steps = completed
                self._active_start = start
                self._active_end = end
                self._materialize_started = True
                try:
                    materialized = self._epoch.materialize_drain(
                        start=start, end=end
                    )
                except BaseException as exc:
                    self._poison_locked(
                        "epoch drain transfer failed after materialization began: "
                        + type(exc).__name__
                    )
                    raise
                if (
                    type(materialized)
                    is not epoch_v1.ActionEpochMaterializedDrain
                    or type(materialized.entries) is not tuple
                    or type(materialized.overflow) is not torch.Tensor
                    or materialized.overflow.dtype != torch.bool
                    or tuple(materialized.overflow.shape)
                    != (self._epoch.num_envs,)
                    or materialized.overflow.device.type != "cpu"
                    or not materialized.overflow.is_contiguous()
                ):
                    self._poison_locked("epoch drain overflow image ABI differs")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "epoch drain overflow image ABI differs"
                    )
                overflowed = bool(materialized.overflow.any())
                self._drain_materialized = True
                host_entries = materialized.entries
                if (
                    len(host_entries) != end - start
                    or any(type(entry) is not epoch_v1.CommitEntry for entry in host_entries)
                    or any(
                        entry.sequence != start + ordinal
                        for ordinal, entry in enumerate(host_entries)
                    )
                ):
                    self._poison_locked(
                        "packed epoch host suffix differs from prepared frontier"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "packed epoch host suffix differs from prepared frontier"
                    )
                try:
                    decoded = drain_v2.decode_epoch_drain_suffix(
                        host_entries,
                        start_commit=start,
                        end_commit=end,
                        previous=self._acked_continuation,
                        milestone_i64=materialized.milestone_i64,
                        milestone_f64=materialized.milestone_f64,
                    )
                except BaseException as exc:
                    self._poison_locked(
                        "epoch row-wise drain decode failed before optimizer: "
                        + type(exc).__name__
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "epoch row-wise drain decode failed before optimizer: "
                        + str(exc)
                    ) from exc
                if type(decoded) is not drain_v2.DecodedEpochDrain:
                    self._poison_locked("epoch drain decoder result type differs")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "epoch drain decoder result type differs"
                    )
                self._active_decoded_drain = decoded
                if overflowed:
                    self._poison_locked("epoch drain decoded a terminal overflow")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "epoch drain decoded a terminal overflow"
                    )
                return boundary
            finally:
                self._leave()

    def _require_active_locked(
        self,
        boundary: object,
        update_index: object,
    ) -> int:
        update = _plain_nonnegative_int(update_index, label="update_index")
        if (
            type(boundary) is not _OptimizerBoundary
            or boundary is not self._active_boundary
            or update != self._active_update_index
        ):
            self._poison_locked("optimizer boundary identity/update is foreign or stale")
            raise ActionBallFullMdpLeanRuntimeError(
                "optimizer boundary identity/update is foreign or stale"
            )
        return update

    def mark_optimizer_returned(
        self,
        boundary: object,
        *,
        update_index: int,
    ) -> None:
        """Mark the irreversible return from ``optimizer.step()`` once."""

        with self._lock:
            self._enter("mark optimizer returned")
            try:
                self.require_healthy()
                self._require_active_locked(boundary, update_index)
                if not self._drain_materialized or self._optimizer_returned:
                    self._poison_locked("optimizer return is reordered or replayed")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "optimizer return is reordered or replayed"
                    )
                self._optimizer_returned = True
            finally:
                self._leave()

    def acknowledge_post_update(
        self,
        boundary: object,
        summary: object,
        *,
        update_index: int,
    ) -> ActionEpochPpoBoundarySummary:
        """Destructively ACK only the exact owner-prepared durable summary."""

        with self._lock:
            self._enter("acknowledge post update")
            try:
                self.require_healthy()
                update = self._require_active_locked(boundary, update_index)
                if not self._optimizer_returned:
                    self._poison_locked("post-update ACK preceded optimizer return")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-update ACK preceded optimizer return"
                    )
                if (
                    type(summary) is not ActionEpochPpoBoundarySummary
                    or summary is not self._active_post_update_summary
                ):
                    self._poison_locked(
                        "post-update ACK summary is foreign, stale, or unprepared"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-update ACK summary is foreign, stale, or unprepared"
                    )
                start = self._active_start
                end = self._active_end
                completed = self._active_completed_environment_steps
                decoded = self._active_decoded_drain
                if (
                    type(start) is not int
                    or type(end) is not int
                    or type(completed) is not int
                    or type(decoded) is not drain_v2.DecodedEpochDrain
                ):
                    self._poison_locked("active epoch drain state is incomplete")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "active epoch drain state is incomplete"
                    )
                frontier = summary.frontier
                operation_sequence = self._operation_sequence + 1
                drain_sequence = self._drain_sequence + 1
                if (
                    type(frontier) is not EpochDrainFrontier
                    or frontier.update_index != update
                    or frontier.completed_environment_steps != completed
                    or frontier.operation_sequence != operation_sequence
                    or frontier.drain_sequence != drain_sequence
                    or frontier.start_commit != start
                    or frontier.end_commit != end
                ):
                    self._poison_locked(
                        "prepared post-update summary chronology changed"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "prepared post-update summary chronology changed"
                    )
                try:
                    self._epoch.acknowledge_drain(start=start, end=end)
                except BaseException as exc:
                    self._poison_locked(
                        "epoch drain ACK failed after optimizer return: "
                        + type(exc).__name__
                    )
                    raise

                self._acked_continuation = decoded.next_continuation
                self._acked_commit_end = end
                self._operation_sequence = operation_sequence
                self._drain_sequence = drain_sequence
                self._next_update_index = update + 1
                self._last_completed_environment_steps = completed
                self._pending_durable_ack_summary = summary
                self._clear_active_locked()
                return summary
            finally:
                self._leave()

    def prepare_post_update_summary(
        self,
        boundary: object,
        *,
        update_index: int,
    ) -> ActionEpochPpoBoundarySummary:
        """Freeze the exact typed summary without ACKing or clearing Epoch."""

        with self._lock:
            self._enter("prepare post-update summary")
            try:
                self.require_healthy()
                update = self._require_active_locked(boundary, update_index)
                if not self._optimizer_returned:
                    self._poison_locked(
                        "post-update summary preceded optimizer return"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-update summary preceded optimizer return"
                    )
                if self._active_post_update_summary is not None:
                    self._poison_locked(
                        "post-update summary preparation was replayed"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-update summary preparation was replayed"
                    )
                start = self._active_start
                end = self._active_end
                completed = self._active_completed_environment_steps
                decoded = self._active_decoded_drain
                if (
                    type(start) is not int
                    or type(end) is not int
                    or type(completed) is not int
                    or type(decoded) is not drain_v2.DecodedEpochDrain
                ):
                    self._poison_locked("active epoch drain state is incomplete")
                    raise ActionBallFullMdpLeanRuntimeError(
                        "active epoch drain state is incomplete"
                    )
                frontier = EpochDrainFrontier(
                    schema_version=DRAIN_SCHEMA_VERSION,
                    kind=DRAIN_SUMMARY_KIND,
                    num_envs=self._epoch.num_envs,
                    shot_slot_capacity=self._epoch.shot_slot_capacity,
                    device_type=self._epoch.device.type,
                    device_index=self._epoch.device.index,
                    update_index=update,
                    next_update_index=update + 1,
                    completed_environment_steps=completed,
                    operation_sequence=self._operation_sequence + 1,
                    drain_sequence=self._drain_sequence + 1,
                    start_commit=start,
                    end_commit=end,
                )
                summary = decoded.with_frontier(frontier)
                if type(summary) is not ActionEpochPpoBoundarySummary:
                    self._poison_locked(
                        "post-update summary constructor result type differs"
                    )
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-update summary constructor result type differs"
                    )
                self._active_post_update_summary = summary
                return summary
            finally:
                self._leave()

    def _clear_active_locked(self) -> None:
        self._active_boundary = None
        self._active_update_index = None
        self._active_completed_environment_steps = None
        self._active_start = None
        self._active_end = None
        self._active_decoded_drain = None
        self._active_post_update_summary = None
        self._materialize_started = False
        self._drain_materialized = False
        self._optimizer_returned = False

    def poison_optimizer_boundary(
        self,
        boundary_or_none: object,
        *,
        update_index: int,
        reason: str,
    ) -> None:
        """Poison sticky; abort only a proven pre-materialization snapshot."""

        with self._lock:
            carry_txn._require_leaf_mutable(self)
            if self._poisoned:
                return
            update_ok = (
                type(update_index) is int
                and update_index >= 0
                and (
                    self._active_update_index is None
                    or update_index == self._active_update_index
                )
            )
            boundary_ok = (
                self._active_boundary is None
                and boundary_or_none is None
            ) or (
                type(boundary_or_none) is _OptimizerBoundary
                and boundary_or_none is self._active_boundary
            )
            clean = reason if type(reason) is str and reason.strip() else (
                "optimizer failed after epoch drain"
            )
            if not update_ok or not boundary_ok:
                clean = "foreign optimizer poison boundary/update"

            if (
                self._active_boundary is not None
                and boundary_ok
                and update_ok
                and not self._materialize_started
                and type(self._active_start) is int
                and type(self._active_end) is int
            ):
                try:
                    self._epoch.abort_drain(
                        start=self._active_start,
                        end=self._active_end,
                    )
                except BaseException as exc:
                    clean += "; pre-materialize abort failed: " + type(exc).__name__
                else:
                    self._clear_active_locked()
            self._poison_locked(clean)

    # An unresolved callpoint always HOLDs before its first producer mutation.
    # This is intentionally narrower than fabricating a portable receipt or a
    # caller verdict around half of the real causal path.
    @staticmethod
    def _runtime_hold(callpoint: str) -> None:
        raise ActionBallFullMdpLeanRuntimeError(
            callpoint + " is not integrated in the lean diagnostic runtime"
        )

    def before_policy_step(self, control_step: int, action: object) -> None:
        with self._lock:
            self._enter("before policy step")
            try:
                self.require_healthy()
                step = _plain_nonnegative_int(control_step, label="control_step")
                if step < 1 or step != self._last_before_policy_control_step + 1:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "before-policy control step was skipped, duplicated, or replayed"
                    )
                if self._pending_after_command_control_step is not None:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "previous command-compute settlement is still pending"
                    )
                if (
                    type(action) is not torch.Tensor
                    or action.ndim != 2
                    or action.shape[0] != self._epoch.num_envs
                    or action.device != self._epoch.device
                    or not action.is_floating_point()
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "policy action must be one floating device tensor [N,A]"
                    )
                if self._active_boundary is not None:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "policy step cannot overlap an optimizer drain"
                    )
                graph = self._reward_graph()
                if graph.poisoned or graph.cycle_open:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "policy step cannot cross poisoned or open Reward debt"
                    )
                self._last_before_policy_control_step = step
                self._pending_after_command_control_step = step
            finally:
                self._leave()

    def before_physics_substep(self, stamp: object) -> None:
        """Offer one exact pre-write opportunity to the Physical launcher."""

        stamp_type = type(stamp)
        stamp_module = sys.modules.get(stamp_type.__module__)
        if (
            stamp_type.__module__
            != "whole_body_tracking.tasks.tracking.full_mdp_env"
            or stamp_type.__qualname__ != "FullMdpPrePhysicsSubstepStamp"
            or stamp_module is None
            or vars(stamp_module).get("FullMdpPrePhysicsSubstepStamp")
            is not stamp_type
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "pre-physics stamp must be the exact environment type"
            )
        control = stamp.control_step
        substep = stamp.physics_substep
        count = stamp.physics_substeps_per_control
        sim_before = stamp.sim_step_before
        if (
            type(control) is not int
            or control < 1
            or type(substep) is not int
            or type(count) is not int
            or count < 1
            or substep < 0
            or substep >= count
            or type(sim_before) is not int
            or sim_before < 0
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "pre-physics stamp scalar chronology differs"
            )
        with self._lock:
            self._enter("before physics substep")
            try:
                self.require_healthy()
                if control != self._last_before_policy_control_step:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "pre-physics control step differs from policy step"
                    )
                previous = self._last_prephysics_position
                expected = (
                    (control, 0)
                    if previous is None or previous[0] != control
                    else (control, previous[1] + 1)
                )
                if (control, substep) != expected:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "pre-physics substep was skipped, duplicated, or replayed"
                    )
                if (
                    previous is not None
                    and previous[0] == control
                    and count != previous[2]
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "pre-physics decimation changed within one control step"
                    )
                if previous is not None and sim_before != previous[3] + 1:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "pre-physics simulator step is not contiguous"
                    )
                launch = self._bound_plain_method(
                    self._physical_ball, "launch_action_epoch"
                )
                if launch() is not None:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "Physical launch opportunity must return None"
                    )
                self._last_prephysics_position = (
                    control,
                    substep,
                    count,
                    sim_before,
                )
            finally:
                self._leave()

    def after_command_compute_before_observation(self, control_step: int) -> None:
        """Run conditional D05 settlement and arm real R03 before physics."""

        with self._lock:
            self._enter("after command compute")
            try:
                self.require_healthy()
                step = _plain_nonnegative_int(control_step, label="control_step")
                genesis = (
                    step == 0
                    and not self._genesis_after_command_completed
                    and self._last_before_policy_control_step == 0
                    and self._pending_after_command_control_step is None
                )
                ordinary = (
                    step >= 1
                    and step == self._last_before_policy_control_step
                    and step == self._pending_after_command_control_step
                )
                if not genesis and not ordinary:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "command-compute settlement is stale, skipped, or replayed"
                    )
                # R05 owns the no-argument full-N transaction.  Internally it
                # asks ActionEpoch to pull Motion's current projection, close
                # and retire prior rows, then settles the one active D05 view.
                # No caller mask/count can become a second row authority.
                settle = self._bound_plain_method(
                    self._r05_runtime,
                    "advance_action_ball_full_mdp_rows",
                )
                try:
                    if settle() is not None:
                        raise ActionBallFullMdpLeanRuntimeError(
                            "D05 row-wise settlement must return None"
                        )
                except BaseException as exc:
                    self._poison_locked(
                        "D05 row-wise settlement failed: "
                        + type(exc).__name__
                    )
                    try:
                        self._epoch.poison_owner_write(
                            "r05_runtime", 24, owner=self._r05_runtime
                        )
                    except BaseException as attribution_exc:
                        raise ActionBallFullMdpLeanRuntimeError(
                            "D05 failure poison attribution failed: "
                            + type(attribution_exc).__name__
                        ) from exc
                    raise
                # Fresh Racket compute deliberately never arms R03.  This is
                # the sole callpoint, after the possible D05 epoch change, for
                # genesis, ordinary not-due cadence and due reveal alike.
                arm = self._bound_plain_method(
                    self._racket,
                    "arm_action_ball_full_mdp_epoch_strike_fact",
                )
                if arm() is not None:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "Racket epoch strike-fact arm must return None"
                    )
                if genesis:
                    self._genesis_after_command_completed = True
                else:
                    self._pending_after_command_control_step = None
            finally:
                self._leave()

    def action_epoch_observation_v1(
        self, record: epoch_v1.ActionEpochRecord
    ) -> object:
        """Delegate exact real-source extraction to the observation ABI."""

        module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_observation_cfg"
        )
        build = getattr(module, "build_direct_action_epoch_observation_facts", None)
        if not callable(build):
            raise ActionBallFullMdpLeanRuntimeError(
                "direct ActionEpoch observation builder is absent"
            )
        return build(runtime_owner=self, record=record)

    def publish_post_physics_substep(self, stamp: object) -> None:
        """Publish the exact post-scene facts in their causal owner order."""

        stamp_type = type(stamp)
        stamp_module = sys.modules.get(stamp_type.__module__)
        if (
            stamp_type.__module__
            != "whole_body_tracking.tasks.tracking.full_mdp_env"
            or stamp_type.__qualname__ != "FullMdpPhysicsSubstepStamp"
            or stamp_module is None
            or vars(stamp_module).get("FullMdpPhysicsSubstepStamp")
            is not stamp_type
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "post-physics stamp must be the exact environment type"
            )
        control = stamp.control_step
        substep = stamp.physics_substep
        count = stamp.physics_substeps_per_control
        sim_step = stamp.sim_step
        phase = stamp.event_phase
        phase_type = vars(stamp_module).get("FullMdpPhysicsEventPhase")
        if (
            type(control) is not int
            or control < 1
            or type(substep) is not int
            or type(count) is not int
            or count < 1
            or substep < 0
            or substep >= count
            or type(sim_step) is not int
            or sim_step < 1
            or phase_type is None
            or type(phase) is not phase_type
            or phase is not phase_type.POST_SCENE_UPDATE
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "post-physics stamp scalar chronology differs"
            )

        with self._lock:
            self._enter("publish post-physics substep")
            try:
                self.require_healthy()
                if control != self._last_before_policy_control_step:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-physics control step differs from policy step"
                    )
                prephysics = self._last_prephysics_position
                if (
                    prephysics is None
                    or prephysics[:2] != (control, substep)
                    or count != prephysics[2]
                    or sim_step != prephysics[3] + 1
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-physics stamp does not follow its exact pre-physics call"
                    )
                previous = self._last_postphysics_position
                expected = (
                    (control, 0)
                    if previous is None or previous[0] != control
                    else (control, previous[1] + 1)
                )
                if (control, substep) != expected:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-physics substep was skipped, duplicated, or replayed"
                    )
                if (
                    previous is not None
                    and previous[0] == control
                    and count != previous[2]
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-physics decimation changed within one control step"
                    )
                if previous is not None and sim_step != previous[3] + 1:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "post-physics simulator step is not contiguous"
                    )

                physical_publish = self._bound_plain_method(
                    self._physical_ball, "publish_action_epoch_post_physics"
                )
                publication_started = False
                try:
                    publication_started = True
                    result = physical_publish(stamp)
                    if result is not None:
                        raise ActionBallFullMdpLeanRuntimeError(
                            "Physical post-physics publisher must return None"
                        )
                    if substep == count - 1:
                        racket_publish = self._bound_plain_method(
                            self._racket,
                            "publish_action_ball_full_mdp_epoch_strike_fact",
                        )
                        if racket_publish(source_step=control) is not None:
                            raise ActionBallFullMdpLeanRuntimeError(
                                "Racket epoch strike-fact publisher must return None"
                            )
                        recovery_publish = self._bound_plain_method(
                            self._r07_recovery, "publish_epoch_reward_facts"
                        )
                        source_step = torch.full(
                            (self._epoch.num_envs,),
                            control - 1,
                            dtype=torch.int64,
                            device=self._epoch.device,
                        )
                        recovery_publish(current_source_step=source_step)
                        project_ready = self._bound_plain_method(
                            self._r07_recovery, "motion_ready_projection"
                        )
                        ready_projection = project_ready()
                        if ready_projection is None:
                            raise ActionBallFullMdpLeanRuntimeError(
                                "R07 Motion-ready projection is absent"
                            )
                        install_ready = self._bound_plain_method(
                            self._motion,
                            "install_action_ball_continuous_r07_ready_projection",
                        )
                        if install_ready(ready_projection) is not None:
                            raise ActionBallFullMdpLeanRuntimeError(
                                "Motion R07-ready installer must return None"
                            )
                except BaseException:
                    if publication_started:
                        self._poison_locked(
                            "post-physics owner chain failed after Physical "
                            "publication began"
                        )
                    raise
                self._last_postphysics_position = (
                    control,
                    substep,
                    count,
                    sim_step,
                )
            finally:
                self._leave()

    def after_reward_close(self, control_step: int) -> None:
        with self._lock:
            self._enter("after Reward close")
            try:
                self.require_healthy()
                step = _plain_nonnegative_int(control_step, label="control_step")
                if (
                    step < 1
                    or step != self._last_before_policy_control_step
                    or step != self._last_reward_control_step + 1
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "after-Reward control step was skipped, duplicated, or replayed"
                    )
                graph = self._reward_graph()
                if graph.poisoned or graph.cycle_open:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "RewardManager left a poisoned or open ActionEpoch cycle"
                    )
                completed = graph.completed_cycle_count
                if (
                    type(completed) is not int
                    or completed != self._last_reward_cycle_count + 1
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "RewardManager skipped, duplicated, or replayed its cycle"
                    )
                if graph.actual_closed_cycle_count != completed:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "RewardManager skipped or replayed actual-buffer close"
                    )
                # ActionEpoch is the sole payment authority.  R06 may close
                # its typed mailbox only after the graph proves that all
                # fourteen payments completed and the epoch cycle is closed.
                # The no-argument R06 method joins the current epoch identity
                # itself; a caller mask/key would create a second authority.
                try:
                    publish_payment = self._bound_plain_method(
                        self._epoch, "publish_reward_payment"
                    )
                    payment = publish_payment(step)
                    payment_type = getattr(
                        epoch_v1, "ActionEpochRewardPaymentRows", None
                    )
                    if type(payment_type) is not type or type(payment) is not payment_type:
                        raise ActionBallFullMdpLeanRuntimeError(
                            "ActionEpoch Reward payment publication type differs"
                        )
                    close_r06 = self._bound_plain_method(
                        self._r06_landing_outcome,
                        "close_action_ball_full_mdp_epoch_reward_rows",
                    )
                    if close_r06() is not None:
                        raise ActionBallFullMdpLeanRuntimeError(
                            "R06 typed epoch Reward close must return None"
                        )
                except BaseException as exc:
                    self._poison_locked(
                        "R06 typed reward close failed after ActionEpoch cycle "
                        "closed: "
                        + type(exc).__name__
                    )
                    raise
                # Commit the top-level chronology only after the typed R06
                # close succeeds.  A failure cannot be replayed as if this
                # cycle had cleanly crossed the external callpoint.
                self._last_reward_cycle_count = completed
                self._last_reward_control_step = step
            finally:
                self._leave()

    def _clear_selected_reset_locked(self) -> None:
        self._selected_reset_event = None
        self._selected_reset_projection = None
        self._selected_reset_prepared = None
        self._selected_reset_child_commits = None
        self._selected_reset_child_commits_started = False
        self._selected_reset_r05_receipt = None
        self._selected_reset_completions = None
        self._selected_reset_env_record = None
        self._selected_reset_live_ledger_identity = None
        self._selected_reset_epoch_prepared = None
        self._selected_reset_packed_preflight = None
        self._selected_reset_leaf_completions_consumed = False

    def _poison_selected_reset_locked(self, reason: str) -> None:
        """Fail-stop every mutation owner after the irreversible arm."""

        self._poison_locked(reason)
        for child, method_name in (
            (self._motion, "poison_global_reveal_epoch"),
            (self._racket, "poison_global_reveal_epoch"),
            (self._physical_ball, "poison_selected_reset"),
            (self._r06_landing_outcome, "poison_selected_reset"),
        ):
            try:
                self._bound_plain_method(child, method_name)(reason)
            except BaseException:
                pass
        try:
            self._bound_plain_method(
                self._r05_runtime, "poison_from_external_failure"
            )(11)
        except BaseException:
            pass
        try:
            self._epoch.poison_owner_write(
                "r05_runtime", 11, owner=self._r05_runtime
            )
        except BaseException:
            pass

    def _abort_selected_reset_precommit_locked(
        self,
        *,
        prepared: object,
        motion_value: object,
        racket_value: object,
        r06_prepared: object,
        physical_value: object,
    ) -> None:
        failures = []
        for label, owner, method_name, value in (
            (
                "physical_ball",
                self._physical_ball,
                "abort_selected_true_reset",
                physical_value,
            ),
            (
                "r06_flight",
                self._r06_landing_outcome,
                "abort_selected_reset",
                r06_prepared,
            ),
            (
                "racket",
                self._racket,
                "abort_prevalidated_action_ball_continuous_racket_selected_reset",
                racket_value,
            ),
            (
                "motion",
                self._motion,
                "abort_prevalidated_action_ball_continuous_motion_selected_reset",
                motion_value,
            ),
        ):
            if value is None:
                continue
            try:
                self._bound_plain_method(owner, method_name)(value)
            except BaseException as exc:
                failures.append(label + ":" + type(exc).__name__)
        if prepared is not None:
            try:
                self._bound_plain_method(
                    self._r05_runtime, "abort_true_reset_many"
                )(prepared)
            except BaseException as exc:
                failures.append("r05_runtime:" + type(exc).__name__)
        epoch_prepared = self._selected_reset_epoch_prepared
        if epoch_prepared is not None:
            try:
                self._epoch.abort_selected_true_reset(
                    owner=self, prepared_reset=epoch_prepared
                )
            except BaseException as exc:
                failures.append("action_epoch:" + type(exc).__name__)
        if failures:
            self._poison_selected_reset_locked(
                "selected-reset precommit abort was incomplete: "
                + ",".join(failures)
            )
        else:
            self._clear_selected_reset_locked()

    @staticmethod
    def _require_reset_tensor(
        value: object,
        *,
        label: str,
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            type(value) is not torch.Tensor
            or value.device != device
            or tuple(value.shape) != shape
            or value.dtype != dtype
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                label + " must be the exact device tensor"
            )
        return value

    def _validate_env_selected_reset_locked(
        self, event: object, projection: object
    ) -> None:
        """Join the clone-only argument back to the env's active record."""

        env_module = importlib.import_module(type(self._env).__module__)
        event_type = getattr(env_module, "FullMdpSelectedResetEvent", None)
        projection_type = getattr(
            env_module, "FullMdpSelectedResetProjection", None
        )
        record_type = getattr(
            env_module, "_FullMdpSelectedResetRecord", None
        )
        record = getattr(
            self._env, "_action_ball_full_mdp_active_reset_record", None
        )
        if (
            event_type is None
            or projection_type is None
            or record_type is None
            or type(event) is not event_type
            or type(projection) is not projection_type
            or type(record) is not record_type
            or record.event is not event
            or record.reset_event_identity is not projection.reset_event_identity
            or record.projected is not True
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "selected-reset event is not the env's exact active projection"
            )
        device = self._epoch.device
        count = self._epoch.num_envs
        index = self._require_reset_tensor(
            projection.selected_env_index,
            label="selected-reset index",
            device=device,
            shape=(projection.selected_env_index.shape[0],)
            if type(projection.selected_env_index) is torch.Tensor
            and projection.selected_env_index.ndim == 1
            else (-1,),
            dtype=torch.int64,
        )
        if index.shape[0] < 1:
            raise ActionBallFullMdpLeanRuntimeError(
                "selected-reset index cannot be empty"
            )
        tensor_specs = (
            ("selected_mask", torch.bool),
            ("generation_before", torch.int64),
            ("generation_after", torch.int64),
            ("generation_overflow_fault", torch.bool),
        )
        for name, dtype in tensor_specs:
            self._require_reset_tensor(
                getattr(projection, name, None),
                label="selected-reset " + name,
                device=device,
                shape=(count,),
                dtype=dtype,
            )
            self._require_reset_tensor(
                getattr(record, name, None),
                label="env selected-reset " + name,
                device=device,
                shape=(count,),
                dtype=dtype,
            )
        self._require_reset_tensor(
            getattr(projection, "terminal_reset_facts_i64", None),
            label="selected-reset terminal_reset_facts_i64",
            device=device,
            shape=(count, 3),
            dtype=torch.int64,
        )
        self._require_reset_tensor(
            getattr(record, "terminal_reset_facts_i64", None),
            label="env selected-reset terminal_reset_facts_i64",
            device=device,
            shape=(count, 3),
            dtype=torch.int64,
        )
        original_index = self._require_reset_tensor(
            record.selected_env_index,
            label="env selected-reset index",
            device=device,
            shape=(index.shape[0],),
            dtype=torch.int64,
        )
        self._selected_reset_event = event
        self._selected_reset_projection = projection
        self._selected_reset_env_record = record

    def project_r05_true_reset(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
        live_reset_ledger_identity: object,
        live_reset_generation: torch.Tensor,
    ) -> device_r05.DeviceTrueResetEventProjection:
        """Project only the env-bound selection after independent preflight."""

        with self._lock:
            carry_txn._require_leaf_mutable(self)
            projection = self._selected_reset_projection
            record = self._selected_reset_env_record
            if (
                receipt is not self._selected_reset_event
                or projection is None
                or record is None
                or self._selected_reset_prepared is not None
                or self._selected_reset_live_ledger_identity is not None
                or type(num_envs) is not int
                or num_envs != self._epoch.num_envs
                or torch.device(device) != self._epoch.device
                or live_reset_ledger_identity is None
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "Device-R05 selected-reset projection is stale or foreign"
                )
            count = self._epoch.num_envs
            index = self._require_reset_tensor(
                projection.selected_env_index,
                label="Device-R05 selected-reset index",
                device=self._epoch.device,
                shape=(projection.selected_env_index.shape[0],),
                dtype=torch.int64,
            )
            selected = self._require_reset_tensor(
                projection.selected_mask,
                label="Device-R05 selected-reset mask",
                device=self._epoch.device,
                shape=(count,),
                dtype=torch.bool,
            )
            live = self._require_reset_tensor(
                live_reset_generation,
                label="Device-R05 live reset generation",
                device=self._epoch.device,
                shape=(count,),
                dtype=torch.int64,
            )
            self._selected_reset_live_ledger_identity = (
                live_reset_ledger_identity
            )
            return device_r05.DeviceTrueResetEventProjection(
                reset_event_identity=projection.reset_event_identity,
                selected_env_index=index.clone(),
                selected_mask=selected.clone(),
            )

    def _packed_selected_reset_preflight_locked(self, prepared: object) -> None:
        """Consume one reset-boundary D2H verdict before any leaf method.

        Every packed lane is computed from independently owned live state:
        the env active record, ActionEpoch, or Device-R05's retained prepare.
        No caller boolean is interpreted as authorization.
        """

        projection = self._selected_reset_projection
        record = self._selected_reset_env_record
        if projection is None or record is None:
            raise ActionBallFullMdpLeanRuntimeError(
                "selected-reset packed preflight lacks env authority"
            )
        claim = self._bound_plain_method(
            self._r05_runtime, "require_owned_prepared_true_reset"
        )(prepared, owner_kind="motion")
        if (
            type(claim) is not device_r05.DeviceR05PreparedTrueResetProjection
            or claim.prepared_true_reset is not prepared
            or claim.owner_kind != "motion"
            or claim.prepared_identity is not prepared
            or claim.reset_event_identity is not projection.reset_event_identity
        ):
            raise ActionBallFullMdpLeanRuntimeError(
                "Device-R05 selected-reset prepare identity differs"
            )
        count = self._epoch.num_envs
        device = self._epoch.device
        index = self._require_reset_tensor(
            projection.selected_env_index,
            label="selected-reset packed index",
            device=device,
            shape=(projection.selected_env_index.shape[0],),
            dtype=torch.int64,
        )
        selected = self._require_reset_tensor(
            projection.selected_mask,
            label="selected-reset packed mask",
            device=device,
            shape=(count,),
            dtype=torch.bool,
        )
        epoch_generation = self._epoch.current().reset_generation
        d05_mask = self._require_reset_tensor(
            claim.selected_mask,
            label="Device-R05 prepared mask",
            device=device,
            shape=(count,),
            dtype=torch.bool,
        )
        d05_before = self._require_reset_tensor(
            claim.generation_before,
            label="Device-R05 prepared generation before",
            device=device,
            shape=(count,),
            dtype=torch.int64,
        )
        d05_after = self._require_reset_tensor(
            claim.generation_after,
            label="Device-R05 prepared generation after",
            device=device,
            shape=(count,),
            dtype=torch.int64,
        )
        d05_fault = self._require_reset_tensor(
            claim.generation_overflow_fault,
            label="Device-R05 prepared generation fault",
            device=device,
            shape=(count,),
            dtype=torch.bool,
        )
        d05_writer_fault = self._require_reset_tensor(
            claim.writer_fault,
            label="Device-R05 prepared writer fault",
            device=device,
            shape=(),
            dtype=torch.bool,
        )
        terminal_reset_facts_i64 = self._require_reset_tensor(
            projection.terminal_reset_facts_i64,
            label="selected-reset packed terminal_reset_facts_i64",
            device=device,
            shape=(count, 3),
            dtype=torch.int64,
        )
        env_terminal_reset_facts_i64 = self._require_reset_tensor(
            record.terminal_reset_facts_i64,
            label="env selected-reset packed terminal_reset_facts_i64",
            device=device,
            shape=(count, 3),
            dtype=torch.int64,
        )
        terminal_common_step = terminal_reset_facts_i64[:, 0]
        terminal_episode_tick = terminal_reset_facts_i64[:, 1]
        terminal_reason_bits = terminal_reset_facts_i64[:, 2]
        unselected = ~selected
        unknown_reason_bits = torch.bitwise_and(
            terminal_reason_bits,
            torch.full_like(terminal_reason_bits, ~31),
        )
        safe_index = torch.clamp(index, min=0, max=count - 1)
        reconstructed = torch.zeros_like(selected)
        reconstructed.index_fill_(0, safe_index, True)
        independent_overflow = selected & d05_before.eq(
            torch.iinfo(torch.int64).max
        )
        increment = selected & ~independent_overflow
        safe_base = torch.where(
            increment, d05_before, torch.zeros_like(d05_before)
        )
        expected_after = torch.where(
            increment, safe_base + torch.ones_like(d05_before), d05_before
        )

        def mismatch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.sum(left != right, dtype=torch.int64)

        packed = torch.stack(
            (
                torch.sum((index < 0) | (index >= count), dtype=torch.int64),
                torch.sum(index[1:] <= index[:-1], dtype=torch.int64),
                torch.sum(reconstructed != selected, dtype=torch.int64),
                (~torch.any(selected)).to(torch.int64),
                mismatch(index, record.selected_env_index),
                mismatch(selected, record.selected_mask),
                mismatch(
                    projection.generation_before, record.generation_before
                ),
                mismatch(
                    projection.generation_after, record.generation_after
                ),
                mismatch(
                    projection.generation_overflow_fault,
                    record.generation_overflow_fault,
                ),
                mismatch(
                    terminal_reset_facts_i64,
                    env_terminal_reset_facts_i64,
                ),
                torch.sum(
                    unselected
                    & (
                        terminal_common_step.ne(-1)
                        | terminal_episode_tick.ne(-1)
                        | terminal_reason_bits.ne(0)
                    ),
                    dtype=torch.int64,
                ),
                torch.sum(
                    selected
                    & (
                        terminal_common_step.lt(1)
                        | terminal_episode_tick.lt(1)
                        | terminal_reason_bits.eq(0)
                        | unknown_reason_bits.ne(0)
                    ),
                    dtype=torch.int64,
                ),
                mismatch(d05_mask, selected),
                mismatch(d05_before, epoch_generation),
                mismatch(d05_before, projection.generation_before),
                mismatch(d05_after, expected_after),
                mismatch(d05_fault, independent_overflow),
                torch.sum(independent_overflow, dtype=torch.int64),
                d05_writer_fault.to(torch.int64),
            )
        ).contiguous()
        packed_host = packed.detach().to(device="cpu", non_blocking=False)
        packed_bytes = bytes(packed_host.untyped_storage())[
            : packed_host.numel() * 8
        ]
        faults = struct.unpack("=" + "q" * packed_host.numel(), packed_bytes)
        if any(value != 0 for value in faults):
            raise ActionBallFullMdpLeanRuntimeError(
                "selected-reset packed preflight rejected env/D05/epoch join"
            )
        self._selected_reset_packed_preflight = object.__new__(
            _SelectedResetPackedPreflight
        )

    def require_owned_r05_true_reset_preflight(
        self,
        prepared: object,
        *,
        preflight_capability: object,
    ) -> device_r05.DeviceTrueResetPreflightProjection:
        """Register the exact clean packed verdict back into Device-R05."""

        with self._lock:
            projection = self._selected_reset_projection
            record = self._selected_reset_env_record
            if (
                prepared is not self._selected_reset_prepared
                or projection is None
                or record is None
                or type(preflight_capability) is not _SelectedResetPackedPreflight
                or preflight_capability
                is not self._selected_reset_packed_preflight
                or self._selected_reset_epoch_prepared is not None
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "Device-R05 packed preflight is stale or foreign"
                )
            return device_r05.DeviceTrueResetPreflightProjection(
                prepared_true_reset=prepared,
                reset_event_identity=record.reset_event_identity,
                preflight_capability=preflight_capability,
            )

    def require_owned_epoch_selected_reset_preflight(
        self,
        preflight: object,
        *,
        selected_env_index: torch.Tensor,
        selected_mask: torch.Tensor,
        generation_before: torch.Tensor,
        generation_after: torch.Tensor,
        generation_overflow_fault: torch.Tensor,
        terminal_reset_facts_i64: torch.Tensor,
    ) -> object:
        """Let ActionEpoch consume only this transaction's clean D2H edge."""

        with self._lock:
            projection = self._selected_reset_projection
            if (
                type(preflight) is not _SelectedResetPackedPreflight
                or preflight is not self._selected_reset_packed_preflight
                or projection is None
                or self._selected_reset_prepared is None
                or self._selected_reset_epoch_prepared is not None
                or selected_env_index is not projection.selected_env_index
                or selected_mask is not projection.selected_mask
                or generation_before is not projection.generation_before
                or generation_after is not projection.generation_after
                or generation_overflow_fault
                is not projection.generation_overflow_fault
                or terminal_reset_facts_i64
                is not projection.terminal_reset_facts_i64
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "ActionEpoch selected-reset preflight is stale or foreign"
                )
            return preflight

    def require_owned_epoch_selected_reset_commit(
        self,
        preflight: object,
        prepared_reset: object,
    ) -> object:
        """Join the epoch publication to real R05-last leaf completions."""

        with self._lock:
            if (
                type(preflight) is not _SelectedResetPackedPreflight
                or preflight is not self._selected_reset_packed_preflight
                or prepared_reset is not self._selected_reset_epoch_prepared
                or self._selected_reset_r05_receipt is None
                or type(self._selected_reset_completions) is not dict
                or not self._selected_reset_leaf_completions_consumed
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "ActionEpoch selected-reset commit lacks R05-last leaf completion"
                )
            return prepared_reset

    def require_owned_r05_true_reset_commit(
        self,
        prepared: object,
        *,
        owner_view: object,
    ) -> device_r05.DeviceTrueResetCommitProjection:
        with self._lock:
            record = self._selected_reset_env_record
            commits = self._selected_reset_child_commits
            if (
                prepared is not self._selected_reset_prepared
                or record is None
                or type(commits) is not tuple
                or len(commits) != len(device_r05.CHILD_OWNER_ORDER)
                or not self._selected_reset_child_commits_started
                or type(self._selected_reset_packed_preflight)
                is not _SelectedResetPackedPreflight
                or type(owner_view) is not device_r05.DeviceR05TrueResetCommitInput
                or owner_view.prepared_true_reset is not prepared
                or owner_view.reset_event_identity
                is not record.reset_event_identity
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "Device-R05 selected-reset writer view differs"
                )
            validators = (
                self._bound_plain_method(
                    self._motion, "require_owned_selected_reset_commit"
                ),
                self._bound_plain_method(
                    self._racket, "require_owned_selected_reset_commit"
                ),
                self._bound_plain_method(
                    self._physical_ball, "require_owned_selected_reset_commit"
                ),
                self._bound_plain_method(
                    self._r06_landing_outcome,
                    "require_owned_selected_reset_commit",
                ),
            )
            owned = tuple(
                validator(commit)
                if kind == "physical_ball"
                else validator(
                    commit, expected_prepared_true_reset=prepared
                )
                for kind, validator, commit in zip(
                    device_r05.CHILD_OWNER_ORDER, validators, commits
                )
            )
            if any(value is not commit for value, commit in zip(owned, commits)):
                raise ActionBallFullMdpLeanRuntimeError(
                    "selected-reset child commit identity differs"
                )
            return device_r05.DeviceTrueResetCommitProjection(
                prepared_true_reset=prepared,
                reset_event_identity=record.reset_event_identity,
                child_kinds=device_r05.CHILD_OWNER_ORDER,
                child_commit_identities=commits,
                preflight_capability=self._selected_reset_packed_preflight,
            )

    def require_owned_r05_true_reset_abort(
        self, prepared: object
    ) -> device_r05.DeviceTrueResetAbortProjection:
        with self._lock:
            projection = self._selected_reset_projection
            if (
                prepared is not self._selected_reset_prepared
                or projection is None
                or self._selected_reset_child_commits_started
                or self._selected_reset_child_commits is not None
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "selected-reset abort proof is unavailable"
                )
            return device_r05.DeviceTrueResetAbortProjection(
                prepared_true_reset=prepared,
                reset_event_identity=projection.reset_event_identity,
                child_commits_started=False,
            )

    def require_owned_r05_true_reset_child_completion(
        self,
        receipt: object,
        *,
        child_kind: str,
        child_receipt: object,
    ) -> device_r05.DeviceTrueResetChildCompletionProjection:
        with self._lock:
            completions = self._selected_reset_completions
            if (
                receipt is not self._selected_reset_r05_receipt
                or type(completions) is not dict
                or child_kind not in device_r05.CHILD_OWNER_ORDER
                or completions.get(child_kind) is not child_receipt
            ):
                raise ActionBallFullMdpLeanRuntimeError(
                    "selected-reset child completion differs"
                )
            return device_r05.DeviceTrueResetChildCompletionProjection(
                true_reset_receipt=receipt,
                child_kind=child_kind,
                child_receipt=child_receipt,
            )

    def selected_true_reset(self, event: object, projection: object) -> None:
        """Commit one env-selected reset in the fixed leaf-last order."""

        with self._lock:
            self._enter("selected true reset")
            prepared = None
            motion_value = None
            racket_value = None
            r06_prepared = None
            physical_value = None
            irreversible = False
            try:
                self.require_healthy()
                if any(
                    value is not None
                    for value in (
                        self._selected_reset_event,
                        self._selected_reset_projection,
                        self._selected_reset_prepared,
                        self._selected_reset_child_commits,
                        self._selected_reset_r05_receipt,
                        self._selected_reset_completions,
                        self._selected_reset_epoch_prepared,
                        self._selected_reset_packed_preflight,
                    )
                ):
                    raise ActionBallFullMdpLeanRuntimeError(
                        "one selected-reset transaction is already active"
                    )
                self._validate_env_selected_reset_locked(event, projection)
                prepared = self._bound_plain_method(
                    self._r05_runtime, "prepare_true_reset_many"
                )(event)
                self._selected_reset_prepared = prepared
                # The sole reset-boundary D2H is intentionally outside the
                # physics/PPO hot paths and precedes every leaf method call.
                self._packed_selected_reset_preflight_locked(prepared)
                self._bound_plain_method(
                    self._r05_runtime, "register_true_reset_preflight"
                )(prepared, self._selected_reset_packed_preflight)
                self._selected_reset_epoch_prepared = (
                    self._epoch.prepare_selected_true_reset(
                        owner=self,
                        top_preflight=self._selected_reset_packed_preflight,
                        selected_env_index=projection.selected_env_index,
                        selected_mask=projection.selected_mask,
                        generation_before=projection.generation_before,
                        generation_after=projection.generation_after,
                        generation_overflow_fault=(
                            projection.generation_overflow_fault
                        ),
                        terminal_reset_facts_i64=(
                            projection.terminal_reset_facts_i64
                        ),
                    )
                )

                motion_stage = self._bound_plain_method(
                    self._motion,
                    "prepare_action_ball_continuous_motion_selected_reset",
                )(prepared)
                motion_value = motion_stage
                motion_prevalidated = self._bound_plain_method(
                    self._motion,
                    "arm_prevalidated_action_ball_continuous_motion_selected_reset",
                )(motion_stage)
                motion_value = motion_prevalidated

                racket_stage = self._bound_plain_method(
                    self._racket,
                    "stage_action_ball_continuous_racket_selected_reset",
                )(prepared)
                racket_value = racket_stage
                racket_prevalidated = self._bound_plain_method(
                    self._racket,
                    "finalize_action_ball_continuous_racket_selected_reset",
                )(racket_stage)
                racket_value = racket_prevalidated

                r06_prepared = self._bound_plain_method(
                    self._r06_landing_outcome, "prepare_selected_reset"
                )(prepared)
                physical_stage = self._bound_plain_method(
                    self._physical_ball, "stage_selected_true_reset"
                )(r06_prepared)
                physical_value = physical_stage
                physical_finalized = self._bound_plain_method(
                    self._physical_ball, "finalize_selected_true_reset"
                )(physical_stage)
                physical_value = physical_finalized

                # R06 arm is the first leaf point with no rollback API.  All
                # four leaf after-images and both independent generation
                # checks exist before this line.
                irreversible = True
                r06_armed = self._bound_plain_method(
                    self._r06_landing_outcome,
                    "arm_prevalidated_selected_reset",
                )(r06_prepared, physical_finalized)
                physical_armed = self._bound_plain_method(
                    self._physical_ball, "prearm_selected_true_reset"
                )(physical_finalized, r06_armed)

                self._selected_reset_child_commits_started = True
                physical_commit = self._bound_plain_method(
                    self._physical_ball,
                    "commit_prevalidated_selected_true_reset",
                )(physical_armed)
                if self._bound_plain_method(
                    self._physical_ball, "require_owned_selected_reset_commit"
                )(physical_commit) is not physical_commit:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "physical selected-reset commit identity differs"
                    )
                r06_commit = self._bound_plain_method(
                    self._r06_landing_outcome,
                    "commit_prevalidated_selected_reset",
                )(r06_armed, physical_commit)
                if self._bound_plain_method(
                    self._r06_landing_outcome,
                    "require_owned_selected_reset_commit",
                )(
                    r06_commit, expected_prepared_true_reset=prepared
                ) is not r06_commit:
                    raise ActionBallFullMdpLeanRuntimeError(
                        "R06 selected-reset commit identity differs"
                    )
                self._bound_plain_method(
                    self._physical_ball,
                    "acknowledge_r06_selected_reset_commit",
                )(physical_commit, r06_commit)
                motion_commit = self._bound_plain_method(
                    self._motion,
                    "commit_prevalidated_action_ball_continuous_motion_selected_reset",
                )(motion_prevalidated)
                racket_commit = self._bound_plain_method(
                    self._racket,
                    "commit_prevalidated_action_ball_continuous_racket_selected_reset",
                )(racket_prevalidated)
                commits = (
                    motion_commit,
                    racket_commit,
                    physical_commit,
                    r06_commit,
                )
                self._selected_reset_child_commits = commits
                # Device-R05 asks this exact top to revalidate the four real
                # owner commits, then publishes last.
                r05_receipt = self._bound_plain_method(
                    self._r05_runtime, "commit_true_reset_many"
                )(prepared)
                self._selected_reset_r05_receipt = r05_receipt

                motion_completion = self._bound_plain_method(
                    self._motion,
                    "complete_action_ball_continuous_motion_selected_reset_after_r05",
                )(motion_commit, r05_receipt)
                racket_completion = self._bound_plain_method(
                    self._racket,
                    "complete_action_ball_continuous_racket_selected_reset_after_r05",
                )(racket_commit, r05_receipt)
                r06_completion = self._bound_plain_method(
                    self._r06_landing_outcome,
                    "complete_selected_reset_after_r05",
                )(r06_commit, r05_receipt)
                physical_completion = self._bound_plain_method(
                    self._physical_ball,
                    "complete_selected_true_reset_after_r05",
                )(physical_commit, r06_commit, r05_receipt)
                completions = {
                    "motion": motion_completion,
                    "racket": racket_completion,
                    "physical_ball": physical_completion,
                    "r06_flight": r06_completion,
                }
                self._selected_reset_completions = completions
                for child_kind in device_r05.CHILD_OWNER_ORDER:
                    self._bound_plain_method(
                        self._r05_runtime,
                        "record_true_reset_child_completion",
                    )(
                        r05_receipt,
                        child_kind=child_kind,
                        child_receipt=completions[child_kind],
                    )

                self._bound_plain_method(
                    self._motion, "consume_owned_selected_reset_completion"
                )(
                    motion_completion,
                    expected_prepared_true_reset=prepared,
                )
                self._bound_plain_method(
                    self._racket, "consume_owned_selected_reset_completion"
                )(
                    racket_completion,
                    expected_prepared_true_reset=prepared,
                )
                self._bound_plain_method(
                    self._physical_ball,
                    "consume_owned_selected_reset_completion",
                )(physical_completion)
                self._bound_plain_method(
                    self._r06_landing_outcome,
                    "consume_owned_selected_reset_completion",
                )(r06_completion)
                self._selected_reset_leaf_completions_consumed = True

                self._epoch.commit_selected_true_reset(
                    owner=self,
                    prepared_reset=self._selected_reset_epoch_prepared,
                )
                self._clear_selected_reset_locked()
                return None
            except BaseException as exc:
                if irreversible:
                    self._poison_selected_reset_locked(
                        "selected reset failed after irreversible arm: "
                        + type(exc).__name__
                    )
                else:
                    self._abort_selected_reset_precommit_locked(
                        prepared=prepared,
                        motion_value=motion_value,
                        racket_value=racket_value,
                        r06_prepared=r06_prepared,
                        physical_value=physical_value,
                    )
                    # A clean rollback of unpublished child after-images does
                    # not make the env event reusable.  Its private record is
                    # already projected and Device-R05 has consumed one
                    # prepare identity.  Keep the top fail-stop so a caller
                    # cannot rebuild a value-equal projection for the same
                    # opaque event and obtain a second prepare.
                    self._poison_locked(
                        "selected reset failed before irreversible arm: "
                        + type(exc).__name__
                    )
                raise
            finally:
                self._leave()


__all__ = [
    "ActionBallFullMdpLeanRuntimeError",
    "ActionBallFullMdpLeanRuntimeOwner",
    "ActionEpochPpoBoundarySummary",
    "DIAGNOSTIC_UNAUTHORIZED",
    "DIAGNOSTIC_DEPENDENCY_KIND",
    "DRAIN_SUMMARY_KIND",
    "DRAIN_SCHEMA_VERSION",
    "EpochDrainFrontier",
    "LAUNCH_AUTHORIZED",
    "RUNTIME_INTEGRATED",
]
