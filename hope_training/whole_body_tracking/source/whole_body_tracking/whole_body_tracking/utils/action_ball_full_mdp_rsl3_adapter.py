"""One RSL-RL 3 optimizer edge for diagnostic FullMDP.

The environment owns every semantic fact.  This adapter owns only the order
between the zero-argument PPO update and the two durable WAL records.  It is
not a checkpoint, a readiness receipt, or a second telemetry authority.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Callable

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


RUN_MODE = "single_action_lean"
TELEMETRY_SCHEMA_VERSION = 10
TELEMETRY_KIND = "action_ball_epoch_optimizer_update_ack_telemetry_v10"


def _bound(owner: object, name: str) -> Callable:
    method = getattr(owner, name, None)
    function = vars(type(owner)).get(name)
    if (
        not callable(method)
        or not callable(function)
        or getattr(method, "__self__", None) is not owner
        or getattr(method, "__func__", None) is not function
    ):
        raise RuntimeError(f"FullMDP owner lacks exact bound method {name}")
    return method


def _shot_row(shot: object, *, lifecycle_flags: tuple[str, ...]) -> dict:
    row = asdict(shot)
    evidence = row["evidence"]
    bits = evidence["lifecycle_bits"]
    lifecycle = {
        name: bool(bits & (1 << ordinal))
        for ordinal, name in enumerate(lifecycle_flags)
    }
    evidence["lifecycle"] = lifecycle
    evidence["contact_face"] = {"availability": "not_produced"}
    evidence["recovery_horizon"] = {"availability": "not_produced"}
    if not lifecycle["physical_launched"]:
        row["target_x_m"] = None
        row["target_y_m"] = None
    return row


def _telemetry(summary: object, runtime_module: object) -> dict:
    """Project one exact owner-produced summary without re-owning its facts."""

    summary_type = runtime_module.ActionEpochPpoBoundarySummary
    frontier_type = runtime_module.EpochDrainFrontier
    if type(summary) is not summary_type or type(summary.frontier) is not frontier_type:
        raise RuntimeError("FullMDP owner returned a foreign summary")
    frontier = summary.frontier
    drain = runtime_module.drain_v2
    lifecycle_flags = drain.SHOT_LIFECYCLE_FLAGS
    if type(lifecycle_flags) is not tuple:
        raise RuntimeError("FullMDP lifecycle ABI differs")
    rewards = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_rewards"
    )
    reset_rows = [asdict(row) for row in summary.terminal_resets]
    reset_counts = {
        "terminal_reset_reason_time_out_count": 0,
        "terminal_reset_reason_base_fell_tilt_count": 0,
        "terminal_reset_reason_base_too_low_count": 0,
        "terminal_reset_reason_joint_qdes_forbidden_count": 0,
        "terminal_reset_reason_robot_hit_table_count": 0,
    }
    for row in reset_rows:
        for name, bit in zip(reset_counts, (1, 2, 4, 8, 16)):
            reset_counts[name] += int(bool(row["reason_bits"] & bit))
    settlement = summary.settlement
    commits = summary.reveal_commit
    lifecycle = summary.lifecycle
    faults = summary.owner_faults
    continuation = summary.continuation
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "kind": TELEMETRY_KIND,
        "diagnostic_unauthorized": True,
        "ppo_update": frontier.update_index,
        "completed_environment_steps": frontier.completed_environment_steps,
        "epoch_operation_sequence": frontier.operation_sequence,
        "epoch_drain_sequence": frontier.drain_sequence,
        "epoch_commit_start": frontier.start_commit,
        "epoch_commit_end": frontier.end_commit,
        "shot_slot_capacity": frontier.shot_slot_capacity,
        "d05_transactions": settlement.transactions,
        "d05_due_rows": settlement.due_rows,
        "d05_selected_rows": settlement.selected_rows,
        "d05_accepted_rows": settlement.accepted,
        "d05_censored_rows": settlement.censored,
        "d05_rejected_rows": settlement.rejected,
        "d05_deferred_rows": settlement.deferred,
        "d05_not_ready_rows": settlement.not_ready,
        "motion_committed_rows": commits.motion_committed_rows,
        "racket_committed_rows": commits.racket_committed_rows,
        "r05_committed_rows": commits.r05_committed_rows,
        "playback_started_rows": lifecycle.playback_started_rows,
        "closed_unplayed_rows": lifecycle.closed_unplayed_rows,
        "physical_launch_rows": lifecycle.physical_launch_rows,
        "outcome_settled_rows": lifecycle.outcome_settled_rows,
        "payment_recorded_rows": lifecycle.payment_recorded_rows,
        "retired_rows": lifecycle.retired_rows,
        "terminal_shot_rows": lifecycle.terminal_shot_rows,
        "attributed_fault_rows": faults.attributed_fault_rows,
        "active_before": continuation.active_before,
        "active_after": continuation.active_after,
        "awaiting_playback_after": continuation.awaiting_playback_after,
        "awaiting_outcome_after": continuation.awaiting_outcome_after,
        "awaiting_payment_after": continuation.awaiting_payment_after,
        "action_opportunities": [asdict(row) for row in summary.action_opportunities],
        "completed_shots": [
            _shot_row(row, lifecycle_flags=lifecycle_flags)
            for row in summary.completed_shots
        ],
        "terminal_shots": [
            _shot_row(row, lifecycle_flags=lifecycle_flags)
            for row in summary.terminal_shots
        ],
        "terminal_resets": reset_rows,
        "terminal_reset_rows": len(reset_rows),
        "milestone": summary.milestone.as_json(tuple(rewards.MANAGER_NAMES)),
        **reset_counts,
    }


class ActionBallFullMdpRsl3Adapter:
    """Bind one exact lean owner to one RSL-RL 3 optimizer call."""

    def __init__(self, *, env: object, owner: object, log_dir: str) -> None:
        runtime = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_runtime"
        )
        if type(owner) is not runtime.ActionBallFullMdpLeanRuntimeOwner:
            raise RuntimeError("single_action_lean requires the exact runtime owner")
        runtime_env = getattr(env, "unwrapped", env)
        lease = getattr(runtime_env, "action_ball_full_mdp_runtime_lease", None)
        getter = getattr(runtime_env, "action_ball_full_mdp_ppo_drain_owner", None)
        if (
            lease is None
            or owner.full_mdp_runtime_env is not runtime_env
            or owner.full_mdp_runtime_lease is not lease
            or not callable(getter)
            or getattr(getter, "__self__", None) is not runtime_env
            or getter(lease) is not owner
            or owner.epoch_owner.num_envs != runtime_env.num_envs
            or owner.epoch_owner.shot_slot_capacity != 1
        ):
            raise RuntimeError("single_action_lean runtime graph identity differs")
        self._runtime = runtime
        self._owner = owner
        self._require_healthy = _bound(owner, "require_healthy")
        self._prepare = _bound(owner, "prepare_pre_optimizer_ppo_boundary")
        self._mark_returned = _bound(owner, "mark_optimizer_returned")
        self._prepare_summary = _bound(owner, "prepare_post_update_summary")
        self._ack = _bound(owner, "acknowledge_post_update")
        self._latch = _bound(owner, "_record_durable_epoch_ack_span")
        self._poison = _bound(owner, "poison_optimizer_boundary")
        self._wal = importlib.import_module(
            "whole_body_tracking.utils.action_ball_full_mdp_durable_wal"
        )
        self._rank = 0
        self._path, self._identity, self._segment = self._create_wal(log_dir)
        self._size = 0
        self._last_update = -1
        self._require_healthy()

    @staticmethod
    def _create_wal(log_dir: str) -> tuple[Path, tuple[int, int], str]:
        if type(log_dir) is not str or not log_dir:
            raise RuntimeError("single_action_lean requires an exact log directory")
        root = Path(log_dir)
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("single_action_lean log directory differs")
        directory = root / "action_ball_epoch_durable_wal"
        directory.mkdir(mode=0o700)
        path = directory / "rank_0000.jsonl"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("single_action_lean WAL is not a unique regular file")
            os.fsync(fd)
        finally:
            os.close(fd)
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        identity = (info.st_dev, info.st_ino)
        return path, identity, f"{info.st_dev:x}:{info.st_ino:x}"

    def _append(self, line: bytes) -> tuple[int, int]:
        if type(line) is not bytes or line.count(b"\n") != 1 or not line.endswith(b"\n"):
            raise RuntimeError("single_action_lean WAL row is not one JSONL line")
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self._path, flags)
        start = self._size
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino) != self._identity
                or info.st_nlink != 1
                or info.st_size != start
            ):
                raise RuntimeError("single_action_lean WAL frontier changed")
            written = os.write(fd, line)
            if written != len(line):
                raise OSError("single_action_lean WAL append was short")
            os.fsync(fd)
            end = start + len(line)
            if os.fstat(fd).st_size != end:
                raise RuntimeError("single_action_lean WAL append frontier differs")
        except BaseException:
            try:
                if os.fstat(fd).st_size > start:
                    os.ftruncate(fd, start)
                    os.fsync(fd)
            except BaseException:
                pass
            raise
        finally:
            os.close(fd)
        self._size = end
        return start, end

    def update(
        self,
        algorithm_update: Callable[[], object],
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        if (
            not callable(algorithm_update)
            or type(update_index) is not int
            or update_index != self._last_update + 1
            or type(completed_environment_steps) is not int
            or completed_environment_steps <= 0
        ):
            raise RuntimeError("single_action_lean optimizer chronology differs")
        boundary = None
        try:
            self._require_healthy()
            boundary = self._prepare(
                update_index=update_index,
                completed_environment_steps=completed_environment_steps,
            )
            result = algorithm_update()
            self._mark_returned(boundary, update_index=update_index)
            summary = self._prepare_summary(boundary, update_index=update_index)
            record = _telemetry(summary, self._runtime)
            canonical, pending_line = self._wal.encode_pending(
                segment_id=self._segment, rank=self._rank, telemetry=record
            )
            pending_start, pending_end = self._append(pending_line)
            if self._ack(boundary, summary, update_index=update_index) is not summary:
                raise RuntimeError("single_action_lean owner changed ACK summary identity")
            ack_line = self._wal.encode_epoch_ack(
                pending_line=pending_line,
                pending_byte_start=pending_start,
                pending_byte_end=pending_end,
            )
            ack_start, ack_end = self._append(ack_line)
            self._latch(
                summary,
                update_index=update_index,
                segment_id=self._segment,
                rank=self._rank,
                pending_byte_start=pending_start,
                pending_byte_end=pending_end,
                ack_byte_start=ack_start,
                ack_byte_end=ack_end,
            )
            marker = "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" + canonical.decode("utf-8") + "\n"
            if sys.stdout.write(marker) != len(marker):
                raise OSError("single_action_lean telemetry stdout was short")
            sys.stdout.flush()
            self._last_update = update_index
            return result
        except BaseException as exc:
            reason = (
                "single_action_lean optimizer boundary failed; retry forbidden: "
                f"{type(exc).__module__}.{type(exc).__qualname__}"
            )
            try:
                self._poison(boundary, update_index=update_index, reason=reason)
            except BaseException:
                pass
            raise RuntimeError(reason) from exc


class ActionBallFullMdpRsl3Runner(OnPolicyRunner):
    """Unmodified RSL-RL 3 loop with one exact optimizer-boundary adapter."""

    def __init__(
        self,
        env: object,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: object = None,
        *,
        training_contract_schema_version: int | None = None,
        training_contract_sha256: str | None = None,
        training_contract_lineage_exact: bool = False,
        training_launch_claim_sha256: str | None = None,
        require_exact_resume_state: bool = False,
        action_ball_r10_checkpoint_adapter: object = None,
        action_ball_r10_cold_restore_capsule: object = None,
        action_ball_full_mdp_runtime_owner: object = None,
        action_ball_full_mdp_run_mode: object = None,
    ) -> None:
        if (
            action_ball_full_mdp_run_mode != RUN_MODE
            or action_ball_full_mdp_runtime_owner is None
            or action_ball_r10_checkpoint_adapter is not None
            or action_ball_r10_cold_restore_capsule is not None
            or type(training_contract_lineage_exact) is not bool
            or type(require_exact_resume_state) is not bool
        ):
            raise RuntimeError("FullMDP RSL3 runner accepts only fresh single_action_lean")
        _bound(action_ball_full_mdp_runtime_owner, "require_healthy")()
        super().__init__(env, train_cfg, log_dir, device)
        if self.is_distributed:
            raise RuntimeError("single_action_lean RSL3 runner is single-process only")
        self.registry_name = registry_name
        self.training_contract_schema_version = training_contract_schema_version
        self.training_contract_sha256 = training_contract_sha256
        self.training_contract_lineage_exact = training_contract_lineage_exact
        self.training_launch_claim_sha256 = training_launch_claim_sha256
        self.require_exact_resume_state = require_exact_resume_state
        self.empirical_normalization = bool(train_cfg.get("empirical_normalization"))
        if log_dir is None:
            raise RuntimeError("single_action_lean RSL3 runner requires log_dir")
        self._full_mdp_adapter = ActionBallFullMdpRsl3Adapter(
            env=env, owner=action_ball_full_mdp_runtime_owner, log_dir=log_dir
        )
        original_update = self.alg.update
        if not callable(original_update) or getattr(original_update, "__self__", None) is not self.alg:
            raise RuntimeError("RSL3 PPO update is not the exact bound algorithm method")

        def update_with_full_mdp_boundary():
            update_index = self._full_mdp_adapter._last_update + 1
            return self._full_mdp_adapter.update(
                original_update,
                update_index=update_index,
                completed_environment_steps=(
                    (update_index + 1) * int(self.env.num_envs) * int(self.num_steps_per_env)
                ),
            )

        self.alg.update = update_with_full_mdp_boundary

    @staticmethod
    def _normalizer_aliases(role: str) -> tuple[str, ...]:
        if role == "actor":
            return ("actor_obs_normalizer",)
        if role == "critic":
            return ("critic_obs_normalizer",)
        raise ValueError("normalizer role must be actor or critic")

    def _resolve_runtime_normalizer(self, role: str):
        aliases = self._normalizer_aliases(role)
        policy = self.alg.policy
        name = aliases[0]
        value = getattr(policy, name, None)
        return name, value, aliases

    def save(self, path: str, infos: dict | None = None) -> None:
        """Deliberately produce no resumable file for the diagnostic lane."""

        print(
            "[ActionBallFullMdpRsl3Runner] checkpoint skipped: "
            "single_action_lean has no complete plant/trainer restore contract",
            flush=True,
        )

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        raise RuntimeError("single_action_lean forbids checkpoint load/resume")
