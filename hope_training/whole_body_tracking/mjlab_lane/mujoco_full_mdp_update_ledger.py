"""Minimal transactional update ledger for portable MuJoCo FullMDP."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, math, os, re
SCHEMA_VERSION, REWARD_TERM_COUNT = 3, 20
EXACT_TERMINATION_BITS = {
    "time_out": 1, "base_fell_tilt": 2, "base_too_low": 4,
    "joint_qdes_forbidden": 8, "robot_hit_table": 16,
}
EXACT_PHASE_CODES = (0, 2, 5, 6, 8)
EVENT_NAMES = (
    "reveal_rows", "reveal_due_rows", "reveal_deferred_rows", "launch_rows", "flight_terminal_rows",
    "shot_retired_rows", "completed_action_epoch_rows", "selected_reset_rows",
    "racket_contact_eligible_rows", "racket_contact_rows", "selected_contact_rows",
    "opposite_contact_rows", "edge_contact_rows", "between_contact_rows", "invalid_contact_rows",
    "r03_present_rows", "r03_physically_valid_rows", "landing_crossing_rows",
    "r06_present_rows", "r06_eligible_rows", "r06_common_rows", "r07_present_rows", "r07_eligible_rows",
    "recovery_success_rows", "recovery_failure_rows", "recovery_timeout_rows",
)
EVENT_FIELDS = tuple((name, "full_a_" + name[:-5] + "_event") for name in EVENT_NAMES)
_MISC_NAMES = (
    "gym_reset_rows", "unknown_terminal_rows", "invalid_done_rows",
    "done_explanation_fault_rows", "time_out_rows", "timeout_fault_rows",
    "selected_reset_fault_rows", "reset_generation_rows",
    "reset_generation_fault_rows", "resolved_table_rows",
    "landing_on_opponent_rows", "landing_opponent_bound_rows",
    "classification_unknown_rows", "event_semantics_fault_rows",
    "reward20_finite_rows",
    "reward20_nonfinite_rows", "actual_reward_finite_rows",
    "actual_reward_nonfinite_rows", "conservation_fault_rows", "slot0_rows",
    "uid_rows", "mount_sign_rows", "identity_rows",
    "outcome_unknown_rows", "outcome_event_code_fault_rows",
    "phase_unknown_rows",
)
_ZERO_FAULTS = (
    "unknown_terminal_rows", "invalid_done_rows", "done_explanation_fault_rows",
    "timeout_fault_rows", "selected_reset_fault_rows", "reset_generation_fault_rows",
    "classification_unknown_rows", "reward20_nonfinite_rows", "actual_reward_nonfinite_rows",
    "conservation_fault_rows", "event_semantics_fault_rows",
    "outcome_unknown_rows", "outcome_event_code_fault_rows",
    "phase_unknown_rows",
)
@dataclass(frozen=True)
class PreparedUpdate:
    update_index: int
    token: int
    payload: bytes
class FullMdpUpdateLedger:
    """Accumulate one rollout on-device, then prepare-before/ACK-after PPO."""
    def __init__(self, *, torch_module, num_envs: int, num_steps_per_env: int,
                 device, termination_bits: dict[str, int], action_slot: int,
                 action_uid: int, mount_normal_sign: int, family: str,
                 initial_reset_generation, run_identity: dict) -> None:
        torch = torch_module
        if (type(num_envs) is not int or num_envs <= 0
                or type(num_steps_per_env) is not int or num_steps_per_env <= 0
                or action_slot != 0 or type(action_uid) is not int
                or mount_normal_sign != 1 or family not in ("forehand", "backhand")):
            raise ValueError("FullMDP ledger dimensions or action identity differ")
        if (type(run_identity) is not dict
                or set(run_identity) != {
                    "source_commit", "run_namespace", "mujoco_warp_runtime"
                }
                or type(run_identity.get("source_commit")) is not str
                or re.fullmatch(r"[0-9a-f]{40}", run_identity["source_commit"]) is None
                or type(run_identity.get("run_namespace")) is not str
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}",
                    run_identity["run_namespace"],
                ) is None
                or type(run_identity.get("mujoco_warp_runtime")) is not dict):
            raise ValueError("FullMDP run identity differs")
        runtime = run_identity["mujoco_warp_runtime"]
        if termination_bits != EXACT_TERMINATION_BITS:
            raise ValueError("termination bit schema differs")
        bits = tuple(sorted(EXACT_TERMINATION_BITS.items(), key=lambda row: row[1]))
        if not (isinstance(initial_reset_generation, torch.Tensor)
                and initial_reset_generation.dtype == torch.long
                and tuple(initial_reset_generation.shape) == (num_envs,)
                and initial_reset_generation.device == device):
            raise ValueError("initial reset generation differs")
        self.torch, self.device = torch, device
        self.num_envs, self.num_steps = num_envs, num_steps_per_env
        self.term_bits, self.known_term_mask = bits, sum(bit for _name, bit in bits)
        self.time_out_bit = termination_bits["time_out"]
        self.action_uid, self.mount_normal_sign = action_uid, mount_normal_sign
        self.family = family
        self.run_identity = {
            "source_commit": run_identity["source_commit"],
            "run_namespace": run_identity["run_namespace"],
            "mujoco_warp_runtime": dict(runtime),
        }
        self._reset_generation = initial_reset_generation.detach().clone()
        counters = (
            len(EVENT_FIELDS), len(bits), 6, 7, len(EXACT_PHASE_CODES), len(_MISC_NAMES),
            REWARD_TERM_COUNT, 1, 3,
        )
        (self._event, self._terminal, self._classification, self._outcome,
         self._phase, self._misc, self._reward, self._actual, self._episode) = (
            torch.zeros(size, dtype=torch.float64, device=device) for size in counters)
        self._episode_return = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self._episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._steps, self._faults, self._next, self._token = 0, set(), 0, 0
        self._pending = None
        self._last_run_elapsed_seconds = 0.0
    def _vector(self, extras, key, dtype):
        value = extras.get(key)
        if not (isinstance(value, self.torch.Tensor) and value.dtype == dtype
                and tuple(value.shape) == (self.num_envs,)
                and value.device == self.device):
            self._faults.add(key)
            return self.torch.zeros(self.num_envs, dtype=dtype, device=self.device)
        return value
    def ingest(self, result) -> None:
        torch = self.torch
        self._steps += 1
        if (not isinstance(result, tuple) or len(result) != 4
                or not isinstance(result[3], dict)):
            self._faults.add("step_result")
            return
        actual, done, extras = result[1], result[2], result[3]
        events = [self._vector(extras, key, torch.bool) for _name, key in EVENT_FIELDS]
        event = dict(zip(EVENT_NAMES, events))
        bits = self._vector(extras, "termination_bits", torch.long)
        if not (isinstance(done, torch.Tensor) and done.dtype in (torch.bool, torch.long)
                and tuple(done.shape) == (self.num_envs,) and done.device == self.device):
            self._faults.add("done")
            done = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        invalid_done = (torch.zeros_like(done, dtype=torch.bool)
                        if done.dtype == torch.bool else ~(done.eq(0) | done.eq(1)))
        done = done if done.dtype == torch.bool else done.eq(1)
        time_outs = self._vector(extras, "time_outs", torch.bool)
        resolved = self._vector(extras, "backend_resolved_table_contact", torch.bool)
        generation = self._vector(extras, "reset_generation", torch.long)
        slot = self._vector(extras, "full_a_action_slot", torch.long)
        uid = self._vector(extras, "full_a_action_uid", torch.long)
        sign = self._vector(extras, "full_a_mount_normal_sign", torch.int8)
        status = self._vector(extras, "full_a_contact_classification_status", torch.int8)
        outcome = self._vector(extras, "full_a_outcome_code", torch.long)
        phase = self._vector(extras, "full_a_phase_before_reset", torch.long)
        landing = self._vector(extras, "full_a_landing_on_opponent", torch.bool)
        opponent_bound = self._vector(extras, "full_a_landing_opponent_bound", torch.bool)
        terms = extras.get("reward_terms")
        if not (isinstance(terms, torch.Tensor) and torch.is_floating_point(terms)
                and tuple(terms.shape) == (self.num_envs, REWARD_TERM_COUNT)
                and terms.device == self.device):
            self._faults.add("reward_terms")
            terms = torch.full((self.num_envs, REWARD_TERM_COUNT), torch.nan,
                               dtype=torch.float32, device=self.device)
        if not (isinstance(actual, torch.Tensor) and torch.is_floating_point(actual)
                and tuple(actual.shape) == (self.num_envs,)
                and actual.device == self.device):
            self._faults.add("actual_reward")
            actual = torch.full(
                (self.num_envs,), torch.nan, dtype=torch.float32, device=self.device
            )
        terms_finite = torch.isfinite(terms).all(1)
        actual_finite = torch.isfinite(actual)
        conserved = terms_finite & actual_finite & torch.isclose(
            actual, terms.sum(1), rtol=1.0e-5, atol=1.0e-7
        )
        selected_reset = event["selected_reset_rows"]
        landing_event = event["landing_crossing_rows"]
        generation_delta = generation - self._reset_generation
        terminal_present = bits.ne(0)
        timeout_bit = torch.bitwise_and(bits, self.time_out_bit).ne(0)
        timeout_fault = time_outs.ne(timeout_bit) | (
            timeout_bit & (bits.ne(self.time_out_bit) | resolved))
        identity = slot.eq(0) & uid.eq(self.action_uid) & sign.eq(self.mount_normal_sign)
        classified = torch.stack(
            [event[name] for name in (
                "selected_contact_rows", "opposite_contact_rows",
                "edge_contact_rows", "between_contact_rows",
                "invalid_contact_rows",
            )]
        )
        classified_count = classified.sum(dim=0, dtype=torch.long)
        expected_status = (
            event["selected_contact_rows"].to(torch.int8)
            + event["opposite_contact_rows"].to(torch.int8) * 2
            + event["edge_contact_rows"].to(torch.int8) * 3
            + event["between_contact_rows"].to(torch.int8) * 4
            + event["invalid_contact_rows"].to(torch.int8) * 5
        )
        recovery_count = torch.stack(
            [event[name] for name in (
                "recovery_success_rows", "recovery_failure_rows",
                "recovery_timeout_rows",
            )]
        ).sum(dim=0, dtype=torch.long)
        natural_recovery = event["recovery_success_rows"] | event["recovery_timeout_rows"]
        phase2, phase5, phase6, phase8 = (phase.eq(code) for code in (2, 5, 6, 8))
        outcome_event = event["flight_terminal_rows"]
        legal_outcome = outcome.eq(3)
        own_table_outcome = outcome.eq(4)
        event_semantics_fault = (
            classified_count.ne(event["racket_contact_rows"].to(torch.long))
            | status.ne(expected_status)
            | event["racket_contact_eligible_rows"].ne(event["launch_rows"])
            | (event["r03_physically_valid_rows"] & ~event["r03_present_rows"])
            | (event["landing_crossing_rows"] & ~event["flight_terminal_rows"])
            | event["r06_present_rows"].ne(event["flight_terminal_rows"])
            | (event["r06_eligible_rows"] & ~event["r06_present_rows"])
            | (event["r06_common_rows"] & ~event["r06_eligible_rows"])
            | (event["r07_eligible_rows"] & ~event["r07_present_rows"])
            | recovery_count.gt(1)
            | event["reveal_due_rows"].ne(
                event["reveal_rows"] | event["reveal_deferred_rows"]
            )
            | (event["reveal_rows"] & event["reveal_deferred_rows"])
            | (event["reveal_rows"] & ~phase2)
            | (event["launch_rows"] & ~phase5)
            | (outcome_event & ~(phase6 | phase8))
            | (event["r06_present_rows"] & ~(phase6 | phase8))
            | (event["r07_present_rows"] & ~(phase6 | phase8))
            | event["shot_retired_rows"].ne(natural_recovery)
            | (event["completed_action_epoch_rows"] & ~event["shot_retired_rows"])
            | (event["recovery_failure_rows"] & ~done)
            | (natural_recovery & done)
            | (event["shot_retired_rows"] & (~phase8 | done))
            | selected_reset.ne(done)
            | (outcome_event & outcome.eq(0))
            | (outcome_event & legal_outcome & ~(landing & opponent_bound))
            | (outcome_event & legal_outcome & ~event["landing_crossing_rows"])
            | (outcome_event & own_table_outcome & ~event["landing_crossing_rows"])
        )
        self._event += torch.stack(events).sum(1, dtype=torch.float64)
        self._terminal += torch.stack(
            [torch.bitwise_and(bits, bit).ne(0) for _name, bit in self.term_bits]
        ).sum(1, dtype=torch.float64)
        self._classification += torch.stack([status.eq(index) for index in range(6)]
                                            ).sum(1, dtype=torch.float64)
        self._outcome += torch.stack(
            [outcome_event & outcome.eq(index) for index in range(7)]
        ).sum(1, dtype=torch.float64)
        self._phase += torch.stack(
            [phase.eq(index) for index in EXACT_PHASE_CODES]
        ).sum(1, dtype=torch.float64)
        self._misc += torch.stack(
            (
                done, torch.bitwise_and(bits, ~self.known_term_mask).ne(0),
                invalid_done, done.ne(terminal_present | resolved),
                time_outs, timeout_fault, selected_reset.ne(done),
                generation_delta, generation_delta.ne(done.to(torch.long)),
                resolved, landing & landing_event, opponent_bound & landing_event,
                status.lt(0) | status.gt(5), event_semantics_fault,
                terms_finite, ~terms_finite, actual_finite, ~actual_finite, ~conserved,
                slot.eq(0), uid.eq(self.action_uid), sign.eq(self.mount_normal_sign),
                identity, outcome.lt(0) | outcome.gt(6), outcome_event & outcome.eq(0),
                ~torch.stack(
                    [phase.eq(index) for index in EXACT_PHASE_CODES]
                ).any(dim=0),
            )
        ).sum(1, dtype=torch.float64)
        self._reward += torch.where(torch.isfinite(terms), terms,
                                    torch.zeros_like(terms)).sum(0, dtype=torch.float64)
        self._actual += torch.where(actual_finite, actual,
                                    torch.zeros_like(actual)).sum(0, dtype=torch.float64)
        safe_actual = torch.where(actual_finite, actual,
                                  torch.zeros_like(actual)).to(torch.float64)
        self._episode_return += safe_actual
        self._episode_length += 1
        self._episode += torch.stack(
            (
                done.sum(dtype=torch.float64),
                torch.where(done, self._episode_return,
                            torch.zeros_like(self._episode_return)).sum(),
                torch.where(done, self._episode_length,
                            torch.zeros_like(self._episode_length)).sum(
                    dtype=torch.float64
                ),
            )
        )
        self._episode_return.masked_fill_(done, 0.0)
        self._episode_length.masked_fill_(done, 0)
        self._reset_generation.copy_(generation)
    def prepare(self, update_index: int, *, environment_steps: int, storage_step: int,
                storage_tensors: dict, policy_std) -> PreparedUpdate:
        torch = self.torch
        if self._pending is not None or update_index != self._next:
            raise RuntimeError("FullMDP ledger update order differs")
        expected_steps = (update_index + 1) * self.num_steps
        storage_names = ("rewards", "returns", "advantages")
        if (type(storage_tensors) is not dict
                or set(storage_tensors) != set(storage_names) or any(
            not isinstance(storage_tensors[name], torch.Tensor)
            or not torch.is_floating_point(storage_tensors[name])
            or storage_tensors[name].device != self.device
            or tuple(storage_tensors[name].shape)
            != (self.num_steps, self.num_envs, 1)
            or not storage_tensors[name].is_contiguous()
            for name in storage_names
        )):
            self._faults.add("storage_tensors")
        if (
            not isinstance(policy_std, torch.Tensor)
            or not torch.is_floating_point(policy_std)
            or policy_std.device != self.device
            or tuple(policy_std.shape) != (self.num_envs, 31)
        ):
            self._faults.add("policy_std")
        if (self._steps != self.num_steps or self._faults or storage_step != self.num_steps
                or environment_steps != expected_steps):
            raise RuntimeError(
                "FullMDP ledger schema fault: "
                + ",".join(sorted(self._faults or {
                    f"steps={self._steps},storage_step={storage_step},environment_steps={environment_steps}"
                }))
            )
        storage_finite = torch.stack(tuple(
            torch.isfinite(storage_tensors[name]).all() for name in storage_names
        )).to(dtype=torch.float64)
        policy_summary = torch.stack((
            (torch.isfinite(policy_std) & policy_std.gt(0)).all().to(torch.float64),
            policy_std.mean().to(torch.float64),
        ))
        host = torch.cat(
            (self._event, self._terminal, self._classification, self._outcome,
             self._phase, self._misc, self._reward, self._actual,
             self._episode, storage_finite, policy_summary)
        ).cpu()
        values = host.tolist()
        cursor = 0
        take = lambda size: values.__getitem__(slice(cursor, cursor + size))
        event_values = take(len(EVENT_FIELDS)); cursor += len(EVENT_FIELDS)
        terminal_values = take(len(self.term_bits)); cursor += len(self.term_bits)
        classification_values = take(6); cursor += 6
        outcome_values = take(7); cursor += 7
        phase_values = take(len(EXACT_PHASE_CODES)); cursor += len(EXACT_PHASE_CODES)
        misc_values = take(len(_MISC_NAMES)); cursor += len(_MISC_NAMES)
        reward_values = take(REWARD_TERM_COUNT); cursor += REWARD_TERM_COUNT
        actual_sum = values[cursor]; cursor += 1
        episode_values = values[cursor:cursor + 3]; cursor += 3
        storage_values = values[cursor:cursor + len(storage_names)]
        cursor += len(storage_names)
        policy_finite, policy_mean_std = values[cursor:cursor + 2]
        if any(value != 1.0 for value in storage_values):
            raise RuntimeError("FullMDP rollout storage is nonfinite")
        if policy_finite != 1.0 or not math.isfinite(policy_mean_std) or policy_mean_std <= 0:
            raise RuntimeError("FullMDP policy std is nonfinite or nonpositive")
        events = {name: int(value) for (name, _key), value
                  in zip(EVENT_FIELDS, event_values)}
        terminal = {name: int(value) for (name, _bit), value
                    in zip(self.term_bits, terminal_values)}
        misc = {name: int(value) for name, value in zip(_MISC_NAMES, misc_values)}
        transitions = self.num_envs * self.num_steps
        bad = [name for name in _ZERO_FAULTS if misc[name] != 0]
        if bad or any(misc[name] != transitions for name in (
            "reward20_finite_rows", "actual_reward_finite_rows",
            "slot0_rows", "uid_rows", "identity_rows",
        )):
            raise RuntimeError("FullMDP ledger rollout evidence differs: "
                               + ",".join(bad or ["identity_or_finite_rows"]))
        family_counts = {"forehand": 0, "backhand": 0}
        family_counts[self.family] = misc["identity_rows"]
        record = {
            "schema_version": SCHEMA_VERSION, "record_type": "mujoco_full_mdp_update_ack",
            "diagnostic_unauthorized": True, "update_index": update_index,
            "run_identity": dict(self.run_identity),
            "num_envs": self.num_envs, "num_steps_per_env": self.num_steps,
            "transitions_delta": transitions, "transitions_cumulative": transitions * (update_index + 1),
            "environment_steps_delta": self.num_steps,
            "environment_steps_cumulative": environment_steps,
            "storage_finite": {name: bool(value) for name, value
                               in zip(storage_names, storage_values)},
            "extras_counts": events, "terminal_bit_counts": terminal,
            "classification_status_counts": {str(index): int(value) for index, value
                                             in enumerate(classification_values)},
            "outcome_code_counts": {str(index): int(value) for index, value
                                    in enumerate(outcome_values)},
            "phase_counts": {str(index): int(value) for index, value
                             in zip(EXACT_PHASE_CODES, phase_values)},
            "episodes": {"completed_count": int(episode_values[0]),
                         "return_sum": float(episode_values[1]),
                         "length_sum": int(episode_values[2])},
            "rollout_policy_mean_std": float(policy_mean_std),
            "selected_reset_rows": events["selected_reset_rows"], "gym_reset_rows": misc["gym_reset_rows"],
            "lifecycle_counts": {name: misc[name] for name in _MISC_NAMES[:14]},
            "reward20": {
                "term_sums": reward_values,
                "actual_reward_sum": actual_sum,
                **{name: misc[name] for name in ("reward20_finite_rows",
                    "reward20_nonfinite_rows", "actual_reward_finite_rows",
                    "actual_reward_nonfinite_rows", "conservation_fault_rows")},
            },
            "action_identity": {
                "action_slot": 0, "action_uid": self.action_uid,
                "mount_normal_sign": self.mount_normal_sign, "family": self.family,
                "family_source": "runner_pinned_identity",
                "observed_rows": transitions, "slot0_rows": misc["slot0_rows"],
                "uid_rows": misc["uid_rows"],
                "mount_sign_rows": misc["mount_sign_rows"],
                "identity_rows": misc["identity_rows"], "family_counts": family_counts,
            },
        }
        self._token += 1
        raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        prepared = PreparedUpdate(update_index, self._token, raw)
        self._pending = prepared
        return prepared
    def ack(self, prepared: PreparedUpdate, *, completed_updates: int, evidence_fd: int,
            optimizer_metrics: dict, learning_rate: float, timings: dict,
            snapshot: dict | None) -> bytes:
        if prepared is not self._pending or completed_updates != self._next + 1:
            raise RuntimeError("FullMDP ledger ACK order differs")
        required = {"value_function", "surrogate", "entropy"}
        if set(optimizer_metrics) != required or any(
            type(optimizer_metrics[name]) not in (int, float) or
            not math.isfinite(float(optimizer_metrics[name])) for name in required
        ) or (type(learning_rate) not in (int, float)
              or not math.isfinite(float(learning_rate)) or learning_rate <= 0):
            raise RuntimeError("FullMDP optimizer metrics differ")
        timing_names = ("collection_seconds", "learning_seconds",
                        "pre_ack_iteration_seconds", "run_elapsed_pre_ack_seconds")
        if (
            set(timings) != set(timing_names)
            or any(
                type(timings[name]) not in (int, float)
                or not math.isfinite(float(timings[name]))
                or float(timings[name]) <= 0
                for name in timing_names
            )
            or float(timings["pre_ack_iteration_seconds"])
            < float(timings["collection_seconds"]) + float(timings["learning_seconds"])
            or float(timings["run_elapsed_pre_ack_seconds"])
            <= self._last_run_elapsed_seconds
        ):
            raise RuntimeError("FullMDP update timings differ")
        if snapshot is not None and (
            type(snapshot) is not dict
            or set(snapshot) != {"name", "bytes", "sha256"}
            or snapshot["name"] != f"model_{prepared.update_index}.pt"
            or type(snapshot["bytes"]) is not int or snapshot["bytes"] <= 0
            or type(snapshot["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", snapshot["sha256"]) is None
        ):
            raise RuntimeError("FullMDP snapshot receipt differs")
        record = json.loads(prepared.payload)
        record["prepared_update_sha256"] = hashlib.sha256(prepared.payload).hexdigest()
        record["snapshot"] = snapshot
        record["optimizer_metrics"] = {name: float(optimizer_metrics[name])
                                       for name in sorted(required)}
        record["learning_rate"] = float(learning_rate)
        record["timings"] = {name: float(timings[name]) for name in timing_names}
        payload = json.dumps(
            record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        pending = memoryview(payload + b"\n")
        while pending:
            written = os.write(evidence_fd, pending)
            if written <= 0:
                raise OSError("FullMDP ledger append made no progress")
            pending = pending[written:]
        os.fsync(evidence_fd)
        self._last_run_elapsed_seconds = float(timings["run_elapsed_pre_ack_seconds"])
        for tensor in (
            self._event, self._terminal, self._classification, self._outcome,
            self._phase, self._misc, self._reward, self._actual, self._episode,
        ):
            tensor.zero_()
        self._steps, self._faults, self._pending = 0, set(), None
        self._next += 1
        return payload
