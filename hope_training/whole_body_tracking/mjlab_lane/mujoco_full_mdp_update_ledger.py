"""Minimal transactional update ledger for portable MuJoCo FullMDP."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib, importlib.util, json, math, os, re, sys
from pathlib import Path
SCHEMA_VERSION = 9

EXACT_RUNTIME_STACK = {
    "schema_version": 1,
    "mujoco_warp": {
        "schema_version": 1,
        "distribution": "mujoco-warp",
        "fork_id": "hope_mujoco_warp_epa48_v1",
        "version": "3.10.0.3+hope.epa48.1",
        "epa_horizon": 48,
        "types_py_sha256": (
            "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696"
        ),
        "wheel_sha256": (
            "58f47b1c3b4249d82666f25d3a302ff5a215043a3d7a3b9445a5ca7ef15b561a"
        ),
        "build_receipt_sha256": (
            "336f6454296d3c062e26fb0c330d6dbca4b2fd0ad4e50f386f8a647db013e041"
        ),
        "import_scope": "fresh_run_local_site",
    },
    "rsl_rl": {
        "distribution": "rsl-rl-lib",
        "version": "3.1.2",
        "wheel_sha256": (
            "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
        ),
        "import_scope": "fresh_run_local_site",
    },
    "mjlab": {
        "schema_version": 1,
        "distribution": "mjlab",
        "version": "1.5.3",
        "import_scope": "verified_venv_distribution",
        "selected_tree_scope": "mjlab/**/*.py+mjlab/scene/scene.xml",
        "selected_file_count": 193,
        "selected_byte_count": 1399177,
        "selected_tree_sha256": (
            "88c9725d0416b4ac3e21f6752ad423c13ea3b8cfb9e23ca664f8aba146cec33d"
        ),
        "mjlab_tasks_entry_point_count": 0,
    },
}


def _load_pinned_local_module(*, source: Path, name: str, subject: str):
    """Load one origin-pinned local module without caching failed imports."""
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError(f"cached {subject} origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {subject}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _plant_contract_module():
    return _load_pinned_local_module(
        source=Path(__file__).with_name(
            "mujoco_full_mdp_plant_contract.py"
        ).resolve(),
        name="_hope_mujoco_full_mdp_plant_contract",
        subject="MuJoCo plant contract",
    )


def _reward_contract_module():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_full_mdp_reward_contract.py"
    ).resolve()
    return _load_pinned_local_module(
        source=source,
        name="_hope_mujoco_ledger_action_ball_full_mdp_reward_contract",
        subject="FullMDP reward contract",
    )


def _portable_catalog_module():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_full_mdp_portable_catalog.py"
    ).resolve()
    return _load_pinned_local_module(
        source=source,
        name="_hope_mujoco_ledger_action_ball_full_mdp_portable_catalog",
        subject="FullMDP portable catalog",
    )


def _portable_observation_module():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_full_mdp_portable_observation.py"
    ).resolve()
    return _load_pinned_local_module(
        source=source,
        name="_hope_mujoco_ledger_action_ball_full_mdp_portable_observation",
        subject="FullMDP portable observation contract",
    )


def _exact_reward_contract() -> tuple[tuple[str, ...], int]:
    contract = _reward_contract_module()
    names = getattr(contract, "MANAGER_NAMES", None)
    count = getattr(contract, "REWARD_TERM_COUNT", None)
    if (
        type(names) is not tuple
        or not names
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
        or type(count) is not int
        or count != len(names)
    ):
        raise RuntimeError("FullMDP reward contract differs")
    return names, count


REWARD_TERM_NAMES, REWARD_TERM_COUNT = _exact_reward_contract()


def _exact_observation_storage_widths() -> tuple[int, int]:
    contract = _portable_observation_module()
    actor_width = getattr(contract, "ACTOR_WIDTH_V3", None)
    critic_width = getattr(contract, "CRITIC_WIDTH_V3", None)
    if (
        getattr(contract, "OBSERVATION_KIND_V3", None)
        != "action_ball_full_mdp_semantic_observation_v3"
        or type(actor_width) is not int
        or type(critic_width) is not int
        or actor_width <= 0
        or critic_width <= actor_width
        or actor_width != sum(
            width for _name, width in contract.ACTOR_LAYOUT_V3
        )
        or critic_width
        != actor_width
        + sum(
            width
            for _name, width in contract.CRITIC_EXTENSION_LAYOUT_V3
        )
    ):
        raise RuntimeError("FullMDP semantic observation V3 contract differs")
    return actor_width, critic_width


(
    FULLMDP_ACTOR_OBSERVATION_WIDTH,
    FULLMDP_CRITIC_OBSERVATION_WIDTH,
) = _exact_observation_storage_widths()


def _exact_paddle_prior_contract():
    contract = _reward_contract_module()
    specs = getattr(contract, "PADDLE_MOTION_PRIOR_SPECS", None)
    if type(specs) is not tuple or not specs:
        raise RuntimeError("FullMDP paddle prior contract differs")
    names = tuple(spec.manager_name for spec in specs)
    weights = tuple(float(spec.manager_weight) for spec in specs)
    start = REWARD_TERM_COUNT - len(names)
    if (
        REWARD_TERM_NAMES[start:] != names
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in weights
        )
    ):
        raise RuntimeError("FullMDP paddle prior contract differs")
    return names, weights, start


(
    PADDLE_PRIOR_TERM_NAMES,
    PADDLE_PRIOR_WEIGHTS,
    PADDLE_PRIOR_TERM_START,
) = _exact_paddle_prior_contract()
PADDLE_PRIOR_TERM_COUNT = len(PADDLE_PRIOR_TERM_NAMES)
FULLMDP_POLICY_STEP_S = float(
    _portable_catalog_module().FRESH_POLICY_STEP_S
)
if not math.isfinite(FULLMDP_POLICY_STEP_S) or FULLMDP_POLICY_STEP_S <= 0.0:
    raise RuntimeError("FullMDP policy step differs")
EXACT_TERMINATION_BITS = {
    "time_out": 1, "base_fell_tilt": 2, "base_too_low": 4,
    "joint_qdes_forbidden": 8, "robot_hit_table": 16,
}
EXACT_PHASE_CODES = (0, 2, 5, 6, 8)
EVENT_NAMES = (
    "scheduled_due_rows", "due_terminal_overlap_rows",
    "reveal_rows", "reveal_due_rows", "reveal_deferred_rows", "launch_rows",
    "missed_launch_rows", "flight_terminal_rows",
    "shot_retired_rows", "completed_action_epoch_rows", "selected_reset_rows",
    "racket_contact_rows", "selected_contact_rows",
    "opposite_contact_rows", "edge_contact_rows", "between_contact_rows", "invalid_contact_rows",
    "actual_hard_edge_rows", "qdes_guard_intervention_rows",
    "r03_present_rows", "r03_physically_valid_rows", "landing_crossing_rows",
    "r06_present_rows", "r06_eligible_rows", "r06_common_rows", "r07_present_rows", "r07_eligible_rows",
    "recovery_success_rows", "recovery_failure_rows", "recovery_timeout_rows",
    "recovery_completion_fault_rows",
)
EVENT_FIELDS = tuple((name, "full_a_" + name[:-5] + "_event") for name in EVENT_NAMES)
STORAGE_FLOAT_WIDTHS = (
    ("observations_policy", FULLMDP_ACTOR_OBSERVATION_WIDTH),
    ("observations_critic", FULLMDP_CRITIC_OBSERVATION_WIDTH),
    ("actions", 31), ("values", 1), ("actions_log_prob", 1),
    ("mu", 31), ("sigma", 31), ("rewards", 1), ("returns", 1),
    ("advantages", 1),
)
STORAGE_DOMAIN_NAMES = ("dones_binary", "sigma_positive")
FACT_INTEGRITY_CAUSES = (
    ("fact_integrity_r03_nonfinite_rows", 1 << 0),
    ("fact_integrity_r06_source_invalid_rows", 1 << 1),
    ("fact_integrity_r07_sequence_rows", 1 << 2),
    ("fact_integrity_r07_nonfinite_rows", 1 << 3),
)
FACT_INTEGRITY_COUNT_NAMES = tuple(
    name for name, _bit in FACT_INTEGRITY_CAUSES
) + ("fact_integrity_unknown_bits_rows",)
FACT_INTEGRITY_KNOWN_MASK = sum(bit for _name, bit in FACT_INTEGRITY_CAUSES)
LIFECYCLE_COUNT_NAMES = (
    "gym_reset_rows", "unknown_terminal_rows", "invalid_done_rows",
    "done_explanation_fault_rows", "time_out_rows", "timeout_fault_rows",
    "selected_reset_fault_rows", "reset_generation_rows",
    "reset_generation_fault_rows", "resolved_table_rows",
    "landing_on_opponent_rows", "landing_opponent_bound_rows",
    "classification_unknown_rows",
)
_MISC_NAMES = LIFECYCLE_COUNT_NAMES + (
    # Internal, rowwise producer-partition checks.  They fail before an ACK is
    # serialized and therefore do not create a second public evidence schema.
    "scheduled_due_partition_fault_rows",
    "reveal_due_partition_fault_rows",
    "r03_subset_fault_rows",
    "r06_subset_fault_rows",
    "reward_terms_finite_rows",
    "reward_terms_nonfinite_rows", "actual_reward_finite_rows",
    "actual_reward_nonfinite_rows", "conservation_fault_rows", "slot0_rows",
    "uid_rows", "mount_sign_rows", "identity_rows",
    "outcome_unknown_rows", "outcome_event_code_fault_rows",
    "phase_unknown_rows",
    *FACT_INTEGRITY_COUNT_NAMES,
)
_ZERO_FAULTS = (
    "unknown_terminal_rows", "invalid_done_rows", "done_explanation_fault_rows",
    "timeout_fault_rows", "selected_reset_fault_rows", "reset_generation_fault_rows",
    "classification_unknown_rows", "reward_terms_nonfinite_rows", "actual_reward_nonfinite_rows",
    "conservation_fault_rows",
    "scheduled_due_partition_fault_rows",
    "reveal_due_partition_fault_rows",
    "r03_subset_fault_rows",
    "r06_subset_fault_rows",
    "outcome_unknown_rows", "outcome_event_code_fault_rows",
    "phase_unknown_rows",
    *FACT_INTEGRITY_COUNT_NAMES,
)
@dataclass(frozen=True)
class PreparedUpdate:
    update_index: int
    token: int
    payload: bytes


def _plant_model_is_exact(value) -> bool:
    return _plant_contract_module().plant_model_identity_is_exact(value)


def _tree_is_exact(value, expected) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _tree_is_exact(value[key], expected[key]) for key in expected
        )
    return value == expected


def runtime_stack_is_exact(value) -> bool:
    return _tree_is_exact(value, EXACT_RUNTIME_STACK)


def storage_schema_is_exact(torch_module, *, num_steps: int, num_envs: int,
                            device, storage_tensors: dict, storage_dones) -> bool:
    """Validate the pinned RSL3 rollout ABI without copying tensor contents."""
    expected = dict(STORAGE_FLOAT_WIDTHS)
    return (
        type(storage_tensors) is dict
        and set(storage_tensors) == set(expected)
        and all(
            isinstance(storage_tensors[name], torch_module.Tensor)
            and torch_module.is_floating_point(storage_tensors[name])
            and storage_tensors[name].device == device
            and tuple(storage_tensors[name].shape)
            == (num_steps, num_envs, width)
            and storage_tensors[name].is_contiguous()
            for name, width in STORAGE_FLOAT_WIDTHS
        )
        and isinstance(storage_dones, torch_module.Tensor)
        and storage_dones.dtype == torch_module.uint8
        and storage_dones.device == device
        and tuple(storage_dones.shape) == (num_steps, num_envs, 1)
        and storage_dones.is_contiguous()
    )


def storage_domain_validity(storage_tensors: dict, storage_dones) -> tuple:
    """Return device predicates in exact ``STORAGE_DOMAIN_NAMES`` order."""
    return (
        (storage_dones.eq(0) | storage_dones.eq(1)).all(),
        storage_tensors["sigma"].gt(0).all(),
    )


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
                    "source_commit", "run_namespace", "runtime_stack",
                    "plant_model",
                }
                or type(run_identity.get("source_commit")) is not str
                or re.fullmatch(r"[0-9a-f]{40}", run_identity["source_commit"]) is None
                or type(run_identity.get("run_namespace")) is not str
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}",
                    run_identity["run_namespace"],
                ) is None
                or not runtime_stack_is_exact(run_identity.get("runtime_stack"))
                or not _plant_model_is_exact(run_identity.get("plant_model"))):
            raise ValueError("FullMDP run identity differs")
        runtime_stack = run_identity["runtime_stack"]
        plant_model = run_identity["plant_model"]
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
        self.table_bit = termination_bits["robot_hit_table"]
        self.action_uid, self.mount_normal_sign = action_uid, mount_normal_sign
        self.family = family
        self.run_identity = {
            "source_commit": run_identity["source_commit"],
            "run_namespace": run_identity["run_namespace"],
            "runtime_stack": copy.deepcopy(runtime_stack),
            "plant_model": (
                _plant_contract_module().clone_plant_model_identity(plant_model)
            ),
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
        self._paddle_playback_rows = torch.zeros(
            1, dtype=torch.float64, device=device
        )
        # Rows: finite count, raw kernel sum/sumsq, domain-violation count.
        self._paddle_playback_stats = torch.zeros(
            (4, PADDLE_PRIOR_TERM_COUNT), dtype=torch.float64, device=device
        )
        self._paddle_max_payment = torch.tensor(
            [
                weight * FULLMDP_POLICY_STEP_S
                for weight in PADDLE_PRIOR_WEIGHTS
            ],
            dtype=torch.float64,
            device=device,
        )
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
        fact_integrity_bits = self._vector(
            extras, "full_a_fact_integrity_fault_bits", torch.long
        )
        paddle_playback = self._vector(
            extras, "full_a_paddle_prior_playback", torch.bool
        )
        fact_integrity = {
            name: torch.bitwise_and(fact_integrity_bits, bit).ne(0)
            for name, bit in FACT_INTEGRITY_CAUSES
        }
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
        table_bit = torch.bitwise_and(bits, self.table_bit).ne(0)
        timeout_fault = time_outs.ne(timeout_bit) | (
            timeout_bit & bits.ne(self.time_out_bit)
        )
        identity = slot.eq(0) & uid.eq(self.action_uid) & sign.eq(self.mount_normal_sign)
        outcome_event = event["flight_terminal_rows"]
        scheduled_due_partition_fault = (
            event["scheduled_due_rows"].ne(
                event["reveal_due_rows"]
                | event["due_terminal_overlap_rows"]
            )
            | (
                event["reveal_due_rows"]
                & event["due_terminal_overlap_rows"]
            )
        )
        reveal_due_partition_fault = (
            event["reveal_due_rows"].ne(
                event["reveal_rows"] | event["reveal_deferred_rows"]
            )
            | (event["reveal_rows"] & event["reveal_deferred_rows"])
        )
        r03_subset_fault = (
            event["r03_physically_valid_rows"] & ~event["r03_present_rows"]
        )
        r06_subset_fault = (
            (
                event["r06_eligible_rows"]
                & ~event["r06_present_rows"]
            )
            | (event["r06_common_rows"] & ~event["r06_eligible_rows"])
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
                invalid_done, done.ne(terminal_present) | (resolved & ~table_bit),
                time_outs, timeout_fault, selected_reset.ne(done),
                generation_delta, generation_delta.ne(done.to(torch.long)),
                resolved, landing & landing_event, opponent_bound & landing_event,
                status.lt(0) | status.gt(5),
                scheduled_due_partition_fault,
                reveal_due_partition_fault,
                r03_subset_fault,
                r06_subset_fault,
                terms_finite, ~terms_finite, actual_finite, ~actual_finite, ~conserved,
                slot.eq(0), uid.eq(self.action_uid), sign.eq(self.mount_normal_sign),
                identity, outcome.lt(0) | outcome.gt(6), outcome_event & outcome.eq(0),
                ~torch.stack(
                    [phase.eq(index) for index in EXACT_PHASE_CODES]
                ).any(dim=0),
                *(
                    fact_integrity[name]
                    for name, _bit in FACT_INTEGRITY_CAUSES
                ),
                torch.bitwise_and(
                    fact_integrity_bits, ~FACT_INTEGRITY_KNOWN_MASK
                ).ne(0),
            )
        ).sum(1, dtype=torch.float64)
        self._reward += torch.where(torch.isfinite(terms), terms,
                                    torch.zeros_like(terms)).sum(0, dtype=torch.float64)
        paddle_terms = terms[
            :,
            PADDLE_PRIOR_TERM_START : (
                PADDLE_PRIOR_TERM_START + PADDLE_PRIOR_TERM_COUNT
            ),
        ].to(dtype=torch.float64)
        paddle_kernel = paddle_terms / self._paddle_max_payment
        paddle_finite = (
            paddle_playback[:, None]
            & torch.isfinite(paddle_kernel)
        )
        finite_kernel = torch.where(
            paddle_finite, paddle_kernel, torch.zeros_like(paddle_kernel)
        )
        paddle_domain_violation = paddle_finite & (
            paddle_kernel.lt(0.0) | paddle_kernel.gt(1.0)
        )
        self._paddle_playback_rows += paddle_playback.sum(
            dtype=torch.float64
        )
        self._paddle_playback_stats += torch.stack(
            (
                paddle_finite.sum(0, dtype=torch.float64),
                finite_kernel.sum(0),
                torch.square(finite_kernel).sum(0),
                paddle_domain_violation.sum(0, dtype=torch.float64),
            )
        )
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
                storage_tensors: dict, storage_dones, policy_std) -> PreparedUpdate:
        torch = self.torch
        if self._pending is not None or update_index != self._next:
            raise RuntimeError("FullMDP ledger update order differs")
        expected_steps = (update_index + 1) * self.num_steps
        storage_names = tuple(name for name, _width in STORAGE_FLOAT_WIDTHS)
        if not storage_schema_is_exact(
            torch, num_steps=self.num_steps, num_envs=self.num_envs,
            device=self.device, storage_tensors=storage_tensors,
            storage_dones=storage_dones,
        ):
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
        storage_domains = torch.stack(
            storage_domain_validity(storage_tensors, storage_dones)
        ).to(dtype=torch.float64)
        policy_summary = torch.stack((
            (torch.isfinite(policy_std) & policy_std.gt(0)).all().to(torch.float64),
            policy_std.mean().to(torch.float64),
        ))
        host = torch.cat(
            (self._event, self._terminal, self._classification, self._outcome,
             self._phase, self._misc, self._reward, self._actual,
             self._paddle_playback_rows,
             self._paddle_playback_stats.reshape(-1),
             self._episode, storage_finite, storage_domains, policy_summary)
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
        paddle_playback_rows = values[cursor]; cursor += 1
        paddle_stat_width = 4 * PADDLE_PRIOR_TERM_COUNT
        paddle_stats = values[cursor:cursor + paddle_stat_width]
        cursor += paddle_stat_width
        paddle_finite = paddle_stats[:PADDLE_PRIOR_TERM_COUNT]
        paddle_kernel_sum = paddle_stats[
            PADDLE_PRIOR_TERM_COUNT : 2 * PADDLE_PRIOR_TERM_COUNT
        ]
        paddle_kernel_sumsq = paddle_stats[
            2 * PADDLE_PRIOR_TERM_COUNT : 3 * PADDLE_PRIOR_TERM_COUNT
        ]
        paddle_domain_violation = paddle_stats[
            3 * PADDLE_PRIOR_TERM_COUNT : 4 * PADDLE_PRIOR_TERM_COUNT
        ]
        episode_values = values[cursor:cursor + 3]; cursor += 3
        storage_values = values[cursor:cursor + len(storage_names)]
        cursor += len(storage_names)
        storage_domain_values = values[cursor:cursor + len(STORAGE_DOMAIN_NAMES)]
        cursor += len(STORAGE_DOMAIN_NAMES)
        policy_finite, policy_mean_std = values[cursor:cursor + 2]
        nonfinite_storage = [name for name, value in zip(
            storage_names, storage_values
        ) if value != 1.0]
        if nonfinite_storage:
            raise RuntimeError(
                "FullMDP rollout storage is nonfinite: "
                + ",".join(nonfinite_storage)
            )
        invalid_domains = [name for name, value in zip(
            STORAGE_DOMAIN_NAMES, storage_domain_values
        ) if value != 1.0]
        if invalid_domains:
            raise RuntimeError(
                "FullMDP rollout storage domain differs: "
                + ",".join(invalid_domains)
            )
        if policy_finite != 1.0 or not math.isfinite(policy_mean_std) or policy_mean_std <= 0:
            raise RuntimeError("FullMDP policy std is nonfinite or nonpositive")
        events = {name: int(value) for (name, _key), value
                  in zip(EVENT_FIELDS, event_values)}
        terminal = {name: int(value) for (name, _bit), value
                    in zip(self.term_bits, terminal_values)}
        misc = {name: int(value) for name, value in zip(_MISC_NAMES, misc_values)}
        transitions = self.num_envs * self.num_steps
        bad = [name for name in _ZERO_FAULTS if misc[name] != 0]
        bad_events = [name for name in (
            "missed_launch_rows", "recovery_completion_fault_rows",
        ) if events[name] != 0]
        if bad or bad_events or any(misc[name] != transitions for name in (
            "reward_terms_finite_rows", "actual_reward_finite_rows",
            "slot0_rows", "uid_rows", "identity_rows",
        )):
            details = (
                [f"{name}={misc[name]}" for name in bad]
                + [f"{name}={events[name]}" for name in bad_events]
            )
            raise RuntimeError(
                "FullMDP ledger rollout evidence differs: "
                + ",".join(details or ["identity_or_finite_rows"])
            )
        family_counts = {"forehand": 0, "backhand": 0}
        family_counts[self.family] = misc["identity_rows"]
        record = {
            "schema_version": SCHEMA_VERSION, "record_type": "mujoco_full_mdp_update_ack",
            "diagnostic_unauthorized": True, "update_index": update_index,
            "run_identity": copy.deepcopy(self.run_identity),
            "num_envs": self.num_envs, "num_steps_per_env": self.num_steps,
            "transitions_delta": transitions, "transitions_cumulative": transitions * (update_index + 1),
            "environment_steps_delta": self.num_steps,
            "environment_steps_cumulative": environment_steps,
            "storage_finite": {name: bool(value) for name, value
                               in zip(storage_names, storage_values)},
            "storage_domains": {name: bool(value) for name, value
                                in zip(STORAGE_DOMAIN_NAMES, storage_domain_values)},
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
            "lifecycle_counts": {
                name: misc[name] for name in LIFECYCLE_COUNT_NAMES
            },
            "fact_integrity_counts": {
                name: misc[name] for name in FACT_INTEGRITY_COUNT_NAMES
            },
            "reward_graph": {
                "term_names": list(REWARD_TERM_NAMES),
                "term_count": REWARD_TERM_COUNT,
                "term_sums": reward_values,
                "actual_reward_sum": actual_sum,
                **{name: misc[name] for name in ("reward_terms_finite_rows",
                    "reward_terms_nonfinite_rows", "actual_reward_finite_rows",
                    "actual_reward_nonfinite_rows", "conservation_fault_rows")},
                "playback_paddle_prior": {
                    "term_names": list(PADDLE_PRIOR_TERM_NAMES),
                    "row_count": int(paddle_playback_rows),
                    "finite_rows": [int(value) for value in paddle_finite],
                    "kernel_sum": paddle_kernel_sum,
                    "kernel_sumsq": paddle_kernel_sumsq,
                    "domain_violation_rows": [
                        int(value) for value in paddle_domain_violation
                    ],
                },
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
            self._paddle_playback_rows, self._paddle_playback_stats,
        ):
            tensor.zero_()
        self._steps, self._faults, self._pending = 0, set(), None
        self._next += 1
        return payload
