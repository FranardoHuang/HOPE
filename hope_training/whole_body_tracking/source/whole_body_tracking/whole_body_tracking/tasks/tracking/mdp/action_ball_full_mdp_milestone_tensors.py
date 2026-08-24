"""Small payment, actual-return and once-only event reductions in Epoch drain."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    from . import action_ball_full_mdp_lean_checkpoint_txn as carry_txn
except ImportError:
    import action_ball_full_mdp_lean_checkpoint_txn as carry_txn

try:
    from . import action_ball_full_mdp_reward_contract as reward_contract
except ImportError:
    import action_ball_full_mdp_reward_contract as reward_contract

SCHEMA_VERSION = 7
REWARD_TERM_COUNT = reward_contract.REWARD_TERM_COUNT
REWARD_I64_COLUMNS = ("evaluated", "eligible", "finite", "nonzero")
REWARD_F64_COLUMNS = (
    "primitive_sum", "primitive_sum_sq", "payment_raw_sum",
    "configured_income_sum", "configured_income_sum_sq",
    "configured_positive_income_sum", "configured_negative_income_abs_sum",
)
ACTUAL_I64_NAMES = (
    "sample_count", "finite_count", "nonfinite_count",
    "conservation_violation_count",
)
ACTUAL_F64_NAMES = (
    "sum", "sum_sq", "residual_sum", "residual_abs_sum",
    "residual_sq_sum", "residual_max_abs", "tolerance_max",
)
EVENT_SPECS = (
    ("d05_due", "env_slot_row_per_D05_SETTLED", "ActionEpoch.settle_d05_transaction", "due_mask_key_may_be_invalid_for_censor"),
    ("d05_selected_opportunity", "env_slot_row_per_D05_SETTLED", "ActionEpoch.settle_d05_transaction", "selected_construct_slot"),
    ("d05_construction_admitted", "env_slot_row_per_D05_SETTLED", "ActionEpoch.settle_d05_transaction", "selected_and_construction_admissible_and_fault_free"),
    ("d05_key_admitted", "full_key_row_per_D05_SETTLED", "ActionEpoch.settle_d05_transaction", "construction_admitted_and_full_key_valid"),
    ("r03_first_physically_valid", "full_key_row_first_sticky_publish", "ActionEpoch.publish_owner_facts:r03_strike_fact", "PRESENT_and_PHYSICALLY_VALID_and_owner_fault_free"),
    ("physical_first_observed", "full_key_row_0_to_1", "ActionEpoch.refresh_physical_postphysics_rows", "PRESENT_0_to_1"),
    ("physical_first_contact", "full_key_row_0_to_1", "ActionEpoch.refresh_physical_postphysics_rows", "SELECTED_CONTACT_0_to_1"),
    ("r06_settled", "full_key_row_per_R06_OUTCOME_ROWS", "ActionEpoch.refresh_r06_outcome_rows", "unique_phase_gated_join"),
    ("r06_contact_valid", "full_key_row_at_unique_settlement", "ActionEpoch.refresh_r06_outcome_rows", "CONTACT_VALID"),
    ("r06_net_crossed", "full_key_row_at_unique_settlement", "ActionEpoch.refresh_r06_outcome_rows", "NET_CROSSED"),
    ("r06_net_clear", "full_key_row_at_unique_settlement", "ActionEpoch.refresh_r06_outcome_rows", "NET_CROSSED_and_NET_CLEAR"),
    ("r06_crossing_valid", "full_key_row_at_unique_settlement", "ActionEpoch.refresh_r06_outcome_rows", "CROSSING_VALID"),
    ("r06_on_opponent_table", "full_key_row_at_unique_settlement", "ActionEpoch.refresh_r06_outcome_rows", "ON_OPPONENT_TABLE"),
    ("r06_common_legal", "full_key_row_at_unique_settlement", "ActionEpoch.refresh_r06_outcome_rows", "COMMON_ON_TABLE"),
    ("r07_recovery_outcome_first_valid", "full_key_row_first_sticky_dense_publish", "ActionEpoch.publish_owner_facts:r07_recovery", "PRESENT_and_NUMERICALLY_VALID_and_reward_eligible_and_facts_valid_and_no_infrastructure_or_owner_fault"),
    ("r07_recovery_first_ready", "full_key_row_per_monotonic_first_ready", "ContinuousRecoveryDeviceCoordinator._publish_action_epoch_motion_readiness", "ready_live_and_first_ready_source_step_unset"),
)
EVENT_NAMES = tuple(spec[0] for spec in EVENT_SPECS)
EPISODE_I64_NAMES = (
    "completed", "length_sum", "reason_time_out", "reason_base_fell_tilt",
    "reason_base_too_low", "reason_joint_qdes_forbidden",
    "reason_robot_hit_table",
)
EPISODE_F64_NAMES = ("return_sum", "return_sum_sq")
PADDLE_PLAYBACK_I64_COLUMNS = (
    "telemetry_unavailable_count",
    "playback_count",
    "finite_count",
    "domain_violation_count",
)
PADDLE_PLAYBACK_F64_COLUMNS = ("kernel_sum", "kernel_sum_sq")
PADDLE_PLAYBACK_TERM_NAMES = reward_contract.PADDLE_MOTION_PRIOR_NAMES
NOT_PRODUCED = (
    "per_action_event_strata", "per_side_event_strata",
    "landing_xy_event_statistics", "recovery_success_event",
    "recovery_component_event_statistics", "all_policy_balance",
    "outgoing_v_plus_w_plus", "action_rate_l2", "multi_action_side_strata",
    "shot_attributed_reward",
)
FLOAT32_REWARD_ACCUMULATION_TOLERANCE_FACTOR = 64.0
_RI = REWARD_TERM_COUNT * len(REWARD_I64_COLUMNS)
_RF = REWARD_TERM_COUNT * len(REWARD_F64_COLUMNS)
_AI = _RI
_EI = _AI + len(ACTUAL_I64_NAMES)
_EPI = _EI + len(EVENT_NAMES)
_AF = _RF
_EPF = _AF + len(ACTUAL_F64_NAMES)
_PI = _EPI + len(EPISODE_I64_NAMES)
_PF = _EPF + len(EPISODE_F64_NAMES)
I64_NUMEL = _PI + len(PADDLE_PLAYBACK_TERM_NAMES) * len(
    PADDLE_PLAYBACK_I64_COLUMNS
)
F64_NUMEL = _PF + len(PADDLE_PLAYBACK_TERM_NAMES) * len(
    PADDLE_PLAYBACK_F64_COLUMNS
)


def _configured_totals(floats: tuple[float, ...]) -> tuple[float, float]:
    signed = sum(floats[index * 7 + 3] for index in range(REWARD_TERM_COUNT))
    absolute = sum(
        floats[index * 7 + 5] + floats[index * 7 + 6]
        for index in range(REWARD_TERM_COUNT)
    )
    return signed, absolute


@dataclass(frozen=True)
class MilestoneWindowTelemetry:
    i64: tuple[int, ...]
    f64: tuple[float, ...]

    def as_json(self, term_names: tuple[str, ...]) -> dict[str, object]:
        if len(term_names) != REWARD_TERM_COUNT:
            raise ValueError("milestone reward term names differ")
        reward_i, reward_f = self.i64[:_RI], self.f64[:_RF]
        rewards = []
        for ordinal, name in enumerate(term_names):
            i0, f0 = ordinal * 4, ordinal * 7
            rewards.append({
                "ordinal": ordinal, "term": name,
                **dict(zip(REWARD_I64_COLUMNS, reward_i[i0:i0 + 4])),
                **dict(zip(REWARD_F64_COLUMNS, reward_f[f0:f0 + 7])),
            })
        configured_sum, configured_abs_sum = _configured_totals(self.f64)
        paddle_playback = []
        for index, name in enumerate(PADDLE_PLAYBACK_TERM_NAMES):
            i0 = _PI + index * len(PADDLE_PLAYBACK_I64_COLUMNS)
            f0 = _PF + index * len(PADDLE_PLAYBACK_F64_COLUMNS)
            paddle_playback.append(
                {
                    "ordinal": (
                        REWARD_TERM_COUNT
                        - len(PADDLE_PLAYBACK_TERM_NAMES)
                        + index
                    ),
                    "term": name,
                    **dict(
                        zip(
                            PADDLE_PLAYBACK_I64_COLUMNS,
                            self.i64[
                                i0 : i0 + len(PADDLE_PLAYBACK_I64_COLUMNS)
                            ],
                        )
                    ),
                    **dict(
                        zip(
                            PADDLE_PLAYBACK_F64_COLUMNS,
                            self.f64[
                                f0 : f0 + len(PADDLE_PLAYBACK_F64_COLUMNS)
                            ],
                        )
                    ),
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_unit": "reward_manager_payment_sample",
            "reward_terms": rewards,
            "actual_reward": {
                **dict(zip(ACTUAL_I64_NAMES, self.i64[_AI:_EI])),
                **dict(zip(ACTUAL_F64_NAMES, self.f64[_AF:_EPF])),
                "configured_income_sum": configured_sum,
                "configured_income_abs_sum": configured_abs_sum,
                "conservation_unit": "per_env_control_step",
            },
            "event_ladder": {
                "scope": "aggregate_current_capacity_rows_not_action_or_side_strata",
                "events": [
                    {
                        "event": name, "count": count, "unit": unit,
                        "producer": producer, "predicate": predicate,
                    }
                    for (name, unit, producer, predicate), count in zip(
                        EVENT_SPECS, self.i64[_EI:_EPI]
                    )
                ],
            },
            "episodes": {
                **dict(zip(EPISODE_I64_NAMES, self.i64[_EPI:_PI])),
                **dict(zip(EPISODE_F64_NAMES, self.f64[_EPF:_PF])),
            },
            "paddle_motion_prior_playback": {
                "sample_unit": "motion_playback_active_reward_row",
                "predicate": (
                    "Motion.action_ball_full_mdp_playback_active_mask"
                ),
                "terms": paddle_playback,
            },
            "not_produced": NOT_PRODUCED,
        }

def decode_host_window(i64: torch.Tensor, f64: torch.Tensor) -> MilestoneWindowTelemetry:
    if (
        type(i64) is not torch.Tensor or i64.device.type != "cpu"
        or i64.dtype is not torch.int64 or tuple(i64.shape) != (I64_NUMEL,)
        or not i64.is_contiguous() or type(f64) is not torch.Tensor
        or f64.device.type != "cpu" or f64.dtype is not torch.float64
        or tuple(f64.shape) != (F64_NUMEL,) or not f64.is_contiguous()
    ):
        raise ValueError("milestone host tensor ABI differs")
    ints, floats = tuple(i64.tolist()), tuple(f64.tolist())
    if any(value < 0 for value in ints) or any(not math.isfinite(v) for v in floats):
        raise ValueError("milestone host tensor values differ")
    for ordinal in range(REWARD_TERM_COUNT):
        offset = ordinal * len(REWARD_F64_COLUMNS)
        if any(floats[offset + column] < 0.0 for column in (1, 4, 5, 6)):
            raise ValueError("milestone nonnegative telemetry differs")
    if any(floats[_AF + column] < 0.0 for column in (1, 3, 4, 5, 6)):
        raise ValueError("milestone nonnegative telemetry differs")
    if floats[_EPF + 1] < 0.0:
        raise ValueError("milestone nonnegative telemetry differs")
    for index in range(len(PADDLE_PLAYBACK_TERM_NAMES)):
        i0 = _PI + index * len(PADDLE_PLAYBACK_I64_COLUMNS)
        f0 = _PF + index * len(PADDLE_PLAYBACK_F64_COLUMNS)
        (
            _telemetry_unavailable_count,
            playback_count,
            finite_count,
            domain_violation_count,
        ) = ints[
            i0 : i0 + len(PADDLE_PLAYBACK_I64_COLUMNS)
        ]
        kernel_sum_sq = floats[f0 + 1]
        if (
            finite_count > playback_count
            or domain_violation_count > finite_count
            or kernel_sum_sq < 0.0
        ):
            raise ValueError("milestone paddle playback telemetry differs")
    nonfinite_count, violations = ints[_AI + 2:_EI]
    if nonfinite_count != 0 or violations != 0:
        raise ValueError("milestone actual reward conservation differs")
    return MilestoneWindowTelemetry(ints, floats)


class MilestoneTensorAccumulator:
    """Fixed-size reductions; Epoch owns freeze, transfer and ACK."""

    __slots__ = (
        "i64", "f64", "open_episode_return", "open_step_configured_income",
        "open_step_configured_abs_income", "r03_seen", "r07_outcome_seen",
        "r07_ready_seen",
        "_num_envs", "_shot_shape", "_device",
        "_scratch_open", "_frozen", "_lean_carry_coordinator",
    )

    def __init__(
        self, num_envs: int, device: torch.device, shot_slot_capacity: int = 1
    ) -> None:
        self._num_envs = num_envs
        self._shot_shape = (num_envs, shot_slot_capacity)
        self._device = torch.device(device)
        with torch.inference_mode(False):
            self.i64 = torch.zeros(I64_NUMEL, dtype=torch.int64, device=self._device)
            self.f64 = torch.zeros(F64_NUMEL, dtype=torch.float64, device=self._device)
            self.open_episode_return = torch.zeros(num_envs, dtype=torch.float64, device=device)
            self.open_step_configured_income = torch.zeros_like(self.open_episode_return)
            self.open_step_configured_abs_income = torch.zeros_like(self.open_episode_return)
            self.r03_seen = torch.zeros(
                self._shot_shape, dtype=torch.bool, device=self._device
            )
            self.r07_outcome_seen = torch.zeros_like(self.r03_seen)
            self.r07_ready_seen = torch.zeros_like(self.r03_seen)
        self._scratch_open = False
        self._frozen = False
        self._lean_carry_coordinator = None

    def _writable(self) -> None:
        carry_txn._require_leaf_mutable(self)
        if self._frozen:
            raise RuntimeError("milestone window is frozen for drain")

    def add_reward(self, ordinal, primitive, payment, eligible, finite, configured_scale) -> None:
        self._writable()
        if ordinal == 0:
            if self._scratch_open:
                raise RuntimeError("milestone configured-income step is still open")
            self._scratch_open = True
        elif not self._scratch_open:
            raise RuntimeError("milestone configured-income step was not opened")
        valid = eligible & finite
        i0, f0 = ordinal * 4, ordinal * 7
        self.i64[i0:i0 + 4].add_(torch.stack((
            torch.full((), primitive.numel(), dtype=torch.int64, device=primitive.device),
            eligible.sum(dtype=torch.int64), valid.sum(dtype=torch.int64),
            (valid & payment.ne(0)).sum(dtype=torch.int64),
        )))
        p = torch.where(valid, primitive, 0).to(torch.float64)
        q = torch.where(valid, payment, 0).to(torch.float64)
        income = q * configured_scale
        self.open_step_configured_income.add_(income)
        self.open_step_configured_abs_income.add_(income.abs())
        self.f64[f0:f0 + 7].add_(torch.stack((
            p.sum(), p.square().sum(), q.sum(), income.sum(), income.square().sum(),
            income.clamp_min(0).sum(), (-income.clamp_max(0)).sum(),
        )))

    def add_paddle_motion_prior_playback(
        self,
        ordinal: int,
        kernel: torch.Tensor,
        playback_active: torch.Tensor,
    ) -> None:
        """Reduce raw motion-prior kernels on Motion-owned playback rows.

        A finite zero kernel is a valid sample.  Values outside the analytic
        Cauchy range are counted for diagnosis but remain telemetry: this leaf
        neither clamps them nor turns that same-writer relationship into a
        training gate.
        """

        self._writable()
        first = REWARD_TERM_COUNT - len(PADDLE_PLAYBACK_TERM_NAMES)
        if type(ordinal) is not int or not first <= ordinal < REWARD_TERM_COUNT:
            raise RuntimeError("milestone paddle playback ordinal differs")
        if (
            type(kernel) is not torch.Tensor
            or type(playback_active) is not torch.Tensor
            or tuple(kernel.shape) != (self._num_envs,)
            or playback_active.shape != kernel.shape
            or kernel.device != self._device
            or playback_active.device != self._device
            or kernel.dtype is not torch.float32
            or playback_active.dtype != torch.bool
        ):
            raise RuntimeError("milestone paddle playback tensor ABI differs")
        finite = playback_active & torch.isfinite(kernel)
        domain_violation = finite & (
            kernel.lt(0.0) | kernel.gt(1.0)
        )
        index = ordinal - first
        i0 = _PI + index * len(PADDLE_PLAYBACK_I64_COLUMNS)
        f0 = _PF + index * len(PADDLE_PLAYBACK_F64_COLUMNS)
        self.i64[i0 : i0 + len(PADDLE_PLAYBACK_I64_COLUMNS)].add_(
            torch.stack(
                (
                    torch.zeros((), dtype=torch.int64, device=kernel.device),
                    playback_active.sum(dtype=torch.int64),
                    finite.sum(dtype=torch.int64),
                    domain_violation.sum(dtype=torch.int64),
                )
            )
        )
        clean_kernel = torch.where(finite, kernel, 0.0).to(torch.float64)
        self.f64[f0 : f0 + len(PADDLE_PLAYBACK_F64_COLUMNS)].add_(
            torch.stack(
                (
                    clean_kernel.sum(),
                    clean_kernel.square().sum(),
                )
            )
        )

    def add_paddle_motion_prior_unavailable(self, ordinal: int) -> None:
        """Record missing optional telemetry without changing training flow."""

        self._writable()
        first = REWARD_TERM_COUNT - len(PADDLE_PLAYBACK_TERM_NAMES)
        if type(ordinal) is not int or not first <= ordinal < REWARD_TERM_COUNT:
            raise RuntimeError("milestone paddle playback ordinal differs")
        index = ordinal - first
        i0 = _PI + index * len(PADDLE_PLAYBACK_I64_COLUMNS)
        self.i64[i0].add_(self._num_envs)

    def close_actual_step(self, reward: torch.Tensor) -> None:
        self._writable()
        if not self._scratch_open:
            raise RuntimeError("milestone actual reward step is not open")
        if (
            type(reward) is not torch.Tensor
            or reward.device != self.open_step_configured_income.device
            or tuple(reward.shape) != tuple(self.open_step_configured_income.shape)
            or reward.dtype is not torch.float32
        ):
            raise RuntimeError("milestone actual reward tensor ABI differs")
        actual = reward.to(torch.float64)
        finite = (
            torch.isfinite(actual)
            & torch.isfinite(self.open_step_configured_income)
            & torch.isfinite(self.open_step_configured_abs_income)
        )
        clean_actual = torch.where(finite, actual, 0)
        residual = torch.where(finite, actual - self.open_step_configured_income, 0)
        residual_abs = residual.abs()
        tolerance = (
            self.open_step_configured_abs_income
            * (
                FLOAT32_REWARD_ACCUMULATION_TOLERANCE_FACTOR
                * torch.finfo(torch.float32).eps
            )
            + REWARD_TERM_COUNT * torch.finfo(torch.float32).tiny
        )
        self.i64[_AI:_EI].add_(torch.stack((
            torch.full((), reward.numel(), dtype=torch.int64, device=reward.device),
            finite.sum(dtype=torch.int64), (~finite).sum(dtype=torch.int64),
            (finite & residual_abs.gt(tolerance)).sum(dtype=torch.int64),
        )))
        self.f64[_AF:_EPF].add_(torch.stack((
            clean_actual.sum(), clean_actual.square().sum(), residual.sum(),
            residual_abs.sum(), residual.square().sum(), residual_abs.amax(),
            tolerance.amax(),
        )))
        self.open_step_configured_income.zero_()
        self.open_step_configured_abs_income.zero_()
        self._scratch_open = False

    def _add_events(self, offset: int, masks: tuple[torch.Tensor, ...]) -> None:
        self._writable()
        self.i64[_EI + offset:_EI + offset + len(masks)].add_(torch.stack(
            tuple(mask.sum(dtype=torch.int64) for mask in masks)
        ))

    def add_d05_events(self, due, opportunity, construction, key_admitted) -> None:
        self._add_events(0, (due, opportunity, construction, key_admitted))

    def add_first_fact_event(self, owner_kind: str, fully_valid) -> None:
        self._writable()
        if owner_kind not in ("r03_strike_fact", "r07_recovery"):
            raise RuntimeError("milestone first-fact producer differs")
        seen = self.r03_seen if owner_kind == "r03_strike_fact" else self.r07_outcome_seen
        first_valid = fully_valid & ~seen
        seen.bitwise_or_(fully_valid)
        self._add_events(4 if owner_kind == "r03_strike_fact" else 14, (first_valid,))

    def reset_event_rows(self, rows: torch.Tensor) -> None:
        self._writable()
        self.r03_seen.masked_fill_(rows, False)
        self.r07_outcome_seen.masked_fill_(rows, False)
        self.r07_ready_seen.masked_fill_(rows, False)

    def reset_event_envs(self, selected: torch.Tensor) -> None:
        self.reset_event_rows(selected[:, None].expand_as(self.r03_seen))

    def add_r07_first_ready(self, rows: torch.Tensor) -> None:
        self._writable()
        first_ready = rows & ~self.r07_ready_seen
        self.r07_ready_seen.bitwise_or_(rows)
        self._add_events(15, (first_ready,))

    def add_physical_events(self, first_observed, first_contact) -> None:
        self._add_events(5, (first_observed, first_contact))

    def add_r06_events(self, *masks) -> None:
        if len(masks) != 7:
            raise RuntimeError("milestone R06 event tuple differs")
        self._add_events(7, masks)

    def add_step_return(self, reward: torch.Tensor) -> None:
        self._writable()
        self.open_episode_return.add_(reward.to(torch.float64))

    def close_episodes(self, selected, lengths, reason_bits) -> None:
        self._writable()
        returns = torch.where(selected, self.open_episode_return, 0)
        self.i64[_EPI:_EPI + 7].add_(torch.stack((
            selected.sum(dtype=torch.int64), torch.where(selected, lengths, 0).sum(),
            *((selected & reason_bits.bitwise_and(bit).ne(0)).sum(dtype=torch.int64)
              for bit in (1, 2, 4, 8, 16)),
        )))
        self.f64[_EPF:_EPF + 2].add_(torch.stack((returns.sum(), returns.square().sum())))
        self.open_episode_return.masked_fill_(selected, 0)

    def pack_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.i64, self.f64

    def freeze_window_(self) -> None:
        carry_txn._require_leaf_mutable(self)
        if self._frozen:
            raise RuntimeError("milestone window is frozen")
        if self._scratch_open:
            raise RuntimeError("milestone configured-income step is still open")
        self._frozen = True

    def abort_window_(self) -> None:
        carry_txn._require_leaf_mutable(self)
        if not self._frozen:
            raise RuntimeError("milestone window is not frozen")
        self._frozen = False

    def clear_window_(self) -> None:
        carry_txn._require_leaf_mutable(self)
        if not self._frozen:
            raise RuntimeError("milestone window is not frozen")
        self.i64.zero_()
        self.f64.zero_()
        self._frozen = False

    def _checkpoint_device_views(
        self, *, allow_prepared: bool = False
    ) -> tuple[torch.Tensor, ...]:
        del allow_prepared
        if self._frozen or self._scratch_open:
            raise RuntimeError("milestone checkpoint boundary is not quiescent")
        live_specs = (
            (self.i64, torch.int64, (I64_NUMEL,)),
            (self.f64, torch.float64, (F64_NUMEL,)),
            (self.open_step_configured_income, torch.float64, (self._num_envs,)),
            (self.open_step_configured_abs_income, torch.float64, (self._num_envs,)),
            (self.open_episode_return, torch.float64, (self._num_envs,)),
            (self.r03_seen, torch.bool, self._shot_shape),
            (self.r07_outcome_seen, torch.bool, self._shot_shape),
            (self.r07_ready_seen, torch.bool, self._shot_shape),
        )
        if any(
            type(value) is not torch.Tensor
            or value.device != self._device
            or value.dtype is not dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            for value, dtype, shape in live_specs
        ):
            raise RuntimeError("milestone live checkpoint tensor ABI differs")
        views = tuple(value for value, _dtype, _shape in live_specs)
        occupied: list[tuple[int, int, int]] = []
        for value in views:
            pointer = value.untyped_storage().data_ptr()
            start = value.storage_offset() * value.element_size()
            end = start + value.numel() * value.element_size()
            if any(
                pointer == prior_pointer
                and start < prior_end
                and prior_start < end
                for prior_pointer, prior_start, prior_end in occupied
            ):
                raise RuntimeError("milestone live checkpoint tensors alias")
            occupied.append((pointer, start, end))
        return views

    def _checkpoint_carry_from_host_views(
        self, values: object, *, dormant: bool
    ) -> None:
        if type(values) is not tuple or len(values) != 8:
            raise RuntimeError("milestone checkpoint host image differs")
        host_specs = (
            (torch.int64, (I64_NUMEL,)),
            (torch.float64, (F64_NUMEL,)),
            (torch.float64, (self._num_envs,)),
            (torch.float64, (self._num_envs,)),
            (torch.float64, (self._num_envs,)),
            (torch.bool, self._shot_shape),
            (torch.bool, self._shot_shape),
            (torch.bool, self._shot_shape),
        )
        if any(
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype is not dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            for value, (dtype, shape) in zip(values, host_specs)
        ):
            raise RuntimeError("milestone checkpoint host tensor ABI differs")
        (
            window_i64,
            window_f64,
            configured,
            configured_abs,
            open_return,
            r03_seen,
            r07_outcome_seen,
            r07_ready_seen,
        ) = values
        if bool(window_i64.any()) or bool(window_f64.ne(0).any()):
            raise RuntimeError("milestone checkpoint window was not ACK-cleared")
        if bool(configured.ne(0).any()) or bool(configured_abs.ne(0).any()):
            raise RuntimeError("milestone configured-income scratch is not closed")
        if not bool(torch.isfinite(open_return).all()):
            raise RuntimeError("milestone open episode return is non-finite")
        if dormant and (
            bool(open_return.ne(0).any())
            or bool(r03_seen.any())
            or bool(r07_outcome_seen.any())
            or bool(r07_ready_seen.any())
        ):
            raise RuntimeError("milestone restore target is not dormant")
    def _lean_carry_schema(self) -> carry_txn._LeanCarrySchema:
        return carry_txn._LeanCarrySchema(
            "milestone", (), (
                carry_txn._LeanCarryTensorSpec(
                    "acked_window_i64", (I64_NUMEL,), torch.int64, "attest"
                ),
                carry_txn._LeanCarryTensorSpec(
                    "acked_window_f64", (F64_NUMEL,), torch.float64, "attest"
                ),
                carry_txn._LeanCarryTensorSpec(
                    "closed_configured_income", (self._num_envs,),
                    torch.float64, "attest",
                ),
                carry_txn._LeanCarryTensorSpec(
                    "closed_configured_abs_income", (self._num_envs,),
                    torch.float64, "attest",
                ),
                carry_txn._LeanCarryTensorSpec(
                    "open_episode_return", (self._num_envs,), torch.float64
                ),
                carry_txn._LeanCarryTensorSpec("r03_seen", self._shot_shape, torch.bool),
                carry_txn._LeanCarryTensorSpec(
                    "r07_outcome_seen", self._shot_shape, torch.bool
                ),
                carry_txn._LeanCarryTensorSpec(
                    "r07_ready_seen", self._shot_shape, torch.bool
                ),
            ),
        )

    def _lean_carry_construction_views(self):
        return self._checkpoint_device_views()

    def _lean_carry_capture(self, lease):
        if getattr(lease, "coordinator", None) is not self._lean_carry_coordinator:
            raise RuntimeError("milestone carry lease differs")
        views = self._checkpoint_device_views()
        return carry_txn._LeanCarryCapture((), views)

    def _lean_carry_stage(self, lease, scalars, host_tensors):
        if scalars != () or getattr(lease, "coordinator", None) is not self._lean_carry_coordinator:
            raise RuntimeError("milestone carry stage differs")
        self._checkpoint_carry_from_host_views(host_tensors, dormant=False)
        targets = self._checkpoint_device_views()
        if any(bool(value.any()) for value in targets):
            raise RuntimeError("milestone carry target is not dormant")
        # Attested fields retain independent source after-images: central can
        # compare them with the fresh target, but never generically copies them.
        staging = tuple(
            value.to(device=self._device, copy=True).contiguous()
            for value in host_tensors
        )
        return carry_txn._LeanCarryStage((), staging, targets)

    def _lean_carry_target_views(self, lease, stage):
        if lease is not self._lean_carry_coordinator._active_lease:
            raise RuntimeError("milestone carry target lease differs")
        return self._lean_carry_construction_views()

    def _lean_carry_apply_scalars(self, lease, stage) -> None:
        if not stage.commit_started or lease is not self._lean_carry_coordinator._active_lease:
            raise RuntimeError("milestone carry commit was not armed")
