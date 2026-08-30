"""Focused mechanics for Racket's row-wise D05 ACCEPT writer.

These are structural/fixed-tape tests, not a safety authorization.  The real
writer consumes an opaque D05 transaction and a full-N typed after-image; the
fixtures below only make that exact call stack reachable so partial/no-op row
semantics can be checked directly.
"""

from __future__ import annotations

import ast
from dataclasses import fields
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

import test_action_ball_motion_rowwise_due_closure as motion_close_test
import test_action_ball_continuous_racket_device_reveal_hold as racket_test
import test_action_ball_continuous_runtime_transaction_device as d05_test


HC = racket_test.HC
device_r05 = racket_test.device_r05
epoch = racket_test.epoch
physical = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_physical_flight_device"
)
landing = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_landing_outcome_device"
)


_FLOAT_WIDTHS = {
    "racket_target_pos_w": 3,
    "racket_target_vel_w": 3,
    "racket_target_normal_w": 3,
    "target_normal_cmd": 3,
    "_action_ball_ball_contact_target_w": 3,
    "_action_ball_face_center_velocity_target_w": 3,
    "_action_ball_racket_command_quat_w": 4,
    "base_target_pos_w": 2,
    "vb_vel_in_w": 3,
    "vb_spin_in_w": 3,
    "_vb_target_xy_per_env": 2,
}

_KEY_DESTINATIONS = {
    "reset_generation": "_action_ball_reset_generation",
    "ball_generation": "_action_ball_full_mdp_racket_ball_generation",
    "action_uid": "_action_ball_action_uid",
    "action_slot": "_action_ball_action_slot",
    "shot_index": "_action_ball_full_mdp_racket_outcome_shot_index",
    "task_identity": "_action_ball_full_mdp_racket_task_identity",
    "outcome_identity": "_action_ball_full_mdp_racket_outcome_identity",
    "ball_identity": "_action_ball_full_mdp_racket_ball_identity",
}

_WRITTEN_FIELDS = (
    *_FLOAT_WIDTHS,
    "time_to_strike",
    "pre_strike",
    "strike_window",
    "strike_window_pos",
    "strike_window_wide",
    "_action_ball_task_valid",
    "_action_ball_attempt_active",
    "_action_ball_attempt_action",
    "_action_ball_attempt_legal",
    "_action_ball_attempt_hit",
    "_action_ball_reset_generation",
    "_action_ball_swing_generation",
    "_action_ball_action_uid",
    "_action_ball_action_slot",
    "_action_ball_continuous_racket_observation_scheduled_ordinal",
    "_action_ball_full_mdp_racket_ball_generation",
    "_action_ball_full_mdp_racket_outcome_shot_index",
    "_action_ball_full_mdp_racket_task_identity",
    "_action_ball_full_mdp_racket_outcome_identity",
    "_action_ball_full_mdp_racket_ball_identity",
    "_counter_rally_reward_terms",
    "_counter_rally_accepted",
    "_counter_rally_legal_first_landing",
    "_counter_rally_primary_reason_code",
    "_action_ball_prev_contact_valid",
)


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _racket_and_sources(n: int, *, runtime_device: str, monkeypatch):
    device = _device(runtime_device)
    harness = d05_test._harness(n, device=str(device))
    epoch_owner = epoch.ActionEpochOwner(
        num_envs=n,
        device=device,
        shot_slot_capacity=1,
        initial_reset_generation=torch.ones(n, dtype=torch.int64, device=device),
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(n, dtype=torch.bool, device=device),
        reset_generation=torch.ones(n, dtype=torch.int64, device=device),
    )
    harness.owner._diagnostic_epoch_owner = epoch_owner

    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = n
    racket.device = device
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_continuous_fresh_racket_lane_bound = False
    for name, width in _FLOAT_WIDTHS.items():
        setattr(racket, name, torch.zeros(n, width, dtype=torch.float32, device=device))
    racket.time_to_strike = torch.zeros(n, dtype=torch.float32, device=device)
    racket.time_left = torch.zeros_like(racket.time_to_strike)
    racket.pre_strike = torch.zeros(n, dtype=torch.bool, device=device)
    racket.strike_window = torch.zeros_like(racket.pre_strike)
    racket.strike_window_pos = torch.zeros_like(racket.pre_strike)
    racket.strike_window_wide = torch.zeros_like(racket.pre_strike)
    racket._counter_rally_reward_terms = torch.zeros(
        n, 5, dtype=torch.float32, device=device
    )
    racket._counter_rally_accepted = torch.zeros_like(racket.pre_strike)
    racket._counter_rally_legal_first_landing = torch.zeros_like(racket.pre_strike)
    racket._counter_rally_primary_reason_code = torch.full(
        (n,), -1, dtype=torch.int64, device=device
    )
    racket._action_ball_prev_contact_valid = torch.zeros_like(racket.pre_strike)
    racket.cfg = SimpleNamespace(
        strike_window_s=0.1,
        strike_window_pos_s=0.05,
        strike_window_wide_s=0.2,
    )
    origins = torch.arange(
        n * 3, dtype=torch.float32, device=device
    ).reshape(n, 3).contiguous()
    racket._env = SimpleNamespace(
        common_step_counter=4,
        scene=SimpleNamespace(env_origins=origins),
    )
    racket.bind_action_ball_full_mdp_racket_epoch_sources(
        harness.owner, epoch_owner
    )

    class _MotionLeaf:
        def commit_action_ball_full_mdp_motion_epoch_rows(self, token):
            harness.owner.require_owned_action_epoch_accepted(
                token, owner_kind="motion"
            )

    class _PhysicalLeaf:
        def retain_action_epoch_launch(self, token):
            harness.owner.require_owned_action_epoch_accepted(
                token, owner_kind="physical_ball"
            )

    def finish_focused_row_transaction(
        owner, record, *, accepted, retain_physical=False
    ):
        """End the mechanics-only row without publishing D05 live state."""

        assert owner is harness.owner
        assert tuple(accepted.shape) == (n,)
        assert retain_physical is True
        owner._diagnostic_physical_owner.retain_action_epoch_launch(
            record.capability
        )
        assert "physical_ball" in record.accepted_consumers
        record.stage = "settled"
        owner._active_row_transaction = None

    monkeypatch.setattr(
        device_r05.DeviceR05Owner,
        "_finish_row_transaction",
        finish_focused_row_transaction,
    )
    harness.owner._diagnostic_motion_owner = _MotionLeaf()
    harness.owner._diagnostic_racket_owner = racket
    harness.owner._diagnostic_physical_owner = _PhysicalLeaf()
    epoch_owner.bind_d05_accept_writers(
        motion_write=harness.owner._commit_action_epoch_motion_write,
        racket_write=harness.owner._commit_action_epoch_racket_write,
        r05_write=harness.owner._commit_action_epoch_r05_write,
    )
    return racket, harness.owner, epoch_owner


def _candidate(n: int, device: torch.device):
    shape = (n, 1)
    row = torch.arange(n, dtype=torch.int64, device=device).reshape(n, 1)
    key = epoch.row_identity.ActionEpochShotKey(
        reset_generation=(row + 1).contiguous(),
        ball_generation=(row + 101).contiguous(),
        action_uid=torch.full_like(row, 201).contiguous(),
        action_slot=torch.zeros_like(row).contiguous(),
        shot_index=(row + 301).contiguous(),
        task_identity=(row + 401).contiguous(),
        outcome_identity=(row + 501).contiguous(),
        ball_identity=(row + 601).contiguous(),
    )
    identity = epoch.EpochIdentityPayload(
        shot_key=key,
        scheduled_ordinal=(row + 701).contiguous(),
        target_generation=(row + 801).contiguous(),
        selected_cell=row.remainder(3).contiguous(),
        candidate_identity=(row + 901).contiguous(),
    )
    clocks = epoch.EpochClockPayload(
        reveal_tick=(row + 1_001).contiguous(),
        contact_tick=(row + 1_011).contiguous(),
        launch_tick=(row + 1_002).contiguous(),
        deadline_tick=(row + 1_021).contiguous(),
        next_reveal_tick=(row + 1_031).contiguous(),
    )
    task_f32 = torch.arange(
        n * epoch.TASK_F32_WIDTH, dtype=torch.float32, device=device
    ).reshape(n, 1, epoch.TASK_F32_WIDTH)
    task_f32 = (task_f32 * 0.01 + 10.0).contiguous()
    task_f32[:, 0, 0] = torch.linspace(0.02, 0.3, n, device=device)
    task = epoch.EpochTaskPayload(
        task_f32=task_f32,
        task_valid=torch.ones(shape, dtype=torch.bool, device=device),
    )
    return epoch.ActionEpochD05CandidateProjection(
        identity=identity,
        clocks=clocks,
        task=task,
        rng_counter=torch.ones(shape, dtype=torch.int64, device=device),
        construction_admissible=torch.ones(shape, dtype=torch.bool, device=device),
        playback_admissible=torch.ones(shape, dtype=torch.bool, device=device),
        owner_fault_bits=torch.zeros(
            n, 1, len(epoch.OWNER_ORDER), dtype=torch.int64, device=device
        ),
    )


def _arm_racket_callback(d05_owner, epoch_owner, candidate, accept_mask):
    """Arm the exact flat D05 owner with a mechanics-only private record.

    The private ``accept_mask`` is deliberately all-zero: the production D05
    accepted-row method must source its sole authorization from Epoch's active
    writer window.  The fixture only supplies candidate payload and Physical's
    landing target; neither is treated as safety evidence.
    """

    n = epoch_owner.num_envs
    token = object.__new__(device_r05.DeviceR05RowTransaction)
    private_zero = torch.zeros(n, dtype=torch.bool, device=epoch_owner.device)
    prepared = SimpleNamespace(
        selected_target_xy_m=torch.arange(
            n * 2, dtype=torch.float32, device=epoch_owner.device
        ).reshape(n, 2).contiguous()
    )
    d05_owner._row_transaction_records = {
        token: device_r05._RowTransactionRecord(
            capability=token,
            candidate=candidate,
            prepared=prepared,
            preview=SimpleNamespace(prepared=prepared),
            due_mask=accept_mask.clone(),
            construct_mask=accept_mask.clone(),
            accept_mask=private_zero,
            reject_mask=private_zero.clone(),
            defer_mask=private_zero.clone(),
            censor_mask=private_zero.clone(),
            candidate_consumed=False,
            accepted_consumers=set(),
            stage="settling",
        )
    }
    d05_owner._active_row_transaction = token
    epoch_owner._active_d05 = epoch._ActiveD05Transaction(
        rows=epoch.ActionEpochDueRows(
            common_step=1,
            due_mask=accept_mask.clone(),
            construct_mask=accept_mask.clone(),
        ),
        publication_ordinal=17,
        base_version=epoch_owner.current().version,
        token=token,
        accept_mask=accept_mask.reshape(n, 1).clone(),
        active_writer_kind="racket",
        next_writer_ordinal=1,
    )
    return token


def _settle_candidate(racket, d05_owner, epoch_owner, candidate, accept_mask):
    """Drive the exact D05 -> Epoch -> Motion/Racket/Physical callback order."""

    n = epoch_owner.num_envs
    mask = accept_mask.reshape(n, 1).contiguous()
    candidate = epoch.ActionEpochD05CandidateProjection(
        identity=candidate.identity,
        clocks=candidate.clocks,
        task=candidate.task,
        rng_counter=candidate.rng_counter,
        construction_admissible=mask.clone(),
        playback_admissible=mask.clone(),
        owner_fault_bits=candidate.owner_fault_bits,
    )
    zero = torch.zeros(n, dtype=torch.bool, device=epoch_owner.device)
    token = _arm_racket_callback(d05_owner, epoch_owner, candidate, accept_mask)
    epoch_owner._active_d05 = epoch._ActiveD05Transaction(
        rows=epoch.ActionEpochDueRows(
            common_step=1,
            due_mask=torch.ones_like(zero),
            construct_mask=torch.ones_like(zero),
        ),
        publication_ordinal=17,
        base_version=epoch_owner.current().version,
    )
    epoch_owner.settle_d05_transaction(token)
    settled = epoch_owner.current()
    return token, settled


class _PhysicalLaunchOwner:
    """Expose one exact Physical-owned launch projection to ActionEpoch."""

    def __init__(self):
        self.launch = None

    def action_epoch_r06_launch_projection(self):
        assert self.launch is not None
        return self.launch


class _R06OutcomeOwner:
    """Expose one production-shaped settled outcome to ActionEpoch."""

    def __init__(self):
        self.outcome = None

    def project_previous_paid_action_epoch_rows(self):
        return None

    def consume_closed_action_epoch_rows(self, _rows):
        return None

    def project_current_action_epoch_outcome_rows(self):
        assert self.outcome is not None
        return self.outcome


def _bind_physical_launch_owner(epoch_owner):
    owner = _PhysicalLaunchOwner()
    epoch_owner.bind_fact_owner("physical_ball", owner)
    epoch_owner.bind_async_owner("physical_ball", owner)
    return owner


def _bind_r06_outcome_owner(epoch_owner):
    owner = _R06OutcomeOwner()
    epoch_owner.bind_fact_owner("r06_landing_outcome", owner)
    epoch_owner.bind_async_owner("r06_landing_outcome", owner)
    return owner


def _refresh_real_physical_launch(epoch_owner, physical_owner, due):
    """Drive the production Physical projection -> Epoch launch transition."""

    current = epoch_owner.current()
    slot = current.current_task_slot[:, None]

    def selected(value):
        return torch.gather(value, 1, slot).squeeze(1).clone().contiguous()

    shot_key = epoch.row_identity.ActionEpochShotKey(
        **{
            field.name: selected(
                getattr(current.identity.shot_key, field.name)
            )
            for field in fields(epoch.row_identity.ActionEpochShotKey)
        }
    )
    n = epoch_owner.num_envs
    device = epoch_owner.device
    minus_one = torch.full((n,), -1, dtype=torch.int64, device=device)
    physical_owner.launch = physical.ActionEpochR06LaunchProjection(
        selected_mask=due.clone(),
        due=due.clone(),
        late_launch=torch.zeros(n, dtype=torch.bool, device=device),
        flight_slot=torch.where(
            due, torch.zeros(n, dtype=torch.int64, device=device), minus_one
        ),
        shot_key=shot_key,
        publication_ordinal=selected(current.publication_ordinal),
        target_xy_m=torch.zeros((n, 2), dtype=torch.float32, device=device),
        launch_control_step=torch.where(
            due, torch.full_like(minus_one, 5), minus_one
        ),
        contact_deadline_control_step=torch.where(
            due, torch.full_like(minus_one, 6), minus_one
        ),
        crossing_horizon_control_step=torch.where(
            due, torch.full_like(minus_one, 7), minus_one
        ),
        physical_owner=physical_owner,
        epoch_owner=epoch_owner,
        owner_identity=physical_owner,
        _token=physical._ACTION_EPOCH_R06_LAUNCH_TOKEN,
    )
    try:
        epoch_owner.refresh_physical_launch_rows()
    finally:
        physical_owner.launch = None
    return epoch_owner.current()


def _refresh_real_r06_outcome(epoch_owner, r06_owner, due):
    """Drive the production R06 projection -> Epoch outcome transition."""

    current = epoch_owner.current()
    slot = current.current_task_slot[:, None]

    def selected(value):
        return torch.gather(value, 1, slot).squeeze(1).clone().contiguous()

    shot_key = epoch.row_identity.ActionEpochShotKey(
        **{
            field.name: selected(
                getattr(current.identity.shot_key, field.name)
            )
            for field in fields(epoch.row_identity.ActionEpochShotKey)
        }
    )
    n = epoch_owner.num_envs
    device = epoch_owner.device
    minus_one = torch.full((n,), -1, dtype=torch.int64, device=device)
    r06_owner.outcome = landing.ActionEpochR06OutcomeRows(
        valid=due.clone(),
        shot_key=shot_key,
        publication_ordinal=selected(current.publication_ordinal),
        settlement_step=torch.where(
            due, torch.full_like(minus_one, 5), minus_one
        ),
        valid_bits=due.to(torch.int64),
        fact_values=torch.zeros(
            (n, epoch.OWNER_FACT_F32_WIDTH),
            dtype=torch.float32,
            device=device,
        ),
        outcome_code=torch.where(
            due, torch.full_like(minus_one, 2), minus_one
        ),
        owner_fault_bits=torch.zeros(n, dtype=torch.int64, device=device),
    )
    epoch_owner.refresh_r06_outcome_rows()
    return epoch_owner.current()


def _bytes(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().view(torch.uint8).reshape(-1).clone()


def _snapshot(racket):
    return {name: _bytes(getattr(racket, name)) for name in _WRITTEN_FIELDS}


def _control_snapshot(racket):
    tensor_names = (
        "_action_ball_continuous_racket_mutation_version_device",
        "_action_ball_continuous_racket_drain_fault_count_device",
        "_action_ball_continuous_racket_drain_invariant_count_device",
        "_action_ball_continuous_racket_terminal_resolution_total_device",
    )
    value_names = (
        "_action_ball_continuous_racket_mutation_version",
        "_action_ball_continuous_racket_terminal_resolution_total",
        "_action_ball_continuous_racket_logical_target_root_sha256",
        "_action_ball_continuous_racket_drain_poisoned",
        "_action_ball_continuous_racket_drain_poison_reason",
        "_action_ball_continuous_racket_drain_last_update_index",
        "_action_ball_continuous_racket_drain_last_completed_environment_steps",
        "_action_ball_continuous_racket_drain_sequence",
        "_action_ball_continuous_racket_drain_last_acknowledged_mutation_version",
        "_action_ball_continuous_racket_checkpoint_live_join_required",
        "_action_ball_continuous_racket_checkpoint_requires_drain_ack",
    )
    identity_names = (
        "_action_ball_continuous_racket_mutation_version_device_receipt",
        "_action_ball_continuous_racket_terminal_resolution_total_device_receipt",
        "_action_ball_continuous_racket_active_ppo_drain_pack",
        "_action_ball_continuous_racket_drain_authority",
        "_action_ball_continuous_racket_drain_last_acknowledged_receipt",
        "_action_ball_continuous_racket_checkpoint_live_epoch_receipts",
    )
    return (
        {name: _bytes(getattr(racket, name)) for name in tensor_names},
        {name: getattr(racket, name) for name in value_names},
        {name: id(getattr(racket, name)) for name in identity_names},
    )


def _assert_snapshot_equal(racket, before, *, row_mask=None):
    for name, expected in before.items():
        actual = _bytes(getattr(racket, name))
        if row_mask is None:
            assert torch.equal(actual, expected), name
            continue
        row_width = actual.numel() // racket.num_envs
        peer = (~row_mask).repeat_interleave(row_width)
        assert torch.equal(actual[peer], expected[peer]), name


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda:0"))
def test_racket_row_writer_partial_n64_preserves_peer_bytes(
    monkeypatch, runtime_device
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        64, runtime_device=runtime_device, monkeypatch=monkeypatch
    )
    device = racket.device
    candidate = _candidate(64, device)
    accept = torch.arange(64, device=device).remainder(3).eq(1)

    # Make peer preservation sensitive to NaN payloads and signed zero.
    racket.racket_target_pos_w[:, 0] = float("nan")
    racket.racket_target_pos_w[:, 1] = -0.0
    racket._vb_target_xy_per_env.fill_(float("nan"))
    before = _snapshot(racket)
    landing_before = _bytes(racket._vb_target_xy_per_env)
    token = _arm_racket_callback(d05_owner, epoch_owner, candidate, accept)
    racket.commit_action_ball_full_mdp_racket_epoch_rows(token)

    _assert_snapshot_equal(racket, before, row_mask=accept)
    # Landing target belongs to Physical, not Racket's 27D slice.
    assert torch.equal(_bytes(racket._vb_target_xy_per_env), landing_before)
    image = candidate.task.task_f32[:, 0, 5:32]
    assert torch.equal(
        racket.racket_target_pos_w[accept],
        (racket._env.scene.env_origins + image[:, 0:3])[accept],
    )
    assert torch.equal(
        racket.time_to_strike[accept],
        candidate.task.task_f32[:, 0, 0][accept],
    )
    for key_name, destination_name in _KEY_DESTINATIONS.items():
        assert torch.equal(
            getattr(racket, destination_name)[accept],
            getattr(candidate.identity.shot_key, key_name)[:, 0][accept],
        )


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda:0"))
def test_full_mdp_racket_timing_keeps_d05_task_clock(
    monkeypatch, runtime_device
):
    """Fresh FullMDP must not replace D05 TTC with legacy strike_phase."""

    racket, _, _ = _racket_and_sources(
        2, runtime_device=runtime_device, monkeypatch=monkeypatch
    )
    device = racket.device
    racket._env.step_dt = 0.02
    racket._action_ball_attempt_active = torch.ones(
        2, dtype=torch.bool, device=device
    )
    racket.cfg.strike_phase = 0.47
    racket.cfg.strike_phase_per_clip = ()
    task_ttc = torch.tensor([0.02, 0.0], dtype=torch.float32, device=device)
    motion = SimpleNamespace(
        motion=SimpleNamespace(time_step_total=57),
        action_ball_task_timing_active=torch.ones(
            2, dtype=torch.bool, device=device
        ),
        action_ball_current_task_receipt_active=torch.ones(
            2, dtype=torch.bool, device=device
        ),
        just_resampled=torch.zeros(2, dtype=torch.bool, device=device),
        action_ball_time_to_contact_remaining_s=task_ttc,
    )
    racket._motion = lambda: motion

    racket._compute_strike_timing()

    assert torch.equal(racket.time_to_strike, task_ttc)
    assert racket.pre_strike.tolist() == [True, False]
    assert racket.strike_window.tolist() == [True, True]
    # The inherited 0.47 phase would choose frame 26 of 57 and return
    # [0.52, 0.52] at frame zero.  Make that regression impossible to hide.
    assert not torch.equal(
        racket.time_to_strike,
        torch.full_like(task_ttc, 0.52),
    )

@pytest.mark.parametrize(
    ("receipt_active", "timing_active", "error"),
    (
        ((False, False), (False, False), None),
        (
            (True, False),
            (False, False),
            "active action-ball task has no receipt-owned Motion timing",
        ),
        (
            (False, False),
            (True, False),
            "FullMDP Motion timing has no current task receipt",
        ),
    ),
)
def test_full_mdp_timing_uses_motion_receipt_not_scoring_attempt(
    monkeypatch,
    receipt_active,
    timing_active,
    error,
):
    racket, _, _ = _racket_and_sources(
        2, runtime_device="cpu", monkeypatch=monkeypatch
    )
    racket._action_ball_attempt_active = torch.ones(2, dtype=torch.bool)
    task_ttc = torch.full((2,), 1.0e6, dtype=torch.float32)
    motion = SimpleNamespace(
        motion=SimpleNamespace(time_step_total=57),
        action_ball_task_timing_active=torch.tensor(
            timing_active, dtype=torch.bool
        ),
        action_ball_current_task_receipt_active=torch.tensor(
            receipt_active, dtype=torch.bool
        ),
        action_ball_time_to_contact_remaining_s=task_ttc,
    )
    racket._motion = lambda: motion
    racket._action_ball_diagnostic_unauthorized = False

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            racket._compute_strike_timing()
        return

    racket._compute_strike_timing()
    assert torch.equal(racket.time_to_strike, task_ttc)


def test_full_mdp_racket_recovery_accepts_natural_motion_suffix(
    monkeypatch,
):
    motion, _ = motion_close_test._fresh_motion(torch.device("cpu"))
    motion_close_test._seed_current_task_rows(motion)
    motion._action_ball_continuous_canonical_playback_started.fill_(True)
    motion._action_ball_continuous_canonical_task_close_tick.fill_(100)
    motion._action_ball_pre_swing_wait_s.zero_()
    motion._action_ball_task_age_s.copy_(
        torch.tensor([1.0, 0.0], dtype=torch.float64)
    )
    motion._env.common_step_counter = 3

    motion._advance_action_ball_continuous_motion_cadence()

    assert motion.action_ball_current_task_receipt_active.tolist() == [
        False,
        True,
    ]
    assert motion.action_ball_task_timing_active.tolist() == [False, True]

    racket, _, _ = _racket_and_sources(
        2, runtime_device="cpu", monkeypatch=monkeypatch
    )
    racket._action_ball_attempt_active = torch.ones(2, dtype=torch.bool)
    racket._action_ball_diagnostic_unauthorized = False
    racket._motion = lambda: motion

    racket._compute_strike_timing()

    assert racket.time_to_strike[0] == 1.0e6


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda:0"))
def test_racket_row_writer_all_zero_is_business_and_drain_byte_noop(
    monkeypatch, runtime_device
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        64, runtime_device=runtime_device, monkeypatch=monkeypatch
    )
    candidate = _candidate(64, racket.device)
    accept = torch.zeros(64, dtype=torch.bool, device=racket.device)
    racket.racket_target_vel_w[:, 0] = float("nan")
    racket.racket_target_vel_w[:, 1] = -0.0
    before = _snapshot(racket)
    controls = _control_snapshot(racket)
    token = _arm_racket_callback(d05_owner, epoch_owner, candidate, accept)
    racket.commit_action_ball_full_mdp_racket_epoch_rows(token)

    _assert_snapshot_equal(racket, before)
    assert all(
        torch.equal(_bytes(getattr(racket, name)), expected)
        for name, expected in controls[0].items()
    )
    assert {
        name: getattr(racket, name) for name in controls[1]
    } == controls[1]
    assert {
        name: id(getattr(racket, name)) for name in controls[2]
    } == controls[2]


@pytest.mark.parametrize(
    "key_name", tuple(epoch.row_identity.ActionEpochShotKey.__dataclass_fields__)
)
def test_each_invalid_shared_key_field_suppresses_only_its_row(
    monkeypatch, key_name
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        3, runtime_device="cpu", monkeypatch=monkeypatch
    )
    candidate = _candidate(3, racket.device)
    key_values = {
        name: getattr(candidate.identity.shot_key, name).clone()
        for name in epoch.row_identity.ActionEpochShotKey.__dataclass_fields__
    }
    key_values[key_name][1, 0] = -1
    candidate = epoch.ActionEpochD05CandidateProjection(
        identity=epoch.EpochIdentityPayload(
            shot_key=epoch.row_identity.ActionEpochShotKey(**key_values),
            scheduled_ordinal=candidate.identity.scheduled_ordinal,
            target_generation=candidate.identity.target_generation,
            selected_cell=candidate.identity.selected_cell,
            candidate_identity=candidate.identity.candidate_identity,
        ),
        clocks=candidate.clocks,
        task=candidate.task,
        rng_counter=candidate.rng_counter,
        construction_admissible=candidate.construction_admissible,
        playback_admissible=candidate.playback_admissible,
        owner_fault_bits=candidate.owner_fault_bits,
    )
    token = _arm_racket_callback(
        d05_owner,
        epoch_owner,
        candidate,
        torch.ones(3, dtype=torch.bool),
    )
    racket.commit_action_ball_full_mdp_racket_epoch_rows(token)
    assert racket._action_ball_task_valid.tolist() == [True, False, True]


def test_row_layout_is_authority_and_replay_foreign_calls_fail_stop(
    monkeypatch,
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        4, runtime_device="cpu", monkeypatch=monkeypatch
    )
    candidate = _candidate(4, racket.device)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    task = epoch.EpochTaskPayload(
        task_f32=candidate.task.task_f32.index_select(0, permutation).contiguous(),
        task_valid=candidate.task.task_valid,
    )
    candidate = epoch.ActionEpochD05CandidateProjection(
        identity=candidate.identity,
        clocks=candidate.clocks,
        task=task,
        rng_counter=candidate.rng_counter,
        construction_admissible=candidate.construction_admissible,
        playback_admissible=candidate.playback_admissible,
        owner_fault_bits=candidate.owner_fault_bits,
    )
    token = _arm_racket_callback(
        d05_owner,
        epoch_owner,
        candidate,
        torch.ones(4, dtype=torch.bool),
    )
    racket.commit_action_ball_full_mdp_racket_epoch_rows(token)
    image = task.task_f32[:, 0, 5:32]
    assert torch.equal(
        racket.racket_target_vel_w, image[:, 3:6]
    )

    with pytest.raises(device_r05.DeviceR05ConflictError):
        racket.commit_action_ball_full_mdp_racket_epoch_rows(token)
    assert racket._action_ball_continuous_racket_poisoned is True

    other, other_d05, other_epoch = _racket_and_sources(
        2, runtime_device="cpu", monkeypatch=monkeypatch
    )
    other_candidate = _candidate(2, other.device)
    live = _arm_racket_callback(
        other_d05,
        other_epoch,
        other_candidate,
        torch.ones(2, dtype=torch.bool),
    )
    foreign = object.__new__(device_r05.DeviceR05RowTransaction)
    with pytest.raises(device_r05.DeviceR05ConflictError):
        other.commit_action_ball_full_mdp_racket_epoch_rows(foreign)
    assert other_epoch._active_d05.token is live
    assert other._action_ball_continuous_racket_poisoned is True


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda:0"))
def test_rowwise_accept_arms_r03_and_publishes_real_fk(
    monkeypatch, runtime_device
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        2, runtime_device=runtime_device, monkeypatch=monkeypatch
    )
    r03_owner = racket_test._bind_epoch_r03(racket, epoch_owner)
    physical_owner = _bind_physical_launch_owner(epoch_owner)
    r06_owner = _bind_r06_outcome_owner(epoch_owner)
    candidate = _candidate(2, racket.device)
    racket_before = _snapshot(racket)
    fact_before = {
        name: _bytes(getattr(epoch_owner.current(), name)[1])
        for name in ("fact_valid_bits", "fact_source_step", "fact_f32")
    }
    _, settled = _settle_candidate(
        racket,
        d05_owner,
        epoch_owner,
        candidate,
        torch.ones(2, dtype=torch.bool, device=racket.device),
    )
    accepted = settled.phase[:, 0].eq(epoch.PHASE_REVEAL_COMMITTED)
    assert accepted.tolist() == [True, False]
    assert settled.phase[1, 0].item() == epoch.PHASE_IDLE
    _assert_snapshot_equal(racket, racket_before, row_mask=accepted)
    launched = _refresh_real_physical_launch(
        epoch_owner, physical_owner, accepted
    )
    assert launched.phase[:, 0].tolist() == [
        epoch.PHASE_LAUNCH_SETTLED,
        epoch.PHASE_IDLE,
    ]
    racket._action_ball_strike_fact_exact_eligibility.copy_(
        torch.tensor([True, False], dtype=torch.bool, device=racket.device)
    )
    racket.arm_action_ball_full_mdp_epoch_strike_fact()
    assert r03_owner._epoch_arm_mask.tolist() == [True, False]
    assert racket._action_ball_strike_fact_expected_publish_step == 5

    # Physical/R06 may settle the launched shot before the final-substep R03
    # publisher runs.  Eligibility was frozen at arm time, so the publication
    # must survive this real lifecycle advance rather than re-gating on phase.
    outcome = _refresh_real_r06_outcome(epoch_owner, r06_owner, accepted)
    assert outcome.phase[:, 0].tolist() == [
        epoch.PHASE_OUTCOME_SETTLED,
        epoch.PHASE_IDLE,
    ]

    achieved_position = torch.full((2, 3), 0.25, device=racket.device)
    achieved_velocity = torch.full_like(achieved_position, 0.5)
    achieved_normal = torch.zeros_like(achieved_position)
    achieved_normal[:, 1] = 1.0
    racket._racket_fk = lambda: (
        achieved_position,
        torch.zeros(2, 4, device=racket.device),
        achieved_velocity,
        achieved_normal,
        achieved_normal,
    )
    racket.publish_action_ball_full_mdp_epoch_strike_fact(source_step=5)
    record = epoch_owner.current()
    owner_slot = epoch.OWNER_ORDER.index("r03_strike_fact")
    assert record.fact_valid_bits[:, 0, owner_slot].tolist() == [3, 0]
    assert torch.equal(
        record.fact_f32[0, 0, owner_slot, 15:18], achieved_position[0]
    )
    assert torch.all(record.fact_f32[1, 0, owner_slot] == 0)
    for name, expected in fact_before.items():
        assert torch.equal(_bytes(getattr(record, name)[1]), expected), name
    assert racket._action_ball_strike_fact_expected_publish_step is None
    assert r03_owner._epoch_arm_identity is None
    sticky = {
        name: _bytes(getattr(record, name)[0])
        for name in ("fact_valid_bits", "fact_source_step", "fact_f32")
    }

    # The producer clears its exact one-shot on the following command step;
    # LAUNCH_SETTLED alone must not make the same strike eligible again.
    racket._action_ball_strike_fact_exact_eligibility.zero_()
    racket.arm_action_ball_full_mdp_epoch_strike_fact()
    assert r03_owner._epoch_arm_mask.tolist() == [False, False]
    racket.publish_action_ball_full_mdp_epoch_strike_fact(source_step=5)
    record = epoch_owner.current()
    assert record.fact_valid_bits[:, 0, owner_slot].tolist() == [3, 0]
    for name, expected in sticky.items():
        assert torch.equal(_bytes(getattr(record, name)[0]), expected), name
    assert r03_owner._epoch_arm_identity is None


@pytest.mark.parametrize(
    "nonlaunch_phase",
    (
        epoch.PHASE_IDLE,
        epoch.PHASE_REVEAL_COMMITTED,
        epoch.PHASE_OUTCOME_SETTLED,
        epoch.PHASE_RETIRED,
    ),
)
def test_r03_exact_strike_rejects_every_nonlaunch_phase(
    monkeypatch, nonlaunch_phase
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        2, runtime_device="cpu", monkeypatch=monkeypatch
    )
    r03_owner = racket_test._bind_epoch_r03(racket, epoch_owner)
    _, settled = _settle_candidate(
        racket,
        d05_owner,
        epoch_owner,
        _candidate(2, racket.device),
        torch.ones(2, dtype=torch.bool),
    )
    assert settled.phase[:, 0].tolist() == [
        epoch.PHASE_REVEAL_COMMITTED,
        epoch.PHASE_IDLE,
    ]

    # Negative-only phase injection isolates the R03 gate.  The eligible case
    # above reaches LAUNCH_SETTLED through the real Physical projection path.
    epoch_owner._publication.current.phase[0, 0] = nonlaunch_phase
    racket._action_ball_strike_fact_exact_eligibility.fill_(True)
    racket.arm_action_ball_full_mdp_epoch_strike_fact()
    assert r03_owner._epoch_arm_mask.tolist() == [False, False]


def test_rowwise_r03_rejects_foreign_racket_and_wrong_postphysics_step(
    monkeypatch,
):
    racket, d05_owner, epoch_owner = _racket_and_sources(
        2, runtime_device="cpu", monkeypatch=monkeypatch
    )
    owner = racket_test._bind_epoch_r03(racket, epoch_owner)
    _settle_candidate(
        racket,
        d05_owner,
        epoch_owner,
        _candidate(2, racket.device),
        torch.ones(2, dtype=torch.bool),
    )
    racket._action_ball_strike_fact_exact_eligibility[0] = True
    identity = racket_test.r03.EpochR03RacketIdentity(
        reset_generation=racket._action_ball_reset_generation,
        action_uid=racket._action_ball_action_uid,
        action_slot=racket._action_ball_action_slot,
        task_identity=racket._action_ball_full_mdp_racket_task_identity,
    )
    with pytest.raises(
        racket_test.r03.StrikeFactDeviceError, match="cold-bound Racket"
    ):
        owner.arm_action_epoch_strike_fact_v1(
            racket_owner=object(),
            source_step=torch.full((2,), 5, dtype=torch.int64),
            racket_identity=identity,
            target_position=racket.racket_target_pos_w,
            target_velocity=racket.racket_target_vel_w,
            target_face_normal=racket.target_normal_cmd,
            ball_position=racket._action_ball_ball_contact_target_w,
            ball_velocity=racket.vb_vel_in_w,
        )
    assert owner._epoch_arm_identity is None
    racket.arm_action_ball_full_mdp_epoch_strike_fact()
    with pytest.raises(RuntimeError, match="source step differs"):
        racket.publish_action_ball_full_mdp_epoch_strike_fact(source_step=6)
    assert owner._epoch_arm_identity is not None


def test_racket_row_writer_call_graph_has_no_host_sync_or_compact_index_path():
    method = HC.RacketTargetCommand.commit_action_ball_full_mdp_racket_epoch_rows
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    forbidden = {
        "item",
        "cpu",
        "numpy",
        "tolist",
        "nonzero",
        "masked_select",
        "_assert_async",
        "equal",
        "index_select",
        "index_copy_",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)
    assert not hasattr(
        HC.RacketTargetCommand,
        "commit_action_ball_full_mdp_racket_epoch_reveal",
    )
    for deleted_name in (
        "stage_action_ball_full_mdp_racket_device_reveal",
        "bind_action_ball_full_mdp_racket_exact_face_timing",
        "prepare_action_ball_full_mdp_exact_face_timing_source",
        "project_action_ball_full_mdp_exact_face_timing_source",
        "project_action_ball_full_mdp_racket_action_reference",
    ):
        assert not hasattr(HC.RacketTargetCommand, deleted_name)
    assert not hasattr(HC, "RacketActionReferenceProjection")
    assert "require_owned_prepared_reveal_for_child" not in inspect.getsource(
        HC.RacketTargetCommand
    )
    assert "selected_env_index" not in inspect.getsource(
        HC.RacketTargetCommand
    )
