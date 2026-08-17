"""Focused fresh Racket selected-reset leaf tests.

Run only on the exact Pod1 Isaac environment.  Most semantic counterexamples
use a diagnostic exact-method fake; one CUDA test crosses the real Device-R05
owner and the drain tests cross the real seven-slot coordinator.  They still do
not prove four real reset children, seven real drain leaves, or authorize a
training launch.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import importlib
from types import SimpleNamespace

import pytest
import torch

import test_action_ball_continuous_racket_device_reveal_hold as racket_test
import test_action_ball_continuous_runtime_transaction_device as device_r05_test


HC = racket_test.HC


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _device_r05_harness_with_open_true_reset_window(
    n: int,
    *,
    device: str,
):
    """Build the real Device-R05 owner before its construction cycle closes.

    The shared Device-R05 test helper construction-binds its reset authority,
    which cannot exercise the production order: each child first consumes its
    opaque genesis capability and the top owner binds true reset only after all
    child genesis joins succeed.  This helper reuses the real test authorities
    but deliberately leaves that one constructor seam open.
    """

    r05 = device_r05_test.r05
    dev = torch.device(device)
    genesis = device_r05_test._Genesis(dev, n)
    cadence = device_r05_test._Cadence(dev, n)
    question = device_r05_test._Question(dev)
    reveal = device_r05_test._Reveal(dev)
    children = tuple(
        device_r05_test._Child(kind) for kind in r05.CHILD_OWNER_ORDER
    )
    reveal.bind_children(children)
    drain = device_r05_test._Drain()
    reset = device_r05_test._Reset(dev, n)
    profile = device_r05_test._ProfileAuthority(dev)
    owner = r05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=12345,
        num_envs=n,
        journal_capacity=64,
        max_reveal_epochs_per_drain=64,
        genesis_authority=genesis,
        genesis_receipt=genesis.receipt,
        cadence_authority=cadence,
        question_authority=question,
        reveal_boundary_authority=reveal,
        child_completion_authorities=children,
        drain_authority=drain,
        true_reset_authority=None,
    )
    reveal.bind_owner(owner)
    reset.bind_owner(owner)
    return device_r05_test._Harness(
        owner,
        dev,
        profile,
        genesis,
        cadence,
        question,
        reveal,
        children,
        drain,
        reset,
    )


def _drain_module():
    try:
        package = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp"
        )
        module = getattr(package, "action_ball_full_mdp_ppo_drain", None)
        if module is not None:
            # ``hope_commands`` resolves this lazy package attribute at call
            # time.  Earlier focused harnesses may install the same source
            # under a synthetic short name, so the package attribute -- not a
            # possibly stale duplicate in ``sys.modules`` -- is the exact
            # production class identity the positive fixture must use.
            return module
        return importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        )
    except (ImportError, ModuleNotFoundError):
        return importlib.import_module("action_ball_full_mdp_ppo_drain")


class _DeviceR05Authority:
    def __init__(
        self,
        *,
        mask: torch.Tensor,
        reset_generation: torch.Tensor,
    ):
        self.prepared = object()
        self.prepared_identity = object()
        self.reset_event_identity = object()
        self.mask = mask
        self.generation_before = reset_generation.clone()
        self.generation_after = torch.where(
            mask,
            self.generation_before + torch.ones_like(self.generation_before),
            self.generation_before,
        )
        self.r05_receipt = object()

    def require_owned_prepared_true_reset(self, prepared, *, owner_kind):
        if prepared is not self.prepared or owner_kind != "racket":
            raise RuntimeError("prepared reset differs")
        return SimpleNamespace(
            prepared_true_reset=prepared,
            owner_kind=owner_kind,
            prepared_identity=self.prepared_identity,
            reset_event_identity=self.reset_event_identity,
            selected_mask=self.mask,
            generation_before=self.generation_before,
            generation_after=self.generation_after,
        )

    def require_owned_true_reset_receipt(
        self,
        receipt,
        *,
        expected_prepared_true_reset,
    ):
        if (
            receipt is not self.r05_receipt
            or expected_prepared_true_reset is not self.prepared
        ):
            raise RuntimeError("R05 reset acknowledgement differs")
        return receipt


def _shape_and_dtype(name: str, rows: int):
    vector3 = {
        "racket_target_pos_w",
        "racket_target_vel_w",
        "racket_target_normal_w",
        "target_normal_cmd",
        "_action_ball_ball_contact_target_w",
        "_action_ball_face_center_velocity_target_w",
        "_action_ball_prev_racket_site_w",
        "_action_ball_prev_racket_site_velocity_w",
        "_action_ball_prev_racket_angular_velocity_w",
        "vb_vel_in_w",
        "vb_spin_in_w",
    }
    vector4 = {
        "_action_ball_racket_command_quat_w",
        "_action_ball_prev_racket_quat_w",
    }
    vector2 = {
        "base_target_pos_w",
        "vb_landing_xy",
        "_vb_target_xy_per_env",
        "_counter_rally_return_direction_env_xy",
        "station_anchor_pos_w",
    }
    vector5 = {"_counter_rally_reward_terms"}
    bool_names = {
        "_action_ball_task_valid",
        "_action_ball_attempt_active",
        "_action_ball_attempt_legal",
        "_action_ball_attempt_hit",
        "_action_ball_resume_reset_exclusion",
        "pre_strike",
        "strike_window",
        "strike_window_pos",
        "strike_window_wide",
        "_post_strike_elapsed_valid",
        "_exact_fired",
        "_action_ball_prev_contact_valid",
        "vb_fired",
        "vb_landing_valid",
        "vb_on_opponent",
        "vb_depth_ok",
        "vb_net_clear",
        "vb_net_crossed",
        "_counter_rally_accepted",
        "_counter_rally_legal_first_landing",
        "_previous_in_hold",
        "_hold_edge_pending",
        "_progress_reset_mask",
        "_swing_start_pending",
        "_action_ball_reference_term_center_latch",
        "_action_ball_strike_fact_exact_eligibility",
    }
    long_names = {
        "_action_ball_task_wait_total_ticks",
        "_action_ball_task_wait_elapsed_ticks",
        "_action_ball_action_uid",
        "_action_ball_action_slot",
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "_action_ball_attempt_action",
        "_prev_motion_steps",
        "_action_ball_prev_attempt_action",
        "_action_ball_prev_reset_generation",
        "_action_ball_prev_swing_generation",
        "_counter_rally_primary_reason_code",
        "_action_ball_strike_fact_source_step",
    }
    shape = (
        (rows, 3)
        if name in vector3
        else (rows, 4)
        if name in vector4
        else (rows, 2)
        if name in vector2
        else (rows, 5)
        if name in vector5
        else (rows,)
    )
    dtype = (
        torch.bool
        if name in bool_names
        else torch.long
        if name in long_names
        else torch.float32
    )
    return shape, dtype


def _seed_reset_surface(racket, *, skip=()):
    skipped = set(skip)
    for index, (name, _disposition) in enumerate(
        HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
    ):
        if name in skipped:
            continue
        if not hasattr(racket, name):
            shape, dtype = _shape_and_dtype(name, racket.num_envs)
            setattr(
                racket,
                name,
                torch.empty(shape, dtype=dtype, device=racket.device),
            )
        value = getattr(racket, name)
        if value.dtype == torch.bool:
            seeded = torch.tensor(
                [bool((index + row) % 2) for row in range(value.numel())],
                dtype=torch.bool,
                device=racket.device,
            ).reshape(value.shape)
        elif value.is_floating_point():
            seeded = (
                torch.arange(
                    value.numel(), dtype=value.dtype, device=racket.device
                )
                .reshape(value.shape)
                .add_(index + 0.25)
            )
        else:
            seeded = (
                torch.arange(
                    value.numel(), dtype=value.dtype, device=racket.device
                )
                .reshape(value.shape)
                .add_(index + 2)
            )
        value.copy_(seeded)


def _racket():
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = "cpu"
    racket._action_ball_enabled = True
    racket._action_ball_full_mdp_enabled = False
    racket._action_ball_strike_fact_device_enabled = False
    racket._action_ball_continuous_fresh_racket_lane_bound = True
    racket.time_left = torch.full((2,), float("inf"))
    racket._action_ball_continuous_fresh_racket_time_left_receipt = (
        HC._tensor_identity_version_receipt(racket.time_left)
    )
    racket._action_ball_continuous_racket_owner_nonce = object()
    racket._action_ball_continuous_racket_poisoned = False
    racket._action_ball_continuous_racket_poison_reason = None
    racket._action_ball_continuous_racket_terminal_resolution_total = 0
    racket._action_ball_continuous_racket_terminal_resolution_total_device = (
        torch.zeros((1,), dtype=torch.int64)
    )
    racket._action_ball_continuous_racket_terminal_resolution_total_device_receipt = (
        HC._action_ball_continuous_tensor_receipt(
            racket._action_ball_continuous_racket_terminal_resolution_total_device
        )
    )
    racket._action_ball_continuous_racket_drain_fault_count_device = (
        torch.zeros((1,), dtype=torch.int64)
    )
    racket._action_ball_continuous_racket_drain_invariant_count_device = (
        torch.zeros((1,), dtype=torch.int64)
    )
    racket._action_ball_continuous_racket_active_ppo_drain_pack = None
    racket._action_ball_continuous_racket_drain_poisoned = False
    racket._action_ball_continuous_racket_drain_poison_reason = None
    racket._action_ball_continuous_racket_drain_last_update_index = -1
    racket._action_ball_continuous_racket_drain_last_completed_environment_steps = (
        -1
    )
    racket._action_ball_continuous_racket_drain_sequence = 0
    racket._action_ball_continuous_racket_drain_last_acknowledged_mutation_version = (
        -1
    )
    racket._action_ball_continuous_racket_checkpoint_requires_drain_ack = False
    racket._action_ball_continuous_racket_selected_reset_authority = None
    racket._action_ball_continuous_racket_selected_reset_prepared_validator = (
        None
    )
    racket._action_ball_continuous_racket_selected_reset_r05_validator = None
    racket._action_ball_continuous_racket_selected_reset_source_sha256 = None
    racket._action_ball_continuous_racket_selected_reset_diagnostic = False
    racket._action_ball_continuous_racket_selected_reset_next_serial = 0
    racket._action_ball_continuous_racket_selected_reset_stage = None
    racket._action_ball_continuous_racket_selected_reset_record = None
    racket._action_ball_continuous_racket_selected_reset_prevalidated = None
    racket._action_ball_continuous_racket_selected_reset_sealed_afterimage = (
        None
    )
    racket._action_ball_continuous_racket_selected_reset_commit_token = None
    racket._action_ball_continuous_racket_selected_reset_completion = None
    racket._action_ball_continuous_racket_selected_reset_completion_prepared = (
        None
    )
    racket._action_ball_continuous_racket_mutation_version = 0
    version = torch.zeros((1,), dtype=torch.long)
    racket._action_ball_continuous_racket_mutation_version_device = version
    racket._action_ball_continuous_racket_mutation_version_device_receipt = None
    racket._action_ball_continuous_racket_logical_target_root_sha256 = _sha(
        "racket-selected-reset-logical-genesis"
    )
    _seed_reset_surface(racket)
    authority = _DeviceR05Authority(
        mask=torch.tensor([False, True], dtype=torch.bool),
        reset_generation=racket._action_ball_reset_generation,
    )
    racket.bind_action_ball_continuous_racket_selected_reset(
        authority,
        prepared_reset_validator=(
            authority.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            authority.require_owned_true_reset_receipt
        ),
        diagnostic=True,
    )
    return racket, authority


def _fresh_r03_racket():
    """Attach the exact construction topology used by the full-MDP factory."""

    racket, authority = _racket()
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_strike_fact_device_enabled = True
    racket._action_ball_strike_fact_expected_publish_step = None
    epoch_owner = racket_test.epoch.ActionEpochOwner(
        num_envs=racket.num_envs,
        device=racket.device,
        shot_slot_capacity=1,
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(racket.num_envs, dtype=torch.bool),
        reset_generation=torch.zeros(racket.num_envs, dtype=torch.int64),
    )
    r03_owner = racket_test.r03.ActionBallStrikeFactDeviceCoordinator(
        num_envs=racket.num_envs,
        device=racket.device,
        observation_projection_mode=(
            racket_test.r03.OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP
        ),
        action_epoch_owner=epoch_owner,
    )
    r03_owner.bind_action_epoch_racket_owner(racket)
    racket._action_ball_full_mdp_racket_epoch_owner = epoch_owner
    racket._action_ball_strike_fact_device_coordinator = r03_owner
    racket._action_ball_strike_fact_target_validity = torch.ones(
        racket.num_envs,
        dtype=torch.bool,
        device=racket.device,
    )
    return racket, authority, epoch_owner, r03_owner


def _snapshot(racket):
    return {
        name: getattr(racket, name).clone()
        for name, _disposition in (
            HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
        )
    }


def _replace_reset_surface_with_inference_tensors(racket):
    """Replace every Racket reset destination with exact inference storage."""

    for name, _disposition in (
        HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
    ):
        value = getattr(racket, name).detach().clone()
        assert torch.is_inference(value)
        setattr(racket, name, value)


def _expected_selected(before: torch.Tensor, disposition: str):
    selected = before[1].clone()
    if disposition == "zero":
        return torch.zeros_like(selected)
    if disposition == "one":
        return torch.ones_like(selected)
    if disposition == "negative_one":
        return torch.full_like(selected, -1)
    if disposition == "increment_one":
        return selected + torch.ones_like(selected)
    if disposition == "identity_quaternion":
        expected = torch.zeros_like(selected)
        expected[0] = 1
        return expected
    raise AssertionError(disposition)


def test_selected_reset_is_prevalidated_selected_only_and_r05_last():
    racket, authority = _racket()
    before = _snapshot(racket)
    logical_before = (
        racket._action_ball_continuous_racket_logical_target_root_sha256
    )

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    assert type(stage) is HC.ActionBallContinuousRacketSelectedResetStage
    assert type(armed) is HC.ActionBallContinuousRacketPrevalidatedSelectedReset
    assert all(
        not torch.is_tensor(getattr(stage, field.name))
        for field in fields(type(stage))
    )
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name

    terminal = racket.commit_prevalidated_selected_reset(armed)
    assert type(terminal) is HC.ActionBallContinuousRacketSelectedResetCommitToken
    assert set(
        field.name
        for field in fields(HC.ActionBallContinuousRacketSelectedResetCommitToken)
    ) == {"_owner_nonce", "serial", "stage_sha256"}
    assert (
        racket._action_ball_continuous_racket_selected_reset_stage is stage
    )
    assert racket.require_owned_selected_reset_commit(
        terminal,
        expected_prepared_true_reset=authority.prepared,
    ) is terminal
    assert racket.require_owned_selected_reset_commit(
        terminal,
        expected_prepared_true_reset=authority.prepared,
    ) is terminal
    with pytest.raises(RuntimeError, match="stale or foreign"):
        racket.require_owned_selected_reset_commit(
            terminal,
            expected_prepared_true_reset=object(),
        )
    assert racket._action_ball_continuous_racket_mutation_version == 1
    assert racket._action_ball_continuous_racket_logical_target_root_sha256 != (
        logical_before
    )
    for name, disposition in (
        HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
    ):
        live = getattr(racket, name)
        assert torch.equal(live[0], before[name][0]), name
        assert torch.equal(
            live[1], _expected_selected(before[name], disposition)
        ), name

    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    assert type(completion) is (
        HC.ActionBallContinuousRacketSelectedResetCompletionToken
    )
    assert {field.name for field in fields(type(completion))} == {
        "_owner_nonce",
        "_completion_identity",
    }
    assert racket.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    ) is completion
    assert racket.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    ) is completion
    with pytest.raises(RuntimeError, match="stale or foreign"):
        racket.require_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=object(),
        )
    assert racket.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    ) is completion
    with pytest.raises(RuntimeError, match="stale or foreign"):
        racket.require_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=authority.prepared,
        )
    assert racket._action_ball_continuous_racket_selected_reset_stage is None


def test_unselected_floating_payload_is_byte_identical_after_reset():
    racket, authority = _racket()
    # Negative zero and a non-canonical quiet-NaN payload catch value-level
    # equality tests that accidentally normalize/canonicalize an unselected
    # row.  Compare the exact int32 view after the full reset settlement.
    raw = racket.racket_target_pos_w.view(torch.int32)
    raw[0].copy_(
        torch.tensor(
            [-2147483648, 2143294004, 1065353216], dtype=torch.int32
        )
    )
    before_bits = raw[0].clone()

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )

    assert torch.equal(raw[0], before_bits)
    assert torch.equal(raw[1], torch.zeros_like(raw[1]))


def test_full_selected_reset_accepts_inference_tensors_and_preserves_peer_bytes():
    racket, authority = _racket()

    # The production runner enters inference_mode before this leaf transaction.
    # Rebuild the complete destination surface there so any accidental
    # Tensor._version receipt reproduces the live RuntimeError instead of being
    # hidden by ordinary eager tensors in the focused fixture.
    with torch.inference_mode():
        _replace_reset_surface_with_inference_tensors(racket)
        raw = racket.racket_target_pos_w.view(torch.int32)
        raw[0].copy_(
            torch.tensor(
                [-2147483648, 2143294004, 1065353216], dtype=torch.int32
            )
        )
        peer_bits_before = raw[0].clone()
        before = _snapshot(racket)
        peer_bytes_before = {
            name: value[0].contiguous().reshape(-1).view(torch.uint8).clone()
            for name, value in before.items()
        }

        stage = racket.prepare_selected_reset(authority.prepared)
        armed = racket.arm_prevalidated_selected_reset(stage)
        terminal = racket.commit_prevalidated_selected_reset(armed)
        completion = racket.complete_selected_reset_after_r05(
            terminal, authority.r05_receipt
        )
        racket.consume_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=authority.prepared,
        )

        assert torch.equal(raw[0], peer_bits_before)
        for name, disposition in (
            HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
        ):
            live = getattr(racket, name)
            assert torch.is_inference(live), name
            # NaN is intentionally present in the peer row, so value-level
            # equality is false even when the payload is preserved exactly.
            # The selected-reset contract is byte preservation.
            assert torch.equal(
                live[0].contiguous().reshape(-1).view(torch.uint8),
                peer_bytes_before[name],
            ), name
            assert torch.equal(
                live[1], _expected_selected(before[name], disposition)
            ), name


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires the exact Pod CUDA lane"
)
def test_fresh_selected_reset_interoperates_with_real_device_r05_owner_cuda():
    harness = _device_r05_harness_with_open_true_reset_window(
        2, device="cuda:0"
    )
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = "cuda:0"
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_strike_fact_device_enabled = False
    fresh_only = {
        "_action_ball_task_valid",
        "_action_ball_task_wait_total_ticks",
        "_action_ball_task_wait_elapsed_ticks",
        "_action_ball_action_uid",
        "_action_ball_action_slot",
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "_action_ball_attempt_active",
        "_action_ball_attempt_action",
        "_action_ball_attempt_legal",
        "_action_ball_attempt_hit",
        "_action_ball_resume_reset_exclusion",
        "_counter_rally_return_direction_env_xy",
        "_counter_rally_target_baseline_speed_mps",
        "_counter_rally_reward_terms",
        "_counter_rally_accepted",
        "_counter_rally_legal_first_landing",
        "_counter_rally_primary_reason_code",
        "_action_ball_reference_term_center_latch",
    }
    _seed_reset_surface(racket, skip=fresh_only)
    racket._initialize_action_ball_full_mdp_racket_protocol_state()
    # Production binding must remain closed until Device-R05 freezes its
    # independent current-task/current-shot observation ABI.  This focused
    # selected-reset fixture seeds only the pre-existing genesis relationship;
    # it is not evidence of R08 production graph closure.
    with pytest.raises(
        HC.ActionBallContinuousRacketObservationHold,
        match="current-observation projection/validator ABI",
    ):
        racket.bind_action_ball_full_mdp_racket_staging(harness.owner)
    assert harness.owner._genesis_child_projections == {}
    genesis_capability = harness.owner.project_owned_genesis_for_child(
        owner_kind="racket"
    )
    genesis = harness.owner.require_owned_genesis_projection(
        genesis_capability,
        owner_kind="racket",
    )
    racket._action_ball_reset_generation.copy_(genesis.reset_generation)
    racket._action_ball_full_mdp_device_r05_owner = harness.owner
    assert torch.equal(
        racket._action_ball_reset_generation,
        harness.genesis.projection.reset_generations,
    )
    harness.owner.bind_true_reset_authority(harness.reset)
    racket.bind_action_ball_continuous_racket_selected_reset(
        harness.owner,
        prepared_reset_validator=(
            harness.owner.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            harness.owner.require_owned_true_reset_receipt
        ),
        authority_source_sha256="bogus-sha-cannot-grant-or-deny-authority",
        diagnostic=False,
    )
    with torch.inference_mode():
        _replace_reset_surface_with_inference_tensors(racket)
        before = _snapshot(racket)
        peer_bytes_before = {
            name: value[0].contiguous().reshape(-1).view(torch.uint8).clone()
            for name, value in before.items()
        }
        event = harness.reset.issue(harness.owner, (1,))
        prepared = harness.owner.prepare_true_reset_many(event)

        stage = racket.prepare_selected_reset(prepared)
        record = racket._action_ball_continuous_racket_selected_reset_record
        assert torch.is_inference(record.selected_mask)
        assert torch.is_inference(record.reset_generation_before)
        assert torch.is_inference(record.reset_generation_after)
        armed = racket.arm_prevalidated_selected_reset(stage)
        sealed = (
            racket._action_ball_continuous_racket_selected_reset_sealed_afterimage
        )
        assert all(torch.is_inference(after) for _name, _live, after in sealed.swaps)
        commit = racket.commit_prevalidated_selected_reset(armed)
        assert racket.require_owned_selected_reset_commit(
            commit,
            expected_prepared_true_reset=prepared,
        ) is commit
        harness.reset.allow_commit(prepared)
        receipt = harness.owner.commit_true_reset_many(prepared)
        completion = racket.complete_selected_reset_after_r05(commit, receipt)
        assert racket.require_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=prepared,
        ) is completion
        racket.consume_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=prepared,
        )

        for name, disposition in (
            HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
        ):
            live = getattr(racket, name)
            assert torch.equal(
                live[0].contiguous().reshape(-1).view(torch.uint8),
                peer_bytes_before[name],
            ), name
            assert torch.equal(
                live[1], _expected_selected(before[name], disposition)
            ), name


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires the exact Pod CUDA lane"
)
def test_fresh_device_r05_binder_holds_before_instance_genesis_replacement():
    harness = _device_r05_harness_with_open_true_reset_window(
        2, device="cuda:0"
    )
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = "cuda:0"
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True

    # Instance replacement must fail before any genesis capability is minted;
    # returning a publicly constructible lookalike view is not owner proof.
    real_project = harness.owner.project_owned_genesis_for_child
    harness.owner.project_owned_genesis_for_child = lambda *, owner_kind: (
        real_project(owner_kind=owner_kind)
    )
    with pytest.raises(
        HC.ActionBallContinuousRacketObservationHold,
        match="current-observation projection/validator ABI",
    ):
        racket.bind_action_ball_full_mdp_racket_staging(harness.owner)
    assert harness.owner._genesis_child_projections == {}


def test_production_selected_reset_rejects_named_bound_method_lookalikes():
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket._action_ball_continuous_fresh_racket_lane_bound = True
    racket._action_ball_full_mdp_enabled = True
    authority = _DeviceR05Authority(
        mask=torch.tensor([False, True], dtype=torch.bool),
        reset_generation=torch.zeros((2,), dtype=torch.int64),
    )
    racket._action_ball_full_mdp_device_r05_owner = authority

    with pytest.raises(RuntimeError, match="exact Device-R05 methods"):
        racket.bind_action_ball_continuous_racket_selected_reset(
            authority,
            prepared_reset_validator=(
                authority.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                authority.require_owned_true_reset_receipt
            ),
            authority_source_sha256=(
                HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_AUTHORITY_API_SHA256
            ),
            diagnostic=False,
        )


def test_deprecated_selected_reset_sha_argument_is_compatibility_only():
    source = inspect.getsource(
        HC.RacketTargetCommand.
        bind_action_ball_continuous_racket_selected_reset
    )
    assert "del authority_source_sha256" in source
    assert "authority_source_sha256" not in source.split(
        "del authority_source_sha256", 1
    )[1]


def test_selection_forgery_abort_and_after_image_mutation_fail_at_prearm():
    racket, authority = _racket()
    before = _snapshot(racket)
    with pytest.raises(RuntimeError, match="not owner-issued"):
        racket.prepare_selected_reset(object())
    assert racket._action_ball_continuous_racket_poisoned is False

    stage = racket.prepare_selected_reset(authority.prepared)
    with pytest.raises(RuntimeError, match="forged or stale"):
        racket.arm_prevalidated_selected_reset(
            replace(stage, serial=stage.serial + 1)
        )
    racket.abort_prevalidated_selected_reset(stage)
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    racket.abort_prevalidated_selected_reset(armed)
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    after_image = (
        racket._action_ball_continuous_racket_selected_reset_sealed_afterimage
        .swaps[0][2]
    )
    after_image.logical_not_()
    # The owner-private after-image cannot be reached by a production caller.
    # If an internal diagnostic tampers with it *after* prearm, commit still
    # performs no new receipt validation below the top irreversible boundary.
    terminal = racket.commit_prevalidated_selected_reset(armed)
    assert type(terminal) is HC.ActionBallContinuousRacketSelectedResetCommitToken
    assert racket._action_ball_continuous_racket_poisoned is False


def test_unconsumed_completion_blocks_second_stage_without_erasing_ack():
    racket, authority = _racket()
    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )

    with pytest.raises(RuntimeError, match="unconsumed completion"):
        racket.prepare_selected_reset(authority.prepared)

    assert racket.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    ) is completion
    assert (
        racket._action_ball_continuous_racket_selected_reset_completion
        is completion
    )
    assert (
        racket._action_ball_continuous_racket_selected_reset_completion_prepared
        is authority.prepared
    )


def test_projection_tensor_alias_mutation_cannot_rewrite_retained_reset_facts():
    racket, authority = _racket()
    before = _snapshot(racket)
    stage = racket.prepare_selected_reset(authority.prepared)

    # The authority projection tensors remain caller-visible, but staging has
    # already cloned them into an owner-private record.  Ordinary public alias
    # mutation therefore cannot change which row the leaf resets.
    authority.mask.logical_not_()
    authority.generation_before.add_(17)
    authority.generation_after.sub_(9)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )

    for name, disposition in (
        HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
    ):
        live = getattr(racket, name)
        assert torch.equal(live[0], before[name][0]), name
        assert torch.equal(
            live[1], _expected_selected(before[name], disposition)
        ), name


def test_wrong_r05_after_commit_is_sticky_poison_and_cannot_retry():
    racket, authority = _racket()
    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)

    with pytest.raises(RuntimeError, match="stale or foreign"):
        racket.complete_selected_reset_after_r05(terminal, object())
    assert racket._action_ball_continuous_racket_poisoned is True
    assert racket._action_ball_continuous_racket_selected_reset_stage is None
    with pytest.raises(RuntimeError, match="stale or foreign"):
        racket.complete_selected_reset_after_r05(
            terminal, authority.r05_receipt
        )


def test_postcommit_live_drift_is_not_a_new_r05_ack_gate():
    racket, authority = _racket()
    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.racket_target_pos_w[1, 0].add_(1)

    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    assert racket.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    ) is completion


def test_postcommit_device_chronology_drift_is_not_a_new_r05_ack_gate():
    racket, authority = _racket()
    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket._action_ball_continuous_racket_mutation_version_device.add_(1)

    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    assert racket.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    ) is completion


def test_host_mutation_chronology_drift_before_arm_is_write_free():
    racket, authority = _racket()
    before = _snapshot(racket)
    stage = racket.prepare_selected_reset(authority.prepared)
    racket._action_ball_continuous_racket_mutation_version += 1

    with pytest.raises(RuntimeError, match="forged or stale"):
        racket.arm_prevalidated_selected_reset(stage)
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_live_reset_generation_drift_sets_fault_and_still_settles_tombstone():
    racket, authority = _racket()
    before = _snapshot(racket)
    stage = racket.prepare_selected_reset(authority.prepared)
    racket._action_ball_reset_generation.add_(1)

    armed = racket.arm_prevalidated_selected_reset(stage)
    assert racket._action_ball_continuous_racket_poisoned is False
    assert torch.equal(
        racket._action_ball_continuous_racket_drain_fault_count_device,
        torch.ones((1,), dtype=torch.int64),
    )
    terminal = racket.commit_prevalidated_selected_reset(armed)
    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    racket.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    )
    assert not bool(racket._action_ball_task_valid[1])
    assert not bool(racket._action_ball_attempt_active[1])
    for name, disposition in (
        HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
    ):
        live = getattr(racket, name)
        if name == "_action_ball_reset_generation":
            # This test deliberately introduced an unselected writer drift;
            # selected reset must not overwrite that unselected row.
            assert torch.equal(live[0], before[name][0] + 1), name
        else:
            assert torch.equal(live[0], before[name][0]), name
        assert torch.equal(
            live[1], _expected_selected(before[name], disposition)
        ), name


def test_selected_generation_max_never_wraps_and_sets_fault_settlement():
    racket, authority = _racket()
    racket._action_ball_reset_generation[1] = torch.iinfo(torch.int64).max
    authority.generation_before = racket._action_ball_reset_generation.clone()
    authority.generation_after = authority.generation_before.clone()
    before = _snapshot(racket)
    stage = racket.prepare_selected_reset(authority.prepared)

    armed = racket.arm_prevalidated_selected_reset(stage)
    assert torch.equal(
        racket._action_ball_continuous_racket_drain_fault_count_device,
        torch.ones((1,), dtype=torch.int64),
    )
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    assert not bool(racket._action_ball_task_valid[1])
    assert not bool(racket._action_ball_attempt_active[1])
    for name, disposition in (
        HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
    ):
        live = getattr(racket, name)
        assert torch.equal(live[0], before[name][0]), name
        expected = (
            before[name][1]
            if name == "_action_ball_reset_generation"
            else _expected_selected(before[name], disposition)
        )
        assert torch.equal(live[1], expected), name


def test_wrapped_generation_projection_is_faulted_but_never_published():
    racket, authority = _racket()
    racket._action_ball_reset_generation[1] = torch.iinfo(torch.int64).max
    authority.generation_before = racket._action_ball_reset_generation.clone()
    authority.generation_after = authority.generation_before.clone()
    authority.generation_after[1] = torch.iinfo(torch.int64).min

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    assert torch.equal(
        racket._action_ball_continuous_racket_drain_fault_count_device,
        torch.ones((1,), dtype=torch.int64),
    )
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    assert int(racket._action_ball_reset_generation[1]) == torch.iinfo(
        torch.int64
    ).max


def test_faulted_settlement_blocks_next_fresh_update_without_business_write():
    racket, authority = _racket()
    racket._action_ball_reset_generation[1] = torch.iinfo(torch.int64).max
    authority.generation_before = racket._action_ball_reset_generation.clone()
    authority.generation_after = authority.generation_before.clone()

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    settled = _snapshot(racket)

    # D05/ActionEpoch owns every fresh task write.  A command tick before the
    # global drain is therefore a write-free no-op; it must preserve both the
    # censored row and its durable fault for the drain owner.
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_runtime_initialized = False
    assert racket._update_command() is None
    assert not bool(racket._action_ball_task_valid[1])
    assert not bool(racket._action_ball_attempt_active[1])
    assert torch.equal(
        racket._action_ball_continuous_racket_drain_fault_count_device,
        torch.ones((1,), dtype=torch.int64),
    )
    for name, expected in settled.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_fresh_command_zero_compute_is_write_free_and_never_prearms_r03(
    monkeypatch,
):
    racket, _authority = _racket()
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    before = _snapshot(racket)
    racket._update_metrics = lambda: None
    monkeypatch.setattr(
        HC,
        "_compute_without_disabled_time_resampling_scan",
        lambda _command, _dt: pytest.fail(
            "fresh Racket fell through to the generic timer lane"
        ),
    )
    racket._arm_action_ball_strike_fact_for_next_transition = lambda: (
        pytest.fail("fresh CommandTerm compute armed R03 before D05 settlement")
    )

    assert racket.compute(0.02) is None

    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_nonfresh_update_still_enters_legacy_initialization():
    racket, _authority = _racket()
    racket._action_ball_continuous_fresh_racket_lane_bound = False

    def legacy_entry():
        raise RuntimeError("legacy initialization reached")

    racket._ensure_action_ball_runtime_initialized = legacy_entry
    with pytest.raises(RuntimeError, match="legacy initialization reached"):
        racket._update_command()


def test_faulted_settlement_cannot_checkpoint_before_global_drain():
    racket, authority = _racket()
    racket._action_ball_reset_generation[1] = torch.iinfo(torch.int64).max
    authority.generation_before = racket._action_ball_reset_generation.clone()
    authority.generation_after = authority.generation_before.clone()

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    racket.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    )

    assert not bool(racket._action_ball_task_valid[1])
    assert not bool(racket._action_ball_attempt_active[1])
    with pytest.raises(RuntimeError, match="globally ACKed mutation frontier"):
        racket._action_ball_continuous_racket_exact_resume_protocol_state()


def test_healthy_global_drain_ack_is_the_only_checkpoint_mutation_frontier():
    racket, authority = _racket()
    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    completion = racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )
    racket.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=authority.prepared,
    )
    with pytest.raises(RuntimeError, match="globally ACKed mutation frontier"):
        racket._action_ball_continuous_racket_exact_resume_protocol_state()

    owner, _leaves = _global_drain_owner_with_real_racket(racket)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)

    state = racket._action_ball_continuous_racket_exact_resume_protocol_state()
    assert state["schema_version"] == 2
    assert state["mutation_version"] == 1
    assert state["drain_last_acknowledged_mutation_version"] == 1
    assert state["drain_sequence"] == 1

    wrong = dict(state)
    wrong["drain_last_acknowledged_mutation_version"] = 0
    with pytest.raises(ValueError, match="lacks its global drain ACK frontier"):
        racket._action_ball_continuous_racket_stage_exact_resume_protocol_state(
            wrong
        )


def test_fresh_racket_exact_resume_remains_explicit_hold_without_fake_hooks():
    assert HC._ACTION_BALL_FULL_MDP_RACKET_EXACT_RESUME_SUPPORTED is False
    assert HC._ACTION_BALL_FULL_MDP_RACKET_EXACT_RESUME_STATUS.startswith(
        "HOLD_"
    )
    assert "R10" in HC._ACTION_BALL_FULL_MDP_RACKET_EXACT_RESUME_STATUS


def test_legacy_resample_is_tombstoned():
    racket, _authority = _racket()
    before = _snapshot(racket)
    with pytest.raises(RuntimeError, match="resample/wrap producer"):
        racket._resample_command((1,))
    assert racket._resample_command(()) is None
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_legacy_enabled_strike_fact_has_no_fresh_selected_reset_authority():
    racket, authority = _racket()
    before = _snapshot(racket)
    racket._action_ball_strike_fact_device_enabled = True
    with pytest.raises(RuntimeError, match="requires the fresh R03 owner"):
        racket.prepare_selected_reset(authority.prepared)
    assert racket._action_ball_continuous_racket_poisoned is False
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_exact_settled_fresh_r03_allows_selected_only_facade_reset():
    racket, authority, _epoch_owner, r03_owner = _fresh_r03_racket()
    racket._action_ball_strike_fact_source_step.copy_(
        torch.tensor([31, 47], dtype=torch.int64)
    )
    racket._action_ball_strike_fact_exact_eligibility.fill_(True)
    # Signed zero and a non-canonical quiet NaN prove the unselected peer is
    # copied byte-for-byte rather than normalized by a broad clear.
    raw = racket.racket_target_pos_w.view(torch.int32)
    raw[0].copy_(
        torch.tensor(
            [-2147483648, 2143294004, 1065353216], dtype=torch.int32
        )
    )
    peer_before = {
        name: getattr(racket, name)[0].clone()
        for name, _disposition in (
            HC._ACTION_BALL_CONTINUOUS_RACKET_SELECTED_RESET_TOMBSTONES
        )
    }
    r03_before = (
        r03_owner._mutation_version,
        r03_owner._reset_total.clone(),
        r03_owner._fault_bits.clone(),
    )

    stage = racket.prepare_selected_reset(authority.prepared)
    armed = racket.arm_prevalidated_selected_reset(stage)
    terminal = racket.commit_prevalidated_selected_reset(armed)
    racket.complete_selected_reset_after_r05(
        terminal, authority.r05_receipt
    )

    assert int(racket._action_ball_strike_fact_source_step[1]) == -1
    assert not bool(racket._action_ball_strike_fact_exact_eligibility[1])
    for name, expected in peer_before.items():
        live = getattr(racket, name)[0]
        if live.is_floating_point():
            assert torch.equal(
                live.view(torch.int32), expected.view(torch.int32)
            ), name
        else:
            assert torch.equal(live, expected), name
    # Fresh R03 has no fifth reset ledger: its persistent facts live in and
    # are reset by ActionEpoch.  The Racket transaction clears only its two
    # independent facade tensors.
    assert r03_owner._mutation_version == r03_before[0]
    assert torch.equal(r03_owner._reset_total, r03_before[1])
    assert torch.equal(r03_owner._fault_bits, r03_before[2])


def test_foreign_or_unsettled_fresh_r03_fails_before_racket_reset_write():
    racket, authority, epoch_owner, r03_owner = _fresh_r03_racket()
    before = _snapshot(racket)

    foreign_epoch = racket_test.epoch.ActionEpochOwner(
        num_envs=racket.num_envs,
        device=racket.device,
        shot_slot_capacity=1,
    )
    foreign_epoch.activate_reset_genesis(
        selected_mask=torch.ones(racket.num_envs, dtype=torch.bool),
        reset_generation=torch.zeros(racket.num_envs, dtype=torch.int64),
    )
    foreign = racket_test.r03.ActionBallStrikeFactDeviceCoordinator(
        num_envs=racket.num_envs,
        device=racket.device,
        observation_projection_mode=(
            racket_test.r03.OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP
        ),
        action_epoch_owner=foreign_epoch,
    )
    foreign.bind_action_epoch_racket_owner(racket)
    racket._action_ball_strike_fact_device_coordinator = foreign
    with pytest.raises(RuntimeError, match="settled exact fresh R03 owner"):
        racket.prepare_selected_reset(authority.prepared)
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name

    racket._action_ball_strike_fact_device_coordinator = r03_owner
    current = epoch_owner.current()
    slot = current.current_task_slot[:, None]
    identity = racket_test.r03.EpochR03RacketIdentity(
        reset_generation=current.reset_generation.clone(),
        action_uid=current.identity.action_uid.gather(1, slot).squeeze(1),
        action_slot=current.identity.action_slot.gather(1, slot).squeeze(1),
        task_identity=current.identity.task_identity.gather(1, slot).squeeze(1),
    )
    zero = torch.zeros((racket.num_envs, 3), dtype=torch.float32)
    racket._action_ball_full_mdp_r03_writer_active = True
    try:
        r03_owner.arm_action_epoch_strike_fact_v1(
            racket_owner=racket,
            source_step=torch.ones(racket.num_envs, dtype=torch.int64),
            racket_identity=identity,
            target_position=zero,
            target_velocity=zero,
            target_face_normal=zero,
            ball_position=zero,
            ball_velocity=zero,
        )
    finally:
        racket._action_ball_full_mdp_r03_writer_active = False
    assert r03_owner._epoch_arm_identity is not None
    with pytest.raises(RuntimeError, match="settled exact fresh R03 owner"):
        racket.prepare_selected_reset(authority.prepared)
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_fresh_r03_owner_drift_after_stage_fails_before_afterimage_publish():
    racket, authority, _epoch_owner, _r03_owner = _fresh_r03_racket()
    before = _snapshot(racket)
    stage = racket.prepare_selected_reset(authority.prepared)
    racket._action_ball_strike_fact_device_coordinator = None
    with pytest.raises(RuntimeError, match="settled exact fresh R03 owner"):
        racket.arm_prevalidated_selected_reset(stage)
    assert racket._action_ball_continuous_racket_poisoned is True
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_fresh_protocol_initializer_owns_every_reset_destination_without_legacy_runtime():
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = "cpu"
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket.racket_target_pos_w = torch.zeros((2, 3), dtype=torch.float32)
    fresh_only = {
        "_action_ball_task_valid",
        "_action_ball_task_wait_total_ticks",
        "_action_ball_task_wait_elapsed_ticks",
        "_action_ball_action_uid",
        "_action_ball_action_slot",
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "_action_ball_attempt_active",
        "_action_ball_attempt_action",
        "_action_ball_attempt_legal",
        "_action_ball_attempt_hit",
        "_action_ball_resume_reset_exclusion",
        "_counter_rally_return_direction_env_xy",
        "_counter_rally_target_baseline_speed_mps",
        "_counter_rally_reward_terms",
        "_counter_rally_accepted",
        "_counter_rally_legal_first_landing",
        "_counter_rally_primary_reason_code",
        "_action_ball_reference_term_center_latch",
    }
    _seed_reset_surface(racket, skip=fresh_only)

    racket._initialize_action_ball_full_mdp_racket_protocol_state()

    assert all(hasattr(racket, name) for name in fresh_only)
    assert (
        racket._action_ball_continuous_racket_selected_reset_destinations()
    )
    assert not hasattr(racket, "_action_ball_broker")


def test_selected_reset_contract_has_no_per_env_host_materialization():
    sources = tuple(
        inspect.getsource(getattr(HC.RacketTargetCommand, name))
        for name in (
            "stage_action_ball_continuous_racket_selected_reset",
            "_action_ball_continuous_racket_require_strike_fact_reset_idle",
            "_validate_action_ball_continuous_racket_selected_reset_stage",
            "finalize_action_ball_continuous_racket_selected_reset",
            "commit_prevalidated_action_ball_continuous_racket_selected_reset",
            "complete_action_ball_continuous_racket_selected_reset_after_r05",
        )
    )
    for source in sources:
        for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy("):
            assert forbidden not in source
        for forbidden in (
            "_action_ball_continuous_tensor_receipt",
            "_action_ball_continuous_tensor_matches_receipt",
            "._version",
        ):
            assert forbidden not in source
    commit_source = sources[4]
    assert "validator(" not in commit_source
    assert "destination.copy_(after)" in commit_source
    assert "_matches_receipt" not in commit_source
    completion_source = sources[5]
    assert "_matches_receipt" not in completion_source
    assert "_selected_reset_destinations" not in completion_source
    assert not hasattr(
        HC.ActionBallContinuousRacketSelectedResetCommitToken,
        "canonical_sha256",
    )
    assert (
        racket_source := inspect.getsource(
            HC.RacketTargetCommand._resample_command
        )
    )
    assert '_action_ball_reject_legacy_fresh_lane("resample/wrap producer")' in (
        racket_source
    )


def _drain_authority(racket, *, drain_owner=None):
    drain = _drain_module()
    schema = drain.LeafDrainSchema(
        owner_kind="racket",
        fields=tuple(
            drain.DeviceDrainFieldSpec(name=name)
            for name in HC._ACTION_BALL_CONTINUOUS_RACKET_DRAIN_FIELD_NAMES
        ),
    )
    return drain.LeafDevicePackAuthority(
        owner_kind="racket",
        schema=schema,
        device=torch.device("cpu"),
        num_envs=racket.num_envs,
        leaf=racket,
        drain_owner=drain_owner,
    )


def _prepare_real_drain_pack(racket, authority):
    authority._open(1)
    try:
        pack = racket.prepare_pre_optimizer_ppo_boundary_device_pack(
            authority=authority,
            update_index=0,
            completed_environment_steps=2,
        )
    finally:
        minted = authority._close()
    assert minted is pack
    values = authority._require(pack, operation_id=1)
    return pack, values


class _ZeroDrainLeaf:
    def __init__(self, schema, *, num_envs, device, total=0):
        self.schema = schema
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.total = total
        self.poison_reason = None

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self, *, authority, update_index, completed_environment_steps
    ):
        del update_index, completed_environment_steps
        values = []
        for field in self.schema.fields:
            value = (
                self.total
                if field.name
                in ("terminal_resolution_total", "policy_opportunity_total")
                else 0
            )
            values.extend((value,) * field.width(self.num_envs))
        values = torch.tensor(
            values, dtype=torch.int64, device=self.device
        )
        return authority.mint_device_pack(leaf=self, values=values)

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack):
        del pack

    def acknowledge_pre_optimizer_ppo_boundary(
        self, *, pack, receipt, owner_row
    ):
        del pack, receipt, owner_row

    def poison_pre_optimizer_ppo_boundary(self, *, reason):
        if self.poison_reason is None:
            self.poison_reason = reason


def _global_drain_owner_with_real_racket(racket, *, terminal_total=0):
    drain = _drain_module()
    schemas = drain.DEFAULT_LEAF_SCHEMAS
    leaves = {
        schema.owner_kind: _ZeroDrainLeaf(
            schema,
            num_envs=racket.num_envs,
            device=racket.device,
            total=terminal_total,
        )
        for schema in schemas
    }
    leaves["racket"] = racket
    owner = drain.ActionBallFullMdpPpoDrainOwner(
        num_envs=racket.num_envs,
        device=racket.device,
        leaves=leaves,
        leaf_schemas=schemas,
        diagnostic_allow_minimal_schemas=True,
    )
    owner.require_exact_leaf_bindings(
        {name: leaves[name] for name in drain.OWNER_ORDER}
    )
    return owner, leaves


def test_k2_racket_old_plus_one_total_is_blocked_by_real_seven_leaf_conservation():
    drain = _drain_module()
    racket, _authority = _racket()
    racket._action_ball_continuous_racket_mutation_version = 1
    racket._action_ball_continuous_racket_mutation_version_device.fill_(1)
    racket._action_ball_continuous_racket_mutation_version_device_receipt = (
        HC._action_ball_continuous_tensor_receipt(
            racket._action_ball_continuous_racket_mutation_version_device
        )
    )
    racket._action_ball_continuous_racket_terminal_resolution_total = 1
    racket._action_ball_continuous_racket_terminal_resolution_total_device.fill_(1)
    racket._action_ball_continuous_racket_terminal_resolution_total_device_receipt = (
        HC._action_ball_continuous_tensor_receipt(
            racket._action_ball_continuous_racket_terminal_resolution_total_device
        )
    )
    owner, _leaves = _global_drain_owner_with_real_racket(
        racket, terminal_total=2
    )
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    with pytest.raises(
        drain.ActionBallFullMdpPpoDrainPoisonedError,
        match="r05_terminal_vs_racket_completion",
    ):
        owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)


def test_global_drain_uses_real_authority_and_abort_is_business_write_free():
    racket, _authority = _racket()
    drain_authority = _drain_authority(racket)
    before = _snapshot(racket)

    pack, values = _prepare_real_drain_pack(racket, drain_authority)
    assert values.dtype == torch.int64
    assert tuple(values.shape) == (4,)
    assert torch.equal(values, torch.zeros((4,), dtype=torch.int64))
    racket.abort_pre_optimizer_ppo_boundary_device_pack(pack=pack)
    drain_authority._retire(pack)
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


def test_global_drain_ack_crosses_real_coordinator_and_exact_seven_slot_join():
    racket, _authority = _racket()
    owner, leaves = _global_drain_owner_with_real_racket(racket)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)

    assert receipt.acknowledged is True
    assert racket._action_ball_continuous_racket_drain_sequence == 1
    assert leaves["racket"] is racket


def test_foreign_real_drain_coordinator_cannot_release_current_racket_pack():
    racket, _authority = _racket()
    owner, leaves = _global_drain_owner_with_real_racket(racket)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    racket_row = next(
        row for row in receipt.owner_rows if row.owner_kind == "racket"
    )
    active = racket._action_ball_continuous_racket_active_ppo_drain_pack

    foreign_racket, _foreign_reset = _racket()
    foreign_owner, _foreign_leaves = _global_drain_owner_with_real_racket(
        foreign_racket
    )
    foreign_prepared = foreign_owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    foreign_receipt = foreign_owner.transfer_decode_pre_optimizer_ppo_boundary(
        foreign_prepared
    )
    foreign_owner.mark_optimizer_returned(foreign_receipt)
    foreign_row = next(
        row
        for row in foreign_receipt.owner_rows
        if row.owner_kind == "racket"
    )
    assert foreign_row.values == racket_row.values

    with pytest.raises(RuntimeError, match="foreign|stale|out of window"):
        racket.acknowledge_pre_optimizer_ppo_boundary(
            pack=active.pack,
            receipt=foreign_receipt,
            owner_row=foreign_row,
        )
    assert racket._action_ball_continuous_racket_drain_poisoned is True
    assert racket._action_ball_continuous_racket_active_ppo_drain_pack is active
    assert active.stage == "poisoned"
    # The real coordinator cannot rescue a partially challenged leaf ACK.
    with pytest.raises(RuntimeError, match="poisoned|acknowledgement differs"):
        racket.acknowledge_pre_optimizer_ppo_boundary(
            pack=active.pack,
            receipt=receipt,
            owner_row=racket_row,
        )
    assert leaves["racket"] is racket


def test_global_drain_rejects_settlement_fault_before_ack_and_stays_poisoned():
    drain = _drain_module()
    racket, _authority = _racket()
    owner, _leaves = _global_drain_owner_with_real_racket(racket)
    racket._action_ball_continuous_racket_drain_fault_count_device.fill_(1)

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    with pytest.raises(
        drain.ActionBallFullMdpPpoDrainPoisonedError,
        match="racket reported a device fault",
    ):
        owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert owner.poisoned is True
    assert racket._action_ball_continuous_racket_poisoned is True
    assert racket._action_ball_continuous_racket_drain_poisoned is True
    assert (
        racket._action_ball_continuous_racket_active_ppo_drain_pack.stage
        == "poisoned"
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        racket.prepare_pre_optimizer_ppo_boundary_device_pack(
            authority=_drain_authority(racket),
            update_index=1,
            completed_environment_steps=4,
        )


def test_caller_assembled_drain_ack_is_rejected_and_sticky_poisons():
    drain = _drain_module()
    racket, _authority = _racket()
    drain_authority = _drain_authority(racket)
    pack, _values = _prepare_real_drain_pack(racket, drain_authority)
    owner_row = drain.OwnerDrainRow(
        owner_kind="racket",
        values=(
            ("mutation_version", 0),
            ("fault_count", 0),
            ("invariant_count", 0),
            ("terminal_resolution_total", 0),
        ),
    )
    with pytest.raises(RuntimeError, match="ACK authority|acknowledgement"):
        racket.acknowledge_pre_optimizer_ppo_boundary(
            pack=pack,
            receipt=SimpleNamespace(
                update_index=0,
                completed_environment_steps=2,
                drain_sequence=1,
                device_to_host_transfers=1,
            ),
            owner_row=owner_row,
        )
    assert racket._action_ball_continuous_racket_drain_poisoned is True
    assert torch.equal(
        racket._action_ball_continuous_racket_drain_fault_count_device,
        torch.ones((1,), dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        racket.prepare_pre_optimizer_ppo_boundary_device_pack(
            authority=drain_authority,
            update_index=0,
            completed_environment_steps=2,
        )


@pytest.mark.parametrize("source_name", ("mutation", "terminal"))
def test_global_drain_device_source_drift_is_sticky_at_abort(source_name):
    racket, _authority = _racket()
    drain_authority = _drain_authority(racket)
    pack, _values = _prepare_real_drain_pack(racket, drain_authority)
    source = (
        racket._action_ball_continuous_racket_mutation_version_device
        if source_name == "mutation"
        else racket._action_ball_continuous_racket_terminal_resolution_total_device
    )
    source.add_(1)

    with pytest.raises(RuntimeError, match="pre-transfer image drifted"):
        racket.abort_pre_optimizer_ppo_boundary_device_pack(pack=pack)
    assert racket._action_ball_continuous_racket_drain_poisoned is True
    assert (
        racket._action_ball_continuous_racket_active_ppo_drain_pack.stage
        == "poisoned"
    )


def test_global_drain_prepare_has_no_host_materialization():
    source = inspect.getsource(
        HC.RacketTargetCommand.prepare_pre_optimizer_ppo_boundary_device_pack
    )
    for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy(", ".to("):
        assert forbidden not in source


def test_global_drain_source_and_ack_api_pins_match_frozen_authority():
    drain = _drain_module()
    source_bytes = inspect.getsource(drain).encode("utf-8")
    assert hashlib.sha256(source_bytes).hexdigest() == (
        HC._ACTION_BALL_CONTINUOUS_RACKET_DRAIN_SOURCE_SHA256
    )

    tree = __import__("ast").parse(source_bytes.decode("utf-8"))
    authority_class = next(
        node
        for node in tree.body
        if isinstance(node, __import__("ast").ClassDef)
        and node.name == "LeafDevicePackAuthority"
    )
    method = next(
        node
        for node in authority_class.body
        if isinstance(node, __import__("ast").FunctionDef)
        and node.name == "require_owned_ack"
    )
    method_source = __import__("ast").get_source_segment(
        source_bytes.decode("utf-8"), method, padded=False
    )
    payload = {
        "fields": (),
        "methods": (("require_owned_ack", method_source),),
    }
    encoded = __import__("json").dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == (
        HC._ACTION_BALL_CONTINUOUS_RACKET_DRAIN_ACK_AUTHORITY_API_SHA256
    )
